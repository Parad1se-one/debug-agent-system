"""Cross-episode/session orchestration for the W7a -> W2 -> W7b chain."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable

from .atomic_case_adapter import W2_ATOMIC_CASE_KINDS
from .contracts import (
    TRACE_ASSEMBLY_CASE_KINDS,
    canonical_hash,
    dedupe_strings,
)
from .orchestrator import W7ShadowOrchestrator
from .source_context import build_episode_source_ledger


AtomicExtractor = Callable[[dict[str, Any]], dict[str, Any]]


def _merge_fragment_rows(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    """Recompose W1 fragments that intentionally reuse one source message ID."""

    identity_fields = ("source_message_id", "create_time", "msg_type")
    if any(
        canonical_hash(left.get(key)) != canonical_hash(right.get(key))
        for key in identity_fields
    ) or canonical_hash(left.get("sender")) != canonical_hash(
        right.get("sender")
    ):
        return None
    left_count = int(left.get("fragment_count") or 1)
    right_count = int(right.get("fragment_count") or 1)
    if max(left_count, right_count) <= 1:
        return None

    def fragments(value: dict[str, Any]) -> list[dict[str, Any]]:
        explicit = value.get("source_fragments")
        if isinstance(explicit, list):
            return [
                deepcopy(item) for item in explicit
                if isinstance(item, dict)
            ]
        return [{
            "fragment_index": int(value.get("fragment_index") or 1),
            "text": str(value.get("text") or ""),
            "content_summary": str(
                value.get("content_summary")
                or value.get("text")
                or ""
            ),
        }]

    by_index: dict[int, dict[str, Any]] = {}
    for fragment in [*fragments(left), *fragments(right)]:
        index = int(fragment.get("fragment_index") or 0)
        current = by_index.get(index)
        if (
            current is not None
            and canonical_hash(current) != canonical_hash(fragment)
        ):
            return None
        by_index[index] = fragment
    values = [by_index[index] for index in sorted(by_index)]
    output = deepcopy(left)
    output.update({
        "fragment_index": 0,
        "fragment_count": max(left_count, right_count, len(values)),
        "source_fragments": values,
        "text": "\n".join(
            str(item.get("text") or "") for item in values
            if str(item.get("text") or "")
        ),
        "content_summary": "\n".join(
            str(
                item.get("content_summary")
                or item.get("text")
                or ""
            )
            for item in values
            if str(
                item.get("content_summary")
                or item.get("text")
                or ""
            )
        ),
    })
    for key in ("attachment_metadata", "attachment_refs", "links"):
        merged: list[Any] = []
        seen: set[str] = set()
        for value in [
            *(left.get(key) or []),
            *(right.get(key) or []),
        ]:
            digest = canonical_hash(value)
            if digest not in seen:
                seen.add(digest)
                merged.append(deepcopy(value))
        output[key] = merged
    return output


def build_batch_source_ledger(
    ledgers: list[dict[str, Any]],
    *,
    batch_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Union source ledgers while rejecting conflicting message identities."""

    row_by_id: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    core_ids: list[str] = []
    source_thread_ids: list[str] = []
    episode_ids: list[str] = []
    for ledger in ledgers:
        source_thread_ids.append(
            str(ledger.get("source_thread_id") or "")
        )
        episode_ids.append(str(ledger.get("episode_id") or ""))
        core_ids.extend(ledger.get("core_message_ids") or [])
        for row in ledger.get("rows") or []:
            if not isinstance(row, dict):
                continue
            message_id = str(row.get("message_id") or "")
            if not message_id:
                issues.append("batch_row_missing_message_id")
                continue
            existing = row_by_id.get(message_id)
            if existing is None:
                row_by_id[message_id] = deepcopy(row)
            elif canonical_hash(existing) != canonical_hash(row):
                merged = _merge_fragment_rows(existing, row)
                if merged is None:
                    issues.append(
                        f"batch_message_identity_conflict:{message_id}"
                    )
                    continue
                row_by_id[message_id] = merged
    rows = sorted(
        row_by_id.values(),
        key=lambda row: (
            str(row.get("create_time") or ""),
            str(row.get("message_id") or ""),
        ),
    )
    allowed_message_ids = [
        str(row.get("message_id") or "") for row in rows
    ]
    allowed_attachment_ids = dedupe_strings(
        attachment.get("attachment_id")
        for row in rows
        for attachment in row.get("attachment_refs") or []
        if isinstance(attachment, dict)
    )
    ledger = {
        "schema_version": "w7.batch_source_ledger.v1",
        "batch_id": str(batch_id or ""),
        "source_thread_ids": dedupe_strings(source_thread_ids),
        "episode_ids": dedupe_strings(episode_ids),
        "source_thread_id": str(batch_id or ""),
        "episode_id": "",
        "rows": rows,
        "allowed_message_ids": allowed_message_ids,
        "allowed_attachment_ids": allowed_attachment_ids,
        "core_message_ids": [
            value
            for value in dedupe_strings(core_ids)
            if value in set(allowed_message_ids)
        ],
        "stats": {
            "source_units": len(ledgers),
            "rows": len(rows),
            "attachments": sum(
                len(row.get("attachment_refs") or [])
                for row in rows
            ),
        },
    }
    ledger["ledger_hash"] = canonical_hash(ledger)
    return ledger, sorted(set(issues))


def _unit_ref(episode: dict[str, Any], index: int) -> str:
    episode_id = str(
        episode.get("episode_id")
        or episode.get("thread_id")
        or f"episode-{index}"
    )
    return f"U{index:03d}-{canonical_hash(episode_id)[:8]}"


def _enrich_unit_card_scope(
    cards: list[dict[str, Any]],
    episode: dict[str, Any],
) -> list[dict[str, Any]]:
    """Propagate W1 source scope to W7b cards without inventing identity.

    A chat name is only a channel-level hint. Field reports copied into a
    customer chat may explicitly describe another site, so a ``raw.chat_name``
    offset must not become the case's authoritative ``site_scope``.
    """

    extracted = (
        episode.get("extracted")
        if isinstance(episode.get("extracted"), dict)
        else {}
    )
    artifacts = (
        extracted.get("artifacts")
        if isinstance(extracted.get("artifacts"), dict)
        else {}
    )
    anchor = (
        episode.get("field_report_anchor")
        if isinstance(episode.get("field_report_anchor"), dict)
        else {}
    )
    extracted_sites = dedupe_strings([
        *(extracted.get("sites") or []),
        *(artifacts.get("sites") or []),
        anchor.get("site"),
    ])
    source_offsets = [
        item
        for item in [
            *(episode.get("source_offsets") or []),
            *(extracted.get("source_offsets") or []),
            *(artifacts.get("source_offsets") or []),
        ]
        if isinstance(item, dict)
        and str(item.get("field") or "") == "sites"
    ]
    channel_sites = dedupe_strings(
        item.get("value")
        for item in source_offsets
        if str(item.get("source") or "") == "message.raw.chat_name"
    )
    content_sites = dedupe_strings(
        item.get("value")
        for item in source_offsets
        if str(item.get("source") or "")
        and str(item.get("source") or "") != "message.raw.chat_name"
    )
    # Older W1 payloads did not carry offset provenance. Preserve their
    # previous contract; only explicitly channel-derived values are weakened.
    if not source_offsets:
        content_sites = extracted_sites
    else:
        channel_site_set = set(channel_sites)
        content_sites = dedupe_strings([
            *content_sites,
            *(
                value
                for value in extracted_sites
                if value not in channel_site_set
            ),
        ])
    devices = dedupe_strings([
        *(extracted.get("devices") or []),
        *(artifacts.get("devices") or []),
    ])
    jira_keys = dedupe_strings([
        *(extracted.get("jira_ids") or []),
        *(artifacts.get("jira_ids") or []),
    ])
    output: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        current = deepcopy(card)
        current["site_scopes"] = dedupe_strings([
            *(current.get("site_scopes") or []),
            current.get("site_scope"),
            *content_sites,
        ])
        current["site_scope"] = (
            str(current.get("site_scope") or "")
            or (current["site_scopes"][0] if current["site_scopes"] else "")
        )
        current["channel_site_scopes"] = dedupe_strings([
            *(current.get("channel_site_scopes") or []),
            *channel_sites,
        ])
        if current["channel_site_scopes"]:
            current["site_scope_provenance"] = (
                "content"
                if current["site_scope"] else "channel_hint_only"
            )
        elif current["site_scope"]:
            current["site_scope_provenance"] = "content_or_legacy"
        current["device_scopes"] = dedupe_strings([
            *(current.get("device_scopes") or []),
            current.get("device_scope"),
            *devices,
        ])
        current["device_scope"] = (
            str(current.get("device_scope") or "")
            or (
                current["device_scopes"][0]
                if current["device_scopes"] else ""
            )
        )
        current["jira_keys"] = dedupe_strings([
            *(current.get("jira_keys") or []),
            *jira_keys,
        ])
        if anchor:
            anchor_site = str(anchor.get("site") or "")
            current["field_report_scope"] = {
                "anchor_id": str(anchor.get("anchor_id") or ""),
                "report_date": str(anchor.get("report_date") or ""),
                "site": (
                    anchor_site if anchor_site in set(content_sites) else ""
                ),
                "channel_site_hint": (
                    anchor_site if anchor_site in set(channel_sites) else ""
                ),
                "anchor_item_index": int(
                    anchor.get("anchor_item_index") or 0
                ),
            }
        output.append(current)
    return output


def _raw_cards(
    *,
    unit_ref: str,
    w7a: dict[str, Any],
    ledger: dict[str, Any],
    include_w2_eligible: bool,
) -> list[dict[str, Any]]:
    boundary = (
        (w7a.get("case_boundary") or {}).get("decision")
        if isinstance(w7a.get("case_boundary"), dict)
        else {}
    )
    anchor = (
        (w7a.get("evidence_anchor") or {}).get("decision")
        if isinstance(w7a.get("evidence_anchor"), dict)
        else {}
    )
    anchor_by_fragment: dict[str, list[dict[str, Any]]] = {}
    for item in anchor.get("anchor_decisions") or []:
        if isinstance(item, dict):
            anchor_by_fragment.setdefault(
                str(item.get("target_fragment_ref") or ""), []
            ).append(item)
    row_by_id = {
        str(row.get("message_id") or ""): row
        for row in ledger.get("rows") or []
        if isinstance(row, dict)
    }
    output: list[dict[str, Any]] = []
    for fragment in boundary.get("case_fragments") or []:
        if not isinstance(fragment, dict):
            continue
        case_kind = str(fragment.get("case_kind") or "")
        if case_kind not in TRACE_ASSEMBLY_CASE_KINDS:
            continue
        if (
            not include_w2_eligible
            and case_kind in W2_ATOMIC_CASE_KINDS
        ):
            continue
        fragment_ref = str(fragment.get("fragment_ref") or "")
        source_ids = dedupe_strings(
            fragment.get("source_message_ids") or []
        )
        anchors = anchor_by_fragment.get(fragment_ref) or []
        anchor_ids = dedupe_strings(
            item.get("evidence_message_id") for item in anchors
        )
        evidence_ids = dedupe_strings([*source_ids, *anchor_ids])
        times = sorted(
            str((row_by_id.get(message_id) or {}).get("create_time") or "")
            for message_id in evidence_ids
            if str(
                (row_by_id.get(message_id) or {}).get("create_time") or ""
            )
        )
        output.append({
            **deepcopy(fragment),
            "case_ref": f"{unit_ref}:{fragment_ref}",
            "w7a_fragment_ref": fragment_ref,
            "case_kind": case_kind,
            "title": str(fragment.get("fault_summary") or ""),
            "fault_summary": str(fragment.get("fault_summary") or ""),
            "parent_episode_id": str(
                ledger.get("episode_id") or ""
            ),
            "source_thread_id": str(
                ledger.get("source_thread_id") or ""
            ),
            "source_message_ids": evidence_ids,
            "evidence_message_ids": evidence_ids,
            "attachment_ids": dedupe_strings(
                attachment_id
                for item in anchors
                for attachment_id in item.get("attachment_ids") or []
            ),
            "start_time": times[0] if times else "",
            "chat_id": str(
                ledger.get("source_thread_id") or ""
            ).split(":", 1)[0].split("_20", 1)[0],
            "w2_projection": "not_applicable",
        })
    return output


def _dedupe_cards(
    cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_ref: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for card in cards:
        case_ref = str(
            card.get("case_ref")
            or card.get("case_item_ref")
            or card.get("fragment_ref")
            or ""
        )
        if not case_ref:
            issues.append("batch_case_card_missing_ref")
            continue
        existing = by_ref.get(case_ref)
        if existing is None:
            by_ref[case_ref] = deepcopy(card)
            continue
        identity_fields = (
            "source_case_id",
            "candidate_id",
            "atomic_episode_id",
            "case_kind",
        )
        if any(
            str(existing.get(key) or "") != str(card.get(key) or "")
            for key in identity_fields
        ):
            issues.append(f"batch_case_ref_conflict:{case_ref}")
            continue
        existing["source_message_ids"] = dedupe_strings([
            *(existing.get("source_message_ids") or []),
            *(card.get("source_message_ids") or []),
        ])
        existing["evidence_message_ids"] = dedupe_strings([
            *(existing.get("evidence_message_ids") or []),
            *(card.get("evidence_message_ids") or []),
        ])
    return [by_ref[key] for key in sorted(by_ref)], sorted(set(issues))


class W7BatchShadowOrchestrator:
    """Run W7a per episode and W7b once over a thread/chat batch."""

    schema_version = "w7.multi_agent_batch_shadow_result.v1"

    def __init__(
        self,
        orchestrator: W7ShadowOrchestrator,
        *,
        decision_workers: int = 1,
        atomic_workers: int = 1,
    ) -> None:
        self.orchestrator = orchestrator
        self.decision_workers = max(1, int(decision_workers))
        self.atomic_workers = max(1, int(atomic_workers))

    def run(
        self,
        *,
        batch_id: str,
        episodes: list[dict[str, Any]],
        atomic_extractor: AtomicExtractor | None = None,
    ) -> dict[str, Any]:
        units: list[dict[str, Any]] = []
        ledgers: list[dict[str, Any]] = []
        cards: list[dict[str, Any]] = []
        all_w2_candidates: list[dict[str, Any]] = []
        issues: list[str] = []

        valid_episodes = [
            (index, episode)
            for index, episode in enumerate(episodes, 1)
            if isinstance(episode, dict)
        ]
        for index, episode in enumerate(episodes, 1):
            if not isinstance(episode, dict):
                issues.append(f"batch_episode_not_object:{index}")

        def run_w7a_unit(
            item: tuple[int, dict[str, Any]],
        ) -> tuple[int, dict[str, Any], dict[str, Any]]:
            index, episode = item
            ledger = build_episode_source_ledger(episode)
            return (
                index,
                ledger,
                self.orchestrator.run_w7a(
                    ledger=ledger, episode=episode
                ),
            )

        if self.decision_workers > 1 and len(valid_episodes) > 1:
            with ThreadPoolExecutor(
                max_workers=self.decision_workers,
                thread_name_prefix="w7a",
            ) as executor:
                w7a_units = list(
                    executor.map(run_w7a_unit, valid_episodes)
                )
        else:
            w7a_units = [
                run_w7a_unit(item) for item in valid_episodes
            ]

        episode_by_index = dict(valid_episodes)
        w2_by_index: dict[int, dict[str, Any]] = {}
        w2_issue_by_index: dict[int, str] = {}

        def extract_atomic_unit(
            item: tuple[int, dict[str, Any], dict[str, Any]],
        ) -> tuple[int, dict[str, Any], str]:
            index, _ledger, w7a = item
            atomic_manifest = (
                (w7a.get("atomic_case_adapter") or {}).get("manifest")
                if isinstance(w7a.get("atomic_case_adapter"), dict)
                else {}
            )
            if atomic_extractor is None or not atomic_manifest:
                return index, {}, ""
            stage = "w2_atomic"
            key = self.orchestrator.checkpoints.key(
                stage=stage,
                input_value=atomic_manifest,
                version="w7-atomic-w2-adapter-v1",
            )
            cached = self.orchestrator.checkpoints.read(
                stage=stage, key=key
            )
            if cached is not None:
                return index, deepcopy(cached.get("output") or {}), ""
            try:
                result = atomic_extractor(atomic_manifest)
            except Exception as exc:
                result = {
                    "schema_version": "w7.atomic_w2_result.v1",
                    "error": f"{type(exc).__name__}:{exc}",
                    "candidates": [],
                    "w7b_case_cards": [],
                    "summary": {
                        "atomic_cases": len(
                            atomic_manifest.get(
                                "w2_atomic_episode_ids"
                            )
                            or []
                        ),
                        "candidates": 0,
                        "schema_valid": 0,
                    },
                }
                return (
                    index,
                    result,
                    f"batch_w2_failed:{type(exc).__name__}:{exc}",
                )
            summary = (
                result.get("summary")
                if isinstance(result.get("summary"), dict)
                else {}
            )
            if (
                not result.get("error")
                and int(summary.get("candidates") or 0)
                == int(summary.get("schema_valid") or 0)
            ):
                self.orchestrator.checkpoints.write(
                    stage=stage,
                    key=key,
                    output=result,
                    issues=[],
                    call={"adapter": "W2.atomic-case", "cached": False},
                )
            return index, result, ""

        if atomic_extractor is not None:
            if self.atomic_workers > 1 and len(w7a_units) > 1:
                with ThreadPoolExecutor(
                    max_workers=self.atomic_workers,
                    thread_name_prefix="w7-w2",
                ) as executor:
                    w2_units = list(
                        executor.map(extract_atomic_unit, w7a_units)
                    )
            else:
                w2_units = [
                    extract_atomic_unit(item) for item in w7a_units
                ]
            for index, w2_result, w2_issue in w2_units:
                w2_by_index[index] = w2_result
                if w2_issue:
                    w2_issue_by_index[index] = w2_issue

        for index, ledger, w7a in w7a_units:
            episode = episode_by_index[index]
            unit_ref = _unit_ref(episode, index)
            ledgers.append(ledger)
            atomic_manifest = (
                (w7a.get("atomic_case_adapter") or {}).get("manifest")
                if isinstance(w7a.get("atomic_case_adapter"), dict)
                else {}
            )
            w2_result: dict[str, Any] = w2_by_index.get(index, {})
            unit_cards: list[dict[str, Any]] = []
            if atomic_extractor is not None and atomic_manifest:
                if index in w2_issue_by_index:
                    issues.append(
                        f"{w2_issue_by_index[index]}:{unit_ref}"
                    )
                unit_cards.extend(
                    item
                    for item in w2_result.get("w7b_case_cards") or []
                    if isinstance(item, dict)
                )
                all_w2_candidates.extend(
                    item
                    for item in w2_result.get("candidates") or []
                    if isinstance(item, dict)
                )
                summary = (
                    w2_result.get("summary")
                    if isinstance(w2_result.get("summary"), dict)
                    else {}
                )
                if int(summary.get("candidates") or 0) != int(
                    summary.get("schema_valid") or 0
                ):
                    issues.append(
                        f"batch_w2_schema_invalid:{unit_ref}:"
                        f"{summary.get('schema_valid') or 0}/"
                        f"{summary.get('candidates') or 0}"
                    )
            unit_cards.extend(_raw_cards(
                unit_ref=unit_ref,
                w7a=w7a,
                ledger=ledger,
                include_w2_eligible=atomic_extractor is None,
            ))
            cards.extend(_enrich_unit_card_scope(unit_cards, episode))
            if not all(
                bool((w7a.get(key) or {}).get("schema_valid"))
                for key in (
                    "case_boundary",
                    "evidence_anchor",
                    "atomic_case_adapter",
                )
            ):
                issues.append(f"batch_w7a_invalid:{unit_ref}")
            units.append({
                "unit_ref": unit_ref,
                "episode_id": str(episode.get("episode_id") or ""),
                "source_thread_id": str(
                    episode.get("source_thread_id")
                    or episode.get("thread_id")
                    or ""
                ),
                "source_ledger_hash": str(
                    ledger.get("ledger_hash") or ""
                ),
                "w7a": w7a,
                "w2": w2_result,
            })
        batch_ledger, ledger_issues = build_batch_source_ledger(
            ledgers, batch_id=batch_id
        )
        issues.extend(ledger_issues)
        cards, card_issues = _dedupe_cards(cards)
        issues.extend(card_issues)
        w7b = self.orchestrator.run_w7b(
            ledger=batch_ledger,
            case_cards=cards,
            prior_decisions={
                "w7a_units": [{
                    "unit_ref": unit["unit_ref"],
                    "episode_id": unit["episode_id"],
                    "case_boundary": (
                        unit["w7a"]["case_boundary"].get("decision")
                        or {}
                    ),
                    "evidence_anchor": (
                        unit["w7a"]["evidence_anchor"].get("decision")
                        or {}
                    ),
                } for unit in units],
            },
            prior_issues=issues,
        )
        w7b_valid = all(
            bool((w7b.get(key) or {}).get("schema_valid"))
            for key in (
                "candidate_graph",
                "neighbor_link",
                "component_consistency",
                "component_bridge",
                "trace_components",
                "trace_phase",
                "outcome_reconciliation",
                "trace_compiler",
            )
        )
        result = {
            "schema_version": self.schema_version,
            "mode": "shadow_multi_agent",
            "batch_scope": "multi_source_unit",
            "batch_id": str(batch_id or ""),
            "source_only": True,
            "promotion_allowed": False,
            "legacy_authoritative": True,
            "source_ledger": batch_ledger,
            "source_ledger_hash": str(
                batch_ledger.get("ledger_hash") or ""
            ),
            "units": units,
            "w2_candidates": all_w2_candidates,
            "case_cards": cards,
            **w7b,
            "schema_valid": not issues and w7b_valid,
            "issues": sorted(set(issues)),
            "review_required": True,
            "fallback_policy": "keep_legacy_w7",
            "stats": {
                "source_units": len(units),
                "messages": len(
                    batch_ledger.get("allowed_message_ids") or []
                ),
                "case_cards": len(cards),
                "w2_candidates": len(all_w2_candidates),
                "traces": len(
                    (
                        (w7b.get("trace_compiler") or {}).get("bundle")
                        or {}
                    ).get("traces")
                    or []
                ),
            },
        }
        result["result_hash"] = canonical_hash(result)
        return result
