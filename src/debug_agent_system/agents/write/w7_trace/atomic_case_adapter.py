"""Deterministic adapter from W7a decisions to W2-sized atomic episodes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import canonical_hash, dedupe_strings


W2_ATOMIC_CASE_KINDS = {
    "diagnostic_case",
    "algorithm_data_request",
    "configuration_issue",
    "operator_error",
    "product_requirement",
}
_MESSAGE_BUCKETS = (
    "messages",
    "fault_description_messages",
    "diagnostic_chain_messages",
    "action_messages",
    "resolution_messages",
    "noise_messages",
    "case_evidence_messages",
    "case_context_messages",
    "context_messages",
)


def _message_id(value: dict[str, Any]) -> str:
    return str(value.get("message_id") or value.get("id") or "")


def _filter_bucket(
    episode: dict[str, Any],
    key: str,
    allowed_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        deepcopy(item)
        for item in episode.get(key) or []
        if isinstance(item, dict) and _message_id(item) in allowed_ids
    ]


def build_atomic_case_manifest(
    *,
    episode: dict[str, Any],
    source_ledger: dict[str, Any],
    case_boundary: dict[str, Any],
    evidence_anchor: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Build isolated W2 episodes without copying unrelated case evidence."""

    issues: list[str] = []
    row_by_id = {
        str(item.get("message_id") or ""): item
        for item in source_ledger.get("rows") or []
        if isinstance(item, dict) and str(item.get("message_id") or "")
    }
    anchors_by_fragment: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_anchor.get("anchor_decisions") or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_fragment_ref") or "")
        anchors_by_fragment.setdefault(target, []).append(item)
    original_episode_id = str(
        episode.get("episode_id") or episode.get("thread_id") or "episode"
    )
    envelopes: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for index, fragment in enumerate(
        case_boundary.get("case_fragments") or [], 1
    ):
        if not isinstance(fragment, dict):
            continue
        fragment_ref = str(fragment.get("fragment_ref") or "")
        if not fragment_ref or fragment_ref in seen_refs:
            issues.append(f"invalid_or_duplicate_fragment:{fragment_ref}")
            continue
        seen_refs.add(fragment_ref)
        source_ids = dedupe_strings(fragment.get("source_message_ids") or [])
        anchor_rows = anchors_by_fragment.get(fragment_ref, [])
        anchor_ids = dedupe_strings(
            item.get("evidence_message_id") for item in anchor_rows
        )
        selected_ids = dedupe_strings([*source_ids, *anchor_ids])
        missing_ids = [
            message_id for message_id in selected_ids
            if message_id not in row_by_id
        ]
        for message_id in missing_ids:
            issues.append(
                f"fragment_unknown_message:{fragment_ref}:{message_id}"
            )
        selected_ids = [
            message_id for message_id in selected_ids
            if message_id in row_by_id
        ]
        selected = set(selected_ids)
        case_kind = str(fragment.get("case_kind") or "")
        atomic_key = canonical_hash({
            "episode_id": original_episode_id,
            "fragment_ref": fragment_ref,
            "source_message_ids": selected_ids,
            "anchor_decisions": anchor_rows,
        })
        atomic_episode_id = (
            f"{original_episode_id}:atomic:{atomic_key[:12]}"
        )
        atomic_episode = deepcopy(episode)
        for key in _MESSAGE_BUCKETS:
            filtered = _filter_bucket(episode, key, selected)
            if filtered:
                atomic_episode[key] = filtered
            else:
                atomic_episode.pop(key, None)
        # The source ledger may contain a message from a bucket not present in
        # the input episode copy.  Keep it visible to W2 as case evidence.
        present = {
            _message_id(item)
            for key in _MESSAGE_BUCKETS
            for item in atomic_episode.get(key) or []
            if isinstance(item, dict)
        }
        missing_rows = [
            deepcopy(row_by_id[message_id])
            for message_id in selected_ids
            if message_id not in present
        ]
        if missing_rows:
            atomic_episode.setdefault(
                "case_evidence_messages", []
            ).extend(missing_rows)
        atomic_episode.update({
            "episode_id": atomic_episode_id,
            "parent_episode_id": original_episode_id,
            "atomic_case_ref": fragment_ref,
            "case_kind": case_kind,
            "message_ids": selected_ids,
            "evidence_message_ids": selected_ids,
            "context_message_ids": [],
            "full_context_message_ids": selected_ids,
            "summary_context_message_ids": selected_ids,
            "message_count": len(selected_ids),
        })
        extracted = (
            deepcopy(atomic_episode.get("extracted"))
            if isinstance(atomic_episode.get("extracted"), dict)
            else {}
        )
        extracted["w7_atomic_case"] = {
            "schema_version": "w7.atomic_case_envelope.v1",
            "parent_episode_id": original_episode_id,
            "atomic_episode_id": atomic_episode_id,
            "fragment_ref": fragment_ref,
            "case_kind": case_kind,
            "fault_summary": str(
                fragment.get("fault_summary") or ""
            ),
            "source_message_ids": source_ids,
            "anchored_evidence_message_ids": anchor_ids,
            "anchor_decisions": deepcopy(anchor_rows),
            "w2_eligible": case_kind in W2_ATOMIC_CASE_KINDS,
            "atomic_content_hash": atomic_key,
        }
        extracted["fault_focus_text"] = str(
            fragment.get("fault_summary") or ""
        )
        atomic_episode["extracted"] = extracted
        envelope = {
            "schema_version": "w7.atomic_case_envelope.v1",
            "fragment_ref": fragment_ref,
            "case_kind": case_kind,
            "w2_eligible": case_kind in W2_ATOMIC_CASE_KINDS,
            "parent_episode_id": original_episode_id,
            "atomic_episode_id": atomic_episode_id,
            "source_message_ids": source_ids,
            "anchored_evidence_message_ids": anchor_ids,
            "atomic_content_hash": atomic_key,
            "episode": atomic_episode,
        }
        envelopes.append(envelope)
    manifest = {
        "schema_version": "w7.atomic_case_manifest.v1",
        "parent_episode_id": original_episode_id,
        "atomic_cases": envelopes,
        "w2_atomic_episode_ids": [
            item["atomic_episode_id"]
            for item in envelopes if item["w2_eligible"]
        ],
        "non_case_message_ids": dedupe_strings(
            case_boundary.get("non_case_message_ids") or []
        ),
        "unassigned_evidence_message_ids": dedupe_strings(
            evidence_anchor.get("unassigned_evidence_message_ids") or []
        ),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return manifest, sorted(set(issues))


def w2_atomic_episodes(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        deepcopy(item["episode"])
        for item in manifest.get("atomic_cases") or []
        if isinstance(item, dict)
        and bool(item.get("w2_eligible"))
        and isinstance(item.get("episode"), dict)
    ]


def w7_case_cards_from_w2_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project W2 output to the bounded card shape consumed by W7b."""

    cards: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict):
            continue
        episode = (
            candidate.get("episode")
            if isinstance(candidate.get("episode"), dict)
            else {}
        )
        extracted = (
            episode.get("extracted")
            if isinstance(episode.get("extracted"), dict)
            else {}
        )
        atomic = (
            extracted.get("w7_atomic_case")
            if isinstance(extracted.get("w7_atomic_case"), dict)
            else {}
        )
        bundle = (
            candidate.get("candidate_draft_v2_normalized_bundle")
            if isinstance(
                candidate.get("candidate_draft_v2_normalized_bundle"), dict
            )
            else {}
        )
        objects = (
            bundle.get("objects")
            if isinstance(bundle.get("objects"), dict)
            else {}
        )
        source_case = next((
            item for item in objects.get("SourceCase") or []
            if isinstance(item, dict)
        ), {})
        variant = next((
            item for item in objects.get("FaultVariant") or []
            if isinstance(item, dict)
        ), {})
        family = next((
            item for item in objects.get("FaultFamily") or []
            if isinstance(item, dict)
        ), {})
        case_ref = str(
            source_case.get("case_id")
            or atomic.get("fragment_ref")
            or candidate.get("candidate_id")
            or f"w2-case-{index}"
        )
        evidence_ids = dedupe_strings(
            episode.get("evidence_message_ids")
            or atomic.get("source_message_ids")
            or []
        )
        cards.append({
            "case_ref": case_ref,
            "source_case_id": str(source_case.get("case_id") or ""),
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "atomic_episode_id": str(episode.get("episode_id") or ""),
            "parent_episode_id": str(
                atomic.get("parent_episode_id")
                or episode.get("parent_episode_id")
                or ""
            ),
            "case_kind": str(
                atomic.get("case_kind") or "diagnostic_case"
            ),
            "title": str(
                atomic.get("fault_summary")
                or variant.get("label")
                or family.get("label")
                or candidate.get("label")
                or ""
            ),
            "fault_summary": str(
                atomic.get("fault_summary")
                or variant.get("summary")
                or candidate.get("label")
                or ""
            ),
            "family_id": str(family.get("family_id") or ""),
            "variant_id": str(variant.get("variant_id") or ""),
            "device_scope": str(
                variant.get("equipment_type")
                or variant.get("scenario")
                or ""
            ),
            "jira_keys": dedupe_strings(
                episode.get("jira_keys")
                or source_case.get("source_tickets")
                or []
            ),
            "attachment_ids": dedupe_strings(
                attachment.get("attachment_id")
                for attachment in episode.get("attachments") or []
                if isinstance(attachment, dict)
            ),
            "start_time": episode.get("start_time") or "",
            "chat_id": episode.get("chat_id") or "",
            "source_message_ids": evidence_ids,
            "evidence_message_ids": evidence_ids,
            "production_schema_valid": bool(
                candidate.get("production_schema_valid")
            ),
        })
    return cards
