"""Build one deterministic typed candidate from W7 batch/W2 outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.validator import validate_graph

from .contracts import canonical_hash, dedupe_strings
from .correction_compiler import materialize_corrected_typed_candidate


def _bundle(candidate: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get("candidate_draft_v2_normalized_bundle")
    return value if isinstance(value, dict) else {}


def _merge_fault_family(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    """Reconcile case-specific wording for one canonical family identity."""

    for key in ("family_id", "label", "source_kind"):
        left_value = str(left.get(key) or "")
        right_value = str(right.get(key) or "")
        if left_value and right_value and left_value != right_value:
            return None
    output = deepcopy(left)
    output["keywords"] = dedupe_strings([
        *(left.get("keywords") or []),
        *(right.get("keywords") or []),
    ])
    for key in (
        "label",
        "source_kind",
        "escalation_target",
    ):
        if not str(output.get(key) or ""):
            output[key] = right.get(key) or ""
    # Family identity is broader than one incident.  Prefer the shortest
    # non-empty description deterministically; retain all case-specific
    # wording in SourceCase/FaultVariant instead of oscillating family fields.
    for key in ("summary", "category", "subsystem", "scenario"):
        values = sorted(
            {
                str(value).strip()
                for value in (left.get(key), right.get(key))
                if str(value or "").strip()
            },
            key=lambda value: (len(value), value),
        )
        output[key] = values[0] if values else ""
    return output


def _merge_bundles(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    objects: dict[str, list[dict[str, Any]]] = {}
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    issues: list[str] = []
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        bundle = _bundle(candidate)
        if not bundle:
            issues.append(
                "w7_batch_w2_normalized_bundle_missing:"
                f"{candidate.get('candidate_id') or ''}"
            )
            continue
        if not bool(bundle.get("schema_valid")):
            issues.extend(
                f"w7_batch_w2_bundle_invalid:{value}"
                for value in bundle.get("schema_issues") or ["unknown"]
            )
        raw_objects = (
            bundle.get("objects")
            if isinstance(bundle.get("objects"), dict)
            else {}
        )
        for object_type, values in raw_objects.items():
            primary_key = V2_PRIMARY_KEYS.get(str(object_type))
            for value in values or []:
                if not isinstance(value, dict):
                    continue
                object_id = str(
                    value.get(primary_key) or ""
                ) if primary_key else canonical_hash(value)
                identity = (str(object_type), object_id)
                current = by_identity.get(identity)
                if current is not None:
                    if canonical_hash(current) != canonical_hash(value):
                        merged = (
                            _merge_fault_family(current, value)
                            if object_type == "FaultFamily"
                            else None
                        )
                        if merged is None:
                            issues.append(
                                f"w7_batch_object_conflict:"
                                f"{object_type}:{object_id}"
                            )
                        else:
                            current.clear()
                            current.update(merged)
                    continue
                copied = deepcopy(value)
                by_identity[identity] = copied
                objects.setdefault(str(object_type), []).append(copied)
        for relation in bundle.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            identity = (
                str(relation.get("from") or ""),
                str(relation.get("to") or ""),
                str(relation.get("relation") or ""),
            )
            if all(identity):
                relations.setdefault(identity, deepcopy(relation))
    for object_type in V2_PRIMARY_KEYS:
        objects.setdefault(object_type, [])
    return (
        objects,
        [relations[key] for key in sorted(relations)],
        sorted(set(issues)),
    )


def _ensure_w7_case_evidence(
    *,
    objects: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, Any]],
    case_cards: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    """Materialize source-bounded W7 anchors missing from atomic W2 output."""

    evidence = objects.setdefault("EvidenceItem", [])
    evidence_by_message: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        for key in (
            "message_id",
            "source_message_id",
            "external_id",
            "source_ref",
        ):
            message_id = str(item.get(key) or "")
            if evidence_id and message_id:
                evidence_by_message.setdefault(message_id, evidence_id)
    row_by_id = {
        str(row.get("message_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("message_id") or "")
    }
    relation_keys = {
        (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("relation") or ""),
        )
        for item in relations
        if isinstance(item, dict)
    }
    for card in case_cards:
        if not isinstance(card, dict):
            continue
        source_case_id = str(card.get("source_case_id") or "")
        for message_id in dedupe_strings(
            card.get("evidence_message_ids")
            or card.get("source_message_ids")
            or []
        ):
            evidence_id = evidence_by_message.get(message_id)
            if not evidence_id:
                row = row_by_id.get(message_id) or {}
                evidence_id = (
                    "evidence:w7:"
                    + canonical_hash({
                        "source_case_id": source_case_id,
                        "message_id": message_id,
                    })[:20]
                )
                attachment_names = dedupe_strings(
                    attachment.get("name")
                    or attachment.get("file_name")
                    or attachment.get("attachment_id")
                    for attachment in row.get("attachment_refs") or []
                    if isinstance(attachment, dict)
                )
                evidence.append({
                    "evidence_id": evidence_id,
                    "source_kind": "chat_message",
                    "external_id": message_id,
                    "title": (
                        " / ".join(attachment_names)
                        or message_id
                    ),
                    "summary": (
                        str(
                            row.get("content_summary")
                            or row.get("text")
                            or ""
                        ).strip()
                        or " / ".join(attachment_names)
                        or f"群聊消息证据 {message_id}"
                    ),
                    "payload_ref": "",
                })
                evidence_by_message[message_id] = evidence_id
            if not source_case_id:
                continue
            relation_key = (
                evidence_id, source_case_id, "evidences"
            )
            if relation_key not in relation_keys:
                relations.append({
                    "from": evidence_id,
                    "to": source_case_id,
                    "relation": "evidences",
                })
                relation_keys.add(relation_key)


def build_w7_batch_typed_candidate(
    batch_result: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Combine atomic W2 bundles and materialize reviewed W7b semantics."""

    candidates = [
        item for item in batch_result.get("w2_candidates") or []
        if isinstance(item, dict)
    ]
    objects, relations, issues = _merge_bundles(candidates)
    ledger = (
        batch_result.get("source_ledger")
        if isinstance(batch_result.get("source_ledger"), dict)
        else {}
    )
    rows = [
        item for item in ledger.get("rows") or []
        if isinstance(item, dict)
    ]
    _ensure_w7_case_evidence(
        objects=objects,
        relations=relations,
        case_cards=[
            item for item in batch_result.get("case_cards") or []
            if isinstance(item, dict)
        ],
        rows=rows,
    )
    message_ids = dedupe_strings(
        ledger.get("allowed_message_ids") or []
    )
    text = "\n".join(
        str(item.get("text") or "").strip()
        for item in rows
        if str(item.get("text") or "").strip()
    )
    batch_id = str(batch_result.get("batch_id") or "")
    identity_hash = canonical_hash({
        "batch_id": batch_id,
        "source_ledger_hash": batch_result.get("source_ledger_hash") or "",
        "candidate_ids": sorted(
            str(item.get("candidate_id") or "") for item in candidates
        ),
    })
    base_candidate = {
        "schema_version": "w7.batch_typed_candidate.v1",
        "candidate_id": f"w7-batch-candidate:{identity_hash[:20]}",
        "intake_id": f"w7-batch-intake:{identity_hash[:20]}",
        "dedupe_key": f"w7-batch:{identity_hash[:20]}",
        "content_hash": "",
        "source_type": "chat",
        "source_kind": "chat",
        "source_ref": {
            "batch_id": batch_id,
            "source_thread_ids": list(
                ledger.get("source_thread_ids") or []
            ),
            "episode_ids": list(ledger.get("episode_ids") or []),
            "message_ids": message_ids,
            "source_ledger_hash": str(
                batch_result.get("source_ledger_hash") or ""
            ),
        },
        "message_ids": message_ids,
        "evidence_message_ids": message_ids,
        "raw_text": text,
        "text": text,
        "payload": {
            "schema_version": "w7.batch_typed_payload.v1",
            "text": text,
            "source_messages": rows,
            "w2_candidate_ids": [
                str(item.get("candidate_id") or "")
                for item in candidates
            ],
        },
        "evidence_pack": {
            "raw_text": text,
            "source_anchor": {
                "batch_id": batch_id,
                "message_ids": message_ids,
            },
            "outcome_evidence": [
                item
                for item in objects.get("ActionOutcome") or []
                if isinstance(item, dict)
            ],
        },
        "lineage": {
            "agent_id": "W7a/W2/W7b/TraceCompiler",
            "source_ledger_hash": str(
                batch_result.get("source_ledger_hash") or ""
            ),
            "w7_result_hash": str(
                batch_result.get("result_hash") or ""
            ),
        },
        "objects": objects,
        "relations": relations,
        "schema_valid": not issues,
        "schema_issues": list(issues),
        "admission_target": "fault_execution",
        "admission_readiness": "execution_ready",
        "context_evidence_policy": "w7_promoted_case_evidence.v1",
        "evidence_disposition": "promoted_case_evidence",
        "operation": "merge_graph",
        "promotion_source": "w7_multi_agent",
    }
    semantic_bundle = (
        (batch_result.get("trace_compiler") or {}).get("bundle")
        if isinstance(batch_result.get("trace_compiler"), dict)
        else {}
    )
    trace_payload = (
        batch_result.get("w6_trace_review_payload")
        if isinstance(
            batch_result.get("w6_trace_review_payload"), dict
        )
        else {}
    )
    if semantic_bundle.get("traces") and not issues:
        materialized, materialize_issues = (
            materialize_corrected_typed_candidate(
                typed_candidate=base_candidate,
                correction_compile_result={
                    "kg_materialization_ready": not issues,
                    "compile_result_hash": str(
                        batch_result.get("result_hash") or ""
                    ),
                    "corrected_trace_review_payload": trace_payload,
                    "corrected_compiled_trace_bundle": semantic_bundle,
                },
            )
        )
        issues.extend(materialize_issues)
        if materialized:
            base_candidate = materialized
    graph_issues = validate_graph(
        base_candidate.get("objects") or {},
        base_candidate.get("relations") or [],
    )
    issues.extend(
        f"kg_v2_validator:{value}" for value in graph_issues
    )
    base_candidate["schema_valid"] = not issues
    base_candidate["schema_issues"] = sorted(set(issues))
    hash_basis = {
        key: value
        for key, value in base_candidate.items()
        if key != "content_hash"
    }
    base_candidate["content_hash"] = (
        "content:w7-batch:" + canonical_hash(hash_basis)
    )
    return base_candidate, sorted(set(issues))
