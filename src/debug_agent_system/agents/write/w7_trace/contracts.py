"""Versioned contracts and local validators for the W7 decision pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Iterable


W7_MODES = (
    "legacy",
    "shadow_multi_agent",
    "assisted",
    "multi_agent",
)
CASE_KINDS = (
    "diagnostic_case",
    "algorithm_data_request",
    "configuration_issue",
    "operator_error",
    "positive_validation",
    "product_requirement",
    "jira_status_update",
    "field_work_report",
    "coordination_only",
    "noise",
)
TRACE_ROOT_CASE_KINDS = {
    "diagnostic_case",
    "algorithm_data_request",
    "configuration_issue",
    "operator_error",
}
# Trace participation is intentionally broader than W2/KG eligibility.
# Field reports, requirements and coordination records may represent a later
# action/validation phase, but they still cannot become a diagnostic root on
# their own because TRACE_ROOT_CASE_KINDS remains strict.
TRACE_ASSEMBLY_CASE_KINDS = set(CASE_KINDS) - {"noise"}
EDGE_DECISIONS = ("must_link", "possible_link", "cannot_link")
TRACE_EVENT_TYPES = (
    "report",
    "diagnosis",
    "action",
    "short_term_recovery",
    "recurrence",
    "resolution",
    "validation",
)
TRACE_RELATION_TYPES = (
    "trace_root",
    "continuation_of",
    "diagnosis_of",
    "action_for",
    "recurrence_of",
    "validation_of",
)
RESOLUTION_STATUSES = (
    "unknown",
    "pending",
    "investigating",
    "mitigation_observed",
    "provisionally_resolved",
    "ineffective",
    "recurrence",
    "verified",
)
ANCHOR_ROLES = (
    "initial_report_attachment",
    "initial_diagnostic_package",
    "diagnostic_evidence",
    "action_evidence",
    "outcome_evidence",
    "context_only",
)
ANCHOR_CONFIDENCE = ("high", "medium", "low")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dedupe_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value or "")))


def resolve_w7_mode(value: str | None = None) -> str:
    mode = str(value if value is not None else os.environ.get("W7_MODE", "legacy"))
    mode = mode.strip().lower() or "legacy"
    if mode not in W7_MODES:
        raise ValueError(f"unsupported_w7_mode:{mode}")
    return mode


def validate_case_boundary_decision(
    raw: dict[str, Any],
    *,
    allowed_message_ids: set[str],
    message_text_lengths: dict[str, int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    fragments: list[dict[str, Any]] = []
    refs: set[str] = set()
    accounted: set[str] = set()
    values = raw.get("case_fragments") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        values = []
        issues.append("case_fragments_not_list")
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"case_fragments[{index}]:not_object")
            continue
        ref = str(value.get("fragment_ref") or "")
        if not ref:
            issues.append(f"case_fragments[{index}]:missing_ref")
            continue
        if ref in refs:
            issues.append(f"case_fragments[{index}]:duplicate_ref:{ref}")
            continue
        refs.add(ref)
        kind = str(value.get("case_kind") or "")
        if kind not in CASE_KINDS:
            issues.append(f"case_fragments[{index}]:invalid_kind:{kind}")
        message_ids = dedupe_strings(value.get("source_message_ids") or [])
        unknown = [item for item in message_ids if item not in allowed_message_ids]
        for message_id in unknown:
            issues.append(
                f"case_fragments[{index}]:unknown_message_id:{message_id}"
            )
        message_ids = [
            item for item in message_ids if item in allowed_message_ids
        ]
        if kind not in {"noise", "coordination_only"} and not message_ids:
            issues.append(f"case_fragments[{index}]:missing_source_message")
        accounted.update(message_ids)
        spans: list[dict[str, Any]] = []
        for span_index, span in enumerate(value.get("evidence_spans") or []):
            if not isinstance(span, dict):
                issues.append(
                    f"case_fragments[{index}].evidence_spans[{span_index}]:not_object"
                )
                continue
            message_id = str(span.get("message_id") or "")
            if message_id not in message_ids:
                issues.append(
                    f"case_fragments[{index}].evidence_spans[{span_index}]:"
                    f"message_not_in_fragment:{message_id}"
                )
                continue
            start = int(span.get("start") or 0)
            end = int(span.get("end") or 0)
            text_length = (
                max(0, int(message_text_lengths.get(message_id) or 0))
                if message_text_lengths is not None
                else None
            )
            # Attachment-only messages have no character range.  They remain
            # in source_message_ids and are bound by EvidenceAnchorAgent.
            if text_length == 0:
                continue
            if start < 0:
                issues.append(
                    f"case_fragments[{index}].evidence_spans[{span_index}]:"
                    f"invalid_offset:{start}:{end}"
                )
                continue
            if end <= start:
                if message_text_lengths is not None:
                    continue
                issues.append(
                    f"case_fragments[{index}].evidence_spans[{span_index}]:"
                    f"invalid_offset:{start}:{end}"
                )
                continue
            elif text_length is not None and start >= text_length:
                # Character offsets are optional advisory anchors.  Drop an
                # impossible model offset while retaining the bounded source
                # message ID; W2 still receives the complete source text.
                continue
            elif text_length is not None and end > text_length:
                # Models count Unicode/punctuation offsets inconsistently.
                # The local compiler may safely clamp an otherwise valid
                # prefix span; it never expands the source evidence range.
                end = text_length
            spans.append({"message_id": message_id, "start": start, "end": end})
        fragments.append({
            "fragment_ref": ref,
            "case_kind": kind,
            "fault_summary": str(value.get("fault_summary") or "").strip(),
            "source_message_ids": message_ids,
            "evidence_spans": spans,
            "uncertainties": dedupe_strings(value.get("uncertainties") or []),
        })
    non_case = dedupe_strings(
        raw.get("non_case_message_ids") or [] if isinstance(raw, dict) else []
    )
    for message_id in non_case:
        if message_id not in allowed_message_ids:
            issues.append(f"non_case_unknown_message_id:{message_id}")
        elif message_id in accounted:
            issues.append(f"message_case_and_non_case:{message_id}")
    accounted.update(item for item in non_case if item in allowed_message_ids)
    unaccounted = sorted(allowed_message_ids - accounted)
    for message_id in unaccounted:
        issues.append(f"message_unaccounted:{message_id}")
    normalized = {
        "schema_version": "w7.case_boundary_decision.v1",
        "case_fragments": fragments,
        "non_case_message_ids": [
            item for item in non_case if item in allowed_message_ids
        ],
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))


def validate_outcome_patch(
    raw: dict[str, Any],
    *,
    allowed_trace_refs: set[str],
    allowed_message_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    operations: list[dict[str, Any]] = []
    values = raw.get("operations") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        values = []
        issues.append("operations_not_list")
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"operations[{index}]:not_object")
            continue
        trace_ref = str(value.get("local_trace_ref") or "")
        if trace_ref not in allowed_trace_refs:
            issues.append(f"operations[{index}]:unknown_trace_ref:{trace_ref}")
            continue
        target = str(value.get("to") or "")
        if target not in RESOLUTION_STATUSES:
            issues.append(f"operations[{index}]:invalid_status:{target}")
        evidence = dedupe_strings(value.get("evidence_message_ids") or [])
        for message_id in evidence:
            if message_id not in allowed_message_ids:
                issues.append(
                    f"operations[{index}]:unknown_message_id:{message_id}"
                )
        evidence = [
            item for item in evidence if item in allowed_message_ids
        ]
        if target == "verified" and not evidence:
            issues.append(f"operations[{index}]:verified_without_evidence")
        if trace_ref in seen:
            issues.append(f"operations[{index}]:duplicate_trace_revision:{trace_ref}")
        seen.add(trace_ref)
        operations.append({
            "op": "revise_trace_status",
            "local_trace_ref": trace_ref,
            "from": str(value.get("from") or ""),
            "to": target,
            "evidence_message_ids": evidence,
            "reason": str(value.get("reason") or "").strip(),
        })
    normalized = {
        "schema_version": "w7.outcome_patch.v1",
        "operations": operations,
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))


def validate_evidence_anchor_decision(
    raw: dict[str, Any],
    *,
    allowed_fragment_refs: set[str],
    candidate_message_ids: set[str],
    allowed_attachment_ids_by_message: dict[str, set[str]],
) -> tuple[dict[str, Any], list[str]]:
    """Validate complete, exclusive accounting of attachment evidence."""

    issues: list[str] = []
    decisions: list[dict[str, Any]] = []
    accounted: set[str] = set()
    values = raw.get("anchor_decisions") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        values = []
        issues.append("anchor_decisions_not_list")
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"anchor_decisions[{index}]:not_object")
            continue
        message_id = str(value.get("evidence_message_id") or "")
        if message_id not in candidate_message_ids:
            issues.append(
                f"anchor_decisions[{index}]:unknown_evidence_message:{message_id}"
            )
            continue
        if message_id in accounted:
            issues.append(
                f"anchor_decisions[{index}]:duplicate_evidence_message:{message_id}"
            )
            continue
        target = str(value.get("target_fragment_ref") or "")
        if target not in allowed_fragment_refs:
            issues.append(
                f"anchor_decisions[{index}]:unknown_fragment_ref:{target}"
            )
        role = str(value.get("role") or "")
        if role not in ANCHOR_ROLES:
            issues.append(f"anchor_decisions[{index}]:invalid_role:{role}")
        confidence = str(value.get("confidence") or "")
        if confidence not in ANCHOR_CONFIDENCE:
            issues.append(
                f"anchor_decisions[{index}]:invalid_confidence:{confidence}"
            )
        attachment_ids = dedupe_strings(value.get("attachment_ids") or [])
        allowed_attachments = allowed_attachment_ids_by_message.get(
            message_id, set()
        )
        for attachment_id in attachment_ids:
            if attachment_id not in allowed_attachments:
                issues.append(
                    f"anchor_decisions[{index}]:unknown_attachment_id:"
                    f"{message_id}:{attachment_id}"
                )
        accounted.add(message_id)
        decisions.append({
            "evidence_message_id": message_id,
            "attachment_ids": [
                item for item in attachment_ids if item in allowed_attachments
            ],
            "target_fragment_ref": target,
            "role": role,
            "confidence": confidence,
            "reasons": dedupe_strings(value.get("reasons") or []),
        })
    unassigned = dedupe_strings(
        raw.get("unassigned_evidence_message_ids") or []
        if isinstance(raw, dict)
        else []
    )
    for message_id in unassigned:
        if message_id not in candidate_message_ids:
            issues.append(
                f"unassigned_unknown_evidence_message:{message_id}"
            )
        elif message_id in accounted:
            issues.append(
                f"evidence_message_assigned_and_unassigned:{message_id}"
            )
    accounted.update(
        item for item in unassigned if item in candidate_message_ids
    )
    for message_id in sorted(candidate_message_ids - accounted):
        issues.append(f"evidence_message_unaccounted:{message_id}")
    normalized = {
        "schema_version": "w7.evidence_anchor_decision.v1",
        "anchor_decisions": decisions,
        "unassigned_evidence_message_ids": [
            item for item in unassigned if item in candidate_message_ids
        ],
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))


def _edge_key(left: Any, right: Any) -> tuple[str, str]:
    values = sorted((str(left or ""), str(right or "")))
    return values[0], values[1]


def must_link_reason_is_contradictory(reasons: Iterable[Any]) -> bool:
    reason_text = " ".join(
        str(value) for value in reasons if str(value or "")
    ).lower()
    return bool(re.search(
        r"不同(?:问题|故障|事件|设备|产线|根因|业务)"
        r"|独立(?:问题|故障|事件)"
        r"|different (?:issue|issues|fault|faults|problem|"
        r"problems|event|events|device|devices|root cause|"
        r"root causes)",
        reason_text,
    ))


def validate_trace_link_decision(
    raw: dict[str, Any],
    *,
    required_edges: set[tuple[str, str]],
    allowed_edges: set[tuple[str, str]],
    allowed_message_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate bounded edge decisions without requiring weak-edge negatives."""

    required = {_edge_key(*edge) for edge in required_edges}
    allowed = {_edge_key(*edge) for edge in allowed_edges}
    issues: list[str] = []
    decisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    values = raw.get("edge_decisions") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        values = []
        issues.append("edge_decisions_not_list")
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"edge_decisions[{index}]:not_object")
            continue
        pair = _edge_key(
            value.get("left_case_ref"), value.get("right_case_ref")
        )
        if not all(pair) or pair not in allowed:
            issues.append(
                f"edge_decisions[{index}]:unknown_edge:{pair[0]}:{pair[1]}"
            )
            continue
        if pair in seen:
            issues.append(
                f"edge_decisions[{index}]:duplicate_edge:{pair[0]}:{pair[1]}"
            )
            continue
        seen.add(pair)
        decision = str(value.get("decision") or "")
        if decision not in EDGE_DECISIONS:
            issues.append(
                f"edge_decisions[{index}]:invalid_decision:{decision}"
            )
        relation = str(value.get("relation_hint") or "")
        if relation and relation not in TRACE_RELATION_TYPES:
            issues.append(
                f"edge_decisions[{index}]:invalid_relation:{relation}"
            )
        evidence = dedupe_strings(value.get("evidence_message_ids") or [])
        for message_id in evidence:
            if message_id not in allowed_message_ids:
                issues.append(
                    f"edge_decisions[{index}]:unknown_message_id:{message_id}"
                )
        reasons = dedupe_strings(value.get("reasons") or [])
        if decision in {"must_link", "possible_link"} and not reasons:
            issues.append(f"edge_decisions[{index}]:link_without_reason")
        if (
            decision == "must_link"
            and must_link_reason_is_contradictory(reasons)
        ):
            issues.append(
                f"edge_decisions[{index}]:must_link_reason_contradiction"
            )
        decisions.append({
            "left_case_ref": pair[0],
            "right_case_ref": pair[1],
            "decision": decision,
            "relation_hint": relation,
            "evidence_message_ids": [
                item for item in evidence if item in allowed_message_ids
            ],
            "reasons": reasons,
        })
    for left, right in sorted(required - seen):
        issues.append(f"required_edge_unaccounted:{left}:{right}")
    normalized = {
        "schema_version": "w7.trace_link_decision.v1",
        "edge_decisions": decisions,
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))


def validate_component_consistency_decision(
    raw: dict[str, Any],
    *,
    required_conflicts: set[tuple[str, str]],
    allowed_message_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate a bounded review of contradictory cannot-link edges."""

    required = {_edge_key(*pair) for pair in required_conflicts}
    issues: list[str] = []
    decisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    values = (
        raw.get("conflict_decisions")
        if isinstance(raw, dict)
        else None
    )
    if not isinstance(values, list):
        values = []
        issues.append("conflict_decisions_not_list")
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"conflict_decisions[{index}]:not_object")
            continue
        pair = _edge_key(
            value.get("left_case_ref"), value.get("right_case_ref")
        )
        if not all(pair) or pair not in required:
            issues.append(
                "conflict_decisions"
                f"[{index}]:unknown_conflict:{pair[0]}:{pair[1]}"
            )
            continue
        if pair in seen:
            issues.append(
                "conflict_decisions"
                f"[{index}]:duplicate_conflict:{pair[0]}:{pair[1]}"
            )
            continue
        seen.add(pair)
        decision = str(value.get("decision") or "")
        if decision not in {"confirmed_cannot", "weak_cannot"}:
            issues.append(
                "conflict_decisions"
                f"[{index}]:invalid_decision:{decision}"
            )
        evidence = dedupe_strings(value.get("evidence_message_ids") or [])
        for message_id in evidence:
            if message_id not in allowed_message_ids:
                issues.append(
                    "conflict_decisions"
                    f"[{index}]:unknown_message_id:{message_id}"
                )
        reasons = dedupe_strings(value.get("reasons") or [])
        if not reasons:
            issues.append(
                f"conflict_decisions[{index}]:decision_without_reason"
            )
        decisions.append({
            "left_case_ref": pair[0],
            "right_case_ref": pair[1],
            "decision": decision,
            "evidence_message_ids": [
                item for item in evidence
                if item in allowed_message_ids
            ],
            "reasons": reasons,
        })
    for left, right in sorted(required - seen):
        issues.append(f"required_conflict_unaccounted:{left}:{right}")
    normalized = {
        "schema_version": "w7.component_consistency_decision.v1",
        "conflict_decisions": decisions,
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))


def validate_component_bridge_decision(
    raw: dict[str, Any],
    *,
    required_bridges: set[tuple[str, str]],
    allowed_message_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate component-level re-review of possible-link bridges."""

    required = {_edge_key(*pair) for pair in required_bridges}
    issues: list[str] = []
    decisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    values = (
        raw.get("bridge_decisions") if isinstance(raw, dict) else None
    )
    if not isinstance(values, list):
        values = []
        issues.append("bridge_decisions_not_list")
    allowed_values = {
        "promote_must",
        "keep_possible",
        "confirm_cannot",
    }
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"bridge_decisions[{index}]:not_object")
            continue
        pair = _edge_key(
            value.get("left_case_ref"), value.get("right_case_ref")
        )
        if not all(pair) or pair not in required:
            issues.append(
                f"bridge_decisions[{index}]:unknown_bridge:"
                f"{pair[0]}:{pair[1]}"
            )
            continue
        if pair in seen:
            issues.append(
                f"bridge_decisions[{index}]:duplicate_bridge:"
                f"{pair[0]}:{pair[1]}"
            )
            continue
        seen.add(pair)
        decision = str(value.get("decision") or "")
        if decision not in allowed_values:
            issues.append(
                f"bridge_decisions[{index}]:invalid_decision:{decision}"
            )
        evidence = dedupe_strings(value.get("evidence_message_ids") or [])
        for message_id in evidence:
            if message_id not in allowed_message_ids:
                issues.append(
                    f"bridge_decisions[{index}]:unknown_message_id:"
                    f"{message_id}"
                )
        reasons = dedupe_strings(value.get("reasons") or [])
        if not reasons:
            issues.append(
                f"bridge_decisions[{index}]:decision_without_reason"
            )
        decisions.append({
            "left_case_ref": pair[0],
            "right_case_ref": pair[1],
            "decision": decision,
            "evidence_message_ids": [
                item for item in evidence
                if item in allowed_message_ids
            ],
            "reasons": reasons,
        })
    for left, right in sorted(required - seen):
        issues.append(f"required_bridge_unaccounted:{left}:{right}")
    normalized = {
        "schema_version": "w7.component_bridge_decision.v1",
        "bridge_decisions": decisions,
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))


def validate_trace_phase_patch(
    raw: dict[str, Any],
    *,
    component_case_refs: set[str],
    allowed_message_ids: set[str],
    allowed_message_ids_by_case: dict[str, set[str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Require a complete, exclusive and acyclic assignment per component."""

    issues: list[str] = []
    values = raw.get("operations") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        values = []
        issues.append("operations_not_list")
    groups: dict[str, list[str]] = {}
    phases: dict[str, dict[str, Any]] = {}
    normalized_ops: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            issues.append(f"operations[{index}]:not_object")
            continue
        op = str(value.get("op") or "")
        trace_ref = str(value.get("local_trace_ref") or "")
        if not trace_ref:
            issues.append(f"operations[{index}]:missing_trace_ref")
            continue
        if op == "create_trace_group":
            case_refs = dedupe_strings(value.get("case_refs") or [])
            if trace_ref in groups:
                issues.append(
                    f"operations[{index}]:duplicate_trace_group:{trace_ref}"
                )
                continue
            unknown = [
                item for item in case_refs
                if item not in component_case_refs
            ]
            for case_ref in unknown:
                issues.append(
                    f"operations[{index}]:unknown_case_ref:{case_ref}"
                )
            case_refs = [
                item for item in case_refs if item in component_case_refs
            ]
            if not case_refs:
                issues.append(f"operations[{index}]:empty_trace_group")
            groups[trace_ref] = case_refs
            normalized_ops.append({
                "op": op,
                "local_trace_ref": trace_ref,
                "case_refs": case_refs,
                "case_ref": "",
                "event_type": "",
                "relation_type": "",
                "phase_index": 0,
                "after_case_ref": "",
                "evidence_message_ids": [],
                "summary": str(value.get("summary") or "").strip(),
            })
        elif op == "set_phase":
            case_ref = str(value.get("case_ref") or "")
            if case_ref not in component_case_refs:
                issues.append(
                    f"operations[{index}]:unknown_case_ref:{case_ref}"
                )
                continue
            if case_ref in phases:
                issues.append(
                    f"operations[{index}]:duplicate_phase:{case_ref}"
                )
                continue
            event_type = str(value.get("event_type") or "")
            relation_type = str(value.get("relation_type") or "")
            try:
                phase_index = int(value.get("phase_index") or 0)
            except (TypeError, ValueError):
                phase_index = 0
                issues.append(
                    f"operations[{index}]:phase_index_not_integer"
                )
            after_case_ref = str(value.get("after_case_ref") or "")
            if event_type not in TRACE_EVENT_TYPES:
                issues.append(
                    f"operations[{index}]:invalid_event_type:{event_type}"
                )
            if relation_type not in TRACE_RELATION_TYPES:
                issues.append(
                    f"operations[{index}]:invalid_relation_type:{relation_type}"
                )
            if phase_index < 1:
                issues.append(
                    f"operations[{index}]:invalid_phase_index:{phase_index}"
                )
            if after_case_ref and after_case_ref not in component_case_refs:
                issues.append(
                    f"operations[{index}]:unknown_after_case:{after_case_ref}"
                )
            evidence = dedupe_strings(
                value.get("evidence_message_ids") or []
            )
            for message_id in evidence:
                if message_id not in allowed_message_ids:
                    issues.append(
                        f"operations[{index}]:unknown_message_id:{message_id}"
                    )
                elif (
                    allowed_message_ids_by_case is not None
                    and case_ref in allowed_message_ids_by_case
                    and message_id
                    not in allowed_message_ids_by_case[case_ref]
                ):
                    issues.append(
                        f"operations[{index}]:message_outside_case:"
                        f"{case_ref}:{message_id}"
                    )
            phase = {
                "op": op,
                "local_trace_ref": trace_ref,
                "case_refs": [],
                "case_ref": case_ref,
                "event_type": event_type,
                "relation_type": relation_type,
                "phase_index": phase_index,
                "after_case_ref": after_case_ref,
                "evidence_message_ids": [
                    item for item in evidence
                    if item in allowed_message_ids
                ],
                "summary": str(value.get("summary") or "").strip(),
            }
            phases[case_ref] = phase
            normalized_ops.append(phase)
        else:
            issues.append(f"operations[{index}]:invalid_op:{op}")
    assigned = [
        case_ref for case_refs in groups.values() for case_ref in case_refs
    ]
    for case_ref in sorted(component_case_refs):
        count = assigned.count(case_ref)
        if count == 0:
            issues.append(f"case_not_assigned_to_trace:{case_ref}")
        elif count > 1:
            issues.append(f"case_assigned_to_multiple_traces:{case_ref}")
        if case_ref not in phases:
            issues.append(f"case_missing_phase:{case_ref}")
    for case_ref, phase in phases.items():
        trace_ref = phase["local_trace_ref"]
        if case_ref not in groups.get(trace_ref, []):
            issues.append(
                f"phase_trace_membership_mismatch:{case_ref}:{trace_ref}"
            )
        after = phase["after_case_ref"]
        if after and after not in groups.get(trace_ref, []):
            issues.append(
                f"phase_after_outside_trace:{case_ref}:{after}"
            )
    for trace_ref, case_refs in groups.items():
        ordinals = [
            int(phases[case_ref]["phase_index"])
            for case_ref in case_refs if case_ref in phases
        ]
        if len(ordinals) != len(set(ordinals)):
            issues.append(f"duplicate_phase_index:{trace_ref}")
        roots = [
            case_ref for case_ref in case_refs
            if case_ref in phases
            and not phases[case_ref]["after_case_ref"]
        ]
        if len(roots) != 1:
            issues.append(f"trace_root_count:{trace_ref}:{len(roots)}")
        for case_ref in case_refs:
            visited: set[str] = set()
            cursor = case_ref
            while cursor and cursor in phases:
                if cursor in visited:
                    issues.append(f"phase_cycle:{trace_ref}:{case_ref}")
                    break
                visited.add(cursor)
                cursor = str(phases[cursor].get("after_case_ref") or "")
    normalized = {
        "schema_version": "w7.trace_phase_patch.v1",
        "operations": normalized_ops,
        "uncertainties": dedupe_strings(
            raw.get("uncertainties") or [] if isinstance(raw, dict) else []
        ),
    }
    normalized["decision_hash"] = canonical_hash(normalized)
    return normalized, sorted(set(issues))
