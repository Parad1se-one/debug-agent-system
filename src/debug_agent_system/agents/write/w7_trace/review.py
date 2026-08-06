"""Content-addressed W7 trace review and correction-event contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .contracts import (
    RESOLUTION_STATUSES,
    TRACE_RELATION_TYPES,
    canonical_hash,
    dedupe_strings,
)


TRACE_CORRECTION_OPERATIONS = (
    "split_case",
    "merge_cases",
    "detach_case",
    "move_phase",
    "change_relation",
    "attach_evidence",
    "detach_evidence",
    "change_status",
)


def trace_review_target_refs(
    trace_review_payload: dict[str, Any],
) -> set[str]:
    refs: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and value:
            if (
                key.endswith("_ref")
                or key.endswith("_id")
                or key in {"case_refs", "source_message_ids"}
            ):
                refs.add(value)

    visit(trace_review_payload)
    return refs


def build_trace_review_payload(
    *,
    source_ledger_hash: str,
    decisions: dict[str, Any],
    compiled_trace_bundle: dict[str, Any],
    dry_run_diff: list[dict[str, Any]] | None = None,
    validator_issues: list[str] | None = None,
    allowed_message_ids: list[str] | None = None,
    case_cards: list[dict[str, Any]] | None = None,
    correction_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = deepcopy(compiled_trace_bundle)
    bundle_hash = canonical_hash(bundle)
    payload = {
        "schema_version": "w7.trace_review_payload.v1",
        "source_ledger_hash": str(source_ledger_hash or ""),
        "decisions": deepcopy(decisions),
        "compiled_trace_bundle": bundle,
        "trace_bundle_hash": bundle_hash,
        "dry_run_diff": deepcopy(dry_run_diff or []),
        "validator_issues": dedupe_strings(validator_issues or []),
        "allowed_message_ids": dedupe_strings(
            allowed_message_ids or []
        ),
        "case_cards": deepcopy(case_cards or []),
    }
    if correction_provenance:
        payload["correction_provenance"] = deepcopy(
            correction_provenance
        )
    payload["review_payload_hash"] = canonical_hash(payload)
    return payload


def correction_event_hash(event: dict[str, Any]) -> str:
    value = {
        key: deepcopy(item)
        for key, item in event.items()
        if key not in {"event_id", "event_hash"}
    }
    return canonical_hash(value)


def approval_subject_hash(item: dict[str, Any]) -> str:
    """Return the exact review subject W6 approves and W5 rechecks."""

    corrected_payload = (
        item.get("corrected_trace_review_payload")
        if isinstance(item.get("corrected_trace_review_payload"), dict)
        else {}
    )
    trace_payload = corrected_payload or (
        item.get("trace_review_payload")
        if isinstance(item.get("trace_review_payload"), dict)
        else {}
    )
    trace_hash = str(
        item.get("trace_bundle_hash")
        or trace_payload.get("trace_bundle_hash")
        or ""
    )
    if trace_hash:
        event_hashes = [
            correction_event_hash(event)
            for event in item.get("correction_events") or []
            if isinstance(event, dict)
        ]
        return "review-subject:" + canonical_hash({
            "trace_bundle_hash": trace_hash,
            "trace_review_payload_hash": canonical_hash(trace_payload),
            "typed_candidate_hash": canonical_hash(
                item.get("typed_candidate")
                if isinstance(item.get("typed_candidate"), dict)
                else {}
            ),
            "correction_event_hashes": event_hashes,
            "applied_correction_overlay_hash": str(
                item.get("applied_correction_overlay_hash") or ""
            ),
        })
    typed = (
        item.get("typed_candidate")
        if isinstance(item.get("typed_candidate"), dict)
        else {}
    )
    candidate = (
        item.get("candidate")
        if isinstance(item.get("candidate"), dict)
        else {}
    )
    return str(
        item.get("content_hash")
        or typed.get("content_hash")
        or candidate.get("content_hash")
        or ""
    )


def correction_chain_subject_hash(
    trace_review_payload: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    """Hash the immutable base payload plus an event prefix.

    Unlike the W6 approval subject, this never switches to the compiled
    corrected payload.  It therefore remains replayable when a reviewer adds
    another event after an intermediate local compilation.
    """

    return approval_subject_hash({
        "trace_review_payload": trace_review_payload,
        "trace_bundle_hash": trace_review_payload.get(
            "trace_bundle_hash"
        ) or "",
        "correction_events": events,
    })


def approval_hash_matches(item: dict[str, Any]) -> bool:
    if not bool(item.get("approval_hash_required")):
        return True
    expected = approval_subject_hash(item)
    approved = str(
        item.get("approved_content_hash")
        or (
            item.get("review_decision", {}).get("approved_content_hash")
            if isinstance(item.get("review_decision"), dict)
            else ""
        )
        or ""
    )
    return bool(expected and approved and expected == approved)


def build_correction_event(
    *,
    review_id: str,
    operation: str,
    target_ref: str,
    payload: dict[str, Any],
    evidence_message_ids: list[str],
    reviewer: str,
    note: str,
    sequence: int,
    base_subject_hash: str,
    allowed_target_refs: set[str] | None = None,
    allowed_message_ids: set[str] | None = None,
    created_at: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    operation_value = str(operation or "")
    if operation_value not in TRACE_CORRECTION_OPERATIONS:
        issues.append(f"unsupported_correction_operation:{operation_value}")
    target = str(target_ref or "")
    if not target:
        issues.append("correction_target_missing")
    if allowed_target_refs is not None and target not in allowed_target_refs:
        issues.append(f"correction_target_unknown:{target}")
    evidence = dedupe_strings(evidence_message_ids)
    if allowed_message_ids is not None:
        for message_id in evidence:
            if message_id not in allowed_message_ids:
                issues.append(
                    f"correction_evidence_unknown:{message_id}"
                )
        evidence = [
            item for item in evidence if item in allowed_message_ids
        ]
    if not isinstance(payload, dict):
        payload = {}
        issues.append("correction_payload_not_object")
    if operation_value == "change_status":
        target_status = str(payload.get("to") or "")
        if target_status not in RESOLUTION_STATUSES:
            issues.append(
                f"correction_invalid_status:{target_status}"
            )
    elif operation_value == "move_phase":
        try:
            phase_index = int(payload.get("phase_index") or 0)
        except (TypeError, ValueError):
            phase_index = 0
        if phase_index < 1:
            issues.append("correction_invalid_phase_index")
    elif operation_value == "change_relation":
        relation = str(payload.get("relation_type") or "")
        if relation not in TRACE_RELATION_TYPES:
            issues.append(
                f"correction_invalid_relation:{relation}"
            )
    elif operation_value in {"attach_evidence", "detach_evidence"}:
        if not evidence:
            issues.append("correction_evidence_required")
    elif operation_value == "split_case":
        new_cases = payload.get("new_cases")
        if not isinstance(new_cases, list) or len(new_cases) < 2:
            issues.append("correction_split_requires_two_cases")
    elif operation_value == "merge_cases":
        case_refs = dedupe_strings(payload.get("case_refs") or [])
        if len(case_refs) < 2:
            issues.append("correction_merge_requires_two_cases")
    sequence_value = int(sequence or 0)
    if sequence_value < 1:
        issues.append(f"correction_sequence_invalid:{sequence_value}")
    event = {
        "schema_version": "w7.human_correction_event.v1",
        "event_id": "",
        "review_id": str(review_id or ""),
        "base_subject_hash": str(base_subject_hash or ""),
        "sequence": sequence_value,
        "operation": operation_value,
        "target_ref": target,
        "payload": deepcopy(payload),
        "evidence_message_ids": evidence,
        "reviewer": str(reviewer or ""),
        "note": str(note or ""),
        "created_at": str(
            created_at or datetime.now(timezone.utc).isoformat()
        ),
    }
    event_hash = correction_event_hash(event)
    event["event_id"] = f"w7-correction:{event_hash[:20]}"
    event["event_hash"] = event_hash
    return event, sorted(set(issues))


def replay_correction_events(
    trace_review_payload: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Replay an immutable correction overlay and produce its content hash.

    The overlay deliberately does not mutate model decisions in place.  A
    later local compiler consumes the ordered operations; W6 approval binds
    the base bundle plus this exact event sequence.
    """

    issues: list[str] = []
    ordered = sorted(
        [deepcopy(item) for item in events if isinstance(item, dict)],
        key=lambda item: int(item.get("sequence") or 0),
    )
    expected_sequence = list(range(1, len(ordered) + 1))
    actual_sequence = [
        int(item.get("sequence") or 0) for item in ordered
    ]
    if actual_sequence != expected_sequence:
        issues.append(
            "correction_sequence_not_contiguous:"
            + ",".join(str(value) for value in actual_sequence)
        )
    seen_ids: set[str] = set()
    replayed_events: list[dict[str, Any]] = []
    for event in ordered:
        event_id = str(event.get("event_id") or "")
        if event_id in seen_ids:
            issues.append(f"duplicate_correction_event:{event_id}")
        seen_ids.add(event_id)
        if str(event.get("operation") or "") not in TRACE_CORRECTION_OPERATIONS:
            issues.append(
                f"unsupported_correction_operation:"
                f"{event.get('operation') or ''}"
            )
        if str(event.get("event_hash") or "") != correction_event_hash(event):
            issues.append(f"correction_event_hash_mismatch:{event_id}")
        expected_base = correction_chain_subject_hash(
            trace_review_payload, replayed_events
        )
        if str(event.get("base_subject_hash") or "") != expected_base:
            issues.append(f"correction_base_hash_mismatch:{event_id}")
        replayed_events.append(event)
    replay = {
        "schema_version": "w7.corrected_trace_overlay.v1",
        "trace_review_payload_hash": str(
            trace_review_payload.get("review_payload_hash")
            or canonical_hash(trace_review_payload)
        ),
        "trace_bundle_hash": str(
            trace_review_payload.get("trace_bundle_hash") or ""
        ),
        "correction_events": ordered,
        "correction_event_hashes": [
            correction_event_hash(event) for event in ordered
        ],
    }
    replay["effective_bundle_hash"] = canonical_hash(replay)
    return replay, sorted(set(issues))
