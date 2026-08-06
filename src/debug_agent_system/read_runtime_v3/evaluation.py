"""Evaluation projection and structural gates for Read Runtime v3.

The formal benchmark owns its Gold and scoring semantics.  This module only
projects one v3 response into that existing prediction contract and checks
v3-specific invariants.  It never reads a benchmark case or Gold fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


_PREDICTION_STATUSES = {
    "answer", "step", "ask_info", "resolved", "escalate", "unsupported",
}


def response_to_formal_prediction(response: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(response.get("baseline_response") or {})
    metadata = dict(baseline.get("metadata") or {})
    evidence_pack = dict(metadata.get("evidence_pack") or {})
    runtime_decision = dict(evidence_pack.get("runtime_decision") or {})
    document_mode = dict(metadata.get("document_answer_mode") or {})
    retrieval = dict(metadata.get("retrieval") or {})
    task = dict(response.get("task") or {})
    answer_plan = dict(response.get("answer_plan") or {})
    context_records = _source_context_records(response)
    kg_records = _kg_candidate_records(response)

    route_ids: list[str] = []
    evidence_ids: list[str] = []
    _extend_ids(route_ids, (
        baseline.get("family_id"), baseline.get("variant_id"), baseline.get("plan_id"),
    ))
    _extend_ids(evidence_ids, baseline.get("evidence_ids") or [])

    for item in evidence_pack.get("source_items") or []:
        if not isinstance(item, dict):
            continue
        _extend_ids(route_ids, (item.get("document_id"), item.get("object_id")))
        _extend_ids(evidence_ids, (
            item.get("document_id"), item.get("object_id"),
            *(item.get("evidence_ids") or []),
            *(item.get("chunk_ids") or []),
        ))
    for item in retrieval.get("supporting_chunks") or []:
        if not isinstance(item, dict):
            continue
        _extend_ids(route_ids, (item.get("document_id"), item.get("object_id")))
        _extend_ids(evidence_ids, (
            item.get("document_id"), item.get("object_id"), item.get("chunk_id"),
            *(item.get("evidence_ids") or []),
        ))
    for candidate in retrieval.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        _extend_ids(evidence_ids, candidate.get("evidence_ids") or [])

    allowed = dict(evidence_pack.get("allowed_references") or {})
    _extend_ids(evidence_ids, allowed.get("evidence_ids") or [])
    _extend_ids(evidence_ids, allowed.get("chunk_ids") or [])
    _extend_ids(
        evidence_ids,
        (record.get("source_ref") for record in context_records),
    )
    for record in kg_records:
        content = dict(record.get("content") or {})
        _extend_ids(route_ids, (content.get("family_id"), content.get("variant_id")))
        _extend_ids(evidence_ids, (
            content.get("family_id"), content.get("variant_id"),
            record.get("source_ref"),
        ))
    diagnostic_trace = dict(evidence_pack.get("diagnostic_trace") or {})
    action_ids = [str(value) for value in diagnostic_trace.get("action_ids") or [] if str(value)]
    traces = [item for item in answer_plan.get("traces") or [] if isinstance(item, dict)]
    for trace in traces:
        _extend_ids(route_ids, (trace.get("trace_id"),))
        _extend_ids(evidence_ids, trace.get("evidence_ids") or [])

    route_type = _route_type(
        baseline=baseline,
        metadata=metadata,
        document_mode=document_mode,
        retrieval=retrieval,
        task=task,
        evidence_pack=evidence_pack,
        context_records=context_records,
        kg_records=kg_records,
    )
    status = _prediction_status(
        baseline=baseline,
        document_mode=document_mode,
        current_action_id=str(baseline.get("current_action_id") or ""),
    )
    return {
        "case_id": "",
        "answer": str((response.get("shadow") or {}).get("proposed_answer") or response.get("answer") or ""),
        "route_type": route_type,
        "route_ids": route_ids,
        "evidence_ids": evidence_ids,
        "family_id": str(
            baseline.get("family_id") or runtime_decision.get("family_id") or ""
        ),
        "variant_id": str(
            baseline.get("variant_id") or runtime_decision.get("variant_id") or ""
        ),
        "first_action_id": str(
            baseline.get("current_action_id")
            or diagnostic_trace.get("current_action_id")
            or (action_ids[0] if action_ids else "")
        ),
        "followup_ids": _followup_ids(baseline),
        "status": status,
        "executed_action_ids": [],
        "trace_count": len(traces) if traces else len(action_ids),
    }


def structural_errors(response: dict[str, Any]) -> list[str]:
    """Return runtime-contract failures independent of benchmark Gold."""

    errors: list[str] = []
    baseline = dict(response.get("baseline_response") or {})
    shadow = dict(response.get("shadow") or {})
    task = dict(response.get("task") or {})
    verification = dict(response.get("verification") or {})
    snapshot = dict(response.get("evidence_snapshot") or {})
    records = list(snapshot.get("records") or [])
    evidence_ids = {
        str(item.get("evidence_id") or "")
        for item in records if isinstance(item, dict)
    }

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    require(
        response.get("schema_version") == "debug_agent_system.read_response.v3",
        "response_schema_mismatch",
    )
    require(bool(str(response.get("query") or "").strip()), "query_missing")
    require(bool(shadow.get("enabled")), "shadow_not_enabled")
    require(response.get("answer") == baseline.get("answer"), "official_answer_drift")
    require(response.get("status") == baseline.get("status"), "official_status_drift")
    require(bool(verification.get("passed")), "answer_plan_verification_failed")
    require(int(snapshot.get("record_count") or 0) == len(records), "record_count_mismatch")
    require(bool(records), "evidence_fabric_empty")
    require(bool(str(snapshot.get("fingerprint") or "")), "evidence_fingerprint_missing")
    facets = list(task.get("facets") or [])
    require(bool(facets), "task_facets_empty")
    require(
        not any(str(value).lstrip().startswith("{") for value in facets),
        "task_facet_serialized_mapping",
    )
    details = list(task.get("facet_details") or [])
    require(len(details) == len(facets), "task_facet_detail_count_mismatch")
    require(
        all(isinstance(item, dict) and item.get("facet_id") and item.get("label") for item in details),
        "task_facet_detail_invalid",
    )
    for section in (response.get("answer_plan") or {}).get("sections") or []:
        if not isinstance(section, dict):
            errors.append("answer_plan_section_invalid")
            continue
        for claim in section.get("claims") or []:
            if not isinstance(claim, dict):
                errors.append("answer_plan_claim_invalid")
                continue
            claim_id = str(claim.get("claim_id") or "missing")
            cited = [str(value) for value in claim.get("evidence_ids") or []]
            require(bool(cited), f"claim_without_evidence:{claim_id}")
            for evidence_id in cited:
                require(
                    evidence_id in evidence_ids,
                    f"claim_unknown_evidence:{claim_id}:{evidence_id}",
                )
    provider_stages = {
        str(item.get("stage") or ""): str(item.get("status") or "")
        for item in response.get("trace") or [] if isinstance(item, dict)
    }
    require(provider_stages.get("provider:baseline") == "ok", "baseline_provider_failed")
    require(provider_stages.get("planner") in {"ok", "fallback"}, "planner_trace_missing")
    return list(dict.fromkeys(errors))


def _route_type(
    *,
    baseline: dict[str, Any],
    metadata: dict[str, Any],
    document_mode: dict[str, Any],
    retrieval: dict[str, Any],
    task: dict[str, Any],
    evidence_pack: dict[str, Any],
    context_records: list[dict[str, Any]],
    kg_records: list[dict[str, Any]],
) -> str:
    failure = str(baseline.get("failure_type") or "").lower()
    scope = dict(metadata.get("query_scope") or {})
    if failure in {"out_of_domain", "unsupported", "unsupported_external_knowledge"}:
        return "out_of_domain"
    if str(scope.get("mode") or "") == "out_of_domain":
        return "out_of_domain"
    if any(bool((item.get("metadata") or {}).get("source_only")) for item in context_records):
        return "source_only_trace_reconstruction"
    if document_mode.get("active") and str(task.get("mode") or "") != "fault_diagnosis":
        return "knowledge_document_section"
    if (
        str(task.get("mode") or "") == "knowledge_lookup"
        and _has_document_evidence(evidence_pack=evidence_pack, retrieval=retrieval)
    ):
        return "knowledge_document_section"
    routes = {
        str(item.get("route") or "")
        for item in retrieval.get("candidates") or [] if isinstance(item, dict)
    }
    if "source_only_trace_reconstruction" in routes:
        return "source_only_trace_reconstruction"
    if baseline.get("variant_id") or baseline.get("family_id") or retrieval or kg_records:
        return "sag_v2_native"
    return ""


def _source_context_records(response: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = dict(response.get("evidence_snapshot") or {})
    return [
        item for item in snapshot.get("records") or []
        if isinstance(item, dict) and item.get("provider") == "request_context"
    ]


def _kg_candidate_records(response: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = dict(response.get("evidence_snapshot") or {})
    return [
        item for item in snapshot.get("records") or []
        if isinstance(item, dict)
        and item.get("provider") == "kg_v2_sag"
        and item.get("kind") == "kg_object"
        and isinstance(item.get("content"), dict)
        and item.get("content", {}).get("variant_id")
    ]


def _has_document_evidence(
    *, evidence_pack: dict[str, Any], retrieval: dict[str, Any]
) -> bool:
    """Return whether selected/retrieved facts have document source anchors."""

    items = [
        *(evidence_pack.get("source_items") or []),
        *(retrieval.get("supporting_chunks") or []),
    ]
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("document_id") or "").strip():
            return True
        values = [
            item.get("document_id"), item.get("object_id"), item.get("chunk_id"),
            *(item.get("chunk_ids") or []),
        ]
        if any("knowledge-document:" in str(value or "") for value in values):
            return True
    return False


def _prediction_status(
    *, baseline: dict[str, Any], document_mode: dict[str, Any], current_action_id: str
) -> str:
    value = str(baseline.get("status") or "")
    if document_mode.get("active") and value == "step" and not current_action_id:
        return "answer"
    if value == "failed" and str(baseline.get("failure_type") or ""):
        return "unsupported"
    return value if value in _PREDICTION_STATUSES else "unsupported"


def _followup_ids(baseline: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in baseline.get("required_data") or []:
        if isinstance(item, dict):
            value = str(item.get("required_info_id") or item.get("id") or "")
        else:
            value = str(item)
        if value and value not in values:
            values.append(value)
    return values


def _extend_ids(target: list[str], values: Iterable[Any]) -> None:
    for value in values:
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)
