"""Generic query/task normalization for Read Runtime v3."""

from __future__ import annotations

import re
from typing import Any

from debug_agent_system.incident_runtime.scope import parse_incident_scope
from debug_agent_system.knowledge_v2.query_scope import analyze_query_scope

from .contracts import ReadRequest, ReadTask


_INFORMATION_QUESTION = re.compile(
    r"(?:如何|怎样|怎么|多少|什么|哪里|哪种|哪个|哪些|是否|吗|[?？])"
)
_OBSERVED_INCIDENT_FRAME = re.compile(
    r"(?:"
    r"(?:现场|客户|产线|售后).{0,8}(?:反馈|报告|发现|复现)|"
    r"(?:设备|系统|软件|程序|工控机|相机).{0,12}(?:出现|发生|提示|报警|报错)|"
    r"(?:日志|转储|事件|诊断包).{0,8}(?:报|提示|显示|记录)|"
    r"(?:已经|已).{0,10}(?:拿到|出现|发生|确认|复现)|"
    r"(?:运行|生产|检测|测试).{0,6}(?:中|过程)"
    r")"
)
_CONDITIONAL_PROCEDURE_FRAME = re.compile(
    r"(?:时|后|之后|以前|之前|前).{0,30}"
    r"(?:如何|怎么|怎样|按什么|按照什么|用什么|选择什么|哪些步骤)"
)


def normalize_task(request: ReadRequest, *, budgets: dict[str, int] | None = None) -> ReadTask:
    scope = analyze_query_scope(request.query)
    mode, request_kind, reconciliation_trace = _reconcile_scope(
        query=request.query,
        mode=scope.mode,
        request_kind=scope.request_kind,
        diagnostic_signals=scope.diagnostic_signals,
    )
    source_only_context = request.routing_context.get("source_only_context")
    if isinstance(source_only_context, dict):
        visibility = str(source_only_context.get("label_visibility") or "")
        if visibility in {"source_records_only", "source_only_no_ground_truth"}:
            mode = "source_only_trace_reconstruction"
            request_kind = "trace_reconstruction"
            reconciliation_trace.append(
                "v3_scope_reconciled:explicit_source_only_context"
            )
    task_model = dict(scope.task_model or {})
    raw_facets = (
        task_model.get("facets")
        or task_model.get("required_facets")
        or task_model.get("operations")
        or []
    )
    facet_details, facets = _normalize_facets(raw_facets)
    if not facets:
        facets = [scope.request_kind]
        facet_details = [{
            "facet_id": f"request_kind:{scope.request_kind}",
            "kind": "request_kind",
            "label": scope.request_kind,
            "match_terms": [],
            "required_for_closure": True,
        }]
    entities = list(dict.fromkeys([
        *scope.strong_identifiers,
        *_strings(task_model.get("entities") or []),
        *_strings(task_model.get("subjects") or []),
    ]))
    resource_hints = [
        str(item.get("name") or item.get("path") or item.get("url") or "")
        for item in request.evidence_resources
        if isinstance(item, dict)
    ]
    incident_scope = parse_incident_scope(request.query, resource_hints)
    resource_ids = [
        str(item.get("resource_id") or item.get("path") or item.get("name") or "")
        for item in request.evidence_resources
        if isinstance(item, dict)
    ]
    incident = bool(request.evidence_resources or request.log_summary or incident_scope.has_time_scope)
    multi_source = len([item for item in resource_ids if item]) > 1
    complexity = (
        "incident" if incident else "multi_source" if multi_source or len(facets) > 2
        else "simple" if len(facets) == 1 and not entities else "standard"
    )
    default_budgets = {
        "provider_calls": 12 if incident else 8,
        "tool_calls": 48 if incident else 24,
        "evidence_records": 512 if incident else 256,
        "planner_rounds": 8 if incident else 4,
    }
    default_budgets.update({
        key: max(1, int(value)) for key, value in (budgets or {}).items()
    })
    return ReadTask(
        query=request.query,
        mode=mode,
        request_kind=request_kind,
        facets=facets,
        facet_details=facet_details,
        entities=entities,
        time_windows=[
            {
                "reference_time": item.reference_time,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "precision": item.precision,
                "source_text": item.source_text,
                "year_inferred": item.year_inferred,
            }
            for item in incident_scope.reference_windows
        ],
        resource_ids=[item for item in resource_ids if item],
        complexity=complexity,  # type: ignore[arg-type]
        budgets=default_budgets,
        normalization_trace=[
            *scope.reasons,
            *reconciliation_trace,
            f"complexity:{complexity}",
            f"time_windows:{len(incident_scope.reference_windows)}",
            f"resources:{len(resource_ids)}",
        ],
    )


def _reconcile_scope(
    *, query: str, mode: str, request_kind: str,
    diagnostic_signals: tuple[str, ...] | list[str],
) -> tuple[str, str, list[str]]:
    """Recover information questions from a conservative fault default.

    The frozen v2 scope intentionally falls back to ``fault_diagnosis``. That
    is safe for execution, but it can classify unseen information questions as
    incidents when their verb is outside its phrase inventory. v3 distinguishes
    three generic discourse forms instead of enumerating individual Queries:

    * an explicit observation/report frame stays an incident;
    * a conditional ``X 时/后如何 Y`` question is a procedure lookup;
    * an information question without a fault observation is a lookup.

    Consequently ``电脑不开机，应该怎么排查`` remains diagnosis (an observed
    symptom), while ``电脑无法开机时应该按什么顺序排查`` is treated as asking
    for a reusable procedure.  A leading ``现场反馈`` always wins over the
    conditional/question form.
    """

    text = str(query or "")
    is_question = bool(_INFORMATION_QUESTION.search(text))
    observed_incident = bool(_OBSERVED_INCIDENT_FRAME.search(text))
    conditional_procedure = bool(_CONDITIONAL_PROCEDURE_FRAME.search(text))
    if observed_incident and mode != "out_of_domain":
        if mode != "fault_diagnosis" or request_kind != "fault_diagnosis":
            return (
                "fault_diagnosis",
                "fault_diagnosis",
                ["v3_scope_reconciled:explicit_incident_observation"],
            )
        return mode, request_kind, []
    if (
        mode == "fault_diagnosis"
        and is_question
        and not observed_incident
        and (not diagnostic_signals or conditional_procedure)
    ):
        reason = (
            "v3_scope_reconciled:conditional_procedure_question"
            if conditional_procedure
            else "v3_scope_reconciled:information_question_without_fault_observation"
        )
        return (
            "knowledge_lookup",
            "procedure_lookup",
            [reason],
        )
    return mode, request_kind, []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(dict.fromkeys(
        str(item).strip() for item in value if str(item).strip()
    ))


def _normalize_facets(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep structured query facets while exposing stable human labels.

    Query scope v2 returns facet dictionaries.  Treating those dictionaries as
    strings loses the closure semantics and makes downstream planners parse a
    Python representation.  Scalar facets remain supported for compatibility.
    """

    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return [], []

    details: list[dict[str, Any]] = []
    labels: list[str] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for index, item in enumerate(value):
        if isinstance(item, dict):
            detail = dict(item)
            label = str(detail.get("label") or detail.get("facet_id") or "").strip()
            facet_id = str(detail.get("facet_id") or f"facet:{index}:{label}").strip()
            if not label:
                continue
            detail["facet_id"] = facet_id
            detail["label"] = label
            detail["match_terms"] = _strings(detail.get("match_terms") or [])
            detail["required_for_closure"] = bool(
                detail.get("required_for_closure", True)
            )
        else:
            label = str(item).strip()
            if not label:
                continue
            facet_id = f"facet:{index}:{label}"
            detail = {
                "facet_id": facet_id,
                "kind": "generic",
                "label": label,
                "match_terms": [],
                "required_for_closure": True,
            }
        normalized_label = label.casefold()
        if facet_id in seen_ids or normalized_label in seen_labels:
            continue
        seen_ids.add(facet_id)
        seen_labels.add(normalized_label)
        details.append(detail)
        labels.append(label)
    return details, labels
