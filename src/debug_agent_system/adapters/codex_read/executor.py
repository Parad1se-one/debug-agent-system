"""Bounded, read-only KG_v2/SAG tools exposed to Codex."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from debug_agent_system.agents.tools.executor import (
    ReadEvidenceToolExecutor,
    parse_evidence_tool_schema,
)
from debug_agent_system.agents.read.codex_answer import (
    CodexEvidenceAnswerVerifier,
)
from debug_agent_system.agents.read.evidence_answer import (
    render_answer_sections,
)
from debug_agent_system.agents.read.evidence_pack import EvidencePack
from debug_agent_system.core.contracts import AnswerSection, to_jsonable
from debug_agent_system.runtime.system import DebugAgentSystem


def _resource_schema() -> dict[str, Any]:
    return dict(
        parse_evidence_tool_schema()["function"]["parameters"]["properties"][
            "resource"
        ]
    )


def _function(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def read_side_tool_schemas() -> list[dict[str, Any]]:
    """Return the complete Codex read surface.

    None of these tools can execute an action, choose a branch, approve risk,
    mutate KG_v2, or mark a diagnosis resolved.
    """

    resource = _resource_schema()
    incident_tools = _incident_tool_schemas(resource)
    return [
        _function(
            "diagnose_start",
            (
                "Start the deterministic KG_v2 diagnosis runtime. Only this "
                "runtime may lock a Variant or compile diagnostic actions."
            ),
            {
                "query": {"type": "string"},
                "interactive": {"type": "boolean"},
                "session_id": {"type": "string"},
                "routing_context": {
                    "type": "object",
                    "properties": {
                        "stage": {"type": "string"},
                        "query_type": {"type": "string"},
                        "interface": {"type": "string"},
                        "side": {"type": "string"},
                    },
                    "required": ["stage", "query_type", "interface", "side"],
                    "additionalProperties": False,
                },
                "evidence_resources": {
                    "type": "array",
                    "items": resource,
                    "maxItems": 12,
                },
            },
            [
                "query",
                "interactive",
                "session_id",
                "routing_context",
                "evidence_resources",
            ],
        ),
        _function(
            "diagnose_step",
            (
                "Continue an existing deterministic diagnosis session using "
                "user text and caller-supplied evidence resources."
            ),
            {
                "session_id": {"type": "string"},
                "user_message": {"type": "string"},
                "evidence_resources": {
                    "type": "array",
                    "items": resource,
                    "maxItems": 12,
                },
            },
            ["session_id", "user_message", "evidence_resources"],
        ),
        _function(
            "retrieve_evidence",
            (
                "Recall KG_v2 FaultVariant candidates and approved source "
                "chunks from both SAG channels without creating a session."
            ),
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            ["query", "limit"],
        ),
        _function(
            "expand_document_context",
            (
                "Expand explicitly selected source documents into their full "
                "approved semantic outline, preserving chunk order and media."
            ),
            {
                "query": {"type": "string"},
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "max_chunks": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                },
            },
            ["query", "document_ids", "max_chunks"],
        ),
        _function(
            "inspect_kg_path",
            (
                "Inspect one KG_v2 Family/Variant diagnostic path including "
                "Trace actions, outcomes, branch conditions, evidence and risk "
                "flags. This tool never selects a branch or executes an action."
            ),
            {
                "family_id": {"type": "string"},
                "variant_id": {"type": "string"},
            },
            ["family_id", "variant_id"],
        ),
        _function(
            "inspect_source_assets",
            (
                "List source images and attachments bound to selected approved "
                "document chunks, including their captions and provenance."
            ),
            {
                "query": {"type": "string"},
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "max_items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            ["query", "document_ids", "max_items"],
        ),
        _function(
            "render_evidence_answer",
            (
                "Submit a source-closed answer plan after investigation. The "
                "local verifier checks every item ID, required fact, query "
                "facet and section type, then renders canonical local text. "
                "This tool cannot change diagnosis or safety state."
            ),
            {
                "session_id": {"type": "string"},
                "answer_sections": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_type": {
                                "type": "string",
                                "enum": [
                                    "known",
                                    "diagnostic_steps",
                                    "document_guidance",
                                    "conditions",
                                    "uncertainty",
                                    "required_info",
                                ],
                            },
                            "source_item_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "section_type",
                            "source_item_ids",
                        ],
                        "additionalProperties": False,
                    },
                },
                "covered_query_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "uncovered_query_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            [
                "session_id",
                "answer_sections",
                "covered_query_facets",
                "uncovered_query_facets",
            ],
        ),
        parse_evidence_tool_schema(),
        *incident_tools,
    ]


def _incident_tool_schemas(resource: dict[str, Any]) -> list[dict[str, Any]]:
    case_id = {"type": "string"}
    return [
        _function(
            "parse_incident_scope",
            "Normalize reference times in a Jira/query into independent bounded local-time windows before package parsing.",
            {
                "query": {"type": "string"},
                "resource_hints": {"type": "array", "items": {"type": "string"}, "maxItems": 48},
            },
            ["query", "resource_hints"],
        ),
        _function(
            "analyze_incident",
            "Create an immutable incident snapshot, parse diagnostic artifacts, query KG_v2 hypotheses and build a locally verified Jira report.",
            {
                "query": {"type": "string"},
                "evidence_resources": {"type": "array", "items": resource, "maxItems": 24},
                "log_summary": {"type": "string", "description": "Optional JSON object serialized as a string."},
            },
            ["query", "evidence_resources", "log_summary"],
        ),
        _function(
            "index_log_package",
            "Alias for incident intake when the caller explicitly wants an immutable diagnostic package index before deeper inspection.",
            {
                "query": {"type": "string"},
                "evidence_resources": {"type": "array", "items": resource, "maxItems": 24},
                "log_summary": {"type": "string", "description": "Optional JSON object serialized as a string."},
            },
            ["query", "evidence_resources", "log_summary"],
        ),
        _function("get_jira_snapshot", "Read the immutable Jira/case snapshot.", {"case_id": case_id}, ["case_id"]),
        _function("get_incident_scope", "Read the normalized query time scope used during artifact selection.", {"case_id": case_id}, ["case_id"]),
        _function("get_incident_evidence_pack", "Return the source-closed Incident Evidence Pack v3 for local or model-side synthesis.", {"case_id": case_id}, ["case_id"]),
        _function("list_artifacts", "List source and archive-member artifacts with hashes and parser states.", {"case_id": case_id}, ["case_id"]),
        _function(
            "inspect_archive_manifest",
            "Inspect archive ancestry and members without executing or extracting attachments to caller paths.",
            {"case_id": case_id, "artifact_id": {"type": "string"}},
            ["case_id", "artifact_id"],
        ),
        _function(
            "search_diagnostic_events",
            "Search normalized diagnostic events by error code, component, function or message.",
            {"case_id": case_id, "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
            ["case_id", "query", "limit"],
        ),
        _function(
            "search_diagnostic_events_by_time",
            "Search normalized events inside an ISO local-time interval after scoped extraction.",
            {
                "case_id": case_id,
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            ["case_id", "start_time", "end_time", "query", "limit"],
        ),
        _function(
            "extract_log_time_windows",
            "Return immutable, source-line-bound windows that were streamed from archive members using the query time scope.",
            {"case_id": case_id},
            ["case_id"],
        ),
        _function(
            "read_log_window",
            "Read a bounded source-log window around a 1-based line number.",
            {
                "case_id": case_id,
                "artifact_id": {"type": "string"},
                "line": {"type": "integer", "minimum": 1},
                "before": {"type": "integer", "minimum": 0, "maximum": 100},
                "after": {"type": "integer", "minimum": 0, "maximum": 200},
            },
            ["case_id", "artifact_id", "line", "before", "after"],
        ),
        _function("build_incident_timeline", "Return the normalized cross-artifact incident timeline.", {"case_id": case_id}, ["case_id"]),
        _function(
            "inspect_stacktrace",
            "Inspect normalized stack frames while preserving detection-point versus root-cause boundaries.",
            {"case_id": case_id, "trace_id": {"type": "string"}},
            ["case_id", "trace_id"],
        ),
        _function("inspect_environment", "Inspect source-bound software, driver, OS and hardware version observations.", {"case_id": case_id}, ["case_id"]),
        _function("inspect_evtx", "Return source-bound Windows provider/event/time/data records, including query-local time alignment, for the selected EVTX artifact.", {"case_id": case_id, "artifact_id": {"type": "string"}}, ["case_id", "artifact_id"]),
        _function("inspect_dump", "Return bounded minidump process, exception, OS and loaded-module metadata; symbolized stacks remain a separately reported capability.", {"case_id": case_id, "artifact_id": {"type": "string"}}, ["case_id", "artifact_id"]),
        _function("query_kg_hypotheses", "Return KG_v2 candidates as a support/contradiction/missing-evidence hypothesis matrix.", {"case_id": case_id}, ["case_id"]),
        _function("retrieve_similar_cases", "Return lexically related SourceCase records as non-authoritative clues, never as formal knowledge or proof.", {"case_id": case_id, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["case_id", "limit"]),
        _function("propose_next_tests", "Return safe next-best tests ranked by information gain, risk and cost.", {"case_id": case_id}, ["case_id"]),
        _function(
            "plan_reproduction",
            "Build a non-executing controlled-reproduction and observation plan. It never controls production equipment or runs attachment scripts.",
            {"case_id": case_id},
            ["case_id"],
        ),
        _function(
            "compare_reproduction_runs",
            "Compare two immutable incident runs for recurring signatures. A match proves recurrence, not automatically controlled reproduction or a verified fix.",
            {"baseline_case_id": case_id, "candidate_case_id": case_id},
            ["baseline_case_id", "candidate_case_id"],
        ),
        _function(
            "compare_incident_environments",
            "Compare two parsed incident environment snapshots without treating differences as root cause.",
            {"left_case_id": case_id, "right_case_id": case_id},
            ["left_case_id", "right_case_id"],
        ),
        _function("render_incident_report", "Return the locally verified Jira-friendly report; this tool cannot change Jira state.", {"case_id": case_id}, ["case_id"]),
    ]


class CodexReadSideToolExecutor:
    """Execute the bounded read-only tool surface."""

    allowed_tools = {
        "diagnose_start",
        "diagnose_step",
        "retrieve_evidence",
        "expand_document_context",
        "inspect_kg_path",
        "inspect_source_assets",
        "render_evidence_answer",
        "parse_evidence",
        "parse_incident_scope",
        "analyze_incident",
        "index_log_package",
        "get_jira_snapshot",
        "get_incident_scope",
        "get_incident_evidence_pack",
        "list_artifacts",
        "inspect_archive_manifest",
        "search_diagnostic_events",
        "search_diagnostic_events_by_time",
        "extract_log_time_windows",
        "read_log_window",
        "build_incident_timeline",
        "inspect_stacktrace",
        "inspect_environment",
        "inspect_evtx",
        "inspect_dump",
        "query_kg_hypotheses",
        "retrieve_similar_cases",
        "propose_next_tests",
        "plan_reproduction",
        "compare_reproduction_runs",
        "compare_incident_environments",
        "render_incident_report",
    }

    def __init__(
        self,
        system: DebugAgentSystem | None = None,
        evidence_executor: ReadEvidenceToolExecutor | None = None,
    ) -> None:
        self.system = system or DebugAgentSystem.from_config()
        self.evidence_executor = evidence_executor or ReadEvidenceToolExecutor()
        self.verifier = CodexEvidenceAnswerVerifier()
        self._responses: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str = "",
    ) -> dict[str, Any]:
        if name not in self.allowed_tools:
            return self._error(name, call_id, "tool_not_allowed")
        if name == "diagnose_start":
            session_id = str(arguments.get("session_id") or "")
            response = self.system.start({
                "query": str(arguments.get("query") or ""),
                "interactive": bool(arguments.get("interactive", True)),
                "session": {"session_id": session_id} if session_id else {},
                "routing_context": dict(arguments.get("routing_context") or {}),
                "evidence_resources": list(
                    arguments.get("evidence_resources") or []
                ),
            })
            self._remember_response(response)
            return response
        if name == "diagnose_step":
            response = self.system.step(
                str(arguments.get("session_id") or ""),
                str(arguments.get("user_message") or ""),
                evidence_resources=list(
                    arguments.get("evidence_resources") or []
                ),
            )
            self._remember_response(response)
            return response
        if name == "retrieve_evidence":
            return self._retrieve(arguments)
        if name == "expand_document_context":
            return self._expand_documents(arguments)
        if name == "inspect_kg_path":
            return self._inspect_kg_path(arguments)
        if name == "inspect_source_assets":
            return self._inspect_assets(arguments)
        if name == "render_evidence_answer":
            return self._render_evidence_answer(arguments)
        if name in {
            "parse_incident_scope",
            "analyze_incident",
            "index_log_package",
            "get_jira_snapshot",
            "get_incident_scope",
            "get_incident_evidence_pack",
            "list_artifacts",
            "inspect_archive_manifest",
            "search_diagnostic_events",
            "search_diagnostic_events_by_time",
            "extract_log_time_windows",
            "read_log_window",
            "build_incident_timeline",
            "inspect_stacktrace",
            "inspect_environment",
            "inspect_evtx",
            "inspect_dump",
            "query_kg_hypotheses",
            "retrieve_similar_cases",
            "propose_next_tests",
            "plan_reproduction",
            "compare_reproduction_runs",
            "compare_incident_environments",
            "render_incident_report",
        }:
            return self._incident(name, arguments)
        result = self.evidence_executor.execute(
            arguments.get("resource") or {},
            tool=str(arguments.get("tool") or "auto"),
            max_bytes=int(arguments.get("max_bytes") or 65536),
            call_id=call_id,
        )
        return to_jsonable(result)

    def _incident(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        runtime = self.system.incident_runtime
        if name == "parse_incident_scope":
            from debug_agent_system.incident_runtime.scope import parse_incident_scope

            return {
                "schema_version": "debug_agent_system.incident_scope.v1",
                "status": "ok",
                "scope": parse_incident_scope(
                    str(arguments.get("query") or ""),
                    list(arguments.get("resource_hints") or []),
                ).to_dict(),
            }
        if name in {"analyze_incident", "index_log_package"}:
            result = runtime.analyze(
                str(arguments.get("query") or ""),
                list(arguments.get("evidence_resources") or []),
                log_summary=_json_object(arguments.get("log_summary")),
            )
            if name == "analyze_incident":
                return result.to_dict()
            return {
                "schema_version": "debug_agent_system.incident_package_index.v1",
                "status": result.status,
                "case_id": result.case.case_id,
                "artifact_count": len(result.case.artifacts),
                "event_count": len(result.events),
                "stack_trace_count": len(result.stack_traces),
                "exclusions": result.exclusions,
            }
        if name == "compare_incident_environments":
            left = runtime.get(str(arguments.get("left_case_id") or ""))
            right = runtime.get(str(arguments.get("right_case_id") or ""))
            if left is None or right is None:
                return self._error(name, "", "incident_case_not_available")
            keys = sorted(set(left.environment.values) | set(right.environment.values))
            differences = [
                {
                    "field": key,
                    "left": left.environment.values.get(key) or [],
                    "right": right.environment.values.get(key) or [],
                }
                for key in keys
                if left.environment.values.get(key) != right.environment.values.get(key)
            ]
            return {
                "schema_version": "debug_agent_system.incident_environment_compare.v1",
                "status": "ok",
                "differences": differences,
                "causality_asserted": False,
            }
        if name == "compare_reproduction_runs":
            left = runtime.get(str(arguments.get("baseline_case_id") or ""))
            right = runtime.get(str(arguments.get("candidate_case_id") or ""))
            if left is None or right is None:
                return self._error(name, "", "incident_case_not_available")
            return runtime.compare_runs(left, right)
        case_id = str(arguments.get("case_id") or "")
        result = runtime.get(case_id)
        if result is None:
            return self._error(name, "", "incident_case_not_available")
        if name == "get_jira_snapshot":
            return {"schema_version": "debug_agent_system.incident_snapshot.v1", "status": "ok", "case": to_jsonable(result.case)}
        if name == "get_incident_scope":
            return {
                "schema_version": "debug_agent_system.incident_scope.v1",
                "status": "ok",
                "scope": runtime.scope(case_id),
            }
        if name == "get_incident_evidence_pack":
            return dict(result.evidence_pack)
        if name == "list_artifacts":
            return {"schema_version": "debug_agent_system.incident_artifacts.v1", "status": "ok", "artifacts": to_jsonable(result.case.artifacts)}
        if name == "inspect_archive_manifest":
            artifact_id = str(arguments.get("artifact_id") or "")
            artifacts = [
                item for item in result.case.artifacts
                if not artifact_id
                or item.artifact_id == artifact_id
                or item.parent_artifact_id == artifact_id
            ]
            return {"schema_version": "debug_agent_system.incident_archive_manifest.v1", "status": "ok", "artifacts": to_jsonable(artifacts)}
        if name == "search_diagnostic_events":
            needle = str(arguments.get("query") or "").lower()
            limit = max(1, min(int(arguments.get("limit") or 50), 200))
            events = [
                event for event in result.events
                if not needle
                or needle in " ".join([
                    event.message,
                    event.component,
                    event.module,
                    event.function,
                    *event.error_codes,
                ]).lower()
            ][:limit]
            return {"schema_version": "debug_agent_system.incident_event_search.v1", "status": "ok", "events": to_jsonable(events)}
        if name == "search_diagnostic_events_by_time":
            start_time = str(arguments.get("start_time") or "")
            end_time = str(arguments.get("end_time") or "")
            needle = str(arguments.get("query") or "").lower()
            limit = max(1, min(int(arguments.get("limit") or 50), 200))
            events = [
                event
                for event in result.events
                if _timestamp_between(event.timestamp_utc or event.timestamp_raw, start_time, end_time)
                and (
                    not needle
                    or needle in " ".join([
                        event.message, event.component, event.module,
                        event.function, *event.error_codes,
                    ]).lower()
                )
            ][:limit]
            return {
                "schema_version": "debug_agent_system.incident_event_time_search.v1",
                "status": "ok",
                "events": to_jsonable(events),
            }
        if name == "extract_log_time_windows":
            artifacts = [
                item
                for item in result.case.artifacts
                if item.metadata.get("derived_by") == "query_time_window_stream"
            ]
            return {
                "schema_version": "debug_agent_system.incident_time_windows.v1",
                "status": "ok",
                "scope": runtime.scope(case_id),
                "artifacts": to_jsonable(artifacts),
                "window_count": len(artifacts),
            }
        if name == "read_log_window":
            return runtime.log_window(
                case_id,
                str(arguments.get("artifact_id") or ""),
                int(arguments.get("line") or 1),
                before=int(arguments.get("before") or 10),
                after=int(arguments.get("after") or 20),
            )
        if name == "build_incident_timeline":
            return {
                "schema_version": "debug_agent_system.incident_timeline.v1",
                "status": "ok",
                "timeline": result.timeline,
                "correlations": result.correlations,
            }
        if name == "inspect_stacktrace":
            trace_id = str(arguments.get("trace_id") or "")
            traces = [trace for trace in result.stack_traces if not trace_id or trace.trace_id == trace_id]
            return {
                "schema_version": "debug_agent_system.incident_stacktrace.v1",
                "status": "ok",
                "stack_traces": to_jsonable(traces),
                "detection_point_is_root_cause": False,
            }
        if name == "inspect_environment":
            return {"schema_version": "debug_agent_system.incident_environment.v1", "status": "ok", "environment": to_jsonable(result.environment)}
        if name == "plan_reproduction":
            signatures = sorted({
                "|".join([*event.error_codes, event.event_kind, event.component, event.function]).strip("|")
                for event in result.events
                if event.error_codes or event.event_kind != "diagnostic_event"
            })
            return {
                "schema_version": "debug_agent_system.reproduction_plan.v1",
                "status": "planned",
                "case_id": case_id,
                "mode": "observe_then_controlled_compare",
                "observed_signatures": [item for item in signatures if item],
                "steps": [
                    {"step": 1, "action": "冻结原始诊断包、hash、参考时间和当前环境版本矩阵", "execution": "read_only"},
                    {"step": 2, "action": "在测试环境启动有界日志、GPU/内存与进程生命周期观测", "execution": "approved_observer_adapter"},
                    {"step": 3, "action": "由人工或已批准设备适配器执行单变量操作路径并记录开始/结束标记", "execution": "human_or_approved_adapter"},
                    {"step": 4, "action": "停止采集并建立新的不可变 Incident Case", "execution": "read_only"},
                    {"step": 5, "action": "调用 compare_reproduction_runs 比较签名、环境和时间闭环", "execution": "read_only"},
                ],
                "automatic_device_control": False,
                "attachment_scripts_executed": False,
                "single_non_occurrence_verifies_fix": False,
            }
        if name in {"inspect_evtx", "inspect_dump"}:
            artifact_id = str(arguments.get("artifact_id") or "")
            artifact = next(
                (item for item in result.case.artifacts if item.artifact_id == artifact_id),
                None,
            )
            if artifact is None:
                return self._error(name, "", "incident_artifact_not_available")
            suffix = str(artifact.name).lower()
            expected = suffix.endswith(".evtx") if name == "inspect_evtx" else suffix.endswith((".dmp", ".mdmp"))
            if not expected:
                return self._error(name, "", "incident_artifact_type_mismatch")
            exclusions = [
                item for item in result.exclusions
                if str(item.get("artifact_id") or "") == artifact_id
            ]
            return {
                "schema_version": f"debug_agent_system.{name}.v1",
                "status": "parsed" if artifact.parser_state == "parsed" else "metadata_only",
                "artifact": to_jsonable(artifact),
                "events": to_jsonable([
                    event for event in result.events if event.artifact_id == artifact_id
                ]),
                "stack_traces": to_jsonable([
                    trace for trace in result.stack_traces if trace.artifact_id == artifact_id
                ]),
                "exclusions": exclusions,
                "attachment_executed": False,
            }
        if name == "query_kg_hypotheses":
            return {
                "schema_version": "debug_agent_system.incident_hypotheses.v1",
                "status": "ok",
                "retrieval": result.retrieval,
                "hypotheses": to_jsonable(result.hypotheses),
            }
        if name == "retrieve_similar_cases":
            limit = max(1, min(int(arguments.get("limit") or 10), 20))
            query_terms = _search_terms(" ".join([
                result.case.query,
                *[
                    str(item.get("value") or "")
                    for item in result.retrieval.get("anchors") or []
                    if item.get("stability") != "volatile"
                ],
            ]))
            ranked: list[tuple[float, str, dict[str, Any]]] = []
            for source_case_id, raw in (self.system.read_model.by_type.get("SourceCase") or {}).items():
                if not isinstance(raw, dict):
                    continue
                text = " ".join(str(value) for value in raw.values() if isinstance(value, (str, int, float)))
                terms = _search_terms(text)
                overlap = len(query_terms & terms)
                if overlap:
                    ranked.append((overlap / max(len(query_terms), 1), str(source_case_id), raw))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            return {
                "schema_version": "debug_agent_system.incident_similar_cases.v1",
                "status": "ok",
                "cases": [
                    {"source_case_id": case_id, "score": round(score, 4), "record": raw}
                    for score, case_id, raw in ranked[:limit]
                ],
                "formal_knowledge": False,
                "causality_asserted": False,
            }
        if name == "propose_next_tests":
            return {"schema_version": "debug_agent_system.incident_next_tests.v1", "status": "ok", "next_tests": to_jsonable(result.next_tests)}
        return {
            "schema_version": "debug_agent_system.incident_report.v1",
            "status": result.status,
            "case_id": result.case.case_id,
            "answer": result.report,
            "report": result.report,
            "verification": result.verification,
            "metadata": {
                "incident_runtime": {
                    "case_id": result.case.case_id,
                    "evidence_pack_schema": result.evidence_pack.get("schema_version"),
                }
            },
            "jira_mutated": False,
        }

    def _remember_response(self, response: dict[str, Any]) -> None:
        session_id = str(response.get("session_id") or "")
        if session_id:
            self._responses[session_id] = response

    def _render_evidence_answer(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        session_id = str(arguments.get("session_id") or "")
        response = self._responses.get(session_id)
        if response is None:
            return self._error(
                "render_evidence_answer",
                "",
                "diagnosis_response_not_available",
            )
        payload = dict(
            (response.get("metadata") or {}).get("evidence_pack") or {}
        )
        raw_items = payload.get("source_items")
        if (
            payload.get("schema_version")
            != "debug_agent_system.answer_evidence_pack.v2"
            or not isinstance(raw_items, list)
        ):
            return self._error(
                "render_evidence_answer",
                "",
                "evidence_pack_not_available",
            )
        source_items = {
            str(item.get("item_id") or ""): dict(item)
            for item in raw_items
            if isinstance(item, dict) and item.get("item_id")
        }
        source_section = _source_section(response)
        pack = EvidencePack(
            payload=payload,
            source_items=source_items,
            source_section=source_section,
            eligible_for_llm=bool(
                (payload.get("budgets") or {}).get("eligible_for_llm")
            ),
            fallback_reason="",
        )
        output = {
            "schema_version": self.verifier.schema_version,
            "answer_sections": list(
                arguments.get("answer_sections") or []
            ),
            "covered_query_facets": list(
                arguments.get("covered_query_facets") or []
            ),
            "uncovered_query_facets": list(
                arguments.get("uncovered_query_facets") or []
            ),
        }
        sections, errors = self.verifier.verify(output, pack)
        if errors or sections is None:
            return {
                "schema_version": (
                    "debug_agent_system.render_evidence_answer.v1"
                ),
                "status": "rejected",
                "failure_type": "answer_plan_verification_failed",
                "verification_errors": errors,
                "session_id": session_id,
            }
        rendered = dict(response)
        rendered["answer"] = render_answer_sections(sections)
        rendered["answer_sections"] = to_jsonable(sections)
        metadata = dict(rendered.get("metadata") or {})
        metadata["answer_composer"] = {
            "provider": "codex",
            "enabled": True,
            "attempted": True,
            "used": True,
            "fallback_used": False,
            "fallback_reason": "",
            "model": self.system.config.read_llm.model,
            "call_count": 0,
            "verification_errors": [],
            "evidence_mode": "tool_harness_canonical_render",
            "selected_item_count": sum(
                len(section.get("source_item_ids") or [])
                for section in output["answer_sections"]
            ),
        }
        rendered["metadata"] = metadata
        self._remember_response(rendered)
        return rendered

    def _retrieve(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        limit = max(1, min(int(arguments.get("limit") or 10), 20))
        candidates = self.system.read_model.search_variants(query, limit=limit)
        retrieval = self.system.read_model.last_retrieval or {}
        return {
            "schema_version": "debug_agent_system.retrieve_evidence.v2",
            "status": "ok",
            "query": query,
            "candidates": to_jsonable(candidates),
            "supporting_chunks": [
                _bounded_chunk(item)
                for item in retrieval.get("chunks") or []
                if isinstance(item, dict)
            ],
            "paths": list(retrieval.get("paths") or [])[:40],
            "trace": dict(retrieval.get("trace") or {}),
        }

    def _expand_documents(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sag = self.system.read_model.sag
        if sag is None:
            return self._error(
                "expand_document_context", "", "sag_v2_unavailable"
            )
        query = str(arguments.get("query") or "")
        document_ids = _dedupe(arguments.get("document_ids") or [])[:8]
        max_chunks = max(
            1, min(int(arguments.get("max_chunks") or 64), 64)
        )
        chunks = sag.expand_source_document_chunks(query, document_ids)
        selected = chunks[:max_chunks]
        return {
            "schema_version": "debug_agent_system.document_context.v1",
            "status": "ok",
            "query": query,
            "document_ids": document_ids,
            "chunks": [_bounded_chunk(item) for item in selected],
            "returned_chunk_count": len(selected),
            "available_chunk_count": len(chunks),
            "truncated": len(selected) < len(chunks),
        }

    def _inspect_kg_path(self, arguments: dict[str, Any]) -> dict[str, Any]:
        family_id = str(arguments.get("family_id") or "")
        variant_id = str(arguments.get("variant_id") or "")
        if not self.system.read_model.has_object(family_id, "FaultFamily"):
            return self._error("inspect_kg_path", "", "unknown_family")
        if not self.system.read_model.has_object(variant_id, "FaultVariant"):
            return self._error("inspect_kg_path", "", "unknown_variant")
        variant = self.system.read_model.get(variant_id) or {}
        if str(variant.get("family_id") or "") != family_id:
            return self._error(
                "inspect_kg_path", "", "family_variant_mismatch"
            )
        plan = self.system.read_model.compile_plan(family_id, variant_id)
        steps: list[dict[str, Any]] = []
        for step in plan.steps[:64]:
            steps.append({
                **to_jsonable(step),
                "outcomes": self.system.read_model.outcomes_for_step(step),
                "branch_rules": self.system.read_model.branch_rules_for_step(
                    step
                ),
                "evidence": self.system.read_model.evidence(
                    step.evidence_ids
                ),
            })
        return {
            "schema_version": "debug_agent_system.kg_path_inspection.v1",
            "status": "ok",
            "family": self.system.read_model.get(family_id),
            "variant": variant,
            "plan": {
                **to_jsonable(plan),
                "steps": steps,
            },
            "safety_contract": {
                "tool_is_read_only": True,
                "branch_selected": False,
                "action_executed": False,
                "verified_fix_asserted": False,
            },
        }

    def _inspect_assets(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sag = self.system.read_model.sag
        if sag is None:
            return self._error(
                "inspect_source_assets", "", "sag_v2_unavailable"
            )
        query = str(arguments.get("query") or "")
        document_ids = _dedupe(arguments.get("document_ids") or [])[:8]
        max_items = max(1, min(int(arguments.get("max_items") or 50), 100))
        chunks = sag.expand_source_document_chunks(query, document_ids)
        assets: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            for raw_media in chunk.get("media_refs") or []:
                if not isinstance(raw_media, dict):
                    continue
                media = dict(raw_media)
                key = (
                    str(media.get("media_kind") or ""),
                    str(
                        media.get("content_hash")
                        or media.get("asset_path")
                        or media.get("archive_path")
                        or media.get("media_id")
                        or ""
                    ),
                )
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                assets.append({
                    "document_id": str(chunk.get("document_id") or ""),
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "source_label": str(chunk.get("source_label") or ""),
                    "context_label": str(
                        media.get("context_label")
                        or chunk.get("source_heading")
                        or chunk.get("source_label")
                        or ""
                    ),
                    "media": media,
                })
                if len(assets) >= max_items:
                    break
            if len(assets) >= max_items:
                break
        return {
            "schema_version": "debug_agent_system.source_assets.v1",
            "status": "ok",
            "query": query,
            "document_ids": document_ids,
            "assets": assets,
            "asset_count": len(assets),
            "truncated": len(assets) >= max_items,
        }

    @staticmethod
    def _error(
        tool: str,
        call_id: str,
        failure_type: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "debug_agent_system.read_tool_error.v1",
            "status": "failed",
            "failure_type": failure_type,
            "tool": tool,
            "call_id": call_id,
        }


def _bounded_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    output = dict(chunk)
    text = str(output.get("text") or "")
    if len(text) > 4000:
        output["text"] = text[:4000]
        output["text_truncated"] = True
    output["media_refs"] = [
        dict(item)
        for item in output.get("media_refs") or []
        if isinstance(item, dict)
    ][:50]
    return output


def _dedupe(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def _source_section(response: dict[str, Any]) -> AnswerSection | None:
    for raw in response.get("answer_sections") or []:
        if (
            isinstance(raw, dict)
            and str(raw.get("section_type") or "") == "sources"
        ):
            return AnswerSection(
                section_type="sources",
                title=str(raw.get("title") or "资料来源"),
                items=[
                    dict(item)
                    for item in raw.get("items") or []
                    if isinstance(item, dict)
                ],
                evidence_ids=[
                    str(value)
                    for value in raw.get("evidence_ids") or []
                    if str(value)
                ],
                chunk_ids=[
                    str(value)
                    for value in raw.get("chunk_ids") or []
                    if str(value)
                ],
            )
    return None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not str(value or "").strip():
        return {}
    import json

    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {"summary": str(value)}
    return dict(decoded) if isinstance(decoded, dict) else {"value": decoded}


def _search_terms(value: str) -> set[str]:
    import re

    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.:-]{2,}|[\u4e00-\u9fff]{2,}", value)
        if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{8,16}", token)
    }


def _timestamp_between(value: str, start: str, end: str) -> bool:
    if not value:
        return False
    try:
        current = datetime.fromisoformat(value.replace("/", "-").replace(",", ".").replace("Z", "+00:00"))
        lower = datetime.fromisoformat(start.replace("/", "-").replace(",", ".").replace("Z", "+00:00"))
        upper = datetime.fromisoformat(end.replace("/", "-").replace(",", ".").replace("Z", "+00:00"))
    except ValueError:
        return False
    if current.tzinfo is not None and lower.tzinfo is None:
        current = current.replace(tzinfo=None)
    if current.tzinfo is None and lower.tzinfo is not None:
        lower = lower.replace(tzinfo=None)
        upper = upper.replace(tzinfo=None)
    return lower <= current <= upper


__all__ = ["CodexReadSideToolExecutor", "read_side_tool_schemas"]
