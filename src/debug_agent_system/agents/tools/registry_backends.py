"""把统一工具注册表接到真实后端（incident runtime / KG_v2 / corpus tools）。

本模块提供 ``build_full_registry(...)``：给定可选的 incident runtime、
KGV2ReadModel、corpus workspace，把注册表中的工具定义接到对应的真实执行器
上，得到可直接被 Agent 循环调用的完整注册表。

所有后端都是只读的，且每个执行结果都尽量带上 evidence_ids/source_ids，
供上层 Evidence Fabric / verifier 使用。任何后端异常都会被统一包装为结构化
失败，绝不向上抛出。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from debug_agent_system.agents.tools.registry import (
    ToolBackend,
    ToolDefinition,
    ToolRegistry,
    ToolParameter,
    _CallableBackend,
    _default_backend,
    build_default_registry,
)


def _evidence_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    """从后端返回里尽力提取证据 ID（来源闭合）。"""

    ids: list[str] = []
    if not isinstance(payload, dict):
        return ids
    for key in ("evidence_ids", "evidence_id", "event_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item))
        elif isinstance(value, str) and value:
            ids.append(value)
    return list(dict.fromkeys(ids))


class IncidentBackend:
    """包装 IncidentEvidenceRuntime 为只读工具后端。"""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._cases: dict[str, dict[str, Any]] = {}

    def _ensure_case(self, query: str, resources: list[dict[str, Any]]) -> str:
        """运行/复用一次 incident 分析，返回 case_id。"""

        digest = hashlib.sha256(
            f"{query}:{[str(r) for r in resources]}".encode()
        ).hexdigest()[:16]
        case_id = f"incident:{digest}"
        if case_id in self._cases:
            return case_id
        result = self.runtime.analyze(query, resources)
        self._cases[case_id] = result
        return case_id

    def parse_evtx_window(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments.get("path") or "")
        if not path:
            return {"error": "missing_evtx_path"}
        # 通过 incident analyze 的通用路径解析 EVTX（与 read_kernel_dump 共用入口）
        case_id = self._ensure_case(
            "解析诊断包事件", [{"resource_id": "evtx", "name": Path(path).name, "path": path}]
        )
        result = self._cases.get(case_id)
        if result is None:
            return {"error": "incident_analysis_failed"}
        events = [
            {
                "event_id": getattr(item, "event_id", ""),
                "timestamp": getattr(item, "timestamp_utc", ""),
                "severity": getattr(item, "severity", ""),
                "component": getattr(item, "component", ""),
                "event_kind": getattr(item, "event_kind", ""),
                "message": str(getattr(item, "message", ""))[:1000],
            }
            for item in getattr(result, "events", [])
        ][: int(arguments.get("max_selected") or 5000)]
        return {
            "case_id": case_id,
            "events": events,
            "returned": len(events),
            "evidence_ids": [
                eid
                for item in getattr(result, "events", [])
                for eid in (getattr(item, "evidence_ids", []) or [])
            ][:200],
        }

    def read_kernel_dump(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments.get("path") or "")
        if not path:
            return {"error": "missing_dump_path"}
        case_id = self._ensure_case(
            "解析内核转储", [{"resource_id": "dmp", "name": Path(path).name, "path": path}]
        )
        result = self._cases.get(case_id)
        if result is None:
            return {"error": "incident_analysis_failed"}
        for manifest in getattr(result, "case", None).artifacts if getattr(result, "case", None) else []:
            kernel = (manifest.metadata or {}).get("kernel_dump") or {}
            if kernel:
                return {"kernel_dump": kernel, "artifact": manifest.name}
            minidump = (manifest.metadata or {}).get("minidump") or {}
            if minidump:
                return {"minidump": minidump, "artifact": manifest.name}
        return {"error": "no_dump_metadata_found"}

    def search_diagnostic_events(self, arguments: dict[str, Any]) -> dict[str, Any]:
        case_id = str(arguments.get("case_id") or "")
        query = str(arguments.get("query") or "")
        limit = min(200, max(1, int(arguments.get("limit") or 50)))
        result = self._cases.get(case_id)
        if result is None:
            return {"error": "unknown_case_id"}
        lowered = query.lower()
        matches = [
            {
                "event_id": getattr(item, "event_id", ""),
                "timestamp": getattr(item, "timestamp_utc", ""),
                "severity": getattr(item, "severity", ""),
                "component": getattr(item, "component", ""),
                "event_kind": getattr(item, "event_kind", ""),
                "message": str(getattr(item, "message", ""))[:1000],
            }
            for item in getattr(result, "events", [])
            if not lowered
            or lowered in str(getattr(item, "message", "")).lower()
            or lowered in str(getattr(item, "event_kind", "")).lower()
            or lowered in str(getattr(item, "component", "")).lower()
        ][:limit]
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": len(getattr(result, "events", [])) > limit,
        }

    def build_incident_timeline(self, arguments: dict[str, Any]) -> dict[str, Any]:
        case_id = str(arguments.get("case_id") or "")
        result = self._cases.get(case_id)
        if result is None:
            return {"error": "unknown_case_id"}
        timeline = list(getattr(result, "timeline", []) or [])
        return {
            "timeline": timeline,
            "returned": len(timeline),
            "correlations": list(getattr(result, "correlations", []) or [])[:50],
        }

    def read_log_window(self, arguments: dict[str, Any]) -> dict[str, Any]:
        case_id = str(arguments.get("case_id") or "")
        artifact_id = str(arguments.get("artifact_id") or "")
        line = int(arguments.get("line") or 1)
        before = int(arguments.get("before") or 10)
        after = int(arguments.get("after") or 20)
        try:
            window = self.runtime.log_window(
                case_id, artifact_id, line, before=before, after=after
            )
            return dict(window)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}:{str(exc)[:200]}"}


class KGV2Backend:
    """包装 KGV2ReadModel 为只读工具后端。"""

    def __init__(self, read_model: Any) -> None:
        self.read_model = read_model

    def kg_search_candidates(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        limit = min(20, max(1, int(arguments.get("limit") or 10)))
        candidates = self.read_model.search_variants(query, limit=limit)
        return {
            "query": query,
            "candidates": [
                {
                    "family_id": getattr(item, "family_id", ""),
                    "variant_id": getattr(item, "variant_id", ""),
                    "family_label": getattr(item, "family_label", ""),
                    "variant_label": getattr(item, "variant_label", ""),
                    "score": getattr(item, "score", 0.0),
                    "matched_fields": list(getattr(item, "matched_fields", []) or []),
                    "evidence_ids": list(getattr(item, "evidence_ids", []) or []),
                }
                for item in candidates
            ],
            "returned": len(candidates),
        }

    def kg_get_object(self, arguments: dict[str, Any]) -> dict[str, Any]:
        object_id = str(arguments.get("object_id") or "")
        value = self.read_model.get(object_id)
        return {"object_id": object_id, "object": value}

    def kg_compile_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        family_id = str(arguments.get("family_id") or "")
        variant_id = str(arguments.get("variant_id") or "")
        try:
            plan = self.read_model.compile_plan(family_id, variant_id)
            return {
                "family_id": family_id,
                "variant_id": variant_id,
                "plan": _to_dict(plan),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}:{str(exc)[:200]}"}


class CorpusBackend:
    """包装 CorpusReadTools 为只读工具后端。"""

    def __init__(self, workspace: str | Path) -> None:
        from debug_agent_system.kg_raw_codex.pipeline import CorpusReadTools

        self.tools = CorpusReadTools(Path(workspace).resolve())

    def _run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result, audit = self.tools.execute(name, arguments)
        result["audit"] = {
            "status": audit.get("status", ""),
            "files_read": sorted(self.tools.files_read),
        }
        return result

    def list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._run("list_files", arguments)

    def search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._run("search_text", arguments)

    def read_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._run("read_text", arguments)


def _to_dict(value: Any) -> dict[str, Any] | list[Any] | Any:
    """把 dataclass 对象递归转 dict（评估/序列化用）。"""

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _to_dict(getattr(value, key))
            for key in getattr(value, "__dataclass_fields__")
        }
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_dict(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_dict(item) for item in value]
    return value


def build_full_registry(
    *,
    incident_runtime: Any | None = None,
    kg_read_model: Any | None = None,
    corpus_workspace: str | Path | None = None,
    system: Any | None = None,
    custom_backend_provider: Callable[[str], Any] | None = None,
) -> ToolRegistry:
    """构建接入真实后端的完整只读工具注册表。

    参数
    ----
    incident_runtime:
        ``IncidentEvidenceRuntime`` 实例（含 analyze/log_window）。
    kg_read_model:
        ``KGV2ReadModel`` 实例。
    corpus_workspace:
        ``CorpusReadTools`` 的语料根（data/raw + data/kg_v2 所在仓库根）。
    system:
        ``DebugAgentSystem`` 实例。提供后，注册表会复用
        ``CodexReadSideToolExecutor`` 的全部 32 个工具（确定性诊断、Incident
        全量面、文档展开等），以及 W9 写侧只读面。
    custom_backend_provider:
        可选的按工具名返回后端的函数（优先于内置/上述后端）。
    """

    codex_executor = None
    if system is not None:
        from debug_agent_system.adapters.codex_read.executor import (
            CodexReadSideToolExecutor,
        )

        codex_executor = CodexReadSideToolExecutor(system)

    def provider(name: str) -> Any:
        if custom_backend_provider is not None:
            backend = custom_backend_provider(name)
            if backend is not None:
                return _as_backend(backend)
        if name in {"parse_evidence", "parse_evidence_context"}:
            return _default_backend(name)  # type: ignore[arg-type]
        # 确定性诊断 / Incident 全量 / 文档展开 → Codex executor
        if codex_executor is not None and name in _CODEX_EXECUTOR_TOOLS:
            return _CallableBackend(
                lambda arguments, _name=name: codex_executor.execute(
                    _name, arguments
                )
            )
        # W9 写侧只读面
        if system is not None and name.startswith("w9_"):
            from debug_agent_system.agents.write.w9_raw_doc_ingest import (
                RawDocIngestAgent,
            )

            w9 = RawDocIngestAgent()
            w9_mapping = {
                "w9_inspect_document": lambda arguments: w9.inspect_document(
                    str(arguments.get("path") or "")
                ),
                "w9_build_structured_sections": lambda arguments: w9.build_structured_sections(
                    str(arguments.get("path") or "")
                ),
                "w9_build_section_cases": lambda arguments: w9.build_section_cases(
                    str(arguments.get("path") or "")
                ),
                "w9_build_root_checklist": lambda arguments: w9.build_root_checklist(
                    str(arguments.get("root") or ""),
                    include_sop=bool(arguments.get("include_sop", False)),
                ),
            }
            if name in w9_mapping:
                return _CallableBackend(w9_mapping[name])
        if incident_runtime is not None:
            incident = IncidentBackend(incident_runtime)
            mapping = {
                "parse_evtx_window": incident.parse_evtx_window,
                "read_kernel_dump": incident.read_kernel_dump,
                "search_diagnostic_events": incident.search_diagnostic_events,
                "build_incident_timeline": incident.build_incident_timeline,
                "read_log_window": incident.read_log_window,
            }
            if name in mapping:
                return _CallableBackend(mapping[name])
        if kg_read_model is not None:
            kg = KGV2Backend(kg_read_model)
            mapping = {
                "kg_search_candidates": kg.kg_search_candidates,
                "kg_get_object": kg.kg_get_object,
                "kg_compile_plan": kg.kg_compile_plan,
            }
            if name in mapping:
                return _CallableBackend(mapping[name])
        if corpus_workspace is not None:
            corpus = CorpusBackend(corpus_workspace)
            mapping = {
                "list_files": corpus.list_files,
                "search_text": corpus.search_text,
                "read_text": corpus.read_text,
            }
            if name in mapping:
                return _CallableBackend(mapping[name])
        return None

    return build_default_registry(backend_provider=provider)


# 这些工具名由 ``CodexReadSideToolExecutor`` 提供真实后端。
_CODEX_EXECUTOR_TOOLS = frozenset({
    "diagnose_start",
    "diagnose_step",
    "retrieve_evidence",
    "expand_document_context",
    "inspect_kg_path",
    "inspect_source_assets",
    "render_evidence_answer",
    "analyze_incident",
    "index_log_package",
    "parse_incident_scope",
    "get_jira_snapshot",
    "get_incident_scope",
    "get_incident_evidence_pack",
    "list_artifacts",
    "inspect_archive_manifest",
    "search_diagnostic_events_by_time",
    "extract_log_time_windows",
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
})


def _as_backend(value: Any) -> Any:
    if isinstance(value, ToolBackend) or hasattr(value, "execute"):
        return value
    if callable(value):
        return _CallableBackend(value)
    return value


__all__ = [
    "IncidentBackend",
    "KGV2Backend",
    "CorpusBackend",
    "build_full_registry",
]
