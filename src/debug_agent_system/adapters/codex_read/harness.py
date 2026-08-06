"""Codex-controlled investigation loop over deterministic read-side tools."""

from __future__ import annotations

import json
from typing import Any, Protocol

from debug_agent_system.agents.tools.executor import tool_call_fingerprint
from debug_agent_system.runtime.system import DebugAgentSystem

from .client import CodexReadClient
from .executor import CodexReadSideToolExecutor, read_side_tool_schemas


class ReadModelClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class CodexReadToolHarness:
    """Let Codex investigate while deterministic runtime owns every decision."""

    system_prompt = """
你是 debug_agent_system 的 Codex 读侧调查控制器。

工作顺序：
1. 用户提供 Jira、日志或诊断包时，先调用 parse_incident_scope 核对参考时间，再调用
   analyze_incident 建立不可变案件证据。运行时会在压缩包 intake 阶段按参考时间流式提取
   日志窗口；按需使用 get_incident_scope、extract_log_time_windows、
   search_diagnostic_events_by_time、read_log_window、build_incident_timeline、
   inspect_stacktrace、inspect_environment、query_kg_hypotheses 和 propose_next_tests 迭代调查。
2. 自然语言知识问答或未提供诊断包时，优先调用 diagnose_start 获取 KG_v2/SAG 的正式运行时结论。
3. 需要核对候选和原文时调用 retrieve_evidence；需要完整文档上下文时再调用
   expand_document_context；Variant 已明确时可调用 inspect_kg_path；需要图片或附件时调用
   inspect_source_assets。
4. parse_evidence 仅用于单文件浅层检查；完整诊断包优先使用 analyze_incident，不得将其
   结构化结果压平成 Query 文本。
   回到确定性运行时。
   对 EVTX/DMP 证据必须分别调用 inspect_evtx/inspect_dump 核对：EVTX 的 provider、
   EventID、UTC/本地时间对齐和事件数据；DMP 的进程、异常码、故障模块与加载模块版本。
   无符号 DMP 的地址归属不是调用栈，也不能单独证明驱动或硬件根因。
5. 普通知识回答结束后必须调用 render_evidence_answer；事故诊断结束后必须调用
   render_incident_report。只能使用工具返回的证据和假设组织说明。
   返回的 Evidence Pack.source_items 中的 ID 组织答案，并逐字回传 Pack 声明的
   supported_facets/unsupported_facets。若本地校验拒绝，不得自行输出自由文本答案。
6. “重复发生”“受控复现”“修复验证”必须区分。需要复现建议时调用 plan_reproduction；
   有两个不可变运行结果时调用 compare_reproduction_runs。相同签名只能证明复发，单次
   未出现不能证明 verified_fix。Tool 不自动控制产线设备，不执行诊断包附件脚本。

边界：
- 不得虚构工具结果或引用。
- 不得自行选择 BranchRule、执行 Action、确认高风险动作或宣称 verified_fix。
- 工具调查用于补充证据和审计；最终诊断状态必须来自 diagnose_start/diagnose_step，
  最终答案只能是原确定性答案或 render_evidence_answer 的本地规范化结果。
""".strip()

    def __init__(
        self,
        system: DebugAgentSystem | None = None,
        *,
        client: ReadModelClient | None = None,
    ) -> None:
        self.system = system or DebugAgentSystem.from_config()
        self.executor = CodexReadSideToolExecutor(self.system)
        self.client = client

    def run(
        self,
        query: str,
        *,
        evidence_resources: list[dict[str, Any]] | None = None,
        interactive: bool = True,
        routing_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resources = list(evidence_resources or [])
        if not self.system.config.read_llm.enabled:
            result = self._start(
                query, resources, interactive, routing_context
            )
            return _with_harness_metadata(result, {
                "provider": "codex",
                "enabled": False,
                "fallback_used": True,
                "reason": "read_llm_disabled",
                "tool_calls": [],
            })
        client = self.client or CodexReadClient(
            model=self.system.config.read_llm.model,
            base_url=self.system.config.read_llm.base_url,
            timeout_seconds=self.system.config.read_llm.timeout_seconds,
            env_file=self.system.config.root / ".env.local",
        )
        tools = read_side_tool_schemas()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "query": query,
                    "interactive": interactive,
                    "routing_context": routing_context or {},
                    "evidence_resources": resources,
                }, ensure_ascii=False),
            },
        ]
        fingerprints: set[str] = set()
        trace: list[dict[str, Any]] = []
        latest_diagnosis: dict[str, Any] | None = None
        canonical_render_used = False
        incident_analysis_started = False
        max_rounds = max(1, self.system.config.read_llm.max_tool_rounds)
        try:
            for round_index in range(max_rounds):
                assistant = client.complete(messages=messages, tools=tools)
                tool_calls = assistant.get("tool_calls") or []
                messages.append(assistant)
                if not isinstance(tool_calls, list) or not tool_calls:
                    if (
                        latest_diagnosis is not None
                        and not canonical_render_used
                        and round_index + 1 < max_rounds
                    ):
                        messages.append({
                            "role": "user",
                            "content": (
                                "你尚未通过 render_incident_report 提交最终案件报告。"
                                "不要输出自由文本；现在必须使用当前 case_id 调用该 Tool。"
                                if incident_analysis_started
                                else (
                                    "你尚未通过 render_evidence_answer 提交最终"
                                    "证据编排。不要输出自由文本；现在必须使用当前"
                                    " diagnosis response 的 Evidence Pack 调用该 Tool。"
                                )
                            ),
                        })
                        continue
                    break
                for call in tool_calls:
                    function = (
                        call.get("function")
                        if isinstance(call, dict)
                        and isinstance(call.get("function"), dict)
                        else {}
                    )
                    name = str(function.get("name") or "")
                    call_id = str(call.get("id") or "")
                    arguments = _arguments(function.get("arguments"))
                    fingerprint = tool_call_fingerprint(name, arguments)
                    if fingerprint in fingerprints:
                        tool_result = {
                            "schema_version": (
                                "debug_agent_system.read_tool_error.v1"
                            ),
                            "status": "skipped",
                            "failure_type": "duplicate_tool_call",
                            "tool": name,
                            "call_id": call_id,
                            "call_fingerprint": fingerprint,
                        }
                    else:
                        fingerprints.add(fingerprint)
                        tool_result = self.executor.execute(
                            name, arguments, call_id=call_id
                        )
                    trace.append({
                        "round": round_index + 1,
                        "tool": name,
                        "call_id": call_id,
                        "call_fingerprint": fingerprint,
                        "status": str(tool_result.get("status") or ""),
                    })
                    if (
                        name == "render_evidence_answer"
                        and tool_result.get("status") != "rejected"
                    ):
                        latest_diagnosis = tool_result
                        canonical_render_used = True
                    elif (
                        name == "render_incident_report"
                        and tool_result.get("status")
                        in {"analyzed", "verification_failed"}
                    ):
                        latest_diagnosis = tool_result
                        canonical_render_used = True
                    elif (
                        name in {"analyze_incident", "index_log_package"}
                        and tool_result.get("status")
                        in {"analyzed", "verification_failed"}
                    ):
                        latest_diagnosis = tool_result
                        incident_analysis_started = True
                    elif name in {
                        "diagnose_start",
                        "diagnose_step",
                    } and tool_result.get("status") != "rejected":
                        latest_diagnosis = tool_result
                    elif (
                        name == "parse_evidence"
                        and latest_diagnosis is not None
                        and latest_diagnosis.get("status") == "ask_info"
                        and tool_result.get("status")
                        in {"parsed", "metadata_only"}
                    ):
                        latest_diagnosis = self.system.step(
                            str(latest_diagnosis.get("session_id") or ""),
                            "",
                            evidence_resources=[
                                dict(arguments.get("resource") or {})
                            ],
                        )
                        trace.append({
                            "round": round_index + 1,
                            "tool": "evidence_gap_resume",
                            "call_id": call_id,
                            "call_fingerprint": fingerprint,
                            "status": str(
                                latest_diagnosis.get("status") or ""
                            ),
                        })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(
                            tool_result, ensure_ascii=False, default=str
                        ),
                    })
        except Exception as exc:  # fail-open is part of the public contract
            fallback = self._start(
                query, resources, interactive, routing_context
            )
            return _with_harness_metadata(fallback, {
                "provider": "codex",
                "enabled": True,
                "fallback_used": True,
                "reason": f"{type(exc).__name__}:{str(exc)[:200]}",
                "tool_calls": trace,
            })
        result = latest_diagnosis or self._start(
            query, resources, interactive, routing_context
        )
        return _with_harness_metadata(result, {
            "provider": "codex",
            "enabled": True,
            "fallback_used": latest_diagnosis is None,
            "reason": (
                "model_returned_no_diagnosis_tool"
                if latest_diagnosis is None
                else (
                    "tool_controlled_canonical_answer"
                    if canonical_render_used
                    else "tool_controlled_deterministic_answer"
                )
            ),
            "canonical_render_used": canonical_render_used,
            "tool_calls": trace,
            "available_tools": sorted(self.executor.allowed_tools),
        })

    def _start(
        self,
        query: str,
        resources: list[dict[str, Any]],
        interactive: bool,
        routing_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self.system.start({
            "query": query,
            "interactive": interactive,
            "routing_context": routing_context or {},
            "evidence_resources": resources,
        })


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("tool_arguments_not_object")
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("tool_arguments_not_object")
    return data


def _with_harness_metadata(
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output = dict(result)
    output_metadata = dict(output.get("metadata") or {})
    output_metadata["codex_tool_harness"] = metadata
    output["metadata"] = output_metadata
    return output


__all__ = ["CodexReadToolHarness", "ReadModelClient"]
