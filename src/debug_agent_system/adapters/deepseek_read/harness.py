"""Optional model-controlled loop over deterministic read-side Tools."""

from __future__ import annotations

import json
from typing import Any, Protocol

from debug_agent_system.agents.tools.executor import tool_call_fingerprint
from debug_agent_system.runtime.system import DebugAgentSystem

from .client import DeepSeekReadClient, DeepSeekReadClientError
from .executor import ReadSideToolExecutor, read_side_tool_schemas


class ReadModelClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class DeepSeekReadToolHarness:
    """Let DeepSeek select Tools while deterministic runtime owns diagnosis."""

    system_prompt = (
        "你是 debug_agent_system 的读侧工具控制器。优先调用 diagnose_start。"
        "只有已有 required_data 且用户提供了资源时才调用 parse_evidence。"
        "不得虚构 Tool 输出；不得选择 BranchRule、宣称 verified_fix、确认高风险动作"
        "或要求执行未暴露的工具。最终诊断文本必须来自 diagnose_start/diagnose_step。"
    )

    def __init__(
        self,
        system: DebugAgentSystem | None = None,
        *,
        client: ReadModelClient | None = None,
    ) -> None:
        self.system = system or DebugAgentSystem.from_config()
        self.executor = ReadSideToolExecutor(self.system)
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
            result = self.system.start(
                {
                    "query": query,
                    "interactive": interactive,
                    "routing_context": routing_context or {},
                    "evidence_resources": resources,
                }
            )
            return _with_harness_metadata(
                result,
                {
                    "enabled": False,
                    "fallback_used": True,
                    "reason": "read_llm_disabled",
                    "tool_calls": [],
                },
            )
        client = self.client or DeepSeekReadClient(
            model=self.system.config.read_llm.model,
            base_url=self.system.config.read_llm.base_url,
            timeout_seconds=self.system.config.read_llm.timeout_seconds,
        )
        tools = read_side_tool_schemas()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "interactive": interactive,
                        "routing_context": routing_context or {},
                        "evidence_resources": resources,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        fingerprints: set[str] = set()
        trace: list[dict[str, Any]] = []
        latest_diagnosis: dict[str, Any] | None = None
        try:
            for round_index in range(
                max(1, self.system.config.read_llm.max_tool_rounds)
            ):
                assistant = client.complete(messages=messages, tools=tools)
                tool_calls = assistant.get("tool_calls") or []
                messages.append(assistant)
                if not isinstance(tool_calls, list) or not tool_calls:
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
                            "schema_version": "debug_agent_system.read_tool_error.v1",
                            "status": "skipped",
                            "failure_type": "duplicate_tool_call",
                            "tool": name,
                            "call_id": call_id,
                            "call_fingerprint": fingerprint,
                        }
                    else:
                        fingerprints.add(fingerprint)
                        tool_result = self.executor.execute(
                            name,
                            arguments,
                            call_id=call_id,
                        )
                    trace.append(
                        {
                            "round": round_index + 1,
                            "tool": name,
                            "call_id": call_id,
                            "call_fingerprint": fingerprint,
                            "status": str(tool_result.get("status") or ""),
                        }
                    )
                    if name in {"diagnose_start", "diagnose_step"}:
                        latest_diagnosis = tool_result
                    elif (
                        name == "parse_evidence"
                        and latest_diagnosis is not None
                        and latest_diagnosis.get("status") == "ask_info"
                        and tool_result.get("status") in {"parsed", "metadata_only"}
                    ):
                        # Feed the original resource, not model-authored facts,
                        # into the source-bound runtime resolver.
                        latest_diagnosis = self.system.step(
                            str(latest_diagnosis.get("session_id") or ""),
                            "",
                            evidence_resources=[
                                dict(arguments.get("resource") or {})
                            ],
                        )
                        trace.append(
                            {
                                "round": round_index + 1,
                                "tool": "evidence_gap_resume",
                                "call_id": call_id,
                                "call_fingerprint": fingerprint,
                                "status": str(
                                    latest_diagnosis.get("status") or ""
                                ),
                            }
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                                default=str,
                            ),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - optional controller must fail open.
            fallback = self.system.start(
                {
                    "query": query,
                    "interactive": interactive,
                    "routing_context": routing_context or {},
                    "evidence_resources": resources,
                }
            )
            return _with_harness_metadata(
                fallback,
                {
                    "enabled": True,
                    "fallback_used": True,
                    "reason": str(exc),
                    "tool_calls": trace,
                },
            )
        result = latest_diagnosis or self.system.start(
            {
                "query": query,
                "interactive": interactive,
                "routing_context": routing_context or {},
                "evidence_resources": resources,
            }
        )
        return _with_harness_metadata(
            result,
            {
                "enabled": True,
                "fallback_used": latest_diagnosis is None,
                "reason": (
                    "model_returned_no_diagnosis_tool"
                    if latest_diagnosis is None
                    else "tool_controlled_deterministic_answer"
                ),
                "tool_calls": trace,
            },
        )


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
    output_metadata["deepseek_tool_harness"] = metadata
    output["metadata"] = output_metadata
    return output


__all__ = ["DeepSeekReadToolHarness", "ReadModelClient"]
