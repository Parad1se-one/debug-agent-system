"""Model boundary for W7 decision agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from debug_agent_system.agents.write.w2_extract.deepseek_client import (
    DeepSeekToolCallError,
    call_json_object,
    call_strict_tool,
)


class DecisionModelClient(Protocol):
    def call_tool(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        tool: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]: ...


@dataclass
class DeepSeekDecisionModelClient:
    api_key: str
    timeout_seconds: float = 180.0
    max_attempts: int = 2

    def call_tool(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        tool: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        try:
            return call_strict_tool(
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_payload=payload,
                tool=tool,
                max_tokens=max_tokens,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.max_attempts,
                user_id=f"w7_multi_agent_{stage}",
            )
        except DeepSeekToolCallError as tool_error:
            repairable = str(tool_error).startswith((
                "deepseek_tool_arguments_json_decode:",
                "deepseek_tool_arguments_empty",
                "deepseek_tool_arguments_not_object",
                "deepseek_missing_tool_call:",
                "deepseek_wrong_tool_call",
            ))
            if not repairable:
                # Authentication, authorization, quota, transport and service
                # failures are not semantic repair cases.  Keep transport
                # retry accounting inside the transport client and fail
                # closed here after it is exhausted.
                raise
            # The DeepSeek beta Tool Calls endpoint occasionally returns a
            # syntactically incomplete ``function.arguments`` value.  Never
            # repair that text locally.  Retry through the independent JSON
            # Output endpoint, then let the stage's local contract validator
            # accept or reject the fresh object.
            function = (
                tool.get("function")
                if isinstance(tool.get("function"), dict)
                else {}
            )
            parameters = (
                function.get("parameters")
                if isinstance(function.get("parameters"), dict)
                else {}
            )
            fallback_prompt = (
                f"{system_prompt}\n\n"
                "工具调用传输失败。现在通过 JSON Output 重新作答。"
                "只输出原工具 arguments 对象本身，不要输出工具名、Markdown 或解释。"
                "必须逐项满足以下 JSON Schema；不确定时使用空数组或 uncertainties，"
                "不得编造 ID：\n"
                f"{json.dumps(parameters, ensure_ascii=False, sort_keys=True)}"
            )
            response = call_json_object(
                api_key=self.api_key,
                system_prompt=fallback_prompt,
                user_payload=payload,
                max_tokens=max_tokens,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.max_attempts,
                user_id=f"w7_multi_agent_{stage}_json_repair",
            )
            response["json_output_fallback"] = True
            response["semantic_repair_count"] = 1
            response["strict_tool_error"] = str(tool_error)
            return response
