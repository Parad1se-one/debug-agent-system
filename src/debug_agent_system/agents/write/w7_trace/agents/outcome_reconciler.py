"""DeepSeek-backed trace outcome/status reconciliation."""

from __future__ import annotations

from typing import Any

from ..contracts import RESOLUTION_STATUSES, validate_outcome_patch
from ..model_client import DecisionModelClient


PROMPT_VERSION = "w7-outcome-reconciler-v2"
SYSTEM_PROMPT = """\
你是工业现场 Trace 的结果状态校对器。输入中的 Trace 已经完成边界和阶段判断；你只能根据
允许的消息证据提出状态修订 patch。

规则：
1. 正常生产、客户验证正常或明确未再复发可作为验证候选。
2. 重启后暂时正常、目前正常、待观察只能到 provisionally_resolved。
3. 后续再次复发必须覆盖更早的暂时恢复，改为 recurrence/ineffective。
4. Jira Resolved 本身不能证明现场 verified。
5. verified 必须引用明确证据消息。
6. 不生成 Trace、Action、Outcome 或 KG ID，只输出状态 patch。

严格调用指定工具，不输出解释文字。"""


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def tool_schema() -> dict[str, Any]:
    operation = _strict_object({
        "op": {"type": "string", "enum": ["revise_trace_status"]},
        "local_trace_ref": {"type": "string"},
        "from": {"type": "string"},
        "to": {"type": "string", "enum": list(RESOLUTION_STATUSES)},
        "evidence_message_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    })
    return {
        "type": "function",
        "function": {
            "name": "reconcile_trace_outcomes",
            "strict": True,
            "description": "Revise trace status from explicit source evidence.",
            "parameters": _strict_object({
                "operations": {"type": "array", "items": operation},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            }),
        },
    }


class OutcomeReconcilerAgent:
    agent_id = "W7b.OutcomeReconciler"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        traces: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        allowed_message_ids: set[str],
        repair_issues: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        trace_refs = {
            str(item.get("local_trace_ref") or item.get("trace_ref") or "")
            for item in traces
            if isinstance(item, dict)
            and str(item.get("local_trace_ref") or item.get("trace_ref") or "")
        }
        response = self.client.call_tool(
            stage="outcome_reconciler",
            system_prompt=(
                SYSTEM_PROMPT
                if not repair_issues
                else (
                    f"{SYSTEM_PROMPT}\n\n"
                    "上一次输出未通过本地契约。必须修正以下问题后重新生成完整"
                    " operations；不要复述问题：\n"
                    + "\n".join(
                        f"- {value}" for value in repair_issues
                    )
                )
            ),
            payload={
                "traces": traces,
                "evidence_rows": evidence_rows,
                "allowed_trace_refs": sorted(trace_refs),
                "allowed_message_ids": sorted(allowed_message_ids),
                "previous_validation_issues": list(
                    repair_issues or []
                ),
            },
            tool=tool_schema(),
            max_tokens=8_192,
        )
        raw = (
            response.get("arguments")
            if isinstance(response.get("arguments"), dict)
            else {}
        )
        normalized, issues = validate_outcome_patch(
            raw,
            allowed_trace_refs=trace_refs,
            allowed_message_ids=allowed_message_ids,
        )
        call = {
            key: value
            for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
