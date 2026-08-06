"""Assign one bounded component to trace groups and ordered phases."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    TRACE_EVENT_TYPES,
    TRACE_RELATION_TYPES,
    validate_trace_phase_patch,
)
from ..model_client import DecisionModelClient


PROMPT_VERSION = "w7-trace-phase-v1"
SYSTEM_PROMPT = """\
你是工业现场纵向 Trace 的阶段判断器。输入是本地程序构造的一个小型、已裁决
case component。你只输出 Trace 分组和 phase patch。

规则：
1. 每个 case 必须且只能进入一个 local trace，并且恰好有一个 set_phase。
2. 同一 component 仍可能包含多个独立 Trace；不要为了减少数量强行合并。
3. 保留 report、diagnosis、action、short_term_recovery、recurrence、
   resolution、validation 的时间与因果边界。
4. 临时恢复后复发必须保留两个 phase；最终验证不能覆盖早期失败。
5. 每条 Trace 只有一个 root；phase_index 从 1 递增，after_case_ref 只能指向
   同 Trace 中更早 case。
6. 只能引用 allowed case/message ID；不生成最终 trace_id、Action、Outcome 或 KG ID。

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
        "op": {
            "type": "string",
            "enum": ["create_trace_group", "set_phase"],
        },
        "local_trace_ref": {"type": "string"},
        "case_refs": {"type": "array", "items": {"type": "string"}},
        "case_ref": {"type": "string"},
        "event_type": {
            "type": "string",
            "enum": ["", *TRACE_EVENT_TYPES],
        },
        "relation_type": {
            "type": "string",
            "enum": ["", *TRACE_RELATION_TYPES],
        },
        "phase_index": {"type": "integer", "minimum": 0},
        "after_case_ref": {"type": "string"},
        "evidence_message_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
    })
    return {
        "type": "function",
        "function": {
            "name": "assign_trace_groups_and_phases",
            "strict": True,
            "description": (
                "Partition one bounded component into local traces and assign "
                "each case one ordered phase."
            ),
            "parameters": _strict_object({
                "operations": {"type": "array", "items": operation},
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
        },
    }


class TracePhaseAgent:
    agent_id = "W7b.TracePhase"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        component: dict[str, Any],
        case_cards: list[dict[str, Any]],
        link_decisions: list[dict[str, Any]],
        allowed_message_ids: set[str],
        repair_issues: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        case_refs = {
            str(value) for value in component.get("case_refs") or []
            if str(value)
        }
        cards = [
            item for item in case_cards
            if isinstance(item, dict)
            and str(
                item.get("case_ref")
                or item.get("case_item_ref")
                or item.get("fragment_ref")
                or ""
            ) in case_refs
        ]
        allowed_by_case: dict[str, set[str]] = {}
        for card in cards:
            case_ref = str(
                card.get("case_ref")
                or card.get("case_item_ref")
                or card.get("fragment_ref")
                or ""
            )
            allowed_by_case[case_ref] = {
                str(value)
                for value in [
                    *(card.get("source_message_ids") or []),
                    *(card.get("evidence_message_ids") or []),
                ]
                if str(value)
            }
        response = self.client.call_tool(
            stage="trace_phase",
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
                "component": component,
                "case_cards": cards,
                "link_decisions": link_decisions,
                "allowed_case_refs": sorted(case_refs),
                "allowed_message_ids": sorted(allowed_message_ids),
                "previous_validation_issues": list(
                    repair_issues or []
                ),
            },
            tool=tool_schema(),
            max_tokens=16_384,
        )
        raw = (
            response.get("arguments")
            if isinstance(response.get("arguments"), dict)
            else {}
        )
        normalized, issues = validate_trace_phase_patch(
            raw,
            component_case_refs=case_refs,
            allowed_message_ids=allowed_message_ids,
            allowed_message_ids_by_case=allowed_by_case,
        )
        call = {
            key: value for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
