"""Review bounded must/cannot contradictions before component compilation."""

from __future__ import annotations

from typing import Any

from ..contracts import validate_component_consistency_decision
from ..model_client import DecisionModelClient


PROMPT_VERSION = "w7-component-consistency-v1"
SYSTEM_PROMPT = """\
你是工业现场纵向 Trace 的局部一致性裁决器。NeighborLink 已分别判断候选边，
但局部结果中出现了“两个 case 被 must_link 路径连通，同时又有直接
cannot_link”的矛盾。你只复审这些 cannot_link 是否可靠。

规则：
1. confirmed_cannot：有明确的不同故障、不同设备/产线、不同 Jira、不同业务目标
   或文本直接说明是另一个问题；本地编译器会保持拆分。
2. weak_cannot：cannot 仅来自时间距离、日报混杂、措辞差异或证据不足，而 must
   路径具有连续诊断、同一 Jira、相同设备故障或复发/验证证据；本地编译器会将
   该 cannot 降为 possible，再重新进行一致性编译。
3. 不得凭“同群/同日”判 weak_cannot，也不得修改输入之外的边。
4. evidence_message_ids 只能引用 allowed message ID；不生成 Trace、phase 或 KG ID。
5. 不确定时选择 confirmed_cannot，保证 fail-closed。

严格调用指定工具，不输出解释文字。"""


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def tool_schema() -> dict[str, Any]:
    decision = _strict_object({
        "left_case_ref": {"type": "string"},
        "right_case_ref": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["confirmed_cannot", "weak_cannot"],
        },
        "evidence_message_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasons": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "review_component_link_conflicts",
            "strict": True,
            "description": (
                "Review cannot-link edges contradicted by a must-link path."
            ),
            "parameters": _strict_object({
                "conflict_decisions": {
                    "type": "array",
                    "items": decision,
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
        },
    }


class ComponentConsistencyAgent:
    agent_id = "W7b.ComponentConsistency"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        conflicts: dict[str, Any],
        case_cards: list[dict[str, Any]],
        allowed_message_ids: set[str],
        repair_issues: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        conflict_values = [
            item for item in conflicts.get("conflicts") or []
            if isinstance(item, dict)
        ]
        required_conflicts = {
            tuple(sorted((
                str(
                    (item.get("cannot_link_edge") or {}).get(
                        "left_case_ref"
                    )
                    or ""
                ),
                str(
                    (item.get("cannot_link_edge") or {}).get(
                        "right_case_ref"
                    )
                    or ""
                ),
            )))
            for item in conflict_values
        }
        conflict_case_refs = {
            str(case_ref)
            for item in conflict_values
            for case_ref in item.get("case_refs") or []
            if str(case_ref)
        }
        bounded_cards = [
            item for item in case_cards
            if isinstance(item, dict)
            and str(
                item.get("case_ref")
                or item.get("case_item_ref")
                or item.get("fragment_ref")
                or ""
            ) in conflict_case_refs
        ]
        response = self.client.call_tool(
            stage="component_consistency",
            system_prompt=(
                SYSTEM_PROMPT
                if not repair_issues
                else (
                    f"{SYSTEM_PROMPT}\n\n"
                    "上一次输出未通过本地契约。请修正以下问题并重新输出全部"
                    " conflict_decisions：\n"
                    + "\n".join(
                        f"- {value}" for value in repair_issues
                    )
                )
            ),
            payload={
                "conflicts": conflicts,
                "case_cards": bounded_cards,
                "allowed_message_ids": sorted(allowed_message_ids),
                "previous_validation_issues": list(repair_issues or []),
            },
            tool=tool_schema(),
            max_tokens=12_288,
        )
        raw = (
            response.get("arguments")
            if isinstance(response.get("arguments"), dict)
            else {}
        )
        normalized, issues = validate_component_consistency_decision(
            raw,
            required_conflicts=required_conflicts,
            allowed_message_ids=allowed_message_ids,
        )
        call = {
            key: value for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
