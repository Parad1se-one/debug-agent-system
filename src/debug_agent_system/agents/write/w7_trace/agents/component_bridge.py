"""Re-review safe possible-link bridges between compiled components."""

from __future__ import annotations

from typing import Any

from ..contracts import validate_component_bridge_decision
from ..model_client import DecisionModelClient


PROMPT_VERSION = "w7-component-bridge-v2"
SYSTEM_PROMPT = """\
你是工业现场纵向 Trace 的组件断桥复审器。输入只包含 NeighborLink 判为
possible_link、且本地程序确认合并不会违反 cannot_link 或组件大小上限的跨组件边。

规则：
1. promote_must 仅用于存在明确纵向连续性：同一 Jira/设备故障、首报→诊断→动作
   →复发/恢复/验证，或附件与后续解释能够闭合。
2. 同群、同日、同一日报、同一人员、相似通用措辞都不能单独作为 promote 依据。
3. confirm_cannot 用于确认不同故障、设备/产线、Jira 或业务目标。
4. 证据不足时 keep_possible，交给人工审核，不能自动物化。
5. 只能处理 input bridge candidates；evidence_message_ids 只能引用 allowed ID；
   不生成 Trace、phase、Action、Outcome 或 KG ID。
6. 相同 site_scope、相同具体故障且时间连续的恢复/验证可以补强 promote_must；
   不同 site_scope 时默认不是同一现场 Trace。site_scope 的来源优先于同一日报中
   其他 case 的地名。channel_site_scopes 只是群名弱提示，不能单独支持
   promote_must，且正文中的明确站点优先。

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
            "enum": [
                "promote_must",
                "keep_possible",
                "confirm_cannot",
            ],
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
            "name": "review_component_possible_bridges",
            "strict": True,
            "description": (
                "Promote, retain, or reject safe possible-link bridges."
            ),
            "parameters": _strict_object({
                "bridge_decisions": {
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


class ComponentBridgeAgent:
    agent_id = "W7b.ComponentBridge"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        bridge_candidates: dict[str, Any],
        case_cards: list[dict[str, Any]],
        allowed_message_ids: set[str],
        repair_issues: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        candidates = [
            item for item in bridge_candidates.get("candidates") or []
            if isinstance(item, dict)
        ]
        required = {
            tuple(sorted((
                str(item.get("left_case_ref") or ""),
                str(item.get("right_case_ref") or ""),
            )))
            for item in candidates
        }
        case_refs = {
            str(case_ref)
            for item in candidates
            for case_ref in [
                *(item.get("left_component_case_refs") or []),
                *(item.get("right_component_case_refs") or []),
            ]
            if str(case_ref)
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
        response = self.client.call_tool(
            stage="component_bridge",
            system_prompt=(
                SYSTEM_PROMPT
                if not repair_issues
                else (
                    f"{SYSTEM_PROMPT}\n\n"
                    "上一次输出未通过本地契约。请修正以下问题并重新输出全部"
                    " bridge_decisions：\n"
                    + "\n".join(
                        f"- {value}" for value in repair_issues
                    )
                )
            ),
            payload={
                "bridge_candidates": bridge_candidates,
                "case_cards": cards,
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
        normalized, issues = validate_component_bridge_decision(
            raw,
            required_bridges=required,
            allowed_message_ids=allowed_message_ids,
        )
        call = {
            key: value for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
