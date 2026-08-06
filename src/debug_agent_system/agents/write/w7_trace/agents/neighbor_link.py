"""Adjudicate bounded sparse-graph candidate edges."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    EDGE_DECISIONS,
    TRACE_RELATION_TYPES,
    validate_trace_link_decision,
)
from ..model_client import DecisionModelClient


PROMPT_VERSION = "w7-neighbor-link-v4"
SYSTEM_PROMPT = """\
你是工业现场纵向 Trace 的候选边裁决器。输入图已经由本地程序做成分块后的
稀疏候选图；其中既有强 identity 候选，也有为了提高召回率保留的弱候选。
你只判断给出的 candidate case pair 是否属于同一纵向业务 Trace。

规则：
1. must_link 需要具体 identity 或连续诊断链证据；同群、同日、同一个日报不是充分条件。
2. 同时处理的不同故障、不同设备/产线或明确独立问题必须 cannot_link。
3. possible_link 用于证据不足但值得人工复核的边，不会自动物化。
4. 同一 Trace 可以跨日期保留 report、action、recurrence、validation phase，
   但不能把不同 case 压成一个扁平 episode。
5. 只能裁决 input candidate_edges，不生成新边、Trace 或 KG ID。
6. evidence_message_ids 只能引用 allowed message ID。
7. shared_message/shared_parent_episode 只表示共同来源。一个日报或一条消息可以同时
   包含多个独立故障；若理由中判断为“不同问题/不同故障”，决策必须是
   cannot_link，不能因为共同工作上下文输出 must_link。
8. site_scope/device_scope/Jira 是来源侧作用域。相同 site + 相同具体故障 +
   连续日期的“排查→更换/调整→恢复→未复发”可构成 must_link；不同 site/device
   时除非有明确迁移/置换证据，否则应 cannot_link。不得从同一消息内其他 case
   的文本覆盖当前 case 自己的 site_scope。channel_site_scopes 只是群名提供的
   弱提示，不是案例事实，不能单独作为 must_link 依据。

严格调用指定工具，不输出解释文字。"""


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def tool_schema() -> dict[str, Any]:
    edge = _strict_object({
        "left_case_ref": {"type": "string"},
        "right_case_ref": {"type": "string"},
        "decision": {"type": "string", "enum": list(EDGE_DECISIONS)},
        "relation_hint": {
            "type": "string",
            "enum": ["", *TRACE_RELATION_TYPES],
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
            "name": "adjudicate_trace_candidate_edges",
            "strict": True,
            "description": (
                "Classify bounded case-pair candidates as must, possible, "
                "or cannot link."
            ),
            "parameters": _strict_object({
                "edge_decisions": {"type": "array", "items": edge},
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
        },
    }


class NeighborLinkAgent:
    agent_id = "W7b.NeighborLink"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        graph: dict[str, Any],
        case_cards: list[dict[str, Any]],
        allowed_message_ids: set[str],
        repair_issues: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        edges = [
            item for item in graph.get("edges") or []
            if isinstance(item, dict)
        ]
        required_edges = {
            tuple(sorted((
                str(item.get("left_case_ref") or ""),
                str(item.get("right_case_ref") or ""),
            )))
            for item in edges
            if bool(item.get("requires_adjudication"))
        }
        allowed_edges = {
            tuple(sorted((
                str(item.get("left_case_ref") or ""),
                str(item.get("right_case_ref") or ""),
            )))
            for item in edges
        }
        response = self.client.call_tool(
            stage="neighbor_link",
            system_prompt=(
                SYSTEM_PROMPT
                if not repair_issues
                else (
                    f"{SYSTEM_PROMPT}\n\n"
                    "上一次输出未通过本地契约。必须修正以下问题并重新输出"
                    "完整 edge_decisions：\n"
                    + "\n".join(
                        f"- {value}" for value in repair_issues
                    )
                )
            ),
            payload={
                "case_cards": case_cards,
                "candidate_edges": [
                    item for item in edges
                    if bool(item.get("requires_adjudication"))
                ],
                "allowed_message_ids": sorted(allowed_message_ids),
                "previous_validation_issues": list(
                    repair_issues or []
                ),
            },
            tool=tool_schema(),
            max_tokens=12_288,
        )
        raw = (
            response.get("arguments")
            if isinstance(response.get("arguments"), dict)
            else {}
        )
        normalized, issues = validate_trace_link_decision(
            raw,
            required_edges=required_edges,
            allowed_edges=allowed_edges,
            allowed_message_ids=allowed_message_ids,
        )
        call = {
            key: value for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
