"""DeepSeek-backed, evidence-bounded case boundary decisions."""

from __future__ import annotations

from typing import Any

from ..contracts import CASE_KINDS, validate_case_boundary_decision
from ..model_client import DecisionModelClient


PROMPT_VERSION = "w7-case-boundary-v1"
SYSTEM_PROMPT = """\
你是工业现场群聊的原子 Case 边界判断器。你只拆分消息中的业务问题，不抽取完整动作、
结果或知识图谱对象。

规则：
1. 一条日报可拆成多个独立 case fragment。
2. 故障、产品需求、正向验证、培训/协调和噪声必须区分。
3. 每个 fragment 必须引用 allowed message ID；仅对非空文本提供字符 offset。附件或
   空文本消息保留在 source_message_ids，禁止生成 0:0 span，后续由 EvidenceAnchor 绑定。
4. 同一消息可支持多个 fragment，但不能把不同设备或不同故障合成一个 fragment。
5. 每条输入消息必须进入 fragment 或 non_case_message_ids。
6. 不生成 Trace、Action、Outcome、KG ID。

严格调用指定工具，不输出解释文字。"""


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def tool_schema() -> dict[str, Any]:
    span = _strict_object({
        "message_id": {"type": "string"},
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 1},
    })
    fragment = _strict_object({
        "fragment_ref": {"type": "string"},
        "case_kind": {"type": "string", "enum": list(CASE_KINDS)},
        "fault_summary": {"type": "string"},
        "source_message_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_spans": {"type": "array", "items": span},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "decide_atomic_case_boundaries",
            "strict": True,
            "description": "Split chat messages into evidence-bounded atomic cases.",
            "parameters": _strict_object({
                "case_fragments": {"type": "array", "items": fragment},
                "non_case_message_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            }),
        },
    }


class CaseBoundaryAgent:
    agent_id = "W7a.CaseBoundary"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self, source_ledger: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        allowed = set(source_ledger.get("allowed_message_ids") or [])
        response = self.client.call_tool(
            stage="case_boundary",
            system_prompt=SYSTEM_PROMPT,
            payload={
                "source_ledger": source_ledger,
                "allowed_message_ids": sorted(allowed),
            },
            tool=tool_schema(),
            max_tokens=16_384,
        )
        raw = (
            response.get("arguments")
            if isinstance(response.get("arguments"), dict)
            else {}
        )
        normalized, issues = validate_case_boundary_decision(
            raw,
            allowed_message_ids=allowed,
            message_text_lengths={
                str(row.get("message_id") or ""): len(
                    str(row.get("text") or "")
                )
                for row in source_ledger.get("rows") or []
                if isinstance(row, dict)
                and str(row.get("message_id") or "")
            },
        )
        call = {
            key: value
            for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
