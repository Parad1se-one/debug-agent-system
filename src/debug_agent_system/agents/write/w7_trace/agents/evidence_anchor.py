"""Attach media and diagnostic packages to atomic case fragments."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    ANCHOR_CONFIDENCE,
    ANCHOR_ROLES,
    validate_evidence_anchor_decision,
)
from ..model_client import DecisionModelClient
from ..source_context import evidence_anchor_candidates


PROMPT_VERSION = "w7-evidence-anchor-v1"
SYSTEM_PROMPT = """\
你是工业现场群聊的附件证据归属判断器。CaseBoundary 已经完成业务问题拆分；
你只判断图片、视频、文件、日志或诊断包属于哪个 fragment，以及它在该问题中
扮演的证据角色。

规则：
1. 特别检查“附件先出现、文字问题稍后说明”的初始诊断包或首报。
2. 只能引用 allowed fragment/message/attachment ID。
3. 同一 evidence message 只能归属一个主 fragment；证据不足必须放入
   unassigned_evidence_message_ids，不能为了覆盖率强行关联。
4. 文件名相同、同群或时间相邻本身不是充分依据；后续回复、Jira、设备、错误码和
   明确指代可作为理由。
5. 不生成 Action、Outcome、Trace 或 KG ID。

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
        "evidence_message_id": {"type": "string"},
        "attachment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "target_fragment_ref": {"type": "string"},
        "role": {"type": "string", "enum": list(ANCHOR_ROLES)},
        "confidence": {
            "type": "string",
            "enum": list(ANCHOR_CONFIDENCE),
        },
        "reasons": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "attach_evidence_to_case_fragments",
            "strict": True,
            "description": (
                "Attach media/log/package evidence to one atomic case fragment "
                "or leave it explicitly unassigned."
            ),
            "parameters": _strict_object({
                "anchor_decisions": {
                    "type": "array",
                    "items": decision,
                },
                "unassigned_evidence_message_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
        },
    }


class EvidenceAnchorAgent:
    agent_id = "W7a.EvidenceAnchor"
    version = PROMPT_VERSION

    def __init__(self, client: DecisionModelClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        source_ledger: dict[str, Any],
        case_boundary: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        candidates = evidence_anchor_candidates(source_ledger)
        fragment_refs = {
            str(item.get("fragment_ref") or "")
            for item in case_boundary.get("case_fragments") or []
            if isinstance(item, dict) and str(item.get("fragment_ref") or "")
        }
        candidate_ids = {
            str(item.get("message_id") or "")
            for item in candidates
            if str(item.get("message_id") or "")
        }
        attachments_by_message = {
            str(item.get("message_id") or ""): {
                str(attachment.get("attachment_id") or "")
                for attachment in item.get("attachment_refs") or []
                if isinstance(attachment, dict)
                and str(attachment.get("attachment_id") or "")
            }
            for item in candidates
        }
        response = self.client.call_tool(
            stage="evidence_anchor",
            system_prompt=SYSTEM_PROMPT,
            payload={
                "case_fragments": case_boundary.get("case_fragments") or [],
                "evidence_candidates": candidates,
                "allowed_fragment_refs": sorted(fragment_refs),
                "allowed_evidence_message_ids": sorted(candidate_ids),
                "allowed_attachment_ids_by_message": {
                    key: sorted(value)
                    for key, value in attachments_by_message.items()
                },
            },
            tool=tool_schema(),
            max_tokens=8_192,
        )
        raw = (
            response.get("arguments")
            if isinstance(response.get("arguments"), dict)
            else {}
        )
        normalized, issues = validate_evidence_anchor_decision(
            raw,
            allowed_fragment_refs=fragment_refs,
            candidate_message_ids=candidate_ids,
            allowed_attachment_ids_by_message=attachments_by_message,
        )
        call = {
            key: value for key, value in response.items()
            if key != "arguments"
        }
        return normalized, issues, call
