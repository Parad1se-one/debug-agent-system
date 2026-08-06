"""Adapter returning qa_agentic_system AgentResult-compatible dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.runtime import DebugAgentSystem


class DebugAgentSystemQARuntime:
    backend = "debug_agent_system"

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.system = DebugAgentSystem.from_config(config_path)

    def answer(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        session: dict[str, Any] | None = None,
        evidence_resources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self.system.diagnose({
            "query": query,
            "chat_history": chat_history or [],
            "session": session or {},
            "routing_context": {"stage": "diagnosis", "query_type": "debug_issue"},
            "evidence_resources": evidence_resources or [],
        })
        obs = payload.get("observability") or {}
        answer_sections = list(payload.get("answer_sections") or [])
        return {
            "agent": "debug_agent",
            "answer": str(payload.get("answer") or ""),
            "sources": [str(x) for x in payload.get("sources") or []],
            "images": _images_from_answer_sections(answer_sections),
            "answer_sections": answer_sections,
            "confidence": float(payload.get("confidence") or 0.0),
            "required_data": [str(x) for x in payload.get("required_data") or []],
            "escalation_target": str(payload.get("escalation_target") or ""),
            "backend": self.backend,
            "observations": [
                {"type": "status", "content": str(payload.get("status") or "")},
                {"type": "debug_agent_system", "content": json.dumps(obs, ensure_ascii=False)},
            ],
        }


def _images_from_answer_sections(sections: list[Any]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if str(value.get("media_kind") or "") == "image":
            path = str(value.get("asset_path") or "").strip()
            asset_id = str(
                value.get("media_id")
                or value.get("content_hash")
                or value.get("archive_path")
                or path
            )
            if path and asset_id and asset_id not in seen:
                seen.add(asset_id)
                images.append({
                    **value,
                    "asset_id": asset_id,
                    "image_id": asset_id,
                    "path": path,
                    "caption": str(
                        value.get("context_label")
                        or value.get("label")
                        or value.get("archive_path")
                        or ""
                    ),
                })
        for item in value.values():
            visit(item)

    visit(sections)
    return images
