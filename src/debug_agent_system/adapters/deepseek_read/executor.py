"""Deterministic implementation behind read-side function tools."""

from __future__ import annotations

from typing import Any

from debug_agent_system.agents.tools.executor import (
    ReadEvidenceToolExecutor,
    parse_evidence_tool_schema,
)
from debug_agent_system.core.contracts import to_jsonable
from debug_agent_system.runtime.system import DebugAgentSystem


def _resource_schema() -> dict[str, Any]:
    return dict(
        parse_evidence_tool_schema()["function"]["parameters"]["properties"][
            "resource"
        ]
    )


def read_side_tool_schemas() -> list[dict[str, Any]]:
    resource = _resource_schema()
    return [
        {
            "type": "function",
            "function": {
                "name": "diagnose_start",
                "description": (
                    "Start the deterministic KG_v2 diagnosis runtime. "
                    "This tool alone may lock a Variant or compile actions."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "interactive": {"type": "boolean"},
                        "session_id": {"type": "string"},
                        "routing_context": {
                            "type": "object",
                            "properties": {
                                "stage": {"type": "string"},
                                "query_type": {"type": "string"},
                                "interface": {"type": "string"},
                                "side": {"type": "string"},
                            },
                            "required": [
                                "stage",
                                "query_type",
                                "interface",
                                "side",
                            ],
                            "additionalProperties": False,
                        },
                        "evidence_resources": {
                            "type": "array",
                            "items": resource,
                            "maxItems": 12,
                        },
                    },
                    "required": [
                        "query",
                        "interactive",
                        "session_id",
                        "routing_context",
                        "evidence_resources",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "diagnose_step",
                "description": (
                    "Continue an existing deterministic diagnosis session with "
                    "user text and/or caller-supplied evidence resources."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "user_message": {"type": "string"},
                        "evidence_resources": {
                            "type": "array",
                            "items": resource,
                            "maxItems": 12,
                        },
                    },
                    "required": [
                        "session_id",
                        "user_message",
                        "evidence_resources",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_evidence",
                "description": (
                    "Read KG_v2/SAG candidates and supporting chunks without "
                    "creating or mutating a diagnosis session."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query", "limit"],
                    "additionalProperties": False,
                },
            },
        },
        parse_evidence_tool_schema(),
    ]


class ReadSideToolExecutor:
    """Execute only the four read-side tools exposed to a model."""

    allowed_tools = {
        "diagnose_start",
        "diagnose_step",
        "retrieve_evidence",
        "parse_evidence",
    }

    def __init__(
        self,
        system: DebugAgentSystem | None = None,
        evidence_executor: ReadEvidenceToolExecutor | None = None,
    ) -> None:
        self.system = system or DebugAgentSystem.from_config()
        self.evidence_executor = evidence_executor or ReadEvidenceToolExecutor()

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str = "",
    ) -> dict[str, Any]:
        if name not in self.allowed_tools:
            return {
                "schema_version": "debug_agent_system.read_tool_error.v1",
                "status": "failed",
                "failure_type": "tool_not_allowed",
                "tool": str(name),
                "call_id": call_id,
            }
        if name == "diagnose_start":
            session_id = str(arguments.get("session_id") or "")
            return self.system.start(
                {
                    "query": str(arguments.get("query") or ""),
                    "interactive": bool(arguments.get("interactive", True)),
                    "session": {"session_id": session_id} if session_id else {},
                    "routing_context": dict(
                        arguments.get("routing_context") or {}
                    ),
                    "evidence_resources": list(
                        arguments.get("evidence_resources") or []
                    ),
                }
            )
        if name == "diagnose_step":
            return self.system.step(
                str(arguments.get("session_id") or ""),
                str(arguments.get("user_message") or ""),
                evidence_resources=list(
                    arguments.get("evidence_resources") or []
                ),
            )
        if name == "retrieve_evidence":
            query = str(arguments.get("query") or "")
            limit = max(1, min(int(arguments.get("limit") or 10), 20))
            candidates = self.system.read_model.search_variants(
                query,
                limit=limit,
            )
            retrieval = self.system.read_model.last_retrieval or {}
            return {
                "schema_version": "debug_agent_system.retrieve_evidence.v1",
                "status": "ok",
                "query": query,
                "candidates": to_jsonable(candidates),
                "supporting_chunks": list(retrieval.get("chunks") or []),
                "trace": dict(retrieval.get("trace") or {}),
            }
        result = self.evidence_executor.execute(
            arguments.get("resource") or {},
            tool=str(arguments.get("tool") or "auto"),
            max_bytes=int(arguments.get("max_bytes") or 65536),
            call_id=call_id,
        )
        return to_jsonable(result)


__all__ = ["ReadSideToolExecutor", "read_side_tool_schemas"]
