"""Stable read-side Tool contracts and bounded parser execution.

The existing parser agents intentionally expose parser-specific dictionaries.
This module adds the uniform envelope consumed by the evidence-gap resolver and
the optional DeepSeek Tool Calling harness.  It does not execute diagnostic
actions and it never turns a parser result into a KG_v2 branch decision.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from debug_agent_system.core.contracts import (
    EvidenceObservation,
    EvidenceResource,
    ToolResultEnvelope,
)

from .router import EvidenceToolAgent

TOOL_RESULT_SCHEMA = "debug_agent_system.read_tool_result.v1"


def parse_evidence_tool_schema() -> dict[str, Any]:
    """Return the strict function schema exposed to DeepSeek."""

    return {
        "type": "function",
        "function": {
            "name": "parse_evidence",
            "description": (
                "Inspect one caller-supplied evidence resource with a bounded, "
                "read-only parser. Images expose metadata only; no OCR is performed."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "attachment",
                            "document",
                            "dmp",
                            "image",
                            "jira",
                            "log_package",
                            "proj",
                        ],
                    },
                    "resource": {
                        "type": "object",
                        "properties": {
                            "resource_id": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "auto",
                                    "attachment",
                                    "document",
                                    "dmp",
                                    "image",
                                    "jira",
                                    "log_package",
                                    "proj",
                                ],
                            },
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "url": {"type": "string"},
                            "text": {"type": "string"},
                            "mime": {"type": "string"},
                            "size": {"type": ["integer", "null"]},
                            "sha256": {"type": "string"},
                            "source_message_id": {"type": "string"},
                            "metadata": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                                "additionalProperties": False,
                            },
                        },
                        "required": [
                            "resource_id",
                            "kind",
                            "name",
                            "path",
                            "url",
                            "text",
                            "mime",
                            "size",
                            "sha256",
                            "source_message_id",
                            "metadata",
                        ],
                        "additionalProperties": False,
                    },
                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": 1048576},
                },
                "required": ["tool", "resource", "max_bytes"],
                "additionalProperties": False,
            },
        },
    }


class ReadEvidenceToolExecutor:
    """Execute existing parsers and normalize their outputs."""

    def __init__(self, tool_agent: EvidenceToolAgent | None = None) -> None:
        self.tool_agent = tool_agent or EvidenceToolAgent()

    def execute(
        self,
        resource: EvidenceResource | dict[str, Any],
        *,
        tool: str = "auto",
        max_bytes: int = 65536,
        call_id: str = "",
    ) -> ToolResultEnvelope:
        normalized = normalize_resource(resource)
        requested_tool = tool if tool and tool != "auto" else normalized.kind
        payload = resource_payload(normalized)
        canonical_tool = (
            self.tool_agent.infer_tool(payload)
            if requested_tool in {"", "auto"}
            else str(requested_tool)
        )
        fingerprint = tool_call_fingerprint(
            "parse_evidence",
            {"tool": canonical_tool, "resource": payload, "max_bytes": int(max_bytes)},
        )
        parser_result = self.tool_agent.parse(
            canonical_tool,
            payload,
            max_bytes=max(1, min(int(max_bytes), 1048576)),
        )
        observations = self._observations(
            normalized,
            canonical_tool,
            parser_result,
            fingerprint,
        )
        parser_status = str(parser_result.get("status") or "")
        if parser_status == "parse_failed":
            status = "parse_failed"
        elif observations:
            status = "parsed"
        else:
            status = "metadata_only"
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for observation in observations
                for evidence_id in observation.evidence_ids
            )
        )
        source_ids = _source_ids(normalized)
        errors = []
        if parser_result.get("error") or parser_result.get("listing_error"):
            errors.append(
                {
                    "code": "parser_error",
                    "message": str(
                        parser_result.get("error")
                        or parser_result.get("listing_error")
                    ),
                }
            )
        return ToolResultEnvelope(
            schema_version=TOOL_RESULT_SCHEMA,
            tool=canonical_tool,
            call_id=call_id or f"call-{fingerprint[:12]}",
            call_fingerprint=fingerprint,
            status=status,  # type: ignore[arg-type]
            resource_id=normalized.resource_id,
            payload=parser_result,
            observations=observations,
            evidence_ids=evidence_ids,
            source_ids=source_ids,
            errors=errors,
            excluded=self._excluded(canonical_tool, parser_result),
            safety={
                "read_only": True,
                "archive_extracted": bool(parser_result.get("archive_extracted")),
                "debugger_executed": bool(parser_result.get("debugger_executed")),
                "full_content_read": bool(parser_result.get("full_content_read")),
                "ocr_performed": bool(parser_result.get("ocr_performed")),
                "mutated": bool(parser_result.get("mutated")),
            },
            observability={
                "agent_id": "READ-TOOL-EXECUTOR",
                "parser_agent_id": str(
                    (parser_result.get("observability") or {}).get("agent_id") or ""
                ),
                "resource_sha256": normalized.sha256,
                "source_message_id": normalized.source_message_id,
            },
        )

    def _observations(
        self,
        resource: EvidenceResource,
        tool: str,
        result: dict[str, Any],
        fingerprint: str,
    ) -> list[EvidenceObservation]:
        values: list[tuple[str, Any, float, str, bool]] = []
        key_hints = (
            result.get("key_hints")
            if isinstance(result.get("key_hints"), dict)
            else {}
        )
        text_hints = (
            result.get("text_hints")
            if isinstance(result.get("text_hints"), dict)
            else {}
        )
        for field in ("error_codes", "error_lines", "versions", "ip_addresses", "phase_hints", "project_names"):
            value = key_hints.get(field) or text_hints.get(field)
            if value:
                values.append((field, value, 0.92, "bounded_text_hint", True))
        for field in ("detected_roles", "dump_kind", "bugcheck_hints"):
            value = result.get(field)
            if value:
                values.append((field, value, 0.9, "bounded_binary_or_manifest_hint", True))
        if tool == "jira":
            for field in ("issue_keys", "version_hints", "site_hints", "summary_hint"):
                value = result.get(field)
                if value:
                    values.append((field, value, 0.9, "offline_jira_metadata", True))
            details = result.get("offline_details")
            if isinstance(details, list):
                for detail in details[:5]:
                    if not isinstance(detail, dict):
                        continue
                    for field in ("summary", "description_preview", "comment_preview_text", "status", "resolution"):
                        value = detail.get(field)
                        if value:
                            values.append((f"jira_{field}", value, 0.92, "offline_jira_detail", True))
        if bool(result.get("text_preview_read")) and result.get("text_preview"):
            values.append(
                (
                    f"{tool}_text_preview",
                    str(result.get("text_preview"))[:2000],
                    0.82,
                    "bounded_text_preview",
                    True,
                )
            )
        if tool == "document" and isinstance(result.get("entries"), list):
            entry_previews = [
                str(item.get("text_preview") or "")
                for item in result["entries"]
                if isinstance(item, dict) and item.get("text_preview_read")
            ]
            if entry_previews:
                values.append(
                    (
                        "document_text_preview",
                        "\n".join(entry_previews)[:4000],
                        0.82,
                        "bounded_document_preview",
                        True,
                    )
                )
        if tool == "image":
            image_meta = {
                key: result.get(key)
                for key in ("format", "width", "height", "orientation")
                if result.get(key) not in (None, "")
            }
            if image_meta:
                values.append(
                    ("image_metadata", image_meta, 0.98, "image_header_only", False)
                )
        source_ids = _source_ids(resource)
        observations: list[EvidenceObservation] = []
        for index, (field, value, confidence, mode, supports_retrieval) in enumerate(values):
            if value in (None, "", []):
                continue
            evidence_id = f"tool-evidence:{fingerprint[:16]}:{index}"
            observations.append(
                EvidenceObservation(
                    observation_id=f"observation:{fingerprint[:16]}:{index}",
                    field=field,
                    value=value,
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    source_ids=source_ids,
                    extraction_mode=mode,
                    supports_retrieval=supports_retrieval,
                )
            )
        return observations

    @staticmethod
    def _excluded(tool: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        excluded: list[dict[str, Any]] = []
        if tool == "image":
            excluded.append(
                {
                    "material": "image_text",
                    "reason": "ocr_not_supported",
                }
            )
        if result.get("text_preview_truncated"):
            excluded.append(
                {
                    "material": "remaining_text",
                    "reason": "bounded_preview_limit",
                }
            )
        if str(result.get("status") or "") == "parse_failed":
            excluded.append(
                {
                    "material": "resource_content",
                    "reason": "parser_failed",
                }
            )
        return excluded


def normalize_resource(
    resource: EvidenceResource | dict[str, Any],
) -> EvidenceResource:
    if isinstance(resource, EvidenceResource):
        normalized = resource
    else:
        data = dict(resource or {})
        path = str(data.get("path") or data.get("relative_path") or "")
        name = str(data.get("name") or data.get("file_key") or Path(path).name)
        identity_seed = "|".join(
            [
                str(data.get("source_message_id") or ""),
                path,
                str(data.get("url") or ""),
                name,
            ]
        )
        resource_id = str(data.get("resource_id") or "")
        if not resource_id:
            resource_id = f"resource:{hashlib.sha256(identity_seed.encode('utf-8')).hexdigest()[:16]}"
        size = data.get("size")
        try:
            normalized_size = int(size) if size is not None else None
        except (TypeError, ValueError):
            normalized_size = None
        normalized = EvidenceResource(
            resource_id=resource_id,
            kind=str(data.get("kind") or data.get("tool") or "auto"),  # type: ignore[arg-type]
            name=name,
            path=path,
            url=str(data.get("url") or ""),
            text=str(data.get("text") or ""),
            mime=str(data.get("mime") or data.get("mime_type") or ""),
            size=normalized_size,
            sha256=str(data.get("sha256") or ""),
            source_message_id=str(data.get("source_message_id") or ""),
            metadata=dict(data.get("metadata") or {}),
        )
    path = Path(normalized.path) if normalized.path else None
    size = normalized.size
    digest = normalized.sha256
    if path is not None and path.exists() and path.is_file():
        size = size if size is not None else path.stat().st_size
        digest = digest or _sha256_file(path)
    mime = normalized.mime or mimetypes.guess_type(normalized.name or normalized.path)[0] or ""
    return replace(normalized, size=size, sha256=digest, mime=mime)


def resource_payload(resource: EvidenceResource) -> dict[str, Any]:
    payload = {
        "resource_id": resource.resource_id,
        "kind": resource.kind,
        "name": resource.name,
        "path": resource.path,
        "url": resource.url,
        "text": resource.text,
        "mime": resource.mime,
        "size": resource.size,
        "sha256": resource.sha256,
        "source_message_id": resource.source_message_id,
        "metadata": resource.metadata,
    }
    if resource.url and not resource.path:
        payload["value"] = resource.url
    return payload


def tool_call_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"name": str(name), "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_ids(resource: EvidenceResource) -> list[str]:
    return list(
        dict.fromkeys(
            item
            for item in (resource.resource_id, resource.source_message_id)
            if item
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ReadEvidenceToolExecutor",
    "TOOL_RESULT_SCHEMA",
    "normalize_resource",
    "parse_evidence_tool_schema",
    "resource_payload",
    "tool_call_fingerprint",
]
