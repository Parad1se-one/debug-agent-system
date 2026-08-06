"""Stable tool-entry router for evidence parsers.

Other agents should call this module instead of instantiating parser internals
ad hoc.  The router is intentionally conservative: unknown tools and parser
exceptions are returned as structured failures so evidence parsing never breaks
read/write orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .attachment_parser import AttachmentParserAgent
from .document_parser import DocumentParserAgent
from .dmp_parser import DmpParserAgent
from .image_parser import ImageParserAgent
from .jira_parser import JiraParserAgent, JIRA_KEY_RE, URL_RE
from .log_package_parser import LogPackageParserAgent
from .proj_parser import ProjParserAgent

TOOL_ALIASES = {
    "attachment": "attachment",
    "attachments": "attachment",
    "file": "attachment",
    "resource": "attachment",
    "resource_file": "attachment",
    "document": "document",
    "doc": "document",
    "office": "document",
    "pdf": "document",
    "spreadsheet": "document",
    "image": "image",
    "sample_image": "image",
    "screenshot": "image",
    "picture": "image",
    "proj": "proj",
    "project": "proj",
    "project_file": "proj",
    "program_file": "proj",
    "jira": "jira",
    "jira_link": "jira",
    "issue": "jira",
    "issue_link": "jira",
    "log": "log_package",
    "logs": "log_package",
    "log_package": "log_package",
    "diagnostic_log": "log_package",
    "dlog": "log_package",
    "dmp": "dmp",
    "dump": "dmp",
    "memory_dump": "dmp",
    "minidump": "dmp",
}


class EvidenceToolAgent:
    """Unified entry point for proj/Jira/attachment parsing tools."""

    schema_version = "debug_agent_system.tool_router.v1"

    def __init__(
        self,
        attachment_parser: AttachmentParserAgent | None = None,
        document_parser: DocumentParserAgent | None = None,
        dmp_parser: DmpParserAgent | None = None,
        image_parser: ImageParserAgent | None = None,
        jira_parser: JiraParserAgent | None = None,
        log_package_parser: LogPackageParserAgent | None = None,
        proj_parser: ProjParserAgent | None = None,
    ) -> None:
        self.attachment_parser = attachment_parser or AttachmentParserAgent()
        self.document_parser = document_parser or DocumentParserAgent()
        self.dmp_parser = dmp_parser or DmpParserAgent()
        self.image_parser = image_parser or ImageParserAgent()
        self.jira_parser = jira_parser or JiraParserAgent()
        self.log_package_parser = log_package_parser or LogPackageParserAgent()
        self.proj_parser = proj_parser or ProjParserAgent()

    def parse(self, tool: str, payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
        """Parse one evidence payload with a named tool.

        Returns parser-native JSON plus `tool_entry` metadata.  Failures are
        structured `parse_failed` results, not raised exceptions.
        """

        canonical = self._canonical_tool(tool)
        if canonical == "attachment":
            result = self._safe("TOOL-ATTACHMENT", self.attachment_parser.parse, payload, max_preview_bytes=max_bytes)
        elif canonical == "document":
            result = self._safe("TOOL-DOCUMENT", self.document_parser.parse, payload, max_preview_bytes=max_bytes)
        elif canonical == "dmp":
            result = self._safe("TOOL-DMP", self.dmp_parser.parse, payload, max_header_bytes=max_bytes)
        elif canonical == "image":
            result = self._safe("TOOL-IMAGE", self.image_parser.parse, payload, max_header_bytes=max_bytes)
        elif canonical == "jira":
            result = self._safe("TOOL-JIRA", self.jira_parser.parse, payload)
        elif canonical == "log_package":
            result = self._safe("TOOL-LOG-PACKAGE", self.log_package_parser.parse, payload, max_preview_bytes=max_bytes)
        elif canonical == "proj":
            proj_payload = str(payload.get("path") or payload.get("relative_path") or payload.get("name") or "") if isinstance(payload, dict) else payload
            if not str(proj_payload or ""):
                result = {
                    "schema_version": self.schema_version,
                    "type": "TOOL-PROJError",
                    "status": "parse_failed",
                    "error": "missing project file path",
                    "source": payload if isinstance(payload, dict) else {"value": str(payload)},
                    "observability": {"agent_id": "TOOL-PROJ", "status": "parse_failed"},
                }
            else:
                result = self._safe("TOOL-PROJ", self.proj_parser.parse, proj_payload, max_bytes=max_bytes)
        else:
            result = {
                "schema_version": self.schema_version,
                "type": "EvidenceToolError",
                "status": "parse_failed",
                "error": f"unknown evidence tool: {tool}",
                "source": payload if isinstance(payload, dict) else {"value": str(payload)},
                "observability": {"agent_id": "TOOL-ROUTER", "status": "parse_failed"},
            }
        result = dict(result)
        result.setdefault("status", "metadata_only")
        result["tool_entry"] = {
            "schema_version": self.schema_version,
            "requested_tool": str(tool or ""),
            "tool": canonical or "unknown",
            "agent_id": "TOOL-ROUTER",
        }
        return result

    def parse_attachment(self, payload: Any) -> dict[str, Any]:
        return self.parse("attachment", payload)

    def parse_document(self, payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
        return self.parse("document", payload, max_bytes=max_bytes)

    def parse_dmp(self, payload: Any, *, max_bytes: int = 1048576) -> dict[str, Any]:
        return self.parse("dmp", payload, max_bytes=max_bytes)

    def parse_image(self, payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
        return self.parse("image", payload, max_bytes=max_bytes)

    def parse_jira(self, payload: Any) -> dict[str, Any]:
        return self.parse("jira", payload)

    def parse_log_package(self, payload: Any) -> dict[str, Any]:
        return self.parse("log_package", payload)

    def parse_proj(self, payload: str | Path, *, max_bytes: int = 65536) -> dict[str, Any]:
        return self.parse("proj", payload, max_bytes=max_bytes)

    def infer_and_parse(self, payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
        return self.parse(self.infer_tool(payload), payload, max_bytes=max_bytes)

    def parse_many(self, items: list[dict[str, Any]] | list[Any], *, max_bytes: int = 65536) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                tool = str(item.get("tool") or item.get("kind") or item.get("evidence_role") or "")
                payload = item.get("payload", item)
            else:
                tool = ""
                payload = item
            results.append(self.parse(tool or self.infer_tool(payload), payload, max_bytes=max_bytes))
        return results

    def infer_tool(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("tool", "kind", "evidence_role", "type"):
                value = str(payload.get(key) or "")
                canonical = self._canonical_tool(value)
                if canonical:
                    return canonical
            name = str(payload.get("name") or payload.get("path") or payload.get("url") or payload.get("text") or "")
        else:
            name = str(payload or "")
        if URL_RE.search(name) or JIRA_KEY_RE.search(name):
            return "jira"
        suffix = Path(name).suffix.lower()
        if suffix in {".dmp", ".mdmp"}:
            return "dmp"
        if suffix in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
            return "document"
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return "image"
        if suffix == ".proj":
            return "proj"
        if suffix in {".zip", ".rar", ".7z", ".log", ".evtx", ".dmp", ".pml"}:
            return "log_package"
        return "attachment"

    def _canonical_tool(self, tool: str) -> str:
        return TOOL_ALIASES.get(str(tool or "").strip().lower(), "")

    def _safe(self, agent_name: str, fn: Any, payload: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(payload, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tool failures must degrade to evidence metadata.
            return {
                "schema_version": self.schema_version,
                "type": f"{agent_name}Error",
                "status": "parse_failed",
                "error": str(exc),
                "source": payload if isinstance(payload, dict) else {"value": str(payload)},
                "observability": {"agent_id": agent_name, "status": "parse_failed"},
            }


def parse_evidence(tool: str, payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
    return EvidenceToolAgent().parse(tool, payload, max_bytes=max_bytes)


def parse_attachment_evidence(payload: Any) -> dict[str, Any]:
    """Stable public entry for attachment metadata parsing by other agents."""

    return EvidenceToolAgent().parse_attachment(payload)


def parse_document_evidence(payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
    """Stable public entry for safe document metadata parsing by other agents."""

    return EvidenceToolAgent().parse_document(payload, max_bytes=max_bytes)


def parse_dmp_evidence(payload: Any, *, max_bytes: int = 1048576) -> dict[str, Any]:
    """Stable public entry for safe DMP header metadata parsing by other agents."""

    return EvidenceToolAgent().parse_dmp(payload, max_bytes=max_bytes)


def parse_image_evidence(payload: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
    """Stable public entry for safe image header parsing by other agents."""

    return EvidenceToolAgent().parse_image(payload, max_bytes=max_bytes)


def parse_jira_evidence(payload: Any) -> dict[str, Any]:
    """Stable public entry for offline Jira evidence parsing by other agents."""

    return EvidenceToolAgent().parse_jira(payload)


def parse_log_package_evidence(payload: Any) -> dict[str, Any]:
    """Stable public entry for safe log-package metadata parsing by other agents."""

    return EvidenceToolAgent().parse_log_package(payload)


def parse_proj_evidence(payload: str | Path, *, max_bytes: int = 65536) -> dict[str, Any]:
    """Stable public entry for bounded `.proj` evidence parsing by other agents."""

    return EvidenceToolAgent().parse_proj(payload, max_bytes=max_bytes)


def parse_json_payload(value: str) -> Any:
    """Decode CLI JSON payloads while keeping plain strings as strings."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


__all__ = [
    "EvidenceToolAgent",
    "TOOL_ALIASES",
    "parse_attachment_evidence",
    "parse_document_evidence",
    "parse_dmp_evidence",
    "parse_evidence",
    "parse_image_evidence",
    "parse_jira_evidence",
    "parse_json_payload",
    "parse_log_package_evidence",
    "parse_proj_evidence",
]
