"""Context-level evidence parser for local imported samples.

This agent does not replace the file parsers.  It reads local context files
such as ``source_manifest.json`` and routes the referenced raw files through
``EvidenceToolAgent`` so W2/W6 can consume one coherent evidence package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.tools.router import EvidenceToolAgent


TOOL_EVIDENCE_KEYS = {
    "attachment": "attachment_parse_results",
    "document": "document_parse_results",
    "dmp": "dmp_parse_results",
    "image": "image_parse_results",
    "jira": "jira_parse_results",
    "log_package": "log_package_parse_results",
    "proj": "proj_parse_results",
}


class EvidenceContextParserAgent:
    """Parse one local evidence context directory into grouped tool evidence."""

    schema_version = "debug_agent_system.tool.evidence_context_parse.v1"

    def __init__(self, tool_agent: EvidenceToolAgent | None = None) -> None:
        self.tool_agent = tool_agent or EvidenceToolAgent()

    def parse_context(self, root: str | Path, *, max_bytes: int = 65536, limit: int = 0) -> dict[str, Any]:
        root_path = Path(root)
        if root_path.is_file():
            contexts = [self._parse_file_context(root_path, max_bytes=max_bytes)]
        else:
            manifests = self._find_manifests(root_path, limit=limit)
            if manifests:
                contexts = [self._parse_manifest(path, max_bytes=max_bytes) for path in manifests]
            elif "jira_offline" in {part.lower() for part in root_path.parts}:
                contexts = self._parse_jira_offline_root(root_path, max_bytes=max_bytes, limit=limit)
            else:
                contexts = [self._parse_directory_context(root_path, max_bytes=max_bytes, limit=limit)]
        return self._result(root_path, contexts)

    def _find_manifests(self, root: Path, *, limit: int) -> list[Path]:
        if not root.exists() or not root.is_dir():
            return []
        if (root / "source_manifest.json").is_file():
            return [root / "source_manifest.json"]
        manifests = sorted(root.glob("*/source_manifest.json"))
        return manifests[:limit] if limit > 0 else manifests

    def _parse_manifest(self, manifest_path: Path, *, max_bytes: int) -> dict[str, Any]:
        manifest = self._read_json(manifest_path)
        context_root = manifest_path.parent
        files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
        file_items: list[dict[str, Any]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            relative_path = str(item.get("relative_path") or item.get("path") or item.get("name") or "")
            file_path = context_root / relative_path
            file_items.append(self._parse_file(file_path, declared=item, max_bytes=max_bytes))
        return self._context(
            context_id=str(manifest.get("sample_id") or context_root.name),
            context_root=context_root,
            source_manifest=manifest,
            files=file_items,
        )

    def _parse_directory_context(self, root: Path, *, max_bytes: int, limit: int) -> dict[str, Any]:
        files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name != "source_manifest.json"]
        if limit > 0:
            files = files[:limit]
        return self._context(
            context_id=root.name,
            context_root=root,
            source_manifest={},
            files=[self._parse_file(path, declared={}, max_bytes=max_bytes) for path in files],
        )

    def _parse_file_context(self, path: Path, *, max_bytes: int) -> dict[str, Any]:
        if path.name == "source_manifest.json":
            return self._parse_manifest(path, max_bytes=max_bytes)
        return self._context(
            context_id=path.stem,
            context_root=path.parent,
            source_manifest={},
            files=[self._parse_file(path, declared={}, max_bytes=max_bytes)],
        )

    def _parse_jira_offline_root(self, root: Path, *, max_bytes: int, limit: int) -> list[dict[str, Any]]:
        candidates = [path for path in sorted(root.rglob("*.json")) if path.is_file()]
        candidates = [path for path in candidates if path.parent.name == "fault_details" or self._looks_like_jira_json(path)]
        if limit > 0:
            candidates = candidates[:limit]
        return [self._parse_jira_file(path, max_bytes=max_bytes) for path in candidates]

    def _parse_jira_file(self, path: Path, *, max_bytes: int) -> dict[str, Any]:
        data = self._read_json(path)
        issue_key = str(data.get("key") or path.stem)
        raw_root = path.parent.parent if path.parent.name == "fault_details" else path.parent
        jira_agent = EvidenceToolAgent()
        jira_agent.jira_parser.offline_root = raw_root
        parsed = jira_agent.parse("jira", issue_key, max_bytes=max_bytes)
        return self._context(
            context_id=issue_key,
            context_root=path.parent,
            source_manifest={
                "sample_id": issue_key,
                "source": "jira_offline",
                "anchor_messages": [str(data.get("summary") or "")],
                "files": [{"name": path.name, "relative_path": path.name, "size": path.stat().st_size if path.exists() else None}],
            },
            files=[{
                "name": path.name,
                "path": str(path),
                "relative_path": path.name,
                "size": path.stat().st_size if path.exists() else None,
                "tool": "jira",
                "parse_result": parsed,
            }],
        )

    def _parse_file(self, path: Path, *, declared: dict[str, Any], max_bytes: int) -> dict[str, Any]:
        payload = self._payload(path, declared)
        tool = self._tool_for(path, payload)
        parsed = self.tool_agent.parse(tool, payload, max_bytes=max_bytes)
        return {
            "name": path.name or str(declared.get("name") or ""),
            "path": str(path),
            "relative_path": str(declared.get("relative_path") or declared.get("path") or path.name),
            "size": path.stat().st_size if path.exists() and path.is_file() else declared.get("size"),
            "tool": parsed.get("tool_entry", {}).get("tool") or tool,
            "parse_result": parsed,
            "evidence_origin": str(declared.get("evidence_origin") or ""),
            "binding_status": str(declared.get("binding_status") or ""),
            "source_message_id": str(declared.get("source_message_id") or ""),
            "source_create_time": str(declared.get("source_create_time") or ""),
        }

    def _payload(self, path: Path, declared: dict[str, Any]) -> dict[str, Any]:
        payload = dict(declared)
        payload.setdefault("name", path.name)
        payload.setdefault("path", str(path))
        payload.setdefault("relative_path", str(declared.get("relative_path") or path.name))
        if path.exists() and path.is_file():
            payload.setdefault("size", path.stat().st_size)
        return payload

    def _tool_for(self, path: Path, payload: dict[str, Any]) -> str:
        suffix = path.suffix.lower()
        if suffix in {".dmp", ".mdmp"}:
            return "dmp"
        if suffix == ".proj":
            return "proj"
        if suffix in {".zip", ".rar", ".7z", ".log", ".evtx", ".pml"}:
            return "log_package"
        if suffix in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
            return "document"
        return self.tool_agent.infer_tool(payload)

    def _context(self, *, context_id: str, context_root: Path, source_manifest: dict[str, Any], files: list[dict[str, Any]]) -> dict[str, Any]:
        tool_evidence = self._group_tool_evidence(files)
        return {
            "schema_version": self.schema_version,
            "type": "EvidenceContext",
            "context_id": context_id,
            "context_root": str(context_root),
            "source_context": {
                "sample_id": source_manifest.get("sample_id") or context_id,
                "source": source_manifest.get("source") or "",
                "chat_name": source_manifest.get("chat_name") or "",
                "segment_id": source_manifest.get("segment_id") or "",
                "anchor_messages": list(source_manifest.get("anchor_messages") or []),
                "expected_tools": list(source_manifest.get("expected_tools") or []),
                "notes": source_manifest.get("notes") or "",
            },
            "source_manifest": source_manifest,
            "files": files,
            "tool_evidence": tool_evidence,
            "summary_hints": self._summary_hints(tool_evidence),
            "safety": self._safety_summary(tool_evidence),
        }

    def _group_tool_evidence(self, files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped = {key: [] for key in TOOL_EVIDENCE_KEYS.values()}
        for item in files:
            parsed = item.get("parse_result")
            if not isinstance(parsed, dict):
                continue
            tool = str(item.get("tool") or parsed.get("tool_entry", {}).get("tool") or "")
            key = TOOL_EVIDENCE_KEYS.get(tool)
            if key:
                grouped[key].append(parsed)
        return grouped

    def _summary_hints(self, tool_evidence: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        hints: dict[str, Any] = {
            "issue_keys": [],
            "versions": [],
            "ip_addresses": [],
            "project_names": [],
            "error_codes": [],
            "phase_hints": [],
            "dump_kinds": [],
            "detected_log_roles": [],
            "has_dmp": False,
            "has_evtx": False,
            "has_startup_log": False,
            "has_dlog": False,
        }
        for result in tool_evidence.get("jira_parse_results") or []:
            hints["issue_keys"].extend(result.get("issue_keys") or [])
            hints["versions"].extend(result.get("version_hints") or [])
        for result in tool_evidence.get("proj_parse_results") or []:
            key_hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
            hints["versions"].extend(key_hints.get("versions") or [])
            hints["ip_addresses"].extend(key_hints.get("ip_addresses") or [])
            hints["project_names"].extend(key_hints.get("project_names") or [])
        for result in tool_evidence.get("dmp_parse_results") or []:
            if result.get("dump_kind"):
                hints["dump_kinds"].append(result["dump_kind"])
            hints["error_codes"].extend(result.get("bugcheck_hints") or [])
        for result in tool_evidence.get("log_package_parse_results") or []:
            text_hints = result.get("text_hints") if isinstance(result.get("text_hints"), dict) else {}
            hints["error_codes"].extend(text_hints.get("error_codes") or [])
            hints["phase_hints"].extend(text_hints.get("phase_hints") or [])
            hints["detected_log_roles"].extend(result.get("detected_roles") or [])
            hints["has_dmp"] = hints["has_dmp"] or bool(result.get("has_dmp"))
            hints["has_evtx"] = hints["has_evtx"] or bool(result.get("has_evtx"))
            hints["has_startup_log"] = hints["has_startup_log"] or bool(result.get("has_startup_log"))
            hints["has_dlog"] = hints["has_dlog"] or bool(result.get("has_dlog"))
        for key, value in list(hints.items()):
            if isinstance(value, list):
                hints[key] = _unique(value, limit=50)
        return hints

    def _safety_summary(self, tool_evidence: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        results = [item for items in tool_evidence.values() for item in items if isinstance(item, dict)]
        return {
            "archive_extracted": any(bool(item.get("archive_extracted")) for item in results),
            "debugger_executed": any(bool(item.get("debugger_executed")) for item in results),
            "full_content_read": any(bool(item.get("full_content_read")) for item in results),
            "ocr_performed": any(bool(item.get("ocr_performed")) for item in results),
            "mutated": any(bool(item.get("mutated")) for item in results),
        }

    def _result(self, root: Path, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        tool_counts = {key: 0 for key in TOOL_EVIDENCE_KEYS.values()}
        for context in contexts:
            for key, items in (context.get("tool_evidence") or {}).items():
                tool_counts[key] = tool_counts.get(key, 0) + len(items or [])
        return {
            "schema_version": self.schema_version,
            "type": "EvidenceContextParseResult",
            "root": str(root),
            "context_count": len(contexts),
            "file_count": sum(len(context.get("files") or []) for context in contexts),
            "tool_counts": tool_counts,
            "contexts": contexts,
            "observability": {
                "agent_id": "TOOL-EVIDENCE-CONTEXT",
                "boundary": "context_manifest_plus_safe_file_parsers",
            },
        }

    def _looks_like_jira_json(self, path: Path) -> bool:
        data = self._read_json(path)
        return bool(isinstance(data, dict) and (data.get("key") or data.get("summary") or data.get("description")))

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}


def parse_evidence_context(root: str | Path, *, max_bytes: int = 65536, limit: int = 0) -> dict[str, Any]:
    return EvidenceContextParserAgent().parse_context(root, max_bytes=max_bytes, limit=limit)


def _unique(values: list[Any], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


__all__ = ["EvidenceContextParserAgent", "parse_evidence_context"]
