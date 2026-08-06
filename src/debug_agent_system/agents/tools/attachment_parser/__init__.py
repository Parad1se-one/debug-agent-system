"""Safe attachment evidence parser.

This tool is intentionally conservative: it classifies attachment evidence from
paths or W1 metadata, and reads only bounded previews from whitelisted text
attachments.  It never executes files, extracts archives, OCRs images, or
follows links.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Any

LOG_PACKAGE_EXTS = {".zip", ".rar", ".7z", ".log", ".evtx", ".dmp", ".pml"}
PROGRAM_FILE_EXTS = {".proj"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
CONFIG_EXTS = {".toml", ".ini", ".cfg", ".json", ".yaml", ".yml", ".reg"}
DATA_EXTS = {".csv", ".txt", ".xls", ".xlsx", ".pdf", ".model"}
TEXT_PREVIEW_EXTS = {".txt", ".csv", ".json", ".ini", ".toml", ".cfg", ".yaml", ".yml", ".reg"}
VERSION_RE = re.compile(r"(?<![\d.])(?:v)?\d{1,2}\.\d+(?:\.\d+){0,2}(?![\d.])", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
JIRA_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}-\d+\b")
URL_RE = re.compile(r"https?://[^\s\]）)，,。；;]+", re.IGNORECASE)
HEX_ERROR_RE = re.compile(r"\b0x[0-9A-Fa-f]{6,}\b")
ERROR_LINE_RE = re.compile(r"(error|exception|failed|fail|timeout|fatal|报错|异常|失败|超时|错误)", re.IGNORECASE)
PHASE_HINTS = {
    "startup": ("startup", "init", "initialize", "初始化", "启动"),
    "network": ("network", "ethernet", "ip", "网卡", "网络"),
    "camera": ("camera", "相机"),
    "detection": ("detect", "inspection", "检测", "复判"),
    "programming": ("program", "recipe", "模板", "配方", "编程"),
}


def classify_attachment(name: str, kind: str = "") -> str:
    ext = Path(str(name or "")).suffix.lower()
    kind_text = str(kind or "").lower()
    if ext in PROGRAM_FILE_EXTS:
        return "program_file"
    if ext in LOG_PACKAGE_EXTS:
        return "log_package"
    if kind_text == "image" or ext in IMAGE_EXTS:
        return "sample_image"
    if ext in CONFIG_EXTS:
        return "environment"
    if ext in DATA_EXTS:
        return "data_file"
    return "attachment"


class AttachmentParserAgent:
    """Tool entry for parsing attachment evidence safely."""

    schema_version = "debug_agent_system.tool.attachment_parse.v1"

    def parse(self, attachment: str | Path | dict[str, Any], *, max_preview_bytes: int = 65536) -> dict[str, Any]:
        if isinstance(attachment, dict):
            path_text = str(attachment.get("path") or "")
            name = str(attachment.get("name") or attachment.get("file_key") or Path(path_text).name)
            kind = str(attachment.get("kind") or "")
            size = attachment.get("size")
            status = str(attachment.get("status") or "metadata_only")
            source = dict(attachment)
        else:
            path = Path(attachment)
            path_text = str(path)
            name = path.name
            kind = ""
            size = path.stat().st_size if path.exists() and path.is_file() else None
            status = "metadata_only"
            source = {"path": path_text}
        ext = Path(name).suffix.lower()
        role = classify_attachment(name, kind)
        safe_to_read = self._safe_to_read_text_preview(role, ext)
        preview = self._text_preview(path_text, max_preview_bytes=max_preview_bytes) if safe_to_read else {}
        content_read = bool(preview.get("text_preview_read"))
        return {
            "schema_version": self.schema_version,
            "type": "AttachmentParseResult",
            "name": name,
            "path": path_text,
            "extension": ext,
            "kind": kind or ("file" if ext else ""),
            "size": size,
            "mime_guess": mimetypes.guess_type(name)[0],
            "evidence_role": role,
            "status": "text_hints" if content_read else status,
            "safe_to_read_text_preview": safe_to_read,
            "content_read": content_read,
            "text_preview_read": content_read,
            "text_preview": preview.get("text_preview", ""),
            "text_preview_truncated": bool(preview.get("text_preview_truncated", False)),
            "key_hints": preview.get("key_hints", {}),
            "archive_extracted": False,
            "link_fetched": False,
            "source": source,
            "observability": {
                "agent_id": "TOOL-ATTACHMENT",
                "boundary": "bounded_text_preview" if content_read else "metadata_only",
            },
        }

    def _safe_to_read_text_preview(self, role: str, ext: str) -> bool:
        if ext not in TEXT_PREVIEW_EXTS:
            return False
        # `.proj`, archives/log dumps, images, Office/PDF/model/binary formats are
        # routed to dedicated parsers or kept metadata-only.
        return role in {"environment", "data_file", "attachment"}

    def _text_preview(self, path_text: str, *, max_preview_bytes: int) -> dict[str, Any]:
        path = Path(path_text)
        if not path_text or not path.exists() or not path.is_file():
            return {"text_preview_read": False, "key_hints": {}}
        try:
            raw = path.read_bytes()[: max(1, max_preview_bytes)]
        except OSError as exc:
            return {"text_preview_read": False, "text_preview_error": str(exc), "key_hints": {}}
        text = raw.decode("utf-8", errors="replace")
        preview = " ".join(text.replace("\r", "\n").split())[:2000]
        return {
            "text_preview_read": bool(raw),
            "text_preview_truncated": bool(path.stat().st_size > len(raw)),
            "text_preview": preview,
            "key_hints": self._key_hints(text),
        }

    def _key_hints(self, text: str) -> dict[str, Any]:
        ips = self._unique(IP_RE.findall(text))
        versions = [value for value in VERSION_RE.findall(text) if value not in ips]
        lines = [line.strip() for line in text.splitlines() if ERROR_LINE_RE.search(line)]
        phases = [
            phase
            for phase, keywords in PHASE_HINTS.items()
            if any(keyword.lower() in text.lower() for keyword in keywords)
        ]
        return {
            "versions": self._unique(versions)[:20],
            "ip_addresses": ips[:20],
            "jira_ids": self._unique(JIRA_KEY_RE.findall(text))[:20],
            "urls": self._unique(URL_RE.findall(text))[:20],
            "error_codes": self._unique(HEX_ERROR_RE.findall(text))[:20],
            "error_lines": self._unique(lines, limit=160)[:20],
            "phase_hints": phases[:20],
        }

    def _unique(self, values: Any, *, limit: int = 80) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text[:limit])
        return out


__all__ = ["AttachmentParserAgent", "classify_attachment"]
