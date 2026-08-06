"""Safe Windows dump evidence parser.

The DMP parser is intentionally metadata/header-only.  It never invokes WinDbg,
never scans the full dump, and never extracts private memory contents.  The goal
is to make dump evidence auditable and ready for an explicitly approved debugger
step, not to infer root cause by itself.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

DMP_EXTS = {".dmp", ".mdmp"}
ASCII_RE = re.compile(rb"[ -~]{4,}")
BUGCHECK_TEXT_RE = re.compile(r"\b(?:bugcheck|stop code|0x[0-9a-fA-F]{6,16})\b", re.IGNORECASE)


class DmpParserAgent:
    """Tool entry for safe DMP metadata and header inspection."""

    schema_version = "debug_agent_system.tool.dmp_parse.v1"

    def parse(self, dump: str | Path | dict[str, Any], *, max_header_bytes: int = 1048576, max_strings: int = 80) -> dict[str, Any]:
        if isinstance(dump, dict):
            path_text = str(dump.get("path") or dump.get("relative_path") or "")
            name = str(dump.get("name") or Path(path_text).name or dump.get("file_key") or "")
            source = dict(dump)
            declared_size = dump.get("size") or dump.get("bytes")
        else:
            path = Path(dump)
            path_text = str(path)
            name = path.name
            source = {"path": path_text}
            declared_size = None
        file_path = Path(path_text) if path_text else Path(name)
        exists = bool(path_text and file_path.exists())
        is_file = file_path.is_file() if exists else False
        suffix = Path(name or path_text).suffix.lower()
        size = file_path.stat().st_size if is_file else _int_or_none(declared_size)
        raw = b""
        read_error = ""
        if is_file:
            try:
                with file_path.open("rb") as f:
                    raw = f.read(max(0, max_header_bytes))
            except OSError as exc:
                read_error = str(exc)
        signature = raw[:8].decode("ascii", errors="replace") if raw else ""
        strings = _ascii_strings(raw, limit=max_strings)
        bugcheck_hints = _bugcheck_hints(strings)
        dump_kind = _dump_kind(signature, suffix)
        architecture_hint = "x64" if "64" in signature else ""
        looks_like_dmp = suffix in DMP_EXTS or bool(dump_kind)
        return {
            "schema_version": self.schema_version,
            "type": "DmpParseResult",
            "name": name or file_path.name,
            "path": path_text,
            "extension": suffix,
            "exists": exists,
            "size": size,
            "status": "header_metadata" if raw else "metadata_only",
            "looks_like_dmp": looks_like_dmp,
            "dump_kind": dump_kind,
            "architecture_hint": architecture_hint,
            "header_signature": signature,
            "header_read": bool(raw),
            "header_bytes_read": len(raw),
            "max_header_bytes": max_header_bytes,
            "header_sha1_prefix": hashlib.sha1(raw).hexdigest()[:12] if raw else "",
            "full_content_read": False,
            "memory_content_extracted": False,
            "strings_sample": strings[:20],
            "bugcheck_hints": bugcheck_hints,
            "debugger_executed": False,
            "windbg_analysis_performed": False,
            "windbg_ready": bool(is_file and looks_like_dmp),
            "suggested_debugger_commands": ["!analyze -v", "lm", "kv"] if is_file and looks_like_dmp else [],
            "read_error": read_error,
            "source": source,
            "observability": {
                "agent_id": "TOOL-DMP",
                "boundary": "header_metadata_only",
            },
        }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ascii_strings(raw: bytes, *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in ASCII_RE.finditer(raw):
        text = match.group(0).decode("ascii", errors="ignore").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:200])
        if len(out) >= limit:
            break
    return out


def _bugcheck_hints(strings: list[str]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for text in strings:
        if not BUGCHECK_TEXT_RE.search(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        hints.append(text[:200])
        if len(hints) >= 20:
            break
    return hints


def _dump_kind(signature: str, suffix: str) -> str:
    sig = str(signature or "")
    if sig.startswith("PAGEDU64"):
        return "windows_kernel_or_complete_dump_64_candidate"
    if sig.startswith("PAGE") and "DU" in sig:
        return "windows_kernel_or_complete_dump_candidate"
    if sig.startswith("MDMP"):
        return "windows_minidump_candidate"
    if suffix in DMP_EXTS:
        return "windows_dump_candidate"
    return ""


__all__ = ["DmpParserAgent"]
