"""Safe log-package evidence parser.

This tool inspects log evidence without extraction or execution.  Zip archives
are listed through the central directory; whitelisted text log entries may be
read as a bounded preview for error/phase hints only.  Binary logs and other
archive formats stay metadata-only.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

TEXT_LOG_EXTS = {".log", ".txt", ".csv"}
LOG_EXTS = {*TEXT_LOG_EXTS, ".evtx", ".dmp", ".pml"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z"}
STARTUP_MARKERS = ("startup", "start", "init", "初始化", "启动")
DLOG_MARKERS = ("dlog", "diagnostic", "诊断")
ERROR_MARKERS = ("error", "exception", "crash", "dump", "bugcheck", "错误", "异常", "蓝屏", "崩溃")
PHASE_MARKERS = {
    "startup": ("startup", "start", "init", "初始化", "启动"),
    "camera": ("camera", "相机", "拍照", "capture"),
    "network": ("ip", "network", "socket", "tcp", "网卡", "网络", "连接"),
    "motion": ("motion", "axis", "servo", "运动", "轴", "伺服"),
    "light": ("light", "光源", "光控"),
    "programming": ("program", "recipe", "cad", "gerber", "编程", "配方"),
    "detection": ("detect", "inspection", "检测", "复判"),
}
ERROR_CODE_RE = re.compile(r"\b(?:0x[0-9a-fA-F]{4,16}|BugCheck\s+[0-9a-fA-Fx]+|APPCRASH|AppHangB1|StackHash|E\d{3,6})\b")


def classify_log_entry(name: str) -> str:
    text = str(name or "").lower()
    suffix = Path(text).suffix.lower()
    if suffix == ".dmp" or "dump" in text or "memory.dmp" in text:
        return "memory_dump"
    if suffix == ".evtx" or "event" in text or "事件" in text:
        return "windows_event_log"
    if any(marker in text for marker in STARTUP_MARKERS):
        return "startup_log"
    if any(marker in text for marker in DLOG_MARKERS):
        return "dlog"
    if any(marker in text for marker in ERROR_MARKERS):
        return "error_log"
    if suffix in {".log", ".txt", ".pml", ".csv"}:
        return "plain_log"
    return "log_artifact"


class LogPackageParserAgent:
    """Tool entry for safe log package metadata and manifest inspection."""

    schema_version = "debug_agent_system.tool.log_package_parse.v1"

    def parse(self, package: str | Path | dict[str, Any], *, max_entries: int = 200, max_preview_bytes: int = 8192, max_text_entries: int = 5) -> dict[str, Any]:
        if isinstance(package, dict):
            path_text = str(package.get("path") or package.get("relative_path") or "")
            name = str(package.get("name") or Path(path_text).name or package.get("file_key") or "")
            source = dict(package)
            declared_size = package.get("size") or package.get("bytes")
        else:
            path = Path(package)
            path_text = str(path)
            name = path.name
            source = {"path": path_text}
            declared_size = None
        file_path = Path(path_text) if path_text else Path(name)
        exists = bool(path_text and file_path.exists())
        is_file = file_path.is_file() if exists else False
        suffix = Path(name or path_text).suffix.lower()
        size = file_path.stat().st_size if is_file else _int_or_none(declared_size)
        archive_supported = suffix == ".zip" or (suffix in {".7z", ".rar"} and bool(shutil.which("bsdtar")))
        entries: list[dict[str, Any]] = []
        listing_error = ""
        archive_format = ""
        if suffix == ".zip" and is_file:
            archive_format = "zip"
            try:
                entries = self._zip_entries(file_path, max_entries=max_entries, max_preview_bytes=max_preview_bytes, max_text_entries=max_text_entries)
            except Exception as exc:  # noqa: BLE001 - degrade to metadata-only.
                listing_error = str(exc)
        elif suffix in {".7z", ".rar"} and is_file and archive_supported:
            archive_format = suffix.lstrip(".")
            try:
                entries = self._external_archive_entries(file_path, max_entries=max_entries)
            except Exception as exc:  # noqa: BLE001 - degrade to metadata-only.
                listing_error = str(exc)
        elif suffix in LOG_EXTS and (name or path_text):
            entry = {
                "name": name or file_path.name,
                "size": size,
                "extension": suffix,
                "role": classify_log_entry(name or path_text),
                "source": "single_file_metadata",
            }
            if suffix in TEXT_LOG_EXTS and is_file:
                preview = self._text_preview_from_file(file_path, max_preview_bytes=max_preview_bytes)
                entry.update(preview)
            entries = [entry]
        roles = sorted({str(item.get("role") or "") for item in entries if item.get("role")})
        text_hints = self._aggregate_text_hints(entries)
        text_preview_read = any(bool(item.get("text_preview_read")) for item in entries)
        return {
            "schema_version": self.schema_version,
            "type": "LogPackageParseResult",
            "name": name or file_path.name,
            "path": path_text,
            "extension": suffix,
            "exists": exists,
            "size": size,
            "status": "metadata_only" if not entries else ("text_hints" if text_preview_read else "manifest_only"),
            "archive_listing_supported": archive_supported,
            "archive_format": archive_format,
            "archive_extracted": False,
            "content_read": False,
            "text_preview_read": text_preview_read,
            "text_hints": text_hints,
            "executed": False,
            "mutated": False,
            "entry_count": len(entries),
            "entries_truncated": bool(len(entries) >= max_entries),
            "entries": entries,
            "detected_roles": roles,
            "has_dmp": "memory_dump" in roles,
            "has_evtx": "windows_event_log" in roles,
            "has_startup_log": "startup_log" in roles,
            "has_dlog": "dlog" in roles,
            "listing_error": listing_error,
            "source": source,
            "observability": {
                "agent_id": "TOOL-LOG-PACKAGE",
                "boundary": "bounded_text_hints" if text_preview_read else ("archive_manifest_only" if archive_supported else "metadata_only"),
            },
        }

    def _zip_entries(self, file_path: Path, *, max_entries: int, max_preview_bytes: int, max_text_entries: int) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        text_entries_read = 0
        with zipfile.ZipFile(file_path) as zf:
            for info in zf.infolist()[:max_entries]:
                if info.is_dir():
                    continue
                name = _repair_zip_name(info)
                entry = {
                    "name": name,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "extension": Path(name).suffix.lower(),
                    "role": classify_log_entry(name),
                    "source": "zip_central_directory",
                    "text_preview_read": False,
                }
                if Path(name).suffix.lower() in TEXT_LOG_EXTS and text_entries_read < max_text_entries:
                    try:
                        with zf.open(info) as f:
                            raw = f.read(max_preview_bytes)
                        entry.update(self._text_preview(raw, max_preview_bytes=max_preview_bytes))
                        text_entries_read += 1
                    except Exception as exc:  # noqa: BLE001 - keep manifest evidence.
                        entry["text_preview_error"] = str(exc)
                entries.append(entry)
        return entries

    def _external_archive_entries(self, file_path: Path, *, max_entries: int) -> list[dict[str, Any]]:
        command = shutil.which("bsdtar")
        if not command:
            return []
        proc = subprocess.run(
            [command, "-tf", str(file_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        entries: list[dict[str, Any]] = []
        for line in proc.stdout.splitlines()[:max_entries]:
            name = line.strip()
            if not name or name.endswith("/"):
                continue
            entries.append({
                "name": name,
                "size": None,
                "compressed_size": None,
                "extension": Path(name).suffix.lower(),
                "role": classify_log_entry(name),
                "source": "external_archive_manifest",
                "text_preview_read": False,
            })
        return entries

    def _text_preview_from_file(self, file_path: Path, *, max_preview_bytes: int) -> dict[str, Any]:
        with file_path.open("rb") as f:
            raw = f.read(max_preview_bytes)
        return self._text_preview(raw, max_preview_bytes=max_preview_bytes)

    def _text_preview(self, raw: bytes, *, max_preview_bytes: int) -> dict[str, Any]:
        text, encoding = _decode(raw)
        compact = " ".join(text.replace("\x00", " ").split())
        preview = compact[:1200]
        return {
            "text_preview_read": bool(raw),
            "preview_bytes": len(raw),
            "max_preview_bytes": max_preview_bytes,
            "preview_sha1_prefix": hashlib.sha1(raw).hexdigest()[:12] if raw else "",
            "encoding": encoding,
            "text_preview": preview,
            "text_hints": _text_hints(text),
        }

    def _aggregate_text_hints(self, entries: list[dict[str, Any]]) -> dict[str, list[str]]:
        error_codes: list[str] = []
        error_lines: list[str] = []
        phase_hints: list[str] = []
        preview_sources: list[str] = []
        for entry in entries:
            hints = entry.get("text_hints") if isinstance(entry.get("text_hints"), dict) else {}
            error_codes.extend(str(x) for x in hints.get("error_codes") or [])
            error_lines.extend(str(x) for x in hints.get("error_lines") or [])
            phase_hints.extend(str(x) for x in hints.get("phase_hints") or [])
            if entry.get("text_preview_read"):
                preview_sources.append(str(entry.get("name") or ""))
        return {
            "error_codes": _unique(error_codes, limit=20),
            "error_lines": _unique(error_lines, limit=8),
            "phase_hints": _unique(phase_hints, limit=12),
            "preview_sources": _unique(preview_sources, limit=20),
        }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decode(raw: bytes) -> tuple[str, str]:
    if not raw:
        return "", ""
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def _repair_zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    # If the UTF-8 flag is unset, Python decodes names as cp437.  Many Feishu
    # zips in this corpus actually store GBK/GB18030 bytes, which renders as
    # mojibake like "╬╩╠Γ".  Re-decode only when this improves Chinese names.
    if info.flag_bits & 0x800:
        return name
    try:
        repaired = name.encode("cp437").decode("gb18030")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    if re.search(r"[\u4e00-\u9fff]", repaired):
        return repaired
    return name


def _text_hints(text: str) -> dict[str, list[str]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    error_lines: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in ERROR_MARKERS) or ERROR_CODE_RE.search(line):
            error_lines.append(" ".join(line.split())[:240])
        if len(error_lines) >= 8:
            break
    compact = "\n".join(lines)
    phases: list[str] = []
    lowered_compact = compact.lower()
    for phase, markers in PHASE_MARKERS.items():
        if any(marker.lower() in lowered_compact for marker in markers):
            phases.append(phase)
    return {
        "error_codes": _unique(ERROR_CODE_RE.findall(compact), limit=20),
        "error_lines": _unique(error_lines, limit=8),
        "phase_hints": phases,
    }


def _unique(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


__all__ = ["LogPackageParserAgent", "classify_log_entry"]
