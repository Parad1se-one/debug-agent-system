"""Safe `.proj` evidence parser.

The parser provides a bounded text preview for project/config files.  It does
not execute, import, or mutate project files.  AOI `.proj` files are often tar
containers; for those, the parser reads only the tar manifest plus bounded
preview of whitelisted small text entries.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

TEXT_ENTRY_EXTS = {".json", ".csv", ".txt", ".ini", ".toml", ".cfg", ".xml"}
SKIP_ENTRY_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tpg", ".detail", ".tar", ".zip", ".rar", ".7z"}
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
VERSION_RE = re.compile(r"(?<![\d.])(?:v|V)?\d+\.\d+(?:\.\d+){0,3}(?![\d.])")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
CONFIG_TERMS = ("Camera", "camera", "相机", "光源", "Light", "PLC", "IO", "Motion", "Axis", "Trigger", "Exposure", "Gain")


class ProjParserAgent:
    """Tool entry for parsing AOI project/program files as evidence."""

    schema_version = "debug_agent_system.tool.proj_parse.v1"

    def parse(self, path: str | Path, *, max_bytes: int = 65536, max_entries: int = 200, max_text_entries: int = 8) -> dict[str, Any]:
        file_path = Path(path)
        raw = b""
        exists = file_path.exists()
        is_file = file_path.is_file() if exists else False
        if is_file:
            with file_path.open("rb") as f:
                raw = f.read(max_bytes)
        text, encoding = self._decode(raw)
        is_tar = bool(is_file and tarfile.is_tarfile(file_path))
        archive_entries: list[dict[str, Any]] = []
        archive_error = ""
        archive_text = ""
        if is_tar:
            try:
                archive_entries, archive_text = self._tar_manifest_and_text(file_path, max_entries=max_entries, max_bytes=max_bytes, max_text_entries=max_text_entries)
            except Exception as exc:  # noqa: BLE001 - degrade to bounded raw preview.
                archive_error = str(exc)
        hint_text = "\n".join([text, archive_text])
        return {
            "schema_version": self.schema_version,
            "type": "ProjParseResult",
            "path": str(file_path),
            "name": file_path.name,
            "exists": exists,
            "size": file_path.stat().st_size if is_file else None,
            "sha1_prefix": hashlib.sha1(raw).hexdigest()[:12] if raw else "",
            "content_read": bool(raw),
            "max_bytes": max_bytes,
            "truncated": bool(is_file and file_path.stat().st_size > len(raw)),
            "encoding": encoding,
            "text_preview": self._preview(hint_text or text),
            "key_hints": self._hints(hint_text, archive_entries),
            "archive_format": "tar" if is_tar else "",
            "archive_manifest_read": bool(archive_entries),
            "archive_extracted": False,
            "archive_error": archive_error,
            "entries": archive_entries,
            "entries_truncated": bool(len(archive_entries) >= max_entries),
            "text_entry_preview_read": any(bool(item.get("text_preview_read")) for item in archive_entries),
            "executed": False,
            "mutated": False,
            "observability": {
                "agent_id": "TOOL-PROJ",
                "boundary": "tar_manifest_bounded_text_preview" if is_tar else "bounded_text_preview",
            },
        }

    def _decode(self, raw: bytes) -> tuple[str, str]:
        if not raw:
            return "", ""
        for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace"), "utf-8-replace"

    def _preview(self, text: str, limit: int = 1200) -> str:
        compact = " ".join(str(text or "").replace("\x00", " ").split())
        return compact[:limit]

    def _tar_manifest_and_text(self, file_path: Path, *, max_entries: int, max_bytes: int, max_text_entries: int) -> tuple[list[dict[str, Any]], str]:
        entries: list[dict[str, Any]] = []
        texts: list[str] = []
        text_reads = 0
        with tarfile.open(file_path, "r:*") as tf:
            for member in tf.getmembers()[:max_entries]:
                suffix = Path(member.name).suffix.lower()
                role = _entry_role(member.name)
                entry = {
                    "name": member.name,
                    "size": member.size,
                    "extension": suffix,
                    "role": role,
                    "is_file": member.isfile(),
                    "text_preview_read": False,
                }
                if member.isfile() and suffix in TEXT_ENTRY_EXTS and member.size <= max_bytes and text_reads < max_text_entries:
                    extracted = tf.extractfile(member)
                    raw = extracted.read(max_bytes) if extracted else b""
                    text, encoding = self._decode(raw)
                    entry.update({
                        "text_preview_read": bool(raw),
                        "encoding": encoding,
                        "preview_bytes": len(raw),
                        "preview_sha1_prefix": hashlib.sha1(raw).hexdigest()[:12] if raw else "",
                        "text_preview": self._preview(text),
                    })
                    texts.append(text)
                    text_reads += 1
                elif member.isfile() and suffix in SKIP_ENTRY_EXTS:
                    entry["skip_reason"] = "binary_or_large_artifact_metadata_only"
                entries.append(entry)
        return entries, "\n".join(texts)

    def _hints(self, text: str, entries: list[dict[str, Any]] | None = None) -> dict[str, list[str] | bool]:
        compact = str(text or "")
        ips = sorted(set(IP_RE.findall(compact)))[:20]
        json_objects = _json_objects(compact)
        project_names: list[str] = []
        app_versions: list[str] = []
        sdk_versions: list[str] = []
        schema_versions: list[str] = []
        pcb_types: list[str] = []
        device_names: list[str] = []
        manufacturers: list[str] = []
        model_types: list[str] = []
        for obj in json_objects:
            project_names.extend(_string_values(obj, ("name", "project_name", "program_name", "recipe_name")))
            app_versions.extend(_string_values(obj, ("app_version", "application_version", "software_version")))
            sdk_versions.extend(_string_values(obj, ("sdk_version",)))
            schema_versions.extend(_string_values(obj, ("version", "schema_version")))
            pcb_types.extend(_string_values(obj, ("pcb_type", "board_type")))
            device_names.extend(_string_values(obj, ("device", "device_name", "machine", "machine_name")))
            manufacturers.extend(_string_values(obj, ("manufacturer",)))
            model_types.extend(_model_types(obj))
        # Tar-based .proj files contain component coordinate tables where values
        # like "101.35" are not software versions.  For structured project
        # archives, trust JSON version fields; use broad regex only for plain
        # text project/config files without JSON metadata.
        regex_versions = [] if json_objects else sorted({value for value in VERSION_RE.findall(compact) if value not in ips})[:30]
        file_roles = sorted({str(entry.get("role") or "") for entry in entries or [] if entry.get("role")})
        entry_names = [str(entry.get("name") or "") for entry in entries or []]
        return {
            "versions": _unique([*app_versions, *sdk_versions, *schema_versions, *regex_versions], limit=30),
            "app_versions": _unique(app_versions, limit=12),
            "sdk_versions": _unique(sdk_versions, limit=12),
            "schema_versions": _unique(schema_versions, limit=12),
            "ip_addresses": ips,
            "project_names": _unique(project_names, limit=12),
            "pcb_types": _unique(pcb_types, limit=12),
            "device_names": _unique(device_names, limit=12),
            "manufacturers": _unique(manufacturers, limit=12),
            "model_types": _unique(model_types, limit=30),
            "revision_ids": _unique(UUID_RE.findall(compact), limit=20),
            "camera_terms": [term for term in CONFIG_TERMS if term in compact],
            "file_roles": file_roles,
            "has_csv": any(Path(name).suffix.lower() == ".csv" for name in entry_names),
            "has_board_images": any(_entry_role(name) == "board_image" for name in entry_names),
            "has_detail": any(Path(name).suffix.lower() == ".detail" for name in entry_names),
        }


def _entry_role(name: str) -> str:
    lower = str(name or "").lower()
    suffix = Path(lower).suffix.lower()
    if Path(lower).name == "meta.json":
        return "project_meta"
    if lower.startswith("rev.") and suffix == ".json":
        return "revision_meta"
    if suffix == ".csv":
        return "component_table"
    if "image_pack" in lower:
        return "image_pack"
    if "image_rgb" in lower or "image_white" in lower or suffix in {".tpg", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return "board_image"
    if suffix == ".detail":
        return "inspection_detail"
    if suffix in TEXT_ENTRY_EXTS:
        return "text_config"
    return "artifact"


def _json_objects(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        raw = line.strip()
        if not raw.startswith("{") or not raw.endswith("}"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out[:20]


def _string_values(obj: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _model_types(obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    models = obj.get("models")
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("type"), str):
                out.append(model["type"])
    return out


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


__all__ = ["ProjParserAgent"]
