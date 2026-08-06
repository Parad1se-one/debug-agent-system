"""Safe document evidence parser.

This parser extracts bounded metadata from PDF/Office-like documents.  It never
executes macros, evaluates formulas, renders pages, or performs OCR.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

DOCUMENT_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
OOXML_EXTS = {".docx", ".xlsx", ".pptx"}
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
PDF_VERSION_RE = re.compile(rb"%PDF-(\d\.\d)")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
OOXML_TEXT_ENTRIES = (
    "docProps/core.xml",
    "docProps/app.xml",
    "word/document.xml",
    "xl/sharedStrings.xml",
    "xl/workbook.xml",
    "ppt/slides/slide1.xml",
)


class DocumentParserAgent:
    """Tool entry for bounded document metadata parsing."""

    schema_version = "debug_agent_system.tool.document_parse.v1"

    def parse(self, document: str | Path | dict[str, Any], *, max_preview_bytes: int = 65536) -> dict[str, Any]:
        if isinstance(document, dict):
            path_text = str(document.get("path") or "")
            name = str(document.get("name") or document.get("file_key") or Path(path_text).name)
            size = document.get("size")
            source = dict(document)
        else:
            path = Path(document)
            path_text = str(path)
            name = path.name
            size = path.stat().st_size if path.exists() and path.is_file() else None
            source = {"path": path_text}
        ext = Path(name).suffix.lower()
        if ext not in DOCUMENT_EXTS:
            return self._base(name, path_text, ext, size, source, status="metadata_only", document_format="")
        if ext == ".pdf":
            return self._parse_pdf(name, path_text, ext, size, source, max_preview_bytes=max_preview_bytes)
        if ext in OOXML_EXTS:
            return self._parse_ooxml(name, path_text, ext, size, source, max_preview_bytes=max_preview_bytes)
        return self._parse_ole(name, path_text, ext, size, source, max_preview_bytes=max_preview_bytes)

    def _base(self, name: str, path_text: str, ext: str, size: Any, source: dict[str, Any], **extra: Any) -> dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "type": "DocumentParseResult",
            "name": name,
            "path": path_text,
            "extension": ext,
            "size": size,
            "document_format": extra.pop("document_format", self._format_from_ext(ext)),
            "status": extra.pop("status", "metadata_only"),
            "header_read": False,
            "text_preview_read": False,
            "text_preview": "",
            "text_preview_truncated": False,
            "entries": [],
            "archive_manifest_read": False,
            "archive_extracted": False,
            "macros_executed": False,
            "formulas_evaluated": False,
            "ocr_performed": False,
            "source": source,
            "observability": {"agent_id": "TOOL-DOCUMENT", "boundary": "metadata_only"},
        }
        out.update(extra)
        return out

    def _parse_pdf(self, name: str, path_text: str, ext: str, size: Any, source: dict[str, Any], *, max_preview_bytes: int) -> dict[str, Any]:
        raw = self._read_bytes(path_text, max_preview_bytes=max_preview_bytes)
        version_match = PDF_VERSION_RE.search(raw)
        text_preview = self._printable_preview(raw)
        out = self._base(
            name,
            path_text,
            ext,
            size,
            source,
            status="bounded_metadata" if raw else "metadata_only",
            document_format="pdf",
            header_read=bool(raw),
            text_preview_read=bool(text_preview),
            text_preview=text_preview,
            text_preview_truncated=self._is_truncated(path_text, raw),
            pdf_version=version_match.group(1).decode("ascii") if version_match else "",
            page_count_hint=len(re.findall(rb"/Type\s*/Page\b", raw)),
            observability={"agent_id": "TOOL-DOCUMENT", "boundary": "bounded_pdf_bytes" if raw else "metadata_only"},
        )
        return out

    def _parse_ooxml(self, name: str, path_text: str, ext: str, size: Any, source: dict[str, Any], *, max_preview_bytes: int) -> dict[str, Any]:
        path = Path(path_text)
        entries: list[dict[str, Any]] = []
        previews: list[str] = []
        if path_text and path.exists() and path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist()[:200]:
                    entry = {"name": info.filename, "size": info.file_size, "source": "zip_central_directory", "text_preview_read": False}
                    if info.filename in OOXML_TEXT_ENTRIES and info.file_size <= max_preview_bytes * 4:
                        raw = zf.read(info.filename)[:max_preview_bytes]
                        preview = self._xml_preview(raw)
                        if preview:
                            entry["text_preview_read"] = True
                            entry["text_preview"] = preview
                            previews.append(preview)
                    entries.append(entry)
        preview_text = " ".join(previews)[:2000]
        return self._base(
            name,
            path_text,
            ext,
            size,
            source,
            status="bounded_ooxml_metadata" if entries else "metadata_only",
            document_format=self._format_from_ext(ext),
            header_read=bool(entries),
            text_preview_read=bool(preview_text),
            text_preview=preview_text,
            entries=entries[:80],
            archive_manifest_read=bool(entries),
            observability={"agent_id": "TOOL-DOCUMENT", "boundary": "ooxml_manifest_bounded_text" if entries else "metadata_only"},
        )

    def _parse_ole(self, name: str, path_text: str, ext: str, size: Any, source: dict[str, Any], *, max_preview_bytes: int) -> dict[str, Any]:
        raw = self._read_bytes(path_text, max_preview_bytes=min(max_preview_bytes, 8192))
        return self._base(
            name,
            path_text,
            ext,
            size,
            source,
            status="ole_header_metadata" if raw.startswith(OLE_MAGIC) else "metadata_only",
            document_format=self._format_from_ext(ext),
            header_read=bool(raw),
            ole_compound_file=raw.startswith(OLE_MAGIC),
            text_preview_read=False,
            observability={"agent_id": "TOOL-DOCUMENT", "boundary": "ole_header_metadata" if raw else "metadata_only"},
        )

    def _read_bytes(self, path_text: str, *, max_preview_bytes: int) -> bytes:
        path = Path(path_text)
        if not path_text or not path.exists() or not path.is_file():
            return b""
        try:
            return path.read_bytes()[: max(1, max_preview_bytes)]
        except OSError:
            return b""

    def _is_truncated(self, path_text: str, raw: bytes) -> bool:
        path = Path(path_text)
        try:
            return path.exists() and path.stat().st_size > len(raw)
        except OSError:
            return False

    def _printable_preview(self, raw: bytes) -> str:
        text = raw.decode("latin-1", errors="ignore")
        chunks = re.findall(r"[\x20-\x7e\u4e00-\u9fff]{4,}", text)
        return SPACE_RE.sub(" ", " ".join(chunks)).strip()[:2000]

    def _xml_preview(self, raw: bytes) -> str:
        text = raw.decode("utf-8", errors="replace")
        text = TAG_RE.sub(" ", text)
        return SPACE_RE.sub(" ", text).strip()[:2000]

    def _format_from_ext(self, ext: str) -> str:
        return {
            ".pdf": "pdf",
            ".doc": "word_ole",
            ".docx": "word_ooxml",
            ".xls": "excel_ole",
            ".xlsx": "excel_ooxml",
            ".ppt": "powerpoint_ole",
            ".pptx": "powerpoint_ooxml",
        }.get(ext, "")


__all__ = ["DocumentParserAgent"]
