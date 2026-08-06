"""Safe image evidence parser.

Reads only bounded image headers to expose format and dimensions for review
evidence.  It never decodes pixels, runs OCR, extracts archives, or calls vision
models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _u16be(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "big")


def _u16le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little")


def _u24le(data: bytes, offset: int) -> int:
    raw = data[offset : offset + 3]
    return raw[0] | (raw[1] << 8) | (raw[2] << 16) if len(raw) == 3 else 0


class ImageParserAgent:
    """Tool entry for safe image header metadata parsing."""

    schema_version = "debug_agent_system.tool.image_parse.v1"

    def parse(self, image: str | Path | dict[str, Any], *, max_header_bytes: int = 65536) -> dict[str, Any]:
        if isinstance(image, dict):
            path_text = str(image.get("path") or "")
            name = str(image.get("name") or image.get("file_key") or Path(path_text).name)
            size = image.get("size")
            source = dict(image)
        else:
            path = Path(image)
            path_text = str(path)
            name = path.name
            size = path.stat().st_size if path.exists() and path.is_file() else None
            source = {"path": path_text}
        ext = Path(name).suffix.lower()
        header = self._read_header(path_text, max_header_bytes=max_header_bytes) if ext in IMAGE_EXTS else b""
        meta = self._image_meta(header)
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        megapixels = round((width * height) / 1_000_000, 4) if width and height else 0.0
        return {
            "schema_version": self.schema_version,
            "type": "ImageParseResult",
            "name": name,
            "path": path_text,
            "extension": ext,
            "size": size,
            "image_format": meta.get("format") or self._format_from_ext(ext),
            "width": width or None,
            "height": height or None,
            "megapixels": megapixels,
            "aspect_ratio": round(width / height, 4) if width and height else None,
            "header_read": bool(header),
            "pixels_read": False,
            "ocr_performed": False,
            "content_read": False,
            "archive_extracted": False,
            "status": "header_metadata" if width and height else "metadata_only",
            "source": source,
            "observability": {
                "agent_id": "TOOL-IMAGE",
                "boundary": "bounded_header_metadata" if header else "metadata_only",
            },
        }

    def _read_header(self, path_text: str, *, max_header_bytes: int) -> bytes:
        path = Path(path_text)
        if not path_text or not path.exists() or not path.is_file():
            return b""
        try:
            return path.read_bytes()[: max(32, max_header_bytes)]
        except OSError:
            return b""

    def _image_meta(self, data: bytes) -> dict[str, Any]:
        if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
            return {"format": "png", "width": int.from_bytes(data[16:20], "big"), "height": int.from_bytes(data[20:24], "big")}
        if len(data) >= 10 and (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
            return {"format": "gif", "width": _u16le(data, 6), "height": _u16le(data, 8)}
        if len(data) >= 30 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return self._webp_meta(data)
        if len(data) >= 4 and data.startswith(b"\xff\xd8"):
            return self._jpeg_meta(data)
        if len(data) >= 26 and data.startswith(b"BM"):
            width = int.from_bytes(data[18:22], "little", signed=True)
            height = abs(int.from_bytes(data[22:26], "little", signed=True))
            return {"format": "bmp", "width": abs(width), "height": height}
        return {}

    def _jpeg_meta(self, data: bytes) -> dict[str, Any]:
        i = 2
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while i + 8 < len(data):
            while i < len(data) and data[i] != 0xFF:
                i += 1
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]
            i += 1
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > len(data):
                break
            length = _u16be(data, i)
            if length < 2 or i + length > len(data):
                break
            segment = i + 2
            if marker in sof_markers and segment + 5 <= len(data):
                return {"format": "jpeg", "width": _u16be(data, segment + 3), "height": _u16be(data, segment + 1)}
            i += length
        return {"format": "jpeg"}

    def _webp_meta(self, data: bytes) -> dict[str, Any]:
        chunk = data[12:16]
        payload = 20
        if chunk == b"VP8X" and len(data) >= 30:
            return {"format": "webp", "width": _u24le(data, 24) + 1, "height": _u24le(data, 27) + 1}
        if chunk == b"VP8 " and len(data) >= payload + 10 and data[payload + 3 : payload + 6] == b"\x9d\x01\x2a":
            return {"format": "webp", "width": _u16le(data, payload + 6) & 0x3FFF, "height": _u16le(data, payload + 8) & 0x3FFF}
        if chunk == b"VP8L" and len(data) >= payload + 5 and data[payload] == 0x2F:
            bits = int.from_bytes(data[payload + 1 : payload + 5], "little")
            return {"format": "webp", "width": (bits & 0x3FFF) + 1, "height": ((bits >> 14) & 0x3FFF) + 1}
        return {"format": "webp"}

    def _format_from_ext(self, ext: str) -> str:
        return {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".gif": "gif", ".bmp": "bmp"}.get(ext, "")


__all__ = ["ImageParserAgent"]
