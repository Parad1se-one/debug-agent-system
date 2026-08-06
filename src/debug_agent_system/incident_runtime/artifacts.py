"""Immutable artifact intake and safe archive member access."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable, Iterator
import zipfile

from .contracts import ArtifactManifest, IncidentCase
from .scope import IncidentScope, name_matches_scope, parse_log_timestamp

_JIRA_KEY = re.compile(r"\b[A-Z][A-Z0-9_]+-\d+\b")
_ARCHIVES = {".zip", ".tar", ".tgz", ".gz", ".7z", ".rar"}


@dataclass(slots=True)
class ArtifactLimits:
    max_package_bytes: int = 512 * 1024 * 1024
    max_member_bytes: int = 32 * 1024 * 1024
    # Kernel dumps are sparse high-value evidence that routinely exceed the
    # generic member cap.  They are parsed path-backed with bounded stream
    # reads, so a large allowance does not imply loading the file into memory.
    max_dump_member_bytes: int = 8 * 1024 * 1024 * 1024
    max_total_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_members: int = 5000
    max_nesting: int = 3
    max_compression_ratio: float = 200.0


@dataclass(slots=True)
class ArtifactContent:
    manifest: ArtifactManifest
    data: bytes


class ArtifactIntake:
    schema_version = "debug_agent_system.incident_artifact_manifest.v1"

    def __init__(self, limits: ArtifactLimits | None = None) -> None:
        self.limits = limits or ArtifactLimits()
        self._temp_files: list[str] = []

    def cleanup(self) -> None:
        """Delete path-backed temp files materialized for large dumps."""

        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files = []

    def create_case(
        self,
        query: str,
        resources: Iterable[dict[str, Any]],
        *,
        log_summary: dict[str, Any] | None = None,
    ) -> tuple[IncidentCase, list[ArtifactContent], list[dict[str, Any]]]:
        normalized = [dict(item) for item in resources if isinstance(item, dict)]
        digest = hashlib.sha256()
        digest.update(str(query).encode("utf-8", errors="replace"))
        for item in normalized:
            digest.update(str(item.get("resource_id") or item.get("path") or item.get("name") or "").encode())
        case_id = f"incident:{digest.hexdigest()[:16]}"
        manifests: list[ArtifactManifest] = []
        contents: list[ArtifactContent] = []
        exclusions: list[dict[str, Any]] = []
        jira_key = ""
        jira_status = ""
        affected_version = ""
        device = ""
        station = ""
        reproduction = ""

        for index, resource in enumerate(normalized):
            resource_id = str(resource.get("resource_id") or f"resource:{index + 1}")
            name = str(resource.get("name") or Path(str(resource.get("path") or "")).name or resource_id)
            kind = str(resource.get("kind") or "auto")
            path_text = str(resource.get("path") or "")
            text = str(resource.get("text") or "")
            metadata = dict(resource.get("metadata") or {})
            manifest = ArtifactManifest(
                artifact_id=f"artifact:{hashlib.sha256(resource_id.encode()).hexdigest()[:16]}",
                resource_id=resource_id,
                name=name,
                kind=kind,
                path=path_text,
                url=str(resource.get("url") or ""),
                mime=str(resource.get("mime") or mimetypes.guess_type(name)[0] or ""),
                size=_int_or_none(resource.get("size")),
                sha256=str(resource.get("sha256") or ""),
                metadata=metadata,
            )
            data = b""
            if text:
                data = text.encode("utf-8")
            elif path_text:
                path = Path(path_text)
                if not path.exists() or not path.is_file():
                    manifest.status = "missing"
                    manifest.parser_state = "not_started"
                    exclusions.append({"artifact_id": manifest.artifact_id, "reason": "file_missing", "path": path_text})
                elif path.stat().st_size > self.limits.max_package_bytes:
                    manifest.status = "rejected"
                    manifest.parser_state = "not_started"
                    manifest.safety_flags.append("package_size_limit")
                    exclusions.append({"artifact_id": manifest.artifact_id, "reason": "package_size_limit"})
                elif _suffix(name) in _ARCHIVES:
                    # Archive roots stay path-backed.  Reading a multi-hundred MB
                    # package into one bytes object defeats bounded member access.
                    manifest.sha256 = _sha256_file(path)
                    manifest.size = path.stat().st_size
                    manifest.parser_state = "available"
                    manifest.metadata["path_backed"] = True
                    contents.append(ArtifactContent(manifest=manifest, data=b""))
                else:
                    data = path.read_bytes()
            if data:
                actual_hash = hashlib.sha256(data).hexdigest()
                if manifest.sha256 and manifest.sha256.lower() != actual_hash:
                    manifest.safety_flags.append("declared_hash_mismatch")
                manifest.sha256 = actual_hash
                manifest.size = len(data)
                manifest.parser_state = "available"
                contents.append(ArtifactContent(manifest=manifest, data=data))
            manifests.append(manifest)

            combined = " ".join([name, text, str(metadata), manifest.url])
            match = _JIRA_KEY.search(combined)
            if match and not jira_key:
                jira_key = match.group(0)
            if kind == "jira":
                jira_status = str(metadata.get("status") or jira_status)
                affected_version = str(metadata.get("affected_version") or metadata.get("version") or affected_version)
                device = str(metadata.get("device") or device)
                station = str(metadata.get("station") or metadata.get("site") or station)
                reproduction = str(metadata.get("reproduction") or reproduction)

        if log_summary:
            raw = _json_bytes(log_summary)
            resource_id = "resource:log-summary"
            manifest = ArtifactManifest(
                artifact_id=f"artifact:{hashlib.sha256(resource_id.encode()).hexdigest()[:16]}",
                resource_id=resource_id,
                name="log_summary.json",
                kind="log_package",
                mime="application/json",
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                parser_state="available",
                metadata={"synthetic_from_input_contract": True},
            )
            manifests.append(manifest)
            contents.append(ArtifactContent(manifest=manifest, data=raw))

        case = IncidentCase(
            case_id=case_id,
            query=query,
            jira_key=jira_key,
            status=jira_status,
            affected_version=affected_version,
            device=device,
            station=station,
            reproduction=reproduction,
            artifacts=manifests,
            metadata={"artifact_manifest_schema": self.schema_version},
        )
        return case, contents, exclusions

    def iter_members(
        self,
        content: ArtifactContent,
        *,
        nesting: int = 0,
    ) -> Iterator[ArtifactContent]:
        """Yield source plus safely materialized archive members in memory."""

        yield content
        suffix = _suffix(content.manifest.name)
        if suffix not in _ARCHIVES:
            return
        if nesting >= self.limits.max_nesting:
            yield self._rejected_child(
                content,
                "<nested-archive>",
                self.limits.max_nesting,
                "archive_nesting_limit",
            )
            return
        if suffix == ".zip":
            yield from self._zip_members(content, nesting=nesting)
        elif suffix in {".tar", ".tgz", ".gz"}:
            yield from self._tar_members(content, nesting=nesting)
        elif suffix in {".7z", ".rar"}:
            yield from self._external_members(content, nesting=nesting)

    def iter_scoped_members(
        self,
        content: ArtifactContent,
        scope: IncidentScope,
    ) -> Iterator[ArtifactContent]:
        """Yield manifests plus bounded time windows for a scoped archive.

        ZIP is handled with true member streaming.  Other formats retain the
        existing bounded fallback until they have an equivalent seekable reader.
        """

        if not scope.has_time_scope or _suffix(content.manifest.name) != ".zip":
            yield from self.iter_members(content)
            return
        yield content
        try:
            with self._open_zip(content) as archive:
                infos = archive.infolist()
                materialized_total = 0
                if len(infos) > self.limits.max_members:
                    yield self._rejected_child(
                        content,
                        "<remaining-members>",
                        self.limits.max_members,
                        "archive_member_count_limit",
                    )
                safe_infos = [
                    (index, info)
                    for index, info in enumerate(infos[: self.limits.max_members])
                    if not info.is_dir() and _safe_member_name(info.filename)
                ]
                dated_text = [
                    (index, info)
                    for index, info in safe_infos
                    if _is_text_name(info.filename) and name_matches_scope(info.filename, scope)
                ]
                # Prefer exact dated files.  If the archive uses undated names,
                # stream text candidates and let timestamps decide.
                selected_text = {
                    info.filename
                    for _, info in (dated_text or [
                        pair for pair in safe_infos if _is_text_name(pair[1].filename)
                    ])
                }
                for index, info in enumerate(infos[: self.limits.max_members]):
                    if info.is_dir():
                        continue
                    if not _safe_member_name(info.filename):
                        yield self._rejected_child(
                            content, info.filename, index, "unsafe_archive_member_path"
                        )
                        continue
                    ratio = info.file_size / max(info.compress_size, 1)
                    if ratio > self.limits.max_compression_ratio:
                        yield self._rejected_child(
                            content, info.filename, index, "compression_ratio_limit"
                        )
                        continue
                    if info.filename in selected_text:
                        yield self._zip_time_window_child(
                            content, archive, info, index, scope
                        )
                        continue
                    suffix = _suffix(info.filename)
                    bypass_reason = ""
                    if suffix == ".evtx":
                        # EVTX carries record timestamps internally; filenames and
                        # ZIP timestamps are not reliable selectors.
                        bypass_reason = "internally_timestamped_binary"
                    elif suffix in {".dmp", ".mdmp"}:
                        # Crash dumps are sparse, high-value evidence and commonly
                        # use UUID filenames.  Their header timestamp is parsed later.
                        bypass_reason = "high_value_crash_artifact"
                    elif _is_static_environment_name(info.filename):
                        # Version/environment snapshots remain relevant to every
                        # reference instant and should not be mistaken for logs.
                        bypass_reason = "static_environment_snapshot"
                    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} and name_matches_scope(info.filename, scope):
                        bypass_reason = "dated_visual_evidence"
                    if bypass_reason and info.file_size <= (
                        self.limits.max_dump_member_bytes
                        if suffix in {".dmp", ".mdmp"}
                        else self.limits.max_member_bytes
                    ):
                        is_dump_bypass = suffix in {".dmp", ".mdmp"}
                        # Path-backed dumps are streamed to disk and parsed with
                        # bounded reads; their uncompressed size is not charged
                        # against the in-memory materialization budget.
                        if (
                            not is_dump_bypass
                            and materialized_total + info.file_size > self.limits.max_total_uncompressed_bytes
                        ):
                            yield self._rejected_child(
                                content, info.filename, index, "archive_total_uncompressed_limit"
                            )
                            continue
                        if is_dump_bypass and info.file_size > self.limits.max_member_bytes:
                            child = self._zip_dump_path_child(
                                content, archive, info, index
                            )
                            child.manifest.metadata.update({
                                "scope_bypass_reason": bypass_reason,
                                "time_scope": scope.to_dict(),
                            })
                            yield child
                            continue
                        data = archive.read(info)
                        materialized_total += len(data)
                        child = self._child(content, info.filename, data, index)
                        child.manifest.metadata.update({
                            "scope_bypass_reason": bypass_reason,
                            "time_scope": scope.to_dict(),
                            "source_member_crc32": f"{info.CRC:08x}",
                            "source_member_size": info.file_size,
                            "source_member_compressed_size": info.compress_size,
                        })
                        yield child
                        continue
                    yield self._metadata_child(
                        content,
                        info.filename,
                        index,
                        size=info.file_size,
                        metadata={
                            "scope_skipped": True,
                            "crc32": f"{info.CRC:08x}",
                            "compressed_size": info.compress_size,
                        },
                    )
        except (OSError, zipfile.BadZipFile):
            yield self._rejected_child(content, "<archive>", 0, "archive_parse_failed")

    @staticmethod
    def _open_zip(content: ArtifactContent) -> zipfile.ZipFile:
        if content.manifest.path and Path(content.manifest.path).is_file():
            return zipfile.ZipFile(content.manifest.path)
        return zipfile.ZipFile(io.BytesIO(content.data))

    def _zip_time_window_child(
        self,
        parent: ArtifactContent,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        index: int,
        scope: IncidentScope,
    ) -> ArtifactContent:
        lines: list[str] = []
        source_lines: list[int] = []
        matched_windows: set[int] = set()
        last_timestamp = None
        latest_end = max(item.end() for item in scope.reference_windows)
        scanned_bytes = 0
        extracted_bytes = 0
        encoding = "utf-8-replace"
        with archive.open(info, "r") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                scanned_bytes += len(raw_line)
                if scanned_bytes > self.limits.max_total_uncompressed_bytes:
                    break
                line = _decode_stream_line(raw_line)
                timestamp = parse_log_timestamp(line)
                if timestamp is not None:
                    last_timestamp = timestamp
                    if timestamp > latest_end and matched_windows:
                        break
                effective = timestamp or last_timestamp
                window_indexes = [
                    window_index
                    for window_index, window in enumerate(scope.reference_windows)
                    if effective is not None and window.contains(effective)
                ]
                if not window_indexes:
                    continue
                matched_windows.update(window_indexes)
                lines.append(line.rstrip("\r\n"))
                source_lines.append(line_no)
                extracted_bytes += len(lines[-1].encode("utf-8", errors="replace")) + 1
                if extracted_bytes > self.limits.max_member_bytes:
                    break
        data = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        identity = hashlib.sha256(
            f"{parent.manifest.sha256}:{info.filename}:{info.CRC}:{scope.to_dict()}".encode()
        ).hexdigest()
        manifest = ArtifactManifest(
            artifact_id=f"artifact:{identity[:16]}:{index}",
            resource_id=parent.manifest.resource_id,
            name=PurePosixPath(info.filename).name,
            kind=_kind_for_name(info.filename),
            mime=mimetypes.guess_type(info.filename)[0] or "text/plain",
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            parent_artifact_id=parent.manifest.artifact_id,
            archive_member=info.filename,
            parser_state="available" if data else "time_window_empty",
            metadata={
                "archive_ancestry": [parent.manifest.artifact_id],
                "derived_by": "query_time_window_stream",
                "source_member_crc32": f"{info.CRC:08x}",
                "source_member_size": info.file_size,
                "source_member_compressed_size": info.compress_size,
                "source_lines": source_lines,
                "source_line_start": source_lines[0] if source_lines else None,
                "source_line_end": source_lines[-1] if source_lines else None,
                "matched_reference_window_indexes": sorted(matched_windows),
                "scanned_uncompressed_bytes": scanned_bytes,
                "encoding": encoding,
                "time_scope": scope.to_dict(),
            },
        )
        return ArtifactContent(manifest=manifest, data=data)

    def _zip_members(self, content: ArtifactContent, *, nesting: int) -> Iterator[ArtifactContent]:
        total = 0
        try:
            with self._open_zip(content) as archive:
                infos = archive.infolist()
                if len(infos) > self.limits.max_members:
                    yield self._rejected_child(
                        content,
                        "<remaining-members>",
                        self.limits.max_members,
                        "archive_member_count_limit",
                    )
                for index, info in enumerate(infos[: self.limits.max_members]):
                    if info.is_dir():
                        continue
                    if not _safe_member_name(info.filename):
                        yield self._rejected_child(
                            content, info.filename, index, "unsafe_archive_member_path"
                        )
                        continue
                    is_dump = _suffix(info.filename) in {".dmp", ".mdmp"}
                    # Path-backed kernel dumps are streamed to a temp file and
                    # parsed with bounded reads, so their uncompressed size does
                    # not consume the in-memory materialization budget.
                    if not is_dump:
                        total += max(0, info.file_size)
                    ratio = info.file_size / max(info.compress_size, 1)
                    member_limit = (
                        self.limits.max_dump_member_bytes
                        if is_dump
                        else self.limits.max_member_bytes
                    )
                    if (
                        info.file_size > member_limit
                        or (total > self.limits.max_total_uncompressed_bytes and not is_dump)
                        or ratio > self.limits.max_compression_ratio
                    ):
                        reason = (
                            "member_size_limit"
                            if info.file_size > member_limit
                            else (
                                "total_uncompressed_size_limit"
                                if total > self.limits.max_total_uncompressed_bytes
                                else "compression_ratio_limit"
                            )
                        )
                        yield self._rejected_child(content, info.filename, index, reason)
                        continue
                    if is_dump and info.file_size > self.limits.max_member_bytes:
                        # Large kernel dumps stay path-backed: stream them to a
                        # temp file instead of materializing GBs into memory.
                        yield self._zip_dump_path_child(
                            content, archive, info, index
                        )
                        continue
                    data = archive.read(info)
                    child = self._child(content, info.filename, data, index)
                    yield child
                    if _suffix(info.filename) in _ARCHIVES:
                        yield from self.iter_members(child, nesting=nesting + 1)
        except (OSError, zipfile.BadZipFile):
            yield self._rejected_child(content, "<archive>", 0, "archive_parse_failed")

    def _zip_dump_path_child(
        self,
        parent: ArtifactContent,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        index: int,
    ) -> ArtifactContent:
        """Stream a large dump member to a temp file and return a path-backed child."""

        fd, tmp_path = tempfile.mkstemp(
            prefix="incident_dump_", suffix=_suffix(info.filename)
        )
        self._temp_files.append(tmp_path)
        digest = hashlib.sha256()
        try:
            with archive.open(info, "r") as source, os.fdopen(fd, "wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
        except (OSError, zipfile.BadZipFile):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        identity = hashlib.sha256(
            f"{parent.manifest.sha256}:{info.filename}:{info.CRC}".encode()
        ).hexdigest()
        manifest = ArtifactManifest(
            artifact_id=f"artifact:{identity[:16]}:{index}",
            resource_id=parent.manifest.resource_id,
            name=PurePosixPath(info.filename).name,
            kind=_kind_for_name(info.filename),
            mime=mimetypes.guess_type(info.filename)[0] or "",
            size=info.file_size,
            sha256=digest.hexdigest(),
            path=tmp_path,
            parent_artifact_id=parent.manifest.artifact_id,
            archive_member=info.filename,
            parser_state="available",
            metadata={
                "archive_ancestry": [parent.manifest.artifact_id],
                "path_backed": True,
                "dump_path_backed": True,
                "source_member_crc32": f"{info.CRC:08x}",
                "source_member_size": info.file_size,
                "source_member_compressed_size": info.compress_size,
            },
        )
        return ArtifactContent(manifest=manifest, data=b"")

    def _tar_members(self, content: ArtifactContent, *, nesting: int) -> Iterator[ArtifactContent]:
        total = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(content.data), mode="r:*") as archive:
                members = archive.getmembers()
                if len(members) > self.limits.max_members:
                    yield self._rejected_child(
                        content,
                        "<remaining-members>",
                        self.limits.max_members,
                        "archive_member_count_limit",
                    )
                for index, info in enumerate(members[: self.limits.max_members]):
                    if info.issym() or info.islnk():
                        yield self._rejected_child(content, info.name, index, "archive_link_rejected")
                        continue
                    if not info.isfile():
                        continue
                    if not _safe_member_name(info.name):
                        yield self._rejected_child(content, info.name, index, "unsafe_archive_member_path")
                        continue
                    total += max(0, info.size)
                    if info.size > self.limits.max_member_bytes or total > self.limits.max_total_uncompressed_bytes:
                        reason = (
                            "member_size_limit"
                            if info.size > self.limits.max_member_bytes
                            else "total_uncompressed_size_limit"
                        )
                        yield self._rejected_child(content, info.name, index, reason)
                        continue
                    handle = archive.extractfile(info)
                    if handle is None:
                        continue
                    data = handle.read(self.limits.max_member_bytes + 1)
                    if len(data) > self.limits.max_member_bytes:
                        yield self._rejected_child(content, info.name, index, "member_size_limit")
                        continue
                    child = self._child(content, info.name, data, index)
                    yield child
                    if _suffix(info.name) in _ARCHIVES:
                        yield from self.iter_members(child, nesting=nesting + 1)
        except (OSError, tarfile.TarError):
            yield self._rejected_child(content, "<archive>", 0, "archive_parse_failed")

    def _external_members(self, content: ArtifactContent, *, nesting: int) -> Iterator[ArtifactContent]:
        command = shutil.which("bsdtar")
        if not command or not content.manifest.path:
            yield self._rejected_child(
                content,
                "<external-archive>",
                0,
                "external_archive_reader_unavailable",
            )
            return
        try:
            listed = subprocess.run(
                [command, "-tf", content.manifest.path],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            yield self._rejected_child(content, "<archive>", 0, "archive_list_failed")
            return
        if len(listed) > self.limits.max_members:
            yield self._rejected_child(
                content,
                "<remaining-members>",
                self.limits.max_members,
                "archive_member_count_limit",
            )
        total = 0
        for index, name in enumerate(listed[: self.limits.max_members]):
            name = name.strip()
            if not name or name.endswith("/"):
                continue
            if not _safe_member_name(name):
                yield self._rejected_child(content, name, index, "unsafe_archive_member_path")
                continue
            try:
                proc = subprocess.run(
                    [command, "-xOf", content.manifest.path, name],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            data = proc.stdout
            total += len(data)
            if len(data) > self.limits.max_member_bytes or total > self.limits.max_total_uncompressed_bytes:
                reason = (
                    "member_size_limit"
                    if len(data) > self.limits.max_member_bytes
                    else "total_uncompressed_size_limit"
                )
                yield self._rejected_child(content, name, index, reason)
                continue
            child = self._child(content, name, data, index)
            yield child
            if _suffix(name) in _ARCHIVES:
                yield from self.iter_members(child, nesting=nesting + 1)

    @staticmethod
    def _child(parent: ArtifactContent, name: str, data: bytes, index: int) -> ArtifactContent:
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"artifact:{digest[:16]}:{index}"
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            resource_id=parent.manifest.resource_id,
            name=PurePosixPath(name).name,
            kind=_kind_for_name(name),
            mime=mimetypes.guess_type(name)[0] or "",
            size=len(data),
            sha256=digest,
            parent_artifact_id=parent.manifest.artifact_id,
            archive_member=name,
            parser_state="available",
            metadata={"archive_ancestry": [parent.manifest.artifact_id]},
        )
        return ArtifactContent(manifest=manifest, data=data)

    @staticmethod
    def _rejected_child(
        parent: ArtifactContent,
        name: str,
        index: int,
        reason: str,
    ) -> ArtifactContent:
        normalized = str(name).replace("\\", "/")[:500]
        identity = hashlib.sha256(
            f"{parent.manifest.artifact_id}:{normalized}:{index}:{reason}".encode()
        ).hexdigest()
        manifest = ArtifactManifest(
            artifact_id=f"artifact:rejected:{identity[:16]}",
            resource_id=parent.manifest.resource_id,
            name=PurePosixPath(normalized).name or "rejected-member",
            kind="attachment",
            parent_artifact_id=parent.manifest.artifact_id,
            archive_member=normalized,
            status="rejected",
            parser_state="not_started",
            safety_flags=[reason],
            metadata={"archive_ancestry": [parent.manifest.artifact_id]},
        )
        return ArtifactContent(manifest=manifest, data=b"")

    @staticmethod
    def _metadata_child(
        parent: ArtifactContent,
        name: str,
        index: int,
        *,
        size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactContent:
        identity = hashlib.sha256(
            f"{parent.manifest.artifact_id}:{name}:{index}:metadata".encode()
        ).hexdigest()
        manifest = ArtifactManifest(
            artifact_id=f"artifact:metadata:{identity[:16]}",
            resource_id=parent.manifest.resource_id,
            name=PurePosixPath(name).name,
            kind=_kind_for_name(name),
            size=size,
            parent_artifact_id=parent.manifest.artifact_id,
            archive_member=name,
            parser_state="scope_skipped",
            metadata={"archive_ancestry": [parent.manifest.artifact_id], **(metadata or {})},
        )
        return ArtifactContent(manifest=manifest, data=b"")


def _safe_member_name(name: str) -> bool:
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(
        normalized
        and not normalized.startswith("/")
        and not re.match(r"^[A-Za-z]:", normalized)
        and ".." not in path.parts
        and "\x00" not in normalized
    )


def _suffix(name: str) -> str:
    lowered = str(name).lower()
    if lowered.endswith(".tar.gz"):
        return ".tgz"
    return Path(lowered).suffix


def _kind_for_name(name: str) -> str:
    suffix = _suffix(name)
    if suffix in {".log", ".txt", ".json", ".jsonl", ".csv", ".xml", ".ini", ".cfg", ".yaml", ".yml"}:
        return "log_package"
    if suffix in {".dmp", ".mdmp"}:
        return "dmp"
    if suffix == ".evtx":
        return "log_package"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return "image"
    return "attachment"


def _is_text_name(name: str) -> bool:
    return _suffix(name) in {
        ".log", ".txt", ".json", ".jsonl", ".csv", ".xml", ".ini",
        ".cfg", ".yaml", ".yml", ".out", ".err", ".trace",
    }


def _is_static_environment_name(name: str) -> bool:
    normalized = str(name).replace("\\", "/").lower()
    basename = PurePosixPath(normalized).name
    if basename in {
        "sysinfo.json", "systeminfo.json", "versions.txt", "version.txt",
        "dxdiag.txt", "dxdiag.xml", "environment.json", "hardware.json",
    }:
        return True
    return bool(re.search(
        r"(?:^|/)(?:system[-_]?info|environment|versions?|dxdiag|gpu[-_]?info)"
        r"(?:[-_.]|$)",
        normalized,
    ))


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_stream_line(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _json_bytes(value: dict[str, Any]) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
