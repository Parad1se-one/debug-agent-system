"""Deterministic diagnostic event, stack, and environment parsers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable

from .artifacts import ArtifactContent
from .contracts import (
    DiagnosticEvent,
    EnvironmentSnapshot,
    EvidenceLink,
    StackFrame,
    StackTrace,
)
from .minidump import (
    MinidumpFormatError,
    parse_kernel_dump_file,
    parse_minidump,
    parse_minidump_file,
)
from .windows_events import (
    EvtxParserUnavailable,
    align_utc_timestamp_to_scope,
    parse_evtx,
)

PARSER_VERSION = "incident-text-v1"
_TEXT_SUFFIXES = {
    ".log", ".txt", ".json", ".jsonl", ".csv", ".xml", ".ini", ".cfg",
    ".yaml", ".yml", ".md", ".out", ".err", ".trace",
}
_TIMESTAMP = re.compile(
    r"(?P<ts>20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_LEVEL = re.compile(r"\b(FATAL|CRITICAL|ERROR|ERR|WARNING|WARN|EXCEPTION|PANIC|INFO|DEBUG)\b", re.I)
_ERROR_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?:0x[0-9a-fA-F]{6,16}|-\d{2,6}|[A-Z][A-Z0-9_]{1,20}[-_:]?\d{2,8}|E\d{3,8})(?![A-Za-z0-9])"
)
_EXPLICIT_CODE_CONTEXT = re.compile(
    r"(?:error|err(?:or)?)[ _-]*(?:code|no)|错误码|异常码|故障码|"
    r"(?:code|errno)\s*[:=]",
    re.I,
)
_EVENT_SIGNAL = re.compile(
    r"error|exception|fatal|panic|failed|failure|illegal memory|access violation|"
    r"timeout|reset|device lost|crash|报错|错误|异常|失败|超时|崩溃|蓝屏|掉线",
    re.I,
)
_LIFECYCLE_SIGNAL = re.compile(
    r"\bApplicationPid\b|\b(?:process|application|service)\b.{0,24}"
    r"\b(?:start(?:ed)?|restart(?:ed)?|exit(?:ed)?|terminated|shutdown)\b|"
    r"进程.{0,16}(?:启动|重启|退出|终止)|程序.{0,16}(?:启动|重启|退出|闪退)",
    re.I,
)
_CONFIG_ASSIGNMENT = re.compile(
    r"^\s*[A-Za-z_][\w.-]{1,100}\s*[:=]\s*[^\s].{0,300}$"
)
_TRACE_HEADER = re.compile(r"stack\s*trace|call\s*stack|trace\s*:|调用栈|堆栈", re.I)
_POSITIVE = re.compile(r"success|succeeded|healthy|normal|passed|恢复正常|成功|正常", re.I)
_THREAD = re.compile(r"\b(?:thread|tid|线程)[ =:#]*(?P<value>0x[0-9a-f]+|\d+)\b", re.I)
_PROCESS = re.compile(r"\b(?:applicationpid|process|pid|进程)[ =:#]*(?P<value>0x[0-9a-f]+|\d+)\b", re.I)
_MODULE_FUNCTION = re.compile(
    r"(?:(?P<module>[A-Za-z0-9_.-]+)[!:\s])?"
    r"(?P<function>[A-Za-z_~][\w~]*(?:::[A-Za-z_~][\w~]*)+(?:<[^>]+>)?|"
    r"(?:cv|cuda|nvidia|symv|torch|tensorflow)(?:::[\w~]+)+)"
)
_FRAME_NUMBERED = re.compile(
    r"^\s*(?P<ordinal>\d+)#\s+(?:(?P<address>(?:0x)?[0-9a-fA-F]{6,16})\s+){0,2}"
    r"(?P<body>.+?)\s*$"
)
_FRAME_AT = re.compile(r"^\s*(?:at\s+|#(?P<ordinal>\d+)\s+)(?P<body>.+)$", re.I)
_SOURCE_LINE = re.compile(
    r"(?P<file>(?:[A-Za-z]:)?[^,\n]+?\.(?:c|cc|cpp|cxx|cu|h|hpp|py|cs|java))"
    r"(?:,?\s*(?:line|:)[ ]*(?P<line>\d+))?",
    re.I,
)
_ADDRESS = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,16}\b")

_ENV_PATTERNS: dict[str, re.Pattern[str]] = {
    "symv_version": re.compile(r"\bSYMV[ /:(]*(\d+(?:\.\d+){1,3})", re.I),
    "opencv_version": re.compile(r"\bOpenCV[ /:=-]*(\d+(?:\.\d+){1,3})", re.I),
    "cuda_version": re.compile(r"\bCUDA(?: Toolkit)?[ /:=-]*(\d+(?:\.\d+){1,3})", re.I),
    "driver_version": re.compile(r"\b(?:NVIDIA|GPU|display)?\s*driver(?: version)?[ /:=-]*(\d+(?:\.\d+){1,4})", re.I),
    "windows_version": re.compile(r"\bWindows(?: 10| 11| Server)?[ /:=-]*([0-9]{4,5}|\d+(?:\.\d+){1,3})", re.I),
    "gpu_model": re.compile(r"\b((?:NVIDIA|GeForce|Quadro|RTX|GTX|Tesla)\s+[A-Za-z0-9 ._-]{2,40})", re.I),
    "commit": re.compile(r"\b(?:commit|revision|rev|build)[ /:=-]*([0-9a-f]{7,40})\b", re.I),
}


@dataclass(slots=True)
class ParsedDiagnostics:
    events: list[DiagnosticEvent] = field(default_factory=list)
    stack_traces: list[StackTrace] = field(default_factory=list)
    environment: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    text_lines: dict[str, list[str]] = field(default_factory=dict)
    exclusions: list[dict[str, Any]] = field(default_factory=list)


class DiagnosticParserRegistry:
    """Read-only parser registry with a stable normalized output."""

    def __init__(
        self,
        *,
        max_text_bytes_per_artifact: int = 32 * 1024 * 1024,
        max_lines_per_artifact: int = 300_000,
        max_events: int = 20_000,
        allow_dump_analysis: bool = False,
        allow_ocr_analysis: bool = False,
    ) -> None:
        self.max_text_bytes_per_artifact = max(1024, int(max_text_bytes_per_artifact))
        self.max_lines_per_artifact = max(100, int(max_lines_per_artifact))
        self.max_events = max(10, int(max_events))
        self.allow_dump_analysis = bool(allow_dump_analysis)
        self.allow_ocr_analysis = bool(allow_ocr_analysis)

    def parse(self, contents: Iterable[ArtifactContent]) -> ParsedDiagnostics:
        output = ParsedDiagnostics()
        for content in contents:
            if content.manifest.parser_state in {"scope_skipped", "time_window_empty"} and not content.data:
                continue
            suffix = Path(content.manifest.name.lower()).suffix
            if suffix == ".evtx":
                self._parse_evtx(content, output)
                continue
            if suffix in {".dmp", ".mdmp"}:
                self._parse_dump_metadata(content, output)
                continue
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                self._parse_image(content, output)
                continue
            if not self._is_text(content):
                continue
            self._parse_text(content, output)
            if len(output.events) >= self.max_events:
                output.exclusions.append({"material": "remaining_events", "reason": "event_budget"})
                break
        output.events = output.events[: self.max_events]
        return output

    def _parse_text(self, content: ArtifactContent, output: ParsedDiagnostics) -> None:
        raw = content.data[: self.max_text_bytes_per_artifact]
        text, encoding = _decode(raw)
        lines = text.replace("\x00", "").splitlines()[: self.max_lines_per_artifact]
        artifact_id = content.manifest.artifact_id
        output.text_lines[artifact_id] = lines
        content.manifest.parser_state = "parsed"
        content.manifest.metadata.update({
            "encoding": encoding,
            "parsed_line_count": len(lines),
            "text_truncated": len(content.data) > len(raw) or len(text.splitlines()) > len(lines),
        })
        current_frames: list[StackFrame] = []
        current_trace_evidence: list[str] = []
        trace_start = 0
        trace_active = False
        known_evidence = {item.evidence_id for item in output.evidence_links}

        def add_evidence(value: EvidenceLink) -> None:
            if value.evidence_id not in known_evidence:
                known_evidence.add(value.evidence_id)
                output.evidence_links.append(value)

        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                if current_frames:
                    self._finish_trace(content, output, current_frames, current_trace_evidence, trace_start)
                    current_frames, current_trace_evidence, trace_start = [], [], 0
                trace_active = False
                continue
            evidence = self._evidence_link(content, line_no, line)
            header = bool(_TRACE_HEADER.search(line))
            frame = self._stack_frame(
                line,
                len(current_frames),
                evidence.evidence_id,
                trace_active=trace_active or header,
            )
            if frame is not None:
                if not current_frames:
                    trace_start = line_no
                current_frames.append(frame)
                current_trace_evidence.append(evidence.evidence_id)
                add_evidence(evidence)
                trace_active = True
            elif current_frames and not header:
                self._finish_trace(content, output, current_frames, current_trace_evidence, trace_start)
                current_frames, current_trace_evidence, trace_start = [], [], 0
                trace_active = False
            elif header:
                trace_active = True

            if frame is not None and not header:
                # Stack frames inherit the parent exception's severity and are
                # evidence for one trace, not independent reset/exception events.
                self._environment(line, evidence.evidence_id, output.environment)
                continue

            level_match = _LEVEL.search(line)
            codes = _extract_error_codes(line)
            level_value = level_match.group(1).upper() if level_match else ""
            severe_level = bool(
                level_match
                and level_value not in {"INFO", "DEBUG"}
            )
            signal = bool(_EVENT_SIGNAL.search(line))
            lifecycle = bool(_LIFECYCLE_SIGNAL.search(line))
            # INFO/DEBUG lines often contain configuration fields named
            # ``timeout``/``reset`` or filenames such as C156.  They are useful
            # context, but are not diagnostic events without a lifecycle signal
            # or explicit error severity.
            signal_is_event = (
                signal
                and (not level_match or severe_level)
                and not (
                    not _TIMESTAMP.search(line)
                    and _CONFIG_ASSIGNMENT.match(line)
                )
            )
            if severe_level or codes or signal_is_event or lifecycle:
                add_evidence(evidence)
                event = self._event(content, len(output.events) + 1, line, line_no, evidence.evidence_id)
                output.events.append(event)
            self._environment(line, evidence.evidence_id, output.environment)
        if current_frames:
            self._finish_trace(content, output, current_frames, current_trace_evidence, trace_start)

    def _event(
        self,
        content: ArtifactContent,
        sequence: int,
        line: str,
        line_no: int,
        evidence_id: str,
    ) -> DiagnosticEvent:
        timestamp_match = _TIMESTAMP.search(line)
        timestamp_raw = timestamp_match.group("ts") if timestamp_match else ""
        level = _LEVEL.search(line)
        severity = (
            level.group(1).upper()
            if level
            else ("INFO" if _LIFECYCLE_SIGNAL.search(line) and not _EVENT_SIGNAL.search(line) else "ERROR")
        )
        thread = _THREAD.search(line)
        process = _PROCESS.search(line)
        function_match = _MODULE_FUNCTION.search(line)
        positive = bool(_POSITIVE.search(line) and not _EVENT_SIGNAL.search(line))
        return DiagnosticEvent(
            event_id=f"event:{hashlib.sha256(f'{content.manifest.artifact_id}:{line_no}:{line}'.encode()).hexdigest()[:16]}",
            artifact_id=content.manifest.artifact_id,
            sequence=sequence,
            severity=severity,
            message=line.strip()[:4000],
            timestamp_raw=timestamp_raw,
            timestamp_utc=_normalize_timestamp(timestamp_raw),
            process_id=process.group("value") if process else "",
            thread_id=thread.group("value") if thread else "",
            component=_component(line),
            module=(function_match.group("module") or "") if function_match else "",
            function=function_match.group("function") if function_match else "",
            error_codes=_extract_error_codes(line)[:20],
            event_kind=_event_kind(line),
            polarity="positive" if positive else "negative",
            evidence_ids=[evidence_id],
        )

    def _stack_frame(
        self,
        line: str,
        ordinal: int,
        evidence_id: str,
        *,
        trace_active: bool = False,
    ) -> StackFrame | None:
        if not trace_active:
            return None
        match = _FRAME_NUMBERED.match(line) or _FRAME_AT.match(line)
        if match is None and not (
            _MODULE_FUNCTION.search(line)
            and (" in " in line or "file " in line.lower() or _ADDRESS.search(line))
        ):
            return None
        body = str(match.groupdict().get("body") or line).strip() if match else line.strip()
        function = _MODULE_FUNCTION.search(body)
        source = _SOURCE_LINE.search(body)
        address = ""
        if match:
            address = str(match.groupdict().get("address") or "")
        if not address:
            found_address = _ADDRESS.search(body)
            address = found_address.group(0) if found_address else ""
        source_file = source.group("file").strip() if source else ""
        line_value = int(source.group("line")) if source and source.group("line") else None
        return StackFrame(
            ordinal=ordinal,
            raw=line.strip()[:2000],
            module=(function.group("module") or "") if function else "",
            function=function.group("function") if function else "",
            source_file=source_file,
            line=line_value,
            address=address,
            stability="volatile" if address and not function else "contextual",
            evidence_ids=[evidence_id],
        )

    def _finish_trace(
        self,
        content: ArtifactContent,
        output: ParsedDiagnostics,
        frames: list[StackFrame],
        evidence_ids: list[str],
        line_start: int,
    ) -> None:
        if not frames:
            return
        trace_id = f"trace:{hashlib.sha256(f'{content.manifest.artifact_id}:{line_start}'.encode()).hexdigest()[:16]}"
        output.stack_traces.append(StackTrace(
            trace_id=trace_id,
            artifact_id=content.manifest.artifact_id,
            frames=list(frames),
            evidence_ids=_dedupe(evidence_ids),
        ))

    def _environment(self, line: str, evidence_id: str, snapshot: EnvironmentSnapshot) -> None:
        for field_name, pattern in _ENV_PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group(1).strip(" ,;)]}")
                values = snapshot.values.setdefault(field_name, [])
                if value and value not in values:
                    values.append(value)
                    snapshot.evidence_ids.setdefault(field_name, []).append(evidence_id)

    def _parse_dump_metadata(self, content: ArtifactContent, output: ParsedDiagnostics) -> None:
        try:
            if content.manifest.path and content.manifest.metadata.get("dump_path_backed"):
                parsed = parse_minidump_file(content.manifest.path)
            else:
                parsed = parse_minidump(content.data)
        except MinidumpFormatError as exc:
            # MEMORY.DMP from a Windows kernel bugcheck is a PAGEDU64 full
            # dump, not a Minidump.  Parse its header for the bugcheck code.
            if (
                content.manifest.path
                and content.manifest.metadata.get("dump_path_backed")
            ):
                kernel = self._parse_kernel_dump_fallback(
                    content, output, original_reason=str(exc)
                )
                if kernel is not None:
                    return
            content.manifest.parser_state = "parse_failed"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "dump_metadata",
                "reason": str(exc),
            })
            return
        content.manifest.parser_state = "parsed"
        alignment = align_utc_timestamp_to_scope(
            str(parsed.get("created_time_utc") or ""),
            content.manifest.metadata.get("time_scope"),
        )
        parsed["time_alignment"] = alignment
        content.manifest.metadata["minidump"] = parsed
        evidence = EvidenceLink(
            evidence_id=f"evidence:{content.manifest.sha256[:16]}:minidump-streams",
            artifact_id=content.manifest.artifact_id,
            source_name=content.manifest.name,
            sha256=content.manifest.sha256,
            byte_start=0,
            byte_end=len(content.data),
            timestamp=str(parsed.get("created_time_utc") or ""),
            extraction_method="bounded_minidump_stream_parser",
            parser_version=PARSER_VERSION,
        )
        output.evidence_links.append(evidence)
        self._minidump_environment(parsed, evidence.evidence_id, output.environment)
        exception = parsed.get("exception") or {}
        if exception:
            code = str(exception.get("code") or "")
            module = str(exception.get("module") or "")
            message = "; ".join(filter(None, [
                "Windows minidump exception",
                f"name={exception.get('name')}" if exception.get("name") else "",
                f"code={code}" if code else "",
                f"module={module}" if module else "",
                f"address={exception.get('address')}" if exception.get("address") else "",
            ]))
            output.events.append(DiagnosticEvent(
                event_id=f"event:{hashlib.sha256(f'{content.manifest.artifact_id}:minidump-exception'.encode()).hexdigest()[:16]}",
                artifact_id=content.manifest.artifact_id,
                sequence=len(output.events) + 1,
                severity="ERROR",
                message=message,
                timestamp_raw=str(parsed.get("created_time_utc") or ""),
                timestamp_utc=str(alignment.get("timestamp_local") or parsed.get("created_time_utc") or ""),
                process_id=str((parsed.get("process") or {}).get("process_id") or ""),
                thread_id=str(exception.get("thread_id") or ""),
                component="application_crash",
                module=module,
                error_codes=[code] if code else [],
                event_kind="crash_dump_exception",
                evidence_ids=[evidence.evidence_id],
            ))
        output.exclusions.append({
            "artifact_id": content.manifest.artifact_id,
            "material": "dump_symbolized_call_stack",
            "reason": "symbolized_debugger_not_configured",
        })
        if not self.allow_dump_analysis:
            return
        debugger = shutil.which("cdb") or shutil.which("windbg")
        if not debugger or not content.manifest.path:
            output.exclusions[-1]["reason"] = "enabled_dump_debugger_unavailable"
            return
        try:
            proc = subprocess.run(
                [debugger, "-z", content.manifest.path, "-c", "!analyze -v; k; q"],
                check=False,
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            output.exclusions[-1]["reason"] = "dump_debugger_failed"
            return
        if not proc.stdout:
            output.exclusions[-1]["reason"] = "dump_debugger_empty_output"
            return
        debug_manifest = replace(
            content.manifest,
            name=f"{content.manifest.name}.debugger.txt",
            mime="text/plain",
            parser_state="available",
            metadata={**content.manifest.metadata, "derived_by": Path(debugger).name},
        )
        self._parse_text(ArtifactContent(debug_manifest, proc.stdout), output)
        output.exclusions[-1]["reason"] = "structured_metadata_and_debugger_output_retained"

    def _parse_kernel_dump_fallback(
        self,
        content: ArtifactContent,
        output: ParsedDiagnostics,
        *,
        original_reason: str,
    ) -> bool:
        """Parse a PAGEDU64 full-kernel-dump header when Minidump parsing fails.

        Returns True when the file is a recognized kernel dump; the caller then
        returns without emitting a generic parse failure.
        """

        try:
            parsed = parse_kernel_dump_file(content.manifest.path)
        except (MinidumpFormatError, OSError):
            return False
        content.manifest.parser_state = "parsed"
        content.manifest.metadata["kernel_dump"] = parsed
        evidence = EvidenceLink(
            evidence_id=f"evidence:{content.manifest.sha256[:16]}:kernel-dump-header",
            artifact_id=content.manifest.artifact_id,
            source_name=content.manifest.name,
            sha256=content.manifest.sha256,
            byte_start=0,
            byte_end=256,
            extraction_method="bounded_kernel_dump_header_parser",
            parser_version=PARSER_VERSION,
        )
        output.evidence_links.append(evidence)
        bugcheck = parsed.get("bugcheck") or {}
        code = str(bugcheck.get("code") or "")
        name = str(bugcheck.get("name") or "")
        message = "; ".join(filter(None, [
            "Windows kernel full dump",
            f"bugcheck={code}" if code else "",
            f"name={name}" if name and name != "unknown_bugcheck" else "",
            f"os={parsed.get('os_version')}" if parsed.get("os_version") else "",
            f"processors={parsed.get('processor_count')}" if parsed.get("processor_count") else "",
        ]))
        output.events.append(DiagnosticEvent(
            event_id=f"event:{hashlib.sha256(f'{content.manifest.artifact_id}:kernel-dump-bugcheck'.encode()).hexdigest()[:16]}",
            artifact_id=content.manifest.artifact_id,
            sequence=len(output.events) + 1,
            severity="CRITICAL",
            message=message[:4000],
            timestamp_raw="",
            timestamp_utc="",
            process_id="",
            thread_id="",
            component="kernel_bugcheck",
            module=name,
            error_codes=_dedupe([f"Bugcheck:{code}" if code else "", *([code] if code else [])]),
            event_kind="kernel_power_loss",
            evidence_ids=[evidence.evidence_id],
        ))
        output.exclusions.append({
            "artifact_id": content.manifest.artifact_id,
            "material": "dump_symbolized_call_stack",
            "reason": "kernel_full_dump_header_only_no_symbolized_debugger",
        })
        return True

    def _parse_evtx(self, content: ArtifactContent, output: ParsedDiagnostics) -> None:
        try:
            parsed = parse_evtx(
                content.data,
                time_scope=content.manifest.metadata.get("time_scope"),
                max_records=self.max_events * 5,
                max_selected_records=self.max_events,
            )
        except EvtxParserUnavailable:
            content.manifest.parser_state = "metadata_only"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "evtx_events",
                "reason": "evtx_native_parser_unavailable",
            })
            return
        except Exception as exc:  # third-party EVTX corruption/parser boundary
            content.manifest.parser_state = "parse_failed"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "evtx_events",
                "reason": f"evtx_parse_failed:{type(exc).__name__}",
            })
            return
        try:
            records = parsed.get("records") or []
        except (AttributeError, TypeError) as exc:
            content.manifest.parser_state = "parse_failed"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "evtx_events",
                "reason": f"evtx_parse_failed:{type(exc).__name__}",
            })
            return
        content.manifest.parser_state = "parsed"
        content.manifest.metadata["evtx"] = {
            key: value for key, value in parsed.items() if key != "records"
        }
        for record in records:
            if not record.get("signal"):
                continue
            evidence = EvidenceLink(
                evidence_id=f"evidence:{content.manifest.sha256[:16]}:evtx:{record.get('record_number')}",
                artifact_id=content.manifest.artifact_id,
                source_name=content.manifest.name,
                sha256=content.manifest.sha256,
                timestamp=str(record.get("timestamp_utc") or ""),
                extraction_method="python_evtx_record_xml",
                parser_version=PARSER_VERSION,
            )
            output.evidence_links.append(evidence)
            event_kind, component, codes, module = _evtx_classification(record)
            event_identity = (
                f"{content.manifest.artifact_id}:evtx:{record.get('record_number')}"
            )
            output.events.append(DiagnosticEvent(
                event_id=f"event:{hashlib.sha256(event_identity.encode()).hexdigest()[:16]}",
                artifact_id=content.manifest.artifact_id,
                sequence=len(output.events) + 1,
                severity=str(record.get("severity") or "ERROR"),
                message=(
                    f"provider={record.get('provider')}; event_id={record.get('event_id')}; "
                    f"{record.get('message') or ''}"
                )[:4000],
                timestamp_raw=str(record.get("timestamp_utc") or ""),
                timestamp_utc=str(record.get("timestamp_local") or record.get("timestamp_utc") or ""),
                process_id=str(record.get("process_id") or ""),
                thread_id=str(record.get("thread_id") or ""),
                component=component,
                module=module,
                error_codes=codes,
                event_kind=event_kind,
                evidence_ids=[evidence.evidence_id],
            ))

    @staticmethod
    def _minidump_environment(
        parsed: dict[str, Any],
        evidence_id: str,
        snapshot: EnvironmentSnapshot,
    ) -> None:
        def add(field: str, value: Any) -> None:
            text = str(value or "").strip()
            if not text:
                return
            values = snapshot.values.setdefault(field, [])
            if text not in values:
                values.append(text)
                snapshot.evidence_ids.setdefault(field, []).append(evidence_id)

        system = parsed.get("system") or {}
        process = parsed.get("process") or {}
        add("windows_version", system.get("os_version"))
        add("architecture", system.get("architecture"))
        add("process_id", process.get("process_id"))
        for module in parsed.get("modules") or []:
            name = str(module.get("name") or "").lower()
            version = str(module.get("file_version") or "")
            if not name or not version or version == "0.0.0.0":
                continue
            add("loaded_module_versions", f"{name}={version}")
            if name == "symv.dll":
                add("symv_version", version)
            elif name.startswith("nvcuda") or name == "nvldumdx.dll":
                add("nvidia_driver_version", version)
            elif name.startswith("cudart"):
                add("cuda_runtime_version", version)

    def _parse_image(self, content: ArtifactContent, output: ParsedDiagnostics) -> None:
        if not self.allow_ocr_analysis:
            content.manifest.parser_state = "metadata_only"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "image_text",
                "reason": "ocr_analysis_disabled",
            })
            return
        tesseract = shutil.which("tesseract")
        if not tesseract:
            content.manifest.parser_state = "metadata_only"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "image_text",
                "reason": "ocr_engine_unavailable",
            })
            return
        try:
            proc = subprocess.run(
                [tesseract, "stdin", "stdout", "-l", "chi_sim+eng"],
                input=content.data,
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is None or not proc.stdout:
            content.manifest.parser_state = "parse_failed"
            output.exclusions.append({
                "artifact_id": content.manifest.artifact_id,
                "material": "image_text",
                "reason": "ocr_failed_or_empty",
            })
            return
        derived = replace(
            content.manifest,
            name=f"{content.manifest.name}.ocr.txt",
            mime="text/plain",
            parser_state="available",
            metadata={**content.manifest.metadata, "derived_by": "tesseract"},
        )
        self._parse_text(ArtifactContent(derived, proc.stdout), output)
    @staticmethod
    def _evidence_link(content: ArtifactContent, line_no: int, line: str) -> EvidenceLink:
        source_lines = content.manifest.metadata.get("source_lines") or []
        source_line = (
            int(source_lines[line_no - 1])
            if isinstance(source_lines, list) and line_no - 1 < len(source_lines)
            else line_no
        )
        suffix = hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest()[:10]
        return EvidenceLink(
            evidence_id=f"evidence:{content.manifest.sha256[:12]}:{source_line}:{suffix}",
            artifact_id=content.manifest.artifact_id,
            source_name=(content.manifest.archive_member or content.manifest.name),
            sha256=content.manifest.sha256,
            line_start=source_line,
            line_end=source_line,
            extraction_method=(
                "query_time_window_stream"
                if content.manifest.metadata.get("derived_by") == "query_time_window_stream"
                else "text_line_parser"
            ),
            parser_version=PARSER_VERSION,
        )

    @staticmethod
    def _is_text(content: ArtifactContent) -> bool:
        suffix = Path(content.manifest.name.lower()).suffix
        if suffix in _TEXT_SUFFIXES or content.manifest.mime.startswith("text/"):
            return True
        sample = content.data[:4096]
        if not sample:
            return False
        return sample.count(b"\x00") / len(sample) < 0.05


def _evtx_classification(
    record: dict[str, Any],
) -> tuple[str, str, list[str], str]:
    provider = str(record.get("provider") or "")
    event_id = str(record.get("event_id") or "")
    message = str(record.get("message") or "")
    lowered = f"{provider} {message}".lower()
    codes = [f"EVTX:{event_id}"] if event_id else []
    codes.extend(f"0x{value.upper()}" for value in re.findall(r"(?i)0x([0-9a-f]{6,16})", message))
    module = "nvlddmkm.sys" if "nvlddmkm" in lowered else ""
    if provider.lower() == "nvlddmkm":
        return "gpu_driver_exception", "gpu_driver", _dedupe(codes), module
    if provider.lower() == "display" and event_id == "4101":
        return "display_driver_reset", "gpu_driver", _dedupe(codes), module
    if "livekernelevent" in lowered and re.search(r"(?:^|\D)141(?:\D|$)", message):
        codes.append("LiveKernelEvent:141")
        return "gpu_live_kernel_event", "gpu_driver", _dedupe(codes), module
    if provider.lower() == "microsoft-windows-kernel-power" and event_id == "41":
        # Event 41 with a non-zero bugcheck code records an unexpected shutdown
        # caused by a bugcheck; the code is the primary root-cause anchor.
        bugcheck = _kernel_power_bugcheck(record)
        if bugcheck:
            codes.append(f"Bugcheck:{bugcheck}")
        return "kernel_power_loss", "kernel_power", _dedupe(codes), module
    if "windows error reporting" in provider.lower() and event_id == "1001" and "bluescreen" in lowered:
        bugcheck = _wer_blue_screen_bugcheck(record)
        if bugcheck:
            codes.append(f"Bugcheck:{bugcheck}")
        return "windows_blue_screen", "windows_error_reporting", _dedupe(codes), module
    if "windows error reporting" in provider.lower():
        return "windows_error_report", "windows_error_reporting", _dedupe(codes), module
    if provider.lower() == "microsoft-windows-ndis" and event_id == "10400":
        # NDIS 10400 reports network interface up/down churn.  It is a weak
        # signal by itself, but strongly relevant to "插网卡即蓝屏" cases.
        return "network_adapter", "network", _dedupe(codes), module
    if provider.lower() in {"e1rexpress", "e2fexpress", "e1i65x64", "ndis"}:
        return "network_driver", "network", _dedupe(codes), module
    if "whea" in lowered:
        return "hardware_error", "hardware", _dedupe(codes), module
    return "windows_system_event", provider or "windows", _dedupe(codes), module


def _kernel_power_bugcheck(record: dict[str, Any]) -> str:
    """Extract the BugcheckCode from a Kernel-Power 41 EventData payload.

    Kernel-Power 41 stores the bugcheck code as a decimal value (for example
    239 == 0x000000EF == CRITICAL_PROCESS_DIED).  It is normalized to its
    canonical 0x form so the cross-source hypothesis can align it with the
    Query stop code.
    """

    raw = ""
    for value in (record.get("data") or {}).get("BugcheckCode", []):
        raw = str(value).strip()
        if raw:
            break
    if not raw:
        # Flattened message is "rendered; values...", e.g. "0; 239; 0xffffad09...".
        # The bugcheck code is the first non-zero decimal field that is not an
        # 0x-prefixed address.
        for field in re.split(r";\s*", str(record.get("message") or "")):
            field = field.strip()
            if not field or field.lower().startswith("0x"):
                continue
            if re.fullmatch(r"\d{1,8}", field) and int(field, 10) > 0:
                raw = field
                break
    try:
        code = int(raw, 10)
    except (TypeError, ValueError):
        return ""
    if code and 0 < code <= 0xFFFF:
        return f"0x{code:08X}"
    return raw


def _wer_blue_screen_bugcheck(record: dict[str, Any]) -> str:
    """Extract the bugcheck code from a WER 1001 BlueScreen event.

    WER 1001 renders the bugcheck code in hexadecimal without the 0x prefix
    (for example ``4e`` == 0x0000004E).  Normalize it to canonical 0x form so
    it aligns with Kernel-Power 41 and the Query stop code.
    """

    raw = ""
    for value in (record.get("data") or {}).get("BugcheckCode", []):
        raw = str(value).strip()
        if raw:
            break
    if not raw:
        # "0; BlueScreen; 不可用; 4e; 99; 1fc16a4; 2; c0004600046a4fd; ..."
        match = re.search(
            r"BlueScreen\s*;\s*[^;]*;\s*([0-9a-fA-F]{1,8})(?:\s*;|$)",
            str(record.get("message") or ""),
        )
        if match:
            raw = match.group(1)
    if raw.lower().startswith("0x"):
        digits = raw[2:]
    else:
        digits = raw
    if not re.fullmatch(r"[0-9a-fA-F]{1,8}", digits):
        return ""
    try:
        code = int(digits, 16)
    except ValueError:
        return ""
    if code and 0 < code <= 0xFFFF:
        return f"0x{code:08X}"
    return raw


def _decode(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def _extract_error_codes(line: str) -> list[str]:
    values: list[str] = []
    level = _LEVEL.search(line)
    severe = bool(level and level.group(1).upper() not in {"INFO", "DEBUG"})
    diagnostic_context = bool(
        severe or _EXPLICIT_CODE_CONTEXT.search(line) or _EVENT_SIGNAL.search(line)
    )
    for value in _ERROR_CODE.findall(str(line)):
        if re.fullmatch(r"[A-Z][A-Z0-9_]+-\d+", value):
            # Jira keys, station IDs and hardware serials are identifiers, not
            # error codes, unless a future format-specific parser says so.
            continue
        if re.search(rf"\[\s*{re.escape(value)}\s*\]\s*(?:ERROR|WARN|INFO|DEBUG)\b", line, re.I):
            # Common logger thread token position: [0x00004774] ERROR.
            continue
        if re.fullmatch(r"(?:0x[0-9a-fA-F]{6,16}|-\d{2,6}|[A-Z]\d{2,8})", value):
            if not diagnostic_context:
                # Short letter-number tokens are commonly image/FOV/part names.
                # Hexadecimal and negative numeric values are also frequently
                # pointers or coordinates.  Promote them only in a diagnostic
                # record or when explicitly labelled as an error code.
                continue
        elif not diagnostic_context and not _EXPLICIT_CODE_CONTEXT.search(line):
            # Version/runtime/library identifiers that happen to end in digits
            # are not error codes in ordinary INFO/DEBUG records.
            continue
        values.append(value)
    return _dedupe(values)


def _normalize_timestamp(value: str) -> str:
    if not value:
        return ""
    candidate = value.replace("/", "-").replace(",", ".")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        return dt.isoformat(timespec="microseconds")
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _component(line: str) -> str:
    lowered = line.lower()
    for signal, component in (
        ("cuda", "gpu_cuda"), ("gpumat", "gpu_cuda"), ("nvidia", "gpu_driver"),
        ("camera", "camera"), ("capture", "camera"), ("network", "network"),
        ("socket", "network"), ("disk", "storage"), ("windows", "operating_system"),
    ):
        if signal in lowered:
            return component
    return ""


def _event_kind(line: str) -> str:
    lowered = line.lower()
    for signal, kind in (
        ("applicationpid", "process_start"),
        ("process started", "process_start"),
        ("application started", "process_start"),
        ("process exited", "process_exit"),
        ("application exited", "process_exit"),
        ("进程启动", "process_start"),
        ("进程退出", "process_exit"),
        ("illegal memory", "illegal_memory_access"),
        ("access violation", "access_violation"),
        ("timeout", "timeout"),
        ("reset", "reset"),
        ("device lost", "device_lost"),
        ("crash", "crash"),
        ("exception", "exception"),
    ):
        if signal in lowered:
            return kind
    return "diagnostic_event"


def _dedupe(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


__all__ = ["DiagnosticParserRegistry", "ParsedDiagnostics", "PARSER_VERSION"]
