"""Cross-artifact incident timeline construction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .contracts import DiagnosticEvent


def build_timeline(events: Iterable[DiagnosticEvent]) -> list[dict[str, Any]]:
    indexed = list(enumerate(events))

    def key(item: tuple[int, DiagnosticEvent]) -> tuple[int, str, int, int]:
        index, event = item
        timestamp = event.timestamp_utc or event.timestamp_raw
        return (0 if timestamp else 1, timestamp, event.sequence, index)

    ordered = [event for _, event in sorted(indexed, key=key)]
    timeline: list[dict[str, Any]] = []
    previous_signature = ""
    duplicate_count = 0
    for event in ordered:
        signature = "|".join([
            event.event_kind,
            event.component,
            event.function,
            ",".join(event.error_codes),
            _compact_message(event.message),
        ])
        if signature == previous_signature and timeline:
            duplicate_count += 1
            timeline[-1]["repeat_count"] = duplicate_count + 1
            timeline[-1]["evidence_ids"] = _dedupe([
                *timeline[-1].get("evidence_ids", []),
                *event.evidence_ids,
            ])
            continue
        previous_signature = signature
        duplicate_count = 0
        timeline.append({
            "event_id": event.event_id,
            "timestamp": event.timestamp_utc or event.timestamp_raw,
            "time_status": "normalized" if event.timestamp_utc else ("raw" if event.timestamp_raw else "unknown"),
            "severity": event.severity,
            "event_kind": event.event_kind,
            "component": event.component,
            "module": event.module,
            "function": event.function,
            "error_codes": list(event.error_codes),
            "message": event.message,
            "artifact_id": event.artifact_id,
            "evidence_ids": list(event.evidence_ids),
            "repeat_count": 1,
        })
    return timeline


def read_log_window(
    text_lines: dict[str, list[str]],
    artifact_id: str,
    line: int,
    *,
    before: int = 10,
    after: int = 20,
) -> dict[str, Any]:
    lines = text_lines.get(artifact_id)
    if lines is None:
        return {"status": "missing", "artifact_id": artifact_id, "lines": []}
    center = max(1, int(line))
    start = max(1, center - max(0, int(before)))
    end = min(len(lines), center + max(0, int(after)))
    return {
        "status": "ok",
        "artifact_id": artifact_id,
        "line_start": start,
        "line_end": end,
        "lines": [
            {"line": index, "text": lines[index - 1]}
            for index in range(start, end + 1)
        ],
    }


def _compact_message(value: str) -> str:
    import re

    text = re.sub(r"\b(?:0x)?[0-9a-fA-F]{8,16}\b", "<address>", str(value))
    text = re.sub(r"\b(?:thread|tid|pid)[ =:#]*\d+\b", "<runtime-id>", text, flags=re.I)
    return " ".join(text.split()).lower()[:500]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = ["build_timeline", "read_log_window"]
