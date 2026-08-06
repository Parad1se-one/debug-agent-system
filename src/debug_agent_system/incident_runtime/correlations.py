"""Deterministic cross-event correlations for incident diagnosis."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Iterable

from .contracts import DiagnosticEvent


_FAILURE_KINDS = {
    "illegal_memory_access", "access_violation", "device_lost", "crash",
    "exception", "timeout", "gpu_driver_exception", "display_driver_reset",
    "gpu_live_kernel_event", "crash_dump_exception",
}


def correlate_incident_events(events: Iterable[DiagnosticEvent]) -> list[dict[str, Any]]:
    ordered = sorted(
        list(events),
        key=lambda item: (
            _timestamp(item) is None,
            item.timestamp_utc or item.timestamp_raw,
            item.sequence,
        ),
    )
    correlations: list[dict[str, Any]] = []
    failures = [item for item in ordered if _is_failure(item)]
    starts = [item for item in ordered if item.event_kind == "process_start"]

    for restart in starts:
        restart_time = _timestamp(restart)
        if restart_time is None:
            continue
        eligible = [
            item
            for item in failures
            if _timestamp(item) is not None
            and 0 <= (restart_time - _timestamp(item)).total_seconds() <= 300  # type: ignore[arg-type]
        ]
        if not eligible:
            continue
        latest_time = max(_timestamp(item) for item in eligible if _timestamp(item) is not None)
        episode = [
            item
            for item in eligible
            if _timestamp(item) is not None
            and 0 <= (latest_time - _timestamp(item)).total_seconds() <= 2  # type: ignore[operator]
        ]
        # A CUDA failure commonly emits several asynchronous errors in the same
        # millisecond.  Prefer the event with the richest stable signature, then
        # the first event in that failure episode.
        failure = sorted(
            episode,
            key=lambda item: (-_specificity(item), _timestamp(item), item.sequence),
        )[0]
        failure_time = _timestamp(failure)
        if failure_time is None:
            continue
        delta = (restart_time - failure_time).total_seconds()
        correlations.append({
            "correlation_id": f"correlation:failure-restart:{failure.event_id}:{restart.event_id}",
            "type": "failure_followed_by_process_start",
            "failure_event_id": failure.event_id,
            "process_start_event_id": restart.event_id,
            "failure_timestamp": failure.timestamp_utc or failure.timestamp_raw,
            "process_start_timestamp": restart.timestamp_utc or restart.timestamp_raw,
            "delta_seconds": round(delta, 3),
            "new_process_id": restart.process_id,
            "signature": _signature(failure),
            "evidence_ids": _dedupe([*failure.evidence_ids, *restart.evidence_ids]),
            "interpretation": "异常后观察到应用进程重新启动；这支持闪退/重启闭环，但不单独证明根因。",
            "root_cause_asserted": False,
        })

    by_signature: dict[str, list[DiagnosticEvent]] = {}
    for failure in failures:
        signature = _signature(failure)
        if signature and _specificity(failure) >= 2:
            by_signature.setdefault(signature, []).append(failure)
    for signature, items in sorted(by_signature.items()):
        distinct_dates = {
            timestamp.date().isoformat()
            for item in items
            if (timestamp := _timestamp(item)) is not None
        }
        if len(distinct_dates) < 2:
            continue
        correlations.append({
            "correlation_id": (
                "correlation:recurrence:"
                + hashlib.sha256(signature.encode()).hexdigest()[:16]
            ),
            "type": "repeated_failure_signature",
            "signature": signature,
            "event_ids": [item.event_id for item in items],
            "dates": sorted(distinct_dates),
            "evidence_ids": _dedupe(
                evidence_id for item in items for evidence_id in item.evidence_ids
            ),
            "interpretation": "相同故障签名在不同日期重复发生；这是复发证据，不等同于受控复现。",
            "controlled_reproduction": False,
        })
    return correlations


def event_signature(event: DiagnosticEvent) -> str:
    return _signature(event)


def _is_failure(event: DiagnosticEvent) -> bool:
    severe = event.severity.upper() in {
        "FATAL", "CRITICAL", "ERROR", "ERR", "EXCEPTION", "PANIC"
    }
    return bool(
        event.event_kind in _FAILURE_KINDS
        or severe
    ) and _specificity(event) >= 1


def _specificity(event: DiagnosticEvent) -> int:
    """Score whether an event can support cross-run/restart correlation."""

    score = 0
    if event.error_codes:
        score += 3
    if event.function:
        score += 2
    if event.component:
        score += 1
    if event.event_kind not in {"diagnostic_event", "timeout", "reset", "crash", "exception"}:
        score += 1
    return score


def _timestamp(event: DiagnosticEvent) -> datetime | None:
    raw = event.timestamp_utc or event.timestamp_raw
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(
            raw.replace("/", "-").replace(",", ".").replace("Z", "+00:00")
        )
        # Diagnostic packages commonly mix local timestamps with normalized UTC
        # timestamps.  Correlation only needs a consistent comparison domain, so
        # normalize aware values to naive UTC while leaving source-local naive
        # values untouched.  The original timestamp remains on the event for
        # audit and display.
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _signature(event: DiagnosticEvent) -> str:
    values = [
        *event.error_codes,
        event.event_kind,
        event.component,
        event.function,
    ]
    return "|".join(_dedupe(value.lower() for value in values if value))


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = ["correlate_incident_events", "event_signature"]
