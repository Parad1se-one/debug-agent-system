"""Read-only Windows EVTX normalization with query-time alignment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import html
import os
import re
from tempfile import NamedTemporaryFile
from typing import Any
import xml.etree.ElementTree as ET


EVTX_PARSER_VERSION = "windows-evtx-events-v1"
_NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
_SIGNAL = re.compile(
    r"error|exception|failed|failure|crash|fault|hang|watchdog|livekernelevent|"
    r"reset|stopped responding|nvlddmkm|display driver|bugcheck|whea|错误|异常|失败|崩溃",
    re.I,
)
_INNER_STRING = re.compile(r"<string>(.*?)</string>", re.I | re.S)
_LEVELS = {1: "CRITICAL", 2: "ERROR", 3: "WARNING", 4: "INFO", 5: "VERBOSE"}


class EvtxParserUnavailable(RuntimeError):
    """Raised when the optional native EVTX reader is not installed."""


def align_utc_timestamp_to_scope(
    timestamp_utc: str,
    time_scope: dict[str, Any] | None,
) -> dict[str, Any]:
    """Align a UTC binary-artifact timestamp to the query's local windows."""

    windows = _scope_windows(time_scope)
    if not windows:
        return {"timestamp_local": timestamp_utc, "local_utc_offset_minutes": 0, "method": "no_query_time_scope"}
    parsed = _parse_utc(timestamp_utc)
    if parsed is None:
        return {"timestamp_local": "", "local_utc_offset_minutes": 0, "method": "timestamp_unavailable"}
    candidates: list[tuple[float, int, datetime]] = []
    for offset in range(-12 * 60, 14 * 60 + 1, 15):
        local = parsed.replace(tzinfo=None) + timedelta(minutes=offset)
        distance = min(
            abs((local - (start + (end - start) / 2)).total_seconds())
            for start, end in windows
        )
        if any(start <= local <= end for start, end in windows):
            candidates.append((distance, offset, local))
    if not candidates:
        return {"timestamp_local": parsed.isoformat(), "local_utc_offset_minutes": 0, "method": "outside_query_time_scope"}
    _, offset, local = min(candidates, key=lambda item: (item[0], abs(item[1])))
    return {
        "timestamp_local": local.isoformat(),
        "local_utc_offset_minutes": offset,
        "method": "reference_window_timestamp_alignment",
    }


def parse_evtx(
    data: bytes,
    *,
    time_scope: dict[str, Any] | None = None,
    max_records: int = 100_000,
    max_selected_records: int = 5_000,
) -> dict[str, Any]:
    """Parse EVTX bytes and retain source-bound events around query times."""

    try:
        from Evtx.Evtx import Evtx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EvtxParserUnavailable("python_evtx_not_installed") from exc

    records: list[dict[str, Any]] = []
    path = ""
    try:
        with NamedTemporaryFile(suffix=".evtx", delete=False) as temp:
            temp.write(data)
            path = temp.name
        with Evtx(path) as handle:
            for index, record in enumerate(handle.records()):
                if index >= max_records:
                    break
                try:
                    normalized = normalize_event_xml(record.xml(), record_number=index + 1)
                except (ET.ParseError, ValueError, TypeError):
                    continue
                records.append(normalized)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    windows = _scope_windows(time_scope)
    alignment = _select_time_alignment(records, windows)
    selected: list[dict[str, Any]] = []
    for record in records:
        local_time = _aligned_time(record.get("timestamp_utc"), alignment["local_utc_offset_minutes"])
        record["timestamp_local"] = local_time.isoformat() if local_time else ""
        record["signal"] = _is_signal(record)
        if windows and (local_time is None or not any(start <= local_time <= end for start, end in windows)):
            continue
        if not windows and not record["signal"]:
            continue
        selected.append(record)
        if len(selected) >= max_selected_records:
            break

    return {
        "schema_version": "debug_agent_system.windows_evtx.v1",
        "parser_version": EVTX_PARSER_VERSION,
        "record_count": len(records),
        "selected_record_count": len(selected),
        "records": selected,
        "time_alignment": alignment,
        "truncated": len(records) >= max_records or len(selected) >= max_selected_records,
    }


def normalize_event_xml(xml: str, *, record_number: int = 0) -> dict[str, Any]:
    """Normalize one rendered EVTX XML record without provider templates."""

    root = ET.fromstring(xml)
    system = root.find("e:System", _NS)
    if system is None:
        raise ValueError("evtx_system_node_missing")
    provider_node = system.find("e:Provider", _NS)
    time_node = system.find("e:TimeCreated", _NS)
    execution = system.find("e:Execution", _NS)
    timestamp_raw = time_node.attrib.get("SystemTime", "") if time_node is not None else ""
    timestamp = _parse_utc(timestamp_raw)
    values: list[str] = []
    named: dict[str, list[str]] = {}
    for node in root.findall(".//e:EventData/e:Data", _NS):
        extracted = _expand_value(node.text or "")
        name = node.attrib.get("Name", "").strip()
        for value in extracted:
            if value:
                values.append(value)
                if name:
                    named.setdefault(name, []).append(value)
    for node in root.findall(".//e:UserData//*", _NS):
        if len(node) or not (node.text or "").strip():
            continue
        name = node.tag.rsplit("}", 1)[-1]
        value = (node.text or "").strip()
        values.append(f"{name}={value}")
        named.setdefault(name, []).append(value)
    rendered = root.findtext(".//e:RenderingInfo/e:Message", default="", namespaces=_NS).strip()
    message_parts = ([rendered] if rendered else []) + values
    return {
        "record_number": record_number,
        "record_id": _text(system, "e:EventRecordID"),
        "timestamp_raw": timestamp_raw,
        "timestamp_utc": timestamp.isoformat() if timestamp else "",
        "provider": provider_node.attrib.get("Name", "") if provider_node is not None else "",
        "provider_guid": provider_node.attrib.get("Guid", "") if provider_node is not None else "",
        "event_id": _text(system, "e:EventID"),
        "level_code": _int(_text(system, "e:Level")),
        "severity": _LEVELS.get(_int(_text(system, "e:Level")), "UNKNOWN"),
        "channel": _text(system, "e:Channel"),
        "computer": _text(system, "e:Computer"),
        "process_id": execution.attrib.get("ProcessID", "") if execution is not None else "",
        "thread_id": execution.attrib.get("ThreadID", "") if execution is not None else "",
        "message": "; ".join(_dedupe(message_parts))[:8000],
        "data": named,
        "values": _dedupe(values)[:200],
    }


def _select_time_alignment(
    records: list[dict[str, Any]],
    windows: list[tuple[datetime, datetime]],
) -> dict[str, Any]:
    if not windows:
        return {
            "local_utc_offset_minutes": 0,
            "method": "no_query_time_scope",
            "score": 0,
        }
    biases: list[int] = []
    for record in records:
        for value in record.get("data", {}).get("CurrentBias", []):
            try:
                biases.append(-int(value))
            except (TypeError, ValueError):
                continue
    if biases:
        offset = max(set(biases), key=biases.count)
        return {
            "local_utc_offset_minutes": offset,
            "method": "windows_current_bias",
            "score": _alignment_score(records, windows, offset),
        }

    candidates = range(-12 * 60, 14 * 60 + 1, 15)
    scored = [(offset, _alignment_score(records, windows, offset)) for offset in candidates]
    offset, score = max(scored, key=lambda item: (item[1], -abs(item[0])))
    return {
        "local_utc_offset_minutes": offset if score else 0,
        "method": "reference_window_signal_alignment" if score else "no_signal_alignment",
        "score": score,
    }


def _alignment_score(
    records: list[dict[str, Any]],
    windows: list[tuple[datetime, datetime]],
    offset: int,
) -> int:
    score = 0
    for record in records:
        local_time = _aligned_time(record.get("timestamp_utc"), offset)
        if local_time is None or not any(start <= local_time <= end for start, end in windows):
            continue
        if _is_signal(record):
            provider = str(record.get("provider") or "").lower()
            message = str(record.get("message") or "").lower()
            score += 5 if any(x in f"{provider} {message}" for x in ("nvlddmkm", "livekernelevent", "watchdog", "bugcheck")) else 2
    return score


def _is_signal(record: dict[str, Any]) -> bool:
    if int(record.get("level_code") or 0) in {1, 2, 3}:
        return True
    value = " ".join([
        str(record.get("provider") or ""),
        str(record.get("event_id") or ""),
        str(record.get("message") or ""),
    ])
    return bool(_SIGNAL.search(value))


def _scope_windows(scope: dict[str, Any] | None) -> list[tuple[datetime, datetime]]:
    result: list[tuple[datetime, datetime]] = []
    for item in (scope or {}).get("reference_windows") or []:
        try:
            result.append((datetime.fromisoformat(item["start_time"]), datetime.fromisoformat(item["end_time"])))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _aligned_time(value: Any, offset_minutes: int) -> datetime | None:
    parsed = _parse_utc(str(value or ""))
    if parsed is None:
        return None
    return parsed.replace(tzinfo=None) + timedelta(minutes=offset_minutes)


def _parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _expand_value(value: str) -> list[str]:
    value = html.unescape(value).strip()
    inner = [html.unescape(item).strip() for item in _INNER_STRING.findall(value)]
    return [item for item in inner if item] or ([value] if value else [])


def _text(node: ET.Element, path: str) -> str:
    return node.findtext(path, default="", namespaces=_NS).strip()


def _int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))
