"""Relation-aware merge and session construction for Xing message history.

This module is deliberately separate from the semantic W1 extractors.  It only
solves source alignment, provenance, reference edges, and session boundaries;
it does not infer fault meaning or mutate KG data.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


RELATION_FIELDS = ("root_id", "parent_id", "thread_id", "upper_message_id")
CONTEXT_CONTINUATION_TERMS = (
    "错件", "漏检", "漏报", "误报", "极反", "极性", "灯芯", "颜色", "色差", "大小", "贴反", "偏位", "错位",
    "连锡", "少锡", "虚焊", "翘脚", "飞件", "OCR", "模型", "算法", "模板", "参数", "焊盘", "框",
    "相机", "拍照", "拍摄", "光源", "初始化", "蓝屏", "黑屏", "重启", "卡死", "闪退", "卡顿", "报错",
    "工控机", "内存", "显卡", "驱动", "网卡", "版本", "主程序", "运控", "日志", "DLOG", "dmp", "proj",
    "Jira", "工单", "金手指",
)
CONTEXT_CONTINUATION_MARKERS = (
    "这个", "那个", "另外", "还是", "也", "目前", "似乎", "然后", "补充", "刚才", "上面", "前面", "同样",
    "不是", "反正", "已经", "继续", "对应", "这种", "所以",
)

_TRACE_RECURRENCE_MARKERS = (
    "复发", "再次出现", "又出现", "仍然", "依旧", "继续排查", "后续反馈", "上次", "之前", "未解决",
)
_TRACE_FAULT_TERMS = (
    "蓝屏", "黑屏", "自动重启", "自动关机", "断电", "无法开机", "拍摄失败", "相机", "残帧",
    "网卡", "网络", "掉线", "重置", "buddy", "保存失败", "http 500", "d盘", "光源初始化失败",
    "user.cfg", "配置加载失败", "卡死", "闪退", "误报", "漏检", "sata", "电源",
)
_EQUIPMENT_KEY_RE = re.compile(
    r"\b(?:AOI[-_ ]?\d{3,6}|SI\d{3,6}[A-Z]?|\d{3,5}T|[A-Z]{2,8}[-_]\d{2,8})\b",
    re.IGNORECASE,
)
_REPORT_FRAGMENT_RE = re.compile(
    r"(?:^|\n|(?<=[。；;]))\s*(?:\d{1,2}|[一二三四五六七八九十]+)[、.．:：]\s*"
)


def semantic_message_fragments(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a mixed report into provenance-preserving semantic fragments."""
    text_value = _text(message.get("text"))
    message_id = _text(message.get("message_id"))
    if not text_value:
        return []
    parts = [part.strip(" ，,。；;\n") for part in _REPORT_FRAGMENT_RE.split(text_value) if part.strip(" ，,。；;\n")]
    if len(parts) <= 1:
        parts = [text_value.strip()]
    out: list[dict[str, Any]] = []
    for index, fragment_text in enumerate(parts, start=1):
        lowered = fragment_text.lower()
        fault_terms = sorted({term for term in _TRACE_FAULT_TERMS if term in lowered})
        equipment_keys = sorted({match.group(0).upper().replace(" ", "") for match in _EQUIPMENT_KEY_RE.finditer(fragment_text)})
        out.append({
            "fragment_id": f"{message_id or 'message'}#fragment:{index}",
            "source_message_id": message_id,
            "fragment_index": index,
            "text": fragment_text,
            "fault_terms": fault_terms,
            "equipment_keys": equipment_keys,
            "mixed_report_fragment": len(parts) > 1,
        })
    return out


def annotate_semantic_fragments(messages: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    fragment_count = 0
    mixed_message_count = 0
    for message in messages:
        row = dict(message)
        fragments = semantic_message_fragments(row)
        row["semantic_fragments"] = fragments
        fragment_count += len(fragments)
        if len(fragments) > 1:
            mixed_message_count += 1
        rows.append(row)
    return rows, {"semantic_fragment_count": fragment_count, "mixed_report_message_count": mixed_message_count}


def infer_cross_window_trace_edges(
    messages: Iterable[dict[str, Any]],
    *,
    max_back_days: float = 30.0,
    max_back_messages: int = 500,
) -> list[dict[str, Any]]:
    """Link high-confidence recurrence updates across ordinary W1 windows."""
    by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        by_chat[_text(message.get("chat_id")) or "default"].append(message)
    edges: list[dict[str, Any]] = []
    for chat_id, rows in by_chat.items():
        rows.sort(key=lambda row: (_text(row.get("create_time")), _text(row.get("message_id"))))
        features = []
        for row in rows:
            value = _text(row.get("text"))
            lowered = value.lower()
            features.append({
                "time": _parse_time(row.get("create_time")),
                "faults": {term for term in _TRACE_FAULT_TERMS if term in lowered},
                "equipment": {match.group(0).upper().replace(" ", "") for match in _EQUIPMENT_KEY_RE.finditer(value)},
                "artifact_payload": attachment_identity_keys(row),
                "recurrence": any(marker in value for marker in _TRACE_RECURRENCE_MARKERS),
                "mixed_report": len(semantic_message_fragments(row)) > 1,
            })
        for index, row in enumerate(rows):
            current = features[index]
            current_id = _text(row.get("message_id"))
            if not current_id or current["time"] is None or current["mixed_report"]:
                continue
            candidates: list[
                tuple[int, float, dict[str, Any], set[str], set[str], set[str]]
            ] = []
            for previous_index in range(index - 1, max(-1, index - max_back_messages - 1), -1):
                previous = features[previous_index]
                previous_row = rows[previous_index]
                if previous["time"] is None or previous["mixed_report"]:
                    continue
                gap_days = (current["time"] - previous["time"]).total_seconds() / 86400
                if gap_days < 0 or gap_days > max_back_days:
                    continue
                shared_equipment = current["equipment"] & previous["equipment"]
                shared_faults = current["faults"] & previous["faults"]
                shared_artifact_payload = current["artifact_payload"] & previous["artifact_payload"]
                # Cross-window merging is intentionally stricter than local
                # continuation.  It needs a stable equipment key, or recurrence
                # language plus at least two shared fault facets, or an exact
                # payload identity together with a shared fault facet.
                if (
                    not shared_equipment
                    and not (current["recurrence"] and len(shared_faults) >= 2)
                    and not (shared_artifact_payload and shared_faults)
                ):
                    continue
                score = (
                    4 * len(shared_equipment)
                    + 2 * len(shared_faults)
                    + 6 * len(shared_artifact_payload)
                    + (2 if current["recurrence"] else 0)
                )
                if score >= 6:
                    candidates.append((
                        score,
                        gap_days,
                        previous_row,
                        shared_equipment,
                        shared_faults,
                        shared_artifact_payload,
                    ))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (-item[0], item[1]))
            score, gap_days, previous_row, shared_equipment, shared_faults, shared_artifact_payload = candidates[0]
            edges.append({
                "source": current_id,
                "target": _text(previous_row.get("message_id")),
                "type": "cross_window_trace_continuation",
                "chat_id": chat_id,
                "target_present": True,
                "source_field": "inferred_semantic_trace",
                "inferred": True,
                "confidence": round(min(1.0, score / 12.0), 4),
                "score": score,
                "gap_days": round(gap_days, 4),
                "reason_codes": [
                    "same_chat",
                    "cross_window",
                    *(["same_equipment"] if shared_equipment else []),
                    *(["same_artifact_payload"] if shared_artifact_payload else []),
                    *(["recurrence_language"] if current["recurrence"] else []),
                ],
                "shared_equipment_keys": sorted(shared_equipment),
                "shared_fault_terms": sorted(shared_faults),
                "shared_artifact_payload_keys": sorted(shared_artifact_payload),
            })
    return edges
REPORT_HARD_BOUNDARY_MARKERS = (
    "今日现状", "今日工作汇总", "今日设备状态", "现场工作汇报", "现场工作汇总", "每日反馈", "工作汇报",
    "现场问题汇总", "现场问题如下", "问题点汇总", "异常点汇报", "异常汇总", "今日3d-demo现状", "今日demo现状",
)
JIRA_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}-\d+\b")
KNOWN_JIRA_KEY_RE = re.compile(r"\b(?:SMTAOITS|TEST)-\d+\b", re.IGNORECASE)
PROJECT_KEY_RE = re.compile(r"\b[A-Za-z0-9_-]{4,}\.(?:proj|dlog|dmp)\b", re.IGNORECASE)
LINE_KEY_RE = re.compile(r"(?:第)?([一二三四五六七八九十]|\d{1,2})线")
_EXPLICIT_OTHER_DEVICE_MARKERS = ("另外一台", "另一台", "另一个设备", "不是这个设备", "不是同一台")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_STABLE_LARK_FILE_KEY_RE = re.compile(r"^(?:file_v\d+_|boxcn)[A-Za-z0-9_-]{10,}$", re.IGNORECASE)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ATTACHMENT_ALLOWED_ROOT = _REPO_ROOT / "data/imports"
_ATTACHMENT_HASH_MAX_BYTES = 8 * 1024 * 1024
_FULL_ATTACHMENT_SOURCE_STATUSES = {"api_ok", "client_cache_full", "client_recovered"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _parse_time(value: Any) -> datetime | None:
    value = _text(value)
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _norm_content(value: Any) -> str:
    value = re.sub(r"\s+", " ", _text(value))
    return value[:2000]


def _message_key(message: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _text(message.get("chat_id")),
        _text(message.get("create_time")),
        _norm_content(message.get("text") or message.get("plain_text") or message.get("content")),
    )


def _stable_id(*values: Any) -> str:
    raw = "|".join(_text(value) for value in values)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _sender_name(message: dict[str, Any]) -> str:
    sender = message.get("sender")
    if isinstance(sender, dict):
        return _text(sender.get("name") or sender.get("id"))
    return _text(sender)


def _mentions(message: dict[str, Any]) -> set[str]:
    values = message.get("mentions") or []
    output: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("id") or ""
        value = _text(value).lstrip("@")
        if value:
            output.add(value)
    return output


def _context_terms(message: dict[str, Any]) -> set[str]:
    text = _text(message.get("text") or message.get("plain_text") or message.get("content"))
    terms = {term.lower() for term in CONTEXT_CONTINUATION_TERMS if term.lower() in text.lower()}
    terms.update(value.lower() for value in _jira_keys(text))
    terms.update(re.findall(r"(?<![\d.])v?\d{1,2}\.\d+(?:\.\d+){0,2}(?![\d.])", text, re.IGNORECASE))
    return terms


def _jira_keys(text: str) -> set[str]:
    """Extract Jira-like keys without treating equipment ids as Jira."""
    return {
        value.upper()
        for value in KNOWN_JIRA_KEY_RE.findall(text)
    }


@lru_cache(maxsize=8192)
def _bounded_attachment_sha256(
    path_value: str,
    size_bytes: int,
    mtime_ns: int,
) -> str:
    # Size and mtime are cache identity inputs. Recheck the file before reading
    # so a replacement cannot reuse a stale digest.
    path = Path(path_value)
    try:
        stat = path.stat()
        if stat.st_size != size_bytes or stat.st_mtime_ns != mtime_ns:
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat_after = path.stat()
    except OSError:
        return ""
    if (
        stat_after.st_size != size_bytes
        or stat_after.st_mtime_ns != mtime_ns
    ):
        return ""
    return digest.hexdigest()


def _local_attachment_sha256(attachment: dict[str, Any]) -> str:
    if _text(attachment.get("source_status")) not in _FULL_ATTACHMENT_SOURCE_STATUSES:
        return ""
    path_value = next(
        (
            attachment.get(key)
            for key in ("path", "local_path", "file_path")
            if attachment.get(key)
        ),
        None,
    )
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        path = path.resolve(strict=True)
        allowed_root = _ATTACHMENT_ALLOWED_ROOT.resolve(strict=True)
        if not path.is_relative_to(allowed_root) or not path.is_file():
            return ""
        stat = path.stat()
    except OSError:
        return ""
    if stat.st_size <= 0 or stat.st_size > _ATTACHMENT_HASH_MAX_BYTES:
        return ""
    return _bounded_attachment_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def attachment_identity_keys(message: dict[str, Any]) -> set[str]:
    """Return exact payload identities, never filename-only similarities."""

    attachments: list[Any] = []
    for key in ("attachments", "attachment_metadata"):
        values = message.get(key)
        if isinstance(values, list):
            attachments.extend(values)
    identities: set[str] = set()
    for value in attachments:
        if not isinstance(value, dict):
            continue
        for key in ("sha256", "file_sha256", "content_sha256", "payload_sha256"):
            digest = _text(value.get(key)).lower()
            if _SHA256_RE.fullmatch(digest):
                identities.add(f"sha256:{digest}")
        file_key = _text(value.get("file_key"))
        if _STABLE_LARK_FILE_KEY_RE.fullmatch(file_key):
            identities.add(f"lark-file-key:{file_key.lower()}")
        local_digest = _local_attachment_sha256(value)
        if local_digest:
            identities.add(f"sha256:{local_digest}")
    return identities


def _is_report_hard_boundary(message: dict[str, Any]) -> bool:
    text = _text(message.get("text") or message.get("plain_text") or message.get("content"))
    return any(marker.lower() in text.lower() for marker in REPORT_HARD_BOUNDARY_MARKERS)


def _identity_features(message: dict[str, Any]) -> dict[str, set[str]]:
    """Extract conservative identity keys used only as merge constraints."""
    text = _text(message.get("text") or message.get("plain_text") or message.get("content"))
    return {
        "equipment": {
            match.group(0).upper().replace(" ", "") for match in _EQUIPMENT_KEY_RE.finditer(text)
            if not KNOWN_JIRA_KEY_RE.fullmatch(match.group(0))
        },
        "line": {match.group(1) for match in LINE_KEY_RE.finditer(text)},
        "jira": _jira_keys(text),
        "artifact": {value.lower() for value in PROJECT_KEY_RE.findall(text)},
        "artifact_payload": attachment_identity_keys(message),
        "fault": {term.lower() for term in CONTEXT_CONTINUATION_TERMS if term.lower() in text.lower()},
        "recurrence": {marker for marker in _TRACE_RECURRENCE_MARKERS if marker in text},
        "explicit_other": {marker for marker in _EXPLICIT_OTHER_DEVICE_MARKERS if marker in text},
    }


def _merge_identity_features(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key, values in _identity_features(row).items():
            result[key].update(values)
    return dict(result)


def _soft_component_conflicts(
    left: dict[str, set[str]],
    right: dict[str, set[str]],
) -> list[str]:
    """Return high-precision cannot-link reasons between two components."""
    reasons: list[str] = []
    shared_strong_identity = bool(
        (left.get("jira", set()) & right.get("jira", set()))
        or (left.get("artifact", set()) & right.get("artifact", set()))
        or (left.get("artifact_payload", set()) & right.get("artifact_payload", set()))
    )
    shared_fault = left.get("fault", set()) & right.get("fault", set())
    recurrence = bool(left.get("recurrence") or right.get("recurrence"))
    left_equipment, right_equipment = left.get("equipment", set()), right.get("equipment", set())
    if (
        left_equipment and right_equipment and left_equipment.isdisjoint(right_equipment)
        and not shared_strong_identity and not (recurrence and shared_fault)
    ):
        reasons.append("conflicting_equipment_identity")
    left_lines, right_lines = left.get("line", set()), right.get("line", set())
    if (
        left_lines and right_lines and left_lines.isdisjoint(right_lines)
        and not shared_strong_identity and not (recurrence and shared_fault)
    ):
        reasons.append("conflicting_line_identity")
    if (left.get("explicit_other") or right.get("explicit_other")) and not shared_strong_identity:
        reasons.append("explicit_other_device")
    # Different Jira keys alone are not a conflict: duplicate Jira issues can
    # describe the same trace.  They become a cannot-link only when fault
    # signatures also disagree and there is no shared device/artifact key.
    if (
        left.get("jira") and right.get("jira") and left["jira"].isdisjoint(right["jira"])
        and not shared_fault
        and not (left_equipment & right_equipment)
        and not (left.get("artifact", set()) & right.get("artifact", set()))
        and not (left.get("artifact_payload", set()) & right.get("artifact_payload", set()))
    ):
        reasons.append("incompatible_jira_and_fault_identity")
    return reasons


def _decode_soft_context_edges(
    rows: list[dict[str, Any]],
    edges: Iterable[dict[str, Any]],
    uf: "_UnionFind",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Constrained Kruskal decoder for inferred W1 edges.

    Native reply/root relations have already formed hard components.  Inferred
    edges are considered by confidence, rejected on component-level identity
    conflicts, and never override a hard report boundary.
    """
    by_id = {_text(row.get("message_id")): row for row in rows}
    report_ids = {message_id for message_id, row in by_id.items() if _is_report_hard_boundary(row)}
    candidates = sorted(
        (dict(edge) for edge in edges),
        key=lambda edge: (
            -float(edge.get("score") or (float(edge.get("confidence") or 0.0) * 8.0)),
            float(edge.get("gap_hours") or (float(edge.get("gap_days") or 0.0) * 24.0)),
            _text(edge.get("source")),
            _text(edge.get("target")),
        ),
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for edge in candidates:
        source, target = _text(edge.get("source")), _text(edge.get("target"))
        if source not in by_id or target not in by_id or source == target:
            rejected.append({**edge, "soft_edge_status": "rejected", "soft_edge_reasons": ["missing_or_invalid_endpoint"]})
            continue
        if source in report_ids or target in report_ids:
            rejected.append({**edge, "soft_edge_status": "rejected", "soft_edge_reasons": ["report_hard_boundary"]})
            continue
        source_root, target_root = uf.find(source), uf.find(target)
        if source_root == target_root:
            accepted.append({**edge, "soft_edge_status": "redundant", "soft_edge_reasons": ["already_hard_connected"]})
            continue
        edge_type = _text(edge.get("type"))
        score = float(edge.get("score") or (float(edge.get("confidence") or 0.0) * 8.0))
        minimum_score = 6.0 if edge_type == "cross_window_trace_continuation" else 4.0
        if score < minimum_score:
            rejected.append({**edge, "soft_edge_status": "rejected", "soft_edge_reasons": ["below_soft_threshold"]})
            continue
        component_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for message_id, row in by_id.items():
            component_rows[uf.find(message_id)].append(row)
        conflicts = _soft_component_conflicts(
            _merge_identity_features(component_rows[source_root]),
            _merge_identity_features(component_rows[target_root]),
        )
        if conflicts:
            rejected.append({**edge, "soft_edge_status": "rejected", "soft_edge_reasons": conflicts})
            continue
        uf.union(source, target)
        accepted.append({**edge, "soft_edge_status": "accepted", "soft_edge_reasons": ["score_and_constraints_passed"]})
    return accepted, rejected


def _has_media_context(message: dict[str, Any]) -> bool:
    text = _text(message.get("text") or message.get("plain_text") or message.get("content"))
    return bool(message.get("attachments") or "[图片]" in text or "[media:" in text.lower() or "[file:" in text.lower())


def infer_context_continuation_edges(
    messages: Iterable[dict[str, Any]],
    *,
    max_back_hours: float = 24.0,
    max_back_messages: int = 12,
    minimum_score: float = 4.0,
    max_edges_per_message: int = 2,
) -> list[dict[str, Any]]:
    """Infer conservative predecessor links not represented by Feishu replies.

    These edges are explicitly marked inferred. They recover fragmented
    descriptions and nearby image/file context without pretending to be
    platform-native reply metadata.
    """
    by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        by_chat[_text(message.get("chat_id")) or "default"].append(message)
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chat_id, rows in by_chat.items():
        rows.sort(key=lambda row: (_text(row.get("create_time")), _text(row.get("message_id"))))
        features = [
            {
                "time": _parse_time(row.get("create_time")),
                "terms": _context_terms(row),
                "sender": _sender_name(row),
                "mentions": _mentions(row),
                "text": _text(row.get("text")),
                "report": _is_report_hard_boundary(row),
                "media": _has_media_context(row),
                "jira": {value for value in _context_terms(row) if JIRA_KEY_RE.fullmatch(value.upper())},
                "artifact_payload": attachment_identity_keys(row),
            }
            for row in rows
        ]
        for index, current in enumerate(rows):
            current_id = _text(current.get("message_id"))
            current_feature = features[index]
            current_time = current_feature["time"]
            if not current_id or current_time is None or current_feature["report"]:
                continue
            current_terms = current_feature["terms"]
            current_sender = current_feature["sender"]
            current_mentions = current_feature["mentions"]
            current_text = current_feature["text"]
            candidates: list[
                tuple[float, float, dict[str, Any], list[str], set[str], set[str]]
            ] = []
            for previous_index in range(index - 1, max(-1, index - max_back_messages - 1), -1):
                previous = rows[previous_index]
                previous_feature = features[previous_index]
                previous_id = _text(previous.get("message_id"))
                previous_time = previous_feature["time"]
                if not previous_id or previous_time is None:
                    continue
                gap_hours = (current_time - previous_time).total_seconds() / 3600
                if gap_hours < 0 or gap_hours > max_back_hours:
                    continue
                if previous_feature["report"]:
                    break
                previous_terms = previous_feature["terms"]
                shared_terms = current_terms & previous_terms
                current_jira = current_feature["jira"]
                previous_jira = previous_feature["jira"]
                shared_artifact_payload = (
                    current_feature["artifact_payload"]
                    & previous_feature["artifact_payload"]
                )
                if current_jira and previous_jira and current_jira.isdisjoint(previous_jira):
                    continue
                same_sender = bool(current_sender and current_sender == previous_feature["sender"])
                shared_mentions = current_mentions & previous_feature["mentions"]
                continuation_language = any(current_text.startswith(marker) or marker in current_text[:24] for marker in CONTEXT_CONTINUATION_MARKERS)
                media_context = current_feature["media"] or previous_feature["media"]
                identifier_support = bool(current_jira and previous_jira and current_jira == previous_jira)
                payload_support = bool(
                    shared_artifact_payload
                    and (shared_terms or continuation_language)
                )
                continuation_has_context = bool(current_terms or previous_terms or media_context)
                short_same_sender_continuation = bool(same_sender and continuation_language and continuation_has_context and gap_hours <= (10 / 60))
                short_mention_continuation = bool(shared_mentions and continuation_language and continuation_has_context and gap_hours <= (10 / 60))
                semantic_support = bool(
                    shared_terms
                    or identifier_support
                    or payload_support
                    or short_same_sender_continuation
                    or short_mention_continuation
                )
                if not semantic_support:
                    continue
                score = 0.0
                reasons: list[str] = ["same_chat"]
                if gap_hours <= 0.5:
                    score += 2.0
                    reasons.append("gap_le_30m")
                elif gap_hours <= 2:
                    score += 1.0
                    reasons.append("gap_le_2h")
                elif gap_hours <= 24:
                    reasons.append("gap_le_24h")
                if len(shared_terms) >= 2:
                    score += 3.0
                    reasons.append("multiple_shared_fault_terms")
                elif shared_terms:
                    score += 2.0
                    reasons.append("shared_fault_term")
                if same_sender:
                    score += 1.0
                    reasons.append("same_sender")
                if shared_mentions:
                    score += 1.0
                    reasons.append("shared_mentions")
                if continuation_language:
                    score += 1.0
                    reasons.append("continuation_language")
                if media_context:
                    score += 0.5
                    reasons.append("media_context")
                if current_jira and previous_jira and current_jira == previous_jira:
                    score += 3.0
                    reasons.append("same_jira")
                if payload_support:
                    score += 4.0
                    reasons.append("same_artifact_payload")
                if score >= minimum_score:
                    candidates.append((
                        score,
                        gap_hours,
                        previous,
                        reasons,
                        set(shared_terms),
                        set(shared_artifact_payload),
                    ))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (-item[0], item[1], _text(item[2].get("message_id"))))
            best_score = candidates[0][0]
            accepted = [candidate for candidate in candidates if candidate[0] >= best_score - 1.5][:max_edges_per_message]
            for (
                score,
                gap_hours,
                previous,
                reasons,
                candidate_shared_terms,
                candidate_shared_artifact_payload,
            ) in accepted:
                previous_id = _text(previous.get("message_id"))
                key = (current_id, previous_id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "source": current_id,
                    "target": previous_id,
                    "type": "context_continuation",
                    "chat_id": chat_id,
                    "target_present": True,
                    "source_field": "inferred_context",
                    "inferred": True,
                    "confidence": round(min(1.0, score / 8.0), 4),
                    "score": score,
                    "gap_hours": round(gap_hours, 4),
                    "reason_codes": reasons,
                    "shared_terms": sorted(candidate_shared_terms),
                    "shared_artifact_payload_keys": sorted(candidate_shared_artifact_payload),
                })
    return edges


def _relation_values(row: dict[str, Any]) -> dict[str, str]:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    return {
        "root_id": _text(row.get("root_id") or raw.get("root_id")),
        "parent_id": _text(row.get("parent_id") or raw.get("parent_id")),
        "thread_id": _text(row.get("relation_thread_id") or raw.get("thread_id")),
        "upper_message_id": _text(row.get("upper_message_id") or raw.get("upper_message_id")),
    }


def _merge_message(old: dict[str, Any], relation: dict[str, Any], *, match_type: str) -> dict[str, Any]:
    """Keep old Xing attachments/evidence, add v3 relation metadata."""
    merged = dict(old)
    merged["relation_source"] = "v3-relations"
    merged["relation_match_type"] = match_type
    relation_values = _relation_values(relation)
    for field, value in relation_values.items():
        if value:
            if field == "thread_id":
                merged["relation_thread_id"] = value
            else:
                merged[field] = value
    raw = dict(merged.get("raw") or {})
    relation_raw = dict(relation.get("raw") or relation)
    raw.update({
        "relation_source": "full-2015-to-2026-07-09-v3-relations",
        "relation_match_type": match_type,
        "relation_fields": {field: value for field, value in relation_values.items() if value},
        "v3_chat_name": relation.get("chat_name") or relation_raw.get("chat_name") or "",
    })
    merged["raw"] = raw
    if not merged.get("text"):
        merged["text"] = relation.get("text") or relation.get("plain_text") or relation.get("content") or ""
    if not merged.get("chat_id"):
        merged["chat_id"] = relation.get("chat_id") or ""
    if not merged.get("create_time"):
        merged["create_time"] = relation.get("create_time") or ""
    return merged


def _dedupe_dicts(items: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = tuple(_text(item.get(field)) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(item))
    return output


def _dedupe_old_xing(messages: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse repeated manifest rows while preserving all resource evidence."""
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    duplicate_rows = 0
    for row in messages:
        item = dict(row)
        message_id = _text(item.get("message_id") or item.get("id"))
        identity = ("id", message_id) if message_id else ("fallback", *_message_key(item))
        existing = grouped.get(identity)
        if existing is None:
            raw = dict(item.get("raw") or {})
            raw["source_thread_ids"] = [_text(item.get("thread_id"))] if _text(item.get("thread_id")) else []
            item["raw"] = raw
            grouped[identity] = item
            continue
        duplicate_rows += 1
        existing["attachments"] = _dedupe_dicts(
            [*(existing.get("attachments") or []), *(item.get("attachments") or [])],
            ("file_key", "path", "name"),
        )
        existing["links"] = _dedupe_dicts(
            [*(existing.get("links") or []), *(item.get("links") or [])],
            ("url", "label", "type"),
        )
        if len(_text(item.get("text"))) > len(_text(existing.get("text"))):
            existing["text"] = item.get("text")
        raw = dict(existing.get("raw") or {})
        source_thread_ids = list(raw.get("source_thread_ids") or [])
        source_thread_id = _text(item.get("thread_id"))
        if source_thread_id and source_thread_id not in source_thread_ids:
            source_thread_ids.append(source_thread_id)
        raw["source_thread_ids"] = source_thread_ids
        raw["duplicate_source_rows"] = int(raw.get("duplicate_source_rows") or 0) + 1
        existing["raw"] = raw
    return list(grouped.values()), duplicate_rows


def merge_xing_relation_history(
    old_messages: Iterable[dict[str, Any]],
    relation_messages: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge v3 relation rows into the old Xing corpus without losing resources.

    Matching precedence is exact ``message_id`` followed by exact
    ``(chat_id, create_time, normalized text)``.  The fallback is intentionally
    conservative: a chat/time match alone is not enough to merge messages.
    """
    old_input = [dict(row) for row in old_messages]
    old, old_duplicate_rows = _dedupe_old_xing(old_input)
    v3 = [dict(row) for row in relation_messages]
    v3_by_id: dict[str, dict[str, Any]] = {}
    v3_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    duplicate_v3_ids = 0
    for row in v3:
        message_id = _text(row.get("message_id") or row.get("id"))
        if message_id:
            if message_id in v3_by_id:
                duplicate_v3_ids += 1
            else:
                v3_by_id[message_id] = row
        v3_by_key[_message_key(row)].append(row)

    output: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    matched_v3_ids: set[str] = set()
    exact_matches = 0
    fallback_matches = 0
    for row in old:
        message_id = _text(row.get("message_id") or row.get("id"))
        relation = v3_by_id.get(message_id)
        match_type = "message_id" if relation else ""
        if relation is None:
            candidates = [candidate for candidate in v3_by_key.get(_message_key(row), []) if _text(candidate.get("message_id")) not in matched_v3_ids]
            if len(candidates) == 1:
                relation = candidates[0]
                match_type = "chat_id_create_time_text"
                fallback_matches += 1
        if relation is not None:
            output.append(_merge_message(row, relation, match_type=match_type))
            matched_ids.add(message_id)
            if match_type == "message_id":
                exact_matches += 1
            if _text(relation.get("message_id")):
                matched_v3_ids.add(_text(relation.get("message_id")))
        else:
            untouched = dict(row)
            untouched.setdefault("relation_source", "unmatched_old_xing")
            output.append(untouched)

    v3_only = 0
    output_keys = {_message_key(item) for item in output}
    for row in v3:
        message_id = _text(row.get("message_id") or row.get("id"))
        if message_id and message_id in matched_v3_ids:
            continue
        if not message_id and _message_key(row) in output_keys:
            continue
        item = dict(row)
        item["relation_thread_id"] = _text(row.get("thread_id"))
        item["relation_source"] = "v3-relations"
        item["relation_match_type"] = "v3_only"
        item.setdefault("attachments", [])
        raw = dict(item.get("raw") or item)
        raw["relation_source"] = "full-2015-to-2026-07-09-v3-relations"
        raw["relation_match_type"] = "v3_only"
        item["raw"] = raw
        output.append(item)
        output_keys.add(_message_key(item))
        v3_only += 1

    output.sort(key=lambda row: (_text(row.get("create_time")), _text(row.get("chat_id")), _text(row.get("message_id"))))
    report = {
        "old_input_rows": len(old_input),
        "old_messages_after_dedupe": len(old),
        "old_duplicate_rows_collapsed": old_duplicate_rows,
        "relation_messages": len(v3),
        "merged_messages": len(output),
        "matched_by_message_id": exact_matches,
        "matched_by_chat_id_create_time_text": fallback_matches,
        "v3_only_messages": v3_only,
        "old_unmatched_messages": len(old) - len(matched_ids),
        "duplicate_v3_message_ids": duplicate_v3_ids,
        "dedupe_policy": ["message_id", "chat_id+create_time+normalized_text"],
    }
    return output, report


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent.get(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent.get(value, value)

    def union(self, left: str, right: str) -> None:
        if left not in self.parent or right not in self.parent:
            return
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_message_reference_graph(
    messages: Iterable[dict[str, Any]],
    *,
    context_edges: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an auditable directed graph from root/parent relation fields."""
    rows = [dict(row) for row in messages]
    node_ids = {_text(row.get("message_id")) for row in rows if _text(row.get("message_id"))}
    edges: list[dict[str, Any]] = []
    edge_seen: set[tuple[str, str, str]] = set()
    for row in rows:
        child = _text(row.get("message_id"))
        if not child:
            continue
        relation = _relation_values(row)
        for field, edge_type in (("parent_id", "reply_to"), ("root_id", "in_thread"), ("upper_message_id", "upper_message")):
            target = relation[field]
            if not target or target == child:
                continue
            key = (child, target, edge_type)
            if key in edge_seen:
                continue
            edge_seen.add(key)
            edges.append({
                "source": child,
                "target": target,
                "type": edge_type,
                "chat_id": _text(row.get("chat_id")),
                "target_present": target in node_ids,
                "source_field": field,
            })
    for edge in context_edges or []:
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        edge_type = _text(edge.get("type")) or "context_continuation"
        key = (source, target, edge_type)
        if not source or not target or source == target or key in edge_seen:
            continue
        edge_seen.add(key)
        edges.append({**dict(edge), "source": source, "target": target, "type": edge_type, "target_present": target in node_ids})
    stats = {
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "reply_edge_count": sum(edge["type"] == "reply_to" for edge in edges),
        "thread_membership_edge_count": sum(edge["type"] == "in_thread" for edge in edges),
        "upper_message_edge_count": sum(edge["type"] == "upper_message" for edge in edges),
        "context_continuation_edge_count": sum(edge["type"] == "context_continuation" for edge in edges),
        "cross_window_trace_continuation_edge_count": sum(
            edge["type"] == "cross_window_trace_continuation" for edge in edges
        ),
        "resolved_edge_count": sum(edge["target_present"] for edge in edges),
        "unresolved_edge_count": sum(not edge["target_present"] for edge in edges),
    }
    return {"schema": "w1.message_reference_graph.v1", "nodes": sorted(node_ids), "edges": edges, "stats": stats}


def _time_segment_id(chat_id: str, items: list[dict[str, Any]], index: int) -> str:
    start = _text(items[0].get("create_time")).replace(" ", "T") if items else ""
    end = _text(items[-1].get("create_time")).replace(" ", "T") if items else ""
    return f"{chat_id}:{start}:{end}:refseg:{index}"


def assign_reference_aware_segments(
    messages: Iterable[dict[str, Any]],
    *,
    quiet_gap_hours: float = 12.0,
    max_messages: int = 120,
    context_attach_minutes: float = 60.0,
    context_edges: Iterable[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign stable session ids, preserving reply/reference components.

    Temporal breaks remain the default for unrelated messages.  Any connected
    root/parent component is treated as a hard session constraint, so a long
    reply chain is not split merely because it crosses the old fixed window.
    """
    rows = [dict(row) for row in messages]
    context_edge_rows = [dict(edge) for edge in (context_edges or [])]
    context_edges_by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in context_edge_rows:
        context_edges_by_chat[_text(edge.get("chat_id")) or "default"].append(edge)
    by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chat[_text(row.get("chat_id")) or "default"].append(row)
    assigned: list[dict[str, Any]] = []
    session_count = 0
    reference_component_count = 0
    parallel_temporal_blocks_split = 0
    isolated_messages_attached = 0
    isolated_session_count = 0
    soft_context_edges_accepted = 0
    soft_context_edges_rejected = 0
    soft_context_edges_redundant = 0
    soft_context_rejection_counts: dict[str, int] = defaultdict(int)
    for chat_id, chat_rows in by_chat.items():
        chat_rows.sort(key=lambda row: (_text(row.get("create_time")), _text(row.get("message_id"))))
        ids = [_text(row.get("message_id")) for row in chat_rows]
        uf = _UnionFind(ids)
        id_set = set(ids)
        report_ids = {_text(row.get("message_id")) for row in chat_rows if _is_report_hard_boundary(row)}
        referenced_ids: set[str] = set()
        unresolved_root_first: dict[str, str] = {}
        for row in chat_rows:
            child = _text(row.get("message_id"))
            relation = _relation_values(row)
            if child in report_ids:
                # A daily/field report is a new multi-issue anchor. Preserve its
                # official outgoing edge in the graph, but do not let that edge
                # merge the report upward into a preceding fault session.
                referenced_ids.add(child)
                continue
            for field in ("parent_id", "root_id", "upper_message_id"):
                target = relation[field]
                if target:
                    referenced_ids.add(child)
                if target in id_set:
                    uf.union(child, target)
            # Children of an unavailable root still belong to one reply tree.
            root_id = relation["root_id"]
            if root_id:
                first = unresolved_root_first.setdefault(root_id, child)
                uf.union(first, child)
        # Inferred context edges are soft evidence.  Decode them only after all
        # platform-native relations have formed hard components, and never let
        # them override component identity conflicts.
        accepted_soft_edges, rejected_soft_edges = _decode_soft_context_edges(
            chat_rows,
            context_edges_by_chat.get(chat_id, []),
            uf,
        )
        soft_context_edges_accepted += sum(edge.get("soft_edge_status") == "accepted" for edge in accepted_soft_edges)
        soft_context_edges_redundant += sum(edge.get("soft_edge_status") == "redundant" for edge in accepted_soft_edges)
        soft_context_edges_rejected += len(rejected_soft_edges)
        for edge in rejected_soft_edges:
            for reason in edge.get("soft_edge_reasons") or []:
                soft_context_rejection_counts[str(reason)] += 1
        for edge in accepted_soft_edges:
            source = _text(edge.get("source"))
            target = _text(edge.get("target"))
            if source in id_set and target in id_set:
                referenced_ids.update((source, target))
        components: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in chat_rows:
            components[uf.find(_text(row.get("message_id")))].append(row)
        reference_components = {
            component_id: component
            for component_id, component in components.items()
            if len(component) > 1 or any(_text(row.get("message_id")) in referenced_ids for row in component)
        }
        reference_component_count += len(reference_components)

        # Start with temporal blocks, then union blocks touched by a reference
        # component. This keeps unrelated chat traffic separated while making
        # reply chains authoritative.
        blocks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous_time: datetime | None = None
        for row in chat_rows:
            current_time = _parse_time(row.get("create_time"))
            break_block = bool(current and (
                len(current) >= max_messages
                or (previous_time is not None and current_time is not None and (current_time - previous_time).total_seconds() > quiet_gap_hours * 3600)
            ))
            if break_block:
                blocks.append(current)
                current = []
            current.append(row)
            if current_time is not None:
                previous_time = current_time
        if current:
            blocks.append(current)

        component_for_message = {
            _text(row.get("message_id")): component_id
            for component_id, component in reference_components.items()
            for row in component
        }
        report_component_ids = {
            component_id
            for component_id, component in reference_components.items()
            if any(_text(row.get("message_id")) in report_ids for row in component)
        }
        session_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for component_id, component in reference_components.items():
            session_buckets[f"ref:{component_id}"].extend(component)

        for block_index, block in enumerate(blocks):
            touched_components = sorted({
                component_for_message[_text(row.get("message_id"))]
                for row in block
                if _text(row.get("message_id")) in component_for_message
            })
            isolated = [row for row in block if _text(row.get("message_id")) not in component_for_message]
            if not isolated:
                if len(touched_components) > 1:
                    parallel_temporal_blocks_split += 1
                continue
            if not touched_components:
                session_buckets[f"time:{block_index}"].extend(isolated)
                isolated_session_count += 1
                continue
            if len(touched_components) == 1 and touched_components[0] not in report_component_ids:
                session_buckets[f"ref:{touched_components[0]}"].extend(isolated)
                isolated_messages_attached += len(isolated)
                continue

            if len(touched_components) == 1 and touched_components[0] in report_component_ids:
                session_buckets[f"time:{block_index}:around_report"].extend(isolated)
                isolated_session_count += 1
                continue

            parallel_temporal_blocks_split += 1
            component_times: dict[str, list[datetime]] = {}
            for component_id in touched_components:
                component_times[component_id] = [
                    parsed
                    for row in reference_components[component_id]
                    for parsed in [_parse_time(row.get("create_time"))]
                    if parsed is not None
                ]
            unassigned: list[dict[str, Any]] = []
            for row in isolated:
                row_time = _parse_time(row.get("create_time"))
                candidates: list[tuple[float, str]] = []
                if row_time is not None:
                    for component_id, times in component_times.items():
                        if times:
                            candidates.append((min(abs((row_time - value).total_seconds()) for value in times), component_id))
                candidates.sort()
                if candidates and candidates[0][0] <= context_attach_minutes * 60:
                    session_buckets[f"ref:{candidates[0][1]}"].append(row)
                    isolated_messages_attached += 1
                else:
                    unassigned.append(row)
            if unassigned:
                session_buckets[f"time:{block_index}:unassigned"].extend(unassigned)
                isolated_session_count += 1

        merged_blocks = [
            sorted(block, key=lambda row: (_text(row.get("create_time")), _text(row.get("message_id"))))
            for block in session_buckets.values()
            if block
        ]
        merged_blocks.sort(key=lambda block: (_text(block[0].get("create_time")), _text(block[0].get("message_id"))))
        for index, block in enumerate(merged_blocks, 1):
            segment_id = _time_segment_id(chat_id, block, index)
            session_count += 1
            for row in block:
                previous_thread_id = row.get("thread_id")
                row["thread_id"] = segment_id
                row["relation_aware_session_id"] = segment_id
                raw = dict(row.get("raw") or {})
                raw["relation_aware_session_id"] = segment_id
                raw["source_thread_id_before_relation_merge"] = previous_thread_id
                row["raw"] = raw
                for link in row.get("links") or []:
                    if isinstance(link, dict):
                        link.setdefault("source_thread_id", link.get("thread_id") or previous_thread_id or "")
                        link["thread_id"] = segment_id
                for attachment in row.get("attachments") or []:
                    if isinstance(attachment, dict):
                        attachment.setdefault("source_thread_id", attachment.get("thread_id") or previous_thread_id or "")
                        attachment["thread_id"] = segment_id
                assigned.append(row)
    assigned.sort(key=lambda row: (_text(row.get("create_time")), _text(row.get("message_id"))))
    report = {
        "messages": len(assigned),
        "chat_count": len(by_chat),
        "reference_component_count": reference_component_count,
        "relation_aware_session_count": session_count,
        "parallel_temporal_blocks_split": parallel_temporal_blocks_split,
        "isolated_messages_attached": isolated_messages_attached,
        "isolated_session_count": isolated_session_count,
        "quiet_gap_hours": quiet_gap_hours,
        "max_messages": max_messages,
        "context_attach_minutes": context_attach_minutes,
        "context_continuation_edge_count": len(context_edge_rows),
        "soft_context_edge_count": len(context_edge_rows),
        "soft_context_edges_accepted": soft_context_edges_accepted,
        "soft_context_edges_rejected": soft_context_edges_rejected,
        "soft_context_edges_redundant": soft_context_edges_redundant,
        "soft_context_rejection_counts": dict(sorted(soft_context_rejection_counts.items())),
        "native_edges_are_hard": True,
        "inferred_edges_are_soft": True,
        "cross_window_trace_continuation_edge_count": sum(
            str(edge.get("type") or "") == "cross_window_trace_continuation" for edge in context_edge_rows
        ),
    }
    return assigned, report


def write_relation_run(out_dir: str | Path, result: dict[str, Any]) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "messages.jsonl").open("w", encoding="utf-8") as handle:
        for row in result.get("messages", []):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("thread_summaries", "episodes", "reference_graph", "run_manifest"):
        (out / f"{name}.json").write_text(json.dumps(result.get(name) or {}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {name: str(out / f"{name}.json") for name in ("thread_summaries", "episodes", "reference_graph", "run_manifest")}
