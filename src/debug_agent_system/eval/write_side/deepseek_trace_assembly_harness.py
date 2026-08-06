"""Evidence-bounded DeepSeek harness for chat case decomposition and Trace assembly.

This module deliberately lives in ``eval.write_side`` first.  It does not
replace W7 and it cannot promote knowledge.  The harness:

1. rebuilds a source ledger from the latest W1 ``messages.jsonl``;
2. asks DeepSeek to decompose that ledger into atomic case items;
3. asks DeepSeek to assemble only those case items into longitudinal traces;
4. validates every reference locally and fails closed;
5. freezes input/output hashes and renders an auditable report.

The model never receives human annotations or gold labels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable

from debug_agent_system.agents.write.w2_extract.deepseek_client import (
    call_json_object,
    call_strict_tool,
    configured_model,
)


SCHEMA_VERSION = "debug_agent_system.deepseek_trace_assembly_harness.v4"
PROMPT_VERSION = "deepseek-trace-assembly-harness-v4"
VALIDATOR_VERSION = "deepseek-trace-assembly-validator-v2"
DEFAULT_MESSAGES = Path(
    "data/results/xing_relation_context_payload_candidate_20260724/messages.jsonl"
)
DEFAULT_OUT = Path("data/results/deepseek_trace_assembly_harness_current")

CASE_KINDS = (
    "diagnostic_case",
    "algorithm_data_request",
    "configuration_issue",
    "operator_error",
    "positive_validation",
    "product_requirement",
    "jira_status_update",
    "field_work_report",
    "coordination_only",
    "noise",
)
TRACE_CASE_KINDS = {
    "diagnostic_case",
    "algorithm_data_request",
    "configuration_issue",
    "operator_error",
}
ASSEMBLY_CASE_KINDS = TRACE_CASE_KINDS | {
    "jira_status_update",
    "positive_validation",
}
RESOLUTION_STATUSES = (
    "unknown",
    "pending",
    "investigating",
    "provisionally_resolved",
    "ineffective",
    "recurrence",
    "verified",
)
EVENT_TYPES = (
    "report",
    "diagnosis",
    "action",
    "short_term_recovery",
    "recurrence",
    "resolution",
    "validation",
)
RELATION_TYPES = (
    "trace_root",
    "continuation_of",
    "diagnosis_of",
    "action_for",
    "recurrence_of",
    "validation_of",
)

FAULT_SIGNAL_RE = re.compile(
    r"(失败|报错|异常|蓝屏|花屏|闪退|卡顿|漏检|误报|不拍|不出板|断连|超时|"
    r"连接不上|连不上|无法|不能|丢失|消失|不准|失真|崩溃|无响应|"
    r"错件|错位|缺件|虚焊|连锡|"
    r"HTTP\\s*500|APPCRASH|AppHang|MEMORY_MANAGEMENT)",
    re.IGNORECASE,
)
SUCCESS_SIGNAL_RE = re.compile(
    r"(正常生产|恢复正常|验证正常|未再出现|不再出现|正常使用|已解决|已修复|"
    r"问题解决|后解决|解决了|测试正常|顺利生产|正常检出)",
    re.IGNORECASE,
)
TEMPORARY_SIGNAL_RE = re.compile(
    r"(暂时|目前|持续观察|继续观察|待观察|短期|偶发|重启后|未反馈)",
    re.IGNORECASE,
)


ToolCaller = Callable[..., dict[str, Any]]
JsonCaller = Callable[..., dict[str, Any]]


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def decomposition_tool_schema() -> dict[str, Any]:
    """Strict schema for atomic case-item decomposition."""

    case_item = _strict_object({
        "case_item_ref": {"type": "string"},
        "case_kind": {"type": "string", "enum": list(CASE_KINDS)},
        "title": {"type": "string"},
        "problem_summary": {"type": "string"},
        "device_scope": {"type": "string"},
        "time_span": {"type": "string"},
        "source_message_ids": {"type": "array", "items": {"type": "string"}},
        "attachment_message_ids": {"type": "array", "items": {"type": "string"}},
        "jira_keys": {"type": "array", "items": {"type": "string"}},
        "duplicate_report_message_ids": {"type": "array", "items": {"type": "string"}},
        "requires_trace_assembly": {"type": "boolean"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "decompose_chat_into_atomic_case_items",
            "description": (
                "Split a complete field-support chat evidence ledger into atomic "
                "fault cases, requirements, validations, status updates, work reports, and noise."
            ),
            "strict": True,
            "parameters": _strict_object({
                "case_items": {"type": "array", "items": case_item},
                "unassigned_message_ids": {"type": "array", "items": {"type": "string"}},
                "global_uncertainties": {"type": "array", "items": {"type": "string"}},
            }),
        },
    }


def assembly_tool_schema() -> dict[str, Any]:
    """Strict schema for assembling validated case items into traces."""

    phase = _strict_object({
        "phase_index": {"type": "integer", "minimum": 1},
        "case_item_ref": {"type": "string"},
        "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
        "relation_type": {"type": "string", "enum": list(RELATION_TYPES)},
        "summary": {"type": "string"},
        "evidence_message_ids": {"type": "array", "items": {"type": "string"}},
    })
    trace = _strict_object({
        "trace_ref": {"type": "string"},
        "title": {"type": "string"},
        "device_scope": {"type": "string"},
        "case_item_refs": {"type": "array", "items": {"type": "string"}},
        "phases": {"type": "array", "items": phase},
        "resolution_status": {"type": "string", "enum": list(RESOLUTION_STATUSES)},
        "resolution_evidence_message_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "link_reasons": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    })
    cannot_link = _strict_object({
        "left_case_item_ref": {"type": "string"},
        "right_case_item_ref": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "assemble_longitudinal_fault_traces",
            "description": (
                "Assemble atomic case items into evidence-bounded longitudinal fault traces "
                "with phases, recurrence, status, and cannot-link decisions."
            ),
            "strict": True,
            "parameters": _strict_object({
                "traces": {"type": "array", "items": trace},
                "standalone_case_item_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cannot_link_pairs": {"type": "array", "items": cannot_link},
                "global_uncertainties": {"type": "array", "items": {"type": "string"}},
            }),
        },
    }


def neighbor_selection_tool_schema() -> dict[str, Any]:
    """Strict schema for selecting longitudinally relevant neighbor cases."""

    selected_link = _strict_object({
        "neighbor_case_item_ref": {"type": "string"},
        "related_core_case_item_refs": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasons": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "select_neighbor_cases_for_trace_assembly",
            "description": (
                "Select only neighbor-window case items that may continue, precede, "
                "or resolve a core-session case item."
            ),
            "strict": True,
            "parameters": _strict_object({
                "selected_links": {"type": "array", "items": selected_link},
                "excluded_neighbor_case_item_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "global_uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }),
        },
    }


DECOMPOSITION_PROMPT = """\
你是工业现场群聊写侧的原子 Case Item 分解器。输入是程序从最新 W1 messages.jsonl
重建的完整证据账本，不是旧 episode 摘要。

必须遵守：
1. 先按业务问题拆原子 case item，再考虑 Trace；同一条日报必须把其中每个故障、诊断、
   动作和恢复分别回连或拆成诊断类 item，不能只生成一个 field_work_report 把故障包住。
2. 首报、诊断包、回复讨论、日报复述、Jira 状态和结果可以属于同一 item，日报复述不是新问题。
3. 区分 diagnostic_case、算法数据需求、配置问题、人员漏判、正向验证、产品需求、
   Jira 状态、普通现场工作、协调和噪声。
4. “设备已检出但人员未看出”是 operator_error/positive_validation，不是设备漏检。
5. 花屏、蓝屏、闪退等词冲突时，以原始首报、附件、专项诊断和 Jira 为优先证据，
   在 uncertainties 记录冲突，不擅自规范成另一类故障。
6. file/image/video 消息即使没有文字也可能是诊断起点；必须结合相邻回复和附件元数据。
7. source_message_ids 只能引用 allowed_message_ids，不得编造；保留跨日复发与最终验证证据。
8. 纯培训、致谢、成员变更、一般生产播报不得伪造成故障。
9. 不预设 case 数量；证据不足就记录 uncertainties。
10. 同一条日报消息可同时被一个 field_work_report 和多个原子诊断 item 引用；这不是重复，
    是为了保留日报中各业务问题的独立 Trace 归属。

严格调用指定工具，不输出解释文字。"""


NEIGHBOR_SELECTION_PROMPT = """\
你是工业现场故障 Trace 的邻域证据筛选器。核心 case items 来自本次待审核 source
session；邻域 case items 来自前后时间窗口，只用于找跨 session 的首报、复发、后续诊断、
动作或最终验证。

必须遵守：
1. 核心 items 永远保留；你只能决定哪些邻域 item 与至少一个核心 item 可能属于同一业务 Trace。
2. 只有同一设备/产线且故障、动作链、Jira、附件诊断、人员上下文或时间延续存在具体联系时才选择。
3. 同群、同日或同一日报不是充分联系；同时处理的不同故障必须排除。
4. 邻域 item 可以是核心故障的更早首报，也可以是更晚复发、诊断、动作或验证。
5. related_core_case_item_refs 是兼容字段名：优先引用 allowed_core_refs；若一个邻域 item
   只能通过另一个已选择邻域 item 回连核心，也可引用 allowed_neighbor_refs，但被引用邻域
   item 必须同样被选择，且整条链最终必须连到核心；不得形成脱离核心的孤立环或发明 ref。
6. 每个邻域 ref 必须且只能出现一次：要么进入 selected_links，要么进入
   excluded_neighbor_case_item_refs。证据不足时排除，并可写入 global_uncertainties。
7. 选择阶段只判断“值得进入最终 Trace 裁决”，不直接断言根因、解决状态或合并结果。

严格调用指定工具，不输出解释文字。"""


ASSEMBLY_PROMPT = """\
你是工业现场故障 Trace 组装器。上游已经给出原子 case items；你只能基于这些 item
和 evidence ledger 进行 Trace 决策，不能发明新消息、动作、原因或结果。

必须遵守：
1. 同一设备、同一故障的首报、诊断、动作、短期恢复、复发和最终验证应合成一个 Trace，
   不按日期、日报、refseg 或 source session 机械拆分。
2. 不同设备或不同故障必须 cannot-link；“同时讨论”不等于同一 Trace。
3. Jira 状态更新、日报复述和附件不是独立 Trace，应回连其业务问题。
4. product_requirement、positive_validation、field_work_report、coordination_only、noise
   通常放 standalone，不得进入诊断执行 Trace。
5. 短期“重启后正常/目前未复发”只能是 provisionally_resolved；后续复发必须把状态修订为
   recurrence/ineffective，不能继续保留 verified。
6. verified 必须有明确、可引用的生产恢复或验证正常证据；计划、建议、Jira Done 本身不足以
   证明现场最终恢复。
7. phases 必须按时间和因果顺序排列；同一 case item 不得进入多个 Trace。
8. 对不确定链接保守处理并记录 uncertainties/cannot-link，不得为了减少 Trace 数量强行合并。
9. case items 是分块产生的阶段性片段，不是最终 Trace；首报、深入诊断、日报复述和次日动作
   经常是多个 item。不得机械地“一 item 一 Trace”，应把跨块延续放进同一 Trace。
10. 后续 item 缺少设备编号不等于设备不同；只有明确的不同设备证据才 cannot-link。若故障关键词、
    时间、参与人和动作链一致，应合并并把设备不确定性写入 uncertainties。
11. 同日同设备的“花屏首报/蓝屏日报”等术语改写，如果恢复动作、Jira 或上下文一致，应合并为
    同一 Trace 并保留词语冲突；不能仅因归一化名称不同而拆 Trace。
12. 输入中的 link_candidates 是本地高精度待裁决对。每一对必须合进同一 Trace，或写入
    cannot_link_pairs 并给出具体反证；不得忽略。

严格调用指定工具，不输出解释文字。"""


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a potentially large source file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


def _load_env(path: Path) -> None:
    """Load simple KEY=VALUE secrets without echoing them."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _source_thread_ids(row: dict[str, Any]) -> set[str]:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    values = raw.get("source_thread_ids") or []
    ids = {str(value) for value in values if value}
    for key in ("segment_id", "source_thread_id_before_relation_merge"):
        if raw.get(key):
            ids.add(str(raw[key]))
    return ids


def _attachment_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "file_key": str(item.get("file_key") or ""),
            "name": str(item.get("name") or ""),
            "kind": str(item.get("kind") or ""),
            "extension": str(item.get("extension") or ""),
            "size": int(item.get("size") or 0),
            "status": str(item.get("status") or ""),
            "source_status": str(item.get("source_status") or ""),
            "evidence_role": str(item.get("evidence_role") or ""),
            "payload_sha256": str(item.get("payload_sha256") or ""),
        }
        for item in row.get("attachments") or []
        if isinstance(item, dict)
    ]


def _ledger_row(row: dict[str, Any], *, region: str) -> dict[str, Any]:
    sender = row.get("sender") if isinstance(row.get("sender"), dict) else {}
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    text = str(row.get("text") or "")
    return {
        "message_id": str(row.get("message_id") or ""),
        "create_time": str(row.get("create_time") or ""),
        "sender": str(sender.get("name") or ""),
        "msg_type": str(row.get("msg_type") or ""),
        "text": text[:8_000],
        "root_id": str(row.get("root_id") or ""),
        "parent_id": str(row.get("parent_id") or ""),
        "relation_aware_session_id": str(row.get("thread_id") or ""),
        "source_thread_ids": sorted(_source_thread_ids(row)),
        "region": region,
        "attachments": _attachment_rows(row),
        "links": [
            {
                "url": str(item.get("url") or ""),
                "label": str(item.get("label") or ""),
            }
            for item in row.get("links") or []
            if isinstance(item, dict)
        ],
        "semantic_fragments": [
            {
                "fragment_id": str(item.get("fragment_id") or ""),
                "text": str(item.get("text") or "")[:2_000],
            }
            for item in row.get("semantic_fragments") or []
            if isinstance(item, dict)
        ],
        "chat_name": str(raw.get("chat_name") or raw.get("v3_chat_name") or ""),
    }


def build_source_ledger(
    messages_path: str | Path,
    source_thread_id: str,
    *,
    neighbor_days: int = 14,
    max_rows: int = 1_500,
) -> dict[str, Any]:
    """Rebuild a complete core source ledger plus same-chat temporal neighbors."""

    messages_path = Path(messages_path)
    core: list[dict[str, Any]] = []
    with messages_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"messages_jsonl_invalid_line:{line_number}:{exc}") from exc
            if not isinstance(row, dict) or not row.get("message_id"):
                continue
            if source_thread_id in _source_thread_ids(row):
                core.append(row)
    if not core:
        raise ValueError(f"source_thread_not_found:{source_thread_id}")

    chat_ids = {str(row.get("chat_id") or "") for row in core if row.get("chat_id")}
    if len(chat_ids) != 1:
        raise ValueError(f"source_thread_chat_id_ambiguous:{sorted(chat_ids)}")
    chat_id = next(iter(chat_ids))
    core_times = [_parse_time(str(row.get("create_time") or "")) for row in core]
    known_times = [value for value in core_times if value is not None]
    if not known_times:
        raise ValueError("source_thread_missing_times")
    core_start = min(known_times)
    core_end = max(known_times)
    window_start = core_start - timedelta(days=max(0, int(neighbor_days)))
    window_end = core_end + timedelta(days=max(0, int(neighbor_days)))

    selected: dict[str, tuple[dict[str, Any], str]] = {}
    core_ids = {str(row.get("message_id") or "") for row in core}
    with messages_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"messages_jsonl_invalid_line:{line_number}:{exc}") from exc
            if (
                not isinstance(row, dict)
                or not row.get("message_id")
                or str(row.get("chat_id") or "") != chat_id
            ):
                continue
            message_id = str(row.get("message_id") or "")
            created = _parse_time(str(row.get("create_time") or ""))
            if message_id in core_ids:
                selected[message_id] = (row, "core")
            elif created is not None and window_start <= created <= window_end:
                selected.setdefault(message_id, (row, "neighbor"))

    ordered = sorted(
        selected.values(),
        key=lambda item: (
            str(item[0].get("create_time") or ""),
            str(item[0].get("message_id") or ""),
        ),
    )
    if len(ordered) > int(max_rows):
        raise ValueError(
            f"source_window_too_large:{len(ordered)}>{int(max_rows)};"
            "refuse_silent_truncation"
        )
    rows = [_ledger_row(row, region=region) for row, region in ordered]
    allowed_ids = [row["message_id"] for row in rows]
    core_allowed_ids = [row["message_id"] for row in rows if row["region"] == "core"]
    attachment_count = sum(len(row["attachments"]) for row in rows)
    orphan_attachment_candidates = [
        row["message_id"]
        for row in rows
        if row["attachments"] and not row["text"].strip()
    ]
    high_signal_core_ids = [
        row["message_id"]
        for row in rows
        if row["region"] == "core"
        and (
            bool(FAULT_SIGNAL_RE.search(row["text"]))
            or any(
                item["evidence_role"] in {"log_package", "diagnostic_screenshot", "log_screenshot"}
                for item in row["attachments"]
            )
        )
    ]
    ledger = {
        "schema_version": "debug_agent_system.trace_source_ledger.v1",
        "source_thread_id": source_thread_id,
        "chat_id": chat_id,
        "chat_name": next((row["chat_name"] for row in rows if row["chat_name"]), ""),
        "core_start": core_start.strftime("%Y-%m-%d %H:%M"),
        "core_end": core_end.strftime("%Y-%m-%d %H:%M"),
        "window_start": window_start.strftime("%Y-%m-%d %H:%M"),
        "window_end": window_end.strftime("%Y-%m-%d %H:%M"),
        "neighbor_days": int(neighbor_days),
        "rows": rows,
        "allowed_message_ids": allowed_ids,
        "core_message_ids": core_allowed_ids,
        "high_signal_core_message_ids": high_signal_core_ids,
        "stats": {
            "rows": len(rows),
            "core_rows": len(core_allowed_ids),
            "neighbor_rows": len(rows) - len(core_allowed_ids),
            "attachments": attachment_count,
            "orphan_attachment_candidates": len(orphan_attachment_candidates),
        },
        "orphan_attachment_candidate_message_ids": orphan_attachment_candidates,
    }
    ledger["ledger_sha256"] = canonical_hash(ledger)
    return ledger


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value or "")))


def validate_decomposition(
    raw: dict[str, Any],
    *,
    allowed_message_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize and validate a decomposition without inventing references."""

    issues: list[str] = []
    items: list[dict[str, Any]] = []
    refs: set[str] = set()
    raw_items = raw.get("case_items") if isinstance(raw, dict) else None
    if not isinstance(raw_items, list):
        return {"case_items": [], "unassigned_message_ids": []}, ["case_items_not_list"]
    for index, value in enumerate(raw_items):
        if not isinstance(value, dict):
            issues.append(f"case_items[{index}]:not_object")
            continue
        item = dict(value)
        ref = str(item.get("case_item_ref") or "")
        if not ref:
            issues.append(f"case_items[{index}]:missing_ref")
            continue
        if ref in refs:
            issues.append(f"case_items[{index}]:duplicate_ref:{ref}")
            continue
        refs.add(ref)
        kind = str(item.get("case_kind") or "")
        if kind not in CASE_KINDS:
            issues.append(f"case_items[{index}]:invalid_kind:{kind}")
        evidence = _dedupe_strings(item.get("source_message_ids") or [])
        attachments = _dedupe_strings(item.get("attachment_message_ids") or [])
        for evidence_id in [*evidence, *attachments]:
            if evidence_id not in allowed_message_ids:
                issues.append(f"case_items[{index}]:unknown_message_id:{evidence_id}")
        evidence = [value for value in evidence if value in allowed_message_ids]
        attachments = [value for value in attachments if value in allowed_message_ids]
        if kind not in {"noise", "coordination_only"} and not evidence:
            issues.append(f"case_items[{index}]:missing_evidence")
        item["source_message_ids"] = evidence
        item["attachment_message_ids"] = attachments
        item["jira_keys"] = _dedupe_strings(item.get("jira_keys") or [])
        item["duplicate_report_message_ids"] = [
            value
            for value in _dedupe_strings(item.get("duplicate_report_message_ids") or [])
            if value in allowed_message_ids
        ]
        item["uncertainties"] = _dedupe_strings(item.get("uncertainties") or [])
        item["requires_trace_assembly"] = bool(item.get("requires_trace_assembly"))
        items.append(item)
    unassigned = _dedupe_strings(raw.get("unassigned_message_ids") or [])
    for evidence_id in unassigned:
        if evidence_id not in allowed_message_ids:
            issues.append(f"unassigned_unknown_message_id:{evidence_id}")
    normalized = {
        "case_items": items,
        "unassigned_message_ids": [
            value for value in unassigned if value in allowed_message_ids
        ],
        "global_uncertainties": _dedupe_strings(raw.get("global_uncertainties") or []),
    }
    return normalized, sorted(set(issues))


def validate_neighbor_selection(
    raw: dict[str, Any],
    *,
    allowed_core_refs: set[str],
    allowed_neighbor_refs: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate complete, exclusive accounting of neighbor case refs."""

    issues: list[str] = []
    selected_links: list[dict[str, Any]] = []
    selected_refs: set[str] = set()
    raw_links = raw.get("selected_links") if isinstance(raw, dict) else None
    if not isinstance(raw_links, list):
        raw_links = []
        issues.append("selected_links_not_list")
    for index, value in enumerate(raw_links):
        if not isinstance(value, dict):
            issues.append(f"selected_links[{index}]:not_object")
            continue
        neighbor_ref = str(value.get("neighbor_case_item_ref") or "")
        if neighbor_ref not in allowed_neighbor_refs:
            issues.append(
                f"selected_links[{index}]:unknown_neighbor_ref:{neighbor_ref}"
            )
            continue
        if neighbor_ref in selected_refs:
            issues.append(
                f"selected_links[{index}]:duplicate_neighbor_ref:{neighbor_ref}"
            )
            continue
        selected_refs.add(neighbor_ref)
        core_refs = _dedupe_strings(value.get("related_core_case_item_refs") or [])
        allowed_related_refs = allowed_core_refs | allowed_neighbor_refs
        unknown_core_refs = [
            ref for ref in core_refs if ref not in allowed_related_refs
        ]
        for core_ref in unknown_core_refs:
            issues.append(
                f"selected_links[{index}]:unknown_core_ref:{core_ref}"
            )
        core_refs = [ref for ref in core_refs if ref in allowed_related_refs]
        if not core_refs:
            issues.append(f"selected_links[{index}]:missing_related_core_ref")
        reasons = _dedupe_strings(value.get("reasons") or [])
        if not reasons:
            issues.append(f"selected_links[{index}]:missing_reasons")
        selected_links.append({
            "neighbor_case_item_ref": neighbor_ref,
            "related_core_case_item_refs": core_refs,
            "reasons": reasons,
        })

    excluded = _dedupe_strings(
        raw.get("excluded_neighbor_case_item_refs") or []
        if isinstance(raw, dict) else []
    )
    excluded_refs: set[str] = set()
    for ref in excluded:
        if ref not in allowed_neighbor_refs:
            issues.append(f"excluded_unknown_neighbor_ref:{ref}")
            continue
        if ref in selected_refs:
            issues.append(f"neighbor_ref_selected_and_excluded:{ref}")
            continue
        excluded_refs.add(ref)
    accounted = selected_refs | excluded_refs
    for ref in sorted(allowed_neighbor_refs - accounted):
        issues.append(f"neighbor_ref_unaccounted:{ref}")
    relations = {
        value["neighbor_case_item_ref"]: set(
            value["related_core_case_item_refs"]
        )
        for value in selected_links
    }
    for neighbor_ref, related_refs in relations.items():
        for related_ref in sorted(related_refs & allowed_neighbor_refs):
            if related_ref not in selected_refs:
                issues.append(
                    "selected_neighbor_points_to_unselected_neighbor:"
                    f"{neighbor_ref}:{related_ref}"
                )
    reachable = set(allowed_core_refs)
    changed = True
    while changed:
        changed = False
        for neighbor_ref, related_refs in relations.items():
            if neighbor_ref not in reachable and related_refs & reachable:
                reachable.add(neighbor_ref)
                changed = True
    for ref in sorted(selected_refs - reachable):
        issues.append(f"selected_neighbor_not_connected_to_core:{ref}")
    normalized = {
        "selected_links": selected_links,
        "excluded_neighbor_case_item_refs": sorted(excluded_refs),
        "global_uncertainties": _dedupe_strings(
            raw.get("global_uncertainties") or []
            if isinstance(raw, dict) else []
        ),
    }
    return normalized, sorted(set(issues))


def _evidence_text_index(ledger: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("message_id") or ""): str(row.get("text") or "")
        for row in ledger.get("rows") or []
        if isinstance(row, dict)
    }


def validate_assembly(
    raw: dict[str, Any],
    *,
    decomposition: dict[str, Any],
    ledger: dict[str, Any],
    link_candidates: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate trace membership, phases, evidence, and terminal status."""

    issues: list[str] = []
    allowed_message_ids = set(ledger.get("allowed_message_ids") or [])
    case_items = {
        str(item.get("case_item_ref") or ""): item
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict) and item.get("case_item_ref")
    }
    evidence_text = _evidence_text_index(ledger)
    traces: list[dict[str, Any]] = []
    trace_refs: set[str] = set()
    assigned: dict[str, str] = {}
    raw_traces = raw.get("traces") if isinstance(raw, dict) else None
    if not isinstance(raw_traces, list):
        return {"traces": []}, ["traces_not_list"]
    for index, value in enumerate(raw_traces):
        if not isinstance(value, dict):
            issues.append(f"traces[{index}]:not_object")
            continue
        trace = dict(value)
        ref = str(trace.get("trace_ref") or "")
        if not ref:
            issues.append(f"traces[{index}]:missing_ref")
            continue
        if ref in trace_refs:
            issues.append(f"traces[{index}]:duplicate_ref:{ref}")
            continue
        trace_refs.add(ref)
        members = _dedupe_strings(trace.get("case_item_refs") or [])
        for member in members:
            if member not in case_items:
                issues.append(f"traces[{index}]:unknown_case_item:{member}")
            elif member in assigned:
                issues.append(
                    f"traces[{index}]:case_item_multiple_traces:{member}:"
                    f"{assigned[member]}:{ref}"
                )
            else:
                assigned[member] = ref
        members = [value for value in members if value in case_items]
        if not members:
            issues.append(f"traces[{index}]:missing_case_items")
        phases: list[dict[str, Any]] = []
        for phase_index, raw_phase in enumerate(trace.get("phases") or []):
            if not isinstance(raw_phase, dict):
                issues.append(f"traces[{index}].phases[{phase_index}]:not_object")
                continue
            phase = dict(raw_phase)
            member = str(phase.get("case_item_ref") or "")
            if member not in members:
                issues.append(
                    f"traces[{index}].phases[{phase_index}]:case_item_not_in_trace:{member}"
                )
            evidence = _dedupe_strings(phase.get("evidence_message_ids") or [])
            for evidence_id in evidence:
                if evidence_id not in allowed_message_ids:
                    issues.append(
                        f"traces[{index}].phases[{phase_index}]:unknown_message_id:{evidence_id}"
                    )
            phase["evidence_message_ids"] = [
                value for value in evidence if value in allowed_message_ids
            ]
            phases.append(phase)
        expected = list(range(1, len(phases) + 1))
        actual = [int(item.get("phase_index") or 0) for item in phases]
        if actual != expected:
            issues.append(f"traces[{index}]:non_contiguous_phases:{actual}")
        resolution_status = str(trace.get("resolution_status") or "")
        resolution_ids = _dedupe_strings(
            trace.get("resolution_evidence_message_ids") or []
        )
        for evidence_id in resolution_ids:
            if evidence_id not in allowed_message_ids:
                issues.append(
                    f"traces[{index}]:unknown_resolution_message_id:{evidence_id}"
                )
        resolution_ids = [
            value for value in resolution_ids if value in allowed_message_ids
        ]
        resolution_text = " ".join(evidence_text.get(value, "") for value in resolution_ids)
        resolution_phase_text = " ".join(
            str(phase.get("summary") or "")
            for phase in phases
            if (
                not resolution_ids
                or set(phase.get("evidence_message_ids") or []) & set(resolution_ids)
            )
        )
        resolution_decision_text = " ".join(
            value for value in (resolution_phase_text, resolution_text) if value
        )
        phase_has_explicit_success = bool(
            SUCCESS_SIGNAL_RE.search(resolution_phase_text)
        )
        if resolution_status == "verified":
            if not resolution_ids:
                issues.append(f"traces[{index}]:verified_without_evidence")
            elif not SUCCESS_SIGNAL_RE.search(resolution_decision_text):
                issues.append(f"traces[{index}]:verified_without_explicit_success_signal")
            if (
                TEMPORARY_SIGNAL_RE.search(resolution_phase_text)
                or (
                    not phase_has_explicit_success
                    and TEMPORARY_SIGNAL_RE.search(resolution_text)
                )
            ):
                issues.append(f"traces[{index}]:verified_from_temporary_signal")
        if any(
            phase.get("event_type") == "recurrence"
            for phase in phases
        ) and resolution_status == "verified":
            last_success_position = max(
                (
                    pos
                    for pos, phase in enumerate(phases)
                    if phase.get("event_type") in {"resolution", "validation"}
                ),
                default=-1,
            )
            last_recurrence_position = max(
                (
                    pos
                    for pos, phase in enumerate(phases)
                    if phase.get("event_type") == "recurrence"
                ),
                default=-1,
            )
            if last_recurrence_position > last_success_position:
                issues.append(f"traces[{index}]:verified_before_latest_recurrence")
        trace["case_item_refs"] = members
        trace["phases"] = phases
        trace["resolution_evidence_message_ids"] = resolution_ids
        trace["link_reasons"] = _dedupe_strings(trace.get("link_reasons") or [])
        trace["uncertainties"] = _dedupe_strings(trace.get("uncertainties") or [])
        traces.append(trace)

    standalone = _dedupe_strings(raw.get("standalone_case_item_refs") or [])
    for ref in standalone:
        if ref not in case_items:
            issues.append(f"standalone_unknown_case_item:{ref}")
        if ref in assigned:
            issues.append(f"standalone_also_assigned:{ref}")
    standalone = [value for value in standalone if value in case_items]
    cannot_link_pairs: list[dict[str, Any]] = []
    for index, value in enumerate(raw.get("cannot_link_pairs") or []):
        if not isinstance(value, dict):
            issues.append(f"cannot_link_pairs[{index}]:not_object")
            continue
        left = str(value.get("left_case_item_ref") or "")
        right = str(value.get("right_case_item_ref") or "")
        if left not in case_items or right not in case_items or left == right:
            issues.append(f"cannot_link_pairs[{index}]:invalid_pair:{left}:{right}")
            continue
        if assigned.get(left) and assigned.get(left) == assigned.get(right):
            issues.append(
                f"cannot_link_pairs[{index}]:pair_linked_in_same_trace:{left}:{right}"
            )
        cannot_link_pairs.append({
            "left_case_item_ref": left,
            "right_case_item_ref": right,
            "reasons": _dedupe_strings(value.get("reasons") or []),
        })
    explicit_cannot_links = {
        frozenset((
            value["left_case_item_ref"],
            value["right_case_item_ref"],
        ))
        for value in cannot_link_pairs
    }
    for candidate in link_candidates or []:
        left = str(candidate.get("left_case_item_ref") or "")
        right = str(candidate.get("right_case_item_ref") or "")
        if left not in case_items or right not in case_items:
            continue
        if assigned.get(left) and assigned.get(left) == assigned.get(right):
            continue
        if frozenset((left, right)) in explicit_cannot_links:
            continue
        issues.append(f"link_candidate_unresolved:{left}:{right}")
    for ref, item in case_items.items():
        kind = str(item.get("case_kind") or "")
        if kind in TRACE_CASE_KINDS and item.get("requires_trace_assembly"):
            if ref not in assigned and ref not in standalone:
                issues.append(f"trace_case_item_unaccounted:{ref}")
    normalized = {
        "traces": traces,
        "standalone_case_item_refs": standalone,
        "cannot_link_pairs": cannot_link_pairs,
        "global_uncertainties": _dedupe_strings(raw.get("global_uncertainties") or []),
    }
    return normalized, sorted(set(issues))


def coverage_metrics(
    ledger: dict[str, Any],
    decomposition: dict[str, Any],
    assembly: dict[str, Any],
) -> dict[str, Any]:
    """Compute source-coverage and trace-accounting diagnostics."""

    assigned_messages = {
        str(message_id)
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict)
        for message_id in item.get("source_message_ids") or []
    }
    diagnostic_assigned_messages = {
        str(message_id)
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict)
        and str(item.get("case_kind") or "") in TRACE_CASE_KINDS
        for message_id in item.get("source_message_ids") or []
    }
    high_signal = set(ledger.get("high_signal_core_message_ids") or [])
    covered_high_signal = high_signal & assigned_messages
    diagnostic_covered_high_signal = high_signal & diagnostic_assigned_messages
    trace_members = {
        str(ref)
        for trace in assembly.get("traces") or []
        if isinstance(trace, dict)
        for ref in trace.get("case_item_refs") or []
    }
    standalone = set(assembly.get("standalone_case_item_refs") or [])
    all_case_refs = {
        str(item.get("case_item_ref") or "")
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict)
    }
    return {
        "core_high_signal_total": len(high_signal),
        "core_high_signal_assigned": len(covered_high_signal),
        "core_high_signal_coverage": (
            round(len(covered_high_signal) / len(high_signal), 4)
            if high_signal else 1.0
        ),
        "uncovered_high_signal_message_ids": sorted(high_signal - assigned_messages),
        "core_high_signal_diagnostic_assigned": len(diagnostic_covered_high_signal),
        "core_high_signal_diagnostic_coverage": (
            round(len(diagnostic_covered_high_signal) / len(high_signal), 4)
            if high_signal else 1.0
        ),
        "high_signal_not_in_diagnostic_case_message_ids": sorted(
            high_signal - diagnostic_assigned_messages
        ),
        "case_item_total": len(all_case_refs),
        "case_items_in_trace": len(trace_members),
        "case_items_standalone": len(standalone),
        "unaccounted_case_item_refs": sorted(all_case_refs - trace_members - standalone),
        "orphan_attachment_candidate_message_ids": list(
            ledger.get("orphan_attachment_candidate_message_ids") or []
        ),
    }


def partition_source_ledger(
    ledger: dict[str, Any],
    *,
    max_rows: int = 80,
    max_chars: int = 160_000,
) -> list[dict[str, Any]]:
    """Partition an evidence ledger for atomic extraction without dropping rows.

    Chunks are non-overlapping.  Cross-chunk continuity is intentionally left
    to the global assembly stage, which sees every extracted case item.
    """

    rows = [row for row in ledger.get("rows") or [] if isinstance(row, dict)]
    if not rows:
        return [dict(ledger)]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    row_limit = max(1, int(max_rows))
    char_limit = max(8_000, int(max_chars))
    for row in rows:
        row_chars = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        if current and (
            len(current) >= row_limit or current_chars + row_chars > char_limit
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars
    if current:
        chunks.append(current)

    core_ids = set(ledger.get("core_message_ids") or [])
    high_signal_ids = set(ledger.get("high_signal_core_message_ids") or [])
    orphan_ids = set(ledger.get("orphan_attachment_candidate_message_ids") or [])
    output: list[dict[str, Any]] = []
    for index, chunk_rows in enumerate(chunks, 1):
        allowed_ids = [str(row.get("message_id") or "") for row in chunk_rows]
        allowed_set = set(allowed_ids)
        chunk = {
            key: value
            for key, value in ledger.items()
            if key not in {
                "rows",
                "allowed_message_ids",
                "core_message_ids",
                "high_signal_core_message_ids",
                "orphan_attachment_candidate_message_ids",
                "stats",
                "ledger_sha256",
            }
        }
        chunk.update({
            "chunk_index": index,
            "chunk_count": len(chunks),
            "rows": chunk_rows,
            "allowed_message_ids": allowed_ids,
            "core_message_ids": [
                value for value in allowed_ids if value in core_ids
            ],
            "high_signal_core_message_ids": [
                value for value in allowed_ids if value in high_signal_ids
            ],
            "orphan_attachment_candidate_message_ids": [
                value for value in allowed_ids if value in orphan_ids
            ],
            "stats": {
                "rows": len(chunk_rows),
                "core_rows": sum(value in core_ids for value in allowed_ids),
                "neighbor_rows": sum(value not in core_ids for value in allowed_ids),
                "attachments": sum(
                    len(row.get("attachments") or []) for row in chunk_rows
                ),
            },
        })
        chunk["ledger_sha256"] = canonical_hash(chunk)
        output.append(chunk)
    return output


def harness_config() -> dict[str, Any]:
    return {
        "chunk_max_rows": int(
            os.environ.get("DEEPSEEK_TRACE_CHUNK_MAX_ROWS", "30")
        ),
        "chunk_max_chars": int(
            os.environ.get("DEEPSEEK_TRACE_CHUNK_MAX_CHARS", "50000")
        ),
        "min_high_signal_coverage": float(
            os.environ.get("DEEPSEEK_TRACE_MIN_HIGH_SIGNAL_COVERAGE", "0.90")
        ),
        "assembly_component_threshold": int(
            os.environ.get("DEEPSEEK_TRACE_ASSEMBLY_COMPONENT_THRESHOLD", "40")
        ),
    }


def _namespace_decomposition(
    decomposition: dict[str, Any],
    *,
    chunk_index: int,
    namespace: bool,
) -> dict[str, Any]:
    if not namespace:
        return decomposition
    prefix = f"C{chunk_index:02d}-"
    items: list[dict[str, Any]] = []
    for raw_item in decomposition.get("case_items") or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item["case_item_ref"] = prefix + str(item.get("case_item_ref") or "")
        items.append(item)
    return {
        "case_items": items,
        "unassigned_message_ids": list(
            decomposition.get("unassigned_message_ids") or []
        ),
        "global_uncertainties": [
            f"chunk_{chunk_index}:{value}"
            for value in decomposition.get("global_uncertainties") or []
        ],
    }


def _compact_assembly_evidence(
    ledger: dict[str, Any],
    decomposition: dict[str, Any],
) -> list[dict[str, Any]]:
    referenced_ids = {
        str(message_id)
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict)
        for field in ("source_message_ids", "attachment_message_ids")
        for message_id in item.get(field) or []
    }
    compact: list[dict[str, Any]] = []
    for row in ledger.get("rows") or []:
        if (
            not isinstance(row, dict)
            or str(row.get("message_id") or "") not in referenced_ids
        ):
            continue
        compact.append({
            "message_id": str(row.get("message_id") or ""),
            "create_time": str(row.get("create_time") or ""),
            "sender": str(row.get("sender") or ""),
            "text": str(row.get("text") or "")[:2_000],
            "region": str(row.get("region") or ""),
            "attachments": row.get("attachments") or [],
        })
    return compact


def _compact_case_items_for_selection(
    items: Iterable[dict[str, Any]],
    *,
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the selector input small while retaining temporal/evidence anchors."""

    compact: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        message_ids = _dedupe_strings([
            *(item.get("source_message_ids") or []),
            *(item.get("attachment_message_ids") or []),
        ])
        evidence = []
        for message_id in message_ids:
            row = rows_by_id.get(message_id) or {}
            evidence.append({
                "message_id": message_id,
                "create_time": str(row.get("create_time") or ""),
                "text": str(row.get("text") or "")[:500],
                "attachment_names": [
                    str(value.get("name") or "")
                    for value in row.get("attachments") or []
                    if isinstance(value, dict)
                ],
            })
        compact.append({
            "case_item_ref": str(item.get("case_item_ref") or ""),
            "case_kind": str(item.get("case_kind") or ""),
            "title": str(item.get("title") or ""),
            "problem_summary": str(item.get("problem_summary") or "")[:1_000],
            "device_scope": str(item.get("device_scope") or ""),
            "time_span": str(item.get("time_span") or ""),
            "jira_keys": _dedupe_strings(item.get("jira_keys") or []),
            "evidence": evidence,
        })
    return compact


def build_link_candidates(
    decomposition: dict[str, Any],
    ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build high-precision identity candidates for DeepSeek to adjudicate."""

    rows_by_id = {
        str(row.get("message_id") or ""): row
        for row in ledger.get("rows") or []
        if isinstance(row, dict)
    }
    prepared: list[dict[str, Any]] = []
    for item in decomposition.get("case_items") or []:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("case_item_ref") or "")
        content = " ".join(
            str(item.get(key) or "")
            for key in ("title", "problem_summary", "device_scope")
        )
        dates = {
            str((rows_by_id.get(str(message_id)) or {}).get("create_time") or "")[:10]
            for message_id in item.get("source_message_ids") or []
            if (rows_by_id.get(str(message_id)) or {}).get("create_time")
        }
        anchors = set(re.findall(
            r"(?:[A-Z]{2,}-\d+|HTTP\s*(?:STATUS)?\s*:?\s*\d{3}|F\d{1,2})",
            content.upper(),
        ))
        prepared.append({
            "ref": ref,
            "kind": str(item.get("case_kind") or ""),
            "content": content,
            "dates": dates,
            "anchors": anchors,
        })
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(prepared):
        for right in prepared[left_index + 1:]:
            if not left["ref"] or not right["ref"]:
                continue
            reasons: list[str] = []
            shared_anchors = left["anchors"] & right["anchors"]
            if shared_anchors:
                reasons.append(
                    "shared_distinctive_anchors:" + ",".join(sorted(shared_anchors))
                )
            screen_alias = (
                ("花屏" in left["content"] and "蓝屏" in right["content"])
                or ("蓝屏" in left["content"] and "花屏" in right["content"])
            )
            if (
                screen_alias
                and left["dates"] & right["dates"]
                and "重启" in left["content"]
                and "重启" in right["content"]
            ):
                reasons.append("same_day_screen_term_conflict_with_same_recovery")
            if not reasons:
                continue
            candidates.append({
                "left_case_item_ref": left["ref"],
                "right_case_item_ref": right["ref"],
                "reasons": reasons,
            })
    return candidates


def partition_assembly_components(
    case_items: list[dict[str, Any]],
    *,
    neighbor_selection: dict[str, Any],
    link_candidates: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Partition a large assembly request by already-audited relation edges."""

    refs = [
        str(item.get("case_item_ref") or "")
        for item in case_items
        if item.get("case_item_ref")
    ]
    item_by_ref = {
        str(item.get("case_item_ref") or ""): item
        for item in case_items
        if item.get("case_item_ref")
    }
    parent = {ref: ref for ref in refs}

    def find(ref: str) -> str:
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            return
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for value in neighbor_selection.get("selected_links") or []:
        if not isinstance(value, dict):
            continue
        neighbor_ref = str(value.get("neighbor_case_item_ref") or "")
        for related_ref in value.get("related_core_case_item_refs") or []:
            union(neighbor_ref, str(related_ref))
    for value in link_candidates:
        union(
            str(value.get("left_case_item_ref") or ""),
            str(value.get("right_case_item_ref") or ""),
        )

    grouped: dict[str, list[str]] = {}
    for ref in refs:
        grouped.setdefault(find(ref), []).append(ref)
    components = list(grouped.values())
    isolated = [component[0] for component in components if len(component) == 1]
    components = [component for component in components if len(component) > 1]
    if isolated:
        components.append(isolated)
    order = {ref: index for index, ref in enumerate(refs)}
    components.sort(key=lambda values: min(order[value] for value in values))
    return [
        [item_by_ref[ref] for ref in sorted(values, key=order.__getitem__)]
        for values in components
    ]


def _call_with_repair(
    *,
    caller: ToolCaller,
    api_key: str,
    system_prompt: str,
    base_payload: dict[str, Any],
    tool: dict[str, Any],
    validator: Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]],
    max_tokens: int,
    user_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    raw: dict[str, Any] = {}
    normalized: dict[str, Any] = {}
    issues: list[str] = []
    calls: list[dict[str, Any]] = []
    for semantic_attempt in range(1, 3):
        payload = dict(base_payload)
        if issues:
            payload["repair_request"] = {
                "instruction": (
                    "只修复本地校验列出的问题；不得新增输入证据之外的事实或ID。"
                ),
                "issues": issues[:80],
            }
        response = caller(
            api_key=api_key,
            system_prompt=system_prompt,
            user_payload=payload,
            tool=tool,
            max_tokens=max_tokens,
            timeout_seconds=float(os.environ.get("DEEPSEEK_TRACE_TIMEOUT", "360")),
            max_attempts=int(
                os.environ.get("DEEPSEEK_TRACE_TRANSPORT_ATTEMPTS", "4")
            ),
            user_id=user_id,
        )
        raw = response.get("arguments") if isinstance(response.get("arguments"), dict) else {}
        calls.append({
            **{key: value for key, value in response.items() if key != "arguments"},
            "semantic_attempt": semantic_attempt,
        })
        normalized, issues = validator(raw)
        if not issues:
            break
    return raw, normalized, issues, calls


def _stage_cache_key(*, stage: str, input_hash: str) -> str:
    return canonical_hash({
        "stage": stage,
        "input_hash": input_hash,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model": configured_model(),
    })


def _read_stage_cache(
    path: Path | None,
    *,
    cache_key: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]] | None:
    if path is None or not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(cached, dict)
        or cached.get("cache_key") != cache_key
        or cached.get("issues")
    ):
        return None
    raw = cached.get("raw")
    normalized = cached.get("normalized")
    calls = cached.get("calls")
    if not isinstance(raw, dict) or not isinstance(normalized, dict):
        return None
    cached_calls = calls if isinstance(calls, list) else []
    return raw, normalized, [], [
        {"stage_cache_hit": True, **call}
        for call in cached_calls
        if isinstance(call, dict)
    ]


def _write_stage_cache(
    path: Path | None,
    *,
    cache_key: str,
    raw: dict[str, Any],
    normalized: dict[str, Any],
    issues: list[str],
    calls: list[dict[str, Any]],
) -> None:
    if path is None or issues:
        return
    _atomic_write_json(path, {
        "schema_version": "debug_agent_system.trace_assembly_stage_cache.v1",
        "cache_key": cache_key,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model": configured_model(),
        "raw": raw,
        "normalized": normalized,
        "issues": issues,
        "calls": calls,
    })


def _slice_source_ledger(
    ledger: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    adaptive_path: str,
) -> dict[str, Any]:
    allowed_ids = [str(row.get("message_id") or "") for row in rows]
    allowed_set = set(allowed_ids)
    sliced = {
        key: value
        for key, value in ledger.items()
        if key not in {
            "rows",
            "allowed_message_ids",
            "core_message_ids",
            "high_signal_core_message_ids",
            "orphan_attachment_candidate_message_ids",
            "stats",
            "ledger_sha256",
        }
    }
    sliced.update({
        "adaptive_path": adaptive_path,
        "rows": rows,
        "allowed_message_ids": allowed_ids,
        "core_message_ids": [
            value
            for value in ledger.get("core_message_ids") or []
            if value in allowed_set
        ],
        "high_signal_core_message_ids": [
            value
            for value in ledger.get("high_signal_core_message_ids") or []
            if value in allowed_set
        ],
        "orphan_attachment_candidate_message_ids": [
            value
            for value in ledger.get("orphan_attachment_candidate_message_ids") or []
            if value in allowed_set
        ],
        "stats": {
            "rows": len(rows),
            "attachments": sum(len(row.get("attachments") or []) for row in rows),
        },
    })
    sliced["ledger_sha256"] = canonical_hash(sliced)
    return sliced


def _prefix_case_item_refs(
    decomposition: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    output = dict(decomposition)
    items: list[dict[str, Any]] = []
    for raw_item in decomposition.get("case_items") or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item["case_item_ref"] = prefix + str(item.get("case_item_ref") or "")
        items.append(item)
    output["case_items"] = items
    return output


def _decompose_with_adaptive_split(
    *,
    chunk: dict[str, Any],
    caller: ToolCaller,
    api_key: str,
    user_id: str,
    adaptive_path: str = "root",
    minimum_rows: int = 8,
    json_caller: JsonCaller = call_json_object,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    allowed = set(chunk.get("allowed_message_ids") or [])
    try:
        return _call_with_repair(
            caller=caller,
            api_key=api_key,
            system_prompt=DECOMPOSITION_PROMPT,
            base_payload={
                "source_ledger": chunk,
                "allowed_message_ids": sorted(allowed),
            },
            tool=decomposition_tool_schema(),
            validator=lambda raw: validate_decomposition(
                raw, allowed_message_ids=allowed
            ),
            max_tokens=int(
                os.environ.get("DEEPSEEK_TRACE_DECOMPOSE_MAX_TOKENS", "32768")
            ),
            user_id=f"{user_id}_{adaptive_path}",
        )
    except Exception:
        rows = [
            row for row in chunk.get("rows") or []
            if isinstance(row, dict)
        ]
        if len(rows) <= max(2, int(minimum_rows)):
            response = json_caller(
                api_key=api_key,
                system_prompt=(
                    DECOMPOSITION_PROMPT
                    + "\nTool Call持续格式失败，改用JSON Output。必须严格输出"
                    "required_output_schema对应的对象，不输出Markdown。"
                ),
                user_payload={
                    "source_ledger": chunk,
                    "allowed_message_ids": sorted(allowed),
                    "required_output_schema": (
                        decomposition_tool_schema()["function"]["parameters"]
                    ),
                },
                max_tokens=int(
                    os.environ.get(
                        "DEEPSEEK_TRACE_DECOMPOSE_MAX_TOKENS", "32768"
                    )
                ),
                timeout_seconds=float(
                    os.environ.get("DEEPSEEK_TRACE_TIMEOUT", "360")
                ),
                max_attempts=int(
                    os.environ.get("DEEPSEEK_TRACE_TRANSPORT_ATTEMPTS", "4")
                ),
                user_id=f"{user_id}_{adaptive_path}_json_fallback",
            )
            raw = (
                response.get("arguments")
                if isinstance(response.get("arguments"), dict) else {}
            )
            normalized, issues = validate_decomposition(
                raw,
                allowed_message_ids=allowed,
            )
            return raw, normalized, issues, [{
                **{
                    key: value
                    for key, value in response.items()
                    if key != "arguments"
                },
                "json_output_fallback": True,
                "semantic_attempt": 1,
            }]
        midpoint = len(rows) // 2
        child_outputs = []
        for suffix, child_rows in (("a", rows[:midpoint]), ("b", rows[midpoint:])):
            child_path = f"{adaptive_path}{suffix}"
            child_chunk = _slice_source_ledger(
                chunk,
                child_rows,
                adaptive_path=child_path,
            )
            child_outputs.append((
                suffix,
                *_decompose_with_adaptive_split(
                    chunk=child_chunk,
                    caller=caller,
                    api_key=api_key,
                    user_id=user_id,
                    adaptive_path=child_path,
                    minimum_rows=minimum_rows,
                    json_caller=json_caller,
                ),
            ))
        raw_children: list[dict[str, Any]] = []
        combined_items: list[dict[str, Any]] = []
        combined_unassigned: list[str] = []
        combined_uncertainties: list[str] = []
        combined_issues: list[str] = []
        combined_calls: list[dict[str, Any]] = []
        for suffix, raw, normalized, issues, calls in child_outputs:
            prefix = f"{suffix.upper()}-"
            normalized = _prefix_case_item_refs(normalized, prefix=prefix)
            raw_children.append({
                "adaptive_part": suffix,
                "raw": raw,
            })
            combined_items.extend(normalized.get("case_items") or [])
            combined_unassigned.extend(
                normalized.get("unassigned_message_ids") or []
            )
            combined_uncertainties.extend(
                f"adaptive_{suffix}:{value}"
                for value in normalized.get("global_uncertainties") or []
            )
            combined_issues.extend(
                f"adaptive_{suffix}:{value}" for value in issues
            )
            combined_calls.extend(
                {"adaptive_part": suffix, **call} for call in calls
            )
        return (
            {"adaptive_split": True, "children": raw_children},
            {
                "case_items": combined_items,
                "unassigned_message_ids": _dedupe_strings(combined_unassigned),
                "global_uncertainties": _dedupe_strings(combined_uncertainties),
            },
            combined_issues,
            combined_calls,
        )


def run_harness(
    ledger: dict[str, Any],
    *,
    api_key: str,
    caller: ToolCaller = call_strict_tool,
    stage_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run both model stages and apply all local gates."""

    allowed = set(ledger.get("allowed_message_ids") or [])
    config = harness_config()
    chunks = partition_source_ledger(
        ledger,
        max_rows=config["chunk_max_rows"],
        max_chars=config["chunk_max_chars"],
    )
    decomposition_raw_chunks: list[dict[str, Any]] = []
    decomposition_calls: list[dict[str, Any]] = []
    decomposition_issues: list[str] = []
    decomposition_items: list[dict[str, Any]] = []
    decomposition_unassigned: list[str] = []
    decomposition_uncertainties: list[str] = []
    namespace = len(chunks) > 1
    for chunk_index, chunk in enumerate(chunks, 1):
        cache_path = (
            stage_cache_dir / f"decomposition_chunk_{chunk_index:03d}.json"
            if stage_cache_dir is not None else None
        )
        cache_key = _stage_cache_key(
            stage=f"decomposition_chunk_{chunk_index}",
            input_hash=str(chunk.get("ledger_sha256") or canonical_hash(chunk)),
        )
        cached_stage = _read_stage_cache(cache_path, cache_key=cache_key)
        if cached_stage is not None:
            raw, normalized, issues, calls = cached_stage
        else:
            try:
                raw, normalized, issues, calls = _decompose_with_adaptive_split(
                    chunk=chunk,
                    caller=caller,
                    api_key=api_key,
                    user_id=f"debug_agent_trace_decomposition_{chunk_index}",
                )
            except Exception as exc:  # noqa: BLE001 - add auditable stage context
                raise RuntimeError(
                    f"decomposition_chunk_{chunk_index}_of_{len(chunks)}:"
                    f"{type(exc).__name__}:{exc}"
                ) from exc
            _write_stage_cache(
                cache_path,
                cache_key=cache_key,
                raw=raw,
                normalized=normalized,
                issues=issues,
                calls=calls,
            )
        normalized = _namespace_decomposition(
            normalized,
            chunk_index=chunk_index,
            namespace=namespace,
        )
        decomposition_raw_chunks.append({
            "chunk_index": chunk_index,
            "chunk_ledger_sha256": chunk.get("ledger_sha256"),
            "output": raw,
        })
        decomposition_calls.extend(
            {"chunk_index": chunk_index, **call} for call in calls
        )
        decomposition_issues.extend(
            f"chunk_{chunk_index}:{issue}" for issue in issues
        )
        decomposition_items.extend(normalized.get("case_items") or [])
        decomposition_unassigned.extend(
            normalized.get("unassigned_message_ids") or []
        )
        decomposition_uncertainties.extend(
            normalized.get("global_uncertainties") or []
        )
    decomposition = {
        "case_items": decomposition_items,
        "unassigned_message_ids": _dedupe_strings(decomposition_unassigned),
        "global_uncertainties": _dedupe_strings(decomposition_uncertainties),
    }
    decomposition_raw: dict[str, Any] = {
        "chunk_count": len(chunks),
        "chunks": decomposition_raw_chunks,
    }
    all_assembly_case_items = [
        item
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict)
        and str(item.get("case_kind") or "") in ASSEMBLY_CASE_KINDS
    ]
    core_message_ids = set(ledger.get("core_message_ids") or [])
    core_assembly_case_items = [
        item
        for item in all_assembly_case_items
        if core_message_ids & set(_dedupe_strings([
            *(item.get("source_message_ids") or []),
            *(item.get("attachment_message_ids") or []),
        ]))
    ]
    neighbor_assembly_case_items = [
        item
        for item in all_assembly_case_items
        if item not in core_assembly_case_items
    ]
    neighbor_selection_raw: dict[str, Any] = {}
    neighbor_selection = {
        "selected_links": [],
        "excluded_neighbor_case_item_refs": [],
        "global_uncertainties": [],
    }
    neighbor_selection_issues: list[str] = []
    neighbor_selection_calls: list[dict[str, Any]] = []
    selected_neighbor_refs: set[str] = set()
    excluded_neighbor_refs: list[str] = []
    if core_assembly_case_items and neighbor_assembly_case_items:
        rows_by_id = {
            str(row.get("message_id") or ""): row
            for row in ledger.get("rows") or []
            if isinstance(row, dict)
        }
        core_selection_items = _compact_case_items_for_selection(
            core_assembly_case_items,
            rows_by_id=rows_by_id,
        )
        neighbor_selection_items = _compact_case_items_for_selection(
            neighbor_assembly_case_items,
            rows_by_id=rows_by_id,
        )
        allowed_core_refs = {
            str(item.get("case_item_ref") or "")
            for item in core_assembly_case_items
            if item.get("case_item_ref")
        }
        allowed_neighbor_refs = {
            str(item.get("case_item_ref") or "")
            for item in neighbor_assembly_case_items
            if item.get("case_item_ref")
        }
        selection_input_hash = canonical_hash({
            "source_thread_id": ledger.get("source_thread_id"),
            "core_case_items": core_selection_items,
            "neighbor_case_items": neighbor_selection_items,
        })
        selection_cache_path = (
            stage_cache_dir / "neighbor_selection.json"
            if stage_cache_dir is not None else None
        )
        selection_cache_key = _stage_cache_key(
            stage="neighbor_selection",
            input_hash=selection_input_hash,
        )
        cached_selection = _read_stage_cache(
            selection_cache_path,
            cache_key=selection_cache_key,
        )
        if cached_selection is not None:
            (
                neighbor_selection_raw,
                neighbor_selection,
                neighbor_selection_issues,
                neighbor_selection_calls,
            ) = cached_selection
        else:
            try:
                (
                    neighbor_selection_raw,
                    neighbor_selection,
                    neighbor_selection_issues,
                    neighbor_selection_calls,
                ) = _call_with_repair(
                    caller=caller,
                    api_key=api_key,
                    system_prompt=NEIGHBOR_SELECTION_PROMPT,
                    base_payload={
                        "source_thread_id": ledger.get("source_thread_id"),
                        "core_case_items": core_selection_items,
                        "neighbor_case_items": neighbor_selection_items,
                        "allowed_core_refs": sorted(allowed_core_refs),
                        "allowed_neighbor_refs": sorted(allowed_neighbor_refs),
                    },
                    tool=neighbor_selection_tool_schema(),
                    validator=lambda raw: validate_neighbor_selection(
                        raw,
                        allowed_core_refs=allowed_core_refs,
                        allowed_neighbor_refs=allowed_neighbor_refs,
                    ),
                    max_tokens=int(
                        os.environ.get(
                            "DEEPSEEK_TRACE_SELECTION_MAX_TOKENS", "16384"
                        )
                    ),
                    user_id="debug_agent_trace_neighbor_selection",
                )
            except Exception as exc:  # noqa: BLE001 - auditable stage context
                raise RuntimeError(
                    f"neighbor_selection:{type(exc).__name__}:{exc}"
                ) from exc
            _write_stage_cache(
                selection_cache_path,
                cache_key=selection_cache_key,
                raw=neighbor_selection_raw,
                normalized=neighbor_selection,
                issues=neighbor_selection_issues,
                calls=neighbor_selection_calls,
            )
        selected_neighbor_refs = {
            str(value.get("neighbor_case_item_ref") or "")
            for value in neighbor_selection.get("selected_links") or []
            if isinstance(value, dict)
        }
        excluded_neighbor_refs = _dedupe_strings(
            neighbor_selection.get("excluded_neighbor_case_item_refs") or []
        )
    else:
        selected_neighbor_refs = {
            str(item.get("case_item_ref") or "")
            for item in neighbor_assembly_case_items
            if item.get("case_item_ref")
        }
    assembly_case_items = [
        *core_assembly_case_items,
        *[
            item
            for item in neighbor_assembly_case_items
            if str(item.get("case_item_ref") or "") in selected_neighbor_refs
        ],
    ]
    assembly_decomposition = {
        "case_items": assembly_case_items,
        "unassigned_message_ids": [],
        "global_uncertainties": [],
    }
    preclassified_standalone_refs = [
        str(item.get("case_item_ref") or "")
        for item in decomposition.get("case_items") or []
        if isinstance(item, dict)
        and str(item.get("case_kind") or "") not in ASSEMBLY_CASE_KINDS
        and item.get("case_item_ref")
    ]
    preclassified_standalone_refs = _dedupe_strings([
        *preclassified_standalone_refs,
        *excluded_neighbor_refs,
    ])
    link_candidates = build_link_candidates(assembly_decomposition, ledger)
    assembly_raw: dict[str, Any] = {}
    assembly: dict[str, Any] = {"traces": []}
    assembly_issues: list[str] = []
    assembly_calls: list[dict[str, Any]] = []
    if not decomposition_issues and not neighbor_selection_issues:
        assembly_evidence = _compact_assembly_evidence(
            ledger, assembly_decomposition
        )
        assembly_input_hash = canonical_hash({
            "source_thread_id": ledger.get("source_thread_id"),
            "decomposition": assembly_decomposition,
            "evidence": assembly_evidence,
            "link_candidates": link_candidates,
            "preclassified_standalone_refs": preclassified_standalone_refs,
        })
        assembly_cache_path = (
            stage_cache_dir / "global_assembly.json"
            if stage_cache_dir is not None else None
        )
        assembly_cache_key = _stage_cache_key(
            stage="global_assembly",
            input_hash=assembly_input_hash,
        )
        cached_assembly = _read_stage_cache(
            assembly_cache_path,
            cache_key=assembly_cache_key,
        )
        if cached_assembly is not None:
            (
                assembly_raw,
                assembly,
                assembly_issues,
                assembly_calls,
            ) = cached_assembly
        else:
            try:
                component_threshold = max(
                    1, int(config["assembly_component_threshold"])
                )
                if len(assembly_case_items) <= component_threshold:
                    assembly_raw, assembly, assembly_issues, assembly_calls = (
                        _call_with_repair(
                            caller=caller,
                            api_key=api_key,
                            system_prompt=ASSEMBLY_PROMPT,
                            base_payload={
                                "source_thread_id": ledger.get("source_thread_id"),
                                "case_items": assembly_case_items,
                                "evidence_ledger": assembly_evidence,
                                "allowed_message_ids": sorted(allowed),
                                "decomposition_chunk_count": len(chunks),
                                "link_candidates": link_candidates,
                            },
                            tool=assembly_tool_schema(),
                            validator=lambda raw: validate_assembly(
                                raw,
                                decomposition=assembly_decomposition,
                                ledger=ledger,
                                link_candidates=link_candidates,
                            ),
                            max_tokens=int(
                                os.environ.get(
                                    "DEEPSEEK_TRACE_ASSEMBLY_MAX_TOKENS", "32768"
                                )
                            ),
                            user_id="debug_agent_trace_assembly",
                        )
                    )
                else:
                    components = partition_assembly_components(
                        assembly_case_items,
                        neighbor_selection=neighbor_selection,
                        link_candidates=link_candidates,
                    )
                    component_raw_outputs: list[dict[str, Any]] = []
                    combined_traces: list[dict[str, Any]] = []
                    combined_standalone: list[str] = []
                    combined_cannot_links: list[dict[str, Any]] = []
                    combined_uncertainties: list[str] = []
                    component_issues: list[str] = []
                    for component_index, component_items in enumerate(
                        components, 1
                    ):
                        component_refs = {
                            str(item.get("case_item_ref") or "")
                            for item in component_items
                        }
                        component_decomposition = {
                            "case_items": component_items,
                            "unassigned_message_ids": [],
                            "global_uncertainties": [],
                        }
                        component_evidence = _compact_assembly_evidence(
                            ledger, component_decomposition
                        )
                        component_links = [
                            value
                            for value in link_candidates
                            if (
                                str(value.get("left_case_item_ref") or "")
                                in component_refs
                                and str(value.get("right_case_item_ref") or "")
                                in component_refs
                            )
                        ]
                        component_input_hash = canonical_hash({
                            "source_thread_id": ledger.get("source_thread_id"),
                            "component_index": component_index,
                            "decomposition": component_decomposition,
                            "evidence": component_evidence,
                            "link_candidates": component_links,
                        })
                        component_cache_path = (
                            stage_cache_dir
                            / f"assembly_component_{component_index:03d}.json"
                            if stage_cache_dir is not None else None
                        )
                        component_cache_key = _stage_cache_key(
                            stage=f"assembly_component_{component_index}",
                            input_hash=component_input_hash,
                        )
                        cached_component = _read_stage_cache(
                            component_cache_path,
                            cache_key=component_cache_key,
                        )
                        if cached_component is not None:
                            raw_part, part, issues_part, calls_part = (
                                cached_component
                            )
                        else:
                            raw_part, part, issues_part, calls_part = (
                                _call_with_repair(
                                    caller=caller,
                                    api_key=api_key,
                                    system_prompt=ASSEMBLY_PROMPT,
                                    base_payload={
                                        "source_thread_id": ledger.get(
                                            "source_thread_id"
                                        ),
                                        "assembly_component_index": component_index,
                                        "assembly_component_count": len(components),
                                        "case_items": component_items,
                                        "evidence_ledger": component_evidence,
                                        "allowed_message_ids": sorted(allowed),
                                        "decomposition_chunk_count": len(chunks),
                                        "link_candidates": component_links,
                                    },
                                    tool=assembly_tool_schema(),
                                    validator=lambda raw, decomposition=(
                                        component_decomposition
                                    ), candidates=component_links: validate_assembly(
                                        raw,
                                        decomposition=decomposition,
                                        ledger=ledger,
                                        link_candidates=candidates,
                                    ),
                                    max_tokens=int(
                                        os.environ.get(
                                            "DEEPSEEK_TRACE_ASSEMBLY_MAX_TOKENS",
                                            "32768",
                                        )
                                    ),
                                    user_id=(
                                        "debug_agent_trace_assembly_component_"
                                        f"{component_index}"
                                    ),
                                )
                            )
                            _write_stage_cache(
                                component_cache_path,
                                cache_key=component_cache_key,
                                raw=raw_part,
                                normalized=part,
                                issues=issues_part,
                                calls=calls_part,
                            )
                        component_raw_outputs.append({
                            "component_index": component_index,
                            "case_item_refs": sorted(component_refs),
                            "output": raw_part,
                        })
                        for trace in part.get("traces") or []:
                            trace = dict(trace)
                            trace["trace_ref"] = (
                                f"A{component_index:02d}-"
                                + str(trace.get("trace_ref") or "")
                            )
                            combined_traces.append(trace)
                        combined_standalone.extend(
                            part.get("standalone_case_item_refs") or []
                        )
                        combined_cannot_links.extend(
                            part.get("cannot_link_pairs") or []
                        )
                        combined_uncertainties.extend(
                            f"component_{component_index}:{value}"
                            for value in part.get("global_uncertainties") or []
                        )
                        component_issues.extend(
                            f"component_{component_index}:{value}"
                            for value in issues_part
                        )
                        assembly_calls.extend(
                            {"component_index": component_index, **call}
                            for call in calls_part
                        )
                    assembly_raw = {
                        "component_count": len(components),
                        "components": component_raw_outputs,
                    }
                    combined_raw = {
                        "traces": combined_traces,
                        "standalone_case_item_refs": _dedupe_strings(
                            combined_standalone
                        ),
                        "cannot_link_pairs": combined_cannot_links,
                        "global_uncertainties": _dedupe_strings(
                            combined_uncertainties
                        ),
                    }
                    assembly, combined_issues = validate_assembly(
                        combined_raw,
                        decomposition=assembly_decomposition,
                        ledger=ledger,
                        link_candidates=link_candidates,
                    )
                    assembly_issues = sorted(set([
                        *component_issues,
                        *combined_issues,
                    ]))
            except Exception as exc:  # noqa: BLE001 - add auditable stage context
                raise RuntimeError(
                    f"global_assembly:{type(exc).__name__}:{exc}"
                ) from exc
            _write_stage_cache(
                assembly_cache_path,
                cache_key=assembly_cache_key,
                raw=assembly_raw,
                normalized=assembly,
                issues=assembly_issues,
                calls=assembly_calls,
            )
        assembly["standalone_case_item_refs"] = _dedupe_strings([
            *(assembly.get("standalone_case_item_refs") or []),
            *preclassified_standalone_refs,
        ])
    coverage = coverage_metrics(ledger, decomposition, assembly)
    quality_issues = [
        *[f"decomposition:{value}" for value in decomposition_issues],
        *[
            f"neighbor_selection:{value}"
            for value in neighbor_selection_issues
        ],
        *[f"assembly:{value}" for value in assembly_issues],
    ]
    min_coverage = config["min_high_signal_coverage"]
    if coverage["core_high_signal_coverage"] < min_coverage:
        quality_issues.append(
            "coverage:core_high_signal_below_threshold:"
            f"{coverage['core_high_signal_coverage']}<{min_coverage}"
        )
    if coverage["core_high_signal_diagnostic_coverage"] < min_coverage:
        quality_issues.append(
            "coverage:core_high_signal_diagnostic_below_threshold:"
            f"{coverage['core_high_signal_diagnostic_coverage']}<{min_coverage}"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "source_thread_id": ledger.get("source_thread_id"),
        "source_ledger_sha256": ledger.get("ledger_sha256"),
        "model": configured_model(),
        "source_only": True,
        "human_annotations_accessed": False,
        "harness_config": config,
        "decomposition_chunk_count": len(chunks),
        "decomposition": decomposition,
        "neighbor_selection": neighbor_selection,
        "link_candidates": link_candidates,
        "assembly": assembly,
        "coverage": coverage,
        "raw_model_outputs": {
            "decomposition": decomposition_raw,
            "neighbor_selection": neighbor_selection_raw,
            "assembly": assembly_raw,
        },
        "calls": {
            "decomposition": decomposition_calls,
            "neighbor_selection": neighbor_selection_calls,
            "assembly": assembly_calls,
        },
        "validation_issues": sorted(set(quality_issues)),
        "schema_valid": not quality_issues,
        "review_required": True,
        "promotion_allowed": False,
        "fallback_policy": "fail_closed_keep_current_w7",
    }
    result["result_sha256"] = canonical_hash(result)
    return result


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def render_markdown(ledger: dict[str, Any], result: dict[str, Any]) -> str:
    """Render a concise, evidence-linked review artifact."""

    rows_by_id = {
        str(row.get("message_id") or ""): row
        for row in ledger.get("rows") or []
        if isinstance(row, dict)
    }
    lines = [
        f"# DeepSeek Trace Assembly：`{ledger.get('source_thread_id')}`",
        "",
        "## Gate",
        "",
        f"- schema_valid: `{result.get('schema_valid')}`",
        f"- review_required: `{result.get('review_required')}`",
        f"- promotion_allowed: `{result.get('promotion_allowed')}`",
        f"- model: `{result.get('model')}`",
        f"- source rows: `{(ledger.get('stats') or {}).get('rows')}`",
        f"- core high-signal coverage: `{(result.get('coverage') or {}).get('core_high_signal_coverage')}`",
        "",
    ]
    issues = result.get("validation_issues") or []
    if issues:
        lines.extend(["## Validation issues", ""])
        lines.extend(f"- `{value}`" for value in issues)
        lines.append("")
    items = {
        str(item.get("case_item_ref") or ""): item
        for item in (result.get("decomposition") or {}).get("case_items") or []
        if isinstance(item, dict)
    }
    lines.extend(["## Atomic case items", ""])
    for ref, item in items.items():
        lines.extend([
            f"### `{ref}` · {item.get('title')}",
            "",
            f"- kind: `{item.get('case_kind')}`",
            f"- device: `{item.get('device_scope')}`",
            f"- summary: {item.get('problem_summary')}",
            f"- uncertainties: {item.get('uncertainties') or []}",
            "",
        ])
        for message_id in item.get("source_message_ids") or []:
            row = rows_by_id.get(str(message_id), {})
            lines.append(
                f"- `{message_id}` · {row.get('create_time')} · {row.get('sender')}: "
                f"{str(row.get('text') or '')[:500]}"
            )
        lines.append("")
    lines.extend(["## Assembled traces", ""])
    for trace in (result.get("assembly") or {}).get("traces") or []:
        lines.extend([
            f"### `{trace.get('trace_ref')}` · {trace.get('title')}",
            "",
            f"- case items: `{trace.get('case_item_refs')}`",
            f"- status: `{trace.get('resolution_status')}`",
            f"- link reasons: {trace.get('link_reasons') or []}",
            f"- uncertainties: {trace.get('uncertainties') or []}",
            "",
        ])
        for phase in trace.get("phases") or []:
            lines.append(
                f"{phase.get('phase_index')}. `{phase.get('event_type')}` / "
                f"`{phase.get('relation_type')}` / `{phase.get('case_item_ref')}` — "
                f"{phase.get('summary')}"
            )
        lines.append("")
    lines.extend([
        "## Coverage diagnostics",
        "",
        "```json",
        json.dumps(result.get("coverage") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
    ])
    return "\n".join(lines)


def run_one(
    *,
    messages_path: Path,
    source_thread_id: str,
    out_dir: Path,
    api_key: str,
    neighbor_days: int,
    force: bool = False,
    caller: ToolCaller = call_strict_tool,
) -> dict[str, Any]:
    ledger = build_source_ledger(
        messages_path,
        source_thread_id,
        neighbor_days=neighbor_days,
    )
    run_key = hashlib.sha256(source_thread_id.encode("utf-8")).hexdigest()[:12]
    result_path = out_dir / "results" / f"{run_key}.json"
    ledger_path = out_dir / "ledgers" / f"{run_key}.json"
    markdown_path = out_dir / "reviews" / f"{run_key}.md"
    if result_path.is_file() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("source_ledger_sha256") == ledger.get("ledger_sha256")
            and existing.get("prompt_version") == PROMPT_VERSION
            and existing.get("validator_version") == VALIDATOR_VERSION
            and existing.get("model") == configured_model()
            and existing.get("harness_config") == harness_config()
            and bool(existing.get("schema_valid"))
        ):
            return {
                "source_thread_id": source_thread_id,
                "status": "cache_hit",
                "result": str(result_path),
                "review": str(markdown_path),
                "schema_valid": bool(existing.get("schema_valid")),
                "trace_count": len((existing.get("assembly") or {}).get("traces") or []),
            }
    _atomic_write_json(ledger_path, ledger)
    result = run_harness(
        ledger,
        api_key=api_key,
        caller=caller,
        stage_cache_dir=out_dir / "stage_cache" / run_key,
    )
    _atomic_write_json(result_path, result)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(ledger, result), encoding="utf-8")
    return {
        "source_thread_id": source_thread_id,
        "status": "completed",
        "ledger": str(ledger_path),
        "result": str(result_path),
        "review": str(markdown_path),
        "schema_valid": bool(result.get("schema_valid")),
        "trace_count": len((result.get("assembly") or {}).get("traces") or []),
        "case_item_count": len((result.get("decomposition") or {}).get("case_items") or []),
        "coverage": (result.get("coverage") or {}).get("core_high_signal_coverage"),
        "result_sha256": result.get("result_sha256"),
    }


def run_batch(
    *,
    messages_path: Path,
    source_thread_ids: list[str],
    out_dir: Path,
    api_key: str,
    neighbor_days: int = 14,
    workers: int = 1,
    force: bool = False,
    caller: ToolCaller = call_strict_tool,
) -> dict[str, Any]:
    if not source_thread_ids:
        raise ValueError("missing_source_thread_ids")
    unique_ids = list(dict.fromkeys(source_thread_ids))
    results_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers)))) as executor:
        futures = {
            executor.submit(
                run_one,
                messages_path=messages_path,
                source_thread_id=source_thread_id,
                out_dir=out_dir,
                api_key=api_key,
                neighbor_days=neighbor_days,
                force=force,
                caller=caller,
            ): source_thread_id
            for source_thread_id in unique_ids
        }
        for future in as_completed(futures):
            source_thread_id = futures[future]
            try:
                results_by_id[source_thread_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - freeze per-session failure
                results_by_id[source_thread_id] = {
                    "source_thread_id": source_thread_id,
                    "status": "failed_closed",
                    "schema_valid": False,
                    "error": f"{type(exc).__name__}:{str(exc)[:2_000]}",
                }
            print(
                f"[trace-harness] {source_thread_id} "
                f"{results_by_id[source_thread_id].get('status')}",
                file=sys.stderr,
                flush=True,
            )
    results = [results_by_id[value] for value in unique_ids]
    manifest = {
        "schema_version": "debug_agent_system.deepseek_trace_assembly_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model": configured_model(),
        "harness_config": harness_config(),
        "messages_path": str(messages_path),
        "messages_file_sha256": file_sha256(messages_path),
        "neighbor_days": int(neighbor_days),
        "source_only": True,
        "human_annotations_accessed": False,
        "promotion_allowed": False,
        "results": results,
        "summary": {
            "sessions": len(results),
            "completed": sum(item.get("status") in {"completed", "cache_hit"} for item in results),
            "schema_valid": sum(bool(item.get("schema_valid")) for item in results),
            "needs_review": sum(
                item.get("status") in {"completed", "cache_hit"}
                and not bool(item.get("schema_valid"))
                for item in results
            ),
            "failed_closed": sum(item.get("status") == "failed_closed" for item in results),
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    _atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepseek-trace-assembly-harness")
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--source-thread-id", action="append", default=[])
    parser.add_argument("--source-thread-id-file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--neighbor-days", type=int, default=14)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    _load_env(args.env_file)
    source_thread_ids = list(args.source_thread_id)
    if args.source_thread_id_file:
        source_thread_ids.extend(
            line.strip()
            for line in args.source_thread_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        parser.error("missing DEEPSEEK_API_KEY; provide it via environment or --env-file")
    manifest = run_batch(
        messages_path=args.messages,
        source_thread_ids=source_thread_ids,
        out_dir=args.out_dir,
        api_key=api_key,
        neighbor_days=args.neighbor_days,
        workers=args.workers,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    summary = manifest["summary"]
    return (
        0
        if not summary["failed_closed"]
        and summary["schema_valid"] == summary["sessions"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
