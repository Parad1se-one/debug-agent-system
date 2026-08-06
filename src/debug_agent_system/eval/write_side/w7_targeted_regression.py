"""Targeted W7 regression over relation-aware W1 episodes.

The script intentionally samples risky session shapes instead of claiming a
full quality score: long reference chains, fragmented sessions, report
contamination, and suspicious resolution statements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from debug_agent_system.agents.write.review_context import (
    _w7_is_question_or_pending,
    refine_episode_group,
)


def iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            position = 0
            if not started:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    if eof:
                        return
                    buffer = buffer[position:]
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected JSON array in {path}")
                started = True
                position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) and buffer[position] == ",":
                    position += 1
                    continue
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if isinstance(item, dict):
                    yield item
                position = end
            buffer = buffer[position:]
            if eof:
                raise ValueError(f"truncated JSON array in {path}")


def parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def episode_texts(episode: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "noise_messages", "case_context_messages"):
        for message in episode.get(key) or []:
            if isinstance(message, dict):
                text = str(message.get("text") or message.get("content_summary") or "").strip()
                if text:
                    texts.append(text)
    return texts


def episode_resolution_texts(episode: dict[str, Any]) -> list[str]:
    return [
        str(message.get("text") or message.get("content_summary") or "")
        for message in episode.get("resolution_messages") or []
        if isinstance(message, dict)
    ]


def duration_hours(episode: dict[str, Any]) -> float:
    start = parse_time(episode.get("start_time"))
    end = parse_time(episode.get("end_time"))
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 3600.0)


def compact_before(episode: dict[str, Any]) -> dict[str, Any]:
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    return {
        "episode_id": episode.get("episode_id"),
        "completeness": episode.get("completeness"),
        "fault_focus": extracted.get("fault_focus_text") or extracted.get("symptom_raw") or "",
        "resolution_messages": episode_resolution_texts(episode),
        "field_report": bool(episode.get("field_report_anchor")),
        "message_count": int(episode.get("message_count") or len(episode_texts(episode))),
        "duration_hours": round(duration_hours(episode), 2),
    }


def compact_after(episode: dict[str, Any]) -> dict[str, Any]:
    cleanup = ((episode.get("extracted") or {}).get("w7_episode_cleanup") or {})
    return {
        "episode_id": episode.get("episode_id"),
        "episode_scope": episode.get("episode_scope"),
        "continuation": bool(episode.get("continuation")),
        "trace_group_id": episode.get("trace_group_id"),
        "trace_phase_index": episode.get("trace_phase_index"),
        "trace_phase_count": episode.get("trace_phase_count"),
        "trace_relation_type": episode.get("trace_relation_type"),
        "trace_link_strength": episode.get("trace_link_strength"),
        "trace_link_reasons": episode.get("trace_link_reasons") or [],
        "trace_link_candidates": episode.get("trace_link_candidates") or [],
        "resolution_status": cleanup.get("resolution_status"),
        "accepted_resolution_message_ids": [str(item.get("message_id") or "") for item in episode.get("resolution_messages") or [] if isinstance(item, dict)],
        "rejected_resolution_message_ids": cleanup.get("rejected_resolution_message_ids") or [],
        "w2_ready": bool(episode.get("w2_ready")),
        "w2_block_reasons": episode.get("w2_block_reasons") or [],
        "fault_focus": ((episode.get("extracted") or {}).get("fault_focus_text") or ""),
    }


def w2_input_projection(episode: dict[str, Any]) -> dict[str, Any]:
    """Return the complete W7 episode shape passed into W2.

    Production additionally injects alignment-only KG context and may promote
    nearby Jira/chat evidence before calling W2.  Those additions do not alter
    the episode boundary decision audited here, so the projection records the
    gap explicitly instead of pretending the audit used a production KG slice.
    """
    out = json.loads(json.dumps(episode, ensure_ascii=False))
    extracted = out.get("extracted") if isinstance(out.get("extracted"), dict) else {}
    extracted.setdefault("review_context", {
        "schema_version": "w7.audit_alignment_placeholder.v1",
        "context_role": "alignment_only",
        "facts_may_not_be_copied_as_new_evidence": True,
        "audit_note": "production pipeline injects the retrieved KG slice after this boundary audit",
    })
    out["extracted"] = extracted
    out["w2_input_contract"] = {
        "producer": "W7",
        "consumer": "W2",
        "schema_version": "w7.w2_episode_input.v1",
        "full_episode_preserved": True,
        "production_additions_not_executed_in_audit": [
            "alignment_only_kg_retrieval",
            "nearby_case_evidence_promotion",
            "offline_jira_evidence_linking",
        ],
    }
    return out


def _message_timeline(episode: dict[str, Any]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for role, key in (
        ("fault", "fault_description_messages"),
        ("diagnostic", "diagnostic_chain_messages"),
        ("resolution", "resolution_messages"),
        ("noise", "noise_messages"),
        ("context", "case_context_messages"),
        ("promoted_evidence", "case_evidence_messages"),
    ):
        for index, message in enumerate(episode.get(key) or []):
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or f"{key}:{index}")
            target = by_id.setdefault(message_id, {"message": message, "roles": []})
            if role not in target["roles"]:
                target["roles"].append(role)
    rows = list(by_id.values())
    rows.sort(key=lambda row: (
        str((row.get("message") or {}).get("create_time") or ""),
        str((row.get("message") or {}).get("message_id") or ""),
    ))
    return rows


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def full_context_markdown(
    thread_id: str,
    source_episodes: list[dict[str, Any]],
    w7_episodes: list[dict[str, Any]],
    json_name: str,
) -> str:
    lines = [
        f"# W7 full-context 审核：`{thread_id}`",
        "",
        f"完整机器可读对照：[`{json_name}`]({json_name})",
        "",
        "## 审核目标",
        "",
        "逐条核对完整 W1 消息证据、W7 边界/状态决策，以及实际传给 W2 的 episode 结构。",
        "",
    ]
    for source, refined in zip(source_episodes, w7_episodes):
        cleanup = ((refined.get("extracted") or {}).get("w7_episode_cleanup") or {})
        lines.extend([
            f"## Episode `{source.get('episode_id')}`",
            "",
            "### 判定对照",
            "",
            "| 字段 | W1 | W7 |",
            "|---|---|---|",
            f"| completeness/scope | {_md_cell(source.get('completeness'))} | {_md_cell(refined.get('episode_scope'))} |",
            f"| fault focus | {_md_cell(((source.get('extracted') or {}).get('fault_focus_text') or (source.get('extracted') or {}).get('symptom_raw')))} | {_md_cell((refined.get('extracted') or {}).get('fault_focus_text'))} |",
            f"| resolution | {_md_cell([m.get('message_id') for m in source.get('resolution_messages') or [] if isinstance(m, dict)])} | status={_md_cell(cleanup.get('resolution_status'))}; accepted={_md_cell([m.get('message_id') for m in refined.get('resolution_messages') or [] if isinstance(m, dict)])}; rejected={_md_cell(cleanup.get('rejected_resolution_message_ids'))} |",
            f"| W2 gate | W1 无此门禁 | ready={refined.get('w2_ready')}; block={_md_cell(refined.get('w2_block_reasons'))} |",
            f"| trace | W1 thread={_md_cell(source.get('thread_id'))} | group={_md_cell(refined.get('trace_group_id'))}; phase={refined.get('trace_phase_index')}/{refined.get('trace_phase_count')}; relation={_md_cell(refined.get('trace_relation_type'))}; strength={_md_cell(refined.get('trace_link_strength'))}; reasons={_md_cell(refined.get('trace_link_reasons'))}; continuation={refined.get('continuation')} |",
            "",
            "### W1 完整消息时间线",
            "",
            "| 时间 | 角色 | 发送人 | message_id | 原文 | 链接/附件 |",
            "|---|---|---|---|---|---|",
        ])
        for row in _message_timeline(source):
            message = row["message"]
            sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
            resources = [str(item.get("url") or item.get("label") or "") for item in message.get("links") or [] if isinstance(item, dict)]
            resources.extend(str(item.get("name") or item.get("file_name") or item.get("label") or "") for item in message.get("attachment_metadata") or [] if isinstance(item, dict))
            lines.append(
                f"| {_md_cell(message.get('create_time'))} | {_md_cell(','.join(row['roles']))} | {_md_cell(sender.get('name') or message.get('sender'))} | "
                f"`{_md_cell(message.get('message_id') or message.get('source_message_id'))}` | {_md_cell(message.get('text') or message.get('content_summary'))} | {_md_cell(resources)} |"
            )
        lines.extend([
            "",
            "### W7 实际交给 W2 的关键分组",
            "",
            f"- `fault_description_messages`: {len(refined.get('fault_description_messages') or [])}",
            f"- `diagnostic_chain_messages`: {len(refined.get('diagnostic_chain_messages') or [])}",
            f"- `resolution_messages`: {len(refined.get('resolution_messages') or [])}",
            f"- `w7_resolution_messages_rejected`: {len(refined.get('w7_resolution_messages_rejected') or [])}",
            f"- `case_context_messages`（W2 中视为不可信上下文）: {len(refined.get('case_context_messages') or [])}",
            f"- `evidence_message_ids`: {len(refined.get('evidence_message_ids') or [])}",
            f"- `source_offsets`: {len(refined.get('source_offsets') or [])}",
            f"- `attachments`: {len(refined.get('attachments') or [])}",
            f"- `trace_link_candidates`（仅审核建议，不共享 evidence/outcome）: {len(refined.get('trace_link_candidates') or [])}",
            "",
            "<details><summary>展开完整 W2 input JSON</summary>",
            "",
            "```json",
            json.dumps(w2_input_projection(refined), ensure_ascii=False, indent=2),
            "```",
            "",
            "</details>",
            "",
        ])
    return "\n".join(lines)


def select_threads(path: Path, *, per_bucket: int) -> tuple[set[str], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "max_duration_hours": 0.0,
        "has_report": False,
        "has_non_report": False,
        "resolution_risk": 0,
        "episode_ids": [],
    })
    total = 0
    for episode in iter_json_array(path):
        total += 1
        thread_id = str(episode.get("thread_id") or episode.get("source_thread_id") or "")
        row = stats[thread_id]
        row["count"] += 1
        row["max_duration_hours"] = max(row["max_duration_hours"], duration_hours(episode))
        has_report = bool(episode.get("field_report_anchor"))
        row["has_report"] = row["has_report"] or has_report
        row["has_non_report"] = row["has_non_report"] or not has_report
        row["episode_ids"].append(str(episode.get("episode_id") or ""))
        for text in episode_resolution_texts(episode):
            if _w7_is_question_or_pending(text):
                row["resolution_risk"] += 1
    rows = list(stats.items())
    selected: set[str] = set()
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "long_reference_chain": sorted(rows, key=lambda item: item[1]["max_duration_hours"], reverse=True),
        "fragmented_session": sorted(rows, key=lambda item: item[1]["count"], reverse=True),
        "report_contamination": sorted((item for item in rows if item[1]["has_report"] and item[1]["has_non_report"]), key=lambda item: item[1]["count"], reverse=True),
        "resolution_risk": sorted((item for item in rows if item[1]["resolution_risk"]), key=lambda item: item[1]["resolution_risk"], reverse=True),
    }
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for bucket, bucket_rows in buckets.items():
        selected_rows[bucket] = []
        for thread_id, row in bucket_rows[:per_bucket]:
            selected.add(thread_id)
            selected_rows[bucket].append({"thread_id": thread_id, **row})
    return selected, {
        "total_episodes": total,
        "total_threads": len(stats),
        "buckets": selected_rows,
    }


def run(
    input_path: Path,
    out_dir: Path,
    *,
    per_bucket: int = 20,
    include_episode_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_context_dir = out_dir / "full_context"
    full_context_dir.mkdir(parents=True, exist_ok=True)
    selected_threads, selection = select_threads(input_path, per_bucket=per_bucket)
    explicit_rows = [
        {
            "thread_id": thread_id,
            "count": len(episodes),
            "episode_ids": [str(item.get("episode_id") or "") for item in episodes],
            "source": "frozen_pipeline_result",
        }
        for thread_id, episodes in sorted((include_episode_groups or {}).items())
    ]
    selection["explicit_include"] = explicit_rows
    selection_reasons: dict[str, list[str]] = defaultdict(list)
    for bucket, rows in selection["buckets"].items():
        for row in rows:
            selection_reasons[str(row.get("thread_id") or "")].append(bucket)
    for row in selection["explicit_include"]:
        selection_reasons[str(row.get("thread_id") or "")].append("fixed173_trace_calibration")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_episode_count = 0
    for episode in iter_json_array(input_path):
        thread_id = str(episode.get("thread_id") or episode.get("source_thread_id") or "")
        if thread_id in selected_threads:
            grouped[thread_id].append(episode)
            selected_episode_count += 1
    for thread_id, episodes in (include_episode_groups or {}).items():
        if thread_id in grouped:
            continue
        grouped[thread_id] = json.loads(json.dumps(episodes, ensure_ascii=False))
        selected_threads.add(thread_id)
        selected_episode_count += len(episodes)

    cases: list[dict[str, Any]] = []
    before_ready = 0
    after_ready = 0
    rejected_resolution_count = 0
    report_only_count = 0
    continuation_count = 0
    w2_input_rows: list[dict[str, Any]] = []
    for thread_id, episodes in sorted(grouped.items()):
        refined = refine_episode_group(episodes)
        before = [compact_before(item) for item in episodes]
        after = [compact_after(item) for item in refined]
        before_ready += sum(bool(row["fault_focus"]) and row["completeness"] != "noise" for row in before)
        after_ready += sum(bool(row["w2_ready"]) for row in refined)
        rejected_resolution_count += sum(len(row["rejected_resolution_message_ids"]) for row in after)
        report_only_count += sum(row["episode_scope"] == "report_only" for row in after)
        continuation_count += sum(bool(row["continuation"]) for row in after)
        digest = hashlib.sha1(thread_id.encode("utf-8")).hexdigest()[:12]
        context_json_name = f"{digest}.json"
        context_md_name = f"{digest}.md"
        context_payload = {
            "thread_id": thread_id,
            "source_episodes": episodes,
            "w7_episodes": refined,
            "w2_inputs": [w2_input_projection(item) for item in refined],
        }
        (full_context_dir / context_json_name).write_text(
            json.dumps(context_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (full_context_dir / context_md_name).write_text(
            full_context_markdown(thread_id, episodes, refined, context_json_name), encoding="utf-8"
        )
        for item in refined:
            w2_input_rows.append(w2_input_projection(item))
        weak_links = sum(
            1
            for item in refined
            for candidate in item.get("trace_link_candidates") or []
            if candidate.get("linked") and candidate.get("link_strength") == "weak"
        )
        accepted_links = sum(
            1 for item in refined
            if str(item.get("trace_link_strength") or "") in {"hard", "strong", "medium"}
        )
        reasons = list(dict.fromkeys(selection_reasons.get(thread_id) or []))
        priority_score = (
            (1000 if "fixed173_trace_calibration" in reasons else 0)
            + min(weak_links, 10) * 8
            + min(accepted_links, 5) * 6
            + min(len(episodes), 20)
            + (8 if any(item.get("episode_scope") == "multi_fault" for item in refined) else 0)
        )
        cases.append({
            "thread_id": thread_id,
            "episode_count": len(episodes),
            "review_priority_score": priority_score,
            "review_priority_reasons": reasons,
            "weak_trace_link_candidate_count": weak_links,
            "accepted_trace_link_count": accepted_links,
            "full_context_markdown": f"full_context/{context_md_name}",
            "full_context_json": f"full_context/{context_json_name}",
            "before": before,
            "after": after,
            "sample_texts": [text for item in episodes for text in episode_texts(item)[:3]][:12],
        })

    summary = {
        "input": str(input_path),
        "selection": {"per_bucket": per_bucket, **selection},
        "selected_threads": len(selected_threads),
        "selected_episodes": selected_episode_count,
        "before_w2_ready_proxy_count": before_ready,
        "after_w2_ready_count": after_ready,
        "resolution_messages_rejected_by_w7": rejected_resolution_count,
        "report_only_episodes": report_only_count,
        "continuation_episodes": continuation_count,
        "quality_claim": "targeted_regression_only_not_full_dataset_score",
    }
    payload = {"summary": summary, "cases": cases}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "review_pack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "w2_inputs.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in w2_input_rows), encoding="utf-8"
    )
    (out_dir / "review_pack.md").write_text(to_markdown(summary, cases), encoding="utf-8")
    (out_dir / "full_context_index.md").write_text(to_full_context_index(summary, cases), encoding="utf-8")
    (out_dir / "audit_conclusion.md").write_text(to_audit_conclusion(summary, cases), encoding="utf-8")
    return summary


def to_markdown(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# W7 定向回归审核",
        "",
        "> 该报告只覆盖高风险 session 抽样，不代表全量质量分数。",
        "",
        "## 汇总",
        "",
        f"- 全量 episode：{summary['selection']['total_episodes']}",
        f"- 全量 session：{summary['selection']['total_threads']}",
        f"- 定向 session：{summary['selected_threads']}",
        f"- 定向 episode：{summary['selected_episodes']}",
        f"- W7 拦截疑问/待验证 resolution：{summary['resolution_messages_rejected_by_w7']}",
        f"- 标记 report_only：{summary['report_only_episodes']}",
        f"- 标记 longitudinal continuation：{summary['continuation_episodes']}",
        "",
        "## 抽样分桶",
        "",
        "| 分桶 | session 数 |",
        "|---|---:|",
    ]
    for bucket, rows in summary["selection"]["buckets"].items():
        lines.append(f"| {bucket} | {len(rows)} |")
    lines.extend(["", "## 复核记录", ""])
    for case in cases:
        lines.extend([
            f"### `{case['thread_id']}`（{case['episode_count']} episodes）",
            "",
            f"完整审核：[`full-context Markdown`]({case['full_context_markdown']}) | [`完整 JSON`]({case['full_context_json']})",
            "",
        ])
        lines.append("原文抽样：")
        for text in case["sample_texts"][:8]:
            lines.append(f"- {text}")
        lines.extend(["", "| episode | W1 观察 | W7 结果 |", "|---|---|---|"])
        for before, after in zip(case["before"], case["after"]):
            lines.append(
                f"| `{before['episode_id']}` | {before['fault_focus'] or '无 fault focus'}；resolution={len(before['resolution_messages'])}；report={before['field_report']} | "
                f"scope={after['episode_scope']}；status={after['resolution_status']}；w2_ready={after['w2_ready']}；block={','.join(after['w2_block_reasons']) or 'none'} |"
            )
        lines.append("")
    lines.extend([
        "## 判定",
        "",
        "W7 当前只做边界与证据门控，不生成 FaultFamily/DiagnosticAction/Outcome。",
        "`w2_ready=true` 仅表示 episode 具备进入 W2 的最低证据条件，不表示 W2 候选可以入图。",
        "",
    ])
    return "\n".join(lines)


def to_full_context_index(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# W7 full-context 审核索引",
        "",
        "`review_pack.md` 只保留摘要；本索引链接到每个 session 的完整 W1→W7→W2 对照。",
        "",
        f"审核 session：{summary['selected_threads']}；episode：{summary['selected_episodes']}",
        "",
        "| session | episode 数 | 完整审核 | JSON |",
        "|---|---:|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| `{case['thread_id']}` | {case['episode_count']} | [{case['full_context_markdown']}]({case['full_context_markdown']}) | [{case['full_context_json']}]({case['full_context_json']}) |"
        )
    lines.extend([
        "",
        "## 审核顺序",
        "",
        "1. 先看完整消息时间线；",
        "2. 判断 W1 episode 边界；",
        "3. 查看 W7 的 scope/status/w2_ready/block；",
        "4. 展开完整 W2 input JSON，确认 W2 获得了哪些消息分组和证据；",
        "5. 记录应合并、应拆分、应放行或应拦截的意见。",
        "",
    ])
    return "\n".join(lines)


def to_audit_conclusion(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    block_counts: dict[str, int] = defaultdict(int)
    scope_counts: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        for row in case.get("after") or []:
            scope_counts[str(row.get("episode_scope") or "unknown")] += 1
            status_counts[str(row.get("resolution_status") or "unknown")] += 1
            for reason in row.get("w2_block_reasons") or []:
                block_counts[str(reason)] += 1
    lines = [
        "# W7 定向回归审核结论",
        "",
        f"本轮覆盖 {summary['selected_threads']} 个高风险 session、{summary['selected_episodes']} 个 episode。该结果是风险定向审核，不是全量质量分数。",
        "",
        "## 结果",
        "",
        f"- `w2_ready=true`：{summary['after_w2_ready_count']}",
        f"- 拦截疑问/待验证 resolution：{summary['resolution_messages_rejected_by_w7']}",
        f"- verified：{status_counts.get('verified', 0)}",
        f"- pending：{status_counts.get('pending', 0)}",
        f"- ineffective：{status_counts.get('ineffective', 0)}",
        f"- multi_fault：{scope_counts.get('multi_fault', 0)}",
        f"- report_only：{scope_counts.get('report_only', 0)}",
        f"- longitudinal_trace：{scope_counts.get('longitudinal_trace', 0)}",
        "",
        "主要阻塞：",
        "",
    ]
    for reason, count in sorted(block_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{reason}`：{count}")
    lines.extend([
        "",
        "## 如何复核",
        "",
        "摘要不足以判定 W7 是否正确。请从 `full_context_index.md` 进入单个 session，逐条查看完整 W1 消息时间线、W7 判定和完整 W2 input JSON。",
        "",
        "## 当前判断",
        "",
        "W7 已能拦截最危险的虚假 resolution、多问题报告和纯汇报内容，但 fault-focus 质量和同一 trace 的阶段合并仍需继续优化。`w2_ready=true` 仅表示达到 W2 最低输入条件，不表示可以入图。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/xing_relation_context_v3_20260715/episodes.json")
    parser.add_argument("--out-dir", default="data/results/w7_targeted_regression_20260715")
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument(
        "--include-pipeline-result",
        action="append",
        default=[],
        help="Also include every thread represented by the episodes in a pipeline_result.json file.",
    )
    args = parser.parse_args()
    include_episode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_text in args.include_pipeline_result:
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
        for item in payload.get("episodes") or []:
            if not isinstance(item, dict):
                continue
            thread_id = str(item.get("thread_id") or item.get("source_thread_id") or "")
            if thread_id:
                include_episode_groups[thread_id].append(item)
    summary = run(
        Path(args.input),
        Path(args.out_dir),
        per_bucket=max(1, args.per_bucket),
        include_episode_groups=dict(include_episode_groups),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
