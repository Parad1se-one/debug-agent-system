"""Reproducible W1/W7 smoke over field-report-centered real chat windows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.people_roles import load_people_role_registry
from debug_agent_system.agents.write.review_context import inject_review_context
from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent, _build_observed_people, _field_report_anchor


def run(
    source: str | Path,
    out_dir: str | Path,
    *,
    limit: int = 500,
    selection_mode: str = "adaptive",
    quiet_gap_hours: float = 12.0,
) -> dict[str, Any]:
    source = Path(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_messages = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in all_messages:
        by_thread[str(message.get("thread_id") or "default")].append(message)
    selected, selection_stats = _select_field_report_messages(
        by_thread,
        limit=limit,
        selection_mode=selection_mode,
        quiet_gap_hours=quiet_gap_hours,
    )
    messages = sorted(
        selected.values(),
        key=lambda item: (str(item.get("thread_id") or ""), str(item.get("create_time") or ""), str(item.get("message_id") or "")),
    )
    collector = ChatCollectAgent()
    summaries = collector.aggregate_threads(messages)
    episodes = [episode for summary in summaries for episode in summary.get("episodes", [])]
    anchors = [anchor for summary in summaries for anchor in summary.get("field_report_anchors", [])]
    observed = _build_observed_people(messages, anchors)
    registry = load_people_role_registry()
    background = {
        "schema_version": "smoke.alignment.v1",
        "context_role": "alignment_only",
        "facts_may_not_be_copied_as_new_evidence": True,
    }
    w7_episodes = [inject_review_context(episode, background, role_registry=registry) for episode in episodes]
    episode_roles: Counter[str] = Counter()
    organization_roles: Counter[str] = Counter()
    for episode in w7_episodes:
        attribution = ((episode.get("extracted") or {}).get("attribution") or {})
        for row in attribution.get("role_assignments") or []:
            episode_roles.update(row.get("episode_roles") or [])
            organization_roles.update(row.get("organization_roles") or [])
    expected_anchor_items = {
        (str(anchor.get("anchor_id") or ""), int(item.get("item_index") or 0))
        for anchor in anchors
        for item in anchor.get("issue_items") or []
        if str(anchor.get("anchor_id") or "") and int(item.get("item_index") or 0)
    }
    observed_anchor_items = [
        (
            str((episode.get("field_report_anchor") or {}).get("anchor_id") or ""),
            int((episode.get("field_report_anchor") or {}).get("anchor_item_index") or 0),
        )
        for episode in episodes
        if episode.get("field_report_anchor")
    ]
    shared_evidence_leaks = 0
    for episode in episodes:
        extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
        unassigned_links = {
            (str(link.get("url") or ""), str(link.get("message_id") or row.get("source_message_id") or ""))
            for row in extracted.get("unassigned_shared_evidence") or []
            if isinstance(row, dict)
            for link in row.get("links") or []
            if isinstance(link, dict) and str(link.get("url") or "")
        }
        active_links = {
            (str(link.get("url") or ""), str(link.get("message_id") or ""))
            for link in extracted.get("links") or []
            if isinstance(link, dict) and str(link.get("url") or "")
        }
        shared_evidence_leaks += len(unassigned_links & active_links)
    role_assignments_missing_evidence = sum(
        not list(row.get("evidence_message_ids") or [])
        for episode in w7_episodes
        for row in (((episode.get("extracted") or {}).get("attribution") or {}).get("role_assignments") or [])
        if isinstance(row, dict)
    )
    confirmed_inferred_fae = sum(
        person.get("status") == "confirmed"
        and not list(person.get("organization_roles") or [])
        and "fae" in (person.get("organization_role_candidates") or [])
        for person in observed
    )
    safety_checks = {
        "anchor_item_coverage_exact": set(observed_anchor_items) == expected_anchor_items,
        "anchor_item_duplicate_count": len(observed_anchor_items) - len(set(observed_anchor_items)),
        "shared_evidence_leak_count": shared_evidence_leaks,
        "role_assignments_missing_message_evidence": role_assignments_missing_evidence,
        "behavior_inferred_fae_confirmed_count": confirmed_inferred_fae,
    }
    safety_checks["passed"] = bool(
        safety_checks["anchor_item_coverage_exact"]
        and safety_checks["anchor_item_duplicate_count"] == 0
        and safety_checks["shared_evidence_leak_count"] == 0
        and safety_checks["role_assignments_missing_message_evidence"] == 0
        and safety_checks["behavior_inferred_fae_confirmed_count"] == 0
    )
    summary = {
        "source": str(source),
        "selection": selection_stats["selection"],
        "selection_stats": selection_stats,
        "counts": {
            "messages": len(messages),
            "threads": len(summaries),
            "field_report_anchors": len(anchors),
            "active_fault_items": sum(int(anchor.get("issue_count") or 0) for anchor in anchors),
            "status_updates": sum(int(anchor.get("status_update_count") or 0) for anchor in anchors),
            "work_items": sum(int(anchor.get("work_item_count") or 0) for anchor in anchors),
            "episodes": len(episodes),
            "anchored_episodes": sum(bool(episode.get("field_report_anchor")) for episode in episodes),
            "observed_people": len(observed),
            "candidate_fae": sum("fae" in (person.get("organization_role_candidates") or []) for person in observed),
        },
        "episode_role_counts": dict(episode_roles),
        "organization_role_counts": dict(organization_roles),
        "safety_checks": safety_checks,
    }
    run_payload = {
        "messages": messages,
        "thread_summaries": summaries,
        "episodes": episodes,
        "field_report_anchors": anchors,
        "observed_people": observed,
        "run_manifest": summary,
    }
    collector.write_run(out / "w1", run_payload)
    (out / "episodes_w7.json").write_text(json.dumps(w7_episodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "quality_report.md").write_text(_markdown_report(summary, anchors, observed), encoding="utf-8")
    return summary


def _select_field_report_messages(
    by_thread: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
    selection_mode: str,
    quiet_gap_hours: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Select complete anchor sessions without cutting an active diagnosis.

    ``limit`` is a soft global message budget.  Once one anchor session has
    started, its full adaptive window is retained even if the total exceeds
    the requested budget.  This avoids the old failure mode where the final
    ``[:limit]`` silently removed the action/outcome half of a conversation.
    """

    if selection_mode not in {"adaptive", "fixed"}:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")
    selected: dict[str, dict[str, Any]] = {}
    window_sizes: list[int] = []
    anchor_ids_selected: set[str] = set()
    for items in by_thread.values():
        items.sort(key=lambda item: str(item.get("create_time") or ""))
        anchor_indexes = [index for index, message in enumerate(items) if _field_report_anchor(message)]
        for anchor_pos, index in enumerate(anchor_indexes):
            if limit and len(selected) >= limit:
                break
            if selection_mode == "fixed":
                window = items[max(0, index - 4): index + 7]
            else:
                next_anchor = anchor_indexes[anchor_pos + 1] if anchor_pos + 1 < len(anchor_indexes) else len(items)
                window = _adaptive_anchor_window(
                    items,
                    anchor_index=index,
                    previous_anchor_index=anchor_indexes[anchor_pos - 1] if anchor_pos else -1,
                    next_anchor_index=next_anchor,
                    quiet_gap_hours=quiet_gap_hours,
                )
            for message in window:
                selected[str(message.get("message_id") or id(message))] = message
            window_sizes.append(len(window))
            anchor_message = items[index]
            anchor = _field_report_anchor(anchor_message)
            anchor_id = str(anchor.get("anchor_id") or anchor_message.get("message_id") or index)
            anchor_ids_selected.add(anchor_id)
        if limit and len(selected) >= limit:
            break
    return selected, {
        "selection": "field_report_adaptive_sessions" if selection_mode == "adaptive" else "field_report_fixed_windows",
        "selection_mode": selection_mode,
        "requested_message_soft_limit": int(limit or 0),
        "selected_messages": len(selected),
        "anchors_selected": len(anchor_ids_selected),
        "quiet_gap_hours": quiet_gap_hours if selection_mode == "adaptive" else None,
        "window_min": min(window_sizes) if window_sizes else 0,
        "window_max": max(window_sizes) if window_sizes else 0,
        "windows_over_11": sum(size > 11 for size in window_sizes),
        "truncated_windows": 0,
    }


def _adaptive_anchor_window(
    items: list[dict[str, Any]],
    *,
    anchor_index: int,
    previous_anchor_index: int,
    next_anchor_index: int,
    quiet_gap_hours: float,
) -> list[dict[str, Any]]:
    start = max(previous_anchor_index + 1, anchor_index - 4, 0)
    end = min(len(items), max(anchor_index + 1, next_anchor_index))
    last_time = _message_time(items[anchor_index])
    for index in range(anchor_index + 1, end):
        current_time = _message_time(items[index])
        if last_time is not None and current_time is not None:
            gap_hours = (current_time - last_time).total_seconds() / 3600.0
            if gap_hours > quiet_gap_hours:
                end = index
                break
        if current_time is not None:
            last_time = current_time
    return items[start:end]


def _message_time(message: dict[str, Any]) -> datetime | None:
    value = str(message.get("create_time") or "").strip()
    if not value:
        return None
    for candidate in (value, value.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _markdown_report(summary: dict[str, Any], anchors: list[dict[str, Any]], observed: list[dict[str, Any]]) -> str:
    lines = ["# W1 FieldReportAnchor / People Role Smoke", ""]
    lines.extend(f"- {key}: {value}" for key, value in summary["counts"].items())
    lines.extend(["", "## Safety checks", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary.get("safety_checks", {}).items())
    lines.extend(["", "## Anchor examples", ""])
    for anchor in anchors[:25]:
        lines.append(
            f"### {anchor.get('author') or '(unknown)'} / {anchor.get('report_date') or ''} / "
            f"fault={anchor.get('issue_count')} / status={anchor.get('status_update_count')} / work={anchor.get('work_item_count')}"
        )
        lines.extend(f"- fault: {item.get('text')}" for item in anchor.get("issue_items") or [])
        lines.extend(f"- status: {item.get('text')}" for item in anchor.get("status_updates") or [])
        lines.extend(f"- work: {item.get('text')}" for item in anchor.get("work_items") or [])
        lines.append("")
    lines.extend(["## Top observed people", ""])
    lines.extend(
        f"- {person['name']}: status={person['status']}, candidate={person.get('organization_role_candidates')}, "
        f"roles={person.get('episode_role_counts')}, report_days={len(person.get('distinct_report_dates') or [])}, "
        f"confidence={person.get('confidence')}"
        for person in observed[:20]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--selection-mode", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--quiet-gap-hours", type=float, default=12.0)
    args = parser.parse_args()
    print(json.dumps(run(
        args.source,
        args.out,
        limit=args.limit,
        selection_mode=args.selection_mode,
        quiet_gap_hours=args.quiet_gap_hours,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
