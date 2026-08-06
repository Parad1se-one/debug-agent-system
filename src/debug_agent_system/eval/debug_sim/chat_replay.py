"""Report-only evaluation for high-confidence chat replay scenarios."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.runtime import DebugAgentSystem

from .runner import run_one, score_transcripts
from .scenario_v2 import load_scenarios


def run_chat_replay(*, config: str | Path, scenario_file: str | Path, limit: int | None = None) -> dict[str, Any]:
    system = DebugAgentSystem(load_config(config))
    scenarios = load_scenarios(scenario_file, limit)
    transcripts = [run_one(system, scenario) for scenario in scenarios]
    details, summary = score_transcripts(scenarios, transcripts, judge_policy="report-only")
    return {
        "schema_version": "debug_agent_system.chat_replay_eval.v1",
        "mode": "report-only",
        "meta": {
            "config": str(config),
            "scenario_file": str(scenario_file),
            "limit": limit,
            "n": len(scenarios),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_commit(),
        },
        "summary": summary,
        "details": details,
        "transcripts": transcripts,
    }


def write_report(report: dict[str, Any], out_json: str | Path, out_md: str | Path) -> None:
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    details = report.get("details") or []
    transcripts = {str(row.get("case_id") or ""): row for row in report.get("transcripts") or []}
    lines = [
        "# Chat Replay Eval Report",
        "",
        f"- Mode: `{report.get('mode') or 'report-only'}`",
        f"- Config: `{(report.get('meta') or {}).get('config') or ''}`",
        f"- Scenario file: `{(report.get('meta') or {}).get('scenario_file') or ''}`",
        f"- Cases: `{summary.get('n')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "strong_cases",
        "target_error_acc",
        "top_error_acc",
        "first_check_acc",
        "effective_result_covered",
        "failure_path_acc",
        "missing_info_request_acc",
        "trace_coverage",
        "latency_ms",
        "chat_replay_composite",
        "simulator_gap",
        "failed",
    ):
        lines.append(f"| `{key}` | {_fmt(summary.get(key))} |")

    lines.extend([
        "",
        "## Cases",
        "",
        "| Case | Top Error | First Check | Final | Diagnosis | Replay Composite | Trace | Latency ms | Notes |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ])
    for detail in details:
        case_id = str(detail.get("case_id") or "")
        transcript = transcripts.get(case_id) or {}
        diagnosis = detail.get("trace_diagnosis") or {}
        lines.append(
            "| "
            + " | ".join([
                _esc(case_id),
                _esc(str(detail.get("top_error_id") or "")),
                _esc(_short(str(detail.get("first_check_text") or ""))),
                _esc(str(detail.get("final_status") or "")),
                _esc(_diagnosis_label(diagnosis)),
                _fmt(detail.get("chat_replay_composite")),
                _fmt(detail.get("trace_coverage")),
                _fmt(detail.get("latency_ms")),
                _esc("; ".join(str(x) for x in detail.get("chat_replay_notes") or []) or "-"),
            ])
            + " |"
        )
        events = transcript.get("replay_events") or []
        if events:
            lines.append("")
            lines.append(f"### {_esc(case_id)} Replay Events")
            lines.append("")
            lines.append("| Kind | Result | Matched Text | Reply |")
            lines.append("|---|---|---|---|")
            for event in events:
                lines.append(
                    "| "
                    + " | ".join([
                        _esc(str(event.get("kind") or "")),
                        _esc(str(event.get("result_type") or event.get("slot") or "")),
                        _esc(_short(str(event.get("check_text") or event.get("question") or event.get("matched_text") or ""))),
                        _esc(_short(str(event.get("reply") or ""))),
                    ])
                    + " |"
                )
            lines.append("")

    failed_or_weak = [
        detail for detail in details
        if str((detail.get("trace_diagnosis") or {}).get("primary_stage") or "") not in {"ok", ""}
        or float(detail.get("chat_replay_composite") or 1.0) < 0.8
    ]
    if failed_or_weak:
        lines.extend([
            "",
            "## Failure Diagnosis",
            "",
            "| Case | Stage | Cause | Top Candidates | Selected Check | Next Action |",
            "|---|---|---|---|---|---|",
        ])
        for detail in failed_or_weak:
            diagnosis = detail.get("trace_diagnosis") or {}
            digest = detail.get("trace_digest") or {}
            lines.append(
                "| "
                + " | ".join([
                    _esc(str(detail.get("case_id") or "")),
                    _esc(str(diagnosis.get("primary_stage") or "")),
                    _esc(str(diagnosis.get("primary_cause") or "")),
                    _esc(_candidate_summary(digest)),
                    _esc(_selected_check_summary(digest)),
                    _esc(_short(str(diagnosis.get("next_debug_action") or ""), 96)),
                ])
                + " |"
            )

    lines.extend([
        "",
        "## Latency",
        "",
        "| Case | Latency ms | Top Error | First Check |",
        "|---|---:|---|---|",
    ])
    for detail in sorted(details, key=lambda row: float(row.get("latency_ms") or 0), reverse=True):
        lines.append(
            "| "
            + " | ".join([
                _esc(str(detail.get("case_id") or "")),
                _fmt(detail.get("latency_ms")),
                _esc(str(detail.get("top_error_id") or "")),
                _esc(_short(str(detail.get("first_check_text") or ""))),
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def _short(text: str, limit: int = 80) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _diagnosis_label(diagnosis: dict[str, Any]) -> str:
    stage = str(diagnosis.get("primary_stage") or "")
    cause = str(diagnosis.get("primary_cause") or "")
    if not stage and not cause:
        return ""
    return f"{stage}:{cause}" if cause else stage


def _candidate_summary(digest: dict[str, Any]) -> str:
    rows = [row for row in digest.get("candidate_scores") or [] if isinstance(row, dict)]
    parts: list[str] = []
    for row in rows[:3]:
        parts.append(f"{row.get('error_id')}@{row.get('final_rank')}:{row.get('final_score')}")
    return "; ".join(parts)


def _selected_check_summary(digest: dict[str, Any]) -> str:
    row = digest.get("selected_check_trace") or {}
    if not isinstance(row, dict):
        return ""
    return f"{row.get('check_id') or ''} / {row.get('source_error_id') or ''} / {row.get('source_tier') or ''}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/debug_agent_system_sag.yaml")
    parser.add_argument("--scenario-file", default="data/eval/scenarios/chat_replay_seed_v1.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-json", default="data/results/chat_replay/latest.json")
    parser.add_argument("--out-md", default="data/results/chat_replay/latest.md")
    args = parser.parse_args(argv)

    limit = args.limit if args.limit > 0 else None
    report = run_chat_replay(config=args.config, scenario_file=args.scenario_file, limit=limit)
    write_report(report, args.out_json, args.out_md)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
