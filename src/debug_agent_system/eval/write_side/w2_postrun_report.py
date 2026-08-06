from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_agent_system.eval.write_side.w2_family_diagnostics import build_report as build_family_report
from debug_agent_system.eval.write_side.w2_quality_diagnostics import build_report as build_quality_report
from debug_agent_system.eval.write_side.w2_quality_gate import gate_report
from debug_agent_system.eval.write_side.w2_split_diagnostics import build_report as build_split_report


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object json: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def build_postrun_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    summary_path = root / "summary.json"
    candidates_path = root / "w2_candidates.jsonl"
    progress_path = root / "progress.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing w2_candidates.jsonl: {candidates_path}")

    summary = _load_json(summary_path)
    progress = _load_json(progress_path) if progress_path.exists() else {}
    rows = _load_jsonl(candidates_path)
    family = build_family_report(rows, sample_limit=25)
    quality = build_quality_report(rows, sample_limit=25)
    split = build_split_report(rows, sample_limit=25)
    quality_gate = gate_report(quality)

    return {
        "schema_version": "debug_agent_system.w2_postrun_report.v1",
        "run_dir": str(root),
        "summary": summary,
        "progress": progress,
        "family_diagnostics": family,
        "quality_diagnostics": quality,
        "split_diagnostics": split,
        "quality_gate": quality_gate,
        "recommended_next_steps": _recommended_next_steps(family, quality, split, quality_gate),
    }


def _recommended_next_steps(
    family: dict[str, Any],
    quality: dict[str, Any],
    split: dict[str, Any],
    quality_gate: dict[str, Any],
) -> list[str]:
    steps: list[str] = []
    family_noncanonical = int(family.get("noncanonical_family_count") or 0)
    quality_counters = quality.get("counters") if isinstance(quality.get("counters"), dict) else {}
    if family_noncanonical:
        steps.append("继续补 family canonicalization / pseudo-family 拦截，优先处理非 canonical family 样本。")
    if int(quality_counters.get("empty_case") or 0):
        steps.append("针对 empty_case 样本补 focused text / no-fault 回退策略，减少空 case。")
    if int(quality_counters.get("report_noise") or 0) or int(quality_counters.get("positive_no_issue") or 0):
        steps.append("继续增强 report/no-status gate，避免状态汇报和跟踪消息污染 fault candidate。")
    if int(quality_counters.get("split_required") or 0):
        top_pairs = split.get("top_split_family_pairs") if isinstance(split.get("top_split_family_pairs"), list) else []
        if top_pairs:
            steps.append(f"针对 split_required 样本补 W3 split-case 规则，优先处理高频 family 对：{top_pairs[0][0]}。")
        else:
            steps.append("针对 split_required 样本补 W3 split-case 规则，减少多故障/汇报混合。")
    if not steps:
        steps.append("当前 run 已通过现有质量门禁；下一步应转入 approved review_queue 抽样和 materialized_execution / SAG 验证。")
    if quality_gate.get("status") == "failed":
        steps.append("当前 quality gate 未通过，暂不建议扩大到主图写入，应继续收敛问题桶后再做批量候选审核。")
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = build_postrun_report(args.run_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
