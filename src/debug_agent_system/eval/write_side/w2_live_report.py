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
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def build_live_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    progress = _load_json(root / "progress.json")
    summary = _load_json(root / "summary.json")
    candidates_path = root / "w2_candidates.jsonl"
    partial_path = root / "w2_candidates.partial.jsonl"
    using_partial = False
    rows = _load_jsonl(candidates_path)
    if not rows:
        rows = _load_jsonl(partial_path)
        using_partial = bool(rows)
    family = build_family_report(rows, sample_limit=15) if rows else {}
    quality = build_quality_report(rows, sample_limit=15) if rows else {}
    split = build_split_report(rows, sample_limit=15) if rows else {}
    quality_gate = gate_report(quality) if quality else {}
    return {
        "schema_version": "debug_agent_system.w2_live_report.v1",
        "run_dir": str(root),
        "status": "completed" if summary else progress.get("status") or "unknown",
        "using_partial_candidates": using_partial,
        "progress": progress,
        "summary": summary,
        "family_diagnostics": family,
        "quality_diagnostics": quality,
        "split_diagnostics": split,
        "quality_gate": quality_gate,
        "paths": {
            "progress": str(root / "progress.json"),
            "summary": str(root / "summary.json"),
            "candidates": str(candidates_path),
            "partial_candidates": str(partial_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = build_live_report(args.run_dir)
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
