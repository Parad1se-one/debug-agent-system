"""Regression gate for real diagnosis eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COMPOSITE_DROP = 0.01
RECALL_DROP = 0.02
TARGET_DROP = 0.02
FLIP_LIMIT = 3


def resolve_run_path(path_or_latest: str | Path) -> Path:
    p = Path(path_or_latest)
    if p.name.endswith(".txt"):
        text = p.read_text(encoding="utf-8").strip()
        return Path(text)
    return p


def load_run(path_or_latest: str | Path) -> dict[str, Any]:
    p = resolve_run_path(path_or_latest)
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_gate(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur = current.get("summary") or current
    base = baseline.get("summary") or baseline
    failures: list[str] = []
    warnings: list[str] = []

    if (cur.get("failed") or 0) > 0:
        failures.append(f"failed cases > 0: {cur.get('failed')}")
    if (cur.get("unsafe_action_rate") or 0.0) > 0:
        failures.append(f"unsafe_action_rate > 0: {cur.get('unsafe_action_rate')}")
    _drop_fail(failures, base, cur, "terminal_ok_rate", 0.0)
    _drop_fail(failures, base, cur, "check_recall", RECALL_DROP)
    _drop_fail(failures, base, cur, "evidence_recall", RECALL_DROP)
    _drop_fail(failures, base, cur, "target_error_acc", TARGET_DROP)
    _drop_fail(failures, base, cur, "composite_gated", COMPOSITE_DROP)
    _drop_warn(warnings, base, cur, "judge_score", 0.02)

    flipped = _flipped_cases(current.get("details") or [], baseline.get("details") or [])
    if len(flipped) > FLIP_LIMIT:
        failures.append(f"weak-case flips > {FLIP_LIMIT}: {len(flipped)}")

    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "flipped_cases": flipped,
        "current_summary": cur,
        "baseline_summary": base,
    }


def _drop_fail(failures: list[str], base: dict[str, Any], cur: dict[str, Any], key: str, allowed: float) -> None:
    old = base.get(key)
    new = cur.get(key)
    if old is None or new is None:
        return
    if float(old) - float(new) > allowed:
        failures.append(f"{key} dropped {float(old):.4f}->{float(new):.4f} > {allowed:.4f}")


def _drop_warn(warnings: list[str], base: dict[str, Any], cur: dict[str, Any], key: str, allowed: float) -> None:
    old = base.get(key)
    new = cur.get(key)
    if old is None or new is None:
        return
    if float(old) - float(new) > allowed:
        warnings.append(f"{key} dropped {float(old):.4f}->{float(new):.4f}; report-only")


def _flipped_cases(current_details: list[dict[str, Any]], baseline_details: list[dict[str, Any]]) -> list[str]:
    base = {str(x.get("case_id")): x for x in baseline_details}
    out: list[str] = []
    for row in current_details:
        cid = str(row.get("case_id") or "")
        prev = base.get(cid)
        if not prev:
            continue
        if (prev.get("composite_gated") or 0.0) >= 0.5 and (row.get("composite_gated") or 0.0) < 0.5:
            out.append(cid)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/results/runs/latest_real.txt")
    parser.add_argument("--baseline", default="data/eval/baselines/real_diag_v1_baseline.json")
    args = parser.parse_args(argv)
    report = evaluate_gate(load_run(args.eval), load_run(args.baseline))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
