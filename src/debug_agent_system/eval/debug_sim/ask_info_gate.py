"""Gate for ask-info scenarios without requiring a historical baseline."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .gate import load_run, resolve_run_path


def evaluate(
    current: dict[str, Any],
    *,
    min_cases: int,
    min_required_info_acc: float,
    min_ask_info_precision: float,
    min_ask_once_then_step_rate: float,
    max_over_ask_rate: float,
) -> dict[str, Any]:
    summary = current.get("summary") or {}
    failures: list[str] = []
    n = int(summary.get("n") or 0)
    if n < min_cases:
        failures.append(f"case count {n} < {min_cases}")
    if (summary.get("failed") or 0) > 0:
        failures.append(f"failed cases > 0: {summary.get('failed')}")
    terminal_ok = summary.get("terminal_ok_rate")
    if terminal_ok is None or float(terminal_ok) < 1.0:
        failures.append(f"terminal_ok_rate < 1.0: {terminal_ok}")
    required_info_acc = summary.get("required_info_acc")
    if required_info_acc is None or float(required_info_acc) < min_required_info_acc:
        failures.append(f"required_info_acc {required_info_acc} < {min_required_info_acc}")
    ask_info_precision = summary.get("ask_info_precision")
    if ask_info_precision is None or float(ask_info_precision) < min_ask_info_precision:
        failures.append(f"ask_info_precision {ask_info_precision} < {min_ask_info_precision}")
    ask_once = summary.get("ask_once_then_step_rate")
    if ask_once is None or float(ask_once) < min_ask_once_then_step_rate:
        failures.append(f"ask_once_then_step_rate {ask_once} < {min_ask_once_then_step_rate}")
    over_ask = summary.get("over_ask_rate")
    if over_ask is not None and float(over_ask) > max_over_ask_rate:
        failures.append(f"over_ask_rate {over_ask} > {max_over_ask_rate}")
    return {"status": "FAIL" if failures else "PASS", "failures": failures, "current_summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/results/runs/latest_real.txt")
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--min-required-info-acc", type=float, default=0.6)
    parser.add_argument("--min-ask-info-precision", type=float, default=0.8)
    parser.add_argument("--min-ask-once-then-step-rate", type=float, default=0.8)
    parser.add_argument("--max-over-ask-rate", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = evaluate(
        load_run(resolve_run_path(args.eval)),
        min_cases=args.min_cases,
        min_required_info_acc=args.min_required_info_acc,
        min_ask_info_precision=args.min_ask_info_precision,
        min_ask_once_then_step_rate=args.min_ask_once_then_step_rate,
        max_over_ask_rate=args.max_over_ask_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
