"""Freeze the latest real diagnosis eval as baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gate import load_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="data/results/runs/latest_real.txt")
    parser.add_argument("--baseline", default="data/eval/baselines/real_diag_v1_baseline.json")
    args = parser.parse_args(argv)
    run = load_run(args.run)
    out = Path(args.baseline)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": str(out), "run_id": run.get("run_id"), "summary": run.get("summary")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
