from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _find_process(run_dir: str, run_pid_path: Path | None = None) -> dict[str, Any]:
    target_pid = ""
    if run_pid_path is not None and run_pid_path.exists():
        target_pid = run_pid_path.read_text(encoding="utf-8").strip()
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,etimes=,args="],
            text=True,
        )
    except Exception:
        return {}
    for line in out.splitlines():
        try:
            pid_str, etimes_str, cmd = line.strip().split(None, 2)
        except Exception:
            continue
        if target_pid and pid_str == target_pid:
            return {
                "pid": int(pid_str),
                "elapsed_sec": int(etimes_str),
                "cmd": cmd,
            }
        if not target_pid and run_dir in cmd:
            return {
                "pid": int(pid_str),
                "elapsed_sec": int(etimes_str),
                "cmd": cmd,
            }
    return {}


def _estimate_eta(progress: dict[str, Any], elapsed_sec: int | None) -> dict[str, Any]:
    total = int(progress.get("episodes_total") or 0)
    completed = int(progress.get("episodes_completed") or 0)
    if not total or not completed or not elapsed_sec or elapsed_sec <= 0:
        return {"rate_eps": 0.0, "eta_sec": None}
    rate = completed / float(elapsed_sec)
    remaining = max(0, total - completed)
    eta = int(round(remaining / rate)) if rate > 0 else None
    return {"rate_eps": round(rate, 4), "eta_sec": eta}


def build_status(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    progress = _read_json(root / "progress.json")
    summary = _read_json(root / "summary.json")
    postrun = _read_json(root / "postrun_report.json")
    process = _find_process(str(root), root / "run.pid")
    estimate = _estimate_eta(progress, process.get("elapsed_sec") if process else None)
    status = "missing"
    if progress:
        status = str(progress.get("status") or "unknown")
    if summary:
        status = "completed"
    if postrun:
        status = "postrun_completed"
    return {
        "schema_version": "debug_agent_system.w2_run_status.v1",
        "run_dir": str(root),
        "status": status,
        "progress": progress,
        "process": process,
        "estimate": estimate,
        "has_summary": bool(summary),
        "has_postrun_report": bool(postrun),
        "paths": {
            "progress": str(root / "progress.json"),
            "summary": str(root / "summary.json"),
            "postrun_report": str(root / "postrun_report.json"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_status(args.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
