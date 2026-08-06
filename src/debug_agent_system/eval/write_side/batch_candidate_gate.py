"""Gate review-only write-side batch candidate artifacts.

The gate validates artifacts produced by:

```
debug-agent-system ingest-xing ... --queue-dir <run>/review_queue > <run>/ingest_xing_summary.json
```

It deliberately checks review queue structure and approved-only safety, not KG
quality.  Semantic quality remains covered by the manual golden gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QUEUE_NAMES = ("candidates", "merge_candidates", "noise_candidates", "ask_info_candidates")


def gate_run_dir(
    run_dir: str | Path,
    *,
    min_candidates: int = 1,
    require_no_applied: bool = True,
    min_deepseek_used_rate: float = 0.0,
) -> dict[str, Any]:
    root = Path(run_dir)
    summary_path = root / "ingest_xing_summary.json"
    queue_dir = root / "review_queue"
    issues: list[str] = []
    if not summary_path.exists():
        return _report(root, {}, {}, issues=[f"missing_summary:{summary_path}"], status="failed")
    summary = _read_json_object(summary_path)
    if not queue_dir.exists():
        issues.append(f"missing_queue_dir:{queue_dir}")
    queues = {name: _read_json_list(queue_dir / f"{name}.json") for name in QUEUE_NAMES}
    summary_counts = summary.get("summary") if isinstance(summary.get("summary"), dict) else {}
    review_summary = summary.get("review_summary") if isinstance(summary.get("review_summary"), dict) else {}
    expected_counts = {
        "candidates": int(review_summary.get("candidates") or 0),
        "merge_candidates": int(review_summary.get("merge_candidates") or 0),
        "noise_candidates": int(review_summary.get("noise_candidates") or 0),
        "ask_info_candidates": int(review_summary.get("ask_info_candidates") or 0),
    }
    queue_counts = {name: len(items) for name, items in queues.items()}
    for name, expected in expected_counts.items():
        if queue_counts.get(name, 0) != expected:
            issues.append(f"queue_count_mismatch:{name}:expected={expected}:actual={queue_counts.get(name, 0)}")
    total_candidates = sum(queue_counts[name] for name in ("candidates", "merge_candidates", "noise_candidates"))
    if total_candidates < min_candidates:
        issues.append(f"insufficient_candidates:min={min_candidates}:actual={total_candidates}")
    if require_no_applied:
        if int(summary_counts.get("applied") or 0) != 0:
            issues.append(f"unexpected_applied:{summary_counts.get('applied')}")
        if int(summary_counts.get("required_info_applied") or 0) != 0:
            issues.append(f"unexpected_required_info_applied:{summary_counts.get('required_info_applied')}")
    structural = _queue_structure_issues(queues)
    issues.extend(structural)
    deepseek = _deepseek_stats(queues)
    if min_deepseek_used_rate > 0 and deepseek["enabled"] > 0 and deepseek["used_rate"] < min_deepseek_used_rate:
        issues.append(f"deepseek_used_rate_below_min:min={min_deepseek_used_rate}:actual={deepseek['used_rate']}")
    if min_deepseek_used_rate > 0 and deepseek["enabled"] == 0:
        issues.append("deepseek_not_enabled")
    return _report(
        root,
        summary,
        {
            "queue_counts": queue_counts,
            "total_candidates": total_candidates,
            "deepseek": deepseek,
        },
        issues=issues,
        status="passed" if not issues else "failed",
    )


def _queue_structure_issues(queues: dict[str, list[dict[str, Any]]]) -> list[str]:
    issues: list[str] = []
    for name, items in queues.items():
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                issues.append(f"{name}[{idx}]:not_object")
                continue
            required = {"review_id", "queue", "episode", "quality_gate", "evidence_pack", "review_actions"}
            missing = sorted(required - set(item))
            if missing:
                issues.append(f"{name}[{idx}]:missing:{','.join(missing)}")
            if item.get("queue") != name:
                issues.append(f"{name}[{idx}]:queue_mismatch:{item.get('queue')}")
            summary = item.get("review_summary") if isinstance(item.get("review_summary"), dict) else {}
            if not summary:
                issues.append(f"{name}[{idx}]:missing_review_summary")
            elif not summary.get("title") and not summary.get("candidate_id"):
                issues.append(f"{name}[{idx}]:bad_review_summary")
            evidence = item.get("evidence_pack") if isinstance(item.get("evidence_pack"), dict) else {}
            if not evidence.get("messages"):
                issues.append(f"{name}[{idx}]:empty_evidence_messages")
            if name == "ask_info_candidates":
                candidate = item.get("required_info_candidate") if isinstance(item.get("required_info_candidate"), dict) else {}
                if not candidate.get("candidate_id"):
                    issues.append(f"{name}[{idx}]:missing_required_info_candidate_id")
                if not candidate.get("slot") or not candidate.get("question"):
                    issues.append(f"{name}[{idx}]:bad_required_info_candidate")
                if not isinstance(item.get("dry_run_required_info_merge_plan"), dict):
                    issues.append(f"{name}[{idx}]:missing_required_info_dry_run")
            else:
                candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
                if not candidate.get("candidate_id"):
                    issues.append(f"{name}[{idx}]:missing_candidate_id")
                if not candidate.get("schema_valid"):
                    issues.append(f"{name}[{idx}]:schema_invalid:{candidate.get('schema_issues')}")
                dry_run = item.get("dry_run_merge_plan") if isinstance(item.get("dry_run_merge_plan"), dict) else {}
                if dry_run.get("status") != "dry_run_merge_plan":
                    issues.append(f"{name}[{idx}]:missing_dry_run_merge_plan")
    return issues


def _deepseek_stats(queues: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidates = [
        item.get("candidate") or {}
        for name in ("candidates", "merge_candidates", "noise_candidates")
        for item in queues.get(name, [])
        if isinstance(item, dict)
    ]
    enabled = sum(1 for candidate in candidates if ((candidate.get("observability") or {}).get("deepseek_enabled")))
    used = sum(1 for candidate in candidates if ((candidate.get("observability") or {}).get("deepseek_used")))
    errors = sum(1 for candidate in candidates if ((candidate.get("observability") or {}).get("deepseek_error")))
    return {
        "candidates": len(candidates),
        "enabled": enabled,
        "used": used,
        "errors": errors,
        "used_rate": round(used / enabled, 4) if enabled else 0.0,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list JSON: {path}")
    return [item for item in data if isinstance(item, dict)]


def _report(
    run_dir: Path,
    summary: dict[str, Any],
    checks: dict[str, Any],
    *,
    issues: list[str],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.write_batch_gate.v1",
        "status": status,
        "run_dir": str(run_dir),
        "summary": summary.get("summary") if isinstance(summary.get("summary"), dict) else {},
        "review_summary": summary.get("review_summary") if isinstance(summary.get("review_summary"), dict) else {},
        "checks": checks,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--allow-applied", action="store_true")
    parser.add_argument("--min-deepseek-used-rate", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = gate_run_dir(
        args.run_dir,
        min_candidates=args.min_candidates,
        require_no_applied=not args.allow_applied,
        min_deepseek_used_rate=args.min_deepseek_used_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
