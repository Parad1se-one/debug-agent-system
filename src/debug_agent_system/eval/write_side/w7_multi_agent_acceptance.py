"""Mode-aware release gate for the W7 multi-agent shadow pipeline.

The existing safety gate intentionally proves *non-mutation* of a shadow run.
It must not be used as a promotion decision by itself.  This module combines
that safety report with the human calibration score and an optional held-out
report, then emits an explicit recommendation.  It never changes W7_MODE or
writes a queue/KG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.w7_trace.contracts import canonical_hash


def _load(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    *,
    calibration: dict[str, Any],
    fixed173_safety: dict[str, Any],
    heldout: dict[str, Any] | None = None,
    min_calibration_strict_rate: float = 1.0,
    min_calibration_trace_f1: float = 0.9,
    require_heldout: bool = True,
) -> dict[str, Any]:
    """Return a fail-closed promotion recommendation.

    ``heldout`` may be either a safety-gate report or a score report.  For a
    score report its own ``gate.status`` must be PASS; for a safety report the
    same applies.  A missing held-out report is a hard failure by default.
    """

    calibration_metrics = (
        calibration.get("metrics")
        if isinstance(calibration.get("metrics"), dict)
        else {}
    )
    strict = (
        calibration_metrics.get("strict_episode_match")
        if isinstance(calibration_metrics.get("strict_episode_match"), dict)
        else {}
    )
    trace = (
        calibration_metrics.get("trace_pairwise")
        if isinstance(calibration_metrics.get("trace_pairwise"), dict)
        else {}
    )
    safety_gate = (
        fixed173_safety.get("gate")
        if isinstance(fixed173_safety.get("gate"), dict)
        else {}
    )
    safety_requirements = (
        fixed173_safety.get("requirements")
        if isinstance(fixed173_safety.get("requirements"), dict)
        else {}
    )
    heldout_gate = (
        heldout.get("gate")
        if isinstance(heldout, dict)
        and isinstance(heldout.get("gate"), dict)
        else {}
    )
    requirements = {
        "calibration_gate_pass": str(
            (calibration.get("gate") or {}).get("status") or ""
        ) == "PASS",
        "calibration_strict_episode_rate": float(
            strict.get("rate") or 0.0
        ) >= float(min_calibration_strict_rate),
        "calibration_trace_pairwise_f1": float(
            trace.get("f1") or 0.0
        ) >= float(min_calibration_trace_f1),
        "fixed173_safety_pass": str(safety_gate.get("status") or "") == "PASS",
        "fixed173_expected_coverage": bool(
            safety_requirements.get("episode_coverage_exact")
        ),
        "fixed173_state_unchanged": bool(
            safety_requirements.get("state_unchanged")
        ),
        "heldout_present": bool(heldout) or not require_heldout,
        "heldout_pass": (
            str(heldout_gate.get("status") or "") == "PASS"
            if heldout
            else not require_heldout
        ),
    }
    report = {
        "schema_version": "w7.multi_agent_acceptance_gate.v1",
        "requirements": requirements,
        "inputs": {
            "calibration_manifest": str(calibration.get("manifest") or ""),
            "fixed173_manifest": str(
                fixed173_safety.get("manifest") or ""
            ),
            "heldout_manifest": str(
                (heldout or {}).get("manifest") or ""
            ),
        },
        "thresholds": {
            "min_calibration_strict_rate": min_calibration_strict_rate,
            "min_calibration_trace_f1": min_calibration_trace_f1,
            "require_heldout": require_heldout,
        },
        "recommendation": {
            "status": "PASS" if all(requirements.values()) else "FAIL",
            "promotion_ready": all(requirements.values()),
            "next_mode": "assisted" if all(requirements.values()) else "shadow_multi_agent",
            "legacy_fallback_required": True,
            "human_approval_required": True,
        },
    }
    report["report_hash"] = canonical_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="w7-multi-agent-acceptance")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--fixed173-safety", type=Path, required=True)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--min-strict-rate", type=float, default=1.0)
    parser.add_argument("--min-trace-f1", type=float, default=0.9)
    parser.add_argument("--allow-missing-heldout", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        calibration=_load(args.calibration),
        fixed173_safety=_load(args.fixed173_safety),
        heldout=_load(args.heldout) if args.heldout else None,
        min_calibration_strict_rate=args.min_strict_rate,
        min_calibration_trace_f1=args.min_trace_f1,
        require_heldout=not args.allow_missing_heldout,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["recommendation"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
