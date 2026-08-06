"""Gate W2 native_v2 quality diagnostics for batch/full runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def gate_report(
    diagnostics: dict[str, Any],
    *,
    max_noncanonical_family_rate: float = 0.02,
    max_pseudo_family_rate: float = 0.0,
    max_long_variant_rate: float = 0.03,
    max_questionish_variant_rate: float = 0.0,
    max_empty_case_rate: float = 0.12,
    max_report_noise_rate: float = 0.08,
    max_positive_status_rate: float = 0.12,
    max_split_required_rate: float = 0.12,
    max_action_duplicates_rate: float = 0.02,
) -> dict[str, Any]:
    episodes = int(diagnostics.get("episodes") or 0)
    counters = diagnostics.get("counters") if isinstance(diagnostics.get("counters"), dict) else {}
    if episodes <= 0:
        return {
            "schema_version": "debug_agent_system.w2_quality_gate.v1",
            "status": "failed",
            "episodes": episodes,
            "issues": ["invalid_episodes"],
            "checks": {},
        }

    def ratio(key: str) -> float:
        return round(float(counters.get(key) or 0) / float(episodes), 6)

    checks = {
        "noncanonical_family_rate": ratio("noncanonical_family"),
        "pseudo_family_rate": ratio("pseudo_family"),
        "long_variant_rate": ratio("long_variant"),
        "questionish_variant_rate": ratio("questionish_variant"),
        "empty_case_rate": ratio("empty_case"),
        "report_noise_rate": ratio("report_noise"),
        "positive_status_rate": ratio("positive_no_issue"),
        "split_required_rate": ratio("split_required"),
        "action_duplicates_rate": ratio("action_duplicates"),
    }
    issues: list[str] = []
    if checks["noncanonical_family_rate"] > max_noncanonical_family_rate:
        issues.append(f"noncanonical_family_rate_exceeded:{checks['noncanonical_family_rate']}>{max_noncanonical_family_rate}")
    if checks["pseudo_family_rate"] > max_pseudo_family_rate:
        issues.append(f"pseudo_family_rate_exceeded:{checks['pseudo_family_rate']}>{max_pseudo_family_rate}")
    if checks["long_variant_rate"] > max_long_variant_rate:
        issues.append(f"long_variant_rate_exceeded:{checks['long_variant_rate']}>{max_long_variant_rate}")
    if checks["questionish_variant_rate"] > max_questionish_variant_rate:
        issues.append(f"questionish_variant_rate_exceeded:{checks['questionish_variant_rate']}>{max_questionish_variant_rate}")
    if checks["empty_case_rate"] > max_empty_case_rate:
        issues.append(f"empty_case_rate_exceeded:{checks['empty_case_rate']}>{max_empty_case_rate}")
    if checks["report_noise_rate"] > max_report_noise_rate:
        issues.append(f"report_noise_rate_exceeded:{checks['report_noise_rate']}>{max_report_noise_rate}")
    if checks["positive_status_rate"] > max_positive_status_rate:
        issues.append(f"positive_status_rate_exceeded:{checks['positive_status_rate']}>{max_positive_status_rate}")
    if checks["split_required_rate"] > max_split_required_rate:
        issues.append(f"split_required_rate_exceeded:{checks['split_required_rate']}>{max_split_required_rate}")
    if checks["action_duplicates_rate"] > max_action_duplicates_rate:
        issues.append(f"action_duplicates_rate_exceeded:{checks['action_duplicates_rate']}>{max_action_duplicates_rate}")
    return {
        "schema_version": "debug_agent_system.w2_quality_gate.v1",
        "status": "passed" if not issues else "failed",
        "episodes": episodes,
        "checks": checks,
        "thresholds": {
            "max_noncanonical_family_rate": max_noncanonical_family_rate,
            "max_pseudo_family_rate": max_pseudo_family_rate,
            "max_long_variant_rate": max_long_variant_rate,
            "max_questionish_variant_rate": max_questionish_variant_rate,
            "max_empty_case_rate": max_empty_case_rate,
            "max_report_noise_rate": max_report_noise_rate,
            "max_positive_status_rate": max_positive_status_rate,
            "max_split_required_rate": max_split_required_rate,
            "max_action_duplicates_rate": max_action_duplicates_rate,
        },
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--max-noncanonical-family-rate", type=float, default=0.02)
    parser.add_argument("--max-pseudo-family-rate", type=float, default=0.0)
    parser.add_argument("--max-long-variant-rate", type=float, default=0.03)
    parser.add_argument("--max-questionish-variant-rate", type=float, default=0.0)
    parser.add_argument("--max-empty-case-rate", type=float, default=0.12)
    parser.add_argument("--max-report-noise-rate", type=float, default=0.08)
    parser.add_argument("--max-positive-status-rate", type=float, default=0.12)
    parser.add_argument("--max-split-required-rate", type=float, default=0.12)
    parser.add_argument("--max-action-duplicates-rate", type=float, default=0.02)
    args = parser.parse_args(argv)
    diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    report = gate_report(
        diagnostics,
        max_noncanonical_family_rate=args.max_noncanonical_family_rate,
        max_pseudo_family_rate=args.max_pseudo_family_rate,
        max_long_variant_rate=args.max_long_variant_rate,
        max_questionish_variant_rate=args.max_questionish_variant_rate,
        max_empty_case_rate=args.max_empty_case_rate,
        max_report_noise_rate=args.max_report_noise_rate,
        max_positive_status_rate=args.max_positive_status_rate,
        max_split_required_rate=args.max_split_required_rate,
        max_action_duplicates_rate=args.max_action_duplicates_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
