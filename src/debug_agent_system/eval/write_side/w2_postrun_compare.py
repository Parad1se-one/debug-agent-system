from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object json: {path}")
    return data


def compare_reports(base_report: dict[str, Any], candidate_report: dict[str, Any]) -> dict[str, Any]:
    base_summary = base_report.get("summary") if isinstance(base_report.get("summary"), dict) else {}
    cand_summary = candidate_report.get("summary") if isinstance(candidate_report.get("summary"), dict) else {}
    base_family = base_report.get("family_diagnostics") if isinstance(base_report.get("family_diagnostics"), dict) else {}
    cand_family = candidate_report.get("family_diagnostics") if isinstance(candidate_report.get("family_diagnostics"), dict) else {}
    base_quality = (base_report.get("quality_diagnostics") or {}).get("counters") if isinstance(base_report.get("quality_diagnostics"), dict) else {}
    cand_quality = (candidate_report.get("quality_diagnostics") or {}).get("counters") if isinstance(candidate_report.get("quality_diagnostics"), dict) else {}
    if not isinstance(base_quality, dict):
        base_quality = {}
    if not isinstance(cand_quality, dict):
        cand_quality = {}

    def delta(base: Any, cand: Any) -> int:
        return int(cand or 0) - int(base or 0)

    metrics = {
        "episodes_delta": delta(base_summary.get("episodes"), cand_summary.get("episodes")),
        "deepseek_used_delta": delta(base_summary.get("deepseek_used"), cand_summary.get("deepseek_used")),
        "noncanonical_family_delta": delta(base_family.get("noncanonical_family_count"), cand_family.get("noncanonical_family_count")),
        "pseudo_family_delta": delta(base_family.get("pseudo_family_count"), cand_family.get("pseudo_family_count")),
        "long_variant_delta": delta(base_family.get("long_variant_count"), cand_family.get("long_variant_count")),
        "questionish_variant_delta": delta(base_family.get("questionish_variant_count"), cand_family.get("questionish_variant_count")),
        "split_required_delta": delta(base_family.get("split_required_count"), cand_family.get("split_required_count")),
        "empty_case_delta": delta(base_quality.get("empty_case"), cand_quality.get("empty_case")),
        "report_noise_delta": delta(base_quality.get("report_noise"), cand_quality.get("report_noise")),
        "positive_no_issue_delta": delta(base_quality.get("positive_no_issue"), cand_quality.get("positive_no_issue")),
        "noncanonical_family_rate_delta": round(
            float((candidate_report.get("quality_gate") or {}).get("checks", {}).get("noncanonical_family_rate") or 0.0)
            - float((base_report.get("quality_gate") or {}).get("checks", {}).get("noncanonical_family_rate") or 0.0),
            6,
        ),
        "empty_case_rate_delta": round(
            float((candidate_report.get("quality_gate") or {}).get("checks", {}).get("empty_case_rate") or 0.0)
            - float((base_report.get("quality_gate") or {}).get("checks", {}).get("empty_case_rate") or 0.0),
            6,
        ),
    }
    improvements = [key for key, value in metrics.items() if key.endswith("_delta") and value < 0]
    regressions = [key for key, value in metrics.items() if key.endswith("_delta") and value > 0 and not key.startswith("episodes_") and not key.startswith("deepseek_used_")]
    return {
        "schema_version": "debug_agent_system.w2_postrun_compare.v1",
        "base_run_dir": base_report.get("run_dir") or "",
        "candidate_run_dir": candidate_report.get("run_dir") or "",
        "metrics": metrics,
        "improvements": improvements,
        "regressions": regressions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = compare_reports(_load(Path(args.base)), _load(Path(args.candidate)))
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
