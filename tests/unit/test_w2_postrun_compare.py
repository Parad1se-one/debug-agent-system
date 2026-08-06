from __future__ import annotations

from debug_agent_system.eval.write_side.w2_postrun_compare import compare_reports


def test_w2_postrun_compare_detects_improvement():
    base = {
        "run_dir": "base",
        "summary": {"episodes": 100, "deepseek_used": 100},
        "family_diagnostics": {
            "noncanonical_family_count": 10,
            "pseudo_family_count": 5,
            "long_variant_count": 8,
            "questionish_variant_count": 2,
            "split_required_count": 15,
        },
        "quality_diagnostics": {
            "counters": {
                "empty_case": 12,
                "report_noise": 9,
                "positive_no_issue": 11,
            }
        },
        "quality_gate": {"checks": {"noncanonical_family_rate": 0.1, "empty_case_rate": 0.12}},
    }
    candidate = {
        "run_dir": "cand",
        "summary": {"episodes": 100, "deepseek_used": 100},
        "family_diagnostics": {
            "noncanonical_family_count": 4,
            "pseudo_family_count": 1,
            "long_variant_count": 3,
            "questionish_variant_count": 0,
            "split_required_count": 10,
        },
        "quality_diagnostics": {
            "counters": {
                "empty_case": 6,
                "report_noise": 4,
                "positive_no_issue": 7,
            }
        },
        "quality_gate": {"checks": {"noncanonical_family_rate": 0.04, "empty_case_rate": 0.06}},
    }
    report = compare_reports(base, candidate)
    assert report["metrics"]["noncanonical_family_delta"] == -6
    assert "noncanonical_family_delta" in report["improvements"]
    assert not report["regressions"]
