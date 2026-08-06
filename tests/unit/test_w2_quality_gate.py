from __future__ import annotations

from debug_agent_system.eval.write_side.w2_quality_gate import gate_report


def test_w2_quality_gate_passes_clean_report():
    diagnostics = {
        "episodes": 100,
        "counters": {
            "noncanonical_family": 1,
            "pseudo_family": 0,
            "long_variant": 1,
            "questionish_variant": 0,
            "empty_case": 5,
            "report_noise": 3,
            "positive_no_issue": 4,
            "split_required": 5,
            "action_duplicates": 1,
        },
    }
    report = gate_report(diagnostics)
    assert report["status"] == "passed"


def test_w2_quality_gate_fails_bad_report():
    diagnostics = {
        "episodes": 100,
        "counters": {
            "noncanonical_family": 10,
            "pseudo_family": 2,
            "long_variant": 8,
            "questionish_variant": 1,
            "empty_case": 20,
            "report_noise": 10,
            "positive_no_issue": 20,
            "split_required": 25,
            "action_duplicates": 5,
        },
    }
    report = gate_report(diagnostics)
    assert report["status"] == "failed"
    assert any("noncanonical_family_rate_exceeded" in item for item in report["issues"])
    assert any("empty_case_rate_exceeded" in item for item in report["issues"])
    assert any("split_required_rate_exceeded" in item for item in report["issues"])
