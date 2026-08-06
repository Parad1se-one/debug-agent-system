from __future__ import annotations

import json
import tempfile
from pathlib import Path

from debug_agent_system.eval.write_side.w2_live_report import build_live_report


def test_w2_live_report_uses_partial_candidates_when_summary_missing():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "episodes_total": 10,
                "episodes_completed": 2,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "candidate_id": "cand:1",
            "label": "客户反馈说今天没有昨天也没有黑屏的情况",
            "symptom_raw": "客户反馈说今天没有昨天也没有黑屏的情况",
            "conclusion": "",
            "case_understanding_card": {"schema_valid": True, "split_required": False},
            "candidate_draft_v2": {
                "split_cases": [
                    {
                        "family": {"label": "工控机异常重启", "subsystem": "工控机/系统运行稳定性"},
                        "variant": {"label": "今天没有昨天也没有黑屏的情况"},
                        "actions": [{"label": "持续观察"}],
                    }
                ]
            },
        }
    ]
    with (root / "w2_candidates.partial.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_live_report(root)
    assert report["using_partial_candidates"] is True
    assert report["progress"]["episodes_completed"] == 2
    assert report["quality_diagnostics"]["counters"]["positive_no_issue"] == 1
    assert report["split_diagnostics"]["split_required_count"] == 0

def test_w2_live_compare_computes_deltas():
    from debug_agent_system.eval.write_side.w2_live_compare import compare_reports
    base = {
        "run_dir": "base",
        "progress": {"episodes_completed": 50},
        "quality_diagnostics": {"counters": {"split_required": 20, "empty_case": 5, "report_noise": 4, "long_variant": 3, "noncanonical_family": 1}},
        "quality_gate": {"checks": {"split_required_rate": 0.2, "report_noise_rate": 0.04, "long_variant_rate": 0.03, "empty_case_rate": 0.05, "noncanonical_family_rate": 0.01}},
        "status": "running",
    }
    candidate = {
        "run_dir": "cand",
        "progress": {"episodes_completed": 80},
        "quality_diagnostics": {"counters": {"split_required": 18, "empty_case": 4, "report_noise": 6, "long_variant": 2, "noncanonical_family": 0}},
        "quality_gate": {"checks": {"split_required_rate": 0.15, "report_noise_rate": 0.05, "long_variant_rate": 0.02, "empty_case_rate": 0.04, "noncanonical_family_rate": 0.0}},
        "status": "running",
    }
    report = compare_reports(base, candidate)
    assert report["metrics"]["base_episodes"] == 50
    assert report["metrics"]["candidate_episodes"] == 80
    assert report["metrics"]["split_required_rate_delta"] == -0.05
    assert report["metrics"]["report_noise_rate_delta"] == 0.01

def test_w2_live_compare_uses_rates_not_raw_counts():
    from debug_agent_system.eval.write_side.w2_live_compare import compare_reports
    base = {
        "run_dir": "base",
        "progress": {"episodes_completed": 100},
        "quality_gate": {"checks": {"split_required_rate": 0.20, "report_noise_rate": 0.05, "long_variant_rate": 0.03, "empty_case_rate": 0.04, "noncanonical_family_rate": 0.01}},
        "quality_diagnostics": {"counters": {"split_required": 20, "report_noise": 5, "long_variant": 3, "empty_case": 4, "noncanonical_family": 1}},
        "status": "running",
    }
    candidate = {
        "run_dir": "cand",
        "progress": {"episodes_completed": 200},
        "quality_gate": {"checks": {"split_required_rate": 0.10, "report_noise_rate": 0.03, "long_variant_rate": 0.02, "empty_case_rate": 0.02, "noncanonical_family_rate": 0.0}},
        "quality_diagnostics": {"counters": {"split_required": 20, "report_noise": 6, "long_variant": 4, "empty_case": 4, "noncanonical_family": 0}},
        "status": "running",
    }
    report = compare_reports(base, candidate)
    assert report["metrics"]["base_episodes"] == 100
    assert report["metrics"]["candidate_episodes"] == 200
    assert report["metrics"]["split_required_rate_delta"] == -0.1
    assert report["metrics"]["report_noise_rate_delta"] == -0.02
