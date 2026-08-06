from __future__ import annotations

import json
import tempfile
from pathlib import Path

from debug_agent_system.eval.write_side.w2_postrun_report import build_postrun_report


def test_w2_postrun_report_composes_summary_and_diagnostics():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "w2_summaries_run.v1",
                "episodes": 2,
                "noncanonical_family_count": 1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "progress.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "episodes_total": 2,
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
            "label": "客户反馈复判站弹窗报错从buddv获取保存路径失败",
            "symptom_raw": "客户反馈复判站弹窗报错从buddv获取保存路径失败",
            "conclusion": "",
            "case_understanding_card": {"schema_valid": True, "split_required": False},
            "candidate_draft_v2": {
                "split_cases": [
                    {
                        "family": {"label": "客户反馈复判站弹窗报错从buddv获取保存路径失败"},
                        "variant": {"label": "复判站弹窗报错从buddv获取保存路径失败"},
                        "actions": [{"label": "导出日志"}],
                    }
                ]
            },
        },
        {
            "candidate_id": "cand:2",
            "label": "客户反馈说今天没有昨天也没有黑屏的情况",
            "symptom_raw": "客户反馈说今天没有昨天也没有黑屏的情况",
            "conclusion": "",
            "case_understanding_card": {"schema_valid": True, "split_required": False},
            "candidate_draft_v2": {
                "split_cases": [
                    {
                        "family": {"label": "工控机异常重启"},
                        "variant": {"label": "今天没有昨天也没有黑屏的情况"},
                        "actions": [{"label": "持续观察"}],
                    }
                ]
            },
        },
    ]
    with (root / "w2_candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_postrun_report(root)
    assert report["summary"]["episodes"] == 2
    assert report["family_diagnostics"]["noncanonical_family_count"] == 1
    assert report["quality_diagnostics"]["counters"]["positive_no_issue"] == 1
    assert report["split_diagnostics"]["split_required_count"] == 0
    assert report["quality_gate"]["status"] == "failed"
    assert report["recommended_next_steps"]
