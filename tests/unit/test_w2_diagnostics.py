from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from debug_agent_system.agents.write import WriteSidePipeline
from debug_agent_system.eval.write_side.w2_family_diagnostics import build_report as build_family_report
from debug_agent_system.eval.write_side.w2_quality_diagnostics import build_report as build_quality_report
from debug_agent_system.knowledge.json_store import JsonKGStore


def _summary_episode() -> dict:
    return {
        "thread_id": "thread:test",
        "episodes": [
            {
                "episode_id": "ep:test",
                "thread_id": "thread:test",
                "completeness": "partial",
                "fault_description_messages": [{"message_id": "m1", "sender": "fae", "text": "设备开机无法启动，插拔内存无效。"}],
                "diagnostic_chain_messages": [{"message_id": "m2", "sender": "dev", "text": "拔除网卡显卡采集卡后仍无法启动。"}],
                "resolution_messages": [],
                "noise_messages": [],
                "evidence_message_ids": ["m1", "m2"],
                "source_offsets": [{"message_id": "m1", "index": 0}],
                "attachments": [],
                "extracted": {
                    "symptom_raw": "设备开机无法启动，插拔内存无效。",
                    "debug_actions": ["拔除网卡显卡采集卡后仍无法启动"],
                    "conclusion": "",
                },
            }
        ],
    }


def test_run_w2_summaries_writes_progress_file():
    tmp = tempfile.TemporaryDirectory()
    out_dir = Path(tmp.name) / "w2_run"
    pipeline = WriteSidePipeline(JsonKGStore("data/kg"), w2_mode="native_v2")
    result = pipeline.run_w2_summaries([_summary_episode()], out_dir=out_dir, w2_mode="native_v2")
    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["episodes_total"] == 1
    assert progress["episodes_completed"] == 1
    assert result["summary"]["top_families"]


def test_w2_family_diagnostics_report_flags_noncanonical_family():
    rows = [
        {
            "candidate_id": "cand:1",
            "label": "客户反馈复判站弹窗报错从buddv获取保存路径失败",
            "candidate_draft_v2": {
                "split_cases": [
                    {
                        "family": {"label": "客户反馈复判站弹窗报错从buddv获取保存路径失败"},
                        "variant": {"label": "复判站弹窗报错从buddv获取保存路径失败"},
                        "actions": [{"label": "导出日志"}],
                    }
                ]
            },
            "case_understanding_card": {"split_required": False},
        }
    ]
    report = build_family_report(rows, sample_limit=5)
    assert report["noncanonical_family_count"] == 1
    assert report["noncanonical_family_samples"][0]["candidate_id"] == "cand:1"


def test_w2_quality_diagnostics_flags_positive_status_noise():
    rows = [
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
        }
    ]
    report = build_quality_report(rows, sample_limit=5)
    assert report["counters"]["positive_no_issue"] == 1
    assert report["samples"]["positive_no_issue"][0]["candidate_id"] == "cand:2"


def test_run_w2_summaries_resume_from_partial_candidates():
    tmp = tempfile.TemporaryDirectory()
    out_dir = Path(tmp.name) / "w2_resume"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "w2_candidates.partial.jsonl"
    partial.write_text(json.dumps({
        "candidate_id": "cand:ep1",
        "source_episode_id": "ep:1",
        "schema_valid": True,
        "case_understanding_card_schema_valid": True,
        "candidate_draft_v2_schema_valid": True,
        "candidate_draft_v2_bundle_schema_valid": True,
        "observability": {"deepseek_used": False},
        "candidate_draft_v2": {
            "split_cases": [{
                "family": {"label": "主程序/系统异常", "subsystem": "主程序/系统"},
                "variant": {"label": "已缓存样本"},
                "actions": [],
                "required_info": []
            }]
        },
        "case_understanding_card": {"split_required": False},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    summaries = [{
        "thread_id": "thread:test",
        "episodes": [
            {
                "episode_id": "ep:1",
                "thread_id": "thread:test",
                "completeness": "partial",
                "fault_description_messages": [{"message_id": "m1", "sender": "fae", "text": "cached"}],
                "diagnostic_chain_messages": [],
                "resolution_messages": [],
                "noise_messages": [],
                "evidence_message_ids": ["m1"],
                "source_offsets": [],
                "attachments": [],
                "extracted": {"symptom_raw": "cached", "debug_actions": [], "conclusion": ""},
            },
            {
                "episode_id": "ep:2",
                "thread_id": "thread:test",
                "completeness": "partial",
                "fault_description_messages": [{"message_id": "m2", "sender": "fae", "text": "fresh"}],
                "diagnostic_chain_messages": [],
                "resolution_messages": [],
                "noise_messages": [],
                "evidence_message_ids": ["m2"],
                "source_offsets": [],
                "attachments": [],
                "extracted": {"symptom_raw": "fresh", "debug_actions": [], "conclusion": ""},
            },
        ],
    }]
    pipeline = WriteSidePipeline(JsonKGStore("data/kg"), w2_mode="native_v2")
    called: list[str] = []

    def fake_extract(episode: dict, *, w2_mode=None):
        called.append(str(episode.get("episode_id") or ""))
        return {
            "candidate_id": f"cand:{episode.get('episode_id')}",
            "source_episode_id": episode.get("episode_id"),
            "schema_valid": True,
            "case_understanding_card_schema_valid": True,
            "candidate_draft_v2_schema_valid": True,
            "candidate_draft_v2_bundle_schema_valid": True,
            "observability": {"deepseek_used": False},
            "candidate_draft_v2": {
                "split_cases": [{
                    "family": {"label": "主程序/系统异常", "subsystem": "主程序/系统"},
                    "variant": {"label": str(episode.get("episode_id"))},
                    "actions": [],
                    "required_info": []
                }]
            },
            "case_understanding_card": {"split_required": False},
        }

    pipeline.w2.extract = fake_extract
    result = pipeline.run_w2_summaries(summaries, out_dir=out_dir, w2_mode="native_v2")
    assert called == ["ep:2"]
    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["resumed_from_existing"] == 1
    assert result["summary"]["resumed_from_existing"] == 1


def test_run_w2_summaries_isolates_empty_episode_without_aborting_batch():
    tmp = tempfile.TemporaryDirectory()
    out_dir = Path(tmp.name) / "w2_empty_episode"
    summaries = [{
        "thread_id": "thread:mixed",
        "episodes": [
            {
                "episode_id": "ep:empty",
                "thread_id": "thread:mixed",
                "completeness": "noise",
                "fault_description_messages": [],
                "diagnostic_chain_messages": [],
                "resolution_messages": [],
                "noise_messages": [],
                "evidence_message_ids": [],
                "source_offsets": [],
                "attachments": [],
                "extracted": {},
            },
            {
                "episode_id": "ep:valid",
                "thread_id": "thread:mixed",
                "completeness": "partial",
                "fault_description_messages": [
                    {"message_id": "m1", "sender": "fae", "text": "工控机蓝屏，错误代码 0x00000139。"}
                ],
                "diagnostic_chain_messages": [],
                "resolution_messages": [],
                "noise_messages": [],
                "evidence_message_ids": ["m1"],
                "source_offsets": [{"message_id": "m1", "index": 0}],
                "attachments": [],
                "extracted": {"symptom_raw": "工控机蓝屏，错误代码 0x00000139。"},
            },
        ],
    }]
    pipeline = WriteSidePipeline(JsonKGStore("data/kg"), w2_mode="native_v2")
    result = pipeline.run_w2_summaries(
        summaries,
        out_dir=out_dir,
        emit_candidates=True,
        w2_workers=2,
        w2_mode="native_v2",
    )

    by_episode = {row["source_episode_id"]: row for row in result["candidates"]}
    assert set(by_episode) == {"ep:empty", "ep:valid"}
    failed = by_episode["ep:empty"]
    assert failed["schema_valid"] is False
    assert failed["extraction_error"]["code"] == "missing_intake_text"
    assert failed["schema_issues"] == ["extraction_invalid:missing_intake_text"]
    assert "extraction_error" not in by_episode["ep:valid"]
    assert result["summary"]["extraction_error_count"] == 1
    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["episodes_completed"] == 2

def test_w2_split_diagnostics_reports_split_clusters():
    from debug_agent_system.eval.write_side.w2_split_diagnostics import build_report
    rows = [{
        "candidate_id": "cand:split",
        "label": "正常测试时设备黑屏自动重启",
        "case_understanding_card": {
            "split_required": True,
            "cases": [
                {
                    "family_hypothesis": {"label": "工控机蓝屏"},
                    "variant_hypothesis": {"label": "0x00000139 关键数据结构损坏蓝屏"},
                },
                {
                    "family_hypothesis": {"label": "工控机异常重启"},
                    "variant_hypothesis": {"label": "正常测试时设备黑屏自动重启"},
                },
            ],
        },
    }]
    report = build_report(rows, sample_limit=5)
    assert report["split_required_count"] == 1
    assert report["top_split_family_pairs"][0][0] == "工控机异常重启 | 工控机蓝屏"


def test_run_candidates_processes_precomputed_rows_without_calling_w2():
    tmp = tempfile.TemporaryDirectory()
    kg_root = Path(tmp.name) / "kg"
    shutil.copytree("data/kg", kg_root)
    queue_dir = Path(tmp.name) / "review_queue"
    pipeline = WriteSidePipeline(JsonKGStore(kg_root), queue_dir=queue_dir, w2_mode="native_v2")

    def should_not_run(*args, **kwargs):
        raise AssertionError("run_candidates should not invoke W2.extract")

    pipeline.w2.extract = should_not_run
    candidate = {
        "candidate_id": "cand:downstream",
        "source_episode_id": "ep:downstream",
        "source_thread_id": "thread:downstream",
        "label": "工控机蓝屏 0x00000139",
        "confidence": 0.92,
        "schema_valid": True,
        "schema_issues": [],
        "category": "工控机/Windows 内核",
        "nodes": [
            {"type": "Error", "error_id": "err:industrial-pc-blue-screen", "label": "工控机蓝屏"},
            {"type": "DiagnosticCheck", "check_id": "check:dump-export", "label": "导出转储文件", "how_to_check": "导出 dmp", "step_order": 1},
            {"type": "Solution", "solution_id": "sol:replace-memory", "label": "更换内存条", "method": "更换内存条", "evidence_level": "field_verified"},
        ],
        "edges": [
            {"from": "err:industrial-pc-blue-screen", "relation": "has_check", "to": "check:dump-export"},
            {"from": "err:industrial-pc-blue-screen", "relation": "resolved_by", "to": "sol:replace-memory"},
        ],
        "evidence_ids": ["m1"],
        "source_offsets": [{"message_id": "m1", "index": 0}],
        "required_info_candidates": [],
        "episode": {
            "episode_id": "ep:downstream",
            "thread_id": "thread:downstream",
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "sender": "fae", "text": "现场蓝屏 0x00000139"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "sender": "dev", "text": "先导出转储文件再判断是否内存故障"}],
            "resolution_messages": [{"message_id": "m3", "sender": "dev", "text": "更换内存条后恢复正常"}],
            "noise_messages": [],
            "attachments": [],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "source_offsets": [{"message_id": "m1", "index": 0}],
        },
    }
    result = pipeline.run_candidates([candidate], kg_mode="legacy", dry_run_merge=False)
    assert result["summary"]["candidates"] == 1
    assert result["review_summary"]["candidates"] == 1
    queue_path = queue_dir / "candidates.json"
    assert queue_path.exists()
    rows = json.loads(queue_path.read_text(encoding="utf-8"))
    assert rows[0]["candidate_id"] == "cand:downstream"
