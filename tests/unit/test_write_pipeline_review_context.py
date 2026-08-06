from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.agents.write import WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore


def _episode() -> dict:
    return {
        "episode_id": "ep-pipe-001",
        "thread_id": "thread-pipe-001",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "text": "@工程师乙 客户反馈设备正常运行中出现蓝屏现象，发生时间12:55左右，麻烦看下。"}
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "text": "远程收集日志并使用蓝屏dmp脚本收集dmp文件"},
            {"message_id": "m3", "text": "分析dmp定位蓝屏原因"},
        ],
        "resolution_messages": [],
        "noise_messages": [],
        "case_context_messages": [],
        "attachments": [],
        "extracted": {},
    }


def test_write_pipeline_prepare_episode_injects_review_context():
    pipeline = WriteSidePipeline(JsonKGStore("data/kg"), w2_mode="native_v2")
    episode = _episode()
    episode["extracted"] = {
        "attribution": {
            "reporter_candidates": [{"name": "现场FAE", "role_type": "reporter", "confidence": 0.6, "reason": ["field_feedback"], "evidence_message_ids": ["m1"]}],
            "owner_candidates": [{"name": "工程师乙", "role_type": "issue_owner", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["m1"]}],
            "owner_assignments": [{"name": "工程师乙", "role_type": "issue_owner", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["m1"]}],
            "responsibility_signals": [{"message_id": "m1", "signal_type": "owner_assignment", "name": "工程师乙", "reason": ["direct_owner_request"]}],
            "classification_hypotheses": [{"name": "工程师乙", "role_type": "issue_owner", "problem_category": "工控机/复判站/编程站及操作系统问题", "confidence": 0.55, "reason": ["matched:蓝屏"], "evidence_message_ids": ["m1"]}],
        }
    }
    prepared = pipeline._prepare_episode_for_w2(episode)
    extracted = prepared["extracted"]
    assert isinstance(extracted.get("review_context"), dict)
    assert extracted["review_context"]["context_role"] == "alignment_only"
    assert extracted["sop_background"] == extracted["review_context"]
    assert extracted["fault_focus_text"] == "客户反馈设备正常运行中出现蓝屏现象，发生时间12:55左右，麻烦看下"
    assert extracted["fault_focus_confidence"] > 0.0
    assert extracted["attribution"]["sanitized_by"] == "W7"
    assert extracted["attribution"]["owner_assignments"][0]["name"] == "工程师乙"
    assert extracted["attribution_raw"]["owner_assignments"][0]["name"] == "工程师乙"


def test_write_pipeline_progress_writer_emits_json(tmp_path: Path):
    path = tmp_path / "pipeline_progress.json"
    WriteSidePipeline._write_progress(path, stage="w2_extract", payload={"episodes_total": 10, "episodes_completed": 3})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "w2_extract"
    assert payload["episodes_total"] == 10
    assert payload["episodes_completed"] == 3
    assert "timestamp" in payload


def test_pipeline_summary_expansion_attaches_w7_trace_metadata_without_merging_episodes():
    left = _episode()
    right = {**_episode(), "episode_id": "ep-pipe-002"}

    episodes = WriteSidePipeline._episodes_from_summaries(
        [{"thread_id": "thread-pipe-001", "episodes": [left, right]}],
        refine_trace=True,
    )

    assert [item["episode_id"] for item in episodes] == ["ep-pipe-001", "ep-pipe-002"]
    assert episodes[0]["trace_group_id"] == episodes[1]["trace_group_id"]
    assert episodes[1]["previous_trace_episode_id"] == "ep-pipe-001"
    assert episodes[1]["trace_link_strength"] == "hard"
    # Trace metadata is navigation/review context; source evidence remains
    # episode-local and is never concatenated into the neighbouring phase.
    assert len(episodes[0]["fault_description_messages"]) == 1
    assert len(episodes[1]["fault_description_messages"]) == 1


def test_hydrate_v2_bundle_uses_full_episode_evidence_boundary():
    episode = _episode()
    episode["evidence_message_ids"] = ["m1", "m2", "m3", "m4"]
    episode["case_evidence_messages"] = [
        {"message_id": "m4", "text": "分析日志后确认是软件 BUG，并创建 Jira。"}
    ]

    hydrated = WriteSidePipeline._hydrate_v2_bundle_identity(
        {"candidate_id": "candidate:test"},
        episode,
        {"objects": {}, "relations": []},
    )

    assert hydrated["source_message_ids"] == ["m1", "m2", "m3", "m4"]
    assert hydrated["source_messages"][-1] == {
        "message_id": "m4",
        "role": "w7_promoted",
        "text": "分析日志后确认是软件 BUG，并创建 Jira。",
    }
