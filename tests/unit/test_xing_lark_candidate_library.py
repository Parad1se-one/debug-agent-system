from datetime import datetime
import json
from pathlib import Path

from debug_agent_system.eval.write_side.build_xing_lark_candidate_library import (
    Anchor,
    _cluster_anchors,
    _issue_tags,
    _score,
    load_known_gold_message_ids,
)
from debug_agent_system.eval.write_side.freeze_xing_lark_heldout import (
    _external_artifacts,
    _linked_jira_issues,
    derive_gold_cutoff,
    load_embedded_gold_time_bounds,
    select_eligible_candidates,
)


def _anchor(message_id: str, session_id: str, minute: int) -> Anchor:
    return Anchor(
        message={
            "message_id": message_id,
            "chat_id": "chat-1",
            "relation_aware_session_id": session_id,
        },
        time=datetime(2026, 7, 1, 10, minute),
        signals={"issue"},
    )


def test_cluster_anchors_keeps_parallel_relation_sessions_separate() -> None:
    clusters = _cluster_anchors([
        _anchor("m-1", "session-a", 0),
        _anchor("m-2", "session-b", 1),
        _anchor("m-3", "session-a", 2),
    ])

    assert len(clusters) == 2
    assert sorted(len(cluster.anchors) for cluster in clusters) == [1, 2]


def test_issue_tags_support_longitudinal_candidate_linking() -> None:
    tags = _issue_tags([
        "硬盘有坏块，克隆完成后仍卡顿",
        "系统黑屏自动重启",
    ])

    assert tags == ["performance", "storage", "system_crash"]


def test_known_gold_overlap_is_strongly_penalized() -> None:
    signals = {"issue", "diagnosis", "action", "resolution", "attachment"}

    assert _score(signals, 8, known_overlap=True, weak_only=False) < _score(
        signals,
        8,
        known_overlap=False,
        weak_only=False,
    )


def test_known_gold_ids_include_gold_v2_inputs(tmp_path: Path) -> None:
    path = tmp_path / "data/annotations/goldcases/gold-v2/inputs/goldcase-020.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"messages": [{"message_id": "om_x100b6cc58251d0a4c3157af3d744af5"}]}), encoding="utf-8")

    assert "om_x100b6cc58251d0a4c3157af3d744af5" in load_known_gold_message_ids(tmp_path)


def test_heldout_selection_requires_complete_session_after_cutoff() -> None:
    library = {
        "candidates": [
            {
                "candidate_id": "old-session",
                "chat_id": "chat",
                "relation_aware_session_ids": ["session-old"],
                "score": 99,
            },
            {
                "candidate_id": "future-session",
                "chat_id": "chat",
                "relation_aware_session_ids": ["session-future"],
                "score": 80,
            },
        ]
    }
    sessions = {
        ("chat", "session-old"): [
            {"message_id": "om_old", "create_time": "2026-07-08 03:26"},
            {"message_id": "om_late", "create_time": "2026-07-09 03:26"},
        ],
        ("chat", "session-future"): [
            {"message_id": "om_future", "create_time": "2026-07-09 04:00"},
        ],
    }

    eligible, rejected = select_eligible_candidates(
        library,
        sessions,
        {"om_old"},
        "2026-07-08 03:26",
    )

    assert [row["candidate_id"] for row in eligible] == ["future-session"]
    old = next(row for row in rejected if row["candidate_id"] == "old-session")
    assert "known_gold_message_overlap" in old["reasons"]
    assert "session_not_strictly_after_cutoff" in old["reasons"]


def test_heldout_cutoff_uses_latest_resolved_gold_message() -> None:
    cutoff, unresolved = derive_gold_cutoff(
        {
            "om_1": {"create_time": "2026-07-01 10:00"},
            "om_2": {"create_time": "2026-07-08 03:26"},
        },
        {"om_1", "om_2", "om_missing"},
    )

    assert cutoff == "2026-07-08 03:26"
    assert unresolved == ["om_missing"]


def test_embedded_gold_parent_relation_provides_safe_time_bound(tmp_path: Path) -> None:
    path = tmp_path / "data/annotations/goldcases/gold-v2/inputs/goldcase-020.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "analysis_window": {"end_inclusive": "2026-07-08 03:26"},
            "messages": [{
                "message_id": "om_x100child",
                "root_id": "om_x100missingroot",
                "create_time": "2026-07-07 09:10",
            }],
        }),
        encoding="utf-8",
    )

    times, bounded = load_embedded_gold_time_bounds(
        tmp_path,
        {"om_x100child", "om_x100missingroot"},
    )

    assert max(times) == "2026-07-08 03:26"
    assert bounded == {"om_x100child", "om_x100missingroot"}


def test_additional_gold_time_bound_can_raise_cutoff() -> None:
    cutoff, unresolved = derive_gold_cutoff(
        {"om_1": {"create_time": "2026-07-01 10:00"}},
        {"om_1", "om_missing"},
        additional_time_bounds=["2026-07-08 03:26"],
    )

    assert cutoff == "2026-07-08 03:26"
    assert unresolved == ["om_missing"]


def test_heldout_evidence_snapshot_hashes_jira_and_available_attachment(tmp_path: Path) -> None:
    jira = tmp_path / "data/imports/jira_offline/raw/fault_details/TEST-1234.json"
    jira.parent.mkdir(parents=True)
    jira.write_text(
        json.dumps({
            "summary": "相机不拍摄",
            "description": "检查采集链",
            "status": "Done",
            "comments": [{"body": "更换线缆后复测"}],
        }),
        encoding="utf-8",
    )
    attachment = tmp_path / "artifacts/diagnostic.zip"
    attachment.parent.mkdir()
    attachment.write_bytes(b"diagnostic-data")
    messages = [{
        "message_id": "om_x100source",
        "text": "已提交 TEST-1234",
        "attachments": [{"name": "diagnostic.zip", "path": str(attachment)}],
    }]

    jira_rows = _linked_jira_issues(messages, tmp_path)
    artifact_rows = _external_artifacts(messages, tmp_path)

    assert jira_rows[0]["retrieval_status"] == "frozen_local_snapshot"
    assert len(jira_rows[0]["source_file_sha256"]) == 64
    assert artifact_rows[0]["retrieval_status"] == "local_file_available"
    assert artifact_rows[0]["size_bytes"] == len(b"diagnostic-data")
    assert len(artifact_rows[0]["file_sha256"]) == 64
