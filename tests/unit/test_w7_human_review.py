from __future__ import annotations

from debug_agent_system.eval.write_side.w7_human_review import (
    BOOLEAN_FIELDS,
    build_adjudicated_dataset,
    build_template,
    merge_structured_corrections,
    validate_annotations,
)


def _pack() -> dict:
    return {
        "summary": {"input": "episodes.json"},
        "cases": [{
            "thread_id": "thread:1",
            "full_context_markdown": "full_context/one.md",
            "full_context_json": "full_context/one.json",
            "after": [{
                "episode_id": "episode:1",
                "episode_scope": "single_fault",
                "continuation": False,
                "trace_group_id": "trace:1",
                "trace_phase_index": 1,
                "trace_phase_count": 1,
                "resolution_status": "pending",
                "w2_ready": False,
                "w2_block_reasons": ["missing_fault_signal"],
                "fault_focus": "相机拍摄失败",
            }],
        }],
    }


def test_w7_human_review_template_is_incomplete_until_human_fields_are_filled() -> None:
    template = build_template(_pack())
    report = validate_annotations(template, min_sessions=1)

    assert template["target_sessions"] == 1
    assert report["status"] == "INCOMPLETE"
    assert report["release_status"] == "INCOMPLETE"
    assert report["completed_sessions"] == 0
    assert report["remaining_sessions"] == 1


def test_w7_human_review_gate_summarizes_completed_annotations() -> None:
    template = build_template(_pack())
    session = template["sessions"][0]
    session.update({
        "reviewer": "human-reviewer",
        "reviewed_at": "2026-07-17T00:00:00Z",
        "session_verdict": "needs_fix",
        "session_issue_tags": ["trace_phase"],
        "session_notes": "trace phase should remain pending",
    })
    episode = session["episodes"][0]
    episode.update({field: True for field in template["boolean_fields"]})
    episode["trace_phase_correct"] = False
    episode["issue_tags"] = ["trace_phase"]
    episode["notes"] = "phase requires correction"

    report = validate_annotations(template, min_sessions=1)

    assert report["status"] == "PASS"
    assert report["completion_status"] == "PASS"
    assert report["quality_status"] == "FAIL"
    assert report["release_status"] == "FAIL"
    assert report["completed_sessions"] == 1
    assert report["verdict_counts"] == {"needs_fix": 1}
    assert report["boolean_failure_counts"] == {"trace_phase_correct": 1}
    assert report["field_quality"]["trace_phase_correct"]["accuracy"] == 0.0
    assert report["field_quality"]["fault_focus_correct"]["accuracy"] == 1.0


def test_w7_human_review_prioritizes_trace_calibration_and_preserves_existing_fields() -> None:
    pack = _pack()
    low = pack["cases"][0]
    high = {
        **low,
        "thread_id": "thread:high",
        "review_priority_score": 120,
        "review_priority_reasons": ["fixed173_trace_calibration"],
        "weak_trace_link_candidate_count": 2,
        "after": [{**low["after"][0], "episode_id": "episode:high"}],
    }
    pack["cases"] = [low, high]
    existing = {
        "required_min_sessions": 1,
        "sessions": [{
            "thread_id": "thread:high",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-07-22T00:00:00Z",
            "session_verdict": "pass",
            "episodes": [{
                "episode_id": "episode:high",
                **{field: True for field in BOOLEAN_FIELDS},
            }],
        }],
    }

    template = build_template(pack, existing)

    assert [row["thread_id"] for row in template["sessions"]] == ["thread:high", "thread:1"]
    first = template["sessions"][0]
    assert first["reviewer"] == "human-reviewer"
    assert first["review_priority_reasons"] == ["fixed173_trace_calibration"]
    assert first["weak_trace_link_candidate_count"] == 2
    assert first["episodes"][0]["trace_group_correct"] is True


def test_adjudicated_export_excludes_incomplete_rows_and_marks_structured_target_coverage() -> None:
    template = build_template(_pack())
    complete = template["sessions"][0]
    complete.update({
        "reviewer": "human-reviewer",
        "reviewed_at": "2026-07-22T00:00:00Z",
        "session_verdict": "needs_fix",
        "session_notes": "trace needs correction",
    })
    episode = complete["episodes"][0]
    episode.update({field: True for field in BOOLEAN_FIELDS})
    episode.update({
        "trace_group_correct": False,
        "corrected_trace_group_id": "trace:A",
        "issue_tags": ["trace_group"],
    })
    incomplete = {**complete, "thread_id": "thread:incomplete", "reviewer": ""}
    template["sessions"].append(incomplete)

    dataset = build_adjudicated_dataset(template)

    assert dataset["completed_sessions"] == 1
    assert dataset["completed_episodes"] == 1
    exported = dataset["sessions"][0]["episodes"][0]
    assert exported["corrections"]["corrected_trace_group_id"] == "trace:A"
    assert exported["structured_targets_complete"] is True
    assert len(dataset["source_annotation_sha256"]) == 64


def test_structured_correction_merge_targets_suffix_without_changing_human_judgment() -> None:
    template = build_template(_pack())
    template["sessions"][0]["episodes"][0]["trace_group_correct"] = False
    corrections = {
        "schema_version": "debug_agent_system.w7_structured_corrections.v1",
        "sessions": [{
            "thread_id": "thread:1",
            "episodes": [{
                "episode_suffix": ":1",
                "corrected_trace_group_id": "human-trace:A",
                "corrected_trace_phase_index": 1,
                "corrected_trace_phase_count": 3,
            }],
        }],
    }

    merged = merge_structured_corrections(template, corrections)
    episode = merged["sessions"][0]["episodes"][0]

    assert episode["trace_group_correct"] is False
    assert episode["corrected_trace_group_id"] == "human-trace:A"
    assert episode["corrected_trace_phase_count"] == 3
    assert merged["structured_correction_sources"][0]["applied_field_count"] == 3
