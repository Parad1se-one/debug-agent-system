from __future__ import annotations

import os

from debug_agent_system.agents.write.w2_extract.deepseek_client import (
    DeepSeekToolCallError,
    _decode_json_frames,
    configured_model,
    model_output_limit,
)
from debug_agent_system.eval.write_side.blind_011_015_deepseek_pro import (
    _detail_prompt_input,
    _validate_boundaries,
    boundary_tool_schema,
)


def test_configured_model_uses_general_v4_pro_setting_when_tool_override_is_absent():
    before_model = os.environ.get("DEEPSEEK_W2_MODEL")
    before_tool = os.environ.pop("DEEPSEEK_W2_TOOL_MODEL", None)
    try:
        os.environ["DEEPSEEK_W2_MODEL"] = "deepseek-v4-pro"
        assert configured_model() == "deepseek-v4-pro"
        assert model_output_limit(configured_model()) == 384_000
        assert model_output_limit("deepseek-chat") == 8_192
    finally:
        if before_model is None:
            os.environ.pop("DEEPSEEK_W2_MODEL", None)
        else:
            os.environ["DEEPSEEK_W2_MODEL"] = before_model
        if before_tool is not None:
            os.environ["DEEPSEEK_W2_TOOL_MODEL"] = before_tool


def test_boundary_schema_is_strict_and_boundary_validation_rejects_unknown_evidence():
    schema = boundary_tool_schema()["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])

    clusters, issues = _validate_boundaries(
        {
            "clusters": [
                {
                    "cluster_ref": "t1",
                    "evidence_ids": ["m1", "invented"],
                }
            ]
        },
        {"m1"},
    )
    assert clusters[0]["evidence_ids"] == ["m1"]
    assert issues == ["clusters[0]:unknown_evidence_id:invented"]


def test_detail_stage_receives_only_boundary_selected_source_evidence():
    prompt_input = {
        "source_episode_id": "goldcase-011",
        "current_episode_messages": [
            {"message_id": "m1", "text": "fault"},
            {"message_id": "m2", "text": "unrelated"},
        ],
        "promoted_case_evidence": [
            {"message_id": "jira:J1", "text": "jira"},
        ],
        "allowed_evidence_ids": ["m1", "m2", "jira:J1"],
    }
    detail = _detail_prompt_input(
        prompt_input,
        {"cluster_ref": "trace-a", "evidence_ids": ["m1", "jira:J1"]},
    )

    assert [item["message_id"] for item in detail["current_episode_messages"]] == ["m1"]
    assert [item["message_id"] for item in detail["promoted_case_evidence"]] == ["jira:J1"]
    assert detail["allowed_evidence_ids"] == ["jira:J1", "m1"]


def test_deepseek_json_frame_decoder_only_dedupes_identical_values():
    value, count = _decode_json_frames('{"ok":true}\n{"ok":true}', label="test")
    assert value == {"ok": True}
    assert count == 2
    try:
        _decode_json_frames('{"ok":true}{"ok":false}', label="test")
    except DeepSeekToolCallError as exc:
        assert "multiple_nonidentical_json_values:2" in str(exc)
    else:
        raise AssertionError("non-identical frames must fail closed")
