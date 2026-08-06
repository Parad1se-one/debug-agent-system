from __future__ import annotations

from debug_agent_system.eval.write_side.gold_prompt_preview import build_preview, disclosure_summary, response_template
from debug_agent_system.eval.write_side.gold_prompt_replay import validate_response_payload


def test_disclosure_summary_detects_retained_diagnostic_details_and_redacted_identity():
    summary = disclosure_summary({
        "current_episode_messages": [{
            "message_id": "m1",
            "sender": "participant_1",
            "text": "版本1.2.8，IP 192.168.1.2，HTTP status:500，@participant",
        }],
        "promoted_case_evidence": [],
    })
    assert summary["contains_ip_address"] is True
    assert summary["contains_software_version"] is True
    assert summary["contains_log_or_error_detail"] is True
    assert summary["contains_unredacted_personal_marker"] is False


def test_gold_prompt_preview_is_offline_loo_and_integrity_checked():
    preview = build_preview(gold_root="data/annotations/goldcases/gold-v1", kg_root="data/kg")

    assert preview["gold_set_id"] == "gold-v1"
    assert preview["gold_set_integrity_ok"] is True
    assert preview["network_io_performed"] is False
    assert preview["request_count"] == 10
    assert all(item["loo_audit"]["current_gold_case_excluded"] for item in preview["requests"])
    assert all(not item["disclosure"]["contains_unredacted_personal_marker"] for item in preview["requests"])
    assert all(item["payload_sha256"] for item in preview["requests"])


def test_gold_prompt_response_template_is_hash_bound_and_replay_validated():
    preview = build_preview(gold_root="data/annotations/goldcases/gold-v1", kg_root="data/kg")
    responses = response_template(preview)
    for item in responses["responses"]:
        item["tool_arguments"] = {"split_required": False, "split_reason": "test", "cases": [], "global_uncertainties": []}

    response_map, issues = validate_response_payload(preview, responses)
    assert issues == []
    assert len(response_map) == 10

    responses["responses"][0]["payload_sha256"] = "tampered"
    _, issues = validate_response_payload(preview, responses)
    assert any(issue.startswith("payload_sha256_mismatch:") for issue in issues)
