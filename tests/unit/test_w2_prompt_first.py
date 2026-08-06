from __future__ import annotations

import json
import os

import debug_agent_system.agents.write.w2_extract as w2_module
from debug_agent_system.agents.write import KnowledgeExtractionAgent
from debug_agent_system.agents.write.w4_quality_gate import QualityGateAgent
from debug_agent_system.agents.write.w2_extract import _tool_schema_strict_issues
from debug_agent_system.agents.write.w2_extract.case_understanding_prompt import (
    build_prompt_input,
    normalize_card,
    tool_schema,
)
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2 import (
    build_candidate_draft_v2_from_case_understanding,
    build_v2_bundle_from_candidate_draft,
    validate_graph,
)


def _episode() -> dict:
    return {
        "episode_id": "prompt-first-episode",
        "thread_id": "prompt-first-thread",
        "completeness": "complete",
        "fault_description_messages": [
            {"message_id": "m1", "text": "设备出现故障，当前无法生产。"},
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "text": "建议检查接口。"},
            {"message_id": "m3", "text": "现场已重新连接接口。"},
        ],
        "resolution_messages": [
            {"message_id": "m4", "text": "重新连接后正常启动，目前已恢复生产。"},
        ],
        "case_evidence_messages": [],
        "case_context_messages": [
            {"message_id": "outside", "text": "其他案例更换设备后解决。"},
        ],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3", "m4"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "设备出现故障，当前无法生产。",
            "debug_actions": ["建议检查接口", "重新连接接口"],
            "conclusion": "重新连接后正常启动，目前已恢复生产。",
            "review_context": {
                "reviewed_case_examples": [
                    {"case_id": "same", "exact_source_match": True, "gold_structure": {"secret": "must not leak"}},
                    {"case_id": "other", "exact_source_match": False, "gold_structure": {"family": {"label": "示例家族"}}},
                ],
            },
        },
    }


def _semantics() -> dict:
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), deepseek_enabled=False)
    return extractor.extract_semantics(_episode())


def _raw_card(*, recommended_verified: bool = False, outcome_evidence: str = "m4") -> dict:
    status = "recommended" if recommended_verified else "actual"
    action_evidence = "m2" if recommended_verified else "m3"
    return {
        "split_required": False,
        "split_reason": "一个连续故障链",
        "cases": [{
            "case_ref": "case_1",
            "candidate_scope": "fault_execution",
            "family_hypothesis": {
                "label": "设备连接异常",
                "summary": "设备连接异常导致无法生产",
                "category": "硬件与运控",
                "subsystem": "设备连接链路",
                "scenario": "生产期间",
                "why_family_not_variant": "稳定故障现象类别",
                "confidence": 0.9,
            },
            "variant_hypothesis": {
                "label": "接口连接异常导致无法生产",
                "summary": "重新连接接口后恢复生产",
                "distinguishing_conditions": ["生产期间", "接口连接"],
                "confidence": 0.9,
            },
            "symptom_summary": "设备出现故障且无法生产",
            "evidence_anchor_ids": ["m1", "m3", "m4"],
            "actions": [{
                "action_ref": "act_1",
                "label": "重新连接接口",
                "summary": "重新连接设备接口",
                "action_role": "change",
                "execution_status": status,
                "atomicity_ok": True,
                "source_evidence_ids": [action_evidence],
                "high_cost": False,
                "destructive": False,
            }],
            "outcomes": [{
                "action_ref": "act_1",
                "outcome_type": "verified_fix",
                "summary": "恢复生产",
                "why_not_other_types": "恢复且有生产验证",
                "source_evidence_ids": [outcome_evidence],
                "high_cost": False,
                "destructive": False,
                "root_cause_summary": "",
            }],
            "required_info": [],
            "hypothesis_timeline": [],
            "uncertainties": [],
        }],
        "global_uncertainties": [],
    }


def test_prompt_first_tool_schema_is_deepseek_strict():
    assert _tool_schema_strict_issues(tool_schema()) == []


def test_prompt_input_excludes_navigation_context_and_exact_source_gold():
    prompt_input = build_prompt_input(_semantics())

    assert {item["message_id"] for item in prompt_input["current_episode_messages"]} == {"m1", "m2", "m3", "m4"}
    assert "outside" not in str(prompt_input)
    assert [item["example_ref"] for item in prompt_input["alignment_examples"]] == ["alignment_example"]
    assert "must not leak" not in str(prompt_input)


def test_prompt_card_rejects_unknown_evidence_and_action_without_outcome():
    raw = _raw_card()
    raw["cases"][0]["actions"][0]["source_evidence_ids"] = ["outside"]
    raw["cases"][0]["outcomes"] = []

    card, issues, _ = normalize_card(raw, _semantics())

    assert card["schema_valid"] is False
    assert any("unknown_evidence_id:outside" in issue for issue in issues)
    assert any("action_without_outcome:act_1" in issue for issue in issues)


def test_prompt_card_downgrades_recommended_verified_fix():
    card, issues, corrections = normalize_card(_raw_card(recommended_verified=True), _semantics())

    assert issues == []
    assert card["cases"][0]["outcomes"][0]["outcome_type"] == "pending_validation"
    assert any("recommended" in correction for correction in corrections)


def test_prompt_card_keeps_verified_fix_only_with_recovery_and_durable_signal():
    card, issues, corrections = normalize_card(_raw_card(), _semantics())

    assert issues == []
    assert corrections == []
    assert card["cases"][0]["outcomes"][0]["outcome_type"] == "verified_fix"


def test_outcome_origin_is_propagated_from_card_through_draft_to_bundle():
    card, issues, _ = normalize_card(_raw_card(), _semantics())
    assert issues == []

    draft = build_candidate_draft_v2_from_case_understanding(card)
    bundle = build_v2_bundle_from_candidate_draft(draft)

    assert card["cases"][0]["actions"][0]["evidence_scope"] == "current_episode_direct"
    assert draft["split_cases"][0]["actions"][0]["evidence_scope"] == "current_episode_direct"
    assert bundle["objects"]["DiagnosticAction"][0]["evidence_scope"] == "current_episode_direct"
    assert draft["split_cases"][0]["outcomes"][0]["outcome_origin"] == "source_extracted"
    assert bundle["objects"]["ActionOutcome"][0]["outcome_origin"] == "source_extracted"

    corrected, issues, _ = normalize_card(_raw_card(recommended_verified=True), _semantics())
    assert issues == []
    corrected_bundle = build_v2_bundle_from_candidate_draft(
        build_candidate_draft_v2_from_case_understanding(corrected)
    )
    assert corrected_bundle["objects"]["ActionOutcome"][0]["outcome_origin"] == "rule_inferred"


def test_candidate_draft_maps_arbitrary_valid_action_refs_to_outcomes():
    raw = _raw_card()
    raw["cases"][0]["actions"][0]["action_ref"] = "action_1"
    raw["cases"][0]["outcomes"][0]["action_ref"] = "action_1"
    card, issues, _ = normalize_card(raw, _semantics())
    assert issues == []

    draft = build_candidate_draft_v2_from_case_understanding(card)
    bundle = build_v2_bundle_from_candidate_draft(draft)

    assert draft["split_cases"][0]["outcomes"][0][
        "action_label"
    ] == "重新连接接口"
    assert len(bundle["objects"]["ActionOutcome"]) == 1
    assert bundle["schema_valid"] is True


def test_prompt_card_accepts_plain_recovery_with_explicit_24h_no_recurrence():
    semantics = _semantics()
    semantics["episode"]["resolution_messages"][0]["text"] = (
        "重新插紧后恢复，连续生产24小时未再出现拍摄失败。"
    )
    card, issues, corrections = normalize_card(_raw_card(), semantics)

    assert issues == []
    assert corrections == []
    assert card["cases"][0]["outcomes"][0]["outcome_type"] == "verified_fix"


def test_prompt_card_does_not_promote_plain_recovery_when_temporary_risk_is_explicit():
    semantics = _semantics()
    semantics["episode"]["resolution_messages"][0]["text"] = (
        "重新插紧后临时恢复，短期可用但仍有复发风险。"
    )
    card, issues, corrections = normalize_card(_raw_card(), semantics)

    assert issues == []
    assert card["cases"][0]["outcomes"][0]["outcome_type"] == "partial_temporary"
    assert corrections

    semantics = _semantics()
    semantics["episode"]["resolution_messages"][0]["text"] = "重新连接后暂时恢复，仍需观察。"
    card, issues, corrections = normalize_card(_raw_card(), semantics)
    assert issues == []
    assert card["cases"][0]["outcomes"][0]["outcome_type"] == "partial_temporary"
    assert corrections


def test_native_v2_uses_prompt_card_as_authoritative_when_deepseek_enabled():
    original = w2_module._call_deepseek_case_understanding_with_hard_timeout
    old_key = os.environ.get("DEEPSEEK_API_KEY")

    def fake_call(prompt_input, *, api_key, repair_issues):
        assert api_key == "test-key"
        assert repair_issues == []
        assert prompt_input["source_episode_id"] == "prompt-first-episode"
        return _raw_card()

    try:
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        w2_module._call_deepseek_case_understanding_with_hard_timeout = fake_call
        result = KnowledgeExtractionAgent(
            JsonKGStore("data/kg"),
            deepseek_enabled=True,
            w2_mode="native_v2",
        ).extract(_episode())
    finally:
        w2_module._call_deepseek_case_understanding_with_hard_timeout = original
        if old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_key

    assert result["case_understanding_extraction"]["case_understanding_source"] == "deepseek_prompt_a"
    assert result["case_understanding_extraction"]["deterministic_compat_fallback"] is False
    assert result["case_understanding_card"]["extraction_source"] == "deepseek_prompt_a"
    assert result["case_understanding_card"]["cases"][0]["family_hypothesis"]["label"] == "设备连接异常"


def test_prompt_first_repairs_malformed_tool_arguments_once():
    original = w2_module._call_deepseek_case_understanding_with_hard_timeout
    calls = []

    def fake_call(prompt_input, *, api_key, repair_issues):
        calls.append(list(repair_issues))
        if len(calls) == 1:
            raise json.JSONDecodeError("Extra data", "{}{}", 2)
        assert repair_issues and repair_issues[0].startswith(
            "tool_arguments_parse_error:JSONDecodeError:"
        )
        return _raw_card()

    try:
        w2_module._call_deepseek_case_understanding_with_hard_timeout = fake_call
        card, attempts, corrections = w2_module._extract_prompt_case_understanding_with_repair(
            _semantics(), api_key="test-key"
        )
    finally:
        w2_module._call_deepseek_case_understanding_with_hard_timeout = original

    assert card["schema_valid"] is True
    assert attempts == 2
    assert corrections == []
    assert calls[0] == []


def test_strict_prompt_first_mode_marks_unavailable_model_invalid():
    result = KnowledgeExtractionAgent(
        JsonKGStore("data/kg"),
        deepseek_enabled=False,
        w2_mode="prompt_first",
    ).extract(_episode())

    assert result["case_understanding_extraction"]["deterministic_compat_fallback"] is True
    assert result["schema_valid"] is False
    assert result["case_understanding_card_schema_valid"] is False
    assert result["production_schema_valid"] is False
    assert "prompt_first_extraction_unavailable" in result["case_understanding_card_schema_issues"]
    gate = QualityGateAgent().score(result)
    assert gate["passed"] is False
    assert "schema_invalid" in gate["issues"]


def test_v2_graph_gate_requires_case_action_evidence_outcome_and_actual_fix():
    card, issues, _ = normalize_card(_raw_card(), _semantics())
    assert issues == []
    bundle = build_v2_bundle_from_candidate_draft(
        build_candidate_draft_v2_from_case_understanding(card)
    )
    assert bundle["schema_valid"] is True, bundle["schema_issues"]

    objects = bundle["objects"]
    action = objects["DiagnosticAction"][0]
    outcome = objects["ActionOutcome"][0]
    action["evidence_ids"] = []
    action["execution_status"] = "recommended"
    objects["ActionOutcome"] = []
    issues = validate_graph(objects, bundle["relations"])

    assert f"case_action_missing_evidence:{action['action_id']}" in issues
    assert f"case_action_missing_outcome:{action['action_id']}" in issues

    objects["ActionOutcome"] = [outcome]
    issues = validate_graph(objects, bundle["relations"])
    assert any(issue.startswith("verified_fix_for_non_actual_action:") for issue in issues)
