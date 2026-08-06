from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.eval.write_side.blind_011_015_prompt_preview import (
    build_preview,
    response_template,
)
from debug_agent_system.eval.write_side.blind_011_015_w1_baseline import (
    predict_source_only,
    score_prediction,
)
from debug_agent_system.eval.write_side.render_blind_ground_truth_review import _validate, render


REVIEW_V3_ROOT = Path("data/annotations/goldcases/review-v3")
ROOT = REVIEW_V3_ROOT


def test_all_review_v3_ground_truth_cases_bind_to_frozen_source_inputs():
    for truth_path in sorted((ROOT / "ground_truth").glob("goldcase-*.json")):
        input_payload = json.loads((ROOT / "inputs" / truth_path.name).read_text(encoding="utf-8"))
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        assert truth["review_status"] == "approved"
        assert _validate(input_payload, truth) == []


def test_blind_prompt_preview_is_source_only_anonymized_and_hash_bound():
    preview = build_preview(REVIEW_V3_ROOT / "inputs")

    assert preview["request_count"] == 5
    assert preview["source_only"] is True
    assert preview["ground_truth_accessed"] is False
    assert preview["network_io_performed"] is False
    assert len(preview["allowed_auxiliary_inputs"]) == 2
    assert all(item["payload_sha256"] for item in preview["requests"])
    assert all(not item["disclosure"]["contains_unredacted_personal_marker"] for item in preview["requests"])
    request_013 = next(item for item in preview["requests"] if item["request_id"] == "goldcase-013")
    promoted_013 = request_013["request"]["prompt_input"]["promoted_case_evidence"]
    assert any(item["message_id"] == "013-artifact-new-device-diagnostic-package" for item in promoted_013)
    assert any(item["message_id"] == "jira:SMTAOITS-1234" for item in promoted_013)
    serialized = json.dumps(preview, ensure_ascii=False)
    assert "review_candidate" not in serialized
    assert "critical_expectations" not in serialized
    assert "ground_truth/" not in serialized

    template = response_template(preview)
    assert len(template["responses"]) == 5
    assert all(item["tool_arguments"] is None for item in template["responses"])


def test_review_v3_binds_confirmed_fae_roles_without_aliasing_names():
    expected = {
        "goldcase-012": {"邓志勇", "廖明森", "孔令明"},
        "goldcase-013": {"方扬皓", "孔令明"},
        "goldcase-014": {"工程师申"},
        "goldcase-015": {"工程师未", "工程师子"},
    }
    for case_id, expected_names in expected.items():
        truth = _strict_json(REVIEW_V3_ROOT / "ground_truth" / f"{case_id}.json")
        confirmed = {
            item["reporter"]
            for item in truth.get("field_report_anchors") or []
            if item.get("role_status") == "confirmed_fae"
        }
        assert expected_names <= confirmed
        assert truth["review_status"] == "approved"
        assert truth["human_review"]["decision"] == "approved"


def test_w1_prediction_has_no_truth_input_and_scores_only_afterwards():
    input_payload = json.loads((ROOT / "inputs" / "goldcase-011.json").read_text(encoding="utf-8"))
    truth = json.loads((ROOT / "ground_truth" / "goldcase-011.json").read_text(encoding="utf-8"))
    prediction = predict_source_only(input_payload)

    assert prediction["source_only"] is True
    assert prediction["ground_truth_accessed"] is False
    assert prediction["prediction_sha256"]
    assert "expected_case_count" not in prediction

    score = score_prediction(prediction, truth)
    assert score["input_hash_match"] is True
    assert score["expected_case_count"] == 3


def test_goldcase_011_v3_is_fae_daily_report_anchored_and_evidence_bound():
    input_payload = json.loads(
        (REVIEW_V3_ROOT / "inputs" / "goldcase-011.json").read_text(encoding="utf-8")
    )
    truth = json.loads(
        (REVIEW_V3_ROOT / "ground_truth" / "goldcase-011.json").read_text(encoding="utf-8")
    )

    assert len(input_payload["messages"]) == 59
    assert len(input_payload["linked_jira_issues"]) == 6
    assert input_payload["fae_daily_summary_message_ids"] == [
        "om_x100b5c9e8b1ac8a4c36c537ffd1a4d1",
        "om_x100b5c8bc3a614a4c365e2815940c81",
    ]
    assert truth["analysis_window"] == {
        "start_inclusive": "2025-12-07 00:00",
        "end_inclusive": "2025-12-12 23:59:59",
        "timezone": "Asia/Shanghai",
    }
    assert truth["case_count"] == 3
    assert _validate(input_payload, truth) == []


def test_goldcase_011_v3_keeps_three_traces_and_occurrence_semantics():
    truth = json.loads(
        (REVIEW_V3_ROOT / "ground_truth" / "goldcase-011.json").read_text(encoding="utf-8")
    )
    cases = {case["case_ref"]: case for case in truth["cases"]}

    assert set(cases) == {"011-a", "011-b", "011-c"}
    assert [item["occurrence_ref"] for item in cases["011-a"]["occurrences"]] == [
        "011-a-e1",
        "011-a-e2",
    ]
    assert [item["occurrence_ref"] for item in cases["011-b"]["occurrences"]] == ["011-b-e1"]
    assert [item["occurrence_ref"] for item in cases["011-c"]["occurrences"]] == [
        "011-c-e1",
        "011-c-e2",
    ]
    assert all(
        item["occurrence_ref"].startswith(case_ref)
        for case_ref, case in cases.items()
        for item in case["hypothesis_timeline"]
    )
    assert {"jira:TEST-1234", "jira:TEST-1234"} <= set(cases["011-b"]["evidence_anchor_ids"])


def test_goldcase_011_v3_marks_only_confirmed_blue_screen_changes_actual():
    truth = json.loads(
        (REVIEW_V3_ROOT / "ground_truth" / "goldcase-011.json").read_text(encoding="utf-8")
    )
    blue = next(case for case in truth["cases"] if case["case_ref"] == "011-c")
    status_by_ref = {action["action_ref"]: action["execution_status"] for action in blue["actions"]}
    outcome_types = {
        action["outcome"]["outcome_type"]
        for case in truth["cases"]
        for action in case["actions"]
    }

    assert status_by_ref["011-c3"] == "actual"
    assert status_by_ref["011-c5"] == "actual"
    assert status_by_ref["011-c4"] == "recommended"
    assert status_by_ref["011-c6"] == "recommended"
    assert status_by_ref["011-c7"] == "recommended"
    assert "verified_fix" not in outcome_types


def _strict_json(path: Path):
    def reject_duplicate_keys(pairs):
        payload = {}
        for key, value in pairs:
            assert key not in payload, f"duplicate JSON key: {key}"
            payload[key] = value
        return payload

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def test_goldcase_012_v3_backtracks_first_week_and_binds_all_evidence():
    input_path = REVIEW_V3_ROOT / "inputs" / "goldcase-012.json"
    truth_path = REVIEW_V3_ROOT / "ground_truth" / "goldcase-012.json"
    input_payload = _strict_json(input_path)
    truth = _strict_json(truth_path)

    assert len(input_payload["messages"]) == 75
    assert len(input_payload["linked_jira_issues"]) == 7
    assert len(input_payload["external_artifacts"]) == 5
    assert input_payload["messages_sha256"] == "54df64b2f1adc134d6e35f192c0ba6aa497ff70f8333f8a46f808b1959e2e426"
    assert input_payload["input_evidence_sha256"] == "331a597c9630320889f6dc28f4f89a67f7b38605179bd803acfaf6728e6babb8"
    assert truth["analysis_window"]["start_inclusive"] == "2025-05-16 00:00"
    assert truth["field_report_anchors"][0]["date"] == "2025-05-21"
    assert _validate(input_payload, truth) == []


def test_goldcase_012_v3_splits_non_atomic_stability_rollup_into_four_traces():
    truth = _strict_json(REVIEW_V3_ROOT / "ground_truth" / "goldcase-012.json")
    cases = {case["case_ref"]: case for case in truth["cases"]}

    assert truth["case_count"] == 4
    assert set(cases) == {"012-a", "012-b", "012-c", "012-d"}
    assert {"jira:TEST-1234", "jira:TEST-1234"} <= set(cases["012-a"]["evidence_anchor_ids"])
    assert "jira:TEST-1234" in cases["012-c"]["evidence_anchor_ids"]
    assert "jira:TEST-1234" in cases["012-d"]["evidence_anchor_ids"]
    assert all(
        action["outcome"]["outcome_type"] != "verified_fix"
        for case in truth["cases"]
        for action in case["actions"]
    )


def test_goldcase_012_v3_uses_jira_descriptions_without_promoting_questions_to_causes():
    input_payload = _strict_json(REVIEW_V3_ROOT / "inputs" / "goldcase-012.json")
    truth = _strict_json(REVIEW_V3_ROOT / "ground_truth" / "goldcase-012.json")
    jira = {item["key"]: item for item in input_payload["linked_jira_issues"]}
    cases = {case["case_ref"]: case for case in truth["cases"]}

    assert "切换rgb/w图" in jira["TEST-1234"]["comments"][0]["body"]
    assert jira["TEST-1234"]["comments"][0]["body"] == "和TEST-6533是同类问题"
    assert "MES的报错问题比较多" in jira["TEST-1234"]["description"]
    assert "wait for exposure end event timed out" in jira["TEST-1234"]["description"]
    mes = next(
        item for item in cases["012-b"]["hypothesis_timeline"]
        if "MES" in item["summary"]
    )
    assert mes["state"] == "unverified"
    assert mes["causal_role"] == "concurrent_candidate"


def test_goldcase_012_v3_keeps_unavailable_artifact_content_out_of_annotation():
    input_path = REVIEW_V3_ROOT / "inputs" / "goldcase-012.json"
    truth_path = REVIEW_V3_ROOT / "ground_truth" / "goldcase-012.json"
    input_payload = _strict_json(input_path)
    truth = _strict_json(truth_path)

    assert all(not item["content_used_for_annotation"] for item in input_payload["external_artifacts"])
    jira_audit = next(
        item for item in truth["artifact_audit"]
        if item["artifact_ref"] == "012-artifact-jira-linked-materials"
    )
    assert "SMB日志目录" in jira_audit["finding"]
    assert "jira:TEST-1234" in jira_audit["source_evidence_ids"]
    excluded = " ".join(item["fragment"] for item in truth["excluded_fragments"])
    assert "气缸顶起不拍照" in excluded
    assert "轨道中间宽度" in excluded
    body = render(input_payload, truth, input_path=input_path, truth_path=truth_path)
    assert "## 外部附件与文档可用性" in body
    assert "## 附件倒查结论" in body
    assert "## 现场报告状态钩子" in body
    assert "TEST-1234" in body


def test_goldcase_013_v3_binds_two_devices_and_three_traces():
    input_path = REVIEW_V3_ROOT / "inputs" / "goldcase-013.json"
    truth_path = REVIEW_V3_ROOT / "ground_truth" / "goldcase-013.json"
    input_payload = _strict_json(input_path)
    truth = _strict_json(truth_path)
    cases = {case["case_ref"]: case for case in truth["cases"]}

    assert len(input_payload["messages"]) == 27
    assert len(input_payload["linked_jira_issues"]) == 2
    assert len(input_payload["external_artifacts"]) == 5
    assert truth["case_count"] == 3
    assert set(cases) == {"013-a", "013-b", "013-c"}
    assert len(truth["device_identity_map"]) == 2
    assert truth["device_identity_map"][0]["trace_refs"] == ["013-a", "013-b"]
    assert truth["device_identity_map"][1]["trace_refs"] == ["013-c"]
    assert _validate(input_payload, truth) == []


def test_goldcase_013_v3_preserves_execution_and_causality_limits():
    truth = _strict_json(REVIEW_V3_ROOT / "ground_truth" / "goldcase-013.json")
    cases = {case["case_ref"]: case for case in truth["cases"]}
    action_by_ref = {
        action["action_ref"]: action
        for case in truth["cases"]
        for action in case["actions"]
    }
    outcome_types = {
        action["outcome"]["outcome_type"]
        for case in truth["cases"]
        for action in case["actions"]
    }

    assert cases["013-a"]["family"]["label"] == "显示器无信号"
    assert action_by_ref["013-b3"]["execution_status"] == "actual"
    assert action_by_ref["013-b4"]["execution_status"] == "recommended"
    assert action_by_ref["013-b5"]["execution_status"] == "uncertain"
    antivirus = cases["013-c"]["hypothesis_timeline"][0]
    assert antivirus["state"] == "unverified"
    assert antivirus["causal_role"] == "preceding_event"
    assert "verified_fix" not in outcome_types

    mark_audit = next(
        item for item in truth["artifact_audit"]
        if item["artifact_ref"] == "013-artifact-new-device-diagnostic-package"
    )
    assert "57次" in mark_audit["finding"]
    assert "318次" in mark_audit["finding"]


def test_goldcase_014_v3_extends_the_flower_screen_trace_to_july():
    input_path = REVIEW_V3_ROOT / "inputs" / "goldcase-014.json"
    truth_path = REVIEW_V3_ROOT / "ground_truth" / "goldcase-014.json"
    input_payload = _strict_json(input_path)
    truth = _strict_json(truth_path)
    cases = {case["case_ref"]: case for case in truth["cases"]}

    assert len(input_payload["messages"]) == 195
    assert len(input_payload["linked_jira_issues"]) == 8
    assert len(input_payload["external_artifacts"]) == 6
    assert truth["analysis_window"]["start_inclusive"] == "2025-12-04 00:00"
    assert truth["analysis_window"]["end_inclusive"] == "2026-07-01 23:59:59"
    assert truth["analysis_window"]["retrospective_first_customer_report"] == "2025-11-04"
    assert truth["case_count"] == 2
    assert set(cases) == {"014-a", "014-b"}
    assert len(cases["014-b"]["occurrences"]) == 5
    assert cases["014-b"]["occurrences"][-1]["state"] == "latest_recurrence_unresolved"
    assert _validate(input_payload, truth) == []


def test_goldcase_014_v3_separates_hdmi_flicker_and_keeps_flower_screen_unresolved():
    truth = _strict_json(REVIEW_V3_ROOT / "ground_truth" / "goldcase-014.json")
    cases = {case["case_ref"]: case for case in truth["cases"]}
    flower = cases["014-b"]
    action_by_ref = {action["action_ref"]: action for action in flower["actions"]}
    outcome_types = {
        action["outcome"]["outcome_type"]
        for case in truth["cases"]
        for action in case["actions"]
    }

    assert cases["014-a"]["family"]["label"] == "显示器黑屏闪烁"
    assert flower["family"]["label"] == "软件界面花屏"
    assert action_by_ref["014-b7"]["outcome"]["outcome_type"] == "regression_observed"
    assert action_by_ref["014-b9"]["execution_status"] == "actual"
    assert action_by_ref["014-b9b"]["execution_status"] == "recommended"
    assert action_by_ref["014-b10"]["action_role"] == "collect"
    assert action_by_ref["014-b12"]["execution_status"] == "recommended"
    assert flower["hypothesis_timeline"][-1]["state"] == "unresolved"
    assert "verified_fix" not in outcome_types

    excluded = {item["evidence_id"] for item in truth["excluded_fragments"]}
    assert {"jira:TEST-1234", "jira:TEST-1234", "jira:SMTAOITS-1234"} <= excluded


def test_goldcase_015_v3_restores_opening_artifacts_local_time_and_jira_link():
    input_path = REVIEW_V3_ROOT / "inputs" / "goldcase-015.json"
    truth_path = REVIEW_V3_ROOT / "ground_truth" / "goldcase-015.json"
    input_payload = _strict_json(input_path)
    truth = _strict_json(truth_path)
    messages = {item["message_id"]: item for item in input_payload["messages"]}

    assert len(input_payload["messages"]) == 65
    assert len(input_payload["linked_jira_issues"]) == 2
    assert len(input_payload["external_artifacts"]) == 9
    assert input_payload["messages_sha256"] == "a725d858e00b900d28d2ac63fd77dd942d6ac3b9743310d5fd0fa7daf34aac6f"
    assert input_payload["input_evidence_sha256"] == "efc3553bc88539bee7793fd7ffe39a4a638734e346813465c75cc60dbc40224b"
    assert len(messages["om_x100b6e4e8a5b0480b2c4617f837b919"]["attachments"]) == 1
    assert len(messages["om_x100b6e4e9af7e944c255af8e67ae80e"]["attachments"]) == 1
    assert len(messages["om_x100b6e4e9afe60f4c3875169fe67fbd"]["attachments"]) == 1
    daily = messages["om_x100b6eb7ba060ca4b11fc13dcbbc4e0"]
    assert daily["create_time"] == "2026-05-28 21:12"
    assert daily["source_create_time_utc"] == "2026-05-28 13:12"
    assert "https://jira.example.com/browse/SMTAOITS-1234" in daily["text"]
    assert _validate(input_payload, truth) == []


def test_goldcase_015_v3_keeps_four_traces_and_attachment_causality_limits():
    input_payload = _strict_json(REVIEW_V3_ROOT / "inputs" / "goldcase-015.json")
    truth = _strict_json(REVIEW_V3_ROOT / "ground_truth" / "goldcase-015.json")
    cases = {case["case_ref"]: case for case in truth["cases"]}
    artifacts = {item["artifact_ref"]: item for item in input_payload["external_artifacts"]}
    outcome_types = {
        action["outcome"]["outcome_type"]
        for case in truth["cases"]
        for action in case["actions"]
    }

    assert truth["case_count"] == 4
    assert set(cases) == {"015-a", "015-b", "015-c", "015-d"}
    assert len(cases["015-a"]["occurrences"]) == 3
    assert len(cases["015-d"]["occurrences"]) == 3
    assert artifacts["015-artifact-may28-diagnostic"]["retrieval_status"] == "referenced_but_payload_missing"
    assert artifacts["015-artifact-may28-diagnostic"]["content_used_for_annotation"] is False
    assert artifacts["015-artifact-blue-screen-dmp-1"]["bugcheck_code"] == "0x4e"
    assert artifacts["015-artifact-blue-screen-dmp-1"]["bugcheck_parameters"][0] == "0x99"
    assert artifacts["015-artifact-blue-screen-dmp-2"]["bugcheck_code"] == "0x1a"
    assert artifacts["015-artifact-blue-screen-dmp-2"]["bugcheck_parameters"][0] == "0x41792"
    assert cases["015-b"]["hypothesis_timeline"][0]["causal_role"] == "proximate_cause"
    assert cases["015-c"]["hypothesis_timeline"][-1]["state"] == "candidate"
    assert "verified_fix" not in outcome_types
