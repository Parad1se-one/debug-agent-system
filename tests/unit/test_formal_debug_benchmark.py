from copy import deepcopy

import pytest

from debug_agent_system.eval.read_side.formal_debug_benchmark import (
    LAYER_QUOTAS,
    VALIDATION_QUOTAS,
    build_broad_pools,
    build_dataset,
    candidate_feature_selftest_shards,
    prediction_template,
    record_current_approval,
    score_predictions,
    validate_broad_pools,
    validate_dataset,
)
from debug_agent_system.eval.read_side.formal_debug_runner import execute_dataset


class _FakeRunner:
    model = "gpt-5.6-luna"
    runtime_metadata = {"engine": "fake"}

    def __init__(self):
        self.prompts = []

    def run(self, *, prompt, workspace, output_schema, timeout_seconds):
        self.prompts.append(prompt)
        return ({
            "answer": "证据不足，需要补充信息。",
            "route_type": "",
            "route_ids": [],
            "evidence_ids": [],
            "family_id": "",
            "variant_id": "",
            "first_action_id": "",
            "followup_ids": [],
            "status": "ask_info",
            "executed_action_ids": [],
            "trace_count": 0,
        }, {"files_read": [], "usage": {}})


def test_formal_core_has_exact_layers_and_split_without_false_gold():
    dataset = build_dataset(approval_path=None)
    report = validate_dataset(dataset)

    assert report["status"] == "passed", report["issues"]
    assert len(dataset["cases"]) == 100
    assert dataset["coverage"]["layer_counts"] == LAYER_QUOTAS
    assert dataset["coverage"]["split_counts"] == {
        "held_out_test": 40,
        "validation": 60,
    }
    assert dataset["coverage"]["independent_frozen_gold_count"] == 10
    assert report["pending_human_freeze_count"] == 90
    assert len({case["query"] for case in dataset["cases"]}) == 100
    route_types = {
        case["expected_route"]["route_type"]
        for case in dataset["cases"]
        if case["capability_layer"] == "routing_domain_boundary"
    }
    assert route_types == {
        "knowledge_document_section", "sag_v2_native", "out_of_domain",
    }
    group_splits = {}
    for case in dataset["cases"]:
        assert case["evidence_gold"]["must_recall_ids"]
        if case["human_review"]["status"] == "frozen":
            assert case["human_review"]["gold_origin"] == (
                "human_frozen_source_only_gold"
            )
        if case["split"] == "held_out_test":
            assert case["optimization_eligible"] is False
        group_splits.setdefault(case["leakage_control"]["group_id"], set()).add(
            case["split"]
        )
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_release_check_fails_closed_until_all_cases_are_human_frozen():
    report = validate_dataset(
        build_dataset(approval_path=None), require_release_ready=True
    )

    assert report["status"] == "failed"
    assert "human_freeze_pending:90" in report["issues"]
    assert report["release_ready"] is False


def test_validator_rejects_thin_evidence_and_fake_frozen_kg_gold():
    dataset = deepcopy(build_dataset(approval_path=None))
    case = dataset["cases"][0]
    case["evidence_gold"]["must_recall_ids"] = []
    case["human_review"]["status"] = "frozen"

    report = validate_dataset(dataset)

    assert any(issue.endswith(":thin_evidence") for issue in report["issues"])
    assert any(
        issue.endswith(":nonhuman_gold_marked_frozen")
        for issue in report["issues"]
    )


def test_broad_pools_keep_three_reports_separate():
    manifest = build_broad_pools()
    report = validate_broad_pools(manifest)

    # The manifest is intentionally fail-closed against the current KG: two
    # historical snapshots have drifted, while the FAE candidate pool remains
    # structurally valid.  A formal release must surface, not hide, that drift.
    assert report["status"] == "passed"
    assert report["issues"] == []
    assert report["runtime_compatibility_status"] == "drift_detected"
    assert set(report["runtime_compatibility_issues"]) == {
        "kg_runtime_contract:current_runtime_compatibility",
        "document_qa:current_runtime_compatibility",
    }
    assert report["pool_reports"]["real_fae_candidates"]["status"] == "passed"
    assert [pool["case_count"] for pool in manifest["pools"]] == [238, 205, 77]
    assert report["aggregation_policy"] == (
        "separate_reports_no_combined_accuracy"
    )


def test_prediction_template_and_scorer_default_to_validation_only():
    dataset = build_dataset()
    template = prediction_template(dataset)
    score = score_predictions(dataset, template, split="validation")

    assert len(score["case_scores"]) == 60
    assert "total_accuracy" not in score
    assert score["aggregation_policy"] == (
        "layer_reports_only_no_single_total_accuracy"
    )
    with pytest.raises(ValueError, match="explicit_allow"):
        score_predictions(dataset, template, split="held_out_test")


def test_batch_executor_hides_gold_and_resumes(tmp_path):
    dataset = build_dataset(approval_path=None)
    dataset["cases"] = dataset["cases"][:2]
    runner = _FakeRunner()

    first = execute_dataset(
        dataset,
        runner=runner,
        workspace=tmp_path,
        run_root=tmp_path / "run",
        split="all",
    )
    second = execute_dataset(
        dataset,
        runner=runner,
        workspace=tmp_path,
        run_root=tmp_path / "run",
        split="all",
    )

    assert len(first["predictions"]) == 2
    assert len(second["predictions"]) == 2
    assert len(runner.prompts) == 2
    assert all("evidence_gold" not in prompt for prompt in runner.prompts)
    assert all("conclusion_gold" not in prompt for prompt in runner.prompts)


def test_explicit_workspace_owner_approval_freezes_without_fake_independence(
    tmp_path,
):
    approval_path = tmp_path / "approval.json"
    record_current_approval(approval_path=approval_path)
    dataset = build_dataset(approval_path=approval_path)
    report = validate_dataset(dataset, require_release_ready=True)

    assert report["status"] == "passed", report["issues"]
    assert report["release_ready"] is True
    assert dataset["release_status"] == "released"
    assert dataset["coverage"]["frozen_case_count"] == 100
    assert dataset["coverage"]["independent_frozen_gold_count"] == 47
    assert dataset["coverage"]["human_approved_kg_conformance_count"] == 53


def test_feature_selftest_projection_matches_operation_agent_shape():
    shards, manifest = candidate_feature_selftest_shards()
    rows = [row for shard in shards for row in shard]
    expected_keys = {
        "id", "query", "product", "module", "query_type",
        "source_document", "source_document_id", "origin",
        "responsibility_scope", "source_readiness",
        "source_text_char_count", "source_media_marker_count",
    }

    assert len(rows) == 192
    assert [len(shard) for shard in shards] == [64, 64, 64]
    assert all(set(row) == expected_keys for row in rows)
    assert all(len({row["id"] for row in shard}) == 64 for shard in shards)
    assert len({row["query"] for row in rows}) == 192
    assert all(row["source_document_id"] for row in rows)
    assert all("dbg-core-trace" not in row["origin"] for row in rows)
    assert all(len(shard) <= 64 for shard in shards)
    assert manifest["source_counts_by_shard"] == [
        {"kg_runtime_contract": 64},
        {"real_fae_candidates": 64},
        {"document_qa": 64},
    ]
    assert {
        row["origin"].split(":")[1] for row in shards[0]
    } == {"kg_runtime_contract"}
    assert {
        row["origin"].split(":")[1] for row in shards[1]
    } == {"real_fae_candidates"}
    assert {
        row["origin"].split(":")[1] for row in shards[2]
    } == {"document_qa"}
    assert all(
        label not in row["query"]
        for row in shards[1]
        for label in (
            "【真实 FAE 现场报告】", "真实 FAE 现场报告",
            "现场原文：", "任务：",
        )
    )
    assert all(
        label not in row["query"]
        for row in shards[0]
        for label in (
            "现场反馈：", "请判断", "所属故障", "具体变体",
            "需要补充的信息", "首个排查动作",
        )
    )
