from copy import deepcopy
import json
from pathlib import Path

from debug_agent_system.eval.read_side.unified_benchmark import (
    FIELD_QUERY_SEEDS,
    build_dataset,
    render_query_answer_markdown,
    score_predictions,
    validate_dataset,
)
from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel


def test_unified_benchmark_is_deterministic_and_valid():
    first = build_dataset()
    second = build_dataset()
    assert first == second

    report = validate_dataset(first)
    assert report["status"] == "passed", report["issues"]
    assert report["coverage"]["case_count"] == 238


def test_unified_benchmark_covers_full_kg_and_keeps_origins_separate():
    dataset = build_dataset()
    coverage = dataset["coverage"]
    model = KGV2ReadModel("data/kg_v2")

    assert coverage["document_count"] == sum(
        bool(item.get("approved"))
        for item in model.by_type["KnowledgeDocument"].values()
    )
    assert coverage["family_count"] == len(model.by_type["FaultFamily"])
    assert coverage["variant_count"] == len(model.by_type["FaultVariant"])
    assert coverage["runtime_variant_case_count"] == 98
    assert coverage["catalog_only_variant_case_count"] == 19
    assert coverage["independent_gold_case_count"] == 10
    assert coverage["field_query_count"] == len(FIELD_QUERY_SEEDS)
    assert set(coverage["expectation_origin_counts"]) == {
        "curated_query_kg_evidence",
        "human_frozen_gold",
        "kg_snapshot_conformance",
    }


def test_shared_queries_rebuild_answers_from_kg_not_historical_answers():
    dataset = build_dataset()
    baseline = json.loads(
        Path(
            "data/eval/scenarios/read_side_shared_query_baseline_v1.json"
        ).read_text(encoding="utf-8")
    )
    historical_answers = {
        str(item["query"]): str(item.get("original_answer_text") or "")
        for item in baseline["records"]
    }
    shared = [
        case
        for case in dataset["cases"]
        if case["source_type"] == "legacy_or_field_query"
    ]

    assert len(shared) == 56
    assert [case["query"] for case in shared[-9:]] == list(FIELD_QUERY_SEEDS)
    for case in shared:
        assert case["quality"]["legacy_answer_ignored"] is True
        assert case["evidence_gold"]["required_object_ids"]
        assert case["answer_gold"]["required_claims"]
        if case["query"] in historical_answers:
            assert (
                case["answer_gold"]["reference_answer"]
                != historical_answers[case["query"]]
            )

    threshold_case = next(
        case for case in shared if case["query"].startswith("坏板阈值")
    )
    assert threshold_case["quality"]["evidence_gap"] is True
    assert len(threshold_case["evidence_gold"]["document_ids"]) == 1
    assert "磁盘分区" not in threshold_case["answer_gold"]["reference_answer"]


def test_catalog_only_variants_fail_closed_and_dangerous_actions_are_gated():
    dataset = build_dataset()
    catalog_cases = [
        case
        for case in dataset["cases"]
        if case["source_type"] == "catalog_only_kg_variant"
    ]
    assert len(catalog_cases) == 19
    for case in catalog_cases:
        execution = case["execution_gold"]
        assert execution["execution_materialize_allowed"] is False
        assert execution["acceptable_action_ids"] == []
        assert execution["allowed_initial_statuses"] == ["ask_info"]
        assert "step" in execution["forbidden_terminal_statuses"]

    dangerous_cases = [
        case
        for case in dataset["cases"]
        if case["execution_gold"].get("dangerous_action_ids")
    ]
    assert dangerous_cases
    for case in dangerous_cases:
        execution = case["execution_gold"]
        assert set(execution["dangerous_action_ids"]) <= set(
            execution["forbidden_action_ids_before_confirmation"]
        )


def test_human_gold_is_source_only_frozen_and_hash_bound():
    dataset = build_dataset()
    gold_cases = [
        case
        for case in dataset["cases"]
        if case["expectation_origin"] == "human_frozen_gold"
    ]

    assert [case["case_id"] for case in gold_cases] == [
        f"gold-source-{number:03d}" for number in range(11, 21)
    ]
    for case in gold_cases:
        assert case["quality"]["independent_semantic_gold"] is True
        assert case["quality"]["graph_ingestion_allowed"] is False
        assert case["source_refs"][0]["kind"] == "gold_source_only_input"
        assert case["source_refs"][1]["kind"] == "gold_truth_label"
        assert case["source_refs"][1]["runtime_visible"] is False


def test_query_answer_markdown_lists_every_case_and_origin_boundary():
    dataset = build_dataset()
    markdown = render_query_answer_markdown(dataset)

    assert markdown.count("\n### ") == dataset["coverage"]["case_count"]
    assert "## 批准文档 Query" in markdown
    assert "## 分享测试集与现场 Query" in markdown
    assert "## 冻结 source-only Gold Query" in markdown
    assert "`human_frozen_gold` 是独立冻结语义 Gold" in markdown
    for query in FIELD_QUERY_SEEDS:
        assert query in markdown
    for case in dataset["cases"]:
        assert f"### {case['case_id']}" in markdown
        assert case["query"] in markdown
        for line in case["answer_gold"]["reference_answer"].splitlines():
            assert f"    {line}" in markdown


def test_validator_rejects_stale_id_hash_and_catalog_plan():
    dataset = build_dataset()
    broken = deepcopy(dataset)
    runtime_case = next(
        case
        for case in broken["cases"]
        if case["source_type"] == "active_kg_variant"
    )
    runtime_case["diagnosis_gold"]["variant_id"] = "variant:missing"
    gold_case = next(
        case
        for case in broken["cases"]
        if case["expectation_origin"] == "human_frozen_gold"
    )
    gold_case["source_refs"][0]["sha256"] = "0" * 64
    catalog_case = next(
        case
        for case in broken["cases"]
        if case["source_type"] == "catalog_only_kg_variant"
    )
    catalog_case["execution_gold"]["plan_id"] = "plan:invented"

    report = validate_dataset(broken)
    assert report["status"] == "failed"
    assert any(issue.endswith(":variant_id") for issue in report["issues"])
    assert any(issue.endswith(":source_ref_hash") for issue in report["issues"])
    assert any(
        issue.endswith(":catalog_variant_has_executable_plan")
        for issue in report["issues"]
    )


def test_scorer_applies_hard_gate_to_unsafe_action_and_false_closure():
    dataset = build_dataset()
    assert score_predictions(dataset, {"predictions": []})[
        "hard_safety_gate_passed"
    ] is False
    dangerous = next(
        case
        for case in dataset["cases"]
        if case["execution_gold"].get("dangerous_action_ids")
    )
    action_id = dangerous["execution_gold"]["dangerous_action_ids"][0]
    catalog = next(
        case
        for case in dataset["cases"]
        if case["source_type"] == "catalog_only_kg_variant"
    )
    predictions = {
        "predictions": [
            {
                "case_id": dangerous["case_id"],
                "executed_action_id": action_id,
                "human_confirmed": False,
                "status": "resolved",
            },
            {
                "case_id": catalog["case_id"],
                "status": "step",
            },
        ]
    }

    score = score_predictions(dataset, predictions)
    assert score["hard_safety_gate_passed"] is False
    assert score["summary"]["T4_unsafe_action_rate"] > 0
    assert score["summary"]["T4_false_resolved_rate"] > 0
    assert score["summary"]["T4_forbidden_status_rate"] > 0
