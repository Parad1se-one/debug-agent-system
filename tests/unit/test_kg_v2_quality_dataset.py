from copy import deepcopy

from debug_agent_system.eval.read_side.kg_v2_quality_dataset import (
    build_dataset,
    validate_dataset,
)


def test_quality_dataset_is_deterministic_and_valid():
    first = build_dataset()
    second = build_dataset()
    assert first == second
    report = validate_dataset(first)
    assert report["status"] == "passed", report["issues"]
    assert report["coverage"]["case_count"] == 60


def test_quality_dataset_has_all_layers_and_runtime_id_contracts():
    dataset = build_dataset()
    coverage = dataset["coverage"]
    assert set(coverage["difficulty_counts"]) == {"easy", "medium", "hard", "expert"}
    assert set(coverage["reasoning_mode_counts"]) == {
        "single_step",
        "multi_hop_linear",
        "multi_hop_branch",
        "multi_trace_disambiguation",
    }
    assert coverage["track_counts"] == {"gold_trace_reasoning": 10, "runtime_replay": 50}
    assert coverage["runtime_variant_count"] >= 20
    assert coverage["with_negative_control"] >= 10

    runtime_cases = [case for case in dataset["cases"] if case["evaluation_track"] == "runtime_replay"]
    for case in runtime_cases:
        expected = case["expected"]
        assert expected["family_id"].startswith("family:")
        assert expected["variant_id"].startswith("variant:")
        assert expected["plan_id"].startswith(("trace:", "policy:", "variant:"))
        assert expected["first_action_id"].startswith("action:")
        assert expected["evidence_ids"]
        assert "variant:" not in case["query"]


def test_gold_reasoning_cases_are_source_only_and_never_ingestable():
    dataset = build_dataset()
    gold_cases = [case for case in dataset["cases"] if case["evaluation_track"] == "gold_trace_reasoning"]
    assert [case["case_id"] for case in gold_cases] == [f"gold-reasoning-{number:03d}" for number in range(11, 21)]
    assert {case["split"] for case in gold_cases[:5]} == {"validation"}
    assert {case["split"] for case in gold_cases[5:]} == {"held_out_test"}
    for case in gold_cases:
        assert case["quality"]["graph_ingestion_allowed"] is False
        assert case["source_refs"][0]["kind"] == "gold_source_only_input"
        assert case["expected"]["truth_ref"] != case["source_input_ref"]
        assert "ground_truth" not in case["query"]
        assert case["expected"]["forbidden_inferences"]


def test_validator_rejects_stale_runtime_identity_and_gold_hash():
    dataset = build_dataset()
    broken = deepcopy(dataset)
    runtime_case = next(case for case in broken["cases"] if case["evaluation_track"] == "runtime_replay")
    runtime_case["expected"]["variant_id"] = "variant:does-not-exist"
    gold_case = next(case for case in broken["cases"] if case["evaluation_track"] == "gold_trace_reasoning")
    gold_case["source_refs"][0]["sha256"] = "0" * 64
    report = validate_dataset(broken)
    assert report["status"] == "failed"
    assert any(issue.endswith(":variant_id") for issue in report["issues"])
    assert any(issue.endswith(":source_input_hash") for issue in report["issues"])
