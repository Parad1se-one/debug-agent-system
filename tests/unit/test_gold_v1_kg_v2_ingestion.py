from __future__ import annotations

from debug_agent_system.eval.write_side.ingest_gold_v1_to_kg_v2 import build_bundle, run
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


def test_gold_v1_bundle_maps_all_ten_cases_without_semantic_inference():
    bundle = build_bundle("data/annotations/goldcases/gold-v1", "data/kg_v2")

    assert [item["case_id"] for item in bundle["report"]["cases"]] == [
        f"goldcase-{index:03d}" for index in range(1, 11)
    ]
    assert len(bundle["objects"]["SourceCase"]) == 10
    assert len(bundle["objects"]["FaultVariant"]) == 10
    assert len(bundle["objects"]["DiagnosticTrace"]) == 10
    assert len(bundle["objects"]["DiagnosticAction"]) == 90
    assert len(bundle["objects"]["ActionOutcome"]) == 90
    assert len(bundle["objects"]["TraceStep"]) == 90
    assert len(bundle["objects"]["ExecutionObservation"]) == 77
    assert len(bundle["objects"]["BranchRule"]) == 90
    assert all(item["approved"] is True for item in bundle["objects"]["SourceCase"])
    assert all(item.get("gold_case_id") != "goldcase-011" for rows in bundle["objects"].values() for item in rows)


def test_gold_v1_ingestion_dry_run_repairs_superseded_case_subgraphs_and_validates():
    report = run(
        gold_root="data/annotations/goldcases/gold-v1",
        kg_root="data/kg_v2",
        audit_out="data/annotations/goldcases/gold-v1/kg_v2_ingestion_manifest.json",
        apply=False,
        authorization="",
    )

    assert report["write_result"]["status"] == "dry_run_valid"
    assert all(
        not case_id.startswith("case:goldcase-")
        for case_id in report["superseded_subgraph_cleanup"]["source_case_ids"]
    )
    assert report["after"]["mode"] == "projected"


def test_active_policy_relations_match_recomputed_policy_objects():
    store = JsonKGV2Store("data/kg_v2")
    policy_ids = {
        str(item.get("policy_id") or "")
        for item in store.objects_by_type["DecisionPolicy"]
    }
    relation_policy_ids = {
        str(item.get("from") or "")
        for item in store.relations
        if item.get("relation") == "for_family"
    }

    assert relation_policy_ids == policy_ids
