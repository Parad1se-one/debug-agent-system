"""Verify that frozen Goldcase 001--010 reached the KG v2 execution view."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write_v2.ingest import _candidate
from debug_agent_system.knowledge_v2.contracts import EXECUTION_CHECK_ROLES, make_id
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


EXPECTED_CASE_IDS = [f"goldcase-{index:03d}" for index in range(1, 11)]
NEGATIVE_OUTCOME_TYPES = {"ineffective", "recurred", "context_not_root_cause", "pending_validation"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _materialized_rows(root: Path, kind: str) -> list[dict[str, Any]]:
    path = root / "materialized_execution" / "instances" / kind / f"{kind}.json"
    payload = _read_json(path)
    return [item for item in payload if isinstance(item, dict)]


def verify(
    *,
    kg_root: str | Path,
    pipeline_report: str | Path,
    review_queue: str | Path,
) -> dict[str, Any]:
    root = Path(kg_root)
    report = _read_json(Path(pipeline_report))
    queue = _read_json(Path(review_queue))
    store = JsonKGV2Store(root)

    issues = validate_graph(store.objects_by_type, store.relations, schema_root=root / "schema")
    by_review_id = {
        str(item.get("review_id") or ""): item
        for item in queue
        if isinstance(item, dict) and str(item.get("review_id") or "")
    }

    errors = _materialized_rows(root, "errors")
    checks = _materialized_rows(root, "checks")
    traces = _materialized_rows(root, "traces")
    outcomes = _materialized_rows(root, "outcomes")
    materialized_trace_steps = _materialized_rows(root, "trace_steps")
    materialized_observations = _materialized_rows(root, "observations")
    materialized_branches = _materialized_rows(root, "branches")
    policies = _materialized_rows(root, "policies")
    variant_ids_in_errors = {str(item.get("_kg_v2_variant_id") or "") for item in errors}
    check_ids = {str(item.get("check_id") or "") for item in checks}
    trace_ids = {str(item.get("trace_id") or "") for item in traces}
    outcome_by_id = {str(item.get("outcome_id") or ""): item for item in outcomes}
    materialized_trace_step_ids = {str(item.get("trace_step_id") or "") for item in materialized_trace_steps}
    materialized_observation_ids = {str(item.get("observation_id") or "") for item in materialized_observations}
    materialized_branch_ids = {str(item.get("branch_rule_id") or "") for item in materialized_branches}
    policy_variant_ids = {str(item.get("_kg_v2_variant_id") or "") for item in policies}
    policy_by_variant_id = {
        str(item.get("_kg_v2_variant_id") or ""): item
        for item in policies
        if str(item.get("_kg_v2_variant_id") or "")
    }
    unsafe_action_ids = {
        str(action.get("action_id") or "")
        for policy in policies
        for action in policy.get("unsafe_actions") or []
        if isinstance(action, dict) and str(action.get("action_id") or "")
    }

    stored_actions = store.object_index("DiagnosticAction")
    stored_cases = store.object_index("SourceCase")
    case_rows: list[dict[str, Any]] = []
    corrections = report.get("corrections") if isinstance(report.get("corrections"), list) else []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        case_id = str(correction.get("case_id") or "")
        review_id = str(correction.get("corrected_review_id") or "")
        item = by_review_id.get(review_id, {})
        graph = _candidate(item)
        objects = graph.get("objects") if isinstance(graph.get("objects"), dict) else {}
        variants = {
            str(value.get("variant_id") or "")
            for value in objects.get("FaultVariant") or []
            if isinstance(value, dict) and str(value.get("variant_id") or "")
        }
        actions = {
            str(value.get("action_id") or ""): value
            for value in objects.get("DiagnosticAction") or []
            if isinstance(value, dict) and str(value.get("action_id") or "")
        }
        eligible_actions = {
            action_id
            for action_id, value in actions.items()
            if str(value.get("action_role") or "") in EXECUTION_CHECK_ROLES
        }
        candidate_traces = {
            str(value.get("trace_id") or "")
            for value in objects.get("DiagnosticTrace") or []
            if isinstance(value, dict) and str(value.get("trace_id") or "")
        }
        candidate_outcomes = {
            str(value.get("outcome_id") or ""): value
            for value in objects.get("ActionOutcome") or []
            if isinstance(value, dict) and str(value.get("outcome_id") or "")
        }
        source_case_ids = {
            str(value.get("case_id") or "")
            for value in objects.get("SourceCase") or []
            if isinstance(value, dict) and str(value.get("case_id") or "")
        }
        candidate_trace_steps = {
            str(value.get("trace_step_id") or ""): value
            for value in objects.get("TraceStep") or []
            if isinstance(value, dict) and str(value.get("trace_step_id") or "")
        }
        candidate_observations = {
            str(value.get("observation_id") or ""): value
            for value in objects.get("ExecutionObservation") or []
            if isinstance(value, dict) and str(value.get("observation_id") or "")
        }
        candidate_branches = {
            str(value.get("branch_rule_id") or ""): value
            for value in objects.get("BranchRule") or []
            if isinstance(value, dict) and str(value.get("branch_rule_id") or "")
        }
        review_basis = ((correction.get("correction") or {}).get("review_basis") or {}) if isinstance(correction.get("correction"), dict) else {}
        expected_actual_execution_count = sum(int(value.get("observation_count") or 0) for value in candidate_observations.values())
        expected_unsafe = {
            str(value.get("action_id") or "")
            for value in candidate_outcomes.values()
            if str(value.get("action_id") or "")
            and (
                bool(value.get("high_cost"))
                or bool(value.get("destructive"))
                or str(value.get("outcome_type") or "") == "pending_validation"
                or bool(actions.get(str(value.get("action_id") or ""), {}).get("high_cost"))
                or bool(actions.get(str(value.get("action_id") or ""), {}).get("destructive"))
            )
        }
        negative_outcome_ids = {
            outcome_id
            for outcome_id, value in candidate_outcomes.items()
            if str(value.get("outcome_type") or "") in NEGATIVE_OUTCOME_TYPES
        }

        facts = {
            "case_id": case_id,
            "w4_decision": str((item.get("quality_gate") or {}).get("decision") or ""),
            "selected_action": str(item.get("selected_action") or ""),
            "human_approved": bool(item.get("human_approved")),
            "variant_count": len(variants),
            "eligible_action_count": len(eligible_actions),
            "trace_count": len(candidate_traces),
            "outcome_count": len(candidate_outcomes),
            "trace_step_count": len(candidate_trace_steps),
            "execution_observation_count": len(candidate_observations),
            "branch_rule_count": len(candidate_branches),
            "all_variants_materialized": bool(variants) and variants <= variant_ids_in_errors,
            "all_variant_policies_materialized": bool(variants) and variants <= policy_variant_ids,
            "all_actions_execution_enabled": bool(actions) and all(
                stored_actions.get(action_id, {}).get("execution_materialize_allowed") is True
                for action_id in actions
            ),
            "all_eligible_actions_materialized": all(
                make_id("checkv2", action_id) in check_ids for action_id in eligible_actions
            ),
            "all_traces_materialized": bool(candidate_traces) and all(
                make_id("tracev2", trace_id) in trace_ids for trace_id in candidate_traces
            ),
            "all_outcomes_materialized": bool(candidate_outcomes) and all(
                make_id("outcomev2", outcome_id) in outcome_by_id for outcome_id in candidate_outcomes
            ),
            "all_trace_steps_materialized": bool(candidate_trace_steps) and all(
                make_id("trace-step-v2", trace_step_id) in materialized_trace_step_ids
                for trace_step_id in candidate_trace_steps
            ),
            "all_observations_materialized": bool(candidate_observations) and all(
                make_id("observation-v2", observation_id) in materialized_observation_ids
                for observation_id in candidate_observations
            ),
            "all_branches_materialized": bool(candidate_branches) and all(
                make_id("branch-v2", branch_id) in materialized_branch_ids
                for branch_id in candidate_branches
            ),
            "all_source_cases_execution_enabled": bool(source_case_ids) and all(
                stored_cases.get(source_case_id, {}).get("execution_materialize_allowed") is True
                for source_case_id in source_case_ids
            ),
            "gold_provenance_persisted": bool(source_case_ids) and all(
                stored_cases.get(source_case_id, {}).get("trust_tier") == "gold"
                and stored_cases.get(source_case_id, {}).get("annotation_set_id") == "gold-v1"
                and stored_cases.get(source_case_id, {}).get("annotation_case_id") == case_id
                and stored_cases.get(source_case_id, {}).get("annotation_sha256") == review_basis.get("annotation_sha256")
                and bool(stored_cases.get(source_case_id, {}).get("review_id"))
                and bool(stored_cases.get(source_case_id, {}).get("ingest_run_id"))
                for source_case_id in source_case_ids
            ),
            "actual_steps_have_one_observation": sum(
                1 for value in candidate_trace_steps.values()
                if str(value.get("execution_status") or "") == "actual"
            ) == expected_actual_execution_count,
            "policy_actual_execution_count_exact": bool(variants) and all(
                int((policy_by_variant_id.get(variant_id) or {}).get("actual_execution_count") or 0)
                == expected_actual_execution_count
                for variant_id in variants
            ),
            "all_risky_or_pending_actions_unsafe": expected_unsafe <= unsafe_action_ids,
            "negative_or_pending_outcomes_have_no_solution": all(
                not outcome_by_id.get(make_id("outcomev2", outcome_id), {}).get("target_solution_id")
                for outcome_id in negative_outcome_ids
            ),
        }
        facts["passed"] = (
            facts["selected_action"] == "approve_for_execution_policy"
            and facts["human_approved"]
            and all(
                facts[key]
                for key in (
                    "all_variants_materialized",
                    "all_variant_policies_materialized",
                    "all_actions_execution_enabled",
                    "all_eligible_actions_materialized",
                    "all_traces_materialized",
                    "all_outcomes_materialized",
                    "all_trace_steps_materialized",
                    "all_observations_materialized",
                    "all_branches_materialized",
                    "all_source_cases_execution_enabled",
                    "gold_provenance_persisted",
                    "actual_steps_have_one_observation",
                    "policy_actual_execution_count_exact",
                    "all_risky_or_pending_actions_unsafe",
                    "negative_or_pending_outcomes_have_no_solution",
                )
            )
        )
        case_rows.append(facts)

    case_rows.sort(key=lambda item: str(item.get("case_id") or ""))
    exact_cases = [item["case_id"] for item in case_rows] == EXPECTED_CASE_IDS
    policy_ids = {
        str(item.get("policy_id") or "")
        for item in store.objects_by_type.get("DecisionPolicy") or []
        if isinstance(item, dict) and str(item.get("policy_id") or "")
    }
    related_policy_ids = {
        str(item.get("from") or "")
        for item in store.relations
        if isinstance(item, dict) and str(item.get("relation") or "") == "for_family"
    }
    passed = exact_cases and not issues and all(item["passed"] for item in case_rows) and policy_ids == related_policy_ids
    return {
        "schema_version": "debug_agent_system.gold_v1_execution_acceptance.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "kg_root": str(root),
        "exact_goldcase_001_010": exact_cases,
        "graph_validation": {"status": "valid" if not issues else "invalid", "issues": issues},
        "policy_relation_exact_match": policy_ids == related_policy_ids,
        "case_count": len(case_rows),
        "passed_case_count": sum(bool(item["passed"]) for item in case_rows),
        "cases": case_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify-gold-v1-execution")
    parser.add_argument("--kg-root", required=True)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = verify(
        kg_root=args.kg_root,
        pipeline_report=args.pipeline_report,
        review_queue=args.review_queue,
    )
    body = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "passed_case_count": result["passed_case_count"],
        "graph_validation": result["graph_validation"]["status"],
        "policy_relation_exact_match": result["policy_relation_exact_match"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
