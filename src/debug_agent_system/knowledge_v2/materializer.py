"""Materialize KG v2 into the current execution-view JSON KG format."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from debug_agent_system.knowledge_v2.contracts import EXECUTION_CHECK_ROLES, EXECUTION_SLOT_MAP, ProjectedPolicy, make_id, trim_text
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


class KGV2Materializer:
    def __init__(self, store: JsonKGV2Store) -> None:
        self.store = store
        self.objects = store.objects_by_type
        self.relations = [item for item in store.relations if isinstance(item, dict)]
        self.index = {
            obj_type: store.object_index(obj_type)
            for obj_type in self.objects
        }

    def materialize(self, out_root: str | Path | None = None) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []
        solutions: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        trace_steps: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        branches: list[dict[str, Any]] = []
        policies: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        family_to_variants = self._family_to_variants()
        for family in self.objects.get("FaultFamily") or []:
            if not isinstance(family, dict):
                continue
            if _execution_materialize_blocked(family):
                continue
            family_error_id = self._family_error_id(family)
            family_error = self._project_error(family, None)
            errors.append(family_error)
            family_projection = self._project_execution_for_target(family, None, family_error_id)
            checks.extend(family_projection["checks"])
            solutions.extend(family_projection["solutions"])
            traces.extend(family_projection["traces"])
            outcomes.extend(family_projection["outcomes"])
            trace_steps.extend(family_projection["trace_steps"])
            observations.extend(family_projection["observations"])
            branches.extend(family_projection["branches"])
            policies.extend(family_projection["policies"])
            edges.extend(family_projection["edges"])
            for variant in family_to_variants.get(str(family.get("family_id") or ""), []):
                if _execution_materialize_blocked(variant):
                    continue
                variant_error_id = self._variant_error_id(variant)
                errors.append(self._project_error(family, variant))
                edges.append({"from": variant_error_id, "to": family_error_id, "relation": "alias_of"})
                if not self._should_project_variant_execution(family, variant):
                    continue
                variant_projection = self._project_execution_for_target(family, variant, variant_error_id)
                checks.extend(variant_projection["checks"])
                solutions.extend(variant_projection["solutions"])
                traces.extend(variant_projection["traces"])
                outcomes.extend(variant_projection["outcomes"])
                trace_steps.extend(variant_projection["trace_steps"])
                observations.extend(variant_projection["observations"])
                branches.extend(variant_projection["branches"])
                policies.extend(variant_projection["policies"])
                edges.extend(variant_projection["edges"])

        result = {
            "errors": _dedupe_by_id(errors, "error_id"),
            "checks": _dedupe_by_id(checks, "check_id"),
            "solutions": _dedupe_by_id(solutions, "solution_id"),
            "traces": _dedupe_by_id(traces, "trace_id"),
            "outcomes": _dedupe_by_id(outcomes, "outcome_id"),
            "trace_steps": _dedupe_by_id(trace_steps, "trace_step_id"),
            "observations": _dedupe_by_id(observations, "observation_id"),
            "branches": _dedupe_by_id(branches, "branch_rule_id"),
            "policies": _dedupe_by_id(policies, "policy_id"),
            "edges": _dedupe_edges(edges),
        }
        if out_root is not None:
            self.write_execution_view(Path(out_root), result)
        return result

    def build_policy_objects(self) -> list[dict[str, Any]]:
        policies: list[dict[str, Any]] = []
        for family in self.objects.get("FaultFamily") or []:
            if not isinstance(family, dict):
                continue
            if _execution_materialize_blocked(family):
                continue
            family_id = str(family.get("family_id") or "")
            actions = [
                item for item in self.objects.get("DiagnosticAction") or []
                if isinstance(item, dict) and str(item.get("family_id") or "") == family_id
                and not _execution_materialize_blocked(item)
            ]
            traces = [
                item for item in self.objects.get("DiagnosticTrace") or []
                if isinstance(item, dict) and str(item.get("family_id") or "") == family_id
                and not _execution_materialize_blocked(item)
            ]
            outcomes = [
                item for item in self.objects.get("ActionOutcome") or []
                if isinstance(item, dict) and str(item.get("family_id") or "") == family_id
                and not _execution_materialize_blocked(item)
            ]
            trace_ids = {str(item.get("trace_id") or "") for item in traces}
            trace_steps = [
                item for item in self.objects.get("TraceStep") or []
                if isinstance(item, dict) and str(item.get("trace_id") or "") in trace_ids
                and not _execution_materialize_blocked(item)
            ]
            trace_step_ids = {str(item.get("trace_step_id") or "") for item in trace_steps}
            observations = [
                item for item in self.objects.get("ExecutionObservation") or []
                if isinstance(item, dict) and str(item.get("trace_step_id") or "") in trace_step_ids
                and not _execution_materialize_blocked(item)
            ]
            branches = [
                item for item in self.objects.get("BranchRule") or []
                if isinstance(item, dict) and str(item.get("trace_id") or "") in trace_ids
                and not _execution_materialize_blocked(item)
            ]
            if not actions and not traces and not outcomes:
                continue
            order_map: dict[str, list[int]] = {}
            for trace in traces:
                for idx, action_id in enumerate(trace.get("recommended_action_ids") or [], start=1):
                    order_map.setdefault(str(action_id), []).append(idx)
            ordered_actions = sorted(
                actions,
                key=lambda item: (
                    mean(order_map.get(str(item.get("action_id") or ""), [float(item.get("step_order") or 999.0)])),
                    int(item.get("step_order") or 999),
                    str(item.get("label") or ""),
                ),
            )
            ineffective_action_ids = sorted({
                str(item.get("action_id") or "")
                for item in outcomes
                if str(item.get("outcome_type") or "") in {"ineffective", "context_not_root_cause"}
                and str(item.get("action_id") or "")
            })
            high_cost_action_ids = sorted({
                str(item.get("action_id") or "")
                for item in outcomes
                if (item.get("high_cost") or item.get("destructive")) and str(item.get("action_id") or "")
            } | {
                str(item.get("action_id") or "")
                for item in actions
                if (item.get("high_cost") or item.get("destructive")) and str(item.get("action_id") or "")
            })
            execution_stats = _execution_stats(ordered_actions, outcomes, observations)
            policy = {
                "policy_id": make_id("policy", family_id),
                "family_id": family_id,
                "source_trace_ids": [str(item.get("trace_id") or "") for item in traces if str(item.get("trace_id") or "")],
                "source_outcome_ids": [str(item.get("outcome_id") or "") for item in outcomes if str(item.get("outcome_id") or "")],
                "ordered_action_ids": [str(item.get("action_id") or "") for item in ordered_actions if str(item.get("action_id") or "")],
                "ineffective_action_ids": ineffective_action_ids,
                "high_cost_action_ids": high_cost_action_ids,
                "source_case_count": len({str(item.get("source_case_id") or "") for item in traces if str(item.get("source_case_id") or "")}),
                "actual_execution_count": sum(int(item.get("observation_count") or 0) for item in observations),
                "execution_stats": execution_stats,
                "observation_ids": [str(item.get("observation_id") or "") for item in observations if str(item.get("observation_id") or "")],
                "branch_rule_ids": [str(item.get("branch_rule_id") or "") for item in branches if str(item.get("branch_rule_id") or "")],
                "deterministic_recompute": True,
            }
            policies.append(policy)
        return _dedupe_by_id(policies, "policy_id")

    def write_execution_view(self, out_root: Path, materialized: dict[str, Any]) -> dict[str, Any]:
        instances = out_root / "instances"
        for folder in ("errors", "checks", "solutions", "traces", "outcomes", "trace_steps", "observations", "branches", "policies", "sites", "versions", "tickets"):
            (instances / folder).mkdir(parents=True, exist_ok=True)
        _write_json(instances / "errors" / "errors.json", materialized["errors"])
        _write_json(instances / "checks" / "checks.json", materialized["checks"])
        _write_json(instances / "solutions" / "solutions.json", materialized["solutions"])
        _write_json(instances / "traces" / "traces.json", materialized["traces"])
        _write_json(instances / "outcomes" / "outcomes.json", materialized["outcomes"])
        _write_json(instances / "trace_steps" / "trace_steps.json", materialized["trace_steps"])
        _write_json(instances / "observations" / "observations.json", materialized["observations"])
        _write_json(instances / "branches" / "branches.json", materialized["branches"])
        _write_json(instances / "policies" / "policies.json", materialized["policies"])
        _write_json(out_root / "edges.json", materialized["edges"])
        (out_root / "review_queue").mkdir(parents=True, exist_ok=True)
        return {
            "status": "written",
            "out_root": str(out_root),
            "counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
        }

    def _project_error(self, family: dict[str, Any], variant: dict[str, Any] | None) -> dict[str, Any]:
        target = variant or family
        required = self._project_required_info(family, variant)
        payload = {
            "_kg_v2_family_id": str(family.get("family_id") or ""),
            "_kg_v2_variant_id": str((variant or {}).get("variant_id") or ""),
        }
        if variant is None:
            return {
                "type": "Error",
                "error_id": self._family_error_id(family),
                "label": trim_text(family.get("label") or "", 60),
                "symptom": trim_text(family.get("summary") or "", 240),
                "category": str(family.get("category") or "系统与软件异常"),
                "subsystem": str(family.get("subsystem") or ""),
                "scenario": str(family.get("scenario") or ""),
                "keywords": list(family.get("keywords") or []),
                "entry_role": "canonical",
                "required_info": [item["question"] for item in required],
                "required_info_schema": required,
                "escalation_target": str(family.get("escalation_target") or ""),
                **payload,
            }
        return {
            "type": "Error",
            "error_id": self._variant_error_id(variant),
            "label": trim_text(target.get("label") or "", 60),
            "symptom": trim_text(target.get("summary") or family.get("summary") or "", 240),
            "category": str(family.get("category") or "系统与软件异常"),
            "subsystem": str(target.get("subsystem") or family.get("subsystem") or ""),
            "scenario": str(target.get("error_phase") or family.get("scenario") or ""),
            "keywords": list(dict.fromkeys([*(family.get("keywords") or []), *(target.get("keywords") or [])])),
            "entry_role": "case_variant",
            "canonical_error_id": self._family_error_id(family),
            "required_info": [item["question"] for item in required],
            "required_info_schema": required,
            "escalation_target": str(target.get("escalation_target") or family.get("escalation_target") or ""),
            **payload,
        }

    def _project_execution_for_target(
        self,
        family: dict[str, Any],
        variant: dict[str, Any] | None,
        error_id: str,
    ) -> dict[str, Any]:
        target_actions = self._target_actions(family, variant)
        ordered_actions = [item for item in self._ordered_actions(target_actions, family, variant) if item.get("action_role") in EXECUTION_CHECK_ROLES]
        checks = []
        traces = []
        outcomes = []
        trace_steps = []
        observations = []
        branches = []
        solutions = []
        edges: list[dict[str, Any]] = []
        check_id_by_action_id: dict[str, str] = {}
        for order, action in enumerate(ordered_actions, start=1):
            check_id = make_id("checkv2", action.get("action_id") or action.get("label") or f"{error_id}-{order}")
            check_id_by_action_id[str(action.get("action_id") or "")] = check_id
            checks.append({
                "type": "DiagnosticCheck",
                "check_id": check_id,
                "label": trim_text(action.get("label") or "", 80),
                "how_to_check": trim_text(action.get("summary") or action.get("label") or "", 240),
                "step_order": order,
                "source": "kg_v2",
                "source_title": str(family.get("label") or ""),
                "action_role": str(action.get("action_role") or ""),
                "stage": str(action.get("stage") or ""),
                "safety_level": str(action.get("safety_level") or "safe"),
                "applicability_condition": str(action.get("applicability_condition") or ""),
                "expected_result": str(action.get("expected_result") or ""),
                "media_refs": [
                    dict(item)
                    for item in action.get("curated_image_refs") or []
                    if isinstance(item, dict)
                ],
                "condition_tags": [str((variant or {}).get("error_phase") or "")] if variant else [],
            })
        if checks:
            edges.append({"from": error_id, "to": checks[0]["check_id"], "relation": "has_check"})
            for prev, nxt in zip(checks, checks[1:]):
                edges.append({"from": prev["check_id"], "to": nxt["check_id"], "relation": "next"})

        target_outcomes = self._target_outcomes(family, variant)
        for item in target_outcomes:
            action_id = str(item.get("action_id") or "")
            outcome_id = make_id("outcomev2", item.get("outcome_id") or action_id)
            check_id = check_id_by_action_id.get(action_id)
            solution_id = ""
            if (
                str(item.get("outcome_type") or "") == "verified_fix"
                and str(item.get("activation_mode") or "")
                != "human_confirmed_runtime"
                and not item.get("high_cost")
                and not item.get("destructive")
            ):
                solution_id = make_id("solv2", item.get("outcome_id") or action_id)
                solutions.append({
                    "type": "Solution",
                    "solution_id": solution_id,
                    "content": trim_text(item.get("summary") or "", 240),
                    "method": str((self.index["DiagnosticAction"].get(action_id) or {}).get("action_role") or "case"),
                    "evidence_level": "case_chat_evidence",
                    "source": "kg_v2",
                    "source_title": str(family.get("label") or ""),
                })
                if check_id:
                    edges.append({"from": check_id, "to": solution_id, "relation": "resolved_by"})
            outcomes.append({
                "type": "DiagnosticOutcome",
                "outcome_id": outcome_id,
                "source_episode_id": str(item.get("source_case_id") or ""),
                "target_error_id": error_id,
                "target_check_id": check_id or "",
                "target_solution_id": solution_id,
                "action_label": trim_text(item.get("summary") or "", 120),
                "outcome_type": str(item.get("outcome_type") or ""),
                "activation_mode": str(item.get("activation_mode") or ""),
                "activation_requirements": dict(item.get("activation_requirements") or {}),
                "evidence_message_ids": list(item.get("evidence_ids") or []),
                "root_cause_summary": str(item.get("root_cause_summary") or ""),
            })
            edges.append({"from": error_id, "to": outcome_id, "relation": "has_outcome"})
            if check_id:
                edges.append({"from": outcome_id, "to": check_id, "relation": "outcome_check"})
            if solution_id:
                edges.append({"from": outcome_id, "to": solution_id, "relation": "outcome_solution"})

        target_traces = self._target_traces(family, variant)
        target_trace_steps = self._target_trace_steps(target_traces)
        target_observations = self._target_observations(target_trace_steps)
        target_branches = self._target_branches(target_traces)
        materialized_step_id_by_source = {
            str(item.get("trace_step_id") or ""): make_id("trace-step-v2", item.get("trace_step_id") or "step")
            for item in target_trace_steps
        }
        for item in target_traces:
            trace_id = make_id("tracev2", item.get("trace_id") or item.get("source_case_id") or error_id)
            traces.append({
                "type": "DiagnosticTrace",
                "trace_id": trace_id,
                "source_episode_id": str(item.get("source_case_id") or ""),
                "target_error_id": error_id,
                "recommended_order": [
                    {"check_id": check_id_by_action_id.get(action_id, ""), "label": self._action_label(action_id), "order": idx}
                    for idx, action_id in enumerate(item.get("recommended_action_ids") or [], start=1)
                    if check_id_by_action_id.get(action_id)
                ],
                "actual_order": [
                    {"check_id": check_id_by_action_id.get(action_id, ""), "label": self._action_label(action_id), "order": idx}
                    for idx, action_id in enumerate(item.get("actual_action_ids") or [], start=1)
                    if check_id_by_action_id.get(action_id)
                ],
                "evidence_message_ids": list(item.get("evidence_ids") or []),
            })
            edges.append({"from": error_id, "to": trace_id, "relation": "has_trace"})

        for item in target_trace_steps:
            source_step_id = str(item.get("trace_step_id") or "")
            source_trace_id = str(item.get("trace_id") or "")
            action_id = str(item.get("action_id") or "")
            step_id = materialized_step_id_by_source[source_step_id]
            projected_trace_id = make_id("tracev2", source_trace_id or item.get("source_case_id") or error_id)
            trace_steps.append({
                "type": "TraceStep",
                "trace_step_id": step_id,
                "source_trace_step_id": source_step_id,
                "source_episode_id": str(item.get("source_case_id") or ""),
                "target_error_id": error_id,
                "target_trace_id": projected_trace_id,
                "target_check_id": check_id_by_action_id.get(action_id, ""),
                "action_id": action_id,
                "ordinal": int(item.get("ordinal") or 0),
                "execution_status": str(item.get("execution_status") or ""),
                "attempt_index": int(item.get("attempt_index") or 0),
                "evidence_message_ids": list(item.get("evidence_ids") or []),
            })
            edges.extend([
                {"from": projected_trace_id, "to": step_id, "relation": "has_trace_step"},
                *([{"from": step_id, "to": check_id_by_action_id[action_id], "relation": "step_check"}] if action_id in check_id_by_action_id else []),
            ])
        for item in target_trace_steps:
            source_step_id = str(item.get("trace_step_id") or "")
            ordinal = int(item.get("ordinal") or 0)
            source_trace_id = str(item.get("trace_id") or "")
            next_source_id = next((
                str(candidate.get("trace_step_id") or "")
                for candidate in target_trace_steps
                if str(candidate.get("trace_id") or "") == source_trace_id
                and int(candidate.get("ordinal") or 0) == ordinal + 1
            ), "")
            if next_source_id:
                edges.append({
                    "from": materialized_step_id_by_source[source_step_id],
                    "to": materialized_step_id_by_source[next_source_id],
                    "relation": "next_trace_step",
                })

        for item in target_observations:
            source_observation_id = str(item.get("observation_id") or "")
            source_step_id = str(item.get("trace_step_id") or "")
            action_id = str(item.get("action_id") or "")
            observation_id = make_id("observation-v2", source_observation_id or source_step_id)
            observations.append({
                "type": "ExecutionObservation",
                "observation_id": observation_id,
                "source_observation_id": source_observation_id,
                "source_episode_id": str(item.get("source_case_id") or ""),
                "target_error_id": error_id,
                "target_trace_step_id": materialized_step_id_by_source.get(source_step_id, ""),
                "target_check_id": check_id_by_action_id.get(action_id, ""),
                "action_id": action_id,
                "attempt_index": int(item.get("attempt_index") or 0),
                "actual_execution_count": int(item.get("observation_count") or 0),
                "source_outcome_ids": list(item.get("outcome_ids") or []),
                "outcome_types": list(item.get("outcome_types") or []),
                "observation_window": str(item.get("observation_window") or ""),
                "evidence_message_ids": list(item.get("evidence_ids") or []),
            })
            if source_step_id in materialized_step_id_by_source:
                edges.append({"from": materialized_step_id_by_source[source_step_id], "to": observation_id, "relation": "has_observation"})

        for item in target_branches:
            source_branch_id = str(item.get("branch_rule_id") or "")
            source_from_id = str(item.get("from_trace_step_id") or "")
            source_to_id = str(item.get("to_trace_step_id") or "")
            branch_id = make_id("branch-v2", source_branch_id or source_from_id)
            branches.append({
                "type": "BranchRule",
                "branch_rule_id": branch_id,
                "source_branch_rule_id": source_branch_id,
                "source_episode_id": str(item.get("source_case_id") or ""),
                "target_error_id": error_id,
                "from_trace_step_id": materialized_step_id_by_source.get(source_from_id, ""),
                "to_trace_step_id": materialized_step_id_by_source.get(source_to_id, ""),
                "from_check_id": check_id_by_action_id.get(str((self.index["TraceStep"].get(source_from_id) or {}).get("action_id") or ""), ""),
                "to_check_id": check_id_by_action_id.get(str((self.index["TraceStep"].get(source_to_id) or {}).get("action_id") or ""), ""),
                "trigger_outcome_types": list(item.get("trigger_outcome_types") or []),
                "condition": str(item.get("condition") or ""),
                "condition_code": str(item.get("condition_code") or ""),
                "branch_kind": str(item.get("branch_kind") or ""),
                "terminal_status": str(item.get("terminal_status") or ""),
                "priority": int(item.get("priority") or 0),
                "evidence_message_ids": list(item.get("evidence_ids") or []),
            })
            if source_from_id in materialized_step_id_by_source:
                edges.append({"from": materialized_step_id_by_source[source_from_id], "to": branch_id, "relation": "has_branch"})
            if source_to_id in materialized_step_id_by_source:
                edges.append({"from": branch_id, "to": materialized_step_id_by_source[source_to_id], "relation": "branch_to"})

        policy = self._project_policy(
            family, variant, error_id, ordered_actions, target_outcomes, target_traces,
            target_observations, target_branches, check_id_by_action_id, materialized_step_id_by_source,
        )
        policies = [policy.payload]
        edges.append({"from": error_id, "to": policy.policy_id, "relation": "has_policy"})

        return {
            "checks": checks,
            "solutions": solutions,
            "traces": traces,
            "outcomes": outcomes,
            "trace_steps": trace_steps,
            "observations": observations,
            "branches": branches,
            "policies": policies,
            "edges": edges,
        }

    def _project_policy(
        self,
        family: dict[str, Any],
        variant: dict[str, Any] | None,
        error_id: str,
        ordered_actions: list[dict[str, Any]],
        target_outcomes: list[dict[str, Any]],
        target_traces: list[dict[str, Any]],
        target_observations: list[dict[str, Any]],
        target_branches: list[dict[str, Any]],
        check_id_by_action_id: dict[str, str],
        materialized_step_id_by_source: dict[str, str],
    ) -> ProjectedPolicy:
        by_action: dict[str, dict[str, Any]] = {}
        order_index = {str(item.get("action_id") or ""): idx for idx, item in enumerate(ordered_actions, start=1)}
        for outcome in target_outcomes:
            action_id = str(outcome.get("action_id") or "")
            entry = by_action.setdefault(action_id, {"verified_fix": 0, "ineffective": 0, "partial_temporary": 0, "pending_validation": 0, "all": 0})
            key = str(outcome.get("outcome_type") or "")
            if (
                key == "verified_fix"
                and str(outcome.get("activation_mode") or "")
                == "human_confirmed_runtime"
            ):
                entry["all"] += 1
                continue
            if key in entry:
                entry[key] += 1
            entry["all"] += 1
        ordered_checks: list[dict[str, Any]] = []
        for action in ordered_actions:
            action_id = str(action.get("action_id") or "")
            stats = by_action.get(action_id, {})
            check_id = check_id_by_action_id.get(action_id)
            if not check_id:
                continue
            trace_orders: list[int] = []
            for trace in target_traces:
                for idx, item in enumerate(trace.get("recommended_action_ids") or [], start=1):
                    if str(item) == action_id:
                        trace_orders.append(idx)
            avg_order = round(mean(trace_orders), 3) if trace_orders else float(order_index.get(action_id) or 999.0)
            policy_prior = round(
                float(stats.get("verified_fix", 0)) * 3.0
                + max(0.0, 2.5 - avg_order / 2.0)
                - float(stats.get("ineffective", 0)) * 2.0
                - float(stats.get("pending_validation", 0)) * 1.0,
                4,
            )
            ordered_checks.append({
                "check_id": check_id,
                "action_id": action_id,
                "avg_order": avg_order,
                "verified_fix_count": int(stats.get("verified_fix", 0)),
                "ineffective_count": int(stats.get("ineffective", 0)),
                "partial_temporary_count": int(stats.get("partial_temporary", 0)),
                "actual_execution_count": sum(
                    int(item.get("observation_count") or 0)
                    for item in target_observations
                    if str(item.get("action_id") or "") == action_id
                ),
                "policy_prior": policy_prior,
            })
        ordered_checks.sort(key=lambda item: (-float(item.get("policy_prior") or 0.0), float(item.get("avg_order") or 999.0), item.get("check_id") or ""))
        solution_stats = []
        unsafe_actions = []
        for outcome in target_outcomes:
            action_id = str(outcome.get("action_id") or "")
            action = self.index["DiagnosticAction"].get(action_id) or {}
            solution_stats.append({
                "action_id": action_id,
                "label": trim_text(outcome.get("summary") or action.get("label") or "", 80),
                "by_outcome_type": {
                    (
                        "conditional_verified_fix"
                        if str(outcome.get("activation_mode") or "")
                        == "human_confirmed_runtime"
                        else str(outcome.get("outcome_type") or "")
                    ): 1
                },
            })
            if outcome.get("high_cost") or outcome.get("destructive") or str(outcome.get("outcome_type") or "") == "pending_validation":
                unsafe_actions.append({
                    "action_id": action_id,
                    "label": trim_text(action.get("label") or outcome.get("summary") or "", 80),
                    "reason": str(outcome.get("outcome_type") or ""),
                })
        policy_node = {
            "type": "DiagnosticPolicy",
            "policy_id": make_id("policy", error_id),
            "target_error_id": error_id,
            "updated_at": "kg_v2_materialized",
            "source_trace_ids": [make_id("tracev2", item.get("trace_id") or item.get("source_case_id") or error_id) for item in target_traces],
            "source_outcome_ids": [make_id("outcomev2", item.get("outcome_id") or item.get("action_id") or error_id) for item in target_outcomes],
            "source_observation_ids": [make_id("observation-v2", item.get("observation_id") or item.get("trace_step_id") or error_id) for item in target_observations],
            "source_case_count": len({str(item.get("source_case_id") or "") for item in target_traces if str(item.get("source_case_id") or "")}),
            "actual_execution_count": sum(int(item.get("observation_count") or 0) for item in target_observations),
            "ordered_checks": ordered_checks,
            "branches": [
                {
                    "branch_rule_id": make_id("branch-v2", item.get("branch_rule_id") or item.get("from_trace_step_id") or error_id),
                    "from_trace_step_id": materialized_step_id_by_source.get(str(item.get("from_trace_step_id") or ""), ""),
                    "to_trace_step_id": materialized_step_id_by_source.get(str(item.get("to_trace_step_id") or ""), ""),
                    "trigger_outcome_types": list(item.get("trigger_outcome_types") or []),
                    "terminal_status": str(item.get("terminal_status") or ""),
                    "branch_kind": str(item.get("branch_kind") or ""),
                }
                for item in target_branches
            ],
            "solution_stats": _merge_solution_stats(solution_stats),
            "unsafe_actions": _dedupe_list_of_dicts(unsafe_actions, "action_id"),
            "deterministic_recompute": True,
            "_kg_v2_family_id": str(family.get("family_id") or ""),
            "_kg_v2_variant_id": str((variant or {}).get("variant_id") or ""),
        }
        return ProjectedPolicy(
            policy_id=policy_node["policy_id"],
            target_error_id=error_id,
            ordered_checks=policy_node["ordered_checks"],
            solution_stats=policy_node["solution_stats"],
            unsafe_actions=policy_node["unsafe_actions"],
            payload=policy_node,
        )

    def _project_required_info(self, family: dict[str, Any], variant: dict[str, Any] | None) -> list[dict[str, Any]]:
        family_id = str(family.get("family_id") or "")
        variant_id = str((variant or {}).get("variant_id") or "")
        items: list[dict[str, Any]] = []
        for item in self.objects.get("RequiredInfoSpec") or []:
            if not isinstance(item, dict):
                continue
            if _execution_materialize_blocked(item):
                continue
            if str(item.get("family_id") or "") != family_id:
                continue
            item_variant_id = str(item.get("variant_id") or "")
            if variant is not None and item_variant_id not in {"", variant_id}:
                continue
            if variant is None and item_variant_id:
                continue
            slot = EXECUTION_SLOT_MAP.get(str(item.get("slot") or "other"), "other")
            items.append({
                "slot": slot,
                "question": trim_text(item.get("question") or "", 100),
                "condition": str(item.get("condition") or ""),
                "blocks": list(item.get("blocks") or []),
                "priority": str(item.get("priority") or "medium"),
                "why_required": trim_text(item.get("why_required") or "", 160),
                "evidence": {
                    "evidence_ids": list(item.get("evidence_ids") or []),
                    "source_variant_id": item_variant_id,
                    "internal_slot": str(item.get("slot") or ""),
                },
            })
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            key = (str(item.get("slot") or ""), str(item.get("question") or ""))
            deduped.setdefault(key, item)
        return list(deduped.values())

    def _family_to_variants(self) -> dict[str, list[dict[str, Any]]]:
        variants = {}
        for relation in self.relations:
            if relation.get("relation") != "has_variant":
                continue
            family_id = str(relation.get("from") or "")
            variant_id = str(relation.get("to") or "")
            variant = self.index["FaultVariant"].get(variant_id)
            if variant:
                variants.setdefault(family_id, []).append(variant)
        return variants

    def _target_actions(self, family: dict[str, Any], variant: dict[str, Any] | None) -> list[dict[str, Any]]:
        family_id = str(family.get("family_id") or "")
        variant_id = str((variant or {}).get("variant_id") or "")
        out = []
        for item in self.objects.get("DiagnosticAction") or []:
            if not isinstance(item, dict):
                continue
            if _execution_materialize_blocked(item):
                continue
            if str(item.get("family_id") or "") != family_id:
                continue
            item_variant = str(item.get("variant_id") or "")
            if variant is None and item_variant:
                continue
            if variant is not None and item_variant not in {"", variant_id}:
                continue
            out.append(item)
        return out

    def _target_outcomes(self, family: dict[str, Any], variant: dict[str, Any] | None) -> list[dict[str, Any]]:
        family_id = str(family.get("family_id") or "")
        variant_id = str((variant or {}).get("variant_id") or "")
        out = []
        for item in self.objects.get("ActionOutcome") or []:
            if not isinstance(item, dict):
                continue
            if _execution_materialize_blocked(item):
                continue
            if str(item.get("family_id") or "") != family_id:
                continue
            item_variant = str(item.get("variant_id") or "")
            if variant is None and item_variant:
                continue
            if variant is not None and item_variant not in {"", variant_id}:
                continue
            out.append(item)
        return out

    def _target_traces(self, family: dict[str, Any], variant: dict[str, Any] | None) -> list[dict[str, Any]]:
        family_id = str(family.get("family_id") or "")
        variant_id = str((variant or {}).get("variant_id") or "")
        out = []
        for item in self.objects.get("DiagnosticTrace") or []:
            if not isinstance(item, dict):
                continue
            if _execution_materialize_blocked(item):
                continue
            if str(item.get("family_id") or "") != family_id:
                continue
            item_variant = str(item.get("variant_id") or "")
            if variant is None and item_variant:
                continue
            if variant is not None and item_variant not in {"", variant_id}:
                continue
            out.append(item)
        return out

    def _target_trace_steps(self, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trace_ids = {str(item.get("trace_id") or "") for item in traces}
        return sorted(
            [
                item for item in self.objects.get("TraceStep") or []
                if isinstance(item, dict)
                and str(item.get("trace_id") or "") in trace_ids
                and not _execution_materialize_blocked(item)
            ],
            key=lambda item: (str(item.get("trace_id") or ""), int(item.get("ordinal") or 0)),
        )

    def _target_observations(self, trace_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        step_ids = {str(item.get("trace_step_id") or "") for item in trace_steps}
        return [
            item for item in self.objects.get("ExecutionObservation") or []
            if isinstance(item, dict)
            and str(item.get("trace_step_id") or "") in step_ids
            and not _execution_materialize_blocked(item)
        ]

    def _target_branches(self, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trace_ids = {str(item.get("trace_id") or "") for item in traces}
        return sorted(
            [
                item for item in self.objects.get("BranchRule") or []
                if isinstance(item, dict)
                and str(item.get("trace_id") or "") in trace_ids
                and not _execution_materialize_blocked(item)
            ],
            key=lambda item: (
                int((self.index["TraceStep"].get(str(item.get("from_trace_step_id") or "")) or {}).get("ordinal") or 0),
                int(item.get("priority") or 0),
                str(item.get("branch_rule_id") or ""),
            ),
        )

    def _should_project_variant_execution(self, family: dict[str, Any], variant: dict[str, Any]) -> bool:
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            return False
        owner_context = str(variant.get("owner_context") or "")
        if owner_context and not owner_context.startswith("SOP:"):
            return True
        family_id = str(family.get("family_id") or "")
        for obj_type in ("DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace", "TraceStep", "ExecutionObservation", "BranchRule"):
            for item in self.objects.get(obj_type) or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("family_id") or "") != family_id:
                    continue
                if str(item.get("variant_id") or "") == variant_id:
                    return True
        return False

    def _ordered_actions(
        self,
        actions: list[dict[str, Any]],
        family: dict[str, Any],
        variant: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        traces = self._target_traces(family, variant)
        order_map: dict[str, list[int]] = {}
        for trace in traces:
            for idx, action_id in enumerate(trace.get("recommended_action_ids") or [], start=1):
                order_map.setdefault(str(action_id), []).append(idx)
        return sorted(
            actions,
            key=lambda item: (
                mean(order_map.get(str(item.get("action_id") or ""), [float(item.get("step_order") or 999.0)])),
                int(item.get("step_order") or 999),
                str(item.get("label") or ""),
            ),
        )

    def _action_label(self, action_id: str) -> str:
        return trim_text((self.index["DiagnosticAction"].get(action_id) or {}).get("label") or action_id, 80)

    @staticmethod
    def _family_error_id(family: dict[str, Any]) -> str:
        return make_id("errv2", family.get("family_id") or family.get("label") or "family")

    @staticmethod
    def _variant_error_id(variant: dict[str, Any]) -> str:
        return make_id("errv2", variant.get("variant_id") or variant.get("label") or "variant")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dedupe_by_id(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        obj_id = str(item.get(key) or "")
        if not obj_id:
            continue
        if obj_id in out:
            out[obj_id].update({k: v for k, v in item.items() if v not in (None, "", [])})
        else:
            out[obj_id] = dict(item)
    return list(out.values())


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        key = (
            str(edge.get("from") or ""),
            str(edge.get("to") or ""),
            str(edge.get("relation") or ""),
            str(edge.get("condition") or ""),
        )
        if not all(key[:3]) or key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _merge_solution_stats(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        action_id = str(item.get("action_id") or "")
        if not action_id:
            continue
        entry = merged.setdefault(action_id, {"action_id": action_id, "label": item.get("label") or "", "by_outcome_type": {}})
        for key, value in (item.get("by_outcome_type") or {}).items():
            entry["by_outcome_type"][key] = int(entry["by_outcome_type"].get(key) or 0) + int(value or 0)
    return list(merged.values())


def _execution_stats(
    actions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actions:
        action_id = str(action.get("action_id") or "")
        action_outcomes = [item for item in outcomes if str(item.get("action_id") or "") == action_id]
        action_observations = [item for item in observations if str(item.get("action_id") or "") == action_id]
        counts: dict[str, int] = {}
        for outcome in action_outcomes:
            outcome_type = str(outcome.get("outcome_type") or "")
            if (
                outcome_type == "verified_fix"
                and str(outcome.get("activation_mode") or "")
                == "human_confirmed_runtime"
            ):
                outcome_type = "conditional_verified_fix"
            counts[outcome_type] = int(counts.get(outcome_type) or 0) + 1
        rows.append({
            "action_id": action_id,
            "source_case_count": len({
                str(item.get("source_case_id") or "")
                for item in action_observations
                if str(item.get("source_case_id") or "")
            }),
            "actual_execution_count": sum(int(item.get("observation_count") or 0) for item in action_observations),
            "outcome_record_count": len(action_outcomes),
            "outcome_counts": counts,
        })
    return rows


def _dedupe_list_of_dicts(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get(key) or "")
        if item_id and item_id not in out:
            out[item_id] = item
    return list(out.values())


def _execution_materialize_blocked(item: dict[str, Any]) -> bool:
    return item.get("execution_materialize_allowed") is False
