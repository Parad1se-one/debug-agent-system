"""Deterministic compiler for reviewed trace execution semantics."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from debug_agent_system.knowledge_v2.contracts import (
    V2_PRIMARY_KEYS,
    make_id,
    trim_text,
)


def _semantic_id(prefix: str, value: str) -> str:
    raw = str(value or "")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return make_id(prefix, f"{raw}:{digest}")


def _branch_destination(outcome_type: str, next_step_id: str) -> tuple[str, str]:
    if outcome_type == "verified_fix":
        return "", "resolved"
    if next_step_id:
        return next_step_id, "continue"
    if outcome_type in {"partial_temporary", "mitigation_observed"}:
        return "", "monitoring"
    return "", "unresolved"


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for relation in relations:
        key = (
            str(relation.get("from") or ""),
            str(relation.get("to") or ""),
            str(relation.get("relation") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        output.append(relation)
    return output


class TraceCompiler:
    """Compile TraceStep/Observation/Branch objects without inventing facts."""

    compiler_version = "w7.trace_compiler.v1"

    @staticmethod
    def rebuild_execution_objects(
        objects: dict[str, list[dict[str, Any]]],
        changes: list[dict[str, Any]] | None = None,
    ) -> None:
        changes = changes if changes is not None else []
        old_steps = [
            item
            for item in objects.get("TraceStep") or []
            if isinstance(item, dict)
        ]
        traces = [
            item
            for item in objects.get("DiagnosticTrace") or []
            if isinstance(item, dict)
        ]
        if not old_steps and not any(
            isinstance(item.get("action_occurrences"), list)
            and item.get("action_occurrences")
            for item in traces
        ):
            return
        old_observations = [
            item
            for item in objects.get("ExecutionObservation") or []
            if isinstance(item, dict)
        ]
        outcome_by_id = {
            str(item.get("outcome_id") or ""): item
            for item in objects.get("ActionOutcome") or []
            if isinstance(item, dict) and str(item.get("outcome_id") or "")
        }
        outcomes_by_action: dict[str, list[dict[str, Any]]] = {}
        for outcome in outcome_by_id.values():
            outcomes_by_action.setdefault(
                str(outcome.get("action_id") or ""), []
            ).append(outcome)
        action_by_id = {
            str(item.get("action_id") or ""): item
            for item in objects.get("DiagnosticAction") or []
            if isinstance(item, dict) and str(item.get("action_id") or "")
        }

        rebuilt_steps: list[dict[str, Any]] = []
        rebuilt_observations: list[dict[str, Any]] = []
        rebuilt_branches: list[dict[str, Any]] = []
        for trace in traces:
            trace_id = str(trace.get("trace_id") or "")
            case_id = str(trace.get("source_case_id") or "")
            recommended = [
                str(value)
                for value in trace.get("recommended_action_ids") or []
                if str(value) in action_by_id
            ]
            recommended = list(dict.fromkeys(recommended))
            actual = [
                str(value)
                for value in trace.get("actual_action_ids") or []
                if str(value) in recommended
            ]
            actual = list(dict.fromkeys(actual))
            trace["recommended_action_ids"] = recommended
            trace["actual_action_ids"] = actual
            raw_occurrences = trace.get("action_occurrences")
            has_explicit_occurrences = (
                isinstance(raw_occurrences, list)
                and bool(raw_occurrences)
            )
            occurrences: list[dict[str, Any]] = []
            if has_explicit_occurrences:
                for raw in raw_occurrences:
                    if not isinstance(raw, dict):
                        continue
                    action_id = str(raw.get("action_id") or "")
                    if action_id not in recommended:
                        continue
                    occurrence_case_id = str(
                        raw.get("source_case_id") or case_id
                    )
                    status = str(raw.get("execution_status") or "")
                    if status not in {"actual", "recommended"}:
                        status = (
                            "actual"
                            if action_id in actual
                            else "recommended"
                        )
                    occurrences.append({
                        "action_id": action_id,
                        "source_case_id": occurrence_case_id,
                        "execution_status": status,
                        "attempt_index": max(
                            0, int(raw.get("attempt_index") or 0)
                        ),
                        "evidence_ids": [
                            str(value)
                            for value in raw.get("evidence_ids") or []
                            if str(value)
                        ],
                        "case_ref": str(raw.get("case_ref") or ""),
                        "phase_index": max(
                            0, int(raw.get("phase_index") or 0)
                        ),
                    })
            else:
                occurrences = [{
                    "action_id": action_id,
                    "source_case_id": case_id,
                    "execution_status": (
                        "actual"
                        if action_id in actual
                        else "recommended"
                    ),
                    "attempt_index": (
                        1 if action_id in actual else 0
                    ),
                    "evidence_ids": [],
                    "case_ref": "",
                    "phase_index": 0,
                } for action_id in recommended]
            if has_explicit_occurrences:
                trace["action_occurrences"] = occurrences
            trace_step_ids: list[str] = []
            trace_steps: list[dict[str, Any]] = []
            occurrence_counts: dict[tuple[str, str], int] = {}
            for ordinal, occurrence in enumerate(occurrences, start=1):
                action_id = str(occurrence.get("action_id") or "")
                occurrence_case_id = str(
                    occurrence.get("source_case_id") or case_id
                )
                occurrence_key = (occurrence_case_id, action_id)
                occurrence_counts[occurrence_key] = (
                    occurrence_counts.get(occurrence_key, 0) + 1
                )
                occurrence_index = occurrence_counts[occurrence_key]
                matching_steps = [
                    item
                    for item in old_steps
                    if str(item.get("trace_id") or "") == trace_id
                    and str(item.get("action_id") or "") == action_id
                    and str(
                        item.get("source_case_id") or occurrence_case_id
                    ) == occurrence_case_id
                    and int(item.get("ordinal") or 0) == ordinal
                ]
                evidence_ids = list(dict.fromkeys(
                    [
                        *(
                            str(value)
                            for value in occurrence.get("evidence_ids") or []
                            if str(value)
                        ),
                        *(
                            evidence_id
                            for item in matching_steps
                            for evidence_id in item.get("evidence_ids") or []
                            if str(evidence_id)
                        ),
                    ]
                )) or list(trace.get("evidence_ids") or [])[:1]
                status = str(
                    occurrence.get("execution_status") or "recommended"
                )
                if (
                    status == "actual"
                    or str(
                        action_by_id[action_id].get("execution_status") or ""
                    ) != "actual"
                ):
                    action_by_id[action_id]["execution_status"] = status
                action_by_id[action_id].setdefault("step_order", ordinal)
                attempt_index = int(
                    occurrence.get("attempt_index") or 0
                )
                if status == "actual" and attempt_index < 1:
                    attempt_index = occurrence_index
                step_id = _semantic_id(
                    "trace-step",
                    f"{trace_id}:{ordinal}:{occurrence_case_id}:"
                    f"{action_id}:{attempt_index}",
                )
                trace_step_ids.append(step_id)
                step = {
                    "trace_step_id": step_id,
                    "trace_id": trace_id,
                    "source_case_id": occurrence_case_id,
                    "action_id": action_id,
                    "ordinal": ordinal,
                    "execution_status": status,
                    "attempt_index": attempt_index,
                    "evidence_ids": evidence_ids,
                    "case_ref": str(occurrence.get("case_ref") or ""),
                    "phase_index": int(
                        occurrence.get("phase_index") or 0
                    ),
                }
                trace_steps.append(step)
                rebuilt_steps.append(step)
                if status == "actual":
                    matching_observations = [
                        item
                        for item in old_observations
                        if str(item.get("source_case_id") or "")
                        == occurrence_case_id
                        and str(item.get("action_id") or "") == action_id
                        and int(item.get("attempt_index") or 1)
                        == attempt_index
                    ]
                    referenced_outcome_ids = list(dict.fromkeys(
                        outcome_id
                        for item in matching_observations
                        for outcome_id in item.get("outcome_ids") or []
                        if str(outcome_id) in outcome_by_id
                    ))
                    if not referenced_outcome_ids:
                        referenced_outcome_ids = [
                            str(item.get("outcome_id") or "")
                            for item in outcomes_by_action.get(action_id) or []
                            if (
                                not str(item.get("source_case_id") or "")
                                or str(item.get("source_case_id") or "")
                                == occurrence_case_id
                            )
                        ]
                    observation_evidence = list(dict.fromkeys(
                        evidence_id
                        for item in matching_observations
                        for evidence_id in item.get("evidence_ids") or []
                        if str(evidence_id)
                    )) or evidence_ids
                    rebuilt_observations.append({
                        "observation_id": _semantic_id(
                            "observation",
                            f"{step_id}:attempt:{attempt_index}",
                        ),
                        "trace_step_id": step_id,
                        "source_case_id": occurrence_case_id,
                        "action_id": action_id,
                        "attempt_index": attempt_index,
                        "observation_count": 1,
                        "outcome_ids": referenced_outcome_ids,
                        "outcome_types": sorted({
                            str(
                                (outcome_by_id.get(outcome_id) or {}).get(
                                    "outcome_type"
                                )
                                or ""
                            )
                            for outcome_id in referenced_outcome_ids
                            if str(
                                (outcome_by_id.get(outcome_id) or {}).get(
                                    "outcome_type"
                                )
                                or ""
                            )
                        }),
                        "evidence_ids": observation_evidence,
                    })
            for ordinal, occurrence in enumerate(occurrences, start=1):
                action_id = str(occurrence.get("action_id") or "")
                step_id = trace_step_ids[ordinal - 1]
                next_step_id = (
                    trace_step_ids[ordinal]
                    if ordinal < len(trace_step_ids)
                    else ""
                )
                status = str(
                    occurrence.get("execution_status") or "recommended"
                )
                step_case_id = str(
                    trace_steps[ordinal - 1].get("source_case_id")
                    or case_id
                )
                outcome_types = sorted({
                    str(item.get("outcome_type") or "")
                    for item in outcomes_by_action.get(action_id) or []
                    if str(item.get("outcome_type") or "")
                    and (
                        not str(item.get("source_case_id") or "")
                        or str(item.get("source_case_id") or "")
                        == step_case_id
                    )
                })
                for priority, outcome_type in enumerate(outcome_types, start=1):
                    matching_outcomes = [
                        item
                        for item in outcomes_by_action.get(action_id) or []
                        if str(item.get("outcome_type") or "") == outcome_type
                        and (
                            not str(item.get("source_case_id") or "")
                            or str(item.get("source_case_id") or "")
                            == step_case_id
                        )
                    ]
                    target, terminal = _branch_destination(
                        outcome_type, next_step_id
                    )
                    evidence_ids = list(dict.fromkeys(
                        evidence_id
                        for item in matching_outcomes
                        for evidence_id in item.get("evidence_ids") or []
                        if str(evidence_id)
                    )) or list(trace_steps[ordinal - 1].get("evidence_ids") or [])
                    rebuilt_branches.append({
                        "branch_rule_id": _semantic_id(
                            "branch-rule",
                            f"{step_id}:{outcome_type}:{target}:{terminal}",
                        ),
                        "trace_id": trace_id,
                        "source_case_id": step_case_id,
                        "from_trace_step_id": step_id,
                        "to_trace_step_id": target,
                        "trigger_outcome_types": [outcome_type],
                        "condition": trim_text(
                            f"outcome_type={outcome_type}", 120
                        ),
                        "branch_kind": (
                            "observed_transition"
                            if status == "actual"
                            else "reviewed_recommendation"
                        ),
                        "terminal_status": terminal,
                        "priority": priority,
                        "evidence_ids": evidence_ids,
                    })
        if len(rebuilt_steps) != len(old_steps):
            changes.append({
                "kind": "trace_steps_rebuilt_after_action_normalization",
                "from": len(old_steps),
                "to": len(rebuilt_steps),
            })
        objects["TraceStep"] = rebuilt_steps
        objects["ExecutionObservation"] = rebuilt_observations
        objects["BranchRule"] = rebuilt_branches

    @staticmethod
    def replace_execution_relations(
        objects: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        execution_ids = {
            str(item.get(V2_PRIMARY_KEYS[obj_type]) or "")
            for obj_type in ("TraceStep", "ExecutionObservation", "BranchRule")
            for item in objects.get(obj_type) or []
            if isinstance(item, dict)
            and str(item.get(V2_PRIMARY_KEYS[obj_type]) or "")
        }
        output = [
            relation
            for relation in relations
            if str(relation.get("from") or "") not in execution_ids
            and str(relation.get("to") or "") not in execution_ids
        ]
        steps_by_trace: dict[str, list[dict[str, Any]]] = {}
        for step in objects.get("TraceStep") or []:
            step_id = str(step.get("trace_step_id") or "")
            trace_id = str(step.get("trace_id") or "")
            case_id = str(step.get("source_case_id") or "")
            steps_by_trace.setdefault(trace_id, []).append(step)
            output.extend([
                {
                    "from": trace_id,
                    "to": step_id,
                    "relation": "has_trace_step",
                },
                {
                    "from": step_id,
                    "to": str(step.get("action_id") or ""),
                    "relation": "step_action",
                },
                {
                    "from": case_id,
                    "to": step_id,
                    "relation": "supports",
                },
                *(
                    {
                        "from": evidence_id,
                        "to": step_id,
                        "relation": "evidences",
                    }
                    for evidence_id in step.get("evidence_ids") or []
                ),
            ])
        for steps in steps_by_trace.values():
            ordered = sorted(
                steps, key=lambda item: int(item.get("ordinal") or 0)
            )
            for previous, following in zip(ordered, ordered[1:]):
                output.append({
                    "from": str(previous["trace_step_id"]),
                    "to": str(following["trace_step_id"]),
                    "relation": "next_trace_step",
                })
        for observation in objects.get("ExecutionObservation") or []:
            observation_id = str(observation.get("observation_id") or "")
            output.extend([
                {
                    "from": str(observation.get("trace_step_id") or ""),
                    "to": observation_id,
                    "relation": "has_observation",
                },
                {
                    "from": str(observation.get("source_case_id") or ""),
                    "to": observation_id,
                    "relation": "supports",
                },
                *(
                    {
                        "from": observation_id,
                        "to": outcome_id,
                        "relation": "observed_outcome",
                    }
                    for outcome_id in observation.get("outcome_ids") or []
                ),
                *(
                    {
                        "from": evidence_id,
                        "to": observation_id,
                        "relation": "evidences",
                    }
                    for evidence_id in observation.get("evidence_ids") or []
                ),
            ])
        for branch in objects.get("BranchRule") or []:
            branch_id = str(branch.get("branch_rule_id") or "")
            output.extend([
                {
                    "from": str(branch.get("trace_id") or ""),
                    "to": branch_id,
                    "relation": "has_branch_rule",
                },
                {
                    "from": branch_id,
                    "to": str(branch.get("from_trace_step_id") or ""),
                    "relation": "branch_from",
                },
                {
                    "from": str(branch.get("source_case_id") or ""),
                    "to": branch_id,
                    "relation": "supports",
                },
                *(
                    {
                        "from": evidence_id,
                        "to": branch_id,
                        "relation": "evidences",
                    }
                    for evidence_id in branch.get("evidence_ids") or []
                ),
            ])
            if str(branch.get("to_trace_step_id") or ""):
                output.append({
                    "from": branch_id,
                    "to": str(branch.get("to_trace_step_id") or ""),
                    "relation": "branch_to",
                })
        return _dedupe_relations(output)

    def compile(
        self,
        objects: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
        compiled = deepcopy(objects)
        changes: list[dict[str, Any]] = []
        self.rebuild_execution_objects(compiled, changes)
        compiled_relations = self.replace_execution_relations(
            compiled, deepcopy(relations)
        )
        return compiled, compiled_relations, changes

    def compile_review_bundle(
        self,
        *,
        case_cards: list[dict[str, Any]],
        phase_patch: dict[str, Any],
        outcome_patch: dict[str, Any],
    ) -> dict[str, Any]:
        """Compile local semantic decisions into a hash-bound review bundle.

        This bundle contains no invented Action or Outcome.  Once W2 objects
        exist, ``compile()`` remains responsible for KG-v2 execution objects.
        """

        cards_by_ref: dict[str, dict[str, Any]] = {}
        for index, card in enumerate(case_cards):
            if not isinstance(card, dict):
                continue
            case_ref = str(
                card.get("case_ref")
                or card.get("case_item_ref")
                or card.get("fragment_ref")
                or f"case-{index + 1}"
            )
            cards_by_ref[case_ref] = deepcopy(card)
        groups: dict[str, list[str]] = {}
        phases_by_trace: dict[str, list[dict[str, Any]]] = {}
        for operation in phase_patch.get("operations") or []:
            if not isinstance(operation, dict):
                continue
            trace_ref = str(operation.get("local_trace_ref") or "")
            if operation.get("op") == "create_trace_group":
                groups[trace_ref] = [
                    str(value) for value in operation.get("case_refs") or []
                    if str(value)
                ]
            elif operation.get("op") == "set_phase":
                phases_by_trace.setdefault(trace_ref, []).append(
                    deepcopy(operation)
                )
        outcome_by_trace = {
            str(operation.get("local_trace_ref") or ""): operation
            for operation in outcome_patch.get("operations") or []
            if isinstance(operation, dict)
            and str(operation.get("local_trace_ref") or "")
        }
        traces: list[dict[str, Any]] = []
        for trace_ref, case_refs in sorted(groups.items()):
            phases = sorted(
                phases_by_trace.get(trace_ref, []),
                key=lambda item: (
                    int(item.get("phase_index") or 0),
                    str(item.get("case_ref") or ""),
                ),
            )
            outcome = outcome_by_trace.get(trace_ref) or {}
            semantic_key = json.dumps({
                "trace_ref": trace_ref,
                "case_refs": case_refs,
                "phases": phases,
                "outcome": outcome,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            traces.append({
                "local_trace_ref": trace_ref,
                "compiled_trace_id": _semantic_id(
                    "trace", semantic_key
                ),
                "case_refs": case_refs,
                "case_cards": [
                    cards_by_ref[case_ref]
                    for case_ref in case_refs if case_ref in cards_by_ref
                ],
                "phases": phases,
                "resolution_status": str(
                    outcome.get("to") or "unknown"
                ),
                "resolution_evidence_message_ids": list(
                    outcome.get("evidence_message_ids") or []
                ),
            })
        bundle = {
            "schema_version": "w7.compiled_trace_bundle.v1",
            "compiler_version": self.compiler_version,
            "traces": traces,
            "unassigned_case_refs": sorted(
                set(cards_by_ref)
                - {
                    case_ref
                    for case_refs in groups.values()
                    for case_ref in case_refs
                }
                - set(phase_patch.get("standalone_case_refs") or [])
            ),
            "standalone_case_refs": sorted(
                set(phase_patch.get("standalone_case_refs") or [])
            ),
            "stats": {
                "traces": len(traces),
                "cases": sum(
                    len(item.get("case_refs") or []) for item in traces
                ),
                "phases": sum(
                    len(item.get("phases") or []) for item in traces
                ),
            },
        }
        bundle["compiled_bundle_hash"] = hashlib.sha256(
            json.dumps(
                bundle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return bundle
