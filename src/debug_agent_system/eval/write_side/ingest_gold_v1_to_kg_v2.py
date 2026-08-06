"""Mechanically map frozen goldcase-001--010 into the active KG v2 graph.

The source annotation files remain immutable.  Their historical
``graph_ingestion=false`` flag is overridden only by an explicit invocation of
``--apply`` and the authorization string recorded in the ingestion manifest.
No semantic inference or compatibility routing is performed here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline
from debug_agent_system.eval.write_side.gold_set import verify_gold_set
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS, make_id, trim_text
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


DEFAULT_GOLD_ROOT = Path("data/annotations/goldcases/gold-v1")
DEFAULT_KG_ROOT = Path("data/kg_v2")
DEFAULT_AUDIT = Path("data/annotations/goldcases/gold-v1/kg_v2_ingestion_manifest.json")


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in V2_PRIMARY_KEYS}


def _evidence_kind(anchor: str) -> str:
    lowered = anchor.lower()
    if lowered.startswith("jira:"):
        return "jira"
    if lowered.startswith("file:"):
        return "attachment"
    if lowered.startswith("msg:"):
        return "chat_message"
    return "manual_review"


def _bundle_for_case(payload: dict[str, Any], store: JsonKGV2Store, annotation_path: Path) -> dict[str, Any]:
    case_ref = str(payload["case_id"])
    gold = payload["gold"]
    family_raw = gold["family"]
    variant_raw = gold["variant"]
    objects = _empty_objects()
    relations: list[dict[str, str]] = []

    existing_family = next(
        (
            item for item in store.objects_by_type.get("FaultFamily") or []
            if str(item.get("label") or "") == str(family_raw.get("label") or "")
        ),
        None,
    )
    family_id = str((existing_family or {}).get("family_id") or make_id("family", family_raw["label"]))
    if existing_family is None:
        objects["FaultFamily"].append({
            "family_id": family_id,
            "label": trim_text(family_raw.get("label"), 40),
            "summary": trim_text(family_raw.get("summary"), 80),
            "category": str(family_raw.get("category") or "系统与软件异常"),
            "subsystem": trim_text(family_raw.get("subsystem"), 40),
            "scenario": trim_text(family_raw.get("scenario"), 60),
            "keywords": [],
            "source_kind": "case",
            "escalation_target": "",
        })

    variant_id = make_id("variant", f"{case_ref}:{variant_raw.get('label')}")
    objects["FaultVariant"].append({
        "variant_id": variant_id,
        "family_id": family_id,
        "label": trim_text(variant_raw.get("label"), 60),
        "summary": trim_text(variant_raw.get("summary"), 180),
        "equipment_type": trim_text(variant_raw.get("equipment_type"), 60),
        "site": trim_text(variant_raw.get("site"), 60),
        "software_version": trim_text(variant_raw.get("software_version"), 60),
        "error_phase": trim_text(variant_raw.get("error_phase"), 40),
        "owner_context": trim_text(variant_raw.get("owner_context"), 80),
        "escalation_target": "",
        "keywords": [],
        "source_kind": "reviewed_gold_case",
        "gold_case_id": case_ref,
    })
    relations.append({"from": family_id, "to": variant_id, "relation": "has_variant"})

    source_case_id = f"case:{case_ref}"
    objects["SourceCase"].append({
        "case_id": source_case_id,
        "source_kind": "manual_review",
        "title": trim_text(f"{case_ref} {variant_raw.get('label')}", 80),
        "summary": trim_text(variant_raw.get("summary") or gold.get("trace", {}).get("summary"), 240),
        "source_ref": str(payload.get("source_episode_id") or case_ref),
        "approved": True,
        "gold_case_id": case_ref,
        "annotation_ref": annotation_path.as_posix(),
        "annotation_sha256": _file_hash(annotation_path),
        "ingestion_authority": "explicit_user_authorization_2026-07-21",
    })
    relations.append({"from": source_case_id, "to": variant_id, "relation": "supports"})

    evidence_by_anchor: dict[str, str] = {}
    for index, (anchor, summary) in enumerate((payload.get("evidence_anchor_map") or {}).items(), start=1):
        evidence_id = make_id("evidence", f"{case_ref}:{anchor}")
        evidence_by_anchor[str(anchor)] = evidence_id
        objects["EvidenceItem"].append({
            "evidence_id": evidence_id,
            "source_kind": _evidence_kind(str(anchor)),
            "external_id": trim_text(anchor, 120),
            "title": trim_text(anchor or f"evidence-{index}", 80),
            "summary": trim_text(summary, 500),
            "payload_ref": annotation_path.as_posix(),
            "gold_case_id": case_ref,
        })
        relations.append({"from": evidence_id, "to": source_case_id, "relation": "evidences"})

    action_by_label: dict[str, str] = {}
    action_evidence_by_label: dict[str, list[str]] = {}
    actual_action_labels = set((gold.get("trace") or {}).get("actual_action_labels") or [])
    for index, action in enumerate(gold.get("actions") or [], start=1):
        label = str(action.get("label") or "")
        action_id = make_id("action", f"{case_ref}:{index}:{label}")
        action_by_label[label] = action_id
        action_evidence = [
            evidence_by_anchor[value]
            for value in action.get("evidence_anchor_ids") or []
            if value in evidence_by_anchor
        ]
        action_evidence_by_label[label] = action_evidence
        objects["DiagnosticAction"].append({
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(label, 60),
            "summary": trim_text(action.get("summary") or label, 180),
            "action_role": str(action.get("action_role") or "inspect"),
            "step_order": int(action.get("step_order") or index),
            "destructive": bool(action.get("destructive")),
            "high_cost": bool(action.get("high_cost")),
            "source_kind": "case",
            "execution_status": "actual" if label in actual_action_labels else "recommended",
            "evidence_ids": action_evidence,
            "evidence_scope": "human_reviewed",
            "gold_case_id": case_ref,
        })

    outcome_action_labels: set[str] = set()
    for index, outcome in enumerate(gold.get("outcomes") or [], start=1):
        label = str(outcome.get("action_label") or "")
        outcome_action_labels.add(label)
        action_id = action_by_label[label]
        evidence_ids = [
            evidence_by_anchor[value]
            for value in outcome.get("evidence_anchor_ids") or []
            if value in evidence_by_anchor
        ]
        outcome_id = make_id("outcome", f"{case_ref}:{index}:{label}:{outcome.get('outcome_type')}")
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": str(outcome.get("outcome_type") or "pending_validation"),
            "outcome_origin": "human_reviewed",
            "summary": trim_text(outcome.get("summary") or label, 200),
            "source_case_id": source_case_id,
            "evidence_ids": evidence_ids,
            "high_cost": bool(outcome.get("high_cost")),
            "destructive": bool(outcome.get("destructive")),
            "root_cause_summary": trim_text(outcome.get("root_cause_summary"), 120),
            "gold_case_id": case_ref,
        })
        relations.extend([
            {"from": variant_id, "to": outcome_id, "relation": "has_outcome"},
            {"from": source_case_id, "to": outcome_id, "relation": "supports"},
            {"from": outcome_id, "to": action_id, "relation": "outcome_of"},
        ])
        relations.extend({"from": evidence_id, "to": outcome_id, "relation": "evidences"} for evidence_id in evidence_ids)

    generated_index = len(gold.get("outcomes") or [])
    action_by_label_payload = {str(item.get("label") or ""): item for item in gold.get("actions") or []}
    for label, action_id in action_by_label.items():
        if label in outcome_action_labels:
            continue
        generated_index += 1
        action = action_by_label_payload[label]
        execution_status = "actual" if label in actual_action_labels else "recommended"
        outcome_type = (
            "diagnostic_method"
            if execution_status == "actual" and str(action.get("action_role") or "") in {"inspect", "collect", "compare"}
            else "pending_validation"
        )
        evidence_ids = action_evidence_by_label[label]
        outcome_id = make_id("outcome", f"{case_ref}:generated:{generated_index}:{label}:{outcome_type}")
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": outcome_type,
            "outcome_origin": "rule_inferred",
            "summary": trim_text(
                f"{label}已记录为诊断动作。" if outcome_type == "diagnostic_method" else f"{label}尚无独立验证结果。",
                200,
            ),
            "source_case_id": source_case_id,
            "evidence_ids": evidence_ids,
            "high_cost": bool(action.get("high_cost")),
            "destructive": bool(action.get("destructive")),
            "root_cause_summary": "",
            "gold_case_id": case_ref,
            "generated_from_missing_explicit_outcome": True,
        })
        relations.extend([
            {"from": variant_id, "to": outcome_id, "relation": "has_outcome"},
            {"from": source_case_id, "to": outcome_id, "relation": "supports"},
            {"from": outcome_id, "to": action_id, "relation": "outcome_of"},
        ])
        relations.extend({"from": evidence_id, "to": outcome_id, "relation": "evidences"} for evidence_id in evidence_ids)

    for index, required in enumerate(gold.get("required_info") or [], start=1):
        evidence_ids = [
            evidence_by_anchor[value]
            for value in required.get("evidence_anchor_ids") or []
            if value in evidence_by_anchor
        ]
        required_id = make_id("required-info", f"{case_ref}:{index}:{required.get('slot')}:{required.get('question')}")
        objects["RequiredInfoSpec"].append({
            "required_info_id": required_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": str(required.get("slot") or "other"),
            "question": trim_text(required.get("question"), 100),
            "why_required": trim_text(required.get("why_required"), 160),
            "condition": trim_text(required.get("condition"), 120),
            "blocks": [str(value) for value in required.get("blocks") or []],
            "priority": str(required.get("priority") or "medium"),
            "evidence_ids": evidence_ids,
            "gold_case_id": case_ref,
        })
        relations.extend([
            {"from": variant_id, "to": required_id, "relation": "has_required_info"},
            {"from": source_case_id, "to": required_id, "relation": "supports"},
        ])
        relations.extend({"from": evidence_id, "to": required_id, "relation": "evidences"} for evidence_id in evidence_ids)

    trace_raw = gold.get("trace") or {}
    trace_id = f"trace:{case_ref}"
    recommended = [action_by_label[label] for label in trace_raw.get("recommended_action_labels") or []]
    actual = [action_by_label[label] for label in trace_raw.get("actual_action_labels") or []]
    trace_evidence = [
        evidence_by_anchor[value]
        for value in trace_raw.get("evidence_anchor_ids") or []
        if value in evidence_by_anchor
    ]
    objects["DiagnosticTrace"].append({
        "trace_id": trace_id,
        "family_id": family_id,
        "variant_id": variant_id,
        "source_case_id": source_case_id,
        "summary": trim_text(trace_raw.get("summary") or variant_raw.get("summary"), 160),
        "recommended_action_ids": recommended,
        "actual_action_ids": actual,
        "evidence_ids": trace_evidence,
        "gold_case_id": case_ref,
    })
    relations.extend([
        {"from": family_id, "to": trace_id, "relation": "has_trace"},
        {"from": variant_id, "to": trace_id, "relation": "has_trace"},
        {"from": source_case_id, "to": trace_id, "relation": "supports"},
    ])
    relations.extend({"from": trace_id, "to": action_id, "relation": "used_action"} for action_id in recommended)

    outcomes_by_action: dict[str, list[dict[str, Any]]] = {}
    for outcome in objects["ActionOutcome"]:
        outcomes_by_action.setdefault(str(outcome.get("action_id") or ""), []).append(outcome)
    previous_trace_step_id = ""
    trace_steps_for_case: list[dict[str, Any]] = []
    for ordinal, label in enumerate(trace_raw.get("recommended_action_labels") or [], start=1):
        action_id = action_by_label[label]
        execution_status = "actual" if label in actual_action_labels else "recommended"
        step_evidence = action_evidence_by_label.get(label) or trace_evidence
        trace_step_id = make_id("trace-step", f"{trace_id}:{ordinal}:{action_id}")
        objects["TraceStep"].append({
            "trace_step_id": trace_step_id,
            "trace_id": trace_id,
            "source_case_id": source_case_id,
            "action_id": action_id,
            "ordinal": ordinal,
            "execution_status": execution_status,
            "attempt_index": 1 if execution_status == "actual" else 0,
            "evidence_ids": step_evidence,
            "gold_case_id": case_ref,
        })
        trace_steps_for_case.append(objects["TraceStep"][-1])
        relations.extend([
            {"from": trace_id, "to": trace_step_id, "relation": "has_trace_step"},
            {"from": trace_step_id, "to": action_id, "relation": "step_action"},
            {"from": source_case_id, "to": trace_step_id, "relation": "supports"},
        ])
        relations.extend(
            {"from": evidence_id, "to": trace_step_id, "relation": "evidences"}
            for evidence_id in step_evidence
        )
        if previous_trace_step_id:
            relations.append({
                "from": previous_trace_step_id,
                "to": trace_step_id,
                "relation": "next_trace_step",
            })
        previous_trace_step_id = trace_step_id
        if execution_status != "actual":
            continue
        action_outcomes = outcomes_by_action.get(action_id) or []
        observation_evidence = list(dict.fromkeys(
            evidence_id
            for outcome in action_outcomes
            for evidence_id in outcome.get("evidence_ids") or []
        )) or step_evidence
        observation_id = make_id("observation", f"{trace_step_id}:attempt:1")
        objects["ExecutionObservation"].append({
            "observation_id": observation_id,
            "trace_step_id": trace_step_id,
            "source_case_id": source_case_id,
            "action_id": action_id,
            "attempt_index": 1,
            "observation_count": 1,
            "outcome_ids": [
                str(outcome.get("outcome_id") or "")
                for outcome in action_outcomes
            ],
            "outcome_types": sorted({
                str(outcome.get("outcome_type") or "")
                for outcome in action_outcomes
            }),
            "observation_window": "",
            "evidence_ids": observation_evidence,
            "gold_case_id": case_ref,
        })
        relations.append({
            "from": trace_step_id,
            "to": observation_id,
            "relation": "has_observation",
        })
        relations.extend(
            {"from": observation_id, "to": str(outcome["outcome_id"]), "relation": "observed_outcome"}
            for outcome in action_outcomes
        )
        relations.extend(
            {"from": evidence_id, "to": observation_id, "relation": "evidences"}
            for evidence_id in observation_evidence
        )

    for index, step in enumerate(trace_steps_for_case):
        action_id = str(step.get("action_id") or "")
        action_outcomes = outcomes_by_action.get(action_id) or []
        outcome_types = sorted({
            str(outcome.get("outcome_type") or "")
            for outcome in action_outcomes
            if str(outcome.get("outcome_type") or "")
        })
        next_step = (
            trace_steps_for_case[index + 1]
            if index + 1 < len(trace_steps_for_case)
            else None
        )
        if next_step is not None:
            terminal_status = "continue"
            condition = "完成当前步骤后，按人工审核的诊断顺序继续。"
        elif "verified_fix" in outcome_types:
            terminal_status = "resolved"
            condition = "人工审核证据确认问题已解决。"
        elif any(value in {"recurred", "mitigation_observed"} for value in outcome_types):
            terminal_status = "monitoring"
            condition = "当前结果仍需继续观察是否复发。"
        else:
            terminal_status = "unresolved"
            condition = "当前步骤没有已审核的最终解决结论。"
        branch_id = make_id(
            "branch",
            f"{trace_id}:{step['trace_step_id']}:"
            f"{str(next_step.get('trace_step_id') or '') if isinstance(next_step, dict) else ''}:reviewed",
        )
        step_evidence = list(step.get("evidence_ids") or []) or trace_evidence
        objects["BranchRule"].append({
            "branch_rule_id": branch_id,
            "trace_id": trace_id,
            "source_case_id": source_case_id,
            "from_trace_step_id": str(step.get("trace_step_id") or ""),
            "to_trace_step_id": (
                str(next_step.get("trace_step_id") or "")
                if isinstance(next_step, dict)
                else ""
            ),
            "trigger_outcome_types": outcome_types or ["pending_validation"],
            "condition": trim_text(condition, 120),
            "branch_kind": "reviewed_recommendation",
            "terminal_status": terminal_status,
            "priority": 1,
            "evidence_ids": step_evidence,
            "gold_case_id": case_ref,
        })
        relations.extend([
            {"from": trace_id, "to": branch_id, "relation": "has_branch_rule"},
            {
                "from": branch_id,
                "to": str(step.get("trace_step_id") or ""),
                "relation": "branch_from",
            },
            {"from": source_case_id, "to": branch_id, "relation": "supports"},
        ])
        if isinstance(next_step, dict):
            relations.append({
                "from": branch_id,
                "to": str(next_step.get("trace_step_id") or ""),
                "relation": "branch_to",
            })
        relations.extend(
            {"from": evidence_id, "to": branch_id, "relation": "evidences"}
            for evidence_id in step_evidence
        )
    return {"objects": objects, "relations": relations}


def build_bundle(gold_root: str | Path, kg_root: str | Path) -> dict[str, Any]:
    gold_root = Path(gold_root)
    integrity = verify_gold_set(gold_root)
    store = JsonKGV2Store(kg_root)
    objects = _empty_objects()
    relations: list[dict[str, str]] = []
    case_rows = []
    for path in sorted(gold_root.glob("goldcase-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_bundle = _bundle_for_case(payload, store, path)
        for key, values in case_bundle["objects"].items():
            objects[key].extend(values)
        relations.extend(case_bundle["relations"])
        case_rows.append({
            "case_id": payload["case_id"],
            "annotation": path.as_posix(),
            "annotation_sha256": _file_hash(path),
            "family": payload["gold"]["family"]["label"],
            "variant": payload["gold"]["variant"]["label"],
        })
    return {
        "objects": objects,
        "relations": relations,
        "report": {
            "gold_set": integrity,
            "cases": case_rows,
            "object_counts": {key: len(value) for key, value in objects.items()},
            "relation_count": len(relations),
        },
    }


def _prune_superseded_gold_sources(
    store: JsonKGV2Store,
    source_episode_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    old_case_ids = {
        str(item.get("case_id") or "")
        for item in store.objects_by_type.get("SourceCase") or []
        if isinstance(item, dict)
        and str(item.get("source_ref") or "") in source_episode_ids
        and not str(item.get("case_id") or "").startswith("case:goldcase-")
    }
    variant_ids = {
        str(item.get("to") or "")
        for item in store.relations
        if isinstance(item, dict)
        and str(item.get("from") or "") in old_case_ids
        and str(item.get("relation") or "") == "supports"
        and str(item.get("to") or "").startswith("variant:")
    }
    trace_ids = {
        str(item.get("trace_id") or "")
        for item in store.objects_by_type.get("DiagnosticTrace") or []
        if isinstance(item, dict) and str(item.get("source_case_id") or "") in old_case_ids
    }
    trace_step_ids = {
        str(item.get("trace_step_id") or "")
        for item in store.objects_by_type.get("TraceStep") or []
        if isinstance(item, dict)
        and (
            str(item.get("source_case_id") or "") in old_case_ids
            or str(item.get("trace_id") or "") in trace_ids
        )
    }
    observation_ids = {
        str(item.get("observation_id") or "")
        for item in store.objects_by_type.get("ExecutionObservation") or []
        if isinstance(item, dict)
        and (
            str(item.get("source_case_id") or "") in old_case_ids
            or str(item.get("trace_step_id") or "") in trace_step_ids
        )
    }
    branch_rule_ids = {
        str(item.get("branch_rule_id") or "")
        for item in store.objects_by_type.get("BranchRule") or []
        if isinstance(item, dict)
        and (
            str(item.get("source_case_id") or "") in old_case_ids
            or str(item.get("trace_id") or "") in trace_ids
            or str(item.get("from_trace_step_id") or "") in trace_step_ids
        )
    }
    outcome_ids = {
        str(item.get("outcome_id") or "")
        for item in store.objects_by_type.get("ActionOutcome") or []
        if isinstance(item, dict) and str(item.get("source_case_id") or "") in old_case_ids
    }
    action_ids = {
        str(item.get("action_id") or "")
        for item in store.objects_by_type.get("DiagnosticAction") or []
        if isinstance(item, dict) and str(item.get("variant_id") or "") in variant_ids
    }
    required_ids = {
        str(item.get("required_info_id") or "")
        for item in store.objects_by_type.get("RequiredInfoSpec") or []
        if isinstance(item, dict) and str(item.get("variant_id") or "") in variant_ids
    }
    evidence_ids = {
        str(item.get("from") or "")
        for item in store.relations
        if isinstance(item, dict)
        and str(item.get("to") or "") in old_case_ids
        and str(item.get("relation") or "") == "evidences"
    }
    removed_ids = (
        old_case_ids | variant_ids | trace_ids | trace_step_ids | observation_ids
        | branch_rule_ids | outcome_ids | action_ids | required_ids | evidence_ids
    )
    # Terminology is a derived layer.  Pruning a superseded case subgraph must
    # also remove concepts/senses whose canonical or source object disappears;
    # otherwise projected validation sees dangling primary_concept targets.
    concept_ids = {
        str(item.get("concept_id") or "")
        for item in store.objects_by_type.get("DebugConcept") or []
        if isinstance(item, dict)
        and (
            str(item.get("canonical_target_id") or "") in removed_ids
            or any(
                str(value) in removed_ids
                for value in item.get("source_object_ids") or []
            )
        )
    }
    sense_ids = {
        str(item.get("sense_id") or "")
        for item in store.objects_by_type.get("TermSense") or []
        if isinstance(item, dict)
        and (
            str(item.get("concept_id") or "") in concept_ids
            or str(item.get("source_object_id") or "") in removed_ids
        )
    }
    removed_ids |= concept_ids | sense_ids
    objects = {key: [] for key in store.objects_by_type}
    for obj_type, rows in store.objects_by_type.items():
        pk = V2_PRIMARY_KEYS[obj_type]
        objects[obj_type] = [
            dict(item) for item in rows
            if isinstance(item, dict) and str(item.get(pk) or "") not in removed_ids
        ]
    remaining_observations = {
        str(item.get("observation_id") or ""): item
        for item in objects.get("ExecutionObservation") or []
        if isinstance(item, dict) and str(item.get("observation_id") or "")
    }
    remaining_branches = {
        str(item.get("branch_rule_id") or "")
        for item in objects.get("BranchRule") or []
        if isinstance(item, dict) and str(item.get("branch_rule_id") or "")
    }
    for policy in objects.get("DecisionPolicy") or []:
        if not isinstance(policy, dict):
            continue
        retained_observation_ids = [
            str(value) for value in policy.get("observation_ids") or []
            if str(value) in remaining_observations
        ]
        if "observation_ids" in policy:
            policy["observation_ids"] = retained_observation_ids
            policy["actual_execution_count"] = sum(
                int(remaining_observations[value].get("observation_count") or 0)
                for value in retained_observation_ids
            )
        if "branch_rule_ids" in policy:
            policy["branch_rule_ids"] = [
                str(value) for value in policy.get("branch_rule_ids") or []
                if str(value) in remaining_branches
            ]
        for key in (
            "source_trace_ids", "source_outcome_ids", "ordered_action_ids",
            "ineffective_action_ids", "high_cost_action_ids",
        ):
            if key in policy:
                policy[key] = [str(value) for value in policy.get(key) or [] if str(value) not in removed_ids]
        for row in policy.get("execution_stats") or []:
            if not isinstance(row, dict):
                continue
            action_id = str(row.get("action_id") or "")
            action_observations = [
                value for value in remaining_observations.values()
                if str(value.get("action_id") or "") == action_id
            ]
            row["actual_execution_count"] = sum(int(value.get("observation_count") or 0) for value in action_observations)
            row["source_case_count"] = len({
                str(value.get("source_case_id") or "") for value in action_observations
                if str(value.get("source_case_id") or "")
            })
    relations = [
        dict(item) for item in store.relations
        if isinstance(item, dict)
        and str(item.get("from") or "") not in removed_ids
        and str(item.get("to") or "") not in removed_ids
    ]
    return objects, relations, {
        "source_case_ids": sorted(old_case_ids),
        "variant_ids": sorted(variant_ids),
        "trace_step_ids": sorted(trace_step_ids),
        "observation_ids": sorted(observation_ids),
        "branch_rule_ids": sorted(branch_rule_ids),
        "removed_object_count": len(removed_ids),
        "removed_relation_count": len(store.relations) - len(relations),
    }


def run(
    *,
    gold_root: str | Path,
    kg_root: str | Path,
    audit_out: str | Path,
    apply: bool,
    authorization: str,
) -> dict[str, Any]:
    if apply and not authorization.strip():
        raise ValueError("explicit_authorization_required")
    gold_root = Path(gold_root)
    kg_root = Path(kg_root)
    audit_out = Path(audit_out)
    store = JsonKGV2Store(kg_root)
    before = {
        "object_counts": {key: len(value) for key, value in store.objects_by_type.items()},
        "relation_count": len(store.relations),
        "graph_sha256": _canonical_hash({"objects": store.objects_by_type, "relations": store.relations}),
    }
    bundle = build_bundle(gold_root, kg_root)
    source_episode_ids = {
        str(json.loads(path.read_text(encoding="utf-8")).get("source_episode_id") or "")
        for path in gold_root.glob("goldcase-*.json")
    }
    merged_objects, merged_relations, pruned = _prune_superseded_gold_sources(store, source_episode_ids)
    for key, values in bundle["objects"].items():
        pk = V2_PRIMARY_KEYS[key]
        index = {str(item.get(pk)): item for item in merged_objects[key]}
        for item in values:
            obj_id = str(item.get(pk) or "")
            if obj_id in index:
                index[obj_id].update({k: v for k, v in item.items() if v not in (None, "", [])})
            else:
                merged_objects[key].append(dict(item))
                index[obj_id] = merged_objects[key][-1]
    merged_relations = [*merged_relations, *bundle["relations"]]
    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in merged_relations:
        if isinstance(item, dict):
            relation_index[(str(item.get("from")), str(item.get("to")), str(item.get("relation")))] = item
    merged_relations = list(relation_index.values())
    schema_root = kg_root / "schema"
    issues = validate_graph(merged_objects, merged_relations, schema_root=schema_root)
    if issues:
        raise ValueError("gold_v1_kg_v2_schema_invalid:" + ";".join(issues[:40]))

    result: dict[str, Any] = {"status": "dry_run_valid", "object_counts": {}, "relation_count": 0}
    materialized: dict[str, Any] = {}
    if apply:
        result = store.replace_graph(merged_objects, merged_relations, validate=True)
        if result.get("status") != "replaced":
            raise ValueError("gold_v1_kg_v2_merge_failed:" + json.dumps(result, ensure_ascii=False))
        pipeline = WriteSideV2Pipeline(kg_root)
        materialized = pipeline.materialize_execution()
        store = JsonKGV2Store(kg_root)

    effective_objects = store.objects_by_type if apply else merged_objects
    effective_relations = store.relations if apply else merged_relations
    after = {
        "mode": "applied" if apply else "projected",
        "object_counts": {key: len(value) for key, value in effective_objects.items()},
        "relation_count": len(effective_relations),
        "graph_sha256": _canonical_hash({"objects": effective_objects, "relations": effective_relations}),
    }
    audit = {
        "schema_version": "debug_agent_system.gold_v1_kg_v2_ingestion.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": apply,
        "authorization": authorization,
        "source_gold_root": gold_root.as_posix(),
        "source_manifest_sha256": _file_hash(gold_root / "gold-v1.manifest.json"),
        "source_policy_override": "Frozen evaluation files remain unchanged; explicit user authorization permits graph ingestion of 001-010 only.",
        "excluded_case_range": "goldcase-011..goldcase-015",
        "bundle": bundle["report"],
        "superseded_subgraph_cleanup": pruned,
        "before": before,
        "after": after,
        "write_result": result,
        "materialized": materialized,
    }
    if apply:
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest-gold-v1-to-kg-v2")
    parser.add_argument("--gold-root", default=str(DEFAULT_GOLD_ROOT))
    parser.add_argument("--kg-root", default=str(DEFAULT_KG_ROOT))
    parser.add_argument("--audit-out", default=str(DEFAULT_AUDIT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--authorization", default="")
    args = parser.parse_args(argv)
    report = run(
        gold_root=args.gold_root,
        kg_root=args.kg_root,
        audit_out=args.audit_out,
        apply=args.apply,
        authorization=args.authorization,
    )
    print(json.dumps({
        "status": report["write_result"]["status"],
        "applied": report["applied"],
        "case_count": len(report["bundle"]["cases"]),
        "bundle_object_counts": report["bundle"]["object_counts"],
        "before": report["before"],
        "after": report["after"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
