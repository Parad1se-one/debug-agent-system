"""Turn a W6 expert correction into a fresh, auditable KG v2 candidate.

The correction never mutates the original candidate.  A provenance rebound
(the reviewed issue belongs to another episode) receives a new review/dedupe
identity so W5 cannot accidentally overwrite the original case boundary.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from debug_agent_system.knowledge_v2.builders import infer_action_role
from debug_agent_system.knowledge_v2.contracts import FAMILY_SUBSYSTEM_EXPECTED, make_id, trim_text
from debug_agent_system.knowledge_v2.validator import validate_graph


def build_expert_corrected_candidate(
    review_item: dict[str, Any],
    correction: dict[str, Any],
) -> dict[str, Any]:
    typed = review_item.get("typed_candidate") if isinstance(review_item.get("typed_candidate"), dict) else {}
    original_payload = typed.get("payload") if isinstance(typed.get("payload"), dict) else {}
    original_objects = original_payload.get("objects") if isinstance(original_payload.get("objects"), dict) else {}
    episode = original_payload.get("episode") if isinstance(original_payload.get("episode"), dict) else {}

    family_label = str(correction.get("family") or "").strip()
    variant_label = str(correction.get("variant") or "").strip()
    if not family_label or not variant_label:
        raise ValueError("expert_correction_missing_family_or_variant")

    rebound = str(correction.get("disposition") or "") == "do_not_apply_original_create_rebound_candidate"
    original_family = _first(original_objects, "FaultFamily")
    family_id = (
        str(original_family.get("family_id") or "")
        if str(original_family.get("label") or "") == family_label
        else make_id("family", family_label)
    )
    variant_id = make_id("variant", f"{family_id}:{variant_label}")
    source_ref = str(correction.get("source_episode_id_original") or episode.get("episode_id") or "")
    if rebound:
        jira_ref = next(
            (str(item.get("external_id") or "") for item in correction.get("evidence_additions") or [] if str(item.get("kind") or "") == "jira"),
            "rebound",
        )
        source_ref = f"rebound:{source_ref}:{jira_ref}"
    case_id = make_id("case", f"expert-review:{source_ref}:{variant_label}")

    evidence_items, evidence_by_ref, source_messages = _evidence_items(correction, episode, case_id)
    all_evidence_ids = [item["evidence_id"] for item in evidence_items]
    objects = _empty_objects()
    objects["FaultFamily"] = [{
        "family_id": family_id,
        "label": trim_text(family_label, 40),
        "summary": trim_text(
            original_family.get("summary") if str(original_family.get("label") or "") == family_label else f"{family_label}相关故障。",
            80,
        ),
        "category": str(original_family.get("category") or "系统与软件异常"),
        "subsystem": trim_text(
            FAMILY_SUBSYSTEM_EXPECTED.get(family_label) or original_family.get("subsystem") or "",
            40,
        ),
        "scenario": trim_text(variant_label, 60),
        "keywords": list(original_family.get("keywords") or []),
        "source_kind": "case",
        "escalation_target": str(original_family.get("escalation_target") or ""),
    }]
    objects["FaultVariant"] = [{
        "variant_id": variant_id,
        "family_id": family_id,
        "label": trim_text(variant_label, 60),
        "summary": trim_text(variant_label, 180),
        "equipment_type": trim_text(correction.get("equipment_type"), 60),
        "site": trim_text(correction.get("site"), 60),
        "software_version": trim_text(correction.get("software_version"), 60),
        "error_phase": trim_text(correction.get("error_phase"), 40),
        "owner_context": trim_text(correction.get("owner_context") or source_ref, 80),
        "escalation_target": str(original_family.get("escalation_target") or ""),
        "keywords": [],
    }]
    review_basis = correction.get("review_basis") if isinstance(correction.get("review_basis"), dict) else {}
    source_case = {
        "case_id": case_id,
        "source_kind": "manual_review",
        "title": trim_text(variant_label, 80),
        "summary": trim_text(_case_summary(correction), 240),
        "source_ref": trim_text(source_ref, 200),
        "approved": False,
        "trust_tier": str(review_basis.get("trust_tier") or "human_reviewed"),
        "review_id": trim_text(correction.get("review_id"), 200),
        "ingest_run_id": trim_text(review_basis.get("ingest_run_id"), 120),
    }
    for key, limit in (("annotation_set_id", 80), ("annotation_case_id", 80), ("annotation_sha256", 64)):
        if str(review_basis.get(key) or ""):
            source_case[key] = trim_text(review_basis.get(key), limit)
    objects["SourceCase"] = [source_case]
    objects["EvidenceItem"] = evidence_items

    actual_orders = {
        int(value) for value in correction.get("actual_action_orders") or []
        if str(value).isdigit()
    }
    if not actual_orders:
        actual_orders = {
            int(item.get("action_order") or 0)
            for item in correction.get("outcomes") or []
            if isinstance(item, dict) and int(item.get("action_order") or 0) > 0
        }
    action_ids: list[str] = []
    action_id_by_order: dict[int, str] = {}
    action_payload_by_order: dict[int, dict[str, Any]] = {}
    for raw in sorted(correction.get("actions") or [], key=lambda item: int(item.get("order") or 9999)):
        order = int(raw.get("order") or len(action_ids) + 1)
        label = str(raw.get("label") or "").strip()
        action_id = make_id("action", f"{variant_id}:{order}:{label}")
        action_ids.append(action_id)
        action_id_by_order[order] = action_id
        action_payload_by_order[order] = raw
        evidence_refs = [str(value) for value in raw.get("evidence_refs") or []]
        action_evidence_ids = [evidence_by_ref[ref] for ref in evidence_refs if ref in evidence_by_ref] or all_evidence_ids[:1]
        action_object = {
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(label, 60),
            "summary": trim_text(raw.get("summary") or label, 180),
            "action_role": str(raw.get("role") or infer_action_role(label)),
            "step_order": order,
            "destructive": bool(raw.get("destructive")),
            "high_cost": bool(raw.get("high_cost")),
            "source_kind": "case",
            "execution_status": "actual" if order in actual_orders else "recommended",
        }
        action_object["evidence_ids"] = action_evidence_ids
        objects["DiagnosticAction"].append(action_object)

    for index, raw in enumerate(correction.get("outcomes") or [], start=1):
        order = int(raw.get("action_order") or 0)
        if order < 1 or order > len(action_ids):
            raise ValueError(f"expert_correction_invalid_outcome_action_order:{order}")
        refs = [str(value) for value in raw.get("evidence_refs") or [] if str(value)]
        evidence_ids = [evidence_by_ref[value] for value in refs if value in evidence_by_ref]
        if not evidence_ids:
            evidence_ids = all_evidence_ids[:1]
        outcome_id = make_id("outcome", f"{case_id}:{index}:{raw.get('outcome_type')}:{raw.get('summary')}")
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_ids[order - 1],
            "outcome_type": str(raw.get("outcome_type") or "pending_validation"),
            "summary": trim_text(raw.get("summary") or "", 200),
            "source_case_id": case_id,
            "evidence_ids": evidence_ids,
            "high_cost": bool(raw.get("high_cost")),
            "destructive": bool(raw.get("destructive")),
            "root_cause_summary": trim_text(raw.get("root_cause_summary"), 120),
        })

    outcome_action_ids = {
        str(item.get("action_id") or "")
        for item in objects["ActionOutcome"]
        if str(item.get("action_id") or "")
    }
    generated_index = len(objects["ActionOutcome"])
    for order in sorted(action_id_by_order):
        action_id = action_id_by_order.get(order)
        if not action_id or action_id in outcome_action_ids:
            continue
        generated_index += 1
        action_payload = action_payload_by_order.get(order) or {}
        action_role = str(action_payload.get("role") or infer_action_role(str(action_payload.get("label") or "")))
        evidence_ids = [
            evidence_by_ref[ref]
            for ref in [str(value) for value in action_payload.get("evidence_refs") or []]
            if ref in evidence_by_ref
        ] or all_evidence_ids[:1]
        outcome_type = (
            "diagnostic_method"
            if order in actual_orders and action_role in {"inspect", "collect", "compare"}
            else "pending_validation"
        )
        outcome_id = make_id("outcome", f"{case_id}:generated:{generated_index}:{order}:{outcome_type}")
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": outcome_type,
            "summary": trim_text(
                f"{action_payload.get('label') or '该动作'}已记录为诊断动作。"
                if outcome_type == "diagnostic_method"
                else f"{action_payload.get('label') or '该动作'}尚无独立验证结果。",
                200,
            ),
            "source_case_id": case_id,
            "evidence_ids": evidence_ids,
            "high_cost": bool(action_payload.get("high_cost")),
            "destructive": bool(action_payload.get("destructive")),
            "root_cause_summary": "",
            "generated_from_missing_explicit_outcome": True,
        })

    for index, raw in enumerate(correction.get("required_info") or [], start=1):
        required_id = make_id("required-info", f"{variant_id}:{raw.get('slot')}:{raw.get('question')}")
        refs = [str(value) for value in raw.get("evidence_refs") or [] if str(value)]
        objects["RequiredInfoSpec"].append({
            "required_info_id": required_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": str(raw.get("slot") or "other"),
            "question": trim_text(raw.get("question") or "", 100),
            "why_required": trim_text(raw.get("why_required") or "", 160),
            "condition": trim_text(raw.get("condition") or "", 120),
            "blocks": [trim_text(raw.get("blocks") or action_ids[0] if action_ids else variant_label, 80)],
            "priority": str(raw.get("priority") or "medium"),
            "evidence_ids": [evidence_by_ref[value] for value in refs if value in evidence_by_ref] or all_evidence_ids,
        })

    trace_id = make_id("trace", case_id)
    actual_action_ids = [
        action_id_by_order[order]
        for order in sorted(actual_orders)
        if order in action_id_by_order
    ]
    objects["DiagnosticTrace"] = [{
        "trace_id": trace_id,
        "family_id": family_id,
        "variant_id": variant_id,
        "source_case_id": case_id,
        "summary": trim_text(_case_summary(correction), 160),
        "recommended_action_ids": action_ids,
        "actual_action_ids": actual_action_ids,
        "evidence_ids": all_evidence_ids,
    }]

    outcomes_by_action: dict[str, list[dict[str, Any]]] = {}
    for outcome in objects["ActionOutcome"]:
        outcomes_by_action.setdefault(str(outcome.get("action_id") or ""), []).append(outcome)
    trace_step_id_by_order: dict[int, str] = {}
    for ordinal, action_id in enumerate(action_ids, start=1):
        action = action_payload_by_order.get(ordinal) or {}
        refs = [str(value) for value in action.get("evidence_refs") or []]
        evidence_ids_for_step = [evidence_by_ref[ref] for ref in refs if ref in evidence_by_ref] or all_evidence_ids[:1]
        execution_status = "actual" if ordinal in actual_orders else "recommended"
        trace_step_id = make_id("trace-step", f"{trace_id}:{ordinal}:{action_id}")
        trace_step_id_by_order[ordinal] = trace_step_id
        objects["TraceStep"].append({
            "trace_step_id": trace_step_id,
            "trace_id": trace_id,
            "source_case_id": case_id,
            "action_id": action_id,
            "ordinal": ordinal,
            "execution_status": execution_status,
            "attempt_index": 1 if execution_status == "actual" else 0,
            "evidence_ids": evidence_ids_for_step,
        })
        if execution_status == "actual":
            action_outcomes = outcomes_by_action.get(action_id) or []
            observation_evidence_ids = list(dict.fromkeys(
                evidence_id
                for outcome in action_outcomes
                for evidence_id in outcome.get("evidence_ids") or []
            )) or evidence_ids_for_step
            objects["ExecutionObservation"].append({
                "observation_id": make_id("observation", f"{trace_step_id}:attempt:1"),
                "trace_step_id": trace_step_id,
                "source_case_id": case_id,
                "action_id": action_id,
                "attempt_index": 1,
                "observation_count": 1,
                "outcome_ids": [str(outcome.get("outcome_id") or "") for outcome in action_outcomes],
                "outcome_types": sorted({str(outcome.get("outcome_type") or "") for outcome in action_outcomes}),
                "evidence_ids": observation_evidence_ids,
            })

    for ordinal, action_id in enumerate(action_ids, start=1):
        trace_step_id = trace_step_id_by_order[ordinal]
        next_step_id = trace_step_id_by_order.get(ordinal + 1, "")
        step = objects["TraceStep"][ordinal - 1]
        action_outcomes = outcomes_by_action.get(action_id) or []
        for priority, outcome_type in enumerate(sorted({str(item.get("outcome_type") or "") for item in action_outcomes}), start=1):
            matching_outcomes = [item for item in action_outcomes if str(item.get("outcome_type") or "") == outcome_type]
            branch_target, terminal_status = _branch_destination(outcome_type, next_step_id)
            branch_evidence_ids = list(dict.fromkeys(
                evidence_id
                for outcome in matching_outcomes
                for evidence_id in outcome.get("evidence_ids") or []
            )) or list(step.get("evidence_ids") or [])
            objects["BranchRule"].append({
                "branch_rule_id": make_id("branch-rule", f"{trace_step_id}:{outcome_type}:{branch_target}:{terminal_status}"),
                "trace_id": trace_id,
                "source_case_id": case_id,
                "from_trace_step_id": trace_step_id,
                "to_trace_step_id": branch_target,
                "trigger_outcome_types": [outcome_type],
                "condition": trim_text(f"outcome_type={outcome_type}", 120),
                "branch_kind": "observed_transition" if str(step.get("execution_status") or "") == "actual" else "reviewed_recommendation",
                "terminal_status": terminal_status,
                "priority": priority,
                "evidence_ids": branch_evidence_ids,
            })

    relations = [
        {"from": family_id, "to": variant_id, "relation": "has_variant"},
        {"from": case_id, "to": variant_id, "relation": "supports"},
        {"from": family_id, "to": trace_id, "relation": "has_trace"},
        {"from": variant_id, "to": trace_id, "relation": "has_trace"},
        {"from": case_id, "to": trace_id, "relation": "supports"},
    ]
    for evidence in evidence_items:
        relations.append({"from": evidence["evidence_id"], "to": case_id, "relation": "evidences"})
    for action_id in action_ids:
        relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})
    for ordinal, step in enumerate(objects["TraceStep"], start=1):
        step_id = str(step["trace_step_id"])
        relations.extend([
            {"from": trace_id, "to": step_id, "relation": "has_trace_step"},
            {"from": step_id, "to": str(step["action_id"]), "relation": "step_action"},
            {"from": case_id, "to": step_id, "relation": "supports"},
            *({"from": evidence_id, "to": step_id, "relation": "evidences"} for evidence_id in step["evidence_ids"]),
        ])
        next_step_id = trace_step_id_by_order.get(ordinal + 1)
        if next_step_id:
            relations.append({"from": step_id, "to": next_step_id, "relation": "next_trace_step"})
    for observation in objects["ExecutionObservation"]:
        observation_id = str(observation["observation_id"])
        relations.extend([
            {"from": str(observation["trace_step_id"]), "to": observation_id, "relation": "has_observation"},
            {"from": case_id, "to": observation_id, "relation": "supports"},
            *({"from": observation_id, "to": outcome_id, "relation": "observed_outcome"} for outcome_id in observation["outcome_ids"]),
            *({"from": evidence_id, "to": observation_id, "relation": "evidences"} for evidence_id in observation["evidence_ids"]),
        ])
    for branch in objects["BranchRule"]:
        branch_id = str(branch["branch_rule_id"])
        relations.extend([
            {"from": trace_id, "to": branch_id, "relation": "has_branch_rule"},
            {"from": branch_id, "to": str(branch["from_trace_step_id"]), "relation": "branch_from"},
            {"from": case_id, "to": branch_id, "relation": "supports"},
            *({"from": evidence_id, "to": branch_id, "relation": "evidences"} for evidence_id in branch["evidence_ids"]),
        ])
        if str(branch.get("to_trace_step_id") or ""):
            relations.append({"from": branch_id, "to": str(branch["to_trace_step_id"]), "relation": "branch_to"})
    for outcome in objects["ActionOutcome"]:
        relations.extend([
            {"from": variant_id, "to": outcome["outcome_id"], "relation": "has_outcome"},
            {"from": case_id, "to": outcome["outcome_id"], "relation": "supports"},
            {"from": outcome["outcome_id"], "to": outcome["action_id"], "relation": "outcome_of"},
            *({"from": evidence_id, "to": outcome["outcome_id"], "relation": "evidences"} for evidence_id in outcome["evidence_ids"]),
        ])
    for required in objects["RequiredInfoSpec"]:
        relations.extend([
            {"from": variant_id, "to": required["required_info_id"], "relation": "has_required_info"},
            {"from": case_id, "to": required["required_info_id"], "relation": "supports"},
        ])

    issues = validate_graph(objects, relations)
    identity = f"expert-rebound:{review_item.get('dedupe_key')}" if rebound else f"expert-corrected:{review_item.get('dedupe_key')}"
    candidate_id = f"candidate:{identity}"
    return {
        "candidate_id": candidate_id,
        "dedupe_key": identity,
        "family_id": family_id,
        "variant_id": variant_id,
        "objects": objects,
        "relations": relations,
        "schema_valid": not issues,
        "schema_issues": issues,
        "source_text": " ".join(item["text"] for item in source_messages),
        "source_message_ids": [item["message_id"] for item in source_messages],
        "source_messages": source_messages,
        "expert_correction": copy.deepcopy(correction),
        "supersedes_review_id": str(correction.get("review_id") or ""),
        "provenance_rebound": rebound,
    }


def _evidence_items(
    correction: dict[str, Any], episode: dict[str, Any], case_id: str
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, str]]]:
    additions = [item for item in correction.get("evidence_additions") or [] if isinstance(item, dict)]
    message_text = _episode_message_text(episode)
    rows: list[dict[str, Any]] = []
    by_ref: dict[str, str] = {}
    source_messages: list[dict[str, str]] = []
    for index, raw in enumerate(additions, start=1):
        external_id = str(raw.get("external_id") or f"expert-evidence-{index}")
        kind = str(raw.get("kind") or "manual_review")
        source_kind = "jira" if kind.startswith("jira") else "chat_message" if kind == "chat_message" else "manual_review"
        summary = str(raw.get("summary") or message_text.get(external_id) or external_id)
        # ``make_id`` removes CJK characters before truncation.  Long case IDs
        # therefore made distinct Chinese anchors such as ``msg:恢复`` and
        # ``msg:复验`` collapse to the same ID.  Bind an ASCII digest of the
        # original anchor before canonicalization so reviewed evidence remains
        # one-to-one and schema-valid.
        anchor_digest = hashlib.sha1(external_id.encode("utf-8")).hexdigest()[:12]
        evidence_id = make_id("evidence", f"{case_id}:{external_id}:{anchor_digest}")
        by_ref[external_id] = evidence_id
        rows.append({
            "evidence_id": evidence_id,
            "source_kind": source_kind,
            "external_id": trim_text(external_id, 120),
            "title": trim_text(external_id, 80),
            "summary": trim_text(summary, 500),
            "payload_ref": trim_text(raw.get("source_path") or source_ref_for_evidence(episode), 200),
        })
        source_messages.append({"message_id": external_id, "role": "expert_evidence", "text": summary})
    for outcome in correction.get("outcomes") or []:
        for ref in outcome.get("evidence_refs") or []:
            ref = str(ref)
            if ref in by_ref:
                continue
            summary = str(outcome.get("summary") or ref)
            anchor_digest = hashlib.sha1(ref.encode("utf-8")).hexdigest()[:12]
            evidence_id = make_id("evidence", f"{case_id}:{ref}:{anchor_digest}")
            by_ref[ref] = evidence_id
            rows.append({
                "evidence_id": evidence_id,
                "source_kind": "jira" if ref.startswith("TEST-") else "chat_message",
                "external_id": trim_text(ref, 120),
                "title": trim_text(ref, 80),
                "summary": trim_text(summary, 500),
                "payload_ref": trim_text(source_ref_for_evidence(episode), 200),
            })
            source_messages.append({"message_id": ref, "role": "expert_evidence", "text": summary})
    return rows, by_ref, source_messages


def _episode_message_text(episode: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_evidence_messages", "case_context_messages"):
        for item in episode.get(key) or []:
            if isinstance(item, dict) and str(item.get("message_id") or ""):
                out[str(item["message_id"])] = str(item.get("text") or item.get("content_summary") or "")
    return out


def _case_summary(correction: dict[str, Any]) -> str:
    outcomes = [str(item.get("summary") or "") for item in correction.get("outcomes") or [] if isinstance(item, dict)]
    return "；".join([str(correction.get("variant") or ""), *outcomes])


def _branch_destination(outcome_type: str, next_step_id: str) -> tuple[str, str]:
    if outcome_type == "verified_fix":
        return "", "resolved"
    if next_step_id:
        return next_step_id, "continue"
    if outcome_type in {"partial_temporary", "mitigation_observed"}:
        return "", "monitoring"
    return "", "unresolved"


def source_ref_for_evidence(episode: dict[str, Any]) -> str:
    return str(episode.get("episode_id") or episode.get("source_thread_id") or "")


def _first(objects: dict[str, Any], kind: str) -> dict[str, Any]:
    return next((item for item in objects.get(kind) or [] if isinstance(item, dict)), {})


def _empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {
        "KnowledgeDocument": [], "KnowledgeSection": [], "ProcedureStep": [],
        "FaultFamily": [], "FaultVariant": [], "DiagnosticAction": [],
        "ActionOutcome": [], "RequiredInfoSpec": [], "DiagnosticTrace": [],
        "TraceStep": [], "ExecutionObservation": [], "BranchRule": [],
        "DecisionPolicy": [], "EvidenceItem": [], "SourceCase": [],
    }


__all__ = ["build_expert_corrected_candidate"]
