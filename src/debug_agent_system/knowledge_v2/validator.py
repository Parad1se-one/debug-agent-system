"""Dependency-free validator for KG v2 graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.core.paths import project_root
from debug_agent_system.knowledge_v2.contracts import (
    ACTION_EVIDENCE_SCOPES,
    BRANCH_KINDS,
    BRANCH_TERMINAL_STATUSES,
    INTERNAL_REQUIRED_INFO_SLOTS,
    OUTCOME_ORIGINS,
    OUTCOME_TYPES,
    TRACE_EXECUTION_STATUSES,
    V2_PRIMARY_KEYS,
)

DEFAULT_SCHEMA_ROOT = project_root(__file__) / "data" / "kg_v2" / "schema"


def load_schema(schema_root: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(schema_root) if schema_root is not None else DEFAULT_SCHEMA_ROOT
    objects = json.loads((root / "object-types.json").read_text(encoding="utf-8"))
    links = json.loads((root / "link-types.json").read_text(encoding="utf-8"))
    return objects, links


def validate_graph(
    objects_by_type: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, Any]],
    *,
    schema_root: str | Path | None = None,
) -> list[str]:
    object_schema, link_schema = load_schema(schema_root)
    object_types = object_schema.get("object_types") or {}
    link_types = link_schema.get("link_types") or {}
    issues: list[str] = []
    seen_ids: set[str] = set()
    obj_type_by_id: dict[str, str] = {}
    obj_by_id: dict[str, dict[str, Any]] = {}
    for obj_type, items in objects_by_type.items():
        schema = object_types.get(obj_type)
        if schema is None:
            # JsonKGV2Store exposes every object collection known by the
            # runtime.  Older, isolated graph drafts may intentionally use a
            # narrower schema; an empty newer collection is not graph data and
            # therefore must not invalidate that draft.
            if isinstance(items, list) and not items:
                continue
            issues.append(f"unknown_object_type:{obj_type}")
            continue
        if not isinstance(items, list):
            issues.append(f"object_type_not_list:{obj_type}")
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                issues.append(f"object_not_dict:{obj_type}:{idx}")
                continue
            pk = str(schema.get("pk") or V2_PRIMARY_KEYS.get(obj_type) or "id")
            obj_id = str(item.get(pk) or "")
            if not obj_id:
                issues.append(f"missing_pk:{obj_type}.{pk}")
                continue
            if obj_id in seen_ids:
                issues.append(f"duplicate_object_id:{obj_id}")
                continue
            seen_ids.add(obj_id)
            obj_type_by_id[obj_id] = obj_type
            obj_by_id[obj_id] = item
            issues.extend(_validate_object(obj_type, item, schema))
    for idx, relation in enumerate(relations):
        if not isinstance(relation, dict):
            issues.append(f"relation_not_dict:{idx}")
            continue
        rel = str(relation.get("relation") or "")
        src = str(relation.get("from") or "")
        dst = str(relation.get("to") or "")
        if not rel or not src or not dst:
            issues.append(f"relation_missing_fields:{idx}")
            continue
        if src not in obj_type_by_id:
            issues.append(f"relation_missing_from:{rel}:{src}")
            continue
        if dst not in obj_type_by_id:
            issues.append(f"relation_missing_to:{rel}:{dst}")
            continue
        schema = link_types.get(rel)
        if schema is None:
            issues.append(f"unknown_relation:{rel}")
            continue
        src_type = obj_type_by_id[src]
        dst_type = obj_type_by_id[dst]
        if not _type_allowed(src_type, schema.get("from")):
            issues.append(f"relation_from_type_mismatch:{rel}:{src_type}")
        if not _type_allowed(dst_type, schema.get("to")):
            issues.append(f"relation_to_type_mismatch:{rel}:{dst_type}")
    issues.extend(_semantic_issues(objects_by_type, relations, obj_by_id))
    return sorted(set(issues))


def validate_case_understanding_card(card: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if str(card.get("schema_version") or "") != "kg_v2.case_understanding.v1":
        issues.append("invalid_schema_version")
    if not card.get("source_episode_id"):
        issues.append("missing_source_episode_id")
    if not card.get("source_thread_id"):
        issues.append("missing_source_thread_id")
    cases = card.get("cases")
    if not isinstance(cases, list) or not cases:
        issues.append("missing_cases")
        return sorted(set(issues))
    for idx, case in enumerate(cases):
        prefix = f"cases[{idx}]"
        if not isinstance(case, dict):
            issues.append(f"{prefix}:not_object")
            continue
        family = case.get("family_hypothesis")
        variant = case.get("variant_hypothesis")
        actions = case.get("actions")
        outcomes = case.get("outcomes")
        required = case.get("required_info")
        if not isinstance(family, dict):
            issues.append(f"{prefix}:missing_family_hypothesis")
        else:
            for key in ("label", "summary", "category"):
                if not family.get(key):
                    issues.append(f"{prefix}:family_missing_{key}")
            if family.get("label") and variant and isinstance(variant, dict) and _same_text(family.get("label"), variant.get("label")):
                issues.append(f"{prefix}:family_variant_label_collision")
        if not isinstance(variant, dict):
            issues.append(f"{prefix}:missing_variant_hypothesis")
        else:
            for key in ("label", "summary"):
                if not variant.get(key):
                    issues.append(f"{prefix}:variant_missing_{key}")
        candidate_scope = str(case.get("candidate_scope") or "fault_execution")
        if not isinstance(actions, list):
            issues.append(f"{prefix}:actions_not_list")
        elif not actions and candidate_scope != "fault_only":
            issues.append(f"{prefix}:missing_actions")
        else:
            for action_idx, action in enumerate(actions):
                ap = f"{prefix}.actions[{action_idx}]"
                if not isinstance(action, dict):
                    issues.append(f"{ap}:not_object")
                    continue
                for key in ("action_ref", "label", "summary", "action_role"):
                    if not action.get(key):
                        issues.append(f"{ap}:missing_{key}")
                if action.get("action_role") and str(action.get("action_role")) not in {"inspect", "collect", "compare", "change", "verify", "observe", "escalate"}:
                    issues.append(f"{ap}:invalid_action_role")
        if not isinstance(outcomes, list):
            issues.append(f"{prefix}:outcomes_not_list")
        else:
            for outcome_idx, outcome in enumerate(outcomes):
                op = f"{prefix}.outcomes[{outcome_idx}]"
                if not isinstance(outcome, dict):
                    issues.append(f"{op}:not_object")
                    continue
                for key in ("action_ref", "outcome_type", "summary"):
                    if not outcome.get(key):
                        issues.append(f"{op}:missing_{key}")
                if outcome.get("outcome_type") and str(outcome.get("outcome_type")) not in OUTCOME_TYPES:
                    issues.append(f"{op}:invalid_outcome_type")
        if not isinstance(required, list):
            issues.append(f"{prefix}:required_info_not_list")
        else:
            for req_idx, req in enumerate(required):
                rp = f"{prefix}.required_info[{req_idx}]"
                if not isinstance(req, dict):
                    issues.append(f"{rp}:not_object")
                    continue
                for key in ("slot_hint", "question", "why_required"):
                    if not req.get(key):
                        issues.append(f"{rp}:missing_{key}")
        if card.get("split_required") and len(cases) < 2:
            issues.append("split_required_but_case_count_lt_2")
    return sorted(set(issues))


def validate_candidate_draft_v2(draft: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if str(draft.get("schema_version") or "") != "kg_v2.candidate_draft.v1":
        issues.append("invalid_schema_version")
    if not draft.get("source_episode_id"):
        issues.append("missing_source_episode_id")
    if not draft.get("source_thread_id"):
        issues.append("missing_source_thread_id")
    split_cases = draft.get("split_cases")
    if not isinstance(split_cases, list) or not split_cases:
        issues.append("missing_split_cases")
        return sorted(set(issues))
    for idx, case in enumerate(split_cases):
        prefix = f"split_cases[{idx}]"
        if not isinstance(case, dict):
            issues.append(f"{prefix}:not_object")
            continue
        for block in ("source_case", "family", "variant", "trace"):
            if not isinstance(case.get(block), dict):
                issues.append(f"{prefix}:missing_{block}")
        family = case.get("family") if isinstance(case.get("family"), dict) else {}
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        if family:
            for key in ("label", "summary", "category"):
                if not family.get(key):
                    issues.append(f"{prefix}.family:missing_{key}")
        if variant:
            for key in ("label", "summary"):
                if not variant.get(key):
                    issues.append(f"{prefix}.variant:missing_{key}")
            if family.get("label") and _same_text(family.get("label"), variant.get("label")):
                issues.append(f"{prefix}:family_variant_label_collision")
        actions = case.get("actions")
        candidate_scope = str(case.get("candidate_scope") or "fault_execution")
        if not isinstance(actions, list):
            issues.append(f"{prefix}:actions_not_list")
        elif not actions and candidate_scope != "fault_only":
            issues.append(f"{prefix}:missing_actions")
        else:
            for action_idx, action in enumerate(actions):
                ap = f"{prefix}.actions[{action_idx}]"
                if not isinstance(action, dict):
                    issues.append(f"{ap}:not_object")
                    continue
                for key in ("label", "summary", "action_role"):
                    if not action.get(key):
                        issues.append(f"{ap}:missing_{key}")
                if action.get("action_role") and str(action.get("action_role")) not in {"inspect", "collect", "compare", "change", "verify", "observe", "escalate"}:
                    issues.append(f"{ap}:invalid_action_role")
        outcomes = case.get("outcomes")
        if not isinstance(outcomes, list):
            issues.append(f"{prefix}:outcomes_not_list")
        else:
            for outcome_idx, outcome in enumerate(outcomes):
                op = f"{prefix}.outcomes[{outcome_idx}]"
                if not isinstance(outcome, dict):
                    issues.append(f"{op}:not_object")
                    continue
                for key in ("action_label", "outcome_type", "summary"):
                    if not outcome.get(key):
                        issues.append(f"{op}:missing_{key}")
                if outcome.get("outcome_type") and str(outcome.get("outcome_type")) not in OUTCOME_TYPES:
                    issues.append(f"{op}:invalid_outcome_type")
        required = case.get("required_info")
        if not isinstance(required, list):
            issues.append(f"{prefix}:required_info_not_list")
        else:
            for req_idx, req in enumerate(required):
                rp = f"{prefix}.required_info[{req_idx}]"
                if not isinstance(req, dict):
                    issues.append(f"{rp}:not_object")
                    continue
                for key in ("slot", "question", "why_required", "blocks", "priority"):
                    if req.get(key) in (None, "", []):
                        issues.append(f"{rp}:missing_{key}")
                slot = str(req.get("slot") or "")
                if slot and slot not in INTERNAL_REQUIRED_INFO_SLOTS:
                    issues.append(f"{rp}:invalid_slot")
                priority = str(req.get("priority") or "")
                if priority and priority not in {"high", "medium", "low"}:
                    issues.append(f"{rp}:invalid_priority")
        trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
        if trace:
            if not isinstance(trace.get("recommended_action_labels"), list):
                issues.append(f"{prefix}.trace:missing_recommended_action_labels")
            if not isinstance(trace.get("actual_action_labels"), list):
                issues.append(f"{prefix}.trace:missing_actual_action_labels")
        evidence = case.get("evidence")
        if not isinstance(evidence, list):
            issues.append(f"{prefix}:evidence_not_list")
    return sorted(set(issues))


def _validate_object(obj_type: str, item: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    properties = schema.get("properties") or {}
    for prop, meta in properties.items():
        required = bool(meta.get("required"))
        value = item.get(prop)
        if required and _missing_required_value(prop, value):
            issues.append(f"missing_required:{obj_type}.{prop}")
            continue
        if value in (None, ""):
            continue
        ptype = str(meta.get("type") or "")
        if ptype == "string":
            if not isinstance(value, str):
                issues.append(f"type_mismatch:{obj_type}.{prop}:string")
            max_length = meta.get("max_length")
            if isinstance(value, str) and isinstance(max_length, int) and len(value) > max_length:
                issues.append(f"text_too_long:{obj_type}.{prop}:{len(value)}>{max_length}")
        elif ptype == "integer":
            if not isinstance(value, int):
                issues.append(f"type_mismatch:{obj_type}.{prop}:integer")
        elif ptype == "boolean":
            if not isinstance(value, bool):
                issues.append(f"type_mismatch:{obj_type}.{prop}:boolean")
        elif ptype == "array":
            if not isinstance(value, list):
                issues.append(f"type_mismatch:{obj_type}.{prop}:array")
            elif meta.get("items") == "string" and any(not isinstance(x, str) for x in value):
                issues.append(f"type_mismatch:{obj_type}.{prop}:array[string]")
        enum = meta.get("enum")
        if enum and value not in enum:
            issues.append(f"enum_mismatch:{obj_type}.{prop}:{value}")
    return issues


def _missing_required_value(prop: str, value: Any) -> bool:
    if value is None:
        return True
    if value == "" and prop not in {"condition"}:
        return True
    return False


def _type_allowed(actual: str, allowed: Any) -> bool:
    if isinstance(allowed, list):
        return actual in allowed
    return actual == str(allowed or "")


def _same_text(a: Any, b: Any) -> bool:
    return " ".join(str(a or "").split()).lower() == " ".join(str(b or "").split()).lower()


def _semantic_issues(
    objects_by_type: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, Any]],
    obj_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    family_ids = {str(item.get("family_id") or "") for item in objects_by_type.get("FaultFamily") or [] if isinstance(item, dict)}
    variant_ids = {str(item.get("variant_id") or "") for item in objects_by_type.get("FaultVariant") or [] if isinstance(item, dict)}
    action_ids = {str(item.get("action_id") or "") for item in objects_by_type.get("DiagnosticAction") or [] if isinstance(item, dict)}
    outcome_ids = {str(item.get("outcome_id") or "") for item in objects_by_type.get("ActionOutcome") or [] if isinstance(item, dict)}
    trace_ids = {str(item.get("trace_id") or "") for item in objects_by_type.get("DiagnosticTrace") or [] if isinstance(item, dict)}
    trace_step_ids = {str(item.get("trace_step_id") or "") for item in objects_by_type.get("TraceStep") or [] if isinstance(item, dict)}
    observation_ids = {str(item.get("observation_id") or "") for item in objects_by_type.get("ExecutionObservation") or [] if isinstance(item, dict)}
    branch_rule_ids = {str(item.get("branch_rule_id") or "") for item in objects_by_type.get("BranchRule") or [] if isinstance(item, dict)}
    case_ids = {str(item.get("case_id") or "") for item in objects_by_type.get("SourceCase") or [] if isinstance(item, dict)}
    evidence_ids = {str(item.get("evidence_id") or "") for item in objects_by_type.get("EvidenceItem") or [] if isinstance(item, dict)}
    document_ids = {str(item.get("document_id") or "") for item in objects_by_type.get("KnowledgeDocument") or [] if isinstance(item, dict)}
    section_ids = {str(item.get("section_id") or "") for item in objects_by_type.get("KnowledgeSection") or [] if isinstance(item, dict)}
    procedure_step_ids = {str(item.get("procedure_step_id") or "") for item in objects_by_type.get("ProcedureStep") or [] if isinstance(item, dict)}
    concept_ids = {str(item.get("concept_id") or "") for item in objects_by_type.get("DebugConcept") or [] if isinstance(item, dict)}
    domain_concept_ids = {
        str(item.get("concept_id") or "")
        for item in objects_by_type.get("DebugConcept") or []
        if isinstance(item, dict)
        and str(item.get("canonical_target_id") or "")
    }
    term_ids = {str(item.get("term_id") or "") for item in objects_by_type.get("TermExpression") or [] if isinstance(item, dict)}
    sense_ids = {str(item.get("sense_id") or "") for item in objects_by_type.get("TermSense") or [] if isinstance(item, dict)}
    domain_ids_by_type = {
        "FaultFamily": family_ids,
        "FaultVariant": variant_ids,
        "DiagnosticAction": action_ids,
    }

    for item in objects_by_type.get("DebugConcept") or []:
        if not isinstance(item, dict):
            continue
        concept_id = str(item.get("concept_id") or "")
        target_type = str(item.get("canonical_target_type") or "")
        target_id = str(item.get("canonical_target_id") or "")
        if (target_type or target_id) and (
            target_type not in domain_ids_by_type
            or target_id not in domain_ids_by_type.get(target_type, set())
        ):
            issues.append(
                f"concept_missing_canonical_target:"
                f"{concept_id}:{target_type}:{target_id}"
            )
    for item in objects_by_type.get("TermSense") or []:
        if not isinstance(item, dict):
            continue
        sense_id = str(item.get("sense_id") or "")
        if str(item.get("term_id") or "") not in term_ids:
            issues.append(f"sense_missing_term:{sense_id}")
        if str(item.get("concept_id") or "") not in concept_ids:
            issues.append(f"sense_missing_concept:{sense_id}")
    if concept_ids or term_ids or sense_ids:
        primary_targets = {
            str(relation.get("to") or "")
            for relation in relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "primary_concept"
        }
        denoted_senses = {
            str(relation.get("from") or "")
            for relation in relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "sense_denotes"
        }
        expressed_senses = {
            str(relation.get("to") or "")
            for relation in relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "expression_has_sense"
        }
        for concept_id in domain_concept_ids - primary_targets:
            issues.append(f"concept_missing_primary_relation:{concept_id}")
        for sense_id in sense_ids - denoted_senses:
            issues.append(f"sense_missing_denotation_relation:{sense_id}")
        for sense_id in sense_ids - expressed_senses:
            issues.append(f"sense_missing_expression_relation:{sense_id}")

    for item in objects_by_type.get("FaultVariant") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("family_id") or "") not in family_ids:
            issues.append(f"variant_missing_family:{item.get('variant_id')}")
    case_action_ids: set[str] = set()
    action_by_id: dict[str, dict[str, Any]] = {}
    for item in objects_by_type.get("DiagnosticAction") or []:
        if not isinstance(item, dict):
            continue
        action_id = str(item.get("action_id") or "")
        action_by_id[action_id] = item
        family_id = str(item.get("family_id") or "")
        variant_id = str(item.get("variant_id") or "")
        if family_id not in family_ids:
            issues.append(f"action_missing_family:{item.get('action_id')}")
        if variant_id and variant_id not in variant_ids:
            issues.append(f"action_missing_variant:{item.get('action_id')}")
        if str(item.get("source_kind") or "") == "case":
            case_action_ids.add(action_id)
            execution_status = str(item.get("execution_status") or "")
            if execution_status not in {"actual", "recommended"}:
                issues.append(f"case_action_invalid_execution_status:{action_id}:{execution_status or 'missing'}")
            action_evidence = [str(value or "") for value in item.get("evidence_ids") or [] if str(value or "")]
            if not action_evidence or any(value not in evidence_ids for value in action_evidence):
                issues.append(f"case_action_missing_evidence:{action_id}")
            evidence_scope = str(item.get("evidence_scope") or "")
            if evidence_scope and evidence_scope not in ACTION_EVIDENCE_SCOPES:
                issues.append(f"invalid_action_evidence_scope:{action_id}:{evidence_scope}")
    outcome_action_ids: set[str] = set()
    outcome_by_id: dict[str, dict[str, Any]] = {}
    for item in objects_by_type.get("ActionOutcome") or []:
        if not isinstance(item, dict):
            continue
        action_id = str(item.get("action_id") or "")
        outcome_by_id[str(item.get("outcome_id") or "")] = item
        outcome_action_ids.add(action_id)
        if action_id not in action_ids:
            issues.append(f"outcome_missing_action:{item.get('outcome_id')}")
        if str(item.get("source_case_id") or "") not in case_ids:
            issues.append(f"outcome_missing_case:{item.get('outcome_id')}")
        outcome_evidence = [str(value or "") for value in item.get("evidence_ids") or [] if str(value or "")]
        if not outcome_evidence or any(value not in evidence_ids for value in outcome_evidence):
            issues.append(f"outcome_missing_evidence:{item.get('outcome_id')}")
        if str(item.get("outcome_type") or "") not in OUTCOME_TYPES:
            issues.append(f"invalid_outcome_type:{item.get('outcome_id')}:{item.get('outcome_type')}")
        outcome_origin = str(item.get("outcome_origin") or "")
        activation_mode = str(item.get("activation_mode") or "")
        if outcome_origin and outcome_origin not in OUTCOME_ORIGINS:
            issues.append(f"invalid_outcome_origin:{item.get('outcome_id')}:{outcome_origin}")
        if outcome_origin == "synthetic_fallback" and str(item.get("outcome_type") or "") != "pending_validation":
            issues.append(f"synthetic_outcome_claims_observation:{item.get('outcome_id')}:{item.get('outcome_type')}")
        action = action_by_id.get(action_id) or {}
        conditional_verified_fix = (
            str(item.get("outcome_type") or "") == "verified_fix"
            and activation_mode == "human_confirmed_runtime"
        )
        if conditional_verified_fix:
            requirements = item.get("activation_requirements")
            groups = (
                requirements.get("all_of_groups")
                if isinstance(requirements, dict)
                else None
            )
            if (
                str(action.get("action_role") or "") != "verify"
                or not isinstance(groups, list)
                or len(groups) < 2
                or any(not isinstance(group, list) or not group for group in groups)
            ):
                issues.append(
                    f"invalid_runtime_verified_fix_template:"
                    f"{item.get('outcome_id')}:{action_id}"
                )
        if (
            str(item.get("outcome_type") or "") == "verified_fix"
            and not conditional_verified_fix
            and str(action.get("execution_status") or "") != "actual"
        ):
            issues.append(f"verified_fix_for_non_actual_action:{item.get('outcome_id')}:{action_id}")
    for action_id in sorted(case_action_ids - outcome_action_ids):
        issues.append(f"case_action_missing_outcome:{action_id}")
    for item in objects_by_type.get("RequiredInfoSpec") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("slot") or "") not in INTERNAL_REQUIRED_INFO_SLOTS:
            issues.append(f"invalid_required_info_slot:{item.get('required_info_id')}:{item.get('slot')}")
        if any(str(x or "") not in evidence_ids for x in item.get("evidence_ids") or []):
            issues.append(f"required_info_missing_evidence:{item.get('required_info_id')}")
    trace_by_id: dict[str, dict[str, Any]] = {}
    trace_occurrences: dict[str, list[dict[str, Any]]] = {}
    for item in objects_by_type.get("DiagnosticTrace") or []:
        if not isinstance(item, dict):
            continue
        trace_by_id[str(item.get("trace_id") or "")] = item
        if str(item.get("source_case_id") or "") not in case_ids:
            issues.append(f"trace_missing_case:{item.get('trace_id')}")
        if any(str(x or "") not in action_ids for x in item.get("recommended_action_ids") or []):
            issues.append(f"trace_missing_recommended_action:{item.get('trace_id')}")
        if any(str(x or "") not in action_ids for x in item.get("actual_action_ids") or []):
            issues.append(f"trace_missing_actual_action:{item.get('trace_id')}")
        raw_occurrences = item.get("action_occurrences")
        if raw_occurrences is not None:
            if not isinstance(raw_occurrences, list) or not raw_occurrences:
                issues.append(
                    f"trace_invalid_action_occurrences:{item.get('trace_id')}"
                )
            else:
                normalized_occurrences = [
                    value for value in raw_occurrences
                    if isinstance(value, dict)
                ]
                trace_occurrences[str(item.get("trace_id") or "")] = (
                    normalized_occurrences
                )
                for index, occurrence in enumerate(
                    normalized_occurrences, 1
                ):
                    action_id = str(
                        occurrence.get("action_id") or ""
                    )
                    case_id = str(
                        occurrence.get("source_case_id")
                        or item.get("source_case_id")
                        or ""
                    )
                    status = str(
                        occurrence.get("execution_status") or ""
                    )
                    if action_id not in action_ids:
                        issues.append(
                            f"trace_occurrence_missing_action:"
                            f"{item.get('trace_id')}:{index}:{action_id}"
                        )
                    if action_id not in {
                        str(value)
                        for value in (
                            item.get("recommended_action_ids") or []
                        )
                    }:
                        issues.append(
                            f"trace_occurrence_not_recommended:"
                            f"{item.get('trace_id')}:{index}:{action_id}"
                        )
                    if case_id not in case_ids:
                        issues.append(
                            f"trace_occurrence_missing_case:"
                            f"{item.get('trace_id')}:{index}:{case_id}"
                        )
                    if status not in TRACE_EXECUTION_STATUSES:
                        issues.append(
                            f"trace_occurrence_invalid_status:"
                            f"{item.get('trace_id')}:{index}:{status}"
                        )
                    if (
                        status == "actual"
                        and int(occurrence.get("attempt_index") or 0) < 1
                    ):
                        issues.append(
                            f"trace_occurrence_invalid_attempt:"
                            f"{item.get('trace_id')}:{index}"
                        )
    steps_by_trace: dict[str, list[dict[str, Any]]] = {}
    step_by_id: dict[str, dict[str, Any]] = {}
    for item in objects_by_type.get("TraceStep") or []:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("trace_step_id") or "")
        trace_id = str(item.get("trace_id") or "")
        action_id = str(item.get("action_id") or "")
        case_id = str(item.get("source_case_id") or "")
        step_by_id[step_id] = item
        steps_by_trace.setdefault(trace_id, []).append(item)
        if trace_id not in trace_ids:
            issues.append(f"trace_step_missing_trace:{step_id}:{trace_id}")
        if action_id not in action_ids:
            issues.append(f"trace_step_missing_action:{step_id}:{action_id}")
        if case_id not in case_ids:
            issues.append(f"trace_step_missing_case:{step_id}:{case_id}")
        if str(item.get("execution_status") or "") not in TRACE_EXECUTION_STATUSES:
            issues.append(f"trace_step_invalid_execution_status:{step_id}")
        if int(item.get("ordinal") or 0) < 1:
            issues.append(f"trace_step_invalid_ordinal:{step_id}")
        if int(item.get("attempt_index") or 0) < 0:
            issues.append(f"trace_step_invalid_attempt_index:{step_id}")
        step_evidence = [str(value or "") for value in item.get("evidence_ids") or [] if str(value or "")]
        if not step_evidence or any(value not in evidence_ids for value in step_evidence):
            issues.append(f"trace_step_missing_evidence:{step_id}")

    for trace_id, steps in steps_by_trace.items():
        trace = trace_by_id.get(trace_id) or {}
        ordered = sorted(steps, key=lambda value: int(value.get("ordinal") or 0))
        ordinals = [int(value.get("ordinal") or 0) for value in ordered]
        if ordinals != list(range(1, len(ordered) + 1)):
            issues.append(f"trace_step_non_contiguous_order:{trace_id}")
        step_actions = [str(value.get("action_id") or "") for value in ordered]
        occurrences = trace_occurrences.get(trace_id)
        expected_step_actions = (
            [
                str(value.get("action_id") or "")
                for value in occurrences
            ]
            if occurrences is not None
            else [
                str(value or "")
                for value in trace.get("recommended_action_ids") or []
            ]
        )
        if step_actions != expected_step_actions:
            issues.append(
                f"trace_step_recommended_order_mismatch:{trace_id}"
            )
        actual_actions = [
            str(value.get("action_id") or "")
            for value in ordered
            if str(value.get("execution_status") or "") == "actual"
        ]
        expected_actual_actions = (
            [
                str(value.get("action_id") or "")
                for value in occurrences
                if str(value.get("execution_status") or "") == "actual"
            ]
            if occurrences is not None
            else [
                str(value or "")
                for value in trace.get("actual_action_ids") or []
            ]
        )
        if actual_actions != expected_actual_actions:
            issues.append(
                f"trace_step_actual_order_mismatch:{trace_id}"
            )
    if trace_step_ids:
        for outcome in objects_by_type.get("ActionOutcome") or []:
            if (
                not isinstance(outcome, dict)
                or str(outcome.get("outcome_type") or "") != "verified_fix"
                or str(outcome.get("activation_mode") or "")
                == "human_confirmed_runtime"
            ):
                continue
            action_id = str(outcome.get("action_id") or "")
            case_id = str(outcome.get("source_case_id") or "")
            if not any(
                str(step.get("action_id") or "") == action_id
                and str(step.get("source_case_id") or "") == case_id
                and str(step.get("execution_status") or "") == "actual"
                for step in step_by_id.values()
            ):
                issues.append(f"verified_fix_without_actual_trace_step:{outcome.get('outcome_id')}:{action_id}")

    observations_by_step: dict[str, list[dict[str, Any]]] = {}
    for item in objects_by_type.get("ExecutionObservation") or []:
        if not isinstance(item, dict):
            continue
        observation_id = str(item.get("observation_id") or "")
        step_id = str(item.get("trace_step_id") or "")
        action_id = str(item.get("action_id") or "")
        case_id = str(item.get("source_case_id") or "")
        observations_by_step.setdefault(step_id, []).append(item)
        step = step_by_id.get(step_id) or {}
        if not step:
            issues.append(f"observation_missing_trace_step:{observation_id}:{step_id}")
        elif str(step.get("execution_status") or "") != "actual":
            issues.append(f"observation_for_non_actual_step:{observation_id}:{step_id}")
        if action_id not in action_ids or (step and str(step.get("action_id") or "") != action_id):
            issues.append(f"observation_action_mismatch:{observation_id}:{action_id}")
        if case_id not in case_ids or (step and str(step.get("source_case_id") or "") != case_id):
            issues.append(f"observation_case_mismatch:{observation_id}:{case_id}")
        if int(item.get("attempt_index") or 0) < 1 or int(item.get("observation_count") or 0) < 1:
            issues.append(f"observation_invalid_count:{observation_id}")
        referenced_outcomes = [str(value or "") for value in item.get("outcome_ids") or [] if str(value or "")]
        if not referenced_outcomes or any(value not in outcome_ids for value in referenced_outcomes):
            issues.append(f"observation_missing_outcome:{observation_id}")
        elif any(str((outcome_by_id.get(value) or {}).get("action_id") or "") != action_id for value in referenced_outcomes):
            issues.append(f"observation_outcome_action_mismatch:{observation_id}")
        expected_types = sorted({str((outcome_by_id.get(value) or {}).get("outcome_type") or "") for value in referenced_outcomes})
        actual_types = sorted({str(value or "") for value in item.get("outcome_types") or [] if str(value or "")})
        if expected_types != actual_types:
            issues.append(f"observation_outcome_type_mismatch:{observation_id}")
        observation_evidence = [str(value or "") for value in item.get("evidence_ids") or [] if str(value or "")]
        if not observation_evidence or any(value not in evidence_ids for value in observation_evidence):
            issues.append(f"observation_missing_evidence:{observation_id}")
    for step_id, step in step_by_id.items():
        if str(step.get("execution_status") or "") == "actual" and not observations_by_step.get(step_id):
            issues.append(f"actual_trace_step_missing_observation:{step_id}")

    branches_by_step: dict[str, list[dict[str, Any]]] = {}
    for item in objects_by_type.get("BranchRule") or []:
        if not isinstance(item, dict):
            continue
        branch_id = str(item.get("branch_rule_id") or "")
        trace_id = str(item.get("trace_id") or "")
        case_id = str(item.get("source_case_id") or "")
        from_step_id = str(item.get("from_trace_step_id") or "")
        to_step_id = str(item.get("to_trace_step_id") or "")
        branches_by_step.setdefault(from_step_id, []).append(item)
        if trace_id not in trace_ids:
            issues.append(f"branch_missing_trace:{branch_id}:{trace_id}")
        if case_id not in case_ids:
            issues.append(f"branch_missing_case:{branch_id}:{case_id}")
        from_step = step_by_id.get(from_step_id) or {}
        to_step = (step_by_id.get(to_step_id) or {}) if to_step_id else {}
        if not from_step or str(from_step.get("trace_id") or "") != trace_id:
            issues.append(f"branch_from_step_mismatch:{branch_id}:{from_step_id}")
        if to_step_id and (not to_step or str(to_step.get("trace_id") or "") != trace_id):
            issues.append(f"branch_to_step_mismatch:{branch_id}:{to_step_id}")
        triggers = [str(value or "") for value in item.get("trigger_outcome_types") or [] if str(value or "")]
        if not triggers or any(value not in OUTCOME_TYPES for value in triggers):
            issues.append(f"branch_invalid_trigger:{branch_id}")
        if str(item.get("branch_kind") or "") not in BRANCH_KINDS:
            issues.append(f"branch_invalid_kind:{branch_id}")
        terminal_status = str(item.get("terminal_status") or "")
        if terminal_status not in BRANCH_TERMINAL_STATUSES:
            issues.append(f"branch_invalid_terminal_status:{branch_id}")
        if bool(to_step_id) != (terminal_status == "continue"):
            issues.append(f"branch_target_terminal_mismatch:{branch_id}")
        branch_evidence = [str(value or "") for value in item.get("evidence_ids") or [] if str(value or "")]
        if not branch_evidence or any(value not in evidence_ids for value in branch_evidence):
            issues.append(f"branch_missing_evidence:{branch_id}")
    for step_id in step_by_id:
        if not branches_by_step.get(step_id):
            issues.append(f"trace_step_missing_branch:{step_id}")
    for item in objects_by_type.get("DecisionPolicy") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("family_id") or "") not in family_ids:
            issues.append(f"policy_missing_family:{item.get('policy_id')}")
        if item.get("deterministic_recompute") is not True:
            issues.append(f"policy_not_deterministic:{item.get('policy_id')}")
        policy_id = str(item.get("policy_id") or "")
        if "observation_ids" in item:
            referenced_observations = [str(value or "") for value in item.get("observation_ids") or [] if str(value or "")]
            if any(value not in observation_ids for value in referenced_observations):
                issues.append(f"policy_missing_observation:{policy_id}")
            expected_count = sum(
                int(value.get("observation_count") or 0)
                for value in objects_by_type.get("ExecutionObservation") or []
                if isinstance(value, dict) and str(value.get("observation_id") or "") in referenced_observations
            )
            if int(item.get("actual_execution_count") or 0) != expected_count:
                issues.append(f"policy_actual_execution_count_mismatch:{policy_id}")
        if "branch_rule_ids" in item and any(
            str(value or "") not in branch_rule_ids for value in item.get("branch_rule_ids") or []
        ):
            issues.append(f"policy_missing_branch_rule:{policy_id}")
    for item in objects_by_type.get("SourceCase") or []:
        if not isinstance(item, dict):
            continue
        if item.get("approved") not in (True, False):
            issues.append(f"case_missing_approved_bool:{item.get('case_id')}")
        if str(item.get("trust_tier") or "") == "gold":
            for key in ("annotation_set_id", "annotation_case_id", "annotation_sha256", "review_id", "ingest_run_id"):
                if not str(item.get(key) or ""):
                    issues.append(f"gold_case_missing_provenance:{item.get('case_id')}:{key}")
    for item in objects_by_type.get("KnowledgeSection") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("document_id") or "") not in document_ids:
            issues.append(f"section_missing_document:{item.get('section_id')}")
    for item in objects_by_type.get("ProcedureStep") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("section_id") or "") not in section_ids:
            issues.append(f"procedure_step_missing_section:{item.get('procedure_step_id')}")

    relation_keys = {
        (
            str(relation.get("from") or ""),
            str(relation.get("to") or ""),
            str(relation.get("relation") or ""),
        )
        for relation in relations
        if isinstance(relation, dict)
    }
    for item in objects_by_type.get("MediaAsset") or []:
        if not isinstance(item, dict):
            continue
        media_id = str(item.get("media_id") or "")
        referenced_documents = {
            str(value or "") for value in item.get("document_ids") or []
            if str(value or "")
        }
        referenced_sections = {
            str(value or "") for value in item.get("section_ids") or []
            if str(value or "")
        }
        referenced_steps = {
            str(value or "") for value in item.get("procedure_step_ids") or []
            if str(value or "")
        }
        referenced_actions = {
            str(value or "") for value in item.get("action_ids") or []
            if str(value or "")
        }
        if not referenced_documents:
            issues.append(f"media_without_document:{media_id}")
        if any(value not in document_ids for value in referenced_documents):
            issues.append(f"media_missing_document:{media_id}")
        if any(value not in section_ids for value in referenced_sections):
            issues.append(f"media_missing_section:{media_id}")
        if any(value not in procedure_step_ids for value in referenced_steps):
            issues.append(f"media_missing_procedure_step:{media_id}")
        if any(value not in action_ids for value in referenced_actions):
            issues.append(f"media_missing_action:{media_id}")
        if not item.get("source_occurrences"):
            issues.append(f"media_without_source_occurrence:{media_id}")
        if len(str(item.get("content_hash") or "")) != 64:
            issues.append(f"media_invalid_content_hash:{media_id}")
        for document_id in referenced_documents:
            if (document_id, media_id, "has_media") not in relation_keys:
                issues.append(f"media_missing_document_relation:{media_id}:{document_id}")
        for section_id in referenced_sections:
            if (section_id, media_id, "section_media") not in relation_keys:
                issues.append(f"media_missing_section_relation:{media_id}:{section_id}")
        for step_id in referenced_steps:
            if (step_id, media_id, "step_media") not in relation_keys:
                issues.append(f"media_missing_step_relation:{media_id}:{step_id}")
        for action_id in referenced_actions:
            if (action_id, media_id, "action_media") not in relation_keys:
                issues.append(f"media_missing_action_relation:{media_id}:{action_id}")

    supports = {(str(rel.get("from") or ""), str(rel.get("to") or "")) for rel in relations if isinstance(rel, dict) and rel.get("relation") == "supports"}
    described_variants = {
        str(rel.get("to") or "")
        for rel in relations
        if isinstance(rel, dict) and rel.get("relation") == "describes_variant"
    }
    for item in objects_by_type.get("FaultVariant") or []:
        if not isinstance(item, dict):
            continue
        variant_id = str(item.get("variant_id") or "")
        if not any(dst == variant_id for _, dst in supports) and variant_id not in described_variants:
            issues.append(f"variant_without_source_case:{variant_id}")
    for evidence_id in evidence_ids:
        if not any(str(rel.get("from") or "") == evidence_id for rel in relations if isinstance(rel, dict) and rel.get("relation") == "evidences"):
            issues.append(f"evidence_without_target:{evidence_id}")

    # Atomic execution-node guard: keep heavy text out of family/action/required-info fields.
    for obj_type in ("FaultFamily", "DiagnosticAction", "RequiredInfoSpec"):
        for item in objects_by_type.get(obj_type) or []:
            if not isinstance(item, dict):
                continue
            for key in ("label", "summary", "question", "why_required"):
                value = str(item.get(key) or "")
                if value.count("\n") > 0:
                    issues.append(f"multiline_atomic_field:{obj_type}.{key}:{item.get(V2_PRIMARY_KEYS[obj_type])}")
    return issues
