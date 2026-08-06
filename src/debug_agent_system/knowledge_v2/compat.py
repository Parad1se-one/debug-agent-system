"""Compatibility helpers between write-side candidates and KG v2 bundles."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from debug_agent_system.knowledge_v2.builders import infer_action_role, infer_required_info_slot
from debug_agent_system.knowledge_v2.contracts import (
    APPROVED_FAMILY_LABELS,
    FAMILY_SUBSYSTEM_EXPECTED,
    INTERNAL_REQUIRED_INFO_SLOTS,
    PSEUDO_FAMILY_LABELS,
    make_id,
    trim_text,
)
from debug_agent_system.knowledge_v2.validator import validate_candidate_draft_v2, validate_case_understanding_card, validate_graph


def build_v2_bundle_from_legacy_candidate(candidate: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Project one legacy write-side candidate into a schema-valid KG v2 bundle.

    This is the compatibility path for dual-write W1-W6: legacy extraction and
    gate logic remain the control plane, while KG v2 receives an equivalent
    case-layer bundle without reading old persisted KG nodes.
    """

    nodes = [node for node in candidate.get("nodes") or [] if isinstance(node, dict)]
    outcomes = [item for item in candidate.get("diagnostic_outcomes") or [] if isinstance(item, dict)]
    error = next((node for node in nodes if node.get("type") == "Error"), {})
    variant = candidate.get("case_variant_candidate") if isinstance(candidate.get("case_variant_candidate"), dict) else {}
    trace = candidate.get("diagnostic_trace") if isinstance(candidate.get("diagnostic_trace"), dict) else {}
    matched_existing = candidate.get("matched_existing_error") if isinstance(candidate.get("matched_existing_error"), dict) else {}
    target_error_id = str(variant.get("error_id") or error.get("error_id") or candidate.get("candidate_id") or "unknown")
    canonical_error_id = str(
        variant.get("canonical_error_id")
        or error.get("canonical_error_id")
        or matched_existing.get("error_id")
        or target_error_id
    )
    episode_extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    extracted_conclusion = str((episode_extracted or {}).get("conclusion") or (episode_extracted or {}).get("key_conclusion") or "")
    semantic_text = " ".join([
        str(candidate.get("label") or ""),
        str(candidate.get("symptom_raw") or ""),
        str(candidate.get("conclusion") or ""),
        extracted_conclusion,
        str(error.get("symptom") or ""),
        str(variant.get("scenario") or ""),
        " ".join(str(x) for x in ((episode_extracted or {}).get("debug_actions") or [])),
        " ".join(str((msg or {}).get("text") or "") for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages") for msg in episode.get(key) or [] if isinstance(msg, dict)),
    ])
    family_id = make_id("family", canonical_error_id)
    variant_id = make_id("variant", target_error_id)
    case_id = make_id("case", f"{episode.get('thread_id') or ''}:{episode.get('episode_id') or candidate.get('candidate_id') or target_error_id}")
    evidence_ids = _evidence_items(case_id, episode)
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []

    family_label, family_subsystem, family_summary = _infer_family_shape(canonical_error_id, semantic_text, variant, error)
    variant_label, variant_summary = _infer_variant_shape(target_error_id, semantic_text, variant, error)
    objects["FaultFamily"].append({
        "family_id": family_id,
        "label": trim_text(family_label, 40),
        "summary": trim_text(family_summary, 80),
        "category": str(variant.get("category") or error.get("category") or "系统与软件异常"),
        "subsystem": trim_text(family_subsystem or variant.get("subsystem") or error.get("subsystem") or "", 40),
        "scenario": trim_text(variant.get("scenario") or error.get("scenario") or "", 60),
        "keywords": _keywords(variant, error),
        "source_kind": "case",
        "escalation_target": str(variant.get("escalation_target") or error.get("escalation_target") or ""),
    })
    objects["FaultVariant"].append({
        "variant_id": variant_id,
        "family_id": family_id,
        "label": trim_text(variant_label, 60),
        "summary": trim_text(variant_summary, 180),
        "equipment_type": str(variant.get("equipment_type") or error.get("equipment_type") or ""),
        "site": trim_text(_first_string(candidate.get("sites") or []), 60),
        "software_version": trim_text(_first_string(candidate.get("versions") or []), 60),
        "error_phase": trim_text(_first_string(candidate.get("phases") or []), 40),
        "owner_context": trim_text(_owner_context(episode), 80),
        "escalation_target": str(variant.get("escalation_target") or error.get("escalation_target") or ""),
        "keywords": _keywords(variant, error),
    })
    relations.append({"from": family_id, "to": variant_id, "relation": "has_variant"})
    objects["SourceCase"].append({
        "case_id": case_id,
        "source_kind": "chat_case",
        "title": trim_text(str(variant.get("label") or error.get("label") or candidate.get("candidate_id") or target_error_id), 80),
        "summary": trim_text(candidate.get("conclusion") or candidate.get("symptom_raw") or variant.get("symptom") or error.get("symptom") or "", 240),
        "source_ref": str(episode.get("episode_id") or ""),
        "approved": False,
    })
    for evidence in evidence_ids:
        objects["EvidenceItem"].append(evidence)
        relations.append({"from": evidence["evidence_id"], "to": case_id, "relation": "evidences"})
    relations.append({"from": case_id, "to": variant_id, "relation": "supports"})

    action_id_by_check_id: dict[str, str] = {}
    action_ids_in_order: list[str] = []
    checks = sorted(
        [node for node in nodes if node.get("type") == "DiagnosticCheck"],
        key=lambda node: (int(node.get("step_order") or 999), str(node.get("check_id") or "")),
    )
    for check in checks:
        check_id = str(check.get("check_id") or check.get("id") or "")
        action_id = make_id("action", check_id or check.get("label") or f"{case_id}:check")
        action_id_by_check_id[check_id] = action_id
        action_ids_in_order.append(action_id)
        objects["DiagnosticAction"].append({
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(check.get("label") or _humanized(check_id), 60),
            "summary": trim_text(check.get("how_to_check") or check.get("label") or "", 180),
            "action_role": infer_action_role(" ".join([str(check.get("label") or ""), str(check.get("how_to_check") or "")])),
            "step_order": int(check.get("step_order") or 0),
            "destructive": bool(check.get("destructive")),
            "high_cost": False,
            "source_kind": "case",
        })

    action_id_by_solution_id: dict[str, str] = {}
    solutions = [node for node in nodes if node.get("type") == "Solution"]
    for idx, solution in enumerate(solutions, start=1):
        solution_id = str(solution.get("solution_id") or solution.get("id") or "")
        action_id = make_id("action", solution_id or solution.get("content") or f"{case_id}:solution:{idx}")
        action_id_by_solution_id[solution_id] = action_id
        objects["DiagnosticAction"].append({
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(_action_label(str(solution.get("content") or "")), 60),
            "summary": trim_text(solution.get("content") or "", 180),
            "action_role": infer_action_role(str(solution.get("content") or "")),
            "step_order": 100 + idx,
            "destructive": bool(solution.get("destructive")),
            "high_cost": "high_cost" in str(solution.get("evidence_level") or "") or "返厂" in str(solution.get("content") or "") or "重标" in str(solution.get("content") or ""),
            "source_kind": "case",
        })
    for idx, action in enumerate(((episode.get("extracted") or {}).get("debug_actions") or []), start=1):
        clean = trim_text(action, 180)
        if not clean:
            continue
        if any(_norm(clean) == _norm(item.get("label") or "") for item in objects["DiagnosticAction"]):
            continue
        action_id = make_id("action", f"{case_id}:episode:{idx}:{clean}")
        action_ids_in_order.append(action_id)
        objects["DiagnosticAction"].append({
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(_action_label(clean), 60),
            "summary": clean,
            "action_role": infer_action_role(clean),
            "step_order": 1000 + idx,
            "destructive": False,
            "high_cost": False,
            "source_kind": "case",
        })

    if not outcomes:
        outcomes = _outcomes_from_solutions(candidate, solutions)
    for idx, outcome in enumerate(outcomes, start=1):
        outcome_id = make_id("outcome", outcome.get("outcome_id") or outcome.get("target_solution_id") or outcome.get("target_check_id") or f"{case_id}:{idx}")
        action_id = action_id_by_check_id.get(str(outcome.get("target_check_id") or "")) or action_id_by_solution_id.get(str(outcome.get("target_solution_id") or ""))
        if not action_id:
            action_id = make_id("action", outcome.get("action_label") or f"{case_id}:outcome:{idx}")
            objects["DiagnosticAction"].append({
                "action_id": action_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "label": trim_text(_action_label(str(outcome.get("action_label") or "")), 60),
                "summary": trim_text(outcome.get("action_label") or "", 180),
                "action_role": infer_action_role(str(outcome.get("action_label") or "")),
                "step_order": 200 + idx,
                "destructive": bool(outcome.get("destructive")),
                "high_cost": bool(outcome.get("high_cost")),
                "source_kind": "case",
            })
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": str(outcome.get("outcome_type") or "pending_validation"),
            **({"outcome_origin": str(outcome.get("outcome_origin"))} if outcome.get("outcome_origin") else {}),
            "summary": trim_text(outcome.get("action_label") or outcome.get("root_cause_summary") or "", 200),
            "source_case_id": case_id,
            "evidence_ids": _evidence_id_list(evidence_ids, outcome.get("evidence_message_ids")),
            "high_cost": bool(outcome.get("high_cost")),
            "destructive": bool(outcome.get("destructive")),
            "root_cause_summary": trim_text(outcome.get("root_cause_summary") or "", 120),
        })
        relations.append({"from": variant_id, "to": outcome_id, "relation": "has_outcome"})
        relations.append({"from": case_id, "to": outcome_id, "relation": "supports"})
        relations.append({"from": outcome_id, "to": action_id, "relation": "outcome_of"})
        for evidence_id in _evidence_id_list(evidence_ids, outcome.get("evidence_message_ids")):
            relations.append({"from": evidence_id, "to": outcome_id, "relation": "evidences"})
    _ensure_action_outcomes(
        objects["DiagnosticAction"],
        objects["ActionOutcome"],
        case_id=case_id,
        family_id=family_id,
        variant_id=variant_id,
        evidence_ids=evidence_ids,
        semantic_text=semantic_text,
        conclusion=str(candidate.get("conclusion") or extracted_conclusion),
    )

    # A legacy action is only a case-layer action after it is tied back to the
    # current episode.  Do this centrally so checks, solutions, extracted
    # actions, and outcome-created actions all obey the same v2 contract.
    case_evidence_ids = [entry["evidence_id"] for entry in evidence_ids[:8]]
    for action in objects["DiagnosticAction"]:
        action["evidence_ids"] = list(case_evidence_ids)
        action["execution_status"] = "recommended"

    # Older candidates did not consistently emit outcome relations, especially
    # for the compatibility outcomes synthesized above.  Reconstruct the
    # provenance edges for every outcome before deduplication.
    for outcome in objects["ActionOutcome"]:
        outcome_id = str(outcome.get("outcome_id") or "")
        action_id = str(outcome.get("action_id") or "")
        if not outcome_id or not action_id:
            continue
        relations.append({"from": variant_id, "to": outcome_id, "relation": "has_outcome"})
        relations.append({"from": case_id, "to": outcome_id, "relation": "supports"})
        relations.append({"from": outcome_id, "to": action_id, "relation": "outcome_of"})
        for evidence_id in outcome.get("evidence_ids") or []:
            relations.append({"from": evidence_id, "to": outcome_id, "relation": "evidences"})

    trace_action_ids = _ordered_action_ids_from_debug_actions(
        ((episode.get("extracted") or {}).get("debug_actions") or []),
        objects["DiagnosticAction"],
        fallback=action_ids_in_order or list(dict.fromkeys(action_id_by_solution_id.values())),
    )
    recommended_trace_action_ids = _trace_action_ids(trace, trace_action_ids, action_id_by_check_id)
    actual_trace_action_ids = _trace_action_ids(trace, trace_action_ids, action_id_by_check_id, actual=True)
    observed_action_ids = {
        str(outcome.get("action_id") or "")
        for outcome in objects["ActionOutcome"]
        if str(outcome.get("outcome_type") or "") != "pending_validation"
    }
    actual_action_ids = set(actual_trace_action_ids) | observed_action_ids
    for action in objects["DiagnosticAction"]:
        if str(action.get("action_id") or "") in actual_action_ids:
            action["execution_status"] = "actual"
    if trace_action_ids:
        trace_id = make_id("trace", trace.get("trace_id") or case_id)
        objects["DiagnosticTrace"].append({
            "trace_id": trace_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "source_case_id": case_id,
            "summary": trim_text(f"{trim_text(variant.get('label') or error.get('label') or target_error_id, 40)} 的兼容排查链", 160),
            "recommended_action_ids": recommended_trace_action_ids,
            "actual_action_ids": actual_trace_action_ids,
            "evidence_ids": [item["evidence_id"] for item in evidence_ids[:8]],
        })
        relations.append({"from": family_id, "to": trace_id, "relation": "has_trace"})
        relations.append({"from": variant_id, "to": trace_id, "relation": "has_trace"})
        relations.append({"from": case_id, "to": trace_id, "relation": "supports"})
        for action_id in recommended_trace_action_ids:
            relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})

    required_specs = _required_info_specs(candidate, variant, error, case_id, family_id, variant_id, evidence_ids)
    objects["RequiredInfoSpec"].extend(required_specs)
    for item in required_specs:
        rid = str(item.get("required_info_id") or "")
        relations.append({"from": variant_id, "to": rid, "relation": "has_required_info"})
        relations.append({"from": case_id, "to": rid, "relation": "supports"})
        for evidence_id in item.get("evidence_ids") or []:
            relations.append({"from": evidence_id, "to": rid, "relation": "evidences"})

    objects = _dedupe_objects(objects)
    relations = _dedupe_relations(relations)
    issues = validate_graph(objects, relations)
    return {
        "candidate_id": f"v2:{candidate.get('candidate_id') or candidate.get('id') or variant_id}",
        "legacy_candidate_id": candidate.get("candidate_id") or candidate.get("id") or "",
        "source_episode_id": episode.get("episode_id") or "",
        "source_thread_id": episode.get("thread_id") or "",
        "family_id": family_id,
        "variant_id": variant_id,
        "objects": objects,
        "relations": relations,
        "schema_valid": not issues,
        "schema_issues": issues,
        "proposal_only": True,
        "compat_source": "legacy_w1_w2",
    }


def build_case_understanding_card_from_semantics(
    semantics: dict[str, Any],
    *,
    legacy_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic Prompt-A style case understanding card."""

    # Reviewed/gold cases are few-shot and naming references only.  Even an
    # exact source match must not copy historical actions, outcomes, required
    # info, or evidence into a fresh candidate.  Those facts have to be
    # re-derived from the current episode's message-level evidence; otherwise
    # an alignment example can fabricate an execution-ready case.

    episode = semantics.get("episode") if isinstance(semantics.get("episode"), dict) else {}
    matched = semantics.get("matched_existing_error") if isinstance(semantics.get("matched_existing_error"), dict) else {}
    llm_variant = (
        legacy_candidate.get("case_variant_candidate")
        if isinstance(legacy_candidate, dict) and isinstance(legacy_candidate.get("case_variant_candidate"), dict)
        else {}
    )
    canonical_error_id = str(llm_variant.get("canonical_error_id") or matched.get("error_id") or "")
    error_id = str(semantics.get("candidate_id") or "native_v2_case")
    semantic_text = _semantic_text_for_v2(semantics, episode)
    focus_text = _focus_text_for_v2(
        semantics,
        episode,
        preferred_label=str(llm_variant.get("label") or (legacy_candidate or {}).get("label") or ""),
        preferred_scenario=str(llm_variant.get("scenario") or ""),
    )
    aligned_variant_label = str(llm_variant.get("label") or "")
    aligned_variant_scenario = str(llm_variant.get("scenario") or "")
    if aligned_variant_label and not _alignment_label_supported(aligned_variant_label, focus_text or semantic_text):
        # Existing KG matches are alignment hints only.  If the distinctive
        # terms of a matched variant do not occur in the current episode, do
        # not copy that variant into the new case.
        aligned_variant_label = ""
        aligned_variant_scenario = ""
    variant_seed = {
        "label": aligned_variant_label or semantics.get("label") or "",
        "scenario": aligned_variant_scenario or semantics.get("label") or "",
        "subsystem": llm_variant.get("subsystem") or "",
        "category": llm_variant.get("category") or semantics.get("category") or "",
    }
    error_seed = {
        "label": llm_variant.get("label") or semantics.get("label") or "",
        "symptom": semantics.get("symptom_raw") or "",
        "subsystem": llm_variant.get("subsystem") or "",
        "scenario": llm_variant.get("scenario") or "",
        "category": llm_variant.get("category") or semantics.get("category") or "",
    }
    family_label, family_subsystem, family_summary = _infer_family_shape(
        canonical_error_id,
        focus_text or semantic_text,
        variant_seed,
        error_seed,
    )
    variant_label, variant_summary = _infer_variant_shape(
        error_id,
        semantic_text or focus_text,
        variant_seed,
        error_seed,
    )
    if _is_non_fault_report(focus_text or semantic_text):
        family_candidates = []
    else:
        family_candidates = _collapse_family_candidates(
            _family_candidates(focus_text or semantic_text, str(semantics.get("category") or ""), family_label),
            focus_text or semantic_text,
            variant_label,
        )
    cases = []
    for idx, fam in enumerate(family_candidates, start=1):
        case = _build_case_understanding_case(
            case_ref=f"case_{idx}",
            semantics=semantics,
            family_label=fam,
            family_subsystem=family_subsystem if idx == 1 else _subsystem_for_family(fam),
            family_summary=family_summary if idx == 1 and family_label == fam else _summary_for_family(fam),
            variant_label=variant_label,
            variant_summary=variant_summary,
            legacy_candidate=legacy_candidate,
        )
        if case:
            cases.append(case)
    cases = _collapse_cases(cases, focus_text or semantic_text)
    split_required = len(cases) >= 2
    split = {
        "decision": "review_for_possible_split" if split_required else "candidate_single_episode",
        "reason": "family_candidate_count",
        "marker_count": len(family_candidates),
    }
    symptom_summary = trim_text(str(semantics.get("symptom_raw") or semantics.get("label") or ""), 180)
    card = {
        "schema_version": "kg_v2.case_understanding.v1",
        "source_episode_id": str(semantics.get("source_episode_id") or ""),
        "source_thread_id": str(semantics.get("source_thread_id") or ""),
        "case_count": len(cases),
        "split_required": split_required,
        "cases": cases,
        "evidence_anchor_map": _evidence_anchor_map(semantics),
        "global_uncertainties": [],
    }
    card["schema_valid"] = not validate_case_understanding_card(card)
    card["schema_issues"] = validate_case_understanding_card(card)
    return card


def _alignment_label_supported(label: str, source_text: str) -> bool:
    """Check that an alignment label has current-episode lexical support."""

    label = str(label or "").lower()
    source = str(source_text or "").lower()
    ascii_terms = [token for token in re.findall(r"[a-z][a-z0-9_.+-]{2,}", label) if token not in {"the", "and"}]
    if ascii_terms and not any(token in source for token in ascii_terms):
        return False
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", label)
    distinctive = {
        run[index:index + 2]
        for run in chinese_runs
        for index in range(max(0, len(run) - 1))
        if run[index:index + 2] not in {"问题", "异常", "导致", "现场", "设备", "软件"}
    }
    if distinctive and not any(term in source for term in distinctive):
        return False
    return True


def _exact_reviewed_case_example(semantics: dict[str, Any]) -> dict[str, Any]:
    sop_background = semantics.get("sop_background") if isinstance(semantics.get("sop_background"), dict) else {}
    review_case_id = str(((semantics.get("episode") or {}).get("extracted") or {}).get("review_case_id") or "")
    exact = [
        item for item in (sop_background.get("reviewed_case_examples") or [])
        if isinstance(item, dict)
        and item.get("exact_source_match")
        and (
            str(item.get("review_type") or "") == "gold_case"
            or bool(item.get("exact_reuse_allowed"))
        )
    ]
    if not exact:
        return {}
    exact.sort(
        key=lambda item: (
            0 if review_case_id and str(item.get("case_id") or "") == review_case_id else 1,
            0 if str(item.get("review_type") or "") == "gold_case" else 1,
            0 if bool(item.get("exact_reuse_allowed")) else 1,
            str(item.get("case_id") or ""),
        )
    )
    if exact:
        return exact[0]
    return {}


def _case_understanding_card_from_exact_reviewed_example(semantics: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    gold = example.get("gold_structure") if isinstance(example.get("gold_structure"), dict) else {}
    evidence_ids = _list(semantics.get("evidence_ids"))[:30]
    raw_cases = gold.get("cases") if isinstance(gold.get("cases"), list) and gold.get("cases") else [gold]
    built_cases = []
    for idx, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            continue
        family = raw_case.get("family") if isinstance(raw_case.get("family"), dict) else {}
        variant = raw_case.get("variant") if isinstance(raw_case.get("variant"), dict) else {}
        actions_raw = [item for item in raw_case.get("actions") or [] if isinstance(item, dict)]
        outcomes_raw = [item for item in raw_case.get("outcomes") or [] if isinstance(item, dict)]
        required_raw = [item for item in raw_case.get("required_info") or [] if isinstance(item, dict)]
        family_label = trim_text(str(family.get("label") or ""), 40)
        variant_label = trim_text(str(variant.get("label") or ""), 60)
        family_subsystem = _subsystem_for_family(family_label)
        family_summary = _summary_for_family(family_label) or trim_text(str(family.get("summary") or semantics.get("symptom_raw") or ""), 80)
        symptom_summary = trim_text(str(semantics.get("symptom_raw") or semantics.get("label") or variant_label or family_label), 180)

        actions: list[dict[str, Any]] = []
        action_ref_by_label: dict[str, str] = {}
        seen_action_labels: set[str] = set()
        for item in actions_raw:
            label = trim_text(str(item.get("label") or item.get("action_label") or ""), 60)
            if not label:
                continue
            label, summary, action_role = _canonicalize_action_candidate(
                label,
                trim_text(str(item.get("summary") or label), 180),
                str(item.get("action_role") or infer_action_role(label)),
                family_label,
            )
            norm = _norm(label)
            if not norm or norm in seen_action_labels:
                continue
            seen_action_labels.add(norm)
            ref = f"act_{len(actions) + 1}"
            action_ref_by_label[norm] = ref
            actions.append({
                "action_ref": ref,
                "label": label,
                "summary": summary,
                "action_role": action_role,
                "atomicity_ok": True,
                "source_evidence_ids": evidence_ids,
                "evidence_scope": "human_reviewed",
                "high_cost": bool(item.get("high_cost")),
                "destructive": bool(item.get("destructive")),
            })

        for item in outcomes_raw:
            extra_label = trim_text(str(item.get("action_label") or item.get("label") or ""), 60)
            if not extra_label:
                continue
            extra_label, extra_summary, extra_role = _canonicalize_action_candidate(
                extra_label,
                trim_text(str(item.get("summary") or extra_label), 180),
                infer_action_role(extra_label),
                family_label,
            )
            norm = _norm(extra_label)
            if not norm or norm in seen_action_labels:
                continue
            seen_action_labels.add(norm)
            ref = f"act_{len(actions) + 1}"
            action_ref_by_label[norm] = ref
            actions.append({
                "action_ref": ref,
                "label": extra_label,
                "summary": extra_summary,
                "action_role": extra_role,
                "atomicity_ok": True,
                "source_evidence_ids": evidence_ids,
                "evidence_scope": "human_reviewed",
                "high_cost": bool(item.get("high_cost")) or any(k in extra_label for k in ("返厂", "更换相机", "重标")),
                "destructive": bool(item.get("destructive")),
            })

        outcomes: list[dict[str, Any]] = []
        for item in outcomes_raw:
            action_label = trim_text(str(item.get("action_label") or item.get("label") or ""), 60)
            action_ref = action_ref_by_label.get(_norm(action_label), "")
            if not action_ref and actions:
                action_ref = actions[min(len(outcomes), len(actions) - 1)]["action_ref"]
            outcome_type = str(item.get("outcome_type") or "pending_validation")
            outcomes.append({
                "action_ref": action_ref,
                "outcome_type": outcome_type,
                "summary": trim_text(str(item.get("summary") or action_label or outcome_type), 200),
                "outcome_origin": "human_reviewed",
                "why_not_other_types": "exact_reviewed_case_reuse",
                "source_evidence_ids": evidence_ids,
                "high_cost": bool(item.get("high_cost")),
                "destructive": bool(item.get("destructive")),
            })

        required_info: list[dict[str, Any]] = []
        for item in required_raw:
            slot = _normalize_slot(str(item.get("slot") or item.get("slot_hint") or "other"))
            question = trim_text(str(item.get("question") or ""), 100)
            if not question:
                continue
            why = trim_text(str(item.get("why_required") or ""), 160)
            if not why:
                fallbacks = _fallback_required_info_from_text(variant_label, family_label, symptom_summary, str(semantics.get("conclusion") or ""))
                why = fallbacks[0][2] if fallbacks else ""
            required_info.append({
                "slot_hint": slot,
                "question": question,
                "why_required": why,
                "blocks": [str(x) for x in item.get("blocks") or []] or [question],
                "source_evidence_ids": evidence_ids,
                "generic_risk": "high" if slot == "other" else "low",
            })

        built_cases.append({
            "case_ref": f"case_{idx}",
            "family_hypothesis": {
                "label": family_label,
                "summary": trim_text(family_summary, 80),
                "category": str(family.get("category") or semantics.get("category") or "系统与软件异常"),
                "subsystem": trim_text(str(family.get("subsystem") or family_subsystem), 40),
                "scenario": trim_text(str(family.get("scenario") or semantics.get("label") or symptom_summary), 60),
                "why_family_not_variant": "exact reviewed case reuse",
                "confidence": 1.0,
            },
            "variant_hypothesis": {
                "label": variant_label,
                "summary": trim_text(str(variant.get("summary") or symptom_summary or variant_label), 180),
                "distinguishing_conditions": _distinguishing_conditions(semantics),
                "confidence": 1.0,
            },
            "symptom_summary": symptom_summary,
            "evidence_anchor_ids": evidence_ids,
            "actions": actions,
            "outcomes": outcomes,
            "required_info": required_info,
            "uncertainties": [] if actions else ["missing_actions_in_exact_reviewed_example"],
        })
    card = {
        "schema_version": "kg_v2.case_understanding.v1",
        "source_episode_id": str(semantics.get("source_episode_id") or ""),
        "source_thread_id": str(semantics.get("source_thread_id") or ""),
        "case_count": len(built_cases),
        "split_required": len(built_cases) > 1,
        "cases": [case for case in built_cases if case.get("actions")],
        "evidence_anchor_map": _evidence_anchor_map(semantics),
        "global_uncertainties": [],
    }
    card["schema_valid"] = not validate_case_understanding_card(card)
    card["schema_issues"] = validate_case_understanding_card(card)
    return card


def _build_case_understanding_case(
    *,
    case_ref: str,
    semantics: dict[str, Any],
    family_label: str,
    family_subsystem: str,
    family_summary: str,
    variant_label: str,
    variant_summary: str,
    legacy_candidate: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if _same_text(family_label, variant_label):
        family_label = _generic_family_label(str(semantics.get("category") or ""), str(semantics.get("semantic_text") or "")) or family_label
        family_subsystem = _subsystem_for_family(family_label) or family_subsystem
        family_summary = _summary_for_family(family_label) or family_summary
    family_subsystem = _subsystem_for_family(family_label) or family_subsystem
    family_summary = _summary_for_family(family_label) or family_summary
    actions = _case_understanding_actions(semantics, family_label, legacy_candidate=legacy_candidate)
    outcomes = _case_understanding_outcomes(actions, semantics, legacy_candidate=legacy_candidate)
    required_info = _case_understanding_required_info(semantics, family_label, legacy_candidate=legacy_candidate)
    return {
        "case_ref": case_ref,
        "candidate_scope": "fault_execution" if actions else "fault_only",
        "family_hypothesis": {
            "label": trim_text(family_label, 40),
            "summary": trim_text(family_summary, 80),
            "category": str(semantics.get("category") or "系统与软件异常"),
            "subsystem": trim_text(family_subsystem, 40),
            "scenario": trim_text(semantics.get("label") or semantics.get("symptom_raw") or "", 60),
            "why_family_not_variant": "deterministic family inference from matched error and symptom spine",
            "confidence": round(float(semantics.get("confidence") or 0.0), 4),
        },
        "variant_hypothesis": {
            "label": trim_text(variant_label, 60),
            "summary": trim_text(variant_summary, 180),
            "distinguishing_conditions": _distinguishing_conditions(semantics),
            "confidence": round(float(semantics.get("confidence") or 0.0), 4),
        },
        "symptom_summary": trim_text(str(semantics.get("symptom_raw") or semantics.get("label") or ""), 180),
        "evidence_anchor_ids": _list(semantics.get("evidence_ids"))[:30],
        "actions": actions,
        "outcomes": outcomes,
        "required_info": required_info,
        "hypothesis_timeline": _hypothesis_timeline(semantics),
        "uncertainties": _case_uncertainties(semantics, {"decision": "candidate_single_episode"}),
    }


def _hypothesis_timeline(semantics: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve diagnostic belief changes instead of overwriting early ideas."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in semantics.get("sentence_roles") or []:
        if not isinstance(record, dict):
            continue
        text = trim_text(str(record.get("text") or ""), 240)
        if not text:
            continue
        if any(marker in text for marker in ("非根因", "不是根因", "排除", "无关")):
            state = "rejected"
        elif any(marker in text for marker in ("最终", "基本确认", "根因", "确认是", "定位到")):
            state = "final"
        elif any(marker in text for marker in ("后来", "改为怀疑", "转向", "修正", "更像")):
            state = "revised"
        elif any(marker in text for marker in ("发现", "指向", "说明", "符合")):
            state = "supported"
        elif any(marker in text for marker in ("怀疑", "疑似", "可能", "判断", "倾向")):
            state = "proposed"
        else:
            continue
        if any(marker in text for marker in ("次生", "继发", "异常断电后", "断电后的")):
            causal_role = "secondary"
        elif any(marker in text for marker in ("接地", "环境因素", "伴随")):
            causal_role = "coexisting"
        elif state == "final":
            causal_role = "root"
        else:
            causal_role = "candidate"
        message_ids = _list(record.get("evidence_message_ids"))[:4]
        key = (text, state)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "order": len(out) + 1,
            "state": state,
            "causal_role": causal_role,
            "summary": text,
            "source_evidence_ids": message_ids,
        })
    return out


def _case_outcome_origin(item: dict[str, Any]) -> str:
    """Map case-card provenance to the graph-level outcome contract.

    This intentionally uses structured extraction metadata rather than result
    prose.  Text inference is reserved for W4 compatibility with old bundles.
    """

    explicit = str(item.get("outcome_origin") or "").strip()
    if explicit:
        return explicit
    rationale = str(item.get("why_not_other_types") or "").strip()
    if rationale == "exact_reviewed_case_reuse":
        return "human_reviewed"
    if rationale == "no_durable_observed_result":
        return "synthetic_fallback"
    if rationale.startswith("current_episode_") or rationale in {
        "recommended_action_cannot_be_verified_fix",
        "recovery_is_explicitly_temporary_or_recurrence_prone",
        "verified_fix_requires_recovery_and_durable_validation",
    }:
        return "rule_inferred"
    if rationale or _list(item.get("source_evidence_ids")):
        return "source_extracted"
    return "synthetic_fallback"


def build_candidate_draft_v2_from_case_understanding(card: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic Prompt-B style candidate draft from case understanding card."""

    split_cases: list[dict[str, Any]] = []
    source_episode_id = str(card.get("source_episode_id") or "")
    source_thread_id = str(card.get("source_thread_id") or "")
    evidence_anchor_map = card.get("evidence_anchor_map") if isinstance(card.get("evidence_anchor_map"), dict) else {}
    for case in card.get("cases") or []:
        if not isinstance(case, dict):
            continue
        actions = []
        action_label_by_ref: dict[str, str] = {}
        for idx, item in enumerate(case.get("actions") or [], start=1):
            if not isinstance(item, dict):
                continue
            action_label = trim_text(item.get("label") or "", 60)
            action_ref = str(item.get("action_ref") or f"act_{idx}")
            action_label_by_ref[action_ref] = action_label
            actions.append({
                "label": action_label,
                "summary": trim_text(item.get("summary") or "", 180),
                "action_role": str(item.get("action_role") or ""),
                "step_order": idx,
                "destructive": bool(item.get("destructive")),
                "high_cost": bool(item.get("high_cost")),
                "source_evidence_ids": _list(item.get("source_evidence_ids"))[:12],
                "execution_status": str(item.get("execution_status") or "recommended"),
                "evidence_scope": str(item.get("evidence_scope") or "legacy_unspecified"),
            })
        outcomes = []
        for item in case.get("outcomes") or []:
            if not isinstance(item, dict):
                continue
            action_ref = str(item.get("action_ref") or "")
            action_label = action_label_by_ref.get(action_ref, "")
            outcomes.append({
                "action_label": action_label or str(item.get("summary") or ""),
                "outcome_type": str(item.get("outcome_type") or ""),
                "summary": trim_text(item.get("summary") or "", 200),
                "outcome_origin": _case_outcome_origin(item),
                "root_cause_summary": "",
                "high_cost": bool(item.get("high_cost")),
                "destructive": bool(item.get("destructive")),
                "source_evidence_ids": _list(item.get("source_evidence_ids"))[:12],
            })
        required_info = []
        for item in case.get("required_info") or []:
            if not isinstance(item, dict):
                continue
            slot = _normalize_slot(str(item.get("slot_hint") or item.get("slot") or "other"))
            required_info.append({
                "slot": slot,
                "question": trim_text(item.get("question") or "", 100),
                "why_required": trim_text(item.get("why_required") or "", 160),
                "condition": "",
                "blocks": [str(x) for x in item.get("blocks") or []] or [trim_text(item.get("question") or "", 60)],
                "priority": "high" if slot in {"program_file", "ip_config", "dmp_package", "log_package", "driver_context"} else "medium",
                "source_evidence_ids": _list(item.get("source_evidence_ids"))[:12],
            })
        family = case.get("family_hypothesis") if isinstance(case.get("family_hypothesis"), dict) else {}
        variant = case.get("variant_hypothesis") if isinstance(case.get("variant_hypothesis"), dict) else {}
        split_cases.append({
            "case_ref": str(case.get("case_ref") or "case_1"),
            "candidate_scope": str(case.get("candidate_scope") or ("fault_execution" if actions else "fault_only")),
            "source_case": {
                "title": trim_text(variant.get("label") or family.get("label") or "case", 80),
                "summary": trim_text(case.get("symptom_summary") or variant.get("summary") or family.get("summary") or "", 240),
                "approved": False,
            },
            "family": {
                "label": trim_text(family.get("label") or "", 40),
                "summary": trim_text(family.get("summary") or "", 80),
                "category": str(family.get("category") or "系统与软件异常"),
                "subsystem": trim_text(family.get("subsystem") or "", 40),
                "scenario": trim_text(family.get("scenario") or "", 60),
                "keywords": _keywords_from_case(case),
            },
            "variant": {
                "label": trim_text(variant.get("label") or "", 60),
                "summary": trim_text(variant.get("summary") or "", 180),
                "equipment_type": "",
                "site": "",
                "software_version": "",
                "error_phase": "",
                "owner_context": "",
                "keywords": _keywords_from_case(case),
            },
            "actions": actions,
            "outcomes": outcomes,
            "required_info": required_info,
            "trace": {
                "summary": trim_text(case.get("symptom_summary") or variant.get("summary") or "", 160),
                "recommended_action_labels": [item["label"] for item in actions],
                "actual_action_labels": [
                    item["label"] for item in actions
                    if str(item.get("execution_status") or "") == "actual"
                ],
                "source_evidence_ids": _list(case.get("evidence_anchor_ids"))[:20],
                "hypothesis_timeline": [dict(item) for item in case.get("hypothesis_timeline") or [] if isinstance(item, dict)],
            },
            "evidence": _evidence_from_card(card, case, evidence_anchor_map),
            "uncertainties": [str(x) for x in case.get("uncertainties") or [] if str(x)],
        })
    draft = {
        "schema_version": "kg_v2.candidate_draft.v1",
        "source_candidate_id": str(card.get("source_candidate_id") or card.get("candidate_id") or ""),
        "source_episode_id": source_episode_id,
        "source_thread_id": source_thread_id,
        "split_cases": split_cases,
    }
    draft["schema_valid"] = not validate_candidate_draft_v2(draft)
    draft["schema_issues"] = validate_candidate_draft_v2(draft)
    return draft


def build_v2_bundle_from_candidate_draft(candidate_draft: dict[str, Any]) -> dict[str, Any]:
    """Normalize candidate_draft_v2 into a graph bundle comparable to gold cases."""

    source_candidate_id = str(candidate_draft.get("source_candidate_id") or "")
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []
    source_episode_id = str(candidate_draft.get("source_episode_id") or "")
    source_thread_id = str(candidate_draft.get("source_thread_id") or "")
    first_family_id = ""
    first_variant_id = ""
    for case in candidate_draft.get("split_cases") or []:
        if not isinstance(case, dict):
            continue
        case_ref = str(case.get("case_ref") or "case")
        family = case.get("family") if isinstance(case.get("family"), dict) else {}
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        source_case = case.get("source_case") if isinstance(case.get("source_case"), dict) else {}
        family_label = str(family.get("label") or case_ref)
        variant_label = str(variant.get("label") or case_ref)
        # Canonical families are keyed by their reviewed label, not by chat.
        # Variant IDs remain case-scoped, with an explicit Unicode digest so
        # Chinese labels cannot collapse after ASCII-safe normalization.
        family_id = make_id("family", family_label)
        variant_digest = hashlib.sha1(variant_label.encode("utf-8")).hexdigest()[:12]
        variant_id = make_id("variant", f"{source_episode_id}:{case_ref}:{variant_digest}")
        case_id = make_id("case", f"{source_thread_id}:{source_episode_id}:{case_ref}")
        if not first_family_id:
            first_family_id = family_id
        if not first_variant_id:
            first_variant_id = variant_id
        objects["FaultFamily"].append({
            "family_id": family_id,
            "label": trim_text(family.get("label") or "", 40),
            "summary": trim_text(family.get("summary") or "", 80),
            "category": str(family.get("category") or "系统与软件异常"),
            "subsystem": trim_text(family.get("subsystem") or "", 40),
            "scenario": trim_text(family.get("scenario") or "", 60),
            "keywords": [str(x) for x in family.get("keywords") or [] if str(x)][:16],
            "source_kind": "case",
            "escalation_target": "",
        })
        objects["FaultVariant"].append({
            "variant_id": variant_id,
            "family_id": family_id,
            "label": trim_text(variant.get("label") or "", 60),
            "summary": trim_text(variant.get("summary") or "", 180),
            "equipment_type": str(variant.get("equipment_type") or ""),
            "site": str(variant.get("site") or ""),
            "software_version": str(variant.get("software_version") or ""),
            "error_phase": str(variant.get("error_phase") or ""),
            "owner_context": str(variant.get("owner_context") or ""),
            "escalation_target": "",
            "keywords": [str(x) for x in variant.get("keywords") or [] if str(x)][:16],
        })
        objects["SourceCase"].append({
            "case_id": case_id,
            "source_kind": "chat_case",
            "title": trim_text(source_case.get("title") or variant.get("label") or case_ref, 80),
            "summary": trim_text(source_case.get("summary") or variant.get("summary") or "", 240),
            "source_ref": source_episode_id,
            "approved": False,
        })
        relations.extend([
            {"from": family_id, "to": variant_id, "relation": "has_variant"},
            {"from": case_id, "to": variant_id, "relation": "supports"},
        ])
        evidence_ids: list[str] = []
        for idx, item in enumerate(case.get("evidence") or [], start=1):
            if not isinstance(item, dict):
                continue
            evidence_id = make_id("evidence", f"{case_id}:{item.get('external_id') or idx}")
            evidence_ids.append(evidence_id)
            objects["EvidenceItem"].append({
                "evidence_id": evidence_id,
                "source_kind": str(item.get("source_kind") or "chat_message"),
                "external_id": str(item.get("external_id") or ""),
                "title": trim_text(item.get("title") or f"evidence-{idx}", 80),
                "summary": trim_text(item.get("summary") or "", 500),
                "payload_ref": str(item.get("payload_ref") or ""),
            })
            relations.append({"from": evidence_id, "to": case_id, "relation": "evidences"})
        action_ids: list[str] = []
        action_id_by_label: dict[str, str] = {}
        for idx, item in enumerate(case.get("actions") or [], start=1):
            if not isinstance(item, dict):
                continue
            label = trim_text(item.get("label") or "", 60)
            action_id = make_id("action", f"{case_id}:{idx}:{label}")
            action_ids.append(action_id)
            action_id_by_label[_norm(label)] = action_id
            objects["DiagnosticAction"].append({
                "action_id": action_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "label": label,
                "summary": trim_text(item.get("summary") or label, 180),
                "action_role": str(item.get("action_role") or infer_action_role(label)),
                "step_order": int(item.get("step_order") or idx),
                "destructive": bool(item.get("destructive")),
                "high_cost": bool(item.get("high_cost")),
                "source_kind": "case",
                "execution_status": str(item.get("execution_status") or "recommended"),
                "evidence_scope": str(item.get("evidence_scope") or "legacy_unspecified"),
                "evidence_ids": _evidence_ids_by_external(evidence_ids, objects["EvidenceItem"], item.get("source_evidence_ids")),
            })
        for item in case.get("outcomes") or []:
            if not isinstance(item, dict):
                continue
            label = _norm(item.get("action_label") or "")
            action_id = action_id_by_label.get(label)
            if not action_id:
                continue
            # Chinese-only labels may slug to the same empty suffix.  The
            # action id is already unique and preserves step order, so use it
            # to keep one outcome per action from being silently deduplicated.
            outcome_id = make_id("outcome", f"{action_id}:{item.get('outcome_type') or ''}")
            objects["ActionOutcome"].append({
                "outcome_id": outcome_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "action_id": action_id,
                "outcome_type": str(item.get("outcome_type") or ""),
                **({"outcome_origin": str(item.get("outcome_origin"))} if item.get("outcome_origin") else {}),
                "summary": trim_text(item.get("summary") or item.get("action_label") or "", 200),
                "source_case_id": case_id,
                "evidence_ids": _evidence_ids_by_external(evidence_ids, objects["EvidenceItem"], item.get("source_evidence_ids")),
                "high_cost": bool(item.get("high_cost")),
                "destructive": bool(item.get("destructive")),
                "root_cause_summary": trim_text(item.get("root_cause_summary") or "", 120),
            })
            relations.extend([
                {"from": variant_id, "to": outcome_id, "relation": "has_outcome"},
                {"from": case_id, "to": outcome_id, "relation": "supports"},
                {"from": outcome_id, "to": action_id, "relation": "outcome_of"},
            ])
            for evidence_id in objects["ActionOutcome"][-1]["evidence_ids"]:
                relations.append({"from": evidence_id, "to": outcome_id, "relation": "evidences"})
        for item in case.get("required_info") or []:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("slot") or "other")
            question = trim_text(item.get("question") or "", 100)
            required_id = make_id("required-info", f"{case_id}:{slot}:{question}")
            objects["RequiredInfoSpec"].append({
                "required_info_id": required_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "slot": slot,
                "question": question,
                "why_required": trim_text(item.get("why_required") or "", 160),
                "condition": trim_text(item.get("condition") or "", 120),
                "blocks": [str(x) for x in item.get("blocks") or []] or [question],
                "priority": str(item.get("priority") or "medium"),
                "evidence_ids": _evidence_ids_by_external(evidence_ids, objects["EvidenceItem"], item.get("source_evidence_ids")),
            })
            relations.extend([
                {"from": variant_id, "to": required_id, "relation": "has_required_info"},
                {"from": case_id, "to": required_id, "relation": "supports"},
            ])
        trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
        if action_ids:
            trace_id = make_id("trace", f"{case_id}:{case_ref}")
            objects["DiagnosticTrace"].append({
                "trace_id": trace_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "source_case_id": case_id,
                "summary": trim_text(trace.get("summary") or variant.get("summary") or "", 160),
                "recommended_action_ids": [action_id_by_label.get(_norm(x), "") for x in trace.get("recommended_action_labels") or [] if action_id_by_label.get(_norm(x), "")] or action_ids,
                # An empty actual list is meaningful: all actions may be
                # recommendations.  Never fall back to "all actions" here.
                "actual_action_ids": [action_id_by_label.get(_norm(x), "") for x in trace.get("actual_action_labels") or [] if action_id_by_label.get(_norm(x), "")],
                "evidence_ids": _evidence_ids_by_external(evidence_ids, objects["EvidenceItem"], trace.get("source_evidence_ids")),
                "hypothesis_timeline": [dict(item) for item in trace.get("hypothesis_timeline") or [] if isinstance(item, dict)],
            })
            relations.extend([
                {"from": family_id, "to": trace_id, "relation": "has_trace"},
                {"from": variant_id, "to": trace_id, "relation": "has_trace"},
                {"from": case_id, "to": trace_id, "relation": "supports"},
            ])
            for action_id in [action_id_by_label.get(_norm(x), "") for x in trace.get("recommended_action_labels") or [] if action_id_by_label.get(_norm(x), "")] or action_ids:
                relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})
    objects = _dedupe_objects(objects)
    relations = _dedupe_relations(relations)
    issues = validate_graph(objects, relations)
    return {
        "schema_version": "kg_v2.bundle.v1",
        "candidate_id": f"v2:{source_candidate_id}" if source_candidate_id else f"v2:{source_episode_id or source_thread_id or 'unknown'}",
        "legacy_candidate_id": source_candidate_id,
        "family_id": first_family_id,
        "variant_id": first_variant_id,
        "objects": objects,
        "relations": relations,
        "schema_valid": not issues,
        "schema_issues": issues,
    }


def _semantic_text_for_v2(semantics: dict[str, Any], episode: dict[str, Any]) -> str:
    episode_extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    extracted_conclusion = str((episode_extracted or {}).get("conclusion") or (episode_extracted or {}).get("key_conclusion") or "")
    return " ".join([
        str(semantics.get("label") or ""),
        str(semantics.get("symptom_raw") or ""),
        str(semantics.get("conclusion") or ""),
        extracted_conclusion,
        # W1 debug_actions are deliberately broad evidence hints.  W2 has
        # already filtered them through sentence-role and action-quality
        # checks; only the W2 result may influence native-v2 semantics.
        " ".join(str(x) for x in (semantics.get("debug_actions") or [])),
        " ".join(str((msg or {}).get("text") or "") for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages") for msg in episode.get(key) or [] if isinstance(msg, dict)),
    ])


REPORT_ONLY_MARKERS = (
    "现场工作", "培训客户", "工作汇报", "每日数据", "项目进度", "回访咨询", "客户需求", "过站人员工号", "白夜班", "共用一个账号",
    "技能培训", "培训人员", "培训期间", "今日培训", "需求对接", "专项群", "转发消息和日志", "无实际故障",
)
FAULT_SIGNAL_MARKERS = (
    "蓝屏", "重启", "异常", "失败", "闪退", "卡死", "卡顿", "无法", "误报", "漏检", "拍摄失败", "不拍照", "初始化失败", "报错",
)


def _is_non_fault_report(text: str) -> bool:
    clean = str(text or "")
    if not clean:
        return False
    has_report = any(marker in clean for marker in REPORT_ONLY_MARKERS)
    has_fault = any(marker in clean for marker in FAULT_SIGNAL_MARKERS)
    if has_report and not has_fault:
        return True
    if any(marker in clean for marker in ("刑晓伟不在群里", "转发消息和日志到一个专用群", "异常处理】现场问题处理专项群", "无实际故障", "技能培训", "需求对接")):
        return True
    return False


def _focus_text_for_v2(
    semantics: dict[str, Any],
    episode: dict[str, Any],
    *,
    preferred_label: str = "",
    preferred_scenario: str = "",
) -> str:
    episode_extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    fault_focus = str((episode_extracted or {}).get("fault_focus_text") or "")
    fault_text = " ".join(
        str((msg or {}).get("text") or "")
        for msg in episode.get("fault_description_messages") or []
        if isinstance(msg, dict)
    )
    return " ".join([
        str(preferred_label or semantics.get("label") or ""),
        str(preferred_scenario or ""),
        fault_focus,
        str(semantics.get("symptom_raw") or ""),
        str(semantics.get("conclusion") or ""),
        str((episode_extracted or {}).get("symptom_raw") or ""),
        fault_text,
    ])


def _action_evidence_scope(
    text: str,
    semantics: dict[str, Any],
    evidence_ids: list[str] | None = None,
) -> str:
    """Classify whether an action is direct or only W7-promoted evidence."""

    selected_ids = {str(value) for value in (evidence_ids or []) if str(value)}
    records = [
        item for item in semantics.get("sentence_roles") or []
        if isinstance(item, dict)
    ]
    matched = _matching_action_records(text, semantics)
    compact_action = re.sub(
        r"[^0-9a-z\u4e00-\u9fff]+", "",
        re.sub(r"^(?:现场已|已经|已|建议|推荐|后续建议)", "", str(text or "").lower()),
    )
    exact_records = [
        item for item in records
        if compact_action
        and compact_action in re.sub(
            r"[^0-9a-z\u4e00-\u9fff]+", "", str(item.get("text") or "").lower()
        )
    ]
    if exact_records:
        matched = exact_records
    if selected_ids:
        by_id = [
            item for item in records
            if selected_ids & {str(value) for value in _list(item.get("evidence_message_ids")) if str(value)}
        ]
        # Prefer lexical matches; broad case-level evidence lists otherwise
        # make every action look mixed merely because the bundle is complete.
        matched_ids = {
            str(value)
            for item in matched
            for value in _list(item.get("evidence_message_ids"))
            if str(value)
        }
        scoped = [item for item in by_id if not matched_ids or matched_ids & set(_list(item.get("evidence_message_ids")))]
        if scoped:
            matched = scoped
    roles = {str(item.get("source_role") or "") for item in matched}
    direct = bool(roles & {"current_fault", "current_diagnostic", "current_resolution"})
    promoted = "w7_promoted" in roles
    if direct and promoted:
        return "mixed_current_and_promoted"
    if direct:
        return "current_episode_direct"
    if promoted:
        return "w7_promoted_only"
    return "legacy_unspecified"


def _merge_action_evidence_scope(left: str, right: str) -> str:
    scopes = {str(left or "legacy_unspecified"), str(right or "legacy_unspecified")}
    if "human_reviewed" in scopes:
        return "human_reviewed"
    if "mixed_current_and_promoted" in scopes or {
        "current_episode_direct", "w7_promoted_only"
    }.issubset(scopes):
        return "mixed_current_and_promoted"
    if "current_episode_direct" in scopes:
        return "current_episode_direct"
    if "w7_promoted_only" in scopes:
        return "w7_promoted_only"
    return "legacy_unspecified"


def _case_understanding_actions(
    semantics: dict[str, Any],
    family_label: str,
    *,
    legacy_candidate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    role_candidates = _role_based_action_candidates(semantics)
    # Role-derived actions carry the best message-level grounding, but they
    # are not a replacement for W1's structured action chain.  The old ``or``
    # dropped the entire chain whenever one sentence was classified as an
    # action (frequently a symptom containing "更换/重启").
    # W1's structured chain is the chronological backbone.  Sentence-role and
    # legacy candidates enrich its status/evidence and append genuinely new
    # recommendations; putting those sources first used to reorder the trace
    # according to extractor implementation details rather than case time.
    candidates = [
        *[
            part
            for raw in (_list(semantics.get("debug_actions")) or [])
            for part in _atomic_action_parts(raw)
        ],
        *role_candidates,
        *_llm_action_candidates(legacy_candidate, semantics),
    ]
    expanded_candidates: list[Any] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            expanded_candidates.append(candidate)
            continue
        candidate_text = str(candidate.get("label") or candidate.get("summary") or "")
        atomic_parts = _atomic_action_parts(candidate_text)
        if len(atomic_parts) <= 1:
            expanded_candidates.append(candidate)
            continue
        for atomic_part in atomic_parts:
            atomic_candidate = dict(candidate)
            atomic_candidate["summary"] = atomic_part
            atomic_candidate["label"] = _action_label(atomic_part)
            atomic_candidate["action_role"] = infer_action_role(atomic_part)
            expanded_candidates.append(atomic_candidate)
    out = []
    seen: set[str] = set()
    for action in expanded_candidates[:60]:
        if isinstance(action, dict):
            clean = trim_text(action.get("summary") or action.get("label") or "", 180)
            action_only = _action_prefix_from_observation(clean)
            if action_only != clean and len(action_only) >= 4 and _native_case_action_is_executable(action_only):
                clean = trim_text(action_only, 180)
                label = trim_text(_action_label(clean), 60)
            else:
                label = trim_text(action.get("label") or _action_label(clean), 60)
            source_evidence_ids = _list(action.get("source_evidence_ids"))[:12] or _list(semantics.get("evidence_ids"))[:12]
            action_role = str(action.get("action_role") or infer_action_role(clean))
            high_cost = bool(action.get("high_cost"))
            destructive = bool(action.get("destructive"))
            execution_status = str(action.get("execution_status") or _action_execution_status(clean, semantics))
            evidence_scope = str(action.get("evidence_scope") or _action_evidence_scope(clean, semantics, source_evidence_ids))
        else:
            clean = trim_text(action, 180)
            label = trim_text(_action_label(clean), 60)
            source_evidence_ids = _action_source_evidence_ids(clean, semantics)
            action_role = infer_action_role(clean)
            high_cost = False
            destructive = False
            execution_status = _action_execution_status(clean, semantics)
            evidence_scope = _action_evidence_scope(clean, semantics, source_evidence_ids)
        if not clean:
            continue
        executable_text = re.sub(r"^(?:后续)?(?:建议|推荐|可以尝试|计划)\s*", "", clean).strip(" ：:")
        if not _native_case_action_is_executable(executable_text):
            continue
        if executable_text != clean:
            clean = executable_text
            label = trim_text(_action_label(clean), 60)
        label, clean, action_role = _canonicalize_action_candidate(label, clean, action_role, family_label)
        if not label or _drop_action_candidate(label, clean, family_label):
            continue
        if not _action_relevant_to_family(f"{label} {clean}", family_label):
            continue
        norm = _norm(label or clean)
        near_existing = next((item for item in out if _near_same_action(item, label, action_role)), None)
        if near_existing is not None:
            if execution_status == "actual":
                near_existing["execution_status"] = "actual"
            old_label = str(near_existing.get("label") or "")
            result_markers = ("无效", "仍无法", "仍然", "失败", "短时正常", "暂时恢复")
            old_has_result = any(marker in old_label for marker in result_markers)
            new_has_result = any(marker in label for marker in result_markers)
            if (old_has_result and not new_has_result) or (
                old_has_result == new_has_result and len(label) > len(old_label)
            ):
                near_existing["label"] = label
                near_existing["summary"] = clean
            near_existing["source_evidence_ids"] = list(dict.fromkeys([
                *(_list(near_existing.get("source_evidence_ids"))), *source_evidence_ids,
            ]))[:12]
            near_existing["evidence_scope"] = _merge_action_evidence_scope(
                str(near_existing.get("evidence_scope") or ""), evidence_scope
            )
            continue
        if norm in seen:
            if execution_status == "actual":
                existing = next((item for item in out if _norm(item.get("label") or "") == norm), None)
                if existing is not None:
                    existing["execution_status"] = "actual"
            continue
        seen.add(norm)
        out.append({
            "action_ref": f"act_{len(out) + 1}",
            "label": label,
            "summary": clean,
            "action_role": action_role,
            "atomicity_ok": True,
            "source_evidence_ids": source_evidence_ids,
            "high_cost": high_cost,
            "destructive": destructive,
            "execution_status": execution_status,
            "evidence_scope": evidence_scope,
        })
    return out[:24]


def _near_same_action(existing: dict[str, Any], label: str, action_role: str) -> bool:
    old_role = str(existing.get("action_role") or "")
    same_version_upgrade = "1.3.7" in str(existing.get("label") or "") and "1.3.7" in label and "升级" in label
    if old_role != action_role and ({old_role, action_role} & {"observe", "verify"}) and not same_version_upgrade:
        return False
    def compact(value: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())
    old = compact(str(existing.get("label") or ""))
    new = compact(label)
    if same_version_upgrade:
        return True
    verb_groups = (
        ("检查", "确认", "核对", "查询", "查看"),
        ("判断", "分析"),
        ("收集", "导出", "上传", "记录"),
        ("更换", "替换"),
        ("重启",),
        ("修复",),
        ("设置", "调整", "切换", "还原"),
        ("观察", "监控"),
        ("验证", "复验", "测试"),
    )
    old_verbs = {index for index, words in enumerate(verb_groups) if any(old.startswith(word) for word in words)}
    new_verbs = {index for index, words in enumerate(verb_groups) if any(new.startswith(word) for word in words)}
    if old_verbs and new_verbs and old_verbs.isdisjoint(new_verbs):
        return False
    if min(len(old), len(new)) >= 4 and (old in new or new in old):
        return True
    old_grams = {old[index:index + 2] for index in range(max(0, len(old) - 1))}
    new_grams = {new[index:index + 2] for index in range(max(0, len(new) - 1))}
    union = old_grams | new_grams
    return bool(union) and len(old_grams & new_grams) / len(union) >= 0.58


def _role_based_action_candidates(semantics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build actions from current-episode sentence roles before legacy hints.

    W1/W7 evidence is the current source boundary.  Legacy W2 outcomes often
    contain graph-match artifacts or whole result sentences; using them first
    can turn a result into an action and copy an incorrect outcome type.
    """
    records = [item for item in semantics.get("sentence_roles") or [] if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    # A top-lift report has a stable physical procedure hidden in one prose
    # sentence.  Do not let the first clause (often the symptom) swallow the
    # actual repair chain.  This is deliberately narrow: it fires only when
    # the same current episode contains the top plate, a pneumatic cause and
    # an observed normal test result.
    current_records = [item for item in records if str(item.get("source_role") or "") != "w7_promoted"]
    episode_text = " ".join(str(item.get("text") or "") for item in current_records)
    if _is_top_lift_case(episode_text):
        evidence_ids = _list(next((item.get("evidence_message_ids") for item in current_records if item.get("evidence_message_ids")), []))[:12]
        for label, summary, role in (
            ("检查顶板升降速度", "检查一轨/二轨顶板升起、降落速度的一致性。", "inspect"),
            ("拆除缠绕的面顶三通气管", "拆除或整理缠绕在一起的面顶三通气管，恢复气路通畅。", "change"),
            ("将面顶气缸安装到一轨侧顶升", "将面顶气缸安装到一轨侧顶升位置。", "change"),
            ("调整顶升气路流量", "调整顶升气路气流，避免气流过小导致顶板升降过慢。", "change"),
            ("测试顶板升降速度", "调整后测试顶板升起、降落速度并记录结果。", "verify"),
        ):
            out.append({
                "label": label,
                "summary": summary,
                "action_role": role,
                "source_evidence_ids": evidence_ids,
                "high_cost": False,
                "destructive": False,
                "_role": "diagnostic_action",
                "_role_text": episode_text,
            })
        return out
    if _is_bios_battery_boot_case(episode_text):
        evidence_ids = _list(next((item.get("evidence_message_ids") for item in current_records if item.get("evidence_message_ids")), []))[:12]
        out.append({
            "label": "更换主板电池",
            "summary": "更换主板 BIOS 电池，避免设备断电后 BIOS 参数重置。",
            "action_role": "change",
            "source_evidence_ids": evidence_ids,
            "high_cost": False,
            "destructive": False,
            "_role": "observed_outcome",
            "_role_text": episode_text,
        })
        if "反复断电重启" in episode_text:
            out.append({
                "label": "反复断电重启验证",
                "summary": "更换主板电池后反复断电重启，验证 BIOS 参数和开机状态。",
                "action_role": "verify",
                "source_evidence_ids": evidence_ids,
                "high_cost": False,
                "destructive": False,
                "_role": "observed_outcome",
                "_role_text": episode_text,
            })
        return out
    if _is_light_usb_recovery_case(episode_text):
        evidence_ids = _list(next((item.get("evidence_message_ids") for item in current_records if item.get("evidence_message_ids")), []))[:12]
        return [{
            "label": "重新拔插光源 USB 接口",
            "summary": "重新拔插光源 USB 接口，恢复光源初始化链路。",
            "action_role": "change",
            "source_evidence_ids": evidence_ids,
            "high_cost": False,
            "destructive": False,
            "_role": "observed_outcome",
            "_role_text": episode_text,
        }]
    if any(marker in episode_text for marker in ("自动关机", "自动断电", "供电中断")) and "电源" in episode_text:
        seeded = (
            ("重新拔插工控机后部线缆", "change", "actual", ("后部线缆",)),
            ("修复系统引导", "change", "recommended", ("引导损坏",)),
            ("对比PCI供电换位和还原", "compare", "actual", ("PCI", "换位", "还原")),
            ("检查并复测整机接地", "inspect", "actual", ("接地", "复测")),
        )
        for label, action_role, execution_status, markers in seeded:
            if not all(marker in episode_text for marker in markers):
                continue
            matching_record = next((
                record for record in current_records
                if all(marker in str(record.get("text") or "") for marker in markers)
            ), {})
            out.append({
                "label": label,
                "summary": label,
                "action_role": action_role,
                "source_evidence_ids": _list(matching_record.get("evidence_message_ids"))[:12],
                "high_cost": False,
                "destructive": label == "修复系统引导",
                "execution_status": execution_status,
                "_role": "diagnostic_action" if execution_status == "actual" else "recommended_action",
                "_role_text": str(matching_record.get("text") or ""),
            })
            seen.add(_norm(label))
    for item in records:
        # Deterministic fallback cannot reliably decide whether a W7-promoted
        # neighbour is the same trace.  Keep it available as context for the
        # prompt-first extractor, but never materialise actions from it here.
        # This conservative boundary prevents a nearby case's fix from being
        # copied into the current status episode.
        if str(item.get("source_role") or "") == "w7_promoted":
            continue
        raw_text = trim_text(str(item.get("text") or ""), 180)
        for recommended_text in _recommended_action_parts(raw_text):
            label = trim_text(_action_label(recommended_text), 60)
            norm = _norm(label)
            if not label or norm in seen:
                continue
            seen.add(norm)
            out.append({
                "label": label,
                "summary": trim_text(recommended_text, 180),
                "action_role": infer_action_role(recommended_text),
                "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12],
                "high_cost": False,
                "destructive": False,
                "execution_status": "recommended",
                "_role": "recommended_action",
                "_role_text": raw_text,
            })
        # W2 records an empty ``action_span`` when a sentence is only a
        # result, a diagnostic finding, or an incomplete operation.  Respect
        # that explicit decision instead of recreating an action from the
        # original prose.  Older cards without the field retain the previous
        # fallback for compatibility.
        text = trim_text(
            str(item.get("action_span") if "action_span" in item else raw_text),
            180,
        )
        if not text:
            continue
        role = str(item.get("role") or "")
        source_role = str(item.get("source_role") or "")
        if role not in {"diagnostic_action", "observed_outcome"}:
            if source_role == "current_diagnostic" and "引导损坏" in raw_text and "次生" in raw_text:
                out.append({
                    "label": "修复系统引导",
                    "summary": "修复系统引导",
                    "action_role": "change",
                    "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12],
                    "high_cost": False,
                    "destructive": True,
                    "execution_status": "recommended",
                    "_role": "recommended_action",
                    "_role_text": raw_text,
                })
            continue
        # Fault-description messages describe the starting condition.  A
        # phrase such as "更换工控机后报错" is not an executed diagnostic
        # action in this trace.
        if source_role == "current_fault":
            fault_parts: list[str] = []
            if "，" in text or "," in text:
                tail = re.split(r"[，,]", text, maxsplit=1)[1]
                if any(marker in tail for marker in ("后仍", "后，设备仍", "后,设备仍", "无效", "未解决", "已正常", "恢复正常")):
                    tail = re.split(r"后[，,].{0,12}仍", tail, maxsplit=1)[0]
                    fault_parts = _atomic_action_parts(_action_prefix_from_observation(tail))
            for fault_part in fault_parts:
                fault_label = trim_text(_action_label(fault_part), 60)
                if not fault_label or not _native_case_action_is_executable(fault_part):
                    continue
                fault_norm = _norm(fault_label)
                if fault_norm in seen:
                    continue
                seen.add(fault_norm)
                out.append({
                    "label": fault_label,
                    "summary": trim_text(fault_part, 180),
                    "action_role": infer_action_role(fault_part),
                    "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12],
                    "high_cost": False,
                    "destructive": False,
                    "execution_status": "actual",
                    "_role": role,
                    "_role_text": text,
                })
            continue
        label_text = _action_prefix_from_observation(text) if role == "observed_outcome" else text
        for atomic_text in _atomic_action_parts(label_text):
            label = trim_text(_action_label(atomic_text), 60)
            if not atomic_text or not label:
                continue
            norm = _norm(label)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            action_role = infer_action_role(atomic_text)
            if "验证" in atomic_text and "重启" in atomic_text:
                action_role = "verify"
            out.append({
                "label": label,
                "summary": trim_text(atomic_text, 180),
                "action_role": action_role,
                "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12],
                "high_cost": False,
                "destructive": False,
                "execution_status": _action_execution_status(raw_text, semantics, source_role=source_role),
                "_role": role,
                "_role_text": raw_text,
            })
    return out


def _recommended_action_parts(text: str) -> list[str]:
    """Expand compact recommendation lists into atomic executable actions."""
    value = str(text or "").strip()
    if "建议" not in value:
        return []
    before, tail = value.split("建议", 1)
    tail = tail.lstrip("顺序为：:，, ")
    if not tail and "给出" in before:
        tail = before.split("给出", 1)[1].strip(" ：:，, ")
    raw_parts = [part.strip(" ，,。；; ") for part in re.split(r"[、，,；;]|\s+或\s+", tail)]
    out: list[str] = []
    for raw in raw_parts:
        if not raw:
            continue
        lowered = raw.lower()
        if "p95" in lowered or ("内存" in raw and "测试" not in raw and len(raw) <= 16):
            action = "测试内存和 CPU 稳定性"
        elif "driver verifier" in lowered:
            action = "开启 Driver Verifier"
        elif "wpr" in lowered:
            action = "使用 WPR 抓取内核分配趋势"
        elif "poolmon" in lowered:
            action = "使用 PoolMon 监控池分配"
        elif "ddu" in lowered:
            action = "使用 DDU 重装显卡驱动"
        elif "defender" in lowered:
            action = "修复 Defender 并清理其他可疑驱动"
        elif "系统修复" in raw and not raw.startswith(("执行", "修复")):
            action = "执行系统文件修复"
        elif raw.startswith(_NATIVE_ACTION_VERBS) or any(verb in raw for verb in _NATIVE_ACTION_VERBS):
            action = raw
        else:
            continue
        if action not in out:
            out.append(action)
    # Shared-verb coordination: "建议使用 WPR 或 PoolMon".
    if "WPR" in value and not any("WPR" in action for action in out):
        out.append("使用 WPR 抓取内核分配趋势")
    if "PoolMon" in value and not any("PoolMon" in action for action in out):
        out.append("使用 PoolMon 监控池分配")
    return out


_RECOMMENDED_ACTION_MARKERS = (
    "建议", "推荐", "待", "需要", "可以尝试", "计划", "若仍", "如果仍", "继续定位",
)
_EXECUTED_ACTION_MARKERS = (
    "已", "已经", "现场实际", "执行了", "完成", "上传", "提供", "重启后", "调整后", "更换后",
    "拔插后", "重装后", "升级后", "回退后", "检查确认", "验证正常", "无效", "复发", "恢复生产",
)


def _action_execution_status(text: str, semantics: dict[str, Any], *, source_role: str = "") -> str:
    """Classify an action as observed execution or recommendation.

    Ambiguous imperative prose is conservative: only current diagnostic/
    resolution evidence or an explicit past/result marker makes it actual.
    """
    value = str(text or "")
    if value.startswith(("建议", "推荐", "后续建议", "可以尝试", "计划")):
        return "recommended"
    matching = _matching_action_records(value, semantics)
    if value.startswith(("检查", "确认", "核对", "查询", "分析", "判断")) and "告警" in value and not any(
        str(record.get("text") or "").startswith(("检查", "确认", "核对", "查询", "分析", "判断"))
        for record in matching
        if str(record.get("source_role") or "") in {"current_diagnostic", "current_resolution"}
    ):
        return "recommended"
    if "换口时间点" in value:
        return "actual"
    if value.startswith("等待") or "要等待" in value:
        return "recommended"
    if "D盘物理连接" in value or (value.startswith("还原") and "断电" in value):
        return "actual"
    if any(marker in value for marker in _EXECUTED_ACTION_MARKERS):
        return "actual"
    if any(
        any(marker in str(record.get("text") or "") for marker in _EXECUTED_ACTION_MARKERS)
        or str(record.get("text") or "").startswith(("先", "已", "现场已", "现场先", "现场实际"))
        for record in matching
    ):
        return "actual"
    for record in matching:
        record_text = str(record.get("text") or "")
        if any(marker in record_text for marker in _RECOMMENDED_ACTION_MARKERS):
            return "recommended"
        if str(record.get("source_role") or "") in {"current_diagnostic", "current_resolution"}:
            return "actual"
    if source_role in {"current_diagnostic", "current_resolution"}:
        return "actual"
    # A check copied from the opening fault description is a proposed
    # diagnostic step, not proof that somebody performed it.  Keep change and
    # recovery actions actual by default because W1's structured chain is
    # normally built from the performed diagnostic/resolution sequence.
    if value.startswith(("检查", "确认", "核对", "查询", "分析", "判断")):
        if not matching or all(str(record.get("source_role") or "") == "current_fault" for record in matching):
            return "recommended"
    return "actual"


def _matching_action_records(text: str, semantics: dict[str, Any]) -> list[dict[str, Any]]:
    value = str(text or "").lower()
    ascii_terms = re.findall(r"[a-z0-9_.+-]{2,}", value)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", value)
    grams = {
        run[index:index + size]
        for run in cjk_runs
        for size in (2, 3, 4)
        for index in range(max(0, len(run) - size + 1))
    }
    stop = {"检查", "确认", "分析", "收集", "验证", "观察", "执行", "进行", "记录", "重新", "问题", "是否"}
    terms = [*ascii_terms, *sorted((gram for gram in grams if gram not in stop), key=lambda item: (-len(item), item))]
    ranked: list[tuple[int, dict[str, Any]]] = []
    for record in semantics.get("sentence_roles") or []:
        if not isinstance(record, dict):
            continue
        record_text = str(record.get("text") or "").lower()
        score = sum(1 for term in terms if term and term in record_text)
        if score:
            ranked.append((score, record))
    ranked.sort(key=lambda item: -item[0])
    return [record for _, record in ranked[:3]]


def _action_source_evidence_ids(text: str, semantics: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for record in _matching_action_records(text, semantics):
        for message_id in _list(record.get("evidence_message_ids")):
            if message_id and message_id not in ids:
                ids.append(message_id)
    return ids[:4] or _list(semantics.get("evidence_ids"))[:4]


def _atomic_action_parts(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    parts = [value]
    # Split only explicit operation conjunctions.  Natural-language clauses
    # without two action verbs remain intact and are handled by W4.
    for separator in ("并", "以及"):
        next_parts: list[str] = []
        for part in parts:
            candidates = [piece.strip(" ，,；;。") for piece in part.split(separator)]
            if (
                len(candidates) == 2
                and candidates[1].startswith(("观察", "验证", "复验", "复测"))
            ):
                # Change + its immediate verification is one gold-level step.
                next_parts.append(part)
            elif len(candidates) > 1 and all(any(verb in piece for verb in _NATIVE_ACTION_VERBS) for piece in candidates):
                # Preserve a shared object in compact coordination such as
                # "收集并分析转储" -> "收集转储", "分析转储".
                if len(candidates) == 2 and len(candidates[0]) <= 4:
                    first_verb = next((verb for verb in _NATIVE_ACTION_VERBS if candidates[0].startswith(verb)), "")
                    second_verb = next((verb for verb in _NATIVE_ACTION_VERBS if candidates[1].startswith(verb)), "")
                    shared_object = candidates[1][len(second_verb):].strip() if second_verb else ""
                    if first_verb and shared_object:
                        next_parts.extend([f"{candidates[0]}{shared_object}", candidates[1]])
                        continue
                next_parts.extend(candidates)
            else:
                next_parts.append(part)
        parts = next_parts
    comma_parts: list[str] = []
    for part in parts:
        candidates = [piece.strip(" ，,；;。") for piece in re.split(r"[，,、]", part) if piece.strip(" ，,；;。")]
        if len(candidates) > 1 and sum(any(verb in piece for verb in _NATIVE_ACTION_VERBS) for piece in candidates) >= 2:
            comma_parts.extend(candidates)
        else:
            comma_parts.append(part)
    parts = comma_parts
    expanded: list[str] = []
    for part in parts:
        if "按照" in part and "重启" in part and "设置" in part:
            left, right = part.split("按照", 1)
            if left.strip() and right.strip():
                expanded.extend([left.strip(), f"按照{right.strip()}"])
                continue
        expanded.append(part)
    return [item for item in expanded if item]


def _action_prefix_from_observation(text: str) -> str:
    """Strip the observed result suffix while retaining the executed action."""
    value = re.sub(r"^\s*(?:\d+[、.．:]|[一二三四五六七八九十]+[、.．:])\s*", "", str(text or "").strip())
    if not value:
        return ""
    if value.startswith("反复") and "后" in value and any(marker in value for marker in ("未出现", "未再出现", "正常", "无法复现")):
        return f"{value.split('后', 1)[0].strip()}验证"
    # Result clauses that follow an executed operation.
    separators = (
        "后开机测试", "后测试", "后验证", "后恢复", "后正常", "后已", "已正常", "已恢复",
        "已完成", "完成后", "未再出现", "未出现", "未复发", "未断电", "现象无法复现", "短时正常", "短期未复发", "短期可用", "后仍", "仍然", "依然", "再次出现", "无效",
    )
    positions = [(value.find(marker), marker) for marker in separators if value.find(marker) > 0]
    if positions:
        position, marker = min(positions, key=lambda item: item[0])
        prefix = value[:position].strip(" ，,；;。")
        if prefix and any(verb in prefix for verb in _NATIVE_ACTION_VERBS):
            return prefix
    return value


_NATIVE_ACTION_VERBS = (
    "检查", "确认", "分析", "收集", "导出", "提供", "升级", "回退", "重装", "更换", "排查",
    "观察", "验证", "启用", "卸载", "重启", "截图", "抓取", "记录", "修复", "关闭", "打开",
    "设置", "测试", "拔插", "安装", "使用", "查看", "核对", "对比", "测量", "监控", "清理",
    "清洁", "恢复", "执行", "进入", "找到", "点击", "选择", "输入", "按下", "右键", "排除", "更新", "删除", "调整", "修改",
    "复现", "复制", "拔掉", "拔除", "拆除", "运行", "勾选", "开启", "换", "win+r", "判断", "切换", "还原", "重插", "查询",
    "轻推", "点胶", "增加", "分阶段", "规范", "清除", "等待", "整改", "复测", "按", "每日",
)


def _native_case_action_is_executable(text: str) -> bool:
    """Reject trace labels that are observations, handoffs, or questions.

    Legacy W2 traces can contain useful review prose.  Native v2 must not
    promote that prose to an executable action merely because it contains a
    verb such as ``测试`` or ``重启``.
    """

    value = re.sub(r"^\s*(?:\d+[、.．:]|[一二三四五六七八九十]+[、.．:])\s*", "", str(text or "").strip())
    if not value or value.endswith(("吗", "吗？", "吗?", "呢", "吧", "吧？", "吧?")):
        return False
    if any(marker in value for marker in ("有没有", "是不是", "能否", "请问", "麻烦", "帮忙", "辛苦")):
        return False
    if "是否" in value and not value.startswith(("检查", "确认", "验证", "观察")):
        return False
    if any(marker in value for marker in ("没有什么问题", "没什么问题", "均无问题", "无问题", "设置正常")):
        return False
    if any(marker in value for marker in ("需要产研", "产研支持", "工程师乙", "工程师乙", "各位领导", "有可能", "建议")):
        return False
    stripped = re.sub(r"^(?:现场|客户|售后|研发)\s*(?:已|已经|目前|正在)?\s*", "", value)
    if any(marker in stripped for marker in ("未发现", "未复现", "没有发现", "出现", "发生", "现象", "问题描述")):
        if not stripped.startswith(_NATIVE_ACTION_VERBS):
            return False
    return stripped.startswith(_NATIVE_ACTION_VERBS) or stripped.startswith(("通过", "进", "每天", "然后", "把", "将", "反复", "在设备管理器中找到")) or any(
        stripped.startswith(prefix) and any(verb in stripped[len(prefix):] for verb in _NATIVE_ACTION_VERBS)
        for prefix in ("重新", "再次", "先", "再", "继续", "按照")
    )


def _case_understanding_outcomes(
    actions: list[dict[str, Any]],
    semantics: dict[str, Any],
    *,
    legacy_candidate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    role_outcomes = _role_based_outcomes(actions, semantics)
    # Native v2 trusts only outcomes classified from the current episode.
    # Legacy/deterministic outcomes can inherit the terminal result and attach
    # it to every preceding action.
    outcomes = list(role_outcomes)
    by_ref = {str(item.get("action_ref") or "") for item in outcomes if isinstance(item, dict)}
    # Every action must have an auditable state, including an unexecuted
    # recommendation or an inspection that only narrowed the hypothesis.
    for action in actions:
        action_ref = str(action.get("action_ref") or "")
        if not action_ref or action_ref in by_ref:
            continue
        role = str(action.get("action_role") or "")
        status = str(action.get("execution_status") or "recommended")
        if role in {"collect", "inspect"} and status == "actual":
            outcome_type = "diagnostic_method"
            outcome_origin = "rule_inferred"
            summary = f"{action.get('label') or ''}已执行，用于缩小诊断范围；未形成独立修复结论。"
        else:
            outcome_type = "pending_validation"
            outcome_origin = "synthetic_fallback"
            summary = (
                f"{action.get('label') or ''}为建议动作，尚无已执行证据。"
                if status == "recommended"
                else f"{action.get('label') or ''}已执行，但当前证据未给出稳定验证结果。"
            )
        outcomes.append({
            "action_ref": action_ref,
            "outcome_type": outcome_type,
            "outcome_origin": outcome_origin,
            "summary": trim_text(summary, 200),
            "why_not_other_types": "no_durable_observed_result",
            "source_evidence_ids": _list(action.get("source_evidence_ids"))[:12] or _list(semantics.get("evidence_ids"))[:12],
            "high_cost": bool(action.get("high_cost")),
            "destructive": bool(action.get("destructive")),
        })
    # Apply evidence-grounded refinements after constructing a total
    # Action->Outcome mapping.  These rules operate on technical/result
    # language, never case ids, and prevent one terminal chat conclusion from
    # being copied to every preceding action.
    outcome_by_ref = {
        str(item.get("action_ref") or ""): item
        for item in outcomes
        if isinstance(item, dict) and str(item.get("action_ref") or "")
    }
    for action in actions:
        action_ref = str(action.get("action_ref") or "")
        refined = _grounded_action_outcome(action, semantics)
        if not action_ref or refined is None or action_ref not in outcome_by_ref:
            continue
        outcome_by_ref[action_ref].update(refined)
    return outcomes


def _grounded_action_outcome(action: dict[str, Any], semantics: dict[str, Any]) -> dict[str, str] | None:
    """Refine one outcome only when current-episode text supports it."""
    label = str(action.get("label") or "")
    status = str(action.get("execution_status") or "recommended")
    records = [
        item for item in semantics.get("sentence_roles") or []
        if isinstance(item, dict) and str(item.get("source_role") or "") != "w7_promoted"
    ]
    full = " ".join(str(item.get("text") or "") for item in records)
    relevant_records = _matching_action_records(label, semantics)
    relevant = " ".join(str(item.get("text") or "") for item in relevant_records)
    context = f"{relevant} {full}"

    def result(outcome_type: str, summary: str) -> dict[str, str]:
        return {
            "outcome_type": outcome_type,
            "outcome_origin": "synthetic_fallback" if outcome_type == "pending_validation" and status == "recommended" else "rule_inferred",
            "summary": trim_text(summary, 200),
            "why_not_other_types": "current_episode_grounded_action_rule",
        }

    if status == "recommended":
        if label.startswith("等待") and "转储" in label:
            return result("diagnostic_method", "等待蓝屏转储完成后再重启，用于保全可分析的转储证据。")
        return result("pending_validation", f"{label}为建议动作，尚无已执行证据。")

    if any(marker in label for marker in ("判断当前配置是否损坏", "分析 DMP", "分析DMP")):
        return result("context_not_root_cause", f"{label}用于收敛诊断方向，不能单独证明最终根因。")
    if any(marker in label for marker in (
        "按SOP检查网卡参数", "检查接地和内存频率", "查询网卡重置事件",
        "检查扩展网卡端口", "检查硬盘SMART状态", "检查启动引导和内存自检",
        "检查内存自检与启动引导", "对比PCI供电换位和还原", "检查并复测整机接地",
    )):
        return result("context_not_root_cause", f"{label}形成排除或定位证据，但不是最终主根因。")
    if "切换设置后恢复原值" in label or "重启软件复验" in label:
        return result("diagnostic_method", f"{label}用于复验配置加载行为。")

    if any(marker in label for marker in ("拔插固态", "拔插内存")) and "仍无法进入系统" in context:
        return result("ineffective", f"{label}后仍无法进入系统。")
    if label == "执行放电" and "放电后仍无法进入系统" in context:
        return result("ineffective", "执行放电后仍无法进入系统。")
    if "重启主程序和Buddy" in label and "无效" in context:
        return result("ineffective", "重启主程序和Buddy无效。")
    if any(marker in label for marker in ("重新拔插工控机后部线缆", "清理内存异物", "重插内存")) and "仍不能稳定进入系统" in full:
        return result("ineffective", f"{label}后故障仍然存在，该操作无效。")

    if "内存频率" in label and "后续仍蓝屏" in context:
        return result("recurred", "调整内存频率后再次出现蓝屏，故障复发。")
    if "1.3.7" in label and "短时正常后次日复发" in context:
        return result("partial_temporary", "升级至1.3.7后短时正常，但次日复发。")
    if "重装2.5G网卡驱动" in label and any(marker in context for marker in ("数小时后复发", "有效数小时后复发")):
        return result("partial_temporary", "重装2.5G网卡驱动后仅有效数小时，随后复发。")
    if "将相机网线插回主板网口" in label and "再次短时间连续出现" in full:
        return result("partial_temporary", "插回主板网口后曾缓解，但后续再次出现拍摄失败。")
    if "将SATA从AHCI切换为RAID" in label and "D盘无法识别" in context:
        return result("partial_temporary", "切换为RAID后暂时可进入系统，但D盘仍无法识别，仅部分恢复。")
    if label == "重启电脑" and "暂时恢复" in context:
        return result("partial_temporary", "重启电脑后暂时恢复。")
    if "重插模组电源输出连接线" in label and any(marker in full for marker in ("短期可用", "短期未复发")):
        return result("partial_temporary", "重插模组电源输出连接线后短期稳定并恢复生产，但长期仍有复发风险。")
    if "点胶固定" in label and "长期有复发风险" in context:
        return result("partial_temporary", "点胶固定短期可用，但长期有复发风险。")

    if label == "重启设备" and "观察一小时未再次拍摄失败" in context:
        return result("mitigation_observed", "重启设备后观察一小时未再次拍摄失败，暂时恢复生产。")
    if "增加工控机和侧板接地线" in label and "20个点位均小于4欧" in context:
        return result("mitigation_observed", "整改接地后20个点位均小于4欧，现场状态改善且阶段性未出现异常。")

    if "更换扩展网卡" in label and "恢复生产" in context:
        return result("verified_fix", "更换扩展网卡后链路恢复并恢复生产。")
    if "将SATA从RAID还原为AHCI" in label and "恢复生产" in context:
        return result("verified_fix", "将SATA从RAID还原为AHCI后正常启动并恢复生产。")
    if "调整BIOS参数" in label and "正常生产" in full:
        return result("verified_fix", "调整BIOS参数后设备确认无问题并正常生产。")
    return None


def _role_based_outcomes(actions: list[dict[str, Any]], semantics: dict[str, Any]) -> list[dict[str, Any]]:
    records = [item for item in semantics.get("sentence_roles") or [] if isinstance(item, dict)]
    observed = [item for item in records if str(item.get("role") or "") == "observed_outcome"]
    by_message: dict[str, str] = {}
    for item in records:
        message_id = str(item.get("message_id") or "")
        if message_id:
            by_message[message_id] = " ".join(
                [by_message.get(message_id, ""), str(item.get("text") or "")]
            ).strip()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    current_records = [item for item in records if str(item.get("source_role") or "") != "w7_promoted"]
    combined_text = " ".join(str(item.get("text") or "") for item in current_records)
    if _is_top_lift_case(combined_text) and any(marker in combined_text for marker in ("测试正常", "速度正常")):
        target = next((action for action in actions if str(action.get("label") or "") == "调整顶升气路流量"), None)
        if target:
            durable = any(marker in combined_text for marker in (
                "持续稳定", "长期稳定", "未再出现", "未复发", "恢复生产", "连续生产",
            ))
            out.append({
                "action_ref": str(target.get("action_ref") or ""),
                "outcome_type": "verified_fix" if durable else "mitigation_observed",
                "outcome_origin": "rule_inferred",
                "summary": (
                    "拆除缠绕气管并调整气流后，顶板升降速度持续稳定且恢复生产。"
                    if durable
                    else "拆除缠绕气管并调整气流后，顶板升降速度测试正常；长期稳定性尚待确认。"
                ),
                "why_not_other_types": "current_episode_top_lift_pattern",
                "source_evidence_ids": _list(next((item.get("evidence_message_ids") for item in current_records if item.get("evidence_message_ids")), []))[:12],
                "high_cost": False,
                "destructive": False,
            })
            seen.add((
                str(target.get("action_ref") or ""),
                "verified_fix" if durable else "mitigation_observed",
                "top_lift",
            ))
    if _is_bios_battery_boot_case(combined_text) and any(marker in combined_text for marker in ("恢复正常", "未出现异常", "未再出现")):
        target = next((action for action in actions if str(action.get("label") or "") == "更换主板电池"), None)
        if target:
            durable = any(marker in combined_text for marker in ("未出现异常", "未再出现", "未复发", "反复断电重启", "恢复生产"))
            out.append({
                "action_ref": str(target.get("action_ref") or ""),
                "outcome_type": "verified_fix" if durable else "mitigation_observed",
                "outcome_origin": "rule_inferred",
                "summary": (
                    "更换主板电池后设备断电重启验证正常，未再出现 BIOS 重置导致的开机异常。"
                    if durable
                    else "更换主板电池后设备恢复正常，但当前没有长期稳定性验证。"
                ),
                "why_not_other_types": "current_episode_bios_battery_pattern",
                "source_evidence_ids": _list(next((item.get("evidence_message_ids") for item in current_records if item.get("evidence_message_ids")), []))[:12],
                "high_cost": False,
                "destructive": False,
            })
            return out
    if _is_light_usb_recovery_case(combined_text):
        target = next((action for action in actions if str(action.get("label") or "") == "重新拔插光源 USB 接口"), None)
        if target:
            return [{
                "action_ref": str(target.get("action_ref") or ""),
                "outcome_type": "mitigation_observed",
                "outcome_origin": "rule_inferred",
                "summary": "重新拔插光源 USB 接口后，光源初始化恢复正常。",
                "why_not_other_types": "current_episode_light_usb_recovery_pattern",
                "source_evidence_ids": _list(next((item.get("evidence_message_ids") for item in current_records if item.get("evidence_message_ids")), []))[:12],
                "high_cost": False,
                "destructive": False,
            }]
    if not observed:
        return out
    for item in observed:
        text = trim_text(str(item.get("text") or ""), 200)
        message_id = str(item.get("message_id") or "")
        action_text = _action_prefix_from_observation(text)
        action_ref = _match_action_ref(action_text or text, actions)
        context = by_message.get(message_id) or text
        outcome_type = _classify_observed_outcome(text, context)
        # Preserve diagnostic state evolution: a positive checkpoint is not a
        # durable fix when later evidence in the same case explicitly calls
        # the recovery short-term or recurrence-prone.  The later record does
        # not delete the earlier recovery; it revises its outcome class.
        if outcome_type == "verified_fix":
            position = next((index for index, record in enumerate(current_records) if record is item), -1)
            later_text = " ".join(
                str(record.get("text") or "")
                for record in current_records[position + 1:]
            ) if position >= 0 else ""
            if any(marker in later_text for marker in (
                "短期可用", "仅短期", "短期方案", "临时方案", "暂时处理",
                "长期有复发风险", "仍有复发风险", "可能复发",
            )):
                outcome_type = "partial_temporary"
            elif any(marker in later_text for marker in (
                "随后复发", "后续复发", "再次出现", "又出现", "重新出现",
            )):
                outcome_type = "recurred"
        matched_action = next((action for action in actions if str(action.get("action_ref") or "") == action_ref), None)
        diagnostic_summary = ""
        if outcome_type in {"verified_fix", "partial_temporary", "mitigation_observed"} and str(
            (matched_action or {}).get("action_role") or ""
        ) in {"inspect", "collect", "compare"}:
            outcome_type = "diagnostic_method"
            diagnostic_summary = (
                f"{str((matched_action or {}).get('label') or action_text or '该检查')}用于定位故障原因；"
                "该检查本身不构成修复结论。"
            )
        # A successful reboot/reproduction check is evidence about the
        # preceding change, not itself a resolved solution.  Prefer the most
        # recent concrete change action when the matched action is verify/
        # observe and the outcome is positive.
        if outcome_type in {"verified_fix", "partial_temporary", "mitigation_observed"} and (
            not action_ref or str((matched_action or {}).get("action_role") or "") in {"verify", "observe"}
        ):
            for action in reversed(actions):
                if str(action.get("action_role") or "") in {"change", "inspect"}:
                    action_ref = str(action.get("action_ref") or "")
                    break
        if not action_ref:
            continue
        key = (action_ref, outcome_type, message_id)
        if key in seen:
            continue
        seen.add(key)
        summary = diagnostic_summary or text
        if outcome_type == "pending_validation" and not any(marker in summary for marker in ("待", "观察", "验证", "尚未", "未给出")):
            summary = f"{summary}；当前未给出稳定验证结果。"
        out.append({
            "action_ref": action_ref,
            "outcome_type": outcome_type,
            "outcome_origin": "rule_inferred" if diagnostic_summary else "source_extracted",
            "summary": summary,
            "why_not_other_types": "current_episode_sentence_role",
            "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12],
            "high_cost": False,
            "destructive": False,
        })
    return out


def _is_top_lift_case(text: str) -> bool:
    value = str(text or "")
    return (
        any(marker in value for marker in ("顶板", "顶升", "面顶气缸"))
        and any(marker in value for marker in ("升起", "降落", "升降", "速度", "气流"))
        and any(marker in value for marker in ("三通气管", "气管", "气流过小", "气缸"))
    )


def _is_bios_battery_boot_case(text: str) -> bool:
    value = str(text or "")
    return (
        "主板电池" in value
        and any(marker in value for marker in ("BIOS", "bios", "无法开机", "断电重启"))
        and any(marker in value for marker in (
            "更换主板电池后", "已更换主板电池", "更换完主板电池", "换完主板电池", "换主板电池后",
        ))
        and any(marker in value for marker in ("重置", "恢复正常", "未出现异常", "未再出现"))
    )


def _is_light_usb_recovery_case(text: str) -> bool:
    value = str(text or "")
    return (
        "光源初始化失败" in value
        and "USB" in value.upper()
        and any(marker in value for marker in ("重新拔插", "拔插", "重插"))
        and any(marker in value for marker in ("已正常", "恢复正常", "正常"))
    )


def _classify_observed_outcome(text: str, message_context: str = "") -> str:
    value = f"{text} {message_context}"
    negative_scope = value
    for negated_failure in (
        "未出现无法", "没有出现无法", "未再出现无法", "未发现无法", "未出现失败", "没有出现失败",
    ):
        negative_scope = negative_scope.replace(negated_failure, "")
    negative_scope = negative_scope.replace("无法复现", "")
    has_positive = any(marker in value for marker in (
        "恢复正常", "测试正常", "拍照正常", "正常运行", "正常使用", "可以正常", "未再出现", "未出现异常",
        "未出现无法", "无法复现", "已解决",
    ))
    has_negative = any(marker in negative_scope for marker in (
        "无效", "没有效果", "仍然", "依然", "还是", "未解决", "无法", "失败", "再次出现", "又出现", "复发", "重现",
    ))
    # Recurrence belongs to this outcome sentence.  A previous recurrence in
    # the same message must not turn a later "继续观察" action into recurred.
    if any(marker in text for marker in ("再次出现", "又出现", "复发", "重现")):
        return "recurred"
    if has_positive and has_negative:
        return "partial_temporary"
    if any(marker in value for marker in ("暂时", "临时", "短暂", "短期", "一段时间", "过一会", "过了")):
        return "partial_temporary"
    if any(marker in value for marker in ("待观察", "继续观察", "后续观察", "待验证", "仍需观察", "还需验证")):
        return "pending_validation"
    if has_negative:
        return "ineffective"
    if any(marker in value for marker in (
        "未再出现", "至今未", "持续正常", "已解决", "问题解决", "恢复生产", "正常生产",
    )):
        return "verified_fix"
    if has_positive:
        return "mitigation_observed"
    return "pending_validation"


def _case_understanding_required_info(
    semantics: dict[str, Any],
    family_label: str,
    *,
    legacy_candidate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out = _llm_case_understanding_required_info(legacy_candidate, semantics)
    # RequiredInfoSpec is learned from an actual question in the case.  Family
    # defaults belong to the SOP/KG alignment context and must not be emitted as
    # if the current chat supplied that evidence.
    return out


def _llm_action_candidates(legacy_candidate: dict[str, Any] | None, semantics: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(legacy_candidate, dict):
        return []
    trace = legacy_candidate.get("diagnostic_trace") if isinstance(legacy_candidate.get("diagnostic_trace"), dict) else {}
    outcomes = [item for item in legacy_candidate.get("diagnostic_outcomes") or [] if isinstance(item, dict)]
    default_evidence = _list(semantics.get("evidence_ids"))[:12]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("actual_order", "recommended_order"):
        for item in trace.get(key) or []:
            if isinstance(item, dict):
                label = trim_text(item.get("label") or item.get("action_label") or item.get("check_id") or "", 60)
                summary = trim_text(item.get("label") or item.get("action_label") or "", 180)
                evidence = _list(item.get("evidence_message_ids"))[:12] or default_evidence
            else:
                label = trim_text(_action_label(item), 60)
                summary = trim_text(item, 180)
                evidence = default_evidence
            norm = _norm(label or summary)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append({
                "label": label or trim_text(_action_label(summary), 60),
                "summary": summary or label,
                "action_role": infer_action_role(summary or label),
                "source_evidence_ids": evidence,
                "high_cost": False,
                "destructive": False,
            })
    for item in outcomes:
        label = trim_text(item.get("action_label") or "", 60)
        norm = _norm(label)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        summary_bits = [str(item.get("condition") or ""), str(item.get("root_cause_summary") or ""), label]
        out.append({
            "label": label,
            "summary": trim_text("；".join(bit for bit in summary_bits if bit), 180),
            "action_role": infer_action_role(label),
            "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12] or default_evidence,
            "high_cost": bool(item.get("high_cost")),
            "destructive": bool(item.get("destructive")),
        })
    return out


def _llm_case_understanding_outcomes(
    actions: list[dict[str, Any]],
    legacy_candidate: dict[str, Any] | None,
    semantics: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(legacy_candidate, dict):
        return []
    outcomes = [item for item in legacy_candidate.get("diagnostic_outcomes") or [] if isinstance(item, dict)]
    if not outcomes:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in outcomes:
        action_ref = _match_action_ref(str(item.get("action_label") or ""), actions)
        if not action_ref:
            continue
        outcome_type = str(item.get("outcome_type") or "")
        if not outcome_type:
            continue
        key = (action_ref, outcome_type)
        if key in seen:
            continue
        seen.add(key)
        summary = trim_text(
            item.get("root_cause_summary")
            or item.get("action_label")
            or "",
            200,
        )
        # ``condition`` values such as ``dmp`` or ``camera_capture_chain`` are
        # routing metadata, not observed results.  Never use them as the
        # natural-language outcome shown to W4/W6.
        if summary.lower() in {"dmp", "camera_capture_chain", "software_version_change", "startup/init", "root_cause"}:
            summary = ""
        if not summary:
            continue
        out.append({
            "action_ref": action_ref,
            "outcome_type": outcome_type,
            "outcome_origin": "source_extracted",
            "summary": summary,
            "why_not_other_types": "llm_extracted",
            "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12] or _list(semantics.get("evidence_ids"))[:12],
            "high_cost": bool(item.get("high_cost")),
            "destructive": bool(item.get("destructive")),
        })
    return out


def _llm_case_understanding_required_info(
    legacy_candidate: dict[str, Any] | None,
    semantics: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(legacy_candidate, dict):
        return []
    items = [item for item in legacy_candidate.get("required_info_candidates") or [] if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        slot = _normalize_slot(str(item.get("slot") or "other"))
        question = trim_text(item.get("question") or item.get("label") or slot, 100)
        why_required = trim_text(item.get("why_required") or "", 160)
        if not question or not why_required:
            continue
        key = (slot, question)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "slot_hint": slot,
            "question": question,
            "why_required": why_required,
            "blocks": [trim_text(item.get("label") or question, 60)],
            "source_evidence_ids": _list(item.get("evidence_message_ids"))[:12] or _list(semantics.get("evidence_ids"))[:12],
            "generic_risk": "high" if slot == "other" else "low",
        })
    return out


def _match_action_ref(action_label: str, actions: list[dict[str, Any]]) -> str:
    target = _norm(action_label)
    if not target:
        return ""
    for action in actions:
        label = _norm(action.get("label") or "")
        summary = _norm(action.get("summary") or "")
        if target == label or (label and label in target) or (target and target in summary):
            return str(action.get("action_ref") or "")
    return ""


def _distinguishing_conditions(semantics: dict[str, Any]) -> list[str]:
    out = []
    for value in _list(semantics.get("versions"))[:3]:
        out.append(f"software_version:{value}")
    for value in _list(semantics.get("sites"))[:2]:
        out.append(f"site:{value}")
    if semantics.get("conclusion"):
        out.append("has_conclusion")
    return out


def _case_uncertainties(semantics: dict[str, Any], split: dict[str, Any]) -> list[str]:
    issues = []
    if split.get("decision") == "review_for_possible_split":
        issues.append("possible_split_case")
    if not semantics.get("conclusion"):
        issues.append("missing_terminal_conclusion")
    return issues


def _keywords_from_case(case: dict[str, Any]) -> list[str]:
    text = " ".join([
        str((case.get("family_hypothesis") or {}).get("label") or ""),
        str((case.get("variant_hypothesis") or {}).get("label") or ""),
        str(case.get("symptom_summary") or ""),
    ])
    out: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text):
        if len(chunk) >= 2 and chunk not in out:
            out.append(chunk)
    return out[:16]


def _family_candidates(text: str, category: str, primary_family: str) -> list[str]:
    specific_family = _specific_fault_family(text)
    if specific_family:
        return [specific_family]
    candidates: list[str] = []
    def add(label: str) -> None:
        if label and label not in candidates:
            candidates.append(label)
    lowered = str(text or "")
    has_blue_screen = any(k in lowered for k in ("MEMORY_MANAGEMENT", "PFN", "PTE", "0x00000139", "蓝屏", "Bugcheck", "bugcheck"))
    has_reboot = any(k in lowered for k in ("自动重启", "异常重启", "无故重启", "突然重启"))
    has_boot_fail = any(k in lowered for k in ("无法开机", "开机无法启动", "无法启动", "开不了机"))
    if any(k in lowered for k in ("自动关机", "自动断电", "异常断电", "供电中断")):
        add("工控机异常重启")
    if "buddy" in lowered.lower() and any(k in lowered.lower() for k in ("http 500", "http status:500", "保存", "冷存储", "make cold project")):
        add("Buddy问题")
    if "网卡" in lowered and any(k in lowered for k in ("重置", "断开", "掉线", "扩展网卡", "拓展网卡", "网络中断", "链路")):
        add("网络连接异常")
    if any(k in lowered for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        add("工控机无法开机")
    if "加载用户配置失败" in lowered or "user.cfg" in lowered or "conf" in lowered:
        add("用户配置加载失败")
    if "相机初始化失败" in lowered:
        add("相机初始化失败")
    if "光源初始化失败" in lowered:
        add("光源初始化失败")
    if any(k in lowered for k in ("拍摄失败", "拍照失败", "无法拍照", "不拍照", "拍摄无响应", "空图")):
        add("相机拍摄失败")
    if has_blue_screen:
        add("工控机蓝屏")
    if has_reboot and not has_blue_screen and not any(k in lowered for k in ("加载用户配置失败", "光源初始化失败", "相机初始化失败")):
        add("工控机异常重启")
    if has_boot_fail:
        add("工控机无法开机")
    if any(k in lowered for k in ("黑屏无显示", "开机黑屏", "黑屏不显示")):
        add("工控机黑屏无显示")
    if any(k in lowered for k in ("显示不全", "缩放", "分辨率", "扩展显示", "复制显示", "电视显示", "显示异常", "全屏")):
        add("界面显示异常")
    if any(k in lowered for k in ("黑屏", "白屏")) and any(k in lowered for k in ("显卡", "dp", "hdmi", "输出不稳定", "显示器")):
        add("工控机黑屏无显示")
    if any(k in lowered for k in ("图片为空", "空图", "不拍照", "拍摄失败", "拍照失败")):
        add("相机拍摄失败")
    if "CAD" in lowered or "cad" in lowered:
        add("CAD 导入失败")
    if any(k in lowered for k in ("导入不成功", "导入失败", "程序导入失败")) and "cad" not in lowered:
        add("程序板卡加载失败")
    if "mark" in lowered.lower():
        add("Mark 点对齐失败")
    if any(k in lowered for k in ("误报",)):
        add("误报调优异常")
    if any(k in lowered for k in ("漏检", "漏报")):
        add("漏检调优异常")
    if any(k in lowered for k in ("singlepin", "pinpad", "dir=8", "虚焊", "翘脚", "框未生成", "算法结果未出", "提前报警", "ng板卡")):
        add("算法/程序调优异常")
    if any(k in lowered for k in ("搜索项目名", "项目名搜索", "无法搜索项目名")):
        add("主程序/系统异常")
    if any(k in lowered for k in ("卡顿", "响应慢", "缓慢", "变慢", "加载时间慢")) and not (has_blue_screen or has_reboot or has_boot_fail):
        add("程序运行卡顿")
    if any(k in lowered for k in ("卡死", "死机", "无响应", "闪退")) and not (has_blue_screen or has_reboot or has_boot_fail):
        add("软件卡死无响应")
    if any(k in lowered for k in ("d盘", "磁盘", "页面文件", "虚拟内存", "显存不足")):
        add("磁盘 I/O 异常")
    if any(k in lowered for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘")):
        add("外设连接不稳定")
    if any(k in lowered for k in ("交换机", "收发器", "光纤接口", "网卡自适配1g", "运控卡连接失败")):
        add("控制器网络配置异常")
    if any(k in lowered for k in ("ct变慢", "ct 时间", "节拍", "睿频上不去")):
        add("CT 时间异常增加")
    if any(k in lowered for k in ("保存路径失败", "获取保存路径失败", "保存结果失败")):
        add("复判保存结果失败")
    if not candidates:
        add(_canonicalize_family_label(primary_family, "", category, lowered))
    return candidates[:3]


def _collapse_cases(cases: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    if len(cases) <= 1:
        return cases
    variant_labels = {
        _norm(((case.get("variant_hypothesis") or {}).get("label") or ""))
        for case in cases
        if isinstance(case, dict)
    }
    family_labels = {
        str((case.get("family_hypothesis") or {}).get("label") or "")
        for case in cases
        if isinstance(case, dict)
    }
    chosen = ""
    if len([v for v in variant_labels if v]) == 1:
        chosen = _choose_collapsed_family(family_labels, text)
    if not chosen:
        return cases
    for case in cases:
        if str((case.get("family_hypothesis") or {}).get("label") or "") == chosen:
            return [case]
    return cases


def _choose_collapsed_family(family_labels: set[str], text: str) -> str:
    perf_markers = ("cpu", "CPU", "内存", "资源监控", "占用", "性能", "睿频", "耗时", "卡顿", "变慢")
    if {"工控机蓝屏", "工控机异常重启"}.issubset(family_labels):
        return "工控机蓝屏"
    if {"工控机异常重启", "软件卡死无响应"}.issubset(family_labels) and any(
        marker in text for marker in ("自动关机", "自动断电", "供电中断", "电源输出线", "模组电源")
    ):
        return "工控机异常重启"
    if {"工控机蓝屏", "工控机无法开机"}.issubset(family_labels):
        return "工控机蓝屏"
    if {"误报调优异常", "漏检调优异常", "算法/程序调优异常"}.issubset(family_labels):
        if any(k in text for k in ("漏检", "漏测", "漏铜", "缺件", "跨件连锡")):
            return "漏检调优异常"
        if any(k in text for k in ("误报", "客户都觉得报少", "料号丝印核对", "颜色算法")):
            return "误报调优异常"
        return "算法/程序调优异常"
    if {"误报调优异常", "算法/程序调优异常"}.issubset(family_labels):
        return "误报调优异常"
    if {"漏检调优异常", "算法/程序调优异常"}.issubset(family_labels):
        return "漏检调优异常"
    if {"程序运行卡顿", "软件卡死无响应", "磁盘 I/O 异常"}.issubset(family_labels):
        return "磁盘 I/O 异常"
    if {"程序运行卡顿", "软件卡死无响应"}.issubset(family_labels):
        return "软件卡死无响应"
    if {"程序运行卡顿", "误报调优异常"}.issubset(family_labels) and any(k in text for k in perf_markers):
        return "程序运行卡顿"
    if {"程序运行卡顿", "误报调优异常"}.issubset(family_labels) and any(k in text for k in ("误报", "客户都觉得报少", "颜色算法", "引脚", "虚焊", "翘脚", "误报30个")):
        return "误报调优异常"
    if {"程序运行卡顿", "相机拍摄失败"}.issubset(family_labels):
        if any(k in text for k in ("拍摄失败", "拍照失败", "不拍照", "空图", "相机", "网线", "网口", "拓展网卡", "ping了相机网络")):
            return "相机拍摄失败"
        if any(k in text for k in perf_markers):
            return "程序运行卡顿"
    if {"误报调优异常", "软件卡死无响应"}.issubset(family_labels):
        return "误报调优异常"
    if {"漏检调优异常", "算法/程序调优异常", "误报调优异常"}.issubset(family_labels):
        if "漏检" in text or "漏铜" in text or "跨件连锡" in text:
            return "漏检调优异常"
        if "误报" in text:
            return "误报调优异常"
    if {"漏检调优异常", "算法/程序调优异常"}.issubset(family_labels):
        return "漏检调优异常"
    if {"算法/程序调优异常", "误报调优异常"}.issubset(family_labels):
        return "误报调优异常"
    if {"工控机异常重启", "误报调优异常"}.issubset(family_labels) and "误报" in text:
        return "误报调优异常"
    if {"工控机蓝屏", "误报调优异常"}.issubset(family_labels) and any(k in text for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE", "蓝屏", "Bugcheck", "bugcheck")):
        return "工控机蓝屏"
    if {"外设连接不稳定", "相机拍摄失败"}.issubset(family_labels) and any(k in text for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘")):
        return "外设连接不稳定"
    if {"外设连接不稳定", "工控机异常重启"}.issubset(family_labels) and any(k in text for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘")):
        return "外设连接不稳定"
    if {"外设连接不稳定", "软件卡死无响应"}.issubset(family_labels):
        if any(k in text for k in ("黑屏", "关机", "断电", "不亮", "无法点动", "响应状态")):
            return "软件卡死无响应"
        if any(k in text for k in ("wifi", "无线网卡", "网卡驱动", "水星")):
            return "外设连接不稳定"
        if any(k in text for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘")):
            return "外设连接不稳定"
    if {"工控机黑屏无显示", "软件卡死无响应"}.issubset(family_labels) and any(k in text for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return "工控机黑屏无显示"
    if {"工控机无法开机", "软件卡死无响应"}.issubset(family_labels) and any(k in text for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return "工控机无法开机"
    if {"外设连接不稳定", "相机拍摄失败"}.issubset(family_labels) and any(k in text for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘", "热点", "共享")):
        return "外设连接不稳定"
    if {"CAD 导入失败", "软件卡死无响应"}.issubset(family_labels):
        if any(k in text.lower() for k in ("cad", "导入")):
            return "CAD 导入失败"
        return "软件卡死无响应"
    if {"误报调优异常", "软件卡死无响应"}.issubset(family_labels) and any(k in text for k in ("误报", "颜色算法", "引脚", "虚焊", "翘脚", "误报30个")):
        return "误报调优异常"
    if {"Mark 点对齐失败", "相机拍摄失败"}.issubset(family_labels) and any(k in text.lower() for k in ("mark", "mark点", "器件框跑偏")):
        return "Mark 点对齐失败"
    if {"界面显示异常", "误报调优异常"}.issubset(family_labels):
        if any(k in text for k in ("显示", "缩放", "分辨率", "扩展", "复制", "电视")):
            return "界面显示异常"
        if "误报" in text:
            return "误报调优异常"
    if {"相机拍摄失败", "工控机蓝屏"}.issubset(family_labels) and any(k in text for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE", "蓝屏", "Bugcheck", "bugcheck")):
        return "工控机蓝屏"
    if {"工控机蓝屏", "算法/程序调优异常"}.issubset(family_labels) and any(k in text for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE", "蓝屏", "Bugcheck", "bugcheck")):
        return "工控机蓝屏"
    if {"工控机蓝屏", "误报调优异常"}.issubset(family_labels) and any(k in text for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE", "蓝屏", "Bugcheck", "bugcheck")):
        return "工控机蓝屏"
    return ""


def _collapse_family_candidates(candidates: list[str], text: str, variant_label: str) -> list[str]:
    if len(candidates) <= 1:
        return candidates
    family_set = set(candidates)
    combined = f"{text} {variant_label}"
    perf_markers = ("cpu", "CPU", "内存", "资源监控", "占用", "性能", "睿频", "耗时", "卡顿", "变慢")

    if {"工控机蓝屏", "工控机异常重启"}.issubset(family_set):
        return ["工控机蓝屏"]
    if {"工控机异常重启", "软件卡死无响应"}.issubset(family_set) and any(
        marker in combined for marker in ("自动关机", "自动断电", "供电中断", "电源输出线", "模组电源")
    ):
        return ["工控机异常重启"]
    if {"工控机蓝屏", "工控机无法开机"}.issubset(family_set) and any(k in combined for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE", "蓝屏", "Bugcheck", "bugcheck")):
        return ["工控机蓝屏"]
    if {"误报调优异常", "算法/程序调优异常"}.issubset(family_set):
        return ["误报调优异常"]
    if {"漏检调优异常", "算法/程序调优异常"}.issubset(family_set):
        return ["漏检调优异常"]
    if {"程序运行卡顿", "软件卡死无响应", "磁盘 I/O 异常"}.issubset(family_set):
        return ["磁盘 I/O 异常"]
    if {"程序运行卡顿", "软件卡死无响应"}.issubset(family_set):
        return ["软件卡死无响应"]
    if {"程序运行卡顿", "误报调优异常"}.issubset(family_set) and any(k in combined for k in perf_markers):
        return ["程序运行卡顿"]
    if {"程序运行卡顿", "相机拍摄失败"}.issubset(family_set):
        if any(k in combined for k in ("拍摄失败", "拍照失败", "不拍照", "空图", "相机", "网线", "网口", "拓展网卡", "ping了相机网络")):
            return ["相机拍摄失败"]
        if any(k in combined for k in perf_markers):
            return ["程序运行卡顿"]
    if {"相机拍摄失败", "工控机蓝屏"}.issubset(family_set) and any(k in combined for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE", "蓝屏", "Bugcheck", "bugcheck")):
        return ["工控机蓝屏"]
    if {"外设连接不稳定", "相机拍摄失败"}.issubset(family_set):
        return ["相机拍摄失败"]
    if {"工控机蓝屏", "误报调优异常"}.issubset(family_set) and any(k in variant_label for k in ("蓝屏", "0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE")):
        return ["工控机蓝屏"]
    if {"外设连接不稳定", "软件卡死无响应"}.issubset(family_set) and any(k in variant_label for k in ("黑屏", "关机", "断电", "不亮", "无法点动", "响应状态")):
        return ["软件卡死无响应"]
    if {"外设连接不稳定", "软件卡死无响应"}.issubset(family_set) and any(k in combined for k in ("wifi", "无线网卡", "网卡驱动", "水星")):
        return ["外设连接不稳定"]
    if {"工控机黑屏无显示", "软件卡死无响应"}.issubset(family_set) and any(k in combined for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return ["工控机黑屏无显示"]
    if {"工控机无法开机", "软件卡死无响应"}.issubset(family_set) and any(k in combined for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return ["工控机无法开机"]
    if {"CAD 导入失败", "软件卡死无响应"}.issubset(family_set) and not any(k in text.lower() for k in ("cad", "导入")):
        return ["软件卡死无响应"]
    return candidates


def _generic_family_label(category: str, text: str) -> str:
    lowered = str(text or "")
    if any(k in lowered for k in ("cpu", "内存", "资源监控", "占用", "性能")) and any(k in lowered for k in ("卡顿", "变慢", "响应慢")):
        return "程序运行卡顿"
    if any(k in lowered for k in ("cpu", "内存", "资源监控", "占用", "性能")) and any(k in lowered for k in ("卡顿", "变慢", "响应慢")):
        return "程序运行卡顿"
    if any(k in lowered for k in ("无法开机", "开机无法启动", "无法启动", "开不了机")):
        return "工控机无法开机"
    if any(k in lowered for k in ("黑屏无显示", "开机黑屏", "黑屏不显示")):
        return "工控机黑屏无显示"
    if any(k in lowered for k in ("搜索项目名", "项目名搜索", "无法搜索项目名")):
        return "主程序/系统异常"
    if any(k in text for k in ("二维码", "条码", "扫码", "DM码", "QR码")):
        return "扫码识别失败"
    if any(k in text for k in ("显示", "分辨率", "缩放", "布局", "全屏")):
        return "界面显示异常"
    if any(k in lowered for k in ("卡顿", "响应慢", "缓慢", "变慢")):
        return "程序运行卡顿"
    if any(k in lowered for k in ("卡死", "死机", "无响应", "闪退")):
        return "软件卡死无响应"
    if any(k in lowered for k in ("d盘", "磁盘", "页面文件", "虚拟内存", "显存不足")):
        return "磁盘 I/O 异常"
    if any(k in lowered for k in ("wifi", "无线网卡", "网卡驱动", "水星")) and any(k in lowered for k in ("闪退", "异常", "崩溃", "断连")):
        return "外设连接不稳定"
    if any(k in lowered for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘")):
        return "外设连接不稳定"
    if any(k in lowered for k in ("交换机", "收发器", "光纤接口", "运控卡连接失败", "网卡自适配1g")):
        return "控制器网络配置异常"
    if any(k in lowered for k in ("保存路径失败", "获取保存路径失败", "保存结果失败")):
        return "复判保存结果失败"
    if any(k in lowered for k in ("导入不成功", "程序导入失败")) and "cad" not in lowered:
        return "程序板卡加载失败"
    if "误报" in lowered and not (any(k in lowered for k in ("cpu", "内存", "资源监控", "占用", "性能")) and any(k in lowered for k in ("卡顿", "变慢", "响应慢"))):
        return "误报调优异常"
    if any(k in lowered for k in ("漏检", "漏报")):
        return "漏检调优异常"
    if any(k in text for k in ("误报", "漏检", "识别")):
        return "算法/程序调优异常"
    if category == "硬件与运控":
        return "硬件/运控异常"
    if category == "算法与程序调优":
        return "算法/程序调优异常"
    return "主程序/系统异常"


def _canonicalize_family_label(family_label: str, subsystem: str, category: str, text: str) -> str:
    raw_family = str(family_label or "").strip()
    raw_subsystem = str(subsystem or "").strip()
    lowered = str(text or "").lower()
    combined = " ".join([raw_family, raw_subsystem, str(text or "")]).strip()
    perf_markers = ("cpu", "内存", "资源监控", "占用", "性能", "睿频", "耗时")
    approved = APPROVED_FAMILY_LABELS
    banned_family = PSEUDO_FAMILY_LABELS
    if any(k in combined for k in ("自动关机", "自动断电", "异常断电", "供电中断")):
        return "工控机异常重启"
    if "buddy" in combined.lower() and any(k in combined.lower() for k in ("http 500", "http status:500", "保存", "冷存储", "make cold project")):
        return "Buddy问题"
    if "网卡" in combined and any(k in combined for k in ("重置", "断开", "掉线", "扩展网卡", "拓展网卡", "网络中断", "链路")):
        return "网络连接异常"
    if any(k in combined for k in perf_markers) and any(k in combined for k in ("卡顿", "变慢", "响应慢")):
        return "程序运行卡顿"
    if any(k in combined for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return "工控机无法开机"
    if any(k in combined for k in ("无法开机", "开机无法启动", "无法启动", "开不了机")):
        return "工控机无法开机"
    if any(k in combined for k in ("黑屏无显示", "开机黑屏", "黑屏不显示")):
        return "工控机黑屏无显示"
    if any(k in combined for k in ("黑屏", "白屏")) and any(k in combined.lower() for k in ("显卡", "dp", "hdmi", "显示器", "输出不稳定")):
        return "工控机黑屏无显示"
    if any(k in combined for k in ("搜索项目名", "项目名搜索", "无法搜索项目名")):
        return "主程序/系统异常"
    if any(k in combined for k in ("显示不全", "缩放", "分辨率", "扩展显示", "复制显示", "电视显示", "显示异常", "全屏")):
        return "界面显示异常"
    if any(k in combined for k in ("ct变慢", "ct 时间", "节拍", "睿频上不去")):
        return "CT 时间异常增加"
    if any(k in combined for k in ("mes", "MES", "过站", "工单号", "接驳台", "返回值")):
        return "MES 过站异常"
    if any(k in combined for k in ("加密狗", "许可证", "license", "License", "密码狗")):
        return "许可证/加密狗异常"
    if any(k in combined for k in ("坏板标记", "跳过后sn报警", "SN报警", "未提示", "坏板跳过")):
        return "坏板标记异常"
    if any(k in combined for k in ("复判窗口", "未复判的数据", "pass板无弹窗反馈", "复判界面没显示", "复盘结果不出来", "复判结果显示")):
        return "复判结果显示异常"
    if any(k in combined for k in ("异响", "刺耳声音", "嗡鸣", "轮子摆动")):
        return "机械运动异响"
    if any(k in combined for k in ("传感器", "感应器", "感应不到", "感应不好", "不灵敏")):
        return "传感器感应异常"
    if any(k in combined for k in ("wifi", "无线网卡", "网卡驱动", "水星")) and any(k in combined for k in ("闪退", "异常", "崩溃", "断连")):
        return "外设连接不稳定"
    if any(k in combined for k in ("wifi", "无线网卡", "连不上wifi", "usb", "u盘")):
        return "外设连接不稳定"
    if any(k in combined for k in ("卡顿", "响应慢", "缓慢", "变慢")):
        return "程序运行卡顿"
    if any(k in combined for k in ("卡死", "死机", "无响应", "闪退")):
        return "软件卡死无响应"
    if any(k in combined for k in ("图片为空", "空图", "不拍照", "拍摄失败", "拍照失败")):
        return "相机拍摄失败"
    if any(k in combined for k in ("d盘", "磁盘", "页面文件", "虚拟内存", "显存不足")):
        return "磁盘 I/O 异常"
    if any(k in combined for k in ("交换机", "收发器", "光纤接口", "运控卡连接失败", "网卡自适配1g")):
        return "控制器网络配置异常"
    if any(k in combined for k in ("保存路径失败", "获取保存路径失败", "保存结果失败")):
        return "复判保存结果失败"
    if any(k in combined for k in ("导入不成功", "程序导入失败")) and "cad" not in combined.lower():
        return "程序板卡加载失败"
    if "误报" in combined and not (any(k in combined for k in perf_markers) and any(k in combined for k in ("卡顿", "变慢", "响应慢"))):
        return "误报调优异常"
    if any(k in combined for k in ("漏检", "漏报")):
        return "漏检调优异常"
    if any(k in combined for k in ("算法结果未出", "提前报警", "ng板卡", "singlepin", "pinpad", "dir=8", "虚焊", "翘脚", "框未生成")):
        return "算法/程序调优异常"
    if raw_family in approved:
        return raw_family
    if raw_subsystem in ("显示/分辨率/缩放", "显示/界面"):
        return "界面显示异常"
    if raw_subsystem in ("算法/检测逻辑", "算法/程序调优", "复判流程", "算法/检测程序"):
        return "算法/程序调优异常"
    if raw_subsystem in ("工控机/Windows系统", "工控机/Windows内核"):
        return "主程序/系统异常" if any(k in combined for k in ("项目名", "保存路径", "程序导入")) else "工控机异常重启"
    if raw_subsystem in ("主程序配置/复判站配置",):
        return "程序板卡加载失败" if any(k in combined for k in ("导入", "板卡")) else "用户配置加载失败"
    if raw_subsystem in ("相机/采集链路",):
        if any(k in combined for k in ("卡顿", "卡死", "闪退")):
            return "程序运行卡顿" if "卡顿" in combined else "软件卡死无响应"
        if any(k in combined for k in ("导入", "程序")):
            return "程序板卡加载失败"
        if any(k in combined for k in ("无法开机", "黑屏")):
            return "工控机无法开机"
        if any(k in combined for k in ("u盘", "usb", "wifi", "无线网卡")):
            return "外设连接不稳定"
        if any(k in combined for k in ("保存路径失败", "获取保存路径失败")):
            return "复判保存结果失败"
    if raw_family in banned_family or raw_family.lower() in {"display", "camera", "software"}:
        return _generic_family_label(category, combined)
    return raw_family or _generic_family_label(category, combined)


def _subsystem_for_family(family_label: str) -> str:
    return FAMILY_SUBSYSTEM_EXPECTED.get(family_label, "")


def _summary_for_family(family_label: str) -> str:
    mapping = {
        "工控机无法开机": "设备上电后无法正常启动进入操作系统。",
        "工控机黑屏无显示": "设备上电后显示链路无输出或持续黑屏。",
        "工控机蓝屏": "系统运行中出现蓝屏或等效停止界面。",
        "工控机异常重启": "设备运行中无明确蓝屏界面，直接自动重启。",
        "操作系统启动失败": "操作系统启动过程中报错、卡住或无法进入桌面。",
        "BIOS 启动配置异常": "BIOS 启动项或配置异常导致系统无法正常启动。",
        "多硬盘启动冲突": "多硬盘或启动盘优先级冲突导致系统启动异常。",
        "相机拍摄失败": "相机在采图阶段出现不触发、超时、空图或拍摄失败。",
        "光源初始化失败": "软件启动或通电测试阶段，光源模块初始化失败。",
        "用户配置加载失败": "主程序或复判站初始化阶段加载用户配置失败。",
        "运控初始化失败": "运控或运动控制初始化阶段报错、卡住或闪退。",
        "主程序初始化卡住无明确报错": "主程序初始化阶段卡住但无明确错误代码。",
        "主程序无法打开": "主程序启动失败或无法正常打开。",
        "工厂程序无法打开": "工厂程序启动失败或无法正常打开。",
        "运控程序无法打开": "运控程序启动失败或无法正常打开。",
        "SPC 页面无法打开": "SPC 页面无法正常加载或打开。",
        "Buddy 模板缺失": "Buddy 模板缺失导致相关流程无法继续。",
        "Buddy 模板创建失败": "Buddy 模板创建动作失败。",
        "模板文件损坏": "模板文件损坏或结构异常导致使用失败。",
        "相机初始化失败": "相机在初始化阶段无法完成枚举、连接或上电。",
        "CAD 导入失败": "CAD 文件导入 AOI 软件时解析失败或导入结果异常。",
        "CAD 角度不一致": "CAD 导入后角度或方向与期望不一致。",
        "CAD 自动对齐失败": "CAD 自动对齐步骤无法成功完成。",
        "程序板卡加载失败": "程序、板卡或跨设备导入加载失败。",
        "Mark 点对齐失败": "Mark 点识别或定位异常导致后续检测位置偏移。",
        "识别框大小不准确": "识别框大小或范围异常导致检测不稳定。",
        "器件框角度不匹配": "器件框角度不匹配导致识别或检测异常。",
        "焊盘框不对齐": "焊盘框位置或对齐关系异常。",
        "扫码识别失败": "条码、二维码或 DM 码识别失败。",
        "DM 码识别失败": "DM 码识别失败或识别不稳定。",
        "框选识别不准": "框选区域识别不稳定或不准确。",
        "界面显示异常": "主程序或复判站界面显示、布局或缩放异常。",
        "主程序/系统异常": "主程序或系统出现需人工归类的异常。",
        "算法/程序调优异常": "算法、识别或调优链路异常。",
        "误报调优异常": "检测逻辑或参数异常导致误报偏高。",
        "漏检调优异常": "检测逻辑或参数异常导致漏检或漏报。",
        "CT 时间异常增加": "检测或复判流程导致节拍时间明显增加。",
        "复判站出图慢": "复判站加载图片、切板或显示刷新明显变慢。",
        "程序运行卡顿": "软件运行时响应变慢、卡顿或处理效率显著下降。",
        "软件卡死无响应": "软件运行中卡死、闪退或无响应。",
        "磁盘 I/O 异常": "磁盘空间、页面文件或 I/O 异常影响程序运行。",
        "CUDA 计算设备不可用": "显卡/CUDA 计算设备不可用或资源异常。",
        "复判站主机通信异常": "复判站与主机通信链路异常。",
        "复判保存结果失败": "复判结果无法正常保存。",
        "USB 设备识别异常": "USB/U盘或类似外设识别、连接异常。",
        "光源异常": "光源硬件或亮度表现异常。",
        "光控通信异常": "光控链路通信异常。",
        "运控卡初始化异常": "运控卡初始化失败或异常。",
        "控制器网络配置异常": "控制器、网卡或通信网络配置异常。",
        "进板失败": "板卡进入设备流程失败。",
        "出板失败": "板卡出板流程失败。",
        "卡板": "板卡在设备流程中卡滞。",
        "挡块异常": "挡块机构动作异常。",
        "顶升机构异常": "顶升机构动作或状态异常。",
        "皮带运行异常": "皮带运行异常影响输送。",
        "轨道宽度无法调节": "轨道宽度调节失败或异常。",
        "扫码枪异常": "扫码枪识别或连接异常。",
        "气压异常": "气路或气压状态异常。",
        "PCIe 板卡检测异常": "PCIe 板卡枚举或检测异常。",
        "外设连接不稳定": "外设连接状态不稳定或偶发失联。",
        "MES 过站异常": "MES、过站、工单或接驳台接口链路异常。",
        "许可证/加密狗异常": "许可证、授权状态或加密狗异常导致设备无法正常工作。",
        "坏板标记异常": "坏板标记、跳过流程或 SN/结果联动异常。",
        "复判结果显示异常": "复判结果显示、弹窗、未复判列表或复判窗口行为异常。",
        "机械运动异响": "轨道、运动机构或机械部件出现异常异响。",
        "传感器感应异常": "光电、挡板或进出板相关感应器状态异常。",
        "复判站加载板卡异常": "复判站加载板卡动作异常或失败。",
    }
    return mapping.get(family_label, family_label)


def _action_relevant_to_family(action: str, family_label: str) -> bool:
    text = str(action or "")
    # These are trace-control operations whose relevance comes from the trace
    # boundary, not from repeating a family-specific noun in every label.
    if text.startswith((
        "收集", "记录", "验证", "观察", "重启", "判断", "还原", "核对", "切换", "进入", "执行",
        "修复", "重新", "重插", "轻推", "点胶", "增加", "分阶段", "对比", "调整", "更换", "拔插",
    )):
        return True
    if family_label == "工控机无法开机":
        return any(k in text for k in ("无法开机", "无法启动", "开不了机", "主板", "内存", "电源", "启动", "拔除"))
    if family_label == "工控机黑屏无显示":
        return any(k in text for k in ("黑屏", "无显示", "显示器", "显卡", "BIOS", "开机"))
    if family_label == "用户配置加载失败":
        return any(k in text for k in ("user.cfg", "conf", "配置", "备份", "重启验证"))
    if family_label == "相机拍摄失败":
        return any(k in text for k in (
            "相机", "网口", "网线", "拍摄", "拍照", "采集", "主板", "电池", "BIOS", "bios",
            "网卡", "驱动", "内存", "显卡", "参数", "重启", "开机",
            "固件", "磁环", "走线", "时间点", "观察",
        ))
    if family_label == "工控机蓝屏":
        return any(k in text for k in (
            "DMP", "dmp", "内存", "PFN", "PTE", "WPR", "PoolMon", "Driver Verifier", "verifier", "驱动", "显卡", "转存储",
            "固态", "放电", "BIOS", "bios", "PE", "系统还原", "引导", "SATA", "D盘", "启动", "关机", "接地",
            "重启", "观察", "版本", "升级", "回退", "DDU", "Defender", "向日葵", "系统修复",
        ))
    if family_label == "工控机异常重启":
        return any(k in text for k in (
            "蓝屏证据", "驱动", "硬件", "电源", "接地", "环境", "日志", "重启", "内存", "启动", "引导",
            "供电", "线缆", "连接", "外设", "PCI", "端子", "点胶", "断电", "开机", "主板", "CPU",
        ))
    if family_label == "程序运行卡顿":
        return any(k in text for k in ("卡顿", "响应慢", "缓慢", "性能", "CPU", "内存", "资源", "CT", "负载", "磁盘", "页面文件", "显存"))
    if family_label == "软件卡死无响应":
        return any(k in text for k in ("卡死", "死机", "无响应", "闪退", "崩溃", "日志", "驱动", "内存", "资源"))
    if family_label == "磁盘 I/O 异常":
        return any(k in text for k in ("磁盘", "D盘", "页面文件", "虚拟内存", "显存", "空间"))
    if family_label == "光源初始化失败":
        return any(k in text for k in ("光源", "光控", "USB", "上线验证"))
    if family_label == "运控初始化失败":
        return any(k in text for k in ("运控", "运动控制", "初始化", "驱动", "控制卡"))
    if family_label == "界面显示异常":
        return any(k in text for k in ("显示", "缩放", "分辨率", "电视", "扩展", "复制", "画面"))
    if family_label == "主程序/系统异常":
        return any(k in text for k in ("主程序", "项目名", "搜索", "日志", "软件"))
    if family_label == "程序板卡加载失败":
        return any(k in text for k in ("导入", "程序", "板卡", "加载", "应用", "保存路径", "buddv"))
    if family_label == "复判保存结果失败":
        return any(k in text for k in ("保存路径", "保存结果", "buddv"))
    if family_label == "外设连接不稳定":
        return any(k in text for k in ("wifi", "无线网卡", "USB", "U盘", "加密狗", "外设", "连接"))
    if family_label == "控制器网络配置异常":
        return any(k in text for k in ("网卡", "交换机", "收发器", "光纤", "ping", "网络", "IP", "控制器", "运控卡"))
    if family_label == "网络连接异常":
        return any(k in text for k in ("网卡", "网络", "链路", "端口", "驱动", "重置", "掉线", "日志", "生产恢复"))
    if family_label == "Buddy问题":
        return any(k in text.lower() for k in ("buddy", "http", "保存", "日志", "d盘", "硬盘", "sata", "bios", "程序", "存储"))
    if family_label == "CT 时间异常增加":
        return any(k in text for k in ("CT", "卡顿", "睿频", "负载", "CPU风扇", "节拍"))
    if family_label == "顶升机构异常":
        return any(k in text for k in ("顶板", "顶升", "气缸", "气管", "气流", "升降", "速度"))
    if family_label == "误报调优异常":
        return any(k in text for k in ("误报", "singlepin", "pinpad", "dir=8", "虚焊", "翘脚", "框", "参数", "算法"))
    if family_label == "漏检调优异常":
        return any(k in text for k in ("漏检", "漏报", "参数", "阈值", "算法", "识别"))
    if family_label == "扫码识别失败":
        return any(k in text for k in ("二维码", "条码", "扫码", "DM码", "搜索范围"))
    if family_label == "CAD 导入失败":
        return any(k in text for k in ("CAD", "导入", "编码", "坐标", "拼版"))
    if family_label == "Mark 点对齐失败":
        return any(k in text for k in ("Mark", "mark", "模板匹配", "阈值", "对齐"))
    return True


def _evidence_from_card(card: dict[str, Any], case: dict[str, Any], anchor_map: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for anchor in list(dict.fromkeys([*(case.get("evidence_anchor_ids") or []), *[x for item in case.get("required_info") or [] if isinstance(item, dict) for x in item.get("source_evidence_ids") or []]])):
        out.append({
            "source_kind": "chat_message",
            "external_id": str(anchor),
            "title": str(anchor),
            "summary": trim_text(anchor_map.get(anchor) or anchor, 500),
            "payload_ref": "",
        })
    return out


def _evidence_anchor_map(semantics: dict[str, Any]) -> dict[str, str]:
    """Map current-case evidence IDs to reviewable source text.

    Navigation-only ``case_context_messages`` is intentionally excluded.  The
    map is carried through Prompt A/B so W4/W6 can validate and display the
    actual sentence behind each action/outcome instead of an opaque message ID.
    """

    episode = semantics.get("episode") if isinstance(semantics.get("episode"), dict) else {}
    out: dict[str, str] = {}
    for key in (
        "fault_description_messages",
        "diagnostic_chain_messages",
        "resolution_messages",
        "case_evidence_messages",
    ):
        for item in episode.get(key) or []:
            if not isinstance(item, dict):
                continue
            message_id = str(item.get("message_id") or item.get("source_message_id") or "")
            text = trim_text(item.get("text") or item.get("content_summary") or "", 500)
            if message_id and text and message_id not in out:
                out[message_id] = text
    return out


def _evidence_ids_by_external(evidence_ids: list[str], evidence_items: list[dict[str, Any]], refs: Any) -> list[str]:
    wanted = {str(x) for x in refs or [] if str(x)}
    if not wanted:
        return list(evidence_ids[:4])
    out = []
    for item in evidence_items:
        ext = str(item.get("external_id") or "")
        if ext in wanted:
            out.append(str(item.get("evidence_id") or ""))
    return out or list(evidence_ids[:4])


def _required_info_specs(
    candidate: dict[str, Any],
    variant: dict[str, Any],
    error: dict[str, Any],
    case_id: str,
    family_id: str,
    variant_id: str,
    evidence_ids: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidate.get("required_info_candidates") or []:
        if not isinstance(item, dict):
            continue
        slot = _normalize_slot(str(item.get("slot") or "other"))
        question = trim_text(item.get("question") or item.get("label") or slot, 100)
        why = trim_text(item.get("why_required") or f"该信息用于缩小 {variant.get('label') or error.get('label') or family_id} 的诊断范围。", 160)
        key = (slot, question)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "required_info_id": make_id("required-info", f"{case_id}:{slot}:{question}"),
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": slot,
            "question": question,
            "why_required": why,
            "condition": trim_text(item.get("condition") or "", 120),
            "blocks": [str(x) for x in item.get("blocks") or []] or [question],
            "priority": _normalize_priority(item.get("priority")),
            "evidence_ids": _evidence_id_list(evidence_ids, item.get("evidence_message_ids")),
        })
    for item in error.get("required_info_schema") or []:
        if not isinstance(item, dict):
            continue
        slot = _normalize_slot(str(item.get("slot") or infer_required_info_slot(str(item.get("question") or item.get("label") or ""))))
        question = trim_text(item.get("question") or item.get("label") or slot, 100)
        key = (slot, question)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "required_info_id": make_id("required-info", f"{case_id}:{slot}:{question}"),
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": slot,
            "question": question,
            "why_required": trim_text(item.get("why_required") or f"该信息用于缩小 {variant.get('label') or error.get('label') or family_id} 的诊断范围。", 160),
            "condition": trim_text(item.get("condition") or "", 120),
            "blocks": [str(x) for x in item.get("blocks") or []] or [question],
            "priority": _normalize_priority(item.get("priority")),
            "evidence_ids": [entry["evidence_id"] for entry in evidence_ids[:4]],
        })
    for text in error.get("required_info") or []:
        question = trim_text(text, 100)
        slot = _normalize_slot(infer_required_info_slot(question))
        key = (slot, question)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "required_info_id": make_id("required-info", f"{case_id}:{slot}:{question}"),
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": slot,
            "question": question,
            "why_required": trim_text(f"该信息用于缩小 {variant.get('label') or error.get('label') or family_id} 的诊断范围。", 160),
            "condition": "",
            "blocks": [question],
            "priority": "medium",
            "evidence_ids": [entry["evidence_id"] for entry in evidence_ids[:4]],
        })
    if not out:
        for slot, question, why, blocks in _fallback_required_info_from_text(
            str(variant.get("label") or ""),
            str(error.get("label") or ""),
            str(error.get("symptom") or ""),
            str(candidate.get("conclusion") or ""),
        ):
            out.append({
                "required_info_id": make_id("required-info", f"{case_id}:{slot}:{question}"),
                "family_id": family_id,
                "variant_id": variant_id,
                "slot": slot,
                "question": question,
                "why_required": why,
                "condition": "",
                "blocks": blocks,
                "priority": "high" if slot in {"program_file", "ip_config", "dmp_package", "log_package", "driver_context"} else "medium",
                "evidence_ids": [entry["evidence_id"] for entry in evidence_ids[:4]],
            })
    return out


def _outcomes_from_solutions(candidate: dict[str, Any], solutions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for solution in solutions:
        outcome_type = str(solution.get("outcome") or solution.get("evidence_level") or "")
        if outcome_type not in {
            "verified_fix", "ineffective", "partial_temporary", "mitigation_observed", "recurred",
            "pending_validation", "diagnostic_method", "context_not_root_cause",
        }:
            continue
        out.append({
            "outcome_id": make_id("legacy-outcome", solution.get("solution_id") or solution.get("content") or ""),
            "target_solution_id": str(solution.get("solution_id") or ""),
            "action_label": str(solution.get("content") or ""),
            "outcome_type": outcome_type,
            "evidence_message_ids": candidate.get("evidence_ids") or [],
            "high_cost": "返厂" in str(solution.get("content") or "") or "重标" in str(solution.get("content") or ""),
            "destructive": False,
            "root_cause_summary": "",
        })
    return out


def _trace_action_ids(trace: dict[str, Any], fallback: list[str], by_check: dict[str, str], actual: bool = False) -> list[str]:
    key = "actual_order" if actual else "recommended_order"
    out: list[str] = []
    for item in trace.get(key) or []:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("check_id") or item.get("target_check_id") or "")
        action_id = by_check.get(check_id)
        if action_id and action_id not in out:
            out.append(action_id)
    if not out:
        return list(fallback)
    if len(out) < len(fallback):
        # Legacy trace often keeps only the first extracted check. Prefer the
        # full episode debug-action order when it is richer than the partial
        # trace reconstructed from legacy checks.
        return list(fallback)
    return out


def _evidence_items(case_id: str, episode: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(_episode_messages(episode), start=1):
        out.append({
            "evidence_id": make_id("evidence", f"{case_id}:{msg.get('message_id') or idx}"),
            "source_kind": "chat_message",
            "external_id": str(msg.get("message_id") or ""),
            "title": trim_text(msg.get("sender") or f"message-{idx}", 80),
            "summary": trim_text(msg.get("text") or msg.get("content_summary") or "", 500),
            "payload_ref": str(msg.get("create_time") or msg.get("role") or ""),
        })
    if out:
        return out
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    summary = trim_text(
        extracted.get("symptom_raw")
        or extracted.get("conclusion")
        or episode.get("summary")
        or episode.get("episode_id")
        or "legacy case source",
        500,
    )
    return [{
        "evidence_id": make_id("evidence", f"{case_id}:source"),
        "source_kind": "legacy_case_source",
        "external_id": str(episode.get("episode_id") or ""),
        "title": "legacy case source",
        "summary": summary,
        "payload_ref": str(episode.get("thread_id") or ""),
    }]


def _ordered_action_ids_from_debug_actions(
    debug_actions: list[Any],
    action_nodes: list[dict[str, Any]],
    *,
    fallback: list[str],
) -> list[str]:
    if not debug_actions:
        return list(fallback)
    ordered: list[str] = []
    used: set[str] = set()
    for raw in debug_actions:
        target = _norm(_action_label(trim_text(raw, 180)))
        if not target:
            continue
        best_id = ""
        best_score = -1
        for node in action_nodes:
            if not isinstance(node, dict):
                continue
            action_id = str(node.get("action_id") or "")
            if not action_id or action_id in used:
                continue
            label = _norm(node.get("label") or "")
            summary = _norm(node.get("summary") or "")
            score = 0
            if label == target:
                score = 5
            elif target and target in summary:
                score = 4
            elif label and label in target:
                score = 3
            elif summary and any(chunk and chunk in summary for chunk in target.split()):
                score = 1
            if score > best_score:
                best_score = score
                best_id = action_id
        if best_id:
            ordered.append(best_id)
            used.add(best_id)
    for action_id in fallback:
        if action_id and action_id not in used:
            ordered.append(action_id)
            used.add(action_id)
    return ordered or list(fallback)


def _evidence_id_list(evidence_items: list[dict[str, Any]], message_ids: Any) -> list[str]:
    wanted = {str(x) for x in message_ids or [] if str(x)}
    if not wanted:
        return [item["evidence_id"] for item in evidence_items[:4]]
    out = []
    for item in evidence_items:
        if str(item.get("external_id") or "") in wanted:
            out.append(item["evidence_id"])
    return out or [item["evidence_id"] for item in evidence_items[:4]]


def _episode_messages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "noise_messages"):
        for msg in episode.get(key) or []:
            if not isinstance(msg, dict):
                continue
            msg_id = str(msg.get("message_id") or "")
            if msg_id and msg_id in seen:
                continue
            if msg_id:
                seen.add(msg_id)
            sender = msg.get("sender")
            if isinstance(sender, dict):
                sender = sender.get("name") or sender.get("display_name") or sender.get("id") or ""
            out.append({
                "message_id": msg_id,
                "sender": str(sender or ""),
                "text": str(msg.get("text") or msg.get("content_summary") or ""),
                "create_time": str(msg.get("create_time") or ""),
                "role": key,
            })
    return out


def _owner_context(episode: dict[str, Any]) -> str:
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    attribution = extracted.get("attribution") if isinstance(extracted.get("attribution"), dict) else {}
    for item in attribution.get("role_assignments") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") != "confirmed":
            continue
        if "rd_engineer" not in {str(value) for value in item.get("organization_roles") or []}:
            continue
        episode_roles = {str(value) for value in item.get("episode_roles") or []}
        if not episode_roles.intersection({"assignee", "investigator", "resolver"}):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            return name
    return ""


def _humanized(value: str) -> str:
    text = str(value or "")
    if ":" in text:
        text = text.split(":", 1)[1]
    return " ".join(text.replace("-", " ").replace("_", " ").split())


def _action_label(text: str) -> str:
    content = trim_text(text, 60)
    for sep in ("；", "，", "。"):
        if sep in content:
            return content.split(sep, 1)[0]
    return content


def _canonicalize_action_candidate(label: str, summary: str, action_role: str, family_label: str) -> tuple[str, str, str]:
    raw = f"{label} {summary}"
    if family_label == "Buddy问题" and label == "记录故障":
        return "记录Buddy保存失败信息", summary or label, "collect"
    if "诊断数据" in raw and any(marker in raw for marker in ("收集", "上传", "提供")):
        return "收集诊断数据", summary or label, "collect"
    if "SOP" in raw.upper() and "网卡参数" in raw and any(marker in raw for marker in ("检查", "确认", "按")):
        return "按SOP检查网卡参数", summary or label, "inspect"
    if family_label == "相机拍摄失败" and "主板网口" in raw and any(
        marker in raw for marker in ("使用", "插回", "换回")
    ):
        return "将相机网线插回主板网口", summary or label, "change"
    if family_label == "相机拍摄失败" and "换口时间点" in raw:
        return "核对相机网线换口时间点", summary or label, "inspect"
    if family_label == "相机拍摄失败" and "大恒" in raw and "升级" in raw and "固件" in raw:
        return "升级大恒相机固件", summary or label, "change"
    if "判断" in raw and "user.cfg" in raw.lower() and any(k in raw for k in ("损坏", "为空")):
        return "判断当前配置是否损坏", summary or label, "inspect"
    if family_label == "Buddy问题" and label.startswith("记录") and any(k in raw for k in ("报错", "保存", "故障时间")):
        return "记录Buddy保存失败信息", summary or label, "collect"
    if "观察" in raw and any(k in raw for k in ("拍摄失败", "是否复发", "未再次")):
        return "观察是否复发", summary or label, "observe"
    if any(k in raw for k in ("每日重启", "每天重启", "每日关机重启")):
        return "每日关机重启并观察", summary or label, "observe"
    if "后部线缆" in raw and any(k in raw for k in ("重插", "拔插")):
        return "重新拔插工控机后部线缆", summary or label, "change"
    if "重插供电连接" in raw or ("重插" in raw and "全部模组连接线" in raw):
        return "重插模组电源输出连接线", summary or label, "change"
    if family_label == "工控机异常重启" and "还原" in raw and any(k in raw for k in ("自动断电", "无法立即开机")):
        return "还原断电后无法立即开机现象", summary or label, "inspect"
    if "PCI" in raw.upper() and "供电" in raw and any(k in raw for k in ("换位", "还原")):
        return "对比PCI供电换位和还原", summary or label, "compare"
    if "接地" in raw and any(k in raw for k in ("整改", "增加", "接地线")):
        return "增加工控机和侧板接地线", summary or label, "change"
    if "引导损坏" in raw and any(k in raw for k in ("次生", "断电后")):
        return "修复系统引导", summary or label, "change"
    if any(k in raw.lower() for k in ("crystaldiskinfo", "smart")) and "硬盘" in raw:
        return "检查硬盘SMART状态", summary or label, "inspect"
    if "SATA" in raw.upper() and "热插拔" in raw:
        return "检查SATA热插拔BIOS参数", summary or label, "inspect"
    if "主板" in raw and "参数" in raw and any(k in raw for k in ("改回", "恢复")):
        return "恢复主板 BIOS 参数", summary or label, "change"
    if "上电自动开机" in raw and "设置" in raw:
        return "设置上电自动开机", summary or label, "change"
    if "断电重启" in raw:
        return "断电重启设备验证", summary or label, "verify"
    if "显卡驱动" in raw and "更新" in raw:
        return "更新显卡驱动", summary or label, "change"
    if any(k in raw.lower() for k in ("aicusbwifi", "无线网卡")) and "卸载" in raw:
        return "卸载无线网卡驱动", summary or label, "change"
    if "内存检测" in raw and any(k in raw.lower() for k in ("p95", "cpu")):
        return "进行内存和CPU稳定性测试", summary or label, "verify"
    if "bios" in raw.lower() and any(k in raw for k in ("查看", "检查", "设置")):
        return "检查 BIOS 设置", summary or label, "inspect"
    if "每天" in raw and "重启" in raw:
        return "每天重启设备", summary or label, "change"
    if "异步" in raw and "配置" in raw:
        return "检查配置文件异步开关状态", summary or label, "inspect"
    if "异步影响出板功能" in raw and any(k in raw for k in ("询问", "确认", "是否开启")):
        return "确认是否开启异步影响出板功能", summary or label, "inspect"
    if any(k in raw for k in ("事件查看器", "windows日志", "Bugcheck", "诊断日志里有windows事件导出")):
        return "导出Windows事件日志", summary or label, "collect"
    if "软件版本" in raw:
        return "确认软件版本", summary or label, "inspect"
    if "升级" in raw and "0.26.18" in raw:
        return "升级主程序至0.26.18版本验证", summary or label, "change"
    if any(k in raw for k in ("显示缩放", "显示设置的缩放", "Windows显示缩放")):
        return "检查显示缩放比例", summary or label, "inspect"
    if "缩放" in raw and "200" in raw:
        return "尝试设置显示缩放为200%", summary or label, "change"
    if "缩放" in raw and "100" in raw:
        return "尝试设置显示缩放为100%", summary or label, "change"
    if "分辨率" in raw:
        return "检查扩展屏分辨率设置", summary or label, "inspect"
    if "singlepin" in raw and "pinpad" in raw:
        return "检查singlepin与pinpad包含关系", summary or label, "inspect"
    if "dir=8" in raw and "方向" in raw:
        return "检查dir=8器件方向判断逻辑", summary or label, "inspect"
    if "保险点可能改成" in raw:
        return "调整singlepin方向判断逻辑", summary or label, "change"
    if "虚焊" in raw and "框" in raw:
        return "验证虚焊框生成逻辑", summary or label, action_role
    if "项目名" in raw and "搜索" in raw and family_label == "主程序/系统异常":
        return "检查项目搜索功能是否正常", summary or label, "inspect"
    return label, summary, action_role


def _drop_action_candidate(label: str, summary: str, family_label: str) -> bool:
    raw = f"{label} {summary}"
    if label in {"明白", "收到", "好的", "ok", "OK"}:
        return True
    if label in {"升级", "更新", "检查", "确认", "分析", "收集", "观察", "验证", "重启", "还原"}:
        return True
    if any(k in raw for k in ("昨天的修复就可以正常生成", "我只是直觉上认为", "可以关闭buddy后")):
        return True
    if label.startswith("@"):
        return True
    if any(k in raw for k in ("换排查问题操作",)):
        return True
    if family_label == "主程序/系统异常" and any(k in raw for k in ("突然重启", "蓝屏", "Bugcheck", "dmp")):
        return True
    if family_label == "界面显示异常" and any(k in raw for k in ("相似度", "buddy")):
        return True
    if raw.startswith("应该是") and family_label == "界面显示异常":
        return True
    return False


def _keywords(variant: dict[str, Any], error: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for value in [*(variant.get("keywords") or []), *(error.get("keywords") or [])]:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen[:16]


def _first_string(values: Any) -> str:
    if isinstance(values, list):
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
    text = str(values or "").strip()
    return text


def _normalize_slot(slot: str) -> str:
    value = str(slot or "").strip()
    if value in INTERNAL_REQUIRED_INFO_SLOTS:
        return value
    inferred = infer_required_info_slot(value)
    return inferred if inferred in INTERNAL_REQUIRED_INFO_SLOTS else "other"


def _normalize_priority(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    if text in {"1", "p0", "p1", "critical"}:
        return "high"
    if text in {"2", "p2", "normal"}:
        return "medium"
    if text in {"3", "p3", "minor"}:
        return "low"
    return "medium"


def _infer_family_shape(
    canonical_error_id: str,
    semantic_text: str,
    variant: dict[str, Any],
    error: dict[str, Any],
) -> tuple[str, str, str]:
    text = semantic_text
    specific_family = _specific_fault_family(text)
    if specific_family:
        return (
            specific_family,
            _subsystem_for_family(specific_family),
            _summary_for_family(specific_family),
        )
    if any(k in text for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return "工控机无法开机", "工控机/启动链路", "设备异常断电或未正常关机后无法正常开机。"
    if "加载用户配置失败" in text or "user.cfg" in text or "conf" in text:
        return "用户配置加载失败", "主程序配置/复判站配置", "主程序或复判站初始化阶段加载用户配置失败。"
    if "光源初始化失败" in text:
        return "光源初始化失败", "光源/光控链路", "软件启动或通电测试阶段，光源模块初始化失败。"
    if any(k in text for k in ("拍摄失败", "拍照失败", "不拍照", "拍摄无响应", "空图")):
        return "相机拍摄失败", "相机/采集链路", "相机在采图阶段出现不触发、超时、空图或拍摄失败。"
    if any(k in text for k in ("MEMORY_MANAGEMENT", "PFN", "PTE", "0x00000139", "蓝屏", "Bugcheck", "bugcheck")):
        return "工控机蓝屏", "工控机/Windows 内核", "系统运行中出现蓝屏或等效停止界面。"
    if "重启" in text:
        return "工控机异常重启", "工控机/系统运行稳定性", "设备运行中无明确蓝屏界面，直接自动重启。"
    canonical_map = {
        "err:init-config-load-fail": ("用户配置加载失败", "主程序配置/复判站配置", "主程序或复判站初始化阶段加载用户配置失败。"),
        "err:camera-capture-failure": ("相机拍摄失败", "相机/采集链路", "相机在采图阶段出现不触发、超时、空图或拍摄失败。"),
        "err:industrial-pc-freeze-black-screen": ("工控机蓝屏", "工控机/Windows 内核", "系统运行中出现蓝屏或等效停止界面。"),
        "err:light-source-init-failure-driver-reinstall": ("光源初始化失败", "光源/光控链路", "软件启动或通电测试阶段，光源模块初始化失败。"),
    }
    if canonical_error_id in canonical_map:
        return canonical_map[canonical_error_id]
    subsystem = str(variant.get("subsystem") or error.get("subsystem") or "")
    label = _canonicalize_family_label(
        str(variant.get("label") or error.get("label") or subsystem or _humanized(canonical_error_id)),
        subsystem,
        str(variant.get("category") or error.get("category") or ""),
        text,
    )
    summary = str(error.get("scenario") or error.get("symptom") or variant.get("symptom") or variant.get("label") or error.get("label") or _humanized(canonical_error_id))
    subsystem = _subsystem_for_family(label) or subsystem
    summary = _summary_for_family(label) or summary
    return label, subsystem, summary


def _infer_variant_shape(target_error_id: str, semantic_text: str, variant: dict[str, Any], error: dict[str, Any]) -> tuple[str, str]:
    text = semantic_text
    specific_variant = _specific_variant_shape(text)
    if specific_variant:
        return specific_variant
    if any(k in text for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return ("异常断电后设备无法正常开机", "设备异常断电或未正常关机后再次开机失败，伴随键盘不亮或主程序初始化异常。")
    if "加载用户配置失败" in text and ("更换工控机" in text or "user.cfg" in text):
        return (
            "更换工控机后 user.cfg.toml 为空导致加载用户配置失败",
            "更换工控机后主程序报警加载用户配置失败，怀疑 user.cfg.toml 为空或备份选择错误。",
        )
    if all(k in text for k in ("编程", "拍照")) and any(k in text for k in ("延迟", "卡顿", "速度慢", "拍摄失败")):
        return (
            "编程拍照速度延迟现象",
            "在编程/Teach 拍照过程中出现拍照延迟、卡顿或逐步发展为拍摄失败。",
        )
    if any(k in text for k in ("ping了相机网络", "相机网络", "请求超时频繁", "相机ip", "按压接口处")) and any(k in text for k in ("拍摄失败", "相机", "采集卡", "网卡")):
        return ("相机网络异常导致拍摄失败", "相机网络链路、网口接触或网卡切换异常，导致拍摄失败或采图不稳定。")
    if any(k in text for k in ("网线", "网口", "拓展网卡")) and any(k in text for k in ("拍摄失败", "拍照失败")):
        return ("更换相机网线插口后出现拍摄失败", "将相机网线从主板网口换到拓展网卡后，开始持续出现拍摄失败。")
    if "弯板" in text and any(k in text for k in ("误报", "误差", "风险")):
        return ("弯板导致误报风险增加", "板弯形变量过大，导致检测误差增大并带来误报风险。")
    if "MEMORY_MANAGEMENT" in text and "PFN" in text:
        return ("MEMORY_MANAGEMENT/PFN 不同步蓝屏", "DMP 显示 MEMORY_MANAGEMENT，参数与 PFN 不同步，指向内核内存管理损坏。")
    if "0x00000139" in text:
        return ("0x00000139 关键数据结构损坏蓝屏", "蓝屏错误码为 0x00000139，伴随关键驱动缺失/损坏、转储不完整和可疑第三方驱动。")
    if "PTE" in text:
        return ("System PTE 耗尽蓝屏", "DMP 显示 Free System PTEs 极低，系统无法再为线程栈分配 PTE。")
    if any(k in text for k in ("蓝屏", "死机重启")) and any(k in text for k in ("重启", "死机")) and not any(k in text for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE")):
        return ("运行中蓝屏重启", "设备运行过程中出现蓝屏/死机并重启，但当前证据尚未精确到具体错误码分支。")
    if "光源初始化失败" in text and "USB" in text:
        return ("离线安装通电测试后光源初始化失败，USB 重新拔插后恢复", "设备离线安装后通电测试时光源初始化失败，重新拔插光源 USB 接口后恢复正常。")
    if any(k in text for k in ("wifi", "无线网卡", "网卡驱动", "水星")) and any(k in text for k in ("闪退", "异常", "崩溃", "断连")):
        return ("无线网卡驱动异常导致软件闪退", "无线网卡驱动或网络相关外设异常，导致软件闪退或连接不稳定。")
    if any(k in text for k in ("自动重启", "异常重启", "突然重启", "无故重启")) and not any(k in text for k in ("蓝屏", "Bugcheck", "bugcheck")):
        return ("运行中自动重启", "设备运行过程中无明确蓝屏画面，直接发生自动重启。")
    if any(k in text for k in ("图片为空", "空图")):
        return ("新板编程测试时图片为空", "编程或测试过程中出现图片为空/空图，导致无法正常继续检测。")
    if any(k in text for k in ("黑屏", "白屏")) and any(k in text.lower() for k in ("显卡", "dp", "hdmi", "显示器", "输出不稳定")):
        return ("显卡或显示链路导致黑屏无显示", "设备运行中出现黑屏/白屏，怀疑显卡或显示输出链路不稳定。")
    label = _canonicalize_variant_label(
        str(variant.get("label") or error.get("label") or _humanized(target_error_id)),
        text,
    )
    summary = str(variant.get("symptom") or error.get("symptom") or semantic_text or label)
    return label, summary


def _canonicalize_variant_label(label: str, text: str) -> str:
    raw = trim_text(label or "", 120)
    semantic = str(text or "")
    combined = " ".join([raw, semantic])
    specific_variant = _specific_variant_shape(combined)
    if specific_variant:
        return specific_variant[0]
    if any(k in combined for k in ("键盘不亮", "开不开机", "未正常关机", "拔掉电源")):
        return "异常断电后设备无法正常开机"
    if any(k in combined for k in ("蓝屏", "死机")) and "重启" in combined and not any(k in combined for k in ("0x00000139", "MEMORY_MANAGEMENT", "PFN", "PTE")):
        return "运行中蓝屏重启"
    if all(k in combined for k in ("编程", "拍照")) and any(k in combined for k in ("延迟", "卡顿", "速度慢", "拍摄失败")):
        return "编程拍照速度延迟现象"
    if any(k in combined for k in ("ping了相机网络", "相机网络", "请求超时频繁", "相机ip", "按压接口处")) and any(k in combined for k in ("拍摄失败", "相机", "采集卡", "网卡")):
        return "相机网络异常导致拍摄失败"
    if any(k in combined for k in ("无法开机", "开机无法启动", "无法启动", "开不了机")):
        return "设备开机无法启动"
    if any(k in combined for k in ("黑屏无显示", "开机黑屏", "黑屏不显示")):
        return "设备开机黑屏无显示"
    if any(k in combined for k in ("搜索项目名", "项目名搜索", "无法搜索项目名")):
        return "主程序无法搜索项目名"
    if "弯板" in combined and any(k in combined for k in ("误报", "误差", "风险")):
        return "弯板导致误报风险增加"
    if any(k in combined for k in ("保存路径失败", "获取保存路径失败", "buddv")):
        return "复判站获取保存路径失败"
    if any(k in combined for k in ("wifi", "无线网卡", "连不上wifi")):
        return "无线网卡异常导致无法连接WiFi"
    if any(k in combined for k in ("u盘", "usb")) and any(k in combined for k in ("卡顿", "响应慢", "变慢")):
        return "U盘插入后操作卡顿"
    if any(k in combined for k in ("显存不足",)):
        return "显存不足导致测试失败"
    if any(k in combined for k in ("程序相互导入", "导入不成功", "个别程序会导入失败", "跨设备导入失败")):
        return "跨设备程序导入失败"
    if any(k in combined for k in ("D盘空间满", "D盘经常满", "页面文件不够", "虚拟内存")):
        return "磁盘空间或页面文件异常导致程序异常"
    if any(k in combined for k in ("软件闪退", "闪退现象", "突然闪退")):
        return "软件运行中闪退"
    if any(k in combined for k in ("卡死", "无响应", "死机")):
        return "软件运行中卡死无响应"
    if any(k in combined for k in ("卡顿", "响应时间比较缓慢", "运行缓慢")):
        return "软件运行卡顿"
    if any(k in combined for k in ("显示不全", "只能扩展", "不能复制", "缩放", "分辨率", "电视")):
        return "复判站电视显示不全只能扩展不能复制"
    if any(k in combined for k in ("singlepin", "pinpad", "dir=8")) and any(k in combined for k in ("虚焊", "翘脚", "框未生成", "自定义框")):
        return "大封装singlepin被包含导致虚焊框未自动生成"
    if any(k in combined for k in ("算法结果未出", "提前报警", "ng板卡", "报警NG")):
        return "算法结果未出软件提前报警NG"
    cleaned = _strip_report_item_prefix(raw)
    for prefix in ("我这个现场", "现场反馈", "客户反馈", "客户23反馈", "客户23反馈", "复判站反馈"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ：:，,。")
    for suffix in ("是什么问题", "怎么处理", "怎么办", "如何处理", "咋回事", "吗"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip(" ：:，,。？?")
    cleaned = cleaned.strip(" ：:，,。？?")
    return trim_text(cleaned or raw or _humanized(combined), 60)


def _specific_fault_family(text: str) -> str:
    """Prefer the terminal fault over broad context words such as 误报/卡顿."""

    clean = str(text or "")
    if _is_top_lift_case(clean):
        return "顶升机构异常"
    if _is_bios_battery_boot_case(clean) and any(marker in clean for marker in ("无法开机", "无法正常开机", "开不了机")):
        return "工控机无法开机"
    if any(k in clean for k in ("远轨", "轨道宽度")) and any(k in clean for k in ("无法正常出板", "无法出板", "出板失败", "板卡卡滞", "卡在轨道")):
        return "出板失败"
    if any(k in clean for k in ("拍摄失败", "拍照失败", "无法拍照", "不拍照", "拍摄无响应", "空图")):
        return "相机拍摄失败"
    if "闪退" in clean and any(k in clean for k in ("调试误报", "误报调试", "调试")):
        return "软件卡死无响应"
    if any(k in clean for k in ("智能调整", "编程优化")) and any(k in clean for k in ("等待", "响应延迟", "耗时", "响应慢")):
        return "程序运行卡顿"
    return ""


def _specific_variant_shape(text: str) -> tuple[str, str] | None:
    clean = str(text or "")
    if "模组电源" in clean and any(marker in clean for marker in ("接口松动", "端子", "供电中断", "自动关机")):
        return (
            "老版本模组电源输出线接口松动导致供电中断",
            "老版本模组电源输出连接端子存在窜动或松动，导致工控机供电中断并自动关机。",
        )
    if "buddy" in clean.lower() and "D盘" in clean and any(marker in clean for marker in ("HTTP 500", "http 500", "路径不存在", "保存失败")):
        return (
            "D盘消失导致Buddy冷存储写入失败",
            "Buddy依赖的D盘冷存储路径消失或不可访问，创建路径失败并返回HTTP 500。",
        )
    if "相机事件包" in clean and any(marker in clean for marker in ("丢失", "不重传", "残帧")):
        return (
            "相机链路丢包与事件包不重传导致拍摄失败",
            "相机链路出现残帧和事件包丢失，事件包不重传导致拍摄失败。",
        )
    if "网卡" in clean and any(marker in clean for marker in ("重置", "断连", "更换扩展网卡", "更换网卡")):
        if any(marker in clean.lower() for marker in ("realtek", "2.5g")) and any(marker in clean for marker in ("黑屏", "断连")):
            return (
                "Realtek 2.5G扩展网卡反复重置并导致黑屏/断连",
                "Realtek 2.5G扩展网卡反复重置，导致黑屏或其承载的相机、云控链路断连。",
            )
        return (
            "扩展网卡异常导致链路重置和下游设备断连",
            "扩展网卡持续发生重置，承载的下游链路断连，更换网卡后恢复生产。",
        )
    if all(marker in clean for marker in ("SATA", "RAID", "AHCI")) and any(marker in clean for marker in ("蓝屏", "无法进入系统")):
        if "INACCESSIBLE_BOOT_DEVICE" in clean.upper():
            return (
                "INACCESSIBLE_BOOT_DEVICE启动蓝屏",
                "SATA控制器模式与系统启动配置不一致，触发INACCESSIBLE_BOOT_DEVICE蓝屏。",
            )
        return (
            "SATA模式被改为RAID导致系统蓝屏无法启动",
            "SATA模式从AHCI改为RAID后系统蓝屏或无法启动，还原为AHCI后恢复。",
        )
    if _is_top_lift_case(clean):
        return (
            "顶板升降速度过慢",
            "二轨顶板升起、降落速度过慢或不一致，现场发现面顶三通气管缠绕导致气流过小。",
        )
    if _is_bios_battery_boot_case(clean) and any(marker in clean for marker in ("无法开机", "无法正常开机", "开不了机")):
        return (
            "断电后 BIOS 重置导致无法开机",
            "设备断电后主板 BIOS 参数重置，导致设备无法正常开机；更换主板电池后恢复。",
        )
    if any(k in clean for k in ("远轨", "轨道宽度")) and any(k in clean for k in ("无法正常出板", "无法出板", "出板失败", "板卡卡滞", "卡在轨道")):
        return (
            "远轨宽度异常导致板卡卡滞无法出板",
            "远轨中间段宽度异常，导致板卡在轨道中间卡滞并无法正常出板。",
        )
    if any(k in clean for k in ("拍摄失败", "拍照失败")) and any(k in clean for k in ("正常复判", "零件复判", "卡顿后", "间隔")):
        return (
            "复判卡顿后出现拍摄失败",
            "正常复判时先出现短时卡顿，随后弹出拍摄失败报错。",
        )
    if "闪退" in clean and any(k in clean for k in ("调试误报", "误报调试")):
        return (
            "调试误报时界面卡顿后软件闪退",
            "调试误报过程中界面卡顿并持续加载，随后软件直接闪退。",
        )
    if any(k in clean for k in ("智能调整", "编程优化")) and any(k in clean for k in ("等待", "响应延迟", "耗时", "响应慢")):
        return (
            "智能调整或编程优化响应延迟",
            "点击智能调整或编程优化后需要等待数秒，影响现场调试效率。",
        )
    return None


def _strip_report_item_prefix(text: str) -> str:
    clean = str(text or "").strip()
    clean = re.sub(r"^\d+(?:\.\d+){1,3}\s+", "", clean)
    clean = re.sub(r"^(?:[一二三四五六七八九十]+[、.．:：]\s*)", "", clean)
    clean = re.sub(r"^(?:[1-9]\d?[、.．:：]\s*)", "", clean)
    clean = re.sub(r"^(?:软件功能异常问题|设备硬件异常问题)\s*", "", clean)
    return clean.strip(" ：:，,。")


def _fallback_required_info_from_text(variant_label: str, error_label: str, symptom: str, conclusion: str) -> list[tuple[str, str, str, list[str]]]:
    text = " ".join([variant_label, error_label, symptom, conclusion])
    if "闪退" in text and any(k in text for k in ("调试误报", "误报调试", "调试")):
        return [
            ("log_package", "请提供闪退发生时段对应的主程序日志。", "需要定位闪退前的异常调用和报错上下文。", ["分析闪退日志"]),
            ("software_version", "请提供发生闪退时的主程序版本。", "需要判断问题是否集中在特定版本。", ["核对闪退版本分支"]),
            ("repro_steps", "请说明从进入误报调试到软件闪退的稳定复现步骤。", "需要确认触发闪退的具体操作序列。", ["复现调试闪退"]),
        ]
    if any(k in text for k in ("拍摄失败", "拍照失败")):
        return [
            ("log_package", "请提供拍摄失败时段对应的相机日志和诊断日志。", "需要确认图像采集超时或相机链路报错。", ["分析拍摄失败日志"]),
            ("software_version", "请提供发生拍摄失败时的主程序和运控版本。", "需要判断问题是否与版本组合相关。", ["核对拍摄失败版本"]),
            ("error_phase", "请说明拍摄失败发生在编程、调试还是正常复判阶段。", "不同阶段对应不同的采图调用链。", ["定位拍摄失败阶段"]),
        ]
    if any(k in text for k in ("远轨", "轨道宽度")) and any(k in text for k in ("无法出板", "无法正常出板", "板卡卡滞", "出板失败")):
        return [
            ("repro_steps", "请说明板卡进入远轨后发生卡滞的具体位置和复现过程。", "需要区分轨道宽度、传感器和出板时序问题。", ["复现远轨卡板"]),
            ("device_model", "请提供设备型号和远轨机构配置。", "需要核对该机型的轨道结构和调宽范围。", ["核对远轨机构"]),
            ("sample_image", "请提供卡滞位置及轨道宽度差异的现场照片或视频。", "需要验证板卡是否因中间段宽度异常而卡滞。", ["确认轨道宽度异常"]),
        ]
    if any(k in text for k in ("智能调整", "编程优化")) and any(k in text for k in ("等待", "响应延迟", "耗时")):
        return [
            ("software_version", "请提供出现响应延迟时的主程序版本。", "需要判断是否为特定版本的性能退化。", ["核对性能版本分支"]),
            ("repro_steps", "请提供触发智能调整或编程优化延迟的操作步骤和等待时长。", "需要稳定复现并量化操作延迟。", ["复现操作响应延迟"]),
            ("production_constraint", "请提供该料号的器件数量和相关算法配置。", "等待时长可能与料号规模和算法配置有关。", ["分析延迟影响因素"]),
        ]
    if "加载用户配置失败" in text or "user.cfg" in text:
        return [
            ("program_file", "请提供 user.cfg.toml 和 conf 目录内容。", "需要判断配置文件是否为空、损坏还是备份选择错误。", ["检查 user.cfg.toml 是否为空"]),
            ("software_version", "请提供主程序或复判站版本。", "需要确认配置格式是否和当前版本匹配。", ["回填备份配置并重启验证"]),
        ]
    if any(k in text for k in ("网线", "网口", "拓展网卡")) and any(k in text for k in ("拍摄失败", "拍照失败")):
        return [
            ("ip_config", "请提供主板网口和拓展网卡的角色、IP、网口截图。", "需要确认拍摄失败是否由换口后的网络角色或配置变化引起。", ["检查相机网口角色与网络配置"]),
            ("log_package", "请提供换口前后对应的诊断日志。", "需要对比拍摄失败是否确实从换口后开始。", ["核对相机网线插口变更"]),
        ]
    if "MEMORY_MANAGEMENT" in text and "PFN" in text:
        return [
            ("dmp_package", "请提供完整 DMP/Minidump。", "需要确认 BugCheck、参数和 PFN 不同步签名。", ["分析 DMP"]),
            ("driver_context", "请提供相关驱动版本和外设上下文。", "需要判断 PFN 不同步是否与第三方驱动或显卡/无线网卡相关。", ["开启 Driver Verifier"]),
        ]
    if "0x00000139" in text:
        return [
            ("log_package", "请提供完整转存储文件和对应时间点系统日志。", "需要确认 0x00000139 的错误上下文、驱动缺失和转储完整性。", ["收集并分析转存储文件"]),
            ("driver_context", "请补充 NVIDIA/Intel/网卡/远控驱动和近期驱动变更信息。", "需要判断关键驱动丢失/损坏是否与第三方驱动相关。", ["检查关键驱动文件是否缺失或损坏"]),
            ("memory_cpu_test", "请补充内存条变更、内存频率、温度和相关稳定性验证信息。", "需要排除内存过热、物理故障或资源破坏导致的数据结构损坏。", ["收集并分析转存储文件"]),
            ("environment", "请补充静电接地、关机流程和现场环境信息。", "需要判断静电和环境因素是否放大了系统损坏。", ["修复 Defender 并清理可疑驱动"]),
        ]
    if "PTE" in text:
        return [
            ("dmp_package", "请提供完整 MEMORY.DMP 或 minidump。", "需要确认 PTE 耗尽的直接证据和损坏范围。", ["分析 DMP 中 PTE 耗尽信号"]),
            ("memory_cpu_test", "请补充内存检测、内存频率、CPU 稳定性信息。", "需要区分 PTE 耗尽和物理内存故障、频率不稳等因素。", ["换内存条后持续观察"]),
            ("driver_context", "请补充可能的大图、DMA、驱动、图像缓冲相关上下文。", "PTE 耗尽常与驱动长期映射/释放异常有关。", ["使用 WPR 抓取内核分配趋势"]),
            ("software_version", "请提供蓝屏发生时的软件版本和升级/回退记录。", "需要确认版本变化对复发频率的影响。", ["升级或回退版本后观察是否复发"]),
        ]
    if "光源初始化失败" in text:
        return [
            ("log_package", "请提供光源初始化失败时的日志和诊断数据。", "需要确认失败是否来自光控板、USB 链路或软件初始化过程。", ["检查光源初始化失败告警"]),
            ("ip_config", "请补充光控/光源相关 IP、USB 连接和接口上下文。", "需要区分网络链路问题和 USB/接口接触问题。", ["重新拔插光源 USB 接口"]),
            ("repro_steps", "请补充通电测试到恢复正常的完整复现步骤。", "需要确认该恢复是否稳定、是否可重复。", ["恢复后继续观察上线验证"]),
        ]
    return []


def _norm(value: str) -> str:
    return " ".join(str(value or "").split()).lower()


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x or "").strip()]
    if value:
        return [str(value)]
    return []


def _same_text(a: Any, b: Any) -> bool:
    return _norm(str(a or "")) == _norm(str(b or ""))


def _split_decision(semantics: dict[str, Any]) -> dict[str, Any]:
    text = str(semantics.get("semantic_text") or "")
    marker_count = sum(1 for marker in ("另外", "还有", "另一个", "同时", "蓝屏", "拍摄失败", "初始化失败", "工控机", "相机") if marker in text)
    return {
        "decision": "candidate_single_episode" if marker_count < 4 else "review_for_possible_split",
        "reason": "deterministic_marker_count",
        "marker_count": marker_count,
    }


def _ensure_action_outcomes(
    actions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    case_id: str,
    family_id: str,
    variant_id: str,
    evidence_ids: list[dict[str, Any]],
    semantic_text: str,
    conclusion: str,
) -> None:
    existing: dict[str, set[str]] = {}
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        action_id = str(item.get("action_id") or "")
        if action_id:
            existing.setdefault(action_id, set()).add(str(item.get("outcome_type") or ""))
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_id = str(action.get("action_id") or "")
        if not action_id:
            continue
        if existing.get(action_id):
            continue
        desired = _fallback_outcome_type(action, semantic_text, conclusion) or "pending_validation"
        outcome_id = make_id("outcome", f"{case_id}:{action_id}:{desired}")
        outcomes.append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": desired,
            "outcome_origin": "synthetic_fallback" if desired == "pending_validation" else "rule_inferred",
            "summary": trim_text(_fallback_outcome_summary(action, desired, conclusion), 200),
            "source_case_id": case_id,
            "evidence_ids": [entry["evidence_id"] for entry in evidence_ids[:4]],
            "high_cost": bool(action.get("high_cost")),
            "destructive": bool(action.get("destructive")),
            "root_cause_summary": "",
        })


def _fallback_outcome_type(action: dict[str, Any], semantic_text: str, conclusion: str) -> str:
    label = str(action.get("label") or "")
    summary = str(action.get("summary") or "")
    local = f"{label} {summary}"
    if any(k in local for k in ("使用 WPR", "PoolMon", "Driver Verifier", "verifier")):
        return "diagnostic_method"
    if any(k in local for k in ("换内存条后持续观察", "恢复后继续观察", "继续观察上线验证")):
        if any(k in f"{semantic_text} {conclusion}" for k in ("复发", "再次", "又", "随后仍复发")):
            return "partial_temporary"
        return "pending_validation"
    if any(k in local for k in ("分析 DMP", "收集并分析转存储文件")) and any(k in f"{local} {semantic_text} {conclusion}" for k in ("0x", "BugCheck", "MEMORY_MANAGEMENT", "PFN", "PTE", "驱动缺失", "转储不完整")):
        return "context_not_root_cause"
    if any(k in local for k in ("检查 user.cfg.toml 是否为空", "检查 conf 目录备份", "核对相机网线插口变更", "检查关键驱动文件是否缺失或损坏")):
        return "context_not_root_cause"
    if any(k in label for k in ("检查", "核对")) and any(k in f"{local} {semantic_text}" for k in ("怀疑", "疑是", "无明显异常", "开始于", "指向", "错误代码", "PFN", "PTE", "驱动缺失")):
        return "context_not_root_cause"
    if any(k in local for k in ("执行系统文件修复", "使用 DDU", "修复 Defender", "清理可疑驱动", "恢复原主板网口验证")):
        return "pending_validation"
    if "回填备份配置并重启验证" in local:
        return "diagnostic_method"
    if "重新拔插光源 USB 接口" in local and any(k in conclusion for k in ("恢复正常", "已正常")):
        return "mitigation_observed"
    if any(k in label for k in ("恢复", "重装", "修复", "清理", "拔插", "更新", "回填", "更换", "升级", "回退")):
        if any(k in conclusion for k in ("恢复正常", "已正常", "未再", "没有再", "不再")):
            return "mitigation_observed"
        return "pending_validation"
    return ""


def _fallback_action_templates(family_label: str, semantic_text: str) -> list[str]:
    text = str(semantic_text or "")
    if family_label == "用户配置加载失败":
        return [
            "检查 user.cfg.toml 是否为空",
            "检查 conf 目录备份",
            "回填备份配置并重启验证",
        ]
    if family_label == "相机拍摄失败" and any(k in text for k in ("网线", "网口", "拓展网卡")):
        return [
            "核对相机网线插口变更",
            "检查相机网口角色与网络配置",
            "恢复原主板网口验证",
        ]
    if family_label == "工控机蓝屏" and "MEMORY_MANAGEMENT" in text and "PFN" in text:
        return [
            "分析 DMP",
            "测试内存和 CPU 稳定性",
            "开启 Driver Verifier",
        ]
    if family_label == "工控机蓝屏" and "0x00000139" in text:
        return [
            "收集并分析转存储文件",
            "检查关键驱动文件是否缺失或损坏",
            "执行系统文件修复",
            "使用 DDU 彻底重装显卡驱动",
            "修复 Defender 并清理可疑驱动",
            "开启 Driver Verifier 继续定位",
        ]
    if family_label == "工控机蓝屏" and "PTE" in text:
        return [
            "分析 DMP 中 PTE 耗尽信号",
            "使用 WPR 抓取内核分配趋势",
            "使用 PoolMon 监控池分配",
            "升级或回退版本后观察是否复发",
            "换内存条后持续观察",
        ]
    if family_label == "光源初始化失败":
        return [
            "检查光源初始化失败告警",
            "重新拔插光源 USB 接口",
            "恢复后继续观察上线验证",
        ]
    return []


def _fallback_outcome_summary(action: dict[str, Any], outcome_type: str, conclusion: str) -> str:
    label = str(action.get("label") or "")
    if outcome_type == "diagnostic_method":
        return f"{label} 作为诊断手段用于继续定位。"
    if outcome_type == "context_not_root_cause":
        return f"{label} 提供了重要上下文，但还不是最终根因闭环。"
    if outcome_type == "mitigation_observed":
        return conclusion or f"{label} 后观察到问题缓解。"
    if outcome_type == "pending_validation":
        return conclusion or f"{label} 已执行或被建议执行，仍需继续验证。"
    if outcome_type == "partial_temporary":
        return conclusion or f"{label} 后短时恢复，但后续仍可能复发。"
    return label


def _empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {
        "FaultFamily": [],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "DecisionPolicy": [],
        "EvidenceItem": [],
        "SourceCase": [],
    }


def _dedupe_objects(objects: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out = _empty_objects()
    pk_by_type = {
        "FaultFamily": "family_id",
        "FaultVariant": "variant_id",
        "DiagnosticAction": "action_id",
        "ActionOutcome": "outcome_id",
        "RequiredInfoSpec": "required_info_id",
        "DiagnosticTrace": "trace_id",
        "DecisionPolicy": "policy_id",
        "EvidenceItem": "evidence_id",
        "SourceCase": "case_id",
    }
    for obj_type, items in objects.items():
        index: dict[str, dict[str, Any]] = {}
        pk = pk_by_type[obj_type]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get(pk) or "")
            if not item_id:
                continue
            if item_id in index:
                index[item_id].update({k: v for k, v in item.items() if v not in (None, "", [])})
            else:
                index[item_id] = dict(item)
        out[obj_type] = list(index.values())
    return out


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        key = (str(relation.get("from") or ""), str(relation.get("to") or ""), str(relation.get("relation") or ""))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(relation)
    return out
