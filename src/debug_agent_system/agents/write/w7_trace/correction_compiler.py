"""Deterministically compile W6 Trace corrections back into W7 decisions.

The model is deliberately absent from this module.  Human correction events
operate on bounded case/trace/message references, are replayed in sequence,
and produce a new content-addressed review payload.  Structural case edits are
compiled at the semantic layer but remain fail-closed for KG materialization
until W2 has re-extracted the new atomic cases.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import (
    TRACE_EVENT_TYPES,
    TRACE_RELATION_TYPES,
    canonical_hash,
    dedupe_strings,
    validate_outcome_patch,
    validate_trace_phase_patch,
)
from .review import (
    build_trace_review_payload,
    correction_event_hash,
    replay_correction_events,
)
from .trace_compiler import TraceCompiler


STRUCTURAL_CASE_OPERATIONS = {"split_case", "merge_cases"}


def _case_ref(value: dict[str, Any]) -> str:
    return str(
        value.get("case_ref")
        or value.get("case_item_ref")
        or value.get("fragment_ref")
        or ""
    )


def _case_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("case_cards")
    if isinstance(explicit, list):
        return [
            deepcopy(item) for item in explicit
            if isinstance(item, dict) and _case_ref(item)
        ]
    bundle = (
        payload.get("compiled_trace_bundle")
        if isinstance(payload.get("compiled_trace_bundle"), dict)
        else {}
    )
    cards: dict[str, dict[str, Any]] = {}
    for trace in bundle.get("traces") or []:
        if not isinstance(trace, dict):
            continue
        for card in trace.get("case_cards") or []:
            if not isinstance(card, dict):
                continue
            ref = _case_ref(card)
            if ref:
                cards[ref] = deepcopy(card)
    boundary = (
        (payload.get("decisions") or {}).get("case_boundary")
        if isinstance(payload.get("decisions"), dict)
        else {}
    )
    if isinstance(boundary, dict):
        for fragment in boundary.get("case_fragments") or []:
            if not isinstance(fragment, dict):
                continue
            ref = str(fragment.get("fragment_ref") or "")
            if ref and ref not in cards:
                cards[ref] = {
                    **deepcopy(fragment),
                    "case_ref": ref,
                    "title": str(fragment.get("fault_summary") or ""),
                }
    return [cards[key] for key in sorted(cards)]


def _phase_state(
    payload: dict[str, Any],
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    set[str],
]:
    decisions = (
        payload.get("decisions")
        if isinstance(payload.get("decisions"), dict)
        else {}
    )
    phase = (
        decisions.get("trace_phase")
        if isinstance(decisions.get("trace_phase"), dict)
        else {}
    )
    groups: dict[str, list[str]] = {}
    phases: dict[str, dict[str, Any]] = {}
    for operation in phase.get("operations") or []:
        if not isinstance(operation, dict):
            continue
        trace_ref = str(operation.get("local_trace_ref") or "")
        if operation.get("op") == "create_trace_group" and trace_ref:
            groups[trace_ref] = dedupe_strings(
                operation.get("case_refs") or []
            )
        elif operation.get("op") == "set_phase":
            case_ref = str(operation.get("case_ref") or "")
            if trace_ref and case_ref:
                phases[case_ref] = deepcopy(operation)
    return groups, phases, set(
        dedupe_strings(phase.get("standalone_case_refs") or [])
    )


def _outcome_state(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions = (
        payload.get("decisions")
        if isinstance(payload.get("decisions"), dict)
        else {}
    )
    outcome = (
        decisions.get("outcome_reconciliation")
        if isinstance(decisions.get("outcome_reconciliation"), dict)
        else {}
    )
    return {
        str(item.get("local_trace_ref") or ""): deepcopy(item)
        for item in outcome.get("operations") or []
        if isinstance(item, dict)
        and str(item.get("local_trace_ref") or "")
    }


def _trace_for_case(
    groups: dict[str, list[str]], case_ref: str
) -> str:
    return next((
        trace_ref
        for trace_ref, refs in groups.items()
        if case_ref in refs
    ), "")


def _trace_aliases(payload: dict[str, Any]) -> dict[str, str]:
    bundle = (
        payload.get("compiled_trace_bundle")
        if isinstance(payload.get("compiled_trace_bundle"), dict)
        else {}
    )
    aliases: dict[str, str] = {}
    for trace in bundle.get("traces") or []:
        if not isinstance(trace, dict):
            continue
        local = str(trace.get("local_trace_ref") or "")
        compiled = str(trace.get("compiled_trace_id") or "")
        if local:
            aliases[local] = local
        if local and compiled:
            aliases[compiled] = local
    return aliases


def _normalize_group_phases(
    groups: dict[str, list[str]],
    phases: dict[str, dict[str, Any]],
) -> None:
    for trace_ref in list(groups):
        refs = dedupe_strings(groups[trace_ref])
        refs = [ref for ref in refs if ref in phases]
        if not refs:
            groups.pop(trace_ref, None)
            continue
        groups[trace_ref] = refs
        previous = ""
        for index, ref in enumerate(refs, 1):
            phase = phases[ref]
            phase.update({
                "op": "set_phase",
                "local_trace_ref": trace_ref,
                "case_ref": ref,
                "case_refs": [],
                "phase_index": index,
                "after_case_ref": previous,
            })
            if index == 1:
                phase["relation_type"] = "trace_root"
            elif str(phase.get("relation_type") or "") == "trace_root":
                phase["relation_type"] = "continuation_of"
            if str(phase.get("event_type") or "") not in TRACE_EVENT_TYPES:
                phase["event_type"] = "report"
            if (
                str(phase.get("relation_type") or "")
                not in TRACE_RELATION_TYPES
            ):
                phase["relation_type"] = (
                    "trace_root" if index == 1 else "continuation_of"
                )
            phase["evidence_message_ids"] = dedupe_strings(
                phase.get("evidence_message_ids") or []
            )
            previous = ref


def _new_case_card(
    raw: dict[str, Any],
    *,
    inherited: dict[str, Any],
) -> dict[str, Any]:
    ref = str(
        raw.get("case_ref")
        or raw.get("fragment_ref")
        or raw.get("case_item_ref")
        or ""
    )
    source_ids = dedupe_strings(
        raw.get("source_message_ids")
        or raw.get("evidence_message_ids")
        or []
    )
    return {
        **deepcopy(inherited),
        **deepcopy(raw),
        "case_ref": ref,
        "source_case_id": "",
        "source_message_ids": source_ids,
        "evidence_message_ids": dedupe_strings(
            raw.get("evidence_message_ids") or source_ids
        ),
        "title": str(
            raw.get("title")
            or raw.get("fault_summary")
            or inherited.get("title")
            or ""
        ),
        "fault_summary": str(
            raw.get("fault_summary")
            or raw.get("title")
            or inherited.get("fault_summary")
            or ""
        ),
    }


def compile_trace_corrections(
    *,
    trace_review_payload: dict[str, Any],
    correction_events: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Apply ordered correction events and return a new review payload.

    The result is safe for another human review.  ``kg_materialization_ready``
    is false for split/merge because those operations change W2 case identity
    and therefore require atomic re-extraction before any KG candidate can be
    updated.
    """

    replay, replay_issues = replay_correction_events(
        trace_review_payload, correction_events
    )
    if replay_issues:
        return {
            "schema_version": "w7.correction_compile_result.v1",
            "status": "failed_closed",
            "correction_overlay": replay,
        }, replay_issues

    cards = _case_cards(trace_review_payload)
    cards_by_ref = {_case_ref(card): card for card in cards}
    groups, phases, standalone = _phase_state(trace_review_payload)
    outcomes = _outcome_state(trace_review_payload)
    aliases = _trace_aliases(trace_review_payload)
    allowed_messages = set(
        dedupe_strings(trace_review_payload.get("allowed_message_ids") or [])
    )
    issues: list[str] = []
    applied: list[str] = []
    requires_w2_reextract: list[str] = []

    for event in sorted(
        correction_events,
        key=lambda item: int(item.get("sequence") or 0),
    ):
        operation = str(event.get("operation") or "")
        target = str(event.get("target_ref") or "")
        data = (
            event.get("payload")
            if isinstance(event.get("payload"), dict)
            else {}
        )
        evidence = dedupe_strings(event.get("evidence_message_ids") or [])
        trace_ref = aliases.get(target, target if target in groups else "")

        if operation == "change_status":
            if not trace_ref:
                issues.append(f"correction_status_target_not_trace:{target}")
                continue
            outcomes[trace_ref] = {
                "op": "revise_trace_status",
                "local_trace_ref": trace_ref,
                "from": str(
                    (outcomes.get(trace_ref) or {}).get("to") or ""
                ),
                "to": str(data.get("to") or ""),
                "evidence_message_ids": evidence,
                "reason": str(data.get("reason") or event.get("note") or ""),
            }
        elif operation == "move_phase":
            if target not in phases:
                issues.append(f"correction_phase_target_not_case:{target}")
                continue
            current_trace = _trace_for_case(groups, target)
            destination = aliases.get(
                str(data.get("local_trace_ref") or ""),
                str(data.get("local_trace_ref") or current_trace),
            )
            if destination not in groups:
                issues.append(
                    f"correction_phase_destination_unknown:{destination}"
                )
                continue
            for refs in groups.values():
                if target in refs:
                    refs.remove(target)
            index = max(1, int(data.get("phase_index") or 1)) - 1
            groups[destination].insert(
                min(index, len(groups[destination])), target
            )
            phases[target]["local_trace_ref"] = destination
        elif operation == "change_relation":
            if target not in phases:
                issues.append(
                    f"correction_relation_target_not_case:{target}"
                )
                continue
            relation = str(data.get("relation_type") or "")
            phases[target]["relation_type"] = relation
            if relation == "trace_root":
                current_trace = _trace_for_case(groups, target)
                if current_trace:
                    groups[current_trace].remove(target)
                    groups[current_trace].insert(0, target)
        elif operation in {"attach_evidence", "detach_evidence"}:
            if target in cards_by_ref:
                card = cards_by_ref[target]
                current = dedupe_strings(
                    card.get("evidence_message_ids")
                    or card.get("source_message_ids")
                    or []
                )
                if operation == "attach_evidence":
                    current = dedupe_strings([*current, *evidence])
                else:
                    current = [
                        value for value in current if value not in set(evidence)
                    ]
                card["evidence_message_ids"] = current
                card["source_message_ids"] = dedupe_strings([
                    *(card.get("source_message_ids") or []),
                    *(
                        evidence
                        if operation == "attach_evidence"
                        else []
                    ),
                ])
                if operation == "detach_evidence":
                    card["source_message_ids"] = [
                        value
                        for value in card["source_message_ids"]
                        if value not in set(evidence)
                    ]
                if target in phases:
                    phase_evidence = dedupe_strings(
                        phases[target].get("evidence_message_ids") or []
                    )
                    phases[target]["evidence_message_ids"] = (
                        dedupe_strings([*phase_evidence, *evidence])
                        if operation == "attach_evidence"
                        else [
                            value for value in phase_evidence
                            if value not in set(evidence)
                        ]
                    )
            elif trace_ref:
                current = dedupe_strings(
                    (outcomes.get(trace_ref) or {}).get(
                        "evidence_message_ids"
                    )
                    or []
                )
                updated = (
                    dedupe_strings([*current, *evidence])
                    if operation == "attach_evidence"
                    else [
                        value for value in current
                        if value not in set(evidence)
                    ]
                )
                outcomes.setdefault(trace_ref, {
                    "op": "revise_trace_status",
                    "local_trace_ref": trace_ref,
                    "from": "",
                    "to": "unknown",
                    "reason": "",
                })["evidence_message_ids"] = updated
            else:
                issues.append(
                    f"correction_evidence_target_unknown:{target}"
                )
                continue
        elif operation == "detach_case":
            if target not in cards_by_ref:
                issues.append(f"correction_detach_case_unknown:{target}")
                continue
            for refs in groups.values():
                if target in refs:
                    refs.remove(target)
            phases.pop(target, None)
            standalone.add(target)
        elif operation == "split_case":
            original = cards_by_ref.get(target)
            new_values = [
                value for value in data.get("new_cases") or []
                if isinstance(value, dict)
            ]
            if original is None or len(new_values) < 2:
                issues.append(f"correction_split_invalid:{target}")
                continue
            new_cards = [
                _new_case_card(value, inherited=original)
                for value in new_values
            ]
            new_refs = [_case_ref(value) for value in new_cards]
            if (
                any(not value for value in new_refs)
                or len(set(new_refs)) != len(new_refs)
                or any(
                    value in cards_by_ref and value != target
                    for value in new_refs
                )
            ):
                issues.append(f"correction_split_case_ref_invalid:{target}")
                continue
            original_ids = set(
                dedupe_strings(
                    original.get("source_message_ids")
                    or original.get("evidence_message_ids")
                    or []
                )
            )
            new_id_sets = [
                set(card.get("source_message_ids") or [])
                for card in new_cards
            ]
            if any(not values for values in new_id_sets):
                issues.append(f"correction_split_empty_source:{target}")
                continue
            if any(
                left & right
                for index, left in enumerate(new_id_sets)
                for right in new_id_sets[index + 1:]
            ):
                issues.append(f"correction_split_source_overlap:{target}")
                continue
            if set().union(*new_id_sets) != original_ids:
                issues.append(f"correction_split_source_not_exhaustive:{target}")
                continue
            current_trace = _trace_for_case(groups, target)
            original_phase = deepcopy(phases.get(target) or {})
            if current_trace:
                index = groups[current_trace].index(target)
                groups[current_trace][index:index + 1] = new_refs
            was_standalone = target in standalone
            standalone.discard(target)
            phases.pop(target, None)
            cards_by_ref.pop(target, None)
            for offset, card in enumerate(new_cards):
                ref = _case_ref(card)
                cards_by_ref[ref] = card
                if current_trace:
                    phases[ref] = {
                        **original_phase,
                        "op": "set_phase",
                        "local_trace_ref": current_trace,
                        "case_ref": ref,
                        "case_refs": [],
                        "phase_index": int(
                            original_phase.get("phase_index") or 1
                        ) + offset,
                        "relation_type": (
                            str(original_phase.get("relation_type") or "")
                            if offset == 0
                            else "continuation_of"
                        ),
                        "evidence_message_ids": dedupe_strings(
                            card.get("evidence_message_ids") or []
                        ),
                        "summary": str(
                            card.get("fault_summary")
                            or card.get("title")
                            or ""
                        ),
                    }
                elif was_standalone:
                    standalone.add(ref)
            requires_w2_reextract.append(operation)
        elif operation == "merge_cases":
            refs = dedupe_strings(data.get("case_refs") or [])
            if len(refs) < 2 or any(
                ref not in cards_by_ref for ref in refs
            ):
                issues.append(f"correction_merge_invalid:{target}")
                continue
            survivor = target if target in refs else refs[0]
            survivor_card = cards_by_ref[survivor]
            merged_ids = dedupe_strings(
                message_id
                for ref in refs
                for message_id in (
                    cards_by_ref[ref].get("source_message_ids")
                    or cards_by_ref[ref].get("evidence_message_ids")
                    or []
                )
            )
            survivor_card["source_message_ids"] = merged_ids
            survivor_card["evidence_message_ids"] = merged_ids
            survivor_card["title"] = str(
                data.get("title") or survivor_card.get("title") or ""
            )
            survivor_card["fault_summary"] = str(
                data.get("fault_summary")
                or survivor_card.get("fault_summary")
                or survivor_card.get("title")
                or ""
            )
            destination = _trace_for_case(groups, survivor)
            if not destination:
                destination = next(
                    (
                        _trace_for_case(groups, ref)
                        for ref in refs
                        if _trace_for_case(groups, ref)
                    ),
                    "",
                )
            phase_template = deepcopy(
                phases.get(survivor)
                or next(
                    (
                        phases[ref] for ref in refs if ref in phases
                    ),
                    {},
                )
            )
            for group_refs in groups.values():
                group_refs[:] = [
                    ref for ref in group_refs if ref not in set(refs)
                ]
            if destination:
                groups[destination].append(survivor)
                phases[survivor] = {
                    **phase_template,
                    "local_trace_ref": destination,
                    "case_ref": survivor,
                    "evidence_message_ids": merged_ids,
                }
            else:
                standalone.add(survivor)
            for ref in refs:
                if ref == survivor:
                    continue
                cards_by_ref.pop(ref, None)
                phases.pop(ref, None)
                standalone.discard(ref)
            requires_w2_reextract.append(operation)
        else:
            issues.append(f"unsupported_correction_operation:{operation}")
            continue
        applied.append(str(event.get("event_id") or operation))

    for message_id in sorted(
        {
            value
            for card in cards_by_ref.values()
            for value in (
                card.get("source_message_ids")
                or card.get("evidence_message_ids")
                or []
            )
        }
        - allowed_messages
    ):
        issues.append(f"corrected_case_unknown_message:{message_id}")

    _normalize_group_phases(groups, phases)
    assigned = {
        case_ref for refs in groups.values() for case_ref in refs
    }
    for case_ref in list(standalone):
        if case_ref not in cards_by_ref:
            standalone.discard(case_ref)
    all_refs = set(cards_by_ref)
    missing = all_refs - assigned - standalone
    for case_ref in sorted(missing):
        issues.append(f"corrected_case_unassigned:{case_ref}")

    phase_operations: list[dict[str, Any]] = []
    for trace_ref, refs in sorted(groups.items()):
        phase_operations.append({
            "op": "create_trace_group",
            "local_trace_ref": trace_ref,
            "case_refs": refs,
            "case_ref": "",
            "event_type": "",
            "relation_type": "",
            "phase_index": 0,
            "after_case_ref": "",
            "evidence_message_ids": [],
            "summary": "",
        })
        phase_operations.extend(
            deepcopy(phases[ref]) for ref in refs if ref in phases
        )
    phase_patch = {
        "schema_version": "w7.trace_phase_patch.v1",
        "operations": phase_operations,
        "standalone_case_refs": sorted(standalone),
        "uncertainties": [],
    }
    phase_patch["decision_hash"] = canonical_hash(phase_patch)
    allowed_by_case = {
        ref: set(
            dedupe_strings(
                card.get("source_message_ids")
                or card.get("evidence_message_ids")
                or []
            )
        )
        for ref, card in cards_by_ref.items()
    }
    _, phase_issues = validate_trace_phase_patch(
        phase_patch,
        component_case_refs=assigned,
        allowed_message_ids=allowed_messages,
        allowed_message_ids_by_case=allowed_by_case,
    )
    issues.extend(phase_issues)

    outcome_patch = {
        "schema_version": "w7.outcome_patch.v1",
        "operations": [
            outcomes[key] for key in sorted(outcomes) if key in groups
        ],
        "uncertainties": [],
    }
    normalized_outcome, outcome_issues = validate_outcome_patch(
        outcome_patch,
        allowed_trace_refs=set(groups),
        allowed_message_ids=allowed_messages,
    )
    issues.extend(outcome_issues)

    cards = [cards_by_ref[key] for key in sorted(cards_by_ref)]
    compiled = TraceCompiler().compile_review_bundle(
        case_cards=cards,
        phase_patch=phase_patch,
        outcome_patch=normalized_outcome,
    )
    decisions = deepcopy(trace_review_payload.get("decisions") or {})
    decisions["trace_phase"] = phase_patch
    decisions["outcome_reconciliation"] = normalized_outcome
    corrected_payload = build_trace_review_payload(
        source_ledger_hash=str(
            trace_review_payload.get("source_ledger_hash") or ""
        ),
        decisions=decisions,
        compiled_trace_bundle=compiled,
        dry_run_diff=list(
            trace_review_payload.get("dry_run_diff") or []
        ),
        validator_issues=sorted(set(issues)),
        allowed_message_ids=sorted(allowed_messages),
        case_cards=cards,
        correction_provenance={
            "base_review_payload_hash": str(
                trace_review_payload.get("review_payload_hash") or ""
            ),
            "correction_overlay_hash": str(
                replay.get("effective_bundle_hash") or ""
            ),
            "correction_event_hashes": [
                correction_event_hash(event)
                for event in correction_events
                if isinstance(event, dict)
            ],
        },
    )
    result = {
        "schema_version": "w7.correction_compile_result.v1",
        "status": "compiled" if not issues else "failed_closed",
        "correction_overlay": replay,
        "corrected_trace_review_payload": corrected_payload,
        "corrected_compiled_trace_bundle": compiled,
        "case_cards": cards,
        "applied_event_ids": applied,
        "requires_w2_reextract": bool(requires_w2_reextract),
        "w2_reextract_operations": dedupe_strings(requires_w2_reextract),
        "kg_materialization_ready": (
            not issues and not requires_w2_reextract
        ),
    }
    result["compile_result_hash"] = canonical_hash(result)
    return result, sorted(set(issues))


def _evidence_by_message_id(
    objects: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for evidence in objects.get("EvidenceItem") or []:
        if not isinstance(evidence, dict):
            continue
        evidence_id = str(evidence.get("evidence_id") or "")
        if not evidence_id:
            continue
        values = dedupe_strings([
            evidence.get("message_id"),
            evidence.get("source_message_id"),
            evidence.get("external_id"),
            evidence.get("source_ref"),
        ])
        for value in values:
            output.setdefault(value, []).append(evidence_id)
    return output


def materialize_corrected_typed_candidate(
    *,
    typed_candidate: dict[str, Any],
    correction_compile_result: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Materialize non-structural corrections into one typed KG candidate.

    Longitudinal traces may contain the same canonical action more than once.
    ``action_occurrences`` preserves those execution instances while
    ``recommended_action_ids`` remains the deduplicated action catalog used by
    existing readers.
    """

    issues: list[str] = []
    if not bool(
        correction_compile_result.get("kg_materialization_ready")
    ):
        return {}, ["correction_requires_w2_reextract"]
    corrected_payload = (
        correction_compile_result.get("corrected_trace_review_payload")
        if isinstance(
            correction_compile_result.get(
                "corrected_trace_review_payload"
            ),
            dict,
        )
        else {}
    )
    semantic_bundle = (
        correction_compile_result.get("corrected_compiled_trace_bundle")
        if isinstance(
            correction_compile_result.get(
                "corrected_compiled_trace_bundle"
            ),
            dict,
        )
        else {}
    )
    objects = (
        deepcopy(typed_candidate.get("objects"))
        if isinstance(typed_candidate.get("objects"), dict)
        else {}
    )
    relations = [
        deepcopy(item)
        for item in typed_candidate.get("relations") or []
        if isinstance(item, dict)
    ]
    if not objects:
        return {}, ["typed_candidate_objects_missing"]
    source_cases = {
        str(item.get("case_id") or ""): item
        for item in objects.get("SourceCase") or []
        if isinstance(item, dict) and str(item.get("case_id") or "")
    }
    actions = {
        str(item.get("action_id") or ""): item
        for item in objects.get("DiagnosticAction") or []
        if isinstance(item, dict) and str(item.get("action_id") or "")
    }
    old_traces = [
        item for item in objects.get("DiagnosticTrace") or []
        if isinstance(item, dict)
    ]
    old_traces_by_case: dict[str, list[dict[str, Any]]] = {}
    for trace in old_traces:
        old_traces_by_case.setdefault(
            str(trace.get("source_case_id") or ""), []
        ).append(trace)
    old_steps_by_trace: dict[str, list[dict[str, Any]]] = {}
    for step in objects.get("TraceStep") or []:
        if isinstance(step, dict):
            old_steps_by_trace.setdefault(
                str(step.get("trace_id") or ""), []
            ).append(step)
    for values in old_steps_by_trace.values():
        values.sort(key=lambda value: int(value.get("ordinal") or 0))
    evidence_by_message = _evidence_by_message_id(objects)
    card_by_ref = {
        _case_ref(card): card
        for card in corrected_payload.get("case_cards") or []
        if isinstance(card, dict) and _case_ref(card)
    }

    affected_trace_ids: set[str] = set()
    affected_execution_ids: set[str] = set()
    new_traces: list[dict[str, Any]] = []
    seed_steps: list[dict[str, Any]] = []
    new_relations: list[dict[str, Any]] = []
    used_source_cases: set[str] = set()
    review_only_trace_refs: list[str] = []

    for semantic_trace in semantic_bundle.get("traces") or []:
        if not isinstance(semantic_trace, dict):
            continue
        local_trace_ref = str(
            semantic_trace.get("local_trace_ref") or ""
        )
        compiled_trace_id = str(
            semantic_trace.get("compiled_trace_id") or ""
        )
        phase_values = [
            value for value in semantic_trace.get("phases") or []
            if isinstance(value, dict)
        ]
        ordered_case_refs = [
            str(value.get("case_ref") or "")
            for value in sorted(
                phase_values,
                key=lambda value: int(value.get("phase_index") or 0),
            )
            if str(value.get("case_ref") or "")
        ]
        source_case_ids = dedupe_strings(
            (card_by_ref.get(case_ref) or {}).get("source_case_id")
            for case_ref in ordered_case_refs
        )
        source_case_ids = [
            case_id for case_id in source_case_ids
            if case_id in source_cases
        ]
        if not source_case_ids:
            # W7 can assemble product requirements, field reports and
            # coordination/validation-only traces for human review even when
            # W2 correctly produced no SourceCase. Keep those semantics in the
            # hash-bound review bundle, but do not fabricate KG-v2 objects.
            review_only_trace_refs.append(local_trace_ref)
            continue
        used_source_cases.update(source_case_ids)
        root_case_id = source_case_ids[0]
        source_trace_values: list[tuple[
            str, str, dict[str, Any]
        ]] = []
        for case_ref in ordered_case_refs:
            case_id = str(
                (card_by_ref.get(case_ref) or {}).get(
                    "source_case_id"
                )
                or ""
            )
            if not case_id:
                continue
            for trace in old_traces_by_case.get(case_id) or []:
                source_trace_values.append((case_ref, case_id, trace))
                affected_trace_ids.add(
                    str(trace.get("trace_id") or "")
                )
        recommended: list[str] = []
        actual: list[str] = []
        trace_evidence: list[str] = []
        occurrences: list[dict[str, Any]] = []
        family_ids: list[str] = dedupe_strings(
            (card_by_ref.get(case_ref) or {}).get("family_id")
            for case_ref in ordered_case_refs
        )
        variant_ids: list[str] = dedupe_strings(
            (card_by_ref.get(case_ref) or {}).get("variant_id")
            for case_ref in ordered_case_refs
        )
        source_trace_ids: list[str] = []
        occurrence_count: dict[tuple[str, str], int] = {}
        phase_index_by_case = {
            str(value.get("case_ref") or ""): int(
                value.get("phase_index") or 0
            )
            for value in phase_values
        }
        for case_ref, case_id, trace in source_trace_values:
            trace_id = str(trace.get("trace_id") or "")
            source_trace_ids.append(trace_id)
            family_ids.append(str(trace.get("family_id") or ""))
            variant_ids.append(str(trace.get("variant_id") or ""))
            trace_evidence.extend(trace.get("evidence_ids") or [])
            trace_actions = dedupe_strings(
                trace.get("recommended_action_ids") or []
            )
            trace_actual = set(
                dedupe_strings(trace.get("actual_action_ids") or [])
            )
            recommended.extend(trace_actions)
            actual.extend(
                action_id
                for action_id in trace_actions
                if action_id in trace_actual
            )
            steps = old_steps_by_trace.get(trace_id) or []
            steps_by_action: dict[str, list[dict[str, Any]]] = {}
            for step in steps:
                steps_by_action.setdefault(
                    str(step.get("action_id") or ""), []
                ).append(step)
                affected_execution_ids.add(
                    str(step.get("trace_step_id") or "")
                )
            for action_id in trace_actions:
                if action_id not in actions:
                    issues.append(
                        f"compiled_trace_missing_action:"
                        f"{local_trace_ref}:{action_id}"
                    )
                    continue
                matching_steps = steps_by_action.get(action_id) or []
                step = matching_steps.pop(0) if matching_steps else {}
                status = (
                    "actual"
                    if action_id in trace_actual
                    else "recommended"
                )
                occurrence_key = (case_id, action_id)
                occurrence_count[occurrence_key] = (
                    occurrence_count.get(occurrence_key, 0) + 1
                )
                attempt_index = int(step.get("attempt_index") or 0)
                if status == "actual" and attempt_index < 1:
                    attempt_index = occurrence_count[occurrence_key]
                evidence_ids = dedupe_strings(
                    step.get("evidence_ids")
                    or actions[action_id].get("evidence_ids")
                    or trace.get("evidence_ids")
                    or []
                )
                occurrence = {
                    "action_id": action_id,
                    "source_case_id": case_id,
                    "execution_status": status,
                    "attempt_index": (
                        attempt_index if status == "actual" else 0
                    ),
                    "evidence_ids": evidence_ids,
                    "case_ref": case_ref,
                    "phase_index": phase_index_by_case.get(case_ref, 0),
                }
                occurrences.append(occurrence)
                seed_steps.append({
                    "trace_step_id": (
                        f"seed:{compiled_trace_id}:"
                        f"{len(occurrences)}"
                    ),
                    "trace_id": compiled_trace_id,
                    "source_case_id": case_id,
                    "action_id": action_id,
                    "ordinal": len(occurrences),
                    "execution_status": status,
                    "attempt_index": occurrence["attempt_index"],
                    "evidence_ids": evidence_ids,
                })

        resolution_messages = dedupe_strings(
            semantic_trace.get("resolution_evidence_message_ids") or []
        )
        resolution_evidence_ids = dedupe_strings(
            evidence_id
            for message_id in resolution_messages
            for evidence_id in evidence_by_message.get(message_id) or []
        )
        if any(
            not evidence_by_message.get(message_id)
            for message_id in resolution_messages
        ):
            issues.append(
                f"resolution_evidence_mapping_incomplete:"
                f"{local_trace_ref}"
            )
        trace_evidence = dedupe_strings([
            *trace_evidence,
            *resolution_evidence_ids,
        ])
        new_trace = {
            "trace_id": compiled_trace_id,
            "family_id": next(
                (value for value in family_ids if value), ""
            ),
            "variant_id": next(
                (value for value in variant_ids if value), ""
            ),
            "source_case_id": root_case_id,
            "source_case_ids": source_case_ids,
            "summary": "；".join(
                dedupe_strings(
                    value.get("summary")
                    for value in phase_values
                )
            )[:160],
            "recommended_action_ids": dedupe_strings(recommended),
            "actual_action_ids": dedupe_strings(actual),
            "evidence_ids": trace_evidence,
            "action_occurrences": occurrences,
            "w7_local_trace_ref": local_trace_ref,
            "w7_source_trace_ids": dedupe_strings(source_trace_ids),
            "w7_case_refs": ordered_case_refs,
            "w7_phase_count": len(phase_values),
            "w7_phases": deepcopy(phase_values),
            "resolution_status": str(
                semantic_trace.get("resolution_status") or "unknown"
            ),
            "resolution_evidence_message_ids": resolution_messages,
            "w7_compiled_bundle_hash": str(
                semantic_bundle.get("compiled_bundle_hash") or ""
            ),
        }
        new_traces.append(new_trace)
        for case_id in source_case_ids:
            new_relations.append({
                "from": case_id,
                "to": compiled_trace_id,
                "relation": "supports",
            })
        for family_id in dedupe_strings(family_ids):
            new_relations.append({
                "from": family_id,
                "to": compiled_trace_id,
                "relation": "has_trace",
            })
        for variant_id in dedupe_strings(variant_ids):
            new_relations.append({
                "from": variant_id,
                "to": compiled_trace_id,
                "relation": "has_trace",
            })
        for action_id in dedupe_strings(recommended):
            new_relations.append({
                "from": compiled_trace_id,
                "to": action_id,
                "relation": "used_action",
            })

    if issues:
        return {}, sorted(set(issues))

    for item in objects.get("ExecutionObservation") or []:
        if (
            isinstance(item, dict)
            and str(item.get("trace_step_id") or "")
            in affected_execution_ids
        ):
            affected_execution_ids.add(
                str(item.get("observation_id") or "")
            )
    for item in objects.get("BranchRule") or []:
        if (
            isinstance(item, dict)
            and str(item.get("trace_id") or "")
            in affected_trace_ids
        ):
            affected_execution_ids.add(
                str(item.get("branch_rule_id") or "")
            )
    objects["DiagnosticTrace"] = [
        item for item in old_traces
        if str(item.get("trace_id") or "") not in affected_trace_ids
    ] + new_traces
    objects["TraceStep"] = [
        item for item in objects.get("TraceStep") or []
        if isinstance(item, dict)
        and str(item.get("trace_id") or "") not in affected_trace_ids
    ] + seed_steps
    objects["ExecutionObservation"] = [
        item for item in objects.get("ExecutionObservation") or []
        if isinstance(item, dict)
        and str(item.get("trace_step_id") or "")
        not in affected_execution_ids
    ]
    objects["BranchRule"] = [
        item for item in objects.get("BranchRule") or []
        if isinstance(item, dict)
        and str(item.get("trace_id") or "") not in affected_trace_ids
    ]
    removed_ids = affected_trace_ids | affected_execution_ids
    relations = [
        relation for relation in relations
        if str(relation.get("from") or "") not in removed_ids
        and str(relation.get("to") or "") not in removed_ids
    ]
    relations.extend(new_relations)
    compiled_objects, compiled_relations, changes = TraceCompiler().compile(
        objects, relations
    )
    candidate = deepcopy(typed_candidate)
    candidate["objects"] = compiled_objects
    candidate["relations"] = compiled_relations
    candidate["w7_compiled_trace_bundle"] = deepcopy(semantic_bundle)
    candidate["w7_correction_compile_result_hash"] = str(
        correction_compile_result.get("compile_result_hash") or ""
    )
    candidate["w7_trace_compiler_changes"] = changes
    candidate["w7_review_only_trace_refs"] = dedupe_strings(
        review_only_trace_refs
    )
    hash_basis = {
        key: value
        for key, value in candidate.items()
        if key not in {"content_hash", "review_id", "review_status"}
    }
    candidate["content_hash"] = (
        "content:w7:" + canonical_hash(hash_basis)
    )
    return candidate, []
