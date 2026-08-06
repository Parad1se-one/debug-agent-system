"""W6 human review queue owner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from debug_agent_system.agents.tools import EvidenceToolAgent
from debug_agent_system.agents.write.w7_trace.review import (
    approval_hash_matches,
    approval_subject_hash,
    build_correction_event,
    correction_chain_subject_hash,
    replay_correction_events,
    trace_review_target_refs,
)
from debug_agent_system.agents.write.w7_trace.correction_compiler import (
    compile_trace_corrections as compile_w7_trace_corrections,
    materialize_corrected_typed_candidate,
)
from debug_agent_system.knowledge.store import KGStore
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS

QUEUE_FILES = {
    "candidates", "merge_candidates", "noise_candidates", "ask_info_candidates",
    "v2_typed_candidates", "atr_weight_proposals",
}
DOCUMENT_FILE_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
APPROVAL_ACTIONS = {"approve", "accept", "merge", "merge_existing", "approve_support_only", "approve_for_execution_policy"}
NON_APPROVAL_STATUS = {
    "reject": "rejected",
    "drop": "rejected",
    "request_more_info": "needs_more_info",
    "needs_owner": "needs_owner",
    "needs_better_evidence": "needs_better_evidence",
}


def _queue_name(name: str) -> str:
    base = str(name or "candidates")
    if base.endswith(".json"):
        base = base[:-5]
    if base not in QUEUE_FILES and base not in {"approved_applied"}:
        base = "candidates"
    return f"{base}.json"


def _logical_queue(name: str) -> str:
    file_name = _queue_name(name)
    return file_name[:-5]


def _item_id(item: dict[str, Any]) -> str:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else item.get("required_info_candidate") if isinstance(item.get("required_info_candidate"), dict) else item
    return str(
        item.get("dedupe_key")
        or item.get("intake_id")
        or item.get("review_id")
        or candidate.get("dedupe_key")
        or candidate.get("intake_id")
        or candidate.get("review_id")
        or candidate.get("candidate_id")
        or candidate.get("id")
        or ""
    )


def _review_content_hash(item: dict[str, Any]) -> str:
    typed = item.get("typed_candidate") if isinstance(item.get("typed_candidate"), dict) else {}
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    return str(item.get("content_hash") or typed.get("content_hash") or candidate.get("content_hash") or "")


def _review_base_hash(item: dict[str, Any]) -> str:
    trace_payload = (
        item.get("trace_review_payload")
        if isinstance(item.get("trace_review_payload"), dict)
        else {}
    )
    return str(
        item.get("trace_bundle_hash")
        or trace_payload.get("trace_bundle_hash")
        or _review_content_hash(item)
        or ""
    )


def _merge_review_state(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    decided = bool(existing.get("review_decision")) or str(existing.get("review_status") or "") not in {"", "pending"}
    if not decided:
        return incoming
    old_hash = _review_base_hash(existing)
    new_hash = _review_base_hash(incoming)
    if old_hash and new_hash and old_hash != new_hash:
        out = dict(incoming)
        out["previous_review_decision"] = {
            "content_hash": old_hash,
            "review_status": existing.get("review_status") or "",
            "selected_action": existing.get("selected_action") or "",
            "review_decision": existing.get("review_decision") or {},
        }
        out["review_status"] = "needs_re_review"
        out["human_approved"] = False
        out.pop("approved_content_hash", None)
        out.pop("selected_action", None)
        out.pop("review_decision", None)
        return out
    out = dict(incoming)
    if str(existing.get("review_item_type") or "") == (
        "w7_trace_bundle_review.v1"
    ):
        for key in (
            "correction_events",
            "correction_event_hashes",
            "correction_overlay",
            "current_approval_subject_hash",
        ):
            if key in existing:
                out[key] = existing[key]
    for key in ("selected_action", "review_status", "human_approved", "review_decision"):
        if key in existing:
            out[key] = existing[key]
    if "approved_content_hash" in existing:
        out["approved_content_hash"] = existing["approved_content_hash"]
    if bool(out.get("approval_hash_required")) and not approval_hash_matches(out):
        out["previous_review_decision"] = {
            "content_hash": approval_subject_hash(existing),
            "review_status": existing.get("review_status") or "",
            "selected_action": existing.get("selected_action") or "",
            "review_decision": existing.get("review_decision") or {},
        }
        out["review_status"] = "needs_re_review"
        out["human_approved"] = False
        out.pop("selected_action", None)
        out.pop("review_decision", None)
        out.pop("approved_content_hash", None)
    return out


def _messages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in (
        "fault_description_messages", "diagnostic_chain_messages", "resolution_messages",
        "case_evidence_messages", "noise_messages", "case_context_messages",
    ):
        for msg in episode.get(key) or []:
            if not isinstance(msg, dict):
                continue
            msg_id = str(msg.get("message_id") or "")
            if msg_id and msg_id in seen:
                continue
            if msg_id:
                seen.add(msg_id)
            text = str(msg.get("text") or "")
            raw_text = str(msg.get("raw_text") or "")
            out.append({
                "message_id": msg_id,
                "sender": msg.get("sender") or {},
                "create_time": msg.get("create_time") or "",
                "content_summary": msg.get("content_summary") or " ".join(text.split())[:240],
                "raw_content_summary": " ".join(raw_text.split())[:500] if raw_text and raw_text != text else "",
                "attachment_metadata": msg.get("attachment_metadata") or [],
                "links": msg.get("links") or [],
                "role": key,
            })
    return out


def _unique_dicts(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        value = str(item.get(key) or "")
        if value and value in seen:
            continue
        if value:
            seen.add(value)
        out.append(item)
    return out


def _dedupe_tool_results(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        signature = ""
        for key in keys:
            value = item
            for part in key.split("."):
                if not isinstance(value, dict):
                    value = {}
                    break
                value = value.get(part)
            if value:
                signature = f"{key}:{value}"
                break
        if not signature and item.get("issue_keys"):
            signature = "issue:" + ",".join(str(x) for x in item.get("issue_keys") or [])
        if not signature:
            signature = repr(sorted((str(k), str(v)) for k, v in item.items() if k in {"type", "name", "path"}))
        if signature in seen:
            continue
        seen.add(signature)
        out.append(item)
    return out


def _merge_tool_evidence(*packs: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "attachment_parse_results": [],
        "document_parse_results": [],
        "dmp_parse_results": [],
        "image_parse_results": [],
        "jira_parse_results": [],
        "proj_parse_results": [],
        "log_package_parse_results": [],
    }
    for pack in packs:
        if not isinstance(pack, dict):
            continue
        for key in merged:
            for item in pack.get(key) or []:
                if isinstance(item, dict):
                    merged[key].append(item)
    merged["attachment_parse_results"] = _dedupe_tool_results(merged["attachment_parse_results"], ("source.file_key", "file_key", "path", "name"))
    merged["document_parse_results"] = _dedupe_tool_results(merged["document_parse_results"], ("source.file_key", "file_key", "path", "name"))
    merged["dmp_parse_results"] = _dedupe_tool_results(merged["dmp_parse_results"], ("source.file_key", "file_key", "path", "name"))
    merged["image_parse_results"] = _dedupe_tool_results(merged["image_parse_results"], ("source.file_key", "file_key", "path", "name"))
    merged["jira_parse_results"] = _dedupe_tool_results(merged["jira_parse_results"], ("source.url", "source.issue_key", "urls.0.url"))
    merged["proj_parse_results"] = _dedupe_tool_results(merged["proj_parse_results"], ("path", "name"))
    merged["log_package_parse_results"] = _dedupe_tool_results(merged["log_package_parse_results"], ("path", "name"))
    return merged


def _tool_evidence(episode: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    artifacts = extracted.get("artifacts") if isinstance(extracted.get("artifacts"), dict) else {}
    existing = _merge_tool_evidence(
        extracted.get("tool_evidence") if isinstance(extracted.get("tool_evidence"), dict) else {},
        artifacts.get("tool_evidence") if isinstance(artifacts.get("tool_evidence"), dict) else {},
    )
    attachments = list(episode.get("attachments") or [])
    attachment_evidence = [x for x in artifacts.get("attachment_evidence") or [] if isinstance(x, dict)]
    attachment_inputs = _unique_dicts([*attachments, *attachment_evidence], "file_key")

    tool_agent = EvidenceToolAgent()
    attachment_results = [tool_agent.parse_attachment(item) for item in attachment_inputs]

    links: list[dict[str, Any]] = []
    links.extend([x for x in extracted.get("jira_links") or [] if isinstance(x, dict)])
    links.extend([x for x in artifacts.get("jira_links") or [] if isinstance(x, dict)])
    for message in messages:
        links.extend([x for x in message.get("links") or [] if isinstance(x, dict)])
    links = _unique_dicts(links, "url")
    jira_results = [tool_agent.parse_jira(item) for item in links]

    document_results: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []
    proj_results: list[dict[str, Any]] = []
    log_package_results: list[dict[str, Any]] = []
    dmp_results: list[dict[str, Any]] = []
    for item in attachment_results:
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        if item.get("evidence_role") == "sample_image":
            image_results.append(tool_agent.parse_image(source or item))
        elif item.get("evidence_role") == "data_file" and Path(str(item.get("name") or item.get("path") or "")).suffix.lower() in DOCUMENT_FILE_EXTS:
            document_results.append(tool_agent.parse_document(source or item))
        elif item.get("evidence_role") == "program_file":
            path = str(item.get("path") or "")
            if path:
                proj_results.append(tool_agent.parse_proj(path))
        elif item.get("evidence_role") == "log_package":
            if Path(str(item.get("name") or item.get("path") or "")).suffix.lower() in {".dmp", ".mdmp"}:
                dmp_results.append(tool_agent.parse_dmp(source or item))
            log_package_results.append(tool_agent.parse_log_package(source or item))
    merged = _merge_tool_evidence(existing, {
        "attachment_parse_results": attachment_results,
        "document_parse_results": document_results,
        "dmp_parse_results": dmp_results,
        "image_parse_results": image_results,
        "jira_parse_results": jira_results,
        "proj_parse_results": proj_results,
        "log_package_parse_results": log_package_results,
    })
    return {
        "attachment_parse_results": merged["attachment_parse_results"],
        "document_parse_results": merged["document_parse_results"],
        "dmp_parse_results": merged["dmp_parse_results"],
        "image_parse_results": merged["image_parse_results"],
        "jira_parse_results": merged["jira_parse_results"],
        "proj_parse_results": merged["proj_parse_results"],
        "log_package_parse_results": merged["log_package_parse_results"],
        "observability": {
            "agent_id": "W6",
            "tool_evidence": {
                "attachments": len(merged["attachment_parse_results"]),
                "documents": len(merged["document_parse_results"]),
                "dmp_files": len(merged["dmp_parse_results"]),
                "images": len(merged["image_parse_results"]),
                "jira_links": len(merged["jira_parse_results"]),
                "proj_files": len(merged["proj_parse_results"]),
                "log_packages": len(merged["log_package_parse_results"]),
            },
        },
    }


def _evidence_pack(episode: dict[str, Any]) -> dict[str, Any]:
    messages = _messages(episode)
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    return {
        "message_ids": list(episode.get("evidence_message_ids") or []),
        "messages": messages,
        "attachments": list(episode.get("attachments") or []),
        "source_offsets": list(episode.get("source_offsets") or []),
        "linked_jira_evidence": list(extracted.get("linked_jira_evidence") or []),
        "tool_evidence": _tool_evidence(episode, messages),
    }




def _node_label(node: dict[str, Any]) -> str:
    return str(node.get("label") or node.get("name") or node.get("content") or node.get("error_id") or node.get("check_id") or node.get("solution_id") or "")


def _candidate_review_summary(candidate: dict[str, Any], episode: dict[str, Any], quality_gate: dict[str, Any], conflict: dict[str, Any], evidence_pack: dict[str, Any]) -> dict[str, Any]:
    nodes = [node for node in candidate.get("nodes") or [] if isinstance(node, dict)]
    errors = [node for node in nodes if node.get("type") == "Error"]
    checks = [node for node in nodes if node.get("type") == "DiagnosticCheck"]
    solutions = [node for node in nodes if node.get("type") == "Solution"]
    outcomes = [node for node in candidate.get("diagnostic_outcomes") or [] if isinstance(node, dict)]
    if not outcomes:
        outcomes = [node for node in nodes if node.get("type") == "DiagnosticOutcome"]
    by_type: dict[str, int] = {}
    fix_evidence_candidates: list[dict[str, Any]] = []
    for outcome in outcomes:
        outcome_type = str(outcome.get("outcome_type") or "unknown")
        by_type[outcome_type] = by_type.get(outcome_type, 0) + 1
        if outcome_type == "verified_fix":
            fix_evidence_candidates.append({
                "outcome_id": outcome.get("outcome_id") or "",
                "action_label": outcome.get("action_label") or "",
                "target_solution_id": outcome.get("target_solution_id") or "",
                "needs_confirmation": bool(outcome.get("needs_confirmation")),
                "evidence_message_ids": outcome.get("evidence_message_ids") or [],
                "observed_duration": outcome.get("observed_duration") or "",
                "root_cause_summary": outcome.get("root_cause_summary") or "",
            })
    risk_flags = []
    if not quality_gate.get("passed"):
        risk_flags.append("quality_gate_failed")
    risk_flags.extend(str(x) for x in quality_gate.get("issues") or [] if str(x))
    risk_flags.extend(str(x) for x in conflict.get("reason_codes") or [] if str(x) in {"high_cost_requires_human", "destructive_requires_human", "non_verified_outcome_not_resolved_by"})
    return {
        "title": _node_label(errors[0]) if errors else str(candidate.get("label") or candidate.get("candidate_id") or ""),
        "queue_hint": "review_merge" if conflict.get("existing_error_id") else "review_new_or_noise",
        "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
        "episode_id": episode.get("episode_id") or "",
        "decision": conflict.get("decision") or "",
        "conflict_type": conflict.get("conflict_type") or "",
        "gate_passed": bool(quality_gate.get("passed")),
        "risk_flags": sorted(set(risk_flags)),
        "node_counts": {"errors": len(errors), "checks": len(checks), "solutions": len(solutions), "outcomes": len(outcomes)},
        "outcome_type_counts": by_type,
        "check_preview": [_node_label(x) for x in checks[:5]],
        "solution_preview": [_node_label(x) for x in solutions[:5]],
        "inferred_conclusion": candidate.get("conclusion") or "",
        "fix_evidence_candidates": fix_evidence_candidates[:5],
        "recommended_reviewer_actions": ["approve"] if fix_evidence_candidates else [],
        "evidence_counts": {
            "messages": len(evidence_pack.get("messages") or []),
            "attachments": len(evidence_pack.get("attachments") or []),
            "source_offsets": len(evidence_pack.get("source_offsets") or []),
        },
    }


def _ask_info_review_summary(required_info_candidate: dict[str, Any], episode: dict[str, Any], quality_gate: dict[str, Any], conflict: dict[str, Any], evidence_pack: dict[str, Any]) -> dict[str, Any]:
    risk_flags = []
    if not quality_gate.get("passed"):
        risk_flags.append("quality_gate_failed")
    risk_flags.extend(str(x) for x in quality_gate.get("issues") or [] if str(x))
    return {
        "title": required_info_candidate.get("label") or required_info_candidate.get("slot") or required_info_candidate.get("candidate_id") or "",
        "queue_hint": "review_required_info",
        "candidate_id": required_info_candidate.get("candidate_id") or "",
        "episode_id": episode.get("episode_id") or "",
        "target_error_id": required_info_candidate.get("target_error_id") or "",
        "slot": required_info_candidate.get("slot") or "",
        "question": required_info_candidate.get("question") or "",
        "why_required": required_info_candidate.get("why_required") or "",
        "condition": required_info_candidate.get("condition") or "",
        "provided_later": bool(required_info_candidate.get("provided_later")),
        "decision": conflict.get("decision") or "",
        "gate_passed": bool(quality_gate.get("passed")),
        "risk_flags": sorted(set(risk_flags)),
        "evidence_counts": {
            "messages": len(evidence_pack.get("messages") or []),
            "attachments": len(evidence_pack.get("attachments") or []),
            "source_offsets": len(evidence_pack.get("source_offsets") or []),
        },
    }

def _policy_preview(candidate: dict[str, Any]) -> dict[str, Any]:
    outcomes = [x for x in candidate.get("diagnostic_outcomes") or [] if isinstance(x, dict)]
    by_type: dict[str, int] = {}
    high_cost: list[dict[str, Any]] = []
    for outcome in outcomes:
        outcome_type = str(outcome.get("outcome_type") or "unknown")
        by_type[outcome_type] = by_type.get(outcome_type, 0) + 1
        if outcome.get("high_cost") or outcome.get("destructive"):
            high_cost.append({
                "outcome_id": outcome.get("outcome_id"),
                "action_label": outcome.get("action_label"),
                "outcome_type": outcome_type,
                "high_cost": bool(outcome.get("high_cost")),
                "destructive": bool(outcome.get("destructive")),
            })
    trace = candidate.get("diagnostic_trace") if isinstance(candidate.get("diagnostic_trace"), dict) else {}
    return {
        "source_trace_id": trace.get("trace_id") or "",
        "outcome_count": len(outcomes),
        "by_outcome_type": by_type,
        "unsafe_or_high_cost_actions": high_cost,
        "resolved_by_allowed_only_for_verified_fix": True,
    }


def _typed_raw_excerpt(envelope: dict[str, Any]) -> dict[str, Any]:
    source = _typed_source(envelope)
    raw_text = str(_typed_first(envelope, "raw_text", "original_text", "text") or source.get("raw_text") or source.get("text") or "")
    compact_raw = " ".join(raw_text.split())
    evidence_text = " ".join(
        str(item.get("summary") or item.get("content_summary") or item.get("text") or "").strip()
        for item in _typed_evidence_items(envelope)
        if isinstance(item, dict)
    ).strip()
    # Long chat-segment context may contain many unrelated issues.  The review
    # header must lead with the message-level evidence actually attached to the
    # candidate; keep the broader segment only as navigation context.
    focused_text = evidence_text if len(compact_raw) > 1000 and evidence_text else compact_raw
    return {
        "text": focused_text[:1000],
        "navigation_context_text": compact_raw[:1000] if focused_text != compact_raw else "",
        "source_ref": _typed_first(envelope, "source_ref") or source.get("source_ref") or source.get("path") or source.get("url") or "",
        "source_kind": _typed_first(envelope, "source_kind", "source_type") or source.get("source_kind") or source.get("source_type") or "",
    }


def _typed_containers(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [envelope]
    for key in ("payload", "evidence_pack"):
        value = envelope.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _typed_first(envelope: dict[str, Any], *keys: str) -> Any:
    for source in _typed_containers(envelope):
        for key in keys:
            value = source.get(key)
            if value not in (None, "", [], {}):
                return value
    return ""


def _typed_list(envelope: dict[str, Any], *keys: str) -> list[Any]:
    out: list[Any] = []
    for source in _typed_containers(envelope):
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                out.extend(value)
            elif value not in (None, "", [], {}):
                out.append(value)
    return out


def _typed_source(envelope: dict[str, Any]) -> dict[str, Any]:
    for source_container in _typed_containers(envelope):
        source = source_container.get("source")
        if isinstance(source, dict):
            return source
    return {}


def _typed_objects(envelope: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    objects: dict[str, list[dict[str, Any]]] = {}
    for source in _typed_containers(envelope):
        raw_objects = source.get("objects") if isinstance(source.get("objects"), dict) else {}
        for object_type, items in raw_objects.items():
            if isinstance(items, list):
                objects.setdefault(str(object_type), []).extend(item for item in items if isinstance(item, dict))
    return objects


def _typed_object_diff(envelope: dict[str, Any]) -> dict[str, Any]:
    objects = _typed_objects(envelope)
    return {
        "object_counts": {object_type: len(items) for object_type, items in sorted(objects.items())},
        "object_previews": {
            object_type: [
                {
                    "id": item.get(V2_PRIMARY_KEYS.get(object_type, "id")) or item.get("id") or item.get("object_id") or "",
                    "label": item.get("label") or item.get("title") or item.get("summary") or "",
                }
                for item in items[:5]
            ]
            for object_type, items in sorted(objects.items())
        },
        "relation_count": len([item for item in _typed_list(envelope, "relations") if isinstance(item, dict)]),
    }


def _typed_evidence_items(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = _typed_first(envelope, "evidence")
    if isinstance(evidence, list):
        return [item for item in evidence if isinstance(item, dict)]
    evidence_items = _typed_first(envelope, "evidence_items")
    if isinstance(evidence_items, list):
        return [item for item in evidence_items if isinstance(item, dict)]
    return list(_typed_objects(envelope).get("EvidenceItem") or [])


def _typed_outcome_evidence(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    objects = _typed_objects(envelope)
    outcomes = objects.get("ActionOutcome") or []
    evidence = _typed_first(envelope, "outcome_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    out: list[dict[str, Any]] = []
    for item in outcomes[:10]:
        out.append({
            "outcome_id": item.get("outcome_id") or item.get("id") or "",
            "outcome_type": item.get("outcome_type") or "",
            "summary": item.get("summary") or "",
            "evidence_ids": item.get("evidence_ids") or item.get("evidence_message_ids") or [],
        })
    for item in evidence[:10]:
        if isinstance(item, dict):
            out.append(item)
    return out


def _typed_kg_alignment(envelope: dict[str, Any], quality_gate: dict[str, Any]) -> dict[str, Any]:
    objects = _typed_objects(envelope)
    return {
        "admission_target": quality_gate.get("admission_target") or _typed_first(envelope, "admission_target") or "",
        "mapping_version": quality_gate.get("mapping_version") or "",
        "object_types": sorted(object_type for object_type, items in objects.items() if items),
        "legacy_fault_variant_forced": False,
        "candidate_family_ids": [
            str(item.get("family_id") or "")
            for item in objects.get("FaultFamily") or []
            if str(item.get("family_id") or "")
        ],
        "candidate_variant_ids": [
            str(item.get("variant_id") or "")
            for item in objects.get("FaultVariant") or []
            if str(item.get("variant_id") or "")
        ],
    }


def _typed_review_context(envelope: dict[str, Any]) -> dict[str, Any]:
    """Separate current evidence from W7/KG/gold references for human review."""
    metadata = _typed_first(envelope, "metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    review_context = metadata.get("review_context") if isinstance(metadata.get("review_context"), dict) else {}
    if not review_context:
        review_context = _typed_first(envelope, "review_context", "alignment_context")
        review_context = review_context if isinstance(review_context, dict) else {}
    payload = _typed_first(envelope, "episode")
    episode = payload if isinstance(payload, dict) else {}
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    attribution = extracted.get("attribution") if isinstance(extracted.get("attribution"), dict) else {}
    return {
        "schema_version": "kg_v2.review_source_separation.v1",
        "current_case_evidence": {
            "source_type": _typed_first(envelope, "source_type") or "",
            "source_ref": _typed_first(envelope, "source_ref") or "",
            "message_ids": _typed_first(envelope, "message_ids") or _typed_first(envelope, "evidence_message_ids") or [],
        },
        "w7_attribution": attribution,
        "w7_trace_context": {
            "trace_group_id": str(episode.get("trace_group_id") or ""),
            "phase_index": int(episode.get("trace_phase_index") or 0),
            "phase_count": int(episode.get("trace_phase_count") or 0),
            "relation_type": str(episode.get("trace_relation_type") or ""),
            "previous_episode_id": str(episode.get("previous_trace_episode_id") or ""),
            "link_strength": str(episode.get("trace_link_strength") or ""),
            "link_reasons": list(episode.get("trace_link_reasons") or []),
            "link_candidates": list(episode.get("trace_link_candidates") or []),
            "evidence_sharing_allowed": False,
            "outcome_sharing_allowed": False,
        },
        "kg_alignment_only": {
            "context_role": review_context.get("context_role") or "alignment_only",
            "baseline_graph_hash": review_context.get("baseline_graph_hash") or "",
            "recalled_background": review_context.get("recalled_background") or review_context.get("top_family_background") or [],
            "facts_may_not_be_copied_as_new_evidence": bool(review_context.get("facts_may_not_be_copied_as_new_evidence", True)),
        },
        "reference_only_examples": [
            {
                "case_id": item.get("case_id") or "",
                "selection_reason": item.get("selection_reason") or "",
                "exact_source_match": bool(item.get("exact_source_match")),
                "graph_ingestion": False,
            }
            for item in review_context.get("reviewed_case_examples") or []
            if isinstance(item, dict)
        ],
    }


class ReviewQueueAgent:
    """W6: writes idempotent human review items; never mutates main KG."""

    def __init__(self, store: KGStore, queue_dir: str | Path | None = None) -> None:
        self.store = store
        self.queue_dir = Path(queue_dir) if queue_dir is not None else None

    def build_review_item(
        self,
        queue: str,
        candidate: dict[str, Any],
        episode: dict[str, Any],
        conflict: dict[str, Any],
        quality_gate: dict[str, Any],
        dry_run_merge_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        logical = _logical_queue(queue)
        candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "")
        review_id = f"review:{candidate_id.replace('chatcand:', '')}" if candidate_id else "review:unknown"
        evidence_pack = _evidence_pack(episode)
        return {
            "review_id": review_id,
            "candidate_id": candidate_id,
            "queue": logical,
            "candidate": candidate,
            "case_variant_candidate": candidate.get("case_variant_candidate") or {},
            "diagnostic_trace": candidate.get("diagnostic_trace") or {},
            "diagnostic_outcomes": candidate.get("diagnostic_outcomes") or [],
            "policy_preview": _policy_preview(candidate),
            "episode": episode,
            "conflict": conflict,
            "quality_gate": quality_gate,
            "dry_run_merge_plan": dry_run_merge_plan or {},
            "evidence_pack": evidence_pack,
            "review_summary": _candidate_review_summary(candidate, episode, quality_gate, conflict, evidence_pack),
            "review_actions": ["approve", "reject", "merge_existing", "request_more_info"],
            "review_status": "pending",
            "observability": {"agent_id": "W6", "queue": logical, "candidate_id": candidate_id},
        }

    def build_ask_info_review_item(
        self,
        required_info_candidate: dict[str, Any],
        episode: dict[str, Any],
        quality_gate: dict[str, Any],
        conflict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_id = str(required_info_candidate.get("candidate_id") or "")
        review_id = f"review:{candidate_id.replace('reqinfo:', '')}" if candidate_id else "review:required-info-unknown"
        evidence_pack = _evidence_pack(episode)
        return {
            "review_id": review_id,
            "candidate_id": candidate_id,
            "queue": "ask_info_candidates",
            "required_info_candidate": required_info_candidate,
            "episode": episode,
            "conflict": conflict or {},
            "quality_gate": quality_gate,
            "evidence_pack": evidence_pack,
            "review_summary": _ask_info_review_summary(required_info_candidate, episode, quality_gate, conflict or {}, evidence_pack),
            "review_actions": ["accept", "merge", "drop", "needs_owner", "needs_better_evidence"],
            "review_status": "pending",
            "observability": {"agent_id": "W6", "queue": "ask_info_candidates", "candidate_id": candidate_id},
        }

    def build_typed_review_item(
        self,
        envelope: dict[str, Any],
        quality_gate: dict[str, Any],
        *,
        dry_run_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dedupe_key = str(_typed_first(envelope, "dedupe_key") or "").strip()
        intake_id = str(_typed_first(envelope, "intake_id", "candidate_id", "bundle_id") or "").strip()
        stable_id = dedupe_key or intake_id or "unknown"
        review_id = str(_typed_first(envelope, "review_id") or f"review:typed:{stable_id}")
        materialize_allowed = bool(quality_gate.get("materialize_allowed"))
        # Older callers may provide a valid typed candidate with a legacy W4
        # payload that predates admission readiness.  Do not turn an absent
        # field into an explicit ``not_ready`` decision here: W5 can infer the
        # readiness from the candidate graph, while an explicit value must be
        # preserved as the authoritative gate result.
        admission_readiness = str(quality_gate.get("admission_readiness") or "")
        item = {
            "review_id": review_id,
            "queue": "v2_typed_candidates",
            "dedupe_key": dedupe_key,
            "intake_id": intake_id,
            "content_hash": str(_typed_first(envelope, "content_hash") or ""),
            "candidate_id": str(_typed_first(envelope, "candidate_id") or ""),
            "typed_candidate": envelope,
            "raw_evidence": _typed_raw_excerpt(envelope),
            "evidence": _typed_evidence_items(envelope),
            "kg_alignment": _typed_kg_alignment(envelope, quality_gate),
            "source_separation": _typed_review_context(envelope),
            "object_diff": _typed_object_diff(envelope),
            "outcome_evidence": _typed_outcome_evidence(envelope),
            "quality_gate": quality_gate,
            "admission_readiness": admission_readiness,
            "merge_allowed": bool(quality_gate.get("merge_allowed")),
            "materialize_allowed": materialize_allowed,
            "dry_run_plan": dry_run_plan or {
                "status": "not_run",
                "materialize_allowed": materialize_allowed,
                "admission_target": quality_gate.get("admission_target") or _typed_first(envelope, "admission_target") or "",
                "admission_readiness": admission_readiness,
            },
            # ``approve`` remains the compatibility action used by the CLI and
            # existing review fixtures.  The two explicit actions let the UI
            # distinguish support-only admission from execution-policy impact.
            "review_actions": ["approve", "approve_support_only", "approve_for_execution_policy", "reject", "request_more_info"],
            "review_status": "pending",
            "observability": {
                "agent_id": "W6",
                "queue": "v2_typed_candidates",
                "dedupe_key": dedupe_key,
                "intake_id": intake_id,
                "admission_readiness": admission_readiness,
            },
        }
        if admission_readiness:
            item["admission_readiness"] = admission_readiness
        else:
            item.pop("admission_readiness", None)
        return item

    def build_w7_trace_review_item(
        self,
        envelope: dict[str, Any],
        quality_gate: dict[str, Any],
        *,
        trace_review_payload: dict[str, Any],
        dry_run_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a hash-bound W6 item for compiled W7 trace semantics."""

        item = self.build_typed_review_item(
            envelope,
            quality_gate,
            dry_run_plan=dry_run_plan,
        )
        trace_payload = (
            dict(trace_review_payload)
            if isinstance(trace_review_payload, dict)
            else {}
        )
        trace_bundle_hash = str(
            trace_payload.get("trace_bundle_hash") or ""
        )
        item.update({
            "review_item_type": "w7_trace_bundle_review.v1",
            "trace_review_payload": trace_payload,
            "trace_bundle_hash": trace_bundle_hash,
            "approval_hash_required": True,
            "correction_events": [],
            "correction_event_hashes": [],
        })
        item["current_approval_subject_hash"] = approval_subject_hash(item)
        item["observability"] = {
            **item.get("observability", {}),
            "trace_bundle_hash": trace_bundle_hash,
            "approval_hash_required": True,
        }
        return item

    def enqueue(self, name: str, item: dict[str, Any]) -> dict[str, Any]:
        file_name = _queue_name(name)
        data = self._read(file_name)
        item_id = _item_id(item)
        if item_id:
            for idx, existing in enumerate(data):
                if _item_id(existing) == item_id:
                    data[idx] = _merge_review_state(existing, item)
                    self._write(file_name, data)
                    return {"status": "updated", "queue": file_name, "size": len(data), "review_id": item_id, "candidate_id": item.get("candidate_id")}
        data.append(item)
        self._write(file_name, data)
        return {"status": "queued", "queue": file_name, "size": len(data), "review_id": item_id, "candidate_id": item.get("candidate_id")}

    def enqueue_many(self, name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.enqueue_batches(name, [items])

    def enqueue_batches(self, name: str, batches: Iterable[list[dict[str, Any]]]) -> dict[str, Any]:
        """Upsert multiple source batches with one queue read/write cycle.

        Full non-SOP ingestion emits independently resumable W2/W3 shards.
        Re-reading a multi-gigabyte canonical JSON queue for every shard is
        quadratic; this boundary keeps the existing queue contract while doing
        one canonical merge at the end of a run.
        """

        file_name = _queue_name(name)
        data = self._read(file_name)
        index = {_item_id(existing): idx for idx, existing in enumerate(data) if _item_id(existing)}
        queued = 0
        updated = 0
        batch_count = 0
        for items in batches:
            batch_count += 1
            for item in items:
                item_id = _item_id(item)
                if item_id and item_id in index:
                    existing = data[index[item_id]]
                    data[index[item_id]] = _merge_review_state(existing, item)
                    updated += 1
                else:
                    if item_id:
                        index[item_id] = len(data)
                    data.append(item)
                    queued += 1
        self._write(file_name, data)
        return {
            "status": "batch_written",
            "queue": file_name,
            "size": len(data),
            "queued": queued,
            "updated": updated,
            "batch_count": batch_count,
        }

    def read_queue(self, name: str) -> list[dict[str, Any]]:
        """Read a review queue through the same queue_dir boundary used by W6 writes."""

        return self._read(_queue_name(name))

    def mark_decision(
        self,
        name: str,
        item_id: str,
        action: str,
        *,
        reviewer: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        """Record a human decision on one review item without applying it.

        W6 owns review state only.  Approval here makes a later W5 approved-only
        apply eligible, but this method never mutates KG instances/edges.
        """

        file_name = _queue_name(name)
        data = self._read(file_name)
        wanted = str(item_id or "").strip()
        selected = str(action or "").strip()
        if not wanted:
            return {"status": "not_found", "reason": "missing_item_id", "queue": file_name}
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            current_id = _item_id(item)
            candidate_id = str(item.get("candidate_id") or "")
            nested = item.get("candidate") if isinstance(item.get("candidate"), dict) else item.get("required_info_candidate") if isinstance(item.get("required_info_candidate"), dict) else {}
            nested_id = str(nested.get("candidate_id") or nested.get("id") or "") if isinstance(nested, dict) else ""
            aliases = {
                current_id,
                candidate_id,
                nested_id,
                str(item.get("review_id") or ""),
                str(item.get("dedupe_key") or ""),
                str(item.get("intake_id") or ""),
            }
            if isinstance(nested, dict):
                aliases.update({
                    str(nested.get("review_id") or ""),
                    str(nested.get("dedupe_key") or ""),
                    str(nested.get("intake_id") or ""),
                })
            if wanted not in aliases:
                continue
            allowed = [str(x) for x in item.get("review_actions") or []]
            if selected not in allowed:
                return {
                    "status": "invalid_action",
                    "queue": file_name,
                    "review_id": current_id,
                    "candidate_id": candidate_id or nested_id,
                    "action": selected,
                    "allowed_actions": allowed,
                }
            approved = selected in APPROVAL_ACTIONS
            updated = dict(item)
            updated["selected_action"] = selected
            updated["review_status"] = "approved" if approved else NON_APPROVAL_STATUS.get(selected, selected or "pending")
            updated["human_approved"] = approved
            updated["review_decision"] = {
                "action": selected,
                "reviewer": reviewer,
                "note": note,
                "decided_at": datetime.now(timezone.utc).isoformat(),
            }
            if approved:
                approved_hash = approval_subject_hash(updated)
                if bool(updated.get("approval_hash_required")) and not approved_hash:
                    return {
                        "status": "invalid_approval_subject",
                        "queue": file_name,
                        "review_id": current_id,
                        "candidate_id": candidate_id or nested_id,
                    }
                updated["approved_content_hash"] = approved_hash
                updated["review_decision"][
                    "approved_content_hash"
                ] = approved_hash
            else:
                updated.pop("approved_content_hash", None)
            data[idx] = updated
            self._write(file_name, data)
            return {
                "status": "decision_recorded",
                "queue": file_name,
                "review_id": current_id,
                "candidate_id": candidate_id or nested_id,
                "selected_action": selected,
                "review_status": updated["review_status"],
                "human_approved": approved,
                "approved_content_hash": (
                    updated.get("approved_content_hash") or ""
                ),
            }
        return {"status": "not_found", "queue": file_name, "item_id": wanted}

    def append_trace_correction(
        self,
        name: str,
        item_id: str,
        operation: str,
        *,
        target_ref: str,
        payload: dict[str, Any] | None = None,
        evidence_message_ids: list[str] | None = None,
        reviewer: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        """Append one immutable correction event and invalidate approval."""

        file_name = _queue_name(name)
        data = self._read(file_name)
        wanted = str(item_id or "").strip()
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            aliases = {
                _item_id(item),
                str(item.get("review_id") or ""),
                str(item.get("candidate_id") or ""),
                str(item.get("dedupe_key") or ""),
                str(item.get("intake_id") or ""),
            }
            if wanted not in aliases:
                continue
            if str(item.get("review_item_type") or "") != (
                "w7_trace_bundle_review.v1"
            ):
                return {
                    "status": "invalid_review_item_type",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            trace_payload = (
                item.get("trace_review_payload")
                if isinstance(item.get("trace_review_payload"), dict)
                else {}
            )
            correction_base_payload = (
                item.get("correction_base_trace_review_payload")
                if isinstance(
                    item.get("correction_base_trace_review_payload"), dict
                )
                else trace_payload
            )
            events = [
                value for value in item.get("correction_events") or []
                if isinstance(value, dict)
            ]
            allowed_message_ids = set(
                trace_payload.get("allowed_message_ids") or []
            )
            current_trace_payload = (
                item.get("corrected_trace_review_payload")
                if isinstance(
                    item.get("corrected_trace_review_payload"), dict
                )
                else trace_payload
            )
            allowed_target_refs = trace_review_target_refs(
                current_trace_payload
            )
            event, event_issues = build_correction_event(
                review_id=str(item.get("review_id") or _item_id(item)),
                operation=operation,
                target_ref=target_ref,
                payload=payload or {},
                evidence_message_ids=evidence_message_ids or [],
                reviewer=reviewer,
                note=note,
                sequence=len(events) + 1,
                base_subject_hash=correction_chain_subject_hash(
                    correction_base_payload, events
                ),
                allowed_target_refs=(
                    allowed_target_refs if allowed_target_refs else None
                ),
                allowed_message_ids=(
                    allowed_message_ids if allowed_message_ids else None
                ),
            )
            if event_issues:
                return {
                    "status": "invalid_correction",
                    "queue": file_name,
                    "review_id": _item_id(item),
                    "issues": event_issues,
                }
            updated = dict(item)
            updated["correction_events"] = [*events, event]
            replay, replay_issues = replay_correction_events(
                correction_base_payload,
                updated["correction_events"],
            )
            if replay_issues:
                return {
                    "status": "invalid_correction_replay",
                    "queue": file_name,
                    "review_id": _item_id(item),
                    "issues": replay_issues,
                }
            if item.get("review_decision") or item.get("review_status"):
                updated["previous_review_decision"] = {
                    "content_hash": approval_subject_hash(item),
                    "review_status": item.get("review_status") or "",
                    "selected_action": item.get("selected_action") or "",
                    "review_decision": item.get("review_decision") or {},
                }
            updated["correction_overlay"] = replay
            updated["correction_base_trace_review_payload"] = (
                correction_base_payload
            )
            updated.pop("corrected_trace_review_payload", None)
            updated.pop("correction_compile_result", None)
            updated.pop("correction_overlay_applied", None)
            updated.pop("applied_correction_overlay_hash", None)
            updated["correction_event_hashes"] = list(
                replay.get("correction_event_hashes") or []
            )
            updated["review_status"] = "needs_re_review"
            updated["human_approved"] = False
            updated.pop("selected_action", None)
            updated.pop("review_decision", None)
            updated.pop("approved_content_hash", None)
            updated["current_approval_subject_hash"] = (
                approval_subject_hash(updated)
            )
            data[idx] = updated
            self._write(file_name, data)
            return {
                "status": "correction_recorded",
                "queue": file_name,
                "review_id": _item_id(updated),
                "event_id": event["event_id"],
                "event_hash": event["event_hash"],
                "review_status": "needs_re_review",
                "current_approval_subject_hash": updated[
                    "current_approval_subject_hash"
                ],
            }
        return {"status": "not_found", "queue": file_name, "item_id": wanted}

    def compile_trace_corrections(
        self,
        name: str,
        item_id: str,
        *,
        quality_gate_scorer: (
            Callable[[dict[str, Any]], dict[str, Any]] | None
        ) = None,
        reextract_callback: (
            Callable[[dict[str, Any]], dict[str, Any]] | None
        ) = None,
    ) -> dict[str, Any]:
        """Compile the immutable event chain into a fresh typed candidate.

        A successful compile always invalidates the previous approval.  The
        supplied W4 scorer is required before the result can be approved and
        consumed by W5.
        """

        file_name = _queue_name(name)
        data = self._read(file_name)
        wanted = str(item_id or "").strip()
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            aliases = {
                _item_id(item),
                str(item.get("review_id") or ""),
                str(item.get("candidate_id") or ""),
                str(item.get("dedupe_key") or ""),
                str(item.get("intake_id") or ""),
            }
            if wanted not in aliases:
                continue
            if str(item.get("review_item_type") or "") != (
                "w7_trace_bundle_review.v1"
            ):
                return {
                    "status": "invalid_review_item_type",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            events = [
                value for value in item.get("correction_events") or []
                if isinstance(value, dict)
            ]
            if not events:
                return {
                    "status": "no_correction_events",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            base_payload = (
                item.get("correction_base_trace_review_payload")
                if isinstance(
                    item.get("correction_base_trace_review_payload"), dict
                )
                else (
                    item.get("trace_review_payload")
                    if isinstance(
                        item.get("trace_review_payload"), dict
                    )
                    else {}
                )
            )
            compile_result, compile_issues = (
                compile_w7_trace_corrections(
                    trace_review_payload=base_payload,
                    correction_events=events,
                )
            )
            if compile_issues:
                return {
                    "status": "correction_compile_failed",
                    "queue": file_name,
                    "review_id": _item_id(item),
                    "issues": compile_issues,
                    "requires_w2_reextract": bool(
                        compile_result.get("requires_w2_reextract")
                    ),
                    "compile_result": compile_result,
                }
            typed_candidate = (
                item.get("typed_candidate")
                if isinstance(item.get("typed_candidate"), dict)
                else {}
            )
            corrected_candidate, materialize_issues = (
                materialize_corrected_typed_candidate(
                    typed_candidate=typed_candidate,
                    correction_compile_result=compile_result,
                )
            )
            if materialize_issues:
                # Structural case edits cannot be projected onto the old
                # SourceCase/Action objects.  Persist an explicit, immutable
                # re-extraction request so a worker can run W2 on the
                # corrected atomic-case manifest and later fulfil it through
                # ``fulfill_w2_reextract``.  Returning only a transient error
                # made the required follow-up easy to lose across restarts.
                if compile_result.get("requires_w2_reextract"):
                    request = {
                        "schema_version": "w7.w2_reextract_request.v1",
                        "request_id": (
                            "w7-w2-reextract:"
                            f"{str(item.get('review_id') or _item_id(item))}:"
                            f"{str(compile_result.get('compile_result_hash') or '')[:20]}"
                        ),
                        "review_id": str(
                            item.get("review_id") or _item_id(item)
                        ),
                        "base_source_ledger_hash": str(
                            base_payload.get("source_ledger_hash") or ""
                        ),
                        "correction_event_hashes": list(
                            item.get("correction_event_hashes") or []
                        ),
                        "operations": list(
                            compile_result.get("w2_reextract_operations")
                            or []
                        ),
                        "case_cards": deepcopy(
                            compile_result.get("case_cards") or []
                        ),
                        "trace_review_payload": deepcopy(
                            compile_result.get(
                                "corrected_trace_review_payload"
                            ) or {}
                        ),
                        "compile_result_hash": str(
                            compile_result.get("compile_result_hash") or ""
                        ),
                        "status": "pending",
                    }
                    updated = dict(item)
                    updated["reextract_request"] = request
                    updated["review_status"] = "needs_w2_reextract"
                    updated["human_approved"] = False
                    updated.pop("selected_action", None)
                    updated.pop("review_decision", None)
                    updated.pop("approved_content_hash", None)
                    updated["correction_compile_result"] = compile_result
                    updated["current_approval_subject_hash"] = (
                        approval_subject_hash(updated)
                    )
                    data[idx] = updated
                    self._write(file_name, data)
                    response = {
                        "status": "requires_w2_reextract",
                        "queue": file_name,
                        "review_id": _item_id(updated),
                        "issues": materialize_issues,
                        "reextract_request": request,
                        "compile_result": compile_result,
                    }
                    # The callback is an explicit worker boundary.  It may
                    # perform W2/W3/W4 outside W6 and return a fresh typed
                    # candidate plus gate/payload; absent a callback the
                    # persisted request remains the durable work item.
                    if reextract_callback is not None:
                        try:
                            reextract_result = reextract_callback(
                                deepcopy(request)
                            )
                        except Exception as exc:
                            response["reextract_worker_error"] = (
                                f"{type(exc).__name__}:{exc}"
                            )
                        else:
                            response["reextract_result"] = reextract_result
                            if isinstance(reextract_result, dict):
                                candidate = reextract_result.get(
                                    "typed_candidate"
                                )
                                gate = reextract_result.get(
                                    "quality_gate"
                                )
                                if (
                                    isinstance(candidate, dict)
                                    and isinstance(gate, dict)
                                ):
                                    response["fulfillment"] = (
                                        self.fulfill_w2_reextract(
                                            name,
                                            item_id,
                                            typed_candidate=candidate,
                                            quality_gate=gate,
                                            trace_review_payload=(
                                                reextract_result.get(
                                                    "trace_review_payload"
                                                )
                                                if isinstance(
                                                    reextract_result.get(
                                                        "trace_review_payload"
                                                    ),
                                                    dict,
                                                )
                                                else None
                                            ),
                                            reextract_request_id=str(
                                                request.get("request_id") or ""
                                            ),
                                        )
                                    )
                    return response
                return {
                    "status": (
                        "requires_w2_reextract"
                        if compile_result.get("requires_w2_reextract")
                        else "typed_materialization_failed"
                    ),
                    "queue": file_name,
                    "review_id": _item_id(item),
                    "issues": materialize_issues,
                    "compile_result": compile_result,
                }
            if quality_gate_scorer is None:
                return {
                    "status": "compiled_needs_w4",
                    "queue": file_name,
                    "review_id": _item_id(item),
                    "corrected_candidate": corrected_candidate,
                    "compile_result": compile_result,
                }
            quality_gate = quality_gate_scorer(corrected_candidate)
            if not isinstance(quality_gate, dict):
                return {
                    "status": "w4_revalidation_invalid",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            corrected_candidate["quality_gate"] = quality_gate
            corrected_payload = dict(
                compile_result["corrected_trace_review_payload"]
            )
            overlay = dict(
                compile_result.get("correction_overlay") or {}
            )
            updated = dict(item)
            updated.update({
                "typed_candidate": corrected_candidate,
                "content_hash": str(
                    corrected_candidate.get("content_hash") or ""
                ),
                "corrected_trace_review_payload": corrected_payload,
                "corrected_trace_bundle_hash": str(
                    corrected_payload.get("trace_bundle_hash") or ""
                ),
                "correction_compile_result": compile_result,
                "correction_overlay": overlay,
                "correction_overlay_applied": True,
                "applied_correction_overlay_hash": str(
                    overlay.get("effective_bundle_hash") or ""
                ),
                "correction_revalidation_required": False,
                "quality_gate": quality_gate,
                "admission_readiness": str(
                    quality_gate.get("admission_readiness") or ""
                ),
                "merge_allowed": bool(
                    quality_gate.get("merge_allowed")
                ),
                "materialize_allowed": bool(
                    quality_gate.get("materialize_allowed")
                ),
                "review_status": "pending",
                "human_approved": False,
            })
            updated.pop("selected_action", None)
            updated.pop("review_decision", None)
            updated.pop("approved_content_hash", None)
            updated["current_approval_subject_hash"] = (
                approval_subject_hash(updated)
            )
            updated["observability"] = {
                **(
                    updated.get("observability")
                    if isinstance(updated.get("observability"), dict)
                    else {}
                ),
                "corrected_trace_bundle_hash": updated[
                    "corrected_trace_bundle_hash"
                ],
                "correction_compile_result_hash": str(
                    compile_result.get("compile_result_hash") or ""
                ),
                "w4_revalidated": True,
            }
            data[idx] = updated
            self._write(file_name, data)
            return {
                "status": "corrections_compiled",
                "queue": file_name,
                "review_id": _item_id(updated),
                "content_hash": updated["content_hash"],
                "corrected_trace_bundle_hash": updated[
                    "corrected_trace_bundle_hash"
                ],
                "current_approval_subject_hash": updated[
                    "current_approval_subject_hash"
                ],
                "w4_decision": str(
                    quality_gate.get("decision") or ""
                ),
            }
        return {
            "status": "not_found",
            "queue": file_name,
            "item_id": wanted,
        }

    def fulfill_w2_reextract(
        self,
        name: str,
        item_id: str,
        *,
        typed_candidate: dict[str, Any],
        quality_gate: dict[str, Any],
        trace_review_payload: dict[str, Any] | None = None,
        reextract_request_id: str = "",
    ) -> dict[str, Any]:
        """Attach a fresh W2 result to a structural correction request.

        W6 does not perform extraction itself.  A caller (normally the W7
        correction worker) supplies the newly extracted, W3-normalized typed
        candidate.  The request identity is checked before replacing the
        candidate, approvals are cleared, and the item returns to ``pending``
        so a human must review the new subject again.
        """

        file_name = _queue_name(name)
        data = self._read(file_name)
        wanted = str(item_id or "").strip()
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            aliases = {
                _item_id(item),
                str(item.get("review_id") or ""),
                str(item.get("candidate_id") or ""),
            }
            if wanted not in aliases:
                continue
            request = (
                item.get("reextract_request")
                if isinstance(item.get("reextract_request"), dict)
                else {}
            )
            if not request:
                return {
                    "status": "no_reextract_request",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            expected_request_id = str(request.get("request_id") or "")
            if reextract_request_id and str(reextract_request_id) != expected_request_id:
                return {
                    "status": "reextract_request_mismatch",
                    "queue": file_name,
                    "review_id": _item_id(item),
                    "expected_request_id": expected_request_id,
                }
            if not isinstance(typed_candidate, dict) or not typed_candidate:
                return {
                    "status": "invalid_reextract_candidate",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            if not isinstance(quality_gate, dict) or not quality_gate:
                return {
                    "status": "invalid_reextract_quality_gate",
                    "queue": file_name,
                    "review_id": _item_id(item),
                }
            updated = dict(item)
            updated["typed_candidate"] = deepcopy(typed_candidate)
            updated["content_hash"] = str(
                typed_candidate.get("content_hash") or ""
            )
            updated["quality_gate"] = deepcopy(quality_gate)
            updated["merge_allowed"] = bool(
                quality_gate.get("merge_allowed")
            )
            updated["materialize_allowed"] = bool(
                quality_gate.get("materialize_allowed")
            )
            updated["admission_readiness"] = str(
                quality_gate.get("admission_readiness") or ""
            )
            if isinstance(trace_review_payload, dict) and trace_review_payload:
                updated["corrected_trace_review_payload"] = deepcopy(
                    trace_review_payload
                )
                updated["corrected_trace_bundle_hash"] = str(
                    trace_review_payload.get("trace_bundle_hash") or ""
                )
            request = deepcopy(request)
            request["status"] = "fulfilled"
            request["fulfilled_content_hash"] = updated["content_hash"]
            request["fulfilled_at"] = datetime.now(timezone.utc).isoformat()
            updated["reextract_request"] = request
            updated["review_status"] = "pending"
            updated["human_approved"] = False
            updated.pop("selected_action", None)
            updated.pop("review_decision", None)
            updated.pop("approved_content_hash", None)
            updated["current_approval_subject_hash"] = (
                approval_subject_hash(updated)
            )
            data[idx] = updated
            self._write(file_name, data)
            return {
                "status": "reextract_fulfilled",
                "queue": file_name,
                "review_id": _item_id(updated),
                "content_hash": updated["content_hash"],
                "review_status": updated["review_status"],
                "current_approval_subject_hash": updated[
                    "current_approval_subject_hash"
                ],
            }
        return {"status": "not_found", "queue": file_name, "item_id": wanted}

    def mark_needs_re_review(
        self,
        name: str,
        item_id: str,
        *,
        reason: str,
        details: Any = None,
    ) -> dict[str, Any]:
        """Invalidate an approval whose reviewed source is no longer current."""

        file_name = _queue_name(name)
        data = self._read(file_name)
        wanted = str(item_id or "").strip()
        if not wanted:
            return {"status": "not_found", "reason": "missing_item_id", "queue": file_name}
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            nested = (
                item.get("candidate")
                if isinstance(item.get("candidate"), dict)
                else item.get("required_info_candidate")
                if isinstance(item.get("required_info_candidate"), dict)
                else {}
            )
            aliases = {
                _item_id(item),
                str(item.get("candidate_id") or ""),
                str(item.get("review_id") or ""),
                str(item.get("dedupe_key") or ""),
                str(item.get("intake_id") or ""),
            }
            if isinstance(nested, dict):
                aliases.update({
                    str(nested.get("candidate_id") or nested.get("id") or ""),
                    str(nested.get("review_id") or ""),
                    str(nested.get("dedupe_key") or ""),
                    str(nested.get("intake_id") or ""),
                })
            if wanted not in aliases:
                continue
            updated = dict(item)
            if item.get("review_decision") or item.get("review_status"):
                updated["previous_review_decision"] = {
                    "content_hash": _review_content_hash(item),
                    "review_status": item.get("review_status") or "",
                    "selected_action": item.get("selected_action") or "",
                    "review_decision": item.get("review_decision") or {},
                }
            updated["review_status"] = "needs_re_review"
            updated["human_approved"] = False
            updated.pop("approved_content_hash", None)
            updated["review_invalidation"] = {
                "reason": str(reason or "reviewed_source_changed"),
                "details": details if details is not None else [],
                "invalidated_at": datetime.now(timezone.utc).isoformat(),
            }
            updated.pop("selected_action", None)
            updated.pop("review_decision", None)
            data[idx] = updated
            self._write(file_name, data)
            return {
                "status": "needs_re_review",
                "queue": file_name,
                "review_id": _item_id(updated),
                "candidate_id": str(updated.get("candidate_id") or ""),
                "reason": updated["review_invalidation"]["reason"],
            }
        return {"status": "not_found", "queue": file_name, "item_id": wanted}

    def _read(self, file_name: str) -> list[dict[str, Any]]:
        if self.queue_dir is None:
            return self.store.read_review_queue(file_name)
        path = self.queue_dir / file_name
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _write(self, file_name: str, data: list[dict[str, Any]]) -> None:
        if self.queue_dir is None:
            self.store.write_review_queue(file_name, data)
            return
        path = self.queue_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
