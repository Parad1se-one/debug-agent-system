"""KG candidate schema and semantic validators.

The validator is deliberately local and dependency-free.  It enforces the write
boundary used by W2/W4/W5/W6: LLM extraction may propose candidates, but only
schema-valid and semantically safe candidates can reach approved merge.
"""

from __future__ import annotations

from typing import Any

PRIMARY_KEYS = {
    "Error": "error_id",
    "DiagnosticCheck": "check_id",
    "Solution": "solution_id",
    "Site": "site_id",
    "SoftwareVersion": "version_id",
    "DiagnosticTrace": "trace_id",
    "DiagnosticOutcome": "outcome_id",
    "DiagnosticPolicy": "policy_id",
}

NON_VERIFIED_OUTCOMES = {
    "ineffective",
    "partial_temporary",
    "mitigation_observed",
    "recurred",
    "pending_validation",
    "diagnostic_method",
    "context_not_root_cause",
}
OUTCOME_TYPES = {"verified_fix", *NON_VERIFIED_OUTCOMES}
REQUIRED_INFO_SLOTS = {
    "log_package",
    "software_version",
    "error_phase",
    "error_message",
    "device_model",
    "site",
    "ip_config",
    "repro_steps",
    "sample_image",
    "program_file",
    "environment",
    "owner_context",
    "other",
}
REQUIRED_INFO_REQUIRED_FIELDS = {
    "slot",
    "question",
    "condition",
    "blocks",
    "priority",
    "why_required",
    "evidence",
}


def node_pk(node: dict[str, Any]) -> str:
    node_type = str(node.get("type") or "")
    key = PRIMARY_KEYS.get(node_type, "id")
    return str(node.get(key) or node.get("id") or "")


def edge_relation(edge: dict[str, Any]) -> str:
    return str(edge.get("relation") or edge.get("type") or "")


def validate_nodes_edges(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    node_by_id: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            issues.append(f"node_not_object:{idx}")
            continue
        node_type = str(node.get("type") or "")
        pk = PRIMARY_KEYS.get(node_type)
        if not pk:
            issues.append(f"unsupported_node_type:{node_type or idx}")
            continue
        value = str(node.get(pk) or "")
        if not value:
            issues.append(f"missing_pk:{node_type}.{pk}")
        else:
            if value in seen_ids:
                issues.append(f"duplicate_node_id:{value}")
            seen_ids.add(value)
            node_by_id[value] = node
        issues.extend(_node_required_issues(node_type, node))
        issues.extend(_node_semantic_issues(node_type, value, node))
    for edge in edges:
        if not isinstance(edge, dict):
            issues.append("invalid_edge:not_object")
            continue
        rel = edge_relation(edge)
        from_id = str(edge.get("from") or "")
        to_id = str(edge.get("to") or "")
        if not from_id or not to_id or not rel:
            issues.append("invalid_edge:missing_from_to_relation")
            continue
        if from_id not in seen_ids:
            issues.append(f"invalid_edge:from_not_in_nodes:{from_id}")
        # alias_of and references to existing canonical KG nodes may point outside this proposal.
        if to_id not in seen_ids and rel not in {"alias_of", "documented_in", "references"}:
            issues.append(f"invalid_edge:to_not_in_nodes:{to_id}")
    return sorted(set(issues))


def validate_candidate(candidate: dict[str, Any]) -> list[str]:
    nodes = [node for node in candidate.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in candidate.get("edges") or [] if isinstance(edge, dict)]
    issues = validate_nodes_edges(nodes, edges)
    issues.extend(semantic_schema_issues(candidate))
    for idx, item in enumerate(candidate.get("required_info_candidates") or []):
        if isinstance(item, dict):
            issues.extend(validate_required_info_candidate(item, prefix=f"required_info_candidate:{idx}"))
        else:
            issues.append(f"required_info_candidate_not_object:{idx}")
    return sorted(set(issues))


def semantic_schema_issues(candidate: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    nodes = [node for node in candidate.get("nodes") or [] if isinstance(node, dict)]
    node_by_id = {node_pk(node): node for node in nodes if node_pk(node)}
    outcomes = [node for node in nodes if node.get("type") == "DiagnosticOutcome"]
    outcomes.extend(x for x in candidate.get("diagnostic_outcomes") or [] if isinstance(x, dict))
    for node in nodes:
        node_type = str(node.get("type") or "")
        nid = node_pk(node)
        if node_type == "Error" and str(node.get("entry_role") or "") == "case_variant":
            has_alias = any(
                isinstance(edge, dict)
                and edge_relation(edge) == "alias_of"
                and str(edge.get("from") or "") == nid
                and edge.get("to")
                for edge in candidate.get("edges") or []
            )
            if not (node.get("canonical_error_id") or has_alias):
                issues.append(f"case_variant_missing_canonical:{nid}")
        if node_type == "DiagnosticPolicy" and not node.get("deterministic_recompute"):
            issues.append(f"policy_node_must_be_deterministic_recompute:{nid}")
    for edge in candidate.get("edges") or []:
        if not isinstance(edge, dict) or edge_relation(edge) != "resolved_by":
            continue
        to_id = str(edge.get("to") or "")
        solution = node_by_id.get(to_id, {})
        if str(solution.get("evidence_level") or "") in NON_VERIFIED_OUTCOMES:
            issues.append(f"resolved_by_non_verified_solution:{to_id}")
        for outcome in outcomes:
            if str(outcome.get("target_solution_id") or "") == to_id and str(outcome.get("outcome_type") or "") != "verified_fix":
                issues.append(f"resolved_by_non_verified_outcome:{to_id}:{outcome.get('outcome_type')}")
    return sorted(set(issues))


def validate_required_info_candidate(item: dict[str, Any], *, prefix: str = "required_info_candidate") -> list[str]:
    issues: list[str] = []
    slot = str(item.get("slot") or "")
    if slot not in REQUIRED_INFO_SLOTS:
        issues.append(f"{prefix}:invalid_slot:{slot}")
    if not item.get("question"):
        issues.append(f"{prefix}:missing_question")
    if not item.get("why_required"):
        issues.append(f"{prefix}:missing_why_required")
    if not item.get("evidence_message_ids"):
        issues.append(f"{prefix}:missing_evidence")
    if not (item.get("target_error_id") or item.get("merge_policy") == "review_only"):
        issues.append(f"{prefix}:missing_target_or_review_only")
    return issues


def validate_required_info_schema_item(item: dict[str, Any], *, prefix: str = "required_info_schema") -> list[str]:
    issues: list[str] = []
    for key in sorted(REQUIRED_INFO_REQUIRED_FIELDS):
        if key not in item:
            issues.append(f"{prefix}:missing_{key}")
    slot = str(item.get("slot") or "")
    if slot and slot not in REQUIRED_INFO_SLOTS:
        issues.append(f"{prefix}:invalid_slot:{slot}")
    return issues


def _node_required_issues(node_type: str, node: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if node_type == "DiagnosticCheck":
        if not node.get("how_to_check"):
            issues.append("missing_required:DiagnosticCheck.how_to_check")
        if node.get("step_order") in (None, ""):
            issues.append("missing_required:DiagnosticCheck.step_order")
    elif node_type == "Solution":
        if not node.get("method"):
            issues.append("missing_required:Solution.method")
        if not node.get("evidence_level"):
            issues.append("missing_required:Solution.evidence_level")
    elif node_type == "DiagnosticTrace":
        for key in ("source_episode_id", "target_error_id", "evidence_message_ids"):
            if not node.get(key):
                issues.append(f"missing_required:DiagnosticTrace.{key}")
    elif node_type == "DiagnosticOutcome":
        for key in ("source_episode_id", "target_error_id", "action_label", "outcome_type", "evidence_message_ids"):
            if not node.get(key):
                issues.append(f"missing_required:DiagnosticOutcome.{key}")
    elif node_type == "DiagnosticPolicy":
        for key in ("target_error_id", "updated_at"):
            if not node.get(key):
                issues.append(f"missing_required:DiagnosticPolicy.{key}")
    return issues


def _node_semantic_issues(node_type: str, node_id: str, node: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if node_type == "DiagnosticOutcome":
        outcome_type = str(node.get("outcome_type") or "")
        if outcome_type not in OUTCOME_TYPES:
            issues.append(f"invalid_outcome_type:{node_id}:{outcome_type}")
        if node.get("high_cost") and outcome_type == "verified_fix" and not node.get("root_cause_summary"):
            issues.append(f"high_cost_verified_fix_missing_root_cause:{node_id}")
    return issues
