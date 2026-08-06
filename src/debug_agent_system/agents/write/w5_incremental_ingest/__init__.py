"""W5 approved-only ingest boundary and dry-run merge planning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from debug_agent_system.knowledge.schema_validator import semantic_schema_issues

from debug_agent_system.knowledge.store import KGStore

PK = {
    "Error": "error_id",
    "DiagnosticCheck": "check_id",
    "Solution": "solution_id",
    "Site": "site_id",
    "SoftwareVersion": "version_id",
    "DiagnosticTrace": "trace_id",
    "DiagnosticOutcome": "outcome_id",
    "DiagnosticPolicy": "policy_id",
}
NODE_INDEX_ATTR = {
    "Error": "errors_by_id",
    "DiagnosticCheck": "checks_by_id",
    "Solution": "solutions_by_id",
    "Site": "sites_by_id",
    "SoftwareVersion": "software_versions_by_id",
    "DiagnosticTrace": "traces_by_id",
    "DiagnosticOutcome": "outcomes_by_id",
    "DiagnosticPolicy": "policies_by_id",
}


def _candidate(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("candidate")
    return nested if isinstance(nested, dict) else payload


def _required_info_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("required_info_candidate")
    return nested if isinstance(nested, dict) else payload


def _pk(node: dict[str, Any]) -> str:
    key = PK.get(str(node.get("type") or ""), "id")
    return str(node.get(key) or node.get("id") or "")


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(edge.get("from") or ""),
        str(edge.get("to") or ""),
        str(edge.get("relation") or ""),
        str(edge.get("condition") or ""),
    )


def _node_has_fillable_delta(existing: dict[str, Any], node: dict[str, Any]) -> bool:
    node_type = str(node.get("type") or "")
    pk = PK.get(node_type, "id")
    for key, value in node.items():
        if key in {"proposal_only", "id", "type"} and key != pk:
            continue
        if value in (None, "", []):
            continue
        if key not in existing or existing.get(key) in (None, "", []):
            return True
    return False


def _approved(payload: dict[str, Any]) -> bool:
    return bool(payload.get("human_approved")) or str(payload.get("review_status") or payload.get("status") or "") in {"approved", "human_approved", "accepted"} or str(payload.get("selected_action") or "") in {"approve", "accept", "merge"}


def _merge_existing_error_id(payload: dict[str, Any], candidate: dict[str, Any], store: KGStore | None = None) -> str:
    conflict = payload.get("conflict") if isinstance(payload.get("conflict"), dict) else {}
    existing = candidate.get("matched_existing_error") if isinstance(candidate.get("matched_existing_error"), dict) else {}
    existing_error_id = str(conflict.get("existing_error_id") or existing.get("error_id") or "")
    if existing_error_id:
        return existing_error_id
    # Fallback: if the candidate points at an existing canonical error, use it.
    error_nodes = [node for node in candidate.get("nodes") or [] if isinstance(node, dict) and node.get("type") == "Error"]
    canonical_ids = [str(node.get("canonical_error_id") or "") for node in error_nodes if str(node.get("canonical_error_id") or "")]
    if not canonical_ids or store is None:
        return ""
    index = getattr(store, NODE_INDEX_ATTR.get("Error", ""), None)
    if not isinstance(index, dict):
        return ""
    for canonical_id in canonical_ids:
        if canonical_id in index:
            return canonical_id
    return ""


def _rewrite_merge_candidate_for_existing_error(payload: dict[str, Any], candidate: dict[str, Any], store: KGStore | None = None) -> dict[str, Any]:
    existing_error_id = _merge_existing_error_id(payload, candidate, store)
    if not existing_error_id:
        return candidate
    rewritten = deepcopy(candidate)
    queue = str(payload.get("queue") or "")
    error_nodes = [node for node in rewritten.get("nodes") or [] if isinstance(node, dict) and node.get("type") == "Error"]
    candidate_error_ids = {
        str(node.get("error_id") or "")
        for node in error_nodes
        if str(node.get("error_id") or "")
    }
    if queue == "merge_candidates":
        rewritten["nodes"] = [
            node
            for node in rewritten.get("nodes") or []
            if not (
                isinstance(node, dict)
                and node.get("type") == "Error"
                and str(node.get("error_id") or "") in candidate_error_ids
            )
        ]
    for node in rewritten.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") in {"DiagnosticTrace", "DiagnosticOutcome", "DiagnosticPolicy"} and node.get("target_error_id"):
            node["target_error_id"] = existing_error_id
    if isinstance(rewritten.get("diagnostic_trace"), dict) and rewritten["diagnostic_trace"].get("target_error_id"):
        rewritten["diagnostic_trace"]["target_error_id"] = existing_error_id
    normalized_outcomes = []
    for outcome in rewritten.get("diagnostic_outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        clean = dict(outcome)
        if clean.get("target_error_id"):
            clean["target_error_id"] = existing_error_id
        normalized_outcomes.append(clean)
    if normalized_outcomes:
        rewritten["diagnostic_outcomes"] = normalized_outcomes
    new_edges = []
    for edge in rewritten.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        clean = dict(edge)
        if str(clean.get("from") or "") in candidate_error_ids and clean.get("relation") in {"has_check", "has_trace", "has_outcome", "has_policy", "affects_version", "related_site"}:
            clean["from"] = existing_error_id
        if str(clean.get("to") or "") in candidate_error_ids and clean.get("relation") in {"alias_of", "same_as"}:
            clean["to"] = existing_error_id
        if clean.get("relation") == "alias_of":
            # legacy read-side does not consume alias_of; for approved merge into existing error this edge is noise
            continue
        new_edges.append(clean)
    rewritten["edges"] = new_edges
    rewritten["merged_into_existing_error_id"] = existing_error_id
    return rewritten


class IncrementalIngestAgent:
    """W5: approved-only write boundary; default path is dry-run/review-log."""

    def __init__(self, store: KGStore) -> None:
        self.store = store

    def dry_run(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return self.dry_run_merge_plan(candidate)

    def dry_run_merge_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = _candidate(payload)
        existing_error_id = _merge_existing_error_id(payload, candidate, self.store)
        candidate = _rewrite_merge_candidate_for_existing_error(payload, candidate, self.store)
        would_create_nodes: list[dict[str, Any]] = []
        would_update_nodes: list[dict[str, Any]] = []
        would_skip_nodes: list[dict[str, Any]] = []
        for node in candidate.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = _pk(node)
            item = {"type": node.get("type"), "id": node_id, "node": node}
            existing_node = self._existing_node(str(node.get("type") or ""), node_id)
            if existing_node and _node_has_fillable_delta(existing_node, node):
                would_update_nodes.append(item)
            elif existing_node:
                would_skip_nodes.append(item)
            else:
                would_create_nodes.append(item)
        existing_edge_keys = self._existing_edge_keys()
        would_create_edges: list[dict[str, Any]] = []
        would_skip_edges: list[dict[str, Any]] = []
        for edge in candidate.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            edge_key = _edge_key(edge)
            if not all(edge_key[:3]) or edge_key in existing_edge_keys:
                would_skip_edges.append(edge)
            else:
                would_create_edges.append(edge)
        duplicate_candidate = self._seen_candidate(candidate)
        schema_issues = [str(x) for x in candidate.get("schema_issues") or []]
        schema_issues.extend(semantic_schema_issues(candidate))
        if not candidate.get("schema_valid") and not schema_issues:
            schema_issues.append("schema_invalid")
        affected_error_ids = sorted({
            str(node.get("target_error_id") or node.get("error_id") or "")
            for node in candidate.get("nodes") or []
            if isinstance(node, dict) and (node.get("type") in {"Error", "DiagnosticTrace", "DiagnosticOutcome"} and (node.get("target_error_id") or node.get("error_id")))
        })
        return {
            "status": "dry_run_merge_plan",
            "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
            "existing_error_id": existing_error_id,
            "would_create_nodes": would_create_nodes,
            "would_update_nodes": would_update_nodes,
            "would_skip_nodes": would_skip_nodes,
            "would_create_edges": would_create_edges,
            "would_update_edges": [],
            "would_skip_edges": would_skip_edges,
            "duplicate_candidate": duplicate_candidate,
            "schema_issues": schema_issues,
            "schema_valid": bool(candidate.get("schema_valid")) and not schema_issues,
            "affects_existing_check_chain": bool(existing_error_id and any(node.get("type") == "DiagnosticCheck" for node in candidate.get("nodes") or [])),
            "would_recompute_policies_for": affected_error_ids,
            "observability": {"agent_id": "W5", "mode": "dry_run_merge_plan"},
        }

    def apply_approved(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = _candidate(payload)
        if not (_approved(payload) or _approved(candidate)):
            return {"status": "skipped", "reason": "not_approved", "candidate_id": candidate.get("candidate_id") or candidate.get("id")}
        if not candidate.get("schema_valid"):
            return {"status": "skipped", "reason": "schema_invalid", "schema_issues": candidate.get("schema_issues") or [], "candidate_id": candidate.get("candidate_id") or candidate.get("id")}
        approved_candidate = _rewrite_merge_candidate_for_existing_error(payload, candidate, self.store)
        approved_candidate["status"] = "approved"
        approved_candidate["human_approved"] = True
        return self.store.apply_approved(approved_candidate)

    def dry_run_required_info_merge(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = _required_info_candidate(payload)
        slot = str(candidate.get("slot") or "")
        target_error_id = str(candidate.get("target_error_id") or "")
        merge_policy = str(candidate.get("merge_policy") or "")
        question = str(candidate.get("question") or candidate.get("label") or "").strip()
        target_error = self._existing_node("Error", target_error_id) if target_error_id else None
        required_info = [str(x) for x in (target_error or {}).get("required_info") or []]
        issues: list[str] = []
        if not slot:
            issues.append("missing_slot")
        if not question:
            issues.append("missing_question")
        if merge_policy == "review_only":
            issues.append("review_only_no_main_graph_write")
        if not target_error_id and merge_policy != "review_only":
            issues.append("missing_target_error_id")
        if target_error_id and merge_policy != "review_only" and not target_error:
            issues.append("target_error_not_found")
        if not candidate.get("evidence_message_ids"):
            issues.append("missing_evidence")
        already_present = bool(question and question in required_info)
        fatal_issues = {issue for issue in issues if issue != "review_only_no_main_graph_write"}
        return {
            "status": "dry_run_required_info_merge",
            "candidate_id": candidate.get("candidate_id") or "",
            "target_error_id": target_error_id,
            "target_error_exists": bool(target_error),
            "slot": slot,
            "merge_policy": merge_policy,
            "would_update_error_id": target_error_id if target_error_id and merge_policy != "review_only" else "",
            "would_append_required_info": bool(target_error and merge_policy != "review_only" and slot and question and not fatal_issues and not already_present),
            "required_info_already_present": already_present,
            "schema_issues": sorted(set(issues)),
            "schema_valid": not fatal_issues,
            "observability": {"agent_id": "W5", "mode": "dry_run_required_info_merge"},
        }

    def apply_approved_required_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = _required_info_candidate(payload)
        if not (_approved(payload) or _approved(candidate)):
            return {"status": "skipped", "reason": "not_approved", "candidate_id": candidate.get("candidate_id") or ""}
        if str(candidate.get("merge_policy") or "") == "review_only":
            return {"status": "skipped", "reason": "review_only", "candidate_id": candidate.get("candidate_id") or ""}
        if not hasattr(self.store, "apply_required_info_approved"):
            return {"status": "skipped", "reason": "store_missing_required_info_merge", "candidate_id": candidate.get("candidate_id") or ""}
        return self.store.apply_required_info_approved(candidate)  # type: ignore[attr-defined]

    def _seen_candidate(self, candidate: dict[str, Any]) -> bool:
        candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "")
        if not candidate_id:
            return False
        for queue in ("candidates.json", "merge_candidates.json", "noise_candidates.json", "ask_info_candidates.json", "approved_applied.json"):
            try:
                rows = self.store.read_review_queue(queue)
            except Exception:  # noqa: BLE001 - store protocol may be minimal in tests
                rows = []
            for row in rows:
                nested = row.get("candidate") if isinstance(row, dict) else None
                row_id = str((nested or row).get("candidate_id") or (nested or row).get("id") or "") if isinstance((nested or row), dict) else ""
                if row_id == candidate_id:
                    return True
        return False

    def _existing_node(self, node_type: str, node_id: str) -> dict[str, Any] | None:
        if not node_type or not node_id:
            return None
        index = getattr(self.store, NODE_INDEX_ATTR.get(node_type, ""), None)
        if isinstance(index, dict):
            node = index.get(node_id)
            return node if isinstance(node, dict) else None
        return None

    def _existing_edge_keys(self) -> set[tuple[str, str, str, str]]:
        edges = getattr(self.store, "edges", None)
        if not isinstance(edges, list):
            return set()
        return {_edge_key(edge) for edge in edges if isinstance(edge, dict)}
