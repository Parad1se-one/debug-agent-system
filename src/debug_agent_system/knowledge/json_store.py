"""JSON-backed KGStore for the independent subsystem.

This module reads `data/kg` only. It deliberately does not import legacy runtime code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge.schema_validator import semantic_schema_issues

from debug_agent_system.core.contracts import Candidate, CheckNode, LockedSubgraph, SolutionNode

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_./:-]+")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_json_files(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*.json")):
        data = _load_json(path, [])
        if isinstance(data, list):
            out.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict):
            if all(isinstance(v, dict) for v in data.values()):
                out.extend(data.values())
            else:
                out.append(data)
    return out


NODE_FILE_BY_TYPE = {
    "Error": ("errors", "errors.json", "error_id"),
    "DiagnosticCheck": ("checks", "checks.json", "check_id"),
    "Solution": ("solutions", "solutions.json", "solution_id"),
    "Site": ("sites", "sites.json", "site_id"),
    "SoftwareVersion": ("versions", "versions.json", "version_id"),
    "DiagnosticTrace": ("traces", "traces.json", "trace_id"),
    "DiagnosticOutcome": ("outcomes", "outcomes.json", "outcome_id"),
    "DiagnosticPolicy": ("policies", "policies.json", "policy_id"),
}


def _node_identity(node: dict[str, Any]) -> tuple[str, str]:
    node_type = str(node.get("type") or "")
    pk = NODE_FILE_BY_TYPE.get(node_type, ("", "", "id"))[2]
    return node_type, str(node.get(pk) or node.get("id") or "")


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(edge.get("from") or ""),
        str(edge.get("to") or ""),
        str(edge.get("relation") or ""),
        str(edge.get("condition") or ""),
    )


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(_WORD.findall(lowered))
    cjk = _CJK.findall(lowered)
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add("".join(cjk[i : i + 2]))
    return {t for t in tokens if t.strip()}


def _node_text(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "label", "symptom", "category", "subsystem", "scenario", "source_title",
        "content", "how_to_check", "summary", "action_label", "outcome_type",
        "root_cause_summary", "condition",
    ):
        value = node.get(key)
        if value:
            parts.append(str(value))
    for key in ("keywords", "required_info", "condition_tags", "source_trace_ids", "source_outcome_ids"):
        value = node.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value)
    for key in ("required_info_schema", "ordered_checks", "solution_stats", "unsafe_actions"):
        value = node.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if isinstance(v, (str, int, float)))
                else:
                    parts.append(str(item))
    return " ".join(parts)


def _score_prepared(query: str, q_tokens: set[str], node: dict[str, Any], text: str, n_tokens: set[str]) -> tuple[float, list[str]]:
    q = query.lower()
    overlap = q_tokens & n_tokens
    score = 0.0
    evidence: list[str] = []
    if overlap:
        score += len(overlap) / max(math.sqrt(len(q_tokens) or 1), 1.0) * 2.0
        evidence.append("token_overlap:" + ",".join(sorted(list(overlap))[:8]))
    for kw in node.get("keywords") or []:
        s = str(kw).lower().strip()
        if s and s in q:
            score += min(4.0, max(1.0, len(s) / 2.5))
            evidence.append(f"keyword:{kw}")
    for key, weight in (("label", 5.0), ("symptom", 3.0), ("source_title", 2.0)):
        value = str(node.get(key) or "").lower().strip()
        if value and (value in q or q in value):
            score += weight
            evidence.append(f"{key}:contains")
    for phrase in ("初始化", "相机", "光源", "复判", "ct", "闪退", "运控", "漏检", "误报", "拍照", "ip"):
        if phrase in q and phrase in text:
            score += 1.5
            evidence.append(f"phrase:{phrase}")
    return score, evidence


def _score(query: str, node: dict[str, Any]) -> tuple[float, list[str]]:
    text = _node_text(node).lower()
    return _score_prepared(query, _tokens(query), node, text, _tokens(text))


class JsonKGStore:
    def __init__(self, kg_root: str | Path) -> None:
        self.root = Path(kg_root)
        self.errors = _flatten_json_files(self.root / "instances" / "errors")
        self.checks = _flatten_json_files(self.root / "instances" / "checks")
        self.solutions = _flatten_json_files(self.root / "instances" / "solutions")
        self.sites = _flatten_json_files(self.root / "instances" / "sites")
        self.software_versions = _flatten_json_files(self.root / "instances" / "versions")
        self.traces = _flatten_json_files(self.root / "instances" / "traces")
        self.outcomes = _flatten_json_files(self.root / "instances" / "outcomes")
        self.policies = _flatten_json_files(self.root / "instances" / "policies")
        self.edges = _load_json(self.root / "edges.json", [])
        self.errors_by_id = {str(x.get("error_id")): x for x in self.errors if x.get("error_id")}
        self.checks_by_id = {str(x.get("check_id")): x for x in self.checks if x.get("check_id")}
        self.solutions_by_id = {str(x.get("solution_id")): x for x in self.solutions if x.get("solution_id")}
        self.sites_by_id = {str(x.get("site_id")): x for x in self.sites if x.get("site_id")}
        self.software_versions_by_id = {str(x.get("version_id")): x for x in self.software_versions if x.get("version_id")}
        self.traces_by_id = {str(x.get("trace_id")): x for x in self.traces if x.get("trace_id")}
        self.outcomes_by_id = {str(x.get("outcome_id")): x for x in self.outcomes if x.get("outcome_id")}
        self.policies_by_id = {str(x.get("policy_id")): x for x in self.policies if x.get("policy_id")}
        self.error_search_index: list[tuple[dict[str, Any], str, set[str]]] = []
        for node in self.errors:
            text = _node_text(node).lower()
            self.error_search_index.append((node, text, _tokens(text)))
        self.edges_from: dict[str, list[dict[str, Any]]] = {}
        for edge in self.edges if isinstance(self.edges, list) else []:
            if isinstance(edge, dict):
                self.edges_from.setdefault(str(edge.get("from") or ""), []).append(edge)

    def search_errors(self, query: str, limit: int = 5) -> list[Candidate]:
        ranked: list[Candidate] = []
        q_tokens = _tokens(query)
        for node, text, n_tokens in self.error_search_index:
            error_id = str(node.get("error_id") or "")
            if not error_id:
                continue
            score, evidence = _score_prepared(query, q_tokens, node, text, n_tokens)
            if str(node.get("entry_role") or "") == "case_variant" and score > 0:
                score += 0.35
                evidence.append("entry_role:case_variant")
            if score <= 0:
                continue
            ranked.append(Candidate(
                error_id=error_id,
                label=str(node.get("label") or node.get("symptom") or error_id),
                score=round(score, 4),
                evidence=evidence,
                payload=node,
            ))
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked[:limit]

    def load_locked_subgraph(self, error_id: str) -> LockedSubgraph:
        error = self.errors_by_id.get(error_id)
        if not error:
            raise KeyError(f"unknown error_id: {error_id}")
        check_refs = self._reachable_check_refs(error_id)
        policy = self._policy_for_error(error_id)
        outcomes = self._outcomes_for_error(error_id)
        outcomes_by_check: dict[str, list[dict[str, Any]]] = {}
        outcomes_by_solution: dict[str, list[dict[str, Any]]] = {}
        for outcome in outcomes:
            if outcome.get("target_check_id"):
                outcomes_by_check.setdefault(str(outcome.get("target_check_id")), []).append(outcome)
            if outcome.get("target_solution_id"):
                outcomes_by_solution.setdefault(str(outcome.get("target_solution_id")), []).append(outcome)
        checks: list[CheckNode] = []
        solutions_by_check: dict[str, list[SolutionNode]] = {}
        for check_id, depth, incoming_edge in check_refs:
            raw = self.checks_by_id.get(check_id)
            if not raw:
                continue
            payload = dict(raw)
            payload["_graph_depth"] = depth
            payload["_incoming_relation"] = str(incoming_edge.get("relation") or "")
            payload["_incoming_condition"] = str(incoming_edge.get("condition") or "")
            payload["_source_error_id"] = error_id
            payload["_source_error_label"] = str(error.get("label") or error_id)
            payload["_historical_outcomes"] = outcomes_by_check.get(check_id, [])
            payload["_diagnostic_policy"] = policy
            check = CheckNode(
                check_id=check_id,
                label=str(raw.get("label") or check_id),
                how_to_check=str(raw.get("how_to_check") or raw.get("label") or check_id),
                step_order=int(raw.get("step_order") or 0),
                destructive=_is_destructive(raw),
                payload=payload,
            )
            checks.append(check)
            sols: list[SolutionNode] = []
            for edge in self.edges_from.get(check_id, []):
                if edge.get("relation") != "resolved_by":
                    continue
                raw_sol = self.solutions_by_id.get(str(edge.get("to")))
                if not raw_sol:
                    continue
                sol_payload = dict(raw_sol)
                sol_payload["_edge_condition"] = str(edge.get("condition") or "")
                sol_payload["_historical_outcomes"] = outcomes_by_solution.get(str(raw_sol.get("solution_id") or edge.get("to")), [])
                sols.append(SolutionNode(
                    solution_id=str(raw_sol.get("solution_id") or edge.get("to")),
                    content=str(raw_sol.get("content") or ""),
                    evidence_level=str(raw_sol.get("evidence_level") or ""),
                    destructive=_is_destructive(raw_sol),
                    payload=sol_payload,
                ))
            check.payload["_solution_text"] = " ".join(
                str(x.content) for x in sols if x.content
            )
            solutions_by_check[check_id] = sols
        checks.sort(key=lambda c: (int(c.payload.get("_graph_depth") or 0), c.step_order or 9999, c.check_id))
        loaded_check_ids = {c.check_id for c in checks}
        next_edges_by_check: dict[str, list[dict[str, Any]]] = {}
        for check in checks:
            for edge in self.edges_from.get(check.check_id, []):
                if edge.get("relation") != "next" or not edge.get("to"):
                    continue
                to_check_id = str(edge.get("to") or "")
                if to_check_id not in loaded_check_ids:
                    continue
                raw_to = self.checks_by_id.get(to_check_id) or {}
                next_edges_by_check.setdefault(check.check_id, []).append({
                    "from_check_id": check.check_id,
                    "to_check_id": to_check_id,
                    "to_label": str(raw_to.get("label") or to_check_id),
                    "condition": str(edge.get("condition") or ""),
                    "relation": "next",
                })
        sources = sorted({str(error.get("source_title") or error.get("source") or "KG")})
        error_payload = dict(error)
        error_payload["_diagnostic_outcomes"] = outcomes
        error_payload["_diagnostic_policy"] = policy
        return LockedSubgraph(
            error_id=error_id,
            label=str(error.get("label") or error_id),
            symptom=str(error.get("symptom") or ""),
            category=str(error.get("category") or ""),
            escalation_target=str(error.get("escalation_target") or ""),
            required_info=_required_info_labels(error),
            checks=checks,
            solutions_by_check=solutions_by_check,
            next_edges_by_check=next_edges_by_check,
            sources=sources,
            payload=error_payload,
        )

    def _reachable_check_refs(self, error_id: str, max_depth: int = 8) -> list[tuple[str, int, dict[str, Any]]]:
        """Return direct `has_check` checks plus reachable `next` checks.

        The read-side agent needs the full diagnostic topology, not just the
        entry checks directly attached to an Error.  Conditions on `next` edges
        are preserved in the check payload so B/D can rank branches later.
        """
        roots = [e for e in self.edges_from.get(error_id, []) if e.get("relation") == "has_check"]
        queue: list[tuple[str, int, dict[str, Any]]] = [
            (str(edge.get("to") or ""), 0, edge) for edge in roots if edge.get("to")
        ]
        out: list[tuple[str, int, dict[str, Any]]] = []
        seen: set[str] = set()
        while queue:
            check_id, depth, incoming_edge = queue.pop(0)
            if not check_id or check_id in seen:
                continue
            seen.add(check_id)
            out.append((check_id, depth, incoming_edge))
            if depth >= max_depth:
                continue
            for edge in self.edges_from.get(check_id, []):
                if edge.get("relation") == "next" and edge.get("to"):
                    queue.append((str(edge.get("to")), depth + 1, edge))
        return out

    def read_review_queue(self, name: str) -> list[dict]:
        path = self.root / "review_queue" / name
        data = _load_json(path, [])
        return data if isinstance(data, list) else []

    def write_review_queue(self, name: str, data: list[dict]) -> None:
        path = self.root / "review_queue" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def dry_run_apply(self, candidate: dict) -> dict:
        return {"status": "dry_run", "would_apply": bool(candidate), "candidate_id": candidate.get("id") or candidate.get("candidate_id")}

    def apply_approved(self, candidate: dict) -> dict:
        if str(candidate.get("status") or "") not in {"approved", "human_approved"} and not candidate.get("human_approved"):
            return {"status": "skipped", "reason": "not_approved"}
        if not candidate.get("schema_valid"):
            return {"status": "skipped", "reason": "schema_invalid", "candidate_id": candidate.get("id") or candidate.get("candidate_id")}
        semantic_issues = semantic_schema_issues(candidate)
        if semantic_issues:
            return {
                "status": "skipped",
                "reason": "semantic_schema_invalid",
                "schema_issues": semantic_issues,
                "candidate_id": candidate.get("id") or candidate.get("candidate_id"),
            }
        candidate_id = str(candidate.get("id") or candidate.get("candidate_id") or "")
        audit = self.read_review_queue("approved_applied.json")
        if candidate_id and any(str((row.get("candidate") if isinstance(row.get("candidate"), dict) else row).get("candidate_id") or (row.get("candidate") if isinstance(row.get("candidate"), dict) else row).get("id") or "") == candidate_id for row in audit if isinstance(row, dict)):
            return {"status": "already_applied", "candidate_id": candidate_id}

        node_result = self._merge_candidate_nodes(candidate)
        edge_result = self._merge_candidate_edges(candidate)
        self.__init__(self.root)
        policy_result = self._recompute_policies_for_candidate(candidate)
        audit_item = {
            "candidate_id": candidate_id,
            "status": "applied",
            "candidate": candidate,
            "node_result": node_result,
            "edge_result": edge_result,
            "policy_result": policy_result,
        }
        audit.append(audit_item)
        self.write_review_queue("approved_applied.json", audit)
        self.__init__(self.root)
        return {
            "status": "applied_to_graph",
            "candidate_id": candidate_id,
            "created_nodes": node_result["created"],
            "updated_nodes": node_result["updated"],
            "created_edges": edge_result["created"],
            "policy_result": policy_result,
        }

    def _merge_candidate_nodes(self, candidate: dict[str, Any]) -> dict[str, int]:
        counts = {"created": 0, "updated": 0, "skipped": 0}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for node in candidate.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_type, node_id = _node_identity(node)
            if not node_type or not node_id or node_type not in NODE_FILE_BY_TYPE:
                counts["skipped"] += 1
                continue
            grouped.setdefault(node_type, []).append(dict(node))
        for node_type, nodes in grouped.items():
            folder, file_name, pk = NODE_FILE_BY_TYPE[node_type]
            path = self.root / "instances" / folder / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = _load_json(path, [])
            if not isinstance(data, list):
                data = list(data.values()) if isinstance(data, dict) else []
            index = {str(item.get(pk) or item.get("id") or ""): item for item in data if isinstance(item, dict)}
            for node in nodes:
                node_id = str(node.get(pk) or node.get("id") or "")
                clean_node = {k: v for k, v in node.items() if k not in {"id", "type", "proposal_only"} or k == pk}
                if node_id in index:
                    existing = index[node_id]
                    changed = False
                    for key, value in clean_node.items():
                        if key not in existing or existing.get(key) in (None, "", []):
                            existing[key] = value
                            changed = True
                    if changed:
                        counts["updated"] += 1
                    else:
                        counts["skipped"] += 1
                else:
                    data.append(clean_node)
                    index[node_id] = clean_node
                    counts["created"] += 1
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return counts

    def _merge_candidate_edges(self, candidate: dict[str, Any]) -> dict[str, int]:
        result = {"created": 0, "skipped": 0}
        path = self.root / "edges.json"
        edges = _load_json(path, [])
        if not isinstance(edges, list):
            edges = []
        seen = {_edge_key(edge) for edge in edges if isinstance(edge, dict)}
        for edge in candidate.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            if not _candidate_edge_allowed(candidate, edge):
                result["skipped"] += 1
                continue
            key = _edge_key(edge)
            if not all(key[:3]) or key in seen:
                result["skipped"] += 1
                continue
            clean_edge = dict(edge)
            clean_edge.pop("proposal_only", None)
            edges.append(clean_edge)
            seen.add(key)
            result["created"] += 1
        path.write_text(json.dumps(edges, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def recompute_diagnostic_policy(self, error_id: str) -> dict[str, Any]:
        error_id = str(error_id or "")
        if not error_id:
            return {"status": "skipped", "reason": "missing_error_id"}
        outcomes = self._outcomes_for_error(error_id)
        traces = self._traces_for_error(error_id)
        policy_id = f"policy:{_safe_id(error_id)}"
        ordered_checks = _aggregate_ordered_checks(traces, outcomes)
        solution_stats = _aggregate_solution_stats(outcomes)
        unsafe_actions = [
            {
                "outcome_id": outcome.get("outcome_id"),
                "action_label": outcome.get("action_label"),
                "outcome_type": outcome.get("outcome_type"),
                "high_cost": bool(outcome.get("high_cost")),
                "destructive": bool(outcome.get("destructive")),
                "evidence_message_ids": outcome.get("evidence_message_ids") or [],
            }
            for outcome in outcomes
            if outcome.get("high_cost") or outcome.get("destructive")
        ]
        policy = {
            "type": "DiagnosticPolicy",
            "policy_id": policy_id,
            "target_error_id": error_id,
            "source_trace_ids": sorted(str(x.get("trace_id")) for x in traces if x.get("trace_id")),
            "source_outcome_ids": sorted(str(x.get("outcome_id")) for x in outcomes if x.get("outcome_id")),
            "ordered_checks": ordered_checks,
            "solution_stats": solution_stats,
            "unsafe_actions": unsafe_actions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "deterministic_recompute": True,
        }
        changed = self._upsert_node("DiagnosticPolicy", policy)
        edge_changed = self._ensure_edge({"from": error_id, "to": policy_id, "relation": "has_policy"})
        self.__init__(self.root)
        return {
            "status": "policy_recomputed",
            "target_error_id": error_id,
            "policy_id": policy_id,
            "trace_count": len(traces),
            "outcome_count": len(outcomes),
            "changed": changed,
            "edge_changed": edge_changed,
        }

    def _recompute_policies_for_candidate(self, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        error_ids: set[str] = set()
        for node in candidate.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            if node.get("type") == "Error" and node.get("error_id"):
                error_ids.add(str(node.get("error_id")))
            if node.get("type") in {"DiagnosticTrace", "DiagnosticOutcome"} and node.get("target_error_id"):
                error_ids.add(str(node.get("target_error_id")))
        for outcome in candidate.get("diagnostic_outcomes") or []:
            if isinstance(outcome, dict) and outcome.get("target_error_id"):
                error_ids.add(str(outcome.get("target_error_id")))
        for trace in [candidate.get("diagnostic_trace")]:
            if isinstance(trace, dict) and trace.get("target_error_id"):
                error_ids.add(str(trace.get("target_error_id")))
        return [self.recompute_diagnostic_policy(error_id) for error_id in sorted(error_ids)]

    def _outcomes_for_error(self, error_id: str) -> list[dict[str, Any]]:
        edge_ids = {
            str(edge.get("to"))
            for edge in self.edges if isinstance(edge, dict)
            if edge.get("relation") == "has_outcome" and str(edge.get("from") or "") == error_id and edge.get("to")
        }
        return [
            outcome for outcome in self.outcomes
            if str(outcome.get("target_error_id") or "") == error_id or str(outcome.get("outcome_id") or "") in edge_ids
        ]

    def _traces_for_error(self, error_id: str) -> list[dict[str, Any]]:
        edge_ids = {
            str(edge.get("to"))
            for edge in self.edges if isinstance(edge, dict)
            if edge.get("relation") == "has_trace" and str(edge.get("from") or "") == error_id and edge.get("to")
        }
        return [
            trace for trace in self.traces
            if str(trace.get("target_error_id") or "") == error_id or str(trace.get("trace_id") or "") in edge_ids
        ]

    def _policy_for_error(self, error_id: str) -> dict[str, Any]:
        edge_policy_ids = [
            str(edge.get("to"))
            for edge in self.edges if isinstance(edge, dict)
            if edge.get("relation") == "has_policy" and str(edge.get("from") or "") == error_id and edge.get("to")
        ]
        for policy_id in edge_policy_ids:
            policy = self.policies_by_id.get(policy_id)
            if policy:
                return policy
        for policy in self.policies:
            if str(policy.get("target_error_id") or "") == error_id:
                return policy
        return {}

    def _upsert_node(self, node_type: str, node: dict[str, Any]) -> bool:
        folder, file_name, pk = NODE_FILE_BY_TYPE[node_type]
        path = self.root / "instances" / folder / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _load_json(path, [])
        if not isinstance(data, list):
            data = list(data.values()) if isinstance(data, dict) else []
        node_id = str(node.get(pk) or "")
        changed = False
        for idx, existing in enumerate(data):
            if isinstance(existing, dict) and str(existing.get(pk) or "") == node_id:
                if existing != node:
                    data[idx] = node
                    changed = True
                break
        else:
            data.append(node)
            changed = True
        if changed:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return changed

    def _ensure_edge(self, edge: dict[str, Any]) -> bool:
        path = self.root / "edges.json"
        edges = _load_json(path, [])
        if not isinstance(edges, list):
            edges = []
        key = _edge_key(edge)
        if key in {_edge_key(item) for item in edges if isinstance(item, dict)}:
            return False
        edges.append(edge)
        path.write_text(json.dumps(edges, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True

    def apply_required_info_approved(self, candidate: dict) -> dict:
        target_error_id = str(candidate.get("target_error_id") or "")
        if not target_error_id:
            return {"status": "skipped", "reason": "missing_target_error_id", "candidate_id": candidate.get("candidate_id") or ""}
        if str(candidate.get("merge_policy") or "") == "review_only":
            return {"status": "skipped", "reason": "review_only", "candidate_id": candidate.get("candidate_id") or ""}
        slot = str(candidate.get("slot") or "other")
        question = str(candidate.get("question") or candidate.get("label") or "").strip()
        if not question:
            return {"status": "skipped", "reason": "missing_question", "candidate_id": candidate.get("candidate_id") or ""}
        for path in sorted((self.root / "instances" / "errors").glob("*.json")):
            data = _load_json(path, [])
            updated = self._merge_required_info_in_payload(data, target_error_id, slot, question, candidate)
            if updated != "not_found":
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.__init__(self.root)
                return {
                    "status": "required_info_already_applied" if updated == "already_applied" else "required_info_applied",
                    "candidate_id": candidate.get("candidate_id") or "",
                    "target_error_id": target_error_id,
                    "slot": slot,
                    "file": str(path),
                }
        return {"status": "skipped", "reason": "target_error_not_found", "target_error_id": target_error_id, "candidate_id": candidate.get("candidate_id") or ""}

    def _merge_required_info_in_payload(self, data: Any, error_id: str, slot: str, question: str, candidate: dict) -> str:
        if isinstance(data, list):
            for node in data:
                if isinstance(node, dict) and str(node.get("error_id") or "") == error_id:
                    return self._merge_required_info_node(node, slot, question, candidate)
        elif isinstance(data, dict):
            if str(data.get("error_id") or "") == error_id:
                return self._merge_required_info_node(data, slot, question, candidate)
            for node in data.values():
                if isinstance(node, dict) and str(node.get("error_id") or "") == error_id:
                    return self._merge_required_info_node(node, slot, question, candidate)
        return "not_found"

    @staticmethod
    def _candidate_evidence_key(candidate: dict) -> str:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id:
            return f"candidate:{candidate_id}"
        episode_id = str(candidate.get("source_episode_id") or "").strip()
        evidence = ",".join(sorted(str(x) for x in candidate.get("evidence_message_ids") or [] if str(x).strip()))
        slot = str(candidate.get("slot") or "other")
        return f"evidence:{episode_id}:{slot}:{evidence}"

    @staticmethod
    def _merge_required_info_node(node: dict[str, Any], slot: str, question: str, candidate: dict) -> str:
        required = node.get("required_info")
        if not isinstance(required, list):
            required = []
        if question not in [str(x) for x in required]:
            required.append(question)
        node["required_info"] = required
        schema = node.get("required_info_schema")
        if not isinstance(schema, list):
            schema = []
        evidence_key = JsonKGStore._candidate_evidence_key(candidate)
        schema_entry = {
            "slot": slot,
            "question": question,
            "condition": str(candidate.get("condition") or ""),
            "blocks": candidate.get("blocks") or ["diagnostic_branch_selection"],
            "priority": candidate.get("priority") or "medium",
            "why_required": str(candidate.get("why_required") or ""),
            "evidence": {
                "source_episode_ids": [candidate.get("source_episode_id")] if candidate.get("source_episode_id") else [],
                "evidence_message_ids": [str(x) for x in candidate.get("evidence_message_ids") or [] if str(x)],
                "candidate_id": str(candidate.get("candidate_id") or ""),
            },
        }
        schema_seen = False
        for item in schema:
            if not isinstance(item, dict):
                continue
            if str(item.get("slot") or "") == slot and str(item.get("condition") or "") == schema_entry["condition"]:
                schema_seen = True
                existing_evidence = item.get("evidence")
                if not isinstance(existing_evidence, dict):
                    existing_evidence = {}
                for key in ("source_episode_ids", "evidence_message_ids"):
                    merged = list(existing_evidence.get(key) or [])
                    for value in schema_entry["evidence"][key]:
                        if value and value not in merged:
                            merged.append(value)
                    existing_evidence[key] = merged
                item.setdefault("question", question)
                item.setdefault("why_required", schema_entry["why_required"])
                item.setdefault("blocks", schema_entry["blocks"])
                item.setdefault("priority", schema_entry["priority"])
                item["evidence"] = existing_evidence
                break
        if not schema_seen:
            schema.append(schema_entry)
        node["required_info_schema"] = schema
        sources = node.get("required_info_sources")
        if not isinstance(sources, dict):
            sources = {}
        entry = sources.get(slot)
        if not isinstance(entry, dict):
            entry = {"slot": slot, "source_episode_ids": [], "evidence_message_ids": [], "occurrence_count": 0}
        for key in ("source_episode_ids", "evidence_message_ids", "applied_candidate_ids", "applied_evidence_keys"):
            if not isinstance(entry.get(key), list):
                entry[key] = []
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        already_applied = evidence_key in entry["applied_evidence_keys"] or bool(candidate_id and candidate_id in entry["applied_candidate_ids"])
        for episode_id in [candidate.get("source_episode_id")]:
            if episode_id and episode_id not in entry["source_episode_ids"]:
                entry["source_episode_ids"].append(episode_id)
        for msg_id in candidate.get("evidence_message_ids") or []:
            value = str(msg_id)
            if value and value not in entry["evidence_message_ids"]:
                entry["evidence_message_ids"].append(value)
        if candidate_id and candidate_id not in entry["applied_candidate_ids"]:
            entry["applied_candidate_ids"].append(candidate_id)
        if evidence_key and evidence_key not in entry["applied_evidence_keys"]:
            entry["applied_evidence_keys"].append(evidence_key)
        if not already_applied:
            entry["occurrence_count"] = int(entry.get("occurrence_count") or 0) + 1
        else:
            entry["occurrence_count"] = int(entry.get("occurrence_count") or 0)
        entry["question"] = question
        entry["condition"] = str(candidate.get("condition") or "")
        sources[slot] = entry
        node["required_info_sources"] = sources
        return "already_applied" if already_applied else "applied"



def _required_info_labels(error: dict[str, Any]) -> list[str]:
    raw = error.get("required_info")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    schema = error.get("required_info_schema")
    labels: list[str] = []
    if isinstance(schema, list):
        for item in schema:
            if isinstance(item, dict):
                question = str(item.get("question") or item.get("label") or item.get("slot") or "")
                if question:
                    labels.append(question)
            elif item:
                labels.append(str(item))
    return labels


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value or "")).strip("-") or "unknown"


def _node_by_id(nodes: list[dict[str, Any]], node_id: str) -> dict[str, Any]:
    for node in nodes:
        if not isinstance(node, dict):
            continue
        _, current = _node_identity(node)
        if current == node_id:
            return node
    return {}


def _candidate_edge_allowed(candidate: dict[str, Any], edge: dict[str, Any]) -> bool:
    if edge.get("relation") != "resolved_by":
        return True
    to_id = str(edge.get("to") or "")
    nodes = [node for node in candidate.get("nodes") or [] if isinstance(node, dict)]
    solution = _node_by_id(nodes, to_id)
    evidence_level = str(solution.get("evidence_level") or "")
    if evidence_level in {"ineffective", "partial_temporary", "mitigation_observed", "recurred", "pending_validation", "diagnostic_method", "context_not_root_cause"}:
        return False
    outcomes = [x for x in candidate.get("diagnostic_outcomes") or [] if isinstance(x, dict)]
    for outcome in outcomes:
        if str(outcome.get("target_solution_id") or "") == to_id and str(outcome.get("outcome_type") or "") != "verified_fix":
            return False
    return True


def _iter_order_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for key in ("recommended_order", "actual_order"):
        for item in trace.get(key) or []:
            if isinstance(item, dict):
                steps.append(item)
            elif item:
                steps.append({"label": str(item)})
    return steps


def _aggregate_ordered_checks(traces: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for trace in traces:
        for index, step in enumerate(_iter_order_steps(trace), start=1):
            check_id = str(step.get("check_id") or step.get("target_check_id") or "")
            label = str(step.get("label") or step.get("action_label") or check_id or "")
            key = check_id or label
            if not key:
                continue
            entry = stats.setdefault(key, {"check_id": check_id, "label": label, "seen_count": 0, "order_sum": 0, "outcome_count": 0, "verified_fix_count": 0, "ineffective_count": 0})
            entry["seen_count"] += 1
            entry["order_sum"] += int(step.get("order") or step.get("step_order") or index)
            if not entry.get("label") and label:
                entry["label"] = label
    for outcome in outcomes:
        check_id = str(outcome.get("target_check_id") or "")
        label = str(outcome.get("action_label") or check_id or "")
        key = check_id or label
        if not key:
            continue
        entry = stats.setdefault(key, {"check_id": check_id, "label": label, "seen_count": 0, "order_sum": 0, "outcome_count": 0, "verified_fix_count": 0, "ineffective_count": 0})
        entry["outcome_count"] += 1
        outcome_type = str(outcome.get("outcome_type") or "")
        if outcome_type == "verified_fix":
            entry["verified_fix_count"] += 1
        if outcome_type == "ineffective":
            entry["ineffective_count"] += 1
    out: list[dict[str, Any]] = []
    for entry in stats.values():
        seen = max(int(entry.get("seen_count") or 0), 1)
        outcome_count = int(entry.get("outcome_count") or 0)
        has_order = bool(entry.get("order_sum"))
        avg_order = (float(entry.get("order_sum") or 0) / seen) if has_order else 999.0
        support = max(seen if has_order else 0, outcome_count)
        order_penalty = avg_order * 0.05 if has_order else (0.1 if outcome_count else 999.0 * 0.05)
        prior = float(entry.get("verified_fix_count") or 0) * 2.0 + support * 0.2 - float(entry.get("ineffective_count") or 0) * 0.8 - order_penalty
        item = dict(entry)
        item["avg_order"] = round(avg_order, 3)
        item["policy_prior"] = round(prior, 4)
        out.append(item)
    out.sort(key=lambda x: (-float(x.get("policy_prior") or 0), float(x.get("avg_order") or 999), str(x.get("check_id") or x.get("label") or "")))
    return out


def _aggregate_solution_stats(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        solution_id = str(outcome.get("target_solution_id") or "")
        action = str(outcome.get("action_label") or solution_id or "")
        key = solution_id or action
        if not key:
            continue
        entry = stats.setdefault(key, {"solution_id": solution_id, "action_label": action, "total": 0, "by_outcome_type": {}, "high_cost": False, "destructive": False})
        entry["total"] += 1
        outcome_type = str(outcome.get("outcome_type") or "unknown")
        by_type = entry["by_outcome_type"]
        by_type[outcome_type] = int(by_type.get(outcome_type) or 0) + 1
        entry["high_cost"] = bool(entry.get("high_cost")) or bool(outcome.get("high_cost"))
        entry["destructive"] = bool(entry.get("destructive")) or bool(outcome.get("destructive"))
    out = list(stats.values())
    out.sort(key=lambda x: (-int((x.get("by_outcome_type") or {}).get("verified_fix") or 0), int((x.get("by_outcome_type") or {}).get("ineffective") or 0), str(x.get("action_label") or "")))
    return out

def _is_destructive(node: dict[str, Any]) -> bool:
    text = _node_text(node)
    return any(word in text for word in ("停机", "拆机", "断电", "删除", "清空", "重装", "格式化"))
