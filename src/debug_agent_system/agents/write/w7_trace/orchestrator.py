"""Stage-level orchestration for the W7 multi-agent shadow pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from .agents import (
    CaseBoundaryAgent,
    ComponentBridgeAgent,
    ComponentConsistencyAgent,
    EvidenceAnchorAgent,
    NeighborLinkAgent,
    OutcomeReconcilerAgent,
    TracePhaseAgent,
)
from .atomic_case_adapter import build_atomic_case_manifest
from .candidate_graph import build_sparse_candidate_graph
from .checkpoint_store import CheckpointStore
from .components import (
    apply_candidate_edge_safety_guards,
    apply_component_bridge_decision,
    apply_component_consistency_decision,
    build_component_bridge_candidates,
    build_component_conflicts,
    build_trace_components,
)
from .contracts import (
    TRACE_ASSEMBLY_CASE_KINDS,
    TRACE_ROOT_CASE_KINDS,
    canonical_hash,
    dedupe_strings,
    must_link_reason_is_contradictory,
    validate_component_bridge_decision,
    validate_component_consistency_decision,
    validate_trace_link_decision,
    validate_trace_phase_patch,
)
from .model_client import DecisionModelClient
from .review import build_trace_review_payload
from .source_context import (
    attach_case_source_context,
    evidence_anchor_candidates,
)
from .trace_compiler import TraceCompiler


class W7ShadowOrchestrator:
    """Run small decision agents without mutating W2, W6, or KG state."""

    schema_version = "w7.multi_agent_shadow_result.v1"

    def __init__(
        self,
        *,
        client: DecisionModelClient | None,
        checkpoint_root: str | Path | None = None,
        boundary_chunk_rows: int = 24,
        candidate_top_k: int = 6,
        neighbor_chunk_edges: int = 24,
        component_workers: int = 1,
    ) -> None:
        self.client = client
        self.checkpoints = CheckpointStore(checkpoint_root)
        self.boundary_chunk_rows = max(1, int(boundary_chunk_rows))
        self.candidate_top_k = max(0, int(candidate_top_k))
        self.neighbor_chunk_edges = max(1, int(neighbor_chunk_edges))
        self.component_workers = max(1, int(component_workers))

    @staticmethod
    def _slice_ledger(
        ledger: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        chunk_index: int,
    ) -> dict[str, Any]:
        allowed = [
            str(row.get("message_id") or "")
            for row in rows
            if str(row.get("message_id") or "")
        ]
        output = {
            key: deepcopy(value)
            for key, value in ledger.items()
            if key not in {
                "rows",
                "allowed_message_ids",
                "allowed_attachment_ids",
                "core_message_ids",
                "ledger_hash",
                "stats",
            }
        }
        output.update({
            "schema_version": "w7.source_ledger.v2",
            "chunk_index": chunk_index,
            "rows": deepcopy(rows),
            "allowed_message_ids": allowed,
            "allowed_attachment_ids": dedupe_strings(
                attachment.get("attachment_id")
                for row in rows
                for attachment in row.get("attachment_refs") or []
                if isinstance(attachment, dict)
            ),
            "core_message_ids": [
                value
                for value in ledger.get("core_message_ids") or []
                if value in set(allowed)
            ],
            "stats": {
                "rows": len(rows),
                "attachments": sum(
                    len(row.get("attachment_refs") or []) for row in rows
                ),
            },
        })
        output["ledger_hash"] = canonical_hash(output)
        return output

    def run_case_boundary(self, ledger: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            return {
                "status": "model_disabled",
                "schema_valid": False,
                "issues": ["case_boundary_model_disabled"],
                "decision": {},
                "calls": [],
            }
        rows = [
            row for row in ledger.get("rows") or [] if isinstance(row, dict)
        ]
        chunks = [
            rows[index:index + self.boundary_chunk_rows]
            for index in range(0, len(rows), self.boundary_chunk_rows)
        ] or [[]]
        all_fragments: list[dict[str, Any]] = []
        all_non_case: list[str] = []
        all_uncertainties: list[str] = []
        all_issues: list[str] = []
        calls: list[dict[str, Any]] = []
        for chunk_index, rows_part in enumerate(chunks, 1):
            chunk = self._slice_ledger(
                ledger, rows_part, chunk_index=chunk_index
            )
            stage = f"case_boundary_{chunk_index:03d}"
            key = self.checkpoints.key(
                stage=stage,
                input_value=chunk,
                version=CaseBoundaryAgent.version,
            )
            cached = self.checkpoints.read(stage=stage, key=key)
            if cached is not None:
                decision = deepcopy(cached.get("output") or {})
                issues: list[str] = []
                call = {
                    "stage_cache_hit": True,
                    **(
                        cached.get("call")
                        if isinstance(cached.get("call"), dict)
                        else {}
                    ),
                }
            else:
                started = time.monotonic()
                try:
                    decision, issues, call = CaseBoundaryAgent(
                        self.client
                    ).decide(chunk)
                except Exception as exc:  # fail closed at the stage boundary
                    decision = {
                        "schema_version": "w7.case_boundary_decision.v1",
                        "case_fragments": [],
                        "non_case_message_ids": [],
                        "uncertainties": [],
                    }
                    issues = [
                        f"model_call_failed:{type(exc).__name__}:{exc}"
                    ]
                    call = {
                        "stage_cache_hit": False,
                        "model_call_failed": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                else:
                    call["elapsed_seconds"] = round(
                        time.monotonic() - started, 6
                    )
                    self.checkpoints.write(
                        stage=stage,
                        key=key,
                        output=decision,
                        issues=issues,
                        call=call,
                    )
            prefix = f"B{chunk_index:03d}-" if len(chunks) > 1 else ""
            for raw_fragment in decision.get("case_fragments") or []:
                fragment = deepcopy(raw_fragment)
                fragment["fragment_ref"] = (
                    prefix + str(fragment.get("fragment_ref") or "")
                )
                all_fragments.append(fragment)
            all_non_case.extend(decision.get("non_case_message_ids") or [])
            all_uncertainties.extend(
                f"chunk_{chunk_index}:{value}"
                for value in decision.get("uncertainties") or []
            )
            all_issues.extend(
                f"chunk_{chunk_index}:{value}" for value in issues
            )
            calls.append({"chunk_index": chunk_index, **call})
        combined = {
            "schema_version": "w7.case_boundary_decision.v1",
            "case_fragments": all_fragments,
            "non_case_message_ids": dedupe_strings(all_non_case),
            "uncertainties": dedupe_strings(all_uncertainties),
        }
        combined["decision_hash"] = canonical_hash(combined)
        return {
            "status": "completed" if not all_issues else "failed_closed",
            "schema_valid": not all_issues,
            "issues": sorted(set(all_issues)),
            "decision": combined,
            "calls": calls,
            "chunk_count": len(chunks),
        }

    def run_evidence_anchor(
        self,
        *,
        ledger: dict[str, Any],
        boundary: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = evidence_anchor_candidates(ledger)
        empty = {
            "schema_version": "w7.evidence_anchor_decision.v1",
            "anchor_decisions": [],
            "unassigned_evidence_message_ids": [],
            "uncertainties": [],
        }
        empty["decision_hash"] = canonical_hash(empty)
        if not candidates:
            return {
                "status": "skipped_no_attachment_evidence",
                "schema_valid": True,
                "issues": [],
                "decision": empty,
                "calls": [],
            }
        if not boundary.get("schema_valid"):
            return {
                "status": "skipped_invalid_boundary",
                "schema_valid": False,
                "issues": ["evidence_anchor_requires_valid_boundary"],
                "decision": empty,
                "calls": [],
            }
        if self.client is None:
            return {
                "status": "model_disabled",
                "schema_valid": False,
                "issues": ["evidence_anchor_model_disabled"],
                "decision": empty,
                "calls": [],
            }
        stage_input = {
            "ledger_hash": ledger.get("ledger_hash") or "",
            "case_boundary": boundary.get("decision") or {},
            "evidence_candidates": candidates,
        }
        key = self.checkpoints.key(
            stage="evidence_anchor",
            input_value=stage_input,
            version=EvidenceAnchorAgent.version,
        )
        cached = self.checkpoints.read(stage="evidence_anchor", key=key)
        if cached is not None:
            return {
                "status": "completed",
                "schema_valid": True,
                "issues": [],
                "decision": deepcopy(cached.get("output") or {}),
                "calls": [{
                    "stage_cache_hit": True,
                    **(
                        cached.get("call")
                        if isinstance(cached.get("call"), dict)
                        else {}
                    ),
                }],
            }
        started = time.monotonic()
        try:
            decision, issues, call = EvidenceAnchorAgent(
                self.client
            ).decide(
                source_ledger=ledger,
                case_boundary=boundary.get("decision") or {},
            )
        except Exception as exc:
            decision = empty
            issues = [
                f"model_call_failed:{type(exc).__name__}:{exc}"
            ]
            call = {
                "stage_cache_hit": False,
                "model_call_failed": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:
            call["elapsed_seconds"] = round(
                time.monotonic() - started, 6
            )
            self.checkpoints.write(
                stage="evidence_anchor",
                key=key,
                output=decision,
                issues=issues,
                call=call,
            )
        return {
            "status": "completed" if not issues else "failed_closed",
            "schema_valid": not issues,
            "issues": issues,
            "decision": decision,
            "calls": [call],
        }

    def build_candidate_graph(
        self, case_cards: list[dict[str, Any]]
    ) -> dict[str, Any]:
        graph = build_sparse_candidate_graph(
            case_cards, top_k=self.candidate_top_k
        )
        return {
            "status": "completed",
            "schema_valid": True,
            "issues": [],
            "graph": graph,
        }

    def run_neighbor_link(
        self,
        *,
        graph: dict[str, Any],
        case_cards: list[dict[str, Any]],
        allowed_message_ids: set[str],
    ) -> dict[str, Any]:
        required = [
            item for item in graph.get("edges") or []
            if isinstance(item, dict)
            and bool(item.get("requires_adjudication"))
        ]
        empty = {
            "schema_version": "w7.trace_link_decision.v1",
            "edge_decisions": [],
            "uncertainties": [],
        }
        empty["decision_hash"] = canonical_hash(empty)
        if not required:
            return {
                "status": "skipped_no_required_edges",
                "schema_valid": True,
                "issues": [],
                "decision": empty,
                "calls": [],
            }
        if self.client is None:
            return {
                "status": "model_disabled",
                "schema_valid": False,
                "issues": ["neighbor_link_model_disabled"],
                "decision": empty,
                "calls": [],
            }
        chunks = [
            required[index:index + self.neighbor_chunk_edges]
            for index in range(0, len(required), self.neighbor_chunk_edges)
        ]
        combined_edges: list[dict[str, Any]] = []
        combined_uncertainties: list[str] = []
        issues: list[str] = []
        calls: list[dict[str, Any]] = []

        def decide_chunk(
            item: tuple[int, list[dict[str, Any]]],
        ) -> tuple[
            int, dict[str, Any], list[str], dict[str, Any]
        ]:
            chunk_index, edge_chunk = item
            chunk_case_refs = {
                str(edge.get(key) or "")
                for edge in edge_chunk
                for key in ("left_case_ref", "right_case_ref")
                if str(edge.get(key) or "")
            }
            chunk_cards = [
                card for card in case_cards
                if str(
                    card.get("case_ref")
                    or card.get("case_item_ref")
                    or card.get("fragment_ref")
                    or ""
                ) in chunk_case_refs
            ]
            chunk_graph = {
                "schema_version": graph.get("schema_version") or "",
                "node_refs": sorted(chunk_case_refs),
                "edges": edge_chunk,
            }
            chunk_graph["graph_hash"] = canonical_hash(chunk_graph)
            stage = f"neighbor_link_{chunk_index:03d}"
            stage_input = {
                "parent_graph_hash": graph.get("graph_hash") or "",
                "chunk_graph": chunk_graph,
                "case_cards": chunk_cards,
            }
            key = self.checkpoints.key(
                stage=stage,
                input_value=stage_input,
                version=NeighborLinkAgent.version,
            )
            cached = self.checkpoints.read(stage=stage, key=key)
            if cached is not None:
                chunk_decision = deepcopy(cached.get("output") or {})
                chunk_issues: list[str] = []
                call = {
                    "stage_cache_hit": True,
                    **(
                        cached.get("call")
                        if isinstance(cached.get("call"), dict)
                        else {}
                    ),
                }
            else:
                started = time.monotonic()
                try:
                    semantic_calls: list[dict[str, Any]] = []
                    repair_issues: list[str] = []
                    for semantic_attempt in range(3):
                        transport_calls: list[dict[str, Any]] = []
                        # Transport retry belongs to DecisionModelClient.
                        # Retrying here would multiply its own attempts.
                        for transport_attempt in range(1):
                            try:
                                (
                                    chunk_decision,
                                    chunk_issues,
                                    attempt_call,
                                ) = NeighborLinkAgent(
                                    self.client
                                ).decide(
                                    graph=chunk_graph,
                                    case_cards=chunk_cards,
                                    allowed_message_ids=(
                                        allowed_message_ids
                                    ),
                                    repair_issues=repair_issues,
                                )
                            except Exception as exc:
                                transport_calls.append({
                                    "attempt": transport_attempt + 1,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                })
                                raise
                            transport_calls.append({
                                "attempt": transport_attempt + 1,
                                "success": True,
                            })
                            break
                        attempt_call["transport_attempts"] = len(
                            transport_calls
                        )
                        attempt_call["transport_calls"] = (
                            transport_calls
                        )
                        local_repairs: list[str] = []
                        safely_projectable = (
                            chunk_issues
                            and all(
                                any(
                                    marker in issue
                                    for marker in (
                                        ":unknown_edge:",
                                        ":duplicate_edge:",
                                    )
                                )
                                for issue in chunk_issues
                            )
                        )
                        if safely_projectable:
                            # The validator has already removed hallucinated
                            # or duplicate extra edges. Revalidating its
                            # canonical projection is safe only when every
                            # required input edge remains accounted for.
                            chunk_decision, repaired_issues = (
                                validate_trace_link_decision(
                                    chunk_decision,
                                    required_edges={
                                        tuple(sorted((
                                            str(
                                                edge.get(
                                                    "left_case_ref"
                                                )
                                                or ""
                                            ),
                                            str(
                                                edge.get(
                                                    "right_case_ref"
                                                )
                                                or ""
                                            ),
                                        )))
                                        for edge in edge_chunk
                                    },
                                    allowed_edges={
                                        tuple(sorted((
                                            str(
                                                edge.get(
                                                    "left_case_ref"
                                                )
                                                or ""
                                            ),
                                            str(
                                                edge.get(
                                                    "right_case_ref"
                                                )
                                                or ""
                                            ),
                                        )))
                                        for edge in edge_chunk
                                    },
                                    allowed_message_ids=(
                                        allowed_message_ids
                                    ),
                                )
                            )
                            if not repaired_issues:
                                if any(
                                    ":unknown_edge:" in issue
                                    for issue in chunk_issues
                                ):
                                    local_repairs.append(
                                        "dropped_hallucinated_extra_edges"
                                    )
                                if any(
                                    ":duplicate_edge:" in issue
                                    for issue in chunk_issues
                                ):
                                    local_repairs.append(
                                        "collapsed_duplicate_edges"
                                    )
                                chunk_issues = []
                        attempt_call["local_structural_repairs"] = (
                            local_repairs
                        )
                        semantic_calls.append(attempt_call)
                        if not chunk_issues:
                            break
                        repair_issues = list(chunk_issues)
                    if (
                        chunk_issues
                        and all(
                            ":must_link_reason_contradiction" in issue
                            for issue in chunk_issues
                        )
                    ):
                        repaired = deepcopy(chunk_decision)
                        repaired_pairs: list[str] = []
                        for edge in repaired.get(
                            "edge_decisions"
                        ) or []:
                            if (
                                not isinstance(edge, dict)
                                or str(edge.get("decision") or "")
                                != "must_link"
                                or not must_link_reason_is_contradictory(
                                    edge.get("reasons") or []
                                )
                            ):
                                continue
                            edge["decision"] = "cannot_link"
                            edge["relation_hint"] = ""
                            repaired_pairs.append(
                                f"{edge.get('left_case_ref')}:"
                                f"{edge.get('right_case_ref')}"
                            )
                        repaired.pop("decision_hash", None)
                        chunk_decision, repaired_issues = (
                            validate_trace_link_decision(
                                repaired,
                                required_edges={
                                    tuple(sorted((
                                        str(
                                            edge.get(
                                                "left_case_ref"
                                            )
                                            or ""
                                        ),
                                        str(
                                            edge.get(
                                                "right_case_ref"
                                            )
                                            or ""
                                        ),
                                    )))
                                    for edge in edge_chunk
                                },
                                allowed_edges={
                                    tuple(sorted((
                                        str(
                                            edge.get(
                                                "left_case_ref"
                                            )
                                            or ""
                                        ),
                                        str(
                                            edge.get(
                                                "right_case_ref"
                                            )
                                            or ""
                                        ),
                                    )))
                                    for edge in edge_chunk
                                },
                                allowed_message_ids=(
                                    allowed_message_ids
                                ),
                            )
                        )
                        if not repaired_issues:
                            chunk_issues = []
                            semantic_calls[-1].setdefault(
                                "local_structural_repairs",
                                [],
                            ).extend(
                                "downgraded_contradictory_must:"
                                f"{pair}"
                                for pair in repaired_pairs
                            )
                    call = {
                        **semantic_calls[-1],
                        "semantic_contract_attempts": len(
                            semantic_calls
                        ),
                        "semantic_repair_count": max(
                            0, len(semantic_calls) - 1
                        ),
                        "attempt_calls": semantic_calls,
                    }
                except Exception as exc:
                    chunk_decision = empty
                    chunk_issues = [
                        f"model_call_failed:{type(exc).__name__}:{exc}"
                    ]
                    call = {
                        "stage_cache_hit": False,
                        "model_call_failed": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                else:
                    call["elapsed_seconds"] = round(
                        time.monotonic() - started, 6
                    )
                    self.checkpoints.write(
                        stage=stage,
                        key=key,
                        output=chunk_decision,
                        issues=chunk_issues,
                        call=call,
                    )
            guarded_decision = apply_candidate_edge_safety_guards(
                chunk_decision,
                edge_chunk,
            )
            guarded_pairs = [
                (
                    f"{edge.get('left_case_ref')}:"
                    f"{edge.get('right_case_ref')}"
                )
                for edge in guarded_decision.get("edge_decisions") or []
                if isinstance(edge, dict)
                and str(edge.get("local_override_reason") or "")
                == "candidate_identity_discontinuity_guard"
            ]
            if guarded_pairs:
                call.setdefault("local_structural_repairs", []).extend(
                    f"downgraded_identity_discontinuity_must:{pair}"
                    for pair in guarded_pairs
                )
            return (
                chunk_index,
                guarded_decision,
                chunk_issues,
                call,
            )

        indexed_chunks = list(enumerate(chunks, 1))
        if self.component_workers > 1 and len(indexed_chunks) > 1:
            with ThreadPoolExecutor(
                max_workers=self.component_workers,
                thread_name_prefix="w7-link",
            ) as executor:
                chunk_results = list(
                    executor.map(decide_chunk, indexed_chunks)
                )
        else:
            chunk_results = [
                decide_chunk(item) for item in indexed_chunks
            ]
        for (
            chunk_index,
            chunk_decision,
            chunk_issues,
            call,
        ) in chunk_results:
            combined_edges.extend(
                item
                for item in chunk_decision.get("edge_decisions") or []
                if isinstance(item, dict)
            )
            combined_uncertainties.extend(
                f"chunk_{chunk_index}:{value}"
                for value in chunk_decision.get("uncertainties") or []
            )
            issues.extend(
                f"chunk_{chunk_index}:{value}"
                for value in chunk_issues
            )
            calls.append({"chunk_index": chunk_index, **call})
        decision, combined_issues = validate_trace_link_decision(
            {
                "edge_decisions": combined_edges,
                "uncertainties": combined_uncertainties,
            },
            required_edges={
                tuple(sorted((
                    str(item.get("left_case_ref") or ""),
                    str(item.get("right_case_ref") or ""),
                )))
                for item in required
            },
            allowed_edges={
                tuple(sorted((
                    str(item.get("left_case_ref") or ""),
                    str(item.get("right_case_ref") or ""),
                )))
                for item in graph.get("edges") or []
                if isinstance(item, dict)
            },
            allowed_message_ids=allowed_message_ids,
        )
        issues.extend(combined_issues)
        issues = sorted(set(issues))
        return {
            "status": "completed" if not issues else "failed_closed",
            "schema_valid": not issues,
            "issues": issues,
            "decision": decision,
            "calls": calls,
            "chunk_count": len(chunks),
        }

    def run_component_consistency(
        self,
        *,
        link_decision: dict[str, Any],
        case_cards: list[dict[str, Any]],
        allowed_message_ids: set[str],
    ) -> dict[str, Any]:
        conflicts = build_component_conflicts(link_decision)
        conflict_values = [
            item for item in conflicts.get("conflicts") or []
            if isinstance(item, dict)
        ]
        empty = {
            "schema_version": "w7.component_consistency_decision.v1",
            "conflict_decisions": [],
            "uncertainties": [],
        }
        empty["decision_hash"] = canonical_hash(empty)
        if not conflict_values:
            return {
                "status": "skipped_no_conflicts",
                "schema_valid": True,
                "issues": [],
                "warnings": [],
                "conflicts": conflicts,
                "decision": empty,
                "revised_link_decision": deepcopy(link_decision),
                "calls": [],
            }

        required_conflicts = {
            tuple(sorted((
                str(
                    (item.get("cannot_link_edge") or {}).get(
                        "left_case_ref"
                    )
                    or ""
                ),
                str(
                    (item.get("cannot_link_edge") or {}).get(
                        "right_case_ref"
                    )
                    or ""
                ),
            )))
            for item in conflict_values
        }

        def safe_fallback() -> dict[str, Any]:
            fallback, fallback_issues = (
                validate_component_consistency_decision(
                    {
                        "conflict_decisions": [{
                            "left_case_ref": left,
                            "right_case_ref": right,
                            "decision": "confirmed_cannot",
                            "evidence_message_ids": [],
                            "reasons": [
                                "模型复审不可用，保留 NeighborLink 的 "
                                "cannot_link 以 fail-closed"
                            ],
                        } for left, right in sorted(required_conflicts)],
                        "uncertainties": [
                            "component_consistency_safe_fallback"
                        ],
                    },
                    required_conflicts=required_conflicts,
                    allowed_message_ids=allowed_message_ids,
                )
            )
            if fallback_issues:
                raise AssertionError(
                    "invalid_component_consistency_fallback:"
                    + ",".join(fallback_issues)
                )
            return fallback

        stage_input = {
            "conflicts": conflicts,
            "case_cards": [
                deepcopy(item) for item in case_cards
                if isinstance(item, dict)
            ],
        }
        key = self.checkpoints.key(
            stage="component_consistency",
            input_value=stage_input,
            version=ComponentConsistencyAgent.version,
        )
        cached = self.checkpoints.read(
            stage="component_consistency",
            key=key,
        )
        warnings: list[str] = []
        if cached is not None:
            decision = deepcopy(cached.get("output") or {})
            call = {
                "stage_cache_hit": True,
                **(
                    cached.get("call")
                    if isinstance(cached.get("call"), dict)
                    else {}
                ),
            }
        elif self.client is None:
            decision = safe_fallback()
            call = {
                "stage_cache_hit": False,
                "model_disabled": True,
                "safe_fallback": True,
            }
            warnings.append("component_consistency_model_disabled")
        else:
            started = time.monotonic()
            attempt_calls: list[dict[str, Any]] = []
            model_issues: list[str] = []
            try:
                repair_issues: list[str] = []
                for _attempt in range(2):
                    decision, model_issues, attempt_call = (
                        ComponentConsistencyAgent(self.client).decide(
                            conflicts=conflicts,
                            case_cards=case_cards,
                            allowed_message_ids=allowed_message_ids,
                            repair_issues=repair_issues,
                        )
                    )
                    attempt_calls.append(attempt_call)
                    if not model_issues:
                        break
                    repair_issues = list(model_issues)
            except Exception as exc:
                decision = safe_fallback()
                call = {
                    "stage_cache_hit": False,
                    "model_call_failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "safe_fallback": True,
                    "attempt_calls": attempt_calls,
                }
                warnings.append(
                    "component_consistency_model_call_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
            else:
                if model_issues:
                    decision = safe_fallback()
                    call = {
                        "stage_cache_hit": False,
                        "semantic_contract_attempts": len(attempt_calls),
                        "semantic_repair_count": max(
                            0, len(attempt_calls) - 1
                        ),
                        "safe_fallback": True,
                        "attempt_calls": attempt_calls,
                    }
                    warnings.extend(
                        f"component_consistency_model_issue:{value}"
                        for value in model_issues
                    )
                else:
                    call = {
                        **attempt_calls[-1],
                        "stage_cache_hit": False,
                        "semantic_contract_attempts": len(attempt_calls),
                        "semantic_repair_count": max(
                            0, len(attempt_calls) - 1
                        ),
                        "attempt_calls": attempt_calls,
                    }
                    call["elapsed_seconds"] = round(
                        time.monotonic() - started, 6
                    )
                    self.checkpoints.write(
                        stage="component_consistency",
                        key=key,
                        output=decision,
                        issues=[],
                        call=call,
                    )
        revised = apply_component_consistency_decision(
            link_decision,
            decision,
        )
        return {
            "status": (
                "completed" if not warnings else "degraded_safe"
            ),
            "schema_valid": True,
            "issues": [],
            "warnings": warnings,
            "conflicts": conflicts,
            "decision": decision,
            "revised_link_decision": revised,
            "calls": [call],
        }

    def run_component_bridges(
        self,
        *,
        components: dict[str, Any],
        link_decision: dict[str, Any],
        case_cards: list[dict[str, Any]],
        allowed_message_ids: set[str],
    ) -> dict[str, Any]:
        candidates = build_component_bridge_candidates(
            components,
            link_decision,
            max_component_size=12,
        )
        candidate_values = [
            item for item in candidates.get("candidates") or []
            if isinstance(item, dict)
        ]
        empty = {
            "schema_version": "w7.component_bridge_decision.v1",
            "bridge_decisions": [],
            "uncertainties": [],
        }
        empty["decision_hash"] = canonical_hash(empty)
        if not candidate_values:
            return {
                "status": "skipped_no_bridges",
                "schema_valid": True,
                "issues": [],
                "warnings": [],
                "candidates": candidates,
                "decision": empty,
                "revised_link_decision": deepcopy(link_decision),
                "calls": [],
            }

        chunks = [
            candidate_values[index:index + self.neighbor_chunk_edges]
            for index in range(
                0,
                len(candidate_values),
                self.neighbor_chunk_edges,
            )
        ]

        def decide_chunk(
            item: tuple[int, list[dict[str, Any]]],
        ) -> tuple[
            int,
            dict[str, Any],
            list[str],
            dict[str, Any],
        ]:
            chunk_index, chunk_values = item
            chunk_candidates = {
                "schema_version": (
                    "w7.component_bridge_candidates.v1"
                ),
                "candidates": chunk_values,
                "max_component_size": 12,
            }
            chunk_candidates["candidates_hash"] = canonical_hash(
                chunk_candidates
            )
            required = {
                tuple(sorted((
                    str(value.get("left_case_ref") or ""),
                    str(value.get("right_case_ref") or ""),
                )))
                for value in chunk_values
            }

            def safe_fallback() -> dict[str, Any]:
                fallback, fallback_issues = (
                    validate_component_bridge_decision(
                        {
                            "bridge_decisions": [{
                                "left_case_ref": left,
                                "right_case_ref": right,
                                "decision": "keep_possible",
                                "evidence_message_ids": [],
                                "reasons": [
                                    "组件断桥复审不可用，保留 possible_link "
                                    "交人工审核"
                                ],
                            } for left, right in sorted(required)],
                            "uncertainties": [
                                "component_bridge_safe_fallback"
                            ],
                        },
                        required_bridges=required,
                        allowed_message_ids=allowed_message_ids,
                    )
                )
                if fallback_issues:
                    raise AssertionError(
                        "invalid_component_bridge_fallback:"
                        + ",".join(fallback_issues)
                    )
                return fallback

            stage = f"component_bridge_{chunk_index:03d}"
            stage_input = {
                "parent_candidates_hash": (
                    candidates.get("candidates_hash") or ""
                ),
                "chunk_candidates": chunk_candidates,
                "case_cards": case_cards,
            }
            key = self.checkpoints.key(
                stage=stage,
                input_value=stage_input,
                version=ComponentBridgeAgent.version,
            )
            cached = self.checkpoints.read(stage=stage, key=key)
            chunk_warnings: list[str] = []
            if cached is not None:
                decision = deepcopy(cached.get("output") or {})
                call = {
                    "stage_cache_hit": True,
                    **(
                        cached.get("call")
                        if isinstance(cached.get("call"), dict)
                        else {}
                    ),
                }
                return (
                    chunk_index,
                    decision,
                    chunk_warnings,
                    call,
                )
            if self.client is None:
                return (
                    chunk_index,
                    safe_fallback(),
                    ["component_bridge_model_disabled"],
                    {
                        "stage_cache_hit": False,
                        "model_disabled": True,
                        "safe_fallback": True,
                    },
                )
            started = time.monotonic()
            attempt_calls: list[dict[str, Any]] = []
            model_issues: list[str] = []
            try:
                repair_issues: list[str] = []
                for _attempt in range(2):
                    decision, model_issues, attempt_call = (
                        ComponentBridgeAgent(self.client).decide(
                            bridge_candidates=chunk_candidates,
                            case_cards=case_cards,
                            allowed_message_ids=allowed_message_ids,
                            repair_issues=repair_issues,
                        )
                    )
                    attempt_calls.append(attempt_call)
                    if not model_issues:
                        break
                    repair_issues = list(model_issues)
            except Exception as exc:
                decision = safe_fallback()
                chunk_warnings.append(
                    "component_bridge_model_call_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
                call = {
                    "stage_cache_hit": False,
                    "model_call_failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "safe_fallback": True,
                    "attempt_calls": attempt_calls,
                }
            else:
                if model_issues:
                    decision = safe_fallback()
                    chunk_warnings.extend(
                        f"component_bridge_model_issue:{value}"
                        for value in model_issues
                    )
                    call = {
                        "stage_cache_hit": False,
                        "semantic_contract_attempts": len(attempt_calls),
                        "semantic_repair_count": max(
                            0, len(attempt_calls) - 1
                        ),
                        "safe_fallback": True,
                        "attempt_calls": attempt_calls,
                    }
                else:
                    call = {
                        **attempt_calls[-1],
                        "stage_cache_hit": False,
                        "semantic_contract_attempts": len(attempt_calls),
                        "semantic_repair_count": max(
                            0, len(attempt_calls) - 1
                        ),
                        "attempt_calls": attempt_calls,
                        "elapsed_seconds": round(
                            time.monotonic() - started,
                            6,
                        ),
                    }
                    self.checkpoints.write(
                        stage=stage,
                        key=key,
                        output=decision,
                        issues=[],
                        call=call,
                    )
            return chunk_index, decision, chunk_warnings, call

        indexed_chunks = list(enumerate(chunks, 1))
        if self.component_workers > 1 and len(indexed_chunks) > 1:
            with ThreadPoolExecutor(
                max_workers=self.component_workers,
                thread_name_prefix="w7-bridge",
            ) as executor:
                results = list(
                    executor.map(decide_chunk, indexed_chunks)
                )
        else:
            results = [
                decide_chunk(item) for item in indexed_chunks
            ]
        combined_values: list[dict[str, Any]] = []
        combined_uncertainties: list[str] = []
        warnings: list[str] = []
        calls: list[dict[str, Any]] = []
        for chunk_index, decision, chunk_warnings, call in results:
            combined_values.extend(
                value
                for value in decision.get("bridge_decisions") or []
                if isinstance(value, dict)
            )
            combined_uncertainties.extend(
                f"chunk_{chunk_index}:{value}"
                for value in decision.get("uncertainties") or []
            )
            warnings.extend(
                f"chunk_{chunk_index}:{value}"
                for value in chunk_warnings
            )
            calls.append({"chunk_index": chunk_index, **call})
        decision, issues = validate_component_bridge_decision(
            {
                "bridge_decisions": combined_values,
                "uncertainties": combined_uncertainties,
            },
            required_bridges={
                tuple(sorted((
                    str(value.get("left_case_ref") or ""),
                    str(value.get("right_case_ref") or ""),
                )))
                for value in candidate_values
            },
            allowed_message_ids=allowed_message_ids,
        )
        if issues:
            # Per-chunk validated outputs should make this unreachable. Keep
            # every bridge possible rather than failing the safe shadow run.
            decision = empty
            warnings.extend(
                f"combined_validation:{value}" for value in issues
            )
        revised = apply_component_bridge_decision(
            link_decision,
            decision,
        )
        return {
            "status": (
                "completed" if not warnings else "degraded_safe"
            ),
            "schema_valid": True,
            "issues": [],
            "warnings": sorted(set(warnings)),
            "candidates": candidates,
            "decision": decision,
            "revised_link_decision": revised,
            "calls": calls,
            "chunk_count": len(chunks),
        }

    def run_trace_phases(
        self,
        *,
        components: dict[str, Any],
        case_cards: list[dict[str, Any]],
        link_decision: dict[str, Any],
        allowed_message_ids: set[str],
    ) -> dict[str, Any]:
        component_values = [
            item for item in components.get("components") or []
            if isinstance(item, dict)
        ]
        combined_operations: list[dict[str, Any]] = []
        combined_uncertainties: list[str] = []
        standalone_case_refs: list[str] = []
        issues: list[str] = []
        calls: list[dict[str, Any]] = []
        if not component_values:
            decision = {
                "schema_version": "w7.trace_phase_patch.v1",
                "operations": [],
                "uncertainties": [],
            }
            decision["decision_hash"] = canonical_hash(decision)
            return {
                "status": "skipped_no_components",
                "schema_valid": True,
                "issues": [],
                "decision": decision,
                "calls": [],
            }
        if self.client is None:
            return {
                "status": "model_disabled",
                "schema_valid": False,
                "issues": ["trace_phase_model_disabled"],
                "decision": {},
                "calls": [],
            }
        def decide_component(
            item: tuple[int, dict[str, Any]],
        ) -> tuple[
            int, bool, set[str], dict[str, Any], list[str], dict[str, Any]
        ]:
            component_index, component = item
            component_refs = set(component.get("case_refs") or [])
            component_cards = [
                item for item in case_cards
                if isinstance(item, dict)
                and str(
                    item.get("case_ref")
                    or item.get("case_item_ref")
                    or item.get("fragment_ref")
                    or ""
                ) in component_refs
            ]
            has_trace_root = any(
                str(item.get("case_kind") or "diagnostic_case")
                in TRACE_ROOT_CASE_KINDS
                for item in component_cards
            )
            if not has_trace_root:
                return (
                    component_index,
                    True,
                    component_refs,
                    {"operations": [], "uncertainties": []},
                    [],
                    {
                    "stage_skipped": True,
                    "reason": "component_without_diagnostic_root",
                    },
                )
            stage = f"trace_phase_{component_index:03d}"
            component_allowed_message_ids = set(dedupe_strings(
                message_id
                for card in component_cards
                for message_id in [
                    *(card.get("source_message_ids") or []),
                    *(card.get("evidence_message_ids") or []),
                ]
                if message_id in allowed_message_ids
            ))
            stage_input = {
                "component": component,
                "case_cards": case_cards,
                "link_decision": link_decision,
                "allowed_message_ids": sorted(
                    component_allowed_message_ids
                ),
            }
            key = self.checkpoints.key(
                stage=stage,
                input_value=stage_input,
                version=TracePhaseAgent.version,
            )
            cached = self.checkpoints.read(stage=stage, key=key)
            if cached is not None:
                decision = deepcopy(cached.get("output") or {})
                stage_issues: list[str] = []
                call = {
                    "stage_cache_hit": True,
                    **(
                        cached.get("call")
                        if isinstance(cached.get("call"), dict)
                        else {}
                    ),
                }
            else:
                started = time.monotonic()
                try:
                    repair_issues: list[str] = []
                    attempt_calls: list[dict[str, Any]] = []
                    # Initial decision plus one schema-grounded semantic
                    # repair. Transport retry is already handled by the
                    # DecisionModelClient and must not multiply this loop.
                    for attempt in range(2):
                        decision, stage_issues, attempt_call = (
                            TracePhaseAgent(self.client).decide(
                                component=component,
                                case_cards=case_cards,
                                link_decisions=list(
                                    link_decision.get(
                                        "edge_decisions"
                                    )
                                    or []
                                ),
                                allowed_message_ids=(
                                    component_allowed_message_ids
                                ),
                                repair_issues=repair_issues,
                            )
                        )
                        attempt_calls.append(attempt_call)
                        if not stage_issues:
                            break
                        repair_issues = list(stage_issues)
                    local_structural_repairs: list[str] = []
                    if (
                        stage_issues
                        and all(
                            ":duplicate_phase:" in issue
                            for issue in stage_issues
                        )
                    ):
                        # The model sometimes expands the internal lifecycle
                        # of one already-atomic case into several set_phase
                        # operations. The contract validator has already kept
                        # the first occurrence and dropped later duplicates.
                        # Revalidate that canonical projection rather than
                        # weakening the one-case/one-phase invariant.
                        repaired = deepcopy(decision)
                        duplicate_case_refs = dedupe_strings(
                            issue.split(":duplicate_phase:", 1)[1]
                            for issue in stage_issues
                        )
                        local_structural_repairs = [
                            f"collapsed_duplicate_phase:{case_ref}"
                            for case_ref in duplicate_case_refs
                        ]
                        repaired["uncertainties"] = dedupe_strings([
                            *(repaired.get("uncertainties") or []),
                            *(
                                f"local_structural_repair:{value}"
                                for value in local_structural_repairs
                            ),
                        ])
                        repaired.pop("decision_hash", None)
                        allowed_by_case = {
                            str(
                                card.get("case_ref")
                                or card.get("case_item_ref")
                                or card.get("fragment_ref")
                                or ""
                            ): set(dedupe_strings([
                                *(card.get("source_message_ids") or []),
                                *(card.get("evidence_message_ids") or []),
                            ]))
                            for card in component_cards
                        }
                        decision, stage_issues = validate_trace_phase_patch(
                            repaired,
                            component_case_refs=component_refs,
                            allowed_message_ids=(
                                component_allowed_message_ids
                            ),
                            allowed_message_ids_by_case=allowed_by_case,
                        )
                    call = {
                        **attempt_calls[-1],
                        "semantic_contract_attempts": len(
                            attempt_calls
                        ),
                        "semantic_repair_count": max(
                            int(
                                attempt_calls[-1].get(
                                    "semantic_repair_count"
                                )
                                or 0
                            ),
                            len(attempt_calls) - 1,
                        ),
                        "attempt_calls": attempt_calls,
                        "local_structural_repairs": (
                            local_structural_repairs
                        ),
                    }
                except Exception as exc:
                    decision = {
                        "operations": [],
                        "uncertainties": [],
                    }
                    stage_issues = [
                        f"model_call_failed:{type(exc).__name__}:{exc}"
                    ]
                    call = {
                        "stage_cache_hit": False,
                        "model_call_failed": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                else:
                    call["elapsed_seconds"] = round(
                        time.monotonic() - started, 6
                    )
                    self.checkpoints.write(
                        stage=stage,
                        key=key,
                        output=decision,
                        issues=stage_issues,
                        call=call,
                    )
            return (
                component_index,
                bool(stage_issues),
                component_refs,
                decision,
                stage_issues,
                call,
            )

        indexed_components = list(enumerate(component_values, 1))
        if self.component_workers > 1 and len(indexed_components) > 1:
            with ThreadPoolExecutor(
                max_workers=self.component_workers,
                thread_name_prefix="w7-phase",
            ) as executor:
                component_results = list(
                    executor.map(decide_component, indexed_components)
                )
        else:
            component_results = [
                decide_component(item) for item in indexed_components
            ]

        for (
            component_index,
            standalone,
            component_refs,
            decision,
            stage_issues,
            call,
        ) in component_results:
            if standalone:
                standalone_case_refs.extend(sorted(component_refs))
                issues.extend(
                    f"component_{component_index}:{value}"
                    for value in stage_issues
                )
                if stage_issues:
                    combined_uncertainties.append(
                        f"component_{component_index}:"
                        "safe_standalone_after_phase_failure"
                    )
                calls.append({"component_index": component_index, **call})
                continue
            prefix = f"C{component_index:03d}-"
            for operation in decision.get("operations") or []:
                if not isinstance(operation, dict):
                    continue
                current = deepcopy(operation)
                current["local_trace_ref"] = (
                    prefix + str(current.get("local_trace_ref") or "")
                )
                combined_operations.append(current)
            combined_uncertainties.extend(
                f"component_{component_index}:{value}"
                for value in decision.get("uncertainties") or []
            )
            issues.extend(
                f"component_{component_index}:{value}"
                for value in stage_issues
            )
            calls.append({"component_index": component_index, **call})
        combined = {
            "schema_version": "w7.trace_phase_patch.v1",
            "operations": combined_operations,
            "standalone_case_refs": dedupe_strings(
                standalone_case_refs
            ),
            "uncertainties": dedupe_strings(combined_uncertainties),
        }
        combined["decision_hash"] = canonical_hash(combined)
        return {
            "status": "completed" if not issues else "failed_closed",
            "schema_valid": not issues,
            "issues": sorted(set(issues)),
            "decision": combined,
            "calls": calls,
            "component_count": len(component_values),
        }

    @staticmethod
    def _trace_cards_from_phase(
        phase_decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        phases_by_trace: dict[str, list[dict[str, Any]]] = {}
        groups: dict[str, list[str]] = {}
        for operation in phase_decision.get("operations") or []:
            if not isinstance(operation, dict):
                continue
            trace_ref = str(operation.get("local_trace_ref") or "")
            if operation.get("op") == "create_trace_group":
                groups[trace_ref] = list(operation.get("case_refs") or [])
            elif operation.get("op") == "set_phase":
                phases_by_trace.setdefault(trace_ref, []).append(operation)
        return [
            {
                "local_trace_ref": trace_ref,
                "case_refs": case_refs,
                "phases": sorted(
                    phases_by_trace.get(trace_ref, []),
                    key=lambda item: int(item.get("phase_index") or 0),
                ),
                "resolution_status": "unknown",
            }
            for trace_ref, case_refs in sorted(groups.items())
        ]

    def run_outcome_reconciliation(
        self,
        *,
        traces: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        allowed_message_ids: set[str],
    ) -> dict[str, Any]:
        if not traces:
            return {
                "status": "skipped_no_traces",
                "schema_valid": True,
                "issues": [],
                "decision": {
                    "schema_version": "w7.outcome_patch.v1",
                    "operations": [],
                    "uncertainties": [],
                },
                "calls": [],
            }
        if self.client is None:
            return {
                "status": "model_disabled",
                "schema_valid": False,
                "issues": ["outcome_reconciler_model_disabled"],
                "decision": {},
                "calls": [],
            }
        stage_input = {
            "traces": traces,
            "evidence_rows": evidence_rows,
            "allowed_message_ids": sorted(allowed_message_ids),
        }
        key = self.checkpoints.key(
            stage="outcome_reconciler",
            input_value=stage_input,
            version=OutcomeReconcilerAgent.version,
        )
        cached = self.checkpoints.read(
            stage="outcome_reconciler", key=key
        )
        if cached is not None:
            decision = deepcopy(cached.get("output") or {})
            issues: list[str] = []
            call = {
                "stage_cache_hit": True,
                **(
                    cached.get("call")
                    if isinstance(cached.get("call"), dict)
                    else {}
                ),
            }
        else:
            started = time.monotonic()
            try:
                repair_issues: list[str] = []
                attempt_calls: list[dict[str, Any]] = []
                for _attempt in range(2):
                    decision, issues, attempt_call = (
                        OutcomeReconcilerAgent(self.client).decide(
                            traces=traces,
                            evidence_rows=evidence_rows,
                            allowed_message_ids=allowed_message_ids,
                            repair_issues=repair_issues,
                        )
                    )
                    attempt_calls.append(attempt_call)
                    if not issues:
                        break
                    repair_issues = list(issues)
                call = {
                    **attempt_calls[-1],
                    "semantic_contract_attempts": len(
                        attempt_calls
                    ),
                    "semantic_repair_count": max(
                        int(
                            attempt_calls[-1].get(
                                "semantic_repair_count"
                            )
                            or 0
                        ),
                        len(attempt_calls) - 1,
                    ),
                    "attempt_calls": attempt_calls,
                }
            except Exception as exc:  # fail closed at the stage boundary
                decision = {
                    "schema_version": "w7.outcome_patch.v1",
                    "operations": [],
                    "uncertainties": [],
                }
                issues = [
                    f"model_call_failed:{type(exc).__name__}:{exc}"
                ]
                call = {
                    "stage_cache_hit": False,
                    "model_call_failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            else:
                call["elapsed_seconds"] = round(
                    time.monotonic() - started, 6
                )
                self.checkpoints.write(
                    stage="outcome_reconciler",
                    key=key,
                    output=decision,
                    issues=issues,
                    call=call,
                )
        return {
            "status": "completed" if not issues else "failed_closed",
            "schema_valid": not issues,
            "issues": issues,
            "decision": decision,
            "calls": [call],
        }

    def run(
        self,
        *,
        ledger: dict[str, Any],
        case_cards: list[dict[str, Any]] | None = None,
        trace_cards: list[dict[str, Any]] | None = None,
        episode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        w7a = self.run_w7a(ledger=ledger, episode=episode)
        boundary = w7a["case_boundary"]
        cards = list(case_cards or [])
        if not cards and boundary.get("schema_valid"):
            cards = [
                {
                    **deepcopy(item),
                    "case_ref": str(item.get("fragment_ref") or ""),
                    "title": str(item.get("fault_summary") or ""),
                    "evidence_message_ids": list(
                        item.get("source_message_ids") or []
                    ),
                }
                for item in (
                    (boundary.get("decision") or {}).get(
                        "case_fragments"
                    )
                    or []
                )
                if isinstance(item, dict)
                and str(item.get("case_kind") or "")
                in TRACE_ASSEMBLY_CASE_KINDS
            ]
        w7b = self.run_w7b(
            ledger=ledger,
            case_cards=cards,
            trace_cards=trace_cards,
            prior_decisions={
                "case_boundary": boundary.get("decision") or {},
                "evidence_anchor": (
                    w7a["evidence_anchor"].get("decision") or {}
                ),
            },
            prior_issues=[
                *w7a["case_boundary"].get("issues", []),
                *w7a["evidence_anchor"].get("issues", []),
                *w7a["atomic_case_adapter"].get("issues", []),
            ],
        )
        required_stages = [
            w7a["case_boundary"],
            w7a["evidence_anchor"],
            w7a["atomic_case_adapter"],
            w7b["candidate_graph"],
            w7b["neighbor_link"],
            w7b["component_consistency"],
            w7b["component_bridge"],
            w7b["trace_components"],
            w7b["trace_phase"],
            w7b["outcome_reconciliation"],
            w7b["trace_compiler"],
        ]
        schema_valid = all(
            bool(stage.get("schema_valid"))
            for stage in required_stages
        )
        result = {
            "schema_version": self.schema_version,
            "mode": "shadow_multi_agent",
            "source_only": True,
            "promotion_allowed": False,
            "legacy_authoritative": True,
            "source_ledger_hash": str(
                ledger.get("ledger_hash") or canonical_hash(ledger)
            ),
            **w7a,
            **w7b,
            "schema_valid": schema_valid,
            "review_required": True,
            "fallback_policy": "keep_legacy_w7",
        }
        result["result_hash"] = canonical_hash(result)
        return result

    def run_w7a(
        self,
        *,
        ledger: dict[str, Any],
        episode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run evidence/case boundary decisions for one source unit."""

        boundary = self.run_case_boundary(ledger)
        anchor = self.run_evidence_anchor(
            ledger=ledger,
            boundary=boundary,
        )
        atomic_manifest: dict[str, Any] = {}
        atomic_issues: list[str] = []
        if boundary.get("schema_valid") and anchor.get("schema_valid"):
            atomic_manifest, atomic_issues = build_atomic_case_manifest(
                episode=deepcopy(episode) if isinstance(episode, dict) else {
                    "episode_id": ledger.get("episode_id") or "",
                    "thread_id": ledger.get("source_thread_id") or "",
                    "messages": list(ledger.get("rows") or []),
                },
                source_ledger=ledger,
                case_boundary=boundary.get("decision") or {},
                evidence_anchor=anchor.get("decision") or {},
            )
        atomic = {
            "status": (
                "completed" if atomic_manifest and not atomic_issues
                else "failed_closed"
            ),
            "schema_valid": bool(atomic_manifest) and not atomic_issues,
            "issues": atomic_issues or (
                [] if atomic_manifest else ["atomic_case_dependencies_invalid"]
            ),
            "manifest": atomic_manifest,
        }
        return {
            "case_boundary": boundary,
            "evidence_anchor": anchor,
            "atomic_case_adapter": atomic,
        }

    def run_w7b(
        self,
        *,
        ledger: dict[str, Any],
        case_cards: list[dict[str, Any]],
        trace_cards: list[dict[str, Any]] | None = None,
        prior_decisions: dict[str, Any] | None = None,
        prior_issues: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run cross-case Trace linking, phase, outcome and compilation."""

        cards = [
            deepcopy(item)
            for item in case_cards
            if isinstance(item, dict)
        ]
        decision_cards = attach_case_source_context(cards, ledger)
        graph = self.build_candidate_graph(cards)
        link = self.run_neighbor_link(
            graph=graph.get("graph") or {},
            case_cards=decision_cards,
            allowed_message_ids=set(ledger.get("allowed_message_ids") or []),
        )
        consistency = self.run_component_consistency(
            link_decision=link.get("decision") or {},
            case_cards=decision_cards,
            allowed_message_ids=set(
                ledger.get("allowed_message_ids") or []
            ),
        )
        revised_link_decision = (
            consistency.get("revised_link_decision") or {}
        )
        initial_components, _initial_component_issues = (
            build_trace_components(
                graph.get("graph") or {},
                revised_link_decision,
                max_component_size=12,
            )
        )
        bridge = self.run_component_bridges(
            components=initial_components,
            link_decision=revised_link_decision,
            case_cards=decision_cards,
            allowed_message_ids=set(
                ledger.get("allowed_message_ids") or []
            ),
        )
        final_link_decision = (
            bridge.get("revised_link_decision") or {}
        )
        components_value, component_issues = build_trace_components(
            graph.get("graph") or {},
            final_link_decision,
            max_component_size=12,
        )
        components = {
            "status": "completed" if not component_issues else "failed_closed",
            "schema_valid": not component_issues,
            "issues": component_issues,
            "graph": components_value,
        }
        phase = self.run_trace_phases(
            components=components_value,
            case_cards=decision_cards,
            link_decision=final_link_decision,
            allowed_message_ids=set(ledger.get("allowed_message_ids") or []),
        )
        phase_trace_cards = (
            self._trace_cards_from_phase(phase.get("decision") or {})
            if phase.get("schema_valid")
            else []
        )
        active_trace_cards = list(trace_cards or phase_trace_cards)
        ledger_allowed_message_ids = set(
            ledger.get("allowed_message_ids") or []
        )
        cards_by_ref = {
            str(
                card.get("case_ref")
                or card.get("case_item_ref")
                or card.get("fragment_ref")
                or ""
            ): card
            for card in cards
            if str(
                card.get("case_ref")
                or card.get("case_item_ref")
                or card.get("fragment_ref")
                or ""
            )
        }
        outcome_allowed_message_ids = set(dedupe_strings(
            message_id
            for trace in active_trace_cards
            for message_id in [
                *(trace.get("evidence_message_ids") or []),
                *(
                    phase_message_id
                    for phase in trace.get("phases") or []
                    if isinstance(phase, dict)
                    for phase_message_id in (
                        phase.get("evidence_message_ids") or []
                    )
                ),
                *(
                    card_message_id
                    for case_ref in trace.get("case_refs") or []
                    for card in [cards_by_ref.get(str(case_ref))]
                    if isinstance(card, dict)
                    for card_message_id in [
                        *(card.get("source_message_ids") or []),
                        *(card.get("evidence_message_ids") or []),
                    ]
                ),
            ]
            if message_id in ledger_allowed_message_ids
        ))
        # Explicit trace cards are a bounded caller contract. Older callers may
        # provide only the trace ref/status, without case/evidence linkage; keep
        # that path usable while automatically assembled traces remain scoped
        # to their own case-card evidence.
        if trace_cards is not None and not outcome_allowed_message_ids:
            outcome_allowed_message_ids = set(
                ledger_allowed_message_ids
            )
        outcome = self.run_outcome_reconciliation(
            traces=active_trace_cards,
            evidence_rows=[
                row for row in ledger.get("rows") or []
                if isinstance(row, dict)
                and str(row.get("message_id") or "")
                in outcome_allowed_message_ids
            ],
            allowed_message_ids=outcome_allowed_message_ids,
        )
        compiled_bundle = TraceCompiler().compile_review_bundle(
            case_cards=cards,
            phase_patch=phase.get("decision") or {},
            outcome_patch=outcome.get("decision") or {},
        )
        compiler_issues = [
            f"compiled_case_unassigned:{case_ref}"
            for case_ref in compiled_bundle.get("unassigned_case_refs") or []
        ]
        compiler = {
            "status": (
                "completed"
                if phase.get("schema_valid")
                and outcome.get("schema_valid")
                and not compiler_issues
                else "failed_closed"
            ),
            "schema_valid": bool(
                phase.get("schema_valid")
                and outcome.get("schema_valid")
                and not compiler_issues
            ),
            "issues": compiler_issues,
            "bundle": compiled_bundle,
        }
        trace_review_payload = build_trace_review_payload(
            source_ledger_hash=str(
                ledger.get("ledger_hash") or canonical_hash(ledger)
            ),
            decisions={
                **deepcopy(prior_decisions or {}),
                "neighbor_link": link.get("decision") or {},
                "component_consistency": (
                    consistency.get("decision") or {}
                ),
                "revised_neighbor_link": revised_link_decision,
                "component_bridge": bridge.get("decision") or {},
                "final_neighbor_link": final_link_decision,
                "trace_phase": phase.get("decision") or {},
                "outcome_reconciliation": outcome.get("decision") or {},
            },
            compiled_trace_bundle=compiled_bundle,
            validator_issues=[
                *(prior_issues or []),
                *link.get("issues", []),
                *components.get("issues", []),
                *phase.get("issues", []),
                *outcome.get("issues", []),
                *compiler_issues,
            ],
            allowed_message_ids=list(
                ledger.get("allowed_message_ids") or []
            ),
            case_cards=cards,
        )
        return {
            "candidate_graph": graph,
            "neighbor_link": link,
            "component_consistency": consistency,
            "component_bridge": bridge,
            "trace_components": components,
            "trace_phase": phase,
            "outcome_reconciliation": outcome,
            "trace_compiler": compiler,
            "w6_trace_review_payload": trace_review_payload,
        }
