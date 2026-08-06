"""KG v2 review/apply boundary for dual-write compatibility."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.w7_trace.review import (
    approval_hash_matches,
)

from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent, _evidence_pack
from debug_agent_system.agents.write.non_sop_intake import (
    SOP_INCREMENTAL_CONTRACT,
    is_sop_source_reference,
)
from debug_agent_system.core.paths import project_root
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.document_links import (
    DOCUMENT_LINK_RELATIONS,
    build_document_link_graph,
)
from debug_agent_system.knowledge_v2.json_store import (
    DERIVED_TERMINOLOGY_RELATIONS,
    JsonKGV2Store,
)
from debug_agent_system.knowledge_v2.materializer import KGV2Materializer
from debug_agent_system.knowledge_v2.provenance import alignment_provenance_issues
from debug_agent_system.knowledge_v2.terminology import (
    write_terminology_layer,
)
from debug_agent_system.knowledge_v2.validator import validate_graph


DOCUMENT_LAYER_OBJECT_TYPES = {
    "KnowledgeDocument",
    "KnowledgeSection",
    "ProcedureStep",
    "EvidenceItem",
}


def build_v2_review_item(
    queue: str,
    bundle: dict[str, Any],
    episode: dict[str, Any],
    legacy_conflict: dict[str, Any],
    legacy_gate: dict[str, Any],
    dry_run_merge_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(bundle.get("candidate_id") or "")
    review_id = f"review:v2:{candidate_id.replace(':', '-')}" if candidate_id else "review:v2:unknown"
    logical = str(queue or "candidates").replace(".json", "")
    evidence_pack = _evidence_pack(episode)
    object_counts = {
        key: len([item for item in value or [] if isinstance(item, dict)])
        for key, value in (bundle.get("objects") or {}).items()
    }
    return {
        "review_id": review_id,
        "candidate_id": candidate_id,
        "dedupe_key": candidate_id,
        "queue": logical,
        "candidate": bundle,
        "episode": episode,
        "conflict": legacy_conflict,
        "quality_gate": {
            **legacy_gate,
            "kg_v2_schema_valid": bool(bundle.get("schema_valid")),
            "kg_v2_schema_issues": list(bundle.get("schema_issues") or []),
        },
        "dry_run_merge_plan": dry_run_merge_plan or {},
        "evidence_pack": evidence_pack,
        "review_summary": {
            "title": candidate_id,
            "legacy_candidate_id": bundle.get("legacy_candidate_id") or "",
            "family_id": bundle.get("family_id") or "",
            "variant_id": bundle.get("variant_id") or "",
            "schema_valid": bool(bundle.get("schema_valid")),
            "schema_issues": list(bundle.get("schema_issues") or []),
            "object_counts": object_counts,
            "relation_count": len(bundle.get("relations") or []),
            "queue_hint": logical,
        },
        "review_actions": ["approve", "reject", "merge_existing", "request_more_info"],
        "review_status": "pending",
        "kg_version": "v2",
        "admission_target": "fault_execution",
        "materialize_allowed": True,
        "observability": {"agent_id": "W6", "queue": logical, "candidate_id": candidate_id, "kg_version": "v2"},
    }


class IncrementalIngestV2Agent:
    """Approved-only ingest for KG v2 review items."""

    def __init__(self, store: JsonKGV2Store) -> None:
        self.store = store

    def dry_run_merge_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = _candidate(payload)
        objects = candidate.get("objects") if isinstance(candidate.get("objects"), dict) else {}
        relations = candidate.get("relations") if isinstance(candidate.get("relations"), list) else []
        issues = validate_graph(objects, relations)
        object_counts = {
            key: len([item for item in value or [] if isinstance(item, dict)])
            for key, value in objects.items()
        }
        duplicate = self._seen_candidate(candidate) or self._seen_dedupe_key(
            _dedupe_key(payload, candidate),
            content_hash=_content_hash(payload, candidate),
            candidate_id=str(candidate.get("candidate_id") or ""),
        )
        return {
            "status": "dry_run_merge_plan_v2",
            "candidate_id": candidate.get("candidate_id") or "",
            "intake_id": _intake_id(payload, candidate),
            "dedupe_key": _dedupe_key(payload, candidate),
            "family_id": candidate.get("family_id") or "",
            "variant_id": candidate.get("variant_id") or "",
            "object_counts": object_counts,
            "relation_count": len(relations),
            "schema_valid": not issues,
            "schema_issues": issues,
            "duplicate_candidate": duplicate,
            "admission_readiness": _admission_readiness(payload),
            "would_materialize_execution": not issues and _materialize_allowed(payload),
            "observability": {"agent_id": "W5", "mode": "dry_run_merge_plan_v2"},
        }

    def apply_approved(self, payload: dict[str, Any], *, materialize: bool = False) -> dict[str, Any]:
        if _is_typed_review_item(payload):
            return self.apply_approved_typed_review_item(payload, materialize=materialize)
        candidate = _candidate(payload)
        candidate_id = str(candidate.get("candidate_id") or "")
        if not _approved(payload):
            return {"status": "skipped", "reason": "not_approved", "candidate_id": candidate_id}
        if not _w4_admitted(payload, require_decision=False):
            return {"status": "skipped", "reason": "w4_not_admitted", "candidate_id": candidate_id}
        provenance_issues = alignment_provenance_issues(payload)
        if provenance_issues:
            return {
                "status": "skipped",
                "reason": "alignment_provenance_invalid",
                "provenance_issues": provenance_issues,
                "candidate_id": candidate_id,
            }
        issues = validate_graph(candidate.get("objects") or {}, candidate.get("relations") or [])
        if issues:
            return {"status": "skipped", "reason": "schema_invalid", "schema_issues": issues, "candidate_id": candidate_id}
        if self._seen_candidate(candidate):
            return {"status": "already_applied", "candidate_id": candidate_id}
        if _blocked_sop_source(payload, candidate):
            return {"status": "skipped", "reason": "sop_source_blocked", "candidate_id": candidate_id}
        before_hash = _graph_hash(self.store)
        before_objects = _object_fingerprint(self.store)
        merge = self.store.merge_graph(
            candidate.get("objects") or {},
            candidate.get("relations") or [],
            replace_document_sources=True,
        )
        if merge.get("status") == "schema_invalid":
            return {"status": "skipped", "reason": "schema_invalid", "schema_issues": merge.get("issues") or [], "candidate_id": candidate_id}
        self.store = JsonKGV2Store(self.store.root)
        document_link_refresh = (
            self._refresh_document_links()
            if (candidate.get("objects") or {}).get("KnowledgeDocument")
            else {}
        )
        terminology_refresh = self._refresh_terminology()
        after_hash = _graph_hash(self.store)
        materialized = self._recompute_and_materialize([payload]) if materialize and (
            _materialize_allowed(payload) or _policy_recompute_allowed(payload)
        ) else {}
        object_diff = _object_diff(before_objects, _object_fingerprint(self.store))
        graph_changed = before_hash != after_hash
        affected_object_types = sorted(object_diff)
        document_index_changed = bool({
            "KnowledgeDocument", "KnowledgeSection", "EvidenceItem", "SourceCase"
        }.intersection(affected_object_types))
        audit = self.store.read_review_queue("approved_applied.json")
        audit.append({
            "intake_id": _intake_id(payload, candidate),
            "mapping_version": _mapping_version(payload),
            "admission_target": _admission_target(payload),
            "admission_readiness": _admission_readiness(payload),
            "materialize_allowed": _materialize_allowed(payload),
            "candidate_id": candidate_id,
            "dedupe_key": candidate_id,
            "status": "applied",
            "candidate": candidate,
            "merge_result": merge,
            "graph_hash_before": before_hash,
            "graph_hash_after": after_hash,
            "object_diff": object_diff,
            "diff": object_diff,
            "graph_changed": graph_changed,
            "requires_sag_publish": graph_changed,
            "document_index_changed": document_index_changed,
            "affected_object_types": affected_object_types,
            "document_link_refresh": document_link_refresh,
            "terminology_refresh": terminology_refresh,
            "reviewer": _reviewer(payload),
            "rollback_anchor": before_hash,
            "applied_at": _now(),
            "materialized_counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
        })
        self.store.write_review_queue("approved_applied.json", audit)
        return {
            "status": "applied_to_graph_v2",
            "candidate_id": candidate_id,
            "merge_result": merge,
            "graph_hash_before": before_hash,
            "graph_hash_after": after_hash,
            "graph_changed": graph_changed,
            "requires_sag_publish": graph_changed,
            "document_index_changed": document_index_changed,
            "affected_object_types": affected_object_types,
            "document_link_refresh": document_link_refresh,
            "terminology_refresh": terminology_refresh,
            "materialized_counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
        }

    def apply_approved_typed_review_item(self, payload: dict[str, Any], *, materialize: bool = False) -> dict[str, Any]:
        candidate = _candidate(payload)
        dedupe_key = _dedupe_key(payload, candidate)
        content_hash = _content_hash(payload, candidate)
        if bool(payload.get("approval_hash_required")) and not approval_hash_matches(payload):
            return {
                "status": "skipped",
                "reason": "approval_content_hash_mismatch",
                "dedupe_key": dedupe_key,
            }
        correction_events = [
            item for item in payload.get("correction_events") or []
            if isinstance(item, dict)
        ]
        if correction_events:
            overlay = (
                payload.get("correction_overlay")
                if isinstance(payload.get("correction_overlay"), dict)
                else {}
            )
            if not bool(payload.get("correction_overlay_applied")):
                return {
                    "status": "skipped",
                    "reason": "correction_events_not_compiled",
                    "dedupe_key": dedupe_key,
                }
            expected_overlay_hash = str(
                overlay.get("effective_bundle_hash") or ""
            )
            applied_overlay_hash = str(
                payload.get("applied_correction_overlay_hash") or ""
            )
            if (
                not expected_overlay_hash
                or applied_overlay_hash != expected_overlay_hash
            ):
                return {
                    "status": "skipped",
                    "reason": "correction_overlay_hash_mismatch",
                    "dedupe_key": dedupe_key,
                }
        if not _approved(payload):
            return {"status": "skipped", "reason": "not_approved", "dedupe_key": dedupe_key}
        if not _w4_admitted(payload, require_decision=True):
            return {"status": "skipped", "reason": "w4_not_admitted", "dedupe_key": dedupe_key}
        if _admission_target(payload) == "fault_execution" and _admission_readiness(payload) == "not_ready":
            return {"status": "skipped", "reason": "admission_not_ready", "dedupe_key": dedupe_key}
        operation = _operation(payload, candidate)
        if operation != "merge_graph":
            return {"status": "skipped", "reason": "unsupported_operation", "operation": operation, "dedupe_key": dedupe_key}
        if _blocked_sop_source(payload, candidate):
            return {"status": "skipped", "reason": "sop_source_blocked", "dedupe_key": dedupe_key}
        objects = candidate.get("objects") if isinstance(candidate.get("objects"), dict) else {}
        relations = candidate.get("relations") if isinstance(candidate.get("relations"), list) else []
        provenance_issues = alignment_provenance_issues(payload)
        if provenance_issues:
            return {
                "status": "skipped",
                "reason": "alignment_provenance_invalid",
                "provenance_issues": provenance_issues,
                "dedupe_key": dedupe_key,
            }
        source_hash_issues = _reviewed_source_hash_issues(self.store, objects, payload)
        if source_hash_issues:
            reasons = {str(item.get("reason") or "") for item in source_hash_issues}
            reason = (
                "source_content_changed_since_review"
                if "source_content_changed_since_review" in reasons
                else "source_content_unavailable_at_apply"
            )
            return {
                "status": "skipped",
                "reason": reason,
                "source_hash_issues": source_hash_issues,
                "dedupe_key": dedupe_key,
            }
        dependency_issues = _fault_mapping_document_dependency_issues(
            self.store,
            objects,
            payload,
        )
        if dependency_issues:
            return {
                "status": "skipped",
                "reason": "fault_mapping_document_layer_not_approved",
                "dependency_issues": dependency_issues,
                "dedupe_key": dedupe_key,
            }
        replacement_mapping_issues = _document_replacement_mapping_issues(
            self.store,
            objects,
            payload,
        )
        if replacement_mapping_issues:
            return {
                "status": "skipped",
                "reason": "document_replacement_requires_mapping_approval",
                "dependency_issues": replacement_mapping_issues,
                "dedupe_key": dedupe_key,
            }
        if self._seen_dedupe_key(
            dedupe_key,
            content_hash=content_hash,
            candidate_id=str(candidate.get("candidate_id") or ""),
        ):
            return {"status": "already_applied", "dedupe_key": dedupe_key}
        issues = validate_graph(objects, relations)
        if issues:
            return {"status": "skipped", "reason": "schema_invalid", "schema_issues": issues, "dedupe_key": dedupe_key}

        before_hash = _graph_hash(self.store)
        before_objects = _object_fingerprint(self.store)
        merge_objects = _objects_for_approved_merge(self.store, objects, payload)
        if _review_scope(payload) == "fault_mapping":
            merge_objects = {
                object_type: [] if object_type in DOCUMENT_LAYER_OBJECT_TYPES else list(items or [])
                for object_type, items in merge_objects.items()
            }
        merge_relations = _relations_for_approved_merge(objects, relations, payload)
        merge = self.store.merge_graph(
            merge_objects,
            merge_relations,
            replace_document_sources=_review_scope(payload) != "fault_mapping",
        )
        if merge.get("status") == "schema_invalid":
            return {"status": "skipped", "reason": "schema_invalid", "schema_issues": merge.get("issues") or [], "dedupe_key": dedupe_key}
        self.store = JsonKGV2Store(self.store.root)
        document_link_refresh = (
            self._refresh_document_links()
            if merge_objects.get("KnowledgeDocument")
            else {}
        )
        terminology_refresh = self._refresh_terminology()
        after_hash = _graph_hash(self.store)
        materialized = self._recompute_and_materialize([payload]) if materialize and (
            _materialize_allowed(payload) or _policy_recompute_allowed(payload)
        ) else {}
        object_diff = _object_diff(before_objects, _object_fingerprint(self.store))
        graph_changed = before_hash != after_hash
        affected_object_types = sorted(object_diff)
        document_index_changed = bool({
            "KnowledgeDocument", "KnowledgeSection", "EvidenceItem", "SourceCase"
        }.intersection(affected_object_types))
        audit_item = {
            "intake_id": _intake_id(payload, candidate),
            "mapping_version": _mapping_version(payload),
            "admission_target": _admission_target(payload),
            "materialize_allowed": _materialize_allowed(payload),
            "dedupe_key": dedupe_key,
            "content_hash": content_hash,
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "status": "applied",
            "operation": operation,
            "review_item_type": str(payload.get("review_item_type") or payload.get("typed_review_item") or ""),
            "reviewer": _reviewer(payload),
            "rollback_anchor": before_hash,
            "graph_hash_before": before_hash,
            "graph_hash_after": after_hash,
            "object_diff": object_diff,
            "diff": object_diff,
            "graph_changed": graph_changed,
            "requires_sag_publish": graph_changed,
            "document_index_changed": document_index_changed,
            "affected_object_types": affected_object_types,
            "document_link_refresh": document_link_refresh,
            "terminology_refresh": terminology_refresh,
            "merge_result": merge,
            "component_dedupe_keys": list(payload.get("component_dedupe_keys") or []),
            "component_content_hashes": dict(payload.get("component_content_hashes") or {}),
            "materialized": bool(materialized),
            "materialized_counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
            "applied_at": _now(),
        }
        audit = self.store.read_review_queue("approved_applied.json")
        audit.append(audit_item)
        self.store.write_review_queue("approved_applied.json", audit)
        return {
            "status": "applied_to_graph_v2",
            "dedupe_key": dedupe_key,
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "merge_result": merge,
            "graph_hash_before": before_hash,
            "graph_hash_after": after_hash,
            "graph_changed": graph_changed,
            "requires_sag_publish": graph_changed,
            "document_index_changed": document_index_changed,
            "affected_object_types": affected_object_types,
            "document_link_refresh": document_link_refresh,
            "terminology_refresh": terminology_refresh,
            "component_dedupe_keys": audit_item["component_dedupe_keys"],
            "materialized_counts": audit_item["materialized_counts"],
        }

    def apply_approved_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        materialize_items: list[dict[str, Any]] = []
        for item in items:
            result = self.apply_approved(item, materialize=False)
            results.append(result)
            if result.get("status") == "applied_to_graph_v2" and (
                _materialize_allowed(item) or _policy_recompute_allowed(item)
            ):
                materialize_items.append(item)
        if materialize_items:
            materialized = self._recompute_and_materialize(materialize_items)
            counts = {key: len(value) for key, value in materialized.items() if isinstance(value, list)}
            results.append({
                "status": "materialized_execution_v2",
                "applied_count": len(materialize_items),
                "materialized_counts": counts,
            })
        return results

    def apply_approved_review_queue(self, review_agent: ReviewQueueAgent) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in review_agent.read_queue("v2_typed_candidates.json"):
            if not isinstance(item, dict) or not _approved(item):
                continue
            items.append(item)
        prepared_items = _prepare_atomic_document_mapping_items(self.store, items)
        results = self.apply_approved_batch(prepared_items)
        invalidation_reasons = {
            "source_content_changed_since_review",
            "source_content_unavailable_at_apply",
        }
        for item, result in zip(prepared_items, results):
            reason = str(result.get("reason") or "")
            if reason not in invalidation_reasons:
                continue
            invalidation_keys = list(item.get("component_dedupe_keys") or [])
            if not invalidation_keys:
                invalidation_keys = [_dedupe_key(item, _candidate(item))]
            for invalidation_key in invalidation_keys:
                review_agent.mark_needs_re_review(
                    "v2_typed_candidates",
                    str(invalidation_key),
                    reason=reason,
                    details=result.get("source_hash_issues") or [],
                )
        return results

    def _seen_candidate(self, candidate: dict[str, Any]) -> bool:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return False
        for row in self.store.read_review_queue("approved_applied.json"):
            nested = row.get("candidate") if isinstance(row, dict) and isinstance(row.get("candidate"), dict) else row
            if isinstance(nested, dict) and str(nested.get("candidate_id") or "") == candidate_id:
                return True
        return False

    def _seen_dedupe_key(
        self,
        dedupe_key: str,
        *,
        content_hash: str = "",
        candidate_id: str = "",
    ) -> bool:
        if not dedupe_key:
            return False
        for row in self.store.read_review_queue("approved_applied.json"):
            if not isinstance(row, dict) or str(row.get("dedupe_key") or "") != dedupe_key:
                component_keys = {
                    str(value)
                    for value in row.get("component_dedupe_keys") or []
                    if str(value)
                }
                if dedupe_key not in component_keys:
                    continue
                component_hashes = (
                    row.get("component_content_hashes")
                    if isinstance(row.get("component_content_hashes"), dict)
                    else {}
                )
                applied_component_hash = str(component_hashes.get(dedupe_key) or "")
                if content_hash and applied_component_hash:
                    if applied_component_hash == content_hash:
                        return True
                    continue
                return True
            if not content_hash:
                return True
            applied_hash = str(row.get("content_hash") or "")
            if applied_hash:
                if applied_hash == content_hash:
                    return True
                continue
            # Backward compatibility for audit rows written before content_hash
            # was recorded: a changed content-addressed candidate id is a new
            # version, while an identical id is still a replay.
            applied_candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id or not applied_candidate_id or applied_candidate_id == candidate_id:
                return True
        return False

    def _refresh_document_links(self) -> dict[str, Any]:
        """Re-resolve cross-document relations after a document-layer merge."""

        objects = {
            object_type: [dict(item) for item in values]
            for object_type, values in self.store.objects_by_type.items()
        }
        link_relations, report = build_document_link_graph(
            project_root(__file__),
            objects.get("KnowledgeDocument") or [],
        )
        relations = [
            dict(item)
            for item in self.store.relations
            if isinstance(item, dict)
            and str(item.get("relation") or "") not in DOCUMENT_LINK_RELATIONS
        ]
        relations.extend(link_relations)
        result = self.store.replace_graph(objects, relations, validate=True)
        if result.get("status") != "replaced":
            raise ValueError(
                "kg_v2_document_link_refresh_failed:"
                + json.dumps(result, ensure_ascii=False, sort_keys=True)
            )
        self.store = JsonKGV2Store(self.store.root)
        return {
            "status": "refreshed",
            "resolved_relation_count": report.get("resolved_relation_count", 0),
            "child_relation_count": report.get("child_relation_count", 0),
            "reference_relation_count": report.get("reference_relation_count", 0),
            "unresolved_count": report.get("unresolved_count", 0),
        }

    def _refresh_terminology(self) -> dict[str, Any]:
        """Rebuild the derived terminology layer after an approved merge."""

        manifest = write_terminology_layer(self.store.root)
        self.store = JsonKGV2Store(self.store.root)
        return {
            "status": "refreshed",
            "terminology_version": manifest.get("terminology_version"),
            "revision": manifest.get("revision"),
            "concept_count": manifest.get("concept_count", 0),
            "expression_count": manifest.get("expression_count", 0),
            "sense_count": manifest.get("sense_count", 0),
            "ambiguous_expression_count": manifest.get(
                "ambiguous_expression_count",
                0,
            ),
        }

    def _recompute_and_materialize(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {}
        self.store = JsonKGV2Store(self.store.root)
        policies = KGV2Materializer(self.store).build_policy_objects()
        objects = {
            object_type: list(values)
            for object_type, values in self.store.objects_by_type.items()
        }
        old_policy_ids = {
            str(item.get("policy_id") or "")
            for item in objects.get("DecisionPolicy") or []
            if isinstance(item, dict)
        }
        objects["DecisionPolicy"] = policies
        relations = [
            dict(item)
            for item in self.store.relations
            if isinstance(item, dict)
            and str(item.get("from") or "") not in old_policy_ids
            and str(item.get("to") or "") not in old_policy_ids
            and str(item.get("relation") or "") != "for_family"
        ]
        relations.extend(
            {
                "from": str(policy.get("policy_id") or ""),
                "to": str(policy.get("family_id") or ""),
                "relation": "for_family",
            }
            for policy in policies
        )
        result = self.store.replace_graph(objects, relations, validate=True)
        if result.get("status") != "replaced":
            raise ValueError(
                "kg_v2_policy_recompute_failed:"
                + json.dumps(result, ensure_ascii=False, sort_keys=True)
            )
        self.store = JsonKGV2Store(self.store.root)
        return KGV2Materializer(self.store).materialize(self.store.materialized_root)


def _candidate(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("candidate")
    if isinstance(nested, dict):
        return nested
    typed = payload.get("typed_candidate")
    if isinstance(typed, dict):
        return _graph_candidate(typed, payload)
    inner = payload.get("payload")
    if isinstance(inner, dict):
        return _graph_candidate(inner, payload)
    graph = payload.get("graph")
    if isinstance(graph, dict):
        return _graph_candidate(graph, payload)
    return _graph_candidate(payload, payload)


def _graph_candidate(source: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    inner = source.get("payload") if isinstance(source.get("payload"), dict) else {}
    if isinstance(source.get("graph"), dict):
        graph = source["graph"]
    elif isinstance(inner.get("graph"), dict):
        graph = inner["graph"]
    elif isinstance(inner.get("objects"), dict) or isinstance(inner.get("relations"), list):
        graph = inner
    else:
        graph = source
    candidate = dict(graph)
    if not isinstance(candidate.get("objects"), dict) and isinstance(source.get("objects"), dict):
        candidate["objects"] = source["objects"]
    if not isinstance(candidate.get("objects"), dict) and isinstance(inner.get("objects"), dict):
        candidate["objects"] = inner["objects"]
    if not isinstance(candidate.get("relations"), list) and isinstance(source.get("relations"), list):
        candidate["relations"] = source["relations"]
    if not isinstance(candidate.get("relations"), list) and isinstance(inner.get("relations"), list):
        candidate["relations"] = inner["relations"]
    for key in ("candidate_id", "dedupe_key", "intake_id", "family_id", "variant_id"):
        if not candidate.get(key):
            candidate[key] = _first_value(envelope, source, key)
    return candidate


def _review_scope(payload: dict[str, Any]) -> str:
    typed = payload.get("typed_candidate") if isinstance(payload.get("typed_candidate"), dict) else {}
    inner = typed.get("payload") if isinstance(typed.get("payload"), dict) else {}
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    return str(
        payload.get("review_scope")
        or typed.get("review_scope")
        or inner.get("review_scope")
        or candidate.get("review_scope")
        or ""
    )


def _reviewed_source_hash_issues(
    store: JsonKGV2Store,
    objects: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pin a document approval to the exact source bytes reviewed by W6."""

    if _review_scope(payload) not in {
        "document_layer",
        "fault_mapping",
        "document_mapping_pair",
    }:
        return []
    issues: list[dict[str, Any]] = []
    for document in objects.get("KnowledgeDocument") or []:
        if not isinstance(document, dict):
            continue
        document_id = str(document.get("document_id") or "")
        source_path = str(document.get("source_path") or "")
        expected_hash = str(document.get("content_hash") or "")
        if not source_path or not expected_hash:
            issues.append({
                "reason": "source_content_unavailable_at_apply",
                "document_id": document_id,
                "source_path": source_path,
                "expected_hash": expected_hash,
            })
            continue
        path = _resolve_reviewed_source_path(store, source_path)
        if path is None:
            issues.append({
                "reason": "source_content_unavailable_at_apply",
                "document_id": document_id,
                "source_path": source_path,
                "expected_hash": expected_hash,
            })
            continue
        try:
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            issues.append({
                "reason": "source_content_unavailable_at_apply",
                "document_id": document_id,
                "source_path": source_path,
                "resolved_path": str(path),
                "expected_hash": expected_hash,
                "error": type(exc).__name__,
            })
            continue
        if current_hash != expected_hash:
            issues.append({
                "reason": "source_content_changed_since_review",
                "document_id": document_id,
                "source_path": source_path,
                "resolved_path": str(path),
                "expected_hash": expected_hash,
                "current_hash": current_hash,
            })
    return issues


def _resolve_reviewed_source_path(store: JsonKGV2Store, source_path: str) -> Path | None:
    path = Path(source_path)
    candidates = [path] if path.is_absolute() else [
        path,
        store.root.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _fault_mapping_document_dependency_issues(
    store: JsonKGV2Store,
    objects: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require the exact document layer before a mapping can reference it."""

    if _review_scope(payload) != "fault_mapping":
        return []
    documents = [
        item
        for item in objects.get("KnowledgeDocument") or []
        if isinstance(item, dict)
    ]
    if not documents:
        typed = (
            payload.get("typed_candidate")
            if isinstance(payload.get("typed_candidate"), dict)
            else {}
        )
        inner = (
            typed.get("payload")
            if isinstance(typed.get("payload"), dict)
            else {}
        )
        source_ref = (
            typed.get("source_ref")
            if isinstance(typed.get("source_ref"), dict)
            else {}
        )
        evidence_pack = (
            typed.get("evidence_pack")
            if isinstance(typed.get("evidence_pack"), dict)
            else {}
        )
        manifest_ref = (
            evidence_pack.get("chunk_manifest_ref")
            if isinstance(evidence_pack.get("chunk_manifest_ref"), dict)
            else {}
        )
        atomic_case = (
            inner.get("atomic_case")
            if isinstance(inner.get("atomic_case"), dict)
            else {}
        )
        source_path = str(
            source_ref.get("path") or manifest_ref.get("source_path") or ""
        )
        source_hash = str(manifest_ref.get("source_file_hash") or "")
        source_section_id = str(atomic_case.get("section_id") or "")
        matching_documents = [
            item
            for item in store.objects_by_type.get("KnowledgeDocument") or []
            if isinstance(item, dict)
            and str(item.get("source_path") or "") == source_path
            and str(item.get("content_hash") or "") == source_hash
            and item.get("approved") is True
        ]
        if matching_documents:
            document_ids = {
                str(item.get("document_id") or "") for item in matching_documents
            }
            if not source_section_id or any(
                str(section.get("document_id") or "") in document_ids
                and source_section_id in {
                    str(value) for value in section.get("source_offsets") or []
                }
                for section in store.objects_by_type.get("KnowledgeSection") or []
                if isinstance(section, dict)
            ):
                return []
            return [{
                "reason": "approved_document_section_missing",
                "source_path": source_path,
                "source_file_hash": source_hash,
                "source_section_id": source_section_id,
            }]
        return [{
            "reason": "missing_document_reference",
            "source_path": source_path,
            "source_file_hash": source_hash,
        }]
    issues: list[dict[str, Any]] = []
    for object_type in DOCUMENT_LAYER_OBJECT_TYPES:
        pk = V2_PRIMARY_KEYS[object_type]
        existing = store.object_index(object_type)
        for item in objects.get(object_type) or []:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get(pk) or "")
            current = existing.get(object_id)
            if not object_id or not isinstance(current, dict):
                issues.append({
                    "reason": "document_layer_object_missing",
                    "object_type": object_type,
                    "object_id": object_id,
                })
                continue
            if object_type == "KnowledgeDocument":
                if current.get("approved") is not True:
                    issues.append({
                        "reason": "document_not_approved",
                        "object_type": object_type,
                        "object_id": object_id,
                    })
                expected_hash = str(item.get("content_hash") or "")
                current_hash = str(current.get("content_hash") or "")
                if not expected_hash or current_hash != expected_hash:
                    issues.append({
                        "reason": "document_version_mismatch",
                        "object_type": object_type,
                        "object_id": object_id,
                        "expected_hash": expected_hash,
                        "current_hash": current_hash,
                    })
    return issues


def _document_replacement_mapping_issues(
    store: JsonKGV2Store,
    objects: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect replacements that would temporarily orphan approved mappings."""

    if _review_scope(payload) != "document_layer":
        return []
    incoming_documents = [
        item
        for item in objects.get("KnowledgeDocument") or []
        if isinstance(item, dict) and str(item.get("source_path") or "")
    ]
    issues: list[dict[str, Any]] = []
    for incoming in incoming_documents:
        source_path = str(incoming.get("source_path") or "")
        incoming_id = str(incoming.get("document_id") or "")
        old_document_ids = {
            str(item.get("document_id") or "")
            for item in store.objects_by_type.get("KnowledgeDocument") or []
            if isinstance(item, dict)
            and str(item.get("source_path") or "") == source_path
            and str(item.get("document_id") or "") != incoming_id
        }
        if not old_document_ids:
            continue
        old_section_ids = {
            str(item.get("section_id") or "")
            for item in store.objects_by_type.get("KnowledgeSection") or []
            if isinstance(item, dict)
            and str(item.get("document_id") or "") in old_document_ids
        }
        old_step_ids = {
            str(item.get("procedure_step_id") or "")
            for item in store.objects_by_type.get("ProcedureStep") or []
            if isinstance(item, dict)
            and str(item.get("section_id") or "") in old_section_ids
        }
        old_evidence_ids = {
            str(relation.get("from") or "")
            for relation in store.relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "evidences"
            and str(relation.get("to") or "") in old_section_ids | old_step_ids
        }
        mapping_relations = [
            relation
            for relation in store.relations
            if isinstance(relation, dict)
            and (
                (
                    str(relation.get("from") or "") in old_section_ids
                    and str(relation.get("relation") or "")
                    in {"applicable_to", "describes_variant"}
                )
                or (
                    str(relation.get("from") or "") in old_step_ids
                    and str(relation.get("relation") or "") == "candidate_action"
                )
                or (
                    str(relation.get("from") or "") in old_evidence_ids
                    and str(relation.get("relation") or "") == "evidences"
                    and str(relation.get("to") or "") not in old_section_ids | old_step_ids
                )
            )
        ]
        if mapping_relations:
            issues.append({
                "reason": "existing_document_has_fault_mapping",
                "source_path": source_path,
                "incoming_document_id": incoming_id,
                "old_document_ids": sorted(old_document_ids),
                "mapping_relation_count": len(mapping_relations),
            })
    return issues


def _prepare_atomic_document_mapping_items(
    store: JsonKGV2Store,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair separately approved document/mapping revisions for one graph commit."""

    mapping_by_document: dict[str, tuple[int, dict[str, Any]]] = {}
    mapping_by_source: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for index, item in enumerate(items):
        if _review_scope(item) != "fault_mapping":
            continue
        candidate = _candidate(item)
        objects = candidate.get("objects") if isinstance(candidate.get("objects"), dict) else {}
        for document in objects.get("KnowledgeDocument") or []:
            if isinstance(document, dict) and str(document.get("document_id") or ""):
                mapping_by_document[str(document["document_id"])] = (index, item)
        source_identity = _review_source_identity(item)
        if all(source_identity) and source_identity not in mapping_by_source:
            mapping_by_source[source_identity] = (index, item)

    pair_by_document_index: dict[int, tuple[int, dict[str, Any]]] = {}
    paired_mapping_indexes: set[int] = set()
    for index, item in enumerate(items):
        if _review_scope(item) != "document_layer":
            continue
        candidate = _candidate(item)
        objects = candidate.get("objects") if isinstance(candidate.get("objects"), dict) else {}
        if not _document_replacement_mapping_issues(store, objects, item):
            continue
        document_ids = [
            str(document.get("document_id") or "")
            for document in objects.get("KnowledgeDocument") or []
            if isinstance(document, dict) and str(document.get("document_id") or "")
        ]
        pair = next(
            (
                mapping_by_document[document_id]
                for document_id in document_ids
                if document_id in mapping_by_document
                and mapping_by_document[document_id][0] not in paired_mapping_indexes
            ),
            None,
        )
        if pair is None:
            source_identity = _review_source_identity(item)
            candidate_pair = mapping_by_source.get(source_identity)
            if (
                candidate_pair is not None
                and candidate_pair[0] not in paired_mapping_indexes
            ):
                pair = candidate_pair
        if pair is None:
            continue
        mapping_index, mapping_item = pair
        pair_by_document_index[index] = (mapping_index, mapping_item)
        paired_mapping_indexes.add(mapping_index)

    prepared: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if index in paired_mapping_indexes:
            continue
        pair = pair_by_document_index.get(index)
        if pair is not None:
            _mapping_index, mapping_item = pair
            prepared.append(_combine_document_mapping_reviews(item, mapping_item))
        else:
            prepared.append(item)
    return prepared


def _combine_document_mapping_reviews(
    document_item: dict[str, Any],
    mapping_item: dict[str, Any],
) -> dict[str, Any]:
    combined = deepcopy(mapping_item)
    document_candidate = _candidate(document_item)
    mapping_candidate = _candidate(mapping_item)
    merged_objects: dict[str, list[dict[str, Any]]] = {}
    for object_type in V2_PRIMARY_KEYS:
        pk = V2_PRIMARY_KEYS[object_type]
        seen: set[str] = set()
        values: list[dict[str, Any]] = []
        for source in (document_candidate, mapping_candidate):
            source_objects = (
                source.get("objects")
                if isinstance(source.get("objects"), dict)
                else {}
            )
            for value in source_objects.get(object_type) or []:
                if not isinstance(value, dict):
                    continue
                object_id = str(value.get(pk) or "")
                if not object_id or object_id in seen:
                    continue
                seen.add(object_id)
                values.append(deepcopy(value))
        merged_objects[object_type] = values
    relation_keys: set[tuple[str, str, str]] = set()
    merged_relations: list[dict[str, Any]] = []
    for source in (document_candidate, mapping_candidate):
        for relation in source.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            key = (
                str(relation.get("from") or ""),
                str(relation.get("to") or ""),
                str(relation.get("relation") or ""),
            )
            if not all(key) or key in relation_keys:
                continue
            relation_keys.add(key)
            merged_relations.append(deepcopy(relation))
    typed_candidate = (
        combined.get("typed_candidate")
        if isinstance(combined.get("typed_candidate"), dict)
        else {}
    )
    typed_payload = (
        typed_candidate.get("payload")
        if isinstance(typed_candidate.get("payload"), dict)
        else {}
    )
    typed_payload.update({
        "objects": merged_objects,
        "relations": merged_relations,
        "schema_valid": True,
        "schema_issues": [],
        "review_scope": "document_mapping_pair",
    })
    typed_candidate["payload"] = typed_payload
    typed_candidate["review_scope"] = "document_mapping_pair"
    combined["typed_candidate"] = typed_candidate
    component_items = [document_item, mapping_item]
    component_dedupe_keys = [
        _dedupe_key(item, _candidate(item))
        for item in component_items
    ]
    component_content_hashes = {
        dedupe_key: _content_hash(item, _candidate(item))
        for dedupe_key, item in zip(component_dedupe_keys, component_items)
    }
    identity = "|".join(component_dedupe_keys)
    content_identity = "|".join(
        component_content_hashes[key]
        for key in component_dedupe_keys
    )
    combined.update({
        "review_scope": "document_mapping_pair",
        "dedupe_key": "dedupe:document-mapping:"
        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        "content_hash": "content:"
        + hashlib.sha256(content_identity.encode("utf-8")).hexdigest()[:24],
        "candidate_id": "document-mapping-pair:"
        + hashlib.sha256(f"{identity}|{content_identity}".encode("utf-8")).hexdigest()[:24],
        "review_id": "review:document-mapping-pair:"
        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        "review_status": "approved",
        "human_approved": True,
        "selected_action": "approve_support_only",
        "component_dedupe_keys": component_dedupe_keys,
        "component_content_hashes": component_content_hashes,
        "component_candidate_ids": [
            str(item.get("candidate_id") or _candidate(item).get("candidate_id") or "")
            for item in component_items
        ],
    })
    return combined


def _review_source_identity(item: dict[str, Any]) -> tuple[str, str]:
    typed = (
        item.get("typed_candidate")
        if isinstance(item.get("typed_candidate"), dict)
        else {}
    )
    source_ref = (
        typed.get("source_ref")
        if isinstance(typed.get("source_ref"), dict)
        else {}
    )
    evidence_pack = (
        typed.get("evidence_pack")
        if isinstance(typed.get("evidence_pack"), dict)
        else {}
    )
    manifest_ref = (
        evidence_pack.get("chunk_manifest_ref")
        if isinstance(evidence_pack.get("chunk_manifest_ref"), dict)
        else {}
    )
    return (
        str(source_ref.get("path") or manifest_ref.get("source_path") or ""),
        str(manifest_ref.get("source_file_hash") or ""),
    )


def _approved(payload: dict[str, Any]) -> bool:
    return bool(payload.get("human_approved")) or str(payload.get("review_status") or payload.get("status") or "") in {"approved", "human_approved", "accepted"} or str(payload.get("selected_action") or "") in {"approve", "accept", "merge"}


def _is_typed_review_item(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("review_item_type", "typed_review_item", "typed_candidate", "operation", "apply_op", "dedupe_key", "intake_id", "admission_target", "mapping_version", "payload"))


def _operation(payload: dict[str, Any], candidate: dict[str, Any]) -> str:
    return str(
        payload.get("operation")
        or payload.get("apply_op")
        or candidate.get("operation")
        or candidate.get("apply_op")
        or "merge_graph"
    )


def _dedupe_key(payload: dict[str, Any], candidate: dict[str, Any]) -> str:
    return str(_first_value(payload, candidate, "dedupe_key") or _intake_id(payload, candidate) or candidate.get("candidate_id") or "")


def _content_hash(payload: dict[str, Any], candidate: dict[str, Any]) -> str:
    typed = payload.get("typed_candidate") if isinstance(payload.get("typed_candidate"), dict) else {}
    return str(
        payload.get("content_hash")
        or typed.get("content_hash")
        or candidate.get("content_hash")
        or ""
    )


def _intake_id(payload: dict[str, Any], candidate: dict[str, Any] | None = None) -> str:
    return str(_first_value(payload, candidate or {}, "intake_id") or "")


def _reviewer(payload: dict[str, Any]) -> str:
    review_decision = payload.get("review_decision")
    if isinstance(review_decision, dict) and review_decision.get("reviewer"):
        return str(review_decision.get("reviewer") or "")
    review = payload.get("review")
    if isinstance(review, dict) and review.get("reviewer"):
        return str(review.get("reviewer") or "")
    return str(payload.get("reviewer") or payload.get("approved_by") or "")


def _materialize_allowed(payload: dict[str, Any]) -> bool:
    quality_gate = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    selected_action = str(payload.get("selected_action") or "")
    if selected_action == "approve_support_only":
        return False
    allowed = payload.get("materialize_allowed")
    if allowed is None:
        allowed = quality_gate.get("materialize_allowed")
    if selected_action == "approve_for_execution_policy":
        # ``route_review`` is W4's request for a human decision, not a
        # permanent execution-view veto.  W6 may explicitly release an
        # execution-ready fault candidate after reviewing the semantic/risk
        # issues.  A plain ``approve`` still cannot do this, and W4 ``reject``
        # remains a hard stop in both this function and ``_w4_admitted``.
        return (
            bool(payload.get("human_approved"))
            and str(quality_gate.get("decision") or "") in {"admit", "route_review"}
            and _admission_target(payload) == "fault_execution"
            and _admission_readiness(payload) == "execution_ready"
        )
    return (
        _admission_target(payload) == "fault_execution"
        and _admission_readiness(payload) == "execution_ready"
        and allowed is True
    )


def _policy_recompute_allowed(payload: dict[str, Any]) -> bool:
    if str(payload.get("selected_action") or "") == "approve_support_only":
        return False
    quality_gate = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    decision = str(quality_gate.get("decision") or payload.get("w4_decision") or payload.get("decision") or "")
    if decision != "admit":
        return False
    if _admission_readiness(payload) != "execution_ready":
        return False
    candidate = _candidate(payload)
    objects = candidate.get("objects") if isinstance(candidate.get("objects"), dict) else {}
    return bool(objects.get("ActionOutcome") or objects.get("DiagnosticTrace"))


def _admission_target(payload: dict[str, Any]) -> str:
    quality_gate = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    dry_run = payload.get("dry_run_plan") if isinstance(payload.get("dry_run_plan"), dict) else {}
    typed = payload.get("typed_candidate") if isinstance(payload.get("typed_candidate"), dict) else {}
    return str(
        payload.get("admission_target")
        or quality_gate.get("admission_target")
        or dry_run.get("admission_target")
        or _first_value(typed, "admission_target")
        or ""
    )


def _mapping_version(payload: dict[str, Any]) -> str:
    quality_gate = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    typed = payload.get("typed_candidate") if isinstance(payload.get("typed_candidate"), dict) else {}
    return str(payload.get("mapping_version") or quality_gate.get("mapping_version") or _first_value(typed, "mapping_version") or "")


def _w4_admitted(payload: dict[str, Any], *, require_decision: bool) -> bool:
    quality_gate = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    decision = str(quality_gate.get("decision") or payload.get("w4_decision") or payload.get("decision") or "")
    if not decision:
        return not require_decision
    # ``route_review`` exists specifically to let W6 supply the missing human
    # judgement.  It may be merged after explicit approval, but W4 keeps
    # materialize_allowed=False so it cannot alter the read-side execution view
    # until corrected and re-gated.  A hard reject is never merge-eligible.
    return decision in {"admit", "route_review"}


def _blocked_sop_source(payload: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if _contains_forbidden_sop_build_path(payload) or (
        _contains_forbidden_sop_build_path(candidate)
    ):
        return True
    if _sop_incremental_source_allowed(payload):
        return False
    return _contains_sop_marker(payload) or _contains_sop_marker(candidate)


def _sop_incremental_source_allowed(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
    if (
        str(value.get("source_type") or "").strip().lower() == "sop_doc"
        and str(value.get("source_kind") or "").strip().lower() == "sop"
        and str(metadata.get("incremental_source_contract") or "")
        == SOP_INCREMENTAL_CONTRACT
    ):
        return True
    return any(
        _sop_incremental_source_allowed(value.get(key))
        for key in ("typed_candidate", "payload", "candidate", "graph")
    )


def _contains_forbidden_sop_build_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_forbidden_sop_build_path(child)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_sop_build_path(item) for item in value)
    return "data/kg_v2_sop_draft_build" in str(value or "").replace(
        "\\", "/"
    ).lower()


def _contains_sop_marker(value: Any, key: str = "") -> bool:
    if isinstance(value, dict):
        return any(_contains_sop_marker(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, list):
        return any(_contains_sop_marker(item, key) for item in value)
    text = str(value or "")
    lowered = text.lower()
    if key.lower() in {"source_type", "source_kind"} and lowered == "sop":
        return True
    if key.lower() in {"source_ref", "source_path", "path", "payload_ref"} and is_sop_source_reference(text):
        return True
    normalized = text.replace("\\", "/").lower()
    return "data/kg_v2_sop_draft_build" in normalized


def _objects_for_approved_merge(
    store: JsonKGV2Store,
    objects: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    allowed = _materialize_allowed(payload)
    admission_target = _admission_target(payload)
    readiness = _admission_readiness(payload)
    readiness_types = {
        "evidence_ready": {"SourceCase", "EvidenceItem"},
        "case_ready": {"FaultFamily", "FaultVariant", "RequiredInfoSpec", "SourceCase", "EvidenceItem"},
        "execution_ready": set(V2_PRIMARY_KEYS),
    }
    permitted_types = readiness_types.get(readiness, set(V2_PRIMARY_KEYS))
    replacement_owned_ids = (
        _replacement_source_owned_object_ids(store, objects)
        if _review_scope(payload) == "document_mapping_pair"
        else {}
    )
    annotated: dict[str, list[dict[str, Any]]] = {}
    for obj_type, items in objects.items():
        annotated[obj_type] = []
        if admission_target == "fault_execution" and obj_type not in permitted_types:
            continue
        pk = V2_PRIMARY_KEYS.get(obj_type, "id")
        existing = store.object_index(obj_type) if obj_type in V2_PRIMARY_KEYS else {}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            obj_id = str(item.get(pk) or "")
            if not allowed and obj_id and obj_id in existing:
                if obj_id in replacement_owned_ids.get(obj_type, set()):
                    pass
                elif obj_type == "KnowledgeDocument":
                    annotated[obj_type].append({
                        pk: obj_id,
                        "approved": True,
                        "_admission_target": admission_target,
                        "execution_materialize_allowed": False,
                    })
                    continue
                else:
                    continue
            annotated[obj_type].append({
                **item,
                **({"approved": True} if obj_type in {"KnowledgeDocument", "SourceCase"} else {}),
                "_admission_target": admission_target,
                "execution_materialize_allowed": allowed,
            })
    return annotated


def _replacement_source_owned_object_ids(
    store: JsonKGV2Store,
    incoming_objects: dict[str, list[dict[str, Any]]],
) -> dict[str, set[str]]:
    """Identify old raw-document mapping objects that graph replacement removes."""

    source_paths = {
        str(item.get("source_path") or "")
        for item in incoming_objects.get("KnowledgeDocument") or []
        if isinstance(item, dict) and str(item.get("source_path") or "")
    }
    old_document_ids = {
        str(item.get("document_id") or "")
        for item in store.objects_by_type.get("KnowledgeDocument") or []
        if isinstance(item, dict)
        and str(item.get("source_path") or "") in source_paths
    }
    old_section_ids = {
        str(item.get("section_id") or "")
        for item in store.objects_by_type.get("KnowledgeSection") or []
        if isinstance(item, dict)
        and str(item.get("document_id") or "") in old_document_ids
    }
    old_step_ids = {
        str(item.get("procedure_step_id") or "")
        for item in store.objects_by_type.get("ProcedureStep") or []
        if isinstance(item, dict)
        and str(item.get("section_id") or "") in old_section_ids
    }
    old_evidence_ids = {
        str(relation.get("from") or "")
        for relation in store.relations
        if isinstance(relation, dict)
        and str(relation.get("relation") or "") == "evidences"
        and str(relation.get("to") or "") in old_section_ids | old_step_ids
    }
    source_layer_ids = (
        old_document_ids
        | old_section_ids
        | old_step_ids
        | old_evidence_ids
    )
    candidate_action_ids = {
        str(relation.get("to") or "")
        for relation in store.relations
        if isinstance(relation, dict)
        and str(relation.get("relation") or "") == "candidate_action"
        and str(relation.get("from") or "") in old_step_ids
    }
    action_ids = {
        str(item.get("action_id") or "")
        for item in store.objects_by_type.get("DiagnosticAction") or []
        if isinstance(item, dict)
        and str(item.get("action_id") or "") in candidate_action_ids
        and str(item.get("source_kind") or "") == "raw_doc"
        and not _has_external_relation(
            str(item.get("action_id") or ""),
            store.relations,
            source_layer_ids,
        )
    }
    candidate_required_info_ids = {
        str(relation.get("to") or "")
        for relation in store.relations
        if isinstance(relation, dict)
        and str(relation.get("relation") or "") == "evidences"
        and str(relation.get("from") or "") in old_evidence_ids
    }
    required_info_ids = {
        str(item.get("required_info_id") or "")
        for item in store.objects_by_type.get("RequiredInfoSpec") or []
        if isinstance(item, dict)
        and str(item.get("required_info_id") or "") in candidate_required_info_ids
        and {
            str(value)
            for value in item.get("evidence_ids") or []
            if str(value)
        }.issubset(old_evidence_ids)
        and not any(
            isinstance(relation, dict)
            and str(relation.get("relation") or "") == "evidences"
            and str(relation.get("to") or "")
            == str(item.get("required_info_id") or "")
            and str(relation.get("from") or "") not in old_evidence_ids
            for relation in store.relations
        )
    }
    return {
        "DiagnosticAction": action_ids,
        "RequiredInfoSpec": required_info_ids,
    }


def _has_external_relation(
    object_id: str,
    relations: list[dict[str, Any]],
    source_layer_ids: set[str],
) -> bool:
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if (
            str(relation.get("relation") or "")
            in DERIVED_TERMINOLOGY_RELATIONS
        ):
            continue
        source = str(relation.get("from") or "")
        target = str(relation.get("to") or "")
        if source == object_id and target not in source_layer_ids:
            return True
        if target == object_id and source not in source_layer_ids:
            return True
    return False


def _relations_for_approved_merge(
    objects: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if _admission_target(payload) != "fault_execution":
        return list(relations)
    readiness = _admission_readiness(payload)
    readiness_types = {
        "evidence_ready": {"SourceCase", "EvidenceItem"},
        "case_ready": {"FaultFamily", "FaultVariant", "RequiredInfoSpec", "SourceCase", "EvidenceItem"},
        "execution_ready": set(V2_PRIMARY_KEYS),
    }
    permitted_types = readiness_types.get(readiness, set())
    disallowed_ids = {
        str(item.get(V2_PRIMARY_KEYS.get(obj_type, "id")) or "")
        for obj_type, items in objects.items()
        if obj_type not in permitted_types
        for item in items or []
        if isinstance(item, dict) and str(item.get(V2_PRIMARY_KEYS.get(obj_type, "id")) or "")
    }
    return [
        dict(relation)
        for relation in relations
        if isinstance(relation, dict)
        and str(relation.get("from") or "") not in disallowed_ids
        and str(relation.get("to") or "") not in disallowed_ids
    ]


def _admission_readiness(payload: dict[str, Any]) -> str:
    quality_gate = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    explicit = str(payload.get("admission_readiness") or quality_gate.get("admission_readiness") or "")
    if explicit in {"evidence_ready", "case_ready", "execution_ready", "not_ready"}:
        return explicit
    candidate = _candidate(payload)
    objects = candidate.get("objects") if isinstance(candidate.get("objects"), dict) else {}
    if not (objects.get("SourceCase") and objects.get("EvidenceItem")):
        return "not_ready"
    if not (objects.get("FaultFamily") and objects.get("FaultVariant")):
        return "evidence_ready"
    actions = objects.get("DiagnosticAction") or []
    outcomes = objects.get("ActionOutcome") or []
    traces = objects.get("DiagnosticTrace") or []
    action_ids = {str(item.get("action_id") or "") for item in actions if isinstance(item, dict) and str(item.get("action_id") or "")}
    linked = bool(outcomes) and all(
        isinstance(item, dict)
        and str(item.get("action_id") or "") in action_ids
        and bool(item.get("evidence_ids") or item.get("evidence_message_ids"))
        for item in outcomes
    )
    return "execution_ready" if actions and traces and linked else "case_ready"


def _first_value(*sources: Any) -> Any:
    if not sources:
        return ""
    *containers, key = sources
    for source in containers:
        if not isinstance(source, dict):
            continue
        value = source.get(key)
        if value not in (None, "", [], {}):
            return value
        payload = source.get("payload")
        if isinstance(payload, dict):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return value
            graph = payload.get("graph")
            if isinstance(graph, dict):
                value = graph.get(key)
                if value not in (None, "", [], {}):
                    return value
        graph = source.get("graph")
        if isinstance(graph, dict):
            value = graph.get(key)
            if value not in (None, "", [], {}):
                return value
    return ""


def _graph_payload(store: JsonKGV2Store) -> dict[str, Any]:
    return {
        "objects": {
            key: list(value)
            for key, value in sorted(store.objects_by_type.items())
        },
        "relations": list(store.relations),
    }


def _graph_hash(store: JsonKGV2Store) -> str:
    payload = json.dumps(_graph_payload(store), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _object_fingerprint(store: JsonKGV2Store) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for obj_type, items in store.objects_by_type.items():
        out[obj_type] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            pk_name = V2_PRIMARY_KEYS[obj_type]
            obj_id = str(item.get(pk_name) or "")
            if obj_id:
                out[obj_type][obj_id] = hashlib.sha256(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
    return out


def _object_diff(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> dict[str, dict[str, int]]:
    diff: dict[str, dict[str, int]] = {}
    for obj_type in sorted(set(before) | set(after)):
        before_items = before.get(obj_type, {})
        after_items = after.get(obj_type, {})
        added = set(after_items) - set(before_items)
        removed = set(before_items) - set(after_items)
        updated = {
            obj_id
            for obj_id in set(before_items) & set(after_items)
            if before_items[obj_id] != after_items[obj_id]
        }
        if added or removed or updated:
            diff[obj_type] = {
                "added": len(added),
                "updated": len(updated),
                "removed": len(removed),
            }
    return diff


def _now() -> str:
    return datetime.now(UTC).isoformat()
