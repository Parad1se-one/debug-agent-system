"""Human review workflow for KG_v2 terminology candidates.

Candidate generation is deterministic and non-authoritative.  It turns
existing search hints and ambiguous expressions into an auditable queue, but
only explicit reviewed decisions are imported into curated_terms.json.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.terminology import (
    CURATED_RELATION_TYPES,
    SAFE_EQUIVALENCE_TYPES,
    build_terminology_layer,
    normalize_term,
    write_terminology_layer,
)


REVIEW_QUEUE_SCHEMA = "kg_v2.terminology_review_queue.v2"
CURATED_SCHEMA = "kg_v2.curated_terminology.v1"
REVIEW_QUEUE_FILE = "terminology_candidates.json"
DECISION_FIELDS = {
    "review_status",
    "selected_action",
    "selected_concept_id",
    "approved_relation_type",
    "reviewed_by",
    "reviewed_at",
    "review_note",
}


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part or "") for part in parts)
    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _concept_summary(concept: dict[str, Any]) -> dict[str, Any]:
    return {
        key: concept.get(key)
        for key in (
            "concept_id",
            "canonical_name",
            "concept_type",
            "canonical_target_type",
            "canonical_target_id",
            "category",
            "subsystem",
            "status",
            "source_object_ids",
        )
    }


def _source_evidence(
    store: JsonKGV2Store,
    senses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sense in senses:
        source_type = str(sense.get("source_object_type") or "")
        source_ids = [
            str(value)
            for value in (
                sense.get("source_object_ids")
                or [sense.get("source_object_id")]
            )
            if str(value or "")
        ]
        if source_type not in V2_PRIMARY_KEYS:
            continue
        index = store.object_index(source_type)
        for source_id in source_ids:
            identity = (source_type, source_id)
            if identity in seen:
                continue
            seen.add(identity)
            source = index.get(source_id) or {}
            output.append({
                "source_object_type": source_type,
                "source_object_id": source_id,
                "label": str(
                    source.get("label")
                    or source.get("title")
                    or source.get("heading")
                    or ""
                ),
                "summary": str(source.get("summary") or "")[:300],
                "family_id": str(source.get("family_id") or ""),
                "variant_id": str(source.get("variant_id") or ""),
                "equipment_type": str(
                    source.get("equipment_type") or ""
                ),
                "error_phase": str(source.get("error_phase") or ""),
            })
    return output


def build_terminology_review_items(
    store_or_root: JsonKGV2Store | str | Path,
    *,
    existing_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build pending review items while preserving unchanged decisions."""

    store = (
        store_or_root
        if isinstance(store_or_root, JsonKGV2Store)
        else JsonKGV2Store(store_or_root)
    )
    built = build_terminology_layer(store)
    objects = built["objects_by_type"]
    concepts = {
        str(item.get("concept_id") or ""): item
        for item in objects.get("DebugConcept") or []
        if isinstance(item, dict) and item.get("concept_id")
    }
    expressions = {
        str(item.get("term_id") or ""): item
        for item in objects.get("TermExpression") or []
        if isinstance(item, dict) and item.get("term_id")
    }
    senses_by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sense in objects.get("TermSense") or []:
        if isinstance(sense, dict) and sense.get("term_id"):
            senses_by_term[str(sense["term_id"])].append(sense)

    existing_by_id = {
        str(item.get("review_id") or ""): item
        for item in existing_items or []
        if isinstance(item, dict) and item.get("review_id")
    }
    items: list[dict[str, Any]] = []
    for term_id, senses in sorted(senses_by_term.items()):
        expression = expressions.get(term_id) or {}
        surface = str(expression.get("surface_form") or "")
        direct = [
            sense
            for sense in senses
            if sense.get("relation_type") in SAFE_EQUIVALENCE_TYPES
        ]
        hints = [
            sense
            for sense in senses
            if sense.get("relation_type") == "search_hint"
        ]
        concept_ids = sorted({
            str(sense.get("concept_id") or "")
            for sense in senses
            if str(sense.get("concept_id") or "")
        })
        direct_ids = {
            str(sense.get("concept_id") or "") for sense in direct
        }
        hint_ids = {
            str(sense.get("concept_id") or "") for sense in hints
        }
        noun_hints = [
            sense
            for sense in hints
            if sense.get("source_object_type") == "RawCorpusCandidate"
        ]
        concept_types = {
            str(concepts.get(concept_id, {}).get("concept_type") or "")
            for concept_id in concept_ids
        }
        if concept_types == {"operation"}:
            continue
        if noun_hints and len(hint_ids) == 1 and not direct_ids:
            candidate = noun_hints[0]
            kind = str(
                candidate.get("candidate_kind") or "noun_alias"
            )
            suggested_concept_id = next(iter(hint_ids))
            suggested_relation_type = str(
                candidate.get("candidate_relation_type")
                or "colloquial_alias"
            )
            risk = str(candidate.get("candidate_risk") or "medium")
        elif len(concept_ids) > 1:
            kind = "ambiguous_expression"
            suggested_concept_id = ""
            suggested_relation_type = ""
            risk = "high" if direct_ids else "medium"
        elif len(hint_ids) == 1 and not direct_ids:
            kind = "alias_promotion"
            suggested_concept_id = next(iter(hint_ids))
            suggested_relation_type = "colloquial_alias"
            risk = "medium"
        else:
            continue
        candidate_senses = direct + hints
        payload = {
            "candidate_kind": kind,
            "term_id": term_id,
            "surface_form": surface,
            "normalized_form": normalize_term(surface),
            "current_relation_types": sorted({
                str(sense.get("relation_type") or "")
                for sense in candidate_senses
            }),
            "candidate_concepts": [
                _concept_summary(concepts[concept_id])
                for concept_id in concept_ids
                if concept_id in concepts
            ],
            "context_options": [
                {
                    "concept_id": str(sense.get("concept_id") or ""),
                    "categories": sense.get("categories") or [],
                    "equipment_types": sense.get("equipment_types") or [],
                    "subsystems": sense.get("subsystems") or [],
                    "phases": sense.get("phases") or [],
                }
                for sense in candidate_senses
            ],
            "source_evidence": _source_evidence(store, candidate_senses),
            "suggested_concept_id": suggested_concept_id,
            "suggested_relation_type": suggested_relation_type,
            "risk": risk,
        }
        if noun_hints:
            payload["review_domain"] = "noun_entity"
            payload["corpus_count"] = max(
                int(sense.get("corpus_count") or 0)
                for sense in noun_hints
            )
            payload["corpus_evidence_paths"] = sorted({
                str(path)
                for sense in noun_hints
                for path in sense.get("corpus_evidence_paths") or []
                if str(path or "")
            })
        else:
            payload["review_domain"] = "diagnostic_term"
        fingerprint = _content_hash(payload)
        review_id = _stable_id(
            "term-review",
            kind,
            term_id,
        )
        item = {
            "schema_version": REVIEW_QUEUE_SCHEMA,
            "review_id": review_id,
            **payload,
            "content_hash": fingerprint,
            "review_status": "pending",
            "allowed_actions": ["approve", "reject", "defer"],
            "approval_requirements": [
                "selected_concept_id",
                "approved_relation_type",
                "reviewed_by",
            ],
        }
        previous = existing_by_id.get(review_id)
        if previous:
            if str(previous.get("content_hash") or "") == fingerprint:
                for field in DECISION_FIELDS:
                    if field in previous:
                        item[field] = previous[field]
            elif str(previous.get("review_status") or "") not in {
                "",
                "pending",
            }:
                item["review_status"] = "needs_re_review"
                item["previous_decision"] = {
                    field: previous.get(field)
                    for field in DECISION_FIELDS
                    if field in previous
                }
        items.append(item)
    priority = {
        "noun_typo": 0,
        "noun_abbreviation": 1,
        "noun_alias": 2,
        "ambiguous_expression": 3,
        "alias_promotion": 4,
    }
    items.sort(key=lambda item: (
        priority.get(str(item.get("candidate_kind") or ""), 9),
        -int(item.get("corpus_count") or 0),
        str(item.get("surface_form") or ""),
    ))
    report = {
        "schema_version": "kg_v2.terminology_review_report.v2",
        "terminology_revision": built["report"]["revision"],
        "candidate_count": len(items),
        "ambiguous_expression_count": sum(
            item["candidate_kind"] == "ambiguous_expression"
            for item in items
        ),
        "alias_promotion_count": sum(
            item["candidate_kind"] == "alias_promotion"
            for item in items
        ),
        "noun_candidate_count": sum(
            item.get("review_domain") == "noun_entity"
            for item in items
        ),
        "noun_typo_count": sum(
            item.get("candidate_kind") == "noun_typo"
            for item in items
        ),
        "noun_alias_count": sum(
            str(item.get("candidate_kind") or "").startswith("noun_")
            for item in items
        ),
        "pending_count": sum(
            item["review_status"] in {"pending", "needs_re_review"}
            for item in items
        ),
    }
    return items, report


def write_terminology_review_queue(
    root: str | Path,
) -> dict[str, Any]:
    store = JsonKGV2Store(root)
    existing = store.read_review_queue(REVIEW_QUEUE_FILE)
    items, report = build_terminology_review_items(
        store,
        existing_items=existing,
    )
    store.write_review_queue(REVIEW_QUEUE_FILE, items)
    return {
        **report,
        "queue_file": f"review_queue/{REVIEW_QUEUE_FILE}",
    }


def approved_entries_from_reviews(
    items: list[dict[str, Any]],
    *,
    concepts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate explicit approvals and return curated entries plus rejects."""

    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        review_status = str(item.get("review_status") or "")
        selected_action = str(item.get("selected_action") or "")
        if review_status not in {
            "approved",
            "human_approved",
        } and selected_action != "approve":
            continue
        review_id = str(item.get("review_id") or "")
        concept_id = str(item.get("selected_concept_id") or "")
        relation_type = str(item.get("approved_relation_type") or "")
        reviewed_by = str(item.get("reviewed_by") or "").strip()
        candidate_ids = {
            str(candidate.get("concept_id") or "")
            for candidate in item.get("candidate_concepts") or []
            if isinstance(candidate, dict)
        }
        reasons: list[str] = []
        if review_status in {"rejected", "deferred", "needs_re_review"}:
            reasons.append("conflicting_review_decision")
        if selected_action and selected_action != "approve":
            reasons.append("selected_action_is_not_approve")
        if concept_id not in concepts or concept_id not in candidate_ids:
            reasons.append("invalid_selected_concept")
        if relation_type not in CURATED_RELATION_TYPES:
            reasons.append("invalid_approved_relation_type")
        if not reviewed_by:
            reasons.append("missing_reviewer")
        if reasons:
            rejected.append({
                "review_id": review_id,
                "reasons": reasons,
            })
            continue
        entries.append({
            "surface_form": str(item.get("surface_form") or ""),
            "relation_type": relation_type,
            "concept_id": concept_id,
            "approved": True,
            "review": {
                "review_id": review_id,
                "reviewed_by": reviewed_by,
                "reviewed_at": str(item.get("reviewed_at") or ""),
                "note": str(item.get("review_note") or ""),
            },
        })
    return entries, rejected


def apply_approved_terminology_reviews(
    root: str | Path,
) -> dict[str, Any]:
    """Import valid human approvals and rebuild the terminology projection."""

    store = JsonKGV2Store(root)
    items = store.read_review_queue(REVIEW_QUEUE_FILE)
    concepts = {
        str(item.get("concept_id") or ""): item
        for item in store.objects_by_type.get("DebugConcept") or []
        if isinstance(item, dict) and item.get("concept_id")
    }
    entries, rejected = approved_entries_from_reviews(
        items,
        concepts=concepts,
    )
    curated_path = store.root / "terminology" / "curated_terms.json"
    existing_payload = (
        json.loads(curated_path.read_text(encoding="utf-8"))
        if curated_path.exists()
        else {"schema_version": CURATED_SCHEMA, "entries": []}
    )
    existing_entries = [
        item
        for item in existing_payload.get("entries") or []
        if isinstance(item, dict)
    ]
    by_identity = {
        (
            normalize_term(item.get("surface_form")),
            str(item.get("concept_id") or ""),
            str(item.get("canonical_target_type") or ""),
            str(item.get("canonical_target_id") or ""),
            str(item.get("relation_type") or ""),
        ): item
        for item in existing_entries
    }
    added = 0
    for entry in entries:
        identity = (
            normalize_term(entry.get("surface_form")),
            str(entry.get("concept_id") or ""),
            "",
            "",
            str(entry.get("relation_type") or ""),
        )
        if identity not in by_identity:
            existing_entries.append(entry)
            by_identity[identity] = entry
            added += 1
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.write_text(json.dumps({
        "schema_version": CURATED_SCHEMA,
        "description": (
            "人工审核后的等价表达。只有 approved=true 的条目参与安全 "
            "Query 扩展；历史 keywords 不应直接复制到这里。"
        ),
        "entries": existing_entries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = write_terminology_layer(store.root)
    return {
        "status": "applied",
        "approved_review_count": len(entries),
        "added_curated_entry_count": added,
        "rejected_approval_count": len(rejected),
        "rejected_approvals": rejected,
        "terminology_revision": manifest["revision"],
    }


__all__ = [
    "REVIEW_QUEUE_FILE",
    "apply_approved_terminology_reviews",
    "approved_entries_from_reviews",
    "build_terminology_review_items",
    "write_terminology_review_queue",
]
