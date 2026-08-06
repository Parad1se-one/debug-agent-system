"""Source-closed Evidence Pack for optional answer organization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from debug_agent_system.agents.read.evidence_answer import ComposedEvidenceAnswer
from debug_agent_system.core.contracts import AnswerSection, SessionState
from debug_agent_system.knowledge_v2.query_scope import (
    analyze_query_scope,
    task_facet_matches_text,
)
from debug_agent_system.knowledge_v2.read_model import V2DiagnosticPlan


_SOURCE_ONLY_SECTIONS = {"sources"}
_NON_EVIDENCE_SECTIONS = {"uncertainty", "required_info", "evidence_gap"}
_CONTENT_SECTIONS = {
    "known", "diagnostic_steps", "document_guidance", "conditions",
}


@dataclass(slots=True)
class EvidencePack:
    payload: dict[str, Any]
    source_items: dict[str, dict[str, Any]]
    source_section: AnswerSection | None
    eligible_for_llm: bool
    fallback_reason: str = ""


class EvidencePackBuilder:
    """Normalize deterministic facts into a bounded, reference-closed pack."""

    schema_version = "debug_agent_system.answer_evidence_pack.v2"

    def __init__(
        self,
        *,
        max_documents: int = 8,
        max_chunks: int = 64,
        max_input_chars: int = 60000,
    ) -> None:
        self.max_documents = max(1, int(max_documents))
        self.max_chunks = max(1, int(max_chunks))
        self.max_input_chars = max(1000, int(max_input_chars))

    def build(
        self,
        *,
        state: SessionState,
        status: str,
        composed: ComposedEvidenceAnswer,
        plan: V2DiagnosticPlan | None,
        required_data: Iterable[str],
    ) -> EvidencePack:
        source_items: dict[str, dict[str, Any]] = {}
        source_section: AnswerSection | None = None
        ordered_items: list[dict[str, Any]] = []
        for section_index, section in enumerate(composed.sections):
            if section.section_type in _SOURCE_ONLY_SECTIONS:
                source_section = section
                continue
            for item_index, raw_item in enumerate(section.items):
                item_id = f"answer-item:{section_index + 1}:{item_index + 1}"
                item = {
                    **dict(raw_item),
                    "item_id": item_id,
                    "original_section_type": section.section_type,
                    "original_section_title": section.title,
                    "original_section_order": section_index,
                    "original_item_order": item_index,
                }
                item["evidence_ids"] = _dedupe(item.get("evidence_ids") or [])
                item["chunk_ids"] = _dedupe(item.get("chunk_ids") or [])
                item["sources"] = _dedupe(item.get("sources") or [])
                item["media_refs"] = [
                    dict(media)
                    for media in item.get("media_refs") or []
                    if isinstance(media, dict)
                ]
                source_items[item_id] = item
                ordered_items.append(item)

        excluded_items = [
            {
                "item_id": str(item.get("id") or ""),
                "reason": str(item.get("reason") or "excluded_upstream"),
                "stage": "deterministic_composer",
            }
            for item in composed.coverage.get("excluded") or []
            if isinstance(item, dict)
        ]
        # A SAG candidate summary is a retrieval hypothesis, not source
        # content.  Even when the candidate carries an EvidenceItem identifier,
        # the generated “候选/匹配信号” sentence itself must not satisfy the
        # answer evidence floor or enter the model-selectable body.  The
        # underlying document/action remains independently selectable when it
        # is actually present.
        selectable_items: list[dict[str, Any]] = []
        for item in ordered_items:
            if (
                item.get("original_section_type") == "conditions"
                and "KG_v2 SAG candidate" in set(item.get("sources") or [])
            ):
                excluded_items.append({
                    "item_id": item["item_id"],
                    "reason": "retrieval_candidate_not_answer_evidence",
                    "stage": "evidence_pack",
                })
                source_items.pop(str(item["item_id"]), None)
                continue
            selectable_items.append(item)
        ordered_items = selectable_items

        scope = analyze_query_scope(state.query)
        task_model = dict(scope.task_model)
        facets = self._query_facets(
            state.query,
            task_model,
            ordered_items,
        )
        supported_facets = [
            facet["facet_id"]
            for facet in facets
            if facet["supported_item_ids"]
        ]
        unsupported_facets = [
            facet["facet_id"]
            for facet in facets
            if not facet["supported_item_ids"]
        ]
        grounded_item_ids = [
            str(item["item_id"])
            for item in ordered_items
            if _is_grounded_content_item(item)
        ]
        evidence_floor_met = bool(grounded_item_ids)
        required_ids = self._required_item_ids(ordered_items, facets)
        for item in ordered_items:
            item_id = str(item["item_id"])
            selection_class = (
                "required" if item_id in required_ids else "optional"
            )
            item["selection_class"] = selection_class
            # Kept for one schema transition so older local clients fail safe
            # instead of silently treating optional items as mandatory.
            item["mandatory"] = selection_class == "required"
        document_ids = _dedupe(
            str(item.get("document_id") or "")
            for item in ordered_items
            if str(item.get("document_id") or "")
        )
        chunk_ids = _dedupe(
            chunk_id
            for item in ordered_items
            for chunk_id in item.get("chunk_ids") or []
        )
        evidence_ids = _dedupe(
            evidence_id
            for item in ordered_items
            for evidence_id in item.get("evidence_ids") or []
        )
        media_ids = _dedupe(
            str(media.get("media_id") or media.get("content_hash") or media.get("asset_path") or "")
            for item in ordered_items
            for media in item.get("media_refs") or []
        )
        evidence_groups = self._groups(ordered_items)
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "query": state.query,
            "query_scope": {
                **scope.to_dict(),
                "task_model": task_model,
                "requested_operations": list(
                    task_model.get("operations") or []
                ),
                "context_operations": list(
                    task_model.get("context_operations") or []
                ),
                "facets": facets,
                "supported_facets": supported_facets,
                "unsupported_facets": unsupported_facets,
                "evidence_floor_met": evidence_floor_met,
                "grounded_item_ids": grounded_item_ids,
            },
            "runtime_decision": {
                "status": status,
                "lock_status": state.lock_status,
                "family_id": state.top_family_id,
                "variant_id": state.top_variant_id,
                "plan_id": state.plan_id,
                **dict(composed.sufficiency),
                "verified_fix": bool(status == "resolved" and state.resolved_action_id),
            },
            "source_items": ordered_items,
            "excluded_items": excluded_items,
            "selection_policy": {
                "required_item_ids": [
                    item["item_id"] for item in ordered_items
                    if item.get("selection_class") == "required"
                ],
                "optional_item_ids": [
                    item["item_id"] for item in ordered_items
                    if item.get("selection_class") == "optional"
                ],
                "excluded_count": len(excluded_items),
                "minimum_grounded_content_items": 1,
            },
            "evidence_groups": evidence_groups,
            "diagnostic_trace": {
                "plan_id": plan.plan_id if plan is not None else "",
                "source_type": plan.source_type if plan is not None else "",
                "action_ids": [
                    step.action_id for step in (plan.steps if plan is not None else [])
                ],
                "current_action_id": state.current_action_id,
                "current_trace_step_id": state.current_trace_step_id,
            },
            "required_data": _dedupe(required_data),
            "allowed_references": {
                "item_ids": list(source_items),
                "evidence_ids": evidence_ids,
                "chunk_ids": chunk_ids,
                "media_ids": media_ids,
            },
            "budgets": {
                "max_documents": self.max_documents,
                "max_chunks": self.max_chunks,
                "max_input_chars": self.max_input_chars,
                "document_count": len(document_ids),
                "chunk_count": len(chunk_ids),
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        payload["budgets"]["input_chars"] = len(encoded)
        reasons: list[str] = []
        if len(document_ids) > self.max_documents:
            reasons.append("max_answer_documents_exceeded")
        if len(chunk_ids) > self.max_chunks:
            reasons.append("max_answer_chunks_exceeded")
        if len(encoded) > self.max_input_chars:
            reasons.append("max_answer_input_chars_exceeded")
        if not evidence_floor_met:
            reasons.append("no_approved_grounded_content")
        # A model may organize supported material, but it must not be used to
        # conceal a missing subtask.  The deterministic answer remains the
        # explicit fail-open result until retrieval/tool completion fills it.
        if unsupported_facets and supported_facets:
            payload["query_scope"]["partial_evidence_closure"] = True
        payload["budgets"]["eligible_for_llm"] = not reasons
        payload["budgets"]["fallback_reasons"] = reasons
        return EvidencePack(
            payload=payload,
            source_items=source_items,
            source_section=source_section,
            eligible_for_llm=not reasons,
            fallback_reason=";".join(reasons),
        )

    @staticmethod
    def _query_facets(
        query: str,
        task_model: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        facets: list[dict[str, Any]] = []
        for task_facet in task_model.get("facets") or []:
            if not bool(task_facet.get("required_for_closure", True)):
                continue
            supported = [
                str(item["item_id"])
                for item in items
                if _is_grounded_content_item(item)
                and task_facet_matches_text(
                    task_facet,
                    _item_corpus(item),
                )
            ]
            facets.append({
                **dict(task_facet),
                "supported_item_ids": supported,
            })
        if not facets:
            content_items = [
                str(item["item_id"])
                for item in items
                if _is_grounded_content_item(item)
            ]
            facets.append({
                "facet_id": "request:primary",
                "kind": "primary_request",
                "label": str(query or "").strip(),
                "supported_item_ids": content_items,
            })
        return facets

    @staticmethod
    def _required_item_ids(
        items: list[dict[str, Any]],
        facets: list[dict[str, Any]],
    ) -> set[str]:
        required: set[str] = set()
        by_id = {str(item["item_id"]): item for item in items}
        for item in items:
            section_type = str(item.get("original_section_type") or "")
            if section_type in _NON_EVIDENCE_SECTIONS:
                required.add(str(item["item_id"]))
            if (
                item.get("action_id")
                or item.get("outcome_type")
                or item.get("safety_guarded")
                or item.get("safety_level") in {"high", "destructive"}
            ):
                required.add(str(item["item_id"]))
        # Require the best attributable anchor for each task facet.  Other
        # supporting facts stay optional so the organizer can remove weak or
        # repetitive wide-recall material without hiding a missing subtask.
        for facet in facets:
            candidates = [
                by_id[item_id]
                for item_id in facet.get("supported_item_ids") or []
                if item_id in by_id
            ]
            if not candidates:
                continue
            chosen = min(candidates, key=_required_item_priority)
            required.add(str(chosen["item_id"]))
        return required

    @staticmethod
    def _groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for item in items:
            document_id = str(item.get("document_id") or "")
            key = document_id or "runtime-or-kg"
            group = groups.setdefault(key, {
                "document_id": document_id,
                "title": _first(item.get("navigation_path") or [])
                or _first(item.get("sources") or [])
                or key,
                "source_item_ids": [],
                "chunk_ids": [],
                "evidence_ids": [],
            })
            group["source_item_ids"].append(item["item_id"])
            group["chunk_ids"] = _dedupe([
                *group["chunk_ids"], *item.get("chunk_ids", []),
            ])
            group["evidence_ids"] = _dedupe([
                *group["evidence_ids"], *item.get("evidence_ids", []),
            ])
        return list(groups.values())


def _item_corpus(item: dict[str, Any]) -> str:
    values: list[str] = [
        str(item.get("text") or ""),
        str(item.get("source_heading") or ""),
        *[str(value) for value in item.get("sources") or []],
        *[str(value) for value in item.get("navigation_path") or []],
    ]
    # Preserve token boundaries for ASCII tools and attachment names such as
    # ``DDU.zip``.  Query-task matching compacts CJK terms itself where needed.
    return "\n".join(values).lower()


def _is_grounded_content_item(item: dict[str, Any]) -> bool:
    if str(item.get("original_section_type") or "") not in _CONTENT_SECTIONS:
        return False
    if item.get("chunk_ids") or item.get("evidence_ids"):
        return True
    # DiagnosticTrace actions and ActionOutcome objects are canonical KG_v2
    # facts even when the materialized action currently has no source chunk.
    return bool(item.get("action_id") or item.get("outcome_type"))


def _required_item_priority(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if item.get("direct_document_match") else 1,
        0 if item.get("action_id") else 1,
        0 if item.get("evidence_ids") else 1,
        0 if item.get("chunk_ids") else 1,
        int(item.get("original_section_order") or 0),
        int(item.get("original_item_order") or 0),
        str(item.get("item_id") or ""),
    )


def _first(values: Iterable[Any]) -> str:
    return next((str(value) for value in values if str(value).strip()), "")


def _dedupe(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "")
        if item and item not in result:
            result.append(item)
    return result
