"""Deterministic, source-complete answer composition for KG_v2 reads."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from debug_agent_system.core.contracts import AnswerSection, SessionState
from debug_agent_system.knowledge_v2.query_scope import (
    analyze_query_scope,
    chunk_matches_named_scope,
)
from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel, V2DiagnosticPlan, V2PlanStep


_MEDIA_PLACEHOLDER = re.compile(r"^\[(?:图片|附件)：[^\]]+\]$")
_FIGURE_CAPTION = re.compile(
    r"^(?:图|figure|fig\.?)\s*"
    r"(?:\d+(?:[-.]\d+)*|[一二三四五六七八九十百]+)"
    r"(?:\s*[：:]\s*.+)?$",
    re.IGNORECASE,
)
_PURE_FIGURE_REFERENCE = re.compile(
    r"^如(?:上|下)?图\s*"
    r"(?:\d+(?:[-.]\d+)*|[一二三四五六七八九十百]+)"
    r"\s*所示[；;。.]?$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ComposedEvidenceAnswer:
    answer: str
    sections: list[AnswerSection]
    coverage: dict[str, Any]
    sufficiency: dict[str, Any]


class EvidenceAnswerComposer:
    """Organize every scoped, attributable fact before asking for more data."""

    def __init__(self, model: KGV2ReadModel) -> None:
        self.model = model

    def compose(
        self,
        *,
        state: SessionState,
        status: str,
        base_answer: str,
        plan: V2DiagnosticPlan | None,
        required_data: Iterable[str] = (),
    ) -> ComposedEvidenceAnswer:
        required = _dedupe(required_data)
        chunks = self._scoped_chunks(state)
        facts, merged_count, excluded = self._facts(state, chunks)
        tool_facts, tool_excluded = self._tool_facts(state)
        facts = [*tool_facts, *facts]
        excluded.extend(tool_excluded)
        sections: list[AnswerSection] = []
        document_mode = bool((state.metadata.get("document_answer_mode") or {}).get("active"))
        document_guidance = [
            fact for fact in facts
            if _is_document_guidance_fact(fact)
        ]
        known_facts = [fact for fact in facts if fact not in document_guidance]

        if known_facts:
            title = "根据直接命中的资料可知" if document_mode else "根据资料可知"
            sections.append(_section("known", title, known_facts))

        if plan is not None and plan.steps:
            step_items = self._step_items(state, plan.steps)
            if step_items:
                sections.append(_section("diagnostic_steps", "建议排查顺序", step_items))
        if document_guidance:
            sections.append(_section(
                "document_guidance",
                "文档建议的处理路径",
                document_guidance,
            ))
        condition_items = self._condition_items(state, plan.steps if plan is not None else [])
        if condition_items:
            sections.append(_section("conditions", "适用条件与不同情况", condition_items))

        uncertainty = self._uncertainty_item(state, status, base_answer, facts, plan)
        if uncertainty:
            sections.append(_section("uncertainty", "尚不能确认", [uncertainty]))

        if required:
            source_ids = _dedupe(state.evidence_ids)
            sections.append(_section("required_info", "需要补充的信息", [
                {
                    "text": item,
                    "evidence_ids": source_ids,
                    "chunk_ids": [],
                    "sources": ["KG_v2 RequiredInfoSpec"],
                }
                for item in required
            ]))

        source_items = self._source_items(facts)
        if source_items:
            sections.append(_section("sources", "资料来源", source_items))

        _dedupe_media_across_sections(sections)
        if sections:
            answer = _render_sections(sections)
        else:
            answer = str(base_answer or "当前没有检索到可引用的相关资料。").strip()

        eligible_evidence_ids = _dedupe(
            evidence_id for fact in facts for evidence_id in fact.get("evidence_ids") or []
        )
        eligible_chunk_ids = _dedupe(
            chunk_id for fact in facts for chunk_id in fact.get("chunk_ids") or []
        )
        coverage = {
            "eligible_evidence_count": len(eligible_evidence_ids),
            "eligible_chunk_count": len(eligible_chunk_ids),
            "eligible_fact_count": len(facts),
            "included_fact_count": len(facts),
            "merged_fact_count": merged_count,
            "excluded": excluded,
            "complete": True,
        }
        current = _current_step(state, plan)
        answerable = any(
            not fact.get("tool_observation")
            or bool(fact.get("supports_retrieval", True))
            for fact in facts
        )
        # A tentative candidate may help formulate a distinguishing question,
        # but it is not yet a diagnosis.  Keep the three sufficiency axes
        # aligned with the actual lock state exposed by the runtime.
        diagnosable = bool(
            state.lock_status == "kg_v2_locked"
            and state.top_variant_id
            and plan is not None
            and plan.steps
        )
        executable = bool(
            diagnosable
            and status == "step"
            and current is not None
            and not current.destructive
            and not current.high_cost
            and not required
        )
        reasons: list[str] = []
        if not answerable:
            reasons.append("no_attributable_retrieval_evidence")
        if not diagnosable:
            reasons.append("variant_not_locked")
        if diagnosable and not executable:
            reasons.append("requires_information_or_safety_gate")
        return ComposedEvidenceAnswer(
            answer=answer,
            sections=sections,
            coverage=coverage,
            sufficiency={
                "answerable": answerable,
                "diagnosable": diagnosable,
                "executable": executable,
                "reasons": reasons,
            },
        )

    @staticmethod
    def _tool_facts(
        state: SessionState,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        resolution = (
            state.metadata.get("evidence_gap_resolution")
            if isinstance(state.metadata.get("evidence_gap_resolution"), dict)
            else {}
        )
        facts: list[dict[str, Any]] = []
        excluded = list(resolution.get("excluded") or [])
        for observation in resolution.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            field = str(observation.get("field") or "")
            value = observation.get("value")
            if not field or value in (None, "", []):
                excluded.append(
                    {
                        "id": str(observation.get("observation_id") or ""),
                        "reason": "empty_tool_observation",
                    }
                )
                continue
            source_ids = _dedupe(observation.get("source_ids") or [])
            facts.append(
                {
                    "text": f"只读工具观察（不等同于根因）— {field}：{_display_tool_value(value)}",
                    "evidence_ids": _dedupe(
                        observation.get("evidence_ids") or []
                    ),
                    "chunk_ids": [],
                    "sources": source_ids or ["read-side bounded parser"],
                    "tool_observation": True,
                    "supports_retrieval": bool(
                        observation.get("supports_retrieval", True)
                    ),
                }
            )
        return facts, excluded

    def _scoped_chunks(self, state: SessionState) -> list[dict[str, Any]]:
        retrieval = state.metadata.get("retrieval") if isinstance(state.metadata.get("retrieval"), dict) else {}
        if "supporting_chunks" in retrieval:
            # An explicitly empty list is a deliberate primary-evidence scope:
            # the intent/entity gate rejected all recalled neighbours.  Falling
            # back to ``model.last_retrieval`` here would reintroduce exactly
            # the off-topic chunks that the gate excluded.
            chunks = list(retrieval.get("supporting_chunks") or [])
        else:
            chunks = list((self.model.last_retrieval or {}).get("chunks") or [])
        if not chunks:
            return []
        variant_id = str(state.top_variant_id or "")
        if variant_id and self.model.sag is not None:
            # Once a variant is locked, a strongly retrieved source-document
            # chunk is an entry point to that reviewed document's complete
            # semantic outline.  Expand only documents already present in the
            # retrieval window and already linked to the locked variant; this
            # prevents a low-ranked 3.3/3.5 section from disappearing behind
            # unrelated high-BM25 snippets.
            linked_documents: dict[str, float] = {}
            for chunk in chunks:
                document_id = str(chunk.get("document_id") or "")
                variant_ids = {
                    str(value) for value in chunk.get("variant_ids") or []
                }
                if (
                    document_id
                    and variant_id in variant_ids
                    and str(chunk.get("chunk_id") or "").startswith("chunk:source:")
                ):
                    linked_documents[document_id] = max(
                        linked_documents.get(document_id, 0.0),
                        float(chunk.get("retrieval_score") or 0.0),
                    )
            selected_documents = [
                document_id
                for document_id, _score in sorted(
                    linked_documents.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:2]
            ]
            if selected_documents:
                expanded = self.model.sag.expand_source_document_chunks(
                    state.query,
                    selected_documents,
                )
                by_chunk_id = {
                    str(item.get("chunk_id") or ""): item
                    for item in chunks
                    if str(item.get("chunk_id") or "")
                }
                for item in expanded:
                    by_chunk_id.setdefault(str(item.get("chunk_id") or ""), item)
                chunks = list(by_chunk_id.values())
        document_mode = state.metadata.get("document_answer_mode") if isinstance(state.metadata.get("document_answer_mode"), dict) else {}
        trace = (retrieval or {}).get("trace") if isinstance((retrieval or {}).get("trace"), dict) else {}
        direct_document_ids = {
            str(item.get("document_id") or "")
            for item in [
                *((document_mode or {}).get("documents") or []),
                *((trace or {}).get("direct_document_matches") or []),
                *((trace or {}).get("navigation_document_matches") or []),
            ]
            if str(item.get("document_id") or "")
        }
        query_scope = analyze_query_scope(state.query)
        scoped: list[dict[str, Any]] = []
        for chunk in chunks:
            if not bool(chunk.get("approved", True)):
                continue
            variant_ids = {str(item) for item in chunk.get("variant_ids") or []}
            # Orphan chunks remain useful for a knowledge-only answer.  Once a
            # variant is locked, linked evidence is preferred, while a very
            # strong orphan result may still explain the queried document.
            matched_terms = _dedupe(chunk.get("matched_terms") or [])
            strong_terms = [term for term in matched_terms if len(str(term)) >= 3]
            score_components = chunk.get("score_components") if isinstance(chunk.get("score_components"), dict) else {}
            query_coverage = (
                float(score_components.get("query_coverage") or 0.0)
                if "query_coverage" in score_components
                else min(1.0, len(strong_terms) / 2.0)
            )
            direct_document_match = bool(
                chunk.get("direct_document_match")
                or str(chunk.get("document_id") or "") in direct_document_ids
            )
            if direct_document_ids:
                if direct_document_match:
                    scoped.append({**chunk, "direct_document_match": True})
                continue
            strong_orphan = bool(
                not variant_ids
                and bool(strong_terms)
                and query_coverage >= 0.25
                and (
                    not query_scope.strong_identifiers
                    or chunk_matches_named_scope(state.query, chunk)
                )
            )
            knowledge_only_match = bool(
                not variant_id
                and bool(strong_terms)
                and query_coverage >= 0.2
            )
            if knowledge_only_match or (
                variant_id and (variant_id in variant_ids or strong_orphan)
            ):
                scoped.append({**chunk, "direct_document_match": direct_document_match})
        if direct_document_ids:
            documents_with_source_chunks = {
                str(item.get("document_id") or "")
                for item in scoped
                if str(item.get("chunk_id") or "").startswith("chunk:source:")
            }
            # Rebuilt source chunks preserve paragraph order and offsets.  If
            # they exist for a directly matched document, do not also render
            # the synthetic KnowledgeSection aggregate, which repeats the same
            # content as one oversized pseudo-fact.
            scoped = [
                item for item in scoped
                if (
                    str(item.get("document_id") or "") not in documents_with_source_chunks
                    or str(item.get("chunk_id") or "").startswith("chunk:source:")
                )
            ]
        scoped.sort(key=lambda item: (
            0 if item.get("direct_document_match") and str(item.get("chunk_id") or "").startswith("chunk:source:") else 1,
            int(item.get("navigation_order") or 999999),
            _chunk_source_order(item) if item.get("direct_document_match") else 999999,
            0 if len(str(item.get("text") or "")) > 80 else 1,
            -float(item.get("retrieval_score") or 0.0),
            str(item.get("chunk_id") or ""),
        ))
        # A direct document match is an explicit request for that document's
        # knowledge, so retain its complete semantic outline.  The smaller
        # bound remains appropriate for ordinary mixed-document retrieval.
        return scoped[:64] if direct_document_ids else scoped[:20]

    def _facts(
        self,
        state: SessionState,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
        raw: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        retrieval = (
            state.metadata.get("retrieval")
            if isinstance(state.metadata.get("retrieval"), dict)
            else {}
        )
        retrieval_trace = (
            retrieval.get("trace")
            if isinstance(retrieval.get("trace"), dict)
            else {}
        )
        direct_document_labels = {
            str(item.get("document_id") or ""): str(item.get("source_label") or "")
            for item in retrieval_trace.get("direct_document_matches") or []
            if isinstance(item, dict)
        }
        has_complete_direct_source = any(
            chunk.get("direct_document_match")
            and str(chunk.get("chunk_id") or "").startswith("chunk:source:")
            for chunk in chunks
        )
        complete_direct_source_paths = {
            str(offset.get("source_path") or "")
            for chunk in chunks
            if (
                chunk.get("direct_document_match")
                and str(chunk.get("chunk_id") or "").startswith("chunk:source:")
            )
            for offset in (
                chunk.get("source_offsets")
                if isinstance(chunk.get("source_offsets"), list)
                else []
            )
            if isinstance(offset, dict) and str(offset.get("source_path") or "")
        }
        complete_direct_source_chunks = [
            chunk
            for chunk in chunks
            if (
                chunk.get("direct_document_match")
                and str(chunk.get("chunk_id") or "").startswith("chunk:source:")
            )
        ]
        complete_source_chunks = [
            chunk
            for chunk in chunks
            if (
                str(chunk.get("chunk_id") or "").startswith("chunk:source:")
                and bool(chunk.get("approved", True))
                and chunk.get("content_hash")
            )
        ]
        evidence_ids = _dedupe(state.evidence_ids)
        for evidence in self.model.evidence(evidence_ids):
            evidence_id = str(evidence.get("evidence_id") or "")
            text = _clean_fact_text(str(evidence.get("summary") or ""))
            if not text:
                excluded.append({"id": evidence_id, "reason": "empty_summary"})
                continue
            if (
                str(evidence.get("source_kind") or "") == "tool_parse"
                and str(evidence.get("payload_ref") or "") in complete_direct_source_paths
            ):
                excluded.append({
                    "id": evidence_id,
                    "reason": "derived_section_summary_superseded_by_direct_source_chunks",
                })
                continue
            if has_complete_direct_source and text.endswith("…"):
                excluded.append({
                    "id": evidence_id,
                    "reason": "truncated_summary_superseded_by_direct_source_chunks",
                })
                continue
            if (
                has_complete_direct_source
                and _summary_superseded_by_source_chunks(
                    text,
                    complete_direct_source_chunks,
                    min_matches=1,
                )
            ):
                excluded.append({
                    "id": evidence_id,
                    "reason": "aggregate_summary_superseded_by_direct_source_chunks",
                })
                continue
            if _summary_superseded_by_source_chunks(
                text,
                complete_source_chunks,
                min_matches=3,
            ):
                excluded.append({
                    "id": evidence_id,
                    "reason": "aggregate_summary_superseded_by_source_chunks",
                })
                continue
            text, _ = _guard_high_risk_document_text(text)
            raw.append({
                "text": text,
                "evidence_ids": [evidence_id],
                "chunk_ids": [],
                "sources": [str(evidence.get("title") or evidence.get("external_id") or evidence_id)],
            })
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            text = _clean_fact_text(str(chunk.get("text") or ""))
            if not bool(chunk.get("approved", True)):
                excluded.append({"id": chunk_id, "reason": "unapproved"})
                continue
            if not text or not chunk.get("content_hash"):
                excluded.append({"id": chunk_id, "reason": "missing_text_or_hash"})
                continue
            object_id = str(chunk.get("object_id") or "")
            document_id = str(chunk.get("document_id") or "")
            source_label = str(chunk.get("source_label") or object_id or chunk_id).lstrip("：: ")
            text, safety_guarded = _guard_high_risk_document_text(text, source_label)
            offsets = chunk.get("source_offsets") if isinstance(chunk.get("source_offsets"), list) else []
            content_blocks = _content_blocks_from_offsets(offsets)
            block_types = {
                str(block_type)
                for offset in offsets
                if isinstance(offset, dict)
                for block_type in offset.get("block_types") or []
            }
            first_line = _first_nonempty_line(text)
            if text.startswith((
                "进阶硬件排查（需人工确认）",
                "特定现象分支（涉及拆装时需人工确认）",
            )):
                source_heading = _first_nonempty_line(text)
            elif "heading" in block_types:
                source_heading = _effective_source_heading(source_label, text)
            elif (
                chunk.get("direct_document_match")
                and _is_procedure_source_label(source_label)
            ):
                # A semantic continuation chunk may no longer contain the
                # heading block itself.  Retain the inherited source label so
                # the continuation can be woven into the same answer item.
                source_heading = source_label
            elif (
                chunk.get("direct_document_match")
                and _document_title_matches(source_label, first_line)
            ):
                source_heading = first_line
            else:
                source_heading = ""
            citation_label = source_label
            if text.startswith("进阶硬件排查（需人工确认）"):
                citation_label = (
                    direct_document_labels.get(document_id)
                    or source_label
                )
                citation_label = f"{citation_label}（进阶硬件排查）"
            elif (
                chunk.get("direct_document_match")
                and direct_document_labels.get(document_id)
                and _document_title_matches(source_label, direct_document_labels[document_id])
            ):
                citation_label = direct_document_labels[document_id]
            fact = {
                "text": text,
                "evidence_ids": [object_id] if chunk.get("object_type") == "EvidenceItem" else [],
                "chunk_ids": [chunk_id],
                "sources": [citation_label],
                "direct_document_match": bool(chunk.get("direct_document_match")),
                "safety_guarded": safety_guarded,
                "source_heading": source_heading,
                "document_id": document_id,
                "source_order": _chunk_source_order(chunk),
                "navigation_order": int(chunk.get("navigation_order") or 999999),
                "navigation_depth": int(chunk.get("navigation_depth") or 0),
                "navigation_path": list(chunk.get("navigation_path") or []),
                "navigation_document_path": list(
                    chunk.get("navigation_document_path") or []
                ),
                "navigation_paths": list(chunk.get("navigation_paths") or []),
                "navigation_branch_score": float(
                    chunk.get("navigation_branch_score") or 0.0
                ),
                "navigation_selection_reason": str(
                    chunk.get("navigation_selection_reason") or ""
                ),
                "media_refs": [
                    dict(item) for item in chunk.get("media_refs") or []
                    if isinstance(item, dict)
                ],
                "content_blocks": [] if safety_guarded else content_blocks,
            }
            leading_context, procedure_text = _split_leading_document_context(
                text,
                source_heading,
            )
            if leading_context:
                context_heading = next(
                    (
                        line.strip()
                        for line in leading_context.splitlines()
                        if line.strip()
                    ),
                    source_label,
                )
                raw.append({
                    "text": leading_context,
                    "evidence_ids": [],
                    "chunk_ids": [chunk_id],
                    "sources": [context_heading],
                    "direct_document_match": bool(chunk.get("direct_document_match")),
                    "safety_guarded": safety_guarded,
                    "source_heading": context_heading,
                    "document_context": True,
                    "document_id": str(chunk.get("document_id") or ""),
                    "source_order": _chunk_source_order(chunk),
                    "navigation_order": int(
                        chunk.get("navigation_order") or 999999
                    ),
                    "navigation_depth": int(chunk.get("navigation_depth") or 0),
                    "navigation_path": list(chunk.get("navigation_path") or []),
                    "navigation_document_path": list(
                        chunk.get("navigation_document_path") or []
                    ),
                    "navigation_paths": list(chunk.get("navigation_paths") or []),
                    "navigation_branch_score": float(
                        chunk.get("navigation_branch_score") or 0.0
                    ),
                    "navigation_selection_reason": str(
                        chunk.get("navigation_selection_reason") or ""
                    ),
                    "media_refs": [],
                    "content_blocks": [],
                })
                fact["text"] = procedure_text
                fact["content_blocks"] = _blocks_after_heading(
                    content_blocks,
                    source_heading,
                )
            raw.append(fact)

        # A DOCX drawing can occupy its own paragraph immediately after the
        # explanatory paragraph.  It is supporting media, not an independent
        # answer fact, so attach it to the preceding fact from the same source.
        normalized_raw: list[dict[str, Any]] = []
        merged_count = 0
        for fact in raw:
            if _is_media_only_fact(str(fact.get("text") or "")):
                target = next((
                    item for item in reversed(normalized_raw)
                    if fact.get("direct_document_match")
                    and item.get("direct_document_match")
                    and fact.get("document_id")
                    and fact.get("document_id") == item.get("document_id")
                ), None)
                if target is not None:
                    merged_count += 1
                    target["chunk_ids"] = _dedupe([
                        *target.get("chunk_ids", []), *fact.get("chunk_ids", []),
                    ])
                    target["media_refs"] = _dedupe_media_refs([
                        *target.get("media_refs", []), *fact.get("media_refs", []),
                    ])
                    continue
            normalized_raw.append(fact)

        # Rebuilt source chunks intentionally split long sections.  Rejoin
        # adjacent continuations that inherit the same structural heading so
        # one stage/method is rendered as one answer item.
        coalesced_raw: list[dict[str, Any]] = []
        for fact in normalized_raw:
            target = coalesced_raw[-1] if coalesced_raw else None
            if (
                target is not None
                and fact.get("direct_document_match")
                and target.get("direct_document_match")
                and fact.get("document_id")
                and fact.get("document_id") == target.get("document_id")
                and fact.get("source_heading")
                and fact.get("source_heading") == target.get("source_heading")
                and not fact.get("document_context")
                and not target.get("document_context")
            ):
                merged_count += 1
                if _fact_key(str(fact.get("text") or "")) != _fact_key(
                    str(target.get("text") or "")
                ):
                    incoming_text = str(fact.get("text") or "").lstrip()
                    if (
                        str(target.get("source_heading") or "").startswith(
                            "特定现象分支（涉及拆装时需人工确认）"
                        )
                        and incoming_text.startswith(
                            "特定现象分支（涉及拆装时需人工确认）"
                        )
                    ):
                        incoming_lines = incoming_text.splitlines()
                        incoming_text = "\n".join(incoming_lines[2:]).lstrip()
                    target["text"] = (
                        f"{str(target.get('text') or '').rstrip()}\n"
                        f"{incoming_text}"
                    )
                target["evidence_ids"] = _dedupe([
                    *target.get("evidence_ids", []), *fact.get("evidence_ids", []),
                ])
                target["chunk_ids"] = _dedupe([
                    *target.get("chunk_ids", []), *fact.get("chunk_ids", []),
                ])
                target["sources"] = _dedupe([
                    *target.get("sources", []), *fact.get("sources", []),
                ])
                target["media_refs"] = _dedupe_media_refs([
                    *target.get("media_refs", []), *fact.get("media_refs", []),
                ])
                target["content_blocks"] = [
                    *target.get("content_blocks", []),
                    *fact.get("content_blocks", []),
                ]
                continue
            coalesced_raw.append(fact)

        merged: dict[str, dict[str, Any]] = {}
        for fact in coalesced_raw:
            key = _fact_key(str(fact.get("text") or ""))
            if not key:
                excluded.append({"id": "", "reason": "empty_after_normalization"})
                continue
            if key not in merged:
                merged[key] = fact
                continue
            merged_count += 1
            current = merged[key]
            current["evidence_ids"] = _dedupe([*current.get("evidence_ids", []), *fact.get("evidence_ids", [])])
            current["chunk_ids"] = _dedupe([*current.get("chunk_ids", []), *fact.get("chunk_ids", [])])
            current["sources"] = _dedupe([*current.get("sources", []), *fact.get("sources", [])])
            current["direct_document_match"] = bool(
                current.get("direct_document_match") or fact.get("direct_document_match")
            )
            current["media_refs"] = _dedupe_media_refs([
                *current.get("media_refs", []),
                *fact.get("media_refs", []),
            ])
            if not current.get("content_blocks"):
                current["content_blocks"] = list(fact.get("content_blocks") or [])
            if not current.get("source_heading"):
                current["source_heading"] = str(fact.get("source_heading") or "")
            current["navigation_paths"] = _dedupe_navigation_paths([
                *current.get("navigation_paths", []),
                *fact.get("navigation_paths", []),
            ])
            if (
                not current.get("navigation_path")
                and fact.get("navigation_path")
            ):
                current["navigation_path"] = list(
                    fact.get("navigation_path") or []
                )
                current["navigation_document_path"] = list(
                    fact.get("navigation_document_path") or []
                )
                current["navigation_depth"] = int(
                    fact.get("navigation_depth") or 0
                )
        facts = list(merged.values())
        _compact_repeated_safety_guards(facts)
        return facts, merged_count, excluded

    @staticmethod
    def _step_items(state: SessionState, steps: list[V2PlanStep]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for step in steps:
            if step.destructive or step.high_cost:
                text = f"需人工确认后才可执行：{step.label}。未确认前仅保留为候选步骤。"
            else:
                prefix = "当前步骤：" if step.action_id == state.current_action_id else ""
                text = f"{prefix}{step.label}：{step.instruction}"
            items.append({
                "text": text,
                "evidence_ids": _dedupe(step.evidence_ids),
                "chunk_ids": [],
                "sources": ["KG_v2 DiagnosticTrace"],
                "action_id": step.action_id,
                "media_refs": _dedupe_media_refs(step.media_refs),
                "stage": step.stage,
                "safety_level": step.safety_level,
                "applicability_condition": step.applicability_condition,
                "expected_result": step.expected_result,
            })
        return items

    def _condition_items(self, state: SessionState, steps: list[V2PlanStep]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        if state.lock_status in {"kg_v2_tentative_ambiguous", "document_answer_only"}:
            retrieval = state.metadata.get("retrieval") if isinstance(state.metadata.get("retrieval"), dict) else {}
            candidates = list((retrieval or {}).get("candidates") or [])[:3]
            top_score = float(candidates[0].get("score") or 0.0) if candidates else 0.0
            for index, candidate in enumerate(candidates):
                if state.lock_status == "document_answer_only":
                    # Document-only mode intentionally avoids turning a broad
                    # token overlap (for example just "USB") into a diagnostic
                    # alternative.  Only a strong, explicit variant-label
                    # match is useful enough to present as a conditional.
                    matched_fields = set(candidate.get("matched_fields") or [])
                    if (
                        float(candidate.get("score") or 0.0) < 12.0
                        or "variant_label" not in matched_fields
                    ):
                        continue
                else:
                    # Ambiguity should expose plausible alternatives, not
                    # every top-k result produced by short Chinese n-grams.
                    # Keep the leader, then require alternatives to be close
                    # in score and to carry a meaningful (3+ char) signal.
                    meaningful = [
                        str(term)
                        for term in candidate.get("matched_entities") or []
                        if len(_fact_key(term)) >= 3
                    ]
                    if index > 0 and (
                        float(candidate.get("score") or 0.0) < top_score * 0.75
                        or not meaningful
                    ):
                        continue
                chunks = list(candidate.get("supporting_chunks") or [])
                items.append({
                    "text": (
                        f"候选：{candidate.get('family_label')} / {candidate.get('variant_label')}；"
                        f"匹配信号：{'、'.join(candidate.get('matched_entities') or candidate.get('matched_fields') or [])}"
                    ),
                    "evidence_ids": _dedupe(candidate.get("evidence_ids") or []),
                    "chunk_ids": _dedupe(str(chunk.get("chunk_id") or "") for chunk in chunks),
                    "sources": ["KG_v2 SAG candidate"],
                })
        for step in steps:
            for outcome in self.model.outcomes_for_step(step):
                text = str(outcome.get("summary") or "").strip()
                if not text or _fact_key(text) in seen:
                    continue
                seen.add(_fact_key(text))
                items.append({
                    "text": f"{outcome.get('outcome_type')}: {text}",
                    "evidence_ids": _dedupe(outcome.get("evidence_ids") or step.evidence_ids),
                    "chunk_ids": [],
                    "sources": ["KG_v2 ActionOutcome"],
                    "outcome_type": str(outcome.get("outcome_type") or ""),
                })
        return items

    @staticmethod
    def _uncertainty_item(
        state: SessionState,
        status: str,
        base_answer: str,
        facts: list[dict[str, Any]],
        plan: V2DiagnosticPlan | None,
    ) -> dict[str, Any] | None:
        if status == "resolved":
            return None
        retrieval = (
            state.metadata.get("retrieval")
            if isinstance(state.metadata.get("retrieval"), dict)
            else {}
        )
        trace = (
            retrieval.get("trace")
            if isinstance(retrieval.get("trace"), dict)
            else {}
        )
        if (
            state.lock_status == "document_answer_only"
            and trace.get("navigation_document_matches")
            and not state.required_data
        ):
            # A resolved navigation request is a knowledge/procedure answer,
            # not a claim that a field fault has a unique root cause.
            return None
        evidence_ids = _dedupe(
            evidence_id for fact in facts for evidence_id in fact.get("evidence_ids") or []
        )
        chunk_ids = _dedupe(chunk_id for fact in facts for chunk_id in fact.get("chunk_ids") or [])
        if not state.top_variant_id or plan is None:
            text = "检索结果可用于说明相关文档知识，但目前还不能将其确定为本次现场故障的唯一根因。"
        elif status == "ask_info":
            text = "以上资料可以先用于排查；缺失信息补齐前，不能确认具体分支或把历史案例直接视为本次根因。"
        elif status == "escalate":
            text = "现有 KG_v2 诊断链尚未形成可验证闭环，需要携带已收集证据升级处理。"
        else:
            text = "当前步骤仍需现场结果验证；建议不能替代已经执行并复测的事实。"
        if not facts and base_answer:
            text = str(base_answer).strip()
        return {
            "text": text,
            "evidence_ids": evidence_ids,
            "chunk_ids": chunk_ids,
            "sources": ["当前检索与会话状态"],
        }

    @staticmethod
    def _source_items(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: dict[str, dict[str, Any]] = {}
        for fact in facts:
            navigation_sources = [
                str(source)
                for path in fact.get("navigation_paths") or []
                if isinstance(path, dict)
                for source in path.get("source_labels") or []
                if str(source).strip()
            ]
            for source in _dedupe([
                *navigation_sources,
                *list(fact.get("sources") or []),
            ]):
                item = sources.setdefault(str(source), {
                    "text": str(source), "evidence_ids": [], "chunk_ids": [], "sources": [str(source)],
                })
                item["evidence_ids"] = _dedupe([*item["evidence_ids"], *fact.get("evidence_ids", [])])
                item["chunk_ids"] = _dedupe([*item["chunk_ids"], *fact.get("chunk_ids", [])])
        return list(sources.values())


def _current_step(state: SessionState, plan: V2DiagnosticPlan | None) -> V2PlanStep | None:
    if plan is None:
        return None
    return next((step for step in plan.steps if step.action_id == state.current_action_id), None)


def _display_tool_value(value: Any) -> str:
    if isinstance(value, dict):
        return "；".join(f"{key}={item}" for key, item in value.items())
    if isinstance(value, list):
        return "、".join(str(item) for item in value)
    return str(value)


def _is_document_guidance_fact(fact: dict[str, Any]) -> bool:
    if not fact.get("direct_document_match"):
        return False
    if fact.get("document_context"):
        return False
    heading = str(fact.get("source_heading") or "").strip()
    if _is_procedure_source_label(heading):
        return True
    if re.match(r"^\d+(?:\.\d+)+(?:[.、：:\s]|$)", heading):
        return True
    return len(str(fact.get("text") or "")) > 80


def _is_procedure_source_label(value: str) -> bool:
    return bool(re.match(
        r"^(?:阶段|方案|方法|情况|结论|步骤)\s*"
        r"(?:[一二三四五六七八九十百]+|\d+)(?:[：:\s(（]|$)",
        str(value or "").strip(),
    ))


def _effective_source_heading(source_label: str, text: str) -> str:
    """Prefer a parent numbered heading when one starts the source chunk.

    A semantic chunk may be aligned to a child section (for example ``3.2.1``)
    while retaining its parent ``3.2`` heading as the first source line.  The
    parent is the correct answer-list title; the child remains visible in the
    body as a nested condition.
    """

    heading = str(source_label or "").strip()
    heading_match = re.match(r"^(\d+(?:\.\d+)+)\b", heading)
    if heading_match is None:
        return heading
    for line in str(text or "").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate_match = re.match(r"^(\d+(?:\.\d+)+)\b", candidate)
        if candidate_match is None:
            return heading
        parent_number = candidate_match.group(1)
        heading_number = heading_match.group(1)
        if heading_number.startswith(f"{parent_number}."):
            return candidate
        return heading
    return heading


def _first_nonempty_line(text: str) -> str:
    return next(
        (line.strip() for line in str(text or "").splitlines() if line.strip()),
        "",
    )


def _document_title_matches(source_label: str, first_line: str) -> bool:
    label = re.sub(r"\s*[（(]\d+[）)]\s*$", "", str(source_label or "")).strip()
    first = str(first_line or "").strip()
    return bool(first and _fact_key(label) == _fact_key(first))


def _split_leading_document_context(
    text: str,
    source_heading: str,
) -> tuple[str, str]:
    """Separate document preamble that precedes a numbered procedure heading."""

    heading = str(source_heading or "").strip()
    if re.match(r"^\d+(?:\.\d+)+\b", heading) is None:
        return "", str(text or "")
    match = re.search(
        rf"(?m)^\s*{re.escape(heading)}\s*$",
        str(text or ""),
    )
    if match is None:
        return "", str(text or "")
    leading = str(text or "")[:match.start()].strip()
    procedure = str(text or "")[match.start():].strip()
    if not leading or not procedure:
        return "", str(text or "")
    return leading, procedure


def _chunk_source_order(chunk: dict[str, Any]) -> int:
    offsets = chunk.get("source_offsets") if isinstance(chunk.get("source_offsets"), list) else []
    first = offsets[0] if offsets and isinstance(offsets[0], dict) else {}
    try:
        return int(first.get("block_start") or first.get("paragraph_start") or 999999)
    except (TypeError, ValueError):
        return 999999


def _section(section_type: str, title: str, items: list[dict[str, Any]]) -> AnswerSection:
    return AnswerSection(
        section_type=section_type,
        title=title,
        items=items,
        evidence_ids=_dedupe(evidence_id for item in items for evidence_id in item.get("evidence_ids") or []),
        chunk_ids=_dedupe(chunk_id for item in items for chunk_id in item.get("chunk_ids") or []),
    )


def _render_sections(sections: list[AnswerSection]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(f"## {section.title}")
        for item in section.items:
            text = str(item.get("text") or "").strip()
            sources = _dedupe(item.get("sources") or [])
            suffix = f"【来源：{'；'.join(sources)}】" if sources else ""
            source_heading = str(item.get("source_heading") or "").strip()
            if source_heading:
                body = re.sub(
                    rf"(?m)^\s*{re.escape(source_heading)}\s*$",
                    "",
                    text,
                    count=1,
                ).strip()
                display_heading = _navigation_display_heading(
                    item,
                    _display_source_heading(source_heading, body),
                )
                lines.append(f"- **{display_heading}**{suffix}")
                inline_media: set[tuple[str, str]] = set()
                if body:
                    formatted_body, inline_media = _format_document_body(
                        body,
                        content_blocks=item.get("content_blocks") or [],
                        source_heading=source_heading,
                        as_steps=section.section_type == "document_guidance",
                        media_refs=item.get("media_refs") or [],
                    )
                    if formatted_body:
                        lines.append(formatted_body)
            else:
                lines.append(f"- {text}{suffix}")
                inline_media = set()
            for media in _dedupe_media_refs(item.get("media_refs") or []):
                if _media_render_key(media) in inline_media:
                    continue
                lines.extend(_render_media(media))
        lines.append("")
    return "\n".join(lines).strip()


def render_answer_sections(sections: list[AnswerSection]) -> str:
    """Render already verified sections with the deterministic formatter."""

    return _render_sections(sections)


def _dedupe_media_across_sections(sections: list[AnswerSection]) -> None:
    """Render each binary asset once while preserving every graph binding.

    The full V2 plan still carries each Action's ``media_refs``.  This pass is
    only for answer presentation, where direct-document facts precede the
    diagnostic plan and therefore retain the first, source-ordered rendering.
    """

    seen: set[tuple[str, str]] = set()
    for section in sections:
        for item in section.items:
            unique: list[dict[str, Any]] = []
            for media in _dedupe_media_refs(item.get("media_refs") or []):
                key = (
                    str(media.get("media_kind") or ""),
                    str(
                        media.get("content_hash")
                        or media.get("asset_path")
                        or media.get("archive_path")
                        or ""
                    ),
                )
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                unique.append(media)
            item["media_refs"] = unique


def _fact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").lower())


def _summary_superseded_by_source_chunks(
    summary: str,
    source_chunks: Iterable[dict[str, Any]],
    *,
    min_matches: int,
) -> bool:
    """Prefer complete source chunks over graph-level aggregate summaries.

    Some EvidenceItems contain a long, pre-composed digest of the same source
    document.  Rendering that digest alongside the rebuilt source chunks both
    duplicates the answer and can bypass the per-line safety guard.  A
    normalized, sufficiently long source excerpt is a deterministic signal
    that the aggregate has already been covered by the canonical chunks.
    """

    summary_key = _fact_key(summary)
    if len(summary_key) < 40:
        return False
    matched = 0
    for chunk in source_chunks:
        chunk_key = _fact_key(str(chunk.get("text") or ""))
        if len(chunk_key) < 40:
            continue
        probe_length = min(96, len(chunk_key))
        if (
            chunk_key[:probe_length] in summary_key
            or (len(chunk_key) <= len(summary_key) and chunk_key in summary_key)
        ):
            matched += 1
            if matched >= max(1, min_matches):
                return True
    return False


def _is_media_only_fact(text: str) -> bool:
    remainder = re.sub(r"\[(?:图片|附件)：[^\]]+\]", "", str(text or ""))
    return not remainder.strip("；;、，,。:.： \t\r\n")


def _display_source_heading(source_heading: str, body: str) -> str:
    heading = str(source_heading or "").strip()
    text = str(body or "").lower()
    if re.fullmatch(r"方法一[：:]?", heading) and "ipconfig /flushdns" in text and "winsock" in text:
        return "方法一：刷新 DNS 并重置 IP/Winsock"
    if re.fullmatch(r"方法二[：:]?", heading) and "vpn" in text:
        return "方法二：重置 VPN 连接状态"
    if re.fullmatch(r"方法三[：:]?", heading) and "代理" in text:
        return "方法三：关闭设置脚本和代理服务器"
    return heading


def _navigation_display_heading(
    item: dict[str, Any],
    display_heading: str,
) -> str:
    """Expose a selected second-hop document as an answer breadcrumb."""

    if int(item.get("navigation_depth") or 0) < 2:
        return str(display_heading or "").strip()
    path = [
        re.sub(r"\.(?:docx?|pdf|md|txt)$", "", str(value).strip(), flags=re.I)
        for value in item.get("navigation_path") or []
        if str(value).strip()
    ]
    if len(path) < 2:
        return str(display_heading or "").strip()
    # The root navigation page is only an entry point.  The useful hierarchy
    # starts at its first selected branch ("可以进系统") and child document.
    breadcrumb = path[-2:]
    heading = str(display_heading or "").strip()
    if heading and _fact_key(heading) != _fact_key(breadcrumb[-1]):
        breadcrumb.append(heading)
    return " → ".join(_dedupe(breadcrumb))


def _dedupe_navigation_paths(
    values: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        key = tuple(str(item) for item in value.get("document_ids") or [])
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _format_document_body(
    text: str,
    *,
    content_blocks: Iterable[dict[str, Any]] = (),
    source_heading: str = "",
    as_steps: bool = False,
    media_refs: Iterable[dict[str, Any]] = (),
) -> tuple[str, set[tuple[str, str]]]:
    """Render source paragraphs as nested Markdown while preserving DOCX lists."""

    blocks = [
        dict(block)
        for block in content_blocks
        if isinstance(block, dict) and str(block.get("text") or "").strip()
    ]
    if blocks:
        blocks = _blocks_after_heading(blocks, source_heading)
        # Guarded or otherwise transformed text must never be replaced by the
        # original source blocks.
        block_text = "\n".join(str(block.get("text") or "") for block in blocks)
        if _fact_key(text) and _fact_key(text) not in _fact_key(block_text):
            blocks = []
    if not blocks:
        fallback = [
            {"text": line.strip(), "kind": "paragraph"}
            for line in str(text or "").splitlines()
            if line.strip()
        ]
        blocks = fallback

    media = _dedupe_media_refs(media_refs)
    media_by_match_key = {
        key: item
        for item in media
        for key in _media_match_keys(item)
    }
    explicit_list = any(block.get("kind") == "list_item" for block in blocks)
    rendered: list[str] = []
    inline_media: set[tuple[str, str]] = set()
    inferred_number = 0
    media_cursor = 0
    skipped_indexes: set[int] = set()
    for index, block in enumerate(blocks):
        if index in skipped_indexes:
            continue
        value = str(block.get("text") or "").strip()
        if not value:
            continue
        kind = str(block.get("kind") or "paragraph")
        is_media_block = kind in {"image", "figure", "media"} or bool(
            _MEDIA_PLACEHOLDER.fullmatch(value)
        )
        if is_media_block:
            selected: list[dict[str, Any]] = []
            for key in block.get("media_keys") or []:
                item = media_by_match_key.get(str(key or ""))
                if item is not None and _media_render_key(item) not in {
                    _media_render_key(value) for value in selected
                }:
                    selected.append(item)
            if not selected:
                while (
                    media_cursor < len(media)
                    and _media_render_key(media[media_cursor]) in inline_media
                ):
                    media_cursor += 1
                if media_cursor < len(media):
                    selected.append(media[media_cursor])
                    media_cursor += 1
            adjacent_caption = ""
            if index + 1 < len(blocks):
                next_value = str(blocks[index + 1].get("text") or "").strip()
                next_kind = str(blocks[index + 1].get("kind") or "")
                if (
                    next_kind == "figure_caption"
                    or _FIGURE_CAPTION.fullmatch(next_value)
                ):
                    adjacent_caption = next_value
                    skipped_indexes.add(index + 1)
            for item in selected:
                label = _media_display_label(item, adjacent_caption)
                rendered.extend(_render_media(item, label=label))
                inline_media.add(_media_render_key(item))
            continue
        if kind == "figure_caption" or (
            media and _FIGURE_CAPTION.fullmatch(value)
        ):
            continue
        if media and _PURE_FIGURE_REFERENCE.fullmatch(value):
            continue
        value = _format_document_line(value)
        if kind == "list_item":
            level = max(int(block.get("list_level") or 0), 0)
            marker = str(block.get("list_marker") or "").strip()
            if str(block.get("list_style") or "") == "bullet":
                marker = "-"
            elif not marker:
                marker = "1."
            rendered.append(f"{'  ' * (level + 1)}{marker} {value}")
            continue
        if kind == "heading":
            rendered.append(f"  - **{value}**")
            continue
        if as_steps and not explicit_list and len(blocks) > 1:
            inferred_number += 1
            rendered.append(f"  {inferred_number}. {value}")
            continue
        if kind == "code_block":
            rendered.append(f"  `{value.strip('`')}`")
            continue
        rendered.append(f"  {value}  ")
    return "\n".join(rendered), inline_media


def _media_match_keys(media: dict[str, Any]) -> list[str]:
    return _dedupe([
        str(media.get("content_hash") or ""),
        str(media.get("media_id") or ""),
        str(media.get("archive_path") or ""),
        str(media.get("asset_path") or ""),
    ])


def _media_render_key(media: dict[str, Any]) -> tuple[str, str]:
    return (
        str(media.get("media_kind") or ""),
        str(
            media.get("content_hash")
            or media.get("media_id")
            or media.get("asset_path")
            or media.get("archive_path")
            or ""
        ),
    )


def _media_display_label(
    media: dict[str, Any],
    adjacent_caption: str = "",
) -> str:
    stored = str(
        media.get("caption")
        or media.get("context_label")
        or media.get("label")
        or media.get("archive_path")
        or "源文档资源"
    ).strip()
    adjacent = str(adjacent_caption or "").strip()
    if adjacent:
        adjacent_number = re.match(
            r"^(?:图|figure|fig\.?)\s*"
            r"(\d+(?:[-.]\d+)*|[一二三四五六七八九十百]+)",
            adjacent,
            flags=re.IGNORECASE,
        )
        stored_number = re.match(
            r"^(?:图|figure|fig\.?)\s*"
            r"(\d+(?:[-.]\d+)*|[一二三四五六七八九十百]+)",
            stored,
            flags=re.IGNORECASE,
        )
        stored_is_descriptive = bool(re.search(r"[：:]\s*\S+", stored))
        if (
            stored_is_descriptive
            and adjacent_number
            and stored_number
            and adjacent_number.group(1) == stored_number.group(1)
        ):
            pass
        else:
            stored = adjacent
    return re.sub(r"[\[\]\r\n]+", " ", stored)


def _render_media(
    media: dict[str, Any],
    *,
    label: str = "",
) -> list[str]:
    display_label = label or _media_display_label(media)
    asset_path = str(media.get("asset_path") or "").strip()
    kind_label = "图片" if media.get("media_kind") == "image" else "附件"
    if not asset_path:
        return [f"  - 【源文档{kind_label}：{display_label}】"]
    target = f"<{asset_path}>" if " " in asset_path else asset_path
    if media.get("media_kind") == "image":
        return [
            f"  - 图片说明：{display_label}",
            f"  ![源文档图片：{display_label}]({target})",
        ]
    return [f"  [源文档附件：{display_label}]({target})"]


def _format_document_line(value: str) -> str:
    stripped = str(value or "").strip()
    if re.fullmatch(r"(?i)(?:ipconfig|netsh)\s+[^`]+", stripped):
        return f"`{stripped}`"
    return stripped


def _content_blocks_from_offsets(
    offsets: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    for offset in offsets:
        if not isinstance(offset, dict):
            continue
        blocks = offset.get("content_blocks")
        if isinstance(blocks, list):
            return [dict(block) for block in blocks if isinstance(block, dict)]
    return []


def _blocks_after_heading(
    blocks: Iterable[dict[str, Any]],
    source_heading: str,
) -> list[dict[str, Any]]:
    values = [dict(block) for block in blocks if isinstance(block, dict)]
    heading = str(source_heading or "").strip()
    if not heading:
        return values
    for index, block in enumerate(values):
        if str(block.get("text") or "").strip() == heading:
            return values[index + 1:]
    return [
        block for block in values
        if str(block.get("text") or "").strip() != heading
    ]


def _clean_fact_text(text: str) -> str:
    value = str(text or "").strip()
    # Some imported rich-text code blocks preserve their language tag directly
    # before the first command (for example ``Bashipconfig``).  The tag is
    # presentation metadata, not part of the command or source fact.
    value = re.sub(
        r"(?i)(?<![a-z])bash(?=(?:ipconfig|netsh)\b)",
        "",
        value,
    )
    value = re.sub(
        r"^(?:(?:om_[a-z0-9]+|msg:[^：:；;\s]+|file:[^：:；;\s]+)(?:[/、](?:om_[a-z0-9]+))?)[：:]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    if re.fullmatch(r"(?:om_[a-z0-9]+|msg:[^\s]+|file:[^\s]+)(?:[/、](?:om_[a-z0-9]+))*", value, re.IGNORECASE):
        return ""
    return value


def _guard_high_risk_document_text(
    text: str,
    source_label: str = "",
) -> tuple[str, bool]:
    """Keep high-risk source knowledge without rendering it as an instruction."""

    value = str(text or "").strip()
    lowered = value.lower()
    if value.startswith("重要安全须知"):
        return value, False
    if value.startswith("应急处理与上报") or "应急处理与上报" in str(source_label or ""):
        value = re.sub(
            r"更换备件[：:]\s*根据排查结果，更换确认损坏的硬件",
            "更换备件：根据排查结果确认故障件；备件型号核对并取得人工确认后再更换",
            value,
        )
        return value, True

    if "特定现象" in value or "特定现象" in str(source_label or ""):
        # Symptom branches are useful for triage, but imported source text
        # often embeds disassembly operations inside parenthetical examples.
        # Preserve the branch distinctions while replacing those embedded
        # procedures with one explicit safety gate.
        guarded = re.sub(
            r"[（(][^（）()]{0,120}(?:重新插拔|换槽|更换|清除\s*cmos|"
            r"断开|短接|拆装)[^（）()]{0,120}[）)]",
            "（涉及拆装的操作需断电、防静电并取得人工确认）",
            value,
            flags=re.IGNORECASE,
        )
        guarded = re.sub(
            r"清除\s*cmos",
            "由有资质人员在断电并确认后执行 CMOS 重置",
            guarded,
            flags=re.IGNORECASE,
        )
        return _summarize_symptom_branches(guarded), True

    risk_marker = re.search(r"如果无法进入安全模式|制作启动盘|选择安装到u盘", lowered)
    has_data_loss_risk = any(token in lowered for token in ("格式化", "数据丢失", "重装系统", "删除所有数据"))
    if risk_marker and has_data_loss_risk:
        safe_prefix = value[:risk_marker.start()].rstrip("；;：: \n")
        guarded = (
            "如果安全模式和常规修复均失败，文档转入制作 PE 启动盘、在 PE 中修复系统/引导，"
            "以及备份后重装的兜底路径。制作启动盘会格式化 U 盘，重装会影响数据和授权；"
            "执行前必须备份、核对目标盘、处理加密狗授权并取得人工确认。"
        )
        return (f"{safe_prefix}\n{guarded}" if safe_prefix else guarded, True)

    # Apply the safety policy at action-line granularity.  Replacing an entire
    # semantic Chunk because its final paragraph contains a risky operation
    # loses the safe checks and structural heading that precede it.  It also
    # turns multiple source sections into identical facts that are later
    # deduplicated.  Line-level redaction preserves complete source structure
    # while withholding only the executable detail.
    guarded, changed = _guard_high_risk_action_lines(value)
    if changed:
        return guarded, True

    if re.search(r"更换(?:m2|pci接口)?网卡|更换网线", lowered):
        return (
            "以下为文档中的硬件更换说明；断电、确认备件型号并取得人工确认后才可执行。\n"
            + value,
            True,
        )
    return value, False


def _guard_high_risk_action_lines(text: str) -> tuple[str, bool]:
    output: list[str] = []
    changed = False
    guarded_categories: set[str] = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        category = _high_risk_action_category(line)
        if not category:
            output.append(line)
            continue
        changed = True
        guard_key = category
        if category == "internal_hardware" and re.search(
            r"(?:测试|验证|观察|判断)",
            line,
        ):
            # Preserve the decision purpose once even when an earlier line in
            # the same chunk already emitted the generic hardware safety gate.
            guard_key = f"{category}:decision"
        # One category-level gate per semantic fact is sufficient.  Repeating
        # the same disclaimer after every safe observation makes a source-
        # complete answer unreadable without adding any safety information.
        if guard_key not in guarded_categories:
            output.append(_guarded_action_summary(line, category))
            guarded_categories.add(guard_key)
    return "\n".join(output), changed


def _high_risk_action_category(line: str) -> str:
    lowered = str(line or "").lower()
    if re.search(
        r"(?:\bdiskpart\b|\blist\s+(?:disk|partition|volume)\b|"
        r"\bsel(?:ect)?\s+(?:disk|partition|volume)\b|"
        r"\bassign(?:\s+letter)?\b|\b(?:active|clean|convert)\b|"
        r"\bdelete\s+partition\b|\bbootrec\b|\bbcdboot\b|"
        r"断开所有非系统(?:盘|硬盘)|拔(?:下|掉)所有(?:非系统)?硬盘|"
        r"清除[^。\n]{0,20}引导分区|"
        r"格式化[^。\n]{0,20}(?:磁盘|分区|u盘)|修复引导记录)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return "storage"
    if re.search(
        r"(?:clr[_ -]?cmos|jbat1|清除\s*cmos|重置\s*(?:cmos|bios)|"
        r"(?:取出|取下|扣下|移除|更换)[^。\n]{0,25}(?:主板[^。\n]{0,10})?"
        r"(?:纽扣|cmos|bios)\s*电池)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return "firmware"
    if (
        "bios" in lowered
        and re.search(r"(?:修改|更改|设为|设置为|开启|禁用|恢复|更新|刷写)", lowered)
    ):
        return "firmware"
    if re.search(
        r"(?:万用表[^。\n]{0,30}(?:插座|市电|电压)|带电[^。\n]{0,20}(?:测量|检查)|"
        r"电源内部|拆开电源)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return "electrical"
    if re.search(
        r"(?:打开机箱|拆开机箱|需开箱|短接|机箱内部|最小化硬件|"
        r"(?:拔下|拔掉|移除|拆除|重新插拔|重新安装|只插(?:入)?|"
        r"逐一添加|每次只添加|更换|替换)[^。\n]{0,50}"
        r"(?:内存(?:条)?|cpu|主板|(?:独立)?显卡|扩展卡|板卡|硬盘|"
        r"m\.?2|sata(?:端口|线)?|内部电源线|内部数据线|前面板连接线)|"
        r"(?:内存(?:条)?|(?:独立)?显卡|扩展卡|板卡|硬盘)[^。\n]{0,35}"
        r"(?:拔下|拔掉|移除|拆除|重新插拔|重新安装|逐一添加)|"
        r"拔(?:下|掉)[^。\n]{0,35}(?:内部电源线|数据线|机箱前面板)|"
        r"金手指|橡皮擦|(?:气枪|软毛刷)[^。\n]{0,30}(?:主板|插槽)|"
        r"重新涂抹[^。\n]{0,20}硅脂|每次只添加一个硬件|"
        r"更换[^。\n]{0,35}(?:内存(?:条)?|cpu|主板|显卡|扩展卡|"
        r"电源(?:模块|供应器|本体|（psu）|\(psu\)|psu)?)(?:[。；，,]|$)|"
        r"psu\s*测试仪|(?:观察|检查)主板[^。\n]{0,25}(?:背面|电容|铜柱)|"
        r"(?:使用|通过)[^。\n]{0,30}电源[^。\n]{0,20}(?:替换|测试)|"
        r"检查内部灰尘|"
        r"(?:散热器|机箱|主板|电源)[^。\n]{0,35}灰尘[^。\n]{0,20}(?:清理|清除)|"
        r"(?:清理|清除)[^。\n]{0,25}(?:机箱|主板|电源)[^。\n]{0,15}灰尘)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return "internal_hardware"
    return ""


def _guarded_action_summary(line: str, category: str) -> str:
    label_match = re.match(r"^([^：:]{2,30})[：:]", str(line or "").strip())
    label = label_match.group(1).strip() if label_match else ""
    if re.search(
        r"(?:打开|拆|拔|移除|重新|更换|替换|清理|清除|短接|扣下|"
        r"最小化|修复|测量|刷写)",
        label,
    ):
        label = ""
    if category == "storage" and re.search(
        r"\b(?:diskpart|list|sel(?:ect)?|assign|active|clean|convert|"
        r"delete|bootrec|bcdboot)\b",
        label,
        flags=re.IGNORECASE,
    ):
        label = ""
    prefix = f"{label}：" if label else ""
    if category == "storage":
        return (
            f"{prefix}涉及断开磁盘、修改启动分区或重建引导，可能造成数据或启动配置损失；"
            "执行前必须确认系统盘、启动模式和目标分区，并取得人工确认。本回答不展开命令。"
        )
    if category == "firmware":
        return (
            f"{prefix}涉及 BIOS/CMOS 修改或复位；必须确认主板型号、记录现有配置、完全断电，"
            "并由有资质人员取得人工确认后执行。本回答不展开跳线、短接或刷写步骤。"
        )
    if category == "electrical":
        return (
            f"{prefix}涉及市电或带电测量，存在触电和设备损坏风险；"
            "仅可由具备资质的人员使用合规仪表，在批准的安全条件下执行。"
        )
    if category == "internal_hardware" and re.search(
        r"(?:测试|验证|观察|判断)",
        str(line or ""),
    ):
        return (
            f"{prefix}文档包含硬件隔离或交叉验证，用于区分部件、插槽/接口及连接问题；"
            "该验证涉及机箱内部拆装，必须先完全断电并做好防静电，"
            "由有资质人员取得人工确认后执行。本回答保留判定目的，不展开拆装细节。"
        )
    return (
        f"{prefix}涉及机箱内部、部件拆装或硬件替换；必须先完全断电并做好防静电，"
        "由有资质人员取得人工确认后执行。本回答不展开拆装细节。"
    )


def _compact_repeated_safety_guards(facts: list[dict[str, Any]]) -> None:
    """Render one full safety gate per category across the whole answer.

    Semantic source chunks are guarded independently before they are merged.
    A long hardware document can therefore repeat the same full disclaimer in
    many stages.  Preserve the first complete gate, then replace later copies
    with a short, explicit reference to that gate.  The source fact and its
    citation stay present, but the answer no longer reads like eight copies of
    the same warning.
    """

    patterns = (
        (
            "storage",
            re.compile(
                r"(?:[^：:\n]{2,30}[：:])?涉及断开磁盘、修改启动分区或重建引导，"
                r"可能造成数据或启动配置损失；执行前必须确认系统盘、启动模式和目标分区，"
                r"并取得人工确认。本回答不展开命令。"
            ),
            "本节涉及同类磁盘或引导高风险操作，沿用前述安全前置条件；未确认前不展开命令。",
        ),
        (
            "firmware",
            re.compile(
                r"(?:[^：:\n]{2,30}[：:])?涉及 BIOS/CMOS 修改或复位；"
                r"必须确认主板型号、记录现有配置、完全断电，"
                r"并由有资质人员取得人工确认后执行。本回答不展开跳线、短接或刷写步骤。"
            ),
            "本节涉及同类 BIOS/CMOS 高风险操作，沿用前述安全前置条件；未确认前不展开执行细节。",
        ),
        (
            "electrical",
            re.compile(
                r"(?:[^：:\n]{2,30}[：:])?涉及市电或带电测量，存在触电和设备损坏风险；"
                r"仅可由具备资质的人员使用合规仪表，在批准的安全条件下执行。"
            ),
            "本节涉及同类市电或带电测量，沿用前述资质与安全前置条件。",
        ),
        (
            "internal_hardware",
            re.compile(
                r"(?:[^：:\n]{2,30}[：:])?涉及机箱内部、部件拆装或硬件替换；"
                r"必须先完全断电并做好防静电，"
                r"由有资质人员取得人工确认后执行。本回答不展开拆装细节。"
            ),
            "本节涉及同类机箱内部或部件拆装操作，沿用前述安全前置条件；未确认前不展开执行细节。",
        ),
    )
    seen: set[str] = set()
    for fact in facts:
        text = str(fact.get("text") or "")
        for category, pattern, compact in patterns:
            if not pattern.search(text):
                continue
            if category in seen:
                without_repeat = pattern.sub("", text).strip()
                # Most source sections also contain safe observations and a
                # structural heading.  In that case the repeated disclaimer
                # adds no information and can be omitted entirely.  Keep a
                # short reference only when removing it would empty the fact.
                text = without_repeat or compact
            else:
                seen.add(category)
        fact["text"] = text


def _summarize_symptom_branches(text: str) -> str:
    """Turn a long symptom/check-point continuation into one item per branch."""

    branches: list[tuple[str, list[str]]] = []
    current_title = ""
    current_details: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_details
        if current_title:
            branches.append((current_title, current_details))
        current_title = ""
        current_details = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip(" \t-；;")
        if not line:
            continue
        symptom = re.match(r"^现象[：:]\s*(.+)$", line)
        if symptom:
            flush()
            current_title = symptom.group(1).strip()
            continue
        if not current_title:
            continue
        if re.match(r"^(?:检查点|行动|操作步骤|详细现象参考)[：:]?$", line):
            continue
        line = re.sub(r"[（(]\s*阶段\s*\d+[^）)]*[）)]", "", line)
        line = re.sub(r"\s*-\s*阶段\s*\d+", "", line)
        if "断开所有非系统盘" in line:
            line = (
                "核对 BIOS 是否识别系统盘及启动顺序；"
                "涉及断开硬盘的测试需人工确认"
            )
        elif "回到最小化系统" in line:
            line = "最小化硬件测试及逐步恢复硬件需人工确认"
        elif "bios恢复" in line.lower():
            line = "BIOS 恢复属于高级操作，需确认主板型号和恢复方案"
        line = line.rstrip("。；; ")
        if line and line not in current_details:
            current_details.append(line)
    flush()

    lines = [
        "特定现象分支（涉及拆装时需人工确认）",
        "先依据电源指示灯、风扇、显示、蜂鸣码和主板 Debug 灯选择分支；"
        "涉及机箱内部供电、内存、CPU、扩展卡或 CMOS 的操作，仅保留为待确认检查项。",
    ]
    for title, details in branches:
        # Keep every branch while bounding repetitive check-point prose.
        selected = details[:8]
        suffix = "；".join(selected)
        lines.append(f"现象“{title}”：{suffix}" if suffix else f"现象“{title}”")
    return "\n".join(lines)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _dedupe_media_refs(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        key = (
            str(value.get("media_kind") or ""),
            str(value.get("content_hash") or value.get("asset_path") or value.get("archive_path") or ""),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result
