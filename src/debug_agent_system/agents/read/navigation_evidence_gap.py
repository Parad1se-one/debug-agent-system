"""Explain query facets that stop at a recalled document-navigation entry."""

from __future__ import annotations

from typing import Any

from debug_agent_system.core.contracts import AnswerSection, SessionState
from debug_agent_system.knowledge_v2.query_scope import task_facet_matches_text
from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel


def navigation_evidence_gap_section(
    *,
    model: KGV2ReadModel,
    state: SessionState,
    evidence_pack: dict[str, Any],
) -> tuple[AnswerSection | None, list[dict[str, Any]]]:
    """Build a source-bound gap for recalled-but-unclosed query facets."""

    query_scope = dict(evidence_pack.get("query_scope") or {})
    unsupported_ids = {
        str(item)
        for item in query_scope.get("unsupported_facets") or []
        if str(item)
    }
    facets = [
        dict(item)
        for item in query_scope.get("facets") or []
        if isinstance(item, dict)
        and str(item.get("facet_id") or "") in unsupported_ids
    ]
    if not facets:
        return None, []

    retrieval = dict(state.metadata.get("retrieval") or {})
    chunks = [
        dict(item)
        for item in retrieval.get("supporting_chunks") or []
        if isinstance(item, dict)
    ]
    resolved_links = _resolved_child_links(model)
    gaps: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        document = model.get(document_id) if document_id else None
        if not isinstance(document, dict):
            continue
        chunk_text = str(chunk.get("text") or "")
        for link in document.get("source_links") or []:
            if not isinstance(link, dict):
                continue
            link_text = str(link.get("link_text") or "")
            context = " ".join((
                link_text,
                str(link.get("source_context") or ""),
            )).strip()
            # A link elsewhere in a large document does not prove that the
            # read side saw it; the recalled chunk must contain the entry.
            if not link_text or link_text not in chunk_text:
                continue
            for facet in facets:
                if not task_facet_matches_text(facet, context):
                    continue
                facet_id = str(facet.get("facet_id") or "")
                token = str(link.get("wiki_token") or "")
                target_url = str(link.get("target_url") or "")
                key = (facet_id, document_id, token or target_url or link_text)
                if key in seen:
                    continue
                seen.add(key)
                resolved = (
                    (document_id, token) in resolved_links
                    or (document_id, target_url) in resolved_links
                )
                relationship_id = str(link.get("relationship_id") or "")
                reason = (
                    "linked_child_content_not_recalled"
                    if resolved
                    else "linked_child_not_indexed"
                )
                anchor = f"，锚点 {relationship_id}" if relationship_id else ""
                title = str(
                    document.get("title") or chunk.get("source_label") or ""
                )
                gaps.append({
                    "text": (
                        f"系统已识别并召回“{link_text}”导航入口"
                        f"（来源：{title}{anchor}），但当前已批准证据中没有"
                        f"支持“{facet.get('label') or facet_id}”的子文档正文，"
                        "因此该任务仍未形成证据闭包；现有回答只组织已取得"
                        "的资料，不补写缺失步骤。"
                    ),
                    "facet_id": facet_id,
                    "facet_kind": str(facet.get("kind") or ""),
                    "facet_label": str(facet.get("label") or ""),
                    "reason": reason,
                    "parent_document_id": document_id,
                    "parent_title": title,
                    "parent_source_path": str(document.get("source_path") or ""),
                    "relationship_id": relationship_id,
                    "link_text": link_text,
                    "target_url": target_url,
                    "wiki_token": token,
                    "chunk_ids": [str(chunk.get("chunk_id") or "")],
                    "evidence_ids": [],
                    "sources": [title],
                })
    if not gaps:
        return None, []
    chunk_ids = list(dict.fromkeys(
        chunk_id
        for item in gaps
        for chunk_id in item.get("chunk_ids") or []
        if chunk_id
    ))
    return AnswerSection(
        section_type="evidence_gap",
        title="资料缺口",
        items=gaps,
        chunk_ids=chunk_ids,
    ), gaps


def insert_navigation_evidence_gap(
    sections: list[AnswerSection],
    section: AnswerSection | None,
) -> list[AnswerSection]:
    """Insert before uncertainty/questions and replace any stale gap section."""

    if section is None:
        return sections
    kept = [item for item in sections if item.section_type != "evidence_gap"]
    index = next(
        (
            position
            for position, item in enumerate(kept)
            if item.section_type in {"uncertainty", "required_info", "sources"}
        ),
        len(kept),
    )
    kept.insert(index, section)
    return kept


def _resolved_child_links(model: KGV2ReadModel) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    for parent_id, edges in model.outgoing.items():
        for edge in edges:
            if str(edge.get("relation") or "") != "has_child_document":
                continue
            for token in edge.get("wiki_tokens") or []:
                if str(token):
                    links.add((parent_id, str(token)))
            for url in edge.get("target_urls") or []:
                if str(url):
                    links.add((parent_id, str(url)))
    return links


__all__ = [
    "insert_navigation_evidence_gap",
    "navigation_evidence_gap_section",
]
