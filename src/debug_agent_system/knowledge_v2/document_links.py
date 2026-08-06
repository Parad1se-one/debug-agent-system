"""Extract and resolve cross-document links from canonical knowledge sources.

DOCX exports preserve Feishu hyperlinks in ``word/document.xml.rels`` even
when the visible document is only a navigation table.  This module turns
those links into auditable KG relations without relying on display titles as
document identity.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET
import zipfile


_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_COPY_SUFFIX_RE = re.compile(r"\s*[\(（]\d+[\)）]\s*$")
_WIKI_TOKEN_RE = re.compile(r"/wiki/([A-Za-z0-9]+)")
_TITLE_NOISE_RE = re.compile(r"[\s_/\\|｜·•：:，,。.!！?？\"“”'‘’（）()\[\]【】<>\-—–]+")

# Four exported documents have deliberately duplicated branch titles.  Their
# Feishu node tokens are the stable identities; ``(1)`` is only the local
# exporter's filename collision suffix.
REVIEWED_WIKI_TARGET_OVERRIDES: dict[str, str] = {
    "ENaFwPWgji30elki5UAcTLl2nxc": "可以进系统.docx",
    "SzFBwdOYiijYV1kBTRAcgzbkn3G": "无法进入系统.docx",
    "MCwGwfrw6iWn1dkRvNsc2R8Xnid": "可以进入系统.docx",
    "NUfmw7PyKiYWGLkEMhqcM9ndnzn": "无法进入系统 (1).docx",
}

DOCUMENT_LINK_RELATIONS = {"has_child_document", "references_document"}


def extract_docx_hyperlinks(path: str | Path) -> list[dict[str, Any]]:
    """Return ordered external hyperlinks with their visible paragraph context."""

    source = Path(path)
    if source.suffix.lower() != ".docx" or not source.is_file():
        return []
    try:
        with zipfile.ZipFile(source) as archive:
            document_xml = archive.read("word/document.xml")
            rel_path = "word/_rels/document.xml.rels"
            if rel_path not in archive.namelist():
                return []
            relations_xml = archive.read(rel_path)
    except (KeyError, OSError, zipfile.BadZipFile):
        return []
    try:
        document_root = ET.fromstring(document_xml)
        relation_root = ET.fromstring(relations_xml)
    except ET.ParseError:
        return []
    targets = {
        str(item.attrib.get("Id") or ""): str(item.attrib.get("Target") or "")
        for item in relation_root
    }
    links: list[dict[str, Any]] = []
    order = 0
    paragraph_order = 0
    for paragraph in document_root.iter(f"{_WORD_NS}p"):
        paragraph_order += 1
        paragraph_text = _normalize_space(
            "".join(node.text or "" for node in paragraph.iter(f"{_WORD_NS}t"))
        )
        for hyperlink in paragraph.iter(f"{_WORD_NS}hyperlink"):
            relationship_id = str(hyperlink.attrib.get(f"{_REL_NS}id") or "")
            target_url = targets.get(relationship_id, "")
            if not target_url.startswith(("http://", "https://")):
                continue
            link_text = _normalize_space(
                "".join(node.text or "" for node in hyperlink.iter(f"{_WORD_NS}t"))
            )
            order += 1
            links.append({
                "relationship_id": relationship_id,
                "link_order": order,
                "paragraph_order": paragraph_order,
                "link_text": link_text,
                "source_context": paragraph_text,
                "target_url": canonical_document_url(target_url),
                "wiki_token": wiki_token(target_url),
                "standalone": bool(link_text and link_text == paragraph_text),
            })
    return links


def canonical_document_url(value: str) -> str:
    """Strip query/fragment noise while retaining the stable document URL."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def wiki_token(value: str) -> str:
    match = _WIKI_TOKEN_RE.search(str(value or ""))
    return match.group(1) if match else ""


def normalized_document_title(value: str, *, strip_copy_suffix: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"\.(?:docx?|pdf|md|txt|xlsx?|pptx?)$", "", text, flags=re.IGNORECASE)
    if strip_copy_suffix:
        text = _COPY_SUFFIX_RE.sub("", text)
    return _TITLE_NOISE_RE.sub("", text).casefold()


def build_document_link_graph(
    repo_root: str | Path,
    documents: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve every DOCX hyperlink that targets another canonical document.

    The returned relations are aggregated by ``(parent, child, relation)``.
    Unresolved and self-referential links remain in the report so publication
    is complete and auditable without inventing a target.
    """

    root = Path(repo_root)
    rows = [dict(item) for item in documents if isinstance(item, dict)]
    document_by_title = {
        str(item.get("title") or ""): item for item in rows
        if str(item.get("title") or "")
    }
    extracted_by_parent: dict[str, list[dict[str, Any]]] = {}
    token_labels: dict[str, set[str]] = defaultdict(set)
    source_status: list[dict[str, Any]] = []
    for document in rows:
        document_id = str(document.get("document_id") or "")
        source_path = str(document.get("source_path") or "")
        path = Path(source_path)
        if not path.is_absolute():
            path = root / path
        links = extract_docx_hyperlinks(path)
        extracted_by_parent[document_id] = links
        for link in links:
            token = str(link.get("wiki_token") or "")
            label = str(link.get("link_text") or "")
            if token and label:
                token_labels[token].add(label)
        source_status.append({
            "document_id": document_id,
            "title": str(document.get("title") or ""),
            "source_path": source_path,
            "source_exists": path.is_file(),
            "external_link_count": len(links),
            "feishu_wiki_link_count": sum(
                bool(str(item.get("wiki_token") or "")) for item in links
            ),
        })

    strict_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relaxed_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in rows:
        title = str(document.get("title") or "")
        strict_index[normalized_document_title(title)].append(document)
        relaxed_index[normalized_document_title(title, strip_copy_suffix=True)].append(document)

    unresolved: list[dict[str, Any]] = []
    external_resources: list[dict[str, Any]] = []
    self_references: list[dict[str, Any]] = []
    resolved_occurrences: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for parent in rows:
        parent_id = str(parent.get("document_id") or "")
        links = extracted_by_parent.get(parent_id, [])
        navigation_parent = _is_navigation_parent(parent, links)
        for link in links:
            token = str(link.get("wiki_token") or "")
            if not token:
                external_resources.append(
                    _audit_link(parent, link, "external_resource")
                )
                continue
            target, resolution = _resolve_target(
                token=token,
                link_text=str(link.get("link_text") or ""),
                token_labels=token_labels,
                document_by_title=document_by_title,
                strict_index=strict_index,
                relaxed_index=relaxed_index,
            )
            if target is None:
                unresolved.append(_audit_link(parent, link, resolution))
                continue
            target_id = str(target.get("document_id") or "")
            occurrence = {
                **_audit_link(parent, link, "resolved"),
                "target_document_id": target_id,
                "target_title": str(target.get("title") or ""),
                "target_source_path": str(target.get("source_path") or ""),
                "resolution_method": resolution,
            }
            if target_id == parent_id:
                self_references.append({**occurrence, "status": "self_reference"})
                continue
            relation = "has_child_document" if navigation_parent else "references_document"
            occurrence["relation"] = relation
            resolved_occurrences.append(occurrence)
            grouped[(parent_id, target_id, relation)].append(occurrence)

    relations: list[dict[str, Any]] = []
    for (parent_id, target_id, relation), occurrences in sorted(grouped.items()):
        occurrences.sort(key=lambda item: int(item.get("link_order") or 0))
        relations.append({
            "from": parent_id,
            "to": target_id,
            "relation": relation,
            "link_texts": _dedupe(item.get("link_text") for item in occurrences),
            "relationship_ids": _dedupe(
                item.get("relationship_id") for item in occurrences
            ),
            "target_urls": _dedupe(item.get("target_url") for item in occurrences),
            "wiki_tokens": _dedupe(item.get("wiki_token") for item in occurrences),
            "source_contexts": _dedupe(item.get("source_context") for item in occurrences),
            "link_orders": [
                int(item.get("link_order") or 0) for item in occurrences
            ],
            "resolution_methods": _dedupe(
                item.get("resolution_method") for item in occurrences
            ),
        })

    return relations, {
        "document_count": len(rows),
        "source_status": source_status,
        "external_link_count": sum(
            len(items) for items in extracted_by_parent.values()
        ),
        "feishu_wiki_link_count": sum(
            bool(str(link.get("wiki_token") or ""))
            for items in extracted_by_parent.values()
            for link in items
        ),
        "resolved_occurrence_count": len(resolved_occurrences),
        "resolved_relation_count": len(relations),
        "child_relation_count": sum(
            item.get("relation") == "has_child_document" for item in relations
        ),
        "reference_relation_count": sum(
            item.get("relation") == "references_document" for item in relations
        ),
        "unresolved_count": len(unresolved),
        "external_resource_count": len(external_resources),
        "self_reference_count": len(self_references),
        "unresolved": unresolved,
        "external_resources": external_resources,
        "self_references": self_references,
        "resolved_occurrences": resolved_occurrences,
    }


def _resolve_target(
    *,
    token: str,
    link_text: str,
    token_labels: dict[str, set[str]],
    document_by_title: dict[str, dict[str, Any]],
    strict_index: dict[str, list[dict[str, Any]]],
    relaxed_index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    override = REVIEWED_WIKI_TARGET_OVERRIDES.get(token)
    if override:
        target = document_by_title.get(override)
        return (target, "reviewed_wiki_token_override") if target else (None, "override_target_missing")

    labels = _dedupe([link_text, *sorted(token_labels.get(token) or [])])
    strict_candidates: dict[str, dict[str, Any]] = {}
    for label in labels:
        for item in strict_index.get(normalized_document_title(label), []):
            strict_candidates[str(item.get("document_id") or "")] = item
    if len(strict_candidates) == 1:
        return next(iter(strict_candidates.values())), "wiki_token_global_exact_title"
    if len(strict_candidates) > 1:
        preferred = _prefer_base_export(strict_candidates.values())
        if preferred is not None:
            return preferred, "wiki_token_exact_title_prefer_base_export"
        return None, "ambiguous_exact_title"

    relaxed_candidates: dict[str, dict[str, Any]] = {}
    for label in labels:
        for item in relaxed_index.get(
            normalized_document_title(label, strip_copy_suffix=True), []
        ):
            relaxed_candidates[str(item.get("document_id") or "")] = item
    if len(relaxed_candidates) == 1:
        return next(iter(relaxed_candidates.values())), "wiki_token_relaxed_title"
    if len(relaxed_candidates) > 1:
        preferred = _prefer_base_export(relaxed_candidates.values())
        if preferred is not None:
            return preferred, "wiki_token_relaxed_title_prefer_base_export"
        return None, "ambiguous_relaxed_title"

    scored: list[tuple[float, dict[str, Any]]] = []
    label_keys = [
        normalized_document_title(label, strip_copy_suffix=True)
        for label in labels if normalized_document_title(label, strip_copy_suffix=True)
    ]
    for candidates in relaxed_index.values():
        for item in candidates:
            title_key = normalized_document_title(
                str(item.get("title") or ""), strip_copy_suffix=True
            )
            score = max(
                (_containment_score(label_key, title_key) for label_key in label_keys),
                default=0.0,
            )
            if score >= 0.82:
                scored.append((score, item))
    if not scored:
        return None, "no_local_title_match"
    scored.sort(
        key=lambda pair: (
            -pair[0],
            _has_copy_suffix(str(pair[1].get("title") or "")),
            str(pair[1].get("title") or ""),
        )
    )
    best_score = scored[0][0]
    best = {
        str(item.get("document_id") or ""): item
        for score, item in scored if abs(score - best_score) < 1e-9
    }
    if len(best) == 1:
        return next(iter(best.values())), "wiki_token_global_containment_title"
    preferred = _prefer_base_export(best.values())
    if preferred is not None:
        return preferred, "wiki_token_containment_prefer_base_export"
    return None, "ambiguous_containment_title"


def _is_navigation_parent(
    document: dict[str, Any],
    links: list[dict[str, Any]],
) -> bool:
    if str(document.get("document_kind") or "") == "document_index_doc":
        return True
    if len(links) < 2:
        return False
    standalone = sum(bool(item.get("standalone")) for item in links)
    return standalone / len(links) >= 0.75


def _audit_link(
    parent: dict[str, Any],
    link: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "parent_document_id": str(parent.get("document_id") or ""),
        "parent_title": str(parent.get("title") or ""),
        "parent_source_path": str(parent.get("source_path") or ""),
        "relationship_id": str(link.get("relationship_id") or ""),
        "link_order": int(link.get("link_order") or 0),
        "paragraph_order": int(link.get("paragraph_order") or 0),
        "link_text": str(link.get("link_text") or ""),
        "source_context": str(link.get("source_context") or ""),
        "target_url": str(link.get("target_url") or ""),
        "wiki_token": str(link.get("wiki_token") or ""),
        "status": status,
    }


def _containment_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) < 4 or shorter not in longer:
        return 0.0
    return len(shorter) / len(longer)


def _prefer_base_export(items: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rows = list(items)
    base = [
        item for item in rows
        if not _has_copy_suffix(str(item.get("title") or ""))
    ]
    return base[0] if len(base) == 1 else None


def _has_copy_suffix(value: str) -> bool:
    title = re.sub(r"\.(?:docx?|pdf|md|txt)$", "", str(value or ""), flags=re.IGNORECASE)
    return bool(_COPY_SUFFIX_RE.search(title))


def _normalize_space(value: str) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _dedupe(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


__all__ = [
    "DOCUMENT_LINK_RELATIONS",
    "REVIEWED_WIKI_TARGET_OVERRIDES",
    "build_document_link_graph",
    "canonical_document_url",
    "extract_docx_hyperlinks",
    "normalized_document_title",
    "wiki_token",
]
