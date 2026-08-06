"""Deterministically rebuild semantic answer chunks from KG_v2 source files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile


_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_W_NS}}}"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_V_NS = "urn:schemas-microsoft-com:vml"
_O_NS = "urn:schemas-microsoft-com:office:office"
_PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_SPACE = re.compile(r"\s+")
_HEADING_PREFIX = re.compile(
    r"^[（(]?(?:情况\s*\d+|第[一二三四五六七八九十百]+[章节步]|\d+(?:\.\d+)*)[）).、：:\s-]*"
)
_SUMMARY_SPLIT = re.compile(r"[；;\n]+")
_QUESTION = re.compile(r"(?:[？?]$|^(?:问题|问)\s*[：:])")
_SEMANTIC_NUMBERED_HEADING = re.compile(
    r"^(?:方案|方法|情况|结论)\s*(?:[一二三四五六七八九十百]+|\d+)\s*[：:]"
)
_FIGURE_CAPTION = re.compile(
    r"^(?:图|figure|fig\.?)\s*"
    r"(?:\d+(?:[-.]\d+)*|[一二三四五六七八九十百]+)"
    r"(?:\s*[：:]\s*.+)?$",
    re.IGNORECASE,
)
_MAX_CHARS = 600
_MAX_BLOCKS = 24
_MAX_TABLE_ROWS = 8

SOURCE_CHUNK_MANIFEST_SCHEMA = "kg_v2.source_chunk_manifest.v1"
SOURCE_CHUNKER_VERSION = "section-faq-table-media-list-figure-anchor.v5"


@dataclass(frozen=True, slots=True)
class _SourceBlock:
    text: str
    kind: str = "paragraph"
    style_name: str = ""
    heading_level: int | None = None
    table_group: int | None = None
    media_refs: tuple[dict[str, Any], ...] = ()
    list_level: int | None = None
    list_style: str = ""
    list_marker: str = ""


def _serialize_source_block(block: _SourceBlock) -> dict[str, Any]:
    """Keep presentation semantics without duplicating binary media payloads."""

    return {
        "text": block.text,
        "kind": block.kind,
        "heading_level": block.heading_level,
        "list_level": block.list_level,
        "list_style": block.list_style,
        "list_marker": block.list_marker,
        "media_keys": _dedupe(
            str(item.get("content_hash") or item.get("media_id") or item.get("archive_path") or "")
            for item in block.media_refs
        ),
    }


def build_staged_chunk_manifest(
    source_path: str | Path,
    structured_sections: Iterable[dict[str, Any]],
    *,
    source_doc_title: str = "",
) -> dict[str, Any]:
    """Build a review-only chunk manifest directly from one source file.

    The manifest is deliberately not a KG object and every staged chunk is
    unapproved.  W10 later replaces the W9 source section ids with the draft
    KnowledgeSection ids; the online SAG is still rebuilt only from the
    approved canonical graph and hash-pinned source files.
    """

    path = Path(source_path)
    source_file_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    normalized_path = str(path)
    staged_document_id = (
        "staged-document:"
        + hashlib.sha256(f"{normalized_path}:{source_file_hash}".encode("utf-8")).hexdigest()[:24]
    )
    source_sections = [item for item in structured_sections if isinstance(item, dict)]
    mapped_sections: list[dict[str, Any]] = []
    for index, section in enumerate(source_sections, start=1):
        source_section_id = str(section.get("section_id") or f"section:{index}")
        body_lines = [str(item).strip() for item in section.get("body_lines") or [] if str(item).strip()]
        mapped_sections.append({
            "section_id": source_section_id,
            "document_id": staged_document_id,
            "heading": str(section.get("section_title") or f"章节 {index}"),
            "section_order": index,
            "summary": "；".join(body_lines) or str(section.get("section_title") or ""),
            "source_offsets": [source_section_id],
        })
    document = {
        "document_id": staged_document_id,
        "title": source_doc_title or path.name,
        "source_path": normalized_path,
        "content_hash": source_file_hash,
        "source_kind": "raw_doc",
        "approved": False,
    }
    chunks, stats = rebuild_source_chunks(Path("."), [document], mapped_sections)
    for chunk in chunks:
        chunk["staging_status"] = "pending_review"
    content = {
        "schema_version": SOURCE_CHUNK_MANIFEST_SCHEMA,
        "chunker_version": SOURCE_CHUNKER_VERSION,
        "binding_status": "source_sections",
        "source_path": normalized_path,
        "source_file_hash": source_file_hash,
        "staged_document_id": staged_document_id,
        "chunks": chunks,
        "stats": {
            **stats,
            "chunk_count": len(chunks),
        },
    }
    manifest_hash = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **content,
        "manifest_id": f"chunk-manifest:{manifest_hash[:24]}",
        "manifest_hash": manifest_hash,
    }


def rebuild_source_chunks(
    project_root: str | Path,
    documents: Iterable[dict[str, Any]],
    sections: Iterable[dict[str, Any]],
    *,
    asset_root: str | Path | None = None,
    media_assets: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse hash-pinned files into Section/FAQ/table-aware chunks."""

    root = Path(project_root)
    media_context_overrides = _media_context_overrides(media_assets)
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        document_id = str(section.get("document_id") or "")
        if document_id:
            by_document[document_id].append(section)
    for values in by_document.values():
        values.sort(key=lambda item: (int(item.get("section_order") or 0), str(item.get("section_id") or "")))

    chunks: list[dict[str, Any]] = []
    aligned_section_ids: set[str] = set()
    directly_aligned_section_ids: set[str] = set()
    all_section_ids = {
        str(item.get("section_id") or "")
        for values in by_document.values()
        for item in values
        if str(item.get("section_id") or "")
    }
    stats = {
        "source_document_count": 0,
        "source_document_missing_count": 0,
        "source_hash_mismatch_count": 0,
        "source_parse_failure_count": 0,
        "source_aligned_chunk_count": 0,
        "source_orphan_chunk_count": 0,
        "source_heading_chunk_count": 0,
        "source_table_chunk_count": 0,
        "source_media_chunk_count": 0,
        "source_image_count": 0,
        "source_attachment_count": 0,
    }
    for document in documents:
        document_id = str(document.get("document_id") or "")
        source_path = str(document.get("source_path") or "")
        if not document_id or not source_path:
            continue
        stats["source_document_count"] += 1
        path = root / source_path
        if not path.is_file():
            stats["source_document_missing_count"] += 1
            continue
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = str(document.get("content_hash") or "")
        hash_matches = not expected_hash or source_hash == expected_hash
        if not hash_matches:
            stats["source_hash_mismatch_count"] += 1
        blocks = _read_blocks(path, asset_root=Path(asset_root) if asset_root is not None else None)
        if not blocks:
            stats["source_parse_failure_count"] += 1
            continue
        if media_context_overrides:
            blocks = _apply_media_context_overrides(blocks, media_context_overrides)

        anchors = _align_sections(blocks, by_document.get(document_id, []))
        for bindings in anchors.values():
            for section, method in bindings:
                section_id = str(section.get("section_id") or "")
                if section_id:
                    aligned_section_ids.add(section_id)
                    if method.startswith("direct_"):
                        directly_aligned_section_ids.add(section_id)

        for start, end, bindings, semantic_label in _chunk_ranges(blocks, anchors, path):
            text = "\n".join(block.text for block in blocks[start:end]).strip()
            if not text:
                continue
            section_ids = _dedupe(
                str(section.get("section_id") or "") for section, _method in bindings
            )
            direct_section_ids = _dedupe(
                str(section.get("section_id") or "")
                for section, method in bindings
                if method.startswith("direct_")
            )
            methods = _dedupe(method for _section, method in bindings)
            primary_section = bindings[-1][0] if bindings else {}
            section_id = str(primary_section.get("section_id") or "")
            media_refs = _dedupe_media_refs(
                media
                for block in blocks[start:end]
                for media in block.media_refs
            )
            block_slice = blocks[start:end]
            content_blocks = [_serialize_source_block(block) for block in block_slice]
            chunk_hash = hashlib.sha256(json.dumps(
                {
                    "text": text,
                    "content_blocks": content_blocks,
                    "media": [
                        {
                            "content_hash": item.get("content_hash"),
                            "archive_path": item.get("archive_path"),
                            "media_kind": item.get("media_kind"),
                        }
                        for item in media_refs
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            stable_key = "|".join((
                document_id, source_hash, str(start + 1), str(end),
                ",".join(section_ids), chunk_hash,
            ))
            chunk_id = f"chunk:source:{hashlib.sha256(stable_key.encode('utf-8')).hexdigest()[:24]}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "section_id": section_id,
                "section_ids": section_ids,
                "direct_section_ids": direct_section_ids,
                "alignment_methods": methods,
                "source_path": source_path,
                "source_label": str(
                    semantic_label or primary_section.get("heading")
                    or document.get("title") or path.name
                ),
                "source_kind": str(document.get("source_kind") or "source_file"),
                "text": text,
                "source_offsets": [{
                    "source_path": source_path,
                    # Logical blocks are paragraphs or complete table rows.
                    "block_start": start + 1,
                    "block_end": end,
                    "paragraph_start": start + 1,
                    "paragraph_end": end,
                    "block_types": _dedupe(block.kind for block in block_slice),
                    "content_blocks": content_blocks,
                    "source_file_hash": source_hash,
                }],
                "content_hash": chunk_hash,
                "source_file_hash": source_hash,
                "media_refs": media_refs,
                "approved": bool(document.get("approved") is not False and hash_matches),
            })
            if section_ids:
                stats["source_aligned_chunk_count"] += 1
            else:
                stats["source_orphan_chunk_count"] += 1
            if any(block.heading_level is not None or _is_faq_question(block, path) for block in block_slice):
                stats["source_heading_chunk_count"] += 1
            if any(block.kind == "table_row" for block in block_slice):
                stats["source_table_chunk_count"] += 1
            if media_refs:
                stats["source_media_chunk_count"] += 1
                stats["source_image_count"] += sum(
                    item.get("media_kind") == "image" for item in media_refs
                )
                stats["source_attachment_count"] += sum(
                    item.get("media_kind") == "attachment" for item in media_refs
                )

    stats.update({
        "source_section_count": len(all_section_ids),
        "source_directly_aligned_section_count": len(directly_aligned_section_ids),
        "source_aligned_section_count": len(aligned_section_ids),
    })
    return chunks, stats


def build_media_asset_graph(
    project_root: str | Path,
    documents: Iterable[dict[str, Any]],
    sections: Iterable[dict[str, Any]],
    *,
    procedure_steps: Iterable[dict[str, Any]] = (),
    actions: Iterable[dict[str, Any]] = (),
    relations: Iterable[dict[str, Any]] = (),
    asset_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    """Build content-addressed media nodes and their canonical graph links.

    A ``MediaAsset`` represents one unique payload.  Every placement in a
    source document remains explicit in ``source_occurrences`` so content
    deduplication never hides repeated use across documents or sections.
    """

    root = Path(project_root).resolve()
    document_list = [dict(item) for item in documents if isinstance(item, dict)]
    section_list = [dict(item) for item in sections if isinstance(item, dict)]
    step_list = [dict(item) for item in procedure_steps if isinstance(item, dict)]
    action_list = [dict(item) for item in actions if isinstance(item, dict)]
    relation_list = [dict(item) for item in relations if isinstance(item, dict)]
    chunks, source_stats = rebuild_source_chunks(
        root,
        document_list,
        section_list,
        asset_root=asset_root,
    )

    document_by_source: dict[str, str] = {}
    for document in document_list:
        document_id = str(document.get("document_id") or "")
        source_path = str(document.get("source_path") or "")
        if document_id and source_path:
            document_by_source[source_path] = document_id
            document_by_source[str((root / source_path).resolve())] = document_id

    media: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    media_sections: dict[str, set[str]] = defaultdict(set)
    media_chunks: dict[str, set[str]] = defaultdict(set)
    media_actions: dict[str, set[str]] = defaultdict(set)
    media_contexts: dict[str, set[str]] = defaultdict(set)

    def register(
        ref: dict[str, Any],
        *,
        document_id: str = "",
        chunk_id: str = "",
        section_ids: Iterable[str] = (),
        action_id: str = "",
    ) -> None:
        content_hash = str(ref.get("content_hash") or "")
        media_id = str(ref.get("media_id") or "")
        if not media_id and content_hash:
            media_id = f"media:{content_hash[:24]}"
        if not media_id or not content_hash:
            return
        source_path = str(ref.get("source_path") or "")
        resolved_document_id = document_id or document_by_source.get(source_path, "")
        if not resolved_document_id and source_path:
            resolved_document_id = document_by_source.get(
                str((root / source_path).resolve()), ""
            )
        archive_path = str(ref.get("archive_path") or "")
        asset_path = str(ref.get("asset_path") or "")
        relative_path = str(ref.get("relative_path") or "")
        if asset_path and not relative_path:
            try:
                relative_path = str(Path(asset_path).resolve().relative_to(root))
            except ValueError:
                relative_path = ""
        item = media.setdefault(media_id, {
            "media_id": media_id,
            "media_kind": str(ref.get("media_kind") or "attachment"),
            "label": str(ref.get("label") or Path(archive_path).name or media_id),
            "content_hash": content_hash,
            "mime_type": str(ref.get("mime_type") or "application/octet-stream"),
            "asset_path": asset_path,
            "relative_path": relative_path,
            "source_occurrences": [],
            "document_ids": [],
            "section_ids": [],
            "procedure_step_ids": [],
            "action_ids": [],
            "source_chunk_ids": [],
            "context_labels": [],
            "approved": True,
        })
        if not item.get("asset_path") and asset_path:
            item["asset_path"] = asset_path
        if not item.get("relative_path") and relative_path:
            item["relative_path"] = relative_path
        context = str(ref.get("caption") or ref.get("context_label") or "")
        if context:
            media_contexts[media_id].add(context)
            current_label = str(item.get("label") or "")
            if _generic_media_label(current_label):
                item["label"] = context
        occurrence_key = (resolved_document_id or source_path, archive_path)
        occurrence = occurrences[media_id].setdefault(occurrence_key, {
            "document_id": resolved_document_id,
            "source_path": source_path,
            "archive_path": archive_path,
            "relationship_ids": [],
            "section_ids": [],
            "source_chunk_ids": [],
        })
        relationship_id = str(ref.get("relationship_id") or "")
        if relationship_id and relationship_id not in occurrence["relationship_ids"]:
            occurrence["relationship_ids"].append(relationship_id)
        normalized_sections = {
            str(value or "") for value in section_ids if str(value or "")
        }
        if normalized_sections:
            media_sections[media_id].update(normalized_sections)
            occurrence["section_ids"] = sorted(
                set(occurrence["section_ids"]) | normalized_sections
            )
        if chunk_id:
            media_chunks[media_id].add(chunk_id)
            occurrence["source_chunk_ids"] = sorted(
                set(occurrence["source_chunk_ids"]) | {chunk_id}
            )
        if action_id:
            media_actions[media_id].add(action_id)

    for chunk in chunks:
        chunk_sections = list(chunk.get("section_ids") or [])
        if not chunk_sections and str(chunk.get("section_id") or ""):
            chunk_sections = [str(chunk.get("section_id") or "")]
        for ref in chunk.get("media_refs") or []:
            if isinstance(ref, dict):
                register(
                    ref,
                    document_id=str(chunk.get("document_id") or ""),
                    chunk_id=str(chunk.get("chunk_id") or ""),
                    section_ids=chunk_sections,
                )

    for action in action_list:
        action_id = str(action.get("action_id") or "")
        for ref in action.get("curated_image_refs") or []:
            if isinstance(ref, dict):
                register(ref, action_id=action_id)

    graph_relations: list[dict[str, str]] = []
    steps_by_section: dict[str, set[str]] = defaultdict(set)
    for step in step_list:
        section_id = str(step.get("section_id") or "")
        step_id = str(step.get("procedure_step_id") or "")
        if section_id and step_id:
            steps_by_section[section_id].add(step_id)
    steps_by_action: dict[str, set[str]] = defaultdict(set)
    actions_by_step: dict[str, set[str]] = defaultdict(set)
    for relation in relation_list:
        if str(relation.get("relation") or "") != "candidate_action":
            continue
        step_id = str(relation.get("from") or "")
        action_id = str(relation.get("to") or "")
        if step_id and action_id:
            steps_by_action[action_id].add(step_id)
            actions_by_step[step_id].add(action_id)

    for media_id, item in media.items():
        occurrence_values = sorted(
            occurrences[media_id].values(),
            key=lambda value: (
                str(value.get("document_id") or ""),
                str(value.get("archive_path") or ""),
            ),
        )
        document_ids = {
            str(value.get("document_id") or "")
            for value in occurrence_values
            if str(value.get("document_id") or "")
        }
        section_ids = set(media_sections.get(media_id) or set())
        action_ids = set(media_actions.get(media_id) or set())
        step_ids = {
            step_id
            for section_id in section_ids
            for step_id in steps_by_section.get(section_id) or set()
        }
        for action_id in action_ids:
            step_ids.update(steps_by_action.get(action_id) or set())
        # SOP/document ingestion already reviews ProcedureStep -> Action
        # mappings as ``candidate_action`` edges.  Reuse that typed bridge so
        # source screenshots become directly traversable from the matching
        # execution actions, not only from their document sections.
        action_ids.update(
            action_id
            for step_id in step_ids
            for action_id in actions_by_step.get(step_id) or set()
        )
        item.update({
            "source_occurrences": occurrence_values,
            "document_ids": sorted(document_ids),
            "section_ids": sorted(section_ids),
            "procedure_step_ids": sorted(step_ids),
            "action_ids": sorted(action_ids),
            "source_chunk_ids": sorted(media_chunks.get(media_id) or set()),
            "context_labels": sorted(media_contexts.get(media_id) or set()),
        })
        graph_relations.extend(
            {"from": document_id, "to": media_id, "relation": "has_media"}
            for document_id in document_ids
        )
        graph_relations.extend(
            {"from": section_id, "to": media_id, "relation": "section_media"}
            for section_id in section_ids
        )
        graph_relations.extend(
            {"from": step_id, "to": media_id, "relation": "step_media"}
            for step_id in step_ids
        )
        graph_relations.extend(
            {
                "from": action_id,
                "to": media_id,
                "relation": "action_media",
                "binding_origin": (
                    "curated_action_ref"
                    if action_id in (media_actions.get(media_id) or set())
                    else "procedure_step_candidate_action"
                ),
            }
            for action_id in action_ids
        )

    deduped_relations: list[dict[str, str]] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in graph_relations:
        key = (relation["from"], relation["to"], relation["relation"])
        if key not in seen_relations:
            seen_relations.add(key)
            deduped_relations.append(relation)

    source_occurrence_count = sum(
        len(value) for value in occurrences.values()
    )
    stats = {
        **source_stats,
        "media_asset_count": len(media),
        "image_asset_count": sum(
            item.get("media_kind") == "image" for item in media.values()
        ),
        "attachment_asset_count": sum(
            item.get("media_kind") == "attachment" for item in media.values()
        ),
        "media_occurrence_count": source_occurrence_count,
        "image_occurrence_count": sum(
            1
            for media_id, values in occurrences.items()
            if media[media_id].get("media_kind") == "image"
            for _value in values.values()
        ),
        "attachment_occurrence_count": sum(
            1
            for media_id, values in occurrences.items()
            if media[media_id].get("media_kind") == "attachment"
            for _value in values.values()
        ),
        "document_with_media_count": len({
            str(value.get("document_id") or "")
            for values in occurrences.values()
            for value in values.values()
            if str(value.get("document_id") or "")
        }),
        "media_relation_count": len(deduped_relations),
        "media_without_document_count": sum(
            not item.get("document_ids") for item in media.values()
        ),
        "media_without_section_count": sum(
            not item.get("section_ids") for item in media.values()
        ),
        "curated_action_media_count": sum(
            bool(media_actions.get(media_id))
            for media_id in media
        ),
        "curated_action_relation_count": sum(
            len(media_actions.get(media_id) or set())
            for media_id in media
        ),
        "action_media_asset_count": sum(
            bool(item.get("action_ids")) for item in media.values()
        ),
        "derived_action_relation_count": sum(
            len(set(item.get("action_ids") or []) - (media_actions.get(media_id) or set()))
            for media_id, item in media.items()
        ),
    }
    return (
        sorted(media.values(), key=lambda item: str(item.get("media_id") or "")),
        deduped_relations,
        stats,
    )


def _read_blocks(path: Path, *, asset_root: Path | None = None) -> list[_SourceBlock]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx_blocks(path, asset_root=asset_root)
    if suffix == ".xlsx":
        return _read_xlsx_blocks(path)
    if suffix == ".pptx":
        return _read_pptx_blocks(path)
    if suffix in {".md", ".txt"}:
        return _read_text_blocks(path)
    return []


def _read_paragraphs(path: Path) -> list[str]:
    """Compatibility helper returning the semantic block text."""

    return [block.text for block in _read_blocks(path)]


def _read_text_blocks(path: Path) -> list[_SourceBlock]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    blocks: list[_SourceBlock] = []
    table_group = 0
    in_table = False
    for raw in lines:
        text = _clean(raw)
        if not text:
            in_table = False
            continue
        heading_match = re.match(r"^(#{1,6})\s+", raw)
        is_table = text.startswith("|") and text.endswith("|")
        if is_table and not in_table:
            table_group += 1
        in_table = is_table
        blocks.append(_SourceBlock(
            text=text,
            kind="table_row" if is_table else "heading" if heading_match else "paragraph",
            style_name="markdown_heading" if heading_match else "",
            heading_level=len(heading_match.group(1)) if heading_match else None,
            table_group=table_group if is_table else None,
        ))
    return _contextualize_media_blocks(blocks)


def _read_xlsx_blocks(path: Path) -> list[_SourceBlock]:
    """Read OOXML spreadsheet cell text without evaluating formulas."""

    try:
        with zipfile.ZipFile(path) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.iter():
                    if item.tag.endswith("}si"):
                        shared.append(_clean("".join(
                            node.text or ""
                            for node in item.iter()
                            if node.tag.endswith("}t")
                        )))
            sheet_paths = sorted(
                name
                for name in archive.namelist()
                if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
            )
            blocks: list[_SourceBlock] = []
            table_group = 0
            for sheet_index, sheet_path in enumerate(sheet_paths, start=1):
                table_group += 1
                blocks.append(_SourceBlock(
                    text=f"Sheet {sheet_index}",
                    kind="heading",
                    style_name="xlsx_sheet",
                    heading_level=1,
                ))
                sheet_root = ET.fromstring(archive.read(sheet_path))
                for row in sheet_root.iter():
                    if not row.tag.endswith("}row"):
                        continue
                    values: list[str] = []
                    for cell in row:
                        if not cell.tag.endswith("}c"):
                            continue
                        cell_type = str(cell.attrib.get("t") or "")
                        text_nodes = [
                            node.text or ""
                            for node in cell.iter()
                            if node.tag.endswith("}t")
                        ]
                        value_node = next((
                            node for node in cell
                            if node.tag.endswith("}v")
                        ), None)
                        value = _clean("".join(text_nodes))
                        if not value and value_node is not None:
                            raw = str(value_node.text or "")
                            if cell_type == "s":
                                try:
                                    value = shared[int(raw)]
                                except (ValueError, IndexError):
                                    value = raw
                            else:
                                value = raw
                        if value:
                            values.append(value)
                    if values:
                        blocks.append(_SourceBlock(
                            text=" | ".join(values),
                            kind="table_row",
                            style_name="xlsx_row",
                            table_group=table_group,
                        ))
            return blocks
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return []


def _read_pptx_blocks(path: Path) -> list[_SourceBlock]:
    """Read all OOXML slide text in stable slide order."""

    try:
        with zipfile.ZipFile(path) as archive:
            slide_paths = sorted(
                (
                    name
                    for name in archive.namelist()
                    if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
                ),
                key=lambda name: int(re.search(r"(\d+)", name).group(1)),
            )
            blocks: list[_SourceBlock] = []
            for slide_index, slide_path in enumerate(slide_paths, start=1):
                root = ET.fromstring(archive.read(slide_path))
                values = [
                    _clean(node.text or "")
                    for node in root.iter()
                    if node.tag.endswith("}t") and _clean(node.text or "")
                ]
                if not values:
                    continue
                blocks.append(_SourceBlock(
                    text=values[0],
                    kind="heading",
                    style_name="pptx_slide_title",
                    heading_level=1,
                ))
                blocks.extend(
                    _SourceBlock(
                        text=value,
                        kind="paragraph",
                        style_name=f"pptx_slide_{slide_index}",
                    )
                    for value in values[1:]
                )
            return blocks
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return []


def read_source_text_lines(path: str | Path) -> list[str]:
    """Public safe text view shared by W9 and the online chunk builder."""

    return [block.text for block in _read_blocks(Path(path)) if block.text]


def _read_docx_blocks(path: Path, *, asset_root: Path | None = None) -> list[_SourceBlock]:
    try:
        with zipfile.ZipFile(path) as archive:
            document_root = ET.fromstring(archive.read("word/document.xml"))
            styles = _docx_styles(archive)
            numbering = _docx_numbering(archive)
            relationships = _docx_relationships(archive)
            source_file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            return _docx_body_blocks(
                document_root,
                styles,
                numbering,
                archive,
                relationships,
                source_path=path,
                source_file_hash=source_file_hash,
                asset_root=asset_root,
            )
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return []


def _docx_body_blocks(
    document_root: ET.Element,
    styles: dict[str, tuple[str, int | None]],
    numbering: dict[tuple[str, int], tuple[str, str]],
    archive: zipfile.ZipFile,
    relationships: dict[str, dict[str, str]],
    *,
    source_path: Path,
    source_file_hash: str,
    asset_root: Path | None,
) -> list[_SourceBlock]:
    body = document_root.find(f"{_W}body")
    if body is None:
        return []
    blocks: list[_SourceBlock] = []
    table_group = 0
    for child in body:
        if child.tag == f"{_W}p":
            media_refs = _docx_media_refs(
                child,
                archive,
                relationships,
                source_path=source_path,
                source_file_hash=source_file_hash,
                asset_root=asset_root,
            )
            block = _docx_paragraph_block(
                child,
                styles,
                numbering=numbering,
                media_refs=media_refs,
            )
            if block is not None:
                blocks.append(block)
        elif child.tag == f"{_W}tbl":
            table_group += 1
            rows = list(child.findall(f"{_W}tr"))
            row_cells: list[tuple[ET.Element, list[str]]] = []
            for row in rows:
                cells: list[str] = []
                for cell in row.findall(f"{_W}tc"):
                    cell_text = _clean(" ".join(_element_text(p) for p in cell.iter(f"{_W}p")))
                    if cell_text:
                        cells.append(cell_text)
                row_cells.append((row, cells))
            # Many operational DOCX files use a one-cell table only to add a
            # grey background around a command.  Treating each such box as a
            # data-table transition fragments one logical method into a title,
            # three commands and three success messages.  Only multi-column
            # rows retain hard table semantics.
            is_data_table = any(len(cells) > 1 for _row, cells in row_cells)
            for row, cells in row_cells:
                media_refs = _docx_media_refs(
                    row,
                    archive,
                    relationships,
                    source_path=source_path,
                    source_file_hash=source_file_hash,
                    asset_root=asset_root,
                )
                if cells or media_refs:
                    blocks.append(_SourceBlock(
                        text=" | ".join(cells) or _media_block_text(media_refs),
                        kind="table_row" if cells and is_data_table else "code_block" if cells else "image",
                        table_group=table_group if is_data_table else None,
                        media_refs=tuple(media_refs),
                    ))
    return _contextualize_media_blocks(blocks)


def _docx_styles(archive: zipfile.ZipFile) -> dict[str, tuple[str, int | None]]:
    try:
        root = ET.fromstring(archive.read("word/styles.xml"))
    except (KeyError, ET.ParseError):
        return {}
    result: dict[str, tuple[str, int | None]] = {}
    for style in root.findall(f".//{_W}style"):
        style_id = str(style.get(f"{_W}styleId") or "")
        name_node = style.find(f"{_W}name")
        outline_node = style.find(f".//{_W}outlineLvl")
        name = str(name_node.get(f"{_W}val") or "") if name_node is not None else ""
        level: int | None = None
        if outline_node is not None:
            try:
                level = int(outline_node.get(f"{_W}val") or "0") + 1
            except ValueError:
                level = None
        if level is None:
            match = re.search(r"(?:heading|标题)\s*([1-9])", name, flags=re.IGNORECASE)
            if match:
                level = int(match.group(1))
        if style_id:
            result[style_id] = (name, level)
    return result


def _docx_numbering(
    archive: zipfile.ZipFile,
) -> dict[tuple[str, int], tuple[str, str]]:
    """Resolve DOCX numId/level pairs to a stable list style and marker.

    Some source documents encode every visible list item with a distinct
    ``numId`` whose abstract numbering starts at the displayed ordinal.  The
    marker therefore has to be read from ``numbering.xml`` instead of being
    regenerated from paragraph order.
    """

    try:
        root = ET.fromstring(archive.read("word/numbering.xml"))
    except (KeyError, ET.ParseError):
        return {}
    abstract: dict[str, dict[int, tuple[str, str]]] = {}
    for item in root.findall(f"{_W}abstractNum"):
        abstract_id = str(item.get(f"{_W}abstractNumId") or "")
        levels: dict[int, tuple[str, str]] = {}
        for level in item.findall(f"{_W}lvl"):
            try:
                level_index = int(level.get(f"{_W}ilvl") or "0")
            except ValueError:
                level_index = 0
            format_node = level.find(f"{_W}numFmt")
            text_node = level.find(f"{_W}lvlText")
            start_node = level.find(f"{_W}start")
            number_format = (
                str(format_node.get(f"{_W}val") or "")
                if format_node is not None else ""
            )
            marker_template = (
                str(text_node.get(f"{_W}val") or "")
                if text_node is not None else ""
            )
            start = (
                str(start_node.get(f"{_W}val") or "1")
                if start_node is not None else "1"
            )
            marker = marker_template.replace(f"%{level_index + 1}", start)
            levels[level_index] = (
                "bullet" if number_format == "bullet" else "ordered",
                marker or ("-" if number_format == "bullet" else f"{start}."),
            )
        if abstract_id:
            abstract[abstract_id] = levels
    result: dict[tuple[str, int], tuple[str, str]] = {}
    for item in root.findall(f"{_W}num"):
        num_id = str(item.get(f"{_W}numId") or "")
        abstract_node = item.find(f"{_W}abstractNumId")
        abstract_id = (
            str(abstract_node.get(f"{_W}val") or "")
            if abstract_node is not None else ""
        )
        for level_index, list_info in abstract.get(abstract_id, {}).items():
            result[(num_id, level_index)] = list_info
    return result


def _docx_paragraph_block(
    paragraph: ET.Element,
    styles: dict[str, tuple[str, int | None]],
    *,
    numbering: dict[tuple[str, int], tuple[str, str]] | None = None,
    media_refs: list[dict[str, Any]] | None = None,
) -> _SourceBlock | None:
    text = _clean(_element_text(paragraph))
    refs = list(media_refs or [])
    if not text and not refs:
        return None
    style_node = paragraph.find(f"./{_W}pPr/{_W}pStyle")
    style_id = str(style_node.get(f"{_W}val") or "") if style_node is not None else ""
    style_name, level = styles.get(style_id, (style_id, None))
    outline_node = paragraph.find(f"./{_W}pPr/{_W}outlineLvl")
    if outline_node is not None:
        try:
            level = int(outline_node.get(f"{_W}val") or "0") + 1
        except ValueError:
            pass
    if level is None and _SEMANTIC_NUMBERED_HEADING.match(text):
        level = 4
    list_level: int | None = None
    list_style = ""
    list_marker = ""
    num_node = paragraph.find(f"./{_W}pPr/{_W}numPr/{_W}numId")
    if num_node is not None and level is None:
        num_id = str(num_node.get(f"{_W}val") or "")
        level_node = paragraph.find(f"./{_W}pPr/{_W}numPr/{_W}ilvl")
        try:
            list_level = int(
                level_node.get(f"{_W}val") or "0"
                if level_node is not None else "0"
            )
        except ValueError:
            list_level = 0
        list_style, list_marker = (numbering or {}).get(
            (num_id, list_level),
            ("ordered", "1."),
        )
    return _SourceBlock(
        text=text or _media_block_text(refs),
        kind=(
            "heading" if level is not None
            else "image" if refs and not text
            else "list_item" if list_level is not None
            else "paragraph"
        ),
        style_name=style_name,
        heading_level=level,
        media_refs=tuple(refs),
        list_level=list_level,
        list_style=list_style,
        list_marker=list_marker,
    )


def _docx_relationships(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    try:
        root = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
    except (KeyError, ET.ParseError):
        return {}
    return {
        str(item.get("Id") or ""): {
            "target": str(item.get("Target") or ""),
            "type": str(item.get("Type") or ""),
            "target_mode": str(item.get("TargetMode") or ""),
        }
        for item in root.findall(f"{{{_PR_NS}}}Relationship")
        if str(item.get("Id") or "")
    }


def _docx_media_refs(
    element: ET.Element,
    archive: zipfile.ZipFile,
    relationships: dict[str, dict[str, str]],
    *,
    source_path: Path,
    source_file_hash: str,
    asset_root: Path | None,
) -> list[dict[str, Any]]:
    ids: list[tuple[str, str, str]] = []
    for node in element.iter(f"{{{_A_NS}}}blip"):
        ids.append((str(node.get(f"{{{_R_NS}}}embed") or ""), "image", ""))
    for node in element.iter(f"{{{_V_NS}}}imagedata"):
        ids.append((str(node.get(f"{{{_R_NS}}}id") or ""), "image", str(node.get(f"{{{_O_NS}}}title") or "")))
    for node in element.iter(f"{{{_O_NS}}}OLEObject"):
        ids.append((str(node.get(f"{{{_R_NS}}}id") or ""), "attachment", str(node.get("ProgID") or "")))
    alt_values = _dedupe(
        str(node.get("descr") or node.get("title") or node.get("name") or "")
        for node in element.iter(f"{{{_WP_NS}}}docPr")
    )
    refs: list[dict[str, Any]] = []
    for relationship_id, media_kind, explicit_label in ids:
        relation = relationships.get(relationship_id) or {}
        target = str(relation.get("target") or "")
        if not relationship_id or not target or relation.get("target_mode") == "External":
            continue
        archive_path = _docx_archive_path(target)
        try:
            payload = archive.read(archive_path)
        except KeyError:
            continue
        content_hash = hashlib.sha256(payload).hexdigest()
        suffix = Path(archive_path).suffix.lower()
        asset_path = ""
        if asset_root is not None:
            destination = asset_root / source_file_hash[:16] / f"{content_hash[:24]}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != content_hash:
                destination.write_bytes(payload)
            asset_path = str(destination.resolve())
        label = explicit_label or (alt_values[0] if alt_values else "") or Path(archive_path).name
        refs.append({
            "media_id": f"media:{content_hash[:24]}",
            "media_kind": media_kind,
            "label": label,
            "relationship_id": relationship_id,
            "archive_path": archive_path,
            "source_path": str(source_path),
            "content_hash": content_hash,
            "mime_type": _media_mime_type(suffix),
            "asset_path": asset_path,
        })
    return _dedupe_media_refs(refs)


def _docx_archive_path(target: str) -> str:
    clean = str(target or "").replace("\\", "/").lstrip("/")
    while clean.startswith("../"):
        clean = clean[3:]
    return clean if clean.startswith("word/") else f"word/{clean}"


def _media_mime_type(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
    }.get(str(suffix or "").lower(), "application/octet-stream")


def _media_block_text(media_refs: Iterable[dict[str, Any]]) -> str:
    labels = _dedupe(
        f"{'图片' if item.get('media_kind') == 'image' else '附件'}：{item.get('label') or item.get('archive_path')}"
        for item in media_refs
    )
    return "；".join(f"[{label}]" for label in labels)


def _contextualize_media_blocks(blocks: list[_SourceBlock]) -> list[_SourceBlock]:
    result: list[_SourceBlock] = list(blocks)
    for index, block in enumerate(blocks):
        if not block.media_refs:
            continue
        following_caption = ""
        if index + 1 < len(blocks):
            candidate = str(blocks[index + 1].text or "").strip()
            if _FIGURE_CAPTION.fullmatch(candidate):
                following_caption = candidate
                result[index + 1] = replace(
                    result[index + 1],
                    kind="figure_caption",
                )
        context = following_caption or _media_context_text(block.text)
        if not context:
            for previous in reversed(blocks[:index]):
                context = _media_context_text(previous.text)
                if context:
                    break
        refs: list[dict[str, Any]] = []
        count = len([item for item in block.media_refs if item.get("media_kind") == "image"])
        image_index = 0
        for media in block.media_refs:
            item = dict(media)
            if item.get("media_kind") == "image":
                image_index += 1
                contextual_label = context or str(item.get("label") or "源文档图片")
                if count > 1:
                    contextual_label = f"{contextual_label}（图{image_index}/{count}）"
                item["context_label"] = contextual_label
                if following_caption:
                    item["caption"] = contextual_label
            refs.append(item)
        result[index] = replace(block, media_refs=tuple(refs))
    return result


def _media_context_overrides(
    media_assets: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Select the most explicit approved caption for each content-addressed asset."""

    result: dict[str, str] = {}
    for asset in media_assets:
        if not isinstance(asset, dict) or asset.get("approved") is False:
            continue
        keys = _dedupe([
            str(asset.get("content_hash") or ""),
            str(asset.get("media_id") or ""),
        ])
        candidates = _dedupe([
            *(str(item or "") for item in asset.get("context_labels") or []),
            str(asset.get("label") or ""),
        ])
        descriptive_figures = [
            value
            for value in candidates
            if re.match(
                r"^(?:图|figure|fig\.?)\s*"
                r"(?:\d+(?:[-.]\d+)*|[一二三四五六七八九十百]+)"
                r"\s*[：:]\s*\S+",
                value,
                flags=re.IGNORECASE,
            )
        ]
        selected = (
            max(descriptive_figures, key=len)
            if descriptive_figures
            else next((value for value in candidates if not _generic_media_label(value)), "")
        )
        if selected:
            for key in keys:
                if key:
                    result[key] = selected
    return result


def _apply_media_context_overrides(
    blocks: Iterable[_SourceBlock],
    overrides: dict[str, str],
) -> list[_SourceBlock]:
    result: list[_SourceBlock] = []
    for block in blocks:
        refs: list[dict[str, Any]] = []
        for raw in block.media_refs:
            item = dict(raw)
            caption = ""
            for key in (
                str(item.get("content_hash") or ""),
                str(item.get("media_id") or ""),
            ):
                if key and key in overrides:
                    caption = overrides[key]
                    break
            if caption:
                item["context_label"] = caption
                item["caption"] = caption
            refs.append(item)
        result.append(
            replace(block, media_refs=tuple(refs))
            if refs
            else block
        )
    return result


def _media_context_text(text: str) -> str:
    lines = [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip() and not re.fullmatch(r"\[(?:图片|附件)：[^\]]+\]", line.strip())
    ]
    if not lines:
        return ""
    value = lines[-1]
    return value if len(value) <= 90 else value[:87].rstrip() + "…"


def _generic_media_label(value: str) -> bool:
    label = str(value or "").strip()
    return bool(
        not label
        or re.fullmatch(r"(?:image|图片|附件)\s*\d*(?:\.[a-z0-9]+)?", label, re.IGNORECASE)
        or label.startswith("Drawing ")
    )


def _element_text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{_W}t"))


def _align_sections(
    blocks: list[_SourceBlock],
    sections: list[dict[str, Any]],
) -> dict[int, list[tuple[dict[str, Any], str]]]:
    """Align KG sections by heading and source-derived summary in document order."""

    anchors: dict[int, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    direct: dict[int, tuple[int, str]] = {}
    seen_fingerprints: dict[tuple[str, str], int] = {}
    seen_summaries: dict[str, int] = {}
    cursor = 0
    for ordinal, section in enumerate(sections):
        fingerprint = _section_fingerprint(section)
        summary_key = fingerprint[1]
        duplicate_index = seen_fingerprints.get(fingerprint)
        if duplicate_index is None and len(summary_key) >= 8:
            duplicate_index = seen_summaries.get(summary_key)
        if duplicate_index is None and len(summary_key) >= 40:
            duplicate_index = next((
                index
                for known_summary, index in seen_summaries.items()
                if min(len(summary_key), len(known_summary)) >= 40
                and (
                    summary_key.startswith(known_summary)
                    or known_summary.startswith(summary_key)
                )
            ), None)
        if duplicate_index is not None:
            direct[ordinal] = (duplicate_index, "direct_duplicate_section")
            continue
        found = _find_section_anchor(blocks, section, cursor)
        if found is None:
            continue
        index, method = found
        direct[ordinal] = (index, method)
        seen_fingerprints[fingerprint] = index
        if len(summary_key) >= 8:
            seen_summaries[summary_key] = index
        cursor = index + 1

    # A few KG sections are structural parents with no literal source line.
    # Bind them to the first following concrete child, but retain the inferred
    # method so callers can audit that it was not an exact textual match.
    for ordinal, section in enumerate(sections):
        if ordinal in direct:
            index, method = direct[ordinal]
        else:
            next_direct = next((direct[i][0] for i in range(ordinal + 1, len(sections)) if i in direct), None)
            previous_direct = next((direct[i][0] for i in range(ordinal - 1, -1, -1) if i in direct), None)
            if next_direct is not None:
                index, method = next_direct, "inferred_next_section"
            elif previous_direct is not None:
                index, method = previous_direct, "inferred_previous_section"
            elif blocks:
                index, method = 0, "inferred_document_start"
            else:
                continue
        anchors[index].append((section, method))
    return dict(anchors)


def _section_fingerprint(section: dict[str, Any]) -> tuple[str, str]:
    return (
        _lookup(str(section.get("heading") or "")),
        _lookup(str(section.get("summary") or "")),
    )


def _find_section_anchor(
    blocks: list[_SourceBlock],
    section: dict[str, Any],
    cursor: int,
) -> tuple[int, str] | None:
    heading = _lookup(str(section.get("heading") or ""))
    bare_heading = _HEADING_PREFIX.sub("", heading).lstrip("：:")
    fragments = _summary_fragments(str(section.get("summary") or ""))

    # Specific source summaries disambiguate generic repeated headings such as
    # “操作” and “判断”; literal headings remain preferable otherwise.
    heading_specs: list[tuple[str, str, int]] = []
    if heading:
        heading_specs.append((heading, "direct_heading", 2))
    if bare_heading and bare_heading != heading:
        heading_specs.append((bare_heading, "direct_bare_heading", 3))
    summary_specs = [(fragment, "direct_summary", 4) for fragment in fragments]
    generic_heading = bare_heading in {
        "操作", "判断", "现象", "目标", "问题", "排查步骤", "诊断步骤", "解决方案", "注意事项",
    }
    search_specs = [*summary_specs, *heading_specs] if generic_heading else [*heading_specs, *summary_specs]
    for needle, method, minimum in search_specs:
        if len(needle) < minimum:
            continue
        for index in range(cursor, len(blocks)):
            candidate = _lookup(blocks[index].text)
            bare_candidate = _HEADING_PREFIX.sub("", candidate).lstrip("：:")
            if _text_matches(needle, candidate) or _text_matches(needle, bare_candidate):
                return index, method
    return None


def _summary_fragments(value: str) -> list[str]:
    fragments: list[str] = []
    for raw in _SUMMARY_SPLIT.split(value):
        item = _lookup(raw)
        if len(item) >= 4 and item not in {"排查步骤", "诊断步骤", "解决方案", "注意事项"}:
            fragments.append(item)
        if len(fragments) >= 4:
            break
    return fragments


def _text_matches(needle: str, candidate: str) -> bool:
    if needle == candidate:
        return True
    short = min(len(needle), len(candidate))
    if short < 4:
        return False
    return candidate.startswith(needle) or needle.startswith(candidate) or (
        len(needle) >= 8 and needle in candidate
    )


def _chunk_ranges(
    blocks: list[_SourceBlock],
    anchors: dict[int, list[tuple[dict[str, Any], str]]],
    path: Path,
) -> list[tuple[int, int, list[tuple[dict[str, Any], str]], str]]:
    """Apply hard semantic boundaries, then bounded within-section splitting."""

    ranges: list[tuple[int, int, list[tuple[dict[str, Any], str]], str]] = []
    current_bindings: list[tuple[dict[str, Any], str]] = []
    current_label = ""
    start = 0
    size = 0
    table_rows = 0

    def flush(end: int) -> None:
        nonlocal start, size, table_rows
        if end > start:
            ranges.append((start, end, list(current_bindings), current_label))
        start = end
        size = 0
        table_rows = 0

    for index, block in enumerate(blocks):
        bindings = anchors.get(index) or []
        semantic_heading = block.heading_level is not None or _is_faq_question(block, path)
        previous = blocks[index - 1] if index > 0 else None
        heading_stack = index > start and all(
            item.heading_level is not None
            or _is_faq_question(item, path)
            or _matches_bound_section_heading(item, current_bindings)
            for item in blocks[start:index]
        )
        table_changed = bool(
            previous is not None
            and block.table_group != previous.table_group
            and (block.kind == "table_row" or previous.kind == "table_row")
        )
        section_changed = bool(bindings and bindings != current_bindings)
        hard_boundary = index > start and not heading_stack and (
            section_changed or semantic_heading or table_changed
        )
        too_large = index > start and (
            size + len(block.text) + 1 > _MAX_CHARS
            or index - start >= _MAX_BLOCKS
            or (block.kind == "table_row" and table_rows >= _MAX_TABLE_ROWS)
        )
        if hard_boundary or too_large:
            flush(index)
        if bindings:
            if hard_boundary or too_large or not current_bindings:
                current_bindings = list(bindings)
            else:
                seen = {
                    (str(section.get("section_id") or ""), method)
                    for section, method in current_bindings
                }
                current_bindings.extend(
                    (section, method)
                    for section, method in bindings
                    if (str(section.get("section_id") or ""), method) not in seen
                )
        if semantic_heading:
            current_label = block.text
        elif not current_label and bindings:
            current_label = str(bindings[-1][0].get("heading") or "")
        size += len(block.text) + 1
        if block.kind == "table_row":
            table_rows += 1
    flush(len(blocks))
    return ranges


def _is_faq_question(block: _SourceBlock, path: Path) -> bool:
    return "faq" in path.stem.lower() and bool(_QUESTION.search(block.text.strip()))


def _matches_bound_section_heading(
    block: _SourceBlock,
    bindings: list[tuple[dict[str, Any], str]],
) -> bool:
    candidate = _HEADING_PREFIX.sub("", _lookup(block.text)).lstrip("：:")
    return any(
        candidate == _HEADING_PREFIX.sub(
            "", _lookup(str(section.get("heading") or ""))
        ).lstrip("：:")
        for section, _method in bindings
        if candidate
    )


def _lookup(value: str) -> str:
    return _SPACE.sub("", str(value or "").strip().lower()).strip("#：:。；;")


def _clean(value: str) -> str:
    return _SPACE.sub(" ", str(value or "").replace("\u200b", "")).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_media_refs(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        key = (
            str(value.get("media_kind") or ""),
            str(value.get("content_hash") or value.get("archive_path") or ""),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result
