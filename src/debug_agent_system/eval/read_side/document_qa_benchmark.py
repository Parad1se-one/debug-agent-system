"""Build an extractive, original-document-grounded AOI QA pilot.

Queries are manually rewritten as natural field questions.  Approved KG v2
manual cards are used only to route each query to a canonical SOP section.
Every answer sentence is then extracted from the original SOP document
snapshot; card actions, required-info fields, and generic governance wording
are deliberately excluded from the answer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA_VERSION = "debug_agent_system.aoi_document_qa_pilot.v1.source_grounded"
BENCHMARK_ID = "aoi-document-qa-pilot-v1"
SEED_SCHEMA_VERSION = "debug_agent_system.document_qa_pilot_seeds.v1"
MIN_CASE_COUNT = 20
LEGACY_SIMILARITY_LIMIT = 0.82

DEFAULT_SEEDS = Path("data/eval/benchmark/document_qa_pilot_v1_seeds.json")
DEFAULT_CARDS_ROOT = Path("data/kg_v2_sop_draft_build")
DEFAULT_SHARED_QUERIES = (
    Path("data/eval/scenarios/read_side_shared_query_baseline_v1.json"),
    Path("data/eval/scenarios/read_side_pure_codex_baseline_v1.json"),
)
DEFAULT_EXISTING_BENCHMARKS = (
    Path("data/eval/benchmark/aoi_debug_benchmark_v1.json"),
    Path("data/eval/benchmark/aoi_fae_report_benchmark_v2.json"),
)
DEFAULT_OUT = Path("data/eval/benchmark/aoi_document_qa_pilot_v1.json")
DEFAULT_REPORT_OUT = Path(
    "data/eval/benchmark/aoi_document_qa_pilot_v1.report.json"
)
DEFAULT_MARKDOWN_OUT = Path(
    "data/results/benchmark_reports/aoi-document-qa-pilot-v1/"
    "Query与答案.md"
)

_SPACE = re.compile(r"\s+")
_NORMALIZE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)
_HEADING_START = re.compile(r"<h(?P<level>[1-6])\b[^>]*>", re.IGNORECASE)
_HEADING_TITLE_END = re.compile(
    r"<(?:h[1-6]|p|ol|ul|li|img|div|table|file|source|quote|pre|blockquote)\b"
    r"|</h[1-6]\s*>",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_DOWNLOAD_FILE = re.compile(
    r"下载\s+([^\s，。；;]+?\.(?:bat|cmd|exe|msi|zip|7z|rar|txt|json))",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _normalized(value: Any) -> str:
    return _NORMALIZE.sub("", _clean(value).lower())


def _trigrams(value: Any) -> set[str]:
    text = _normalized(value)
    if len(text) < 3:
        return {text} if text else set()
    return {text[index : index + 3] for index in range(len(text) - 2)}


def _similarity(left: Any, right: Any) -> float:
    left_grams = _trigrams(left)
    right_grams = _trigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _load_query_records(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records") or payload.get("cases") or []
    return [
        _clean(row.get("query"))
        for row in rows
        if isinstance(row, dict) and _clean(row.get("query"))
    ]


def _card_index(cards_root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    for path in sorted(
        path
        for path in cards_root.rglob("*.json")
        if "manual_cards" in path.parts
    ):
        candidates.setdefault(path.name, []).append(path)
    index: dict[str, Path] = {}
    for name, paths in candidates.items():
        approved = [
            path
            for path in paths
            if json.loads(path.read_text(encoding="utf-8")).get("status")
            == "approved_for_phase1_build"
        ]
        if len(approved) == 1:
            index[name] = approved[0]
    return index


class _SourceBlockParser(HTMLParser):
    """Extract source text blocks while preserving their opening order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._sequence = 0
        self._stack: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._hidden_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._hidden_depth += 1
            return
        if self._hidden_depth:
            return
        if tag in {"p", "li"}:
            record = {
                "sequence": self._sequence,
                "tag": tag,
                "parts": [],
            }
            self._sequence += 1
            self._records.append(record)
            self._stack.append(record)
        elif tag == "br" and self._stack:
            self._stack[-1]["parts"].append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)
            return
        if self._hidden_depth or tag not in {"p", "li"}:
            return
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index]["tag"] == tag:
                del self._stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self._hidden_depth or not _clean(data):
            return
        if self._stack:
            self._stack[-1]["parts"].append(data)
            return
        self._records.append(
            {
                "sequence": self._sequence,
                "tag": "text",
                "parts": [data],
            }
        )
        self._sequence += 1

    @property
    def blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        seen_adjacent = ""
        for record in sorted(self._records, key=lambda item: item["sequence"]):
            text = _clean("".join(record["parts"]))
            if not text or text == seen_adjacent:
                continue
            blocks.append(
                {
                    "block_index": len(blocks) + 1,
                    "source_tag": record["tag"],
                    "text": text,
                }
            )
            seen_adjacent = text
        return blocks


class _SourceAssetParser(HTMLParser):
    """Retain image/attachment attributes that text extraction cannot see."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.assets: list[dict[str, Any]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag not in {"img", "source", "file"}:
            return
        self.assets.append(
            {
                "source_tag": tag,
                "attributes": {
                    str(key): str(value or "")
                    for key, value in attrs
                    if key
                },
            }
        )

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)


def _plain_heading_title(fragment: str) -> str:
    return _clean(html.unescape(_TAG.sub("", fragment)).replace("\xa0", " "))


def _load_source_html(fetch_path: Path) -> tuple[str, str]:
    raw = json.loads(fetch_path.read_text(encoding="utf-8"))
    content = str(
        ((((raw.get("data") or {}).get("document") or {}).get("content")) or "")
    )
    if not content:
        raise ValueError(f"missing_document_content:{fetch_path}")
    return content, hashlib.sha256(content.encode("utf-8")).hexdigest()


def _source_heading_records(content: str) -> list[dict[str, Any]]:
    starts = list(_HEADING_START.finditer(content))
    records: list[dict[str, Any]] = []
    path_stack: list[tuple[int, str]] = []
    for index, match in enumerate(starts):
        level = int(match.group("level"))
        title_start = match.end()
        stop = _HEADING_TITLE_END.search(content, title_start)
        title_end = stop.start() if stop else len(content)
        title = _plain_heading_title(content[title_start:title_end])
        if not title:
            continue
        path_stack = [
            (prior_level, prior_title)
            for prior_level, prior_title in path_stack
            if prior_level < level
        ]
        path_stack.append((level, title))
        next_boundary = len(content)
        for later in starts[index + 1 :]:
            if int(later.group("level")) <= level:
                next_boundary = later.start()
                break
        body_html = content[title_end:next_boundary]
        parser = _SourceBlockParser()
        parser.feed(body_html)
        asset_parser = _SourceAssetParser()
        asset_parser.feed(body_html)
        records.append(
            {
                "level": level,
                "title": title,
                "title_path": [value for _, value in path_stack],
                "source_span": {
                    "html_char_start": title_end,
                    "html_char_end": next_boundary,
                },
                "source_fragment_sha256": hashlib.sha256(
                    body_html.encode("utf-8")
                ).hexdigest(),
                "image_count": len(
                    re.findall(r"<img\b", body_html, re.IGNORECASE)
                ),
                "assets": asset_parser.assets,
                "blocks": parser.blocks,
            }
        )
    return records


def _source_section_specs() -> dict[str, dict[str, Any]]:
    # SECTION_MAP is routing metadata only.  Its manual-card actions and
    # required-info fields are never read by this benchmark builder.
    from debug_agent_system.eval.write_side.kg_v2_main_program_manual_build import (
        SECTION_MAP,
        SECTION_RAW_TEXTS,
    )

    return {
        str(item["canonical_section_id"]): {
            **item,
            "source_transcript": str(
                SECTION_RAW_TEXTS.get(
                    str(item["canonical_section_id"]),
                    "",
                )
            ),
        }
        for item in SECTION_MAP
        if item.get("canonical_section_id")
    }


def _path_suffix_score(actual: list[str], expected: list[str]) -> int:
    score = 0
    for left, right in zip(reversed(actual), reversed(expected)):
        if _normalized(left) != _normalized(right):
            break
        score += 1
    return score


def _resolve_source_section(
    section_id: str,
    section_specs: dict[str, dict[str, Any]],
    heading_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if section_id not in section_specs:
        raise ValueError(f"missing_source_section_spec:{section_id}")
    spec = section_specs[section_id]
    title = str(spec.get("title") or "")
    candidates = [
        record
        for record in heading_records
        if _normalized(record.get("title")) == _normalized(title)
    ]
    if not candidates:
        raise ValueError(f"source_heading_not_found:{section_id}:{title}")
    expected_path = list(spec.get("source_title_path") or [])
    candidates.sort(
        key=lambda record: _path_suffix_score(
            list(record.get("title_path") or []),
            expected_path,
        ),
        reverse=True,
    )
    best_score = _path_suffix_score(
        list(candidates[0].get("title_path") or []),
        expected_path,
    )
    if (
        len(candidates) > 1
        and _path_suffix_score(
            list(candidates[1].get("title_path") or []),
            expected_path,
        )
        == best_score
    ):
        raise ValueError(f"ambiguous_source_heading:{section_id}:{title}")
    record = candidates[0]
    if not record.get("blocks"):
        raise ValueError(f"source_section_without_text:{section_id}:{title}")
    return spec, record


def _reference_answer(excerpts: list[dict[str, Any]]) -> str:
    texts = [_clean(item.get("text")) for item in excerpts if _clean(item.get("text"))]
    if len(texts) == 1:
        return texts[0]
    return "\n".join(f"- {text}" for text in texts)


def _reference_answer_with_attachments(
    excerpts: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
) -> str:
    answer = _reference_answer(excerpts)
    display_names = [
        _clean(item.get("display_name"))
        for item in attachments
        if _clean(item.get("display_name"))
    ]
    if len(display_names) == 1 and "下载以下文件" in answer:
        return answer.replace(
            "下载以下文件",
            f"下载附件 {display_names[0]} ",
        )
    missing_names = [name for name in display_names if name not in answer]
    if missing_names:
        return answer + "\n\n原文附件：" + "、".join(missing_names)
    return answer


def _embedded_asset_path(
    fetch_path: Path,
    attributes: dict[str, str],
) -> Path | None:
    media_root = fetch_path.parent / "embedded_media"
    if not media_root.exists():
        return None
    identifiers = [
        value
        for value in (
            attributes.get("token"),
            attributes.get("src"),
            attributes.get("name"),
        )
        if value
    ]
    candidates = sorted(path for path in media_root.iterdir() if path.is_file())
    for identifier in identifiers:
        matches = [path for path in candidates if identifier in path.name]
        if len(matches) == 1:
            return matches[0]
    return None


def _source_images(
    card: dict[str, Any],
    section_id: str,
    source_record: dict[str, Any],
    fetch_path: Path,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for action in card.get("actions") or []:
        for ref in action.get("curated_image_refs") or []:
            if str(ref.get("source_section_id") or "") != section_id:
                continue
            path = Path(str(ref.get("relative_path") or ""))
            if not path.exists() or str(path) in seen_paths:
                continue
            images.append(
                {
                    "image_id": str(ref.get("image_id") or ""),
                    "path": str(path),
                    "sha256": _sha256(path),
                    "caption": _clean(ref.get("caption")) or path.name,
                    "binding_origin": "approved_source_media_binding",
                }
            )
            seen_paths.add(str(path))
    for asset in source_record.get("assets") or []:
        if asset.get("source_tag") != "img":
            continue
        attributes = dict(asset.get("attributes") or {})
        path = _embedded_asset_path(fetch_path, attributes)
        if path is None or str(path) in seen_paths:
            continue
        images.append(
            {
                "image_id": (
                    f"raw-image:{section_id}:{len(images) + 1}"
                ),
                "path": str(path),
                "sha256": _sha256(path),
                "caption": _clean(attributes.get("name")) or path.name,
                "binding_origin": "original_fetch_html_asset",
            }
        )
        seen_paths.add(str(path))
    return images


def _source_attachments(
    transcript: str,
    section_id: str,
    source_record: dict[str, Any],
    fetch_path: Path,
) -> list[dict[str, Any]]:
    display_names = _DOWNLOAD_FILE.findall(transcript)
    attachments: list[dict[str, Any]] = []
    for asset in source_record.get("assets") or []:
        if asset.get("source_tag") not in {"source", "file"}:
            continue
        attributes = dict(asset.get("attributes") or {})
        path = _embedded_asset_path(fetch_path, attributes)
        if path is None:
            continue
        display_name = (
            display_names[len(attachments)]
            if len(display_names) > len(attachments)
            else path.name
        )
        attachments.append(
            {
                "attachment_id": (
                    f"source-attachment:{section_id}:{len(attachments) + 1}"
                ),
                "display_name": display_name,
                "path": str(path),
                "sha256": _sha256(path),
                "source_token": str(attributes.get("token") or ""),
                "binding_origin": "original_fetch_html_asset",
            }
        )
    return attachments


def _aggregate_card_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_dataset(
    seeds_path: str | Path = DEFAULT_SEEDS,
    cards_root: str | Path = DEFAULT_CARDS_ROOT,
    *,
    shared_query_paths: Iterable[str | Path] = DEFAULT_SHARED_QUERIES,
    existing_benchmark_paths: Iterable[str | Path] = DEFAULT_EXISTING_BENCHMARKS,
) -> dict[str, Any]:
    seeds_path = Path(seeds_path)
    cards_root = Path(cards_root)
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    if seeds.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError("invalid_seed_schema")
    source_document = Path(str(seeds.get("source_document") or ""))
    if not source_document.exists():
        raise FileNotFoundError(source_document)
    source_document_sha256 = _sha256(source_document)

    historical_queries: list[str] = []
    historical_paths: list[Path] = []
    for raw_path in (*shared_query_paths, *existing_benchmark_paths):
        path = Path(raw_path)
        if path.exists():
            historical_paths.append(path)
            historical_queries.extend(_load_query_records(path))

    card_index = _card_index(cards_root)
    section_specs = _source_section_specs()
    source_cache: dict[
        Path,
        tuple[str, str, list[dict[str, Any]]],
    ] = {}
    cases: list[dict[str, Any]] = []
    used_paths: list[Path] = []
    used_fetch_paths: list[Path] = []
    for index, seed in enumerate(seeds.get("records") or [], 1):
        card_name = str(seed.get("card") or "")
        if card_name not in card_index:
            raise ValueError(f"missing_or_ambiguous_card:{card_name}")
        card_path = card_index[card_name]
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card.get("status") != "approved_for_phase1_build":
            raise ValueError(f"card_not_approved:{card_name}")
        source_sections = list(card.get("source_sections") or [])
        if len(source_sections) != 1:
            raise ValueError(
                f"card_source_section_count:{card_name}:{len(source_sections)}"
            )
        section_id = str(source_sections[0])
        if section_id not in section_specs:
            raise ValueError(f"missing_source_section_spec:{section_id}")
        fetch_path = Path(
            str(
                (section_specs[section_id].get("source_locator") or {}).get(
                    "fetch_json_path"
                )
                or ""
            )
        )
        if not fetch_path.exists():
            raise FileNotFoundError(fetch_path)
        if fetch_path not in source_cache:
            source_html, source_html_sha256 = _load_source_html(fetch_path)
            source_cache[fetch_path] = (
                source_html,
                source_html_sha256,
                _source_heading_records(source_html),
            )
        source_html, source_html_sha256, heading_records = source_cache[
            fetch_path
        ]
        section_spec, source_record = _resolve_source_section(
            section_id,
            section_specs,
            heading_records,
        )
        span = source_record["source_span"]
        source_fragment = source_html[
            int(span["html_char_start"]) : int(span["html_char_end"])
        ]
        if (
            hashlib.sha256(source_fragment.encode("utf-8")).hexdigest()
            != source_record["source_fragment_sha256"]
        ):
            raise ValueError(f"source_fragment_hash_mismatch:{section_id}")
        query = _clean(seed.get("query"))
        legacy_max = max(
            (_similarity(query, historical) for historical in historical_queries),
            default=0.0,
        )
        raw_html_excerpts = list(source_record.get("blocks") or [])
        source_transcript = str(
            section_spec.get("source_transcript") or ""
        ).strip()
        source_images = _source_images(
            card,
            section_id,
            source_record,
            fetch_path,
        )
        source_attachments = _source_attachments(
            source_transcript,
            section_id,
            source_record,
            fetch_path,
        )
        excerpts = raw_html_excerpts
        reference_answer = _reference_answer_with_attachments(
            excerpts,
            source_attachments,
        )
        case_id = f"doc-qa-{index:03d}"
        cases.append(
            {
                "case_id": case_id,
                "split": "pilot_review",
                "source_type": "original_sop_document_section_multimodal",
                "expectation_origin":
                    "extractive_original_text_and_original_media",
                "tracks": ["T0_evidence_retrieval", "T1_grounded_answer"],
                "isolated_session": True,
                "query": query,
                "source_refs": {
                    "document_path": str(source_document),
                    "document_sha256": source_document_sha256,
                    "fetch_json_path": str(fetch_path),
                    "fetch_json_sha256": _sha256(fetch_path),
                    "source_html_sha256": source_html_sha256,
                    "canonical_section_id": section_id,
                    "source_heading_title": str(
                        source_record.get("title") or ""
                    ),
                    "source_heading_path": list(
                        source_record.get("title_path") or []
                    ),
                    "mapped_title_path": list(
                        section_spec.get("source_title_path") or []
                    ),
                    "source_span": dict(span),
                    "source_fragment_sha256": str(
                        source_record.get("source_fragment_sha256") or ""
                    ),
                    "source_image_count": int(
                        source_record.get("image_count") or 0
                    ),
                    "included_image_count": len(source_images),
                    "source_attachment_count": len(source_attachments),
                    "source_transcript_origin":
                        "kg_v2_main_program_manual_build.SECTION_RAW_TEXTS",
                    "source_transcript_sha256": hashlib.sha256(
                        source_transcript.encode("utf-8")
                    ).hexdigest(),
                    "manual_card_path": str(card_path),
                    "manual_card_sha256": _sha256(card_path),
                    "manual_card_status": str(card.get("status") or ""),
                    "manual_card_role":
                        "section_routing_and_source_media_binding",
                },
                "answer_gold": {
                    "answer_mode":
                        "extractive_source_text_with_original_media",
                    "reference_answer": reference_answer,
                    "evidence_excerpts": excerpts,
                    "source_images": source_images,
                    "source_attachments": source_attachments,
                    "card_action_text_used_in_answer": False,
                    "source_transcript_usage":
                        "attachment_display_name_only"
                        if source_attachments
                        else "none",
                    "image_content_transcribed": False,
                },
                "quality": {
                    "query_manual_curated": True,
                    "answer_source_section_grounded": True,
                    "source_media_preserved": (
                        len(source_images)
                        >= int(source_record.get("image_count") or 0)
                    ),
                    "answer_has_non_source_claims": False,
                    "independent_expert_gold": False,
                    "requires_human_review": True,
                    "graph_ingestion_allowed": False,
                    "legacy_max_similarity": round(legacy_max, 6),
                },
            }
        )
        used_paths.append(card_path)
        used_fetch_paths.append(fetch_path)

    category_counts = Counter(
        str((case["source_refs"].get("mapped_title_path") or ["unknown"])[1])
        if len(case["source_refs"].get("mapped_title_path") or []) > 1
        else "unknown"
        for case in cases
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "build_policy": {
            "query_style": "manual_natural_field_question",
            "answer_source":
                "original_section_text_plus_original_images_and_attachments",
            "source_transcript_role":
                "attachment_display_name_recovery_only",
            "manual_card_role":
                "canonical_section_routing_and_source_media_binding",
            "manual_card_actions_allowed_in_answer": False,
            "manual_card_required_info_allowed_in_answer": False,
            "generic_verification_or_governance_allowed_in_answer": False,
            "source_images_and_attachments_must_be_preserved": True,
            "source_image_ocr_required": False,
            "draft_cards_allowed": False,
            "xing_lark_source_allowed": False,
            "existing_queries_excluded_by_similarity": True,
            "independent_expert_gold_claim_allowed": False,
            "graph_ingestion_allowed": False,
        },
        "source_manifest": {
            "seeds": str(seeds_path),
            "seeds_sha256": _sha256(seeds_path),
            "source_document": str(source_document),
            "source_document_sha256": source_document_sha256,
            "source_fetch_json_files": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for path in sorted(set(used_fetch_paths))
            ],
            "section_routing_map":
                "src/debug_agent_system/eval/write_side/"
                "kg_v2_main_program_manual_build.py:SECTION_MAP",
            "approved_manual_card_count": len(set(used_paths)),
            "approved_manual_cards_sha256": _aggregate_card_hash(used_paths),
            "historical_query_sources": [str(path) for path in historical_paths],
        },
        "cases": cases,
        "coverage": {
            "case_count": len(cases),
            "unique_manual_card_count": len(set(used_paths)),
            "unique_source_section_count": len(
                {
                    case["source_refs"]["canonical_section_id"]
                    for case in cases
                }
            ),
            "source_area_counts": dict(sorted(category_counts.items())),
            "answer_evidence_block_count": sum(
                len(case["answer_gold"]["evidence_excerpts"])
                for case in cases
            ),
            "included_source_image_count": sum(
                len(case["answer_gold"]["source_images"])
                for case in cases
            ),
            "included_source_attachment_count": sum(
                len(case["answer_gold"]["source_attachments"])
                for case in cases
            ),
            "cases_with_source_images": sum(
                int(case["source_refs"]["source_image_count"]) > 0
                for case in cases
            ),
            "legacy_similarity_max": max(
                (
                    float(case["quality"]["legacy_max_similarity"])
                    for case in cases
                ),
                default=0.0,
            ),
        },
    }


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version")
    if dataset.get("benchmark_id") != BENCHMARK_ID:
        issues.append("benchmark_id")
    cases = dataset.get("cases") or []
    if len(cases) < MIN_CASE_COUNT:
        issues.append("case_count")
    seen_queries: set[str] = set()
    seen_cards: set[str] = set()
    seen_sections: set[str] = set()
    source_html_cache: dict[Path, tuple[str, str]] = {}
    section_specs = _source_section_specs()
    for case in cases:
        case_id = str(case.get("case_id") or "")
        prefix = f"{case_id}:"
        query = _clean(case.get("query"))
        query_norm = _normalized(query)
        refs = case.get("source_refs") or {}
        quality = case.get("quality") or {}
        answer_gold = case.get("answer_gold") or {}
        answer = str(answer_gold.get("reference_answer") or "")
        excerpts = list(answer_gold.get("evidence_excerpts") or [])
        source_images = list(answer_gold.get("source_images") or [])
        source_attachments = list(
            answer_gold.get("source_attachments") or []
        )
        card_path = str(refs.get("manual_card_path") or "")
        section_id = str(refs.get("canonical_section_id") or "")
        if len(query) < 20 or not query.endswith(("？", "?")):
            issues.append(prefix + "query_quality")
        if query_norm in seen_queries:
            issues.append(prefix + "duplicate_query")
        seen_queries.add(query_norm)
        if card_path in seen_cards:
            issues.append(prefix + "duplicate_card")
        seen_cards.add(card_path)
        if "遇到“" in query or "根据批准资料应确认" in query:
            issues.append(prefix + "template_query")
        if float(quality.get("legacy_max_similarity") or 0.0) >= LEGACY_SIMILARITY_LIMIT:
            issues.append(prefix + "legacy_duplicate")
        if not quality.get("query_manual_curated"):
            issues.append(prefix + "query_not_curated")
        if quality.get("answer_source_section_grounded") is not True:
            issues.append(prefix + "answer_not_source_section_grounded")
        if quality.get("answer_has_non_source_claims") is not False:
            issues.append(prefix + "non_source_claim_boundary")
        if quality.get("independent_expert_gold") is not False:
            issues.append(prefix + "gold_boundary")
        if quality.get("graph_ingestion_allowed") is not False:
            issues.append(prefix + "graph_ingestion")
        if refs.get("manual_card_status") != "approved_for_phase1_build":
            issues.append(prefix + "card_status")
        if refs.get("manual_card_role") != (
            "section_routing_and_source_media_binding"
        ):
            issues.append(prefix + "manual_card_role")
        if not section_id:
            issues.append(prefix + "source_section")
        seen_sections.add(section_id)
        if answer_gold.get("answer_mode") != (
            "extractive_source_text_with_original_media"
        ):
            issues.append(prefix + "answer_mode")
        if answer_gold.get("card_action_text_used_in_answer") is not False:
            issues.append(prefix + "card_action_content_in_answer")
        if not excerpts or not answer:
            issues.append(prefix + "reference_answer")
        transcript = str(
            (section_specs.get(section_id) or {}).get("source_transcript") or ""
        ).strip()
        if not transcript:
            issues.append(prefix + "missing_source_transcript")
        if hashlib.sha256(transcript.encode("utf-8")).hexdigest() != refs.get(
            "source_transcript_sha256"
        ):
            issues.append(prefix + "source_transcript_sha256")
        if answer != _reference_answer_with_attachments(
            excerpts,
            source_attachments,
        ):
            issues.append(prefix + "answer_evidence_mismatch")
        expected_transcript_usage = (
            "attachment_display_name_only"
            if source_attachments
            else "none"
        )
        if (
            answer_gold.get("source_transcript_usage")
            != expected_transcript_usage
        ):
            issues.append(prefix + "source_transcript_usage")
        if any(
            marker in answer
            for marker in (
                "执行前需要确认：",
                "验证与边界：",
                "不代表现场已经执行",
                "verified_fix",
            )
        ):
            issues.append(prefix + "legacy_generated_wording")

        fetch_path = Path(str(refs.get("fetch_json_path") or ""))
        if not fetch_path.exists():
            issues.append(prefix + "fetch_json_path")
            continue
        if fetch_path not in source_html_cache:
            source_html_cache[fetch_path] = _load_source_html(fetch_path)
        source_html, source_html_sha256 = source_html_cache[fetch_path]
        if _sha256(fetch_path) != refs.get("fetch_json_sha256"):
            issues.append(prefix + "fetch_json_sha256")
        if source_html_sha256 != refs.get("source_html_sha256"):
            issues.append(prefix + "source_html_sha256")
        span = refs.get("source_span") or {}
        try:
            start = int(span.get("html_char_start"))
            end = int(span.get("html_char_end"))
        except (TypeError, ValueError):
            issues.append(prefix + "source_span")
            continue
        fragment = source_html[start:end]
        if (
            hashlib.sha256(fragment.encode("utf-8")).hexdigest()
            != refs.get("source_fragment_sha256")
        ):
            issues.append(prefix + "source_fragment_sha256")
        parser = _SourceBlockParser()
        parser.feed(fragment)
        if parser.blocks != excerpts:
            issues.append(prefix + "raw_html_excerpt_not_reproducible")
        if len(source_images) != int(refs.get("included_image_count") or 0):
            issues.append(prefix + "included_image_count")
        if len(source_images) < int(refs.get("source_image_count") or 0):
            issues.append(prefix + "source_image_loss")
        if len(source_attachments) != int(
            refs.get("source_attachment_count") or 0
        ):
            issues.append(prefix + "source_attachment_count")
        for media in (*source_images, *source_attachments):
            media_path = Path(str(media.get("path") or ""))
            if not media_path.exists():
                issues.append(prefix + "missing_media:" + str(media_path))
            elif _sha256(media_path) != media.get("sha256"):
                issues.append(prefix + "media_sha256:" + str(media_path))
    coverage = dataset.get("coverage") or {}
    if int(coverage.get("case_count") or 0) != len(cases):
        issues.append("coverage_case_count")
    if int(coverage.get("unique_manual_card_count") or 0) != len(seen_cards):
        issues.append("coverage_card_count")
    if int(coverage.get("unique_source_section_count") or 0) != len(
        seen_sections
    ):
        issues.append("coverage_source_section_count")
    return {
        "schema_version":
            "debug_agent_system.aoi_document_qa_pilot.validation.v2",
        "benchmark_id": BENCHMARK_ID,
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
        "coverage": coverage,
    }


def render_markdown(dataset: dict[str, Any]) -> str:
    cases = dataset.get("cases") or []
    coverage = dataset.get("coverage") or {}
    lines = [
        "# AOI Document QA Pilot v1（原文重做版）：Query 与参考答案",
        "",
        "> Query 为人工改写的自然现场问题；答案来自原始 SOP 对应 Section 的正文与媒体。",
        "> 答案同时保留 Section 原图与附件；图片原样展示但不做 OCR 推断。",
        "> KG v2 卡片的 actions、required_info 以及通用“验证与边界”话术",
        "> 均不进入答案；卡片只辅助 Section 路由和源图片绑定。",
        "> 本批次是文档参考答案，不是独立专家诊断 Gold。",
        "",
        f"- Case 总数：{len(cases)}",
        f"- 独立原文 Section：{coverage.get('unique_source_section_count', 0)}",
        f"- 路由卡片：{coverage.get('unique_manual_card_count', 0)}",
        f"- 原文图片：{coverage.get('included_source_image_count', 0)}",
        f"- 原文附件：{coverage.get('included_source_attachment_count', 0)}",
        f"- 与历史 Query 的最高相似度：{coverage.get('legacy_similarity_max', 0)}",
        "",
    ]
    for case in cases:
        refs = case["source_refs"]
        lines.extend(
            [
                f"## {case['case_id']} · {refs['source_heading_title']}",
                "",
                f"- SOP Section：`{refs['canonical_section_id']}`",
                "- 原文路径：" + " > ".join(refs["source_heading_path"]),
                f"- 原文快照：`{refs['fetch_json_path']}`",
                "- 路由及源媒体绑定："
                f"`{refs['manual_card_path']}`",
                f"- 原文图片：{refs['included_image_count']} 张",
                f"- 原文附件：{refs['source_attachment_count']} 个",
                "",
                "**Query**",
                "",
                case["query"],
                "",
                "**参考答案**",
                "",
                case["answer_gold"]["reference_answer"],
                "",
            ]
        )
        images = case["answer_gold"].get("source_images") or []
        if images:
            lines.extend(["**原文图片证据**", ""])
            for image in images:
                caption = _clean(image.get("caption")).replace("[", "（").replace(
                    "]", "）"
                )
                lines.extend(
                    [
                        f"![{caption}](<../{image['path']}>)",
                        "",
                    ]
                )
        attachments = case["answer_gold"].get("source_attachments") or []
        if attachments:
            lines.extend(["**原文附件**", ""])
            for attachment in attachments:
                display_name = _clean(attachment.get("display_name"))
                lines.append(
                    f"- [{display_name}](<../{attachment['path']}>)"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="aoi-document-qa-pilot")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--cards-root", type=Path, default=DEFAULT_CARDS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        dataset = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        dataset = build_dataset(args.seeds, args.cards_root)
        _write_json(args.out, dataset)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(dataset), encoding="utf-8")
    report = validate_dataset(dataset)
    _write_json(args.report_out, report)
    print(
        json.dumps(
            {
                "dataset": str(args.out),
                "report": str(args.report_out),
                "markdown": str(args.markdown_out),
                "status": report["status"],
                "coverage": report["coverage"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
