"""Corpus-backed noun discovery and review workflow for KG_v2.

The authoritative terminology projection intentionally accepts only
structured KG fields and reviewed ontology entries.  This module fills the
gap before review: it scans heterogeneous corpora, attaches auditable
frequency/context evidence to domain noun candidates, proposes surface
variants and typed relationships, and writes a non-authoritative review
queue.

No proposal mutates ``entity_ontology.json`` until a reviewer explicitly
approves it.  Corpus frequency is evidence of usage, never proof that two
expressions are equivalent or that a relationship is true.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from debug_agent_system.knowledge_v2.entity_terminology import (
    APPROVED_ALIAS_RELATION_TYPES,
    ENTITY_RELATION_TYPES,
    NOUN_CONCEPT_TYPES,
)
from debug_agent_system.knowledge_v2.terminology import (
    normalize_term,
    write_terminology_layer,
)


DISCOVERY_CONFIG_SCHEMA = "kg_v2.noun_discovery_config.v1"
DISCOVERY_QUEUE_SCHEMA = "kg_v2.noun_discovery_queue.v1"
DISCOVERY_REPORT_SCHEMA = "kg_v2.noun_discovery_report.v1"
NOUN_INVENTORY_SCHEMA = "kg_v2.noun_terminology_inventory.v1"
DISCOVERY_QUEUE_FILE = "noun_discovery_candidates.json"
ENTITY_ONTOLOGY_SCHEMA = "kg_v2.entity_ontology.v1"

_FULL_MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<prefix>SI|SY)[-_]?"
    r"(?P<base>\d{3,4}[TDCEL]?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BARE_MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<base>\d{3,4}[TDCEL])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_MODEL_CONTEXT_RE = re.compile(
    r"(?:设备|型号|机型|整机|签单|发货|交付|产品|"
    r"订单|项目|现场|demo|样机)",
    re.IGNORECASE,
)
_FILE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9._-])"
    r"(?P<token>[A-Za-z][A-Za-z0-9_-]{0,48}"
    r"(?:\.[A-Za-z0-9_-]{1,24}){1,4})"
    r"(?![A-Za-z0-9._-])"
)
_DYNAMIC_TOKEN_RE = re.compile(
    r"(?:19|20)\d{2}[-_.]?\d{2}[-_.]?\d{2}|(?<!\d)\d{8}(?!\d)"
)
_EXPLICIT_ALIAS_RE = re.compile(
    r"(?:^|[，。；：、;:\s])"
    r"(?P<left>[\u4e00-\u9fffA-Za-z0-9+._-]{2,24})"
    r"\s*(?P<marker>简称|又称|俗称|也叫)\s*"
    r"(?P<right>[\u4e00-\u9fffA-Za-z0-9+._-]{2,24})"
    r"(?=$|[，。；、;:\s])",
    re.IGNORECASE,
)
_ARTIFACT_EXTENSIONS = {
    "cfg",
    "csv",
    "db",
    "ini",
    "json",
    "log",
    "sqlite",
    "toml",
    "xml",
    "yaml",
    "yml",
}
_SOFTWARE_EXTENSIONS = {"dll", "exe"}

REVIEW_FIELDS = {
    "review_status",
    "selected_action",
    "selected_canonical_name",
    "selected_concept_type",
    "selected_relation",
    "selected_target_key",
    "selected_concept_key",
    "approved_relation_type",
    "reviewed_by",
    "reviewed_at",
    "review_note",
}


@dataclass(frozen=True)
class CorpusRecord:
    source_kind: str
    source_id: str
    source_path: str
    text: str


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_noun_discovery_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "terminology" / "noun_discovery_config.json"
    payload = _load_json(path, {})
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != DISCOVERY_CONFIG_SCHEMA
    ):
        raise ValueError("invalid_noun_discovery_config")
    for field in ("candidate_terms", "variant_groups"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"invalid_noun_discovery_config_field:{field}")
    concept_keys: set[str] = set()
    for item in payload["candidate_terms"]:
        if not isinstance(item, dict):
            raise ValueError("invalid_noun_candidate_spec")
        name = str(item.get("canonical_name") or "").strip()
        concept_type = str(item.get("concept_type") or "")
        if not name or concept_type not in NOUN_CONCEPT_TYPES:
            raise ValueError(f"invalid_noun_candidate_spec:{name}")
        key = _concept_key(concept_type, name)
        if key in concept_keys:
            raise ValueError(f"duplicate_noun_candidate_key:{key}")
        concept_keys.add(key)
        relation = item.get("relation")
        if relation is not None and (
            not isinstance(relation, dict)
            or str(relation.get("relation") or "")
            not in ENTITY_RELATION_TYPES
            or not str(relation.get("target_key") or "").strip()
        ):
            raise ValueError(f"invalid_noun_candidate_relation:{key}")
    for group in payload["variant_groups"]:
        if not isinstance(group, dict):
            raise ValueError("invalid_noun_variant_group")
        relation_type = str(
            group.get("suggested_relation_type") or ""
        )
        surfaces = group.get("surface_forms")
        if (
            relation_type not in APPROVED_ALIAS_RELATION_TYPES
            or not isinstance(surfaces, list)
            or not any(str(value or "").strip() for value in surfaces)
        ):
            raise ValueError("invalid_noun_variant_group")
    open_discovery = payload.get("open_discovery") or {}
    if not isinstance(open_discovery, dict):
        raise ValueError("invalid_open_noun_discovery_config")
    associations = open_discovery.get("associations") or {}
    if not isinstance(associations, dict):
        raise ValueError("invalid_noun_association_config")
    return payload


def _iter_chat_records(
    data_root: Path,
    patterns: Iterable[str],
) -> Iterator[CorpusRecord]:
    seen_message_ids: set[str] = set()
    for pattern in patterns:
        for path in sorted(data_root.glob(pattern)):
            if not path.is_file():
                continue
            with path.open(encoding="utf-8", errors="ignore") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    message_id = str(
                        item.get("message_id")
                        or f"{path}:{line_number}"
                    )
                    if message_id in seen_message_ids:
                        continue
                    seen_message_ids.add(message_id)
                    text = str(
                        item.get("plain_text")
                        or item.get("content")
                        or ""
                    ).strip()
                    if not text:
                        continue
                    yield CorpusRecord(
                        source_kind="group_chat",
                        source_id=message_id,
                        source_path=str(path.relative_to(data_root)),
                        text=text,
                    )


def _iter_chunk_records(
    data_root: Path,
    patterns: Iterable[str],
) -> Iterator[CorpusRecord]:
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(data_root.glob(pattern)):
            payload = _load_json(path, [])
            if not isinstance(payload, list):
                continue
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata") or {}
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                source_id = str(
                    metadata.get("chunk_id")
                    or metadata.get("source_id")
                    or f"{path}:{index}"
                )
                identity = _stable_id("chunk-record", source_id, text)
                if identity in seen:
                    continue
                seen.add(identity)
                yield CorpusRecord(
                    source_kind="document_chunk",
                    source_id=source_id,
                    source_path=str(path.relative_to(data_root)),
                    text=text,
                )


def _iter_support_records(
    data_root: Path,
    patterns: Iterable[str],
) -> Iterator[CorpusRecord]:
    seen_record_ids: set[str] = set()
    ignored_fields = {
        "处理人",
        "现场",
        "SourceID",
        "图片",
        "未命名",
    }
    for pattern in patterns:
        for path in sorted(data_root.glob(pattern)):
            payload = _load_json(path, {})
            if not isinstance(payload, dict):
                continue
            for index, item in enumerate(payload.get("objects") or []):
                if not isinstance(item, dict):
                    continue
                record_id = str(
                    item.get("record_id") or f"{path}:{index}"
                )
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                fields = item.get("fields") or {}
                text = " ".join(
                    str(value)
                    for key, value in fields.items()
                    if key not in ignored_fields
                    and isinstance(value, (str, int, float))
                    and str(value).strip()
                ).strip()
                if not text:
                    continue
                yield CorpusRecord(
                    source_kind="support_record",
                    source_id=record_id,
                    source_path=str(path.relative_to(data_root)),
                    text=text,
                )


def iter_corpus_records(
    root: str | Path,
    config: dict[str, Any],
) -> Iterator[CorpusRecord]:
    data_root = Path(root).parent
    sources = config.get("corpus_sources") or {}
    yield from _iter_chat_records(
        data_root,
        sources.get("chat_jsonl") or [],
    )
    yield from _iter_chunk_records(
        data_root,
        sources.get("document_chunks") or [],
    )
    yield from _iter_support_records(
        data_root,
        sources.get("support_records") or [],
    )


def _term_pattern(surface: str) -> re.Pattern[str]:
    escaped = re.escape(surface)
    if re.fullmatch(r"[A-Za-z0-9+._ -]+", surface):
        return re.compile(
            rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    return re.compile(escaped, re.IGNORECASE)


def _snippet(text: str, start: int, end: int, radius: int = 72) -> str:
    compact = " ".join(text.split())
    # Offsets may move after whitespace compaction; locating the surface again
    # is more useful than preserving byte-accurate offsets in a review sample.
    needle = " ".join(text[start:end].split())
    position = compact.lower().find(needle.lower()) if needle else -1
    if position < 0:
        position = max(0, min(start, len(compact)))
    left = max(0, position - radius)
    right = min(len(compact), position + len(needle) + radius)
    return compact[left:right]


def _collect_term_evidence(
    records: Iterable[CorpusRecord],
    terms: list[str],
    *,
    max_examples: int,
) -> tuple[
    dict[str, Counter[str]],
    dict[str, set[str]],
    dict[str, list[dict[str, str]]],
    Counter[str],
]:
    patterns = {term: _term_pattern(term) for term in terms}
    occurrences: dict[str, Counter[str]] = {
        term: Counter() for term in terms
    }
    record_ids: dict[str, set[str]] = {
        term: set() for term in terms
    }
    examples: dict[str, list[dict[str, str]]] = {
        term: [] for term in terms
    }
    corpus_counts: Counter[str] = Counter()
    for record in records:
        corpus_counts[record.source_kind] += 1
        for term, pattern in patterns.items():
            matches = list(pattern.finditer(record.text))
            if not matches:
                continue
            occurrences[term][record.source_kind] += len(matches)
            record_ids[term].add(
                f"{record.source_kind}:{record.source_id}"
            )
            if len(examples[term]) >= max_examples:
                continue
            match = matches[0]
            examples[term].append({
                "source_kind": record.source_kind,
                "source_id": record.source_id,
                "source_path": record.source_path,
                "text": _snippet(
                    record.text,
                    match.start(),
                    match.end(),
                ),
            })
    return occurrences, record_ids, examples, corpus_counts


def _concept_key(concept_type: str, canonical_name: str) -> str:
    return f"{concept_type}:{normalize_term(canonical_name)}"


def _existing_concept_keys(root: Path) -> tuple[set[str], dict[str, str]]:
    concepts = _load_json(root / "objects" / "debug_concepts.json", [])
    keys: set[str] = set()
    key_by_normalized_name: dict[str, str] = {}
    if not isinstance(concepts, list):
        return keys, key_by_normalized_name
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        concept_type = str(concept.get("concept_type") or "")
        name = str(concept.get("canonical_name") or "")
        if concept_type not in NOUN_CONCEPT_TYPES or not name:
            continue
        key = _concept_key(concept_type, name)
        keys.add(key)
        key_by_normalized_name.setdefault(normalize_term(name), key)

    # The discovery queue is an audit surface, not a second terminology
    # authority.  A surface form that is already an approved alias, or is
    # already being reviewed as a context-constrained alias, must therefore
    # resolve to the target concept identity instead of being proposed again
    # as a competing noun concept.  Candidate aliases remain non-authoritative
    # in the runtime projection; this mapping only de-duplicates review work.
    ontology = _load_json(
        root / "terminology" / "entity_ontology.json",
        {},
    )
    alias_targets: dict[str, set[str]] = defaultdict(set)
    if isinstance(ontology, dict):
        # Ontology keys are authoritative and are not required to equal the
        # key mechanically derived from the canonical label (for example,
        # ``software:display driver uninstaller`` versus a whitespace-folded
        # generated key). Keep both as known identities, but resolve names and
        # aliases to the curated ontology key.
        for item in ontology.get("concepts") or []:
            if not isinstance(item, dict) or item.get("approved") is not True:
                continue
            concept_type = str(item.get("concept_type") or "")
            name = str(item.get("canonical_name") or "").strip()
            ontology_key = str(item.get("key") or "")
            if (
                concept_type not in NOUN_CONCEPT_TYPES
                or not name
                or not ontology_key
            ):
                continue
            keys.add(ontology_key)
            key_by_normalized_name[normalize_term(name)] = ontology_key
        alias_entries = list(ontology.get("aliases") or [])
        alias_entries.extend(ontology.get("alias_candidates") or [])
        for alias in alias_entries:
            if not isinstance(alias, dict):
                continue
            surface = normalize_term(alias.get("surface_form"))
            target_key = str(alias.get("concept_key") or "")
            if not surface or target_key not in keys:
                continue
            alias_targets[surface].add(target_key)
    for surface, targets in alias_targets.items():
        # Canonical names take precedence. Ambiguous aliases must not silently
        # choose a concept merely to make the queue shorter.
        if surface not in key_by_normalized_name and len(targets) == 1:
            key_by_normalized_name[surface] = next(iter(targets))
    return keys, key_by_normalized_name


def _existing_alias_mappings(root: Path) -> set[tuple[str, str]]:
    """Return settled surface-to-concept mappings used for queue de-duplication.

    This deliberately keeps the target concept in the identity.  A surface may
    also be a legacy canonical label, or may legitimately resolve to more than
    one concept under different contexts.  Such ambiguity must remain visible
    to runtime resolution, but it must not cause an already reviewed exact
    mapping to be proposed for review again.
    """

    ontology = _load_json(
        root / "terminology" / "entity_ontology.json",
        {},
    )
    if not isinstance(ontology, dict):
        return set()
    entries = list(ontology.get("aliases") or [])
    entries.extend(ontology.get("alias_candidates") or [])
    return {
        (
            normalize_term(item.get("surface_form")),
            str(item.get("concept_key") or ""),
        )
        for item in entries
        if isinstance(item, dict)
        and normalize_term(item.get("surface_form"))
        and str(item.get("concept_key") or "")
    }


def _existing_relation_triples(root: Path) -> set[tuple[str, str, str]]:
    """Return approved ontology relations already settled by review."""

    ontology = _load_json(
        root / "terminology" / "entity_ontology.json",
        {},
    )
    if not isinstance(ontology, dict):
        return set()
    return {
        (
            str(item.get("from_key") or ""),
            str(item.get("relation") or ""),
            str(item.get("to_key") or ""),
        )
        for item in ontology.get("relations") or []
        if isinstance(item, dict)
        and item.get("approved") is True
        and item.get("from_key")
        and item.get("relation")
        and item.get("to_key")
    }


def _new_observation(
    *,
    canonical_name: str,
    concept_type: str,
    discovery_method: str,
) -> dict[str, Any]:
    return {
        "canonical_name": canonical_name,
        "concept_type": concept_type,
        "discovery_method": discovery_method,
        "occurrences": Counter(),
        "record_ids": set(),
        "examples": [],
        "surface_counts": Counter(),
        "surface_occurrences": defaultdict(Counter),
        "surface_record_ids": defaultdict(set),
        "surface_examples": defaultdict(list),
    }


def _add_observation(
    observation: dict[str, Any],
    *,
    record: CorpusRecord,
    surface: str,
    start: int,
    end: int,
    max_examples: int,
) -> None:
    record_key = f"{record.source_kind}:{record.source_id}"
    observation["occurrences"][record.source_kind] += 1
    observation["record_ids"].add(record_key)
    observation["surface_counts"][surface] += 1
    observation["surface_occurrences"][surface][record.source_kind] += 1
    observation["surface_record_ids"][surface].add(record_key)
    example = {
        "source_kind": record.source_kind,
        "source_id": record.source_id,
        "source_path": record.source_path,
        "text": _snippet(record.text, start, end),
    }
    if len(observation["examples"]) < max_examples:
        observation["examples"].append(example)
    if len(observation["surface_examples"][surface]) < max_examples:
        observation["surface_examples"][surface].append(example)


def _model_name_and_key(
    base: str,
    *,
    prefix: str,
    key_by_name: dict[str, str],
) -> tuple[str, str]:
    normalized_base = base.upper()
    prefixed = f"{prefix.upper()}{normalized_base}" if prefix else ""
    for name in tuple(
        value for value in (prefixed, normalized_base) if value
    ):
        key = key_by_name.get(normalize_term(name), "")
        if key:
            return name, key
    canonical = prefixed or normalized_base
    return canonical, _concept_key("product_model", canonical)


def _merge_observation(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    max_examples: int,
) -> None:
    target["occurrences"].update(source["occurrences"])
    target["record_ids"].update(source["record_ids"])
    target["surface_counts"].update(source["surface_counts"])
    for surface, counts in source["surface_occurrences"].items():
        target["surface_occurrences"][surface].update(counts)
    for surface, record_ids in source["surface_record_ids"].items():
        target["surface_record_ids"][surface].update(record_ids)
    for example in source["examples"]:
        if len(target["examples"]) >= max_examples:
            break
        target["examples"].append(example)
    for surface, examples in source["surface_examples"].items():
        for example in examples:
            if len(target["surface_examples"][surface]) >= max_examples:
                break
            target["surface_examples"][surface].append(example)


def _file_concept_type(token: str) -> str:
    extension = token.rsplit(".", 1)[-1].lower()
    if extension in _SOFTWARE_EXTENSIONS:
        return "software"
    return "data_artifact"


def _scan_open_nouns(
    records: Iterable[CorpusRecord],
    *,
    key_by_name: dict[str, str],
    max_examples: int,
    open_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Discover high-precision model/file nouns and explicit alias claims.

    The extractors intentionally target domain-shaped identifiers instead of
    arbitrary n-grams.  Every result remains a review candidate.
    """

    model_minimum = int(open_config.get("model_minimum_count") or 3)
    artifact_minimum = int(
        open_config.get("artifact_minimum_count") or 3
    )
    observations: dict[str, dict[str, Any]] = {}
    explicit_aliases: list[dict[str, Any]] = []
    explicit_seen: set[tuple[str, str, str, str]] = set()

    for record in records:
        full_spans: list[tuple[int, int]] = []
        for match in _FULL_MODEL_RE.finditer(record.text):
            base = match.group("base").upper()
            if base[-1].isdigit():
                context_left = max(0, match.start() - 36)
                context_right = min(len(record.text), match.end() + 36)
                if not _MODEL_CONTEXT_RE.search(
                    record.text[context_left:context_right]
                ):
                    continue
            canonical, concept_key = _model_name_and_key(
                base,
                prefix=match.group("prefix"),
                key_by_name=key_by_name,
            )
            identity = concept_key
            observation = observations.setdefault(
                identity,
                _new_observation(
                    canonical_name=canonical,
                    concept_type="product_model",
                    discovery_method="product_model_pattern",
                ),
            )
            surface = match.group(0)
            _add_observation(
                observation,
                record=record,
                surface=surface,
                start=match.start(),
                end=match.end(),
                max_examples=max_examples,
            )
            full_spans.append(match.span())

        for match in _BARE_MODEL_RE.finditer(record.text):
            if any(
                left <= match.start() and match.end() <= right
                for left, right in full_spans
            ):
                continue
            left = max(0, match.start() - 36)
            right = min(len(record.text), match.end() + 36)
            if not _MODEL_CONTEXT_RE.search(record.text[left:right]):
                continue
            base = match.group("base").upper()
            canonical, concept_key = _model_name_and_key(
                base,
                prefix="",
                key_by_name=key_by_name,
            )
            observation = observations.setdefault(
                concept_key,
                _new_observation(
                    canonical_name=canonical,
                    concept_type="product_model",
                    discovery_method="contextual_product_model_pattern",
                ),
            )
            _add_observation(
                observation,
                record=record,
                surface=match.group(0),
                start=match.start(),
                end=match.end(),
                max_examples=max_examples,
            )

        for match in _FILE_TOKEN_RE.finditer(record.text):
            token = match.group("token")
            extension = token.rsplit(".", 1)[-1].lower()
            if extension not in _ARTIFACT_EXTENSIONS | _SOFTWARE_EXTENSIONS:
                continue
            if _DYNAMIC_TOKEN_RE.search(token):
                continue
            concept_type = _file_concept_type(token)
            normalized = normalize_term(token)
            concept_key = key_by_name.get(
                normalized,
                _concept_key(concept_type, token),
            )
            observation = observations.setdefault(
                concept_key,
                _new_observation(
                    canonical_name=token,
                    concept_type=concept_type,
                    discovery_method="program_or_artifact_filename",
                ),
            )
            _add_observation(
                observation,
                record=record,
                surface=token,
                start=match.start(),
                end=match.end(),
                max_examples=max_examples,
            )

        for match in _EXPLICIT_ALIAS_RE.finditer(record.text):
            left = match.group("left").strip()
            right = match.group("right").strip()
            marker = match.group("marker")
            identity = (
                normalize_term(left),
                normalize_term(right),
                marker,
                f"{record.source_kind}:{record.source_id}",
            )
            if identity in explicit_seen:
                continue
            explicit_seen.add(identity)
            explicit_aliases.append({
                "canonical_name": left,
                "surface_form": right,
                "marker": marker,
                "source_kind": record.source_kind,
                "source_id": record.source_id,
                "source_path": record.source_path,
                "text": _snippet(
                    record.text,
                    match.start(),
                    match.end(),
                ),
            })

    # Merge a contextual bare form only when the corpus itself contains one
    # unambiguous formal prefix for the same base.  This preserves SY-2600D
    # and avoids inventing SI2600D from a bare 2600D mention.
    prefixed_by_base: dict[str, list[str]] = defaultdict(list)
    bare_by_base: dict[str, str] = {}
    for key, observation in observations.items():
        if observation["concept_type"] != "product_model":
            continue
        name = str(observation["canonical_name"]).upper()
        prefixed_match = re.fullmatch(r"(SI|SY)(\d{3,4}[TDCEL]?)", name)
        if prefixed_match:
            prefixed_by_base[prefixed_match.group(2)].append(key)
        elif re.fullmatch(r"\d{3,4}[TDCEL]", name):
            bare_by_base[name] = key
    for base, bare_key in list(bare_by_base.items()):
        prefixed_keys = sorted(set(prefixed_by_base.get(base) or []))
        if len(prefixed_keys) != 1 or bare_key not in observations:
            continue
        target_key = prefixed_keys[0]
        _merge_observation(
            observations[target_key],
            observations.pop(bare_key),
            max_examples=max_examples,
        )

    return {
        key: value
        for key, value in observations.items()
        if sum(value["occurrences"].values()) >= (
            model_minimum
            if value["concept_type"] == "product_model"
            else artifact_minimum
        )
    }, explicit_aliases


def _proposal_evidence(
    observation: dict[str, Any],
) -> dict[str, Any]:
    occurrences = observation["occurrences"]
    return {
        "corpus_count": sum(occurrences.values()),
        "corpus_counts": dict(sorted(occurrences.items())),
        "source_record_count": len(observation["record_ids"]),
        "source_kind_count": sum(
            count > 0 for count in occurrences.values()
        ),
        "corpus_examples": observation["examples"],
    }


def _build_association_items(
    records: Iterable[CorpusRecord],
    *,
    surface_to_keys: dict[str, set[str]],
    canonical_by_key: dict[str, str],
    concept_type_by_key: dict[str, str],
    association_config: dict[str, Any],
    previous_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build review-only noun associations from record-level co-occurrence."""

    if not association_config.get("enabled", False):
        return []
    unique_surface_to_key = {
        surface: next(iter(keys))
        for surface, keys in surface_to_keys.items()
        if surface and len(keys) == 1
    }
    if not unique_surface_to_key:
        return []
    ordered = sorted(
        unique_surface_to_key,
        key=lambda value: (-len(value), value),
    )
    matcher = re.compile(
        "|".join(re.escape(surface) for surface in ordered),
        re.IGNORECASE,
    )
    normalized_lookup = {
        normalize_term(surface): key
        for surface, key in unique_surface_to_key.items()
    }
    document_frequency: Counter[str] = Counter()
    pair_frequency: Counter[tuple[str, str]] = Counter()
    pair_source_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_examples: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    max_terms_per_record = int(
        association_config.get("max_terms_per_record") or 20
    )
    max_examples = int(
        association_config.get("max_examples_per_candidate") or 3
    )
    excluded_types = {
        str(value)
        for value in association_config.get(
            "excluded_concept_types",
            ["subsystem"],
        )
    }
    for record in records:
        keys = {
            normalized_lookup.get(normalize_term(match.group(0)), "")
            for match in matcher.finditer(record.text)
        }
        keys.discard("")
        keys = {
            key for key in keys
            if concept_type_by_key.get(key, "") not in excluded_types
        }
        if len(keys) < 2 or len(keys) > max_terms_per_record:
            continue
        for key in keys:
            document_frequency[key] += 1
        for pair in combinations(sorted(keys), 2):
            pair_frequency[pair] += 1
            pair_source_kinds[pair].add(record.source_kind)
            if len(pair_examples[pair]) < max_examples:
                compact = " ".join(record.text.split())
                pair_examples[pair].append({
                    "source_kind": record.source_kind,
                    "source_id": record.source_id,
                    "source_path": record.source_path,
                    "text": compact[:240],
                })

    minimum_count = int(
        association_config.get("minimum_record_count") or 8
    )
    minimum_source_kinds = int(
        association_config.get("minimum_source_kinds") or 2
    )
    minimum_jaccard = float(
        association_config.get("minimum_jaccard") or 0.03
    )
    max_neighbors = int(
        association_config.get("max_neighbors_per_concept") or 3
    )
    max_candidates = int(
        association_config.get("max_candidates") or 120
    )
    ranked: list[tuple[float, int, tuple[str, str]]] = []
    metrics_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, count in pair_frequency.items():
        if (
            concept_type_by_key.get(pair[0]) == "product_model"
            and concept_type_by_key.get(pair[1]) == "product_model"
        ):
            continue
        if count < minimum_count:
            continue
        source_kind_count = len(pair_source_kinds[pair])
        if source_kind_count < minimum_source_kinds:
            continue
        denominator = (
            document_frequency[pair[0]]
            + document_frequency[pair[1]]
            - count
        )
        jaccard = count / denominator if denominator else 0.0
        if jaccard < minimum_jaccard:
            continue
        confidence_left = count / document_frequency[pair[0]]
        confidence_right = count / document_frequency[pair[1]]
        metrics_by_pair[pair] = {
            "cooccurrence_record_count": count,
            "source_kind_count": source_kind_count,
            "jaccard": round(jaccard, 6),
            "confidence_from": round(confidence_left, 6),
            "confidence_to": round(confidence_right, 6),
        }
        ranked.append((jaccard, count, pair))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))

    selected: list[tuple[str, str]] = []
    neighbor_counts: Counter[str] = Counter()
    for _, _, pair in ranked:
        if (
            neighbor_counts[pair[0]] >= max_neighbors
            or neighbor_counts[pair[1]] >= max_neighbors
        ):
            continue
        selected.append(pair)
        neighbor_counts[pair[0]] += 1
        neighbor_counts[pair[1]] += 1
        if len(selected) >= max_candidates:
            break

    items: list[dict[str, Any]] = []
    for from_key, to_key in selected:
        pair = (from_key, to_key)
        metrics = metrics_by_pair[pair]
        review_id = _stable_id(
            "noun-discovery",
            "association",
            from_key,
            to_key,
        )
        proposal = {
            "schema_version": DISCOVERY_QUEUE_SCHEMA,
            "review_id": review_id,
            "candidate_kind": "noun_association",
            "review_domain": "noun_association",
            "surface_form": canonical_by_key.get(from_key, from_key),
            "target_surface_form": canonical_by_key.get(to_key, to_key),
            "proposed_from_key": from_key,
            "proposed_relation": "associated_with",
            "proposed_to_key": to_key,
            "basis": "record_level_corpus_cooccurrence",
            "risk": "high",
            "corpus_count": metrics["cooccurrence_record_count"],
            "source_kind_count": metrics["source_kind_count"],
            "association_metrics": metrics,
            "corpus_examples": pair_examples[pair],
            "review_status": "pending",
            "allowed_actions": ["approve", "reject", "defer"],
            "approval_requirements": [
                "selected_relation",
                "selected_target_key",
                "reviewed_by",
            ],
            "non_authoritative_note": (
                "共现只表示语料关联，不证明部件、隶属或因果关系。"
            ),
        }
        items.append(_restore_review(
            proposal,
            previous_by_id.get(review_id),
        ))
    return items


def _restore_review(
    item: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in item.items()
        if key not in REVIEW_FIELDS and key != "content_hash"
    }
    fingerprint = _content_hash(payload)
    item["content_hash"] = fingerprint
    if not previous:
        return item
    if str(previous.get("content_hash") or "") == fingerprint:
        for field in REVIEW_FIELDS:
            if field in previous:
                item[field] = previous[field]
    elif str(previous.get("review_status") or "") not in {
        "",
        "pending",
    }:
        item["review_status"] = "needs_re_review"
        item["previous_decision"] = {
            field: previous.get(field)
            for field in REVIEW_FIELDS
            if field in previous
        }
    return item


def build_noun_discovery_items(
    root: str | Path,
    *,
    existing_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build evidence-backed entity, variant and relation proposals."""

    kg_root = Path(root)
    config = load_noun_discovery_config(kg_root)
    specs = [
        item
        for item in config["candidate_terms"]
        if isinstance(item, dict)
        and str(item.get("canonical_name") or "").strip()
    ]
    variant_groups = [
        item
        for item in config["variant_groups"]
        if isinstance(item, dict)
    ]
    all_terms = sorted({
        str(item["canonical_name"]).strip()
        for item in specs
    } | {
        str(surface).strip()
        for group in variant_groups
        for surface in group.get("surface_forms") or []
        if str(surface).strip()
    })
    max_examples = int(config.get("max_examples_per_candidate") or 5)
    occurrences, record_ids, examples, corpus_counts = (
        _collect_term_evidence(
            iter_corpus_records(kg_root, config),
            all_terms,
            max_examples=max_examples,
        )
    )
    existing_keys, key_by_name = _existing_concept_keys(kg_root)
    existing_alias_mappings = _existing_alias_mappings(kg_root)
    existing_relation_triples = _existing_relation_triples(kg_root)
    existing_concepts = _load_json(
        kg_root / "objects" / "debug_concepts.json",
        [],
    )
    canonical_by_key = {
        _concept_key(
            str(item.get("concept_type") or ""),
            str(item.get("canonical_name") or ""),
        ): str(item.get("canonical_name") or "")
        for item in existing_concepts
        if isinstance(item, dict)
        and str(item.get("concept_type") or "") in NOUN_CONCEPT_TYPES
        and str(item.get("canonical_name") or "").strip()
    }
    concept_type_by_key = {
        _concept_key(
            str(item.get("concept_type") or ""),
            str(item.get("canonical_name") or ""),
        ): str(item.get("concept_type") or "")
        for item in existing_concepts
        if isinstance(item, dict)
        and str(item.get("concept_type") or "") in NOUN_CONCEPT_TYPES
        and str(item.get("canonical_name") or "").strip()
    }
    spec_by_name = {
        normalize_term(item["canonical_name"]): item
        for item in specs
    }
    proposed_keys = {
        key_by_name.get(
            normalize_term(item.get("canonical_name")),
            _concept_key(
                str(item.get("concept_type") or ""),
                str(item.get("canonical_name") or ""),
            ),
        )
        for item in specs
    }
    for spec in specs:
        resolved_key = key_by_name.get(
            normalize_term(spec.get("canonical_name")),
            _concept_key(
                str(spec.get("concept_type") or ""),
                str(spec.get("canonical_name") or ""),
            ),
        )
        canonical_by_key.setdefault(
            resolved_key,
            str(spec.get("canonical_name") or ""),
        )
        concept_type_by_key.setdefault(
            resolved_key,
            str(spec.get("concept_type") or ""),
        )
    known_keys = existing_keys | proposed_keys
    previous_by_id = {
        str(item.get("review_id") or ""): item
        for item in existing_items or []
        if isinstance(item, dict) and item.get("review_id")
    }
    items: list[dict[str, Any]] = []
    open_config = config.get("open_discovery") or {}
    open_observations: dict[str, dict[str, Any]] = {}
    explicit_aliases: list[dict[str, Any]] = []
    if open_config.get("enabled", False):
        open_observations, explicit_aliases = _scan_open_nouns(
            iter_corpus_records(kg_root, config),
            key_by_name=key_by_name,
            max_examples=max_examples,
            open_config=open_config,
        )
        known_keys |= set(open_observations)

    for spec in specs:
        name = str(spec["canonical_name"]).strip()
        concept_type = str(spec.get("concept_type") or "")
        if concept_type not in NOUN_CONCEPT_TYPES:
            raise ValueError(
                f"invalid_discovery_concept_type:{concept_type}:{name}"
            )
        total = sum(occurrences[name].values())
        source_kinds = sum(
            count > 0 for count in occurrences[name].values()
        )
        minimum_count = int(spec.get("minimum_count") or 2)
        minimum_source_kinds = int(
            spec.get("minimum_source_kinds") or 1
        )
        if total < minimum_count or source_kinds < minimum_source_kinds:
            continue
        concept_key = key_by_name.get(
            normalize_term(name),
            _concept_key(concept_type, name),
        )
        if concept_key not in existing_keys:
            proposal = {
                "schema_version": DISCOVERY_QUEUE_SCHEMA,
                "review_id": _stable_id(
                    "noun-discovery",
                    "new_concept",
                    concept_key,
                ),
                "candidate_kind": "new_noun_concept",
                "review_domain": "noun_entity",
                "canonical_name": name,
                "normalized_form": normalize_term(name),
                "proposed_concept_key": concept_key,
                "proposed_concept_type": concept_type,
                "definition": str(
                    spec.get("definition")
                    or f"从 Debug 语料发现的候选名词实体：{name}"
                ),
                "risk": str(spec.get("risk") or "medium"),
                "corpus_count": total,
                "corpus_counts": dict(sorted(occurrences[name].items())),
                "source_record_count": len(record_ids[name]),
                "source_kind_count": source_kinds,
                "corpus_examples": examples[name],
                "review_status": "pending",
                "allowed_actions": ["approve", "reject", "defer"],
                "approval_requirements": [
                    "selected_canonical_name",
                    "selected_concept_type",
                    "reviewed_by",
                ],
            }
            items.append(_restore_review(
                proposal,
                previous_by_id.get(proposal["review_id"]),
            ))

        relation = spec.get("relation")
        if not isinstance(relation, dict):
            continue
        relation_name = str(relation.get("relation") or "")
        target_key = str(relation.get("target_key") or "")
        if (
            relation_name not in ENTITY_RELATION_TYPES
            or target_key not in known_keys
        ):
            continue
        relation_item = {
            "schema_version": DISCOVERY_QUEUE_SCHEMA,
            "review_id": _stable_id(
                "noun-discovery",
                "relation",
                concept_key,
                relation_name,
                target_key,
            ),
            "candidate_kind": "noun_relation",
            "review_domain": "noun_relation",
            "surface_form": name,
            "proposed_from_key": concept_key,
            "proposed_relation": relation_name,
            "proposed_to_key": target_key,
            "basis": "corpus_candidate_catalog",
            "risk": str(relation.get("risk") or spec.get("risk") or "medium"),
            "corpus_count": total,
            "corpus_counts": dict(sorted(occurrences[name].items())),
            "source_record_count": len(record_ids[name]),
            "corpus_examples": examples[name],
            "review_status": "pending",
            "allowed_actions": ["approve", "reject", "defer"],
            "approval_requirements": [
                "selected_relation",
                "selected_target_key",
                "reviewed_by",
            ],
        }
        items.append(_restore_review(
            relation_item,
            previous_by_id.get(relation_item["review_id"]),
        ))

    for group in variant_groups:
        canonical_name = str(
            group.get("canonical_name") or ""
        ).strip()
        target_key = str(group.get("concept_key") or "")
        if not target_key:
            target_key = key_by_name.get(normalize_term(canonical_name), "")
        if not target_key:
            target_spec = spec_by_name.get(normalize_term(canonical_name))
            if target_spec:
                target_key = _concept_key(
                    str(target_spec.get("concept_type") or ""),
                    canonical_name,
                )
        if not target_key or target_key not in known_keys:
            continue
        for surface in group.get("surface_forms") or []:
            surface = str(surface).strip()
            if not surface or normalize_term(surface) == normalize_term(
                canonical_name
            ):
                continue
            # Do not ask reviewers to approve the same mapping twice. This
            # includes curated aliases and context-constrained alias
            # candidates; the latter remain non-authoritative at runtime.
            if (
                key_by_name.get(normalize_term(surface)) == target_key
                or (normalize_term(surface), target_key)
                in existing_alias_mappings
            ):
                continue
            total = sum(occurrences[surface].values())
            if total < int(group.get("minimum_count") or 2):
                continue
            relation_type = str(
                group.get("suggested_relation_type")
                or "colloquial_alias"
            )
            proposal = {
                "schema_version": DISCOVERY_QUEUE_SCHEMA,
                "review_id": _stable_id(
                    "noun-discovery",
                    "surface_variant",
                    normalize_term(surface),
                    target_key,
                ),
                "candidate_kind": "noun_surface_variant",
                "review_domain": "noun_variant",
                "surface_form": surface,
                "normalized_form": normalize_term(surface),
                "suggested_canonical_name": canonical_name,
                "suggested_concept_key": target_key,
                "suggested_relation_type": relation_type,
                "risk": str(group.get("risk") or "medium"),
                "corpus_count": total,
                "corpus_counts": dict(
                    sorted(occurrences[surface].items())
                ),
                "source_record_count": len(record_ids[surface]),
                "corpus_examples": examples[surface],
                "review_status": "pending",
                "allowed_actions": ["approve", "reject", "defer"],
                "approval_requirements": [
                    "selected_concept_key",
                    "approved_relation_type",
                    "reviewed_by",
                ],
            }
            items.append(_restore_review(
                proposal,
                previous_by_id.get(proposal["review_id"]),
            ))

    for concept_key, observation in open_observations.items():
        canonical_name = str(observation["canonical_name"])
        concept_type = str(observation["concept_type"])
        evidence = _proposal_evidence(observation)
        canonical_by_key.setdefault(concept_key, canonical_name)
        concept_type_by_key.setdefault(concept_key, concept_type)
        key_by_name.setdefault(
            normalize_term(canonical_name),
            concept_key,
        )
        if (
            concept_key not in existing_keys
            and concept_key not in proposed_keys
        ):
            review_id = _stable_id(
                "noun-discovery",
                "new_concept",
                concept_key,
            )
            proposal = {
                "schema_version": DISCOVERY_QUEUE_SCHEMA,
                "review_id": review_id,
                "candidate_kind": "new_noun_concept",
                "review_domain": "noun_entity",
                "canonical_name": canonical_name,
                "normalized_form": normalize_term(canonical_name),
                "proposed_concept_key": concept_key,
                "proposed_concept_type": concept_type,
                "definition": (
                    "从 Debug 多源语料的开放式专名模式发现："
                    f"{canonical_name}"
                ),
                "discovery_method": observation["discovery_method"],
                "risk": (
                    "medium"
                    if concept_type == "product_model"
                    else "high"
                ),
                **evidence,
                "review_status": "pending",
                "allowed_actions": ["approve", "reject", "defer"],
                "approval_requirements": [
                    "selected_canonical_name",
                    "selected_concept_type",
                    "reviewed_by",
                ],
            }
            items.append(_restore_review(
                proposal,
                previous_by_id.get(review_id),
            ))
            if (
                concept_type == "product_model"
                and "equipment:aoi设备" in known_keys
            ):
                relation_id = _stable_id(
                    "noun-discovery",
                    "relation",
                    concept_key,
                    "model_of",
                    "equipment:aoi设备",
                )
                relation_item = {
                    "schema_version": DISCOVERY_QUEUE_SCHEMA,
                    "review_id": relation_id,
                    "candidate_kind": "noun_relation",
                    "review_domain": "noun_relation",
                    "surface_form": canonical_name,
                    "proposed_from_key": concept_key,
                    "proposed_relation": "model_of",
                    "proposed_to_key": "equipment:aoi设备",
                    "basis": "product_model_name_pattern",
                    "risk": "medium",
                    **evidence,
                    "review_status": "pending",
                    "allowed_actions": [
                        "approve",
                        "reject",
                        "defer",
                    ],
                    "approval_requirements": [
                        "selected_relation",
                        "selected_target_key",
                        "reviewed_by",
                    ],
                }
                items.append(_restore_review(
                    relation_item,
                    previous_by_id.get(relation_id),
                ))

        variant_minimum = int(
            open_config.get("variant_minimum_count") or 2
        )
        for surface, count in observation["surface_counts"].items():
            if (
                count < variant_minimum
                or normalize_term(surface)
                == normalize_term(canonical_name)
                or (normalize_term(surface), concept_key)
                in existing_alias_mappings
            ):
                continue
            relation_type = (
                "abbreviation"
                if (
                    concept_type == "product_model"
                    and not str(surface).upper().startswith("SI")
                )
                else "exact_synonym"
            )
            review_id = _stable_id(
                "noun-discovery",
                "surface_variant",
                normalize_term(surface),
                concept_key,
            )
            proposal = {
                "schema_version": DISCOVERY_QUEUE_SCHEMA,
                "review_id": review_id,
                "candidate_kind": "noun_surface_variant",
                "review_domain": "noun_variant",
                "surface_form": surface,
                "normalized_form": normalize_term(surface),
                "suggested_canonical_name": canonical_name,
                "suggested_concept_key": concept_key,
                "suggested_relation_type": relation_type,
                "discovery_method": observation["discovery_method"],
                "risk": "medium",
                "corpus_count": count,
                "corpus_counts": dict(sorted(
                    observation["surface_occurrences"][surface].items()
                )),
                "source_record_count": len(
                    observation["surface_record_ids"][surface]
                ),
                "corpus_examples": (
                    observation["surface_examples"][surface]
                ),
                "review_status": "pending",
                "allowed_actions": ["approve", "reject", "defer"],
                "approval_requirements": [
                    "selected_concept_key",
                    "approved_relation_type",
                    "reviewed_by",
                ],
            }
            items.append(_restore_review(
                proposal,
                previous_by_id.get(review_id),
            ))

    explicit_minimum = int(
        open_config.get("explicit_alias_minimum_count") or 2
    )
    explicit_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for alias in explicit_aliases:
        explicit_groups[(
            str(alias["canonical_name"]),
            str(alias["surface_form"]),
            str(alias["marker"]),
        )].append(alias)
    relation_by_marker = {
        "简称": "abbreviation",
        "又称": "exact_synonym",
        "俗称": "colloquial_alias",
        "也叫": "colloquial_alias",
    }
    for (
        canonical_name,
        surface,
        marker,
    ), alias_evidence in explicit_groups.items():
        if len(alias_evidence) < explicit_minimum:
            continue
        target_key = key_by_name.get(
            normalize_term(canonical_name),
            "",
        )
        if (
            target_key
            and (normalize_term(surface), target_key)
            in existing_alias_mappings
        ):
            continue
        review_id = _stable_id(
            "noun-discovery",
            "surface_variant",
            normalize_term(surface),
            target_key or normalize_term(canonical_name),
        )
        proposal = {
            "schema_version": DISCOVERY_QUEUE_SCHEMA,
            "review_id": review_id,
            "candidate_kind": "noun_surface_variant",
            "review_domain": "noun_variant",
            "surface_form": surface,
            "normalized_form": normalize_term(surface),
            "suggested_canonical_name": canonical_name,
            "suggested_concept_key": target_key,
            "suggested_relation_type": relation_by_marker[marker],
            "discovery_method": "explicit_alias_statement",
            "risk": "high",
            "corpus_count": len(alias_evidence),
            "corpus_counts": dict(sorted(Counter(
                item["source_kind"] for item in alias_evidence
            ).items())),
            "source_record_count": len({
                f"{item['source_kind']}:{item['source_id']}"
                for item in alias_evidence
            }),
            "corpus_examples": alias_evidence[:max_examples],
            "review_status": "pending",
            "allowed_actions": ["approve", "reject", "defer"],
            "approval_requirements": [
                "selected_concept_key",
                "approved_relation_type",
                "reviewed_by",
            ],
        }
        items.append(_restore_review(
            proposal,
            previous_by_id.get(review_id),
        ))

    surface_to_keys: dict[str, set[str]] = defaultdict(set)
    for key, name in canonical_by_key.items():
        if "/" in name or "／" in name:
            continue
        surface_to_keys[name].add(key)
    for item in items:
        kind = str(item.get("candidate_kind") or "")
        if kind == "new_noun_concept":
            key = str(item.get("proposed_concept_key") or "")
            name = str(item.get("canonical_name") or "")
            if key and name:
                canonical_by_key.setdefault(key, name)
                concept_type_by_key.setdefault(
                    key,
                    str(item.get("proposed_concept_type") or ""),
                )
                surface_to_keys[name].add(key)
        elif kind == "noun_surface_variant":
            key = str(item.get("suggested_concept_key") or "")
            surface = str(item.get("surface_form") or "")
            if key and surface:
                surface_to_keys[surface].add(key)
    association_items = _build_association_items(
        iter_corpus_records(kg_root, config),
        surface_to_keys=surface_to_keys,
        canonical_by_key=canonical_by_key,
        concept_type_by_key=concept_type_by_key,
        association_config=open_config.get("associations") or {},
        previous_by_id=previous_by_id,
    )
    items.extend(association_items)
    # Do not ask reviewers to approve a relation that is already present in
    # the authoritative ontology.  This is deliberately exact: a different
    # endpoint, relation type, scope or unresolved topology remains visible.
    items = [
        item
        for item in items
        if item.get("candidate_kind") not in {
            "noun_relation",
            "noun_association",
        }
        or (
            str(item.get("proposed_from_key") or ""),
            str(item.get("proposed_relation") or ""),
            str(item.get("proposed_to_key") or ""),
        ) not in existing_relation_triples
    ]
    # A catalogued term can also be rediscovered by an open extractor.  The
    # stable review ID represents one decision, so retain the first (curated
    # catalog) proposal instead of presenting duplicate review rows.
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in items:
        deduplicated.setdefault(str(item["review_id"]), item)
    items = list(deduplicated.values())

    priority = {
        "new_noun_concept": 0,
        "noun_surface_variant": 1,
        "noun_relation": 2,
        "noun_association": 3,
    }
    items.sort(key=lambda item: (
        priority.get(str(item.get("candidate_kind") or ""), 9),
        -int(item.get("corpus_count") or 0),
        str(
            item.get("canonical_name")
            or item.get("surface_form")
            or ""
        ),
    ))
    report = {
        "schema_version": DISCOVERY_REPORT_SCHEMA,
        "candidate_count": len(items),
        "new_concept_count": sum(
            item["candidate_kind"] == "new_noun_concept"
            for item in items
        ),
        "surface_variant_count": sum(
            item["candidate_kind"] == "noun_surface_variant"
            for item in items
        ),
        "relation_candidate_count": sum(
            item["candidate_kind"] == "noun_relation"
            for item in items
        ),
        "association_candidate_count": sum(
            item["candidate_kind"] == "noun_association"
            for item in items
        ),
        "pending_count": sum(
            item.get("review_status") in {"pending", "needs_re_review"}
            for item in items
        ),
        "corpus_record_counts": dict(sorted(corpus_counts.items())),
        "candidate_concept_type_counts": dict(sorted(Counter(
            str(item.get("proposed_concept_type") or "")
            for item in items
            if item.get("candidate_kind") == "new_noun_concept"
        ).items())),
    }
    return items, report


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_noun_discovery_markdown(
    items: list[dict[str, Any]],
    report: dict[str, Any],
) -> str:
    """Render the complete candidate concept/variant/relation review table."""

    source_counts = report.get("corpus_record_counts") or {}
    lines = [
        "# KG_v2 多源名词发现与审核清单",
        "",
        "> 该清单来自群聊、文档 Chunk 和技术支持记录；全部条目在人工审核前均为"
        "非权威候选，不可锁定 Variant 或生成诊断动作。",
        "",
        "## 汇总",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 去重群聊记录 | {int(source_counts.get('group_chat') or 0)} |",
        f"| 文档 Chunk | {int(source_counts.get('document_chunk') or 0)} |",
        f"| 去重支持记录 | {int(source_counts.get('support_record') or 0)} |",
        f"| 新名词概念 | {int(report.get('new_concept_count') or 0)} |",
        f"| 变体叫法 | {int(report.get('surface_variant_count') or 0)} |",
        f"| 名词关系 | {int(report.get('relation_candidate_count') or 0)} |",
        f"| 语料共现关联 | {int(report.get('association_candidate_count') or 0)} |",
        f"| 总候选 | {int(report.get('candidate_count') or 0)} |",
        "",
        "## 新名词概念",
        "",
        "| 名词 | 类型 | 总次数 | 群聊 | 文档 | 支持记录 | 来源种类 | 风险 | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in items:
        if item.get("candidate_kind") != "new_noun_concept":
            continue
        counts = item.get("corpus_counts") or {}
        lines.append(
            "| {name} | `{kind}` | {total} | {chat} | {doc} | {support} | "
            "{source_kinds} | `{risk}` | `{status}` |".format(
                name=_markdown_cell(item.get("canonical_name")),
                kind=_markdown_cell(item.get("proposed_concept_type")),
                total=int(item.get("corpus_count") or 0),
                chat=int(counts.get("group_chat") or 0),
                doc=int(counts.get("document_chunk") or 0),
                support=int(counts.get("support_record") or 0),
                source_kinds=int(item.get("source_kind_count") or 0),
                risk=_markdown_cell(item.get("risk")),
                status=_markdown_cell(item.get("review_status")),
            )
        )

    lines.extend([
        "",
        "## 变体叫法",
        "",
        "| 现场叫法 | 建议规范名 | 建议关系 | 总次数 | 风险 | 状态 |",
        "|---|---|---|---:|---|---|",
    ])
    for item in items:
        if item.get("candidate_kind") != "noun_surface_variant":
            continue
        lines.append(
            "| {surface} | {canonical} | `{relation}` | {total} | "
            "`{risk}` | `{status}` |".format(
                surface=_markdown_cell(item.get("surface_form")),
                canonical=_markdown_cell(
                    item.get("suggested_canonical_name")
                ),
                relation=_markdown_cell(
                    item.get("suggested_relation_type")
                ),
                total=int(item.get("corpus_count") or 0),
                risk=_markdown_cell(item.get("risk")),
                status=_markdown_cell(item.get("review_status")),
            )
        )

    lines.extend([
        "",
        "## 名词关系",
        "",
        "| 起点 | 关系 | 终点 | 总次数 | 风险 | 状态 |",
        "|---|---|---|---:|---|---|",
    ])
    for item in items:
        if item.get("candidate_kind") != "noun_relation":
            continue
        lines.append(
            "| `{source}` | `{relation}` | `{target}` | {total} | "
            "`{risk}` | `{status}` |".format(
                source=_markdown_cell(item.get("proposed_from_key")),
                relation=_markdown_cell(item.get("proposed_relation")),
                target=_markdown_cell(item.get("proposed_to_key")),
                total=int(item.get("corpus_count") or 0),
                risk=_markdown_cell(item.get("risk")),
                status=_markdown_cell(item.get("review_status")),
            )
        )
    lines.extend([
        "",
        "## 语料共现关联（非结构事实）",
        "",
        "> 共现只说明两个名词经常出现在同一条记录中；不能直接推出"
        "`part_of`、`connected_to` 或因果关系。",
        "",
        "| 名词 A | 名词 B | 同记录次数 | Jaccard | 来源种类 | 风险 | 状态 |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for item in items:
        if item.get("candidate_kind") != "noun_association":
            continue
        metrics = item.get("association_metrics") or {}
        lines.append(
            "| {source} | {target} | {total} | {jaccard:.4f} | "
            "{source_kinds} | `{risk}` | `{status}` |".format(
                source=_markdown_cell(item.get("surface_form")),
                target=_markdown_cell(item.get("target_surface_form")),
                total=int(metrics.get("cooccurrence_record_count") or 0),
                jaccard=float(metrics.get("jaccard") or 0.0),
                source_kinds=int(metrics.get("source_kind_count") or 0),
                risk=_markdown_cell(item.get("risk")),
                status=_markdown_cell(item.get("review_status")),
            )
        )
    lines.extend([
        "",
        "完整上下文样例、来源路径、记录 ID、审核字段和稳定 `content_hash` 见 "
        "`../review_queue/noun_discovery_candidates.json`。",
        "",
    ])
    return "\n".join(lines)


def build_noun_terminology_inventory(
    root: str | Path,
    *,
    discovery_items: list[dict[str, Any]],
    discovery_report: dict[str, Any],
) -> dict[str, Any]:
    """Combine formal noun graph and pending proposals into one inventory."""

    kg_root = Path(root)
    concepts = [
        item
        for item in _load_json(
            kg_root / "objects" / "debug_concepts.json",
            [],
        )
        if isinstance(item, dict)
        and str(item.get("concept_type") or "") in NOUN_CONCEPT_TYPES
    ]
    expressions = {
        str(item.get("term_id") or ""): item
        for item in _load_json(
            kg_root / "objects" / "term_expressions.json",
            [],
        )
        if isinstance(item, dict)
    }
    aliases_by_concept: dict[str, list[dict[str, str]]] = defaultdict(list)
    alias_seen: set[tuple[str, str, str]] = set()
    for sense in _load_json(
        kg_root / "objects" / "term_senses.json",
        [],
    ):
        if (
            not isinstance(sense, dict)
            or not sense.get("approved")
            or str(sense.get("relation_type") or "")
            not in APPROVED_ALIAS_RELATION_TYPES
        ):
            continue
        expression = expressions.get(str(sense.get("term_id") or ""))
        surface = str((expression or {}).get("surface_form") or "")
        concept_id = str(sense.get("concept_id") or "")
        relation_type = str(sense.get("relation_type") or "")
        identity = (concept_id, normalize_term(surface), relation_type)
        if not surface or identity in alias_seen:
            continue
        alias_seen.add(identity)
        aliases_by_concept[concept_id].append({
            "surface_form": surface,
            "relation_type": relation_type,
        })

    noun_by_id = {
        str(item.get("concept_id") or ""): item
        for item in concepts
    }
    formal_relations: list[dict[str, Any]] = []
    for relation in _load_json(
        kg_root / "relations" / "edges.json",
        [],
    ):
        if not isinstance(relation, dict):
            continue
        from_id = str(relation.get("from") or "")
        to_id = str(relation.get("to") or "")
        relation_type = str(relation.get("relation") or "")
        if (
            from_id not in noun_by_id
            or to_id not in noun_by_id
            or relation_type not in ENTITY_RELATION_TYPES
        ):
            continue
        projected_relation: dict[str, Any] = {
            "from_concept_id": from_id,
            "from_name": str(noun_by_id[from_id]["canonical_name"]),
            "relation": relation_type,
            "to_concept_id": to_id,
            "to_name": str(noun_by_id[to_id]["canonical_name"]),
        }
        # Preserve instance-safety qualifiers in the human review view.
        # Without these fields a reusable type pattern such as
        # "camera connected_via CXP" is easily mistaken for an observed
        # connection on every concrete camera.
        for field in (
            "scope",
            "direction",
            "evidence_required",
            "notes",
        ):
            if field in relation:
                projected_relation[field] = relation[field]
        formal_relations.append(projected_relation)

    authoritative = []
    for concept in sorted(
        concepts,
        key=lambda item: (
            str(item.get("concept_type") or ""),
            str(item.get("canonical_name") or ""),
        ),
    ):
        concept_id = str(concept.get("concept_id") or "")
        authoritative.append({
            "concept_id": concept_id,
            "canonical_name": str(concept.get("canonical_name") or ""),
            "concept_type": str(concept.get("concept_type") or ""),
            "definition": str(concept.get("definition") or ""),
            "aliases": sorted(
                aliases_by_concept.get(concept_id, []),
                key=lambda item: (
                    item["relation_type"],
                    item["surface_form"],
                ),
            ),
        })

    candidates = [{
        key: item.get(key)
        for key in (
            "review_id",
            "candidate_kind",
            "canonical_name",
            "surface_form",
            "target_surface_form",
            "proposed_concept_key",
            "proposed_concept_type",
            "suggested_canonical_name",
            "suggested_concept_key",
            "suggested_relation_type",
            "proposed_from_key",
            "proposed_relation",
            "proposed_to_key",
            "discovery_method",
            "corpus_count",
            "source_kind_count",
            "association_metrics",
            "risk",
            "review_status",
        )
        if item.get(key) is not None and item.get(key) != ""
    } for item in discovery_items]
    pending = [
        item for item in candidates
        if str(item.get("review_status") or "")
        in {"pending", "needs_re_review"}
    ]
    review_status_counts = Counter(
        str(item.get("review_status") or "pending")
        for item in candidates
    )

    return {
        "schema_version": NOUN_INVENTORY_SCHEMA,
        "authoritative_concept_count": len(authoritative),
        "authoritative_alias_count": sum(
            len(item["aliases"]) for item in authoritative
        ),
        "authoritative_relation_count": len(formal_relations),
        "review_candidate_count": len(candidates),
        "pending_candidate_count": len(pending),
        "reviewed_candidate_count": len(candidates) - len(pending),
        "review_status_counts": dict(sorted(review_status_counts.items())),
        "discovery_summary": discovery_report,
        "authoritative_concepts": authoritative,
        "authoritative_relations": sorted(
            formal_relations,
            key=lambda item: (
                item["from_name"],
                item["relation"],
                item["to_name"],
            ),
        ),
        "pending_candidates": pending,
        "review_candidates": candidates,
    }


def render_noun_terminology_inventory_markdown(
    inventory: dict[str, Any],
) -> str:
    aliases_by_name = {
        str(item["canonical_name"]): ", ".join(
            f"{alias['surface_form']}({alias['relation_type']})"
            for alias in item.get("aliases") or []
        )
        for item in inventory.get("authoritative_concepts") or []
    }
    relations_by_name: dict[str, list[str]] = defaultdict(list)
    for relation in inventory.get("authoritative_relations") or []:
        qualifiers: list[str] = []
        if str(relation.get("scope") or "") == "type_pattern":
            qualifiers.append("类型模板")
        if relation.get("evidence_required") is True:
            qualifiers.append("需实例证据")
        direction = str(relation.get("direction") or "").strip()
        if direction:
            qualifiers.append(f"方向={direction}")
        suffix = f" [{', '.join(qualifiers)}]" if qualifiers else ""
        relations_by_name[str(relation["from_name"])].append(
            f"{relation['relation']} → {relation['to_name']}{suffix}"
        )
    lines = [
        "# KG_v2 Debug 名词术语总表与关系图清单",
        "",
        "> 本表把正式术语层与发现审核层放在同一视图中。"
        "未批准条目不参与确定性诊断和安全等价扩展。",
        "",
        "## 总览",
        "",
        "| 层级 | 概念/候选 | 变体 | 关系 |",
        "|---|---:|---:|---:|",
        "| 正式层 | {concepts} | {aliases} | {relations} |".format(
            concepts=int(
                inventory.get("authoritative_concept_count") or 0
            ),
            aliases=int(inventory.get("authoritative_alias_count") or 0),
            relations=int(
                inventory.get("authoritative_relation_count") or 0
            ),
        ),
        "| 发现审核队列 | {reviewed}（待审 {pending}） | 见候选类型 | 见候选类型 |".format(
            reviewed=int(inventory.get("review_candidate_count") or 0),
            pending=int(inventory.get("pending_candidate_count") or 0),
        ),
        "",
        "## 正式名词、变体与出边",
        "",
        "> `[类型模板, 需实例证据]` 表示允许的连接或组成模式，"
        "并非对每个现场实例都成立；实例化必须由 BOM、型号、日志、"
        "照片或原文连接说明等证据确认。",
        "",
        "| 规范名 | 类型 | 已批准变体 | 已批准关系 |",
        "|---|---|---|---|",
    ]
    for concept in inventory.get("authoritative_concepts") or []:
        name = str(concept["canonical_name"])
        lines.append(
            "| {name} | `{kind}` | {aliases} | {relations} |".format(
                name=_markdown_cell(name),
                kind=_markdown_cell(concept.get("concept_type")),
                aliases=_markdown_cell(aliases_by_name.get(name) or "—"),
                relations=_markdown_cell(
                    "; ".join(relations_by_name.get(name) or []) or "—"
                ),
            )
        )

    lines.extend([
        "",
        "## 待审核名词与变体大表",
        "",
        "| 类型 | 名称/叫法 | 建议目标 | 建议关系 | 语料次数 | 风险 | 状态 |",
        "|---|---|---|---|---:|---|---|",
    ])
    for item in inventory.get("pending_candidates") or []:
        kind = str(item.get("candidate_kind") or "")
        if kind == "new_noun_concept":
            name = str(item.get("canonical_name") or "")
            target = str(item.get("proposed_concept_type") or "")
            relation = "—"
        elif kind == "noun_surface_variant":
            name = str(item.get("surface_form") or "")
            target = str(
                item.get("suggested_canonical_name")
                or item.get("suggested_concept_key")
                or "待人工选择"
            )
            relation = str(
                item.get("suggested_relation_type") or ""
            )
        else:
            name = str(
                item.get("surface_form")
                or item.get("proposed_from_key")
                or ""
            )
            target = str(
                item.get("target_surface_form")
                or item.get("proposed_to_key")
                or ""
            )
            relation = str(item.get("proposed_relation") or "")
        lines.append(
            "| `{kind}` | {name} | {target} | `{relation}` | {count} | "
            "`{risk}` | `{status}` |".format(
                kind=_markdown_cell(kind),
                name=_markdown_cell(name),
                target=_markdown_cell(target),
                relation=_markdown_cell(relation),
                count=int(item.get("corpus_count") or 0),
                risk=_markdown_cell(item.get("risk")),
                status=_markdown_cell(item.get("review_status")),
            )
        )
    lines.extend([
        "",
        "详细语料证据与审核字段见 "
        "`../review_queue/noun_discovery_candidates.json`。",
        "",
    ])
    return "\n".join(lines)


def write_noun_discovery_queue(root: str | Path) -> dict[str, Any]:
    kg_root = Path(root)
    path = kg_root / "review_queue" / DISCOVERY_QUEUE_FILE
    existing = _load_json(path, [])
    items, report = build_noun_discovery_items(
        kg_root,
        existing_items=existing if isinstance(existing, list) else [],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = (
        kg_root / "terminology" / "noun_discovery_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = (
        kg_root / "terminology" / "noun_discovery_report.md"
    )
    markdown_path.write_text(
        render_noun_discovery_markdown(items, report),
        encoding="utf-8",
    )
    inventory = build_noun_terminology_inventory(
        kg_root,
        discovery_items=items,
        discovery_report=report,
    )
    inventory_path = (
        kg_root / "terminology" / "noun_terminology_inventory.json"
    )
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    inventory_markdown_path = (
        kg_root / "terminology" / "noun_terminology_inventory.md"
    )
    inventory_markdown_path.write_text(
        render_noun_terminology_inventory_markdown(inventory),
        encoding="utf-8",
    )
    return {
        **report,
        "queue_file": f"review_queue/{DISCOVERY_QUEUE_FILE}",
        "report_file": "terminology/noun_discovery_report.json",
        "report_markdown_file": "terminology/noun_discovery_report.md",
        "inventory_file": "terminology/noun_terminology_inventory.json",
        "inventory_markdown_file": (
            "terminology/noun_terminology_inventory.md"
        ),
    }


def _is_approved(item: dict[str, Any]) -> bool:
    return (
        str(item.get("review_status") or "")
        in {"approved", "human_approved"}
        or str(item.get("selected_action") or "") == "approve"
    )


def _has_decision_conflict(item: dict[str, Any]) -> bool:
    status = str(item.get("review_status") or "")
    action = str(item.get("selected_action") or "")
    approved_status = status in {"approved", "human_approved"}
    rejected_status = status in {"rejected", "deferred"}
    return (
        (approved_status and action in {"reject", "defer"})
        or (rejected_status and action == "approve")
    )


def apply_approved_noun_discovery(root: str | Path) -> dict[str, Any]:
    """Apply explicitly reviewed noun concepts, variants and relationships."""

    kg_root = Path(root)
    queue_path = kg_root / "review_queue" / DISCOVERY_QUEUE_FILE
    items = _load_json(queue_path, [])
    ontology_path = kg_root / "terminology" / "entity_ontology.json"
    ontology = _load_json(ontology_path, {})
    if ontology.get("schema_version") != ENTITY_ONTOLOGY_SCHEMA:
        raise ValueError("invalid_entity_ontology_file")
    concepts = [
        item for item in ontology.get("concepts") or []
        if isinstance(item, dict)
    ]
    relations = [
        item for item in ontology.get("relations") or []
        if isinstance(item, dict)
    ]
    approved_aliases = [
        item for item in ontology.get("aliases") or []
        if isinstance(item, dict)
    ]
    alias_candidates = [
        item for item in ontology.get("alias_candidates") or []
        if isinstance(item, dict)
    ]
    known_keys = {
        str(item.get("key") or "") for item in concepts
    }
    existing_keys, _ = _existing_concept_keys(kg_root)
    known_keys |= existing_keys
    rejected: list[dict[str, Any]] = []
    added = Counter()

    approved_concepts: list[dict[str, Any]] = []
    approved_variants: list[dict[str, Any]] = []
    approved_relations: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not _is_approved(item):
            continue
        if _has_decision_conflict(item):
            rejected.append({
                "review_id": str(item.get("review_id") or ""),
                "reasons": ["decision_conflict"],
            })
            continue
        kind = str(item.get("candidate_kind") or "")
        reviewer = str(item.get("reviewed_by") or "").strip()
        reasons: list[str] = []
        if not reviewer:
            reasons.append("missing_reviewer")
        if kind == "new_noun_concept":
            existing_concept_key = str(
                item.get("selected_concept_key") or ""
            ).strip()
            name = str(item.get("selected_canonical_name") or "").strip()
            concept_type = str(item.get("selected_concept_type") or "")
            relation_type = str(
                item.get("approved_relation_type") or ""
            ).strip()
            if existing_concept_key:
                if existing_concept_key not in known_keys:
                    reasons.append("invalid_concept_key")
                if relation_type not in APPROVED_ALIAS_RELATION_TYPES:
                    reasons.append("invalid_alias_relation_type")
                if not reasons:
                    approved_variants.append({
                        "surface_form": str(
                            item.get("canonical_name") or ""
                        ),
                        "concept_key": existing_concept_key,
                        "relation_type": relation_type,
                        "approved": True,
                        "review": {
                            "review_id": str(item.get("review_id") or ""),
                            "reviewed_by": reviewer,
                            "note": str(item.get("review_note") or ""),
                        },
                    })
                if reasons:
                    rejected.append({
                        "review_id": str(item.get("review_id") or ""),
                        "reasons": reasons,
                    })
                continue
            if not name:
                reasons.append("missing_canonical_name")
            if concept_type not in NOUN_CONCEPT_TYPES:
                reasons.append("invalid_concept_type")
            key = _concept_key(concept_type, name)
            if not reasons:
                approved_concepts.append({
                    "key": key,
                    "canonical_name": name,
                    "concept_type": concept_type,
                    "definition": str(
                        item.get("definition")
                        or f"经人工审核的 Debug 场景名词实体：{name}"
                    ),
                    "approved": True,
                    "review": {
                        "review_id": str(item.get("review_id") or ""),
                        "reviewed_by": reviewer,
                        "reviewed_at": str(item.get("reviewed_at") or ""),
                        "note": str(item.get("review_note") or ""),
                    },
                })
                known_keys.add(key)
                source_name = str(item.get("canonical_name") or "").strip()
                if (
                    source_name
                    and normalize_term(source_name) != normalize_term(name)
                    and relation_type in APPROVED_ALIAS_RELATION_TYPES
                ):
                    approved_variants.append({
                        "surface_form": source_name,
                        "concept_key": key,
                        "relation_type": relation_type,
                        "approved": True,
                        "review": {
                            "review_id": str(item.get("review_id") or ""),
                            "reviewed_by": reviewer,
                            "note": "source_name_for_normalized_concept",
                        },
                    })
        elif kind == "noun_surface_variant":
            concept_key = str(item.get("selected_concept_key") or "")
            relation_type = str(item.get("approved_relation_type") or "")
            if concept_key not in known_keys:
                reasons.append("invalid_concept_key")
            if relation_type not in APPROVED_ALIAS_RELATION_TYPES:
                reasons.append("invalid_alias_relation_type")
            if not reasons:
                approved_variants.append({
                    "surface_form": str(item.get("surface_form") or ""),
                    "concept_key": concept_key,
                    "relation_type": relation_type,
                    "approved": True,
                    "review": {
                        "review_id": str(item.get("review_id") or ""),
                        "reviewed_by": reviewer,
                    },
                })
        elif kind in {"noun_relation", "noun_association"}:
            from_key = str(item.get("proposed_from_key") or "")
            relation = str(item.get("selected_relation") or "")
            to_key = str(item.get("selected_target_key") or "")
            if from_key not in known_keys:
                reasons.append("invalid_from_key")
            if to_key not in known_keys:
                reasons.append("invalid_to_key")
            if relation not in ENTITY_RELATION_TYPES:
                reasons.append("invalid_relation")
            if not reasons:
                approved_relations.append({
                    "from_key": from_key,
                    "to_key": to_key,
                    "relation": relation,
                    "basis": "human_reviewed_corpus_discovery",
                    "approved": True,
                    "review": {
                        "review_id": str(item.get("review_id") or ""),
                        "reviewed_by": reviewer,
                    },
                })
        else:
            reasons.append("unsupported_candidate_kind")
        if reasons:
            rejected.append({
                "review_id": str(item.get("review_id") or ""),
                "reasons": reasons,
            })

    concept_identities = {
        str(item.get("key") or "") for item in concepts
    }
    for item in approved_concepts:
        if item["key"] not in concept_identities:
            concepts.append(item)
            concept_identities.add(item["key"])
            added["concept"] += 1
    alias_identities = {
        (
            normalize_term(item.get("surface_form")),
            str(item.get("concept_key") or ""),
            str(item.get("relation_type") or ""),
        )
        for item in approved_aliases
    }
    for item in approved_variants:
        identity = (
            normalize_term(item["surface_form"]),
            item["concept_key"],
            item["relation_type"],
        )
        if identity not in alias_identities:
            approved_aliases.append(item)
            alias_identities.add(identity)
            added["variant"] += 1
    relation_identities = {
        (
            str(item.get("from_key") or ""),
            str(item.get("relation") or ""),
            str(item.get("to_key") or ""),
        )
        for item in relations
    }
    for item in approved_relations:
        identity = (
            item["from_key"],
            item["relation"],
            item["to_key"],
        )
        if identity not in relation_identities:
            relations.append(item)
            relation_identities.add(identity)
            added["relation"] += 1

    ontology["concepts"] = concepts
    ontology["relations"] = relations
    ontology["aliases"] = approved_aliases
    ontology["alias_candidates"] = alias_candidates
    ontology_path.write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = write_terminology_layer(kg_root)
    return {
        "status": "applied",
        "approved_candidate_count": (
            len(approved_concepts)
            + len(approved_variants)
            + len(approved_relations)
        ),
        "added_concept_count": added["concept"],
        "added_surface_variant_count": added["variant"],
        "added_relation_count": added["relation"],
        "rejected_approval_count": len(rejected),
        "rejected_approvals": rejected,
        "terminology_revision": manifest["revision"],
    }


__all__ = [
    "DISCOVERY_CONFIG_SCHEMA",
    "DISCOVERY_QUEUE_FILE",
    "DISCOVERY_QUEUE_SCHEMA",
    "apply_approved_noun_discovery",
    "build_noun_discovery_items",
    "iter_corpus_records",
    "load_noun_discovery_config",
    "render_noun_discovery_markdown",
    "write_noun_discovery_queue",
]
