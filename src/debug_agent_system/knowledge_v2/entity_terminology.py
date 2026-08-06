"""Noun-centric entity projection for the KG_v2 terminology layer.

The legacy terminology projection treated structured fields such as
``SI2020T/工控机`` and ``复判站/软件`` as opaque strings.  This module keeps
those strings as context anchors for compatibility, while also emitting
atomic noun concepts and typed relationships between them.

All authoritative concepts and relationships come from structured KG fields
or the reviewed entity ontology file.  Corpus-derived aliases remain
``search_hint`` candidates until a human promotes them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable


ENTITY_ONTOLOGY_SCHEMA = "kg_v2.entity_ontology.v1"
ENTITY_RELATION_TYPES = {
    "is_a",
    "model_of",
    "part_of",
    "has_component",
    "runs_on",
    "deployed_at",
    "connected_to",
    "has_interface",
    "connected_via",
    "installed_in",
    "uses_protocol",
    "powered_by",
    "signals_to",
    "endpoint_of",
    "processed_by",
    "communicates_with",
    "identifies",
    "input_of",
    "output_of",
    "associated_with",
    "context_member",
    "driver_of",
    "firmware_of",
    "sdk_for",
    "compatible_with",
    "artifact_of",
    "process_instance_of",
    "configuration_of",
    "database_of",
    "produced_by",
    "collected_from",
    "evidence_for",
    "has_package_type",
}
NOUN_CONCEPT_TYPES = {
    "equipment",
    "product_model",
    "station",
    "workstation",
    "component",
    "software",
    "driver",
    "firmware",
    "sdk",
    "software_artifact",
    "runtime_process",
    "configuration_file",
    "database_file",
    "log_artifact",
    "diagnostic_artifact",
    "interface",
    "connection",
    "protocol",
    "workpiece",
    "subsystem",
    "external_system",
    "inspection_object",
    "package_type",
    "data_artifact",
    "identifier",
    "material",
}
APPROVED_ALIAS_RELATION_TYPES = {
    "exact_synonym",
    "colloquial_alias",
    "abbreviation",
    "english_equivalent",
    "historical_name",
    "typo_variant",
}

_MODEL_RE = re.compile(
    r"(?i)^(?:[a-z]{0,5}[-_.]?)?\d{2,5}[a-z]?(?:[-_.]\d+)?$"
    r"|^[a-z]{1,5}\d{1,5}[a-z0-9._-]*$"
)
_STATION_RE = re.compile(r"站$")
_INTERFACE_RE = re.compile(
    r"(?i)(?:usb|cxp)(?:接口|链路|连接)?$"
    r"|(?:网口|接口|协议|联网|外部触发|连接)$"
)
_SOFTWARE_RE = re.compile(
    r"(?i)(?:windows|buddy|spc)"
    r"|(?:软件|程序|驱动|固件|配置|模板|内核|"
    r"数据采集|导出|冷存储)$"
)
_DRIVER_RE = re.compile(r"(?i)(?:驱动|驱动程序)$")
_FIRMWARE_RE = re.compile(r"(?i)(?:固件|bios)$")
_SDK_RE = re.compile(r"(?i)sdk$")
_SOFTWARE_ARTIFACT_RE = re.compile(r"(?i)\.(?:exe|dll|msi)$")
_DATABASE_FILE_RE = re.compile(r"(?i)\.db$")
_COMPONENT_RE = re.compile(
    r"(?i)(?:cpu|gpu|arm|ssd)"
    r"|(?:网卡|显卡|光源|光控|"
    r"传感器|感应器|皮带|轨道|挡块|气缸|外设)$"
)
_EQUIPMENT_RE = re.compile(
    r"(?i)(?:工控机|(?:2d|3d)?相机|扫码枪|aoi设备)$"
)
_WORKPIECE_RE = re.compile(r"(?i)^(?:pcb|pcb板|线路板|板子|板卡)$")
_GENERIC_CHILD_NAMES = {
    "软件",
    "配置",
    "配置链路",
    "启动链路",
    "显示链路",
    "网络链路",
    "采集链路",
    "系统运行稳定性",
    "Windows 内核",
    "CPU散热",
    "USB外设",
}
_TEXT_SUFFIXES = ("站", "机", "卡", "板", "器", "源", "枪")


def _read_ontology(root: Path) -> dict[str, Any]:
    path = root / "terminology" / "entity_ontology.json"
    if not path.exists():
        return {
            "schema_version": ENTITY_ONTOLOGY_SCHEMA,
            "concepts": [],
            "relations": [],
            "aliases": [],
            "alias_candidates": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != ENTITY_ONTOLOGY_SCHEMA
    ):
        raise ValueError("invalid_entity_ontology_file")
    # ``aliases`` was added compatibly to v1: older checked-in ontologies
    # simply have no approved noun aliases yet.
    payload.setdefault("aliases", [])
    for field in ("concepts", "relations", "aliases", "alias_candidates"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"invalid_entity_ontology_field:{field}")
    return payload


def _split_context(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[/／]+", str(value or ""))
        if part.strip()
    ]


def _classify_entity(name: str) -> str:
    cleaned = str(name or "").strip()
    if _MODEL_RE.fullmatch(cleaned):
        return "product_model"
    if _STATION_RE.search(cleaned):
        return "station"
    if _WORKPIECE_RE.fullmatch(cleaned):
        return "workpiece"
    if _EQUIPMENT_RE.fullmatch(cleaned):
        return "equipment"
    if _INTERFACE_RE.search(cleaned):
        return "interface"
    if _DRIVER_RE.search(cleaned):
        return "driver"
    if _FIRMWARE_RE.search(cleaned):
        return "firmware"
    if _SDK_RE.search(cleaned):
        return "sdk"
    if _SOFTWARE_ARTIFACT_RE.search(cleaned):
        return "software_artifact"
    if _DATABASE_FILE_RE.search(cleaned):
        return "database_file"
    if _SOFTWARE_RE.search(cleaned):
        return "software"
    if _COMPONENT_RE.search(cleaned):
        return "component"
    if cleaned == "设备":
        return "equipment"
    return "subsystem"


def _qualified_child(parent: str, child: str) -> str:
    if child in _GENERIC_CHILD_NAMES:
        return f"{parent}{child}"
    return child


def _relation_for(parent_type: str, child_type: str) -> str:
    if child_type == "product_model" and parent_type == "equipment":
        return "model_of"
    if child_type == "software" and parent_type in {
        "equipment",
        "station",
    }:
        return "runs_on"
    if parent_type in {"equipment", "station", "software"}:
        return "part_of"
    return "connected_to"


def _iter_raw_text_files(raw_root: Path) -> Iterable[Path]:
    if not raw_root.exists():
        return []
    suffixes = {".json", ".jsonl", ".md", ".txt", ".csv", ".html", ".mhtml"}
    return (
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _corpus_counts(
    raw_root: Path,
    surfaces: set[str],
    *,
    max_file_bytes: int = 32 * 1024 * 1024,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    if not surfaces:
        return {}, {}
    ordered = sorted(surfaces, key=lambda item: (-len(item), item))
    for path in _iter_raw_text_files(raw_root):
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = str(path.relative_to(raw_root.parent))
        for surface in ordered:
            count = text.count(surface)
            if not count:
                continue
            counts[surface] += count
            if len(evidence[surface]) < 5:
                evidence[surface].append(relative)
    return dict(counts), dict(evidence)


def build_entity_projection(
    *,
    root: Path,
    objects: dict[str, list[dict[str, Any]]],
    stable_id: Callable[..., str],
    clean_text: Callable[..., str],
    normalize_term: Callable[[Any], str],
    terminology_version: str,
) -> dict[str, Any]:
    """Build atomic noun concepts, context anchors and typed relations."""

    ontology = _read_ontology(root)
    concept_by_key: dict[str, dict[str, Any]] = {}
    concepts: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    context_anchors: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_sources: list[dict[str, Any]] = []

    def ensure_concept(
        *,
        key: str,
        name: str,
        concept_type: str,
        definition: str,
        source_ids: Iterable[str],
        status: str = "approved",
    ) -> dict[str, Any]:
        if concept_type not in NOUN_CONCEPT_TYPES:
            raise ValueError(
                f"invalid_entity_concept_type:{concept_type}:{name}"
            )
        concept = concept_by_key.get(key)
        cleaned_sources = sorted({
            str(value) for value in source_ids if str(value or "")
        })
        if concept is None:
            concept = {
                "concept_id": stable_id("concept", "entity", key),
                "canonical_name": clean_text(name, limit=160),
                "concept_type": concept_type,
                "definition": clean_text(definition, limit=500),
                "status": status,
                "terminology_version": terminology_version,
                "source_object_ids": cleaned_sources,
            }
            concepts.append(concept)
            concept_by_key[key] = concept
        else:
            concept["source_object_ids"] = sorted(set(
                concept["source_object_ids"] + cleaned_sources
            ))
        return concept

    for item in ontology["concepts"]:
        if not isinstance(item, dict) or item.get("approved") is not True:
            continue
        name = str(item.get("canonical_name") or "").strip()
        key = str(item.get("key") or normalize_term(name))
        concept = ensure_concept(
            key=key,
            name=name,
            concept_type=str(item.get("concept_type") or ""),
            definition=str(
                item.get("definition")
                or f"经审核的 Debug 场景名词实体：{name}"
            ),
            source_ids=item.get("source_object_ids") or ["entity_ontology"],
        )
        canonical_sources.append({
            "surface": name,
            "concept": concept,
            "source_type": "EntityOntology",
            "source_id": key,
            "source": {},
            "relation_type": "canonical",
        })

    structured_specs = (
        ("equipment", "FaultVariant", "variant_id", "equipment_type"),
        ("subsystem", "FaultFamily", "family_id", "subsystem"),
    )
    for context_kind, source_type, pk, field in structured_specs:
        grouped: dict[str, list[str]] = defaultdict(list)
        for source in objects.get(source_type) or []:
            if not isinstance(source, dict):
                continue
            source_id = str(source.get(pk) or "")
            value = str(source.get(field) or "").strip()
            if source_id and value:
                grouped[value].append(source_id)
        for value, source_ids in sorted(grouped.items()):
            parts = _split_context(value)
            parent_name = parts[0] if parts else value
            atomic: list[dict[str, Any]] = []
            for index, raw_part in enumerate(parts or [value]):
                part = (
                    _qualified_child(parent_name, raw_part)
                    if index > 0
                    else raw_part
                )
                concept_type = _classify_entity(part)
                entity_key = f"{concept_type}:{normalize_term(part)}"
                atom = ensure_concept(
                    key=entity_key,
                    name=part,
                    concept_type=concept_type,
                    definition=(
                        f"从 KG_v2 {field}={value} 确定性拆分出的"
                        f"{concept_type} 名词实体：{part}"
                    ),
                    source_ids=source_ids,
                )
                atomic.append(atom)
                canonical_sources.append({
                    "surface": part,
                    "concept": atom,
                    "source_type": source_type,
                    "source_id": source_ids[0],
                    "source": next(
                        (
                            source
                            for source in objects.get(source_type) or []
                            if str(source.get(pk) or "") == source_ids[0]
                        ),
                        {},
                    ),
                    "relation_type": "canonical",
                })

            if len(atomic) == 1:
                # A scalar such as ``工控机`` is already an atomic entity.
                # Reusing it as the context anchor avoids artificial
                # ambiguity between two approved concepts with one surface.
                anchor = atomic[0]
            else:
                anchor_key = (
                    f"context:{context_kind}:{normalize_term(value)}"
                )
                anchor = ensure_concept(
                    key=anchor_key,
                    name=value,
                    concept_type=context_kind,
                    definition=(
                        f"KG_v2 {field} 字段的复合上下文锚点：{value}。"
                        "原子实体及其关系见 context_member。"
                    ),
                    source_ids=source_ids,
                    status="legacy",
                )
                for atom in atomic:
                    relations.append({
                        "from": anchor["concept_id"],
                        "to": atom["concept_id"],
                        "relation": "context_member",
                        "basis": (
                            f"structured_field:{source_type}.{field}"
                        ),
                    })
            context_anchors[(context_kind, value)] = anchor

            if context_kind == "equipment":
                equipment = next(
                    (
                        atom
                        for atom in atomic
                        if atom["concept_type"] == "equipment"
                    ),
                    None,
                )
                if equipment:
                    for atom in atomic:
                        if atom["concept_type"] == "product_model":
                            relations.append({
                                "from": atom["concept_id"],
                                "to": equipment["concept_id"],
                                "relation": "model_of",
                                "basis": f"structured_field:{source_type}.{field}",
                            })
            elif len(atomic) > 1:
                parent = atomic[0]
                for child in atomic[1:]:
                    relation = _relation_for(
                        str(parent["concept_type"]),
                        str(child["concept_type"]),
                    )
                    if relation in {"part_of", "runs_on", "model_of"}:
                        src, dst = child, parent
                    else:
                        src, dst = parent, child
                    relations.append({
                        "from": src["concept_id"],
                        "to": dst["concept_id"],
                        "relation": relation,
                        "basis": f"structured_field:{source_type}.{field}",
                    })

    for item in ontology["relations"]:
        if not isinstance(item, dict) or item.get("approved") is not True:
            continue
        relation = str(item.get("relation") or "")
        src = concept_by_key.get(str(item.get("from_key") or ""))
        dst = concept_by_key.get(str(item.get("to_key") or ""))
        if relation not in ENTITY_RELATION_TYPES:
            raise ValueError(f"invalid_entity_relation:{relation}")
        if src is None or dst is None:
            raise ValueError(
                "entity_relation_missing_concept:"
                f"{item.get('from_key')}:{item.get('to_key')}"
            )
        projected_relation = {
            "from": src["concept_id"],
            "to": dst["concept_id"],
            "relation": relation,
            "basis": str(item.get("basis") or "approved_entity_ontology"),
        }
        # Type-level connection patterns are useful for retrieval and answer
        # composition, but they must not be mistaken for observed instance
        # wiring.  Preserve their review guards in the generated graph so
        # downstream consumers can distinguish a reusable topology template
        # from a site-specific fact.
        for field in (
            "scope",
            "direction",
            "evidence_required",
            "notes",
        ):
            value = item.get(field)
            if value not in (None, "", []):
                projected_relation[field] = value
        relations.append(projected_relation)

    approved_alias_specs: list[dict[str, Any]] = []
    for item in ontology["aliases"]:
        if not isinstance(item, dict) or item.get("approved") is not True:
            continue
        surface = str(item.get("surface_form") or "").strip()
        concept = concept_by_key.get(str(item.get("concept_key") or ""))
        relation_type = str(item.get("relation_type") or "")
        if not surface or concept is None:
            raise ValueError("approved_entity_alias_missing_concept")
        if relation_type not in APPROVED_ALIAS_RELATION_TYPES:
            raise ValueError(
                f"invalid_approved_entity_alias_relation:{relation_type}"
            )
        approved_alias_specs.append({
            "surface": surface,
            "concept": concept,
            "source_type": "EntityOntologyAlias",
            "source_id": stable_id(
                "entity-approved-alias",
                normalize_term(surface),
                concept["concept_id"],
                relation_type,
            ),
            "source": {},
            "relation_type": relation_type,
            "approved": True,
        })

    alias_specs: list[dict[str, Any]] = []
    alias_surfaces: set[str] = set()
    for item in ontology["alias_candidates"]:
        if not isinstance(item, dict):
            continue
        surface = str(item.get("surface_form") or "").strip()
        concept = concept_by_key.get(str(item.get("concept_key") or ""))
        if not surface or concept is None:
            continue
        alias_surfaces.add(surface)
        alias_specs.append({
            "surface": surface,
            "concept": concept,
            "source_type": "RawCorpusCandidate",
            "source_id": stable_id(
                "entity-alias-candidate",
                normalize_term(surface),
                concept["concept_id"],
            ),
            "source": {},
            "relation_type": "search_hint",
            "candidate_relation_type": str(
                item.get("suggested_relation_type")
                or "colloquial_alias"
            ),
            "candidate_kind": str(
                item.get("candidate_kind") or "noun_alias"
            ),
            "risk": str(item.get("risk") or "medium"),
        })

    counts, evidence = _corpus_counts(root.parent / "raw", alias_surfaces)
    for item in alias_specs:
        item["corpus_count"] = int(counts.get(item["surface"], 0))
        item["corpus_evidence_paths"] = evidence.get(item["surface"], [])

    return {
        "concepts": concepts,
        "relations": relations,
        "context_anchors": context_anchors,
        "canonical_sources": canonical_sources,
        "alias_sources": approved_alias_specs + alias_specs,
        "concept_by_key": concept_by_key,
        "report": {
            "noun_concept_count": len(concepts),
            "noun_concept_counts": dict(sorted(Counter(
                str(item["concept_type"]) for item in concepts
            ).items())),
            "entity_relation_count": len(relations),
            "entity_relation_counts": dict(sorted(Counter(
                str(item["relation"]) for item in relations
            ).items())),
            "composite_context_count": sum(
                item.get("status") == "legacy" for item in concepts
            ),
            "approved_noun_alias_count": len(approved_alias_specs),
            "noun_alias_candidate_count": len(alias_specs),
            "noun_alias_candidate_occurrence_count": sum(
                int(item.get("corpus_count") or 0) for item in alias_specs
            ),
        },
    }


__all__ = [
    "ENTITY_ONTOLOGY_SCHEMA",
    "ENTITY_RELATION_TYPES",
    "NOUN_CONCEPT_TYPES",
    "APPROVED_ALIAS_RELATION_TYPES",
    "build_entity_projection",
]
