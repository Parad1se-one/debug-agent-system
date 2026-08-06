"""Deterministic terminology projection and resolver for KG_v2.

The terminology layer does not replace FaultFamily, FaultVariant or
DiagnosticAction.  It gives those canonical domain objects stable concept
identities, explicit expressions and auditable senses.  In particular, legacy
``keywords`` are projected as ``search_hint`` senses and are never silently
treated as synonyms.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.contracts import (
    APPROVED_FAMILY_LABELS,
    V2_PRIMARY_KEYS,
)
from debug_agent_system.knowledge_v2.entity_terminology import (
    ENTITY_RELATION_TYPES,
    NOUN_CONCEPT_TYPES,
    build_entity_projection,
)
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


TERMINOLOGY_VERSION = "kg_v2.debug_terminology.v4"
TERMINOLOGY_OBJECT_TYPES = {
    "DebugConcept",
    "TermExpression",
    "TermSense",
}
TERMINOLOGY_RELATIONS = {
    "primary_concept",
    "expression_has_sense",
    "sense_denotes",
    "broader_concept",
    "concept_context",
    "mentions_concept",
    *ENTITY_RELATION_TYPES,
}
SAFE_EQUIVALENCE_TYPES = {
    "canonical",
    "exact_synonym",
    "colloquial_alias",
    "abbreviation",
    "english_equivalent",
    "historical_name",
    "typo_variant",
}
DOMAIN_PROJECTIONS = {
    "FaultFamily": {
        "pk": "family_id",
        "name": "label",
        "definition": "summary",
        "concept_type": "symptom",
    },
    "FaultVariant": {
        "pk": "variant_id",
        "name": "label",
        "definition": "summary",
        "concept_type": "fault_variant",
    },
    "DiagnosticAction": {
        "pk": "action_id",
        "name": "label",
        "definition": "summary",
        "concept_type": "operation",
    },
}
MENTION_SOURCE_FIELDS = {
    "KnowledgeDocument": ("title",),
    "KnowledgeSection": ("heading", "summary"),
    "ProcedureStep": (
        "label",
        "instruction",
        "details",
        "expected_result",
        "prerequisites",
    ),
    "EvidenceItem": ("title", "summary"),
    "SourceCase": ("title", "summary"),
}
CURATED_RELATION_TYPES = SAFE_EQUIVALENCE_TYPES - {"canonical"}
CONTEXT_FIELDS = (
    "categories",
    "equipment_types",
    "subsystems",
    "phases",
    "signals",
)
CONTEXT_WEIGHTS = {
    "categories": 1.5,
    "equipment_types": 3.0,
    "subsystems": 3.0,
    "phases": 2.0,
    "signals": 2.0,
}


def normalize_term(value: Any) -> str:
    """Normalize an expression for identity and deterministic matching."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(re.findall(r"[a-z0-9+#._-]+|[\u4e00-\u9fff]+", text))


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part or "") for part in parts)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _clean_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif value in (None, ""):
        raw = []
    else:
        raw = [value]
    return list(
        dict.fromkeys(
            cleaned
            for item in raw
            if (cleaned := _clean_text(item, limit=200))
        )
    )


def _language(value: str) -> str:
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", value))
    has_latin = bool(re.search(r"[A-Za-z]", value))
    if has_zh and has_latin:
        return "mixed"
    if has_zh:
        return "zh-CN"
    if re.fullmatch(r"[A-Za-z0-9_+.#-]+", value.strip()):
        return "code"
    return "en"


def _source_text(item: dict[str, Any], fields: Iterable[str]) -> str:
    values: list[str] = []
    for field in fields:
        value = item.get(field)
        if isinstance(value, list):
            values.extend(str(part or "") for part in value)
        elif value not in (None, ""):
            values.append(str(value))
    return normalize_term(" ".join(values))


def _is_distinctive_mention(value: str) -> bool:
    normalized = normalize_term(value)
    if not normalized:
        return False
    if re.search(r"[\u4e00-\u9fff]", normalized):
        return len(normalized) >= 4
    return len(normalized) >= 3


def _local_schema_root(root: Path) -> Path | None:
    schema = root / "schema"
    return schema if (schema / "object-types.json").exists() else None


def _load_curated_entries(root: Path) -> list[dict[str, Any]]:
    path = root / "terminology" / "curated_terms.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("entries"), list
    ):
        raise ValueError("invalid_curated_terminology_file")
    return [
        dict(item)
        for item in payload["entries"]
        if isinstance(item, dict) and item.get("approved") is True
    ]


def _load_context_policies(root: Path) -> list[dict[str, Any]]:
    """Load declarative noun-context gates without embedding query rules."""

    path = root / "terminology" / "context_policies.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("policies"), list
    ):
        raise ValueError("invalid_terminology_context_policies_file")
    policies: list[dict[str, Any]] = []
    for raw in payload["policies"]:
        if not isinstance(raw, dict):
            continue
        canonical = _clean_text(raw.get("canonical_name"), limit=160)
        if not canonical:
            continue
        policies.append({
            "canonical_name": canonical,
            "require_allow_if_bare": bool(
                raw.get("require_allow_if_bare", False)
            ),
            "allow_if_any": _string_list(raw.get("allow_if_any")),
            "block_if_any": _string_list(raw.get("block_if_any")),
            "reason": _clean_text(
                raw.get("reason") or "context_policy_blocked",
                limit=300,
            ),
        })
    return policies


def build_terminology_layer(
    store_or_root: JsonKGV2Store | str | Path,
) -> dict[str, Any]:
    """Build a complete terminology projection without mutating the store."""

    store = (
        store_or_root
        if isinstance(store_or_root, JsonKGV2Store)
        else JsonKGV2Store(store_or_root)
    )
    objects = deepcopy(store.objects_by_type)
    relations = [
        dict(relation)
        for relation in store.relations
        if isinstance(relation, dict)
        and str(relation.get("relation") or "") not in TERMINOLOGY_RELATIONS
    ]

    family_by_id = {
        str(item.get("family_id") or ""): item
        for item in objects.get("FaultFamily") or []
        if isinstance(item, dict) and item.get("family_id")
    }
    concept_by_target: dict[tuple[str, str], dict[str, Any]] = {}
    context_concept_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    concepts: list[dict[str, Any]] = []

    # Families and variants are canonical diagnosis objects, so their concept
    # identity remains target-specific.
    for obj_type in ("FaultFamily", "FaultVariant"):
        projection = DOMAIN_PROJECTIONS[obj_type]
        pk = str(projection["pk"])
        for item in sorted(
            (
                value
                for value in objects.get(obj_type) or []
                if isinstance(value, dict) and value.get(pk)
            ),
            key=lambda value: str(value.get(pk) or ""),
        ):
            target_id = str(item[pk])
            name = _clean_text(item.get(projection["name"]), limit=160)
            if not name:
                continue
            family = (
                family_by_id.get(str(item.get("family_id") or ""))
                if obj_type != "FaultFamily"
                else item
            ) or {}
            family_label = str(family.get("label") or "")
            status = (
                "approved"
                if obj_type != "FaultFamily"
                or family_label in APPROVED_FAMILY_LABELS
                else "legacy"
            )
            definition = _clean_text(
                item.get(projection["definition"]) or name,
                limit=500,
            )
            concept = {
                "concept_id": _stable_id(
                    "concept",
                    obj_type,
                    target_id,
                ),
                "canonical_name": name,
                "concept_type": str(projection["concept_type"]),
                "definition": definition,
                "canonical_target_type": obj_type,
                "canonical_target_id": target_id,
                "status": status,
                "terminology_version": TERMINOLOGY_VERSION,
                "category": _clean_text(
                    family.get("category"),
                    limit=80,
                ),
                "subsystem": _clean_text(
                    family.get("subsystem"),
                    limit=120,
                ),
                "source_object_ids": [target_id],
            }
            concepts.append(concept)
            concept_by_target[(obj_type, target_id)] = concept
            relations.append({
                "from": target_id,
                "to": concept["concept_id"],
                "relation": "primary_concept",
                "basis": "deterministic_domain_projection",
            })

    # DiagnosticAction is an occurrence in a concrete trace.  Multiple traces
    # may instantiate the same semantic operation, so operation concepts are
    # keyed by normalized label plus action role rather than action_id.
    action_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for action in objects.get("DiagnosticAction") or []:
        if not isinstance(action, dict) or not action.get("action_id"):
            continue
        label = _clean_text(action.get("label"), limit=160)
        normalized = normalize_term(label)
        if not normalized:
            continue
        action_groups[
            (normalized, str(action.get("action_role") or ""))
        ].append(action)
    for (normalized, action_role), grouped_actions in sorted(
        action_groups.items()
    ):
        grouped_actions = sorted(
            grouped_actions,
            key=lambda item: str(item.get("action_id") or ""),
        )
        action_ids = [
            str(item["action_id"])
            for item in grouped_actions
        ]
        labels = sorted({
            _clean_text(item.get("label"), limit=160)
            for item in grouped_actions
            if _clean_text(item.get("label"), limit=160)
        })
        canonical_name = labels[0]
        families = [
            family_by_id.get(str(item.get("family_id") or "")) or {}
            for item in grouped_actions
        ]
        categories = sorted({
            value
            for family in families
            for value in _string_list(family.get("category"))
        })
        subsystems = sorted({
            value
            for family in families
            for value in _string_list(family.get("subsystem"))
        })
        summaries = sorted({
            _clean_text(item.get("summary"), limit=500)
            for item in grouped_actions
            if _clean_text(item.get("summary"), limit=500)
        })
        definition = (
            summaries[0]
            if len(grouped_actions) == 1 and summaries
            else _clean_text(
                f"可复用诊断操作：{canonical_name}。"
                f"由 {len(grouped_actions)} 个 KG_v2 Action 实例实现，"
                "具体条件、步骤和安全要求以各 Action 所属 Trace 为准。",
                limit=500,
            )
        )
        concept = {
            "concept_id": _stable_id(
                "concept",
                "operation",
                normalized,
                action_role,
            ),
            "canonical_name": canonical_name,
            "concept_type": "operation",
            "definition": definition,
            "status": "approved",
            "terminology_version": TERMINOLOGY_VERSION,
            "category": categories[0] if len(categories) == 1 else "",
            "subsystem": subsystems[0] if len(subsystems) == 1 else "",
            "source_object_ids": action_ids,
        }
        concepts.append(concept)
        for action_id in action_ids:
            concept_by_target[("DiagnosticAction", action_id)] = concept
            relations.append({
                "from": action_id,
                "to": concept["concept_id"],
                "relation": "primary_concept",
                "basis": "semantic_operation_projection",
            })

    variant_by_id = {
        str(item.get("variant_id") or ""): item
        for item in objects.get("FaultVariant") or []
        if isinstance(item, dict) and item.get("variant_id")
    }

    entity_projection = build_entity_projection(
        root=store.root,
        objects=objects,
        stable_id=_stable_id,
        clean_text=_clean_text,
        normalize_term=normalize_term,
        terminology_version=TERMINOLOGY_VERSION,
    )
    concepts.extend(entity_projection["concepts"])
    relations.extend(entity_projection["relations"])
    context_concept_by_key.update(entity_projection["context_anchors"])

    # Categories and phases are scalar facets. Equipment and subsystem fields
    # are noun-rich composite paths and are handled by entity_projection.
    context_specs = (
        ("category", "category", "FaultFamily", "category"),
        ("phase", "phase", "FaultVariant", "error_phase"),
    )
    for context_kind, concept_type, source_type, field in context_specs:
        pk = V2_PRIMARY_KEYS[source_type]
        sources_by_value: dict[str, list[str]] = defaultdict(list)
        for source in objects.get(source_type) or []:
            if not isinstance(source, dict):
                continue
            source_id = str(source.get(pk) or "")
            for value in _string_list(source.get(field)):
                if source_id:
                    sources_by_value[value].append(source_id)
        for value, source_ids in sorted(sources_by_value.items()):
            concept = {
                "concept_id": _stable_id(
                    "concept",
                    "native",
                    context_kind,
                    normalize_term(value),
                ),
                "canonical_name": _clean_text(value, limit=160),
                "concept_type": concept_type,
                "definition": _clean_text(
                    f"KG_v2 结构化字段中的{context_kind}概念：{value}",
                    limit=500,
                ),
                "status": "approved",
                "terminology_version": TERMINOLOGY_VERSION,
                "category": (
                    _clean_text(value, limit=80)
                    if context_kind == "category"
                    else ""
                ),
                "subsystem": (
                    _clean_text(value, limit=120)
                    if context_kind == "subsystem"
                    else ""
                ),
                "source_object_ids": sorted(set(source_ids)),
            }
            concepts.append(concept)
            context_concept_by_key[(context_kind, value)] = concept

    expressions_by_normalized: dict[str, dict[str, Any]] = {}
    senses_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_expression_and_sense(
        *,
        surface: str,
        concept: dict[str, Any],
        relation_type: str,
        source_type: str,
        source_id: str,
        source: dict[str, Any],
        sense_metadata: dict[str, Any] | None = None,
    ) -> None:
        cleaned = _clean_text(surface, limit=200)
        normalized = normalize_term(cleaned)
        if not normalized:
            return
        expression = expressions_by_normalized.get(normalized)
        expression_type = (
            "canonical"
            if relation_type == "canonical"
            else "keyword"
            if relation_type == "search_hint"
            else "abbreviation"
            if relation_type == "abbreviation"
            else "typo"
            if relation_type == "typo_variant"
            else "alias"
        )
        if expression is None:
            expression = {
                "term_id": _stable_id("term", normalized),
                "surface_form": cleaned,
                "normalized_form": normalized,
                "language": _language(cleaned),
                "expression_type": expression_type,
                "terminology_version": TERMINOLOGY_VERSION,
                "source_object_ids": [source_id],
            }
            expressions_by_normalized[normalized] = expression
        else:
            if source_id not in expression["source_object_ids"]:
                expression["source_object_ids"].append(source_id)
            if relation_type == "canonical":
                expression["expression_type"] = "canonical"
                expression["surface_form"] = cleaned
        term_id = str(expression["term_id"])
        concept_id = str(concept["concept_id"])
        key = (term_id, concept_id, relation_type)
        family = (
            family_by_id.get(str(source.get("family_id") or ""))
            if source_type != "FaultFamily"
            else source
        ) or {}
        variant = (
            source
            if source_type == "FaultVariant"
            else variant_by_id.get(str(source.get("variant_id") or "")) or {}
        )
        context = {
            "categories": _string_list(family.get("category")),
            "equipment_types": _string_list(
                variant.get("equipment_type")
                or source.get("equipment_type")
            ),
            "subsystems": _string_list(family.get("subsystem")),
            "phases": _string_list(
                variant.get("error_phase")
                or source.get("error_phase")
                or source.get("stage")
            ),
        }
        existing_sense = senses_by_key.get(key)
        if existing_sense is not None:
            for field, values in context.items():
                existing_sense[field] = sorted(set(
                    _string_list(existing_sense.get(field)) + values
                ))
            existing_sense["source_object_ids"] = sorted(set(
                _string_list(existing_sense.get("source_object_ids"))
                + [source_id]
            ))
            if sense_metadata:
                existing_sense.update(sense_metadata)
            return
        senses_by_key[key] = {
            "sense_id": _stable_id(
                "sense",
                term_id,
                concept_id,
                relation_type,
            ),
            "term_id": term_id,
            "concept_id": concept_id,
            "relation_type": relation_type,
            "approved": True,
            "source_object_type": source_type,
            "source_object_id": source_id,
            "source_object_ids": [source_id],
            **context,
            "required_signals": [],
            "excluded_signals": [],
            "terminology_version": TERMINOLOGY_VERSION,
            **(sense_metadata or {}),
        }

    seen_canonical_concepts: set[str] = set()
    for (obj_type, target_id), concept in sorted(
        concept_by_target.items(),
        key=lambda item: item[0],
    ):
        source = next(
            (
                item
                for item in objects.get(obj_type) or []
                if isinstance(item, dict)
                and str(item.get(V2_PRIMARY_KEYS[obj_type]) or "") == target_id
            ),
            {},
        )
        concept_id = str(concept["concept_id"])
        if concept_id not in seen_canonical_concepts:
            add_expression_and_sense(
                surface=str(concept["canonical_name"]),
                concept=concept,
                relation_type="canonical",
                source_type=obj_type,
                source_id=target_id,
                source=source,
            )
            seen_canonical_concepts.add(concept_id)
        elif obj_type == "DiagnosticAction":
            # Merge the context and provenance of every Action occurrence into
            # the shared canonical operation sense.
            add_expression_and_sense(
                surface=str(concept["canonical_name"]),
                concept=concept,
                relation_type="canonical",
                source_type=obj_type,
                source_id=target_id,
                source=source,
            )
        if obj_type in {"FaultFamily", "FaultVariant"}:
            for keyword in _string_list(source.get("keywords")):
                if normalize_term(keyword) == normalize_term(
                    concept["canonical_name"]
                ):
                    continue
                add_expression_and_sense(
                    surface=keyword,
                    concept=concept,
                    relation_type="search_hint",
                    source_type=obj_type,
                    source_id=target_id,
                    source=source,
                )

    for (context_kind, value), concept in sorted(
        context_concept_by_key.items(),
        key=lambda item: item[0],
    ):
        if concept.get("status") == "legacy":
            # Composite structured paths are graph-only compatibility
            # anchors. Their normalized surface can equal a qualified atomic
            # noun (for example 复判站/软件 -> 复判站软件), so exposing both as
            # canonical senses would create artificial ambiguity.
            continue
        source_type = (
            "FaultFamily"
            if context_kind in {"category", "subsystem"}
            else "FaultVariant"
        )
        source_id = str(concept["source_object_ids"][0])
        source = next(
            (
                item
                for item in objects.get(source_type) or []
                if isinstance(item, dict)
                and str(item.get(V2_PRIMARY_KEYS[source_type]) or "")
                == source_id
            ),
            {},
        )
        add_expression_and_sense(
            surface=value,
            concept=concept,
            relation_type="canonical",
            source_type=source_type,
            source_id=source_id,
            source=source,
        )

    for item in entity_projection["canonical_sources"]:
        add_expression_and_sense(
            surface=str(item["surface"]),
            concept=item["concept"],
            relation_type=str(item["relation_type"]),
            source_type=str(item["source_type"]),
            source_id=str(item["source_id"]),
            source=item["source"],
        )

    for item in entity_projection["alias_sources"]:
        relation_type = str(item.get("relation_type") or "search_hint")
        is_candidate = relation_type == "search_hint"
        metadata = (
            {
                "candidate_kind": str(item["candidate_kind"]),
                "candidate_relation_type": str(
                    item["candidate_relation_type"]
                ),
                "candidate_risk": str(item["risk"]),
                "corpus_count": int(item.get("corpus_count") or 0),
                "corpus_evidence_paths": item.get(
                    "corpus_evidence_paths"
                ) or [],
            }
            if is_candidate
            else {
                "review_authority": "human_approved_entity_ontology",
            }
        )
        add_expression_and_sense(
            surface=str(item["surface"]),
            concept=item["concept"],
            relation_type=relation_type,
            source_type=str(item["source_type"]),
            source_id=str(item["source_id"]),
            source=item["source"],
            sense_metadata=metadata,
        )

    concept_by_id = {
        str(item.get("concept_id") or ""): item
        for item in concepts
        if isinstance(item, dict) and item.get("concept_id")
    }
    for entry in _load_curated_entries(store.root):
        relation_type = str(entry.get("relation_type") or "")
        target_type = str(entry.get("canonical_target_type") or "")
        target_id = str(entry.get("canonical_target_id") or "")
        selected_concept_id = str(entry.get("concept_id") or "")
        surface = str(entry.get("surface_form") or "")
        if relation_type not in CURATED_RELATION_TYPES:
            raise ValueError(
                f"invalid_curated_relation_type:{relation_type}"
            )
        concept = (
            concept_by_id.get(selected_concept_id)
            if selected_concept_id
            else concept_by_target.get((target_type, target_id))
        )
        if concept is None:
            raise ValueError(
                "invalid_curated_target:"
                f"{selected_concept_id or target_type + ':' + target_id}"
            )
        if not normalize_term(surface):
            raise ValueError("invalid_curated_surface_form")
        source_refs: list[tuple[str, str, dict[str, Any]]] = []
        candidate_types = (
            [target_type]
            if target_type in DOMAIN_PROJECTIONS
            else list(DOMAIN_PROJECTIONS)
        )
        source_ids = (
            [target_id]
            if target_id
            else _string_list(concept.get("source_object_ids"))
        )
        for source_id in source_ids:
            for source_type in candidate_types:
                pk = V2_PRIMARY_KEYS[source_type]
                source = next(
                    (
                        item
                        for item in objects.get(source_type) or []
                        if isinstance(item, dict)
                        and str(item.get(pk) or "") == source_id
                    ),
                    None,
                )
                if source is not None:
                    source_refs.append(
                        (source_type, source_id, source)
                    )
                    break
        if not source_refs:
            source_refs = [(
                "DebugConcept",
                str(concept["concept_id"]),
                {},
            )]
        for source_type, source_id, source in source_refs:
            add_expression_and_sense(
                surface=surface,
                concept=concept,
                relation_type=relation_type,
                source_type=source_type,
                source_id=source_id,
                source=source,
            )

    expressions = sorted(
        expressions_by_normalized.values(),
        key=lambda item: str(item["term_id"]),
    )
    for expression in expressions:
        expression["source_object_ids"] = sorted(
            set(expression["source_object_ids"])
        )
    senses = sorted(
        senses_by_key.values(),
        key=lambda item: str(item["sense_id"]),
    )
    for sense in senses:
        relations.extend([
            {
                "from": sense["term_id"],
                "to": sense["sense_id"],
                "relation": "expression_has_sense",
            },
            {
                "from": sense["sense_id"],
                "to": sense["concept_id"],
                "relation": "sense_denotes",
            },
        ])

    for variant in objects.get("FaultVariant") or []:
        if not isinstance(variant, dict):
            continue
        variant_concept = concept_by_target.get((
            "FaultVariant",
            str(variant.get("variant_id") or ""),
        ))
        family_concept = concept_by_target.get((
            "FaultFamily",
            str(variant.get("family_id") or ""),
        ))
        if variant_concept and family_concept:
            relations.append({
                "from": variant_concept["concept_id"],
                "to": family_concept["concept_id"],
                "relation": "broader_concept",
                "basis": "fault_variant_family",
            })

    for (obj_type, target_id), concept in concept_by_target.items():
        source = next(
            (
                item
                for item in objects.get(obj_type) or []
                if isinstance(item, dict)
                and str(item.get(V2_PRIMARY_KEYS[obj_type]) or "")
                == target_id
            ),
            {},
        )
        family = (
            family_by_id.get(str(source.get("family_id") or ""))
            if obj_type != "FaultFamily"
            else source
        ) or {}
        contexts = [
            ("category", value)
            for value in _string_list(family.get("category"))
        ]
        contexts.extend(
            ("subsystem", value)
            for value in _string_list(family.get("subsystem"))
        )
        if obj_type == "FaultVariant":
            contexts.extend(
                ("equipment", value)
                for value in _string_list(source.get("equipment_type"))
            )
            contexts.extend(
                ("phase", value)
                for value in _string_list(source.get("error_phase"))
            )
        for context_kind, value in contexts:
            context_concept = context_concept_by_key.get(
                (context_kind, value)
            )
            if context_concept:
                relations.append({
                    "from": concept["concept_id"],
                    "to": context_concept["concept_id"],
                    "relation": "concept_context",
                    "context_kind": context_kind,
                })

    mention_concepts = [
        concept
        for concept in concepts
        if concept.get("canonical_target_type") in {
            "FaultFamily",
            "FaultVariant",
        }
        and _is_distinctive_mention(str(concept["canonical_name"]))
    ]
    for source_type, fields in MENTION_SOURCE_FIELDS.items():
        pk = V2_PRIMARY_KEYS[source_type]
        for source in objects.get(source_type) or []:
            if not isinstance(source, dict):
                continue
            source_id = str(source.get(pk) or "")
            haystack = _source_text(source, fields)
            if not source_id or not haystack:
                continue
            for concept in mention_concepts:
                needle = normalize_term(concept["canonical_name"])
                if needle and needle in haystack:
                    relations.append({
                        "from": source_id,
                        "to": concept["concept_id"],
                        "relation": "mentions_concept",
                        "basis": "exact_canonical_name",
                    })

    objects["DebugConcept"] = sorted(
        concepts,
        key=lambda item: str(item["concept_id"]),
    )
    objects["TermExpression"] = expressions
    objects["TermSense"] = senses
    relations = _deduplicate_relations(relations)

    schema_root = _local_schema_root(store.root)
    issues = validate_graph(
        objects,
        relations,
        schema_root=schema_root,
    )
    if issues:
        raise ValueError(
            "terminology_graph_invalid:" + ",".join(issues[:50])
        )

    report = terminology_quality_report(
        objects,
        relations,
    )
    return {
        "objects_by_type": objects,
        "relations": relations,
        "report": report,
    }


def _deduplicate_relations(
    relations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for relation in relations:
        key = (
            str(relation.get("from") or ""),
            str(relation.get("to") or ""),
            str(relation.get("relation") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        output.append(dict(relation))
    return output


def terminology_quality_report(
    objects_by_type: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    concepts = objects_by_type.get("DebugConcept") or []
    expressions = objects_by_type.get("TermExpression") or []
    senses = objects_by_type.get("TermSense") or []
    concepts_by_term: dict[str, set[str]] = defaultdict(set)
    relation_types_by_term: dict[str, set[str]] = defaultdict(set)
    for sense in senses:
        if not isinstance(sense, dict):
            continue
        term_id = str(sense.get("term_id") or "")
        concepts_by_term[term_id].add(
            str(sense.get("concept_id") or "")
        )
        relation_types_by_term[term_id].add(
            str(sense.get("relation_type") or "")
        )
    expression_by_id = {
        str(item.get("term_id") or ""): item
        for item in expressions
        if isinstance(item, dict)
    }
    ambiguous = [
        {
            "term_id": term_id,
            "surface_form": str(
                expression_by_id.get(term_id, {}).get(
                    "surface_form"
                ) or ""
            ),
            "concept_count": len(concept_ids),
            "relation_types": sorted(relation_types_by_term[term_id]),
        }
        for term_id, concept_ids in concepts_by_term.items()
        if len(concept_ids) > 1
    ]
    legacy = [
        str(item.get("concept_id") or "")
        for item in concepts
        if isinstance(item, dict) and item.get("status") == "legacy"
    ]
    terminology_relations = [
        relation
        for relation in relations
        if str(relation.get("relation") or "")
        in TERMINOLOGY_RELATIONS
    ]
    revision_payload = {
        "version": TERMINOLOGY_VERSION,
        "concepts": concepts,
        "expressions": expressions,
        "senses": senses,
        "relations": terminology_relations,
    }
    revision = hashlib.sha256(
        json.dumps(
            revision_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    relation_counts: dict[str, int] = defaultdict(int)
    for relation in terminology_relations:
        relation_counts[str(relation.get("relation") or "")] += 1
    sense_counts: dict[str, int] = defaultdict(int)
    for sense in senses:
        sense_counts[str(sense.get("relation_type") or "")] += 1
    return {
        "schema_version": "kg_v2.terminology_report.v2",
        "terminology_version": TERMINOLOGY_VERSION,
        "revision": revision,
        "concept_count": len(concepts),
        "expression_count": len(expressions),
        "sense_count": len(senses),
        "concept_counts": dict(sorted(
            (
                concept_type,
                sum(
                    item.get("concept_type") == concept_type
                    for item in concepts
                ),
            )
            for concept_type in sorted({
                str(item.get("concept_type") or "")
                for item in concepts
                if isinstance(item, dict)
            })
        )),
        "sense_counts": dict(sorted(sense_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "ambiguous_expression_count": len(ambiguous),
        "ambiguous_expressions": sorted(
            ambiguous,
            key=lambda item: (
                -int(item["concept_count"]),
                str(item["surface_form"]),
            ),
        )[:100],
        "legacy_concept_count": len(legacy),
        "legacy_concept_ids": sorted(legacy),
        "operation_instance_count": sum(
            len(_string_list(item.get("source_object_ids")))
            for item in concepts
            if isinstance(item, dict)
            and item.get("concept_type") == "operation"
        ),
        "operation_concept_count": sum(
            item.get("concept_type") == "operation"
            for item in concepts
            if isinstance(item, dict)
        ),
        "shared_operation_concept_count": sum(
            item.get("concept_type") == "operation"
            and len(_string_list(item.get("source_object_ids"))) > 1
            for item in concepts
            if isinstance(item, dict)
        ),
        "merged_operation_instance_count": sum(
            max(
                0,
                len(_string_list(item.get("source_object_ids"))) - 1,
            )
            for item in concepts
            if isinstance(item, dict)
            and item.get("concept_type") == "operation"
        ),
        "noun_concept_count": sum(
            item.get("concept_type") in NOUN_CONCEPT_TYPES
            for item in concepts
            if isinstance(item, dict)
        ),
        "noun_concept_counts": dict(sorted(
            (
                concept_type,
                sum(
                    item.get("concept_type") == concept_type
                    for item in concepts
                    if isinstance(item, dict)
                ),
            )
            for concept_type in sorted(NOUN_CONCEPT_TYPES)
        )),
        "entity_relation_count": sum(
            str(relation.get("relation") or "") in ENTITY_RELATION_TYPES
            for relation in terminology_relations
        ),
        "entity_relation_counts": dict(sorted(
            (
                relation_type,
                sum(
                    relation.get("relation") == relation_type
                    for relation in terminology_relations
                ),
            )
            for relation_type in sorted(ENTITY_RELATION_TYPES)
        )),
        "noun_alias_candidate_count": sum(
            sense.get("relation_type") == "search_hint"
            and sense.get("source_object_type") == "RawCorpusCandidate"
            for sense in senses
            if isinstance(sense, dict)
        ),
        "approved_noun_alias_count": sum(
            sense.get("source_object_type") == "EntityOntologyAlias"
            and sense.get("relation_type") in SAFE_EQUIVALENCE_TYPES
            for sense in senses
            if isinstance(sense, dict)
        ),
    }


def write_terminology_layer(
    root: str | Path,
) -> dict[str, Any]:
    """Rebuild, validate and persist the terminology projection."""

    store = JsonKGV2Store(root)
    built = build_terminology_layer(store)
    result = store.replace_graph(
        built["objects_by_type"],
        built["relations"],
        validate=True,
    )
    if result.get("status") != "replaced":
        raise ValueError(
            "terminology_write_failed:"
            + json.dumps(result, ensure_ascii=False)
        )
    manifest_root = Path(root) / "terminology"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(built["report"])
    manifest["object_files"] = {
        "DebugConcept": "objects/debug_concepts.json",
        "TermExpression": "objects/term_expressions.json",
        "TermSense": "objects/term_senses.json",
    }
    manifest["curated_terms_file"] = "terminology/curated_terms.json"
    manifest["relation_file"] = "relations/edges.json"
    (manifest_root / "terminology_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


class TerminologyResolver:
    """Read-only resolver over the generated terminology objects."""

    def __init__(
        self,
        *,
        concepts: list[dict[str, Any]],
        expressions: list[dict[str, Any]],
        senses: list[dict[str, Any]],
        relations: list[dict[str, Any]] | None = None,
        context_policies: list[dict[str, Any]] | None = None,
    ) -> None:
        self.concepts = {
            str(item.get("concept_id") or ""): dict(item)
            for item in concepts
            if isinstance(item, dict) and item.get("concept_id")
        }
        self.expressions = {
            str(item.get("term_id") or ""): dict(item)
            for item in expressions
            if isinstance(item, dict) and item.get("term_id")
        }
        self.senses_by_term: dict[str, list[dict[str, Any]]] = (
            defaultdict(list)
        )
        self.senses_by_concept: dict[str, list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for sense in senses:
            if not isinstance(sense, dict) or not sense.get("approved"):
                continue
            copied = dict(sense)
            self.senses_by_term[str(sense.get("term_id") or "")].append(
                copied
            )
            self.senses_by_concept[
                str(sense.get("concept_id") or "")
            ].append(copied)
        self.entity_relations = [
            dict(relation)
            for relation in relations or []
            if str(relation.get("relation") or "")
            in ENTITY_RELATION_TYPES - {"context_member"}
        ]
        self.context_policies = {
            normalize_term(str(item.get("canonical_name") or "")): dict(item)
            for item in context_policies or []
            if normalize_term(str(item.get("canonical_name") or ""))
        }

    @classmethod
    def from_root(cls, root: str | Path) -> "TerminologyResolver":
        store = JsonKGV2Store(root)
        return cls(
            concepts=store.objects_by_type.get("DebugConcept") or [],
            expressions=store.objects_by_type.get("TermExpression") or [],
            senses=store.objects_by_type.get("TermSense") or [],
            relations=store.relations,
            context_policies=_load_context_policies(store.root),
        )

    def resolve(
        self,
        text: str,
        *,
        limit: int = 30,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_query = normalize_term(text)
        detected_context = self._detect_context(
            normalized_query,
            context or {},
        )
        matched: list[dict[str, Any]] = []
        supporting: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        ambiguous_supporting: list[dict[str, Any]] = []
        safe_expansions: list[str] = []
        retrieval_expansions: list[dict[str, Any]] = []
        blocked_expansions: list[dict[str, Any]] = []

        for term_id, expression in self.expressions.items():
            normalized = str(expression.get("normalized_form") or "")
            if not _expression_matches(
                normalized, normalized_query, original_query=text
            ):
                continue
            senses = self.senses_by_term.get(term_id) or []
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
            direct_concepts = sorted({
                str(sense.get("concept_id") or "")
                for sense in direct
            })
            if len(direct_concepts) > 1:
                # When one sense is a trivial self-reference (canonical name
                # equals the surface form) and another is a non-canonical
                # approved equivalence, prefer the non-trivial mapping.
                trivial_self = None
                non_trivial: list[str] = []
                for concept_id in direct_concepts:
                    concept = self.concepts.get(concept_id) or {}
                    if (
                        str(concept.get("canonical_name") or "") == str(
                            expression.get("surface_form") or ""
                        )
                        and any(
                            s.get("relation_type") == "canonical"
                            for s in senses
                            if s.get("concept_id") == concept_id
                        )
                    ):
                        trivial_self = concept_id
                    else:
                        non_trivial.append(concept_id)
                if trivial_self and len(non_trivial) == 1:
                    # Prefer the non-trivial mapping (e.g. abbreviation)
                    concept_id = non_trivial[0]
                    matched.append({
                        "term_id": term_id,
                        "surface_form": expression.get("surface_form"),
                        "normalized_form": normalized,
                        "relation_types": sorted({
                            str(sense.get("relation_type") or "")
                            for sense in direct
                            if str(sense.get("concept_id") or "")
                            == concept_id
                        }),
                        "concept": self._concept_summary(concept_id),
                        "resolution_method": "prefer_non_trivial_mapping",
                    })
                    safe_expansions.extend(
                        self._safe_surface_forms(
                            concept_id,
                            exclude=normalized,
                        )
                    )
                    continue

                ranked = self._rank_candidates(
                    direct_concepts,
                    direct,
                    detected_context,
                )
                best = ranked[0]
                second_score = (
                    float(ranked[1]["score"])
                    if len(ranked) > 1
                    else 0.0
                )
                margin = float(best["score"]) - second_score
                if (
                    int(best["context_match_count"]) > 0
                    and float(best["score"]) >= 2.0
                    and margin >= 2.0
                ):
                    concept_id = str(best["concept"]["concept_id"])
                    matched.append({
                        "term_id": term_id,
                        "surface_form": expression.get("surface_form"),
                        "normalized_form": normalized,
                        "relation_types": sorted({
                            str(sense.get("relation_type") or "")
                            for sense in direct
                            if str(sense.get("concept_id") or "")
                            == concept_id
                        }),
                        "concept": self._concept_summary(concept_id),
                        "resolution_method": "context_disambiguation",
                        "context_score": best["score"],
                        "top_margin": round(margin, 4),
                        "matched_context": best["matched_context"],
                    })
                    safe_expansions.extend(
                        self._safe_surface_forms(
                            concept_id,
                            exclude=normalized,
                        )
                    )
                else:
                    ambiguous.append({
                        "term_id": term_id,
                        "surface_form": expression.get("surface_form"),
                        "candidate_concepts": ranked,
                        "top_margin": round(margin, 4),
                        "reason": (
                            "context_margin_insufficient"
                            if any(
                                item["context_match_count"]
                                for item in ranked
                            )
                            else "context_required"
                        ),
                        "required_context": self._required_context(
                            ranked,
                        ),
                    })
            elif direct_concepts:
                concept_id = direct_concepts[0]
                matched.append({
                    "term_id": term_id,
                    "surface_form": expression.get("surface_form"),
                    "normalized_form": normalized,
                    "relation_types": sorted({
                        str(sense.get("relation_type") or "")
                        for sense in direct
                    }),
                    "concept": self._concept_summary(concept_id),
                    "resolution_method": "unique_approved_sense",
                })
                safe_expansions.extend(
                    self._safe_surface_forms(concept_id, exclude=normalized)
                )
            hint_concepts = sorted({
                str(sense.get("concept_id") or "")
                for sense in hints
            })
            ranked_hints = self._rank_candidates(
                hint_concepts,
                hints,
                detected_context,
            )
            rank_by_concept = {
                str(item["concept"]["concept_id"]): item
                for item in ranked_hints
            }
            for sense in hints:
                concept_id = str(sense.get("concept_id") or "")
                ranked_hint = rank_by_concept.get(concept_id) or {}
                supporting.append({
                    "term_id": term_id,
                    "surface_form": expression.get("surface_form"),
                    "relation_type": "search_hint",
                    "concept": self._concept_summary(concept_id),
                    "context_score": ranked_hint.get("score", 0.0),
                    "matched_context": ranked_hint.get(
                        "matched_context",
                        [],
                    ),
                    "can_lock_variant": False,
                })
                canonical_name = str(
                    self.concepts.get(concept_id, {}).get(
                        "canonical_name"
                    ) or ""
                )
                if canonical_name:
                    retrieval_expansions.append({
                        "text": canonical_name,
                        "authority": "search_hint",
                        "source_surface_form": expression.get(
                            "surface_form"
                        ),
                        "concept_id": concept_id,
                        "can_lock_variant": False,
                    })
            if len(hint_concepts) > 1:
                ambiguous_supporting.append({
                    "term_id": term_id,
                    "surface_form": expression.get("surface_form"),
                    "candidate_concepts": ranked_hints,
                    "reason": "search_hint_has_multiple_senses",
                    "can_lock_variant": False,
                })

        (
            matched,
            supporting,
            safe_expansions,
            retrieval_expansions,
            blocked_expansions,
        ) = self._apply_context_policies(
            normalized_query=normalized_query,
            matched=matched,
            supporting=supporting,
            safe_expansions=safe_expansions,
            retrieval_expansions=retrieval_expansions,
            ambiguous=ambiguous,
            blocked_expansions=blocked_expansions,
        )

        matched.sort(
            key=lambda item: (
                -len(str(item.get("normalized_form") or "")),
                str(item.get("term_id") or ""),
            )
        )
        supporting.sort(
            key=lambda item: (
                -len(normalize_term(item.get("surface_form"))),
                -float(item.get("context_score") or 0.0),
                str(item.get("term_id") or ""),
                str(item.get("concept", {}).get("concept_id") or ""),
            )
        )
        unique_safe_expansions = list(dict.fromkeys(
            item
            for item in safe_expansions
            if normalize_term(item) != normalized_query
        ))[:limit]
        retrieval_expansions = [
            {
                "text": item,
                "authority": "approved_equivalence",
                "can_lock_variant": False,
            }
            for item in unique_safe_expansions
        ] + retrieval_expansions

        mentioned_concept_ids = {
            str(item.get("concept", {}).get("concept_id") or "")
            for item in matched + supporting
            if str(item.get("concept", {}).get("concept_id") or "")
        }
        retrieval_expansions.extend(
            self._relational_retrieval_expansions(
                mentioned_concept_ids,
            )
        )
        unique_retrieval: list[dict[str, Any]] = []
        seen_retrieval: set[tuple[str, str]] = set()
        for item in retrieval_expansions:
            identity = (
                normalize_term(item.get("text")),
                str(item.get("authority") or ""),
            )
            if not identity[0] or identity in seen_retrieval:
                continue
            seen_retrieval.add(identity)
            unique_retrieval.append(item)
        entity_relations = self._related_entities(
            mentioned_concept_ids,
            limit=limit,
        )

        # ── Structured query_expansions consumed directly by the search
        # contract and audit layer.  Every sub-field has a deterministic
        # contract that the model must observe.
        matched_terms: list[dict[str, Any]] = []
        canonical_entities: list[dict[str, Any]] = []
        for mention in matched[:limit]:
            concept = mention.get("concept") or {}
            concept_id = str(concept.get("concept_id") or "")
            canonical = str(concept.get("canonical_name") or "")
            surface = str(mention.get("surface_form") or "")
            relation_types = list(mention.get("relation_types") or [])
            matched_terms.append({
                "surface_form": surface,
                "canonical_name": canonical,
                "concept_id": concept_id,
                "concept_type": str(concept.get("concept_type") or ""),
                "relation_types": relation_types,
                "resolution_method": str(
                    mention.get("resolution_method") or ""
                ),
                "can_lock_variant": (
                    "canonical" in relation_types
                    and not any(
                        rt in ("search_hint", "colloquial_alias",
                               "abbreviation", "english_equivalent")
                        for rt in relation_types
                    )
                ),
            })
            if canonical and canonical not in {
                item["canonical_name"] for item in canonical_entities
            }:
                canonical_entities.append({
                    "canonical_name": canonical,
                    "concept_id": concept_id,
                    "concept_type": str(concept.get("concept_type") or ""),
                    "category": str(concept.get("category") or ""),
                    "subsystem": str(concept.get("subsystem") or ""),
                    "aliases": sorted({
                        surface
                        for m in matched[:limit]
                        if str((m.get("concept") or {}).get(
                            "canonical_name") or "") == canonical
                        for surface in [str(m.get("surface_form") or "")]
                        if surface and surface != canonical
                    }),
                })

        # Build required_pairs from two sources:
        # 1. matched_terms where surface ≠ canonical (direct alias match)
        # 2. retrieval_expansions where approved_equivalence source appears in query
        required_pairs: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for term in matched_terms:
            source = term["surface_form"]
            canonical = term["canonical_name"]
            if source == canonical:
                continue
            key = (normalize_term(source), normalize_term(canonical))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            required_pairs.append({
                "source": source,
                "canonical": canonical,
                "obligation": "must_search_both",
            })
        # Also pick up approved equivalence expansions present in the query
        query_key = normalize_term(text)
        for exp in unique_retrieval[:limit]:
            if str(exp.get("authority") or "") != "approved_equivalence":
                continue
            src = str(exp.get("source_surface_form") or exp.get("text") or "")
            if not src or normalize_term(src) not in query_key:
                continue
            canonical = str(exp.get("text") or "")
            if normalize_term(src) == normalize_term(canonical):
                continue
            key = (normalize_term(src), normalize_term(canonical))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            required_pairs.append({
                "source": src,
                "canonical": canonical,
                "obligation": "must_search_both",
            })

        query_expansions: dict[str, Any] = {
            "schema_version": "kg_v2.query_expansions.v1",
            "matched_terms": matched_terms,
            "canonical_entities": canonical_entities,
            "blocked_expansions": [
                {
                    "surface_form": str(item.get("surface_form") or ""),
                    "canonical_name": str(item.get("canonical_name") or ""),
                    "reason": str(item.get("reason") or ""),
                }
                for item in blocked_expansions
            ],
            "ambiguous_surfaces": [
                {
                    "surface_form": str(item.get("surface_form") or ""),
                    "reason": str(item.get("reason") or ""),
                    "required_context": list(
                        item.get("required_context") or []
                    ),
                }
                for item in ambiguous[:limit]
            ],
            "search_obligations": {
                "required_pairs": required_pairs,
                "optional_expansions": [
                    {"text": item["text"], "authority": item["authority"]}
                    for item in unique_retrieval[:limit]
                    if item.get("authority") == "search_hint"
                ],
                "governance_only": (
                    "query_expansions 不包含事实证据；"
                    "matched_terms 只能用于扩展搜索词，不能用于锁定 Variant"
                ),
            },
        }

        return {
            "schema_version": "kg_v2.term_resolution.v4",
            "terminology_version": TERMINOLOGY_VERSION,
            "query": text,
            "normalized_query": normalized_query,
            "detected_context": detected_context,
            "resolved_mentions": matched[:limit],
            "ambiguous_mentions": ambiguous[:limit],
            "ambiguous_supporting_mentions": ambiguous_supporting[:limit],
            "supporting_concepts": supporting[:limit],
            "safe_expansions": unique_safe_expansions,
            "retrieval_expansions": unique_retrieval[:limit],
            "entity_relations": entity_relations,
            "query_expansions": query_expansions,
            "safety": {
                "search_hint_can_expand_retrieval": True,
                "search_hint_can_lock_variant": False,
                "ambiguous_term_requires_context": True,
                "context_disambiguation_min_score": 2.0,
                "context_disambiguation_min_margin": 2.0,
                "blocked_expansions": blocked_expansions,
            },
        }

    def _apply_context_policies(
        self,
        *,
        normalized_query: str,
        matched: list[dict[str, Any]],
        supporting: list[dict[str, Any]],
        safe_expansions: list[str],
        retrieval_expansions: list[dict[str, Any]],
        ambiguous: list[dict[str, Any]],
        blocked_expansions: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[str],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Apply reviewed noun context gates to direct alias matches.

        A blocked alias is moved to ``ambiguous_mentions`` and can still be
        searched as a candidate.  It is never emitted as a resolved concept
        and therefore cannot lock a Variant or create an unsafe expansion.
        The gates are data-driven and concept-scoped; no query-specific rules
        are embedded in this resolver.
        """

        retained: list[dict[str, Any]] = []
        blocked_concepts: set[str] = set()
        for mention in matched:
            concept = mention.get("concept") or {}
            concept_id = str(concept.get("concept_id") or "")
            canonical = str(concept.get("canonical_name") or "")
            policy = self.context_policies.get(normalize_term(canonical))
            if not policy:
                retained.append(mention)
                continue
            allow_terms = [
                normalize_term(value)
                for value in policy.get("allow_if_any") or []
                if normalize_term(value)
            ]
            block_terms = [
                normalize_term(value)
                for value in policy.get("block_if_any") or []
                if normalize_term(value)
            ]
            matched_allow = [
                value for value in allow_terms if value in normalized_query
            ]
            matched_block = [
                value for value in block_terms if value in normalized_query
            ]
            blocked = bool(matched_block) or bool(
                policy.get("require_allow_if_bare") and not matched_allow
            )
            if not blocked:
                retained.append(mention)
                continue
            blocked_concepts.add(concept_id)
            surface = str(mention.get("surface_form") or "")
            reason = str(policy.get("reason") or "context_policy_blocked")
            candidate = {
                "concept": self._concept_summary(concept_id),
                "score": 0.0,
                "context_match_count": 0,
                "matched_context": [],
                "conflicting_context": [{
                    "field": "context_policy",
                    "values": matched_block,
                    "score": -4.0,
                }],
                "candidate_context": {},
            }
            ambiguous.append({
                "term_id": mention.get("term_id"),
                "surface_form": surface,
                "candidate_concepts": [candidate],
                "top_margin": 0.0,
                "reason": "context_policy_blocked",
                "required_context": list(policy.get("allow_if_any") or []),
            })
            blocked_expansions.append({
                "surface_form": surface,
                "canonical_name": canonical,
                "concept_id": concept_id,
                "matched_block_context": matched_block,
                "required_context": list(policy.get("allow_if_any") or []),
                "reason": reason,
                "can_lock_variant": False,
            })

        if blocked_concepts:
            supporting = [
                item
                for item in supporting
                if str((item.get("concept") or {}).get("concept_id") or "")
                not in blocked_concepts
            ]
            blocked_surfaces = {
                normalize_term(str(item.get("canonical_name") or ""))
                for item in blocked_expansions
            }
            blocked_aliases = {
                normalize_term(str(item.get("surface_form") or ""))
                for item in blocked_expansions
            }
            for concept_id in blocked_concepts:
                blocked_aliases.update(
                    normalize_term(value)
                    for value in self._safe_surface_forms(
                        concept_id,
                        exclude="",
                    )
                    if normalize_term(value)
                )
            safe_expansions = [
                item
                for item in safe_expansions
                if normalize_term(item) not in blocked_surfaces
                and normalize_term(item) not in blocked_aliases
            ]
            retrieval_expansions = [
                item
                for item in retrieval_expansions
                if str(item.get("concept_id") or "") not in blocked_concepts
                and normalize_term(item.get("text"))
                not in blocked_surfaces
            ]
        return (
            retained,
            supporting,
            safe_expansions,
            retrieval_expansions,
            blocked_expansions,
        )

    def _detect_context(
        self,
        normalized_query: str,
        supplied: dict[str, Any],
    ) -> dict[str, list[str]]:
        detected: dict[str, list[str]] = {
            field: [] for field in CONTEXT_FIELDS
        }
        concept_field = {
            "category": "categories",
            "equipment": "equipment_types",
            "product_model": "equipment_types",
            "station": "equipment_types",
            "workstation": "equipment_types",
            "subsystem": "subsystems",
            "component": "subsystems",
            "software": "subsystems",
            "interface": "subsystems",
            "connection": "subsystems",
            "protocol": "subsystems",
            "workpiece": "subsystems",
            "inspection_object": "subsystems",
            "package_type": "subsystems",
            "external_system": "subsystems",
            "data_artifact": "subsystems",
            "identifier": "subsystems",
            "material": "subsystems",
            "phase": "phases",
        }
        for concept in self.concepts.values():
            field = concept_field.get(
                str(concept.get("concept_type") or "")
            )
            if not field:
                continue
            name = str(concept.get("canonical_name") or "")
            normalized = normalize_term(name)
            if _expression_matches(normalized, normalized_query):
                detected[field].append(name)
        aliases = {
            "category": "categories",
            "equipment": "equipment_types",
            "equipment_type": "equipment_types",
            "subsystem": "subsystems",
            "phase": "phases",
            "signal": "signals",
        }
        for raw_field, values in supplied.items():
            field = aliases.get(str(raw_field), str(raw_field))
            if field not in detected:
                continue
            detected[field].extend(_string_list(values))
        return {
            field: list(dict.fromkeys(values))
            for field, values in detected.items()
            if values
        }

    def _rank_candidates(
        self,
        concept_ids: list[str],
        senses: list[dict[str, Any]],
        detected_context: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        senses_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(
            list
        )
        for sense in senses:
            senses_by_concept[
                str(sense.get("concept_id") or "")
            ].append(sense)
        ranked: list[dict[str, Any]] = []
        for concept_id in concept_ids:
            concept = self.concepts.get(concept_id) or {}
            candidate_context: dict[str, list[str]] = {
                field: [] for field in CONTEXT_FIELDS
            }
            candidate_context["categories"].extend(
                _string_list(concept.get("category"))
            )
            candidate_context["subsystems"].extend(
                _string_list(concept.get("subsystem"))
            )
            for sense in senses_by_concept.get(concept_id) or []:
                for field in CONTEXT_FIELDS[:-1]:
                    candidate_context[field].extend(
                        _string_list(sense.get(field))
                    )
                candidate_context["signals"].extend(
                    _string_list(sense.get("required_signals"))
                )
            candidate_context = {
                field: list(dict.fromkeys(values))
                for field, values in candidate_context.items()
                if values
            }
            score = 0.0
            matched_context: list[dict[str, Any]] = []
            conflicting_context: list[dict[str, Any]] = []
            for field, query_values in detected_context.items():
                query_normalized = {
                    normalize_term(value)
                    for value in query_values
                    if normalize_term(value)
                }
                candidate_values = candidate_context.get(field) or []
                candidate_normalized = {
                    normalize_term(value)
                    for value in candidate_values
                    if normalize_term(value)
                }
                overlap = query_normalized & candidate_normalized
                if overlap:
                    contribution = CONTEXT_WEIGHTS[field] * len(overlap)
                    score += contribution
                    matched_context.append({
                        "field": field,
                        "values": sorted(overlap),
                        "score": contribution,
                    })
                elif query_normalized and candidate_normalized:
                    score -= 1.0
                    conflicting_context.append({
                        "field": field,
                        "query_values": sorted(query_normalized),
                        "candidate_values": sorted(candidate_normalized),
                        "score": -1.0,
                    })
            excluded = {
                normalize_term(value)
                for sense in senses_by_concept.get(concept_id) or []
                for value in _string_list(sense.get("excluded_signals"))
                if normalize_term(value)
            }
            query_signals = {
                normalize_term(value)
                for value in detected_context.get("signals") or []
                if normalize_term(value)
            }
            excluded_overlap = query_signals & excluded
            if excluded_overlap:
                score -= 4.0 * len(excluded_overlap)
                conflicting_context.append({
                    "field": "excluded_signals",
                    "values": sorted(excluded_overlap),
                    "score": -4.0 * len(excluded_overlap),
                })
            ranked.append({
                "concept": self._concept_summary(concept_id),
                "score": round(score, 4),
                "context_match_count": len(matched_context),
                "matched_context": matched_context,
                "conflicting_context": conflicting_context,
                "candidate_context": candidate_context,
            })
        return sorted(
            ranked,
            key=lambda item: (
                -float(item["score"]),
                str(item["concept"].get("concept_id") or ""),
            ),
        )

    @staticmethod
    def _required_context(
        ranked: list[dict[str, Any]],
    ) -> list[str]:
        differing: list[str] = []
        for field in CONTEXT_FIELDS[:-1]:
            values = {
                tuple(
                    normalize_term(value)
                    for value in item.get(
                        "candidate_context",
                        {},
                    ).get(field, [])
                )
                for item in ranked
            }
            if len(values) > 1:
                differing.append(field)
        return differing

    def _concept_summary(self, concept_id: str) -> dict[str, Any]:
        concept = self.concepts.get(concept_id) or {}
        return {
            key: concept.get(key)
            for key in (
                "concept_id",
                "canonical_name",
                "concept_type",
                "canonical_target_type",
                "canonical_target_id",
                "status",
                "category",
                "subsystem",
                "source_object_ids",
            )
        }

    def _safe_surface_forms(
        self,
        concept_id: str,
        *,
        exclude: str,
    ) -> list[str]:
        output: list[str] = []
        for sense in self.senses_by_concept.get(concept_id) or []:
            if sense.get("relation_type") not in SAFE_EQUIVALENCE_TYPES:
                continue
            expression = self.expressions.get(
                str(sense.get("term_id") or "")
            ) or {}
            if str(expression.get("normalized_form") or "") == exclude:
                continue
            surface = str(expression.get("surface_form") or "")
            if surface:
                output.append(surface)
        return output

    def _related_entities(
        self,
        concept_ids: set[str],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        expandable = {
            "model_of",
            "is_a",
            "part_of",
            "runs_on",
            "driver_of",
            "firmware_of",
            "sdk_for",
            "artifact_of",
            "process_instance_of",
            "configuration_of",
            "database_of",
            "produced_by",
            "collected_from",
        }
        for relation in self.entity_relations:
            source_id = str(relation.get("from") or "")
            target_id = str(relation.get("to") or "")
            if source_id not in concept_ids and target_id not in concept_ids:
                continue
            relation_type = str(relation.get("relation") or "")
            output.append({
                "relation": relation_type,
                "source": self._concept_summary(source_id),
                "target": self._concept_summary(target_id),
                "basis": str(relation.get("basis") or ""),
                "scope": str(relation.get("scope") or "fact"),
                "direction": str(relation.get("direction") or ""),
                "evidence_required": bool(
                    relation.get("evidence_required", False)
                ),
                "can_expand_retrieval": (
                    source_id in concept_ids
                    and relation_type in expandable
                ),
                "can_lock_variant": False,
            })
        return sorted(
            output,
            key=lambda item: (
                str(item["relation"]),
                str(item["source"].get("canonical_name") or ""),
                str(item["target"].get("canonical_name") or ""),
            ),
        )[:limit]

    def _relational_retrieval_expansions(
        self,
        concept_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Expand a mentioned noun to its reviewed semantic parent.

        Reverse expansion from a host such as ``工控机`` to every child
        subsystem is intentionally excluded because it would broaden most
        equipment queries without evidence that the child is relevant.
        """

        eligible = {
            "model_of",
            "is_a",
            "part_of",
            "runs_on",
            "driver_of",
            "firmware_of",
            "sdk_for",
            "artifact_of",
            "process_instance_of",
            "configuration_of",
            "database_of",
            "produced_by",
            "collected_from",
        }
        output: list[dict[str, Any]] = []
        for relation in self.entity_relations:
            relation_type = str(relation.get("relation") or "")
            source_id = str(relation.get("from") or "")
            target_id = str(relation.get("to") or "")
            if source_id not in concept_ids or relation_type not in eligible:
                continue
            target = self.concepts.get(target_id) or {}
            canonical_name = str(target.get("canonical_name") or "")
            if not canonical_name:
                continue
            output.append({
                "text": canonical_name,
                "authority": "entity_relation",
                "relation": relation_type,
                "source_concept_id": source_id,
                "concept_id": target_id,
                "can_lock_variant": False,
            })
        return output


def _expression_matches(
    expression: str,
    query: str,
    *,
    original_query: str = "",
) -> bool:
    """Check whether *expression* is a safe token-level match in *query*.

    *query* is the normalized (whitespace-stripped) form, while
    *original_query* preserves the original whitespace and case for
    token-boundary checks on short Latin abbreviations.
    """

    if not expression or not query:
        return False
    if expression == query:
        return True
    if re.search(r"[\u4e00-\u9fff]", expression):
        return len(expression) >= 2 and expression in query
    # 2-char Latin abbreviations (DL, PE, etc.) are only safe when they
    # appear as a standalone token in the original query.
    if (
        len(expression) == 2
        and expression.isascii()
        and expression.isalpha()
        and original_query
    ):
        pattern = re.compile(
            r"(?:^|\s)" + re.escape(expression) + r"(?:\s|$)",
            flags=re.I,
        )
        return bool(pattern.search(original_query))
    return len(expression) >= 3 and expression in query


__all__ = [
    "CURATED_RELATION_TYPES",
    "SAFE_EQUIVALENCE_TYPES",
    "TERMINOLOGY_RELATIONS",
    "TERMINOLOGY_VERSION",
    "TerminologyResolver",
    "build_terminology_layer",
    "normalize_term",
    "terminology_quality_report",
    "write_terminology_layer",
]
