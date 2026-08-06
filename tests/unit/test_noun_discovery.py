from __future__ import annotations

import json
from pathlib import Path

import debug_agent_system.knowledge_v2.noun_discovery as discovery
from debug_agent_system.knowledge_v2.entity_terminology import (
    build_entity_projection,
)
from debug_agent_system.knowledge_v2.noun_discovery import (
    apply_approved_noun_discovery,
    build_noun_discovery_items,
)
from debug_agent_system.knowledge_v2.terminology import (
    TERMINOLOGY_VERSION,
    _clean_text,
    _stable_id,
    normalize_term,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    kg_root = data_root / "kg_v2"
    _write_json(
        kg_root / "terminology" / "noun_discovery_config.json",
        {
            "schema_version": "kg_v2.noun_discovery_config.v1",
            "max_examples_per_candidate": 3,
            "corpus_sources": {
                "chat_jsonl": ["imports/chat.jsonl"],
                "document_chunks": ["raw/chunks.json"],
                "support_records": ["raw/records.json"],
            },
            "candidate_terms": [
                {
                    "canonical_name": "轨道传感器",
                    "concept_type": "component",
                    "minimum_count": 3,
                    "minimum_source_kinds": 2,
                    "relation": {
                        "relation": "part_of",
                        "target_key": "equipment:aoi设备",
                    },
                }
            ],
            "variant_groups": [
                {
                    "canonical_name": "轨道传感器",
                    "surface_forms": ["感应头"],
                    "suggested_relation_type": "colloquial_alias",
                    "minimum_count": 2,
                }
            ],
        },
    )
    _write_json(
        kg_root / "objects" / "debug_concepts.json",
        [
            {
                "concept_id": "concept:aoi",
                "canonical_name": "AOI设备",
                "concept_type": "equipment",
            }
        ],
    )
    chat_path = data_root / "imports" / "chat.jsonl"
    chat_path.parent.mkdir(parents=True, exist_ok=True)
    chat_path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in [
                {
                    "message_id": "m1",
                    "plain_text": "检查轨道传感器和感应头",
                },
                {
                    "message_id": "m2",
                    "plain_text": "轨道传感器没有触发，感应头灯不亮",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        data_root / "raw" / "chunks.json",
        [
            {
                "text": "轨道传感器负责确认板件到位。",
                "metadata": {"chunk_id": "c1"},
            }
        ],
    )
    _write_json(
        data_root / "raw" / "records.json",
        {
            "objects": [
                {
                    "record_id": "r1",
                    "fields": {"问题": "轨道传感器无信号"},
                }
            ]
        },
    )
    _write_json(
        kg_root / "terminology" / "entity_ontology.json",
        {
            "schema_version": "kg_v2.entity_ontology.v1",
            "concepts": [
                {
                    "key": "equipment:aoi设备",
                    "canonical_name": "AOI设备",
                    "concept_type": "equipment",
                    "approved": True,
                }
            ],
            "relations": [],
            "aliases": [],
            "alias_candidates": [],
        },
    )
    return kg_root


def test_inventory_marks_type_patterns_as_unobserved_instance_facts() -> None:
    markdown = discovery.render_noun_terminology_inventory_markdown({
        "authoritative_concept_count": 2,
        "authoritative_alias_count": 0,
        "authoritative_relation_count": 1,
        "pending_candidate_count": 0,
        "authoritative_concepts": [
            {
                "concept_id": "concept:camera",
                "canonical_name": "相机",
                "concept_type": "equipment",
                "aliases": [],
            },
            {
                "concept_id": "concept:cxp",
                "canonical_name": "CXP采集连接",
                "concept_type": "connection",
                "aliases": [],
            },
        ],
        "authoritative_relations": [{
            "from_concept_id": "concept:camera",
            "from_name": "相机",
            "relation": "connected_via",
            "to_concept_id": "concept:cxp",
            "to_name": "CXP采集连接",
            "scope": "type_pattern",
            "evidence_required": True,
        }],
        "pending_candidates": [],
    })

    assert (
        "connected_via → CXP采集连接 [类型模板, 需实例证据]"
        in markdown
    )
    assert "并非对每个现场实例都成立" in markdown


def test_noun_discovery_is_multisource_and_non_authoritative(
    tmp_path: Path,
) -> None:
    items, report = build_noun_discovery_items(_fixture_root(tmp_path))

    assert report["corpus_record_counts"] == {
        "document_chunk": 1,
        "group_chat": 2,
        "support_record": 1,
    }
    concept = next(
        item for item in items
        if item["candidate_kind"] == "new_noun_concept"
    )
    assert concept["canonical_name"] == "轨道传感器"
    assert concept["corpus_counts"] == {
        "document_chunk": 1,
        "group_chat": 2,
        "support_record": 1,
    }
    assert concept["review_status"] == "pending"
    assert len(concept["corpus_examples"]) == 3
    assert {
        item["candidate_kind"] for item in items
    } == {
        "new_noun_concept",
        "noun_surface_variant",
        "noun_relation",
    }


def test_approved_or_pending_alias_deduplicates_new_concept_but_keeps_relation(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    concepts_path = root / "objects" / "debug_concepts.json"
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    concepts.append({
        "concept_id": "concept:legacy-sensor-head",
        "canonical_name": "感应头",
        "concept_type": "subsystem",
    })
    _write_json(concepts_path, concepts)
    ontology_path = root / "terminology" / "entity_ontology.json"
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology["concepts"][0]["key"] = "equipment:aoi-device"
    ontology["aliases"] = [
        {
            "surface_form": "轨道传感器",
            "concept_key": "equipment:aoi-device",
            "relation_type": "colloquial_alias",
            "approved": True,
        },
        {
            "surface_form": "感应头",
            "concept_key": "equipment:aoi-device",
            "relation_type": "colloquial_alias",
            "approved": True,
        },
    ]
    _write_json(ontology_path, ontology)
    config_path = root / "terminology" / "noun_discovery_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["candidate_terms"][0]["relation"][
        "target_key"
    ] = "equipment:aoi-device"
    _write_json(config_path, config)

    items, _ = build_noun_discovery_items(root)

    assert not any(
        item["candidate_kind"] == "new_noun_concept"
        and item.get("canonical_name") == "轨道传感器"
        for item in items
    )
    relation = next(
        item for item in items
        if item["candidate_kind"] == "noun_relation"
    )
    assert relation["proposed_from_key"] == "equipment:aoi-device"
    assert not any(
        item["candidate_kind"] == "noun_surface_variant"
        for item in items
    )

    ontology["aliases"] = []
    ontology["alias_candidates"] = [
        {
            "surface_form": "轨道传感器",
            "concept_key": "equipment:aoi-device",
            "candidate_kind": "noun_alias",
            "suggested_relation_type": "colloquial_alias",
            "risk": "medium",
        },
        {
            "surface_form": "感应头",
            "concept_key": "equipment:aoi-device",
            "candidate_kind": "noun_alias",
            "suggested_relation_type": "colloquial_alias",
            "risk": "medium",
        },
    ]
    _write_json(ontology_path, ontology)

    pending_items, _ = build_noun_discovery_items(root)
    assert not any(
        item["candidate_kind"] == "new_noun_concept"
        and item.get("canonical_name") == "轨道传感器"
        for item in pending_items
    )
    assert not any(
        item["candidate_kind"] == "noun_surface_variant"
        for item in pending_items
    )


def test_approved_exact_relation_is_removed_from_review_queue(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    ontology_path = root / "terminology" / "entity_ontology.json"
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology["relations"] = [{
        "from_key": "component:轨道传感器",
        "relation": "part_of",
        "to_key": "equipment:aoi设备",
        "basis": "approved_test_relation",
        "approved": True,
    }]
    _write_json(ontology_path, ontology)

    items, _ = build_noun_discovery_items(root)

    assert not any(
        item["candidate_kind"] == "noun_relation"
        and item.get("proposed_from_key") == "component:轨道传感器"
        and item.get("proposed_relation") == "part_of"
        and item.get("proposed_to_key") == "equipment:aoi设备"
        for item in items
    )
    assert any(
        item["candidate_kind"] == "new_noun_concept"
        for item in items
    )


def test_open_discovery_normalizes_models_files_and_separates_association(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    config_path = root / "terminology" / "noun_discovery_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["open_discovery"] = {
        "enabled": True,
        "model_minimum_count": 2,
        "artifact_minimum_count": 2,
        "variant_minimum_count": 1,
        "explicit_alias_minimum_count": 2,
        "associations": {
            "enabled": True,
            "minimum_record_count": 2,
            "minimum_source_kinds": 2,
            "minimum_jaccard": 0.01,
            "max_neighbors_per_concept": 5,
        },
    }
    _write_json(config_path, config)
    concepts_path = root / "objects" / "debug_concepts.json"
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    concepts.append({
        "concept_id": "concept:si2020t",
        "canonical_name": "SI2020T",
        "concept_type": "product_model",
    })
    _write_json(concepts_path, concepts)
    ontology_path = root / "terminology" / "entity_ontology.json"
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology["aliases"].append({
        "surface_form": "SI-2020T",
        "concept_key": "product_model:si2020t",
        "relation_type": "exact_synonym",
        "approved": True,
    })
    _write_json(ontology_path, ontology)
    chat_path = root.parent / "imports" / "chat.jsonl"
    with chat_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "message_id": "m3",
            "plain_text": (
                "AOI设备 SI-2020T 使用 smt-aoi.exe，"
                "忽略 smt-aoi-2025-09-24.log"
            ),
        }, ensure_ascii=False) + "\n")
        stream.write(json.dumps({
            "message_id": "m4",
            "plain_text": "设备型号2020T仍写作SI2020T",
        }, ensure_ascii=False) + "\n")
        stream.write(json.dumps({
            "message_id": "m5",
            "plain_text": "验证码:SY2023，识别码:544845472",
        }, ensure_ascii=False) + "\n")
        stream.write(json.dumps({
            "message_id": "m6",
            "plain_text": "长期验证码:SY2023",
        }, ensure_ascii=False) + "\n")
    chunks_path = root.parent / "raw" / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks.append({
        "text": "AOI设备 SI-2020T 通过 smt-aoi.exe 启动。",
        "metadata": {"chunk_id": "c2"},
    })
    _write_json(chunks_path, chunks)

    items, report = build_noun_discovery_items(root)

    assert not any(
        item.get("canonical_name") == "SI2020T"
        and item["candidate_kind"] == "new_noun_concept"
        for item in items
    )
    model_variants = {
        item["surface_form"]
        for item in items
        if item["candidate_kind"] == "noun_surface_variant"
        and item.get("suggested_concept_key") == "product_model:si2020t"
    }
    assert model_variants == {"2020T"}
    executable = next(
        item for item in items
        if item["candidate_kind"] == "new_noun_concept"
        and item.get("canonical_name") == "smt-aoi.exe"
    )
    assert executable["proposed_concept_type"] == "software"
    assert not any(
        "2025-09-24.log" in str(item.get("canonical_name") or "")
        for item in items
    )
    assert not any(
        item.get("canonical_name") == "SY2023"
        for item in items
    )
    association = next(
        item for item in items
        if item["candidate_kind"] == "noun_association"
        and {
            item["proposed_from_key"],
            item["proposed_to_key"],
        } == {
            "equipment:aoi设备",
            "software:smt-aoi.exe",
        }
    )
    assert association["proposed_relation"] == "associated_with"
    assert association["risk"] == "high"
    assert report["association_candidate_count"] >= 1


def test_only_fully_reviewed_nouns_enter_authoritative_ontology(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fixture_root(tmp_path)
    items, _ = build_noun_discovery_items(root)
    for item in items:
        item["selected_action"] = "approve"
        item["reviewed_by"] = "reviewer"
        if item["candidate_kind"] == "new_noun_concept":
            item["selected_canonical_name"] = item["canonical_name"]
            item["selected_concept_type"] = item[
                "proposed_concept_type"
            ]
        elif item["candidate_kind"] == "noun_surface_variant":
            item["selected_concept_key"] = item[
                "suggested_concept_key"
            ]
            item["approved_relation_type"] = item[
                "suggested_relation_type"
            ]
        else:
            item["selected_relation"] = item["proposed_relation"]
            item["selected_target_key"] = item["proposed_to_key"]
    _write_json(
        root / "review_queue" / "noun_discovery_candidates.json",
        items,
    )
    monkeypatch.setattr(
        discovery,
        "write_terminology_layer",
        lambda _root: {"revision": "test-revision"},
    )

    result = apply_approved_noun_discovery(root)
    ontology = json.loads(
        (root / "terminology" / "entity_ontology.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["rejected_approval_count"] == 0
    assert result["added_concept_count"] == 1
    assert result["added_surface_variant_count"] == 1
    assert result["added_relation_count"] == 1
    assert ontology["aliases"][0]["surface_form"] == "感应头"
    assert ontology["aliases"][0]["relation_type"] == "colloquial_alias"
    assert ontology["alias_candidates"] == []

    projection = build_entity_projection(
        root=root,
        objects={
            "FaultVariant": [],
            "FaultFamily": [],
        },
        stable_id=_stable_id,
        clean_text=_clean_text,
        normalize_term=normalize_term,
        terminology_version=TERMINOLOGY_VERSION,
    )
    approved_alias = next(
        item for item in projection["alias_sources"]
        if item["surface"] == "感应头"
    )
    assert approved_alias["relation_type"] == "colloquial_alias"
    assert approved_alias["approved"] is True
    assert projection["report"]["approved_noun_alias_count"] == 1


def test_approval_without_explicit_selection_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fixture_root(tmp_path)
    items, _ = build_noun_discovery_items(root)
    concept = next(
        item for item in items
        if item["candidate_kind"] == "new_noun_concept"
    )
    concept["selected_action"] = "approve"
    concept["reviewed_by"] = "reviewer"
    _write_json(
        root / "review_queue" / "noun_discovery_candidates.json",
        [concept],
    )
    monkeypatch.setattr(
        discovery,
        "write_terminology_layer",
        lambda _root: {"revision": "test-revision"},
    )

    result = apply_approved_noun_discovery(root)

    assert result["approved_candidate_count"] == 0
    assert result["rejected_approval_count"] == 1
    assert result["rejected_approvals"][0]["reasons"] == [
        "missing_canonical_name",
        "invalid_concept_type",
    ]


def test_conflicting_review_status_and_action_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fixture_root(tmp_path)
    items, _ = build_noun_discovery_items(root)
    concept = next(
        item for item in items
        if item["candidate_kind"] == "new_noun_concept"
    )
    concept.update({
        "review_status": "rejected",
        "selected_action": "approve",
        "selected_canonical_name": concept["canonical_name"],
        "selected_concept_type": concept["proposed_concept_type"],
        "reviewed_by": "reviewer",
    })
    _write_json(
        root / "review_queue" / "noun_discovery_candidates.json",
        [concept],
    )
    monkeypatch.setattr(
        discovery,
        "write_terminology_layer",
        lambda _root: {"revision": "test-revision"},
    )

    result = apply_approved_noun_discovery(root)

    assert result["approved_candidate_count"] == 0
    assert result["rejected_approval_count"] == 1
    assert result["rejected_approvals"][0]["reasons"] == [
        "decision_conflict",
    ]


def test_new_concept_candidate_can_merge_into_existing_concept(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fixture_root(tmp_path)
    candidate = {
        "review_id": "noun-discovery:merge-aoi",
        "candidate_kind": "new_noun_concept",
        "canonical_name": "AOI",
        "review_status": "approved",
        "selected_action": "approve",
        "selected_concept_key": "equipment:aoi设备",
        "approved_relation_type": "abbreviation",
        "reviewed_by": "reviewer",
    }
    _write_json(
        root / "review_queue" / "noun_discovery_candidates.json",
        [candidate],
    )
    monkeypatch.setattr(
        discovery,
        "write_terminology_layer",
        lambda _root: {"revision": "test-revision"},
    )

    result = apply_approved_noun_discovery(root)
    ontology = json.loads(
        (root / "terminology" / "entity_ontology.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["rejected_approval_count"] == 0
    assert result["added_concept_count"] == 0
    assert result["added_surface_variant_count"] == 1
    assert ontology["aliases"] == [
        {
            "surface_form": "AOI",
            "concept_key": "equipment:aoi设备",
            "relation_type": "abbreviation",
            "approved": True,
            "review": {
                "review_id": "noun-discovery:merge-aoi",
                "reviewed_by": "reviewer",
                "note": "",
            },
        }
    ]
