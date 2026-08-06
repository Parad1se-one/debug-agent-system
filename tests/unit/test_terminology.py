from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import debug_agent_system.knowledge_v2.terminology as terminology_module
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.terminology import (
    TerminologyResolver,
    build_terminology_layer,
    normalize_term,
)
from debug_agent_system.knowledge_v2.terminology_review import (
    approved_entries_from_reviews,
    build_terminology_review_items,
)
from debug_agent_system.knowledge_v2.validator import validate_graph


REPO_ROOT = Path(__file__).resolve().parents[2]
KG_ROOT = REPO_ROOT / "data/kg_v2"


def test_current_graph_has_total_deterministic_domain_projection() -> None:
    store = JsonKGV2Store(KG_ROOT)
    first = build_terminology_layer(store)
    second = build_terminology_layer(store)

    expected = sum(
        len(store.objects_by_type[object_type])
        for object_type in (
            "FaultFamily",
            "FaultVariant",
            "DiagnosticAction",
        )
    )
    assert first["report"]["concept_count"] >= expected
    assert first["report"]["revision"] == second["report"]["revision"]
    assert first["report"]["relation_counts"]["primary_concept"] == expected
    assert first["report"]["operation_instance_count"] == len(
        store.objects_by_type["DiagnosticAction"]
    )
    assert (
        first["report"]["operation_concept_count"]
        < first["report"]["operation_instance_count"]
    )
    assert first["report"]["shared_operation_concept_count"] > 0
    assert {
        item["concept_type"]
        for item in first["objects_by_type"]["DebugConcept"]
        if not item.get("canonical_target_id")
    } >= {"category", "subsystem", "equipment", "phase"}
    assert validate_graph(
        first["objects_by_type"],
        first["relations"],
        schema_root=KG_ROOT / "schema",
    ) == []


def test_noun_projection_atomizes_context_and_links_entities() -> None:
    built = build_terminology_layer(KG_ROOT)
    concepts = built["objects_by_type"]["DebugConcept"]
    by_name = {
        str(item["canonical_name"]): item
        for item in concepts
    }
    relation_keys = {
        (
            str(relation["from"]),
            str(relation["relation"]),
            str(relation["to"]),
        )
        for relation in built["relations"]
    }

    assert by_name["工控机"]["concept_type"] == "equipment"
    assert by_name["SI2020T"]["concept_type"] == "product_model"
    assert by_name["复判站"]["concept_type"] == "station"
    assert by_name["复判站软件"]["concept_type"] == "software"
    assert by_name["CXP链路"]["concept_type"] == "interface"
    assert by_name["PCB"]["concept_type"] == "workpiece"
    assert by_name["SI2020T/工控机"]["status"] == "legacy"
    assert len([
        item
        for item in concepts
        if item["canonical_name"] == "工控机"
    ]) == 1
    assert not any(
        item["canonical_name"] == "软件"
        and item["concept_type"] == "software"
        for item in concepts
    )
    assert (
        by_name["SI2020T"]["concept_id"],
        "model_of",
        by_name["工控机"]["concept_id"],
    ) in relation_keys
    assert (
        by_name["复判站软件"]["concept_id"],
        "runs_on",
        by_name["复判站"]["concept_id"],
    ) in relation_keys
    assert (
        by_name["3D相机"]["concept_id"],
        "is_a",
        by_name["相机"]["concept_id"],
    ) in relation_keys
    assert (
        by_name["PCB"]["concept_id"],
        "processed_by",
        by_name["AOI设备"]["concept_id"],
    ) in relation_keys


def test_workstation_and_connection_patterns_preserve_instance_guards() -> None:
    built = build_terminology_layer(KG_ROOT)
    by_name = {
        str(item["canonical_name"]): item
        for item in built["objects_by_type"]["DebugConcept"]
    }
    relations = built["relations"]

    assert by_name["工作站"]["concept_type"] == "workstation"
    assert by_name["USB外设连接"]["concept_type"] == "connection"
    assert by_name["USB协议"]["concept_type"] == "protocol"

    def relation(
        source: str,
        relation_type: str,
        target: str,
    ) -> dict[str, object]:
        return next(
            item
            for item in relations
            if item.get("from") == by_name[source]["concept_id"]
            and item.get("relation") == relation_type
            and item.get("to") == by_name[target]["concept_id"]
        )

    assert relation(
        "复判工作站",
        "deployed_at",
        "复判站",
    )["basis"] == "approved_workstation_deployment"
    assert relation(
        "复判站软件",
        "runs_on",
        "复判工作站",
    )["basis"] == "approved_workstation_software_layer"

    usb_mouse = relation("鼠标", "connected_via", "USB外设连接")
    assert usb_mouse["scope"] == "type_pattern"
    assert usb_mouse["evidence_required"] is True
    assert relation(
        "USB接口",
        "endpoint_of",
        "USB外设连接",
    )["scope"] == "type_pattern"
    assert relation(
        "USB外设连接",
        "uses_protocol",
        "USB协议",
    )["scope"] == "type_pattern"

    upstream = relation("产线上游设备", "signals_to", "AOI设备")
    assert upstream["direction"] == "upstream_to_aoi"
    assert upstream["evidence_required"] is True


def test_software_stack_artifacts_and_data_files_are_not_flattened() -> None:
    built = build_terminology_layer(KG_ROOT)
    by_name = {
        str(item["canonical_name"]): item
        for item in built["objects_by_type"]["DebugConcept"]
    }
    relations = built["relations"]

    assert by_name["显卡驱动"]["concept_type"] == "driver"
    assert by_name["BIOS"]["concept_type"] == "firmware"
    assert by_name["相机 SDK"]["concept_type"] == "sdk"
    assert by_name["smt-aoi.exe"]["concept_type"] == "software_artifact"
    assert by_name["软件运行进程"]["concept_type"] == "runtime_process"
    assert by_name["machine.toml"]["concept_type"] == "configuration_file"
    assert by_name["host.db"]["concept_type"] == "database_file"
    assert by_name["日志产物"]["concept_type"] == "log_artifact"
    assert by_name["sysinfo.json"]["concept_type"] == "diagnostic_artifact"
    assert by_name["PROJ工程数据"]["concept_type"] == "data_artifact"
    assert by_name["CAD工程数据"]["concept_type"] == "data_artifact"
    assert by_name["Gerber工程数据"]["concept_type"] == "data_artifact"
    assert by_name["BOM物料清单"]["concept_type"] == "data_artifact"
    assert by_name["图像产物"]["concept_type"] == "data_artifact"
    assert by_name["整板大图"]["concept_type"] == "data_artifact"
    assert by_name["白图"]["concept_type"] == "data_artifact"
    assert by_name["RGB图"]["concept_type"] == "data_artifact"
    assert by_name["explorer.exe"]["concept_type"] == "runtime_process"
    assert by_name["SPI设备"]["concept_type"] == "equipment"
    assert by_name["上板机"]["concept_type"] == "equipment"
    assert by_name["收板机"]["concept_type"] == "equipment"
    assert by_name["接驳台"]["concept_type"] == "equipment"
    assert by_name["缓存机"]["concept_type"] == "equipment"
    assert by_name["回流焊设备"]["concept_type"] == "equipment"
    assert by_name["工业相机"]["concept_type"] == "equipment"

    def relation(
        source: str,
        relation_type: str,
        target: str,
    ) -> dict[str, object]:
        return next(
            item
            for item in relations
            if item.get("from") == by_name[source]["concept_id"]
            and item.get("relation") == relation_type
            and item.get("to") == by_name[target]["concept_id"]
        )

    assert relation(
        "显卡驱动",
        "driver_of",
        "显卡",
    )["basis"] == "approved_device_driver_binding"
    assert relation(
        "BIOS",
        "firmware_of",
        "主板",
    )["basis"] == "approved_device_firmware_binding"
    assert relation(
        "相机 SDK",
        "sdk_for",
        "相机",
    )["evidence_required"] is True
    assert relation(
        "smt-aoi.exe",
        "artifact_of",
        "AOI主程序",
    )["basis"] == "approved_deployment_artifact_binding"
    assert relation(
        "machined.toml",
        "configuration_of",
        "Machine服务",
    )["basis"] == "approved_configuration_binding"
    assert relation(
        "整板大图",
        "is_a",
        "图像产物",
    )["basis"] == "approved_image_artifact_taxonomy"
    assert relation(
        "白图",
        "is_a",
        "图像产物",
    )["basis"] == "approved_image_artifact_taxonomy"
    assert relation(
        "RGB图",
        "is_a",
        "图像产物",
    )["basis"] == "approved_image_artifact_taxonomy"
    assert relation(
        "explorer.exe",
        "runs_on",
        "Windows",
    )["basis"] == "approved_operating_system_process_binding"
    assert relation(
        "工业相机",
        "is_a",
        "相机",
    )["basis"] == "approved_camera_taxonomy"
    for adjacent_equipment in (
        "SPI设备",
        "上板机",
        "收板机",
        "接驳台",
        "缓存机",
        "回流焊设备",
    ):
        assert not any(
            item.get("from") == by_name[adjacent_equipment]["concept_id"]
            and item.get("relation") in {
                "connected_to",
                "signals_to",
            }
            and item.get("to") == by_name["AOI设备"]["concept_id"]
            for item in relations
        )
    assert not any(
        item.get("from") == by_name["machine.toml"]["concept_id"]
        and item.get("relation") == "configuration_of"
        for item in relations
    )
    assert not any(
        item.get("from") == by_name["host.db"]["concept_id"]
        and item.get("relation") == "database_of"
        for item in relations
    )

    resolver = TerminologyResolver(
        concepts=built["objects_by_type"]["DebugConcept"],
        expressions=built["objects_by_type"]["TermExpression"],
        senses=built["objects_by_type"]["TermSense"],
        relations=relations,
    )
    ddu = resolver.resolve("使用 DDU 清理显卡驱动")
    assert any(
        item["concept"]["canonical_name"] == "Display Driver Uninstaller"
        for item in ddu["resolved_mentions"]
    )
    artifact = resolver.resolve("smt-aoi.exe 启动失败")
    assert any(
        item.get("authority") == "entity_relation"
        and item.get("relation") == "artifact_of"
        and item.get("text") == "AOI主程序"
        for item in artifact["retrieval_expansions"]
    )


def test_reviewed_noun_alias_resolves_while_unreviewed_alias_stays_hint() -> None:
    built = build_terminology_layer(KG_ROOT)
    by_name = {
        str(item["canonical_name"]): item
        for item in built["objects_by_type"]["DebugConcept"]
    }
    resolver = TerminologyResolver(
        concepts=built["objects_by_type"]["DebugConcept"],
        expressions=built["objects_by_type"]["TermExpression"],
        senses=built["objects_by_type"]["TermSense"],
        relations=built["relations"],
    )

    result = resolver.resolve("复盘站不出图")

    mention = next(
        item
        for item in result["resolved_mentions"]
        if item["surface_form"] == "复盘站"
    )
    assert mention["concept"]["canonical_name"] == "复判站"
    assert "typo_variant" in mention["relation_types"]
    expansion = next(
        item
        for item in result["retrieval_expansions"]
        if item["text"] == "复判站"
    )
    assert expansion["text"] == "复判站"
    assert expansion["authority"] == "approved_equivalence"
    assert expansion["can_lock_variant"] is False
    assert any(
        item["source"]["canonical_name"] == "复判站"
        and item["relation"] == "connected_to"
        and item["target"]["canonical_name"] == "AOI设备"
        for item in result["entity_relations"]
    )
    assert not any(
        item["relation"] == "context_member"
        for item in result["entity_relations"]
    )
    unreviewed = resolver.resolve("主机不开机")
    assert not any(
        item["concept"]["canonical_name"] == "工控机"
        for item in unreviewed["resolved_mentions"]
    )
    unreviewed_hint = next(
        item
        for item in unreviewed["supporting_concepts"]
        if item["surface_form"] == "主机"
    )
    assert unreviewed_hint["concept"]["canonical_name"] == "工控机"
    assert unreviewed_hint["can_lock_variant"] is False
    model_result = resolver.resolve("SI2020T不开机")
    model_expansion = next(
        item
        for item in model_result["retrieval_expansions"]
        if item.get("authority") == "entity_relation"
    )
    assert model_expansion["text"] == "工控机"
    assert model_expansion["relation"] == "model_of"
    assert model_expansion["can_lock_variant"] is False

    association_resolver = TerminologyResolver(
        concepts=built["objects_by_type"]["DebugConcept"],
        expressions=built["objects_by_type"]["TermExpression"],
        senses=built["objects_by_type"]["TermSense"],
        relations=built["relations"] + [{
            "from": by_name["复判站"]["concept_id"],
            "to": by_name["PCB"]["concept_id"],
            "relation": "associated_with",
            "basis": "human_reviewed_corpus_discovery",
        }],
    )
    association_result = association_resolver.resolve("复判站不出图")
    association = next(
        item
        for item in association_result["entity_relations"]
        if item["relation"] == "associated_with"
    )
    assert association["target"]["canonical_name"] == "PCB"
    assert association["can_expand_retrieval"] is False
    assert association["can_lock_variant"] is False
    assert not any(
        item.get("authority") == "entity_relation"
        and item.get("text") == "PCB"
        for item in association_result["retrieval_expansions"]
    )


def test_noun_review_queue_uses_corpus_evidence_and_skips_operations() -> None:
    items, report = build_terminology_review_items(KG_ROOT)
    noun_items = [
        item
        for item in items
        if item.get("review_domain") == "noun_entity"
    ]

    assert report["noun_candidate_count"] >= 3
    assert noun_items
    assert all(
        item["candidate_kind"].startswith("noun_")
        or item["candidate_kind"] == "ambiguous_expression"
        for item in noun_items
    )
    surfaces = {item["surface_form"] for item in noun_items}
    assert {"主机", "板子", "PE"} <= surfaces
    assert not {"复盘站", "IPC", "PCB板"} & surfaces
    assert not any(
        {
            str(concept.get("concept_type") or "")
            for concept in item["candidate_concepts"]
        } == {"operation"}
        for item in items
    )


def test_legacy_keywords_are_search_hints_not_equivalences() -> None:
    built = build_terminology_layer(KG_ROOT)
    senses = built["objects_by_type"]["TermSense"]
    keyword_senses = [
        item
        for item in senses
        if item["relation_type"] == "search_hint"
    ]

    assert keyword_senses
    assert all(item["approved"] is True for item in keyword_senses)
    assert not any(
        item["relation_type"] in {
            "exact_synonym",
            "colloquial_alias",
            "abbreviation",
        }
        for item in keyword_senses
    )


def test_resolver_exposes_ambiguity_and_does_not_promote_search_hint() -> None:
    shared_term = {
        "term_id": "term:shared",
        "surface_form": "初始化异常",
        "normalized_form": normalize_term("初始化异常"),
        "expression_type": "canonical",
    }
    hint_term = {
        "term_id": "term:hint",
        "surface_form": "通信",
        "normalized_form": normalize_term("通信"),
        "expression_type": "keyword",
    }
    concepts = [
        {
            "concept_id": "concept:a",
            "canonical_name": "设备甲初始化异常",
            "concept_type": "fault_variant",
            "canonical_target_type": "FaultVariant",
            "canonical_target_id": "variant:a",
            "status": "approved",
        },
        {
            "concept_id": "concept:b",
            "canonical_name": "设备乙初始化异常",
            "concept_type": "fault_variant",
            "canonical_target_type": "FaultVariant",
            "canonical_target_id": "variant:b",
            "status": "approved",
        },
    ]
    senses = [
        {
            "sense_id": "sense:a",
            "term_id": "term:shared",
            "concept_id": "concept:a",
            "relation_type": "canonical",
            "approved": True,
        },
        {
            "sense_id": "sense:b",
            "term_id": "term:shared",
            "concept_id": "concept:b",
            "relation_type": "canonical",
            "approved": True,
        },
        {
            "sense_id": "sense:hint",
            "term_id": "term:hint",
            "concept_id": "concept:a",
            "relation_type": "search_hint",
            "approved": True,
        },
    ]
    resolver = TerminologyResolver(
        concepts=deepcopy(concepts),
        expressions=[shared_term, hint_term],
        senses=senses,
    )

    result = resolver.resolve("初始化异常并伴随通信提示")

    assert result["ambiguous_mentions"][0]["surface_form"] == "初始化异常"
    assert result["supporting_concepts"][0]["can_lock_variant"] is False
    assert result["safe_expansions"] == []


def test_approved_curated_alias_becomes_safe_expansion(monkeypatch) -> None:
    store = JsonKGV2Store(KG_ROOT)
    family = store.objects_by_type["FaultFamily"][0]
    alias = "经审核的测试别名"
    monkeypatch.setattr(
        terminology_module,
        "_load_curated_entries",
        lambda _root: [{
            "surface_form": alias,
            "relation_type": "colloquial_alias",
            "canonical_target_type": "FaultFamily",
            "canonical_target_id": family["family_id"],
            "approved": True,
        }],
    )

    built = build_terminology_layer(store)
    resolver = TerminologyResolver(
        concepts=built["objects_by_type"]["DebugConcept"],
        expressions=built["objects_by_type"]["TermExpression"],
        senses=built["objects_by_type"]["TermSense"],
    )
    result = resolver.resolve(alias)

    assert result["resolved_mentions"][0]["concept"][
        "canonical_target_id"
    ] == family["family_id"]
    assert family["label"] in result["safe_expansions"]


def test_repeated_action_occurrences_share_semantic_operation_concept() -> None:
    store = JsonKGV2Store(KG_ROOT)
    built = build_terminology_layer(store)
    primary = {
        str(relation.get("from") or ""): str(relation.get("to") or "")
        for relation in built["relations"]
        if relation.get("relation") == "primary_concept"
    }
    actions = store.objects_by_type["DiagnosticAction"]
    repeated = [
        item
        for item in actions
        if item.get("label") == "观察是否复发"
        and item.get("action_role") == "observe"
    ]

    assert len(repeated) > 1
    assert len({
        primary[str(item["action_id"])] for item in repeated
    }) == 1
    concept_id = primary[str(repeated[0]["action_id"])]
    concept = next(
        item
        for item in built["objects_by_type"]["DebugConcept"]
        if item["concept_id"] == concept_id
    )
    assert concept["source_object_ids"] == sorted(
        str(item["action_id"]) for item in repeated
    )
    assert not concept.get("canonical_target_id")

    same_label_different_roles = [
        item
        for item in actions
        if item.get("label") == "检查电源线"
    ]
    assert {
        str(item.get("action_role") or "")
        for item in same_label_different_roles
    } >= {"inspect", "change"}
    assert len({
        primary[str(item["action_id"])]
        for item in same_label_different_roles
    }) >= 2


def test_context_disambiguation_resolves_only_with_sufficient_margin() -> None:
    expressions = [{
        "term_id": "term:shared",
        "surface_form": "初始化异常",
        "normalized_form": normalize_term("初始化异常"),
        "expression_type": "canonical",
    }]
    concepts = [
        {
            "concept_id": "concept:ipc",
            "canonical_name": "工控机初始化异常",
            "concept_type": "fault_variant",
            "canonical_target_type": "FaultVariant",
            "canonical_target_id": "variant:ipc",
            "status": "approved",
        },
        {
            "concept_id": "concept:camera",
            "canonical_name": "相机初始化异常",
            "concept_type": "fault_variant",
            "canonical_target_type": "FaultVariant",
            "canonical_target_id": "variant:camera",
            "status": "approved",
        },
        {
            "concept_id": "concept:equipment-camera",
            "canonical_name": "相机",
            "concept_type": "equipment",
            "status": "approved",
        },
    ]
    senses = [
        {
            "sense_id": "sense:ipc",
            "term_id": "term:shared",
            "concept_id": "concept:ipc",
            "relation_type": "canonical",
            "approved": True,
            "equipment_types": ["工控机"],
            "subsystems": ["Windows"],
            "phases": ["开机"],
        },
        {
            "sense_id": "sense:camera",
            "term_id": "term:shared",
            "concept_id": "concept:camera",
            "relation_type": "canonical",
            "approved": True,
            "equipment_types": ["相机"],
            "subsystems": ["采集链路"],
            "phases": ["检测"],
        },
    ]
    resolver = TerminologyResolver(
        concepts=concepts,
        expressions=expressions,
        senses=senses,
    )

    unresolved = resolver.resolve("初始化异常")
    resolved = resolver.resolve(
        "初始化异常",
        context={
            "equipment_types": ["相机"],
            "subsystems": ["采集链路"],
        },
    )
    auto_resolved = resolver.resolve("相机初始化异常")

    assert unresolved["resolved_mentions"] == []
    assert unresolved["ambiguous_mentions"][0][
        "reason"
    ] == "context_required"
    assert resolved["ambiguous_mentions"] == []
    assert resolved["resolved_mentions"][0][
        "resolution_method"
    ] == "context_disambiguation"
    assert resolved["resolved_mentions"][0]["concept"][
        "canonical_target_id"
    ] == "variant:camera"
    assert resolved["resolved_mentions"][0]["top_margin"] >= 2.0
    assert auto_resolved["detected_context"]["equipment_types"] == ["相机"]
    assert auto_resolved["resolved_mentions"][0]["concept"][
        "canonical_target_id"
    ] == "variant:camera"


def test_review_queue_is_pending_and_preserves_explicit_decision() -> None:
    first, report = build_terminology_review_items(KG_ROOT)

    assert report["alias_promotion_count"] > 0
    assert report["ambiguous_expression_count"] > 0
    assert all(item["review_status"] == "pending" for item in first)
    decided = deepcopy(first)
    decided[0]["review_status"] = "rejected"
    decided[0]["selected_action"] = "reject"
    decided[0]["reviewed_by"] = "domain-reviewer"

    second, _ = build_terminology_review_items(
        KG_ROOT,
        existing_items=decided,
    )
    preserved = next(
        item
        for item in second
        if item["review_id"] == decided[0]["review_id"]
    )

    assert preserved["review_status"] == "rejected"
    assert preserved["reviewed_by"] == "domain-reviewer"


def test_review_approval_requires_target_relation_and_reviewer() -> None:
    items, _ = build_terminology_review_items(KG_ROOT)
    candidate = deepcopy(next(
        item
        for item in items
        if item["candidate_kind"] == "alias_promotion"
    ))
    concept_id = str(candidate["suggested_concept_id"])
    concepts = {
        concept_id: next(
            item
            for item in candidate["candidate_concepts"]
            if item["concept_id"] == concept_id
        )
    }
    candidate.update({
        "review_status": "approved",
        "selected_concept_id": concept_id,
        "approved_relation_type": "colloquial_alias",
        "reviewed_by": "domain-reviewer",
    })

    entries, rejected = approved_entries_from_reviews(
        [candidate],
        concepts=concepts,
    )
    invalid = deepcopy(candidate)
    invalid["reviewed_by"] = ""
    _, invalid_rejected = approved_entries_from_reviews(
        [invalid],
        concepts=concepts,
    )

    assert rejected == []
    assert entries[0]["concept_id"] == concept_id
    assert entries[0]["approved"] is True
    assert invalid_rejected[0]["reasons"] == ["missing_reviewer"]

    conflicting = deepcopy(candidate)
    conflicting["review_status"] = "rejected"
    conflicting["selected_action"] = "approve"
    _, conflict_rejected = approved_entries_from_reviews(
        [conflicting],
        concepts=concepts,
    )
    assert conflict_rejected[0]["reasons"] == [
        "conflicting_review_decision",
    ]
