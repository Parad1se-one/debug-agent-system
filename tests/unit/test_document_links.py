from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from debug_agent_system.knowledge_v2.document_links import (
    DOCUMENT_LINK_RELATIONS,
    build_document_link_graph,
    extract_docx_hyperlinks,
)
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import (
    SqliteSAGV2,
    build_sqlite_sag_v2,
)
from debug_agent_system.knowledge_v2.validator import validate_graph


REPO_ROOT = Path(__file__).parents[2]
RAW_ROOT = REPO_ROOT / "data/raw/aoi_debug_agent_sources"


def _documents() -> list[dict]:
    return json.loads(
        (REPO_ROOT / "data/kg_v2/objects/knowledge_documents.json").read_text(
            encoding="utf-8"
        )
    )


def _title_by_id(documents: list[dict]) -> dict[str, str]:
    return {
        str(item.get("document_id") or ""): str(item.get("title") or "")
        for item in documents
    }


def test_docx_navigation_links_preserve_distinct_feishu_targets() -> None:
    links = extract_docx_hyperlinks(RAW_ROOT / "如何进入安全模式.docx")

    assert [(item["link_text"], item["wiki_token"]) for item in links] == [
        ("可以进入系统", "MCwGwfrw6iWn1dkRvNsc2R8Xnid"),
        ("无法进入系统", "NUfmw7PyKiYWGLkEMhqcM9ndnzn"),
    ]
    assert all(item["standalone"] for item in links)
    assert [item["relationship_id"] for item in links] == ["rId4", "rId5"]


def test_dism_backup_navigation_preserves_relationship_anchor() -> None:
    links = extract_docx_hyperlinks(RAW_ROOT / "Dism++软件使用教程.docx")
    backup = next(item for item in links if item["link_text"] == "制作镜像/备份镜像")

    assert backup["relationship_id"] == "rId5"
    assert backup["wiki_token"] == "XuDgwZpkjiFtKQkinnJc46dGnPh"


def test_all_available_canonical_document_links_are_resolved_and_typed() -> None:
    documents = _documents()
    title_by_id = _title_by_id(documents)
    relations, report = build_document_link_graph(REPO_ROOT, documents)
    by_pair = {
        (
            title_by_id.get(str(item.get("from") or "")),
            title_by_id.get(str(item.get("to") or "")),
        ): item
        for item in relations
    }

    assert report["feishu_wiki_link_count"] == 58
    assert report["resolved_occurrence_count"] == 47
    assert report["resolved_relation_count"] == 44
    assert report["unresolved_count"] == 9
    assert by_pair[
        ("如何进入安全模式.docx", "可以进入系统.docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("如何进入安全模式.docx", "无法进入系统 (1).docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("Windows系统_引导修复.docx", "可以进系统.docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("Windows系统_引导修复.docx", "无法进入系统.docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("可以进系统.docx", "快速系统文件修复.docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("可以进系统.docx", "修复系统.docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("可以进系统.docx", "修复引导.docx")
    ]["relation"] == "has_child_document"
    assert by_pair[
        ("无法进入系统.docx", "开机后一直转圈无法进去系统.docx")
    ]["relation"] == "references_document"
    assert not any(item.get("from") == item.get("to") for item in relations)


def test_document_link_relations_validate_with_the_current_graph() -> None:
    store = JsonKGV2Store(REPO_ROOT / "data/kg_v2")
    objects = {
        object_type: list(items)
        for object_type, items in store.objects_by_type.items()
    }
    relations, _report = build_document_link_graph(
        REPO_ROOT,
        objects["KnowledgeDocument"],
    )
    proposed = [
        dict(item)
        for item in store.relations
        if str(item.get("relation") or "") not in DOCUMENT_LINK_RELATIONS
    ]
    proposed.extend(relations)

    assert validate_graph(
        objects,
        proposed,
        schema_root=REPO_ROOT / "data/kg_v2/schema",
    ) == []


def test_sag_expands_only_the_two_safe_mode_navigation_children() -> None:
    source_store = JsonKGV2Store(REPO_ROOT / "data/kg_v2")
    objects = {
        object_type: list(items)
        for object_type, items in source_store.objects_by_type.items()
    }
    link_relations, _report = build_document_link_graph(
        REPO_ROOT,
        objects["KnowledgeDocument"],
    )
    relations = [
        dict(item)
        for item in source_store.relations
        if str(item.get("relation") or "") not in DOCUMENT_LINK_RELATIONS
    ]
    relations.extend(link_relations)

    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "data") as tmp:
        root = Path(tmp)
        shutil.copytree(REPO_ROOT / "data/kg_v2/schema", root / "schema")
        assert JsonKGV2Store(root).replace_graph(
            objects, relations, validate=True
        )["status"] == "replaced"
        sag_path = root / "read.sqlite"
        build_sqlite_sag_v2(root, sag_path, reset=True)
        retrieval = SqliteSAGV2(sag_path).retrieve(
            "如何进入安全模式", chunk_limit=64
        )

    direct_titles = [
        item["source_label"]
        for item in retrieval["trace"]["direct_document_matches"]
    ]
    navigation_titles = [
        item["source_label"]
        for item in retrieval["trace"]["navigation_document_matches"]
    ]
    answer_text = "\n".join(
        str(item.get("text") or "") for item in retrieval["chunks"]
        if item.get("direct_document_match")
    )
    assert direct_titles == ["如何进入安全模式"]
    assert navigation_titles == ["可以进入系统.docx", "无法进入系统 (1).docx"]
    assert "按住Shift" in answer_text
    assert "4 或 F4" in answer_text
    assert "5 或 F5" in answer_text
    assert "制作启动盘" not in answer_text
    assert "重装系统" not in answer_text
    assert "名称 | 所有者 | 修改时间" not in answer_text
    assert all(
        item.get("navigation_document_match")
        for item in retrieval["chunks"]
        if item.get("direct_document_match")
    )


def test_sag_does_not_expand_a_partially_exported_navigation_page() -> None:
    sag = SqliteSAGV2(REPO_ROOT / "data/kg_v2_sag/debug_agent_v2.sqlite")
    retrieval = sag.retrieve("Dism++软件使用教程", chunk_limit=64)

    assert retrieval["trace"]["direct_document_matches"]
    assert retrieval["trace"]["navigation_document_matches"] == []
    direct_text = "\n".join(
        str(item.get("text") or "")
        for item in retrieval["chunks"]
        if item.get("direct_document_match")
    )
    assert "一般电脑选择X64版本" in direct_text
    assert "制作镜像/备份镜像" in direct_text
    assert any(
        item.get("reason") == "partial_navigation_document"
        for item in retrieval["trace"]["navigation_excluded"]
    )


def test_sag_selects_query_relevant_second_hop_repair_documents() -> None:
    sag = SqliteSAGV2(REPO_ROOT / "data/kg_v2_sag/debug_agent_v2.sqlite")
    retrieval = sag.retrieve(
        "如何进行Windows系统/引导修复",
        chunk_limit=64,
    )

    navigation = retrieval["trace"]["navigation_document_matches"]
    assert [
        (item["navigation_depth"], item["source_label"])
        for item in navigation
    ] == [
        (1, "可以进系统.docx"),
        (2, "快速系统文件修复.docx"),
        (2, "修复系统.docx"),
        (2, "修复引导.docx"),
        (1, "无法进入系统.docx"),
    ]
    assert navigation[1]["navigation_path"] == [
        "Windows系统_引导修复.docx",
        "可以进系统.docx",
        "快速系统文件修复.docx",
    ]
    assert all(
        item.get("selection_reason") == "query_branch_match"
        for item in navigation
        if item["navigation_depth"] == 2
    )
    assert any(
        item.get("source_label") == "Dism++软件使用教程.docx"
        and item.get("reason") == "query_branch_mismatch"
        for item in retrieval["trace"]["navigation_excluded"]
    )
    assert retrieval["trace"]["navigation_max_depth"] == 2
    answer_text = "\n".join(
        str(item.get("text") or "")
        for item in retrieval["chunks"]
        if item.get("direct_document_match")
    )
    assert "sfc /scannow" in answer_text
    assert "修复系统" in answer_text
    assert "修复引导" in answer_text


def test_sag_second_hop_respects_system_and_boot_branch_intent() -> None:
    sag = SqliteSAGV2(REPO_ROOT / "data/kg_v2_sag/debug_agent_v2.sqlite")

    system_retrieval = sag.retrieve(
        "如何修复Windows系统",
        chunk_limit=64,
    )
    system_titles = {
        str(item["source_label"]).removesuffix(".docx")
        for item in [
            *system_retrieval["trace"]["direct_document_matches"],
            *system_retrieval["trace"]["navigation_document_matches"],
        ]
    }
    assert "快速系统文件修复" in system_titles
    assert "修复系统" in system_titles
    assert "修复引导" not in system_titles

    boot_retrieval = sag.retrieve(
        "如何修复Windows引导",
        chunk_limit=64,
    )
    boot_titles = {
        str(item["source_label"]).removesuffix(".docx")
        for item in [
            *boot_retrieval["trace"]["direct_document_matches"],
            *boot_retrieval["trace"]["navigation_document_matches"],
        ]
    }
    assert "修复引导" in boot_titles
    assert "修复系统" not in boot_titles
    assert "快速系统文件修复" not in boot_titles
