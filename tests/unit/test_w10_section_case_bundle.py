from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.agents.write import RawDocIngestAgent, SectionCaseBundleAgent
from debug_agent_system.adapters.cli import _refine_and_gate_v2_bundle


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_w10_cpu_section_cases_to_bundle_schema_valid():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    assert out["schema_valid"] is True
    assert [item["label"] for item in out["objects"]["FaultFamily"]] == ["CPU温度异常"]
    assert [item["label"] for item in out["objects"]["FaultVariant"]] == ["CPU温度异常升高"]
    assert out["report"]["document_count"] == 1
    assert out["report"]["section_count"] == 16
    assert out["report"]["procedure_step_count"] == 15
    assert out["report"]["trace_count"] == 0
    assert out["report"]["source_case_count"] == 0


def test_w10_binds_chunk_manifest_to_draft_knowledge_sections():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    manifest = out["chunk_manifest"]
    document_id = out["objects"]["KnowledgeDocument"][0]["document_id"]
    section_ids = {item["section_id"] for item in out["objects"]["KnowledgeSection"]}

    assert manifest["binding_status"] == "draft_kg_sections"
    assert manifest["document_id"] == document_id
    assert manifest["source_manifest_id"] == payload["chunk_manifest"]["manifest_id"]
    assert manifest["stats"]["bound_section_count"] == len(section_ids)
    assert manifest["stats"]["unbound_chunk_count"] == 0
    assert all(chunk["document_id"] == document_id for chunk in manifest["chunks"])
    assert all(set(chunk["section_ids"]).issubset(section_ids) for chunk in manifest["chunks"])
    assert all(chunk["approved"] is False for chunk in manifest["chunks"])
    assert all(chunk["staged_chunk_id"] != chunk["chunk_id"] for chunk in manifest["chunks"])


def test_w10_keeps_raw_document_links_on_the_document_layer():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/如何进入安全模式.docx"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    document = out["objects"]["KnowledgeDocument"][0]

    assert out["schema_valid"] is True
    assert [item["link_text"] for item in document["source_links"]] == [
        "可以进入系统",
        "无法进入系统",
    ]
    assert all(item["wiki_token"] for item in document["source_links"])


def test_w10_cpu_hardware_steps_are_parent_actions_with_nested_details():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    actions = out["objects"]["DiagnosticAction"]
    clean = next(item for item in actions if item["label"] == "清洁除尘")

    assert clean["procedure_instruction"] == "使用压缩空气或软毛刷清理"
    assert clean["procedure_details"] == ["散热器鳍片", "风扇叶片", "机箱防尘网", "主板供电区域"]
    assert not any(item["label"] in {"散热器鳍片", "风扇叶片", "机箱防尘网", "主板供电区域"} for item in actions)


def test_w10_usb_section_cases_to_bundle_schema_valid():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/USB设备问题解决方案.docx"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    assert out["schema_valid"] is True
    assert out["report"]["document_count"] == 1
    assert out["report"]["procedure_step_count"] >= 1
    assert out["report"]["variant_count"] == 0


def test_w10_boot_manual_section_cases_to_bundle_schema_valid():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/工控机不开机手册.md"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    assert out["schema_valid"] is True
    assert out["report"]["variant_count"] >= 3
    assert out["report"]["required_info_count"] >= 1


def test_w10_numbered_blue_screen_manual_keeps_four_distinct_variants():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/工控机异常(蓝屏&重启&死机）手册.docx"
    )
    out = SectionCaseBundleAgent().build_bundle(payload)
    assert out["schema_valid"] is True
    labels = [item["label"] for item in out["objects"]["FaultVariant"]]
    assert labels == ["蓝屏", "无限蓝屏重启循环", "重启", "死机（完全卡死）"]
    assert len({item["variant_id"] for item in out["objects"]["FaultVariant"]}) == 4


def test_w10_non_sop_fault_documents_map_to_specific_families():
    cases = {
        "检测界面出现拍照失败问题处理.docx": "相机拍摄失败",
        "无法上网_显示_无Internet_.docx": "网络连接异常",
        "键盘随机按键 _ 无响应.docx": "键盘输入异常",
    }
    for name, expected_family in cases.items():
        payload = RawDocIngestAgent().build_section_cases(
            f"data/raw/aoi_debug_agent_sources/{name}"
        )
        out = SectionCaseBundleAgent().build_bundle(payload)
        assert out["schema_valid"] is True
        assert [item["label"] for item in out["objects"]["FaultFamily"]] == [expected_family]
        assert out["report"]["action_count"] > 0


def test_w10_empty_section_cases_is_invalid():
    out = SectionCaseBundleAgent().build_bundle({
        "name": "empty-doc",
        "path": "empty",
        "strategy": {"strategy_id": "unclassified_doc"},
        "section_cases": [],
    })
    assert out["schema_valid"] is False
    assert out["schema_issues"] == ["empty_section_cases"]


def test_w10_empty_bundle_stays_invalid_after_w3_w4():
    raw = SectionCaseBundleAgent().build_bundle({
        "name": "empty-doc",
        "path": "empty",
        "strategy": {"strategy_id": "unclassified_doc"},
        "section_cases": [],
    })
    out = _refine_and_gate_v2_bundle(raw)
    assert out["schema_valid"] is False
    assert "empty_v2_document_bundle" in out["schema_issues"]
    assert out["quality_gate"]["passed"] is False


def test_w10_bundle_enters_w3_refinement_and_w4_gate():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx"
    )
    raw = SectionCaseBundleAgent().build_bundle(payload)
    out = _refine_and_gate_v2_bundle(raw)
    assert out["type"] == "W3NormalizedKGV2Bundle"
    assert out["schema_valid"] is True
    assert out["w3_refinement"]["agent_id"] == "W3"
    assert out["quality_gate"]["observability"]["agent_id"] == "W4"
    assert out["quality_gate"]["passed"] is True
    assert "kg_v2_ambiguous_family_scope" not in out["quality_gate"]["issues"]
    assert "kg_v2_high_cost_action_requires_human" in out["quality_gate"]["issues"]


def test_w10_non_fault_playbook_is_not_admitted_to_fault_graph():
    payload = _load("data/results/w9_usb_device_20260710/section_cases.json")
    raw = SectionCaseBundleAgent().build_bundle(payload)
    out = _refine_and_gate_v2_bundle(raw)
    assert out["quality_gate"]["passed"] is False
    assert "kg_v2_non_fault_output_mode" in out["quality_gate"]["issues"]


def test_w10_sop_atomic_bundles_are_independent_mapping_only_drafts():
    payload = RawDocIngestAgent().build_section_cases(
        "data/raw/aoi_debug_agent_sources/异常处理 - 标准操作流程（SOP）.docx"
    )
    bundles = SectionCaseBundleAgent().build_atomic_case_bundles(payload)

    assert len(bundles) >= 40
    assert all(item["schema_valid"] for item in bundles)
    assert len({item["bundle_id"] for item in bundles}) == len(bundles)
    assert len({item["atomic_case"]["variant_id"] for item in bundles}) == len(bundles)
    for bundle in bundles:
        objects = bundle["objects"]
        assert len(objects["FaultFamily"]) == 1
        assert len(objects["FaultVariant"]) == 1
        assert len(objects["DiagnosticAction"]) >= 1
        assert len(objects["SourceCase"]) == 1
        assert objects["KnowledgeDocument"] == []
        assert objects["KnowledgeSection"] == []
        assert objects["ProcedureStep"] == []
        assert len(objects["EvidenceItem"]) == 1
        assert all(item.get("evidence_ids") for item in objects["DiagnosticAction"])
        assert any(rel["relation"] == "supports" for rel in bundle["relations"])
