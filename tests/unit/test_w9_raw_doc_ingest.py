from __future__ import annotations

from pathlib import Path
import hashlib
import tempfile
import zipfile

from debug_agent_system.agents.write import RawDocIngestAgent
from debug_agent_system.agents.write.pipeline import WriteSidePipeline


RAW_ROOT = Path("data/raw/aoi_debug_agent_sources")


def test_w9_classifies_cpu_overheat_doc_as_troubleshooting_topic():
    out = RawDocIngestAgent().inspect_document(RAW_ROOT / "CPU温度过高问题处理指南.docx")
    assert out["strategy"]["strategy_id"] == "troubleshooting_topic_doc"
    assert out["excluded_from_w9"] is False


def test_w9_classifies_usb_solution_doc_as_repair_playbook():
    out = RawDocIngestAgent().inspect_document(RAW_ROOT / "USB设备问题解决方案.docx")
    assert out["strategy"]["strategy_id"] == "repair_playbook_doc"


def test_w9_classifies_boot_manual_as_fault_manual_numbered():
    out = RawDocIngestAgent().inspect_document(RAW_ROOT / "工控机不开机手册.md")
    assert out["strategy"]["strategy_id"] == "fault_manual_numbered"


def test_w9_classifies_spec_and_procedure_docs():
    spec_out = RawDocIngestAgent().inspect_document(RAW_ROOT / "机械硬盘技术要求.docx")
    proc_out = RawDocIngestAgent().inspect_document(RAW_ROOT / "更换_加装内存教程.docx")
    assert spec_out["strategy"]["strategy_id"] == "spec_doc"
    assert proc_out["strategy"]["strategy_id"] == "procedure_doc"


def test_w9_classifies_navigation_docs_as_document_indexes():
    for name in (
        "Windows系统_引导修复.docx",
        "关闭快速启动.docx",
        "内存检测.docx",
        "可以进系统.docx",
        "如何进入安全模式.docx",
    ):
        out = RawDocIngestAgent().inspect_document(RAW_ROOT / name)
        assert out["strategy"]["strategy_id"] == "document_index_doc"


def test_w9_extracts_navigation_hyperlinks_with_stable_wiki_tokens():
    out = RawDocIngestAgent().build_section_cases(
        RAW_ROOT / "如何进入安全模式.docx"
    )

    assert [
        (item["link_text"], item["wiki_token"])
        for item in out["document_links"]
    ] == [
        ("可以进入系统", "MCwGwfrw6iWn1dkRvNsc2R8Xnid"),
        ("无法进入系统", "NUfmw7PyKiYWGLkEMhqcM9ndnzn"),
    ]


def test_w9_batch_checklist_excludes_sop_by_default():
    out = RawDocIngestAgent().build_root_checklist(RAW_ROOT)
    names = {item["name"] for item in out["documents"]}
    assert "异常处理 - 标准操作流程（SOP）.docx" not in names
    assert out["counts_by_strategy"]["troubleshooting_topic_doc"] >= 1
    assert "进板失败SOP--20250521.docx" not in names


def test_w9_builds_cpu_overheat_structured_sections():
    out = RawDocIngestAgent().build_structured_sections(RAW_ROOT / "CPU温度过高问题处理指南.docx")
    sections = out["structured_sections"]
    assert out["strategy"]["strategy_id"] == "troubleshooting_topic_doc"
    assert any(section["section_title"] == "文档目的" for section in sections)
    assert any(section["section_title"] == "诊断步骤" for section in sections)
    assert any(section["section_title"] == "解决方案" for section in sections)
    assert any(section["section_kind"] == "threshold_reference" for section in sections)
    assert any(section["section_kind"] == "diagnostic_actions" for section in sections)
    assert any(section["section_kind"] == "solution_playbook" for section in sections)


def test_w9_builds_cpu_overheat_section_cases():
    out = RawDocIngestAgent().build_section_cases(RAW_ROOT / "CPU温度过高问题处理指南.docx")
    cases = out["section_cases"]
    by_kind = {row["section_case_kind"]: row for row in cases}
    assert "threshold_reference" in by_kind
    assert "diagnostic_actions" in by_kind
    assert "solution_playbook" in by_kind
    assert "preventive_note" in by_kind
    assert "operator_caution" in by_kind
    diag = by_kind["diagnostic_actions"]
    assert diag["family_scope_candidates"] == ["CPU温度异常"]
    assert diag["variant_candidate"] == "CPU温度异常升高"
    assert diag["actions"]
    assert diag["required_info"]


def test_w9_stages_deterministic_unapproved_semantic_chunk_manifest():
    path = RAW_ROOT / "CPU温度过高问题处理指南.docx"
    first = RawDocIngestAgent().build_section_cases(path)["chunk_manifest"]
    second = RawDocIngestAgent().build_section_cases(path)["chunk_manifest"]

    assert first["schema_version"] == "kg_v2.source_chunk_manifest.v1"
    assert first["binding_status"] == "source_sections"
    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["source_file_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first["stats"]["chunk_count"] == len(first["chunks"]) > 0
    assert first["stats"]["source_aligned_section_count"] == len(
        RawDocIngestAgent().build_section_cases(path)["structured_sections"]
    )
    assert all(chunk["approved"] is False for chunk in first["chunks"])
    assert all(chunk["staging_status"] == "pending_review" for chunk in first["chunks"])


def test_w9_reads_all_xlsx_rows_into_source_aligned_chunks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "相机故障排查.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                "<sst xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>"
                "<si><t>问题现象</t></si><si><t>相机频繁断连</t></si>"
                "<si><t>排查步骤</t></si><si><t>查询网卡重置事件</t></si>"
                "</sst>",
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>"
                "<sheetData><row><c t='s'><v>0</v></c><c t='s'><v>1</v></c></row>"
                "<row><c t='s'><v>2</v></c><c t='s'><v>3</v></c></row>"
                "</sheetData></worksheet>",
            )
        result = RawDocIngestAgent().build_section_cases(path)
        text = "\n".join(
            chunk["text"] for chunk in result["chunk_manifest"]["chunks"]
        )
        assert "问题现象 | 相机频繁断连" in text
        assert "排查步骤 | 查询网卡重置事件" in text
        assert result["chunk_manifest"]["stats"]["source_parse_failure_count"] == 0


def test_w9_reads_every_pptx_slide_not_only_first_slide_preview() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "相机故障排查.pptx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "ppt/slides/slide1.xml",
                "<p:sld xmlns:p='p' xmlns:a='a'><a:t>问题现象</a:t>"
                "<a:t>相机频繁断连</a:t></p:sld>",
            )
            archive.writestr(
                "ppt/slides/slide2.xml",
                "<p:sld xmlns:p='p' xmlns:a='a'><a:t>排查步骤</a:t>"
                "<a:t>重新安装网卡驱动</a:t></p:sld>",
            )
        result = RawDocIngestAgent().build_section_cases(path)
        text = "\n".join(
            chunk["text"] for chunk in result["chunk_manifest"]["chunks"]
        )
        assert "相机频繁断连" in text
        assert "重新安装网卡驱动" in text
        assert result["chunk_manifest"]["stats"]["source_parse_failure_count"] == 0


def test_w9_routes_unrendered_pdf_to_review_only_instead_of_guessing_actions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "相机排查手册.pdf"
        path.write_bytes(b"%PDF-1.7\n/Type /Page\ncompressed-content")
        inspection = RawDocIngestAgent().inspect_document(path)
        result = RawDocIngestAgent().build_section_cases(path)

        assert inspection["structure_parse_status"] == "review_only"
        assert inspection["strategy"]["strategy_id"] == "unclassified_doc"
        assert result["strategy"]["kg_output_mode"] == "review_only"
        assert result["chunk_manifest"]["chunks"] == []


def test_w9_preserves_cpu_hardware_operation_step_hierarchy():
    out = RawDocIngestAgent().build_section_cases(RAW_ROOT / "CPU温度过高问题处理指南.docx")
    case = next(row for row in out["section_cases"] if row["section_title"] == "硬件操作（逐步执行）")

    assert case["actions"] == ["清洁除尘", "重新涂抹硅脂", "检查散热器固定", "优化风道"]
    assert len(case["procedure_steps"]) == 4
    first = case["procedure_steps"][0]
    assert first["step_order"] == 1
    assert first["label"] == "清洁除尘"
    assert first["instruction"] == "使用压缩空气或软毛刷清理"
    assert first["details"] == ["散热器鳍片", "风扇叶片", "机箱防尘网", "主板供电区域"]


def test_w9_builds_usb_solution_sections_and_cases():
    sections_out = RawDocIngestAgent().build_structured_sections(RAW_ROOT / "USB设备问题解决方案.docx")
    assert sections_out["strategy"]["strategy_id"] == "repair_playbook_doc"
    sections = sections_out["structured_sections"]
    assert any(section["section_title"].startswith("静电干扰") for section in sections)
    assert any(section["section_title"].startswith("卸载重装") for section in sections)
    cases_out = RawDocIngestAgent().build_section_cases(RAW_ROOT / "USB设备问题解决方案.docx")
    cases = cases_out["section_cases"]
    assert len(cases) >= 5
    first = cases[0]
    assert first["section_case_kind"] == "solution_playbook"
    assert first["family_scope_candidates"] == ["USB设备异常"]
    assert first["actions"]


def test_w9_extracts_unnumbered_procedure_actions_without_flattening_explanations():
    out = RawDocIngestAgent().build_section_cases(
        RAW_ROOT / "Windows内存检测方法.docx"
    )
    cases = out["section_cases"]
    assert cases
    steps = [step for case in cases for step in case.get("procedure_steps") or []]
    assert any(step["label"].startswith("鼠标右击") for step in steps)
    assert any(step["label"].startswith("按下F1") for step in steps)
    assert not any(step["label"].startswith("重要提醒") for step in steps)


def test_w9_camera_manual_keeps_numeric_hierarchy_and_ignores_model_numbers_as_headings():
    out = RawDocIngestAgent().build_structured_sections(
        RAW_ROOT / "检测界面出现拍照失败问题处理.docx"
    )
    sections = out["structured_sections"]
    assert not any(section["heading_number"] in {"252", "1020"} for section in sections)
    network = next(section for section in sections if section["heading_number"] == "2.2.1")
    assert network["section_kind"] == "diagnostic_actions"
    assert any(title.startswith("排查步骤") for title in network["path_titles"])


def test_w9_builds_boot_manual_sections_and_cases():
    sections_out = RawDocIngestAgent().build_structured_sections(RAW_ROOT / "工控机不开机手册.md")
    assert sections_out["strategy"]["strategy_id"] == "fault_manual_numbered"
    sections = sections_out["structured_sections"]
    assert any(section["section_title"] == "设备完全无通电反应" for section in sections)
    assert any(section["section_title"] == "显示器无显示（基础连接问题）" for section in sections)
    cases_out = RawDocIngestAgent().build_section_cases(RAW_ROOT / "工控机不开机手册.md")
    cases = cases_out["section_cases"]
    assert len(cases) >= 3
    first = cases[0]
    assert first["section_case_kind"] == "fault_case"
    assert first["family_scope_candidates"] == ["工控机无法开机"]
    assert first["actions"]
    assert first["support_notes"]


def test_non_sop_batch_semantic_hash_deduplicates_boot_manual_docx_and_markdown():
    agent = RawDocIngestAgent()
    docx = agent.build_section_cases(RAW_ROOT / "工控机不开机手册.docx")
    markdown = agent.build_section_cases(RAW_ROOT / "工控机不开机手册.md")
    assert WriteSidePipeline._semantic_document_hash(docx) == WriteSidePipeline._semantic_document_hash(markdown)


def test_semantic_duplicate_is_represented_as_evidence_alias_not_duplicate_semantics():
    canonical = {
        "document_id": "knowledge-document:boot-manual",
        "title": "工控机不开机手册.docx",
        "document_kind": "fault_manual_numbered",
        "source_path": "data/raw/aoi_debug_agent_sources/工控机不开机手册.docx",
        "content_hash": "a" * 64,
        "version": "",
        "owner": "",
        "approved": True,
        "source_kind": "raw_doc",
    }
    bundle = WriteSidePipeline._semantic_duplicate_evidence_bundle(
        duplicate_path="data/raw/aoi_debug_agent_sources/工控机不开机手册.md",
        semantic_hash="b" * 64,
        canonical_path=canonical["source_path"],
        canonical_document=canonical,
    )
    assert bundle["schema_valid"] is True, bundle["schema_issues"]
    assert set(bundle["objects"]) == {"KnowledgeDocument", "EvidenceItem"}
    assert len(bundle["objects"]["EvidenceItem"]) == 1
    assert bundle["objects"]["EvidenceItem"][0]["payload_ref"].endswith("工控机不开机手册.md")
    assert bundle["relations"] == [{
        "from": bundle["objects"]["EvidenceItem"][0]["evidence_id"],
        "to": canonical["document_id"],
        "relation": "evidences",
    }]


def test_w9_preserves_numbered_blue_screen_manual_hierarchy_without_component_variants():
    path = RAW_ROOT / "工控机异常(蓝屏&重启&死机）手册.docx"
    sections_out = RawDocIngestAgent().build_structured_sections(path)
    sections = sections_out["structured_sections"]
    fault_titles = [
        section["section_title"]
        for section in sections
        if section["section_kind"] == "fault_case"
    ]
    assert fault_titles == ["蓝屏", "无限蓝屏重启循环", "重启", "死机（完全卡死）"]
    component_titles = [
        section["section_title"]
        for section in sections
        if section["section_kind"] == "cause_support"
    ]
    assert component_titles == ["CPU", "内存", "硬盘", "显卡", "网卡", "电源", "软件、驱动与系统配置冲突"]

    cases = RawDocIngestAgent().build_section_cases(path)["section_cases"]
    fault_cases = [case for case in cases if case["section_case_kind"] == "fault_case"]
    assert [case["family_scope_candidates"] for case in fault_cases] == [
        ["工控机蓝屏"],
        ["工控机蓝屏"],
        ["工控机异常重启"],
        ["工控机死机"],
    ]
    support_cases = [case for case in cases if case["section_case_kind"] == "component_support"]
    assert len(support_cases) == 7
    assert all(case["fault_mapping_allowed"] is False for case in support_cases)
    assert all(not case["variant_candidate"] for case in support_cases)


def test_w9_build_not_entered_docs_excludes_manifest_entered_sources():
    with tempfile.TemporaryDirectory() as tmp:
        out = RawDocIngestAgent().build_not_entered_docs(
            RAW_ROOT,
            out_root=Path(tmp) / "w9_not_entered",
        )
        names = {item["name"] for item in out["documents"]}
        assert "异常处理 - 标准操作流程（SOP）.docx" not in names
        assert "工控机不开机手册.md" not in names
        assert "CPU温度过高问题处理指南.docx" in names
        assert out["summary"]["doc_count"] > 0


def test_w9_sop_catalog_is_split_into_section_local_atomic_fault_cases():
    path = RAW_ROOT / "异常处理 - 标准操作流程（SOP）.docx"
    out = RawDocIngestAgent().build_section_cases(path)
    atomic_cases = [
        item
        for item in out["section_cases"]
        if item.get("fault_mapping_allowed")
    ]

    assert out["strategy"]["strategy_id"] == "sop_fault_catalog_doc"
    assert len(atomic_cases) >= 40
    assert len({item["family_scope_candidates"][0] for item in atomic_cases}) >= 8
    assert all(item["variant_candidate"] != path.stem for item in atomic_cases)
    assert all(item["procedure_steps"] for item in atomic_cases)
    assert all(item["atomic_case_id"] for item in atomic_cases)

    two_d_cases = [
        item for item in atomic_cases
        if "2D" in item.get("variant_candidate", "")
    ]
    assert len(two_d_cases) >= 2
    assert len({item["variant_candidate"] for item in two_d_cases}) == len(two_d_cases)
