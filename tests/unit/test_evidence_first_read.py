from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from debug_agent_system.adapters.qa_supervisor import DebugAgentSystemQARuntime
from debug_agent_system.agents.read.evidence_answer import EvidenceAnswerComposer
from debug_agent_system.core.config import load_config
from debug_agent_system.core.contracts import SessionState
from debug_agent_system.runtime.system import DebugAgentSystem
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import kg_v2_source_revision


def _system(tmp_path: Path) -> DebugAgentSystem:
    config = load_config("config/debug_agent_system.yaml")
    config.session_store = tmp_path / "sessions"
    return DebugAgentSystem(config)


def test_required_info_keeps_document_answer_before_questions(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "工控机运行中蓝屏，BugCheck 0x00000139，转储提示关键数据结构损坏。"
            "当前缺少日志包、驱动上下文和内存测试结果。"
        ),
        "session": {"session_id": "evidence-first-required-info"},
    })

    assert out["status"] == "ask_info"
    assert out["required_data"]
    titles = [section["title"] for section in out["answer_sections"]]
    assert "根据资料可知" in titles
    assert "建议排查顺序" in titles
    assert "尚不能确认" in titles
    assert "需要补充的信息" in titles
    assert out["answer"].index("根据资料可知") < out["answer"].index("需要补充的信息")
    assert "0x00000139" in out["answer"]
    assert out["metadata"]["sufficiency"] == {
        "answerable": True,
        "diagnosable": True,
        "executable": False,
        "reasons": ["requires_information_or_safety_gate"],
    }
    coverage = out["metadata"]["answer_coverage"]
    assert coverage["complete"] is True
    assert coverage["included_fact_count"] == coverage["eligible_fact_count"]


def test_document_described_incident_uses_specific_source_and_plan(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "更换工控机后主程序报加载用户配置失败，检查发现 user.cfg.toml 为空或已损坏。",
        "session": {"session_id": "evidence-first-user-config"},
    })

    assert out["variant_id"] == "variant:family::efc23af9fc66:-user.cfg.toml-:1869ae54a49d"
    assert "user.cfg.toml为空或损坏" in out["answer"]
    assert "建议排查顺序" in out["answer"]
    assert "【来源：" in out["answer"]
    assert out["metadata"]["answer_coverage"]["eligible_fact_count"] > 0


def test_sag_rebuilds_original_chunk_from_current_hash_pinned_source(tmp_path: Path) -> None:
    system = _system(tmp_path)
    assert system.read_model.sag is not None
    assert system.read_model.sag.source_revision() == kg_v2_source_revision(
        system.config.knowledge.kg_v2_root
    )
    with sqlite3.connect(system.read_model.sag.sqlite_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM source_chunks WHERE chunk_id LIKE 'chunk:source:%'"
        ).fetchone()[0] > 0
        assert connection.execute(
            "SELECT count(*) FROM source_chunks WHERE chunk_id LIKE 'chunk:raw:%'"
        ).fetchone()[0] == 0
    system.read_model.search_variants("CPU温度过高，OCCT和AIDA64应如何确认封装温度？", limit=5)

    source_chunks = [
        chunk for chunk in system.read_model.last_retrieval["chunks"]
        if str(chunk.get("chunk_id") or "").startswith("chunk:source:")
        and "CPU温度过高问题处理指南.docx" in str(chunk.get("source_offsets") or "")
    ]
    assert source_chunks
    top = source_chunks[0]
    source_path = Path(__file__).parents[2] / top["source_offsets"][0]["source_path"]
    assert "OCCT" in top["text"]
    assert "CPU封装温度" in top["text"]
    assert len(top["content_hash"]) == 64
    assert top["source_file_hash"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert top["source_offsets"][0]["paragraph_start"] > 0
    assert top["source_offsets"][0]["paragraph_end"] >= top["source_offsets"][0]["paragraph_start"]
    assert top["variant_ids"]
    assert top["retrieval_paths"]


def test_unrelated_query_does_not_turn_weak_chunk_into_answer(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "完全未知的稀奇问题 foobarzzzz",
        "session": {"session_id": "evidence-first-no-match"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    # No candidate is a distinct retrieval outcome from finding a candidate
    # whose score is below the admission threshold.
    assert out["failure_type"] == "no_kg_v2_variant_match"
    assert out["metadata"]["sufficiency"]["answerable"] is False
    assert out["metadata"]["answer_coverage"]["eligible_fact_count"] == 0
    assert "建议排查顺序" not in out["answer"]


def test_relevant_source_chunk_answers_without_reliable_variant(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "Dism++软件怎么下载，x64、x86、ARM应该选哪个版本？",
        "session": {"session_id": "evidence-first-chunk-only"},
    })

    assert out["status"] == "step"
    assert out["variant_id"] == ""
    assert out["metadata"]["sufficiency"]["answerable"] is True
    assert out["metadata"]["sufficiency"]["diagnosable"] is False
    assert "一般电脑选择X64版本" in out["answer"]
    assert "建议排查顺序" not in out["answer"]
    assert any(
        str(chunk.get("chunk_id") or "").startswith("chunk:source:")
        for chunk in out["metadata"]["retrieval"]["supporting_chunks"]
    )


def test_named_knowledge_queries_do_not_lock_neighboring_fault_variants(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    cases = (
        (
            "软件卸载后有残留，如何彻底清理？",
            "卸载软件",
            "IP冲突",
        ),
        (
            "MemTest86 如何检测内存并判断 PASS/FAIL？",
            "MemTest86",
            "D盘",
        ),
        (
            "机械硬盘技术要求和容量规格是什么？",
            "机械硬盘技术要求",
            "扩展D盘",
        ),
    )

    for index, (query, expected, forbidden) in enumerate(cases):
        out = system.start({
            "query": query,
            "interactive": False,
            "session": {"session_id": f"knowledge-scope-{index}"},
        })
        assert out["status"] == "step"
        assert out["variant_id"] == ""
        assert out["metadata"]["query_scope"]["mode"] == "knowledge_lookup"
        assert out["metadata"]["sufficiency"]["answerable"] is True
        assert expected in out["answer"]
        assert forbidden not in out["answer"]
        assert out["required_data"] == []


def test_named_scope_without_index_coverage_does_not_substitute_variant(
    tmp_path: Path,
) -> None:
    out = _system(tmp_path).start({
        "query": "技嘉 B760 GAMING X DDR4 主板如何核对型号？",
        "interactive": False,
        "session": {"session_id": "knowledge-scope-uncovered-model"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    assert out["failure_type"] == "knowledge_scope_not_covered"
    assert out["metadata"]["answer_coverage"]["eligible_fact_count"] == 0
    assert "光源控制器" not in out["answer"]
    assert "主板型号" not in out["answer"]


def test_overloaded_fault_term_requires_compound_subject_compatibility(
    tmp_path: Path,
) -> None:
    out = _system(tmp_path).start({
        "query": "维修板误报如何排查？",
        "interactive": False,
        "session": {"session_id": "diagnostic-compound-subject-mismatch"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    assert out["failure_type"] == "candidate_subject_scope_mismatch"
    assert "轨道传感器" not in out["answer"]
    assert "已阻止" in out["answer"]


def test_license_scope_separates_software_and_hardware_dongles(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    software = system.start({
        "query": "软件许可证授权如何处理？",
        "interactive": False,
        "session": {"session_id": "knowledge-scope-soft-license"},
    })
    hardware = system.start({
        "query": "硬件加密狗授权如何处理？",
        "interactive": False,
        "session": {"session_id": "knowledge-scope-hard-dongle"},
    })

    assert software["status"] == "step"
    assert software["variant_id"] == ""
    assert "加密狗软狗授权步骤" in software["answer"]
    assert "localhost:1947" in software["answer"]
    assert hardware["status"] == "ask_info"
    assert hardware["variant_id"] == ""
    assert "软狗授权步骤" not in hardware["answer"]
    assert "localhost:1947" not in hardware["answer"]


def test_direct_document_match_uses_body_before_unlinked_variant_plan(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "开机后一直转圈无法进入系统",
        "interactive": False,
        "session": {"session_id": "evidence-first-direct-document"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    assert out["observability"]["lock_status"] == "document_answer_only"
    assert out["metadata"]["document_answer_mode"]["active"] is True
    assert out["metadata"]["sufficiency"] == {
        "answerable": True,
        "diagnosable": False,
        "executable": False,
        "reasons": ["variant_not_locked"],
    }
    assert "根据直接命中的资料可知" in out["answer"]
    assert "文档建议的处理路径" in out["answer"]
    assert "自动修复/高级启动选项" in out["answer"]
    assert "卸载最近安装的软件/驱动" in out["answer"]
    assert "制作启动盘" in out["answer"]
    assert "执行前必须备份" in out["answer"]
    assert "按F1进入系统并Esc不保存退出" not in out["answer"]
    assert out["answer"].index("自动修复/高级启动选项") < out["answer"].index("制作启动盘")
    assert "能否进入自动修复（WinRE）或安全模式" in out["required_data"][0]


def test_no_boot_query_uses_one_source_ordered_document_and_safety_gates(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "电脑不开机，应该怎么排查？",
        "interactive": False,
        "session": {"session_id": "evidence-first-no-boot-document"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    assert out["observability"]["lock_status"] == "document_answer_only"
    assert out["metadata"]["sufficiency"] == {
        "answerable": True,
        "diagnosable": False,
        "executable": False,
        "reasons": ["variant_not_locked"],
    }
    direct = out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    assert len(direct) == 1
    assert direct[0]["source_label"] == "电脑不开机排查"
    answer = out["answer"]
    assert answer.count("**重要安全须知**") == 1
    assert answer.index("阶段 0：快速检查") < answer.index("阶段 1：观察现象")
    assert answer.index("阶段 1：观察现象") < answer.index("阶段 5：特定现象深入排查")
    assert "按下电源键完全无反应" in answer
    assert "风扇转一下立即停止" in answer
    assert "有蜂鸣码" in answer
    assert "卡在主板Logo/BIOS界面" in answer
    assert "应急处理与上报" in answer
    assert "短接主板" not in answer
    assert "CLR_CMOS" not in answer
    assert "JBAT1" not in answer
    assert "拔掉所有硬盘" not in answer
    assert "重新涂抹硅脂" not in answer
    assert "SPC页面打不开" not in answer
    assert answer.count("本回答不展开拆装细节") <= 7
    assert "当前已完成哪些无需拆机的检查" in out["required_data"][0]
    coverage = out["metadata"]["answer_coverage"]
    assert coverage["complete"] is True
    assert coverage["included_fact_count"] == coverage["eligible_fact_count"]


def test_locked_document_answer_leads_with_compact_diagnostic_sequence(
    tmp_path: Path,
) -> None:
    out = _system(tmp_path).start({
        "query": "工控机不开机，怎么解决？",
        "interactive": False,
        "session": {"session_id": "evidence-first-locked-order"},
    })

    titles = [section["title"] for section in out["answer_sections"]]
    assert titles.index("建议排查顺序") < titles.index("文档建议的处理路径")
    assert out["answer"].count("本回答不展开拆装细节") <= 8


def test_locked_variant_keeps_complete_document_and_requires_branch_context(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "工控机不开机，怎么解决？",
        "interactive": False,
        "session": {"session_id": "evidence-first-locked-no-boot"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == "variant:1c1e5db082d5"
    assert out["observability"]["lock_status"] == "kg_v2_locked"
    assert out["metadata"]["sufficiency"] == {
        "answerable": True,
        "diagnosable": True,
        "executable": False,
        "reasons": ["requires_information_or_safety_gate"],
    }
    assert out["required_data"]
    assert "风扇/指示灯/Debug灯/屏幕显示/蜂鸣声" in out["required_data"][0]

    answer = out["answer"]
    headings = [f"情况 {number}：" for number in range(1, 12)]
    positions = [answer.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "检查外部供电" in answer
    assert "涉及机箱内部、部件拆装或硬件替换" in answer
    assert "涉及 BIOS/CMOS 修改或复位" in answer
    assert "涉及断开磁盘、修改启动分区或重建引导" in answer
    assert "涉及市电或带电测量" in answer
    for unsafe_detail in (
        "打开机箱侧板",
        "CLR_CMOS",
        "JBAT1",
        "重新涂抹 CPU 导热硅脂",
        "使用橡皮擦",
        "断开所有非系统硬盘",
        "bootrec /fixboot",
        "只插入一条内存",
        "逐一添加其他内存条",
        "移除独立显卡",
        "重新插拔显卡",
        "重新插拔硬盘",
        "扣下主板上的纽扣电池",
        "重新插拔所有连接线",
    ):
        assert unsafe_detail not in answer
    coverage = out["metadata"]["answer_coverage"]
    assert coverage["complete"] is True
    assert coverage["included_fact_count"] == coverage["eligible_fact_count"]


def test_locked_variant_observation_can_satisfy_branch_context(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "工控机不开机，风扇持续转动但屏幕无显示，怎么排查？",
        "interactive": False,
        "session": {"session_id": "evidence-first-locked-no-boot-observed"},
    })

    assert out["status"] == "step"
    assert out["variant_id"] == "variant:1c1e5db082d5"
    assert out["required_data"] == []
    assert out["metadata"]["sufficiency"]["diagnosable"] is True
    assert out["metadata"]["sufficiency"]["executable"] is True
    assert "情况 2：显示器无显示" in out["answer"]
    assert "打开机箱侧板" not in out["answer"]
    assert "bootrec /fixboot" not in out["answer"]


def test_safe_mode_navigation_expands_only_its_two_child_documents(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "如何进入安全模式",
        "interactive": False,
        "session": {"session_id": "evidence-first-safe-mode-navigation"},
    })

    assert out["status"] == "step"
    assert out["variant_id"] == ""
    assert out["observability"]["lock_status"] == "document_answer_only"
    assert out["required_data"] == []
    assert out["metadata"]["sufficiency"]["answerable"] is True
    assert out["metadata"]["sufficiency"]["diagnosable"] is False
    assert out["metadata"]["sufficiency"]["executable"] is False
    navigation = out["metadata"]["retrieval"]["trace"][
        "navigation_document_matches"
    ]
    assert [item["source_label"] for item in navigation] == [
        "可以进入系统.docx",
        "无法进入系统 (1).docx",
    ]
    assert "按住Shift" in out["answer"]
    assert "4 或 F4" in out["answer"]
    assert "5 或 F5" in out["answer"]
    assert "制作启动盘" not in out["answer"]
    assert "重装系统" not in out["answer"]
    assert "名称 | 所有者 | 修改时间" not in out["answer"]
    assert "尚不能确认" not in out["answer"]
    assert "  1. 按住Shift" in out["answer"]
    assert "- **无法进入系统**" in out["answer"]
    assert "  - Windows 10/11:" in out["answer"]


def test_windows_repair_navigation_weaves_selected_second_hop_documents(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "如何进行Windows系统/引导修复",
        "interactive": False,
        "session": {"session_id": "evidence-first-windows-repair-navigation"},
    })

    assert out["status"] == "step"
    assert out["variant_id"] == ""
    assert out["observability"]["lock_status"] == "document_answer_only"
    assert out["required_data"] == []
    navigation = out["metadata"]["retrieval"]["trace"][
        "navigation_document_matches"
    ]
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
    assert out["metadata"]["retrieval"]["trace"]["navigation_max_depth"] == 2
    assert "可以进系统 → 快速系统文件修复 → SFC" in out["answer"]
    assert "sfc /scannow" in out["answer"]
    assert "DISM /Online /Cleanup-Image /RestoreHealth" in out["answer"]
    assert "可以进系统 → 修复系统" in out["answer"]
    assert "选择“修复受损”" in out["answer"]
    assert "可以进系统 → 修复引导" in out["answer"]
    assert "选择“引导修复”" in out["answer"]
    assert "方法一：命令行修复" in out["answer"]
    assert "执行前必须确认系统盘" in out["answer"]
    assert "bootrec /scanos" not in out["answer"]
    assert "bootrec /rebuildbcd" not in out["answer"]
    guidance = next(
        section for section in out["answer_sections"]
        if section["section_type"] == "document_guidance"
    )
    second_hop_items = [
        item for item in guidance["items"]
        if int(item.get("navigation_depth") or 0) == 2
    ]
    assert second_hop_items
    assert all(len(item.get("navigation_document_path") or []) == 3 for item in second_hop_items)
    assert sum(
        len(item.get("media_refs") or [])
        for item in guidance["items"]
    ) >= 5
    coverage = out["metadata"]["answer_coverage"]
    assert coverage["complete"] is True
    assert coverage["included_fact_count"] == coverage["eligible_fact_count"]


def test_document_disk_commands_are_explained_but_not_rendered_for_execution(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "无法进入系统",
        "interactive": False,
        "session": {"session_id": "evidence-first-risk-guard"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    assert "涉及断开磁盘、修改启动分区或重建引导" in out["answer"]
    assert "执行前必须确认系统盘" in out["answer"]
    assert "list disk" not in out["answer"]
    assert "sel disk 0" not in out["answer"]
    assert "sel partition 1" not in out["answer"]
    assert "assign letter" not in out["answer"]
    assert "bcdboot" not in out["answer"].lower()


def test_short_reference_bm25_is_bounded_below_informative_body(tmp_path: Path) -> None:
    system = _system(tmp_path)
    assert system.read_model.sag is not None
    retrieval = system.read_model.sag.retrieve(
        "开机后一直转圈无法进入系统",
        chunk_limit=100,
    )
    reference = next(
        chunk for chunk in retrieval["chunks"]
        if chunk.get("source_label") == "制作PE和如何进入PE环境"
        and "参考：开机后一直转圈无法进去系统" in str(chunk.get("text") or "")
    )
    body = next(
        chunk for chunk in retrieval["chunks"]
        if "自动修复/高级启动选项" in str(chunk.get("text") or "")
        and str(chunk.get("chunk_id") or "").startswith("chunk:source:")
    )

    assert reference["raw_retrieval_score"] > 100
    assert reference["score_components"]["reference_only"] is True
    assert reference["retrieval_score"] < 6
    assert body["retrieval_score"] > reference["retrieval_score"]


def test_section_title_can_activate_orphan_document_answer(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "USB设备问题",
        "interactive": False,
        "session": {"session_id": "evidence-first-section-document"},
    })

    assert out["status"] == "ask_info"
    assert out["variant_id"] == ""
    assert out["observability"]["lock_status"] == "document_answer_only"
    assert out["metadata"]["document_answer_mode"]["documents"][0]["entry_object_type"] == "KnowledgeSection"
    assert "USB设备问题解决方案" in out["answer"]
    assert "Windows开启了自动省电导致供电不足" in out["answer"]
    assert "光源初始化失败" not in out["answer"]
    assert out["metadata"]["sufficiency"]["answerable"] is True
    assert out["metadata"]["sufficiency"]["diagnosable"] is False


def test_document_answer_renders_subheadings_and_materialized_images(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "电脑卡顿",
        "interactive": False,
        "session": {"session_id": "evidence-first-document-media"},
    })

    assert "强制关闭卡死程序" in out["answer"]
    assert "一键深度清理（每周1次）" in out["answer"]
    assert "自动清理" in out["answer"]
    assert "防卡顿黄金法则" in out["answer"]
    assert out["answer"].count("![源文档图片：") == 2
    media_refs = [
        media
        for section in out["answer_sections"]
        for item in section["items"]
        for media in item.get("media_refs") or []
    ]
    image_refs = [item for item in media_refs if item.get("media_kind") == "image"]
    assert len({item["content_hash"] for item in image_refs}) == 2
    assert all(Path(item["asset_path"]).is_file() for item in image_refs)


def test_usb_document_renders_each_numbered_solution_as_one_item(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "USB设备问题",
        "interactive": False,
        "session": {"session_id": "evidence-first-usb-solutions"},
    })

    guidance = next(
        section for section in out["answer_sections"]
        if section["section_type"] == "document_guidance"
    )
    headings = [str(item.get("source_heading") or "") for item in guidance["items"]]
    assert [heading[:3] for heading in headings] == [
        "方案一", "方案二", "方案三", "方案四", "方案五",
    ]
    assert out["answer"].count("- **方案") == 5
    for heading in ("方案一", "方案二", "方案三", "方案四", "方案五"):
        start = out["answer"].index(f"- **{heading}")
        following = out["answer"][start:]
        assert "\n  1. " in following.split("\n- **方案", 1)[0]


def test_web_document_weaves_reason_then_all_methods_in_source_order(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "网页打不开但微信/飞书能用",
        "interactive": False,
        "session": {"session_id": "evidence-first-web-weaving"},
    })

    known = next(
        section for section in out["answer_sections"]
        if section["section_type"] == "known"
    )
    guidance = next(
        section for section in out["answer_sections"]
        if section["section_type"] == "document_guidance"
    )
    assert any("原因：DNS故障" in item["text"] for item in known["items"])
    assert [str(item.get("source_heading") or "") for item in guidance["items"]] == [
        "方法一：", "方法二：", "方法三：",
    ]
    assert "Bashipconfig" not in out["answer"]
    assert "方法一：刷新 DNS 并重置 IP/Winsock" in out["answer"]
    assert "方法二：重置 VPN 连接状态" in out["answer"]
    assert "方法三：关闭设置脚本和代理服务器" in out["answer"]
    assert "`ipconfig /flushdns`" in out["answer"]
    assert out["answer"].index("方法一：") < out["answer"].index("方法二：") < out["answer"].index("方法三：")


def test_numbered_sop_headings_stay_in_source_order_without_derived_duplicates(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "板子到达进板口，皮带不转",
        "interactive": False,
        "session": {"session_id": "evidence-first-infeed-heading-order"},
    })

    known = next(
        section for section in out["answer_sections"]
        if section["section_type"] == "known"
    )
    guidance = next(
        section for section in out["answer_sections"]
        if section["section_type"] == "document_guidance"
    )
    headings = [str(item.get("source_heading") or "") for item in guidance["items"]]

    assert len(known["items"]) == 2
    assert known["items"][1]["source_heading"] == "进板失败SOP--20250521"
    assert "问题现象：板子到达进板口，皮带不转" in known["items"][1]["text"]
    assert headings == [
        "3.1 检查出板口板子是否已出板",
        "3.2 检查进板传感器",
        "3.3 检查皮带运转情况",
        "3.4 检查IO点位信号",
        "3.5 检查皮带电机：重新拔插电机线（在设备后部），如图8所示：",
    ]
    assert out["answer"].index("**3.1 ") < out["answer"].index("**3.2 ")
    assert out["answer"].index("**3.2 ") < out["answer"].index("**3.3 ")
    assert out["answer"].index("**3.3 ") < out["answer"].index("**3.4 ")
    assert out["answer"].index("**3.4 ") < out["answer"].index("**3.5 ")
    first_step = guidance["items"][0]["text"]
    assert first_step.startswith("3.1 检查出板口板子是否已出板")
    assert "问题现象：" not in first_step
    assert out["answer"].count("![源文档图片：") == 7
    assert any(
        item.get("reason")
        == "derived_section_summary_superseded_by_direct_source_chunks"
        for item in out["metadata"]["answer_coverage"]["excluded"]
    )


def test_camera_failure_answer_covers_direct_document_and_explains_images(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "检测界面出现拍照失败问题",
        "interactive": False,
        "session": {"session_id": "evidence-first-camera-document"},
    })

    for expected in (
        "相机SDK升级", "相机固件升级", "检查相机网口配置", "其余网卡配置", "电源模式",
        "系统日志检查", "更换M2网卡", "更换PCI接口网卡", "相机网线走线", "工控机机箱接地",
        "系统驱动清理", "老化验证",
    ):
        assert expected in out["answer"], expected
    assert out["answer"].count("![源文档图片：") == 47
    assert out["answer"].count("图片说明：") == 47
    assert "图片说明：Drawing" not in out["answer"]
    image_refs = [
        media
        for section in out["answer_sections"]
        for item in section["items"]
        for media in item.get("media_refs") or []
        if media.get("media_kind") == "image"
    ]
    assert all(str(item.get("context_label") or "").strip() for item in image_refs)
    assert len(image_refs) == len({
        str(item.get("content_hash") or "") for item in image_refs
    }) == 47
    assert len(out["answer"]) < 20000
    assert "需人工确认" in out["answer"]
    assert out["metadata"]["answer_coverage"]["eligible_chunk_count"] >= 20
    assert any(
        item.get("reason") == "truncated_summary_superseded_by_direct_source_chunks"
        for item in out["metadata"]["answer_coverage"]["excluded"]
    )


def test_semantic_source_chunks_recall_faq_table_and_section_content(tmp_path: Path) -> None:
    system = _system(tmp_path)
    probes = [
        ("CAD自动对齐失败后界面固定在CAD视图，应该怎么处理？", "若自动对齐仍无法将所有拼板对齐"),
        ("CPU温度过高时如何用OCCT和AIDA64确认封装温度？", "CPU封装温度"),
        ("Dism++制作镜像和修复系统分别是谁维护的？", "制作镜像/备份镜像 | 工程师乙"),
    ]

    for query, expected_text in probes:
        system.read_model.search_variants(query, limit=5)
        source_chunks = [
            chunk for chunk in system.read_model.last_retrieval["chunks"]
            if str(chunk.get("chunk_id") or "").startswith("chunk:source:")
        ]
        assert any(expected_text in str(chunk.get("text") or "") for chunk in source_chunks), query


def test_composer_merges_duplicate_sources_keeps_conflicts_and_rejects_unapproved(tmp_path: Path) -> None:
    system = _system(tmp_path)
    composer = EvidenceAnswerComposer(system.read_model)
    state = SessionState(session_id="composer-coverage", query="相机链路丢包")
    state.metadata["retrieval"] = {"supporting_chunks": [
        {
            "chunk_id": "chunk:a", "object_id": "section:a", "object_type": "KnowledgeSection",
            "source_label": "文档A", "text": "检查事件包是否丢失并确认是否重传。",
            "content_hash": "hash-a", "approved": True, "variant_ids": [],
            "matched_terms": ["事件包", "丢失", "重传"], "retrieval_score": 9.0,
        },
        {
            "chunk_id": "chunk:b", "object_id": "section:b", "object_type": "KnowledgeSection",
            "source_label": "文档B", "text": "检查事件包是否丢失并确认是否重传。",
            "content_hash": "hash-b", "approved": True, "variant_ids": [],
            "matched_terms": ["事件包", "丢失", "重传"], "retrieval_score": 8.5,
        },
        {
            "chunk_id": "chunk:c", "object_id": "section:c", "object_type": "KnowledgeSection",
            "source_label": "文档C", "text": "另一版本要求先检查网线换口时间点。",
            "content_hash": "hash-c", "approved": True, "variant_ids": [],
            "matched_terms": ["检查网线", "换口时间"], "retrieval_score": 8.0,
        },
        {
            "chunk_id": "chunk:bad", "object_id": "section:bad", "object_type": "KnowledgeSection",
            "source_label": "未批准文档", "text": "未经审核的结论。",
            "content_hash": "hash-bad", "approved": False, "variant_ids": [],
            "matched_terms": ["事件包", "重传"], "retrieval_score": 10.0,
        },
    ]}

    result = composer.compose(
        state=state,
        status="ask_info",
        base_answer="需要补充现场信息。",
        plan=None,
        required_data=["请提供相机日志。"],
    )

    known = next(section for section in result.sections if section.section_type == "known")
    assert len(known.items) == 2
    duplicate = next(item for item in known.items if "事件包" in item["text"])
    assert duplicate["chunk_ids"] == ["chunk:a", "chunk:b"]
    assert duplicate["sources"] == ["文档A", "文档B"]
    assert any("网线换口" in item["text"] for item in known.items)
    assert result.coverage["merged_fact_count"] == 1
    assert all("未经审核" not in item["text"] for item in known.items)


def test_high_risk_document_guidance_never_bypasses_confirmation(tmp_path: Path) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "光源初始化异常，准备退出软件并断电后重启。",
        "session": {"session_id": "evidence-first-safety"},
    })

    assert out["status"] == "ask_info"
    assert out["metadata"]["pending_confirmation_action_id"] == out["current_action_id"]
    assert "需人工确认后才可执行" in out["answer"]
    assert "未确认前仅保留为候选步骤" in out["answer"]
    assert out["metadata"]["sufficiency"]["executable"] is False


def test_internal_qa_adapter_preserves_answer_sections(tmp_path: Path) -> None:
    config = load_config("config/debug_agent_system.yaml")
    config.session_store = tmp_path / "sessions"
    runtime = DebugAgentSystemQARuntime()
    runtime.system = DebugAgentSystem(config)

    out = runtime.answer(
        "更换工控机后user.cfg.toml为空，主程序加载配置失败。",
        session={"session_id": "evidence-first-adapter"},
    )

    assert out["answer_sections"]
    assert any(section["section_type"] == "known" for section in out["answer_sections"])
    assert out["sources"]


def test_long_procedure_query_is_not_swallowed_by_short_related_title(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "新工控机或更换硬件后出现蓝屏、死机、重启或程序报错，"
            "怎样做兼容性和稳定性测试？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-procedure-operation-scope"},
    })

    assert "一、硬件兼容性验证" in out["answer"]
    assert "二、系统稳定性验证" in out["answer"]
    assert "三、实际压力测试" in out["answer"]
    assert "四、其他测试" in out["answer"]
    assert "五、理想结果" in out["answer"]
    assert "相机拍摄失败综合排查" not in out["answer"]
    assert all(
        item["source_label"] != "更换工控机"
        for item in out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    )


def test_named_component_prefers_whole_topic_document_over_incidental_section(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "工控机 USB 设备无法识别或工作异常时怎么办？",
        "interactive": False,
        "session": {"session_id": "evidence-first-named-component-document"},
    })

    assert out["answer"].count("- **方案") == 5
    assert "USB设备问题解决方案" in out["answer"]
    assert "键盘随机按键 / 无响应" not in out["answer"]
    direct = out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    assert [item["source_label"] for item in direct] == [
        "USB设备问题解决方案.docx"
    ]


def test_comparison_query_does_not_mix_generic_device_solution_document(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "AOI 设备出现 D 盘空间不足故障，"
            "0.2X 和 1.0 版本分别应该怎样处理？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-comparison-topic-scope"},
    })

    assert "0.2X版本的数据清理" in out["answer"]
    assert "1.0版本的数据清理" in out["answer"]
    assert "USB设备问题解决方案" not in out["answer"]
    assert out["answer"].count("方案一：静电干扰") == 0


def test_named_identifier_does_not_override_more_specific_document_topic(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "工控机键盘随机按键或无响应时，如何判断是键盘、"
            "转接链路、USB 口还是系统问题？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-specific-usb-topic"},
    })

    assert "直连法（最关键一步）" in out["answer"]
    assert "键盘交叉验证" in out["answer"]
    assert "USB设备问题解决方案" not in out["answer"]
    direct = out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    assert [item["source_label"] for item in direct] == [
        "键盘随机按键 _ 无响应"
    ]


def test_navigation_directory_selects_requested_child_for_non_comparison(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "电脑频繁蓝屏、死机或怀疑内存故障时，怎样用 "
            "Windows 内存诊断检查并查看结果？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-navigation-branch"},
    })

    assert "Windows内存检测方法" in out["answer"]
    assert "内存诊断的使用" in out["answer"]
    assert "内存诊断结果的查看" in out["answer"]
    assert "memtest86使用方法" not in out["answer"].lower()
    navigation = out["metadata"]["retrieval"]["trace"]["navigation_document_matches"]
    assert [item["source_label"] for item in navigation] == [
        "Windows内存检测方法.docx"
    ]
    assert navigation[0]["selection_reason"] == "query_guided_first_hop"


def test_named_command_prefers_matching_procedure_title_over_incidental_body_hit(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "Windows 报磁盘文件系统错误、磁盘异常或 SMART 状态异常时，"
            "怎样用 CHKDSK 检查修复？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-command-topic-title"},
    })

    assert "磁盘文件系统检测和修复" in out["answer"]
    assert "chkdsk" in out["answer"].lower()
    assert "wmic diskdrive get status" in out["answer"].lower()
    assert "阶段 0：快速检查" not in out["answer"]


def test_post_install_fault_anchors_causal_procedure_document(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "更换或加装内存后，电脑无法开机或频繁蓝屏，"
            "如何逐条测试内存和插槽？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-causal-procedure"},
    })

    assert "更换_加装内存教程" in out["answer"]
    assert "硬件隔离或交叉验证" in out["answer"]
    assert "人工确认" in out["answer"]
    assert "新硬件稳定性测试" not in out["answer"]


def test_tool_name_does_not_hide_descriptive_uninstall_document(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": "如何使用 DDU 彻底卸载显卡驱动？",
        "interactive": False,
        "session": {"session_id": "evidence-first-ddu-procedure"},
    })

    assert "彻底卸载显卡驱动" in out["answer"]
    assert "DDU" in out["answer"]
    assert "工控机异常(蓝屏&重启&死机）手册" not in out["answer"]


def test_generic_component_heading_cannot_override_named_tool_document(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "工控机高负载时死机、重启或计算报错，怀疑 CPU 不稳定时"
            "怎样用 Prime95 检测？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-prime95-domain"},
    })

    assert "P95使用文档" in out["answer"]
    assert "just strss testing" in out["answer"]
    assert "产品使用 - FAQ" not in out["answer"]
    assert "工控机主板接线规范" not in out["answer"]


def test_specification_role_outweighs_generic_replacement_operation(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "AOI 工控机加装机械硬盘后出现型号不合规或无法满足数据"
            "存储要求，应该更换为什么规格？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-disk-spec-domain"},
    })

    assert "机械硬盘技术要求" in out["answer"]
    assert "NAS级或企业级机械硬盘" in out["answer"]
    assert "更换_加装内存教程" not in out["answer"]
    assert "检测界面拍照失败" not in out["answer"]


def test_escalation_request_uses_source_heading_metadata(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "现场软件 Bug 或设备故障反馈后无法推进、责任人不明确时，"
            "应该怎样提交 JIRA 和升级处理？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-escalation-domain"},
    })

    assert "现场问题反馈流程" in out["answer"]
    assert "提交JIRA并附上所需资料" in out["answer"]
    assert "指定正确的负责人" in out["answer"]


def test_exact_error_heading_outweighs_generic_faq_category(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "主程序报“拍摄失败：set light source params failed”时，"
            "工控机主板串口线和端口配置应怎样检查？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-light-source-domain"},
    })

    assert "工控机主板接线规范" in out["answer"]
    assert "set light source params failed" in out["answer"]
    assert "主板接口示意图" in out["answer"]
    assert "产品使用 - FAQ" not in out["answer"]
    direct = out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    wiring = next(
        item for item in direct
        if item["source_label"].startswith("工控机主板接线规范")
    )
    assert wiring["expansion_scope"] == "document"


def test_resource_occupancy_query_prefers_lag_document_over_memory_test(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "电脑严重卡顿、程序无响应或 CPU/内存占用异常时，"
            "应该怎样定位并清理系统？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-resource-occupancy-domain"},
    })

    assert "电脑卡顿" in out["answer"]
    assert "强制关闭卡死程序" in out["answer"]
    assert "Windows内存检测方法" not in out["answer"]
    assert "memtest86" not in out["answer"].lower()


def test_internal_faq_hit_expands_section_not_entire_handbook(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "AOI 出现偶发报错、误报或现场故障但研发无法复现时，"
            "应该怎样采集整图、SPC、FOV 和日志？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-section-scope"},
    })

    assert "ctrlX图导出" in out["answer"]
    assert "AOI设备FOV碎图导出" in out["answer"]
    assert "误报导出" in out["answer"]
    assert "编程阶段，算法生成的框不准确" not in out["answer"]
    assert "特殊LED的极性检测" not in out["answer"]
    direct = out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    faq = next(
        item for item in direct
        if item["source_label"] == "产品使用 - FAQ"
    )
    assert faq["expansion_scope"] == "section"


def test_source_document_title_anchor_expands_complete_procedure(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "Windows 蓝屏、启动异常或系统文件损坏时，"
            "怎样使用 WinDbg 收集 DMP、配置环境并分析？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-source-title-scope"},
    })

    assert "如何进行MEMORY.DMP文件的分析" in out["answer"]
    assert "如何收集DMP文件" in out["answer"]
    assert "环境配置" in out["answer"]
    assert "进行DMP文件分析" in out["answer"]
    direct = out["metadata"]["retrieval"]["trace"]["direct_document_matches"]
    windbg = next(
        item for item in direct
        if item["source_label"] == "如何收集DMP文件"
    )
    assert windbg["expansion_scope"] == "document"


def test_comparison_lookup_keeps_descriptive_and_named_operands(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "内存不稳定时，Windows 内存诊断与 MemTest86 "
            "各适合什么情况，应该如何选择？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-comparison-operands"},
    })

    assert "内存诊断的使用" in out["answer"]
    assert "memtest86使用方法" in out["answer"].lower()
    assert "PASS" in out["answer"]


def test_compound_procedure_covers_each_requested_operation(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "显卡驱动持续异常时，如何先彻底卸载旧驱动，"
            "再重新安装显卡驱动？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-compound-procedure"},
    })

    assert "卸载并重装显卡驱动" in out["answer"]
    assert "安装显卡驱动" in out["answer"]


def test_recalled_navigation_entry_explains_unclosed_query_facet(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "Windows 启动异常、系统文件损坏或引导报错时，"
            "怎样用 Dism++ 备份并修复系统或引导？"
        ),
        "interactive": False,
        "session": {"session_id": "navigation-evidence-gap"},
    })

    coverage = out["metadata"]["answer_coverage"]
    assert coverage["query_facets_complete"] is False
    assert "operation:备份" in coverage["uncovered_query_facets"]
    assert "资料缺口" in out["answer"]
    assert "制作镜像/备份镜像" in out["answer"]
    assert "rId5" in out["answer"]
    assert "不补写缺失步骤" in out["answer"]
    gaps = coverage["navigation_evidence_gaps"]
    assert gaps[0]["facet_id"] == "operation:备份"
    assert gaps[0]["relationship_id"] == "rId5"
    assert gaps[0]["reason"] == "linked_child_not_indexed"


def test_authorization_update_is_not_routed_to_generic_data_export(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    out = system.start({
        "query": (
            "已授权的软件许可证即将到期，需要续期时，"
            "如何导出申请文件并应用更新授权文件？"
        ),
        "interactive": False,
        "session": {"session_id": "evidence-first-license-renewal"},
    })

    assert "软狗更新教程" in out["answer"]
    assert "C2V" in out["answer"]
    assert "更新/依附" in out["answer"]
    assert "SPC数据导出" not in out["answer"]
    assert "FOV碎图导出" not in out["answer"]
