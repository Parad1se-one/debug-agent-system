from __future__ import annotations

from debug_agent_system.agents.write import ChatCollectAgent, KnowledgeExtractionAgent
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.compat import (
    _canonicalize_family_label,
    _canonicalize_variant_label,
    _owner_context,
    build_case_understanding_card_from_semantics,
    build_v2_bundle_from_candidate_draft,
)


def _episode() -> dict:
    return {
        "episode_id": "native-v2-001",
        "thread_id": "native-v2-thread",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "sender": "fae", "text": "现场更换工控机后打开主程序报警加载用户配置失败，怀疑 user.cfg.toml 是空白文件。"}
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "sender": "dev", "text": "检查 user.cfg.toml 是否为空。"},
            {"message_id": "m3", "sender": "dev", "text": "检查 conf 目录备份。"},
            {"message_id": "m4", "sender": "dev", "text": "回填备份配置并重启验证。"},
        ],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3", "m4"],
        "source_offsets": [{"message_id": "m1", "index": 0}],
        "attachments": [],
        "extracted": {
            "symptom_raw": "更换工控机后加载用户配置失败。",
            "debug_actions": ["检查 user.cfg.toml 是否为空", "检查 conf 目录备份", "回填备份配置并重启验证"],
            "conclusion": ""
        },
    }


def test_w2_native_v2_mode_emits_case_understanding_and_draft():
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2")
    result = extractor.extract(_episode())
    assert result["w2_mode"] == "native_v2"
    assert result["case_understanding_card"]["schema_version"] == "kg_v2.case_understanding.v1"
    assert result["candidate_draft_v2"]["schema_version"] == "kg_v2.candidate_draft.v1"
    assert result["candidate_draft_v2_schema_valid"] is True
    assert result["candidate_draft_v2_normalized_bundle"]["schema_valid"] is True
    case = result["candidate_draft_v2"]["split_cases"][0]
    assert case["family"]["label"] == "用户配置加载失败"
    assert case["variant"]["label"] == "更换工控机后 user.cfg.toml 为空导致加载用户配置失败"
    assert case["required_info"] == []


def test_w2_compare_mode_keeps_legacy_and_native_payloads():
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="compare")
    result = extractor.extract(_episode())
    assert result["w2_mode"] == "compare"
    assert result["schema_valid"] is True
    assert result["case_understanding_card_schema_valid"] is True
    assert result["candidate_draft_v2_bundle_schema_valid"] is True
    assert result["observability"]["context_evidence_policy"] == "current_episode_only.v1"
    assert result["candidate_draft_v2_normalized_bundle"]["extraction_metadata"]["context_evidence_policy"] == "current_episode_only.v1"


def test_w2_marks_w7_promoted_messages_as_audited_case_evidence():
    episode = _episode()
    episode["case_evidence_messages"] = [
        {"message_id": "m5", "text": "后续分析确认是软件 BUG，并创建 Jira。"}
    ]
    episode["evidence_message_ids"].append("m5")

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)

    assert result["observability"]["context_evidence_policy"] == "w7_promoted_case_evidence.v1"
    assert result["candidate_draft_v2_normalized_bundle"]["extraction_metadata"]["context_evidence_policy"] == "w7_promoted_case_evidence.v1"


def test_w2_owner_context_requires_confirmed_w7_assignment():
    episode = _episode()
    episode["extracted"]["attribution"] = {
        "role_assignments": [
            {
                "name": "现场张工",
                "status": "inferred",
                "organization_roles": [],
                "episode_roles": ["reporter", "field_report_author"],
                "confidence": 0.95,
            },
            {
                "name": "工程师丑",
                "status": "confirmed",
                "organization_roles": ["rd_engineer"],
                "episode_roles": ["assignee", "investigator"],
                "confidence": 0.8,
            },
        ]
    }

    assert _owner_context(episode) == "工程师丑"

    episode["extracted"]["attribution"]["role_assignments"] = [
        {
            "name": "现场张工",
            "status": "inferred",
            "organization_roles": [],
            "episode_roles": ["assignee"],
            "confidence": 0.95,
        }
    ]
    assert _owner_context(episode) == ""

    episode["extracted"]["attribution"]["role_assignments"] = [
        {
            "name": "工程师癸",
            "status": "confirmed",
            "organization_roles": ["delivery_pm"],
            "episode_roles": ["investigator"],
            "confidence": 0.9,
        }
    ]
    assert _owner_context(episode) == ""


def test_v2_bundle_keeps_multiple_chinese_split_cases_distinct():
    draft = {
        "source_candidate_id": "candidate:multi",
        "source_episode_id": "chat:episode:1",
        "source_thread_id": "chat:thread:1",
        "split_cases": [
            {
                "case_ref": "case_1",
                "source_case": {"title": "调试卡顿", "summary": "智能调整等待时间长", "approved": False},
                "family": {"label": "程序运行卡顿", "summary": "程序响应慢", "category": "系统与软件异常", "subsystem": "主程序/运行性能"},
                "variant": {"label": "智能调整等待时间长", "summary": "等待5秒"},
                "actions": [{"label": "记录智能调整耗时", "summary": "记录耗时", "action_role": "collect", "step_order": 1}],
                "outcomes": [], "required_info": [], "evidence": [], "trace": {},
            },
            {
                "case_ref": "case_2",
                "source_case": {"title": "拍摄失败", "summary": "复判时拍摄失败", "approved": False},
                "family": {"label": "相机拍摄失败", "summary": "相机不拍照", "category": "硬件与运控", "subsystem": "相机/采集链路"},
                "variant": {"label": "复判卡顿后拍摄失败", "summary": "复判时弹出拍摄失败"},
                "actions": [{"label": "检查图像采集日志", "summary": "检查日志", "action_role": "inspect", "step_order": 1}],
                "outcomes": [], "required_info": [], "evidence": [], "trace": {},
            },
        ],
    }

    bundle = build_v2_bundle_from_candidate_draft(draft)

    assert len(bundle["objects"]["FaultFamily"]) == 2
    assert len(bundle["objects"]["FaultVariant"]) == 2
    assert len({item["family_id"] for item in bundle["objects"]["FaultFamily"]}) == 2
    assert len({item["variant_id"] for item in bundle["objects"]["FaultVariant"]}) == 2


def test_w1_w2_preserve_four_faults_from_one_nested_field_report():
    report = (
        "升级1.3.3版本现场问题汇总 "
        "一、软件功能异常问题 "
        "1. 操作响应延迟，在处理误报时，点击智能调整或编程优化后，应用等待时间较长，约5–10秒。 "
        "2. 调试误报闪退，界面出现卡顿加载转圈，约10秒后软件直接闪退。 "
        "3. 卡顿后出现拍摄失败，正常复判时零件复判卡顿2–3秒，随后弹出拍摄失败报错。 "
        "二、设备硬件异常问题 远轨中间段轨道宽度异常，导致板卡卡滞、无法正常出板。"
        "临时将轨道宽度小幅收窄后可正常测试。以上信息请各位领导知悉！"
    )
    collector = ChatCollectAgent()
    messages = collector.normalize_messages([
        {"message_id": "m-multi", "thread_id": "t-multi", "sender": "fae", "content": report}
    ])
    episodes = collector.aggregate_threads(messages)[0]["episodes"]
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2")

    drafts = [extractor.extract(episode)["candidate_draft_v2"] for episode in episodes]
    cases = [draft["split_cases"][0] for draft in drafts]

    assert len(episodes) == 4
    assert len(cases) == 4
    assert [case["family"]["label"] for case in cases] == [
        "程序运行卡顿",
        "软件卡死无响应",
        "相机拍摄失败",
        "出板失败",
    ]
    assert [case["variant"]["label"] for case in cases] == [
        "智能调整或编程优化响应延迟",
        "调试误报时界面卡顿后软件闪退",
        "复判卡顿后出现拍摄失败",
        "远轨宽度异常导致板卡卡滞无法出板",
    ]
    assert all(case["required_info"] == [] for case in cases)
    assert all(draft["schema_valid"] for draft in drafts)


def test_w2_does_not_extract_actions_or_outcomes_from_untrusted_case_context():
    episode = {
        "episode_id": "native-v2-context-boundary",
        "thread_id": "native-v2-context-thread",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m-current", "text": "现场相机偶发拍摄失败，当前还没有完成排查。"}
        ],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m-current"],
        "source_offsets": [{"message_id": "m-current", "index": 0}],
        "attachments": [],
        "case_context_messages": [
            {"message_id": "m-unrelated", "text": "另一个现场更换内存条后蓝屏未再出现，问题已解决。"}
        ],
        "extracted": {
            "symptom_raw": "现场相机偶发拍摄失败。",
            "debug_actions": [],
            "conclusion": "",
        },
    }
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2")

    semantics = extractor.extract_semantics(episode)
    result = extractor.extract(episode)

    assert "更换内存条" not in semantics["semantic_text"]
    assert semantics["debug_actions"] == []
    assert semantics["conclusion"] == ""
    assert semantics["untrusted_case_context_message_count"] == 1
    bundle = result["candidate_draft_v2_normalized_bundle"]
    assert all("更换内存条" not in str(item) for item in bundle["objects"]["DiagnosticAction"])
    assert all("蓝屏未再出现" not in str(item) for item in bundle["objects"]["ActionOutcome"])
    evidence = bundle["objects"]["EvidenceItem"]
    assert evidence
    assert evidence[0]["external_id"] == "m-current"
    assert "现场相机偶发拍摄失败" in evidence[0]["summary"]
    assert all("另一个现场" not in item["summary"] for item in evidence)


def test_w2_native_v2_preserves_fault_only_case_without_inventing_actions():
    episode = {
        "episode_id": "native-v2-weak-001",
        "thread_id": "native-v2-weak-thread",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "sender": "fae", "text": "大量的器件3D成像异常误报3D共面算法了。"}
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "sender": "dev", "text": "我懂，但是没有图，没有对应发生问题时的日志，这个没法排查的"},
            {"message_id": "m3", "sender": "dev", "text": "另外，运控卡供应商今天也提到我们这台机器的操作系统是iot版本的windows，这个也不太符合我的认知，也能顺便确认下吗"},
        ],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [{"message_id": "m1", "index": 0}],
        "attachments": [],
        "extracted": {
            "symptom_raw": "大量器件3D成像异常误报。",
            "debug_actions": [],
            "conclusion": "",
        },
    }
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2")
    result = extractor.extract(episode)
    assert result["candidate_draft_v2_schema_valid"] is True
    assert result["production_schema_valid"] is True
    assert result["schema_valid"] is True
    case = result["candidate_draft_v2"]["split_cases"][0]
    assert case["candidate_scope"] == "fault_only"
    assert case["actions"] == []
    assert case["outcomes"] == []


def test_w2_native_v2_collapses_camera_failure_over_perf_family_when_network_change_is_explicit():
    episode = {
        "episode_id": "native-v2-camera-001",
        "thread_id": "native-v2-camera-thread",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "text": "更换相机网线插口后出现拍摄失败，ping相机网络请求超时频繁。"}
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "text": "重新插拔了网卡，ping了相机网络，请求超时频繁。"},
            {"message_id": "m3", "text": "检查相机网口角色与网络配置。"},
        ],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [{"message_id": "m1", "index": 0}],
        "attachments": [],
        "extracted": {
            "symptom_raw": "更换相机网线插口后出现拍摄失败。",
            "debug_actions": ["检查相机网口角色与网络配置", "重新插拔网卡验证"],
            "conclusion": "",
        },
    }
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2")
    result = extractor.extract(episode)
    draft = result["candidate_draft_v2"]
    assert draft["schema_valid"] is True
    assert len(draft["split_cases"]) == 1
    assert draft["split_cases"][0]["family"]["label"] == "相机拍摄失败"
    bundle_families = [item["label"] for item in result["candidate_draft_v2_normalized_bundle"]["objects"]["FaultFamily"]]
    assert bundle_families == ["相机拍摄失败"]


def test_w2_native_v2_treats_reboot_as_condition_when_terminal_fault_is_cannot_capture():
    episode = {
        "episode_id": "native-v2-camera-after-reboot",
        "thread_id": "native-v2-camera-after-reboot-thread",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "text": "设备断电重启后无法拍照。"}
        ],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {"symptom_raw": "设备断电重启后无法拍照。", "debug_actions": [], "conclusion": ""},
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)

    assert result["candidate_draft_v2"]["split_cases"][0]["family"]["label"] == "相机拍摄失败"


def test_w2_sentence_roles_do_not_promote_fault_observation_to_action():
    episode = {
        "episode_id": "native-v2-role-symptom",
        "thread_id": "native-v2-role-thread",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "text": "测试这种产品的时候调试误报会有卡顿现象。"}
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m1", "text": "测试这种产品的时候调试误报会有卡顿现象。"}
        ],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {"symptom_raw": "调试误报时软件卡顿", "debug_actions": [], "conclusion": ""},
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)

    assert result["debug_actions"] == []
    assert any(item["role"] == "symptom" for item in result["sentence_roles"])
    assert result["candidate_draft_v2"]["split_cases"][0]["candidate_scope"] == "fault_only"


def test_w2_outcome_requires_a_linkable_action_and_message_evidence():
    episode = {
        "episode_id": "native-v2-role-outcome",
        "thread_id": "native-v2-role-thread",
        "completeness": "complete",
        "fault_description_messages": [{"message_id": "m1", "text": "工控机频繁蓝屏。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "更换内存条后继续观察。"}],
        "resolution_messages": [{"message_id": "m3", "text": "更换内存条后至今未再出现蓝屏。"}],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {"symptom_raw": "工控机频繁蓝屏", "debug_actions": [], "conclusion": "更换内存条后至今未再出现蓝屏"},
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)

    assert any(item["role"] == "observed_outcome" and item["message_id"] == "m3" for item in result["sentence_roles"])
    assert result["diagnostic_outcomes"]
    assert all(item["target_check_id"] for item in result["diagnostic_outcomes"])
    assert all(item["evidence_message_ids"] for item in result["diagnostic_outcomes"])


def test_canonicalize_variant_label_shortens_camera_network_timeout_case():
    raw = "2.重新插拔了网卡，ping了相机网络，请求超时频繁，用万用表排查相机接口处电压，无漏电情况"
    assert _canonicalize_variant_label(raw, raw) == "相机网络异常导致拍摄失败"


def test_canonicalize_variant_label_shortens_warp_board_false_positive_case():
    raw = "我们可以尝试重建，但是风险我上面也提了（误差和误报），因为这个现场的弯板有点离谱，有5mm的下弯"
    assert _canonicalize_variant_label(raw, raw) == "弯板导致误报风险增加"


def test_legacy_leaf_enrichment_cannot_override_current_episode_outcome_state():
    semantics = {
        "candidate_id": "chatcand:test-llm",
        "source_episode_id": "ep-llm-001",
        "source_thread_id": "thread-llm-001",
        "label": "原始deterministic标签",
        "category": "系统与软件异常",
        "symptom_raw": "软件启动后拍照失败，现场让检查相机配置。",
        "conclusion": "更换网口后恢复。",
        "semantic_text": "软件启动后拍照失败，现场让检查相机配置，更换网口后恢复。",
        "evidence_ids": ["m1", "m2", "m3"],
        "episode": {
            "episode_id": "ep-llm-001",
            "thread_id": "thread-llm-001",
            "fault_description_messages": [{"message_id": "m1", "text": "软件启动后拍照失败"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查相机IP并更换网口验证"}],
            "resolution_messages": [{"message_id": "m3", "text": "更换网口后恢复"}],
            "noise_messages": [],
            "extracted": {"debug_actions": ["检查相机"], "conclusion": "更换网口后恢复"},
        },
    }
    legacy_candidate = {
        "case_variant_candidate": {
            "label": "相机网口切换后拍摄失败",
            "category": "系统与软件异常",
            "subsystem": "相机/采集链路",
            "scenario": "启动后拍摄失败",
            "canonical_error_id": "err:camera-capture-failure",
            "escalation_target": "motion_control",
        },
        "diagnostic_trace": {
            "recommended_order": [
                {"label": "检查相机 IP 配置", "evidence_message_ids": ["m2"]},
                {"label": "更换网口验证", "evidence_message_ids": ["m3"]},
            ],
            "actual_order": [
                {"label": "检查相机 IP 配置", "evidence_message_ids": ["m2"]},
                {"label": "更换网口验证", "evidence_message_ids": ["m3"]},
            ],
            "summary": "先检查配置，再做换口验证。",
        },
        "diagnostic_outcomes": [
            {
                "action_label": "更换网口验证",
                "outcome_type": "mitigation_observed",
                "condition": "更换后恢复",
                "root_cause_summary": "故障与当前网口/网络角色配置相关。",
                "evidence_message_ids": ["m3"],
                "high_cost": False,
                "destructive": False,
            }
        ],
        "required_info_candidates": [
            {
                "slot": "ip_config",
                "label": "相机IP配置",
                "question": "请提供主板网口和拓展网卡的 IP / 角色配置截图。",
                "why_required": "需要确认拍摄失败是否由换口后的网络角色或 IP 配置变化引起。",
                "evidence_message_ids": ["m2"],
            }
        ],
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    case = card["cases"][0]
    assert case["variant_hypothesis"]["label"] == "更换相机网线插口后出现拍摄失败"
    assert [item["label"] for item in case["actions"][:2]] == ["检查相机 IP 配置", "更换网口验证"]
    changed_ref = next(item["action_ref"] for item in case["actions"] if item["label"] == "更换网口验证")
    changed_outcome = next(item for item in case["outcomes"] if item["action_ref"] == changed_ref)
    assert changed_outcome["outcome_type"] == "pending_validation"
    assert "故障与当前网口/网络角色配置相关" not in changed_outcome["summary"]
    assert case["required_info"][0]["slot_hint"] == "ip_config"
    assert "IP / 角色配置截图" in case["required_info"][0]["question"]


def test_case_understanding_builder_canonicalizes_display_family_from_llm_subsystem_hint():
    semantics = {
        "candidate_id": "chatcand:test-display",
        "source_episode_id": "ep-display-001",
        "source_thread_id": "thread-display-001",
        "label": "55寸电视扩展显示不全无法复制",
        "category": "系统与软件异常",
        "symptom_raw": "外接55寸电视只能扩展不能复制，画面显示不全。",
        "conclusion": "",
        "semantic_text": "55寸电视 扩展 显示不全 无法复制 缩放 分辨率 电视",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-display-001",
            "thread_id": "thread-display-001",
            "fault_description_messages": [{"message_id": "m1", "text": "55寸电视扩展显示不全无法复制"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": ["检查显示缩放设置"]},
        },
    }
    legacy_candidate = {
        "case_variant_candidate": {
            "label": "复判站外接4K电视显示缩放设置不当导致显示不全",
            "category": "系统与软件异常",
            "subsystem": "显示/分辨率/缩放",
            "scenario": "显示不全",
            "canonical_error_id": "",
            "escalation_target": "",
        }
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    assert card["cases"][0]["family_hypothesis"]["label"] == "界面显示异常"


def test_case_understanding_builder_canonicalizes_project_search_issue_to_main_program_family():
    semantics = {
        "candidate_id": "chatcand:test-project-search",
        "source_episode_id": "ep-proj-001",
        "source_thread_id": "thread-proj-001",
        "label": "主程序无法搜索项目名",
        "category": "系统与软件异常",
        "symptom_raw": "主程序无法搜索项目名。",
        "conclusion": "",
        "semantic_text": "主程序 无法 搜索 项目名 windows 事件 日志",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-proj-001",
            "thread_id": "thread-proj-001",
            "fault_description_messages": [{"message_id": "m1", "text": "主程序无法搜索项目名"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": ["导出windows事件日志"]},
        },
    }
    legacy_candidate = {
        "case_variant_candidate": {
            "label": "主程序无法搜索项目名",
            "category": "系统与软件异常",
            "subsystem": "AOI检测软件",
            "scenario": "搜索项目名失败",
            "canonical_error_id": "",
            "escalation_target": "",
        }
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    assert card["cases"][0]["family_hypothesis"]["label"] == "主程序/系统异常"


def test_case_understanding_builder_canonicalizes_variant_label_from_question_style_text():
    semantics = {
        "candidate_id": "chatcand:test-variant-short",
        "source_episode_id": "ep-var-001",
        "source_thread_id": "thread-var-001",
        "label": "我这个现场炉前2D，使用的模式1，算法结果还没出来，软件就报警NG板卡了是什么问题",
        "category": "算法与程序调优",
        "symptom_raw": "使用模式1时算法结果未出，软件提前报警NG板卡。",
        "conclusion": "",
        "semantic_text": "我这个现场炉前2D，使用的模式1，算法结果还没出来，软件就报警NG板卡了是什么问题",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-var-001",
            "thread_id": "thread-var-001",
            "fault_description_messages": [{"message_id": "m1", "text": "算法结果还没出来软件就报警NG板卡"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": ["检查算法结果返回时序"]},
        },
    }
    legacy_candidate = {
        "case_variant_candidate": {
            "label": "我这个现场炉前2D，使用的模式1，算法结果还没出来，软件就报警NG板卡了是什么问题",
            "category": "算法与程序调优",
            "subsystem": "算法/检测逻辑",
            "scenario": "算法结果未出",
            "canonical_error_id": "",
            "escalation_target": "",
        }
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    assert card["cases"][0]["variant_hypothesis"]["label"] == "算法结果未出软件提前报警NG"


def test_case_understanding_builder_filters_and_canonicalizes_noisy_actions():
    semantics = {
        "candidate_id": "chatcand:test-actions",
        "source_episode_id": "ep-act-001",
        "source_thread_id": "thread-act-001",
        "label": "55寸电视扩展显示不全无法复制",
        "category": "系统与软件异常",
        "symptom_raw": "55寸电视扩展显示不全无法复制。",
        "conclusion": "",
        "debug_actions": [
            "设置成200%",
            "设置显示缩放为200%",
            "设置缩放100了还是不行",
            "扩展屏的分辨率设置的是多少",
            "明白",
        ],
        "semantic_text": "55寸电视 显示不全 复制 扩展 缩放 100 200 分辨率",
        "evidence_ids": ["m1", "m2"],
        "episode": {
            "episode_id": "ep-act-001",
            "thread_id": "thread-act-001",
            "fault_description_messages": [{"message_id": "m1", "text": "55寸电视扩展显示不全无法复制"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "设置成200%，设置100%，问分辨率"}],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {
                "debug_actions": [
                    "设置成200%",
                    "设置显示缩放为200%",
                    "设置缩放100了还是不行",
                    "扩展屏的分辨率设置的是多少",
                    "明白",
                ]
            },
        },
    }
    legacy_candidate = {
        "case_variant_candidate": {
            "label": "复判站电视显示不全只能扩展不能复制",
            "category": "系统与软件异常",
            "subsystem": "显示/分辨率/缩放",
            "scenario": "显示不全",
            "canonical_error_id": "",
            "escalation_target": "",
        }
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    labels = [item["label"] for item in card["cases"][0]["actions"]]
    assert "明白" not in labels
    assert "检查显示缩放比例" in labels
    assert "尝试设置显示缩放为100%" in labels
    assert "检查扩展屏分辨率设置" not in labels


def test_case_understanding_builder_uses_llm_preferred_label_for_focus_routing():
    semantics = {
        "candidate_id": "chatcand:test-focus",
        "source_episode_id": "ep-focus-001",
        "source_thread_id": "thread-focus-001",
        "label": "误报优化下呗",
        "category": "硬件与运控",
        "symptom_raw": "误报优化下呗",
        "conclusion": "",
        "semantic_text": "误报优化下呗 显示设置 缩放 分辨率",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-focus-001",
            "thread_id": "thread-focus-001",
            "fault_description_messages": [
                {"message_id": "m1", "text": "@罗新忠 罗工误报优化下呗"},
                {"message_id": "m2", "text": "没有问题"},
            ],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {
                "debug_actions": [
                    "设置成200%",
                    "扩展屏的分辨率设置的是多少",
                ]
            },
        },
    }
    legacy_candidate = {
        "label": "复判站大电视缩放设置不当导致显示不全",
        "case_variant_candidate": {
            "label": "复判站大电视缩放设置不当导致显示不全",
            "category": "系统与软件异常",
            "subsystem": "显示/分辨率/缩放",
            "scenario": "55寸大电视用作复判站显示，缩放/分辨率设置不当导致显示不全",
            "canonical_error_id": "err:red-glue-board-false-call",
            "escalation_target": "",
        },
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    assert card["schema_valid"] is True
    assert card["cases"][0]["family_hypothesis"]["label"] == "界面显示异常"


def test_case_understanding_builder_canonicalizes_boot_failure_family():
    semantics = {
        "candidate_id": "chatcand:test-boot",
        "source_episode_id": "ep-boot-001",
        "source_thread_id": "thread-boot-001",
        "label": "设备开机无法启动",
        "category": "硬件与运控",
        "symptom_raw": "设备开机无法启动，插拔内存无效。",
        "conclusion": "",
        "semantic_text": "设备开机无法启动 插拔内存无效 拔除网卡显卡采集卡后仍无法启动",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-boot-001",
            "thread_id": "thread-boot-001",
            "fault_description_messages": [{"message_id": "m1", "text": "设备开机无法启动，插拔内存无效"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": ["拔除网卡显卡采集卡后仍无法启动"]},
        },
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=None)
    assert card["cases"][0]["family_hypothesis"]["label"] == "工控机无法开机"


def test_case_understanding_builder_canonicalizes_import_failure_variant():
    semantics = {
        "candidate_id": "chatcand:test-import",
        "source_episode_id": "ep-import-001",
        "source_thread_id": "thread-import-001",
        "label": "客户现场有两台设备，程序相互导入，老设备给新设备导入不成功",
        "category": "系统与软件异常",
        "symptom_raw": "老设备给新设备程序导入不成功。",
        "conclusion": "",
        "semantic_text": "程序相互导入 老设备给新设备导入不成功 个别程序会导入失败",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-import-001",
            "thread_id": "thread-import-001",
            "fault_description_messages": [{"message_id": "m1", "text": "程序相互导入，老设备给新设备导入不成功"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": ["检查导入失败日志"]},
        },
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=None)
    assert card["cases"][0]["variant_hypothesis"]["label"] == "跨设备程序导入失败"


def test_family_and_variant_canonicalization_samples():
    assert _canonicalize_family_label(
        "客户反馈复判站弹窗报错从buddv获取保存路径失败",
        "工控机/Windows系统",
        "系统与软件异常",
        "客户反馈复判站弹窗报错从buddv获取保存路径失败",
    ) == "复判保存结果失败"
    assert _canonicalize_variant_label(
        "客户反馈复判站弹窗报错从buddv获取保存路径失败",
        "客户反馈复判站弹窗报错从buddv获取保存路径失败",
    ) == "复判站获取保存路径失败"
    assert _canonicalize_family_label(
        "客户现场有两台设备，程序相互导入，老设备给新设备导入不成功",
        "相机/采集链路",
        "系统与软件异常",
        "客户现场有两台设备，程序相互导入，老设备给新设备导入不成功",
    ) == "程序板卡加载失败"
    assert _canonicalize_variant_label(
        "客户现场有两台设备，程序相互导入，老设备给新设备导入不成功",
        "客户现场有两台设备，程序相互导入，老设备给新设备导入不成功",
    ) == "跨设备程序导入失败"
    assert _canonicalize_family_label(
        "客户u盘一插上，设备就开始操作卡顿",
        "相机/采集链路",
        "硬件与运控",
        "客户u盘一插上，设备就开始操作卡顿",
    ) == "外设连接不稳定"
    assert _canonicalize_variant_label(
        "客户u盘一插上，设备就开始操作卡顿",
        "客户u盘一插上，设备就开始操作卡顿",
    ) == "U盘插入后操作卡顿"


def test_case_understanding_builder_does_not_split_blue_screen_and_reboot_when_blue_screen_evidence_exists():
    semantics = {
        "candidate_id": "chatcand:test-blue-reboot",
        "source_episode_id": "ep-blue-001",
        "source_thread_id": "thread-blue-001",
        "label": "正常测试时设备黑屏自动重启",
        "category": "硬件与运控",
        "symptom_raw": "今天上午9:49正常测试时设备直接黑屏，自动重启，主程序日志没什么异常，版本0.25.4",
        "conclusion": "",
        "semantic_text": "正常测试时设备黑屏自动重启",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-blue-001",
            "thread_id": "thread-blue-001",
            "fault_description_messages": [{"message_id": "m1", "text": "正常测试时设备黑屏自动重启"}],
            "diagnostic_chain_messages": [
                {"message_id": "m2", "text": "分析 DMP"},
                {"message_id": "m3", "text": "开启 Driver Verifier"},
            ],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": ["分析 DMP", "开启 Driver Verifier"]},
        },
    }
    legacy_candidate = {
        "case_variant_candidate": {
            "label": "正常测试时设备黑屏自动重启",
            "category": "硬件与运控",
            "subsystem": "工控机/Windows内核",
            "scenario": "正常测试过程中设备突然黑屏并自动重启，Windows日志报0x00000139错误",
            "canonical_error_id": "err:industrial-pc-unexpected-reboot",
            "escalation_target": "硬件工程师",
        }
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=legacy_candidate)
    assert card["split_required"] is False
    assert [((c.get("family_hypothesis") or {}).get("label")) for c in card.get("cases") or []] == ["工控机蓝屏"]

def test_family_candidate_collapse_rules():
    from debug_agent_system.knowledge_v2.compat import _collapse_family_candidates
    assert _collapse_family_candidates(["工控机蓝屏", "工控机异常重启"], "0x00000139 蓝屏 自动重启", "0x00000139 关键数据结构损坏蓝屏") == ["工控机蓝屏"]
    assert _collapse_family_candidates(["误报调优异常", "算法/程序调优异常"], "颜色算法误报", "立贴USB引脚短3D成像看不到导致颜色算法误报") == ["误报调优异常"]
    assert _collapse_family_candidates(["程序运行卡顿", "软件卡死无响应", "磁盘 I/O 异常"], "虚拟内存关闭导致系统卡顿闪退", "磁盘空间或页面文件异常导致程序异常") == ["磁盘 I/O 异常"]
    assert _collapse_family_candidates(["相机拍摄失败", "工控机蓝屏"], "拍照失败后蓝屏 0x00000139", "显卡驱动损坏导致拍照失败后蓝屏") == ["工控机蓝屏"]

def test_case_understanding_builder_collapses_duplicate_variant_cases():
    from debug_agent_system.knowledge_v2.compat import _collapse_cases
    cases = [
        {
            "family_hypothesis": {"label": "工控机蓝屏"},
            "variant_hypothesis": {"label": "0x00000139 关键数据结构损坏蓝屏"},
        },
        {
            "family_hypothesis": {"label": "工控机异常重启"},
            "variant_hypothesis": {"label": "0x00000139 关键数据结构损坏蓝屏"},
        },
    ]
    collapsed = _collapse_cases(cases, "0x00000139 蓝屏 自动重启")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "工控机蓝屏"


def test_case_understanding_builder_collapses_resource_triad_cases():
    from debug_agent_system.knowledge_v2.compat import _collapse_cases
    cases = [
        {
            "family_hypothesis": {"label": "程序运行卡顿"},
            "variant_hypothesis": {"label": "磁盘空间或页面文件异常导致程序异常"},
        },
        {
            "family_hypothesis": {"label": "软件卡死无响应"},
            "variant_hypothesis": {"label": "磁盘空间或页面文件异常导致程序异常"},
        },
        {
            "family_hypothesis": {"label": "磁盘 I/O 异常"},
            "variant_hypothesis": {"label": "磁盘空间或页面文件异常导致程序异常"},
        },
    ]
    collapsed = _collapse_cases(cases, "虚拟内存关闭导致系统卡顿闪退 页面文件 磁盘")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "磁盘 I/O 异常"

def test_case_understanding_builder_filters_non_fault_report_messages():
    semantics = {
        "candidate_id": "chatcand:test-report-only",
        "source_episode_id": "ep-report-001",
        "source_thread_id": "thread-report-001",
        "label": "客户需求表格中mes需要过站人员工号，本机设置账号登陆过站，如果绑定到每个测试人员，那就要设置太多账号了，能不能白夜班分别共用一个账号",
        "category": "系统与软件异常",
        "symptom_raw": "客户需求表格中mes需要过站人员工号，白夜班分别共用一个账号。",
        "conclusion": "",
        "semantic_text": "客户需求表格中mes需要过站人员工号，白夜班分别共用一个账号",
        "evidence_ids": ["m1"],
        "episode": {
            "episode_id": "ep-report-001",
            "thread_id": "thread-report-001",
            "fault_description_messages": [{"message_id": "m1", "text": "客户需求表格中mes需要过站人员工号，本机设置账号登陆过站，如果绑定到每个测试人员，那就要设置太多账号了，能不能白夜班分别共用一个账号"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "extracted": {"debug_actions": []},
        },
    }
    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=None)
    assert card["schema_valid"] is False
    assert card["cases"] == []
    assert "missing_cases" in card["schema_issues"]

def test_family_candidate_collapse_additional_pairs():
    from debug_agent_system.knowledge_v2.compat import _collapse_family_candidates
    assert _collapse_family_candidates(["工控机蓝屏", "误报调优异常"], "2.跟线生产，误报调试，异常处理", "复判站内存管理错误导致蓝屏") == ["工控机蓝屏"]
    assert _collapse_family_candidates(["外设连接不稳定", "软件卡死无响应"], "弹窗后点击后软件无响应 黑屏关机 断电后显示器不亮", "弹窗后黑屏关机显示器不亮") == ["软件卡死无响应"]
    assert _collapse_family_candidates(["CAD 导入失败", "软件卡死无响应"], "朗特一线设备报错下道集成失败 AOI正常出板 ngbuff没有条码", "软件运行中卡死无响应") == ["软件卡死无响应"]

def test_family_candidate_collapse_more_split_pairs():
    from debug_agent_system.knowledge_v2.compat import _collapse_cases
    cases = [
        {"family_hypothesis": {"label": "误报调优异常"}, "variant_hypothesis": {"label": "chip料电极损键误报过高"}},
        {"family_hypothesis": {"label": "软件卡死无响应"}, "variant_hypothesis": {"label": "chip料电极损键误报过高"}},
    ]
    collapsed = _collapse_cases(cases, "chip料 电极 损键 误报")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "误报调优异常"

    cases = [
        {"family_hypothesis": {"label": "CAD 导入失败"}, "variant_hypothesis": {"label": "跨设备程序导入失败"}},
        {"family_hypothesis": {"label": "软件卡死无响应"}, "variant_hypothesis": {"label": "跨设备程序导入失败"}},
    ]
    collapsed = _collapse_cases(cases, "CAD 导入 程序 导入失败")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "CAD 导入失败"

def test_choose_collapsed_family_additional_pairs():
    from debug_agent_system.knowledge_v2.compat import _choose_collapsed_family
    assert _choose_collapsed_family({"工控机蓝屏", "误报调优异常"}, "复判站内存管理错误导致蓝屏 误报调试") == "工控机蓝屏"
    assert _choose_collapsed_family({"外设连接不稳定", "软件卡死无响应"}, "弹窗后黑屏关机显示器不亮") == "软件卡死无响应"
    assert _choose_collapsed_family({"CAD 导入失败", "软件卡死无响应"}, "朗特一线设备报错下道集成失败 ngbuff没有条码") == "软件卡死无响应"

def test_choose_collapsed_family_more_split_pairs():
    from debug_agent_system.knowledge_v2.compat import _choose_collapsed_family
    assert _choose_collapsed_family({"误报调优异常", "软件卡死无响应"}, "颜色算法误报很多 误报30个 引脚短 虚焊") == "误报调优异常"
    assert _choose_collapsed_family({"误报调优异常", "漏检调优异常", "算法/程序调优异常"}, "客户都觉得报少 料号丝印核对 误报30个") == "误报调优异常"
    assert _choose_collapsed_family({"外设连接不稳定", "相机拍摄失败"}, "USB网络共享 热点 导致拍摄失败") == "外设连接不稳定"

def test_family_candidate_collapse_split_pairs_more_cases():
    from debug_agent_system.knowledge_v2.compat import _collapse_cases
    cases = [
        {"family_hypothesis": {"label": "误报调优异常"}, "variant_hypothesis": {"label": "立贴USB引脚短3D成像看不到导致颜色算法误报"}},
        {"family_hypothesis": {"label": "算法/程序调优异常"}, "variant_hypothesis": {"label": "立贴USB引脚短3D成像看不到导致颜色算法误报"}},
    ]
    collapsed = _collapse_cases(cases, "立贴USB 引脚短 颜色算法 误报")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "误报调优异常"

    cases = [
        {"family_hypothesis": {"label": "漏检调优异常"}, "variant_hypothesis": {"label": "整板漏铜漏检仅检出1个"}},
        {"family_hypothesis": {"label": "算法/程序调优异常"}, "variant_hypothesis": {"label": "整板漏铜漏检仅检出1个"}},
        {"family_hypothesis": {"label": "误报调优异常"}, "variant_hypothesis": {"label": "整板漏铜漏检仅检出1个"}},
    ]
    collapsed = _collapse_cases(cases, "整板 漏铜 漏检")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "漏检调优异常"

    cases = [
        {"family_hypothesis": {"label": "工控机蓝屏"}, "variant_hypothesis": {"label": "显卡驱动损坏导致拍照失败后蓝屏"}},
        {"family_hypothesis": {"label": "算法/程序调优异常"}, "variant_hypothesis": {"label": "显卡驱动损坏导致拍照失败后蓝屏"}},
    ]
    collapsed = _collapse_cases(cases, "拍照失败后蓝屏 0x00000139")
    assert len(collapsed) == 1
    assert collapsed[0]["family_hypothesis"]["label"] == "工控机蓝屏"

def test_report_markers_and_perf_misreport_collapse():
    from debug_agent_system.knowledge_v2.compat import _is_non_fault_report, _choose_collapsed_family
    assert _is_non_fault_report('客户培训期间无实际故障 技能培训 需求对接') is True
    assert _choose_collapsed_family({"程序运行卡顿", "误报调优异常"}, '颜色算法误报很多 客户都觉得报少 引脚虚焊 误报30个') == '误报调优异常'


def test_w2_outcome_stays_pending_for_try_fix_without_closed_loop():
    episode = {
        "episode_id": "ep-outcome-001",
        "thread_id": "thread-outcome-001",
        "completeness": "partial",
        "fault_description_messages": [
            {"message_id": "m1", "sender": "fae", "text": "日志频繁显示网卡断连导致拍照失败。"}
        ],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "sender": "dev", "text": "尝试修复了驱动。"},
            {"message_id": "m3", "sender": "dev", "text": "仍然存在断连情况。"},
        ],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "网卡断连导致拍照失败。",
            "fault_focus_text": "网卡断连导致拍照失败。",
            "fault_focus_confidence": 0.9,
            "debug_actions": ["尝试修复了驱动"],
            "conclusion": "仍然存在断连情况。",
        },
    }
    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)
    outcomes = [x for x in result.get("diagnostic_outcomes") or [] if isinstance(x, dict)]
    assert all(str(item.get("outcome_type") or "") != "verified_fix" for item in outcomes)
    by_label = {str(item.get("action_label") or ""): str(item.get("outcome_type") or "") for item in outcomes}
    assert by_label["尝试修复了驱动"] == "ineffective"
    assert "仍然存在断连情况" not in by_label


def test_native_v2_does_not_turn_w1_hints_or_family_templates_into_case_facts():
    semantics = {
        "candidate_id": "chatcand:no-fabricated-case-facts",
        "source_episode_id": "ep:no-fabricated-case-facts",
        "source_thread_id": "thread:no-fabricated-case-facts",
        "label": "相机网络异常导致拍摄失败",
        "category": "硬件与运控",
        "symptom_raw": "相机网络异常导致拍摄失败",
        "conclusion": "",
        "debug_actions": [],
        "semantic_text": "相机网络异常导致拍摄失败，现场尚未开始排查。",
        "evidence_ids": ["m1"],
        "confidence": 0.8,
        "episode": {
            "episode_id": "ep:no-fabricated-case-facts",
            "thread_id": "thread:no-fabricated-case-facts",
            "fault_description_messages": [{"message_id": "m1", "text": "相机网络异常导致拍摄失败，现场尚未开始排查。"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "extracted": {
                # W1 may retain broad hints for review. Native W2 must not
                # promote them after its own action filter rejected them.
                "debug_actions": ["请老师帮忙看看是不是网卡问题"],
                "missing_info_requests": [],
            },
        },
    }

    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=None)

    assert card["schema_valid"] is True, card["schema_issues"]
    assert len(card["cases"]) == 1
    assert card["cases"][0]["candidate_scope"] == "fault_only"
    assert card["cases"][0]["actions"] == []
    assert card["cases"][0]["outcomes"] == []
    assert card["cases"][0]["required_info"] == []


def test_native_v2_gold_alignment_does_not_copy_historical_facts():
    semantics = {
        "candidate_id": "chatcand:gold-alignment-only",
        "source_episode_id": "ep:gold-alignment-only",
        "source_thread_id": "thread:gold-alignment-only",
        "label": "光源初始化失败",
        "category": "硬件与运控",
        "symptom_raw": "现场报告光源初始化失败，但尚未记录排查动作。",
        "conclusion": "",
        "debug_actions": [],
        "semantic_text": "现场报告光源初始化失败，但尚未记录排查动作。",
        "evidence_ids": ["m1"],
        "confidence": 0.8,
        "episode": {
            "episode_id": "ep:gold-alignment-only",
            "thread_id": "thread:gold-alignment-only",
            "fault_description_messages": [{"message_id": "m1", "text": "现场报告光源初始化失败，但尚未记录排查动作。"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "extracted": {},
        },
        "sop_background": {
            "context_role": "alignment_only",
            "reviewed_case_examples": [{
                "exact_source_match": True,
                "review_type": "gold_case",
                "gold_structure": {
                    "cases": [{
                        "family": {"label": "光源初始化失败"},
                        "actions": [{"label": "重新拔插光源 USB 接口"}],
                        "outcomes": [{"action_label": "重新拔插光源 USB 接口", "outcome_type": "verified_fix"}],
                    }]
                },
            }],
        },
    }

    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=None)

    assert card["schema_valid"] is True, card["schema_issues"]
    assert card["cases"][0]["actions"] == []
    assert card["cases"][0]["outcomes"] == []


def test_native_v2_binds_temporary_and_stable_results_to_atomic_actions():
    episode = {
        "episode_id": "ep:camera-reboot-chain",
        "thread_id": "thread:camera-reboot-chain",
        "completeness": "complete",
        "fault_description_messages": [{"message_id": "m1", "text": "设备断电重启后无法拍照。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [{
            "message_id": "m3",
            "text": "重新改回主板 BIOS 参数后，反复断电重启5次均未出现无法拍照异常，目前正常使用。",
        }],
        "case_evidence_messages": [{
            "message_id": "m2",
            "text": (
                "更换主板电池后开机测试拍照正常。然后断电重启并设置上电自动开机，设备再次无法拍照。"
                "把主板参数改回去后重启。"
            ),
        }],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "设备断电重启后无法拍照。",
            "fault_focus_text": "设备断电重启后无法拍照。",
            "fault_focus_confidence": 0.95,
            "debug_actions": [],
            "conclusion": "反复断电重启5次均未出现无法拍照异常。",
        },
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)
    case = result["case_understanding_card"]["cases"][0]
    labels = [item["label"] for item in case["actions"]]
    outcomes = {(item["action_ref"], item["outcome_type"], item["summary"]) for item in case["outcomes"]}

    assert any(label.startswith("更换主板电池") for label in labels)
    assert "恢复主板 BIOS 参数" in labels
    assert all("并" not in label for label in labels)
    battery_ref = next(item["action_ref"] for item in case["actions"] if item["label"].startswith("更换主板电池"))
    bios_ref = next(item["action_ref"] for item in case["actions"] if item["label"] == "恢复主板 BIOS 参数")
    assert any(ref == battery_ref and kind == "partial_temporary" for ref, kind, _ in outcomes)
    assert any(ref == bios_ref and kind in {"mitigation_observed", "verified_fix"} for ref, kind, _ in outcomes)


def test_native_v2_extracts_top_lift_family_atomic_trace_and_observed_mitigation():
    episode = {
        "episode_id": "ep:top-lift-slow",
        "thread_id": "thread:top-lift-slow",
        "completeness": "complete",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": (
                "客户反馈顶板升起速度不一致，排查二轨顶板升起降落速度过慢，"
                "原因为客户将面顶三通气管缠在了一起，气流过小，拆掉面顶测试速度正常。"
            ),
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m1",
            "text": "将面顶气缸安装到一轨侧顶升，安装后调整气流测试正常。",
        }],
        "resolution_messages": [],
        "case_evidence_messages": [],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "顶板升起速度不一致，二轨顶板升降速度过慢。",
            "fault_focus_text": "顶板升起速度不一致，二轨顶板升降速度过慢。",
            "fault_focus_confidence": 0.95,
            "debug_actions": [],
            "conclusion": "调整气流后测试正常。",
        },
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)
    case = result["case_understanding_card"]["cases"][0]
    labels = [item["label"] for item in case["actions"]]

    assert case["family_hypothesis"]["label"] == "顶升机构异常"
    assert case["variant_hypothesis"]["label"] == "顶板升降速度过慢"
    expected_labels = [
        "检查顶板升降速度",
        "拆除缠绕的面顶三通气管",
        "将面顶气缸安装到一轨侧顶升",
        "调整顶升气路流量",
        "测试顶板升降速度",
    ]
    assert all(label in labels for label in expected_labels)
    airflow_ref = next(item["action_ref"] for item in case["actions"] if item["label"] == "调整顶升气路流量")
    assert any(
        item["action_ref"] == airflow_ref and item["outcome_type"] == "mitigation_observed"
        for item in case["outcomes"]
    )


def test_native_v2_extracts_bios_battery_boot_fix_without_conflicting_outcome():
    episode = {
        "episode_id": "ep:bios-battery-boot",
        "thread_id": "thread:bios-battery-boot",
        "completeness": "complete",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "设备断电后主板 BIOS 就会重置，导致设备无法开机，更换主板电池后恢复正常。",
        }],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "case_evidence_messages": [],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "设备断电后 BIOS 重置导致无法开机。",
            "fault_focus_text": "设备断电后 BIOS 重置导致无法开机。",
            "fault_focus_confidence": 0.95,
            "debug_actions": [],
            "conclusion": "更换主板电池后恢复正常。",
        },
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)
    case = result["case_understanding_card"]["cases"][0]

    assert case["family_hypothesis"]["label"] == "工控机无法开机"
    assert case["variant_hypothesis"]["label"] == "断电后 BIOS 重置导致无法开机"
    assert [item["label"] for item in case["actions"]] == ["更换主板电池"]
    # 一次“已正常”只有短期恢复信号，不能当作长期 verified_fix。
    assert [item["outcome_type"] for item in case["outcomes"]] == ["mitigation_observed"]


def test_native_v2_does_not_treat_planned_bios_battery_change_as_executed():
    episode = {
        "episode_id": "ep:bios-battery-planned",
        "thread_id": "thread:bios-battery-planned",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "设备断电后 BIOS 重置，无法正常开机。"}],
        "diagnostic_chain_messages": [{
            "message_id": "m2",
            "text": "排查是设备断电后主板 BIOS 会重置，待换主板电池验证。已远程指导恢复正常。",
        }],
        "resolution_messages": [],
        "case_evidence_messages": [],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "断电后 BIOS 重置导致无法开机。",
            "debug_actions": ["待换主板电池验证"],
            "conclusion": "已远程指导恢复正常。",
        },
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)
    case = result["case_understanding_card"]["cases"][0]

    assert [item["label"] for item in case["actions"]] == ["更换主板电池验证"]
    assert [item["execution_status"] for item in case["actions"]] == ["recommended"]
    assert [item["outcome_type"] for item in case["outcomes"]] == ["pending_validation"]
    assert [item["outcome_origin"] for item in case["outcomes"]] == ["synthetic_fallback"]


def test_native_v2_extracts_light_usb_recovery_from_field_report_sentence():
    episode = {
        "episode_id": "ep:light-usb-recovery",
        "thread_id": "thread:light-usb-recovery",
        "completeness": "complete",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "设备离线安装通电测试后光源初始化失败，重新拔插光源USB接口已正常。",
        }],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "case_evidence_messages": [],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "通电测试后光源初始化失败。",
            "fault_focus_text": "通电测试后光源初始化失败。",
            "fault_focus_confidence": 0.9,
            "debug_actions": [],
            "conclusion": "重新拔插光源 USB 接口后恢复正常。",
        },
    }

    result = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2").extract(episode)
    case = result["case_understanding_card"]["cases"][0]
    assert case["family_hypothesis"]["label"] == "光源初始化失败"
    assert [item["label"] for item in case["actions"]] == ["重新拔插光源 USB 接口"]
    assert [item["outcome_type"] for item in case["outcomes"]] == ["mitigation_observed"]


def test_native_v2_does_not_promote_neighbor_light_usb_fix_into_status_episode():
    semantics = {
        "candidate_id": "chatcand:neighbor-fix",
        "source_episode_id": "ep:neighbor-fix",
        "source_thread_id": "thread:neighbor-fix",
        "label": "昨天光源初始化失败后重启设备",
        "category": "硬件与运控",
        "symptom_raw": "昨天光源初始化失败后重启设备。",
        "conclusion": "",
        "debug_actions": [],
        "semantic_text": "昨天光源初始化失败后重启设备。",
        "evidence_ids": ["m-current", "m-neighbor"],
        "confidence": 0.8,
        "sentence_roles": [
            {
                "message_id": "m-current",
                "text": "昨天光源初始化失败后重启设备。",
                "role": "diagnostic_action",
                "source_role": "current_fault",
                "evidence_message_ids": ["m-current"],
            },
            {
                "message_id": "m-neighbor",
                "text": "通电测试后光源初始化失败，重新拔插光源 USB 接口已正常。",
                "role": "diagnostic_action",
                "source_role": "w7_promoted",
                "evidence_message_ids": ["m-neighbor"],
            },
        ],
        "episode": {
            "episode_id": "ep:neighbor-fix",
            "thread_id": "thread:neighbor-fix",
            "fault_description_messages": [{"message_id": "m-current", "text": "昨天光源初始化失败后重启设备。"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "case_evidence_messages": [{"message_id": "m-neighbor", "text": "通电测试后光源初始化失败，重新拔插光源 USB 接口已正常。"}],
            "extracted": {},
        },
    }

    card = build_case_understanding_card_from_semantics(semantics, legacy_candidate=None)
    assert card["cases"][0]["actions"] == []
    assert card["cases"][0]["outcomes"] == []


def test_native_v2_keeps_promoted_only_action_as_audit_hint_not_execution():
    episode = {
        "episode_id": "ep:wireless-status",
        "thread_id": "thread:wireless",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m-status", "text": "软件偶发闪退，持续跟踪。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "case_evidence_messages": [{
            "message_id": "m-owner",
            "text": "现场已卸载无线网卡驱动，后续继续观察。",
        }],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m-status", "m-owner"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "软件偶发闪退。",
            "debug_actions": [],
            "conclusion": "持续跟踪。",
        },
    }

    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), w2_mode="native_v2")
    semantics = extractor.extract_semantics(episode, deepseek_enrich=False)
    result = extractor.extract(episode)

    assert semantics["debug_actions"] == ["现场已卸载无线网卡驱动"]
    assert semantics["promoted_action_hints"] == ["现场已卸载无线网卡驱动"]
    actions = result["case_understanding_card"]["cases"][0]["actions"]
    assert [item["label"] for item in actions] == ["卸载无线网卡驱动"]
    assert actions[0]["evidence_scope"] == "w7_promoted_only"
    bundle_actions = result["candidate_draft_v2_normalized_bundle"]["objects"]["DiagnosticAction"]
    assert bundle_actions[0]["evidence_scope"] == "w7_promoted_only"


def test_w2_normalizes_action_spans_before_native_v2_materialization():
    raw_actions = [
        "更换主板电池后开机测试拍照正常",
        "排查是设备断电后主板 BIOS 就会重置。待换主板电池验证",
        "更换主板电池后恢复正常",
        "重启后解决",
        "调整后解决",
        "重新开启软件后无法测试",
        "更换硬盘后",
        "重启一下",
        "重启一下，然后设备正常运行",
        "重启电脑后暂时恢复",
    ]
    episode = {
        "episode_id": "ep:action-span-cleanup",
        "thread_id": "thread:action-span-cleanup",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "设备运行异常。"}],
        "diagnostic_chain_messages": [
            {"message_id": f"m{index}", "text": text}
            for index, text in enumerate(raw_actions, start=1)
        ],
        "resolution_messages": [],
        "case_evidence_messages": [],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m0", *[f"m{index}" for index in range(1, len(raw_actions) + 1)]],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "设备运行异常。",
            "debug_actions": raw_actions,
            "conclusion": "",
        },
    }

    semantics = KnowledgeExtractionAgent(
        JsonKGStore("data/kg"), w2_mode="native_v2"
    ).extract_semantics(episode)

    assert semantics["debug_actions"] == [
        "更换主板电池",
        "建议更换主板电池验证",
        "重新开启软件",
        "重启电脑",
    ]
    role_by_text = {item["text"]: item for item in semantics["sentence_roles"]}
    assert role_by_text["更换主板电池后开机测试拍照正常"]["action_span"] == "更换主板电池"
    assert role_by_text["重新开启软件后无法测试"]["action_span"] == "重新开启软件"
    assert role_by_text["重启后解决"]["role"] == "observed_outcome"
    assert role_by_text["重启后解决"]["action_span"] == ""
    assert role_by_text["更换硬盘后"]["action_span"] == ""
    assert role_by_text["重启一下，然后设备正常运行"]["action_span"] == ""
    assert role_by_text["重启电脑后暂时恢复"]["action_span"] == "重启电脑"

    result = KnowledgeExtractionAgent(
        JsonKGStore("data/kg"), w2_mode="native_v2"
    ).extract(episode)
    assert [
        item["label"]
        for item in result["candidate_draft_v2_normalized_bundle"]["objects"]["DiagnosticAction"]
    ] == ["更换主板电池", "更换主板电池验证", "重新开启软件", "重启电脑"]


def test_w2_action_span_cleanup_keeps_scoped_diagnostic_operations():
    episode = {
        "episode_id": "ep:scoped-actions",
        "thread_id": "thread:scoped-actions",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "相机网络异常。"}],
        "diagnostic_chain_messages": [
            {"message_id": "m1", "text": "排查相机网络"},
            {"message_id": "m2", "text": "断电重启"},
            {"message_id": "m3", "text": "重启相机服务"},
        ],
        "resolution_messages": [],
        "case_evidence_messages": [],
        "case_context_messages": [],
        "noise_messages": [],
        "evidence_message_ids": ["m0", "m1", "m2", "m3"],
        "source_offsets": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "相机网络异常。",
            "debug_actions": ["排查相机网络", "断电重启", "重启相机服务"],
            "conclusion": "",
        },
    }

    semantics = KnowledgeExtractionAgent(
        JsonKGStore("data/kg"), w2_mode="native_v2"
    ).extract_semantics(episode)

    assert semantics["debug_actions"] == ["排查相机网络", "断电重启", "重启相机服务"]
