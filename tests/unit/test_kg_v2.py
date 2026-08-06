from __future__ import annotations

import tempfile
from pathlib import Path
import json
import shutil

from debug_agent_system.agents.write import KnowledgeExtractionAgent, WriteSidePipeline
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.adapters.kg_v2_adapter import KGv2Adapter
from debug_agent_system.agents.write_v2 import WriteSideV2Pipeline
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2 import JsonKGV2Store, SqliteSAGV2, build_manual_case_seed, build_sop_seed, build_v2_bundle_from_legacy_candidate, validate_graph
from debug_agent_system.eval.write_side.kg_v2_overview import write_overview


def _dual_episode() -> dict:
    return {
        "episode_id": "ep:dual-1",
        "thread_id": "thread:dual-1",
        "completeness": "complete",
        "fault_description_messages": [{"message_id": "m1", "sender": "fae", "text": "客户反馈设备卡顿后蓝屏，MEMORY_MANAGEMENT。"}],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "sender": "dev", "text": "请提供 DMP 和系统日志，先分析蓝屏根因。"},
            {"message_id": "m3", "sender": "dev", "text": "检查内存和驱动稳定性，必要时更换内存条验证。"},
        ],
        "resolution_messages": [{"message_id": "m4", "sender": "fae", "text": "现场更换内存条后未再出现蓝屏。"}],
        "noise_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3", "m4"],
        "source_offsets": [{"message_id": "m1", "index": 0}],
        "attachments": [],
        "extracted": {
            "symptom_raw": "设备卡顿后蓝屏，MEMORY_MANAGEMENT。",
            "debug_actions": ["收集 DMP 和日志", "检查内存和驱动稳定性", "更换内存条验证"],
            "conclusion": "更换内存条后未再出现蓝屏。",
            "missing_info_requests": [{
                "message_id": "m2",
                "text": "请提供 DMP 和系统日志，先分析蓝屏根因。",
                "thread_id": "thread:dual-1",
                "context_before": [],
                "context_after": [],
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }


def _init_non_sop_v2_root(root: Path) -> None:
    shutil.copytree("data/kg_v2/schema", root / "schema")
    (root / "gold_cases").mkdir(parents=True, exist_ok=True)


def test_kg_v2_schema_rejects_heavy_execution_text():
    objects = {
        "FaultFamily": [
            {
                "family_id": "family:test",
                "label": "过长节点",
                "summary": "x" * 120,
                "category": "系统与软件异常",
                "source_kind": "case",
            }
        ],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "DecisionPolicy": [],
        "EvidenceItem": [],
        "SourceCase": [],
    }
    issues = validate_graph(objects, [])
    assert any(item.startswith("text_too_long:FaultFamily.summary") for item in issues)


def test_legacy_candidate_bridge_to_v2_schema_valid():
    episode = _dual_episode()
    candidate = KnowledgeExtractionAgent().extract(episode)
    bundle = build_v2_bundle_from_legacy_candidate(candidate, episode)
    assert bundle["schema_valid"] is True
    assert bundle["objects"]["FaultFamily"]
    assert bundle["objects"]["FaultVariant"]
    assert bundle["objects"]["DiagnosticAction"]
    assert bundle["objects"]["ActionOutcome"]
    assert bundle["objects"]["RequiredInfoSpec"]


def test_kg_v2_validator_checks_outcome_origin_contract():
    episode = _dual_episode()
    candidate = KnowledgeExtractionAgent().extract(episode)
    bundle = build_v2_bundle_from_legacy_candidate(candidate, episode)
    outcome = bundle["objects"]["ActionOutcome"][0]

    outcome["outcome_origin"] = "unknown_origin"
    issues = validate_graph(bundle["objects"], bundle["relations"])
    assert any(issue.startswith("enum_mismatch:ActionOutcome.outcome_origin") for issue in issues)
    assert any(issue.startswith("invalid_outcome_origin:") for issue in issues)

    outcome["outcome_origin"] = "synthetic_fallback"
    outcome["outcome_type"] = "verified_fix"
    issues = validate_graph(bundle["objects"], bundle["relations"])
    assert any(issue.startswith("synthetic_outcome_claims_observation:") for issue in issues)

    outcome["outcome_type"] = "pending_validation"
    issues = validate_graph(bundle["objects"], bundle["relations"])
    assert not any(issue.startswith("synthetic_outcome_claims_observation:") for issue in issues)


def test_kg_v2_validator_checks_action_evidence_scope_contract():
    episode = _dual_episode()
    candidate = KnowledgeExtractionAgent().extract(episode)
    bundle = build_v2_bundle_from_legacy_candidate(candidate, episode)
    action = bundle["objects"]["DiagnosticAction"][0]
    action["evidence_scope"] = "unknown_scope"

    issues = validate_graph(bundle["objects"], bundle["relations"])

    assert any(issue.startswith("enum_mismatch:DiagnosticAction.evidence_scope") for issue in issues)
    assert any(issue.startswith("invalid_action_evidence_scope:") for issue in issues)


def test_kg_v2_builder_rejects_legacy_instances_path():
    try:
        build_sop_seed("data/kg/instances/errors/hardware-motion.json")
    except ValueError as exc:
        assert "legacy_kg_input_forbidden" in str(exc)
    else:
        raise AssertionError("expected legacy input rejection")


def test_kg_v2_sop_seed_uses_raw_chunks():
    bundle = build_sop_seed("data/raw/aoi_debug_agent_sources/chunks/debug_chunks.json", limit=8)
    assert bundle["objects"]["FaultFamily"]
    assert bundle["objects"]["DiagnosticAction"]
    assert bundle["objects"]["DiagnosticTrace"]
    assert bundle["report"]["source"] == "sop"


def test_kg_v2_manual_seed_rebuilds_case001_into_atomic_objects():
    bundle = build_manual_case_seed("data/kg/review_queue/manual_review_examples")
    variants = bundle["objects"]["FaultVariant"]
    actions = bundle["objects"]["DiagnosticAction"]
    outcomes = bundle["objects"]["ActionOutcome"]
    required = bundle["objects"]["RequiredInfoSpec"]
    target = next(item for item in variants if item["label"] == "编程拍照速度延迟现象")
    assert target["summary"]
    assert len(target["summary"]) <= 180
    assert any("检查采集卡" in item["label"] for item in actions)
    assert any(item["outcome_type"] == "ineffective" for item in outcomes)
    assert required


def test_kg_v2_manual_seed_maps_case_verified_fix_to_verified_fix():
    bundle = build_manual_case_seed("data/kg/review_queue/manual_review_examples")
    outcomes = [item for item in bundle["objects"]["ActionOutcome"] if isinstance(item, dict)]
    assert any(item["outcome_type"] == "verified_fix" for item in outcomes)


def test_kg_v2_sop_seed_canonicalizes_and_skips_non_fault_cases():
    tmp = tempfile.TemporaryDirectory()
    chunks_path = Path(tmp.name) / "chunks.json"
    chunks = [
        {
            "metadata": {"category": "debug", "title": "MES过站报错", "section_num": "1", "keywords": []},
            "text": "MES过站报错\n\n检查 MES 返回值",
        },
        {
            "metadata": {"category": "debug", "title": "加密码狗续期", "section_num": "2", "keywords": []},
            "text": "加密码狗续期\n\n联系相关负责人续期",
        },
    ]
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    bundle = build_sop_seed(chunks_path)
    families = [item for item in bundle["objects"]["FaultFamily"] if isinstance(item, dict)]
    variants = [item for item in bundle["objects"]["FaultVariant"] if isinstance(item, dict)]
    labels = {item["label"] for item in families}
    assert "MES 过站异常" in labels
    assert all(item["label"] != "加密码狗续期" for item in families)
    assert any("MES过站报错" in item["label"] for item in variants)
    assert bundle["report"]["skipped_non_fault"] == 1


def test_kg_v2_sop_seed_maps_license_and_review_result_families():
    tmp = tempfile.TemporaryDirectory()
    chunks_path = Path(tmp.name) / "chunks.json"
    chunks = [
        {
            "metadata": {"category": "debug", "title": "加密狗过期", "section_num": "1", "keywords": []},
            "text": "加密狗过期\n\n检查许可证状态",
        },
        {
            "metadata": {"category": "debug", "title": "复判站pass板无弹窗反馈", "section_num": "2", "keywords": []},
            "text": "复判站pass板无弹窗反馈\n\n复判窗口不弹出",
        },
    ]
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    bundle = build_sop_seed(chunks_path)
    labels = {item["label"] for item in bundle["objects"]["FaultFamily"] if isinstance(item, dict)}
    assert "许可证/加密狗异常" in labels
    assert "复判结果显示异常" in labels


def test_kg_v2_sop_seed_maps_result_display_and_algo_variants():
    tmp = tempfile.TemporaryDirectory()
    chunks_path = Path(tmp.name) / "chunks.json"
    chunks = [
        {
            "metadata": {"category": "debug", "title": "复盘ok后显示NG", "section_num": "1", "keywords": []},
            "text": "复盘ok后显示NG\n\n复判结果显示异常",
        },
        {
            "metadata": {"category": "debug", "title": "假焊没报出来", "section_num": "2", "keywords": []},
            "text": "假焊没报出来\n\n需要继续优化算法",
        },
    ]
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    bundle = build_sop_seed(chunks_path)
    labels = {item["label"] for item in bundle["objects"]["FaultFamily"] if isinstance(item, dict)}
    assert "复判结果显示异常" in labels
    assert "算法/程序调优异常" in labels


def test_kg_v2_store_pipeline_materialize_and_json_store_runtime_view():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "kg_v2"
    pipeline = WriteSideV2Pipeline(root)
    seeded = pipeline.seed_all(
        chunks_path="data/raw/aoi_debug_agent_sources/chunks/debug_chunks.json",
        manual_root="data/kg/review_queue/manual_review_examples",
        sop_limit=6,
        manual_limit=0,
        replace=True,
    )
    assert seeded["status"] == "replaced"
    materialized = pipeline.materialize_execution()
    assert materialized["status"] == "materialized"
    store = JsonKGStore(root / "materialized_execution")
    candidates = store.search_errors("编程拍照速度延迟，相机拍摄失败")
    assert candidates
    subgraph = store.load_locked_subgraph(candidates[0].error_id)
    assert subgraph.checks
    assert subgraph.required_info


def test_kg_v2_materializer_projects_only_verified_fix_to_resolved_by():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "kg_v2"
    store = JsonKGV2Store(root)
    objects = {
        "FaultFamily": [
            {"family_id": "family:blue-screen", "label": "蓝屏", "summary": "蓝屏", "category": "系统与软件异常", "source_kind": "case"}
        ],
        "FaultVariant": [
            {"variant_id": "variant:memory", "family_id": "family:blue-screen", "label": "换内存后蓝屏未复现", "summary": "换内存后蓝屏未复现", "equipment_type": "通用", "site": "", "software_version": "", "error_phase": "", "owner_context": "", "escalation_target": "", "keywords": ["蓝屏", "内存"]}
        ],
        "DiagnosticAction": [
            {"action_id": "action:collect-dmp", "family_id": "family:blue-screen", "variant_id": "variant:memory", "label": "收集DMP", "summary": "收集DMP并分析", "action_role": "collect", "step_order": 1, "source_kind": "case", "evidence_ids": ["evidence:1"], "execution_status": "actual"},
            {"action_id": "action:replace-memory", "family_id": "family:blue-screen", "variant_id": "variant:memory", "label": "更换内存条", "summary": "更换内存条验证", "action_role": "change", "step_order": 2, "source_kind": "case", "evidence_ids": ["evidence:1"], "execution_status": "actual"},
            {"action_id": "action:replace-capture", "family_id": "family:blue-screen", "variant_id": "variant:memory", "label": "更换采集卡", "summary": "更换采集卡验证", "action_role": "change", "step_order": 3, "source_kind": "case", "evidence_ids": ["evidence:1"], "execution_status": "actual"},
        ],
        "ActionOutcome": [
            {"outcome_id": "outcome:dmp-method", "family_id": "family:blue-screen", "variant_id": "variant:memory", "action_id": "action:collect-dmp", "outcome_type": "diagnostic_method", "summary": "已收集 DMP 用于分析", "source_case_id": "case:1", "evidence_ids": ["evidence:1"], "high_cost": False, "destructive": False, "root_cause_summary": ""},
            {"outcome_id": "outcome:memory-fix", "family_id": "family:blue-screen", "variant_id": "variant:memory", "action_id": "action:replace-memory", "outcome_type": "verified_fix", "summary": "更换内存条后未再出现蓝屏", "source_case_id": "case:1", "evidence_ids": ["evidence:1"], "high_cost": False, "destructive": False, "root_cause_summary": "内存问题"},
            {"outcome_id": "outcome:capture-bad", "family_id": "family:blue-screen", "variant_id": "variant:memory", "action_id": "action:replace-capture", "outcome_type": "ineffective", "summary": "更换采集卡无效", "source_case_id": "case:1", "evidence_ids": ["evidence:1"], "high_cost": False, "destructive": False, "root_cause_summary": ""},
        ],
        "RequiredInfoSpec": [
            {"required_info_id": "required:1", "family_id": "family:blue-screen", "variant_id": "variant:memory", "slot": "dmp_package", "question": "请提供 DMP", "why_required": "用于判断蓝屏根因", "condition": "", "blocks": ["收集DMP"], "priority": "high", "evidence_ids": ["evidence:1"]}
        ],
        "DiagnosticTrace": [
            {"trace_id": "trace:1", "family_id": "family:blue-screen", "variant_id": "variant:memory", "source_case_id": "case:1", "summary": "蓝屏排查链", "recommended_action_ids": ["action:collect-dmp", "action:replace-memory", "action:replace-capture"], "actual_action_ids": ["action:collect-dmp", "action:replace-memory", "action:replace-capture"], "evidence_ids": ["evidence:1"]}
        ],
        "DecisionPolicy": [],
        "EvidenceItem": [
            {"evidence_id": "evidence:1", "source_kind": "chat_message", "external_id": "m1", "title": "蓝屏信息", "summary": "蓝屏后更换内存条未复发", "payload_ref": ""}
        ],
        "SourceCase": [
            {"case_id": "case:1", "source_kind": "manual_review", "title": "蓝屏案例", "summary": "蓝屏案例", "source_ref": "ep:1", "approved": True}
        ],
    }
    relations = [
        {"from": "family:blue-screen", "to": "variant:memory", "relation": "has_variant"},
        {"from": "variant:memory", "to": "required:1", "relation": "has_required_info"},
        {"from": "variant:memory", "to": "trace:1", "relation": "has_trace"},
        {"from": "variant:memory", "to": "outcome:dmp-method", "relation": "has_outcome"},
        {"from": "variant:memory", "to": "outcome:memory-fix", "relation": "has_outcome"},
        {"from": "variant:memory", "to": "outcome:capture-bad", "relation": "has_outcome"},
        {"from": "trace:1", "to": "action:collect-dmp", "relation": "used_action"},
        {"from": "trace:1", "to": "action:replace-memory", "relation": "used_action"},
        {"from": "trace:1", "to": "action:replace-capture", "relation": "used_action"},
        {"from": "outcome:dmp-method", "to": "action:collect-dmp", "relation": "outcome_of"},
        {"from": "outcome:memory-fix", "to": "action:replace-memory", "relation": "outcome_of"},
        {"from": "outcome:capture-bad", "to": "action:replace-capture", "relation": "outcome_of"},
        {"from": "case:1", "to": "variant:memory", "relation": "supports"},
        {"from": "case:1", "to": "trace:1", "relation": "supports"},
        {"from": "case:1", "to": "outcome:dmp-method", "relation": "supports"},
        {"from": "case:1", "to": "outcome:memory-fix", "relation": "supports"},
        {"from": "case:1", "to": "outcome:capture-bad", "relation": "supports"},
        {"from": "case:1", "to": "required:1", "relation": "supports"},
        {"from": "evidence:1", "to": "case:1", "relation": "evidences"},
        {"from": "evidence:1", "to": "outcome:dmp-method", "relation": "evidences"},
        {"from": "evidence:1", "to": "outcome:memory-fix", "relation": "evidences"},
        {"from": "evidence:1", "to": "required:1", "relation": "evidences"},
    ]
    result = store.replace_graph(objects, relations)
    assert result["status"] == "replaced"
    adapter = KGv2Adapter(root)
    out = adapter.materialize_execution_view(root / "materialized_execution")
    assert out["status"] == "materialized"
    edges = JsonKGStore(root / "materialized_execution").edges
    resolved_targets = {(edge["from"], edge["to"]) for edge in edges if edge.get("relation") == "resolved_by"}
    assert len(resolved_targets) == 1
    assert not any("replace-capture" in dst for _, dst in resolved_targets)


def test_kg_v2_materializer_policy_prioritizes_verified_fix_and_marks_unsafe_actions():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "kg_v2"
    store = JsonKGV2Store(root)
    objects = {
        "FaultFamily": [
            {"family_id": "family:camera", "label": "相机拍摄失败", "summary": "拍摄失败", "category": "系统与软件异常", "subsystem": "相机/采集链路", "source_kind": "case"}
        ],
        "FaultVariant": [
            {"variant_id": "variant:camera", "family_id": "family:camera", "label": "换线后仍拍摄失败", "summary": "换线后仍拍摄失败", "equipment_type": "", "site": "", "software_version": "", "error_phase": "", "owner_context": "", "escalation_target": "", "keywords": ["拍摄失败"]}
        ],
        "DiagnosticAction": [
            {"action_id": "action:check-ip", "family_id": "family:camera", "variant_id": "variant:camera", "label": "检查相机IP配置", "summary": "检查相机IP配置", "action_role": "inspect", "step_order": 1, "source_kind": "case", "evidence_ids": ["e:1"], "execution_status": "actual"},
            {"action_id": "action:replace-cable", "family_id": "family:camera", "variant_id": "variant:camera", "label": "更换CXP线验证", "summary": "更换CXP线验证", "action_role": "change", "step_order": 2, "source_kind": "case", "evidence_ids": ["e:1"], "execution_status": "actual"},
            {"action_id": "action:replace-camera", "family_id": "family:camera", "variant_id": "variant:camera", "label": "更换相机返厂验证", "summary": "更换相机返厂验证", "action_role": "change", "step_order": 3, "source_kind": "case", "evidence_ids": ["e:1"], "execution_status": "recommended"},
        ],
        "ActionOutcome": [
            {"outcome_id": "outcome:ip-bad", "family_id": "family:camera", "variant_id": "variant:camera", "action_id": "action:check-ip", "outcome_type": "ineffective", "summary": "检查 IP 无异常", "source_case_id": "case:1", "evidence_ids": ["e:1"], "high_cost": False, "destructive": False, "root_cause_summary": ""},
            {"outcome_id": "outcome:cable-fix", "family_id": "family:camera", "variant_id": "variant:camera", "action_id": "action:replace-cable", "outcome_type": "verified_fix", "summary": "更换 CXP 线后恢复正常", "source_case_id": "case:1", "evidence_ids": ["e:1"], "high_cost": False, "destructive": False, "root_cause_summary": ""},
            {"outcome_id": "outcome:camera-pending", "family_id": "family:camera", "variant_id": "variant:camera", "action_id": "action:replace-camera", "outcome_type": "pending_validation", "summary": "更换相机需要返厂重标，待验证", "source_case_id": "case:1", "evidence_ids": ["e:1"], "high_cost": True, "destructive": False, "root_cause_summary": ""},
        ],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [
            {"trace_id": "trace:1", "family_id": "family:camera", "variant_id": "variant:camera", "source_case_id": "case:1", "summary": "排查链", "recommended_action_ids": ["action:check-ip", "action:replace-cable", "action:replace-camera"], "actual_action_ids": ["action:check-ip", "action:replace-cable"], "evidence_ids": ["e:1"]}
        ],
        "DecisionPolicy": [],
        "EvidenceItem": [{"evidence_id": "e:1", "source_kind": "chat_message", "external_id": "m1", "title": "消息", "summary": "消息", "payload_ref": ""}],
        "SourceCase": [{"case_id": "case:1", "source_kind": "manual_review", "title": "case", "summary": "case", "source_ref": "ep:1", "approved": True}],
    }
    relations = [
        {"from": "family:camera", "to": "variant:camera", "relation": "has_variant"},
        {"from": "variant:camera", "to": "trace:1", "relation": "has_trace"},
        {"from": "variant:camera", "to": "outcome:ip-bad", "relation": "has_outcome"},
        {"from": "variant:camera", "to": "outcome:cable-fix", "relation": "has_outcome"},
        {"from": "variant:camera", "to": "outcome:camera-pending", "relation": "has_outcome"},
        {"from": "trace:1", "to": "action:check-ip", "relation": "used_action"},
        {"from": "trace:1", "to": "action:replace-cable", "relation": "used_action"},
        {"from": "trace:1", "to": "action:replace-camera", "relation": "used_action"},
        {"from": "outcome:ip-bad", "to": "action:check-ip", "relation": "outcome_of"},
        {"from": "outcome:cable-fix", "to": "action:replace-cable", "relation": "outcome_of"},
        {"from": "outcome:camera-pending", "to": "action:replace-camera", "relation": "outcome_of"},
        {"from": "case:1", "to": "variant:camera", "relation": "supports"},
        {"from": "case:1", "to": "trace:1", "relation": "supports"},
        {"from": "case:1", "to": "outcome:ip-bad", "relation": "supports"},
        {"from": "case:1", "to": "outcome:cable-fix", "relation": "supports"},
        {"from": "case:1", "to": "outcome:camera-pending", "relation": "supports"},
        {"from": "e:1", "to": "case:1", "relation": "evidences"},
    ]
    assert store.replace_graph(objects, relations)["status"] == "replaced"
    materialized = KGv2Adapter(root).preview()
    policies = [item for item in materialized["policies"] if item.get("ordered_checks")]
    assert policies
    policy = policies[0]
    ordered_checks = policy["ordered_checks"]
    assert ordered_checks[0]["verified_fix_count"] >= ordered_checks[-1]["verified_fix_count"]
    assert ordered_checks[0]["policy_prior"] >= ordered_checks[-1]["policy_prior"]
    unsafe = {item["label"] for item in policy["unsafe_actions"]}
    assert "更换相机返厂验证" in unsafe


def test_kg_v2_materializer_maps_required_info_slots_for_runtime():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "kg_v2"
    store = JsonKGV2Store(root)
    objects = {
        "FaultFamily": [
            {"family_id": "family:boot", "label": "工控机蓝屏", "summary": "蓝屏", "category": "系统与软件异常", "subsystem": "工控机/Windows 内核", "source_kind": "case"}
        ],
        "FaultVariant": [
            {"variant_id": "variant:boot", "family_id": "family:boot", "label": "MEMORY_MANAGEMENT/PFN 不同步蓝屏", "summary": "蓝屏", "equipment_type": "", "site": "", "software_version": "", "error_phase": "", "owner_context": "", "escalation_target": "", "keywords": ["蓝屏"]}
        ],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [
            {"required_info_id": "req:dmp", "family_id": "family:boot", "variant_id": "variant:boot", "slot": "dmp_package", "question": "请提供完整 DMP", "why_required": "用于分析蓝屏根因", "condition": "", "blocks": ["分析 DMP"], "priority": "high", "evidence_ids": ["e:1"]},
            {"required_info_id": "req:driver", "family_id": "family:boot", "variant_id": "variant:boot", "slot": "driver_context", "question": "请提供驱动版本信息", "why_required": "用于判断驱动关联", "condition": "", "blocks": ["确认驱动版本"], "priority": "medium", "evidence_ids": ["e:1"]},
        ],
        "DiagnosticTrace": [],
        "DecisionPolicy": [],
        "EvidenceItem": [{"evidence_id": "e:1", "source_kind": "chat_message", "external_id": "m1", "title": "消息", "summary": "消息", "payload_ref": ""}],
        "SourceCase": [{"case_id": "case:1", "source_kind": "manual_review", "title": "case", "summary": "case", "source_ref": "ep:1", "approved": True}],
    }
    relations = [
        {"from": "family:boot", "to": "variant:boot", "relation": "has_variant"},
        {"from": "variant:boot", "to": "req:dmp", "relation": "has_required_info"},
        {"from": "variant:boot", "to": "req:driver", "relation": "has_required_info"},
        {"from": "case:1", "to": "variant:boot", "relation": "supports"},
        {"from": "case:1", "to": "req:dmp", "relation": "supports"},
        {"from": "case:1", "to": "req:driver", "relation": "supports"},
        {"from": "e:1", "to": "case:1", "relation": "evidences"},
    ]
    assert store.replace_graph(objects, relations)["status"] == "replaced"
    preview = KGv2Adapter(root).preview()
    variant_error = next(item for item in preview["errors"] if item.get("entry_role") == "case_variant")
    slots = [item["slot"] for item in variant_error["required_info_schema"]]
    assert "log_package" in slots
    assert "owner_context" in slots


def test_kg_v2_sqlite_sag_build_and_search():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "kg_v2"
    pipeline = WriteSideV2Pipeline(root)
    pipeline.seed_all(
        chunks_path="data/raw/aoi_debug_agent_sources/chunks/debug_chunks.json",
        manual_root="data/kg/review_queue/manual_review_examples",
        sop_limit=4,
        manual_limit=0,
        replace=True,
    )
    built = pipeline.build_sqlite_sag(root.parent / "kg_v2.sqlite")
    assert built["status"] == "built"
    rows = SqliteSAGV2(root.parent / "kg_v2.sqlite").search("编程拍照速度延迟", limit=5)
    assert rows


def test_write_pipeline_both_mode_writes_legacy_and_v2_review_items():
    tmp = tempfile.TemporaryDirectory()
    queue_dir = Path(tmp.name) / "legacy_queue"
    kg_v2_root = Path(tmp.name) / "kg_v2"
    _init_non_sop_v2_root(kg_v2_root)
    pipeline = WriteSidePipeline(
        JsonKGStore("data/kg"),
        queue_dir=queue_dir,
        kg_v2_root=kg_v2_root,
    )
    result = pipeline.run_summaries([_dual_episode()], kg_mode="both", dry_run_merge=True)
    assert result["kg_mode"] == "both"
    assert result["queue_writes"]
    assert result["v2_queue_writes"]
    assert result["summary"]["v2_candidates"] == 1
    assert result["summary"]["v2_dry_run"] == 1
    v2_queue = JsonKGV2Store(kg_v2_root).read_review_queue("v2_typed_candidates.json")
    assert v2_queue


def test_write_pipeline_v2_reviewed_bad_action_labels_do_not_materialize_execution():
    tmp = tempfile.TemporaryDirectory()
    kg_v2_root = Path(tmp.name) / "kg_v2"
    _init_non_sop_v2_root(kg_v2_root)
    pipeline = WriteSidePipeline(
        JsonKGStore("data/kg"),
        queue_dir=Path(tmp.name) / "legacy_queue",
        kg_v2_root=kg_v2_root,
    )
    result = pipeline.run_summaries([_dual_episode()], kg_mode="v2", dry_run_merge=True)
    store_v2 = JsonKGV2Store(kg_v2_root)
    review_agent = ReviewQueueAgent(store_v2)
    rows = store_v2.read_review_queue("v2_typed_candidates.json")
    assert rows[0]["quality_gate"]["decision"] == "route_review"
    assert rows[0]["materialize_allowed"] is False
    target = str(rows[0].get("review_id") or rows[0].get("candidate_id") or "") if rows else ""
    if target:
        review_agent.mark_decision("v2_typed_candidates", target, "approve", reviewer="test")
    assert target
    applied = pipeline.apply_approved_review_queue(kg_mode="v2")
    assert any(item.get("status") == "applied_to_graph_v2" for item in applied)
    materialized_store = JsonKGStore(kg_v2_root / "materialized_execution")
    candidates = materialized_store.search_errors("蓝屏 MEMORY_MANAGEMENT 更换内存条")
    assert candidates == []


def test_kg_v2_overview_writes_snapshot_and_html():
    out = write_overview(
        kg_v2_root="data/kg_v2",
        pinned_run_dir="data/results/w2_native_v2_full_pinned_20260708_010455",
        snapshot_out="tmp/kg_v2_overview_snapshot_test.json",
        html_out="tmp/kg_v2_overview_test.html",
    )
    assert out["family_count"] > 0
    assert out["variant_count"] > 0
    assert Path("tmp/kg_v2_overview_snapshot_test.json").exists()
    assert Path("tmp/kg_v2_overview_test.html").exists()
