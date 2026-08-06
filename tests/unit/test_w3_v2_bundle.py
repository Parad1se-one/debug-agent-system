from __future__ import annotations

import tempfile
from pathlib import Path

from debug_agent_system.agents.write import WriteSidePipeline
from debug_agent_system.agents.write.w3_conflict import ConflictResolutionAgent
from debug_agent_system.knowledge.json_store import JsonKGStore


def _raw_bundle() -> dict:
    return {
        "type": "W10SectionCaseBundleDraft",
        "bundle_id": "w10:test",
        "source_doc_title": "相机拍摄失败问题处理",
        "strategy": {"kg_output_mode": "family_support_bundle"},
        "schema_valid": False,
        "schema_issues": ["invalid_required_info_slot:req:1:DLOG"],
        "objects": {
            "FaultFamily": [{
                "family_id": "family:camera",
                "label": "camera",
                "summary": "相机拍摄失败问题",
                "category": "硬件与运控",
                "subsystem": "camera",
                "scenario": "拍摄失败",
                "keywords": ["拍摄失败"],
                "source_kind": "hybrid",
                "escalation_target": "",
            }],
            "FaultVariant": [
                {
                    "variant_id": "variant:1",
                    "family_id": "family:camera",
                    "label": "现场反馈相机网络异常导致拍摄失败是什么问题",
                    "summary": "相机网络异常导致拍摄失败",
                    "equipment_type": "",
                    "site": "",
                    "software_version": "",
                    "error_phase": "",
                    "owner_context": "",
                    "escalation_target": "",
                    "keywords": ["相机网络"],
                },
                {
                    "variant_id": "variant:2",
                    "family_id": "family:camera",
                    "label": "相机网络异常导致拍摄失败",
                    "summary": "相机网络异常导致拍摄失败",
                    "equipment_type": "",
                    "site": "",
                    "software_version": "",
                    "error_phase": "",
                    "owner_context": "",
                    "escalation_target": "",
                    "keywords": ["相机网络"],
                },
            ],
            "DiagnosticAction": [
                {
                    "action_id": "action:1",
                    "family_id": "family:camera",
                    "variant_id": "variant:1",
                    "label": "检查相机网口角色与网络配置",
                    "summary": "检查相机网口角色与网络配置",
                    "action_role": "inspect",
                    "step_order": 7,
                    "destructive": False,
                    "high_cost": False,
                    "source_kind": "hybrid",
                },
                {
                    "action_id": "action:2",
                    "family_id": "family:camera",
                    "variant_id": "variant:2",
                    "label": "检查相机网口角色与网络配置",
                    "summary": "检查相机网口角色与网络配置",
                    "action_role": "inspect",
                    "step_order": 9,
                    "destructive": False,
                    "high_cost": False,
                    "source_kind": "hybrid",
                },
            ],
            "ActionOutcome": [],
            "RequiredInfoSpec": [{
                "required_info_id": "req:1",
                "family_id": "family:camera",
                "variant_id": "variant:1",
                "slot": "DLOG",
                "question": "请提供拍摄失败时的 DLOG。",
                "why_required": "判断相机链路在哪一步失败。",
                "condition": "拍摄失败",
                "blocks": ["相机链路定位"],
                "priority": "high",
                "evidence_ids": ["evidence:1"],
            }],
            "DiagnosticTrace": [{
                "trace_id": "trace:1",
                "family_id": "family:camera",
                "variant_id": "variant:1",
                "source_case_id": "case:1",
                "summary": "相机拍摄失败排查链",
                "recommended_action_ids": ["action:1", "action:2"],
                "actual_action_ids": ["action:1", "action:2"],
                "evidence_ids": ["evidence:1"],
            }],
            "DecisionPolicy": [],
            "EvidenceItem": [{
                "evidence_id": "evidence:1",
                "source_kind": "manual_review",
                "external_id": "m1",
                "title": "相机拍摄失败消息",
                "summary": "现场反馈相机网络异常导致拍摄失败。",
                "payload_ref": "m1",
            }],
            "SourceCase": [{
                "case_id": "case:1",
                "source_kind": "manual_review",
                "title": "相机拍摄失败案例",
                "summary": "相机网络异常导致拍摄失败。",
                "source_ref": "m1",
                "approved": False,
            }],
        },
        "relations": [
            {"from": "family:camera", "to": "variant:1", "relation": "has_variant"},
            {"from": "family:camera", "to": "variant:2", "relation": "has_variant"},
            {"from": "case:1", "to": "variant:1", "relation": "supports"},
            {"from": "case:1", "to": "variant:2", "relation": "supports"},
            {"from": "evidence:1", "to": "case:1", "relation": "evidences"},
            {"from": "variant:1", "to": "req:1", "relation": "has_required_info"},
            {"from": "case:1", "to": "req:1", "relation": "supports"},
            {"from": "evidence:1", "to": "req:1", "relation": "evidences"},
            {"from": "family:camera", "to": "trace:1", "relation": "has_trace"},
            {"from": "variant:1", "to": "trace:1", "relation": "has_trace"},
            {"from": "case:1", "to": "trace:1", "relation": "supports"},
            {"from": "trace:1", "to": "action:1", "relation": "used_action"},
            {"from": "trace:1", "to": "action:2", "relation": "used_action"},
        ],
    }


def test_w3_normalizes_and_dedupes_v2_bundle():
    result = ConflictResolutionAgent().normalize_v2_bundle(_raw_bundle())

    assert result["type"] == "W3NormalizedKGV2Bundle"
    assert result["schema_valid"] is True
    assert result["schema_issues"] == []
    assert [item["label"] for item in result["objects"]["FaultFamily"]] == ["相机拍摄失败"]
    assert [item["subsystem"] for item in result["objects"]["FaultFamily"]] == ["相机/采集链路"]
    assert len(result["objects"]["FaultVariant"]) == 1
    assert result["objects"]["FaultVariant"][0]["label"] == "相机网络异常导致拍摄失败"
    assert len(result["objects"]["DiagnosticAction"]) == 1
    assert result["objects"]["DiagnosticTrace"][0]["recommended_action_ids"] == [
        result["objects"]["DiagnosticAction"][0]["action_id"]
    ]
    assert result["objects"]["RequiredInfoSpec"][0]["slot"] == "log_package"
    assert result["w3_refinement"]["change_count"] >= 4


def test_w3_keeps_source_bundle_immutable():
    source = _raw_bundle()
    ConflictResolutionAgent().normalize_v2_bundle(source)

    assert source["objects"]["FaultFamily"][0]["label"] == "camera"
    assert len(source["objects"]["FaultVariant"]) == 2
    assert source["objects"]["RequiredInfoSpec"][0]["slot"] == "DLOG"


def test_w3_refines_approved_umbrella_family_from_variant_signal():
    source = _raw_bundle()
    source["objects"]["FaultFamily"][0].update({
        "label": "用户配置加载失败",
        "subsystem": "主程序配置/复判站配置",
    })
    for variant in source["objects"]["FaultVariant"]:
        variant.update({
            "label": "MES数据上传报错且弹窗无具体信息",
            "summary": "MES站位配置异常导致数据上传报错",
        })

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert [item["label"] for item in result["objects"]["FaultFamily"]] == ["MES 过站异常"]
    assert [item["subsystem"] for item in result["objects"]["FaultFamily"]] == ["MES/接口链路"]


def test_w3_refines_vram_umbrella_family_to_cuda():
    source = _raw_bundle()
    source["objects"]["FaultFamily"][0].update({
        "label": "磁盘 I/O 异常",
        "subsystem": "磁盘/存储链路",
    })
    for variant in source["objects"]["FaultVariant"]:
        variant.update({
            "label": "显存不足导致测试失败",
            "summary": "CUDA爆显存后测试失败",
        })

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert [item["label"] for item in result["objects"]["FaultFamily"]] == ["CUDA 计算设备不可用"]


def test_w3_uses_focused_source_case_summary_for_family_refinement():
    source = _raw_bundle()
    source["objects"]["FaultFamily"][0].update({
        "label": "软件卡死无响应",
        "subsystem": "主程序/运行稳定性",
    })
    for variant in source["objects"]["FaultVariant"]:
        variant.update({"label": "软件运行异常", "summary": "软件运行异常"})
    source["objects"]["SourceCase"][0].update({
        "title": "SPC 单板分析打不开",
        "summary": "1.3.8 版本 SPC 打不开单板分析页面。",
    })

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert [item["label"] for item in result["objects"]["FaultFamily"]] == ["SPC 页面无法打开"]
    assert result["objects"]["FaultFamily"][0]["summary"] == "SPC 页面无法正常加载或打开。"


def test_w3_separates_action_result_suffix_from_atomic_action():
    source = _raw_bundle()
    source["objects"]["DiagnosticAction"][0].update({
        "label": "执行智能调试两三张板子后误报恢复正常",
        "summary": "执行智能调试两三张板子后误报恢复正常",
    })

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert result["objects"]["DiagnosticAction"][0]["label"] == "执行智能调试两三张板子"
    assert result["objects"]["DiagnosticAction"][0]["summary"] == "执行智能调试两三张板子后误报恢复正常"


def test_w3_drops_result_statement_that_leaves_only_generic_verb():
    source = _raw_bundle()
    source["objects"]["DiagnosticAction"] = [source["objects"]["DiagnosticAction"][0]]
    source["objects"]["DiagnosticAction"][0].update({
        "label": "更换后编程依然会闪退",
        "summary": "更换后编程依然会闪退",
    })

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert result["objects"]["DiagnosticAction"] == []
    assert all(item["to"] != "action:1" for item in result["relations"])
    assert any(
        item["kind"] == "underspecified_action_removed"
        for item in result["w3_refinement"]["changes"]
    )


def test_w3_removes_support_only_heading_from_fault_variants_but_keeps_evidence():
    source = _raw_bundle()
    source["objects"]["FaultVariant"].append({
        "variant_id": "variant:support",
        "family_id": "family:camera",
        "label": "常见原因参考",
        "summary": "仅作为知识背景",
        "equipment_type": "",
        "site": "",
        "software_version": "",
        "error_phase": "",
        "owner_context": "",
        "escalation_target": "",
        "keywords": [],
    })
    source["relations"].extend([
        {"from": "family:camera", "to": "variant:support", "relation": "has_variant"},
        {"from": "case:1", "to": "variant:support", "relation": "supports"},
    ])

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert result["schema_valid"] is True
    assert all(item["label"] != "常见原因参考" for item in result["objects"]["FaultVariant"])
    assert result["objects"]["EvidenceItem"][0]["evidence_id"] == "evidence:1"
    assert any(item["kind"] == "support_only_variant_removed" for item in result["w3_refinement"]["changes"])


def test_w3_raw_doc_promotes_questions_and_removes_tool_or_list_fragments():
    source = _raw_bundle()
    source["objects"]["DiagnosticAction"] = [
        {
            "action_id": "action:question",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "label": "是否刚开机温度就偏高（>60°C）？",
            "summary": "是否刚开机温度就偏高（>60°C）？",
            "action_role": "inspect",
            "step_order": 1,
            "destructive": False,
            "high_cost": False,
            "source_kind": "raw_doc",
        },
        {
            "action_id": "action:tool",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "label": "OCCT",
            "summary": "OCCT",
            "action_role": "inspect",
            "step_order": 2,
            "destructive": False,
            "high_cost": False,
            "source_kind": "raw_doc",
        },
        {
            "action_id": "action:step",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "label": "第一步",
            "summary": "第一步：清洁除尘",
            "action_role": "inspect",
            "step_order": 3,
            "destructive": False,
            "high_cost": False,
            "source_kind": "raw_doc",
        },
    ]
    source["objects"]["DiagnosticTrace"][0]["recommended_action_ids"] = [
        "action:question", "action:tool", "action:step"
    ]
    source["objects"]["DiagnosticTrace"][0]["actual_action_ids"] = [
        "action:question", "action:tool", "action:step"
    ]
    source["relations"] = [
        rel for rel in source["relations"] if rel.get("relation") != "used_action"
    ] + [
        {"from": "trace:1", "to": action_id, "relation": "used_action"}
        for action_id in ("action:question", "action:tool", "action:step")
    ]

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    labels = [item["label"] for item in result["objects"]["DiagnosticAction"]]
    assert labels == ["清洁除尘"]
    assert all(not label.startswith("是否") for label in labels)
    promoted = [
        item for item in result["objects"]["RequiredInfoSpec"]
        if item["question"].startswith("是否刚开机温度")
    ]
    assert len(promoted) == 1
    assert promoted[0]["slot"] == "environment"
    trace = result["objects"]["DiagnosticTrace"][0]
    assert trace["recommended_action_ids"] == [result["objects"]["DiagnosticAction"][0]["action_id"]]
    assert any(
        item["kind"] == "document_question_promoted_to_required_info"
        for item in result["w3_refinement"]["changes"]
    )


def test_write_pipeline_routes_native_v2_bundle_through_w3_before_w4():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = {
            "candidate_id": "candidate:test",
            "schema_valid": True,
            "schema_issues": [],
            "confidence": 0.8,
            "label": "相机拍摄失败",
            "category": "硬件与运控",
            "evidence_ids": ["m1"],
            "source_offsets": [{"message_id": "m1", "index": 0}],
            "nodes": [{
                "type": "DiagnosticCheck",
                "check_id": "check:test",
                "id": "check:test",
                "label": "检查相机网口角色与网络配置",
            }],
            "edges": [],
            "candidate_draft_v2_normalized_bundle": _raw_bundle(),
            "required_info_candidates": [],
        }
        episode = {
            "episode_id": "episode:test",
            "thread_id": "thread:test",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": "m1", "text": "相机网络异常导致拍摄失败"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1"],
            "source_offsets": [{"message_id": "m1", "index": 0}],
        }
        pipeline = WriteSidePipeline(
            JsonKGStore(root / "legacy"),
            kg_v2_root=root / "kg_v2",
            kg_v2_queue_dir=root / "kg_v2" / "review_queue",
        )

        result = pipeline._run_candidate_episode_pairs(
            [episode],
            [candidate],
            apply_approved=False,
            emit_episodes=False,
            dry_run_merge=False,
            kg_mode="v2",
            summary_counts={"summaries": 1, "episodes": 1},
        )

        assert candidate["candidate_draft_v2_w3_bundle"]["type"] == "W3NormalizedKGV2Bundle"
        assert result["summary"]["v2_candidates"] == 1
        assert result["details"][0]["gate"]["kg_v2_semantic_gate"]["observability"]["agent_id"] == "W4"


def test_w3_dedupes_same_case_outcome_after_action_merge():
    source = _raw_bundle()
    source["objects"]["ActionOutcome"] = [
        {
            "outcome_id": "outcome:1",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "action_id": "action:1",
            "outcome_type": "ineffective",
            "outcome_origin": "source_extracted",
            "summary": "检查后仍拍摄失败",
            "source_case_id": "case:1",
            "evidence_ids": ["evidence:1"],
            "high_cost": False,
            "destructive": False,
            "root_cause_summary": "",
        },
        {
            "outcome_id": "outcome:2",
            "family_id": "family:camera",
            "variant_id": "variant:2",
            "action_id": "action:2",
            "outcome_type": "ineffective",
            "outcome_origin": "source_extracted",
            "summary": "检查后仍拍摄失败",
            "source_case_id": "case:1",
            "evidence_ids": ["evidence:1"],
            "high_cost": False,
            "destructive": False,
            "root_cause_summary": "",
        },
    ]
    source["relations"].extend([
        {"from": "variant:1", "to": "outcome:1", "relation": "has_outcome"},
        {"from": "variant:2", "to": "outcome:2", "relation": "has_outcome"},
        {"from": "outcome:1", "to": "action:1", "relation": "outcome_of"},
        {"from": "outcome:2", "to": "action:2", "relation": "outcome_of"},
        {"from": "case:1", "to": "outcome:1", "relation": "supports"},
        {"from": "case:1", "to": "outcome:2", "relation": "supports"},
        {"from": "evidence:1", "to": "outcome:1", "relation": "evidences"},
        {"from": "evidence:1", "to": "outcome:2", "relation": "evidences"},
    ])

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert result["schema_valid"] is True
    assert len(result["objects"]["ActionOutcome"]) == 1
    assert result["objects"]["ActionOutcome"][0]["outcome_origin"] == "source_extracted"
    assert any(item["kind"] == "outcome_exact_merged" for item in result["w3_refinement"]["changes"])


def test_w3_downgrades_verified_fix_without_fix_evidence_semantics():
    source = _raw_bundle()
    source["objects"]["ActionOutcome"] = [
        {
            "outcome_id": "outcome:placeholder",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "action_id": "action:1",
            "outcome_type": "verified_fix",
            "summary": "camera_capture_chain",
            "source_case_id": "case:1",
            "evidence_ids": ["evidence:1"],
        },
        {
            "outcome_id": "outcome:not-reproduced",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "action_id": "action:1",
            "outcome_type": "verified_fix",
            "summary": "压力测试未复现故障",
            "source_case_id": "case:1",
            "evidence_ids": ["evidence:1"],
        },
        {
            "outcome_id": "outcome:recurred",
            "family_id": "family:camera",
            "variant_id": "variant:1",
            "action_id": "action:1",
            "outcome_type": "verified_fix",
            "summary": "就重启了",
            "source_case_id": "case:1",
            "evidence_ids": ["evidence:1"],
        },
    ]

    result = ConflictResolutionAgent().normalize_v2_bundle(source)

    assert [item["outcome_type"] for item in result["objects"]["ActionOutcome"]] == [
        "pending_validation", "pending_validation", "recurred"
    ]
    assert sum(
        item["kind"] == "outcome_type_evidence_normalized"
        for item in result["w3_refinement"]["changes"]
    ) == 3
