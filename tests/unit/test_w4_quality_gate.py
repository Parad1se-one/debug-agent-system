from __future__ import annotations

from debug_agent_system.agents.write.w4_quality_gate import QualityGateAgent


def _bundle(family_label: str, subsystem: str, *, variant_label: str = "") -> dict:
    if not variant_label:
        variant_label = {
            "界面显示异常": "界面显示不全",
            "工控机异常重启": "设备运行中自动重启",
            "相机拍摄失败": "相机拍摄超时失败",
            "工控机蓝屏": "设备运行中蓝屏",
            "算法/程序调优异常": "算法模型调优异常",
        }.get(family_label, "示例变体")
    return {
        "candidate_id": "v2:test",
        "schema_valid": True,
        "schema_issues": [],
        "objects": {
            "FaultFamily": [{
                "family_id": "family:test",
                "label": family_label,
                "summary": "summary",
                "category": "系统与软件异常",
                "subsystem": subsystem,
                "scenario": "scene",
            }],
            "FaultVariant": [{
                "variant_id": "variant:test",
                "family_id": "family:test",
                "label": variant_label,
                "summary": "variant-summary",
            }],
            "DiagnosticAction": [{
                "action_id": "action:test",
                "family_id": "family:test",
                "variant_id": "variant:test",
                "label": "检查显示缩放设置",
                "summary": "检查显示缩放设置",
                "action_role": "inspect",
                "execution_status": "actual",
                "evidence_ids": ["evidence:test"],
            }],
            "ActionOutcome": [{
                "outcome_id": "outcome:test",
                "family_id": "family:test",
                "variant_id": "variant:test",
                "action_id": "action:test",
                "outcome_type": "diagnostic_method",
                "summary": "排查中",
                "source_case_id": "case:test",
                "evidence_ids": ["evidence:test"],
            }],
            "RequiredInfoSpec": [{
                "required_info_id": "req:test",
                "family_id": "family:test",
                "variant_id": "variant:test",
                "slot": "software_version",
                "question": "当前软件版本是多少？",
                "why_required": "确认是否受版本影响",
                "condition": "",
                "blocks": ["当前软件版本"],
                "priority": "high",
            }],
            "SourceCase": [{
                "case_id": "case:test",
                "source_kind": "manual_review",
                "title": "示例案例",
                "summary": "示例案例证据",
                "source_ref": "test",
                "approved": False,
            }],
            "EvidenceItem": [{
                "evidence_id": "evidence:test",
                "source_kind": "manual_review",
                "external_id": "m1",
                "title": "示例证据",
                "summary": "示例消息证据",
                "payload_ref": "test",
            }],
        },
        "relations": [
            {"from": "family:test", "to": "variant:test", "relation": "has_variant"},
            {"from": "variant:test", "to": "action:test", "relation": "used_action"},
            {"from": "variant:test", "to": "req:test", "relation": "has_required_info"},
            {"from": "case:test", "to": "variant:test", "relation": "supports"},
            {"from": "evidence:test", "to": "case:test", "relation": "evidences"},
        ],
    }


def test_score_v2_bundle_rejects_pseudo_family():
    gate = QualityGateAgent()
    result = gate.score_v2_bundle(_bundle("display", "显示/分辨率/缩放"))
    assert result["passed"] is False
    assert "kg_v2_noncanonical_family" in result["issues"]
    assert "kg_v2_pseudo_family" in result["issues"]


def test_score_v2_bundle_accepts_canonical_family():
    gate = QualityGateAgent()
    result = gate.score_v2_bundle(_bundle("界面显示异常", "显示/界面"))
    assert result["passed"] is True
    assert "kg_v2_noncanonical_family" not in result["issues"]


def test_score_v2_bundle_routes_fault_only_candidate_to_review():
    gate = QualityGateAgent()
    bundle = _bundle("软件卡死无响应", "主程序/运行稳定性", variant_label="调试误报时界面卡顿后软件闪退")
    bundle["objects"]["DiagnosticAction"] = []
    bundle["objects"]["ActionOutcome"] = []

    result = gate.score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_missing_actions" in result["issues"]


def test_score_v2_bundle_rejects_questionish_long_variant():
    gate = QualityGateAgent()
    result = gate.score_v2_bundle(
        _bundle(
            "算法/程序调优异常",
            "算法/程序调优",
            variant_label="我这个现场炉前2D，使用的模式1，算法结果还没出来，软件就报警NG板卡了是什么问题",
        )
    )
    assert result["passed"] is False
    assert "kg_v2_long_variant_label" in result["issues"]
    assert "kg_v2_questionish_variant_label" in result["issues"]


def test_score_v2_bundle_rejects_conversational_variant_label():
    gate = QualityGateAgent()
    result = gate.score_v2_bundle(
        _bundle(
            "工控机异常重启",
            "工控机/系统运行稳定性",
            variant_label="设备重启了应该不会打不开啊",
        )
    )
    assert result["passed"] is False
    assert "kg_v2_conversational_variant_label" in result["issues"]


def test_score_v2_bundle_rejects_non_atomic_actions():
    gate = QualityGateAgent()
    bundle = _bundle("工控机蓝屏", "工控机/Windows 内核", variant_label="运行中蓝屏重启")
    bundle["objects"]["DiagnosticAction"] = [
        {
            "action_id": "action:test:1",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "这个是重启后报的错",
            "summary": "这个是重启后报的错",
            "action_role": "inspect",
        },
        {
            "action_id": "action:test:2",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "@邢工 帮忙看一下",
            "summary": "@邢工 帮忙看一下",
            "action_role": "inspect",
        },
    ]
    result = gate.score_v2_bundle(bundle)
    assert result["passed"] is False
    assert "kg_v2_non_atomic_actions" in result["issues"]


def test_score_v2_bundle_rejects_weak_variant_prefix():
    gate = QualityGateAgent()
    result = gate.score_v2_bundle(
        _bundle(
            "相机拍摄失败",
            "相机/采集链路",
            variant_label="中午之后频繁出现拍摄失败，到达现场",
        )
    )
    assert result["passed"] is False
    assert "kg_v2_weak_variant_label" in result["issues"]


def test_score_v2_bundle_allows_version_or_scale_prefixed_fault_variant():
    gate = QualityGateAgent()
    for label in (
        "0.27.44 四线设备首个 FOV 不拍照并报拍摄失败",
        "7175 点产品误报调试时界面严重卡顿",
    ):
        result = gate.score_v2_bundle(_bundle("程序运行卡顿", "主程序/运行性能", variant_label=label))
        assert "kg_v2_weak_variant_label" not in result["issues"]


def test_score_v2_bundle_rejects_noisy_action_labels():
    gate = QualityGateAgent()
    bundle = _bundle("相机拍摄失败", "相机/采集链路", variant_label="相机网络异常导致拍摄失败")
    bundle["objects"]["DiagnosticAction"] = [
        {
            "action_id": "action:test:1",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "今日反馈表格已更新(20251011)",
            "summary": "今日反馈表格已更新(20251011)",
            "action_role": "inspect",
        },
        {
            "action_id": "action:test:2",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "检查相机网口角色与网络配置",
            "summary": "检查相机网口角色与网络配置",
            "action_role": "inspect",
        },
    ]
    result = gate.score_v2_bundle(bundle)
    assert result["passed"] is False
    assert "kg_v2_noisy_action_labels" in result["issues"]


def test_score_v2_bundle_rejects_chat_result_statements_as_actions():
    bundle = _bundle("工控机异常重启", "工控机/系统运行稳定性", variant_label="运行中自动重启")
    bundle["objects"]["DiagnosticAction"] = [
        {
            "action_id": "action:statement",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "发生时间：9:40 上面是软件导出日志",
            "summary": "发生时间：9:40 上面是软件导出日志",
            "action_role": "collect",
        },
        {
            "action_id": "action:recurred",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "就重启了",
            "summary": "就重启了",
            "action_role": "observe",
        },
    ]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_non_action_labels" in result["issues"]


def test_v2_executable_action_gate_accepts_reviewed_engineering_verbs():
    from debug_agent_system.agents.write.w4_quality_gate import _kg_v2_executable_action_label

    assert _kg_v2_executable_action_label("核对相机网线插口变更") is True
    assert _kg_v2_executable_action_label("开启 Driver Verifier") is True
    assert _kg_v2_executable_action_label("换内存条后持续观察") is True
    assert _kg_v2_executable_action_label("更换内存条后恢复正常") is False
    assert _kg_v2_executable_action_label("更换后编程依然会闪退") is False
    assert _kg_v2_executable_action_label("排查是设备断电后 BIOS 会重置") is False
    assert _kg_v2_executable_action_label("重新开启软件后无法测试") is False
    assert _kg_v2_executable_action_label("更换硬盘后") is False
    assert _kg_v2_executable_action_label("重启一下") is False
    assert _kg_v2_executable_action_label("调整后解决") is False
    assert _kg_v2_executable_action_label("等下我和夜班说下提供一下日志和具体时间") is False


def test_score_v2_bundle_rejects_unsubstantiated_verified_fix():
    bundle = _bundle("相机拍摄失败", "相机/采集链路", variant_label="相机拍摄超时")
    bundle["objects"]["ActionOutcome"] = [{
        "outcome_id": "outcome:placeholder",
        "family_id": "family:test",
        "variant_id": "variant:test",
        "action_id": "action:test",
        "outcome_type": "verified_fix",
        "summary": "camera_capture_chain",
        "source_case_id": "case:test",
        "evidence_ids": ["evidence:test"],
    }]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_unsubstantiated_verified_fix" in result["issues"]


def test_score_v2_bundle_rejects_synthetic_bridge_outcome():
    bundle = _bundle("软件卡死无响应", "主程序/运行稳定性", variant_label="软件运行中卡死无响应")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "context_not_root_cause",
        "summary": "检查系统日志 提供了重要上下文，但还不是最终根因闭环。",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_synthetic_outcome" in result["issues"]


def test_score_v2_bundle_rejects_placeholder_outcome_for_any_type():
    bundle = _bundle("工控机蓝屏", "工控机/Windows 内核", variant_label="设备运行中蓝屏")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "diagnostic_method",
        "summary": "dmp",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_synthetic_outcome" in result["issues"]


def test_score_v2_bundle_rejects_non_fault_project_variant():
    bundle = _bundle(
        "复判站加载板卡异常",
        "复判/板卡加载",
        variant_label="设备交付现场安装调试与培训安排",
    )

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_non_fault_variant" in result["issues"]


def test_score_v2_bundle_rejects_missing_observed_resolution():
    bundle = _bundle(
        "相机拍摄失败",
        "相机/采集链路",
        variant_label="杀毒后光源配置丢失导致拍摄失败",
    )
    bundle["objects"]["FaultVariant"][0]["summary"] = "重新设定光源路径后正常拍图。"
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "diagnostic_method",
        "summary": "杀毒导致光源配置丢失。",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_missing_observed_resolution" in result["issues"]


def test_score_v2_bundle_accepts_observed_resolution_with_temporary_outcome():
    bundle = _bundle(
        "相机拍摄失败",
        "相机/采集链路",
        variant_label="相机拍摄超时失败",
    )
    bundle["objects"]["SourceCase"][0]["summary"] = "重启软件后拍摄恢复正常。"
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "partial_temporary",
        "summary": "重启软件可临时恢复，根因未明确。",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is True
    assert "kg_v2_missing_observed_resolution" not in result["issues"]


def test_score_v2_bundle_rejects_result_statement_as_action():
    bundle = _bundle("进板失败", "轨道/进出板", variant_label="扫码后不进板")
    bundle["objects"]["DiagnosticAction"][0]["label"] = "分析日志发现串口对象在错误线程被访问"

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_result_statement_actions" in result["issues"]


def test_score_v2_bundle_rejects_partial_temporary_without_mitigation_semantics():
    bundle = _bundle("程序板卡加载失败", "程序/板卡加载", variant_label="导入插件解析json报错")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "partial_temporary",
        "summary": "1.3.8版本导入插件解析json报错。",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_mitigation_without_observed_improvement():
    bundle = _bundle("光源初始化失败", "光源/运控链路", variant_label="TF卡损坏导致光源初始化失败")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "mitigation_observed",
        "summary": "TF卡损坏导致ARM板系统无法启动。",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_pending_validation_without_uncertainty():
    bundle = _bundle("界面显示异常", "显示/界面", variant_label="HDMI线导致花屏异常")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "pending_validation",
        "summary": "劣质HDMI线导致显示器黑屏闪烁。",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_multiple_operations_in_one_action():
    bundle = _bundle("气压异常", "气路/顶升链路", variant_label="气压波动导致顶板失败")
    bundle["objects"]["DiagnosticAction"][0]["label"] = "检查气压并降低阈值"

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_multi_operation_actions" in result["issues"]


def test_score_v2_bundle_allows_temporal_prefix_for_executable_action():
    bundle = _bundle("程序运行卡顿", "主程序/运行性能", variant_label="大点数产品误报调试卡顿")
    bundle["objects"]["DiagnosticAction"][0]["label"] = "在卡顿时打开任务管理器性能页"

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert "kg_v2_non_action_labels" not in result["issues"]


def test_score_v2_bundle_rejects_action_chain_copied_from_unrelated_context():
    bundle = _bundle("软件卡死无响应", "主程序/运行稳定性", variant_label="AOI设备闪退日志无错误")
    bundle["source_text"] = "AOI设备发生闪退，现有日志中未发现错误。"
    bundle["objects"]["DiagnosticAction"] = [
        {"action_id": "a1", "family_id": "family:test", "variant_id": "variant:test", "label": "检查Windows系统日志", "summary": "检查Windows系统日志", "action_role": "inspect"},
        {"action_id": "a2", "family_id": "family:test", "variant_id": "variant:test", "label": "检查内存使用与泄漏", "summary": "检查内存使用与泄漏", "action_role": "inspect"},
        {"action_id": "a3", "family_id": "family:test", "variant_id": "variant:test", "label": "检查驱动兼容性", "summary": "检查驱动兼容性", "action_role": "inspect"},
    ]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_ungrounded_action_chain" in result["issues"]


def test_score_v2_bundle_allows_mostly_grounded_action_chain():
    bundle = _bundle("相机拍摄失败", "相机/采集链路", variant_label="相机CXP连接异常导致拍摄失败")
    bundle["source_text"] = "现场检查相机CXP连接状态，重启软件后拍摄临时恢复，版本0.27.44。"
    bundle["objects"]["DiagnosticAction"] = [
        {"action_id": "a1", "family_id": "family:test", "variant_id": "variant:test", "label": "检查相机CXP连接状态", "summary": "检查相机CXP连接状态", "action_role": "inspect", "execution_status": "actual", "evidence_ids": ["evidence:test"]},
        {"action_id": "a2", "family_id": "family:test", "variant_id": "variant:test", "label": "重启软件", "summary": "重启软件", "action_role": "change", "execution_status": "actual", "evidence_ids": ["evidence:test"]},
        {"action_id": "a3", "family_id": "family:test", "variant_id": "variant:test", "label": "确认软件版本", "summary": "确认软件版本", "action_role": "inspect", "execution_status": "actual", "evidence_ids": ["evidence:test"]},
    ]
    bundle["objects"]["ActionOutcome"] = [
        {"outcome_id": "o1", "family_id": "family:test", "variant_id": "variant:test", "action_id": "a1", "outcome_type": "diagnostic_method", "summary": "检查相机CXP连接状态用于定位链路。", "source_case_id": "case:test", "evidence_ids": ["evidence:test"]},
        {"outcome_id": "o2", "family_id": "family:test", "variant_id": "variant:test", "action_id": "a2", "outcome_type": "partial_temporary", "summary": "重启软件后拍摄临时恢复。", "source_case_id": "case:test", "evidence_ids": ["evidence:test"]},
        {"outcome_id": "o3", "family_id": "family:test", "variant_id": "variant:test", "action_id": "a3", "outcome_type": "diagnostic_method", "summary": "确认软件版本用于版本排查。", "source_case_id": "case:test", "evidence_ids": ["evidence:test"]},
    ]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is True
    assert "kg_v2_ungrounded_action_chain" not in result["issues"]


def test_score_v2_bundle_accepts_reviewed_user_config_variant_shape():
    bundle = _bundle(
        "复判站加载板卡异常",
        "复判/板卡加载",
        variant_label="更换工控机后 user.cfg.toml 为空导致加载用户配置失败",
    )
    bundle["objects"]["ActionOutcome"][0]["summary"] = "现场怀疑 user.cfg.toml 为空，但仍需验证。"

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is True
    assert "kg_v2_non_fault_variant" not in result["issues"]


def test_score_v2_bundle_rejects_family_variant_semantic_mismatch():
    bundle = _bundle(
        "软件卡死无响应",
        "主程序/运行稳定性",
        variant_label="显存不足导致测试失败",
    )

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_family_variant_semantic_mismatch" in result["issues"]


def test_score_v2_bundle_prefers_specific_mes_family():
    bundle = _bundle(
        "用户配置加载失败",
        "主程序配置/复判站配置",
        variant_label="MES数据上传报错且弹窗无具体信息",
    )

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_more_specific_family_available" in result["issues"]


def test_score_v2_bundle_rejects_truncated_action_parentheses():
    bundle = _bundle("相机拍摄失败", "相机/采集链路")
    bundle["objects"]["DiagnosticAction"][0]["label"] = "检查系统资源使用情况（内存/CPU"

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_malformed_action_labels" in result["issues"]


def test_score_v2_bundle_rejects_duplicate_outcomes_across_actions():
    bundle = _bundle("相机拍摄失败", "相机/采集链路")
    duplicate = dict(bundle["objects"]["ActionOutcome"][0])
    duplicate.update({"outcome_id": "outcome:duplicate", "action_id": "action:other"})
    bundle["objects"]["ActionOutcome"].append(duplicate)

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_duplicate_outcomes" in result["issues"]


def test_score_v2_bundle_rejects_duplicate_actions():
    bundle = _bundle("相机拍摄失败", "相机/采集链路")
    duplicate = dict(bundle["objects"]["DiagnosticAction"][0])
    duplicate["action_id"] = "action:duplicate"
    bundle["objects"]["DiagnosticAction"].append(duplicate)

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_duplicate_actions" in result["issues"]


def test_score_v2_bundle_rejects_near_duplicate_action_stems():
    bundle = _bundle("磁盘 I/O 异常", "磁盘/存储链路", variant_label="硬盘损坏导致磁盘读写异常")
    bundle["objects"]["DiagnosticAction"] = [
        {
            "action_id": "action:first",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "检查磁盘管理器",
            "summary": "检查磁盘管理器",
            "action_role": "inspect",
        },
        {
            "action_id": "action:second",
            "family_id": "family:test",
            "variant_id": "variant:test",
            "label": "检查磁盘管理器是否识别硬盘",
            "summary": "检查磁盘管理器是否识别硬盘",
            "action_role": "inspect",
        },
    ]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_near_duplicate_actions" in result["issues"]


def test_score_v2_bundle_rejects_pending_outcome_that_claims_resolved():
    bundle = _bundle("程序运行卡顿", "主程序/运行性能", variant_label="编程调试画面卡顿")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "pending_validation",
        "summary": "已解决，1.2.6之后修复了",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_unexecuted_verified_fix():
    bundle = _bundle("工控机蓝屏", "工控机/Windows 内核")
    bundle["source_text"] = "现场无法停线，内存诊断待后续执行。"
    bundle["objects"]["ActionOutcome"] = [{
        "outcome_id": "outcome:unexecuted",
        "family_id": "family:test",
        "variant_id": "variant:test",
        "action_id": "action:test",
        "outcome_type": "verified_fix",
        "summary": "内存诊断需数小时，现场无法停线执行",
        "source_case_id": "case:test",
        "evidence_ids": ["evidence:test"],
    }]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_outcome_evidence_from_other_episode():
    bundle = _bundle("相机拍摄失败", "相机/采集链路")
    bundle["source_message_ids"] = ["m1"]
    bundle["objects"]["EvidenceItem"].append({
        "evidence_id": "evidence:other",
        "source_kind": "chat_message",
        "external_id": "m-other",
        "title": "other episode",
        "summary": "other episode evidence",
        "payload_ref": "",
    })
    bundle["objects"]["ActionOutcome"][0]["evidence_ids"] = ["evidence:other"]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_evidence_outside_source_episode" in result["issues"]


def test_score_v2_bundle_checks_all_families_not_only_first():
    bundle = _bundle("界面显示异常", "显示/界面")
    bundle["objects"]["FaultFamily"].append({
        "family_id": "family:bad",
        "label": "自造的非规范故障名",
        "summary": "summary",
        "category": "系统与软件异常",
        "subsystem": "unknown",
        "scenario": "scene",
    })
    result = QualityGateAgent().score_v2_bundle(bundle)
    assert result["passed"] is False
    assert "kg_v2_noncanonical_family" in result["issues"]
    assert any(item["object_id"] == "family:bad" for item in result["observability"]["item_issues"])


def test_score_v2_bundle_rejects_non_fault_document_output_mode():
    bundle = _bundle("界面显示异常", "显示/界面")
    bundle["strategy"] = {"kg_output_mode": "policy_template_only"}
    result = QualityGateAgent().score_v2_bundle(bundle)
    assert result["passed"] is False
    assert "kg_v2_non_fault_output_mode" in result["issues"]


def test_score_v2_bundle_rejects_ambiguous_family_scope():
    bundle = _bundle("工控机异常重启", "工控机/系统运行稳定性")
    bundle["w3_refinement"] = {
        "review_flags": ["ambiguous_family_scope"],
        "family_scope_candidates": ["工控机异常重启", "工控机蓝屏"],
    }
    result = QualityGateAgent().score_v2_bundle(bundle)
    assert result["passed"] is False
    assert "kg_v2_ambiguous_family_scope" in result["issues"]


def test_score_candidate_rejects_positive_status_update():
    candidate = {
        "candidate_id": "cand:test",
        "label": "客户反馈说今天没有昨天也没有黑屏的情况",
        "symptom_raw": "客户反馈说今天没有昨天也没有黑屏的情况",
        "conclusion": "",
        "category": "系统与软件异常",
        "confidence": 0.9,
        "evidence_ids": ["m1"],
        "source_offsets": [{"message_id": "m1", "index": 0}],
        "schema_valid": True,
        "nodes": [
            {"type": "Error", "label": "客户反馈说今天没有昨天也没有黑屏的情况"},
            {"type": "DiagnosticCheck", "label": "持续观察"},
        ],
        "edges": [{"from": "a", "to": "b", "relation": "has_check"}],
        "episode": {
            "completeness": "partial",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈说今天没有昨天也没有黑屏的情况"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
        },
    }
    result = QualityGateAgent().score(candidate)
    assert result["passed"] is False
    assert "positive_status_not_fault" in result["issues"]


def test_score_v2_bundle_rejects_ineffective_that_only_describes_fault_failure():
    bundle = _bundle("相机拍摄失败", "相机/采集链路")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "ineffective",
        "summary": "二轨上板机卡板，开后门让一轨报错拍摄失败",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_pending_analysis_as_diagnostic_method():
    bundle = _bundle("误报调优异常", "算法/误报调优", variant_label="LED OCR 识别不稳定导致误报")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "diagnostic_method",
        "summary": "待算法团队分析，可能涉及模型识别稳定性",
    })

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_type_conflict" in result["issues"]


def test_score_v2_bundle_rejects_outcome_supported_only_by_image_placeholder():
    bundle = _bundle("程序运行卡顿", "主程序/运行性能", variant_label="大点数产品误报调试卡顿")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "diagnostic_method",
        "summary": "任务管理器显示 CPU 和内存均未占满",
        "evidence_ids": ["evidence:test"],
    })
    bundle["objects"]["EvidenceItem"][0]["external_id"] = "m-image"
    bundle["source_message_ids"] = ["m-image"]
    bundle["source_messages"] = [{"message_id": "m-image", "role": "fault", "text": "[Image: task-manager.png]"}]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is False
    assert "kg_v2_outcome_evidence_without_text_support" in result["issues"]


def test_score_v2_bundle_accepts_outcome_with_textual_followup_evidence():
    bundle = _bundle("程序运行卡顿", "主程序/运行性能", variant_label="大点数产品误报调试卡顿")
    bundle["objects"]["ActionOutcome"][0].update({
        "outcome_type": "diagnostic_method",
        "summary": "任务管理器显示 CPU 和内存均未占满",
        "evidence_ids": ["evidence:test"],
    })
    bundle["objects"]["EvidenceItem"][0]["external_id"] = "m-result"
    bundle["source_message_ids"] = ["m-result"]
    bundle["source_messages"] = [{"message_id": "m-result", "role": "w7_promoted", "text": "看上去没有什么资源占满了。"}]

    result = QualityGateAgent().score_v2_bundle(bundle)

    assert result["passed"] is True
    assert "kg_v2_outcome_evidence_without_text_support" not in result["issues"]
