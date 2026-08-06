from __future__ import annotations

from debug_agent_system.eval.write_side.kg_v2_gold_compare import (
    build_prompt_a_input,
    build_prompt_b_input,
    compare_gold_case,
    load_gold_cases,
    run_legacy_bridge_baseline,
)
from debug_agent_system.eval.write_side.gold_set import verify_gold_set


def test_gold_v1_manifest_is_immutable_and_matches_all_ten_cases():
    report = verify_gold_set()
    assert report["gold_set_id"] == "gold-v1"
    assert report["immutable"] is True
    assert report["case_count"] == 10
    assert report["ok"] is True


def test_load_kg_v2_gold_cases_index():
    cases = load_gold_cases("data/annotations/goldcases/gold-v1")
    assert len(cases) == 10
    assert cases[0].case_id == "goldcase-001"


def test_xing_lark_seed_gold_cases_preserve_split_and_network_topology_semantics():
    cases = load_gold_cases("data/annotations/goldcases/gold-v1")
    network = cases[6].payload
    blue_screen = cases[7].payload

    assert network["source_group_id"] == blue_screen["source_group_id"] == "xing-lark-seed-session-001"
    assert network["split_required"] is True
    assert blue_screen["split_required"] is True
    network_actions = [item["label"] for item in network["gold"]["actions"]]
    assert "检查扩展网卡端口及下游链路状态" in network_actions
    assert "检查相机故障" not in network_actions
    blue_outcomes = {item["action_label"]: item["outcome_type"] for item in blue_screen["gold"]["outcomes"]}
    assert blue_outcomes["将SATA从RAID还原为AHCI"] == "verified_fix"


def test_macroyingda_gold_cases_preserve_independent_traces_and_buddy_storage_semantics():
    cases = load_gold_cases("data/annotations/goldcases/gold-v1")
    light = cases[5].payload
    buddy = cases[8].payload

    assert light["source_group_id"] == buddy["source_group_id"] == "xing-lark-macroyingda-20260324-20260328"
    assert light["sibling_case_ids"] == ["goldcase-009"]
    assert buddy["sibling_case_ids"] == ["goldcase-006"]
    assert buddy["merge_required"] is True
    assert buddy["gold"]["family"]["label"] == "Buddy问题"
    outcomes = {item["action_label"]: item["outcome_type"] for item in buddy["gold"]["outcomes"]}
    assert outcomes["重启主程序和Buddy"] == "ineffective"
    assert outcomes["重启电脑"] == "partial_temporary"
    assert outcomes["调整BIOS参数"] == "verified_fix"
    assert "外部网络" in buddy["gold"]["outcomes"][2]["summary"]


def test_reviewed_gold_cases_001_005_separate_observed_actions_from_recommendations():
    cases = {case.case_id: case.payload for case in load_gold_cases("data/annotations/goldcases/gold-v1")}

    case1 = cases["goldcase-001"]
    outcomes1 = {item["action_label"]: item["outcome_type"] for item in case1["gold"]["outcomes"]}
    assert outcomes1["使用最近诊断日志中的user.cfg.toml"] == "verified_fix"

    case2 = cases["goldcase-002"]
    outcomes2 = {item["action_label"]: item["outcome_type"] for item in case2["gold"]["outcomes"]}
    assert outcomes2["将相机网线插回主板网口"] == "partial_temporary"
    assert "升级大恒相机固件" not in case2["gold"]["trace"]["actual_action_labels"]

    case3 = cases["goldcase-003"]
    actual3 = case3["gold"]["trace"]["actual_action_labels"]
    assert "卸载可疑无线网卡驱动" in actual3
    assert "测试内存和 CPU 稳定性" not in actual3
    assert "开启 Driver Verifier" not in actual3
    outcomes3 = {item["action_label"]: item["outcome_type"] for item in case3["gold"]["outcomes"]}
    assert outcomes3["每日关机重启并观察"] == "pending_validation"

    case4 = cases["goldcase-004"]
    assert case4["source_episode_id"] != "pending_repo_binding"
    actual4 = case4["gold"]["trace"]["actual_action_labels"]
    assert "清除向日葵驱动" in actual4
    assert "使用 DDU 重装显卡驱动" not in actual4
    assert "开启 Driver Verifier" not in actual4

    case5 = cases["goldcase-005"]
    outcomes5 = {item["action_label"]: item["outcome_type"] for item in case5["gold"]["outcomes"]}
    assert outcomes5["将内存频率改为 2666 并观察"] == "recurred"
    assert outcomes5["升级至 1.3.7 并观察"] == "partial_temporary"
    actual5 = case5["gold"]["trace"]["actual_action_labels"]
    assert "使用 WPR 抓取内核分配趋势" not in actual5
    assert "使用 PoolMon 监控池分配" not in actual5
    assert "等待蓝屏转储完成后再重启" not in actual5


def test_nanjing_gaoxi_power_trace_preserves_diagnostic_evolution_and_temporary_fix():
    cases = {case.case_id: case.payload for case in load_gold_cases("data/annotations/goldcases/gold-v1")}
    power = cases["goldcase-010"]

    assert power["merge_required"] is True
    assert power["gold"]["family"]["label"] == "工控机异常重启"
    assert "模组电源输出线接口松动" in power["gold"]["variant"]["label"]
    outcomes = {item["action_label"]: item["outcome_type"] for item in power["gold"]["outcomes"]}
    assert outcomes["重新拔插四根内存条"] == "ineffective"
    assert outcomes["重插模组电源输出连接线"] == "partial_temporary"
    assert outcomes["检查并复测整机接地"] == "context_not_root_cause"
    assert outcomes["对松动端子点胶固定"] == "partial_temporary"
    assert outcomes["更换工控机电源"] == "pending_validation"
    assert "verified_fix" not in outcomes.values()
    actual = power["gold"]["trace"]["actual_action_labels"]
    assert "修复系统引导" not in actual
    assert "更换工控机电源" not in actual


def test_native_w2_preserves_hypothesis_evolution_and_causal_roles_for_power_trace():
    report = run_legacy_bridge_baseline(
        gold_root="data/annotations/goldcases/gold-v1",
        kg_root="data/kg",
        runner_mode="native_v2",
        with_w7_loo=True,
    )
    detail = next(item for item in report["details"] if item["case_id"] == "goldcase-010")
    timeline = detail["candidate_draft_v2"]["split_cases"][0]["trace"]["hypothesis_timeline"]
    assert any(item["causal_role"] == "root" and "模组电源" in item["summary"] for item in timeline)
    assert any(item["causal_role"] == "coexisting" and "接地" in item["summary"] for item in timeline)
    assert any(item["causal_role"] == "secondary" and "引导" in item["summary"] for item in timeline)


def test_gold_v1_zero_critical_gate_catches_no_temporary_fix_promotions():
    report = run_legacy_bridge_baseline(
        gold_root="data/annotations/goldcases/gold-v1",
        kg_root="data/kg",
        runner_mode="native_v2",
        with_w7_loo=True,
    )

    assert report["summary"]["critical_error_cases"] == 0
    assert report["summary"]["critical_error_counts"] == {}
    power = next(item for item in report["details"] if item["case_id"] == "goldcase-010")
    assert not any(
        outcome.get("outcome_type") == "verified_fix"
        for case in power["candidate_draft_v2"]["split_cases"]
        for outcome in case.get("outcomes") or []
    )


def test_prompt_inputs_build_from_gold_case():
    case = load_gold_cases("data/annotations/goldcases/gold-v1")[0]
    prompt_a = build_prompt_a_input(case)
    prompt_b = build_prompt_b_input(case)
    assert prompt_a["schema_version"] == "kg_v2.prompt_a_input.v1"
    assert prompt_b["schema_version"] == "kg_v2.prompt_b_input.v1"
    assert prompt_a["source_case_draft"]["source_episode_id"]
    assert prompt_b["case_understanding_card"]["family_hypothesis"]["label"] == "用户配置加载失败"


def test_compare_gold_case_perfect_match():
    case = load_gold_cases("data/annotations/goldcases/gold-v1")[5]
    bundle = {
        "objects": {
            "FaultFamily": [{"label": "光源初始化失败"}],
            "FaultVariant": [{"label": "离线安装通电测试后光源初始化失败，USB 重新拔插后恢复"}],
            "DiagnosticAction": [
                {"label": "检查光源初始化失败告警"},
                {"label": "重新拔插光源 USB 接口"},
                {"label": "恢复后继续观察上线验证"},
            ],
            "ActionOutcome": [
                {"summary": "重新拔插光源 USB 接口后已恢复正常。", "outcome_type": "mitigation_observed"},
                {"summary": "恢复后仍需 1-2 天上线跟线验证确认是否稳定。", "outcome_type": "pending_validation"},
            ],
            "RequiredInfoSpec": [
                {"slot": "log_package"},
                {"slot": "ip_config"},
                {"slot": "repro_steps"},
            ],
            "DiagnosticTrace": [{"recommended_action_ids": ["检查光源初始化失败告警", "重新拔插光源 USB 接口", "恢复后继续观察上线验证"]}],
        }
    }
    detail = compare_gold_case(case, bundle)
    assert detail["family_match"] is True
    assert detail["variant_match"] is True
    assert detail["action_metrics"]["recall"] == 1.0
    assert detail["required_info_metrics"]["recall"] == 1.0


def test_native_v2_runner_mode_emits_prompt_payloads():
    report = run_legacy_bridge_baseline(
        gold_root="data/annotations/goldcases/gold-v1",
        kg_root="data/kg",
        emit_prompt_inputs=True,
        runner_mode="native_v2",
    )
    assert report["runner_mode"] == "native_v2"
    assert report["summary"]["family_exact_rate"] >= 0.9
    detail = report["details"][0]
    assert detail["case_understanding_card"]["schema_version"] == "kg_v2.case_understanding.v1"
    assert detail["candidate_draft_v2"]["schema_version"] == "kg_v2.candidate_draft.v1"
