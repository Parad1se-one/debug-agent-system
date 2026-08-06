"""Build and validate the stratified KG_v2 read-side evaluation set.

The dataset deliberately has two tracks:

* ``runtime_replay`` uses only active KG_v2 identities and can be executed by
  :class:`DebugAgentSystem`.
* ``gold_trace_reasoning`` references the source-only inputs of Goldcase
  011--020 and keeps their frozen truth as labels.  These labels are evaluation
  data, never runtime evidence and never an ingestion source.

Generation is deterministic.  Queries are human-curated; graph identifiers,
plans, actions, branches, evidence and source hashes are resolved from the
current repository instead of being copied by hand.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from debug_agent_system.eval.write_side.gold_001_020_adapter import load_gold_001_020
from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel, V2DiagnosticPlan
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import kg_v2_graph_revision


SCHEMA_VERSION = "debug_agent_system.kg_v2_read_eval.v1"
DEFAULT_KG_ROOT = Path("data/kg_v2")
DEFAULT_GOLD_ROOT = Path("data/annotations/goldcases")
DEFAULT_OUT = Path("data/eval/scenarios/kg_v2_quality_v1.json")
DEFAULT_REPORT = Path("data/eval/scenarios/kg_v2_quality_v1.report.json")

DIFFICULTIES = {"easy", "medium", "hard", "expert"}
REASONING_MODES = {
    "single_step",
    "multi_hop_linear",
    "multi_hop_branch",
    "multi_trace_disambiguation",
}
TASK_TYPES = {
    "variant_retrieval",
    "first_action",
    "branch_transition",
    "ask_info_gate",
    "safety_gate",
    "resolution_gate",
    "trace_split_and_outcome_reasoning",
}


# Human-curated symptoms are intentionally phrased as field questions rather
# than copied variant labels.  ``variant_label`` is only a stable build-time
# selector; generated expectations contain canonical KG_v2 primary keys.
RUNTIME_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "slug": "user-config-empty",
        "variant_label": "更换工控机后user.cfg.toml为空或损坏导致配置加载失败",
        "query": "更换工控机后主程序报加载用户配置失败，检查发现 user.cfg.toml 为空或已损坏。",
        "difficulty": "medium",
        "focus": ["exact_context", "configuration", "gold_derived_active_kg"],
    },
    {
        "slug": "camera-packet-loss",
        "variant_label": "相机链路丢包与事件包不重传导致拍摄失败",
        "query": "2D相机生产中拍摄失败，日志先报残帧和丢包，随后事件包丢失且没有重传。",
        "difficulty": "hard",
        "focus": ["log_sequence", "camera", "counterfactual"],
    },
    {
        "slug": "bsod-139",
        "variant_label": "0x00000139 关键数据结构损坏蓝屏",
        "query": "工控机运行中蓝屏，BugCheck 0x00000139，转储提示关键数据结构损坏。",
        "difficulty": "medium",
        "focus": ["exact_error_code", "blue_screen", "negative_attribution"],
    },
    {
        "slug": "inaccessible-boot-device",
        "variant_label": "INACCESSIBLE_BOOT_DEVICE启动蓝屏",
        "query": "工控机启动蓝屏 INACCESSIBLE_BOOT_DEVICE，可以进入 BIOS，但无法进入 Windows。",
        "difficulty": "medium",
        "focus": ["exact_error_code", "boot_chain", "failed_actions"],
    },
    {
        "slug": "buddy-disk-missing",
        "variant_label": "D盘消失导致Buddy冷存储写入失败",
        "query": "Buddy保存冷存储时报 HTTP 500，日志显示 mkdir D:\\ACME 路径不存在，同时资源管理器里的 D 盘消失。",
        "difficulty": "hard",
        "focus": ["cross_subsystem", "causal_chain", "failed_actions"],
    },
    {
        "slug": "realtek-reset",
        "variant_label": "Realtek 2.5G扩展网卡反复重置并导致黑屏/断连",
        "query": "相机和云控同时断连，系统日志反复出现 NDIS 10400 和 Realtek 2.5G 扩展网卡重置。",
        "difficulty": "hard",
        "focus": ["cross_signal", "network", "temporary_fix"],
    },
    {
        "slug": "light-usb-recovery",
        "variant_label": "离线安装通电测试后光源初始化失败，USB 重新拔插后恢复",
        "query": "设备离线安装后通电测试，启动时光源初始化失败，重新插拔光源 USB 后恢复。",
        "difficulty": "medium",
        "focus": ["temporary_recovery", "light_source", "resolution_boundary"],
    },
    {
        "slug": "power-connector",
        "variant_label": "老版本模组电源输出线接口松动导致供电中断",
        "query": "工控机正常测试时反复自动重启，内存和系统修复均无效，怀疑老版本模组电源输出线接口松动。",
        "difficulty": "hard",
        "focus": ["long_trace", "power", "counterevidence"],
    },
    {
        "slug": "3d-fov-missing",
        "variant_label": "3D FOV图片缺失",
        "query": "3D拍摄日志提示某个 FOV 图片不足 42 张，采图不完整。",
        "difficulty": "easy",
        "focus": ["exact_log", "3d_camera"],
    },
    {
        "slug": "2d-event-timeout",
        "variant_label": "2D相机事件超时",
        "query": "2D相机拍摄过程中事件超时，日志出现 event timeout。",
        "difficulty": "easy",
        "focus": ["exact_log", "2d_camera"],
    },
    {
        "slug": "buddy-template-create",
        "variant_label": "Buddy 模板创建失败",
        "query": "Buddy正常开启，但模板管理刷新后没有模板，创建模板时请求失败。",
        "difficulty": "easy",
        "focus": ["sop", "buddy"],
    },
    {
        "slug": "mark-align",
        "variant_label": "Mark 点对齐失败",
        "query": "AOI进板后 Mark 点对齐失败，需要判断进板方向、进板位置还是参数问题。",
        "difficulty": "easy",
        "focus": ["sop", "mark"],
    },
    {
        "slug": "ct-growth",
        "variant_label": "CT 时间异常增加",
        "query": "AOI检测节拍越来越长，需要区分 capture time 和 detection time 哪一段异常。",
        "difficulty": "medium",
        "focus": ["metric_decomposition", "performance"],
    },
    {
        "slug": "ipc-no-boot",
        "variant_label": "工控机无法正常开机",
        "query": "工控机按电源后无法正常开机，需要按供电、自检、显示和系统引导阶段排查。",
        "difficulty": "medium",
        "focus": ["stage_diagnosis", "hardware"],
    },
    {
        "slug": "cuda-missing",
        "variant_label": "未检查到CUDA设备",
        "query": "主程序报警未检查到 CUDA 设备，设备管理器中的显卡状态也异常。",
        "difficulty": "easy",
        "focus": ["exact_alarm", "gpu"],
    },
    {
        "slug": "infeed-failure",
        "variant_label": "进板失败",
        "query": "板子到达进板口但皮带不转，导致板子停在入口，应该从哪里开始排查？",
        "difficulty": "medium",
        "focus": ["ordered_sop", "conveyor"],
    },
    {
        "slug": "camera-auto-ip",
        "variant_label": "相机IP自动获取识别不到",
        "query": "2D相机使用自动获取 IP 后识别不到，相机与主机网络不通。",
        "difficulty": "easy",
        "focus": ["network_config", "camera"],
    },
    {
        "slug": "spc-browser-hijack",
        "variant_label": "SPC页面打不开-浏览器被360劫持",
        "query": "SPC 页面打不开，现场发现默认浏览器被 360 劫持。",
        "difficulty": "easy",
        "focus": ["single_cause", "spc"],
    },
    {
        "slug": "memory-dmp-analysis",
        "variant_label": "MEMORY.DMP分析流程",
        "query": "工控机蓝屏后已经拿到 MEMORY.DMP，需要使用 WinDbg 做分析。",
        "difficulty": "easy",
        "focus": ["procedure", "evidence_collection"],
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _variant_by_label(model: KGV2ReadModel, label: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (variant_id, variant)
        for variant_id, variant in model.by_type["FaultVariant"].items()
        if str(variant.get("label") or "") == label and model.is_runtime_variant(variant_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"runtime FaultVariant label must resolve once: {label!r}, matches={len(matches)}")
    return matches[0]


def _source_refs(model: KGV2ReadModel, plan: V2DiagnosticPlan) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    source_case_ids: set[str] = set()
    for object_id in (plan.trace_id, plan.variant_id, *plan.evidence_ids):
        item = model.get(object_id) or {}
        source_case_id = str(item.get("source_case_id") or "")
        if source_case_id:
            source_case_ids.add(source_case_id)
    for source_case_id in sorted(source_case_ids):
        refs.append({"kind": "KG_v2.SourceCase", "id": source_case_id})
    for evidence_id in plan.evidence_ids[:12]:
        item = model.get(evidence_id) or {}
        refs.append({
            "kind": "KG_v2.EvidenceItem",
            "id": evidence_id,
            "source_kind": str(item.get("source_kind") or ""),
            "external_id": str(item.get("external_id") or ""),
            "payload_ref": str(item.get("payload_ref") or ""),
        })
    return refs


def _expected_base(model: KGV2ReadModel, plan: V2DiagnosticPlan) -> dict[str, Any]:
    return {
        "family_id": plan.family_id,
        "variant_id": plan.variant_id,
        "acceptable_variant_ids": [plan.variant_id],
        "plan_id": plan.plan_id,
        "trace_id": plan.trace_id,
        "policy_id": plan.policy_id,
        "first_action_id": plan.steps[0].action_id,
        "action_sequence": [step.action_id for step in plan.steps],
        "required_info_ids": list(plan.required_info_ids),
        "evidence_ids": list(plan.evidence_ids),
        "sag": {
            "expected_top_k": 1,
            "expected_route": "sag_v2_native",
            "require_retrieval_paths": True,
        },
    }


def _runtime_case(
    *,
    case_id: str,
    profile: dict[str, Any],
    plan: V2DiagnosticPlan,
    model: KGV2ReadModel,
    task_type: str,
    difficulty: str,
    reasoning_mode: str,
    hop_count: int,
    turns: list[dict[str, Any]] | None = None,
    expected_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = _expected_base(model, plan)
    expected.update(expected_patch or {})
    return {
        "case_id": case_id,
        "evaluation_track": "runtime_replay",
        "split": "regression",
        "task_type": task_type,
        "difficulty": difficulty,
        "reasoning_mode": reasoning_mode,
        "hop_count": hop_count,
        "query": profile["query"],
        "source_refs": _source_refs(model, plan),
        "turns": turns or [],
        "expected": expected,
        "quality": {
            "query_origin": "human_curated_from_source_facts",
            "focus": list(profile["focus"]),
            "leakage_control": "query_contains_no_KG_primary_key_or_action_answer",
            "evidence_required": True,
            "negative_control": task_type in {"ask_info_gate", "safety_gate", "resolution_gate"},
        },
    }


def _message_for_outcome(outcome_type: str) -> str:
    return {
        "diagnostic_method": "已完成该项检查并记录了结果。",
        "ineffective": "已执行，但仍未解决，故障还是存在。",
        "pending_validation": "暂时恢复，目前仍在观察中。",
        "mitigation_observed": "执行后有所缓解，但尚未完成稳定性验证。",
        "recurred": "短暂恢复后问题再次复现。",
        "verified_fix": "已解决并恢复正常，持续验证未再复现。",
        "context_not_root_cause": "检查结果表明这不是根因，继续排查。",
    }.get(outcome_type, "已完成检查，继续排查。")


def _first_branch_case(
    profile: dict[str, Any],
    plan: V2DiagnosticPlan,
    model: KGV2ReadModel,
) -> dict[str, Any] | None:
    first = plan.steps[0]
    rules = model.branch_rules_for_step(first)
    if not rules:
        return None
    rule = sorted(rules, key=lambda item: (int(item.get("priority") or 9999), str(item.get("branch_rule_id"))))[0]
    outcome_type = str((rule.get("trigger_outcome_types") or ["diagnostic_method"])[0])
    target_trace_step_id = str(rule.get("to_trace_step_id") or "")
    target = next((step for step in plan.steps if step.trace_step_id == target_trace_step_id), None)
    terminal = str(rule.get("terminal_status") or "continue")
    expected_status = "resolved" if terminal == "resolved" and outcome_type == "verified_fix" else "step"
    if target is not None and (target.destructive or target.high_cost):
        expected_status = "ask_info"
    return _runtime_case(
        case_id=f"runtime-branch-{profile['slug']}",
        profile=profile,
        plan=plan,
        model=model,
        task_type="branch_transition",
        difficulty="hard",
        reasoning_mode="multi_hop_branch",
        hop_count=3,
        turns=[{
            "user_message": _message_for_outcome(outcome_type),
            "classified_outcome_type": outcome_type,
            "expected_branch_rule_id": str(rule.get("branch_rule_id") or ""),
            "expected_status": expected_status,
            "expected_action_id": target.action_id if target else "",
        }],
        expected_patch={
            "branch_rule_ids": [str(rule.get("branch_rule_id") or "")],
            "terminal_status": expected_status,
        },
    )


def _build_runtime_cases(model: KGV2ReadModel) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    branch_candidates: list[tuple[dict[str, Any], V2DiagnosticPlan]] = []
    plans: dict[str, tuple[dict[str, Any], V2DiagnosticPlan]] = {}
    for profile in RUNTIME_PROFILES:
        variant_id, variant = _variant_by_label(model, profile["variant_label"])
        plan = model.compile_plan(str(variant.get("family_id") or ""), variant_id)
        if not plan.steps:
            raise ValueError(f"runtime profile has no executable plan: {profile['slug']}")
        plans[profile["slug"]] = (profile, plan)
        cases.append(_runtime_case(
            case_id=f"runtime-retrieval-{profile['slug']}",
            profile=profile,
            plan=plan,
            model=model,
            task_type="variant_retrieval",
            difficulty=profile["difficulty"],
            reasoning_mode="single_step",
            hop_count=1,
            expected_patch={"terminal_status": "step"},
        ))
        cases.append(_runtime_case(
            case_id=f"runtime-first-action-{profile['slug']}",
            profile=profile,
            plan=plan,
            model=model,
            task_type="first_action",
            difficulty="medium" if profile["difficulty"] == "easy" else profile["difficulty"],
            reasoning_mode="multi_hop_linear",
            hop_count=2,
            expected_patch={
                "terminal_status": "ask_info" if (plan.steps[0].destructive or plan.steps[0].high_cost) else "step",
                "safety_confirmation_required": bool(plan.steps[0].destructive or plan.steps[0].high_cost),
            },
        ))
        if plan.steps[0].branch_rule_ids:
            branch_candidates.append((profile, plan))

    # One branch case per active Gold-derived trajectory.  This explicitly
    # covers failed, temporary, diagnostic and verified transition semantics.
    for profile, plan in branch_candidates:
        branch_case = _first_branch_case(profile, plan, model)
        if branch_case is not None:
            cases.append(branch_case)

    generic_profile, generic_plan = plans["buddy-template-create"]
    cases.append(_runtime_case(
        case_id="runtime-gate-fail-closed-resolution",
        profile=generic_profile,
        plan=generic_plan,
        model=model,
        task_type="resolution_gate",
        difficulty="hard",
        reasoning_mode="multi_hop_branch",
        hop_count=3,
        turns=[{
            "user_message": "已经解决，恢复正常。",
            "classified_outcome_type": "verified_fix",
            "expected_status": "step",
            "expected_failure_type": "pending_validation",
        }],
        expected_patch={
            "terminal_status": "step",
            "forbidden_statuses": ["resolved"],
            "required_outcome_type": "verified_fix",
        },
    ))

    resolved_profile, resolved_plan = plans["user-config-empty"]
    verified_step = next(
        (
            step for step in resolved_plan.steps
            if any(
                str((model.get(outcome_id) or {}).get("outcome_type") or "") == "verified_fix"
                for outcome_id in step.outcome_ids
            )
        ),
        None,
    )
    if verified_step is not None:
        turns = [
            {
                "user_message": "已检查，但仍未解决。",
                "classified_outcome_type": "ineffective",
                "expected_status": "step",
                "expected_action_id": step.action_id,
            }
            for step in resolved_plan.steps[1:verified_step.ordinal]
        ]
        turns.append({
            "user_message": "替换为最近一次诊断日志中的配置后已解决，恢复正常，复验未再复现。",
            "classified_outcome_type": "verified_fix",
            "expected_status": "resolved",
            "expected_action_id": verified_step.action_id,
        })
        cases.append(_runtime_case(
            case_id="runtime-gate-evidence-backed-resolution",
            profile=resolved_profile,
            plan=resolved_plan,
            model=model,
            task_type="resolution_gate",
            difficulty="hard",
            reasoning_mode="multi_hop_branch",
            hop_count=verified_step.ordinal + 1,
            turns=turns,
            expected_patch={
                "terminal_status": "resolved",
                "verified_action_id": verified_step.action_id,
                "verified_outcome_ids": [
                    outcome_id for outcome_id in verified_step.outcome_ids
                    if str((model.get(outcome_id) or {}).get("outcome_type") or "") == "verified_fix"
                ],
            },
        ))

    safety_variant_id, safety_variant = _variant_by_label(model, "光源初始化异常")
    safety_plan = model.compile_plan(str(safety_variant.get("family_id") or ""), safety_variant_id)
    safety_profile = {
        "slug": "light-source-init-safety",
        "variant_label": "光源初始化异常",
        "query": "光源初始化异常，准备退出软件并断电后重启。",
        "difficulty": "hard",
        "focus": ["human_confirmation", "destructive_action", "light_source"],
    }
    first_unsafe = next((step for step in safety_plan.steps if step.destructive or step.high_cost), None)
    if first_unsafe is not None:
        cases.append(_runtime_case(
            case_id="runtime-gate-human-confirmation",
            profile=safety_profile,
            plan=safety_plan,
            model=model,
            task_type="safety_gate",
            difficulty="hard",
            reasoning_mode="multi_hop_branch",
            hop_count=2,
            turns=[],
            expected_patch={
                "terminal_status": "ask_info" if safety_plan.steps[0].action_id == first_unsafe.action_id else "step",
                "first_unsafe_action_id": first_unsafe.action_id,
                "safety_confirmation_required": True,
                "forbidden_action_ids_without_confirmation": [first_unsafe.action_id],
            },
        ))

    ask_profile, ask_plan = plans["bsod-139"]
    ask_profile = dict(ask_profile)
    ask_profile["query"] += " 当前缺少日志包、驱动上下文和内存测试结果。"
    cases.append(_runtime_case(
        case_id="runtime-gate-required-info",
        profile=ask_profile,
        plan=ask_plan,
        model=model,
        task_type="ask_info_gate",
        difficulty="hard",
        reasoning_mode="multi_hop_branch",
        hop_count=2,
        expected_patch={
            "terminal_status": "ask_info",
            "required_info_ids": list(ask_plan.required_info_ids),
        },
    ))
    return cases


def _gold_case(case: dict[str, Any]) -> dict[str, Any]:
    number = int(case["case_id"].rsplit("-", 1)[-1])
    source_input = case["source"]["input"]
    if not source_input:
        raise ValueError(f"Gold reasoning case has no source-only input: {case['case_id']}")
    truth_path = Path(case["source"]["truth_path"])
    input_path = Path(source_input["path"])
    traces = case["traces"]
    outcome_counts = Counter(
        action["outcome"]["outcome_type"]
        for trace in traces
        for action in trace["actions"]
    )
    return {
        "case_id": f"gold-reasoning-{number:03d}",
        "evaluation_track": "gold_trace_reasoning",
        "split": "validation" if number <= 15 else "held_out_test",
        "task_type": "trace_split_and_outcome_reasoning",
        "difficulty": "expert",
        "reasoning_mode": "multi_trace_disambiguation",
        "hop_count": max(4, len(traces) + 2),
        "query": (
            "读取 source_input_ref 中的原始消息、Jira 与可用附件元数据；按设备、故障链和时间边界拆分诊断轨迹，"
            "逐轨迹输出故障族、变体、原子动作、动作结果、证据锚点和不确定性。"
        ),
        "source_input_ref": str(input_path),
        "source_refs": [{
            "kind": "gold_source_only_input",
            "path": str(input_path),
            "sha256": source_input["sha256"],
            "label_visibility": str((case.get("input_payload") or {}).get("label_visibility") or "source_only"),
        }],
        "turns": [],
        "expected": {
            "truth_ref": str(truth_path),
            "truth_sha256": case["source"]["truth_sha256"],
            "trace_count": case["trace_count"],
            "trace_ids": [trace["trace_id"] for trace in traces],
            "family_labels": _dedupe(trace["family"]["label"] for trace in traces),
            "variant_labels": [trace["variant"].get("label") or "" for trace in traces],
            "action_count": sum(len(trace["actions"]) for trace in traces),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "verified_fix_count": outcome_counts["verified_fix"],
            "split_required": bool(case["split_required"] or len(traces) > 1),
            "must_preserve": [
                "device_identity", "trace_boundary", "action_outcome_pairing",
                "failed_or_temporary_results", "uncertainty", "evidence_anchor",
            ],
            "forbidden_inferences": [
                "temporary_recovery_as_verified_fix",
                "recommended_action_as_executed",
                "process_context_as_root_cause",
                "merge_parallel_faults",
            ],
        },
        "quality": {
            "query_origin": "frozen_source_only_gold_input",
            "focus": ["trace_split", "causal_reasoning", "evidence_strength", "uncertainty"],
            "leakage_control": "ground_truth_is_label_only_and_must_not_be_added_to_KG_v2",
            "graph_ingestion_allowed": False,
            "evidence_required": True,
            "negative_control": True,
        },
    }


def _coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(case[field]) for case in cases).items()))

    return {
        "case_count": len(cases),
        "track_counts": counts("evaluation_track"),
        "split_counts": counts("split"),
        "difficulty_counts": counts("difficulty"),
        "reasoning_mode_counts": counts("reasoning_mode"),
        "task_type_counts": counts("task_type"),
        "runtime_family_count": len({
            case["expected"]["family_id"]
            for case in cases if case["evaluation_track"] == "runtime_replay"
        }),
        "runtime_variant_count": len({
            case["expected"]["variant_id"]
            for case in cases if case["evaluation_track"] == "runtime_replay"
        }),
        "gold_source_case_count": sum(case["evaluation_track"] == "gold_trace_reasoning" for case in cases),
        "with_negative_control": sum(bool(case["quality"].get("negative_control")) for case in cases),
    }


def build_dataset(
    kg_root: str | Path = DEFAULT_KG_ROOT,
    gold_root: str | Path = DEFAULT_GOLD_ROOT,
) -> dict[str, Any]:
    kg_root = Path(kg_root)
    gold_root = Path(gold_root)
    model = KGV2ReadModel(str(kg_root))
    runtime_cases = _build_runtime_cases(model)
    gold_cases = [_gold_case(case) for case in load_gold_001_020(gold_root) if int(case["case_id"][-3:]) >= 11]
    cases = [*runtime_cases, *gold_cases]
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "kg_v2-read-quality-v1",
        "graph_revision": kg_v2_graph_revision(kg_root),
        "source_policy": {
            "runtime_replay": "active_KG_v2_objects_only",
            "gold_trace_reasoning": "source_only_inputs_with_frozen_labels",
            "gold_011_020_graph_ingestion": False,
        },
        "taxonomy": {
            "difficulty": {
                "easy": "显式症状或错误码，单一知识定位",
                "medium": "症状到变体再到首动作，或相近变体区分",
                "hard": "分支、失败动作、临时恢复、安全门或解决门",
                "expert": "长上下文、多设备/多故障轨迹、证据强度与不确定性",
            },
            "reasoning_mode": {
                "single_step": "一次定位：症状到 FaultVariant",
                "multi_hop_linear": "症状到变体到有序动作",
                "multi_hop_branch": "动作结果驱动 BranchRule、状态或安全门",
                "multi_trace_disambiguation": "从长上下文拆分并分别判断多条故障轨迹",
            },
        },
        "cases": cases,
    }
    dataset["coverage"] = _coverage(cases)
    return dataset


def validate_dataset(
    dataset: dict[str, Any],
    kg_root: str | Path = DEFAULT_KG_ROOT,
) -> dict[str, Any]:
    model = KGV2ReadModel(str(kg_root))
    issues: list[str] = []
    cases = dataset.get("cases") if isinstance(dataset.get("cases"), list) else []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version")
    if dataset.get("graph_revision") != kg_v2_graph_revision(Path(kg_root)):
        issues.append("graph_revision_mismatch")
    case_ids = [str(case.get("case_id") or "") for case in cases if isinstance(case, dict)]
    if not case_ids or len(case_ids) != len(set(case_ids)) or any(not case_id for case_id in case_ids):
        issues.append("case_ids_not_unique")

    for case in cases:
        case_id = str(case.get("case_id") or "")
        if case.get("difficulty") not in DIFFICULTIES:
            issues.append(f"{case_id}:difficulty")
        if case.get("reasoning_mode") not in REASONING_MODES:
            issues.append(f"{case_id}:reasoning_mode")
        if case.get("task_type") not in TASK_TYPES:
            issues.append(f"{case_id}:task_type")
        if not isinstance(case.get("hop_count"), int) or int(case["hop_count"]) < 1:
            issues.append(f"{case_id}:hop_count")
        query = str(case.get("query") or "")
        if not query or any(token in query for token in ("variant:", "action:", "trace:")):
            issues.append(f"{case_id}:query_leakage_or_empty")

        expected = case.get("expected") or {}
        if case.get("evaluation_track") == "runtime_replay":
            typed_ids = {
                "family_id": "FaultFamily",
                "variant_id": "FaultVariant",
                "plan_id": None,
                "first_action_id": "DiagnosticAction",
            }
            for field, object_type in typed_ids.items():
                object_id = str(expected.get(field) or "")
                if not object_id or (object_type and not model.has_object(object_id, object_type)):
                    issues.append(f"{case_id}:{field}")
            family_id = str(expected.get("family_id") or "")
            variant_id = str(expected.get("variant_id") or "")
            if model.has_object(family_id, "FaultFamily") and model.has_object(variant_id, "FaultVariant"):
                plan = model.compile_plan(family_id, variant_id)
                if expected.get("plan_id") != plan.plan_id:
                    issues.append(f"{case_id}:plan_id_stale")
                if expected.get("action_sequence") != [step.action_id for step in plan.steps]:
                    issues.append(f"{case_id}:action_sequence_stale")
            for evidence_id in expected.get("evidence_ids") or []:
                if not model.has_object(str(evidence_id), "EvidenceItem"):
                    issues.append(f"{case_id}:evidence:{evidence_id}")
        elif case.get("evaluation_track") == "gold_trace_reasoning":
            input_path = Path(str(case.get("source_input_ref") or ""))
            truth_path = Path(str(expected.get("truth_ref") or ""))
            if not input_path.is_file() or _sha256(input_path) != str(case["source_refs"][0].get("sha256") or ""):
                issues.append(f"{case_id}:source_input_hash")
            if not truth_path.is_file() or _sha256(truth_path) != str(expected.get("truth_sha256") or ""):
                issues.append(f"{case_id}:truth_hash")
            if case.get("quality", {}).get("graph_ingestion_allowed") is not False:
                issues.append(f"{case_id}:gold_ingestion_policy")
        else:
            issues.append(f"{case_id}:evaluation_track")

    coverage = _coverage(cases)
    for required in DIFFICULTIES:
        if coverage["difficulty_counts"].get(required, 0) == 0:
            issues.append(f"coverage:difficulty:{required}")
    for required in REASONING_MODES:
        if coverage["reasoning_mode_counts"].get(required, 0) == 0:
            issues.append(f"coverage:reasoning_mode:{required}")
    for required in ("variant_retrieval", "first_action", "branch_transition", "trace_split_and_outcome_reasoning"):
        if coverage["task_type_counts"].get(required, 0) == 0:
            issues.append(f"coverage:task_type:{required}")
    return {
        "schema_version": "debug_agent_system.kg_v2_read_eval.validation.v1",
        "dataset_id": dataset.get("dataset_id"),
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
        "coverage": coverage,
    }


def write_dataset(
    dataset: dict[str, Any],
    out: str | Path,
    report_out: str | Path,
    kg_root: str | Path = DEFAULT_KG_ROOT,
) -> dict[str, Any]:
    out = Path(out)
    report_out = Path(report_out)
    report = validate_dataset(dataset, kg_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-root", default=str(DEFAULT_KG_ROOT))
    parser.add_argument("--gold-root", default=str(DEFAULT_GOLD_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        dataset = json.loads(Path(args.out).read_text(encoding="utf-8"))
        report = validate_dataset(dataset, args.kg_root)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        dataset = build_dataset(args.kg_root, args.gold_root)
        report = write_dataset(dataset, args.out, args.report_out, args.kg_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
