from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.contracts import make_id, trim_text
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.materializer import KGV2Materializer
from debug_agent_system.knowledge_v2.validator import validate_graph


DEFAULT_ROOT = "data/kg_v2_sop_ipc_manual_draft"
DEFAULT_SUMMARY = "data/results/kg_v2_sop_ipc_manual_draft_summary.json"


def _hid(prefix: str, *parts: str) -> str:
    raw = ' | '.join(str(x or '') for x in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _family(
    label: str,
    summary: str,
    *,
    category: str,
    subsystem: str,
    scenario: str,
    source_kind: str,
    keywords: list[str] | None = None,
    escalation_target: str = "",
) -> dict[str, Any]:
    return {
        "family_id": make_id("family", label),
        "label": trim_text(label, 40),
        "summary": trim_text(summary, 80),
        "category": category,
        "subsystem": trim_text(subsystem, 40),
        "scenario": trim_text(scenario, 60),
        "keywords": list(keywords or []),
        "source_kind": source_kind,
        "escalation_target": trim_text(escalation_target, 40),
    }


def _variant(
    family_id: str,
    label: str,
    summary: str,
    *,
    error_phase: str,
    owner_context: str,
    source_kind: str,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "variant_id": _hid("variant", family_id, label, error_phase, owner_context),
        "family_id": family_id,
        "label": trim_text(label, 60),
        "summary": trim_text(summary, 180),
        "equipment_type": "",
        "site": "",
        "software_version": "",
        "error_phase": trim_text(error_phase, 40),
        "owner_context": trim_text(owner_context, 80),
        "escalation_target": "",
        "keywords": list(keywords or []),
        "_source_kind": source_kind,
    }


def _action(
    family_id: str,
    variant_id: str,
    label: str,
    summary: str,
    *,
    action_role: str,
    step_order: int,
    source_kind: str,
    destructive: bool = False,
    high_cost: bool = False,
) -> dict[str, Any]:
    return {
        "action_id": _hid("action", family_id, variant_id, str(step_order), label, action_role),
        "family_id": family_id,
        "variant_id": variant_id,
        "label": trim_text(label, 60),
        "summary": trim_text(summary, 180),
        "action_role": action_role,
        "step_order": step_order,
        "destructive": destructive,
        "high_cost": high_cost,
        "source_kind": source_kind,
    }


def _required(
    family_id: str,
    variant_id: str,
    slot: str,
    question: str,
    why_required: str,
    *,
    condition: str,
    blocks: list[str],
    priority: str,
    evidence_ids: list[str],
) -> dict[str, Any]:
    return {
        "required_info_id": _hid("required-info", family_id, variant_id, slot, question, condition),
        "family_id": family_id,
        "variant_id": variant_id,
        "slot": slot,
        "question": trim_text(question, 100),
        "why_required": trim_text(why_required, 160),
        "condition": trim_text(condition, 120),
        "blocks": [trim_text(x, 120) for x in blocks],
        "priority": priority,
        "evidence_ids": evidence_ids,
    }


def _outcome(
    family_id: str,
    variant_id: str,
    action_id: str,
    source_case_id: str,
    summary: str,
    *,
    outcome_type: str,
    evidence_ids: list[str],
    root_cause_summary: str = "",
    destructive: bool = False,
    high_cost: bool = False,
) -> dict[str, Any]:
    return {
        "outcome_id": _hid("outcome", family_id, variant_id, action_id, outcome_type, summary),
        "family_id": family_id,
        "variant_id": variant_id,
        "action_id": action_id,
        "outcome_type": outcome_type,
        "summary": trim_text(summary, 200),
        "source_case_id": source_case_id,
        "evidence_ids": evidence_ids,
        "high_cost": high_cost,
        "destructive": destructive,
        "root_cause_summary": trim_text(root_cause_summary, 120),
    }


def _trace(
    family_id: str,
    variant_id: str,
    source_case_id: str,
    summary: str,
    *,
    recommended_action_ids: list[str],
    actual_action_ids: list[str],
    evidence_ids: list[str],
) -> dict[str, Any]:
    return {
        "trace_id": _hid("trace", family_id, variant_id, source_case_id, summary),
        "family_id": family_id,
        "variant_id": variant_id,
        "source_case_id": source_case_id,
        "summary": trim_text(summary, 160),
        "recommended_action_ids": recommended_action_ids,
        "actual_action_ids": actual_action_ids,
        "evidence_ids": evidence_ids,
    }


def _case(title: str, summary: str, *, source_kind: str, source_ref: str) -> dict[str, Any]:
    return {
        "case_id": _hid("case", source_kind, source_ref, title, summary),
        "source_kind": source_kind,
        "title": trim_text(title, 80),
        "summary": trim_text(summary, 240),
        "source_ref": trim_text(source_ref, 200),
        "approved": True,
    }


def _evidence(title: str, summary: str, *, source_kind: str, external_id: str, payload_ref: str) -> dict[str, Any]:
    return {
        "evidence_id": _hid("evidence", source_kind, external_id, title, summary),
        "source_kind": source_kind,
        "external_id": trim_text(external_id, 120),
        "title": trim_text(title, 80),
        "summary": trim_text(summary, 500),
        "payload_ref": trim_text(payload_ref, 200),
    }


def build_manual_graph() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    objects: dict[str, list[dict[str, Any]]] = {
        "FaultFamily": [],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "DecisionPolicy": [],
        "EvidenceItem": [],
        "SourceCase": [],
    }
    relations: list[dict[str, Any]] = []

    def add_relation(a: str, b: str, rel: str) -> None:
        relations.append({"from": a, "to": b, "relation": rel})

    def add_variant_bundle(
        family: dict[str, Any],
        *,
        variant: dict[str, Any],
        case: dict[str, Any],
        evidence: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
        required_info: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> None:
        objects["FaultVariant"].append(variant)
        objects["SourceCase"].append(case)
        objects["EvidenceItem"].extend(evidence)
        objects["DiagnosticAction"].extend(actions)
        objects["ActionOutcome"].extend(outcomes)
        objects["RequiredInfoSpec"].extend(required_info)
        objects["DiagnosticTrace"].append(trace)

        add_relation(family["family_id"], variant["variant_id"], "has_variant")
        add_relation(family["family_id"], trace["trace_id"], "has_trace")
        add_relation(case["case_id"], variant["variant_id"], "supports")
        add_relation(case["case_id"], trace["trace_id"], "supports")
        for ev in evidence:
            add_relation(ev["evidence_id"], case["case_id"], "evidences")
        for req in required_info:
            add_relation(variant["variant_id"], req["required_info_id"], "has_required_info")
            add_relation(case["case_id"], req["required_info_id"], "supports")
            for ev in evidence:
                add_relation(ev["evidence_id"], req["required_info_id"], "evidences")
        for act in actions:
            add_relation(trace["trace_id"], act["action_id"], "used_action")
        for outcome in outcomes:
            add_relation(variant["variant_id"], outcome["outcome_id"], "has_outcome")
            add_relation(case["case_id"], outcome["outcome_id"], "supports")
            add_relation(outcome["outcome_id"], outcome["action_id"], "outcome_of")
            for ev in evidence:
                add_relation(ev["evidence_id"], outcome["outcome_id"], "evidences")

    # 1. 用户配置加载失败
    fam = _family(
        "用户配置加载失败",
        "主程序或复判站初始化阶段无法加载用户配置。",
        category="系统与软件异常",
        subsystem="主程序配置/复判站配置",
        scenario="初始化阶段配置加载失败",
        source_kind="sop",
        keywords=["conf", "user.cfg.toml", "配置加载失败", "初始化失败"],
    )
    objects["FaultFamily"].append(fam)
    variant = _variant(
        fam["family_id"],
        "更换工控机后 user.cfg.toml 异常导致配置加载失败",
        "更换工控机后主程序报警加载用户配置失败，怀疑 conf 残留或 user.cfg.toml 异常。",
        error_phase="初始化阶段",
        owner_context="SOP:1.1.3.1.2",
        source_kind="sop",
        keywords=["user.cfg.toml", "conf", "更换工控机"],
    )
    case = _case("加载用户配置失败处理思路", "清空 conf、检查 user.cfg.toml、使用最近一次正常配置替换验证。", source_kind="sop", source_ref="1.1.3.1.2")
    ev = [_evidence("SOP 1.1.3.1.2", "删除 conf 文件夹内所有文件后重启；客户08案例中，用最近一次诊断日志的 user.cfg.toml 替换后恢复。", source_kind="sop", external_id="1.1.3.1.2", payload_ref="异常处理 - 标准操作流程（SOP）")]
    a1 = _action(fam["family_id"], variant["variant_id"], "备份并清空 conf 目录", "操作前先备份 conf，然后删除 conf 文件夹内所有文件并重启软件。", action_role="change", step_order=1, source_kind="sop", destructive=True)
    a2 = _action(fam["family_id"], variant["variant_id"], "检查 user.cfg.toml 是否为空或损坏", "若清空 conf 后仍失败，重点检查 user.cfg.toml 是否为空白或异常。", action_role="inspect", step_order=2, source_kind="sop")
    a3 = _action(fam["family_id"], variant["variant_id"], "用最近一次诊断日志中的配置文件替换验证", "使用最近一次正常诊断日志中的 user.cfg.toml 替换，并修改一项配置再改回后重启验证。", action_role="verify", step_order=3, source_kind="sop")
    r1 = _required(fam["family_id"], variant["variant_id"], "program_file", "请提供 conf 目录与 user.cfg.toml 当前内容。", "需要判断是配置残留、空白文件还是配置损坏。", condition="初始化阶段报加载用户配置失败", blocks=[a1["label"], a2["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]])
    r2 = _required(fam["family_id"], variant["variant_id"], "log_package", "请提供最近一次正常诊断日志及其中的配置文件。", "需要用已知正常配置回填并验证是否恢复。", condition="清空 conf 后仍失败", blocks=[a3["label"]], priority="medium", evidence_ids=[ev[0]["evidence_id"]])
    o1 = _outcome(fam["family_id"], variant["variant_id"], a1["action_id"], case["case_id"], "清空 conf 是 SOP 默认修复起点，用于清除残留配置。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]])
    o2 = _outcome(fam["family_id"], variant["variant_id"], a3["action_id"], case["case_id"], "在客户08案例中，替换正常 user.cfg.toml 后已恢复进入软件。", outcome_type="verified_fix", evidence_ids=[ev[0]["evidence_id"]], root_cause_summary="user.cfg.toml 异常或残留配置导致加载失败")
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先清空 conf 重启，再查 user.cfg.toml，最后用历史正常配置替换验证。", recommended_action_ids=[a1["action_id"], a2["action_id"], a3["action_id"]], actual_action_ids=[a1["action_id"], a3["action_id"]], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=[a1,a2,a3], outcomes=[o1,o2], required_info=[r1,r2], trace=t)

    # 2. 相机拍摄失败
    fam = _family(
        "相机拍摄失败",
        "初始化或拍摄过程中出现拍摄失败、操作失败或拍摄卡顿。",
        category="硬件与运控",
        subsystem="相机/采集链路",
        scenario="初始化或检测阶段拍摄失败",
        source_kind="sop",
        keywords=["拍摄失败", "相机事件超时", "CXP", "SDK", "残帧"],
    )
    objects["FaultFamily"].append(fam)
    # variant init
    variant = _variant(fam["family_id"], "初始化阶段提示拍摄失败", "软件启动初始化过程中直接提示拍摄失败。", error_phase="初始化阶段", owner_context="SOP:2.1.1", source_kind="sop", keywords=["初始化", "拍摄失败", "Galaxy Viewer", "MVS"])
    case = _case("初始化阶段提示拍摄失败", "优先用相机软件检查拍摄，再查供电/IP/光控/自动曝光。", source_kind="sop", source_ref="2.1.1")
    ev = [_evidence("SOP 2.1.1", "2D 优先 Galaxy Viewer，3D 优先 MVS；检查相机上电、2D 相机 IP、光源控制器、3D 自动曝光状态。", source_kind="sop", external_id="2.1.1", payload_ref="异常处理 - 标准操作流程（SOP）")]
    a1 = _action(fam["family_id"], variant["variant_id"], "使用相机软件检查拍摄是否正常", "2D 优先 Galaxy Viewer，3D 优先 MVS。", action_role="inspect", step_order=1, source_kind="sop")
    a2 = _action(fam["family_id"], variant["variant_id"], "检查相机上电状态", "确认相机供电与指示状态正常。", action_role="inspect", step_order=2, source_kind="sop")
    a3 = _action(fam["family_id"], variant["variant_id"], "检查 2D 相机 IP 配置", "2D 相机初始化失败优先排除 IP 错配。", action_role="inspect", step_order=3, source_kind="sop")
    a4 = _action(fam["family_id"], variant["variant_id"], "检查光源控制器状态", "2D 初始化失败时检查光控链路。", action_role="inspect", step_order=4, source_kind="sop")
    a5 = _action(fam["family_id"], variant["variant_id"], "检查 3D 自动曝光是否关闭", "3D 相机不应处于自动曝光模式。", action_role="inspect", step_order=5, source_kind="sop")
    r1 = _required(fam["family_id"], variant["variant_id"], "ip_config", "请提供相机 IP 配置与网络连接状态。", "初始化阶段需要优先排除相机 IP 识别不到。", condition="2D 初始化提示拍摄失败", blocks=[a3["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]])
    r2 = _required(fam["family_id"], variant["variant_id"], "log_package", "请提供初始化阶段的主程序日志与运控日志。", "需要区分相机未上电、IP 异常、光控异常还是 3D 曝光配置异常。", condition="初始化阶段提示拍摄失败", blocks=[a1["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]])
    o1 = _outcome(fam["family_id"], variant["variant_id"], a1["action_id"], case["case_id"], "优先用相机侧工具区分软件链路问题与相机本体问题。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]])
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先用相机软件验证拍摄，再查供电/IP/光控/曝光配置。", recommended_action_ids=[a1["action_id"], a2["action_id"], a3["action_id"], a4["action_id"], a5["action_id"]], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=[a1,a2,a3,a4,a5], outcomes=[o1], required_info=[r1,r2], trace=t)
    # variant run
    variant = _variant(fam["family_id"], "运行中拍摄失败或拍摄卡顿", "编程或检测过程中出现拍摄失败、操作失败、残帧、事件超时或拍摄越来越慢。", error_phase="编程/检测阶段", owner_context="SOP:2.1.2", source_kind="sop", keywords=["残帧", "block discarded", "事件超时", "CXP", "停稳耗时"])
    case = _case("运行中拍摄失败或拍摄卡顿", "按故障阶段分流：开始即失败、拍摄中失败、拍大板越来越慢。", source_kind="sop", source_ref="2.1.2")
    ev = [_evidence("SOP 2.1.2", "3D 检测头复位、检查 cyclops-lighter 版本、紧固 CXP 线；2D 残帧关注 SDK 版本与网口配置；拍大板越来越慢时看 Camera steady costs。", source_kind="sop", external_id="2.1.2", payload_ref="异常处理 - 标准操作流程（SOP）")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "执行 3D 检测头复位", "3D 拍摄开始即失败时优先尝试复位。", action_role="change", step_order=1, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "检查 cyclops-lighter 版本与光源亮度读写", "确认版本、亮度读写、是否需要重新烧录。", action_role="inspect", step_order=2, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "紧固或重新插拔 CXP 线", "3D 图片缺失或拍摄开始无图返回时重点检查 CXP 线，断电后操作。", action_role="change", step_order=3, source_kind="sop", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "检查相机 SDK 版本", "2D 残帧时检查是否高于等于 2.4.2503.9201。", action_role="inspect", step_order=4, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "检查网口配置并开启巨帧", "2D block discarded 或残帧时重点排查网口配置。", action_role="inspect", step_order=5, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "升级相机固件", "2D 相机事件超时时联系 FAE 升级固件。", action_role="change", step_order=6, source_kind="sop", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "设置进程绑核", "若日志报光源切换事件超时，可尝试用绑核缓解。", action_role="change", step_order=7, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "检查运控配置项 timeout_wait_for_camera_event_in_millisec", "3D 曝光结束超时时配置应为 10000。", action_role="inspect", step_order=8, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "查看相机停稳耗时并联系 FAE 排查伺服参数", "拍大板越来越慢时关注 Camera steady costs。", action_role="observe", step_order=9, source_kind="sop"),
    ]
    r1 = _required(fam["family_id"], variant["variant_id"], "log_package", "请提供主程序日志、运控日志，以及具体报错片段。", "需要按残帧 / block discarded / event timeout / fov 图片缺失进行分型。", condition="运行中拍摄失败或拍摄卡顿", blocks=[actions[3]["label"], actions[4]["label"], actions[7]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]])
    r2 = _required(fam["family_id"], variant["variant_id"], "software_version", "请提供相机 SDK 版本、相机固件版本、cyclops-lighter 版本。", "需要确认是否命中已知版本问题。", condition="运行中拍摄失败或拍摄卡顿", blocks=[actions[1]["label"], actions[3]["label"], actions[5]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]])
    r3 = _required(fam["family_id"], variant["variant_id"], "ip_config", "请提供相机网口配置、巨帧配置和相关网络截图。", "2D 残帧与 block discarded 经常与网口配置相关。", condition="2D 拍摄失败 / block discarded / 残帧", blocks=[actions[4]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]])
    r4 = _required(fam["family_id"], variant["variant_id"], "repro_steps", "请说明故障发生在拍摄开始即失败、拍摄中失败、还是拍大板逐渐变慢。", "不同阶段对应不同动作链与负责人。", condition="运行中拍摄失败或拍摄卡顿", blocks=[actions[0]["label"], actions[8]["label"]], priority="medium", evidence_ids=[ev[0]["evidence_id"]])
    outcomes = [
        _outcome(fam["family_id"], variant["variant_id"], actions[2]["action_id"], case["case_id"], "3D FOV 图片缺失或拍摄开始无图返回时，CXP 线是重点怀疑路径。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]], destructive=True),
        _outcome(fam["family_id"], variant["variant_id"], actions[3]["action_id"], case["case_id"], "2D 残帧的已知原因之一是旧版本 SDK 缺少重传机制。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
        _outcome(fam["family_id"], variant["variant_id"], actions[5]["action_id"], case["case_id"], "事件超时场景中，升级固件属于需 FAE 配合的修复路径。", outcome_type="pending_validation", evidence_ids=[ev[0]["evidence_id"]], destructive=True),
        _outcome(fam["family_id"], variant["variant_id"], actions[8]["action_id"], case["case_id"], "拍摄越来越慢与伺服停稳耗时、流控误触发有关。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "按故障阶段分流：开始即失败看复位/光源/CXP，拍摄中失败看 SDK/网口/固件/timeout，大板变慢看停稳耗时和伺服参数。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=[r1,r2,r3,r4], trace=t)

    # 3. 光源初始化失败
    fam = _family("光源初始化失败", "初始化阶段光源模块或光控链路初始化失败。", category="硬件与运控", subsystem="光源/光控链路", scenario="初始化阶段光源失败", source_kind="sop", keywords=["光控", "防火墙", "ARM", "断电重启"])
    objects["FaultFamily"].append(fam)
    variant = _variant(fam["family_id"], "初始化阶段光源问题导致软件无法继续启动", "光源初始化失败时，需先做断电、光控重插、网络与防火墙检查，再反馈项目群联系硬件。", error_phase="初始化阶段", owner_context="SOP:1.4.1.1.2", source_kind="sop", keywords=["光源问题", "光控", "ARM", "防火墙"])
    case = _case("光源初始化失败处理链路", "断电重启、插拔光控、查网络与防火墙、收集日志交硬件。", source_kind="sop", source_ref="1.4.1.1.2")
    ev = [_evidence("SOP 1.4.1.1.2", "将软件全部退出，断电 1 分钟后重启；2D 设备插拔光控；查看系统 IP 连接、ARM 连接、防火墙；收集日志并联系硬件。", source_kind="sop", external_id="1.4.1.1.2", payload_ref="异常处理 - 标准操作流程（SOP）")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "软件全部退出并断电 1 分钟后重启", "先执行完整断电重启，排除临时初始化异常。", action_role="change", step_order=1, source_kind="sop", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "插拔 2D 设备光控", "从设备前面屏幕下方开门后，对光控链路进行重新插拔。", action_role="change", step_order=2, source_kind="sop", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "检查系统 IP / ARM 连接 / 防火墙", "确认网络链路和系统安全策略未阻断光控连接。", action_role="inspect", step_order=3, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "收集日志并反馈项目群联系硬件", "若仍未恢复，保留日志并交给硬件线继续处理。", action_role="escalate", step_order=4, source_kind="sop"),
    ]
    reqs = [
        _required(fam["family_id"], variant["variant_id"], "log_package", "请提供初始化失败时的主程序日志、运控日志和光控相关日志。", "需要判断是光控链路异常、网络异常还是更底层硬件问题。", condition="初始化阶段光源失败", blocks=[actions[2]["label"], actions[3]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "ip_config", "请提供网络连接、ARM 连接与防火墙状态截图。", "需要排除网络与系统策略造成的光源初始化失败。", condition="初始化阶段光源失败", blocks=[actions[2]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    outcomes = [
        _outcome(fam["family_id"], variant["variant_id"], actions[0]["action_id"], case["case_id"], "断电重启是 SOP 给出的第一反应动作。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]], destructive=True),
        _outcome(fam["family_id"], variant["variant_id"], actions[3]["action_id"], case["case_id"], "若基础链路排查无果，需进入硬件升级路径。", outcome_type="pending_validation", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "断电重启 → 插拔光控 → 查网络/防火墙 → 收集日志交硬件。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=reqs, trace=t)

    # 4. 运控卡初始化异常
    fam = _family("运控卡初始化异常", "初始化阶段卡在运动控制卡或运控闪退。", category="硬件与运控", subsystem="运控卡/初始化链路", scenario="初始化阶段卡在运控卡", source_kind="sop", keywords=["运控", "100M", "Speed & Duplex", "网速异常"])
    objects["FaultFamily"].append(fam)
    variant = _variant(fam["family_id"], "初始化卡在运动控制卡，疑似网口速率异常", "运控日志无明显异常，但初始化卡在运动控制卡或运控闪退，需要重点检查运控卡网口速率。", error_phase="初始化阶段", owner_context="SOP:1.4.1.1.4", source_kind="sop", keywords=["100M", "Speed & Duplex", "运控闪退"])
    case = _case("运控卡初始化异常处理链路", "先看运控日志，再确认网口速率，最后调整双工模式。", source_kind="sop", source_ref="1.4.1.1.4")
    ev = [_evidence("SOP 1.4.1.1.4", "查看运控日志是否有异常；运控卡需要 100M；若网速异常则进入网卡属性调整 Speed & Duplex。", source_kind="sop", external_id="1.4.1.1.4", payload_ref="异常处理 - 标准操作流程（SOP）")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "检查运控日志是否有异常", "先看运控日志是否已有明确报错。", action_role="inspect", step_order=1, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "检查网络适配器网速是否为 100M", "运控卡需要 100M，优先检查网口速率。", action_role="inspect", step_order=2, source_kind="sop"),
        _action(fam["family_id"], variant["variant_id"], "在网卡属性中调整 Speed & Duplex", "若网速异常，进入相关网口属性，调整连接速度和双工模式。", action_role="change", step_order=3, source_kind="sop"),
    ]
    reqs = [
        _required(fam["family_id"], variant["variant_id"], "ip_config", "请提供运控卡相关网口的速率、双工模式和网络适配器截图。", "需要确认运控卡是否跑在 100M 正常模式。", condition="初始化卡在运动控制卡或运控闪退", blocks=[actions[1]["label"], actions[2]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "log_package", "请提供初始化阶段运控日志。", "需要判断是否真的是无异常日志场景，还是已存在明确错误码。", condition="初始化卡在运动控制卡或运控闪退", blocks=[actions[0]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    outcomes = [_outcome(fam["family_id"], variant["variant_id"], actions[1]["action_id"], case["case_id"], "该类问题的 SOP 主判断点是运控卡网速异常。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]])]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先看运控日志，再确认网口速率，最后调整双工模式。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=reqs, trace=t)

    # 5. 工控机蓝屏
    fam = _family("工控机蓝屏", "系统运行中出现蓝屏并可见错误代码或驱动文件名。", category="系统与软件异常", subsystem="工控机/Windows 内核", scenario="运行中蓝屏", source_kind="hybrid", keywords=["蓝屏", "dmp", "WinDbg", "错误代码", ".sys"])
    objects["FaultFamily"].append(fam)
    # generic bsod
    variant = _variant(fam["family_id"], "运行中蓝屏并显示错误代码或 .sys 文件", "系统突然蓝屏，需先保留错误码，再通过事件查看器与 dmp 文件做进一步定位。", error_phase="运行中", owner_context="MANUAL:蓝屏", source_kind="hybrid", keywords=["错误代码", ".sys", "Minidump", "WinDbg"])
    case = _case("工控机蓝屏通用排查", "先拍照记录屏幕，再查事件查看器，再分析 Minidump，然后按错误码走驱动/系统/硬件分支。", source_kind="manual_review", source_ref="IPC-1")
    ev = [_evidence("工控机异常手册·蓝屏", "立即拍照记录屏幕上的错误代码和失败文件名；重启后可通过事件查看器补证据；高级分析可用 Minidump 和 WinDbg。", source_kind="manual_review", external_id="IPC-1", payload_ref="工控机异常(蓝屏&重启&死机）手册")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "拍照记录蓝屏错误代码与失败文件名", "第一时间保留错误代码、二维码、.sys 文件名。", action_role="collect", step_order=1, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "通过事件查看器查看崩溃时间点日志", "若未记录屏幕，重启后从 Windows 日志 -> 系统中找关键日志。", action_role="inspect", step_order=2, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "分析 Minidump 转储文件", "分析 C:\\Windows\\Minidump\\ 下的 .dmp 文件，使用 WinDbg 定位故障模块。", action_role="inspect", step_order=3, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "按错误代码回退驱动或修复系统文件", "针对固定错误代码，回退/卸载驱动或执行 sfc /scannow。", action_role="change", step_order=4, source_kind="hybrid", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "转向内存/硬盘/显卡/网卡硬件排查", "若证据指向硬件不稳定，则进入硬件分线排查。", action_role="verify", step_order=5, source_kind="hybrid"),
    ]
    reqs = [
        _required(fam["family_id"], variant["variant_id"], "dmp_package", "请提供 C:\\Windows\\Minidump\\ 下的 .dmp 转储文件。", "蓝屏场景的核心证据来自转储文件定位故障模块。", condition="蓝屏且有错误代码", blocks=[actions[2]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "error_message", "请提供蓝屏错误代码、失败文件名和照片。", "需要先按错误码把问题落到驱动、系统文件或硬件方向。", condition="蓝屏且现场仍能看到屏幕", blocks=[actions[0]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "log_package", "请导出事件查看器中崩溃时间点附近的系统日志。", "若现场未及时拍照，事件日志是补证据的主要路径。", condition="蓝屏后已重启", blocks=[actions[1]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    outcomes = [
        _outcome(fam["family_id"], variant["variant_id"], actions[0]["action_id"], case["case_id"], "这是整个蓝屏链路的关键证据入口。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
        _outcome(fam["family_id"], variant["variant_id"], actions[2]["action_id"], case["case_id"], "WinDbg / dmp 是高级定位主手段。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
        _outcome(fam["family_id"], variant["variant_id"], actions[3]["action_id"], case["case_id"], "驱动回退或系统文件修复后仍需在后续复现/观察中确认是否真正消除蓝屏。", outcome_type="pending_validation", evidence_ids=[ev[0]["evidence_id"]], destructive=True),
    ]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先保留蓝屏证据，再查事件日志和 dmp，随后按错误码走驱动/系统/硬件分支。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=reqs, trace=t)
    # loop
    variant = _variant(fam["family_id"], "蓝屏后无限重启循环", "蓝屏后反复自动重启，无法稳定进入桌面，需要先进入安全模式中断循环。", error_phase="启动后循环重启", owner_context="MANUAL:无限蓝屏重启循环", source_kind="hybrid", keywords=["安全模式", "重启循环", "错误代码固定", "内存", "电源", "主板"])
    case = _case("无限蓝屏重启循环排查", "先通过自动修复进入安全模式，再根据错误码固定与否分流到软件修复或硬件优先路径。", source_kind="manual_review", source_ref="IPC-2")
    ev = [_evidence("工控机异常手册·无限蓝屏重启循环", "先进入安全模式中断循环；固定错误码走驱动/系统修复，不固定或无代码时优先排查内存、电源、主板。", source_kind="manual_review", external_id="IPC-2", payload_ref="工控机异常(蓝屏&重启&死机）手册")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "多次强制关机触发自动修复并进入安全模式", "通过自动修复路径进入安全模式或带网络安全模式。", action_role="change", step_order=1, source_kind="hybrid", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "在安全模式下按固定错误代码做针对性处理", "若错误代码固定，则卸载驱动、卸载软件或执行系统修复。", action_role="inspect", step_order=2, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "转向内存/电源/主板硬件优先排查", "若错误代码不固定或没有代码，这是严重硬件不稳定信号。", action_role="verify", step_order=3, source_kind="hybrid"),
    ]
    reqs = [
        _required(fam["family_id"], variant["variant_id"], "error_message", "请确认循环中蓝屏错误代码是否固定。", "固定与不固定将决定走软件修复还是硬件优先路径。", condition="蓝屏后无限重启循环", blocks=[actions[1]["label"], actions[2]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "memory_cpu_test", "请提供内存条测试、电源替换和主板状态检查结果。", "循环蓝屏且错误不固定时，硬件稳定性是最高优先级。", condition="错误代码不固定或无代码", blocks=[actions[2]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    outcomes = [
        _outcome(fam["family_id"], variant["variant_id"], actions[0]["action_id"], case["case_id"], "无限重启循环的首要目标是获得稳定环境。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]], destructive=True),
        _outcome(fam["family_id"], variant["variant_id"], actions[2]["action_id"], case["case_id"], "错误码不固定时，应优先怀疑硬件不稳定。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先进入安全模式中断循环，再根据错误码固定与否分流到软件修复或硬件排查。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=reqs, trace=t)

    # 6. 工控机异常重启
    fam = _family("工控机异常重启", "系统运行中无提示自动重启。", category="系统与软件异常", subsystem="工控机/系统运行稳定性", scenario="运行中自动重启", source_kind="hybrid", keywords=["事件 6008", "自动重新启动", "温度", "电源", "干净启动"])
    objects["FaultFamily"].append(fam)
    variant = _variant(fam["family_id"], "运行中无提示自动重启", "工控机在运行时直接重启，需区分未显示蓝屏的保护性重启与电源/温度/硬件问题。", error_phase="运行中", owner_context="MANUAL:重启", source_kind="hybrid", keywords=["6008", "自动重启", "干净启动", "温度", "电源"])
    case = _case("工控机异常重启排查", "先让蓝屏显性化，再查事件日志，随后排后台冲突、温度和电源。", source_kind="manual_review", source_ref="IPC-3")
    ev = [_evidence("工控机异常手册·重启", "取消自动重新启动让真实错误显现；重点查看事件 ID 6008；做干净启动；监控 CPU/GPU 温度；使用替换法测试电源。", source_kind="manual_review", external_id="IPC-3", payload_ref="工控机异常(蓝屏&重启&死机）手册")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "取消系统自动重新启动", "让下次故障时尽量显示蓝屏而不是直接重启。", action_role="change", step_order=1, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "查看事件查看器中重启时间点日志", "重点关注 6008 及其前后的关联错误。", action_role="inspect", step_order=2, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "执行干净启动排除后台冲突", "通过 msconfig 禁用非 Microsoft 服务与启动项。", action_role="inspect", step_order=3, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "监控 CPU/GPU 温度", "排查过热触发保护重启。", action_role="observe", step_order=4, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "替换法测试电源", "电源老化或功率不足是最常见硬件原因之一。", action_role="verify", step_order=5, source_kind="hybrid"),
    ]
    reqs = [
        _required(fam["family_id"], variant["variant_id"], "log_package", "请提供重启时间点附近的系统日志与事件查看器导出。", "需要确认是否为保护性重启及其前序错误。", condition="运行中无提示自动重启", blocks=[actions[1]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "environment", "请提供市电、接地、现场温度与散热状态信息。", "重启常与过热、接地不良、市电波动有关。", condition="运行中无提示自动重启", blocks=[actions[3]["label"], actions[4]["label"]], priority="medium", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "memory_cpu_test", "请提供电源替换测试、CPU/GPU 温度监控结果。", "需要区分供电问题和过热保护。", condition="运行中无提示自动重启", blocks=[actions[3]["label"], actions[4]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    outcomes = [
        _outcome(fam["family_id"], variant["variant_id"], actions[0]["action_id"], case["case_id"], "目标是把隐藏的蓝屏重新显性化。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
        _outcome(fam["family_id"], variant["variant_id"], actions[4]["action_id"], case["case_id"], "电源是异常重启的高优先级怀疑对象。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先让蓝屏显性化，再看事件日志，随后排后台冲突、温度和电源。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=reqs, trace=t)

    # 7. 工控机死机
    fam = _family("工控机死机", "系统完全卡死，键鼠无响应。", category="系统与软件异常", subsystem="工控机/系统运行稳定性", scenario="运行中完全卡死", source_kind="hybrid", keywords=["Ctrl+Alt+Del", "完全无响应", "任务管理器", "散热", "内存"])
    objects["FaultFamily"].append(fam)
    variant = _variant(fam["family_id"], "屏幕定格且键盘鼠标无响应", "完全死机时需先区分资源占满的软件层问题与核心硬件无响应。", error_phase="运行中", owner_context="MANUAL:死机", source_kind="hybrid", keywords=["死机", "键鼠无响应", "任务管理器", "散热", "内存"])
    case = _case("工控机死机排查", "先判断能否调出安全界面，能则优先软件排查，不能则直接转硬件稳定性排查。", source_kind="manual_review", source_ref="IPC-4")
    ev = [_evidence("工控机异常手册·死机", "先尝试 Ctrl + Alt + Del；若仍完全无响应，则硬件故障概率很大，应系统性检查散热、内存、主板和电源。", source_kind="manual_review", external_id="IPC-4", payload_ref="工控机异常(蓝屏&重启&死机）手册")]
    actions = [
        _action(fam["family_id"], variant["variant_id"], "尝试按 Ctrl + Alt + Del", "先判断是否还能调出安全界面。", action_role="inspect", step_order=1, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "在任务管理器结束高占用或未响应进程", "若能进任务管理器，则优先结束异常进程。", action_role="change", step_order=2, source_kind="hybrid"),
        _action(fam["family_id"], variant["variant_id"], "进入安全模式卸载最近安装的软件/驱动", "从软件层排除最近变更带来的资源锁死。", action_role="change", step_order=3, source_kind="hybrid", destructive=True),
        _action(fam["family_id"], variant["variant_id"], "系统性检查散热、内存、主板和电源", "完全无响应时需转入硬件稳定性排查。", action_role="verify", step_order=4, source_kind="hybrid"),
    ]
    reqs = [
        _required(fam["family_id"], variant["variant_id"], "repro_steps", "请说明死机前是否有高负载操作、特定软件或固定触发步骤。", "需要判断是否存在可归因的软件触发。", condition="运行中完全卡死", blocks=[actions[1]["label"], actions[2]["label"]], priority="medium", evidence_ids=[ev[0]["evidence_id"]]),
        _required(fam["family_id"], variant["variant_id"], "memory_cpu_test", "请提供死机前后的 CPU/内存占用、温度、内存稳定性测试结果。", "需要区分资源耗尽与硬件停止响应。", condition="运行中完全卡死", blocks=[actions[3]["label"]], priority="high", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    outcomes = [
        _outcome(fam["family_id"], variant["variant_id"], actions[0]["action_id"], case["case_id"], "能否弹出安全界面决定先走软件还是硬件路径。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
        _outcome(fam["family_id"], variant["variant_id"], actions[3]["action_id"], case["case_id"], "完全无响应场景中，硬件概率显著升高。", outcome_type="diagnostic_method", evidence_ids=[ev[0]["evidence_id"]]),
    ]
    t = _trace(fam["family_id"], variant["variant_id"], case["case_id"], "先判断能否调出安全界面，能则优先软件排查，不能则直接硬件稳定性排查。", recommended_action_ids=[x["action_id"] for x in actions], actual_action_ids=[], evidence_ids=[ev[0]["evidence_id"]])
    add_variant_bundle(fam, variant=variant, case=case, evidence=ev, actions=actions, outcomes=outcomes, required_info=reqs, trace=t)

    # policies
    for family in objects["FaultFamily"]:
        family_id = family["family_id"]
        fam_actions = [a for a in objects["DiagnosticAction"] if a["family_id"] == family_id]
        fam_outcomes = [o for o in objects["ActionOutcome"] if o["family_id"] == family_id]
        fam_traces = [t for t in objects["DiagnosticTrace"] if t["family_id"] == family_id]
        ineffective = sorted({o["action_id"] for o in fam_outcomes if o["outcome_type"] in {"ineffective", "context_not_root_cause"}})
        high_cost = sorted({o["action_id"] for o in fam_outcomes if o.get("high_cost") or o.get("destructive")} | {a["action_id"] for a in fam_actions if a.get("high_cost") or a.get("destructive")})
        ordered = [a["action_id"] for a in sorted(fam_actions, key=lambda x: (int(x.get("step_order") or 999), x.get("label") or ""))]
        policy = {
            "policy_id": make_id("policy", family_id),
            "family_id": family_id,
            "source_trace_ids": [t["trace_id"] for t in fam_traces],
            "source_outcome_ids": [o["outcome_id"] for o in fam_outcomes],
            "ordered_action_ids": ordered,
            "ineffective_action_ids": ineffective,
            "high_cost_action_ids": high_cost,
            "deterministic_recompute": True,
        }
        objects["DecisionPolicy"].append(policy)
        add_relation(policy["policy_id"], family_id, "for_family")

    return objects, relations


def write_manual_graph(root: str | Path = DEFAULT_ROOT, summary_out: str | Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    objects, relations = build_manual_graph()
    issues = validate_graph(objects, relations)
    if issues:
        raise RuntimeError("schema validation failed: " + "; ".join(issues[:20]))
    store = JsonKGV2Store(root)
    replaced = store.replace_graph(objects, relations, validate=True)
    if replaced.get("status") != "replaced":
        raise RuntimeError(f"replace_graph failed: {replaced}")
    materialized = KGV2Materializer(store).materialize(store.materialized_root)
    summary = {
        "root": str(Path(root)),
        "replace": replaced,
        "materialized": materialized,
        "counts": {k: len(v) for k, v in objects.items()},
        "relation_count": len(relations),
        "families": [x["label"] for x in objects["FaultFamily"]],
    }
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    out = write_manual_graph(args.root, args.summary_out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
