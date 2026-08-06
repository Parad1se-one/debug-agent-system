"""Seed builders for KG v2 from SOP chunks and manual review cases."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from debug_agent_system.knowledge_v2.contracts import (
    ACTION_ROLES,
    APPROVED_FAMILY_LABELS,
    FAMILY_SUBSYSTEM_EXPECTED,
    INTERNAL_REQUIRED_INFO_SLOTS,
    PSEUDO_FAMILY_LABELS,
    V2_PRIMARY_KEYS,
    make_id,
    trim_text,
    humanize_id,
)

_LEGACY_KG_MARKER = ("data", "kg", "instances")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]|[0-9]+[:：.)]?)\s*")


def assert_not_legacy_kg_input(path: str | Path) -> None:
    parts = tuple(Path(path).resolve().parts)
    marker = tuple(_LEGACY_KG_MARKER)
    for idx in range(len(parts) - len(marker) + 1):
        if tuple(parts[idx : idx + len(marker)]) == marker:
            raise ValueError(f"legacy_kg_input_forbidden:{Path(path)}")


def build_sop_seed(chunks_path: str | Path, *, limit: int = 0) -> dict[str, Any]:
    assert_not_legacy_kg_input(chunks_path)
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("sop_chunks_not_list")
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    count = 0
    skipped_non_fault = 0
    for raw in chunks:
        if not isinstance(raw, dict):
            continue
        meta = raw.get("metadata") or {}
        if str(meta.get("category") or "") != "debug":
            continue
        title = str(meta.get("title") or "").strip()
        if not title:
            continue
        text_body = str(raw.get("text") or "")
        if _is_non_fault_sop_case(title, text_body):
            skipped_non_fault += 1
            continue
        if limit and count >= limit:
            break
        count += 1
        case_ref = str(meta.get("section_num") or title)
        case_id = make_id("case", f"sop-{case_ref}")
        evidence_id = make_id("evidence", f"sop-{case_ref}")
        title_short = _short_title(title)
        family_label = _canonical_family_for_seed(title_short, title, text_body)
        family_id = make_id("family", family_label)
        family_summary = trim_text(_family_summary_for_seed(family_label, title), 80)
        variant_label = _variant_label_for_sop(title, family_label, case_ref)
        variant_id = make_id("variant", f"sop:{case_ref}:{variant_label}")
        category = _family_category(title, text_body)
        subsystem = _subsystem_for_seed(family_label, title, text_body)
        if family_id not in seen_families:
            objects["FaultFamily"].append({
                "family_id": family_id,
                "label": family_label,
                "summary": family_summary,
                "category": category,
                "subsystem": subsystem,
                "scenario": trim_text(title, 60),
                "keywords": [str(x) for x in meta.get("keywords") or []][:12],
                "source_kind": "sop",
                "escalation_target": "",
            })
            seen_families.add(family_id)
        objects["FaultVariant"].append({
            "variant_id": variant_id,
            "family_id": family_id,
            "label": trim_text(variant_label, 60),
            "summary": trim_text(title, 180),
            "equipment_type": "",
            "site": "",
            "software_version": "",
            "error_phase": trim_text(title, 40),
            "owner_context": f"SOP:{case_ref}",
            "escalation_target": "",
            "keywords": [str(x) for x in meta.get("keywords") or []][:16],
        })
        relations.append({"from": family_id, "to": variant_id, "relation": "has_variant"})
        body = _body_lines(text_body)
        action_ids: list[str] = []
        for order, line in enumerate(body, start=1):
            action_role = infer_action_role(line)
            action_id = make_id("action", f"{case_id}:{order}:{line}")
            action_ids.append(action_id)
            objects["DiagnosticAction"].append({
                "action_id": action_id,
                "family_id": family_id,
                "label": trim_text(_action_label(line), 60),
                "summary": trim_text(line, 180),
                "action_role": action_role,
                "step_order": order,
                "destructive": _is_destructive(line),
                "high_cost": _is_high_cost(line),
                "source_kind": "sop",
            })
            if _looks_like_required_info(line):
                required_id = make_id("required-info", f"{case_id}:{line}")
                objects["RequiredInfoSpec"].append({
                    "required_info_id": required_id,
                    "family_id": family_id,
                    "variant_id": "",
                    "slot": infer_required_info_slot(line),
                    "question": trim_text(line, 100),
                    "why_required": trim_text(f"缺少该信息会影响 {family_label} 的诊断分流。", 160),
                    "condition": "",
                    "blocks": [_action_label(line)],
                    "priority": "high" if "日志" in line or "报错" in line or "版本" in line else "medium",
                    "evidence_ids": [evidence_id],
                })
                relations.append({"from": family_id, "to": required_id, "relation": "has_required_info"})
                relations.append({"from": case_id, "to": required_id, "relation": "supports"})
                relations.append({"from": evidence_id, "to": required_id, "relation": "evidences"})
        source_summary = trim_text(" ".join(body) or title, 240)
        objects["SourceCase"].append({
            "case_id": case_id,
            "source_kind": "sop",
            "title": trim_text(title, 80),
            "summary": source_summary,
            "source_ref": str(meta.get("section_num") or ""),
            "approved": True,
        })
        objects["EvidenceItem"].append({
            "evidence_id": evidence_id,
            "source_kind": "sop",
            "external_id": str(meta.get("section_num") or ""),
            "title": trim_text(title, 80),
            "summary": trim_text(str(raw.get("text") or ""), 500),
            "payload_ref": str(meta.get("source") or "SOP"),
        })
        relations.append({"from": evidence_id, "to": case_id, "relation": "evidences"})
        relations.append({"from": case_id, "to": variant_id, "relation": "supports"})
        if action_ids:
            trace_id = make_id("trace", case_id)
            objects["DiagnosticTrace"].append({
                "trace_id": trace_id,
                "family_id": family_id,
                "variant_id": "",
                "source_case_id": case_id,
                "summary": trim_text(f"{family_label} 的 SOP 标准排查骨架", 160),
                "recommended_action_ids": action_ids,
                "actual_action_ids": action_ids,
                "evidence_ids": [evidence_id],
            })
            relations.append({"from": family_id, "to": trace_id, "relation": "has_trace"})
            relations.append({"from": case_id, "to": trace_id, "relation": "supports"})
            for action_id in action_ids:
                relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})
    return {
        "objects": _dedupe_objects(objects),
        "relations": _dedupe_relations(relations),
        "report": {"source": "sop", "cases": count, "skipped_non_fault": skipped_non_fault},
    }


def build_manual_case_seed(manual_root: str | Path, *, limit: int = 0) -> dict[str, Any]:
    assert_not_legacy_kg_input(manual_root)
    root = Path(manual_root)
    files = sorted(root.glob("*.json"))
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []
    count = 0
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        for view in _manual_case_views(raw):
            if limit and count >= limit:
                break
            count += 1
            case_objects, case_relations = _manual_case_to_v2(view)
            for key, items in case_objects.items():
                objects[key].extend(items)
            relations.extend(case_relations)
        if limit and count >= limit:
            break
    return {"objects": _dedupe_objects(objects), "relations": _dedupe_relations(relations), "report": {"source": "manual_review", "cases": count}}


def merge_bundles(*bundles: dict[str, Any]) -> dict[str, Any]:
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for bundle in bundles:
        for key, items in (bundle.get("objects") or {}).items():
            if key in objects:
                objects[key].extend(item for item in items or [] if isinstance(item, dict))
        relations.extend(item for item in bundle.get("relations") or [] if isinstance(item, dict))
        if bundle.get("report"):
            report.append(bundle["report"])
    return {"objects": _dedupe_objects(objects), "relations": _dedupe_relations(relations), "report": report}


def infer_action_role(text: str) -> str:
    t = str(text or "")
    if any(word in t for word in ("联系", "升级给", "@", "负责人", "返厂")):
        return "escalate"
    if any(word in t for word in ("观察", "监控", "持续看", "未再出现", "观察未复发")):
        return "observe"
    if any(word in t for word in ("验证", "确认是否恢复", "确认是否正常")):
        return "verify"
    if any(word in t for word in (
        "更换", "删除", "关闭", "清空", "回退", "升级", "卸载", "复位", "替换", "重启",
        "设置", "优化", "恢复", "限制", "清洁", "清理", "涂抹", "安装", "拆下", "拆卸", "整理",
        "调整", "修复", "还原", "拔插", "拔除", "重插", "清除", "规范", "点胶", "增加",
    )):
        return "change"
    if any(word in t for word in ("对比", "比较", "区分", "排除")):
        return "compare"
    if any(word in t for word in ("导出", "收集", "提供", "记录", "抓取", "上传", "发送")):
        return "collect"
    if any(word in t for word in ("检查", "查看", "核对", "分析", "确认", "读取")):
        return "inspect"
    return "inspect"


def infer_required_info_slot(text: str) -> str:
    t = str(text or "").lower()
    if "dmp" in t or "dump" in t:
        return "dmp_package"
    if "日志" in t or "dlog" in t or "诊断数据" in t or "数据包" in t:
        return "log_package"
    if "版本" in t:
        return "software_version"
    if "错误码" in t or "错误代码" in t or "报错" in t:
        return "error_message"
    if "阶段" in t or "初始化" in t or "扫码" in t or "复判" in t or "检测" in t:
        return "error_phase"
    if "ip" in t or "网络" in t:
        return "ip_config"
    if "复现" in t or "步骤" in t:
        return "repro_steps"
    if "图片" in t or "截图" in t or "样本" in t:
        return "sample_image"
    if "程序" in t or "配方" in t or "proj" in t:
        return "program_file"
    if "内存" in t or "cpu" in t or "接地" in t or "环境" in t or "显示器" in t or "视频线" in t or "hdmi" in t or "vga" in t or "dp" in t:
        return "environment"
    if "驱动" in t or "无线网卡" in t or "显卡" in t:
        return "driver_context"
    if "停线" in t or "生产" in t:
        return "production_constraint"
    if "客户" in t or "现场" in t or "归属" in t:
        return "owner_context"
    return "other"


def _manual_case_to_v2(raw: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []
    manual = raw.get("manual_decision") or {}
    refined = raw.get("refined_merge_proposal") or {}
    nodes = [item for item in refined.get("nodes") or [] if isinstance(item, dict)]
    error = next((item for item in nodes if item.get("type") == "Error"), {})
    target_error_id = str(manual.get("target_error_id") or error.get("error_id") or raw.get("sample_id") or "unknown")
    canonical_error_id = str(manual.get("canonical_error_id") or error.get("canonical_error_id") or target_error_id)
    family_id = make_id("family", canonical_error_id)
    variant_id = make_id("variant", target_error_id)
    case_id = make_id("case", raw.get("sample_id") or raw.get("source_episode_id") or target_error_id)
    family_label = trim_text(error.get("subsystem") or error.get("label") or humanize_id(canonical_error_id), 40)
    family_summary = trim_text(error.get("scenario") or error.get("symptom") or error.get("label") or humanize_id(canonical_error_id), 80)
    objects["FaultFamily"].append({
        "family_id": family_id,
        "label": family_label,
        "summary": family_summary,
        "category": str(error.get("category") or "系统与软件异常"),
        "subsystem": trim_text(error.get("subsystem") or "", 40),
        "scenario": trim_text(error.get("scenario") or error.get("label") or "", 60),
        "keywords": [str(x) for x in error.get("keywords") or []][:12],
        "source_kind": "case",
        "escalation_target": str(error.get("escalation_target") or ""),
    })
    objects["FaultVariant"].append({
        "variant_id": variant_id,
        "family_id": family_id,
        "label": trim_text(error.get("label") or humanize_id(target_error_id), 60),
        "summary": trim_text(error.get("symptom") or error.get("label") or humanize_id(target_error_id), 180),
        "equipment_type": str(error.get("equipment_type") or ""),
        "site": "",
        "software_version": "",
        "error_phase": trim_text(error.get("scenario") or "", 40),
        "owner_context": trim_text(raw.get("source_thread_id") or "", 80),
        "escalation_target": str(error.get("escalation_target") or ""),
        "keywords": [str(x) for x in error.get("keywords") or []][:16],
    })
    relations.append({"from": family_id, "to": variant_id, "relation": "has_variant"})
    case_summary = trim_text("；".join(str(item.get("summary") or "") for item in raw.get("evidence_findings") or [] if isinstance(item, dict)) or error.get("symptom") or target_error_id, 240)
    objects["SourceCase"].append({
        "case_id": case_id,
        "source_kind": "manual_review",
        "title": trim_text(raw.get("sample_id") or target_error_id, 80),
        "summary": case_summary,
        "source_ref": str(raw.get("source_episode_id") or ""),
        "approved": True,
    })
    evidence_ids: list[str] = []
    for idx, finding in enumerate(raw.get("evidence_findings") or [], start=1):
        if not isinstance(finding, dict):
            continue
        evidence_id = make_id("evidence", f"{case_id}:{finding.get('message_id') or idx}")
        evidence_ids.append(evidence_id)
        objects["EvidenceItem"].append({
            "evidence_id": evidence_id,
            "source_kind": "chat_message",
            "external_id": str(finding.get("message_id") or ""),
            "title": trim_text(finding.get("finding") or f"evidence-{idx}", 80),
            "summary": trim_text(finding.get("summary") or "", 500),
            "payload_ref": str(finding.get("time") or ""),
        })
        relations.append({"from": evidence_id, "to": case_id, "relation": "evidences"})
    if not evidence_ids:
        evidence_id = make_id("evidence", f"{case_id}:manual-review")
        evidence_ids.append(evidence_id)
        objects["EvidenceItem"].append({
            "evidence_id": evidence_id,
            "source_kind": "manual_review",
            "external_id": str(raw.get("sample_id") or ""),
            "title": trim_text(raw.get("sample_id") or "manual review", 80),
            "summary": case_summary,
            "payload_ref": str(raw.get("source_episode_id") or ""),
        })
        relations.append({"from": evidence_id, "to": case_id, "relation": "evidences"})
    relations.append({"from": case_id, "to": variant_id, "relation": "supports"})

    action_ids_in_order: list[str] = []
    all_action_ids: list[str] = []
    check_nodes = sorted(
        [item for item in nodes if item.get("type") == "DiagnosticCheck"],
        key=lambda item: int(item.get("step_order") or 9999),
    )
    for check in check_nodes:
        action_id = make_id("action", check.get("check_id") or check.get("id") or check.get("label"))
        action_ids_in_order.append(action_id)
        all_action_ids.append(action_id)
        objects["DiagnosticAction"].append({
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(check.get("label") or "", 60),
            "summary": trim_text(check.get("how_to_check") or check.get("label") or "", 180),
            "action_role": infer_action_role(str(check.get("label") or "") + " " + str(check.get("how_to_check") or "")),
            "step_order": int(check.get("step_order") or 0),
            "destructive": _is_destructive(str(check.get("how_to_check") or "")),
            "high_cost": _is_high_cost(str(check.get("how_to_check") or "")),
            "source_kind": "case",
            "evidence_ids": evidence_ids[:8],
            "execution_status": "actual",
            "evidence_scope": "human_reviewed",
        })
    solution_nodes = [item for item in nodes if item.get("type") == "Solution"]
    for order, solution in enumerate(solution_nodes, start=1):
        action_id = make_id("action", solution.get("solution_id") or solution.get("id") or solution.get("content"))
        all_action_ids.append(action_id)
        outcome_type = _normalize_manual_outcome_type(
            str(solution.get("outcome") or solution.get("evidence_level") or "pending_validation")
        )
        objects["DiagnosticAction"].append({
            "action_id": action_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "label": trim_text(_action_label(str(solution.get("content") or "")), 60),
            "summary": trim_text(solution.get("content") or "", 180),
            "action_role": infer_action_role(str(solution.get("content") or "")),
            "step_order": 100 + order,
            "destructive": _is_destructive(str(solution.get("content") or "")),
            "high_cost": _is_high_cost(str(solution.get("content") or "")),
            "source_kind": "case",
            "evidence_ids": evidence_ids[:8],
            "execution_status": "recommended" if outcome_type == "pending_validation" else "actual",
            "evidence_scope": "human_reviewed",
        })
        outcome_id = make_id("outcome", solution.get("solution_id") or solution.get("id") or solution.get("content"))
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": outcome_type,
            "outcome_origin": "human_reviewed",
            "summary": trim_text(solution.get("content") or "", 200),
            "source_case_id": case_id,
            "evidence_ids": evidence_ids[:8],
            "high_cost": _is_high_cost(str(solution.get("content") or "")),
            "destructive": _is_destructive(str(solution.get("content") or "")),
            "root_cause_summary": trim_text(error.get("label") or "", 120),
        })
        relations.append({"from": variant_id, "to": outcome_id, "relation": "has_outcome"})
        relations.append({"from": case_id, "to": outcome_id, "relation": "supports"})
        relations.append({"from": outcome_id, "to": action_id, "relation": "outcome_of"})
        for evidence_id in evidence_ids[:8]:
            relations.append({"from": evidence_id, "to": outcome_id, "relation": "evidences"})

    outcome_action_ids = {
        str(item.get("action_id") or "")
        for item in objects["ActionOutcome"]
        if isinstance(item, dict)
    }
    for action in objects["DiagnosticAction"]:
        action_id = str(action.get("action_id") or "")
        if not action_id or action_id in outcome_action_ids:
            continue
        outcome_id = make_id("outcome", f"{case_id}:{action_id}:pending-validation")
        objects["ActionOutcome"].append({
            "outcome_id": outcome_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "action_id": action_id,
            "outcome_type": "pending_validation",
            "outcome_origin": "synthetic_fallback",
            "summary": trim_text(f"{action.get('label') or '该动作'}的执行结果待进一步确认", 200),
            "source_case_id": case_id,
            "evidence_ids": evidence_ids[:8],
            "high_cost": bool(action.get("high_cost")),
            "destructive": bool(action.get("destructive")),
            "root_cause_summary": "",
        })
        relations.append({"from": variant_id, "to": outcome_id, "relation": "has_outcome"})
        relations.append({"from": case_id, "to": outcome_id, "relation": "supports"})
        relations.append({"from": outcome_id, "to": action_id, "relation": "outcome_of"})
        for evidence_id in evidence_ids[:8]:
            relations.append({"from": evidence_id, "to": outcome_id, "relation": "evidences"})

    required_specs = list(error.get("required_info_schema") or [])
    required_texts = list(error.get("required_info") or [])
    for idx, item in enumerate(required_specs, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("label") or item.get("slot") or "").strip()
        if not question:
            continue
        required_id = make_id("required-info", f"{case_id}:{question}")
        objects["RequiredInfoSpec"].append({
            "required_info_id": required_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": _normalize_internal_slot(str(item.get("slot") or "other")),
            "question": trim_text(question, 100),
            "why_required": trim_text(str(item.get("why_required") or f"该信息用于缩小 {family_label} 的诊断范围。"), 160),
            "condition": trim_text(str(item.get("condition") or ""), 120),
            "blocks": [str(x) for x in item.get("blocks") or []] or [trim_text(question, 60)],
            "priority": str(item.get("priority") or "medium"),
            "evidence_ids": evidence_ids[:8],
        })
        relations.append({"from": variant_id, "to": required_id, "relation": "has_required_info"})
        relations.append({"from": case_id, "to": required_id, "relation": "supports"})
        for evidence_id in evidence_ids[:4]:
            relations.append({"from": evidence_id, "to": required_id, "relation": "evidences"})
    if not required_specs:
        for text in required_texts:
            question = str(text or "").strip()
            if not question:
                continue
            required_id = make_id("required-info", f"{case_id}:{question}")
            objects["RequiredInfoSpec"].append({
                "required_info_id": required_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "slot": infer_required_info_slot(question),
                "question": trim_text(question, 100),
                "why_required": trim_text(f"该信息用于缩小 {family_label} 的诊断范围。", 160),
                "condition": "",
                "blocks": [trim_text(question, 60)],
                "priority": "medium",
                "evidence_ids": evidence_ids[:8],
            })
            relations.append({"from": variant_id, "to": required_id, "relation": "has_required_info"})
            relations.append({"from": case_id, "to": required_id, "relation": "supports"})
            for evidence_id in evidence_ids[:4]:
                relations.append({"from": evidence_id, "to": required_id, "relation": "evidences"})
    trace_action_ids = all_action_ids
    actual_action_ids = [
        str(item.get("action_id") or "")
        for item in objects["DiagnosticAction"]
        if str(item.get("execution_status") or "") == "actual"
    ]
    if trace_action_ids:
        trace_id = make_id("trace", case_id)
        objects["DiagnosticTrace"].append({
            "trace_id": trace_id,
            "family_id": family_id,
            "variant_id": variant_id,
            "source_case_id": case_id,
            "summary": trim_text(f"{trim_text(error.get('label') or target_error_id, 40)} 的人工审核排查链", 160),
            "recommended_action_ids": trace_action_ids,
            "actual_action_ids": actual_action_ids,
            "evidence_ids": evidence_ids[:8],
        })
        relations.append({"from": family_id, "to": trace_id, "relation": "has_trace"})
        relations.append({"from": variant_id, "to": trace_id, "relation": "has_trace"})
        relations.append({"from": case_id, "to": trace_id, "relation": "supports"})
        for action_id in trace_action_ids:
            relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})
    return _dedupe_objects(objects), _dedupe_relations(relations)


def _empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {
        "FaultFamily": [],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "TraceStep": [],
        "ExecutionObservation": [],
        "BranchRule": [],
        "DecisionPolicy": [],
        "EvidenceItem": [],
        "SourceCase": [],
    }


def _manual_case_views(raw: dict[str, Any]) -> list[dict[str, Any]]:
    refined = raw.get("refined_merge_proposal") or {}
    primary = refined.get("primary_candidate") if isinstance(refined.get("primary_candidate"), dict) else None
    secondary = refined.get("secondary_candidate") if isinstance(refined.get("secondary_candidate"), dict) else None
    if not primary and not secondary:
        return [raw]
    views: list[dict[str, Any]] = []
    for branch_name, candidate in (("primary", primary), ("secondary", secondary)):
        if not isinstance(candidate, dict):
            continue
        manual = dict(raw.get("manual_decision") or {})
        target_key = "target_error_id" if branch_name == "primary" else "secondary_target_error_id"
        canonical_key = "canonical_error_id" if branch_name == "primary" else "secondary_canonical_error_id"
        manual["target_error_id"] = candidate.get("target_error_id") or manual.get(target_key) or manual.get("target_error_id")
        manual["canonical_error_id"] = candidate.get("canonical_error_id") or manual.get(canonical_key) or manual.get("canonical_error_id")
        view = dict(raw)
        view["sample_id"] = f"{raw.get('sample_id')}:{branch_name}"
        view["manual_decision"] = manual
        view["refined_merge_proposal"] = candidate
        views.append(view)
    return views or [raw]


def _family_category(title: str, text: str) -> str:
    merged = f"{title} {text}"
    if any(token in merged for token in ("蓝屏", "闪退", "初始化", "软件", "版本", "驱动", "内存", "工控机")):
        return "系统与软件异常"
    if any(token in merged for token in ("误报", "漏检", "算法", "调参", "识别", "cad", "spc")):
        return "算法与程序调优"
    return "硬件与运控"


def _infer_subsystem(title: str, text: str) -> str:
    merged = f"{title} {text}"
    for token, label in (
        ("相机", "相机/采集链路"),
        ("蓝屏", "工控机/Windows"),
        ("cad", "CAD/程序导入"),
        ("运控", "运控"),
        ("光源", "光源"),
        ("日志", "日志/诊断"),
    ):
        if token.lower() in merged.lower():
            return label
    return ""


def _short_title(title: str) -> str:
    text = str(title or "").replace("【SOP】", "").strip("。 ")
    for sep in ("，", "；", ":", "：", "/", "（"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return trim_text(text, 40)


def _canonical_family_for_seed(title_short: str, title: str, text: str) -> str:
    combined = " ".join([title_short, title, text]).strip()
    raw_family = str(title_short or "").strip()
    lowered = combined.lower()
    if any(k in combined for k in ("用户配置", "加载用户配置失败", "user.cfg", "cfg.toml", "conf")):
        return "用户配置加载失败"
    if any(k in combined for k in ("残帧", "block discarded", "相机事件超时", "discardtrigger ready event timed out", "incomplete frame")):
        return "相机拍摄失败"
    if "显示器无显示" in combined or "基础连接问题" in combined:
        return "工控机黑屏无显示"
    if "间歇性黑屏或死机" in combined:
        return "工控机黑屏无显示"
    if (
        any(k in combined for k in ("主程序无法打开", "主程序打不开", "打开主程序失败", "unknown error", "应用错误", "内部错误", "无法正常进行初始化操作"))
        or ("主程序" in combined and any(k in combined for k in ("无法打开", "打不开", "打开失败")))
    ):
        return "主程序无法打开"
    if any(k in combined for k in ("工厂程序无法打开", "工厂程序打不开")):
        return "工厂程序无法打开"
    if any(k in combined for k in ("运控打不开", "运控程序错误", "运控程序无法打开")):
        return "运控程序无法打开"
    if any(k in combined for k in ("回不了原点", "运控错误", "com17连接失败", "COM17连接失败", "软复位", "硬复位", "复位失败", "运控卡初始化", "运控没有就绪", "运控未就绪", "需要重置")):
        return "运控初始化失败"
    if any(k in combined for k in ("mes", "MES", "过站", "工单号", "接驳台", "返回值", "SN卡控", "SN报警", "ok信号", "OK信号", "ng信号", "NG信号", "OK/NG", "模式5", "请求要板", "下道", "上道")):
        return "MES 过站异常"
    if any(k in combined for k in ("加密狗", "许可证", "license", "License", "密码狗")):
        return "许可证/加密狗异常"
    if any(k in combined for k in ("坏板标记", "跳过后sn报警", "SN报警", "坏板跳过", "坏板标记无效", "坏板标记", "坏板跳过后")):
        return "坏板标记异常"
    if any(k in combined for k in ("复判窗口", "未复判的数据", "pass板无弹窗反馈", "复判界面没显示", "复盘结果不出来", "复判结果显示", "复判结果", "复盘ok后显示ng", "未弹出复判窗口", "未提示", "pass板", "未复判")):
        return "复判结果显示异常"
    if any(k in combined for k in ("异响", "刺耳声音", "嗡鸣", "轮子摆动")):
        return "机械运动异响"
    if any(k in combined for k in ("传感器", "感应器", "感应不到", "感应不好", "不灵敏")):
        return "传感器感应异常"
    if any(k in combined for k in ("进入bios", "BIOS页面", "BIOS 启动", "启动项", "bios设置", "BIOS设置")):
        return "BIOS 启动配置异常"
    if any(k in combined for k in ("进入系统", "不进入系统", "无法进入系统", "启动修复", "系统修复")):
        return "操作系统启动失败"
    if any(k in combined for k in ("spc页面", "SPC页面", "SPC 页面", "页面进不去", "加载不出来", "spc查询报错", "SPC查询报错", "单板分析无法查看", "单班分析无法打开")):
        return "SPC 页面无法打开"
    if any(k in combined for k in ("复判站和主机连接异常", "复盘站和主机连接异常", "主机连接异常", "复判站主机通信", "断联", "连接不了", "复判站连接不上", "复盘站连接不上", "连接不到主机", "ip为空", "不显示数据")):
        return "复判站主机通信异常"
    if any(k in combined for k in ("保存路径失败", "获取保存路径失败", "保存结果失败", "推送结果至buddy失败", "保存检测结果失败", "保存数据失败")):
        return "复判保存结果失败"
    if any(k in lowered for k in ("cuda", "未检查到cuda设备")):
        return "CUDA 计算设备不可用"
    if any(k in combined for k in ("加载板卡失败", "板卡加载失败", "加载已复判列表失败", "加载最新模块失败", "无法加载出板卡", "复判站报错加载失败", "加载失败")):
        return "复判站加载板卡异常" if "复判站" in combined else "程序板卡加载失败"
    if any(k in combined for k in ("应用异常", "应用不成功", "加载板卡时间过长")):
        return "程序板卡加载失败"
    if any(k in combined for k in ("进板失败", "不进板", "轨道有板", "进到轨道内一半", "没有感应到进板", "第一片板不会进板")):
        return "进板失败"
    if any(k in combined for k in ("出板失败", "飞板", "出板传感器")):
        return "出板失败"
    if any(k in combined for k in ("卡板", "挡板卡住", "板子无法在轨道中滑动", "挡住板卡", "板边掉进去")):
        return "卡板"
    if "挡块" in combined or "夹爪" in combined:
        return "挡块异常"
    if "顶升" in combined:
        return "顶升机构异常"
    if "皮带" in combined:
        return "皮带运行异常"
    if any(k in combined for k in ("轨道宽度", "轨道小0.5", "喇叭口")):
        return "轨道宽度无法调节"
    if "气压" in combined:
        return "气压异常"
    if any(k in combined for k in ("扫码枪", "扫码", "二维码", "条码", "DM码", "dm码", "SN搜索")):
        return "扫码识别失败"
    if any(k in combined for k in ("焊盘框", "器件框角度", "识别框", "框选")):
        if "焊盘框" in combined:
            return "焊盘框不对齐"
        if "角度" in combined:
            return "器件框角度不匹配"
        if "大小" in combined or "框不准确" in combined:
            return "识别框大小不准确"
        return "框选识别不准"
    if any(k in combined for k in ("mark点", "Mark点", "mark 点", "多拼板", "定位点")):
        return "Mark 点对齐失败"
    if any(k in combined for k in ("误报", "自动变成OK", "报错件", "误判")):
        return "误报调优异常"
    if any(k in combined for k in ("漏检", "漏报", "飞件未检出", "脏污未检出", "连锡未检出", "缺件", "未报出", "没检出", "极反偶尔漏报", "错件漏报", "多件未检出")):
        return "漏检调优异常"
    if any(k in combined for k in ("ocr", "OCR", "极反", "多料", "脏污", "连锡", "飞件", "金手指", "红胶", "pad", "PAD", "错位报的较多", "智能调整出现", "丝印", "浮高", "器件错位", "少锡", "单引脚", "不溶锡", "假焊", "未检出", "没打下去", "镜像模糊")):
        return "算法/程序调优异常"
    if any(k in combined for k in ("相机", "拍摄", "拍照", "拍图", "fov图片缺失", "fov", "FOV", "拍照模糊", "debayering", "拼接异常", "拼图错位", "图片拼接错位", "成像")):
        if any(k in combined for k in ("初始化", "枚举", "连接相机")):
            return "相机初始化失败"
        return "相机拍摄失败"
    if any(k in combined for k in ("显示器无显示", "无显示")) and any(k in combined for k in ("风扇一直转", "键盘鼠标灯亮", "视频线", "信号源", "核显输出", "主板")):
        return "工控机黑屏无显示"
    if any(k in combined for k in ("间歇性黑屏", "突然黑屏")) and any(k in combined for k in ("风扇", "死机", "运行一段时间后")):
        return "工控机黑屏无显示"
    if any(k in combined for k in ("显示器", "主机都不亮", "未显示结果", "未显示报错框", "工单号面别没有显示", "置灰", "花屏", "屏卡住了", "显示空白", "不全屏显示", "器件视图打开没有图像", "打开没有图像", "卡在一个界面", "界面卡住")):
        return "界面显示异常"
    if any(k in combined for k in ("显示不全", "缩放", "分辨率", "扩展", "复制", "电视", "花屏", "黑屏", "软件都没显示是OK还是ng")):
        return "界面显示异常" if "花屏" not in combined and "黑屏无显示" not in combined else "工控机黑屏无显示"
    if any(k in combined for k in ("卡顿", "响应慢", "缓慢", "变慢", "很慢", "时间长", "出图慢", "反应慢", "传输速率慢", "保存慢", "出数据慢", "等待时间过长", "检测时间变长")):
        if any(k in combined for k in ("ct", "CT", "节拍", "出图慢", "传输速率慢", "保存慢", "出数据慢", "等待时间过长", "检测时间变长")):
            return "CT 时间异常增加"
        return "程序运行卡顿"
    if any(k in combined for k in ("卡死", "死机", "无响应", "闪退", "动不了了", "停不下来", "强制关软件")):
        return "软件卡死无响应"
    if any(k in combined for k in ("无法开机", "无法正常开机", "不能正常开机", "开机无法启动", "无法启动", "开不了机", "不能启动", "启动不了了", "无法正常开启", "电脑不能正常开机")):
        return "工控机无法开机"
    if any(k in combined for k in ("黑屏无显示", "开机黑屏", "黑屏不显示")):
        return "工控机黑屏无显示"
    if any(k in combined for k in ("蓝屏", "bugcheck", "BugCheck")):
        return "工控机蓝屏"
    if any(k in combined for k in ("自动重启", "异常重启", "重启", "黑屏自动重启", "自动关机", "莫名重启")):
        return "工控机异常重启"
    if any(k in combined for k in ("搜索项目名", "项目名搜索", "无法搜索项目名", "项目名")):
        return "主程序/系统异常"
    if any(k in lowered for k in ("wifi", "usb", "u盘", "无线网卡")):
        return "外设连接不稳定"
    if any(k in combined for k in ("c盘", "d盘", "磁盘", "页面文件", "虚拟内存", "显存不足", "空间不足", "挂载不了", "盘满了")):
        return "磁盘 I/O 异常"
    if any(k in combined for k in ("交换机", "收发器", "光纤接口", "运控卡连接失败", "网卡自适配1g")):
        return "控制器网络配置异常"
    if any(k in combined for k in ("网卡驱动掉线", "驱动掉线", "网络掉线")):
        return "控制器网络配置异常"
    if any(k in combined.lower() for k in ("cad", "导入")):
        return "CAD 导入失败"
    if any(k in combined for k in ("拼接问题", "白彩图错位", "拼板错位", "拼图错位")):
        return "相机拍摄失败"
    if any(k in combined for k in ("算法", "虚焊", "翘脚", "框未生成", "singlepin", "pinpad", "识别")):
        return "算法/程序调优异常"
    if any(k in combined for k in ("运控初始化", "运动控制初始化")):
        return "运控初始化失败"
    if any(k in combined for k in ("光源初始化", "光控初始化")):
        return "光源初始化失败"
    if any(k in combined for k in ("buddy", "模板")):
        if any(k in combined for k in ("缺失", "没有模板")):
            return "Buddy 模板缺失"
        if any(k in combined for k in ("创建失败", "模板管理")):
            return "Buddy 模板创建失败"
        return "模板文件损坏"
    if raw_family in APPROVED_FAMILY_LABELS:
        return raw_family
    if raw_family in PSEUDO_FAMILY_LABELS:
        return _generic_seed_family(title, text)
    return _generic_seed_family(title, text)


def _is_non_fault_sop_case(title: str, text: str) -> bool:
    merged = " ".join([str(title or ""), str(text or "")]).lower()
    hard_skip_markers = (
        "加密码狗续期",
        "需要api接口读取数据",
        "现场需要mes连接",
        "配置mes",
        "近日有时间帮忙协助做下设备cpk和grr数据",
        "指导对接接驳台信号线",
        "咨询dip后端是否可以检出",
        "弯板需要邮寄顶pin",
        "需要顶pin",
        "打开7个g的excel文件",
        "需要mes连接",
        "api接口",
        "cpk和grr",
        "问题总结",
        "现场问题总结",
        "相关问题总结",
        "咨询aoi信号线与送板机线如何接",
        "那两根是ok信号线",
        "下游们信号线34短接",
        "轨道速度更改",
        "调整进出板速度",
        "传板速度是在哪里改的",
        "同时接分屏和复判站",
        "gerber是原理图中导出",
        "鼠标坏了",
        "定位块弯的位置有没有长一点的",
        "连接器需优化",
        "要设置共享文件夹",
        "咨询sn信息和检查结果是否可以储存在csv文件中",
        "程序覆盖率表格导出需求",
        "spc能不能导出大图",
        "检测板整图保存",
    )
    if any(marker.lower() in merged for marker in hard_skip_markers):
        return True
    if "mes" in merged and not any(k in merged for k in ("报错", "异常", "失败", "计数有问题")):
        return True
    return False


def _generic_seed_family(title: str, text: str) -> str:
    category = _family_category(title, text)
    if "硬件" in category or "运控" in category:
        return "主程序/系统异常"
    if "算法" in category:
        return "算法/程序调优异常"
    return "主程序/系统异常"


def _subsystem_for_seed(family_label: str, title: str, text: str) -> str:
    return FAMILY_SUBSYSTEM_EXPECTED.get(family_label) or _infer_subsystem(title, text)


def _family_summary_for_seed(family_label: str, title: str) -> str:
    if family_label in APPROVED_FAMILY_LABELS:
        return family_label
    return trim_text(title, 80)


def _variant_label_for_sop(title: str, family_label: str, case_ref: str) -> str:
    clean = trim_text(str(title or "").strip(), 60)
    if not clean:
        return f"{family_label}（SOP {case_ref}）"
    if clean == family_label:
        return f"{family_label}（SOP {case_ref}）"
    return clean


def _body_lines(text: str) -> list[str]:
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text
    lines: list[str] = []
    for raw in body.splitlines():
        line = _BULLET_PREFIX.sub("", raw).strip()
        if line:
            lines.append(line)
    return lines


def _action_label(text: str) -> str:
    raw = str(text or "").strip().lstrip("，。；;：:、 ")
    line = trim_text(raw, 60)
    for sep in ("；", "，", "。"):
        if sep in line:
            head = line.split(sep, 1)[0].strip().lstrip("，。；;：:、 ")
            if head:
                return head
    return line


def _is_destructive(text: str) -> bool:
    return any(token in str(text or "") for token in ("拆机", "返厂", "格式化", "删除conf", "清空"))


def _is_high_cost(text: str) -> bool:
    return any(token in str(text or "") for token in ("返厂", "重标", "高成本", "更换相机", "更换工控机"))


def _looks_like_required_info(text: str) -> bool:
    return infer_action_role(text) == "collect" or any(token in str(text or "") for token in ("日志", "DMP", "版本", "报错", "截图", "样本"))


def _normalize_internal_slot(slot: str) -> str:
    value = str(slot or "").strip()
    return value if value in INTERNAL_REQUIRED_INFO_SLOTS else infer_required_info_slot(value)


def _normalize_manual_outcome_type(value: str) -> str:
    raw = str(value or "").strip()
    if raw in {"verified_fix", "ineffective", "partial_temporary", "mitigation_observed", "recurred", "pending_validation", "diagnostic_method", "context_not_root_cause"}:
        return raw
    mapping = {
        "case_verified_fix": "verified_fix",
        "temporary_then_recurred": "recurred",
        "mitigation_uncertain": "mitigation_observed",
        "temporary_recovery": "partial_temporary",
        "partial_then_recurred": "recurred",
        "workaround": "mitigation_observed",
        "pending_rnd_investigation": "pending_validation",
        "candidate_final_fix_high_cost": "pending_validation",
        "mitigation_observed_then_recurred": "recurred",
        "recommended_pending_validation": "pending_validation",
    }
    return mapping.get(raw, "pending_validation")


def _dedupe_objects(objects: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out = _empty_objects()
    for obj_type, items in objects.items():
        seen: dict[str, dict[str, Any]] = {}
        pk = V2_PRIMARY_KEYS[obj_type]
        for item in items:
            if not isinstance(item, dict):
                continue
            obj_id = str(item.get(pk) or "")
            if not obj_id:
                continue
            if obj_id in seen:
                seen[obj_id].update({k: v for k, v in item.items() if v not in (None, "", [])})
            else:
                seen[obj_id] = dict(item)
        out[obj_type] = list(seen.values())
    return out


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        key = (str(rel.get("from") or ""), str(rel.get("to") or ""), str(rel.get("relation") or ""))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out
