from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from debug_agent_system.knowledge_v2.builders import (
    _body_lines,
    _canonical_family_for_seed,
    _family_category,
    _family_summary_for_seed,
    _short_title,
    _subsystem_for_seed,
    infer_action_role,
    infer_required_info_slot,
)
from debug_agent_system.knowledge_v2.contracts import APPROVED_FAMILY_LABELS, make_id, trim_text

DEFAULT_SOP_CHUNKS = "data/raw/aoi_debug_agent_sources/chunks/debug_chunks.json"
DEFAULT_OUT = "data/results/kg_v2_sop_seed_draft_manual.json"

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\\S+")
_FILE_ONLY = re.compile(r"^[\\w./\\\\:-]+\\.(?:exe|zip|bat|json|toml|md|txt)$", re.I)
_SECTION_CASE_RE = re.compile(r"^[0-9]+(?:\\.[0-9]+)+$")
_NOISE_PREFIXES = (
    "处理结果",
    "备注",
    "描述",
    "持续跟进",
    "原因",
    "已知bug",
    "已提jira",
    "jira",
    "bug",
    "sop",
)
_LOW_VALUE_ACTION_PREFIXES = (
    "复制以下命令",
    "搜索并打开",
    "在“高级”",
    "选择",
    "点击",
    "主页面",
    "打开windows powershell",
)

_ACTION_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(收集|提供|导出).*(dmp|dump|转储)", re.I), "收集 DMP / 转储文件", "collect"),
    (re.compile(r"(分析|查看).*(dmp|dump|转储)", re.I), "分析 DMP / 转储文件", "inspect"),
    (re.compile(r"(收集|提供|导出|上传|发送).*(日志|dlog|诊断数据|数据包)", re.I), "收集日志 / 诊断数据包", "collect"),
    (re.compile(r"(查看|检查|分析).*(日志|dlog|诊断数据)", re.I), "查看日志 / 诊断数据", "inspect"),
    (re.compile(r"相机ip.*(改为|固定|自动获取)", re.I), "调整相机 IP 配置验证", "change"),
    (re.compile(r"(浏览器访问|检查).*(cyclops|lighter|光源控制器).*(版本|亮度|状态)", re.I), "检查光源控制器版本 / 状态", "inspect"),
    (re.compile(r"(检测|检查|确认).*(相机sdk|sdk版本)", re.I), "检查相机 SDK 版本", "inspect"),
    (re.compile(r"(打开|使用).*(galaxy viewer)", re.I), "使用 Galaxy Viewer 排查相机状态", "inspect"),
    (re.compile(r"(参考此文档|逐项排查).*(相机网口|电源模式|配置)", re.I), "按文档排查相机网口与电源配置", "inspect"),
    (re.compile(r"(检测头复位|3d检测头复位)", re.I), "执行 3D 检测头复位", "change"),
    (re.compile(r"(检查|确认|查看).*(版本|sdk)", re.I), "确认软件 / SDK 版本", "inspect"),
    (re.compile(r"(回退|降级).*(版本|machine|软件)", re.I), "回退版本验证", "change"),
    (re.compile(r"(升级|更新).*(版本|sdk|驱动)", re.I), "升级版本 / 驱动验证", "change"),
    (re.compile(r"(检查|确认|查看).*(ip|网口|网络|网卡|控制器)", re.I), "检查网络 / IP 配置", "inspect"),
    (re.compile(r"(重启|断电重启|上电重启)", re.I), "重启后复现验证", "verify"),
    (re.compile(r"(彻底关机|断电并等待|上电重启)", re.I), "断电重启验证", "verify"),
    (re.compile(r"(检查|查看|确认).*(相机|采集卡|cxp线|网卡|光源控制器|运控卡)", re.I), "检查采集链路硬件状态", "inspect"),
    (re.compile(r"(紧固|重新插拔|插拔).*(cxp|网线|网卡|接口|线缆)", re.I), "重新插拔 / 紧固连接线缆", "change"),
    (re.compile(r"(删除|清空).*(conf|配置)", re.I), "清空配置目录并重启验证", "change"),
    (re.compile(r"(替换|回填).*(cfg|配置文件|toml)", re.I), "替换配置文件并重启验证", "change"),
    (re.compile(r"(检查|查看|确认).*(conf|user\\.cfg|cfg\\.toml)", re.I), "检查配置文件内容", "inspect"),
    (re.compile(r"(检查|确认|查看).*(cad|坐标|角度|拼版|位号)", re.I), "检查 CAD 坐标 / 角度 / 拼版配置", "inspect"),
    (re.compile(r"(检查|确认|查看).*(mark|定位点)", re.I), "检查 Mark 点定位配置", "inspect"),
    (re.compile(r"(虚拟内存|页面文件)", re.I), "调整虚拟内存设置", "change"),
    (re.compile(r"(内存|显存).*(是否正常|配置|频率)", re.I), "检查内存 / 显存配置", "inspect"),
    (re.compile(r"(联系|通知|升级给|联系fae|联系研发|返厂)", re.I), "升级给 FAE / 研发进一步排查", "escalate"),
]

_FAMILY_REQUIRED_INFO_PRIORS: dict[str, list[str]] = {
    "工控机蓝屏": ["dmp_package", "log_package", "software_version"],
    "工控机异常重启": ["log_package", "software_version", "environment"],
    "相机拍摄失败": ["log_package", "software_version", "ip_config", "error_phase"],
    "相机初始化失败": ["log_package", "software_version", "ip_config"],
    "用户配置加载失败": ["program_file", "software_version"],
    "CAD 导入失败": ["program_file", "sample_image", "software_version"],
    "Mark 点对齐失败": ["program_file", "sample_image"],
    "程序运行卡顿": ["log_package", "software_version", "environment"],
    "软件卡死无响应": ["log_package", "software_version", "environment"],
    "光源初始化失败": ["log_package", "software_version", "error_phase"],
}

_SLOT_QUESTION = {
    "log_package": "请提供诊断日志 / DLOG / 诊断数据包。",
    "dmp_package": "请提供 DMP / 转储文件。",
    "software_version": "请提供软件版本 / SDK 版本 / machine 版本。",
    "error_phase": "请说明故障发生阶段和复现时机。",
    "error_message": "请提供完整报错信息或错误代码。",
    "device_model": "请提供设备型号 / 相机型号 / 板卡型号。",
    "site": "请提供现场站点 / 项目信息。",
    "ip_config": "请提供 IP / 网卡 / 控制器网络配置。",
    "repro_steps": "请提供稳定复现步骤。",
    "sample_image": "请提供样图 / 截图 / 现象图片。",
    "program_file": "请提供程序文件 / CAD / 配方 / 配置文件。",
    "environment": "请提供环境信息，如内存、磁盘、温度、供电、接地等。",
    "owner_context": "请提供现场上下文和责任归属信息。",
    "memory_cpu_test": "请提供内存 / CPU / 压测结果。",
    "driver_context": "请提供相关驱动版本和驱动状态。",
    "production_constraint": "请说明停线压力、是否可停机和生产约束。",
    "other": "请补充与本故障定位直接相关的关键信息。",
}

_SLOT_WHY = {
    "log_package": "需要从日志里确认报错点、阶段和模块边界。",
    "dmp_package": "需要从转储里定位蓝屏 / 崩溃指向的驱动或内核模块。",
    "software_version": "需要判断是否命中特定版本行为或已知版本差异。",
    "error_phase": "需要用故障发生阶段缩小诊断路径。",
    "error_message": "需要用错误码 / 报错文本收敛故障分支。",
    "device_model": "需要确认是否是特定型号或硬件代际差异。",
    "site": "需要确认是否和特定现场 / 项目环境相关。",
    "ip_config": "需要判断是否是网络配置或链路稳定性问题。",
    "repro_steps": "需要稳定复现路径才能验证修复和定位分支。",
    "sample_image": "需要从现象图片判断是成像问题、算法问题还是显示问题。",
    "program_file": "需要核对程序 / CAD / 配方 / 配置是否本身异常。",
    "environment": "需要排查供电、温度、磁盘、内存等环境因素。",
    "owner_context": "需要确认责任边界和现场处理限制。",
    "memory_cpu_test": "需要判断是否是性能 / 内存 / 计算稳定性问题。",
    "driver_context": "需要判断是否是驱动版本或驱动状态导致。",
    "production_constraint": "需要确定是否允许执行停机、重启、替换等动作。",
    "other": "需要补足当前证据不足的关键上下文。",
}


def _clean_text(text: str) -> str:
    text = _URL.sub("", str(text or ""))
    text = text.replace("【SOP】", "")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"（[^）]*仅[^）]*）", "", text)
    text = re.sub(r"\([^)]*only[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\([^)]*仅[^)]*\)", "", text)
    text = text.strip("，。；;：:、- ")
    text = _WS.sub(" ", text)
    return text.strip()


def _is_noise_line(line: str) -> bool:
    text = _clean_text(line)
    if not text:
        return True
    lowered = text.lower()
    if _FILE_ONLY.match(text):
        return True
    if _SECTION_CASE_RE.match(text):
        return True
    if len(text) <= 2:
        return True
    if lowered.startswith(("http://", "https://")):
        return True
    if any(lowered.startswith(prefix) for prefix in _NOISE_PREFIXES):
        return True
    if text in {"主页面", "SOP", "注意"}:
        return True
    if "已提JIRA" in text or "持续跟进" in text:
        return True
    if (
        any(token in text for token in ("通常会有", "可能是", "异常原因", "客户版本是", "偶发", "已知bug", "说明：", "描述：", "原因："))
        and not any(token in text for token in ("检查", "查看", "收集", "提供", "导出", "重启", "升级", "更新", "删除", "清空", "联系", "回退"))
    ):
        return True
    if "导致" in text and not any(token in text for token in ("检查", "查看", "收集", "提供", "导出", "重启", "升级", "更新", "删除", "清空", "联系", "回退", "改为", "插拔", "紧固", "执行")):
        return True
    if text.startswith(("如果", "正常没有这种", "客户版本是")) and not any(token in text for token in ("检查", "改为", "联系", "升级", "回退")):
        return True
    return False


def _is_low_value_action_label(label: str) -> bool:
    text = str(label or "").strip()
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in _LOW_VALUE_ACTION_PREFIXES):
        return True
    if any(token in text for token in ("导致", "问题处理", "哈")):
        return True
    return False


def _fallback_action_label(line: str) -> str:
    text = _clean_text(line)
    for sep in ("；", "。", "，"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    text = re.sub(r"^如果.*?(则|就)", "", text)
    text = re.sub(r"^(点击|打开|进入|在).*?(，|,)", "", text)
    text = trim_text(text.strip("，。；;：:、- "), 40)
    return text


def _normalize_action(line: str) -> tuple[str, str, str] | None:
    text = _clean_text(line)
    if _is_noise_line(text):
        return None
    lowered = text.lower()
    for pattern, label, role in _ACTION_RULES:
        if pattern.search(lowered) or pattern.search(text):
            return label, trim_text(text, 160), role
    label = _fallback_action_label(text)
    if not label or len(label) < 2:
        return None
    return label, trim_text(text, 160), infer_action_role(text)


def _slot_payload(slot: str) -> tuple[str, str]:
    return _SLOT_QUESTION.get(slot, _SLOT_QUESTION["other"]), _SLOT_WHY.get(slot, _SLOT_WHY["other"])


def _load_sop_chunks(path: str | Path) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("sop_chunks_not_list")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") or {}
        if str(meta.get("source") or "") != "SOP":
            continue
        if str(meta.get("category") or "") != "debug":
            continue
        out.append(row)
    return out


def _seed_family_for_sop_case(title: str, text: str) -> str:
    merged = f"{title} {text}"
    if any(token in title for token in ("C盘空间占用异常", "D盘满了", "磁盘保存失败", "硬盘挂载不了")):
        return "磁盘 I/O 异常"
    if "蓝屏" in title:
        return "工控机蓝屏"
    if "黑屏" in title and "花屏" not in title:
        return "工控机黑屏无显示"
    if "CT时间长" in title:
        return "CT 时间异常增加"
    if "出图慢" in title:
        return "复判站出图慢"
    if "误报" in title:
        return "误报调优异常"
    if any(token in title for token in ("漏检", "漏报", "缺件")):
        return "漏检调优异常"
    if "扫码枪" in title:
        return "扫码枪异常"
    if any(token in title for token in ("DM码", "dm码")):
        return "DM 码识别失败"
    if "二维码" in title and "扫码枪" not in title:
        return "扫码识别失败"
    return _canonical_family_for_seed(_short_title(title), title, text)


def build_sop_seed_draft(chunks_path: str | Path = DEFAULT_SOP_CHUNKS) -> dict[str, Any]:
    chunks = _load_sop_chunks(chunks_path)
    family_records: dict[str, dict[str, Any]] = {}

    for row in chunks:
        meta = row.get("metadata") or {}
        title = str(meta.get("title") or "").strip()
        text = str(row.get("text") or "")
        case_ref = str(meta.get("section_num") or title)
        if not title:
            continue
        family_label = _seed_family_for_sop_case(title, text)
        family_id = make_id("family", family_label)
        family = family_records.setdefault(
            family_id,
            {
                "family": {
                    "family_id": family_id,
                    "label": family_label,
                    "summary": trim_text(_family_summary_for_seed(family_label, title), 80),
                    "category": _family_category(title, text),
                    "subsystem": _subsystem_for_seed(family_label, title, text),
                    "scenario": trim_text(title, 60),
                    "keywords": [str(x) for x in meta.get("keywords") or []][:12],
                    "source_kind": "sop_seed",
                    "escalation_target": "",
                },
                "section_refs": [],
                "actions": defaultdict(lambda: {"count": 0, "role": Counter(), "summaries": [], "sections": set(), "step_orders": [], "high_cost": False, "destructive": False}),
                "required_info": defaultdict(lambda: {"count": 0, "sections": set(), "question": "", "why_required": ""}),
            },
        )

        family["section_refs"].append({"section_num": case_ref, "title": trim_text(title, 80)})
        lines = _body_lines(text)
        for step_order, line in enumerate(lines, start=1):
            action = _normalize_action(line)
            if action is not None:
                label, summary, role = action
                slot = infer_required_info_slot(line)
                payload = family["actions"][label]
                payload["count"] += 1
                payload["role"][role] += 1
                payload["summaries"].append(summary)
                payload["sections"].add(case_ref)
                payload["step_orders"].append(step_order)
                payload["high_cost"] = payload["high_cost"] or any(token in line for token in ("返厂", "更换相机", "更换工控机", "重标"))
                payload["destructive"] = payload["destructive"] or any(token in line for token in ("删除", "清空", "格式化", "拆机"))
                if slot != "other":
                    req = family["required_info"][slot]
                    req["count"] += 1
                    req["sections"].add(case_ref)
                    req["question"], req["why_required"] = _slot_payload(slot)
            else:
                slot = infer_required_info_slot(line)
                if slot != "other":
                    req = family["required_info"][slot]
                    req["count"] += 1
                    req["sections"].add(case_ref)
                    req["question"], req["why_required"] = _slot_payload(slot)

        for slot in _FAMILY_REQUIRED_INFO_PRIORS.get(family_label, []):
            req = family["required_info"][slot]
            if not req["question"]:
                req["question"], req["why_required"] = _slot_payload(slot)
            if req["count"] == 0:
                req["count"] = 1
                req["sections"].add(case_ref)

    families_view: list[dict[str, Any]] = []
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
    missing_approved = sorted(APPROVED_FAMILY_LABELS - {rec["family"]["label"] for rec in family_records.values()})

    for family_id, record in sorted(
        family_records.items(),
        key=lambda item: (-len(item[1]["section_refs"]), item[1]["family"]["label"]),
    ):
        family_obj = record["family"]
        objects["FaultFamily"].append(family_obj)

        action_rows = []
        for label, payload in record["actions"].items():
            support_count = int(payload["count"])
            if support_count <= 0:
                continue
            if _is_low_value_action_label(label):
                continue
            role = payload["role"].most_common(1)[0][0] if payload["role"] else "inspect"
            avg_step = round(mean(payload["step_orders"]), 2) if payload["step_orders"] else 999.0
            action_id = make_id("action-template", f"{family_id}:{label}")
            summary = min(payload["summaries"], key=len) if payload["summaries"] else label
            action_rows.append({
                "action_id": action_id,
                "label": label,
                "summary": trim_text(summary, 180),
                "action_role": role,
                "support_count": support_count,
                "avg_step_order": avg_step,
                "source_sections": sorted(payload["sections"])[:8],
                "high_cost": bool(payload["high_cost"]),
                "destructive": bool(payload["destructive"]),
            })

        action_rows.sort(key=lambda item: (item["avg_step_order"], -item["support_count"], item["label"]))
        action_rows = action_rows[:12]
        for idx, action in enumerate(action_rows, start=1):
            objects["DiagnosticAction"].append({
                "action_id": action["action_id"],
                "family_id": family_id,
                "label": action["label"],
                "summary": action["summary"],
                "action_role": action["action_role"],
                "step_order": idx,
                "destructive": action["destructive"],
                "high_cost": action["high_cost"],
                "source_kind": "sop_seed",
                "support_count": action["support_count"],
                "source_sections": action["source_sections"],
            })

        req_rows = []
        for slot, payload in record["required_info"].items():
            if not payload["question"]:
                continue
            req_id = make_id("required-info-template", f"{family_id}:{slot}")
            req_rows.append({
                "required_info_id": req_id,
                "slot": slot,
                "question": payload["question"],
                "why_required": payload["why_required"],
                "support_count": int(payload["count"]),
                "source_sections": sorted(payload["sections"])[:8],
            })
        req_rows.sort(key=lambda item: (-item["support_count"], item["slot"]))
        req_rows = req_rows[:8]
        for req in req_rows:
            objects["RequiredInfoSpec"].append({
                "required_info_id": req["required_info_id"],
                "family_id": family_id,
                "variant_id": "",
                "slot": req["slot"],
                "question": req["question"],
                "why_required": req["why_required"],
                "condition": "",
                "blocks": [a["label"] for a in action_rows[:3]],
                "priority": "high" if req["slot"] in {"log_package", "dmp_package", "software_version", "program_file"} else "medium",
                "evidence_ids": [],
                "support_count": req["support_count"],
                "source_sections": req["source_sections"],
            })
            relations.append({"from": family_id, "to": req["required_info_id"], "relation": "has_required_info"})

        trace_id = make_id("trace-template", family_id)
        ordered_action_ids = [item["action_id"] for item in action_rows]
        objects["DiagnosticTrace"].append({
            "trace_id": trace_id,
            "family_id": family_id,
            "variant_id": "",
            "source_case_id": "",
            "summary": trim_text(f"{family_obj['label']} 的 SOP 标准排查骨架", 160),
            "recommended_action_ids": ordered_action_ids,
            "actual_action_ids": [],
            "evidence_ids": [],
            "source_sections": [item["section_num"] for item in record["section_refs"][:8]],
        })
        relations.append({"from": family_id, "to": trace_id, "relation": "has_trace"})
        for action_id in ordered_action_ids:
            relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})

        policy_id = make_id("policy-template", family_id)
        objects["DecisionPolicy"].append({
            "policy_id": policy_id,
            "family_id": family_id,
            "source_trace_ids": [trace_id],
            "source_outcome_ids": [],
            "ordered_action_ids": ordered_action_ids,
            "ineffective_action_ids": [],
            "high_cost_action_ids": [item["action_id"] for item in action_rows if item["high_cost"]],
            "deterministic_recompute": False,
            "source_kind": "sop_seed",
        })

        families_view.append({
            "family": family_obj,
            "sop_case_count": len(record["section_refs"]),
            "source_sections": record["section_refs"][:8],
            "action_templates": action_rows,
            "required_info_templates": req_rows,
            "trace_template": {
                "trace_id": trace_id,
                "summary": trim_text(f"{family_obj['label']} 的 SOP 标准排查骨架", 160),
                "recommended_action_ids": ordered_action_ids,
                "recommended_action_labels": [item["label"] for item in action_rows],
                "ordering_basis": "avg_step_order_then_support_count",
            },
        })

    return {
        "schema_version": "kg_v2.sop_only_seed_draft.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "kind": "SOP",
            "chunks_path": str(chunks_path),
            "filters": {"metadata.source": "SOP", "metadata.category": "debug"},
            "generation_mode": "family_level_sop_seed",
        },
        "counts": {
            "source_sop_cases": len(chunks),
            "fault_families": len(objects["FaultFamily"]),
            "diagnostic_actions": len(objects["DiagnosticAction"]),
            "required_info_specs": len(objects["RequiredInfoSpec"]),
            "diagnostic_traces": len(objects["DiagnosticTrace"]),
            "decision_policies": len(objects["DecisionPolicy"]),
        },
        "coverage": {
            "approved_family_count": len(APPROVED_FAMILY_LABELS),
            "covered_approved_family_count": len(APPROVED_FAMILY_LABELS) - len(missing_approved),
            "missing_approved_families": missing_approved,
        },
        "objects": objects,
        "relations": relations,
        "family_seed_view": families_view,
    }


def write_sop_seed_draft(
    chunks_path: str | Path = DEFAULT_SOP_CHUNKS,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, Any]:
    payload = build_sop_seed_draft(chunks_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "out": str(out_path),
        "counts": payload["counts"],
        "coverage": payload["coverage"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", default=DEFAULT_SOP_CHUNKS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = write_sop_seed_draft(args.chunks_path, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
