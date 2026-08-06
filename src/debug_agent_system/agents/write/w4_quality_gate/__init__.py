"""W4 quality gate for chat-derived schema candidates."""

from __future__ import annotations

import re
from typing import Any

from debug_agent_system.knowledge_v2.contracts import APPROVED_FAMILY_LABELS, FAMILY_SUBSYSTEM_EXPECTED, PSEUDO_FAMILY_LABELS, trim_text
from debug_agent_system.knowledge_v2.provenance import alignment_provenance_issues
from debug_agent_system.agents.write.non_sop_intake import is_sop_source_reference
from debug_agent_system.agents.write.non_sop_intake import (
    SOP_INCREMENTAL_CONTRACT,
)

NOISE_MARKERS = (
    "invited", "updated the group name", "joined", "撤回", "谢谢", "收到", "辛苦", "好的", "ok", "OK",
)
PROJECT_NOISE_MARKERS = ("需求", "排期", "会议", "工时", "上线", "验收", "合同", "报价", "jira")
REPORT_DOMINANT_MARKERS = (
    "每日反馈", "每日数据", "工作汇报", "现场工作汇报", "现场工作汇总", "现场工作：", "现场工作内容",
    "各位领导", "请领导知悉", "以上请领导知悉", "签单设备", "客户带板测试满意", "双章合同", "附条件采购合同",
    "交付信息", "项目背景", "项目信息背景", "项目重新启动", "客户地址", "收货地址", "培训员工", "培训人员",
    "客户人员", "需求记录", "项目进度", "上线计划",
)
POSITIVE_STATUS_MARKERS = (
    "没有问题", "未出现", "未再出现", "没发生过", "恢复正常", "正常测试", "持续观察", "观察未出现", "已恢复", "今天没有", "没有昨天也没有",
)
KNOWLEDGE_MARKERS = (
    "报错", "异常", "失败", "卡死", "漏检", "误报", "检查", "排查", "解决", "恢复", "日志", "诊断数据",
    "版本", "现场", "相机", "光源", "服务", "初始化", "IP", "ip",
)
ASK_INFO_GENERIC_MARKERS = ("发日志", "提供日志", "上传日志", "给日志", "发资料", "提供资料")
STRONG_DIAGNOSTIC_MARKERS = (
    "根因", "原因是", "定位到", "已解决", "恢复正常", "解决方案", "处理方案", "重装",
    "事件查看器", "Bugcheck", "bugcheck", "dmp", "DMP", "dump", "DLOG", "诊断数据",
)
PERSON_ONLY_LABEL_SUFFIXES = ("老师", "经理", "总", "工", "哥", "姐", "大佬")
PLACEHOLDER_LABELS = ("群聊噪声/待人工确认",)
NON_VERIFIED_OUTCOME_TYPES = {
    "ineffective", "partial_temporary", "mitigation_observed", "recurred",
    "pending_validation", "diagnostic_method", "context_not_root_cause",
}
WEAK_HANDOFF_LABEL_MARKERS = (
    "准备好了", "帮忙看一下", "帮忙看下", "麻烦看下", "麻烦帮忙", "邢工远程看下", "看下这个问题", "要一个报告",
)
V2_VARIANT_CONVERSATIONAL_MARKERS = (
    "@", "大佬", "帮忙", "看一下", "看下", "麻烦", "辛苦", "应该", "是不是", "还需要", "要一个报告",
    "客户没反馈", "今天没有", "未再出现", "恢复正常", "这个是", "捂脸",
)
V2_ACTION_NON_ATOMIC_MARKERS = (
    "@", "各位领导", "帮忙", "看一下", "看下", "麻烦", "辛苦", "这个是", "应该", "大佬", "要一个报告",
)
V2_ACTION_VERBS = (
    "检查", "确认", "分析", "收集", "导出", "提供", "升级", "回退", "重装", "更换", "排查", "观察", "验证",
    "启用", "卸载", "重启", "截图", "抓取", "记录", "修复", "关闭", "打开", "设置", "测试", "拔插", "安装",
    "使用", "查看", "核对", "对比", "比较", "测量", "监控", "清理", "清洁", "拆除", "调整", "恢复", "执行", "联系", "触摸",
    "优化", "限制", "降低", "调低", "拆下", "涂抹", "整理", "送修", "定位",
    "进入", "找到", "点击", "选择", "输入", "按下", "右键", "排除", "复现", "判断", "根据", "逐行",
    "复制", "将", "拔掉", "运行", "勾选", "开启", "换", "win+r", "切换", "还原", "重插", "查询", "更新",
    "轻推", "点胶", "增加", "分阶段", "拔除", "规范", "清除", "等待",
)
V2_VARIANT_WEAK_PREFIXES = (
    "各位领导", "今日反馈表格", "今日工作汇报", "今天", "中午之后", "此外", "4，此外", "这个是", "到达现场",
)
V2_ACTION_NOISY_MARKERS = (
    "各位领导", "今日反馈表格", "今日工作汇报", "问题已解决", "帮忙看", "帮忙分析", "请联系", "我远程看看",
    "有可能", "我们分析下", "尝试复现下",
)
V2_ACTION_HISTORY_MARKERS = (
    "已经", "曾经", "之前", "此前", "年前", "这台设备", "现场已", "客户已", "换过", "做过", "处理过", "执行过",
)
V2_FAMILY_VARIANT_MARKERS = {
    "工控机无法开机": ("无法开机", "不开机", "启动不了", "不能启动", "BIOS", "bios"),
    "工控机蓝屏": ("蓝屏", "bugcheck", "pfn", "pte"),
    "工控机异常重启": ("重启", "自动重启", "掉电", "断电", "关机", "供电中断", "电源"),
    "工控机黑屏无显示": ("黑屏", "无显示", "显示器", "屏幕"),
    "用户配置加载失败": ("配置", "加载", "user.cfg", "MES", "mes", "模板"),
    "程序运行卡顿": ("卡顿", "延迟", "耗时", "运行慢", "响应慢"),
    "软件卡死无响应": ("卡死", "无响应", "闪退", "崩溃", "退出"),
    "磁盘 I/O 异常": ("磁盘", "硬盘", "存储", "读写", "i/o", "io", "C盘", "D盘", "空间占用"),
    "顶升机构异常": ("顶板", "顶升", "气缸", "气管", "气流", "升降"),
    "相机拍摄失败": ("相机", "拍摄", "拍照", "残帧", "掉帧", "收图"),
    "扫码识别失败": ("扫码", "识别", "dm码", "二维码", "条码识别"),
    "外设连接不稳定": ("外设", "连接", "断连", "网卡", "wifi", "usb", "cxp", "接口"),
    "界面显示异常": ("显示", "界面", "花屏", "缩放", "分辨率", "显示不全"),
    "误报调优异常": ("误报", "调优"),
    "漏检调优异常": ("漏检", "调优"),
    "算法/程序调优异常": ("算法", "调优", "模型", "模板", "焊盘", "识别"),
    "复判站加载板卡异常": ("复判站", "加载板卡", "板卡加载", "加载用户配置", "user.cfg"),
}
V2_MORE_SPECIFIC_FAMILY_HINTS = (
    (("mes",), {"MES 过站异常"}),
    (("复判站", "加载"), {"复判站加载板卡异常"}),
    (("计算时间变长",), {"CT 时间异常增加", "程序运行卡顿"}),
)
V2_FAULT_OUTPUT_MODES = {
    "family_support_bundle",
    "variant_case_bundle",
    "atomic_case_bundle",
}
V2_STRONG_FAULT_VARIANT_MARKERS = (
    "报错", "异常", "失败", "卡死", "死机", "闪退", "崩溃", "蓝屏", "黑屏", "花屏",
    "重启", "无法", "不能", "不拍", "不响", "不亮", "不显示", "显示不全", "不进板", "不出板", "不测试",
    "掉线", "断连", "误报", "漏检", "超时", "残帧", "掉帧", "卡顿", "过热", "温度",
    "丢失", "损坏", "错误", "停止", "耗尽", "告警", "失效", "不稳定", "变长", "过长", "过慢", "不一致",
    "中断", "松动", "关机", "断电", "缺失", "报警", "频闪", "卡住", "不流畅", "模糊", "跑偏", "过低", "错位",
)
V2_NON_FAULT_VARIANT_MARKERS = (
    "交付安排", "培训安排", "培训人员不足", "架设完成", "验证功能无异常", "上线正常",
    "正常使用无故障", "现场安装调试", "设备交付", "培训与交付", "收集异常数据",
    "之后再没出现", "后未再出现", "回退版本后正常",
)
V2_SYNTHETIC_OUTCOME_MARKERS = (
    "提供了重要上下文，但还不是最终根因闭环",
    "作为诊断手段用于继续定位",
)
V2_PLACEHOLDER_OUTCOME_VALUES = {
    "camera_capture_chain", "software_version_change", "startup/init", "startup/init phase",
    "dmp", "root_cause", "verified", "fixed", "resolved",
}
TYPED_DECISION_VERSION = "w4_typed_decision.v1"
TYPED_MAPPING_VERSION = "kg_v2_typed_admission.v1"
TYPED_ADMISSION_TARGETS = {
    "fault_execution",
    "fault_support",
    "playbook",
    "procedure_library",
    "reference_constraint",
    "policy_template",
    "overlay",
    "evidence_only",
}
TYPED_TARGET_OBJECTS = {
    "KnowledgeDocument": "procedure_library",
    "KnowledgeSection": "procedure_library",
    "FaultFamily": "fault_execution",
    "FaultVariant": "fault_execution",
    "DiagnosticAction": "fault_execution",
    "ActionOutcome": "fault_support",
    "RequiredInfoSpec": "fault_support",
    "DiagnosticTrace": "fault_support",
    "Playbook": "playbook",
    "Procedure": "procedure_library",
    "ProcedureStep": "procedure_library",
    "ReferenceConstraint": "reference_constraint",
    "PolicyTemplate": "policy_template",
    "Overlay": "overlay",
    "EvidenceItem": "evidence_only",
}
TYPED_REQUIRED_EVIDENCE = {
    "fault_execution": ("raw_text", "source_anchor"),
    "fault_support": ("raw_text", "source_anchor", "evidence"),
    "playbook": ("raw_text", "source_anchor", "evidence"),
    "procedure_library": ("raw_text", "source_anchor", "evidence"),
    "reference_constraint": ("raw_text", "source_ref"),
    "policy_template": ("raw_text", "source_ref"),
    "overlay": ("raw_text", "source_ref"),
    "evidence_only": ("raw_text", "evidence"),
}
V2_SYNTHETIC_PENDING_SUMMARY_MARKERS = (
    # Backward compatibility for bundles created before ``outcome_origin``
    # was propagated.  New bundles must carry the explicit provenance field.
    "为建议动作，尚无已执行证据",
    "已执行，但当前证据未给出稳定验证结果",
    "的执行结果待进一步确认",
    "已执行或被建议执行，仍需继续验证",
)


def _nodes(candidate: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [node for node in candidate.get("nodes") or [] if node.get("type") == node_type]


def _candidate_text(candidate: dict[str, Any]) -> str:
    parts = [str(candidate.get(k) or "") for k in ("label", "symptom_raw", "conclusion", "category")]
    parts.extend(str(x) for x in candidate.get("debug_actions") or [])
    episode = candidate.get("episode") if isinstance(candidate.get("episode"), dict) else {}
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "noise_messages"):
        for msg in episode.get(key) or []:
            if isinstance(msg, dict):
                parts.append(str(msg.get("text") or msg.get("content_summary") or ""))
    return " ".join(parts)


def _diagnostic_outcomes(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node for node in _nodes(candidate, "DiagnosticOutcome")]
    for item in candidate.get("diagnostic_outcomes") or []:
        if isinstance(item, dict):
            out.append(item)
    return out


def _weak_action_chain(candidate: dict[str, Any]) -> bool:
    checks = _nodes(candidate, "DiagnosticCheck")
    if len(checks) >= 2:
        return False
    labels = [str(node.get("label") or node.get("how_to_check") or "").strip() for node in checks]
    labels = [label for label in labels if label]
    if not labels:
        labels = [str(item).strip() for item in candidate.get("debug_actions") or [] if str(item).strip()]
    if not labels:
        return True
    clean = labels[0]
    if len(clean) < 4:
        return True
    if any(marker in clean for marker in ("工作汇报", "建议可以", "看看", "有关系吗", "帮忙", "辛苦", "麻烦")):
        return True
    return False


def _fault_focus_confidence(candidate: dict[str, Any]) -> float:
    episode = candidate.get("episode") if isinstance(candidate.get("episode"), dict) else {}
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    try:
        return float(extracted.get("fault_focus_confidence") or 0.0)
    except Exception:
        return 0.0


def _edge_points_to_non_verified_outcome(candidate: dict[str, Any]) -> bool:
    solution_ids = {str(edge.get("to") or "") for edge in candidate.get("edges") or [] if isinstance(edge, dict) and edge.get("relation") == "resolved_by"}
    if not solution_ids:
        return False
    for outcome in _diagnostic_outcomes(candidate):
        if str(outcome.get("target_solution_id") or "") in solution_ids and str(outcome.get("outcome_type") or "") in NON_VERIFIED_OUTCOME_TYPES:
            return True
    return False


def _project_report_dominant(text: str, *, has_check: bool, has_solution: bool) -> bool:
    clean = str(text or "")
    if not clean:
        return False
    marker_count = sum(1 for marker in REPORT_DOMINANT_MARKERS if marker in clean)
    if marker_count == 0:
        return False
    strong_count = sum(1 for marker in STRONG_DIAGNOSTIC_MARKERS if marker in clean)
    fault_count = sum(1 for marker in ("报错", "异常", "失败", "卡死", "闪退", "蓝屏", "白屏", "漏检", "误报", "无法开机") if marker in clean)
    # Work reports often contain one true fault sentence mixed with project
    # tracking.  Keep them review-only unless they also carry concrete root
    # cause / diagnostic evidence or a proper check+solution chain.
    if marker_count >= 2 and strong_count == 0:
        return True
    if marker_count >= 1 and not (has_check and has_solution) and strong_count == 0 and len(clean) > 180:
        return True
    if marker_count >= 1 and fault_count <= 1 and strong_count == 0 and len(clean) > 120:
        return True
    return False


def _person_only_label(label: str, symptom: str) -> bool:
    clean = str(label or "").strip()
    if not clean:
        return False
    if len(clean) > 12:
        return False
    if not clean.endswith(PERSON_ONLY_LABEL_SUFFIXES):
        return False
    signal_text = f"{clean} {symptom}"
    return not any(marker in signal_text for marker in ("报错", "异常", "失败", "卡死", "闪退", "蓝屏", "白屏", "重启", "无法", "不能", "漏检", "误报"))


def _weak_fault_label(label: str, text: str) -> bool:
    clean = str(label or "").strip()
    if not clean:
        return True
    if clean in PLACEHOLDER_LABELS:
        return True
    if clean.startswith(("Image:", "File:", "Media:", "Video:")):
        return True
    if any(marker in clean for marker in WEAK_HANDOFF_LABEL_MARKERS):
        return True
    if any(marker in clean for marker in ("麻烦", "辛苦", "帮忙", "看一下", "看下", "看看")) and not any(marker in clean for marker in KNOWLEDGE_MARKERS):
        return True
    # A strong symptom elsewhere in the candidate text can still be reviewed,
    # but it must not pass the automatic quality gate with a placeholder label.
    return False


def _positive_status_update(label: str, text: str) -> bool:
    combined = f"{label} {text}"
    if not any(marker in combined for marker in POSITIVE_STATUS_MARKERS):
        return False
    if any(marker in combined for marker in (
        "异常", "故障", "失败", "蓝屏", "拍摄失败", "误报", "漏检", "卡死", "闪退",
        "无法开机", "初始化失败", "无检测图", "报错",
    )):
        # Keep historical fault summaries that also mention a true fault cause;
        # only pure status/no-issue updates should be gated as noise.
        if any(marker in label for marker in POSITIVE_STATUS_MARKERS):
            return True
        if label.startswith(("客户反馈说今天没有", "未再出现", "恢复正常", "正常测试")):
            return True
        return False
    return True


def _kg_v2_conversational_variant_label(label: str) -> bool:
    clean = str(label or "").strip()
    if not clean:
        return False
    if clean.startswith(("群聊候选", "[Image", "[File", "[Media")):
        return True
    if any(marker in clean for marker in V2_VARIANT_CONVERSATIONAL_MARKERS):
        return True
    if clean.endswith(("啊", "吗", "么", "呢")):
        return True
    return False


def _kg_v2_weak_variant_label(label: str) -> bool:
    clean = str(label or "").strip()
    if not clean:
        return False
    if clean.startswith(V2_VARIANT_WEAK_PREFIXES):
        return True
    # A leading software version or board-point count is a valid condition
    # when the rest of the label still names a concrete fault.  Reject only
    # numbered-list fragments that lack fault semantics.
    if (
        clean[:1].isdigit()
        or clean.startswith(("1.", "2.", "3.", "4.", "1，", "2，", "3，", "4，"))
    ) and not any(marker in clean for marker in V2_STRONG_FAULT_VARIANT_MARKERS):
        return True
    if any(marker in clean for marker in ("到达现场", "链接远程之前", "客户重启了设备", "中午之后频繁出现", "今天没有", "客户没反馈")):
        return True
    return False


def _kg_v2_non_atomic_actions(actions: list[dict[str, Any]]) -> bool:
    if not actions:
        return False
    suspicious = 0
    concise = 0
    for action in actions:
        label = str(action.get("label") or "").strip()
        if not label:
            suspicious += 1
            continue
        has_verb = any(verb in label for verb in V2_ACTION_VERBS)
        if len(label) <= 28 and has_verb and not any(marker in label for marker in V2_ACTION_NON_ATOMIC_MARKERS):
            concise += 1
        if (
            len(label) > 40
            or "\n" in label
            or label.startswith("@")
            or any(marker in label for marker in V2_ACTION_NON_ATOMIC_MARKERS)
            or label.endswith(("啊", "吗", "么", "呢"))
        ):
            suspicious += 1
    if concise == 0:
        return True
    return suspicious >= max(1, len(actions) // 2 + len(actions) % 2)


def _kg_v2_noisy_action_labels(actions: list[dict[str, Any]]) -> bool:
    for action in actions:
        label = str(action.get("label") or "").strip()
        if not label:
            continue
        if label.startswith("@") or label.startswith(("[Image", "[File", "[Media")):
            return True
        if any(marker in label for marker in V2_ACTION_NOISY_MARKERS):
            return True
        if len(label) > 60 and any(token in label for token in ("：", ":", "。", "；", ";")):
            return True
    return False


def _kg_v2_malformed_action_labels(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for action in actions:
        label = str(action.get("label") or "").strip()
        if not label:
            continue
        if any(label.count(left) != label.count(right) for left, right in (("(", ")"), ("（", "）"), ("[", "]"), ("【", "】"))):
            malformed.append(action)
            continue
        if label.endswith(("，然", ",然", "，然后", ",然后", "并", "以及")):
            malformed.append(action)
    return malformed


def _kg_v2_family_variant_mismatches(
    families: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    family_labels = {
        str(item.get("family_id") or ""): str(item.get("label") or "")
        for item in families
    }
    mismatches: list[dict[str, Any]] = []
    for variant in variants:
        family_label = family_labels.get(str(variant.get("family_id") or ""), "")
        markers = V2_FAMILY_VARIANT_MARKERS.get(family_label)
        if not markers:
            continue
        label = str(variant.get("label") or "")
        lowered = label.lower()
        if not any(marker.lower() in lowered for marker in markers):
            mismatches.append(variant)
    return mismatches


def _kg_v2_more_specific_family_mismatches(
    families: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    family_labels = {
        str(item.get("family_id") or ""): str(item.get("label") or "")
        for item in families
    }
    mismatches: list[dict[str, Any]] = []
    for variant in variants:
        family_label = family_labels.get(str(variant.get("family_id") or ""), "")
        lowered = str(variant.get("label") or "").lower()
        for required_markers, allowed_families in V2_MORE_SPECIFIC_FAMILY_HINTS:
            if all(marker.lower() in lowered for marker in required_markers) and family_label not in allowed_families:
                mismatches.append(variant)
                break
    return mismatches


def _kg_v2_duplicate_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    duplicates: list[dict[str, Any]] = []
    for outcome in outcomes:
        key = (
            str(outcome.get("outcome_type") or ""),
            "".join(str(outcome.get("summary") or "").lower().split()),
        )
        if not key[1]:
            continue
        if key in seen:
            duplicates.append(outcome)
        else:
            seen.add(key)
    return duplicates


def _kg_v2_duplicate_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    duplicates: list[dict[str, Any]] = []
    for action in actions:
        key = "".join(str(action.get("label") or "").lower().split())
        if not key:
            continue
        if key in seen:
            duplicates.append(action)
        else:
            seen.add(key)
    return duplicates


def _kg_v2_near_duplicate_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def core(label: str) -> str:
        value = "".join(ch for ch in label.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        for verb in sorted(V2_ACTION_VERBS, key=len, reverse=True):
            if value.startswith(verb.lower()):
                return value[len(verb):]
        return value

    duplicates: list[dict[str, Any]] = []
    seen: list[tuple[str, str, dict[str, Any]]] = []
    for action in actions:
        if str(action.get("source_kind") or "") in {"raw_doc", "hybrid"}:
            continue
        label = str(action.get("label") or "").strip()
        current = core(label)
        if len(current) < 4:
            seen.append((current, str(action.get("action_role") or ""), action))
            continue
        current_role = str(action.get("action_role") or "")
        for previous, previous_role, _ in seen:
            if (
                min(len(current), len(previous)) >= 4
                and (current in previous or previous in current)
            ):
                # The same target may legitimately be inspected, changed and
                # then verified.  Near-duplicate detection only applies within
                # the same semantic role.
                if current_role != previous_role:
                    continue
                duplicates.append(action)
                break
        seen.append((current, current_role, action))
    return duplicates


def _kg_v2_outcome_type_conflicts(outcomes: list[dict[str, Any]], source_text: str) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    source = str(source_text or "")
    success_markers = (
        "已解决", "解决了", "问题解决", "恢复正常", "已正常", "未再出现", "后正常", "后未复现",
        "测试正常", "速度正常", "验证正常",
    )
    recurrence_markers = ("复发", "再次出现", "重新出现", "又出现", "仍然", "再现", "recur")
    temporary_markers = ("暂时", "短期", "后续又", "仍出现", "复发", "待观察", "继续观察")
    pending_markers = ("待", "可能", "怀疑", "需验证", "仍需", "尚需", "尚未", "尚无", "继续", "观察", "未确认", "未定位", "不明确", "偶发", "未给出稳定验证结果")
    mitigation_markers = (
        "恢复", "缓解", "改善", "正常使用", "可正常", "后正常", "测试正常", "拍照正常", "开机正常",
        "正常运行", "未出现", "无法复现", "可用", "可继续", "绕过",
    )
    for outcome in outcomes:
        outcome_type = str(outcome.get("outcome_type") or "")
        summary = str(outcome.get("summary") or "").strip()
        if not summary:
            continue
        if outcome_type == "verified_fix":
            unexecuted = any(marker in summary for marker in ("未执行", "无法停线", "待现场", "需数小时", "计划执行", "建议执行"))
            source_ungrounded = bool(source) and not any(marker in source for marker in success_markers)
            if unexecuted or source_ungrounded:
                conflicts.append(outcome)
        elif outcome_type == "diagnostic_method":
            if any(marker in summary for marker in success_markers):
                conflicts.append(outcome)
            elif any(marker in summary for marker in ("待分析", "待确认", "可能涉及", "可能是", "尚未分析")):
                conflicts.append(outcome)
        elif outcome_type == "pending_validation":
            if not any(marker in summary for marker in pending_markers):
                conflicts.append(outcome)
        elif outcome_type == "ineffective" and not any(marker in summary for marker in (
            "无效", "未解决", "未能解决", "仍然存在", "仍会", "后仍", "不生效", "没用", "非根因", "没有改善",
        )):
            conflicts.append(outcome)
        elif outcome_type == "partial_temporary":
            has_temporary_evidence = any(marker in summary for marker in (*temporary_markers, *mitigation_markers))
            has_later_failure = any(marker in source for marker in (*recurrence_markers, "然后设备无法", "随后无法", "之后无法", "仍无法"))
            if not has_temporary_evidence and not has_later_failure:
                conflicts.append(outcome)
        elif outcome_type == "mitigation_observed" and not any(marker in summary for marker in mitigation_markers):
            conflicts.append(outcome)
        elif outcome_type == "recurred" and not any(marker in summary.lower() for marker in recurrence_markers):
            conflicts.append(outcome)
    return conflicts


def _kg_v2_non_fault_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject status/project labels that are not technical fault variants.

    A chat segment may contain delivery or training updates next to a real
    fault.  Such updates are useful evidence, but must not become executable
    fault variants merely because the extraction bridge supplied actions.
    """

    invalid: list[dict[str, Any]] = []
    for variant in variants:
        label = str(variant.get("label") or "").strip()
        if not label:
            continue
        if any(marker in label for marker in V2_NON_FAULT_VARIANT_MARKERS) or not any(
            marker in label for marker in V2_STRONG_FAULT_VARIANT_MARKERS
        ):
            invalid.append(variant)
    return invalid


def _kg_v2_synthetic_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find bridge-generated prose/placeholders that are not observed outcomes."""

    invalid: list[dict[str, Any]] = []
    for outcome in outcomes:
        summary = str(outcome.get("summary") or "").strip()
        if (
            any(marker in summary for marker in V2_SYNTHETIC_OUTCOME_MARKERS)
            or summary.lower() in V2_PLACEHOLDER_OUTCOME_VALUES
        ):
            invalid.append(outcome)
    return invalid


def _kg_v2_focused_case_text(
    variants: list[dict[str, Any]], traces: list[dict[str, Any]], source_cases: list[dict[str, Any]]
) -> str:
    parts: list[str] = []
    for item in variants:
        parts.extend(str(item.get(key) or "") for key in ("label", "summary"))
    for item in traces:
        parts.append(str(item.get("summary") or ""))
    for item in source_cases:
        parts.extend(str(item.get(key) or "") for key in ("title", "summary"))
    return " ".join(part for part in parts if part)


def _kg_v2_missing_observed_resolution(
    outcomes: list[dict[str, Any]], focused_case_text: str
) -> bool:
    """Detect a source-observed recovery that W2 failed to model as an outcome."""

    text = str(focused_case_text or "")
    observed = any(marker in text for marker in (
        "恢复正常", "恢复使用", "正常拍图", "正常测试", "不再闪退", "未再出现", "再没出现",
        "问题已解决", "问题解决", "重启恢复", "后恢复",
    ))
    if not observed:
        return False
    return not any(
        str(item.get("outcome_type") or "") in {"verified_fix", "partial_temporary", "mitigation_observed"}
        for item in outcomes
    )


def _kg_v2_result_statement_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Action labels must say what to do, not embed the observed result."""

    invalid: list[dict[str, Any]] = []
    for action in actions:
        label = str(action.get("label") or "").strip()
        if any(marker in label for marker in ("分析日志发现", "检查日志发现", "查看日志发现", "排查发现")):
            invalid.append(action)
    return invalid


def _kg_v2_multi_operation_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject one node that combines independently executable operations."""

    invalid: list[dict[str, Any]] = []
    for action in actions:
        label = str(action.get("label") or "").strip()
        if label.startswith("等待") and "后再" in label:
            # This is one conditional operation (wait for dump completion,
            # then reboot), not two independently executable nodes.
            continue
        for connector in ("并", "然后", "再"):
            if connector not in label:
                continue
            clauses = [part.strip() for part in label.split(connector) if part.strip()]
            if len(clauses) == 2 and clauses[1].startswith(("观察", "验证", "复验", "复测")):
                continue
            if len(clauses) >= 2 and sum(any(verb in part for verb in V2_ACTION_VERBS) for part in clauses) >= 2:
                invalid.append(action)
                break
    return invalid


V2_ACTION_GROUNDING_STOP_BIGRAMS = {
    "检查", "确认", "分析", "查看", "进行", "当前", "相关", "状态", "情况", "问题", "现场",
    "设备", "软件", "程序", "日志", "版本", "是否", "使用", "结果", "异常",
}


def _kg_v2_ungrounded_actions(actions: list[dict[str, Any]], source_text: str) -> list[dict[str, Any]]:
    """Return actions whose distinctive terms are absent from current evidence.

    This is deliberately a majority gate.  One inferred follow-up check may be
    useful, but a candidate whose action chain is mostly copied from alignment
    context must not auto-admit as a historical trace.
    """

    source = str(source_text or "").lower()
    if not source or not actions:
        return []
    ungrounded: list[dict[str, Any]] = []
    for action in actions:
        label = str(action.get("label") or "").strip().lower()
        ascii_terms = {
            token for token in re.findall(r"[a-z][a-z0-9_.+-]{2,}", label)
            if token not in {"aoi", "the", "and"}
        }
        chinese_terms: set[str] = set()
        for run in re.findall(r"[\u4e00-\u9fff]+", label):
            for index in range(max(0, len(run) - 1)):
                term = run[index:index + 2]
                if len(term) == 2 and term not in V2_ACTION_GROUNDING_STOP_BIGRAMS:
                    chinese_terms.add(term)
        terms = ascii_terms | chinese_terms
        if not terms or not any(term in source for term in terms):
            ungrounded.append(action)
    return ungrounded if len(ungrounded) > len(actions) // 2 else []


def _kg_v2_outcome_evidence_outside_source(
    outcomes: list[dict[str, Any]], evidence_items: list[dict[str, Any]], source_message_ids: list[str]
) -> list[dict[str, Any]]:
    allowed = {str(value) for value in source_message_ids if str(value)}
    if not allowed:
        return []
    external_id_by_evidence = {
        str(item.get("evidence_id") or ""): str(item.get("external_id") or "")
        for item in evidence_items
        if str(item.get("evidence_id") or "")
    }
    leaked: list[dict[str, Any]] = []
    for outcome in outcomes:
        for evidence_id in outcome.get("evidence_ids") or []:
            external_id = external_id_by_evidence.get(str(evidence_id), "")
            if external_id and external_id not in allowed:
                leaked.append(outcome)
                break
    return leaked


def _kg_v2_outcome_evidence_without_text_support(
    outcomes: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    source_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject semantic outcomes supported only by an image/file placeholder.

    Metadata proves that an attachment exists, not what is visible inside it.
    Without OCR/tool extraction or a textual follow-up, W2 must not infer a
    resource reading or diagnostic result from ``[Image: ...]`` alone.
    """

    if not source_messages:
        return []
    external_by_evidence = {
        str(item.get("evidence_id") or ""): str(item.get("external_id") or "")
        for item in evidence_items
        if str(item.get("evidence_id") or "")
    }
    text_by_message = {
        str(item.get("message_id") or ""): str(item.get("text") or "").strip()
        for item in source_messages
        if str(item.get("message_id") or "")
    }

    def substantive(value: str) -> bool:
        clean = re.sub(r"\[(?:Image|Media|File):[^\]]+\]", " ", str(value or ""), flags=re.IGNORECASE)
        clean = re.sub(r"@\S+", " ", clean)
        clean = " ".join(clean.split())
        return len(clean) >= 8

    unsupported: list[dict[str, Any]] = []
    for outcome in outcomes:
        evidence_ids = [str(value) for value in outcome.get("evidence_ids") or [] if str(value)]
        if not evidence_ids:
            continue
        external_ids = [external_by_evidence.get(value, "") for value in evidence_ids]
        texts = [text_by_message.get(value, "") for value in external_ids if value]
        if not texts or not any(substantive(value) for value in texts):
            unsupported.append(outcome)
    return unsupported


def _kg_v2_executable_action_label(value: str) -> bool:
    """Require an executable instruction, not a chat/result statement."""

    label = str(value or "").strip()
    if not label or label.endswith(("?", "？")):
        return False
    if re.match(
        r"^(?:初步)?(?:排查是|排查发现|分析发现|检查发现|查看日志发现|确认是|判断为|定位到)",
        label,
    ):
        return False
    if re.search(r"(?:后|之后|然后|再然后)\s*$", label):
        return False
    if re.search(
        r"(?:后|之后)(?:开机测试.*正常|测试.*正常|验证.*正常|恢复|正常|已|未|仍|依然|"
        r"无法|不能|解决|无效|失败|再次出现|复发)",
        label,
    ):
        return False
    weak = re.sub(r"^(?:后续)?(?:建议|推荐|可以尝试|计划|先|再|继续)\s*", "", label).strip()
    if re.fullmatch(
        r"(?:重新)?(?:重启|调整|更换|替换|检查|确认|测试|验证|观察|分析|排查)(?:下|一下|看看|看下|看一下|试试)?",
        weak,
    ):
        return False
    if label.startswith((
        "如果", "若", "是否", "有无", "能否", "问题", "目标", "结论", "现象", "发生时间",
        "现场", "客户", "我的", "这个", "还是", "是我们", "应该", "已经", "曾经", "之前", "目前",
        "确认客户认为", "确认客户反馈", "确认现场反馈",
    )):
        return False
    if any(marker in label for marker in (
        "后恢复正常", "后故障消除", "后未再", "后就没", "已经解决", "已解决", "就重启了", "已升级到",
    )):
        return False
    if "后" in label and any(marker in label for marker in ("依然", "仍然", "仍会", "未解决", "无效", "失败", "闪退", "报错")):
        return False
    if any(marker in label for marker in ("等下我", "和夜班说", "提供一下日志和具体时间")):
        return False
    if label.startswith(("每日", "每天")) and "重启" in label:
        return True
    core = label
    temporal = re.match(r"^在[^，,。；;]{1,16}时", core)
    if temporal:
        core = core[temporal.end():].lstrip()
    for prefix in ("重新", "反复", "先", "再", "同时", "逐行", "彻底", "手动", "临时", "每日", "每天", "按SOP"):
        if core.startswith(prefix):
            core = core[len(prefix):].lstrip()
            break
    if core.startswith(V2_ACTION_VERBS):
        return True
    return len(label) <= 40 and label.rstrip("：:").endswith(("测试", "验证", "检查", "分析", "排查", "确认", "修复"))


def _typed_objects(envelope: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    objects: dict[str, list[dict[str, Any]]] = {}
    for source in _typed_containers(envelope):
        raw_objects = source.get("objects") if isinstance(source.get("objects"), dict) else {}
        for object_type, items in raw_objects.items():
            if not isinstance(items, list):
                continue
            objects.setdefault(str(object_type), []).extend(item for item in items if isinstance(item, dict))
    return objects


def _typed_containers(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [envelope]
    for key in ("payload", "evidence_pack"):
        value = envelope.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _typed_first(envelope: dict[str, Any], *keys: str) -> Any:
    for source in _typed_containers(envelope):
        for key in keys:
            value = source.get(key)
            if value not in (None, "", [], {}):
                return value
    return ""


def _typed_list(envelope: dict[str, Any], *keys: str) -> list[Any]:
    out: list[Any] = []
    for source in _typed_containers(envelope):
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                out.extend(value)
            elif value not in (None, "", [], {}):
                out.append(value)
    return out


def _typed_source_kinds(envelope: dict[str, Any]) -> set[str]:
    kinds: set[str] = set()
    for source_container in _typed_containers(envelope):
        for key in ("source_type", "source_kind"):
            value = str(source_container.get(key) or "").strip().lower()
            if value:
                kinds.add(value)
        source = source_container.get("source") if isinstance(source_container.get("source"), dict) else {}
        for key in ("source_type", "source_kind", "kind", "type"):
            value = str(source.get(key) or "").strip().lower()
            if value:
                kinds.add(value)
        for key in ("source_ref", "path", "url"):
            raw_value = source_container.get(key) or source.get(key) or ""
            # Structured source_ref carries message IDs as well as paths.  Do
            # not stringify the whole mapping: a legitimate field message such
            # as ``msg:SOP参数无异常`` would otherwise be mistaken for an SOP
            # document source.  Only path-bearing members participate in the
            # SOP source boundary.
            if isinstance(raw_value, dict):
                values = [
                    raw_value.get(ref_key)
                    for ref_key in ("path", "source_path", "payload_ref", "url")
                ]
            else:
                values = [raw_value]
            if any(is_sop_source_reference(str(value or "").strip().lower()) for value in values):
                kinds.add("sop")
    for items in _typed_objects(envelope).values():
        for item in items:
            for key in ("source_type", "source_kind"):
                value = str(item.get(key) or "").strip().lower()
                if value:
                    kinds.add(value)
            source_ref = str(item.get("source_ref") or "").strip().lower()
            if is_sop_source_reference(source_ref):
                kinds.add("sop")
    return kinds


def _typed_sop_incremental_allowed(envelope: dict[str, Any]) -> bool:
    """Recognize only the explicit, versioned SOP document intake path."""

    for container in _typed_containers(envelope):
        if str(container.get("source_type") or "").strip().lower() != "sop_doc":
            continue
        metadata = (
            container.get("metadata")
            if isinstance(container.get("metadata"), dict)
            else {}
        )
        if (
            str(container.get("source_kind") or "").strip().lower() == "sop"
            and str(metadata.get("incremental_source_contract") or "")
            == SOP_INCREMENTAL_CONTRACT
        ):
            return True
    return False


def _typed_text(envelope: dict[str, Any]) -> str:
    parts: list[str] = []
    for source_container in _typed_containers(envelope):
        parts.extend(str(source_container.get(key) or "") for key in ("raw_text", "text", "original_text", "title", "summary"))
        source = source_container.get("source") if isinstance(source_container.get("source"), dict) else {}
        parts.extend(str(source.get(key) or "") for key in ("raw_text", "text", "title", "summary", "source_ref"))
    for items in _typed_objects(envelope).values():
        for item in items:
            parts.extend(str(item.get(key) or "") for key in ("label", "title", "summary", "content", "source_ref"))
    return " ".join(part for part in parts if part)


TYPED_FAULT_RELEVANCE_MARKERS = (
    "异常", "故障", "报错", "错误", "失败", "无法", "不能", "卡死", "卡顿",
    "崩溃", "闪退", "蓝屏", "重启", "死机", "漏检", "误报", "超时", "断连",
    "丢失", "损坏", "不响应", "不拍照", "不出图", "未识别", "报警",
    "error", "failed", "failure", "fault", "crash", "timeout", "exception",
    "disconnect", "missing", "corrupt", "hang", "freeze", "reboot", "bsod",
)


def _typed_fault_relevant(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in TYPED_FAULT_RELEVANCE_MARKERS)


def _typed_has_evidence(envelope: dict[str, Any], evidence_name: str) -> bool:
    if evidence_name == "raw_text":
        return bool(str(_typed_first(envelope, "raw_text", "text", "original_text") or "").strip())
    if evidence_name == "source_case":
        return bool(_typed_objects(envelope).get("SourceCase") or _typed_list(envelope, "source_case", "source_cases"))
    if evidence_name == "source_anchor":
        return bool(
            _typed_objects(envelope).get("SourceCase")
            or _typed_objects(envelope).get("KnowledgeDocument")
            or _typed_list(envelope, "source_case", "source_cases", "knowledge_document")
        )
    if evidence_name == "outcome_evidence":
        explicit = _typed_list(envelope, "outcome_evidence", "outcome_evidence_items")
        if explicit:
            return True
        outcomes = _typed_objects(envelope).get("ActionOutcome") or []
        return bool(outcomes) and all(
            item.get("evidence_ids") or item.get("evidence_message_ids")
            for item in outcomes
        )
    if evidence_name == "source_ref":
        for source_container in _typed_containers(envelope):
            source = source_container.get("source") if isinstance(source_container.get("source"), dict) else {}
            if source_container.get("source_ref") or source.get("source_ref") or source.get("url") or source.get("path"):
                return True
        return False
    if evidence_name == "evidence":
        return bool(_typed_objects(envelope).get("EvidenceItem") or _typed_list(envelope, "evidence", "evidence_items"))
    return False


def _infer_typed_admission_target(envelope: dict[str, Any]) -> str:
    explicit = str(_typed_first(envelope, "admission_target", "target") or "").strip()
    if explicit:
        return explicit
    for source_container in _typed_containers(envelope):
        strategy = source_container.get("strategy") if isinstance(source_container.get("strategy"), dict) else {}
        strategy_target = str(strategy.get("admission_target") or strategy.get("kg_output_mode") or "").strip()
        if strategy_target in TYPED_ADMISSION_TARGETS:
            return strategy_target
    for object_type, items in _typed_objects(envelope).items():
        if items and object_type in TYPED_TARGET_OBJECTS:
            return TYPED_TARGET_OBJECTS[object_type]
    return "evidence_only"


def _typed_admission_readiness(
    envelope: dict[str, Any],
    *,
    schema_valid: bool,
    missing_evidence: list[str],
) -> str:
    """Return the highest graph layer supported by current-source evidence."""

    if not schema_valid or "raw_text" in missing_evidence:
        return "not_ready"
    objects = _typed_objects(envelope)
    has_evidence = bool(objects.get("SourceCase") and objects.get("EvidenceItem"))
    if not has_evidence:
        return "not_ready"
    families = objects.get("FaultFamily") or []
    variants = objects.get("FaultVariant") or []
    if not (families and variants):
        return "evidence_ready"
    actions = objects.get("DiagnosticAction") or []
    outcomes = objects.get("ActionOutcome") or []
    traces = objects.get("DiagnosticTrace") or []
    action_ids = {str(item.get("action_id") or "") for item in actions if str(item.get("action_id") or "")}
    outcomes_linked = bool(outcomes) and all(
        str(item.get("action_id") or "") in action_ids
        and bool(item.get("evidence_ids") or item.get("evidence_message_ids"))
        for item in outcomes
    )
    if actions and traces and outcomes_linked and "outcome_evidence" not in missing_evidence:
        return "execution_ready"
    return "case_ready"


def _typed_outcome_origin(outcome: dict[str, Any]) -> str:
    """Return explicit outcome provenance, with legacy-only text fallback."""

    explicit = str(outcome.get("outcome_origin") or "").strip()
    if explicit:
        return explicit
    summary = str(outcome.get("summary") or "")
    if str(outcome.get("outcome_type") or "") == "pending_validation" and any(
        marker in summary for marker in V2_SYNTHETIC_PENDING_SUMMARY_MARKERS
    ):
        return "synthetic_fallback"
    return "legacy_unspecified"


def _typed_execution_policy_readiness(envelope: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Classify policy evidence separately from graph-structure readiness.

    ``execution_ready`` means the complete case/trace graph can be retained.
    It does not imply that a reusable execution policy may be projected.  At
    least one actual action with a non-pending observed outcome is required
    for automatic policy materialization; pending-only traces remain valuable
    support/history and can still be reviewed by W6.
    """

    objects = _typed_objects(envelope)
    actions = [item for item in objects.get("DiagnosticAction") or [] if isinstance(item, dict)]
    outcomes = [item for item in objects.get("ActionOutcome") or [] if isinstance(item, dict)]
    traces = [item for item in objects.get("DiagnosticTrace") or [] if isinstance(item, dict)]
    if not actions or not traces:
        return "not_applicable", {
            "outcome_count": len(outcomes),
            "pending_count": sum(str(item.get("outcome_type") or "") == "pending_validation" for item in outcomes),
            "synthetic_pending_count": sum(
                str(item.get("outcome_type") or "") == "pending_validation"
                and _typed_outcome_origin(item) == "synthetic_fallback"
                for item in outcomes
            ),
            "observed_actual_count": 0,
        }
    actual_action_ids = {
        str(item.get("action_id") or "")
        for item in actions
        if str(item.get("execution_status") or "") == "actual"
    }
    actual_action_ids.update(
        str(action_id)
        for trace in traces
        for action_id in trace.get("actual_action_ids") or []
        if str(action_id)
    )
    pending = [item for item in outcomes if str(item.get("outcome_type") or "") == "pending_validation"]
    synthetic_pending = [item for item in pending if _typed_outcome_origin(item) == "synthetic_fallback"]
    promoted_only_action_ids = {
        str(item.get("action_id") or "")
        for item in actions
        if str(item.get("evidence_scope") or "") == "w7_promoted_only"
    }
    observed_actual = [
        item for item in outcomes
        if str(item.get("outcome_type") or "") != "pending_validation"
        and str(item.get("action_id") or "") in actual_action_ids
        and _typed_outcome_origin(item) != "synthetic_fallback"
    ]
    counts = {
        "outcome_count": len(outcomes),
        "pending_count": len(pending),
        "synthetic_pending_count": len(synthetic_pending),
        "observed_actual_count": len(observed_actual),
        "promoted_only_action_count": len(promoted_only_action_ids),
    }
    if promoted_only_action_ids:
        return "contains_promoted_only_action", counts
    if observed_actual:
        return "observed_execution", counts
    if outcomes and len(pending) == len(outcomes):
        return "pending_only", counts
    return "unobserved_execution", counts


class QualityGateAgent:
    """W4: confidence/clarity/relevance/schema gate before review queues."""

    def __init__(self, *, threshold: float = 0.65, min_confidence: float = 0.35) -> None:
        self.threshold = threshold
        self.min_confidence = min_confidence

    def score(self, candidate: dict[str, Any]) -> dict[str, Any]:
        confidence = float(candidate.get("confidence") or candidate.get("score") or 0.0)
        text = _candidate_text(candidate)
        episode = candidate.get("episode") if isinstance(candidate.get("episode"), dict) else {}
        completeness = str(episode.get("completeness") or "")
        has_nodes = bool(candidate.get("nodes"))
        has_edges = bool(candidate.get("edges"))
        has_evidence = bool(candidate.get("evidence_ids") or candidate.get("source_offsets"))
        has_check = bool(_nodes(candidate, "DiagnosticCheck"))
        has_solution = bool(_nodes(candidate, "Solution"))
        schema_valid = bool(candidate.get("schema_valid"))
        schema_issues = [str(x) for x in candidate.get("schema_issues") or []]
        person_only_label = _person_only_label(str(candidate.get("label") or ""), str(candidate.get("symptom_raw") or ""))
        weak_fault_label = _weak_fault_label(str(candidate.get("label") or ""), text)
        fault_focus_conf = _fault_focus_confidence(candidate)

        clarity = 0.2
        if candidate.get("label") or candidate.get("symptom_raw"):
            clarity += 0.25
        if has_check:
            clarity += 0.2
        if has_solution:
            clarity += 0.15
        if has_evidence:
            clarity += 0.15
        if completeness == "complete":
            clarity += 0.05
        clarity = min(1.0, clarity)

        relevance = 0.2
        if any(k in text for k in KNOWLEDGE_MARKERS):
            relevance += 0.35
        if candidate.get("category"):
            relevance += 0.15
        if has_nodes and has_edges:
            relevance += 0.15
        if candidate.get("log_paths") or candidate.get("versions") or candidate.get("sites"):
            relevance += 0.15
        project_noise = any(k.lower() in text.lower() for k in PROJECT_NOISE_MARKERS) and not any(k in text for k in KNOWLEDGE_MARKERS)
        project_report_noise = _project_report_dominant(text, has_check=has_check, has_solution=has_solution)
        generic_noise = any(k.lower() in text.lower() for k in NOISE_MARKERS) and not has_check and not has_solution
        positive_status_noise = _positive_status_update(str(candidate.get("label") or ""), text)
        if project_noise or project_report_noise or generic_noise or positive_status_noise:
            relevance -= 0.45
        relevance = max(0.0, min(1.0, relevance))

        schema_validity = 1.0 if schema_valid else 0.0
        weighted = round(confidence * 0.35 + clarity * 0.25 + relevance * 0.25 + schema_validity * 0.15, 4)
        issues: list[str] = []
        if confidence < self.min_confidence:
            issues.append("low_confidence")
        if not has_evidence:
            issues.append("missing_evidence")
        if not has_check and not has_solution:
            issues.append("missing_check_or_solution")
        if _weak_action_chain(candidate):
            issues.append("weak_action_chain")
        if fault_focus_conf and fault_focus_conf < 0.45:
            issues.append("weak_fault_focus")
        if completeness == "noise":
            issues.append("noise_episode")
        if project_noise or project_report_noise or generic_noise:
            issues.append("review_only_noise")
        if positive_status_noise:
            issues.append("positive_status_not_fault")
        if person_only_label:
            issues.append("weak_person_only_label")
        if weak_fault_label:
            issues.append("weak_fault_label")
        if not schema_valid:
            issues.append("schema_invalid")
            issues.extend(schema_issues)
        if _edge_points_to_non_verified_outcome(candidate):
            issues.append("resolved_by_non_verified_outcome")
        for outcome in _diagnostic_outcomes(candidate):
            outcome_type = str(outcome.get("outcome_type") or "")
            if outcome_type in NON_VERIFIED_OUTCOME_TYPES:
                issues.append(f"historical_outcome:{outcome_type}")
            if outcome_type == "verified_fix":
                issues.append("historical_outcome:verified_fix")
                if outcome.get("needs_confirmation"):
                    issues.append("verified_fix_requires_human_confirmation")
            if outcome.get("high_cost"):
                issues.append("high_cost_requires_human")
            if outcome.get("destructive"):
                issues.append("destructive_requires_human")
            if outcome_type == "diagnostic_method":
                diagnostic_solution_id = str(outcome.get("target_solution_id") or "")
                if diagnostic_solution_id and any(
                    isinstance(edge, dict) and edge.get("relation") == "resolved_by" and str(edge.get("to") or "") == diagnostic_solution_id
                    for edge in candidate.get("edges") or []
                ):
                    issues.append("diagnostic_method_not_solution")
        hard_fail = any(issue in set(issues) for issue in ("missing_evidence", "missing_check_or_solution", "weak_action_chain", "weak_fault_focus", "noise_episode", "review_only_noise", "positive_status_not_fault", "weak_person_only_label", "weak_fault_label", "schema_invalid", "resolved_by_non_verified_outcome", "diagnostic_method_not_solution"))
        passed = weighted >= self.threshold and confidence >= self.min_confidence and not hard_fail
        return {
            "confidence": round(confidence, 4),
            "clarity": round(clarity, 4),
            "relevance": round(relevance, 4),
            "schema_validity": round(schema_validity, 4),
            "weighted_sum": weighted,
            "threshold": self.threshold,
            "passed": passed,
            "issues": sorted(set(issues)),
            "observability": {
                "agent_id": "W4",
                "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
                "episode_completeness": completeness,
            },
        }

    def score_v2_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        objects = bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {}
        relations = bundle.get("relations") if isinstance(bundle.get("relations"), list) else []
        schema_valid = bool(bundle.get("schema_valid"))
        schema_issues = [str(x) for x in bundle.get("schema_issues") or []]
        families = [item for item in objects.get("FaultFamily") or [] if isinstance(item, dict)]
        variants = [item for item in objects.get("FaultVariant") or [] if isinstance(item, dict)]
        actions = [item for item in objects.get("DiagnosticAction") or [] if isinstance(item, dict)]
        outcomes = [item for item in objects.get("ActionOutcome") or [] if isinstance(item, dict)]
        required_info = [item for item in objects.get("RequiredInfoSpec") or [] if isinstance(item, dict)]
        traces = [item for item in objects.get("DiagnosticTrace") or [] if isinstance(item, dict)]
        source_cases = [item for item in objects.get("SourceCase") or [] if isinstance(item, dict)]
        knowledge_documents = [item for item in objects.get("KnowledgeDocument") or [] if isinstance(item, dict)]
        evidence_items = [item for item in objects.get("EvidenceItem") or [] if isinstance(item, dict)]
        strategy = bundle.get("strategy") if isinstance(bundle.get("strategy"), dict) else {}
        output_mode = str(strategy.get("kg_output_mode") or "")
        refinement = bundle.get("w3_refinement") if isinstance(bundle.get("w3_refinement"), dict) else {}
        refinement_flags = [str(item) for item in refinement.get("review_flags") or []]
        source_text = str(bundle.get("source_text") or bundle.get("text") or "")
        source_message_ids = [str(value) for value in bundle.get("source_message_ids") or [] if str(value)]
        source_messages = [item for item in bundle.get("source_messages") or [] if isinstance(item, dict)]

        family_labels = [str(item.get("label") or "") for item in families]
        variant_labels = [str(item.get("label") or "") for item in variants]
        canonical_family_count = sum(1 for label in family_labels if label in APPROVED_FAMILY_LABELS)
        evidence_relations = [item for item in relations if item.get("relation") == "evidences"]

        clarity = 0.2
        if families and all(family_labels):
            clarity += 0.25
        if variants and all(variant_labels):
            clarity += 0.15
        if actions:
            clarity += 0.15
        if required_info:
            clarity += 0.1
        if outcomes:
            clarity += 0.1
        if (source_cases or knowledge_documents) and evidence_items and evidence_relations:
            clarity += 0.1
        clarity = min(1.0, clarity)

        relevance = 0.2
        if families and canonical_family_count == len(families):
            relevance += 0.35
        elif families:
            relevance += 0.1
        if all(str(item.get("subsystem") or "") for item in families):
            relevance += 0.1
        if actions:
            relevance += 0.15
        if required_info:
            relevance += 0.1
        if outcomes:
            relevance += 0.1
        relevance = min(1.0, relevance)

        confidence = 0.4
        if families and canonical_family_count == len(families):
            confidence += 0.25
        if schema_valid:
            confidence += 0.15
        if actions and outcomes:
            confidence += 0.1
        if required_info:
            confidence += 0.05
        confidence = min(1.0, confidence)

        schema_validity = 1.0 if schema_valid else 0.0
        weighted = round(confidence * 0.35 + clarity * 0.25 + relevance * 0.25 + schema_validity * 0.15, 4)
        issues: list[str] = []
        item_issues: list[dict[str, str]] = []
        if not families or any(not label for label in family_labels):
            issues.append("kg_v2_missing_family")
        if not variants or any(not label for label in variant_labels):
            issues.append("kg_v2_missing_variant")
        for family in families:
            family_id = str(family.get("family_id") or "")
            label = str(family.get("label") or "")
            subsystem = str(family.get("subsystem") or "")
            if label and label not in APPROVED_FAMILY_LABELS:
                issues.append("kg_v2_noncanonical_family")
                item_issues.append({"object_id": family_id, "issue": "noncanonical_family", "value": label})
            if label in PSEUDO_FAMILY_LABELS:
                issues.append("kg_v2_pseudo_family")
                item_issues.append({"object_id": family_id, "issue": "pseudo_family", "value": label})
            expected = FAMILY_SUBSYSTEM_EXPECTED.get(label)
            if expected and subsystem and subsystem != expected:
                issues.append("kg_v2_family_subsystem_mismatch")
                item_issues.append({"object_id": family_id, "issue": "family_subsystem_mismatch", "value": subsystem})
        family_label_by_id = {
            str(item.get("family_id") or ""): str(item.get("label") or "")
            for item in families
        }
        for variant in variants:
            variant_id = str(variant.get("variant_id") or "")
            label = str(variant.get("label") or "")
            family_label = family_label_by_id.get(str(variant.get("family_id") or ""), "")
            checks: list[str] = []
            if family_label and label == family_label:
                checks.append("family_variant_label_collision")
                issues.append("kg_v2_family_variant_label_collision")
            if len(label) > 40:
                checks.append("long_variant_label")
                issues.append("kg_v2_long_variant_label")
            if _kg_v2_conversational_variant_label(label):
                checks.append("conversational_variant_label")
                issues.append("kg_v2_conversational_variant_label")
            if _kg_v2_weak_variant_label(label):
                checks.append("weak_variant_label")
                issues.append("kg_v2_weak_variant_label")
            if label.startswith(("我这个现场", "现场反馈", "客户反馈")) or label.endswith(("是什么问题", "怎么处理", "怎么办", "如何处理", "吗", "么")):
                checks.append("questionish_variant_label")
                issues.append("kg_v2_questionish_variant_label")
            item_issues.extend({"object_id": variant_id, "issue": check, "value": label} for check in checks)
        family_variant_mismatches = _kg_v2_family_variant_mismatches(families, variants)
        if family_variant_mismatches:
            issues.append("kg_v2_family_variant_semantic_mismatch")
            item_issues.extend(
                {
                    "object_id": str(item.get("variant_id") or ""),
                    "issue": "family_variant_semantic_mismatch",
                    "value": str(item.get("label") or ""),
                }
                for item in family_variant_mismatches
            )
        more_specific_family_mismatches = _kg_v2_more_specific_family_mismatches(families, variants)
        if more_specific_family_mismatches:
            issues.append("kg_v2_more_specific_family_available")
            item_issues.extend(
                {
                    "object_id": str(item.get("variant_id") or ""),
                    "issue": "more_specific_family_available",
                    "value": str(item.get("label") or ""),
                }
                for item in more_specific_family_mismatches
            )
        non_fault_variants = _kg_v2_non_fault_variants(variants)
        if non_fault_variants:
            issues.append("kg_v2_non_fault_variant")
            item_issues.extend(
                {
                    "object_id": str(item.get("variant_id") or ""),
                    "issue": "non_fault_variant",
                    "value": str(item.get("label") or ""),
                }
                for item in non_fault_variants
            )
        active_variant_ids = {
            str(item.get("variant_id") or "")
            for item in [*actions, *outcomes, *required_info, *traces]
            if str(item.get("variant_id") or "")
        }
        for variant in variants:
            variant_id = str(variant.get("variant_id") or "")
            if variant_id not in active_variant_ids:
                issues.append("kg_v2_orphan_variant")
                item_issues.append({"object_id": variant_id, "issue": "orphan_variant", "value": str(variant.get("label") or "")})
        if not actions:
            issues.append("kg_v2_missing_actions")
        if _kg_v2_non_atomic_actions(actions):
            issues.append("kg_v2_non_atomic_actions")
            item_issues.extend(
                {"object_id": str(item.get("action_id") or ""), "issue": "non_atomic_action", "value": str(item.get("label") or "")}
                for item in actions
                if len(str(item.get("label") or "")) > 40 or any(marker in str(item.get("label") or "") for marker in V2_ACTION_NON_ATOMIC_MARKERS)
            )
        if _kg_v2_noisy_action_labels(actions):
            issues.append("kg_v2_noisy_action_labels")
        duplicate_actions = _kg_v2_duplicate_actions(actions)
        if duplicate_actions:
            issues.append("kg_v2_duplicate_actions")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "duplicate_action",
                    "value": str(item.get("label") or ""),
                }
                for item in duplicate_actions
            )
        near_duplicate_actions = _kg_v2_near_duplicate_actions(actions)
        if near_duplicate_actions:
            issues.append("kg_v2_near_duplicate_actions")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "near_duplicate_action",
                    "value": str(item.get("label") or ""),
                }
                for item in near_duplicate_actions
            )
        malformed_action_labels = _kg_v2_malformed_action_labels(actions)
        if malformed_action_labels:
            issues.append("kg_v2_malformed_action_labels")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "malformed_action_label",
                    "value": str(item.get("label") or ""),
                }
                for item in malformed_action_labels
            )
        weak_action_labels = [
            item for item in actions
            if not _kg_v2_executable_action_label(str(item.get("label") or ""))
        ]
        if weak_action_labels:
            issues.append("kg_v2_non_action_labels")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "non_action_label",
                    "value": str(item.get("label") or ""),
                }
                for item in weak_action_labels
            )
        historical_statement_actions = [
            item for item in actions
            if any(marker in str(item.get("label") or "") for marker in V2_ACTION_HISTORY_MARKERS)
            and not str(item.get("label") or "").strip().startswith(V2_ACTION_VERBS)
        ]
        if historical_statement_actions:
            issues.append("kg_v2_historical_statement_actions")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "historical_statement_action",
                    "value": str(item.get("label") or ""),
                }
                for item in historical_statement_actions
            )
        action_ids = {str(item.get("action_id") or "") for item in actions if str(item.get("action_id") or "")}
        outcome_action_ids = {str(item.get("action_id") or "") for item in outcomes if str(item.get("action_id") or "")}
        trace_action_ids = {
            str(action_id)
            for trace in traces
            for action_id in [*(trace.get("recommended_action_ids") or []), *(trace.get("actual_action_ids") or [])]
            if str(action_id)
        }
        document_source_kinds = {"raw_doc", "sop", "hybrid"}
        case_actions = [
            item for item in actions
            if str(item.get("source_kind") or "") == "case"
            or str(item.get("action_id") or "") in trace_action_ids
            or (
                bool(source_cases or traces)
                and str(item.get("source_kind") or "") not in document_source_kinds
            )
        ]
        document_actions = [
            item for item in actions
            if str(item.get("source_kind") or "") in document_source_kinds
            and str(item.get("action_id") or "") not in trace_action_ids
        ]
        actions_without_evidence = [item for item in case_actions if not item.get("evidence_ids")]
        if actions_without_evidence:
            issues.append("kg_v2_action_missing_evidence")
            item_issues.extend({
                "object_id": str(item.get("action_id") or ""),
                "issue": "action_missing_evidence",
                "value": str(item.get("label") or ""),
            } for item in actions_without_evidence)
        actions_without_outcome = [
            item for item in case_actions
            if str(item.get("action_id") or "") not in outcome_action_ids
        ]
        if actions_without_outcome:
            issues.append("kg_v2_action_missing_outcome")
            item_issues.extend({
                "object_id": str(item.get("action_id") or ""),
                "issue": "action_missing_outcome",
                "value": str(item.get("label") or ""),
            } for item in actions_without_outcome)
        procedure_step_ids = {
            str(item.get("procedure_step_id") or "")
            for item in objects.get("ProcedureStep") or []
            if isinstance(item, dict) and str(item.get("procedure_step_id") or "")
        }
        documented_action_ids = {
            str(relation.get("to") or "")
            for relation in relations
            if str(relation.get("relation") or "") == "candidate_action"
            and str(relation.get("from") or "") in procedure_step_ids
        }
        document_actions_without_basis = [
            item for item in document_actions
            if not item.get("evidence_ids")
            and str(item.get("action_id") or "") not in documented_action_ids
        ]
        if document_actions_without_basis:
            issues.append("kg_v2_document_action_missing_basis")
            item_issues.extend({
                "object_id": str(item.get("action_id") or ""),
                "issue": "document_action_missing_basis",
                "value": str(item.get("label") or ""),
            } for item in document_actions_without_basis)
        invalid_actual_actions = []
        action_status = {str(item.get("action_id") or ""): str(item.get("execution_status") or "") for item in actions}
        for trace in traces:
            invalid_actual_actions.extend(
                action_id for action_id in trace.get("actual_action_ids") or []
                if action_id not in action_ids or action_status.get(str(action_id)) != "actual"
            )
        if invalid_actual_actions:
            issues.append("kg_v2_actual_trace_contains_unexecuted_action")
            item_issues.extend({
                "object_id": str(action_id),
                "issue": "actual_trace_contains_unexecuted_action",
                "value": action_status.get(str(action_id), "missing"),
            } for action_id in invalid_actual_actions)
        weak_verified_fixes = []
        for outcome in outcomes:
            if str(outcome.get("outcome_type") or "") != "verified_fix":
                continue
            summary = str(outcome.get("summary") or "").strip()
            lowered = summary.lower()
            if (
                not outcome.get("evidence_ids")
                or lowered in {
                    "camera_capture_chain", "software_version_change", "startup/init", "startup/init phase",
                    "root_cause", "verified", "fixed", "resolved",
                }
                or any(marker in summary for marker in (
                    "未复现", "无法复现", "未发现异常", "仍需验证", "待验证", "应该能解决", "可能解决",
                    "非根因", "不是根因", "非直接原因",
                ))
                or summary in {"就重启了", "又重启了", "仍然重启", "再次出现", "问题复发"}
                or not any(marker in summary for marker in (
                    "持续", "至今", "长期", "未再", "不再", "没有再", "无复发", "正常生产",
                    "恢复生产", "验证通过", "稳定运行", "反复验证", "最终确认", "最终解决",
                ))
            ):
                weak_verified_fixes.append(outcome)
        if weak_verified_fixes:
            issues.append("kg_v2_unsubstantiated_verified_fix")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "unsubstantiated_verified_fix",
                    "value": str(item.get("summary") or ""),
                }
                for item in weak_verified_fixes
            )
        duplicate_outcomes = _kg_v2_duplicate_outcomes(outcomes)
        if duplicate_outcomes:
            issues.append("kg_v2_duplicate_outcomes")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "duplicate_outcome",
                    "value": str(item.get("summary") or ""),
                }
                for item in duplicate_outcomes
            )
        outcome_type_conflicts = _kg_v2_outcome_type_conflicts(outcomes, source_text)
        if outcome_type_conflicts:
            issues.append("kg_v2_outcome_type_conflict")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "outcome_type_conflict",
                    "value": str(item.get("summary") or ""),
                }
                for item in outcome_type_conflicts
            )
        synthetic_outcomes = _kg_v2_synthetic_outcomes(outcomes)
        if synthetic_outcomes:
            issues.append("kg_v2_synthetic_outcome")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "synthetic_outcome",
                    "value": str(item.get("summary") or ""),
                }
                for item in synthetic_outcomes
            )
        synthetic_observation_claims = [
            item for item in outcomes
            if str(item.get("outcome_origin") or "") == "synthetic_fallback"
            and str(item.get("outcome_type") or "") != "pending_validation"
        ]
        if synthetic_observation_claims:
            issues.append("kg_v2_synthetic_outcome_claims_observation")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "synthetic_outcome_claims_observation",
                    "value": str(item.get("outcome_type") or ""),
                }
                for item in synthetic_observation_claims
            )
        focused_case_text = _kg_v2_focused_case_text(variants, traces, source_cases)
        if case_actions and _kg_v2_missing_observed_resolution(outcomes, focused_case_text):
            issues.append("kg_v2_missing_observed_resolution")
            item_issues.append({
                "object_id": str(bundle.get("candidate_id") or bundle.get("bundle_id") or ""),
                "issue": "missing_observed_resolution",
                "value": trim_text(focused_case_text, 180),
            })
        result_statement_actions = _kg_v2_result_statement_actions(actions)
        if result_statement_actions:
            issues.append("kg_v2_result_statement_actions")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "result_statement_action",
                    "value": str(item.get("label") or ""),
                }
                for item in result_statement_actions
            )
        multi_operation_actions = _kg_v2_multi_operation_actions(actions)
        if multi_operation_actions:
            issues.append("kg_v2_multi_operation_actions")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "multi_operation_action",
                    "value": str(item.get("label") or ""),
                }
                for item in multi_operation_actions
            )
        ungrounded_actions = _kg_v2_ungrounded_actions(actions, source_text)
        if ungrounded_actions:
            issues.append("kg_v2_ungrounded_action_chain")
            item_issues.extend(
                {
                    "object_id": str(item.get("action_id") or ""),
                    "issue": "ungrounded_action",
                    "value": str(item.get("label") or ""),
                }
                for item in ungrounded_actions
            )
        outside_source_outcomes = _kg_v2_outcome_evidence_outside_source(
            outcomes, evidence_items, source_message_ids
        )
        if outside_source_outcomes:
            issues.append("kg_v2_outcome_evidence_outside_source_episode")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "outcome_evidence_outside_source_episode",
                    "value": str(item.get("summary") or ""),
                }
                for item in outside_source_outcomes
            )
        unsupported_text_outcomes = _kg_v2_outcome_evidence_without_text_support(
            outcomes, evidence_items, source_messages
        )
        if unsupported_text_outcomes:
            issues.append("kg_v2_outcome_evidence_without_text_support")
            item_issues.extend(
                {
                    "object_id": str(item.get("outcome_id") or ""),
                    "issue": "outcome_evidence_without_text_support",
                    "value": str(item.get("summary") or ""),
                }
                for item in unsupported_text_outcomes
            )
        if any(bool(item.get("destructive")) for item in actions):
            issues.append("kg_v2_destructive_action_requires_human")
        if any(bool(item.get("high_cost")) for item in actions):
            issues.append("kg_v2_high_cost_action_requires_human")
        for required in required_info:
            required_id = str(required.get("required_info_id") or "")
            slot = str(required.get("slot") or "other")
            question = str(required.get("question") or "").strip()
            why = str(required.get("why_required") or "").strip()
            if slot == "other":
                issues.append("kg_v2_required_info_other_slot")
                item_issues.append({"object_id": required_id, "issue": "required_info_other_slot", "value": question})
            if not question or not why or not required.get("blocks"):
                issues.append("kg_v2_incomplete_required_info")
                item_issues.append({"object_id": required_id, "issue": "incomplete_required_info", "value": question})
            if question in ASK_INFO_GENERIC_MARKERS or question.rstrip("。？?") in ASK_INFO_GENERIC_MARKERS:
                issues.append("kg_v2_generic_required_info")
                item_issues.append({"object_id": required_id, "issue": "generic_required_info", "value": question})
        if not (source_cases or knowledge_documents) or not evidence_items or not evidence_relations:
            issues.append("kg_v2_missing_evidence_pack")
        if any(item.get("relation") == "resolved_by" for item in relations):
            issues.append("kg_v2_resolved_by_relation_forbidden")
        if output_mode and output_mode not in V2_FAULT_OUTPUT_MODES:
            issues.append("kg_v2_non_fault_output_mode")
            item_issues.append({"object_id": str(bundle.get("candidate_id") or bundle.get("bundle_id") or ""), "issue": "non_fault_output_mode", "value": output_mode})
        if "ambiguous_family_scope" in refinement_flags:
            issues.append("kg_v2_ambiguous_family_scope")
            item_issues.append({
                "object_id": str(bundle.get("candidate_id") or bundle.get("bundle_id") or ""),
                "issue": "ambiguous_family_scope",
                "value": ", ".join(str(item) for item in refinement.get("family_scope_candidates") or []),
            })
        if not schema_valid:
            issues.append("kg_v2_schema_invalid")
            issues.extend(f"kg_v2:{issue}" for issue in schema_issues)
        hard_fail = any(issue in {
            "kg_v2_missing_family",
            "kg_v2_missing_variant",
            "kg_v2_noncanonical_family",
            "kg_v2_pseudo_family",
            "kg_v2_family_variant_label_collision",
            "kg_v2_family_variant_semantic_mismatch",
            "kg_v2_more_specific_family_available",
            "kg_v2_non_fault_variant",
            "kg_v2_long_variant_label",
            "kg_v2_conversational_variant_label",
            "kg_v2_weak_variant_label",
            "kg_v2_questionish_variant_label",
            "kg_v2_non_atomic_actions",
            "kg_v2_missing_actions",
            "kg_v2_noisy_action_labels",
            "kg_v2_duplicate_actions",
            "kg_v2_near_duplicate_actions",
            "kg_v2_malformed_action_labels",
            "kg_v2_non_action_labels",
            "kg_v2_historical_statement_actions",
            "kg_v2_action_missing_evidence",
            "kg_v2_action_missing_outcome",
            "kg_v2_document_action_missing_basis",
            "kg_v2_actual_trace_contains_unexecuted_action",
            "kg_v2_unsubstantiated_verified_fix",
            "kg_v2_duplicate_outcomes",
            "kg_v2_outcome_type_conflict",
            "kg_v2_synthetic_outcome",
            "kg_v2_synthetic_outcome_claims_observation",
            "kg_v2_missing_observed_resolution",
            "kg_v2_result_statement_actions",
            "kg_v2_multi_operation_actions",
            "kg_v2_ungrounded_action_chain",
            "kg_v2_outcome_evidence_outside_source_episode",
            "kg_v2_outcome_evidence_without_text_support",
            "kg_v2_required_info_other_slot",
            "kg_v2_incomplete_required_info",
            "kg_v2_generic_required_info",
            "kg_v2_missing_evidence_pack",
            "kg_v2_resolved_by_relation_forbidden",
            "kg_v2_non_fault_output_mode",
            "kg_v2_ambiguous_family_scope",
            "kg_v2_orphan_variant",
            "kg_v2_schema_invalid",
        } for issue in issues)
        return {
            "confidence": round(confidence, 4),
            "clarity": round(clarity, 4),
            "relevance": round(relevance, 4),
            "schema_validity": round(schema_validity, 4),
            "weighted_sum": weighted,
            "threshold": self.threshold,
            "passed": weighted >= self.threshold and not hard_fail,
            "issues": sorted(set(issues)),
            "observability": {
                "agent_id": "W4",
                "candidate_id": bundle.get("candidate_id") or bundle.get("bundle_id") or "",
                "kg_version": "v2",
                "family_labels": family_labels,
                "variant_labels": variant_labels,
                "source_output_mode": output_mode,
                "requires_human": True,
                "item_issues": item_issues,
            },
        }

    def score_typed_candidate(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Score a non-SOP typed KG v2 admission envelope without forcing fault shape."""

        objects = _typed_objects(envelope)
        admission_target = _infer_typed_admission_target(envelope)
        source_kinds = _typed_source_kinds(envelope)
        sop_incremental_allowed = _typed_sop_incremental_allowed(envelope)
        schema_valid = bool(_typed_first(envelope, "schema_valid") if _typed_first(envelope, "schema_valid") != "" else True)
        schema_issues = [str(x) for x in _typed_list(envelope, "schema_issues")]
        required_evidence = list(TYPED_REQUIRED_EVIDENCE.get(admission_target, ("raw_text", "evidence")))
        if admission_target == "fault_execution" and objects.get("ActionOutcome"):
            required_evidence.append("outcome_evidence")
        missing_evidence = [name for name in required_evidence if not _typed_has_evidence(envelope, name)]
        admission_readiness = _typed_admission_readiness(
            envelope,
            schema_valid=schema_valid,
            missing_evidence=missing_evidence,
        )
        policy_readiness, policy_evidence_counts = _typed_execution_policy_readiness(envelope)
        text = _typed_text(envelope)
        evidence_disposition = str(_typed_first(envelope, "evidence_disposition") or "")
        context_evidence_policy = str(_typed_first(envelope, "context_evidence_policy") or "")
        provenance_issues = alignment_provenance_issues(envelope)
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        strategy = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}
        strategy_id = str(strategy.get("strategy_id") or "")
        procedure_steps = objects.get("ProcedureStep") or []
        knowledge_sections = objects.get("KnowledgeSection") or []
        has_fault_mapping = bool(objects.get("FaultFamily") or objects.get("FaultVariant"))

        issues: list[str] = []
        if admission_target not in TYPED_ADMISSION_TARGETS:
            issues.append("typed_unknown_admission_target")
        if "sop" in source_kinds and not sop_incremental_allowed:
            issues.append("typed_sop_source_rejected")
        if not schema_valid:
            issues.append("typed_schema_invalid")
            issues.extend(f"typed:{issue}" for issue in schema_issues)
        issues.extend(f"typed_missing_evidence:{name}" for name in missing_evidence)
        issues.extend(f"typed_{issue}" for issue in provenance_issues)
        if not text.strip():
            issues.append("typed_missing_raw_text")
        if evidence_disposition == "reject_review_only":
            issues.append("typed_evidence_requires_review")
        low_fault_relevance = (
            admission_target == "evidence_only"
            and "jira" in source_kinds
            and not _typed_fault_relevant(text)
        )
        if low_fault_relevance:
            issues.append("typed_low_fault_relevance")
        untrusted_chat_context = (
            "chat" in source_kinds
            and admission_target == "fault_execution"
            and context_evidence_policy not in {
                "current_episode_only.v1",
                "w7_promoted_case_evidence.v1",
            }
        )
        if untrusted_chat_context:
            issues.append("typed_untrusted_context_evidence_policy")
        structural_review = False
        if admission_target == "fault_execution" and not objects.get("DiagnosticAction"):
            issues.append("typed_fault_execution_missing_actions")
            structural_review = True
        execution_policy_review = (
            admission_target == "fault_execution"
            and admission_readiness == "execution_ready"
            and policy_readiness != "observed_execution"
        )
        if execution_policy_review:
            if policy_readiness == "pending_only":
                issues.append("typed_execution_policy_pending_only")
            elif policy_readiness == "contains_promoted_only_action":
                issues.append("typed_promoted_only_action_evidence")
            else:
                issues.append("typed_execution_policy_without_observed_actual_outcome")
            if (
                policy_readiness == "pending_only"
                and policy_evidence_counts.get("synthetic_pending_count", 0)
                == policy_evidence_counts.get("outcome_count", 0)
            ):
                issues.append("typed_synthetic_pending_outcome_only")
        if admission_target in {"procedure_library", "playbook", "policy_template"} and not procedure_steps:
            issues.append("typed_document_missing_procedure_steps")
            structural_review = True
        if (
            admission_target == "fault_support"
            and strategy_id == "troubleshooting_topic_doc"
            and not has_fault_mapping
            and len(knowledge_sections) <= 1
        ):
            issues.append("typed_fault_support_missing_structured_mapping")
            structural_review = True

        object_count = sum(len(items) for items in objects.values())
        evidence_strength = 1.0 - (len(missing_evidence) / max(1, len(required_evidence)))
        object_fit = 0.35
        if admission_target in TYPED_ADMISSION_TARGETS:
            object_fit += 0.25
        if object_count:
            object_fit += 0.25
        if admission_target == "evidence_only" and objects.get("EvidenceItem"):
            object_fit += 0.15
        object_fit = min(1.0, object_fit)
        source_fit = 0.25
        if source_kinds:
            source_fit += 0.25
        if source_kinds and (
            "sop" not in source_kinds or sop_incremental_allowed
        ):
            source_fit += 0.25
        if text.strip():
            source_fit += 0.25
        source_fit = min(1.0, source_fit)
        schema_fit = 1.0 if schema_valid else 0.0
        weighted = round(evidence_strength * 0.35 + object_fit * 0.25 + source_fit * 0.25 + schema_fit * 0.15, 4)

        hard_reject = any(issue in set(issues) for issue in {
            "typed_unknown_admission_target",
            "typed_sop_source_rejected",
            "typed_schema_invalid",
            "typed_missing_raw_text",
        })
        hard_reject = hard_reject or bool(provenance_issues)
        if hard_reject:
            decision = "reject"
        elif missing_evidence or low_fault_relevance or untrusted_chat_context or structural_review or execution_policy_review or evidence_disposition in {"evidence_only", "reject_review_only"}:
            decision = "route_review"
        else:
            decision = "admit"
        merge_allowed = decision != "reject" and admission_readiness != "not_ready"
        materialize_allowed = (
            decision == "admit"
            and admission_target == "fault_execution"
            and admission_readiness == "execution_ready"
            and policy_readiness == "observed_execution"
        )
        return {
            "decision": decision,
            "admission_target": admission_target if admission_target in TYPED_ADMISSION_TARGETS else "evidence_only",
            "materialize_allowed": materialize_allowed,
            "merge_allowed": merge_allowed,
            "admission_readiness": admission_readiness,
            "policy_readiness": policy_readiness,
            "required_evidence": required_evidence,
            "issues": sorted(set(issues)),
            "decision_version": TYPED_DECISION_VERSION,
            "mapping_version": TYPED_MAPPING_VERSION,
            "weighted_sum": weighted,
            "threshold": 0.62,
            "passed": decision != "reject",
            "scores": {
                "evidence_strength": round(evidence_strength, 4),
                "object_fit": round(object_fit, 4),
                "source_fit": round(source_fit, 4),
                "schema_fit": round(schema_fit, 4),
            },
            "observability": {
                "agent_id": "W4",
                "candidate_id": _typed_first(envelope, "candidate_id", "intake_id", "bundle_id") or "",
                "source_kinds": sorted(source_kinds),
                "object_types": sorted(object_type for object_type, items in objects.items() if items),
                "evidence_disposition": evidence_disposition,
                "admission_readiness": admission_readiness,
                "policy_readiness": policy_readiness,
                "policy_evidence_counts": policy_evidence_counts,
            },
        }

    def score_required_info(self, required_info_candidate: dict[str, Any]) -> dict[str, Any]:
        slot = str(required_info_candidate.get("slot") or "other")
        question = str(required_info_candidate.get("question") or "")
        why = str(required_info_candidate.get("why_required") or "")
        condition = str(required_info_candidate.get("condition") or "")
        target_error_id = str(required_info_candidate.get("target_error_id") or "")
        merge_policy = str(required_info_candidate.get("merge_policy") or "")
        evidence_ids = [str(x) for x in required_info_candidate.get("evidence_message_ids") or [] if str(x)]
        provided_tool_roles = [str(x) for x in required_info_candidate.get("provided_tool_roles") or [] if str(x)]
        has_slot_match_field = "provided_slot_match_roles" in required_info_candidate
        provided_slot_match_roles = [str(x) for x in required_info_candidate.get("provided_slot_match_roles") or [] if str(x)]
        request = required_info_candidate.get("source_request") if isinstance(required_info_candidate.get("source_request"), dict) else {}
        text = " ".join(str(x or "") for x in (
            required_info_candidate.get("label"),
            question,
            why,
            condition,
            request.get("text"),
        ))
        generic = any(k in text for k in ASK_INFO_GENERIC_MARKERS) and not condition and len(text) < 60

        slot_specificity = 0.15 if slot == "other" else 0.65
        if condition:
            slot_specificity += 0.2
        if any(k in text for k in ("DLOG", "诊断数据", "主程序版本", "算法包版本", "报错码", "IP", "初始化", "dmp")):
            slot_specificity += 0.1
        if generic:
            slot_specificity -= 0.3
        slot_specificity = max(0.0, min(1.0, slot_specificity))

        diagnostic_relevance = 0.25
        if why:
            diagnostic_relevance += 0.25
        if any(k in text for k in ("缩小", "诊断", "判断", "定位", "确认", "分支", "阶段", "相机", "控制器", "蓝屏", "初始化")):
            diagnostic_relevance += 0.3
        if target_error_id:
            diagnostic_relevance += 0.1
        if generic:
            diagnostic_relevance -= 0.25
        diagnostic_relevance = max(0.0, min(1.0, diagnostic_relevance))

        evidence_strength = min(1.0, 0.25 + 0.2 * len(evidence_ids)) if evidence_ids else 0.0
        if provided_slot_match_roles:
            evidence_strength = max(evidence_strength, min(1.0, 0.55 + 0.1 * len(provided_slot_match_roles)))
        elif provided_tool_roles and has_slot_match_field:
            evidence_strength = min(evidence_strength, 0.45)
        elif provided_tool_roles:
            evidence_strength = max(evidence_strength, min(1.0, 0.55 + 0.1 * len(provided_tool_roles)))
        schema_fit = 1.0 if slot != "other" else 0.35
        if not question:
            schema_fit -= 0.3
        if not (target_error_id or merge_policy == "review_only"):
            schema_fit -= 0.4
        schema_fit = max(0.0, min(1.0, schema_fit))

        weighted = round(slot_specificity * 0.25 + diagnostic_relevance * 0.35 + evidence_strength * 0.2 + schema_fit * 0.2, 4)
        issues: list[str] = []
        if slot == "other":
            issues.append("slot_other")
        if not evidence_ids:
            issues.append("missing_evidence")
        if not question:
            issues.append("missing_question")
        if not why:
            issues.append("missing_why_required")
        if not (target_error_id or merge_policy == "review_only"):
            issues.append("missing_target_or_review_policy")
        if generic:
            issues.append("generic_log_request")
        if provided_tool_roles and has_slot_match_field and not provided_slot_match_roles:
            issues.append("provided_tool_roles_mismatch")
        if merge_policy == "review_only" and not target_error_id:
            issues.append("review_only_no_target_error")
        hard_fail = any(x in set(issues) for x in ("slot_other", "missing_evidence", "missing_question", "missing_why_required", "missing_target_or_review_policy", "generic_log_request"))
        passed = weighted >= 0.62 and not hard_fail
        return {
            "slot_specificity": round(slot_specificity, 4),
            "diagnostic_relevance": round(diagnostic_relevance, 4),
            "evidence_strength": round(evidence_strength, 4),
            "schema_fit": round(schema_fit, 4),
            "weighted_sum": weighted,
            "threshold": 0.62,
            "passed": passed,
            "issues": sorted(set(issues)),
            "observability": {
                "agent_id": "W4",
                "candidate_id": required_info_candidate.get("candidate_id") or "",
                "slot": slot,
                "provided_tool_roles": provided_tool_roles,
                "provided_slot_match_roles": provided_slot_match_roles,
            },
        }
