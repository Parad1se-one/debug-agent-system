from __future__ import annotations

import json
import re
import hashlib
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.compat import _canonicalize_family_label
from debug_agent_system.knowledge_v2.builders import infer_required_info_slot
from debug_agent_system.agents.tools.jira_parser import JiraParserAgent
from debug_agent_system.agents.write.w1_message_relations import attachment_identity_keys
from debug_agent_system.agents.write.people_roles import DEFAULT_ROLE_REGISTRY, load_people_role_registry, people_index

_WORD = re.compile(r"[A-Za-z0-9_.:-]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_NUMBERED_BULLET = re.compile(r"\b[0-9]+[、.．]")
_ADDITIONAL10_NOISE_TOKENS = (
    "现场工作汇报",
    "今日工作情况",
    "今日反馈表格",
    "日常数据回传",
    "每日反馈表格",
    "夜班数据返回",
    "现场问题",
    "问题点",
    "工作汇报",
    "各领导请查阅",
    "集中培训",
    "上手实操",
    "表示认可",
)
_FAULT_HINTS = (
    "报错", "异常", "失败", "蓝屏", "黑屏", "重启", "卡死", "闪退", "不拍照", "拍摄失败",
    "空图", "图片为空", "错位", "误报", "漏检", "不稳定", "延迟", "卡顿", "初始化失败",
    "无法开机", "打不开", "驱动", "dmp", "DMP", "图像", "ocr", "OCR",
)
_HELPY_HINTS = (
    "看看能不能", "看有必要", "帮得上忙", "帮忙看", "帮忙分析", "帮忙确认",
    "麻烦", "辛苦", "能不能试试", "远程看了一下", "排查建议", "什么排查的动作", "帮分析看看怎么解决",
)
_NON_CONCRETE_ACTION_MARKERS = (
    "没法排查",
    "无法排查",
    "顺便确认",
    "帮忙确认",
    "确认下吗",
    "确认一下吗",
    "还请",
)
_NON_FAULT_PRIMARY_HINTS = (
    "建议可以", "批量回传", "开启存ok件", "上传导出", "交叉验证", "不如直接相信模型了",
    "今日工作汇总如下", "工作汇总如下", "现场工作汇总如下",
)
_FAULT_STATUS_UPDATE_MARKERS = (
    "未再出现", "暂无", "暂未", "恢复正常", "正常开关机", "正常测试", "持续观察", "观察中", "已撤离现场",
    "没反馈", "后续观察", "到现在", "至今", "已更换", "更换完成后", "处理后应该", "正常用",
)
_ACTION_HINTS = (
    "检查", "确认", "分析", "收集", "导出", "提供", "升级", "回退", "重装", "更换",
    "排查", "观察", "验证", "启用", "卸载", "重启", "截图", "抓取", "记录", "修复",
)
_ALLOW_SINGLE_ACTION_FAULT_MARKERS = (
    "图片为空", "空图", "拍摄失败", "拍照失败", "保存结果失败", "扫码失败", "二维码识别失败",
    "蓝屏", "黑屏", "自动重启", "无法开机",
)
_MEDIA_RE = re.compile(r"\[(?:Media|Image|File):[^\]]+\]", re.IGNORECASE)
_MENTION_PREFIX_RE = re.compile(r"^(?:@\S+\s*)+")
_CLAUSE_SPLIT_RE = re.compile(r"[。！？!?\n；;，,]+")
_COORDINATION_MARKERS = (
    "发货", "包装", "物流", "审批", "付款", "交付", "demo", "培训", "签到", "回执", "进厂", "离场",
    "排期", "会议", "安排", "表格", "日报", "工作汇报", "今日工作", "计划", "上线", "验收",
    "发版后", "升级一下版本", "升级到什么版本", "可以升级", "确认发货",
)
_OWNER_ACTION_MARKERS = ("看下", "处理", "排查", "确认", "支持", "分析", "定位", "跟进", "负责")
_TAKEOVER_MARKERS = ("我看下", "我来处理", "我来跟进", "我排查", "我确认下", "我负责", "我这边处理", "我来分析")
_TRANSFER_MARKERS = ("转发给", "转给", "对应负责人", "责任人", "负责人")

DEFAULT_GOLD_ROOT = "data/annotations/goldcases/gold-v1"
DEFAULT_MANUAL_ROOT = "data/kg/review_queue/manual_review_examples"
DEFAULT_EPISODES_JSON = "data/results/w1_full_jira_enriched_20260703_074700/w1/episodes.json"
DEFAULT_SOP_SEED_JSON = "data/results/kg_v2_sop_seed_draft_manual.json"
DEFAULT_TEXT_HISTORY_ROOT = "data/imports/full-2015-to-2026-07-09-v2/messages_by_chat"
DEFAULT_JIRA_OFFLINE_ROOT = "data/imports/jira_offline/raw"

_CHAT_ID_RE = re.compile(r"(oc_[a-f0-9]{32})", re.IGNORECASE)
_JIRA_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}-\d+\b")
_CASE_EVIDENCE_MARKERS = (
    "检查", "排查", "分析", "日志", "定位", "原因", "根因", "恢复", "解决", "修复", "验证",
    "重启", "升级", "回退", "无效", "仍然", "复现", "bug", "BUG", "jira", "JIRA", "配置",
    "资源", "版本", "超时", "404", "驱动", "触发", "识别", "扫码", "拍摄", "卡顿", "闪退",
)
_CASE_RESULT_MARKERS = (
    "分析了下", "判断为", "确认是", "发现是", "原因是", "已解决", "恢复正常", "没有占满",
    "没什么资源", "没有什么资源", "资源占满", "软件有BUG", "软件有 bug", "配置文件问题", "无需修复", "问题重复",
)
_CASE_STRONG_RESULT_MARKERS = (
    "已解决", "恢复正常", "没什么资源", "没有什么资源", "软件有BUG", "软件有 bug", "配置文件问题", "无需修复",
)
_CASE_CONTEXT_NOISE = (
    "invited", "updated the group name", "各位领导晚上好", "今日工作汇报", "培训客户", "发货",
    "各位领导，晚上好", "今天现场和客户沟通了", "现场工作：", "现场工作内容：", "问题汇总：",
    "会议", "排期", "合同", "付款", "撤场", "签到",
)
_CASE_FAULT_SIGNATURE_MARKERS = (
    "扫码", "条码", "二维码", "进板", "出板", "卡板", "拍摄", "拍照", "fov", "相机", "蓝屏", "黑屏",
    "重启", "卡顿", "闪退", "崩溃", "误报", "漏检", "复判", "结果等待", "暂停退出", "初始化",
    "配置文件", "user.cfg", "内存", "显卡", "驱动", "光源", "mes", "spc", "cad", "mark", "ct", "ocr", "识别", "提示框", "连锡",
    "网卡", "排线", "请求超时", "拍摄失败",
)
_GENERIC_CASE_SIGNATURES = {
    "重启", "初始化", "驱动", "配置文件", "卡顿", "异常", "复判",
}

_RESOLUTION_QUESTION_MARKERS = (
    "解决了吗", "解决了么", "有没有解决", "是否解决", "问题解决没", "现在解决", "能解决吗",
)
_RESOLUTION_PENDING_MARKERS = (
    "待解决", "等待解决", "需要再讨论", "如果解决", "还在排查", "继续排查", "后续验证",
    "待验证", "观察中", "持续观察", "暂时", "临时", "应该可以", "是否可行", "需要确认",
    "强制重启后正常", "重启后正常", "请现场确认", "请现场验证",
)
_RESOLUTION_POSITIVE_MARKERS = (
    "已解决", "解决了", "恢复正常", "已恢复", "处理完成", "验证通过", "没问题了", "未再出现",
    "不再出现", "未复现", "没有复现", "运行正常", "测试正常", "拍摄正常", "可以正常使用",
    "正常拍照", "无异常情况出现", "没有异常情况", "没有报拍摄失败", "未反馈拍摄失败",
)
_RESOLUTION_ACTION_MARKERS = (
    "更换", "修复", "升级", "回退", "重装", "卸载", "清理", "调整", "设置", "拔插", "替换",
    "恢复", "关闭", "启用", "更新", "重启", "处理", "下降", "固定", "打胶", "安装", "拔掉", "拔出",
)
_INEFFECTIVE_MARKERS = (
    "无效", "没有效果", "仍然出现", "依旧出现", "还是出现", "仍然存在", "依旧存在", "还是存在",
    "仍会", "还是会", "未解决", "没有解决", "验证失败",
)
_REPORT_ONLY_MARKERS = (
    "今日现状", "今日工作", "现场工作汇报", "每日反馈", "工作汇总", "现场问题汇总", "问题汇总",
)
_W7_PRIMARY_FAULT_SIGNATURES = {
    "蓝屏", "黑屏", "重启", "卡顿", "闪退", "漏检", "误报", "拍摄", "拍照", "不拍照",
    "崩溃", "残帧", "丢包", "初始化", "报错", "失败", "空图", "错位", "掉板", "进板", "出板",
    "断层", "ocr", "识别", "提示框", "连锡",
}
_W7_TRACE_DISTINCTIVE_SIGNATURES = {
    "相机", "网卡", "排线", "请求超时", "拍摄失败", "ocr", "提示框", "连锡", "漏检", "误报",
    "蓝屏", "黑屏", "内存", "显卡", "驱动", "配置文件", "二维码", "扫码", "初始化",
}
_W7_VALIDATION_MARKERS = (
    "验证通过", "复测", "重新测试", "原条件", "连续", "运行", "生产", "未再出现", "不再出现",
    "未复现", "没有复现", "无异常", "正常拍照", "正常测试", "正常使用", "可以正常使用",
)
_W7_RECURRENCE_MARKERS = (
    "复发", "再次出现", "又出现", "再次报", "仍然出现", "依旧出现", "还是出现", "还是一样",
)
_W7_NON_EXECUTION_MARKERS = (
    "待后续", "后续将", "后续进行", "计划", "建议", "可以升级", "用于验证", "用于测试",
    "如果没有问题", "考虑", "请现场确认", "请现场验证", "能否", "是否",
)
_W7_EQUIPMENT_RE = re.compile(r"\b(?:AOI[-_ ]?\d{1,6}|SI\d{3,6}[A-Z]?|\d{3,5}T|[A-Z]{2,8}[-_]\d{2,8})\b", re.IGNORECASE)
_W7_ARTIFACT_RE = re.compile(r"\b[A-Za-z0-9_-]{4,}\.(?:proj|dlog|dmp|log)\b", re.IGNORECASE)
_W7_LINE_RE = re.compile(r"(?:第)?([一二三四五六七八九十]|\d{1,2})线")
_W7_JIRA_RE = re.compile(r"\b(?:SMTAOITS|TEST)-\d+\b", re.IGNORECASE)


def norm_text(text: Any) -> str:
    value = str(text or "").lower()
    return " ".join(_WORD.findall(value) + _CJK.findall(value))


def episode_text(episode: dict[str, Any]) -> str:
    ext = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    parts: list[str] = []
    for key in ("symptom_raw", "conclusion", "key_conclusion"):
        parts.append(str(ext.get(key) or ""))
    parts.extend(str(x) for x in ext.get("debug_actions") or [])
    seen: set[str] = set()
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_evidence_messages", "case_context_messages"):
        for msg in episode.get(key) or []:
            if isinstance(msg, dict):
                text = str(msg.get("text") or msg.get("content_summary") or "").strip()
                key_id = str(msg.get("message_id") or msg.get("source_message_id") or text)
                if not text or key_id in seen:
                    continue
                seen.add(key_id)
                parts.append(text)
    return " ".join(parts)


def _clean_fault_candidate(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("**", " ")
    value = _MEDIA_RE.sub("", value)
    value = _MENTION_PREFIX_RE.sub("", value).strip()
    value = re.sub(r"\[[^\]]+\]\([^)]*\)", "", value)
    for marker in ("问题点", "还有一个", "现场测试时"):
        idx = value.rfind(marker)
        if idx > 0:
            tail = value[idx:].strip()
            if any(token in tail for token in _FAULT_HINTS):
                value = tail
                break
    value = " ".join(value.split())
    return value.strip(" ，,。；;：:")


def _fault_focus_score(text: str) -> int:
    clean = _clean_fault_candidate(text)
    if not clean:
        return -100
    if any(token in clean for token in _ADDITIONAL10_NOISE_TOKENS + _NON_FAULT_PRIMARY_HINTS):
        return -100
    score = 0
    for token in _FAULT_HINTS:
        if token in clean:
            score += 4
    if len(clean) <= 120:
        score += 4
    if any(token in clean for token in _HELPY_HINTS):
        score -= 4
    if any(token in clean for token in _FAULT_STATUS_UPDATE_MARKERS):
        score -= 8
    return score


def derive_fault_focus_text(episode: dict[str, Any]) -> str:
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    raw_candidates = [
        extracted.get("fault_focus_text"),
        extracted.get("symptom_raw"),
        *[
            (msg.get("text") or msg.get("content_summary") or "")
            for msg in episode.get("fault_description_messages") or []
            if isinstance(msg, dict)
        ],
    ]
    candidates: list[tuple[int, str]] = []
    for raw in raw_candidates:
        clean = _clean_fault_candidate(str(raw or ""))
        if not clean:
            continue
        # split to avoid long combined report text dominating
        clauses = [
            x.strip()
            for x in re.split(r"[。！？!?\n；;]+|(?:^|\s)[0-9]+[、.．]|(?:^|\s)[一二三四五六七八九十]+[、.．]", clean)
            if x.strip()
        ] or [clean]
        for clause in clauses:
            score = _fault_focus_score(clause)
            if score > -100:
                candidates.append((score, clause))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1]


def primary_fault_text(episode: dict[str, Any]) -> str:
    return derive_fault_focus_text(episode)


def has_fault_signal(text: str) -> bool:
    value = _clean_fault_candidate(str(text or ""))
    return any(token in value for token in _FAULT_HINTS)


def has_action_signal(text: str) -> bool:
    value = str(text or "")
    return any(token in value for token in _ACTION_HINTS)


def split_clauses(text: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(str(text or "")) if part.strip()]
    return parts or [str(text or "").strip()]


def is_coordination_text(text: str) -> bool:
    value = str(text or "")
    return any(token in value for token in _COORDINATION_MARKERS)


def episode_has_strong_fault_basis(episode: dict[str, Any]) -> bool:
    if str(episode.get("completeness") or "") == "noise":
        return False
    if has_fault_signal(primary_fault_text(episode)):
        return True
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages"):
        for msg in episode.get(key) or []:
            if isinstance(msg, dict) and has_fault_signal(str(msg.get("text") or msg.get("content_summary") or "")):
                return True
    return False


def episode_is_coordination_heavy(episode: dict[str, Any]) -> bool:
    if episode_has_strong_fault_basis(episode):
        return False
    texts: list[str] = []
    for key in ("fault_description_messages", "diagnostic_chain_messages", "case_context_messages"):
        for msg in episode.get(key) or []:
            if isinstance(msg, dict):
                text = str(msg.get("text") or msg.get("content_summary") or "")
                if text:
                    texts.append(text)
    if not texts:
        return False
    hits = sum(1 for text in texts if is_coordination_text(text))
    return hits >= max(1, len(texts) // 2)


def _dedupe_role_rows(rows: list[dict[str, Any]], *, role_type: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in rows:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        target = grouped.setdefault(name, {
            "name": name,
            "role_type": role_type,
            "confidence": 0.0,
            "reason": [],
            "evidence_message_ids": [],
        })
        target["confidence"] = max(float(target.get("confidence") or 0.0), float(item.get("confidence") or 0.0))
        target["reason"] = sorted(set([*target.get("reason", []), *item.get("reason", [])]))
        target["evidence_message_ids"] = sorted(set([*target.get("evidence_message_ids", []), *item.get("evidence_message_ids", [])]))
    out = list(grouped.values())
    out.sort(key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("name") or "")))
    return out


def _has_local_owner_request(text: str, name: str) -> bool:
    value = str(text or "")
    clauses = split_clauses(value)
    for clause in clauses:
        if (name in clause or f"@{name}" in clause) and any(marker in clause for marker in _OWNER_ACTION_MARKERS):
            return True
    escaped = re.escape(name)
    action_alt = "|".join(re.escape(marker) for marker in _OWNER_ACTION_MARKERS)
    return bool(
        re.search(rf"(?:@?{escaped}).{{0,96}}(?:{action_alt})", value)
        or re.search(rf"(?:{action_alt}).{{0,96}}(?:@?{escaped})", value)
    )


def sanitize_attribution(episode: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    raw = extracted.get("attribution") if isinstance(extracted.get("attribution"), dict) else {}
    raw_copy = json.loads(json.dumps(raw, ensure_ascii=False)) if raw else {
        "reporter_candidates": [],
        "owner_candidates": [],
        "owner_assignments": [],
        "responsibility_signals": [],
        "classification_hypotheses": [],
    }
    strong_fault = episode_has_strong_fault_basis(episode)
    coordination_heavy = episode_is_coordination_heavy(episode)
    if str(episode.get("completeness") or "") == "noise":
        return raw_copy, {
            "reporter_candidates": [],
            "owner_candidates": [],
            "owner_assignments": [],
            "responsibility_signals": [],
            "classification_hypotheses": [],
            "sanitized_by": "W7",
            "fault_basis": False,
            "coordination_heavy": True,
        }

    faultish_texts: list[str] = []
    for key in ("fault_description_messages", "diagnostic_chain_messages", "case_context_messages"):
        for msg in episode.get(key) or []:
            if isinstance(msg, dict):
                text = str(msg.get("text") or msg.get("content_summary") or "")
                if text:
                    faultish_texts.append(text)

    reporter_rows: list[dict[str, Any]] = []
    for item in raw_copy.get("reporter_candidates") or []:
        reasons = {str(x) for x in item.get("reason") or []}
        if coordination_heavy and "jira_submission" not in reasons:
            continue
        if not strong_fault and reasons <= {"field_feedback"}:
            continue
        reporter_rows.append(dict(item))

    hard_owner_rows: list[dict[str, Any]] = []
    for item in raw_copy.get("owner_assignments") or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        matched = False
        for text in faultish_texts:
            if _has_local_owner_request(text, name) and (has_fault_signal(text) or has_action_signal(text) or strong_fault):
                matched = True
                break
        if matched and not coordination_heavy:
            hard_owner_rows.append(dict(item))

    owner_rows = list(hard_owner_rows)
    hard_names = {str(item.get("name") or "") for item in hard_owner_rows}
    top_class_names = {str(item.get("name") or "") for item in (raw_copy.get("classification_hypotheses") or [])[:2]}
    for item in raw_copy.get("owner_candidates") or []:
        name = str(item.get("name") or "").strip()
        if not name or name in hard_names:
            continue
        reasons = {str(x) for x in item.get("reason") or []}
        keep = False
        if "diagnostic_takeover" in reasons and strong_fault:
            keep = True
        elif "responsibility_transfer" in reasons and strong_fault and not coordination_heavy:
            keep = True
        elif "direct_owner_request" in reasons and strong_fault and name in top_class_names and not coordination_heavy:
            keep = True
        if keep:
            owner_rows.append(dict(item))

    owner_assignments = _dedupe_role_rows(hard_owner_rows, role_type="issue_owner")
    owner_candidates = _dedupe_role_rows(owner_rows, role_type="issue_owner")
    reporter_candidates = _dedupe_role_rows(reporter_rows, role_type="reporter")

    kept_names = {str(item.get("name") or "") for item in [*owner_candidates, *reporter_candidates]}
    responsibility_signals = []
    for item in raw_copy.get("responsibility_signals") or []:
        name = str(item.get("name") or "").strip()
        sig = str(item.get("signal_type") or "")
        if sig == "reporter_signal" and reporter_candidates:
            responsibility_signals.append(dict(item))
        elif name and name in kept_names:
            responsibility_signals.append(dict(item))

    classification_rows: list[dict[str, Any]] = []
    for item in raw_copy.get("classification_hypotheses") or []:
        category = str(item.get("problem_category") or "")
        if not strong_fault:
            continue
        if coordination_heavy and not owner_candidates:
            continue
        if category == "其他问题及无法分类问题" and len(raw_copy.get("classification_hypotheses") or []) > 1:
            continue
        if owner_candidates and str(item.get("name") or "") not in {str(x.get("name") or "") for x in owner_candidates}:
            if float(item.get("confidence") or 0.0) < 0.55:
                continue
        classification_rows.append(dict(item))
    if not classification_rows and strong_fault and owner_candidates:
        owner_names = {str(item.get("name") or "") for item in owner_candidates}
        classification_rows = [dict(item) for item in (raw_copy.get("classification_hypotheses") or []) if str(item.get("name") or "") in owner_names][:2]

    sanitized = {
        "reporter_candidates": reporter_candidates,
        "owner_candidates": owner_candidates,
        "owner_assignments": owner_assignments,
        "responsibility_signals": responsibility_signals[:20],
        "classification_hypotheses": classification_rows[:3],
        "sanitized_by": "W7",
        "fault_basis": strong_fault,
        "coordination_heavy": coordination_heavy,
    }
    return raw_copy, sanitized


def resolve_people_roles(
    episode: dict[str, Any],
    attribution: dict[str, Any],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve stable organization roles separately from per-episode behavior."""

    explicit = people_index(registry)
    rows: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        episode_role: str,
        *,
        confidence: float,
        evidence_message_ids: list[str] | None = None,
        responsibility_scopes: list[str] | None = None,
    ) -> None:
        clean = str(name or "").strip()
        if not clean:
            return
        registry_row = explicit.get(clean) or {}
        target = rows.setdefault(clean, {
            "name": clean,
            "organization_roles": list(registry_row.get("organization_roles") or []),
            "episode_roles": [],
            "responsibility_scopes": list(registry_row.get("responsibility_scopes") or []),
            "confidence": 0.0,
            "status": "confirmed" if registry_row else "inferred",
            "evidence_message_ids": [],
        })
        if episode_role and episode_role not in target["episode_roles"]:
            target["episode_roles"].append(episode_role)
        target["confidence"] = max(float(target.get("confidence") or 0.0), confidence)
        target["evidence_message_ids"] = sorted(set([
            *target.get("evidence_message_ids", []),
            *(value for value in (evidence_message_ids or []) if value),
        ]))
        for scope in responsibility_scopes or []:
            if scope and scope not in target["responsibility_scopes"]:
                target["responsibility_scopes"].append(scope)

    classifications = attribution.get("classification_hypotheses") or []
    scopes_by_name: dict[str, list[str]] = {}
    for item in classifications:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        category = str(item.get("problem_category") or "")
        if name and category:
            scopes_by_name.setdefault(name, []).append(category)

    for item in attribution.get("reporter_candidates") or []:
        if isinstance(item, dict):
            add(
                str(item.get("name") or ""),
                "reporter",
                confidence=float(item.get("confidence") or 0.0),
                evidence_message_ids=[str(x) for x in item.get("evidence_message_ids") or []],
            )
    for item in attribution.get("owner_candidates") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        reasons = {str(x) for x in item.get("reason") or []}
        add(
            name,
            "assignee",
            confidence=float(item.get("confidence") or 0.0),
            evidence_message_ids=[str(x) for x in item.get("evidence_message_ids") or []],
            responsibility_scopes=scopes_by_name.get(name, []),
        )
        if "diagnostic_takeover" in reasons:
            add(name, "investigator", confidence=0.75, evidence_message_ids=[str(x) for x in item.get("evidence_message_ids") or []])

    anchor = episode.get("field_report_anchor") if isinstance(episode.get("field_report_anchor"), dict) else {}
    if anchor.get("author"):
        add(
            str(anchor.get("author") or ""),
            "field_report_author",
            confidence=0.8,
            evidence_message_ids=[str(anchor.get("message_id") or "")],
        )

    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages"):
        for message in episode.get(key) or []:
            if not isinstance(message, dict):
                continue
            sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
            name = str(sender.get("name") or "")
            message_id = str(message.get("message_id") or "")
            text = str(message.get("text") or message.get("content_summary") or "")
            if key == "diagnostic_chain_messages":
                add(name, "investigator", confidence=0.65, evidence_message_ids=[message_id])
            if key == "resolution_messages":
                add(name, "resolver", confidence=0.75, evidence_message_ids=[message_id])
                if any(marker in text for marker in ("复测", "验证", "现场", "未再出现", "恢复正常")):
                    add(name, "validator", confidence=0.75, evidence_message_ids=[message_id])
            if any(marker in text for marker in ("已上传", "已提供", "已发", "已导出", "日志已", "数据包已", "已提交JIRA", "已提交Jira")):
                add(name, "evidence_provider", confidence=0.7, evidence_message_ids=[message_id])

    return sorted(rows.values(), key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("name") or "")))


def concrete_action_texts(episode: dict[str, Any]) -> list[str]:
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    diags = [x for x in episode.get("diagnostic_chain_messages") or [] if isinstance(x, dict)]
    candidates = [
        *[str(x) for x in extracted.get("debug_actions") or [] if str(x).strip()],
        *[str(x.get("text") or x.get("content_summary") or "") for x in diags],
    ]
    out: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        clean = " ".join(str(text or "").split()).strip()
        if not clean:
            continue
        if any(marker in clean for marker in _ADDITIONAL10_NOISE_TOKENS + _NON_FAULT_PRIMARY_HINTS):
            continue
        if any(marker in clean for marker in _HELPY_HINTS):
            continue
        if any(marker in clean for marker in _NON_CONCRETE_ACTION_MARKERS):
            continue
        if clean.startswith(("另外，", "另外,", "此外，", "此外,")) and "确认下" in clean:
            continue
        if len(clean) > 260 and any(marker in clean for marker in ("1：", "1:", "2：", "2:", "[Media:", "匹配算法", "尽量选择")):
            continue
        if not has_action_signal(clean):
            continue
        if len(clean) > 180 and any(marker in clean for marker in ("客户反馈", "现场工作", "工作汇总", "项目")):
            continue
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def is_review_ready_episode(episode: dict[str, Any]) -> bool:
    if str(episode.get("completeness") or "") == "noise":
        return False
    faults = [x for x in episode.get("fault_description_messages") or [] if isinstance(x, dict)]
    diags = [x for x in episode.get("diagnostic_chain_messages") or [] if isinstance(x, dict)]
    if len(faults) != 1:
        return False
    if len(diags) <= 0 or len(diags) > 8:
        return False
    fault_text = primary_fault_text(episode)
    if len(fault_text) < 8 or len(fault_text) > 180:
        return False
    if any(token in fault_text for token in _ADDITIONAL10_NOISE_TOKENS):
        return False
    if len(_NUMBERED_BULLET.findall(fault_text)) >= 2:
        return False
    if fault_text.count("：") >= 3 or fault_text.count(":") >= 3:
        return False
    if fault_text.count("，") > 8:
        return False
    if any(token in fault_text for token in ("确认下是蓝屏还是黑屏", "让客户确认下", "还需要进行其他方面排查不")):
        return False
    if fault_text.startswith("@") and not has_fault_signal(fault_text):
        return False
    if any(token in fault_text for token in _NON_FAULT_PRIMARY_HINTS):
        return False
    if any(token in fault_text for token in _FAULT_STATUS_UPDATE_MARKERS):
        return False
    if any(token in fault_text for token in _HELPY_HINTS) and not has_fault_signal(fault_text):
        return False
    if not has_fault_signal(fault_text):
        return False
    action_count = len(concrete_action_texts(episode))
    if action_count == 0:
        return False
    if action_count < 2 and not any(marker in fault_text for marker in _ALLOW_SINGLE_ACTION_FAULT_MARKERS):
        return False
    return True


def review_ready_episode_score(episode: dict[str, Any]) -> float:
    faults = [x for x in episode.get("fault_description_messages") or [] if isinstance(x, dict)]
    diags = [x for x in episode.get("diagnostic_chain_messages") or [] if isinstance(x, dict)]
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    fault_text = primary_fault_text(episode)
    debug_actions = [str(x) for x in extracted.get("debug_actions") or [] if str(x).strip()]
    attachments = [x for x in episode.get("attachments") or [] if isinstance(x, dict)]
    score = 0.0
    score += min(len(fault_text), 160) / 40.0
    score += min(len(diags), 8) * 1.5
    score += min(len(debug_actions), 6) * 1.8
    score += min(len(attachments), 4) * 0.5
    if extracted.get("conclusion"):
        score += 2.0
    return round(score, 3)


def derive_fault_focus_confidence(episode: dict[str, Any]) -> float:
    focus = derive_fault_focus_text(episode)
    score = _fault_focus_score(focus)
    if score <= 0:
        return 0.0
    return round(min(score / 20.0, 1.0), 4)


def load_episode_index(path: str | Path = DEFAULT_EPISODES_JSON) -> dict[str, dict[str, Any]]:
    episodes = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item.get("episode_id") or ""): item for item in episodes if isinstance(item, dict) and item.get("episode_id")}


def _manual_family_label(canonical_error_id: str, error: dict[str, Any], payload: dict[str, Any]) -> str:
    canonical_map = {
        "err:camera-capture-failure": "相机拍摄失败",
        "err:industrial-pc-freeze-black-screen": "工控机蓝屏",
    }
    if canonical_error_id in canonical_map:
        return canonical_map[canonical_error_id]
    return _canonicalize_family_label(
        str(error.get("label") or canonical_error_id or ""),
        str(error.get("subsystem") or ""),
        str(error.get("category") or ""),
        " ".join([
            str(error.get("label") or ""),
            str(error.get("symptom") or ""),
            str(payload.get("review_summary") or ""),
        ]),
    )


def _manual_solution_outcome_type(text: str) -> str:
    raw = str(text or "")
    if any(k in raw for k in ("无效", "验证失败", "未解决", "仍出现", "仍复发")):
        return "ineffective"
    if any(k in raw for k in ("短时正常", "一度未出现", "暂未复发", "临时恢复")):
        return "partial_temporary"
    if any(k in raw for k in ("用于", "抓取", "定位", "排查", "分析")) and not any(k in raw for k in ("最终", "解决")):
        return "diagnostic_method"
    if any(k in raw for k in ("需人工确认", "待验证", "pending", "需人工")):
        return "pending_validation"
    if any(k in raw for k in ("最终判断", "根因", "最终解决")):
        return "pending_validation"
    return "pending_validation"


def _manual_outcome_type(value: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "ineffective": "ineffective",
        "partial_temporary": "partial_temporary",
        "candidate_final_fix_high_cost": "pending_validation",
        "temporary_recovery": "partial_temporary",
        "partial_then_recurred": "recurred",
        "mitigation_observed": "mitigation_observed",
        "workaround": "mitigation_observed",
        "pending_validation": "pending_validation",
        "pending_rnd_investigation": "pending_validation",
        "cleared_not_root_cause": "context_not_root_cause",
        "temporary_then_recurred": "recurred",
        "mitigation_uncertain": "mitigation_observed",
        "case_verified_fix": "verified_fix",
        "mitigation_observed_then_recurred": "recurred",
        "recommended_pending_validation": "pending_validation",
        "context_not_root_cause": "context_not_root_cause",
        "diagnostic_method": "diagnostic_method",
    }
    return mapping.get(raw, _manual_solution_outcome_type(raw))


def _manual_gold_structure(payload: dict[str, Any], family_label: str, error: dict[str, Any], checks: list[dict[str, Any]], solutions: list[dict[str, Any]]) -> dict[str, Any]:
    human = payload.get("human_correction") if isinstance(payload.get("human_correction"), dict) else {}
    required_info = [{"slot": infer_required_info_slot(str(text)), "question": str(text)} for text in (error.get("required_info") or [])[:8]]
    if human.get("correct_modeling") == "split_episode_into_two_candidates":
        primary_variant = str(human.get("primary_error_label") or "")
        secondary_variant = str(human.get("secondary_error_label") or "")
        primary_family = "相机拍摄失败" if any(k in primary_variant for k in ("拍摄失败", "不拍照", "相机")) else _canonicalize_family_label(primary_variant, "", "", primary_variant)
        secondary_family = "工控机蓝屏" if any(k in secondary_variant for k in ("蓝屏", "igdkmdn64", "驱动")) else _canonicalize_family_label(secondary_variant, "", "", secondary_variant)
        primary_actions = [{"label": str(x)} for x in human.get("primary_check_nodes") or [] if str(x).strip()]
        secondary_actions = [{"label": str(x)} for x in human.get("secondary_check_nodes") or [] if str(x).strip()]
        raw_outcomes = [x for x in human.get("solution_or_outcome_nodes") or [] if isinstance(x, dict)]
        primary_outcomes = []
        secondary_outcomes = []
        for item in raw_outcomes:
            entry = {
                "action_label": str(item.get("label") or ""),
                "outcome_type": _manual_outcome_type(str(item.get("outcome") or "")),
                "summary": str(item.get("note") or ""),
            }
            label = str(item.get("label") or "")
            if "过滤驱动" in label or "拍照" in label:
                primary_outcomes.append(entry)
            else:
                secondary_outcomes.append(entry)
        primary_required = [
            {"slot": "ip_config", "question": "请提供相机网卡与非相机网卡的区分、过滤驱动勾选状态和网口截图。"},
            {"slot": "log_package", "question": "请提供拍摄失败时的诊断日志和报错截图。"},
        ]
        secondary_required = [
            {"slot": "dmp_package", "question": "请提供蓝屏对应的DMP/minidump文件。"},
            {"slot": "software_version", "question": "请提供Intel核显驱动版本、系统版本和近期驱动变更记录。"},
        ]
        return {
            "cases": [
                {
                    "family": {"label": primary_family},
                    "variant": {"label": primary_variant},
                    "actions": primary_actions,
                    "outcomes": primary_outcomes,
                    "required_info": primary_required,
                    "trace": {"recommended_action_labels": [x["label"] for x in primary_actions]},
                },
                {
                    "family": {"label": secondary_family},
                    "variant": {"label": secondary_variant},
                    "actions": secondary_actions,
                    "outcomes": secondary_outcomes,
                    "required_info": secondary_required,
                    "trace": {"recommended_action_labels": [x["label"] for x in secondary_actions]},
                },
            ]
        }

    if human:
        clean_variant = str(human.get("correct_error_label") or error.get("label") or "")
        clean_actions = [{"label": str(x)} for x in human.get("check_nodes") or [] if str(x).strip()]
        clean_outcomes = []
        for item in human.get("solution_or_outcome_nodes") or []:
            if not isinstance(item, dict):
                continue
            clean_outcomes.append({
                "action_label": str(item.get("label") or ""),
                "outcome_type": _manual_outcome_type(str(item.get("outcome") or "")),
                "summary": str(item.get("note") or ""),
            })
        return {
            "family": {"label": family_label},
            "variant": {"label": clean_variant},
            "actions": clean_actions,
            "outcomes": clean_outcomes,
            "required_info": required_info,
            "trace": {"recommended_action_labels": [x["label"] for x in clean_actions]},
        }

    return {
        "family": {"label": family_label},
        "variant": {"label": str(error.get("label") or "")},
        "actions": [{"label": str(item.get("label") or "")} for item in checks[:12]],
        "outcomes": [{"action_label": str(item.get("content") or ""), "outcome_type": _manual_solution_outcome_type(str(item.get("content") or ""))} for item in solutions[:12]],
        "required_info": required_info,
        "trace": {"recommended_action_labels": [str(item.get("label") or "") for item in checks[:12]]},
    }


def manual_example_to_reviewed(payload: dict[str, Any], source_file: str) -> dict[str, Any]:
    refined = payload.get("refined_merge_proposal") if isinstance(payload.get("refined_merge_proposal"), dict) else {}
    nodes = [item for item in refined.get("nodes") or [] if isinstance(item, dict)]
    error = next((item for item in nodes if item.get("type") == "Error"), {})
    checks = [item for item in nodes if item.get("type") == "DiagnosticCheck"]
    solutions = [item for item in nodes if item.get("type") == "Solution"]
    evidence_findings = [item for item in payload.get("evidence_findings") or [] if isinstance(item, dict)]
    canonical_error_id = str(refined.get("canonical_error_id") or payload.get("manual_decision", {}).get("canonical_error_id") or "")
    family_label = _manual_family_label(canonical_error_id, error, payload)
    exact_reuse_allowed = bool(error and checks)
    gold_structure = _manual_gold_structure(payload, family_label, error, checks, solutions)
    if gold_structure.get("cases"):
        exact_reuse_allowed = True
    return {
        "case_id": str(payload.get("sample_id") or Path(source_file).stem),
        "source_episode_id": str(payload.get("source_episode_id") or ""),
        "family_label": family_label,
        "variant_label": str(error.get("label") or ""),
        "source_excerpt": [str(item.get("summary") or item.get("finding") or "") for item in evidence_findings[:6]],
        "evidence_anchor_map": {str(item.get("message_id") or f"m{idx+1}"): str(item.get("summary") or item.get("finding") or "") for idx, item in enumerate(evidence_findings[:12])},
        "gold": gold_structure,
        "review_type": "manual_review",
        "exact_reuse_allowed": exact_reuse_allowed,
        "source_file": source_file,
    }


def load_reviewed_examples(gold_root: str | Path = DEFAULT_GOLD_ROOT, manual_root: str | Path = DEFAULT_MANUAL_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(gold_root).glob("goldcase-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        gold = payload.get("gold") if isinstance(payload.get("gold"), dict) else {}
        family = gold.get("family") if isinstance(gold.get("family"), dict) else {}
        variant = gold.get("variant") if isinstance(gold.get("variant"), dict) else {}
        rows.append({
            "case_id": str(payload.get("case_id") or path.stem),
            "source_episode_id": str(payload.get("source_episode_id") or ""),
            "family_label": str(family.get("label") or ""),
            "variant_label": str(variant.get("label") or ""),
            "source_excerpt": payload.get("source_excerpt") or [],
            "evidence_anchor_map": payload.get("evidence_anchor_map") or {},
            "gold": gold,
            "review_type": "gold_case",
            "source_file": str(path),
        })
    for path in sorted(Path(manual_root).glob("chat-rank-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(manual_example_to_reviewed(payload, str(path)))
    return rows


def load_sop_seed_background(path: str | Path = DEFAULT_SOP_SEED_JSON) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    families = [x for x in payload.get("family_seed_view") or [] if isinstance(x, dict)]
    return {
        "root": str(path),
        "families": families,
    }


def score_family(episode_text_value: str, family_view: dict[str, Any]) -> float:
    text = norm_text(episode_text_value)
    if not text:
        return 0.0
    family = family_view.get("family") if isinstance(family_view.get("family"), dict) else {}
    actions = family_view.get("action_templates") if isinstance(family_view.get("action_templates"), list) else []
    reqs = family_view.get("required_info_templates") if isinstance(family_view.get("required_info_templates"), list) else []
    generic_penalty = 0.0
    family_label = str(family.get("label") or "")
    if family_label in {"算法/程序调优异常", "主程序/系统异常", "硬件/运控异常"}:
        generic_penalty = 4.0
    features = [
        str(family.get("label") or ""),
        str(family.get("summary") or ""),
        str(family.get("subsystem") or ""),
        str(family.get("scenario") or ""),
        *[str(x) for x in family.get("keywords") or []],
        *[str((x or {}).get("title") or "") for x in family_view.get("source_sections") or []],
        *[str(x.get("label") or "") for x in actions[:12]],
        *[str(x.get("slot") or "") for x in reqs[:8]],
    ]
    score = 0.0
    for feat in features:
        norm = norm_text(feat)
        if not norm:
            continue
        token_hits = sum(1 for token in norm.split() if token and token in text)
        if token_hits:
            score += token_hits
        elif norm in text:
            score += 2.0
    return max(0.0, score - generic_penalty)


def score_reviewed_example(episode_text_value: str, example: dict[str, Any], top_family_labels: set[str]) -> float:
    text = norm_text(episode_text_value)
    if not text:
        return 0.0
    score = 0.0
    family_label = str(example.get("family_label") or "")
    variant_label = str(example.get("variant_label") or "")
    if family_label and family_label in top_family_labels:
        score += 8.0
    if str(example.get("review_type") or "") == "manual_review":
        score += 2.0
    blob = " ".join([
        family_label,
        variant_label,
        *[str(x) for x in example.get("source_excerpt") or [] if isinstance(x, str)],
        *[str((x or {}).get("label") or "") for x in (example.get("gold") or {}).get("actions") or [] if isinstance(x, dict)],
        *[str((x or {}).get("question") or "") for x in (example.get("gold") or {}).get("required_info") or [] if isinstance(x, dict)],
    ])
    norm = norm_text(blob)
    if norm:
        score += sum(1 for token in norm.split() if token and token in text)
    return score


def reviewed_examples_for_episode(episode: dict[str, Any], examples: list[dict[str, Any]], top_families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = episode_text(episode)
    top_family_labels = {str(item.get("label") or "") for item in top_families}
    ranked: list[tuple[float, dict[str, Any]]] = []
    episode_id = str(episode.get("episode_id") or "")
    for example in examples:
        score = score_reviewed_example(text, example, top_family_labels)
        exact_source_match = (
            str(example.get("source_episode_id") or "") == episode_id
            or (
                str(example.get("source_episode_id") or "") == "pending_repo_binding"
                and str(example.get("case_id") or "") == episode_id
            )
        )
        if exact_source_match:
            score += 1000.0
        if score <= 0:
            continue
        ranked.append((score, {**example, "_exact_source_match": exact_source_match}))
    ranked.sort(
        key=lambda item: (
            0 if item[1].get("_exact_source_match") and str(item[1].get("review_type") or "") == "gold_case" else
            1 if item[1].get("_exact_source_match") else
            2,
            0 if str(item[1].get("review_type") or "") == "gold_case" else 1,
            -item[0],
            str(item[1].get("case_id") or ""),
        )
    )
    out = []
    for score, example in ranked[:3]:
        gold = example.get("gold") if isinstance(example.get("gold"), dict) else {}
        out.append({
            "score": round(score, 3),
            "case_id": example.get("case_id") or "",
            "source_episode_id": example.get("source_episode_id") or "",
            "exact_source_match": bool(example.get("_exact_source_match")),
            "review_type": str(example.get("review_type") or ""),
            "exact_reuse_allowed": bool(example.get("exact_reuse_allowed")),
            "family_label": example.get("family_label") or "",
            "variant_label": example.get("variant_label") or "",
            "source_excerpt": example.get("source_excerpt") or [],
            "gold_structure": {
                "cases": gold.get("cases") or [],
                "family": gold.get("family") or {},
                "variant": gold.get("variant") or {},
                "actions": gold.get("actions") or [],
                "outcomes": gold.get("outcomes") or [],
                "required_info": gold.get("required_info") or [],
                "trace": gold.get("trace") or {},
            },
        })
    return out


def build_sop_background_for_episode(episode: dict[str, Any], sop: dict[str, Any], reviewed_examples: list[dict[str, Any]]) -> dict[str, Any]:
    text = episode_text(episode)
    ranked = []
    for family_view in sop["families"]:
        family = family_view.get("family") if isinstance(family_view.get("family"), dict) else {}
        actions = family_view.get("action_templates") if isinstance(family_view.get("action_templates"), list) else []
        reqs = family_view.get("required_info_templates") if isinstance(family_view.get("required_info_templates"), list) else []
        score = score_family(text, family_view)
        if score > 0:
            ranked.append((score, family, actions, reqs, family_view))
    ranked.sort(key=lambda item: item[0], reverse=True)
    top = []
    for score, family, actions, reqs, family_view in ranked[:4]:
        top.append({
            "score": round(score, 3),
            "family_id": family.get("family_id") or "",
            "label": family.get("label") or "",
            "summary": family.get("summary") or "",
            "category": family.get("category") or "",
            "subsystem": family.get("subsystem") or "",
            "scenario": family.get("scenario") or "",
            "source_sections": family_view.get("source_sections") or [],
            "suggested_actions": [
                {
                    "label": item.get("label") or "",
                    "summary": item.get("summary") or "",
                    "action_role": item.get("action_role") or "",
                    "support_count": item.get("support_count") or 0,
                }
                for item in actions[:8]
            ],
            "suggested_required_info": [
                {
                    "slot": item.get("slot") or "",
                    "question": item.get("question") or "",
                    "why_required": item.get("why_required") or "",
                    "support_count": item.get("support_count") or 0,
                }
                for item in reqs[:8]
            ],
            "trace_template": family_view.get("trace_template") or {},
        })
    return {
        "source": sop.get("root") or DEFAULT_SOP_SEED_JSON,
        "top_family_background": top,
        "reviewed_case_examples": reviewed_examples_for_episode(episode, reviewed_examples, top),
    }


def _chat_id(episode: dict[str, Any]) -> str:
    for value in (episode.get("source_chat_id"), episode.get("chat_id"), episode.get("thread_id"), episode.get("episode_id")):
        match = _CHAT_ID_RE.search(str(value or ""))
        if match:
            return match.group(1)
    return ""


@lru_cache(maxsize=256)
def _history_file(text_history_root: str, chat_id: str) -> str:
    root = Path(text_history_root)
    if not root.exists() or not chat_id:
        return ""
    matches = sorted(root.glob(f"*{chat_id}.jsonl"))
    return str(matches[0]) if matches else ""


@lru_cache(maxsize=256)
def _read_history_chat(path_text: str) -> tuple[dict[str, Any], ...]:
    if not path_text:
        return ()
    path = Path(path_text)
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            rows.append({
                "message_id": str(raw.get("message_id") or ""),
                "create_time": str(raw.get("create_time") or ""),
                "sender": str(raw.get("sender") or ""),
                "text": str(raw.get("plain_text") or raw.get("content") or "").strip(),
                "chat_id": str(raw.get("chat_id") or ""),
                "chat_name": str(raw.get("chat_name") or ""),
                "_history_index": index,
                "_evidence_source": "text_history_same_chat",
            })
    return tuple(rows)


def _case_terms(value: str) -> set[str]:
    text = str(value or "").lower()
    terms = {
        token for token in re.findall(r"[a-z][a-z0-9_.+-]{2,}", text)
        if token not in {"aoi", "the", "and", "jira"}
    }
    stop = {"客户", "现场", "设备", "问题", "反馈", "老师", "程序", "软件", "异常", "发生", "时间"}
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        for index in range(max(0, len(run) - 1)):
            term = run[index:index + 2]
            if term and term not in stop:
                terms.add(term)
    return terms


def _fault_signature(value: str) -> set[str]:
    lowered = str(value or "").lower()
    return {marker for marker in _CASE_FAULT_SIGNATURE_MARKERS if marker.lower() in lowered}


def _current_case_messages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, key in (
        ("fault", "fault_description_messages"),
        ("diagnostic", "diagnostic_chain_messages"),
        ("resolution", "resolution_messages"),
    ):
        for item in episode.get(key) or []:
            if isinstance(item, dict):
                rows.append({**item, "_case_role": role})
    return rows


def _supplemental_history_window(
    episode: dict[str, Any], *, text_history_root: str = DEFAULT_TEXT_HISTORY_ROOT
) -> list[dict[str, Any]]:
    rows = list(_read_history_chat(_history_file(str(text_history_root), _chat_id(episode))))
    if not rows:
        return []
    current_ids = {
        str(item.get("message_id") or "")
        for item in _current_case_messages(episode)
        if str(item.get("message_id") or "")
    }
    positions = [index for index, item in enumerate(rows) if str(item.get("message_id") or "") in current_ids]
    if not positions:
        return []
    start = max(0, min(positions) - 16)
    end = min(len(rows), max(positions) + 32)
    anchor = min(positions)
    return [
        {**item, "_context_distance": index - anchor}
        for index, item in enumerate(rows[start:end], start=start)
    ]


def _local_case_context_window(episode: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in episode.get("case_context_messages") or [] if isinstance(item, dict)]
    if not rows:
        return []
    current_ids = {
        str(item.get("message_id") or "")
        for item in _current_case_messages(episode)
        if str(item.get("message_id") or "")
    }
    positions = [index for index, item in enumerate(rows) if str(item.get("message_id") or "") in current_ids]
    if not positions:
        return []
    start = max(0, min(positions) - 12)
    end = min(len(rows), max(positions) + 24)
    anchor = min(positions)
    return [
        {**item, "_context_distance": index - anchor, "_history_index": index}
        for index, item in enumerate(rows[start:end], start=start)
    ]


def _case_relevant_fragment(text: str, anchor_signature: set[str]) -> str:
    """Keep only the numbered clause relevant to the current episode.

    W7 may find a same-chat message that contains several daily-report items.
    Message-level promotion is useful for provenance, but passing the whole
    message to W2 turns unrelated problems into current-case evidence.  Keep
    the matching numbered items while retaining the original in ``raw_text``.
    """
    value = str(text or "").strip()
    matches = list(re.finditer(r"(?:(?<=^)|(?<=[\s。；;：:]))([1-9]\d?)[、.．:：/]\s*", value))
    if len(matches) < 2 or not anchor_signature:
        return value
    sections: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        section = value[match.start():end].strip(" \n。；;")
        if section and _fault_signature(section) & anchor_signature:
            sections.append(section)
    return " ".join(sections) if sections else value


def promote_case_evidence(
    episode: dict[str, Any],
    *,
    supplemental_messages: list[dict[str, Any]] | None = None,
    text_history_root: str = DEFAULT_TEXT_HISTORY_ROOT,
    jira_offline_root: str = DEFAULT_JIRA_OFFLINE_ROOT,
    limit: int = 20,
) -> dict[str, Any]:
    """Promote nearby same-case chat/Jira evidence without changing W1 roles."""

    out = json.loads(json.dumps(episode, ensure_ascii=False))
    current = _current_case_messages(out)
    current_ids = {str(item.get("message_id") or "") for item in current if item.get("message_id")}
    anchor_text = " ".join(
        [derive_fault_focus_text(out), *[str(item.get("text") or item.get("content_summary") or "") for item in current]]
    )
    anchor_terms = _case_terms(anchor_text)
    anchor_signature = _fault_signature(anchor_text)
    candidates = (
        supplemental_messages
        if supplemental_messages is not None
        else _supplemental_history_window(out, text_history_root=text_history_root)
    )
    candidates = [*candidates, *_local_case_context_window(out)]
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("message_id") or "")
        if not message_id or message_id in current_ids:
            continue
        text = str(item.get("text") or item.get("plain_text") or item.get("content_summary") or "").strip()
        if not text:
            continue
        existing = unique.get(message_id)
        # Prefer records carrying a real local distance over a longer but
        # unbounded copy from the original context list.
        if existing is None or (
            "_context_distance" in item and "_context_distance" not in existing
        ) or (
            ("_context_distance" in item) == ("_context_distance" in existing)
            and len(text) > len(str(existing.get("text") or ""))
        ):
            unique[message_id] = {**item, "message_id": message_id, "text": text}

    jira_parser = JiraParserAgent(offline_root=jira_offline_root)
    jira_details_by_message: dict[str, list[dict[str, Any]]] = {}
    for message_id, item in unique.items():
        text = str(item.get("text") or "")
        if not _JIRA_KEY_RE.search(text) and "jira" not in text.lower():
            continue
        parsed = jira_parser.parse(text)
        relevant = [
            detail
            for detail in parsed.get("offline_details") or []
            if anchor_signature & _fault_signature(str(detail.get("summary") or ""))
        ]
        if relevant:
            jira_details_by_message[message_id] = relevant

    # One field-report item may sit near several later incidents with the same
    # broad symptom (for example many unrelated "闪退" tickets).  Bind the
    # promotion window to the current Jira when present; otherwise infer at
    # most one nearest, best-overlap Jira as the case anchor.
    allowed_jira_keys = set(_JIRA_KEY_RE.findall(anchor_text))
    if not allowed_jira_keys and jira_details_by_message:
        ranked_jira: list[tuple[int, int, str]] = []
        for message_id, details in jira_details_by_message.items():
            item = unique.get(message_id) or {}
            distance = abs(int(item.get("_context_distance") or 999))
            for detail in details:
                key = str(detail.get("issue_key") or "")
                summary = str(detail.get("summary") or "")
                overlap_score = len(anchor_terms & _case_terms(summary))
                if key:
                    ranked_jira.append((-overlap_score, distance, key))
        if ranked_jira:
            ranked_jira.sort()
            allowed_jira_keys = {ranked_jira[0][2]}
    if allowed_jira_keys:
        jira_details_by_message = {
            message_id: [
                detail for detail in details
                if str(detail.get("issue_key") or "") in allowed_jira_keys
            ]
            for message_id, details in jira_details_by_message.items()
        }
        jira_details_by_message = {
            message_id: details for message_id, details in jira_details_by_message.items() if details
        }

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for item in unique.values():
        text = str(item.get("text") or "")
        lowered = text.lower()
        overlap = anchor_terms & _case_terms(text)
        signature_overlap = anchor_signature & _fault_signature(text)
        distinctive_signature_overlap = signature_overlap - _GENERIC_CASE_SIGNATURES
        jira_keys = _JIRA_KEY_RE.findall(text)
        distance = abs(int(item.get("_context_distance") or 999))
        result_marker = any(marker.lower() in lowered for marker in _CASE_RESULT_MARKERS)
        strong_result_marker = any(marker.lower() in lowered for marker in _CASE_STRONG_RESULT_MARKERS)
        report_noise = any(marker.lower() in lowered for marker in _CASE_CONTEXT_NOISE)
        jira_relevant = bool(jira_keys and jira_details_by_message.get(str(item.get("message_id") or "")))
        if allowed_jira_keys:
            if jira_keys and allowed_jira_keys.isdisjoint(jira_keys):
                continue
            if not jira_keys and (
                distance > 3
                or not (result_marker or any(marker.lower() in lowered for marker in _CASE_EVIDENCE_MARKERS))
            ):
                continue
        if report_noise:
            continue
        # Generic markers such as ``重启`` or ``初始化`` occur in many
        # neighbouring field reports.  They cannot establish same-case
        # evidence by themselves; require a distinctive fault marker, a
        # case-matched Jira, or a strong result sentence with concrete term
        # overlap.
        if not distinctive_signature_overlap and not jira_relevant and not (strong_result_marker and len(overlap) >= 2):
            continue
        if distance > 12 and not result_marker and not jira_relevant:
            continue
        score = min(6, len(overlap)) + min(6, len(signature_overlap) * 2)
        if jira_relevant:
            score += 4
        if any(marker.lower() in lowered for marker in _CASE_EVIDENCE_MARKERS):
            score += 2
        if result_marker:
            score += 4
        if distance <= 8:
            score += 2
        elif distance <= 20:
            score += 1
        if score >= 4:
            scored.append((score, int(item.get("_history_index") or 10**9), item))

    selected = [item for _, _, item in sorted(scored, key=lambda row: (-row[0], row[1]))[: max(0, limit)]]
    selected.sort(key=lambda item: (int(item.get("_history_index") or 10**9), str(item.get("create_time") or "")))
    for item in selected:
        item["promotion_reason"] = "same_chat_case_evidence"
        original_text = str(item.get("text") or "")
        relevant_fragment = _case_relevant_fragment(original_text, anchor_signature)
        if relevant_fragment != original_text:
            item["raw_text"] = original_text
            item["text"] = relevant_fragment
            item["promotion_reason"] = "same_chat_case_evidence:fragment_filtered"
        details = jira_details_by_message.get(str(item.get("message_id") or ""), [])
        jira_keys = _JIRA_KEY_RE.findall(str(item.get("text") or ""))
        if details and len(set(jira_keys)) > len(details):
            # Keep the original message for audit, but expose only the
            # case-matched Jira summaries to W2.  A single chat message often
            # batches several unrelated Jira links.
            item["raw_text"] = str(item.get("text") or "")
            item["text"] = "；".join(
                f"相关问题已提交 {detail.get('issue_key')}：{detail.get('summary')}"
                for detail in details
            )
            item["promotion_reason"] = "same_chat_case_evidence:jira_filtered"

    jira_evidence: list[dict[str, Any]] = []
    seen_jira: set[str] = set()
    for item in selected:
        for detail in jira_details_by_message.get(str(item.get("message_id") or ""), []):
            key = str(detail.get("issue_key") or "")
            if not key or key in seen_jira:
                continue
            seen_jira.add(key)
            jira_evidence.append(detail)

    out["case_evidence_messages"] = selected
    evidence_ids = [str(value) for value in out.get("evidence_message_ids") or [] if str(value)]
    evidence_ids.extend(str(item.get("message_id") or "") for item in selected if item.get("message_id"))
    out["evidence_message_ids"] = list(dict.fromkeys(evidence_ids))
    extracted = out.get("extracted") if isinstance(out.get("extracted"), dict) else {}
    extracted["linked_jira_evidence"] = jira_evidence
    extracted["case_evidence_promotion"] = {
        "agent_id": "W7",
        "policy": "same_chat_nearby_diagnostic_and_jira.v1",
        "promoted_message_ids": [str(item.get("message_id") or "") for item in selected],
        "linked_jira_keys": [str(item.get("issue_key") or "") for item in jira_evidence],
        "split_risk": len(jira_evidence) > 1,
    }
    out["extracted"] = extracted
    return out


def _w7_message_text(message: dict[str, Any]) -> str:
    return str(message.get("text") or message.get("content_summary") or "").strip()


def _w7_is_question_or_pending(text: str) -> bool:
    value = str(text or "")
    return (
        any(marker in value for marker in _RESOLUTION_QUESTION_MARKERS)
        or any(marker in value for marker in _RESOLUTION_PENDING_MARKERS)
        or value.rstrip().endswith(("?", "？"))
    )


def _w7_has_executed_fix_action(text: str) -> bool:
    value = str(text or "")
    return bool(
        any(marker in value for marker in _RESOLUTION_ACTION_MARKERS)
        and not any(marker in value for marker in _W7_NON_EXECUTION_MARKERS)
        and not value.rstrip().endswith(("?", "？"))
    )


def _w7_message_outcome_type(text: str, *, prior_fix_action: bool = False) -> tuple[str, list[str]]:
    """Classify one action/outcome event into the canonical W7 taxonomy."""
    value = str(text or "")
    reasons: list[str] = []
    has_fix_action = _w7_has_executed_fix_action(value)
    has_positive = any(marker in value for marker in _RESOLUTION_POSITIVE_MARKERS) or bool(
        re.search(r"后.{0,24}(?:恢复)?正常", value)
        or re.search(r"后.{0,24}(?:能|可以)(?:正常)?(?:进入|运行|使用|连接|拍照|拍摄|测试)", value)
    )
    has_recurrence = any(marker in value for marker in (*_INEFFECTIVE_MARKERS, *_W7_RECURRENCE_MARKERS))
    if value.count("[TEST-") + value.count("[SMTAOITS-") >= 2:
        return "diagnostic_method", ["multi_jira_context_not_direct_validation"]
    if any(marker in value for marker in _REPORT_ONLY_MARKERS) and len(re.findall(r"(?:^|\s)[一二三四五六七八九十\d]+[、.．]", value)) >= 2:
        return "pending_validation", ["multi_issue_report_not_direct_validation"]
    if has_positive and has_recurrence:
        return "partial_temporary", ["recovery_followed_by_recurrence"]
    if has_recurrence:
        return "ineffective", ["failure_or_recurrence_observed"]
    if _w7_is_question_or_pending(value):
        return "pending_validation", ["question_or_pending_language"]
    duration = re.search(r"(\d+|[一二三四五六七八九十百]+)\s*(分钟|小时)", value)
    if duration and any(marker in value for marker in ("正常", "未反馈", "没有报", "无异常")):
        raw_amount, unit = duration.group(1), duration.group(2)
        amount = int(raw_amount) if raw_amount.isdigit() else (1 if raw_amount in {"一", "壹"} else 2)
        if unit == "分钟" or amount < 2:
            return "partial_temporary", ["short_observation_window"]
        reasons.append("observation_window_at_least_two_hours")
    if has_positive:
        if not (has_fix_action or prior_fix_action):
            return "pending_validation", ["positive_claim_without_executed_fix"]
        action_positions = [value.find(marker) for marker in _RESOLUTION_ACTION_MARKERS if marker in value]
        action_position = min(action_positions) if action_positions else -1
        validation_positions = [value.find(marker) for marker in _W7_VALIDATION_MARKERS if marker in value]
        has_validation = any(
            position >= 0 and (prior_fix_action or action_position < 0 or position > action_position)
            for position in validation_positions
        )
        explicit_closed = any(marker in value for marker in ("已解决", "解决了", "验证通过"))
        workaround_only = any(marker in value for marker in ("重启", "断电", "拔插", "重新打开", "强制重启"))
        if workaround_only and not has_validation:
            return "partial_temporary", ["restart_or_replug_without_validation"]
        if has_validation or (explicit_closed and prior_fix_action) or reasons:
            return "verified_fix", [*reasons, "action_and_validation_present"]
        return "mitigation_observed", ["immediate_recovery_without_validation"]
    if has_fix_action:
        return "pending_validation", ["action_without_recorded_outcome"]
    if any(marker in value for marker in ("检查", "分析", "日志", "确认", "排查", "收集")):
        return "diagnostic_method", ["diagnostic_action"]
    return "pending_validation", ["no_outcome_evidence"]


def _w7_resolution_status(text: str) -> str:
    """Legacy four-state projection retained for downstream compatibility."""
    outcome_type, _ = _w7_message_outcome_type(text)
    if outcome_type == "verified_fix":
        return "verified"
    # Compatibility for legacy consumers that treated a targeted same-message
    # action/result as verified.  The canonical outcome remains
    # mitigation_observed until W7 sees explicit validation.
    if outcome_type == "mitigation_observed" and re.search(r"后.{0,24}(?:恢复)?正常", str(text or "")):
        return "verified"
    if outcome_type == "ineffective":
        return "ineffective"
    if outcome_type in {"partial_temporary", "mitigation_observed", "pending_validation"}:
        return "pending"
    return "unknown"


def _w7_action_outcome_state(episode: dict[str, Any]) -> dict[str, Any]:
    """Build an ordered action -> outcome -> validation -> recurrence state."""
    messages = _w7_episode_messages(episode)
    messages.sort(key=lambda item: (str(item.get("create_time") or ""), str(item.get("message_id") or "")))
    events: list[dict[str, Any]] = []
    prior_fix_action = False
    verified_index: int | None = None
    for index, message in enumerate(messages):
        text = _w7_message_text(message)
        has_fix_action = _w7_has_executed_fix_action(text)
        outcome_type, reasons = _w7_message_outcome_type(text, prior_fix_action=prior_fix_action)
        if has_fix_action:
            prior_fix_action = True
        if outcome_type == "verified_fix":
            verified_index = index
        events.append({
            "message_id": str(message.get("message_id") or message.get("source_message_id") or ""),
            "create_time": str(message.get("create_time") or ""),
            "stage": (
                "validation" if outcome_type == "verified_fix" else
                "recurrence" if outcome_type in {"ineffective", "partial_temporary"} else
                "outcome" if outcome_type == "mitigation_observed" else
                "action" if has_fix_action else "diagnostic"
            ),
            "action_executed": has_fix_action,
            "outcome_type": outcome_type,
            "reason_codes": reasons,
        })
    later_recurrence = bool(
        verified_index is not None
        and any(event["outcome_type"] in {"ineffective", "partial_temporary"} for event in events[verified_index + 1:])
    )
    if later_recurrence:
        final_outcome = "partial_temporary"
    elif any(event["outcome_type"] == "verified_fix" for event in events):
        final_outcome = "verified_fix"
    elif any(event["outcome_type"] == "partial_temporary" for event in events):
        final_outcome = "partial_temporary"
    elif any(event["outcome_type"] == "mitigation_observed" for event in events):
        final_outcome = "mitigation_observed"
    elif any(event["outcome_type"] == "ineffective" for event in events):
        final_outcome = "ineffective"
    elif any(event["outcome_type"] == "execution_failed" for event in events):
        final_outcome = "execution_failed"
    elif any(event["outcome_type"] == "diagnostic_method" for event in events):
        final_outcome = "diagnostic_method"
    else:
        final_outcome = "pending_validation"
    return {
        "schema_version": "w7.action_outcome_state.v1",
        "events": events,
        "final_outcome_type": final_outcome,
        "verified_fix_requirements": {
            "executed_fix_action": prior_fix_action,
            "validation_or_observation": any(event["outcome_type"] == "verified_fix" for event in events),
            "no_later_recurrence": not later_recurrence,
        },
    }


def _w7_is_outcome_statement(text: str) -> bool:
    value = str(text or "")
    if _w7_is_question_or_pending(value) and not any(marker in value for marker in ("暂时正常", "临时恢复", "未再出现", "不再出现")):
        return False
    return any(marker in value for marker in (*_INEFFECTIVE_MARKERS, *_RESOLUTION_POSITIVE_MARKERS)) or bool(
        re.search(r"(?:处理|更换|调整|升级|回退|重装|安装).{0,24}后.{0,24}(?:正常|无异常|仍|还是|依旧)", value)
    )


def _w7_episode_messages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "outcome_messages", "noise_messages", "case_context_messages"):
        for message in episode.get(key) or []:
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            if message_id and message_id in seen:
                continue
            if message_id:
                seen.add(message_id)
            result.append(message)
    return result


def _w7_core_messages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "outcome_messages"):
        for message in episode.get(key) or []:
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            if message_id and message_id in seen:
                continue
            if message_id:
                seen.add(message_id)
            result.append(message)
    return result


def _w7_is_multi_fault_episode(episode: dict[str, Any]) -> bool:
    anchor = episode.get("field_report_anchor") or {}
    if isinstance(anchor, dict) and int(anchor.get("issue_count") or 0) > 1:
        return True
    for message in _w7_core_messages(episode):
        text = _w7_message_text(message)
        sections = [
            part.strip()
            for part in re.split(r"(?:^|[\s。；;,，])[一二三四五六七八九十]+[、.．,，]|(?:^|[\s。；;,，])\d+[、.．,，]", text)
            if part.strip()
        ]
        signature_sets = [
            frozenset(_fault_signature(part) & _W7_PRIMARY_FAULT_SIGNATURES)
            for part in sections
            if _fault_signature(part) & _W7_PRIMARY_FAULT_SIGNATURES
        ]
        if len(set(signature_sets)) >= 2 and len(set().union(*signature_sets)) >= 2:
            return True
    return False


def _w7_numbered_fault_sections(episode: dict[str, Any]) -> list[tuple[str, set[str], str]]:
    """Extract fault sections from report-like messages, not procedure steps.

    A numbered troubleshooting procedure must remain one action chain.  This
    helper only splits messages already classified as fault descriptions and
    only keeps sections carrying distinct fault signatures.
    """
    sections: list[tuple[str, set[str], str]] = []
    for message in episode.get("fault_description_messages") or []:
        if not isinstance(message, dict):
            continue
        text = _w7_message_text(message)
        parts = [
            part.strip(" ：:，,。；;\n")
            for part in re.split(
                r"(?:^|[\s。；;,，])(?:[一二三四五六七八九十]+|\d+)[、.．,，]",
                text,
            )
            if part.strip()
        ]
        if len(parts) < 2:
            continue
        for part in parts:
            signature = _fault_signature(part) & _W7_PRIMARY_FAULT_SIGNATURES
            if not signature:
                continue
            sections.append((part, signature, str(message.get("message_id") or message.get("source_message_id") or "")))
    return sections


def _w7_salvage_report_fault_sections(episode: dict[str, Any]) -> list[tuple[str, set[str], str]]:
    """Recover concrete fault clauses embedded in an otherwise report-only message."""
    sections: list[tuple[str, set[str], str]] = []
    seen: set[tuple[str, str]] = set()
    for message in _w7_episode_messages(episode):
        text = _w7_message_text(message)
        if not text or not any(marker in text for marker in _REPORT_ONLY_MARKERS):
            continue
        source_id = str(message.get("message_id") or message.get("source_message_id") or "")
        clauses = [
            clause.strip(" ：:，,。；;\n")
            for clause in re.split(r"[。；;\n]+", text)
            if clause.strip(" ：:，,。；;\n")
        ]
        for clause in clauses:
            signature = _fault_signature(clause) & _W7_PRIMARY_FAULT_SIGNATURES
            if not signature or not has_fault_signal(clause):
                continue
            # A report heading or generic status line is not a recoverable
            # case.  Require a concrete diagnostic/handling signal or a
            # sufficiently specific fault clause.
            if not has_action_signal(clause) and len(signature) < 2:
                continue
            key = (source_id, clause)
            if key in seen:
                continue
            seen.add(key)
            sections.append((clause, signature, source_id))
    return sections


def _w7_build_case_items(
    episode: dict[str, Any],
    *,
    multi_fault: bool,
    forced_sections: list[tuple[str, set[str], str]] | None = None,
) -> list[dict[str, Any]]:
    """Produce W2-sized case items while preserving episode-level trace data."""
    core = _w7_core_messages(episode)
    summary_context = [
        item for item in episode.get("summary_context_messages") or []
        if isinstance(item, dict)
    ]
    local_context = [
        item for item in episode.get("case_context_messages") or []
        if isinstance(item, dict)
    ]
    all_ids = list(dict.fromkeys([
        *list(episode.get("full_context_message_ids") or []),
        *list(episode.get("evidence_message_ids") or []),
        *list(episode.get("summary_context_message_ids") or []),
    ]))
    sections = list(forced_sections or [])
    if not sections and multi_fault:
        sections = _w7_numbered_fault_sections(episode)
    if not sections:
        focus = str(((episode.get("extracted") or {}).get("fault_focus_text")) or derive_fault_focus_text(episode) or "")
        sections = [(focus, _fault_signature(focus) & _W7_PRIMARY_FAULT_SIGNATURES, "")]

    # Keep repeated sections in one case when they describe the same fault.
    grouped: list[tuple[str, set[str], list[str]]] = []
    for text, signature, source_id in sections:
        target = next((row for row in grouped if row[1] & signature), None)
        if target is None:
            grouped.append((text, set(signature), [source_id] if source_id else []))
        else:
            target[1].update(signature)
            if source_id:
                target[2].append(source_id)

    items: list[dict[str, Any]] = []
    for index, (problem, signature, source_ids) in enumerate(grouped, start=1):
        matched_diag: list[dict[str, Any]] = []
        matched_outcome: list[dict[str, Any]] = []
        matched_context: list[dict[str, Any]] = []
        candidate_messages = [*core, *local_context, *summary_context]
        seen_candidates: set[str] = set()
        for message in candidate_messages:
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            if message_id and message_id in seen_candidates:
                continue
            if message_id:
                seen_candidates.add(message_id)
            text = _w7_message_text(message)
            overlap = _fault_signature(text) & signature
            if not overlap and len(grouped) > 1:
                continue
            matched_context.append(message)
            is_explicit_diagnostic = message in (episode.get("diagnostic_chain_messages") or [])
            is_context_action = message in summary_context and has_action_signal(text)
            if is_explicit_diagnostic or is_context_action:
                matched_diag.append(message)
            resolution_status = _w7_resolution_status(text)
            if message in (episode.get("resolution_messages") or []) or message in (episode.get("outcome_messages") or []) or resolution_status in {"verified", "ineffective"}:
                matched_outcome.append(message)
        matched_context_ids = [
            str(item.get("message_id") or item.get("source_message_id") or "")
            for item in matched_context
            if item.get("message_id") or item.get("source_message_id")
        ]
        if len(grouped) == 1 and not matched_context_ids:
            matched_context_ids = all_ids
        message_ids = list(dict.fromkeys([
            *source_ids,
            *[str(item.get("message_id") or item.get("source_message_id") or "") for item in matched_diag],
            *[str(item.get("message_id") or item.get("source_message_id") or "") for item in matched_outcome],
        ]))
        ready = bool(problem and message_ids and (matched_diag or matched_outcome))
        items.append({
            "case_item_id": f"{episode.get('episode_id', '')}:case:{index}",
            "problem_statement": problem,
            "fault_focus": problem,
            "fault_signatures": sorted(signature),
            "message_ids": message_ids,
            "context_message_ids": list(dict.fromkeys([*source_ids, *matched_context_ids])),
            "summary_context_message_ids": [
                message_id for message_id in matched_context_ids
                if message_id in set(episode.get("summary_context_message_ids") or [])
            ],
            "diagnostic_message_ids": [str(item.get("message_id") or item.get("source_message_id") or "") for item in matched_diag],
            "outcome_message_ids": [str(item.get("message_id") or item.get("source_message_id") or "") for item in matched_outcome],
            "trace_stage": "resolution" if matched_outcome else ("diagnostic" if matched_diag else "fault"),
            "relation_to_other_cases": "independent" if len(grouped) > 1 else "single_episode",
            "w2_ready": ready,
            "w2_block_reasons": [] if ready else ["case_item_missing_diagnostic_or_outcome"],
        })
    return items


def _w7_is_report_only(episode: dict[str, Any], fault_focus: str) -> bool:
    anchor = episode.get("field_report_anchor") or {}
    if not isinstance(anchor, dict):
        anchor = {}
    texts = [_w7_message_text(item) for item in _w7_episode_messages(episode)]
    report_hits = sum(any(marker in text for marker in _REPORT_ONLY_MARKERS) for text in texts)
    has_fault = bool(fault_focus and has_fault_signal(fault_focus))
    return bool(anchor.get("anchor_id") and not has_fault) or (report_hits > 0 and not has_fault)


def _w7_parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _w7_date_span_hours(episode: dict[str, Any]) -> float:
    start = _w7_parse_time(episode.get("start_time"))
    end = _w7_parse_time(episode.get("end_time"))
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 3600.0)


def refine_episode_for_w2(episode: dict[str, Any]) -> dict[str, Any]:
    """Apply W7's conservative episode-level gate without extracting KG semantics.

    This deliberately does not invent a family, action, or outcome.  It only
    repairs resolution status, marks report/longitudinal scope, and decides
    whether the current episode is safe to hand to W2.
    """
    out = json.loads(json.dumps(episode, ensure_ascii=False))
    extracted = out.get("extracted") if isinstance(out.get("extracted"), dict) else {}
    initial_messages = []
    initial_seen: set[str] = set()
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "noise_messages"):
        for message in out.get(key) or []:
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            if message_id and message_id in initial_seen:
                continue
            if message_id:
                initial_seen.add(message_id)
            initial_messages.append(message)
    before_resolution = list(out.get("resolution_messages") or [])
    accepted_resolution: list[dict[str, Any]] = []
    pending_or_invalid: list[dict[str, Any]] = []
    resolution_statuses: list[str] = []
    for message in before_resolution:
        text = _w7_message_text(message)
        status = _w7_resolution_status(text)
        resolution_statuses.append(status)
        if status == "verified":
            accepted_resolution.append(message)
        else:
            pending_or_invalid.append(message)
    out["resolution_messages"] = accepted_resolution

    diagnostic = list(out.get("diagnostic_chain_messages") or [])
    existing_ids = {str(item.get("message_id") or "") for item in diagnostic if isinstance(item, dict)}
    for message in pending_or_invalid:
        if not isinstance(message, dict):
            continue
        message_id = str(message.get("message_id") or "")
        if message_id and message_id not in existing_ids:
            diagnostic.append(message)
            existing_ids.add(message_id)
    out["diagnostic_chain_messages"] = diagnostic

    # W1 is intentionally coarse.  Reclassify explicit observations from any
    # message role so that recurrence/ineffective attempts and verified
    # recovery cannot remain hidden in noise or fault-only buckets.
    outcome_messages: list[dict[str, Any]] = []
    outcome_statuses: list[str] = []
    outcome_ids: set[str] = set()
    for message in initial_messages:
        status = _w7_resolution_status(_w7_message_text(message))
        if status == "unknown" or not _w7_is_outcome_statement(_w7_message_text(message)):
            continue
        message_id = str(message.get("message_id") or message.get("source_message_id") or "")
        if message_id and message_id in outcome_ids:
            continue
        if message_id:
            outcome_ids.add(message_id)
        outcome_messages.append(message)
        outcome_statuses.append(status)
        if status == "verified" and message not in accepted_resolution:
            accepted_resolution.append(message)
        if has_action_signal(_w7_message_text(message)) and message_id not in existing_ids:
            diagnostic.append(message)
            if message_id:
                existing_ids.add(message_id)
    out["outcome_messages"] = outcome_messages
    if outcome_ids:
        for key in ("fault_description_messages", "noise_messages"):
            out[key] = [
                message for message in out.get(key) or []
                if str(message.get("message_id") or message.get("source_message_id") or "") not in outcome_ids
            ]
    out["diagnostic_chain_messages"] = diagnostic

    fault_focus = derive_fault_focus_text(out)
    messages = _w7_episode_messages(out)
    core_messages = _w7_core_messages(out)
    evidence_ids = list(dict.fromkeys(
        [str(value) for value in out.get("evidence_message_ids") or [] if str(value)]
        + [str(item.get("message_id") or item.get("source_message_id") or "") for item in messages if isinstance(item, dict) and (item.get("message_id") or item.get("source_message_id"))]
    ))
    report_only = _w7_is_report_only(out, fault_focus)
    salvaged_report_sections = _w7_salvage_report_fault_sections(out) if report_only else []
    if salvaged_report_sections:
        report_only = False
        fault_focus = salvaged_report_sections[0][0]
    multi_fault = _w7_is_multi_fault_episode(out)
    has_action = any(has_action_signal(_w7_message_text(item)) for item in core_messages)
    has_fault = bool(fault_focus and has_fault_signal(fault_focus)) or bool(salvaged_report_sections) or any(has_fault_signal(_w7_message_text(item)) for item in core_messages)

    # A short confirmation such as "已解决" is usable only when the same
    # episode already contains a concrete diagnostic action.  This supports
    # cross-message action -> confirmation without accepting isolated claims.
    if has_action:
        for message in list(pending_or_invalid):
            text = _w7_message_text(message)
            if (
                any(marker in text for marker in _RESOLUTION_POSITIVE_MARKERS)
                and not _w7_is_question_or_pending(text)
                and not any(marker in text for marker in _INEFFECTIVE_MARKERS)
            ):
                accepted_resolution.append(message)
                pending_or_invalid.remove(message)

    embedded_statuses = [_w7_resolution_status(_w7_message_text(item)) for item in core_messages]
    for message, status in zip(core_messages, embedded_statuses):
        if status == "verified" and message not in accepted_resolution:
            accepted_resolution.append(message)
    out["resolution_messages"] = accepted_resolution

    if report_only:
        scope = "report_only"
    elif multi_fault:
        scope = "multi_fault"
    elif _w7_date_span_hours(out) > 24 * 7:
        scope = "longitudinal_trace"
    else:
        scope = "single_fault"
    block_reasons: list[str] = []
    if report_only:
        block_reasons.append("report_only_or_coordination_context")
    if not has_fault:
        block_reasons.append("missing_fault_signal")
    if not evidence_ids:
        block_reasons.append("missing_message_evidence")
    if not has_action and not accepted_resolution:
        block_reasons.append("no_diagnostic_action_or_verified_outcome")
    if str(out.get("completeness") or "") == "noise":
        block_reasons.append("w1_marked_noise")

    case_items = _w7_build_case_items(
        out,
        multi_fault=multi_fault,
        forced_sections=salvaged_report_sections,
    )
    inherited_case_blocks: list[str] = []
    if report_only:
        inherited_case_blocks.append("parent_episode_report_only")
    if not has_fault:
        inherited_case_blocks.append("parent_episode_missing_fault_signal")
    if str(out.get("completeness") or "") == "noise":
        inherited_case_blocks.append("parent_episode_w1_noise")
    if inherited_case_blocks:
        for case_item in case_items:
            case_item["w2_ready"] = False
            case_item["w2_block_reasons"] = list(dict.fromkeys([
                *list(case_item.get("w2_block_reasons") or []),
                *inherited_case_blocks,
            ]))
    ready_case_count = sum(bool(item.get("w2_ready")) for item in case_items)
    if multi_fault and len(case_items) >= 2:
        # Keep the legacy episode gate conservative for callers that still
        # submit one episode at a time.  W2 consumers should iterate the
        # explicit case_items instead of treating the whole report as one KG
        # candidate.
        block_reasons.append("multi_fault_requires_case_item_iteration")
    out["case_items"] = case_items
    out["case_item_count"] = len(case_items)
    out["case_items_w2_ready_count"] = ready_case_count

    continuation = scope == "longitudinal_trace"
    action_outcome_state = _w7_action_outcome_state(out)
    final_outcome_type = str(action_outcome_state.get("final_outcome_type") or "pending_validation")
    if final_outcome_type == "verified_fix" and not has_fault:
        final_outcome_type = "pending_validation"
        action_outcome_state["final_outcome_type"] = final_outcome_type
        action_outcome_state["verified_fix_requirements"]["fault_identity_present"] = False
        action_outcome_state["downgrade_reason"] = "missing_fault_identity"
    legacy_embedded_verified = bool(
        final_outcome_type == "mitigation_observed"
        and any(_w7_resolution_status(_w7_message_text(item)) == "verified" for item in _w7_core_messages(out))
    )
    if final_outcome_type != "verified_fix" and not legacy_embedded_verified:
        # Historical/temporary recovery remains available in outcome_events,
        # but must not leak into the current verified resolution channel.
        accepted_resolution = []
        out["resolution_messages"] = []
    legacy_resolution_status = (
        "verified" if final_outcome_type == "verified_fix" else
        "ineffective" if final_outcome_type == "ineffective" else
        "verified" if legacy_embedded_verified else
        "pending" if final_outcome_type in {"partial_temporary", "mitigation_observed", "pending_validation"} else
        "unknown"
    )
    extracted["w7_episode_cleanup"] = {
        "agent_id": "W7",
        "schema_version": "w7.episode_cleanup.v2",
        "resolution_status": legacy_resolution_status,
        "outcome_type": final_outcome_type,
        "action_outcome_state": action_outcome_state,
        "legacy_resolution_compatibility": legacy_embedded_verified,
        "resolution_statuses_by_message": resolution_statuses,
        "embedded_resolution_statuses": embedded_statuses,
        "outcome_statuses_by_message": outcome_statuses,
        "rejected_resolution_message_ids": [str(item.get("message_id") or "") for item in pending_or_invalid if isinstance(item, dict)],
        "episode_scope": scope,
        "continuation": continuation,
        "continuation_reason": "episode_duration_over_7_days" if continuation else "",
        "w2_ready": not block_reasons,
        "w2_block_reasons": block_reasons,
        "evidence_message_ids": evidence_ids,
        "case_item_count": len(case_items),
        "case_items_w2_ready_count": ready_case_count,
        "case_item_policy": "numbered_fault_sections_only.v1",
        "report_case_salvaged": bool(salvaged_report_sections),
        "salvaged_report_case_count": len(salvaged_report_sections),
    }
    extracted["fault_focus_text"] = fault_focus
    extracted["fault_focus_confidence"] = derive_fault_focus_confidence(out)
    out["extracted"] = extracted
    out["w2_ready"] = not block_reasons
    out["w2_block_reasons"] = block_reasons
    out["episode_scope"] = scope
    out["continuation"] = continuation
    out["continuation_reason"] = "episode_duration_over_7_days" if continuation else ""
    out["evidence_message_ids"] = evidence_ids
    out["w7_resolution_messages_rejected"] = pending_or_invalid
    return out


def _w7_trace_signature(episode: dict[str, Any]) -> set[str]:
    texts = [
        str(((episode.get("extracted") or {}).get("fault_focus_text")) or ""),
        *[_w7_message_text(item) for item in _w7_core_messages(episode)],
    ]
    return set().union(*(_fault_signature(text) for text in texts if text)) if any(texts) else set()


def _w7_episode_source_message_ids(episode: dict[str, Any]) -> set[str]:
    return {
        str(item.get("source_message_id") or item.get("message_id") or "")
        for item in _w7_core_messages(episode)
        if item.get("source_message_id") or item.get("message_id")
    }


def _w7_trace_identity(episode: dict[str, Any]) -> dict[str, set[str]]:
    messages = _w7_episode_messages(episode)
    texts = [_w7_message_text(item) for item in messages]
    joined = "\n".join(texts)
    chat_ids = {_chat_id(episode)} - {""}
    artifact_payload = attachment_identity_keys(episode)
    for message in messages:
        artifact_payload.update(attachment_identity_keys(message))
    return {
        "chat": chat_ids,
        "equipment": {
            match.group(0).upper().replace(" ", "") for match in _W7_EQUIPMENT_RE.finditer(joined)
            if not _W7_JIRA_RE.fullmatch(match.group(0))
        },
        "line": {match.group(1) for match in _W7_LINE_RE.finditer(joined)},
        "jira": {
            match.upper() for match in _W7_JIRA_RE.findall(joined)
        },
        "artifact": {match.lower() for match in _W7_ARTIFACT_RE.findall(joined)},
        "artifact_payload": artifact_payload,
        "signature": _w7_trace_signature(episode),
        "recurrence": {marker for marker in _W7_RECURRENCE_MARKERS if marker in joined},
    }


def _w7_trace_link_candidate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Score one longitudinal trace candidate with explicit cannot-links."""
    reasons: list[str] = []
    cannot_links: list[str] = []
    if left.get("episode_scope") in {"report_only", "multi_fault"} or right.get("episode_scope") in {"report_only", "multi_fault"}:
        return {
            "linked": False,
            "relation_type": "unrelated",
            "link_strength": "none",
            "reasons": [],
            "cannot_link_reasons": ["non_atomic_episode_scope"],
        }
    left_sources = _w7_episode_source_message_ids(left)
    right_sources = _w7_episode_source_message_ids(right)
    shared_sources = left_sources & right_sources
    left_identity, right_identity = _w7_trace_identity(left), _w7_trace_identity(right)
    left_focus_signature = _fault_signature(str(((left.get("extracted") or {}).get("fault_focus_text")) or ""))
    right_focus_signature = _fault_signature(str(((right.get("extracted") or {}).get("fault_focus_text")) or ""))
    shared_focus_signature = left_focus_signature & right_focus_signature
    shared_distinctive_focus = shared_focus_signature & _W7_TRACE_DISTINCTIVE_SIGNATURES
    exact_focus_signature = bool(
        shared_focus_signature
        and left_focus_signature == right_focus_signature
    )
    focus_compatible = bool(shared_distinctive_focus or exact_focus_signature)
    if left_identity["chat"] and right_identity["chat"] and left_identity["chat"].isdisjoint(right_identity["chat"]):
        cannot_links.append("different_chat")
    shared_jira = left_identity["jira"] & right_identity["jira"]
    shared_artifact = left_identity["artifact"] & right_identity["artifact"]
    shared_artifact_payload = (
        left_identity["artifact_payload"]
        & right_identity["artifact_payload"]
    )
    shared_equipment = left_identity["equipment"] & right_identity["equipment"]
    shared_signature = left_identity["signature"] & right_identity["signature"]
    shared_distinctive = shared_signature & _W7_TRACE_DISTINCTIVE_SIGNATURES
    strong_identity = bool(shared_jira or shared_artifact or shared_artifact_payload)
    recurrence = bool(right_identity["recurrence"])
    if (
        left_identity["equipment"] and right_identity["equipment"]
        and left_identity["equipment"].isdisjoint(right_identity["equipment"])
        and not strong_identity and not (recurrence and shared_distinctive)
    ):
        cannot_links.append("conflicting_equipment_identity")
    if (
        left_identity["line"] and right_identity["line"]
        and left_identity["line"].isdisjoint(right_identity["line"])
        and not strong_identity and not (recurrence and shared_distinctive)
    ):
        cannot_links.append("conflicting_line_identity")
    if cannot_links:
        return {
            "linked": False,
            "relation_type": "unrelated",
            "link_strength": "none",
            "reasons": [],
            "cannot_link_reasons": cannot_links,
        }
    if shared_jira:
        reasons.extend(["shared_jira", *sorted(shared_jira)])
    if shared_artifact:
        reasons.extend(["shared_artifact", *sorted(shared_artifact)])
    if shared_artifact_payload:
        reasons.extend(["shared_artifact_payload", *sorted(shared_artifact_payload)])
    if shared_equipment:
        reasons.extend(["shared_equipment", *sorted(shared_equipment)])
    if shared_distinctive:
        reasons.extend(["shared_distinctive_fault_signature", *sorted(shared_distinctive)])
    if shared_sources and focus_compatible:
        reasons.append("shared_source_message_with_compatible_fault_focus")
    if shared_distinctive_focus:
        reasons.extend(["shared_distinctive_fault_focus", *sorted(shared_distinctive_focus)])
    same_session = bool(
        str(left.get("thread_id") or left.get("source_thread_id") or "")
        == str(right.get("thread_id") or right.get("source_thread_id") or "")
    )
    exact_compact_signature = bool(
        shared_signature
        and left_identity["signature"] == right_identity["signature"]
        and len(shared_signature) <= 2
    )
    exact_compact_focus = bool(
        exact_focus_signature
        and len(left_focus_signature) <= 2
    )
    left_focus_terms = _case_terms(
        str(((left.get("extracted") or {}).get("fault_focus_text")) or "")
    )
    right_focus_terms = _case_terms(
        str(((right.get("extracted") or {}).get("fault_focus_text")) or "")
    )
    shared_focus_terms = left_focus_terms & right_focus_terms
    focus_term_union = left_focus_terms | right_focus_terms
    focus_term_similarity = (
        len(shared_focus_terms) / len(focus_term_union)
        if focus_term_union else 0.0
    )
    left_end = _w7_parse_time(left.get("end_time"))
    right_start = _w7_parse_time(right.get("start_time"))
    gap_hours = (
        max(0.0, (right_start - left_end).total_seconds() / 3600.0)
        if left_end is not None and right_start is not None
        else None
    )
    # Taxonomy signatures such as “误报” or “漏检” are common within a busy
    # support session.  They can authorize a merge only when the full fault
    # focus also overlaps materially and the phases are temporally local.
    same_session_exact_focus_support = bool(
        same_session
        and exact_compact_focus
        and gap_hours is not None
        and gap_hours <= 24.0
        and len(shared_focus_terms) >= 3
        and focus_term_similarity >= 0.10
    )
    if same_session_exact_focus_support:
        reasons.append("same_session_exact_focus_with_text_overlap")
    linked = bool(
        (shared_sources and focus_compatible)
        or (strong_identity and (shared_focus_signature or shared_signature))
        or (shared_equipment and (shared_distinctive_focus or shared_distinctive))
        or (recurrence and (shared_distinctive_focus or shared_distinctive))
        or same_session_exact_focus_support
        or (same_session and (shared_distinctive_focus or exact_focus_signature or exact_compact_signature))
    )
    if recurrence and linked:
        relation_type = "recurrence_of"
        reasons.append("recurrence_language")
    else:
        right_outcome = str(((right.get("extracted") or {}).get("w7_episode_cleanup") or {}).get("outcome_type") or "")
        relation_type = "validation_of" if linked and right_outcome == "verified_fix" else ("continuation_of" if linked else "uncertain")
    if not linked:
        link_strength = "none"
    elif shared_sources and focus_compatible:
        link_strength = "hard"
    elif strong_identity and (shared_focus_signature or shared_signature):
        link_strength = "strong"
    elif (
        shared_equipment and (shared_distinctive_focus or shared_distinctive)
    ) or (
        recurrence and (shared_distinctive_focus or shared_distinctive)
    ) or (
        same_session_exact_focus_support
    ):
        link_strength = "medium"
    else:
        # Same-session lexical continuity is useful navigation context, but is
        # deliberately too weak to authorize evidence or outcome sharing.
        link_strength = "weak"
    return {
        "linked": linked,
        "relation_type": relation_type,
        "link_strength": link_strength,
        "reasons": list(dict.fromkeys(reasons)),
        "cannot_link_reasons": [],
    }


def _w7_should_link_same_trace(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, list[str]]:
    """Conservatively link phases without merging their outcomes."""
    candidate = _w7_trace_link_candidate(left, right)
    reasons = [*candidate["reasons"], *(f"cannot_link:{value}" for value in candidate["cannot_link_reasons"])]
    return bool(candidate["linked"]), reasons


def refine_episode_group(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refine one W1 relation-aware session while preserving trace membership.

    W7 does not blindly concatenate all same-root messages.  It emits a stable
    trace group and leaves explicit phase/item boundaries for later W2 handling.
    """
    refined = [refine_episode_for_w2(item) for item in episodes]
    if not refined:
        return refined
    thread_ids = sorted({str(item.get("thread_id") or item.get("source_thread_id") or "") for item in refined})
    parent = list(range(len(refined)))
    link_reasons: dict[tuple[int, int], list[str]] = {}
    link_types: dict[tuple[int, int], str] = {}
    link_strengths: dict[tuple[int, int], str] = {}
    candidate_audit: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for right_index in range(1, len(refined)):
        right = refined[right_index]
        for left_index in range(right_index - 1, -1, -1):
            candidate = _w7_trace_link_candidate(refined[left_index], right)
            candidate_audit[right_index].append({
                "candidate_episode_id": str(refined[left_index].get("episode_id") or ""),
                **candidate,
            })
            if candidate["linked"] and candidate["link_strength"] in {"hard", "strong", "medium"}:
                union(left_index, right_index)
                link_reasons[(left_index, right_index)] = list(candidate["reasons"])
                link_types[(left_index, right_index)] = str(candidate["relation_type"])
                link_strengths[(left_index, right_index)] = str(candidate["link_strength"])
                break

    components: dict[int, list[int]] = {}
    for index in range(len(refined)):
        components.setdefault(find(index), []).append(index)
    for component_indexes in components.values():
        episode_ids = [str(refined[index].get("episode_id") or "") for index in component_indexes]
        trace_seed = f"{'|'.join(thread_ids)}|{'|'.join(episode_ids)}"
        trace_group_id = f"w7-trace:{hashlib.sha1(trace_seed.encode('utf-8')).hexdigest()[:16]}"
        phase_count = len(component_indexes)
        for phase_index, item_index in enumerate(component_indexes, start=1):
            item = refined[item_index]
            previous_index = component_indexes[phase_index - 2] if phase_index > 1 else None
            reasons = link_reasons.get((previous_index, item_index), []) if previous_index is not None else []
            item["trace_group_id"] = trace_group_id
            item["trace_phase_index"] = phase_index
            item["trace_phase_count"] = phase_count
            item["trace_relation"] = "trace_root" if phase_index == 1 else "same_trace"
            item["trace_relation_type"] = "trace_root" if phase_index == 1 else link_types.get((previous_index, item_index), "continuation_of")
            item["trace_link_strength"] = "root" if phase_index == 1 else link_strengths.get((previous_index, item_index), "weak")
            item["trace_link_reasons"] = reasons
            item["trace_link_candidates"] = candidate_audit.get(item_index, [])
            item["previous_trace_episode_id"] = (
                str(refined[previous_index].get("episode_id") or "") if previous_index is not None else ""
            )
            cleanup = (item.get("extracted") or {}).get("w7_episode_cleanup") or {}
            cleanup["trace_group_id"] = trace_group_id
            cleanup["trace_phase_index"] = phase_index
            cleanup["trace_phase_count"] = phase_count
            cleanup["trace_relation"] = item["trace_relation"]
            cleanup["trace_relation_type"] = item["trace_relation_type"]
            cleanup["trace_link_strength"] = item["trace_link_strength"]
            cleanup["trace_link_reasons"] = reasons
            cleanup["trace_link_candidates"] = item["trace_link_candidates"]
            item.setdefault("extracted", {})["w7_episode_cleanup"] = cleanup
    return refined


class ReviewContextAgent:
    """W7: sanitize W1 attribution and inject review/SOP context before W2."""

    agent_id = "W7"

    def __init__(
        self,
        *,
        text_history_root: str | Path = DEFAULT_TEXT_HISTORY_ROOT,
        jira_offline_root: str | Path = DEFAULT_JIRA_OFFLINE_ROOT,
        role_registry_path: str | Path = DEFAULT_ROLE_REGISTRY,
    ) -> None:
        self.text_history_root = str(text_history_root)
        self.jira_offline_root = str(jira_offline_root)
        self.role_registry_path = str(role_registry_path)

    def prepare_episode(self, episode: dict[str, Any], background: dict[str, Any], *, review_case_id: str = "") -> dict[str, Any]:
        cleaned = refine_episode_for_w2(episode)
        promoted = promote_case_evidence(
            cleaned,
            text_history_root=self.text_history_root,
            jira_offline_root=self.jira_offline_root,
        )
        return inject_review_context(
            promoted,
            background,
            review_case_id=review_case_id,
            role_registry=load_people_role_registry(self.role_registry_path),
        )


def inject_review_context(
    episode: dict[str, Any],
    background: dict[str, Any],
    *,
    review_case_id: str = "",
    role_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = json.loads(json.dumps(episode, ensure_ascii=False))
    extracted = out.get("extracted") if isinstance(out.get("extracted"), dict) else {}
    # ``sop_background`` predates non-SOP incremental ingestion.  The active
    # pipeline now injects an alignment-only KG slice plus reviewed examples,
    # so the neutral name is the contract.  Keep the old key as a read-only
    # compatibility alias while W2 callers migrate.
    review_context = dict(background) if isinstance(background, dict) else {"value": background}
    review_context.setdefault("schema_version", "kg_v2.review_context.v1")
    review_context.setdefault("context_role", "alignment_only")
    review_context.setdefault("facts_may_not_be_copied_as_new_evidence", True)
    extracted["review_context"] = review_context
    extracted["sop_background"] = review_context
    extracted["sop_background_compatibility_alias"] = True
    extracted["fault_focus_text"] = derive_fault_focus_text(out)
    extracted["fault_focus_confidence"] = derive_fault_focus_confidence(out)
    attribution_raw, attribution = sanitize_attribution(out)
    extracted["attribution_raw"] = attribution_raw
    attribution["role_assignments"] = resolve_people_roles(out, attribution, role_registry or load_people_role_registry())
    attribution["role_registry_version"] = str((role_registry or load_people_role_registry()).get("schema_version") or "")
    extracted["attribution"] = attribution
    if review_case_id:
        extracted["review_case_id"] = review_case_id
    out["extracted"] = extracted
    return out


def build_non_sop_alignment_background(envelope: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from debug_agent_system.agents.write.non_sop_intake import build_alignment_only_background

    return build_alignment_only_background(envelope, **kwargs)


def gold_review_fallback_episode(case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    episode = payload.get("episode_input") if isinstance(payload.get("episode_input"), dict) else {}
    if not episode:
        return {}
    out = json.loads(json.dumps(episode, ensure_ascii=False))
    out["episode_id"] = str(payload.get("source_episode_id") or out.get("episode_id") or case_id)
    out["thread_id"] = str(out.get("thread_id") or payload.get("source_thread_id") or f"gold-thread:{case_id}")
    out["completeness"] = str(out.get("completeness") or "partial")
    out["case_context_messages"] = [
        {
            "message_id": f"goldctx:{idx+1}",
            "create_time": "",
            "text": str(text),
        }
        for idx, text in enumerate(payload.get("source_excerpt") or [])
        if str(text).strip()
    ]
    extracted = out.get("extracted") if isinstance(out.get("extracted"), dict) else {}
    if payload.get("source_excerpt"):
        extracted["source_excerpt"] = payload.get("source_excerpt")
    out["extracted"] = extracted
    return out


def manual_review_fallback_episode(case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    refined = payload.get("refined_merge_proposal") if isinstance(payload.get("refined_merge_proposal"), dict) else {}
    nodes = [item for item in refined.get("nodes") or [] if isinstance(item, dict)]
    error = next((item for item in nodes if item.get("type") == "Error"), {})
    checks = [item for item in nodes if item.get("type") == "DiagnosticCheck"]
    solutions = [item for item in nodes if item.get("type") == "Solution"]
    evidence = [item for item in payload.get("evidence_findings") or [] if isinstance(item, dict)]
    if not error and not checks and not solutions and not evidence:
        return {}
    fault_messages = []
    if error:
        fault_messages.append({
            "message_id": "manual:error",
            "create_time": payload.get("reviewed_at") or "",
            "text": str(error.get("symptom") or error.get("label") or payload.get("review_summary") or ""),
        })
    diagnostic_messages = []
    for idx, check in enumerate(checks, start=1):
        diagnostic_messages.append({
            "message_id": f"manual:check:{idx}",
            "create_time": payload.get("reviewed_at") or "",
            "text": str(check.get("how_to_check") or check.get("label") or ""),
        })
    for idx, finding in enumerate(evidence, start=1):
        diagnostic_messages.append({
            "message_id": str(finding.get("message_id") or f"manual:evidence:{idx}"),
            "create_time": str(finding.get("time") or payload.get("reviewed_at") or ""),
            "text": str(finding.get("summary") or finding.get("finding") or ""),
        })
    resolution_messages = []
    for idx, solution in enumerate(solutions, start=1):
        resolution_messages.append({
            "message_id": f"manual:solution:{idx}",
            "create_time": payload.get("reviewed_at") or "",
            "text": str(solution.get("content") or ""),
        })
    if payload.get("review_summary"):
        resolution_messages.append({
            "message_id": "manual:review_summary",
            "create_time": payload.get("reviewed_at") or "",
            "text": str(payload.get("review_summary") or ""),
        })
    evidence_ids = [str(item.get("message_id") or "") for item in evidence if item.get("message_id")]
    return {
        "episode_id": str(payload.get("source_episode_id") or case_id),
        "thread_id": payload.get("source_thread_id") or "",
        "completeness": "manual_fallback",
        "fault_description_messages": fault_messages[:8],
        "diagnostic_chain_messages": diagnostic_messages[:24],
        "resolution_messages": resolution_messages[:16],
        "noise_messages": [],
        "evidence_message_ids": evidence_ids[:32],
        "source_offsets": [],
        "attachments": [],
        "case_context_messages": [],
        "extracted": {
            "symptom_raw": str(error.get("symptom") or error.get("label") or ""),
            "debug_actions": [str(item.get("label") or "") for item in checks[:12]],
            "conclusion": str(payload.get("review_summary") or ""),
            "required_info": list(error.get("required_info") or []),
            "source_kind": "manual_review_fallback",
        },
    }
