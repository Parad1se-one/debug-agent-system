"""W2 extraction from W1 fault episodes into schema-valid KG candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import threading
import urllib.error
import urllib.request
from typing import Any

from debug_agent_system.knowledge.store import KGStore
from debug_agent_system.knowledge.schema_validator import validate_nodes_edges, semantic_schema_issues
from debug_agent_system.knowledge_v2 import (
    build_candidate_draft_v2_from_case_understanding,
    build_case_understanding_card_from_semantics,
    build_v2_bundle_from_candidate_draft,
)
from .case_understanding_prompt import (
    PROMPT_VERSION as CASE_UNDERSTANDING_PROMPT_VERSION,
    SYSTEM_PROMPT as CASE_UNDERSTANDING_SYSTEM_PROMPT,
    build_prompt_input as build_case_understanding_prompt_input,
    normalize_card as normalize_prompt_case_understanding_card,
    tool_schema as case_understanding_tool_schema,
)

CATEGORY_RULES = (
    ("硬件与运控", ("电机", "运控", "轴", "相机", "光源", "拍照", "工控机", "内存", "板卡", "IO", "传感器", "IP", "网线")),
    ("系统与软件异常", ("软件", "程序", "卡死", "闪退", "初始化", "数据库", "服务", "登录", "系统", "崩溃", "报错", "启动")),
    ("算法与程序调优", ("漏检", "误报", "阈值", "算法", "框", "识别", "OCR", "BGA", "焊", "极性", "检测", "模型", "调试")),
)

PRIMARY_KEYS = {
    "Error": "error_id",
    "DiagnosticCheck": "check_id",
    "Solution": "solution_id",
    "Site": "site_id",
    "SoftwareVersion": "version_id",
    "DiagnosticTrace": "trace_id",
    "DiagnosticOutcome": "outcome_id",
    "DiagnosticPolicy": "policy_id",
}

REQUIRED_INFO_SLOTS = {
    "log_package",
    "software_version",
    "error_phase",
    "error_message",
    "device_model",
    "site",
    "ip_config",
    "repro_steps",
    "sample_image",
    "program_file",
    "environment",
    "owner_context",
    "other",
}

SLOT_KEYWORDS = (
    ("log_package", ("DLOG", "dlog", "诊断数据", "日志", "数据包", "转存储", "转储", "log", "startup", "init", "dmp", "DMP", "dump", "evtx")),
    ("software_version", ("软件版本", "主程序版本", "算法包版本", "版本", "驱动", "固件")),
    ("error_phase", ("阶段", "初始化", "开机", "启动", "扫码", "检测", "复判")),
    ("error_message", ("报错", "错误码", "弹窗", "截图", "异常信息")),
    ("device_model", ("型号", "相机", "光源", "控制器", "工控机", "设备")),
    ("site", ("现场", "客户", "线体", "设备编号", "项目")),
    ("ip_config", ("IP", "ip", "网段", "防火墙", "端口", "ping", "Ping", "网络")),
    ("repro_steps", ("复现", "必现", "偶发", "怎么操作", "操作步骤", "复现步骤")),
    ("sample_image", ("原图", "缺陷图", "误检图", "漏检图", "图片", "样本")),
    ("program_file", ("程序", "模板", "配方", "板型", "工程文件")),
    ("environment", ("电源", "温度", "磁盘", "内存", "系统环境", "环境")),
    ("owner_context", ("负责人", "找谁", "责任归属", "归属模块", "归属")),
)

SLOT_LABELS = {
    "log_package": "诊断数据包/日志",
    "software_version": "软件版本",
    "error_phase": "故障发生阶段",
    "error_message": "完整报错信息",
    "device_model": "设备型号",
    "site": "现场/客户信息",
    "ip_config": "IP/网络配置",
    "repro_steps": "复现步骤",
    "sample_image": "样本/截图",
    "program_file": "程序/配方文件",
    "environment": "运行环境",
    "owner_context": "责任归属上下文",
    "other": "补充信息",
}

QUESTION_TEMPLATES = {
    "log_package": "请提供该故障对应的诊断数据包或日志。",
    "software_version": "请提供主程序、算法包或相关软件版本。",
    "error_phase": "请说明故障发生在启动、初始化、扫码、检测还是复判阶段。",
    "error_message": "请提供完整报错文本或报错截图。",
    "device_model": "请提供相关设备型号和硬件对象。",
    "site": "请提供现场、客户、线体或设备编号信息。",
    "ip_config": "请提供相关相机/控制器/工控机 IP、网段和网络连通性信息。",
    "repro_steps": "请补充复现步骤、频率以及是否必现。",
    "sample_image": "请提供对应原图、样本图或缺陷截图。",
    "program_file": "请提供对应程序、模板、配方或板型文件。",
    "environment": "请补充系统环境、电源、磁盘、内存或运行环境信息。",
    "owner_context": "请补充当前处理人、责任模块或已确认的归属信息。",
    "other": "请补充现场排查所需信息。",
}

WHY_TEMPLATES = {
    "log_package": "日志能定位故障阶段、错误码和底层模块，避免只按现象猜测。",
    "software_version": "版本信息用于判断是否命中已知缺陷、兼容性问题或升级/回退路径。",
    "error_phase": "发生阶段能缩小诊断分支，区分启动、初始化、检测和复判链路。",
    "error_message": "完整报错可直接匹配 KG 中的 Error/LogPattern，减少误召回。",
    "device_model": "设备型号决定相机、光源、控制器、工控机等检查路径。",
    "site": "现场信息用于约束版本、设备配置和历史复发上下文。",
    "ip_config": "网络配置能验证相机/控制器连接异常是否由 IP、网段或端口导致。",
    "repro_steps": "复现方式能区分偶发环境问题、稳定软件缺陷和操作路径问题。",
    "sample_image": "样本图能支持算法/误检/漏检类问题的定位。",
    "program_file": "程序或配方文件能定位模板、板型和参数差异。",
    "environment": "运行环境能排除电源、磁盘、内存、系统等非业务配置问题。",
    "owner_context": "责任归属上下文用于把无法自助诊断的 case 升级给正确负责人。",
    "other": "该信息可能补充现场上下文，但诊断用途需要人工确认。",
}

SLOT_PROVIDED_ROLE_MAP = {
    "log_package": {
        "log_package",
        "log_package_manifest",
        "log_text_hints",
        "log_phase_hints",
        "log_manifest_has_dmp",
        "log_manifest_has_evtx",
        "log_manifest_has_startup_log",
        "log_manifest_has_dlog",
        "dmp_metadata",
        "dmp_bugcheck_hints",
        "attachment_error_hints",
        "attachment_phase_hints",
    },
    "software_version": {"software_version", "attachment_text_preview", "document_text_preview", "proj_parsed", "jira_issue_key"},
    "error_phase": {"log_phase_hints", "attachment_phase_hints", "log_manifest_has_startup_log", "sample_image_metadata", "document_text_preview"},
    "error_message": {"log_text_hints", "dmp_bugcheck_hints", "attachment_error_hints", "sample_image_metadata", "image_dimensions", "document_text_preview"},
    "device_model": {"project_name", "proj_manifest", "document_text_preview", "attachment_text_preview", "proj_parsed"},
    "site": {"jira_issue_key", "jira_link", "document_text_preview", "attachment_text_preview"},
    "ip_config": {"ip_config", "proj_parsed", "attachment_text_preview", "document_text_preview"},
    "repro_steps": {"document_text_preview", "attachment_text_preview", "jira_issue_key"},
    "sample_image": {"sample_image", "sample_image_metadata", "image_dimensions"},
    "program_file": {"program_file", "proj_parsed", "proj_manifest", "project_name", "proj_component_table", "proj_board_images"},
    "environment": {
        "environment",
        "document_metadata",
        "document_text_preview",
        "attachment_text_preview",
        "software_version",
        "ip_config",
        "log_text_hints",
    },
    "owner_context": {"jira_link", "jira_issue_key"},
}


def _summary_extracted(item: dict[str, Any]) -> dict[str, Any]:
    extracted = item.get("extracted")
    return extracted if isinstance(extracted, dict) else item


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x or "").strip()]
    if value:
        return [str(value)]
    return []


def _one_line(value: Any, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


_NOISE_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_NOISE_BRACKET_RE = re.compile(r"\[(?:Image|Media|File|Video):[^\]]*\]", re.IGNORECASE)
_NOISE_BARE_ATTACHMENT_RE = re.compile(r"\b(?:Image|Media|File|Video):\s*[\w.\-]+", re.IGNORECASE)
_NOISE_RESOURCE_ID_RE = re.compile(r"\b(?:img|image|media|file|video)_v?[A-Za-z0-9][A-Za-z0-9_.-]{8,}\b", re.IGNORECASE)
_NOISE_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NOISE_MENTION_RE = re.compile(r"@[A-Za-z0-9_\-\u4e00-\u9fff]+")
_NOISE_JIRA_TITLE_RE = re.compile(r"\bJira\b", re.IGNORECASE)
_LOG_LINE_LABEL_RE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d{3}|"
    r"\b(?:DEBUG|INFO|WARN|ERROR|TRACE)\b|"
    r"\b[A-Za-z_][\w.]+\.(?:cpp|h|hpp|py|cs|cc):\d+\b|"
    r"\b(?:timeUsage|runInspect|ResourceController)\b)",
    re.IGNORECASE,
)
_WEAK_ACTION_LABEL_RE = re.compile(
    r"^(?:你|先|再|可以)?(?:重启|重启下|重启一下|试试|看看|看下|看一下|发我|发一下|发下|传一下|给一下)(?:吧|下|一下)?$"
)
_DANGLING_ACTION_TAIL_RE = re.compile(r"(?:后|之后|然后|再然后)\s*$")
_DIAGNOSTIC_FINDING_PREFIX_RE = re.compile(
    r"^(?:初步)?(?:排查|分析|检查|确认|判断|发现|定位)(?:结果|结论)?(?:发现|确认|是|为|到|出)[：:\s]*"
)
_ACTION_RESULT_MARKERS = (
    "正常", "恢复", "解决", "无效", "失败", "无法", "不能", "未能", "仍", "依然", "异常",
    "未再", "未出现", "没出现", "不再", "复发", "消失", "短时", "短期", "可用", "没问题", "无问题",
    "不拍", "卡死", "花屏", "蓝屏", "报错", "退出", "闪退",
)
_ACTION_RESULT_SEPARATORS = ("完成后", "之后", "，然后", ",然后", "然后", "后")


def _clean_chat_text(value: Any, limit: int = 500) -> str:
    text = str(value or "")
    text = _NOISE_MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)
    text = _NOISE_BRACKET_RE.sub(" ", text)
    text = _NOISE_BARE_ATTACHMENT_RE.sub(" ", text)
    text = _NOISE_RESOURCE_ID_RE.sub(" ", text)
    text = _NOISE_URL_RE.sub(" ", text)
    text = _NOISE_MENTION_RE.sub(" ", text)
    text = _NOISE_JIRA_TITLE_RE.sub(" ", text)
    text = text.replace("[[", " ").replace("]]", " ").replace("[", " ").replace("]", " ")
    tokens: list[str] = []
    for token in text.split():
        if tokens and token == tokens[-1]:
            continue
        tokens.append(token)
    joined = " ".join(tokens)
    previous = None
    while previous != joined:
        previous = joined
        joined = re.sub(r"([\u4e00-\u9fffA-Za-z0-9_-]{2,24})\s+\1(?=$|\s|[，,。；;:：])", r"\1", joined)
    return _one_line(joined, limit).strip(" -_，,。；;:：()（）")


def _clean_list(values: list[str], *, limit: int = 500) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean_chat_text(value, limit)
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


_FAULT_SENTENCE_KEYWORDS = (
    "报错", "异常", "失败", "闪退", "卡死", "黑屏", "蓝屏", "重启", "不亮", "无法", "不能", "连接异常", "应用异常",
    "拍照失败", "拍摄失败", "拍摄延迟", "拍照延迟", "延迟", "卡顿", "残帧", "丢包", "马赛克",
    "漏检", "误报", "初始化", "显示器", "断电", "关机",
)
_FAULT_OBJECT_KEYWORDS = (
    "显卡", "内存", "主板", "硬盘", "磁盘", "网卡", "相机", "光源", "控制器", "运控", "采集卡",
    "工控机", "电脑", "主机", "CPU", "cpu", "GPU", "gpu", "驱动", "电源", "供电", "软件", "主程序",
)
_FAULT_STATE_KEYWORDS = (
    "有问题", "问题", "损坏", "坏了", "坏", "异常", "报错", "失败", "找不到", "丢失", "不亮", "不稳定",
    "卡死", "重启", "蓝屏", "黑屏", "闪退", "打不开", "启动不了", "无法", "不能",
)
_ACTION_SENTENCE_KEYWORDS = (
    "检查", "确认", "判断", "排查", "导出", "提供", "上传", "重启", "设置", "替换", "更换", "拔插", "安装", "升级", "回退", "收集", "运行", "修复", "更新", "删除", "卸载", "清理", "调整", "开启",
)
_CONCRETE_ACTION_KEYWORDS = (
    "检查", "确认", "判断", "排查", "导出", "提供", "上传", "重启", "设置", "替换", "更换", "拔插", "安装", "升级", "回退", "收集", "运行", "修复", "更新", "删除", "卸载", "清理", "调整",
    "打开", "开启", "关闭", "截图", "抓取", "抓", "复现", "测试", "搜索", "双击", "点击", "查看", "事件查看器",
)
_STRUCTURED_ACTION_KEYWORDS = (
    # These concise imperative verbs are valid in W1's structured action
    # projection, but are too broad for arbitrary chat prose (for example,
    # "使用配置后能正常进入软件" is an outcome, not another action).
    "使用", "验证", "核对", "插回", "观察", "分析", "拔除", "规范关机", "清除", "改为", "查询",
    "重装", "执行", "进入PE", "执行系统", "切换", "还原", "记录故障", "重插", "连接外设", "点胶",
)
_HANDOFF_ONLY_KEYWORDS = ("麻烦", "辛苦", "幸苦", "帮忙", "看一下", "看看", "看下", "排查一下原因", "排查一下异常", "根据日志")
_LEADING_NOISE_PREFIXES = ("各位领导", "下午好", "上午好", "客户描述", "补充", "现场反馈")
_LEADING_PERSON_PREFIX_RE = re.compile(
    r"^(?:@?\s*)?"
    r"(?:(?:[A-Za-z0-9_\-\u4e00-\u9fff]{1,8}\s+)?"
    r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,8}(?:老师|工|经理|总|哥|姐|大佬)[，,、\s]*){1,3}"
)
_LEADING_HELP_PREFIX_RE = re.compile(
    r"^(?:麻烦|辛苦|幸苦|请|劳烦|还请)?(?:帮忙)?(?:看一下|看下|看看|帮忙看下|帮忙看一下)(?:这个|下)?(?:问题点?|异常)?[，,：:\s]*"
)
_LEADING_OPINION_PREFIX_RE = re.compile(
    r"^(?:我想把|我想|我觉得|我看|看起来|感觉|应该是|应该|像是|怀疑是|推测是|估计是|可能是|貌似|看着像)[，,：:\s]*"
)
_REPORT_PREFIXES = ("今日汇报", "现场工作汇总", "一、现场工作", "二、问题收集", "问题汇总", "每日数据")
_FAULT_LABEL_NOISE_MARKERS = (
    "收到", "我来协调", "付款流程", "如沟通", "准备好了", "请领导知悉", "以上请领导知悉", "工作汇报",
    "现场工作汇总", "每日反馈", "签单", "合同", "客户地址", "行情 报价", "京东", "淘宝", "邀请", "要一个报告",
)
_VERSION_CONTEXT_MARKERS = ("版本", "version", "Version", "主程序", "算法包", "软件包", "固件", "驱动", "私包", "包")
_TIME_CONTEXT_MARKERS = ("早上", "上午", "下午", "晚上", "凌晨", "夜班", "白班", "发生时间", "时间", "左右", "点", "点钟", "：", ":")
_HELP_TAIL_RE = re.compile(
    r"(?:，|,|。|；|;|\s|@)*"
    r"(?:麻烦|辛苦|幸苦|帮忙|请|劳烦|还请)?"
    r"[^。！？!?；;，,]{0,30}"
    r"(?:帮忙)?"
    r"(?:继续跟踪一下|根据日志(?:查询|排查|分析)?一下?异常(?:引起)?原因|根据日志(?:查询|排查|分析)?异常(?:引起)?原因|"
    r"排查一下异常(?:引起)?原因|排查异常(?:引起)?原因|排查一下原因|分析异常原因|"
    r"帮忙看一下是什么问题|看一下是什么问题|需要收集什么信息排查故障|从硬件上(?:看一下)?(?:是什么问题)?|"
    r"帮忙核对一下|核对一下|帮忙看看|帮忙看下|帮忙看一下|提供一下|收集一下|明确下|给一下|介入处理|"
    r"安排给客户更新一下最新版本|协助一下|看看吧|看看|看一下|看下).*$"
)
_RETURN_RESULT_TAIL_RE = re.compile(r"(把结果返回我(?:看下|看一下)|把结果发我(?:看下|看一下)?|结果返回我(?:看下|看一下)).*$")
_TRAILING_LOOK_RE = re.compile(r"(?:看看|看下|看一下|看一看)$")
_TRAILING_PERSON_REQUEST_RE = re.compile(
    r"(?:麻烦|辛苦|幸苦|请|还请|劳烦)?[A-Za-z0-9_\-\u4e00-\u9fff]{0,8}(?:老师|工|哥|姐|总|大佬)?(?:帮忙)?(?:确认下|确认一下|看一下|看下|看看)?$"
)
_LEADING_HANDOFF_RE = re.compile(
    r"^(?:麻烦|辛苦|帮忙|请|劳烦)?"
    r"(?:[A-Za-z0-9_\-\u4e00-\u9fff]{0,8}(?:老师|工|哥|姐|总|大佬)|邢工|蒙老师|韦工|阿神|健哥)?[，,：:\s]*"
    r"(?:帮忙)?(?:看一下|看下|看看)(?:这个|下)?(?:报警|问题|异常)?(?:需要处理吗|是什么问题)?$"
)
_PRAISE_TOKEN_RE = re.compile(r":?YouAreTheBest:?|送你小红花|碰拳")
_SOLUTION_START_RE = re.compile(r"(?:解决方案|处理方案|暂时的解决方案|操作)[:：]")


def _remove_help_tail(value: Any) -> str:
    text = str(value or "").strip()
    # Strip attachment placeholders before tail removal.  Otherwise a trailing
    # "看下/看一下" pattern can cut after the closing bracket and leave an
    # orphan "Image: xxx" as the fault label.
    text = _NOISE_BRACKET_RE.sub(" ", text)
    text = _NOISE_BARE_ATTACHMENT_RE.sub(" ", text)
    text = _PRAISE_TOKEN_RE.sub(" ", text)
    previous = None
    while previous != text:
        previous = text
        text = _RETURN_RESULT_TAIL_RE.sub("", text).strip(" ，,。；;:：@")
        if any(k in text for k in ("升级", "拔插", "重启", "测试", "复测", "观察")):
            text = _TRAILING_LOOK_RE.sub("", text).strip(" ，,。；;:：@")
        text = _HELP_TAIL_RE.sub("", text).strip(" ，,。；;:：@")
        if any(k in text for k in ("麻烦", "辛苦", "幸苦", "帮忙", "还请", "劳烦")):
            text = _TRAILING_PERSON_REQUEST_RE.sub("", text).strip(" ，,。；;:：@")
    return text


def _split_sentences(text: str) -> list[str]:
    clean = _clean_chat_text(text, 1200)
    parts = re.split(r"[。！？!?；;\n]+", clean)
    return [p.strip(" ，,：:-") for p in parts if p.strip(" ，,：:-")]


def _split_action_clauses(text: str) -> list[str]:
    clean = _clean_chat_text(text, 1200)
    clean = _SOLUTION_START_RE.sub("。解决方案：", clean)
    clean = re.sub(r"(?:（|\()\s*\d+\s*(?:）|\))\s*[：:]?", "。", clean)
    clean = re.sub(r"(?:(?<=^)|(?<=[。；;\s]))[一二三四五六七八九十\d]{1,2}[、.．:：]\s*", "。", clean)
    parts = re.split(r"[。！？!?；;\n，,]+", clean)
    return [p.strip(" ，,：:-") for p in parts if p.strip(" ，,：:-")]


def _fault_label_signal_score(text: str) -> int:
    clean = _clean_chat_text(text, 240)
    if not clean:
        return 0
    hits = sum(1 for k in _FAULT_SENTENCE_KEYWORDS if k in clean)
    if any(obj in clean for obj in _FAULT_OBJECT_KEYWORDS) and any(state in clean for state in _FAULT_STATE_KEYWORDS):
        hits += 1
    if re.search(r"\b0x[0-9a-fA-F]{6,}\b", clean):
        hits += 1
    return hits


def _has_fault_label_signal(text: str) -> bool:
    return _fault_label_signal_score(text) > 0


def _strip_leading_noise(sentence: str) -> str:
    out = sentence.strip()
    changed = True
    while changed:
        changed = False
        person_stripped = _LEADING_PERSON_PREFIX_RE.sub("", out).strip(" ，,、：:-")
        if person_stripped != out and _has_fault_label_signal(person_stripped):
            out = person_stripped
            changed = True
        opinion_stripped = _LEADING_OPINION_PREFIX_RE.sub("", out).strip(" ，,、：:-")
        if opinion_stripped != out and _has_fault_label_signal(opinion_stripped):
            out = opinion_stripped
            changed = True
        help_stripped = _LEADING_HELP_PREFIX_RE.sub("", out).strip(" ，,、：:-")
        if help_stripped != out and _has_fault_label_signal(help_stripped):
            out = help_stripped
            changed = True
        for prefix in _LEADING_NOISE_PREFIXES:
            if out.startswith(prefix):
                out = out[len(prefix):].strip(" ，,：:-")
                changed = True
    return out


def _trim_fault_fact(value: str, limit: int = 160) -> str:
    raw_clean = _clean_chat_text(value, max(limit, 240))
    clean = _clean_chat_text(_remove_help_tail(value), max(limit, 240))
    if _reject_fault_label(clean or raw_clean):
        return ""
    if not clean and _has_fault_label_signal(raw_clean):
        clean = raw_clean
    if "客户反馈" in clean:
        tail = clean.split("客户反馈", 1)[1].strip(" ，,：:-")
        tail = re.split(
            r"(?:麻烦|辛苦|幸苦|帮忙提交|售后同事帮忙|请帮忙|还请|感谢|谢谢|客户今天反馈时间)",
            tail,
            maxsplit=1,
        )[0].strip(" ，,。；;:：")
        if tail and _has_fault_label_signal(tail):
            return _clean_chat_text("客户反馈" + tail, limit)
    trimmed = re.split(
        r"(?:麻烦|辛苦|幸苦|帮忙提交|售后同事帮忙|请帮忙|还请|感谢|谢谢)",
        clean,
        maxsplit=1,
    )[0].strip(" ，,。；;:：")
    if trimmed and _has_fault_label_signal(trimmed):
        return _clean_chat_text(trimmed, limit)
    return _clean_chat_text(clean, limit)


def _best_fault_sentence(text: str, fallback: str, limit: int = 160) -> str:
    candidates = []
    for sentence in _split_sentences(text):
        raw_stripped = _strip_leading_noise(sentence)
        stripped = _remove_help_tail(raw_stripped)
        if not stripped and any(k in raw_stripped for k in _FAULT_SENTENCE_KEYWORDS):
            stripped = raw_stripped
        if _has_fault_label_signal(stripped):
            candidates.append(stripped)
    if candidates:
        # Prefer concise sentences with concrete failure keywords, not greetings/status preambles.
        candidates.sort(key=lambda s: (len(s) > limit, len(s)))
        return _trim_fault_fact(candidates[0], limit)
    return _trim_fault_fact(str(fallback or text), limit)


def _fault_label_score(sentence: str) -> int:
    clean = _clean_chat_text(sentence, 240)
    if not clean:
        return -100
    if _reject_fault_label(clean):
        return -100
    score = 0
    fault_hits = _fault_label_signal_score(clean)
    score += fault_hits * 10
    if any(k in clean for k in ("客户反馈", "现场反馈", "问题点", "故障", "报错", "异常")):
        score += 8
    if any(k in clean for k in ("原因是", "根因", "定位到", "已解决", "恢复正常")):
        score += 3
    if any(k in clean for k in _FAULT_LABEL_NOISE_MARKERS):
        score -= 20
    if any(k in clean for k in ("麻烦", "辛苦", "帮忙", "看下", "看一下", "看看")) and fault_hits == 0:
        score -= 10
    if len(clean) < 4:
        score -= 10
    if len(clean) > 180:
        score -= 4
    return score


def _reject_fault_label(label: str) -> bool:
    clean = _clean_chat_text(label, 240)
    if not clean:
        return True
    fault_hits = _fault_label_signal_score(clean)
    if clean.startswith("群聊候选"):
        return True
    if _WEAK_ACTION_LABEL_RE.fullmatch(clean):
        return True
    if _LOG_LINE_LABEL_RE.search(clean) and fault_hits == 0:
        return True
    if re.search(r"\b0x[0-9a-fA-F]{6,}\b", clean) and _LOG_LINE_LABEL_RE.search(clean):
        return True
    if fault_hits == 0 and (
        re.search(r"\b(?:png|jpe?g|webp|gif|bmp)\b", clean, re.IGNORECASE)
        or re.search(r"\b\d{2,5}x\d{2,5}\b", clean)
    ):
        return True
    try:
        if _has_request_focus(clean) and not any(k in clean for k in ("报错", "异常", "失败", "蓝屏", "黑屏", "重启", "卡死", "闪退", "延迟", "卡顿", "误报", "漏检", "虚焊", "马赛克", "残帧", "丢包")):
            return True
    except NameError:
        pass
    compact = re.sub(r"[\W_]+", "", clean, flags=re.UNICODE)
    if len(compact) < 4:
        return True
    return False


def _best_episode_fault_label(fault_messages: list[str], extracted_symptom: str, semantic_text: str, *, limit: int = 160) -> str:
    scored: list[tuple[int, int, str]] = []
    label_sources = [*fault_messages, extracted_symptom]
    if not any(_clean_chat_text(value, 240) for value in label_sources):
        label_sources.append(semantic_text)
    for source_rank, value in enumerate(label_sources):
        for sentence in _split_sentences(value):
            raw_stripped = _strip_leading_noise(sentence)
            stripped = _remove_help_tail(raw_stripped)
            if not stripped and any(k in raw_stripped for k in _FAULT_SENTENCE_KEYWORDS):
                stripped = raw_stripped
            if not stripped:
                continue
            candidate = _trim_fault_fact(stripped, limit)
            score = _fault_label_score(candidate)
            if score > 0:
                scored.append((score, -source_rank, candidate))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
        return _clean_chat_text(scored[0][2], limit)
    fallback = extracted_symptom or " ".join(fault_messages) or semantic_text
    return _best_fault_sentence(fallback, extracted_symptom or " ".join(fault_messages) or semantic_text, limit)


def _label_is_handoff_noise(label: str, semantic_text: str) -> bool:
    clean = _clean_chat_text(label, 120)
    if not clean:
        return True
    if clean in {"麻烦工程师乙", "辛苦工程师乙", "幸苦工程师乙", "辛苦", "幸苦", "这个辛苦", "这个幸苦", "麻烦", "工程师乙 工程师乙，麻烦", "工程师乙，麻烦"}:
        return True
    if any(k in clean for k in ("麻烦", "辛苦", "幸苦")) and not _has_fault_label_signal(clean):
        return True
    if any(k in clean for k in ("帮忙查一下", "帮忙安排", "收货地址", "是否需要寄回", "可以提供一个收货地址", "要一个报告")):
        return True
    if any(k in semantic_text for k in ("收货地址", "是否需要寄回", "需要寄回", "寄回")) and any(k in clean for k in ("更换", "换下", "内存条", "显卡", "主板")):
        return True
    report_markers = ("各位领导", "现场工作汇报", "现场工作汇总", "今日现场情况", "工作进展", "培训方面", "日常数据回传")
    if any(k in clean[:120] for k in report_markers):
        fault_hits = _fault_label_signal_score(clean)
        if fault_hits < 2 or any(k in clean for k in ("培训", "工作进展", "日常数据", "需求记录", "交付情况")):
            return True
    if any(k in clean for k in ("没有收到新的注册信息", "注册信息")) and "报错" not in semantic_text and "异常" not in semantic_text:
        return True
    return False


def _action_sentences(values: list[str], *, limit: int = 500) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for sentence in _split_action_clauses(value):
            stripped = _remove_help_tail(_strip_leading_noise(sentence))
            stripped = _normalise_action_span(stripped)
            if not _is_review_grade_action(stripped):
                continue
            clean = _clean_chat_text(stripped, limit)
            if clean and clean not in seen:
                seen.add(clean)
                out.append(clean)
    return out


def _has_explicit_outcome_signal(text: str) -> bool:
    clean = str(text or "")
    return bool(
        _has_verified_fix_evidence(clean)
        or _has_action_failure(clean)
        or _has_any(clean, _TEMPORARY_MARKERS)
        or _has_any(clean, _NO_RECURRENCE_OBSERVATION_MARKERS)
        or any(marker in clean for marker in (
            "恢复正常", "恢复生产", "正常生产", "恢复后", "仍需观察", "继续观察", "再次出现", "又出现",
        ))
    )


def _sentence_role(text: str) -> str:
    """Classify one evidence sentence before graph-object extraction.

    The classifier is intentionally conservative: a fault/status sentence may
    remain evidence, but it must not become a DiagnosticAction merely because
    it contains words such as ``测试`` or ``重启``.
    """

    clean = _clean_chat_text(text, 500)
    if not clean:
        return "noise"
    if _has_request_focus(clean) and any(marker in clean for marker in (
        "日志", "DLOG", "dmp", "DMP", "版本", "IP", "截图", "报错", "数据包", "程序文件", "复现步骤",
    )):
        return "required_info"
    # Action cleanup works clause-by-clause.  Running ``_remove_help_tail`` on
    # the whole sentence can erase a legitimate action such as
    # "升级 V0.27.43 再观察一段看看".  Reuse the production action splitter so
    # sentence-role classification and later extraction share one contract.
    binary_clarification = bool(re.search(r"(?:是.+还是.+|是.+还是.+了|还是.+\?)", clean)) or any(
        marker in clean for marker in ("是设备重启了还是软件退出了", "是网络问题吗", "还需要进行其他方面排查不")
    )
    action_span = _normalise_action_span(clean)
    is_action = bool(action_span) and _is_review_grade_action(action_span)
    if not is_action and not binary_clarification:
        is_action = bool(_action_sentences([clean], limit=500))
    if is_action:
        return "observed_outcome" if action_span != clean or _has_explicit_outcome_signal(clean) else "diagnostic_action"
    # Retain a result-only clause as evidence even when its action span is too
    # weak or incomplete to materialise.  Downstream action extraction must
    # use `action_span`, not promote the full result sentence.
    if any(
        separator in clean and any(marker in clean.split(separator, 1)[1] for marker in _ACTION_RESULT_MARKERS)
        for separator in _ACTION_RESULT_SEPARATORS
    ):
        return "observed_outcome"
    if _has_fault_label_signal(clean):
        return "symptom"
    if any(marker in clean for marker in ("培训", "交付", "付款", "发货", "会议", "工作计划", "客户需求")):
        return "noise"
    return "context"


def _sentence_role_records(episode: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_role, key in (
        ("current_fault", "fault_description_messages"),
        ("current_diagnostic", "diagnostic_chain_messages"),
        ("current_resolution", "resolution_messages"),
        ("w7_promoted", "case_evidence_messages"),
    ):
        for message in episode.get(key) or []:
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            raw = str(message.get("text") or message.get("content_summary") or "")
            for sentence in _split_sentences(raw):
                clean = _clean_chat_text(sentence, 500)
                role = _sentence_role(clean)
                if source_role == "current_diagnostic" and _has_explicit_outcome_signal(clean):
                    role = "observed_outcome"
                if source_role == "current_resolution" and _has_explicit_outcome_signal(clean):
                    role = "observed_outcome"
                dedupe = (message_id, clean, role)
                if not clean or dedupe in seen:
                    continue
                seen.add(dedupe)
                records.append({
                    "message_id": message_id,
                    "text": clean,
                    "action_span": _normalise_action_span(clean),
                    "role": role,
                    "source_role": source_role,
                    "evidence_message_ids": [message_id] if message_id else [],
                })
    return records


def _is_field_report_action_text(text: str) -> bool:
    """Allow explicit operations embedded in a field-report fault message."""

    value = _clean_chat_text(text, 500)
    value = re.sub(r"^\s*(?:\d+[、.．:]|[一二三四五六七八九十]+[、.．:])\s*", "", value)
    value = re.sub(r"^(?:现场|客户|售后)\s*(?:已|已经|目前|正在)?\s*", "", value)
    return value.startswith(_CONCRETE_ACTION_KEYWORDS)


def _review_grade_actions(values: list[str], *, limit: int = 500) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean_chat_text(_remove_help_tail(value), limit)
        if not _is_review_grade_action(clean):
            continue
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _merge_action_candidates(*groups: list[str], limit: int = 500) -> list[str]:
    """Preserve explicit extracted actions first, then append novel actions.

    Gold-case episode inputs often carry curated `debug_actions` order.  W2
    should not drop those steps simply because later sentence heuristics are
    conservative about certain diagnostic verbs or long clauses.
    """

    out: list[str] = []
    seen: set[str] = set()
    for group_index, group in enumerate(groups):
        for value in group:
            # A W1 structured hint can still be a multi-sentence diagnostic
            # finding followed by a recommendation.  Split sentence-level
            # boundaries before validation so one valid verb cannot promote
            # the entire finding/result paragraph to an action.
            spans = _split_sentences(value) if group_index == 0 else [str(value or "")]
            for span in spans:
                clean = _clean_chat_text(_remove_help_tail(span), limit)
                clean = _normalise_action_span(clean, structured=group_index == 0)
                if not _is_review_grade_action(clean, structured=group_index == 0):
                    continue
                if clean and clean not in seen:
                    seen.add(clean)
                    out.append(clean)
    return out


def _normalise_action_span(text: str, *, structured: bool = False) -> str:
    """Return only the executable span of an action-like clause.

    Chat summaries frequently encode ``operation + 后 + observed result`` in
    one sentence.  Keeping the complete sentence makes the result look like a
    second executable instruction.  Conversely, a bare trailing ``后`` is an
    incomplete fragment and cannot prove what was performed.  This helper is
    deliberately syntactic and domain-independent; outcome strength remains
    the responsibility of the outcome extractor.
    """

    clean = _clean_chat_text(_remove_help_tail(text), 500)
    if not clean:
        return ""
    clean = re.sub(r"^\s*(?:\d+[、.．:]|[一二三四五六七八九十]+[、.．:])\s*", "", clean).strip()

    # Engineering shorthand such as ``待换电池验证`` denotes a recommendation,
    # not an already executed action.  Expand only the leading shorthand so
    # downstream execution-status logic can retain the recommendation state.
    clean = re.sub(r"^(?:后续)?待\s*换(?=\S)", "建议更换", clean)

    if _DANGLING_ACTION_TAIL_RE.search(clean):
        return ""

    # A diagnostic finding states what was inferred, not what an operator
    # should execute.  A later sentence may still independently provide the
    # proposed action and is handled by `_merge_action_candidates` splitting.
    finding_tail = _DIAGNOSTIC_FINDING_PREFIX_RE.sub("", clean, count=1)
    if finding_tail != clean:
        return ""

    # Trigger descriptions often use ``执行 A 的时候出现故障``.  Preserve the
    # complete operation sequence and remove only the terminal observation;
    # splitting at an earlier ``之后`` would incorrectly discard A's second
    # operation (for example search -> open details -> crash).
    # ``时`` is a trigger delimiter in “执行操作时发生故障”, but not in
    # lexical result markers such as “暂时恢复/短时正常”.
    trigger_match = re.match(r"^(.+?)(?:(?<![暂短同有])时|的时候)(.+)$", clean)
    if trigger_match and any(marker in trigger_match.group(2) for marker in _ACTION_RESULT_MARKERS):
        trigger_prefix = trigger_match.group(1).strip(" ，,；;。")
        if (
            _action_starts_at_clause_head(trigger_prefix, structured=structured)
            and _has_explicit_action_object(trigger_prefix, structured=structured)
        ):
            clean = trigger_prefix
        else:
            return ""

    # Prefer the earliest result boundary whose suffix is recognisably a
    # state/observation.  Do not split ``升级后检查日志`` because the latter is
    # another operation rather than a result.
    result_boundaries: list[tuple[int, str]] = []
    for separator in _ACTION_RESULT_SEPARATORS:
        start = 0
        while True:
            position = clean.find(separator, start)
            if position < 0:
                break
            if separator == "后" and position > 0 and clean[position - 1] == "然":
                start = position + len(separator)
                continue
            suffix = clean[position + len(separator):].strip(" ，,；;。")
            if suffix and any(marker in suffix for marker in _ACTION_RESULT_MARKERS):
                result_boundaries.append((position, separator))
            start = position + len(separator)
    if result_boundaries:
        position, _ = min(result_boundaries, key=lambda item: item[0])
        prefix = clean[:position].strip(" ，,；;。")
        if _has_explicit_action_object(prefix, structured=structured):
            clean = prefix
        else:
            return ""

    if not _has_explicit_action_object(clean, structured=structured):
        return ""
    return clean


def _has_explicit_action_object(text: str, *, structured: bool = False) -> bool:
    """Reject weak verb-only utterances while retaining scoped operations."""

    clean = str(text or "").strip()
    if not clean:
        return False
    value = re.sub(r"^(?:后续)?(?:建议|推荐|可以尝试|计划|先|再|继续)\s*", "", clean).strip()
    # A reboot mode itself supplies scope even when the device is implicit.
    if value.startswith(("断电重启", "每日重启", "每天重启")):
        return True
    weak_verbs = ("重启", "调整", "更换", "替换", "检查", "确认", "测试", "验证", "观察", "分析", "排查")
    for verb in weak_verbs:
        if not value.startswith(verb):
            continue
        remainder = value[len(verb):].strip(" ：:，,。")
        if remainder in {"", "下", "一下", "看看", "看下", "看一下", "试试", "验证"}:
            return False
        if remainder.startswith(("后", "之后", "然后")):
            return False
        return True
    # Other accepted action forms still need the existing review-grade verb
    # contract; this function only adds a minimum completeness invariant.
    keywords = (*_CONCRETE_ACTION_KEYWORDS, *(_STRUCTURED_ACTION_KEYWORDS if structured else ()))
    return any(keyword in value for keyword in keywords)


def _action_starts_at_clause_head(text: str, *, structured: bool = False) -> bool:
    """Separate an operator step from a scenario ending in ``测试时...``."""

    value = str(text or "").strip()
    value = re.sub(r"^(?:现场|客户|售后|研发)\s*(?:已|已经|正在)?\s*", "", value)
    value = re.sub(r"^(?:已|已经|先|再|继续|建议|推荐|计划|可以尝试)\s*", "", value)
    keywords = (*_CONCRETE_ACTION_KEYWORDS, *(_STRUCTURED_ACTION_KEYWORDS if structured else ()))
    if value.startswith(keywords):
        return True
    if value.startswith(("重新", "再次")):
        return value[2:].startswith(keywords)
    if value.startswith(("把", "将")):
        return any(keyword in value for keyword in keywords)
    return False


def _solution_sentences(values: list[str], *, limit: int = 500) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for sentence in _split_action_clauses(value):
            clean = _clean_chat_text(_remove_help_tail(sentence), limit)
            if not clean or clean in seen:
                continue
            if (
                any(k in clean for k in ("解决方案", "处理方案", "暂时的解决方案", "修复系统文件", "重装显卡驱动", "启用TLS", "禁止Windows更新"))
                or _has_strong_fix_evidence(clean)
            ):
                seen.add(clean)
                out.append(clean)
    return out


def _is_review_grade_action(text: str, *, structured: bool = False) -> bool:
    clean = _remove_help_tail(str(text or "").strip())
    if not clean:
        return False
    if "解决方案" in clean or "处理方案" in clean or clean.startswith("操作："):
        return False
    if any(clean.startswith(prefix) for prefix in _REPORT_PREFIXES) or any(prefix in clean[:80] for prefix in _REPORT_PREFIXES):
        return False
    if _LEADING_HANDOFF_RE.match(clean):
        return False
    if any(k in clean for k in ("让邢工看一下", "让现场提供啥信息", "麻烦邢工", "辛苦帮忙看下", "辛苦嘉兴", "幸苦", "麻烦商务", "辛苦大家", "麻烦收集", "辛苦谢工", "需要收集什么信息")):
        return False
    if "帮忙确认" in clean or "帮忙确定" in clean or clean.endswith(("帮忙确认下", "帮忙确认一下", "帮忙确定下", "帮忙确定一下")):
        return False
    if any(k in clean for k in ("还请", "请邢", "请 帮忙", "麻烦辛苦确定")) and any(
        k in clean for k in ("是否", "是还是", "点是", "点否", "异常原因", "物理损坏", "更换", "有没有录到", "方案", "原因")
    ):
        return False
    if any(k in clean for k in ("帮忙沟通", "请帮忙沟通", "协助沟通")):
        return False
    if re.search(r"(?:是.+还是.+|是.+还是.+了|还是.+\?)", clean) or any(k in clean for k in ("是设备重启了还是软件退出了", "是网络问题吗", "还需要进行其他方面排查不")):
        return False
    if any(k in clean for k in ("日志正在联系客户", "收集日志", "收集数据", "收集以上日志", "DMP文件和日志", "dmp文件和日志", "售后同事已帮忙收集数据")):
        if any(k in clean for k in ("帮忙", "联系客户", "可以", "已", "正在")) or not any(
            k in clean for k in ("检查", "设置", "替换", "拔插", "安装", "升级", "回退", "重启", "运行", "修复", "搜索", "筛选", "事件查看器")
        ):
            return False
    if len(clean) > 180 and any(k in clean for k in ("客户反馈", "发生时间", "每日数据", "现场问题", "现场工作")):
        return False
    if len(clean) > 180 and any(k in clean for k in ("问题本质", "根因定性", "高概率嫌疑", "责任归属", "调用链", "实际问题")):
        return False
    if any(k in clean for k in ("支持循环覆盖", "京东", "淘宝", "行情 报价", "图片 价格", "采购")):
        return False
    if any(k in clean for k in ("现场工作汇报", "现场工作", "日常数据回传", "需求记录", "培训员工", "综合对比")):
        return False
    if re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", clean):
        return False
    if any(k in clean for k in ("ci-robot", "mentioned this issue in commit", "修复/完成版本", "{quote}fix:")):
        return False
    if len(clean) > 180 and any(k in clean for k in ("网卡1", "网卡2", "正确MAC", "ARP 缓存", "表现出来的现象", "此时会发生")):
        return False
    if any(k in clean for k in ("确认时报错", "确认时报", "确认时弹", "页面确认时报错")):
        return False
    if any(k in clean for k in _HANDOFF_ONLY_KEYWORDS) and any(k in clean for k in ("是什么问题", "需要处理吗", "排查一下", "看一下", "看下", "看看", "明确下", "提供啥信息")):
        if not any(k in clean for k in ("检查", "确认", "导出", "提供", "上传", "设置", "替换", "拔插", "安装", "升级", "回退", "收集", "截图", "抓取", "事件查看器")):
            return False
    if any(k in clean for k in ("日志已上传", "蓝屏问题提供下日志", "又出现蓝屏")) and any(k in clean for k in ("看一下", "看下", "麻烦", "辛苦")):
        return False
    if any(k in clean for k in ("已远程收集", "收集了日志", "远程收集了日志", "使用蓝屏dmp脚本收集", "DMP文件和日志", "dmp文件", "DMP文件")):
        return False
    if any(k in clean for k in ("从日志上看", "从软件日志上看", "没有检测到原因", "查看日志发现", "收集日志发现")) and not any(k in clean for k in ("建议", "重装", "安装", "更换", "替换", "设置")):
        return False
    if any(k in clean for k in ("突然闪退", "测试时突然闪退", "客户反馈未进行任何操作")) and not any(k in clean for k in ("重启软件", "检查", "导出", "设置", "安装", "更换", "替换", "运行")):
        return False
    if re.search(r"(?:断电)?重启(?:后|之后).{0,28}(?:无法|失败|异常|不拍照|不能)", clean) and not any(
        marker in clean for marker in ("重启了一下", "执行重启", "进行重启", "尝试重启", "重新重启")
    ):
        return False
    if clean.startswith("测试") and any(marker in clean for marker in ("卡死", "闪退", "卡顿", "异常", "失败", "误报")) and any(
        marker in clean for marker in ("有一台", "有两台", "有三台", "的时候", "现象", "设备")
    ):
        return False
    has_concrete = any(k in clean for k in _CONCRETE_ACTION_KEYWORDS)
    if structured and not has_concrete:
        has_concrete = any(k in clean for k in _STRUCTURED_ACTION_KEYWORDS)
    if not has_concrete:
        return False
    if "重启" in clean and not any(k in clean for k in ("重启软件", "重启工控机", "重启服务", "断电重启", "重启相机", "重启AOI", "重启aoi")):
        if any(k in clean for k in ("蓝屏", "已重启", "重启三次", "重启多次", "自动重启", "突然重启", "无故重启", "异常重启", "白班", "夜班", "发生时间", "恢复", "正常使用", "正常")) and not clean.startswith(("每日", "每天")):
            return False
    if any(k in clean for k in ("联系", "介入处理", "安排给客户", "反馈的蓝屏问题", "研发建议")) and not any(k in clean for k in ("检查", "导出", "设置", "安装", "更换", "替换", "拔插", "运行")):
        return False
    handoff_only = any(k in clean for k in _HANDOFF_ONLY_KEYWORDS) and not any(
        k in clean for k in ("检查", "确认", "导出", "提供", "上传", "重启", "设置", "替换", "拔插", "安装", "升级", "回退", "收集", "运行", "截图", "抓取", "事件查看器")
    )
    if handoff_only:
        return False
    # Pure symptom/reassignment clauses are evidence for Error, not DiagnosticCheck.
    if any(k in clean for k in ("客户反馈", "现场反馈", "发生闪退", "发生异常", "没发现报错", "没有检测到原因")) and not any(
        k in clean for k in ("检查", "确认", "导出", "提供", "上传", "设置", "替换", "拔插", "截图", "事件查看器", "重启软件", "重启工控机", "重启服务", "断电重启")
    ):
        return False
    return True


def _is_likely_software_version(version: str, semantic_text: str) -> bool:
    value = str(version or "").strip()
    if not value:
        return False
    stripped = value[1:] if value[:1].lower() == "v" else value
    parts = stripped.split(".")
    if len(parts) == 2 and not value[:1].lower() == "v":
        # Two-part bare numbers are usually time fragments or truncated semver fragments.
        # Keep them only when the version marker is tight and the numeric shape looks like an AOI release.
        if not (parts[0] in {"0", "1"} and len(parts[1]) <= 2):
            return False
        text = str(semantic_text or "")
        value_pattern = re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d.])")
        for match in value_pattern.finditer(text):
            before = text[max(0, match.start() - 8):match.start()]
            if any(k in before for k in _VERSION_CONTEXT_MARKERS):
                break
        else:
            return False
    text = str(semantic_text or "")
    value_pattern = re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d.])")
    for match in value_pattern.finditer(text):
        start = max(0, match.start() - 12)
        end = min(len(text), match.end() + 12)
        window = text[start:end]
        before = text[max(0, match.start() - 10):match.start()]
        after = text[match.end():min(len(text), match.end() + 10)]
        has_version_before = any(k in before for k in _VERSION_CONTEXT_MARKERS)
        has_version_context = has_version_before or any(k in window for k in _VERSION_CONTEXT_MARKERS)
        has_time_context = any(k in f"{before}{after}" for k in _TIME_CONTEXT_MARKERS)
        env_context = any(k in window for k in ("系统", "镜像", "ddr", "DDR", "CUDA", "cuda", "显卡", "驱动"))
        app_context = any(k in window for k in ("软件", "主程序", "算法包", "包版本", "后升级", "恢复"))
        if len(parts) == 2 and value == "1.0" and env_context and not app_context:
            continue
        if has_time_context and not has_version_before:
            continue
        if has_version_context:
            return True
    if len(parts) >= 3:
        return True
    if len(parts) != 2:
        return False
    # Date/time fragments such as 05.16 or 11.02 are common in Jira titles and should not become SoftwareVersion nodes.
    if any(part.startswith("0") and len(part) > 1 for part in parts):
        return False
    return value[:1].lower() == "v"


def _slug(value: Any, *, limit: int = 60) -> str:
    text = _one_line(value, limit)
    if not text:
        return "unknown"
    safe = re.sub(r"[^0-9A-Za-z_.:\-\u4e00-\u9fff]+", "-", text).strip("-").lower()
    return safe[:limit] or "unknown"


def _candidate_id(source_id: str, label: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{label}".encode("utf-8")).hexdigest()[:12]
    return f"chatcand:{digest}"


def _required_info_id(source_id: str, message_id: str, slot: str, condition: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{message_id}|{slot}|{condition}".encode("utf-8")).hexdigest()[:12]
    return f"reqinfo:{digest}"


def _classify_category(text: str) -> str:
    lowered = text.lower()
    scores: dict[str, int] = {name: 0 for name, _ in CATEGORY_RULES}
    for name, keywords in CATEGORY_RULES:
        scores[name] = sum(1 for kw in keywords if kw.lower() in lowered)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0] if ranked and ranked[0][1] > 0 else "系统与软件异常"


def _first_sentence(text: str) -> str:
    parts = re.split(r"[。！？!?\n]", text)
    for part in parts:
        compact = _one_line(part, 120)
        if compact:
            return compact
    return _one_line(text, 120)


def _message_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [_one_line(m.get("text") or m.get("content_summary"), 500) for m in messages if _one_line(m.get("text") or m.get("content_summary"))]


def _episode_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("episode_id"):
        return summary
    episodes = [x for x in summary.get("episodes") or [] if isinstance(x, dict)]
    for episode in episodes:
        if episode.get("completeness") != "noise":
            return episode
    if episodes:
        return episodes[0]
    return summary


def _semantic_text(item: dict[str, Any], extracted: dict[str, Any]) -> str:
    parts: list[str] = []
    if extracted.get("fault_focus_text"):
        parts.append(str(extracted.get("fault_focus_text")))
    for key in ("symptom_raw", "conclusion", "key_conclusion", "content", "text"):
        if extracted.get(key):
            parts.append(str(extracted.get(key)))
        if item.get(key):
            parts.append(str(item.get(key)))
    parts.extend(_list(extracted.get("debug_actions")))
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_evidence_messages"):
        parts.extend(_message_texts([x for x in item.get(key) or [] if isinstance(x, dict)]))
    # ``case_context_messages`` remains navigation-only.  W7 explicitly
    # promotes a small auditable subset into ``case_evidence_messages``.
    # It may span unrelated faults in a busy group.  Facts extracted from it
    # would be impossible to audit against ``evidence_message_ids``.
    artifacts = extracted.get("artifacts") if isinstance(extracted.get("artifacts"), dict) else {}
    for key in ("sites", "versions", "devices", "log_paths", "project_files", "attachment_names"):
        parts.extend(_list(extracted.get(key)))
        parts.extend(_list(artifacts.get(key)))
    tool_evidence = _tool_evidence(extracted)
    for key in ("versions", "ip_configs", "project_files", "log_paths"):
        if key == "versions":
            parts.extend(_tool_evidence_versions(tool_evidence))
        elif key == "ip_configs":
            parts.extend(_tool_evidence_ip_configs(tool_evidence))
        elif key == "project_files":
            parts.extend(_tool_evidence_project_files(tool_evidence))
        elif key == "log_paths":
            parts.extend(_tool_evidence_log_paths(tool_evidence))
    log_hints = _tool_evidence_log_hints(tool_evidence)
    parts.extend(log_hints.get("error_lines") or [])
    parts.extend(log_hints.get("error_codes") or [])
    parts.extend(log_hints.get("phase_hints") or [])
    dmp_hints = _tool_evidence_dmp_hints(tool_evidence)
    parts.extend(dmp_hints.get("dump_kinds") or [])
    parts.extend(dmp_hints.get("bugcheck_hints") or [])
    project_hints = _tool_evidence_project_hints(tool_evidence)
    for values in project_hints.values():
        parts.extend(values)
    jira_hints = _tool_evidence_jira_hints(tool_evidence)
    for values in jira_hints.values():
        parts.extend(values)
    attachment_hints = _tool_evidence_attachment_hints(tool_evidence)
    parts.extend(attachment_hints.get("text_previews") or [])
    parts.extend(attachment_hints.get("error_lines") or [])
    parts.extend(attachment_hints.get("error_codes") or [])
    parts.extend(attachment_hints.get("phase_hints") or [])
    image_hints = _tool_evidence_image_hints(tool_evidence)
    parts.extend(image_hints.get("formats") or [])
    parts.extend(image_hints.get("dimension_labels") or [])
    document_hints = _tool_evidence_document_hints(tool_evidence)
    parts.extend(document_hints.get("formats") or [])
    parts.extend(document_hints.get("text_previews") or [])
    parts.extend(document_hints.get("page_count_labels") or [])
    for link in [*(extracted.get("jira_links") or []), *(artifacts.get("jira_links") or []), *_tool_evidence_jira_links(tool_evidence)]:
        if isinstance(link, dict):
            parts.append(str(link.get("label") or ""))
            parts.append(str(link.get("url") or ""))
    return " ".join(parts)


def _request_context_text(request: dict[str, Any]) -> str:
    parts = [str(request.get("text") or "")]
    for key in ("context_before", "context_after"):
        for msg in request.get(key) or []:
            if isinstance(msg, dict):
                parts.append(str(msg.get("text") or msg.get("content_summary") or ""))
    return " ".join(parts)


_INFO_FILE_HINT_RE = re.compile(r"[\w\-.\u4e00-\u9fff\[\]()（）]+\.(?:zip|rar|7z|log|evtx|dmp|proj|csv|toml|pml)", re.IGNORECASE)
_REQUEST_FOCUS_VERBS = (
    "提供", "上传", "补充", "导出", "截图", "打包", "发一下", "发下", "发我", "发给", "发来",
    "传一下", "给一下", "给我", "说明", "确认", "确认下", "确认一下", "看下", "看一下",
    "有没有", "是否有", "能否", "需要",
)
_REQUEST_FOCUS_OBJECTS = (
    "日志", "DLOG", "dlog", "诊断数据", "数据包", "版本", "IP", "ip", "报错", "错误码", "截图", "图片",
    "dmp", "DMP", "dump", "Dump", "转存储", "转储", "配方", "程序文件", "工程文件", "模板", "板型", "样本", "文件",
    "阶段", "复现", "操作步骤", "设备型号", "相机型号", "控制器型号", "工控机型号", "固件",
    "现场信息", "客户信息", "客户名称", "线体", "设备编号", "项目名", "电源", "磁盘", "内存", "系统环境",
    "日期", "时间", "负责人", "责任归属", "归属模块",
)
_BROAD_CONTEXT_SLOTS = {"site", "device_model", "environment", "owner_context", "program_file"}
_REPORT_LIKE_MARKERS = (
    "工作汇报",
    "现场工作汇总",
    "现场工作：",
    "现场工作内容",
    "今日现状已更新",
    "请查阅",
    "请领导知悉",
    "以上请领导知悉",
    "各位领导",
)
_DIRECT_INFO_REQUEST_MARKERS = (
    "请提供",
    "麻烦提供",
    "辛苦提供",
    "请上传",
    "麻烦上传",
    "请补充",
    "麻烦补充",
    "请导出",
    "麻烦导出",
    "请截图",
    "麻烦截图",
    "请打包",
    "麻烦打包",
    "发我",
    "发给我",
    "给我看看",
    "给我看下",
    "给我看一下",
)
_NOT_REQUEST_FOCUS_MARKERS = (
    "没有截图",
    "暂无截图",
    "未截图",
    "没截图",
    "没有拍照",
    "暂无图片",
    "截图能看得到",
    "截图能看到",
    "截图可以看到",
    "截图上能看到",
    "截图里能看到",
    "截图显示",
    "图片显示",
    "图片能看到",
    "截图如下",
    "图片如下",
)


def _request_focus_text(text: str) -> str:
    """Keep only the clauses that actually ask for diagnostic information.

    Real chat messages often mix root-cause explanations, handoff chatter and a
    final ask like "把 MEMORY.DMP 发给我".  Slot extraction must run on the ask
    clause, not the whole paragraph, otherwise context words such as 客户/设备/内存
    become fake required-info candidates.
    """

    raw_text = str(text or "")
    if any(k in raw_text for k in ("没有关系", "无关", "关系不大")):
        return ""
    if _is_report_like_without_direct_request(raw_text):
        return ""
    clauses = _split_action_clauses(text)
    selected: list[str] = []
    for clause in clauses:
        clean = _clean_chat_text(clause, 240)
        if not clean:
            continue
        if _is_explanatory_not_request_clause(clean):
            continue
        has_request = _has_request_focus(clean)
        has_object = any(obj in clean for obj in _REQUEST_FOCUS_OBJECTS) or bool(_INFO_FILE_HINT_RE.search(clean))
        if has_request and has_object:
            selected.append(clean)
    if not selected:
        clean = _clean_chat_text(text, 500)
        if _is_explanatory_not_request_clause(clean):
            return ""
        has_request = _has_request_focus(clean)
        has_object = any(obj in clean for obj in _REQUEST_FOCUS_OBJECTS) or bool(_INFO_FILE_HINT_RE.search(clean))
        if has_request and has_object:
            selected.append(clean)
    return " ".join(selected)


def _is_explanatory_not_request_clause(text: str) -> bool:
    if _is_report_like_without_direct_request(text):
        return True
    if any(k in text for k in _NOT_REQUEST_FOCUS_MARKERS):
        return True
    return any(k in text for k in (
        "无法上传",
        "不能上传",
        "无法补充",
        "不能补充",
        "相关信息已收集",
        "相关数据已收集",
        "数据已收集",
        "信息已收集",
        "无法提供",
        "不能提供",
        "点击提供程序",
        "按提供程序",
        "客户提供",
        "厂家提供",
        "你提供的",
        "您提供的",
        "已使用你提供",
        "已使用您提供",
        "提供给研发",
        "提供给产研",
        "提供给客户",
        "所有发给",
        "可以倒推",
        "可以优化",
        "提jira跟踪",
        "提JIRA跟踪",
        "请知悉",
        "已知bug",
        "已知BUG",
        "明确说明",
        "没有关系",
        "问题驱动",
        "这意味着",
        "这指出了",
        "总结就是",
        "通常不是问题的根源",
        "属于Windows核心组件",
        "工作范围严格限定",
        "不会触及",
    ))


def _is_report_like_without_direct_request(text: str) -> bool:
    clean = str(text or "")
    if not any(marker in clean for marker in _REPORT_LIKE_MARKERS):
        return False
    return not any(marker in clean for marker in _DIRECT_INFO_REQUEST_MARKERS)


def _has_request_focus(text: str) -> bool:
    if any(k in text for k in _NOT_REQUEST_FOCUS_MARKERS):
        return False
    if re.search(r"(请|麻烦|辛苦|劳烦).{0,32}(提供|上传|补充|导出|截图|打包|发我|发给我|传一下|给我|说明|确认)", text):
        return True
    if re.search(r"(把|将).{0,48}(发给我|发我|给我|上传|提供)", text):
        return True
    if re.search(r"(提供|上传|补充|导出|打包).{0,24}(一下|下|给我|发我|看看|看下|看一下|吗|？|\\?|$)", text):
        return True
    if re.search(r"(?:请|麻烦|辛苦|劳烦|帮忙).{0,24}截图(?:一下|下|给我|发我|看看|看下|看一下|吗|？|\\?|$)", text):
        return True
    if any(k in text for k in ("发我", "发给我", "给我看看", "给我看下", "给我看一下")):
        return True
    if "?" in text or "？" in text or any(k in text for k in ("有没有", "是否有", "有无", "还需要", "需要再", "能否")):
        object_alt = "|".join(re.escape(k) for k in _REQUEST_FOCUS_OBJECTS)
        return bool(
            re.search(rf"(有没有|是否有|有无|还需要|需要再|能否).{{0,24}}(?:{object_alt})", text)
            or re.search(rf"(?:{object_alt}).{{0,24}}(有没有|是否有|有无|还需要|需要再|能否)", text)
        )
    return False


def _slot_allowed_by_focus(slot: str, focus_text: str) -> bool:
    """Reject broad context slots unless the ask clause names that slot exactly."""

    if slot not in _BROAD_CONTEXT_SLOTS:
        return True
    text = str(focus_text or "")
    if slot == "site":
        return any(k in text for k in ("现场信息", "客户信息", "客户名称", "线体", "设备编号", "项目名", "哪个现场", "哪个客户"))
    if slot == "device_model":
        return any(k in text for k in ("设备型号", "相机型号", "光源型号", "控制器型号", "工控机型号", "硬件型号", "型号"))
    if slot == "environment":
        return any(k in text for k in ("系统环境", "运行环境", "电源", "磁盘", "内存", "温度", "日期", "时间"))
    if slot == "owner_context":
        return any(k in text for k in ("负责人", "责任归属", "归属模块", "找谁", "当前处理人"))
    if slot == "program_file":
        if "日志" in text and not any(k in text for k in ("程序文件", "工程文件", "配方", "模板", "板型", ".proj")):
            return False
        return any(k in text for k in ("程序文件", "工程文件", "配方", "模板", "板型", ".proj"))
    return True


def _tool_evidence(extracted: dict[str, Any]) -> dict[str, Any]:
    direct = extracted.get("tool_evidence") if isinstance(extracted.get("tool_evidence"), dict) else {}
    artifacts = extracted.get("artifacts") if isinstance(extracted.get("artifacts"), dict) else {}
    nested = artifacts.get("tool_evidence") if isinstance(artifacts.get("tool_evidence"), dict) else {}
    return direct or nested or {}


def _tool_evidence_versions(tool_evidence: dict[str, Any]) -> list[str]:
    versions: list[str] = []
    for result in tool_evidence.get("proj_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        versions.extend(_list(hints.get("versions")))
    for result in tool_evidence.get("jira_parse_results") or []:
        if not isinstance(result, dict):
            continue
        versions.extend(_list(result.get("version_hints")))
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        versions.extend(_list(hints.get("versions")))
    return _clean_list(versions, limit=80)


def _tool_evidence_ip_configs(tool_evidence: dict[str, Any]) -> list[str]:
    ips: list[str] = []
    for result in tool_evidence.get("proj_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        ips.extend(_list(hints.get("ip_addresses")))
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        ips.extend(_list(hints.get("ip_addresses")))
    return _clean_list(ips, limit=80)


def _tool_evidence_project_hints(tool_evidence: dict[str, Any]) -> dict[str, list[str]]:
    project_names: list[str] = []
    model_types: list[str] = []
    file_roles: list[str] = []
    pcb_types: list[str] = []
    device_names: list[str] = []
    manufacturers: list[str] = []
    for result in tool_evidence.get("proj_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        project_names.extend(_list(hints.get("project_names")))
        model_types.extend(_list(hints.get("model_types")))
        file_roles.extend(_list(hints.get("file_roles")))
        pcb_types.extend(_list(hints.get("pcb_types")))
        device_names.extend(_list(hints.get("device_names")))
        manufacturers.extend(_list(hints.get("manufacturers")))
    return {
        "project_names": _clean_list(project_names, limit=160)[:20],
        "model_types": _clean_list(model_types, limit=160)[:30],
        "file_roles": _clean_list(file_roles, limit=120)[:20],
        "pcb_types": _clean_list(pcb_types, limit=120)[:20],
        "device_names": _clean_list(device_names, limit=120)[:20],
        "manufacturers": _clean_list(manufacturers, limit=120)[:20],
    }


def _tool_evidence_project_files(tool_evidence: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for result in tool_evidence.get("attachment_parse_results") or []:
        if isinstance(result, dict) and result.get("evidence_role") == "program_file":
            files.append(str(result.get("name") or result.get("path") or ""))
    return _clean_list(files, limit=160)


def _tool_evidence_log_paths(tool_evidence: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for result in tool_evidence.get("attachment_parse_results") or []:
        if isinstance(result, dict) and result.get("evidence_role") == "log_package":
            files.append(str(result.get("name") or result.get("path") or ""))
    for result in tool_evidence.get("log_package_parse_results") or []:
        if not isinstance(result, dict):
            continue
        files.append(str(result.get("name") or result.get("path") or ""))
        for entry in result.get("entries") or []:
            if isinstance(entry, dict):
                files.append(str(entry.get("name") or ""))
    return _clean_list(files, limit=160)


def _tool_evidence_log_hints(tool_evidence: dict[str, Any]) -> dict[str, list[str]]:
    error_lines: list[str] = []
    error_codes: list[str] = []
    phase_hints: list[str] = []
    for result in tool_evidence.get("log_package_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hint_sources = []
        if isinstance(result.get("text_hints"), dict):
            hint_sources.append(result.get("text_hints") or {})
        for entry in result.get("entries") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text_hints"), dict):
                hint_sources.append(entry.get("text_hints") or {})
        for hints in hint_sources:
            error_lines.extend(_list(hints.get("error_lines")))
            error_codes.extend(_list(hints.get("error_codes")))
            phase_hints.extend(_list(hints.get("phase_hints")))
    return {
        "error_lines": _clean_list(error_lines, limit=240)[:12],
        "error_codes": _clean_list(error_codes, limit=80)[:20],
        "phase_hints": _clean_list(phase_hints, limit=80)[:20],
    }


def _tool_evidence_dmp_hints(tool_evidence: dict[str, Any]) -> dict[str, list[str]]:
    dump_kinds: list[str] = []
    bugcheck_hints: list[str] = []
    for result in tool_evidence.get("dmp_parse_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("dump_kind"):
            dump_kinds.append(str(result.get("dump_kind")))
        bugcheck_hints.extend(_list(result.get("bugcheck_hints")))
    return {
        "dump_kinds": _clean_list(dump_kinds, limit=120)[:10],
        "bugcheck_hints": _clean_list(bugcheck_hints, limit=240)[:20],
    }


def _tool_evidence_files_by_role(tool_evidence: dict[str, Any], roles: set[str]) -> list[str]:
    files: list[str] = []
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        if str(result.get("evidence_role") or "") in roles:
            files.append(str(result.get("name") or result.get("path") or ""))
    return _clean_list(files, limit=160)


def _tool_evidence_jira_links(tool_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in tool_evidence.get("jira_parse_results") or []:
        if not isinstance(result, dict):
            continue
        for url in result.get("urls") or []:
            if not isinstance(url, dict):
                continue
            value = str(url.get("url") or "")
            if value and value not in seen:
                seen.add(value)
                out.append(url)
    return out


def _tool_evidence_jira_ids(tool_evidence: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for result in tool_evidence.get("jira_parse_results") or []:
        if not isinstance(result, dict):
            continue
        ids.extend(_list(result.get("issue_keys")))
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        ids.extend(_list(hints.get("jira_ids")))
    return _clean_list(ids, limit=80)


def _tool_evidence_jira_hints(tool_evidence: dict[str, Any]) -> dict[str, list[str]]:
    titles: list[str] = []
    versions: list[str] = []
    sites: list[str] = []
    descriptions: list[str] = []
    comments: list[str] = []
    for result in tool_evidence.get("jira_parse_results") or []:
        if not isinstance(result, dict):
            continue
        titles.extend(_list(result.get("title_hints")))
        versions.extend(_list(result.get("version_hints")))
        sites.extend(_list(result.get("site_hints")))
        descriptions.extend(_list(result.get("description_hints")))
        comments.extend(_list(result.get("comment_hints")))
    return {
        "titles": _clean_list(titles, limit=240)[:20],
        "versions": _clean_list(versions, limit=80)[:20],
        "sites": _clean_list(sites, limit=80)[:20],
        "descriptions": _clean_list(descriptions, limit=1200)[:10],
        "comments": _clean_list(comments, limit=1200)[:10],
    }


def _tool_evidence_attachment_hints(tool_evidence: dict[str, Any]) -> dict[str, list[str]]:
    previews: list[str] = []
    error_lines: list[str] = []
    error_codes: list[str] = []
    phase_hints: list[str] = []
    urls: list[str] = []
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("text_preview_read") and result.get("text_preview"):
            previews.append(str(result.get("text_preview") or ""))
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        error_lines.extend(_list(hints.get("error_lines")))
        error_codes.extend(_list(hints.get("error_codes")))
        phase_hints.extend(_list(hints.get("phase_hints")))
        urls.extend(_list(hints.get("urls")))
    return {
        "text_previews": _clean_list(previews, limit=500)[:20],
        "error_lines": _clean_list(error_lines, limit=200)[:20],
        "error_codes": _clean_list(error_codes, limit=80)[:20],
        "phase_hints": _clean_list(phase_hints, limit=80)[:20],
        "urls": _clean_list(urls, limit=200)[:20],
    }


def _tool_evidence_image_hints(tool_evidence: dict[str, Any]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    formats: list[str] = []
    dimension_labels: list[str] = []
    for result in tool_evidence.get("image_parse_results") or []:
        if not isinstance(result, dict):
            continue
        item = {
            "name": str(result.get("name") or ""),
            "image_format": str(result.get("image_format") or ""),
            "width": result.get("width"),
            "height": result.get("height"),
            "megapixels": result.get("megapixels"),
            "header_read": bool(result.get("header_read")),
            "ocr_performed": bool(result.get("ocr_performed")),
            "pixels_read": bool(result.get("pixels_read")),
        }
        images.append(item)
        if item["image_format"]:
            formats.append(item["image_format"])
        if item.get("width") and item.get("height"):
            dimension_labels.append(f"{item['width']}x{item['height']}")
    return {
        "images": images[:50],
        "formats": _clean_list(formats, limit=40)[:20],
        "dimension_labels": _clean_list(dimension_labels, limit=40)[:20],
    }


def _tool_evidence_document_hints(tool_evidence: dict[str, Any]) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    formats: list[str] = []
    previews: list[str] = []
    page_count_labels: list[str] = []
    for result in tool_evidence.get("document_parse_results") or []:
        if not isinstance(result, dict):
            continue
        item = {
            "name": str(result.get("name") or ""),
            "document_format": str(result.get("document_format") or ""),
            "extension": str(result.get("extension") or ""),
            "pdf_version": str(result.get("pdf_version") or ""),
            "page_count_hint": result.get("page_count_hint"),
            "header_read": bool(result.get("header_read")),
            "text_preview_read": bool(result.get("text_preview_read")),
            "archive_manifest_read": bool(result.get("archive_manifest_read")),
            "ole_compound_file": bool(result.get("ole_compound_file")),
            "macros_executed": bool(result.get("macros_executed")),
            "formulas_evaluated": bool(result.get("formulas_evaluated")),
            "ocr_performed": bool(result.get("ocr_performed")),
        }
        documents.append(item)
        if item["document_format"]:
            formats.append(item["document_format"])
        if result.get("text_preview_read") and result.get("text_preview"):
            previews.append(str(result.get("text_preview") or ""))
        if result.get("page_count_hint"):
            page_count_labels.append(str(result.get("page_count_hint")))
    return {
        "documents": documents[:50],
        "formats": _clean_list(formats, limit=40)[:20],
        "text_previews": _clean_list(previews, limit=500)[:20],
        "page_count_labels": _clean_list(page_count_labels, limit=40)[:20],
    }


def _provided_roles_from_tool_evidence(tool_evidence: dict[str, Any]) -> list[str]:
    """Infer provided-evidence roles from a per-message W1 tool evidence pack."""

    roles: list[str] = []
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        role = str(result.get("evidence_role") or "")
        if role:
            roles.append(role)
        if result.get("text_preview_read"):
            roles.append("attachment_text_preview")
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        if hints.get("versions"):
            roles.append("software_version")
        if hints.get("ip_addresses"):
            roles.append("ip_config")
        if hints.get("jira_ids"):
            roles.append("jira_issue_key")
        if hints.get("error_lines") or hints.get("error_codes"):
            roles.append("attachment_error_hints")
        if hints.get("phase_hints"):
            roles.append("attachment_phase_hints")
    for result in tool_evidence.get("image_parse_results") or []:
        if not isinstance(result, dict):
            continue
        roles.append("sample_image_metadata")
        if result.get("width") and result.get("height"):
            roles.append("image_dimensions")
    for result in tool_evidence.get("document_parse_results") or []:
        if not isinstance(result, dict):
            continue
        roles.append("document_metadata")
        if result.get("text_preview_read"):
            roles.append("document_text_preview")
        if result.get("archive_manifest_read"):
            roles.append("document_manifest")
        if result.get("ole_compound_file"):
            roles.append("document_ole_header")
    for result in tool_evidence.get("proj_parse_results") or []:
        if not isinstance(result, dict):
            continue
        roles.append("proj_parsed")
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        if hints.get("versions"):
            roles.append("software_version")
        if hints.get("ip_addresses"):
            roles.append("ip_config")
        if hints.get("project_names"):
            roles.append("project_name")
        if hints.get("file_roles"):
            roles.append("proj_manifest")
        if "component_table" in set(_list(hints.get("file_roles"))):
            roles.append("proj_component_table")
        if hints.get("has_board_images"):
            roles.append("proj_board_images")
    for result in tool_evidence.get("log_package_parse_results") or []:
        if not isinstance(result, dict):
            continue
        roles.append("log_package_manifest")
        text_hints = result.get("text_hints") if isinstance(result.get("text_hints"), dict) else {}
        if text_hints.get("error_lines") or text_hints.get("error_codes"):
            roles.append("log_text_hints")
        if text_hints.get("phase_hints"):
            roles.append("log_phase_hints")
        if result.get("has_dmp"):
            roles.append("log_manifest_has_dmp")
        if result.get("has_evtx"):
            roles.append("log_manifest_has_evtx")
        if result.get("has_startup_log"):
            roles.append("log_manifest_has_startup_log")
        if result.get("has_dlog"):
            roles.append("log_manifest_has_dlog")
    for result in tool_evidence.get("dmp_parse_results") or []:
        if not isinstance(result, dict):
            continue
        roles.append("dmp_metadata")
        if result.get("looks_like_dmp"):
            roles.append("log_manifest_has_dmp")
        if result.get("bugcheck_hints"):
            roles.append("dmp_bugcheck_hints")
    for result in tool_evidence.get("jira_parse_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("urls"):
            roles.append("jira_link")
        if result.get("issue_keys"):
            roles.append("jira_issue_key")
    return roles


def _slot_matched_provided_roles(slot: str, provided_roles: list[str]) -> list[str]:
    allowed = SLOT_PROVIDED_ROLE_MAP.get(str(slot or ""), set())
    if not allowed:
        return []
    return sorted({role for role in provided_roles if role in allowed})


def _quality_score(quality: dict[str, Any]) -> float:
    return round(
        float(quality.get("evidence_strength") or 0.0) * 0.35
        + float(quality.get("diagnostic_relevance") or 0.0) * 0.4
        + float(quality.get("slot_specificity") or 0.0) * 0.25,
        4,
    )


def _provided_tool_roles(request: dict[str, Any], tool_evidence: dict[str, Any]) -> list[str]:
    provided_ids = set(_list(request.get("provided_evidence_message_ids")))
    if not provided_ids:
        return []
    roles: list[str] = []
    for result in tool_evidence.get("attachment_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        if str(source.get("message_id") or "") in provided_ids:
            role = str(result.get("evidence_role") or "")
            if role:
                roles.append(role)
            if result.get("text_preview_read"):
                roles.append("attachment_text_preview")
            hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
            if hints.get("versions"):
                roles.append("software_version")
            if hints.get("ip_addresses"):
                roles.append("ip_config")
            if hints.get("jira_ids"):
                roles.append("jira_issue_key")
            if hints.get("error_lines") or hints.get("error_codes"):
                roles.append("attachment_error_hints")
            if hints.get("phase_hints"):
                roles.append("attachment_phase_hints")
    for result in tool_evidence.get("image_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        if str(source.get("message_id") or "") not in provided_ids:
            continue
        roles.append("sample_image_metadata")
        if result.get("width") and result.get("height"):
            roles.append("image_dimensions")
    for result in tool_evidence.get("document_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        if str(source.get("message_id") or "") not in provided_ids:
            continue
        roles.append("document_metadata")
        if result.get("text_preview_read"):
            roles.append("document_text_preview")
        if result.get("archive_manifest_read"):
            roles.append("document_manifest")
        if result.get("ole_compound_file"):
            roles.append("document_ole_header")
    for result in tool_evidence.get("proj_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        source_message_id = str(source.get("message_id") or "")
        if source_message_id and source_message_id not in provided_ids:
            continue
        path = str(result.get("path") or "")
        if not (source_message_id or any(role == "program_file" for role in roles)):
            continue
        if path and (source_message_id or any(role == "program_file" for role in roles)):
            roles.append("proj_parsed")
        hints = result.get("key_hints") if isinstance(result.get("key_hints"), dict) else {}
        if hints.get("project_names"):
            roles.append("project_name")
        if hints.get("file_roles"):
            roles.append("proj_manifest")
        if "component_table" in set(_list(hints.get("file_roles"))):
            roles.append("proj_component_table")
        if hints.get("has_board_images"):
            roles.append("proj_board_images")
    for result in tool_evidence.get("log_package_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        if str(source.get("message_id") or "") not in provided_ids:
            continue
        roles.append("log_package_manifest")
        text_hints = result.get("text_hints") if isinstance(result.get("text_hints"), dict) else {}
        if text_hints.get("error_lines") or text_hints.get("error_codes"):
            roles.append("log_text_hints")
        if text_hints.get("phase_hints"):
            roles.append("log_phase_hints")
        if result.get("has_dmp"):
            roles.append("log_manifest_has_dmp")
        if result.get("has_evtx"):
            roles.append("log_manifest_has_evtx")
        if result.get("has_startup_log"):
            roles.append("log_manifest_has_startup_log")
        if result.get("has_dlog"):
            roles.append("log_manifest_has_dlog")
    for result in tool_evidence.get("dmp_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        if str(source.get("message_id") or "") not in provided_ids:
            continue
        roles.append("dmp_metadata")
        if result.get("looks_like_dmp"):
            roles.append("log_manifest_has_dmp")
        if result.get("bugcheck_hints"):
            roles.append("dmp_bugcheck_hints")
    for result in tool_evidence.get("jira_parse_results") or []:
        if not isinstance(result, dict):
            continue
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        source_message_id = str(source.get("message_id") or "")
        if source_message_id and source_message_id in provided_ids:
            roles.append("jira_link")
            if result.get("issue_keys"):
                roles.append("jira_issue_key")
            continue
        for url in result.get("urls") or []:
            if isinstance(url, dict) and str(url.get("message_id") or "") in provided_ids:
                roles.append("jira_link")
                break
    for provided in request.get("provided_evidence") or []:
        if not isinstance(provided, dict) or str(provided.get("message_id") or "") not in provided_ids:
            continue
        hints = provided.get("text_hints") if isinstance(provided.get("text_hints"), dict) else {}
        if hints.get("jira_ids"):
            roles.append("jira_issue_key")
        if hints.get("versions"):
            roles.append("software_version")
        if hints.get("ip_config"):
            roles.append("ip_config")
        if hints.get("log_paths"):
            roles.append("log_package")
        if hints.get("project_files"):
            roles.append("program_file")
        local_tool_evidence = provided.get("tool_evidence") if isinstance(provided.get("tool_evidence"), dict) else {}
        roles.extend(_provided_roles_from_tool_evidence(local_tool_evidence))
    return sorted(set(roles))


def _slots_for(text: str) -> list[str]:
    slots: list[str] = []
    for slot, keywords in SLOT_KEYWORDS:
        if any(kw in text for kw in keywords) and slot not in slots:
            slots.append(slot)
    return slots or ["other"]


def _condition_for(slot: str, text: str) -> str:
    lowered = text.lower()
    service_restart = any(k in text for k in (
        "重启相机服务", "重启服务", "重启主程序", "重启软件", "重启程序",
        "相机服务重启", "服务重启", "主程序重启", "软件重启", "程序重启",
    ))
    manual_reboot_action = any(k in text for k in ("重启设备", "设备重启", "重启机器", "机器重启", "重启电脑", "电脑重启")) and not any(
        k in text for k in ("自动重启", "突然重启", "无故重启", "异常重启", "莫名重启")
    )
    explicit_reboot_failure = any(k in text for k in ("自动重启", "突然重启", "无故重启", "异常重启", "莫名重启"))
    system_reboot = (
        "重启" in text
        and not service_restart
        and not manual_reboot_action
        and (
            explicit_reboot_failure
            or any(k in text for k in ("蓝屏", "工控机", "系统", "死机", "黑屏", "BugCheck", "bugcheck"))
        )
    )
    if slot in {"log_package", "error_message", "error_phase"} and (
        "蓝屏" in text or system_reboot or "dmp" in lowered or "dump" in lowered or "转存储" in text or "转储" in text
    ):
        return "dmp"
    if slot in {"log_package", "error_message", "error_phase"} and ("初始化" in text or "启动" in text or "startup" in lowered or "init" in lowered):
        return "startup/init log" if slot == "log_package" else "startup/init phase"
    if slot == "ip_config" or any(k in text for k in ("相机", "控制器", "IP", "ip", "网段")):
        return "camera/controller network config" if slot == "ip_config" else ""
    return ""


def _priority(slot: str, condition: str, text: str) -> int:
    if slot in {"log_package", "error_message"} and condition:
        return 1
    if slot in {"log_package", "software_version", "ip_config", "error_phase"}:
        return 2
    if slot == "other":
        return 5
    if "为什么" in text or "判断" in text or "定位" in text or "确认" in text:
        return 2
    return 3


def _required_info_quality(slot: str, condition: str, target_error_id: str, evidence_ids: list[str], text: str) -> dict[str, Any]:
    generic = slot == "log_package" and any(k in text for k in ("发日志", "提供日志", "上传日志")) and not condition and len(text) < 18
    specificity = 0.2 if slot == "other" else 0.65
    if condition:
        specificity += 0.2
    if any(k in text for k in ("DLOG", "诊断数据", "主程序版本", "算法包版本", "IP", "报错码")):
        specificity += 0.1
    relevance = 0.35
    if any(k in text for k in ("判断", "定位", "确认", "排查", "初始化", "相机", "控制器", "蓝屏", "重启")):
        relevance += 0.35
    if target_error_id:
        relevance += 0.15
    evidence = min(1.0, 0.25 + 0.2 * len(evidence_ids)) if evidence_ids else 0.0
    if generic:
        specificity -= 0.25
        relevance -= 0.25
    score = round(max(0.0, min(1.0, specificity)) * 0.3 + max(0.0, min(1.0, relevance)) * 0.4 + evidence * 0.3, 4)
    return {
        "slot_specificity": round(max(0.0, min(1.0, specificity)), 4),
        "diagnostic_relevance": round(max(0.0, min(1.0, relevance)), 4),
        "evidence_strength": round(evidence, 4),
        "score": score,
        "generic_request": generic,
    }




def _ask_info_target_error_id(matched: dict[str, Any] | None, slot: str, condition: str, focus_text: str, context_text: str) -> str:
    """Return a safe target Error for required-info, or blank for review-only.

    The write side must not append ask-info slots to an unrelated existing Error
    just because a long chat paragraph had token overlap.  For high-collision
    asks such as dmp/blue-screen/reboot logs, require the matched KG label/id
    to be in the same fault family as the ask clause itself.
    """

    if not isinstance(matched, dict):
        return ""
    target_error_id = str(matched.get("error_id") or "")
    if not target_error_id:
        return ""
    label = str(matched.get("label") or "")
    haystack = f"{target_error_id} {label}".lower()
    text = f"{focus_text} {context_text}"
    lowered = text.lower()
    asks_dump = condition == "dmp" or any(k in lowered for k in ("dmp", "dump", "memory.dmp")) or any(k in text for k in ("蓝屏", "转存储", "转储"))
    asks_reboot = any(k in text for k in ("自动重启", "突然重启", "无故重启", "异常重启", "莫名重启"))
    if asks_dump:
        if "wifi-card" in haystack or "无线网卡" in haystack or "usb" in haystack:
            mentions_usb = any(k in lowered for k in ("usb", "wireless", "wifi")) or any(k in text for k in ("无线网卡", "无线网", "网卡", "USB", "usb口"))
            negates_usb = any(k in text for k in ("usb问题目前没问题", "USB问题目前没问题", "usb目前没问题", "USB目前没问题", "无线网卡目前没问题"))
            return target_error_id if mentions_usb and not negates_usb else ""
        if any(k in haystack for k in ("blue-screen", "bsod", "reboot", "蓝屏", "重启", "unexpected-reboot", "wifi-card")):
            return target_error_id
        return ""
    if asks_reboot and not any(k in haystack for k in ("reboot", "重启", "blue-screen", "蓝屏", "bsod")):
        return ""
    if slot == "ip_config" and any(k in text for k in ("相机", "控制器", "网段", "ping", "Ping")):
        if not any(k in haystack for k in ("ip", "network", "camera", "controller", "相机", "控制器", "网络", "连不上", "连接")):
            return ""
    if any(k in text for k in ("硬盘", "磁盘", "分区", "Disk", "disk", "partition")):
        if not any(k in haystack for k in ("disk", "硬盘", "磁盘", "分区", "507", "space", "空间", "partition")):
            return ""
    if condition == "startup/init log" or condition == "startup/init phase":
        if not any(k in haystack for k in ("init", "startup", "boot", "初始化", "启动", "开机")):
            return ""
    if slot == "software_version":
        version_context = any(k in text for k in ("升级", "版本", "兼容", "认证配置", "算法包", "主程序版本", "软件版本", "固件", "驱动"))
        version_target = any(k in haystack for k in ("version", "版本", "upgrade", "升级", "compat", "兼容", "config", "配置", "license", "认证", "login", "登录"))
        return target_error_id if version_context and version_target else ""
    if slot == "environment":
        if any(k in text for k in ("堆损坏", "0xc0000374", "ntdll", "STATUS_HEAP_CORRUPTION", "应用程序发生")):
            return target_error_id if any(k in haystack for k in ("crash", "崩溃", "闪退", "heap", "堆", "white-screen", "白屏", "app")) else ""
        if any(k in text for k in ("硬盘", "磁盘", "分区", "空间不足", "系统盘")):
            return target_error_id if any(k in haystack for k in ("disk", "硬盘", "磁盘", "507", "空间")) else ""
    if slot == "error_message":
        if any(k in text for k in ("截图没截全", "有没有空", "找找原因", "很忙")) and not any(k in text for k in ("报错", "错误码", "异常代码", "弹窗")):
            return ""
    return target_error_id

def _node_pk(node: dict[str, Any]) -> str:
    node_type = str(node.get("type") or "")
    key = PRIMARY_KEYS.get(node_type, "id")
    return str(node.get(key) or node.get("id") or "")


def _validate_schema(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    return validate_nodes_edges(nodes, edges)


def _context_evidence_policy(episode: dict[str, Any]) -> str:
    return "w7_promoted_case_evidence.v1" if episode.get("case_evidence_messages") else "current_episode_only.v1"


class KnowledgeExtractionAgent:
    """W2: extract episode semantics, then normalize to strict KG schema draft."""

    def __init__(
        self,
        store: KGStore | None = None,
        *,
        match_threshold: float = 4.0,
        deepseek_enabled: bool | None = None,
        w2_mode: str = "legacy_only",
    ) -> None:
        self.store = store
        self.match_threshold = match_threshold
        self.deepseek_enabled = bool(os.environ.get("DEBUG_AGENT_SYSTEM_W2_DEEPSEEK") == "1") if deepseek_enabled is None else bool(deepseek_enabled)
        self.w2_mode = w2_mode
        self._match_cache: dict[str, dict[str, Any] | None] = {}

    def extract(self, summary_or_episode: dict[str, Any], *, w2_mode: str | None = None) -> dict[str, Any]:
        episode = _episode_from_summary(summary_or_episode)
        mode = str(w2_mode or self.w2_mode or "legacy_only")
        # native_v2 uses one complete Prompt-A call.  Do not first invoke the
        # older leaf-enrichment prompt and then ask a second model call to
        # reinterpret its output.
        semantics = self.extract_semantics(
            episode,
            deepseek_enrich=mode not in {"native_v2", "prompt_first"},
        )
        legacy_candidate = self.normalize_to_kg_schema(semantics)
        legacy_candidate["w2_mode"] = mode
        if mode == "legacy_only":
            return legacy_candidate

        case_understanding, extraction_metadata = self._case_understanding_card(
            semantics,
            legacy_candidate=legacy_candidate,
            prompt_required=mode == "prompt_first",
        )
        case_understanding["source_candidate_id"] = str(legacy_candidate.get("candidate_id") or legacy_candidate.get("id") or "")
        candidate_draft_v2 = build_candidate_draft_v2_from_case_understanding(case_understanding)
        native_bundle = build_v2_bundle_from_candidate_draft(candidate_draft_v2)
        native_bundle["extraction_metadata"] = {
            "context_evidence_policy": _context_evidence_policy(episode),
            "untrusted_case_context_message_count": int(semantics.get("untrusted_case_context_message_count") or 0),
            **extraction_metadata,
        }
        legacy_candidate["case_understanding_extraction"] = extraction_metadata
        legacy_candidate["case_understanding_card"] = case_understanding
        legacy_candidate["case_understanding_card_schema_valid"] = bool(case_understanding.get("schema_valid"))
        legacy_candidate["case_understanding_card_schema_issues"] = list(case_understanding.get("schema_issues") or [])
        legacy_candidate["candidate_draft_v2"] = candidate_draft_v2
        legacy_candidate["candidate_draft_v2_schema_valid"] = bool(candidate_draft_v2.get("schema_valid"))
        legacy_candidate["candidate_draft_v2_schema_issues"] = list(candidate_draft_v2.get("schema_issues") or [])
        legacy_candidate["candidate_draft_v2_normalized_bundle"] = native_bundle
        legacy_candidate["candidate_draft_v2_bundle_schema_valid"] = bool(native_bundle.get("schema_valid"))
        legacy_candidate["candidate_draft_v2_bundle_schema_issues"] = list(native_bundle.get("schema_issues") or [])
        production_issues: list[str] = []
        if not legacy_candidate["case_understanding_card_schema_valid"]:
            production_issues.append("native_case_understanding_invalid")
        if not legacy_candidate["candidate_draft_v2_schema_valid"]:
            production_issues.append("native_candidate_draft_invalid")
            production_issues.extend(
                f"native_candidate_draft:{issue}"
                for issue in legacy_candidate["candidate_draft_v2_schema_issues"]
            )
        if not legacy_candidate["candidate_draft_v2_bundle_schema_valid"]:
            production_issues.append("native_bundle_invalid")
            production_issues.extend(
                f"native_bundle:{issue}"
                for issue in legacy_candidate["candidate_draft_v2_bundle_schema_issues"]
            )
        legacy_candidate["production_schema_valid"] = not production_issues
        legacy_candidate["production_schema_issues"] = sorted(set(production_issues))
        if mode in {"native_v2", "prompt_first"} and production_issues:
            legacy_candidate["schema_valid"] = False
            legacy_candidate["schema_issues"] = sorted(set([*legacy_candidate.get("schema_issues", []), *production_issues]))
        return legacy_candidate

    def extract_semantics(self, episode: dict[str, Any], *, deepseek_enrich: bool = True) -> dict[str, Any]:
        extracted = _summary_extracted(episode)
        text = _semantic_text(episode, extracted)
        source_id = str(episode.get("episode_id") or episode.get("thread_id") or episode.get("source_thread_id") or "")
        thread_id = str(episode.get("thread_id") or episode.get("source_thread_id") or source_id)
        fault_message_texts = _message_texts([x for x in episode.get("fault_description_messages") or [] if isinstance(x, dict)])
        context_message_texts = _message_texts([x for x in episode.get("case_context_messages") or [] if isinstance(x, dict)])
        promoted_message_texts = _message_texts([x for x in episode.get("case_evidence_messages") or [] if isinstance(x, dict)])
        sentence_roles = _sentence_role_records(episode)
        fault_text = " ".join(fault_message_texts)
        raw_symptom = _clean_chat_text(extracted.get("fault_focus_text") or extracted.get("symptom_raw") or fault_text or _first_sentence(text), 500)
        current_symptom = _best_episode_fault_label(fault_message_texts, raw_symptom, fault_text or raw_symptom, limit=160)
        if current_symptom and _has_fault_label_signal(current_symptom) and not _label_is_handoff_noise(current_symptom, fault_text):
            symptom = current_symptom
        else:
            symptom = _best_episode_fault_label(fault_message_texts, raw_symptom, text, limit=160)
        raw_action_messages = _message_texts([x for x in episode.get("diagnostic_chain_messages") or [] if isinstance(x, dict)])
        raw_resolution_messages = _message_texts([x for x in episode.get("resolution_messages") or [] if isinstance(x, dict)])
        role_action_messages = [
            str(item.get("text") or "")
            for item in sentence_roles
            if item.get("role") in {"diagnostic_action", "observed_outcome"}
            and (
                item.get("source_role") != "current_fault"
                or _is_field_report_action_text(str(item.get("text") or ""))
            )
        ]
        promoted_role_action_messages = [
            str(item.get("text") or "")
            for item in sentence_roles
            if item.get("role") in {"diagnostic_action", "observed_outcome"}
            and item.get("source_role") == "w7_promoted"
        ]
        role_outcome_messages = [
            str(item.get("text") or "")
            for item in sentence_roles
            if item.get("role") == "observed_outcome"
        ]
        # Promoted evidence remains available to Prompt-A and outcome/trace
        # review, but it must not silently create a second executable action
        # in a neighbouring episode.  Keep it as an explicit audit hint.
        promoted_action_messages = _action_sentences(promoted_role_action_messages, limit=500)
        promoted_resolution_messages = _solution_sentences(promoted_message_texts, limit=500)
        # ``debug_actions`` is W1's structured action projection.  Re-running
        # prose sentence-role classification over it used to discard concise
        # but valid operations such as "使用配置文件" and "验证启动".  Preserve
        # structured actions here; native-v2 still applies its executable,
        # relevance and atomicity checks before materialising them.
        extracted_actions = [
            _clean_chat_text(value, 500)
            for value in _list(extracted.get("debug_actions"))
            if _clean_chat_text(value, 500)
        ]
        conclusion = _clean_chat_text(
            extracted.get("conclusion")
            or extracted.get("key_conclusion")
            or " ".join([*raw_resolution_messages, *role_outcome_messages, *promoted_resolution_messages])
            or " ".join(_solution_sentences([*raw_action_messages, *promoted_message_texts, *extracted_actions], limit=500)[:2]),
            500,
        )
        action_messages = _action_sentences(role_action_messages, limit=500)
        # W1 ``debug_actions`` are structural hints, not trusted executable
        # actions.  They may still contain hand-off requests, fault text,
        # questions, daily reports, or result statements.  Send both sources
        # through the same review-grade filter before creating checks.
        actions = _merge_action_candidates(extracted_actions, action_messages)
        artifacts = extracted.get("artifacts") if isinstance(extracted.get("artifacts"), dict) else {}
        tool_evidence = _tool_evidence(extracted)
        sites = _list(extracted.get("sites")) or _list(artifacts.get("sites"))
        raw_versions = _list(extracted.get("versions")) or _list(artifacts.get("versions"))
        raw_versions = _clean_list([*raw_versions, *_tool_evidence_versions(tool_evidence)], limit=80)
        versions = [version for version in raw_versions if _is_likely_software_version(version, text)]
        devices = _list(extracted.get("devices")) or _list(artifacts.get("devices"))
        log_paths = _clean_list([*(_list(extracted.get("log_paths")) or _list(artifacts.get("log_paths"))), *_tool_evidence_log_paths(tool_evidence)], limit=200)
        log_hints = _tool_evidence_log_hints(tool_evidence)
        project_hints = _tool_evidence_project_hints(tool_evidence)
        jira_hints = _tool_evidence_jira_hints(tool_evidence)
        linked_jira = [item for item in extracted.get("linked_jira_evidence") or [] if isinstance(item, dict)]
        if linked_jira:
            jira_hints = {
                **jira_hints,
                "titles": _clean_list([*(jira_hints.get("titles") or []), *(str(item.get("summary") or "") for item in linked_jira)], limit=80),
                "versions": _clean_list([
                    *(jira_hints.get("versions") or []),
                    *(
                        value
                        for item in linked_jira
                        for value in re.findall(r"(?<![\d.])(?:v)?\d{1,2}\.\d+(?:\.\d+){0,2}(?![\d.])", str(item.get("summary") or ""), re.IGNORECASE)
                    ),
                ], limit=80),
            }
        attachment_hints = _tool_evidence_attachment_hints(tool_evidence)
        image_hints = _tool_evidence_image_hints(tool_evidence)
        document_hints = _tool_evidence_document_hints(tool_evidence)
        sites = _clean_list([*sites, *(jira_hints.get("sites") or [])], limit=80)
        project_files = _clean_list([*(_list(extracted.get("project_files")) or _list(artifacts.get("project_files"))), *_tool_evidence_project_files(tool_evidence)], limit=200)
        sample_images = _clean_list([*(_list(extracted.get("sample_images")) or _list(artifacts.get("sample_images"))), *_tool_evidence_files_by_role(tool_evidence, {"sample_image"})], limit=200)
        environment_files = _clean_list([*(_list(extracted.get("environment_files")) or _list(artifacts.get("environment_files"))), *_tool_evidence_files_by_role(tool_evidence, {"environment"})], limit=200)
        data_files = _clean_list([*(_list(extracted.get("data_files")) or _list(artifacts.get("data_files"))), *_tool_evidence_files_by_role(tool_evidence, {"data_file"})], limit=200)
        jira_links = list(extracted.get("jira_links") or artifacts.get("jira_links") or [])
        jira_links = [*jira_links, *_tool_evidence_jira_links(tool_evidence)]
        jira_ids = _clean_list([
            *(_list(extracted.get("jira_ids")) or _list(artifacts.get("jira_ids"))),
            *_tool_evidence_jira_ids(tool_evidence),
            *(str(item.get("issue_key") or "") for item in linked_jira),
        ], limit=80)
        attachment_evidence = list(artifacts.get("attachment_evidence") or [])
        evidence_ids = _list(episode.get("evidence_message_ids")) or _list(extracted.get("evidence_message_ids"))
        label = _best_fault_sentence(_remove_help_tail(symptom or conclusion or text), symptom or conclusion or text, 80) or f"群聊候选 {source_id or thread_id}"
        category = _classify_category(text)
        matched = self._match_existing(symptom or text)
        candidate_id = _candidate_id(source_id or thread_id, label)
        semantics = {
            "candidate_id": candidate_id,
            "source_episode_id": source_id,
            "source_thread_id": thread_id,
            "episode": episode,
            "label": label,
            "category": category,
            "symptom_raw": symptom,
            "conclusion": conclusion,
            "debug_actions": actions[:30],
            "sites": sites[:5],
            "versions": versions[:5],
            "devices": devices[:10],
            "log_paths": log_paths[:20],
            "log_error_hints": (log_hints.get("error_lines") or [])[:12],
            "log_error_codes": (log_hints.get("error_codes") or [])[:20],
            "log_phase_hints": (log_hints.get("phase_hints") or [])[:20],
            "project_files": project_files[:20],
            "project_names": (project_hints.get("project_names") or [])[:20],
            "project_model_types": (project_hints.get("model_types") or [])[:30],
            "project_file_roles": (project_hints.get("file_roles") or [])[:20],
            "project_pcb_types": (project_hints.get("pcb_types") or [])[:20],
            "project_device_names": (project_hints.get("device_names") or [])[:20],
            "project_manufacturers": (project_hints.get("manufacturers") or [])[:20],
            "sample_images": sample_images[:20],
            "sample_image_metadata": (image_hints.get("images") or [])[:50],
            "sample_image_dimensions": (image_hints.get("dimension_labels") or [])[:20],
            "sample_image_formats": (image_hints.get("formats") or [])[:20],
            "document_metadata": (document_hints.get("documents") or [])[:50],
            "document_formats": (document_hints.get("formats") or [])[:20],
            "document_text_previews": (document_hints.get("text_previews") or [])[:20],
            "document_page_count_hints": (document_hints.get("page_count_labels") or [])[:20],
            "environment_files": environment_files[:20],
            "data_files": data_files[:20],
            "jira_titles": (jira_hints.get("titles") or [])[:20],
            "jira_site_hints": (jira_hints.get("sites") or [])[:20],
            "jira_version_hints": (jira_hints.get("versions") or [])[:20],
            "attachment_text_previews": (attachment_hints.get("text_previews") or [])[:20],
            "attachment_error_hints": (attachment_hints.get("error_lines") or [])[:20],
            "attachment_error_codes": (attachment_hints.get("error_codes") or [])[:20],
            "attachment_phase_hints": (attachment_hints.get("phase_hints") or [])[:20],
            "jira_links": jira_links[:20],
            "jira_ids": jira_ids[:20],
            "attachment_evidence": attachment_evidence[:50],
            "tool_evidence": tool_evidence,
            "ip_configs": _tool_evidence_ip_configs(tool_evidence)[:20],
            "evidence_ids": evidence_ids[:30],
            "source_offsets": episode.get("source_offsets") or extracted.get("source_offsets") or [],
            "matched_existing_error": matched,
            "confidence": self._confidence(symptom=symptom, actions=actions, conclusion=conclusion, evidence_ids=evidence_ids, log_paths=log_paths, matched=matched),
            "semantic_text": text,
            "sentence_roles": sentence_roles,
            "promoted_action_hints": promoted_action_messages,
            "untrusted_case_context_message_count": len(context_message_texts),
            "promoted_case_evidence_message_count": len(promoted_message_texts),
            "linked_jira_evidence": linked_jira,
            "review_context": extracted.get("review_context") if isinstance(extracted.get("review_context"), (dict, list)) else {},
            # Compatibility only.  New paths consume review_context; historical
            # W2 artifacts still expose sop_background.
            "sop_background": extracted.get("review_context") if isinstance(extracted.get("review_context"), (dict, list)) else (extracted.get("sop_background") if isinstance(extracted.get("sop_background"), (dict, list)) else {}),
        }
        return self._maybe_deepseek_enrich(semantics) if deepseek_enrich else semantics

    def _case_understanding_card(
        self,
        semantics: dict[str, Any],
        *,
        legacy_candidate: dict[str, Any],
        prompt_required: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run Prompt A when configured, otherwise expose an explicit fallback.

        The fallback remains available for offline regression and migration,
        but is labelled review-only.  It is never presented as evidence that
        the prompt-first extractor generalized.
        """

        fallback_reason = "deepseek_disabled"
        if self.deepseek_enabled:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                fallback_reason = "missing_DEEPSEEK_API_KEY"
            else:
                try:
                    card, attempts, corrections = _extract_prompt_case_understanding_with_repair(
                        semantics,
                        api_key=api_key,
                    )
                    return card, {
                        "case_understanding_source": "deepseek_prompt_a",
                        "case_understanding_prompt_version": CASE_UNDERSTANDING_PROMPT_VERSION,
                        "case_understanding_prompt_attempts": attempts,
                        "case_understanding_corrections": corrections,
                        "deterministic_compat_fallback": False,
                        "requires_human_review": True,
                    }
                except Exception as exc:  # noqa: BLE001 - fallback is explicit and review-only
                    fallback_reason = f"{type(exc).__name__}:{exc}"

        fallback = build_case_understanding_card_from_semantics(
            semantics,
            legacy_candidate=legacy_candidate,
        )
        fallback["extraction_source"] = "deterministic_compat_fallback"
        fallback["prompt_version"] = CASE_UNDERSTANDING_PROMPT_VERSION
        fallback["requires_human_review"] = True
        fallback["fallback_reason"] = fallback_reason
        metadata = {
            "case_understanding_source": "deterministic_compat_fallback",
            "case_understanding_prompt_version": CASE_UNDERSTANDING_PROMPT_VERSION,
            "case_understanding_prompt_attempts": 0,
            "case_understanding_corrections": [],
            "deterministic_compat_fallback": True,
            "requires_human_review": True,
            "fallback_reason": fallback_reason,
        }
        if prompt_required:
            # Strict prompt_first mode is intended for production evaluation:
            # an unavailable/invalid LLM result must not masquerade as success.
            fallback["schema_valid"] = False
            fallback["schema_issues"] = sorted(set([
                *fallback.get("schema_issues", []),
                "prompt_first_extraction_unavailable",
            ]))
        return fallback, metadata

    def normalize_to_kg_schema(self, semantics: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(semantics.get("candidate_id") or "chatcand:unknown")
        digest = candidate_id.replace("chatcand:", "") or hashlib.sha1(candidate_id.encode("utf-8")).hexdigest()[:12]
        matched = semantics.get("matched_existing_error") if isinstance(semantics.get("matched_existing_error"), dict) else None
        matched_error_id = str(matched.get("error_id") if matched else "")
        error_id = f"err:candidate-{digest}"
        raw_label = str(semantics.get("label") or error_id)
        label = _trim_fault_fact(raw_label, 80)
        if _label_is_handoff_noise(label, str(semantics.get("semantic_text") or "")):
            label = ""
        nodes: list[dict[str, Any]] = [{
            "type": "Error",
            "id": error_id,
            "error_id": error_id,
            "label": label or "群聊噪声/待人工确认",
            "category": semantics.get("category") or "系统与软件异常",
            "symptom": _clean_chat_text(_remove_help_tail(semantics.get("symptom_raw") or label), 500),
            "source": "chat",
            "source_title": label,
            "source_episode_id": semantics.get("source_episode_id") or "",
            "source_thread_id": semantics.get("source_thread_id") or "",
            "entry_role": "case_variant" if matched_error_id else "canonical",
            "canonical_error_id": matched_error_id,
            "occurrence_count": 1,
            "proposal_only": True,
        }]
        for idx, action in enumerate(_list(semantics.get("debug_actions"))[:5], 1):
            check_id = f"check:candidate-{digest}:{idx}"
            nodes.append({
                "type": "DiagnosticCheck",
                "id": check_id,
                "check_id": check_id,
                "label": _clean_chat_text(action, 120) or f"群聊排查步骤 {idx}",
                "how_to_check": _clean_chat_text(action, 500) or f"按群聊证据执行排查步骤 {idx}",
                "step_order": idx,
                "source": "chat",
                "source_title": label,
                "proposal_only": True,
            })
        conclusion = _clean_chat_text(semantics.get("conclusion"), 500)
        if conclusion:
            solution_id = f"solution:candidate-{digest}:1"
            nodes.append({
                "type": "Solution",
                "id": solution_id,
                "solution_id": solution_id,
                "content": conclusion,
                "method": self._solution_method(conclusion),
                "evidence_level": "case_chat_evidence",
                "source": "chat",
                "source_title": label,
                "proposal_only": True,
            })
        for site in _list(semantics.get("sites"))[:5]:
            site_id = f"site:{_slug(site)}"
            nodes.append({
                "type": "Site",
                "id": site_id,
                "site_id": site_id,
                "name": site,
                "short_name": site,
                "source": "chat",
                "proposal_only": True,
            })
        for version in _list(semantics.get("versions"))[:5]:
            version_id = f"version:{_slug(version)}"
            nodes.append({
                "type": "SoftwareVersion",
                "id": version_id,
                "version_id": version_id,
                "version_string": version,
                "source": "chat",
                "proposal_only": True,
            })

        diagnostic_trace = _merge_llm_trace(semantics, _diagnostic_trace_candidate(semantics, error_id, digest, nodes))
        _sync_trace_steps_to_check_nodes(nodes, diagnostic_trace)
        diagnostic_outcomes = _merge_llm_outcomes(
            semantics,
            _diagnostic_outcome_candidates(semantics, error_id, digest, nodes, conclusion),
            error_id,
            digest,
            nodes,
        )
        nodes.extend([diagnostic_trace, *diagnostic_outcomes])

        edges: list[dict[str, Any]] = []
        solution_ids = [_node_pk(node) for node in nodes if node.get("type") == "Solution"]
        verified_solution_ids = _verified_solution_ids(solution_ids, diagnostic_outcomes, conclusion)
        for node in nodes:
            node_type = node.get("type")
            node_id = _node_pk(node)
            if node_type == "DiagnosticCheck":
                edges.append({"from": error_id, "to": node_id, "relation": "has_check", "proposal_only": True})
                for solution_id in verified_solution_ids:
                    edges.append({"from": node_id, "to": solution_id, "relation": "resolved_by", "proposal_only": True})
            elif node_type == "Site":
                edges.append({"from": error_id, "to": node_id, "relation": "occurs_at", "proposal_only": True})
            elif node_type == "SoftwareVersion":
                edges.append({"from": error_id, "to": node_id, "relation": "affects_version", "proposal_only": True})
            elif node_type == "DiagnosticTrace":
                edges.append({"from": error_id, "to": node_id, "relation": "has_trace", "proposal_only": True})
            elif node_type == "DiagnosticOutcome":
                edges.append({"from": error_id, "to": node_id, "relation": "has_outcome", "proposal_only": True})
                if node.get("target_check_id"):
                    edges.append({"from": node_id, "to": node["target_check_id"], "relation": "outcome_check", "proposal_only": True})
                if node.get("target_solution_id"):
                    edges.append({"from": node_id, "to": node["target_solution_id"], "relation": "outcome_solution", "proposal_only": True})
        if matched_error_id and matched_error_id != error_id:
            edges.append({"from": error_id, "to": matched_error_id, "relation": "alias_of", "proposal_only": True})
        case_variant_candidate = _merge_llm_case_variant(semantics, _case_variant_candidate(semantics, error_id, matched))
        if nodes and nodes[0].get("type") == "Error":
            llm_label = _trim_fault_fact(str(case_variant_candidate.get("label") or ""), 80)
            if llm_label and not _label_is_handoff_noise(llm_label, str(semantics.get("semantic_text") or "")):
                label = llm_label
                nodes[0]["label"] = llm_label
                nodes[0]["source_title"] = llm_label
                nodes[0]["symptom"] = _clean_chat_text(semantics.get("symptom_raw") or llm_label, 500)
            for key in ("category", "subsystem", "scenario", "escalation_target"):
                if case_variant_candidate.get(key):
                    nodes[0][key] = case_variant_candidate[key]
        issues = _validate_schema(nodes, edges)
        issues.extend(semantic_schema_issues({"nodes": nodes, "edges": edges, "diagnostic_outcomes": diagnostic_outcomes}))
        required_info_candidates = [*self._required_info_candidates(semantics, matched), *_llm_required_info_candidates(semantics, matched)]
        candidate = {
            "type": "SchemaValidCandidate",
            "candidate_type": "ChatKnowledgeCandidate",
            "candidate_id": candidate_id,
            "id": candidate_id,
            "status": "pending_review",
            "auto_ingest": False,
            "proposal_only": True,
            "source": semantics.get("source_episode_id") or semantics.get("source_thread_id") or "",
            "source_episode_id": semantics.get("source_episode_id") or "",
            "source_thread_id": semantics.get("source_thread_id") or "",
            "label": label or "群聊噪声/待人工确认",
            "category": semantics.get("category") or "系统与软件异常",
            "symptom_raw": semantics.get("symptom_raw") or "",
            "conclusion": conclusion,
            "debug_actions": _list(semantics.get("debug_actions")),
            "sites": _list(semantics.get("sites")),
            "versions": _list(semantics.get("versions")),
            "devices": _list(semantics.get("devices")),
            "log_paths": _list(semantics.get("log_paths")),
            "log_error_hints": _list(semantics.get("log_error_hints")),
            "log_error_codes": _list(semantics.get("log_error_codes")),
            "log_phase_hints": _list(semantics.get("log_phase_hints")),
            "project_files": _list(semantics.get("project_files")),
            "project_names": _list(semantics.get("project_names")),
            "project_model_types": _list(semantics.get("project_model_types")),
            "project_file_roles": _list(semantics.get("project_file_roles")),
            "project_pcb_types": _list(semantics.get("project_pcb_types")),
            "project_device_names": _list(semantics.get("project_device_names")),
            "project_manufacturers": _list(semantics.get("project_manufacturers")),
            "sample_images": _list(semantics.get("sample_images")),
            "sample_image_metadata": list(semantics.get("sample_image_metadata") or []),
            "sample_image_dimensions": _list(semantics.get("sample_image_dimensions")),
            "sample_image_formats": _list(semantics.get("sample_image_formats")),
            "document_metadata": list(semantics.get("document_metadata") or []),
            "document_formats": _list(semantics.get("document_formats")),
            "document_text_previews": _list(semantics.get("document_text_previews")),
            "document_page_count_hints": _list(semantics.get("document_page_count_hints")),
            "environment_files": _list(semantics.get("environment_files")),
            "data_files": _list(semantics.get("data_files")),
            "jira_titles": _list(semantics.get("jira_titles")),
            "jira_site_hints": _list(semantics.get("jira_site_hints")),
            "jira_version_hints": _list(semantics.get("jira_version_hints")),
            "attachment_text_previews": _list(semantics.get("attachment_text_previews")),
            "attachment_error_hints": _list(semantics.get("attachment_error_hints")),
            "attachment_error_codes": _list(semantics.get("attachment_error_codes")),
            "attachment_phase_hints": _list(semantics.get("attachment_phase_hints")),
            "jira_links": list(semantics.get("jira_links") or []),
            "jira_ids": _list(semantics.get("jira_ids")),
            "attachment_evidence": list(semantics.get("attachment_evidence") or []),
            "tool_evidence": semantics.get("tool_evidence") or {},
            "ip_configs": _list(semantics.get("ip_configs")),
            "evidence_ids": _list(semantics.get("evidence_ids")),
            "source_offsets": semantics.get("source_offsets") or [],
            "matched_existing_error": matched,
            "deepseek_extraction": semantics.get("deepseek_extraction") or {},
            "sentence_roles": list(semantics.get("sentence_roles") or []),
            "case_variant_candidate": case_variant_candidate,
            "diagnostic_trace": diagnostic_trace,
            "diagnostic_outcomes": diagnostic_outcomes,
            "split_decision": _split_decision(semantics),
            "nodes": nodes,
            "edges": edges,
            "required_info_candidates": required_info_candidates,
            "schema_valid": not issues,
            "schema_issues": issues,
            "confidence": semantics.get("confidence") or 0.0,
            "episode": semantics.get("episode") or {},
            "observability": {
                "agent_id": "W2",
                "episode_id": semantics.get("source_episode_id") or "",
                "thread_id": semantics.get("source_thread_id") or "",
                "matched_existing": bool(matched),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "required_info_candidate_count": len(required_info_candidates),
                "diagnostic_outcome_count": len(diagnostic_outcomes),
                "has_diagnostic_trace": bool(diagnostic_trace),
                "schema_valid": not issues,
                "deepseek_enabled": self.deepseek_enabled,
                "deepseek_used": bool(semantics.get("deepseek_extraction")) and not bool(semantics.get("deepseek_error")),
                "deepseek_error": str(semantics.get("deepseek_error") or ""),
                "context_evidence_policy": _context_evidence_policy(semantics.get("episode") or {}),
                "untrusted_case_context_message_count": int(semantics.get("untrusted_case_context_message_count") or 0),
            },
        }
        return candidate

    def _maybe_deepseek_enrich(self, semantics: dict[str, Any]) -> dict[str, Any]:
        if not self.deepseek_enabled:
            return semantics
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            enriched = dict(semantics)
            enriched["deepseek_error"] = "missing_DEEPSEEK_API_KEY"
            return enriched
        try:
            extraction = _call_deepseek_w2_extractor_with_hard_timeout(semantics, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - optional extractor must not block deterministic W2
            enriched = dict(semantics)
            enriched["deepseek_error"] = f"{type(exc).__name__}:{exc}"
            return enriched
        extraction = _sanitize_deepseek_extraction(extraction, semantics)
        issues = _validate_deepseek_extraction(extraction)
        enriched = dict(semantics)
        if issues:
            enriched["deepseek_error"] = "schema_invalid:" + ",".join(issues)
            enriched["deepseek_invalid_extraction"] = {"raw": extraction, "schema_issues": issues}
            return enriched
        enriched["deepseek_extraction"] = extraction
        return enriched

    def _required_info_candidates(self, semantics: dict[str, Any], matched: dict[str, Any] | None) -> list[dict[str, Any]]:
        episode = semantics.get("episode") if isinstance(semantics.get("episode"), dict) else {}
        extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
        requests = [x for x in extracted.get("missing_info_requests") or [] if isinstance(x, dict)]
        tool_evidence = semantics.get("tool_evidence") if isinstance(semantics.get("tool_evidence"), dict) else {}
        out: list[dict[str, Any]] = []
        source_episode_id = str(semantics.get("source_episode_id") or "")
        source_thread_id = str(semantics.get("source_thread_id") or "")
        for request in requests:
            request_text = str(request.get("text") or "")
            focus_text = _request_focus_text(request_text)
            if not focus_text:
                continue
            context_text = _request_context_text(request)
            for slot in _slots_for(focus_text):
                if not _slot_allowed_by_focus(slot, focus_text):
                    continue
                condition = _condition_for(slot, f"{focus_text} {context_text}")
                target_error_id = _ask_info_target_error_id(matched, slot, condition, focus_text, context_text)
                evidence_ids = _list(request.get("evidence_message_ids")) or [str(request.get("message_id") or "")]
                provided_ids = _list(request.get("provided_evidence_message_ids"))
                question = QUESTION_TEMPLATES.get(slot, QUESTION_TEMPLATES["other"])
                if condition == "dmp" and slot == "log_package":
                    question = "请提供蓝屏或重启对应的 dmp/系统日志。"
                elif condition == "startup/init log" and slot == "log_package":
                    question = "请提供启动/初始化阶段的 DLOG 或诊断数据包。"
                elif slot == "software_version" and "算法包" in focus_text:
                    question = "请提供主程序版本和算法包版本。"
                provided_tool_roles = _provided_tool_roles(request, tool_evidence)
                provided_slot_match_roles = _slot_matched_provided_roles(slot, provided_tool_roles)
                quality = _required_info_quality(slot, condition, target_error_id, evidence_ids, f"{focus_text} {context_text}")
                if provided_tool_roles:
                    quality = dict(quality)
                    quality["provided_tool_roles"] = provided_tool_roles
                    quality["provided_slot_match_roles"] = provided_slot_match_roles
                    if provided_slot_match_roles:
                        quality["evidence_strength"] = max(float(quality.get("evidence_strength") or 0.0), min(1.0, 0.55 + 0.1 * len(provided_slot_match_roles)))
                    else:
                        quality["provided_tool_roles_mismatch"] = True
                        quality["evidence_strength"] = min(float(quality.get("evidence_strength") or 0.0), 0.45)
                    quality["score"] = max(float(quality.get("score") or 0.0), _quality_score(quality)) if provided_slot_match_roles else _quality_score(quality)
                merge_policy = "append_to_required_info" if target_error_id else "review_only"
                if slot == "other":
                    merge_policy = "review_only"
                candidate_id = _required_info_id(source_episode_id or source_thread_id, str(request.get("message_id") or ""), slot, condition)
                out.append({
                    "candidate_id": candidate_id,
                    "source_episode_id": source_episode_id,
                    "source_thread_id": source_thread_id or str(request.get("thread_id") or ""),
                    "target_error_id": target_error_id,
                    "acceptable_error_ids": [target_error_id] if target_error_id else [],
                    "slot": slot if slot in REQUIRED_INFO_SLOTS else "other",
                    "label": SLOT_LABELS.get(slot, SLOT_LABELS["other"]),
                    "question": question,
                    "why_required": WHY_TEMPLATES.get(slot, WHY_TEMPLATES["other"]),
                    "condition": condition,
                    "priority": _priority(slot, condition, context_text),
                    "evidence_message_ids": evidence_ids[:12],
                    "source_offsets": semantics.get("source_offsets") or [],
                    "provided_later": bool(request.get("provided_later")),
                    "provided_evidence_message_ids": provided_ids[:12],
                    "provided_tool_roles": provided_tool_roles,
                    "provided_slot_match_roles": provided_slot_match_roles,
                    "diagnostic_outcome_after_provided": _one_line(semantics.get("conclusion"), 240),
                    "merge_policy": merge_policy,
                    "quality": quality,
                    "request_focus_text": focus_text,
                    "source_request": request,
                })
        return out

    def _match_existing(self, query: str) -> dict[str, Any] | None:
        normalized = _one_line(query, 600)
        if not self.store or not normalized.strip():
            return None
        if normalized in self._match_cache:
            return self._match_cache[normalized]
        candidates = self.store.search_errors(normalized, limit=1)
        if not candidates or candidates[0].score < self.match_threshold:
            self._match_cache[normalized] = None
            return None
        top = candidates[0]
        result = {"error_id": top.error_id, "label": top.label, "score": top.score, "route": top.route, "evidence": top.evidence}
        self._match_cache[normalized] = result
        return result

    @staticmethod
    def _solution_method(text: str) -> str:
        lowered = text.lower()
        if "重启" in text or "restart" in lowered:
            return "restart_or_recover_service"
        if "升级" in text or "版本" in text:
            return "upgrade_or_version_fix"
        if "配置" in text or "ip" in lowered or "IP" in text:
            return "configuration_fix"
        if "替换" in text or "更换" in text:
            return "replace_component"
        return "chat_reported_resolution"

    @staticmethod
    def _confidence(*, symptom: str, actions: list[str], conclusion: str, evidence_ids: list[str], log_paths: list[str], matched: dict[str, Any] | None) -> float:
        score = 0.25
        if symptom:
            score += 0.2
        if actions:
            score += min(0.2, 0.05 * len(actions))
        if conclusion:
            score += 0.15
        if evidence_ids:
            score += min(0.1, 0.02 * len(evidence_ids))
        if log_paths:
            score += 0.05
        if matched:
            score += 0.1
        return round(min(score, 0.95), 4)

OUTCOME_TYPES = {
    "verified_fix", "ineffective", "partial_temporary", "mitigation_observed",
    "recurred", "pending_validation", "diagnostic_method", "context_not_root_cause",
}


def _case_variant_candidate(semantics: dict[str, Any], error_id: str, matched: dict[str, Any] | None) -> dict[str, Any]:
    canonical = str((matched or {}).get("error_id") or "")
    return {
        "error_id": error_id,
        "entry_role": "case_variant" if canonical and canonical != error_id else "canonical",
        "canonical_error_id": canonical if canonical and canonical != error_id else "",
        "category": semantics.get("category") or "",
        "subsystem": _infer_subsystem(str(semantics.get("semantic_text") or "")),
        "scenario": _one_line(semantics.get("label") or semantics.get("symptom_raw"), 120),
        "source_episode_id": semantics.get("source_episode_id") or "",
        "source_thread_id": semantics.get("source_thread_id") or "",
    }


def _infer_subsystem(text: str) -> str:
    if any(k in text for k in ("相机", "采集卡", "CXP", "cxp", "拍照", "图像")):
        return "相机/采集链路"
    if any(k in text for k in ("工控机", "内存", "硬盘", "蓝屏", "dmp", "DMP")):
        return "工控机/Windows系统"
    if any(k in text for k in ("光源", "光机", "控制板")):
        return "光源/光机控制"
    if any(k in text for k in ("算法", "漏检", "误报", "模型")):
        return "算法/检测程序"
    return ""


def _diagnostic_trace_candidate(semantics: dict[str, Any], error_id: str, digest: str, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [node for node in nodes if node.get("type") == "DiagnosticCheck"]
    order = []
    for idx, check in enumerate(checks, start=1):
        order.append({
            "order": idx,
            "check_id": check.get("check_id") or check.get("id"),
            "label": check.get("label") or "",
            "evidence_message_ids": _list(semantics.get("evidence_ids"))[:12],
        })
    trace_id = f"trace:candidate-{digest}:1"
    return {
        "type": "DiagnosticTrace",
        "id": trace_id,
        "trace_id": trace_id,
        "source_episode_id": semantics.get("source_episode_id") or "",
        "source_thread_id": semantics.get("source_thread_id") or "",
        "target_error_id": error_id,
        "recommended_order": order,
        "actual_order": order,
        "evidence_message_ids": _list(semantics.get("evidence_ids"))[:30],
        "source_offsets": semantics.get("source_offsets") or [],
        "summary": _one_line(semantics.get("conclusion") or semantics.get("symptom_raw"), 240),
        "proposal_only": True,
    }


def _check_for_outcome_action(action: str, checks: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    if not checks:
        return {}
    if idx - 1 < len(checks):
        return checks[idx - 1]
    action_terms = _outcome_match_terms(action)
    best: tuple[int, dict[str, Any]] = (0, {})
    for check in checks:
        label = str(check.get("label") or check.get("how_to_check") or "")
        score = sum(1 for term in action_terms if term and term in label)
        if score > best[0]:
            best = (score, check)
    return best[1] if best[0] > 0 else checks[-1]


def _outcome_match_terms(text: str) -> list[str]:
    clean = str(text or "")
    preferred = [
        term for term in (
            "user.cfg", "cfg", "toml", "conf",
            "内存条", "内存", "采集卡", "CXP", "相机", "工控机", "驱动", "版本", "光源", "网卡", "主板", "硬盘", "显卡",
            "PoolMon", "poolmon", "WPR", "wpr", "Driver Verifier", "verifier", "DMP", "dmp",
            "PTE", "PFN", "MEMORY_MANAGEMENT", "0x00000139", "USB"
        ) if term in clean
    ]
    if preferred:
        return preferred
    stop = {"检查", "分析", "确认", "测试", "开启", "使用", "执行", "继续", "观察", "验证", "恢复", "是否", "然后", "之后", "后续", "进行", "继续定位"}
    out = []
    for chunk in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", clean):
        chunk = chunk.strip()
        if len(chunk) < 2 or chunk in stop:
            continue
        out.append(chunk)
    return out[:6]


def _diagnostic_outcome_candidates(
    semantics: dict[str, Any],
    error_id: str,
    digest: str,
    nodes: list[dict[str, Any]],
    conclusion: str,
) -> list[dict[str, Any]]:
    checks = [node for node in nodes if node.get("type") == "DiagnosticCheck"]
    solutions = [node for node in nodes if node.get("type") == "Solution"]
    out: list[dict[str, Any]] = []
    evidence = _list(semantics.get("evidence_ids"))[:30]
    actions = _list(semantics.get("debug_actions"))
    conclusion_is_concrete_fix = (
        _classify_outcome_type(conclusion, conclusion) == "verified_fix"
        and (
            _has_change_action(conclusion)
            or any(marker in conclusion for marker in ("恢复", "修复", "清理", "重新配置", "改回", "纠正"))
        )
    )
    for idx, action in enumerate(actions, start=1):
        outcome_context = _outcome_result_context(action, conclusion)
        if (
            not outcome_context
            and len(actions) == 1
            and _has_explicit_outcome_signal(conclusion)
            and not _looks_like_diagnostic_method(action)
        ):
            outcome_context = conclusion
        outcome_type = _classify_outcome_type(action, outcome_context)
        if outcome_type == "verified_fix" and (_looks_like_diagnostic_method(action) or _has_any(action, _NO_PROBLEM_MARKERS)):
            outcome_type = "pending_validation"
        if outcome_type == "diagnostic_method" and conclusion_is_concrete_fix and len(actions) == 1:
            # The inspection belongs in DiagnosticTrace.  The separately
            # evidenced resolution below owns the terminal ActionOutcome.
            continue
        if not _should_emit_outcome(action, outcome_context, outcome_type):
            continue
        check = _check_for_outcome_action(action, checks, idx)
        target_solution_id = ""
        if outcome_type == "verified_fix" and solutions:
            target_solution_id = str(solutions[0].get("solution_id") or solutions[0].get("id") or "")
        outcome_id = f"outcome:candidate-{digest}:{len(out) + 1}"
        out.append({
            "type": "DiagnosticOutcome",
            "id": outcome_id,
            "outcome_id": outcome_id,
            "source_episode_id": semantics.get("source_episode_id") or "",
            "source_thread_id": semantics.get("source_thread_id") or "",
            "target_error_id": error_id,
            "target_check_id": check.get("check_id") or check.get("id") or "",
            "target_solution_id": target_solution_id,
            "action_label": _clean_chat_text(action, 180),
            "outcome_type": outcome_type,
            "condition": _outcome_condition(action, outcome_context),
            "high_cost": _is_high_cost_action(action),
            "destructive": _is_destructive_action(action),
            "observed_duration": _observed_duration(f"{action} {outcome_context}"),
            "root_cause_summary": _root_cause_summary(outcome_context),
            "needs_confirmation": (
                _looks_like_fix_confirmation_question(action)
                or _looks_like_fix_confirmation_question(outcome_context)
            ) and outcome_type == "verified_fix",
            "evidence_message_ids": evidence,
            "source_offsets": semantics.get("source_offsets") or [],
            "proposal_only": True,
        })
    # A resolution message can state an explicit executed fix even when the
    # diagnostic chain contains only an inspection step.  Preserve that as its
    # own evidenced outcome instead of (a) promoting the inspection to a fix or
    # (b) dropping the resolution entirely.  This is deliberately narrower than
    # the former generic "conclusion as action" behavior: a concrete change and
    # verified result are both required.
    if (
        conclusion
        and solutions
        and conclusion_is_concrete_fix
        and not any(item.get("outcome_type") == "verified_fix" for item in out)
    ):
        resolution_evidence = [
            str(item.get("message_id") or item.get("source_message_id") or "")
            for item in ((semantics.get("episode") or {}).get("resolution_messages") or [])
            if isinstance(item, dict) and str(item.get("message_id") or item.get("source_message_id") or "")
        ]
        check = checks[-1] if checks else {}
        solution_id = str(solutions[0].get("solution_id") or solutions[0].get("id") or "")
        outcome_id = f"outcome:candidate-{digest}:{len(out) + 1}"
        out.append({
            "type": "DiagnosticOutcome",
            "id": outcome_id,
            "outcome_id": outcome_id,
            "source_episode_id": semantics.get("source_episode_id") or "",
            "source_thread_id": semantics.get("source_thread_id") or "",
            "target_error_id": error_id,
            "target_check_id": check.get("check_id") or check.get("id") or "",
            "target_solution_id": solution_id,
            "action_label": _clean_chat_text(conclusion, 180),
            "outcome_type": "verified_fix",
            "condition": _outcome_condition(conclusion, conclusion),
            "high_cost": _is_high_cost_action(conclusion),
            "destructive": _is_destructive_action(conclusion),
            "observed_duration": _observed_duration(conclusion),
            "root_cause_summary": _root_cause_summary(conclusion),
            "needs_confirmation": _looks_like_fix_confirmation_question(conclusion),
            "evidence_message_ids": resolution_evidence or evidence,
            "source_offsets": semantics.get("source_offsets") or [],
            "proposal_only": True,
        })
    return out


def _should_emit_outcome(action: str, outcome_context: str, outcome_type: str) -> bool:
    """Only emit outcomes when the action carries a result signal.

    Pure inspection / collection steps without an explicit result create noisy
    pseudo-outcomes and hurt v2 gold alignment. Diagnostic methods, explicit
    failures, explicit pending observations, or conclusion-qualified changes are
    still worth materializing as outcomes.
    """

    text = str(action or "")
    if not text:
        return False
    if outcome_context:
        if str(action or "").startswith(("检查", "确认")) and outcome_type in {"ineffective", "diagnostic_method", "context_not_root_cause"}:
            if not any(k in outcome_context for k in ("怀疑", "疑似", "错误代码", "PFN", "PTE", "丢失", "损坏", "无法加载", "为空", "空白文件", "驱动")):
                return False
        return True
    if outcome_type in {"verified_fix", "ineffective", "partial_temporary", "mitigation_observed", "recurred"}:
        return True
    if outcome_type == "pending_validation":
        return bool(_has_change_action(text) or "观察" in text or _has_any(text, _PENDING_MARKERS))
    if outcome_type == "diagnostic_method":
        return _looks_like_diagnostic_method(text)
    if outcome_type == "context_not_root_cause":
        return any(k in text for k in ("怀疑", "疑似", "可能", "原因", "导致", "指向", "说明", "怀疑是"))
    return False


def _verified_solution_ids(solution_ids: list[str], outcomes: list[dict[str, Any]], conclusion: str) -> list[str]:
    if not solution_ids:
        return []
    if any(outcome.get("target_solution_id") in solution_ids and outcome.get("outcome_type") == "verified_fix" for outcome in outcomes):
        return sorted({str(outcome.get("target_solution_id")) for outcome in outcomes if outcome.get("target_solution_id") in solution_ids and outcome.get("outcome_type") == "verified_fix"})
    return solution_ids if _classify_outcome_type(conclusion, conclusion) == "verified_fix" else []


_ACTION_FAILURE_MARKERS = ("无效", "没效果", "未解决", "不行", "无改善", "未改善", "验证失败", "未排查出", "不能解决", "无法解决", "仍然存在", "依然存在", "仍存在", "依旧存在", "仍旧存在")
_PENDING_MARKERS = ("待验证", "需要验证", "未验证", "待执行", "暂无法执行", "无法执行", "双休", "放假", "继续观察", "明天", "计划", "建议", "可以先", "先", "请问", "能否", "是否", "确认一下", "沟通", "尝试", "先观察")
_DIAGNOSTIC_METHOD_MARKERS = ("WPR", "PoolMon", "poolmon", "Driver Verifier", "verifier", "抓取", "收集", "导出", "日志", "诊断数据", "DMP", "dmp", "dump", "memtest", "MemTest", "查看", "检查", "分析")
_STRONG_FIX_MARKERS = (
    "已解决", "解决了", "最终解决", "恢复正常", "更换后正常", "已恢复", "修复", "验证通过", "长期稳定",
    "不再复发", "未再复发", "无复发", "不再出现", "未再出现", "没有再出现", "到现在未再出现", "至今未再出现",
    "不再发生", "未再发生", "没有再发生", "没有复现", "未复现", "未再复现",
)
_VERIFIED_FIX_MARKERS = (
    "已解决", "解决了", "最终解决", "验证通过", "长期稳定", "恢复生产", "正常生产",
    "不再复发", "未再复发", "无复发", "不再出现", "未再出现", "没有再出现", "到现在未再出现", "至今未再出现",
    "不再发生", "未再发生", "没有再发生", "没有复现", "未复现", "未再复现",
)
_TEMPORARY_MARKERS = ("暂时", "临时", "短时", "短期", "缓解", "可临时", "重启后正常", "重启恢复", "重启可恢复")
_NO_RECURRENCE_OBSERVATION_MARKERS = (
    "暂未复发", "暂时未复发", "目前未复发", "还未复发", "观察未复发", "现场观察暂未复发",
    "暂未出现", "目前未出现", "观察未出现", "均未出现", "反复验证未出现", "运行未复现", "运行未再出现",
)
_NO_PROBLEM_MARKERS = ("没有什么问题", "没什么问题", "均无问题", "无问题", "不是问题", "接地没问题", "正常识别", "设置正常")
_CONFIRMATION_QUESTION_MARKERS = ("吧", "吗", "是否", "确认一下", "确认下", "还会不会", "有没有再")
_HIGH_COST_PENDING_MARKERS = ("返厂", "重标", "成本高", "高成本", "更换相机")


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _has_action_failure(text: str) -> bool:
    if _has_any(text, _ACTION_FAILURE_MARKERS):
        return True
    return bool(re.search(r"(更换|替换|回退|降级|升级|调整|重装|卸载|拔插|修复|恢复).{0,24}失败", text))


def _has_change_action(text: str) -> bool:
    return any(k in text for k in ("更换", "替换", "卸载", "更新", "调整", "回退", "降级", "升级", "重装", "修改", "取消", "勾选"))


def _outcome_result_context(action: str, conclusion: str) -> str:
    """Return conclusion context only when it can reasonably qualify this action.

    A strong final conclusion should not turn every earlier diagnostic method
    into a verified fix.  It can qualify the concrete executed change it refers
    to, e.g. "现场已更换内存条" + "更换后未再出现蓝屏".
    """

    action_text = str(action or "")
    conclusion_text = str(conclusion or "")
    if not conclusion_text:
        return ""
    if _has_action_failure(action_text):
        return ""
    if _looks_like_diagnostic_method(action_text):
        return ""
    if any(k in action_text for k in ("建议", "可以先", "计划", "请问", "能否", "是否", "需要怎么操作", "商务")) and not _has_strong_fix_evidence(action_text):
        return ""
    if _has_strong_fix_evidence(action_text):
        return conclusion_text
    if not _has_change_action(action_text):
        return ""
    terms = [term for term in ("内存", "采集卡", "CXP", "相机", "工控机", "驱动", "版本", "光源", "网卡", "主板", "硬盘", "显卡") if term in action_text]
    if terms and any(term in conclusion_text for term in terms):
        return conclusion_text
    return ""


def _has_strong_fix_evidence(text: str) -> bool:
    if _has_any(text, _STRONG_FIX_MARKERS):
        return True
    if re.search(r"(更换|替换|修复|处理|调整|重装|回退|降级|升级).{0,32}(?:后|之后).{0,18}(?:未再|没有再|不再)(?:出现|发生|复发|复现)", text):
        return True
    if re.search(r"(?:运行|观察|测试).{0,12}(?:\d+(?:h|H|小时|天)|到现在|至今).{0,18}(?:未再|没有再|不再|未)(?:出现|发生|复发|复现)", text):
        return True
    # "正常" alone is too weak in chat logs (often means short smoke/after reboot).
    return bool(re.search(r"(更换|替换|修复|处理|调整|重装|回退|降级|升级).{0,18}(后)?(恢复)?正常", text))


def _has_verified_fix_evidence(text: str) -> bool:
    if _has_any(text, _VERIFIED_FIX_MARKERS):
        return True
    if re.search(r"(更换|替换|修复|处理|调整|重装|回退|降级|升级).{0,24}(后|之后).{0,12}(恢复)?正常", text):
        return True
    return bool(re.search(r"(更换|替换|修复|处理|调整|重装|回退|降级|升级).{0,24}(后|之后).{0,18}(?:未再|没有再|不再)(?:出现|发生|复发|复现)", text))


def _looks_like_fix_confirmation_question(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    if clean.endswith(("吧", "吧？", "吧?", "吗", "吗？", "吗?")):
        return True
    return any(marker in clean for marker in ("是否未再", "确认一下", "确认下", "还会不会", "有没有再"))


def _looks_like_diagnostic_method(text: str) -> bool:
    return _has_any(text, _DIAGNOSTIC_METHOD_MARKERS) and not _has_strong_fix_evidence(text)


def _normalize_outcome_type_from_text(action: str, conclusion: str = "", proposed: str = "") -> str:
    text = f"{action} {conclusion}"
    lowered = text.lower()
    proposed = proposed if proposed in OUTCOME_TYPES else ""
    if _has_any(text, _NO_PROBLEM_MARKERS):
        return "context_not_root_cause"
    if _has_action_failure(text):
        return "ineffective"
    if "尝试修复" in text and not _has_verified_fix_evidence(text):
        return "pending_validation"
    if any(k in action for k in ("尝试修复", "建议更换", "建议升级", "建议重装", "建议回退")) and not _has_strong_fix_evidence(text):
        return "pending_validation"
    if any(k in action for k in ("user.cfg", "cfg", "conf")) and any(k in text for k in ("空白文件", "为空", "备份")):
        return "context_not_root_cause"
    if any(k in action for k in ("回填", "重启验证")) and any(k in text for k in ("是否正常", "看是否正常", "重启软件")):
        return "diagnostic_method"
    if any(k in action for k in ("WPR", "PoolMon", "poolmon", "Driver Verifier", "verifier")):
        return "diagnostic_method"
    if "重新拔插" in action and any(k in text for k in ("恢复正常", "已正常", "恢复")):
        if any(k in text for k in ("仍需观察", "后续观察", "持续观察", "继续观察")):
            return "mitigation_observed"
        return "verified_fix"
    if any(k in action for k in ("继续观察", "持续观察", "上线验证", "跟线验证")):
        return "pending_validation"
    if any(k in action for k in ("分析 DMP", "收集并分析转存储文件", "检查关键驱动文件是否缺失或损坏", "分析 DMP 中 PTE 耗尽信号")) and any(
        k in text for k in ("错误代码", "BugCheck", "PFN", "PTE", "驱动无法加载", "丢失", "损坏", "极低", "只剩", "空白文件")
    ):
        return "context_not_root_cause"
    if any(k in action for k in ("观察是否复发", "继续观察", "持续观察")) and any(k in text for k in ("复发", "又出现", "再次", "仍出现")):
        return "partial_temporary"
    if any(k in text for k in ("仍需观察", "后续观察", "持续观察", "继续观察")) and _has_change_action(text):
        return "mitigation_observed"
    if "恢复正常" in text and "观察" in text and _has_change_action(text):
        return "mitigation_observed"
    if any(k in text for k in ("待执行", "暂无法执行", "无法执行", "双休", "放假")):
        return "pending_validation"
    if "接地" in text and any(k in text for k in ("正常", "不是", "非根因", "不是根因")):
        return "context_not_root_cause"
    if ("每天" in text or "周期" in text or "一星期" in text) and any(k in text for k in ("重启", "断电")):
        return "mitigation_observed"
    if _has_any(text, _NO_RECURRENCE_OBSERVATION_MARKERS):
        return "mitigation_observed"
    if any(k in text for k in ("复发", "又出现", "再次", "过一会", "一段时间后", "2h", "2小时", "两小时")):
        if _has_change_action(text) or any(k in text for k in ("正常", "恢复", "缓解", "好了", "稳定")):
            return "partial_temporary"
        return "recurred"
    if _has_any(text, _TEMPORARY_MARKERS) or ("重启" in text and any(k in text for k in ("正常", "恢复", "好了", "开机"))):
        return "partial_temporary"
    if "观察" in text and _has_change_action(text):
        return "mitigation_observed"
    if _has_any(text, _HIGH_COST_PENDING_MARKERS) and not _has_strong_fix_evidence(text):
        return "pending_validation"
    if _looks_like_diagnostic_method(text):
        return "diagnostic_method"
    if "验证" in action and any(k in text for k in ("是否正常", "看是否正常", "验证")) and not _has_strong_fix_evidence(text):
        return "diagnostic_method"
    if _has_any(text, _PENDING_MARKERS) and not _has_strong_fix_evidence(text):
        return "pending_validation"
    if any(k in lowered for k in ("context", "background")):
        return "context_not_root_cause"
    if _has_verified_fix_evidence(text):
        return "verified_fix"
    if _has_strong_fix_evidence(text):
        return "mitigation_observed"
    if proposed == "verified_fix":
        if not _has_verified_fix_evidence(text):
            return "pending_validation"
        if _has_any(text, _NO_PROBLEM_MARKERS) or _looks_like_diagnostic_method(action):
            return "context_not_root_cause"
    # DeepSeek may over-promote actions without local evidence.  Keep non-terminal
    # labels only when the text supports them; otherwise review as pending.
    if proposed == "ineffective" and not _has_action_failure(text):
        return "pending_validation"
    if proposed == "mitigation_observed" and not (_has_any(text, _NO_RECURRENCE_OBSERVATION_MARKERS) or _has_any(text, _TEMPORARY_MARKERS) or "观察" in text):
        return "pending_validation"
    return proposed or "pending_validation"


def _classify_outcome_type(action: str, conclusion: str = "") -> str:
    return _normalize_outcome_type_from_text(action, conclusion)


def _outcome_condition(action: str, conclusion: str) -> str:
    text = f"{action} {conclusion}"
    if any(k in text for k in ("dmp", "DMP", "蓝屏", "dump")):
        return "dmp"
    if any(k in text for k in ("初始化", "启动", "startup", "init")):
        return "startup/init"
    if any(k in text for k in ("CXP", "采集卡", "相机", "拍照")):
        return "camera_capture_chain"
    if any(k in text for k in ("版本", "回退", "升级", "降级")):
        return "software_version_change"
    return ""


def _is_high_cost_action(text: str) -> bool:
    return any(k in str(text or "") for k in ("返厂", "重标", "更换相机", "更换工控机", "停线", "成本高"))


def _is_destructive_action(text: str) -> bool:
    return any(k in str(text or "") for k in ("停机", "拆机", "断电", "删除", "清空", "重装", "格式化", "返厂", "重标"))


def _observed_duration(text: str) -> str:
    match = re.search(r"\b\d+\s*(?:h|H|小时|分钟|min)\b", str(text or ""))
    if match:
        return match.group(0)
    match = re.search(r"[一二两三四五六七八九十]+小时", str(text or ""))
    return match.group(0) if match else ""


def _root_cause_summary(text: str) -> str:
    value = str(text or "")
    for marker in ("原因是", "根因是", "最终", "定位到", "问题是", "实际是"):
        if marker in value:
            return _clean_chat_text(value[value.index(marker):], 180)
    return ""


def _candidate_semantic_issues(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    node_by_id = {_node_pk(node): node for node in nodes if _node_pk(node)}
    for node in nodes:
        if node.get("type") == "DiagnosticOutcome":
            if str(node.get("outcome_type") or "") not in OUTCOME_TYPES:
                issues.append(f"invalid_outcome_type:{node.get('outcome_id')}")
            if not node.get("evidence_message_ids"):
                issues.append(f"outcome_missing_evidence:{node.get('outcome_id')}")
        if node.get("type") == "Error" and str(node.get("entry_role") or "") == "case_variant":
            if not node.get("canonical_error_id"):
                issues.append(f"case_variant_missing_canonical:{node.get('error_id')}")
    for edge in edges:
        if edge.get("relation") != "resolved_by":
            continue
        solution = node_by_id.get(str(edge.get("to") or ""), {})
        if str(solution.get("evidence_level") or "") in OUTCOME_TYPES - {"verified_fix"}:
            issues.append(f"resolved_by_non_verified_solution:{edge.get('to')}")
    return sorted(set(issues))


def _split_decision(semantics: dict[str, Any]) -> dict[str, Any]:
    text = str(semantics.get("semantic_text") or "")
    multi_signals = sum(1 for marker in ("另外", "还有", "另一个", "同时", "蓝屏", "采集", "驱动", "相机") if marker in text)
    return {
        "decision": "candidate_single_episode" if multi_signals < 3 else "review_for_possible_split",
        "reason": "deterministic_marker_count",
        "marker_count": multi_signals,
    }

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_BETA_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/beta/chat/completions"


def _extract_prompt_case_understanding_with_repair(
    semantics: dict[str, Any],
    *,
    api_key: str,
) -> tuple[dict[str, Any], int, list[str]]:
    """Run Prompt A and allow one schema-grounding repair attempt."""

    prompt_input = build_case_understanding_prompt_input(semantics)
    repair_issues: list[str] = []
    all_corrections: list[str] = []
    max_attempts = max(1, min(2, int(os.environ.get("DEEPSEEK_W2_PROMPT_ATTEMPTS", "2"))))
    for attempt in range(1, max_attempts + 1):
        try:
            raw = _call_deepseek_case_understanding_with_hard_timeout(
                prompt_input,
                api_key=api_key,
                repair_issues=repair_issues,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            repair_issues = [
                f"tool_arguments_parse_error:{type(exc).__name__}:{str(exc)[:240]}"
            ]
            if attempt < max_attempts:
                continue
            raise ValueError("prompt_a_invalid:" + repair_issues[0]) from exc
        card, issues, corrections = normalize_prompt_case_understanding_card(raw, semantics)
        all_corrections.extend(corrections)
        if not issues:
            return card, attempt, sorted(set(all_corrections))
        repair_issues = issues[:40]
    raise ValueError("prompt_a_invalid:" + ",".join(repair_issues[:40]))


def _call_deepseek_case_understanding_with_hard_timeout(
    prompt_input: dict[str, Any],
    *,
    api_key: str,
    repair_issues: list[str],
) -> dict[str, Any]:
    hard_timeout = float(os.environ.get("DEEPSEEK_W2_HARD_TIMEOUT", "360"))
    if hard_timeout <= 0 or os.name == "nt" or threading.current_thread() is not threading.main_thread():
        return _call_deepseek_case_understanding(
            prompt_input,
            api_key=api_key,
            repair_issues=repair_issues,
        )

    def _raise_timeout(signum: int, frame: Any) -> None:  # pragma: no cover
        raise _DeepSeekHardTimeoutError(f"deepseek_wall_clock_timeout>{hard_timeout}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, hard_timeout)
    try:
        return _call_deepseek_case_understanding(
            prompt_input,
            api_key=api_key,
            repair_issues=repair_issues,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _call_deepseek_case_understanding(
    prompt_input: dict[str, Any],
    *,
    api_key: str,
    repair_issues: list[str] | None = None,
) -> dict[str, Any]:
    """Call DeepSeek with the complete Prompt-A CaseUnderstanding schema."""

    user_payload: dict[str, Any] = {"input": prompt_input}
    if repair_issues:
        user_payload["repair_request"] = {
            "instruction": "上一次输出未通过本地证据/结构校验。只修复下列问题，不增加证据中不存在的事实。",
            "issues": repair_issues,
        }
    schema = case_understanding_tool_schema()
    schema_issues = _tool_schema_strict_issues(schema)
    if schema_issues:
        raise ValueError("invalid_case_understanding_tool_schema:" + ",".join(schema_issues))
    from debug_agent_system.agents.write.w2_extract.deepseek_client import (
        call_json_object,
        call_strict_tool,
        configured_model,
        model_output_limit,
    )

    model = configured_model()
    default_tokens = 32_768 if model.startswith("deepseek-v4-") else 8_192
    requested_tokens = int(os.environ.get("DEEPSEEK_W2_MAX_TOKENS", str(default_tokens)))
    common = {
        "api_key": api_key,
        "max_tokens": min(model_output_limit(model), requested_tokens),
        "max_attempts": max(1, min(3, int(os.environ.get("DEEPSEEK_W2_TRANSPORT_ATTEMPTS", "3")))),
    }
    if model.startswith("deepseek-v4-"):
        response = call_json_object(
            **common,
            system_prompt=(
                CASE_UNDERSTANDING_SYSTEM_PROMPT.replace("严格按工具 schema 输出，不输出解释文字。", "")
                + "\n只输出一个符合 output_json_schema 的 JSON object，不输出 Markdown 或解释文字。"
            ),
            user_payload={
                **user_payload,
                "output_json_schema": schema["function"]["parameters"],
            },
        )
    else:
        response = call_strict_tool(
            **common,
            system_prompt=CASE_UNDERSTANDING_SYSTEM_PROMPT,
            user_payload=user_payload,
            tool=schema,
        )
    return response["arguments"]


def _deepseek_w2_tool_schema() -> dict[str, Any]:
    """Strict DeepSeek tool-call schema for W2 extraction.

    DeepSeek strict mode requires object schemas to list all properties in
    ``required`` and to set ``additionalProperties`` to false.  Keep every
    field optional in semantics by allowing empty strings/lists, not by omitting
    required schema keys.
    """

    outcome_types = sorted(OUTCOME_TYPES)
    slots = sorted(REQUIRED_INFO_SLOTS)
    return {
        "type": "function",
        "function": {
            "name": "extract_w2_kg_candidate",
            "description": "Extract AOI fault case variant, diagnostic trace, outcomes, and required info from one chat episode.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "case_variant_candidate": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "category": {"type": "string"},
                            "subsystem": {"type": "string"},
                            "scenario": {"type": "string"},
                            "canonical_error_id": {"type": "string"},
                            "escalation_target": {"type": "string"},
                        },
                        "required": ["label", "category", "subsystem", "scenario", "canonical_error_id", "escalation_target"],
                    },
                    "diagnostic_trace": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "recommended_order": {"type": "array", "items": {"type": "string"}},
                            "actual_order": {"type": "array", "items": {"type": "string"}},
                            "summary": {"type": "string"},
                        },
                        "required": ["recommended_order", "actual_order", "summary"],
                    },
                    "diagnostic_outcomes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "action_label": {"type": "string"},
                                "outcome_type": {"type": "string", "enum": outcome_types},
                                "condition": {"type": "string"},
                                "target_check_id": {"type": "string"},
                                "target_solution_id": {"type": "string"},
                                "high_cost": {"type": "boolean"},
                                "destructive": {"type": "boolean"},
                                "observed_duration": {"type": "string"},
                                "root_cause_summary": {"type": "string"},
                                "evidence_message_ids": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["action_label", "outcome_type", "condition", "target_check_id", "target_solution_id", "high_cost", "destructive", "observed_duration", "root_cause_summary", "evidence_message_ids"],
                        },
                    },
                    "required_info_candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "slot": {"type": "string", "enum": slots},
                                "label": {"type": "string"},
                                "question": {"type": "string"},
                                "why_required": {"type": "string"},
                                "condition": {"type": "string"},
                                "priority": {"type": "string"},
                                "target_error_id": {"type": "string"},
                                "provided_later": {"type": "boolean"},
                                "provided_evidence_message_ids": {"type": "array", "items": {"type": "string"}},
                                "evidence_message_ids": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["slot", "label", "question", "why_required", "condition", "priority", "target_error_id", "provided_later", "provided_evidence_message_ids", "evidence_message_ids"],
                        },
                    },
                    "split_decision": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "should_split": {"type": "boolean"},
                            "reason": {"type": "string"},
                            "marker_count": {"type": "integer"},
                        },
                        "required": ["should_split", "reason", "marker_count"],
                    },
                },
                "required": ["case_variant_candidate", "diagnostic_trace", "diagnostic_outcomes", "required_info_candidates", "split_decision"],
            },
        },
    }


def _tool_schema_strict_issues(schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
            required = node.get("required") if isinstance(node.get("required"), list) else []
            if node.get("additionalProperties") is not False:
                issues.append(f"{path}:additionalProperties_not_false")
            missing = sorted(set(props) - set(required))
            if missing:
                issues.append(f"{path}:properties_not_required:{','.join(missing)}")
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{path}.{key}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    walk(item, f"{path}.{key}[{idx}]")

    walk(schema.get("function", {}).get("parameters", {}), "parameters")
    return issues


class _DeepSeekHardTimeoutError(TimeoutError):
    pass


def _call_deepseek_w2_extractor_with_hard_timeout(semantics: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    """Apply an outer wall-clock timeout around the DeepSeek call.

    In long-running production runs we have seen individual requests stall far
    longer than the nominal socket timeout. On Unix main-thread execution, use
    `SIGALRM` as a hard wall-clock guard; otherwise fall back to the normal
    request path.
    """

    hard_timeout = float(os.environ.get("DEEPSEEK_W2_HARD_TIMEOUT", "75"))
    if hard_timeout <= 0:
        return _call_deepseek_w2_extractor(semantics, api_key=api_key)
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        return _call_deepseek_w2_extractor(semantics, api_key=api_key)

    def _raise_timeout(signum: int, frame: Any) -> None:  # pragma: no cover
        raise _DeepSeekHardTimeoutError(f"deepseek_wall_clock_timeout>{hard_timeout}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, hard_timeout)
    try:
        return _call_deepseek_w2_extractor(semantics, api_key=api_key)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _call_deepseek_w2_extractor(semantics: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    """Optional W2 LLM leaf extractor.

    DeepSeek's official API is OpenAI-compatible and supports JSON output via
    `response_format={"type":"json_object"}`.  This hook stays dependency-free
    and default-off; deterministic extraction remains the source of truth when
    the API is unavailable or the returned JSON fails local validation.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "你是 AOI 故障诊断 KG 写侧抽取器。只抽取结构化候选并以 JSON 输出，不做合并决策。"
                "禁止把无效尝试、临时缓解、待验证高成本动作、诊断方法标成 verified_fix。"
                "case_variant_candidate.label 必须是故障现象短名称，优先 8 到 24 个汉字；"
                "禁止复制某个检查动作、替换动作、追问句或长日志句作为 label；"
                "禁止以“请/能否/如果/将/更换/检查/2）/3.”开头。"
                "你现在不直接输出 FaultFamily，但必须让 case_variant_candidate.label 隐含地落在一个正确的故障家族下面；"
                "不要把 family 写成模块名、系统名、英文词或产品别名。"
                "尤其禁止把 label 或 subsystem 写成“AOI_复判站”“AOI检测软件”“display”“camera”“software”这类伪家族词。"
                "category 只能使用：系统与软件异常、硬件与运控、算法与程序调优。"
                "subsystem 必须是中文业务域短语，例如：相机/采集链路、主程序配置/复判站配置、显示/分辨率/缩放、工控机/Windows 内核、算法/检测逻辑、复判流程。"
                "如果现象是电视扩展/复制/缩放/分辨率导致显示不全，按“界面显示异常”家族理解，subsystem 用“显示/分辨率/缩放”；"
                "如果现象是算法结果未出、误报、漏检、框逻辑、复判判定异常，按“算法与程序调优”理解，不要写成 AOI_复判站 或 AOI检测软件。"
                "如果现象是拍照失败/空图/不拍照/超时，按“相机拍摄失败”理解；如果是启动初始化阶段相机未枚举/未连接，按“相机初始化失败”理解。"
                "如果提供了 review_context，它是 alignment_only 背景：其中 KG alignment 只允许复用 family 命名、动作粒度、ID 和追问方式；绝不能把背景中的 outcome/evidence 当成本轮事实。"
                "review_context.reviewed_case_examples 是已经人工标注过的示例；只把它们当成命名风格、动作粒度、outcome 类型和 required_info 组织方式的参考，绝不能把示例里的事实、SourceCase 或 EvidenceItem 直接搬到当前案例。"
                "如果 review_context.reviewed_case_examples 中存在 exact_source_match=true 的样本，说明这是同一条已审核案例；优先复用该样本的 family / variant 命名、action 粒度、outcome 类型和 required_info 结构，只在当前证据明确矛盾时才偏离。"
                "高优先级 family 参考：工控机蓝屏、工控机异常重启、用户配置加载失败、运控初始化失败、光源初始化失败、主程序初始化卡住无明确报错、相机拍摄失败、相机初始化失败、CAD 导入失败、Mark 点对齐失败、扫码识别失败、界面显示异常、误报调优异常、漏检调优异常、CT 时间异常增加、复判站出图慢、进板失败、出板失败。"
                "case_variant_candidate.label 是 variant，不是 family 本身；它应体现该 family 下的关键条件，但不要复述整段聊天。"
                "current_episode_messages、w7_promoted_case_evidence、linked_jira_evidence 和 evidence_ids 是当前候选 episode 唯一允许生成 action/outcome 的证据；"
                "W7 提升的消息必须保留原 message_id；Jira 的后续评论和最终状态优先于更早的群聊猜测，但冲突证据必须显式保留，不能静默覆盖。"
                "case_context_messages 只是导航背景，未列入 evidence_ids 的消息不得生成动作、结果或根因。"
                "如果背景出现其他 Jira/故障，必须忽略，不得拼入当前案例。"
                "label 示例：编程拍照速度延迟现象、设备卡顿后蓝屏、System PTE 耗尽导致蓝屏、相机网卡过滤驱动取消勾选导致拍摄失败。"
                "diagnostic_trace 里使用归一化排查步骤，如“检查采集卡”“检查驱动”“更换 CXP 线验证”，不要保留整段聊天原文。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "source_episode_id": semantics.get("source_episode_id"),
                "source_thread_id": semantics.get("source_thread_id"),
                "label": semantics.get("label"),
                "symptom_raw": semantics.get("symptom_raw"),
                "debug_actions": semantics.get("debug_actions"),
                "conclusion": semantics.get("conclusion"),
                "evidence_ids": semantics.get("evidence_ids"),
                "source_offsets": semantics.get("source_offsets"),
                "category_enum": ["系统与软件异常", "硬件与运控", "算法与程序调优"],
                "family_seed_catalog": [
                    "工控机蓝屏",
                    "工控机异常重启",
                    "用户配置加载失败",
                    "运控初始化失败",
                    "光源初始化失败",
                    "主程序初始化卡住无明确报错",
                    "相机拍摄失败",
                    "相机初始化失败",
                    "CAD 导入失败",
                    "Mark 点对齐失败",
                    "扫码识别失败",
                    "界面显示异常",
                    "误报调优异常",
                    "漏检调优异常",
                    "CT 时间异常增加",
                    "复判站出图慢",
                    "进板失败",
                    "出板失败",
                ],
                "subsystem_style_rules": [
                    "必须用中文业务域短语",
                    "不能用英文单词或产品代号当 subsystem",
                    "不能用 AOI_复判站、AOI检测软件、display 这类伪家族词",
                ],
                "current_episode_messages": [
                    {
                        "message_id": msg.get("message_id"),
                        "time": msg.get("create_time"),
                        "role": role,
                        "text": _one_line(msg.get("text") or msg.get("content_summary"), 500),
                    }
                    for role, messages_for_role in (
                        ("fault", (semantics.get("episode") or {}).get("fault_description_messages") or []),
                        ("diagnostic", (semantics.get("episode") or {}).get("diagnostic_chain_messages") or []),
                        ("resolution", (semantics.get("episode") or {}).get("resolution_messages") or []),
                    )
                    for msg in messages_for_role
                    if isinstance(msg, dict)
                ][:40],
                "w7_promoted_case_evidence": [
                    {
                        "message_id": msg.get("message_id"),
                        "time": msg.get("create_time"),
                        "sender": msg.get("sender"),
                        "text": _one_line(msg.get("text") or msg.get("content_summary"), 800),
                        "promotion_reason": msg.get("promotion_reason"),
                    }
                    for msg in ((semantics.get("episode") or {}).get("case_evidence_messages") or [])
                    if isinstance(msg, dict)
                ][:30],
                "linked_jira_evidence": semantics.get("linked_jira_evidence") or [],
                "case_context_messages": [
                    {
                        "message_id": msg.get("message_id"),
                        "time": msg.get("create_time"),
                        "text": _one_line(msg.get("text") or msg.get("content_summary"), 500),
                    }
                    for msg in ((semantics.get("episode") or {}).get("case_context_messages") or [])
                    if isinstance(msg, dict)
                    and str(msg.get("message_id") or "") in set(_list(semantics.get("evidence_ids")))
                ][:60],
                "semantic_text": _one_line(semantics.get("semantic_text"), 4000),
                "review_context": semantics.get("review_context") or semantics.get("sop_background") or {},
            }, ensure_ascii=False),
        },
    ]
    use_tools = os.environ.get("DEEPSEEK_W2_USE_TOOLS", "0") == "1"
    model = os.environ.get("DEEPSEEK_W2_MODEL", "deepseek-chat")
    url = DEEPSEEK_CHAT_COMPLETIONS_URL
    if use_tools:
        model = os.environ.get("DEEPSEEK_W2_TOOL_MODEL", "deepseek-chat")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": messages,
    }
    if use_tools:
        tool_schema = _deepseek_w2_tool_schema()
        schema_issues = _tool_schema_strict_issues(tool_schema)
        if schema_issues:
            raise ValueError("invalid_deepseek_tool_schema:" + ",".join(schema_issues))
        payload.update({
            "tools": [tool_schema],
            "tool_choice": {"type": "function", "function": {"name": "extract_w2_kg_candidate"}},
        })
        url = DEEPSEEK_BETA_CHAT_COMPLETIONS_URL
    else:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("DEEPSEEK_W2_TIMEOUT", "30"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit user-enabled external API
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"deepseek_http_{exc.code}:{body}") from exc
    message = ((raw.get("choices") or [{}])[0].get("message") or {})
    if use_tools:
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {}
        function = (tool_calls[0] or {}).get("function") or {}
        parsed = json.loads(function.get("arguments") or "{}")
    else:
        parsed = json.loads(message.get("content") or "{}")
    return parsed if isinstance(parsed, dict) else {}



LLM_SLOT_ALIASES = {
    "dmp_package": "log_package",
    "dump_package": "log_package",
    "diagnostic_data": "log_package",
    "diagnostic_log": "log_package",
    "driver_allocation_trace": "log_package",
    "wpr_trace": "log_package",
    "poolmon_trace": "log_package",
    "blue_screen_code": "error_message",
    "bugcheck_code": "error_message",
    "error_code": "error_message",
    "pte_exhaustion_signals": "error_message",
    "memory_config": "environment",
    "memory_cpu_test": "environment",
    "driver_context": "environment",
    "version_and_memory_context": "environment",
    "production_constraint": "environment",
    "graphics_driver_version": "software_version",
    "driver_version": "software_version",
    "recurrence_after_driver_change": "repro_steps",
    "recurrence_after_mitigation": "repro_steps",
    "capture_behavior_after_toggle": "repro_steps",
    "nic_role_map": "ip_config",
    "filter_driver_binding": "ip_config",
    "network_config": "ip_config",
    "route_config": "ip_config",
    "routing_config": "ip_config",
}


def _normalize_llm_required_info_slot(slot: Any) -> str:
    """Map LLM/tool-call slot drift back into the fixed required-info enum."""

    normalized = LLM_SLOT_ALIASES.get(str(slot or "other"), str(slot or "other"))
    return normalized if normalized in REQUIRED_INFO_SLOTS else "other"


def _outcome_context_terms(action: str) -> list[str]:
    text = str(action or "")
    terms = set(re.findall(r"[A-Za-z0-9_.:-]{2,}", text))
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    stop = {"检查", "排查", "更换", "替换", "验证", "测试", "分析", "收集", "导出", "确认", "是否", "进行", "问题", "设备"}
    for chunk in cjk:
        if len(chunk) <= 2 and chunk in stop:
            continue
        if len(chunk) >= 2:
            terms.add(chunk)
        for size in (2, 3, 4):
            for idx in range(len(chunk) - size + 1):
                gram = chunk[idx:idx+size]
                if gram not in stop:
                    terms.add(gram)
    # Set iteration made sanitizer behaviour depend on PYTHONHASHSEED and could
    # drop the action's most specific phrase from the first 12 terms.  Prefer
    # longer terms, then lexical order, so local-evidence matching is stable.
    return sorted(
        (term for term in terms if term and term not in stop),
        key=lambda term: (-len(term), term),
    )[:12]


def _local_outcome_context(action: str, semantics: dict[str, Any]) -> str:
    terms = _outcome_context_terms(action)
    if not terms:
        return ""
    text = str(semantics.get("semantic_text") or "")
    if not text:
        return ""
    hits: list[str] = []
    for sent in _split_sentences(text):
        if any(term in sent for term in terms):
            hits.append(sent)
        if len(hits) >= 4:
            break
    return _clean_chat_text(" ".join(hits), 600)


def _sanitize_deepseek_extraction(extraction: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(extraction, dict):
        return {}
    # DeepSeek strict tool-calls still occasionally echo input-only fields or
    # choose over-optimistic outcome labels.  Keep the local contract strict by
    # dropping non-output keys and normalising unsafe labels before validation.
    allowed = {"case_variant_candidate", "diagnostic_trace", "diagnostic_outcomes", "required_info_candidates", "split_decision"}
    sanitized = {key: value for key, value in extraction.items() if key in allowed}
    for key in ("case_variant_candidate", "diagnostic_trace", "split_decision"):
        if key in sanitized and not isinstance(sanitized.get(key), dict):
            sanitized[key] = {}
    default_evidence = _list(semantics.get("evidence_ids"))[:12]
    required = []
    for item in sanitized.get("required_info_candidates") or []:
        if not isinstance(item, dict):
            required.append(item)
            continue
        fixed = dict(item)
        fixed["slot"] = _normalize_llm_required_info_slot(fixed.get("slot"))
        if not fixed.get("evidence_message_ids") and default_evidence:
            fixed["evidence_message_ids"] = default_evidence
        if "provided_evidence_message_ids" not in fixed:
            fixed["provided_evidence_message_ids"] = []
        if "provided_later" not in fixed:
            fixed["provided_later"] = False
        required.append(fixed)
    if "required_info_candidates" in sanitized:
        sanitized["required_info_candidates"] = required
    outcomes = []
    for item in sanitized.get("diagnostic_outcomes") or []:
        if not isinstance(item, dict):
            outcomes.append(item)
            continue
        fixed = dict(item)
        action_text = str(fixed.get("action_label") or "")
        context_text = " ".join(str(fixed.get(key) or "") for key in ("condition", "observed_duration", "root_cause_summary"))
        local_context = _local_outcome_context(action_text, semantics)
        if not local_context:
            # The LLM may have copied an outcome from untrusted navigation
            # context.  Without a supporting sentence in the primary episode,
            # dropping it is safer than fabricating evidence IDs.
            continue
        fixed["outcome_type"] = _normalize_outcome_type_from_text(action_text, f"{context_text} {local_context}", str(fixed.get("outcome_type") or ""))
        if fixed["outcome_type"] == "verified_fix" and (not fixed.get("target_solution_id") or _looks_like_diagnostic_method(action_text) or _has_any(action_text, _NO_PROBLEM_MARKERS)):
            fixed["outcome_type"] = "pending_validation"
        if fixed["outcome_type"] != "verified_fix":
            fixed["target_solution_id"] = ""
        if not fixed.get("evidence_message_ids") and default_evidence:
            fixed["evidence_message_ids"] = default_evidence
        outcomes.append(fixed)
    if "diagnostic_outcomes" in sanitized:
        sanitized["diagnostic_outcomes"] = outcomes
    return sanitized

def _validate_deepseek_extraction(extraction: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    allowed = {"case_variant_candidate", "diagnostic_trace", "diagnostic_outcomes", "required_info_candidates", "split_decision"}
    for key in extraction:
        if key not in allowed:
            issues.append(f"unexpected_key:{key}")
    outcomes = extraction.get("diagnostic_outcomes") or []
    if not isinstance(outcomes, list):
        issues.append("diagnostic_outcomes_not_list")
        outcomes = []
    for idx, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            issues.append(f"diagnostic_outcome_not_object:{idx}")
            continue
        outcome_type = str(outcome.get("outcome_type") or "")
        if outcome_type not in OUTCOME_TYPES:
            issues.append(f"invalid_outcome_type:{idx}:{outcome_type}")
        if outcome_type == "verified_fix" and _has_action_failure(str(outcome.get("action_label") or "")):
            issues.append(f"verified_fix_conflicts_with_action_text:{idx}")
        if not outcome.get("evidence_message_ids"):
            issues.append(f"outcome_missing_evidence:{idx}")
    required = extraction.get("required_info_candidates") or []
    if not isinstance(required, list):
        issues.append("required_info_candidates_not_list")
    else:
        for idx, item in enumerate(required):
            if not isinstance(item, dict):
                issues.append(f"required_info_not_object:{idx}")
                continue
            slot = str(item.get("slot") or "other")
            if slot not in REQUIRED_INFO_SLOTS:
                issues.append(f"invalid_required_info_slot:{idx}:{slot}")
            if not item.get("evidence_message_ids"):
                issues.append(f"required_info_missing_evidence:{idx}")
    trace = extraction.get("diagnostic_trace")
    if trace is not None and not isinstance(trace, dict):
        issues.append("diagnostic_trace_not_object")
    variant = extraction.get("case_variant_candidate")
    if variant is not None and not isinstance(variant, dict):
        issues.append("case_variant_candidate_not_object")
    return sorted(set(issues))


def _merge_llm_trace(semantics: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    extraction = semantics.get("deepseek_extraction") if isinstance(semantics.get("deepseek_extraction"), dict) else {}
    llm_trace = extraction.get("diagnostic_trace") if isinstance(extraction.get("diagnostic_trace"), dict) else {}
    if not llm_trace:
        return trace
    merged = dict(trace)
    evidence = _list(semantics.get("evidence_ids"))[:12]
    if llm_trace.get("recommended_order"):
        merged["recommended_order"] = _normalise_llm_trace_steps(
            llm_trace.get("recommended_order"),
            trace.get("recommended_order") if isinstance(trace.get("recommended_order"), list) else [],
            evidence,
            str(semantics.get("semantic_text") or ""),
        )
    if llm_trace.get("actual_order"):
        merged["actual_order"] = _normalise_llm_trace_steps(
            llm_trace.get("actual_order"),
            trace.get("actual_order") if isinstance(trace.get("actual_order"), list) else [],
            evidence,
            str(semantics.get("semantic_text") or ""),
        )
    if llm_trace.get("summary"):
        merged["summary"] = _one_line(llm_trace.get("summary"), 240)
    return merged


def _normalise_llm_trace_steps(value: Any, fallback: list[Any], evidence: list[str], semantic_text: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return [x for x in fallback if isinstance(x, dict)]
    out: list[dict[str, Any]] = []
    fallback_steps = [x for x in fallback if isinstance(x, dict)]
    for idx, item in enumerate(value, start=1):
        base = fallback_steps[idx - 1] if idx - 1 < len(fallback_steps) else {}
        if isinstance(item, dict):
            label = _clean_chat_text(item.get("label") or item.get("action_label") or item.get("check") or base.get("label"), 180)
            check_id = str(item.get("check_id") or item.get("target_check_id") or base.get("check_id") or "")
            step_evidence = _list(item.get("evidence_message_ids")) or _list(base.get("evidence_message_ids")) or evidence
            order = int(item.get("order") or item.get("step_order") or idx)
        else:
            label = _clean_chat_text(item, 180) or str(base.get("label") or "")
            check_id = str(base.get("check_id") or "")
            step_evidence = _list(base.get("evidence_message_ids")) or evidence
            order = idx
        if not label and not check_id:
            continue
        terms = _outcome_context_terms(label)
        if label and (not terms or not any(term in semantic_text for term in terms)):
            continue
        out.append({
            "order": order,
            "check_id": check_id,
            "label": label or check_id,
            "evidence_message_ids": step_evidence,
        })
    return out or fallback_steps


def _sync_trace_steps_to_check_nodes(nodes: list[dict[str, Any]], trace: dict[str, Any]) -> None:
    """Use LLM-normalized trace labels as DiagnosticCheck labels.

    Deterministic clause splitting is deliberately conservative but often keeps
    whole chat clauses like "machine版本从8.0.2版本退回...".  When DeepSeek
    returns a clean trace, the check nodes should expose those normalized
    steps so W6 reviewers and read-side policy see the same semantics.
    """

    checks = [node for node in nodes if node.get("type") == "DiagnosticCheck"]
    if not checks or not isinstance(trace, dict):
        return
    steps = trace.get("actual_order") if isinstance(trace.get("actual_order"), list) and trace.get("actual_order") else trace.get("recommended_order")
    if not isinstance(steps, list) or not steps:
        return
    for idx, step in enumerate(steps[:len(checks)]):
        if isinstance(step, dict):
            label = _clean_chat_text(step.get("label") or step.get("action_label") or "", 120)
        else:
            label = _clean_chat_text(step, 120)
        if not label or not _is_review_grade_action(label):
            # Keep concise diagnostic nouns such as "检查驱动" even though they
            # are too short for some action filters.
            if not any(k in label for k in ("检查", "分析", "确认", "更换", "回退", "升级", "测试", "收集", "导出")):
                continue
        checks[idx]["label"] = label
        checks[idx]["how_to_check"] = label


def _merge_llm_outcomes(
    semantics: dict[str, Any],
    outcomes: list[dict[str, Any]],
    error_id: str,
    digest: str,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    extraction = semantics.get("deepseek_extraction") if isinstance(semantics.get("deepseek_extraction"), dict) else {}
    llm_outcomes = extraction.get("diagnostic_outcomes") if isinstance(extraction.get("diagnostic_outcomes"), list) else []
    if not llm_outcomes:
        return outcomes
    checks = [node for node in nodes if node.get("type") == "DiagnosticCheck"]
    check_ids = {str(node.get("check_id") or node.get("id") or "") for node in checks}
    solution_ids = {
        str(node.get("solution_id") or node.get("id") or "")
        for node in nodes
        if node.get("type") == "Solution"
    }
    out = list(outcomes)
    seen = {str(item.get("action_label") or "") for item in out}
    evidence_default = _list(semantics.get("evidence_ids"))[:30]
    for idx, item in enumerate(llm_outcomes, start=1):
        if not isinstance(item, dict):
            continue
        action = _clean_chat_text(item.get("action_label"), 180)
        if not action or action in seen:
            continue
        seen.add(action)
        check = checks[min(idx - 1, len(checks) - 1)] if checks else {}
        target_check_id = str(item.get("target_check_id") or "")
        if target_check_id not in check_ids:
            target_check_id = str(check.get("check_id") or check.get("id") or "")
        target_solution_id = str(item.get("target_solution_id") or "")
        if target_solution_id not in solution_ids:
            target_solution_id = ""
        outcome_id = f"outcome:candidate-{digest}:llm:{idx}"
        out.append({
            "type": "DiagnosticOutcome",
            "id": outcome_id,
            "outcome_id": outcome_id,
            "source_episode_id": semantics.get("source_episode_id") or "",
            "source_thread_id": semantics.get("source_thread_id") or "",
            "target_error_id": error_id,
            "target_check_id": target_check_id,
            "target_solution_id": target_solution_id,
            "action_label": action,
            "outcome_type": str(item.get("outcome_type") or "pending_validation"),
            "condition": str(item.get("condition") or ""),
            "high_cost": bool(item.get("high_cost")),
            "destructive": bool(item.get("destructive")),
            "observed_duration": str(item.get("observed_duration") or ""),
            "root_cause_summary": str(item.get("root_cause_summary") or ""),
            "needs_confirmation": bool(item.get("needs_confirmation")) or (_looks_like_fix_confirmation_question(action) and str(item.get("outcome_type") or "") == "verified_fix"),
            "evidence_message_ids": _list(item.get("evidence_message_ids")) or evidence_default,
            "source_offsets": semantics.get("source_offsets") or [],
            "proposal_only": True,
        })
    return out


def _merge_llm_case_variant(semantics: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    extraction = semantics.get("deepseek_extraction") if isinstance(semantics.get("deepseek_extraction"), dict) else {}
    llm_variant = extraction.get("case_variant_candidate") if isinstance(extraction.get("case_variant_candidate"), dict) else {}
    if not llm_variant:
        return variant
    merged = dict(variant)
    for key in ("label", "category", "subsystem", "scenario", "escalation_target"):
        if llm_variant.get(key):
            merged[key] = llm_variant[key]
    return merged


def _llm_required_info_candidates(semantics: dict[str, Any], matched: dict[str, Any] | None) -> list[dict[str, Any]]:
    extraction = semantics.get("deepseek_extraction") if isinstance(semantics.get("deepseek_extraction"), dict) else {}
    items = extraction.get("required_info_candidates") if isinstance(extraction.get("required_info_candidates"), list) else []
    out: list[dict[str, Any]] = []
    source_episode_id = str(semantics.get("source_episode_id") or "")
    target_error_id = str((matched or {}).get("error_id") or "")
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        slot = _normalize_llm_required_info_slot(item.get("slot"))
        evidence = _list(item.get("evidence_message_ids")) or _list(semantics.get("evidence_ids"))[:12]
        question = str(item.get("question") or QUESTION_TEMPLATES.get(slot, QUESTION_TEMPLATES["other"]))
        out.append({
            "candidate_id": f"reqinfo:{hashlib.sha1(f'{source_episode_id}:llm:{idx}:{slot}:{question}'.encode('utf-8')).hexdigest()[:12]}",
            "source_episode_id": source_episode_id,
            "source_thread_id": str(semantics.get("source_thread_id") or ""),
            "target_error_id": str(item.get("target_error_id") or target_error_id),
            "acceptable_error_ids": [str(item.get("target_error_id") or target_error_id)] if (item.get("target_error_id") or target_error_id) else [],
            "slot": slot,
            "label": str(item.get("label") or SLOT_LABELS.get(slot, SLOT_LABELS["other"])),
            "question": question,
            "why_required": str(item.get("why_required") or WHY_TEMPLATES.get(slot, WHY_TEMPLATES["other"])),
            "condition": str(item.get("condition") or ""),
            "priority": item.get("priority") or _priority(slot, str(item.get("condition") or ""), ""),
            "evidence_message_ids": evidence,
            "source_offsets": semantics.get("source_offsets") or [],
            "provided_later": bool(item.get("provided_later")),
            "provided_evidence_message_ids": _list(item.get("provided_evidence_message_ids"))[:12],
            "diagnostic_outcome_after_provided": _one_line(semantics.get("conclusion"), 240),
            "merge_policy": "append_to_required_info" if (item.get("target_error_id") or target_error_id) and slot != "other" else "review_only",
            "quality": item.get("quality") if isinstance(item.get("quality"), dict) else {},
            "source_request": {"text": "deepseek_structured_extraction"},
        })
    return out
