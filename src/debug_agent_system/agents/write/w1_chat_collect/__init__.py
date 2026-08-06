"""W1 chat collection and real Feishu/Xing archive adapter.

W1 owns only collection, message attribution, episode segmentation, and evidence
packaging.  It is deliberately stdlib-only and only reads bounded, allowlisted
resource hints through tool agents; it never writes KG data, performs OCR,
executes macros/scripts, or extracts archives to disk.
"""

from __future__ import annotations

import csv
from datetime import datetime
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from debug_agent_system.agents.tools import parse_attachment_evidence, parse_dmp_evidence, parse_document_evidence, parse_image_evidence, parse_jira_evidence, parse_log_package_evidence, parse_proj_evidence
from debug_agent_system.agents.write.people_roles import load_people_role_registry, people_index
from debug_agent_system.core.paths import project_root

JIRA_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}-\d+\b")
VERSION_RE = re.compile(r"(?<![\d.])(?:v)?\d{1,2}\.\d+(?:\.\d+){0,2}(?![\d.])", re.IGNORECASE)
LOG_FILE_RE = re.compile(
    r"[\w\-.\u4e00-\u9fff\[\]()（）]+\.(?:zip|rar|7z|log|evtx|dmp|proj|csv|toml|pml)",
    re.IGNORECASE,
)
DEVICE_RE = re.compile(r"\b(?:AOI|SPI|SI|X|SMT|PC|IPC)[-_A-Z0-9]*\d{3,}[A-Z0-9_-]*\b", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BRACKET_SITE_RE = re.compile(r"现场[【\[]([^】\]]+)[】\]]")
MENTION_RE = re.compile(r"@([^\s,，:：]+)")
URL_RE = re.compile(r"https?://[^\s\]\(\)（）），,。；;]+|www\.[^\s\]\(\)（）），,。；;]+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[(?P<label>.*?)\]\((?P<url>https?://[^)\s]+)\)", re.IGNORECASE | re.DOTALL)
FILE_TAG_RE = re.compile(r"<file\b[^>]*\bkey=[\"'](?P<key>[^\"']+)[\"'][^>]*\bname=[\"'](?P<name>[^\"']+)[\"'][^>]*/?>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
LOG_PACKAGE_EXTS = {".zip", ".rar", ".7z", ".log", ".evtx", ".dmp", ".pml"}
PROJECT_FILE_EXTS = {".proj"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
CONFIG_EXTS = {".toml", ".ini", ".cfg", ".json", ".yaml", ".yml", ".reg"}
DATA_FILE_EXTS = {".csv", ".txt", ".xls", ".xlsx", ".pdf", ".model"}
DOCUMENT_FILE_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
PROJECT_ROOT = project_root(__file__)
RESPONSIBILITY_FLOW_PATH = PROJECT_ROOT / "data" / "raw" / "现场问题反馈流程.md"
TEXT_HISTORY_SEGMENT_GAP_HOURS = 12
TEXT_HISTORY_SEGMENT_MAX_MESSAGES = 120
TEXT_HISTORY_SEGMENT_ID_FMT = "%Y%m%d%H%M%S"
TEXT_HISTORY_SOURCE = "text_jsonl_history"

USEFUL_KEYWORDS = (
    "客户反馈", "现场", "报错", "异常", "失败", "卡死", "闪退", "漏检", "误报", "无法", "不能",
    "检查", "排查", "定位", "原因", "解决", "恢复", "正常", "日志", "诊断数据", "远程", "版本",
    "相机", "光源", "工控机", "主程序", "服务", "IP", "ip",
)
FAULT_KEYWORDS = (
    "客户反馈", "报错", "异常", "失败", "卡死", "闪退", "漏检", "误报", "无法", "不能", "不出图",
    "打不开", "连不上", "初始化", "报警", "停机", "问题", "故障", "蓝屏", "黑屏", "重启",
    "卡顿", "延迟", "拍摄失败", "拍照失败", "残帧", "丢包", "马赛克",
)
ACTION_KEYWORDS = (
    "检查", "排查", "确认", "打开", "关闭", "重启", "升级", "回退", "替换", "拔插", "调整", "设置",
    "发", "提供", "上传", "远程", "看", "定位", "分析", "验证", "复现", "抓", "导出",
)
PROVIDED_INFO_MARKERS = (
    "已上传", "已提供", "已发", "发了", "上传了", "提供了", "已经上传", "已经提供", "已经发",
    "已导出", "导出了", "已经导出", "日志已", "诊断数据已", "数据包已",
)
CONCRETE_DIAGNOSTIC_VERBS = (
    "检查", "排查", "确认", "打开", "关闭", "设置", "替换", "拔插", "调整", "升级", "回退",
    "远程", "定位", "分析", "验证", "复现", "抓",
)
DIAGNOSTIC_QUERY_RE = re.compile(
    r"(?:查询|查看|搜索|检索).{0,16}(?:系统日志|事件日志|事件查看器|诊断日志|日志|DMP|dmp|错误码|报错信息)"
    r"|(?:系统日志|事件日志|事件查看器|诊断日志|日志|DMP|dmp|错误码|报错信息).{0,16}(?:查询|查看|搜索|检索)"
)
REBOOT_SYMPTOM_RE = re.compile(
    r"(?:自动|突然|无故|异常|蓝屏|黑屏|频繁|一直|正常测试中|运行中).{0,12}重启|"
    r"重启.{0,8}(?:报错|异常|失败|蓝屏|黑屏)|"
    r"重启之后.{0,12}(?:报错|打不开|异常)"
)
REBOOT_ACTION_RE = re.compile(r"(?:先|再|请|可以|建议|尝试|需要|执行|手动|断电)?重启(?:相机服务|服务|主程序|软件|程序|设备|机器|电脑|工控机)?")
CONCLUSION_KEYWORDS = (
    "已解决", "解决了", "恢复正常", "已恢复", "定位到", "根因", "原因是", "修复", "好了", "处理完成",
    "验证通过", "可以了", "没问题了", "解决方案",
)
INEFFECTIVE_OUTCOME_MARKERS = (
    "无效", "没有效果", "仍然出现", "依旧出现", "还是出现", "仍会", "还是会", "未解决", "没有解决", "验证失败",
)
OBSERVED_OUTCOME_MARKERS = (
    "恢复正常", "已恢复", "验证通过", "测试验证可以正常", "可以正常导出", "可以正常拍照", "运行正常",
    "测试正常", "拍摄正常", "未再出现", "不再出现", "未复现", "没有复现", "无异常情况出现", "没有异常情况",
)
MISSING_INFO_REQUEST_VERBS = (
    "提供", "上传", "补充", "导出", "截图", "打包", "发一下", "发下", "发我", "发给", "传一下", "给一下",
)
MISSING_INFO_OBJECTS = (
    "日志", "DLOG", "dlog", "诊断数据", "数据包", "版本", "IP", "ip", "报错", "错误码", "截图", "图片",
    "dmp", "DMP", "dump", "Dump", "转存储", "转储", "配方", "程序", "样本", "资料", "文件",
    "JIRA", "jira", "Jira", "工单", "缺陷单",
)
MISSING_INFO_INTERROGATIVE_OBJECTS = (
    "日志", "DLOG", "dlog", "诊断数据", "数据包", "版本", "IP", "ip", "报错", "错误码", "截图", "图片",
    "dmp", "DMP", "dump", "Dump", "转存储", "转储", "配方", "程序", "样本", "文件",
    "JIRA", "jira", "Jira", "工单", "缺陷单",
)
NOISE_KEYWORDS = (
    "谢谢", "收到", "辛苦", "好的", "ok", "OK", "会议", "排期", "需求", "jira", "JIRA", "工时", "上线",
    "拉群", "进群", "改群名", "joined", "invited", "updated the group name", "撤回",
)
PROJECT_NOISE_KEYWORDS = ("需求", "排期", "会议", "工时", "上线", "jira", "JIRA", "验收", "合同", "报价")
FAULT_FOCUS_NOISE_MARKERS = (
    "现场工作汇报", "现场工作汇总", "今日工作情况", "今日工作汇总", "每日反馈", "每日数据", "夜班数据返回",
    "建议可以", "看看能不能", "看有必要", "帮得上忙", "有关系吗", "工作汇总如下", "日常数据回传",
)
FAULT_STATUS_UPDATE_MARKERS = (
    "未再出现", "暂无", "暂未", "恢复正常", "正常开关机", "正常测试", "持续观察", "观察中", "已撤离现场",
    "没反馈", "后续观察", "到现在", "至今", "已更换", "更换完成后", "处理后应该",
)
MULTI_ISSUE_REPORT_MARKERS = (
    "异常点汇报", "异常点", "问题汇总", "问题如下", "问题收集", "异常汇总", "故障汇总", "客户问题与培训",
    "今日现状", "今日工作汇总", "今日设备状态", "现场情况反馈", "现场问题如下", "现场问题汇总", "异常反馈",
)
MULTI_ISSUE_EXCLUDE_MARKERS = (
    "解决步骤", "解决方案", "处理方案", "检查方案", "排查原因", "可能原因", "原因：", "原因:", "故障原理",
    "从转存储文件来看", "从日志", "日志分析", "总结就是", "结合以上信息", "诊断信息",
    "各位领导", "现场工作", "今日情况如下", "每日反馈", "每日数据", "工作汇报", "上传网盘",
    "请领导知悉", "以上信息请", "以上请",
)
MULTI_ISSUE_PROCEDURE_MARKERS = (
    "解决步骤", "解决方案", "处理方案", "检查方案", "排查原因", "可能原因", "原因：", "原因:",
    "故障原理", "从转存储文件来看", "结合以上信息", "诊断信息",
)
MULTI_ISSUE_KEYWORDS = (
    "报错", "异常", "失败", "卡死", "闪退", "漏检", "误报", "无法", "不能", "蓝屏", "白屏", "重启",
    "时间长", "出结果慢", "响应延迟", "等待", "计算时间", "IP互换", "ip互换", "识别不准", "识别不精准", "虚焊", "检测",
    "黑屏", "卡住", "卡顿", "崩溃", "打不开", "连不上",
)
NUMBERED_ITEM_RE = re.compile(r"(?:(?<=^)|(?<=[\s\n。；;：:]))([1-9]\d?)[、.．:：/](?!\d)\s*")
REPORT_HEADING_RE = re.compile(r"(?:(?<=^)|(?<=[\s\n。；;：:]))([一二三四五六七八九十]+)[、.．:：，,]\s*")
REPORT_END_MARKERS = ("以上信息请", "以上请", "请各位领导知悉", "请领导知悉")
RESPONSIBILITY_RULE_FALLBACK = {
    "owner_name": "工程师午",
    "owner_handle": "@工程师午",
    "category": "其他问题及无法分类问题",
    "description": "如卡顿等无法明确分类的问题。",
}
RESPONSIBILITY_CATEGORY_RULES = {
    "软件功能咨询/需求类": ("咨询", "需求", "新需求", "是否支持", "支持某功能", "功能支持"),
    "主程序软件问题": ("主程序", "软件异常", "闪退", "花屏", "产品功能异常", "打不开", "崩溃"),
    "复判站软件问题": ("复判", "复判站", "复判软件", "复判页面"),
    "软件使用及调试问题": ("误报", "漏检", "调试", "参数", "答疑", "功能使用", "算法参数", "全局参数"),
    "工控机/复判站/编程站及操作系统问题": (
        "蓝屏", "死机", "自动重启", "无法开机", "黑屏", "内存", "显卡", "系统中毒", "工控机", "操作系统", "网络问题",
    ),
    "运控问题": (
        "拍摄", "拍照", "不拍照", "拍照停顿", "拍照失败", "成像异常", "进板", "出板", "停板", "飞板",
        "HERMES", "hermes", "运控", "运控程序", "运动控制卡", "运控闪退",
    ),
    "标定问题（原则上设备出厂标定不应该有问题，如果需要研发支持，找@刘亚林）": (
        "标定", "原点标定", "角度标定", "大板标定", "光源一致性",
    ),
    "硬件问题": ("传感器", "电机", "流道", "光源", "打光", "硬件故障", "ARM"),
    "模型优化问题": (
        "模型", "整版", "ODA", "二维码", "OCR", "缺陷", "SINGLEPIN", "PAD", "body", "极性", "红胶", "金手指",
    ),
    "3D成像问题": ("3D", "3d", "点云", "成像参数"),
    "MES问题": ("MES", "BAK"),
    "SPC问题": ("SPC",),
    "Buddy问题": ("Buddy", "buddy", "CFX", "EAP", "500错误"),
    "迁移工具": ("迁移", "0.x", "1.x", "proj迁移", "旧proj", "res归档", "统计数据迁移"),
    "外部对接设备": ("扫码枪",),
    "其他问题及无法分类问题": ("卡顿",),
}
REPORTER_SIGNAL_MARKERS = (
    "现场反馈", "现场情况反馈", "客户反馈", "每日反馈", "现场工作", "已到现场", "客户现场", "今日反馈", "明天的工作",
    "今日现状", "今日工作汇总", "现场问题如下", "异常反馈",
    "现场版本已更新", "现场先使用", "待问题解决后升级", "验证现场", "已验证",
)
REPORTER_JIRA_MARKERS = ("已提交JIRA", "已提交Jira", "已提交jira", "提交JIRA", "提交Jira", "提交jira")
OWNER_REQUEST_MARKERS = ("看下", "处理", "排查", "确认", "负责", "支持", "分析", "定位", "跟进", "安排")
OWNER_TAKEOVER_MARKERS = ("我看下", "我来处理", "我来跟进", "我排查", "我确认下", "我负责", "我这边处理", "我来分析")
DAILY_REPORT_MARKERS = (
    "每日反馈", "现场工作", "明天的工作", "今日反馈", "今日现状", "今日工作汇总", "今日设备状态",
    "现场工作：", "现场情况反馈", "问题反馈：", "现场问题如下", "现场问题汇总", "异常反馈",
)
FIELD_REPORT_ACTIVE_FAULT_MARKERS = (
    "报错", "异常", "失败", "卡死", "闪退", "漏检", "漏报", "无法", "不能", "蓝屏", "黑屏", "自动重启",
    "卡顿", "不拍照", "拍摄失败", "拍照失败", "卡板", "识别不准", "识别不到", "未检出", "无检测框",
    "识别不佳", "不显示", "不一致", "偏位", "经常满", "误报过高", "误报增加", "误报无法降低", "出结果慢",
    "结果等待时间长", "响应延迟", "等待时间较长", "初始化失败",
)
FIELD_REPORT_STATUS_MARKERS = (
    "未再出现", "暂未出现", "未出现", "未发现异常", "无异常", "无卡板", "恢复正常", "已恢复", "已解决", "正常使用", "正常生产", "持续观察", "跟踪无异常",
)
FIELD_REPORT_WORK_MARKERS = (
    "培训", "讲解", "教学视频", "工作计划", "前往", "到达现场", "辅助客户", "安装成功", "版本已更新", "版本升级完成",
    "追溯", "整合文档", "设备稳定性", "咨询客户", "申请加密狗", "日常数据",
)
JIRA_SUBMISSION_MARKERS = ("已提交JIRA", "已提交Jira", "已提交jira", "提交JIRA", "提交Jira", "提交jira", "JIRA如下", "jira如下")
RESPONSIBILITY_DEBUG_CONTEXT_MARKERS = (
    "日志", "版本", "IP", "相机", "运控", "蓝屏", "重启", "误报", "漏检", "标定", "光源",
    "主程序", "JIRA", "jira", "工单", "报错", "异常", "失败", "拍照", "拍摄", "闪退",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _chat_id_from_segment(segment_id: str) -> str:
    match = re.match(r"^(oc_[^_]+)_", segment_id or "")
    return match.group(1) if match else ""


def _clean_text(text: Any) -> str:
    """Clean Feishu post/html noise while preserving paragraph boundaries."""
    raw = str(text or "")
    if not raw:
        return ""
    raw = html.unescape(raw)
    raw = re.sub(r"</(?:p|div|li|h\d)>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<(?:br|br/)\s*/?>", "\n", raw, flags=re.IGNORECASE)
    raw = TAG_RE.sub("", raw)
    raw = URL_RE.sub("", raw)
    lines = [" ".join(line.replace("\r", " ").split()) for line in raw.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _normalize_url(url: Any) -> str:
    value = str(url or "").strip().rstrip(")，,。；;]")
    jira_match = re.match(r"(?P<url>https?://[^\s]+?/browse/[A-Z][A-Z0-9]{2,}-\d+)", value, re.IGNORECASE)
    return jira_match.group("url") if jira_match else value


def _link_type(url: str, label: str = "") -> str:
    text = f"{url} {label}".lower()
    if "jira" in text or JIRA_RE.search(url) or JIRA_RE.search(label):
        return "jira"
    if "feishu" in text or "larksuite" in text:
        return "feishu"
    return "other"


def _extract_links(raw_text: Any, *, message_id: str = "", thread_id: str = "") -> list[dict[str, Any]]:
    """Extract link metadata before URL stripping.

    W1 records URLs as evidence metadata only.  It never fetches linked content.
    """
    raw = str(raw_text or "")
    links: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for match in MARKDOWN_LINK_RE.finditer(raw):
        url = _normalize_url(match.group("url"))
        if not url or url in seen_urls:
            continue
        label = _one_line(_clean_text(match.group("label")), 240)
        seen_urls.add(url)
        links.append({
            "url": url,
            "label": label,
            "type": _link_type(url, label),
            "message_id": message_id,
            "thread_id": thread_id,
            "source": "message.raw_content.link",
            "start": match.start("url"),
            "end": match.end("url"),
            "reason": "metadata_only_not_fetched",
        })
    for match in URL_RE.finditer(raw):
        url = _normalize_url(match.group(0))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        links.append({
            "url": url,
            "label": "",
            "type": _link_type(url),
            "message_id": message_id,
            "thread_id": thread_id,
            "source": "message.raw_content.url",
            "start": match.start(),
            "end": match.end(),
            "reason": "metadata_only_not_fetched",
        })
    return links


def _extract_embedded_file_metadata(raw_text: Any, *, message_id: str = "", thread_id: str = "") -> list[dict[str, Any]]:
    """Recover metadata-only files even when the crawler did not copy bytes."""
    raw = str(raw_text or "")
    rows: list[dict[str, Any]] = []
    for match in FILE_TAG_RE.finditer(raw):
        name = html.unescape(str(match.group("name") or "")).strip()
        file_key = str(match.group("key") or "").strip()
        if not name and not file_key:
            continue
        rows.append({
            "file_key": file_key or name,
            "kind": "file",
            "name": name or file_key,
            "mime": None,
            "size": 0,
            "path": None,
            "status": "metadata_only",
            "source_status": "embedded_file_tag_not_copied",
            "extension": Path(name or file_key).suffix.lower(),
            "evidence_role": _attachment_evidence_role(name or file_key, "file"),
            "message_id": message_id,
            "thread_id": thread_id,
            "reason": "message_raw_content_file_tag_metadata_only",
        })
    return rows


def _attachment_evidence_role(name: Any, kind: Any = "") -> str:
    ext = Path(str(name or "")).suffix.lower()
    kind_text = str(kind or "").lower()
    if ext in PROJECT_FILE_EXTS:
        return "program_file"
    if ext in LOG_PACKAGE_EXTS:
        return "log_package"
    if kind_text == "image" or ext in IMAGE_EXTS:
        return "sample_image"
    if ext in CONFIG_EXTS:
        return "environment"
    if ext in DATA_FILE_EXTS:
        return "data_file"
    return "attachment"


def _attachment_evidence(att: dict[str, Any]) -> dict[str, Any]:
    name = str(att.get("name") or att.get("file_key") or "")
    role = str(att.get("evidence_role") or _attachment_evidence_role(name, att.get("kind")))
    return {
        "file_key": att.get("file_key", ""),
        "name": name,
        "kind": att.get("kind") or "file",
        "extension": att.get("extension") or Path(name).suffix.lower(),
        "evidence_role": role,
        "size": att.get("size", 0),
        "path": att.get("path"),
        "status": att.get("status", ""),
        "source_status": att.get("source_status", ""),
        "message_id": att.get("message_id", ""),
        "thread_id": att.get("thread_id", ""),
        "reason": att.get("reason", "pre_crawled_resource_metadata_only"),
    }


def _one_line(text: Any, limit: int = 500) -> str:
    compact = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    return compact[:limit]


def _unique(values: Iterable[Any], *, limit: int = 50) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _parse_md_table_cell(text: str) -> str:
    clean = str(text or "").replace("\\", "")
    clean = clean.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    clean = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip(" -_，,。；;|")


def _load_responsibility_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if RESPONSIBILITY_FLOW_PATH.exists():
        for raw_line in RESPONSIBILITY_FLOW_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith("|@"):
                continue
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) < 3:
                continue
            owner_name = _parse_md_table_cell(parts[0]).lstrip("@")
            category = _parse_md_table_cell(parts[1])
            description = _parse_md_table_cell(parts[2])
            if not owner_name or not category:
                continue
            rules.append({
                "owner_name": owner_name,
                "owner_handle": f"@{owner_name}",
                "category": category,
                "description": description,
            })
    if not rules:
        rules.append(dict(RESPONSIBILITY_RULE_FALLBACK))
    if not any(rule.get("category") == RESPONSIBILITY_RULE_FALLBACK["category"] for rule in rules):
        rules.append(dict(RESPONSIBILITY_RULE_FALLBACK))
    return rules


def _text_history_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _text_history_segment_id(chat_id: str, start_time: str, end_time: str, index: int) -> str:
    start = _text_history_timestamp(start_time)
    end = _text_history_timestamp(end_time)
    start_key = start.strftime(TEXT_HISTORY_SEGMENT_ID_FMT) if start is not None else "00000000000000"
    end_key = end.strftime(TEXT_HISTORY_SEGMENT_ID_FMT) if end is not None else start_key
    return f"{chat_id}_{start_key}_{end_key}_{index}"


def _message_is_daily_report(text: str) -> bool:
    clean = str(text or "")
    return any(marker in clean for marker in DAILY_REPORT_MARKERS)


def _message_has_jira_submission_signal(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if any(marker in text for marker in JIRA_SUBMISSION_MARKERS):
        return True
    if JIRA_RE.search(text) and any(marker in text for marker in ("已提交", "提交", "工单")):
        return True
    return any(link.get("type") == "jira" for link in msg.get("links") or [] if isinstance(link, dict))


def _has_responsibility_debug_context(text: str) -> bool:
    clean = str(text or "")
    return _has_fault_symptom_signal(clean) or any(marker in clean for marker in RESPONSIBILITY_DEBUG_CONTEXT_MARKERS)


def _message_is_reporter_signal(msg: dict[str, Any]) -> tuple[bool, list[str]]:
    text = str(msg.get("text") or "")
    reasons: list[str] = []
    if any(marker in text for marker in REPORTER_SIGNAL_MARKERS):
        reasons.append("field_feedback")
    if _message_has_jira_submission_signal(msg):
        reasons.append("jira_submission")
    if any(marker in text for marker in ("已收集", "已上传", "已提供", "已验证", "复测", "升级完成")):
        reasons.append("evidence_collection_or_validation")
    if reasons and not (_has_responsibility_debug_context(text) or _message_has_jira_submission_signal(msg)):
        return False, []
    return bool(reasons), reasons


def _message_has_owner_assignment_signal(msg: dict[str, Any]) -> tuple[bool, list[str]]:
    text = str(msg.get("text") or "")
    mentions = [item for item in msg.get("mentions") or [] if isinstance(item, dict) and str(item.get("name") or "").strip()]
    reasons: list[str] = []
    clauses = _missing_info_request_clauses(text)
    if mentions:
        for clause in clauses:
            if not any(f"@{item.get('name')}" in clause or str(item.get("name") or "") in clause for item in mentions):
                continue
            if any(marker in clause for marker in OWNER_REQUEST_MARKERS) and _has_responsibility_debug_context(clause):
                reasons.append("direct_owner_request")
                break
    if any(marker in text for marker in ("转发给", "转给", "对应负责人", "责任人", "负责人")) and _has_responsibility_debug_context(text):
        reasons.append("responsibility_transfer")
    return bool(reasons), reasons


def _message_has_owner_takeover_signal(msg: dict[str, Any]) -> tuple[bool, list[str]]:
    text = str(msg.get("text") or "")
    reasons = [marker for marker in OWNER_TAKEOVER_MARKERS if marker in text and _has_responsibility_debug_context(text)]
    if not reasons and _is_diagnostic_action(msg) and any(marker in text for marker in ("我这边", "我先", "我去")) and _has_responsibility_debug_context(text):
        reasons.append("diagnostic_takeover")
    return bool(reasons), reasons


def _signal_offsets(field: str, markers: tuple[str, ...], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    for msg in messages:
        content = str(msg.get("text") or "")
        for marker in markers:
            start = content.find(marker)
            if start < 0:
                continue
            offsets.append({
                "field": field,
                "value": marker,
                "source": "message.text",
                "message_id": msg.get("message_id", ""),
                "thread_id": msg.get("thread_id", ""),
                "start": start,
                "end": start + len(marker),
            })
    return offsets


def _aggregate_role_candidates(items: list[dict[str, Any]], *, role_type: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
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
        target["reason"] = _unique([*target.get("reason", []), *item.get("reason", [])], limit=10)
        target["evidence_message_ids"] = _unique([*target.get("evidence_message_ids", []), *item.get("evidence_message_ids", [])], limit=20)
    out = list(grouped.values())
    out.sort(key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("name") or "")))
    return out


def _classification_hypotheses(messages: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus = "\n".join(str(msg.get("text") or "") for msg in messages)
    evidence_ids = _unique((msg.get("message_id") for msg in messages if _message_useful(msg)), limit=20)
    if not any(_has_responsibility_debug_context(str(msg.get("text") or "")) or _message_has_jira_submission_signal(msg) for msg in messages):
        fallback = dict(RESPONSIBILITY_RULE_FALLBACK)
        return [{
            "name": fallback["owner_name"],
            "role_type": "issue_owner",
            "problem_category": fallback["category"],
            "confidence": 0.2,
            "reason": ["fallback_unclassified"],
            "evidence_message_ids": evidence_ids,
        }]
    hypotheses: list[dict[str, Any]] = []
    for rule in rules:
        category = str(rule.get("category") or "")
        keywords = RESPONSIBILITY_CATEGORY_RULES.get(category, ())
        matched = [keyword for keyword in keywords if keyword and keyword in corpus]
        if not matched:
            continue
        confidence = min(1.0, 0.35 + 0.1 * len(matched))
        hypotheses.append({
            "name": str(rule.get("owner_name") or ""),
            "role_type": "issue_owner",
            "problem_category": category,
            "confidence": round(confidence, 4),
            "reason": [f"matched:{keyword}" for keyword in matched[:8]],
            "evidence_message_ids": evidence_ids,
        })
    if not hypotheses:
        fallback = dict(RESPONSIBILITY_RULE_FALLBACK)
        hypotheses.append({
            "name": fallback["owner_name"],
            "role_type": "issue_owner",
            "problem_category": fallback["category"],
            "confidence": 0.2,
            "reason": ["fallback_unclassified"],
            "evidence_message_ids": evidence_ids,
        })
    hypotheses.sort(key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("problem_category") or "")))
    return hypotheses[:5]


def _build_attribution(messages: list[dict[str, Any]]) -> dict[str, Any]:
    rules = _load_responsibility_rules()
    reporter_events: list[dict[str, Any]] = []
    owner_events: list[dict[str, Any]] = []
    owner_assignments: list[dict[str, Any]] = []
    responsibility_signals: list[dict[str, Any]] = []
    requested_names: set[str] = set()
    for msg in messages:
        sender = msg.get("sender") or {}
        sender_name = str(sender.get("name") or "").strip()
        text = str(msg.get("text") or "")
        message_id = str(msg.get("message_id") or "")
        is_reporter, reporter_reasons = _message_is_reporter_signal(msg)
        if is_reporter and sender_name:
            reporter_events.append({
                "name": sender_name,
                "confidence": min(1.0, 0.45 + 0.15 * len(reporter_reasons)),
                "reason": reporter_reasons,
                "evidence_message_ids": [message_id],
            })
            responsibility_signals.append({
                "message_id": message_id,
                "signal_type": "reporter_signal",
                "sender": sender_name,
                "content_summary": _one_line(text, 160),
                "reason": reporter_reasons,
            })
        has_assignment, assignment_reasons = _message_has_owner_assignment_signal(msg)
        if has_assignment:
            for mention in msg.get("mentions") or []:
                if not isinstance(mention, dict):
                    continue
                name = str(mention.get("name") or "").strip()
                if not name:
                    continue
                requested_names.add(name)
                event = {
                    "name": name,
                    "role_type": "issue_owner",
                    "confidence": 0.8,
                    "reason": assignment_reasons,
                    "evidence_message_ids": [message_id],
                }
                owner_events.append(dict(event))
                owner_assignments.append(dict(event))
                responsibility_signals.append({
                    "message_id": message_id,
                    "signal_type": "owner_assignment",
                    "sender": sender_name,
                    "name": name,
                    "content_summary": _one_line(text, 160),
                    "reason": assignment_reasons,
                })
        has_takeover, takeover_reasons = _message_has_owner_takeover_signal(msg)
        if has_takeover and sender_name and (sender_name in requested_names or _is_diagnostic_action(msg)):
            owner_events.append({
                "name": sender_name,
                "role_type": "issue_owner",
                "confidence": 0.7,
                "reason": takeover_reasons or ["diagnostic_takeover"],
                "evidence_message_ids": [message_id],
            })
            responsibility_signals.append({
                "message_id": message_id,
                "signal_type": "owner_takeover",
                "sender": sender_name,
                "content_summary": _one_line(text, 160),
                "reason": takeover_reasons or ["diagnostic_takeover"],
            })
    return {
        "reporter_candidates": _aggregate_role_candidates(reporter_events, role_type="reporter"),
        "owner_candidates": _aggregate_role_candidates(owner_events, role_type="issue_owner"),
        "owner_assignments": owner_assignments[:20],
        "responsibility_signals": responsibility_signals[:50],
        "classification_hypotheses": _classification_hypotheses(messages, rules),
    }


def _site_from_chat_name(chat_name: str) -> str:
    text = str(chat_name or "").strip()
    if not text:
        return ""
    if "】" in text:
        text = text.split("】", 1)[1]
    text = re.sub(r"（.*?）|\(.*?\)", "", text)
    for suffix in ("项目沟通群", "项目群", "沟通群", "交流群", "群"):
        if suffix in text:
            text = text.split(suffix, 1)[0]
            break
    return text.strip(" -_，,。；;")


def _find_offsets(field: str, pattern: re.Pattern[str], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    for msg in messages:
        content = str(msg.get("text") or "")
        for match in pattern.finditer(content):
            offsets.append({
                "field": field,
                "value": match.group(0),
                "source": "message.text",
                "message_id": msg.get("message_id", ""),
                "thread_id": msg.get("thread_id", ""),
                "start": match.start(),
                "end": match.end(),
            })
    return offsets


def _message_useful(msg: dict[str, Any]) -> bool:
    if str(msg.get("msg_type") or "") == "system":
        return False
    text = str(msg.get("text") or "")
    if not text:
        return bool(msg.get("attachments"))
    return any(k in text for k in USEFUL_KEYWORDS) or bool(msg.get("attachments"))


def _message_has_missing_info_request(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    not_request_markers = (
        "补充JIRA", "补充jira", "JIRA如下", "jira如下", "远程码", "远程代码", "远程ID", "远程id",
        "commit还不对", "没发给过你", "我没发给过你", "怎么会用到",
        "请查阅", "每日反馈", "现场工作汇总", "现场工作内容", "工作汇报", "今日现状已更新",
        "各位领导", "请领导知悉", "以上请领导知悉",
    )
    direct_request_markers = (
        "请提供", "麻烦提供", "辛苦提供", "请上传", "麻烦上传", "请补充", "麻烦补充",
        "请导出", "麻烦导出", "请截图", "麻烦截图", "请打包", "麻烦打包",
        "发我", "发给我", "给我看看", "给我看下", "给我看一下",
    )
    if any(k in clean for k in not_request_markers):
        # A report can still contain a concrete ask in a later clause; do not
        # use this as a whole-message hard fail.
        pass
    provided_markers = (
        "已上传", "已提供", "已发", "发了", "上传了", "提供了", "已经上传", "已经提供", "已经发",
        "已导出", "导出了", "已经导出", "日志里有", "有windows事件导出", "有Windows事件导出", "有日志",
        "日志已", "诊断数据已", "数据包已", "将日志发给", "把日志发给",
    )
    resolution_statement = any(k in clean for k in ("可以正常", "正常使用", "恢复正常", "已解决", "解决了", "可以用了"))
    if clean.startswith("补充") and resolution_statement:
        return False
    report_like = any(k in clean for k in not_request_markers)
    for clause in _missing_info_request_clauses(clean):
        if any(k in clause for k in provided_markers) and not any(k in clause for k in ("还需要", "是否还需要", "需要再", "再发")):
            continue
        if report_like and not any(k in clause for k in direct_request_markers):
            continue
        if _is_missing_info_request_clause(clause):
            return True
    return False


def _missing_info_request_clauses(text: str) -> list[str]:
    raw = TAG_RE.sub(" ", str(text or ""))
    parts = re.split(r"[。！？!?；;\n，,]+", raw)
    clauses: list[str] = []
    for part in parts:
        clause = _one_line(part, 260).strip(" -_，,。；;:：()（）")
        if clause:
            clauses.append(clause)
    return clauses or [_one_line(text, 260)]


def _is_missing_info_request_clause(clause: str) -> bool:
    if any(k in clause for k in (
        "无法提供", "不能提供", "提供程序", "点击提供程序", "按提供程序", "客户提供", "厂家提供",
        "你提供的", "您提供的", "已使用", "提供给研发", "提供给产研", "提供给客户", "没有关系",
    )):
        return False
    has_object = any(k in clause for k in MISSING_INFO_OBJECTS) or bool(LOG_FILE_RE.search(clause) or VERSION_RE.search(clause) or IP_RE.search(clause))
    if not has_object:
        return False
    # Version/action approval and troubleshooting advice questions are not
    # "please provide information" requests.
    if re.search(r"(更新|升级|安装|回退).{0,24}(?:版本|v?\d+\.\d+).{0,12}(可以吗|可不可以|能不能|是否可以)", clause, re.IGNORECASE):
        return False
    if re.search(r"(有没有|是否有|有无).{0,12}(好的|合适的|推荐的).{0,12}(驱动|软件)?版本", clause):
        return False
    if "什么原因" in clause and not any(k in clause for k in MISSING_INFO_REQUEST_VERBS):
        return False
    explicit_request = any(k in clause for k in (
        "请", "麻烦", "辛苦", "能否", "可以发", "发一下", "发下", "发我", "给我", "补充一下", "导出一下", "上传一下", "截图一下", "打包一下",
    ))
    if not explicit_request and re.search(r"(?:导出|上传|提供|发送).{0,24}(?:后|了|已|显示|报错|失败|正常)", clause):
        return False
    if any(k in clause for k in MISSING_INFO_REQUEST_VERBS):
        return True
    interrogative = "?" in clause or "？" in clause or any(k in clause for k in ("有没有", "是否有", "有无", "还需要", "需要再"))
    if not interrogative:
        return False
    object_alt = "|".join(re.escape(k) for k in MISSING_INFO_INTERROGATIVE_OBJECTS)
    return bool(
        re.search(rf"(有没有|是否有|有无|还需要|需要再).{{0,16}}(?:{object_alt})", clause)
        or re.search(rf"(?:{object_alt}).{{0,16}}(有没有|是否有|有无|还需要|需要再)", clause)
    )


def _message_has_provided_info(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if msg.get("attachments"):
        return True
    if any(link.get("type") == "jira" for link in msg.get("links") or [] if isinstance(link, dict)):
        return True
    return bool(
        VERSION_RE.search(text)
        or LOG_FILE_RE.search(text)
        or JIRA_RE.search(text)
        or IP_RE.search(text)
        or any(k in text for k in ("截图", "图片", "版本", "日志", "诊断数据", "DLOG", "dlog", "dmp", "DMP", "报错", "错误码", "IP", "ip", "JIRA", "jira", "Jira", "工单", "缺陷单"))
    )


def _provided_evidence_brief(msg: dict[str, Any]) -> dict[str, Any]:
    text = str(msg.get("text") or "")
    raw_attachments = [att for att in msg.get("attachments") or [] if isinstance(att, dict)]
    attachment_metadata = [_attachment_evidence(att) for att in raw_attachments]
    links = [link for link in msg.get("links") or [] if isinstance(link, dict)]
    link_text = " ".join(str(link.get("url") or "") + " " + str(link.get("label") or "") for link in links)
    ip_values = _unique(IP_RE.findall(text), limit=20)
    jira_ids = _unique([*JIRA_RE.findall(text), *JIRA_RE.findall(link_text)], limit=20)
    return {
        "message_id": str(msg.get("message_id") or ""),
        "sender": msg.get("sender") or {},
        "create_time": str(msg.get("create_time") or ""),
        "content_summary": _one_line(text, 240),
        "attachment_metadata": attachment_metadata,
        "links": links,
        "tool_evidence": _build_tool_evidence(raw_attachments, links, jira_ids),
        "text_hints": {
            "jira_ids": jira_ids,
            "versions": _unique((value for value in VERSION_RE.findall(text) if value not in ip_values), limit=20),
            "ip_config": ip_values,
            "log_paths": _unique((m.group(0) for m in LOG_FILE_RE.finditer(text) if _attachment_evidence_role(m.group(0), "file") == "log_package"), limit=20),
            "project_files": _unique((m.group(0) for m in LOG_FILE_RE.finditer(text) if _attachment_evidence_role(m.group(0), "file") == "program_file"), limit=20),
        },
    }


def _is_fault_description(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if str(msg.get("msg_type") or "") == "system":
        return False
    if msg.get("field_report_item_kind") == "fault_case":
        return True
    if msg.get("fragment_count") and any(k in text for k in MULTI_ISSUE_KEYWORDS):
        return True
    if _message_has_missing_info_request(text) or _is_provided_info_statement(msg):
        return False
    if _is_resolution(msg):
        return False
    if _has_concrete_diagnostic_action(text) and not _has_fault_symptom_signal(text):
        return False
    if _is_generic_problem_statement(text):
        return False
    return any(k in text for k in FAULT_KEYWORDS) or bool(JIRA_RE.search(text) and any(k in text for k in ("报错", "异常", "失败", "问题")))


def _has_fault_symptom_signal(text: str) -> bool:
    clean = str(text or "")
    return any(k in clean for k in (
        "客户反馈", "现场反馈", "报错", "异常", "失败", "蓝屏", "黑屏", "突然重启", "自动重启",
        "无故重启", "卡死", "闪退", "漏检", "误报", "延迟", "卡顿", "残帧", "丢包", "马赛克",
        "无法", "不能", "不出图", "打不开", "连不上",
    ))


_GENERIC_PROBLEM_PATTERNS = (
    "不是软件问题",
    "有个问题",
    "这个问题",
    "问题点",
    "问题本质",
    "问题已",
    "问题暂时",
    "之前的设备问题有点多",
)


def _is_generic_problem_statement(text: str) -> bool:
    clean = _one_line(text, 260)
    if not clean:
        return False
    strong = any(k in clean for k in (
        "报错", "异常", "失败", "蓝屏", "黑屏", "重启", "卡死", "闪退", "漏检", "误报",
        "延迟", "卡顿", "拍摄失败", "拍照失败", "残帧", "丢包", "无法", "不能",
    ))
    if strong:
        return False
    return any(k in clean for k in _GENERIC_PROBLEM_PATTERNS)


def _is_diagnostic_action(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if str(msg.get("msg_type") or "") == "system":
        return False
    if _is_pure_handoff(text):
        return False
    if _message_has_missing_info_request(text):
        return False
    if _is_provided_info_statement(msg):
        return False
    return _has_concrete_diagnostic_action(text)


def _is_provided_info_statement(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if any(k in text for k in PROVIDED_INFO_MARKERS):
        return True
    if msg.get("attachments") and not any(k in text for k in CONCRETE_DIAGNOSTIC_VERBS):
        return True
    return False


def _has_concrete_diagnostic_action(text: str) -> bool:
    if any(k in text for k in CONCRETE_DIAGNOSTIC_VERBS):
        return True
    if DIAGNOSTIC_QUERY_RE.search(str(text or "")):
        return True
    if "重启" in text and REBOOT_ACTION_RE.search(text) and not REBOOT_SYMPTOM_RE.search(text):
        return True
    return False


def _is_pure_handoff(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    handoff = any(k in clean for k in ("帮忙看", "看一下", "排查一下原因", "看看是什么问题", "麻烦"))
    concrete = any(k in clean for k in ("检查", "确认", "导出", "提供", "上传", "重启", "设置", "替换", "拔插", "日志", "版本", "IP", "截图", "报错"))
    return handoff and not concrete


def _is_resolution(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if str(msg.get("msg_type") or "") == "system":
        return False
    if any(k in text for k in ("没有检测到原因", "未找到原因", "看一下是什么问题", "麻烦", "请提供")):
        return False
    if "解决方案" in text and "等待解决方案" not in text:
        return True
    if any(k in text for k in INEFFECTIVE_OUTCOME_MARKERS):
        return True
    if any(k in text for k in OBSERVED_OUTCOME_MARKERS):
        return True
    if any(k in text for k in ("已解决", "解决了", "恢复正常", "已恢复", "处理完成", "验证通过", "可以了", "没问题了")):
        return True
    if any(k in text for k in ("原因是", "根因", "定位到")) and any(k in text for k in ("恢复", "解决", "修复", "正常")):
        return True
    return False


def _is_noise(msg: dict[str, Any]) -> bool:
    text = str(msg.get("text") or "")
    if str(msg.get("msg_type") or "") == "system":
        return True
    if _is_resolution(msg):
        return False
    if not text and not msg.get("attachments"):
        return True
    if any(k in text for k in NOISE_KEYWORDS) and not any(k in text for k in USEFUL_KEYWORDS):
        return True
    if any(k in text for k in PROJECT_NOISE_KEYWORDS) and not any(k in text for k in FAULT_KEYWORDS + ACTION_KEYWORDS + CONCLUSION_KEYWORDS):
        return True
    return False


def _is_fault_focus_text(text: str) -> bool:
    clean = _one_line(text, 260)
    if not clean:
        return False
    if not _has_fault_symptom_signal(clean):
        return False
    if any(marker in clean for marker in FAULT_FOCUS_NOISE_MARKERS):
        return False
    if clean.startswith("@") and not any(k in clean for k in ("报错", "异常", "失败", "蓝屏", "黑屏", "重启", "卡死", "闪退", "漏检", "误报", "延迟", "卡顿", "拍摄失败", "拍照失败", "残帧", "丢包", "图片为空", "空图", "错位")):
        return False
    return True


def _fault_focus_score(text: str) -> int:
    clean = _one_line(text, 260)
    if not _is_fault_focus_text(clean):
        return -100
    score = 0
    for marker in ("报错", "异常", "失败", "蓝屏", "黑屏", "重启", "卡死", "闪退", "拍摄失败", "拍照失败", "图片为空", "空图", "误报", "漏检", "错位"):
        if marker in clean:
            score += 5
    if len(clean) <= 120:
        score += 5
    if any(marker in clean for marker in ("客户反馈", "现场反馈", "发生时间", "错误代码", "软件版本")):
        score += 2
    if any(marker in clean for marker in FAULT_STATUS_UPDATE_MARKERS):
        score -= 8
    return score


def _best_fault_focus(messages: list[dict[str, Any]], useful: list[dict[str, Any]]) -> str:
    candidates = []
    for msg in [*messages, *useful]:
        text = str(msg.get("text") or "")
        score = _fault_focus_score(text)
        if score > -100:
            candidates.append((score, _one_line(text, 1000)))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1]


def _fault_focus_confidence(text: str) -> float:
    score = _fault_focus_score(text)
    if score <= 0:
        return 0.0
    if score >= 20:
        return 1.0
    return round(score / 20.0, 4)


def _is_action_focus_text(text: str) -> bool:
    clean = _one_line(text, 260)
    if not clean:
        return False
    if any(marker in clean for marker in ("现场工作汇报", "工作汇总", "每日反馈", "夜班数据返回", "建议可以", "不如直接相信模型", "有关系吗")):
        return False
    if not _has_concrete_diagnostic_action(clean):
        return False
    if len(clean) > 180 and any(marker in clean for marker in ("客户反馈", "现场工作", "工作汇总", "项目")):
        return False
    return True



def _message_brief(msg: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_id": str(msg.get("message_id") or ""),
        "source_message_id": str(msg.get("source_message_id") or msg.get("message_id") or ""),
        "fragment_index": msg.get("fragment_index"),
        "fragment_count": msg.get("fragment_count"),
        "sender": msg.get("sender") or {},
        "create_time": str(msg.get("create_time") or ""),
        "msg_type": str(msg.get("msg_type") or ""),
        "text": str(msg.get("text") or ""),
        "content_summary": _one_line(msg.get("text"), 240),
        "attachment_metadata": list(msg.get("attachments") or []),
        "links": list(msg.get("links") or []),
    }


def _message_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return stable message ids while preserving source order."""
    return _unique(
        str(row.get("message_id") or row.get("source_message_id") or "")
        for row in rows
        if isinstance(row, dict)
    )


def _multi_issue_fragments(msg: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(msg.get("text") or "")
    daily_report = _message_is_daily_report(text)
    if len(text) < 80 and not daily_report:
        return [msg]
    report_positions = [text.find(marker) for marker in MULTI_ISSUE_REPORT_MARKERS if marker in text]
    report_start = min((value for value in report_positions if value >= 0), default=-1)
    if daily_report and report_start < 0:
        report_start = 0
    report_like = report_start >= 0
    if not daily_report and any(marker in text for marker in MULTI_ISSUE_PROCEDURE_MARKERS):
        return [msg]
    scope = text[report_start:] if report_like else text
    for marker in REPORT_END_MARKERS:
        if marker in scope:
            scope = scope.split(marker, 1)[0]
    if daily_report:
        anchor = _field_report_anchor(msg)
        issue_sections = [str(item.get("text") or "") for item in anchor.get("issue_items") or []]
    else:
        sections = _report_issue_sections(scope)
        issue_sections = [section for section in sections if any(keyword in section for keyword in MULTI_ISSUE_KEYWORDS)]
    if len(issue_sections) < 2 and not (daily_report and len(issue_sections) == 1):
        return [msg]
    if not report_like:
        strong_fault_sections = [
            section for section in issue_sections
            if any(keyword in section for keyword in ("报错", "异常", "失败", "蓝屏", "黑屏", "重启", "卡死", "闪退", "图片为空", "空图", "不拍照", "拍摄失败", "无法", "不能"))
        ]
        if len(strong_fault_sections) < 2:
            return [msg]

    prefix = text[:report_start] if report_like else text[: max(0, text.find(issue_sections[0]))]
    context = " ".join(_version_values(prefix))
    out: list[dict[str, Any]] = []
    source_id = str(msg.get("source_message_id") or msg.get("message_id") or "")
    shared_links = [dict(item) for item in msg.get("links") or [] if isinstance(item, dict)]
    shared_attachments = [dict(item) for item in msg.get("attachments") or [] if isinstance(item, dict)]
    for idx, section in enumerate(issue_sections, 1):
        fragment = dict(msg)
        fragment["text"] = f"{context} {section}".strip() if context else section
        fragment["source_message_id"] = source_id
        fragment["fragment_index"] = idx
        fragment["fragment_count"] = len(issue_sections)
        if daily_report:
            fragment["field_report_item_kind"] = "fault_case"
            fragment["field_report_item_index"] = idx
        fragment["links"] = []
        fragment["attachments"] = []
        fragment["shared_source_evidence_unassigned"] = {
            "source_message_id": source_id,
            "links": shared_links,
            "attachments": shared_attachments,
            "reason": "multi_issue_message_requires_case_level_evidence_assignment",
        }
        fragment["attachments_shared_across_fragments"] = bool(shared_attachments)
        fragment.setdefault("raw", dict(msg.get("raw") or {}))
        fragment["raw"] = {**dict(fragment.get("raw") or {}), "source_message_id": source_id, "fragment_index": idx, "fragment_count": len(issue_sections)}
        out.append(fragment)
    return out or [msg]


def _report_issue_sections(scope: str) -> list[str]:
    """Return leaf fault items while preserving report heading hierarchy."""

    top = list(REPORT_HEADING_RE.finditer(scope))
    top_chunks: list[str] = []
    if top:
        prefix = scope[:top[0].start()].strip(" \n。；;")
        if prefix:
            top_chunks.append(prefix)
        for index, match in enumerate(top):
            end = top[index + 1].start() if index + 1 < len(top) else len(scope)
            chunk = scope[match.end():end].strip(" \n。；;")
            if chunk:
                top_chunks.append(chunk)
    else:
        top_chunks = [scope]

    sections: list[str] = []
    for chunk in top_chunks:
        numbered = list(NUMBERED_ITEM_RE.finditer(chunk))
        if not numbered:
            sections.append(chunk)
            continue
        for index, match in enumerate(numbered):
            end = numbered[index + 1].start() if index + 1 < len(numbered) else len(chunk)
            section = chunk[match.start():end].strip(" \n。；;")
            if section:
                sections.append(section)
    return sections


SUMMARY_RELATION_MARKERS = (
    "汇总", "每日反馈", "今日反馈", "今日现状", "现场情况反馈", "问题汇总", "异常汇总",
)


def _is_summary_message(message: dict[str, Any]) -> bool:
    text = str(message.get("text") or "")
    return any(marker in text for marker in SUMMARY_RELATION_MARKERS) and (
        bool(re.search(r"(?:一|二|三|四|五|1|2|3|4|5)[、.．:：]", text))
        or any(marker in text for marker in ("问题", "异常", "漏检", "误报", "报错", "失败", "JIRA", "jira"))
    )


def _summary_relation_terms(text: str) -> set[str]:
    value = str(text or "").lower()
    terms = set(JIRA_RE.findall(value.upper()))
    terms.update(VERSION_RE.findall(value))
    for marker in (
        "ocr", "led", "连锡", "漏检", "漏报", "误报", "提示框", "卡顿", "拍摄失败", "拍照失败",
        "双轨", "异步", "同步", "导出", "乱码", "相机", "网卡", "排线", "复判", "模型", "光源",
    ):
        if marker.lower() in value:
            terms.add(marker.lower())
    return terms


def _attach_summary_relations(summaries: list[dict[str, Any]], messages: list[dict[str, Any]]) -> None:
    """Link report summaries to earlier sessions without rewriting native threads."""
    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        by_thread[str(message.get("thread_id") or "default")].append(message)
    thread_meta: dict[str, dict[str, Any]] = {}
    for thread_id, rows in by_thread.items():
        rows.sort(key=lambda item: str(item.get("create_time") or ""))
        thread_meta[thread_id] = {
            "messages": rows,
            "start": _text_history_timestamp(rows[0].get("create_time")) if rows else None,
            "end": _text_history_timestamp(rows[-1].get("create_time")) if rows else None,
            "terms": _summary_relation_terms(" ".join(str(item.get("text") or "") for item in rows)),
        }
    summary_by_thread = {str(summary.get("thread_id") or ""): summary for summary in summaries}
    for summary in summaries:
        thread_id = str(summary.get("thread_id") or "")
        current_rows = thread_meta.get(thread_id, {}).get("messages") or []
        report_messages = [item for item in current_rows if _is_summary_message(item)]
        if not report_messages:
            continue
        relations: list[dict[str, Any]] = []
        for report in report_messages:
            report_time = _text_history_timestamp(report.get("create_time"))
            if report_time is None:
                continue
            report_terms = _summary_relation_terms(str(report.get("text") or ""))
            candidates: list[tuple[float, dict[str, Any], list[str], str]] = []
            for candidate_thread, meta in thread_meta.items():
                if candidate_thread == thread_id:
                    continue
                end_time = meta.get("end")
                if end_time is None or end_time > report_time:
                    continue
                gap_hours = (report_time - end_time).total_seconds() / 3600.0
                if gap_hours > 24 * 14:
                    continue
                shared = report_terms & set(meta.get("terms") or set())
                if not shared:
                    continue
                score = 0.0
                reasons = ["same_chat"]
                if report_time.date() == end_time.date():
                    score += 3.0
                    reasons.append("same_day")
                elif gap_hours <= 24:
                    score += 2.0
                    reasons.append("gap_le_24h")
                else:
                    reasons.append("historical_window")
                if gap_hours <= 6:
                    score += 1.5
                    reasons.append("gap_le_6h")
                elif gap_hours <= 24:
                    score += 0.5
                jira_shared = {item for item in shared if JIRA_RE.fullmatch(item.upper())}
                if jira_shared:
                    score += 4.0
                    reasons.append("shared_jira")
                version_shared = {item for item in shared if VERSION_RE.fullmatch(item)}
                if version_shared:
                    score += 1.0
                    reasons.append("shared_version")
                score += min(4.0, len(shared) * 0.75)
                reasons.append("shared_fault_terms")
                relation = "summary_of" if gap_hours <= 24 else "historical_related"
                threshold = 4.0 if relation == "summary_of" else 5.5
                if relation == "summary_of" and not jira_shared and len(shared) < 2:
                    continue
                if relation == "historical_related" and not jira_shared and len(shared) < 2:
                    continue
                if score >= threshold:
                    candidates.append((score, {"thread_id": candidate_thread, **meta}, reasons, relation))
            candidates.sort(key=lambda item: (-item[0], -((item[1].get("end") or datetime.min).timestamp())))
            for score, candidate, reasons, relation in candidates[:3]:
                target_thread = str(candidate.get("thread_id") or "")
                target_messages = candidate.get("messages") or []
                relations.append({
                    "relation_id": f"summary:{report.get('message_id')}:{target_thread}",
                    "relation": relation,
                    "source_message_id": str(report.get("message_id") or ""),
                    "source_thread_id": thread_id,
                    "target_thread_id": target_thread,
                    "target_message_ids": [str(item.get("message_id") or "") for item in target_messages],
                    "shared_terms": sorted(report_terms & set(candidate.get("terms") or set())),
                    "score": round(score, 4),
                    "confidence": round(min(1.0, score / 10.0), 4),
                    "reason_codes": reasons + [relation],
                    "inferred": True,
                    "platform_reference": False,
                })
        unique = {str(item.get("relation_id")): item for item in relations}
        relations = list(unique.values())
        if not relations:
            continue
        extracted = summary.get("extracted") if isinstance(summary.get("extracted"), dict) else {}
        extracted["summary_relations"] = relations
        extracted["summary_relation_policy"] = "same_day_summary_of_plus_14d_historical_related.v1"
        summary["extracted"] = extracted
        summary["summary_relations"] = relations
        summary["summary_context_messages"] = [
            _message_brief(item)
            for relation in relations
            if relation.get("relation") == "summary_of"
            for item in (thread_meta.get(str(relation.get("target_thread_id") or ""), {}).get("messages") or [])
        ]
        summary["summary_context_message_ids"] = [
            str(item.get("message_id") or "") for item in summary.get("summary_context_messages") or []
        ]
        for episode in summary.get("episodes") or []:
            episode_message_ids = {
                str(message.get("message_id") or message.get("source_message_id") or "")
                for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "noise_messages", "case_context_messages")
                for message in episode.get(key) or []
                if isinstance(message, dict)
            }
            episode_text = " ".join(
                str(message.get("text") or message.get("content_summary") or "")
                for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_context_messages")
                for message in episode.get(key) or []
                if isinstance(message, dict)
            )
            episode_terms = _summary_relation_terms(episode_text)
            episode_relations = []
            for relation in relations:
                if str(relation.get("source_message_id") or "") not in episode_message_ids:
                    continue
                relation_terms = {str(item).lower() for item in relation.get("shared_terms") or []}
                matched_terms = episode_terms & relation_terms
                # The source report message may have been split into several
                # case items that all share one source_message_id.  Attach the
                # relation only to the fragment that actually carries the
                # matching fault vocabulary; source identity alone is unsafe.
                if not matched_terms:
                    continue
                episode_relations.append({**relation, "episode_shared_terms": sorted(matched_terms)})
            if not episode_relations:
                continue
            episode_context_messages = [
                _message_brief(item)
                for relation in episode_relations
                if relation.get("relation") == "summary_of"
                for item in (thread_meta.get(str(relation.get("target_thread_id") or ""), {}).get("messages") or [])
                if _summary_relation_terms(str(item.get("text") or "")) & set(relation.get("episode_shared_terms") or [])
                if str(item.get("msg_type") or "") != "system"
            ]
            episode_extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
            episode_extracted["summary_relations"] = episode_relations
            episode_extracted["summary_evidence_message_ids"] = [
                str(item.get("message_id") or "") for item in episode_context_messages
            ]
            episode_extracted["summary_relation_policy"] = extracted["summary_relation_policy"]
            episode["extracted"] = episode_extracted
            episode["summary_relations"] = episode_relations
            episode["summary_context_messages"] = episode_context_messages
            episode["summary_context_message_ids"] = [
                str(item.get("message_id") or "") for item in episode_context_messages
            ]
            message_ids = list(episode.get("message_ids") or [])
            context_message_ids = list(episode.get("context_message_ids") or [])
            episode["full_context_message_ids"] = _unique([
                *message_ids,
                *context_message_ids,
                *episode["summary_context_message_ids"],
            ], limit=500)
            episode["message_refs"] = {
                "message_ids": message_ids,
                "context_message_ids": context_message_ids,
                "summary_context_message_ids": episode["summary_context_message_ids"],
                "full_context_message_ids": episode["full_context_message_ids"],
            }


def _field_report_anchor(msg: dict[str, Any]) -> dict[str, Any]:
    text = str(msg.get("text") or "")
    if not _message_is_daily_report(text):
        return {}
    positions = [text.find(marker) for marker in DAILY_REPORT_MARKERS if marker in text]
    report_start = min((value for value in positions if value >= 0), default=0)
    scope = text[report_start:]
    for marker in REPORT_END_MARKERS:
        if marker in scope:
            scope = scope.split(marker, 1)[0]
    sections = [_one_line(section, 1000) for section in _report_issue_sections(scope)]
    classified = [
        {"text": section, "kind": _field_report_section_kind(section)}
        for section in sections
        if section
    ]
    issue_sections = [item["text"] for item in classified if item["kind"] == "fault_case"]
    status_sections = [item["text"] for item in classified if item["kind"] == "status_update"]
    work_sections = [item["text"] for item in classified if item["kind"] == "work_item"]
    sender = msg.get("sender") if isinstance(msg.get("sender"), dict) else {}
    message_id = str(msg.get("source_message_id") or msg.get("message_id") or "")
    raw = msg.get("raw") if isinstance(msg.get("raw"), dict) else {}
    report_date = str(msg.get("create_time") or "")[:10]
    return {
        "anchor_id": f"field-report:{message_id}",
        "message_id": message_id,
        "author": str(sender.get("name") or ""),
        "author_id": str(sender.get("id") or ""),
        "report_date": report_date,
        "site": _site_from_chat_name(str(raw.get("chat_name") or "")),
        "chat_id": str(msg.get("chat_id") or raw.get("source_chat_id") or ""),
        "issue_count": len(issue_sections),
        "status_update_count": len(status_sections),
        "work_item_count": len(work_sections),
        "item_count_total": len(classified),
        "issue_items": [
            {
                "item_index": index,
                "text": section,
                "evidence_message_ids": [message_id],
            }
            for index, section in enumerate(issue_sections, 1)
        ],
        "status_updates": [
            {"item_index": index, "text": section, "evidence_message_ids": [message_id]}
            for index, section in enumerate(status_sections, 1)
        ],
        "work_items": [
            {"item_index": index, "text": section, "evidence_message_ids": [message_id]}
            for index, section in enumerate(work_sections, 1)
        ],
        "confidence": round(min(1.0, 0.65 + 0.08 * len(issue_sections)), 4),
    }


def _field_report_section_kind(section: str) -> str:
    text = str(section or "")
    has_status = any(marker in text for marker in FIELD_REPORT_STATUS_MARKERS)
    has_work = any(marker in text for marker in FIELD_REPORT_WORK_MARKERS)
    active_text = text
    for marker in FIELD_REPORT_STATUS_MARKERS:
        active_text = active_text.replace(marker, "")
    has_active_fault = any(marker in active_text for marker in FIELD_REPORT_ACTIVE_FAULT_MARKERS)
    has_case_context = any(marker in text for marker in (
        "客户反馈", "现场反馈", "出现", "发生", "导致", "排查", "原因", "更换", "处理", "修复", "调整", "报警", "报错",
    ))
    if any(marker in text for marker in ("培训", "讲解")) and not any(marker in text for marker in (
        "客户反馈", "客户提到", "现场反馈", "现场测板", "出现", "发生", "无法", "失败", "卡死", "闪退", "蓝屏",
        "报错", "误报增加", "误报过高", "识别不准", "漏检测", "卡板的情况",
    )):
        return "work_item"
    if any(marker in text for marker in ("追溯", "整合文档")) and not any(marker in text for marker in ("报错", "无法", "失败", "卡死", "闪退", "蓝屏", "原因是", "排查为")):
        return "work_item"
    if has_status and any(marker in text for marker in ("跟踪", "观察", "截至目前", "到现在")) and not any(marker in text for marker in ("排查为", "原因是", "导致", "修复", "解决")):
        return "status_update"
    if has_status and not has_active_fault:
        return "status_update"
    if has_active_fault and (has_case_context or not has_work):
        return "fault_case"
    if has_status:
        return "status_update"
    return "work_item"


def _build_observed_people(messages: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = people_index(load_people_role_registry())
    grouped: dict[str, dict[str, Any]] = {}

    def add(name: str, *, episode_role: str, message_id: str, signal: str, create_time: str, organization_candidate: str = "") -> None:
        clean_name = str(name or "").strip()
        if not clean_name:
            return
        explicit = registry.get(clean_name) or {}
        target = grouped.setdefault(clean_name, {
            "name": clean_name,
            "organization_roles": list(explicit.get("organization_roles") or []),
            "organization_role_candidates": [],
            "responsibility_scopes": list(explicit.get("responsibility_scopes") or []),
            "episode_role_counts": {},
            "evidence": [],
            "distinct_report_dates": [],
            "status": "confirmed" if explicit else "observed",
        })
        counts = target["episode_role_counts"]
        counts[episode_role] = int(counts.get(episode_role) or 0) + 1
        evidence = {
            "message_id": message_id,
            "signal": signal,
            "episode_role": episode_role,
        }
        if evidence not in target["evidence"]:
            target["evidence"].append(evidence)
        if episode_role == "field_report_author" and create_time:
            report_date = str(create_time)[:10]
            if report_date and report_date not in target["distinct_report_dates"]:
                target["distinct_report_dates"].append(report_date)
        if organization_candidate and organization_candidate not in target["organization_roles"]:
            candidates = target["organization_role_candidates"]
            if organization_candidate not in candidates:
                candidates.append(organization_candidate)

    anchor_by_message = {str(item.get("message_id") or ""): item for item in anchors}
    for msg in messages:
        sender = msg.get("sender") if isinstance(msg.get("sender"), dict) else {}
        name = str(sender.get("name") or "")
        message_id = str(msg.get("message_id") or "")
        create_time = str(msg.get("create_time") or "")
        text = str(msg.get("text") or "")
        if message_id in anchor_by_message:
            add(name, episode_role="field_report_author", message_id=message_id, signal="field_report_anchor", create_time=create_time, organization_candidate="fae")
        reporter, reasons = _message_is_reporter_signal(msg)
        if reporter:
            add(name, episode_role="reporter", message_id=message_id, signal="+".join(reasons), create_time=create_time)
        if _message_has_jira_submission_signal(msg) or _message_has_provided_info(msg):
            add(name, episode_role="evidence_provider", message_id=message_id, signal="jira_or_evidence_provided", create_time=create_time)
        if _is_diagnostic_action(msg):
            add(name, episode_role="investigator", message_id=message_id, signal="diagnostic_action", create_time=create_time)
        if _is_resolution(msg):
            add(name, episode_role="resolver", message_id=message_id, signal="resolution_statement", create_time=create_time)
            if any(marker in text for marker in ("复测", "验证", "现场", "未再出现", "恢复正常")):
                add(name, episode_role="validator", message_id=message_id, signal="field_validation", create_time=create_time)

    for item in grouped.values():
        roles = item.get("episode_role_counts") or {}
        fae_evidence = int(roles.get("field_report_author") or 0) + int(roles.get("evidence_provider") or 0) + int(roles.get("validator") or 0)
        item["confidence"] = round(min(0.95, 0.25 + 0.12 * fae_evidence + (0.2 if item["status"] == "confirmed" else 0.0)), 4)
        item["evidence"] = item["evidence"][:50]
        item["distinct_report_dates"].sort()
    return sorted(grouped.values(), key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("name") or "")))


def _expand_multi_issue_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for msg in messages:
        expanded.extend(_multi_issue_fragments(msg))
    return expanded


def _is_next_fragment_boundary(current: list[dict[str, Any]], msg: dict[str, Any]) -> bool:
    source_id = str(msg.get("source_message_id") or "")
    fragment_index = msg.get("fragment_index")
    if not source_id or not fragment_index:
        return False
    return any(str(item.get("source_message_id") or "") == source_id and item.get("fragment_index") != fragment_index for item in current)


def _fault_topic_key(msg: dict[str, Any]) -> str:
    text = str(msg.get("text") or "")
    rules = (
        ("ipc_system", ("蓝屏", "自动重启", "异常重启", "死机重启", "Bugcheck", "0x00000139")),
        ("camera_capture", ("拍摄失败", "拍照失败", "不拍照", "空图", "相机初始化失败")),
        ("transport_outfeed", ("无法出板", "出板失败", "卡板", "轨道宽度", "板卡卡滞")),
        ("light_init", ("光源初始化失败", "光控初始化失败", "光源异常")),
        ("software_crash", ("软件闪退", "直接闪退", "卡死", "无响应")),
        ("performance", ("操作响应延迟", "智能调整", "编程优化", "运行卡顿", "响应慢")),
        ("tuning", ("误报", "漏检", "识别不准", "算法异常")),
        ("config", ("加载用户配置失败", "配置文件", "user.cfg")),
        ("program_export", ("导出中文程序", "导出乱码", "显示乱码", "导入提示失败", "导出错误", "无法导出")),
        ("dual_track_test", ("双轨", "异步测试不出检测结果", "同步测试不出检测结果", "正在暂停", "无法开启测试")),
        ("network", ("网络异常", "连接失败", "请求超时", "IP冲突")),
    )
    for topic, markers in rules:
        if any(marker in text for marker in markers):
            return topic
    return ""


def _is_distinct_fault_topic_boundary(current: list[dict[str, Any]], msg: dict[str, Any]) -> bool:
    incoming = _fault_topic_key(msg)
    if not incoming:
        return False
    existing = {
        topic
        for item in current
        if _is_fault_description(item)
        for topic in [_fault_topic_key(item)]
        if topic
    }
    return bool(existing and incoming not in existing)


def _is_stale_context_boundary(current: list[dict[str, Any]], msg: dict[str, Any]) -> bool:
    """Detach a late, unclassified utterance from an already complete case.

    Long-running chats often resume days later with a short question such as
    ``又压排线了吗`` before the next concrete fault description.  Keeping that
    utterance in the old episode leaks the previous fault into the new case.
    Explicit result messages remain attached to the old case even across a
    long gap.
    """
    if not current or _is_resolution(msg):
        return False
    previous_time = _text_history_timestamp(current[-1].get("create_time"))
    incoming_time = _text_history_timestamp(msg.get("create_time"))
    if previous_time is None or incoming_time is None:
        return False
    gap_hours = (incoming_time - previous_time).total_seconds() / 3600.0
    if gap_hours <= 72:
        return False
    return any(_is_fault_description(item) or _is_diagnostic_action(item) or _is_resolution(item) for item in current)


def _episode_local_messages(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the evidence-local part of one already split episode.

    ``split_fault_episodes`` has already established the episode boundary.  A
    second, global ``messages +/- N`` context window is therefore unsafe: in a
    long field-report chat it re-attaches unrelated numbered items and later
    cases to the current episode.  Keep at most a small lead-in immediately
    before the first fault anchor for local context, then retain the whole
    episode chunk.  The complete original segment remains available in the
    parent thread summary for audit/navigation.
    """
    if not chunk:
        return []
    fault_positions = [
        index
        for index, item in enumerate(chunk)
        if _is_fault_description(item) or item.get("field_report_item_kind") == "fault_case"
    ]
    if not fault_positions:
        return list(chunk)
    # Fault descriptions are the semantic anchor.  Earlier messages remain in
    # the parent thread summary as navigation context, but are not current-case
    # evidence unless a later W7/Jira/attachment linker explicitly assigns
    # them.  This conservative boundary avoids silently attaching procurement,
    # delivery, training, or a previous fault to the new episode.
    start = fault_positions[0]
    return list(chunk[start:])


def _missing_info_requests(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        text = str(msg.get("text") or "")
        if not _message_has_missing_info_request(text):
            continue
        later = [m for m in messages[idx + 1 :] if _message_has_provided_info(m)]
        provided_ids = _unique((m.get("message_id") for m in later), limit=10)
        provided_evidence = [_provided_evidence_brief(m) for m in later[:10]]
        request_id = str(msg.get("message_id") or "")
        requests.append({
            "message_id": request_id,
            "sender": msg.get("sender") or {},
            "timestamp": str(msg.get("create_time") or ""),
            "create_time": str(msg.get("create_time") or ""),
            "text": text,
            "thread_id": str(msg.get("thread_id") or ""),
            "context_before": [_message_brief(m) for m in messages[max(0, idx - 2) : idx]],
            "context_after": [_message_brief(m) for m in messages[idx + 1 : idx + 3]],
            "evidence_message_ids": _unique([request_id, *provided_ids], limit=12),
            "provided_later": bool(provided_ids),
            "provided_evidence_message_ids": provided_ids,
            "provided_evidence": provided_evidence,
        })
    return requests


def _dedupe_offsets(offsets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in offsets:
        key = (str(item.get("field") or ""), str(item.get("value") or ""), str(item.get("message_id") or ""), str(item.get("attachment_key") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _version_values(text: str) -> list[str]:
    ips = set(IP_RE.findall(str(text or "")))
    return _unique((value for value in VERSION_RE.findall(str(text or "")) if value not in ips), limit=80)


def _find_version_offsets(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    for msg in messages:
        content = str(msg.get("text") or "")
        ip_spans = [match.span() for match in IP_RE.finditer(content)]
        for match in VERSION_RE.finditer(content):
            start, end = match.span()
            if any(ip_start <= start and end <= ip_end for ip_start, ip_end in ip_spans):
                continue
            offsets.append({
                "field": "versions",
                "value": match.group(0),
                "source": "message.text",
                "message_id": msg.get("message_id", ""),
                "thread_id": msg.get("thread_id", ""),
                "start": start,
                "end": end,
            })
    return offsets


def _unique_dicts(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        value = str(item.get(key) or "")
        if value and value in seen:
            continue
        if value:
            seen.add(value)
        out.append(item)
    return out


def _safe_parse_attachment(att: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_attachment_evidence(att)
    except Exception as exc:  # noqa: BLE001 - W1 must keep evidence metadata even if a tool fails.
        return {
            "type": "AttachmentParseResult",
            "status": "parse_failed",
            "error": str(exc),
            "source": dict(att),
            "observability": {"agent_id": "TOOL-ATTACHMENT", "status": "parse_failed"},
        }


def _safe_parse_jira(link: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_jira_evidence(link)
    except Exception as exc:  # noqa: BLE001 - W1 must not fail on evidence parsing.
        return {
            "type": "JiraParseResult",
            "status": "parse_failed",
            "error": str(exc),
            "source": dict(link),
            "observability": {"agent_id": "TOOL-JIRA", "status": "parse_failed"},
        }


def _safe_parse_image(att: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_image_evidence(att)
    except Exception as exc:  # noqa: BLE001 - W1 degrades image evidence to metadata-only.
        return {
            "type": "ImageParseResult",
            "status": "parse_failed",
            "error": str(exc),
            "pixels_read": False,
            "ocr_performed": False,
            "archive_extracted": False,
            "source": dict(att),
            "observability": {"agent_id": "TOOL-IMAGE", "status": "parse_failed"},
        }


def _safe_parse_document(att: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_document_evidence(att)
    except Exception as exc:  # noqa: BLE001 - W1 degrades document evidence to metadata-only.
        return {
            "type": "DocumentParseResult",
            "status": "parse_failed",
            "error": str(exc),
            "archive_extracted": False,
            "macros_executed": False,
            "formulas_evaluated": False,
            "ocr_performed": False,
            "source": dict(att),
            "observability": {"agent_id": "TOOL-DOCUMENT", "status": "parse_failed"},
        }


def _safe_parse_dmp(att: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_dmp_evidence(att)
    except Exception as exc:  # noqa: BLE001 - W1 degrades DMP evidence to metadata-only.
        return {
            "type": "DmpParseResult",
            "status": "parse_failed",
            "error": str(exc),
            "full_content_read": False,
            "debugger_executed": False,
            "source": dict(att),
            "observability": {"agent_id": "TOOL-DMP", "status": "parse_failed"},
        }


def _safe_parse_proj(path: str, source: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        result = parse_proj_evidence(path)
        if source:
            result = dict(result)
            result["source"] = dict(source)
        return result
    except Exception as exc:  # noqa: BLE001 - W1 degrades to metadata-only evidence.
        return {
            "type": "ProjParseResult",
            "status": "parse_failed",
            "path": path,
            "error": str(exc),
            "executed": False,
            "mutated": False,
            "source": dict(source or {}),
            "observability": {"agent_id": "TOOL-PROJ", "status": "parse_failed"},
        }


def _safe_parse_log_package(att: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_log_package_evidence(att)
    except Exception as exc:  # noqa: BLE001 - W1 degrades to metadata-only evidence.
        return {
            "type": "LogPackageParseResult",
            "status": "parse_failed",
            "error": str(exc),
            "archive_extracted": False,
            "content_read": False,
            "source": dict(att),
            "observability": {"agent_id": "TOOL-LOG-PACKAGE", "status": "parse_failed"},
        }


def _jira_tool_inputs(links: list[dict[str, Any]], jira_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Build offline Jira parser inputs from both URLs and bare issue keys."""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for link in links:
        if not isinstance(link, dict) or link.get("type") != "jira":
            continue
        link_text = " ".join(str(link.get(k) or "") for k in ("url", "issue_key", "label", "text"))
        link_keys = JIRA_RE.findall(link_text)
        signature = f"issue:{link_keys[0]}" if link_keys else str(link.get("url") or link.get("issue_key") or link.get("label") or link.get("text") or "")
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        for key in link_keys:
            seen.add(f"issue:{key}")
        out.append(link)
    for issue_key in jira_ids or []:
        key = str(issue_key or "").strip()
        if not key:
            continue
        signature = f"issue:{key}"
        if signature in seen:
            continue
        seen.add(signature)
        out.append({
            "type": "jira",
            "text": key,
            "issue_key": key,
            "source": "message.text.jira_key",
            "reason": "offline_issue_key_metadata",
        })
    return out


def _build_tool_evidence(attachments: list[dict[str, Any]], links: list[dict[str, Any]], jira_ids: list[str] | None = None) -> dict[str, Any]:
    attachment_inputs = _unique_dicts([x for x in attachments if isinstance(x, dict)], "file_key")
    jira_inputs = _jira_tool_inputs(links, jira_ids)
    attachment_results = [_safe_parse_attachment(att) for att in attachment_inputs]
    jira_results = [_safe_parse_jira(link) for link in jira_inputs]
    document_results: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []
    dmp_results: list[dict[str, Any]] = []
    proj_results: list[dict[str, Any]] = []
    log_package_results: list[dict[str, Any]] = []
    for result in attachment_results:
        source = result.get("source") if isinstance(result.get("source"), dict) else {}
        if result.get("evidence_role") == "sample_image":
            image_results.append(_safe_parse_image(source or result))
        elif result.get("evidence_role") == "data_file" and Path(str(result.get("name") or result.get("path") or "")).suffix.lower() in DOCUMENT_FILE_EXTS:
            document_results.append(_safe_parse_document(source or result))
        elif result.get("evidence_role") == "program_file":
            path = str(result.get("path") or "")
            if path:
                proj_results.append(_safe_parse_proj(path, source or result))
        elif result.get("evidence_role") == "log_package":
            if Path(str(result.get("name") or result.get("path") or "")).suffix.lower() in {".dmp", ".mdmp"}:
                dmp_results.append(_safe_parse_dmp(source or result))
            log_package_results.append(_safe_parse_log_package(source or result))
    return {
        "attachment_parse_results": attachment_results,
        "document_parse_results": document_results,
        "image_parse_results": image_results,
        "dmp_parse_results": dmp_results,
        "jira_parse_results": jira_results,
        "proj_parse_results": proj_results,
        "log_package_parse_results": log_package_results,
        "observability": {
            "agent_id": "W1",
            "tool_evidence": {
                "attachments": len(attachment_results),
                "documents": len(document_results),
                "images": len(image_results),
                "dmp_files": len(dmp_results),
                "jira_links": len(jira_results),
                "proj_files": len(proj_results),
                "log_packages": len(log_package_results),
            },
        },
    }


class ChatCollectAgent:
    """W1: normalize chat messages into thread summaries plus fault episodes."""

    def collect(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.aggregate_threads(self.normalize_messages(messages))

    def normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(messages):
            message_id = str(row.get("message_id") or row.get("id") or f"msg-{idx}")
            thread_id = str(row.get("thread_id") or row.get("segment_id") or row.get("chat_id") or "default")
            sender = row.get("sender")
            if isinstance(sender, dict):
                sender_obj = {"id": str(sender.get("id") or ""), "name": str(sender.get("name") or ""), "type": str(sender.get("type") or "user")}
            else:
                sender_obj = {"id": "", "name": str(sender or row.get("sender_name") or ""), "type": "user"}
            raw = dict(row.get("raw") or row)
            raw_content = str(raw.get("raw_content") or raw.get("content") or row.get("content") or "")
            raw_text = str(row.get("text") or row.get("plain_text") or row.get("content") or "")
            clean_text = _clean_text(raw_text)
            attachments = [item for item in (row.get("attachments") or []) if isinstance(item, dict)]
            embedded_attachments = _extract_embedded_file_metadata(raw_content, message_id=message_id, thread_id=thread_id)
            attachment_keys = {str(item.get("file_key") or item.get("name") or "") for item in attachments}
            attachments.extend(item for item in embedded_attachments if str(item.get("file_key") or item.get("name") or "") not in attachment_keys)
            links = []
            for item in row.get("links") or []:
                if not isinstance(item, dict):
                    continue
                link = dict(item)
                link["url"] = _normalize_url(link.get("url"))
                links.append(link)
            existing_urls = {str(item.get("url") or "") for item in links if isinstance(item, dict)}
            for link in _extract_links(raw_content or raw_text, message_id=message_id, thread_id=thread_id):
                if str(link.get("url") or "") not in existing_urls:
                    links.append(link)
                    existing_urls.add(str(link.get("url") or ""))
            raw.setdefault("raw_content", raw_content or raw_text)
            normalized.append({
                "message_id": message_id,
                "thread_id": thread_id,
                "chat_id": str(row.get("chat_id") or _chat_id_from_segment(thread_id)),
                "sender": sender_obj,
                "create_time": str(row.get("create_time") or row.get("timestamp") or ""),
                "msg_type": str(row.get("msg_type") or "text"),
                "text": clean_text,
                "mentions": list(row.get("mentions") or _unique(MENTION_RE.findall(clean_text or raw_text))),
                "attachments": list(attachments) if isinstance(attachments, list) else [],
                "links": links,
                "root_id": str(row.get("root_id") or ""),
                "parent_id": str(row.get("parent_id") or ""),
                "upper_message_id": str(row.get("upper_message_id") or ""),
                "relation_thread_id": str(row.get("relation_thread_id") or ""),
                "relation_source": str(row.get("relation_source") or ""),
                "raw": raw,
            })
        return normalized

    def import_xing_upload(
        self,
        import_root: str | Path,
        *,
        limit: int = 0,
        hits_only: bool = False,
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        import_root = Path(import_root).resolve()
        messages = self.load_xing_messages(import_root, limit=limit, hits_only=hits_only)
        summaries = self.aggregate_threads(messages)
        episodes = [episode for summary in summaries for episode in summary.get("episodes", [])]
        field_report_anchors = [anchor for summary in summaries for anchor in summary.get("field_report_anchors", [])]
        observed_people = _build_observed_people(messages, field_report_anchors)
        manifest = self.build_manifest(import_root, messages, summaries, episodes, limit=limit, hits_only=hits_only)
        manifest["counts"]["field_report_anchors"] = len(field_report_anchors)
        manifest["counts"]["observed_people"] = len(observed_people)
        result = {
            "messages": messages,
            "thread_summaries": summaries,
            "episodes": episodes,
            "field_report_anchors": field_report_anchors,
            "observed_people": observed_people,
            "run_manifest": manifest,
        }
        if out_dir is not None:
            self.write_run(out_dir, result)
        return result

    def import_text_history(
        self,
        import_root: str | Path,
        *,
        limit: int = 0,
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        import_root = Path(import_root).resolve()
        source_rows, source_path = self.load_text_history_rows(import_root, limit=limit)
        prepared_rows = [self._prepare_text_history_row(row) for row in source_rows]
        messages = self._segment_text_history_messages(self.normalize_messages(prepared_rows))
        summaries = self.aggregate_threads(messages)
        episodes = [episode for summary in summaries for episode in summary.get("episodes", [])]
        field_report_anchors = [anchor for summary in summaries for anchor in summary.get("field_report_anchors", [])]
        observed_people = _build_observed_people(messages, field_report_anchors)
        manifest = self.build_text_history_manifest(import_root, source_path, source_rows, messages, summaries, episodes, limit=limit)
        manifest["counts"]["field_report_anchors"] = len(field_report_anchors)
        manifest["counts"]["observed_people"] = len(observed_people)
        result = {
            "messages": messages,
            "thread_summaries": summaries,
            "episodes": episodes,
            "field_report_anchors": field_report_anchors,
            "observed_people": observed_people,
            "run_manifest": manifest,
        }
        if out_dir is not None:
            self.write_run(out_dir, result)
        return result

    def import_xing_with_relations(
        self,
        xing_import_root: str | Path,
        relation_import_root: str | Path,
        *,
        limit: int = 0,
        hits_only: bool = False,
        quiet_gap_hours: float = 12.0,
        max_messages: int = 120,
        context_attach_minutes: float = 60.0,
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Merge old Xing resources with v3 message relations, then re-segment.

        The old archive remains the authority for downloaded attachments.  The
        v3 corpus supplies text coverage and root/parent relation metadata.
        """
        from debug_agent_system.agents.write.w1_message_relations import (
            annotate_semantic_fragments,
            assign_reference_aware_segments,
            build_message_reference_graph,
            infer_cross_window_trace_edges,
            infer_context_continuation_edges,
            merge_xing_relation_history,
        )

        xing_import_root = Path(xing_import_root).resolve()
        relation_import_root = Path(relation_import_root).resolve()
        old_messages = self.load_xing_messages(xing_import_root, limit=limit, hits_only=hits_only)
        relation_rows, relation_source = self.load_text_history_rows(relation_import_root, limit=0)
        merged_rows, merge_report = merge_xing_relation_history(old_messages, relation_rows)
        normalized = self.normalize_messages(merged_rows)
        normalized, semantic_fragment_report = annotate_semantic_fragments(normalized)
        context_edges = [
            *infer_context_continuation_edges(normalized),
            *infer_cross_window_trace_edges(normalized),
        ]
        messages, segmentation_report = assign_reference_aware_segments(
            normalized,
            quiet_gap_hours=quiet_gap_hours,
            max_messages=max_messages,
            context_attach_minutes=context_attach_minutes,
            context_edges=context_edges,
        )
        reference_graph = build_message_reference_graph(messages, context_edges=context_edges)
        summaries = self.aggregate_threads(messages)
        episodes = [episode for summary in summaries for episode in summary.get("episodes", [])]
        field_report_anchors = [anchor for summary in summaries for anchor in summary.get("field_report_anchors", [])]
        observed_people = _build_observed_people(messages, field_report_anchors)
        manifest = {
            "run_id": "xing_relation_merged" if not limit else f"xing_relation_merged_sample_{limit}",
            "source": "xing_upload_plus_text_v3_relations",
            "xing_import_root": str(xing_import_root),
            "relation_import_root": str(relation_import_root),
            "relation_source": str(relation_source),
            "options": {
                "limit": limit,
                "hits_only": hits_only,
                "quiet_gap_hours": quiet_gap_hours,
                "max_messages": max_messages,
                "context_attach_minutes": context_attach_minutes,
            },
            "counts": {
                "messages": len(messages),
                "threads": len(summaries),
                "episodes": len(episodes),
                "attachments": sum(len(message.get("attachments") or []) for message in messages),
                "field_report_anchors": len(field_report_anchors),
                "observed_people": len(observed_people),
            },
            "merge_report": merge_report,
            "segmentation_report": segmentation_report,
            "semantic_fragment_report": semantic_fragment_report,
            "reference_graph_stats": reference_graph.get("stats") or {},
            "episode_completeness": dict(Counter(str(episode.get("completeness") or "") for episode in episodes)),
        }
        result = {
            "messages": messages,
            "thread_summaries": summaries,
            "episodes": episodes,
            "field_report_anchors": field_report_anchors,
            "observed_people": observed_people,
            "reference_graph": reference_graph,
            "run_manifest": manifest,
        }
        if out_dir is not None:
            self.write_run(out_dir, result)
            Path(out_dir, "message_reference_graph.json").write_text(
                json.dumps(reference_graph, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return result

    def load_text_history_rows(self, import_root: str | Path, *, limit: int = 0) -> tuple[list[dict[str, Any]], Path]:
        import_root = Path(import_root).resolve()
        if import_root.is_file():
            sources = [import_root]
            source_path = import_root
        elif (import_root / "all_text_messages.jsonl").exists():
            sources = [import_root / "all_text_messages.jsonl"]
            source_path = sources[0]
        elif (import_root / "messages_by_chat").exists():
            sources = sorted((import_root / "messages_by_chat").glob("*.jsonl"))
            source_path = import_root / "messages_by_chat"
        else:
            raise FileNotFoundError(f"text history source not found: {import_root}")
        rows: list[dict[str, Any]] = []
        for source in sources:
            with source.open(encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        rows.append(payload)
                    if limit and len(rows) >= limit:
                        return rows, source_path
        return rows, source_path

    def _prepare_text_history_row(self, row: dict[str, Any]) -> dict[str, Any]:
        raw = dict(row)
        raw.setdefault("source_chat_id", str(row.get("chat_id") or ""))
        raw.setdefault("source_chat_name", str(row.get("chat_name") or ""))
        raw.setdefault("source_import", TEXT_HISTORY_SOURCE)
        return {
            **row,
            "text": str(row.get("plain_text") or row.get("content") or ""),
            "attachments": [],
            "relation_thread_id": str(row.get("thread_id") or ""),
            "raw": raw,
        }

    def load_xing_messages(self, import_root: str | Path, *, limit: int = 0, hits_only: bool = False) -> list[dict[str, Any]]:
        import_root = Path(import_root).resolve()
        manifest_dir = import_root / "_MANIFEST"
        if not manifest_dir.exists():
            raise FileNotFoundError(f"manifest directory not found: {manifest_dir}")
        resource_results = self.load_xing_resources(import_root)
        rows = _read_csv(manifest_dir / "xing_messages.csv")
        hit_segments = {row.get("segment_id", "") for row in rows if _bool_text(row.get("is_hit"))} if hits_only else set()
        messages: list[dict[str, Any]] = []
        for row in rows:
            segment_id = row.get("segment_id", "")
            message_id = row.get("message_id", "")
            attachments = resource_results.get(message_id, [])
            if hits_only:
                keep = (
                    _bool_text(row.get("is_hit"))
                    or _bool_text(row.get("is_xing_related"))
                    or (segment_id in hit_segments and bool(attachments))
                )
                if not keep:
                    continue
            raw_content = row.get("content", "")
            clean_text = _clean_text(raw_content)
            links = _extract_links(raw_content, message_id=message_id, thread_id=segment_id)
            messages.append({
                "message_id": message_id,
                "thread_id": segment_id,
                "chat_id": _chat_id_from_segment(segment_id),
                "sender": {"id": "", "name": row.get("sender", ""), "type": "user"},
                "create_time": row.get("create_time", ""),
                "msg_type": row.get("msg_type", "text") or "text",
                "text": clean_text,
                "mentions": _unique(MENTION_RE.findall(clean_text or raw_content)),
                "attachments": attachments,
                "links": links,
                "raw": {
                    **row,
                    "raw_content": raw_content,
                    "import_source": "lark_xing_crawl_output_xing_upload",
                    "resource_paths": [att.get("path") for att in attachments if att.get("path")],
                },
            })
            if limit and len(messages) >= limit:
                break
        return messages

    def load_xing_resources(self, import_root: str | Path) -> dict[str, list[dict[str, Any]]]:
        import_root = Path(import_root).resolve()
        rows = _read_csv(import_root / "_MANIFEST" / "xing_resource_files.csv")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            message_id = row.get("message_id", "")
            if not message_id:
                continue
            rel = row.get("relative_path", "")
            size_text = row.get("copied_bytes") or row.get("bytes") or "0"
            try:
                size = int(size_text)
            except ValueError:
                size = 0
            name = row.get("name") or Path(rel).name
            ext = Path(name or rel).suffix.lower()
            role = _attachment_evidence_role(name or rel, row.get("type"))
            grouped[message_id].append({
                "file_key": rel or row.get("name", ""),
                "kind": row.get("type") or "file",
                "name": name,
                "mime": None,
                "size": size,
                "path": str((import_root / rel).resolve()) if rel else None,
                "status": "metadata_only" if _bool_text(row.get("copied")) else "unavailable",
                "source_status": row.get("status", ""),
                "extension": ext,
                "evidence_role": role,
                "message_id": message_id,
                "thread_id": row.get("segment_id", ""),
                "reason": "pre_crawled_resource_metadata_only",
            })
        return dict(grouped)

    def _segment_text_history_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for msg in messages:
            grouped[str(msg.get("chat_id") or (msg.get("raw") or {}).get("source_chat_id") or "default")].append(msg)
        segmented: list[dict[str, Any]] = []
        for chat_id, items in grouped.items():
            items.sort(key=lambda msg: str(msg.get("create_time") or ""))
            segments: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            prev_time: datetime | None = None
            for msg in items:
                msg_time = _text_history_timestamp(msg.get("create_time"))
                should_break = False
                if current:
                    if len(current) >= TEXT_HISTORY_SEGMENT_MAX_MESSAGES:
                        should_break = True
                    elif prev_time is not None and msg_time is not None and (msg_time - prev_time).total_seconds() > TEXT_HISTORY_SEGMENT_GAP_HOURS * 3600:
                        should_break = True
                    elif self._should_break_text_history_segment(current, msg):
                        should_break = True
                if should_break:
                    segments.append(current)
                    current = []
                current.append(msg)
                if msg_time is not None:
                    prev_time = msg_time
            if current:
                segments.append(current)
            for index, segment in enumerate(segments, 1):
                segment_id = _text_history_segment_id(chat_id, str(segment[0].get("create_time") or ""), str(segment[-1].get("create_time") or ""), index)
                for msg in segment:
                    msg["thread_id"] = segment_id
                    raw = dict(msg.get("raw") or {})
                    raw["segment_id"] = segment_id
                    raw["source_chat_id"] = str(msg.get("chat_id") or raw.get("source_chat_id") or chat_id)
                    raw["source_chat_name"] = str(raw.get("chat_name") or raw.get("source_chat_name") or "")
                    raw["source_import"] = TEXT_HISTORY_SOURCE
                    msg["raw"] = raw
                    for link in msg.get("links") or []:
                        if isinstance(link, dict):
                            link["thread_id"] = segment_id
                    segmented.append(msg)
        segmented.sort(key=lambda msg: (str(msg.get("create_time") or ""), str(msg.get("message_id") or "")))
        return segmented

    def _should_break_text_history_segment(self, current: list[dict[str, Any]], msg: dict[str, Any]) -> bool:
        starts_new_fault = _is_fault_description(msg)
        current_has_fault = any(_is_fault_description(item) for item in current)
        current_has_progress = any(
            _is_diagnostic_action(item)
            or _is_resolution(item)
            or ((not _is_fault_description(item)) and _message_has_missing_info_request(str(item.get("text") or "")))
            or ((not _is_fault_description(item)) and _message_has_provided_info(item))
            for item in current
        )
        return bool(current and starts_new_fault and current_has_fault and current_has_progress)

    def aggregate_threads(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for msg in messages:
            grouped[str(msg.get("thread_id") or "default")].append(msg)
        summaries: list[dict[str, Any]] = []
        for thread_id, items in grouped.items():
            items.sort(key=lambda m: str(m.get("create_time") or ""))
            attachments = [att for msg in items for att in msg.get("attachments", [])]
            extracted = self.extract_fields(items, attachments)
            episodes = self.split_fault_episodes(thread_id, items)
            field_report_anchors = [anchor for msg in items for anchor in [_field_report_anchor(msg)] if anchor]
            summaries.append({
                "thread_id": thread_id,
                "participants": _unique((msg.get("sender") or {}).get("name") for msg in items),
                "start_time": str(items[0].get("create_time") or "") if items else "",
                "end_time": str(items[-1].get("create_time") or "") if items else "",
                "message_count": len(items),
                "attachments": attachments,
                "evidence_message_ids": self.evidence_ids(items, extracted),
                "extracted": extracted,
                "episodes": episodes,
                "field_report_anchors": field_report_anchors,
            })
        summaries.sort(key=lambda s: (str(s.get("start_time") or ""), str(s.get("thread_id") or "")))
        _attach_summary_relations(summaries, messages)
        return summaries

    def split_fault_episodes(self, thread_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deterministically split one chat segment into one or more fault episodes."""
        if not messages:
            return []
        field_report_anchors = {
            str(anchor.get("message_id") or ""): anchor
            for msg in messages
            for anchor in [_field_report_anchor(msg)]
            if anchor
        }
        messages = [dict(msg, _w1_seq_index=idx) for idx, msg in enumerate(_expand_multi_issue_messages(messages))]
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in messages:
            starts_new_fault = _is_fault_description(msg)
            starts_field_report_fault = msg.get("field_report_item_kind") == "fault_case"
            current_has_fault = any(_is_fault_description(m) for m in current)
            current_has_field_report_fault = any(
                item.get("field_report_item_kind") == "fault_case"
                for item in current
            )
            current_has_progress = any(
                _is_diagnostic_action(m)
                or _is_resolution(m)
                or ((not _is_fault_description(m)) and _message_has_missing_info_request(str(m.get("text") or "")))
                or ((not _is_fault_description(m)) and _message_has_provided_info(m))
                for m in current
            )
            next_fragment_boundary = _is_next_fragment_boundary(current, msg)
            distinct_fault_topic = _is_distinct_fault_topic_boundary(current, msg)
            stale_context_boundary = _is_stale_context_boundary(current, msg)
            if (
                current
                and (
                    stale_context_boundary
                    or (
                        starts_new_fault
                        and (
                            starts_field_report_fault
                            or (
                                current_has_fault
                                and (
                                    current_has_progress
                                    or next_fragment_boundary
                                    or distinct_fault_topic
                                    or current_has_field_report_fault
                                )
                            )
                        )
                    )
                )
            ):
                chunks.append(current)
                current = []
            current.append(msg)
        if current:
            chunks.append(current)

        episodes: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, 1):
            # The chunk is the authoritative W1 episode boundary.  Do not
            # rebuild context from the entire segment: that would leak other
            # field-report items back into this episode.
            episode_messages = _episode_local_messages(chunk)
            attachments = [att for msg in episode_messages for att in msg.get("attachments", [])]
            extracted = self.extract_fields(episode_messages, attachments)
            anchor_message = next((
                item for item in chunk
                if item.get("field_report_item_kind") == "fault_case"
                if str(item.get("source_message_id") or item.get("message_id") or "") in field_report_anchors
            ), None)
            anchor = field_report_anchors.get(str((anchor_message or {}).get("source_message_id") or (anchor_message or {}).get("message_id") or ""), {})
            anchor_item_index = int((anchor_message or {}).get("fragment_index") or 1) if anchor else 0
            if anchor:
                extracted["field_report_anchor"] = {
                    "anchor_id": anchor.get("anchor_id"),
                    "message_id": anchor.get("message_id"),
                    "author": anchor.get("author"),
                    "report_date": anchor.get("report_date"),
                    "site": anchor.get("site"),
                    "anchor_item_index": anchor_item_index,
                }
            fault_messages = [_message_brief(m) for m in episode_messages if _is_fault_description(m)]
            diagnostic_messages = [_message_brief(m) for m in episode_messages if _is_diagnostic_action(m)]
            resolution_messages = [_message_brief(m) for m in episode_messages if _is_resolution(m)]
            noise_messages = [_message_brief(m) for m in episode_messages if _is_noise(m)]
            evidence_message_ids = self.evidence_ids(episode_messages, extracted)
            message_ids = _message_ids(episode_messages)
            context_messages = [_message_brief(m) for m in episode_messages]
            context_message_ids = _message_ids(context_messages)
            has_fault = bool(fault_messages)
            has_check = bool(diagnostic_messages)
            has_solution = bool(resolution_messages)
            if not has_fault and not has_check and not has_solution:
                completeness = "noise"
            elif has_fault and has_check and has_solution:
                completeness = "complete"
            else:
                completeness = "partial"
            episodes.append({
                "episode_id": f"{thread_id}:episode:{index}",
                "thread_id": thread_id,
                "source_thread_id": thread_id,
                "completeness": completeness,
                "fault_description_messages": fault_messages,
                "diagnostic_chain_messages": diagnostic_messages,
                "resolution_messages": resolution_messages,
                "noise_messages": noise_messages,
                "case_context_messages": context_messages,
                "message_ids": message_ids,
                "context_message_ids": context_message_ids,
                "summary_context_message_ids": [],
                "full_context_message_ids": _unique([*message_ids, *context_message_ids], limit=500),
                "message_refs": {
                    "message_ids": message_ids,
                    "context_message_ids": context_message_ids,
                    "summary_context_message_ids": [],
                    "full_context_message_ids": _unique([*message_ids, *context_message_ids], limit=500),
                },
                "evidence_message_ids": evidence_message_ids,
                "source_offsets": extracted.get("source_offsets") or [],
                "attachments": attachments,
                "extracted": extracted,
                "field_report_anchor": extracted.get("field_report_anchor") or {},
                "message_count": len(chunk),
                "start_time": str(chunk[0].get("create_time") or "") if chunk else "",
                "end_time": str(chunk[-1].get("create_time") or "") if chunk else "",
            })
        return episodes

    def extract_fields(self, messages: list[dict[str, Any]], attachments: list[dict[str, Any]]) -> dict[str, Any]:
        corpus = "\n".join(str(m.get("text") or "") for m in messages)
        unassigned_shared_evidence = [
            dict(msg.get("shared_source_evidence_unassigned") or {})
            for msg in messages
            if isinstance(msg.get("shared_source_evidence_unassigned"), dict)
        ]
        chat_names = [str((m.get("raw") or {}).get("chat_name") or "") for m in messages]
        sites = list(BRACKET_SITE_RE.findall(corpus))
        sites.extend(_site_from_chat_name(name) for name in chat_names)
        attachment_names = [str(att.get("name") or att.get("file_key") or "") for att in attachments]
        attachment_evidence = [_attachment_evidence(att) for att in attachments]
        text_artifacts = list(LOG_FILE_RE.findall(corpus))
        text_project_files = [path for path in text_artifacts if _attachment_evidence_role(path, "file") == "program_file"]
        text_log_paths = [path for path in text_artifacts if _attachment_evidence_role(path, "file") == "log_package"]
        project_files = [*text_project_files, *(item["name"] for item in attachment_evidence if item.get("evidence_role") == "program_file")]
        log_artifacts = [item["name"] for item in attachment_evidence if item.get("evidence_role") == "log_package"]
        links = [link for msg in messages for link in list(msg.get("links") or [])]
        jira_links = [link for link in links if link.get("type") == "jira"]
        jira_ids = _unique(JIRA_RE.findall(corpus), limit=100)
        tool_evidence = _build_tool_evidence(attachments, links, jira_ids)
        log_paths = [*text_log_paths, *log_artifacts]
        attribution = _build_attribution(messages)
        jira_submission_signals = [_message_brief(msg) for msg in messages if _message_has_jira_submission_signal(msg)]
        daily_report_signals = [_message_brief(msg) for msg in messages if _message_is_daily_report(str(msg.get("text") or ""))]
        owner_handoff_signals = [_message_brief(msg) for msg in messages if _message_has_owner_assignment_signal(msg)[0] or _message_has_owner_takeover_signal(msg)[0]]
        tool_evidence["text_only_signals"] = {
            "jira_submission_signals": jira_submission_signals[:20],
            "daily_report_signals": daily_report_signals[:20],
            "owner_handoff_signals": owner_handoff_signals[:20],
        }
        source_offsets: list[dict[str, Any]] = []
        source_offsets.extend(_find_offsets("jira_ids", JIRA_RE, messages))
        source_offsets.extend(_find_version_offsets(messages))
        source_offsets.extend(_find_offsets("ip_config", IP_RE, messages))
        source_offsets.extend(_signal_offsets("jira_submission_signals", JIRA_SUBMISSION_MARKERS, messages))
        source_offsets.extend(_signal_offsets("daily_report_signals", DAILY_REPORT_MARKERS, messages))
        source_offsets.extend(_signal_offsets("owner_handoff_signals", OWNER_REQUEST_MARKERS + OWNER_TAKEOVER_MARKERS, messages))
        for msg in messages:
            content = str(msg.get("text") or "")
            for match in LOG_FILE_RE.finditer(content):
                value = match.group(0)
                role = _attachment_evidence_role(value, "file")
                field = "project_files" if role == "program_file" else "log_paths"
                source_offsets.append({
                    "field": field,
                    "value": value,
                    "source": "message.text",
                    "message_id": msg.get("message_id", ""),
                    "thread_id": msg.get("thread_id", ""),
                    "start": match.start(),
                    "end": match.end(),
                })
        for link in links:
            field = "jira_links" if link.get("type") == "jira" else "links"
            source_offsets.append({
                "field": field,
                "value": link.get("url", ""),
                "source": link.get("source", "message.raw_content.link"),
                "message_id": link.get("message_id", ""),
                "thread_id": link.get("thread_id", ""),
                "start": link.get("start", 0),
                "end": link.get("end", 0),
            })
        for msg in messages:
            site = _site_from_chat_name(str((msg.get("raw") or {}).get("chat_name") or ""))
            if site:
                source_offsets.append({
                    "field": "sites",
                    "value": site,
                    "source": "message.raw.chat_name",
                    "message_id": msg.get("message_id", ""),
                    "thread_id": msg.get("thread_id", ""),
                    "start": 0,
                    "end": len(site),
                })
        for att in attachments:
            name = str(att.get("name") or att.get("file_key") or "")
            role = str(att.get("evidence_role") or _attachment_evidence_role(name, att.get("kind")))
            if role in {"log_package", "program_file", "sample_image", "environment", "data_file"}:
                field = {
                    "log_package": "log_paths",
                    "program_file": "project_files",
                    "sample_image": "sample_images",
                    "environment": "environment_files",
                    "data_file": "data_files",
                }.get(role, "attachment_evidence")
                source_offsets.append({
                    "field": field,
                    "value": name,
                    "source": "attachment.name",
                    "attachment_key": att.get("file_key", ""),
                    "message_id": att.get("message_id", ""),
                    "thread_id": att.get("thread_id", ""),
                    "start": 0,
                    "end": len(name),
                })
        useful = [m for m in messages if _message_useful(m)]
        debug_actions = [_one_line(m.get("text")) for m in useful if _is_action_focus_text(str(m.get("text") or ""))]
        conclusion = ""
        for msg in reversed(useful):
            text = _one_line(msg.get("text"))
            if _is_resolution(msg):
                conclusion = text
                break
        symptom_raw = _one_line(useful[0].get("text"), 1000) if useful else _one_line(messages[0].get("text"), 1000) if messages else ""
        fault_focus_text = _best_fault_focus(messages, useful) or symptom_raw
        fault_focus_confidence = _fault_focus_confidence(fault_focus_text)
        missing_info_requests = _missing_info_requests(messages)
        missing_info = [_one_line(item.get("text")) for item in missing_info_requests]
        return {
            "jira_ids": jira_ids,
            "versions": _version_values(corpus),
            "devices": _unique(DEVICE_RE.findall(corpus), limit=20),
            "sites": _unique(sites, limit=20),
            "log_paths": _unique(log_paths, limit=50),
            "project_files": _unique(project_files, limit=50),
            "links": links[:200],
            "jira_links": jira_links[:100],
            "tool_evidence": tool_evidence,
            "source_offsets": _dedupe_offsets(source_offsets),
            "unassigned_shared_evidence": unassigned_shared_evidence,
            "symptom_raw": symptom_raw,
            "fault_focus_text": fault_focus_text,
            "fault_focus_confidence": fault_focus_confidence,
            "debug_actions": _unique(debug_actions, limit=12),
            "conclusion": conclusion,
            "key_conclusion": conclusion,
            "missing_info": _unique(missing_info, limit=10),
            "missing_info_requests": missing_info_requests,
            "attribution": attribution,
            "jira_submission_signals": jira_submission_signals[:20],
            "daily_report_signals": daily_report_signals[:20],
            "owner_handoff_signals": owner_handoff_signals[:20],
            "artifacts": {
                "jira_ids": jira_ids,
                "versions": _version_values(corpus),
                "devices": _unique(DEVICE_RE.findall(corpus), limit=20),
                "sites": _unique(sites, limit=20),
                "log_paths": _unique(log_paths, limit=50),
                "project_files": _unique(project_files, limit=50),
                "links": links[:200],
                "jira_links": jira_links[:100],
                "tool_evidence": tool_evidence,
                "attribution": attribution,
                "attachment_evidence": attachment_evidence[:200],
                "attachment_keys": _unique((att.get("file_key") for att in attachments), limit=200),
                "attachment_names": _unique(attachment_names, limit=200),
            },
        }

    def evidence_ids(self, messages: list[dict[str, Any]], extracted: dict[str, Any]) -> list[str]:
        signal_ids = {str(item.get("message_id") or "") for item in extracted.get("source_offsets") or [] if item.get("message_id")}
        useful_ids = [str(m.get("message_id") or "") for m in messages if _message_useful(m)]
        hit_ids = [str(m.get("message_id") or "") for m in messages if _bool_text((m.get("raw") or {}).get("is_hit")) or _bool_text((m.get("raw") or {}).get("is_xing_related"))]
        return _unique([*signal_ids, *useful_ids, *hit_ids], limit=30)

    def build_manifest(
        self,
        import_root: Path,
        messages: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        *,
        limit: int,
        hits_only: bool,
    ) -> dict[str, Any]:
        type_counts = Counter(str(m.get("msg_type") or "") for m in messages)
        episode_counts = Counter(str(e.get("completeness") or "") for e in episodes)
        return {
            "run_id": "xing_upload_real" if not limit else f"xing_upload_sample_{limit}",
            "source": "lark_xing_crawl_output_xing_upload",
            "import_root": str(import_root),
            "options": {"limit": limit, "hits_only": hits_only},
            "counts": {
                "messages": len(messages),
                "threads": len(summaries),
                "episodes": len(episodes),
                "attachments": sum(len(m.get("attachments") or []) for m in messages),
                "hits": sum(1 for m in messages if _bool_text((m.get("raw") or {}).get("is_hit"))),
            },
            "episode_completeness": dict(episode_counts),
            "message_types": dict(type_counts),
        }

    def build_text_history_manifest(
        self,
        import_root: Path,
        source_path: Path,
        source_rows: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        *,
        limit: int,
    ) -> dict[str, Any]:
        type_counts = Counter(str(m.get("msg_type") or "") for m in messages)
        episode_counts = Counter(str(e.get("completeness") or "") for e in episodes)
        chat_ids = {str((m.get("raw") or {}).get("source_chat_id") or m.get("chat_id") or "") for m in messages if str((m.get("raw") or {}).get("source_chat_id") or m.get("chat_id") or "")}
        owner_assignment_count = sum(len(((episode.get("extracted") or {}).get("attribution") or {}).get("owner_assignments") or []) for episode in episodes)
        reporter_candidate_count = sum(len(((episode.get("extracted") or {}).get("attribution") or {}).get("reporter_candidates") or []) for episode in episodes)
        classified_issue_count = sum(1 for episode in episodes if (((episode.get("extracted") or {}).get("attribution") or {}).get("classification_hypotheses") or []))
        return {
            "run_id": "text_history_real" if not limit else f"text_history_sample_{limit}",
            "source": TEXT_HISTORY_SOURCE,
            "import_root": str(import_root),
            "source_path": str(source_path),
            "options": {"limit": limit},
            "counts": {
                "messages": len(messages),
                "threads": len(summaries),
                "episodes": len(episodes),
                "attachments": 0,
                "hits": 0,
                "chat_count": len(chat_ids),
                "segment_count": len(summaries),
                "messages_without_thread_id": sum(1 for row in source_rows if not str(row.get("thread_id") or "").strip()),
                "owner_assignment_count": owner_assignment_count,
                "reporter_candidate_count": reporter_candidate_count,
                "classified_issue_count": classified_issue_count,
            },
            "episode_completeness": dict(episode_counts),
            "message_types": dict(type_counts),
        }

    def write_run(self, out_dir: str | Path, run: dict[str, Any]) -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        messages = run.get("messages") or []
        summaries = run.get("thread_summaries") or []
        episodes = run.get("episodes") or []
        manifest = run.get("run_manifest") or {}
        field_report_anchors = run.get("field_report_anchors") or []
        observed_people = run.get("observed_people") or []
        with (out / "messages.jsonl").open("w", encoding="utf-8") as f:
            for row in messages:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out / "thread_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "episodes.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "field_report_anchors.json").write_text(json.dumps(field_report_anchors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "observed_people.json").write_text(json.dumps(observed_people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "messages": str(out / "messages.jsonl"),
            "thread_summaries": str(out / "thread_summaries.json"),
            "episodes": str(out / "episodes.json"),
            "field_report_anchors": str(out / "field_report_anchors.json"),
            "observed_people": str(out / "observed_people.json"),
            "run_manifest": str(out / "run_manifest.json"),
        }
