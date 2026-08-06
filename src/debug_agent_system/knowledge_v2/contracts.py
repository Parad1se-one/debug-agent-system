"""Shared constants and lightweight contracts for KG v2."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any

V2_PRIMARY_KEYS = {
    "KnowledgeDocument": "document_id",
    "MediaAsset": "media_id",
    "KnowledgeSection": "section_id",
    "ProcedureStep": "procedure_step_id",
    "FaultFamily": "family_id",
    "FaultVariant": "variant_id",
    "DiagnosticAction": "action_id",
    "ActionOutcome": "outcome_id",
    "RequiredInfoSpec": "required_info_id",
    "DiagnosticTrace": "trace_id",
    "TraceStep": "trace_step_id",
    "ExecutionObservation": "observation_id",
    "BranchRule": "branch_rule_id",
    "DecisionPolicy": "policy_id",
    "EvidenceItem": "evidence_id",
    "SourceCase": "case_id",
    "DebugConcept": "concept_id",
    "TermExpression": "term_id",
    "TermSense": "sense_id",
}
TRACE_EXECUTION_STATUSES = {"actual", "recommended"}
BRANCH_KINDS = {"observed_transition", "reviewed_recommendation"}
BRANCH_TERMINAL_STATUSES = {"continue", "resolved", "monitoring", "unresolved", "escalated"}

ACTION_ROLES = {"inspect", "collect", "compare", "change", "verify", "observe", "escalate"}
OUTCOME_TYPES = {
    "verified_fix",
    "ineffective",
    "partial_temporary",
    "mitigation_observed",
    "recurred",
    "pending_validation",
    "diagnostic_method",
    "context_not_root_cause",
}
OUTCOME_ORIGINS = {
    # Accepted by a human reviewer (for example a frozen Gold case).
    "human_reviewed",
    # Explicitly extracted from source text or a model's evidence-bound output.
    "source_extracted",
    # Deterministically inferred from source-grounded action/result semantics.
    "rule_inferred",
    # Added only to keep the Action -> Outcome audit mapping total when the
    # source does not provide a durable observed result.
    "synthetic_fallback",
}
ACTION_EVIDENCE_SCOPES = {
    # The action is stated by a message assigned directly to this episode.
    "current_episode_direct",
    # Both direct and W7-promoted messages support the same action.
    "mixed_current_and_promoted",
    # The action exists only in neighbouring evidence promoted by W7.  It is
    # reviewable case evidence, but not an automatic execution-policy fact.
    "w7_promoted_only",
    # Frozen expert-reviewed actions are authoritative within their Gold case.
    "human_reviewed",
    # Compatibility value for older bundles without message-role provenance.
    "legacy_unspecified",
}
INTERNAL_REQUIRED_INFO_SLOTS = {
    "log_package",
    "dmp_package",
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
    "memory_cpu_test",
    "driver_context",
    "production_constraint",
    "other",
}
EXECUTION_SLOT_MAP = {
    "log_package": "log_package",
    "dmp_package": "log_package",
    "software_version": "software_version",
    "error_phase": "error_phase",
    "error_message": "error_message",
    "device_model": "device_model",
    "site": "site",
    "ip_config": "ip_config",
    "repro_steps": "repro_steps",
    "sample_image": "sample_image",
    "program_file": "program_file",
    "environment": "environment",
    "owner_context": "owner_context",
    "memory_cpu_test": "environment",
    "driver_context": "owner_context",
    "production_constraint": "owner_context",
    "other": "other",
}
EXECUTION_CHECK_ROLES = {"inspect", "collect", "compare", "change", "verify"}
UNSAFE_OUTCOME_TYPES = {"pending_validation"}
APPROVED_FAMILY_LABELS = {
    "CPU温度异常",
    "工控机无法开机",
    "工控机黑屏无显示",
    "工控机蓝屏",
    "工控机异常重启",
    "工控机死机",
    "网络连接异常",
    "Buddy问题",
    "键盘输入异常",
    "操作系统启动失败",
    "BIOS 启动配置异常",
    "多硬盘启动冲突",
    "用户配置加载失败",
    "运控初始化失败",
    "光源初始化失败",
    "主程序初始化卡住无明确报错",
    "主程序无法打开",
    "工厂程序无法打开",
    "运控程序无法打开",
    "SPC 页面无法打开",
    "Buddy 模板缺失",
    "Buddy 模板创建失败",
    "模板文件损坏",
    "程序运行卡顿",
    "软件卡死无响应",
    "磁盘 I/O 异常",
    "CUDA 计算设备不可用",
    "复判站主机通信异常",
    "复判保存结果失败",
    "USB 设备识别异常",
    "相机拍摄失败",
    "相机初始化失败",
    "光源异常",
    "光控通信异常",
    "运控卡初始化异常",
    "控制器网络配置异常",
    "进板失败",
    "出板失败",
    "卡板",
    "挡块异常",
    "顶升机构异常",
    "皮带运行异常",
    "轨道宽度无法调节",
    "扫码枪异常",
    "气压异常",
    "PCIe 板卡检测异常",
    "外设连接不稳定",
    "MES 过站异常",
    "许可证/加密狗异常",
    "坏板标记异常",
    "复判结果显示异常",
    "机械运动异响",
    "传感器感应异常",
    "CAD 导入失败",
    "CAD 角度不一致",
    "CAD 自动对齐失败",
    "程序板卡加载失败",
    "Mark 点对齐失败",
    "识别框大小不准确",
    "器件框角度不匹配",
    "焊盘框不对齐",
    "扫码识别失败",
    "DM 码识别失败",
    "框选识别不准",
    "界面显示异常",
    "误报调优异常",
    "漏检调优异常",
    "CT 时间异常增加",
    "复判站出图慢",
    "复判站加载板卡异常",
    "主程序/系统异常",
    "算法/程序调优异常",
    "2D成像一致性异常",
    "图像拼接错位",
    "相机成像模糊",
    "标定问题",
}
PSEUDO_FAMILY_LABELS = {
    "AOI_复判站",
    "AOI检测软件",
    "display",
    "camera",
    "software",
    "算法/检测逻辑",
    "显示/分辨率/缩放",
    "显示/界面",
    "复判流程",
    "硬件/运控异常",
}
FAMILY_SUBSYSTEM_EXPECTED = {
    "CPU温度异常": "工控机/CPU散热",
    "工控机无法开机": "工控机/启动链路",
    "工控机黑屏无显示": "工控机/显示链路",
    "工控机蓝屏": "工控机/Windows 内核",
    "工控机异常重启": "工控机/系统运行稳定性",
    "工控机死机": "工控机/系统运行稳定性",
    "网络连接异常": "工控机/网络链路",
    "Buddy问题": "Buddy/模板与冷存储",
    "键盘输入异常": "工控机/USB外设",
    "2D成像一致性异常": "2D相机/光学链路",
    "图像拼接错位": "相机/拼图链路",
    "相机成像模糊": "相机/光学成像",
    "标定问题": "运控/业务原点标定",
    "操作系统启动失败": "工控机/操作系统启动",
    "BIOS 启动配置异常": "工控机/BIOS启动",
    "多硬盘启动冲突": "工控机/BIOS启动",
    "用户配置加载失败": "主程序配置/复判站配置",
    "运控初始化失败": "运控/初始化链路",
    "光源初始化失败": "光源/光控链路",
    "主程序初始化卡住无明确报错": "主程序/初始化链路",
    "主程序无法打开": "主程序/启动链路",
    "工厂程序无法打开": "工厂程序/启动链路",
    "运控程序无法打开": "运控程序/启动链路",
    "SPC 页面无法打开": "SPC/页面链路",
    "Buddy 模板缺失": "模板/Buddy",
    "Buddy 模板创建失败": "模板/Buddy",
    "模板文件损坏": "模板/文件链路",
    "程序运行卡顿": "主程序/运行性能",
    "软件卡死无响应": "主程序/运行稳定性",
    "磁盘 I/O 异常": "磁盘/存储链路",
    "CUDA 计算设备不可用": "显卡/CUDA链路",
    "复判站主机通信异常": "复判/主机通信",
    "复判保存结果失败": "复判/结果保存",
    "USB 设备识别异常": "USB/外设链路",
    "相机拍摄失败": "相机/采集链路",
    "相机初始化失败": "相机/初始化链路",
    "光源异常": "光源/硬件链路",
    "光控通信异常": "光源/通信链路",
    "运控卡初始化异常": "运控卡/初始化链路",
    "控制器网络配置异常": "控制器/网络链路",
    "进板失败": "进出板/轨道链路",
    "出板失败": "进出板/轨道链路",
    "卡板": "进出板/轨道链路",
    "挡块异常": "轨道/挡块机构",
    "顶升机构异常": "轨道/顶升机构",
    "皮带运行异常": "轨道/皮带机构",
    "轨道宽度无法调节": "轨道/宽度调节",
    "扫码枪异常": "扫码枪/外设链路",
    "气压异常": "气路/气压链路",
    "PCIe 板卡检测异常": "PCIe/板卡链路",
    "外设连接不稳定": "外设/连接链路",
    "MES 过站异常": "MES/接口链路",
    "许可证/加密狗异常": "授权/许可链路",
    "坏板标记异常": "坏板标记/流程链路",
    "复判结果显示异常": "复判/结果显示",
    "机械运动异响": "轨道/机械运动",
    "传感器感应异常": "传感器/感应链路",
    "CAD 导入失败": "CAD/程序导入",
    "CAD 角度不一致": "CAD/程序导入",
    "CAD 自动对齐失败": "CAD/自动对齐",
    "程序板卡加载失败": "程序/板卡加载",
    "Mark 点对齐失败": "Mark/定位对齐",
    "识别框大小不准确": "识别框/几何参数",
    "器件框角度不匹配": "识别框/角度参数",
    "焊盘框不对齐": "焊盘框/几何参数",
    "扫码识别失败": "扫码/识别链路",
    "DM 码识别失败": "扫码/识别链路",
    "框选识别不准": "框选/识别链路",
    "界面显示异常": "显示/界面",
    "误报调优异常": "算法/误报调优",
    "漏检调优异常": "算法/漏检调优",
    "CT 时间异常增加": "节拍/CT",
    "复判站出图慢": "复判/显示性能",
    "复判站加载板卡异常": "复判/板卡加载",
    "主程序/系统异常": "主程序/系统",
    "算法/程序调优异常": "算法/程序调优",
}

_NON_ID = re.compile(r"[^A-Za-z0-9_.:-]+")


@dataclass(slots=True)
class ProjectedError:
    error_id: str
    label: str
    symptom: str
    category: str
    subsystem: str = ""
    scenario: str = ""
    keywords: list[str] = field(default_factory=list)
    required_info: list[str] = field(default_factory=list)
    required_info_schema: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedCheck:
    check_id: str
    label: str
    how_to_check: str
    step_order: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedSolution:
    solution_id: str
    content: str
    method: str
    evidence_level: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedRequiredInfo:
    slot: str
    question: str
    condition: str
    blocks: list[str]
    priority: str
    why_required: str
    evidence: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedPolicy:
    policy_id: str
    target_error_id: str
    ordered_checks: list[dict[str, Any]]
    solution_stats: list[dict[str, Any]]
    unsafe_actions: list[dict[str, Any]]
    payload: dict[str, Any] = field(default_factory=dict)


def make_id(prefix: str, value: str, *, limit: int = 72) -> str:
    safe = _NON_ID.sub("-", str(value or "").strip()).strip("-").lower()
    if not safe:
        safe = hashlib.sha1(str(value or prefix).encode("utf-8")).hexdigest()[:12]
    if len(safe) > limit:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
        safe = safe[: limit - 11].rstrip("-") + "-" + digest
    return f"{prefix}:{safe}"


def make_family_id(label: str) -> str:
    """Build a collision-safe family id while retaining legacy CJK ids."""

    value = str(label or "").strip()
    if re.search(r"[A-Za-z0-9]", value) and any(ord(char) > 127 for char in value):
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
        return make_id("family", f"{value}:{digest}")
    return make_id("family", value)


def trim_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def humanize_id(value: str) -> str:
    text = str(value or "")
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace("-", " ").replace("_", " ")
    return " ".join(part for part in text.split() if part)
