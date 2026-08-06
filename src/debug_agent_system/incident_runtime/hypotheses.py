"""Evidence matrix and next-best-test planning."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel

from .contracts import DiagnosticEvent, DiagnosticHypothesis, DiagnosticTest, EnvironmentSnapshot

# 常见 Windows 终止代码（蓝屏 bugcheck）名称 → 规范化 0x 代码。
# 仅用于把 query 中的终止代码与 EVTX Kernel-Power 41 / WER 1001 的
# bugcheck code 对齐，不宣称根因。
_STOP_CODE_MAP = {
    "CRITICAL PROCESS DIED": "0x000000EF",
    "CRITICAL_PROCESS_DIED": "0x000000EF",
    "DRIVER_IRQL_NOT_LESS_OR_EQUAL": "0x000000D1",
    "PAGE_FAULT_IN_NONPAGED_AREA": "0x00000050",
    "KERNEL_DATA_INPAGE_ERROR": "0x0000007A",
    "IRQL_NOT_LESS_OR_EQUAL": "0x0000000A",
    "SYSTEM_SERVICE_EXCEPTION": "0x0000003B",
    "KMODE_EXCEPTION_NOT_HANDLED": "0x0000001E",
    "BAD_POOL_CALLER": "0x000000C2",
    "UNEXPECTED_KERNEL_MODE_TRAP": "0x0000007F",
    "KERNEL_AUTO_BOOST_LOCK_ACQUISITION_WITH_RAISED_IRQL": "0x0000008E",
    "DRIVER_CORRUPTED_EXPOOL": "0x000000C9",
    "THREAD_STUCK_IN_DEVICE_DRIVER": "0x000000EA",
    "KERNEL_SECURITY_CHECK_FAILURE": "0x00000139",
    "VIDEO_TDR_FAILURE": "0x00000116",
}


class HypothesisRuntime:
    schema_version = "debug_agent_system.incident_hypotheses.v1"

    def __init__(self, read_model: KGV2ReadModel) -> None:
        self.read_model = read_model

    def build(
        self,
        retrieval: dict[str, Any],
        environment: EnvironmentSnapshot,
        *,
        events: Iterable[DiagnosticEvent] = (),
        correlations: Iterable[dict[str, Any]] = (),
        max_hypotheses: int = 5,
        query: str = "",
    ) -> list[DiagnosticHypothesis]:
        hypotheses: list[DiagnosticHypothesis] = []
        event_list = list(events)
        correlation_list = list(correlations)
        candidates = [
            raw
            for raw in retrieval.get("candidates") or []
            if isinstance(raw, dict)
            and raw.get("support_evidence_ids")
            and raw.get("matched_incident_anchors")
        ]
        for raw in candidates[:max_hypotheses]:
            if not isinstance(raw, dict):
                continue
            variant_id = str(raw.get("variant_id") or "")
            family_id = str(raw.get("family_id") or "")
            variant = self.read_model.get(variant_id) or {}
            family = self.read_model.get(family_id) or {}
            support = _dedupe(raw.get("support_evidence_ids") or [])
            missing = _missing_items(raw.get("required_info") or [], environment)
            score = float(raw.get("score") or 0.0)
            anchor_count = len(raw.get("matched_incident_anchors") or [])
            confidence = min(0.79, 0.12 + min(score, 50.0) / 100.0 + min(anchor_count, 4) * 0.08)
            if not support:
                confidence = min(confidence, 0.34)
            status = "supported" if support and not missing else ("needs_evidence" if missing else "candidate")
            hypotheses.append(DiagnosticHypothesis(
                hypothesis_id=f"hypothesis:{variant_id or hashlib.sha256(str(raw).encode()).hexdigest()[:12]}",
                label=str(raw.get("variant_label") or variant.get("label") or variant_id),
                failure_mechanism=str(variant.get("summary") or raw.get("variant_label") or "待验证的故障机制"),
                suspected_component=str(family.get("subsystem") or family.get("label") or "待确认组件"),
                family_id=family_id,
                variant_id=variant_id,
                support_evidence_ids=support,
                contradict_evidence_ids=[],
                missing_evidence=missing,
                confidence=round(confidence, 4),
                status=status,  # type: ignore[arg-type]
                retrieval_score=score,
                source_ids=_dedupe(raw.get("source_ids") or []),
            ))
        if not hypotheses:
            fused = _cross_source_hypothesis(event_list, correlation_list, environment)
            if fused is not None:
                hypotheses.append(fused)
        if not hypotheses:
            blue_screen = _windows_blue_screen_hypothesis(
                event_list, correlation_list, query
            )
            if blue_screen is not None:
                hypotheses.append(blue_screen)
        if not hypotheses:
            anchors = [
                item
                for item in retrieval.get("anchors") or []
                if isinstance(item, dict)
                and item.get("stability") != "volatile"
                and item.get("value")
            ]
            diagnostic_anchors = [
                item
                for item in anchors
                if item.get("kind") in {"error_code", "function", "component"}
                or (
                    item.get("kind") == "event_kind"
                    and item.get("value") not in {
                        "diagnostic_event", "timeout", "reset", "crash",
                        "exception", "process_start", "process_exit",
                    }
                )
            ]
            support = _dedupe(
                evidence_id
                for item in diagnostic_anchors
                for evidence_id in item.get("evidence_ids") or []
            )[:32]
            signatures = _dedupe(
                item.get("value")
                for item in diagnostic_anchors
                if item.get("kind") in {"error_code", "event_kind", "function", "component"}
            )
            label = (
                "案件错误签名待定位：" + " / ".join(signatures[:3])
                if signatures
                else "当前证据不足以映射到 KG_v2 故障候选"
            )
            hypotheses.append(DiagnosticHypothesis(
                hypothesis_id="hypothesis:incident-signature-unmapped",
                label=label,
                failure_mechanism=(
                    "案件已证明这些错误签名同时出现，但尚无足够证据把异常检测点解释为唯一根因"
                    if support
                    else "需要补充可复现条件、完整日志和环境信息"
                ),
                suspected_component=next(
                    (
                        str(item.get("value"))
                        for item in diagnostic_anchors
                        if item.get("kind") == "component"
                    ),
                    "未定位",
                ),
                support_evidence_ids=support,
                missing_evidence=[
                    "异常发生前后至少 100 行完整日志",
                    "产品、操作系统、GPU/驱动及相关运行库版本矩阵",
                    "稳定复现步骤以及正常设备/异常设备对照结果",
                ],
                confidence=0.25 if support else 0.0,
                status="needs_evidence" if support else "inconclusive",
                source_ids=[],
            ))
        return hypotheses

    def propose_tests(
        self,
        hypotheses: list[DiagnosticHypothesis],
        environment: EnvironmentSnapshot,
        *,
        max_tests: int = 8,
    ) -> list[DiagnosticTest]:
        tests: list[DiagnosticTest] = []
        all_ids = [item.hypothesis_id for item in hypotheses if item.variant_id]
        missing = _dedupe(item for hypothesis in hypotheses for item in hypothesis.missing_evidence)
        for index, item in enumerate(missing[:max_tests]):
            tests.append(DiagnosticTest(
                test_id=f"test:required-info:{index + 1}",
                title=f"补齐证据：{item[:80]}",
                instruction=f"只读采集并记录“{item}”，保留原始文件、时间戳和来源；不要先执行修复动作。",
                distinguishes_hypothesis_ids=all_ids,
                expected_observations=["观察结果应能支持、削弱或排除至少一个候选假设"],
                evidence_required=[item],
                information_gain=max(0.2, round(1.0 - index * 0.08, 2)),
                cost="low",
                risk="safe",
            ))
        core_environment = {
            "windows_version", "driver_version", "nvidia_driver_version",
            "cuda_version", "cuda_runtime_version", "opencv_version", "gpu_model",
        }
        if not core_environment.intersection(environment.values):
            tests.insert(0, DiagnosticTest(
                test_id="test:environment-snapshot",
                title="采集环境版本矩阵",
                instruction="记录产品、业务模块、操作系统、驱动、CUDA/OpenCV、硬件型号和配置版本。",
                distinguishes_hypothesis_ids=all_ids,
                expected_observations=["确认问题是否与特定版本、驱动或设备组合相关"],
                evidence_required=["环境版本矩阵"],
                information_gain=0.95,
                cost="low",
                risk="safe",
            ))
        if len(hypotheses) > 1:
            tests.append(DiagnosticTest(
                test_id="test:controlled-reproduction",
                title="最小化、受控复现",
                instruction="在保留现场证据后，对单次与连续运行、正常设备与异常设备进行对照，不修改多个变量。",
                distinguishes_hypothesis_ids=all_ids,
                expected_observations=["记录首次异常前后的日志、组件状态和复现边界"],
                evidence_required=["复现步骤", "对照结果", "异常时间窗口"],
                information_gain=0.9,
                cost="medium",
                risk="controlled",
            ))
        tests.sort(key=lambda item: (-item.information_gain, item.cost, item.test_id))
        return tests[:max_tests]


def _missing_items(raw_items: Iterable[Any], environment: EnvironmentSnapshot) -> list[str]:
    result: list[str] = []
    environment_text = " ".join(value for values in environment.values.values() for value in values).lower()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("question") or raw.get("label") or raw.get("summary") or raw.get("required_info") or "").strip()
        if text and text.lower() not in environment_text and text not in result:
            result.append(text)
    return result


def _stop_code_from_query(query: str) -> list[str]:
    """Normalize stop codes mentioned in the Query into 0x form.

    Supports both the English bugcheck name ("CRITICAL PROCESS DIED") and an
    explicit 0x value ("0x000000EF").  These become alignment anchors for the
    EVTX bugcheck codes rather than standalone conclusions.
    """

    text = str(query or "").upper()
    codes: list[str] = []
    for name, code in _STOP_CODE_MAP.items():
        if name in text:
            codes.append(code)
    for match in re.finditer(r"(?i)0x([0-9a-f]{6,10})", text):
        value = match.group(1).upper().rjust(8, "0")
        codes.append(f"0x{value}")
    return list(dict.fromkeys(codes))


def _normalize_bugcheck(value: Any) -> str:
    """Normalize a bugcheck code to canonical 0x00000000 form."""

    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("0x"):
        digits = re.sub(r"(?i)0x", "", text).upper()
        return f"0x{digits.rjust(8, '0')}"
    if re.fullmatch(r"\d{1,8}", text):
        return f"0x{int(text, 10):08X}"
    return text.upper()


def _windows_blue_screen_hypothesis(
    events: list[DiagnosticEvent],
    correlations: list[dict[str, Any]],
    query: str,
) -> DiagnosticHypothesis | None:
    """Fuse EVTX bugcheck signals into a Windows blue-screen hypothesis.

    This fires when the diagnostic package contains a Kernel-Power 41
    unexpected-shutdown record, a WER 1001 BlueScreen report, or both.  When
    the Query states a stop code, it is aligned against the EVTX bugcheck
    codes so the hypothesis names the exact termination code the customer
    reported instead of guessing a signature.
    """

    by_kind: dict[str, list[DiagnosticEvent]] = {}
    for event in events:
        by_kind.setdefault(event.event_kind, []).append(event)
    power_loss = by_kind.get("kernel_power_loss", [])
    blue_screen = by_kind.get("windows_blue_screen", [])
    if not power_loss and not blue_screen:
        return None
    support = _dedupe(
        evidence_id
        for event in [*power_loss, *blue_screen]
        for evidence_id in event.evidence_ids
    )[:64]
    evtx_codes = _dedupe(
        code
        for event in [*power_loss, *blue_screen]
        for code in event.error_codes
        if code.lower().startswith("bugcheck:")
    )
    evtx_normalized = {_normalize_bugcheck(code.split(":", 1)[-1]) for code in evtx_codes}
    query_codes = {_normalize_bugcheck(code) for code in _stop_code_from_query(query)}
    aligned = sorted(evtx_normalized.intersection(query_codes)) if query_codes else []
    recurrent = any(
        item.get("type") == "repeated_failure_signature"
        for item in correlations
    )
    confidence = 0.42
    confidence += 0.08 if power_loss else 0.0
    confidence += 0.08 if blue_screen else 0.0
    confidence += 0.10 if aligned else 0.0
    confidence += 0.05 if recurrent else 0.0
    confidence = min(confidence, 0.75)
    if aligned:
        label = f"Windows 蓝屏：终止代码 {aligned[0]}"
        mechanism = (
            "EVTX 同时记录非正常关机（Kernel-Power 41）与 WER 1001 BlueScreen 报告，"
            "且终止代码与客户报告一致，确认蓝屏事件本身真实发生。"
            "触发组件仍需结合崩溃模块、驱动版本与复现条件定位。"
        )
        missing = [
            "MEMORY.DMP 符号化调用栈或崩溃驱动模块",
            "触发时正在运行的应用/驱动及版本",
            "正常设备对照与稳定复现步骤",
        ]
    else:
        label = "Windows 蓝屏重启（终止代码待与现场确认对齐）"
        mechanism = (
            "EVTX 记录非正常关机（Kernel-Power 41）与 WER 1001 BlueScreen 报告，"
            "蓝屏重启事件真实发生；终止代码与触发组件待定位。"
        )
        missing = [
            "客户报告的终止代码原文",
            "MEMORY.DMP 符号化调用栈或崩溃驱动模块",
            "触发时正在运行的应用/驱动及版本",
        ]
    if aligned and "0x000000EF" in aligned:
        suspected_component = "network_driver_or_system_component"
        mechanism += " CRITICAL_PROCESS_DIED 通常指向关键系统进程被终止，需检查崩溃进程及其依赖驱动（如客户更换/插入的网卡驱动）。"
    else:
        suspected_component = "windows_kernel"
    return DiagnosticHypothesis(
        hypothesis_id="hypothesis:windows-blue-screen-bugcheck",
        label=label,
        failure_mechanism=mechanism,
        suspected_component=suspected_component,
        support_evidence_ids=support,
        contradict_evidence_ids=[],
        missing_evidence=missing,
        confidence=round(confidence, 4),
        status="observed_support",
        retrieval_score=0.0,
        source_ids=[],
    )


def _cross_source_hypothesis(
    events: list[DiagnosticEvent],
    correlations: list[dict[str, Any]],
    environment: EnvironmentSnapshot,
) -> DiagnosticHypothesis | None:
    """Fuse normalized event semantics, not query phrases or file names."""

    by_kind: dict[str, list[DiagnosticEvent]] = {}
    for event in events:
        by_kind.setdefault(event.event_kind, []).append(event)
    driver_kinds = {
        "gpu_driver_exception", "display_driver_reset", "gpu_live_kernel_event",
    }
    driver_events = [event for kind in driver_kinds for event in by_kind.get(kind, [])]
    cuda_events = by_kind.get("illegal_memory_access", []) + by_kind.get("device_lost", [])
    dump_events = by_kind.get("crash_dump_exception", [])
    if not driver_events or not cuda_events:
        return None
    artifact_ids = {event.artifact_id for event in [*driver_events, *cuda_events, *dump_events]}
    observed_kinds = {event.event_kind for event in [*driver_events, *cuda_events, *dump_events]}
    if len(artifact_ids) < 2 or len(observed_kinds) < 2:
        return None
    support = _dedupe(
        evidence_id
        for event in [*driver_events, *cuda_events, *dump_events]
        for evidence_id in event.evidence_ids
    )[:64]
    recurrent = any(
        item.get("type") == "repeated_failure_signature"
        for item in correlations
    )
    has_reset = bool(by_kind.get("display_driver_reset"))
    has_kernel_report = bool(by_kind.get("gpu_live_kernel_event"))
    has_dump = bool(dump_events)
    confidence = 0.54
    confidence += 0.06 if has_reset else 0.0
    confidence += 0.06 if has_kernel_report else 0.0
    confidence += 0.04 if has_dump else 0.0
    confidence += 0.04 if recurrent else 0.0
    confidence = min(confidence, 0.78)
    versions = environment.values
    missing = [
        "区分驱动缺陷与 GPU 硬件/供电/PCIe 不稳定的单变量对照结果",
        "Windows WATCHDOG 内核转储的符号化分析结果",
        "修复后在相同负载和复现条件下的长时间验证记录",
    ]
    if not versions.get("nvidia_driver_version"):
        missing.insert(0, "NVIDIA 显示驱动完整版本")
    return DiagnosticHypothesis(
        hypothesis_id="hypothesis:cross-source-gpu-driver-reset-chain",
        label="GPU/显示驱动异常触发 CUDA 失败与应用退出",
        failure_mechanism=(
            "同一参考时刻同时出现 GPU 驱动图形异常、CUDA 非法内存访问"
            + ("、显示驱动超时恢复" if has_reset else "")
            + ("、LiveKernelEvent/WATCHDOG" if has_kernel_report else "")
            + ("和应用 fatal-exit 转储" if has_dump else "")
            + "。这将故障域收敛到 GPU/显示驱动执行链，但仍不能仅凭这些证据区分驱动软件缺陷、GPU 硬件、供电或 PCIe 稳定性问题。"
        ),
        suspected_component="gpu_driver",
        support_evidence_ids=support,
        contradict_evidence_ids=[],
        missing_evidence=missing,
        confidence=round(confidence, 4),
        status="supported",
        retrieval_score=0.0,
        source_ids=[],
    )


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


__all__ = ["HypothesisRuntime"]
