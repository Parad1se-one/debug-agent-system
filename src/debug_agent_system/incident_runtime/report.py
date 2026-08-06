"""Deterministic, Jira-friendly incident report and local verification."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .contracts import (
    DiagnosticHypothesis,
    DiagnosticTest,
    IncidentCase,
)
from .parsers import ParsedDiagnostics


def render_incident_report(
    case: IncidentCase,
    parsed: ParsedDiagnostics,
    timeline: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    hypotheses: list[DiagnosticHypothesis],
    next_tests: list[DiagnosticTest],
    exclusions: list[dict[str, Any]],
) -> str:
    available = [item for item in case.artifacts if item.status == "available"]
    unavailable = [item for item in case.artifacts if item.status != "available"]
    lines = ["# Jira 诊断分析", "", "## 问题摘要", ""]
    lines.append(f"- Query：{_query_summary(case.query)}")
    if case.jira_key:
        lines.append(f"- Jira：{case.jira_key}")
    if case.status:
        lines.append(f"- Jira 状态：{case.status}（状态不作为修复已验证的证据）")
    incident_scope = case.metadata.get("incident_scope") or {}
    windows = incident_scope.get("reference_windows") or []
    if windows:
        lines.append(
            "- 参考时间窗："
            + "；".join(
                f"{item.get('start_time')}—{item.get('end_time')}"
                for item in windows
                if isinstance(item, dict)
            )
            + "（按独立时间点处理）"
        )
    lines.extend([
        f"- 已接收资料：{len(available)} 项；不可用或被拒绝：{len(unavailable)} 项。",
        f"- 已提取诊断事件：{len(parsed.events)} 条；调用栈：{len(parsed.stack_traces)} 组。",
        "",
        "## 诊断资料与解析边界",
        "",
    ])
    visible_artifacts = [
        artifact
        for artifact in case.artifacts
        if not artifact.metadata.get("scope_skipped")
    ]
    skipped_artifact_count = len(case.artifacts) - len(visible_artifacts)
    for artifact in visible_artifacts[:40]:
        location = f"（压缩包成员：{artifact.archive_member}）" if artifact.archive_member else ""
        lines.append(
            f"- `{artifact.name}`{location}：{artifact.status}/{artifact.parser_state}，"
            f"SHA256 `{artifact.sha256 or '未取得'}`。"
        )
    if len(visible_artifacts) > 40:
        lines.append(f"- 其余已纳入资料 {len(visible_artifacts) - 40} 项，详见 ArtifactManifest。")
    if skipped_artifact_count:
        lines.append(
            f"- 另有 {skipped_artifact_count} 个压缩包成员因不属于参考时间窗，"
            "仅保留元数据而未解析正文。"
        )
    if exclusions:
        lines.append("")
        lines.append("未解析或受限内容：")
        for item in exclusions[:20]:
            lines.append(f"- {item.get('material') or item.get('artifact_id') or '资料'}：{item.get('reason') or '未说明'}")

    lines.extend(["", "## 关键时间线与错误签名", ""])
    if not timeline:
        lines.append("- 当前资料没有形成可排序的诊断事件时间线。")
    critical_timeline = _critical_timeline(timeline, correlations)
    for item in critical_timeline[:30]:
        when = item.get("timestamp") or "时间未知"
        codes = ", ".join(item.get("error_codes") or [])
        suffix = f"；错误码：`{codes}`" if codes else ""
        repeats = f"；重复 {item.get('repeat_count')} 次" if int(item.get("repeat_count") or 1) > 1 else ""
        evidence = ", ".join(item.get("evidence_ids") or [])
        lines.append(f"- {when} [{item.get('severity')}] {item.get('message')}{suffix}{repeats}【证据：{evidence}】")
    omitted_timeline = len(timeline) - len(critical_timeline[:30])
    if omitted_timeline > 0:
        lines.append(
            f"- 另有 {omitted_timeline} 条低优先级或重复上下文事件未在正文展开，"
            "仍保留在结构化时间线中。"
        )
    if correlations:
        lines.extend(["", "时间关联："])
        for item in correlations[:20]:
            evidence = ", ".join(item.get("evidence_ids") or [])
            lines.append(
                f"- {item.get('interpretation')}"
                + (f" 间隔 {item.get('delta_seconds')} 秒。" if item.get("delta_seconds") is not None else "")
                + (f"【证据：{evidence}】" if evidence else "")
            )

    lines.extend(["", "## 当前定位", ""])
    if parsed.stack_traces:
        first = parsed.stack_traces[0]
        detection = next((frame for frame in first.frames if frame.function), first.frames[0] if first.frames else None)
        if detection:
            lines.append(
                f"- 异常检测点：`{detection.function or detection.module or detection.raw}`。"
                "该位置表示异常被观察到的位置，不自动等同于根因位置。"
            )
    else:
        lines.append(
            "- 当前没有完整调用栈，不能可靠定位异常检测点；即使后续取得检测点，"
            "异常被观察到的位置，不自动等同于根因位置。"
        )
    lines.append("- 当前结论属于候选定位；只有复现、修复和验证条件同时闭环后才能标记为已验证根因。")

    lines.extend(["", "## 候选假设与证据", ""])
    for index, hypothesis in enumerate(hypotheses, start=1):
        lines.extend([
            f"### 假设 {index}：{hypothesis.label}", "",
            f"- 状态：`{hypothesis.status}`；置信度：{hypothesis.confidence:.2f}。",
            f"- 故障机制：{hypothesis.failure_mechanism}",
            f"- 疑似组件：{hypothesis.suspected_component}",
            "- 支持证据：" + (", ".join(hypothesis.support_evidence_ids) or "无直接案件证据，仅为检索候选"),
            "- 反证：" + (", ".join(hypothesis.contradict_evidence_ids) or "尚未取得"),
            "- 缺失证据：" + ("；".join(hypothesis.missing_evidence) or "无明确缺口，但仍需修复后验证"),
            "- KG 来源：" + (", ".join(hypothesis.source_ids) or "未取得正式 EvidenceItem"),
            "",
        ])

    lines.extend(["## 建议的下一步检查", ""])
    if not next_tests:
        lines.append("- 当前没有可安全自动规划的检查项，请由人工结合现场环境补充诊断计划。")
    for index, test in enumerate(next_tests, start=1):
        lines.extend([
            f"### {index}. {test.title}", "",
            f"- 操作：{test.instruction}",
            f"- 风险/成本：`{test.risk}` / `{test.cost}`。",
            f"- 预期观察：{'；'.join(test.expected_observations)}",
            "",
        ])

    lines.extend([
        "## 验证与关闭条件", "",
        "- 在同一复现条件下确认问题不再出现。",
        "- 修复后保留不少于一个完整运行周期的日志，并与修复前时间线对照。",
        "- 由现场人员确认业务功能恢复；Jira 状态变更本身不等同于验证证据。",
        "- 若只能暂时恢复或缺少复现验证，应保持 `待验证`，不能声明根因已闭环。",
        "",
        "## 来源说明", "",
        "- 事实来自带哈希、文件名及行号/字节范围的案件证据；KG 内容只作为候选故障、检查和验证知识。",
        "- 模型或检索候选不得替代原始诊断数据，也不得自行确认高风险动作。",
    ])
    return "\n".join(lines).strip()


def _query_summary(value: str) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= 800:
        return compact
    return compact[:800] + "…（完整输入保留在案件快照）"


def _critical_timeline(
    timeline: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    correlated_ids = {
        str(value)
        for item in correlations
        for key in ("failure_event_id", "process_start_event_id")
        if (value := item.get(key))
    }
    important_kinds = {
        "illegal_memory_access", "access_violation", "device_lost",
        "process_start", "process_exit", "crash",
    }
    selected = [
        item
        for item in timeline
        if item.get("event_id") in correlated_ids
        or item.get("event_kind") in important_kinds
        or bool(item.get("error_codes"))
        or str(item.get("severity") or "").upper() in {
            "FATAL", "CRITICAL", "ERROR", "ERR", "EXCEPTION", "PANIC"
        }
    ]
    return selected or timeline[:30]


class IncidentVerifier:
    schema_version = "debug_agent_system.incident_verification.v1"

    def verify(
        self,
        parsed: ParsedDiagnostics,
        hypotheses: list[DiagnosticHypothesis],
        report: str,
    ) -> dict[str, Any]:
        known_evidence = {item.evidence_id for item in parsed.evidence_links}
        errors: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            unknown = [
                item for item in [*hypothesis.support_evidence_ids, *hypothesis.contradict_evidence_ids]
                if item not in known_evidence
            ]
            if unknown:
                errors.append({
                    "code": "unknown_hypothesis_evidence",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "evidence_ids": unknown,
                })
            if hypothesis.status == "locked" and not hypothesis.support_evidence_ids:
                errors.append({"code": "locked_without_case_evidence", "hypothesis_id": hypothesis.hypothesis_id})
            if hypothesis.confidence > 0.8 and hypothesis.status != "locked":
                errors.append({"code": "unlocked_confidence_too_high", "hypothesis_id": hypothesis.hypothesis_id})
        if "异常被观察到的位置，不自动等同于根因位置" not in report:
            errors.append({"code": "detection_root_cause_boundary_missing"})
        if "- Jira 状态：" in report and "不作为修复已验证的证据" not in report:
            errors.append({"code": "jira_status_verification_boundary_missing"})
        return {
            "schema_version": self.schema_version,
            "passed": not errors,
            "errors": errors,
            "claim_policy": {
                "facts_require_evidence_link": True,
                "detection_point_is_root_cause": False,
                "jira_status_is_verified_fix": False,
                "canonical_kg_mutated": False,
            },
        }


__all__ = ["IncidentVerifier", "render_incident_report"]
