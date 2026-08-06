"""O-EG: bounded, source-bound evidence gap completion."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from debug_agent_system.agents.tools.executor import ReadEvidenceToolExecutor
from debug_agent_system.core.contracts import (
    EvidenceGapResolution,
    EvidenceObservation,
    EvidenceResource,
)

_WORD = re.compile(r"[A-Za-z0-9_.:+-]{2,}|[\u4e00-\u9fff]{2,}")
_GENERIC = {
    "请补充",
    "当前",
    "现场",
    "信息",
    "情况",
    "明确",
    "提供",
    "完整",
    "问题",
    "故障",
}
_CATEGORY_SIGNALS = {
    "error": ("报错", "错误", "异常", "失败", "error", "exception", "bugcheck", "stop code", "错误码"),
    "version": ("版本", "version", "固件", "firmware", "sdk"),
    "network": ("网络", "网卡", "ip", "地址", "丢包", "ping", "socket", "tcp"),
    "camera": ("相机", "拍照", "camera", "capture"),
    "stage": ("阶段", "发生", "启动", "初始化", "检测", "复判", "startup", "phase"),
    "interface": ("接口", "usb", "m.2", "pci", "端口"),
    "system": ("系统", "windows", "蓝屏", "安全模式", "winre", "启动"),
    "hardware": ("硬件", "设备", "板卡", "电源", "网线"),
    "site": ("现场", "站点", "客户", "jira"),
    "log": ("日志", "log", "dlog", "evtx", "dump", "dmp"),
}


class EvidenceGapResolver:
    """Inspect supplied resources without crossing diagnostic safety gates."""

    schema_version = "debug_agent_system.evidence_gap_resolution.v1"

    def __init__(self, executor: ReadEvidenceToolExecutor | None = None) -> None:
        self.executor = executor or ReadEvidenceToolExecutor()

    def resolve(
        self,
        required_data: Iterable[str],
        resources: Iterable[EvidenceResource | dict[str, Any]],
        *,
        max_resources: int = 12,
        max_bytes: int = 65536,
        max_rounds: int = 2,
        processed_fingerprints: Iterable[str] = (),
    ) -> EvidenceGapResolution:
        questions = [str(item).strip() for item in required_data if str(item).strip()]
        resource_items = list(resources)[: max(0, int(max_resources))]
        processed = set(str(item) for item in processed_fingerprints if str(item))
        tool_results = []
        observations: list[EvidenceObservation] = []
        excluded: list[dict[str, Any]] = []
        if not questions:
            return EvidenceGapResolution(
                schema_version=self.schema_version,
                attempted=False,
                round_count=0,
                required_data_before=[],
                stop_reason="no_required_data",
            )
        if not resource_items:
            return EvidenceGapResolution(
                schema_version=self.schema_version,
                attempted=False,
                round_count=0,
                required_data_before=questions,
                unresolved_items=questions,
                stop_reason="no_evidence_resources",
            )

        # Local parsers complete their work in one bounded round.  The second
        # configured round is reserved for a model-controlled tool loop; it is
        # never used to repeat an identical local call.
        round_count = min(1, max(0, int(max_rounds)))
        for index, resource in enumerate(resource_items):
            result = self.executor.execute(
                resource,
                max_bytes=max_bytes,
                call_id=f"evidence-gap-{index + 1}",
            )
            if result.call_fingerprint in processed:
                excluded.append(
                    {
                        "resource_id": result.resource_id,
                        "reason": "duplicate_tool_call",
                        "call_fingerprint": result.call_fingerprint,
                    }
                )
                continue
            processed.add(result.call_fingerprint)
            tool_results.append(result)
            observations.extend(result.observations)
            excluded.extend(result.excluded)

        resolved: list[str] = []
        unresolved: list[str] = []
        relevant_observation_ids: set[str] = set()
        for question in questions:
            matched = [
                observation
                for observation in observations
                if _observation_answers(question, observation)
            ]
            if matched:
                resolved.append(question)
                relevant_observation_ids.update(
                    observation.observation_id for observation in matched
                )
            else:
                unresolved.append(question)

        context_observations = [
            observation
            for observation in observations
            if observation.supports_retrieval
            and (
                observation.observation_id in relevant_observation_ids
                or any(
                    _observation_answers(question, observation)
                    for question in questions
                )
            )
        ]
        retrieval_context = _render_retrieval_context(context_observations)
        if not tool_results:
            stop_reason = "all_tool_calls_deduplicated"
        elif not observations:
            stop_reason = "no_parser_observations"
        elif not retrieval_context:
            stop_reason = "observations_not_relevant_to_required_data"
        elif unresolved:
            stop_reason = "partial_evidence_completion"
        else:
            stop_reason = "required_data_supported_by_evidence"
        return EvidenceGapResolution(
            schema_version=self.schema_version,
            attempted=True,
            round_count=round_count,
            required_data_before=questions,
            resolved_items=resolved,
            unresolved_items=unresolved,
            observations=observations,
            tool_results=tool_results,
            retrieval_context=retrieval_context,
            excluded=excluded,
            stop_reason=stop_reason,
        )


def _observation_answers(
    question: str,
    observation: EvidenceObservation,
) -> bool:
    question_text = str(question or "").lower()
    observation_text = " ".join(
        [
            observation.field,
            json.dumps(observation.value, ensure_ascii=False, default=str),
        ]
    ).lower()
    question_categories = _categories(question_text)
    observation_categories = _categories(observation_text)
    if question_categories and question_categories & observation_categories:
        return True
    question_tokens = _tokens(question_text)
    observation_tokens = _tokens(observation_text)
    return bool(question_tokens and question_tokens & observation_tokens)


def _categories(text: str) -> set[str]:
    lowered = str(text or "").lower()
    return {
        category
        for category, signals in _CATEGORY_SIGNALS.items()
        if any(signal.lower() in lowered for signal in signals)
    }


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD.findall(str(text or ""))
        if token not in _GENERIC and len(token) >= 2
    }


def _render_retrieval_context(
    observations: list[EvidenceObservation],
    *,
    max_chars: int = 6000,
) -> str:
    lines: list[str] = []
    for observation in observations:
        value = json.dumps(
            observation.value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        source = ",".join(observation.source_ids)
        lines.append(
            f"[工具证据 source={source} field={observation.field}] {value}"
        )
    return "\n".join(lines)[:max_chars]


__all__ = ["EvidenceGapResolver"]
