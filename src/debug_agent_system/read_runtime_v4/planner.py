from __future__ import annotations

import hashlib
import re
from typing import Any

from debug_agent_system.read_runtime_v3.contracts import HypothesisRecord
from debug_agent_system.read_runtime_v3.fabric import EvidenceFabric

from .contracts import (
    EvidenceGap,
    InvestigationFact,
    InvestigationState,
    InvestigationTask,
    V4AnswerPlan,
    V4AnswerSection,
)


class InvestigationPlanner:
    """Deterministic v4 planner.

    This is intentionally generic: it never recognizes a particular product,
    error code or Query.  It prioritizes scoped incident evidence, then uses
    baseline/KG material as reference evidence.  An agentic planner can emit
    the same state contract later without changing the renderer or verifier.
    """

    name = "deterministic_investigation"

    def build(
        self,
        *,
        task: InvestigationTask,
        fabric: EvidenceFabric,
        baseline_result: dict[str, Any] | None = None,
        kg_result: dict[str, Any] | None = None,
        incident_result: dict[str, Any] | None = None,
        raw_result: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> V4AnswerPlan:
        state = InvestigationState(task=task)
        incident = dict(incident_result or {})
        result = dict(incident.get("result") or {})
        if result:
            self._add_incident_state(state, result, incident, fabric)
        self._add_provider_gaps(state, result, incident, kg_result)
        if not result:
            # A procedure/evidence query already has a complete, source-backed
            # frozen answer.  It must not be reclassified as an observed
            # diagnostic fact (which used to produce an empty
            # “诊断数据中的直接观测” section).  Incident tasks may still use
            # the baseline as reference evidence while building their state.
            if task.output_contract == "incident_report":
                self._add_baseline_facts(state, baseline_result or {}, fabric)
        else:
            # Baseline is retained as reference evidence but not rendered into
            # an incident answer unless it has direct, scoped support.
            state.excluded_evidence.append({
                "provider": "frozen_read_pipeline",
                "reason": "incident_answer_prioritizes_scoped_case_evidence",
            })
        state.selected_evidence_ids = list(dict.fromkeys(
            item.evidence_id for item in fabric.records()
            if item.provider in {"incident_evidence_runtime", "raw_corpus", "kg_v2_sag"}
        ))
        sections = self._sections(state, baseline_result or {}, fabric)
        status = "ask_info"
        if not state.gaps and any(h.state in {"observed_support", "kg_supported"} for h in state.hypotheses):
            status = "step"
        return V4AnswerPlan(
            task=task,
            sections=sections,
            state=state,
            proposed_status=status,
            answerable=bool(
                state.facts
                or state.hypotheses
                or _baseline_answer(baseline_result or {})
            ),
            diagnosable=any(h.state == "locked_root_cause" for h in state.hypotheses),
            executable=False,
            verified_fix=any(h.state == "verified_fix" for h in state.hypotheses),
        )


    def _add_incident_state(
        self, state: InvestigationState, result: dict[str, Any], payload: dict[str, Any], fabric: EvidenceFabric
    ) -> None:
        event_ids = list(payload.get("event_evidence_ids") or [])
        events = list(result.get("events") or [])
        for index, event in enumerate(events):
            evidence_id = event_ids[index] if index < len(event_ids) else ""
            if not evidence_id:
                continue
            timestamp = str(event.get("timestamp_utc") or event.get("timestamp_raw") or "")
            text = str(event.get("message") or event.get("event_id") or "").strip()
            if not text:
                continue
            state.facts.append(InvestigationFact(
                fact_id=_id("fact", evidence_id), text=text, evidence_ids=[evidence_id],
                assertion="observed", relevance=_event_relevance(event),
                temporal_match=bool(timestamp), source_kind="diagnostic_event",
            ))
        for hypothesis in payload.get("hypotheses") or []:
            if isinstance(hypothesis, HypothesisRecord):
                state.hypotheses.append(hypothesis)
        report_evidence_id = str(payload.get("report_evidence_id") or "")
        if result:
            # This is a generic incident-safety action, not a diagnosis rule.
            # It keeps the evidence boundary intact when the caller cannot
            # provide any further context and does not change system state.
            state.next_tests.append(_normalize_action({
                "test_id": "action:preserve-incident-evidence",
                "kind": "containment",
                "title": "保全当前诊断证据",
                "instruction": (
                    "在继续操作或重复复现前，复制并校验当前日志、转储和诊断包，"
                    "保留原始文件、时间戳和哈希。"
                ),
                "preconditions": ["保留原始诊断包，不覆盖原文件"],
                "expected_observations": ["后续每次动作都能与变更前时间线对照"],
                "rollback": "不改变系统状态",
                "information_gain": 0.95,
                "risk": "safe",
                "cost": "low",
                "requires_confirmation": False,
                "source_ids": ["v4:incident_safety_policy"],
                "evidence_ids": [report_evidence_id] if report_evidence_id else [],
                "generated_by": "v4:incident_safety_policy",
            }, 0, fabric=fabric))
        for index, item in enumerate(result.get("next_tests") or [], start=1):
            if isinstance(item, dict):
                state.next_tests.append(_normalize_action(
                    item, index, fabric=fabric, default_evidence_id=report_evidence_id,
                ))
        state.next_tests.sort(key=_action_sort_key)
        for item in result.get("exclusions") or []:
            if isinstance(item, dict):
                state.excluded_evidence.append(dict(item))

    def _add_baseline_facts(self, state: InvestigationState, baseline: dict[str, Any], fabric: EvidenceFabric) -> None:
        answer_id = str(baseline.get("answer_evidence_id") or "")
        answer = _baseline_answer(baseline)
        if answer_id and fabric.get(answer_id) and answer:
            state.facts.append(InvestigationFact(
                fact_id=_id("fact", answer_id), text=answer,
                evidence_ids=[answer_id], assertion="derived", relevance=0.8,
                source_kind="baseline_answer",
            ))

    def _add_provider_gaps(
        self, state: InvestigationState, result: dict[str, Any], incident: dict[str, Any], kg_result: dict[str, Any] | None
    ) -> None:
        for item in result.get("required_evidence") or []:
            text = str(item).strip()
            if text:
                state.gaps.append(EvidenceGap(_id("gap", text), text, "diagnosis", "warning"))
        for hypothesis in state.hypotheses:
            for item in hypothesis.missing_evidence:
                state.gaps.append(EvidenceGap(_id("gap", item), item, hypothesis.hypothesis_id, "warning"))
        trace = dict((kg_result or {}).get("retrieval_trace") or {})
        if trace and trace.get("top_margin") is not None and float(trace.get("top_margin") or 0) < 0.1:
            state.gaps.append(EvidenceGap(
                _id("gap", "ambiguous_kg_retrieval"),
                "KG 候选区分度不足，不能把 Top1 当作当前案件根因",
                "diagnosis", "info", "kg_search_candidates",
            ))

    def _sections(self, state: InvestigationState, baseline: dict[str, Any], fabric: EvidenceFabric) -> list[V4AnswerSection]:
        sections: list[V4AnswerSection] = []
        if state.task.task.time_windows:
            sections.append(V4AnswerSection(
                "time_scope", "时间对齐", "time_scope",
                [
                    f"{item.get('source_text') or item.get('reference_time')}: "
                    f"{item.get('start_time')} 至 {item.get('end_time')}"
                    for item in state.task.task.time_windows
                ],
                [],
            ))
        if state.facts:
            ordered = sorted(state.facts, key=lambda item: (-item.relevance, not item.temporal_match))
            selected: list[InvestigationFact] = []
            seen: set[str] = set()
            for fact in ordered:
                key = _dedupe_key(fact.text)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(fact)
                if len(selected) >= 20:
                    break
            sections.append(V4AnswerSection(
                "observations", "诊断数据中的直接观测", "observations",
                [_compact_fact(item.text) for item in selected],
                [eid for item in selected for eid in item.evidence_ids],
            ))
        supported = [item for item in state.hypotheses if item.support_evidence_ids]
        if supported:
            items = [
                f"{item.label}：{item.mechanism}（状态：{item.state}，置信度：{item.confidence:.2f}）"
                for item in supported
            ]
            sections.append(V4AnswerSection(
                "hypotheses", "综合判断与候选假设", "hypotheses", items,
                [eid for item in supported for eid in item.support_evidence_ids],
            ))
        if state.next_tests:
            grouped = (
                ("containment", "建议立即采取", "containment"),
                ("diagnosis", "下一步验证", "next_tests"),
                ("remediation", "候选修复动作", "remediation"),
                ("verification", "修复后验证", "verification"),
            )
            for kind, title, section_type in grouped:
                actions = [
                    item for item in state.next_tests
                    if item.get("kind") == kind and item.get("risk") != "destructive"
                ]
                if not actions:
                    continue
                risk = _max_action_risk(actions)
                status = "risk_controlled" if risk in {"controlled", "destructive"} else "expanded"
                sections.append(V4AnswerSection(
                    f"actions_{kind}", title, section_type,
                    [_render_action(item) for item in actions[:10]],
                    [eid for item in actions for eid in item.get("evidence_ids") or []],
                    status, risk,
                ))
            destructive = [item for item in state.next_tests if item.get("risk") == "destructive"]
            if destructive:
                sections.append(V4AnswerSection(
                    "actions_blocked", "需人工确认的高风险动作", "blocked_actions",
                    [_render_action(item) for item in destructive[:10]],
                    [eid for item in destructive for eid in item.get("evidence_ids") or []],
                    "risk_controlled", "destructive",
                ))
        if state.gaps:
            unique = list(dict.fromkeys(item.description for item in state.gaps))
            sections.append(V4AnswerSection(
                "gaps", "仍需补充的证据", "evidence_gaps", unique,
                [], "omitted_evidence_gap",
            ))
        baseline_answer = _baseline_answer(baseline)
        if not state.facts and baseline_answer:
            sections.insert(0, V4AnswerSection(
                "reference", "根据资料可知", "reference",
                [baseline_answer],
                [str(baseline.get("answer_evidence_id") or "")] if baseline.get("answer_evidence_id") else [],
            ))
        return sections


class CodexInvestigationPlanner(InvestigationPlanner):
    """Optional Codex planner that emits the same v4 answer contract.

    The existing v3 read-only tool runner is reused as an adapter. Codex can
    search and expand evidence, but local v4 state and verification remain the
    authority. Any client/tool/schema failure falls back to the deterministic
    investigation plan.
    """

    name = "codex_investigation"

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.last_trace: list[dict[str, Any]] = []

    def build(self, *, request, task, tool_registry, **kwargs):
        deterministic = super().build(task=task, **kwargs)
        try:
            from debug_agent_system.read_runtime_v3.agentic import answer_plan_from_payload

            payload = self.runner.run(
                request=request,
                task=task.task,
                fabric=kwargs["fabric"],
                tools=tool_registry,
            )
            self.last_trace = list(getattr(self.runner, "last_trace", []))
            model_plan = answer_plan_from_payload(task.task, payload)
            if not model_plan.sections:
                return deterministic
            sections = [
                V4AnswerSection(
                    section_id=item.section_id,
                    title=item.title,
                    section_type=item.section_type,
                    items=[claim.text for claim in item.claims] or list(item.items),
                    evidence_ids=list(item.evidence_ids)
                    or [eid for claim in item.claims for eid in claim.evidence_ids],
                    status=item.status,
                    risk=item.risk,
                )
                for item in model_plan.sections
            ]
            state = deterministic.state
            if model_plan.hypotheses:
                state.hypotheses = model_plan.hypotheses
            state.gaps = [
                EvidenceGap(_id("gap", item), item, "diagnosis", "warning")
                for item in model_plan.unresolved_gaps
            ]
            state.planner_trace = self.last_trace
            return V4AnswerPlan(
                task=task,
                sections=sections,
                state=state,
                proposed_status=model_plan.proposed_status or deterministic.proposed_status,
                answerable=deterministic.answerable,
                diagnosable=deterministic.diagnosable,
                executable=False,
                verified_fix=False,
            )
        except Exception as exc:
            self.last_trace = [
                {"status": "fallback", "error": f"{type(exc).__name__}:{str(exc)[:240]}"}
            ]
            deterministic.state.planner_trace = self.last_trace
            return deterministic


def render_answer(plan: V4AnswerPlan) -> str:
    lines: list[str] = []
    for section in plan.sections:
        lines.extend([f"## {section.title}", ""])
        for item in section.items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _id(prefix: str, value: str) -> str:
    return f"{prefix}:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _baseline_answer(baseline: dict[str, Any]) -> str:
    """Read the frozen provider's nested response without flattening it.

    Provider adapters intentionally return metadata alongside ``response``;
    treating the wrapper as if it were the response silently dropped answers
    for non-incident tasks.  Keeping this accessor centralized prevents that
    class of contract mismatch in both planning and rendering.
    """
    response = baseline.get("response")
    if isinstance(response, dict):
        return str(response.get("answer") or "").strip()
    return str(baseline.get("answer") or "").strip()


def _event_relevance(event: dict[str, Any]) -> float:
    """Score how directly an event points at the reported incident.

    Kernel bugchecks, unexpected power loss and network-adapter churn are
    given priority over generic Windows error-reporting or update noise so
    the answer highlights the evidence most likely to explain the case.
    """

    text = " ".join(str(event.get(key) or "") for key in ("message", "event_id", "provider", "code")).lower()
    kind = str(event.get("event_kind") or "").lower()
    score = 0.5
    kind_bonus = {
        "windows_blue_screen": 0.55,
        "kernel_power_loss": 0.5,
        "gpu_live_kernel_event": 0.5,
        "network_driver": 0.4,
        "network_adapter": 0.35,
        "gpu_driver_exception": 0.45,
        "display_driver_reset": 0.45,
        "crash_dump_exception": 0.4,
        "hardware_error": 0.4,
    }
    if kind in kind_bonus:
        score += kind_bonus[kind]
    for token, weight in (
        ("bugcheck", 0.5), ("bluescreen", 0.5), ("kernel-power", 0.45),
        ("livekernelevent", 0.5), ("watchdog", 0.5), ("cuda", 0.45),
        ("e1rexpress", 0.4), ("e2fexpress", 0.4), ("ndis", 0.3),
        ("fatal", 0.35), ("crash", 0.3), ("critical process died", 0.5),
        ("network", 0.15), ("error", 0.15),
    ):
        if token in text:
            score += weight
    return min(1.0, score)


def _dedupe_key(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    # WER report/archive paths and GUIDs identify the same event family but
    # make otherwise identical observations appear as separate bullets.
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<guid>", text)
    text = re.sub(r"(?:reportqueue|reportarchive)\S+", "<wer-report>", text)
    return text


def _compact_fact(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"(?:\\\?\\)?[A-Za-z]:\\(?:[^ ]+\\){2,}[^ ]+", "<path>", text)
    if len(text) > 420:
        text = text[:417].rstrip() + "..."
    return text


_ACTION_KINDS = {"containment", "diagnosis", "remediation", "verification"}
_ACTION_STATUSES = {"recommended", "conditional", "blocked", "omitted"}
_RISK_ORDER = {"safe": 0, "controlled": 1, "destructive": 2}
_COST_ORDER = {"low": 0, "medium": 1, "high": 2}


def _normalize_action(
    value: dict[str, Any],
    index: int,
    *,
    fabric: EvidenceFabric,
    default_evidence_id: str = "",
) -> dict[str, Any]:
    """Normalize incident tests and planner actions into one v4-compatible record.

    This deliberately returns a dict instead of introducing a new top-level
    action class. Existing v3 DiagnosticTest payloads remain valid, while v4
    can still enforce ordering, provenance and risk semantics.
    """
    raw = dict(value)
    kind = _action_kind(raw)
    risk = str(raw.get("risk") or "safe").lower()
    if risk not in _RISK_ORDER:
        risk = "safe"
    status = str(raw.get("status") or "").lower()
    if status not in _ACTION_STATUSES:
        status = "blocked" if risk == "destructive" else "recommended"
    evidence_ids = [
        str(item) for item in raw.get("evidence_ids") or []
        if str(item) in {record.evidence_id for record in fabric.records()}
    ]
    if not evidence_ids and default_evidence_id and fabric.get(default_evidence_id):
        evidence_ids = [default_evidence_id]
    source_ids = [
        str(item) for item in raw.get("source_ids") or [] if str(item).strip()
    ]
    if raw.get("source_id"):
        source_ids.append(str(raw["source_id"]))
    priority = _coerce_int(raw.get("priority"), 100 + index)
    if kind == "containment":
        priority = min(priority, 10)
    elif kind == "diagnosis":
        priority = min(priority, 100 + index)
    elif kind == "remediation":
        priority = min(priority, 200 + index)
    elif kind == "verification":
        priority = min(priority, 300 + index)
    preconditions = _as_text_list(raw.get("preconditions"))
    expected = _as_text_list(raw.get("expected_observations"))
    evidence_required = _as_text_list(raw.get("evidence_required"))
    distinguishes = _as_text_list(raw.get("distinguishes_hypothesis_ids"))
    requires_confirmation = bool(
        raw.get("requires_confirmation", risk in {"controlled", "destructive"})
    )
    rollback = str(raw.get("rollback") or "").strip()
    if kind == "remediation" and not rollback:
        rollback = "未提供回滚步骤，需人工补充后再执行"
        status = "conditional" if status == "recommended" else status
    normalized = {
        "test_id": str(raw.get("test_id") or f"action:{kind}:{index}"),
        "kind": kind,
        "title": str(raw.get("title") or "验证").strip(),
        "instruction": str(raw.get("instruction") or "").strip(),
        "priority": priority,
        "status": status,
        "preconditions": preconditions,
        "expected_observations": expected,
        "rollback": rollback,
        "distinguishes_hypothesis_ids": distinguishes,
        "evidence_required": evidence_required,
        "information_gain": _coerce_float(raw.get("information_gain"), 0.0),
        "cost": str(raw.get("cost") or "low") if str(raw.get("cost") or "low") in _COST_ORDER else "low",
        "risk": risk,
        "requires_confirmation": requires_confirmation,
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "source_ids": list(dict.fromkeys(source_ids)),
        "generated_by": str(raw.get("generated_by") or "incident_evidence_runtime"),
        "plan_group": str(raw.get("plan_group") or raw.get("alternative_group") or "").strip(),
        "alternative_index": _coerce_int(raw.get("alternative_index"), 0),
        "expected_alternatives": _coerce_int(raw.get("expected_alternatives"), 0),
    }
    return normalized


def _action_kind(value: dict[str, Any]) -> str:
    explicit = str(value.get("kind") or value.get("action_kind") or value.get("category") or "").lower()
    aliases = {
        "contain": "containment", "containment_action": "containment",
        "test": "diagnosis", "diagnostic": "diagnosis",
        "fix": "remediation", "repair": "remediation",
        "verify": "verification", "validation": "verification",
    }
    kind = aliases.get(explicit, explicit)
    if kind in _ACTION_KINDS:
        return kind
    text = " ".join(str(value.get(key) or "") for key in ("title", "instruction")).lower()
    if any(token in text for token in ("保全", "暂停", "隔离", "避免覆盖")):
        return "containment"
    if any(token in text for token in ("修复后", "长时间验证", "验证是否", "回归验证")):
        return "verification"
    # Evidence-gathering instructions can mention a repair only to defer it
    # (for example, "不要先执行修复动作"). Classify by the operation's
    # intent, not by an incidental verb in a safety warning.
    if any(token in text for token in (
        "补齐证据", "只读采集", "证据", "对照", "区分", "符号化",
        "检查", "分析结果", "不要先执行修复", "先不要修复",
    )):
        return "diagnosis"
    if any(token in text for token in ("回退", "重装", "更换", "修复", "恢复")):
        return "remediation"
    return "diagnosis"


def _action_sort_key(value: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        _coerce_int(value.get("priority"), 1000),
        _RISK_ORDER.get(str(value.get("risk") or "safe"), 0),
        -round(_coerce_float(value.get("information_gain"), 0.0) * 1000),
        str(value.get("test_id") or ""),
    )


def _max_action_risk(actions: list[dict[str, Any]]) -> str:
    return max(
        (str(item.get("risk") or "safe") for item in actions),
        key=lambda item: _RISK_ORDER.get(item, 0),
        default="safe",
    )


def _render_action(value: dict[str, Any]) -> str:
    priority = _coerce_int(value.get("priority"), 0)
    status = str(value.get("status") or "recommended")
    risk = str(value.get("risk") or "safe")
    title = str(value.get("title") or "验证")
    instruction = str(value.get("instruction") or "")
    parts = [f"P{priority} {title}（{status}，风险：{risk}）"]
    if instruction:
        parts.append(instruction)
    preconditions = _as_text_list(value.get("preconditions"))
    if preconditions:
        parts.append("前置条件：" + "；".join(preconditions))
    expected = _as_text_list(value.get("expected_observations"))
    if expected:
        parts.append("预期观察：" + "；".join(expected))
    rollback = str(value.get("rollback") or "").strip()
    if rollback:
        parts.append("回滚：" + rollback)
    if value.get("requires_confirmation"):
        parts.append("需人工确认")
    return "；".join(parts)


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
