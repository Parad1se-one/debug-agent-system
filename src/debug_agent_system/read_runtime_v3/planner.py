"""Evidence-first bootstrap planner for Read Runtime v3.

The deterministic planner is the safe shadow-mode bootstrap.  Its output is
the same Answer Plan contract that a future Codex planner must emit, so the
policy and verifier do not depend on which planner produced it.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .contracts import (
    AnswerClaim,
    AnswerPlan,
    AnswerPlanSection,
    HypothesisRecord,
    ReadTask,
)
from .fabric import EvidenceFabric


class EvidenceFirstPlanner:
    name = "evidence_first_bootstrap"

    def build(
        self,
        *,
        task: ReadTask,
        fabric: EvidenceFabric,
        baseline_result: dict[str, Any],
        kg_result: dict[str, Any] | None,
        incident_result: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> AnswerPlan:
        baseline = dict(baseline_result.get("response") or {})
        answer_evidence_id = str(baseline_result.get("answer_evidence_id") or "")
        baseline_answer = str(baseline.get("answer") or "").strip()
        sections: list[AnswerPlanSection] = []
        if baseline_answer and answer_evidence_id:
            sections.append(AnswerPlanSection(
                section_id="baseline_answer",
                title="现有读侧回答",
                section_type="baseline_answer",
                claims=[AnswerClaim(
                    claim_id=_id("baseline", baseline_answer),
                    text=baseline_answer,
                    evidence_ids=[answer_evidence_id],
                    assertion="derived",
                    confidence=float(baseline.get("confidence") or 0.0),
                )],
                items=[baseline_answer],
                evidence_ids=[answer_evidence_id],
            ))

        hypotheses: list[HypothesisRecord] = []
        unresolved: list[str] = [str(item) for item in baseline.get("required_data") or []]
        incident_payload = dict(incident_result or {})
        if incident_payload and not incident_payload.get("skipped"):
            result = dict(incident_payload.get("result") or {})
            event_ids = list(incident_payload.get("event_evidence_ids") or [])
            events = list(result.get("events") or [])
            if events and event_ids:
                claims = []
                for event, evidence_id in zip(events[:12], event_ids[:12]):
                    claims.append(AnswerClaim(
                        claim_id=_id("event", str(event.get("event_id") or evidence_id)),
                        text=str(event.get("message") or ""),
                        evidence_ids=[evidence_id],
                        assertion="observed",
                        confidence=1.0,
                    ))
                sections.append(AnswerPlanSection(
                    section_id="incident_observations",
                    title="诊断数据中的直接观测",
                    section_type="observations",
                    claims=claims,
                    items=[claim.text for claim in claims],
                    evidence_ids=[item for claim in claims for item in claim.evidence_ids],
                ))
            hypotheses = list(incident_payload.get("hypotheses") or [])
            for hypothesis in hypotheses:
                unresolved.extend(hypothesis.missing_evidence)
            if hypotheses:
                claims = [
                    AnswerClaim(
                        claim_id=_id("hypothesis", hypothesis.hypothesis_id),
                        text=(
                            f"{hypothesis.label}：{hypothesis.mechanism}"
                            f"（状态：{hypothesis.state}，置信度：{hypothesis.confidence:.2f}）"
                        ),
                        evidence_ids=list(hypothesis.support_evidence_ids),
                        assertion="inferred",
                        confidence=hypothesis.confidence,
                    )
                    for hypothesis in hypotheses
                    if hypothesis.support_evidence_ids
                ]
                if claims:
                    sections.append(AnswerPlanSection(
                        section_id="incident_hypotheses",
                        title="综合判断与候选假设",
                        section_type="hypotheses",
                        claims=claims,
                        items=[claim.text for claim in claims],
                        evidence_ids=[item for claim in claims for item in claim.evidence_ids],
                    ))
            next_tests = list(result.get("next_tests") or [])
            if next_tests:
                report_evidence_id = str(incident_payload.get("report_evidence_id") or "")
                sections.append(AnswerPlanSection(
                    section_id="incident_next_tests",
                    title="下一步验证",
                    section_type="next_tests",
                    claims=[AnswerClaim(
                        claim_id=_id("test", str(item.get("test_id") or index)),
                        text=(
                            f"{item.get('title') or '验证'}：{item.get('instruction') or ''}"
                        ),
                        evidence_ids=[report_evidence_id] if report_evidence_id else [],
                        assertion="derived",
                        confidence=float(item.get("information_gain") or 0.0),
                    ) for index, item in enumerate(next_tests[:10], start=1)],
                    items=[
                        f"{item.get('title') or '验证'}：{item.get('instruction') or ''}"
                        for item in next_tests[:10]
                    ],
                    evidence_ids=[report_evidence_id] if report_evidence_id else [],
                    risk=(
                        "destructive" if any(item.get("risk") == "destructive" for item in next_tests)
                        else "controlled" if any(item.get("risk") == "controlled" for item in next_tests)
                        else "safe"
                    ),
                ))

        unresolved = list(dict.fromkeys(item for item in unresolved if item.strip()))
        if unresolved:
            sections.append(AnswerPlanSection(
                section_id="evidence_gaps",
                title="仍需补充的证据",
                section_type="evidence_gaps",
                items=unresolved,
                evidence_ids=[],
                status="omitted_evidence_gap",
            ))
        proposed = str(baseline.get("status") or "failed")
        return AnswerPlan(
            task=task,
            sections=sections,
            hypotheses=hypotheses,
            unresolved_gaps=unresolved,
            baseline_status=str(baseline.get("status") or "failed"),
            proposed_status=proposed,
        )


def render_answer(plan: AnswerPlan) -> str:
    lines: list[str] = []
    for section in plan.sections:
        if section.section_type == "baseline_answer":
            lines.extend(section.items)
            continue
        lines.extend(["", f"## {section.title}", ""])
        rendered_items = (
            [claim.text for claim in section.claims]
            if section.claims
            else section.items if section.section_type == "evidence_gaps"
            else []
        )
        for item in rendered_items:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def _id(prefix: str, value: str) -> str:
    return f"{prefix}:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
