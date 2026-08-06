"""Fail-closed policy decisions for Read Runtime v3."""

from __future__ import annotations

from typing import Any

from .contracts import AnswerPlan, PolicyDecision, ReadRequest


_STATUS_RANK = {
    "failed": 0,
    "ask_info": 1,
    "step": 2,
    "escalate": 2,
    "resolved": 3,
}


class ReadPolicyEngine:
    """Separate answerability, diagnosis and execution authority."""

    def decide(
        self,
        *,
        request: ReadRequest,
        baseline: dict[str, Any],
        plan: AnswerPlan,
        shadow_mode: bool,
    ) -> PolicyDecision:
        status = str(baseline.get("status") or "failed")
        answer = str(baseline.get("answer") or "").strip()
        family_id = str(baseline.get("family_id") or "")
        variant_id = str(baseline.get("variant_id") or "")
        evidence_ids = [str(item) for item in baseline.get("evidence_ids") or []]
        metadata = dict(baseline.get("metadata") or {})
        sufficiency = dict(metadata.get("sufficiency") or {})

        answerable = bool(answer and status != "failed")
        if "answerable" in sufficiency:
            answerable = bool(sufficiency["answerable"])
        diagnosable = bool(family_id and variant_id)
        if "diagnosable" in sufficiency:
            diagnosable = bool(sufficiency["diagnosable"])
        executable = bool(diagnosable and status in {"step", "resolved"})
        if "executable" in sufficiency:
            executable = bool(sufficiency["executable"])
        verified_fix = bool(
            status == "resolved"
            and variant_id
            and evidence_ids
            and str(baseline.get("resolution") or "").strip()
        )

        reasons: list[str] = []
        proposed_status = str(plan.proposed_status or status or "failed")
        if _STATUS_RANK.get(proposed_status, 0) > _STATUS_RANK.get(status, 0):
            reasons.append("v3_may_not_upgrade_frozen_runtime_status_without_policy_evidence")
            proposed_status = status
        if any(item.state == "locked_root_cause" for item in plan.hypotheses) and not diagnosable:
            reasons.append("root_cause_lock_rejected_without_frozen_variant_lock")
            for hypothesis in plan.hypotheses:
                if hypothesis.state == "locked_root_cause":
                    hypothesis.state = "needs_evidence"
        if any(item.state == "verified_fix" for item in plan.hypotheses) and not verified_fix:
            reasons.append("verified_fix_rejected_without_frozen_closure")
            for hypothesis in plan.hypotheses:
                if hypothesis.state == "verified_fix":
                    hypothesis.state = "needs_evidence"

        blocked_actions: list[dict[str, Any]] = []
        allow_destructive = bool(request.controls.get("allow_destructive_actions", False))
        for section in plan.sections:
            if section.risk == "destructive" and not allow_destructive:
                blocked_actions.append({
                    "section_id": section.section_id,
                    "reason": "destructive_action_requires_explicit_authority",
                })
                section.status = "risk_controlled"

        return PolicyDecision(
            answerable=answerable,
            diagnosable=diagnosable,
            executable=executable,
            verified_fix=verified_fix,
            official_status=status,
            proposed_status=proposed_status,
            active_answer_source="frozen_read_pipeline" if shadow_mode else "read_runtime_v3",
            reasons=reasons,
            blocked_actions=blocked_actions,
        )

