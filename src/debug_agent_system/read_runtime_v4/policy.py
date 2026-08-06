from __future__ import annotations

from typing import Any

from .contracts import V4AnswerPlan, V4Policy


class InvestigationPolicy:
    """Policy outside the model; answers and actions have separate authority."""

    def decide(self, plan: V4AnswerPlan, baseline: dict[str, Any] | None = None) -> V4Policy:
        baseline = baseline or {}
        reasons: list[str] = []
        blocked: list[dict[str, Any]] = []
        status = "ask_info"
        if plan.proposed_status == "step" and plan.diagnosable:
            status = "step"
        if plan.verified_fix:
            # v4 cannot promote a fix without explicit frozen closure.
            reasons.append("v4_verified_fix_requires_external_verification_closure")
        if any(section.risk == "destructive" for section in plan.sections):
            blocked.append({
                "section_id": next(section.section_id for section in plan.sections if section.risk == "destructive"),
                "reason": "destructive_action_requires_explicit_authority",
            })
        for action in (plan.state.next_tests if plan.state else []):
            if str(action.get("risk") or "safe") == "destructive":
                blocked.append({
                    "action_id": str(action.get("test_id") or ""),
                    "reason": "destructive_action_requires_explicit_authority",
                })
            elif bool(action.get("requires_confirmation")):
                reasons.append(
                    f"action_requires_confirmation:{str(action.get('test_id') or '')}"
                )
        return V4Policy(
            status=status,
            answerable=plan.answerable,
            diagnosable=plan.diagnosable,
            executable=False,
            verified_fix=False,
            blocked_actions=blocked,
            reasons=reasons,
        )
