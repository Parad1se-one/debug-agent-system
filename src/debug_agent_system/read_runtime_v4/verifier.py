from __future__ import annotations

from .contracts import V4AnswerPlan, V4Verification
from debug_agent_system.read_runtime_v3.fabric import EvidenceFabric


class InvestigationVerifier:
    def verify(self, plan: V4AnswerPlan, fabric: EvidenceFabric) -> V4Verification:
        known = {record.evidence_id for record in fabric.records()}
        errors: list[str] = []
        warnings: list[str] = []
        checked_facts = 0
        for fact in (plan.state.facts if plan.state else []):
            checked_facts += 1
            missing = [item for item in fact.evidence_ids if item not in known]
            if missing:
                errors.append(f"fact_unknown_evidence:{fact.fact_id}:{','.join(missing)}")
        for section in plan.sections:
            missing = [item for item in section.evidence_ids if item and item not in known]
            if missing:
                errors.append(f"section_unknown_evidence:{section.section_id}:{','.join(missing)}")
            if section.risk == "destructive" and section.status != "risk_controlled":
                errors.append(f"destructive_section_not_controlled:{section.section_id}")
        hypotheses = plan.state.hypotheses if plan.state else []
        for hypothesis in hypotheses:
            for evidence_id in [*hypothesis.support_evidence_ids, *hypothesis.contradict_evidence_ids]:
                if evidence_id not in known:
                    errors.append(f"hypothesis_unknown_evidence:{hypothesis.hypothesis_id}:{evidence_id}")
        actions = plan.state.next_tests if plan.state else []
        alternative_groups: dict[str, list[dict]] = {}
        for action in actions:
            action_id = str(action.get("test_id") or "")
            kind = str(action.get("kind") or "diagnosis")
            status = str(action.get("status") or "recommended")
            risk = str(action.get("risk") or "safe")
            if kind not in {"containment", "diagnosis", "remediation", "verification"}:
                errors.append(f"action_invalid_kind:{action_id}:{kind}")
            if status not in {"recommended", "conditional", "blocked", "omitted"}:
                errors.append(f"action_invalid_status:{action_id}:{status}")
            try:
                priority = int(action.get("priority"))
                if priority < 0:
                    errors.append(f"action_invalid_priority:{action_id}")
            except (TypeError, ValueError):
                errors.append(f"action_invalid_priority:{action_id}")
            missing = [item for item in action.get("evidence_ids") or [] if item not in known]
            if missing:
                errors.append(f"action_unknown_evidence:{action_id}:{','.join(missing)}")
            if risk == "destructive" and status != "blocked":
                warnings.append(f"destructive_action_not_blocked:{action_id}")
            group = str(action.get("plan_group") or "").strip()
            if group:
                alternative_groups.setdefault(group, []).append(action)
        for group, members in alternative_groups.items():
            expected = max(int(item.get("expected_alternatives") or 0) for item in members)
            if expected and len(members) < expected:
                warnings.append(
                    f"parallel_plan_incomplete:{group}:expected={expected}:actual={len(members)}"
                )
        if not plan.sections:
            warnings.append("answer_plan_has_no_sections")
        return V4Verification(
            passed=not errors,
            errors=errors,
            warnings=warnings,
            checked_facts=checked_facts,
            checked_hypotheses=len(hypotheses),
        )
