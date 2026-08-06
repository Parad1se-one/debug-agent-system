"""Deterministic release checks for a v3 Answer Plan."""

from __future__ import annotations

from .contracts import AnswerPlan, PolicyDecision, VerificationReport
from .fabric import EvidenceFabric


class AnswerPlanVerifier:
    def verify(
        self,
        *,
        plan: AnswerPlan,
        policy: PolicyDecision,
        fabric: EvidenceFabric,
    ) -> VerificationReport:
        errors: list[str] = []
        warnings: list[str] = []
        checked_claims = 0
        known = {record.evidence_id for record in fabric.records()}
        for section in plan.sections:
            missing_section = [item for item in section.evidence_ids if item not in known]
            if missing_section:
                errors.append(
                    f"section_unknown_evidence:{section.section_id}:{','.join(missing_section)}"
                )
            if section.risk == "destructive" and section.status != "risk_controlled":
                errors.append(f"destructive_section_not_controlled:{section.section_id}")
            for claim in section.claims:
                checked_claims += 1
                if not claim.evidence_ids:
                    errors.append(f"claim_without_evidence:{claim.claim_id}")
                    continue
                missing = [item for item in claim.evidence_ids if item not in known]
                if missing:
                    errors.append(
                        f"claim_unknown_evidence:{claim.claim_id}:{','.join(missing)}"
                    )
            if section.items and not section.claims and section.section_type != "evidence_gaps":
                warnings.append(f"section_items_without_claims:{section.section_id}")
        for hypothesis in plan.hypotheses:
            for evidence_id in [
                *hypothesis.support_evidence_ids,
                *hypothesis.contradict_evidence_ids,
            ]:
                if evidence_id not in known:
                    errors.append(
                        f"hypothesis_unknown_evidence:{hypothesis.hypothesis_id}:{evidence_id}"
                    )
            if hypothesis.state == "locked_root_cause" and not policy.diagnosable:
                errors.append(f"uncontrolled_root_cause_lock:{hypothesis.hypothesis_id}")
            if hypothesis.state == "verified_fix" and not policy.verified_fix:
                errors.append(f"uncontrolled_verified_fix:{hypothesis.hypothesis_id}")
        seen_trace_ids: set[str] = set()
        for trace in plan.traces:
            if not trace.trace_id:
                errors.append("trace_id_missing")
            elif trace.trace_id in seen_trace_ids:
                errors.append(f"trace_id_duplicate:{trace.trace_id}")
            seen_trace_ids.add(trace.trace_id)
            if not trace.evidence_ids:
                errors.append(f"trace_without_evidence:{trace.trace_id or 'missing'}")
            for evidence_id in trace.evidence_ids:
                if evidence_id not in known:
                    errors.append(
                        f"trace_unknown_evidence:{trace.trace_id or 'missing'}:{evidence_id}"
                    )
        return VerificationReport(
            passed=not errors,
            errors=errors,
            warnings=warnings,
            checked_claims=checked_claims,
            checked_hypotheses=len(plan.hypotheses),
        )
