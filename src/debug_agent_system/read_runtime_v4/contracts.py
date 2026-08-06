from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from debug_agent_system.read_runtime_v3.contracts import (
    EvidenceAssertion,
    HypothesisRecord,
    ReadTask,
    to_jsonable,
)


@dataclass(slots=True)
class InvestigationTask:
    task: ReadTask
    goal: str
    output_contract: str
    risk_scope: str = "safe"
    requested_sections: list[str] = field(default_factory=list)
    parser_hints: list[str] = field(default_factory=list)
    schema_version: str = "debug_agent_system.investigation_task.v4"


@dataclass(slots=True)
class InvestigationFact:
    fact_id: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    assertion: EvidenceAssertion = "observed"
    relevance: float = 1.0
    temporal_match: bool = False
    source_kind: str = ""


@dataclass(slots=True)
class EvidenceGap:
    gap_id: str
    description: str
    required_for: str = "diagnosis"
    severity: Literal["info", "warning", "blocking"] = "warning"
    suggested_tool: str = ""


@dataclass(slots=True)
class InvestigationState:
    task: InvestigationTask
    facts: list[InvestigationFact] = field(default_factory=list)
    hypotheses: list[HypothesisRecord] = field(default_factory=list)
    gaps: list[EvidenceGap] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    next_tests: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence_ids: list[str] = field(default_factory=list)
    excluded_evidence: list[dict[str, Any]] = field(default_factory=list)
    planner_trace: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "debug_agent_system.investigation_state.v4"


@dataclass(slots=True)
class V4AnswerSection:
    section_id: str
    title: str
    section_type: str
    items: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    status: Literal["expanded", "risk_controlled", "omitted_evidence_gap"] = "expanded"
    risk: Literal["safe", "controlled", "destructive"] = "safe"


@dataclass(slots=True)
class V4AnswerPlan:
    task: InvestigationTask
    sections: list[V4AnswerSection] = field(default_factory=list)
    state: InvestigationState | None = None
    proposed_status: str = "ask_info"
    answerable: bool = True
    diagnosable: bool = False
    executable: bool = False
    verified_fix: bool = False
    schema_version: str = "debug_agent_system.answer_plan.v4"


@dataclass(slots=True)
class V4Policy:
    status: str
    answerable: bool
    diagnosable: bool
    executable: bool
    verified_fix: bool
    blocked_actions: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    schema_version: str = "debug_agent_system.policy_decision.v4"


@dataclass(slots=True)
class V4Verification:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_facts: int = 0
    checked_hypotheses: int = 0
    schema_version: str = "debug_agent_system.answer_verification.v4"


@dataclass(slots=True)
class V4Response:
    query: str
    status: str
    answer: str
    task: InvestigationTask
    state: InvestigationState
    answer_plan: V4AnswerPlan
    policy: V4Policy
    verification: V4Verification
    evidence_snapshot: dict[str, Any]
    provider_results: dict[str, Any] = field(default_factory=dict)
    baseline_response: dict[str, Any] = field(default_factory=dict)
    shadow: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "debug_agent_system.read_response.v4"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
