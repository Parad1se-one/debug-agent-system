"""Stable contracts for the evidence-orchestrated Read Runtime v3.

The v3 package is intentionally additive.  It consumes frozen read-side
outputs but does not alter their contracts or execution semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


EvidenceKind = Literal[
    "source_artifact",
    "document_chunk",
    "kg_object",
    "kg_edge",
    "diagnostic_event",
    "stack_trace",
    "environment_fact",
    "media_asset",
    "runtime_decision",
    "answer_fragment",
    "exclusion",
]
EvidenceAssertion = Literal["observed", "source_asserted", "derived", "inferred"]
EvidenceRelation = Literal[
    "supports",
    "contradicts",
    "derived_from",
    "same_as",
    "contains",
    "temporal_neighbor",
]
HypothesisState = Literal[
    "candidate",
    "observed_support",
    "kg_supported",
    "needs_evidence",
    "contradicted",
    "locked_root_cause",
    "verified_fix",
]
RiskLevel = Literal["safe", "controlled", "destructive"]


@dataclass(slots=True)
class ReadRequest:
    query: str
    interactive: bool = False
    session: dict[str, Any] = field(default_factory=dict)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    log_summary: dict[str, Any] = field(default_factory=dict)
    routing_context: dict[str, Any] = field(default_factory=dict)
    evidence_resources: list[dict[str, Any]] = field(default_factory=list)
    controls: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "debug_agent_system.read_request.v3"

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | "ReadRequest") -> "ReadRequest":
        if isinstance(payload, cls):
            return payload
        return cls(
            query=str(payload.get("query") or payload.get("original_query") or ""),
            interactive=bool(payload.get("interactive", False)),
            session=dict(payload.get("session") or {}),
            chat_history=list(payload.get("chat_history") or []),
            log_summary=dict(payload.get("log_summary") or {}),
            routing_context=dict(payload.get("routing_context") or {}),
            evidence_resources=list(payload.get("evidence_resources") or []),
            controls=dict(payload.get("controls") or {}),
        )

    def to_baseline_payload(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "interactive": self.interactive,
            "session": dict(self.session),
            "chat_history": list(self.chat_history),
            "log_summary": dict(self.log_summary),
            "routing_context": dict(self.routing_context),
            "evidence_resources": list(self.evidence_resources),
        }


@dataclass(slots=True)
class ReadTask:
    query: str
    mode: str
    request_kind: str
    facets: list[str] = field(default_factory=list)
    facet_details: list[dict[str, Any]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    time_windows: list[dict[str, Any]] = field(default_factory=list)
    resource_ids: list[str] = field(default_factory=list)
    complexity: Literal["simple", "standard", "multi_source", "incident"] = "standard"
    budgets: dict[str, int] = field(default_factory=dict)
    normalization_trace: list[str] = field(default_factory=list)
    schema_version: str = "debug_agent_system.read_task.v3"


@dataclass(slots=True)
class SourceAnchor:
    path: str = ""
    source_id: str = ""
    line_start: int | None = None
    line_end: int | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    timestamp: str = ""
    object_id: str = ""
    chunk_id: str = ""
    artifact_id: str = ""


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    kind: EvidenceKind
    provider: str
    source_ref: str
    assertion: EvidenceAssertion
    summary: str
    content: Any = None
    content_sha256: str = ""
    anchors: list[SourceAnchor] = field(default_factory=list)
    confidence: float = 1.0
    source_revision: str = ""
    parser_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "debug_agent_system.evidence_record.v3"


@dataclass(slots=True)
class EvidenceLink:
    link_id: str
    relation: EvidenceRelation
    from_evidence_id: str
    to_evidence_id: str
    explanation: str = ""
    confidence: float = 1.0
    schema_version: str = "debug_agent_system.evidence_link.v3"


@dataclass(slots=True)
class HypothesisRecord:
    hypothesis_id: str
    label: str
    mechanism: str = ""
    suspected_component: str = ""
    state: HypothesisState = "candidate"
    confidence: float = 0.0
    support_evidence_ids: list[str] = field(default_factory=list)
    contradict_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    family_id: str = ""
    variant_id: str = ""
    source_provider: str = ""
    schema_version: str = "debug_agent_system.hypothesis_record.v3"


@dataclass(slots=True)
class TraceCandidate:
    """One evidence-bounded incident chain reconstructed from source records."""

    trace_id: str
    title: str
    device_scope: str = ""
    failure_chain: str = ""
    time_boundary: str = ""
    action_results: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    uncertainty: str = ""
    state: Literal["candidate", "needs_evidence", "closed"] = "candidate"
    schema_version: str = "debug_agent_system.trace_candidate.v3"


@dataclass(slots=True)
class AnswerClaim:
    claim_id: str
    text: str
    evidence_ids: list[str]
    assertion: EvidenceAssertion = "source_asserted"
    confidence: float = 1.0


@dataclass(slots=True)
class AnswerPlanSection:
    section_id: str
    title: str
    section_type: str
    claims: list[AnswerClaim] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    risk: RiskLevel = "safe"
    status: Literal["expanded", "risk_controlled", "omitted_evidence_gap"] = "expanded"


@dataclass(slots=True)
class AnswerPlan:
    task: ReadTask
    sections: list[AnswerPlanSection] = field(default_factory=list)
    hypotheses: list[HypothesisRecord] = field(default_factory=list)
    traces: list[TraceCandidate] = field(default_factory=list)
    unresolved_gaps: list[str] = field(default_factory=list)
    baseline_status: str = ""
    proposed_status: str = ""
    schema_version: str = "debug_agent_system.answer_plan.v3"


@dataclass(slots=True)
class PolicyDecision:
    answerable: bool
    diagnosable: bool
    executable: bool
    verified_fix: bool
    official_status: str
    proposed_status: str
    active_answer_source: str
    reasons: list[str] = field(default_factory=list)
    blocked_actions: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "debug_agent_system.policy_decision.v3"


@dataclass(slots=True)
class VerificationReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_claims: int = 0
    checked_hypotheses: int = 0
    schema_version: str = "debug_agent_system.answer_verification.v3"


@dataclass(slots=True)
class ReadResponse:
    query: str
    status: str
    answer: str
    task: ReadTask
    answer_plan: AnswerPlan
    policy: PolicyDecision
    verification: VerificationReport
    evidence_snapshot: dict[str, Any]
    baseline_response: dict[str, Any] = field(default_factory=dict)
    shadow: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "debug_agent_system.read_response.v3"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
