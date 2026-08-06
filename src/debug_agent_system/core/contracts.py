"""Shared structured contracts for the independent debug agent system."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

AgentStatus = Literal["ask_info", "step", "resolved", "escalate", "failed"]
EvidenceResourceKind = Literal[
    "attachment",
    "document",
    "dmp",
    "image",
    "jira",
    "log_package",
    "proj",
    "auto",
]
ToolExecutionStatus = Literal[
    "parsed",
    "metadata_only",
    "parse_failed",
    "skipped",
]


@dataclass(slots=True)
class AnswerSection:
    section_type: str
    title: str
    items: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DebugAgentInput:
    query: str
    interactive: bool = True
    session: dict[str, Any] = field(default_factory=dict)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    log_summary: dict[str, Any] = field(default_factory=dict)
    routing_context: dict[str, Any] = field(default_factory=dict)
    evidence_resources: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceResource:
    """Caller-supplied evidence that may be inspected by a bounded parser."""

    resource_id: str
    kind: EvidenceResourceKind = "auto"
    name: str = ""
    path: str = ""
    url: str = ""
    text: str = ""
    mime: str = ""
    size: int | None = None
    sha256: str = ""
    source_message_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceObservation:
    """One source-bound fact normalized from a parser result."""

    observation_id: str
    field: str
    value: Any
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    extraction_mode: str = "bounded_parser"
    supports_retrieval: bool = True


@dataclass(slots=True)
class ToolResultEnvelope:
    """Stable read-side Tool result; parser-native output stays in ``payload``."""

    schema_version: str
    tool: str
    call_id: str
    call_fingerprint: str
    status: ToolExecutionStatus
    resource_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    observations: list[EvidenceObservation] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    safety: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceGapResolution:
    """Result of a bounded evidence-completion round."""

    schema_version: str
    attempted: bool
    round_count: int
    required_data_before: list[str] = field(default_factory=list)
    resolved_items: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    observations: list[EvidenceObservation] = field(default_factory=list)
    tool_results: list[ToolResultEnvelope] = field(default_factory=list)
    retrieval_context: str = ""
    excluded: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""


@dataclass(slots=True)
class Candidate:
    error_id: str
    label: str
    score: float
    route: str = "lexical_kg"
    evidence: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckNode:
    check_id: str
    label: str
    how_to_check: str
    step_order: int = 0
    destructive: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SolutionNode:
    solution_id: str
    content: str
    evidence_level: str = ""
    destructive: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LockedSubgraph:
    error_id: str
    label: str
    symptom: str = ""
    category: str = ""
    escalation_target: str = ""
    required_info: list[str] = field(default_factory=list)
    checks: list[CheckNode] = field(default_factory=list)
    solutions_by_check: dict[str, list[SolutionNode]] = field(default_factory=dict)
    next_edges_by_check: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FaultEpisode:
    episode_id: str
    thread_id: str
    completeness: Literal["complete", "partial", "noise"]
    fault_description_messages: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_chain_messages: list[dict[str, Any]] = field(default_factory=list)
    resolution_messages: list[dict[str, Any]] = field(default_factory=list)
    noise_messages: list[dict[str, Any]] = field(default_factory=list)
    evidence_message_ids: list[str] = field(default_factory=list)
    source_offsets: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    extracted: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SchemaValidCandidate:
    candidate_id: str
    source_episode_id: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    schema_valid: bool = False
    schema_issues: list[str] = field(default_factory=list)
    proposal_only: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionState:
    session_id: str
    query: str
    status: AgentStatus = "step"
    top_error_id: str = ""
    top_error_label: str = ""
    retrieval_route: str = ""
    lock_status: str = ""
    current_check_id: str = ""
    current_check: str = ""
    current_index: int = 0
    checks_presented: list[str] = field(default_factory=list)
    check_results: dict[str, str] = field(default_factory=dict)
    ruled_out: list[str] = field(default_factory=list)
    which_check_solved: str = ""
    required_data: list[str] = field(default_factory=list)
    resolution: str = ""
    escalation_target: str = ""
    confidence: float = 0.0
    failure_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # KG_v2-native runtime identity.  The legacy-named fields above remain
    # serialization aliases only; runtime decisions must use these fields.
    top_family_id: str = ""
    top_variant_id: str = ""
    top_family_label: str = ""
    top_variant_label: str = ""
    plan_id: str = ""
    plan_source_type: str = ""
    current_action_id: str = ""
    current_trace_step_id: str = ""
    actions_presented: list[str] = field(default_factory=list)
    action_results: dict[str, str] = field(default_factory=dict)
    resolved_action_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    turn_count: int = 0


@dataclass(slots=True)
class AgentResponse:
    schema_version: str
    session_id: str
    status: AgentStatus
    answer: str
    required_data: list[str] = field(default_factory=list)
    current_check_id: str = ""
    current_check: str = ""
    resolution: str = ""
    confidence: float = 0.0
    escalation_target: str = ""
    sources: list[str] = field(default_factory=list)
    failure_type: str = ""
    observability: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    family_id: str = ""
    variant_id: str = ""
    plan_id: str = ""
    current_action_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    answer_sections: list[AnswerSection] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value
