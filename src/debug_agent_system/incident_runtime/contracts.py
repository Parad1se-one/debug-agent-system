"""Contracts for source-bound incident diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AnchorStability = Literal["stable", "contextual", "volatile"]
ArtifactStatus = Literal[
    "available", "missing", "unsupported", "rejected", "parse_failed"
]
HypothesisStatus = Literal[
    "candidate", "needs_evidence", "supported", "locked", "ruled_out", "inconclusive"
]


@dataclass(slots=True)
class EvidenceLink:
    evidence_id: str
    artifact_id: str
    source_name: str
    sha256: str = ""
    line_start: int | None = None
    line_end: int | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    timestamp: str = ""
    extraction_method: str = ""
    parser_version: str = ""


@dataclass(slots=True)
class ArtifactManifest:
    artifact_id: str
    resource_id: str
    name: str
    kind: str
    path: str = ""
    url: str = ""
    mime: str = ""
    size: int | None = None
    sha256: str = ""
    parent_artifact_id: str = ""
    archive_member: str = ""
    status: ArtifactStatus = "available"
    parser_state: str = "pending"
    safety_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StackFrame:
    ordinal: int
    raw: str
    module: str = ""
    function: str = ""
    source_file: str = ""
    line: int | None = None
    address: str = ""
    stability: AnchorStability = "contextual"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StackTrace:
    trace_id: str
    artifact_id: str
    frames: list[StackFrame] = field(default_factory=list)
    thread_id: str = ""
    exception_code: str = ""
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticEvent:
    event_id: str
    artifact_id: str
    sequence: int
    severity: str
    message: str
    timestamp_raw: str = ""
    timestamp_utc: str = ""
    process: str = ""
    process_id: str = ""
    thread_id: str = ""
    component: str = ""
    module: str = ""
    function: str = ""
    error_codes: list[str] = field(default_factory=list)
    event_kind: str = "diagnostic_event"
    polarity: Literal["positive", "negative", "neutral"] = "negative"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EnvironmentSnapshot:
    values: dict[str, list[str]] = field(default_factory=dict)
    evidence_ids: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class IncidentCase:
    case_id: str
    query: str
    jira_key: str = ""
    snapshot_time: str = ""
    status: str = ""
    affected_version: str = ""
    device: str = ""
    station: str = ""
    reproduction: str = ""
    artifacts: list[ArtifactManifest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiagnosticHypothesis:
    hypothesis_id: str
    label: str
    failure_mechanism: str
    suspected_component: str
    family_id: str = ""
    variant_id: str = ""
    support_evidence_ids: list[str] = field(default_factory=list)
    contradict_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: HypothesisStatus = "candidate"
    retrieval_score: float = 0.0
    source_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticTest:
    test_id: str
    title: str
    instruction: str
    distinguishes_hypothesis_ids: list[str] = field(default_factory=list)
    expected_observations: list[str] = field(default_factory=list)
    evidence_required: list[str] = field(default_factory=list)
    information_gain: float = 0.0
    cost: Literal["low", "medium", "high"] = "low"
    risk: Literal["safe", "controlled", "destructive"] = "safe"


@dataclass(slots=True)
class IncidentResult:
    schema_version: str
    status: str
    case: IncidentCase
    events: list[DiagnosticEvent] = field(default_factory=list)
    stack_traces: list[StackTrace] = field(default_factory=list)
    environment: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    correlations: list[dict[str, Any]] = field(default_factory=list)
    case_graph: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    evidence_pack: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)
    next_tests: list[DiagnosticTest] = field(default_factory=list)
    report: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    observability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
