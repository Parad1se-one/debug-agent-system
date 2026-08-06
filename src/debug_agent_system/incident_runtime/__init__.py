"""Structured Jira/diagnostic-package evidence runtime.

This package is deliberately independent from the KG_v2 decision runtime.  It
turns caller supplied artifacts into source-bound observations, queries KG_v2
with stable diagnostic anchors, and builds an auditable hypothesis matrix.  It
never executes an attachment or mutates the canonical graph.
"""

from .contracts import (
    ArtifactManifest,
    DiagnosticEvent,
    DiagnosticHypothesis,
    DiagnosticTest,
    EnvironmentSnapshot,
    EvidenceLink,
    IncidentCase,
    IncidentResult,
    StackFrame,
    StackTrace,
)
from .runtime import IncidentEvidenceRuntime
from .evidence_pack import SCHEMA_VERSION as INCIDENT_EVIDENCE_PACK_SCHEMA
from .scope import IncidentScope, ReferenceTimeWindow, parse_incident_scope

__all__ = [
    "ArtifactManifest",
    "DiagnosticEvent",
    "DiagnosticHypothesis",
    "DiagnosticTest",
    "EnvironmentSnapshot",
    "EvidenceLink",
    "IncidentCase",
    "IncidentEvidenceRuntime",
    "INCIDENT_EVIDENCE_PACK_SCHEMA",
    "IncidentResult",
    "IncidentScope",
    "ReferenceTimeWindow",
    "StackFrame",
    "StackTrace",
    "parse_incident_scope",
]
