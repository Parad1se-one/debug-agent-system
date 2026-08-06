"""Source-closed Evidence Pack for incident analysis and LLM orchestration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import (
    DiagnosticHypothesis,
    DiagnosticTest,
    EvidenceLink,
    IncidentCase,
)
from .parsers import ParsedDiagnostics


SCHEMA_VERSION = "debug_agent_system.incident_evidence_pack.v3"


def build_incident_evidence_pack(
    case: IncidentCase,
    parsed: ParsedDiagnostics,
    timeline: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    case_graph: dict[str, Any],
    retrieval: dict[str, Any],
    hypotheses: list[DiagnosticHypothesis],
    next_tests: list[DiagnosticTest],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an immutable, source-indexed pack; no claim is added here."""

    source_index = {
        link.evidence_id: _source_entry(link)
        for link in parsed.evidence_links
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "case": asdict(case),
        "artifact_manifest": [asdict(item) for item in case.artifacts],
        "diagnostic_events": [asdict(item) for item in parsed.events],
        "stack_traces": [asdict(item) for item in parsed.stack_traces],
        "environment": asdict(parsed.environment),
        "timeline": timeline,
        "correlations": correlations,
        "case_graph": case_graph,
        "kg_retrieval": retrieval,
        "hypothesis_matrix": [asdict(item) for item in hypotheses],
        "next_best_tests": [asdict(item) for item in next_tests],
        "source_index": source_index,
        "exclusions": list(exclusions),
        "claim_policy": {
            "facts_require_source_index_entry": True,
            "kg_candidates_are_case_facts": False,
            "similar_cases_are_formal_knowledge": False,
            "detection_point_is_root_cause": False,
            "jira_status_is_verified_fix": False,
            "canonical_kg_mutated": False,
        },
    }


def verify_incident_evidence_pack(pack: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if pack.get("schema_version") != SCHEMA_VERSION:
        errors.append({"code": "incident_evidence_pack_schema_mismatch"})
    source_index = pack.get("source_index") or {}
    if not isinstance(source_index, dict):
        errors.append({"code": "incident_source_index_invalid"})
        source_index = {}
    known = set(source_index)
    for hypothesis in pack.get("hypothesis_matrix") or []:
        if not isinstance(hypothesis, dict):
            errors.append({"code": "incident_hypothesis_invalid"})
            continue
        references = [
            *(hypothesis.get("support_evidence_ids") or []),
            *(hypothesis.get("contradict_evidence_ids") or []),
        ]
        unknown = [str(item) for item in references if str(item) not in known]
        if unknown:
            errors.append({
                "code": "incident_hypothesis_source_not_closed",
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "evidence_ids": unknown,
            })
    return errors


def _source_entry(link: EvidenceLink) -> dict[str, Any]:
    item = asdict(link)
    item["locator"] = _locator(link)
    return item


def _locator(link: EvidenceLink) -> str:
    if link.line_start is not None:
        end = link.line_end if link.line_end is not None else link.line_start
        return f"{link.source_name}:L{link.line_start}-L{end}"
    if link.byte_start is not None:
        end = link.byte_end if link.byte_end is not None else link.byte_start
        return f"{link.source_name}:bytes={link.byte_start}-{end}"
    return link.source_name


__all__ = [
    "SCHEMA_VERSION",
    "build_incident_evidence_pack",
    "verify_incident_evidence_pack",
]
