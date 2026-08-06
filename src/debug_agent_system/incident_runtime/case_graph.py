"""Ephemeral case evidence graph; never writes canonical KG_v2."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import IncidentCase
from .parsers import ParsedDiagnostics


def build_case_graph(case: IncidentCase, parsed: ParsedDiagnostics) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [{
        "id": case.case_id,
        "type": "IncidentCase",
        "label": case.jira_key or case.query[:120],
        "properties": {
            "jira_key": case.jira_key,
            "status": case.status,
            "affected_version": case.affected_version,
        },
    }]
    edges: list[dict[str, Any]] = []
    for artifact in case.artifacts:
        nodes.append({
            "id": artifact.artifact_id,
            "type": "Artifact",
            "label": artifact.archive_member or artifact.name,
            "properties": {
                "sha256": artifact.sha256,
                "status": artifact.status,
                "parser_state": artifact.parser_state,
            },
        })
        edges.append({"from": case.case_id, "relation": "has_artifact", "to": artifact.artifact_id})
        if artifact.parent_artifact_id:
            edges.append({"from": artifact.parent_artifact_id, "relation": "contains", "to": artifact.artifact_id})
    for event in parsed.events:
        nodes.append({
            "id": event.event_id,
            "type": "DiagnosticEvent",
            "label": event.event_kind,
            "properties": {
                "timestamp": event.timestamp_utc or event.timestamp_raw,
                "severity": event.severity,
                "component": event.component,
                "function": event.function,
                "error_codes": list(event.error_codes),
                "evidence_ids": list(event.evidence_ids),
            },
        })
        edges.append({"from": event.artifact_id, "relation": "observed_event", "to": event.event_id})
    for trace in parsed.stack_traces:
        nodes.append({
            "id": trace.trace_id,
            "type": "StackTrace",
            "label": f"{len(trace.frames)} frames",
            "properties": {"evidence_ids": list(trace.evidence_ids)},
        })
        edges.append({"from": trace.artifact_id, "relation": "has_stacktrace", "to": trace.trace_id})
        for frame in trace.frames:
            frame_id = f"{trace.trace_id}:frame:{frame.ordinal}"
            nodes.append({
                "id": frame_id,
                "type": "StackFrame",
                "label": frame.function or frame.module or frame.raw[:120],
                "properties": asdict(frame),
            })
            edges.append({"from": trace.trace_id, "relation": "has_frame", "to": frame_id, "ordinal": frame.ordinal})
    if parsed.environment.values:
        environment_id = f"{case.case_id}:environment"
        nodes.append({
            "id": environment_id,
            "type": "EnvironmentSnapshot",
            "label": "诊断环境快照",
            "properties": asdict(parsed.environment),
        })
        edges.append({"from": case.case_id, "relation": "has_environment", "to": environment_id})
    return {
        "schema_version": "debug_agent_system.case_evidence_graph.v1",
        "ephemeral": True,
        "canonical_kg_mutated": False,
        "nodes": nodes,
        "edges": edges,
    }


__all__ = ["build_case_graph"]
