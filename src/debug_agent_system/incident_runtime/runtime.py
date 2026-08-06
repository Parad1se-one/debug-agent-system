"""End-to-end incident evidence runtime."""

from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel
from debug_agent_system.agents.tools.jira_parser import JiraParserAgent

from .artifacts import ArtifactIntake, ArtifactLimits
from .case_graph import build_case_graph
from .correlations import correlate_incident_events, event_signature
from .contracts import IncidentResult
from .evidence_pack import (
    build_incident_evidence_pack,
    verify_incident_evidence_pack,
)
from .hypotheses import HypothesisRuntime
from .kg_bridge import IncidentKGBridge
from .parsers import DiagnosticParserRegistry
from .report import IncidentVerifier, render_incident_report
from .timeline import build_timeline, read_log_window
from .scope import parse_incident_scope


class IncidentEvidenceRuntime:
    """Parse artifacts, retrieve KG hypotheses, and build a verified report."""

    schema_version = "debug_agent_system.incident_result.v1"

    def __init__(
        self,
        read_model: KGV2ReadModel,
        *,
        limits: ArtifactLimits | None = None,
        allow_dump_analysis: bool = False,
        allow_ocr_analysis: bool = False,
    ) -> None:
        self.read_model = read_model
        self.intake = ArtifactIntake(limits)
        self.parsers = DiagnosticParserRegistry(
            max_text_bytes_per_artifact=(limits.max_member_bytes if limits else 32 * 1024 * 1024),
            allow_dump_analysis=allow_dump_analysis,
            allow_ocr_analysis=allow_ocr_analysis,
        )
        self.kg_bridge = IncidentKGBridge(read_model)
        self.hypotheses = HypothesisRuntime(read_model)
        self.verifier = IncidentVerifier()
        self._results: dict[str, IncidentResult] = {}
        self._text_lines: dict[str, dict[str, list[str]]] = {}
        self._scopes: dict[str, dict[str, Any]] = {}

    def analyze(
        self,
        query: str,
        resources: Iterable[dict[str, Any]],
        *,
        log_summary: dict[str, Any] | None = None,
    ) -> IncidentResult:
        started = monotonic()
        normalized_resources = [dict(item) for item in resources if isinstance(item, dict)]
        resource_hints = [
            " ".join(str(item.get(key) or "") for key in ("name", "path", "url"))
            for item in normalized_resources
        ]
        scope = parse_incident_scope(query, resource_hints)
        jira_input = " ".join([
            query,
            *[
                " ".join(str(item.get(key) or "") for key in ("name", "url", "text"))
                for item in normalized_resources
            ],
        ])
        jira_snapshot = JiraParserAgent().parse(jira_input)
        if jira_snapshot.get("issue_keys"):
            detail = next(iter(jira_snapshot.get("offline_details") or []), {})
            normalized_resources.append({
                "resource_id": "resource:jira-offline-snapshot",
                "kind": "jira",
                "name": "jira_offline_snapshot.json",
                "text": json.dumps(jira_snapshot, ensure_ascii=False, default=str),
                "metadata": {
                    "status": detail.get("status") or "",
                    "affected_version": next(iter(jira_snapshot.get("version_hints") or []), ""),
                    "site": next(iter(jira_snapshot.get("site_hints") or []), ""),
                    "reproduction": detail.get("description_preview") or "",
                    "offline_snapshot": True,
                    "fetched": False,
                },
            })
        if not normalized_resources and not log_summary and _looks_like_diagnostic_payload(query):
            normalized_resources.append({
                "resource_id": "resource:query-diagnostic-payload",
                "kind": "log_package",
                "name": "query_diagnostic_payload.log",
                "text": query,
                "metadata": {"synthetic_from_query": True},
            })
        case, root_contents, exclusions = self.intake.create_case(
            query,
            normalized_resources,
            log_summary=log_summary,
        )
        case.metadata["incident_scope"] = scope.to_dict()
        all_contents = []
        seen: set[str] = set()
        known_artifact_ids = {artifact.artifact_id for artifact in case.artifacts}
        for root in root_contents:
            iterator = (
                self.intake.iter_scoped_members(root, scope)
                if scope.has_time_scope
                else self.intake.iter_members(root)
            )
            for item in iterator:
                key = f"{item.manifest.artifact_id}:{item.manifest.archive_member}"
                if key in seen:
                    continue
                seen.add(key)
                all_contents.append(item)
                if item.manifest.artifact_id not in known_artifact_ids:
                    case.artifacts.append(item.manifest)
                    known_artifact_ids.add(item.manifest.artifact_id)
        parsed = self.parsers.parse(all_contents)
        exclusions.extend(parsed.exclusions)
        exclusions.extend(
            {
                "artifact_id": item.manifest.artifact_id,
                "material": item.manifest.archive_member or item.manifest.name,
                "reason": item.manifest.safety_flags[0],
            }
            for item in all_contents
            if item.manifest.status == "rejected" and item.manifest.safety_flags
        )
        timeline = build_timeline(parsed.events)
        correlations = correlate_incident_events(parsed.events)
        graph = build_case_graph(case, parsed)
        graph["incident_correlations"] = correlations
        retrieval = self.kg_bridge.retrieve(query, parsed.events, parsed.stack_traces)
        hypotheses = self.hypotheses.build(
            retrieval,
            parsed.environment,
            events=parsed.events,
            correlations=correlations,
            query=query,
        )
        next_tests = self.hypotheses.propose_tests(hypotheses, parsed.environment)
        evidence_pack = build_incident_evidence_pack(
            case,
            parsed,
            timeline,
            correlations,
            graph,
            retrieval,
            hypotheses,
            next_tests,
            exclusions,
        )
        report = render_incident_report(
            case,
            parsed,
            timeline,
            correlations,
            hypotheses,
            next_tests,
            exclusions,
        )
        verification = self.verifier.verify(parsed, hypotheses, report)
        pack_errors = verify_incident_evidence_pack(evidence_pack)
        if pack_errors:
            verification["errors"] = [*verification.get("errors", []), *pack_errors]
            verification["passed"] = False
        result = IncidentResult(
            schema_version=self.schema_version,
            status="analyzed" if verification["passed"] else "verification_failed",
            case=case,
            events=parsed.events,
            stack_traces=parsed.stack_traces,
            environment=parsed.environment,
            evidence_links=parsed.evidence_links,
            timeline=timeline,
            correlations=correlations,
            case_graph=graph,
            retrieval=retrieval,
            evidence_pack=evidence_pack,
            hypotheses=hypotheses,
            next_tests=next_tests,
            report=report,
            verification=verification,
            exclusions=exclusions,
            observability={
                "agent_id": "INCIDENT-EVIDENCE-RUNTIME",
                "elapsed_ms": round((monotonic() - started) * 1000, 3),
                "artifact_count": len(case.artifacts),
                "event_count": len(parsed.events),
                "stack_trace_count": len(parsed.stack_traces),
                "hypothesis_count": len(hypotheses),
                "correlation_count": len(correlations),
                "time_scoped": scope.has_time_scope,
                "time_window_artifact_count": sum(
                    1
                    for artifact in case.artifacts
                    if artifact.metadata.get("derived_by") == "query_time_window_stream"
                ),
                "canonical_kg_mutated": False,
            },
        )
        self._results[case.case_id] = result
        self._text_lines[case.case_id] = parsed.text_lines
        self._scopes[case.case_id] = scope.to_dict()
        # Parsed minidump/kernel-dump metadata is retained on the manifests, so
        # the path-backed temp files are no longer needed once analysis returns.
        self.intake.cleanup()
        return result

    def get(self, case_id: str) -> IncidentResult | None:
        return self._results.get(case_id)

    def scope(self, case_id: str) -> dict[str, Any]:
        return dict(self._scopes.get(case_id) or {})

    @staticmethod
    def compare_runs(left: IncidentResult, right: IncidentResult) -> dict[str, Any]:
        failure_levels = {"FATAL", "CRITICAL", "ERROR", "ERR", "EXCEPTION", "PANIC"}
        left_signatures = sorted({
            event_signature(item)
            for item in left.events
            if item.severity.upper() in failure_levels and event_signature(item)
        })
        right_signatures = sorted({
            event_signature(item)
            for item in right.events
            if item.severity.upper() in failure_levels and event_signature(item)
        })
        common = sorted(set(left_signatures).intersection(right_signatures))
        return {
            "schema_version": "debug_agent_system.reproduction_compare.v1",
            "status": "ok",
            "baseline_case_id": left.case.case_id,
            "candidate_case_id": right.case.case_id,
            "common_failure_signatures": common,
            "baseline_only_signatures": sorted(set(left_signatures) - set(right_signatures)),
            "candidate_only_signatures": sorted(set(right_signatures) - set(left_signatures)),
            "signature_reproduced": bool(common),
            "controlled_reproduction": False,
            "fix_verified": False,
            "interpretation": (
                "相同签名再次出现，支持故障复发；只有具备受控前置条件和操作记录时，才能称为受控复现。"
                if common
                else "本次未观察到相同签名；单次未出现不足以证明修复有效。"
            ),
        }

    def log_window(
        self,
        case_id: str,
        artifact_id: str,
        line: int,
        *,
        before: int = 10,
        after: int = 20,
    ) -> dict[str, Any]:
        return read_log_window(
            self._text_lines.get(case_id) or {},
            artifact_id,
            line,
            before=before,
            after=after,
        )


_DIAGNOSTIC_PAYLOAD = re.compile(
    r"(?:\[[12]\d{3}-\d{2}-\d{2}|\b(?:ERROR|FATAL|EXCEPTION)\b|"
    r"\b0x[0-9a-fA-F]{6,16}\b|\b\d+#\s+[0-9a-fA-F]{6,16}\b|"
    r"stack\s*trace|调用栈|trace:)",
    re.I,
)


def _looks_like_diagnostic_payload(query: str) -> bool:
    return bool(_DIAGNOSTIC_PAYLOAD.search(str(query)))


__all__ = ["IncidentEvidenceRuntime"]
