"""Shadow-first orchestration runtime for Read Runtime v3."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from .config import ReadRuntimeV3Options
from .contracts import ReadRequest, ReadResponse
from .fabric import EvidenceFabric
from .planner import EvidenceFirstPlanner, render_answer
from .policy import ReadPolicyEngine
from .providers import (
    FrozenPipelineProvider,
    IncidentProvider,
    KGSAGProvider,
    RawCorpusProvider,
    RequestContextProvider,
    ReadToolRegistry,
)
from .tasking import normalize_task
from .verifier import AnswerPlanVerifier


class ReadRuntimeV3:
    """Coordinate frozen providers without changing the official path."""

    def __init__(
        self,
        *,
        baseline: FrozenPipelineProvider,
        options: ReadRuntimeV3Options | None = None,
        kg_sag: KGSAGProvider | None = None,
        raw: RawCorpusProvider | None = None,
        incident: IncidentProvider | None = None,
        request_context: RequestContextProvider | None = None,
        planner: Any | None = None,
        policy: ReadPolicyEngine | None = None,
        verifier: AnswerPlanVerifier | None = None,
    ) -> None:
        self.baseline = baseline
        self.options = options or ReadRuntimeV3Options()
        self.kg_sag = kg_sag
        self.raw = raw
        self.incident = incident
        self.request_context = request_context or RequestContextProvider()
        self.planner = planner or EvidenceFirstPlanner()
        self.policy = policy or ReadPolicyEngine()
        self.verifier = verifier or AnswerPlanVerifier()

    @classmethod
    def from_system(
        cls,
        system: Any,
        *,
        options: ReadRuntimeV3Options | None = None,
        workspace: str | Path | None = None,
    ) -> "ReadRuntimeV3":
        selected = options or ReadRuntimeV3Options()
        planner: Any = EvidenceFirstPlanner()
        if selected.planner == "codex_agentic":
            from debug_agent_system.adapters.codex_read.client import CodexResponsesClient
            from .agentic import AgenticEvidencePlanner, CodexResponsesPlanRunner

            repo_root = Path(workspace or Path.cwd()).resolve()
            planner = AgenticEvidencePlanner(CodexResponsesPlanRunner(
                CodexResponsesClient(
                    env_file=repo_root / ".env.local",
                    timeout_seconds=selected.timeout_seconds,
                ),
                model=selected.model,
                reasoning_effort=selected.reasoning_effort,
                max_tool_rounds=selected.max_tool_rounds,
                max_tool_calls=selected.max_tool_calls,
            ))
        return cls(
            baseline=FrozenPipelineProvider(system.start),
            options=selected,
            kg_sag=(
                KGSAGProvider(system.read_model, kg_root=system.config.knowledge.kg_v2_root)
                if selected.kg_sag_enabled else None
            ),
            raw=(
                RawCorpusProvider(workspace=workspace or Path.cwd())
                if selected.raw_enabled else None
            ),
            incident=(
                IncidentProvider(system.analyze_incident)
                if selected.incident_enabled else None
            ),
            planner=planner,
        )

    def run(self, payload: dict[str, Any] | ReadRequest) -> dict[str, Any]:
        request = ReadRequest.from_payload(payload)
        if not request.query.strip():
            raise ValueError("read_runtime_v3_requires_query")
        task = normalize_task(request, budgets=self.options.budgets)
        fabric = EvidenceFabric()
        trace: list[dict[str, Any]] = []

        baseline_result = self._collect(
            "baseline", self.baseline, request, task, fabric, trace, required=True
        )
        baseline_response = dict(baseline_result.get("response") or {})
        context_result = self._collect(
            "request_context", self.request_context, request, task, fabric, trace
        )
        kg_result = self._collect(
            "kg_sag", self.kg_sag, request, task, fabric, trace
        ) if self.kg_sag and self.options.kg_sag_enabled else None
        raw_result = self._collect(
            "raw", self.raw, request, task, fabric, trace
        ) if self.raw and self.options.raw_enabled else None
        incident_result = self._collect(
            "incident", self.incident, request, task, fabric, trace
        ) if self.incident and self.options.incident_enabled else None

        tool_registry = self.tool_registry(fabric)
        planner_started = time.perf_counter()
        try:
            plan = self.planner.build(
                request=request,
                task=task,
                fabric=fabric,
                tool_registry=tool_registry,
                baseline_result=baseline_result,
                kg_result=kg_result,
                incident_result=incident_result,
            )
            trace.append({
                "stage": "planner",
                "planner": str(getattr(self.planner, "name", type(self.planner).__name__)),
                "status": "ok",
                "elapsed_ms": round((time.perf_counter() - planner_started) * 1000, 3),
                "tool_trace": list(getattr(self.planner, "last_trace", [])),
            })
        except Exception as exc:
            fabric.create_record(
                kind="exclusion",
                provider="planner",
                source_ref=str(getattr(self.planner, "name", type(self.planner).__name__)),
                assertion="derived",
                summary=f"Planner failed; deterministic fallback used: {type(exc).__name__}",
                content={"error_type": type(exc).__name__, "message": str(exc)[:240]},
            )
            trace.append({
                "stage": "planner",
                "planner": str(getattr(self.planner, "name", type(self.planner).__name__)),
                "status": "fallback",
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                "elapsed_ms": round((time.perf_counter() - planner_started) * 1000, 3),
            })
            plan = EvidenceFirstPlanner().build(
                request=request,
                task=task,
                fabric=fabric,
                tool_registry=tool_registry,
                baseline_result=baseline_result,
                kg_result=kg_result,
                incident_result=incident_result,
            )
        policy = self.policy.decide(
            request=request,
            baseline=baseline_response,
            plan=plan,
            shadow_mode=self.options.shadow_mode,
        )
        verification = self.verifier.verify(plan=plan, policy=policy, fabric=fabric)
        proposed_answer = render_answer(plan)
        baseline_answer = str(baseline_response.get("answer") or "")
        baseline_status = str(baseline_response.get("status") or "failed")
        use_v3 = bool(
            not self.options.shadow_mode
            and verification.passed
            and proposed_answer
        )
        answer = proposed_answer if use_v3 else baseline_answer
        status = policy.proposed_status if use_v3 else baseline_status
        if not verification.passed and not self.options.fail_open_to_baseline:
            status = "failed"
            answer = "Read Runtime v3 answer verification failed."

        snapshot = fabric.snapshot()
        response = ReadResponse(
            query=request.query,
            status=status,
            answer=answer,
            task=task,
            answer_plan=plan,
            policy=policy,
            verification=verification,
            evidence_snapshot=snapshot,
            baseline_response=baseline_response,
            shadow={
                "enabled": self.options.shadow_mode,
                "proposed_answer": proposed_answer,
                "proposed_status": policy.proposed_status,
                "answer_changed": proposed_answer != baseline_answer,
                "status_changed": policy.proposed_status != baseline_status,
                "raw_provider": raw_result or {},
                "kg_provider": kg_result or {},
                "incident_provider": _summary(incident_result or {}),
                "request_context_provider": context_result,
            },
            trace=trace,
        )
        return response.to_dict()

    def tool_registry(self, fabric: EvidenceFabric | None = None) -> ReadToolRegistry:
        return ReadToolRegistry(
            kg=self.kg_sag,
            raw=self.raw,
            incident=self.incident,
            fabric=fabric,
        )

    @staticmethod
    def _collect(
        label: str,
        provider: Any,
        request: ReadRequest,
        task: Any,
        fabric: EvidenceFabric,
        trace: list[dict[str, Any]],
        *,
        required: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = dict(provider.collect(request, task, fabric) or {})
            trace.append({
                "stage": f"provider:{label}",
                "status": "ok",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return result
        except Exception as exc:
            trace.append({
                "stage": f"provider:{label}",
                "status": "failed",
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            fabric.create_record(
                kind="exclusion",
                provider=f"provider:{label}",
                source_ref=label,
                assertion="derived",
                summary=f"Provider failed: {type(exc).__name__}",
                content={"error_type": type(exc).__name__, "message": str(exc)[:240]},
                confidence=1.0,
            )
            if required:
                raise
            return {"failed": True, "error_type": type(exc).__name__}


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("skipped") or value.get("failed"):
        return dict(value)
    result = dict(value.get("result") or {})
    return {
        "status": result.get("status"),
        "event_count": len(result.get("events") or []),
        "stack_trace_count": len(result.get("stack_traces") or []),
        "hypothesis_count": len(value.get("hypotheses") or []),
        "next_test_count": len(result.get("next_tests") or []),
    }
