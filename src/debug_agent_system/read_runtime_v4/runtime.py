from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from debug_agent_system.read_runtime_v3.contracts import ReadRequest
from debug_agent_system.read_runtime_v3.fabric import EvidenceFabric
from debug_agent_system.read_runtime_v3.providers import (
    FrozenPipelineProvider,
    IncidentProvider,
    KGSAGProvider,
    RawCorpusProvider,
    RequestContextProvider,
    ReadToolRegistry,
)

from .config import ReadRuntimeV4Options
from .contracts import V4Response
from .planner import CodexInvestigationPlanner, InvestigationPlanner, render_answer
from .policy import InvestigationPolicy
from .tasking import compile_task
from .verifier import InvestigationVerifier


class ReadRuntimeV4:
    """Evidence-first investigation runtime, additive to the frozen path."""

    def __init__(self, *, baseline=None, kg_sag=None, raw=None, incident=None, options=None, planner=None):
        self.baseline = baseline
        self.kg_sag = kg_sag
        self.raw = raw
        self.incident = incident
        self.options = options or ReadRuntimeV4Options()
        self.planner = planner or InvestigationPlanner()
        self.policy = InvestigationPolicy()
        self.verifier = InvestigationVerifier()
        self.request_context = RequestContextProvider()

    @classmethod
    def from_system(cls, system: Any, *, options: ReadRuntimeV4Options | None = None, workspace: str | Path | None = None):
        selected = options or ReadRuntimeV4Options()
        root = Path(workspace or Path.cwd()).resolve()
        planner = InvestigationPlanner()
        if selected.planner == "codex_investigation":
            from debug_agent_system.adapters.codex_read.client import CodexResponsesClient
            from debug_agent_system.read_runtime_v3.agentic import CodexResponsesPlanRunner
            planner = CodexInvestigationPlanner(CodexResponsesPlanRunner(
                CodexResponsesClient(env_file=root / ".env.local", timeout_seconds=selected.timeout_seconds),
                model=selected.model,
                reasoning_effort=selected.reasoning_effort,
                max_tool_rounds=selected.max_tool_rounds,
                max_tool_calls=selected.max_tool_calls,
            ))
        return cls(
            baseline=FrozenPipelineProvider(system.start) if selected.baseline_enabled else None,
            kg_sag=KGSAGProvider(system.read_model, kg_root=system.config.knowledge.kg_v2_root) if selected.kg_sag_enabled else None,
            raw=RawCorpusProvider(workspace=root) if selected.raw_enabled else None,
            incident=IncidentProvider(system.analyze_incident) if selected.incident_enabled else None,
            options=selected,
            planner=planner,
        )

    def run(self, payload: dict[str, Any] | ReadRequest) -> dict[str, Any]:
        request = ReadRequest.from_payload(payload)
        if not request.query.strip():
            raise ValueError("read_runtime_v4_requires_query")
        task = compile_task(request, self.options.budgets)
        fabric = EvidenceFabric()
        trace: list[dict[str, Any]] = []
        provider_results: dict[str, Any] = {}

        # Incident evidence is collected before optional baseline reference so
        # the case scope, not the frozen answer, determines the investigation.
        if self.incident and self.options.incident_enabled:
            provider_results["incident"] = self._collect("incident", self.incident, request, task.task, fabric, trace)
        provider_results["request_context"] = self._collect("request_context", self.request_context, request, task.task, fabric, trace)
        if self.kg_sag and self.options.kg_sag_enabled:
            provider_results["kg_sag"] = self._collect("kg_sag", self.kg_sag, request, task.task, fabric, trace)
        if self.raw and self.options.raw_enabled:
            provider_results["raw"] = self._collect("raw", self.raw, request, task.task, fabric, trace)
        if self.baseline and self.options.baseline_enabled:
            provider_results["baseline"] = self._collect("baseline", self.baseline, request, task.task, fabric, trace)

        incident_payload = provider_results.get("incident") or {}
        baseline_payload = provider_results.get("baseline") or {}
        tool_registry = ReadToolRegistry(
            kg=self.kg_sag, raw=self.raw, incident=self.incident, fabric=fabric,
        )
        plan = self.planner.build(
            request=request,
            task=task,
            fabric=fabric,
            tool_registry=tool_registry,
            baseline_result=baseline_payload,
            kg_result=provider_results.get("kg_sag"),
            incident_result=incident_payload,
            raw_result=provider_results.get("raw"),
        )
        policy = self.policy.decide(plan, baseline_payload.get("response") or {})
        verification = self.verifier.verify(plan, fabric)
        proposed_answer = render_answer(plan)
        baseline_response = dict(baseline_payload.get("response") or {})
        baseline_answer = str(baseline_response.get("answer") or "")
        # v4 is primarily an incident investigator.  For procedure/evidence
        # queries, preserve the already verified frozen answer until a
        # procedure planner is enabled; otherwise active v4 would replace a
        # complete answer with an empty incident-shaped rendering.
        non_incident_compat = (
            task.output_contract != "incident_report" and bool(baseline_answer.strip())
        )
        if self.options.shadow_mode:
            answer = baseline_answer if baseline_answer else proposed_answer
            status = str(baseline_response.get("status") or "ask_info")
            source = "frozen_read_pipeline"
        elif non_incident_compat:
            answer = baseline_answer
            status = str(baseline_response.get("status") or "ask_info")
            source = "read_runtime_v4_non_incident_baseline_compat"
        elif verification.passed:
            answer = proposed_answer
            status = policy.status
            source = "read_runtime_v4"
        elif self.options.fail_open_to_v3:
            answer = baseline_answer or "Read Runtime v4 answer verification failed."
            status = str(baseline_response.get("status") or "failed")
            source = "frozen_read_pipeline_fallback"
        else:
            answer = "Read Runtime v4 answer verification failed."
            status = "failed"
            source = "read_runtime_v4_verifier"
        result = V4Response(
            query=request.query,
            status=status,
            answer=answer,
            task=task,
            state=plan.state,
            answer_plan=plan,
            policy=policy,
            verification=verification,
            evidence_snapshot=fabric.snapshot(),
            provider_results=_summarize_provider_results(provider_results),
            baseline_response=baseline_response,
            shadow={
                "enabled": self.options.shadow_mode,
                "proposed_answer": proposed_answer,
                "proposed_status": policy.status,
                "active_answer_source": source,
                "answer_changed": proposed_answer != baseline_answer,
                "trace": trace,
            },
        )
        return result.to_dict()

    @staticmethod
    def _collect(label, provider, request, task, fabric, trace):
        started = time.perf_counter()
        try:
            result = dict(provider.collect(request, task, fabric) or {})
            trace.append({"stage": f"provider:{label}", "status": "ok", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "timestamp": datetime.now(timezone.utc).isoformat()})
            return result
        except Exception as exc:
            trace.append({"stage": f"provider:{label}", "status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:240]}", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
            return {"failed": True, "error_type": type(exc).__name__, "error": str(exc)[:240]}


def _summarize_provider_results(results: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in results.items():
        if not isinstance(value, dict):
            output[name] = value
            continue
        if name == "incident":
            result = value.get("result") or {}
            output[name] = {
                "status": result.get("status"),
                "event_count": len(result.get("events") or []),
                "stack_trace_count": len(result.get("stack_traces") or []),
                "hypothesis_count": len(result.get("hypotheses") or []),
                "next_test_count": len(result.get("next_tests") or []),
            }
        elif name == "baseline":
            response = value.get("response") or {}
            output[name] = {"status": response.get("status"), "family_id": response.get("family_id"), "variant_id": response.get("variant_id")}
        else:
            output[name] = {key: value[key] for key in value if key.endswith("_ids") or key in {"skipped", "agent_tools_available", "retrieval_trace"}}
    return output
