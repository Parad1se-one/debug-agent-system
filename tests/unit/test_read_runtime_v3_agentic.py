from __future__ import annotations

from debug_agent_system.read_runtime_v3.agentic import (
    AgenticEvidencePlanner,
    answer_plan_from_payload,
    answer_plan_schema,
)
from debug_agent_system.read_runtime_v3.config import ReadRuntimeV3Options
from debug_agent_system.read_runtime_v3.contracts import ReadRequest
from debug_agent_system.read_runtime_v3.fabric import EvidenceFabric
from debug_agent_system.read_runtime_v3.providers import (
    FrozenPipelineProvider,
    ReadToolRegistry,
)
from debug_agent_system.read_runtime_v3.runtime import ReadRuntimeV3
from debug_agent_system.read_runtime_v3.tasking import normalize_task


class _FakePlanRunner:
    def __init__(self):
        self.last_trace = []

    def run(self, *, request, task, fabric, tools):
        evidence_id = next(
            record.evidence_id
            for record in fabric.records(kind="answer_fragment")
        )
        self.last_trace = [{"tool": "evidence_query", "status": "ok"}]
        return {
            "sections": [{
                "section_id": "answer",
                "title": "回答",
                "section_type": "answer",
                "claims": [{
                    "claim_id": "claim:1",
                    "text": "根据冻结回答组织。",
                    "evidence_ids": [evidence_id],
                    "assertion": "derived",
                    "confidence": 1.0,
                }],
                "items": ["根据冻结回答组织。"],
                "evidence_ids": [evidence_id],
                "risk": "safe",
                "status": "expanded",
            }],
            "hypotheses": [],
            "traces": [],
            "unresolved_gaps": [],
            "baseline_status": "ask_info",
            "proposed_status": "ask_info",
        }


def _baseline(_payload):
    return {
        "schema_version": "debug_agent_system.answer_pack.v2",
        "session_id": "s1",
        "status": "ask_info",
        "answer": "冻结回答",
        "required_data": [],
        "sources": ["data/raw/a.docx"],
        "evidence_ids": [],
        "metadata": {},
    }


def test_agentic_planner_submits_evidence_bound_answer_plan():
    planner = AgenticEvidencePlanner(_FakePlanRunner())
    runtime = ReadRuntimeV3(
        baseline=FrozenPipelineProvider(_baseline),
        planner=planner,
        options=ReadRuntimeV3Options(
            shadow_mode=True,
            kg_sag_enabled=False,
            raw_enabled=False,
            incident_enabled=False,
        ),
    )
    result = runtime.run({"query": "如何进入安全模式"})
    assert result["answer"] == "冻结回答"
    assert result["shadow"]["proposed_answer"] == "## 回答\n\n- 根据冻结回答组织。"
    assert result["verification"]["passed"] is True
    assert result["trace"][-1]["planner"] == "codex_agentic"
    assert result["trace"][-1]["tool_trace"][0]["tool"] == "evidence_query"


def test_answer_plan_schema_is_strict_at_every_object_boundary():
    schema = answer_plan_schema()
    assert schema["additionalProperties"] is False
    section = schema["properties"]["sections"]["items"]
    claim = section["properties"]["claims"]["items"]
    hypothesis = schema["properties"]["hypotheses"]["items"]
    trace = schema["properties"]["traces"]["items"]
    assert section["additionalProperties"] is False
    assert claim["additionalProperties"] is False
    assert hypothesis["additionalProperties"] is False
    assert trace["additionalProperties"] is False


class _FakeRaw:
    def search_text(self, **_arguments):
        return {
            "matches": [{
                "path": "data/raw/a.md",
                "line": 7,
                "excerpt": "7:安全模式",
            }],
            "returned": 1,
            "truncated": False,
        }


def test_tool_registry_returns_envelope_and_adds_raw_evidence():
    fabric = EvidenceFabric()
    registry = ReadToolRegistry(raw=_FakeRaw(), fabric=fabric)
    result = registry.execute("raw_search_text", {
        "query": "安全模式",
        "path_glob": "data/raw/**",
        "regex": False,
        "case_sensitive": False,
        "max_matches": 10,
        "context_lines": 1,
    })
    assert result["schema_version"] == "debug_agent_system.read_tool_result.v3"
    assert result["capability"] == {
        "read_only": True,
        "side_effect": False,
        "approval_required": False,
    }
    assert len(result["evidence_ids"]) == 1
    assert fabric.get(result["evidence_ids"][0]).anchors[0].line_start == 7
