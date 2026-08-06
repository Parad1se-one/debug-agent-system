from __future__ import annotations

from debug_agent_system.read_runtime_v3.config import ReadRuntimeV3Options
from debug_agent_system.read_runtime_v3.planner import EvidenceFirstPlanner
from debug_agent_system.read_runtime_v3.providers import (
    FrozenPipelineProvider,
    IncidentProvider,
)
from debug_agent_system.read_runtime_v3.runtime import ReadRuntimeV3


def _baseline(_payload):
    return {
        "schema_version": "debug_agent_system.answer_pack.v2",
        "session_id": "frozen-session",
        "status": "ask_info",
        "answer": "这是冻结管线的正式回答。",
        "required_data": ["请补充复现条件"],
        "confidence": 0.0,
        "sources": ["data/raw/source.docx"],
        "evidence_ids": ["evidence:old:1"],
        "family_id": "",
        "variant_id": "",
        "metadata": {
            "sufficiency": {
                "answerable": True,
                "diagnosable": False,
                "executable": False,
            }
        },
    }


def _incident(_payload):
    return {
        "schema_version": "debug_agent_system.incident_evidence_pack.v3",
        "status": "needs_evidence",
        "case": {"case_id": "incident:1"},
        "evidence_links": [{
            "evidence_id": "native:line:1",
            "artifact_id": "artifact:log",
            "source_name": "runtime.log",
            "line_start": 10,
            "line_end": 10,
            "timestamp": "2026-08-01T21:30:00",
            "parser_version": "test.v1",
        }],
        "events": [{
            "event_id": "event:1",
            "artifact_id": "artifact:log",
            "message": "Gpu API call: illegal memory access",
            "timestamp_raw": "2026-08-01 21:30:00",
            "polarity": "negative",
            "severity": "error",
            "evidence_ids": ["native:line:1"],
        }],
        "stack_traces": [],
        "environment": {"values": {}, "evidence_ids": {}},
        "hypotheses": [{
            "hypothesis_id": "hypothesis:gpu",
            "label": "GPU 上下文或驱动链异常",
            "failure_mechanism": "GPU API illegal memory access",
            "suspected_component": "GPU runtime",
            "support_evidence_ids": ["native:line:1"],
            "contradict_evidence_ids": [],
            "missing_evidence": ["GPU 型号与驱动版本"],
            "confidence": 0.8,
            "status": "locked",
        }],
        "next_tests": [{
            "test_id": "test:gpu",
            "title": "核对驱动环境",
            "instruction": "导出 GPU 型号、驱动版本和 CUDA 运行时版本。",
            "information_gain": 0.9,
            "risk": "safe",
        }],
        "report": "incident report",
    }


def test_shadow_runtime_keeps_official_answer_and_exposes_proposed_plan():
    runtime = ReadRuntimeV3(
        baseline=FrozenPipelineProvider(_baseline),
        incident=IncidentProvider(_incident),
        options=ReadRuntimeV3Options(
            shadow_mode=True,
            kg_sag_enabled=False,
            raw_enabled=False,
            incident_enabled=True,
        ),
    )
    result = runtime.run({
        "query": "设备在 2026-08-01 21:30 闪退",
        "evidence_resources": [{
            "resource_id": "pkg:1", "kind": "log_package", "path": "/tmp/a.zip",
        }],
    })
    assert result["answer"] == "这是冻结管线的正式回答。"
    assert result["status"] == "ask_info"
    assert result["shadow"]["enabled"] is True
    assert "诊断数据中的直接观测" in result["shadow"]["proposed_answer"]
    assert result["verification"]["passed"] is True
    assert result["policy"]["diagnosable"] is False
    assert result["answer_plan"]["hypotheses"][0]["state"] == "needs_evidence"
    assert result["evidence_snapshot"]["providers"]["incident_evidence_runtime"] >= 3


class _UpgradePlanner(EvidenceFirstPlanner):
    def build(self, **kwargs):
        plan = super().build(**kwargs)
        plan.proposed_status = "resolved"
        return plan


def test_policy_rejects_unproven_status_upgrade_in_active_mode():
    runtime = ReadRuntimeV3(
        baseline=FrozenPipelineProvider(_baseline),
        planner=_UpgradePlanner(),
        options=ReadRuntimeV3Options(
            shadow_mode=False,
            kg_sag_enabled=False,
            raw_enabled=False,
            incident_enabled=False,
        ),
    )
    result = runtime.run({"query": "如何进入安全模式"})
    assert result["status"] == "ask_info"
    assert result["policy"]["proposed_status"] == "ask_info"
    assert "v3_may_not_upgrade_frozen_runtime_status_without_policy_evidence" in result["policy"]["reasons"]
    assert result["verification"]["passed"] is True


def test_non_incident_request_does_not_call_incident_provider():
    calls = []

    def incident(payload):
        calls.append(payload)
        return _incident(payload)

    runtime = ReadRuntimeV3(
        baseline=FrozenPipelineProvider(_baseline),
        incident=IncidentProvider(incident),
        options=ReadRuntimeV3Options(
            shadow_mode=True,
            kg_sag_enabled=False,
            raw_enabled=False,
            incident_enabled=True,
        ),
    )
    result = runtime.run({"query": "如何进入安全模式"})
    assert calls == []
    assert result["shadow"]["incident_provider"]["skipped"] is True


def test_source_only_request_context_enters_evidence_fabric_without_inference():
    runtime = ReadRuntimeV3(
        baseline=FrozenPipelineProvider(_baseline),
        options=ReadRuntimeV3Options(
            shadow_mode=True,
            kg_sag_enabled=False,
            raw_enabled=False,
            incident_enabled=False,
        ),
    )
    result = runtime.run({
        "query": "请按时间边界拆分现场记录",
        "routing_context": {
            "source_only_context_ref": {"path": "input.json", "sha256": "abc"},
            "source_only_context": {
                "case_id": "case:source-only",
                "label_visibility": "source_records_only",
                "messages": [{
                    "message_id": "message:1",
                    "create_time": "2026-08-01 21:30",
                    "text": "设备运行中闪退",
                    "attachments": [{"name": "runtime.log", "path": "/tmp/runtime.log"}],
                }],
            },
        },
    })
    records = result["evidence_snapshot"]["records"]
    context_records = [item for item in records if item["provider"] == "request_context"]
    assert {item["kind"] for item in context_records} == {
        "source_artifact", "diagnostic_event", "media_asset",
    }
    assert result["shadow"]["request_context_provider"]["source_message_ids"] == [
        "message:1"
    ]
    assert not result["answer_plan"]["hypotheses"]
