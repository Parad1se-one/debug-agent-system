from __future__ import annotations

from debug_agent_system.read_runtime_v3.providers import FrozenPipelineProvider, IncidentProvider
from debug_agent_system.read_runtime_v4 import ReadRuntimeV4, ReadRuntimeV4Options


def _baseline(_payload):
    return {
        "status": "ask_info",
        "answer": "相机拍摄失败参考资料。",
        "required_data": ["现场确认"],
        "sources": ["camera.docx"],
        "metadata": {"sufficiency": {"answerable": True, "diagnosable": False, "executable": False}},
    }


def _incident(_payload):
    return {
        "status": "needs_evidence",
        "case": {"case_id": "case:1"},
        "evidence_links": [{"evidence_id": "src:1", "source_name": "runtime.log", "line_start": 10, "line_end": 10, "timestamp": "2026-08-01T21:30:00", "artifact_id": "log:1"}],
        "events": [{"event_id": "event:1", "artifact_id": "log:1", "message": "LiveKernelEvent 141 CUDA crash", "timestamp_raw": "2026-08-01 21:30:00", "severity": "error", "evidence_ids": ["src:1"]}],
        "stack_traces": [],
        "environment": {"values": {}, "evidence_ids": {}},
        "hypotheses": [{"hypothesis_id": "hyp:gpu", "label": "GPU/显示驱动执行链异常", "failure_mechanism": "GPU watchdog and CUDA crash", "support_evidence_ids": ["src:1"], "missing_evidence": ["符号化 dump"], "confidence": 0.74, "status": "supported"}],
        "next_tests": [{"test_id": "test:dump", "title": "符号化分析", "instruction": "分析 WATCHDOG dump", "risk": "safe"}],
    }


def test_v4_incident_answer_prioritizes_case_evidence_over_baseline():
    runtime = ReadRuntimeV4(
        baseline=FrozenPipelineProvider(_baseline),
        incident=IncidentProvider(_incident),
        options=ReadRuntimeV4Options(shadow_mode=True, kg_sag_enabled=False, raw_enabled=False),
    )
    result = runtime.run({"query": "设备在 2026-08-01 21:30 闪退", "evidence_resources": [{"path": "/tmp/a.zip"}]})
    proposed = result["shadow"]["proposed_answer"]
    assert "LiveKernelEvent 141 CUDA crash" in proposed
    assert "相机拍摄失败参考资料" not in proposed
    assert result["answer"] == "相机拍摄失败参考资料。"
    assert result["state"]["hypotheses"][0]["state"] == "observed_support"
    assert result["verification"]["passed"] is True
    assert result["state"]["next_tests"][0]["kind"] == "containment"
    assert result["state"]["next_tests"][0]["priority"] < result["state"]["next_tests"][1]["priority"]
    assert all(item["kind"] == "diagnosis" for item in result["state"]["next_tests"][1:2])
    assert "建议立即采取" in proposed


def test_v4_active_mode_uses_compiled_answer_only_after_verification():
    runtime = ReadRuntimeV4(
        baseline=FrozenPipelineProvider(_baseline),
        incident=IncidentProvider(_incident),
        options=ReadRuntimeV4Options(shadow_mode=False, kg_sag_enabled=False, raw_enabled=False),
    )
    result = runtime.run({"query": "设备在 2026-08-01 21:30 闪退", "evidence_resources": [{"path": "/tmp/a.zip"}]})
    assert result["status"] == "ask_info"
    assert "诊断数据中的直接观测" in result["answer"]
    assert result["shadow"]["active_answer_source"] == "read_runtime_v4"


def test_v4_procedure_task_has_procedure_output_contract():
    runtime = ReadRuntimeV4(
        baseline=FrozenPipelineProvider(_baseline),
        options=ReadRuntimeV4Options(shadow_mode=True, kg_sag_enabled=False, raw_enabled=False, incident_enabled=False),
    )
    result = runtime.run({"query": "如何进入安全模式"})
    assert result["task"]["output_contract"] == "procedure_answer"


def test_v4_non_incident_active_mode_preserves_nested_frozen_answer():
    """A wrapper provider must not turn a complete procedure answer into an empty observation."""
    runtime = ReadRuntimeV4(
        baseline=FrozenPipelineProvider(_baseline),
        options=ReadRuntimeV4Options(
            shadow_mode=False,
            kg_sag_enabled=False,
            raw_enabled=False,
            incident_enabled=False,
        ),
    )
    result = runtime.run({"query": "如何进入安全模式"})
    assert result["answer"] == "相机拍摄失败参考资料。"
    assert result["shadow"]["active_answer_source"] == "read_runtime_v4_non_incident_baseline_compat"
    assert result["verification"]["passed"] is True


def test_v4_destructive_action_is_blocked_without_new_top_level_object():
    def incident(_payload):
        value = _incident(_payload)
        value["next_tests"] = [{
            "test_id": "action:reinstall",
            "kind": "remediation",
            "title": "重装系统",
            "instruction": "执行系统重装",
            "risk": "destructive",
            "cost": "high",
            "evidence_required": ["人工授权"],
        }]
        return value

    runtime = ReadRuntimeV4(
        baseline=FrozenPipelineProvider(_baseline),
        incident=IncidentProvider(incident),
        options=ReadRuntimeV4Options(shadow_mode=False, kg_sag_enabled=False, raw_enabled=False),
    )
    result = runtime.run({"query": "设备闪退", "evidence_resources": [{"path": "/tmp/a.zip"}]})
    action = next(item for item in result["state"]["next_tests"] if item["test_id"] == "action:reinstall")
    assert action["status"] == "blocked"
    assert any(item.get("action_id") == "action:reinstall" for item in result["policy"]["blocked_actions"])
    assert "需人工确认的高风险动作" in result["answer"]
