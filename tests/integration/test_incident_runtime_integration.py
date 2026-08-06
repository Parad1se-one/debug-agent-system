from __future__ import annotations

from pathlib import Path

from debug_agent_system.adapters.codex_read import CodexReadSideToolExecutor
from debug_agent_system.core.config import load_config
from debug_agent_system.runtime import DebugAgentSystem


LOG = """[2026-08-01 21:30:00,093] ERROR SYMV(1.1.0) subsystem error (-217): CUDA illegal memory access
in cv::cuda::GpuMat::upload, file C:\\build\\opencv\\gpu_mat.cu, line 240, trace:
0# 18b97e5 7ffbcb6797e5 symv::cuda_init in symv
1# 1ba15c8 7ffbcb9615c8 symv::template_match in symv
"""


def _system(tmp_path: Path, *, enabled: bool, shadow_mode: bool = True) -> DebugAgentSystem:
    config = load_config()
    config.session_store = tmp_path / ("enabled" if enabled else "disabled")
    config.incident_runtime.enabled = enabled
    config.incident_runtime.shadow_mode = shadow_mode
    config.read_llm.enabled = False
    return DebugAgentSystem(config)


def _resource() -> dict[str, object]:
    return {
        "resource_id": "resource:cuda-log",
        "kind": "log_package",
        "name": "symv.log",
        "text": LOG,
        "metadata": {},
    }


def test_disabled_incident_runtime_preserves_baseline_response(tmp_path: Path) -> None:
    out = _system(tmp_path, enabled=False).start({
        "query": "GPU 报错怎么排查",
        "interactive": False,
        "evidence_resources": [_resource()],
    })

    assert "incident_runtime" not in out["metadata"]
    assert out["schema_version"] == "debug_agent_system.response.v2"


def test_enabled_shadow_and_active_answer_modes(tmp_path: Path) -> None:
    shadow = _system(tmp_path, enabled=True, shadow_mode=True).start({
        "query": "GPU 报错怎么排查",
        "interactive": False,
        "evidence_resources": [_resource()],
    })
    incident = shadow["metadata"]["incident_runtime"]
    assert incident["evidence_pack"]["schema_version"] == "debug_agent_system.incident_evidence_pack.v3"
    assert incident["observability"]["canonical_kg_mutated"] is False

    active = _system(tmp_path, enabled=True, shadow_mode=False).start({
        "query": "GPU 报错怎么排查",
        "interactive": False,
        "evidence_resources": [_resource()],
    })
    assert active["answer"].startswith("# Jira 诊断分析")
    assert active["metadata"]["incident_runtime"]["active_answer"] is True


def test_codex_incident_tools_iterate_over_one_immutable_case(tmp_path: Path) -> None:
    executor = CodexReadSideToolExecutor(_system(tmp_path, enabled=False))
    analyzed = executor.execute(
        "analyze_incident",
        {"query": "GPU 报错怎么排查", "evidence_resources": [_resource()], "log_summary": ""},
    )
    case_id = analyzed["case"]["case_id"]

    pack = executor.execute("get_incident_evidence_pack", {"case_id": case_id})
    events = executor.execute(
        "search_diagnostic_events",
        {"case_id": case_id, "query": "-217", "limit": 20},
    )
    report = executor.execute("render_incident_report", {"case_id": case_id})
    scope = executor.execute("get_incident_scope", {"case_id": case_id})
    reproduction = executor.execute("plan_reproduction", {"case_id": case_id})
    comparison = executor.execute(
        "compare_reproduction_runs",
        {"baseline_case_id": case_id, "candidate_case_id": case_id},
    )

    assert pack["schema_version"] == "debug_agent_system.incident_evidence_pack.v3"
    assert events["events"]
    assert report["answer"].startswith("# Jira 诊断分析")
    assert report["jira_mutated"] is False
    assert scope["schema_version"] == "debug_agent_system.incident_scope.v1"
    assert reproduction["automatic_device_control"] is False
    assert comparison["signature_reproduced"] is True
    assert comparison["controlled_reproduction"] is False
    assert comparison["fix_verified"] is False


def test_codex_can_parse_reference_scope_before_analyzing(tmp_path: Path) -> None:
    executor = CodexReadSideToolExecutor(_system(tmp_path, enabled=False))
    parsed = executor.execute(
        "parse_incident_scope",
        {
            "query": "参考时间：8月1日21：30，8月3日6：04",
            "resource_hints": ["DLOG_[20260801-20260803].zip"],
        },
    )

    assert parsed["status"] == "ok"
    assert parsed["scope"]["time_semantics"] == "independent_points"
    assert len(parsed["scope"]["reference_windows"]) == 2
