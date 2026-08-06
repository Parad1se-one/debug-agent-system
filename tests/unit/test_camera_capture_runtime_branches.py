from __future__ import annotations

from pathlib import Path

from debug_agent_system.core.config import load_config
from debug_agent_system.runtime.system import DebugAgentSystem


FAMILY_ID = "family:5274d74078aa"
VARIANT_ID = "variant:505989010b74"


def _system(tmp_path: Path) -> DebugAgentSystem:
    config = load_config("config/debug_agent_system.yaml")
    config.session_store = tmp_path / "sessions"
    return DebugAgentSystem(config)


def _session_at_action(
    system: DebugAgentSystem,
    *,
    session_id: str,
    action_label: str,
) -> str:
    system.start({
        "query": "检测界面出现拍照失败问题",
        "interactive": True,
        "session": {"session_id": session_id},
    })
    plan = system.read_model.compile_plan(FAMILY_ID, VARIANT_ID)
    step = next(item for item in plan.steps if item.label == action_label)
    state = system.sessions.get(session_id)
    assert state is not None
    state.status = "step"
    state.required_data = []
    state.failure_type = ""
    state.current_action_id = step.action_id
    state.current_trace_step_id = step.trace_step_id
    state.current_index = step.ordinal - 1
    state.current_check_id = step.action_id
    state.current_check = step.label
    system.sessions.save(state)
    return step.action_id


def _applied_condition_code(
    system: DebugAgentSystem, session_id: str
) -> str:
    state = system.sessions.get(session_id)
    assert state is not None
    branch_ids = state.metadata.get("applied_branch_rule_ids") or []
    assert branch_ids
    branch = system.read_model.get(str(branch_ids[-1]))
    assert branch is not None
    return str(branch.get("condition_code") or "")


def test_runtime_selects_pci_and_m2_from_structured_branch_signals(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)

    _session_at_action(
        system,
        session_id="camera-pci",
        action_label="确认网卡类型和更换条件",
    )
    pci = system.step("camera-pci", "现场确认是 PCI 接口网卡，满足更换条件")
    assert pci["status"] == "ask_info"
    assert pci["current_check"] == "更换PCI接口网卡"
    assert _applied_condition_code(system, "camera-pci") == "pci_nic_confirmed"

    _session_at_action(
        system,
        session_id="camera-m2",
        action_label="确认网卡类型和更换条件",
    )
    m2 = system.step("camera-m2", "现场确认是 M.2 网卡，满足更换条件")
    assert m2["status"] == "ask_info"
    assert m2["current_check"] == "更换M.2网卡"
    assert _applied_condition_code(system, "camera-m2") == "m2_nic_confirmed"


def test_runtime_selects_camera_and_ipc_replacement_from_evidence(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)

    _session_at_action(
        system,
        session_id="camera-hardware",
        action_label="汇总前序证据并判断部件更换条件",
    )
    camera = system.step("camera-hardware", "前序证据明确指向相机硬件")
    assert camera["status"] == "ask_info"
    assert camera["current_check"] == "更换相机"
    assert _applied_condition_code(
        system, "camera-hardware"
    ) == "evidence_points_to_camera"

    _session_at_action(
        system,
        session_id="camera-ipc",
        action_label="汇总前序证据并判断部件更换条件",
    )
    ipc = system.step("camera-ipc", "前序证据明确指向工控机侧")
    assert ipc["status"] == "ask_info"
    assert ipc["current_check"] == "更换工控机"
    assert _applied_condition_code(
        system, "camera-ipc"
    ) == "evidence_points_to_ipc"


def test_runtime_asks_instead_of_guessing_when_branch_signal_is_missing(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    _session_at_action(
        system,
        session_id="camera-ambiguous-nic",
        action_label="确认网卡类型和更换条件",
    )

    out = system.step("camera-ambiguous-nic", "已经确认满足网卡更换条件")

    assert out["status"] == "ask_info"
    assert out["failure_type"] == "branch_condition_required"
    assert "M.2" in out["required_data"][0]
    assert "PCI" in out["required_data"][0]
    state = system.sessions.get("camera-ambiguous-nic")
    assert state is not None
    assert not state.metadata.get("applied_branch_rule_ids")

    clarified = system.step(
        "camera-ambiguous-nic", "补充确认：是 PCI 接口网卡"
    )
    assert clarified["status"] == "ask_info"
    assert clarified["current_check"] == "更换PCI接口网卡"
    assert _applied_condition_code(
        system, "camera-ambiguous-nic"
    ) == "pci_nic_confirmed"


def test_aging_resolves_only_after_duration_no_drop_and_human_confirmation(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    _session_at_action(
        system,
        session_id="camera-aging-complete",
        action_label="执行拍照老化并记录丢帧结果",
    )

    resolved = system.step(
        "camera-aging-complete",
        "老化一小时无丢帧，现场确认问题已解决",
    )

    assert resolved["status"] == "resolved"
    assert _applied_condition_code(
        system, "camera-aging-complete"
    ) == "aging_passed_requires_human_closure"
    verification = resolved["metadata"]["verification"]
    assert verification["outcome_type"] == "verified_fix"
    assert verification["activation_mode"] == "human_confirmed_runtime"
    assert "一小时无丢帧" in verification["runtime_confirmation"]

    _session_at_action(
        system,
        session_id="camera-aging-incomplete",
        action_label="执行拍照老化并记录丢帧结果",
    )
    incomplete = system.step(
        "camera-aging-incomplete",
        "老化一小时无丢帧，问题已解决",
    )
    assert incomplete["status"] == "ask_info"
    assert incomplete["failure_type"] == "branch_condition_required"
