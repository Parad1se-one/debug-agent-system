import json
from pathlib import Path

from debug_agent_system import DebugAgentSystem
from debug_agent_system.adapters.qa_supervisor import DebugAgentSystemQARuntime


CONFIG = "config/debug_agent_system.yaml"


def _system() -> DebugAgentSystem:
    return DebugAgentSystem.from_config(CONFIG)


def test_runtime_primary_identity_candidates_plan_and_evidence_are_kg_v2():
    system = _system()
    out = system.start({
        "query": "AOI主程序初始化失败，提示加载用户配置失败，user.cfg.toml为空。",
        "session": {"session_id": "runtime-v2-primary-contract"},
    })

    assert out["schema_version"] == "debug_agent_system.response.v2"
    assert out["family_id"].startswith("family:")
    assert out["variant_id"].startswith("variant:")
    assert out["plan_id"].startswith(("trace:", "policy:", "variant:"))
    assert out["current_action_id"].startswith("action:")
    assert out["current_check_id"] == out["current_action_id"]
    assert out["observability"]["top_error_id"] == out["variant_id"]

    candidates = out["metadata"]["retrieval"]["candidates"]
    assert candidates
    assert all(item["family_id"].startswith("family:") for item in candidates)
    assert all(item["variant_id"].startswith("variant:") for item in candidates)
    assert all("error_id" not in item for item in candidates)

    plan = out["metadata"]["diagnostic_plan"]
    assert plan["family_id"] == out["family_id"]
    assert plan["variant_id"] == out["variant_id"]
    assert plan["plan_id"] == out["plan_id"]
    assert plan["steps"]
    assert all(step["action_id"].startswith("action:") for step in plan["steps"])
    assert all("check_id" not in step for step in plan["steps"])

    evidence_index = system.read_model.by_type["EvidenceItem"]
    assert out["evidence_ids"]
    assert all(evidence_id in evidence_index for evidence_id in out["evidence_ids"])
    assert {item["evidence_id"] for item in out["metadata"]["evidence"]} == set(out["evidence_ids"])
    assert out["metadata"]["runtime_invariants"]["legacy_graph_used"] is False


def test_sag_v2_is_revision_pinned_and_returns_native_candidate_paths():
    system = _system()
    out = system.start({
        "query": "2D相机拍摄失败，提示操作失败并出现残帧。",
        "session": {"session_id": "runtime-v2-sag-path"},
    })
    retrieval = out["metadata"]["retrieval"]
    assert retrieval["backend"] == "sqlite_sag_v2"
    assert len(retrieval["graph_revision"]) == 64
    assert retrieval["candidates"][0]["route"] == "sag_v2_native"
    paths = retrieval["candidates"][0]["retrieval_paths"]
    assert paths
    assert all(path["variant_id"].startswith("variant:") for path in paths)
    assert Path(system.config.knowledge.kg_v2_sqlite_path).exists()


def test_exact_error_code_beats_generic_blue_screen_variants():
    system = _system()
    out = system.start({
        "query": "工控机运行中蓝屏，BugCheck错误代码0x00000139，已提供系统日志。",
        "session": {"session_id": "runtime-v2-exact-code"},
    })
    assert out["variant_id"] == "variant:family::77f1b67eafb3:0x00000139-:5d7bc980660a"
    assert out["family_id"] == "family::77f1b67eafb3"


def test_step_progression_uses_only_actions_from_locked_plan():
    system = _system()
    first = system.start({
        "query": "主程序加载用户配置失败，报错提示user.cfg.toml异常。",
        "session": {"session_id": "runtime-v2-plan-progression"},
    })
    plan_action_ids = first["metadata"]["plan_action_ids"]
    second = system.step(first["session_id"], "已检查但仍未解决")
    assert second["status"] in {"step", "ask_info"}
    assert second["current_action_id"] in plan_action_ids
    assert second["current_action_id"] != first["current_action_id"]
    assert all(action_id.startswith("action:") for action_id in second["metadata"]["plan_action_ids"])
    if second["status"] == "ask_info":
        assert second["metadata"]["pending_confirmation_action_id"] == second["current_action_id"]


def test_resolution_requires_verified_fix_outcome_and_evidence():
    system = _system()
    out = system.start({
        "query": "更换工控机后user.cfg.toml为空或损坏导致配置加载失败",
        "session": {"session_id": "runtime-v2-verified-resolution"},
    })
    for message in (
        "已收集但未解决",
        "检查后仍未解决",
        "确认配置损坏但未解决",
    ):
        out = system.step(out["session_id"], message)
        assert out["status"] == "step"

    out = system.step(out["session_id"], "使用最近诊断日志中的user.cfg.toml后已解决，恢复正常")
    assert out["status"] == "resolved"
    verification = out["metadata"]["verification"]
    assert verification["outcome_type"] == "verified_fix"
    assert verification["outcome_id"].startswith("outcome:")
    assert verification["action_id"] == out["current_action_id"]
    assert verification["evidence_ids"]
    assert out["metadata"]["applied_branch_rule_ids"]
    assert all(rule_id.startswith("branch-rule:") for rule_id in out["metadata"]["applied_branch_rule_ids"])


def test_user_says_solved_without_kg_verified_fix_stays_pending():
    system = _system()
    first = system.start({
        "query": "AOI主程序初始化失败，加载用户配置失败。",
        "session": {"session_id": "runtime-v2-fail-closed-resolution"},
    })
    out = system.step(first["session_id"], "已经解决，恢复正常")
    assert out["status"] == "step"
    assert out["failure_type"] == "pending_validation"
    assert out["metadata"]["verification"]["supported"] is False
    assert out["metadata"]["verification"]["required_outcome_type"] == "verified_fix"


def test_destructive_action_requires_explicit_human_confirmation():
    system = _system()
    first = system.start({
        "query": "光源初始化异常，退出软件并断电后重启",
        "session": {"session_id": "runtime-v2-safety-confirm"},
    })
    assert first["status"] == "ask_info"
    assert first["current_action_id"].startswith("action:")
    assert first["metadata"]["pending_confirmation_action_id"] == first["current_action_id"]

    repeated = system.step(first["session_id"], "暂时不确定")
    assert repeated["status"] == "ask_info"
    confirmed = system.step(first["session_id"], "人工已批准，可以执行")
    assert confirmed["status"] == "step"
    assert confirmed["current_action_id"] == first["current_action_id"]


def test_required_info_questions_and_support_are_kg_v2_objects():
    system = _system()
    first = system.start({
        "query": "主程序加载用户配置失败，当前缺少软件版本和完整报错。",
        "session": {"session_id": "runtime-v2-required-info"},
    })
    assert first["status"] == "ask_info"
    required_ids = first["metadata"]["required_info_ids"]
    assert required_ids
    assert all(system.read_model.has_object(item_id, "RequiredInfoSpec") for item_id in required_ids)
    assert all(system.read_model.has_object(item_id, "EvidenceItem") for item_id in first["evidence_ids"])

    second = system.step(first["session_id"], "已提供版本0.27.42，完整报错为加载用户配置失败。")
    assert second["status"] in {"step", "ask_info"}
    assert second["variant_id"].startswith("variant:")


def test_qa_adapter_preserves_v2_runtime_payload():
    runtime = DebugAgentSystemQARuntime(CONFIG)
    out = runtime.answer("2D相机拍照失败怎么排查？", [], {"session_id": "runtime-v2-qa-adapter"})
    assert out["backend"] == "debug_agent_system"
    assert out["agent"] == "debug_agent"
    assert out["observations"]
    observation = next(item for item in out["observations"] if item["type"] == "debug_agent_system")
    response = json.loads(observation["content"])
    assert response["variant_id"].startswith("variant:")
    assert response["current_action_id"].startswith("action:")


def test_session_fails_closed_when_pinned_kg_revision_changes():
    system = _system()
    first = system.start({
        "query": "主程序加载用户配置失败",
        "session": {"session_id": "runtime-v2-revision-pin"},
    })
    state = system.sessions.get(first["session_id"])
    assert state is not None
    state.metadata["graph_revision"] = "stale-revision"
    system.sessions.save(state)

    out = system.step(first["session_id"], "继续")
    assert out["status"] == "failed"
    assert out["failure_type"] == "invalid_kg_v2_session"
