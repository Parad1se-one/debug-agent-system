from debug_agent_system.knowledge.json_store import JsonKGStore


def test_search_and_lock_camera_case():
    store = JsonKGStore('data/kg')
    candidates = store.search_errors('AOI主程序初始化失败，相机连接异常，请检查相机IP')
    assert candidates
    assert candidates[0].score > 0
    subgraph = store.load_locked_subgraph(candidates[0].error_id)
    assert subgraph.error_id == candidates[0].error_id
    assert subgraph.checks


def test_lock_loads_recursive_next_check_chain():
    store = JsonKGStore('data/kg')
    subgraph = store.load_locked_subgraph('err:industrial-pc-no-boot')
    check_ids = {check.check_id for check in subgraph.checks}
    assert 'check:industrial-pc-no-boot-step1' in check_ids
    assert 'check:industrial-pc-no-boot-step2a' in check_ids
    assert any(check.payload.get('_incoming_relation') == 'next' for check in subgraph.checks)


def test_lock_exposes_structured_next_edges_with_conditions():
    store = JsonKGStore('data/kg')
    subgraph = store.load_locked_subgraph('err:industrial-pc-blue-screen')
    edges = subgraph.next_edges_by_check.get('check:industrial-pc-blue-screen-step2') or []
    targets = {edge['to_check_id']: edge for edge in edges}
    assert 'check:industrial-pc-blue-screen-step2boot' in targets
    assert '0xc0000001' in targets['check:industrial-pc-blue-screen-step2boot']['condition']
    assert all('condition' in edge for edge in edges)


def test_approved_outcomes_recompute_policy_and_do_not_create_non_verified_resolved_by():
    import tempfile
    from pathlib import Path
    tmp = tempfile.TemporaryDirectory()
    kg = Path(tmp.name) / "kg"
    (kg / "review_queue").mkdir(parents=True)
    candidate = {
        "candidate_id": "chatcand:case001-policy",
        "status": "approved",
        "schema_valid": True,
        "nodes": [
            {"type": "Error", "error_id": "err:programming-capture-speed-delay", "label": "编程拍照速度延迟现象", "symptom": "编程拍照速度延迟卡顿", "category": "硬件与运控", "entry_role": "case_variant", "canonical_error_id": "err:camera-capture-failure"},
            {"type": "DiagnosticCheck", "check_id": "check:case001:capture-card", "label": "检查采集卡", "how_to_check": "检查采集卡状态并替换验证", "step_order": 1},
            {"type": "Solution", "solution_id": "sol:case001:replace-capture-card", "content": "更换采集卡", "method": "external", "evidence_level": "case_chat_evidence"},
            {"type": "DiagnosticTrace", "trace_id": "trace:case001", "source_episode_id": "ep:case001", "target_error_id": "err:programming-capture-speed-delay", "recommended_order": [{"check_id": "check:case001:capture-card", "label": "检查采集卡", "order": 1}], "actual_order": [{"check_id": "check:case001:capture-card", "label": "检查采集卡", "order": 1}], "evidence_message_ids": ["m1"]},
            {"type": "DiagnosticOutcome", "outcome_id": "outcome:case001:capture-card", "source_episode_id": "ep:case001", "target_error_id": "err:programming-capture-speed-delay", "target_check_id": "check:case001:capture-card", "target_solution_id": "sol:case001:replace-capture-card", "action_label": "更换采集卡", "outcome_type": "ineffective", "evidence_message_ids": ["m2"]},
        ],
        "diagnostic_outcomes": [
            {"outcome_id": "outcome:case001:capture-card", "target_error_id": "err:programming-capture-speed-delay", "target_check_id": "check:case001:capture-card", "target_solution_id": "sol:case001:replace-capture-card", "action_label": "更换采集卡", "outcome_type": "ineffective", "evidence_message_ids": ["m2"]}
        ],
        "edges": [
            {"from": "err:programming-capture-speed-delay", "to": "check:case001:capture-card", "relation": "has_check"},
            {"from": "err:programming-capture-speed-delay", "to": "trace:case001", "relation": "has_trace"},
            {"from": "err:programming-capture-speed-delay", "to": "outcome:case001:capture-card", "relation": "has_outcome"},
            {"from": "outcome:case001:capture-card", "to": "check:case001:capture-card", "relation": "outcome_check"},
            {"from": "outcome:case001:capture-card", "to": "sol:case001:replace-capture-card", "relation": "outcome_solution"},
        ],
    }
    result = JsonKGStore(kg).apply_approved(candidate)
    assert result["status"] == "applied_to_graph"
    reloaded = JsonKGStore(kg)
    assert "outcome:case001:capture-card" in reloaded.outcomes_by_id
    assert "policy:err:programming-capture-speed-delay" in reloaded.policies_by_id
    assert not any(edge.get("relation") == "resolved_by" for edge in reloaded.edges)
    policy = reloaded.policies_by_id["policy:err:programming-capture-speed-delay"]
    assert policy["type"] == "DiagnosticPolicy"
    assert policy["deterministic_recompute"] is True
    assert policy["solution_stats"][0]["by_outcome_type"]["ineffective"] == 1


def test_policy_prior_keeps_outcome_only_verified_fix_positive():
    import tempfile
    from pathlib import Path
    tmp = tempfile.TemporaryDirectory()
    kg = Path(tmp.name) / "kg"
    (kg / "review_queue").mkdir(parents=True)
    candidate = {
        "candidate_id": "chatcand:outcome-only-fix",
        "status": "approved",
        "schema_valid": True,
        "nodes": [
            {"type": "Error", "error_id": "err:outcome-only-fix", "label": "蓝屏更换内存后未复现", "symptom": "蓝屏", "category": "硬件与运控"},
            {"type": "DiagnosticCheck", "check_id": "check:collect-dmp", "label": "收集DMP", "how_to_check": "收集DMP", "step_order": 1},
            {"type": "DiagnosticCheck", "check_id": "check:replace-memory", "label": "更换内存条", "how_to_check": "更换内存条验证", "step_order": 2},
            {"type": "Solution", "solution_id": "sol:replace-memory", "content": "更换内存条", "method": "replace", "evidence_level": "case_chat_evidence"},
            {"type": "DiagnosticTrace", "trace_id": "trace:outcome-only-fix", "source_episode_id": "ep:outcome-only-fix", "target_error_id": "err:outcome-only-fix", "recommended_order": [{"check_id": "check:collect-dmp", "label": "收集DMP", "order": 1}], "actual_order": [{"check_id": "check:collect-dmp", "label": "收集DMP", "order": 1}], "evidence_message_ids": ["m1"]},
            {"type": "DiagnosticOutcome", "outcome_id": "outcome:replace-memory", "source_episode_id": "ep:outcome-only-fix", "target_error_id": "err:outcome-only-fix", "target_check_id": "check:replace-memory", "target_solution_id": "sol:replace-memory", "action_label": "现场已更换内存条，未再出现蓝屏", "outcome_type": "verified_fix", "evidence_message_ids": ["m2"]},
        ],
        "diagnostic_outcomes": [
            {"outcome_id": "outcome:replace-memory", "target_error_id": "err:outcome-only-fix", "target_check_id": "check:replace-memory", "target_solution_id": "sol:replace-memory", "action_label": "现场已更换内存条，未再出现蓝屏", "outcome_type": "verified_fix", "evidence_message_ids": ["m2"]},
        ],
        "edges": [
            {"from": "err:outcome-only-fix", "to": "check:collect-dmp", "relation": "has_check"},
            {"from": "err:outcome-only-fix", "to": "check:replace-memory", "relation": "has_check"},
            {"from": "err:outcome-only-fix", "to": "trace:outcome-only-fix", "relation": "has_trace"},
            {"from": "err:outcome-only-fix", "to": "outcome:replace-memory", "relation": "has_outcome"},
            {"from": "outcome:replace-memory", "to": "check:replace-memory", "relation": "outcome_check"},
            {"from": "outcome:replace-memory", "to": "sol:replace-memory", "relation": "outcome_solution"},
            {"from": "check:replace-memory", "to": "sol:replace-memory", "relation": "resolved_by"},
        ],
    }
    result = JsonKGStore(kg).apply_approved(candidate)
    assert result["status"] == "applied_to_graph"
    policy = JsonKGStore(kg).policies_by_id["policy:err:outcome-only-fix"]
    by_check = {item["check_id"]: item for item in policy["ordered_checks"]}
    assert by_check["check:replace-memory"]["verified_fix_count"] == 1
    assert by_check["check:replace-memory"]["avg_order"] == 999.0
    assert by_check["check:replace-memory"]["policy_prior"] > 0
    assert policy["ordered_checks"][0]["check_id"] == "check:replace-memory"


def test_case_variant_requires_canonical_or_alias():
    import tempfile
    from pathlib import Path
    tmp = tempfile.TemporaryDirectory()
    kg = Path(tmp.name) / "kg"
    (kg / "review_queue").mkdir(parents=True)
    candidate = {
        "candidate_id": "chatcand:bad-variant",
        "status": "approved",
        "schema_valid": True,
        "nodes": [
            {"type": "Error", "error_id": "err:bad-variant", "label": "孤立变体", "symptom": "孤立变体", "category": "系统与软件异常", "entry_role": "case_variant"},
        ],
        "edges": [],
    }
    result = JsonKGStore(kg).apply_approved(candidate)
    assert result["status"] == "skipped"
    assert result["reason"] == "semantic_schema_invalid"
    assert "case_variant_missing_canonical:err:bad-variant" in result["schema_issues"]


def test_schema_declares_case_experience_extension_contracts():
    import json
    from pathlib import Path
    objects = json.loads(Path("data/kg/schema/object-types.json").read_text())
    links = json.loads(Path("data/kg/schema/link-types.json").read_text())
    for node_type in ("DiagnosticTrace", "DiagnosticOutcome", "DiagnosticPolicy"):
        assert node_type in objects["object_types"]
    req = objects["object_types"]["Error"]["properties"]["required_info_schema"]
    assert set(req["item_required"]) == {"slot", "question", "condition", "blocks", "priority", "why_required", "evidence"}
    outcome_enum = set(objects["object_types"]["DiagnosticOutcome"]["properties"]["outcome_type"]["enum"])
    assert "verified_fix" in outcome_enum
    assert "ineffective" in outcome_enum
    assert "pending_validation" in outcome_enum
    resolved = links["link_types"]["resolved_by"]
    assert "forbidden_outcome_types" in resolved["constraints"]
    assert "ineffective" in resolved["constraints"]["forbidden_outcome_types"]


def test_schema_validator_rejects_non_verified_resolved_by_and_llm_policy_nodes():
    from debug_agent_system.knowledge.schema_validator import validate_candidate
    candidate = {
        "nodes": [
            {"type": "Error", "error_id": "err:v", "label": "变体", "symptom": "变体", "category": "硬件与运控", "entry_role": "case_variant", "canonical_error_id": "err:c"},
            {"type": "DiagnosticCheck", "check_id": "check:v:1", "label": "检查", "how_to_check": "检查", "step_order": 1},
            {"type": "Solution", "solution_id": "sol:v:bad", "content": "更换采集卡无效", "method": "replace", "evidence_level": "ineffective"},
            {"type": "DiagnosticOutcome", "outcome_id": "outcome:v:bad", "source_episode_id": "ep:v", "target_error_id": "err:v", "target_check_id": "check:v:1", "target_solution_id": "sol:v:bad", "action_label": "更换采集卡无效", "outcome_type": "ineffective", "evidence_message_ids": ["m1"]},
            {"type": "DiagnosticPolicy", "policy_id": "policy:err:v", "target_error_id": "err:v", "updated_at": "2026-07-01T00:00:00Z"},
        ],
        "edges": [
            {"from": "err:v", "to": "check:v:1", "relation": "has_check"},
            {"from": "check:v:1", "to": "sol:v:bad", "relation": "resolved_by"},
            {"from": "err:v", "to": "outcome:v:bad", "relation": "has_outcome"},
            {"from": "outcome:v:bad", "to": "sol:v:bad", "relation": "outcome_solution"},
        ],
        "diagnostic_outcomes": [{"target_solution_id": "sol:v:bad", "outcome_type": "ineffective"}],
    }
    issues = validate_candidate(candidate)
    assert "resolved_by_non_verified_solution:sol:v:bad" in issues
    assert "resolved_by_non_verified_outcome:sol:v:bad:ineffective" in issues
    assert "policy_node_must_be_deterministic_recompute:policy:err:v" in issues


def test_required_info_candidate_validator_enforces_slot_evidence_and_target_policy():
    from debug_agent_system.knowledge.schema_validator import validate_required_info_candidate, validate_required_info_schema_item
    issues = validate_required_info_candidate({"slot": "bad", "question": "", "why_required": "", "evidence_message_ids": []})
    assert "required_info_candidate:invalid_slot:bad" in issues
    assert "required_info_candidate:missing_question" in issues
    assert "required_info_candidate:missing_evidence" in issues
    assert "required_info_candidate:missing_target_or_review_only" in issues
    schema_issues = validate_required_info_schema_item({"slot": "log_package", "question": "请提供日志"})
    assert "required_info_schema:missing_evidence" in schema_issues
    assert "required_info_schema:missing_why_required" in schema_issues


def test_w5_approved_merge_is_idempotent_and_policy_not_double_counted():
    import tempfile
    from pathlib import Path
    tmp = tempfile.TemporaryDirectory()
    kg = Path(tmp.name) / "kg"
    (kg / "review_queue").mkdir(parents=True)
    candidate = {
        "candidate_id": "chatcand:idempotent-policy",
        "status": "approved",
        "schema_valid": True,
        "nodes": [
            {"type": "Error", "error_id": "err:idempotent", "label": "幂等测试", "symptom": "幂等测试", "category": "硬件与运控", "entry_role": "case_variant", "canonical_error_id": "err:canonical"},
            {"type": "DiagnosticCheck", "check_id": "check:idempotent:1", "label": "检查采集卡", "how_to_check": "检查采集卡", "step_order": 1},
            {"type": "Solution", "solution_id": "sol:idempotent:1", "content": "更换采集卡无效", "method": "replace", "evidence_level": "ineffective"},
            {"type": "DiagnosticTrace", "trace_id": "trace:idempotent", "source_episode_id": "ep:idempotent", "target_error_id": "err:idempotent", "recommended_order": [{"check_id": "check:idempotent:1", "label": "检查采集卡", "order": 1}], "actual_order": [{"check_id": "check:idempotent:1", "label": "检查采集卡", "order": 1}], "evidence_message_ids": ["m1"]},
            {"type": "DiagnosticOutcome", "outcome_id": "outcome:idempotent:1", "source_episode_id": "ep:idempotent", "target_error_id": "err:idempotent", "target_check_id": "check:idempotent:1", "target_solution_id": "sol:idempotent:1", "action_label": "更换采集卡无效", "outcome_type": "ineffective", "evidence_message_ids": ["m2"]},
        ],
        "diagnostic_outcomes": [{"target_solution_id": "sol:idempotent:1", "outcome_type": "ineffective"}],
        "edges": [
            {"from": "err:idempotent", "to": "err:canonical", "relation": "alias_of"},
            {"from": "err:idempotent", "to": "check:idempotent:1", "relation": "has_check"},
            {"from": "err:idempotent", "to": "trace:idempotent", "relation": "has_trace"},
            {"from": "err:idempotent", "to": "outcome:idempotent:1", "relation": "has_outcome"},
            {"from": "outcome:idempotent:1", "to": "check:idempotent:1", "relation": "outcome_check"},
            {"from": "outcome:idempotent:1", "to": "sol:idempotent:1", "relation": "outcome_solution"},
        ],
    }
    store = JsonKGStore(kg)
    first = store.apply_approved(candidate)
    second = JsonKGStore(kg).apply_approved(candidate)
    assert first["status"] == "applied_to_graph"
    assert second["status"] == "already_applied"
    reloaded = JsonKGStore(kg)
    policy = reloaded.policies_by_id["policy:err:idempotent"]
    assert policy["deterministic_recompute"] is True
    assert policy["solution_stats"][0]["by_outcome_type"]["ineffective"] == 1
    assert len(reloaded.outcomes) == 1


def test_read_side_policy_prior_and_already_tried_penalty_affect_order():
    from debug_agent_system.agents.read.bd_traversal import TopologyTraversalAgent
    from debug_agent_system.core.contracts import CheckNode, LockedSubgraph, SessionState
    checks = [
        CheckNode("check:capture-card", "检查采集卡", "更换采集卡并验证", 1, payload={"_historical_outcomes": [{"action_label": "更换采集卡", "outcome_type": "ineffective"}]}),
        CheckNode("check:camera", "检查相机组件", "检查相机组件和相机日志", 2),
    ]
    subgraph = LockedSubgraph(
        error_id="err:case001",
        label="编程拍照速度延迟现象",
        checks=checks,
        payload={"_diagnostic_policy": {"ordered_checks": [{"check_id": "check:capture-card", "policy_prior": 4.0}]}},
    )
    agent = TopologyTraversalAgent()
    state1 = SessionState("s1", "编程拍照速度延迟")
    first = agent.first_step(state1, subgraph)
    assert first.check and first.check.check_id == "check:capture-card"
    state2 = SessionState("s2", "编程拍照速度延迟，更换采集卡无效")
    second = agent.first_step(state2, subgraph)
    assert second.check and second.check.check_id == "check:camera"
    checks = [
        CheckNode("check:cxp", "检查CXP线", "检查CXP线连接并更换验证", 1),
        CheckNode("check:camera", "检查相机组件", "检查相机组件和相机日志", 2),
    ]
    subgraph = LockedSubgraph(error_id="err:case001", label="编程拍照速度延迟现象", checks=checks)
    state3 = SessionState("s3", "编程拍照速度延迟，更换CXP线失败，怀疑相机组件问题")
    third = agent.first_step(state3, subgraph)
    assert third.check and third.check.check_id == "check:camera"
