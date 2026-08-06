import ast
from collections import Counter
from pathlib import Path

from debug_agent_system import DebugAgentSystem
from debug_agent_system.core.config import load_config


def test_every_runtime_variant_has_native_trace_actions_and_evidence():
    system = DebugAgentSystem.from_config("config/debug_agent_system.yaml")
    model = system.read_model
    counts = Counter()

    for variant_id, variant in model.by_type["FaultVariant"].items():
        if not model.is_runtime_variant(variant_id):
            counts["support_only"] += 1
            continue
        counts["runtime"] += 1
        plan = model.compile_plan(str(variant["family_id"]), variant_id)
        assert plan.source_type == "DiagnosticTrace"
        assert model.has_object(plan.plan_id, "DiagnosticTrace")
        assert plan.steps
        for step in plan.steps:
            action = model.get(step.action_id)
            assert action is not None
            assert action["variant_id"] == variant_id
            assert action.get("execution_materialize_allowed") is not False
            assert step.evidence_ids
            assert all(model.has_object(item_id, "EvidenceItem") for item_id in step.evidence_ids)
            if step.trace_step_id:
                trace_step = model.get(step.trace_step_id)
                assert trace_step is not None
                assert trace_step["trace_id"] == plan.trace_id
                assert trace_step["action_id"] == step.action_id
            for rule_id in step.branch_rule_ids:
                rule = model.get(rule_id)
                assert rule is not None
                assert rule["from_trace_step_id"] == step.trace_step_id

    assert counts["runtime"] == 98
    assert counts["support_only"] == 19


def test_all_runtime_configs_are_kg_v2_only():
    for path in (
        "config/debug_agent_system.yaml",
        "config/debug_agent_system_sag.yaml",
        "config/debug_agent_system_json.yaml",
    ):
        config = load_config(path)
        assert config.knowledge.store in {"sqlite_sag_v2", "kg_v2_json"}
        assert config.knowledge.kg_v2_root.name == "kg_v2"


def test_runtime_module_has_no_legacy_knowledge_imports():
    path = Path("src/debug_agent_system/runtime/system.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith("debug_agent_system.knowledge.") for name in imports)
    assert "debug_agent_system.knowledge_v2.read_model" in imports
