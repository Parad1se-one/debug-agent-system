from __future__ import annotations

from pathlib import Path

from debug_agent_system.kg_raw_codex.pipeline import CorpusReadTools
from debug_agent_system.kg_raw_codex.terminology_harness import (
    execute_terminology_search_contract,
)
from debug_agent_system.knowledge_v2.terminology import TerminologyResolver
from debug_agent_system.kg_raw_codex.terminology_contract import (
    build_terminology_search_contract,
)


def test_harness_executes_every_required_term_and_records_zero_hits() -> None:
    tools = CorpusReadTools(Path.cwd())
    execution = execute_terminology_search_contract({
        "required_search_groups": [{
            "source_surface_form": "旧称",
            "canonical_name": "规范名",
            "required_terms": ["旧称", "不存在的规范名"],
        }],
    }, tools, path_glob="data/kg_v2/terminology/*.json")

    assert execution["task_count"] == 2
    assert [task["term"] for task in execution["tasks"]] == [
        "旧称",
        "不存在的规范名",
    ]
    assert len(execution["tool_trace"]) == 2
    assert all(
        trace["origin"] == "deterministic_terminology_harness"
        for trace in execution["tool_trace"]
    )
    assert execution["results"][1]["returned"] == 0


def test_ambiguous_candidates_are_searchable_but_non_locking() -> None:
    resolver = TerminologyResolver.from_root(Path("data/kg_v2"))
    resolution = resolver.resolve("运控卡故障")
    contract = build_terminology_search_contract(
        "运控卡故障",
        resolution,
    )

    groups = [
        group for group in contract["required_search_groups"]
        if group.get("resolution_status") == "ambiguous"
    ]
    assert groups
    assert any(
        group["required_terms"] == ["运控卡", "运动控制卡"]
        and group["can_lock_variant"] is False
        for group in groups
    )


def test_context_policy_blocks_incompatible_noun_expansions() -> None:
    resolver = TerminologyResolver.from_root(Path("data/kg_v2"))

    protocol = resolver.resolve("USB协议栈枚举失败")
    assert not any(
        mention["concept"].get("canonical_name") == "USB接口"
        for mention in protocol["resolved_mentions"]
    )
    assert any(
        item["canonical_name"] == "USB接口"
        for item in protocol["safety"]["blocked_expansions"]
    )

    device = resolver.resolve("USB设备问题")
    assert any(
        mention["concept"].get("canonical_name") == "USB接口"
        for mention in device["resolved_mentions"]
    )

    business = resolver.resolve("驱动业务增长的因素")
    assert not any(
        mention["concept"].get("canonical_name") == "设备驱动程序"
        for mention in business["resolved_mentions"]
    )
