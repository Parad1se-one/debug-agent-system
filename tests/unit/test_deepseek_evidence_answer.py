from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.knowledge_v2.query_scope import analyze_query_task
from debug_agent_system.runtime import DebugAgentSystem


class CompleteOrganizer:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls += 1
        pack = json.loads(messages[-1]["content"])
        grouped: dict[str, list[str]] = {}
        order: list[str] = []
        for item in pack["source_items"]:
            section_type = item["original_section_type"]
            if section_type not in grouped:
                grouped[section_type] = []
                order.append(section_type)
            grouped[section_type].append(item["item_id"])
        return {
            "schema_version": "debug_agent_system.llm_answer_composition.v2",
            "answer_sections": [
                {
                    "section_type": section_type,
                    "source_item_ids": grouped[section_type],
                }
                for section_type in order
            ],
            "covered_query_facets": pack["query_scope"]["supported_facets"],
            "uncovered_query_facets": pack["query_scope"]["unsupported_facets"],
        }


class UnknownReferenceOrganizer:
    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pack = json.loads(messages[-1]["content"])
        first = pack["source_items"][0]
        return {
            "schema_version": "debug_agent_system.llm_answer_composition.v2",
            "answer_sections": [{
                "section_type": first["original_section_type"],
                "source_item_ids": ["answer-item:not-in-pack"],
            }],
            "covered_query_facets": pack["query_scope"]["supported_facets"],
            "uncovered_query_facets": pack["query_scope"]["unsupported_facets"],
        }


class RequiredOnlyOrganizer:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls += 1
        pack = json.loads(messages[-1]["content"])
        grouped: dict[str, list[str]] = {}
        for item in pack["source_items"]:
            if item["selection_class"] != "required":
                continue
            grouped.setdefault(item["original_section_type"], []).append(
                item["item_id"]
            )
        return {
            "schema_version": "debug_agent_system.llm_answer_composition.v2",
            "answer_sections": [
                {"section_type": section_type, "source_item_ids": item_ids}
                for section_type, item_ids in grouped.items()
            ],
            "covered_query_facets": pack["query_scope"]["supported_facets"],
            "uncovered_query_facets": pack["query_scope"]["unsupported_facets"],
        }


class ContentRegroupingOrganizer:
    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pack = json.loads(messages[-1]["content"])
        grouped: dict[str, list[str]] = {}
        for item in pack["source_items"]:
            original = item["original_section_type"]
            target = (
                original
                if original in {"uncertainty", "required_info"}
                else "document_guidance"
            )
            grouped.setdefault(target, []).append(item["item_id"])
        order = [
            "document_guidance",
            "uncertainty",
            "required_info",
        ]
        return {
            "schema_version": "debug_agent_system.llm_answer_composition.v2",
            "answer_sections": [
                {
                    "section_type": section_type,
                    "source_item_ids": grouped[section_type],
                }
                for section_type in order
                if grouped.get(section_type)
            ],
            "covered_query_facets": pack["query_scope"]["supported_facets"],
            "uncovered_query_facets": pack["query_scope"][
                "unsupported_facets"
            ],
        }


def _config(tmp_path: Path):
    config = load_config()
    config.session_store = tmp_path / "sessions"
    config.read_llm.enabled = True
    config.read_llm.answer_composer_enabled = True
    return config


def test_compound_query_is_composed_once_with_complete_facet_closure(
    tmp_path: Path,
) -> None:
    client = CompleteOrganizer()
    result = DebugAgentSystem(
        _config(tmp_path),
        answer_model_client=client,
    ).start({
        "query": "显卡驱动持续异常时，如何先彻底卸载旧驱动，再重新安装显卡驱动？",
        "interactive": False,
    })

    assert client.calls == 1
    assert result["metadata"]["answer_composer"]["used"] is True
    assert result["metadata"]["answer_composer"]["call_count"] == 1
    assert result["metadata"]["answer_coverage"]["query_facets_complete"] is True
    assert set(
        result["metadata"]["answer_coverage"]["supported_query_facets"]
    ) == {"operation:卸载", "operation:安装"}
    assert "卸载" in result["answer"]
    assert "安装" in result["answer"]
    assert result["status"] == "step"


def test_unknown_reference_fails_open_to_deterministic_answer(
    tmp_path: Path,
) -> None:
    result = DebugAgentSystem(
        _config(tmp_path),
        answer_model_client=UnknownReferenceOrganizer(),
    ).start({
        "query": "如何进入安全模式",
        "interactive": False,
    })

    composer = result["metadata"]["answer_composer"]
    assert composer["used"] is False
    assert composer["fallback_used"] is True
    assert composer["fallback_reason"] == "verification_failed"
    assert any(
        "unknown_source_item_ids" in error
        for error in composer["verification_errors"]
    )
    assert "安全模式" in result["answer"]


def test_missing_api_key_fails_open_without_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config = _config(tmp_path)
    # Do not read the developer's project-local credentials in this negative
    # contract test.
    config.root = tmp_path
    result = DebugAgentSystem(config).start({
        "query": "如何进入安全模式",
        "interactive": False,
    })

    composer = result["metadata"]["answer_composer"]
    assert composer["used"] is False
    assert composer["attempted"] is True
    assert composer["call_count"] == 1
    assert "missing_OPENAI_API_KEY" in composer["fallback_reason"]
    assert "安全模式" in result["answer"]


def test_optional_items_may_be_dropped_without_losing_facet_closure(
    tmp_path: Path,
) -> None:
    client = RequiredOnlyOrganizer()
    result = DebugAgentSystem(
        _config(tmp_path),
        answer_model_client=client,
    ).start({
        "query": "USB设备问题如何排查？",
        "interactive": False,
    })

    pack = result["metadata"]["evidence_pack"]
    assert pack["schema_version"] == "debug_agent_system.answer_evidence_pack.v2"
    assert pack["selection_policy"]["required_item_ids"]
    assert pack["selection_policy"]["optional_item_ids"]
    assert client.calls == 1
    assert result["metadata"]["answer_composer"]["used"] is True
    assert result["metadata"]["answer_coverage"]["evidence_floor_met"] is True


def test_grounded_content_can_be_regrouped_but_control_sections_stay_fixed(
    tmp_path: Path,
) -> None:
    result = DebugAgentSystem(
        _config(tmp_path),
        answer_model_client=ContentRegroupingOrganizer(),
    ).start({
        "query": "电脑不开机，应该怎么排查？",
        "interactive": False,
    })

    assert result["metadata"]["answer_composer"]["used"] is True
    section_types = [
        section["section_type"] for section in result["answer_sections"]
    ]
    assert section_types[0] == "document_guidance"
    assert section_types[-1] == "sources"
    assert section_types.index("uncertainty") < section_types.index(
        "required_info"
    )


def test_no_grounded_evidence_never_calls_answer_model(
    tmp_path: Path,
) -> None:
    client = CompleteOrganizer()
    result = DebugAgentSystem(
        _config(tmp_path),
        answer_model_client=client,
    ).start({
        "query": "技嘉 B760 GAMING X DDR4 主板如何核对型号？",
        "interactive": False,
    })

    assert client.calls == 0
    assert result["metadata"]["answer_coverage"]["evidence_floor_met"] is False
    assert result["metadata"]["answer_coverage"]["query_facets_complete"] is False
    assert (
        result["metadata"]["answer_composer"]["fallback_reason"]
        == "no_approved_grounded_content"
    )


def test_query_task_model_preserves_conditions_sequence_and_deliverable() -> None:
    task = analyze_query_task(
        "Windows 启动异常时，可以进入系统和无法进入系统分别如何先修复系统，"
        "再修复引导？"
    )

    assert task["deliverable"] == "comparison"
    assert task["sequence"]
    assert {
        item["condition_id"] for item in task["conditions"]
    } >= {"can_enter_system", "cannot_enter_system"}
    assert {"修复"} <= set(task["operations"])
    assert {
        facet["kind"] for facet in task["facets"]
    } >= {"operation", "condition"}
