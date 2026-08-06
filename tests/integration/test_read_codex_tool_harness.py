from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.adapters.codex_read import (
    CodexReadSideToolExecutor,
    CodexReadToolHarness,
    read_side_tool_schemas,
)
from debug_agent_system.core.config import load_config
from debug_agent_system.runtime import DebugAgentSystem


class MockToolClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = list(messages)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assert tools
        return self.messages.pop(0)


def _system(tmp_path: Path) -> DebugAgentSystem:
    config = load_config()
    config.session_store = tmp_path / "sessions"
    config.read_llm.enabled = True
    config.read_llm.answer_composer_enabled = False
    config.read_llm.max_tool_rounds = 2
    return DebugAgentSystem(config)


def _resource(path: Path) -> dict[str, Any]:
    return {
        "resource_id": "resource:camera-log",
        "kind": "log_package",
        "name": path.name,
        "path": str(path),
        "url": "",
        "text": "",
        "mime": "text/plain",
        "size": path.stat().st_size,
        "sha256": "",
        "source_message_id": "message:camera-log",
        "metadata": {},
    }


def test_codex_read_surface_has_eight_bounded_read_tools(
    tmp_path: Path,
) -> None:
    executor = CodexReadSideToolExecutor(_system(tmp_path))
    names = {
        item["function"]["name"] for item in read_side_tool_schemas()
    }

    assert names == executor.allowed_tools
    assert {
        "expand_document_context",
        "inspect_kg_path",
        "inspect_source_assets",
        "render_evidence_answer",
    } <= names
    assert "select_branch" not in names
    assert "mark_resolved" not in names
    assert "execute_action" not in names


def test_render_evidence_answer_uses_closed_pack_and_local_verifier(
    tmp_path: Path,
) -> None:
    executor = CodexReadSideToolExecutor(_system(tmp_path))
    response = executor.execute(
        "diagnose_start",
        {
            "query": "如何进入安全模式",
            "interactive": False,
            "session_id": "codex-render-test",
            "routing_context": {
                "stage": "knowledge",
                "query_type": "knowledge_lookup",
                "interface": "",
                "side": "",
            },
            "evidence_resources": [],
        },
    )
    pack = response["metadata"]["evidence_pack"]
    grouped: dict[str, list[str]] = {}
    for item in pack["source_items"]:
        grouped.setdefault(item["original_section_type"], []).append(
            item["item_id"]
        )
    order = [
        "known",
        "diagnostic_steps",
        "document_guidance",
        "conditions",
        "uncertainty",
        "required_info",
    ]
    rendered = executor.execute(
        "render_evidence_answer",
        {
            "session_id": response["session_id"],
            "answer_sections": [
                {
                    "section_type": section_type,
                    "source_item_ids": grouped[section_type],
                }
                for section_type in order
                if grouped.get(section_type)
            ],
            "covered_query_facets": pack["query_scope"][
                "supported_facets"
            ],
            "uncovered_query_facets": pack["query_scope"][
                "unsupported_facets"
            ],
        },
    )

    assert rendered["status"] == response["status"]
    assert rendered["metadata"]["answer_composer"]["provider"] == "codex"
    assert rendered["metadata"]["answer_composer"]["used"] is True
    assert rendered["answer"]
    assert rendered["answer"] != "rejected"


def test_harness_reminds_codex_to_close_answer_through_renderer(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path)
    system.config.read_llm.max_tool_rounds = 3
    executor = CodexReadSideToolExecutor(system)
    diagnosis = executor.execute(
        "diagnose_start",
        {
            "query": "如何进入安全模式",
            "interactive": False,
            "session_id": "codex-harness-render-test",
            "routing_context": {
                "stage": "knowledge",
                "query_type": "knowledge_lookup",
                "interface": "",
                "side": "",
            },
            "evidence_resources": [],
        },
    )
    pack = diagnosis["metadata"]["evidence_pack"]
    grouped: dict[str, list[str]] = {}
    for item in pack["source_items"]:
        grouped.setdefault(item["original_section_type"], []).append(
            item["item_id"]
        )
    answer_sections = [
        {
            "section_type": section_type,
            "source_item_ids": item_ids,
        }
        for section_type, item_ids in grouped.items()
    ]
    client = MockToolClient([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-diagnose",
                "type": "function",
                "function": {
                    "name": "diagnose_start",
                    "arguments": json.dumps({
                        "query": "如何进入安全模式",
                        "interactive": False,
                        "session_id": "codex-harness-render-test",
                        "routing_context": {
                            "stage": "knowledge",
                            "query_type": "knowledge_lookup",
                            "interface": "",
                            "side": "",
                        },
                        "evidence_resources": [],
                    }, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "assistant",
            "content": "先直接回答一段自由文本。",
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-render",
                "type": "function",
                "function": {
                    "name": "render_evidence_answer",
                    "arguments": json.dumps({
                        "session_id": diagnosis["session_id"],
                        "answer_sections": answer_sections,
                        "covered_query_facets": pack["query_scope"][
                            "supported_facets"
                        ],
                        "uncovered_query_facets": pack["query_scope"][
                            "unsupported_facets"
                        ],
                    }, ensure_ascii=False),
                },
            }],
        },
    ])

    result = CodexReadToolHarness(system, client=client).run(
        "如何进入安全模式",
        interactive=False,
    )

    harness = result["metadata"]["codex_tool_harness"]
    assert harness["canonical_render_used"] is True
    assert harness["reason"] == "tool_controlled_canonical_answer"
    assert result["metadata"]["answer_composer"]["used"] is True
    assert [item["tool"] for item in harness["tool_calls"]] == [
        "diagnose_start",
        "render_evidence_answer",
    ]


def test_document_and_kg_inspection_tools_are_source_bounded(
    tmp_path: Path,
) -> None:
    executor = CodexReadSideToolExecutor(_system(tmp_path))
    retrieval = executor.execute(
        "retrieve_evidence",
        {"query": "检测界面出现拍照失败问题", "limit": 5},
    )
    assert retrieval["status"] == "ok"
    assert retrieval["supporting_chunks"]
    documents = list(dict.fromkeys(
        str(item.get("document_id") or "")
        for item in retrieval["supporting_chunks"]
        if str(item.get("document_id") or "")
    ))
    expanded = executor.execute(
        "expand_document_context",
        {
            "query": "检测界面出现拍照失败问题",
            "document_ids": documents[:2],
            "max_chunks": 64,
        },
    )
    assert expanded["status"] == "ok"
    assert all(
        item.get("approved") is not False
        for item in expanded["chunks"]
    )
    candidate = retrieval["candidates"][0]
    path = executor.execute(
        "inspect_kg_path",
        {
            "family_id": candidate["family_id"],
            "variant_id": candidate["variant_id"],
        },
    )
    assert path["status"] == "ok"
    assert path["safety_contract"] == {
        "tool_is_read_only": True,
        "branch_selected": False,
        "action_executed": False,
        "verified_fix_asserted": False,
    }


def test_mock_codex_can_select_parser_and_resume_diagnosis(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "camera.log"
    log_path.write_text(
        "检测界面 camera capture failed E1005 拍照失败\n",
        encoding="utf-8",
    )
    resource = _resource(log_path)
    client = MockToolClient([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-diagnose",
                "type": "function",
                "function": {
                    "name": "diagnose_start",
                    "arguments": json.dumps({
                        "query": "现场出现问题",
                        "interactive": True,
                        "session_id": "",
                        "routing_context": {
                            "stage": "diagnosis",
                            "query_type": "debug_issue",
                            "interface": "",
                            "side": "",
                        },
                        "evidence_resources": [],
                    }, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-parse",
                "type": "function",
                "function": {
                    "name": "parse_evidence",
                    "arguments": json.dumps({
                        "tool": "log_package",
                        "resource": resource,
                        "max_bytes": 65536,
                    }, ensure_ascii=False),
                },
            }],
        },
    ])

    result = CodexReadToolHarness(
        _system(tmp_path), client=client
    ).run("现场出现问题", evidence_resources=[resource])

    harness = result["metadata"]["codex_tool_harness"]
    assert harness["enabled"] is True
    assert harness["fallback_used"] is False
    assert [item["tool"] for item in harness["tool_calls"]] == [
        "diagnose_start",
        "parse_evidence",
        "evidence_gap_resume",
    ]
    assert result["metadata"]["evidence_gap_resolution"]["attempted"] is True
    assert "只读工具观察" in result["answer"]
