from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.adapters.deepseek_read import (
    DeepSeekReadToolHarness,
    ReadSideToolExecutor,
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


def _system(tmp_path: Path) -> DebugAgentSystem:
    config = load_config()
    config.session_store = tmp_path / "sessions"
    config.read_llm.enabled = True
    config.read_llm.max_tool_rounds = 2
    return DebugAgentSystem(config)


def test_read_tool_surface_has_only_bounded_tools(tmp_path: Path) -> None:
    executor = ReadSideToolExecutor(_system(tmp_path))
    names = {
        item["function"]["name"]
        for item in read_side_tool_schemas()
    }
    assert names == executor.allowed_tools
    assert "select_branch" not in names
    assert "mark_resolved" not in names
    assert "execute_action" not in names


def test_mock_deepseek_can_select_parser_and_resume_diagnosis(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "camera.log"
    log_path.write_text(
        "检测界面 camera capture failed E1005 拍照失败\n",
        encoding="utf-8",
    )
    resource = _resource(log_path)
    client = MockToolClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-diagnose",
                        "type": "function",
                        "function": {
                            "name": "diagnose_start",
                            "arguments": json.dumps(
                                {
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
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-parse",
                        "type": "function",
                        "function": {
                            "name": "parse_evidence",
                            "arguments": json.dumps(
                                {
                                    "tool": "log_package",
                                    "resource": resource,
                                    "max_bytes": 65536,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
        ]
    )
    result = DeepSeekReadToolHarness(
        _system(tmp_path),
        client=client,
    ).run("现场出现问题", evidence_resources=[resource])
    harness = result["metadata"]["deepseek_tool_harness"]
    assert harness["enabled"] is True
    assert harness["fallback_used"] is False
    assert [item["tool"] for item in harness["tool_calls"]] == [
        "diagnose_start",
        "parse_evidence",
        "evidence_gap_resume",
    ]
    gap = result["metadata"]["evidence_gap_resolution"]
    assert gap["attempted"] is True
    assert any(
        item["source_ids"]
        for item in gap["observations"]
    )
    assert "只读工具观察" in result["answer"]


def test_deepseek_failure_falls_back_to_deterministic_runtime(
    tmp_path: Path,
) -> None:
    client = MockToolClient([])
    system = _system(tmp_path)
    result = DeepSeekReadToolHarness(system, client=client).run(
        "如何进入安全模式"
    )
    harness = result["metadata"]["deepseek_tool_harness"]
    assert harness["fallback_used"] is True
    assert result["status"] in {"ask_info", "step"}
