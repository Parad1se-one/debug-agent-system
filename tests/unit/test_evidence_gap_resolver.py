from __future__ import annotations

from pathlib import Path

from debug_agent_system.agents.read.o_evidence_gap import EvidenceGapResolver


def test_resolver_uses_relevant_source_bound_log_observations(tmp_path: Path) -> None:
    log_path = tmp_path / "startup.log"
    log_path.write_text(
        "camera capture failed E1005 during startup\n",
        encoding="utf-8",
    )
    result = EvidenceGapResolver().resolve(
        ["请补充明确报错文本和发生阶段"],
        [
            {
                "resource_id": "resource:startup-log",
                "kind": "log_package",
                "name": log_path.name,
                "path": str(log_path),
                "source_message_id": "message:1",
            }
        ],
    )
    assert result.attempted is True
    assert result.resolved_items == ["请补充明确报错文本和发生阶段"]
    assert result.unresolved_items == []
    assert "resource:startup-log" in result.retrieval_context
    assert "E1005" in result.retrieval_context
    assert all(item.source_ids for item in result.observations)


def test_resolver_deduplicates_identical_tool_call(tmp_path: Path) -> None:
    log_path = tmp_path / "error.log"
    log_path.write_text("error E1005\n", encoding="utf-8")
    resource = {
        "resource_id": "resource:error-log",
        "kind": "log_package",
        "name": log_path.name,
        "path": str(log_path),
    }
    first = EvidenceGapResolver().resolve(["请补充错误码"], [resource])
    second = EvidenceGapResolver().resolve(
        ["请补充错误码"],
        [resource],
        processed_fingerprints=[
            first.tool_results[0].call_fingerprint
        ],
    )
    assert second.tool_results == []
    assert second.stop_reason == "all_tool_calls_deduplicated"


def test_image_metadata_does_not_resolve_screenshot_text_gap(tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x02\x00\x00\x00\x03"
        b"\x08\x02\x00\x00\x00"
    )
    result = EvidenceGapResolver().resolve(
        ["请补充截图中的报错文本"],
        [
            {
                "resource_id": "resource:screen",
                "kind": "image",
                "name": image_path.name,
                "path": str(image_path),
            }
        ],
    )
    assert result.resolved_items == []
    assert result.unresolved_items == ["请补充截图中的报错文本"]
    assert result.retrieval_context == ""
    assert result.stop_reason == "observations_not_relevant_to_required_data"
