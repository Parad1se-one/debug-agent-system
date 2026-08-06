from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from debug_agent_system.agents.tools.executor import (
    ReadEvidenceToolExecutor,
    parse_evidence_tool_schema,
)


def test_parse_evidence_schema_is_strict() -> None:
    function = parse_evidence_tool_schema()["function"]
    assert function["strict"] is True
    parameters = function["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(parameters["properties"])
    resource = parameters["properties"]["resource"]
    assert resource["additionalProperties"] is False
    assert set(resource["required"]) == set(resource["properties"])


def test_log_tool_returns_source_bound_uniform_envelope(tmp_path: Path) -> None:
    log_path = tmp_path / "camera.log"
    log_path.write_text(
        "2026-07-24 camera capture failed E1005 ip=192.168.1.8\n",
        encoding="utf-8",
    )
    result = ReadEvidenceToolExecutor().execute(
        {
            "resource_id": "resource:camera-log",
            "kind": "log_package",
            "name": log_path.name,
            "path": str(log_path),
            "source_message_id": "message:42",
        }
    )
    payload = asdict(result)
    assert payload["schema_version"] == "debug_agent_system.read_tool_result.v1"
    assert payload["status"] == "parsed"
    assert payload["source_ids"] == ["resource:camera-log", "message:42"]
    assert payload["call_fingerprint"]
    assert payload["evidence_ids"]
    assert any(item["field"] == "error_codes" for item in payload["observations"])
    assert all(item["source_ids"] for item in payload["observations"])
    assert payload["safety"]["read_only"] is True
    assert payload["safety"]["mutated"] is False


def test_image_tool_never_claims_ocr(tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x02\x00\x00\x00\x03"
        b"\x08\x02\x00\x00\x00"
    )
    result = ReadEvidenceToolExecutor().execute(
        {
            "resource_id": "resource:screen",
            "kind": "image",
            "name": image_path.name,
            "path": str(image_path),
        }
    )
    assert result.safety["ocr_performed"] is False
    assert any(item["reason"] == "ocr_not_supported" for item in result.excluded)
    assert all(
        observation.supports_retrieval is False
        for observation in result.observations
    )
