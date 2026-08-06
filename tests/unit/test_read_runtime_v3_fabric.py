from __future__ import annotations

import pytest

from debug_agent_system.read_runtime_v3.contracts import EvidenceRecord, SourceAnchor
from debug_agent_system.read_runtime_v3.fabric import EvidenceFabric


def test_evidence_fabric_is_content_addressed_and_deduplicated():
    fabric = EvidenceFabric()
    first = fabric.create_record(
        kind="diagnostic_event",
        provider="test",
        source_ref="log:1",
        assertion="observed",
        summary="illegal memory access",
        content={"line": 7, "message": "illegal memory access"},
        anchors=[SourceAnchor(path="app.log", line_start=7, line_end=7)],
    )
    second = fabric.create_record(
        kind="diagnostic_event",
        provider="test",
        source_ref="log:1",
        assertion="observed",
        summary="illegal memory access",
        content={"line": 7, "message": "illegal memory access"},
        anchors=[SourceAnchor(path="app.log", line_start=7, line_end=7)],
    )
    assert first.evidence_id == second.evidence_id
    assert fabric.snapshot()["record_count"] == 1


def test_evidence_fabric_rejects_identity_collision_and_unknown_links():
    fabric = EvidenceFabric()
    record = fabric.create_record(
        kind="source_artifact",
        provider="test",
        source_ref="a",
        assertion="observed",
        summary="a",
        content="a",
    )
    with pytest.raises(ValueError, match="evidence_identity_collision"):
        fabric.add_record(EvidenceRecord(
            evidence_id=record.evidence_id,
            kind="source_artifact",
            provider="test",
            source_ref="a",
            assertion="observed",
            summary="different",
            content="different",
        ))
    with pytest.raises(KeyError, match="unknown_evidence"):
        fabric.link("supports", record.evidence_id, "ev3:missing")


def test_evidence_fabric_snapshot_fingerprint_is_stable():
    def build():
        fabric = EvidenceFabric()
        source = fabric.create_record(
            kind="source_artifact", provider="p", source_ref="x",
            assertion="observed", summary="x", content={"x": 1},
        )
        claim = fabric.create_record(
            kind="answer_fragment", provider="p", source_ref="a",
            assertion="derived", summary="a", content={"a": 1},
        )
        fabric.link("derived_from", claim.evidence_id, source.evidence_id)
        return fabric.snapshot()

    first = build()
    second = build()
    assert first["fingerprint"] == second["fingerprint"]
    assert first["record_count"] == 2
    assert first["link_count"] == 1

