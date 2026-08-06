from __future__ import annotations

from debug_agent_system.read_runtime_v3.evaluation import (
    response_to_formal_prediction,
    structural_errors,
)


def _response() -> dict:
    return {
        "schema_version": "debug_agent_system.read_response.v3",
        "query": "如何进入安全模式",
        "status": "step",
        "answer": "grounded answer",
        "task": {
            "facets": ["进入"],
            "facet_details": [{"facet_id": "operation:进入", "label": "进入"}],
        },
        "baseline_response": {
            "status": "step",
            "answer": "grounded answer",
            "family_id": "",
            "variant_id": "",
            "current_action_id": "",
            "evidence_ids": [],
            "metadata": {
                "document_answer_mode": {"active": True},
                "evidence_pack": {
                    "source_items": [{
                        "document_id": "doc:1",
                        "object_id": "section:1",
                        "chunk_ids": ["chunk:1"],
                        "evidence_ids": [],
                    }],
                    "allowed_references": {"chunk_ids": ["chunk:1"]},
                },
                "retrieval": {"supporting_chunks": []},
            },
        },
        "answer_plan": {"sections": [{
            "claims": [{"claim_id": "claim:1", "evidence_ids": ["ev3:1"]}],
        }]},
        "verification": {"passed": True},
        "evidence_snapshot": {
            "record_count": 1,
            "records": [{"evidence_id": "ev3:1"}],
            "fingerprint": "abc",
        },
        "shadow": {"enabled": True, "answer_changed": False, "status_changed": False},
        "trace": [
            {"stage": "provider:baseline", "status": "ok"},
            {"stage": "planner", "status": "ok"},
        ],
    }


def test_projection_preserves_document_route_and_grounding_ids():
    prediction = response_to_formal_prediction(_response())
    assert prediction["route_type"] == "knowledge_document_section"
    assert prediction["status"] == "answer"
    assert prediction["route_ids"] == ["doc:1", "section:1"]
    assert set(prediction["evidence_ids"]) == {"doc:1", "section:1", "chunk:1"}
    assert prediction["executed_action_ids"] == []


def test_projection_recovers_document_route_from_v3_information_task():
    response = _response()
    response["baseline_response"]["metadata"]["document_answer_mode"] = {}
    response["task"]["mode"] = "knowledge_lookup"
    prediction = response_to_formal_prediction(response)
    assert prediction["route_type"] == "knowledge_document_section"


def test_projection_does_not_turn_document_backed_fault_into_document_route():
    response = _response()
    response["baseline_response"]["metadata"]["document_answer_mode"] = {}
    response["baseline_response"]["family_id"] = "family:1"
    response["task"]["mode"] = "fault_diagnosis"
    prediction = response_to_formal_prediction(response)
    assert prediction["route_type"] == "sag_v2_native"


def test_projection_counts_v3_kg_candidates_as_retrieved_not_locked():
    response = _response()
    response["task"]["mode"] = "fault_diagnosis"
    response["baseline_response"]["metadata"]["document_answer_mode"] = {"active": True}
    response["baseline_response"]["family_id"] = ""
    response["baseline_response"]["variant_id"] = ""
    response["evidence_snapshot"] = {
        "record_count": 2,
        "records": [
            {"evidence_id": "ev3:1"},
            {
                "evidence_id": "ev3:kg",
                "provider": "kg_v2_sag",
                "kind": "kg_object",
                "source_ref": "variant:retrieved",
                "content": {
                    "family_id": "family:retrieved",
                    "variant_id": "variant:retrieved",
                },
            },
        ],
        "fingerprint": "abc",
    }
    prediction = response_to_formal_prediction(response)
    assert prediction["route_type"] == "sag_v2_native"
    assert prediction["family_id"] == ""
    assert prediction["variant_id"] == ""
    assert {"family:retrieved", "variant:retrieved"} <= set(prediction["route_ids"])
    assert "variant:retrieved" in prediction["evidence_ids"]


def test_projection_recognizes_grounded_source_only_context():
    response = _response()
    response["baseline_response"]["metadata"]["document_answer_mode"] = {}
    response["evidence_snapshot"] = {
        "record_count": 2,
        "records": [
            {"evidence_id": "ev3:1"},
            {
                "evidence_id": "ev3:2",
                "provider": "request_context",
                "source_ref": "message:1",
                "metadata": {"source_only": True},
            },
        ],
        "fingerprint": "abc",
    }
    prediction = response_to_formal_prediction(response)
    assert prediction["route_type"] == "source_only_trace_reconstruction"
    assert "message:1" in prediction["evidence_ids"]


def test_projection_uses_structured_trace_count_and_ids():
    response = _response()
    response["answer_plan"]["traces"] = [{
        "trace_id": "trace:derived:1",
        "evidence_ids": ["ev3:1"],
    }]
    prediction = response_to_formal_prediction(response)
    assert prediction["trace_count"] == 1
    assert "trace:derived:1" in prediction["route_ids"]
    assert "ev3:1" in prediction["evidence_ids"]


def test_structural_gate_accepts_closed_shadow_response():
    assert structural_errors(_response()) == []


def test_structural_gate_rejects_answer_drift_and_unclosed_claim():
    response = _response()
    response["answer"] = "changed"
    response["answer_plan"]["sections"][0]["claims"][0]["evidence_ids"] = ["ev3:missing"]
    assert structural_errors(response) == [
        "official_answer_drift",
        "claim_unknown_evidence:claim:1:ev3:missing",
    ]
