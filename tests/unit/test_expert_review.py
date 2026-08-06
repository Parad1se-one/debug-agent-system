from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.agents.write_v2.expert_review import build_expert_corrected_candidate
from debug_agent_system.knowledge.json_store import JsonKGStore


def _review_item() -> dict:
    return {
        "review_id": "review:typed:dedupe:old",
        "dedupe_key": "dedupe:old",
        "typed_candidate": {
            "payload": {
                "objects": {
                    "FaultFamily": [{
                        "family_id": "family:old",
                        "label": "进板失败",
                        "summary": "板卡进入设备流程失败。",
                        "category": "系统与软件异常",
                        "subsystem": "扫码/进板",
                        "keywords": [],
                    }],
                },
                "episode": {
                    "episode_id": "chat:episode:1",
                    "fault_description_messages": [{"message_id": "m1", "text": "扫码后不进板。"}],
                },
            },
        },
    }


def _correction() -> dict:
    return {
        "review_id": "review:typed:dedupe:old",
        "disposition": "replace_root_cause_and_trace_from_jira",
        "family": "进板失败",
        "variant": "扫码枪配置异常导致扫码后不进板",
        "actions": [
            {"order": 1, "label": "核对扫码枪配置文件", "role": "inspect"},
            {"order": 2, "label": "重启主程序并验证进板", "role": "verify"},
        ],
        "outcomes": [{
            "action_order": 2,
            "outcome_type": "partial_temporary",
            "summary": "重启后临时恢复进板。",
            "evidence_refs": ["TEST-1"],
        }],
        "required_info": [{
            "slot": "program_file",
            "question": "请提供扫码枪配置文件。",
            "why_required": "核对触发配置。",
        }],
        "evidence_additions": [{
            "kind": "jira",
            "external_id": "TEST-1",
            "summary": "扫码枪配置文件问题。",
        }],
        "source_episode_id_original": "chat:episode:1",
        "review_basis": {
            "trust_tier": "gold",
            "annotation_set_id": "gold-v1",
            "annotation_case_id": "goldcase-011",
            "annotation_sha256": "a" * 64,
            "ingest_run_id": "test-run",
        },
    }


def test_expert_correction_builds_fresh_schema_valid_candidate():
    result = build_expert_corrected_candidate(_review_item(), _correction())

    assert result["schema_valid"] is True
    assert result["dedupe_key"] == "expert-corrected:dedupe:old"
    assert result["objects"]["FaultVariant"][0]["label"] == "扫码枪配置异常导致扫码后不进板"
    assert len(result["objects"]["DiagnosticAction"]) == 2
    assert result["objects"]["ActionOutcome"][0]["outcome_type"] == "partial_temporary"
    assert result["objects"]["SourceCase"][0]["annotation_case_id"] == "goldcase-011"
    assert result["objects"]["SourceCase"][0]["trust_tier"] == "gold"
    assert [item["ordinal"] for item in result["objects"]["TraceStep"]] == [1, 2]
    assert len(result["objects"]["ExecutionObservation"]) == 1
    assert result["objects"]["ExecutionObservation"][0]["observation_count"] == 1
    assert len(result["objects"]["BranchRule"]) == 2
    assert result["objects"]["BranchRule"][-1]["terminal_status"] == "monitoring"


def test_expert_rebound_gets_new_identity_and_keeps_original_review_reference():
    correction = _correction()
    correction["disposition"] = "do_not_apply_original_create_rebound_candidate"

    result = build_expert_corrected_candidate(_review_item(), correction)

    assert result["provenance_rebound"] is True
    assert result["dedupe_key"] == "expert-rebound:dedupe:old"
    assert result["supersedes_review_id"] == "review:typed:dedupe:old"
    assert result["objects"]["SourceCase"][0]["source_ref"].startswith("rebound:")


def test_expert_correction_reenters_w4_w6_canonical_queue_without_mutating_original():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        queue_dir = root / "queue"
        shutil.copytree("data/kg_v2/schema", root / "kg_v2" / "schema")
        original = _review_item()
        snapshot = json.loads(json.dumps(original, ensure_ascii=False))
        pipeline = WriteSidePipeline(
            JsonKGStore(root / "legacy"),
            kg_v2_root=root / "kg_v2",
            kg_v2_queue_dir=queue_dir,
        )

        result = pipeline.run_expert_correction(original, _correction())

        assert original == snapshot
        assert result["candidate"]["dedupe_key"] == "expert-corrected:dedupe:old"
        assert result["quality_gate"]["decision"] in {"admit", "route_review"}
        assert result["quality_gate"]["admission_readiness"] == "execution_ready"
        assert result["review_item"]["queue"] == "v2_typed_candidates"
        assert result["review_item"]["review_status"] == "pending"
        assert result["review_item"]["typed_candidate"]["lineage"]["agent_id"] == "W6-EXPERT-REVIEW"
        rows = json.loads((queue_dir / "v2_typed_candidates.json").read_text(encoding="utf-8"))
        assert len(rows) == 1
        assert rows[0]["dedupe_key"] == "expert-corrected:dedupe:old"
        assert rows[0]["typed_candidate"]["payload"]["supersedes_review_id"] == "review:typed:dedupe:old"
