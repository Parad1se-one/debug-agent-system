from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


def _pipeline(root: Path) -> WriteSidePipeline:
    shutil.copytree("data/kg_v2/schema", root / "kg_v2" / "schema")
    return WriteSidePipeline(
        JsonKGStore(root / "legacy"),
        kg_v2_root=root / "kg_v2",
        kg_v2_queue_dir=root / "queue",
    )


def test_diagnostic_feedback_enters_canonical_queue_as_evidence_only() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pipeline = _pipeline(root)
        transcript = {
            "session_id": "session:camera:1",
            "query": "相机请求超时并拍摄失败",
            "top_error_id": "camera-capture-failed",
            "final_status": "unresolved",
            "check_results": {"check-network": "failed"},
        }

        first = pipeline.run_diagnostic_feedback(transcript)
        replay = pipeline.run_diagnostic_feedback(transcript)

        assert first["proposal"]["type"] == "DiagnosticFeedback"
        assert first["quality_gate"]["decision"] == "route_review"
        assert first["quality_gate"]["admission_target"] == "evidence_only"
        assert first["quality_gate"]["admission_readiness"] == "evidence_ready"
        assert first["quality_gate"]["materialize_allowed"] is False
        assert replay["queue_write"]["status"] == "updated"
        rows = json.loads((root / "queue" / "v2_typed_candidates.json").read_text(encoding="utf-8"))
        assert len(rows) == 1
        candidate = rows[0]["typed_candidate"]
        assert candidate["source_type"] == "diagnostic_feedback"
        assert candidate["payload"]["objects"]["SourceCase"]
        assert candidate["payload"]["objects"]["EvidenceItem"]
        assert not candidate["payload"]["objects"].get("DiagnosticAction")


def test_log_pattern_enters_canonical_queue_without_creating_log_pattern_node() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pipeline = _pipeline(root)
        summary = {
            "signature_id": "sig:camera-timeout",
            "pattern": "GrabImage timeout",
            "matched": False,
            "source_files": ["DLOG/main.log"],
        }

        result = pipeline.run_log_pattern(summary)

        assert result["proposal"]["type"] == "LogPatternCandidate"
        assert result["quality_gate"]["decision"] == "route_review"
        assert result["quality_gate"]["admission_readiness"] == "evidence_ready"
        candidate = result["review_item"]["typed_candidate"]
        assert candidate["source_type"] == "log_pattern"
        assert set(candidate["payload"]["objects"]) == {"SourceCase", "EvidenceItem"}
        assert candidate["materialize_allowed"] is False


def test_loop_evidence_requires_w6_approval_and_never_materializes_execution() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pipeline = _pipeline(root)
        result = pipeline.run_diagnostic_feedback({
            "session_id": "session:approval:1",
            "query": "相机拍摄失败",
            "final_status": "unresolved",
        })
        pending = result["review_item"]

        skipped = pipeline.w5_v2.apply_approved_typed_review_item(pending, materialize=True)
        assert skipped["status"] == "skipped"
        assert skipped["reason"] == "not_approved"
        assert JsonKGV2Store(root / "kg_v2").objects_by_type["SourceCase"] == []

        decision = pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            pending["dedupe_key"],
            "approve_support_only",
            reviewer="loop-reviewer",
        )
        assert decision["human_approved"] is True
        approved = json.loads((root / "queue" / "v2_typed_candidates.json").read_text(encoding="utf-8"))[0]
        applied = pipeline.w5_v2.apply_approved_typed_review_item(approved, materialize=True)
        replay = pipeline.w5_v2.apply_approved_typed_review_item(approved, materialize=True)

        assert applied["status"] == "applied_to_graph_v2"
        assert applied["materialized_counts"] == {}
        assert replay == {"status": "already_applied", "dedupe_key": approved["dedupe_key"]}
        refreshed = JsonKGV2Store(root / "kg_v2")
        assert refreshed.objects_by_type["SourceCase"][0]["approved"] is True
        assert refreshed.objects_by_type["EvidenceItem"]
        assert refreshed.objects_by_type["DiagnosticAction"] == []
        assert refreshed.objects_by_type["ActionOutcome"] == []


def test_loop_content_change_keeps_identity_but_forces_re_review() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pipeline = _pipeline(root)
        base = {
            "session_id": "session:changed:1",
            "query": "相机拍摄失败",
            "final_status": "unresolved",
        }
        first = pipeline.run_diagnostic_feedback(base)
        pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            first["review_item"]["dedupe_key"],
            "approve_support_only",
            reviewer="first-reviewer",
        )

        changed = pipeline.run_diagnostic_feedback({**base, "final_status": "resolved", "which_check_solved": "check-network"})

        assert changed["review_item"]["dedupe_key"] == first["review_item"]["dedupe_key"]
        rows = json.loads((root / "queue" / "v2_typed_candidates.json").read_text(encoding="utf-8"))
        assert len(rows) == 1
        assert rows[0]["review_status"] == "needs_re_review"
        assert rows[0]["human_approved"] is False
        assert rows[0]["previous_review_decision"]["selected_action"] == "approve_support_only"


def test_atr_proposal_has_idempotent_review_queue_but_no_w5_operation() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pipeline = _pipeline(root)
        feedback = {
            "session_id": "session:atr:1",
            "top_error_id": "camera-capture-failed",
            "which_check_solved": "check-network",
            "check_results": {"check-network": "failed"},
        }

        first = pipeline.run_atr_weight_proposal(feedback)
        replay = pipeline.run_atr_weight_proposal(feedback)

        assert first["proposal"]["type"] == "ATRWeightProposal"
        assert first["review_item"]["queue"] == "atr_weight_proposals"
        assert first["review_item"]["application_boundary"]["operation"] == "none"
        assert first["review_item"]["application_boundary"]["w5_eligible"] is False
        assert replay["queue_write"]["status"] == "updated"
        rows = json.loads((root / "queue" / "atr_weight_proposals.json").read_text(encoding="utf-8"))
        assert len(rows) == 1
        decision = pipeline.w6_v2.mark_decision(
            "atr_weight_proposals",
            rows[0]["dedupe_key"],
            "accept",
            reviewer="atr-reviewer",
        )
        assert decision["human_approved"] is True
        assert JsonKGV2Store(root / "kg_v2").all_objects() == []
