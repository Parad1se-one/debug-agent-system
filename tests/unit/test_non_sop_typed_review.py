from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from debug_agent_system.agents.write.w4_quality_gate import QualityGateAgent
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.knowledge import JsonKGStore


def _playbook_envelope() -> dict:
    return {
        "intake_id": "intake:playbook:1",
        "dedupe_key": "typed:playbook:camera-reconnect",
        "source_kind": "chat_review",
        "admission_target": "playbook",
        "schema_valid": True,
        "raw_text": "现场相机掉线时，先确认网口灯和相机 IP，再重启采集服务。",
        "objects": {
            "Playbook": [
                {
                    "playbook_id": "playbook:camera-reconnect",
                    "title": "相机掉线排查流程",
                    "summary": "确认链路后重启采集服务。",
                    "source_kind": "chat_review",
                }
            ],
            "SourceCase": [
                {
                    "case_id": "case:playbook:1",
                    "source_kind": "chat_review",
                    "title": "现场相机掉线排查",
                    "summary": "聊天记录中的排查步骤。",
                }
            ],
            "EvidenceItem": [
                {
                    "evidence_id": "evidence:playbook:1",
                    "source_kind": "chat_review",
                    "summary": "现场相机掉线时的排查原文。",
                }
            ],
            "ProcedureStep": [
                {
                    "procedure_step_id": "procedure-step:playbook:1",
                    "section_id": "section:playbook:1",
                    "label": "确认网口灯和相机 IP",
                    "instruction": "确认网口灯和相机 IP",
                    "step_order": 1,
                }
            ],
        },
        "relations": [
            {"from": "case:playbook:1", "to": "playbook:camera-reconnect", "relation": "supports"},
            {"from": "evidence:playbook:1", "to": "case:playbook:1", "relation": "evidences"},
        ],
    }


def _fault_execution_envelope(*, outcome_type: str, outcome_origin: str) -> dict:
    return {
        "source_kind": "chat_review",
        "admission_target": "fault_execution",
        "schema_valid": True,
        "raw_text": "现场已卸载无线网卡驱动，但当前证据未给出稳定验证结果。",
        "objects": {
            "FaultFamily": [{"family_id": "family:pending", "label": "网络连接异常"}],
            "FaultVariant": [{"variant_id": "variant:pending", "family_id": "family:pending", "label": "无线网卡连接异常"}],
            "DiagnosticAction": [{
                "action_id": "action:pending",
                "label": "卸载无线网卡驱动",
                "execution_status": "actual",
                "evidence_ids": ["evidence:pending"],
            }],
            "ActionOutcome": [{
                "outcome_id": "outcome:pending",
                "action_id": "action:pending",
                "outcome_type": outcome_type,
                "outcome_origin": outcome_origin,
                "summary": "卸载无线网卡驱动已执行，但当前证据未给出稳定验证结果。",
                "evidence_ids": ["evidence:pending"],
            }],
            "DiagnosticTrace": [{
                "trace_id": "trace:pending",
                "recommended_action_ids": ["action:pending"],
                "actual_action_ids": ["action:pending"],
                "evidence_ids": ["evidence:pending"],
            }],
            "SourceCase": [{"case_id": "case:pending", "summary": "现场执行历史"}],
            "EvidenceItem": [{"evidence_id": "evidence:pending", "summary": "原始消息"}],
        },
    }


def test_w4_typed_playbook_does_not_force_fault_variant():
    result = QualityGateAgent().score_typed_candidate(_playbook_envelope())

    assert result["decision"] == "admit"
    assert result["admission_target"] == "playbook"
    assert result["materialize_allowed"] is False
    assert result["decision_version"] == "w4_typed_decision.v1"
    assert result["mapping_version"] == "kg_v2_typed_admission.v1"
    assert "FaultVariant" not in result["observability"]["object_types"]
    assert not any("FaultVariant" in issue for issue in result["issues"])


def test_w4_typed_rejects_sop_source():
    envelope = _playbook_envelope()
    envelope["source_kind"] = "sop"

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "reject"
    assert result["materialize_allowed"] is False
    assert "typed_sop_source_rejected" in result["issues"]


def test_w4_typed_reads_thin_payload_and_routes_missing_evidence():
    envelope = {
        "intake_id": "intake:thin:1",
        "payload": {
            "source_kind": "chat_review",
            "admission_target": "fault_execution",
            "schema_valid": True,
            "raw_text": "蓝屏后提示 MEMORY_MANAGEMENT，更换内存后未再出现。",
            "objects": {
                "FaultFamily": [{"family_id": "family:thin", "label": "工控机蓝屏"}],
                "FaultVariant": [{"variant_id": "variant:thin", "label": "蓝屏 MEMORY_MANAGEMENT"}],
                "DiagnosticAction": [{"action_id": "action:thin", "label": "更换内存条验证"}],
                "ActionOutcome": [
                    {
                        "outcome_id": "outcome:thin",
                        "action_id": "action:thin",
                        "outcome_type": "verified_fix",
                        "summary": "更换内存后未再出现",
                        "evidence_ids": ["evidence:thin"],
                    }
                ],
                "DiagnosticTrace": [
                    {
                        "trace_id": "trace:thin",
                        "recommended_action_ids": ["action:thin"],
                        "actual_action_ids": ["action:thin"],
                        "evidence_ids": ["evidence:thin"],
                    }
                ],
                "SourceCase": [{"case_id": "case:thin", "summary": "聊天记录证据"}],
                "EvidenceItem": [{"evidence_id": "evidence:thin", "summary": "原始消息"}],
            },
        },
        "evidence_pack": {
            "outcome_evidence": [{"outcome_id": "outcome:thin", "summary": "更换内存后未再出现"}],
        },
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "admit"
    assert result["admission_target"] == "fault_execution"
    assert result["admission_readiness"] == "execution_ready"
    assert result["materialize_allowed"] is True
    assert "FaultVariant" in result["observability"]["object_types"]

    envelope["evidence_pack"] = {}
    envelope["payload"]["objects"]["ActionOutcome"][0].pop("evidence_ids")
    result = QualityGateAgent().score_typed_candidate(envelope)
    assert result["decision"] == "route_review"
    assert result["materialize_allowed"] is False
    assert result["admission_readiness"] == "case_ready"
    assert "typed_missing_evidence:outcome_evidence" in result["issues"]


def test_w4_pending_only_trace_is_mergeable_history_but_not_execution_policy():
    result = QualityGateAgent().score_typed_candidate(
        _fault_execution_envelope(
            outcome_type="pending_validation",
            outcome_origin="synthetic_fallback",
        )
    )

    assert result["decision"] == "route_review"
    assert result["admission_readiness"] == "execution_ready"
    assert result["policy_readiness"] == "pending_only"
    assert result["merge_allowed"] is True
    assert result["materialize_allowed"] is False
    assert "typed_execution_policy_pending_only" in result["issues"]
    assert "typed_synthetic_pending_outcome_only" in result["issues"]
    assert result["observability"]["policy_evidence_counts"] == {
        "outcome_count": 1,
        "pending_count": 1,
        "synthetic_pending_count": 1,
        "observed_actual_count": 0,
        "promoted_only_action_count": 0,
    }


def test_w4_promoted_only_action_stays_reviewable_but_cannot_materialize():
    envelope = _fault_execution_envelope(
        outcome_type="ineffective",
        outcome_origin="source_extracted",
    )
    envelope["objects"]["DiagnosticAction"][0]["evidence_scope"] = "w7_promoted_only"

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "route_review"
    assert result["admission_readiness"] == "execution_ready"
    assert result["policy_readiness"] == "contains_promoted_only_action"
    assert result["merge_allowed"] is True
    assert result["materialize_allowed"] is False
    assert "typed_promoted_only_action_evidence" in result["issues"]


def test_w4_observed_negative_supports_policy_while_pending_branch_is_preserved():
    envelope = _fault_execution_envelope(
        outcome_type="ineffective",
        outcome_origin="source_extracted",
    )
    objects = envelope["objects"]
    objects["ActionOutcome"][0]["summary"] = "卸载无线网卡驱动后故障仍然存在，该操作无效。"
    objects["DiagnosticAction"].append({
        "action_id": "action:observe",
        "label": "继续观察是否复发",
        "execution_status": "recommended",
        "evidence_ids": ["evidence:pending"],
    })
    objects["ActionOutcome"].append({
        "outcome_id": "outcome:observe",
        "action_id": "action:observe",
        "outcome_type": "pending_validation",
        "outcome_origin": "synthetic_fallback",
        "summary": "继续观察是否复发为建议动作，尚无已执行证据。",
        "evidence_ids": ["evidence:pending"],
    })
    objects["DiagnosticTrace"][0]["recommended_action_ids"].append("action:observe")

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "admit"
    assert result["policy_readiness"] == "observed_execution"
    assert result["materialize_allowed"] is True
    assert result["observability"]["policy_evidence_counts"]["pending_count"] == 1
    assert result["observability"]["policy_evidence_counts"]["observed_actual_count"] == 1


def test_w4_fault_only_candidate_does_not_require_outcome_evidence_but_cannot_materialize():
    envelope = {
        "source_kind": "chat_review",
        "admission_target": "fault_execution",
        "schema_valid": True,
        "raw_text": "现场反馈相机拍摄失败，暂未形成排查结论。",
        "objects": {
            "FaultFamily": [{"family_id": "family:camera-capture", "label": "相机拍摄失败"}],
            "FaultVariant": [{"variant_id": "variant:camera-capture:1", "label": "生产中相机拍摄失败"}],
            "SourceCase": [{"case_id": "case:camera-capture:1", "summary": "群聊现场反馈"}],
            "EvidenceItem": [{"evidence_id": "evidence:camera-capture:1", "summary": "原始消息"}],
        },
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "route_review"
    assert result["admission_readiness"] == "case_ready"
    assert result["materialize_allowed"] is False
    assert "typed_fault_execution_missing_actions" in result["issues"]
    assert "typed_missing_evidence:outcome_evidence" not in result["issues"]
    assert "outcome_evidence" not in result["required_evidence"]


def test_w4_evidence_only_fault_record_is_evidence_ready():
    envelope = {
        "source_kind": "chat_review",
        "admission_target": "fault_execution",
        "schema_valid": True,
        "raw_text": "现场只确认发生异常，问题边界尚不完整。",
        "objects": {
            "SourceCase": [{"case_id": "case:evidence-only", "summary": "现场异常"}],
            "EvidenceItem": [{"evidence_id": "evidence:evidence-only", "summary": "原始消息"}],
        },
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["admission_readiness"] == "evidence_ready"
    assert result["merge_allowed"] is True
    assert result["materialize_allowed"] is False


def test_w4_outcome_candidate_still_requires_message_level_outcome_evidence():
    envelope = {
        "source_kind": "chat_review",
        "admission_target": "fault_execution",
        "schema_valid": True,
        "raw_text": "现场更换内存条后蓝屏未再出现。",
        "objects": {
            "FaultVariant": [{"variant_id": "variant:bsod:1", "label": "生产中工控机蓝屏"}],
            "DiagnosticAction": [{"action_id": "action:replace-memory", "label": "更换内存条验证"}],
            "ActionOutcome": [
                {
                    "outcome_id": "outcome:replace-memory:1",
                    "action_id": "action:replace-memory",
                    "outcome_type": "verified_fix",
                    "summary": "更换后未再蓝屏",
                }
            ],
            "SourceCase": [{"case_id": "case:bsod:1", "summary": "群聊现场反馈"}],
        },
    }

    result = QualityGateAgent().score_typed_candidate(envelope)
    assert result["decision"] == "route_review"
    assert "typed_missing_evidence:outcome_evidence" in result["issues"]
    assert "outcome_evidence" in result["required_evidence"]

    envelope["objects"]["ActionOutcome"][0]["evidence_message_ids"] = ["om_bsod_1"]
    result = QualityGateAgent().score_typed_candidate(envelope)
    assert result["decision"] == "admit"
    assert "typed_missing_evidence:outcome_evidence" not in result["issues"]


def test_w4_typed_rejects_sop_source_ref_in_payload():
    envelope = {
        "payload": {
            "source_kind": "manual_review",
            "source_ref": "data/raw/aoi_debug_agent_sources/进板失败SOP--20250521.docx",
            "admission_target": "playbook",
            "raw_text": "按 SOP 步骤处理。",
            "objects": {"SourceCase": [{"case_id": "case:sop"}]},
        }
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "reject"
    assert result["materialize_allowed"] is False
    assert "typed_sop_source_rejected" in result["issues"]


def test_w4_typed_does_not_reject_non_sop_path_token():
    envelope = _playbook_envelope()
    envelope["source_ref"] = "data/non_sop/camera_guide.docx"

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "admit"
    assert "typed_sop_source_rejected" not in result["issues"]


def test_w4_jira_evidence_without_fault_semantics_requires_review():
    envelope = {
        "source_type": "jira",
        "source_ref": {"path": "/tmp/SMTAOI-1.json"},
        "admission_target": "evidence_only",
        "schema_valid": True,
        "raw_text": "Collect all SDK test cases",
        "objects": {
            "SourceCase": [{"case_id": "case:jira:SMTAOI-1", "source_kind": "jira_case"}],
            "EvidenceItem": [{"evidence_id": "evidence:jira:SMTAOI-1", "source_kind": "jira", "payload_ref": "/tmp/SMTAOI-1.json", "summary": "Collect all SDK test cases"}],
        },
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "route_review"
    assert result["materialize_allowed"] is False
    assert "typed_low_fault_relevance" in result["issues"]


def test_w4_jira_fault_evidence_can_enter_support_review_batch():
    envelope = {
        "source_type": "jira",
        "source_ref": {"path": "/tmp/AOI-139.json"},
        "admission_target": "evidence_only",
        "schema_valid": True,
        "raw_text": "设备检测过程中蓝屏重启，错误码 0x139",
        "objects": {
            "SourceCase": [{"case_id": "case:jira:AOI-139", "source_kind": "jira_case"}],
            "EvidenceItem": [{"evidence_id": "evidence:jira:AOI-139", "source_kind": "jira", "payload_ref": "/tmp/AOI-139.json", "summary": "蓝屏重启"}],
        },
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "admit"
    assert result["materialize_allowed"] is False
    assert "typed_low_fault_relevance" not in result["issues"]


def test_w4_routes_procedure_document_without_steps_to_human_review():
    envelope = {
        "source_type": "manual_review",
        "admission_target": "procedure_library",
        "schema_valid": True,
        "raw_text": "显卡驱动安装教程",
        "payload": {
            "strategy": {"strategy_id": "procedure_doc"},
            "objects": {
                "KnowledgeDocument": [{"document_id": "doc:driver"}],
                "KnowledgeSection": [{"section_id": "section:driver"}],
                "EvidenceItem": [{"evidence_id": "evidence:driver"}],
                "ProcedureStep": [],
            },
        },
        "evidence_pack": {"structured_sections": [{"section_title": "安装教程"}]},
    }

    result = QualityGateAgent().score_typed_candidate(envelope)

    assert result["decision"] == "route_review"
    assert "typed_document_missing_procedure_steps" in result["issues"]


def test_w6_canonical_typed_queue_is_idempotent_by_dedupe_key_and_intake_id():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = JsonKGStore(tmp_path / "kg")
        queue = ReviewQueueAgent(store)
        gate = QualityGateAgent().score_typed_candidate(_playbook_envelope())

        first = queue.build_typed_review_item(_playbook_envelope(), gate)
        second_envelope = _playbook_envelope()
        second_envelope["raw_text"] = "更新后的原文证据。"
        second = queue.build_typed_review_item(second_envelope, gate)

        assert queue.enqueue("v2_typed_candidates", first)["status"] == "queued"
        assert queue.enqueue("v2_typed_candidates", second)["status"] == "updated"
        rows = store.read_review_queue("v2_typed_candidates.json")
        assert len(rows) == 1
        assert rows[0]["raw_evidence"]["text"] == "更新后的原文证据。"
        assert rows[0]["kg_alignment"]["legacy_fault_variant_forced"] is False
        assert "FaultVariant" not in rows[0]["object_diff"]["object_counts"]
        assert rows[0]["materialize_allowed"] is False
        assert rows[0]["dry_run_plan"]["admission_target"] == "playbook"
        decision = queue.mark_decision("v2_typed_candidates", rows[0]["review_id"], "approve", reviewer="unit")
        assert decision["status"] == "decision_recorded"
        assert decision["human_approved"] is True
        assert queue.enqueue("v2_typed_candidates", second)["status"] == "updated"
        preserved = store.read_review_queue("v2_typed_candidates.json")[0]
        assert preserved["review_status"] == "approved"
        assert preserved["human_approved"] is True
        assert preserved["review_decision"]["reviewer"] == "unit"

        no_dedupe = _playbook_envelope()
        no_dedupe.pop("dedupe_key")
        no_dedupe["intake_id"] = "intake:playbook:stable"
        item = queue.build_typed_review_item(no_dedupe, QualityGateAgent().score_typed_candidate(no_dedupe))
        updated = queue.build_typed_review_item(no_dedupe, QualityGateAgent().score_typed_candidate(no_dedupe))
        updated["raw_evidence"]["text"] = "same intake replacement"
        assert queue.enqueue("v2_typed_candidates", item)["status"] == "queued"
        assert queue.enqueue("v2_typed_candidates", updated)["status"] == "updated"
        rows = store.read_review_queue("v2_typed_candidates.json")
        assert len(rows) == 2

        thin = {
            "dedupe_key": "typed:thin:fault",
            "payload": {
                "intake_id": "intake:thin:fault",
                "admission_target": "fault_execution",
                "source_kind": "chat_review",
                "raw_text": "蓝屏后提示 MEMORY_MANAGEMENT，更换内存后未再出现。",
                "objects": {
                    "FaultFamily": [{"family_id": "family:thin", "label": "工控机蓝屏"}],
                    "FaultVariant": [{"variant_id": "variant:thin", "label": "蓝屏 MEMORY_MANAGEMENT"}],
                    "DiagnosticAction": [{"action_id": "action:thin", "label": "更换内存条验证"}],
                    "ActionOutcome": [
                        {
                            "outcome_id": "outcome:thin",
                            "action_id": "action:thin",
                            "outcome_type": "verified_fix",
                            "summary": "更换内存后未再出现",
                            "evidence_ids": ["evidence:thin"],
                        }
                    ],
                    "DiagnosticTrace": [
                        {
                            "trace_id": "trace:thin",
                            "recommended_action_ids": ["action:thin"],
                            "actual_action_ids": ["action:thin"],
                            "evidence_ids": ["evidence:thin"],
                        }
                    ],
                    "SourceCase": [{"case_id": "case:thin"}],
                    "EvidenceItem": [{"evidence_id": "evidence:thin", "summary": "原始消息"}],
                },
            },
            "evidence_pack": {
                "outcome_evidence": [{"outcome_id": "outcome:thin", "summary": "更换内存后未再出现"}],
            },
        }
        thin_gate = QualityGateAgent().score_typed_candidate(thin)
        thin_item = queue.build_typed_review_item(thin, thin_gate)
        assert thin_item["intake_id"] == "intake:thin:fault"
        assert thin_item["raw_evidence"]["text"] == "蓝屏后提示 MEMORY_MANAGEMENT，更换内存后未再出现。"
        assert thin_item["object_diff"]["object_counts"]["FaultVariant"] == 1
        assert thin_item["outcome_evidence"][0]["outcome_id"] == "outcome:thin"
        assert thin_item["materialize_allowed"] is True


def test_w6_changed_content_invalidates_previous_approval() -> None:
    with TemporaryDirectory() as tmp:
        store = JsonKGStore(Path(tmp) / "kg")
        queue = ReviewQueueAgent(store)
        first_envelope = _playbook_envelope()
        first_envelope["content_hash"] = "content:v1"
        gate = QualityGateAgent().score_typed_candidate(first_envelope)
        first = queue.build_typed_review_item(first_envelope, gate)
        queue.enqueue("v2_typed_candidates", first)
        queue.mark_decision("v2_typed_candidates", first["review_id"], "approve", reviewer="unit")

        changed_envelope = _playbook_envelope()
        changed_envelope["content_hash"] = "content:v2"
        changed_envelope["raw_text"] = "内容已经改变。"
        changed = queue.build_typed_review_item(changed_envelope, QualityGateAgent().score_typed_candidate(changed_envelope))
        queue.enqueue("v2_typed_candidates", changed)

        row = store.read_review_queue("v2_typed_candidates.json")[0]
        assert row["review_status"] == "needs_re_review"
        assert row["human_approved"] is False
        assert "review_decision" not in row
        assert row["previous_review_decision"]["review_status"] == "approved"


def test_w6_enqueue_batches_reads_and_writes_canonical_queue_once_semantics() -> None:
    with TemporaryDirectory() as tmp:
        store = JsonKGStore(Path(tmp) / "kg")
        queue = ReviewQueueAgent(store)
        gate = QualityGateAgent().score_typed_candidate(_playbook_envelope())
        first = queue.build_typed_review_item(_playbook_envelope(), gate)
        second_envelope = _playbook_envelope()
        second_envelope["dedupe_key"] = "dedupe:second"
        second_envelope["intake_id"] = "intake:second"
        second = queue.build_typed_review_item(second_envelope, QualityGateAgent().score_typed_candidate(second_envelope))
        replacement = dict(first)
        replacement["raw_evidence"] = {**first["raw_evidence"], "text": "replacement"}

        result = queue.enqueue_batches("v2_typed_candidates", [[first, second], [replacement]])

        assert result == {
            "status": "batch_written",
            "queue": "v2_typed_candidates.json",
            "size": 2,
            "queued": 2,
            "updated": 1,
            "batch_count": 2,
        }
        rows = store.read_review_queue("v2_typed_candidates.json")
        assert len(rows) == 2
        assert rows[0]["raw_evidence"]["text"] == "replacement"
