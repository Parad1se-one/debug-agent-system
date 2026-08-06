from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from debug_agent_system.adapters.cli import _kg_v2_sag_publish_required, build_parser
from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph
from debug_agent_system.agents.write.w4_quality_gate import _kg_v2_outcome_type_conflicts


def _pipeline(root: Path) -> WriteSidePipeline:
    return WriteSidePipeline(JsonKGStore(root / "legacy"), kg_v2_root=root / "kg_v2")


def _file(tool: str, result: dict) -> dict:
    return {"name": f"sample.{tool}", "path": f"/tmp/sample.{tool}", "tool": tool, "parse_result": result}


def test_jira_detail_creates_reviewable_source_case_with_evidence_links() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        context = {
            "context_id": "SMTAOI-100",
            "context_root": "/tmp/jira",
            "source_context": {"anchor_messages": ["现场蓝屏"]},
            "source_manifest": {},
            "files": [_file("jira", {
                "status": "offline_detail_found",
                "offline_details": [{"issue_key": "SMTAOI-100", "summary": "现场蓝屏", "description_preview": "上传 DMP", "comment_preview_text": "待分析"}],
            })],
            "tool_evidence": {},
        }

        envelope = pipeline._evidence_context_envelope(context)
        payload = envelope["payload"]

        assert payload["evidence_disposition"] == "new_source_case"
        assert payload["objects"]["SourceCase"][0]["source_kind"] == "jira_case"
        assert payload["objects"]["SourceCase"][0]["approved"] is False
        assert payload["relations"][0]["relation"] == "evidences"


def test_known_source_case_routes_to_merge_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pipeline = _pipeline(root)
        case = {"case_id": "case:known", "source_kind": "chat_case", "title": "已知案例", "summary": "已知案例", "source_ref": "ep:1", "approved": True}
        JsonKGV2Store(root / "kg_v2").merge_graph({"SourceCase": [case]}, [])
        pipeline.kg_v2_store = JsonKGV2Store(root / "kg_v2")
        context = {
            "context_id": "attachment-1",
            "context_root": "/tmp/attachment",
            "source_context": {"anchor_messages": ["补充诊断数据"]},
            "source_manifest": {"source_case_id": "case:known"},
            "files": [_file("log_package", {"status": "manifest_only", "exists": True})],
            "tool_evidence": {},
        }

        envelope = pipeline._evidence_context_envelope(context)

        assert envelope["payload"]["evidence_disposition"] == "merge_evidence"
        assert envelope["payload"]["objects"]["SourceCase"][0]["case_id"] == "case:known"


def test_unanchored_attachment_stays_evidence_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        context = {
            "context_id": "orphan-attachment",
            "context_root": "/tmp/attachment",
            "source_context": {"anchor_messages": []},
            "source_manifest": {},
            "files": [_file("dmp", {"status": "header_metadata", "exists": True, "dump_kind": "kernel"})],
            "tool_evidence": {},
        }

        envelope = pipeline._evidence_context_envelope(context)
        gate = pipeline.w4.score_typed_candidate(envelope)

        assert envelope["payload"]["evidence_disposition"] == "evidence_only"
        assert envelope["payload"]["objects"]["SourceCase"] == []
        assert envelope["payload"]["objects"]["EvidenceItem"] == []
        assert envelope["evidence_pack"]["evidence_items"]
        assert gate["decision"] == "route_review"


def test_unusable_parse_routes_to_review_and_cannot_materialize() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        context = {
            "context_id": "missing-file",
            "context_root": "/tmp/attachment",
            "source_context": {"anchor_messages": ["现场故障"]},
            "source_manifest": {},
            "files": [_file("attachment", {"status": "missing", "exists": False})],
            "tool_evidence": {},
        }

        envelope = pipeline._evidence_context_envelope(context)
        gate = pipeline.w4.score_typed_candidate(envelope)

        assert envelope["payload"]["evidence_disposition"] == "reject_review_only"
        assert gate["decision"] == "route_review"
        assert gate["materialize_allowed"] is False
        assert "typed_evidence_requires_review" in gate["issues"]


def test_cli_exposes_all_non_sop_incremental_entrypoints() -> None:
    parser = build_parser()

    text_args = parser.parse_args(["ingest-text-history", "data/imports/text", "--kg-v2-root", "data/kg_v2"])
    doc_args = parser.parse_args(["ingest-non-sop-doc", "guide.docx"])
    sop_doc_args = parser.parse_args(["ingest-sop-doc", "guide-SOP.docx"])
    sop_sync_args = parser.parse_args(["sync-sop-docs", "data/raw"])
    evidence_args = parser.parse_args(["ingest-evidence-context", "data/imports/tool_samples"])
    expert_args = parser.parse_args(["ingest-expert-correction", "review.json", "correction.json"])
    feedback_args = parser.parse_args(["ingest-diagnostic-feedback", "transcript.json"])
    log_pattern_args = parser.parse_args(["ingest-log-pattern", "log-summary.json"])
    atr_args = parser.parse_args(["ingest-atr-weight-proposal", "feedback.json"])
    review_args = parser.parse_args(["review-decision", "v2_typed_candidates", "review:1", "approve", "--kg-version", "v2"])
    apply_args = parser.parse_args(["apply-approved-queue", "--kg-version", "v2"])

    assert text_args.w2_mode == "native_v2"
    assert doc_args.kg_v2_root == "data/kg_v2"
    assert sop_doc_args.kg_v2_root == "data/kg_v2"
    assert sop_sync_args.limit == 0
    assert evidence_args.max_bytes == 65536
    assert expert_args.kg_v2_root == "data/kg_v2"
    assert feedback_args.kg_v2_root == "data/kg_v2"
    assert log_pattern_args.kg_v2_root == "data/kg_v2"
    assert atr_args.kg_v2_root == "data/kg_v2"
    assert review_args.queue == "v2_typed_candidates"
    assert apply_args.kg_v2_sag_out == "data/kg_v2_sag/debug_agent_v2.sqlite"


def test_cli_publishes_sag_for_any_v2_graph_change_including_support_only() -> None:
    support_only = [{
        "status": "applied_to_graph_v2",
        "requires_sag_publish": True,
        "document_index_changed": True,
        "materialized_counts": {},
    }]

    assert _kg_v2_sag_publish_required(support_only, "v2") is True
    assert _kg_v2_sag_publish_required(support_only, "both") is True
    assert _kg_v2_sag_publish_required(support_only, "v1") is False
    assert _kg_v2_sag_publish_required([{"status": "already_applied"}], "v2") is False


def test_pipeline_rejects_sop_build_path_before_w9_reads_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        with patch("debug_agent_system.agents.write.pipeline.RawDocIngestAgent.inspect_document", side_effect=AssertionError("must not read")):
            try:
                pipeline.run_non_sop_document("data/kg_v2_sop_draft_build/card.docx")
            except Exception as exc:
                assert getattr(exc, "code", "") == "sop_source_rejected"
            else:
                raise AssertionError("expected SOP build path to be rejected")


def test_pipeline_allows_non_sop_directory_name_before_w9_reads_document() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        path = Path(tmp) / "non_sop" / "camera_guide.docx"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not-a-real-docx")
        with patch.object(pipeline, "_preflight_non_sop_kg_v2", return_value={}), patch(
            "debug_agent_system.agents.write.pipeline.RawDocIngestAgent.inspect_document",
            side_effect=RuntimeError("reached_w9"),
        ):
            try:
                pipeline.run_non_sop_document(path)
            except RuntimeError as exc:
                assert str(exc) == "reached_w9"
            else:
                raise AssertionError("expected W9 probe")


def test_non_sop_raw_doc_provenance_never_leaks_sop_source_kind() -> None:
    bundle = {
        "objects": {
            "FaultFamily": [{"family_id": "family:cpu", "label": "CPU温度过高", "summary": "CPU温度异常升高", "category": "系统与软件异常", "source_kind": "sop"}],
            "FaultVariant": [],
            "DiagnosticAction": [{"action_id": "action:cpu", "family_id": "family:cpu", "label": "检查散热", "summary": "检查风扇和散热器", "action_role": "inspect", "source_kind": "sop"}],
            "ActionOutcome": [], "RequiredInfoSpec": [], "DiagnosticTrace": [], "DecisionPolicy": [],
            "EvidenceItem": [],
            "SourceCase": [{"case_id": "case:cpu", "source_kind": "sop", "title": "CPU温度指南", "summary": "CPU温度指南", "source_ref": "sop", "approved": True}],
        },
        "relations": [],
    }

    converted = WriteSidePipeline._mark_bundle_as_non_sop_raw_doc(bundle, Path("guide.docx"))

    assert converted["schema_valid"] is True
    assert converted["objects"]["FaultFamily"][0]["source_kind"] == "raw_doc"
    assert converted["objects"]["DiagnosticAction"][0]["source_kind"] == "raw_doc"
    assert converted["objects"]["SourceCase"][0]["source_kind"] == "raw_doc"
    assert converted["objects"]["SourceCase"][0]["approved"] is False
    assert validate_graph(converted["objects"], converted["relations"]) == []


def test_non_fault_raw_doc_uses_source_case_as_safe_evidence_target() -> None:
    bundle = WriteSidePipeline._document_evidence_bundle(
        {
            "name": "D盘扩容方法.docx",
            "path": "/tmp/D盘扩容方法.docx",
            "section_cases": [
                {
                    "case_id": "doc:d-drive:section:1",
                    "section_title": "修改盘符",
                    "variant_candidate": "数据盘盘符调整",
                    "actions": ["打开磁盘管理并修改非系统盘盘符"],
                    "cause_notes": [],
                }
            ],
        },
        knowledge_kind="procedure",
        admission_target="procedure_library",
    )

    assert bundle["schema_valid"] is True
    assert bundle["objects"]["FaultFamily"] == []
    assert bundle["objects"]["DiagnosticAction"] == []
    assert bundle["objects"]["SourceCase"][0]["source_kind"] == "raw_doc"
    assert bundle["objects"]["SourceCase"][0]["approved"] is False
    assert bundle["relations"] == [{
        "from": bundle["objects"]["EvidenceItem"][0]["evidence_id"],
        "to": bundle["objects"]["SourceCase"][0]["case_id"],
        "relation": "evidences",
    }]
    assert validate_graph(bundle["objects"], bundle["relations"]) == []


def test_typed_fault_candidate_requires_v2_semantic_gate_not_only_shape_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        envelope = {
            "source_type": "raw_doc",
            "source_ref": {"path": "/tmp/semantic-weak.docx"},
            "knowledge_kind": "fault_case",
            "text": "文档提到一个尚未校准的故障",
            "candidate_id": "candidate:semantic-weak",
            "admission_target": "fault_execution",
            "payload": {
                "text": "文档提到一个尚未校准的故障",
                "schema_valid": True,
                "schema_issues": [],
                "objects": {
                    "FaultFamily": [{"family_id": "family:weak", "label": "尚未校准的复杂故障名称", "summary": "故障", "category": "系统与软件异常", "source_kind": "raw_doc"}],
                    "FaultVariant": [],
                    "DiagnosticAction": [],
                    "ActionOutcome": [{"outcome_id": "outcome:weak", "family_id": "family:weak", "action_id": "action:missing", "outcome_type": "pending_validation", "summary": "待验证", "source_case_id": "case:weak", "evidence_ids": ["evidence:weak"]}],
                    "RequiredInfoSpec": [], "DiagnosticTrace": [], "DecisionPolicy": [],
                    "EvidenceItem": [{"evidence_id": "evidence:weak", "source_kind": "tool_parse", "title": "原文", "summary": "原文", "payload_ref": "/tmp/semantic-weak.docx"}],
                    "SourceCase": [{"case_id": "case:weak", "source_kind": "raw_doc", "title": "文档", "summary": "文档", "approved": False}],
                },
                "relations": [],
            },
            "evidence_pack": {"evidence_items": [{"evidence_id": "evidence:weak"}]},
        }

        gate = pipeline._score_typed_envelope(envelope)

        assert gate["decision"] == "route_review"
        assert gate["materialize_allowed"] is False
        assert gate["kg_v2_semantic_gate"]["passed"] is False
        assert "kg_v2_semantic_gate_failed" in gate["issues"]


def test_chat_execution_candidate_without_current_episode_evidence_policy_routes_review() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        envelope = {
            "source_type": "chat",
            "source_ref": {"episode_id": "episode:legacy"},
            "knowledge_kind": "fault_case",
            "text": "现场相机拍摄失败，检查相机网络后恢复。",
            "candidate_id": "candidate:legacy-context",
            "admission_target": "fault_execution",
            "schema_valid": True,
            "payload": {
                "text": "现场相机拍摄失败，检查相机网络后恢复。",
                "schema_valid": True,
                "objects": {
                    "FaultFamily": [{"family_id": "family:camera", "label": "相机拍摄失败", "summary": "相机拍摄失败", "category": "硬件与运控", "source_kind": "case"}],
                    "FaultVariant": [{"variant_id": "variant:camera", "family_id": "family:camera", "label": "相机网络异常导致拍摄失败", "summary": "相机网络异常"}],
                    "DiagnosticAction": [{"action_id": "action:camera", "family_id": "family:camera", "variant_id": "variant:camera", "label": "检查相机网络", "summary": "检查相机网络", "action_role": "inspect", "source_kind": "case"}],
                    "ActionOutcome": [{
                        "outcome_id": "outcome:camera",
                        "family_id": "family:camera",
                        "variant_id": "variant:camera",
                        "action_id": "action:camera",
                        "outcome_type": "verified_fix",
                        "summary": "检查并修正相机网络配置后拍摄恢复正常",
                        "source_case_id": "case:chat",
                        "evidence_ids": ["evidence:chat"],
                    }], "RequiredInfoSpec": [], "DiagnosticTrace": [{
                        "trace_id": "trace:camera",
                        "family_id": "family:camera",
                        "variant_id": "variant:camera",
                        "source_case_id": "case:chat",
                        "summary": "检查相机网络",
                        "recommended_action_ids": ["action:camera"],
                        "actual_action_ids": ["action:camera"],
                        "evidence_ids": ["evidence:chat"],
                    }], "DecisionPolicy": [],
                    "EvidenceItem": [{"evidence_id": "evidence:chat", "source_kind": "chat_message", "title": "消息", "summary": "现场相机拍摄失败"}],
                    "SourceCase": [{"case_id": "case:chat", "source_kind": "chat_case", "title": "案例", "summary": "案例", "approved": False}],
                },
                "relations": [{"from": "evidence:chat", "to": "case:chat", "relation": "evidences"}],
            },
            "evidence_pack": {"message_ids": ["m1"]},
        }

        legacy_gate = pipeline.w4.score_typed_candidate(envelope)
        envelope["payload"]["context_evidence_policy"] = "current_episode_only.v1"
        current_gate = pipeline.w4.score_typed_candidate(envelope)

        assert legacy_gate["decision"] == "route_review"
        assert legacy_gate["materialize_allowed"] is False
        assert "typed_untrusted_context_evidence_policy" in legacy_gate["issues"]
        assert current_gate["decision"] == "admit"
        assert current_gate["materialize_allowed"] is True


def test_v2_semantic_gate_rejects_heading_and_tool_names_as_actions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        bundle = {
            "candidate_id": "candidate:weak-actions",
            "schema_valid": True,
            "schema_issues": [],
            "objects": {
                "FaultFamily": [{"family_id": "family:restart", "label": "工控机异常重启", "summary": "工控机异常重启", "category": "系统与软件异常", "source_kind": "raw_doc"}],
                "FaultVariant": [{"variant_id": "variant:cpu-hot", "family_id": "family:restart", "label": "CPU温度异常升高后重启", "summary": "CPU高温后重启"}],
                "DiagnosticAction": [
                    {"action_id": "action:notice", "family_id": "family:restart", "variant_id": "variant:cpu-hot", "label": "注意", "summary": "注意事项", "action_role": "observe", "source_kind": "raw_doc"},
                    {"action_id": "action:occt", "family_id": "family:restart", "variant_id": "variant:cpu-hot", "label": "OCCT", "summary": "压力测试", "action_role": "inspect", "source_kind": "raw_doc"},
                ],
                "ActionOutcome": [], "RequiredInfoSpec": [],
                "DiagnosticTrace": [{"trace_id": "trace:cpu", "family_id": "family:restart", "variant_id": "variant:cpu-hot", "source_case_id": "case:cpu", "summary": "排查链", "recommended_action_ids": ["action:notice", "action:occt"], "actual_action_ids": [], "evidence_ids": ["evidence:cpu"]}],
                "DecisionPolicy": [],
                "EvidenceItem": [{"evidence_id": "evidence:cpu", "source_kind": "tool_parse", "title": "指南", "summary": "指南"}],
                "SourceCase": [{"case_id": "case:cpu", "source_kind": "raw_doc", "title": "指南", "summary": "指南", "approved": False}],
            },
            "relations": [{"from": "evidence:cpu", "to": "case:cpu", "relation": "evidences"}],
            "strategy": {"kg_output_mode": "family_support_bundle"},
        }

        gate = pipeline.w4.score_v2_bundle(bundle)

        assert gate["passed"] is False
        assert "kg_v2_non_action_labels" in gate["issues"]


def test_v2_semantic_gate_rejects_historical_result_statements_as_actions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        bundle = {
            "candidate_id": "candidate:history-as-action",
            "schema_valid": True,
            "schema_issues": [],
            "objects": {
                "FaultFamily": [{"family_id": "family:camera", "label": "相机拍摄失败", "summary": "相机拍摄失败", "category": "硬件与运控", "source_kind": "case"}],
                "FaultVariant": [{"variant_id": "variant:camera", "family_id": "family:camera", "label": "更换采集卡后仍频繁拍照失败", "summary": "更换采集卡后仍频繁失败"}],
                "DiagnosticAction": [{"action_id": "action:history", "family_id": "family:camera", "variant_id": "variant:camera", "label": "年前现场已经更换过采集卡了", "summary": "历史处理记录", "action_role": "change", "source_kind": "case"}],
                "ActionOutcome": [], "RequiredInfoSpec": [],
                "DiagnosticTrace": [{"trace_id": "trace:camera", "family_id": "family:camera", "variant_id": "variant:camera", "source_case_id": "case:camera", "summary": "排查链", "recommended_action_ids": ["action:history"], "actual_action_ids": ["action:history"], "evidence_ids": ["evidence:camera"]}],
                "DecisionPolicy": [],
                "EvidenceItem": [{"evidence_id": "evidence:camera", "source_kind": "chat_message", "title": "消息", "summary": "消息"}],
                "SourceCase": [{"case_id": "case:camera", "source_kind": "chat_case", "title": "案例", "summary": "案例", "approved": False}],
            },
            "relations": [{"from": "evidence:camera", "to": "case:camera", "relation": "evidences"}],
            "strategy": {"kg_output_mode": "variant_case_bundle"},
        }

        gate = pipeline.w4.score_v2_bundle(bundle)

        assert gate["passed"] is False
        assert "kg_v2_historical_statement_actions" in gate["issues"]


def test_v2_verified_fix_accepts_current_test_normal_evidence() -> None:
    outcomes = [{
        "outcome_type": "verified_fix",
        "summary": "拆除缠绕气管并调整气流后，顶板升降速度测试正常。",
    }]
    assert _kg_v2_outcome_type_conflicts(outcomes, "现场发现气流过小，拆掉面顶测试速度正常；调整气流测试正常") == []
    assert _kg_v2_outcome_type_conflicts(
        [{"outcome_type": "verified_fix", "summary": "重新拔插光源 USB 接口后恢复正常。"}],
        "通电测试后光源初始化失败，在进行光源USB接口重新拔插已正常",
    ) == []


def test_gold_alignment_context_is_allowed_but_cannot_become_candidate_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        envelope = {
            "source_type": "chat",
            "source_ref": {"episode_id": "ep:current", "message_ids": ["m-current"]},
            "knowledge_kind": "fault_case",
            "text": "现场蓝屏，更换内存条后未再出现。",
            "metadata": {
                "alignment_context": {
                    "context_role": "alignment_only",
                    "reviewed_case_examples": [{"case_id": "goldcase-001", "graph_ingestion": False}],
                }
            },
            "payload": {
                "text": "现场蓝屏，更换内存条后未再出现。",
                "schema_valid": True,
                "episode": {"episode_id": "ep:current", "evidence_message_ids": ["m-current"]},
                "objects": {
                    "EvidenceItem": [{
                        "evidence_id": "evidence:current",
                        "source_kind": "chat_message",
                        "external_id": "m-current",
                        "title": "当前消息",
                        "summary": "当前消息",
                        "payload_ref": "",
                    }],
                    "SourceCase": [{
                        "case_id": "case:current",
                        "source_kind": "chat_case",
                        "title": "当前案例",
                        "summary": "当前案例",
                        "source_ref": "ep:current",
                        "approved": False,
                    }],
                },
                "relations": [],
            },
        }

        allowed = pipeline.w4.score_typed_candidate(envelope)
        assert not any("alignment_evidence" in issue for issue in allowed["issues"])

        envelope["payload"]["objects"]["EvidenceItem"][0].update({
            "source_kind": "gold_case",
            "external_id": "goldcase-001",
            "payload_ref": "data/annotations/goldcases/gold-v1/goldcase-001.json",
        })
        rejected = pipeline.w4.score_typed_candidate(envelope)
        assert rejected["decision"] == "reject"
        assert rejected["materialize_allowed"] is False
        assert any("typed_alignment_evidence" in issue for issue in rejected["issues"])


def test_provenance_rejects_chat_evidence_without_current_message_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        envelope = {
            "source_type": "manual_review",
            "source_ref": {"episode_id": "ep:manual"},
            "knowledge_kind": "fault_case",
            "text": "人工案例蓝屏。",
            "payload": {
                "text": "人工案例蓝屏。",
                "schema_valid": True,
                "objects": {
                    "EvidenceItem": [{
                        "evidence_id": "evidence:copied",
                        "source_kind": "chat_message",
                        "external_id": "m-from-example",
                        "title": "示例消息",
                        "summary": "示例消息",
                        "payload_ref": "",
                    }]
                },
                "relations": [],
            },
        }

        gate = pipeline.w4.score_typed_candidate(envelope)

        assert gate["decision"] == "reject"
        assert "typed_alignment_evidence_missing_current_message_ids" in gate["issues"]


def test_raw_doc_provenance_requires_tool_parse_from_current_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _pipeline(Path(tmp))
        envelope = {
            "source_type": "raw_doc",
            "source_ref": {"path": "data/raw/guide.docx"},
            "knowledge_kind": "support",
            "text": "CPU 温度过高处理指南。",
            "payload": {
                "text": "CPU 温度过高处理指南。",
                "schema_valid": True,
                "objects": {
                    "EvidenceItem": [{
                        "evidence_id": "evidence:guide",
                        "source_kind": "tool_parse",
                        "external_id": "section:1",
                        "title": "指南",
                        "summary": "指南",
                        "payload_ref": "data/raw/guide.docx",
                    }]
                },
                "relations": [],
            },
        }

        allowed = pipeline.w4.score_typed_candidate(envelope)
        assert not any("alignment_evidence" in issue for issue in allowed["issues"])

        envelope["payload"]["objects"]["EvidenceItem"][0].update({
            "source_kind": "chat_message",
            "payload_ref": "data/annotations/goldcases/gold-v1/example.json",
        })
        rejected = pipeline.w4.score_typed_candidate(envelope)
        assert rejected["decision"] == "reject"
        assert any("typed_alignment_evidence" in issue for issue in rejected["issues"])
