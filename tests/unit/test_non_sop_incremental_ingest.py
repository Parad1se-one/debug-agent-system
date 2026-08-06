from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import shutil

from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.agents.write_v2.ingest import IncrementalIngestV2Agent
from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


def test_non_sop_document_batch_splits_document_and_fault_mapping_review_scopes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        shutil.copytree("data/kg_v2/schema", root / "schema")
        queue_dir = Path(tmp) / "queue"
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=queue_dir,
        )

        result = pipeline.run_non_sop_document(
            "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx",
            split_review_scopes=True,
        )

        rows = json.loads((queue_dir / "v2_typed_candidates.json").read_text(encoding="utf-8"))
        assert {row["review_scope"] for row in rows} == {"document_layer", "fault_mapping"}
        document = next(row for row in rows if row["review_scope"] == "document_layer")
        semantic = next(row for row in rows if row["review_scope"] == "fault_mapping")
        document_objects = document["typed_candidate"]["payload"]["objects"]
        semantic_objects = semantic["typed_candidate"]["payload"]["objects"]
        assert document_objects["KnowledgeDocument"]
        assert document_objects["DiagnosticAction"] == []
        assert semantic_objects["DiagnosticAction"]
        document_manifest = document["typed_candidate"]["payload"]["chunk_manifest"]
        assert document_manifest["binding_status"] == "draft_kg_sections"
        assert document_manifest["chunks"]
        assert semantic["typed_candidate"]["payload"]["chunk_manifest"] == {}
        assert document["typed_candidate"]["evidence_pack"]["chunk_manifest_ref"]["manifest_hash"] == document_manifest["manifest_hash"]
        assert semantic["typed_candidate"]["evidence_pack"]["chunk_manifest_ref"]["manifest_hash"] == document_manifest["manifest_hash"]
        assert all(chunk["approved"] is False for chunk in document_manifest["chunks"])
        assert len(result["review_scopes"]) == 2


def test_fault_mapping_approval_requires_exact_approved_document_layer() -> None:
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        root = Path(tmp)
        shutil.copytree("data/kg_v2/schema", root / "schema")
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=root / "pending_review",
        )
        result = pipeline.run_non_sop_document(
            "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx",
            split_review_scopes=True,
        )
        document_review = next(
            item["review_item"]
            for item in result["review_scopes"]
            if item["review_item"]["review_scope"] == "document_layer"
        )
        mapping_review = next(
            item["review_item"]
            for item in result["review_scopes"]
            if item["review_item"]["review_scope"] == "fault_mapping"
        )

        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            mapping_review["review_id"],
            "approve_support_only",
            reviewer="unit-reviewer",
        )["status"] == "decision_recorded"
        blocked = pipeline.apply_approved_review_queue(kg_mode="v2")

        assert len(blocked) == 1
        assert blocked[0]["status"] == "skipped"
        assert blocked[0]["reason"] == "fault_mapping_document_layer_not_approved"
        assert blocked[0]["dependency_issues"]
        assert blocked[0]["dedupe_key"] == mapping_review["dedupe_key"]
        assert JsonKGV2Store(root).objects_by_type["KnowledgeDocument"] == []
        assert JsonKGV2Store(root).objects_by_type["FaultFamily"] == []

        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            document_review["review_id"],
            "approve_support_only",
            reviewer="unit-reviewer",
        )["status"] == "decision_recorded"
        applied = pipeline.apply_approved_review_queue(kg_mode="v2")
        document_result = next(
            item for item in applied
            if item.get("candidate_id") == document_review["candidate_id"]
        )
        mapping_result = next(
            item for item in applied
            if item.get("candidate_id") == mapping_review["candidate_id"]
        )

        assert document_result["status"] == "applied_to_graph_v2"
        assert mapping_result["status"] == "applied_to_graph_v2"
        assert mapping_result["document_index_changed"] is False
        assert not {
            "KnowledgeDocument", "KnowledgeSection", "ProcedureStep", "EvidenceItem"
        }.intersection(mapping_result["affected_object_types"])
        store = JsonKGV2Store(root)
        assert len(store.objects_by_type["KnowledgeDocument"]) == 1
        assert store.objects_by_type["KnowledgeDocument"][0]["approved"] is True
        assert store.objects_by_type["FaultFamily"]
        assert store.objects_by_type["DiagnosticAction"]


def test_approved_document_layer_rebuilds_current_source_chunks_into_sag() -> None:
    # Keep the KG root two levels below the repository so SAG source-path
    # resolution has the same layout as data/kg_v2.
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        root = Path(tmp)
        shutil.copytree("data/kg_v2/schema", root / "schema")
        queue_dir = root / "pending_review"
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=queue_dir,
        )
        result = pipeline.run_non_sop_document(
            "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx",
            split_review_scopes=True,
        )
        document_review = next(
            item["review_item"]
            for item in result["review_scopes"]
            if item["review_item"]["review_scope"] == "document_layer"
        )
        decision = pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            document_review["review_id"],
            "approve_support_only",
            reviewer="unit-reviewer",
        )

        assert decision["status"] == "decision_recorded"
        apply_results = pipeline.apply_approved_review_queue(kg_mode="v2")
        assert len(apply_results) == 1
        assert apply_results[0]["requires_sag_publish"] is True
        assert apply_results[0]["document_index_changed"] is True

        report = WriteSideV2Pipeline(root).build_sqlite_sag(root / "read.sqlite", reset=True)
        manifest = result["bundle"]["chunk_manifest"]
        assert report["status"] == "built"
        assert report["source_hash_mismatch_count"] == 0
        assert report["source_parse_failure_count"] == 0
        assert report["source_aligned_chunk_count"] == len(manifest["chunks"])
        assert report["source_aligned_section_count"] == manifest["stats"]["bound_section_count"]


def test_approved_new_source_version_replaces_old_document_layer_and_chunks() -> None:
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        root = Path(tmp)
        shutil.copytree("data/kg_v2/schema", root / "schema")
        source_path = root / "增量索引更新教程.md"
        source_ref = source_path.as_posix()
        source_path.write_text(
            "# 旧版专属说明\n\n旧版唯一标记 OLD-CHUNK-ONLY。\n\n# 操作步骤\n\n1. 检查旧配置。\n",
            encoding="utf-8",
        )
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=root / "pending_review",
        )

        first = pipeline.run_non_sop_document(source_ref, split_review_scopes=True)
        first_review = first["review_scopes"][0]["review_item"]
        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates", first_review["review_id"], "approve_support_only", reviewer="unit-reviewer"
        )["status"] == "decision_recorded"
        assert any(item.get("requires_sag_publish") for item in pipeline.apply_approved_review_queue(kg_mode="v2"))

        source_path.write_text(
            "# 新版专属说明\n\n新版唯一标记 NEW-CHUNK-ONLY。\n\n# 操作步骤\n\n1. 检查新配置。\n",
            encoding="utf-8",
        )
        second = pipeline.run_non_sop_document(source_ref, split_review_scopes=True)
        second_review = second["review_scopes"][0]["review_item"]
        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates", second_review["review_id"], "approve_support_only", reviewer="unit-reviewer"
        )["status"] == "decision_recorded"
        apply_results = pipeline.apply_approved_review_queue(kg_mode="v2")
        updated = next(item for item in apply_results if item.get("candidate_id") == second_review["candidate_id"])

        replacement = updated["merge_result"]["document_source_replacement"]
        assert replacement["removed_document_count"] == 1
        assert replacement["removed_section_count"] > 0
        documents = [
            item for item in JsonKGV2Store(root).objects_by_type["KnowledgeDocument"]
            if item.get("source_path") == source_ref
        ]
        assert len(documents) == 1
        assert documents[0]["content_hash"] == second["bundle"]["chunk_manifest"]["source_file_hash"]

        sag_path = root / "read.sqlite"
        WriteSideV2Pipeline(root).build_sqlite_sag(sag_path, reset=True)
        with sqlite3.connect(sag_path) as conn:
            old_count = conn.execute(
                "SELECT COUNT(*) FROM source_chunks WHERE approved=1 AND text LIKE '%OLD-CHUNK-ONLY%'"
            ).fetchone()[0]
            new_count = conn.execute(
                "SELECT COUNT(*) FROM source_chunks WHERE approved=1 AND text LIKE '%NEW-CHUNK-ONLY%'"
            ).fetchone()[0]
        assert old_count == 0
        assert new_count > 0


def test_document_replacement_waits_for_mapping_and_removes_old_source_only_objects_atomically() -> None:
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        root = Path(tmp)
        shutil.copytree("data/kg_v2/schema", root / "schema")
        source_path = root / "CPU温度过高问题处理指南.docx"
        shutil.copyfile(
            "data/raw/aoi_debug_agent_sources/CPU温度过高问题处理指南.docx",
            source_path,
        )
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=root / "pending_review",
        )

        first = pipeline.run_non_sop_document(
            source_path.as_posix(),
            split_review_scopes=True,
        )
        for scoped in first["review_scopes"]:
            review = scoped["review_item"]
            assert pipeline.w6_v2.mark_decision(
                "v2_typed_candidates",
                review["review_id"],
                "approve_support_only",
                reviewer="unit-reviewer",
            )["status"] == "decision_recorded"
        first_apply = pipeline.apply_approved_review_queue(kg_mode="v2")
        assert sum(item.get("status") == "applied_to_graph_v2" for item in first_apply) == 2
        first_store = JsonKGV2Store(root)
        old_action_ids = {
            item["action_id"] for item in first_store.objects_by_type["DiagnosticAction"]
        }
        old_required_info_ids = {
            item["required_info_id"] for item in first_store.objects_by_type["RequiredInfoSpec"]
        }
        assert old_action_ids
        assert old_required_info_ids

        # Appending bytes changes the reviewed file hash while keeping the DOCX
        # ZIP readable, which models a new source revision with the same content.
        source_path.write_bytes(source_path.read_bytes() + b"\nreview-version-two\n")
        second = pipeline.run_non_sop_document(
            source_path.as_posix(),
            split_review_scopes=True,
        )
        second_document = next(
            item["review_item"]
            for item in second["review_scopes"]
            if item["review_item"]["review_scope"] == "document_layer"
        )
        second_mapping = next(
            item["review_item"]
            for item in second["review_scopes"]
            if item["review_item"]["review_scope"] == "fault_mapping"
        )
        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            second_document["review_id"],
            "approve_support_only",
            reviewer="unit-reviewer",
        )["status"] == "decision_recorded"

        document_apply = pipeline.apply_approved_review_queue(kg_mode="v2")
        assert document_apply[0]["status"] == "skipped"
        assert document_apply[0]["reason"] == "document_replacement_requires_mapping_approval"
        unchanged_store = JsonKGV2Store(root)
        assert old_action_ids == {
            item["action_id"] for item in unchanged_store.objects_by_type["DiagnosticAction"]
        }

        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            second_mapping["review_id"],
            "approve_support_only",
            reviewer="unit-reviewer",
        )["status"] == "decision_recorded"
        mapping_apply = pipeline.apply_approved_review_queue(kg_mode="v2")
        pair_result = next(
            item for item in mapping_apply
            if item.get("status") == "applied_to_graph_v2"
        )
        assert set(pair_result["component_dedupe_keys"]) == {
            second_document["dedupe_key"],
            second_mapping["dedupe_key"],
        }
        replacement = pair_result["merge_result"]["document_source_replacement"]
        assert replacement["removed_source_only_action_count"] == len(old_action_ids)
        assert replacement["removed_source_only_required_info_count"] == len(old_required_info_ids)
        final_store = JsonKGV2Store(root)
        assert final_store.objects_by_type["DiagnosticAction"]
        assert final_store.objects_by_type["RequiredInfoSpec"]
        new_mapping_objects = second_mapping["typed_candidate"]["payload"]["objects"]
        assert {
            item["action_id"] for item in final_store.objects_by_type["DiagnosticAction"]
        } == {
            item["action_id"] for item in new_mapping_objects["DiagnosticAction"]
        }
        assert {
            item["required_info_id"] for item in final_store.objects_by_type["RequiredInfoSpec"]
        } == {
            item["required_info_id"] for item in new_mapping_objects["RequiredInfoSpec"]
        }
        assert final_store.objects_by_type["FaultFamily"]
        assert final_store.objects_by_type["FaultVariant"]
        replay = pipeline.apply_approved_review_queue(kg_mode="v2")
        assert len(replay) == 2
        assert {item["status"] for item in replay} == {"already_applied"}
        assert {item["dedupe_key"] for item in replay} == {
            second_document["dedupe_key"],
            second_mapping["dedupe_key"],
        }


def test_source_change_after_review_invalidates_approval_before_w5_merge() -> None:
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        root = Path(tmp)
        shutil.copytree("data/kg_v2/schema", root / "schema")
        source_path = root / "审核期间变化.md"
        source_path.write_text(
            "# 初始版本\n\n审核时看到的是 V1-CONTENT。\n\n# 操作步骤\n\n1. 检查旧配置。\n",
            encoding="utf-8",
        )
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=root / "pending_review",
        )
        result = pipeline.run_non_sop_document(
            source_path.as_posix(),
            split_review_scopes=True,
        )
        document_review = next(
            item["review_item"]
            for item in result["review_scopes"]
            if item["review_item"]["review_scope"] == "document_layer"
        )
        assert pipeline.w6_v2.mark_decision(
            "v2_typed_candidates",
            document_review["review_id"],
            "approve_support_only",
            reviewer="unit-reviewer",
        )["status"] == "decision_recorded"

        source_path.write_text(
            "# 新版本\n\n批准前已经变为 V2-CONTENT。\n\n# 操作步骤\n\n1. 检查新配置。\n",
            encoding="utf-8",
        )
        apply_results = pipeline.apply_approved_review_queue(kg_mode="v2")

        assert apply_results[0]["status"] == "skipped"
        assert apply_results[0]["reason"] == "source_content_changed_since_review"
        assert apply_results[0]["source_hash_issues"][0]["expected_hash"]
        assert apply_results[0]["source_hash_issues"][0]["current_hash"]
        assert JsonKGV2Store(root).objects_by_type["KnowledgeDocument"] == []
        queued = pipeline.w6_v2.read_queue("v2_typed_candidates")
        invalidated = next(
            item for item in queued
            if item.get("review_id") == document_review["review_id"]
        )
        assert invalidated["review_status"] == "needs_re_review"
        assert invalidated["human_approved"] is False
        assert invalidated["review_invalidation"]["reason"] == "source_content_changed_since_review"
        assert "review_decision" not in invalidated


def test_sop_document_incremental_update_uses_reviewed_atomic_replacement() -> None:
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        root = Path(tmp) / "kg_v2"
        shutil.copytree("data/kg_v2/schema", root / "schema")
        source_root = Path(tmp) / "sources"
        source_root.mkdir()
        source_path = source_root / "相机断连标准操作流程.md"
        source_path.write_text(
            "# 相机断连排查\n\n"
            "情况 1：相机频繁断连\n\n"
            "表现：相机网卡反复重置。\n\n"
            "排查步骤：检查网线；查询网卡重置事件；验证相机恢复。\n",
            encoding="utf-8",
        )
        pipeline = WriteSidePipeline(
            JsonKGStore("data/kg"),
            kg_v2_root=root,
            kg_v2_queue_dir=root / "pending_review",
        )

        first = pipeline.run_sop_document(source_path)
        assert first["run_manifest"]["source_type"] == "sop_doc"
        assert first["bundle"]["schema_valid"] is True
        assert first["quality_gate"]["decision"] != "reject"
        assert {
            item["review_item"]["review_scope"]
            for item in first["review_scopes"]
        } == {"document_layer", "fault_mapping"}
        for scoped in first["review_scopes"]:
            review = scoped["review_item"]
            assert pipeline.w6_v2.mark_decision(
                "v2_typed_candidates",
                review["review_id"],
                "approve_support_only",
                reviewer="unit-reviewer",
            )["status"] == "decision_recorded"
        assert any(
            item.get("status") == "applied_to_graph_v2"
            for item in pipeline.apply_approved_review_queue(kg_mode="v2")
        )
        stored = JsonKGV2Store(root)
        document = next(
            item
            for item in stored.objects_by_type["KnowledgeDocument"]
            if item.get("source_path") == str(source_path)
        )
        assert document["source_kind"] == "sop"
        first_document_id = document["document_id"]

        unchanged = pipeline.run_sop_documents(source_root)
        assert unchanged["summary"]["unchanged"] == 1
        assert unchanged["documents"][0]["status"] == "unchanged"

        source_path.write_text(
            "# 相机断连排查 V2\n\n"
            "情况 1：相机频繁断连\n\n"
            "表现：相机网卡反复重置。\n\n"
            "排查步骤：检查网线；重新安装网卡驱动；验证生产恢复。\n",
            encoding="utf-8",
        )
        changed = pipeline.run_sop_documents(source_root)
        assert changed["summary"]["queued_update"] == 1
        queued = pipeline.w6_v2.read_queue("v2_typed_candidates")
        current = [
            item for item in queued
            if str(item.get("typed_candidate", {}).get("source_ref", {}).get("path") or "")
            == str(source_path)
            and item.get("review_status") in {"pending", "needs_re_review"}
        ]
        assert {item["review_scope"] for item in current} == {
            "document_layer", "fault_mapping"
        }
        for review in current:
            assert pipeline.w6_v2.mark_decision(
                "v2_typed_candidates",
                review["review_id"],
                "approve_support_only",
                reviewer="unit-reviewer",
            )["status"] == "decision_recorded"
        applied = pipeline.apply_approved_review_queue(kg_mode="v2")
        pair = next(
            item for item in applied
            if item.get("status") == "applied_to_graph_v2"
        )
        assert pair["merge_result"]["document_source_replacement"][
            "removed_document_count"
        ] == 1
        refreshed = JsonKGV2Store(root)
        documents = [
            item for item in refreshed.objects_by_type["KnowledgeDocument"]
            if item.get("source_path") == str(source_path)
        ]
        assert len(documents) == 1
        assert documents[0]["document_id"] != first_document_id


def test_approved_typed_merge_applies_once_and_writes_audit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        store = JsonKGV2Store(root)
        agent = IncrementalIngestV2Agent(store)

        item = _review_item(_graph("memory", "verified_fix"), dedupe_key="case-memory-v1")
        result = agent.apply_approved_typed_review_item(item)
        replay = agent.apply_approved_typed_review_item(item)

        assert result["status"] == "applied_to_graph_v2"
        assert result["graph_changed"] is True
        assert result["requires_sag_publish"] is True
        assert result["document_index_changed"] is True
        assert replay == {"status": "already_applied", "dedupe_key": "case-memory-v1"}
        refreshed = JsonKGV2Store(root)
        assert len(refreshed.objects_by_type["FaultFamily"]) == 1
        assert refreshed.objects_by_type["SourceCase"][0]["approved"] is True

        audit = refreshed.read_review_queue("approved_applied.json")
        assert len(audit) == 1
        row = audit[0]
        assert row["intake_id"] == "intake:case-memory-v1"
        assert row["mapping_version"] == "kg_v2_typed_admission.v1"
        assert row["admission_target"] == "fault_execution"
        assert row["materialize_allowed"] is True
        assert row["dedupe_key"] == "case-memory-v1"
        assert row["reviewer"] == "unit-reviewer"
        assert row["rollback_anchor"] == row["graph_hash_before"]
        assert row["graph_hash_before"] != row["graph_hash_after"]
        assert row["object_diff"]["FaultFamily"]["added"] == 1
        assert row["object_diff"]["ActionOutcome"]["added"] == 1
        assert row["requires_sag_publish"] is True


def test_support_only_document_approval_sets_approved_true_for_new_and_existing_documents() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        shutil.copytree("data/kg_v2/schema", root / "schema")
        store = JsonKGV2Store(root)
        document = {
            "document_id": "knowledge-document:approval-test",
            "title": "审批状态测试文档",
            "document_kind": "procedure_doc",
            "source_path": "data/raw/approval-test.docx",
            "content_hash": "a" * 64,
            "version": "",
            "owner": "",
            "approved": False,
            "source_kind": "raw_doc",
        }
        base = _review_item({
            "candidate_id": "document-approval-new",
            "objects": {"KnowledgeDocument": [document]},
            "relations": [],
        }, dedupe_key="document-approval-new")
        base["quality_gate"]["admission_target"] = "procedure_library"
        base["quality_gate"]["materialize_allowed"] = False
        base["admission_target"] = "procedure_library"
        base["materialize_allowed"] = False
        base["selected_action"] = "approve_support_only"
        new_result = IncrementalIngestV2Agent(store).apply_approved_typed_review_item(base)
        assert new_result["requires_sag_publish"] is True
        assert new_result["document_index_changed"] is True
        assert new_result["affected_object_types"] == ["KnowledgeDocument"]
        assert JsonKGV2Store(root).object_index("KnowledgeDocument")[document["document_id"]]["approved"] is True

        update = _review_item({
            "candidate_id": "document-approval-existing",
            "objects": {"KnowledgeDocument": [{**document, "approved": False}]},
            "relations": [],
        }, dedupe_key="document-approval-existing")
        update["quality_gate"]["admission_target"] = "procedure_library"
        update["quality_gate"]["materialize_allowed"] = False
        update["admission_target"] = "procedure_library"
        update["materialize_allowed"] = False
        update["selected_action"] = "approve_support_only"
        IncrementalIngestV2Agent(JsonKGV2Store(root)).apply_approved_typed_review_item(update)
        assert JsonKGV2Store(root).object_index("KnowledgeDocument")[document["document_id"]]["approved"] is True


def test_case_ready_approval_merges_case_layer_but_not_execution_objects() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        store = JsonKGV2Store(root)
        item = _review_item(_graph("case-ready", "pending_validation"), dedupe_key="case-ready")
        item["admission_readiness"] = "case_ready"
        item["quality_gate"]["admission_readiness"] = "case_ready"
        item["quality_gate"]["materialize_allowed"] = False
        item["materialize_allowed"] = False

        result = IncrementalIngestV2Agent(store).apply_approved_typed_review_item(item)

        assert result["status"] == "applied_to_graph_v2"
        refreshed = JsonKGV2Store(root)
        assert len(refreshed.objects_by_type["FaultFamily"]) == 1
        assert len(refreshed.objects_by_type["FaultVariant"]) == 1
        assert len(refreshed.objects_by_type["SourceCase"]) == 1
        assert len(refreshed.objects_by_type["EvidenceItem"]) == 1
        assert refreshed.objects_by_type["DiagnosticAction"] == []
        assert refreshed.objects_by_type["ActionOutcome"] == []
        assert refreshed.objects_by_type["DiagnosticTrace"] == []
        assert all(item.get("execution_materialize_allowed") is False for item in refreshed.objects_by_type["FaultFamily"])


def test_not_ready_approval_cannot_mutate_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        store = JsonKGV2Store(root)
        item = _review_item(_graph("not-ready", "pending_validation"), dedupe_key="not-ready")
        item["admission_readiness"] = "not_ready"
        item["quality_gate"]["admission_readiness"] = "not_ready"

        result = IncrementalIngestV2Agent(store).apply_approved_typed_review_item(item)

        assert result == {"status": "skipped", "reason": "admission_not_ready", "dedupe_key": "not-ready"}
        refreshed = JsonKGV2Store(root)
        assert all(not items for items in refreshed.objects_by_type.values())
        assert refreshed.relations == []

def test_pending_rejected_replace_and_sop_sources_do_not_mutate_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        store = JsonKGV2Store(root)
        agent = IncrementalIngestV2Agent(store)

        pending = _review_item(_graph("pending", "verified_fix"), dedupe_key="pending", status="pending")
        rejected = _review_item(_graph("rejected", "verified_fix"), dedupe_key="rejected", status="rejected")
        replace = _review_item(_graph("replace", "verified_fix"), dedupe_key="replace", operation="replace_graph")
        sop_kind = _review_item(_graph("sop-kind", "verified_fix"), dedupe_key="sop-kind")
        sop_kind["source_kind"] = "sop"
        sop_path = _review_item(_graph("sop-path", "verified_fix"), dedupe_key="sop-path")
        sop_path["source_path"] = "data/kg_v2_sop_draft_build/source.json"
        sop_document = _review_item(_graph("sop-document", "verified_fix"), dedupe_key="sop-document")
        sop_document["source_path"] = "data/raw/进板失败SOP--20250521.docx"
        w4_reject = _review_item(_graph("w4-reject", "verified_fix"), dedupe_key="w4-reject")
        w4_reject["quality_gate"]["decision"] = "reject"

        results = [
            agent.apply_approved_typed_review_item(pending),
            agent.apply_approved_typed_review_item(rejected),
            agent.apply_approved_typed_review_item(replace),
            agent.apply_approved_typed_review_item(sop_kind),
            agent.apply_approved_typed_review_item(sop_path),
            agent.apply_approved_typed_review_item(sop_document),
            agent.apply_approved_typed_review_item(w4_reject),
        ]

        assert [item["reason"] for item in results] == [
            "not_approved",
            "not_approved",
            "unsupported_operation",
            "sop_source_blocked",
            "sop_source_blocked",
            "sop_source_blocked",
            "w4_not_admitted",
        ]
        refreshed = JsonKGV2Store(root)
        assert refreshed.all_objects() == []
        assert refreshed.read_review_queue("approved_applied.json") == []


def test_human_approved_route_review_merges_without_execution_materialization() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        item = _review_item(_graph("human-reviewed", "pending_validation"), dedupe_key="human-reviewed")
        item["quality_gate"]["decision"] = "route_review"
        item["quality_gate"]["materialize_allowed"] = False
        item["materialize_allowed"] = False

        results = agent.apply_approved_batch([item])

        assert [row["status"] for row in results] == ["applied_to_graph_v2"]
        assert JsonKGV2Store(root).object_index("FaultFamily").get("family:human-reviewed")
        assert not (root / "materialized_execution" / "instances" / "errors" / "errors.json").exists()


def test_w5_rejects_gold_alignment_materialized_as_candidate_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        graph = _graph("gold-leak", "verified_fix")
        graph["objects"]["EvidenceItem"][0].update({
            "source_kind": "gold_case",
            "external_id": "goldcase-001",
            "payload_ref": "data/annotations/goldcases/gold-v1/goldcase-001.json",
        })
        item = _review_item(graph, dedupe_key="gold-leak")

        result = agent.apply_approved_typed_review_item(item)

        assert result["status"] == "skipped"
        assert result["reason"] == "alignment_provenance_invalid"
        assert result["provenance_issues"]
        assert JsonKGV2Store(root).all_objects() == []


def test_batch_materializes_only_approved_fault_execution_items() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))

        non_fault = _review_item(_graph("review-only", "verified_fix"), dedupe_key="review-only")
        non_fault["admission_target"] = "review_only"
        non_fault["materialize_allowed"] = True
        fault = _review_item(_graph("fault", "verified_fix"), dedupe_key="fault")
        results = agent.apply_approved_batch([non_fault, fault])

        assert [item["status"] for item in results] == [
            "applied_to_graph_v2",
            "applied_to_graph_v2",
            "materialized_execution_v2",
        ]
        materialized = JsonKGStore(root / "materialized_execution")
        candidates = materialized.search_errors("fault 更换内存条")
        assert candidates
        materialized_errors = json.loads((root / "materialized_execution" / "instances" / "errors" / "errors.json").read_text(encoding="utf-8"))
        assert {item["_kg_v2_family_id"] for item in materialized_errors} == {"family:fault"}
        policies = json.loads((root / "objects" / "decision_policies.json").read_text(encoding="utf-8"))
        assert len(policies) == 1
        assert policies[0]["family_id"] == "family:fault"


def test_fault_support_reusing_existing_family_does_not_disable_execution_projection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        fault = _review_item(_graph("shared", "verified_fix"), dedupe_key="shared-fault")
        assert agent.apply_approved_batch([fault])[-1]["status"] == "materialized_execution_v2"

        support = _review_item(_graph("shared", "verified_fix"), dedupe_key="shared-support")
        support["admission_target"] = "fault_support"
        support["materialize_allowed"] = False
        support["quality_gate"]["admission_target"] = "fault_support"
        support["quality_gate"]["materialize_allowed"] = False
        support_results = agent.apply_approved_batch([support])
        assert [item["status"] for item in support_results] == ["applied_to_graph_v2", "materialized_execution_v2"]

        refreshed = JsonKGV2Store(root)
        family = refreshed.object_index("FaultFamily")["family:shared"]
        assert family.get("execution_materialize_allowed") is True
        errors = json.loads((root / "materialized_execution" / "instances" / "errors" / "errors.json").read_text(encoding="utf-8"))
        assert any(item.get("_kg_v2_family_id") == "family:shared" for item in errors)


def test_single_apply_does_not_materialize_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))

        result = agent.apply_approved_typed_review_item(_review_item(_graph("single", "verified_fix"), dedupe_key="single"))

        assert result["status"] == "applied_to_graph_v2"
        assert result["materialized_counts"] == {}
        assert not (root / "materialized_execution" / "instances" / "errors" / "errors.json").exists()


def test_pending_validation_outcome_does_not_project_solution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))

        item = _review_item(_graph("pending-outcome", "pending_validation"), dedupe_key="pending-outcome")
        results = agent.apply_approved_batch([item])

        assert results[-1]["status"] == "materialized_execution_v2"
        materialized = JsonKGStore(root / "materialized_execution")
        assert materialized.solutions == []
        outcomes = json.loads((root / "materialized_execution" / "instances" / "outcomes" / "outcomes.json").read_text(encoding="utf-8"))
        assert outcomes[0]["outcome_type"] == "pending_validation"
        assert outcomes[0]["target_solution_id"] == ""


def test_approve_support_only_never_recomputes_or_materializes_execution_view() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        item = _review_item(_graph("support-only", "pending_validation"), dedupe_key="support-only")
        item["graph"]["objects"]["ActionOutcome"][0]["outcome_origin"] = "synthetic_fallback"
        item["quality_gate"].update({
            "decision": "route_review",
            "policy_readiness": "pending_only",
            "materialize_allowed": False,
        })
        item["materialize_allowed"] = False
        item["selected_action"] = "approve_support_only"
        item["human_approved"] = True

        results = agent.apply_approved_batch([item])

        assert [result["status"] for result in results] == ["applied_to_graph_v2"]
        refreshed = JsonKGV2Store(root)
        assert refreshed.objects_by_type["DiagnosticAction"]
        assert refreshed.objects_by_type["ActionOutcome"][0]["outcome_type"] == "pending_validation"
        assert refreshed.objects_by_type["DiagnosticTrace"]
        assert all(
            row.get("execution_materialize_allowed") is False
            for object_type in ("DiagnosticAction", "ActionOutcome", "DiagnosticTrace")
            for row in refreshed.objects_by_type[object_type]
        )
        assert refreshed.objects_by_type["DecisionPolicy"] == []
        assert not (root / "materialized_execution" / "instances").exists()


def test_approve_for_execution_policy_materializes_w4_admitted_fault_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        item = _review_item(_graph("execution-policy", "verified_fix"), dedupe_key="execution-policy")
        item["selected_action"] = "approve_for_execution_policy"
        item["human_approved"] = True

        results = agent.apply_approved_batch([item])

        assert results[-1]["status"] == "materialized_execution_v2"
        assert JsonKGV2Store(root).objects_by_type["DecisionPolicy"]


def test_approve_for_execution_policy_materializes_w6_approved_route_review() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        item = _review_item(_graph("reviewed-execution-policy", "pending_validation"), dedupe_key="reviewed-execution-policy")
        item["graph"]["objects"]["ActionOutcome"][0]["outcome_origin"] = "human_reviewed"
        item["quality_gate"]["decision"] = "route_review"
        item["quality_gate"]["policy_readiness"] = "pending_only"
        item["quality_gate"]["materialize_allowed"] = False
        item["materialize_allowed"] = False
        item["selected_action"] = "approve_for_execution_policy"
        item["human_approved"] = True

        results = agent.apply_approved_batch([item])

        assert results[-1]["status"] == "materialized_execution_v2"
        refreshed = JsonKGV2Store(root)
        assert refreshed.objects_by_type["DecisionPolicy"]
        assert all(
            row.get("execution_materialize_allowed") is True
            for row in refreshed.objects_by_type["FaultFamily"]
        )


def test_route_review_execution_action_without_human_approval_does_not_apply() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        agent = IncrementalIngestV2Agent(JsonKGV2Store(root))
        item = _review_item(_graph("unapproved-execution-policy", "verified_fix"), dedupe_key="unapproved-execution-policy")
        item["quality_gate"]["decision"] = "route_review"
        item["quality_gate"]["materialize_allowed"] = False
        item["materialize_allowed"] = False
        item["selected_action"] = "approve_for_execution_policy"
        item["human_approved"] = False
        item["review_status"] = "pending"

        results = agent.apply_approved_batch([item])

        assert results == [{
            "status": "skipped",
            "reason": "not_approved",
            "dedupe_key": "unapproved-execution-policy",
        }]
        assert JsonKGV2Store(root).all_objects() == []


def test_dry_run_supports_typed_and_thin_payload_graph_envelopes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = IncrementalIngestV2Agent(JsonKGV2Store(Path(tmp) / "kg_v2"))

        typed = {
            "typed_candidate": {
                "intake_id": "intake:typed:dry-run",
                "dedupe_key": "typed:dry-run",
                "payload": _graph("typed-dry-run", "verified_fix"),
            },
            "quality_gate": _quality_gate(),
        }
        thin = {
            "dedupe_key": "typed:thin-graph",
            "payload": {
                "graph": _graph("thin-graph", "verified_fix"),
            },
        }

        typed_plan = agent.dry_run_merge_plan(typed)
        thin_plan = agent.dry_run_merge_plan(thin)

        assert typed_plan["schema_valid"] is True
        assert typed_plan["intake_id"] == "intake:typed:dry-run"
        assert typed_plan["dedupe_key"] == "typed:dry-run"
        assert thin_plan["schema_valid"] is True
        assert thin_plan["object_counts"]["FaultFamily"] == 1


def test_w6_typed_queue_is_canonical_apply_path_and_audit_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "kg_v2"
        store = JsonKGV2Store(root)
        agent = IncrementalIngestV2Agent(store)
        review = ReviewQueueAgent(store)
        gate = _quality_gate()
        envelope = {
            "intake_id": "intake:w6:fault",
            "dedupe_key": "typed:w6:fault",
            "source_kind": "chat_review",
            "payload": {
                **_graph("w6-fault", "verified_fix"),
                "admission_target": "fault_execution",
                "raw_text": "蓝屏 MEMORY_MANAGEMENT，更换内存后未再出现。",
            },
        }
        typed_item = review.build_typed_review_item(
            envelope,
            gate,
            dry_run_plan=agent.dry_run_merge_plan({"typed_candidate": envelope, "quality_gate": gate}),
        )
        assert "typed_candidate" in typed_item
        assert "candidate" not in typed_item
        assert review.enqueue("v2_typed_candidates", typed_item)["status"] == "queued"
        assert review.mark_decision("v2_typed_candidates", typed_item["review_id"], "approve", reviewer="qa")["human_approved"] is True

        legacy_item = _review_item(_graph("legacy-queue", "verified_fix"), dedupe_key="legacy-queue")
        assert review.enqueue("candidates", legacy_item)["status"] == "queued"

        results = agent.apply_approved_review_queue(review)

        assert [item["status"] for item in results] == ["applied_to_graph_v2", "materialized_execution_v2"]
        materialized_errors = json.loads((root / "materialized_execution" / "instances" / "errors" / "errors.json").read_text(encoding="utf-8"))
        assert {item["_kg_v2_family_id"] for item in materialized_errors} == {"family:w6-fault"}
        audit = JsonKGV2Store(root).read_review_queue("approved_applied.json")
        assert len(audit) == 1
        row = audit[0]
        assert row["intake_id"] == "intake:w6:fault"
        assert row["mapping_version"] == "kg_v2_typed_admission.v1"
        assert row["admission_target"] == "fault_execution"
        assert row["materialize_allowed"] is True
        assert row["reviewer"] == "qa"
        assert row["graph_hash_before"] != row["graph_hash_after"]
        assert row["rollback_anchor"] == row["graph_hash_before"]
        assert row["object_diff"]["FaultFamily"]["added"] == 1


def _review_item(
    graph: dict,
    *,
    dedupe_key: str,
    status: str = "approved",
    operation: str = "merge_graph",
) -> dict:
    return {
        "review_item_type": "kg_v2.typed_review_item.v1",
        "operation": operation,
        "dedupe_key": dedupe_key,
        "intake_id": f"intake:{dedupe_key}",
        "review_status": status,
        "reviewer": "unit-reviewer",
        "source_type": "case",
        "source_path": f"data/non_sop/{dedupe_key}.json",
        "admission_target": "fault_execution",
        "materialize_allowed": True,
        "quality_gate": _quality_gate(),
        "graph": graph,
    }


def _quality_gate() -> dict:
    return {
        "decision": "admit",
        "admission_target": "fault_execution",
        "materialize_allowed": True,
        "mapping_version": "kg_v2_typed_admission.v1",
    }


def _graph(suffix: str, outcome_type: str) -> dict:
    family_id = f"family:{suffix}"
    variant_id = f"variant:{suffix}"
    action_id = f"action:{suffix}:replace-memory"
    outcome_id = f"outcome:{suffix}"
    trace_id = f"trace:{suffix}"
    evidence_id = f"evidence:{suffix}"
    case_id = f"case:{suffix}"
    objects = {
        "FaultFamily": [
            {"family_id": family_id, "label": f"{suffix} 蓝屏", "summary": "蓝屏", "category": "系统与软件异常", "source_kind": "case"}
        ],
        "FaultVariant": [
            {"variant_id": variant_id, "family_id": family_id, "label": f"{suffix} 换内存后未复现", "summary": "换内存后未复现", "equipment_type": "", "site": "", "software_version": "", "error_phase": "", "owner_context": "", "escalation_target": "", "keywords": [suffix, "蓝屏"]}
        ],
        "DiagnosticAction": [
            {
                "action_id": action_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "label": "更换内存条",
                "summary": "更换内存条验证",
                "action_role": "change",
                "execution_status": "actual",
                "step_order": 1,
                "source_kind": "case",
                "evidence_ids": [evidence_id],
            }
        ],
        "ActionOutcome": [
            {"outcome_id": outcome_id, "family_id": family_id, "variant_id": variant_id, "action_id": action_id, "outcome_type": outcome_type, "summary": "更换内存条后观察结果", "source_case_id": case_id, "evidence_ids": [evidence_id], "high_cost": False, "destructive": False, "root_cause_summary": "内存问题" if outcome_type == "verified_fix" else ""}
        ],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [
            {"trace_id": trace_id, "family_id": family_id, "variant_id": variant_id, "source_case_id": case_id, "summary": "排查链", "recommended_action_ids": [action_id], "actual_action_ids": [action_id], "evidence_ids": [evidence_id]}
        ],
        "DecisionPolicy": [],
        "EvidenceItem": [
            {"evidence_id": evidence_id, "source_kind": "chat_message", "external_id": f"m:{suffix}", "title": "消息", "summary": "消息", "payload_ref": ""}
        ],
        "SourceCase": [
            {"case_id": case_id, "source_kind": "manual_review", "title": "case", "summary": "case", "source_ref": f"ep:{suffix}", "approved": True}
        ],
    }
    relations = [
        {"from": family_id, "to": variant_id, "relation": "has_variant"},
        {"from": variant_id, "to": trace_id, "relation": "has_trace"},
        {"from": variant_id, "to": outcome_id, "relation": "has_outcome"},
        {"from": trace_id, "to": action_id, "relation": "used_action"},
        {"from": outcome_id, "to": action_id, "relation": "outcome_of"},
        {"from": case_id, "to": variant_id, "relation": "supports"},
        {"from": case_id, "to": trace_id, "relation": "supports"},
        {"from": case_id, "to": outcome_id, "relation": "supports"},
        {"from": evidence_id, "to": case_id, "relation": "evidences"},
        {"from": evidence_id, "to": outcome_id, "relation": "evidences"},
    ]
    return {"objects": objects, "relations": relations}
