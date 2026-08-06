from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from debug_agent_system.agents.write import non_sop_intake as nsi
from debug_agent_system.agents.write import review_context as rc
from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_minimal_kg_v2(root: Path) -> None:
    _write_json(root / "schema" / "object-types.json", {"schema_version": "test.objects.v1", "types": ["FaultFamily"]})
    _write_json(root / "schema" / "link-types.json", {"schema_version": "test.links.v1", "links": ["has_variant"]})
    _write_json(root / "objects" / "fault_families.json", [
        {
            "family_id": "family:user-config",
            "label": "用户配置加载失败",
            "summary": "主程序初始化阶段加载 user.cfg.toml 失败。",
            "category": "系统与软件异常",
            "subsystem": "主程序配置",
        }
    ])
    _write_json(root / "objects" / "fault_variants.json", [
        {
            "variant_id": "variant:user-cfg-empty",
            "family_id": "family:user-config",
            "label": "更换工控机后 user.cfg.toml 为空导致加载用户配置失败",
            "summary": "更换工控机后主程序报警加载用户配置失败。",
            "error_phase": "初始化阶段",
            "keywords": ["user.cfg.toml", "加载用户配置失败"],
        }
    ])
    _write_json(root / "objects" / "diagnostic_actions.json", [
        {
            "action_id": "action:check-user-cfg",
            "family_id": "family:user-config",
            "variant_id": "variant:user-cfg-empty",
            "label": "检查 user.cfg.toml 是否为空",
            "summary": "确认配置文件是否为空白或损坏。",
            "action_role": "inspect",
            "step_order": 1,
        }
    ])
    _write_json(root / "objects" / "required_info_specs.json", [
        {
            "required_info_id": "required-info:program-file",
            "family_id": "family:user-config",
            "variant_id": "variant:user-cfg-empty",
            "slot": "program_file",
            "question": "请提供 user.cfg.toml 和 conf 目录内容。",
            "why_required": "判断配置文件为空、损坏还是备份选择错误。",
            "priority": "high",
        }
    ])
    for name in ("action_outcomes", "decision_policies", "diagnostic_traces", "evidence_items", "source_cases"):
        _write_json(root / "objects" / f"{name}.json", [])
    _write_json(root / "relations" / "edges.json", [
        {"from": "family:user-config", "to": "variant:user-cfg-empty", "relation": "has_variant"}
    ])
    _write_json(root / "materialized_execution" / "edges.json", [
        {"from": "errv2:variant:user-cfg-empty", "to": "checkv2:action:check-user-cfg", "relation": "has_check"}
    ])
    _write_json(root / "gold_cases" / "goldcase-001.json", {
        "schema_version": "kg_v2.gold_case.v1",
        "case_id": "goldcase-001",
        "status": "reviewed",
        "source_episode_id": "ep:gold",
        "source_excerpt": ["更换工控机后主程序报警加载用户配置失败，怀疑 user.cfg.toml 为空。"],
        "gold": {
            "cases": [{"family": {"label": "用户配置加载失败"}, "variant": {"label": "配置为空"}}],
            "family": {"label": "用户配置加载失败"},
            "variant": {"label": "更换工控机后 user.cfg.toml 为空导致加载用户配置失败"},
            "actions": [{"label": "检查 user.cfg.toml 是否为空"}],
            "outcomes": [{"action_label": "检查 user.cfg.toml 是否为空", "outcome_type": "diagnostic_method"}],
            "required_info": [{"slot": "program_file", "question": "请提供 user.cfg.toml 和 conf 目录内容。"}],
            "trace": {"recommended_action_labels": ["检查 user.cfg.toml 是否为空"]},
        },
    })


def test_write_intake_envelope_has_contract_fields_and_stable_ids() -> None:
    first = nsi.build_write_intake_envelope(
        source_type="chat",
        source_ref={"thread_id": "t1"},
        knowledge_kind="support",
        payload={"text": "加载用户配置失败", "messages": [{"id": "m1", "text": "加载用户配置失败"}]},
        evidence_pack={"anchors": [{"message_id": "m1"}]},
        lineage={"collector": "unit-test"},
        metadata={"compat": True},
    )
    second = nsi.build_write_intake_envelope(
        source_type="chat",
        source_ref={"thread_id": "t1"},
        knowledge_kind="support",
        payload={"messages": [{"text": "加载用户配置失败", "id": "m1"}], "text": "加载用户配置失败"},
        evidence_pack={"anchors": [{"message_id": "m1"}]},
        lineage={"collector": "unit-test"},
        metadata={"compat": True},
    )
    assert set([
        "intake_id",
        "source_type",
        "source_ref",
        "knowledge_kind",
        "payload",
        "evidence_pack",
        "lineage",
        "dedupe_key",
    ]).issubset(first)
    assert first["schema_version"] == "debug_agent_system.write_intake_envelope.v1"
    assert first["intake_id"] == second["intake_id"]
    assert first["dedupe_key"] == second["dedupe_key"]
    assert first["intake_id"].startswith("intake:")
    assert first["dedupe_key"].startswith("dedupe:")
    assert first["knowledge_kind"] == "support"
    assert first["payload"]["text"] == "加载用户配置失败"
    assert first["text"] == "加载用户配置失败"
    assert first["metadata"] == {"compat": True}


def test_write_intake_ids_ignore_scalar_evidence_list_order() -> None:
    first = nsi.build_write_intake_envelope(
        source_type="chat",
        source_ref={"episode_id": "ep:1", "message_ids": ["m2", "m1"]},
        payload={"text": "蓝屏", "evidence_message_ids": ["m2", "m1"]},
        evidence_pack={"message_ids": ["m2", "m1"]},
    )
    second = nsi.build_write_intake_envelope(
        source_type="chat",
        source_ref={"episode_id": "ep:1", "message_ids": ["m1", "m2"]},
        payload={"text": "蓝屏", "evidence_message_ids": ["m1", "m2"]},
        evidence_pack={"message_ids": ["m1", "m2"]},
    )

    assert first["intake_id"] == second["intake_id"]
    assert first["dedupe_key"] == second["dedupe_key"]


def test_same_source_updates_one_intake_while_content_hash_tracks_candidate_change() -> None:
    first = nsi.build_write_intake_envelope(
        source_type="text_history",
        source_ref={"episode_id": "ep:1", "message_ids": ["m1"]},
        payload={"text": "蓝屏", "objects": {"DiagnosticAction": [{"label": "检查内存"}]}},
    )
    second = nsi.build_write_intake_envelope(
        source_type="text_history",
        source_ref={"episode_id": "ep:1", "message_ids": ["m1"]},
        payload={"text": "蓝屏", "objects": {"DiagnosticAction": [{"label": "更换内存"}]}},
    )

    assert first["intake_id"] == second["intake_id"]
    assert first["dedupe_key"] == second["dedupe_key"]
    assert first["content_hash"] != second["content_hash"]


def test_content_hash_preserves_diagnostic_action_order() -> None:
    first = nsi.build_write_intake_envelope(
        source_type="text_history",
        source_ref={"episode_id": "ep:1", "message_ids": ["m1"]},
        payload={"text": "蓝屏", "recommended_action_ids": ["action:memory", "action:driver"]},
    )
    second = nsi.build_write_intake_envelope(
        source_type="text_history",
        source_ref={"episode_id": "ep:1", "message_ids": ["m1"]},
        payload={"text": "蓝屏", "recommended_action_ids": ["action:driver", "action:memory"]},
    )

    assert first["dedupe_key"] == second["dedupe_key"]
    assert first["content_hash"] != second["content_hash"]


def test_write_intake_rejects_sop_source_and_sop_build_ref() -> None:
    try:
        nsi.build_write_intake_envelope(source_type="sop", text="加载用户配置失败")
    except nsi.NonSopIntakeError as exc:
        assert exc.to_dict()["code"] == "sop_source_rejected"
    else:
        raise AssertionError("expected sop source_type to be rejected")

    try:
        nsi.build_write_intake_envelope(source_type="chat", source_kind="sop", text="加载用户配置失败")
    except nsi.NonSopIntakeError as exc:
        assert exc.to_dict()["code"] == "sop_source_rejected"
    else:
        raise AssertionError("expected sop source_kind to be rejected")

    result = nsi.try_build_write_intake_envelope(
        source_type="raw_doc",
        text="加载用户配置失败",
        source_ref={"path": "data/kg_v2_sop_draft_build/cards/source.json"},
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "sop_build_path_rejected"


def test_write_intake_accepts_only_explicit_versioned_sop_document_contract() -> None:
    envelope = nsi.build_write_intake_envelope(
        source_type="sop_doc",
        source_kind="sop",
        source_ref={"path": "data/raw/相机标准操作流程.md"},
        text="相机断连排查步骤",
        metadata={
            "incremental_source_contract": nsi.SOP_INCREMENTAL_CONTRACT,
        },
    )
    assert envelope["source_type"] == "sop_doc"
    assert envelope["source_kind"] == "sop"

    try:
        nsi.build_write_intake_envelope(
            source_type="sop_doc",
            source_kind="sop",
            source_ref={"path": "data/raw/相机标准操作流程.md"},
            text="相机断连排查步骤",
        )
    except nsi.NonSopIntakeError as exc:
        assert exc.code == "invalid_sop_incremental_contract"
    else:
        raise AssertionError("expected explicit SOP contract rejection")

    try:
        nsi.build_write_intake_envelope(
            source_type="sop_doc",
            source_kind="sop",
            source_ref={
                "path": "data/kg_v2_sop_draft_build/manual_card.json"
            },
            text="相机断连排查步骤",
            metadata={
                "incremental_source_contract": nsi.SOP_INCREMENTAL_CONTRACT,
            },
        )
    except nsi.NonSopIntakeError as exc:
        assert exc.code == "sop_build_path_rejected"
    else:
        raise AssertionError("expected curated build path rejection")


def test_kg_v2_graph_hash_is_stable_and_includes_materialized_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kg_root = Path(tmp) / "kg_v2"
        _write_minimal_kg_v2(kg_root)

        first = nsi.compute_kg_v2_graph_hash(kg_root)
        second = nsi.compute_kg_v2_graph_hash(kg_root)
        assert first == second
        assert len(first) == 64

        _write_json(kg_root / "materialized_execution" / "edges.json", [
            {"relation": "has_check", "to": "checkv2:action:check-user-cfg", "from": "errv2:variant:user-cfg-empty"},
            {"from": "checkv2:action:check-user-cfg", "to": "tracev2:trace:user-config", "relation": "supports_trace"},
        ])
        assert nsi.compute_kg_v2_graph_hash(kg_root) != first


def test_alignment_background_uses_gold_cases_as_reference_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kg_root = Path(tmp) / "kg_v2"
        _write_minimal_kg_v2(kg_root)
        envelope = nsi.build_write_intake_envelope(
            source_type="chat",
            text="更换工控机后主程序报警加载用户配置失败，user.cfg.toml 可能为空。",
        )

        background = nsi.build_alignment_only_background(envelope, kg_v2_root=kg_root, gold_root=kg_root / "gold_cases")
        assert background["graph_ingestion"] is False
        assert background["reviewed_case_examples"][0]["case_id"] == "goldcase-001"
        assert background["reviewed_case_examples"][0]["graph_ingestion"] is False
        gold_structure = background["reviewed_case_examples"][0]["gold_structure"]
        assert gold_structure["outcomes"][0]["outcome_type"] == "diagnostic_method"
        assert gold_structure["cases"][0]["family"]["label"] == "用户配置加载失败"


def test_alignment_background_prioritizes_exact_gold_source_episode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kg_root = Path(tmp) / "kg_v2"
        _write_minimal_kg_v2(kg_root)
        envelope = nsi.build_write_intake_envelope(
            source_type="manual_review",
            source_ref={"episode_id": "ep:gold"},
            text="与词面无关但属于同一条已审核案例",
        )

        background = nsi.build_alignment_only_background(envelope, kg_v2_root=kg_root, gold_root=kg_root / "gold_cases")

        assert background["reviewed_case_examples"][0]["case_id"] == "goldcase-001"
        assert background["reviewed_case_examples"][0]["exact_source_match"] is True
        assert background["reviewed_case_examples"][0]["selection_reason"] == "exact_source_match"
        assert background["reviewed_case_examples"][0]["graph_ingestion"] is False


def test_alignment_background_empty_recall_allows_new_family() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kg_root = Path(tmp) / "kg_v2"
        _write_minimal_kg_v2(kg_root)
        envelope = nsi.build_write_intake_envelope(
            source_type="attachment",
            text="zzzz-unmatched-novel-fault",
        )

        background = nsi.build_alignment_only_background(envelope, kg_v2_root=kg_root, gold_root=kg_root / "gold_cases")
        assert background["context_role"] == "alignment_only"
        assert background["allows_new_family"] is True
        assert background["recalled_background"] == []
        assert background["reviewed_case_examples"]
        assert background["reviewed_case_examples"][0]["selection_reason"] == "fallback_style_reference"
        assert all(item["graph_ingestion"] is False for item in background["reviewed_case_examples"])
        assert background["intake_id"] == envelope["intake_id"]
        assert background["dedupe_key"] == envelope["dedupe_key"]


def test_w7_injects_gold_case_reference_into_w2_episode_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        kg_v2_root = root / "kg_v2"
        _write_minimal_kg_v2(kg_v2_root)
        pipeline = WriteSidePipeline(
            JsonKGStore(root / "legacy_kg"),
            kg_v2_root=kg_v2_root,
            review_context_gold_root=kg_v2_root / "gold_cases",
        )
        episode = {
            "episode_id": "ep:new-family",
            "thread_id": "thread:1",
            "fault_description_messages": [{"message_id": "m1", "text": "完全陌生的新故障现象"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "evidence_message_ids": ["m1"],
            "source_offsets": [{"message_id": "m1", "index": 0}],
            "extracted": {},
        }

        prepared = pipeline._prepare_episode_for_w2(episode, source_type="chat")
        background = prepared["extracted"]["review_context"]

        assert background["context_role"] == "alignment_only"
        assert background["facts_may_not_be_copied_as_new_evidence"] is True
        assert background["reviewed_case_examples"]
        assert background["reviewed_case_examples"][0]["graph_ingestion"] is False


def test_alignment_background_marks_non_evidence_and_recalls_kg_rows_without_sop_build_read() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kg_root = Path(tmp) / "kg_v2"
        _write_minimal_kg_v2(kg_root)
        forbidden = Path(tmp) / "data" / "kg_v2_sop_draft_build" / "poison.json"
        _write_json(forbidden, {"must_not_read": True})
        original_read_text = Path.read_text

        def guarded_read_text(self: Path, *args: object, **kwargs: object) -> str:
            assert "data/kg_v2_sop_draft_build" not in self.as_posix()
            return original_read_text(self, *args, **kwargs)

        envelope = nsi.build_write_intake_envelope(
            source_type="jira",
            text="加载用户配置失败，请检查 user.cfg.toml 和 conf 目录。",
            source_ref={"jira_id": "JIRA-1"},
        )

        with patch.object(Path, "read_text", guarded_read_text):
            background = rc.build_non_sop_alignment_background(envelope, kg_v2_root=kg_root, gold_root=kg_root / "gold_cases")
        row = background["recalled_background"][0]
        assert background["context_role"] == "alignment_only"
        assert background["facts_may_not_be_copied_as_new_evidence"] is True
        assert row["family"]["label"] == "用户配置加载失败"
        assert row["variant"]["label"] == "更换工控机后 user.cfg.toml 为空导致加载用户配置失败"
        assert row["actions"][0]["label"] == "检查 user.cfg.toml 是否为空"
        assert row["required_info"][0]["slot"] == "program_file"
