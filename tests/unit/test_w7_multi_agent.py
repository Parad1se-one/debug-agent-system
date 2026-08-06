from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from debug_agent_system.agents.write import WriteSidePipeline
from debug_agent_system.agents.write.w7_trace.candidate_graph import (
    build_sparse_candidate_graph,
)
from debug_agent_system.agents.write.w7_trace.atomic_case_adapter import (
    build_atomic_case_manifest,
    w2_atomic_episodes,
)
from debug_agent_system.agents.write.w7_trace.components import (
    apply_candidate_edge_safety_guards,
    apply_component_bridge_decision,
    apply_component_consistency_decision,
    build_component_bridge_candidates,
    build_component_conflicts,
    build_trace_components,
)
from debug_agent_system.agents.write.w7_trace.contracts import (
    TRACE_ASSEMBLY_CASE_KINDS,
    resolve_w7_mode,
    validate_case_boundary_decision,
    validate_evidence_anchor_decision,
    validate_outcome_patch,
    validate_trace_link_decision,
    validate_trace_phase_patch,
)
from debug_agent_system.agents.write.w7_trace.model_client import (
    DeepSeekDecisionModelClient,
)
from debug_agent_system.agents.write.w2_extract.deepseek_client import (
    DeepSeekToolCallError,
)
from debug_agent_system.agents.write.w7_trace.orchestrator import (
    W7ShadowOrchestrator,
)
from debug_agent_system.agents.write.w7_trace.source_context import (
    attach_case_source_context,
    build_episode_source_ledger,
    evidence_anchor_candidates,
)
from debug_agent_system.agents.write.w7_trace.trace_compiler import TraceCompiler
from debug_agent_system.agents.write.w7_trace.checkpoint_store import (
    CheckpointStore,
)
from debug_agent_system.agents.write.w7_trace.review import (
    approval_hash_matches,
    build_correction_event,
    build_trace_review_payload,
    correction_chain_subject_hash,
)
from debug_agent_system.agents.write.w7_trace.correction_compiler import (
    compile_trace_corrections,
    materialize_corrected_typed_candidate,
)
from debug_agent_system.agents.write.w7_trace.batch_orchestrator import (
    W7BatchShadowOrchestrator,
    _enrich_unit_card_scope,
    build_batch_source_ledger,
)
from debug_agent_system.agents.write.w7_trace.batch_candidate import (
    build_w7_batch_typed_candidate,
)
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.agents.write_v2.ingest import IncrementalIngestV2Agent
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2 import JsonKGV2Store
from debug_agent_system.eval.write_side.build_w7_calibration_input import (
    build as build_w7_calibration_input,
)
from debug_agent_system.eval.write_side.w7_multi_agent_score import (
    dedupe_strings_for_score,
    score as score_w7_multi_agent,
)
from debug_agent_system.eval.write_side.w7_multi_agent_safety_gate import (
    build_report as build_w7_safety_report,
)


class FakeDecisionClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call_tool(self, **kwargs):
        name = kwargs["tool"]["function"]["name"]
        self.calls.append(name)
        if name == "decide_atomic_case_boundaries":
            rows = kwargs["payload"]["source_ledger"]["rows"]
            fragments = []
            for index, row in enumerate(rows, 1):
                text = str(row.get("text") or "")
                fragments.append({
                    "fragment_ref": f"F{index}",
                    "case_kind": "diagnostic_case",
                    "fault_summary": text,
                    "source_message_ids": [row["message_id"]],
                    "evidence_spans": [{
                        "message_id": row["message_id"],
                        "start": 0,
                        "end": max(1, len(text)),
                    }],
                    "uncertainties": [],
                })
            arguments = {
                "case_fragments": fragments,
                "non_case_message_ids": [],
                "uncertainties": [],
            }
        elif name == "attach_evidence_to_case_fragments":
            candidates = kwargs["payload"]["evidence_candidates"]
            fragment_ref = kwargs["payload"]["allowed_fragment_refs"][0]
            arguments = {
                "anchor_decisions": [{
                    "evidence_message_id": item["message_id"],
                    "attachment_ids": [
                        attachment["attachment_id"]
                        for attachment in item.get("attachment_refs") or []
                    ],
                    "target_fragment_ref": fragment_ref,
                    "role": "initial_diagnostic_package",
                    "confidence": "high",
                    "reasons": ["后续消息明确描述该诊断包"],
                } for item in candidates],
                "unassigned_evidence_message_ids": [],
                "uncertainties": [],
            }
        elif name == "adjudicate_trace_candidate_edges":
            arguments = {
                "edge_decisions": [{
                    "left_case_ref": item["left_case_ref"],
                    "right_case_ref": item["right_case_ref"],
                    "decision": "must_link",
                    "relation_hint": "continuation_of",
                    "evidence_message_ids": [],
                    "reasons": ["共享 Jira 和设备"],
                } for item in kwargs["payload"]["candidate_edges"]],
                "uncertainties": [],
            }
        elif name == "assign_trace_groups_and_phases":
            case_refs = kwargs["payload"]["allowed_case_refs"]
            operations = [{
                "op": "create_trace_group",
                "local_trace_ref": "T1",
                "case_refs": case_refs,
                "case_ref": "",
                "event_type": "",
                "relation_type": "",
                "phase_index": 0,
                "after_case_ref": "",
                "evidence_message_ids": [],
                "summary": "同一纵向问题",
            }]
            for index, case_ref in enumerate(case_refs, 1):
                operations.append({
                    "op": "set_phase",
                    "local_trace_ref": "T1",
                    "case_refs": [],
                    "case_ref": case_ref,
                    "event_type": "report" if index == 1 else "validation",
                    "relation_type": (
                        "trace_root" if index == 1 else "validation_of"
                    ),
                    "phase_index": index,
                    "after_case_ref": (
                        "" if index == 1 else case_refs[index - 2]
                    ),
                    "evidence_message_ids": [],
                    "summary": case_ref,
                })
            arguments = {"operations": operations, "uncertainties": []}
        elif name == "reconcile_trace_outcomes":
            trace_ref = kwargs["payload"]["allowed_trace_refs"][0]
            evidence_id = (
                "m2"
                if "m2" in kwargs["payload"]["allowed_message_ids"]
                else kwargs["payload"]["allowed_message_ids"][0]
            )
            arguments = {
                "operations": [{
                    "op": "revise_trace_status",
                    "local_trace_ref": trace_ref,
                    "from": "provisionally_resolved",
                    "to": "verified",
                    "evidence_message_ids": [evidence_id],
                    "reason": "设备恢复正常生产",
                }],
                "uncertainties": [],
            }
        else:
            raise AssertionError(name)
        return {
            "arguments": arguments,
            "model": "fake",
            "finish_reason": "tool_calls",
            "usage": {},
        }


class FailingDecisionClient:
    def call_tool(self, **kwargs):
        raise RuntimeError("malformed tool arguments")


class DuplicateNeighborDecisionClient:
    def call_tool(self, **kwargs):
        candidates = kwargs["payload"]["candidate_edges"]
        values = []
        for item in candidates:
            decision = {
                "left_case_ref": item["left_case_ref"],
                "right_case_ref": item["right_case_ref"],
                "decision": "must_link",
                "relation_hint": "continuation_of",
                "evidence_message_ids": [],
                "reasons": ["同一故障"],
            }
            values.extend([decision, deepcopy(decision)])
        return {
            "arguments": {
                "edge_decisions": values,
                "uncertainties": [],
            },
            "model": "fake",
            "finish_reason": "tool_calls",
            "usage": {},
        }


class FlakyNeighborDecisionClient(FakeDecisionClient):
    def __init__(self) -> None:
        super().__init__()
        self.neighbor_attempts = 0

    def call_tool(self, **kwargs):
        if (
            kwargs["tool"]["function"]["name"]
            == "adjudicate_trace_candidate_edges"
        ):
            self.neighbor_attempts += 1
            if self.neighbor_attempts < 3:
                raise RuntimeError("temporary transport failure")
        return super().call_tool(**kwargs)


class ContradictoryNeighborDecisionClient:
    def __init__(self, *, repair_on_retry: bool = True) -> None:
        self.calls = 0
        self.repair_on_retry = repair_on_retry

    def call_tool(self, **kwargs):
        self.calls += 1
        item = kwargs["payload"]["candidate_edges"][0]
        decision = (
            "cannot_link"
            if self.repair_on_retry and self.calls > 1
            else "must_link"
        )
        reasons = (
            ["同一日报中的两个不同问题，仅共享工作上下文"]
            if self.calls == 1
            else ["相机断连与电感漏检是不同故障"]
        )
        return {
            "arguments": {
                "edge_decisions": [{
                    "left_case_ref": item["left_case_ref"],
                    "right_case_ref": item["right_case_ref"],
                    "decision": decision,
                    "relation_hint": (
                        "continuation_of"
                        if decision == "must_link"
                        else ""
                    ),
                    "evidence_message_ids": [],
                    "reasons": reasons,
                }],
                "uncertainties": [],
            },
            "model": "fake",
            "finish_reason": "tool_calls",
            "usage": {},
        }


class RepairingTracePhaseClient:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, **kwargs):
        self.calls += 1
        operations = [{
            "op": "create_trace_group",
            "local_trace_ref": "T1",
            "case_refs": ["C1"],
            "case_ref": "",
            "event_type": "",
            "relation_type": "",
            "phase_index": 0,
            "after_case_ref": "",
            "evidence_message_ids": [],
            "summary": "蓝屏",
        }, {
            "op": "set_phase",
            "local_trace_ref": "T1",
            "case_refs": [],
            "case_ref": "C1",
            "event_type": "report",
            "relation_type": "trace_root",
            "phase_index": 1,
            "after_case_ref": "",
            "evidence_message_ids": [],
            "summary": "首报",
        }]
        if self.calls == 1:
            operations.append({
                **operations[-1],
                "phase_index": 2,
                "summary": "重复分配",
            })
        return {
            "arguments": {
                "operations": operations,
                "uncertainties": [],
            },
            "model": "fake",
            "finish_reason": "tool_calls",
            "usage": {},
        }


class DuplicateTracePhaseClient:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, **kwargs):
        self.calls += 1
        return {
            "arguments": {
                "operations": [{
                    "op": "create_trace_group",
                    "local_trace_ref": "T1",
                    "case_refs": ["C1"],
                    "case_ref": "",
                    "event_type": "",
                    "relation_type": "",
                    "phase_index": 0,
                    "after_case_ref": "",
                    "evidence_message_ids": [],
                    "summary": "蓝屏",
                }, {
                    "op": "set_phase",
                    "local_trace_ref": "T1",
                    "case_refs": [],
                    "case_ref": "C1",
                    "event_type": "report",
                    "relation_type": "trace_root",
                    "phase_index": 1,
                    "after_case_ref": "",
                    "evidence_message_ids": [],
                    "summary": "首报",
                }, {
                    "op": "set_phase",
                    "local_trace_ref": "T1",
                    "case_refs": [],
                    "case_ref": "C1",
                    "event_type": "resolution",
                    "relation_type": "trace_root",
                    "phase_index": 2,
                    "after_case_ref": "",
                    "evidence_message_ids": [],
                    "summary": "同一原子案例内的恢复",
                }],
                "uncertainties": [],
            },
            "model": "fake",
            "finish_reason": "tool_calls",
            "usage": {},
        }


def _episode() -> dict:
    return {
        "episode_id": "ep-1",
        "thread_id": "thread-1",
        "fault_description_messages": [
            {"message_id": "m1", "text": "设备蓝屏"}
        ],
        "resolution_messages": [
            {"message_id": "m2", "text": "更换内存后恢复正常生产"}
        ],
    }


def _attachment_episode() -> dict:
    return {
        "episode_id": "ep-attachment",
        "thread_id": "thread-1",
        "fault_description_messages": [{
            "message_id": "m0",
            "text": "",
            "message_type": "file",
            "attachment_metadata": [{
                "file_key": "file-diagnostic-1",
                "name": "DiagnosticData.zip",
            }],
        }, {
            "message_id": "m1",
            "text": "设备蓝屏，请分析前面的诊断数据",
        }],
    }


def test_w7_mode_defaults_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("W7_MODE", raising=False)
    assert resolve_w7_mode() == "legacy"
    assert resolve_w7_mode("shadow_multi_agent") == "shadow_multi_agent"
    with pytest.raises(ValueError, match="unsupported_w7_mode"):
        resolve_w7_mode("unsafe")


def test_pipeline_keeps_legacy_authoritative_for_shadow_mode(tmp_path: Path):
    pipeline = WriteSidePipeline(
        JsonKGStore(tmp_path / "kg"),
        w7_mode="shadow_multi_agent",
    )
    assert pipeline.w7_shadow_enabled is True
    assert pipeline.w7_legacy_authoritative is True
    with pytest.raises(ValueError, match="w7_mode_not_yet_promotable"):
        WriteSidePipeline(
            JsonKGStore(tmp_path / "kg2"),
            w7_mode="multi_agent",
        )


def test_pipeline_runs_shadow_batches_without_touching_legacy_state(
    tmp_path: Path,
):
    pipeline = WriteSidePipeline(
        JsonKGStore(tmp_path / "kg"),
        w7_mode="shadow_multi_agent",
        w7_decision_client=FakeDecisionClient(),
        review_context_enabled=False,
        w7_shadow_out_dir=tmp_path / "shadow",
    )

    def fake_atomic(manifest, **_kwargs):
        return {
            "schema_version": "w7.atomic_w2_result.v1",
            "manifest_hash": manifest.get("manifest_hash") or "",
            "parent_episode_id": manifest.get("parent_episode_id") or "",
            "atomic_episode_ids": [],
            "candidates": [],
            "w7b_case_cards": [],
            "summary": {"atomic_cases": 0, "candidates": 0, "schema_valid": 0},
            "queue_written": False,
            "kg_mutated": False,
        }

    pipeline.extract_w7_atomic_cases = fake_atomic  # type: ignore[method-assign]
    episodes = [{
        "episode_id": "ep-shadow-a",
        "thread_id": "thread-shadow",
        "fault_description_messages": [{
            "message_id": "m-shadow-a",
            "text": "设备蓝屏",
        }],
    }, {
        "episode_id": "ep-shadow-b",
        "thread_id": "thread-shadow",
        "resolution_messages": [{
            "message_id": "m-shadow-b",
            "text": "恢复正常生产",
        }],
    }]
    before = pipeline.store.tree_hash() if hasattr(pipeline.store, "tree_hash") else None
    manifest = pipeline._run_w7_shadow_batches(
        episodes,
        source_type="chat",
        out_dir=tmp_path,
    )
    assert manifest["status"] == "completed"
    assert manifest["legacy_authoritative"] is True
    assert manifest["promotion_allowed"] is False
    assert manifest["batch_count"] == 1
    assert (tmp_path / "w7_shadow" / "manifest.json").is_file()
    after = pipeline.store.tree_hash() if hasattr(pipeline.store, "tree_hash") else None
    assert before == after


def test_case_boundary_contract_requires_complete_message_accounting():
    normalized, issues = validate_case_boundary_decision(
        {
            "case_fragments": [{
                "fragment_ref": "F1",
                "case_kind": "diagnostic_case",
                "fault_summary": "设备蓝屏",
                "source_message_ids": ["m1"],
                "evidence_spans": [
                    {"message_id": "m1", "start": 0, "end": 4}
                ],
                "uncertainties": [],
            }],
            "non_case_message_ids": [],
            "uncertainties": [],
        },
        allowed_message_ids={"m1", "m2"},
    )
    assert normalized["case_fragments"][0]["fragment_ref"] == "F1"
    assert "message_unaccounted:m2" in issues


def test_case_boundary_drops_empty_attachment_span_and_clamps_text_span():
    normalized, issues = validate_case_boundary_decision(
        {
            "case_fragments": [{
                "fragment_ref": "F1",
                "case_kind": "diagnostic_case",
                "fault_summary": "设备蓝屏",
                "source_message_ids": ["attachment", "text"],
                "evidence_spans": [{
                    "message_id": "attachment",
                    "start": 0,
                    "end": 0,
                }, {
                    "message_id": "text",
                    "start": 0,
                    "end": 20,
                }],
                "uncertainties": [],
            }],
            "non_case_message_ids": [],
            "uncertainties": [],
        },
        allowed_message_ids={"attachment", "text"},
        message_text_lengths={"attachment": 0, "text": 4},
    )
    assert normalized["case_fragments"][0]["evidence_spans"] == [{
        "message_id": "text",
        "start": 0,
        "end": 4,
    }]
    assert issues == []


def test_outcome_patch_rejects_verified_without_evidence():
    _, issues = validate_outcome_patch(
        {
            "operations": [{
                "op": "revise_trace_status",
                "local_trace_ref": "T1",
                "from": "pending",
                "to": "verified",
                "evidence_message_ids": [],
                "reason": "",
            }],
            "uncertainties": [],
        },
        allowed_trace_refs={"T1"},
        allowed_message_ids={"m1"},
    )
    assert "operations[0]:verified_without_evidence" in issues


def test_sparse_candidate_graph_keeps_identity_edges_and_bounds_semantic_edges():
    items = [
        {
            "case_ref": "C1",
            "title": "SI2030TWR250078设备蓝屏",
            "device_scope": "SI2030TWR250078",
            "jira_keys": ["SMTAOITS-1234"],
            "start_time": "2026-05-20 12:55",
        },
        {
            "case_ref": "C2",
            "title": "更换内存后设备再次蓝屏",
            "device_scope": "SI2030TWR250078",
            "jira_keys": ["SMTAOITS-1234"],
            "start_time": "2026-05-21 12:55",
        },
        *[
            {
                "case_ref": f"N{index}",
                "title": f"设备{index}相机拍摄异常",
                "device_scope": f"AOI-{index}",
                "start_time": "2026-05-21 13:00",
            }
            for index in range(3, 12)
        ],
    ]
    graph = build_sparse_candidate_graph(items, top_k=2)
    identity = [
        edge
        for edge in graph["edges"]
        if edge["edge_class"] == "identity_edge"
    ]
    assert [(edge["left_case_ref"], edge["right_case_ref"]) for edge in identity] == [
        ("C1", "C2")
    ]
    assert graph["stats"]["edges"] < len(items) * (len(items) - 1) // 2
    assert all(
        edge["requires_adjudication"]
        for edge in graph["edges"]
    )


def test_field_work_report_can_participate_in_trace_without_being_root():
    assert "field_work_report" in TRACE_ASSEMBLY_CASE_KINDS


def test_shared_message_is_provenance_hint_not_strong_trace_identity():
    graph = build_sparse_candidate_graph([{
        "case_ref": "C1",
        "parent_episode_id": "E1",
        "source_message_ids": ["m1"],
        "fault_summary": "相机网卡断连导致拍摄失败",
    }, {
        "case_ref": "C2",
        "parent_episode_id": "E1",
        "source_message_ids": ["m1"],
        "fault_summary": "电感未检出已提交算法数据",
    }])
    edge = graph["edges"][0]
    assert edge["edge_class"] == "weak_retrieval_edge"
    assert edge["score"] < graph["config"]["strong_threshold"]
    assert graph["config"]["shared_message_weight"] == 1.5
    assert graph["config"]["shared_parent_episode_weight"] == 0.5


def test_device_identity_shift_blocks_automatic_semantic_merge():
    graph = build_sparse_candidate_graph([{
        "case_ref": "C1",
        "fault_summary": "原设备相机网卡断连导致拍摄失败",
        "start_time": "2026-07-30 10:00",
    }, {
        "case_ref": "C2",
        "fault_summary": "另外一台设备开始拍摄失败",
        "start_time": "2026-07-30 11:00",
    }])
    edge = graph["edges"][0]
    assert edge["auto_merge_blockers"] == [
        "device_identity_shift_without_shared_device"
    ]
    guarded = apply_candidate_edge_safety_guards(
        {
            "edge_decisions": [{
                "left_case_ref": "C1",
                "right_case_ref": "C2",
                "decision": "must_link",
                "relation_hint": "recurrence_of",
                "evidence_message_ids": [],
                "reasons": ["同一故障类型"],
            }],
            "uncertainties": [],
        },
        graph["edges"],
    )
    decision = guarded["edge_decisions"][0]
    assert decision["decision"] == "possible_link"
    assert (
        decision["local_override_reason"]
        == "candidate_identity_discontinuity_guard"
    )
    components, _ = build_trace_components(
        graph,
        guarded,
        core_case_refs={"C1", "C2"},
    )
    bridge = build_component_bridge_candidates(components, guarded)
    assert bridge["candidates"] == []


def test_shared_explicit_device_does_not_trigger_identity_shift_guard():
    graph = build_sparse_candidate_graph([{
        "case_ref": "C1",
        "fault_summary": "新交付的一台2020T设备出现拍摄失败",
        "start_time": "2026-07-30 10:00",
    }, {
        "case_ref": "C2",
        "fault_summary": "2020T设备确认置换",
        "start_time": "2026-07-30 11:00",
    }])
    assert graph["edges"][0]["auto_merge_blockers"] == []


def test_w1_scope_is_propagated_to_cards_and_candidate_graph():
    cards = _enrich_unit_card_scope(
        [{"case_ref": "C1", "fault_summary": "相机拍摄失败"}],
        {
            "extracted": {
                "sites": ["客户03"],
                "devices": ["AOI-01"],
                "jira_ids": ["SMTAOITS-1"],
            },
            "field_report_anchor": {
                "anchor_id": "field-report:m1",
                "report_date": "2026-07-30",
                "site": "客户03",
                "anchor_item_index": 1,
            },
        },
    )
    assert cards[0]["site_scope"] == "客户03"
    assert cards[0]["device_scope"] == "AOI-01"
    assert cards[0]["jira_keys"] == ["SMTAOITS-1"]
    assert cards[0]["field_report_scope"]["anchor_item_index"] == 1
    graph = build_sparse_candidate_graph([
        cards[0],
        {
            "case_ref": "C2",
            "fault_summary": "更换网卡后拍摄恢复",
            "site_scope": "客户03",
        },
    ])
    assert "shared_site:客户03" in graph["edges"][0]["reasons"]
    assert graph["config"]["shared_site_weight"] == 1.0


def test_chat_name_site_is_only_a_channel_hint():
    cards = _enrich_unit_card_scope(
        [{
            "case_ref": "C1",
            "fault_summary": "客户17AB面程序与MES文本不一致",
        }],
        {
            "source_offsets": [{
                "field": "sites",
                "value": "客户03",
                "source": "message.raw.chat_name",
            }],
            "extracted": {"sites": ["客户03"]},
            "field_report_anchor": {
                "anchor_id": "field-report:m1",
                "report_date": "2026-07-30",
                "site": "客户03",
                "anchor_item_index": 2,
            },
        },
    )
    assert cards[0]["site_scope"] == ""
    assert cards[0]["site_scopes"] == []
    assert cards[0]["channel_site_scopes"] == ["客户03"]
    assert cards[0]["site_scope_provenance"] == "channel_hint_only"
    assert cards[0]["field_report_scope"]["site"] == ""
    assert (
        cards[0]["field_report_scope"]["channel_site_hint"]
        == "客户03"
    )


def test_score_string_dedupe_preserves_trace_refs_and_status_words():
    assert dedupe_strings_for_score(
        "trace:case_refs-:-case:source-123"
    ) == ["trace:case_refs-:-case:source-123"]
    assert dedupe_strings_for_score("investigating") == ["investigating"]
    assert dedupe_strings_for_score("A, B；C") == ["A", "B", "C"]


def test_neighbor_link_chunks_bounded_candidate_edges(tmp_path: Path):
    client = FakeDecisionClient()
    cards = [{
        "case_ref": f"C{index}",
        "case_kind": "diagnostic_case",
        "title": f"相机拍摄失败阶段{index}",
    } for index in range(1, 5)]
    edges = [{
        "left_case_ref": "C1",
        "right_case_ref": "C2",
        "score": 2.0,
        "reasons": ["semantic"],
        "edge_class": "weak_retrieval_edge",
        "requires_adjudication": True,
    }, {
        "left_case_ref": "C2",
        "right_case_ref": "C3",
        "score": 2.0,
        "reasons": ["semantic"],
        "edge_class": "weak_retrieval_edge",
        "requires_adjudication": True,
    }, {
        "left_case_ref": "C3",
        "right_case_ref": "C4",
        "score": 2.0,
        "reasons": ["semantic"],
        "edge_class": "weak_retrieval_edge",
        "requires_adjudication": True,
    }]
    graph = {
        "schema_version": "w7.sparse_candidate_graph.v1",
        "node_refs": ["C1", "C2", "C3", "C4"],
        "edges": edges,
        "graph_hash": "graph",
    }
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
        neighbor_chunk_edges=2,
        component_workers=2,
    ).run_neighbor_link(
        graph=graph,
        case_cards=cards,
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert stage["chunk_count"] == 2
    assert len(stage["calls"]) == 2
    assert len(stage["decision"]["edge_decisions"]) == 3


@pytest.mark.parametrize(
    ("client", "expected_repair", "expected_attempts"),
    [
        (
            DuplicateNeighborDecisionClient(),
            "collapsed_duplicate_edges",
            1,
        ),
    ],
)
def test_neighbor_link_repairs_duplicate_without_nested_transport_retry(
    tmp_path: Path,
    client,
    expected_repair: str | None,
    expected_attempts: int,
):
    graph = {
        "schema_version": "w7.sparse_candidate_graph.v1",
        "node_refs": ["C1", "C2"],
        "edges": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "score": 2.0,
            "reasons": ["semantic"],
            "edge_class": "weak_retrieval_edge",
            "requires_adjudication": True,
        }],
        "graph_hash": "graph",
    }
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run_neighbor_link(
        graph=graph,
        case_cards=[{"case_ref": "C1"}, {"case_ref": "C2"}],
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert len(stage["decision"]["edge_decisions"]) == 1
    assert stage["calls"][0]["transport_attempts"] == expected_attempts
    repairs = stage["calls"][0]["local_structural_repairs"]
    if expected_repair:
        assert expected_repair in repairs
    else:
        assert repairs == []


def test_neighbor_link_transport_failure_is_not_retried_by_agent_layer(
    tmp_path: Path,
):
    client = FlakyNeighborDecisionClient()
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run_neighbor_link(
        graph={
            "schema_version": "w7.sparse_candidate_graph.v4",
            "node_refs": ["C1", "C2"],
            "edges": [{
                "left_case_ref": "C1",
                "right_case_ref": "C2",
                "score": 2.0,
                "reasons": ["semantic"],
                "edge_class": "weak_retrieval_edge",
                "requires_adjudication": True,
            }],
            "graph_hash": "graph",
        },
        case_cards=[
            {"case_ref": "C1", "title": "故障"},
            {"case_ref": "C2", "title": "后续"},
        ],
        allowed_message_ids=set(),
    )
    assert client.neighbor_attempts == 1
    assert stage["schema_valid"] is False
    assert stage["status"] == "failed_closed"
    assert "temporary transport failure" in stage["issues"][0]


def test_neighbor_link_semantically_retries_contradictory_must_reason(
    tmp_path: Path,
):
    client = ContradictoryNeighborDecisionClient()
    graph = {
        "schema_version": "w7.sparse_candidate_graph.v2",
        "node_refs": ["C1", "C2"],
        "edges": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "score": 2.0,
            "reasons": ["shared_message:m1"],
            "edge_class": "weak_retrieval_edge",
            "requires_adjudication": True,
        }],
        "graph_hash": "graph",
    }
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run_neighbor_link(
        graph=graph,
        case_cards=[{"case_ref": "C1"}, {"case_ref": "C2"}],
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert client.calls == 2
    assert stage["decision"]["edge_decisions"][0][
        "decision"
    ] == "cannot_link"
    assert stage["calls"][0]["semantic_contract_attempts"] == 2
    assert stage["calls"][0]["semantic_repair_count"] == 1


def test_neighbor_link_fail_closed_downgrades_persistent_contradictory_must(
    tmp_path: Path,
):
    client = ContradictoryNeighborDecisionClient(
        repair_on_retry=False
    )
    graph = {
        "schema_version": "w7.sparse_candidate_graph.v2",
        "node_refs": ["C1", "C2"],
        "edges": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "score": 2.0,
            "reasons": ["shared_message:m1"],
            "edge_class": "weak_retrieval_edge",
            "requires_adjudication": True,
        }],
        "graph_hash": "graph",
    }
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run_neighbor_link(
        graph=graph,
        case_cards=[{"case_ref": "C1"}, {"case_ref": "C2"}],
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert client.calls == 3
    assert stage["decision"]["edge_decisions"][0][
        "decision"
    ] == "cannot_link"
    assert any(
        str(value).startswith("downgraded_contradictory_must:")
        for value in stage["calls"][0]["local_structural_repairs"]
    )


def test_component_builder_downgrades_overflowing_must_link_edge():
    graph = {
        "node_refs": ["C1", "C2", "C3"],
        "edges": [],
    }
    decision = {
        "edge_decisions": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "decision": "must_link",
            "relation_hint": "continuation_of",
            "evidence_message_ids": [],
            "reasons": ["same fault"],
        }, {
            "left_case_ref": "C2",
            "right_case_ref": "C3",
            "decision": "must_link",
            "relation_hint": "continuation_of",
            "evidence_message_ids": [],
            "reasons": ["same fault"],
        }],
    }
    components, issues = build_trace_components(
        graph,
        decision,
        max_component_size=2,
    )
    assert issues == []
    assert [item["case_refs"] for item in components["components"]] == [
        ["C1", "C2"],
        ["C3"],
    ]
    assert components["overflow_edges"][0][
        "local_downgrade_reason"
    ] == "component_size_limit"


def test_component_builder_downgrades_must_link_that_conflicts_with_cannot():
    graph = {
        "node_refs": ["C1", "C2", "C3"],
        "edges": [],
    }
    decision = {
        "edge_decisions": [{
            "left_case_ref": "C2",
            "right_case_ref": "C3",
            "decision": "must_link",
            "relation_hint": "continuation_of",
            "evidence_message_ids": [],
            "reasons": ["same fault"],
        }, {
            "left_case_ref": "C1",
            "right_case_ref": "C3",
            "decision": "cannot_link",
            "relation_hint": "independent",
            "evidence_message_ids": [],
            "reasons": ["different faults"],
        }, {
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "decision": "must_link",
            "relation_hint": "continuation_of",
            "evidence_message_ids": [],
            "reasons": ["same fault"],
        }],
    }
    components, issues = build_trace_components(graph, decision)
    assert issues == []
    assert [item["case_refs"] for item in components["components"]] == [
        ["C1", "C2"],
        ["C3"],
    ]
    assert components["conflict_edges"] == [{
        **decision["edge_decisions"][0],
        "decision": "possible_link",
        "local_downgrade_reason": "cannot_link_conflict",
        "blocking_cannot_pairs": [["C1", "C3"]],
    }]
    assert components["downgraded_edges"] == components["conflict_edges"]


def test_component_builder_is_deterministic_across_edge_order():
    graph = {
        "node_refs": ["C1", "C2", "C3"],
        "edges": [],
    }
    edges = [{
        "left_case_ref": "C2",
        "right_case_ref": "C3",
        "decision": "must_link",
    }, {
        "left_case_ref": "C1",
        "right_case_ref": "C3",
        "decision": "cannot_link",
    }, {
        "left_case_ref": "C1",
        "right_case_ref": "C2",
        "decision": "must_link",
    }]
    forward, forward_issues = build_trace_components(
        graph,
        {"edge_decisions": edges},
    )
    reverse, reverse_issues = build_trace_components(
        graph,
        {"edge_decisions": list(reversed(edges))},
    )
    assert forward_issues == reverse_issues == []
    assert forward["components"] == reverse["components"]
    assert forward["downgraded_edges"] == reverse["downgraded_edges"]
    assert forward["components_hash"] == reverse["components_hash"]


def test_component_consistency_finds_transitive_conflict_and_applies_override():
    decision = {
        "edge_decisions": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "decision": "must_link",
        }, {
            "left_case_ref": "C2",
            "right_case_ref": "C3",
            "decision": "must_link",
        }, {
            "left_case_ref": "C1",
            "right_case_ref": "C3",
            "decision": "cannot_link",
        }],
        "uncertainties": [],
    }
    conflicts = build_component_conflicts(decision)
    assert len(conflicts["conflicts"]) == 1
    assert [
        (
            edge["left_case_ref"],
            edge["right_case_ref"],
        )
        for edge in conflicts["conflicts"][0]["must_link_path"]
    ] == [("C1", "C2"), ("C2", "C3")]
    revised = apply_component_consistency_decision(
        decision,
        {
            "conflict_decisions": [{
                "left_case_ref": "C1",
                "right_case_ref": "C3",
                "decision": "weak_cannot",
            }],
            "uncertainties": [],
        },
    )
    assert revised["edge_decisions"][2]["decision"] == "possible_link"
    assert revised["edge_decisions"][2][
        "original_decision"
    ] == "cannot_link"
    components, issues = build_trace_components(
        {"node_refs": ["C1", "C2", "C3"]},
        revised,
    )
    assert issues == []
    assert components["components"][0]["case_refs"] == [
        "C1", "C2", "C3",
    ]


def test_component_consistency_model_failure_keeps_safe_split(tmp_path: Path):
    decision = {
        "edge_decisions": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "decision": "must_link",
        }, {
            "left_case_ref": "C2",
            "right_case_ref": "C3",
            "decision": "must_link",
        }, {
            "left_case_ref": "C1",
            "right_case_ref": "C3",
            "decision": "cannot_link",
        }],
        "uncertainties": [],
    }
    stage = W7ShadowOrchestrator(
        client=FailingDecisionClient(),
        checkpoint_root=tmp_path / "checkpoints",
    ).run_component_consistency(
        link_decision=decision,
        case_cards=[
            {"case_ref": "C1"},
            {"case_ref": "C2"},
            {"case_ref": "C3"},
        ],
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert stage["status"] == "degraded_safe"
    assert stage["decision"]["conflict_decisions"][0][
        "decision"
    ] == "confirmed_cannot"
    assert stage["revised_link_decision"]["edge_decisions"] == (
        decision["edge_decisions"]
    )


def test_component_bridge_candidates_exclude_cannot_cross_component():
    components = {
        "components": [{
            "component_ref": "A",
            "case_refs": ["C1", "C2"],
        }, {
            "component_ref": "B",
            "case_refs": ["C3"],
        }, {
            "component_ref": "C",
            "case_refs": ["C4"],
        }],
    }
    decision = {
        "edge_decisions": [{
            "left_case_ref": "C2",
            "right_case_ref": "C3",
            "decision": "possible_link",
        }, {
            "left_case_ref": "C1",
            "right_case_ref": "C3",
            "decision": "cannot_link",
        }, {
            "left_case_ref": "C2",
            "right_case_ref": "C4",
            "decision": "possible_link",
        }],
    }
    candidates = build_component_bridge_candidates(
        components,
        decision,
    )
    assert [
        (item["left_case_ref"], item["right_case_ref"])
        for item in candidates["candidates"]
    ] == [("C2", "C4")]
    revised = apply_component_bridge_decision(
        decision,
        {
            "bridge_decisions": [{
                "left_case_ref": "C2",
                "right_case_ref": "C4",
                "decision": "promote_must",
                "evidence_message_ids": [],
                "reasons": ["same Jira"],
            }],
            "uncertainties": [],
        },
    )
    assert revised["edge_decisions"][2]["decision"] == "must_link"
    assert revised["edge_decisions"][2][
        "original_decision"
    ] == "possible_link"


def test_component_bridge_model_failure_keeps_possible(tmp_path: Path):
    decision = {
        "edge_decisions": [{
            "left_case_ref": "C1",
            "right_case_ref": "C2",
            "decision": "possible_link",
        }],
        "uncertainties": [],
    }
    stage = W7ShadowOrchestrator(
        client=FailingDecisionClient(),
        checkpoint_root=tmp_path / "checkpoints",
    ).run_component_bridges(
        components={
            "components": [{
                "component_ref": "A",
                "case_refs": ["C1"],
            }, {
                "component_ref": "B",
                "case_refs": ["C2"],
            }],
        },
        link_decision=decision,
        case_cards=[{"case_ref": "C1"}, {"case_ref": "C2"}],
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert stage["status"] == "degraded_safe"
    assert stage["decision"]["bridge_decisions"][0][
        "decision"
    ] == "keep_possible"
    assert stage["revised_link_decision"]["edge_decisions"] == (
        decision["edge_decisions"]
    )


def test_shadow_orchestrator_caches_valid_boundary_stage(tmp_path: Path):
    client = FakeDecisionClient()
    ledger = build_episode_source_ledger(_episode())
    orchestrator = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    )
    first = orchestrator.run(ledger=ledger)
    second = orchestrator.run(ledger=ledger)
    assert first["schema_valid"] is True
    assert second["schema_valid"] is True
    assert client.calls == [
        "decide_atomic_case_boundaries",
        "assign_trace_groups_and_phases",
        "assign_trace_groups_and_phases",
        "reconcile_trace_outcomes",
    ]
    assert (
        second["case_boundary"]["calls"][0]["stage_cache_hit"] is True
    )
    assert len(first["case_boundary"]["decision"]["case_fragments"]) == 2
    assert first["atomic_case_adapter"]["schema_valid"] is True
    assert first["trace_phase"]["schema_valid"] is True
    assert first["trace_compiler"]["schema_valid"] is True
    assert first["w6_trace_review_payload"]["trace_bundle_hash"]


def test_checkpoint_store_keeps_multiple_content_keys_per_stage(
    tmp_path: Path,
):
    store = CheckpointStore(tmp_path / "checkpoints")
    first_key = store.key(
        stage="case_boundary_001",
        input_value={"episode": "one"},
        version="v1",
    )
    second_key = store.key(
        stage="case_boundary_001",
        input_value={"episode": "two"},
        version="v1",
    )
    store.write(
        stage="case_boundary_001",
        key=first_key,
        output={"episode": "one"},
        issues=[],
        call={"model": "fake"},
    )
    store.write(
        stage="case_boundary_001",
        key=second_key,
        output={"episode": "two"},
        issues=[],
        call={"model": "fake"},
    )
    assert store.read(
        stage="case_boundary_001", key=first_key
    )["output"]["episode"] == "one"
    assert store.read(
        stage="case_boundary_001", key=second_key
    )["output"]["episode"] == "two"
    assert len(list(
        (tmp_path / "checkpoints" / "case_boundary_001").glob("*.json")
    )) == 2


def test_evidence_anchor_requires_complete_exclusive_accounting():
    normalized, issues = validate_evidence_anchor_decision(
        {
            "anchor_decisions": [],
            "unassigned_evidence_message_ids": [],
            "uncertainties": [],
        },
        allowed_fragment_refs={"F1"},
        candidate_message_ids={"m0"},
        allowed_attachment_ids_by_message={"m0": {"file-1"}},
    )
    assert normalized["anchor_decisions"] == []
    assert "evidence_message_unaccounted:m0" in issues


def test_case_source_context_binds_verbatim_rows_without_mutating_card():
    ledger = build_episode_source_ledger({
        "episode_id": "ep-1",
        "thread_id": "thread-1",
        "messages": [{
            "message_id": "m1",
            "create_time": "2026-07-27 10:00",
            "sender": {"name": "FAE"},
            "msg_type": "text",
            "text": "设备蓝屏后更换内存条，恢复正常生产",
        }],
    })
    cards = [{
        "case_ref": "C1",
        "source_message_ids": ["m1"],
    }]
    enriched = attach_case_source_context(cards, ledger)
    assert "source_context_rows" not in cards[0]
    assert enriched[0]["source_context_rows"] == [{
        "message_id": "m1",
        "create_time": "2026-07-27 10:00",
        "sender": "FAE",
        "message_type": "text",
        "text": "设备蓝屏后更换内存条，恢复正常生产",
        "attachments": [],
    }]


def test_evidence_anchor_finds_attachment_before_text_and_atomic_adapter():
    episode = _attachment_episode()
    ledger = build_episode_source_ledger(episode)
    candidates = evidence_anchor_candidates(ledger)
    assert candidates[0]["message_id"] == "m0"
    assert candidates[0]["attachment_refs"][0]["attachment_id"] == (
        "file-diagnostic-1"
    )
    boundary = {
        "case_fragments": [{
            "fragment_ref": "F1",
            "case_kind": "diagnostic_case",
            "fault_summary": "设备蓝屏",
            "source_message_ids": ["m1"],
            "evidence_spans": [],
            "uncertainties": [],
        }],
        "non_case_message_ids": [],
        "uncertainties": [],
    }
    anchor = {
        "anchor_decisions": [{
            "evidence_message_id": "m0",
            "attachment_ids": ["file-diagnostic-1"],
            "target_fragment_ref": "F1",
            "role": "initial_diagnostic_package",
            "confidence": "high",
            "reasons": ["后续消息明确指向前面的诊断数据"],
        }],
        "unassigned_evidence_message_ids": [],
        "uncertainties": [],
    }
    manifest, issues = build_atomic_case_manifest(
        episode=episode,
        source_ledger=ledger,
        case_boundary=boundary,
        evidence_anchor=anchor,
    )
    atomic = w2_atomic_episodes(manifest)
    assert issues == []
    assert len(atomic) == 1
    assert atomic[0]["evidence_message_ids"] == ["m1", "m0"]
    assert atomic[0]["extracted"]["w7_atomic_case"][
        "anchored_evidence_message_ids"
    ] == ["m0"]


def test_full_shadow_runs_evidence_anchor_stage(tmp_path: Path):
    result = W7ShadowOrchestrator(
        client=FakeDecisionClient(),
        checkpoint_root=tmp_path / "checkpoints",
    ).run(ledger=build_episode_source_ledger(_attachment_episode()))
    assert result["schema_valid"] is True
    anchor = result["evidence_anchor"]["decision"]["anchor_decisions"][0]
    assert anchor["evidence_message_id"] == "m0"
    assert anchor["role"] == "initial_diagnostic_package"
    assert result["atomic_case_adapter"]["manifest"]["atomic_cases"]


def test_neighbor_link_and_component_builder_only_materialize_must_link():
    graph = build_sparse_candidate_graph([
        {
            "case_ref": "C1",
            "jira_keys": ["SMTAOITS-1"],
            "title": "设备蓝屏",
        },
        {
            "case_ref": "C2",
            "jira_keys": ["SMTAOITS-1"],
            "title": "蓝屏再次复发",
        },
        {"case_ref": "C3", "title": "相机曝光异常"},
    ])
    required = {
        tuple(sorted((
            edge["left_case_ref"], edge["right_case_ref"]
        )))
        for edge in graph["edges"] if edge["requires_adjudication"]
    }
    decision, issues = validate_trace_link_decision(
        {
            "edge_decisions": [{
                "left_case_ref": "C1",
                "right_case_ref": "C2",
                "decision": "must_link",
                "relation_hint": "recurrence_of",
                "evidence_message_ids": [],
                "reasons": ["同一 Jira"],
            }],
            "uncertainties": [],
        },
        required_edges=required,
        allowed_edges={
            tuple(sorted((
                edge["left_case_ref"], edge["right_case_ref"]
            ))) for edge in graph["edges"]
        },
        allowed_message_ids=set(),
    )
    components, component_issues = build_trace_components(graph, decision)
    assert issues == []
    assert component_issues == []
    assert [item["case_refs"] for item in components["components"]] == [
        ["C1", "C2"],
        ["C3"],
    ]


def test_full_shadow_links_diagnostic_root_to_later_validation(
    tmp_path: Path,
):
    result = W7ShadowOrchestrator(
        client=FakeDecisionClient(),
        checkpoint_root=tmp_path / "checkpoints",
    ).run(
        ledger=build_episode_source_ledger(_episode()),
        case_cards=[{
            "case_ref": "C1",
            "case_kind": "diagnostic_case",
            "title": "设备蓝屏",
            "jira_keys": ["SMTAOITS-1"],
            "source_message_ids": ["m1"],
        }, {
            "case_ref": "C2",
            "case_kind": "positive_validation",
            "title": "恢复正常生产",
            "jira_keys": ["SMTAOITS-1"],
            "source_message_ids": ["m2"],
        }],
    )
    assert result["schema_valid"] is True
    edge = result["neighbor_link"]["decision"]["edge_decisions"][0]
    assert edge["decision"] == "must_link"
    assert result["trace_components"]["graph"]["components"][0][
        "case_refs"
    ] == ["C1", "C2"]
    trace = result["trace_compiler"]["bundle"]["traces"][0]
    assert trace["case_refs"] == ["C1", "C2"]
    assert trace["resolution_status"] == "verified"


def test_trace_phase_contract_rejects_missing_case_and_cycles():
    _, issues = validate_trace_phase_patch(
        {
            "operations": [{
                "op": "create_trace_group",
                "local_trace_ref": "T1",
                "case_refs": ["C1"],
                "case_ref": "",
                "event_type": "",
                "relation_type": "",
                "phase_index": 0,
                "after_case_ref": "",
                "evidence_message_ids": [],
                "summary": "",
            }, {
                "op": "set_phase",
                "local_trace_ref": "T1",
                "case_refs": [],
                "case_ref": "C1",
                "event_type": "report",
                "relation_type": "trace_root",
                "phase_index": 1,
                "after_case_ref": "C1",
                "evidence_message_ids": [],
                "summary": "",
            }],
            "uncertainties": [],
        },
        component_case_refs={"C1", "C2"},
        allowed_message_ids=set(),
    )
    assert "case_not_assigned_to_trace:C2" in issues
    assert "case_missing_phase:C2" in issues
    assert "phase_cycle:T1:C1" in issues


def test_trace_phase_harness_retries_contract_invalid_patch(
    tmp_path: Path,
):
    client = RepairingTracePhaseClient()
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run_trace_phases(
        components={
            "components": [{
                "component_ref": "component-1",
                "case_refs": ["C1"],
            }],
        },
        case_cards=[{
            "case_ref": "C1",
            "case_kind": "diagnostic_case",
            "title": "设备蓝屏",
        }],
        link_decision={"edge_decisions": []},
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert client.calls == 2
    assert stage["calls"][0]["semantic_contract_attempts"] == 2
    assert stage["calls"][0]["semantic_repair_count"] == 1


def test_trace_phase_harness_canonically_collapses_duplicate_atomic_phases(
    tmp_path: Path,
):
    client = DuplicateTracePhaseClient()
    stage = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run_trace_phases(
        components={
            "components": [{
                "component_ref": "component-1",
                "case_refs": ["C1"],
            }],
        },
        case_cards=[{
            "case_ref": "C1",
            "case_kind": "diagnostic_case",
            "title": "设备蓝屏后恢复",
        }],
        link_decision={"edge_decisions": []},
        allowed_message_ids=set(),
    )
    assert stage["schema_valid"] is True
    assert client.calls == 2
    assert [
        operation["event_type"]
        for operation in stage["decision"]["operations"]
        if operation["op"] == "set_phase"
    ] == ["report"]
    assert stage["calls"][0]["local_structural_repairs"] == [
        "collapsed_duplicate_phase:C1"
    ]


def test_trace_phase_model_failure_preserves_cases_as_safe_standalone(
    tmp_path: Path,
):
    cards = [{
        "case_ref": "C1",
        "case_kind": "diagnostic_case",
        "title": "翘脚高度测量偏差",
        "source_message_ids": ["m1"],
        "evidence_message_ids": ["m1"],
    }]
    stage = W7ShadowOrchestrator(
        client=FailingDecisionClient(),
        checkpoint_root=tmp_path / "checkpoints",
    ).run_trace_phases(
        components={
            "components": [{
                "component_ref": "component-1",
                "case_refs": ["C1"],
            }],
        },
        case_cards=cards,
        link_decision={"edge_decisions": []},
        allowed_message_ids={"m1"},
    )
    assert stage["schema_valid"] is False
    assert stage["decision"]["standalone_case_refs"] == ["C1"]
    assert "safe_standalone_after_phase_failure" in (
        stage["decision"]["uncertainties"][0]
    )
    compiled = TraceCompiler().compile_review_bundle(
        case_cards=cards,
        phase_patch=stage["decision"],
        outcome_patch={"operations": [], "uncertainties": []},
    )
    assert compiled["unassigned_case_refs"] == []
    assert compiled["standalone_case_refs"] == ["C1"]


def test_shadow_orchestrator_persists_auditable_fail_closed_result(
    tmp_path: Path,
):
    ledger = build_episode_source_ledger(_episode())
    result = W7ShadowOrchestrator(
        client=FailingDecisionClient(),
        checkpoint_root=tmp_path / "checkpoints",
    ).run(ledger=ledger)
    boundary = result["case_boundary"]
    assert result["schema_valid"] is False
    assert result["promotion_allowed"] is False
    assert result["fallback_policy"] == "keep_legacy_w7"
    assert boundary["status"] == "failed_closed"
    assert boundary["calls"][0]["model_call_failed"] is True
    assert "RuntimeError" in boundary["issues"][0]


def test_deepseek_client_uses_fresh_json_output_after_malformed_tool_call(
    monkeypatch,
):
    import debug_agent_system.agents.write.w7_trace.model_client as module

    def broken_tool_call(**kwargs):
        raise DeepSeekToolCallError(
            "deepseek_tool_arguments_json_decode:Expecting value"
        )

    def json_repair(**kwargs):
        assert "只输出原工具 arguments 对象本身" in kwargs["system_prompt"]
        return {
            "arguments": {
                "case_fragments": [],
                "non_case_message_ids": ["m1"],
                "uncertainties": [],
            },
            "model": "deepseek-v4-pro",
            "finish_reason": "stop",
        }

    monkeypatch.setattr(module, "call_strict_tool", broken_tool_call)
    monkeypatch.setattr(module, "call_json_object", json_repair)
    result = DeepSeekDecisionModelClient(api_key="test-key").call_tool(
        stage="case_boundary",
        system_prompt="decide",
        payload={"allowed_message_ids": ["m1"]},
        tool={
            "function": {
                "name": "decide",
                "parameters": {"type": "object"},
            }
        },
        max_tokens=1024,
    )
    assert result["json_output_fallback"] is True
    assert result["semantic_repair_count"] == 1
    assert "json_decode" in result["strict_tool_error"]
    assert result["arguments"]["non_case_message_ids"] == ["m1"]


def test_deepseek_client_does_not_treat_transport_failure_as_semantic_repair(
    monkeypatch,
):
    import debug_agent_system.agents.write.w7_trace.model_client as module

    def transport_failure(**kwargs):
        raise DeepSeekToolCallError("deepseek_http_401:unauthorized")

    def unexpected_json_repair(**kwargs):
        raise AssertionError("JSON repair must not run for transport failures")

    monkeypatch.setattr(module, "call_strict_tool", transport_failure)
    monkeypatch.setattr(module, "call_json_object", unexpected_json_repair)
    with pytest.raises(DeepSeekToolCallError, match="deepseek_http_401"):
        DeepSeekDecisionModelClient(api_key="bad-key").call_tool(
            stage="case_boundary",
            system_prompt="decide",
            payload={},
            tool={
                "function": {
                    "name": "decide",
                    "parameters": {"type": "object"},
                }
            },
            max_tokens=1024,
        )


def test_shadow_orchestrator_reconciles_explicit_production_recovery(tmp_path: Path):
    client = FakeDecisionClient()
    ledger = build_episode_source_ledger(_episode())
    result = W7ShadowOrchestrator(
        client=client,
        checkpoint_root=tmp_path / "checkpoints",
    ).run(
        ledger=ledger,
        trace_cards=[{
            "local_trace_ref": "T1",
            "resolution_status": "provisionally_resolved",
        }],
    )
    operation = result["outcome_reconciliation"]["decision"]["operations"][0]
    assert result["schema_valid"] is True
    assert operation["to"] == "verified"
    assert operation["evidence_message_ids"] == ["m2"]


def test_w6_trace_corrections_invalidate_and_rebind_approval_hash(
    tmp_path: Path,
):
    store = JsonKGV2Store(tmp_path / "kg-v2")
    queue = ReviewQueueAgent(store)
    envelope = {
        "intake_id": "intake:test",
        "dedupe_key": "dedupe:test",
        "content_hash": "content:test",
        "candidate_id": "candidate:test",
        "objects": {},
        "relations": [],
    }
    trace_payload = build_trace_review_payload(
        source_ledger_hash="ledger:1",
        decisions={
            "trace_phase": {
                "operations": [{
                    "op": "create_trace_group",
                    "local_trace_ref": "T1",
                    "case_refs": ["C1"],
                }]
            }
        },
        compiled_trace_bundle={"objects": {}, "relations": []},
        allowed_message_ids=["m2"],
    )
    item = queue.build_w7_trace_review_item(
        envelope,
        {
            "decision": "admit",
            "admission_readiness": "execution_ready",
            "merge_allowed": True,
            "materialize_allowed": True,
        },
        trace_review_payload=trace_payload,
    )
    queue.enqueue("v2_typed_candidates", item)
    approved = queue.mark_decision(
        "v2_typed_candidates",
        item["review_id"],
        "approve",
        reviewer="human",
    )
    assert approved["approved_content_hash"].startswith("review-subject:")
    row = queue.read_queue("v2_typed_candidates")[0]
    assert approval_hash_matches(row) is True

    corrected = queue.append_trace_correction(
        "v2_typed_candidates",
        item["review_id"],
        "change_status",
        target_ref="T1",
        payload={"to": "recurrence"},
        evidence_message_ids=["m2"],
        reviewer="human",
    )
    assert corrected["status"] == "correction_recorded"
    row = queue.read_queue("v2_typed_candidates")[0]
    assert row["review_status"] == "needs_re_review"
    assert row["human_approved"] is False
    assert "approved_content_hash" not in row

    queue.mark_decision(
        "v2_typed_candidates",
        item["review_id"],
        "approve",
        reviewer="human",
    )
    row = queue.read_queue("v2_typed_candidates")[0]
    assert approval_hash_matches(row) is True

    same_item = queue.build_w7_trace_review_item(
        envelope,
        {
            "decision": "admit",
            "admission_readiness": "execution_ready",
            "merge_allowed": True,
            "materialize_allowed": True,
        },
        trace_review_payload=trace_payload,
    )
    queue.enqueue("v2_typed_candidates", same_item)
    row = queue.read_queue("v2_typed_candidates")[0]
    assert len(row["correction_events"]) == 1
    assert row["review_status"] == "approved"
    assert approval_hash_matches(row) is True
    blocked = IncrementalIngestV2Agent(
        store
    ).apply_approved_typed_review_item(row)
    assert blocked["reason"] == "correction_events_not_compiled"

    row["correction_events"][0]["payload"]["to"] = "verified"
    assert approval_hash_matches(row) is False
    applied = IncrementalIngestV2Agent(
        store
    ).apply_approved_typed_review_item(row)
    assert applied["reason"] == "approval_content_hash_mismatch"


def test_pipeline_w2_atomic_adapter_only_extracts_eligible_cases(
    tmp_path: Path,
):
    pipeline = WriteSidePipeline(JsonKGStore(tmp_path / "legacy"))

    class FakeW2:
        def extract(self, episode, w2_mode=None):
            return {
                "source_episode_id": episode["episode_id"],
                "production_schema_valid": True,
            }

    pipeline.w2 = FakeW2()
    episode = _attachment_episode()
    ledger = build_episode_source_ledger(episode)
    manifest, issues = build_atomic_case_manifest(
        episode=episode,
        source_ledger=ledger,
        case_boundary={
            "case_fragments": [{
                "fragment_ref": "F1",
                "case_kind": "diagnostic_case",
                "fault_summary": "设备蓝屏",
                "source_message_ids": ["m1"],
            }, {
                "fragment_ref": "F2",
                "case_kind": "coordination_only",
                "fault_summary": "请老师查看",
                "source_message_ids": ["m0"],
            }],
            "non_case_message_ids": [],
        },
        evidence_anchor={
            "anchor_decisions": [],
            "unassigned_evidence_message_ids": ["m0"],
        },
    )
    assert issues == []
    result = pipeline.extract_w7_atomic_cases(
        manifest,
        w2_mode="native_v2",
    )
    assert result["summary"] == {
        "atomic_cases": 1,
        "candidates": 1,
        "schema_valid": 1,
    }
    assert result["queue_written"] is False
    assert result["kg_mutated"] is False
    assert len(result["w7b_case_cards"]) == 1


def _execution_objects() -> dict:
    return {
        "DiagnosticAction": [
            {"action_id": "a1"},
            {"action_id": "a2"},
        ],
        "ActionOutcome": [
            {
                "outcome_id": "o1",
                "action_id": "a1",
                "outcome_type": "partial_temporary",
                "evidence_ids": ["e1"],
            },
            {
                "outcome_id": "o2",
                "action_id": "a2",
                "outcome_type": "verified_fix",
                "evidence_ids": ["e2"],
            },
        ],
        "DiagnosticTrace": [{
            "trace_id": "t1",
            "source_case_id": "c1",
            "recommended_action_ids": ["a1", "a2"],
            "actual_action_ids": ["a1", "a2"],
            "evidence_ids": ["e1"],
        }],
        "TraceStep": [
            {
                "trace_step_id": "old1",
                "trace_id": "t1",
                "source_case_id": "c1",
                "action_id": "a1",
                "evidence_ids": ["e1"],
            },
            {
                "trace_step_id": "old2",
                "trace_id": "t1",
                "source_case_id": "c1",
                "action_id": "a2",
                "evidence_ids": ["e2"],
            },
        ],
        "ExecutionObservation": [],
        "BranchRule": [],
    }


def test_trace_compiler_is_deterministic_and_keeps_input_immutable():
    source = _execution_objects()
    original = deepcopy(source)
    first_objects, first_relations, _ = TraceCompiler().compile(source, [])
    second_objects, second_relations, _ = TraceCompiler().compile(
        first_objects, first_relations
    )
    assert source == original
    assert first_objects == second_objects
    assert first_relations == second_relations
    assert [item["ordinal"] for item in first_objects["TraceStep"]] == [1, 2]
    assert first_objects["BranchRule"][-1]["terminal_status"] == "resolved"


def test_trace_compiler_review_bundle_is_content_addressed():
    phase = {
        "operations": [{
            "op": "create_trace_group",
            "local_trace_ref": "T1",
            "case_refs": ["C1"],
        }, {
            "op": "set_phase",
            "local_trace_ref": "T1",
            "case_ref": "C1",
            "phase_index": 1,
            "event_type": "report",
            "relation_type": "trace_root",
            "after_case_ref": "",
            "evidence_message_ids": ["m1"],
            "summary": "设备蓝屏",
        }],
    }
    outcome = {
        "operations": [{
            "local_trace_ref": "T1",
            "to": "verified",
            "evidence_message_ids": ["m2"],
        }],
    }
    compiler = TraceCompiler()
    first = compiler.compile_review_bundle(
        case_cards=[{"case_ref": "C1", "title": "设备蓝屏"}],
        phase_patch=phase,
        outcome_patch=outcome,
    )
    second = compiler.compile_review_bundle(
        case_cards=[{"case_ref": "C1", "title": "设备蓝屏"}],
        phase_patch=deepcopy(phase),
        outcome_patch=deepcopy(outcome),
    )
    assert first == second
    assert first["traces"][0]["resolution_status"] == "verified"
    assert first["compiled_bundle_hash"]


def test_shadow_cli_input_shape_can_be_serialized(tmp_path: Path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [_episode()]}, ensure_ascii=False))
    assert json.loads(path.read_text())["episodes"][0]["episode_id"] == "ep-1"


def _correction_review_payload() -> dict:
    cards = [{
        "case_ref": "C1",
        "source_case_id": "c1",
        "case_kind": "diagnostic_case",
        "title": "首次蓝屏",
        "source_message_ids": ["m1"],
        "evidence_message_ids": ["m1"],
    }, {
        "case_ref": "C2",
        "source_case_id": "c2",
        "case_kind": "diagnostic_case",
        "title": "蓝屏复发",
        "source_message_ids": ["m2"],
        "evidence_message_ids": ["m2"],
    }]
    phase = {
        "schema_version": "w7.trace_phase_patch.v1",
        "operations": [{
            "op": "create_trace_group",
            "local_trace_ref": "T1",
            "case_refs": ["C1", "C2"],
        }, {
            "op": "set_phase",
            "local_trace_ref": "T1",
            "case_ref": "C1",
            "phase_index": 1,
            "event_type": "report",
            "relation_type": "trace_root",
            "after_case_ref": "",
            "evidence_message_ids": ["m1"],
            "summary": "首次蓝屏",
        }, {
            "op": "set_phase",
            "local_trace_ref": "T1",
            "case_ref": "C2",
            "phase_index": 2,
            "event_type": "recurrence",
            "relation_type": "recurrence_of",
            "after_case_ref": "C1",
            "evidence_message_ids": ["m2"],
            "summary": "蓝屏复发",
        }],
        "standalone_case_refs": [],
        "uncertainties": [],
    }
    outcome = {
        "schema_version": "w7.outcome_patch.v1",
        "operations": [{
            "op": "revise_trace_status",
            "local_trace_ref": "T1",
            "from": "provisionally_resolved",
            "to": "recurrence",
            "evidence_message_ids": ["m2"],
            "reason": "再次蓝屏",
        }],
        "uncertainties": [],
    }
    compiled = TraceCompiler().compile_review_bundle(
        case_cards=cards,
        phase_patch=phase,
        outcome_patch=outcome,
    )
    return build_trace_review_payload(
        source_ledger_hash="ledger:correction",
        decisions={
            "trace_phase": phase,
            "outcome_reconciliation": outcome,
        },
        compiled_trace_bundle=compiled,
        allowed_message_ids=["m1", "m2"],
        case_cards=cards,
    )


def _correction_typed_candidate() -> dict:
    return {
        "candidate_id": "candidate:correction",
        "intake_id": "intake:correction",
        "dedupe_key": "dedupe:correction",
        "content_hash": "content:before",
        "objects": {
            "FaultFamily": [{"family_id": "f1"}],
            "FaultVariant": [
                {"variant_id": "v1", "family_id": "f1"}
            ],
            "SourceCase": [
                {"case_id": "c1"},
                {"case_id": "c2"},
            ],
            "EvidenceItem": [{
                "evidence_id": "e1",
                "external_id": "m1",
            }, {
                "evidence_id": "e2",
                "external_id": "m2",
            }],
            "DiagnosticAction": [{
                "action_id": "a1",
                "family_id": "f1",
                "variant_id": "v1",
                "source_kind": "case",
                "execution_status": "actual",
                "evidence_ids": ["e1", "e2"],
            }],
            "ActionOutcome": [{
                "outcome_id": "o1",
                "action_id": "a1",
                "source_case_id": "c1",
                "outcome_type": "partial_temporary",
                "evidence_ids": ["e1"],
            }, {
                "outcome_id": "o2",
                "action_id": "a1",
                "source_case_id": "c2",
                "outcome_type": "ineffective",
                "evidence_ids": ["e2"],
            }],
            "DiagnosticTrace": [{
                "trace_id": "old-t1",
                "family_id": "f1",
                "variant_id": "v1",
                "source_case_id": "c1",
                "recommended_action_ids": ["a1"],
                "actual_action_ids": ["a1"],
                "evidence_ids": ["e1"],
            }, {
                "trace_id": "old-t2",
                "family_id": "f1",
                "variant_id": "v1",
                "source_case_id": "c2",
                "recommended_action_ids": ["a1"],
                "actual_action_ids": ["a1"],
                "evidence_ids": ["e2"],
            }],
            "TraceStep": [{
                "trace_step_id": "old-s1",
                "trace_id": "old-t1",
                "source_case_id": "c1",
                "action_id": "a1",
                "ordinal": 1,
                "execution_status": "actual",
                "attempt_index": 1,
                "evidence_ids": ["e1"],
            }, {
                "trace_step_id": "old-s2",
                "trace_id": "old-t2",
                "source_case_id": "c2",
                "action_id": "a1",
                "ordinal": 1,
                "execution_status": "actual",
                "attempt_index": 1,
                "evidence_ids": ["e2"],
            }],
            "ExecutionObservation": [],
            "BranchRule": [],
            "RequiredInfoSpec": [],
        },
        "relations": [],
    }


def test_correction_compiler_materializes_repeated_action_occurrences():
    payload = _correction_review_payload()
    event, event_issues = build_correction_event(
        review_id="review:correction",
        operation="move_phase",
        target_ref="C2",
        payload={"phase_index": 1},
        evidence_message_ids=[],
        reviewer="human",
        note="复发应作为首个可见阶段",
        sequence=1,
        base_subject_hash=correction_chain_subject_hash(payload, []),
        allowed_target_refs={"C1", "C2", "T1"},
        allowed_message_ids={"m1", "m2"},
        created_at="2026-07-27T00:00:00+00:00",
    )
    assert event_issues == []
    result, issues = compile_trace_corrections(
        trace_review_payload=payload,
        correction_events=[event],
    )
    assert issues == []
    assert result["kg_materialization_ready"] is True
    phases = result["corrected_compiled_trace_bundle"]["traces"][0][
        "phases"
    ]
    assert [value["case_ref"] for value in phases] == ["C2", "C1"]

    candidate, materialize_issues = materialize_corrected_typed_candidate(
        typed_candidate=_correction_typed_candidate(),
        correction_compile_result=result,
    )
    assert materialize_issues == []
    trace = candidate["objects"]["DiagnosticTrace"][0]
    assert trace["source_case_ids"] == ["c2", "c1"]
    assert [value["action_id"] for value in trace["action_occurrences"]] == [
        "a1", "a1",
    ]
    assert [
        value["source_case_id"]
        for value in candidate["objects"]["TraceStep"]
    ] == ["c2", "c1"]
    assert [value["ordinal"] for value in candidate["objects"]["TraceStep"]] == [
        1, 2,
    ]


def test_w6_compiles_corrections_revalidates_and_rebinds_hash(
    tmp_path: Path,
):
    store = JsonKGV2Store(tmp_path / "kg-v2")
    queue = ReviewQueueAgent(store)
    item = queue.build_w7_trace_review_item(
        _correction_typed_candidate(),
        {
            "decision": "admit",
            "admission_readiness": "execution_ready",
            "merge_allowed": True,
            "materialize_allowed": True,
        },
        trace_review_payload=_correction_review_payload(),
    )
    queue.enqueue("v2_typed_candidates", item)
    appended = queue.append_trace_correction(
        "v2_typed_candidates",
        item["review_id"],
        "change_status",
        target_ref="T1",
        payload={"to": "ineffective"},
        evidence_message_ids=["m2"],
        reviewer="human",
    )
    assert appended["status"] == "correction_recorded"
    compiled = queue.compile_trace_corrections(
        "v2_typed_candidates",
        item["review_id"],
        quality_gate_scorer=lambda candidate: {
            "decision": "admit",
            "admission_readiness": "execution_ready",
            "merge_allowed": True,
            "materialize_allowed": True,
        },
    )
    assert compiled["status"] == "corrections_compiled"
    row = queue.read_queue("v2_typed_candidates")[0]
    assert row["correction_overlay_applied"] is True
    assert row["review_status"] == "pending"
    assert row["typed_candidate"]["content_hash"].startswith("content:w7:")
    queue.mark_decision(
        "v2_typed_candidates",
        item["review_id"],
        "approve",
        reviewer="human",
    )
    row = queue.read_queue("v2_typed_candidates")[0]
    assert approval_hash_matches(row) is True
    row["typed_candidate"]["objects"]["DiagnosticTrace"][0][
        "resolution_status"
    ] = "verified"
    assert approval_hash_matches(row) is False


def test_structural_split_compiles_semantics_but_requires_w2_reextract():
    payload = _correction_review_payload()
    payload = {
        **payload,
        "allowed_message_ids": ["m1", "m2", "m3"],
        "case_cards": [{
            **payload["case_cards"][0],
            "source_message_ids": ["m1", "m3"],
            "evidence_message_ids": ["m1", "m3"],
        }, payload["case_cards"][1]],
    }

    event, event_issues = build_correction_event(
        review_id="review:correction",
        operation="split_case",
        target_ref="C1",
        payload={"new_cases": [{
            "case_ref": "C1a",
            "title": "首次蓝屏",
            "source_message_ids": ["m1"],
        }, {
            "case_ref": "C1b",
            "title": "独立供电异常",
            "source_message_ids": ["m3"],
        }]},
        evidence_message_ids=[],
        reviewer="human",
        note="一条消息含两个问题",
        sequence=1,
        base_subject_hash=correction_chain_subject_hash(payload, []),
        allowed_target_refs={"C1", "C2", "T1"},
        allowed_message_ids={"m1", "m2", "m3"},
        created_at="2026-07-27T00:00:00+00:00",
    )
    assert event_issues == []
    result, issues = compile_trace_corrections(
        trace_review_payload=payload,
        correction_events=[event],
    )
    assert issues == []
    assert result["requires_w2_reextract"] is True
    assert result["kg_materialization_ready"] is False


def test_structural_correction_persists_and_fulfills_w2_reextract_request(
    tmp_path: Path,
):
    store = JsonKGV2Store(tmp_path / "kg-v2")
    queue = ReviewQueueAgent(store)
    base = _correction_review_payload()
    payload = {
        **base,
        "allowed_message_ids": ["m1", "m2", "m3"],
        "case_cards": [{
            **base["case_cards"][0],
            "source_message_ids": ["m1", "m3"],
            "evidence_message_ids": ["m1", "m3"],
        }, base["case_cards"][1]],
    }
    item = queue.build_w7_trace_review_item(
        _correction_typed_candidate(),
        {
            "decision": "admit",
            "admission_readiness": "execution_ready",
            "merge_allowed": True,
            "materialize_allowed": True,
        },
        trace_review_payload=payload,
    )
    queue.enqueue("v2_typed_candidates", item)
    event, issues = build_correction_event(
        review_id=item["review_id"],
        operation="split_case",
        target_ref="C1",
        payload={"new_cases": [{
            "case_ref": "C1a",
            "title": "首次蓝屏",
            "source_message_ids": ["m1"],
        }, {
            "case_ref": "C1b",
            "title": "独立供电异常",
            "source_message_ids": ["m3"],
        }]},
        evidence_message_ids=[],
        reviewer="human",
        note="拆分",
        sequence=1,
        base_subject_hash=correction_chain_subject_hash(payload, []),
        allowed_target_refs={"C1", "C2", "T1"},
        allowed_message_ids={"m1", "m2", "m3"},
        created_at="2026-07-27T00:00:00+00:00",
    )
    assert issues == []
    # Use the public append path to ensure the persisted request is created by
    # the same code path used by W6.
    appended = queue.append_trace_correction(
        "v2_typed_candidates",
        item["review_id"],
        "split_case",
        target_ref="C1",
        payload=event["payload"],
        reviewer="human",
    )
    assert appended["status"] == "correction_recorded"
    compiled = queue.compile_trace_corrections(
        "v2_typed_candidates", item["review_id"]
    )
    assert compiled["status"] == "requires_w2_reextract"
    request = compiled["reextract_request"]
    row = queue.read_queue("v2_typed_candidates")[0]
    assert row["review_status"] == "needs_w2_reextract"
    assert row["reextract_request"]["status"] == "pending"

    fulfilled = queue.fulfill_w2_reextract(
        "v2_typed_candidates",
        item["review_id"],
        typed_candidate={
            **_correction_typed_candidate(),
            "content_hash": "content:w7:reextracted",
        },
        quality_gate={
            "decision": "route_review",
            "admission_readiness": "review_only",
            "merge_allowed": True,
            "materialize_allowed": False,
        },
        trace_review_payload=request["trace_review_payload"],
        reextract_request_id=request["request_id"],
    )
    assert fulfilled["status"] == "reextract_fulfilled"
    row = queue.read_queue("v2_typed_candidates")[0]
    assert row["review_status"] == "pending"
    assert row["reextract_request"]["status"] == "fulfilled"
    assert row["typed_candidate"]["content_hash"] == "content:w7:reextracted"


def test_batch_orchestrator_runs_w7a_per_episode_and_w7b_across_session(
    tmp_path: Path,
):
    episodes = [{
        "episode_id": "ep-a",
        "thread_id": "thread-1",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "设备蓝屏首次发生",
        }],
    }, {
        "episode_id": "ep-b",
        "thread_id": "thread-1",
        "fault_description_messages": [{
            "message_id": "m2",
            "text": "设备蓝屏处理后恢复正常生产",
        }],
    }]
    extracted: list[str] = []

    def fake_atomic_extractor(manifest):
        parent = str(manifest["parent_episode_id"])
        extracted.append(parent)
        atomic = manifest["atomic_cases"][0]
        message_ids = list(atomic["source_message_ids"])
        return {
            "schema_version": "w7.atomic_w2_result.v1",
            "candidates": [{
                "candidate_id": f"candidate:{parent}",
                "production_schema_valid": True,
            }],
            "w7b_case_cards": [{
                "case_ref": f"case:{parent}",
                "source_case_id": f"case:{parent}",
                "candidate_id": f"candidate:{parent}",
                "atomic_episode_id": atomic["atomic_episode_id"],
                "parent_episode_id": parent,
                "case_kind": "diagnostic_case",
                "title": "设备蓝屏",
                "fault_summary": "设备蓝屏",
                "jira_keys": ["SMTAOITS-1"],
                "source_message_ids": message_ids,
                "evidence_message_ids": message_ids,
                "production_schema_valid": True,
            }],
            "summary": {
                "atomic_cases": 1,
                "candidates": 1,
                "schema_valid": 1,
            },
        }

    result = W7BatchShadowOrchestrator(
        W7ShadowOrchestrator(
            client=FakeDecisionClient(),
            checkpoint_root=tmp_path / "checkpoints",
        )
    ).run(
        batch_id="thread-1",
        episodes=episodes,
        atomic_extractor=fake_atomic_extractor,
    )
    assert result["schema_valid"] is True
    assert extracted == ["ep-a", "ep-b"]
    assert result["stats"]["source_units"] == 2
    assert result["stats"]["case_cards"] == 2
    assert result["trace_compiler"]["bundle"]["traces"][0][
        "case_refs"
    ] == ["case:ep-a", "case:ep-b"]
    assert result["w6_trace_review_payload"]["decisions"][
        "w7a_units"
    ][0]["episode_id"] == "ep-a"


def test_batch_source_ledger_rejects_conflicting_duplicate_message():
    left = build_episode_source_ledger({
        "episode_id": "a",
        "thread_id": "t",
        "messages": [{"message_id": "m1", "text": "蓝屏"}],
    })
    right = build_episode_source_ledger({
        "episode_id": "b",
        "thread_id": "t",
        "messages": [{"message_id": "m1", "text": "相机异常"}],
    })
    ledger, issues = build_batch_source_ledger(
        [left, right], batch_id="t"
    )
    assert ledger["allowed_message_ids"] == ["m1"]
    assert issues == ["batch_message_identity_conflict:m1"]


def test_batch_source_ledger_recomposes_w1_fragments_with_same_message_id():
    common = {
        "message_id": "m1",
        "source_message_id": "m1",
        "create_time": "2026-01-01 10:00",
        "msg_type": "text",
        "sender": {"name": "FAE"},
        "fragment_count": 2,
        "attachment_refs": [],
    }
    left = {
        "source_thread_id": "t",
        "episode_id": "ep-1",
        "core_message_ids": ["m1"],
        "rows": [{
            **common,
            "fragment_index": 1,
            "text": "1.相机拍摄失败",
            "content_summary": "1.相机拍摄失败",
        }],
    }
    right = {
        "source_thread_id": "t",
        "episode_id": "ep-2",
        "core_message_ids": ["m1"],
        "rows": [{
            **common,
            "fragment_index": 2,
            "text": "2.蓝屏",
            "content_summary": "2.蓝屏",
        }],
    }
    ledger, issues = build_batch_source_ledger(
        [left, right], batch_id="t"
    )
    assert issues == []
    assert ledger["allowed_message_ids"] == ["m1"]
    assert ledger["rows"][0]["fragment_index"] == 0
    assert ledger["rows"][0]["text"] == "1.相机拍摄失败\n2.蓝屏"
    assert [
        item["fragment_index"]
        for item in ledger["rows"][0]["source_fragments"]
    ] == [1, 2]


def test_batch_typed_candidate_materializes_w7b_trace_semantics():
    payload = _correction_review_payload()
    base = _correction_typed_candidate()
    batch_result = {
        "batch_id": "thread-1",
        "source_ledger_hash": "ledger:correction",
        "result_hash": "result:correction",
        "source_ledger": {
            "source_thread_ids": ["thread-1"],
            "episode_ids": ["ep-a", "ep-b"],
            "allowed_message_ids": ["m1", "m2"],
            "rows": [
                {"message_id": "m1", "text": "首次蓝屏"},
                {"message_id": "m2", "text": "蓝屏复发"},
            ],
        },
        "w2_candidates": [{
            "candidate_id": "w2:combined",
            "candidate_draft_v2_normalized_bundle": {
                "schema_valid": True,
                "schema_issues": [],
                "objects": base["objects"],
                "relations": base["relations"],
            },
        }],
        "case_cards": payload["case_cards"],
        "trace_compiler": {
            "bundle": payload["compiled_trace_bundle"],
        },
        "w6_trace_review_payload": payload,
    }
    candidate, issues = build_w7_batch_typed_candidate(batch_result)
    assert candidate["content_hash"].startswith("content:w7-batch:")
    assert candidate["w7_compiled_trace_bundle"][
        "compiled_bundle_hash"
    ]
    assert len(candidate["objects"]["DiagnosticTrace"]) == 1
    assert candidate["objects"]["DiagnosticTrace"][0][
        "source_case_ids"
    ] == ["c1", "c2"]
    assert {
        item["external_id"]
        for item in candidate["objects"]["EvidenceItem"]
    } >= {"m1", "m2"}
    assert not any(
        issue.startswith("resolution_evidence_mapping_incomplete:")
        for issue in issues
    )
    # The deliberately compact fixture omits descriptive schema fields; the
    # builder must retain the graph while surfacing validator failures.
    assert any(
        issue.startswith("kg_v2_validator:")
        for issue in issues
    )


def test_materializer_supports_actionless_w2_source_case_trace():
    payload = _correction_review_payload()
    base = _correction_typed_candidate()
    for object_type in (
        "DiagnosticAction",
        "ActionOutcome",
        "DiagnosticTrace",
        "TraceStep",
        "ExecutionObservation",
        "BranchRule",
    ):
        base["objects"][object_type] = []
    materialized, issues = materialize_corrected_typed_candidate(
        typed_candidate=base,
        correction_compile_result={
            "kg_materialization_ready": True,
            "compile_result_hash": "compile:actionless",
            "corrected_trace_review_payload": payload,
            "corrected_compiled_trace_bundle": payload[
                "compiled_trace_bundle"
            ],
        },
    )
    assert not any(
        issue.startswith("compiled_trace_without_w2_trace:")
        for issue in issues
    )
    assert len(materialized["objects"]["DiagnosticTrace"]) == 1
    assert materialized["objects"]["DiagnosticTrace"][0][
        "recommended_action_ids"
    ] == []


def test_batch_candidate_reconciles_case_wording_for_same_fault_family():
    payload = _correction_review_payload()
    base = _correction_typed_candidate()
    first = deepcopy(base)
    second = deepcopy(base)
    first["objects"]["FaultFamily"][0].update({
        "label": "相机拍摄失败",
        "category": "硬件与运控",
        "source_kind": "case",
        "summary": "生产中相机拍摄失败",
        "keywords": ["相机"],
    })
    second["objects"]["FaultFamily"][0].update({
        "label": "相机拍摄失败",
        "category": "系统与软件异常",
        "source_kind": "case",
        "summary": "相机在采图阶段出现超时或空图",
        "keywords": ["超时"],
    })
    candidate, issues = build_w7_batch_typed_candidate({
        "batch_id": "thread-1",
        "source_ledger_hash": "ledger",
        "result_hash": "result",
        "source_ledger": {
            "source_thread_ids": ["thread-1"],
            "episode_ids": ["ep-1"],
            "allowed_message_ids": ["m1", "m2"],
            "rows": [
                {"message_id": "m1", "text": "相机异常"},
                {"message_id": "m2", "text": "恢复"},
            ],
        },
        "w2_candidates": [{
            "candidate_id": "one",
            "candidate_draft_v2_normalized_bundle": {
                "schema_valid": True,
                "schema_issues": [],
                "objects": first["objects"],
                "relations": first["relations"],
            },
        }, {
            "candidate_id": "two",
            "candidate_draft_v2_normalized_bundle": {
                "schema_valid": True,
                "schema_issues": [],
                "objects": second["objects"],
                "relations": second["relations"],
            },
        }],
        "case_cards": payload["case_cards"],
        "trace_compiler": {
            "bundle": payload["compiled_trace_bundle"],
        },
        "w6_trace_review_payload": payload,
    })
    assert not any(
        issue.startswith("w7_batch_object_conflict:FaultFamily:")
        for issue in issues
    )
    family = candidate["objects"]["FaultFamily"][0]
    assert family["summary"] == "生产中相机拍摄失败"
    assert family["category"] == "硬件与运控"
    assert family["keywords"] == ["相机", "超时"]


def test_calibration_input_contains_sources_but_not_human_labels(
    tmp_path: Path,
):
    review_root = tmp_path / "review"
    context_root = review_root / "full_context"
    context_root.mkdir(parents=True)
    source_episode = {
        "episode_id": "ep-1",
        "thread_id": "thread-1",
        "messages": [{"message_id": "m1", "text": "设备蓝屏"}],
    }
    (context_root / "session.json").write_text(
        json.dumps({"source_episodes": [source_episode]}),
        encoding="utf-8",
    )
    annotations = {
        "sessions": [{
            "thread_id": "thread-1",
            "reviewer": "human",
            "session_verdict": "needs_fix",
            "full_context_json": "full_context/session.json",
            "episodes": [{
                "episode_id": "ep-1",
                "corrected_fault_focus": "不应泄露给模型",
            }],
        }],
    }
    annotations_path = review_root / "human_annotations.json"
    annotations_path.write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    out_path = tmp_path / "source_input.json"
    result = build_w7_calibration_input(
        annotations_path=annotations_path,
        out_path=out_path,
        limit=1,
    )
    rendered = out_path.read_text(encoding="utf-8")
    assert result["summary"] == {"sessions": 1, "episodes": 1}
    assert result["episodes"] == [source_episode]
    assert "不应泄露给模型" not in rendered
    assert result["selection_policy"][
        "labels_excluded_from_model_input"
    ] is True


def test_calibration_scorer_uses_episode_pairwise_trace_relationships(
    tmp_path: Path,
):
    thread_id = "thread-1"
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({
            "units": [{
                "source_thread_id": thread_id,
                "episode_id": "ep-1",
            }, {
                "source_thread_id": thread_id,
                "episode_id": "ep-2",
            }],
            "case_cards": [{
                "case_ref": "C1",
                "parent_episode_id": "ep-1",
                "fault_summary": "设备蓝屏",
                "production_schema_valid": True,
            }, {
                "case_ref": "C2",
                "parent_episode_id": "ep-2",
                "fault_summary": "设备蓝屏复发",
                "production_schema_valid": True,
            }],
            "trace_compiler": {
                "bundle": {
                    "traces": [{
                        "compiled_trace_id": "model:T1",
                        "resolution_status": "verified",
                        "resolution_evidence_message_ids": ["m2"],
                        "phases": [{
                            "case_ref": "C1",
                            "phase_index": 1,
                        }, {
                            "case_ref": "C2",
                            "phase_index": 2,
                        }],
                    }],
                },
            },
        }),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"results": [{"result": str(result_path)}]}),
        encoding="utf-8",
    )
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps({
            "sessions": [{
                "thread_id": thread_id,
                "reviewer": "human",
                "session_verdict": "needs_fix",
                "episodes": [{
                    "episode_id": "ep-1",
                    "fault_focus_correct": False,
                    "corrected_fault_focus": "设备蓝屏",
                    "trace_group_correct": False,
                    "corrected_trace_group_id": "human:T1",
                    "trace_phase_correct": False,
                    "corrected_trace_phase_index": 1,
                    "corrected_trace_phase_count": 2,
                    "resolution_status_correct": False,
                    "corrected_resolution_status": "verified",
                    "resolution_evidence_correct": False,
                    "corrected_resolution_evidence_message_ids": ["m2"],
                    "w2_readiness_correct": False,
                    "corrected_w2_readiness": True,
                }, {
                    "episode_id": "ep-2",
                    "fault_focus_correct": False,
                    "corrected_fault_focus": "设备蓝屏复发",
                    "trace_group_correct": False,
                    "corrected_trace_group_id": "human:T1",
                    "trace_phase_correct": False,
                    "corrected_trace_phase_index": 2,
                    "corrected_trace_phase_count": 2,
                    "resolution_status_correct": False,
                    "corrected_resolution_status": "verified",
                    "resolution_evidence_correct": False,
                    "corrected_resolution_evidence_message_ids": ["m2"],
                    "w2_readiness_correct": False,
                    "corrected_w2_readiness": True,
                }],
            }],
        }),
        encoding="utf-8",
    )
    report = score_w7_multi_agent(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        session_limit=1,
    )
    assert report["gate"]["status"] == "PASS"
    assert report["metrics"]["strict_episode_match"]["rate"] == 1.0
    assert report["metrics"]["trace_pairwise"]["f1"] == 1.0


def test_calibration_scorer_reports_source_observable_trace_metric(
    tmp_path: Path,
):
    thread_id = "thread-1"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "units": [
            {"source_thread_id": thread_id, "episode_id": "ep-1"},
            {"source_thread_id": thread_id, "episode_id": "ep-2"},
            {"source_thread_id": thread_id, "episode_id": "ep-3"},
        ],
        "case_cards": [{
            "case_ref": "C2",
            "parent_episode_id": "ep-2",
            "fault_summary": "设备蓝屏",
            "production_schema_valid": True,
        }, {
            "case_ref": "C3",
            "parent_episode_id": "ep-3",
            "fault_summary": "相机异常",
            "production_schema_valid": True,
        }],
        "trace_compiler": {"bundle": {"traces": [{
            "compiled_trace_id": "model:blue",
            "phases": [{"case_ref": "C2", "phase_index": 1}],
        }, {
            "compiled_trace_id": "model:camera",
            "phases": [{"case_ref": "C3", "phase_index": 1}],
        }]}},
    }), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "results": [{"result": str(result_path)}],
    }), encoding="utf-8")

    def annotation(
        episode_id: str,
        group: str,
        *,
        source_gap: bool = False,
    ) -> dict:
        return {
            "episode_id": episode_id,
            "w7_snapshot": {
                "fault_focus": "",
                "trace_group_id": group,
                "trace_phase_index": 0,
                "trace_phase_count": 0,
                "resolution_status": "",
                "w2_ready": False,
            },
            "issue_tags": ["source_context"] if source_gap else [],
        }

    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps({
        "sessions": [{
            "thread_id": thread_id,
            "reviewer": "human",
            "session_verdict": "needs_fix",
            "episodes": [
                annotation("ep-1", "human:blue", source_gap=True),
                annotation("ep-2", "human:blue"),
                annotation("ep-3", "human:camera"),
            ],
        }],
    }), encoding="utf-8")
    report = score_w7_multi_agent(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        session_limit=1,
    )
    assert report["metrics"]["trace_pairwise"]["recall"] == 0.0
    observable = report["metrics"]["trace_pairwise_input_observable"]
    assert observable["f1"] == 1.0
    assert observable["observable_episodes"] == 2
    assert observable["excluded_source_context_gap_episodes"] == 1
    assert (
        report["details"][0]["error_attribution"][0]
        == "upstream_source_context_gap"
    )


def test_calibration_scorer_uses_phase_local_resolution_status(
    tmp_path: Path,
):
    thread_id = "thread-1"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "units": [
            {"source_thread_id": thread_id, "episode_id": "ep-1"},
            {"source_thread_id": thread_id, "episode_id": "ep-2"},
            {"source_thread_id": thread_id, "episode_id": "ep-3"},
        ],
        "case_cards": [{
            "case_ref": f"C{index}",
            "parent_episode_id": f"ep-{index}",
            "fault_summary": "相机拍摄失败",
            "production_schema_valid": True,
        } for index in range(1, 4)],
        "trace_compiler": {"bundle": {"traces": [{
            "compiled_trace_id": "model:T1",
            "resolution_status": "recurrence",
            "phases": [{
                "case_ref": "C1",
                "phase_index": 1,
                "event_type": "report",
            }, {
                "case_ref": "C2",
                "phase_index": 2,
                "event_type": "short_term_recovery",
            }, {
                "case_ref": "C3",
                "phase_index": 3,
                "event_type": "recurrence",
            }],
        }]}},
    }), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "results": [{"result": str(result_path)}],
    }), encoding="utf-8")
    expected = ["pending", "ineffective", "recurrence"]
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps({
        "sessions": [{
            "thread_id": thread_id,
            "reviewer": "human",
            "session_verdict": "needs_fix",
            "episodes": [{
                "episode_id": f"ep-{index}",
                "w7_snapshot": {
                    "fault_focus": "相机拍摄失败",
                    "trace_group_id": "human:T1",
                    "trace_phase_index": index,
                    "trace_phase_count": 3,
                    "resolution_status": expected[index - 1],
                    "w2_ready": True,
                },
            } for index in range(1, 4)],
        }],
    }), encoding="utf-8")
    report = score_w7_multi_agent(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        session_limit=1,
    )
    assert report["metrics"]["resolution_status_match"]["rate"] == 1.0
    assert (
        report["metrics"]["trace_terminal_projection_match"]["rate"]
        == pytest.approx(1 / 3, abs=0.0001)
    )


def test_calibration_scorer_aligns_multi_case_episode_by_fault_focus(
    tmp_path: Path,
):
    thread_id = "thread-1"
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({
            "units": [{
                "source_thread_id": thread_id,
                "episode_id": "ep-1",
            }, {
                "source_thread_id": thread_id,
                "episode_id": "ep-2",
            }],
            "case_cards": [{
                "case_ref": "C1-unrelated",
                "parent_episode_id": "ep-1",
                "case_kind": "diagnostic_case",
                "fault_summary": "相机曝光异常",
                "production_schema_valid": True,
            }, {
                "case_ref": "C1-blue",
                "parent_episode_id": "ep-1",
                "case_kind": "diagnostic_case",
                "fault_summary": "设备蓝屏无法启动",
                "production_schema_valid": True,
            }, {
                "case_ref": "C2-blue",
                "parent_episode_id": "ep-2",
                "case_kind": "diagnostic_case",
                "fault_summary": "蓝屏问题再次复发",
                "production_schema_valid": True,
            }],
            "trace_compiler": {"bundle": {"traces": [{
                "compiled_trace_id": "model:camera",
                "resolution_status": "unknown",
                "phases": [{
                    "case_ref": "C1-unrelated",
                    "phase_index": 1,
                }],
            }, {
                "compiled_trace_id": "model:blue",
                "resolution_status": "unknown",
                "phases": [{
                    "case_ref": "C1-blue",
                    "phase_index": 1,
                }, {
                    "case_ref": "C2-blue",
                    "phase_index": 2,
                }],
            }]}},
        }),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"results": [{"result": str(result_path)}]}),
        encoding="utf-8",
    )

    def annotation(
        episode_id: str,
        focus: str,
        phase_index: int,
    ) -> dict:
        return {
            "episode_id": episode_id,
            "fault_focus_correct": False,
            "corrected_fault_focus": focus,
            "trace_group_correct": False,
            "corrected_trace_group_id": "human:blue",
            "trace_phase_correct": False,
            "corrected_trace_phase_index": phase_index,
            "corrected_trace_phase_count": 2,
            "resolution_status_correct": False,
            "corrected_resolution_status": "unknown",
            "resolution_evidence_correct": True,
            "w2_readiness_correct": False,
            "corrected_w2_readiness": True,
        }

    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps({"sessions": [{
            "thread_id": thread_id,
            "reviewer": "human",
            "session_verdict": "needs_fix",
            "episodes": [
                annotation("ep-1", "设备蓝屏无法启动", 1),
                annotation("ep-2", "蓝屏问题再次复发", 2),
            ],
        }]}),
        encoding="utf-8",
    )
    report = score_w7_multi_agent(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        session_limit=1,
    )
    first = report["details"][0]
    assert first["selected_case_ref"] == "C1-blue"
    assert first["predicted_trace_ref"] == "model:blue"
    assert first["predicted_trace_refs"] == ["model:blue"]
    assert report["metrics"]["trace_pairwise"]["f1"] == 1.0


def test_fixed_set_safety_gate_checks_coverage_and_no_mutation(
    tmp_path: Path,
):
    result_path = tmp_path / "result.json"
    valid_stage = {"schema_valid": True}
    result_path.write_text(
        json.dumps({
            "batch_id": "thread-1",
            "schema_valid": True,
            "promotion_allowed": False,
            "legacy_authoritative": True,
            "queue_written": False,
            "kg_mutated": False,
            "units": [{
                "episode_id": "ep-1",
                "w7a": {
                    stage: valid_stage for stage in (
                        "case_boundary",
                        "evidence_anchor",
                        "atomic_case_adapter",
                    )
                },
            }, {
                "episode_id": "ep-2",
                "w7a": {
                    stage: valid_stage for stage in (
                        "case_boundary",
                        "evidence_anchor",
                        "atomic_case_adapter",
                    )
                },
            }],
            "candidate_graph": {
                "schema_valid": True,
                "graph": {"edges": []},
            },
            "neighbor_link": valid_stage,
            "trace_components": {
                "schema_valid": True,
                "graph": {
                    "components": [{"case_refs": ["C1", "C2"]}]
                },
            },
            "trace_phase": valid_stage,
            "outcome_reconciliation": valid_stage,
            "trace_compiler": valid_stage,
            "stats": {"case_cards": 2},
            "typed_candidate": {"schema_valid": True},
            "typed_candidate_build_issues": [],
            "quality_gate": {"decision": "admit"},
        }),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "promotion_allowed": False,
            "legacy_authoritative": True,
            "state_hashes": {"unchanged": True},
            "results": [{"result": str(result_path)}],
        }),
        encoding="utf-8",
    )
    report = build_w7_safety_report(
        manifest_path=manifest_path,
        expected_episodes=2,
    )
    assert report["gate"]["status"] == "PASS"
    assert report["coverage"]["unique_episodes"] == 2
    assert report["graph"]["max_component_size"] == 2
