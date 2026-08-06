from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import debug_agent_system.agents.write.w1_message_relations as message_relations
from debug_agent_system.agents.write.w1_message_relations import (
    attachment_identity_keys,
    annotate_semantic_fragments,
    assign_reference_aware_segments,
    build_message_reference_graph,
    infer_cross_window_trace_edges,
    infer_context_continuation_edges,
    merge_xing_relation_history,
)
from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent


def test_mixed_daily_report_is_split_into_provenance_preserving_fragments() -> None:
    rows, report = annotate_semantic_fragments([
        _message("r1", "c1", "2026-01-01 09:00", "今日工作汇总：1. 相机拍摄失败；2. 工控机蓝屏")
    ])

    fragments = rows[0]["semantic_fragments"]
    assert len(fragments) == 2
    assert {item["source_message_id"] for item in fragments} == {"r1"}
    assert fragments[0]["fragment_id"] != fragments[1]["fragment_id"]
    assert report["mixed_report_message_count"] == 1


def test_cross_window_recurrence_links_same_equipment_without_joining_other_fault() -> None:
    rows = [
        _message("a1", "c1", "2026-01-01 09:00", "2030T 自动关机，检查模组电源"),
        _message("b1", "c1", "2026-01-03 09:00", "相机拍摄失败，检查网线"),
        _message("a2", "c1", "2026-01-10 09:00", "2030T 自动关机再次出现，仍然怀疑电源"),
    ]

    edges = infer_cross_window_trace_edges(rows)

    assert [(edge["source"], edge["target"]) for edge in edges] == [("a2", "a1")]


def test_cross_window_edge_rejoins_one_trace_but_preserves_parallel_fault() -> None:
    rows = [
        _message("a1", "c1", "2026-01-01 09:00", "2030T 自动关机，检查模组电源"),
        _message("b1", "c1", "2026-01-03 09:00", "相机拍摄失败，检查网线"),
        _message("a2", "c1", "2026-01-10 09:00", "2030T 自动关机再次出现，仍然怀疑电源"),
    ]
    edges = infer_cross_window_trace_edges(rows)
    segmented, report = assign_reference_aware_segments(rows, context_edges=edges)
    sessions = {row["message_id"]: row["thread_id"] for row in segmented}

    assert sessions["a1"] == sessions["a2"]
    assert sessions["b1"] != sessions["a1"]
    assert report["cross_window_trace_continuation_edge_count"] == 1


def test_mixed_daily_report_materializes_distinct_fault_episodes() -> None:
    message = _message(
        "r1",
        "c1",
        "2026-01-01 09:00",
        "今日工作汇总：1. D052相机拍摄失败，正在检查网线；2. T81工控机蓝屏，正在收集DMP。",
    )
    message["thread_id"] = "c1:report"
    message["sender"] = {"name": "FAE"}
    episodes = ChatCollectAgent().split_fault_episodes("c1:report", [message])

    assert len(episodes) == 2
    assert {episode["message_ids"][0] for episode in episodes} == {"r1"}
    fault_texts = [
        " ".join(item["text"] for item in episode["fault_description_messages"])
        for episode in episodes
    ]
    assert any("拍摄失败" in text for text in fault_texts)
    assert any("蓝屏" in text for text in fault_texts)


def _message(message_id: str, chat_id: str, time: str, text: str, **relations: str) -> dict:
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "create_time": time,
        "text": text,
        "attachments": [],
        **relations,
    }


def test_merge_prefers_message_id_and_keeps_old_attachment() -> None:
    old = [_message("m1", "c1", "2026-01-01 09:00", "蓝屏", attachments=[{"name": "MEMORY.DMP"}])]
    relation = [_message("m1", "c1", "2026-01-01 09:00", "蓝屏", root_id="r1", parent_id="p1")]

    merged, report = merge_xing_relation_history(old, relation)

    assert len(merged) == 1
    assert merged[0]["attachments"] == [{"name": "MEMORY.DMP"}]
    assert merged[0]["root_id"] == "r1"
    assert merged[0]["parent_id"] == "p1"
    assert report["matched_by_message_id"] == 1
    assert report["v3_only_messages"] == 0


def test_reference_graph_and_session_keep_reply_chain_together() -> None:
    rows = [
        _message("m1", "c1", "2026-01-01 09:00", "故障开始"),
        _message("m2", "c1", "2026-01-01 23:30", "回复排查", parent_id="m1", root_id="m1"),
        _message("m3", "c1", "2026-01-02 12:00", "无关新故障"),
    ]

    graph = build_message_reference_graph(rows)
    segmented, report = assign_reference_aware_segments(rows, quiet_gap_hours=12, max_messages=120)

    assert graph["stats"]["reply_edge_count"] == 1
    assert graph["stats"]["thread_membership_edge_count"] == 1
    assert segmented[0]["thread_id"] == segmented[1]["thread_id"]
    assert segmented[1]["thread_id"] != segmented[2]["thread_id"]
    assert report["reference_component_count"] == 1


def test_v3_only_messages_are_added_without_inventing_attachments() -> None:
    old = [_message("m1", "c1", "2026-01-01 09:00", "旧消息")]
    relation = [
        _message("m1", "c1", "2026-01-01 09:00", "旧消息", parent_id="p1"),
        _message("m2", "c1", "2026-01-01 09:01", "新消息", root_id="m1"),
    ]

    merged, report = merge_xing_relation_history(old, relation)

    assert {row["message_id"] for row in merged} == {"m1", "m2"}
    assert next(row for row in merged if row["message_id"] == "m2")["attachments"] == []
    assert report["v3_only_messages"] == 1


def test_duplicate_old_rows_collapse_and_preserve_distinct_attachments() -> None:
    old = [
        _message("m1", "c1", "2026-01-01 09:00", "蓝屏", attachments=[{"file_key": "a", "name": "a.dmp"}]),
        _message("m1", "c1", "2026-01-01 09:00", "蓝屏", attachments=[{"file_key": "b", "name": "b.log"}]),
    ]
    relation = [_message("m1", "c1", "2026-01-01 09:00", "蓝屏", parent_id="p1")]

    merged, report = merge_xing_relation_history(old, relation)

    assert len(merged) == 1
    assert {item["file_key"] for item in merged[0]["attachments"]} == {"a", "b"}
    assert report["old_duplicate_rows_collapsed"] == 1
    assert report["old_messages_after_dedupe"] == 1


def test_parallel_reference_components_split_one_temporal_window() -> None:
    rows = [
        _message("a1", "c1", "2026-01-01 09:00", "相机故障"),
        _message("b1", "c1", "2026-01-01 09:01", "蓝屏故障"),
        _message("a2", "c1", "2026-01-01 09:02", "相机排查", parent_id="a1", root_id="a1"),
        _message("b2", "c1", "2026-01-01 09:03", "蓝屏排查", parent_id="b1", root_id="b1"),
    ]

    segmented, report = assign_reference_aware_segments(rows)

    sessions = {row["message_id"]: row["thread_id"] for row in segmented}
    assert sessions["a1"] == sessions["a2"]
    assert sessions["b1"] == sessions["b2"]
    assert sessions["a1"] != sessions["b1"]
    assert report["parallel_temporal_blocks_split"] == 1


def test_inferred_context_continuation_recovers_pre_root_description() -> None:
    rows = [
        _message("m1", "c1", "2026-01-01 09:00", "这个错件两种物料太相似了，默认算法漏检"),
        _message("m2", "c1", "2026-01-01 09:03", "不是，那个灯芯大小不一样"),
        _message("m3", "c1", "2026-01-01 09:05", "似乎也反了，灯芯大小和颜色深浅默认参数漏检"),
        _message("m4", "c1", "2026-01-01 09:07", "我们的错件自定义框默认都是颜色匹配算法"),
    ]

    context_edges = infer_context_continuation_edges(rows)
    segmented, report = assign_reference_aware_segments(rows, context_edges=context_edges)
    graph = build_message_reference_graph(segmented, context_edges=context_edges)

    assert context_edges
    assert any(edge["shared_terms"] for edge in context_edges)
    assert len({row["thread_id"] for row in segmented}) == 1
    assert graph["stats"]["context_continuation_edge_count"] >= 2
    assert report["context_continuation_edge_count"] == len(context_edges)


def test_exact_lark_payload_identity_recovers_attachment_followup() -> None:
    shared = {"file_key": "file_v3_00abcdef1234567890", "name": "诊断数据.zip"}
    rows = [
        _message(
            "m1",
            "c1",
            "2026-01-01 09:00",
            "相机拍摄失败",
            attachments=[shared],
        ),
        _message(
            "m2",
            "c1",
            "2026-01-01 09:05",
            "补充这个附件",
            attachments=[shared],
        ),
    ]

    edges = infer_context_continuation_edges(rows)
    segmented, report = assign_reference_aware_segments(rows, context_edges=edges)

    assert [(edge["source"], edge["target"]) for edge in edges] == [("m2", "m1")]
    assert "same_artifact_payload" in edges[0]["reason_codes"]
    assert edges[0]["shared_artifact_payload_keys"] == [
        "lark-file-key:file_v3_00abcdef1234567890"
    ]
    assert len({row["thread_id"] for row in segmented}) == 1
    assert report["soft_context_edges_accepted"] == 1


def test_filename_match_without_payload_identity_does_not_link() -> None:
    rows = [
        _message(
            "m1",
            "c1",
            "2026-01-01 09:00",
            "相机拍摄失败",
            attachments=[{"file_key": "file_v3_00firstpayload1234", "name": "诊断数据.zip"}],
        ),
        _message(
            "m2",
            "c1",
            "2026-01-01 09:05",
            "补充这个附件",
            attachments=[{"file_key": "file_v3_00secondpayload123", "name": "诊断数据.zip"}],
        ),
    ]

    assert infer_context_continuation_edges(rows) == []


def test_payload_identity_alone_does_not_isolate_media_only_messages() -> None:
    shared = {"file_key": "file_v3_00abcdef1234567890", "name": "截图.jpg"}
    rows = [
        _message("m1", "c1", "2026-01-01 09:00", "[Image: first]", attachments=[shared]),
        _message("m2", "c1", "2026-01-01 09:00", "[Image: second]", attachments=[shared]),
    ]

    assert infer_context_continuation_edges(rows) == []


def test_exact_payload_hash_supports_conservative_cross_window_recurrence() -> None:
    digest = "a" * 64
    rows = [
        _message(
            "m1",
            "c1",
            "2026-01-01 09:00",
            "设备蓝屏",
            attachments=[{"file_key": "old-key", "file_sha256": digest}],
        ),
        _message(
            "m2",
            "c1",
            "2026-01-10 09:00",
            "设备蓝屏再次出现",
            attachments=[{"file_key": "new-key", "sha256": digest}],
        ),
    ]

    edges = infer_cross_window_trace_edges(rows)

    assert [(edge["source"], edge["target"]) for edge in edges] == [("m2", "m1")]
    assert "same_artifact_payload" in edges[0]["reason_codes"]
    assert edges[0]["shared_artifact_payload_keys"] == [f"sha256:{digest}"]


def test_small_fully_retrieved_local_payload_gets_bounded_content_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"same-diagnostic-payload")
    second.write_bytes(b"same-diagnostic-payload")
    message_relations._bounded_attachment_sha256.cache_clear()
    with patch.object(message_relations, "_ATTACHMENT_ALLOWED_ROOT", tmp_path):
        left = attachment_identity_keys({
            "attachments": [{
                "file_key": "message-one/file.bin",
                "path": str(first),
                "source_status": "api_ok",
            }]
        })
        right = attachment_identity_keys({
            "attachments": [{
                "file_key": "message-two/file.bin",
                "path": str(second),
                "source_status": "api_ok",
            }]
        })

    shared = left & right
    assert len(shared) == 1
    assert next(iter(shared)).startswith("sha256:")


def test_daily_report_is_not_inferred_as_context_continuation() -> None:
    rows = [
        _message("m1", "c1", "2026-01-01 09:00", "相机拍摄失败，正在排查"),
        _message("m2", "c1", "2026-01-01 09:10", "今日工作汇总：1.相机拍摄失败 2.工控机蓝屏"),
    ]

    context_edges = infer_context_continuation_edges(rows)

    assert context_edges == []


def test_daily_report_does_not_merge_upward_through_its_platform_root() -> None:
    rows = [
        _message("m1", "c1", "2026-01-01 09:00", "相机拍摄失败"),
        _message("m2", "c1", "2026-01-01 09:10", "今日工作汇总：1.相机拍摄失败 2.工控机蓝屏", root_id="m1", parent_id="m1"),
        _message("m3", "c1", "2026-01-01 09:11", "蓝屏继续排查", root_id="m2", parent_id="m2"),
    ]

    segmented, _ = assign_reference_aware_segments(rows)
    sessions = {row["message_id"]: row["thread_id"] for row in segmented}

    assert sessions["m2"] == sessions["m3"]
    assert sessions["m1"] != sessions["m2"]
    infer_cross_window_trace_edges,
