from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.eval.write_side.deepseek_trace_assembly_harness import (
    _decompose_with_adaptive_split,
    build_source_ledger,
    build_link_candidates,
    coverage_metrics,
    partition_assembly_components,
    partition_source_ledger,
    run_harness,
    validate_assembly,
    validate_decomposition,
    validate_neighbor_selection,
)


def _row(
    message_id: str,
    create_time: str,
    text: str,
    *,
    source_thread_id: str = "legacy-thread",
    chat_id: str = "chat-1",
    attachments: list[dict] | None = None,
) -> dict:
    return {
        "message_id": message_id,
        "thread_id": f"{chat_id}:refseg:1",
        "chat_id": chat_id,
        "sender": {"name": "FAE"},
        "create_time": create_time,
        "msg_type": "file" if attachments else "text",
        "text": text,
        "attachments": attachments or [],
        "links": [],
        "root_id": "",
        "parent_id": "",
        "semantic_fragments": [],
        "raw": {
            "chat_name": "测试群",
            "source_thread_ids": [source_thread_id],
            "segment_id": source_thread_id,
        },
    }


def test_source_ledger_keeps_orphan_attachment_and_temporal_neighbors(tmp_path: Path):
    path = tmp_path / "messages.jsonl"
    rows = [
        _row(
            "m1",
            "2026-05-20 18:51",
            "",
            attachments=[{
                "file_key": "f1",
                "name": "诊断数据.zip",
                "kind": "file",
                "extension": ".zip",
                "size": 10,
                "status": "metadata_only",
                "source_status": "api_ok",
                "evidence_role": "log_package",
            }],
        ),
        _row("m2", "2026-05-20 18:57", "设备正常运行中出现蓝屏"),
        _row(
            "m3",
            "2026-05-29 11:23",
            "更换内存",
            source_thread_id="next-thread",
        ),
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    ledger = build_source_ledger(path, "legacy-thread", neighbor_days=14)
    assert ledger["stats"]["rows"] == 3
    assert ledger["stats"]["core_rows"] == 2
    assert ledger["orphan_attachment_candidate_message_ids"] == ["m1"]
    assert {"m1", "m2"} <= set(ledger["high_signal_core_message_ids"])
    assert next(row for row in ledger["rows"] if row["message_id"] == "m3")["region"] == "neighbor"


def test_decomposition_rejects_unknown_evidence_ids():
    normalized, issues = validate_decomposition(
        {
            "case_items": [{
                "case_item_ref": "c1",
                "case_kind": "diagnostic_case",
                "title": "蓝屏",
                "problem_summary": "设备蓝屏",
                "device_scope": "设备A",
                "time_span": "",
                "source_message_ids": ["m1", "invented"],
                "attachment_message_ids": [],
                "jira_keys": [],
                "duplicate_report_message_ids": [],
                "requires_trace_assembly": True,
                "uncertainties": [],
            }],
            "unassigned_message_ids": [],
            "global_uncertainties": [],
        },
        allowed_message_ids={"m1"},
    )
    assert normalized["case_items"][0]["source_message_ids"] == ["m1"]
    assert "case_items[0]:unknown_message_id:invented" in issues


def test_neighbor_selection_requires_complete_exclusive_accounting():
    normalized, issues = validate_neighbor_selection(
        {
            "selected_links": [{
                "neighbor_case_item_ref": "n1",
                "related_core_case_item_refs": ["c1"],
                "reasons": ["同一设备同一蓝屏问题后续复发"],
            }],
            "excluded_neighbor_case_item_refs": ["n2"],
            "global_uncertainties": [],
        },
        allowed_core_refs={"c1"},
        allowed_neighbor_refs={"n1", "n2", "n3"},
    )
    assert normalized["selected_links"][0]["neighbor_case_item_ref"] == "n1"
    assert "neighbor_ref_unaccounted:n3" in issues


def test_neighbor_selection_accepts_auditable_multihop_path_to_core():
    _, issues = validate_neighbor_selection(
        {
            "selected_links": [
                {
                    "neighbor_case_item_ref": "n1",
                    "related_core_case_item_refs": ["c1"],
                    "reasons": ["首报回连核心复发"],
                },
                {
                    "neighbor_case_item_ref": "n2",
                    "related_core_case_item_refs": ["n1"],
                    "reasons": ["同一首报的诊断延续"],
                },
            ],
            "excluded_neighbor_case_item_refs": ["n3"],
            "global_uncertainties": [],
        },
        allowed_core_refs={"c1"},
        allowed_neighbor_refs={"n1", "n2", "n3"},
    )
    assert issues == []


def test_assembly_rejects_verified_without_explicit_success():
    decomposition = {
        "case_items": [{
            "case_item_ref": "c1",
            "case_kind": "diagnostic_case",
            "requires_trace_assembly": True,
            "source_message_ids": ["m1"],
        }]
    }
    ledger = {
        "allowed_message_ids": ["m1"],
        "rows": [{"message_id": "m1", "text": "建议后续持续观察"}],
    }
    normalized, issues = validate_assembly(
        {
            "traces": [{
                "trace_ref": "t1",
                "title": "蓝屏",
                "device_scope": "设备A",
                "case_item_refs": ["c1"],
                "phases": [{
                    "phase_index": 1,
                    "case_item_ref": "c1",
                    "event_type": "validation",
                    "relation_type": "trace_root",
                    "summary": "观察",
                    "evidence_message_ids": ["m1"],
                }],
                "resolution_status": "verified",
                "resolution_evidence_message_ids": ["m1"],
                "link_reasons": [],
                "uncertainties": [],
            }],
            "standalone_case_item_refs": [],
            "cannot_link_pairs": [],
            "global_uncertainties": [],
        },
        decomposition=decomposition,
        ledger=ledger,
    )
    assert normalized["traces"]
    assert "traces[0]:verified_without_explicit_success_signal" in issues
    assert "traces[0]:verified_from_temporary_signal" in issues


def test_verified_uses_case_phase_summary_not_unrelated_daily_report_clause():
    decomposition = {
        "case_items": [{
            "case_item_ref": "c1",
            "case_kind": "configuration_issue",
            "requires_trace_assembly": True,
            "source_message_ids": ["m1"],
        }]
    }
    ledger = {
        "allowed_message_ids": ["m1"],
        "rows": [{
            "message_id": "m1",
            "text": (
                "1.复判站连接不上，打开主程序后解决；"
                "2.另一台设备待更换电池并持续观察"
            ),
        }],
    }
    _, issues = validate_assembly(
        {
            "traces": [{
                "trace_ref": "t1",
                "title": "复判站连接不上",
                "device_scope": "复判站",
                "case_item_refs": ["c1"],
                "phases": [{
                    "phase_index": 1,
                    "case_item_ref": "c1",
                    "event_type": "report",
                    "relation_type": "trace_root",
                    "summary": "主程序未打开，打开主程序后解决",
                    "evidence_message_ids": ["m1"],
                }],
                "resolution_status": "verified",
                "resolution_evidence_message_ids": ["m1"],
                "link_reasons": [],
                "uncertainties": [],
            }],
            "standalone_case_item_refs": [],
            "cannot_link_pairs": [],
            "global_uncertainties": [],
        },
        decomposition=decomposition,
        ledger=ledger,
    )
    assert not [issue for issue in issues if "verified_" in issue]


def test_run_harness_repairs_and_gates_with_mock_tool_caller(tmp_path: Path):
    ledger = {
        "source_thread_id": "legacy-thread",
        "ledger_sha256": "sha",
        "allowed_message_ids": ["m1", "m2"],
        "core_message_ids": ["m1", "m2"],
        "high_signal_core_message_ids": ["m1"],
        "orphan_attachment_candidate_message_ids": [],
        "rows": [
            {"message_id": "m1", "text": "设备蓝屏", "attachments": []},
            {"message_id": "m2", "text": "更换内存后正常生产", "attachments": []},
        ],
    }
    calls = []

    def caller(**kwargs):
        name = kwargs["tool"]["function"]["name"]
        calls.append(name)
        if name == "decompose_chat_into_atomic_case_items":
            arguments = {
                "case_items": [{
                    "case_item_ref": "c1",
                    "case_kind": "diagnostic_case",
                    "title": "蓝屏",
                    "problem_summary": "设备蓝屏后更换内存",
                    "device_scope": "设备A",
                    "time_span": "",
                    "source_message_ids": ["m1", "m2"],
                    "attachment_message_ids": [],
                    "jira_keys": [],
                    "duplicate_report_message_ids": [],
                    "requires_trace_assembly": True,
                    "uncertainties": [],
                }],
                "unassigned_message_ids": [],
                "global_uncertainties": [],
            }
        else:
            arguments = {
                "traces": [{
                    "trace_ref": "t1",
                    "title": "蓝屏",
                    "device_scope": "设备A",
                    "case_item_refs": ["c1"],
                    "phases": [{
                        "phase_index": 1,
                        "case_item_ref": "c1",
                        "event_type": "resolution",
                        "relation_type": "trace_root",
                        "summary": "更换内存后恢复",
                        "evidence_message_ids": ["m1", "m2"],
                    }],
                    "resolution_status": "verified",
                    "resolution_evidence_message_ids": ["m2"],
                    "link_reasons": ["same_device_same_fault"],
                    "uncertainties": [],
                }],
                "standalone_case_item_refs": [],
                "cannot_link_pairs": [],
                "global_uncertainties": [],
            }
        return {
            "arguments": arguments,
            "model": "mock",
            "finish_reason": "tool_calls",
            "usage": {},
        }

    stage_cache_dir = tmp_path / "stage-cache"
    result = run_harness(
        ledger,
        api_key="fake",
        caller=caller,
        stage_cache_dir=stage_cache_dir,
    )
    assert calls == [
        "decompose_chat_into_atomic_case_items",
        "assemble_longitudinal_fault_traces",
    ]
    assert result["schema_valid"] is True
    assert result["promotion_allowed"] is False
    assert result["coverage"]["core_high_signal_coverage"] == 1.0

    calls.clear()
    cached_result = run_harness(
        ledger,
        api_key="fake",
        caller=caller,
        stage_cache_dir=stage_cache_dir,
    )
    assert calls == []
    assert cached_result["schema_valid"] is True
    assert cached_result["calls"]["decomposition"][0]["stage_cache_hit"] is True
    assert cached_result["calls"]["assembly"][0]["stage_cache_hit"] is True


def test_run_harness_scopes_neighbor_items_before_global_assembly():
    ledger = {
        "source_thread_id": "legacy-thread",
        "ledger_sha256": "sha",
        "allowed_message_ids": ["m1", "m2", "m3"],
        "core_message_ids": ["m1"],
        "high_signal_core_message_ids": ["m1"],
        "orphan_attachment_candidate_message_ids": [],
        "rows": [
            {"message_id": "m1", "text": "设备蓝屏", "attachments": []},
            {"message_id": "m2", "text": "次日同设备再次蓝屏", "attachments": []},
            {"message_id": "m3", "text": "另一设备相机不拍摄", "attachments": []},
        ],
    }
    seen_assembly_refs = []

    def case_item(ref, title, message_id):
        return {
            "case_item_ref": ref,
            "case_kind": "diagnostic_case",
            "title": title,
            "problem_summary": title,
            "device_scope": "设备A" if ref != "n2" else "设备B",
            "time_span": "",
            "source_message_ids": [message_id],
            "attachment_message_ids": [],
            "jira_keys": [],
            "duplicate_report_message_ids": [],
            "requires_trace_assembly": True,
            "uncertainties": [],
        }

    def caller(**kwargs):
        name = kwargs["tool"]["function"]["name"]
        if name == "decompose_chat_into_atomic_case_items":
            arguments = {
                "case_items": [
                    case_item("c1", "设备蓝屏", "m1"),
                    case_item("n1", "同设备蓝屏复发", "m2"),
                    case_item("n2", "另一设备相机不拍摄", "m3"),
                ],
                "unassigned_message_ids": [],
                "global_uncertainties": [],
            }
        elif name == "select_neighbor_cases_for_trace_assembly":
            arguments = {
                "selected_links": [{
                    "neighbor_case_item_ref": "n1",
                    "related_core_case_item_refs": ["c1"],
                    "reasons": ["同设备同故障复发"],
                }],
                "excluded_neighbor_case_item_refs": ["n2"],
                "global_uncertainties": [],
            }
        else:
            seen_assembly_refs.extend(
                item["case_item_ref"]
                for item in kwargs["user_payload"]["case_items"]
            )
            arguments = {
                "traces": [{
                    "trace_ref": "t1",
                    "title": "蓝屏复发",
                    "device_scope": "设备A",
                    "case_item_refs": ["c1", "n1"],
                    "phases": [
                        {
                            "phase_index": 1,
                            "case_item_ref": "c1",
                            "event_type": "report",
                            "relation_type": "trace_root",
                            "summary": "首次蓝屏",
                            "evidence_message_ids": ["m1"],
                        },
                        {
                            "phase_index": 2,
                            "case_item_ref": "n1",
                            "event_type": "recurrence",
                            "relation_type": "recurrence_of",
                            "summary": "次日复发",
                            "evidence_message_ids": ["m2"],
                        },
                    ],
                    "resolution_status": "recurrence",
                    "resolution_evidence_message_ids": [],
                    "link_reasons": ["同设备同故障"],
                    "uncertainties": [],
                }],
                "standalone_case_item_refs": [],
                "cannot_link_pairs": [],
                "global_uncertainties": [],
            }
        return {
            "arguments": arguments,
            "model": "mock",
            "finish_reason": "tool_calls",
            "usage": {},
        }

    result = run_harness(ledger, api_key="fake", caller=caller)
    assert result["schema_valid"] is True
    assert seen_assembly_refs == ["c1", "n1"]
    assert "n2" in result["assembly"]["standalone_case_item_refs"]
    assert result["coverage"]["unaccounted_case_item_refs"] == []


def test_coverage_reports_unassigned_high_signal():
    metrics = coverage_metrics(
        {
            "high_signal_core_message_ids": ["m1", "m2"],
            "orphan_attachment_candidate_message_ids": [],
        },
        {
            "case_items": [{
                "case_item_ref": "c1",
                "source_message_ids": ["m1"],
            }]
        },
        {"traces": [], "standalone_case_item_refs": ["c1"]},
    )
    assert metrics["core_high_signal_coverage"] == 0.5
    assert metrics["uncovered_high_signal_message_ids"] == ["m2"]
    assert metrics["core_high_signal_diagnostic_coverage"] == 0.0


def test_partition_source_ledger_is_lossless_and_non_overlapping():
    rows = [
        {
            "message_id": f"m{index}",
            "text": "故障" * 20,
            "attachments": [],
        }
        for index in range(5)
    ]
    ledger = {
        "source_thread_id": "legacy",
        "rows": rows,
        "allowed_message_ids": [row["message_id"] for row in rows],
        "core_message_ids": ["m0", "m1", "m2"],
        "high_signal_core_message_ids": ["m1"],
        "orphan_attachment_candidate_message_ids": [],
        "stats": {},
    }
    chunks = partition_source_ledger(ledger, max_rows=2, max_chars=100_000)
    assert len(chunks) == 3
    flattened = [
        row["message_id"]
        for chunk in chunks
        for row in chunk["rows"]
    ]
    assert flattened == ["m0", "m1", "m2", "m3", "m4"]
    assert chunks[0]["high_signal_core_message_ids"] == ["m1"]
    assert chunks[2]["core_message_ids"] == []


def test_screen_term_conflict_becomes_adjudication_candidate():
    decomposition = {
        "case_items": [
            {
                "case_item_ref": "C01-a",
                "case_kind": "diagnostic_case",
                "title": "复判界面花屏，重启后正常",
                "problem_summary": "现场花屏，重启后恢复",
                "device_scope": "6线炉后",
                "source_message_ids": ["m1"],
            },
            {
                "case_item_ref": "C02-b",
                "case_kind": "diagnostic_case",
                "title": "复判界面蓝屏，重启后解决",
                "problem_summary": "日报复述蓝屏，重启后解决",
                "device_scope": "复判站",
                "source_message_ids": ["m2"],
            },
        ]
    }
    ledger = {
        "rows": [
            {"message_id": "m1", "create_time": "2026-03-23 11:20"},
            {"message_id": "m2", "create_time": "2026-03-23 20:19"},
        ]
    }
    candidates = build_link_candidates(decomposition, ledger)
    assert candidates == [{
        "left_case_item_ref": "C01-a",
        "right_case_item_ref": "C02-b",
        "reasons": ["same_day_screen_term_conflict_with_same_recovery"],
    }]


def test_partition_assembly_components_keeps_linked_chain_and_batches_isolates():
    items = [{"case_item_ref": ref} for ref in ("c1", "n1", "n2", "c2", "c3")]
    components = partition_assembly_components(
        items,
        neighbor_selection={
            "selected_links": [
                {
                    "neighbor_case_item_ref": "n1",
                    "related_core_case_item_refs": ["c1"],
                },
                {
                    "neighbor_case_item_ref": "n2",
                    "related_core_case_item_refs": ["n1"],
                },
            ]
        },
        link_candidates=[],
    )
    assert [
        [item["case_item_ref"] for item in component]
        for component in components
    ] == [["c1", "n1", "n2"], ["c2", "c3"]]


def test_decomposition_adaptively_splits_persistently_failing_chunk():
    rows = [
        {"message_id": f"m{index}", "text": f"故障{index}", "attachments": []}
        for index in range(4)
    ]
    chunk = {
        "source_thread_id": "legacy",
        "rows": rows,
        "allowed_message_ids": [row["message_id"] for row in rows],
        "core_message_ids": [row["message_id"] for row in rows],
        "high_signal_core_message_ids": [row["message_id"] for row in rows],
        "orphan_attachment_candidate_message_ids": [],
    }
    requested_sizes = []

    def caller(**kwargs):
        source_rows = kwargs["user_payload"]["source_ledger"]["rows"]
        requested_sizes.append(len(source_rows))
        if len(source_rows) > 2:
            raise RuntimeError("synthetic_tool_arguments_truncated")
        ids = [row["message_id"] for row in source_rows]
        return {
            "arguments": {
                "case_items": [{
                    "case_item_ref": "ci",
                    "case_kind": "diagnostic_case",
                    "title": "故障",
                    "problem_summary": "故障",
                    "device_scope": "",
                    "time_span": "",
                    "source_message_ids": ids,
                    "attachment_message_ids": [],
                    "jira_keys": [],
                    "duplicate_report_message_ids": [],
                    "requires_trace_assembly": True,
                    "uncertainties": [],
                }],
                "unassigned_message_ids": [],
                "global_uncertainties": [],
            },
            "model": "mock",
            "finish_reason": "tool_calls",
            "usage": {},
        }

    _, normalized, issues, _ = _decompose_with_adaptive_split(
        chunk=chunk,
        caller=caller,
        api_key="fake",
        user_id="test",
        minimum_rows=2,
    )
    assert requested_sizes == [4, 2, 2]
    assert issues == []
    assert [item["case_item_ref"] for item in normalized["case_items"]] == [
        "A-ci",
        "B-ci",
    ]
