from __future__ import annotations

import unittest

from debug_agent_system.agents.write.review_context import refine_episode_for_w2, refine_episode_group
from debug_agent_system.agents.write.w1_message_relations import assign_reference_aware_segments
from debug_agent_system.eval.write_side.gold_001_020_adapter import (
    CANONICAL_OUTCOMES,
    adapter_summary,
    load_gold_001_020,
)
from debug_agent_system.eval.write_side.w1_w7_gold_benchmark import score_cases, semantic_regression


def _message(message_id: str, text: str, *, time: str = "2026-01-01 10:00", **extra: str) -> dict:
    return {
        "message_id": message_id,
        "source_message_id": message_id,
        "chat_id": "oc_0123456789abcdef0123456789abcdef",
        "create_time": time,
        "text": text,
        "content_summary": text,
        "attachments": [],
        **extra,
    }


def _episode(episode_id: str, thread_id: str, fault: str, action: str) -> dict:
    return {
        "episode_id": episode_id,
        "thread_id": thread_id,
        "chat_id": "oc_0123456789abcdef0123456789abcdef",
        "completeness": "partial",
        "fault_description_messages": [_message(f"{episode_id}-fault", fault)],
        "diagnostic_chain_messages": [_message(f"{episode_id}-action", action)],
        "resolution_messages": [],
        "noise_messages": [],
        "evidence_message_ids": [f"{episode_id}-fault", f"{episode_id}-action"],
        "start_time": "2026-01-01 10:00",
        "end_time": "2026-01-01 11:00",
        "extracted": {},
    }


class GoldAdapterTests(unittest.TestCase):
    def test_loads_all_frozen_schemas_without_writing_sources(self) -> None:
        cases_before = load_gold_001_020()
        hashes_before = {case["case_id"]: case["source"]["truth_sha256"] for case in cases_before}
        cases_after = load_gold_001_020()
        hashes_after = {case["case_id"]: case["source"]["truth_sha256"] for case in cases_after}

        self.assertEqual(hashes_before, hashes_after)
        self.assertEqual(adapter_summary(cases_after)["case_count"], 20)
        self.assertEqual(adapter_summary(cases_after)["trace_count"], 35)
        self.assertEqual(adapter_summary(cases_after)["action_count"], 274)
        self.assertTrue(semantic_regression(cases_after)["passed"])
        self.assertTrue(all(case["graph_ingestion"] is False for case in cases_after))
        self.assertTrue(all(
            action["outcome"]["outcome_type"] in CANONICAL_OUTCOMES
            for case in cases_after for trace in case["traces"] for action in trace["actions"]
        ))

    def test_narrative_outcome_columns_in_019_020_are_recovered_auditably(self) -> None:
        cases = {case["case_id"]: case for case in load_gold_001_020()}
        reconnect = next(
            action
            for trace in cases["goldcase-020"]["traces"]
            for action in trace["actions"]
            if action["label"] == "拔插运控网线"
        )
        driver = next(
            action
            for trace in cases["goldcase-019"]["traces"]
            for action in trace["actions"]
            if "重装 Intel/NVIDIA" in action["label"]
        )
        self.assertEqual(reconnect["outcome"]["outcome_type"], "partial_temporary")
        self.assertEqual(driver["outcome"]["outcome_type"], "partial_temporary")
        self.assertIn("recovered_from_assessment", reconnect["outcome"]["normalization_reason"])


class W1SoftDecoderTests(unittest.TestCase):
    def test_inferred_edge_cannot_override_conflicting_equipment_identity(self) -> None:
        rows = [
            _message("m1", "AOI-101 相机拍摄失败"),
            _message("m1-child", "AOI-101 补充相机日志", time="2026-01-01 10:00", parent_id="m1", root_id="m1"),
            _message("m2", "AOI-202 工控机蓝屏", time="2026-01-01 10:01"),
            _message("m2-child", "AOI-202 补充蓝屏DMP", time="2026-01-01 10:01", parent_id="m2", root_id="m2"),
        ]
        edge = {"source": "m2", "target": "m1", "chat_id": rows[0]["chat_id"], "type": "context_continuation", "inferred": True, "score": 8}

        segmented, report = assign_reference_aware_segments(rows, context_edges=[edge])

        sessions = {row["message_id"]: row["thread_id"] for row in segmented}
        self.assertNotEqual(sessions["m1"], sessions["m2"])
        self.assertEqual(report["soft_context_edges_rejected"], 1)
        self.assertEqual(report["soft_context_rejection_counts"]["conflicting_equipment_identity"], 1)

    def test_native_reply_remains_hard_despite_identity_conflict(self) -> None:
        rows = [
            _message("m1", "AOI-1 相机拍摄失败"),
            _message("m2", "AOI-2 补充回复", time="2026-01-01 10:01", parent_id="m1", root_id="m1"),
        ]

        segmented, report = assign_reference_aware_segments(rows)

        self.assertEqual(segmented[0]["thread_id"], segmented[1]["thread_id"])
        self.assertTrue(report["native_edges_are_hard"])
        self.assertTrue(report["inferred_edges_are_soft"])


class W7StateAndLinkTests(unittest.TestCase):
    def test_temporary_recovery_then_recurrence_is_not_verified(self) -> None:
        episode = _episode("ep", "session-1", "运控初始化失败", "拔插网线后能进入主程序")
        episode["diagnostic_chain_messages"].append(_message("ep-recur", "重启后再次报运动控制初始化失败", time="2026-01-01 11:00"))
        episode["evidence_message_ids"].append("ep-recur")

        refined = refine_episode_for_w2(episode)
        cleanup = refined["extracted"]["w7_episode_cleanup"]

        self.assertEqual(cleanup["outcome_type"], "partial_temporary")
        self.assertEqual(cleanup["resolution_status"], "pending")
        self.assertFalse(cleanup["action_outcome_state"]["verified_fix_requirements"]["validation_or_observation"])

    def test_cross_session_same_device_jira_trace_is_linked_with_relation_type(self) -> None:
        left = _episode("ep-left", "session-left", "AOI-3 相机拍摄失败 SMTAOITS-1234", "收集日志")
        right = _episode("ep-right", "session-right", "AOI-3 相机拍摄失败再次出现 SMTAOITS-1234", "继续排查")

        refined = refine_episode_group([left, right])

        self.assertEqual(len({item["trace_group_id"] for item in refined}), 1)
        self.assertEqual(refined[1]["trace_relation_type"], "recurrence_of")
        self.assertIn("shared_jira", refined[1]["trace_link_reasons"])


class BenchmarkMetricTests(unittest.TestCase):
    def test_perfect_prediction_exercises_all_named_metrics(self) -> None:
        trace = {
            "trace_id": "t1",
            "evidence": {"message_ids": ["m1", "m2"]},
            "actions": [{
                "action_id": "a1",
                "label": "更换网卡",
                "action_role": "change",
                "outcome": {"outcome_type": "verified_fix"},
            }],
        }
        case = {"case_id": "goldcase-test", "trace_count": 1, "traces": [trace]}
        prediction = {
            "case_id": "goldcase-test",
            "w7_trace_groups": [{
                "trace_group_id": "p1",
                "message_ids": ["m1", "m2"],
                "actions": [{"label": "更换网卡", "action_role": "change", "outcome_type": "verified_fix"}],
                "outcome_type": "verified_fix",
            }],
        }

        summary = score_cases([case], [prediction], stage="w7")["summary"]

        self.assertEqual(summary["anchor_pair_recall"], 1.0)
        self.assertEqual(summary["best_cluster_trace_coverage"], 1.0)
        self.assertEqual(summary["same_trace_precision"], 1.0)
        self.assertEqual(summary["cross_trace_contamination"], 0.0)
        self.assertEqual(summary["cannot_link_violation_rate"], 0.0)
        self.assertEqual(summary["false_verified_fix_count"], 0)
        self.assertEqual(summary["action_role_macro_f1"], 1.0)
        self.assertEqual(summary["outcome_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
