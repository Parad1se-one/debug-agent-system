import unittest

from debug_agent_system.agents.write.review_context import refine_episode_for_w2, refine_episode_group


def message(message_id: str, text: str) -> dict:
    return {
        "message_id": message_id,
        "source_message_id": message_id,
        "create_time": "2026-01-01 10:00",
        "text": text,
        "content_summary": text,
        "msg_type": "text",
    }


class W7EpisodeCleanupTests(unittest.TestCase):
    def test_question_is_not_verified_resolution(self):
        episode = {
            "episode_id": "ep-question",
            "thread_id": "thread-question",
            "completeness": "complete",
            "fault_description_messages": [message("m1", "设备蓝屏自动重启")],
            "diagnostic_chain_messages": [],
            "resolution_messages": [message("m2", "这个问题解决了吗？")],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["resolution_messages"], [])
        self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "pending")
        self.assertIn("m2", cleaned["extracted"]["w7_episode_cleanup"]["rejected_resolution_message_ids"])

    def test_verified_fix_requires_action_or_observation(self):
        episode = {
            "episode_id": "ep-fixed",
            "thread_id": "thread-fixed",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "设备蓝屏自动重启")],
            "diagnostic_chain_messages": [message("m2", "更换内存条")],
            "resolution_messages": [message("m3", "更换内存条后连续运行2小时未再出现蓝屏")],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 13:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual([m["message_id"] for m in cleaned["resolution_messages"]], ["m3"])
        self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "verified")
        self.assertTrue(cleaned["w2_ready"])

    def test_short_confirmation_can_follow_an_action(self):
        episode = {
            "episode_id": "ep-confirmed",
            "thread_id": "thread-confirmed",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "键盘数字键失灵")],
            "diagnostic_chain_messages": [message("m2", "重启设备并拔插键鼠USB接口")],
            "resolution_messages": [message("m3", "已解决，谢谢")],
            "noise_messages": [],
            "case_context_messages": [message("m4", "其他群聊上下文包含蓝屏和检查动作")],
            "evidence_message_ids": ["m1", "m2", "m3", "m4"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "verified")
        self.assertTrue(cleaned["w2_ready"])

    def test_embedded_action_and_result_is_detected(self):
        text = "板卡测试报错，经排查是一轨皮带跑偏，把张紧轮下降一个孔位后正常"
        episode = {
            "episode_id": "ep-embedded",
            "thread_id": "thread-embedded",
            "completeness": "partial",
            "fault_description_messages": [message("m1", text)],
            "diagnostic_chain_messages": [message("m1", text)],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "verified")
        self.assertEqual([row["message_id"] for row in cleaned["resolution_messages"]], ["m1"])

    def test_multi_fault_report_is_blocked_until_split(self):
        episode = {
            "episode_id": "ep-multi",
            "thread_id": "thread-multi",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "1、相机拍摄失败。2、设备蓝屏自动重启。")],
            "diagnostic_chain_messages": [message("m2", "检查相机驱动并收集蓝屏日志")],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["episode_scope"], "multi_fault")
        self.assertIn("multi_fault_requires_case_item_iteration", cleaned["w2_block_reasons"])
        self.assertEqual(cleaned["case_item_count"], 2)
        self.assertEqual([item["fault_focus"] for item in cleaned["case_items"]], ["相机拍摄失败", "设备蓝屏自动重启"])
        self.assertTrue(all("message_ids" in item for item in cleaned["case_items"]))

    def test_case_context_cannot_make_isolated_question_ready(self):
        episode = {
            "episode_id": "ep-context-leak",
            "thread_id": "thread-context-leak",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "@工程师 这个数据在哪里？")],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "case_context_messages": [message("m2", "之前相机拍摄失败，检查驱动后恢复正常")],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertFalse(cleaned["w2_ready"])
        self.assertIn("missing_fault_signal", cleaned["w2_block_reasons"])

    def test_long_session_gets_trace_metadata(self):
        base = {
            "thread_id": "thread-long",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "拍摄失败")],
            "diagnostic_chain_messages": [message("m2", "检查相机驱动")],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-10 10:00",
            "extracted": {},
        }
        episodes = [{**base, "episode_id": "ep1"}, {**base, "episode_id": "ep2"}]
        refined = refine_episode_group(episodes)
        self.assertEqual(len({item["trace_group_id"] for item in refined}), 1)
        self.assertTrue(all(item["continuation"] for item in refined))
        self.assertEqual([item["trace_phase_index"] for item in refined], [1, 2])
        self.assertEqual([item["trace_relation"] for item in refined], ["trace_root", "same_trace"])
        self.assertEqual([item["trace_link_strength"] for item in refined], ["root", "hard"])

    def test_distinct_faults_get_independent_trace_groups(self):
        camera = {
            "episode_id": "ep-camera",
            "thread_id": "thread-mixed",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "相机网卡断连导致拍摄失败")],
            "diagnostic_chain_messages": [message("m2", "检查网卡排线")],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        export = {
            **camera,
            "episode_id": "ep-export",
            "fault_description_messages": [message("m3", "中文程序导出后显示乱码")],
            "diagnostic_chain_messages": [message("m4", "升级版本验证导出")],
            "evidence_message_ids": ["m3", "m4"],
        }
        refined = refine_episode_group([camera, export])
        self.assertNotEqual(refined[0]["trace_group_id"], refined[1]["trace_group_id"])
        self.assertEqual([item["trace_relation"] for item in refined], ["trace_root", "trace_root"])

    def test_exact_attachment_payload_links_same_fault_across_sessions(self):
        chat_id = "oc_11111111111111111111111111111111"
        shared = {"file_key": "file_v3_00abcdef1234567890", "name": "诊断数据.zip"}
        first_fault = message("m1", "设备蓝屏")
        first_fault["attachments"] = [shared]
        second_fault = message("m2", "设备蓝屏再次出现")
        second_fault["attachments"] = [shared]
        first = {
            "episode_id": "ep-payload-1",
            "thread_id": f"{chat_id}:session:1",
            "chat_id": chat_id,
            "completeness": "partial",
            "fault_description_messages": [first_fault],
            "diagnostic_chain_messages": [message("m1-action", "收集蓝屏日志")],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m1-action"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        second = {
            **first,
            "episode_id": "ep-payload-2",
            "thread_id": f"{chat_id}:session:2",
            "fault_description_messages": [second_fault],
            "diagnostic_chain_messages": [message("m2-action", "继续检查蓝屏日志")],
            "evidence_message_ids": ["m2", "m2-action"],
            "start_time": "2026-01-10 10:00",
            "end_time": "2026-01-10 11:00",
        }

        refined = refine_episode_group([first, second])

        self.assertEqual(refined[0]["trace_group_id"], refined[1]["trace_group_id"])
        self.assertEqual(refined[1]["trace_link_strength"], "strong")
        self.assertIn("shared_artifact_payload", refined[1]["trace_link_reasons"])

    def test_same_session_exact_compact_fault_focus_is_medium_link(self):
        first = {
            "episode_id": "ep-black-screen-1",
            "thread_id": "thread-black-screen",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "设备频繁黑屏闪烁")],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 10:10",
            "extracted": {"fault_focus_text": "设备频繁黑屏闪烁"},
        }
        second = {
            **first,
            "episode_id": "ep-black-screen-2",
            "fault_description_messages": [message("m2", "另一台设备也频繁黑屏闪烁")],
            "evidence_message_ids": ["m2"],
            "start_time": "2026-01-01 10:20",
            "end_time": "2026-01-01 10:30",
            "extracted": {"fault_focus_text": "另一台设备也频繁黑屏闪烁"},
        }

        refined = refine_episode_group([first, second])

        self.assertEqual(refined[0]["trace_group_id"], refined[1]["trace_group_id"])
        self.assertEqual(refined[1]["trace_link_strength"], "medium")
        self.assertIn(
            "same_session_exact_focus_with_text_overlap",
            refined[1]["trace_link_reasons"],
        )

    def test_generic_focus_does_not_merge_from_context_signature_alone(self):
        first = {
            "episode_id": "ep-camera-1",
            "thread_id": "thread-camera",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "相机拍摄失败")],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1"],
            "extracted": {"fault_focus_text": "相机拍摄失败"},
        }
        second = {
            **first,
            "episode_id": "ep-camera-2",
            "fault_description_messages": [message("m2", "拿到别的设备也是同样报错")],
            "diagnostic_chain_messages": [message("m3", "继续排查相机拍摄失败日志")],
            "evidence_message_ids": ["m2", "m3"],
            "extracted": {"fault_focus_text": "拿到别的设备也是同样报错"},
        }

        refined = refine_episode_group([first, second])

        self.assertNotEqual(refined[0]["trace_group_id"], refined[1]["trace_group_id"])
        self.assertFalse(refined[1]["trace_link_candidates"][0]["linked"])

    def test_broad_exact_focus_stays_weak_without_independent_identity(self):
        focus = "相机到第一个FOV后不拍照并报拍摄失败"
        first = {
            "episode_id": "ep-broad-1",
            "thread_id": "thread-broad",
            "completeness": "partial",
            "fault_description_messages": [message("m1", focus)],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1"],
            "extracted": {"fault_focus_text": focus},
        }
        second = {
            **first,
            "episode_id": "ep-broad-2",
            "fault_description_messages": [message("m2", focus)],
            "evidence_message_ids": ["m2"],
        }

        refined = refine_episode_group([first, second])

        self.assertNotEqual(refined[0]["trace_group_id"], refined[1]["trace_group_id"])
        self.assertEqual(refined[1]["trace_link_candidates"][0]["link_strength"], "weak")

    def test_numbered_procedure_is_not_split_into_fault_cases(self):
        procedure = "排查拍摄失败：1、打开设备管理器；2、卸载网卡驱动；3、重新安装驱动。"
        episode = {
            "episode_id": "ep-procedure",
            "thread_id": "thread-procedure",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "相机拍摄失败")],
            "diagnostic_chain_messages": [message("m2", procedure)],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["case_item_count"], 1)
        self.assertEqual(cleaned["case_items"][0]["fault_focus"], "相机拍摄失败")

    def test_numbered_field_summary_splits_distinct_faults(self):
        report = (
            "20250925汇总 一，更新0.27.30验证OCR识别为空时提示框缺失；"
            "二，LED模型连锡误报或者漏检；三，客户培训；"
            "四，LED连锡误报得分过高。"
        )
        episode = {
            "episode_id": "ep-summary",
            "thread_id": "thread-summary",
            "completeness": "partial",
            "fault_description_messages": [message("m1", report)],
            "diagnostic_chain_messages": [message("m2", "收集OCR与LED样本并建立Jira")],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "summary_context_messages": [
                message("m-ocr", "OCR识别为空时无法显示提示框，已收集对应图片"),
                message("m-led", "LED连锡漏检，已提交样本到Jira"),
            ],
            "summary_context_message_ids": ["m-ocr", "m-led"],
            "start_time": "2025-09-25 14:41",
            "end_time": "2025-09-25 14:41",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["episode_scope"], "multi_fault")
        self.assertEqual(cleaned["case_item_count"], 2)
        self.assertIn("OCR识别为空时提示框缺失", cleaned["case_items"][0]["problem_statement"])
        self.assertIn("LED模型连锡误报或者漏检", cleaned["case_items"][1]["problem_statement"])
        self.assertIn("m-ocr", cleaned["case_items"][0]["context_message_ids"])
        self.assertNotIn("m-led", cleaned["case_items"][0]["context_message_ids"])
        self.assertIn("m-led", cleaned["case_items"][1]["context_message_ids"])
        self.assertNotIn("m-ocr", cleaned["case_items"][1]["context_message_ids"])

    def test_report_only_parent_blocks_case_item(self):
        episode = {
            "episode_id": "ep-report-only",
            "thread_id": "thread-report-only",
            "completeness": "partial",
            "field_report_anchor": {"anchor_id": "field-report:m1", "issue_count": 0},
            "fault_description_messages": [],
            "diagnostic_chain_messages": [message("m1", "今日现场工作汇报，客户培训和设备交付安排")],
            "resolution_messages": [],
            "noise_messages": [],
            "evidence_message_ids": ["m1"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertFalse(cleaned["case_items"][0]["w2_ready"])
        self.assertIn("parent_episode_report_only", cleaned["case_items"][0]["w2_block_reasons"])

    def test_restart_workaround_and_short_observation_are_pending(self):
        for text in (
            "任务管理器强制重启后正常，可升级版本继续验证",
            "调整排线后生产四十分钟正常没有报拍摄失败",
        ):
            episode = {
                "episode_id": f"ep-{len(text)}",
                "thread_id": "thread-pending",
                "completeness": "partial",
                "fault_description_messages": [message("m1", "设备拍摄失败")],
                "diagnostic_chain_messages": [message("m2", text)],
                "resolution_messages": [message("m2", text)],
                "noise_messages": [],
                "evidence_message_ids": ["m1", "m2"],
                "start_time": "2026-01-01 10:00",
                "end_time": "2026-01-01 11:00",
                "extracted": {},
            }
            cleaned = refine_episode_for_w2(episode)
            self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "pending")

    def test_action_followed_by_normal_operation_is_verified(self):
        text = "重新安装网卡并调整排线位置后可以正常拍照，无异常情况出现"
        episode = {
            "episode_id": "ep-normal-operation",
            "thread_id": "thread-normal-operation",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "相机请求超时并频繁拍摄失败")],
            "diagnostic_chain_messages": [message("m2", text)],
            "resolution_messages": [message("m2", text)],
            "noise_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-02 10:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "verified")

    def test_outcome_statement_is_salvaged_from_noise(self):
        ineffective = message("m2", "更换网卡后还是会偶发请求超时")
        episode = {
            "episode_id": "ep-outcome-noise",
            "thread_id": "thread-outcome-noise",
            "completeness": "partial",
            "fault_description_messages": [message("m1", "相机请求超时并拍摄失败")],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [ineffective],
            "case_context_messages": [ineffective],
            "evidence_message_ids": ["m1", "m2"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertEqual([row["message_id"] for row in cleaned["outcome_messages"]], ["m2"])
        self.assertEqual(cleaned["noise_messages"], [])
        self.assertEqual(cleaned["extracted"]["w7_episode_cleanup"]["resolution_status"], "ineffective")

    def test_report_only_salvages_embedded_fault_case(self):
        report = message(
            "m1",
            "今日工作汇报。客户培训和设备接线。收板机远轨进框会出现卡顿，排查发现轨道喇叭口一边宽一边窄，已告知客户安排人员处理。",
        )
        episode = {
            "episode_id": "ep-report-salvage",
            "thread_id": "thread-report-salvage",
            "completeness": "partial",
            "field_report_anchor": {"anchor_id": "field-report:m1", "issue_count": 0},
            "fault_description_messages": [],
            "diagnostic_chain_messages": [report],
            "resolution_messages": [],
            "noise_messages": [],
            "case_context_messages": [report],
            "evidence_message_ids": ["m1"],
            "start_time": "2026-01-01 10:00",
            "end_time": "2026-01-01 11:00",
            "extracted": {},
        }
        cleaned = refine_episode_for_w2(episode)
        self.assertNotEqual(cleaned["episode_scope"], "report_only")
        self.assertEqual(cleaned["case_item_count"], 1)
        self.assertIn("收板机远轨进框会出现卡顿", cleaned["case_items"][0]["problem_statement"])
        self.assertTrue(cleaned["case_items"][0]["w2_ready"])
        self.assertTrue(cleaned["extracted"]["w7_episode_cleanup"]["report_case_salvaged"])


if __name__ == "__main__":
    unittest.main()
