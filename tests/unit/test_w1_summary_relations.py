import unittest

from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent


def msg(message_id: str, thread_id: str, time: str, text: str) -> dict:
    return {
        "message_id": message_id,
        "thread_id": thread_id,
        "chat_id": "oc_test",
        "sender": {"id": "", "name": "FAE", "type": "user"},
        "create_time": time,
        "msg_type": "text",
        "text": text,
        "mentions": [],
        "attachments": [],
        "links": [],
        "raw": {"chat_name": "客户03项目群"},
    }


class W1SummaryRelationTests(unittest.TestCase):
    def test_normalize_recovers_embedded_file_and_jira_metadata(self):
        row = msg(
            "m-resource",
            "thread-resource",
            "2025-09-25 17:04",
            "@工程师午 已上传相关数据",
        )
        row["raw"] = {
            "raw_content": '<file key="k1" name="case.dmp"/> https://jira.example.com/browse/TEST-1',
            "chat_name": "客户03项目群",
        }
        normalized = ChatCollectAgent().normalize_messages([row])[0]
        self.assertEqual(normalized["attachments"][0]["name"], "case.dmp")
        self.assertEqual(normalized["attachments"][0]["status"], "metadata_only")
        self.assertEqual(normalized["links"][0]["type"], "jira")

    def test_same_day_summary_links_prior_fault_session(self):
        previous = msg(
            "m-prev",
            "thread-prev",
            "2025-09-25 09:21",
            "更新0.27.30验证：OCR识别为空时提示框缺失，LED连锡漏检并出现误报。",
        )
        summary = msg(
            "m-summary",
            "thread-summary",
            "2025-09-25 14:41",
            "20250925汇总 一，0.27.30验证OCR提示框缺失；二，LED连锡误报和漏检数据已采集。",
        )
        rows = ChatCollectAgent().aggregate_threads([previous, summary])
        current = next(row for row in rows if row["thread_id"] == "thread-summary")
        relations = current.get("summary_relations") or []
        self.assertEqual([row["relation"] for row in relations], ["summary_of"])
        self.assertEqual(relations[0]["target_thread_id"], "thread-prev")
        self.assertEqual(current["summary_context_message_ids"], ["m-prev"])
        episode = current["episodes"][0]
        self.assertEqual(episode["summary_context_message_ids"], ["m-prev"])
        self.assertEqual(episode["message_ids"], ["m-summary"])
        self.assertEqual(episode["full_context_message_ids"], ["m-summary", "m-prev"])
        self.assertEqual(episode["message_refs"]["summary_context_message_ids"], ["m-prev"])

    def test_summary_does_not_link_unrelated_prior_session(self):
        previous = msg("m-prev", "thread-prev", "2025-09-25 09:21", "客户培训和发货安排。")
        summary = msg(
            "m-summary",
            "thread-summary",
            "2025-09-25 14:41",
            "今日汇总：OCR识别为空时提示框缺失，LED连锡漏检。",
        )
        rows = ChatCollectAgent().aggregate_threads([previous, summary])
        current = next(row for row in rows if row["thread_id"] == "thread-summary")
        self.assertEqual(current.get("summary_relations") or [], [])

    def test_summary_relation_is_filtered_per_case_item(self):
        previous = msg(
            "m-prev",
            "thread-prev",
            "2025-09-25 09:21",
            "相机拍摄失败，检查网卡和排线。",
        )
        summary = msg(
            "m-summary",
            "thread-summary",
            "2025-09-25 14:41",
            "今日工作汇总 一、相机拍摄失败，检查网卡排线；二、收板机远轨进板卡顿，喇叭口一边宽一边窄。",
        )
        rows = ChatCollectAgent().aggregate_threads([previous, summary])
        current = next(row for row in rows if row["thread_id"] == "thread-summary")
        self.assertEqual(len(current["episodes"]), 2)
        camera, transport = current["episodes"]
        self.assertEqual(camera["summary_context_message_ids"], ["m-prev"])
        self.assertEqual(transport["summary_context_message_ids"], [])
        self.assertEqual(transport.get("summary_relations") or [], [])

    def test_sequential_distinct_faults_are_split_and_long_gap_context_is_not_leaked(self):
        rows = [
            msg("m1", "thread-mixed", "2025-09-26 03:11", "双轨异步测试不出检测结果，关闭复判后一直暂停，无法开启测试。"),
            msg("m2", "thread-mixed", "2025-09-26 03:13", "升级0.27.31后暂时正常。"),
            msg("m3", "thread-mixed", "2025-09-26 03:24", "0.27.31导出中文程序后显示乱码，再导入提示失败。"),
            msg("m4", "thread-mixed", "2025-09-30 02:48", "V0.27.32版本已修复，可升级验证。"),
            msg("m5", "thread-mixed", "2025-10-11 17:14", "又压排线了吗？"),
            msg("m6", "thread-mixed", "2025-10-11 17:51", "网卡排线有明显压痕，频繁报拍摄失败。"),
        ]
        summary = ChatCollectAgent().aggregate_threads(rows)[0]
        episodes = summary["episodes"]
        self.assertGreaterEqual(len(episodes), 3)
        texts = [" ".join(item["text"] for item in episode["case_context_messages"]) for episode in episodes]
        self.assertTrue(any("双轨异步" in text and "导出中文程序" not in text for text in texts))
        self.assertTrue(any("导出中文程序" in text and "双轨异步" not in text for text in texts))
        camera = next(text for text in texts if "拍摄失败" in text)
        self.assertNotIn("导出中文程序", camera)

    def test_observed_results_are_outcomes_not_noise(self):
        rows = [
            msg("m1", "thread-outcome", "2025-10-12 12:35", "更换网卡测试后还是会偶发请求超时。"),
            msg("m2", "thread-outcome", "2025-10-12 17:01", "更新0.27.33版本后测试验证可以正常导出。"),
            msg("m3", "thread-outcome", "2025-10-12 19:45", "重新安装网卡并调整排线后可以正常拍照，未再出现异常。"),
        ]
        summary = ChatCollectAgent().aggregate_threads(rows)[0]
        outcomes = {
            item["message_id"]
            for episode in summary["episodes"]
            for item in episode["resolution_messages"]
        }
        noise = {
            item["message_id"]
            for episode in summary["episodes"]
            for item in episode["noise_messages"]
        }
        self.assertEqual(outcomes, {"m1", "m2", "m3"})
        self.assertTrue(outcomes.isdisjoint(noise))


if __name__ == "__main__":
    unittest.main()
