from __future__ import annotations

import json

from debug_agent_system.agents.write import review_context as rc


def _episode(*, fault: str, diags: list[str], extracted: dict | None = None) -> dict:
    return {
        "episode_id": "ep:test",
        "thread_id": "thread:test",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": fault}],
        "diagnostic_chain_messages": [{"message_id": f"m{idx+2}", "text": text} for idx, text in enumerate(diags)],
        "resolution_messages": [],
        "noise_messages": [],
        "case_context_messages": [],
        "attachments": [],
        "extracted": extracted or {},
    }


def test_review_ready_episode_rejects_report_like_fault_message():
    episode = _episode(
        fault="今日工作汇总如下：问题点 1.现场测试时整板异物检测误报飞件，未报脏污。",
        diags=["新板编程完，测试时弹出图片为空，已搜集日志"],
    )
    assert rc.is_review_ready_episode(episode) is False


def test_review_ready_episode_accepts_real_fault_with_concrete_actions():
    episode = _episode(
        fault="客户反馈设备正常运行中出现蓝屏现象，发生时间12:55左右。",
        diags=["远程收集日志并使用蓝屏dmp脚本收集dmp文件", "分析dmp定位蓝屏原因"],
    )
    assert rc.is_review_ready_episode(episode) is True
    assert rc.primary_fault_text(episode) == "客户反馈设备正常运行中出现蓝屏现象，发生时间12:55左右"


def test_inject_review_context_derives_fault_focus_text_and_confidence():
    episode = _episode(
        fault="@工程师丑 设备运行中拍摄失败，发生时间19:17。",
        diags=["检查相机网口角色与网络配置", "收集主程序和运控日志"],
        extracted={"symptom_raw": "@工程师丑 设备运行中拍摄失败，发生时间19:17。"},
    )
    enriched = rc.inject_review_context(episode, {"top_family_background": []})
    extracted = enriched["extracted"]
    assert extracted["review_context"]["context_role"] == "alignment_only"
    assert extracted["review_context"]["facts_may_not_be_copied_as_new_evidence"] is True
    assert extracted["sop_background"] == extracted["review_context"]
    assert extracted["sop_background_compatibility_alias"] is True
    assert extracted["fault_focus_text"] == "设备运行中拍摄失败，发生时间19:17"
    assert extracted["fault_focus_confidence"] > 0.0


def test_inject_review_context_drops_noise_reporter_and_classification():
    episode = {
        "episode_id": "ep:noise",
        "thread_id": "thread:noise",
        "completeness": "noise",
        "fault_description_messages": [],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "noise_messages": [{"message_id": "m1", "text": "各位领导晚上好，今日现场反馈已记录表格"}],
        "case_context_messages": [],
        "attachments": [],
        "extracted": {
            "symptom_raw": "各位领导晚上好，今日现场反馈已记录表格",
            "attribution": {
                "reporter_candidates": [{"name": "工程师D", "role_type": "reporter", "confidence": 0.6, "reason": ["field_feedback"], "evidence_message_ids": ["m1"]}],
                "owner_candidates": [],
                "owner_assignments": [],
                "responsibility_signals": [{"message_id": "m1", "signal_type": "reporter_signal", "sender": "工程师D", "reason": ["field_feedback"]}],
                "classification_hypotheses": [{"name": "工程师午", "role_type": "issue_owner", "problem_category": "其他问题及无法分类问题", "confidence": 0.2, "reason": ["fallback_unclassified"], "evidence_message_ids": ["m1"]}],
            },
        },
    }
    enriched = rc.inject_review_context(episode, {"top_family_background": []})
    attr = enriched["extracted"]["attribution"]
    assert attr["reporter_candidates"] == []
    assert attr["owner_candidates"] == []
    assert attr["classification_hypotheses"] == []
    assert enriched["extracted"]["attribution_raw"]["reporter_candidates"][0]["name"] == "工程师D"


def test_inject_review_context_keeps_hard_owner_assignment_but_drops_coordination_owner():
    good = _episode(
        fault="@工程师乙 客户反馈工控机蓝屏自动重启，麻烦看下异常原因。",
        diags=["先收集dmp和系统日志", "分析蓝屏原因"],
        extracted={
            "attribution": {
                "reporter_candidates": [],
                "owner_candidates": [{"name": "工程师乙", "role_type": "issue_owner", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["m1"]}],
                "owner_assignments": [{"name": "工程师乙", "role_type": "issue_owner", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["m1"]}],
                "responsibility_signals": [{"message_id": "m1", "signal_type": "owner_assignment", "name": "工程师乙", "reason": ["direct_owner_request"]}],
                "classification_hypotheses": [{"name": "工程师乙", "role_type": "issue_owner", "problem_category": "工控机/复判站/编程站及操作系统问题", "confidence": 0.55, "reason": ["matched:蓝屏"], "evidence_message_ids": ["m1"]}],
            }
        },
    )
    bad = _episode(
        fault="@崔桂彬 麻烦安排升级一下版本，现场经常出现闪退。",
        diags=["收到，请教一下升级到什么版本"],
        extracted={
            "attribution": {
                "reporter_candidates": [],
                "owner_candidates": [{"name": "崔桂彬", "role_type": "issue_owner", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["m1"]}],
                "owner_assignments": [{"name": "崔桂彬", "role_type": "issue_owner", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["m1"]}],
                "responsibility_signals": [{"message_id": "m1", "signal_type": "owner_assignment", "name": "崔桂彬", "reason": ["direct_owner_request"]}],
                "classification_hypotheses": [{"name": "工程师庚", "role_type": "issue_owner", "problem_category": "主程序软件问题", "confidence": 0.45, "reason": ["matched:闪退"], "evidence_message_ids": ["m1"]}],
            }
        },
    )
    good_attr = rc.inject_review_context(good, {"top_family_background": []})["extracted"]["attribution"]
    bad_attr = rc.inject_review_context(bad, {"top_family_background": []})["extracted"]["attribution"]
    assert good_attr["owner_assignments"][0]["name"] == "工程师乙"
    assert bad_attr["owner_assignments"] == []
    assert bad_attr["owner_candidates"] == []


def test_inject_review_context_keeps_soft_owner_takeover_without_direct_assignment():
    episode = _episode(
        fault="@杨明俊 有proj或者ctrlx给我先验证一下吗？看起来是ocr误报",
        diags=["我先看看日志吧", "先验证一下ocr误报路径"],
        extracted={
            "attribution": {
                "reporter_candidates": [],
                "owner_candidates": [{"name": "工程师E", "role_type": "issue_owner", "confidence": 0.7, "reason": ["diagnostic_takeover"], "evidence_message_ids": ["m2"]}],
                "owner_assignments": [],
                "responsibility_signals": [{"message_id": "m2", "signal_type": "owner_takeover", "sender": "工程师E", "reason": ["diagnostic_takeover"]}],
                "classification_hypotheses": [{"name": "工程师丁", "role_type": "issue_owner", "problem_category": "软件使用及调试问题", "confidence": 0.45, "reason": ["matched:误报"], "evidence_message_ids": ["m1"]}],
            }
        },
    )
    attr = rc.inject_review_context(episode, {"top_family_background": []})["extracted"]["attribution"]
    assert attr["owner_assignments"] == []
    assert attr["owner_candidates"][0]["name"] == "工程师E"


def test_review_ready_episode_allows_single_action_for_strong_fault_markers():
    episode = _episode(
        fault="刚才关机时出现了一次蓝屏现象，现场有接了地线。",
        diags=["收集蓝屏转存储文件和系统日志"],
    )
    assert rc.is_review_ready_episode(episode) is True


def test_review_ready_episode_rejects_non_concrete_meta_actions():
    episode = _episode(
        fault="大量的器件3D成像异常误报3D共面算法了。",
        diags=[
            "我懂，但是没有图，没有对应发生问题时的日志，这个没法排查的",
            "另外，运控卡供应商今天也提到我们这台机器的操作系统是iot版本的windows，这个也不太符合我的认知，@工程师乙 也能顺便确认下吗",
        ],
    )
    assert rc.concrete_action_texts(episode) == []
    assert rc.is_review_ready_episode(episode) is False


def test_primary_fault_text_pulls_tail_issue_out_of_daily_report():
    episode = _episode(
        fault="今日反馈表格已更新，各领导请查阅。二.问题点 1.现场测试时整板异物检测误报飞件，未报脏污，已收集相关数据提交jira。",
        diags=["已收集相关数据提交jira。", "@工程师庚 帮分析看看怎么解决"],
    )
    assert rc.primary_fault_text(episode) == "现场测试时整板异物检测误报飞件，未报脏污，已收集相关数据提交jira"
    assert rc.is_review_ready_episode(episode) is False


def test_review_ready_episode_rejects_strong_fault_without_any_concrete_action():
    episode = _episode(
        fault="设备突然蓝屏，和邢工一起排查疑似显卡问题。",
        diags=["各位领导，今日工作汇报。"],
    )
    assert rc.concrete_action_texts(episode) == []
    assert rc.is_review_ready_episode(episode) is False


def test_w7_promotes_same_case_followup_and_offline_jira(tmp_path):
    episode = _episode(
        fault="客户反馈外置扫码枪扫到码后不进板，重启主程序后临时恢复。",
        diags=["检查主程序是否收到条码。"],
    )
    episode["evidence_message_ids"] = ["m1", "m2"]
    jira_root = tmp_path / "jira"
    (jira_root / "fault_details").mkdir(parents=True)
    (jira_root / "fault_details" / "TEST-1234.json").write_text(json.dumps({
        "key": "TEST-1234",
        "summary": "外置扫码枪扫码后无法进板",
        "status": "Rejected",
        "resolution": "User Operations",
        "comments": [{"author": "hejie", "body": "扫码枪配置文件问题"}],
    }, ensure_ascii=False), encoding="utf-8")
    (jira_root / "fault_details" / "TEST-1234.json").write_text(json.dumps({
        "key": "TEST-1234",
        "summary": "主程序闪退",
        "status": "Closed",
        "comments": [],
    }, ensure_ascii=False), encoding="utf-8")
    supplemental = [
        {"message_id": "m3", "text": "扫码枪配置软件显示已经识码，但主程序没有进板。", "_context_distance": 2, "_history_index": 3},
        {"message_id": "m4", "text": "这里分析了下日志，先核对扫码枪配置文件。", "_context_distance": 3, "_history_index": 4},
        {"message_id": "m5", "text": "相关问题已提交 TEST-1234 https://jira.example.com/browse/TEST-1234；另有 TEST-1234 https://jira.example.com/browse/TEST-1234", "_context_distance": 4, "_history_index": 5},
        {"message_id": "m6", "text": "各位领导晚上好，今天完成客户培训。", "_context_distance": 5, "_history_index": 6},
    ]

    promoted = rc.promote_case_evidence(
        episode,
        supplemental_messages=supplemental,
        jira_offline_root=jira_root,
    )

    promoted_ids = [item["message_id"] for item in promoted["case_evidence_messages"]]
    assert promoted_ids == ["m3", "m4", "m5"]
    assert "m5" in promoted["evidence_message_ids"]
    assert promoted["extracted"]["linked_jira_evidence"][0]["issue_key"] == "TEST-1234"
    assert promoted["extracted"]["linked_jira_evidence"][0]["comments_preview"][0]["body_preview"] == "扫码枪配置文件问题"
    promoted_jira = promoted["case_evidence_messages"][-1]
    assert "TEST-1234" in promoted_jira["text"]
    assert "TEST-1234" not in promoted_jira["text"]
    assert "TEST-1234" in promoted_jira["raw_text"]


def test_w7_does_not_promote_mixed_daily_report_as_case_evidence():
    episode = _episode(
        fault="客户生产7175点产品时误报调试特别卡顿。",
        diags=["卡顿时打开任务管理器检查资源占用。"],
    )
    supplemental = [
        {
            "message_id": "m-report",
            "text": "各位领导，晚上好：一、现场工作：指导客户调试。二、问题汇总：7175点产品误报调试卡顿；MES图片输出待确认。",
            "_context_distance": 3,
            "_history_index": 8,
        },
        {
            "message_id": "m-result",
            "text": "任务管理器看上去没有什么资源占满，其他旧驱动可用DriverStoreExplorer清理。",
            "_context_distance": 2,
            "_history_index": 7,
        },
    ]

    promoted = rc.promote_case_evidence(episode, supplemental_messages=supplemental)

    assert [item["message_id"] for item in promoted["case_evidence_messages"]] == ["m-result"]


def test_w7_promotes_only_matching_numbered_fragment_from_multi_issue_message():
    episode = _episode(
        fault="设备通电测试时光源初始化失败。",
        diags=["重新拔插光源 USB 接口后恢复正常。"],
    )
    episode["evidence_message_ids"] = ["m1", "m2"]
    supplemental = [{
        "message_id": "m-mixed",
        "text": (
            "1、多次记录到 U 盘加载，随后 BUDDY 异常并重启电脑。 "
            "2、晚上没有关机，第二天继续使用设备。 "
            "3、昨天光源初始化失败，重新拔插光源 USB 接口后恢复正常。"
        ),
        "_context_distance": 2,
        "_history_index": 3,
    }]

    promoted = rc.promote_case_evidence(episode, supplemental_messages=supplemental)

    assert [item["message_id"] for item in promoted["case_evidence_messages"]] == ["m-mixed"]
    item = promoted["case_evidence_messages"][0]
    assert item["text"].startswith("3、昨天光源初始化失败")
    assert "BUDDY" not in item["text"]
    assert "BUDDY" in item["raw_text"]
    assert item["promotion_reason"].endswith("fragment_filtered")
