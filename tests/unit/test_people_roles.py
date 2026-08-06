from __future__ import annotations

import hashlib
import json

from debug_agent_system.agents.write.people_roles import load_people_role_registry, people_index
from debug_agent_system.agents.write.review_context import resolve_people_roles
from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent, _build_observed_people, _is_diagnostic_action


def _daily_report_messages() -> list[dict]:
    collector = ChatCollectAgent()
    return collector.normalize_messages([
        {
            "message_id": "report-1",
            "thread_id": "field-thread",
            "sender": {"id": "fae-1", "name": "现场张工"},
            "create_time": "2026-07-14 18:30:00",
            "content": (
                "今日现状：一、软件问题 1. 正常复判时频繁出现拍摄失败。"
                "二、设备问题 1. 远轨宽度异常导致板卡卡滞无法出板。"
                "以上信息请各位领导知悉。"
            ),
        },
        {
            "message_id": "jira-1",
            "thread_id": "field-thread",
            "sender": {"id": "fae-1", "name": "现场张工"},
            "create_time": "2026-07-14 18:40:00",
            "content": "拍摄失败问题已提交JIRA TEST-1234，并已上传日志。",
        },
    ])


def test_people_registry_matches_feedback_document_and_uses_confirmed_fae_members():
    registry = load_people_role_registry()
    index = people_index(registry)
    document_hash = hashlib.sha256(open("docs/现场问题反馈流程.md", "rb").read()).hexdigest()

    assert registry["source_document"]["sha256"] == document_hash
    assert index["工程师丑"]["responsibility_scopes"] == ["运控问题"]
    assert index["工程师癸"]["organization_roles"] == ["delivery_pm"]
    fae_team = next(item for item in registry["teams"] if item["role"] == "fae")
    roster_hash = hashlib.sha256(open("data/annotations/fae_engineers_2026-07-21.csv", "rb").read()).hexdigest()
    assert len(fae_team["confirmed_members"]) == 37
    assert registry["fae_roster"] == {
        "path": "data/annotations/fae_engineers_2026-07-21.csv",
        "sha256": roster_hash,
        "snapshot_date": "2026-07-21",
        "record_count": 37,
        "active_count": 37,
    }
    assert fae_team["source_sha256"] == roster_hash
    assert index["工程师巳"]["organization_roles"] == ["fae"]
    assert index["工程师己"]["organization_roles"] == ["fae"]
    assert index["工程师巳"]["status"] == "confirmed"
    assert index["工程师己"]["status"] == "confirmed"
    assert index["邓志勇"]["organization_roles"] == ["fae"]
    assert index["工程师子"]["open_ids"] == ["ou_e0389d136f7967cb0a740216f322bbe5"]
    assert index["工程师申"]["departments"] == ["华南一组"]
    assert "工程师辛" in index
    assert "王工程师酉" not in index


def test_w1_recognizes_fae_log_query_as_diagnostic_action_without_generic_view_false_positive():
    assert _is_diagnostic_action({"msg_type": "text", "text": "查询系统日志有大量网卡重置报错"})
    assert _is_diagnostic_action({"msg_type": "text", "text": "查看事件查看器中的错误码"})
    assert not _is_diagnostic_action({"msg_type": "text", "text": "查看一下现场照片"})


def test_w1_field_report_anchor_splits_issue_items_and_marks_episode_lineage():
    collector = ChatCollectAgent()
    summaries = collector.aggregate_threads(_daily_report_messages())

    assert len(summaries[0]["field_report_anchors"]) == 1
    anchor = summaries[0]["field_report_anchors"][0]
    assert anchor["author"] == "现场张工"
    assert anchor["issue_count"] == 2
    assert len(summaries[0]["episodes"]) >= 2
    anchored = [episode for episode in summaries[0]["episodes"] if episode.get("field_report_anchor")]
    assert len(anchored) == 2
    assert {episode["field_report_anchor"]["anchor_item_index"] for episode in anchored} == {1, 2}


def test_w1_inline_chinese_heading_report_splits_each_fault_item():
    collector = ChatCollectAgent()
    messages = collector.normalize_messages([
        {
            "message_id": "prelude",
            "thread_id": "inline-thread",
            "sender": {"name": "研发工程师"},
            "create_time": "2025-09-11 16:33:00",
            "content": "不用复现了，先确认现象。",
        },
        {
            "message_id": "inline-report",
            "thread_id": "inline-thread",
            "sender": {"name": "蒋万涛"},
            "create_time": "2025-09-11 22:35:00",
            "content": (
                "一，现场工作： 1.跟进客户生产，指导客户人员使用设备 "
                "2.收集数据，上传百度云盘 二，现场问题点： "
                "1.器件引脚识别不佳。JIRA： 2.单边复制功能无法使用，锚定后仍无法使用，需排查原因。"
            ),
        },
        {
            "message_id": "next-fault",
            "thread_id": "inline-thread",
            "sender": {"name": "现场张工"},
            "create_time": "2025-09-12 13:10:00",
            "content": "设备断电重启后相机拍摄失败。",
        },
    ])
    report = next(item for item in messages if item["message_id"] == "inline-report")
    report["links"] = [
        {"url": "https://jira.example.com/browse/TEST-1234", "type": "jira", "message_id": "inline-report"},
        {"url": "https://jira.example.com/browse/TEST-1234", "type": "jira", "message_id": "inline-report"},
    ]
    report["attachments"] = [{"name": "ambiguous.zip", "file_key": "file-1", "message_id": "inline-report"}]

    summary = collector.aggregate_threads(messages)[0]
    anchor = summary["field_report_anchors"][0]
    anchored = [episode for episode in summary["episodes"] if episode.get("field_report_anchor")]

    assert anchor["issue_count"] == 2
    assert [item["text"] for item in anchor["issue_items"]] == [
        "1.器件引脚识别不佳。JIRA：",
        "2.单边复制功能无法使用，锚定后仍无法使用，需排查原因",
    ]
    assert len(anchored) == 2
    assert [episode["field_report_anchor"]["anchor_item_index"] for episode in anchored] == [1, 2]
    assert all(episode["message_count"] == 1 for episode in anchored)
    assert all(
        [message["message_id"] for message in episode["case_context_messages"]]
        == ["inline-report"]
        for episode in anchored
    )
    assert all("prelude" not in episode["evidence_message_ids"] for episode in anchored)
    assert all("next-fault" not in episode["evidence_message_ids"] for episode in anchored)
    assert all(episode["attachments"] == [] for episode in anchored)
    assert all((episode["extracted"] or {}).get("jira_links") == [] for episode in anchored)
    assert all((episode["extracted"] or {}).get("unassigned_shared_evidence") for episode in anchored)


def test_w1_episode_context_does_not_reintroduce_unrelated_segment_messages():
    collector = ChatCollectAgent()
    messages = collector.normalize_messages([
        {
            "message_id": "old-1",
            "thread_id": "mixed-thread",
            "sender": {"name": "现场"},
            "create_time": "2026-07-14 08:00:00",
            "content": "另一个问题：磁盘消失，已联系负责人。",
        },
        {
            "message_id": "old-2",
            "thread_id": "mixed-thread",
            "sender": {"name": "现场"},
            "create_time": "2026-07-14 08:10:00",
            "content": "项目协调：安排明天培训和版本升级。",
        },
        {
            "message_id": "report-1",
            "thread_id": "mixed-thread",
            "sender": {"name": "现场张工"},
            "create_time": "2026-07-14 18:30:00",
            "content": "今日现状：一、现场问题点：1.光源初始化失败，重连USB后恢复正常。",
        },
        {
            "message_id": "later-1",
            "thread_id": "mixed-thread",
            "sender": {"name": "研发"},
            "create_time": "2026-07-14 18:40:00",
            "content": "另一个问题：扫码枪配置文件加载失败。",
        },
    ])
    episodes = collector.aggregate_threads(messages)[0]["episodes"]
    light = next(ep for ep in episodes if "光源初始化失败" in " ".join(item["text"] for item in ep["fault_description_messages"]))
    context_ids = {item["message_id"] for item in light["case_context_messages"]}
    assert "report-1" in context_ids
    assert "old-1" not in context_ids
    assert "old-2" not in context_ids
    assert "later-1" not in context_ids


def test_w1_observed_people_keeps_fae_as_candidate_not_confirmed_identity():
    messages = _daily_report_messages()
    summary = ChatCollectAgent().aggregate_threads(messages)[0]
    observed = _build_observed_people(messages, summary["field_report_anchors"])
    person = next(item for item in observed if item["name"] == "现场张工")

    assert person["status"] == "observed"
    assert "fae" in person["organization_role_candidates"]
    assert "fae" not in person["organization_roles"]
    assert person["episode_role_counts"]["field_report_author"] == 1
    assert person["episode_role_counts"]["evidence_provider"] >= 1


def test_w1_single_fault_daily_report_does_not_feed_work_items_into_episode():
    collector = ChatCollectAgent()
    messages = collector.normalize_messages([
        {
            "message_id": "single-report",
            "thread_id": "single-thread",
            "sender": {"name": "现场张工"},
            "content": (
                "今日工作汇总：一、培训客户快速编程和误报调试。"
                "二、现场问题：1. 客户反馈正常复判时频繁拍摄失败。"
                "三、日常数据已上传网盘。"
            ),
        }
    ])

    summary = collector.aggregate_threads(messages)[0]
    fault_text = summary["episodes"][0]["fault_description_messages"][0]["text"]

    assert "拍摄失败" in fault_text
    assert "培训客户" not in fault_text
    assert "上传网盘" not in fault_text


def test_w7_resolves_organization_and_episode_roles_on_separate_axes():
    episode = {
        "field_report_anchor": {"author": "现场张工", "message_id": "report-1"},
        "fault_description_messages": [],
        "diagnostic_chain_messages": [
            {"message_id": "owner-1", "sender": {"name": "工程师丑"}, "text": "检查相机采集日志。"}
        ],
        "resolution_messages": [],
    }
    attribution = {
        "reporter_candidates": [{"name": "现场张工", "confidence": 0.7, "evidence_message_ids": ["report-1"]}],
        "owner_candidates": [{"name": "工程师丑", "confidence": 0.8, "reason": ["direct_owner_request"], "evidence_message_ids": ["owner-1"]}],
        "classification_hypotheses": [{"name": "工程师丑", "problem_category": "运控问题"}],
    }

    assignments = resolve_people_roles(episode, attribution, load_people_role_registry())
    by_name = {item["name"]: item for item in assignments}

    assert by_name["现场张工"]["organization_roles"] == []
    assert set(by_name["现场张工"]["episode_roles"]) == {"reporter", "field_report_author"}
    assert by_name["工程师丑"]["organization_roles"] == ["rd_engineer"]
    assert set(by_name["工程师丑"]["episode_roles"]) == {"assignee", "investigator"}
    assert by_name["工程师丑"]["responsibility_scopes"] == ["运控问题"]
    assert all(item["evidence_message_ids"] for item in assignments)


def test_w1_write_run_emits_people_and_anchor_artifacts(tmp_path):
    collector = ChatCollectAgent()
    messages = _daily_report_messages()
    summaries = collector.aggregate_threads(messages)
    anchors = summaries[0]["field_report_anchors"]
    run = {
        "messages": messages,
        "thread_summaries": summaries,
        "episodes": summaries[0]["episodes"],
        "field_report_anchors": anchors,
        "observed_people": _build_observed_people(messages, anchors),
        "run_manifest": {"counts": {}},
    }

    files = collector.write_run(tmp_path, run)

    assert json.loads((tmp_path / "field_report_anchors.json").read_text(encoding="utf-8"))[0]["issue_count"] == 2
    assert json.loads((tmp_path / "observed_people.json").read_text(encoding="utf-8"))[0]["name"] == "现场张工"
    assert files["field_report_anchors"].endswith("field_report_anchors.json")
    assert files["observed_people"].endswith("observed_people.json")
