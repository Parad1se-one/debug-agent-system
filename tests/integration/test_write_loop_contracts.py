from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from debug_agent_system.adapters.cli import main as cli_main
from debug_agent_system.agents.loop import ATRWeightingAgent, DiagnosticFeedbackAgent, LogPatternAgent
from debug_agent_system.agents.write import (
    ChatCollectAgent,
    ConflictResolutionAgent,
    IncrementalIngestAgent,
    KnowledgeExtractionAgent,
    QualityGateAgent,
    ReviewQueueAgent,
    WriteSidePipeline,
)
from debug_agent_system.eval.debug_sim.ask_info_candidates import build_scenarios, main as ask_info_candidates_main
from debug_agent_system.knowledge.json_store import JsonKGStore


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00"


def _msg(seg: str, idx: int, content: str, *, msg_type: str = "text", resources: str = "0") -> dict[str, str]:
    return {
        "segment_id": seg,
        "message_id": f"om_test_{idx}",
        "chat_name": "【1.0已签单】江苏客户01项目群",
        "create_time": f"2026-06-01 09:{idx:02d}",
        "sender": "工程师丁" if idx % 2 else "工程师乙",
        "msg_type": msg_type,
        "is_hit": "true",
        "is_xing_related": "true",
        "resource_count": resources,
        "content": content,
    }


def _fake_xing_upload(root: Path) -> Path:
    manifest = root / "_MANIFEST"
    seg = "oc_test_202606010900000800_202606011000000800_1"
    rows = [
        _msg(
            seg,
            1,
            "<p>@工程师乙 客户反馈 AOI 主程序初始化失败，相机连接异常，SMTAOITS-1234。</p>"
            "<p>飞书链接 https://example.feishu.cn/file/noise</p>",
            msg_type="post",
        ),
        _msg(seg, 2, "麻烦提供 DLOG 诊断数据包和主程序版本，用于判断初始化阶段。"),
        _msg(seg, 3, "已上传 DLOG_AOI-4294_20260601.zip，版本25.7.1，IP 192.168.1.10。", resources="1"),
        _msg(seg, 4, "先检查相机IP配置和网线连接，再重启相机服务。"),
        _msg(seg, 5, "已解决，恢复正常，原因是现场相机IP被改。"),
        _msg(seg, 6, "收到，谢谢，需求排期会后确认。"),
        _msg(seg, 7, "客户反馈 AOI 光源控制器异常，无法拍照，版本25.7.2。"),
        _msg(seg, 8, "检查光源控制器配置，确认触发线和IO状态。"),
        _msg(seg, 9, "已解决，恢复正常，原因是光源控制器触发配置错误。"),
    ]
    _write_csv(manifest / "xing_messages.csv", rows)
    _write_csv(manifest / "xing_resource_files.csv", [
        {
            "relative_path": "om_test_3/DLOG_AOI-4294_20260601.zip",
            "status": "api_ok",
            "type": "file",
            "bytes": "100",
            "name": "DLOG_AOI-4294_20260601.zip",
            "message_id": "om_test_3",
            "segment_id": seg,
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "create_time": "2026-06-01 09:03",
            "sender": "工程师丁",
            "copied": "true",
            "copied_bytes": "100",
        }
    ])
    _write_csv(manifest / "xing_segments.csv", [
        {
            "segment_id": seg,
            "index": "1",
            "chat_id": "oc_test",
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "start": "2026-06-01T09:00:00+08:00",
            "end": "2026-06-01T09:30:00+08:00",
            "messages": str(len(rows)),
            "hits": str(len(rows)),
            "resources": "1",
            "uploadable": "1",
            "unavailable": "0",
            "xing_rule_ok": "true",
            "xing_before_first": "0",
            "xing_after_last": "0",
            "xing_total": str(len(rows)),
            "xing_count": str(len(rows)),
        }
    ])
    return root


def _node(candidate: dict, node_type: str) -> dict:
    for node in candidate["nodes"]:
        if node["type"] == node_type:
            return node
    raise AssertionError(f"missing node type {node_type}")


def test_write_agents_return_structured_outputs():
    extracted = KnowledgeExtractionAgent().extract({
        "thread_id": "t1",
        "extracted": {"symptom_raw": "相机IP异常", "debug_actions": ["检查IP"], "conclusion": "恢复正常"},
        "evidence_message_ids": ["m1"],
    })
    assert set(extracted) >= {"nodes", "edges", "confidence", "source", "schema_valid", "schema_issues"}
    assert extracted["type"] == "SchemaValidCandidate"
    assert extracted["schema_valid"] is True
    assert "Version" not in {node["type"] for node in extracted["nodes"]}
    assert _node(extracted, "Error")["error_id"]
    assert _node(extracted, "DiagnosticCheck")["check_id"]
    assert _node(extracted, "DiagnosticCheck")["how_to_check"]
    assert _node(extracted, "DiagnosticCheck")["step_order"] == 1
    assert _node(extracted, "Solution")["solution_id"]
    assert _node(extracted, "Solution")["method"]
    assert _node(extracted, "Solution")["evidence_level"]
    gate = QualityGateAgent().score(extracted)
    assert gate["passed"] is True
    assert "schema_validity" in gate
    conflict = ConflictResolutionAgent().resolve(extracted, None)
    assert conflict["decision"] == "Agree"
    assert conflict["conflict_type"] == "new_error"
    assert conflict["requires_human"] is True



def test_w1_does_not_treat_diagnostic_handoff_as_missing_info_request():
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈设备蓝屏重启。"},
        {"message_id": "m2", "thread_id": "t", "sender": "fae", "content": "麻烦从硬件上看一下是什么问题。"},
        {"message_id": "m3", "thread_id": "t", "sender": "dev", "content": "先检查工控机硬盘和内存。"},
    ])
    summary = ChatCollectAgent().aggregate_threads(messages)[0]
    requests = summary["episodes"][0]["extracted"]["missing_info_requests"]
    assert requests == []

def test_w1_does_not_treat_provided_artifact_statement_as_missing_info_request():
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈设备蓝屏重启。"},
        {"message_id": "m2", "thread_id": "t", "sender": "fae", "content": "诊断日志里有windows事件导出。"},
        {"message_id": "m3", "thread_id": "t", "sender": "fae", "content": "日志已上传，客户描述弹窗后黑屏关机。"},
        {"message_id": "m4", "thread_id": "t", "sender": "dev", "content": "请提供windows日志中的Bugcheck错误截图。"},
    ])
    summary = ChatCollectAgent().aggregate_threads(messages)[0]
    requests = summary["episodes"][0]["extracted"]["missing_info_requests"]
    assert [r["message_id"] for r in requests] == ["m4"]


def test_w1_does_not_treat_jira_or_remote_code_or_package_chat_as_missing_info_request():
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈软件闪退。"},
        {"message_id": "m2", "thread_id": "t", "sender": "fae", "content": "@张超 张工，补充JIRA如下[[TEST-1234] 0.27.44，客户15，软件闪退 - Jira]()"},
        {"message_id": "m3", "thread_id": "t", "sender": "dev", "content": "远程码发我一下吧，这是系统文件有损坏。"},
        {"message_id": "m4", "thread_id": "t", "sender": "dev", "content": "我的天，你咋会用到9.4.0的包，commit还不对，我没发给过你吧。"},
        {"message_id": "m5", "thread_id": "t", "sender": "dev", "content": "请提供完整 DLOG 日志用于判断闪退阶段。"},
    ])
    summary = ChatCollectAgent().aggregate_threads(messages)[0]
    requests = [r for episode in summary["episodes"] for r in episode["extracted"]["missing_info_requests"]]
    assert [r["message_id"] for r in requests] == ["m5"]


def test_w2_required_info_slots_use_request_text_not_surrounding_context_noise():
    episode = {
        "episode_id": "ep:req",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场蓝屏重启，@硬件同事看一下。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场蓝屏重启，@硬件同事看一下。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供windows日志中的Bugcheck错误截图。",
                "thread_id": "t",
                "context_before": [{"text": "现场蓝屏重启，@硬件同事看一下。"}],
                "context_after": [],
                "evidence_message_ids": ["m1"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    slots = {item["slot"] for item in candidate["required_info_candidates"]}
    assert slots == {"log_package", "error_message"}


def test_w1_does_not_treat_resolution_supplement_as_missing_info_request():
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈主程序打不开。"},
        {"message_id": "m2", "thread_id": "t", "sender": "fae", "content": "补充：使用相机重拍之后，直接再次打开主程序，可以正常使用了。"},
    ])
    summary = ChatCollectAgent().aggregate_threads(messages)[0]
    requests = summary["episodes"][0]["extracted"]["missing_info_requests"]
    assert requests == []


def test_w2_does_not_make_owner_context_from_mentions_only():
    episode = {
        "episode_id": "ep:req-owner",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场应用日志没有对应时间点。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场应用日志没有对应时间点。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "@工程师丑 安全和更新日志是单独的，得发我一下。",
                "thread_id": "t",
                "context_before": [],
                "context_after": [],
                "evidence_message_ids": ["m1"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    slots = {item["slot"] for item in candidate["required_info_candidates"]}
    assert "log_package" in slots
    assert "owner_context" not in slots


def test_w1_extracted_tool_evidence_parses_attachments_jira_and_proj_metadata_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606040900000800_202606041000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(
                seg,
                1,
                "客户反馈导入程序后相机 IP 异常。"
                " [[LOCALTEST-999999] 程序导入异常 - Jira](https://jira.example.com/browse/LOCALTEST-999999)",
                resources="1",
            ),
            _msg(seg, 2, "请提供配方程序文件和相机 IP 配置。"),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [
            {
                "relative_path": "om_test_1/line_recipe.proj",
                "status": "api_ok",
                "type": "file",
                "bytes": "42",
                "name": "line_recipe.proj",
                "message_id": "om_test_1",
                "segment_id": seg,
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-04 09:01",
                "sender": "工程师丁",
                "copied": "true",
                "copied_bytes": "42",
            }
        ])
        (root / "om_test_1").mkdir(parents=True)
        (root / "om_test_1" / "line_recipe.proj").write_text("Version=4.5.6\nCameraIP=10.20.30.40\n", encoding="utf-8")
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-04T09:00:00+08:00",
                "end": "2026-06-04T10:00:00+08:00",
                "messages": "2",
                "hits": "2",
                "resources": "1",
                "uploadable": "1",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "2",
                "xing_count": "2",
            }
        ])

        run = ChatCollectAgent().import_xing_upload(root)
        tool_evidence = run["episodes"][0]["extracted"]["tool_evidence"]
        assert tool_evidence["attachment_parse_results"][0]["evidence_role"] == "program_file"
        assert tool_evidence["jira_parse_results"][0]["issue_keys"] == ["LOCALTEST-999999"]
        assert tool_evidence["jira_parse_results"][0]["title_hints"] == ["程序导入异常"]
        proj = tool_evidence["proj_parse_results"][0]
        assert proj["type"] == "ProjParseResult"
        assert proj["executed"] is False
        assert proj["mutated"] is False
        assert "4.5.6" in proj["key_hints"]["versions"]
        assert "10.20.30.40" in proj["key_hints"]["ip_addresses"]


def test_w1_extracted_tool_evidence_parses_log_package_text_hints_safely():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606041100000800_202606041200000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈软件蓝屏重启。", resources="1"),
            _msg(seg, 2, "请提供蓝屏对应的 dmp 和系统日志。"),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [
            {
                "relative_path": "om_test_1/DLOG_blue_screen.zip",
                "status": "api_ok",
                "type": "file",
                "bytes": "42",
                "name": "DLOG_blue_screen.zip",
                "message_id": "om_test_1",
                "segment_id": seg,
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-04 11:01",
                "sender": "工程师丁",
                "copied": "true",
                "copied_bytes": "42",
            }
        ])
        (root / "om_test_1").mkdir(parents=True)
        with zipfile.ZipFile(root / "om_test_1" / "DLOG_blue_screen.zip", "w") as zf:
            zf.writestr("startup/init.log", "2026-06-04 ERROR startup camera failed code 0x80070005")
            zf.writestr("crash/MEMORY.DMP", b"not read")
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-04T11:00:00+08:00",
                "end": "2026-06-04T12:00:00+08:00",
                "messages": "2",
                "hits": "2",
                "resources": "1",
                "uploadable": "1",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "2",
                "xing_count": "2",
            }
        ])

        run = ChatCollectAgent().import_xing_upload(root)
        evidence = run["episodes"][0]["extracted"]["tool_evidence"]
        log_package = evidence["log_package_parse_results"][0]
        assert log_package["type"] == "LogPackageParseResult"
        assert log_package["archive_extracted"] is False
        assert log_package["content_read"] is False
        assert log_package["text_preview_read"] is True
        assert "0x80070005" in log_package["text_hints"]["error_codes"]
        assert "startup" in log_package["text_hints"]["phase_hints"]
        assert log_package["has_dmp"] is True
        assert log_package["has_startup_log"] is True


def test_w2_and_w4_use_later_tool_evidence_for_required_info_quality():
    episode = {
        "episode_id": "ep:req-tool",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场导入配方后相机 IP 异常。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场导入配方后相机 IP 异常。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供配方程序文件和相机 IP 配置。",
                "thread_id": "t",
                "context_before": [{"text": "现场导入配方后相机 IP 异常。"}],
                "context_after": [{"message_id": "m2", "text": "已上传 line_recipe.proj。"}],
                "evidence_message_ids": ["m1", "m2"],
                "provided_later": True,
                "provided_evidence_message_ids": ["m2"],
            }],
            "tool_evidence": {
                "attachment_parse_results": [{
                    "type": "AttachmentParseResult",
                    "name": "line_recipe.proj",
                    "path": "/tmp/line_recipe.proj",
                    "evidence_role": "program_file",
                    "source": {"message_id": "m2"},
                    "content_read": False,
                    "archive_extracted": False,
                }],
                "jira_parse_results": [],
                "proj_parse_results": [{
                    "type": "ProjParseResult",
                    "path": "/tmp/line_recipe.proj",
                    "executed": False,
                    "mutated": False,
                    "key_hints": {"versions": ["4.5.6"], "ip_addresses": ["10.20.30.40"]},
                }],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert "4.5.6" in candidate["versions"]
    assert "10.20.30.40" in candidate["ip_configs"]
    program = next(item for item in candidate["required_info_candidates"] if item["slot"] == "program_file")
    assert "program_file" in program["provided_tool_roles"]
    assert "proj_parsed" in program["provided_tool_roles"]
    assert program["quality"]["evidence_strength"] >= 0.7
    gate = QualityGateAgent().score_required_info(program)
    assert gate["evidence_strength"] >= 0.7


def test_w2_and_w4_use_later_log_package_manifest_for_required_info_quality():
    episode = {
        "episode_id": "ep:req-log-tool",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场蓝屏重启。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场蓝屏重启。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供蓝屏对应的 dmp 和系统日志。",
                "thread_id": "t",
                "context_before": [{"text": "现场蓝屏重启。"}],
                "context_after": [{"message_id": "m2", "text": "已上传 DLOG_blue_screen.zip。"}],
                "evidence_message_ids": ["m1", "m2"],
                "provided_later": True,
                "provided_evidence_message_ids": ["m2"],
            }],
            "tool_evidence": {
                "attachment_parse_results": [{
                    "type": "AttachmentParseResult",
                    "name": "DLOG_blue_screen.zip",
                    "path": "/tmp/DLOG_blue_screen.zip",
                    "evidence_role": "log_package",
                    "source": {"message_id": "m2"},
                    "content_read": False,
                    "archive_extracted": False,
                }],
                "jira_parse_results": [],
                "proj_parse_results": [],
                "log_package_parse_results": [{
                    "type": "LogPackageParseResult",
                    "name": "DLOG_blue_screen.zip",
                    "source": {"message_id": "m2"},
                    "has_dmp": True,
                    "has_evtx": False,
                    "has_startup_log": True,
                    "has_dlog": True,
                    "text_hints": {
                        "error_codes": ["BugCheck 0x0000007E"],
                        "error_lines": ["BugCheck 0x0000007E caused reboot"],
                        "phase_hints": ["startup"],
                    },
                    "entries": [{"name": "crash/MEMORY.DMP", "role": "memory_dump"}],
                }],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    required = next(item for item in candidate["required_info_candidates"] if item["slot"] == "log_package")
    assert "DLOG_blue_screen.zip" in candidate["log_paths"]
    assert "crash/MEMORY.DMP" in candidate["log_paths"]
    assert "BugCheck 0x0000007E caused reboot" in candidate["log_error_hints"]
    assert "BugCheck 0x0000007E" in candidate["log_error_codes"]
    assert "startup" in candidate["log_phase_hints"]
    assert "log_package" in required["provided_tool_roles"]
    assert "log_package_manifest" in required["provided_tool_roles"]
    assert "log_text_hints" in required["provided_tool_roles"]
    assert "log_manifest_has_dmp" in required["provided_tool_roles"]
    assert required["quality"]["evidence_strength"] >= 0.7
    gate = QualityGateAgent().score_required_info(required)
    assert gate["evidence_strength"] >= 0.7


def test_w2_and_w4_do_not_boost_required_info_with_mismatched_later_tool_evidence():
    episode = {
        "episode_id": "ep:req-tool-mismatch",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场蓝屏重启。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场蓝屏重启。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供蓝屏对应的 dmp 和系统日志。",
                "thread_id": "t",
                "context_before": [{"text": "现场蓝屏重启。"}],
                "context_after": [{"message_id": "m2", "text": "已上传 line_recipe.proj。"}],
                "evidence_message_ids": ["m1", "m2"],
                "provided_later": True,
                "provided_evidence_message_ids": ["m2"],
            }],
            "tool_evidence": {
                "attachment_parse_results": [{
                    "type": "AttachmentParseResult",
                    "name": "line_recipe.proj",
                    "path": "/tmp/line_recipe.proj",
                    "evidence_role": "program_file",
                    "source": {"message_id": "m2"},
                    "content_read": False,
                    "archive_extracted": False,
                }],
                "proj_parse_results": [{
                    "type": "ProjParseResult",
                    "path": "/tmp/line_recipe.proj",
                    "source": {"message_id": "m2"},
                    "executed": False,
                    "mutated": False,
                    "key_hints": {"versions": ["4.5.6"], "ip_addresses": ["10.20.30.40"]},
                }],
                "log_package_parse_results": [],
                "jira_parse_results": [],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    required = next(item for item in candidate["required_info_candidates"] if item["slot"] == "log_package")
    assert "program_file" in required["provided_tool_roles"]
    assert required["provided_slot_match_roles"] == []
    assert required["quality"]["provided_tool_roles_mismatch"] is True
    assert required["quality"]["evidence_strength"] <= 0.45
    gate = QualityGateAgent().score_required_info(required)
    assert "provided_tool_roles_mismatch" in gate["issues"]
    assert gate["evidence_strength"] <= 0.45


def test_w2_ignores_global_proj_evidence_not_linked_to_provided_message():
    episode = {
        "episode_id": "ep:req-unrelated-global-proj",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场蓝屏重启。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场蓝屏重启。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供蓝屏对应的 dmp 和系统日志。",
                "thread_id": "t",
                "context_before": [{"text": "现场蓝屏重启。"}],
                "context_after": [{"message_id": "m2", "text": "稍后提供。"}],
                "evidence_message_ids": ["m1", "m2"],
                "provided_later": True,
                "provided_evidence_message_ids": ["m2"],
            }],
            "tool_evidence": {
                "attachment_parse_results": [{
                    "type": "AttachmentParseResult",
                    "name": "old_recipe.proj",
                    "path": "/tmp/old_recipe.proj",
                    "evidence_role": "program_file",
                    "source": {"message_id": "m0"},
                    "content_read": False,
                    "archive_extracted": False,
                }],
                "proj_parse_results": [{
                    "type": "ProjParseResult",
                    "path": "/tmp/old_recipe.proj",
                    "source": {"message_id": "m0"},
                    "executed": False,
                    "mutated": False,
                    "key_hints": {"versions": ["9.9.9"], "ip_addresses": ["10.99.99.99"]},
                }],
                "log_package_parse_results": [],
                "jira_parse_results": [],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    required = next(item for item in candidate["required_info_candidates"] if item["slot"] == "log_package")
    assert required["provided_tool_roles"] == []
    assert required["provided_slot_match_roles"] == []
    assert required["quality"].get("provided_tool_roles_mismatch") is None


def test_w2_does_not_treat_screenshot_statements_as_missing_info_focus():
    for text in ("客户数据已收集，现场没有截图。", "客户今早反馈软件闪退，日志在上面，发生时间截图能看得到。"):
        episode = {
            "episode_id": f"ep:not-request:{hash(text)}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": "m0", "text": "客户反馈软件闪退。"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "evidence_message_ids": ["m1"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈软件闪退。",
                "debug_actions": [],
                "conclusion": "",
                "missing_info_requests": [{
                    "message_id": "m1",
                    "text": text,
                    "thread_id": "t",
                    "evidence_message_ids": ["m1"],
                    "provided_later": False,
                    "provided_evidence_message_ids": [],
                }],
            },
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        assert candidate["required_info_candidates"] == []


def test_w2_does_not_treat_collected_but_unuploadable_report_as_ask_info():
    episode = {
        "episode_id": "ep:not-request-collected-unuploadable",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "黑色内存条卡座虚焊检测存在漏检。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "黑色内存条卡座虚焊检测存在漏检。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "相关信息已收集，飞书内存不足，无法上传。问题如下 1/黑色内存条卡座虚焊检测存在漏检。",
                "thread_id": "t",
                "evidence_message_ids": ["m1"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["required_info_candidates"] == []


def test_w2_disk_screenshot_request_does_not_append_to_unrelated_matched_error():
    class FakeStore:
        def search_errors(self, query: str, limit: int = 5):
            class C:
                error_id = "err:app-freeze-during-test"
                label = "测试过程中开始菜单和任务栏卡死"
                score = 99.0
                route = "test"
                evidence = ["forced"]
            return [C()]

    episode = {
        "episode_id": "ep:req-disk-screenshot-unrelated-target",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "第三块物理硬盘的第三个分区有错误。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "第三块物理硬盘的第三个分区有错误。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供磁盘管理器界面的截图，我看下是哪个分区。",
                "thread_id": "t",
                "evidence_message_ids": ["m1"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent(FakeStore(), match_threshold=0.1).extract(episode)
    required = candidate["required_info_candidates"]
    assert required
    assert required[0]["target_error_id"] == ""
    assert required[0]["merge_policy"] == "review_only"


def test_w1_imports_real_xing_manifest_shape_and_splits_episodes():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "upload")
        run = ChatCollectAgent().import_xing_upload(root)
        assert run["run_manifest"]["counts"] == {
            "messages": 9,
            "threads": 1,
            "episodes": 2,
            "attachments": 1,
            "hits": 9,
            "field_report_anchors": 0,
            "observed_people": 2,
        }
        summary = run["thread_summaries"][0]
        assert len(summary["episodes"]) == 2
        assert {episode["completeness"] for episode in summary["episodes"]} == {"complete"}
        first = summary["episodes"][0]
        assert first["fault_description_messages"]
        assert first["diagnostic_chain_messages"]
        assert first["resolution_messages"]
        assert first["noise_messages"]
        assert "<p>" not in first["extracted"]["symptom_raw"]
        assert "http" not in first["extracted"]["symptom_raw"]
        assert first["attachments"][0]["status"] == "metadata_only"
        requests = first["extracted"]["missing_info_requests"]
        assert len(requests) == 1
        assert requests[0]["message_id"] == "om_test_2"
        assert requests[0]["provided_later"] is True
        assert "om_test_3" in requests[0]["provided_evidence_message_ids"]
        extracted = summary["extracted"]
        assert extracted["sites"] == ["江苏客户01"]
        assert "25.7.1" in extracted["versions"]
        assert "SMTAOITS-1234" in extracted["jira_ids"]
        assert any("DLOG_AOI-4294_20260601.zip" in x for x in extracted["log_paths"])
        assert summary["evidence_message_ids"]


def test_w1_imports_text_history_and_builds_segments_with_attribution():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "history"
        _write_jsonl(root / "all_text_messages.jsonl", [
            {
                "message_id": "m1",
                "thread_id": "",
                "chat_id": "oc_hist",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-01 09:00",
                "sender": "工程师B",
                "sender_id": "u_fae",
                "msg_type": "post",
                "mentions": [{"name": "工程师丑"}],
                "content": "<p>@工程师丑 现场每日反馈：客户反馈设备拍照停顿，运控闪退，已提交JIRA SMTAOITS-1234</p>",
                "plain_text": "@工程师丑 现场每日反馈：客户反馈设备拍照停顿，运控闪退，已提交JIRA SMTAOITS-1234",
            },
            {
                "message_id": "m2",
                "thread_id": "",
                "chat_id": "oc_hist",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-01 09:05",
                "sender": "工程师丑",
                "sender_id": "u_owner",
                "msg_type": "text",
                "mentions": [],
                "content": "我看下，先检查相机IP和运控日志",
                "plain_text": "我看下，先检查相机IP和运控日志",
            },
            {
                "message_id": "m3",
                "thread_id": "",
                "chat_id": "oc_hist",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-01 09:10",
                "sender": "工程师B",
                "sender_id": "u_fae",
                "msg_type": "text",
                "mentions": [],
                "content": "已提供相机IP 192.168.0.10 和运控日志包",
                "plain_text": "已提供相机IP 192.168.0.10 和运控日志包",
            },
            {
                "message_id": "m4",
                "thread_id": "",
                "chat_id": "oc_hist",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-02 22:30",
                "sender": "工程师B",
                "sender_id": "u_fae",
                "msg_type": "text",
                "mentions": [{"name": "工程师乙"}],
                "content": "@工程师乙 客户反馈工控机蓝屏自动重启，帮忙看下",
                "plain_text": "@工程师乙 客户反馈工控机蓝屏自动重启，帮忙看下",
            },
        ])
        run = ChatCollectAgent().import_text_history(root)
        assert run["run_manifest"]["source"] == "text_jsonl_history"
        assert run["run_manifest"]["counts"]["messages"] == 4
        assert run["run_manifest"]["counts"]["threads"] == 2
        assert run["run_manifest"]["counts"]["segment_count"] == 2
        assert run["run_manifest"]["counts"]["messages_without_thread_id"] == 4
        first = run["thread_summaries"][0]["extracted"]
        assert first["symptom_raw"].startswith("@工程师丑 现场每日反馈")
        assert first["daily_report_signals"]
        assert first["jira_submission_signals"]
        assert first["owner_handoff_signals"]
        assert any(item["name"] == "工程师B" for item in first["attribution"]["reporter_candidates"])
        assert any(item["name"] == "工程师丑" for item in first["attribution"]["owner_candidates"])
        assert any(item["problem_category"] == "运控问题" and item["name"] == "工程师丑" for item in first["attribution"]["classification_hypotheses"])
        second = run["thread_summaries"][1]["extracted"]
        assert any(item["problem_category"] == "工控机/复判站/编程站及操作系统问题" and item["name"] == "工程师乙" for item in second["attribution"]["classification_hypotheses"])


def test_w1_imports_text_history_messages_by_chat_directory():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "history"
        _write_jsonl(root / "messages_by_chat" / "sample.jsonl", [
            {
                "message_id": "m1",
                "thread_id": "",
                "chat_id": "oc_hist",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-01 09:00",
                "sender": "工程师B",
                "sender_id": "u_fae",
                "msg_type": "text",
                "mentions": [],
                "content": "客户反馈设备拍照失败",
                "plain_text": "客户反馈设备拍照失败",
            },
        ])
        run = ChatCollectAgent().import_text_history(root)
        assert run["run_manifest"]["counts"]["messages"] == 1
        assert run["messages"][0]["thread_id"].startswith("oc_hist_")


def test_cli_extract_text_w1_writes_expected_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "history"
        out_dir = Path(tmp) / "out"
        _write_jsonl(root / "all_text_messages.jsonl", [
            {
                "message_id": "m1",
                "thread_id": "",
                "chat_id": "oc_hist",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-01 09:00",
                "sender": "工程师B",
                "sender_id": "u_fae",
                "msg_type": "text",
                "mentions": [{"name": "工程师丑"}],
                "content": "@工程师丑 客户反馈设备拍照失败",
                "plain_text": "@工程师丑 客户反馈设备拍照失败",
            },
        ])
        buf = StringIO()
        with redirect_stdout(buf):
            assert cli_main(["extract-text-w1", str(root), "--out-dir", str(out_dir)]) == 0
        out = json.loads(buf.getvalue())
        assert out["run_manifest"]["source"] == "text_jsonl_history"
        assert (out_dir / "messages.jsonl").exists()
        assert (out_dir / "thread_summaries.json").exists()
        assert (out_dir / "episodes.json").exists()
        assert (out_dir / "run_manifest.json").exists()


def test_w1_missing_info_request_carries_later_evidence_pack_for_tools():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606050900000800_202606051000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈导入程序后初始化失败，相机 IP 异常。"),
            _msg(seg, 2, "请提供配方程序文件、相机 IP 配置和 JIRA 工单。"),
            _msg(
                seg,
                3,
                "已上传 line_recipe.proj，IP 10.20.30.40，JIRA https://jira.example.com/browse/SMTAOITS-1234",
                resources="1",
            ),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [
            {
                "relative_path": "om_test_3/line_recipe.proj",
                "status": "api_ok",
                "type": "file",
                "bytes": "42",
                "name": "line_recipe.proj",
                "message_id": "om_test_3",
                "segment_id": seg,
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-05 09:03",
                "sender": "工程师丁",
                "copied": "true",
                "copied_bytes": "42",
            }
        ])
        (root / "om_test_3").mkdir(parents=True)
        (root / "om_test_3" / "line_recipe.proj").write_text("Version=4.5.6\nCameraIP=10.20.30.40\n", encoding="utf-8")
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-05T09:00:00+08:00",
                "end": "2026-06-05T10:00:00+08:00",
                "messages": "3",
                "hits": "3",
                "resources": "1",
                "uploadable": "1",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "3",
                "xing_count": "3",
            }
        ])

        episode = ChatCollectAgent().import_xing_upload(root)["episodes"][0]
        request = episode["extracted"]["missing_info_requests"][0]
        assert request["provided_later"] is True
        assert request["provided_evidence_message_ids"] == ["om_test_3"]
        provided = request["provided_evidence"][0]
        assert provided["message_id"] == "om_test_3"
        assert provided["attachment_metadata"][0]["evidence_role"] == "program_file"
        assert provided["tool_evidence"]["attachment_parse_results"][0]["evidence_role"] == "program_file"
        assert provided["tool_evidence"]["proj_parse_results"][0]["type"] == "ProjParseResult"
        assert provided["tool_evidence"]["proj_parse_results"][0]["source"]["message_id"] == "om_test_3"
        assert provided["tool_evidence"]["proj_parse_results"][0]["executed"] is False
        assert provided["tool_evidence"]["jira_parse_results"][0]["issue_keys"] == ["SMTAOITS-1234"]
        assert provided["text_hints"]["jira_ids"] == ["SMTAOITS-1234"]
        assert provided["text_hints"]["ip_config"] == ["10.20.30.40"]

        candidate = KnowledgeExtractionAgent().extract(episode)
        program = next(item for item in candidate["required_info_candidates"] if item["slot"] == "program_file")
        assert "program_file" in program["provided_tool_roles"]
        assert "proj_parsed" in program["provided_tool_roles"]
        assert "jira_issue_key" in program["provided_tool_roles"]
        assert "ip_config" in program["provided_tool_roles"]


def test_w2_uses_request_local_provided_tool_evidence_without_global_episode_tool_evidence():
    episode = {
        "episode_id": "ep:req-local-tool",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场导入配方后初始化失败，相机 IP 异常。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场导入配方后初始化失败，相机 IP 异常。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供配方程序文件、相机 IP 配置和 JIRA 工单。",
                "thread_id": "t",
                "context_before": [{"text": "现场导入配方后初始化失败，相机 IP 异常。"}],
                "context_after": [{"message_id": "m2", "text": "已上传 line_recipe.proj。"}],
                "evidence_message_ids": ["m1", "m2"],
                "provided_later": True,
                "provided_evidence_message_ids": ["m2"],
                "provided_evidence": [{
                    "message_id": "m2",
                    "content_summary": "已上传 line_recipe.proj，JIRA https://jira.example.com/browse/SMTAOITS-1234",
                    "text_hints": {
                        "jira_ids": ["SMTAOITS-1234"],
                        "versions": [],
                        "ip_config": [],
                        "log_paths": [],
                        "project_files": ["line_recipe.proj"],
                    },
                    "tool_evidence": {
                        "attachment_parse_results": [{
                            "type": "AttachmentParseResult",
                            "name": "line_recipe.proj",
                            "path": "/tmp/line_recipe.proj",
                            "evidence_role": "program_file",
                            "source": {"message_id": "m2"},
                            "content_read": False,
                            "archive_extracted": False,
                        }],
                        "jira_parse_results": [{
                            "type": "JiraParseResult",
                            "issue_keys": ["SMTAOITS-1234"],
                            "urls": [{"url": "https://jira.example.com/browse/SMTAOITS-1234", "type": "jira"}],
                            "fetched": False,
                        }],
                        "proj_parse_results": [{
                            "type": "ProjParseResult",
                            "path": "/tmp/line_recipe.proj",
                            "executed": False,
                            "mutated": False,
                            "key_hints": {"versions": ["4.5.6"], "ip_addresses": ["10.20.30.40"]},
                        }],
                        "log_package_parse_results": [],
                    },
                }],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    program = next(item for item in candidate["required_info_candidates"] if item["slot"] == "program_file")
    assert "program_file" in program["provided_tool_roles"]
    assert "proj_parsed" in program["provided_tool_roles"]
    assert "software_version" in program["provided_tool_roles"]
    assert "ip_config" in program["provided_tool_roles"]
    assert "jira_link" in program["provided_tool_roles"]
    assert "jira_issue_key" in program["provided_tool_roles"]
    assert program["quality"]["evidence_strength"] >= 0.7


def test_w1_preserves_attachment_and_jira_link_evidence_metadata_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606020900000800_202606021000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(
                seg,
                1,
                "客户反馈 AOI 导入程序文件后应用异常。"
                " [[SMTAOITS-1234] 程序文件导入异常 - Jira](https://jira.example.com/browse/SMTAOITS-1234)",
                resources="1",
            )
        ])
        _write_csv(manifest / "xing_resource_files.csv", [
            {
                "relative_path": "om_test_1/line_recipe.proj",
                "status": "api_ok",
                "type": "file",
                "bytes": "1234",
                "name": "line_recipe.proj",
                "message_id": "om_test_1",
                "segment_id": seg,
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-02 09:01",
                "sender": "工程师丁",
                "copied": "true",
                "copied_bytes": "1234",
            }
        ])
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-02T09:00:00+08:00",
                "end": "2026-06-02T10:00:00+08:00",
                "messages": "1",
                "hits": "1",
                "resources": "1",
                "uploadable": "1",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "1",
                "xing_count": "1",
            }
        ])
        run = ChatCollectAgent().import_xing_upload(root)
        summary = run["thread_summaries"][0]
        extracted = summary["extracted"]
        attachment = summary["attachments"][0]
        assert attachment["status"] == "metadata_only"
        assert attachment["evidence_role"] == "program_file"
        assert attachment["extension"] == ".proj"
        assert "line_recipe.proj" in extracted["project_files"]
        assert not extracted["log_paths"]
        assert extracted["jira_ids"] == ["SMTAOITS-1234"]
        assert extracted["jira_links"][0]["type"] == "jira"
        assert extracted["jira_links"][0]["url"] == "https://jira.example.com/browse/SMTAOITS-1234"
        assert extracted["artifacts"]["attachment_evidence"][0]["reason"] == "pre_crawled_resource_metadata_only"
        assert any(item["field"] == "project_files" and item["source"] == "attachment.name" for item in extracted["source_offsets"])
        assert any(item["field"] == "jira_links" and item["source"] == "message.raw_content.link" for item in extracted["source_offsets"])


def test_w1_hits_only_keeps_non_hit_resource_messages_in_hit_segments():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606040900000800_202606041000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈导入程序后应用异常。"),
            {
                **_msg(seg, 2, '<file key="file_proj" name="recipe.proj"/>', resources="1"),
                "is_hit": "false",
                "is_xing_related": "false",
            },
            _msg(seg, 3, "检查导入的程序文件和版本。"),
            {
                **_msg(seg, 4, "普通闲聊消息，不应因同 segment 被带入。"),
                "is_hit": "false",
                "is_xing_related": "false",
            },
        ])
        _write_csv(manifest / "xing_resource_files.csv", [
            {
                "relative_path": "om_test_2/recipe.proj",
                "status": "api_ok",
                "type": "file",
                "bytes": "20",
                "name": "recipe.proj",
                "message_id": "om_test_2",
                "segment_id": seg,
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-04 09:02",
                "sender": "工程师乙",
                "copied": "true",
                "copied_bytes": "20",
            }
        ])
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-04T09:00:00+08:00",
                "end": "2026-06-04T10:00:00+08:00",
                "messages": "3",
                "hits": "2",
                "resources": "1",
                "uploadable": "1",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "2",
                "xing_count": "2",
            }
        ])
        run = ChatCollectAgent().import_xing_upload(root, hits_only=True)
        assert run["run_manifest"]["counts"]["messages"] == 3
        assert run["run_manifest"]["counts"]["hits"] == 2
        assert all(m["message_id"] != "om_test_4" for m in run["messages"])
        extracted = run["thread_summaries"][0]["extracted"]
        assert extracted["project_files"] == ["recipe.proj"]


def test_w2_normalizes_candidate_to_kg_schema():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "upload")
        episode = ChatCollectAgent().import_xing_upload(root)["episodes"][0]
        candidate = KnowledgeExtractionAgent().extract(episode)
        node_types = {node["type"] for node in candidate["nodes"]}
        assert "SoftwareVersion" in node_types
        assert "Version" not in node_types
        assert candidate["schema_valid"] is True
        assert candidate["schema_issues"] == []
        assert _node(candidate, "Error")["error_id"].startswith("err:")
        assert _node(candidate, "DiagnosticCheck")["how_to_check"]
        assert _node(candidate, "Solution")["method"]
        assert QualityGateAgent().score(candidate)["passed"] is True
        required = candidate["required_info_candidates"]
        assert required
        assert {item["slot"] for item in required} <= {
            "log_package",
            "software_version",
            "error_phase",
            "error_message",
            "device_model",
            "site",
            "ip_config",
            "repro_steps",
            "sample_image",
            "program_file",
            "environment",
            "owner_context",
            "other",
        }
        log_req = required[0]
        assert log_req["slot"] == "log_package"
        assert log_req["merge_policy"] == "review_only"
        assert log_req["provided_later"] is True
        assert "om_test_3" in log_req["provided_evidence_message_ids"]


def test_w3_w4_required_info_normalizes_and_gates_specific_requests():
    required = {
        "candidate_id": "reqinfo:test",
        "target_error_id": "err:init",
        "slot": "other",
        "label": "DLOG",
        "question": "请提供 DLOG 诊断数据包。",
        "why_required": "用于判断初始化阶段并缩小诊断分支。",
        "condition": "",
        "evidence_message_ids": ["m1"],
        "merge_policy": "append_to_required_info",
        "source_request": {"text": "麻烦提供 DLOG 诊断数据包，用于判断初始化阶段。"},
    }
    conflict = ConflictResolutionAgent().resolve_required_info(required)
    normalized = conflict["candidate"]
    assert normalized["slot"] == "log_package"
    assert normalized["condition"] == "startup/init log"
    gate = QualityGateAgent().score_required_info(normalized)
    assert gate["passed"] is True
    generic = dict(normalized, candidate_id="reqinfo:generic", question="发日志", why_required="", condition="", evidence_message_ids=[])
    generic_gate = QualityGateAgent().score_required_info(generic)
    assert generic_gate["passed"] is False
    assert "missing_evidence" in generic_gate["issues"]




def test_w3_required_info_maps_domain_specific_slot_words_to_reader_slots():
    cases = [
        ("需要 DMP 和 WPR PoolMon 记录", "other", "log_package"),
        ("请提供显卡驱动版本", "other", "software_version"),
        ("请提供蓝屏代码和 PTE 信号", "other", "error_message"),
        ("请说明相机网卡过滤驱动绑定情况", "other", "ip_config"),
        ("请说明驱动更改后是否复发", "other", "repro_steps"),
        ("请说明现场生产约束和内存测试窗口", "other", "environment"),
    ]
    for text, slot, expected in cases:
        required = {
            "candidate_id": f"reqinfo:{expected}",
            "target_error_id": "err:test",
            "slot": slot,
            "label": text,
            "question": text,
            "why_required": "用于缩小诊断分支。",
            "condition": "",
            "evidence_message_ids": ["m1"],
            "merge_policy": "append_to_required_info",
            "source_request": {"text": text},
        }
        normalized = ConflictResolutionAgent().resolve_required_info(required)["candidate"]
        assert normalized["slot"] == expected, (text, normalized)

def test_w3_required_info_maps_domain_specific_slot_words_to_reader_slots():
    cases = [
        ("需要 DMP 和 WPR PoolMon 记录", "other", "log_package"),
        ("请提供显卡驱动版本", "other", "software_version"),
        ("请提供蓝屏代码和 PTE 信号", "other", "error_message"),
        ("请说明相机网卡过滤驱动绑定情况", "other", "ip_config"),
        ("请说明驱动更改后是否复发", "other", "repro_steps"),
        ("请说明现场生产约束和内存测试窗口", "other", "environment"),
    ]
    for text, slot, expected in cases:
        required = {
            "candidate_id": f"reqinfo:{expected}",
            "target_error_id": "err:test",
            "slot": slot,
            "label": text,
            "question": text,
            "why_required": "用于缩小诊断分支。",
            "condition": "",
            "evidence_message_ids": ["m1"],
            "merge_policy": "append_to_required_info",
            "source_request": {"text": text},
        }
        normalized = ConflictResolutionAgent().resolve_required_info(required)["candidate"]
        assert normalized["slot"] == expected, (text, normalized)


def test_w4_routes_noise_episode_to_noise_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        noise_episode = {
            "episode_id": "thread:episode:noise",
            "thread_id": "thread",
            "completeness": "noise",
            "fault_description_messages": [],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [{"message_id": "n1", "sender": {"name": "u"}, "create_time": "t", "text": "谢谢，需求排期会议后确认。", "content_summary": "谢谢，需求排期会议后确认。"}],
            "evidence_message_ids": [],
            "source_offsets": [],
            "attachments": [],
            "extracted": {"symptom_raw": "谢谢，需求排期会议后确认。"},
        }
        out = WriteSidePipeline(store).run_summaries([{"thread_id": "thread", "episodes": [noise_episode]}], emit_episodes=True)
        assert out["review_summary"]["noise_candidates"] == 1
        queued = store.read_review_queue("noise_candidates.json")
        assert queued[0]["queue"] == "noise_candidates"
        assert "candidate" in queued[0]
        assert "episode" in queued[0]
        assert "conflict" in queued[0]
        assert "quality_gate" in queued[0]
        assert "evidence_pack" in queued[0]
        assert "noise_episode" in queued[0]["quality_gate"]["issues"]


def test_write_side_pipeline_queues_review_items_without_graph_mutation_and_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "upload")
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        out = WriteSidePipeline(store).run_xing_upload(root, emit_episodes=True, dry_run_merge=True, w2_workers=2)
        assert out["summary"]["summaries"] == 1
        assert out["summary"]["episodes"] == 2
        assert out["summary"]["candidates"] == 2
        assert out["summary"]["required_info_candidates"] >= 1
        assert out["review_summary"]["candidates"] == 1
        assert out["review_summary"]["noise_candidates"] == 1
        assert out["review_summary"]["ask_info_candidates"] >= 1
        assert out["review_summary"]["dry_run_merge_plans"] == 2
        queued = store.read_review_queue("candidates.json")
        assert len(queued) == 1
        item = queued[0]
        assert set(item) >= {"review_id", "queue", "candidate", "episode", "conflict", "quality_gate", "evidence_pack", "review_actions"}
        assert item["review_actions"] == ["approve", "reject", "merge_existing", "request_more_info"]
        assert item["evidence_pack"]["messages"]
        assert item["dry_run_merge_plan"]["status"] == "dry_run_merge_plan"
        assert item["candidate"]["type"] == "SchemaValidCandidate"
        WriteSidePipeline(store).run_xing_upload(root, emit_episodes=True, dry_run_merge=True)
        assert len(store.read_review_queue("candidates.json")) == 1
        ask_items = store.read_review_queue("ask_info_candidates.json")
        assert ask_items
        ask_item = ask_items[0]
        assert set(ask_item) >= {"review_id", "queue", "required_info_candidate", "episode", "quality_gate", "evidence_pack", "review_actions"}
        assert ask_item["queue"] == "ask_info_candidates"
        assert ask_item["review_actions"] == ["accept", "merge", "drop", "needs_owner", "needs_better_evidence"]
        assert len(store.read_review_queue("ask_info_candidates.json")) == len(ask_items)
        assert not (kg / "edges.json").exists()


def test_w6_review_item_contains_tool_evidence_for_project_and_jira_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606030900000800_202606031000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(
                seg,
                1,
                "客户反馈 AOI 导入程序文件后应用异常。"
                " [[SMTAOITS-1234] 程序文件导入异常 - Jira](https://jira.example.com/browse/SMTAOITS-1234)",
                resources="1",
            ),
            _msg(seg, 2, "检查导入的程序文件和相机配置。"),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [
            {
                "relative_path": "om_test_1/line_recipe.proj",
                "status": "api_ok",
                "type": "file",
                "bytes": "42",
                "name": "line_recipe.proj",
                "message_id": "om_test_1",
                "segment_id": seg,
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "create_time": "2026-06-03 09:01",
                "sender": "工程师丁",
                "copied": "true",
                "copied_bytes": "42",
            }
        ])
        (root / "om_test_1").mkdir(parents=True)
        (root / "om_test_1" / "line_recipe.proj").write_text("Version=1.3.5\nCameraIP=192.168.1.10\n", encoding="utf-8")
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-03T09:00:00+08:00",
                "end": "2026-06-03T10:00:00+08:00",
                "messages": "2",
                "hits": "2",
                "resources": "1",
                "uploadable": "1",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "2",
                "xing_count": "2",
            }
        ])
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        WriteSidePipeline(store).run_xing_upload(root, emit_episodes=True, dry_run_merge=True)
        queued = (
            store.read_review_queue("candidates.json")
            or store.read_review_queue("merge_candidates.json")
            or store.read_review_queue("noise_candidates.json")
        )
        assert queued
        evidence = queued[0]["evidence_pack"]["tool_evidence"]
        assert evidence["attachment_parse_results"][0]["evidence_role"] == "program_file"
        assert evidence["jira_parse_results"][0]["issue_keys"] == ["SMTAOITS-1234"]
        assert evidence["proj_parse_results"][0]["type"] == "ProjParseResult"
        assert evidence["proj_parse_results"][0]["executed"] is False


def test_bare_jira_issue_key_flows_from_w1_to_w2_and_w6_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_test_202606031100000800_202606031200000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈 AOI 初始化失败，已提 SMTAOITS-1234 跟踪。"),
            _msg(seg, 2, "检查初始化日志并确认主程序版本。"),
            _msg(seg, 3, "已解决，恢复正常，原因是版本配置不一致。"),
        ])
        (manifest / "xing_resource_files.csv").write_text(
            "relative_path,status,type,bytes,name,message_id,segment_id,chat_name,create_time,sender,copied,copied_bytes\n",
            encoding="utf-8",
        )
        _write_csv(manifest / "xing_segments.csv", [
            {
                "segment_id": seg,
                "index": "1",
                "chat_id": "oc_test",
                "chat_name": "【1.0已签单】江苏客户01项目群",
                "start": "2026-06-03T11:00:00+08:00",
                "end": "2026-06-03T12:00:00+08:00",
                "messages": "3",
                "hits": "3",
                "resources": "0",
                "uploadable": "0",
                "unavailable": "0",
                "xing_rule_ok": "true",
                "xing_before_first": "0",
                "xing_after_last": "0",
                "xing_total": "3",
                "xing_count": "3",
            }
        ])

        run = ChatCollectAgent().import_xing_upload(root)
        episode = run["episodes"][0]
        extracted = episode["extracted"]
        assert extracted["jira_ids"] == ["SMTAOITS-1234"]
        assert extracted["tool_evidence"]["jira_parse_results"][0]["issue_keys"] == ["SMTAOITS-1234"]

        candidate = KnowledgeExtractionAgent().extract(episode)
        assert candidate["jira_ids"] == ["SMTAOITS-1234"]

        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        WriteSidePipeline(store).run_xing_upload(root, emit_episodes=True, dry_run_merge=True)
        queued = store.read_review_queue("candidates.json")
        assert queued
        evidence = queued[0]["evidence_pack"]["tool_evidence"]
        assert evidence["jira_parse_results"][0]["issue_keys"] == ["SMTAOITS-1234"]
        assert evidence["observability"]["tool_evidence"]["jira_links"] >= 1


def test_w2_candidate_surfaces_image_environment_and_data_attachments_as_evidence_fields():
    episode = {
        "episode_id": "ep-attachment-evidence",
        "thread_id": "t-attachment-evidence",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户反馈 AOI 漏检，需要看样本图和环境配置。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查样本图、环境配置和导出的数据文件。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "attachments": [
            {"file_key": "m1/capture.webp", "name": "capture.webp", "kind": "image", "message_id": "m1"},
            {"file_key": "m1/config.toml", "name": "config.toml", "kind": "file", "message_id": "m1"},
            {"file_key": "m1/export.csv", "name": "export.csv", "kind": "file", "message_id": "m1"},
        ],
        "extracted": {
            "symptom_raw": "客户反馈 AOI 漏检，需要看样本图和环境配置。",
            "debug_actions": ["检查样本图、环境配置和导出的数据文件。"],
            "conclusion": "",
            "tool_evidence": {
                "attachment_parse_results": [
                    {"type": "AttachmentParseResult", "name": "capture.webp", "evidence_role": "sample_image", "content_read": False},
                    {"type": "AttachmentParseResult", "name": "config.toml", "evidence_role": "environment", "content_read": False},
                    {"type": "AttachmentParseResult", "name": "export.csv", "evidence_role": "data_file", "content_read": False},
                ],
                "jira_parse_results": [],
                "proj_parse_results": [],
                "log_package_parse_results": [],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["sample_images"] == ["capture.webp"]
    assert candidate["environment_files"] == ["config.toml"]
    assert candidate["data_files"] == ["export.csv"]
    assert candidate["tool_evidence"]["attachment_parse_results"][0]["content_read"] is False


def test_w4_routes_project_report_dominant_candidate_to_review_only_noise():
    text = (
        "各位领导晚上好，今日现场工作汇报：1.客户带板测试满意，直接签单，签单设备SI-2020C。"
        "2.设备交付信息已替换，客户地址已确认。3.现场需要培训客户人员，项目进度已更新。"
        "4.客户反馈偶发卡顿，后续继续观察。以上请领导知悉。"
    )
    candidate = {
        "candidate_id": "chatcand:project-report",
        "label": text,
        "symptom_raw": text,
        "conclusion": "",
        "category": "系统与软件异常",
        "confidence": 0.8,
        "evidence_ids": ["m1"],
        "source_offsets": [{"message_id": "m1", "field": "sites", "value": "x"}],
        "nodes": [
            {"type": "Error", "error_id": "err:candidate-report"},
            {"type": "DiagnosticCheck", "check_id": "check:candidate-report:1"},
        ],
        "edges": [{"from": "err:candidate-report", "to": "check:candidate-report:1", "relation": "has_check"}],
        "schema_valid": True,
        "episode": {
            "completeness": "partial",
            "fault_description_messages": [{"message_id": "m1", "text": text}],
            "diagnostic_chain_messages": [{"message_id": "m1", "text": "项目进度已更新"}],
            "resolution_messages": [],
        },
    }
    gate = QualityGateAgent().score(candidate)
    assert gate["passed"] is False
    assert "review_only_noise" in gate["issues"]


def test_w4_keeps_report_with_concrete_root_cause_and_solution_reviewable():
    text = (
        "今日现场工作汇报：客户反馈复判站无检测图数据异常，检查日志后定位到配置文件损坏，"
        "已解决，恢复正常。"
    )
    candidate = {
        "candidate_id": "chatcand:report-with-root-cause",
        "label": "复判站无检测图数据异常",
        "symptom_raw": "客户反馈复判站无检测图数据异常",
        "conclusion": "检查日志后定位到配置文件损坏，已解决，恢复正常。",
        "debug_actions": ["检查日志和配置文件"],
        "category": "系统与软件异常",
        "confidence": 0.82,
        "evidence_ids": ["m1"],
        "source_offsets": [{"message_id": "m1", "field": "log_paths", "value": "startup.log"}],
        "log_paths": ["startup.log"],
        "nodes": [
            {"type": "Error", "error_id": "err:candidate-report-root-cause"},
            {"type": "DiagnosticCheck", "check_id": "check:candidate-report-root-cause:1"},
            {"type": "Solution", "solution_id": "solution:candidate-report-root-cause:1"},
        ],
        "edges": [
            {"from": "err:candidate-report-root-cause", "to": "check:candidate-report-root-cause:1", "relation": "has_check"},
            {"from": "check:candidate-report-root-cause:1", "to": "solution:candidate-report-root-cause:1", "relation": "resolved_by"},
        ],
        "schema_valid": True,
        "episode": {
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "text": text}],
            "diagnostic_chain_messages": [{"message_id": "m1", "text": "检查日志和配置文件"}],
            "resolution_messages": [{"message_id": "m1", "text": "已解决，恢复正常"}],
        },
    }
    gate = QualityGateAgent().score(candidate)
    assert gate["passed"] is True
    assert "review_only_noise" not in gate["issues"]


def test_w2_label_strips_leading_mentions_and_keeps_fault_fact():
    episode = {
        "episode_id": "ep-mention-prefix",
        "thread_id": "t-mention-prefix",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "@工程师乙 工程师乙，麻烦帮忙看下这台复判站问题点，重启以后不关机，正常测试的时候任务管理器设置页面打不开。",
        }],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "打开任务管理器确认进程占用。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [{"message_id": "m1", "field": "sites", "value": "x"}],
        "extracted": {
            "symptom_raw": "@工程师乙 工程师乙，麻烦帮忙看下这台复判站问题点，重启以后不关机，正常测试的时候任务管理器设置页面打不开。",
            "debug_actions": ["打开任务管理器确认进程占用。"],
            "conclusion": "",
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert "工程师乙" not in candidate["label"]
    assert "重启以后不关机" in candidate["label"]
    assert candidate["label"] != "群聊噪声/待人工确认"


def test_w4_rejects_person_only_label_even_with_actions():
    candidate = {
        "candidate_id": "chatcand:person-only-label",
        "label": "工程师乙",
        "symptom_raw": "工程师乙",
        "debug_actions": ["打开磁盘管理截个图发我", "能打开任务管理器吗"],
        "category": "系统与软件异常",
        "confidence": 0.7,
        "evidence_ids": ["m1"],
        "source_offsets": [{"message_id": "m1", "field": "sites", "value": "x"}],
        "nodes": [
            {"type": "Error", "error_id": "err:person-only"},
            {"type": "DiagnosticCheck", "check_id": "check:person-only:1"},
        ],
        "edges": [{"from": "err:person-only", "to": "check:person-only:1", "relation": "has_check"}],
        "schema_valid": True,
        "episode": {"completeness": "partial"},
    }
    gate = QualityGateAgent().score(candidate)
    assert gate["passed"] is False
    assert "weak_person_only_label" in gate["issues"]


def test_w4_does_not_reject_short_fault_label_as_person_only():
    candidate = {
        "candidate_id": "chatcand:short-fault-label",
        "label": "蓝屏重启",
        "symptom_raw": "蓝屏重启",
        "debug_actions": ["导出 dmp 文件"],
        "category": "系统与软件异常",
        "confidence": 0.75,
        "evidence_ids": ["m1"],
        "source_offsets": [{"message_id": "m1", "field": "log_paths", "value": "MEMORY.DMP"}],
        "log_paths": ["MEMORY.DMP"],
        "nodes": [
            {"type": "Error", "error_id": "err:short-fault"},
            {"type": "DiagnosticCheck", "check_id": "check:short-fault:1"},
        ],
        "edges": [{"from": "err:short-fault", "to": "check:short-fault:1", "relation": "has_check"}],
        "schema_valid": True,
        "episode": {"completeness": "partial"},
    }
    gate = QualityGateAgent().score(candidate)
    assert "weak_person_only_label" not in gate["issues"]


def test_w2_label_prefers_fault_message_over_coordination_reply():
    episode = {
        "episode_id": "ep-label-coordination",
        "thread_id": "t-label-coordination",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": (
                "客户目前比较关心4项问题：1/优化改进大批量备份导出到其他盘，"
                "2/系统还原一键ghost，3/误报调试时间长，4/QFP内存卡槽虚焊问题。"
            ),
        }],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "@左璐 收到璐哥，我来协调，该客户已进入付款流程"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [{"message_id": "m1", "field": "sites", "value": "x"}],
        "extracted": {
            "symptom_raw": "收到璐哥，我来协调，该客户已进入付款流程",
            "debug_actions": ["商务积极沟通中，问题点 1/优化改进大批量备份导出到其他盘"],
            "conclusion": "",
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert "付款流程" not in candidate["label"]
    assert "误报调试时间长" in candidate["label"] or "内存卡槽虚焊" in candidate["label"]


def test_w2_label_prefers_fault_summary_over_handoff_instruction():
    episode = {
        "episode_id": "ep-label-handoff",
        "thread_id": "t-label-handoff",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": (
                "各位领导，客户16设备D68拍照失败排查总结："
                "按照SOP排查处理后拍照失败未消失，更新 intel MEI 驱动并断电重启后拍照失败消失。"
            ),
        }],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "如沟通，到了先做这三件事：重传间隔设置检查一下，把CPU和内存监控打开"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [{"message_id": "m1", "field": "log_paths", "value": "DLOG.zip"}],
        "extracted": {
            "symptom_raw": "如沟通",
            "debug_actions": ["重传间隔设置检查一下", "把CPU和内存监控打开"],
            "conclusion": "",
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["label"] != "如沟通"
    assert "拍照失败" in candidate["label"]


def test_existing_error_condition_branch_routes_to_merge_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "upload")
        kg = Path(tmp) / "kg"
        (kg / "instances" / "errors").mkdir(parents=True)
        (kg / "review_queue").mkdir(parents=True)
        (kg / "instances" / "errors" / "errors.json").write_text(
            '[{"error_id":"err:camera-ip","label":"相机IP异常 初始化失败","symptom":"AOI 主程序初始化失败 相机连接异常","category":"硬件与运控","keywords":["相机","初始化","IP"]}]\n',
            encoding="utf-8",
        )
        store = JsonKGStore(kg)
        out = WriteSidePipeline(store, match_threshold=0.1).run_xing_upload(root, emit_episodes=True, dry_run_merge=True)
        assert out["review_summary"]["merge_candidates"] >= 1
        queued = store.read_review_queue("merge_candidates.json")
        assert queued
        assert queued[0]["conflict"]["existing_error_id"] == "err:camera-ip"
        assert queued[0]["dry_run_merge_plan"]["affects_existing_check_chain"] is True
        ask_items = store.read_review_queue("ask_info_candidates.json")
        assert ask_items
        assert ask_items[0]["required_info_candidate"]["target_error_id"] == "err:camera-ip"


def test_w5_approved_required_info_merge_only_updates_error_required_info():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "instances" / "errors").mkdir(parents=True)
        (kg / "review_queue").mkdir(parents=True)
        error_path = kg / "instances" / "errors" / "errors.json"
        error_path.write_text('[{"error_id":"err:init","label":"初始化失败","required_info":[]}]\n', encoding="utf-8")
        store = JsonKGStore(kg)
        item = {
            "review_status": "approved",
            "required_info_candidate": {
                "candidate_id": "reqinfo:init-log",
                "target_error_id": "err:init",
                "slot": "log_package",
                "question": "请提供启动/初始化阶段的 DLOG 或诊断数据包。",
                "merge_policy": "append_to_required_info",
                "source_episode_id": "ep1",
                "evidence_message_ids": ["m1", "m2"],
            },
        }
        result = IncrementalIngestAgent(store).apply_approved_required_info(item)
        assert result["status"] == "required_info_applied"
        data = json.loads(error_path.read_text(encoding="utf-8"))
        assert data[0]["required_info"] == ["请提供启动/初始化阶段的 DLOG 或诊断数据包。"]
        assert data[0]["required_info_sources"]["log_package"]["occurrence_count"] == 1
        assert data[0]["required_info_sources"]["log_package"]["applied_candidate_ids"] == ["reqinfo:init-log"]
        repeat = IncrementalIngestAgent(store).apply_approved_required_info(item)
        assert repeat["status"] == "required_info_already_applied"
        data = json.loads(error_path.read_text(encoding="utf-8"))
        assert data[0]["required_info_sources"]["log_package"]["occurrence_count"] == 1


def test_ask_info_scenario_builder_uses_review_queue_items():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "upload")
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        WriteSidePipeline(store).run_xing_upload(root, emit_episodes=True, dry_run_merge=True)
        scenarios = build_scenarios(store.read_review_queue("ask_info_candidates.json"), limit=30)
        assert scenarios
        assert scenarios[0]["expected_status"] == "ask_info"
        assert scenarios[0]["required_info"]
        assert "设备出现故障，需要诊断；当前缺少完整现象和诊断资料" not in scenarios[0]["query"]
        assert "已提供 DLOG 诊断数据包" in scenarios[0]["user_turns"][0]["reply"]
        assert "MEMORY.DMP" not in scenarios[0]["user_turns"][0]["reply"]
        assert any(term.startswith("slot:") for term in scenarios[0]["required_info"])
        assert any("question:" in term for term in scenarios[0]["required_info"])


def test_ask_info_scenario_builder_strips_raw_resource_placeholders():
    item = {
        "review_id": "review:reqinfo:image",
        "queue": "ask_info_candidates",
        "required_info_candidate": {
            "candidate_id": "reqinfo:image",
            "target_error_id": "err:image",
            "acceptable_error_ids": ["err:image"],
            "slot": "sample_image",
            "label": "样本图",
            "question": "请提供样本图。",
            "why_required": "分析重传失败的具体表现",
            "condition": "重传失败发生",
            "merge_policy": "append_to_required_info",
            "evidence_message_ids": ["m1"],
        },
        "quality_gate": {"passed": True},
        "episode": {
            "episode_id": "ep:image",
            "thread_id": "thread:image",
            "fault_description_messages": [
                {"message_id": "m1", "content_summary": "[Image: img_v3_02qv_3b3529a5-6c38-4644-84df-5a0c1241e9ag] 看着还是重传失败的问题"}
            ],
            "diagnostic_chain_messages": [
                {"message_id": "m2", "content_summary": "已上传 [File: file_v3_abcdef1234567890.zip] [Media: f"}
            ],
        },
    }
    scenario = build_scenarios([item], limit=1)[0]
    dumped = json.dumps(scenario, ensure_ascii=False)
    assert "img_v3_" not in dumped
    assert "file_v3_" not in dumped
    assert "[Image:" not in dumped
    assert "[Media:" not in dumped
    assert "截图" in scenario["query"]


def test_loop_agents_make_review_candidates():
    feedback = DiagnosticFeedbackAgent().build_candidate({"case_id": "x"})
    assert feedback["status"] == "pending_review"
    assert ATRWeightingAgent().propose(feedback)["type"] == "ATRWeightProposal"
    assert LogPatternAgent().propose({"pattern": "camera timeout"})["type"] == "LogPatternCandidate"

def test_w3_does_not_override_explicit_required_info_slot_without_rewriting_question():
    required = {
        "candidate_id": "reqinfo:phase-dmp-context",
        "target_error_id": "err:ipc-reboot",
        "slot": "error_phase",
        "label": "故障发生阶段",
        "question": "请说明故障发生在启动、初始化、扫码、检测还是复判阶段。",
        "why_required": "用于判断故障发生阶段。",
        "condition": "",
        "evidence_message_ids": ["m1"],
        "merge_policy": "append_to_required_info",
        "source_request": {"text": "蓝屏重启后，请说明故障发生阶段。"},
    }
    conflict = ConflictResolutionAgent().resolve_required_info(required)
    normalized = conflict["candidate"]
    assert normalized["slot"] == "error_phase"
    assert normalized["question"] == required["question"]
    assert normalized["condition"] == "dmp"


def test_ask_info_scenario_generator_does_not_overwrite_out_on_empty_queue_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        queue = Path(tmp) / "empty_queue.json"
        out = Path(tmp) / "scenarios.json"
        queue.write_text("[]\n", encoding="utf-8")
        out.write_text("[{\"case_id\":\"keep\"}]\n", encoding="utf-8")
        code = ask_info_candidates_main(["--queue", str(queue), "--out", str(out), "--limit", "30"])
        assert code == 1
        assert json.loads(out.read_text(encoding="utf-8")) == [{"case_id": "keep"}]
        code = ask_info_candidates_main(["--queue", str(queue), "--out", str(out), "--limit", "30", "--allow-empty"])
        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8")) == []



def test_read_side_sufficiency_gate_asks_for_explicit_missing_slot_before_retrieval_confidence():
    from debug_agent_system.agents.read.c_sufficiency import SufficiencyGate
    from debug_agent_system.core.contracts import Candidate, LockedSubgraph

    candidate = Candidate(error_id="err:x", label="蓝屏重启", score=20.0, route="test", evidence=[], payload={})
    gate = SufficiencyGate().decide("现场反馈：设备蓝屏重启；当前缺少蓝屏或重启对应的 dmp/系统日志。", [candidate], None)
    assert gate.sufficient is False
    assert "诊断数据包/日志" in gate.required_info
    assert gate.reason == "explicit_missing_required_info"
    gate = SufficiencyGate().decide("现场反馈：设备蓝屏；当前缺少蓝屏错误代码。", [candidate], None)
    assert gate.sufficient is False
    assert "故障现象/完整报错文本" in gate.required_info
    assert gate.reason == "explicit_missing_required_info"
    subgraph = LockedSubgraph(
        error_id="err:x",
        label="蓝屏",
        required_info=[
            "拍摄蓝屏界面，记录 Stop Code、错误文件名、二维码或故障模块信息",
            "导出 C:\\Windows\\Minidump 或 MEMORY.DMP",
            "记录蓝屏发生频率、复现步骤和触发场景",
            "蓝屏的具体错误代码是什么？",
        ],
    )
    gate = SufficiencyGate(max_required_items=3).decide("现场反馈：设备蓝屏；当前缺少蓝屏错误代码。", [candidate], subgraph)
    assert gate.sufficient is False
    assert gate.required_info[0] == "蓝屏的具体错误代码是什么？"


def test_w1_does_not_mark_fault_report_with_normal_word_as_resolution():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "fae",
            "content": "正常测试中突然重启，从软件日志上看，没有检测到原因，麻烦从硬件上看一下是什么问题。",
        },
        {"message_id": "m2", "thread_id": "t", "sender": "dev", "content": "请提供windows日志中的Bugcheck错误截图。"},
    ])
    episode = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"][0]
    assert episode["resolution_messages"] == []
    assert episode["completeness"] == "partial"


def test_w2_candidate_text_is_review_grade_not_raw_chat_dump():
    episode = {
        "episode_id": "ep:noisy-text",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "@工程师庚 [[SMTAOITS-1234] 1.3.5 客户02 设备报错“应用异常”，之后闪退 - Jira](https://jira/x) [Image: img_noise]",
            "content_summary": "@工程师庚 [[SMTAOITS-1234] 1.3.5 客户02 设备报错“应用异常”，之后闪退 - Jira](https://jira/x) [Image: img_noise]",
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m2",
            "text": "@工程师癸 搜索U37器件之后，双击查看详情的时候闪退 [Media: file_noise]",
            "content_summary": "@工程师癸 搜索U37器件之后，双击查看详情的时候闪退 [Media: file_noise]",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "@工程师庚 [[SMTAOITS-1234] 1.3.5 客户02 设备报错“应用异常”，之后闪退 - Jira](https://jira/x) [Image: img_noise]",
            "debug_actions": ["@工程师癸 搜索U37器件之后，双击查看详情的时候闪退 [Media: file_noise]"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    error = _node(candidate, "Error")
    check = _node(candidate, "DiagnosticCheck")
    for value in (candidate["label"], error["label"], error["symptom"], check["label"], check["how_to_check"]):
        assert "@" not in value
        assert "[Image:" not in value
        assert "[Media:" not in value
        assert "http" not in value
        assert "Jira](" not in value
    assert len(error["label"]) <= 80
    assert len(check["label"]) <= 120
    assert "设备报错“应用异常”，之后闪退" in error["symptom"]
    assert "搜索U37器件之后" in check["how_to_check"]


def test_w2_fault_label_filters_resource_ids_raw_log_lines_and_weak_actions():
    noisy_fault = "那属于是路由配置的问题，导致连EAP的网络也尝试走你的无线网卡了。正确的网络配置可以避免这样的问题。江苏客户01 江苏客户01 img_v3_02p5_088b5036-56af-4022-adb2-076a59219b0g"
    candidate = KnowledgeExtractionAgent().extract({
        "episode_id": "ep:label-resource-id",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": noisy_fault, "content_summary": noisy_fault}],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "text": "检查有线网段和无线网卡路由配置。", "content_summary": "检查有线网段和无线网卡路由配置。"},
            {"message_id": "m3", "text": "你重启下", "content_summary": "你重启下"},
        ],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": noisy_fault,
            "debug_actions": ["检查有线网段和无线网卡路由配置。", "你重启下"],
            "missing_info_requests": [],
            "tool_evidence": {
                "image_parse_results": [{
                    "name": "截图.webp",
                    "image_format": "webp",
                    "width": 750,
                    "height": 406,
                    "header_read": True,
                    "ocr_performed": False,
                    "pixels_read": False,
                }]
            },
        },
    })
    assert "img_v3" not in candidate["label"]
    assert "江苏客户01 江苏客户01" not in candidate["label"]
    assert "webp" not in candidate["label"]
    assert "750x406" not in candidate["label"]
    assert "你重启下" not in candidate["label"]
    assert "路由配置" in candidate["label"]

    log_line = "1. 2025-10-27 13:48:04,259 ACME.qt.rc.win 0x00005b48 DEBUG - ResourceController.cpp:127 OCR finished timeUsage:30708.107ms Total runInspect:80319.09"
    log_candidate = KnowledgeExtractionAgent().extract({
        "episode_id": "ep:label-raw-log-line",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": log_line, "content_summary": log_line}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {"symptom_raw": log_line, "debug_actions": [], "missing_info_requests": []},
    })
    assert log_candidate["label"] == "群聊噪声/待人工确认"
    assert QualityGateAgent().score(log_candidate)["passed"] is False

    weak_candidate = KnowledgeExtractionAgent().extract({
        "episode_id": "ep:label-weak-action",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "你重启下", "content_summary": "你重启下"}],
        "diagnostic_chain_messages": [{"message_id": "m1", "text": "你重启下", "content_summary": "你重启下"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {"symptom_raw": "你重启下", "debug_actions": ["你重启下"], "missing_info_requests": []},
    })
    assert weak_candidate["label"] == "群聊噪声/待人工确认"
    assert QualityGateAgent().score(weak_candidate)["passed"] is False

    image_only_candidate = KnowledgeExtractionAgent().extract({
        "episode_id": "ep:label-image-metadata-only",
        "thread_id": "t",
        "completeness": "noise",
        "fault_description_messages": [],
        "diagnostic_chain_messages": [{"message_id": "m1", "text": "[Image: img_v3_0210h_222]", "content_summary": "[Image: img_v3_0210h_222]"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "sites": ["泰国朗特"],
            "tool_evidence": {
                "image_parse_results": [{
                    "name": "截图.webp",
                    "image_format": "webp",
                    "width": 750,
                    "height": 563,
                    "header_read": True,
                    "ocr_performed": False,
                    "pixels_read": False,
                }]
            },
            "debug_actions": [],
            "missing_info_requests": [],
        },
    })
    assert image_only_candidate["label"] == "群聊噪声/待人工确认"
    assert QualityGateAgent().score(image_only_candidate)["passed"] is False


def test_w1_does_not_treat_pure_handoff_as_diagnostic_check():
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈设备开机黑屏。"},
        {"message_id": "m2", "thread_id": "t", "sender": "fae", "content": "麻烦工程师乙，许老师排查一下原因。"},
    ])
    episode = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"][0]
    assert episode["diagnostic_chain_messages"] == []
    assert episode["completeness"] == "partial"


def test_w1_does_not_duplicate_fault_reboot_or_provided_artifact_as_diagnostic_check():
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈正常测试中突然蓝屏重启，软件日志未发现原因。"},
        {"message_id": "m2", "thread_id": "t", "sender": "dev", "content": "请提供Windows日志中的BugCheck错误截图。"},
        {"message_id": "m3", "thread_id": "t", "sender": "fae", "content": "已上传DLOG_AOI.zip和蓝屏截图。", "attachments": [{"name": "DLOG_AOI.zip", "kind": "file"}]},
        {"message_id": "m4", "thread_id": "t", "sender": "dev", "content": "先检查Windows事件查看器中的BugCheck记录。"},
    ])
    episode = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"][0]
    assert [m["message_id"] for m in episode["fault_description_messages"]] == ["m1"]
    assert [m["message_id"] for m in episode["diagnostic_chain_messages"]] == ["m4"]
    assert "m3" in episode["evidence_message_ids"]
    requests = episode["extracted"]["missing_info_requests"]
    assert requests[0]["message_id"] == "m2"
    assert requests[0]["provided_later"] is True
    assert "m3" in requests[0]["provided_evidence_message_ids"]


def test_w1_splits_single_message_numbered_multi_fault_report_into_episodes():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "fae",
            "content": (
                "今日异常点汇报："
                " 1. 检测出结果计算时间长 - 重新编程，关闭整版异物检测没有效果。"
                " 2. 外置网卡上的两个网口重启电脑后IP互换 - 更新网卡驱动无效。"
                " 3. 复判界面点智能调试等待时间长 - 异常日志已反馈售后技术支持。"
            ),
        }
    ])
    episodes = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"]
    assert len(episodes) == 3
    texts = [" ".join(m["text"] for m in ep["fault_description_messages"]) for ep in episodes]
    assert "检测出结果计算时间长" in texts[0]
    assert "IP互换" in texts[1]
    assert "等待时间长" in texts[2]
    assert [ep["fault_description_messages"][0]["source_message_id"] for ep in episodes] == ["m1", "m1", "m1"]
    assert [ep["fault_description_messages"][0]["fragment_index"] for ep in episodes] == [1, 2, 3]


def test_w1_does_not_split_numbered_root_cause_or_solution_list_as_multi_faults():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "dev",
            "content": (
                "从转存储文件来看："
                " 1. 驱动文件丢失/损坏，多个关键驱动无法加载。"
                " 2. 内存转储不完整，Page not present in dump。"
                " 3. 可疑驱动存在异常。"
                " 解决步骤：运行 sfc /scannow，重装显卡驱动。"
            ),
        }
    ])
    episodes = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["fault_description_messages"][0]["fragment_index"] is None


def test_w1_splits_only_fault_section_of_daily_work_report():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "fae",
            "content": (
                "各位领导晚上好，今日情况如下： 一.现场工作："
                " 1.收集测试数据上传网盘。"
                " 二.现场异常点 1.员工反馈出结果慢，已跟技术员解释原因。"
                " 2.主程序闪退，已收集相关数据反馈研发。以上请领导知悉。"
            ),
        }
    ])
    episodes = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"]
    assert len(episodes) == 2
    texts = [episode["fault_description_messages"][0]["text"] for episode in episodes]
    assert "出结果慢" in texts[0]
    assert "主程序闪退" in texts[1]
    assert all("收集测试数据上传网盘" not in text for text in texts)


def test_w1_splits_leadership_problem_summary_report_by_heading():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "fae",
            "content": (
                "升级1.3.3版本现场问题汇总 一、软件功能异常问题 "
                "3. 卡顿后出现拍摄失败，出现时段15:30左右。"
                "二、设备硬件异常问题 远轨间断出现2次板卡停在轨道中间、无法正常出板问题。"
                "以上信息请各位领导知悉！"
            ),
        }
    ])
    episodes = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"]
    assert len(episodes) == 2
    texts = [episode["fault_description_messages"][0]["text"] for episode in episodes]
    assert "拍摄失败" in texts[0]
    assert "无法正常出板" in texts[1]


def test_w1_splits_nested_software_and_hardware_problem_summary():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "fae",
            "content": (
                "升级1.3.3版本现场问题汇总 "
                "一、软件功能异常问题 "
                "1. 操作响应延迟，智能调整等待5–10秒。 "
                "2. 调试误报闪退，界面卡顿后直接闪退。 "
                "3. 卡顿后出现拍摄失败，正常复判时弹出拍摄失败。 "
                "二、设备硬件异常问题 远轨宽度异常导致板卡卡滞、无法正常出板。"
                "以上信息请各位领导知悉！"
            ),
        }
    ])

    episodes = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"]

    assert len(episodes) == 4
    texts = [episode["fault_description_messages"][0]["text"] for episode in episodes]
    assert "操作响应延迟" in texts[0]
    assert "调试误报闪退" in texts[1]
    assert "拍摄失败" in texts[2]
    assert "无法正常出板" in texts[3]


def test_w1_separates_concurrent_distinct_fault_topics_without_progress_messages():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m-camera",
            "thread_id": "t-concurrent",
            "sender": "fae",
            "content": "客户现场正常复判时频繁弹出拍摄失败。",
        },
        {
            "message_id": "m-blue-screen",
            "thread_id": "t-concurrent",
            "sender": "fae",
            "content": "同一时间另一台工控机出现蓝屏并自动重启。",
        },
    ])

    episodes = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"]

    assert len(episodes) == 2
    assert "拍摄失败" in episodes[0]["fault_description_messages"][0]["text"]
    assert "蓝屏" in episodes[1]["fault_description_messages"][0]["text"]


def test_w1_missing_info_filters_daily_report_and_version_upgrade_question():
    messages = ChatCollectAgent().normalize_messages([
        {
            "message_id": "m1",
            "thread_id": "t",
            "sender": "fae",
            "content": "客户反馈设备蓝屏重启。",
        },
        {
            "message_id": "m2",
            "thread_id": "t",
            "sender": "fae",
            "content": "各位领导晚上好，今日现状已更新，请查阅。一、现场工作汇总：升级两台设备到0.27.53版本，日常数据收集上传工作，收集显卡驱动版本及windows系统版本。",
        },
        {
            "message_id": "m3",
            "thread_id": "t",
            "sender": "fae",
            "content": "现在软件版本是0.27.10，我现在直接更新到0.27.12版本可以吗？",
        },
        {
            "message_id": "m4",
            "thread_id": "t",
            "sender": "dev",
            "content": "请提供windows日志中的Bugcheck错误截图，截图发给我看看。",
        },
    ])
    episode = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"][0]
    requests = episode["extracted"]["missing_info_requests"]
    assert [item["message_id"] for item in requests] == ["m4"]


def test_w1_missing_info_filters_work_report_upload_to_jira_and_rnd():
    report = (
        "各位领导，晚上好：深圳客户现场2025年12月22日工作汇报。"
        "现场工作：1.处理客户反馈复判卡顿。2.软件版本由0.27.29升级至0.27.44。"
        "现场异常：J1元件识别不准，收集数据上传至jira；设备闪退数据提供给研发老师排查。以上请领导知悉。"
    )
    messages = ChatCollectAgent().normalize_messages([
        {"message_id": "m1", "thread_id": "t", "sender": "fae", "content": "客户反馈复判站卡顿。"},
        {"message_id": "m2", "thread_id": "t", "sender": "fae", "content": report},
    ])
    episode = ChatCollectAgent().aggregate_threads(messages)[0]["episodes"][0]
    assert episode["extracted"]["missing_info_requests"] == []


def test_w2_required_info_filters_work_report_blob_even_if_w1_request_slips_through():
    report = (
        "各位领导，晚上好：深圳客户现场2025年12月22日工作汇报。"
        "现场工作：1.处理客户反馈复判站检测产品时卡顿。"
        "2.软件版本由0.27.29升级至0.27.44，已登记至操作变更记录表格。"
        "现场异常：J1元件焊盘框识别不准，收集数据上传至jira；以上请领导知悉。"
    )
    episode = {
        "episode_id": "ep:report-req-slip",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户反馈复判站卡顿。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": report}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户反馈复判站卡顿。",
            "debug_actions": [report],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "text": report,
                "thread_id": "t",
                "context_before": [{"text": "客户反馈复判站卡顿。"}],
                "context_after": [],
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["required_info_candidates"] == []


def test_w2_prefers_fault_sentence_over_status_update_blob():
    episode = {
        "episode_id": "ep:long-status",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "各位领导下午好，昨晚3线炉前2D设备在20点18分左右，新建调宽时出现应用异常，日志已上传。客户描述：弹窗后点击是后，软件没有立即闪退，但是鼠标无法点动，然后出现响应状态，然后黑屏关机，断电几分钟后再开机显示器不亮。",
            "content_summary": "各位领导下午好，昨晚3线炉前2D设备在20点18分左右，新建调宽时出现应用异常，日志已上传。客户描述：弹窗后点击是后，软件没有立即闪退，但是鼠标无法点动，然后出现响应状态，然后黑屏关机，断电几分钟后再开机显示器不亮。",
        }],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查显卡驱动和显示输出接口。", "content_summary": "检查显卡驱动和显示输出接口。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "各位领导下午好，昨晚3线炉前2D设备在20点18分左右，新建调宽时出现应用异常，日志已上传。客户描述：弹窗后点击是后，软件没有立即闪退，但是鼠标无法点动，然后出现响应状态，然后黑屏关机，断电几分钟后再开机显示器不亮。",
            "debug_actions": ["检查显卡驱动和显示输出接口。"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    error = _node(candidate, "Error")
    assert len(error["symptom"]) <= 160
    assert "各位领导" not in error["label"]
    assert "客户描述" not in error["label"]
    assert "应用异常" in error["symptom"] or "黑屏关机" in error["symptom"]
    assert _node(candidate, "DiagnosticCheck")["how_to_check"] == "检查显卡驱动和显示输出接口"


def test_w2_does_not_create_check_from_pure_handoff_request():
    episode = {
        "episode_id": "ep:pure-handoff",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "测试中发生闪退，没发现报错。"}],
        "diagnostic_chain_messages": [{"message_id": "m1", "text": "麻烦健哥看一下，测试中发生闪退，没发现报错。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "麻烦健哥看一下，测试中发生闪退，没发现报错。",
            "debug_actions": ["麻烦健哥看一下，测试中发生闪退，没发现报错。"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []


def test_w2_filters_jira_title_date_time_fragments_from_software_versions():
    episode = {
        "episode_id": "ep:jira-version-noise",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "SMTAOITS-1234 1.3.5 客户02 2026.05.16 11.02测试中软件闪退。",
        }],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "搜索U37器件之后，双击查看详情的时候闪退。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "SMTAOITS-1234 1.3.5 客户02 2026.05.16 11.02测试中软件闪退。",
            "versions": ["1.3.5", "05.16", "11.02"],
            "debug_actions": ["搜索U37器件之后，双击查看详情的时候闪退。"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert versions == ["1.3.5"]


def test_w2_strips_handoff_tail_from_fault_label_and_diagnostic_check():
    episode = {
        "episode_id": "ep:handoff-tail",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "客户反馈在编程阶段蓝屏重启.早上9.05和9.33蓝屏重启了两次 软件版本0.27.33 麻烦蒙老师看一下",
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m2",
            "text": "客户反馈在编程阶段蓝屏重启.早上9.05和9.33蓝屏重启了两次 软件版本0.27.33 麻烦蒙老师看一下",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户反馈在编程阶段蓝屏重启.早上9.05和9.33蓝屏重启了两次 软件版本0.27.33 麻烦蒙老师看一下",
            "versions": ["9.05", "9.33", "0.27.33", "27.33"],
            "debug_actions": ["客户反馈在编程阶段蓝屏重启.早上9.05和9.33蓝屏重启了两次 软件版本0.27.33 麻烦蒙老师看一下"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    error = _node(candidate, "Error")
    assert "麻烦" not in error["label"]
    assert "看一下" not in error["symptom"]
    assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert versions == ["0.27.33"]


def test_w2_does_not_turn_daily_report_blob_into_diagnostic_check():
    episode = {
        "episode_id": "ep:daily-report",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "今日汇报如下：一、现场工作汇总 1早上到现场升级软件版本。二、问题收集反馈：四线测试过程中突然重启。",
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m1",
            "text": "今日汇报如下：一、现场工作汇总 1早上到现场升级软件版本。二、问题收集反馈：四线测试过程中突然重启。",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "今日汇报如下：一、现场工作汇总 1早上到现场升级软件版本。二、问题收集反馈：四线测试过程中突然重启。",
            "debug_actions": ["今日汇报如下：一、现场工作汇总 1早上到现场升级软件版本。二、问题收集反馈：四线测试过程中突然重启。"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []


def test_w2_filters_log_handoff_tail_and_bare_version_substrings():
    episode = {
        "episode_id": "ep:log-handoff",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "朗特一线设备夜班客户反馈新做程序时设备自动重启.软件版本0.27.7 发生时间05:47左右 辛苦阿神根据日志排查异常引起原因 送你小红花",
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m1",
            "text": "朗特一线设备夜班客户反馈新做程序时设备自动重启.软件版本0.27.7 发生时间05:47左右 辛苦阿神根据日志排查异常引起原因 送你小红花",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "朗特一线设备夜班客户反馈新做程序时设备自动重启.软件版本0.27.7 发生时间05:47左右 辛苦阿神根据日志排查异常引起原因 送你小红花",
            "versions": ["0.27.7", "27.7", "05.47"],
            "debug_actions": ["朗特一线设备夜班客户反馈新做程序时设备自动重启.软件版本0.27.7 发生时间05:47左右 辛苦阿神根据日志排查异常引起原因 送你小红花"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    error = _node(candidate, "Error")
    assert "辛苦" not in error["label"]
    assert "送你小红花" not in error["symptom"]
    assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert versions == ["0.27.7"]


def test_w2_required_info_uses_request_clause_not_whole_explanation_blob():
    episode = {
        "episode_id": "ep:req-focus-dmp",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户反馈复判站蓝屏重启。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "可能原因：电源电压不稳、内存损坏、硬盘坏道。把 MEMORY.DMP 文件发给我。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户反馈复判站蓝屏重启。",
            "debug_actions": ["可能原因：电源电压不稳、内存损坏、硬盘坏道。把 MEMORY.DMP 文件发给我。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "可能原因：电源电压不稳、内存损坏、硬盘坏道。把 MEMORY.DMP 文件发给我。",
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    required = candidate["required_info_candidates"]
    assert [item["slot"] for item in required] == ["log_package"]
    assert required[0]["condition"] == "dmp"
    assert required[0]["request_focus_text"] == "把 MEMORY.DMP 文件发给我"


def test_w2_required_info_does_not_promote_context_words_to_site_or_device_slots():
    episode = {
        "episode_id": "ep:req-context-noise",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户现场设备刚交付，复判站下午蓝屏。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "该客户现场设备刚交付，辛苦刘工明天去现场协助排查。请提供windows日志中的Bugcheck错误截图。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户现场设备刚交付，复判站下午蓝屏。",
            "debug_actions": ["请提供windows日志中的Bugcheck错误截图。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "该客户现场设备刚交付，辛苦刘工明天去现场协助排查。请提供windows日志中的Bugcheck错误截图。",
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    slots = [item["slot"] for item in candidate["required_info_candidates"]]
    assert slots == ["log_package", "error_message"]


def test_w2_required_info_filters_explanatory_clauses_with_provide_or_ip_words():
    noisy_texts = [
        "这次dump已经没法定位具体驱动。下一步点击提供程序，按提供程序排序，不要选Microsoft Corporation。",
        "所有发给相机IP地址的数据包都直接走网卡发送，可以倒推是因为运控卡ip变动导致程序崩溃。",
        "我继续看看有没有可以优化的内存分配方式。",
        "提jira跟踪了哈，请知悉 TEST-1234 机器访问不到的认证IP地址同步过来了。",
        "从你提供的 ipconfig 看，以太网 6 的子网掩码是 255.255.255.0。",
        "收集日志和验证视频提供给产研。",
        "和物理内存容量、页面文件有没有开启，没有关系。",
    ]
    for idx, text in enumerate(noisy_texts):
        episode = {
            "episode_id": f"ep:req-explain-noise-{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈软件闪退。"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": text}],
            "resolution_messages": [],
            "evidence_message_ids": ["m1", "m2"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈软件闪退。",
                "debug_actions": [text],
                "conclusion": "",
                "missing_info_requests": [{
                    "message_id": "m2",
                    "thread_id": "t",
                    "text": text,
                    "evidence_message_ids": ["m2"],
                    "provided_later": False,
                    "provided_evidence_message_ids": [],
                }],
            },
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        assert candidate["required_info_candidates"] == []


def test_w2_required_info_maps_memory_dump_alias_to_log_package():
    episode = {
        "episode_id": "ep:req-dump-alias",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户反馈蓝屏，错误代码0x00000139。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "把转存储文件发我。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户反馈蓝屏，错误代码0x00000139。",
            "debug_actions": ["把转存储文件发我。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "把转存储文件发我。",
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    required = candidate["required_info_candidates"]
    assert [item["slot"] for item in required] == ["log_package"]
    assert required[0]["condition"] == "dmp"


def test_w2_routes_solution_like_text_to_solution_not_check():
    episode = {
        "episode_id": "ep:solution-text",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户反馈系统文件损坏导致软件闪退。"}],
        "diagnostic_chain_messages": [{
            "message_id": "m2",
            "text": "解决方案：启用TLS 1.2 操作：打开控制面板-查看大图标-internet选项--高级--取消勾选TLS1.0。",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户反馈系统文件损坏导致软件闪退。",
            "debug_actions": ["解决方案：启用TLS 1.2 操作：打开控制面板-查看大图标-internet选项--高级--取消勾选TLS1.0。"],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    checks = [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"]
    solutions = [node for node in candidate["nodes"] if node["type"] == "Solution"]
    assert checks == []
    assert solutions
    assert "启用TLS" in solutions[0]["content"]


def test_pipeline_apply_approved_does_not_auto_apply_fresh_pending_items():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "upload")
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        out = WriteSidePipeline(store).run_xing_upload(root, emit_episodes=True, dry_run_merge=True, apply_approved=True)
        assert out["summary"]["applied"] == 0
        assert out["summary"]["required_info_applied"] == 0
        assert out["approved_apply_results"] == []
        assert store.read_review_queue("approved_applied.json") == []
        assert all(detail["ingest"]["reason"] == "pending_review_item_not_auto_applied" for detail in out["details"])


def test_pipeline_apply_approved_reads_existing_approved_ask_info_queue_only():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "instances" / "errors").mkdir(parents=True)
        (kg / "review_queue").mkdir(parents=True)
        error_path = kg / "instances" / "errors" / "errors.json"
        error_path.write_text('[{"error_id":"err:init","label":"初始化失败","required_info":[]}]\n', encoding="utf-8")
        approved_item = {
            "review_id": "review:reqinfo:init-log",
            "review_status": "approved",
            "required_info_candidate": {
                "candidate_id": "reqinfo:init-log",
                "target_error_id": "err:init",
                "slot": "log_package",
                "question": "请提供启动/初始化阶段的 DLOG 或诊断数据包。",
                "merge_policy": "append_to_required_info",
                "source_episode_id": "ep1",
                "evidence_message_ids": ["m1"],
            },
        }
        pending_item = {
            "review_id": "review:reqinfo:pending",
            "review_status": "pending",
            "required_info_candidate": {
                "candidate_id": "reqinfo:pending",
                "target_error_id": "err:init",
                "slot": "software_version",
                "question": "请提供软件版本。",
                "merge_policy": "append_to_required_info",
                "source_episode_id": "ep2",
                "evidence_message_ids": ["m2"],
            },
        }
        store = JsonKGStore(kg)
        store.write_review_queue("ask_info_candidates.json", [approved_item, pending_item])
        results = WriteSidePipeline(store).apply_approved_review_queue()
        assert [item["status"] for item in results] == ["required_info_applied"]
        data = json.loads(error_path.read_text(encoding="utf-8"))
        assert data[0]["required_info"] == ["请提供启动/初始化阶段的 DLOG 或诊断数据包。"]
        assert "请提供软件版本。" not in data[0]["required_info"]


def test_pipeline_apply_approved_honors_external_queue_dir_for_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        queue_dir = Path(tmp) / "external_review_queue"
        queue_dir.mkdir(parents=True)
        store = JsonKGStore(kg)
        candidate = KnowledgeExtractionAgent().extract({
            "episode_id": "ep:external-approved-merge",
            "thread_id": "t",
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈相机初始化失败。"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查相机IP配置。"}],
            "resolution_messages": [{"message_id": "m3", "text": "已解决，恢复相机IP后正常。"}],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈相机初始化失败。",
                "debug_actions": ["检查相机IP配置。"],
                "conclusion": "已解决，恢复相机IP后正常。",
                "missing_info_requests": [],
            },
        })
        (queue_dir / "candidates.json").write_text(
            json.dumps([{"review_status": "approved", "candidate": candidate}], ensure_ascii=False),
            encoding="utf-8",
        )

        results = WriteSidePipeline(store, queue_dir=queue_dir).apply_approved_review_queue()
        assert [item["status"] for item in results] == ["applied_to_graph"]
        assert store.read_review_queue("candidates.json") == []
        reloaded = JsonKGStore(kg)
        assert _node(candidate, "Error")["error_id"] in reloaded.errors_by_id
        assert reloaded.read_review_queue("approved_applied.json")[0]["candidate_id"] == candidate["candidate_id"]


def test_pipeline_apply_approved_honors_external_queue_dir_for_required_info():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        queue_dir = Path(tmp) / "external_review_queue"
        queue_dir.mkdir(parents=True)
        (kg / "instances" / "errors").mkdir(parents=True)
        error_path = kg / "instances" / "errors" / "errors.json"
        error_path.write_text('[{"error_id":"err:init","label":"初始化失败","required_info":[]}]\n', encoding="utf-8")
        (queue_dir / "ask_info_candidates.json").write_text(
            json.dumps([
                {
                    "review_id": "review:reqinfo:init-log-external",
                    "review_status": "approved",
                    "required_info_candidate": {
                        "candidate_id": "reqinfo:init-log-external",
                        "target_error_id": "err:init",
                        "slot": "log_package",
                        "question": "请提供启动/初始化阶段的 DLOG 或诊断数据包。",
                        "merge_policy": "append_to_required_info",
                        "source_episode_id": "ep1",
                        "evidence_message_ids": ["m1"],
                    },
                }
            ], ensure_ascii=False),
            encoding="utf-8",
        )

        results = WriteSidePipeline(JsonKGStore(kg), queue_dir=queue_dir).apply_approved_review_queue()
        assert [item["status"] for item in results] == ["required_info_applied"]
        data = json.loads(error_path.read_text(encoding="utf-8"))
        assert data[0]["required_info"] == ["请提供启动/初始化阶段的 DLOG 或诊断数据包。"]


def test_w2_removes_log_handoff_tail_and_does_not_create_check_from_reboot_symptom():
    episode = {
        "episode_id": "ep:blue-screen-handoff",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "不会自动跳到下一个复判数据 软件版本0.27.7 白班9.29左右蓝屏重启，昨天开始重启之后到现在已重启三次 麻烦蒙老师根据日志查询一下异常引起原因",
            "content_summary": "不会自动跳到下一个复判数据 软件版本0.27.7 白班9.29左右蓝屏重启，昨天开始重启之后到现在已重启三次 麻烦蒙老师根据日志查询一下异常引起原因",
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m1",
            "text": "不会自动跳到下一个复判数据 软件版本0.27.7 白班9.29左右蓝屏重启，昨天开始重启之后到现在已重启三次 麻烦蒙老师根据日志查询一下异常引起原因",
            "content_summary": "不会自动跳到下一个复判数据 软件版本0.27.7 白班9.29左右蓝屏重启，昨天开始重启之后到现在已重启三次 麻烦蒙老师根据日志查询一下异常引起原因",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "不会自动跳到下一个复判数据 软件版本0.27.7 白班9.29左右蓝屏重启，昨天开始重启之后到现在已重启三次 麻烦蒙老师根据日志查询一下异常引起原因",
            "debug_actions": ["不会自动跳到下一个复判数据 软件版本0.27.7 白班9.29左右蓝屏重启，昨天开始重启之后到现在已重启三次 麻烦蒙老师根据日志查询一下异常引起原因"],
            "versions": ["0.27.7", "27.7", "9.29"],
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert "麻烦" not in candidate["label"]
    assert "根据日志" not in candidate["label"]
    assert not [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"]
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert versions == ["0.27.7"]


def test_w2_strips_handoff_tail_but_keeps_concrete_restart_repro_check():
    episode = {
        "episode_id": "ep:app-exception-repro",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "朗特三线设备在主机点到复判界面查看3d成像时报错应用异常，软件卡死之后白屏。",
            "content_summary": "朗特三线设备在主机点到复判界面查看3d成像时报错应用异常，软件卡死之后白屏。",
        }],
        "diagnostic_chain_messages": [{
            "message_id": "m2",
            "text": "重启软件再次进行同样操作未复现，软件版本配套软件0.27.26 主程序是左经理给的私包 麻烦蒙老师排查一下异常引起原因",
            "content_summary": "重启软件再次进行同样操作未复现，软件版本配套软件0.27.26 主程序是左经理给的私包 麻烦蒙老师排查一下异常引起原因",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "朗特三线设备在主机点到复判界面查看3d成像时报错应用异常，软件卡死之后白屏。",
            "debug_actions": ["重启软件再次进行同样操作未复现，软件版本配套软件0.27.26 主程序是左经理给的私包 麻烦蒙老师排查一下异常引起原因"],
            "versions": ["0.27.26", "27.26"],
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    check = _node(candidate, "DiagnosticCheck")
    assert "重启软件再次进行同样操作未复现" in check["how_to_check"]
    assert "麻烦" not in check["how_to_check"]
    assert "排查一下" not in check["how_to_check"]
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert versions == ["0.27.26"]


def test_w2_filters_remaining_handoff_info_requests_from_diagnostic_checks():
    samples = [
        "辛苦谢工提供一下此现象发生的时间",
        "工程师乙，这个故障复判站客户放了一上午，现在可以进系统用了，麻烦看下你这边需要收集什么信息排查故障",
        "好的 辛苦大家了 明天如有问题随时沟通~ 咱们会议上的事项确认好后",
        "正常测试中突然重启 从软件日志上看，没有检测到原因 麻烦从硬件上",
        "吐槽镭晨编程麻烦、界面复杂 2、D101 复判页面确认时报错“保存结果失败”",
    ]
    for idx, text in enumerate(samples):
        episode = {
            "episode_id": f"ep:handoff-leftover:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "diagnostic_chain_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": text, "debug_actions": [text], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        checks = [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"]
        assert checks == []


def test_w2_strips_general_help_tail_without_dropping_poolmon_command():
    episode = {
        "episode_id": "ep:poolmon",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "设备蓝屏", "content_summary": "设备蓝屏"}],
        "diagnostic_chain_messages": [{
            "message_id": "m2",
            "text": "已管理员权限打开cmd 运行下 poolmon /p 回车 把结果返回我看下",
            "content_summary": "已管理员权限打开cmd 运行下 poolmon /p 回车 把结果返回我看下",
        }],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "设备蓝屏",
            "debug_actions": ["已管理员权限打开cmd 运行下 poolmon /p 回车 把结果返回我看下"],
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    check = _node(candidate, "DiagnosticCheck")
    assert "poolmon /p" in check["how_to_check"]
    assert "把结果返回我看下" not in check["how_to_check"]


def test_w2_filters_more_handoff_tail_symptoms_from_diagnostic_checks():
    samples = [
        "客户反馈设备自动重启，次数频繁，自动重启后会有报错，如图一所示，辛苦售后同学介入，还请产研大佬帮忙看看此异常，感谢",
        "工程师乙，上午好，昨晚客户反馈设备蓝屏，已远程收集DMP文件和日志，辛苦有空帮忙看看，谢谢",
        "客户反馈四线设备，正常运行中黑屏重启，软件版本0.28.1，发生时间12.39左右，麻烦邢工排查一下",
        "设备出现蓝屏现象，重启后正常，报告显示是内存违规访问导致，麻烦工程师乙帮忙看一下了",
        "炉前2D在19.40分和20.15左右测试时突然闪退。客户反馈未进行任何操作。请蒙老师根据日志排查闪退原因。",
    ]
    for idx, text in enumerate(samples):
        episode = {
            "episode_id": f"ep:handoff-general:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "diagnostic_chain_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": text, "debug_actions": [text], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        checks = [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"]
        assert checks == []


def test_w2_filters_real_archive_handoff_checks_but_keeps_concrete_actions_clean():
    negative_samples = [
        "日志正在联系客户帮忙导出",
        "这几个方案还请邢工帮忙确认一波",
        "请工程师乙帮忙确认点是还是点否",
        "还请帮忙确认一下是否更换显卡",
        "收集日志 帮忙排查",
        "请帮忙确认是我司设备原因",
        "请尽快帮忙确认一下异常原因",
        "工程师乙帮忙确认下",
        "售后同事已帮忙收集数据并反馈 ： 今日SPC数据",
        "帮忙确认下这个是显示器物理损坏吗",
        "收集以上日志可以帮忙分析定位下原因吗",
    ]
    for idx, text in enumerate(negative_samples):
        episode = {
            "episode_id": f"ep:archive-handoff:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": "客户反馈设备异常。", "content_summary": "客户反馈设备异常。"}],
            "diagnostic_chain_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": "客户反馈设备异常。", "debug_actions": [text], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        checks = [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"]
        assert checks == []

    keep = "这个问题可以升级V0.27.43再观察一段看看，按理是不会出现闪退了"
    episode = {
        "episode_id": "ep:archive-keep-upgrade",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "客户反馈软件闪退。", "content_summary": "客户反馈软件闪退。"}],
        "diagnostic_chain_messages": [{"message_id": "m1", "text": keep, "content_summary": keep}],
        "resolution_messages": [],
        "evidence_message_ids": ["m0", "m1"],
        "source_offsets": [],
        "extracted": {"symptom_raw": "客户反馈软件闪退。", "debug_actions": [keep], "missing_info_requests": []},
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    check = _node(candidate, "DiagnosticCheck")
    assert "升级V0.27.43再观察一段" in check["how_to_check"]
    assert "看看" not in check["how_to_check"]


def test_w2_filters_communication_request_and_demotes_daily_report_labels():
    episode = {
        "episode_id": "ep:switch-communication",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "客户反馈交换机相关异常。", "content_summary": "客户反馈交换机相关异常。"}],
        "diagnostic_chain_messages": [{"message_id": "m1", "text": "请帮忙沟通交换机是否提供 6、", "content_summary": "请帮忙沟通交换机是否提供 6、"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m0", "m1"],
        "source_offsets": [],
        "extracted": {"symptom_raw": "客户反馈交换机相关异常。", "debug_actions": ["请帮忙沟通交换机是否提供 6、"], "missing_info_requests": []},
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []

    report = "各位领导晚上好，今日现状已更新，请查阅 一、现场工作汇总： 1.上午去现场跟线，目前设备暂未出现花屏情况持续观察中。 2.缓存机通讯问题同步代理商。"
    episode = {
        "episode_id": "ep:daily-label",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m2", "text": report, "content_summary": report}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": report, "content_summary": report}],
        "resolution_messages": [],
        "evidence_message_ids": ["m2"],
        "source_offsets": [],
        "extracted": {"symptom_raw": report, "debug_actions": [report], "missing_info_requests": []},
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["label"] == "群聊噪声/待人工确认"
    gate = QualityGateAgent().score(candidate)
    assert gate["passed"] is False


def test_w2_filters_long_explanatory_or_procurement_blobs_but_keeps_repro_test_check():
    negative_samples = [
        "就会: 网卡1: 192.168.1.1 设备A 假设正确IP:192.168.1.10 正确MAC:AA-AA-AA 网卡2: 192.168.2.1 假设设备B正确IP:192.168.2.20 正确MAC:BB-BB-BB 设备A被错误设置成192.168.2.20 设备B被错误配置成192.168.1.10 表现出来的现象 设备A 声明自己的IP是192.168.2.20 设备B 声明自己的IP是192.168.1.10 此时会发生： 1. Windows 的 ARP 缓存中： 192.168.1.10错误映射到 BB-BB-BB",  # noqa: E501
        "但都不支持循环覆盖 支持循环覆盖：【九音九视HD60】九音九视HD60P高清HDMI带屏幕视频录像录制盒机顶盒电脑课程加密手机老磁带翻录【行情 报价 价格 评测】-京东 900左右 淘宝上： 1080P录像盒直播录制游戏HDMI音视频U盘硬盘存储抓拍照片采集卡器-淘宝网 340 Image: img",  # noqa: E501
        "WfpProcessOutTransportStackIndication 三、问题本质 网络路径WFP中的内存被破坏 出现非法指针访问 四、根因定性 第三方网络过滤驱动异常（非系统模块） 可能机制：Use-After-Free 五、责任归属 tcpip.sys：受害者 实际问题：第三方驱动",  # noqa: E501
        "预计明后天会开始进行综合对比 2、配合客户持续优化默认参数并同步所有设备 3、大板车间处理客户认证权限无法登录（局域网被禁用）、勿开启首件检测、设备测试卡顿以及培训员工软件操作 4、日常数据回传： 二、需求记录",  # noqa: E501
    ]
    for idx, text in enumerate(negative_samples):
        episode = {
            "episode_id": f"ep:long-noise:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": "客户反馈设备异常。", "content_summary": "客户反馈设备异常。"}],
            "diagnostic_chain_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": "客户反馈设备异常。", "debug_actions": [text], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []

    keep = "第三次测试：aida64+occt测试无复现 （20分钟） 换回原来8pin供电线： 第一次测试:aida64+OCCT测试复现(18秒） 紧固供电线 第二次测试：aida64+OCCT测试无复现（20分钟） 第三次测试：aida64+OCCT+cpuburner+memtest64测试复现(3分钟) 发现CPU频率在4.3GHz以上容易复现"  # noqa: E501
    episode = {
        "episode_id": "ep:repro-stress-test",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "客户反馈设备重启。", "content_summary": "客户反馈设备重启。"}],
        "diagnostic_chain_messages": [{"message_id": "m1", "text": keep, "content_summary": keep}],
        "resolution_messages": [],
        "evidence_message_ids": ["m0", "m1"],
        "source_offsets": [],
        "extracted": {"symptom_raw": "客户反馈设备重启。", "debug_actions": [keep], "missing_info_requests": []},
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    check = _node(candidate, "DiagnosticCheck")
    assert "aida64+occt测试无复现" in check["how_to_check"]
    assert "紧固供电线" in check["how_to_check"]


def test_w2_filters_1_0_system_environment_version_but_keeps_explicit_app_versions():
    episode = {
        "episode_id": "ep:version-env",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "今天安装1.0新系统，软件版本0.27.46报模型缺失，后升级版本27.48正常。", "content_summary": "今天安装1.0新系统，软件版本0.27.46报模型缺失，后升级版本27.48正常。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m1"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "今天安装1.0新系统，软件版本0.27.46报模型缺失，后升级版本27.48正常。",
            "versions": ["1.0", "0.27.46", "27.48"],
            "debug_actions": [],
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert "1.0" not in versions
    assert "0.27.46" in versions

    episode["episode_id"] = "ep:version-short-real"
    episode["fault_description_messages"][0]["text"] = "现场测板时软件卡死自动退出，使用版本0.28，时间12：00左右。"
    episode["fault_description_messages"][0]["content_summary"] = "现场测板时软件卡死自动退出，使用版本0.28，时间12：00左右。"
    episode["extracted"]["symptom_raw"] = "现场测板时软件卡死自动退出，使用版本0.28，时间12：00左右。"
    episode["extracted"]["versions"] = ["0.28"]
    candidate = KnowledgeExtractionAgent().extract(episode)
    versions = [node["version_string"] for node in candidate["nodes"] if node["type"] == "SoftwareVersion"]
    assert versions == ["0.28"]


def test_w2_filters_nested_timestamp_chat_and_ci_commit_records_from_checks():
    samples = [
        "今天也安装了那个1.0新系统 2026-01-16T16:11:13+08:00 罗新忠: 但是安装完他模型又缺一个 2026-01-16T16:11:18+08:00 罗新忠: 版本27.46 2026-01-16T16:13:38+08:00 工程师午: 1.0的系统",
        "之后程序崩溃 产研结论: ci-robot: zhoutan| mentioned this issue in commit 54dac620| of qt / smt-aoi| on branch test| :{quote}fix: SMTAOITS-1234 {quote} 修复/完成版本: v1.4.0 请现场确认相关信息并跟进处理",
    ]
    for idx, text in enumerate(samples):
        episode = {
            "episode_id": f"ep:nested-record:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": "客户反馈程序异常。", "content_summary": "客户反馈程序异常。"}],
            "diagnostic_chain_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": "客户反馈程序异常。", "debug_actions": [text], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        assert [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"] == []


def test_w2_fault_label_prefers_customer_fault_fact_over_help_or_jira_tail():
    samples = [
        (
            "最近硬姐这台双轨3d设备进场交付后多次频繁报拍照失败，客户都是断电重启才正常，，，麻烦各位大佬排查一下 ，客户今天反馈时间8：23，今天软件+系统+系统应用日志",
            "麻烦",
        ),
        (
            "1.客户反馈导入acd后，编程进度条跑完后报错编程失败，重启软件后正常识别，但是主程序出现闪退情况，再次打开软件后正常.售后同事帮忙提交jira链接： 2.3.",
            "帮忙提交",
        ),
    ]
    for idx, (text, forbidden) in enumerate(samples):
        episode = {
            "episode_id": f"ep:label-tail:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": text, "debug_actions": [], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        assert any(k in candidate["label"] for k in ("客户反馈", "报错", "异常", "失败", "闪退", "重启"))
        assert forbidden not in candidate["label"]


def test_w2_demotes_confirmation_and_logistics_labels_to_review_noise():
    samples = [
        "工程师乙 这个原因可能帮忙查一下，是不是镜像设置问题？",
        "周工，现场内存条损坏，需要更换，帮忙安排一下 邓工，可以提供一个收货地址 工程师乙，换下的内存条是否需要寄回？",
    ]
    for idx, text in enumerate(samples):
        episode = {
            "episode_id": f"ep:label-confirm-logistics:{idx}",
            "thread_id": "t",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": f"m{idx}", "text": text, "content_summary": text}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "evidence_message_ids": [f"m{idx}"],
            "source_offsets": [],
            "extracted": {"symptom_raw": text, "debug_actions": [], "missing_info_requests": []},
        }
        candidate = KnowledgeExtractionAgent().extract(episode)
        assert candidate["label"] == "群聊噪声/待人工确认"
        gate = QualityGateAgent().score(candidate)
        assert gate["passed"] is False


def test_w5_approved_candidate_merges_nodes_edges_and_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        candidate = KnowledgeExtractionAgent().extract({
            "episode_id": "ep:approved-merge",
            "thread_id": "t",
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈相机初始化失败。"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查相机IP配置和网线连接。"}],
            "resolution_messages": [{"message_id": "m3", "text": "已解决，原因是相机IP被改，恢复正确IP后正常。"}],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈相机初始化失败。",
                "debug_actions": ["检查相机IP配置和网线连接。"],
                "conclusion": "已解决，原因是相机IP被改，恢复正确IP后正常。",
                "missing_info_requests": [],
            },
        })
        item = {"review_status": "approved", "candidate": candidate}
        result = IncrementalIngestAgent(store).apply_approved(item)
        assert result["status"] == "applied_to_graph"
        assert result["created_nodes"] >= 5
        assert result["created_edges"] >= 6

        reloaded = JsonKGStore(kg)
        error_id = _node(candidate, "Error")["error_id"]
        check_id = _node(candidate, "DiagnosticCheck")["check_id"]
        solution_id = _node(candidate, "Solution")["solution_id"]
        assert error_id in reloaded.errors_by_id
        assert check_id in reloaded.checks_by_id
        assert solution_id in reloaded.solutions_by_id
        assert any(edge["from"] == error_id and edge["to"] == check_id and edge["relation"] == "has_check" for edge in reloaded.edges)
        assert any(edge["from"] == check_id and edge["to"] == solution_id and edge["relation"] == "resolved_by" for edge in reloaded.edges)
        assert len(reloaded.read_review_queue("approved_applied.json")) == 1

        repeat = IncrementalIngestAgent(reloaded).apply_approved(item)
        assert repeat["status"] == "already_applied"
        reread = JsonKGStore(kg)
        assert len(reread.errors) == 1
        assert len(reread.checks) == 1
        assert len(reread.solutions) == 1
        assert len(reread.edges) == 7
        assert len(reread.read_review_queue("approved_applied.json")) == 1


def test_w5_dry_run_reports_real_node_edge_deltas_against_current_kg():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "instances" / "errors").mkdir(parents=True)
        (kg / "instances" / "checks").mkdir(parents=True)
        (kg / "instances" / "solutions").mkdir(parents=True)
        (kg / "review_queue").mkdir(parents=True)
        error_id = "err:existing"
        check_id = "check:existing:1"
        solution_id = "sol:new"
        (kg / "instances" / "errors" / "errors.json").write_text(
            json.dumps([{"error_id": error_id, "label": "相机初始化失败"}], ensure_ascii=False),
            encoding="utf-8",
        )
        (kg / "instances" / "checks" / "checks.json").write_text(
            json.dumps([{"check_id": check_id, "label": "检查相机IP", "how_to_check": "检查相机IP", "step_order": 1}], ensure_ascii=False),
            encoding="utf-8",
        )
        (kg / "instances" / "solutions" / "solutions.json").write_text("[]\n", encoding="utf-8")
        (kg / "edges.json").write_text(
            json.dumps([{"from": error_id, "to": check_id, "relation": "has_check"}], ensure_ascii=False),
            encoding="utf-8",
        )

        candidate = {
            "candidate_id": "chatcand:dry-run-delta",
            "schema_valid": True,
            "matched_existing_error": {"error_id": error_id},
            "nodes": [
                {"type": "Error", "error_id": error_id, "label": "相机初始化失败", "symptom": "启动时相机初始化失败"},
                {"type": "DiagnosticCheck", "check_id": check_id, "label": "检查相机IP", "how_to_check": "检查相机IP", "step_order": 1},
                {"type": "Solution", "solution_id": solution_id, "content": "恢复相机IP", "method": "恢复相机IP", "evidence_level": "chat_derived"},
            ],
            "edges": [
                {"from": error_id, "to": check_id, "relation": "has_check"},
                {"from": check_id, "to": solution_id, "relation": "resolved_by"},
            ],
        }
        plan = IncrementalIngestAgent(JsonKGStore(kg)).dry_run_merge_plan(candidate)

        assert [item["id"] for item in plan["would_update_nodes"]] == [error_id]
        assert [item["id"] for item in plan["would_create_nodes"]] == [solution_id]
        assert [item["id"] for item in plan["would_skip_nodes"]] == [check_id]
        assert plan["would_create_edges"] == [{"from": check_id, "to": solution_id, "relation": "resolved_by"}]
        assert plan["would_skip_edges"] == [{"from": error_id, "to": check_id, "relation": "has_check"}]
        assert plan["affects_existing_check_chain"] is True


def test_w5_approved_candidate_merges_software_version_into_existing_versions_folder():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        version_id = "ver:9.9.9"
        candidate = {
            "candidate_id": "chatcand:version-path",
            "status": "approved",
            "schema_valid": True,
            "nodes": [
                {"type": "Error", "error_id": "err:version-path", "label": "版本相关初始化失败", "symptom": "初始化失败"},
                {"type": "DiagnosticCheck", "check_id": "check:version-path:1", "label": "检查版本", "how_to_check": "检查软件版本", "step_order": 1},
                {"type": "Solution", "solution_id": "sol:version-path", "content": "升级版本", "method": "升级版本", "evidence_level": "chat_derived"},
                {"type": "SoftwareVersion", "version_id": version_id, "version_string": "9.9.9"},
            ],
            "edges": [
                {"from": "err:version-path", "to": "check:version-path:1", "relation": "has_check"},
                {"from": "check:version-path:1", "to": "sol:version-path", "relation": "resolved_by"},
                {"from": "err:version-path", "to": version_id, "relation": "affects_version"},
            ],
        }
        result = IncrementalIngestAgent(JsonKGStore(kg)).apply_approved(candidate)

        assert result["status"] == "applied_to_graph"
        versions_path = kg / "instances" / "versions" / "versions.json"
        assert versions_path.exists()
        assert not (kg / "instances" / "software_versions").exists()
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
        assert any(item["version_id"] == version_id for item in versions)
        assert version_id in JsonKGStore(kg).software_versions_by_id


def test_w5_merge_candidate_apply_attaches_trace_outcome_and_checks_to_existing_error():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "instances" / "errors").mkdir(parents=True, exist_ok=True)
        (kg / "instances" / "checks").mkdir(parents=True, exist_ok=True)
        (kg / "instances" / "solutions").mkdir(parents=True, exist_ok=True)
        (kg / "instances" / "traces").mkdir(parents=True, exist_ok=True)
        (kg / "instances" / "outcomes").mkdir(parents=True, exist_ok=True)
        (kg / "instances" / "policies").mkdir(parents=True, exist_ok=True)
        (kg / "review_queue").mkdir(parents=True, exist_ok=True)
        (kg / "instances" / "errors" / "errors.json").write_text(json.dumps([
            {"error_id": "err:industrial-pc-blue-screen", "label": "工控机蓝屏", "symptom": "蓝屏", "category": "硬件与运控"}
        ], ensure_ascii=False), encoding="utf-8")
        (kg / "edges.json").write_text("[]\n", encoding="utf-8")

        item = {
            "queue": "merge_candidates",
            "review_status": "approved",
            "human_approved": True,
            "conflict": {"existing_error_id": "err:industrial-pc-blue-screen"},
            "candidate": {
                "candidate_id": "chatcand:merge-existing",
                "status": "pending_review",
                "schema_valid": True,
                "matched_existing_error": {"error_id": "err:industrial-pc-blue-screen"},
                "nodes": [
                    {"type": "Error", "error_id": "err:candidate-blue", "label": "蓝屏后更换内存恢复", "symptom": "蓝屏", "category": "硬件与运控", "entry_role": "case_variant", "canonical_error_id": "err:industrial-pc-blue-screen"},
                    {"type": "DiagnosticCheck", "check_id": "check:merge-existing:1", "label": "更换内存条", "how_to_check": "更换内存条并观察", "step_order": 1},
                    {"type": "Solution", "solution_id": "sol:merge-existing:1", "content": "更换内存条", "method": "replace", "evidence_level": "case_chat_evidence"},
                    {"type": "DiagnosticTrace", "trace_id": "trace:merge-existing", "source_episode_id": "ep:merge-existing", "target_error_id": "err:candidate-blue", "recommended_order": [{"check_id": "check:merge-existing:1", "label": "更换内存条", "order": 1}], "actual_order": [{"check_id": "check:merge-existing:1", "label": "更换内存条", "order": 1}], "evidence_message_ids": ["m1"]},
                    {"type": "DiagnosticOutcome", "outcome_id": "outcome:merge-existing:1", "source_episode_id": "ep:merge-existing", "target_error_id": "err:candidate-blue", "target_check_id": "check:merge-existing:1", "target_solution_id": "sol:merge-existing:1", "action_label": "现场已更换内存条", "outcome_type": "verified_fix", "evidence_message_ids": ["m2"]},
                ],
                "diagnostic_outcomes": [
                    {"outcome_id": "outcome:merge-existing:1", "target_error_id": "err:candidate-blue", "target_check_id": "check:merge-existing:1", "target_solution_id": "sol:merge-existing:1", "action_label": "现场已更换内存条", "outcome_type": "verified_fix", "evidence_message_ids": ["m2"]},
                ],
                "edges": [
                    {"from": "err:candidate-blue", "to": "check:merge-existing:1", "relation": "has_check"},
                    {"from": "err:candidate-blue", "to": "trace:merge-existing", "relation": "has_trace"},
                    {"from": "err:candidate-blue", "to": "outcome:merge-existing:1", "relation": "has_outcome"},
                    {"from": "check:merge-existing:1", "to": "sol:merge-existing:1", "relation": "resolved_by"},
                ],
            },
        }

        result = IncrementalIngestAgent(JsonKGStore(kg)).apply_approved(item)
        assert result["status"] == "applied_to_graph"

        reloaded = JsonKGStore(kg)
        assert "err:candidate-blue" not in reloaded.errors_by_id
        assert "check:merge-existing:1" in reloaded.checks_by_id
        assert "trace:merge-existing" in reloaded.traces_by_id
        assert "outcome:merge-existing:1" in reloaded.outcomes_by_id
        assert reloaded.traces_by_id["trace:merge-existing"]["target_error_id"] == "err:industrial-pc-blue-screen"
        assert reloaded.outcomes_by_id["outcome:merge-existing:1"]["target_error_id"] == "err:industrial-pc-blue-screen"
        assert any(edge["from"] == "err:industrial-pc-blue-screen" and edge["to"] == "check:merge-existing:1" and edge["relation"] == "has_check" for edge in reloaded.edges)
        assert any(edge["from"] == "err:industrial-pc-blue-screen" and edge["to"] == "trace:merge-existing" and edge["relation"] == "has_trace" for edge in reloaded.edges)
        assert any(edge["from"] == "err:industrial-pc-blue-screen" and edge["to"] == "outcome:merge-existing:1" and edge["relation"] == "has_outcome" for edge in reloaded.edges)
        assert "policy:err:industrial-pc-blue-screen" in reloaded.policies_by_id


def test_pipeline_apply_approved_reads_existing_approved_candidate_queue_and_merges_graph():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        (kg / "review_queue").mkdir(parents=True)
        store = JsonKGStore(kg)
        candidate = KnowledgeExtractionAgent().extract({
            "episode_id": "ep:pipeline-approved-merge",
            "thread_id": "t",
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈光源初始化失败。"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查光源控制器IP和触发线。"}],
            "resolution_messages": [{"message_id": "m3", "text": "已解决，恢复光源触发配置后正常。"}],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈光源初始化失败。",
                "debug_actions": ["检查光源控制器IP和触发线。"],
                "conclusion": "已解决，恢复光源触发配置后正常。",
                "missing_info_requests": [],
            },
        })
        store.write_review_queue("candidates.json", [{"review_status": "approved", "candidate": candidate}])
        results = WriteSidePipeline(store).apply_approved_review_queue()
        assert [item["status"] for item in results] == ["applied_to_graph"]
        reloaded = JsonKGStore(kg)
        assert _node(candidate, "Error")["error_id"] in reloaded.errors_by_id
        assert reloaded.read_review_queue("approved_applied.json")[0]["candidate_id"] == candidate["candidate_id"]


def test_cli_review_decision_marks_external_queue_without_applying_graph():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        queue_dir = Path(tmp) / "external_review_queue"
        config = Path(tmp) / "config.yaml"
        config.write_text(f"kg_root: {kg}\nsession_store: {Path(tmp) / 'sessions'}\n", encoding="utf-8")
        queue_dir.mkdir(parents=True)
        candidate = KnowledgeExtractionAgent().extract({
            "episode_id": "ep:cli-review-decision",
            "thread_id": "t",
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈相机初始化失败。"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查相机IP配置。"}],
            "resolution_messages": [{"message_id": "m3", "text": "已解决，恢复相机IP后正常。"}],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈相机初始化失败。",
                "debug_actions": ["检查相机IP配置。"],
                "conclusion": "已解决，恢复相机IP后正常。",
                "missing_info_requests": [],
            },
        })
        review_id = "review:cli-review-decision"
        (queue_dir / "candidates.json").write_text(
            json.dumps([{
                "review_id": review_id,
                "candidate_id": candidate["candidate_id"],
                "review_actions": ["approve", "reject", "merge_existing", "request_more_info"],
                "review_status": "pending",
                "candidate": candidate,
            }], ensure_ascii=False),
            encoding="utf-8",
        )

        buf = StringIO()
        with redirect_stdout(buf):
            assert cli_main([
                "--config",
                str(config),
                "review-decision",
                "candidates",
                review_id,
                "approve",
                "--queue-dir",
                str(queue_dir),
                "--reviewer",
                "qa",
                "--note",
                "evidence checked",
            ]) == 0
        out = json.loads(buf.getvalue())
        assert out["status"] == "decision_recorded"
        assert out["review_status"] == "approved"
        assert out["human_approved"] is True

        queued = json.loads((queue_dir / "candidates.json").read_text(encoding="utf-8"))
        assert queued[0]["selected_action"] == "approve"
        assert queued[0]["review_decision"]["reviewer"] == "qa"
        assert not JsonKGStore(kg).errors

        apply_buf = StringIO()
        with redirect_stdout(apply_buf):
            assert cli_main(["--config", str(config), "apply-approved-queue", "--queue-dir", str(queue_dir)]) == 0
        results = json.loads(apply_buf.getvalue())
        assert [item["status"] for item in results] == ["applied_to_graph"]
        assert _node(candidate, "Error")["error_id"] in JsonKGStore(kg).errors_by_id


def test_cli_review_decision_then_apply_handles_candidate_and_ask_info_together():
    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        queue_dir = Path(tmp) / "external_review_queue"
        config = Path(tmp) / "config.yaml"
        config.write_text(f"kg_root: {kg}\nsession_store: {Path(tmp) / 'sessions'}\n", encoding="utf-8")
        queue_dir.mkdir(parents=True)
        candidate = KnowledgeExtractionAgent().extract({
            "episode_id": "ep:cli-review-combined-candidate",
            "thread_id": "t",
            "completeness": "complete",
            "fault_description_messages": [{"message_id": "m1", "text": "客户反馈相机初始化失败。"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查相机IP配置。"}],
            "resolution_messages": [{"message_id": "m3", "text": "已解决，恢复相机IP后正常。"}],
            "evidence_message_ids": ["m1", "m2", "m3"],
            "source_offsets": [],
            "extracted": {
                "symptom_raw": "客户反馈相机初始化失败。",
                "debug_actions": ["检查相机IP配置。"],
                "conclusion": "已解决，恢复相机IP后正常。",
                "missing_info_requests": [],
            },
        })
        error_id = _node(candidate, "Error")["error_id"]
        required_info = {
            "candidate_id": "reqinfo:combined-init-log",
            "target_error_id": error_id,
            "slot": "log_package",
            "question": "请提供启动/初始化阶段的 DLOG 或诊断数据包。",
            "merge_policy": "append_to_required_info",
            "source_episode_id": "ep:cli-review-combined-candidate",
            "evidence_message_ids": ["m4"],
        }
        (queue_dir / "candidates.json").write_text(
            json.dumps([{
                "review_id": "review:combined-candidate",
                "candidate_id": candidate["candidate_id"],
                "review_actions": ["approve", "reject", "merge_existing", "request_more_info"],
                "review_status": "pending",
                "candidate": candidate,
            }], ensure_ascii=False),
            encoding="utf-8",
        )
        (queue_dir / "ask_info_candidates.json").write_text(
            json.dumps([{
                "review_id": "review:combined-ask-info",
                "candidate_id": required_info["candidate_id"],
                "review_actions": ["accept", "merge", "drop", "needs_owner", "needs_better_evidence"],
                "review_status": "pending",
                "required_info_candidate": required_info,
            }], ensure_ascii=False),
            encoding="utf-8",
        )

        for queue, review_id, action in (
            ("candidates", "review:combined-candidate", "approve"),
            ("ask_info_candidates", "review:combined-ask-info", "accept"),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                assert cli_main([
                    "--config",
                    str(config),
                    "review-decision",
                    queue,
                    review_id,
                    action,
                    "--queue-dir",
                    str(queue_dir),
                    "--reviewer",
                    "qa",
                ]) == 0
            out = json.loads(buf.getvalue())
            assert out["status"] == "decision_recorded"
            assert out["review_status"] == "approved"

        assert not JsonKGStore(kg).errors_by_id
        apply_buf = StringIO()
        with redirect_stdout(apply_buf):
            assert cli_main(["--config", str(config), "apply-approved-queue", "--queue-dir", str(queue_dir)]) == 0
        results = json.loads(apply_buf.getvalue())
        assert [item["status"] for item in results] == ["applied_to_graph", "required_info_applied"]
        reloaded = JsonKGStore(kg)
        assert error_id in reloaded.errors_by_id
        assert reloaded.errors_by_id[error_id]["required_info"] == ["请提供启动/初始化阶段的 DLOG 或诊断数据包。"]
        repeat_buf = StringIO()
        with redirect_stdout(repeat_buf):
            assert cli_main(["--config", str(config), "apply-approved-queue", "--queue-dir", str(queue_dir)]) == 0
        repeat_results = json.loads(repeat_buf.getvalue())
        assert [item["status"] for item in repeat_results] == ["already_applied", "required_info_already_applied"]
        reread = JsonKGStore(kg)
        assert reread.errors_by_id[error_id]["required_info_sources"]["log_package"]["occurrence_count"] == 1


def test_w2_manual_reboot_device_request_does_not_become_dmp_condition():
    episode = {
        "episode_id": "ep:req-manual-reboot-init-log",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "2D设备软件开启报错运动初始化失败，检查运控网络正常。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "重启设备会第一时间报错以上2张截图。请提供完整报错截图和日志。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "2D设备软件开启报错运动初始化失败，检查运控网络正常。",
            "debug_actions": ["重启设备会第一时间报错以上2张截图。请提供完整报错截图和日志。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "重启设备会第一时间报错以上2张截图。请提供完整报错截图和日志。",
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    conditions = {item["slot"]: item["condition"] for item in candidate["required_info_candidates"]}
    assert conditions["error_message"] != "dmp"
    assert conditions["log_package"] != "dmp"


def test_w2_dmp_ask_info_does_not_append_to_unrelated_matched_error():
    class FakeStore:
        def search_errors(self, query: str, limit: int = 5):
            class C:
                error_id = "err:license-validation-persistent-failure"
                label = "授权校验持续失败"
                score = 99.0
                route = "test"
                evidence = ["forced"]
            return [C()]

    episode = {
        "episode_id": "ep:req-dmp-unrelated-target",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "错误代码为 0x00000139，内核检测到了关键数据结构被破坏。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "把转存储文件发我。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "错误代码为 0x00000139，内核检测到了关键数据结构被破坏。",
            "debug_actions": ["把转存储文件发我。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "把转存储文件发我。",
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent(FakeStore(), match_threshold=0.1).extract(episode)
    required = candidate["required_info_candidates"]
    assert required
    assert required[0]["condition"] == "dmp"
    assert required[0]["target_error_id"] == ""
    assert required[0]["acceptable_error_ids"] == []
    assert required[0]["merge_policy"] == "review_only"


def test_w2_dmp_ask_info_allows_same_family_blue_screen_target():
    class FakeStore:
        def search_errors(self, query: str, limit: int = 5):
            class C:
                error_id = "err:industrial-pc-blue-screen"
                label = "工控机蓝屏"
                score = 99.0
                route = "test"
                evidence = ["forced"]
            return [C()]

    episode = {
        "episode_id": "ep:req-dmp-blue-screen-target",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "客户反馈工控机蓝屏。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "请提供蓝屏后的 dmp 文件。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户反馈工控机蓝屏。",
            "debug_actions": ["请提供蓝屏后的 dmp 文件。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "请提供蓝屏后的 dmp 文件。",
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent(FakeStore(), match_threshold=0.1).extract(episode)
    required = candidate["required_info_candidates"]
    assert required
    assert required[0]["condition"] == "dmp"
    assert required[0]["target_error_id"] == "err:industrial-pc-blue-screen"
    assert required[0]["merge_policy"] == "append_to_required_info"


def test_w2_dmp_condition_does_not_leak_to_software_version_slot():
    episode = {
        "episode_id": "ep:req-version-with-reboot-word",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "导入配置后软件重启，账户登录不上。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "请补充主程序版本和算法包版本。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "导入配置后软件重启，账户登录不上。",
            "debug_actions": ["请补充主程序版本和算法包版本。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "请补充主程序版本和算法包版本。",
                "context_before": [{"text": "导入配置后软件重启，账户登录不上。"}],
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    version = next(item for item in candidate["required_info_candidates"] if item["slot"] == "software_version")
    assert version["condition"] == ""


def test_ask_info_environment_supplement_keeps_disk_case_as_disk_not_kernel_power():
    items = [{
        "review_id": "review:reqinfo:disk-env",
        "queue": "ask_info_candidates",
        "quality_gate": {"passed": True},
        "required_info_candidate": {
            "candidate_id": "reqinfo:disk-env",
            "slot": "environment",
            "label": "运行环境",
            "question": "请补充系统环境、电源、磁盘、内存或运行环境信息。",
            "why_required": "运行环境能排除电源、磁盘、内存、系统等非业务配置问题。",
            "condition": "",
            "target_error_id": "err:disk-full-error-507",
            "acceptable_error_ids": ["err:disk-full-error-507"],
        },
        "episode": {
            "fault_description_messages": [{"text": "硬盘错误，第三块物理硬盘的第三个分区有错误。"}],
            "diagnostic_chain_messages": [],
        },
    }]
    scenario = build_scenarios(items, limit=1)[0]
    reply = scenario["user_turns"][0]["reply"]
    assert "磁盘管理器截图" in reply
    assert "Kernel-Power" not in reply


def test_w2_software_version_request_does_not_inherit_startup_condition_from_context():
    episode = {
        "episode_id": "ep:req-version-init-context",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "软件启动后登录失败。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "请提供主程序版本。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "软件启动后登录失败。",
            "debug_actions": ["请提供主程序版本。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "请提供主程序版本。",
                "context_before": [{"text": "软件启动后登录失败。"}],
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    version = next(item for item in candidate["required_info_candidates"] if item["slot"] == "software_version")
    assert version["condition"] == ""


def test_ask_info_software_version_supplement_does_not_inject_bugcheck_from_reboot_context():
    items = [{
        "review_id": "review:reqinfo:version-reboot",
        "queue": "ask_info_candidates",
        "quality_gate": {"passed": True},
        "required_info_candidate": {
            "candidate_id": "reqinfo:version-reboot",
            "slot": "software_version",
            "label": "软件版本",
            "question": "请提供主程序、算法包或相关软件版本。",
            "why_required": "版本信息用于判断是否命中已知缺陷、兼容性问题或升级/回退路径。",
            "condition": "",
            "target_error_id": "err:version-bug",
            "acceptable_error_ids": ["err:version-bug"],
        },
        "episode": {
            "fault_description_messages": [{"text": "软件重启之后原账户密码登录不上。"}],
            "diagnostic_chain_messages": [],
        },
    }]
    scenario = build_scenarios(items, limit=1)[0]
    reply = scenario["user_turns"][0]["reply"]
    assert "已提供主程序版本" in reply
    assert "BugCheck" not in reply


def test_w3_required_info_does_not_set_dmp_condition_on_software_version_slot():
    required = {
        "candidate_id": "reqinfo:version-reboot-context",
        "target_error_id": "err:version-bug",
        "slot": "software_version",
        "label": "软件版本",
        "question": "请提供主程序版本和算法包版本。",
        "why_required": "版本信息用于判断升级兼容性问题。",
        "condition": "",
        "evidence_message_ids": ["m1"],
        "merge_policy": "append_to_required_info",
        "source_request": {"text": "软件重启后登录失败，请提供主程序版本和算法包版本。"},
    }
    out = ConflictResolutionAgent().resolve_required_info(required)
    normalized = out["candidate"]
    assert normalized["slot"] == "software_version"
    assert normalized["condition"] == ""


def test_w3_manual_reboot_device_context_does_not_set_dmp_condition():
    required = {
        "candidate_id": "reqinfo:error-message-manual-reboot",
        "target_error_id": "err:init",
        "slot": "error_message",
        "label": "完整报错信息",
        "question": "请提供完整报错文本或报错截图。",
        "why_required": "完整报错可减少误召回。",
        "condition": "",
        "evidence_message_ids": ["m1"],
        "merge_policy": "append_to_required_info",
        "source_request": {"text": "重启设备会第一时间报错以上2张截图，请提供完整报错截图。"},
    }
    out = ConflictResolutionAgent().resolve_required_info(required)
    normalized = out["candidate"]
    assert normalized["slot"] == "error_message"
    assert normalized["condition"] == ""


def test_w2_software_version_target_guard_blocks_unrelated_board_loading_match():
    class FakeStore:
        def search_errors(self, query: str, limit: int = 5):
            class C:
                error_id = "err:board-loading-slow-fail"
                label = "上板慢导致失败"
                score = 99.0
                route = "test"
                evidence = ["forced"]
            return [C()]

    episode = {
        "episode_id": "ep:req-version-unrelated-target",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "导入认证配置后原账户密码登录不上。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "请提供主程序版本和算法包版本。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "导入认证配置后原账户密码登录不上。",
            "debug_actions": ["请提供主程序版本和算法包版本。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "请提供主程序版本和算法包版本。",
                "context_before": [{"text": "导入认证配置后原账户密码登录不上。"}],
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent(FakeStore(), match_threshold=0.1).extract(episode)
    version = next(item for item in candidate["required_info_candidates"] if item["slot"] == "software_version")
    assert version["target_error_id"] == ""
    assert version["merge_policy"] == "review_only"


def test_w2_environment_target_guard_allows_heap_crash_but_blocks_unrelated_match():
    class FakeStore:
        def search_errors(self, query: str, limit: int = 5):
            class C:
                error_id = "err:industrial-pc-blue-screen"
                label = "工控机蓝屏"
                score = 99.0
                route = "test"
                evidence = ["forced"]
            return [C()]

    episode = {
        "episode_id": "ep:req-heap-env-unrelated-target",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "smt-aoi.exe 应用程序发生堆损坏，错误代码 0xc0000374。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "请补充系统环境和内存状态。"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "smt-aoi.exe 应用程序发生堆损坏，错误代码 0xc0000374。",
            "debug_actions": ["请补充系统环境和内存状态。"],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": "请补充系统环境和内存状态。",
                "context_before": [{"text": "smt-aoi.exe 应用程序发生堆损坏，错误代码 0xc0000374。"}],
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent(FakeStore(), match_threshold=0.1).extract(episode)
    env = next(item for item in candidate["required_info_candidates"] if item["slot"] == "environment")
    assert env["target_error_id"] == ""
    assert env["merge_policy"] == "review_only"


def test_w2_required_info_filters_technical_explanation_not_request():
    text = "这意味着应用程序 (smt-aoi.exe) 在尝试使用 Windows 提供的内存管理（堆）时，破坏了堆的结构。"
    episode = {
        "episode_id": "ep:req-tech-explanation",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "smt-aoi.exe 应用程序发生堆损坏，错误代码 0xc0000374。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": text}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "smt-aoi.exe 应用程序发生堆损坏，错误代码 0xc0000374。",
            "debug_actions": [text],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m2",
                "thread_id": "t",
                "text": text,
                "evidence_message_ids": ["m2"],
                "provided_later": False,
                "provided_evidence_message_ids": [],
            }],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["required_info_candidates"] == []


def test_ask_info_error_message_supplement_does_not_inject_bugcheck_for_plain_restart_crash():
    items = [{
        "review_id": "review:reqinfo:error-message-restart-crash",
        "queue": "ask_info_candidates",
        "quality_gate": {"passed": True},
        "required_info_candidate": {
            "candidate_id": "reqinfo:error-message-restart-crash",
            "slot": "error_message",
            "label": "完整报错信息",
            "question": "请提供完整报错文本或报错截图。",
            "why_required": "完整报错可直接匹配 KG 中的 Error/LogPattern，减少误召回。",
            "condition": "",
            "target_error_id": "err:software-crash",
            "acceptable_error_ids": ["err:software-crash"],
        },
        "episode": {
            "fault_description_messages": [{"text": "打开主程序日志，看重启前闪退的那段，截图发出来。"}],
            "diagnostic_chain_messages": [],
        },
    }]
    scenario = build_scenarios(items, limit=1)[0]
    reply = scenario["user_turns"][0]["reply"]
    assert "完整报错文本" in reply or "完整报错信息" in reply
    assert "BugCheck" not in reply


def test_w2_preserves_fault_fact_after_image_placeholder_when_help_tail_present():
    text = "[Image: img_v3_02tn_e6f1ef4f-4508-43c2-b19b-2d739f036b8g] 要不看下程序呢，日志里是程序卡死，Windows系统并没有卡死"
    episode = {
        "episode_id": "ep:image-placeholder-fault",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": text, "content_summary": text}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "客户16现场设备T91一直重启", "content_summary": "客户16现场设备T91一直重启"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {"symptom_raw": text, "debug_actions": ["客户16现场设备T91一直重启"], "missing_info_requests": []},
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["label"] != "Image: img_v3_02tn_e6f1ef4f-4"
    assert "Image:" not in candidate["label"]
    assert "img_v3" not in candidate["label"]
    assert "程序卡死" in candidate["label"]


def test_w4_rejects_placeholder_or_handoff_labels_even_with_check_nodes():
    for label in ["群聊噪声/待人工确认", "Image: img_v3_0211m_423c8e00-ded8", "一会罗工准备好了"]:
        candidate = {
            "candidate_id": f"chatcand:{label}",
            "label": label,
            "symptom_raw": "一会罗工准备好了",
            "confidence": 0.9,
            "schema_valid": True,
            "evidence_ids": ["m1"],
            "source_offsets": [{"message_id": "m1", "field": "symptom", "value": label}],
            "nodes": [
                {"type": "Error", "error_id": "err:test", "label": label},
                {"type": "DiagnosticCheck", "check_id": "check:test", "how_to_check": "查看网卡设置和事件", "step_order": 1},
            ],
            "edges": [{"from": "err:test", "to": "check:test", "relation": "has_check"}],
            "episode": {"completeness": "partial"},
        }
        gate = QualityGateAgent().score(candidate)
        assert gate["passed"] is False
        assert "weak_fault_label" in gate["issues"]




def test_w2_does_not_create_check_from_binary_clarification_question():
    episode = {
        "episode_id": "ep:binary-question-not-check",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "已多次培训客户软件打不开时在任务管理器中退出残留进程操作。但是客户夜班工程就是遇事不决就重启设备。",
            "content_summary": "已多次培训客户软件打不开时在任务管理器中退出残留进程操作。但是客户夜班工程就是遇事不决就重启设备。",
        }],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "是设备重启了还是软件退出了？", "content_summary": "是设备重启了还是软件退出了？"}],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "已多次培训客户软件打不开时在任务管理器中退出残留进程操作。但是客户夜班工程就是遇事不决就重启设备。",
            "debug_actions": ["是设备重启了还是软件退出了？"],
            "missing_info_requests": [],
        },
    }

    candidate = KnowledgeExtractionAgent().extract(episode)

    assert not [node for node in candidate["nodes"] if node["type"] == "DiagnosticCheck"]
    gate = QualityGateAgent().score(candidate)
    assert gate["passed"] is False
    assert "missing_check_or_solution" in gate["issues"]

def test_w2_label_strips_person_and_opinion_prefix_from_hardware_fault():
    episode = {
        "episode_id": "ep-person-hardware",
        "thread_id": "thread-person-hardware",
        "completeness": "partial",
        "fault_description_messages": [{
            "message_id": "m1",
            "text": "工程师乙，看起来显卡有问题",
            "content_summary": "工程师乙，看起来显卡有问题",
        }],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "extracted": {
            "symptom_raw": "工程师乙，看起来显卡有问题",
            "debug_actions": [],
            "missing_info_requests": [],
        },
        "evidence_message_ids": ["m1"],
    }

    candidate = KnowledgeExtractionAgent().extract(episode)

    assert candidate["label"] == "显卡有问题"
    assert candidate["symptom_raw"] == "显卡有问题"


def test_w2_label_strips_first_person_prefix_from_blue_screen_cause():
    text = "我想把蓝屏的原因找到了，看看给他锁个频，能不能稳定一点。"
    episode = {
        "episode_id": "ep-first-person-bsod",
        "thread_id": "thread-first-person-bsod",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": text, "content_summary": text}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "extracted": {"symptom_raw": text, "debug_actions": [], "missing_info_requests": []},
        "evidence_message_ids": ["m1"],
    }

    candidate = KnowledgeExtractionAgent().extract(episode)

    assert candidate["label"] == "蓝屏的原因找到了"
    assert candidate["symptom_raw"] == "蓝屏的原因找到了"

def test_w2_consumes_proj_manifest_hints_as_candidate_evidence_and_required_info_roles():
    episode = {
        "episode_id": "ep:req-proj-manifest-hints",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场导入配方后检测框异常。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m0", "m1", "m2"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场导入配方后检测框异常。",
            "debug_actions": [],
            "conclusion": "",
            "missing_info_requests": [{
                "message_id": "m1",
                "text": "请提供程序文件和配方。",
                "thread_id": "t",
                "context_before": [{"text": "现场导入配方后检测框异常。"}],
                "context_after": [{"message_id": "m2", "text": "已上传 recipe.proj。"}],
                "evidence_message_ids": ["m1", "m2"],
                "provided_later": True,
                "provided_evidence_message_ids": ["m2"],
            }],
            "tool_evidence": {
                "attachment_parse_results": [{
                    "type": "AttachmentParseResult",
                    "name": "recipe.proj",
                    "path": "/tmp/recipe.proj",
                    "evidence_role": "program_file",
                    "source": {"message_id": "m2"},
                    "content_read": False,
                    "archive_extracted": False,
                }],
                "jira_parse_results": [],
                "log_package_parse_results": [],
                "proj_parse_results": [{
                    "type": "ProjParseResult",
                    "path": "/tmp/recipe.proj",
                    "archive_format": "tar",
                    "archive_manifest_read": True,
                    "archive_extracted": False,
                    "key_hints": {
                        "versions": ["1.3.5"],
                        "app_versions": ["1.3.5"],
                        "ip_addresses": ["10.20.30.40"],
                        "project_names": ["BOARD_AOI_TEST"],
                        "model_types": ["MODEL_TYPE_DETECTION_COMPONENT"],
                        "file_roles": ["project_meta", "revision_meta", "component_table", "board_image"],
                        "has_board_images": True,
                    },
                }],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert "recipe.proj" in candidate["project_files"]
    assert "BOARD_AOI_TEST" in candidate["project_names"]
    assert "MODEL_TYPE_DETECTION_COMPONENT" in candidate["project_model_types"]
    assert "component_table" in candidate["project_file_roles"]
    assert "10.20.30.40" in candidate["ip_configs"]
    required = next(item for item in candidate["required_info_candidates"] if item["slot"] == "program_file")
    assert "program_file" in required["provided_tool_roles"]
    assert "proj_parsed" in required["provided_tool_roles"]
    assert "project_name" in required["provided_tool_roles"]
    assert "proj_component_table" in required["provided_tool_roles"]
    assert "proj_board_images" in required["provided_tool_roles"]


def test_w2_consumes_jira_title_hints_as_candidate_evidence():
    episode = {
        "episode_id": "ep:jira-title-hints",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m0", "text": "现场反馈软件闪退，JIRA 已跟踪。"}],
        "diagnostic_chain_messages": [],
        "resolution_messages": [],
        "evidence_message_ids": ["m0"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "现场反馈软件闪退，JIRA 已跟踪。",
            "debug_actions": [],
            "conclusion": "",
            "tool_evidence": {
                "attachment_parse_results": [],
                "log_package_parse_results": [],
                "proj_parse_results": [],
                "jira_parse_results": [{
                    "type": "JiraParseResult",
                    "issue_keys": ["SMTAOITS-1234"],
                    "urls": [{"url": "https://jira.example.com/browse/SMTAOITS-1234", "type": "jira"}],
                    "title_hints": ["1.3.5 客户02 设备报错“应用异常”，之后闪退"],
                    "version_hints": ["1.3.5"],
                    "site_hints": ["客户02"],
                    "fetched": False,
                }],
            },
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["jira_ids"] == ["SMTAOITS-1234"]
    assert candidate["jira_titles"] == ["1.3.5 客户02 设备报错“应用异常”，之后闪退"]
    assert "1.3.5" in candidate["versions"]
    assert "客户02" in candidate["sites"]
    assert any(node["type"] == "SoftwareVersion" and node["version_string"] == "1.3.5" for node in candidate["nodes"])
    assert any(node["type"] == "Site" and node["name"] == "客户02" for node in candidate["nodes"])


def test_w1_w2_consume_safe_text_attachment_hints():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_text_attachment_202606050900000800_202606051000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈 AOI 主程序初始化失败，相机异常。"),
            _msg(seg, 2, "请提供版本、IP 和错误信息，用于判断初始化阶段。"),
            _msg(seg, 3, "已上传现场记录文本。", resources="1"),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [{
            "relative_path": "om_test_3/现场记录.txt",
            "status": "api_ok",
            "type": "file",
            "bytes": "120",
            "name": "现场记录.txt",
            "message_id": "om_test_3",
            "segment_id": seg,
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "create_time": "2026-06-05 09:03",
            "sender": "工程师丁",
            "copied": "true",
            "copied_bytes": "120",
        }])
        (root / "om_test_3").mkdir(parents=True)
        (root / "om_test_3" / "现场记录.txt").write_text(
            "version=1.3.5\ncamera_ip=192.168.1.10\nERROR init camera failed 0x80070005\nJIRA=SMTAOITS-1234\n",
            encoding="utf-8",
        )
        _write_csv(manifest / "xing_segments.csv", [{
            "segment_id": seg,
            "index": "1",
            "chat_id": "oc_text_attachment",
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "start": "2026-06-05T09:00:00+08:00",
            "end": "2026-06-05T10:00:00+08:00",
            "messages": "3",
            "hits": "3",
            "resources": "1",
            "uploadable": "1",
            "unavailable": "0",
            "xing_rule_ok": "true",
            "xing_before_first": "0",
            "xing_after_last": "0",
            "xing_total": "3",
            "xing_count": "3",
        }])

        run = ChatCollectAgent().import_xing_upload(root)
        episode = run["episodes"][0]
        attachment_result = episode["extracted"]["tool_evidence"]["attachment_parse_results"][0]
        assert attachment_result["evidence_role"] == "data_file"
        assert attachment_result["text_preview_read"] is True
        assert "0x80070005" in attachment_result["key_hints"]["error_codes"]

        candidate = KnowledgeExtractionAgent().extract(episode)
        assert "1.3.5" in candidate["versions"]
        assert "192.168.1.10" in candidate["ip_configs"]
        assert "SMTAOITS-1234" in candidate["jira_ids"]
        assert "0x80070005" in candidate["attachment_error_codes"]
        assert any("camera failed" in line for line in candidate["attachment_error_hints"])
        required = candidate["required_info_candidates"][0]
        assert "attachment_text_preview" in required["provided_tool_roles"]
        assert "software_version" in required["provided_tool_roles"]
        assert "ip_config" in required["provided_tool_roles"]
        assert "attachment_error_hints" in required["provided_tool_roles"]


def test_w1_w2_w6_consume_image_header_metadata_without_ocr():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_image_attachment_202606060900000800_202606061000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈 AOI 漏检，已上传缺陷截图。", resources="1"),
            _msg(seg, 2, "请提供对应原图或缺陷截图。"),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [{
            "relative_path": "om_test_1/缺陷截图.png",
            "status": "api_ok",
            "type": "image",
            "bytes": "64",
            "name": "缺陷截图.png",
            "message_id": "om_test_1",
            "segment_id": seg,
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "create_time": "2026-06-06 09:01",
            "sender": "工程师丁",
            "copied": "true",
            "copied_bytes": "64",
        }])
        (root / "om_test_1").mkdir(parents=True)
        (root / "om_test_1" / "缺陷截图.png").write_bytes(_png_header(1280, 720))
        _write_csv(manifest / "xing_segments.csv", [{
            "segment_id": seg,
            "index": "1",
            "chat_id": "oc_image_attachment",
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "start": "2026-06-06T09:00:00+08:00",
            "end": "2026-06-06T10:00:00+08:00",
            "messages": "2",
            "hits": "2",
            "resources": "1",
            "uploadable": "1",
            "unavailable": "0",
            "xing_rule_ok": "true",
            "xing_before_first": "0",
            "xing_after_last": "0",
            "xing_total": "2",
            "xing_count": "2",
        }])

        run = ChatCollectAgent().import_xing_upload(root)
        episode = run["episodes"][0]
        image_result = episode["extracted"]["tool_evidence"]["image_parse_results"][0]
        assert image_result["image_format"] == "png"
        assert image_result["width"] == 1280
        assert image_result["height"] == 720
        assert image_result["pixels_read"] is False
        assert image_result["ocr_performed"] is False

        candidate = KnowledgeExtractionAgent().extract(episode)
        assert candidate["sample_images"] == ["缺陷截图.png"]
        assert candidate["sample_image_dimensions"] == ["1280x720"]
        assert candidate["sample_image_formats"] == ["png"]
        assert candidate["sample_image_metadata"][0]["width"] == 1280

        store = JsonKGStore(Path(tmp) / "kg")
        queue = ReviewQueueAgent(store).build_review_item(
            "candidates",
            candidate,
            episode,
            {"decision": "Agree", "requires_human": True},
            {"passed": True, "issues": []},
        )
        evidence = queue["evidence_pack"]["tool_evidence"]
        assert evidence["image_parse_results"][0]["width"] == 1280
        assert evidence["image_parse_results"][0]["ocr_performed"] is False


def test_w1_w2_w6_consume_document_metadata_without_ocr_or_execution():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "upload"
        manifest = root / "_MANIFEST"
        seg = "oc_document_attachment_202606070900000800_202606071000000800_1"
        _write_csv(manifest / "xing_messages.csv", [
            _msg(seg, 1, "客户反馈 AOI 主程序初始化失败，相机异常。"),
            _msg(seg, 2, "请提供诊断报告和日志文件，用于判断初始化阶段。"),
            _msg(seg, 3, "已上传现场报告。", resources="1"),
        ])
        _write_csv(manifest / "xing_resource_files.csv", [{
            "relative_path": "om_test_3/现场报告.pdf",
            "status": "api_ok",
            "type": "file",
            "bytes": "96",
            "name": "现场报告.pdf",
            "message_id": "om_test_3",
            "segment_id": seg,
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "create_time": "2026-06-07 09:03",
            "sender": "工程师丁",
            "copied": "true",
            "copied_bytes": "96",
        }])
        (root / "om_test_3").mkdir(parents=True)
        (root / "om_test_3" / "现场报告.pdf").write_bytes(
            b"%PDF-1.7\n1 0 obj << /Type /Page >> endobj\nBT (AOI init failed) Tj ET\n%%EOF"
        )
        _write_csv(manifest / "xing_segments.csv", [{
            "segment_id": seg,
            "index": "1",
            "chat_id": "oc_document_attachment",
            "chat_name": "【1.0已签单】江苏客户01项目群",
            "start": "2026-06-07T09:00:00+08:00",
            "end": "2026-06-07T10:00:00+08:00",
            "messages": "3",
            "hits": "3",
            "resources": "1",
            "uploadable": "1",
            "unavailable": "0",
            "xing_rule_ok": "true",
            "xing_before_first": "0",
            "xing_after_last": "0",
            "xing_total": "3",
            "xing_count": "3",
        }])

        run = ChatCollectAgent().import_xing_upload(root)
        episode = run["episodes"][0]
        document_result = episode["extracted"]["tool_evidence"]["document_parse_results"][0]
        assert document_result["document_format"] == "pdf"
        assert document_result["pdf_version"] == "1.7"
        assert document_result["page_count_hint"] == 1
        assert document_result["ocr_performed"] is False
        assert document_result["archive_extracted"] is False
        assert document_result["macros_executed"] is False

        candidate = KnowledgeExtractionAgent().extract(episode)
        assert candidate["data_files"] == ["现场报告.pdf"]
        assert candidate["document_formats"] == ["pdf"]
        assert candidate["document_page_count_hints"] == ["1"]
        assert candidate["document_metadata"][0]["ocr_performed"] is False
        assert any("AOI init failed" in preview for preview in candidate["document_text_previews"])

        required = candidate["required_info_candidates"][0]
        assert "document_metadata" in required["provided_tool_roles"]
        assert "document_text_preview" in required["provided_tool_roles"]

        store = JsonKGStore(Path(tmp) / "kg")
        queue = ReviewQueueAgent(store).build_review_item(
            "candidates",
            candidate,
            episode,
            {"decision": "Agree", "requires_human": True},
            {"passed": True, "issues": []},
        )
        evidence = queue["evidence_pack"]["tool_evidence"]
        assert evidence["document_parse_results"][0]["document_format"] == "pdf"
        assert evidence["document_parse_results"][0]["ocr_performed"] is False


def test_w2_outputs_trace_and_non_verified_outcome_without_resolved_by_for_failed_attempt():
    episode = {
        "episode_id": "ep:case001-like",
        "thread_id": "t",
        "completeness": "partial",
        "fault_description_messages": [{"message_id": "m1", "text": "编程拍照速度延迟卡顿。"}],
        "diagnostic_chain_messages": [{"message_id": "m2", "text": "更换采集卡无效，继续检查CXP线。"}],
        "resolution_messages": [{"message_id": "m3", "text": "最终判断相机组件有问题，更换相机需要返厂重标，成本高。"}],
        "evidence_message_ids": ["m1", "m2", "m3"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "编程拍照速度延迟卡顿。",
            "debug_actions": ["更换采集卡无效", "检查CXP线失败"],
            "conclusion": "最终判断相机组件有问题，更换相机需要返厂重标，成本高。",
            "missing_info_requests": [],
        },
    }
    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["diagnostic_trace"]["trace_id"]
    outcome_types = {item["action_label"]: item["outcome_type"] for item in candidate["diagnostic_outcomes"]}
    assert outcome_types["更换采集卡无效"] == "ineffective"
    assert not any(edge["relation"] == "resolved_by" for edge in candidate["edges"])
    gate = QualityGateAgent().score(candidate)
    assert "historical_outcome:ineffective" in gate["issues"]


def test_w6_evidence_pack_includes_case_context_messages_for_noise_review():
    item = ReviewQueueAgent(JsonKGStore("data/kg")).build_review_item(
        "noise_candidates",
        {"candidate_id": "chatcand:context-only", "nodes": [], "edges": [], "schema_valid": True},
        {
            "episode_id": "ep:context-only",
            "evidence_message_ids": ["m-context"],
            "case_context_messages": [{"message_id": "m-context", "sender": {"name": "现场"}, "create_time": "2026-01-01", "text": "备注一下后续再看", "content_summary": "备注一下后续再看"}],
            "source_offsets": [{"message_id": "m-context"}],
        },
        {"decision": "Insufficient", "requires_human": True},
        {"passed": False, "issues": ["noise_episode"]},
    )
    assert item["evidence_pack"]["messages"][0]["message_id"] == "m-context"
    assert item["evidence_pack"]["messages"][0]["role"] == "case_context_messages"
    assert item["review_summary"]["evidence_counts"]["messages"] == 1


def test_w6_review_item_exposes_trace_outcomes_and_policy_preview():
    candidate = {
        "candidate_id": "chatcand:review-trace-outcome",
        "diagnostic_trace": {"trace_id": "trace:x"},
        "diagnostic_outcomes": [{"outcome_id": "outcome:x", "action_label": "更换采集卡", "outcome_type": "ineffective", "high_cost": False}],
        "case_variant_candidate": {"entry_role": "case_variant", "canonical_error_id": "err:camera"},
    }
    item = ReviewQueueAgent(JsonKGStore("data/kg")).build_review_item("candidates", candidate, {"episode_id": "ep:x"}, {}, {})
    assert item["diagnostic_trace"]["trace_id"] == "trace:x"
    assert item["diagnostic_outcomes"][0]["outcome_type"] == "ineffective"
    assert item["policy_preview"]["by_outcome_type"]["ineffective"] == 1
    assert item["review_summary"]["candidate_id"] == "chatcand:review-trace-outcome"
    assert item["review_summary"]["outcome_type_counts"]["ineffective"] == 1


def test_w2_w4_w6_promote_no_recurrence_fix_as_reviewable_verified_fix():
    episode = {
        "episode_id": "ep:macro-blue-screen-memory",
        "thread_id": "thread:macro-blue-screen-memory",
        "completeness": "complete",
        "fault_description_messages": [{"message_id": "m1", "text": "客户22设备蓝屏。"}],
        "diagnostic_chain_messages": [
            {"message_id": "m2", "text": "已管理员权限打开cmd 运行下 poolmon /p 回车"},
            {"message_id": "m3", "text": "我建议更换内存"},
            {"message_id": "m4", "text": "现场已更换内存条"},
            {"message_id": "m5", "text": "更换内存条后到现在未再出现蓝屏吧"},
        ],
        "resolution_messages": [],
        "evidence_message_ids": ["m1", "m2", "m3", "m4", "m5"],
        "source_offsets": [],
        "extracted": {
            "symptom_raw": "客户22设备蓝屏",
            "debug_actions": [
                "已管理员权限打开cmd 运行下 poolmon /p 回车",
                "我建议更换内存",
                "现场已更换内存条",
                "更换内存条后到现在未再出现蓝屏吧",
            ],
            "conclusion": "",
            "missing_info_requests": [],
        },
    }

    candidate = KnowledgeExtractionAgent().extract(episode)
    assert candidate["schema_valid"] is True, candidate["schema_issues"]
    assert candidate["conclusion"] == "更换内存条后到现在未再出现蓝屏吧"

    outcomes = {item["action_label"]: item for item in candidate["diagnostic_outcomes"]}
    assert outcomes["已管理员权限打开cmd 运行下 poolmon /p 回车"]["outcome_type"] == "diagnostic_method"
    assert outcomes["我建议更换内存"]["outcome_type"] == "pending_validation"
    assert outcomes["现场已更换内存条"]["outcome_type"] == "verified_fix"
    assert "更换内存条后到现在未再出现蓝屏吧" not in outcomes
    assert outcomes["更换内存条"]["outcome_type"] == "verified_fix"
    assert outcomes["更换内存条"]["needs_confirmation"] is True

    gate = QualityGateAgent().score(candidate)
    assert gate["passed"] is True
    assert "verified_fix_requires_human_confirmation" in gate["issues"]
    assert "diagnostic_method_not_solution" not in gate["issues"]

    item = ReviewQueueAgent(JsonKGStore("data/kg")).build_review_item(
        "candidates",
        candidate,
        episode,
        {"decision": "Agree", "requires_human": True},
        gate,
    )
    assert item["review_summary"]["inferred_conclusion"] == "更换内存条后到现在未再出现蓝屏吧"
    assert item["review_summary"]["fix_evidence_candidates"]
    assert any(x["needs_confirmation"] for x in item["review_summary"]["fix_evidence_candidates"])


def test_w2_deepseek_hook_is_default_safe_and_missing_key_does_not_block():
    candidate = KnowledgeExtractionAgent(deepseek_enabled=True).extract({
        "thread_id": "t-deepseek-safe",
        "extracted": {"symptom_raw": "相机拍照失败", "debug_actions": ["检查相机连接"], "conclusion": ""},
        "evidence_message_ids": ["m1"],
    })
    assert candidate["schema_valid"] is True
    assert candidate["observability"]["deepseek_enabled"] is True
    assert candidate["observability"]["deepseek_error"] == "missing_DEEPSEEK_API_KEY"


def test_deepseek_w2_tool_schema_is_strict_and_validates_locally():
    from debug_agent_system.agents.write.w2_extract import _deepseek_w2_tool_schema, _tool_schema_strict_issues

    schema = _deepseek_w2_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["strict"] is True
    assert schema["function"]["name"] == "extract_w2_kg_candidate"
    assert _tool_schema_strict_issues(schema) == []
    params = schema["function"]["parameters"]
    assert set(params["properties"]) == set(params["required"])
    assert params["additionalProperties"] is False


def test_deepseek_w2_tool_call_response_is_parsed_without_network_leak():
    import os
    import urllib.request
    from debug_agent_system.agents.write import w2_extract as w2_mod

    original_urlopen = urllib.request.urlopen
    original_use_tools = os.environ.get("DEEPSEEK_W2_USE_TOOLS")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "function": {
                                "name": "extract_w2_kg_candidate",
                                "arguments": json.dumps({
                                    "case_variant_candidate": {"label": "相机拍照失败", "category": "硬件与运控", "subsystem": "相机", "scenario": "拍照失败", "canonical_error_id": "", "escalation_target": "hardware"},
                                    "diagnostic_trace": {"recommended_order": ["检查相机IP"], "actual_order": ["检查相机IP"], "summary": "检查相机链路"},
                                    "diagnostic_outcomes": [{"action_label": "检查相机IP", "outcome_type": "diagnostic_method", "condition": "", "target_check_id": "", "target_solution_id": "", "high_cost": False, "destructive": False, "observed_duration": "", "root_cause_summary": "", "evidence_message_ids": ["m1"]}],
                                    "required_info_candidates": [{"slot": "log_package", "label": "日志", "question": "请提供日志", "why_required": "定位拍照失败阶段", "condition": "", "priority": "high", "target_error_id": "", "provided_later": False, "provided_evidence_message_ids": [], "evidence_message_ids": ["m1"]}],
                                    "split_decision": {"should_split": False, "reason": "single", "marker_count": 1},
                                }, ensure_ascii=False),
                            }
                        }]
                    }
                }]
            }, ensure_ascii=False).encode("utf-8")

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    try:
        os.environ["DEEPSEEK_W2_USE_TOOLS"] = "1"
        urllib.request.urlopen = fake_urlopen
        out = w2_mod._call_deepseek_w2_extractor({
            "source_episode_id": "ep:tool",
            "source_thread_id": "thread:tool",
            "label": "相机拍照失败",
            "symptom_raw": "相机拍照失败",
            "debug_actions": ["检查相机IP"],
            "conclusion": "",
            "evidence_ids": ["m1"],
            "source_offsets": [],
            "semantic_text": "相机拍照失败，检查相机IP",
        }, api_key="fake-key")
    finally:
        urllib.request.urlopen = original_urlopen
        if original_use_tools is None:
            os.environ.pop("DEEPSEEK_W2_USE_TOOLS", None)
        else:
            os.environ["DEEPSEEK_W2_USE_TOOLS"] = original_use_tools

    assert captured["url"].endswith("/beta/chat/completions")
    assert captured["payload"]["tools"][0]["function"]["strict"] is True
    assert out["diagnostic_outcomes"][0]["outcome_type"] == "diagnostic_method"
    assert out["required_info_candidates"][0]["slot"] == "log_package"


def test_extract_xing_w1w2_cli_writes_real_stage_outputs_for_fake_archive():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_xing_upload(Path(tmp) / "xing")
        out_dir = Path(tmp) / "out"
        stdout = StringIO()
        with redirect_stdout(stdout):
            rc = cli_main(["extract-xing-w1w2", str(root), "--out-dir", str(out_dir), "--sample-limit", "2"])
        assert rc == 0
        result = json.loads(stdout.getvalue())
        assert result["summary"]["messages"] == 9
        assert result["summary"]["episodes"] >= 2
        assert result["summary"]["w2_candidates"] == result["summary"]["episodes"]
        assert Path(result["output_files"]["w2_candidates"]).exists()
        assert Path(result["output_files"]["w1_w2_summary"]).exists()
        assert result["samples"]


def _manual_review_candidates() -> list[tuple[str, dict]]:
    root = Path("data/kg/review_queue/manual_review_examples")
    out: list[tuple[str, dict]] = []
    for path in sorted(root.glob("chat-rank-*.json")):
        payload = json.loads(path.read_text())
        proposal = payload.get("refined_merge_proposal") or {}
        sub_candidates = []
        if proposal.get("nodes"):
            sub_candidates.append(proposal)
        for key in ("primary_candidate", "secondary_candidate"):
            if isinstance(proposal.get(key), dict):
                sub_candidates.append(proposal[key])
        for idx, sub in enumerate(sub_candidates, start=1):
            edges = []
            for edge in sub.get("edges") or []:
                clean = dict(edge)
                if "relation" not in clean and clean.get("type"):
                    clean["relation"] = clean["type"]
                edges.append(clean)
            nodes = [dict(node) for node in sub.get("nodes") or []]
            outcomes = [node for node in nodes if node.get("type") == "DiagnosticOutcome"]
            out.append((f"{path.name}:{idx}", {
                "candidate_id": f"manual:{path.stem}:{idx}",
                "schema_valid": True,
                "status": "pending_review",
                "source_episode_id": payload.get("source_episode_id") or "",
                "source_thread_id": payload.get("source_thread_id") or "",
                "nodes": nodes,
                "edges": edges,
                "diagnostic_outcomes": outcomes,
                "case_variant_candidate": next((node for node in nodes if node.get("type") == "Error"), {}),
                "evidence_ids": sub.get("evidence_message_ids") or proposal.get("evidence_message_ids") or [],
            }))
    return out


def test_four_manual_review_examples_are_schema_dry_run_compatible_and_safe():
    candidates = _manual_review_candidates()
    assert len(candidates) >= 4
    seen_files = {name.split(":", 1)[0] for name, _ in candidates}
    assert seen_files == {
        "chat-rank-240b3ff8f1e9.json",
        "chat-rank-68b3b3d0da80.json",
        "chat-rank-aa7f9f81327e.json",
        "chat-rank-b8f3c02dbdaf.json",
    }
    agent = IncrementalIngestAgent(JsonKGStore("data/kg"))
    for name, candidate in candidates:
        plan = agent.dry_run_merge_plan(candidate)
        assert plan["status"] == "dry_run_merge_plan", name
        assert plan["schema_valid"] is True, (name, plan["schema_issues"])
        for edge in candidate["edges"]:
            if edge.get("relation") == "resolved_by":
                solution_id = edge.get("to")
                linked = [out for out in candidate["diagnostic_outcomes"] if out.get("target_solution_id") == solution_id]
                assert not linked or all(out.get("outcome_type") == "verified_fix" for out in linked), (name, solution_id, linked)


def test_manual_golden_compare_reports_auto_vs_manual_metrics():
    from debug_agent_system.eval.write_side.manual_golden_compare import compare_manual_cases

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "manual"
        root.mkdir()
        episode_id = "thread:manual:episode:1"
        thread_id = "thread:manual"
        (root / "index.json").write_text(json.dumps({
            "cases": [{
                "case_id": "manual-001",
                "sample_id": "chat-rank:manual",
                "source_episode_id": episode_id,
                "source_thread_id": thread_id,
                "file": "case.json",
            }]
        }, ensure_ascii=False), encoding="utf-8")
        (root / "case.json").write_text(json.dumps({
            "sample_id": "chat-rank:manual",
            "source_episode_id": episode_id,
            "source_thread_id": thread_id,
            "refined_merge_proposal": {
                "nodes": [
                    {"type": "Error", "error_id": "err:manual-camera", "label": "相机拍照失败", "symptom": "相机拍照失败", "category": "硬件与运控", "entry_role": "canonical", "required_info_schema": [{"slot": "dmp_package", "question": "DMP?"}, {"slot": "graphics_driver_version", "question": "驱动版本?"}]},
                    {"type": "DiagnosticCheck", "check_id": "check:manual-camera-ip", "label": "检查相机IP", "how_to_check": "检查相机IP", "step_order": 1},
                    {"type": "DiagnosticOutcome", "outcome_id": "outcome:manual-camera-ip", "source_episode_id": episode_id, "target_error_id": "err:manual-camera", "target_check_id": "check:manual-camera-ip", "action_label": "检查相机IP", "outcome_type": "diagnostic_method", "evidence_message_ids": ["m1"]},
                    {"type": "DiagnosticTrace", "trace_id": "trace:manual-camera", "source_episode_id": episode_id, "target_error_id": "err:manual-camera", "recommended_order": [{"order": 1, "check_id": "check:manual-camera-ip", "label": "检查相机IP", "evidence_message_ids": ["m1"]}], "actual_order": [{"order": 1, "check_id": "check:manual-camera-ip", "label": "检查相机IP", "evidence_message_ids": ["m1"]}], "evidence_message_ids": ["m1"]},
                ],
                "edges": [
                    {"from": "err:manual-camera", "to": "check:manual-camera-ip", "relation": "has_check"},
                    {"from": "err:manual-camera", "to": "trace:manual-camera", "relation": "has_trace"},
                    {"from": "err:manual-camera", "to": "outcome:manual-camera-ip", "relation": "has_outcome"},
                ],
            },
        }, ensure_ascii=False), encoding="utf-8")
        episodes = [{
            "episode_id": episode_id,
            "thread_id": thread_id,
            "completeness": "partial",
            "extracted": {"symptom_raw": "相机拍照失败", "debug_actions": ["检查相机IP"], "conclusion": "", "missing_info_requests": [{"message_id": "m1", "text": "请提供 DMP 和驱动版本", "evidence_message_ids": ["m1"]}]},
            "evidence_message_ids": ["m1"],
            "fault_description_messages": [{"message_id": "m1", "text": "相机拍照失败"}],
            "diagnostic_chain_messages": [{"message_id": "m2", "text": "检查相机IP"}],
            "resolution_messages": [],
            "source_offsets": [{"message_id": "m1"}],
        }]
        episodes_path = Path(tmp) / "episodes.json"
        episodes_path.write_text(json.dumps(episodes, ensure_ascii=False), encoding="utf-8")
        report = compare_manual_cases(manual_root=root, episodes_path=episodes_path, deepseek=False)

        assert report["summary"]["cases"] == 1
        assert report["summary"]["compared"] == 1
        assert report["summary"]["schema_valid_rate"] == 1.0
        assert report["details"][0]["scores"]["check_label_recall"] >= 1.0
        assert report["details"][0]["scores"]["required_info_slot_recall"] >= 0.5
        assert "ready_for_batch_candidates" in report["summary"]


def test_write_batch_gate_validates_review_only_batch_artifacts():
    from debug_agent_system.eval.write_side.batch_candidate_gate import gate_run_dir

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "write_batch"
        queue_dir = run / "review_queue"
        queue_dir.mkdir(parents=True)
        (run / "ingest_xing_summary.json").write_text(json.dumps({
            "summary": {"applied": 0, "required_info_applied": 0},
            "review_summary": {
                "candidates": 0,
                "merge_candidates": 1,
                "noise_candidates": 0,
                "ask_info_candidates": 1,
            },
        }, ensure_ascii=False), encoding="utf-8")
        common_evidence = {
            "messages": [{"message_id": "m1", "text": "相机拍摄失败"}],
            "attachments": [],
            "source_offsets": [{"message_id": "m1"}],
        }
        (queue_dir / "merge_candidates.json").write_text(json.dumps([{
            "review_id": "review:batch-1",
            "queue": "merge_candidates",
            "candidate": {
                "candidate_id": "chatcand:batch-1",
                "schema_valid": True,
                "observability": {"deepseek_enabled": True, "deepseek_used": True, "deepseek_error": ""},
            },
            "episode": {"episode_id": "ep:1"},
            "quality_gate": {"passed": True},
            "evidence_pack": common_evidence,
            "review_actions": ["approve", "reject", "merge_existing", "request_more_info"],
            "review_summary": {"title": "相机拍摄失败", "candidate_id": "chatcand:batch-1"},
            "dry_run_merge_plan": {"status": "dry_run_merge_plan"},
        }], ensure_ascii=False), encoding="utf-8")
        (queue_dir / "ask_info_candidates.json").write_text(json.dumps([{
            "review_id": "review:req-1",
            "queue": "ask_info_candidates",
            "required_info_candidate": {"candidate_id": "reqinfo:1", "slot": "log_package", "question": "请提供日志。"},
            "episode": {"episode_id": "ep:1"},
            "quality_gate": {"passed": True},
            "evidence_pack": common_evidence,
            "review_actions": ["accept", "merge", "drop", "needs_owner", "needs_better_evidence"],
            "review_summary": {"title": "日志", "candidate_id": "reqinfo:1"},
            "dry_run_required_info_merge_plan": {"status": "dry_run_required_info_merge_plan"},
        }], ensure_ascii=False), encoding="utf-8")

        report = gate_run_dir(run, min_candidates=1, min_deepseek_used_rate=1.0)
        assert report["status"] == "passed", report["issues"]
        assert report["checks"]["queue_counts"]["merge_candidates"] == 1
        assert report["checks"]["deepseek"]["used_rate"] == 1.0


def test_four_manual_review_examples_apply_recompute_policy_and_feed_read_side():
    from debug_agent_system.agents.read.bd_traversal import TopologyTraversalAgent
    from debug_agent_system.core.contracts import SessionState

    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "kg"
        for folder in ("errors", "checks", "solutions", "sites", "versions", "traces", "outcomes", "policies"):
            (kg / "instances" / folder).mkdir(parents=True, exist_ok=True)
        (kg / "review_queue").mkdir(parents=True)
        (kg / "edges.json").write_text("[]\n", encoding="utf-8")

        candidates = _manual_review_candidates()
        for _, candidate in candidates:
            approved = dict(candidate)
            approved["status"] = "approved"
            approved["human_approved"] = True
            plan = IncrementalIngestAgent(JsonKGStore(kg)).dry_run_merge_plan(approved)
            assert plan["schema_valid"] is True, plan["schema_issues"]
            result = JsonKGStore(kg).apply_approved(approved)
            assert result["status"] == "applied_to_graph", result

        store = JsonKGStore(kg)
        assert len(store.errors_by_id) >= 4
        assert len(store.traces_by_id) == len(candidates)
        assert len(store.policies_by_id) == len(store.errors_by_id)
        assert all(policy.get("deterministic_recompute") is True for policy in store.policies_by_id.values())

        case001 = store.load_locked_subgraph("err:programming-capture-speed-delay")
        assert case001.payload["_diagnostic_policy"]["solution_stats"]
        assert any(outcome.get("outcome_type") == "ineffective" for outcome in case001.payload["_diagnostic_outcomes"])
        state = SessionState("s:case001", "编程拍照速度延迟，更换采集卡无效，排查驱动无效")
        decision = TopologyTraversalAgent().first_step(state, case001)
        assert decision.status == "step"
        assert decision.check is not None
        assert decision.check.check_id not in {"check:programming-capture-delay-frame-grabber", "check:programming-capture-delay-driver"}


def test_case001_manual_review_preserves_failed_trials_as_outcomes_not_resolved_by():
    case001 = [candidate for name, candidate in _manual_review_candidates() if name.startswith("chat-rank-aa7f9f81327e")][0]
    outcomes = {str(item.get("action_label") or item.get("label") or ""): item.get("outcome_type") for item in case001["diagnostic_outcomes"]}
    joined = "\n".join(f"{k}:{v}" for k, v in outcomes.items())
    assert "更换采集卡" in joined and "ineffective" in joined
    assert "排查驱动" in joined and "ineffective" in joined
    assert "更换工控机" in joined and "ineffective" in joined
    assert "更换 CXP" in joined and "ineffective" in joined
    assert "更换相机" in joined and "pending_validation" in joined
    assert not any(edge.get("relation") == "resolved_by" for edge in case001["edges"])


def test_deepseek_extraction_sanitizes_slot_aliases_and_missing_evidence():
    from debug_agent_system.agents.write.w2_extract import _sanitize_deepseek_extraction, _validate_deepseek_extraction
    extraction = {
        "source_episode_id": "echoed-input-only-key",
        "case_variant_candidate": None,
        "diagnostic_trace": [],
        "diagnostic_outcomes": [{
            "action_label": "更换采集卡无效",
            "outcome_type": "verified_fix",
            "condition": "",
            "target_check_id": "",
            "target_solution_id": "",
            "high_cost": False,
            "destructive": False,
            "observed_duration": "",
            "root_cause_summary": "",
        }],
        "required_info_candidates": [
            {"slot": "memory_config", "label": "内存配置", "question": "内存频率？", "why_required": "判断内存稳定性", "condition": "", "priority": "high", "target_error_id": "", "provided_later": False, "provided_evidence_message_ids": []},
            {"slot": "dmp_package", "label": "DMP", "question": "DMP？", "why_required": "分析蓝屏", "condition": "", "priority": "high", "target_error_id": "", "provided_later": False, "provided_evidence_message_ids": []},
            {"slot": "graphics_driver_version", "label": "显卡驱动版本", "question": "显卡驱动版本？", "why_required": "判断驱动兼容性", "condition": "", "priority": "high", "target_error_id": "", "provided_later": False, "provided_evidence_message_ids": []},
            {"slot": "recurrence_after_driver_change", "label": "驱动变更后复发", "question": "变更后是否复发？", "why_required": "判断方案是否有效", "condition": "", "priority": "high", "target_error_id": "", "provided_later": False, "provided_evidence_message_ids": []},
            {"slot": "network_config", "label": "网络配置", "question": "网络配置？", "why_required": "判断路由和网段问题", "condition": "", "priority": "high", "target_error_id": "", "provided_later": False, "provided_evidence_message_ids": []},
        ],
        "split_decision": {},
    }
    fixed = _sanitize_deepseek_extraction(extraction, {
        "evidence_ids": ["m1", "m2"],
        "semantic_text": "更换采集卡无效。",
    })
    assert "source_episode_id" not in fixed
    assert fixed["case_variant_candidate"] == {}
    assert fixed["diagnostic_trace"] == {}
    assert fixed["diagnostic_outcomes"][0]["outcome_type"] == "ineffective"
    assert fixed["diagnostic_outcomes"][0]["target_solution_id"] == ""
    assert fixed["diagnostic_outcomes"][0]["evidence_message_ids"] == ["m1", "m2"]
    assert fixed["required_info_candidates"][0]["slot"] == "environment"
    assert fixed["required_info_candidates"][1]["slot"] == "log_package"
    assert fixed["required_info_candidates"][2]["slot"] == "software_version"
    assert fixed["required_info_candidates"][3]["slot"] == "repro_steps"
    assert fixed["required_info_candidates"][4]["slot"] == "ip_config"
    assert fixed["required_info_candidates"][0]["evidence_message_ids"] == ["m1", "m2"]
    assert _validate_deepseek_extraction(fixed) == []



def test_outcome_normalizer_handles_pending_mitigation_and_context_not_root_cause():
    from debug_agent_system.agents.write.w2_extract import _sanitize_deepseek_extraction

    extraction = {
        "diagnostic_outcomes": [
            {"action_label": "内存检测P95待双休执行", "outcome_type": "mitigation_observed", "target_solution_id": "sol:memtest"},
            {"action_label": "每天断电重启一次", "outcome_type": "partial_temporary", "target_solution_id": "sol:reboot"},
            {"action_label": "检查现场接地情况", "outcome_type": "mitigation_observed", "target_solution_id": "sol:ground"},
        ]
    }
    semantics = {"evidence_ids": ["m1"], "semantic_text": "内存检测P95待双休执行。每天断电重启一次用于保障生产连续性。检查现场接地情况，接地测量正常，不是根因。"}
    fixed = _sanitize_deepseek_extraction(extraction, semantics)["diagnostic_outcomes"]
    by_action = {item["action_label"]: item for item in fixed}
    assert by_action["内存检测P95待双休执行"]["outcome_type"] == "pending_validation"
    assert by_action["每天断电重启一次"]["outcome_type"] == "mitigation_observed"
    assert by_action["检查现场接地情况"]["outcome_type"] == "context_not_root_cause"
    assert all(not item.get("target_solution_id") for item in fixed)


def test_deepseek_sanitizer_uses_local_context_for_outcome_type():
    from debug_agent_system.agents.write.w2_extract import _sanitize_deepseek_extraction

    semantics = {
        "evidence_ids": ["m1"],
        "semantic_text": "排查驱动，暂未排查出驱动问题。machine版本从8.0.2版本退回7.2.3版本测试，拍摄速度正常2h后再次开始延迟。调整内存频率从自动改为2666Hz后观察。",
    }
    extraction = {
        "diagnostic_outcomes": [
            {"action_label": "检查驱动", "outcome_type": "diagnostic_method", "target_solution_id": "sol:driver"},
            {"action_label": "machine版本从8.0.2版本退回7.2.3版本测试", "outcome_type": "pending_validation", "target_solution_id": "sol:version"},
            {"action_label": "调整内存频率从自动改为2666Hz", "outcome_type": "ineffective", "target_solution_id": "sol:memory"},
        ]
    }
    fixed = _sanitize_deepseek_extraction(extraction, semantics)["diagnostic_outcomes"]
    by_action = {item["action_label"]: item for item in fixed}
    assert by_action["检查驱动"]["outcome_type"] == "ineffective"
    assert by_action["检查驱动"]["target_solution_id"] == ""
    assert by_action["machine版本从8.0.2版本退回7.2.3版本测试"]["outcome_type"] == "partial_temporary"
    assert by_action["machine版本从8.0.2版本退回7.2.3版本测试"]["target_solution_id"] == ""
    assert by_action["调整内存频率从自动改为2666Hz"]["outcome_type"] == "mitigation_observed"
    assert by_action["调整内存频率从自动改为2666Hz"]["target_solution_id"] == ""


def test_deepseek_sanitizer_demotes_unverified_success_claims():
    from debug_agent_system.agents.write.w2_extract import _sanitize_deepseek_extraction

    extraction = {
        "diagnostic_outcomes": [
            {"action_label": "时间17:25 重启工控机正常开机", "outcome_type": "verified_fix", "target_solution_id": "sol:reboot"},
            {"action_label": "卸载并重装显卡驱动", "outcome_type": "verified_fix", "target_solution_id": "sol:driver"},
            {"action_label": "更换采集卡后恢复正常", "outcome_type": "verified_fix", "target_solution_id": "sol:capture-card"},
            {"action_label": "收集DMP文件分析", "outcome_type": "verified_fix", "target_solution_id": "sol:dmp"},
        ]
    }
    fixed = _sanitize_deepseek_extraction(extraction, {
        "evidence_ids": ["m1"],
        "semantic_text": (
            "时间17:25 重启工控机正常开机。卸载并重装显卡驱动。"
            "更换采集卡后恢复正常。收集DMP文件分析。"
        ),
    })["diagnostic_outcomes"]
    by_action = {item["action_label"]: item for item in fixed}
    assert by_action["时间17:25 重启工控机正常开机"]["outcome_type"] == "partial_temporary"
    assert by_action["时间17:25 重启工控机正常开机"]["target_solution_id"] == ""
    assert by_action["卸载并重装显卡驱动"]["outcome_type"] == "pending_validation"
    assert by_action["卸载并重装显卡驱动"]["target_solution_id"] == ""
    assert by_action["更换采集卡后恢复正常"]["outcome_type"] == "verified_fix"
    assert by_action["更换采集卡后恢复正常"]["target_solution_id"] == "sol:capture-card"
    assert by_action["收集DMP文件分析"]["outcome_type"] == "diagnostic_method"
    assert by_action["收集DMP文件分析"]["target_solution_id"] == ""

def test_w2_deepseek_enrichment_applies_sanitizer_before_validation():
    import os
    from debug_agent_system.agents.write import w2_extract as w2_mod

    original_key = os.environ.get("DEEPSEEK_API_KEY")
    original_call = w2_mod._call_deepseek_w2_extractor

    def fake_call(semantics, *, api_key):
        assert api_key == "fake-key"
        return {
            "case_variant_candidate": {
                "label": "工控机蓝屏重启",
                "category": "硬件与运控",
                "subsystem": "工控机/Windows系统",
                "scenario": "蓝屏重启",
                "canonical_error_id": "",
                "escalation_target": "industrial_pc",
            },
            "diagnostic_trace": {
                "recommended_order": ["收集 DMP", "分析事件查看器"],
                "actual_order": ["收集 DMP"],
                "summary": "先收集蓝屏 DMP 再分析系统日志。",
            },
            "diagnostic_outcomes": [
                {
                    "action_label": "收集 DMP",
                    "outcome_type": "diagnostic_method",
                    "condition": "dmp",
                    "target_check_id": "unknown-from-llm",
                    "target_solution_id": "unknown-solution-from-llm",
                    "high_cost": False,
                    "destructive": False,
                    "observed_duration": "",
                    "root_cause_summary": "",
                    "evidence_message_ids": [],
                }
            ],
            "required_info_candidates": [
                {
                    "slot": "dmp_package",
                    "label": "DMP",
                    "question": "请提供 DMP。",
                    "why_required": "分析蓝屏或重启原因。",
                    "condition": "",
                    "priority": "high",
                    "target_error_id": "",
                    "provided_later": False,
                    "provided_evidence_message_ids": [],
                },
                {
                    "slot": "root_cause_analysis",
                    "label": "根因分析",
                    "question": "请补充根因分析。",
                    "why_required": "DeepSeek 非枚举槽位应降级为 review-only。",
                    "condition": "",
                    "priority": "low",
                    "target_error_id": "",
                    "provided_later": False,
                    "provided_evidence_message_ids": [],
                }
            ],
            "split_decision": {},
        }

    try:
        os.environ["DEEPSEEK_API_KEY"] = "fake-key"
        w2_mod._call_deepseek_w2_extractor = fake_call
        candidate = w2_mod.KnowledgeExtractionAgent(deepseek_enabled=True).extract({
            "thread_id": "t-deepseek-sanitize",
            "extracted": {
                "symptom_raw": "工控机蓝屏重启",
                "debug_actions": ["收集 DMP", "分析事件查看器"],
                "conclusion": "",
            },
            "evidence_message_ids": ["m1"],
        })
    finally:
        w2_mod._call_deepseek_w2_extractor = original_call
        if original_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = original_key

    assert candidate["observability"]["deepseek_used"] is True
    assert candidate["observability"]["deepseek_error"] == ""
    req = candidate["deepseek_extraction"]["required_info_candidates"][0]
    assert req["slot"] == "log_package"
    assert req["evidence_message_ids"] == ["m1"]
    invalid_req = candidate["deepseek_extraction"]["required_info_candidates"][1]
    assert invalid_req["slot"] == "other"
    assert candidate["case_variant_candidate"]["subsystem"] == "工控机/Windows系统"
    assert candidate["nodes"][0]["subsystem"] == "工控机/Windows系统"
    assert candidate["diagnostic_trace"]["recommended_order"][0]["label"] == "收集 DMP"
    assert candidate["diagnostic_outcomes"][0]["action_label"] == "收集 DMP"
    assert candidate["diagnostic_outcomes"][0]["target_solution_id"] == ""
    assert candidate["schema_valid"] is True
