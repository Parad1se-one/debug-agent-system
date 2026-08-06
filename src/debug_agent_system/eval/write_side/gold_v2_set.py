"""Build and verify the immutable human-approved Goldcase 016--020 batch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDCASES_ROOT = REPO_ROOT / "data/annotations/goldcases"
SOURCE_ROOT = GOLDCASES_ROOT / "candidates/xing-lark-v1"
DEFAULT_ROOT = GOLDCASES_ROOT / "gold-v2"
MESSAGES_PATH = REPO_ROOT / "data/results/xing_relation_context_final_20260717/messages.jsonl"
MANIFEST_NAME = "gold-v2-016-020.manifest.json"
CASE_IDS = [f"goldcase-{number:03d}" for number in range(16, 21)]
MESSAGE_RE = re.compile(r"om_x[0-9a-z]{24,}")
JIRA_RE = re.compile(r"(?:SMTAOITS|TEST)-\d+")
TRACE_RE = re.compile(r"^## Trace\s+`?([0-9]{3}(?:-[a-z])?)`?(?:：|\s|$)", re.MULTILINE)


TRACE_TAXONOMY: dict[str, dict[str, str]] = {
    "016-a": {"family": "存储介质异常", "category": "硬件与运控", "subsystem": "工控机磁盘/文件系统", "variant": "CRC与Disk Event 7伴随蓝屏和系统不稳定"},
    "016-b": {"family": "系统与软件卡顿", "category": "系统与软件异常", "subsystem": "工控机图形软件/驱动与后台进程", "variant": "换盘后后台进程堆积和显卡软件清理后的短期恢复"},
    "017-a": {"family": "相机采集链路异常", "category": "硬件与运控", "subsystem": "CXP线缆/接头固定与图像采集", "variant": "首个FOV停滞且无拍摄失败报警"},
    "017-b": {"family": "拍照后板卡流转异常", "category": "系统与软件异常", "subsystem": "运控状态机与主程序生命周期", "variant": "现场误报不拍照但日志显示已完成拍照"},
    "017-c": {"family": "工控机异常重启", "category": "硬件与运控", "subsystem": "CPU散热链/工控机稳定性", "variant": "正常测试中黑屏自动重启（CPU过温保护）"},
    "018": {"family": "项目/模板敏感型检测启动失败", "category": "系统与软件异常", "subsystem": "项目模板/器件库数据生成与SDK项目加载", "variant": "特定.proj跨设备导致相机不运动并报拍摄失败"},
    "019-a": {"family": "同步复判流程卡死", "category": "系统与软件异常", "subsystem": "同步复判/误报处理流程", "variant": "未配置复判站时误报转圈并自动退出"},
    "019-b": {"family": "模型加载与系统稳定性异常", "category": "系统与软件异常", "subsystem": "模型认证/内存与本地数据完整性", "variant": "跨模型认证失败、加载模板闪退及内存硬件故障"},
    "020-a": {"family": "运动控制初始化失败", "category": "硬件与运控", "subsystem": "运动控制卡访问链与网卡驱动", "variant": "正常关机后开机时AOI与E450均无法连接运控卡"},
}


class GoldV2IntegrityError(ValueError):
    """Raised when the frozen batch or its source review contract is invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _all_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_strings(child)]
    return []


def _evidence_ids(text: str) -> list[str]:
    result = list(dict.fromkeys(MESSAGE_RE.findall(text)))
    result.extend(f"jira:{key}" for key in JIRA_RE.findall(text))
    return list(dict.fromkeys(result))


def _split_row(line: str) -> list[str]:
    return [cell.strip().replace("\\|", "|") for cell in line.strip().strip("|").split("|")]


def _table_after(block: str, heading_patterns: tuple[str, ...]) -> list[dict[str, str]]:
    lines = block.splitlines()
    start = -1
    for index, line in enumerate(lines):
        if line.startswith("### ") and any(pattern in line for pattern in heading_patterns):
            start = index + 1
            break
    if start < 0:
        return []
    while start < len(lines) and not lines[start].startswith("|"):
        start += 1
    if start + 1 >= len(lines):
        return []
    headers = _split_row(lines[start])
    rows: list[dict[str, str]] = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = _split_row(line)
        if len(cells) != len(headers):
            raise GoldV2IntegrityError(f"markdown_table_width_mismatch:{headers}:{cells}")
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _trace_blocks(review: str) -> dict[str, str]:
    matches = list(TRACE_RE.finditer(review))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(review)
        blocks[match.group(1)] = review[match.start():end]
    return blocks


def _clean_code(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def _normalise_occurrences(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        occurrence_ref = _clean_code(row.get("occurrence", ""))
        state = _clean_code(row.get("状态", row.get("现象/状态", "reported")))
        summary = row.get("摘要", row.get("现象/状态", ""))
        evidence_text = row.get("证据", row.get("关键证据", ""))
        result.append({
            "occurrence_ref": occurrence_ref,
            "time_range": row.get("时间范围", row.get("时间", "")),
            "device_scope": row.get("设备", ""),
            "state": state,
            "summary": summary,
            "source_evidence_ids": _evidence_ids(evidence_text),
            "reviewed_source_row": row,
        })
    return result


def _infer_role(label: str) -> str:
    for token, role in (("收集", "collect"), ("检查", "inspect"), ("核查", "inspect"), ("观察", "observe"), ("验证", "verify"), ("重装", "change"), ("更换", "change"), ("升级", "change"), ("提交", "escalate")):
        if token in label:
            return role
    return "act"


def _normalise_actions(rows: list[dict[str, str]], trace_ref: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        label = row.get("Action", "")
        outcome_type = _clean_code(row.get("Outcome", "pending_validation"))
        assessment = row.get("判定", "")
        result.append({
            "action_ref": f"{trace_ref}-action-{index}",
            "occurrence_ref": _clean_code(row.get("occurrence", "")) or None,
            "label": label,
            "action_role": _clean_code(row.get("role", "")) or _infer_role(label),
            "execution_status": _clean_code(row.get("execution", "")) or "reviewed_unspecified",
            "source_evidence_ids": _evidence_ids(" ".join(row.values())),
            "outcome": {
                "outcome_type": outcome_type,
                "summary": assessment,
                "source_evidence_ids": _evidence_ids(assessment),
            },
            "reviewed_source_row": row,
        })
    return result


def _normalise_timeline(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        evidence = row.get("证据", "")
        result.append({
            "order": int(row.get("顺序", index)) if str(row.get("顺序", index)).isdigit() else index,
            "state": _clean_code(row.get("state", row.get("状态", ""))),
            "causal_role": _clean_code(row.get("causal_role", "")),
            "summary": row.get("判断", ""),
            "source_evidence_ids": _evidence_ids(evidence),
            "reviewed_source_row": row,
        })
    return result


def _load_messages() -> dict[str, dict[str, Any]]:
    messages: dict[str, dict[str, Any]] = {}
    with MESSAGES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            message_id = str(item.get("message_id") or "")
            if not message_id:
                continue
            messages[message_id] = {
                key: item.get(key)
                for key in (
                    "message_id", "thread_id", "chat_id", "sender", "create_time", "msg_type",
                    "text", "attachments", "links", "root_id", "parent_id", "upper_message_id",
                    "relation_thread_id", "relation_source", "relation_aware_session_id",
                )
            }
    return messages


def _compact_jira(key: str) -> dict[str, Any] | None:
    path = REPO_ROOT / f"data/imports/jira_offline/raw/fault_details/{key}.json"
    if not path.is_file():
        return None
    item = json.loads(path.read_text(encoding="utf-8"))
    return {
        "evidence_id": f"jira:{key}",
        "key": key,
        "summary": item.get("summary"),
        "description": item.get("description"),
        "status": item.get("status"),
        "resolution": item.get("resolution"),
        "created": item.get("created"),
        "updated": item.get("updated"),
        "components": item.get("components") or [],
        "fix_versions": item.get("fix_versions") or [],
        "issue_links": item.get("issue_links") or [],
        "comments": item.get("comments") or [],
        "source_file": str(path.relative_to(REPO_ROOT)),
        "source_file_sha256": _sha256(path),
    }


def _external_artifacts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for message in messages:
        for index, attachment in enumerate(message.get("attachments") or [], start=1):
            if not isinstance(attachment, dict):
                attachment = {"value": attachment}
            row: dict[str, Any] = {
                "artifact_ref": f"{message['message_id']}:attachment:{index}",
                "source_message_ids": [message["message_id"]],
                "kind": attachment.get("mime_type") or attachment.get("type") or "attachment",
                "retrieval_status": "metadata_only",
                "content_used_for_annotation": False,
                "source_metadata": attachment,
            }
            local_value = next((attachment.get(key) for key in ("path", "local_path", "file_path") if attachment.get(key)), None)
            if local_value:
                local_path = Path(str(local_value))
                if not local_path.is_absolute():
                    local_path = REPO_ROOT / local_path
                if local_path.is_file():
                    row.update({
                        "path": str(local_path.relative_to(REPO_ROOT)),
                        "retrieval_status": "local_file_available",
                        "file_sha256": _sha256(local_path),
                        "size_bytes": local_path.stat().st_size,
                    })
            artifacts.append(row)
    return artifacts


def _make_trace(trace: dict[str, Any], block: str) -> dict[str, Any]:
    trace_ref = str(trace["trace_ref"])
    taxonomy = TRACE_TAXONOMY[trace_ref]
    occurrences = _normalise_occurrences(_table_after(block, ("Occurrences",)))
    actions = _normalise_actions(_table_after(block, ("Actions 与 Outcomes", "原子 Actions 与 Outcomes")), trace_ref)
    timeline = _normalise_timeline(_table_after(block, ("诊断状态演化", "诊断状态")))
    if not occurrences or not actions:
        raise GoldV2IntegrityError(f"{trace_ref}:missing_occurrences_or_actions")
    anchors = list(dict.fromkeys(
        list(trace.get("key_evidence_message_ids") or [])
        + [f"jira:{key}" for key in trace.get("jira_keys") or []]
        + [value for item in occurrences for value in item["source_evidence_ids"]]
    ))
    return {
        "trace_ref": trace_ref,
        "family": {key: taxonomy[key] for key in ("family", "category", "subsystem")},
        "variant": {"label": taxonomy["variant"]},
        "label": trace.get("label"),
        "time_range": trace.get("time_range"),
        "outcome_state": trace.get("outcome_state"),
        "root_cause_state": trace.get("root_cause_state"),
        "jira_keys": trace.get("jira_keys") or [],
        "evidence_anchor_ids": anchors,
        "occurrences": occurrences,
        "actions": actions,
        "hypothesis_timeline": timeline,
        "required_info": [
            {"slot": f"validation_gap_{index}", "question": text}
            for index, text in enumerate(trace.get("counterevidence") or [], start=1)
        ],
        "uncertainties": trace.get("counterevidence") or [],
    }


def _build_case(case_id: str, all_messages: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    selection_path = SOURCE_ROOT / "selections" / f"{case_id}.json"
    review_path = SOURCE_ROOT / "reviews" / f"{case_id}.md"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    review = review_path.read_text(encoding="utf-8")
    if selection.get("selection_status") != "accepted" or selection.get("graph_ingestion") is not False:
        raise GoldV2IntegrityError(f"{case_id}:selection_not_accepted_or_ingestion_enabled")
    review_blocks = _trace_blocks(review)
    traces = []
    for trace in selection.get("proposed_traces") or []:
        trace_ref = str(trace.get("trace_ref") or "")
        if trace_ref not in review_blocks:
            raise GoldV2IntegrityError(f"{case_id}:review_trace_missing:{trace_ref}")
        traces.append(_make_trace(trace, review_blocks[trace_ref]))

    source_text = review + "\n" + "\n".join(_all_strings(selection))
    message_ids = sorted(set(MESSAGE_RE.findall(source_text)), key=lambda value: (all_messages.get(value, {}).get("create_time") or "", value))
    missing = [message_id for message_id in message_ids if message_id not in all_messages]
    messages = [all_messages[message_id] for message_id in message_ids if message_id in all_messages]
    jira_keys = sorted(set(JIRA_RE.findall(source_text)))
    jira = [item for key in jira_keys if (item := _compact_jira(key)) is not None]
    artifacts = _external_artifacts(messages)
    messages_sha256 = _canonical_hash(messages)
    evidence_sha256 = _canonical_hash({
        "messages": messages,
        "linked_jira_issues": jira,
        "external_artifacts": artifacts,
        "unresolved_message_ids": missing,
    })
    input_payload = {
        "schema_version": "kg_v2.gold_input.v1",
        "case_id": case_id,
        "batch_id": "gold-v2-016-020",
        "source_kind": "human_curated_xing_lark_chat",
        "label_visibility": "source_only_no_ground_truth",
        "graph_ingestion": False,
        "chat_ids": [selection["chat"]["chat_id"]],
        "analysis_window": selection.get("analysis_window"),
        "selection_policy": selection.get("message_selection_policy"),
        "messages_sha256": messages_sha256,
        "input_evidence_sha256": evidence_sha256,
        "messages": messages,
        "linked_jira_issues": jira,
        "external_artifacts": artifacts,
        "unresolved_message_ids": missing,
    }
    truth_payload = {
        "schema_version": "kg_v2.gold_ground_truth.v1",
        "case_id": case_id,
        "batch_id": "gold-v2-016-020",
        "input_messages_sha256": messages_sha256,
        "input_evidence_sha256": evidence_sha256,
        "annotation_source": "interactive_human_review_with_message_attachment_and_jira_audit",
        "review_status": "approved",
        "graph_ingestion": False,
        "split_required": len(traces) > 1,
        "split_reason": selection.get("selection_reason"),
        "analysis_window": selection.get("analysis_window"),
        "analysis_method": selection.get("message_selection_policy"),
        "device_identity_map": selection.get("device_identity_map") or [],
        "trace_count": len(traces),
        "traces": traces,
        "excluded_parallel_faults": selection.get("excluded_parallel_faults") or [],
        "critical_expectations": [
            "Trace边界、设备身份、Action与Outcome不得跨Trace串用。",
            "保留失败动作、短暂恢复、复发、混杂因素和缺失验证。",
            "probable或unverified根因不得提升为confirmed。",
            "Goldcase接受不构成KG写入授权。",
        ],
        "source_review": {"file": f"reviews/{case_id}.md", "sha256": _sha256(review_path)},
        "source_selection": {"file": str(selection_path.relative_to(REPO_ROOT)), "sha256": _sha256(selection_path)},
        "human_review": {
            "decision": "approved",
            "reviewer": "user:workspace_owner",
            "reviewed_at": "2026-07-21",
            "basis": "interactive_manual_review_and_explicit_acceptance_of_goldcases_016_020",
        },
    }
    return input_payload, truth_payload, str(review_path), str(selection_path)


def build(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        raise GoldV2IntegrityError("batch_already_frozen; run verifier instead")
    for folder in (root / "inputs", root / "ground_truth", root / "reviews"):
        folder.mkdir(parents=True, exist_ok=True)
    all_messages = _load_messages()
    case_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        input_payload, truth_payload, review_source, selection_source = _build_case(case_id, all_messages)
        input_path = root / "inputs" / f"{case_id}.json"
        truth_path = root / "ground_truth" / f"{case_id}.json"
        review_path = root / "reviews" / f"{case_id}.md"
        _write_json(input_path, input_payload)
        shutil.copyfile(review_source, review_path)
        truth_payload["source_review"]["sha256"] = _sha256(review_path)
        _write_json(truth_path, truth_payload)
        input_rows.append({
            "case_id": case_id,
            "file": f"{case_id}.json",
            "message_count": len(input_payload["messages"]),
            "linked_jira_count": len(input_payload["linked_jira_issues"]),
            "external_artifact_count": len(input_payload["external_artifacts"]),
            "messages_sha256": input_payload["messages_sha256"],
            "input_evidence_sha256": input_payload["input_evidence_sha256"],
            "file_sha256": _sha256(input_path),
        })
        case_rows.append({
            "case_id": case_id,
            "input_file": f"inputs/{case_id}.json",
            "input_sha256": _sha256(input_path),
            "truth_file": f"ground_truth/{case_id}.json",
            "truth_sha256": _sha256(truth_path),
            "review_file": f"reviews/{case_id}.md",
            "review_sha256": _sha256(review_path),
            "selection_file": str(Path(selection_source).relative_to(REPO_ROOT)),
            "selection_sha256": _sha256(Path(selection_source)),
            "human_review": truth_payload["human_review"],
        })
    input_manifest = {
        "schema_version": "kg_v2.gold_input_manifest.v1",
        "batch_id": "gold-v2-016-020",
        "immutable": True,
        "contains_ground_truth": False,
        "graph_ingestion": False,
        "cases": input_rows,
    }
    input_manifest_path = root / "inputs/manifest.json"
    _write_json(input_manifest_path, input_manifest)
    readme = (
        "# Gold v2：Goldcase 016–020\n\n"
        "本批次固定 2026-07-21 经用户逐案确认的 Goldcase 016–020。"
        "`inputs/` 是 source-only 证据，`ground_truth/` 是结构化人工答案，`reviews/` 是冻结审核稿。\n\n"
        "该批次默认且明确 `graph_ingestion=false`；接受与冻结均不构成 KG 写入授权。\n"
    )
    readme_path = root / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    manifest = {
        "schema_version": "debug_agent_system.gold_set_manifest.v2",
        "gold_set_id": "gold-v2-016-020",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "immutable": True,
            "graph_ingestion": False,
            "notes": "Human-approved Goldcases 016-020. Any KG ingestion requires separate explicit authorization.",
        },
        "input_manifest": "inputs/manifest.json",
        "input_manifest_sha256": _sha256(input_manifest_path),
        "batch_files": [{"file": "README.md", "sha256": _sha256(readme_path)}],
        "cases": case_rows,
    }
    _write_json(manifest_path, manifest)
    return verify(root)


def verify(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise GoldV2IntegrityError(f"manifest_missing:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if manifest.get("policy", {}).get("immutable") is not True or manifest.get("policy", {}).get("graph_ingestion") is not False:
        issues.append("manifest_policy_invalid")
    input_manifest_path = root / str(manifest.get("input_manifest") or "")
    if not input_manifest_path.is_file() or _sha256(input_manifest_path) != manifest.get("input_manifest_sha256"):
        issues.append("input_manifest_hash_mismatch")
    rows = manifest.get("cases") or []
    if [row.get("case_id") for row in rows] != CASE_IDS:
        issues.append("case_ids_invalid")
    for row in rows:
        case_id = str(row.get("case_id") or "")
        for file_key, hash_key in (("input_file", "input_sha256"), ("truth_file", "truth_sha256"), ("review_file", "review_sha256")):
            path = root / str(row.get(file_key) or "")
            if not path.is_file() or _sha256(path) != row.get(hash_key):
                issues.append(f"{case_id}:{hash_key}_mismatch")
        selection_path = REPO_ROOT / str(row.get("selection_file") or "")
        if not selection_path.is_file() or _sha256(selection_path) != row.get("selection_sha256"):
            issues.append(f"{case_id}:selection_sha256_mismatch")
            continue
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selection_status") != "accepted" or selection.get("graph_ingestion") is not False:
            issues.append(f"{case_id}:selection_policy_invalid")
        input_path = root / str(row.get("input_file") or "")
        truth_path = root / str(row.get("truth_file") or "")
        if input_path.is_file() and truth_path.is_file():
            source = json.loads(input_path.read_text(encoding="utf-8"))
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            if source.get("graph_ingestion") is not False or truth.get("graph_ingestion") is not False:
                issues.append(f"{case_id}:graph_ingestion_enabled")
            if source.get("messages_sha256") != _canonical_hash(source.get("messages") or []):
                issues.append(f"{case_id}:messages_sha256_mismatch")
            evidence = {key: source.get(key) or [] for key in ("messages", "linked_jira_issues", "external_artifacts", "unresolved_message_ids")}
            if source.get("input_evidence_sha256") != _canonical_hash(evidence):
                issues.append(f"{case_id}:input_evidence_sha256_mismatch")
            if truth.get("input_messages_sha256") != source.get("messages_sha256") or truth.get("input_evidence_sha256") != source.get("input_evidence_sha256"):
                issues.append(f"{case_id}:truth_input_hash_mismatch")
            if truth.get("review_status") != "approved" or truth.get("human_review", {}).get("decision") != "approved":
                issues.append(f"{case_id}:human_review_invalid")
            traces = truth.get("traces") or []
            if truth.get("trace_count") != len(traces) or bool(truth.get("split_required")) != (len(traces) > 1):
                issues.append(f"{case_id}:trace_count_invalid")
            if any(not trace.get("occurrences") or not trace.get("actions") for trace in traces):
                issues.append(f"{case_id}:empty_trace_contract")
    for item in manifest.get("batch_files") or []:
        path = root / str(item.get("file") or "")
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            issues.append(f"batch_file_hash_mismatch:{item.get('file')}")
    report = {"ok": not issues, "manifest": str(manifest_path), "case_count": len(rows), "issues": sorted(set(issues))}
    if issues:
        raise GoldV2IntegrityError(json.dumps(report, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gold-v2-set")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    report = build(args.root) if args.build else verify(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
