"""Validate and render the 011--015 blind ground-truth review documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _cell(value: Any, limit: int = 600) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", "<br>").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _validate(input_payload: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    message_ids = {
        str(message.get("message_id") or "")
        for message in input_payload.get("messages") or []
        if isinstance(message, dict)
    }
    linked_evidence_ids = {
        str(item.get("evidence_id") or "")
        for item in input_payload.get("linked_jira_issues") or []
        if isinstance(item, dict)
    }
    artifact_evidence_ids = {
        str(item.get("artifact_ref") or "")
        for item in input_payload.get("external_artifacts") or []
        if isinstance(item, dict)
    }
    evidence_ids = message_ids | linked_evidence_ids | artifact_evidence_ids
    if truth.get("input_messages_sha256") != input_payload.get("messages_sha256"):
        issues.append("input_messages_sha256_mismatch")
    if truth.get("input_evidence_sha256") is not None and truth.get("input_evidence_sha256") != input_payload.get("input_evidence_sha256"):
        issues.append("input_evidence_sha256_mismatch")
    cases = [item for item in truth.get("cases") or [] if isinstance(item, dict)]
    if int(truth.get("case_count") or 0) != len(cases):
        issues.append("case_count_mismatch")
    if bool(truth.get("split_required")) != (len(cases) > 1):
        issues.append("split_flag_case_count_mismatch")
    for anchor_type in ("daily_report_anchors", "field_report_anchors"):
        for anchor in truth.get(anchor_type) or []:
            if not isinstance(anchor, dict):
                issues.append(f"{anchor_type[:-1]}_not_object")
                continue
            evidence_id = str(anchor.get("message_id") or "")
            if evidence_id not in message_ids:
                issues.append(f"unknown_{anchor_type[:-1]}:{evidence_id}")
    for audit in truth.get("artifact_audit") or []:
        if not isinstance(audit, dict):
            issues.append("artifact_audit_not_object")
            continue
        for evidence_id in audit.get("source_evidence_ids") or []:
            if evidence_id not in evidence_ids:
                issues.append(f"unknown_artifact_audit_evidence:{evidence_id}")
    for case in cases:
        case_ref = str(case.get("case_ref") or "missing")
        occurrences = [item for item in case.get("occurrences") or [] if isinstance(item, dict)]
        occurrence_refs = {str(item.get("occurrence_ref") or "") for item in occurrences}
        if len(occurrence_refs) != len(occurrences) or "" in occurrence_refs:
            issues.append(f"{case_ref}:invalid_or_duplicate_occurrence_ref")
        for occurrence in occurrences:
            occurrence_ref = str(occurrence.get("occurrence_ref") or "missing")
            for evidence_id in occurrence.get("source_evidence_ids") or []:
                if evidence_id not in evidence_ids:
                    issues.append(f"{case_ref}:{occurrence_ref}:unknown_occurrence_evidence:{evidence_id}")
        for evidence_id in case.get("evidence_anchor_ids") or []:
            if evidence_id not in evidence_ids:
                issues.append(f"{case_ref}:unknown_case_evidence:{evidence_id}")
        for action in case.get("actions") or []:
            if not isinstance(action, dict):
                issues.append(f"{case_ref}:action_not_object")
                continue
            action_ref = str(action.get("action_ref") or "missing")
            action_occurrence_ref = str(action.get("occurrence_ref") or "")
            if occurrence_refs and action_occurrence_ref not in occurrence_refs:
                issues.append(f"{case_ref}:{action_ref}:unknown_occurrence_ref:{action_occurrence_ref}")
            if not isinstance(action.get("outcome"), dict):
                issues.append(f"{case_ref}:{action_ref}:missing_outcome")
            for key in ("source_evidence_ids",):
                for evidence_id in action.get(key) or []:
                    if evidence_id not in evidence_ids:
                        issues.append(f"{case_ref}:{action_ref}:unknown_action_evidence:{evidence_id}")
            for evidence_id in (action.get("outcome") or {}).get("source_evidence_ids") or []:
                if evidence_id not in evidence_ids:
                    issues.append(f"{case_ref}:{action_ref}:unknown_outcome_evidence:{evidence_id}")
        for item in case.get("hypothesis_timeline") or []:
            if not isinstance(item, dict):
                issues.append(f"{case_ref}:hypothesis_not_object")
                continue
            if occurrence_refs:
                semantic_occurrence_ref = str(item.get("occurrence_ref") or "")
                if semantic_occurrence_ref not in occurrence_refs:
                    issues.append(f"{case_ref}:hypothesis_unknown_occurrence_ref:{semantic_occurrence_ref}")
            for evidence_id in item.get("source_evidence_ids") or []:
                if evidence_id not in evidence_ids:
                    issues.append(f"{case_ref}:unknown_semantic_evidence:{evidence_id}")
        for item in case.get("required_info") or []:
            if not isinstance(item, dict):
                issues.append(f"{case_ref}:required_info_not_object")
                continue
            for evidence_id in item.get("source_evidence_ids") or []:
                if evidence_id not in evidence_ids:
                    issues.append(f"{case_ref}:unknown_semantic_evidence:{evidence_id}")
    for item in truth.get("excluded_fragments") or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or item.get("message_id") or "")
        if evidence_id not in evidence_ids:
            issues.append(f"unknown_excluded_evidence:{evidence_id}")
    return sorted(set(issues))


def _json_shape(truth: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": truth.get("schema_version"),
        "case_id": truth.get("case_id"),
        "input_messages_sha256": truth.get("input_messages_sha256"),
        "input_evidence_sha256": truth.get("input_evidence_sha256"),
        "review_status": truth.get("review_status"),
        "graph_ingestion": truth.get("graph_ingestion"),
        "split_required": truth.get("split_required"),
        "case_count": truth.get("case_count"),
        "analysis_window": truth.get("analysis_window"),
        "daily_report_anchors": truth.get("daily_report_anchors") or [],
        "field_report_anchors": truth.get("field_report_anchors") or [],
        "artifact_audit": truth.get("artifact_audit") or [],
        "cases": [
            {
                "case_ref": case.get("case_ref"),
                "family": case.get("family"),
                "variant": case.get("variant"),
                "occurrences": case.get("occurrences") or [],
                "actions": [
                    {
                        "action_ref": action.get("action_ref"),
                        "occurrence_ref": action.get("occurrence_ref"),
                        "label": action.get("label"),
                        "execution_status": action.get("execution_status"),
                        "outcome": action.get("outcome"),
                    }
                    for action in case.get("actions") or []
                    if isinstance(action, dict)
                ],
            }
            for case in truth.get("cases") or []
            if isinstance(case, dict)
        ],
        "excluded_fragments": truth.get("excluded_fragments") or [],
        "critical_expectations": truth.get("critical_expectations") or [],
    }


def render(
    input_payload: dict[str, Any],
    truth: dict[str, Any],
    *,
    input_path: Path,
    truth_path: Path,
    baseline_prediction: dict[str, Any] | None = None,
    baseline_score: dict[str, Any] | None = None,
) -> str:
    issues = _validate(input_payload, truth)
    lines = [
        f"# {truth.get('case_id')} 盲测 Ground Truth 审核稿",
        "",
        "> 本文由冻结的 source-only 输入和独立 ground truth JSON 自动生成。DeepSeek 运行时只能读取输入文件，不能读取本页或 ground truth JSON。",
        "",
        "## 文件与冻结状态",
        "",
        f"- 原始输入 JSON：`{input_path}`",
        f"- Ground truth JSON：`{truth_path}`",
        f"- 输入消息 SHA-256：`{truth.get('input_messages_sha256')}`",
        *([f"- 全部输入证据 SHA-256：`{truth.get('input_evidence_sha256')}`"] if truth.get("input_evidence_sha256") else []),
        f"- 审核状态：`{truth.get('review_status')}`",
        f"- 是否允许入活动 KG：`{str(bool(truth.get('graph_ingestion'))).lower()}`",
        f"- 校验：`{'pass' if not issues else 'fail'}`",
        "",
        "## 原始候选消息",
        "",
        "| 时间 | message_id | 发送人 | root / parent | 原文 | 附件数 |",
        "|---|---|---|---|---|---:|",
    ]
    for message in input_payload.get("messages") or []:
        sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
        relation = " / ".join(filter(None, [str(message.get("root_id") or ""), str(message.get("parent_id") or "")])) or "—"
        lines.append(
            f"| {_cell(message.get('create_time'))} | `{_cell(message.get('message_id'))}` | "
            f"{_cell(sender.get('name') or sender.get('id'))} | {_cell(relation, 100)} | "
            f"{_cell(message.get('text'))} | {len(message.get('attachments') or [])} |"
        )
    linked_jira = [item for item in input_payload.get("linked_jira_issues") or [] if isinstance(item, dict)]
    if linked_jira:
        lines.extend([
            "",
            "## 关联 Jira 原始记录",
            "",
            "| evidence_id | summary | description | status / resolution | comments |",
            "|---|---|---|---|---|",
        ])
        for issue in linked_jira:
            comments = "<br>".join(
                f"{item.get('created')}: {item.get('body')}"
                for item in issue.get("comments") or []
                if isinstance(item, dict)
            ) or "—"
            lines.append(
                f"| `{issue.get('evidence_id')}` | {_cell(issue.get('summary'), 360)} | "
                f"{_cell(issue.get('description'), 700)} | {_cell(issue.get('status'))} / "
                f"{_cell(issue.get('resolution'))} | {_cell(comments, 700)} |"
            )
    external_artifacts = [item for item in input_payload.get("external_artifacts") or [] if isinstance(item, dict)]
    if external_artifacts:
        lines.extend([
            "",
            "## 外部附件与文档可用性",
            "",
            "| artifact_ref | 类型 | 获取状态 | 用于标注 | 来源消息 | URL / 备注 |",
            "|---|---|---|---:|---|---|",
        ])
        for artifact in external_artifacts:
            lines.append(
                f"| `{_cell(artifact.get('artifact_ref'))}` | `{_cell(artifact.get('kind'))}` | "
                f"`{_cell(artifact.get('retrieval_status'))}` | "
                f"`{str(bool(artifact.get('content_used_for_annotation'))).lower()}` | "
                f"{_cell(', '.join(artifact.get('source_message_ids') or []), 300)} | "
                f"{_cell(artifact.get('url') or artifact.get('content_summary') or artifact.get('scope_limit') or artifact.get('note') or artifact.get('path'), 700)} |"
            )
    report_anchors = truth.get("daily_report_anchors") or truth.get("field_report_anchors") or []
    lines.extend([
        "",
        "## 现场报告状态钩子",
        "",
        f"- 分析窗口：`{(truth.get('analysis_window') or {}).get('start_inclusive')}` 至 `{(truth.get('analysis_window') or {}).get('end_inclusive')}`",
        f"- 方法：`{truth.get('analysis_method')}`",
        "",
        "| 日期 | 报告人 | 角色状态 | message_id | 状态检查点 | 关联 Trace |",
        "|---|---|---|---|---|---|",
    ])
    for anchor in report_anchors:
        lines.append(
            f"| {_cell(anchor.get('date'))} | {_cell(anchor.get('fae') or anchor.get('reporter'))} | "
            f"`{_cell(anchor.get('role_status') or 'confirmed_fae')}` | `{_cell(anchor.get('message_id'))}` | "
            f"{_cell(anchor.get('state_checkpoint'), 500)} | {_cell(', '.join(anchor.get('trace_refs') or []), 120)} |"
        )
    artifact_audit = [item for item in truth.get("artifact_audit") or [] if isinstance(item, dict)]
    if artifact_audit:
        lines.extend([
            "",
            "## 附件倒查结论",
            "",
            "| artifact_ref | 结论 | 证据 |",
            "|---|---|---|",
        ])
        for audit in artifact_audit:
            lines.append(
                f"| `{_cell(audit.get('artifact_ref'))}` | {_cell(audit.get('finding'), 700)} | "
                f"{_cell(', '.join(audit.get('source_evidence_ids') or []), 400)} |"
            )
    lines.extend([
        "",
        "## Trace 边界判断",
        "",
        f"- `split_required`: `{str(bool(truth.get('split_required'))).lower()}`",
        f"- `case_count`: `{truth.get('case_count')}`",
        f"- 理由：{truth.get('split_reason')}",
    ])
    if baseline_prediction is not None and baseline_score is not None:
        errors = [
            str(item.get("code") or "")
            for item in baseline_score.get("critical_errors") or []
            if isinstance(item, dict)
        ]
        lines.extend([
            "",
            "## 当前 W1 首轮盲测基线对照",
            "",
            "> 此处只用于展示首轮 W1 与候选标注的差异，不会反向修改候选答案。",
            "",
            f"- 候选 Trace 数：`{baseline_score.get('expected_case_count')}`",
            f"- W1 active episode 数：`{baseline_score.get('predicted_active_episode_count')}`",
            f"- 候选要求拆分：`{str(bool(baseline_score.get('expected_split_required'))).lower()}`",
            f"- W1 判断拆分：`{str(bool(baseline_score.get('predicted_split_required'))).lower()}`",
            f"- Critical errors：{', '.join(f'`{item}`' for item in errors) if errors else '无'}",
            f"- W1 prediction SHA-256：`{baseline_prediction.get('prediction_sha256')}`",
            "",
            "### W1 预测 episodes",
            "",
            "| completeness | fault focus | evidence message IDs |",
            "|---|---|---|",
        ])
        for episode in baseline_prediction.get("episodes") or []:
            if not isinstance(episode, dict):
                continue
            lines.append(
                f"| `{episode.get('completeness')}` | {_cell(episode.get('fault_focus_text') or '（无）', 360)} | "
                f"{_cell(', '.join(episode.get('evidence_message_ids') or []), 320)} |"
            )
    for case in truth.get("cases") or []:
        family = case.get("family") if isinstance(case.get("family"), dict) else {}
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        lines.extend([
            "",
            f"## Trace `{case.get('case_ref')}`",
            "",
            f"- family：`{family.get('label')}`",
            f"- variant：`{variant.get('label')}`",
            f"- subsystem：`{family.get('subsystem')}`",
            f"- symptom：{case.get('symptom_summary')}",
            f"- evidence：{', '.join(f'`{x}`' for x in case.get('evidence_anchor_ids') or [])}",
            "",
            "### Occurrences / 同一 Trace 内事件",
            "",
            "| occurrence | 时间范围 | 设备范围 | 状态 | 摘要 | 证据 |",
            "|---|---|---|---|---|---|",
        ])
        for occurrence in case.get("occurrences") or []:
            lines.append(
                f"| `{occurrence.get('occurrence_ref')}` | {_cell(occurrence.get('time_range'), 160)} | "
                f"{_cell(occurrence.get('device_scope'), 140)} | `{occurrence.get('state')}` | "
                f"{_cell(occurrence.get('summary'), 360)} | "
                f"{_cell(', '.join(occurrence.get('source_evidence_ids') or []), 260)} |"
            )
        lines.extend([
            "",
            "### Actions 与 Outcomes",
            "",
            "| 顺序 | occurrence | Action | role | execution | Outcome | 结果说明 | Action证据 | Outcome证据 |",
            "|---:|---|---|---|---|---|---|---|---|",
        ])
        for index, action in enumerate(case.get("actions") or [], start=1):
            outcome = action.get("outcome") if isinstance(action.get("outcome"), dict) else {}
            lines.append(
                f"| {index} | `{action.get('occurrence_ref')}` | {_cell(action.get('label'), 180)} | `{action.get('action_role')}` | "
                f"`{action.get('execution_status')}` | `{outcome.get('outcome_type')}` | "
                f"{_cell(outcome.get('summary'), 300)} | "
                f"{_cell(', '.join(action.get('source_evidence_ids') or []), 220)} | "
                f"{_cell(', '.join(outcome.get('source_evidence_ids') or []), 220)} |"
            )
        lines.extend(["", "### 诊断状态演化", ""])
        timeline = case.get("hypothesis_timeline") or []
        if timeline:
            lines.extend(["| 顺序 | occurrence | state | causal_role | 判断 | 证据 |", "|---:|---|---|---|---|---|"])
            for item in timeline:
                lines.append(
                    f"| {item.get('order')} | `{item.get('occurrence_ref')}` | `{item.get('state')}` | `{item.get('causal_role')}` | "
                    f"{_cell(item.get('summary'), 350)} | {_cell(', '.join(item.get('source_evidence_ids') or []), 220)} |"
                )
        else:
            lines.append("- 无证据支持具体根因或假设演化。")
        lines.extend(["", "### Required info / 不确定性", ""])
        for item in case.get("required_info") or []:
            lines.append(f"- `{item.get('slot')}`：{item.get('question')}")
        for item in case.get("uncertainties") or []:
            lines.append(f"- 不确定性：{item}")
    lines.extend(["", "## 应排除的内容", ""])
    for item in truth.get("excluded_fragments") or []:
        evidence_id = item.get("evidence_id") or item.get("message_id")
        lines.append(f"- `{evidence_id}`：{item.get('fragment')}（`{item.get('reason')}`）")
    lines.extend(["", "## Critical 验收要求", ""])
    for item in truth.get("critical_expectations") or []:
        lines.append(f"- {item}")
    if issues:
        lines.extend(["", "## 校验错误", ""] + [f"- `{issue}`" for issue in issues])
    lines.extend([
        "",
        "## Ground truth JSON 关键结构",
        "",
        "```json",
        json.dumps(_json_shape(truth), ensure_ascii=False, indent=2),
        "```",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-blind-ground-truth-review")
    parser.add_argument("--root", default="data/annotations/goldcases/review-v3")
    parser.add_argument("--out", default="data/annotations/goldcases/review-v3/reviews")
    parser.add_argument("--baseline", default="data/results/gold-011-015-review-v3-w1-baseline.json")
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}
    prediction_by_case = {
        str(item.get("case_id") or ""): item
        for item in baseline.get("predictions") or []
        if isinstance(item, dict)
    }
    score_by_case = {
        str(item.get("case_id") or ""): item
        for item in baseline.get("scores") or []
        if isinstance(item, dict)
    }
    rows = []
    all_issues: list[str] = []
    for truth_path in sorted((root / "ground_truth").glob("goldcase-*.json")):
        input_path = root / "inputs" / truth_path.name
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        issues = _validate(input_payload, truth)
        all_issues.extend(f"{truth_path.stem}:{issue}" for issue in issues)
        md_path = out / f"{truth_path.stem}.md"
        body = render(
            input_payload,
            truth,
            input_path=input_path,
            truth_path=truth_path,
            baseline_prediction=prediction_by_case.get(truth_path.stem),
            baseline_score=score_by_case.get(truth_path.stem),
        )
        md_path.write_text(body, encoding="utf-8")
        rows.append({
            "case_id": truth_path.stem,
            "document": str(md_path),
            "ground_truth": str(truth_path),
            "case_count": truth.get("case_count"),
            "split_required": truth.get("split_required"),
            "validation": "pass" if not issues else "fail",
            "w1_critical_errors": [
                str(item.get("code") or "")
                for item in (score_by_case.get(truth_path.stem) or {}).get("critical_errors") or []
                if isinstance(item, dict)
            ],
            "document_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        })
    index_lines = [
        "# goldcase-011–015 Ground Truth 审核索引",
        "",
        "| 样本 | Trace数 | 需拆分 | 校验 | W1 critical errors | 文档 | Ground truth JSON |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        index_lines.append(
            f"| {row['case_id']} | {row['case_count']} | {str(bool(row['split_required'])).lower()} | "
            f"{row['validation']} | {', '.join(row['w1_critical_errors']) or '—'} | "
            f"[{row['case_id']}.md]({Path(row['document']).name}) | `{row['ground_truth']}` |"
        )
    (out / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    report = {"documents": rows, "validation_issues": all_issues}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
