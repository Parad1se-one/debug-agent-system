from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASE_IDS = ("011", "012", "013", "014", "015")

CASE_STATUS: dict[str, dict[str, str]] = {
    "011": {
        "gold_batch": "gold-011-015-review-v3",
        "review_state": "已完成本轮逐条复审；当前仍为 review_candidate，尚未写入生产 KG。",
        "comparison_note": "当前 Gold 已扩展为 12 月 7 日至 12 日、由 FAE 日报串联的三条并行故障链；冻结基线的旧评分基于 blind-v1，不再代表当前 Gold。",
    },
    "012": {
        "gold_batch": "gold-011-015-review-v3",
        "review_state": "已完成本轮跨时间窗、附件与 Jira 描述复审；当前仍为 review_candidate，尚未写入生产 KG。",
        "comparison_note": "当前 Gold 使用跨时间窗证据重建四条 Trace；冻结基线的旧评分基于 blind-v1，只作为历史记录展示。",
    },
    "013": {
        "gold_batch": "gold-011-015-review-v3",
        "review_state": "已完成本轮跨群聊、设备身份与附件复审；当前仍为 review_candidate，尚未写入生产 KG。",
        "comparison_note": "当前 Gold 区分两台物理设备并保留三条Trace：新设备无显示、新设备Mark对位失败、旧DEMO设备光源配置丢失导致拍摄失败。冻结基线评分基于单条日报和blind-v1，只作为历史记录展示。",
    },
    "014": {
        "gold_batch": "gold-011-015-review-v3",
        "review_state": "已完成本轮长期群聊、Jira与附件可用性复审；当前仍为 review_candidate，尚未写入生产 KG。",
        "comparison_note": "当前 Gold 将软件花屏扩展为2025-12-04至2026-07-01的长期复发链，并单独保留2026-05-08的HDMI物理黑屏闪烁。冻结基线评分只覆盖2026年5月局部输入，仅作为历史记录。",
    },
    "015": {
        "gold_batch": "gold-011-015-review-v3",
        "review_state": "已完成本轮纵向群聊、Jira、时区与附件可用性复审；当前仍为 review_candidate，尚未写入生产 KG。",
        "comparison_note": "当前 Gold 补回015开头的诊断包与两张照片引用、北京时间21:12日报和Jira链接，并将Buddy与蓝屏重建为多occurrence Trace；缺失附件内容不作臆测。冻结基线评分仅覆盖blind-v1局部输入，只作为历史记录。",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _candidate_summary(prediction: dict[str, Any]) -> list[str]:
    lines = [
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| 输入消息数 | {prediction.get('message_count', 0)} |",
        f"| 候选 episode 总数 | {prediction.get('episode_count', 0)} |",
        f"| 活跃 episode 数 | {prediction.get('active_episode_count', 0)} |",
        f"| 噪声 episode 数 | {prediction.get('noise_episode_count', 0)} |",
        f"| 输入消息哈希 | `{prediction.get('input_messages_sha256', '')}` |",
        f"| 候选结果哈希 | `{prediction.get('prediction_sha256', '')}` |",
    ]
    return lines


def _gold_summary(gold: dict[str, Any], batch: str) -> list[str]:
    return [
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| Gold 批次 | `{batch}` |",
        f"| review_status | `{gold.get('review_status', '')}` |",
        f"| graph_ingestion | `{str(gold.get('graph_ingestion', False)).lower()}` |",
        f"| split_required | `{str(gold.get('split_required', False)).lower()}` |",
        f"| 人工 Case 数 | {gold.get('case_count', len(gold.get('cases') or []))} |",
        f"| 输入消息哈希 | `{gold.get('input_messages_sha256', '')}` |",
        f"| 输入证据哈希 | `{gold.get('input_evidence_sha256', '未记录')}` |",
    ]


def _case_rows(gold: dict[str, Any]) -> list[str]:
    lines = [
        "| case_ref | FaultFamily | FaultVariant | 症状摘要 |",
        "| --- | --- | --- | --- |",
    ]
    for case in gold.get("cases") or []:
        family = (case.get("family") or {}).get("label", "")
        variant = (case.get("variant") or {}).get("label", "")
        symptom = str(case.get("symptom_summary") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {case.get('case_ref', '')} | {family} | {variant} | {symptom} |"
        )
    return lines


def _score_summary(score: dict[str, Any], *, stale: bool) -> list[str]:
    errors = score.get("critical_errors") or []
    lines = [
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| 评分口径 | `gold-011-015-blind-v1`{'（已过期，仅供追溯）' if stale else ''} |",
        f"| 输入哈希匹配 | `{str(score.get('input_hash_match', False)).lower()}` |",
        f"| 当时预期 Case 数 | {score.get('expected_case_count', '')} |",
        f"| 候选活跃 episode 数 | {score.get('predicted_active_episode_count', '')} |",
        f"| 关键错误数 | {len(errors)} |",
    ]
    return lines


def render_case(
    case_id: str,
    prediction: dict[str, Any],
    score: dict[str, Any],
    gold: dict[str, Any],
) -> str:
    status = CASE_STATUS[case_id]
    batch = status["gold_batch"]
    stale_score = batch != "gold-011-015-blind-v1"
    hashes_match = prediction.get("input_messages_sha256") == gold.get("input_messages_sha256")
    hash_note = (
        "候选和 Gold 使用同一输入消息哈希，可以直接比较切分结果。"
        if hashes_match
        else "候选与当前 Gold 的输入消息哈希不同：当前 Gold 扩展了消息、Jira 或附件证据。本页是历史候选对当前人工结论的审计视图，不能把旧评分当作当前可比指标。"
    )

    lines = [
        f"# goldcase-{case_id}：候选结果 vs Gold JSON",
        "",
        "> 本页用于人工审核，不参与 KG 写入。候选是冻结的 **W1 source-only 基线**，不是 DeepSeek 输出；仓库中的 DeepSeek response template 目前仍为空。",
        "",
        "## 版本与审核状态",
        "",
        f"- 当前 Gold 来源：`{batch}`",
        f"- 状态：{status['review_state']}",
        f"- 比较说明：{status['comparison_note']}",
        f"- 输入口径：{hash_note}",
        "",
        "## 一眼看懂",
        "",
        f"W1 候选切出了 **{prediction.get('active_episode_count', 0)}** 个活跃 episode；当前展示的人工 Gold 有 **{gold.get('case_count', 0)}** 个业务 Case。",
        "",
        *_candidate_summary(prediction),
        "",
        "### W1 候选完整 JSON",
        "",
        _json_block(prediction),
        "",
        "## 当前人工 Gold",
        "",
        *_gold_summary(gold, batch),
        "",
        "### Gold Case 摘要",
        "",
        *_case_rows(gold),
        "",
        "### Gold 完整 JSON",
        "",
        _json_block(gold),
        "",
        "## 冻结基线评分记录",
        "",
        *_score_summary(score, stale=stale_score),
        "",
        "### 评分记录完整 JSON",
        "",
        _json_block(score),
        "",
        "## 审核时应关注",
        "",
        "1. 业务 Trace 数量和边界是否正确，是否把同一问题拆碎或把并行问题合并。",
        "2. `actual`、`recommended`、`planned` 是否与原文一致，不能把建议写成已执行。",
        "3. 当次恢复、短期未复发与 `verified_fix` 必须区分，不能直接升级为最终解决。",
        "4. FaultFamily、FaultVariant、DiagnosticAction、ActionOutcome 和证据锚点是否能逐项回到原始消息、Jira 或附件。",
        "",
    ]
    return "\n".join(lines)


def render_all(root: Path, output_dir: Path) -> list[Path]:
    baseline_path = root / "data/results/gold-011-015-blind-w1-baseline.json"
    baseline = _load_json(baseline_path)
    predictions = {item["case_id"]: item for item in baseline["predictions"]}
    scores = {item["case_id"]: item for item in baseline["scores"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case_id in CASE_IDS:
        full_id = f"goldcase-{case_id}"
        batch = CASE_STATUS[case_id]["gold_batch"]
        gold_path = root / f"data/kg_v2/blind_cases/{batch}/ground_truth/{full_id}.json"
        if full_id not in predictions or full_id not in scores:
            raise KeyError(f"missing frozen candidate or score for {full_id}")
        if not gold_path.exists():
            raise FileNotFoundError(gold_path)
        document = render_case(
            case_id,
            predictions[full_id],
            scores[full_id],
            _load_json(gold_path),
        )
        output_path = output_dir / f"{full_id}.md"
        output_path.write_text(document, encoding="utf-8")
        written.append(output_path)

    readme = [
        "# goldcase-011–015 候选结果 vs Gold JSON",
        "",
        "这五份文档把冻结的 W1 source-only 候选、当前可用的人工 Gold JSON 和冻结基线评分放在同一页，便于逐案审核。它们不是 DeepSeek 输出，也不会写入生产 KG。",
        "",
        "| Case | 当前展示的 Gold | 状态 | 文档 |",
        "| --- | --- | --- | --- |",
    ]
    for case_id in CASE_IDS:
        status = CASE_STATUS[case_id]
        readme.append(
            f"| goldcase-{case_id} | `{status['gold_batch']}` | {status['review_state']} | [查看](goldcase-{case_id}.md) |"
        )
    readme.extend(
        [
            "",
            "注意：011–015 均已完成本轮 review-v3 复审；原 blind-v1 评分只保留作历史审计，当前结果仍是 review_candidate，尚未冻结入生产 KG。",
            "",
        ]
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme), encoding="utf-8")
    written.insert(0, readme_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/xing_lark_blind_011_015_candidate_vs_gold"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    written = render_all(root, output_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
