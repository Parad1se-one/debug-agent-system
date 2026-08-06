from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.contracts import APPROVED_FAMILY_LABELS, PSEUDO_FAMILY_LABELS

POSITIVE_NO_ISSUE_MARKERS = (
    "没有问题",
    "未出现",
    "没发生过",
    "恢复正常",
    "正常测试",
    "持续观察",
    "未再出现",
)
REPORT_NOISE_MARKERS = (
    "现场工作",
    "培训客户",
    "工作汇报",
    "每日数据",
    "项目进度",
    "回访咨询",
)
FAULT_MARKERS = (
    "蓝屏",
    "重启",
    "异常",
    "失败",
    "闪退",
    "卡死",
    "卡顿",
    "无法",
    "误报",
    "漏检",
    "不拍照",
    "拍摄失败",
    "初始化失败",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _first_split_case(row: dict[str, Any]) -> dict[str, Any]:
    draft = row.get("candidate_draft_v2") if isinstance(row.get("candidate_draft_v2"), dict) else {}
    cases = draft.get("split_cases") if isinstance(draft.get("split_cases"), list) else []
    return cases[0] if cases and isinstance(cases[0], dict) else {}


def _actions(case: dict[str, Any]) -> list[str]:
    return [str(item.get("label") or "") for item in case.get("actions") or [] if isinstance(item, dict)]


def _dedupe_count(values: list[str]) -> int:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return len(seen)


def _is_positive_no_issue(text: str) -> bool:
    if not any(marker in text for marker in POSITIVE_NO_ISSUE_MARKERS):
        return False
    if any(marker in text for marker in FAULT_MARKERS):
        positive_prefixes = (
            "客户反馈说今天没有",
            "今天没有",
            "未再出现",
            "恢复正常",
            "正常测试后未再出现",
            "持续观察未再出现",
        )
        return text.startswith(positive_prefixes)
    return True


def build_report(rows: list[dict[str, Any]], *, sample_limit: int = 25) -> dict[str, Any]:
    counters = Counter()
    buckets: dict[str, list[dict[str, Any]]] = {
        "noncanonical_family": [],
        "pseudo_family": [],
        "long_variant": [],
        "questionish_variant": [],
        "positive_no_issue": [],
        "report_noise": [],
        "split_required": [],
        "action_duplicates": [],
        "empty_case": [],
    }

    for row in rows:
        card = row.get("case_understanding_card") if isinstance(row.get("case_understanding_card"), dict) else {}
        case = _first_split_case(row)
        family = case.get("family") if isinstance(case.get("family"), dict) else {}
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        family_label = str(family.get("label") or "")
        variant_label = str(variant.get("label") or "")
        label = str(row.get("label") or "")
        split_required = bool(card.get("split_required"))
        action_labels = _actions(case)
        action_count = len(action_labels)
        action_unique = _dedupe_count(action_labels)
        text = " ".join([
            label,
            str(row.get("symptom_raw") or ""),
            str(row.get("conclusion") or ""),
            " ".join(action_labels),
        ])
        payload = {
            "candidate_id": str(row.get("candidate_id") or ""),
            "label": label,
            "family": family_label,
            "variant": variant_label,
            "split_required": split_required,
            "action_count": action_count,
            "action_unique": action_unique,
        }

        if not card.get("schema_valid"):
            counters["empty_case"] += 1
            if len(buckets["empty_case"]) < sample_limit:
                buckets["empty_case"].append(payload)
        if family_label and family_label not in APPROVED_FAMILY_LABELS:
            counters["noncanonical_family"] += 1
            if len(buckets["noncanonical_family"]) < sample_limit:
                buckets["noncanonical_family"].append(payload)
        if family_label in PSEUDO_FAMILY_LABELS:
            counters["pseudo_family"] += 1
            if len(buckets["pseudo_family"]) < sample_limit:
                buckets["pseudo_family"].append(payload)
        if len(variant_label) > 40:
            counters["long_variant"] += 1
            if len(buckets["long_variant"]) < sample_limit:
                buckets["long_variant"].append(payload)
        if variant_label.startswith(("我这个现场", "现场反馈", "客户反馈")) or variant_label.endswith(("是什么问题", "怎么处理", "怎么办", "如何处理", "吗", "么")):
            counters["questionish_variant"] += 1
            if len(buckets["questionish_variant"]) < sample_limit:
                buckets["questionish_variant"].append(payload)
        if _is_positive_no_issue(text):
            counters["positive_no_issue"] += 1
            if len(buckets["positive_no_issue"]) < sample_limit:
                buckets["positive_no_issue"].append(payload)
        if any(marker in text for marker in REPORT_NOISE_MARKERS):
            counters["report_noise"] += 1
            if len(buckets["report_noise"]) < sample_limit:
                buckets["report_noise"].append(payload)
        if split_required:
            counters["split_required"] += 1
            if len(buckets["split_required"]) < sample_limit:
                buckets["split_required"].append(payload)
        if action_count > action_unique:
            counters["action_duplicates"] += 1
            if len(buckets["action_duplicates"]) < sample_limit:
                buckets["action_duplicates"].append(payload)

    return {
        "episodes": len(rows),
        "counters": dict(counters),
        "samples": buckets,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--out", default="")
    parser.add_argument("--sample-limit", type=int, default=25)
    args = parser.parse_args(argv)
    report = build_report(_load_rows(Path(args.input_jsonl)), sample_limit=args.sample_limit)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
