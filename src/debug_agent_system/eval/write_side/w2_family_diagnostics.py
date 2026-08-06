from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from debug_agent_system.knowledge_v2.contracts import APPROVED_FAMILY_LABELS, PSEUDO_FAMILY_LABELS
QUESTIONISH_SUFFIXES = ("是什么问题", "怎么处理", "怎么办", "如何处理", "吗", "么")


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


def build_report(rows: list[dict[str, Any]], *, sample_limit: int = 20) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    subsystem_counts: Counter[str] = Counter()
    noncanonical: list[dict[str, Any]] = []
    pseudo: list[dict[str, Any]] = []
    long_variant: list[dict[str, Any]] = []
    questionish_variant: list[dict[str, Any]] = []
    split_cases: list[dict[str, Any]] = []

    for row in rows:
        case = _first_split_case(row)
        family = case.get("family") if isinstance(case.get("family"), dict) else {}
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        family_label = str(family.get("label") or "")
        subsystem = str(family.get("subsystem") or "")
        variant_label = str(variant.get("label") or "")
        candidate_id = str(row.get("candidate_id") or "")

        if family_label:
            family_counts[family_label] += 1
        if subsystem:
            subsystem_counts[subsystem] += 1

        payload = {
            "candidate_id": candidate_id,
            "label": row.get("label") or "",
            "family": family_label,
            "subsystem": subsystem,
            "variant": variant_label,
            "split_required": bool((row.get("case_understanding_card") or {}).get("split_required")),
        }
        if family_label and family_label not in APPROVED_FAMILY_LABELS:
            noncanonical.append(payload)
        if family_label in PSEUDO_FAMILY_LABELS:
            pseudo.append(payload)
        if len(variant_label) > 40:
            long_variant.append(payload)
        if variant_label.endswith(QUESTIONISH_SUFFIXES) or variant_label.startswith(("我这个现场", "现场反馈", "客户反馈")):
            questionish_variant.append(payload)
        if payload["split_required"]:
            split_cases.append(payload)

    return {
        "episodes": len(rows),
        "top_families": family_counts.most_common(30),
        "top_subsystems": subsystem_counts.most_common(30),
        "noncanonical_family_count": len(noncanonical),
        "pseudo_family_count": len(pseudo),
        "long_variant_count": len(long_variant),
        "questionish_variant_count": len(questionish_variant),
        "split_required_count": len(split_cases),
        "noncanonical_family_samples": noncanonical[:sample_limit],
        "pseudo_family_samples": pseudo[:sample_limit],
        "long_variant_samples": long_variant[:sample_limit],
        "questionish_variant_samples": questionish_variant[:sample_limit],
        "split_required_samples": split_cases[:sample_limit],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--out", default="")
    parser.add_argument("--sample-limit", type=int, default=20)
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
