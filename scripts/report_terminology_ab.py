#!/usr/bin/env python3
"""Aggregate deterministic metrics for the KG_v2 terminology A/B suite.

The script intentionally does not manufacture route or answer-quality scores.
Those two metrics require a blind evaluator.  It emits a blinded review packet
for that follow-up while computing publication, resolver, search-contract,
evidence-recall, and safety metrics directly from the frozen run artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.terminology import TerminologyResolver
from debug_agent_system.kg_raw_codex.terminology_contract import (
    build_terminology_search_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "data/eval/terminology_ab_v1/cases.jsonl"
DEFAULT_KG_ROOT = REPO_ROOT / "data/kg_v2"
ARMS = ("A_control", "B_treatment")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalise(value: str) -> str:
    return "".join(re.findall(r"[0-9a-z\u3400-\u9fff]+", value.lower()))


def _canonical_names(resolution: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for mention in resolution.get("resolved_mentions") or []:
        concept = mention.get("concept") or {}
        name = str(concept.get("canonical_name") or "").strip()
        if name:
            names.add(name)
    return names


def _classify_error(error: str) -> list[str]:
    labels: list[str] = []
    if "terminology_required_search_missing:" in error:
        labels.append("required_search_not_executed")
    if "external_artifact_" in error:
        labels.append("external_artifact_verification")
    if "unexpected_procedure_variant:" in error:
        labels.append("unexpected_procedure_variant")
    if "missing_terminology" in error or "terminology_" in error and not labels:
        labels.append("other_terminology_contract")
    if not labels:
        labels.append("other_runtime_or_verification")
    return labels


def _artifact_path(raw: str, run_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    direct = REPO_ROOT / path
    if direct.exists():
        return direct
    return run_dir / path


def _source_used(
    artifact: dict[str, Any], gold_sources: list[str]
) -> tuple[bool | None, list[str]]:
    if not gold_sources:
        return None, []
    files = [str(item) for item in artifact.get("files_read") or []]
    answer = str(artifact.get("answer") or "")
    answer_norm = _normalise(answer)
    matched: list[str] = []
    for gold in gold_sources:
        gold_norm = _normalise(gold)
        matching_files = [path for path in files if gold_norm in _normalise(path)]
        if not matching_files:
            continue
        if any(
            _normalise(Path(path).name) in answer_norm
            or _normalise(path) in answer_norm
            for path in matching_files
        ):
            matched.append(gold)
    return bool(matched), matched


def _runtime_search_compliance(
    artifact: dict[str, Any] | None,
    required_pairs: list[list[str]],
) -> bool | None:
    if not required_pairs:
        return None
    if artifact is None:
        return False
    audit = artifact.get("terminology_search_audit") or {}
    if not audit.get("complete"):
        return False
    searched: set[str] = set()
    for group in audit.get("groups") or []:
        for term in group.get("terms") or []:
            if term.get("searched"):
                searched.add(str(term.get("term") or ""))
    return all(left in searched and right in searched for left, right in required_pairs)


def _load_results(run_dirs: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_json(manifest_path)
        manifests.append(manifest)
        for row in manifest.get("results") or []:
            copied = dict(row)
            copied["_run_dir"] = str(run_dir)
            indexed[(str(row["case_id"]), str(row["arm"]))] = copied
    return indexed, manifests


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _delta(b: float | None, a: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _mean(values)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_blind_packet(
    cases: list[dict[str, Any]],
    per_case: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260804)
    packet: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        arms = per_case.get(case_id) or {}
        if not all(arms.get(arm, {}).get("published") for arm in ARMS):
            continue
        sides = list(ARMS)
        rng.shuffle(sides)
        blinded = {"X": sides[0], "Y": sides[1]}
        packet.append({
            "case_id": case_id,
            "category": case["category"],
            "query": case["query"],
            "gold_source_documents": case.get("gold_source_documents") or [],
            "X_answer": arms[sides[0]]["answer"],
            "Y_answer": arms[sides[1]]["answer"],
            "rubric": {
                "route_accuracy": "0/1：对象、故障域和资料域是否正确",
                "factual_correctness": "0/1",
                "request_coverage": "0/1",
                "actionable_structure": "0/1",
                "evidence_constraint": "0/1",
                "wrong_variant_lock": "0/1：是否在证据不足时确定性锁定根因",
            },
        })
        mapping.append({"case_id": case_id, **blinded})
    packet_path = output_dir / "blind_review_packet.jsonl"
    packet_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
        encoding="utf-8",
    )
    _write_json(output_dir / "blind_review_mapping.json", mapping)
    return {"pair_count": len(packet)}


def aggregate(
    *, cases_path: Path, run_dirs: list[Path], output_dir: Path
) -> dict[str, Any]:
    cases = _read_cases(cases_path)
    indexed, manifests = _load_results(run_dirs)
    resolver = TerminologyResolver.from_root(DEFAULT_KG_ROOT)
    per_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for case in cases:
        case_id = str(case["id"])
        expected = case.get("expected") or {}
        offline_resolution = resolver.resolve(str(case["query"]), limit=30, context={})
        offline_names = _canonical_names(offline_resolution)
        offline_contract = build_terminology_search_contract(
            str(case["query"]), offline_resolution
        )
        actual_pairs = {
            tuple(str(term) for term in group.get("required_terms") or [])
            for group in offline_contract.get("required_search_groups") or []
            if len(group.get("required_terms") or []) == 2
        }
        expected_pairs = {
            tuple(str(term) for term in pair)
            for pair in expected.get("required_search_pairs") or []
        }
        must_resolve = set(expected.get("must_resolve") or [])
        forbidden = set(expected.get("must_not_resolve") or [])
        for arm in ARMS:
            result = indexed.get((case_id, arm))
            artifact: dict[str, Any] | None = None
            error = ""
            status = "missing"
            if result:
                status = str(result.get("status") or "missing")
                run_dir = Path(str(result["_run_dir"]))
                artifact_candidate = _artifact_path(
                    str(result.get("artifact") or ""), run_dir
                )
                if artifact_candidate.is_file():
                    artifact = _read_json(artifact_candidate)
                error = str(result.get("error") or "")
            published = bool(
                artifact
                and (artifact.get("verification") or {}).get("passed")
                and status in {"completed", "reused"}
            )
            evidence, matched_sources = (
                _source_used(artifact, case.get("gold_source_documents") or [])
                if published and artifact
                else (
                    (False, [])
                    if case.get("gold_source_documents")
                    else (None, [])
                )
            )
            runtime_names = (
                _canonical_names(artifact.get("terminology_resolution") or {})
                if artifact and arm == "B_treatment"
                else set()
            )
            row = {
                "status": status,
                "published": published,
                "error": error,
                "failure_labels": _classify_error(error) if error else [],
                "evidence_recall": evidence,
                "matched_gold_sources": matched_sources,
                "runtime_resolved_names": sorted(runtime_names),
                "offline_resolved_names": sorted(offline_names),
                "offline_required_search_pairs": sorted(actual_pairs),
                "unexpected_required_search_pairs": (
                    sorted(actual_pairs - expected_pairs)
                    if arm == "B_treatment"
                    else []
                ),
                "missing_required_search_pairs": (
                    sorted(expected_pairs - actual_pairs)
                    if arm == "B_treatment"
                    else []
                ),
                "must_resolve_complete_offline": (
                    must_resolve.issubset(offline_names) if arm == "B_treatment" else None
                ),
                "unsafe_expansions_offline": (
                    sorted(forbidden & offline_names) if arm == "B_treatment" else []
                ),
                "required_search_compliance": (
                    _runtime_search_compliance(
                        artifact if published else None,
                        expected.get("required_search_pairs") or [],
                    )
                    if arm == "B_treatment"
                    else None
                ),
                "answer": str(artifact.get("answer") or "") if artifact else "",
            }
            per_case[case_id][arm] = row

    categories = [
        "canonical_name",
        "field_alias_abbreviation_typo",
        "english_or_mixed",
        "safety_ambiguity_negative",
    ]
    summaries: dict[str, Any] = {}
    for category in ["all", *categories]:
        selected = [
            case
            for case in cases
            if category == "all" or case["category"] == category
        ]
        arm_summary: dict[str, Any] = {}
        for arm in ARMS:
            rows = [per_case[str(case["id"])][arm] for case in selected]
            error_labels = Counter(
                label for row in rows for label in row["failure_labels"]
            )
            arm_summary[arm] = {
                "case_count": len(rows),
                "published_count": sum(bool(row["published"]) for row in rows),
                "publish_rate": _mean([float(row["published"]) for row in rows]),
                "evidence_recall": _rate(rows, "evidence_recall"),
                "failure_labels": dict(sorted(error_labels.items())),
            }
        summaries[category] = {
            **arm_summary,
            "delta_B_minus_A": {
                "publish_rate": _delta(
                    arm_summary["B_treatment"]["publish_rate"],
                    arm_summary["A_control"]["publish_rate"],
                ),
                "evidence_recall": _delta(
                    arm_summary["B_treatment"]["evidence_recall"],
                    arm_summary["A_control"]["evidence_recall"],
                ),
            },
        }

    treatment_rows = [per_case[str(case["id"])]["B_treatment"] for case in cases]
    required_rows = [
        per_case[str(case["id"])]["B_treatment"]
        for case in cases
        if (case.get("expected") or {}).get("required_search_pairs")
    ]
    safety_rows = [
        per_case[str(case["id"])]["B_treatment"]
        for case in cases
        if case["category"] == "safety_ambiguity_negative"
    ]
    resolver_summary = {
        "must_resolve_case_count": sum(
            bool((case.get("expected") or {}).get("must_resolve")) for case in cases
        ),
        "must_resolve_complete_count": sum(
            bool((case.get("expected") or {}).get("must_resolve"))
            and row["must_resolve_complete_offline"] is True
            for case, row in zip(cases, treatment_rows)
        ),
        "must_resolve_incomplete_case_ids": [
            str(case["id"])
            for case, row in zip(cases, treatment_rows)
            if (case.get("expected") or {}).get("must_resolve")
            and row["must_resolve_complete_offline"] is not True
        ],
        "required_search_case_count": len(required_rows),
        "required_search_complete_count": sum(
            row["required_search_compliance"] is True for row in required_rows
        ),
        "required_search_compliance": _mean([
            float(row["required_search_compliance"] is True) for row in required_rows
        ]),
        "required_search_failed_case_ids": [
            str(case["id"])
            for case in cases
            if (case.get("expected") or {}).get("required_search_pairs")
            and per_case[str(case["id"])]["B_treatment"]["required_search_compliance"]
            is not True
        ],
        "unsafe_expansion_count": sum(
            len(row["unsafe_expansions_offline"]) for row in safety_rows
        ),
        "unsafe_expansion_cases": {
            str(case["id"]): per_case[str(case["id"])]["B_treatment"][
                "unsafe_expansions_offline"
            ]
            for case in cases
            if case["category"] == "safety_ambiguity_negative"
            and per_case[str(case["id"])]["B_treatment"][
                "unsafe_expansions_offline"
            ]
        },
        "search_contract_expectation_mismatch_count": sum(
            bool(row["unexpected_required_search_pairs"])
            or bool(row["missing_required_search_pairs"])
            for row in treatment_rows
        ),
        "search_contract_expectation_mismatches": {
            str(case["id"]): {
                "unexpected": row["unexpected_required_search_pairs"],
                "missing": row["missing_required_search_pairs"],
            }
            for case, row in zip(cases, treatment_rows)
            if row["unexpected_required_search_pairs"]
            or row["missing_required_search_pairs"]
        },
    }

    fingerprints = sorted({str(item.get("run_fingerprint") or "") for item in manifests})
    frozen_hashes = sorted({
        hashlib.sha256(
            json.dumps(item.get("frozen") or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for item in manifests
    })
    completeness = {
        "expected_arm_results": len(cases) * 2,
        "observed_arm_results": len(indexed),
        "complete": len(indexed) == len(cases) * 2,
        "missing": [
            {"case_id": str(case["id"]), "arm": arm}
            for case in cases
            for arm in ARMS
            if (str(case["id"]), arm) not in indexed
        ],
        "run_fingerprints": fingerprints,
        "single_frozen_configuration": len(fingerprints) == 1 and len(frozen_hashes) == 1,
    }
    blind = _build_blind_packet(cases, per_case, output_dir)

    public_case_results = {
        case_id: {
            arm: {
                key: value
                for key, value in row.items()
                if key != "answer"
            }
            for arm, row in arms.items()
        }
        for case_id, arms in per_case.items()
    }
    return {
        "schema_version": "debug_agent_system.terminology_ab_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path.relative_to(REPO_ROOT)),
        "run_dirs": [str(path.relative_to(REPO_ROOT)) for path in run_dirs],
        "completeness": completeness,
        "summary": summaries,
        "terminology_contract": resolver_summary,
        "blind_review": {
            **blind,
            "route_accuracy": "pending_blind_review",
            "answer_quality": "pending_blind_review",
            "semantic_wrong_variant_lock": "pending_blind_review",
        },
        "case_results": public_case_results,
    }


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def render_markdown(report: dict[str, Any]) -> str:
    complete = report["completeness"]
    contract = report["terminology_contract"]
    lines = [
        "# KG_v2 读侧术语层 A/B 全量测试报告",
        "",
        f"生成时间：`{report['generated_at']}`",
        "",
        "## 结论状态",
        "",
        (
            f"已收齐 `{complete['observed_arm_results']}/{complete['expected_arm_results']}` "
            f"个 arm 结果；全量完整性：`{complete['complete']}`；冻结配置一致："
            f"`{complete['single_frozen_configuration']}`。"
        ),
        "",
        "路由正确率、回答质量和语义级错误根因锁定需要盲评，当前报告不以启发式规则伪造这些分数。"
        "已生成 `blind_review_packet.jsonl`；只有 A/B 两边均成功发布的题进入盲评包。",
        "",
        "## 自动指标",
        "",
        "| 子集 | A 发布率 | B 发布率 | B-A | A 证据召回 | B 证据召回 | B-A |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "all": "全量",
        "canonical_name": "规范名",
        "field_alias_abbreviation_typo": "别称/缩写/错拼",
        "english_or_mixed": "英文/混写",
        "safety_ambiguity_negative": "歧义/安全负例",
    }
    for key, label in labels.items():
        row = report["summary"][key]
        a = row["A_control"]
        b = row["B_treatment"]
        delta = row["delta_B_minus_A"]
        lines.append(
            f"| {label} | {_fmt_rate(a['publish_rate'])} | {_fmt_rate(b['publish_rate'])} | "
            f"{_fmt_rate(delta['publish_rate'])} | {_fmt_rate(a['evidence_recall'])} | "
            f"{_fmt_rate(b['evidence_recall'])} | {_fmt_rate(delta['evidence_recall'])} |"
        )
    lines.extend([
        "",
        "证据召回按严格口径计算：金标来源必须同时出现在 `files_read` 和最终答案引用中；"
        "发布失败按 0 计，无码金标来源的题不参与该指标。",
        "",
        "## 术语契约与安全门禁",
        "",
        f"- 离线解析应命中题：`{contract['must_resolve_complete_count']}/"
        f"{contract['must_resolve_case_count']}` 完整命中。",
        f"- 强制双词检索：`{contract['required_search_complete_count']}/"
        f"{contract['required_search_case_count']}` 完成，合规率 "
        f"`{_fmt_rate(contract['required_search_compliance'])}`。",
        f"- 安全负例错误扩展：`{contract['unsafe_expansion_count']}` 次。",
        f"- 生成的硬检索契约与测试预期不一致："
        f"`{contract['search_contract_expectation_mismatch_count']}` 题。",
        "- 机制级锁定权限：需要结合逐题 artifact 检查 `can_lock_variant=false`；"
        "语义级确定性根因仍由盲评确认。",
        "",
        "### 未完整解析题",
        "",
        ", ".join(f"`{item}`" for item in contract["must_resolve_incomplete_case_ids"])
        or "无。",
        "",
        "### 强制检索未完成题",
        "",
        ", ".join(f"`{item}`" for item in contract["required_search_failed_case_ids"])
        or "无。",
        "",
        "### 安全错误扩展",
        "",
        json.dumps(contract["unsafe_expansion_cases"], ensure_ascii=False, indent=2),
        "",
        "## 发布失败分类",
        "",
        "| 子集 | Arm | required search 未执行 | 外链审计 | 方法变体 | 其他 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for key, label in labels.items():
        for arm in ARMS:
            counts = report["summary"][key][arm]["failure_labels"]
            lines.append(
                f"| {label} | {arm} | {counts.get('required_search_not_executed', 0)} | "
                f"{counts.get('external_artifact_verification', 0)} | "
                f"{counts.get('unexpected_procedure_variant', 0)} | "
                f"{counts.get('other_runtime_or_verification', 0) + counts.get('other_terminology_contract', 0)} |"
            )
    lines.extend([
        "",
        "## 尚待盲评",
        "",
        f"可盲评配对数：`{report['blind_review']['pair_count']}`。完成盲评后才能正式计算 "
        "`route_accuracy`、`answer_quality`、逐题胜负和语义级 `wrong_variant_lock_count`。",
        "",
        "因此，在盲评回填前，只能对自动门禁作结论，不能声称整体回答质量已经提升。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = [path.resolve() for path in args.run_dir]
    report = aggregate(
        cases_path=args.cases.resolve(),
        run_dirs=run_dirs,
        output_dir=args.output_dir.resolve(),
    )
    _write_json(args.output_dir.resolve() / "report.json", report)
    (args.output_dir.resolve() / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "complete": report["completeness"]["complete"],
        "observed": report["completeness"]["observed_arm_results"],
        "expected": report["completeness"]["expected_arm_results"],
        "report": str(args.output_dir.resolve() / "report.md"),
    }, ensure_ascii=False))
    return 0 if report["completeness"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
