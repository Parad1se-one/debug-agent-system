#!/usr/bin/env python3
"""Validate the KG_v2 terminology A/B dataset and optional live resolver fit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DEFAULT_ROOT = REPO_ROOT / "data/eval/terminology_ab_v1"
EXPECTED_QUOTAS = {
    "canonical_name": 15,
    "field_alias_abbreviation_typo": 20,
    "english_or_mixed": 15,
    "safety_ambiguity_negative": 10,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"case_object_required:{line_number}")
        cases.append(value)
    return cases


def _ontology_inventory() -> tuple[set[str], set[tuple[str, str]]]:
    inventory = _load_json(
        REPO_ROOT
        / "data/kg_v2/terminology/noun_terminology_inventory.json"
    )
    authoritative = [
        item
        for item in inventory.get("authoritative_concepts") or []
        if isinstance(item, dict)
    ]
    concepts = {
        str(item.get("canonical_name") or "").strip()
        for item in authoritative
    }
    aliases = {
        (
            str(alias.get("surface_form") or "").strip(),
            str(item.get("canonical_name") or "").strip(),
        )
        for item in authoritative
        for alias in item.get("aliases") or []
        if isinstance(alias, dict)
    }
    return concepts, aliases


def _known_source_documents() -> set[str]:
    documents: set[str] = set()
    root = REPO_ROOT / "data/eval/formal_debug_benchmark_v1"
    for path in root.glob("feature_selftest_queries_*.jsonl"):
        for record in _load_cases(path):
            value = str(record.get("source_document") or "").strip()
            if value:
                documents.add(value)
    return documents


def validate_dataset(root: Path) -> dict[str, Any]:
    cases = _load_cases(root / "cases.jsonl")
    experiment = _load_json(root / "experiment.json")
    concepts, aliases = _ontology_inventory()
    sources = _known_source_documents()
    issues: list[str] = []

    counts = Counter(str(case.get("category") or "") for case in cases)
    if len(cases) != 60:
        issues.append(f"case_count:{len(cases)}!=60")
    if dict(counts) != EXPECTED_QUOTAS:
        issues.append(f"category_quotas:{dict(counts)}")
    if experiment.get("case_count") != len(cases):
        issues.append("experiment_case_count_mismatch")
    if experiment.get("category_quotas") != EXPECTED_QUOTAS:
        issues.append("experiment_category_quotas_mismatch")

    ids = [str(case.get("id") or "") for case in cases]
    queries = [str(case.get("query") or "").strip() for case in cases]
    if len(set(ids)) != len(ids):
        issues.append("duplicate_case_id")
    if len(set(queries)) != len(queries):
        issues.append("duplicate_query")

    arms = (experiment.get("paired_design") or {}).get("arms") or {}
    if arms != {
        "A_control": {"terminology_enabled": False},
        "B_treatment": {"terminology_enabled": True},
    }:
        issues.append("arms_must_differ_only_by_terminology_enabled")

    required_keys = {
        "must_resolve",
        "must_not_resolve",
        "required_search_pairs",
        "ambiguous_surfaces",
        "must_not_lock_variant",
    }
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or f"line-{index}")
        expected_id = f"term-ab-{index:03d}"
        if case_id != expected_id:
            issues.append(f"{case_id}:expected_id:{expected_id}")
        if not queries[index - 1]:
            issues.append(f"{case_id}:empty_query")
        expected = case.get("expected")
        if not isinstance(expected, dict) or set(expected) != required_keys:
            issues.append(f"{case_id}:invalid_expected_shape")
            continue
        if expected.get("must_not_lock_variant") is not True:
            issues.append(f"{case_id}:terminology_may_lock_variant")
        for field in ("must_resolve", "must_not_resolve"):
            for concept in expected.get(field) or []:
                if concept not in concepts:
                    issues.append(f"{case_id}:{field}:unknown:{concept}")
        for pair in expected.get("required_search_pairs") or []:
            if not isinstance(pair, list) or len(pair) != 2:
                issues.append(f"{case_id}:invalid_search_pair:{pair}")
                continue
            if tuple(pair) not in aliases:
                issues.append(
                    f"{case_id}:search_pair_not_approved:{pair[0]}->{pair[1]}"
                )
        for source in case.get("gold_source_documents") or []:
            if source not in sources:
                issues.append(f"{case_id}:unknown_gold_source:{source}")
        axes = set(case.get("outcome_axes") or [])
        if not axes:
            issues.append(f"{case_id}:missing_outcome_axes")
        if case.get("category") == "safety_ambiguity_negative":
            if not {"unsafe_expansion", "wrong_variant_lock"} <= axes:
                issues.append(f"{case_id}:missing_safety_axes")
            if case.get("gold_source_documents"):
                issues.append(f"{case_id}:safety_case_has_gold_source")

    return {
        "schema_version": "debug_agent_system.terminology_ab_validation.v1",
        "status": "passed" if not issues else "failed",
        "case_count": len(cases),
        "category_counts": dict(counts),
        "issue_count": len(issues),
        "issues": issues,
    }


def check_current_resolver(root: Path) -> dict[str, Any]:
    from debug_agent_system.kg_raw_codex.coverage import build_answer_scope
    from debug_agent_system.kg_raw_codex.terminology_contract import (
        build_resolver_context,
        build_terminology_search_contract,
    )
    from debug_agent_system.knowledge_v2.terminology import TerminologyResolver

    resolver = TerminologyResolver.from_root(REPO_ROOT / "data/kg_v2")
    failures: list[dict[str, Any]] = []
    for case in _load_cases(root / "cases.jsonl"):
        query = str(case["query"])
        scope = build_answer_scope(query)
        resolution = resolver.resolve(
            query,
            limit=30,
            context=build_resolver_context(scope),
        )
        contract = build_terminology_search_contract(query, resolution)
        resolved = {
            str((item.get("concept") or {}).get("canonical_name") or "")
            for item in resolution.get("resolved_mentions") or []
        }
        actual_pairs = {
            (
                str(group.get("source_surface_form") or ""),
                str(group.get("canonical_name") or ""),
            )
            for group in contract.get("required_search_groups") or []
        }
        expected = case["expected"]
        missing = sorted(set(expected["must_resolve"]) - resolved)
        forbidden = sorted(set(expected["must_not_resolve"]) & resolved)
        missing_pairs = sorted(
            {tuple(pair) for pair in expected["required_search_pairs"]}
            - actual_pairs
        )
        if missing or forbidden or missing_pairs:
            failures.append({
                "id": case["id"],
                "category": case["category"],
                "missing_concepts": missing,
                "forbidden_concepts": forbidden,
                "missing_search_pairs": [list(pair) for pair in missing_pairs],
            })
    return {
        "schema_version": (
            "debug_agent_system.terminology_ab_resolver_compatibility.v1"
        ),
        "status": "passed" if not failures else "failed",
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check-current-resolver", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(args.root)
    if args.check_current_resolver:
        report["current_resolver"] = check_current_resolver(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
