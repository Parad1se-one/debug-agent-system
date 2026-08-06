"""Retrieval-only comparison for JSON KG vs SQLite SAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.read.o_kg_retrieval import KGRetrievalAgent
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge.sqlite_sag import SqliteSAGStore

from .scenario_v2 import load_scenarios


def run_retrieval_compare(
    *,
    scenario_file: str | Path,
    kg_root: str | Path,
    sqlite_sag_path: str | Path,
    limit: int = 150,
    top_k: int = 5,
) -> dict[str, Any]:
    scenarios = load_scenarios(scenario_file, limit)
    json_agent = KGRetrievalAgent(JsonKGStore(kg_root))
    sag_store = SqliteSAGStore(sqlite_sag_path, kg_root=kg_root)
    sag_agent = KGRetrievalAgent(sag_store)
    details: list[dict[str, Any]] = []
    for scenario in scenarios:
        expected = {scenario.target_error_id, *scenario.acceptable_error_ids} - {""}
        json_candidates = json_agent.retrieve(scenario.query, limit=top_k)
        sag_candidates = sag_agent.retrieve(scenario.query, limit=top_k)
        json_ids = [c.error_id for c in json_candidates]
        sag_ids = [c.error_id for c in sag_candidates]
        trace = getattr(sag_store, "last_retrieval_trace", {}) or {}
        details.append({
            "case_id": scenario.case_id,
            "query": scenario.query,
            "target_error_id": scenario.target_error_id,
            "acceptable_error_ids": scenario.acceptable_error_ids,
            "json_top_ids": json_ids,
            "sag_top_ids": sag_ids,
            "json_target_at_1": _hit(json_ids[:1], expected),
            "sag_target_at_1": _hit(sag_ids[:1], expected),
            "json_target_at_k": _hit(json_ids, expected),
            "sag_target_at_k": _hit(sag_ids, expected),
            "sag_trace_present": bool(trace.get("candidate_paths")),
            "sag_trace": trace,
        })
    return {
        "schema_version": "debug_agent_system.sag_retrieval_eval.v1",
        "scenario_file": str(scenario_file),
        "kg_root": str(kg_root),
        "sqlite_sag_path": str(sqlite_sag_path),
        "summary": _summary(details),
        "details": details,
    }


def _hit(ids: list[str], expected: set[str]) -> float:
    if not expected:
        return 0.0
    return 1.0 if any(item in expected for item in ids) else 0.0


def _summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(details)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "json_target_at_1": round(sum(x["json_target_at_1"] for x in details) / n, 4),
        "sag_target_at_1": round(sum(x["sag_target_at_1"] for x in details) / n, 4),
        "json_target_at_k": round(sum(x["json_target_at_k"] for x in details) / n, 4),
        "sag_target_at_k": round(sum(x["sag_target_at_k"] for x in details) / n, 4),
        "sag_trace_coverage": round(sum(1 for x in details if x["sag_trace_present"]) / n, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", default="data/eval/scenarios/broad_debug_v1.json")
    parser.add_argument("--kg-root", default="data/kg")
    parser.add_argument("--sqlite-sag-path", default="data/kg_sag/debug_agent.sqlite")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="data/results/sag_retrieval/latest.json")
    args = parser.parse_args(argv)
    report = run_retrieval_compare(
        scenario_file=args.scenario_file,
        kg_root=args.kg_root,
        sqlite_sag_path=args.sqlite_sag_path,
        limit=args.limit,
        top_k=args.top_k,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
