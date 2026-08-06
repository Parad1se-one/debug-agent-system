"""Replay the executable track of the KG_v2 quality dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.runtime.system import DebugAgentSystem


DEFAULT_DATASET = Path("data/eval/scenarios/kg_v2_quality_v1.json")
DEFAULT_OUT = Path("data/results/kg_v2_read_eval/latest.json")


def _assertion(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {"name": name, "passed": actual == expected, "expected": expected, "actual": actual}


def _score_case(system: DebugAgentSystem, case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    response = system.start({
        "query": case["query"],
        "session": {"session_id": f"eval-{case['case_id']}"},
    })
    assertions = [
        _assertion("family_id", expected["family_id"], response.get("family_id")),
        _assertion("variant_top1", expected["variant_id"], response.get("variant_id")),
        _assertion("plan_id", expected["plan_id"], response.get("plan_id")),
    ]
    retrieval = (response.get("metadata") or {}).get("retrieval") or {}
    candidates = retrieval.get("candidates") or []
    top_candidate = candidates[0] if candidates else {}
    assertions.extend([
        _assertion("sag_route", expected["sag"]["expected_route"], top_candidate.get("route")),
        {
            "name": "sag_retrieval_paths",
            "passed": bool(top_candidate.get("retrieval_paths")),
            "expected": "non_empty",
            "actual": len(top_candidate.get("retrieval_paths") or []),
        },
        {
            "name": "evidence_non_empty",
            "passed": bool(response.get("evidence_ids")),
            "expected": "non_empty",
            "actual": len(response.get("evidence_ids") or []),
        },
    ])

    task_type = case["task_type"]
    if task_type not in {"ask_info_gate"}:
        assertions.append(_assertion("first_action_id", expected["first_action_id"], response.get("current_action_id")))
    if task_type in {"first_action", "ask_info_gate", "safety_gate"}:
        assertions.append(_assertion("initial_status", expected["terminal_status"], response.get("status")))

    # Do not feed labels for a different plan into the runtime.  Retrieval
    # failures remain visible and downstream checks are explicitly skipped.
    turns: list[dict[str, Any]] = []
    if response.get("variant_id") == expected["variant_id"]:
        for index, turn in enumerate(case.get("turns") or [], start=1):
            response = system.step(str(response["session_id"]), str(turn["user_message"]))
            turn_assertions = [_assertion("status", turn.get("expected_status"), response.get("status"))]
            if turn.get("expected_action_id"):
                turn_assertions.append(_assertion(
                    "action_id", turn["expected_action_id"], response.get("current_action_id")
                ))
            if turn.get("expected_branch_rule_id"):
                applied = (response.get("metadata") or {}).get("applied_branch_rule_ids") or []
                turn_assertions.append({
                    "name": "branch_rule_id",
                    "passed": turn["expected_branch_rule_id"] in applied,
                    "expected": turn["expected_branch_rule_id"],
                    "actual": applied,
                })
            if turn.get("expected_failure_type"):
                turn_assertions.append(_assertion(
                    "failure_type", turn["expected_failure_type"], response.get("failure_type")
                ))
            assertions.extend({**item, "turn": index} for item in turn_assertions)
            turns.append({
                "turn": index,
                "user_message": turn["user_message"],
                "status": response.get("status"),
                "action_id": response.get("current_action_id"),
                "failure_type": response.get("failure_type"),
            })

    failed = [item for item in assertions if not item["passed"]]
    return {
        "case_id": case["case_id"],
        "task_type": task_type,
        "difficulty": case["difficulty"],
        "reasoning_mode": case["reasoning_mode"],
        "passed": not failed,
        "assertion_count": len(assertions),
        "failed_assertion_count": len(failed),
        "assertions": assertions,
        "turns": turns,
    }


def replay_dataset(
    dataset: dict[str, Any],
    config_path: str | Path = "config/debug_agent_system.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    with tempfile.TemporaryDirectory(prefix="kg-v2-read-eval-") as temp_dir:
        config.session_store = Path(temp_dir) / "sessions"
        system = DebugAgentSystem(config)
        results = [
            _score_case(system, case)
            for case in dataset["cases"]
            if case.get("evaluation_track") == "runtime_replay"
        ]

    passed = sum(item["passed"] for item in results)
    breakdown: dict[str, dict[str, dict[str, int]]] = {}
    for field in ("task_type", "difficulty", "reasoning_mode"):
        groups: dict[str, Counter[str]] = defaultdict(Counter)
        for item in results:
            groups[str(item[field])]["passed" if item["passed"] else "failed"] += 1
            groups[str(item[field])]["total"] += 1
        breakdown[field] = {key: dict(value) for key, value in sorted(groups.items())}
    failed_assertions = Counter(
        assertion["name"]
        for result in results
        for assertion in result["assertions"]
        if not assertion["passed"]
    )
    return {
        "schema_version": "debug_agent_system.kg_v2_read_eval.replay.v1",
        "dataset_id": dataset.get("dataset_id"),
        "graph_revision": dataset.get("graph_revision"),
        "runtime_case_count": len(results),
        "gold_reasoning_case_count": sum(
            case.get("evaluation_track") == "gold_trace_reasoning" for case in dataset["cases"]
        ),
        "passed_case_count": passed,
        "failed_case_count": len(results) - passed,
        "case_pass_rate": round(passed / len(results), 4) if results else 0.0,
        "failed_assertion_counts": dict(sorted(failed_assertions.items())),
        "breakdown": breakdown,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--config", default="config/debug_agent_system.yaml")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    report = replay_dataset(dataset, args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"results"}}, ensure_ascii=False, indent=2))
    # Baseline replay is diagnostic: gaps are expected and written to report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
