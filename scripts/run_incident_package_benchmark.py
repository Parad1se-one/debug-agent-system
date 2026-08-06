#!/usr/bin/env python3
"""Run public structural checks for the Incident Evidence Runtime.

This runner intentionally rejects non-validation inputs and never reads the
formal benchmark's private or held-out Gold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import kg_v2_graph_revision
from debug_agent_system.runtime import DebugAgentSystem


DEFAULT_INPUT = Path(
    "data/eval/formal_debug_benchmark_v1/incident_package_validation.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/debug_agent_system.yaml")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default="")
    return parser


def _contains_environment(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    values = (result.get("environment") or {}).get("values") or {}
    return all(str(value) in [str(item) for item in values.get(key, [])] for key, value in expected.items())


def _case_errors(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    events = result.get("events") or []
    stacks = result.get("stack_traces") or []
    retrieval_query = str((result.get("retrieval") or {}).get("retrieval_query") or "")
    event_codes = {
        str(code)
        for event in events
        for code in (event.get("error_codes") or [])
        if isinstance(event, dict)
    }
    require(result.get("status") == expected.get("status"), "status_mismatch")
    require(len(events) >= int(expected.get("minimum_events", 0)), "event_count_below_minimum")
    require(len(stacks) >= int(expected.get("minimum_stack_traces", 0)), "stack_count_below_minimum")
    require(
        _contains_environment(result, expected.get("environment_contains") or {}),
        "environment_value_missing",
    )
    for code in expected.get("event_error_codes_include") or []:
        require(str(code) in event_codes, f"event_error_code_missing:{code}")
    for value in expected.get("retrieval_query_excludes") or []:
        require(str(value) not in retrieval_query, f"volatile_value_in_retrieval_query:{value}")
    require(
        (result.get("evidence_pack") or {}).get("schema_version")
        == expected.get("evidence_pack_schema"),
        "evidence_pack_schema_mismatch",
    )
    require(
        bool((result.get("verification") or {}).get("passed"))
        is bool(expected.get("verification_passed")),
        "verification_status_mismatch",
    )
    require(
        bool((result.get("observability") or {}).get("canonical_kg_mutated"))
        is bool(expected.get("canonical_kg_mutated")),
        "canonical_kg_mutation_flag_mismatch",
    )
    report = str(result.get("report") or "")
    for value in expected.get("report_contains") or []:
        require(str(value) in report, f"report_text_missing:{value}")
    return errors


def main() -> int:
    args = _parser().parse_args()
    source_path = Path(args.input)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("split") != "validation":
        raise ValueError("incident_benchmark_only_allows_public_validation_split")
    if payload.get("held_out_gold_required") is not False:
        raise ValueError("incident_benchmark_must_not_require_held_out_gold")

    config = load_config(args.config)
    kg_root = config.knowledge.kg_v2_root
    before_revision = kg_v2_graph_revision(kg_root)
    case_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="incident-benchmark-sessions-") as temp_dir:
        config.session_store = Path(temp_dir)
        system = DebugAgentSystem(config)
        for case in payload.get("cases") or []:
            result = system.analyze_incident({
                "query": case.get("query") or "",
                "interactive": False,
                "evidence_resources": case.get("resources") or [],
            })
            errors = _case_errors(result, case.get("assertions") or {})
            case_results.append({
                "case_id": case.get("case_id"),
                "passed": not errors,
                "errors": errors,
                "observed": {
                    "status": result.get("status"),
                    "event_count": len(result.get("events") or []),
                    "stack_trace_count": len(result.get("stack_traces") or []),
                    "hypothesis_count": len(result.get("hypotheses") or []),
                    "evidence_pack_schema": (result.get("evidence_pack") or {}).get("schema_version"),
                },
            })

    after_revision = kg_v2_graph_revision(kg_root)
    if before_revision != after_revision:
        for item in case_results:
            item["passed"] = False
            item["errors"].append("canonical_kg_revision_changed")
    output = {
        "schema_version": "debug_agent_system.incident_package_benchmark_result.v1",
        "source": str(source_path),
        "split": "validation",
        "included_in_formal_accuracy": False,
        "held_out_gold_read": False,
        "canonical_kg_revision_before": before_revision,
        "canonical_kg_revision_after": after_revision,
        "passed": all(item["passed"] for item in case_results),
        "cases": case_results,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
