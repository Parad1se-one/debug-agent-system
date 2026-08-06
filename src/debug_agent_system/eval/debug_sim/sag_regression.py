"""Focused read-side regression checks for the SQLite SAG store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.runtime import DebugAgentSystem
from .scenario_v2 import ScenarioV2
from .scorer import score_case
from .trace_diagnosis import build_trace_digest


def run_regression(*, config: str | Path, scenario_file: str | Path) -> dict[str, Any]:
    system = DebugAgentSystem(load_config(config))
    cases = _read_cases(scenario_file)
    details: list[dict[str, Any]] = []
    for case in cases:
        details.extend(_run_case(system, case))
    failed = [row for row in details if row["violations"]]
    branch_required = sum(len(row["required_branch_targets"]) for row in details)
    branch_hits = sum(row["branch_target_hits"] for row in details)
    return {
        "schema_version": "debug_agent_system.sag_regression.v1",
        "config": str(config),
        "scenario_file": str(scenario_file),
        "summary": {
            "n": len(details),
            "passed": len(details) - len(failed),
            "failed": len(failed),
            "trace_coverage": _rate(sum(1 for row in details if row["trace_present"]), len(details)),
            "final_trace_alignment_rate": _rate(sum(1 for row in details if row["final_trace_aligned"]), len(details)),
            "d_only_top_candidate_rate": _rate(sum(1 for row in details if row["d_only_top_candidate"]), len(details)),
            "source_mismatch_first_check_rate": _rate(sum(1 for row in details if row["source_mismatch_first_check"]), len(details)),
            "family_canonical_hit_rate": _rate(sum(1 for row in details if row["family_canonical_hit"]), len([row for row in details if row["expected_top_error_ids"]])),
            "forbidden_current_check_violations": sum(1 for row in details for item in row["violations"] if item.startswith("forbidden_current_check_id:")),
            "branch_target_recall": _rate(branch_hits, branch_required),
            "tier_d_executable_links": _tier_d_executable_links(system),
        },
        "details": details,
    }


def _run_case(system: DebugAgentSystem, case: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = case.get("variants") or [{"name": "default", "interactive": True}]
    for variant in variants:
        out = system.start({
            "query": str(case.get("query") or ""),
            "interactive": bool(variant.get("interactive", True)),
            "session": {"session_id": f"eval-{case.get('case_id')}-{variant.get('name')}"},
        })
        rows.append(_check(case, variant, out))
    return rows


def _check(case: dict[str, Any], variant: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    metadata = out.get("metadata") or {}
    observability = out.get("observability") or {}
    current_check_id = str(out.get("current_check_id") or "")
    presented_ids = {str(item) for item in metadata.get("presented_check_ids") or [] if str(item)}
    branch_targets = {
        str(item.get("to_check_id") or "")
        for item in metadata.get("branch_options") or []
        if isinstance(item, dict)
    }
    top_error_id = str(observability.get("top_error_id") or "")
    trace = metadata.get("retrieval_trace") or {}
    trace_summary = trace.get("summary") or {}
    violations: list[str] = []

    expected_top = set(_strings(case.get("expected_top_error_ids")))
    if expected_top and top_error_id not in expected_top:
        violations.append(f"top_error_id:{top_error_id}:expected={sorted(expected_top)}")

    expected_statuses = set(_strings(variant.get("expected_statuses")))
    status = str(out.get("status") or "")
    if expected_statuses and status not in expected_statuses:
        violations.append(f"status:{status}:expected={sorted(expected_statuses)}")
    forbidden_statuses = set(_strings(variant.get("forbidden_statuses")))
    if forbidden_statuses and status in forbidden_statuses:
        violations.append(f"forbidden_status:{status}")

    expected_checks = set(_strings(variant.get("expected_current_check_ids")))
    if expected_checks and current_check_id not in expected_checks:
        violations.append(f"current_check_id:{current_check_id}:expected={sorted(expected_checks)}")

    forbidden_checks = set(_strings(variant.get("forbidden_current_check_ids")))
    if forbidden_checks and current_check_id in forbidden_checks:
        violations.append(f"forbidden_current_check_id:{current_check_id}")

    required_branch_targets = set(_strings(variant.get("required_branch_targets")))
    missing_branch_targets = sorted(required_branch_targets - branch_targets)
    if missing_branch_targets:
        violations.append(f"missing_branch_targets:{missing_branch_targets}")

    required_presented = set(_strings(variant.get("required_presented_check_ids")))
    missing_presented = sorted(required_presented - presented_ids - {current_check_id})
    if missing_presented:
        violations.append(f"missing_presented_check_ids:{missing_presented}")

    if bool(case.get("trace_required")):
        if not trace.get("candidate_paths"):
            violations.append("missing_retrieval_trace_candidate_paths")
        if not bool(trace_summary.get("final_trace_aligned")):
            violations.append("retrieval_trace_not_aligned_with_final_candidates")

    source_mismatch_first_check = bool(metadata.get("source_mismatch_first_check"))
    if source_mismatch_first_check and not bool(variant.get("allow_source_mismatch_first_check")):
        violations.append("source_mismatch_first_check")
    scenario = _scenario_from_case(case, variant)
    transcript = {
        "case_id": scenario.case_id,
        "query": scenario.query,
        "expected_status": scenario.expected_status,
        "final_status": out.get("status"),
        "checks_presented": sorted(presented_ids),
        "required_checks": [],
        "terminal_ok": out.get("status") in {"resolved", "escalate"},
        "simulator_gap": False,
        "latency_ms": None,
        "first_check_id": current_check_id,
        "first_check_text": str(out.get("current_check") or ""),
        "top_error_id": top_error_id,
        "retrieval_trace_present": bool(trace.get("candidate_paths")),
        "current_check_id": current_check_id,
        "current_check_text": str(out.get("current_check") or ""),
        "presented_check_trace": metadata.get("presented_check_trace") or [],
        "selected_check_trace": metadata.get("selected_check_trace") or {},
        "branch_trace": metadata.get("branch_trace") or [],
        "branch_options": metadata.get("branch_options") or [],
        "trace_digest": build_trace_digest({
            "turns": [{"actor": "agent", "response": out}],
            "current_check_id": current_check_id,
            "first_check_id": current_check_id,
            "presented_check_trace": metadata.get("presented_check_trace") or [],
            "selected_check_trace": metadata.get("selected_check_trace") or {},
            "branch_trace": metadata.get("branch_trace") or [],
            "branch_options": metadata.get("branch_options") or [],
        }),
        "replay_events": [],
        "turns": [{"actor": "agent", "response": out}],
    }
    detail = score_case(scenario, transcript)
    return {
        "case_id": str(case.get("case_id") or ""),
        "variant": str(variant.get("name") or ""),
        "interactive": bool(variant.get("interactive", True)),
        "status": status,
        "expected_top_error_ids": sorted(expected_top),
        "top_error_id": top_error_id,
        "family_canonical_hit": bool(expected_top and top_error_id in expected_top),
        "current_check_id": current_check_id,
        "retrieval_route": str(observability.get("retrieval_route") or ""),
        "branch_targets": sorted(branch_targets),
        "required_branch_targets": sorted(required_branch_targets),
        "branch_target_hits": len(required_branch_targets & branch_targets),
        "presented_check_ids": sorted(presented_ids),
        "presented_check_trace": metadata.get("presented_check_trace") or [],
        "source_mismatch_first_check": source_mismatch_first_check,
        "trace_present": bool(trace.get("candidate_paths")),
        "final_trace_aligned": bool(trace_summary.get("final_trace_aligned")),
        "d_only_top_candidate": bool(trace_summary.get("d_only_top_candidate")),
        "trace_summary": trace_summary,
        "candidate_ids": [
            str(item.get("error_id") or "")
            for item in metadata.get("retrieval_candidates") or []
            if isinstance(item, dict)
        ][:8],
        "trace_digest": detail.get("trace_digest") or {},
        "trace_diagnosis": detail.get("trace_diagnosis") or {},
        "failure_stage": detail.get("failure_stage") or "",
        "failure_cause": detail.get("failure_cause") or "",
        "violations": violations,
    }


def _read_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("cases") or []
    else:
        rows = data
    return [row for row in rows if isinstance(row, dict)]


def _strings(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _scenario_from_case(case: dict[str, Any], variant: dict[str, Any]) -> ScenarioV2:
    return ScenarioV2(
        case_id=str(case.get("case_id") or ""),
        query=str(case.get("query") or ""),
        source="sag_regression",
        query_type="debug",
        target_error_id=str((case.get("expected_top_error_ids") or [""])[0] or ""),
        acceptable_error_ids=[str(x) for x in (case.get("expected_top_error_ids") or [])[1:] if str(x)],
        expected_status=str((variant.get("expected_statuses") or [""])[0] or ""),
        metadata={"required_branch_targets": list(variant.get("required_branch_targets") or [])},
    )


def _tier_d_executable_links(system: DebugAgentSystem) -> int | None:
    conn = getattr(getattr(system, "store", None), "conn", None)
    if conn is None:
        return None
    return int(conn.execute(
        """
        SELECT COUNT(*) FROM event_links
        WHERE source_tier = 'D'
          AND relation IN ('has_check', 'next', 'resolved_by', 'requires_info')
        """
    ).fetchone()[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/debug_agent_system_sag.yaml")
    parser.add_argument("--scenario-file", default="data/eval/scenarios/sag_regression_v1.json")
    parser.add_argument("--out", default="data/results/sag_regression/latest.json")
    args = parser.parse_args(argv)
    report = run_regression(config=args.config, scenario_file=args.scenario_file)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    tier_d_executable_links = report["summary"].get("tier_d_executable_links")
    if tier_d_executable_links:
        return 1
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
