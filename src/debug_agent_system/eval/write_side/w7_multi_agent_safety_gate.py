"""Aggregate fixed-set W7 shadow safety metrics and enforce release gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


W7A_STAGES = (
    "case_boundary",
    "evidence_anchor",
    "atomic_case_adapter",
)
W7B_STAGES = (
    "candidate_graph",
    "neighbor_link",
    "component_consistency",
    "component_bridge",
    "trace_components",
    "trace_phase",
    "outcome_reconciliation",
    "trace_compiler",
)


def _load_result_paths(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in manifest.get("results") or []:
        if not isinstance(row, dict):
            continue
        value = Path(str(row.get("result") or ""))
        candidates = (
            [value]
            if value.is_absolute()
            else [manifest_path.parent / value, value]
        )
        result_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if result_path is None:
            missing.append(str(value))
            continue
        output.append(json.loads(result_path.read_text(encoding="utf-8")))
    return output, missing


def build_report(
    *,
    manifest_path: Path,
    expected_episodes: int = 173,
    min_schema_valid_rate: float = 1.0,
    require_state_hash: bool = True,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results, missing_result_files = _load_result_paths(
        manifest_path, manifest
    )
    episode_ids: list[str] = []
    stage_failures: Counter[str] = Counter()
    quality_decisions: Counter[str] = Counter()
    typed_issue_counts: Counter[str] = Counter()
    total_edges = identity_edges = semantic_edges = 0
    components = standalone_components = multi_case_components = 0
    max_component_size = 0
    max_edge_density = 0.0
    mutation_flags: list[str] = []
    invalid_batches: list[str] = []
    typed_schema_valid = 0
    typed_candidates = 0

    for result in results:
        batch_id = str(result.get("batch_id") or "")
        if not bool(result.get("schema_valid")):
            invalid_batches.append(batch_id)
        if bool(result.get("promotion_allowed")):
            mutation_flags.append(f"promotion_allowed:{batch_id}")
        if not bool(result.get("legacy_authoritative")):
            mutation_flags.append(f"legacy_not_authoritative:{batch_id}")
        if bool(result.get("queue_written")):
            mutation_flags.append(f"queue_written:{batch_id}")
        if bool(result.get("kg_mutated")):
            mutation_flags.append(f"kg_mutated:{batch_id}")

        for unit in result.get("units") or []:
            if not isinstance(unit, dict):
                continue
            episode_id = str(unit.get("episode_id") or "")
            if episode_id:
                episode_ids.append(episode_id)
            w7a = unit.get("w7a") if isinstance(unit.get("w7a"), dict) else {}
            for stage in W7A_STAGES:
                if not bool((w7a.get(stage) or {}).get("schema_valid")):
                    stage_failures[stage] += 1
        for stage in W7B_STAGES:
            if not bool((result.get(stage) or {}).get("schema_valid")):
                stage_failures[stage] += 1

        graph = (
            (result.get("candidate_graph") or {}).get("graph")
            if isinstance(result.get("candidate_graph"), dict)
            else {}
        ) or {}
        cards = int((result.get("stats") or {}).get("case_cards") or 0)
        edges = [
            item for item in graph.get("edges") or []
            if isinstance(item, dict)
        ]
        total_edges += len(edges)
        identity_edges += sum(
            item.get("edge_class") == "identity_edge"
            for item in edges
        )
        semantic_edges += sum(
            item.get("edge_class") != "identity_edge"
            for item in edges
        )
        possible_edges = cards * (cards - 1) // 2
        density = len(edges) / possible_edges if possible_edges else 0.0
        max_edge_density = max(max_edge_density, density)

        component_values = (
            ((result.get("trace_components") or {}).get("graph") or {}).get(
                "components"
            )
            if isinstance(result.get("trace_components"), dict)
            else []
        ) or []
        for component in component_values:
            if not isinstance(component, dict):
                continue
            size = len(component.get("case_refs") or [])
            components += 1
            max_component_size = max(max_component_size, size)
            if size <= 1:
                standalone_components += 1
            else:
                multi_case_components += 1

        if isinstance(result.get("typed_candidate"), dict):
            typed_candidates += 1
            typed_schema_valid += bool(
                result["typed_candidate"].get("schema_valid")
            )
        for issue in result.get("typed_candidate_build_issues") or []:
            typed_issue_counts[str(issue).split(":", 1)[0]] += 1
        quality = (
            result.get("quality_gate")
            if isinstance(result.get("quality_gate"), dict)
            else {}
        )
        if quality:
            quality_decisions[str(quality.get("decision") or "unknown")] += 1

    duplicates = sorted(
        episode_id
        for episode_id, count in Counter(episode_ids).items()
        if count > 1
    )
    observed_episodes = len(episode_ids)
    schema_valid_batches = len(results) - len(invalid_batches)
    schema_valid_rate = (
        schema_valid_batches / len(results) if results else 0.0
    )
    state_hashes = (
        manifest.get("state_hashes")
        if isinstance(manifest.get("state_hashes"), dict)
        else {}
    )
    requirements = {
        "result_files_complete": not missing_result_files,
        "episode_coverage_exact": observed_episodes == expected_episodes,
        "episode_ids_unique": not duplicates,
        "manifest_shadow_only": (
            not bool(manifest.get("promotion_allowed"))
            and bool(manifest.get("legacy_authoritative"))
        ),
        "result_shadow_only": not mutation_flags,
        "state_unchanged": (
            bool(state_hashes.get("unchanged"))
            if require_state_hash else True
        ),
        "schema_valid_rate": (
            schema_valid_rate >= min_schema_valid_rate
        ),
        "component_size_bounded": max_component_size <= 12,
        "typed_candidates_valid": (
            typed_candidates == 0
            or typed_schema_valid == typed_candidates
        ),
    }
    return {
        "schema_version": "w7.multi_agent_safety_gate.v1",
        "manifest": str(manifest_path),
        "manifest_hash": str(manifest.get("manifest_hash") or ""),
        "input_sha256": str(manifest.get("input_sha256") or ""),
        "configuration": {
            "expected_episodes": expected_episodes,
            "min_schema_valid_rate": min_schema_valid_rate,
            "require_state_hash": require_state_hash,
        },
        "coverage": {
            "episodes": observed_episodes,
            "unique_episodes": len(set(episode_ids)),
            "duplicate_episode_ids": duplicates,
            "batches": len(results),
            "missing_result_files": missing_result_files,
        },
        "schema": {
            "valid_batches": schema_valid_batches,
            "invalid_batches": invalid_batches,
            "valid_rate": round(schema_valid_rate, 6),
            "stage_failures": dict(sorted(stage_failures.items())),
            "typed_candidates": typed_candidates,
            "typed_schema_valid": typed_schema_valid,
            "typed_issue_counts": dict(sorted(typed_issue_counts.items())),
        },
        "graph": {
            "edges": total_edges,
            "identity_edges": identity_edges,
            "semantic_edges": semantic_edges,
            "max_edge_density": round(max_edge_density, 6),
            "components": components,
            "standalone_components": standalone_components,
            "multi_case_components": multi_case_components,
            "max_component_size": max_component_size,
        },
        "quality_gate_decisions": dict(sorted(quality_decisions.items())),
        "safety": {
            "mutation_flags": mutation_flags,
            "state_hashes": state_hashes,
        },
        "requirements": requirements,
        "gate": {
            "status": (
                "PASS" if all(requirements.values()) else "FAIL"
            ),
            "failed_requirements": [
                key for key, value in requirements.items() if not value
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="w7-multi-agent-safety-gate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=173)
    parser.add_argument(
        "--min-schema-valid-rate", type=float, default=1.0
    )
    parser.add_argument("--allow-missing-state-hash", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        manifest_path=args.manifest,
        expected_episodes=max(0, int(args.expected_episodes)),
        min_schema_valid_rate=max(
            0.0, min(1.0, float(args.min_schema_valid_rate))
        ),
        require_state_hash=not args.allow_missing_state_hash,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "coverage": report["coverage"],
        "schema": report["schema"],
        "graph": report["graph"],
        "gate": report["gate"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
