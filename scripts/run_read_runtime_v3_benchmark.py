#!/usr/bin/env python3
"""Run Read Runtime v3 shadow mode on the existing formal validation set.

The default is intentionally the optimization-eligible validation split.
Held-out inputs require an explicit flag and are never scored or used for
iteration by this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from debug_agent_system.core.config import load_config
from debug_agent_system.eval.read_side.formal_debug_benchmark import score_predictions
from debug_agent_system.read_runtime_v3 import ReadRuntimeV3, load_options
from debug_agent_system.read_runtime_v3.evaluation import (
    response_to_formal_prediction,
    structural_errors,
)
from debug_agent_system.runtime import DebugAgentSystem


DEFAULT_VALIDATION = REPO_ROOT / "data/eval/formal_debug_benchmark_v1/core_validation.json"
DEFAULT_HELD_OUT = REPO_ROOT / "data/eval/formal_debug_benchmark_v1/core_test_inputs.json"
DEFAULT_RESULTS = REPO_ROOT / "data/results/read_runtime_v3/formal_validation"
FROZEN_MANIFEST = REPO_ROOT / "config/read_side_frozen_manifest_v2.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--system-config", type=Path, default=REPO_ROOT / "config/debug_agent_system.yaml")
    parser.add_argument("--v3-config", type=Path, default=REPO_ROOT / "config/read_runtime_v3.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-held-out-inputs", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    dataset = _load(dataset_path)
    split = str(dataset.get("split") or _single_split(dataset))
    if split == "held_out_test" and not args.allow_held_out_inputs:
        raise ValueError("held_out_inputs_require_explicit_allow_flag")
    if split not in {"validation", "held_out_test"}:
        raise ValueError(f"unsupported_split:{split}")
    cases = list(dataset.get("cases") or [])
    selected = cases[max(0, args.start - 1):]
    if args.limit is not None:
        selected = selected[:max(0, args.limit)]
    if split == "validation" and not all(case.get("optimization_eligible") is True for case in selected):
        raise ValueError("validation_case_not_optimization_eligible")
    if split == "held_out_test" and not all(case.get("optimization_eligible") is False for case in selected):
        raise ValueError("held_out_case_optimization_policy_invalid")

    run_fingerprint = _run_fingerprint(
        dataset_path=dataset_path,
        v3_config=args.v3_config.resolve(),
        selected=selected,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else DEFAULT_RESULTS / f"{stamp}_{run_fingerprint[:12]}"
    )
    output.mkdir(parents=True, exist_ok=True)

    before_freeze = _verify_freeze()
    if not before_freeze.get("frozen"):
        raise RuntimeError("frozen_read_pipeline_drift_before_v3_benchmark")
    options = load_options(args.v3_config.resolve())
    if not options.enabled or not options.shadow_mode:
        raise ValueError("v3_benchmark_requires_enabled_shadow_mode")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="read-runtime-v3-eval-") as sessions:
        config = load_config(args.system_config.resolve())
        config.session_store = Path(sessions)
        system = DebugAgentSystem(config)
        runtime = ReadRuntimeV3.from_system(system, options=options, workspace=REPO_ROOT)
        for index, case in enumerate(selected, start=max(1, args.start)):
            case_id = str(case.get("case_id") or f"case-{index}")
            artifact = output / "cases" / f"{case_id}.json"
            if args.resume and _reusable(
                artifact, case=case, run_fingerprint=run_fingerprint
            ):
                saved = _load(artifact)
                results.append({**saved, "execution_status": "reused"})
                print(json.dumps({"case_id": case_id, "status": "reused"}, ensure_ascii=False), flush=True)
                continue
            started = time.perf_counter()
            try:
                response = runtime.run(_request(case))
                prediction = response_to_formal_prediction(response)
                prediction["case_id"] = case_id
                errors = structural_errors(response)
                payload = {
                    "schema_version": "debug_agent_system.read_runtime_v3_case_result.v1",
                    "case_id": case_id,
                    "query_sha256": _sha_text(str(case.get("query") or "")),
                    "run_fingerprint": run_fingerprint,
                    "execution_status": "passed" if not errors else "contract_failed",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    "structural_errors": errors,
                    "prediction": prediction,
                    "response": response,
                }
                _write(artifact, payload)
                results.append(payload)
                print(json.dumps({
                    "case_id": case_id,
                    "status": payload["execution_status"],
                    "elapsed_ms": payload["elapsed_ms"],
                    "errors": errors,
                }, ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve per-case datum.
                failures.append({
                    "case_id": case_id,
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                })
                print(json.dumps({"case_id": case_id, "status": "failed", "error": failures[-1]["error"]}, ensure_ascii=False), flush=True)

    after_freeze = _verify_freeze()
    benchmark_compatibility = _benchmark_compatibility(selected)
    predictions = {
        "schema_version": "debug_agent_system.formal_debug_predictions.v1",
        "run_manifest": {
            **dict(dataset.get("version_manifest") or {}),
            "run_id": output.name,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "v3_runtime": asdict(options),
            "v3_run_fingerprint": run_fingerprint,
            "split": split,
            "requested_case_count": len(selected),
            "completed_case_count": len(results),
            "failed_case_count": len(failures),
        },
        "predictions": [item["prediction"] for item in results],
        "failures": failures,
    }
    subset = {**dataset, "cases": selected, "case_count": len(selected)}
    score = (
        score_predictions(subset, predictions, split="validation")
        if split == "validation" else None
    )
    summary = _summary(
        dataset_path=dataset_path,
        split=split,
        results=results,
        failures=failures,
        before_freeze=before_freeze,
        after_freeze=after_freeze,
        score=score,
        run_fingerprint=run_fingerprint,
        benchmark_compatibility=benchmark_compatibility,
    )
    _write(output / "predictions.json", predictions)
    if score is not None:
        _write(output / "score.json", score)
    _write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


def _request(case: dict[str, Any]) -> dict[str, Any]:
    routing_context = {
        "benchmark_case_id": str(case.get("case_id") or ""),
        "capability_layer": str(case.get("capability_layer") or ""),
        "scenario_type": str(case.get("scenario_type") or ""),
    }
    context_ref = dict(case.get("input_context_ref") or {})
    if context_ref:
        context_path = (REPO_ROOT / str(context_ref.get("path") or "")).resolve()
        if REPO_ROOT.resolve() not in context_path.parents or not context_path.is_file():
            raise ValueError(f"invalid_input_context_ref:{case.get('case_id')}")
        actual_sha = hashlib.sha256(context_path.read_bytes()).hexdigest()
        if actual_sha != str(context_ref.get("sha256") or ""):
            raise ValueError(f"input_context_sha256_mismatch:{case.get('case_id')}")
        context = _load(context_path)
        if context.get("label_visibility") not in {
            "source_records_only", "source_only_no_ground_truth",
        }:
            raise ValueError(f"input_context_not_source_only:{case.get('case_id')}")
        routing_context["source_only_context_ref"] = context_ref
        routing_context["source_only_context"] = context
    return {
        "query": str(case.get("query") or ""),
        "interactive": False,
        "chat_history": list(case.get("turns") or []),
        "routing_context": routing_context,
    }


def _summary(
    *, dataset_path: Path, split: str, results: list[dict[str, Any]],
    failures: list[dict[str, str]], before_freeze: dict[str, Any],
    after_freeze: dict[str, Any], score: dict[str, Any] | None,
    run_fingerprint: str, benchmark_compatibility: dict[str, Any],
) -> dict[str, Any]:
    structural = [item for result in results for item in result.get("structural_errors") or []]
    elapsed = [float(item.get("elapsed_ms") or 0.0) for item in results]
    official_parity: list[bool] = []
    proposed_answer_changes: list[bool] = []
    proposed_status_changes: list[bool] = []
    for item in results:
        response = dict(item.get("response") or {})
        baseline = dict(response.get("baseline_response") or {})
        shadow = dict(response.get("shadow") or {})
        official_parity.append(
            response.get("answer") == baseline.get("answer")
            and response.get("status") == baseline.get("status")
        )
        proposed_answer_changes.append(bool(shadow.get("answer_changed")))
        proposed_status_changes.append(bool(shadow.get("status_changed")))
    return {
        "schema_version": "debug_agent_system.read_runtime_v3_benchmark_summary.v1",
        "dataset": _relative(dataset_path),
        "split": split,
        "run_fingerprint": run_fingerprint,
        "passed": bool(
            results and not failures and not structural and before_freeze.get("frozen")
            and after_freeze.get("frozen")
        ),
        "requested_case_count": len(results) + len(failures),
        "completed_case_count": len(results),
        "failure_count": len(failures),
        "contract_failure_count": sum(bool(item.get("structural_errors")) for item in results),
        "official_answer_and_status_parity_rate": (
            sum(official_parity) / len(official_parity) if official_parity else 0.0
        ),
        "proposed_answer_change_rate": (
            sum(proposed_answer_changes) / len(proposed_answer_changes)
            if proposed_answer_changes else 0.0
        ),
        "proposed_status_change_rate": (
            sum(proposed_status_changes) / len(proposed_status_changes)
            if proposed_status_changes else 0.0
        ),
        "verification_pass_rate": (
            sum(bool((item.get("response") or {}).get("verification", {}).get("passed")) for item in results) / len(results)
            if results else 0.0
        ),
        "latency_ms": {
            "minimum": min(elapsed) if elapsed else 0.0,
            "mean": sum(elapsed) / len(elapsed) if elapsed else 0.0,
            "maximum": max(elapsed) if elapsed else 0.0,
        },
        "structural_errors": sorted(set(structural)),
        "failures": failures,
        "freeze_before": before_freeze,
        "freeze_after": after_freeze,
        "benchmark_compatibility": benchmark_compatibility,
        "formal_score_interpretation": (
            "comparable"
            if benchmark_compatibility.get("status") == "compatible"
            else "raw_score_contains_stale_source_id_cases"
        ),
        "formal_score": score,
    }


def _benchmark_compatibility(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit stale benchmark IDs after execution, never as runtime input."""

    current_ids: set[str] = set()
    for path in sorted((REPO_ROOT / "data/kg_v2/objects").glob("*.json")):
        _collect_canonical_ids(_load_any(path), current_ids)

    stale_cases: list[dict[str, Any]] = []
    for case in cases:
        expected = dict(case.get("expected_route") or {})
        required = [str(value) for value in expected.get("required_target_ids") or []]
        canonical = [value for value in required if _looks_like_canonical_kg_id(value)]
        missing = [value for value in canonical if value not in current_ids]
        if not missing:
            continue
        raw_source_refs = (case.get("source") or {}).get("source_refs") or {}
        source_refs = dict(raw_source_refs) if isinstance(raw_source_refs, dict) else {}
        source_path = str(source_refs.get("document_path") or "")
        expected_hash = str(
            source_refs.get("document_content_hash")
            or source_refs.get("document_sha256")
            or ""
        )
        actual_hash = ""
        if source_path:
            candidate = (REPO_ROOT / source_path).resolve()
            if REPO_ROOT.resolve() in candidate.parents and candidate.is_file():
                actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        stale_cases.append({
            "case_id": str(case.get("case_id") or ""),
            "missing_current_ids": missing,
            "source_path": source_path,
            "expected_source_sha256": expected_hash,
            "current_source_sha256": actual_hash,
            "source_content_changed": bool(
                expected_hash and actual_hash and expected_hash != actual_hash
            ),
        })
    return {
        "status": "compatible" if not stale_cases else "stale_source_ids",
        "checked_case_count": len(cases),
        "stale_case_count": len(stale_cases),
        "stale_cases": stale_cases,
        "runtime_prediction_influence": "none_post_score_audit_only",
    }


def _collect_canonical_ids(value: Any, target: set[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _collect_canonical_ids(item, target)
    elif isinstance(value, list):
        for item in value:
            _collect_canonical_ids(item, target)
    elif isinstance(value, str) and ":" in value:
        target.add(value)


def _looks_like_canonical_kg_id(value: str) -> bool:
    """Exclude evaluator-local opaque labels such as ``011-a``.

    The compatibility audit is only about content-addressed/canonical KG ids;
    source-only human Trace labels are not KG objects and must not be reported
    as stale merely because they are absent from ``data/kg_v2/objects``.
    """

    return ":" in str(value or "")


def _verify_freeze() -> dict[str, Any]:
    from scripts.verify_frozen_read_pipeline import verify
    return verify(FROZEN_MANIFEST)


def _single_split(dataset: dict[str, Any]) -> str:
    values = {str(case.get("split") or "") for case in dataset.get("cases") or []}
    return next(iter(values)) if len(values) == 1 else ""


def _run_fingerprint(*, dataset_path: Path, v3_config: Path, selected: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for path in (FROZEN_MANIFEST, v3_config, Path(__file__).resolve()):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for path in sorted((REPO_ROOT / "src/debug_agent_system/read_runtime_v3").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(dataset_path.relative_to(REPO_ROOT).as_posix().encode())
    for case in selected:
        digest.update(str(case.get("case_id") or "").encode())
        digest.update(_sha_text(str(case.get("query") or "")).encode())
    return digest.hexdigest()


def _reusable(path: Path, *, case: dict[str, Any], run_fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        value.get("case_id") == case.get("case_id")
        and value.get("query_sha256") == _sha_text(str(case.get("query") or ""))
        and value.get("run_fingerprint") == run_fingerprint
        and value.get("execution_status") == "passed"
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _load_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
