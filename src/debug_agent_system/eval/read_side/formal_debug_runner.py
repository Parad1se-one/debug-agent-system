"""Execute and score the formal Debug Benchmark with one pinned model run.

The model sees only benchmark inputs, never ``*_gold`` fields.  It investigates
the read-only raw/KG corpus and emits the common prediction schema directly;
the deterministic scorer then reports each capability layer separately.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.adapters.codex_read.client import (
    CodexResponsesClient,
    _read_local_env,
)
from debug_agent_system.eval.read_side.formal_debug_benchmark import (
    CORE_PATH,
    PREDICTION_TEMPLATE_PATH,
    ROOT,
    SCORE_PATH,
    _load,
    _write_json,
    score_predictions,
)
from debug_agent_system.kg_raw_codex.pipeline import (
    AgentRunner,
    CodexCliAgentRunner,
    CodexResponsesAgentRunner,
    prepared_corpus_workspace,
)


MODEL = "gpt-5.6-luna"
RUN_ROOT = ROOT / "data/results/formal_debug_benchmark_v1/latest_run"
PREDICTIONS_PATH = RUN_ROOT / "predictions.json"
EXECUTION_CONTRACT = """You are running one AOI Debug Benchmark case.
Investigate only the supplied query/turns and the read-only corpus. Never infer
benchmark Gold or inspect evaluation files. Return both a concise grounded
answer and the structured IDs you actually support. Use empty strings/lists
when the evidence is insufficient. Do not claim an action was executed. Use
status=resolved only when the input contains an explicit verified-fix outcome.
For evidence_ids and route_ids, copy canonical IDs exactly from source files.
"""
EXECUTION_PROMPT_SHA256 = hashlib.sha256(
    EXECUTION_CONTRACT.encode("utf-8")
).hexdigest()


def _prediction_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "answer": {"type": "string"},
        "route_type": {
            "type": "string",
            "enum": ["knowledge_document_section", "sag_v2_native", "source_only_trace_reconstruction", "out_of_domain", ""],
        },
        "route_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "family_id": {"type": "string"},
        "variant_id": {"type": "string"},
        "first_action_id": {"type": "string"},
        "followup_ids": {"type": "array", "items": {"type": "string"}},
        "status": {
            "type": "string",
            "enum": ["answer", "step", "ask_info", "resolved", "escalate", "unsupported"],
        },
        "executed_action_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "trace_count": {"type": "integer", "minimum": 0},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _prompt(case: dict[str, Any]) -> str:
    visible = {
        "case_id": case["case_id"],
        "capability_layer": case["capability_layer"],
        "scenario_type": case["scenario_type"],
        "query": case["query"],
        "turns": case.get("turns") or [],
    }
    context_ref = case.get("input_context_ref") or {}
    if context_ref:
        context_path = (ROOT / str(context_ref.get("path") or "")).resolve()
        if ROOT.resolve() not in context_path.parents or not context_path.is_file():
            raise ValueError(f"invalid_input_context_ref:{case['case_id']}")
        actual_sha = hashlib.sha256(context_path.read_bytes()).hexdigest()
        if actual_sha != context_ref.get("sha256"):
            raise ValueError(f"input_context_sha256_mismatch:{case['case_id']}")
        context = _load(context_path)
        if context.get("label_visibility") not in {
            "source_records_only", "source_only_no_ground_truth",
        }:
            raise ValueError(f"input_context_not_source_only:{case['case_id']}")
        visible["source_only_context"] = context
    return (
        EXECUTION_CONTRACT
        + "\n\nBENCHMARK INPUT (this contains no Gold):\n"
        + json.dumps(visible, ensure_ascii=False, indent=2)
    )


def _artifact_path(run_root: Path, case_id: str) -> Path:
    return run_root / "cases" / f"{case_id}.json"


def _reusable(path: Path, *, case: dict[str, Any], model: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("case_id") == case["case_id"]
        and payload.get("query_sha256")
        == hashlib.sha256(case["query"].encode("utf-8")).hexdigest()
        and payload.get("model") == model
        and payload.get("execution_prompt_sha256") == EXECUTION_PROMPT_SHA256
        and isinstance(payload.get("prediction"), dict)
    )


def _execute_one(
    *,
    case: dict[str, Any],
    runner: AgentRunner,
    workspace: Path,
    run_root: Path,
    resume: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    artifact = _artifact_path(run_root, case["case_id"])
    if resume and _reusable(artifact, case=case, model=runner.model):
        payload = _load(artifact)
        return {**payload, "execution_status": "reused"}
    prediction, audit = runner.run(
        prompt=_prompt(case),
        workspace=workspace,
        output_schema=_prediction_schema(),
        timeout_seconds=timeout_seconds,
    )
    prediction["case_id"] = case["case_id"]
    payload = {
        "schema_version": "debug_agent_system.formal_debug_case_run.v1",
        "case_id": case["case_id"],
        "query_sha256": hashlib.sha256(case["query"].encode("utf-8")).hexdigest(),
        "model": runner.model,
        "execution_prompt_sha256": EXECUTION_PROMPT_SHA256,
        "execution_status": "passed",
        "prediction": prediction,
        "audit": audit,
    }
    _write_json(artifact, payload)
    return payload


def execute_dataset(
    dataset: dict[str, Any],
    *,
    runner: AgentRunner,
    workspace: Path,
    run_root: Path,
    split: str = "validation",
    allow_held_out_test: bool = False,
    workers: int = 1,
    resume: bool = True,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    if split == "held_out_test" and not allow_held_out_test:
        raise ValueError("held_out_test_requires_explicit_allow_flag")
    cases = [
        case for case in dataset["cases"]
        if split == "all" or case["split"] == split
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def run_case(case: dict[str, Any]) -> dict[str, Any]:
        return _execute_one(
            case=case,
            runner=runner,
            workspace=workspace,
            run_root=run_root,
            resume=resume,
            timeout_seconds=timeout_seconds,
        )

    if workers <= 1:
        for case in cases:
            try:
                results.append(run_case(case))
            except Exception as exc:  # noqa: BLE001 - preserve per-case failure.
                failures.append({
                    "case_id": case["case_id"],
                    "error": f"{type(exc).__name__}:{exc}",
                })
    else:
        with ThreadPoolExecutor(max_workers=min(int(workers), 4)) as pool:
            future_cases = {pool.submit(run_case, case): case for case in cases}
            for future in as_completed(future_cases):
                case = future_cases[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append({
                        "case_id": case["case_id"],
                        "error": f"{type(exc).__name__}:{exc}",
                    })

    results.sort(key=lambda item: item["case_id"])
    by_id = {item["case_id"]: item["prediction"] for item in results}
    predictions = {
        "schema_version": "debug_agent_system.formal_debug_predictions.v1",
        "run_manifest": {
            **dataset["version_manifest"],
            "run_id": run_root.name,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "model": runner.model,
            "execution_prompt_sha256": EXECUTION_PROMPT_SHA256,
            "execution_runtime": runner.runtime_metadata,
            "split": split,
            "requested_case_count": len(cases),
            "completed_case_count": len(results),
            "failed_case_count": len(failures),
        },
        "predictions": [by_id[case["case_id"]] for case in cases if case["case_id"] in by_id],
        "failures": failures,
    }
    _write_json(run_root / "predictions.json", predictions)
    return predictions


def _runner(runtime: str, model: str, reasoning_effort: str) -> AgentRunner:
    config = _load(ROOT / "config/kg_v2_raw_codex.json")
    if runtime == "codex_cli":
        return CodexCliAgentRunner(
            model=model,
            reasoning_effort=reasoning_effort,
            codex_binary=str(config.get("codex_binary") or "codex"),
        )
    local_env = _read_local_env(ROOT / ".env.local")
    return CodexResponsesAgentRunner(
        client=CodexResponsesClient(
            api_key=local_env.get("OPENAI_API_KEY", ""),
            base_url=local_env.get("OPENAI_BASE_URL", ""),
            timeout_seconds=int(config.get("timeout_seconds") or 600),
        ),
        model=model,
        reasoning_effort=reasoning_effort,
        max_tool_rounds=int(config.get("max_tool_rounds") or 24),
        max_tool_calls=int(config.get("max_tool_calls") or 80),
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="run-formal-debug-benchmark")
    parser.add_argument("--dataset", type=Path, default=CORE_PATH)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--score-out", type=Path, default=SCORE_PATH)
    parser.add_argument("--split", choices=("validation", "held_out_test", "all"), default="validation")
    parser.add_argument("--allow-held-out-test", action="store_true")
    parser.add_argument("--runtime", choices=("codex_cli", "responses_api"), default="codex_cli")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    dataset = _load(args.dataset)
    runner = _runner(args.runtime, args.model, args.reasoning_effort)
    with prepared_corpus_workspace(
        asset_root=args.run_root / "corpus_assets"
    ) as corpus:
        predictions = execute_dataset(
            dataset,
            runner=runner,
            workspace=corpus.root,
            run_root=args.run_root,
            split=args.split,
            allow_held_out_test=args.allow_held_out_test,
            workers=args.workers,
            resume=not args.no_resume,
            timeout_seconds=args.timeout_seconds,
        )
    score = score_predictions(
        dataset,
        predictions,
        split=args.split,
        allow_held_out_test=args.allow_held_out_test,
    )
    _write_json(args.score_out, score)
    print(json.dumps({
        "status": "passed" if not predictions["failures"] else "partial",
        "completed": predictions["run_manifest"]["completed_case_count"],
        "failed": predictions["run_manifest"]["failed_case_count"],
        "predictions": (args.run_root / "predictions.json").as_posix(),
        "score": args.score_out.as_posix(),
    }, ensure_ascii=False))
    return 0 if not predictions["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
