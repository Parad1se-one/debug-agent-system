"""Run the independent KG_v2+raw Codex pipeline over a saved query set.

Every query has an independent answer artifact so the batch is resumable and
one failed query does not discard successful results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from debug_agent_system.kg_raw_codex.pipeline import run
from debug_agent_system.kg_raw_codex.prompt import SYSTEM_PROMPT_VERSION


DEFAULT_INPUT = (
    REPO_ROOT
    / "data/results/read_side_codex_comparison_20260730"
    / "comparison_results.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data/results/read_side_codex_comparison_20260730"
    / "kg_v2_raw_codex_answers"
)


def _queries(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("input_json_requires_records")
    queries = [
        str(item.get("query") or "").strip()
        for item in records
        if isinstance(item, dict) and str(item.get("query") or "").strip()
    ]
    if not queries:
        raise ValueError("input_json_has_no_queries")
    return list(dict.fromkeys(queries))


def _artifact_path(output_dir: Path, index: int, query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return output_dir / f"{index:02d}_{digest}.json"


def _failure_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failure.json")


def _reusable(path: Path, query: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    verification = (
        payload.get("verification") if isinstance(payload, dict) else None
    )
    return bool(
        payload.get("query") == query
        and isinstance(prompt, dict)
        and prompt.get("system_version") == SYSTEM_PROMPT_VERSION
        and isinstance(verification, dict)
        and verification.get("passed") is True
        and str(payload.get("answer") or "").strip()
    )


def _is_transient_transport_error(exc: Exception) -> bool:
    """Classify retryable gateway/network failures without coupling to a client."""
    name = type(exc).__name__
    detail = str(exc).lower()
    if name in {"CodexResponsesAgentError", "CodexCliAgentError"} and any(
        marker in detail
        for marker in (
            "codex_responses_transport",
            "codex_responses_http_408",
            "codex_responses_http_429",
            "codex_responses_http_500",
            "codex_responses_http_502",
            "codex_responses_http_503",
            "codex_responses_http_504",
            "connection reset",
            "stream disconnected before completion",
            "error sending request",
            "temporarily unavailable",
            "timed out",
        )
    ):
        return True
    if name in {"RemoteDisconnected", "ConnectionResetError", "TimeoutError"}:
        return True
    return any(
        marker in detail
        for marker in (
            "remote end closed connection",
            "connection reset",
            "temporarily unavailable",
            "timed out",
        )
    )


def _run_one(
    index: int,
    query: str,
    output: Path,
    retries: int,
    runtime: str | None,
    model: str | None,
) -> dict[str, Any]:
    failure: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            payload = run(
                query,
                output,
                runtime=runtime,
                model=model,
            )
            break
        except Exception as exc:  # noqa: BLE001 - record per-query failure.
            failure = exc
            transient = _is_transient_transport_error(exc)
            if not transient or attempt >= max(0, retries):
                result = {
                    "index": index,
                    "query": query,
                    "status": "failed",
                    "artifact": output.relative_to(REPO_ROOT).as_posix(),
                    "attempts": attempt + 1,
                    "error": f"{type(exc).__name__}:{exc}",
                }
                _failure_path(output).write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return result
            time.sleep(min(2 ** attempt, 8))
    else:  # pragma: no cover - loop always returns or breaks.
        raise AssertionError(failure)
    _failure_path(output).unlink(missing_ok=True)
    return {
        "index": index,
        "query": query,
        "status": "passed",
        "artifact": output.relative_to(REPO_ROOT).as_posix(),
        "attempts": attempt + 1,
        "answer_length": len(str(payload.get("answer") or "")),
        "files_read": len(payload.get("files_read") or []),
        "media_exposed": len(payload.get("media_exposed") or []),
        "tool_calls": len(payload.get("tool_trace") or []),
        "usage": dict(payload.get("usage") or {}),
    }


def run_batch(
    *,
    input_path: Path,
    output_dir: Path,
    start_index: int = 1,
    limit: int | None = None,
    workers: int = 1,
    resume: bool = True,
    retries: int = 2,
    runtime: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    queries = _queries(input_path)
    selected = [
        (index, query)
        for index, query in enumerate(queries, start=1)
        if index >= max(1, start_index)
    ]
    if limit is not None:
        selected = selected[: max(0, limit)]
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    pending: list[tuple[int, str, Path]] = []
    for index, query in selected:
        output = _artifact_path(output_dir, index, query)
        if resume and _reusable(output, query):
            payload = json.loads(output.read_text(encoding="utf-8"))
            result = {
                "index": index,
                "query": query,
                "status": "reused",
                "artifact": output.relative_to(REPO_ROOT).as_posix(),
                "answer_length": len(str(payload.get("answer") or "")),
                "files_read": len(payload.get("files_read") or []),
                "media_exposed": len(payload.get("media_exposed") or []),
                "tool_calls": len(payload.get("tool_trace") or []),
                "usage": dict(payload.get("usage") or {}),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        else:
            pending.append((index, query, output))

    if workers <= 1:
        for index, query, output in pending:
            result = _run_one(
                index, query, output, retries, runtime, model
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
            futures: dict[Future[dict[str, Any]], int] = {
                pool.submit(
                    _run_one,
                    index,
                    query,
                    output,
                    retries,
                    runtime,
                    model,
                ): index
                for index, query, output in pending
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    results.sort(key=lambda item: int(item["index"]))
    summary = {
        "schema_version": "debug_agent_system.kg_raw_codex_batch.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "runtime": runtime or "configured_default",
        "model": model or "configured_default",
        "input": input_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "selected_count": len(selected),
        "passed": sum(item["status"] in {"passed", "reused"} for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "total_usage": {
            key: sum(
                int((item.get("usage") or {}).get(key) or 0)
                for item in results
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "results": results,
    }
    manifest = output_dir / "batch_manifest.json"
    manifest.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--runtime",
        choices=("responses_api", "codex_cli"),
        help="Override the configured transport.",
    )
    parser.add_argument("--model", help="Override the selected model.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun even when a verified artifact uses the current prompt",
    )
    args = parser.parse_args()
    summary = run_batch(
        input_path=args.input,
        output_dir=args.output_dir,
        start_index=args.start_index,
        limit=args.limit,
        workers=args.workers,
        resume=not args.no_resume,
        retries=args.retries,
        runtime=args.runtime,
        model=args.model,
    )
    print(json.dumps(
        {
            "selected": summary["selected_count"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "total_usage": summary["total_usage"],
            "manifest": str(args.output_dir / "batch_manifest.json"),
        },
        ensure_ascii=False,
    ))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
