#!/usr/bin/env python3
"""Build a benchmark by asking the model to read one chunk per iteration."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

from debug_agent_system.eval.read_side.chunk_qa_generator import (
    GENERATOR_VERSION,
    build_case_record,
    chunk_sha256,
    generate_one_chunk,
    make_cli_client,
    make_client,
    similarity,
    validate_model_payload,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/raw/aoi_debug_agent_sources/chunks/debug_chunks.json"
DEFAULT_OUT = ROOT / "tests/feature_selftest_queries_from_raw_and_fae.jsonl"
DEFAULT_MD = ROOT / "tests/feature_selftest_queries_from_raw_and_fae.md"
DEFAULT_REPORT = ROOT / "tests/feature_selftest_queries_from_raw_and_fae_report.json"
DEFAULT_AUDIT = ROOT / "tests/feature_selftest_queries_from_raw_and_fae.audit.jsonl"
DEFAULT_HISTORICAL_QUERY_PATHS = (
    ROOT / "data/eval/benchmark/aoi_debug_benchmark_v1.json",
    ROOT / "data/eval/benchmark/aoi_fae_report_benchmark_v2.json",
    ROOT / "data/eval/scenarios/kg_v2_quality_v1.json",
)


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("chunks_root_must_be_array")
    return [item for item in payload if isinstance(item, dict)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_historical_queries(paths: tuple[Path, ...]) -> list[str]:
    queries: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        containers = []
        if isinstance(payload, dict):
            for key in ("cases", "records", "scenarios"):
                value = payload.get(key)
                if isinstance(value, list):
                    containers.extend(value)
        elif isinstance(payload, list):
            containers.extend(payload)
        for row in containers:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            if query:
                queries.append(query)
    return list(dict.fromkeys(queries))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_body(value: Any) -> str:
    text = str(value or "")
    return text.split("\n\n", 1)[-1].strip()


def _dedupe_rows_by_source_body(
    rows: list[dict[str, Any]],
    *,
    similarity_limit: float = 0.90,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    bodies: list[str] = []
    for row in rows:
        body = _source_body((row.get("source") or {}).get("chunk_text"))
        if body and any(similarity(body, prior) >= similarity_limit for prior in bodies):
            continue
        kept.append(row)
        bodies.append(body)
    for index, row in enumerate(kept, start=1):
        row["case_id"] = f"chunk-qa-{index:04d}"
    return kept, len(rows) - len(kept)


def _dedupe_rows_against_queries(
    rows: list[dict[str, Any]],
    historical_queries: list[str],
    *,
    similarity_limit: float = 0.84,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    query_pool = list(historical_queries)
    for row in rows:
        query = str(row.get("query") or "")
        if any(similarity(query, prior) >= similarity_limit for prior in query_pool):
            continue
        kept.append(row)
        query_pool.append(query)
    for index, row in enumerate(kept, start=1):
        row["case_id"] = f"chunk-qa-{index:04d}"
    return kept, len(rows) - len(kept)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Feature Selftest：逐 Chunk 阅读生成的 Query-Answer",
        "",
        f"共 {len(rows)} 条。每条由模型独立阅读一个 chunk 后生成，并经过逐字证据校验。",
        "",
    ]
    for row in rows:
        source = row["source"]
        gold = row["answer_gold"]
        lines.extend([
            f"## {row['case_id']} · {source.get('title') or '无标题'}",
            "",
            f"- 来源：`{source.get('origin')}`",
            f"- Chunk index：`{source.get('chunk_index')}`",
            f"- Section：`{source.get('section_num')}`",
            "",
            "**Query**",
            "",
            row["query"],
            "",
            "**参考答案**",
            "",
            gold["reference_answer"],
            "",
            "**原文证据**",
            "",
        ])
        for excerpt in gold["evidence_excerpts"]:
            lines.append(f"> {str(excerpt).replace(chr(10), chr(10) + '> ')}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _report(
    rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    model: str,
    historical_query_count: int,
) -> dict[str, Any]:
    origins = Counter(str(row["source"].get("origin") or "") for row in rows)
    return {
        "schema_version": "debug_agent_system.chunk_qa_benchmark_report.v1",
        "generator_version": GENERATOR_VERSION,
        "model": model,
        "historical_query_reuse": {
            "mode": "dedupe_and_coverage_reference_only",
            "query_count": historical_query_count,
            "sources": [
                path.relative_to(ROOT).as_posix()
                for path in DEFAULT_HISTORICAL_QUERY_PATHS
            ],
        },
        "accepted": len(rows),
        "chunks_decided": sum(
            1 for row in audit_rows if row.get("decision") in {"accept", "reject"}
        ),
        "audit_rows": len(audit_rows),
        "rejected": sum(1 for row in audit_rows if row.get("decision") == "reject"),
        "transient_errors": sum(
            1 for row in audit_rows if row.get("decision") == "error"
        ),
        "origin_counts": dict(sorted(origins.items())),
        "total_attempts": sum(len(row.get("attempts") or []) for row in audit_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read each chunk with Codex and build grounded QA pairs",
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument(
        "--runtime",
        choices=("responses_api", "codex_cli"),
        default="responses_api",
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--review-model", default="")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--candidate-order",
        choices=("source_richness", "input"),
        default="source_richness",
        help=(
            "Keep SOP/FAQ source order, then review richer tech-support chunks "
            "first; use input to preserve raw file order"
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent one-chunk model calls to run concurrently",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = _load_chunks(args.chunks)
    historical_queries = _load_historical_queries(DEFAULT_HISTORICAL_QUERY_PATHS)
    rows = _load_jsonl(args.out) if args.resume else []
    if args.resume and rows:
        rows, removed_duplicates = _dedupe_rows_by_source_body(rows)
        rows, removed_history_duplicates = _dedupe_rows_against_queries(
            rows,
            historical_queries,
        )
        removed_duplicates += removed_history_duplicates
        if removed_duplicates:
            _write_jsonl(args.out, rows)
            print(
                f"resume_dedupe_removed:{removed_duplicates}",
                file=sys.stderr,
                flush=True,
            )
    audit_rows = _load_jsonl(args.audit) if args.resume else []
    if not args.resume:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        args.audit.write_text("", encoding="utf-8")

    processed_hashes = {
        str(row.get("chunk_sha256") or "")
        for row in audit_rows
        if row.get("decision") in {"accept", "reject"}
    }
    existing_queries = historical_queries + [str(row.get("query") or "") for row in rows]
    processed_this_run = 0
    circuit_breaker_reason = ""

    candidates = [
        (chunk_index, chunk)
        for chunk_index, chunk in enumerate(chunks)
        if chunk_index >= args.start_index
        and chunk_sha256(chunk) not in processed_hashes
    ]
    if args.candidate_order == "source_richness":
        source_rank = {"SOP": 0, "FAQ": 1, "tech_support": 2}

        def candidate_key(item: tuple[int, dict[str, Any]]):
            chunk_index, chunk = item
            metadata = chunk.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            source = str(metadata.get("source") or "")
            rank = source_rank.get(source, 3)
            richness = len(str(chunk.get("text") or ""))
            return (
                rank,
                chunk_index if rank < 2 else -richness,
                chunk_index,
            )

        candidates.sort(key=candidate_key)
    workers = max(1, int(args.workers))

    def run_one(item: tuple[int, dict[str, Any]], query_snapshot: list[str]):
        chunk_index, chunk = item
        client = (
            make_cli_client(timeout_seconds=args.timeout_seconds)
            if args.runtime == "codex_cli"
            else make_client(
                env_file=args.env_file,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return generate_one_chunk(
            client,
            model=args.model,
            chunk=chunk,
            chunk_index=chunk_index,
            existing_queries=query_snapshot,
            reasoning_effort=args.reasoning_effort,
            max_attempts=args.max_attempts,
            review=not args.skip_review,
            review_model=args.review_model or None,
        )

    cursor = 0
    while cursor < len(candidates) and len(rows) < args.count:
        remaining = (
            args.max_chunks - processed_this_run
            if args.max_chunks
            else workers
        )
        if remaining <= 0:
            break
        batch_size = min(workers, remaining, len(candidates) - cursor)
        batch = candidates[cursor : cursor + batch_size]
        cursor += batch_size
        query_snapshot = list(existing_queries)

        def guarded(item: tuple[int, dict[str, Any]]):
            try:
                return item, run_one(item, query_snapshot), ""
            except Exception as exc:  # keep long benchmark runs resumable
                return item, None, f"{type(exc).__name__}:{exc}"

        if workers == 1:
            results = [guarded(batch[0])]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(guarded, batch))

        batch_error_count = 0
        for (chunk_index, chunk), generated, error_text in results:
            processed_this_run += 1
            digest = chunk_sha256(chunk)
            if generated is None:
                batch_error_count += 1
                audit_row = {
                    "generator_version": GENERATOR_VERSION,
                    "model": args.model,
                    "chunk_index": chunk_index,
                    "chunk_sha256": digest,
                    "attempts": [],
                    "decision": "error",
                    "rejection_reason": error_text[:300],
                }
                decision = "error"
            else:
                decision = str(generated.payload.get("decision") or "reject")
                # Calls in one parallel batch share a query snapshot. Recheck
                # accepted cases serially so cross-batch duplicates cannot pass.
                if decision == "accept":
                    duplicate_errors = validate_model_payload(
                        generated.payload,
                        chunk=chunk,
                        existing_queries=existing_queries,
                    )
                    if duplicate_errors:
                        decision = "reject"
                        generated.payload["decision"] = "reject"
                        generated.payload["rejection_reason"] = (
                            "post_batch_validation:" + ",".join(duplicate_errors)
                        )
                    else:
                        body = _source_body(chunk.get("text"))
                        duplicate_source = next(
                            (
                                row["case_id"]
                                for row in rows
                                if similarity(
                                    body,
                                    _source_body(row["source"].get("chunk_text")),
                                ) >= 0.90
                            ),
                            "",
                        )
                        if duplicate_source:
                            decision = "reject"
                            generated.payload["decision"] = "reject"
                            generated.payload["rejection_reason"] = (
                                "near_duplicate_source_body:" + duplicate_source
                            )
                audit_row = {
                    **generated.audit,
                    "decision": decision,
                    "rejection_reason": generated.payload.get("rejection_reason"),
                }

            _append_jsonl(args.audit, audit_row)
            audit_rows.append(audit_row)
            if decision in {"accept", "reject"}:
                processed_hashes.add(digest)

            if decision == "accept" and generated is not None and len(rows) < args.count:
                row = build_case_record(
                    generated,
                    chunk=chunk,
                    case_number=len(rows) + 1,
                )
                _append_jsonl(args.out, row)
                rows.append(row)
                existing_queries.append(row["query"])

            print(
                json.dumps({
                    "chunk_index": chunk_index,
                    "decision": decision,
                    "accepted": len(rows),
                    "processed": len(audit_rows),
                }, ensure_ascii=False),
                flush=True,
            )

        if batch_error_count >= max(2, (batch_size + 1) // 2):
            circuit_breaker_reason = (
                f"batch_error_circuit_breaker:{batch_error_count}/{batch_size}"
            )
            print(circuit_breaker_reason, file=sys.stderr, flush=True)
            break

    args.markdown.write_text(_render_markdown(rows), encoding="utf-8")
    args.report.write_text(
        json.dumps(
            _report(
                rows,
                audit_rows,
                model=args.model,
                historical_query_count=len(historical_queries),
            ),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if len(rows) < args.count:
        print(
            f"incomplete: accepted {len(rows)} of requested {args.count}; "
            "rerun with --resume to continue"
            + (f" ({circuit_breaker_reason})" if circuit_breaker_reason else ""),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
