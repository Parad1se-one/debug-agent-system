"""Rewrite selected FAE source inputs into natural benchmark queries.

Only ``source_input`` is exposed to the model. Follow-up evidence and
``answer_gold`` are deliberately excluded so the rewrite cannot leak answers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from debug_agent_system.kg_raw_codex.pipeline import CodexCliAgentRunner


FAE_POOL = ROOT / "data/eval/benchmark/aoi_fae_report_benchmark_v2.json"
BENCHMARK_ROOT = ROOT / "data/eval/formal_debug_benchmark_v1"
SHARDS = (
    BENCHMARK_ROOT / "feature_selftest_queries_kg_runtime.jsonl",
    BENCHMARK_ROOT / "feature_selftest_queries_fae.jsonl",
    BENCHMARK_ROOT / "feature_selftest_queries_document_qa.jsonl",
)
OUTPUT = BENCHMARK_ROOT / "fae_query_rewrites.json"
MODEL = "gpt-5.6-luna"
PROMPT_CONTRACT = """将每条 AOI 现场 source_input 改写成工程师会直接问 Debug Agent 的自然 Query。
硬性要求：
1. 只能使用 source_input 中当时已知的事实，不得读取或推测后续消息、答案、根因或修复结果。
2. 删除“真实 FAE 现场报告”“现场原文”“任务：”等数据集标签，也删除寒暄、@人名、领导称呼和表情。
3. 不要整段复制日报。保留诊断所需的设备/软件版本、时间、错误码、症状、已做动作及其结果。
4. 单故障改写成一个自然排障问题；原文确有多个并行故障时，明确要求先拆分问题再给低风险首步。
5. 不得把建议写成已执行，不得把短暂恢复写成已解决，不得补造原文没有的日志或参数。
6. Query 通常 40～260 个汉字，复杂多故障最多 420 字；不要加标题或解释，只输出 Query。
"""
PROMPT_SHA256 = hashlib.sha256(PROMPT_CONTRACT.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def selected_case_ids() -> list[str]:
    ids: list[str] = []
    for path in SHARDS:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            origin = str(row.get("origin") or "")
            marker = "formal_debug_benchmark_v1:real_fae_candidates:"
            if origin.startswith(marker):
                ids.append(origin.removeprefix(marker))
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError(f"expected_64_unique_fae_cases:{len(ids)}:{len(set(ids))}")
    return ids


def source_sha256(case: dict[str, Any]) -> str:
    payload = json.dumps(
        case.get("source_input") or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rewrites": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["case_id", "query"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rewrites"],
        "additionalProperties": False,
    }


def validate_query(query: str, *, source_text: str) -> None:
    banned = ("【真实 FAE 现场报告】", "真实 FAE 现场报告", "现场原文：", "任务：")
    if any(value in query for value in banned):
        raise ValueError("rewrite_contains_dataset_label")
    if not 20 <= len(query) <= 500:
        raise ValueError(f"rewrite_length:{len(query)}")
    if len(source_text) >= 140 and query.strip() == source_text.strip():
        raise ValueError("rewrite_copies_full_source")


def write_output(records: dict[str, dict[str, Any]]) -> None:
    ordered = [records[key] for key in sorted(records)]
    payload = {
        "schema_version": "debug_agent_system.formal_fae_query_rewrites.v1",
        "model": MODEL,
        "prompt_sha256": PROMPT_SHA256,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(ordered),
        "records": ordered,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    pool = {case["case_id"]: case for case in load_json(FAE_POOL)["cases"]}
    selected = [pool[case_id] for case_id in selected_case_ids()]
    existing: dict[str, dict[str, Any]] = {}
    if OUTPUT.is_file() and not args.no_resume:
        prior = load_json(OUTPUT)
        if prior.get("model") == args.model and prior.get("prompt_sha256") == PROMPT_SHA256:
            for record in prior.get("records") or []:
                case_id = str(record.get("case_id") or "")
                case = pool.get(case_id)
                if case and record.get("source_input_sha256") == source_sha256(case):
                    existing[case_id] = record

    pending = [case for case in selected if case["case_id"] not in existing]
    runner = CodexCliAgentRunner(model=args.model, reasoning_effort="medium")
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        inputs = [
            {
                "case_id": case["case_id"],
                "source_input": case.get("source_input") or {},
            }
            for case in batch
        ]
        prompt = (
            PROMPT_CONTRACT
            + "\n请逐条返回，与输入 case_id 一一对应。\n\nINPUTS:\n"
            + json.dumps(inputs, ensure_ascii=False, indent=2)
        )
        result, audit = runner.run(
            prompt=prompt,
            workspace=ROOT,
            output_schema=output_schema(),
            timeout_seconds=900,
        )
        rewrites = result.get("rewrites") or []
        by_id = {str(item.get("case_id") or ""): item for item in rewrites}
        if set(by_id) != {case["case_id"] for case in batch}:
            raise ValueError("rewrite_batch_case_id_mismatch")
        for case in batch:
            case_id = case["case_id"]
            query = str(by_id[case_id].get("query") or "").strip()
            source_text = str((case.get("source_input") or {}).get("text") or "")
            validate_query(query, source_text=source_text)
            existing[case_id] = {
                "case_id": case_id,
                "query": query,
                "source_input_sha256": source_sha256(case),
                "model": args.model,
                "prompt_sha256": PROMPT_SHA256,
                "audit": {
                    "usage": audit.get("usage") or {},
                    "files_read": audit.get("files_read") or [],
                },
            }
        write_output(existing)
        print(json.dumps({
            "completed": len(existing),
            "target": len(selected),
            "batch": len(batch),
        }, ensure_ascii=False), flush=True)

    if set(existing) != {case["case_id"] for case in selected}:
        raise ValueError("rewrite_output_incomplete")
    write_output(existing)
    print(json.dumps({"status": "passed", "count": len(existing), "output": str(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
