"""Naturalize the selected KG/runtime benchmark queries without Gold access."""

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


KG_POOL = ROOT / "data/eval/benchmark/aoi_debug_benchmark_v1.json"
SHARD = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "feature_selftest_queries_kg_runtime.jsonl"
)
OUTPUT = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "kg_query_rewrites.json"
)
MODEL = "gpt-5.6-luna"
PROMPT_CONTRACT = """把每条 AOI KG/runtime 候选 Query 改写成现场工程师会自然提出的问题。
硬性要求：
1. 只能使用输入 Query 自身，不得读取 KG Gold、参考答案、故障族、变体或首动作字段。
2. 删除“现场反馈：”“请判断所属故障及具体变体，并说明需要补充的信息和首个排查动作”等测试任务模板。
3. Query 不能指定评分目标，不得出现“所属故障”“具体变体”“必须召回”“需要补充的信息”“首个排查动作”等措辞。
4. 若输入包含“需排查某原因”“通常是某原因”“需要执行某动作”等答案提示，应改成中性的可观察现象或用户目标；不得把这些提示保留为已知根因。
5. 保留现场真正可观察的现象、错误码、版本、日志片段、已做动作及结果。已经自然的问题只需轻量润色。
6. 不得发明新的日志、数值、动作结果或已解决结论。通常 20～160 个汉字，只输出 Query，不加标题或说明。
"""
PROMPT_SHA256 = hashlib.sha256(PROMPT_CONTRACT.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def selected_cases() -> list[dict[str, Any]]:
    pool = {case["case_id"]: case for case in load_json(KG_POOL)["cases"]}
    ids: list[str] = []
    marker = "formal_debug_benchmark_v1:kg_runtime_contract:"
    for line in SHARD.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        origin = str(row.get("origin") or "")
        if origin.startswith(marker):
            ids.append(origin.removeprefix(marker))
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError(f"expected_64_unique_kg_cases:{len(ids)}:{len(set(ids))}")
    return [pool[case_id] for case_id in ids]


def source_sha256(case: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "case_id": case.get("case_id"),
            "source_type": case.get("source_type"),
            "query": case.get("query"),
        },
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


def validate_query(query: str) -> None:
    banned = (
        "现场反馈：", "请判断", "所属故障", "具体变体",
        "必须召回", "需要补充的信息", "首个排查动作",
    )
    if any(value in query for value in banned):
        raise ValueError("rewrite_contains_task_scaffold")
    if not 12 <= len(query) <= 260:
        raise ValueError(f"rewrite_length:{len(query)}")


def write_output(records: dict[str, dict[str, Any]], *, model: str) -> None:
    ordered = [records[key] for key in sorted(records)]
    payload = {
        "schema_version": "debug_agent_system.formal_kg_query_rewrites.v1",
        "model": model,
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

    selected = selected_cases()
    by_case = {case["case_id"]: case for case in selected}
    existing: dict[str, dict[str, Any]] = {}
    if OUTPUT.is_file() and not args.no_resume:
        prior = load_json(OUTPUT)
        if prior.get("model") == args.model and prior.get("prompt_sha256") == PROMPT_SHA256:
            for record in prior.get("records") or []:
                case_id = str(record.get("case_id") or "")
                case = by_case.get(case_id)
                if case and record.get("source_query_sha256") == source_sha256(case):
                    existing[case_id] = record

    pending = [case for case in selected if case["case_id"] not in existing]
    runner = CodexCliAgentRunner(model=args.model, reasoning_effort="medium")
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        inputs = [
            {
                "case_id": case["case_id"],
                "source_type": case.get("source_type"),
                "query": case.get("query"),
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
        result_by_id = {
            str(item.get("case_id") or ""): item for item in rewrites
        }
        if set(result_by_id) != {case["case_id"] for case in batch}:
            raise ValueError("rewrite_batch_case_id_mismatch")
        for case in batch:
            case_id = case["case_id"]
            query = str(result_by_id[case_id].get("query") or "").strip()
            validate_query(query)
            existing[case_id] = {
                "case_id": case_id,
                "query": query,
                "source_query_sha256": source_sha256(case),
                "model": args.model,
                "prompt_sha256": PROMPT_SHA256,
                "audit": {
                    "usage": audit.get("usage") or {},
                    "files_read": audit.get("files_read") or [],
                },
            }
        write_output(existing, model=args.model)
        print(json.dumps({
            "completed": len(existing),
            "target": len(selected),
            "batch": len(batch),
        }, ensure_ascii=False), flush=True)

    if set(existing) != set(by_case):
        raise ValueError("rewrite_output_incomplete")
    write_output(existing, model=args.model)
    print(json.dumps({
        "status": "passed", "count": len(existing), "output": str(OUTPUT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
