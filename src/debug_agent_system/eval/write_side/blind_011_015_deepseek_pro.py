"""Two-stage, source-only DeepSeek Pro blind prediction for review-v3.

Stage 1 discovers trace boundaries from a whole session. Stage 2 expands each
predicted trace independently into the normal CaseUnderstandingCard schema.
No ground-truth file is opened by this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from debug_agent_system.agents.write.w2_extract.case_understanding_prompt import (
    SYSTEM_PROMPT,
    normalize_card,
    tool_schema,
)
from debug_agent_system.agents.write.w2_extract.deepseek_client import (
    call_json_object,
    call_strict_tool,
    configured_model,
)
from debug_agent_system.eval.write_side.blind_011_015_deepseek_prediction import (
    _normalization_semantics,
)
from debug_agent_system.eval.write_side.blind_011_015_prompt_preview import build_preview


PIPELINE_VERSION = "deepseek-v4-pro-two-stage-v4"
DEFAULT_ROOT = Path("data/annotations/goldcases/review-v3")
DEFAULT_RUN_ROOT = Path("data/results/blind_runs/gold-011-015-review-v3") / PIPELINE_VERSION


BOUNDARY_SYSTEM_PROMPT = """\
你是工业现场故障群聊的 Trace 边界识别器。只根据输入 source_session 中的证据聚类，不做详细诊断建议。

通用边界规则：
1. 同一设备、同一故障现象及其跨日复发、排查、恢复链合并为一个 trace，不能按日期或消息窗口机械切分。
2. 不同设备、不同故障现象或互不依赖的排查链必须拆开；同时讨论不等于同一 trace。
3. FAE 日报、Jira、回复关系和附件分析可作为跨时间连接钩子，本身不是新的 trace。
4. 提问、建议、计划、培训、协调、致谢、普通生产播报和无故障附件不得单独形成 trace。
5. evidence_ids 应覆盖后续还原现象、诊断演化、实际/建议动作和结果所需的全部证据；不得编造 ID。
6. 不预设 trace 数量。证据不足时记录 uncertainties，不用常识补全。

严格调用工具输出 JSON，不输出解释文字。"""


DETAIL_SUFFIX = """

本次输入已经由上游边界识别器确定为一个候选 trace。只输出一个 cases 条目：
- 保留该 trace 内跨日复发、假设演化和恢复链；
- 不把日报/Jira/附件各自拆成 case；
- 如果发现它实际混入独立故障，不擅自新增 case，在 uncertainties 中指出需要重新切分；
- 所有字段只能引用本候选的 allowed_evidence_ids。
"""


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def boundary_tool_schema() -> dict[str, Any]:
    cluster = _strict_object({
        "cluster_ref": {"type": "string"},
        "title": {"type": "string"},
        "symptom_summary": {"type": "string"},
        "device_scope": {"type": "string"},
        "time_span": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "type": "function",
        "function": {
            "name": "discover_fault_trace_boundaries",
            "description": "Cluster a long source-only field-support session into independent fault traces.",
            "strict": True,
            "parameters": _strict_object({
                "clusters": {"type": "array", "items": cluster},
                "excluded_evidence_ids": {"type": "array", "items": {"type": "string"}},
                "global_uncertainties": {"type": "array", "items": {"type": "string"}},
            }),
        },
    }


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows(prompt_input: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for key in ("current_episode_messages", "promoted_case_evidence")
        for item in prompt_input.get(key) or []
        if isinstance(item, dict) and item.get("message_id")
    ]


def _validate_boundaries(raw: dict[str, Any], allowed: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    clusters: list[dict[str, Any]] = []
    refs: set[str] = set()
    raw_clusters = raw.get("clusters") if isinstance(raw, dict) else None
    if not isinstance(raw_clusters, list):
        return [], ["boundary_clusters_not_list"]
    for index, item in enumerate(raw_clusters):
        if not isinstance(item, dict):
            issues.append(f"clusters[{index}]:not_object")
            continue
        cluster = dict(item)
        ref = str(cluster.get("cluster_ref") or f"trace-{index + 1}")
        if ref in refs:
            issues.append(f"clusters[{index}]:duplicate_ref:{ref}")
            continue
        refs.add(ref)
        evidence: list[str] = []
        for value in cluster.get("evidence_ids") or []:
            evidence_id = str(value or "")
            if evidence_id not in allowed:
                issues.append(f"clusters[{index}]:unknown_evidence_id:{evidence_id}")
            elif evidence_id not in evidence:
                evidence.append(evidence_id)
        if not evidence:
            issues.append(f"clusters[{index}]:missing_evidence")
        cluster["cluster_ref"] = ref
        cluster["evidence_ids"] = evidence
        clusters.append(cluster)
    if not clusters:
        issues.append("boundary_no_fault_clusters")
    return clusters, sorted(set(issues))


def _usage_sum(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = {
        key
        for row in rows
        for key, value in (row.get("usage") or {}).items()
        if isinstance(value, int)
    }
    return {key: sum(int((row.get("usage") or {}).get(key) or 0) for row in rows) for key in sorted(keys)}


def _discover_boundaries(
    prompt_input: dict[str, Any], *, api_key: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    allowed = {str(value) for value in prompt_input.get("allowed_evidence_ids") or []}
    source_session = {
        "source_episode_id": prompt_input.get("source_episode_id"),
        "source_thread_id": prompt_input.get("source_thread_id"),
        "source_rows": _source_rows(prompt_input),
        "allowed_evidence_ids": sorted(allowed),
    }
    repair_issues: list[str] = []
    last_meta: dict[str, Any] = {}
    last_raw: dict[str, Any] = {}
    for semantic_attempt in range(1, 3):
        user_payload: dict[str, Any] = {"source_session": source_session}
        if repair_issues:
            user_payload["repair_request"] = {
                "instruction": "只修复边界结构或证据 ID 问题，不增加新事实。",
                "issues": repair_issues[:40],
            }
        response = call_strict_tool(
            api_key=api_key,
            system_prompt=BOUNDARY_SYSTEM_PROMPT,
            user_payload=user_payload,
            tool=boundary_tool_schema(),
            max_tokens=int(os.environ.get("DEEPSEEK_BLIND_BOUNDARY_MAX_TOKENS", "16384")),
            max_attempts=int(os.environ.get("DEEPSEEK_W2_TRANSPORT_ATTEMPTS", "3")),
        )
        last_raw = response["arguments"]
        last_meta = {key: value for key, value in response.items() if key != "arguments"}
        last_meta["semantic_attempt"] = semantic_attempt
        clusters, repair_issues = _validate_boundaries(last_raw, allowed)
        if not repair_issues:
            return clusters, {"raw": last_raw, "transport": last_meta}, []
    return [], {"raw": last_raw, "transport": last_meta}, repair_issues


def _detail_prompt_input(prompt_input: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any]:
    selected = set(cluster.get("evidence_ids") or [])
    current = [
        item for item in prompt_input.get("current_episode_messages") or []
        if str(item.get("message_id") or "") in selected
    ]
    promoted = [
        item for item in prompt_input.get("promoted_case_evidence") or []
        if str(item.get("message_id") or "") in selected
    ]
    return {
        **prompt_input,
        "input_mode": "single_predicted_trace",
        "source_episode_id": f"{prompt_input.get('source_episode_id')}:{cluster.get('cluster_ref')}",
        "current_episode_messages": current,
        "promoted_case_evidence": promoted,
        "allowed_evidence_ids": sorted(selected),
        "boundary_cluster": cluster,
    }


def _extract_detail(
    prompt_input: dict[str, Any], cluster: dict[str, Any], *, api_key: str
) -> dict[str, Any]:
    detail_input = _detail_prompt_input(prompt_input, cluster)
    semantics = _normalization_semantics(detail_input)
    repair_issues: list[str] = []
    raw: dict[str, Any] = {}
    card: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    corrections: list[str] = []
    for semantic_attempt in range(1, 3):
        user_payload: dict[str, Any] = {"input": detail_input}
        if repair_issues:
            user_payload["repair_request"] = {
                "instruction": "只修复本地校验列出的问题，不改变 trace 边界，不增加证据外事实。",
                "issues": repair_issues[:40],
            }
        detail_schema = tool_schema()["function"]["parameters"]
        response = call_json_object(
            api_key=api_key,
            system_prompt=(
                SYSTEM_PROMPT.replace("严格按工具 schema 输出，不输出解释文字。", "")
                + DETAIL_SUFFIX
                + "\n只输出一个符合 output_json_schema 的 JSON object，不输出 Markdown 或解释文字。"
            ),
            user_payload={**user_payload, "output_json_schema": detail_schema},
            max_tokens=int(os.environ.get("DEEPSEEK_BLIND_DETAIL_MAX_TOKENS", "32768")),
            max_attempts=int(os.environ.get("DEEPSEEK_W2_TRANSPORT_ATTEMPTS", "3")),
        )
        raw = response["arguments"]
        calls.append({key: value for key, value in response.items() if key != "arguments"})
        card, repair_issues, attempt_corrections = normalize_card(raw, semantics)
        corrections.extend(attempt_corrections)
        if len(card.get("cases") or []) != 1:
            repair_issues = [*repair_issues, "detail_must_return_exactly_one_case"]
        repair_issues = sorted(set(repair_issues))
        if not repair_issues:
            break
    return {
        "cluster_ref": cluster.get("cluster_ref"),
        "cluster": cluster,
        "raw_tool_arguments": raw,
        "normalized_card": card,
        "schema_valid": bool(card and not repair_issues),
        "validation_issues": repair_issues,
        "safety_corrections": sorted(set(corrections)),
        "calls": calls,
    }


def predict_request(request: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    prompt_input = request["request"]["prompt_input"]
    case_id = str(request.get("request_id") or "")
    print(f"[{case_id}] boundary:start", file=sys.stderr, flush=True)
    calls: list[dict[str, Any]] = []
    error = ""
    boundary: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    try:
        clusters, boundary, boundary_issues = _discover_boundaries(prompt_input, api_key=api_key)
        print(f"[{case_id}] boundary:done clusters={len(clusters)} issues={len(boundary_issues)}", file=sys.stderr, flush=True)
        if boundary.get("transport"):
            calls.append(boundary["transport"])
        if boundary_issues:
            raise ValueError("boundary_invalid:" + ",".join(boundary_issues[:40]))
        for cluster in clusters:
            print(f"[{case_id}] detail:start cluster={cluster.get('cluster_ref')}", file=sys.stderr, flush=True)
            try:
                detail = _extract_detail(prompt_input, cluster, api_key=api_key)
            except Exception as exc:  # noqa: BLE001 - isolate one trace from its siblings
                detail = {
                    "cluster_ref": cluster.get("cluster_ref"),
                    "cluster": cluster,
                    "raw_tool_arguments": {},
                    "normalized_card": {},
                    "schema_valid": False,
                    "validation_issues": [f"{type(exc).__name__}:{str(exc)[:1000]}"],
                    "safety_corrections": [],
                    "calls": [],
                    "error": f"{type(exc).__name__}:{str(exc)[:1000]}",
                }
            details.append(detail)
            calls.extend(detail.get("calls") or [])
            print(
                f"[{case_id}] detail:done cluster={cluster.get('cluster_ref')} valid={detail.get('schema_valid')}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - failures are frozen as blind results
        error = f"{type(exc).__name__}:{str(exc)[:1000]}"

    cases = [
        case
        for detail in details
        if detail.get("schema_valid")
        for case in (detail.get("normalized_card") or {}).get("cases") or []
        if isinstance(case, dict)
    ]
    validation_issues = [
        f"{detail.get('cluster_ref')}:{issue}"
        for detail in details
        for issue in detail.get("validation_issues") or []
    ]
    if error:
        validation_issues.append(error)
    card = {
        "schema_version": "kg_v2.case_understanding.v1",
        "source_episode_id": str(prompt_input.get("source_episode_id") or ""),
        "source_thread_id": str(prompt_input.get("source_thread_id") or ""),
        "case_count": len(cases),
        "split_required": len(cases) > 1,
        "split_reason": "two_stage_source_only_boundary_discovery",
        "cases": cases,
        "evidence_anchor_map": {
            str(item.get("message_id") or ""): str(item.get("text") or "")[:1200]
            for item in _source_rows(prompt_input)
        },
        "global_uncertainties": list((boundary.get("raw") or {}).get("global_uncertainties") or []),
        "prompt_version": PIPELINE_VERSION,
        "extraction_source": "deepseek_pro_two_stage",
        "schema_issues": sorted(set(validation_issues)),
        "schema_valid": bool(details and not validation_issues and len(cases) == len(details)),
    }
    row = {
        "case_id": str(request.get("request_id") or ""),
        "input_messages_sha256": str(request.get("input_messages_sha256") or ""),
        "prompt_payload_sha256": str(request.get("payload_sha256") or ""),
        "source_only": True,
        "ground_truth_accessed": False,
        "pipeline_version": PIPELINE_VERSION,
        "boundary_prediction": boundary,
        "detail_predictions": details,
        "normalized_card": card,
        "schema_valid": card["schema_valid"],
        "validation_issues": card["schema_issues"],
        "safety_corrections": sorted({
            correction
            for detail in details
            for correction in detail.get("safety_corrections") or []
        }),
        "calls": calls,
        "usage": _usage_sum(calls),
        "error": error,
    }
    row["prediction_sha256"] = _canonical_hash(row)
    print(
        f"[{case_id}] prediction:done traces={card['case_count']} valid={row['schema_valid']} error={bool(error)}",
        file=sys.stderr,
        flush=True,
    )
    return row


def run_and_freeze(
    *, root: str | Path, out: str | Path, manifest_out: str | Path, workers: int = 2
) -> dict[str, Any]:
    root = Path(root)
    out = Path(out)
    manifest_out = Path(manifest_out)
    if out.exists() or manifest_out.exists():
        if not out.is_file() or not manifest_out.is_file():
            raise ValueError("partial_frozen_prediction")
        manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
        if _file_hash(out) != manifest.get("prediction_file_sha256"):
            raise ValueError("frozen_prediction_sha256_mismatch")
        return {"status": "already_frozen", "out": str(out), "manifest": str(manifest_out)}
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("missing_DEEPSEEK_API_KEY")

    preview = build_preview(root / "inputs")
    predictions_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(5, int(workers)))) as executor:
        futures = {
            executor.submit(predict_request, request, api_key=api_key): str(request.get("request_id") or "")
            for request in preview["requests"]
        }
        for future in as_completed(futures):
            predictions_by_id[futures[future]] = future.result()
    predictions = [predictions_by_id[str(item.get("request_id") or "")] for item in preview["requests"]]

    # Ground-truth JSON is never opened. The frozen manifest is hashed only
    # after every source-only prediction exists and cannot influence output.
    gold_manifest = root / "gold-011-015-review-v3.manifest.json"
    report = {
        "schema_version": "kg_v2.blind_deepseek_two_stage_predictions.v1",
        "batch_id": str(preview.get("batch_id") or ""),
        "pipeline_version": PIPELINE_VERSION,
        "immutable": True,
        "source_only": True,
        "ground_truth_accessed": False,
        "prediction_frozen_before_ground_truth_load": True,
        "model": configured_model(),
        "input_manifest_sha256": _file_hash(root / "inputs" / "manifest.json"),
        "gold_manifest_sha256_after_prediction": _file_hash(gold_manifest),
        "prompt_preview_sha256": _canonical_hash(preview),
        "predictions": predictions,
        "usage": _usage_sum([{"usage": item.get("usage") or {}} for item in predictions]),
    }
    report["prediction_batch_sha256"] = _canonical_hash(predictions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "kg_v2.blind_deepseek_two_stage_predictions_manifest.v1",
        "batch_id": report["batch_id"],
        "pipeline_version": PIPELINE_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "prediction_file": str(out),
        "prediction_file_sha256": _file_hash(out),
        "prediction_batch_sha256": report["prediction_batch_sha256"],
        "input_manifest_sha256": report["input_manifest_sha256"],
        "gold_manifest_sha256_after_prediction": report["gold_manifest_sha256_after_prediction"],
        "policy": "Never overwrite or tune this run after scoring against review-v3 truth.",
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "frozen",
        "out": str(out),
        "manifest": str(manifest_out),
        "prediction_count": len(predictions),
        "schema_valid_count": sum(bool(item.get("schema_valid")) for item in predictions),
        "predicted_trace_counts": {
            item["case_id"]: int((item.get("normalized_card") or {}).get("case_count") or 0)
            for item in predictions
        },
        "usage": report["usage"],
        "prediction_batch_sha256": report["prediction_batch_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blind-011-015-deepseek-pro")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_RUN_ROOT / "predictions.json"))
    parser.add_argument("--manifest-out", default=str(DEFAULT_RUN_ROOT / "predictions.manifest.json"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(run_and_freeze(
        root=args.root,
        out=args.out,
        manifest_out=args.manifest_out,
        workers=args.workers,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
