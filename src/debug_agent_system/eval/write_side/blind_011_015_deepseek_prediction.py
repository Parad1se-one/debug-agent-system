"""Run and immutably freeze source-only DeepSeek predictions for 011--015.

This module never opens ``ground_truth/``. Scoring is deliberately implemented
in a separate module so the prediction/truth phase boundary remains auditable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.w2_extract import (
    _call_deepseek_case_understanding_with_hard_timeout,
)
from debug_agent_system.agents.write.w2_extract.case_understanding_prompt import normalize_card
from debug_agent_system.eval.write_side.blind_011_015_prompt_preview import build_preview


class BlindDeepSeekPredictionError(ValueError):
    pass


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalization_semantics(prompt_input: dict[str, Any]) -> dict[str, Any]:
    current = [dict(item) for item in prompt_input.get("current_episode_messages") or [] if isinstance(item, dict)]
    promoted = [dict(item) for item in prompt_input.get("promoted_case_evidence") or [] if isinstance(item, dict)]
    return {
        "source_episode_id": str(prompt_input.get("source_episode_id") or ""),
        "source_thread_id": str(prompt_input.get("source_thread_id") or ""),
        "evidence_ids": [str(value) for value in prompt_input.get("allowed_evidence_ids") or []],
        "episode": {
            "fault_description_messages": current,
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "case_evidence_messages": promoted,
        },
    }


def _predict_one(request: dict[str, Any], *, api_key: str, max_attempts: int) -> dict[str, Any]:
    prompt_input = request["request"]["prompt_input"]
    semantics = _normalization_semantics(prompt_input)
    repair_issues: list[str] = []
    corrections: list[str] = []
    raw: dict[str, Any] | None = None
    card: dict[str, Any] | None = None
    error = ""
    last_validation_issues: list[str] = []
    attempt = 0
    for attempt in range(1, max_attempts + 1):
        try:
            raw = _call_deepseek_case_understanding_with_hard_timeout(
                prompt_input,
                api_key=api_key,
                repair_issues=repair_issues,
            )
            card, repair_issues, attempt_corrections = normalize_card(raw, semantics)
            last_validation_issues = list(repair_issues)
            corrections.extend(attempt_corrections)
            if not repair_issues:
                break
        except Exception as exc:  # noqa: BLE001 - freeze model/API failure as an evaluation result
            error = f"{type(exc).__name__}:{str(exc)[:600]}"
            repair_issues = [*last_validation_issues, f"repair_attempt_error:{error}"]
    row = {
        "case_id": str(request.get("request_id") or ""),
        "input_messages_sha256": str(request.get("input_messages_sha256") or ""),
        "prompt_payload_sha256": str(request.get("payload_sha256") or ""),
        "source_only": True,
        "ground_truth_accessed": False,
        "attempt_count": attempt,
        "raw_tool_arguments": raw,
        "normalized_card": card,
        "schema_valid": bool(card and not repair_issues),
        "validation_issues": repair_issues,
        "safety_corrections": sorted(set(corrections)),
        "error": error,
    }
    row["prediction_sha256"] = _canonical_hash(row)
    return row


def run_and_freeze(
    *,
    root: str | Path,
    gold_manifest: str | Path,
    out: str | Path,
    manifest_out: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    gold_manifest = Path(gold_manifest)
    out = Path(out)
    manifest_out = Path(manifest_out)
    if out.exists() or manifest_out.exists():
        if not out.is_file() or not manifest_out.is_file():
            raise BlindDeepSeekPredictionError("partial_frozen_prediction")
        manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
        if _file_hash(out) != manifest.get("prediction_file_sha256"):
            raise BlindDeepSeekPredictionError("frozen_prediction_sha256_mismatch")
        return {"status": "already_frozen", "out": str(out), "manifest": str(manifest_out)}

    if not gold_manifest.is_file():
        raise BlindDeepSeekPredictionError("frozen_gold_manifest_missing")
    frozen_gold = json.loads(gold_manifest.read_text(encoding="utf-8"))
    if frozen_gold.get("policy", {}).get("immutable") is not True:
        raise BlindDeepSeekPredictionError("gold_set_not_immutable")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise BlindDeepSeekPredictionError("missing_DEEPSEEK_API_KEY")

    preview = build_preview(root / "inputs")
    predictions = [
        _predict_one(
            request,
            api_key=api_key,
            max_attempts=max(1, min(2, int(os.environ.get("DEEPSEEK_W2_PROMPT_ATTEMPTS", "2")))),
        )
        for request in preview["requests"]
    ]
    report = {
        "schema_version": "kg_v2.blind_deepseek_initial_predictions.v1",
        "batch_id": str(preview.get("batch_id") or ""),
        "immutable": True,
        "source_only": True,
        "ground_truth_accessed": False,
        "prediction_frozen_before_ground_truth_load": True,
        "prompt_version": str(preview.get("prompt_version") or ""),
        "model": os.environ.get("DEEPSEEK_W2_TOOL_MODEL", "deepseek-chat"),
        "generation_config": {
            "temperature": 0,
            "max_tokens": max(1024, min(8192, int(os.environ.get("DEEPSEEK_W2_MAX_TOKENS", "8192")))),
            "request_timeout_seconds": float(os.environ.get("DEEPSEEK_W2_TIMEOUT", "30")),
            "max_attempts": max(1, min(2, int(os.environ.get("DEEPSEEK_W2_PROMPT_ATTEMPTS", "2")))),
        },
        "gold_set_manifest_sha256": _file_hash(gold_manifest),
        "input_manifest_sha256": _file_hash(root / "inputs" / "manifest.json"),
        "prompt_preview_sha256": _canonical_hash(preview),
        "predictions": predictions,
    }
    report["prediction_batch_sha256"] = _canonical_hash(predictions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "kg_v2.blind_deepseek_initial_predictions_manifest.v1",
        "batch_id": report["batch_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "prediction_file": str(out),
        "prediction_file_sha256": _file_hash(out),
        "prediction_batch_sha256": report["prediction_batch_sha256"],
        "gold_set_manifest": str(gold_manifest),
        "gold_set_manifest_sha256": report["gold_set_manifest_sha256"],
        "input_manifest_sha256": report["input_manifest_sha256"],
        "prompt_preview_sha256": report["prompt_preview_sha256"],
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "frozen",
        "out": str(out),
        "manifest": str(manifest_out),
        "prediction_count": len(predictions),
        "schema_valid_count": sum(bool(item.get("schema_valid")) for item in predictions),
        "prediction_batch_sha256": report["prediction_batch_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blind-011-015-deepseek-prediction")
    parser.add_argument("--root", default="data/annotations/goldcases/review-v3")
    parser.add_argument("--gold-manifest", default="data/annotations/goldcases/review-v3/gold-011-015-review-v3.manifest.json")
    parser.add_argument("--out", default="data/annotations/goldcases/review-v3/predictions/deepseek-prompt-v1-initial.json")
    parser.add_argument("--manifest-out", default="data/annotations/goldcases/review-v3/predictions/deepseek-prompt-v1-initial.manifest.json")
    args = parser.parse_args(argv)
    print(json.dumps(run_and_freeze(
        root=args.root,
        gold_manifest=args.gold_manifest,
        out=args.out,
        manifest_out=args.manifest_out,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
