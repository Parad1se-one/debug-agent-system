"""Freeze the first source-only W1 predictions for blind cases 011--015."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


class BlindPredictionFreezeError(ValueError):
    """Raised when an initial blind prediction cannot be safely frozen."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(report: dict[str, Any], input_manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected_ids = [f"goldcase-{index:03d}" for index in range(11, 16)]
    input_rows = [item for item in input_manifest.get("cases") or [] if isinstance(item, dict)]
    input_by_case = {str(item.get("case_id") or ""): item for item in input_rows}
    predictions = [item for item in report.get("predictions") or [] if isinstance(item, dict)]
    prediction_ids = [str(item.get("case_id") or "") for item in predictions]
    if prediction_ids != expected_ids:
        issues.append(f"prediction_case_ids:{','.join(prediction_ids)}")
    if sorted(input_by_case) != expected_ids:
        issues.append(f"input_case_ids:{','.join(sorted(input_by_case))}")
    if report.get("prediction_frozen_before_ground_truth_load") is not True:
        issues.append("prediction_phase_not_frozen_before_truth_load")
    if report.get("prediction_batch_sha256") != _canonical_hash(predictions):
        issues.append("prediction_batch_sha256_mismatch")
    for prediction in predictions:
        case_id = str(prediction.get("case_id") or "")
        source_row = input_by_case.get(case_id) or {}
        if prediction.get("source_only") is not True:
            issues.append(f"{case_id}:not_source_only")
        if prediction.get("ground_truth_accessed") is not False:
            issues.append(f"{case_id}:ground_truth_accessed")
        if prediction.get("input_messages_sha256") != source_row.get("messages_sha256"):
            issues.append(f"{case_id}:input_messages_sha256_mismatch")
        projected = dict(prediction)
        expected_hash = str(projected.pop("prediction_sha256", ""))
        if not expected_hash or expected_hash != _canonical_hash(projected):
            issues.append(f"{case_id}:prediction_sha256_mismatch")
    forbidden = json.dumps(predictions, ensure_ascii=False)
    for marker in ('"expected_case_count"', '"critical_errors"', '"review_status"'):
        if marker in forbidden:
            issues.append(f"label_bearing_marker:{marker}")
    return sorted(set(issues))


def freeze(
    report_path: str | Path,
    input_manifest_path: str | Path,
    out_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    report_path = Path(report_path)
    input_manifest_path = Path(input_manifest_path)
    out_path = Path(out_path)
    manifest_path = Path(manifest_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    issues = _validate(report, input_manifest)
    if issues:
        raise BlindPredictionFreezeError(json.dumps({"issues": issues}, ensure_ascii=False))

    if out_path.exists() or manifest_path.exists():
        if not out_path.is_file() or not manifest_path.is_file():
            raise BlindPredictionFreezeError("partial_frozen_artifact")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = str(manifest.get("prediction_file_sha256") or "")
        actual_hash = _file_hash(out_path)
        frozen = json.loads(out_path.read_text(encoding="utf-8"))
        if expected_hash != actual_hash:
            raise BlindPredictionFreezeError("frozen_prediction_file_sha256_mismatch")
        if frozen.get("prediction_batch_sha256") != report.get("prediction_batch_sha256"):
            raise BlindPredictionFreezeError("current_prediction_differs_from_frozen_initial_prediction")
        return {
            "status": "already_frozen",
            "out": str(out_path),
            "manifest": str(manifest_path),
            "prediction_batch_sha256": frozen.get("prediction_batch_sha256"),
        }

    frozen = {
        "schema_version": "kg_v2.blind_w1_initial_predictions.v1",
        "batch_id": str(report.get("batch_id") or ""),
        "immutable": True,
        "contains_ground_truth": False,
        "prediction_frozen_before_ground_truth_load": True,
        "prediction_batch_sha256": report.get("prediction_batch_sha256"),
        "source_input_manifest_sha256": _file_hash(input_manifest_path),
        "predictions": report.get("predictions") or [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "kg_v2.blind_w1_initial_predictions_manifest.v1",
        "batch_id": frozen["batch_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "prediction_file": out_path.name,
        "prediction_file_sha256": _file_hash(out_path),
        "prediction_batch_sha256": frozen["prediction_batch_sha256"],
        "source_input_manifest": str(input_manifest_path),
        "source_input_manifest_sha256": frozen["source_input_manifest_sha256"],
        "policy": "Never overwrite this first source-only W1 prediction after pipeline tuning.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "frozen",
        "out": str(out_path),
        "manifest": str(manifest_path),
        "prediction_batch_sha256": frozen["prediction_batch_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freeze-blind-w1-predictions")
    parser.add_argument("--report", default="data/results/gold-011-015-review-v3-w1-baseline.json")
    parser.add_argument(
        "--input-manifest",
        default="data/annotations/goldcases/review-v3/inputs/manifest.json",
    )
    parser.add_argument(
        "--out",
        default="data/results/gold-011-015-review-v3/w1-initial-predictions.json",
    )
    parser.add_argument(
        "--manifest",
        default="data/results/gold-011-015-review-v3/w1-initial-predictions.manifest.json",
    )
    args = parser.parse_args(argv)
    result = freeze(args.report, args.input_manifest, args.out, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
