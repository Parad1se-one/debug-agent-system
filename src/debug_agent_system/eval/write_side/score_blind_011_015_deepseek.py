"""Score an already-frozen DeepSeek prediction against frozen review-v3 truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(*, root: str | Path, prediction_path: str | Path, prediction_manifest: str | Path) -> dict[str, Any]:
    root = Path(root)
    prediction_path = Path(prediction_path)
    prediction_manifest = Path(prediction_manifest)
    frozen = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    if _file_hash(prediction_path) != frozen.get("prediction_file_sha256"):
        raise ValueError("frozen_prediction_file_sha256_mismatch")
    report = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows = []
    for prediction in report.get("predictions") or []:
        case_id = str(prediction.get("case_id") or "")
        truth = json.loads((root / "ground_truth" / f"{case_id}.json").read_text(encoding="utf-8"))
        card = prediction.get("normalized_card") if isinstance(prediction.get("normalized_card"), dict) else {}
        predicted_cases = [item for item in card.get("cases") or [] if isinstance(item, dict)]
        boundary_clusters = [
            item
            for item in ((prediction.get("boundary_prediction") or {}).get("raw") or {}).get("clusters") or []
            if isinstance(item, dict)
        ]
        detail_predictions = [
            item for item in prediction.get("detail_predictions") or [] if isinstance(item, dict)
        ]
        valid_detail_count = sum(bool(item.get("schema_valid")) for item in detail_predictions)
        expected_cases = [item for item in truth.get("cases") or [] if isinstance(item, dict)]
        expected_families = [str((item.get("family") or {}).get("label") or "") for item in expected_cases]
        predicted_families = [str((item.get("family_hypothesis") or {}).get("label") or "") for item in predicted_cases]
        expected_evidence = {
            str(value)
            for item in expected_cases
            for value in item.get("evidence_anchor_ids") or []
        }
        predicted_evidence = {
            str(value)
            for item in predicted_cases
            for value in item.get("evidence_anchor_ids") or []
        }
        rows.append({
            "case_id": case_id,
            "schema_valid": bool(prediction.get("schema_valid")),
            "expected_case_count": len(expected_cases),
            "predicted_case_count": len(predicted_cases),
            "exact_case_count": len(expected_cases) == len(predicted_cases),
            "boundary_predicted_case_count": len(boundary_clusters),
            "exact_boundary_case_count": len(expected_cases) == len(boundary_clusters),
            "detail_prediction_count": len(detail_predictions),
            "valid_detail_count": valid_detail_count,
            "detail_schema_valid_rate": round(valid_detail_count / len(detail_predictions), 4) if detail_predictions else 0.0,
            "expected_families": expected_families,
            "predicted_families": predicted_families,
            "exact_family_label_overlap": len(set(expected_families) & set(predicted_families)),
            "expected_evidence_count": len(expected_evidence),
            "predicted_evidence_count": len(predicted_evidence),
            "evidence_anchor_recall": round(len(expected_evidence & predicted_evidence) / len(expected_evidence), 4) if expected_evidence else 1.0,
            "validation_issues": prediction.get("validation_issues") or [],
            "safety_corrections": prediction.get("safety_corrections") or [],
            "prediction_sha256": prediction.get("prediction_sha256"),
        })
    return {
        "schema_version": "kg_v2.blind_deepseek_score.v1",
        "batch_id": report.get("batch_id"),
        "prediction_file_sha256": _file_hash(prediction_path),
        "scores": rows,
        "summary": {
            "case_count": len(rows),
            "schema_valid_count": sum(item["schema_valid"] for item in rows),
            "exact_case_count_matches": sum(item["exact_case_count"] for item in rows),
            "exact_boundary_case_count_matches": sum(item["exact_boundary_case_count"] for item in rows),
            "valid_detail_count": sum(item["valid_detail_count"] for item in rows),
            "detail_prediction_count": sum(item["detail_prediction_count"] for item in rows),
            "detail_schema_valid_rate": round(
                sum(item["valid_detail_count"] for item in rows)
                / sum(item["detail_prediction_count"] for item in rows),
                4,
            ) if sum(item["detail_prediction_count"] for item in rows) else 0.0,
            "mean_evidence_anchor_recall": round(sum(item["evidence_anchor_recall"] for item in rows) / len(rows), 4) if rows else 0.0,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# goldcase-011–015 DeepSeek 新管线盲测结果",
        "",
        "| case | session schema | expected/boundary/valid traces | detail valid | expected families | predicted families | evidence recall |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for row in report.get("scores") or []:
        lines.append(
            f"| {row['case_id']} | {'pass' if row['schema_valid'] else 'fail'} | "
            f"{row['expected_case_count']}/{row['boundary_predicted_case_count']}/{row['predicted_case_count']} | "
            f"{row['valid_detail_count']}/{row['detail_prediction_count']} | "
            f"{'；'.join(row['expected_families'])} | {'；'.join(row['predicted_families'])} | "
            f"{row['evidence_anchor_recall']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score-blind-011-015-deepseek")
    parser.add_argument("--root", default="data/annotations/goldcases/review-v3")
    parser.add_argument("--prediction", default="data/annotations/goldcases/review-v3/predictions/deepseek-prompt-v1-initial.json")
    parser.add_argument("--prediction-manifest", default="data/annotations/goldcases/review-v3/predictions/deepseek-prompt-v1-initial.manifest.json")
    parser.add_argument("--out", default="data/results/gold-011-015-review-v3-deepseek-score.json")
    parser.add_argument("--md-out", default="data/results/gold-011-015-review-v3-deepseek-score.md")
    args = parser.parse_args(argv)
    report = score(root=args.root, prediction_path=args.prediction, prediction_manifest=args.prediction_manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out = Path(args.md_out)
    md_out.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"out": str(out), "md_out": str(md_out), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
