"""Human-review gate and immutable manifest for blind gold cases 011--015."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.eval.write_side.render_blind_ground_truth_review import _validate


DEFAULT_ROOT = Path("data/annotations/goldcases/review-v3")
DEFAULT_MANIFEST = "gold-011-015-review-v3.manifest.json"


class BlindGoldSetIntegrityError(ValueError):
    """Raised when review or immutable-integrity requirements are not met."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_gate(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = Path(root)
    expected_ids = [f"goldcase-{index:03d}" for index in range(11, 16)]
    input_manifest_path = root / "inputs" / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    input_rows = [item for item in input_manifest.get("cases") or [] if isinstance(item, dict)]
    input_by_case = {str(item.get("case_id") or ""): item for item in input_rows}
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if sorted(input_by_case) != expected_ids:
        issues.append(f"input_case_ids:{','.join(sorted(input_by_case))}")
    if input_manifest.get("immutable") is not True or input_manifest.get("contains_ground_truth") is not False:
        issues.append("source_input_manifest_policy_invalid")
    auxiliary_rows: list[dict[str, Any]] = []
    for item in input_manifest.get("allowed_auxiliary_inputs") or []:
        if not isinstance(item, dict):
            issues.append("invalid_allowed_auxiliary_input")
            continue
        path = Path(str(item.get("path") or ""))
        if not path.is_file():
            issues.append(f"allowed_auxiliary_input_missing:{path}")
            continue
        actual_hash = _file_hash(path)
        if actual_hash != item.get("sha256"):
            issues.append(f"allowed_auxiliary_input_sha256_mismatch:{path}")
        auxiliary_rows.append({**item, "actual_sha256": actual_hash})

    truth_paths = sorted((root / "ground_truth").glob("goldcase-*.json"))
    if [path.stem for path in truth_paths] != expected_ids:
        issues.append(f"truth_case_ids:{','.join(path.stem for path in truth_paths)}")
    for truth_path in truth_paths:
        case_id = truth_path.stem
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        input_path = root / "inputs" / truth_path.name
        input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        source_row = input_by_case.get(case_id) or {}
        case_issues = _validate(input_payload, truth)
        if truth.get("case_id") != case_id:
            case_issues.append("case_id_mismatch")
        if _file_hash(input_path) != source_row.get("file_sha256"):
            case_issues.append("frozen_input_file_sha256_mismatch")
        if truth.get("review_status") != "approved":
            case_issues.append("review_status_not_approved")
        if truth.get("graph_ingestion") is not False:
            case_issues.append("gold_evaluation_case_must_not_enter_active_kg")
        human_review = truth.get("human_review") if isinstance(truth.get("human_review"), dict) else {}
        if human_review.get("decision") != "approved":
            case_issues.append("human_review_decision_not_approved")
        if not str(human_review.get("reviewer") or "").strip():
            case_issues.append("human_reviewer_missing")
        if not str(human_review.get("reviewed_at") or "").strip():
            case_issues.append("human_reviewed_at_missing")
        case_issues = sorted(set(case_issues))
        issues.extend(f"{case_id}:{item}" for item in case_issues)
        rows.append({
            "case_id": case_id,
            "input_file": f"inputs/{truth_path.name}",
            "input_sha256": _file_hash(input_path),
            "truth_file": f"ground_truth/{truth_path.name}",
            "truth_sha256": _file_hash(truth_path),
            "review_status": truth.get("review_status"),
            "human_review": human_review,
            "issues": case_issues,
        })
    return {
        "batch_id": str(input_manifest.get("batch_id") or ""),
        "ready_to_freeze": not issues,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": _file_hash(input_manifest_path),
        "allowed_auxiliary_inputs": auxiliary_rows,
        "cases": rows,
        "issues": sorted(set(issues)),
    }


def freeze_gold_set(
    root: str | Path = DEFAULT_ROOT,
    manifest_name: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / manifest_name
    gate = review_gate(root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures: list[str] = []
        for row in manifest.get("cases") or []:
            if not isinstance(row, dict):
                continue
            for file_key, hash_key in (("input_file", "input_sha256"), ("truth_file", "truth_sha256")):
                path = root / str(row.get(file_key) or "")
                if not path.is_file() or _file_hash(path) != row.get(hash_key):
                    failures.append(f"{row.get('case_id')}:{file_key}_sha256_mismatch")
        if failures:
            raise BlindGoldSetIntegrityError(json.dumps({"failures": failures}, ensure_ascii=False))
        return {
            "status": "already_frozen",
            "manifest": str(manifest_path),
            "case_count": len(manifest.get("cases") or []),
            "ok": True,
        }
    if not gate["ready_to_freeze"]:
        raise BlindGoldSetIntegrityError(json.dumps(gate, ensure_ascii=False))
    manifest = {
        "schema_version": "debug_agent_system.blind_gold_set_manifest.v1",
        "gold_set_id": gate["batch_id"],
        "source_batch_id": gate["batch_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "immutable": True,
            "graph_ingestion": False,
            "notes": "Human-approved blind truth. Preserve the initial source-only prediction and never tune before approval.",
        },
        "input_manifest": "inputs/manifest.json",
        "input_manifest_sha256": gate["input_manifest_sha256"],
        "allowed_auxiliary_inputs": gate["allowed_auxiliary_inputs"],
        "cases": [
            {key: row[key] for key in ("case_id", "input_file", "input_sha256", "truth_file", "truth_sha256", "human_review")}
            for row in gate["cases"]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "frozen", "manifest": str(manifest_path), "case_count": len(manifest["cases"]), "ok": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blind-gold-set")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze:
        report = freeze_gold_set(args.root, args.manifest_name)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = review_gate(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_to_freeze"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
