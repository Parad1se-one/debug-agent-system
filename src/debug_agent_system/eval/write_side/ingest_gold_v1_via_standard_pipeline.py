"""Re-enter frozen Goldcase 001--010 through W2->W3->W4->W6->W5.

The frozen annotations are used only at the W6 expert-review boundary.  W2
first receives the source episode without the ``gold`` object or reviewed-case
alignment examples.  The original W2 proposal remains in the review queue and
is rejected as superseded; a corrected candidate is then normalized by W3,
gated by W4, explicitly approved in W6, and applied only by W5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.eval.write_side.gold_set import verify_gold_set
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


DEFAULT_GOLD_ROOT = Path("data/annotations/goldcases/gold-v1")
DEFAULT_KG_ROOT = Path("data/kg_v2")
DEFAULT_LEGACY_KG_ROOT = Path("data/kg")
DEFAULT_OUT = Path("data/results/gold_v1_standard_ingest")


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _graph_state(root: str | Path) -> dict[str, Any]:
    store = JsonKGV2Store(root)
    return {
        "object_counts": {key: len(value) for key, value in store.objects_by_type.items()},
        "relation_count": len(store.relations),
        "graph_sha256": _canonical_hash({"objects": store.objects_by_type, "relations": store.relations}),
    }


def _load_cases(root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(root.glob("goldcase-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or path.stem)
        if case_id < "goldcase-001" or case_id > "goldcase-010":
            continue
        episode = payload.get("episode_input") if isinstance(payload.get("episode_input"), dict) else {}
        if not episode:
            raise ValueError(f"{case_id}:missing_episode_input")
        cases.append({"case_id": case_id, "path": path, "payload": payload, "episode": episode})
    if [item["case_id"] for item in cases] != [f"goldcase-{index:03d}" for index in range(1, 11)]:
        raise ValueError("gold_v1_expected_exactly_001_010")
    return cases


def _typed_episode(review_item: dict[str, Any]) -> dict[str, Any]:
    typed = review_item.get("typed_candidate") if isinstance(review_item.get("typed_candidate"), dict) else {}
    payload = typed.get("payload") if isinstance(typed.get("payload"), dict) else {}
    return payload.get("episode") if isinstance(payload.get("episode"), dict) else {}


def _case_by_episode(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in cases:
        for value in (
            item["case_id"],
            item["payload"].get("source_episode_id"),
            item["episode"].get("episode_id"),
        ):
            if str(value or ""):
                out[str(value)] = item
    return out


def _anchor_kind(anchor: str) -> str:
    if anchor.startswith("jira:"):
        return "jira"
    return "manual_review_anchor"


def build_gold_expert_correction(case: dict[str, Any], review_item: dict[str, Any]) -> dict[str, Any]:
    """Convert frozen truth into a W6 correction, never a direct graph bundle."""

    payload = case["payload"]
    gold = payload.get("gold") if isinstance(payload.get("gold"), dict) else {}
    family = gold.get("family") if isinstance(gold.get("family"), dict) else {}
    variant = gold.get("variant") if isinstance(gold.get("variant"), dict) else {}
    actions = [item for item in gold.get("actions") or [] if isinstance(item, dict)]
    action_order = {str(item.get("label") or ""): index for index, item in enumerate(actions, start=1)}
    actual_labels = set((gold.get("trace") or {}).get("actual_action_labels") or [])
    anchors = payload.get("evidence_anchor_map") if isinstance(payload.get("evidence_anchor_map"), dict) else {}
    return {
        "review_id": str(review_item.get("review_id") or ""),
        "disposition": "apply_corrected_same_case",
        "source_episode_id_original": str(payload.get("source_episode_id") or ""),
        "family": str(family.get("label") or ""),
        "variant": str(variant.get("label") or ""),
        "equipment_type": str(variant.get("equipment_type") or ""),
        "site": str(variant.get("site") or ""),
        "software_version": str(variant.get("software_version") or ""),
        "error_phase": str(variant.get("error_phase") or ""),
        "owner_context": str(variant.get("owner_context") or ""),
        "actions": [
            {
                "order": index,
                "label": str(item.get("label") or ""),
                "summary": str(item.get("summary") or item.get("label") or ""),
                "role": str(item.get("action_role") or "inspect"),
                "destructive": bool(item.get("destructive")),
                "high_cost": bool(item.get("high_cost")),
                "evidence_refs": [str(value) for value in item.get("evidence_anchor_ids") or []],
            }
            for index, item in enumerate(actions, start=1)
        ],
        "actual_action_orders": [
            index for label, index in action_order.items() if label in actual_labels
        ],
        "outcomes": [
            {
                "action_order": action_order[str(item.get("action_label") or "")],
                "outcome_type": str(item.get("outcome_type") or "pending_validation"),
                "summary": str(item.get("summary") or item.get("action_label") or ""),
                "root_cause_summary": str(item.get("root_cause_summary") or ""),
                "evidence_refs": [str(value) for value in item.get("evidence_anchor_ids") or []],
                "destructive": bool(item.get("destructive")),
                "high_cost": bool(item.get("high_cost")),
            }
            for item in gold.get("outcomes") or []
            if isinstance(item, dict) and str(item.get("action_label") or "") in action_order
        ],
        "required_info": [
            {
                "slot": str(item.get("slot") or "other"),
                "question": str(item.get("question") or ""),
                "why_required": str(item.get("why_required") or ""),
                "condition": str(item.get("condition") or ""),
                "blocks": "; ".join(str(value) for value in item.get("blocks") or []),
                "priority": str(item.get("priority") or "medium"),
                "evidence_refs": [str(value) for value in item.get("evidence_anchor_ids") or []],
            }
            for item in gold.get("required_info") or []
            if isinstance(item, dict)
        ],
        "evidence_additions": [
            {
                "external_id": str(anchor),
                "kind": _anchor_kind(str(anchor)),
                "summary": str(summary),
                # Point to the current source episode, never to the Gold JSON.
                # The frozen annotation is the W6 decision basis, not evidence
                # that may be copied into the candidate graph.
                "source_path": str(payload.get("source_episode_id") or ""),
            }
            for anchor, summary in anchors.items()
        ],
        "review_basis": {
            "trust_tier": "gold",
            "annotation_set_id": "gold-v1",
            "annotation_case_id": case["case_id"],
            "annotation_sha256": hashlib.sha256(case["path"].read_bytes()).hexdigest(),
            "ingest_run_id": "gold-v1:w7-w2-w3-w4-w6-w5:kg-v2.1",
            "policy": "frozen_gold_used_only_at_w6_expert_review",
        },
    }


def _w6_action_for_corrected_gate(gate: dict[str, Any]) -> str:
    """Choose the explicit W6 scope after expert correction and re-gating.

    W4 ``route_review`` means that automatic materialization is forbidden
    until W6 decides.  It does not mean that an execution-ready candidate must
    remain support-only after the requested expert review has completed.
    """

    decision = str(gate.get("decision") or "")
    if decision == "reject":
        raise ValueError("corrected_candidate_rejected_by_w4")
    if (
        decision in {"admit", "route_review"}
        and str(gate.get("admission_target") or "") == "fault_execution"
        and str(gate.get("admission_readiness") or "") == "execution_ready"
    ):
        return "approve_for_execution_policy"
    return "approve_support_only"


def run(
    *,
    gold_root: str | Path = DEFAULT_GOLD_ROOT,
    kg_root: str | Path = DEFAULT_KG_ROOT,
    legacy_kg_root: str | Path = DEFAULT_LEGACY_KG_ROOT,
    out_dir: str | Path = DEFAULT_OUT,
    approve: bool = False,
    authorization: str = "",
    w2_workers: int = 1,
) -> dict[str, Any]:
    gold_root = Path(gold_root)
    kg_root = Path(kg_root)
    out_dir = Path(out_dir)
    if approve and not authorization.strip():
        raise ValueError("explicit_authorization_required_for_w6_approval")
    integrity = verify_gold_set(gold_root)
    if not integrity.get("ok"):
        raise ValueError("gold_v1_integrity_failed")
    cases = _load_cases(gold_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = out_dir / "review_queue"
    empty_gold = out_dir / "alignment_inputs" / "empty_gold"
    empty_manual = out_dir / "alignment_inputs" / "empty_manual"
    empty_gold.mkdir(parents=True, exist_ok=True)
    empty_manual.mkdir(parents=True, exist_ok=True)

    before = _graph_state(kg_root)
    pipeline = WriteSidePipeline(
        JsonKGStore(legacy_kg_root),
        kg_v2_root=kg_root,
        kg_v2_queue_dir=queue_dir,
        w2_mode="native_v2",
        review_context_enabled=True,
        review_context_gold_root=empty_gold,
        review_context_manual_root=empty_manual,
    )
    w2_path = out_dir / "w2_candidates.jsonl"
    proposal_run = pipeline.run_summaries(
        [{"thread_id": "gold-v1-source-only", "episodes": [item["episode"] for item in cases]}],
        apply_approved=False,
        emit_episodes=True,
        dry_run_merge=True,
        w2_workers=w2_workers,
        kg_mode="v2",
        w2_mode="native_v2",
        source_type="chat",
        partial_candidates_path=w2_path,
    )

    original_items = pipeline.w6_v2.read_queue("v2_typed_candidates.json")
    by_episode = _case_by_episode(cases)
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in original_items:
        episode_id = str(_typed_episode(item).get("episode_id") or "")
        case = by_episode.get(episode_id)
        if case is not None and not str(item.get("candidate_id") or "").startswith("candidate:expert-corrected"):
            matched.append((case, item))
    if len(matched) != 10 or len({case["case_id"] for case, _ in matched}) != 10:
        raise ValueError(f"expected_10_one_to_one_w2_review_items:actual={len(matched)}")

    corrections: list[dict[str, Any]] = []
    for case, original in sorted(matched, key=lambda row: row[0]["case_id"]):
        correction = build_gold_expert_correction(case, original)
        corrected = pipeline.run_expert_correction(original, correction, dry_run_merge=True)
        corrected_item = corrected["review_item"]
        corrections.append({
            "case_id": case["case_id"],
            "original_review_id": original.get("review_id") or "",
            "original_candidate_id": original.get("candidate_id") or "",
            "original_gate": original.get("quality_gate") or {},
            "corrected_review_id": corrected_item.get("review_id") or "",
            "corrected_candidate_id": corrected_item.get("candidate_id") or "",
            "corrected_gate": corrected_item.get("quality_gate") or {},
            "correction": correction,
        })

    apply_results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    if approve:
        for row in corrections:
            rejected = pipeline.w6_v2.mark_decision(
                "v2_typed_candidates",
                str(row["original_review_id"]),
                "reject",
                reviewer=authorization,
                note="Superseded by frozen Gold Ground Truth correction at W6.",
            )
            gate = row["corrected_gate"] if isinstance(row["corrected_gate"], dict) else {}
            try:
                action = _w6_action_for_corrected_gate(gate)
            except ValueError as exc:
                raise ValueError(f"{row['case_id']}:{exc}") from exc
            approved = pipeline.w6_v2.mark_decision(
                "v2_typed_candidates",
                str(row["corrected_review_id"]),
                action,
                reviewer=authorization,
                note=(
                    f"{row['case_id']} frozen Gold reviewed; W2 proposal superseded by W6 correction. "
                    f"W4 review issues acknowledged: {', '.join(str(value) for value in gate.get('issues') or []) or 'none'}."
                ),
            )
            decisions.append({"case_id": row["case_id"], "original": rejected, "corrected": approved})
        apply_results = pipeline.apply_approved_review_queue(kg_mode="v2")

    store = JsonKGV2Store(kg_root)
    issues = validate_graph(store.objects_by_type, store.relations, schema_root=kg_root / "schema")
    after = _graph_state(kg_root)
    report = {
        "schema_version": "debug_agent_system.gold_v1_standard_ingest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization": authorization,
        "approved": approve,
        "gold_integrity": integrity,
        "source_policy": {
            "w2_input": "episode_input_only",
            "gold_visible_to_w2": False,
            "reviewed_examples_visible_to_w2": False,
            "gold_use": "W6_expert_review_only",
        },
        "pipeline": "W7->W2(native_v2/DeepSeek)->W3->W4->W6(original reject + corrected approve)->W5",
        "proposal_summary": proposal_run.get("summary") or {},
        "corrections": corrections,
        "decisions": decisions,
        "apply_status_counts": dict(Counter(str(item.get("status") or "") for item in apply_results)),
        "apply_results": apply_results,
        "before": before,
        "after": after,
        "graph_validation": {"status": "valid" if not issues else "invalid", "issues": issues},
    }
    (out_dir / "standard_ingestion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest-gold-v1-via-standard-pipeline")
    parser.add_argument("--gold-root", default=str(DEFAULT_GOLD_ROOT))
    parser.add_argument("--kg-root", default=str(DEFAULT_KG_ROOT))
    parser.add_argument("--legacy-kg-root", default=str(DEFAULT_LEGACY_KG_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--w2-workers", type=int, default=1)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--authorization", default="")
    args = parser.parse_args(argv)
    report = run(
        gold_root=args.gold_root,
        kg_root=args.kg_root,
        legacy_kg_root=args.legacy_kg_root,
        out_dir=args.out_dir,
        approve=args.approve,
        authorization=args.authorization,
        w2_workers=args.w2_workers,
    )
    print(json.dumps({
        "status": report["graph_validation"]["status"],
        "approved": report["approved"],
        "case_count": len(report["corrections"]),
        "apply_status_counts": report["apply_status_counts"],
        "before": report["before"],
        "after": report["after"],
    }, ensure_ascii=False, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
