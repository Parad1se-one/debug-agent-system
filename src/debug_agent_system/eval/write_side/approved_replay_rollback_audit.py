"""Isolated approved/replay/snapshot-rollback audit for the KG v2 write boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


def _semantic_state(store: JsonKGV2Store) -> dict[str, Any]:
    materialized: dict[str, Any] = {}
    if store.materialized_root.exists():
        for path in sorted(store.materialized_root.rglob("*.json")):
            materialized[path.relative_to(store.materialized_root).as_posix()] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return {
        "objects": {
            key: deepcopy(value)
            for key, value in sorted(store.objects_by_type.items())
        },
        "relations": deepcopy(store.relations),
        "materialized_execution": materialized,
    }


def _semantic_hash(state: dict[str, Any]) -> str:
    canonical = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_audit(work_root: str | Path, *, schema_root: str | Path = "data/kg_v2/schema") -> dict[str, Any]:
    root = Path(work_root)
    kg_root = root / "kg_v2"
    queue_root = root / "review_queue"
    shutil.copytree(schema_root, kg_root / "schema", dirs_exist_ok=True)
    pipeline = WriteSidePipeline(
        JsonKGStore(root / "legacy"),
        kg_v2_root=kg_root,
        kg_v2_queue_dir=queue_root,
    )
    initial_store = JsonKGV2Store(kg_root)
    initial_state = _semantic_state(initial_store)
    initial_hash = _semantic_hash(initial_state)
    initial_apply_audit = deepcopy(initial_store.read_review_queue("approved_applied.json"))

    queued = pipeline.run_diagnostic_feedback({
        "session_id": "session:rollback-audit:1",
        "query": "相机拍摄失败，当前诊断仍未解决",
        "top_error_id": "err:camera-capture-failure",
        "final_status": "unresolved",
        "check_results": {"check-network": "failed"},
    })
    pending = queued["review_item"]
    pending_result = pipeline.w5_v2.apply_approved_typed_review_item(pending, materialize=True)
    pending_hash = _semantic_hash(_semantic_state(JsonKGV2Store(kg_root)))

    decision = pipeline.w6_v2.mark_decision(
        "v2_typed_candidates",
        pending["dedupe_key"],
        "approve_support_only",
        reviewer="rollback-audit-reviewer",
        note="isolated approval/replay/rollback audit",
    )
    approved_rows = pipeline.w6_v2.read_queue("v2_typed_candidates")
    approved = next(row for row in approved_rows if row.get("dedupe_key") == pending["dedupe_key"])
    applied = pipeline.w5_v2.apply_approved_typed_review_item(approved, materialize=True)
    applied_store = JsonKGV2Store(kg_root)
    applied_state = _semantic_state(applied_store)
    applied_hash = _semantic_hash(applied_state)
    apply_audit = applied_store.read_review_queue("approved_applied.json")

    replay = pipeline.w5_v2.apply_approved_typed_review_item(approved, materialize=True)
    replay_hash = _semantic_hash(_semantic_state(JsonKGV2Store(kg_root)))

    rollback_store = JsonKGV2Store(kg_root)
    rollback_result = rollback_store.replace_graph(
        deepcopy(initial_state["objects"]),
        deepcopy(initial_state["relations"]),
    )
    if rollback_store.materialized_root.exists():
        shutil.rmtree(rollback_store.materialized_root)
    rollback_store.materialized_root.mkdir(parents=True, exist_ok=True)
    rollback_store.write_review_queue("approved_applied.json", initial_apply_audit)
    restored_state = _semantic_state(JsonKGV2Store(kg_root))
    restored_hash = _semantic_hash(restored_state)

    checks = {
        "pending_rejected": pending_result.get("status") == "skipped"
        and pending_result.get("reason") == "not_approved",
        "pending_graph_unchanged": pending_hash == initial_hash,
        "human_decision_recorded": decision.get("human_approved") is True
        and decision.get("selected_action") == "approve_support_only"
        and (approved.get("review_decision") or {}).get("reviewer") == "rollback-audit-reviewer",
        "approved_applied": applied.get("status") == "applied_to_graph_v2",
        "approved_graph_changed": applied_hash != initial_hash,
        "support_only_not_materialized": applied.get("materialized_counts") == {}
        and applied_state["materialized_execution"] == {},
        "apply_audit_recorded": len(apply_audit) == 1
        and apply_audit[0].get("rollback_anchor") == applied.get("graph_hash_before"),
        "replay_idempotent": replay == {
            "status": "already_applied",
            "dedupe_key": approved["dedupe_key"],
        },
        "replay_graph_unchanged": replay_hash == applied_hash,
        "snapshot_restore_succeeded": rollback_result.get("status") == "replaced"
        and restored_hash == initial_hash,
    }
    return {
        "schema_version": "debug_agent_system.write_rollback_audit.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "rollback_mode": "isolated_snapshot_restore",
        "production_rollback_api": False,
        "work_root": str(root),
        "dedupe_key": approved["dedupe_key"],
        "hashes": {
            "initial": initial_hash,
            "pending": pending_hash,
            "applied": applied_hash,
            "replay": replay_hash,
            "restored": restored_hash,
        },
        "results": {
            "pending": pending_result,
            "decision": {
                "review_status": decision.get("review_status"),
                "selected_action": decision.get("selected_action"),
                "reviewer": (approved.get("review_decision") or {}).get("reviewer"),
            },
            "applied": applied,
            "replay": replay,
            "rollback": rollback_result,
        },
        "checks": checks,
        "limitations": [
            "This audit restores an isolated temporary graph from an in-process snapshot.",
            "The active repository does not yet expose a production rollback command or durable snapshot registry.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", default="")
    parser.add_argument("--schema-root", default="data/kg_v2/schema")
    parser.add_argument("--out", default="data/results/write_rollback_audit/latest.json")
    args = parser.parse_args(argv)
    if args.work_root:
        report = run_audit(args.work_root, schema_root=args.schema_root)
    else:
        with tempfile.TemporaryDirectory(prefix="debug-agent-write-rollback-") as tmp:
            report = run_audit(tmp, schema_root=args.schema_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
