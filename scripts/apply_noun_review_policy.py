#!/usr/bin/env python3
"""Apply a complete, reviewable decision policy to the noun queue.

The compact policy records human decisions by stable candidate identity.  This
script expands it into one decision per current queue item and refuses to write
when a concept decision is missing or a policy entry no longer matches the
queue.  Corpus-only relations and associations may use an explicit default
decision, but the expanded artifact still records their individual review IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from debug_agent_system.knowledge_v2.noun_discovery import (  # noqa: E402
    build_noun_terminology_inventory,
    render_noun_discovery_markdown,
    render_noun_terminology_inventory_markdown,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relation_identity(item: dict[str, Any]) -> str:
    return "|".join(
        str(item.get(key) or "")
        for key in ("proposed_from_key", "proposed_relation", "proposed_to_key")
    )


def _decision_index(
    policy: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    index: dict[str, dict[str, Any]] = {}
    declared: set[str] = set()
    for section in ("concept_decisions", "variant_decisions"):
        for decision in policy.get(section) or []:
            identity = str(decision.get("identity") or "")
            if not identity or identity in index:
                raise ValueError(f"invalid_or_duplicate_policy_identity:{identity}")
            index[identity] = decision
            declared.add(f"{section}:{identity}")
    for decision in policy.get("approved_relations") or []:
        identity = str(decision.get("identity") or "")
        if not identity or identity in index:
            raise ValueError(f"invalid_or_duplicate_policy_identity:{identity}")
        index[identity] = decision
        declared.add(f"approved_relations:{identity}")
    return index, declared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg-v2-root",
        type=Path,
        default=REPO_ROOT / "data" / "kg_v2",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=(
            REPO_ROOT
            / "data/kg_v2/terminology/noun_review_policy_2026-08-03.json"
        ),
    )
    args = parser.parse_args()
    root = args.kg_v2_root.resolve()
    policy = _load(args.policy.resolve())
    queue_path = root / "review_queue/noun_discovery_candidates.json"
    queue = _load(queue_path)
    index, declared = _decision_index(policy)
    reviewer = str(policy.get("reviewed_by") or "").strip()
    reviewed_at = str(policy.get("reviewed_at") or "").strip() or datetime.now(
        timezone.utc
    ).isoformat()
    if not reviewer:
        raise ValueError("missing_policy_reviewer")

    used: set[str] = set()
    expanded: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for item in queue:
        kind = str(item.get("candidate_kind") or "")
        if kind == "new_noun_concept":
            identity = str(item.get("canonical_name") or "")
            decision = index.get(identity)
            source = "concept_decisions"
            if decision is None:
                raise ValueError(f"unreviewed_concept:{identity}")
        elif kind == "noun_surface_variant":
            identity = str(item.get("surface_form") or "")
            decision = index.get(identity)
            source = "variant_decisions"
            if decision is None:
                default = policy.get("variant_default") or {}
                decision = {"action": default.get("action"), "note": default.get("note")}
                source = "variant_default"
        elif kind == "noun_relation":
            identity = _relation_identity(item)
            decision = index.get(identity)
            source = "approved_relations"
            if decision is None:
                default = policy.get("relation_default") or {}
                decision = {"action": default.get("action"), "note": default.get("note")}
                source = "relation_default"
        elif kind == "noun_association":
            identity = _relation_identity(item)
            default = policy.get("association_default") or {}
            decision = {"action": default.get("action"), "note": default.get("note")}
            source = "association_default"
        else:
            raise ValueError(f"unsupported_candidate_kind:{kind}")

        action = str(decision.get("action") or "")
        if action not in {"approve", "reject", "defer"}:
            raise ValueError(f"invalid_decision:{kind}:{identity}:{action}")
        if source in {"concept_decisions", "variant_decisions", "approved_relations"}:
            used.add(f"{source}:{identity}")
        item.update({
            "review_status": {
                "approve": "approved",
                "reject": "rejected",
                "defer": "deferred",
            }[action],
            "selected_action": action,
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "review_note": str(decision.get("note") or ""),
        })
        if action == "approve":
            if kind == "new_noun_concept":
                if decision.get("merge_into"):
                    item["selected_concept_key"] = decision["merge_into"]
                    item["approved_relation_type"] = decision["relation_type"]
                else:
                    item["selected_canonical_name"] = decision["canonical_name"]
                    item["selected_concept_type"] = decision["concept_type"]
                    if decision.get("relation_type"):
                        item["approved_relation_type"] = decision["relation_type"]
                    if decision.get("definition"):
                        item["definition"] = decision["definition"]
            elif kind == "noun_surface_variant":
                item["selected_concept_key"] = decision["concept_key"]
                item["approved_relation_type"] = decision["relation_type"]
            else:
                item["selected_relation"] = str(
                    decision.get("relation") or item.get("proposed_relation") or ""
                )
                item["selected_target_key"] = str(
                    decision.get("target_key") or item.get("proposed_to_key") or ""
                )
        counts[item["review_status"]] = counts.get(item["review_status"], 0) + 1
        expanded.append({
            "review_id": item.get("review_id"),
            "candidate_kind": kind,
            "identity": identity,
            "action": action,
            "decision_source": source,
            "note": item["review_note"],
        })

    unused = sorted(declared - used)
    if unused and not policy.get("allow_applied_entries_absent"):
        raise ValueError("policy_entries_not_in_queue:" + ",".join(unused))
    if len(expanded) != len(queue):
        raise AssertionError("decision_count_mismatch")

    _write(queue_path, queue)
    report_path = root / "terminology/noun_discovery_report.json"
    report = _load(report_path) if report_path.exists() else {}
    kind_counts = Counter(
        str(item.get("candidate_kind") or "") for item in queue
    )
    report.update({
        "candidate_count": len(queue),
        "new_concept_count": kind_counts["new_noun_concept"],
        "surface_variant_count": kind_counts["noun_surface_variant"],
        "relation_candidate_count": kind_counts["noun_relation"],
        "association_candidate_count": kind_counts["noun_association"],
        "pending_count": sum(
            item.get("review_status") in {"pending", "needs_re_review"}
            for item in queue
        ),
        "review_status_counts": dict(sorted(counts.items())),
        "review_policy_file": str(args.policy.resolve()),
    })
    _write(report_path, report)
    (root / "terminology/noun_discovery_report.md").write_text(
        render_noun_discovery_markdown(queue, report),
        encoding="utf-8",
    )
    inventory = build_noun_terminology_inventory(
        root,
        discovery_items=queue,
        discovery_report=report,
    )
    _write(root / "terminology/noun_terminology_inventory.json", inventory)
    (root / "terminology/noun_terminology_inventory.md").write_text(
        render_noun_terminology_inventory_markdown(inventory),
        encoding="utf-8",
    )
    artifact_path = root / str(
        policy.get("expanded_decision_file")
        or "terminology/noun_review_decisions_2026-08-03.json"
    )
    _write(artifact_path, {
        "schema_version": "kg_v2.noun_review_decisions.v1",
        "policy_file": str(args.policy.resolve()),
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "candidate_count": len(expanded),
        "status_counts": counts,
        "policy_entries_absent_after_application": unused,
        "decisions": expanded,
    })
    print(json.dumps({
        "status": "applied",
        "candidate_count": len(expanded),
        "status_counts": counts,
        "policy_entries_absent_after_application": len(unused),
        "expanded_decision_file": str(artifact_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
