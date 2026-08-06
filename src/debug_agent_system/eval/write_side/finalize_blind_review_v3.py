"""Finalize the reviewed 011--015 truth after binding the explicit FAE roster."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.people_roles import load_people_role_registry, people_index


PEOPLE_ROLE_NOTES = {
    "goldcase-012": "邓志勇、廖明森、孔令明均由data/annotations/fae_engineers_2026-07-21.csv确认为FAE；其日报和现场回复作为现场报告证据，案例行为角色仍与组织角色分轴记录。",
    "goldcase-013": "方扬皓、孔令明均由data/annotations/fae_engineers_2026-07-21.csv确认为FAE；其日报和现场回复作为现场报告证据，案例行为角色仍与组织角色分轴记录。",
    "goldcase-014": "工程师申由FAE名单确认为FAE，工程师丁和工程师庚由现场问题反馈流程确认为研发工程师；工程师酉、工程师辰、工程师戌未在当前注册表中确认，仍只按消息内容使用其现场事实。",
    "goldcase-015": "工程师未、工程师子均由data/annotations/fae_engineers_2026-07-21.csv确认为FAE；其现场报告作为证据，案例行为角色仍与组织角色分轴记录。",
}


def finalize(
    root: str | Path,
    *,
    reviewer: str,
    reviewed_at: str,
) -> dict[str, Any]:
    root = Path(root)
    registry = load_people_role_registry()
    explicit = people_index(registry)
    rows = []
    for path in sorted((root / "ground_truth").glob("goldcase-*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(truth.get("case_id") or path.stem)
        changed_roles: list[str] = []
        for key in ("daily_report_anchors", "field_report_anchors"):
            for anchor in truth.get(key) or []:
                if not isinstance(anchor, dict):
                    continue
                name = str(anchor.get("reporter") or anchor.get("fae") or "").strip()
                registered = explicit.get(name) or {}
                organization_roles = set(registered.get("organization_roles") or [])
                if "fae" in organization_roles:
                    anchor["role_status"] = "confirmed_fae"
                    changed_roles.append(name)
                elif "rd_engineer" in organization_roles and anchor.get("role_status") == "unconfirmed_field_reporter":
                    anchor["role_status"] = "confirmed_rd_engineer"
                    changed_roles.append(name)
        if case_id in PEOPLE_ROLE_NOTES:
            truth["people_role_note"] = PEOPLE_ROLE_NOTES[case_id]
        truth["review_status"] = "approved"
        truth["human_review"] = {
            "decision": "approved",
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "basis": "interactive_manual_review_011_015_with_final_fae_roster_binding",
        }
        path.write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({"case_id": case_id, "confirmed_role_names": sorted(set(changed_roles))})
    return {"case_count": len(rows), "cases": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finalize-blind-review-v3")
    parser.add_argument("--root", default="data/annotations/goldcases/review-v3")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", default="")
    args = parser.parse_args(argv)
    reviewed_at = args.reviewed_at or datetime.now(timezone.utc).isoformat()
    report = finalize(args.root, reviewer=args.reviewer, reviewed_at=reviewed_at)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
