from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.materializer import KGV2Materializer
from debug_agent_system.knowledge_v2.validator import validate_graph
from debug_agent_system.core.paths import project_root


BUILD_ROOT = Path("data/kg_v2_sop_draft_build/hardware_camera")
TARGET_ROOT = Path("data/kg_v2_sop_draft_hardware_camera")
SUMMARY_OUT = Path("data/results/kg_v2_sop_draft_hardware_camera_summary.json")
INVENTORY_PATH = BUILD_ROOT / "section_inventory_hardware_camera.json"
SECTION_MAP_PATH = BUILD_ROOT / "section_id_map_hardware_camera.json"
FAMILY_MAP_PATH = BUILD_ROOT / "family_map_hardware_camera.json"
MANUAL_CARD_ROOT = BUILD_ROOT / "manual_cards"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hid(prefix: str, *parts: str) -> str:
    raw = " | ".join(str(x or "") for x in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _family_id(label: str) -> str:
    return _hid("family", label)


def _variant_id(family_label: str, variant_label: str) -> str:
    return _hid("variant", family_label, variant_label)


def _action_id(variant_id: str, step_order: int, label: str) -> str:
    return _hid("action", variant_id, str(step_order), label)


def _required_id(variant_id: str, slot: str, question: str) -> str:
    return _hid("required-info", variant_id, slot, question)


def _trace_id(variant_id: str, summary: str) -> str:
    return _hid("trace", variant_id, summary)


def _policy_id(family_id: str) -> str:
    return _hid("policy", family_id)


def _case_id(section_id: str, family_label: str, variant_label: str) -> str:
    return _hid("case", section_id, family_label, variant_label)


def _evidence_id(section_id: str, family_label: str) -> str:
    return _hid("evidence", section_id, family_label)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path in sorted(MANUAL_CARD_ROOT.glob("*.json")):
        cards.append(_load_json(path))
    return cards


def _inventory_text_map() -> dict[str, str]:
    payload = _load_json(INVENTORY_PATH)
    return {str(item["canonical_section_id"]): str(item.get("raw_text") or "") for item in payload.get("items", [])}


def _validate_image_paths(cards: list[dict[str, Any]]) -> list[str]:
    repo_root = project_root(__file__)
    issues: list[str] = []
    for card in cards:
        for action in card.get("actions", []):
            for ref in action.get("curated_image_refs", []) or []:
                rel = str(ref.get("relative_path") or "")
                if rel and not (repo_root / rel).exists():
                    issues.append(f"missing_image:{card['variants'][0]['label']}:{rel}")
    return issues


def build_graph() -> dict[str, Any]:
    cards = _load_cards()
    image_issues = _validate_image_paths(cards)
    if image_issues:
        raise RuntimeError("image validation failed: " + "; ".join(image_issues[:20]))

    text_map = _inventory_text_map()
    objects: dict[str, list[dict[str, Any]]] = {
        "FaultFamily": [],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "DecisionPolicy": [],
        "EvidenceItem": [],
        "SourceCase": [],
    }
    relations: list[dict[str, Any]] = []

    def add_rel(src: str, dst: str, rel: str) -> None:
        relations.append({"from": src, "to": dst, "relation": rel})

    for card in cards:
        family_raw = card["family"]
        family = {
            "family_id": _family_id(family_raw["label"]),
            "label": family_raw["label"],
            "summary": family_raw["summary"],
            "category": family_raw["category"],
            "subsystem": family_raw["subsystem"],
            "scenario": family_raw["scenario"],
            "keywords": [],
            "source_kind": family_raw["source_kind"],
            "escalation_target": family_raw["escalation_target"],
            "owner_domain": family_raw["owner_domain"],
        }
        if not any(x["family_id"] == family["family_id"] for x in objects["FaultFamily"]):
            objects["FaultFamily"].append(family)

        for variant_raw in card["variants"]:
            variant = {
                "variant_id": _variant_id(family["label"], variant_raw["label"]),
                "family_id": family["family_id"],
                "label": variant_raw["label"],
                "summary": variant_raw["summary"],
                "equipment_type": "",
                "site": "",
                "software_version": "",
                "error_phase": variant_raw["error_phase"],
                "owner_context": variant_raw["owner_context"],
                "escalation_target": family["escalation_target"],
                "keywords": variant_raw["keywords"],
            }
            objects["FaultVariant"].append(variant)
            add_rel(family["family_id"], variant["variant_id"], "has_variant")

            source_section = card["source_sections"][0]
            case = {
                "case_id": _case_id(source_section, family["label"], variant["label"]),
                "source_kind": "sop",
                "title": variant["label"],
                "summary": variant["summary"],
                "source_ref": source_section,
                "approved": False,
            }
            evidence = {
                "evidence_id": _evidence_id(source_section, family["label"]),
                "source_kind": "sop",
                "external_id": source_section,
                "title": f"SOP {source_section}",
                "summary": text_map.get(source_section, source_section),
                "payload_ref": "异常处理 - 标准操作流程（SOP）",
            }
            objects["SourceCase"].append(case)
            objects["EvidenceItem"].append(evidence)
            add_rel(case["case_id"], variant["variant_id"], "supports")
            add_rel(evidence["evidence_id"], case["case_id"], "evidences")

            action_label_to_id: dict[str, str] = {}
            for action_raw in card.get("actions", []):
                action = {
                    "action_id": _action_id(variant["variant_id"], int(action_raw["step_order"]), action_raw["label"]),
                    "family_id": family["family_id"],
                    "variant_id": variant["variant_id"],
                    "label": action_raw["label"],
                    "summary": action_raw["summary"],
                    "action_role": action_raw["action_role"],
                    "step_order": int(action_raw["step_order"]),
                    "destructive": bool(action_raw["destructive"]),
                    "high_cost": bool(action_raw["high_cost"]),
                    "source_kind": "sop",
                    "source_section_id": action_raw["source_section_id"],
                    "curated_image_refs": list(action_raw.get("curated_image_refs") or []),
                }
                objects["DiagnosticAction"].append(action)
                action_label_to_id[action["label"]] = action["action_id"]

            for req_raw in card.get("required_info", []):
                req = {
                    "required_info_id": _required_id(variant["variant_id"], req_raw["slot"], req_raw["question"]),
                    "family_id": family["family_id"],
                    "variant_id": variant["variant_id"],
                    "slot": req_raw["slot"],
                    "question": req_raw["question"],
                    "why_required": req_raw["why_required"],
                    "condition": req_raw["condition"],
                    "blocks": req_raw["blocks"],
                    "priority": req_raw["priority"],
                    "evidence_ids": [evidence["evidence_id"]],
                }
                objects["RequiredInfoSpec"].append(req)
                add_rel(variant["variant_id"], req["required_info_id"], "has_required_info")
                add_rel(case["case_id"], req["required_info_id"], "supports")
                add_rel(evidence["evidence_id"], req["required_info_id"], "evidences")

            trace_raw = card["trace"]
            trace = {
                "trace_id": _trace_id(variant["variant_id"], trace_raw["summary"]),
                "family_id": family["family_id"],
                "variant_id": variant["variant_id"],
                "source_case_id": case["case_id"],
                "summary": trace_raw["summary"],
                "recommended_action_ids": [action_label_to_id[x] for x in trace_raw["recommended_action_labels"] if x in action_label_to_id],
                "actual_action_ids": [action_label_to_id[x] for x in trace_raw.get("actual_action_labels", []) if x in action_label_to_id],
                "evidence_ids": [evidence["evidence_id"]],
            }
            objects["DiagnosticTrace"].append(trace)
            add_rel(variant["variant_id"], trace["trace_id"], "has_trace")
            add_rel(case["case_id"], trace["trace_id"], "supports")
            for action_id in trace["recommended_action_ids"]:
                add_rel(trace["trace_id"], action_id, "used_action")

    for family in objects["FaultFamily"]:
        family_id = family["family_id"]
        fam_actions = [a for a in objects["DiagnosticAction"] if a["family_id"] == family_id]
        fam_traces = [t for t in objects["DiagnosticTrace"] if t["family_id"] == family_id]
        policy = {
            "policy_id": _policy_id(family_id),
            "family_id": family_id,
            "source_trace_ids": [t["trace_id"] for t in fam_traces],
            "source_outcome_ids": [],
            "ordered_action_ids": [a["action_id"] for a in sorted(fam_actions, key=lambda x: (int(x.get("step_order") or 999), x.get("label") or ""))],
            "ineffective_action_ids": [],
            "high_cost_action_ids": [a["action_id"] for a in fam_actions if a.get("high_cost") or a.get("destructive")],
            "deterministic_recompute": True,
        }
        objects["DecisionPolicy"].append(policy)
        add_rel(policy["policy_id"], family_id, "for_family")

    issues = validate_graph(objects, relations)
    if issues:
        raise RuntimeError("schema validation failed: " + "; ".join(issues[:20]))

    store = JsonKGV2Store(TARGET_ROOT)
    replace = store.replace_graph(objects, relations, validate=True)
    if replace.get("status") != "replaced":
        raise RuntimeError(f"replace_graph failed: {replace}")
    materialized = KGV2Materializer(store).materialize(store.materialized_root)

    summary = {
        "schema_version": "debug_agent_system.hardware_camera_manual_build_summary.v1",
        "scope": "SOP/2.硬件与系统/2.1相机/first-batch-draft",
        "build_root": str(BUILD_ROOT),
        "target_root": str(TARGET_ROOT),
        "section_map": str(SECTION_MAP_PATH),
        "inventory": str(INVENTORY_PATH),
        "family_map": str(FAMILY_MAP_PATH),
        "manual_card_root": str(MANUAL_CARD_ROOT),
        "section_ids": sorted({x for card in cards for x in card.get("source_sections", [])}),
        "counts": {k: len(v) for k, v in objects.items()},
        "relation_count": len(relations),
        "replace": replace,
        "materialized": materialized,
        "families": [x["label"] for x in objects["FaultFamily"]],
        "variants": [x["label"] for x in objects["FaultVariant"]],
    }
    _write_json(SUMMARY_OUT, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="build hardware camera draft graph")
    args = parser.parse_args(argv)
    summary = build_graph()
    print(json.dumps({"build": bool(args.build), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
