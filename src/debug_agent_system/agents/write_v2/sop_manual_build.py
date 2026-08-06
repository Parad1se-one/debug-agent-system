from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from typing import Any

from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.materializer import KGV2Materializer
from debug_agent_system.knowledge_v2.validator import validate_graph
from debug_agent_system.core.paths import project_root


DEFAULT_TARGET_ROOT = Path("data/kg_v2")
DEFAULT_BUILD_ROOT = Path("data/kg_v2_sop_draft_build")
DEFAULT_GOLD_ROOT = Path("data/annotations/goldcases/gold-v1")
DEFAULT_SUMMARY_OUT = Path("data/results/kg_v2_write_side_build_summary.json")


def build_sources(build_root: str | Path = DEFAULT_BUILD_ROOT) -> list[dict[str, Path | str]]:
    root = Path(build_root)
    return [
    {
        "name": "main_program",
        "inventory": root / "section_inventory_main_program.json",
        "manual_card_root": root / "manual_cards/main_program",
        "section_map": root / "section_id_map_main_program.json",
        "family_map": root / "family_map_main_program.json",
    },
    {
        "name": "hardware_camera",
        "inventory": root / "hardware_camera/section_inventory_hardware_camera.json",
        "manual_card_root": root / "hardware_camera/manual_cards",
        "section_map": root / "hardware_camera/section_id_map_hardware_camera.json",
        "family_map": root / "hardware_camera/family_map_hardware_camera.json",
    },
    {
        "name": "updown_connection",
        "inventory": root / "updown_connection/section_inventory_updown_connection.json",
        "manual_card_root": root / "updown_connection/manual_cards",
        "section_map": root / "updown_connection/section_id_map_updown_connection.json",
        "family_map": root / "updown_connection/family_map_updown_connection.json",
    },
    {
        "name": "track_belt",
        "inventory": root / "track_belt/section_inventory_track_belt.json",
        "manual_card_root": root / "track_belt/manual_cards",
        "section_map": root / "track_belt/section_id_map_track_belt.json",
        "family_map": root / "track_belt/family_map_track_belt.json",
    },
    {
        "name": "sensors",
        "inventory": root / "sensors/section_inventory_sensors.json",
        "manual_card_root": root / "sensors/manual_cards",
        "section_map": root / "sensors/section_id_map_sensors.json",
        "family_map": root / "sensors/family_map_sensors.json",
    },
    {
        "name": "stoppers_lift",
        "inventory": root / "stoppers_lift/section_inventory_stoppers_lift.json",
        "manual_card_root": root / "stoppers_lift/manual_cards",
        "section_map": root / "stoppers_lift/section_id_map_stoppers_lift.json",
        "family_map": root / "stoppers_lift/family_map_stoppers_lift.json",
    },
    {
        "name": "system_ops",
        "inventory": root / "system_ops/section_inventory_system_ops.json",
        "manual_card_root": root / "system_ops/manual_cards",
        "section_map": root / "system_ops/section_id_map_system_ops.json",
        "family_map": root / "system_ops/family_map_system_ops.json",
    },
    {
        "name": "ipc",
        "inventory": root / "ipc/section_inventory_ipc.json",
        "manual_card_root": root / "ipc/manual_cards",
        "section_map": root / "ipc/section_id_map_ipc.json",
        "family_map": root / "ipc/family_map_ipc.json",
    },
    {
        "name": "review_station",
        "inventory": root / "review_station/section_inventory_review_station.json",
        "manual_card_root": root / "review_station/manual_cards",
        "section_map": root / "review_station/section_id_map_review_station.json",
        "family_map": root / "review_station/family_map_review_station.json",
    },
    {
        "name": "gas_pressure",
        "inventory": root / "gas_pressure/section_inventory_gas_pressure.json",
        "manual_card_root": root / "gas_pressure/manual_cards",
        "section_map": root / "gas_pressure/section_id_map_gas_pressure.json",
        "family_map": root / "gas_pressure/family_map_gas_pressure.json",
    },
    {
        "name": "calibration",
        "inventory": root / "calibration/section_inventory_calibration.json",
        "manual_card_root": root / "calibration/manual_cards",
        "section_map": root / "calibration/section_id_map_calibration.json",
        "family_map": root / "calibration/family_map_calibration.json",
    },
    {
        "name": "buddy",
        "inventory": root / "buddy/section_inventory_buddy.json",
        "manual_card_root": root / "buddy/manual_cards",
        "section_map": root / "buddy/section_id_map_buddy.json",
        "family_map": root / "buddy/family_map_buddy.json",
    },
    {
        "name": "motion_control",
        "inventory": root / "motion_control/section_inventory_motion_control.json",
        "manual_card_root": root / "motion_control/manual_cards",
        "section_map": root / "motion_control/section_id_map_motion_control.json",
        "family_map": root / "motion_control/family_map_motion_control.json",
    },
    {
        "name": "spc",
        "inventory": root / "spc/section_inventory_spc.json",
        "manual_card_root": root / "spc/manual_cards",
        "section_map": root / "spc/section_id_map_spc.json",
        "family_map": root / "spc/family_map_spc.json",
    },
    ]


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


def _trace_step_id(trace_id: str, ordinal: int, action_id: str) -> str:
    return _hid("trace-step", trace_id, str(ordinal), action_id)


def _branch_id(trace_id: str, source_step_id: str, condition: str, target_step_id: str) -> str:
    return _hid("branch-rule", trace_id, source_step_id, condition, target_step_id)


def _outcome_id(variant_id: str, action_id: str, outcome_type: str, summary: str) -> str:
    return _hid("outcome", variant_id, action_id, outcome_type, summary)


def _policy_id(family_id: str) -> str:
    return _hid("policy", family_id)


def _case_id(section_id: str, family_label: str, variant_label: str) -> str:
    return _hid("case", section_id, family_label, variant_label)


def _evidence_id(section_id: str, family_label: str) -> str:
    return _hid("evidence", section_id, family_label)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_text(text: str, limit: int = 500) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _inventory_text_map(sources: list[dict[str, Path | str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in sources:
        payload = _load_json(Path(source["inventory"]))
        for item in payload.get("items", []):
            merged[str(item["canonical_section_id"])] = str(item.get("raw_text") or "")
    return merged


def _load_cards(sources: list[dict[str, Path | str]]) -> tuple[list[dict[str, Any]], list[str]]:
    cards: list[dict[str, Any]] = []
    card_paths: list[str] = []
    for source in sources:
        for path in sorted(Path(source["manual_card_root"]).glob("*.json")):
            cards.append(_load_json(path))
            card_paths.append(str(path))
    return cards, card_paths


def _image_binding_refs(card: dict[str, Any], action_label: str) -> list[dict[str, Any]]:
    source_path = str(card.get("source_document") or "")
    if not source_path:
        return []
    repo_root = project_root(__file__)
    docx_path = repo_root / source_path
    if not docx_path.is_file():
        return []
    document_hash = hashlib.sha256(docx_path.read_bytes()).hexdigest()
    refs: list[dict[str, Any]] = []
    with zipfile.ZipFile(docx_path) as archive:
        for binding in card.get("image_bindings", []) or []:
            if str(binding.get("action_label") or "") != action_label:
                continue
            archive_path = str(binding.get("archive_path") or "")
            if not archive_path:
                image_number = int(binding.get("image_number") or 0)
                matches = sorted(
                    name for name in archive.namelist()
                    if name.startswith(f"word/media/image{image_number}.")
                )
                archive_path = matches[0] if matches else ""
            if not archive_path or archive_path not in archive.namelist():
                continue
            payload = archive.read(archive_path)
            content_hash = hashlib.sha256(payload).hexdigest()
            suffix = Path(archive_path).suffix.lower()
            relative_path = (
                f"data/kg_v2_sag/assets/{document_hash[:16]}/{content_hash[:24]}{suffix}"
            )
            refs.append({
                "media_id": f"media:{content_hash[:24]}",
                "media_kind": "image",
                "label": str(binding.get("label") or Path(archive_path).name),
                "relationship_id": str(binding.get("relationship_id") or ""),
                "archive_path": archive_path,
                "source_path": source_path,
                "content_hash": content_hash,
                "relative_path": relative_path,
                "asset_path": str((repo_root / relative_path).resolve()),
                "context_label": str(binding.get("caption") or action_label),
                "caption": str(binding.get("caption") or action_label),
                "role": str(binding.get("role") or "procedure_operation"),
                "source_section_id": str(binding.get("source_section_id") or ""),
                "primary": bool(binding.get("primary", True)),
            })
    return refs


def _validate_image_paths(cards: list[dict[str, Any]]) -> list[str]:
    repo_root = project_root(__file__)
    issues: list[str] = []
    for card in cards:
        source_path = str(card.get("source_document") or "")
        source_archive_names: set[str] = set()
        if source_path:
            source_doc = repo_root / source_path
            if not source_doc.is_file():
                issues.append(f"missing_source_document:{source_path}")
            else:
                with zipfile.ZipFile(source_doc) as archive:
                    source_archive_names = set(archive.namelist())
        seen_images: set[int] = set()
        action_labels = {
            str(action.get("label") or "") for action in card.get("actions", []) or []
        }
        for binding in card.get("image_bindings", []) or []:
            image_number = int(binding.get("image_number") or 0)
            if image_number:
                if image_number in seen_images:
                    issues.append(f"duplicate_image_binding:{card['variants'][0]['label']}:{image_number}")
                seen_images.add(image_number)
            action_label = str(binding.get("action_label") or "")
            if action_label not in action_labels:
                issues.append(f"image_binding_missing_action:{card['variants'][0]['label']}:{action_label}")
            archive_path = str(binding.get("archive_path") or "")
            if archive_path and archive_path not in source_archive_names:
                issues.append(f"missing_archive_image:{card['variants'][0]['label']}:{archive_path}")
            if image_number and not archive_path and not any(
                name.startswith(f"word/media/image{image_number}.")
                for name in source_archive_names
            ):
                issues.append(
                    f"missing_archive_image_number:"
                    f"{card['variants'][0]['label']}:{image_number}"
                )
        expected_count = int(card.get("expected_image_count") or 0)
        if expected_count and len(seen_images) != expected_count:
            issues.append(
                f"image_binding_count:{card['variants'][0]['label']}:{len(seen_images)}!={expected_count}"
            )
        for action in card.get("actions", []):
            for ref in action.get("curated_image_refs", []) or []:
                rel = str(ref.get("relative_path") or "")
                if rel and not (repo_root / rel).exists():
                    issues.append(f"missing_image:{card['variants'][0]['label']}:{rel}")
    return issues


def _copy_raw_gold_cases(gold_root: Path, target_root: Path) -> list[str]:
    index = _load_json(gold_root / "index.json")
    raw_cases = []
    case_ids = []
    for item in index.get("cases", []):
        if not isinstance(item, dict) or not item.get("case_id"):
            continue
        case_id = str(item["case_id"])
        payload = _load_json(gold_root / str(item.get("file") or f"{case_id}.json"))
        case_ids.append(case_id)
        raw_cases.append({
            "case_id": case_id,
            "source_episode_id": str(payload.get("source_episode_id") or ""),
            "source_thread_id": str(payload.get("source_thread_id") or ""),
            "source_file": str(payload.get("source_file") or ""),
            "source_excerpt": [str(x) for x in payload.get("source_excerpt") or []],
            "evidence_anchor_map": payload.get("evidence_anchor_map") or {},
        })
    target = target_root / "gold_cases"
    if target.resolve() != gold_root.resolve():
        shutil.copytree(gold_root, target, dirs_exist_ok=True)
    _write_json(target / "raw_source_texts.json", {
        "schema_version": "kg_v2.gold_case_raw_sources.v1",
        "graph_ingestion": False,
        "cases": raw_cases,
    })
    return case_ids


def build_graph(
    *,
    target_root: str | Path = DEFAULT_TARGET_ROOT,
    build_root: str | Path = DEFAULT_BUILD_ROOT,
    gold_root: str | Path = DEFAULT_GOLD_ROOT,
    summary_out: str | Path = DEFAULT_SUMMARY_OUT,
) -> dict[str, Any]:
    target = Path(target_root)
    schema_root = Path(build_root) / "schema"
    if not (schema_root / "object-types.json").exists() or not (schema_root / "link-types.json").exists():
        raise FileNotFoundError(f"curated_schema_missing:{schema_root}")
    target_schema = target / "schema"
    # A legacy bootstrap must not silently downgrade a newer active schema.
    # In particular, the versioned document layer (KnowledgeDocument/Section/
    # Step and terminology objects) is required by the SOP incremental path.
    # Fresh staging roots still receive the reviewed build schema.
    if not (
        (target_schema / "object-types.json").is_file()
        and (target_schema / "link-types.json").is_file()
    ):
        shutil.copytree(schema_root, target_schema, dirs_exist_ok=True)
    sources = build_sources(build_root)
    cards, card_paths = _load_cards(sources)
    image_issues = _validate_image_paths(cards)
    if image_issues:
        raise RuntimeError("image validation failed: " + "; ".join(image_issues[:20]))

    text_map = _inventory_text_map(sources)
    objects: dict[str, list[dict[str, Any]]] = {
        "FaultFamily": [],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "TraceStep": [],
        "ExecutionObservation": [],
        "BranchRule": [],
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
                "approved": bool(card.get("source_case_approved", False)),
            }
            evidence = {
                "evidence_id": _evidence_id(source_section, family["label"]),
                "source_kind": "sop",
                "external_id": source_section,
                "title": f"SOP {source_section}",
                "summary": _clip_text(text_map.get(source_section, source_section)),
                "payload_ref": "异常处理 - 标准操作流程（SOP）",
            }
            objects["SourceCase"].append(case)
            objects["EvidenceItem"].append(evidence)
            add_rel(case["case_id"], variant["variant_id"], "supports")
            add_rel(evidence["evidence_id"], case["case_id"], "evidences")

            action_label_to_id: dict[str, str] = {}
            for action_raw in card.get("actions", []):
                image_refs = [
                    *list(action_raw.get("curated_image_refs") or []),
                    *_image_binding_refs(card, str(action_raw["label"])),
                ]
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
                    "stage": str(action_raw.get("stage") or ""),
                    "safety_level": str(action_raw.get("safety_level") or "safe"),
                    "applicability_condition": str(action_raw.get("applicability_condition") or ""),
                    "expected_result": str(action_raw.get("expected_result") or ""),
                    "curated_image_refs": image_refs,
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

            trace_step_by_action_label: dict[str, dict[str, Any]] = {}
            previous_step: dict[str, Any] | None = None
            for ordinal, action_label in enumerate(trace_raw["recommended_action_labels"], start=1):
                action_id = action_label_to_id.get(action_label)
                if not action_id:
                    continue
                step = {
                    "trace_step_id": _trace_step_id(trace["trace_id"], ordinal, action_id),
                    "trace_id": trace["trace_id"],
                    "source_case_id": case["case_id"],
                    "action_id": action_id,
                    "ordinal": ordinal,
                    "execution_status": "recommended",
                    "attempt_index": 1,
                    "evidence_ids": [evidence["evidence_id"]],
                }
                objects["TraceStep"].append(step)
                trace_step_by_action_label[action_label] = step
                add_rel(trace["trace_id"], step["trace_step_id"], "has_trace_step")
                add_rel(step["trace_step_id"], action_id, "step_action")
                add_rel(case["case_id"], step["trace_step_id"], "supports")
                add_rel(evidence["evidence_id"], step["trace_step_id"], "evidences")
                if previous_step is not None:
                    add_rel(previous_step["trace_step_id"], step["trace_step_id"], "next_trace_step")
                previous_step = step

            for branch_raw in card.get("branches", []) or []:
                source_step = trace_step_by_action_label.get(
                    str(branch_raw.get("from_action_label") or "")
                )
                target_step = trace_step_by_action_label.get(
                    str(branch_raw.get("to_action_label") or "")
                )
                if source_step is None:
                    continue
                condition = str(branch_raw.get("condition") or "")
                branch = {
                    "branch_rule_id": _branch_id(
                        trace["trace_id"],
                        source_step["trace_step_id"],
                        condition,
                        str((target_step or {}).get("trace_step_id") or ""),
                    ),
                    "trace_id": trace["trace_id"],
                    "source_case_id": case["case_id"],
                    "from_trace_step_id": source_step["trace_step_id"],
                    "to_trace_step_id": str((target_step or {}).get("trace_step_id") or ""),
                    "trigger_outcome_types": list(branch_raw.get("trigger_outcome_types") or []),
                    "condition": condition,
                    "condition_code": str(branch_raw.get("condition_code") or ""),
                    "match_any": list(branch_raw.get("match_any") or []),
                    "match_all": list(branch_raw.get("match_all") or []),
                    "match_all_groups": list(
                        branch_raw.get("match_all_groups") or []
                    ),
                    "exclude_any": list(branch_raw.get("exclude_any") or []),
                    "branch_kind": str(branch_raw.get("branch_kind") or "reviewed_recommendation"),
                    "terminal_status": str(branch_raw.get("terminal_status") or "continue"),
                    "priority": int(branch_raw.get("priority") or 100),
                    "evidence_ids": [evidence["evidence_id"]],
                }
                objects["BranchRule"].append(branch)
                add_rel(trace["trace_id"], branch["branch_rule_id"], "has_branch_rule")
                add_rel(branch["branch_rule_id"], source_step["trace_step_id"], "branch_from")
                if target_step is not None:
                    add_rel(branch["branch_rule_id"], target_step["trace_step_id"], "branch_to")
                add_rel(case["case_id"], branch["branch_rule_id"], "supports")
                add_rel(evidence["evidence_id"], branch["branch_rule_id"], "evidences")

            branched_from_ids = {
                str(item.get("from_trace_step_id") or "")
                for item in objects["BranchRule"]
                if str(item.get("trace_id") or "") == trace["trace_id"]
            }
            ordered_trace_steps = sorted(
                trace_step_by_action_label.values(),
                key=lambda item: int(item.get("ordinal") or 0),
            )
            for index, source_step in enumerate(ordered_trace_steps):
                source_step_id = str(source_step["trace_step_id"])
                if source_step_id in branched_from_ids:
                    continue
                target_step = (
                    ordered_trace_steps[index + 1]
                    if index + 1 < len(ordered_trace_steps)
                    else None
                )
                condition = (
                    "当前动作完成后按默认顺序进入下一未尝试动作"
                    if target_step is not None
                    else "默认排查轨迹已执行完毕但尚无现场验证闭环"
                )
                branch = {
                    "branch_rule_id": _branch_id(
                        trace["trace_id"],
                        source_step_id,
                        condition,
                        str((target_step or {}).get("trace_step_id") or ""),
                    ),
                    "trace_id": trace["trace_id"],
                    "source_case_id": case["case_id"],
                    "from_trace_step_id": source_step_id,
                    "to_trace_step_id": str((target_step or {}).get("trace_step_id") or ""),
                    "trigger_outcome_types": ["pending_validation"],
                    "condition": condition,
                    "condition_code": (
                        "default_next_untried"
                        if target_step is not None
                        else "default_unresolved_terminal"
                    ),
                    "branch_kind": "reviewed_recommendation",
                    "terminal_status": "continue" if target_step is not None else "unresolved",
                    "priority": 1000,
                    "evidence_ids": [evidence["evidence_id"]],
                }
                objects["BranchRule"].append(branch)
                add_rel(trace["trace_id"], branch["branch_rule_id"], "has_branch_rule")
                add_rel(branch["branch_rule_id"], source_step_id, "branch_from")
                if target_step is not None:
                    add_rel(
                        branch["branch_rule_id"],
                        str(target_step["trace_step_id"]),
                        "branch_to",
                    )
                add_rel(case["case_id"], branch["branch_rule_id"], "supports")
                add_rel(evidence["evidence_id"], branch["branch_rule_id"], "evidences")

            for outcome_raw in card.get("outcomes", []) or []:
                action_label = str(outcome_raw.get("action_label") or "")
                action_id = action_label_to_id.get(action_label)
                if not action_id:
                    continue
                outcome_type = str(outcome_raw.get("outcome_type") or "pending_validation")
                summary = str(outcome_raw.get("summary") or "")
                outcome = {
                    "outcome_id": _outcome_id(
                        variant["variant_id"], action_id, outcome_type, summary
                    ),
                    "family_id": family["family_id"],
                    "variant_id": variant["variant_id"],
                    "action_id": action_id,
                    "outcome_type": outcome_type,
                    "outcome_origin": str(outcome_raw.get("outcome_origin") or "rule_inferred"),
                    "activation_mode": str(outcome_raw.get("activation_mode") or ""),
                    "activation_requirements": dict(
                        outcome_raw.get("activation_requirements") or {}
                    ),
                    "summary": summary,
                    "source_case_id": case["case_id"],
                    "evidence_ids": [evidence["evidence_id"]],
                    "high_cost": bool(outcome_raw.get("high_cost")),
                    "destructive": bool(outcome_raw.get("destructive")),
                    "root_cause_summary": str(outcome_raw.get("root_cause_summary") or ""),
                }
                objects["ActionOutcome"].append(outcome)
                add_rel(variant["variant_id"], outcome["outcome_id"], "has_outcome")
                add_rel(case["case_id"], outcome["outcome_id"], "supports")
                add_rel(outcome["outcome_id"], action_id, "outcome_of")
                add_rel(evidence["evidence_id"], outcome["outcome_id"], "evidences")

    for family in objects["FaultFamily"]:
        family_id = family["family_id"]
        fam_actions = [a for a in objects["DiagnosticAction"] if a["family_id"] == family_id]
        fam_traces = [t for t in objects["DiagnosticTrace"] if t["family_id"] == family_id]
        policy = {
            "policy_id": _policy_id(family_id),
            "family_id": family_id,
            "source_trace_ids": [t["trace_id"] for t in fam_traces],
            "source_outcome_ids": [
                item["outcome_id"]
                for item in objects["ActionOutcome"]
                if item["family_id"] == family_id
            ],
            "ordered_action_ids": [a["action_id"] for a in sorted(fam_actions, key=lambda x: (int(x.get("step_order") or 999), x.get("label") or ""))],
            "ineffective_action_ids": [],
            "high_cost_action_ids": [a["action_id"] for a in fam_actions if a.get("high_cost") or a.get("destructive")],
            "branch_rule_ids": [
                item["branch_rule_id"]
                for item in objects["BranchRule"]
                if item["trace_id"] in {trace["trace_id"] for trace in fam_traces}
            ],
            "deterministic_recompute": True,
        }
        objects["DecisionPolicy"].append(policy)
        add_rel(policy["policy_id"], family_id, "for_family")

    issues = validate_graph(objects, relations, schema_root=schema_root)
    if issues:
        raise RuntimeError("schema validation failed: " + "; ".join(issues[:20]))

    store = JsonKGV2Store(target)
    replace = store.replace_graph(objects, relations, validate=True)
    if replace.get("status") != "replaced":
        raise RuntimeError(f"replace_graph failed: {replace}")
    materialized = KGV2Materializer(store).materialize(store.materialized_root)
    materialized_summary = {
        "status": "written",
        "out_root": str(store.materialized_root),
        "counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
    }
    raw_gold_case_ids = _copy_raw_gold_cases(Path(gold_root), target)

    summary = {
        "schema_version": "debug_agent_system.kg_v2_write_side_build_summary.v1",
        "scope": "SOP/manual-cards;gold-cases-raw-only",
        "target_root": str(target),
        "build_root": str(build_root),
        "gold_root": str(gold_root),
        "raw_gold_case_ids": raw_gold_case_ids,
        "gold_cases_ingested": False,
        "build_sources": [
            {
                "name": source["name"],
                "inventory": str(source["inventory"]),
                "manual_card_root": str(source["manual_card_root"]),
                "section_map": str(source["section_map"]),
                "family_map": str(source["family_map"]),
            }
            for source in sources
        ],
        "card_paths": card_paths,
        "section_ids": sorted({x for card in cards for x in card.get("source_sections", [])}),
        "counts": {k: len(v) for k, v in objects.items()},
        "relation_count": len(relations),
        "replace": replace,
        "materialized": materialized_summary,
        "families": [x["label"] for x in objects["FaultFamily"]],
        "variants": [x["label"] for x in objects["FaultVariant"]],
    }
    _write_json(Path(summary_out), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD_ROOT))
    parser.add_argument("--gold-root", default=str(DEFAULT_GOLD_ROOT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    args = parser.parse_args(argv)
    summary = build_graph(
        target_root=args.target_root,
        build_root=args.build_root,
        gold_root=args.gold_root,
        summary_out=args.summary_out,
    )
    print(json.dumps({"build": True, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
