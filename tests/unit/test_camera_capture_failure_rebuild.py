from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.agents.write_v2.sop_manual_build import (
    _image_binding_refs,
    _validate_image_paths,
)


CARD_PATH = Path(
    "data/kg_v2_sop_draft_build/hardware_camera/manual_cards/"
    "family_硬件问题_相机拍摄失败综合排查.json"
)


def _card() -> dict:
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


def test_camera_capture_failure_card_has_complete_reviewed_structure():
    card = _card()
    actions = card["actions"]
    labels = {item["label"] for item in actions}

    assert card["status"] == "reviewed_for_curated_build"
    assert card["source_case_approved"] is True
    assert card["family"]["label"] == "相机拍摄失败"
    assert card["family"]["subsystem"] == "相机/采集链路"
    assert card["family"]["source_kind"] == "hybrid"
    assert card["variants"][0]["label"] == "相机拍摄失败综合排查"
    assert len(actions) == 34
    assert {item["stage"] for item in actions} >= {
        "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9"
    }
    assert card["trace"]["recommended_action_labels"] == [
        item["label"] for item in actions
    ]
    assert len(card["branches"]) >= 20
    assert all(item["from_action_label"] in labels for item in card["branches"])
    assert all(
        not item.get("to_action_label") or item["to_action_label"] in labels
        for item in card["branches"]
    )
    assert {item["outcome_type"] for item in card["outcomes"]} == {
        "pending_validation",
        "verified_fix",
    }
    verified = next(
        item for item in card["outcomes"]
        if item["outcome_type"] == "verified_fix"
    )
    assert verified["activation_mode"] == "human_confirmed_runtime"
    assert len(verified["activation_requirements"]["all_of_groups"]) == 3
    assert all(
        item["safety_level"] == "human_confirmation"
        for item in actions
        if item["destructive"] or item["high_cost"]
    )


def test_camera_capture_failure_all_47_images_bind_directly_to_actions():
    card = _card()
    assert _validate_image_paths([card]) == []

    refs = [
        ref
        for action in card["actions"]
        for ref in _image_binding_refs(card, action["label"])
    ]
    assert len(refs) == 47
    assert len({item["archive_path"] for item in refs}) == 47
    assert len({item["content_hash"] for item in refs}) == 47
    assert all(item["source_section_id"] for item in refs)
    assert all(item["asset_path"] and Path(item["asset_path"]).is_file() for item in refs)


def test_camera_capture_failure_required_info_is_branch_specific():
    card = _card()
    required = card["required_info"]
    slots = {item["slot"] for item in required}
    assert slots >= {
        "log_package",
        "software_version",
        "device_model",
        "ip_config",
        "driver_context",
        "environment",
        "production_constraint",
    }
    production = next(
        item for item in required if item["slot"] == "production_constraint"
    )
    assert "更换工控机" in production["blocks"]
    assert "更换相机" in production["blocks"]
    assert all(item["condition"] and item["blocks"] for item in required)
