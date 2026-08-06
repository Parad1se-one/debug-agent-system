from __future__ import annotations

from debug_agent_system.eval.write_side.w7_multi_agent_acceptance import (
    build_report,
)


def _calibration(status: str = "PASS"):
    return {
        "manifest": "calibration/manifest.json",
        "gate": {"status": status},
        "metrics": {
            "strict_episode_match": {"rate": 1.0},
            "trace_pairwise": {"f1": 0.95},
        },
    }


def _safety(status: str = "PASS"):
    return {
        "manifest": "fixed173/manifest.json",
        "gate": {"status": status},
        "requirements": {
            "episode_coverage_exact": True,
            "state_unchanged": True,
        },
    }


def test_acceptance_requires_calibration_safety_and_heldout():
    report = build_report(
        calibration=_calibration(),
        fixed173_safety=_safety(),
        heldout={"manifest": "heldout/manifest.json", "gate": {"status": "PASS"}},
    )
    assert report["recommendation"]["promotion_ready"] is True
    assert report["recommendation"]["next_mode"] == "assisted"
    assert report["recommendation"]["human_approval_required"] is True


def test_acceptance_stays_shadow_when_calibration_or_heldout_fails():
    report = build_report(
        calibration=_calibration("FAIL"),
        fixed173_safety=_safety(),
        heldout={"manifest": "heldout/manifest.json", "gate": {"status": "FAIL"}},
    )
    assert report["recommendation"]["promotion_ready"] is False
    assert report["recommendation"]["next_mode"] == "shadow_multi_agent"
    assert report["recommendation"]["legacy_fallback_required"] is True

