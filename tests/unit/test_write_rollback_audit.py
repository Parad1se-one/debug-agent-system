from __future__ import annotations

from tempfile import TemporaryDirectory

from debug_agent_system.eval.write_side.approved_replay_rollback_audit import run_audit


def test_approved_replay_and_snapshot_rollback_audit_passes() -> None:
    with TemporaryDirectory() as tmp:
        report = run_audit(tmp)

    assert report["status"] == "PASS"
    assert report["rollback_mode"] == "isolated_snapshot_restore"
    assert report["production_rollback_api"] is False
    assert all(report["checks"].values())
    assert report["hashes"]["initial"] == report["hashes"]["pending"]
    assert report["hashes"]["applied"] == report["hashes"]["replay"]
    assert report["hashes"]["restored"] == report["hashes"]["initial"]
