from __future__ import annotations

import json
import tempfile
from pathlib import Path

from debug_agent_system.eval.write_side.w2_run_status import build_status


def test_w2_run_status_reads_progress_and_completion():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "episodes_total": 10,
                "episodes_completed": 3,
                "w2_mode": "native_v2",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = build_status(root)
    assert status["status"] == "running"
    assert status["progress"]["episodes_completed"] == 3

    (root / "summary.json").write_text(json.dumps({"episodes": 10}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = build_status(root)
    assert status["status"] == "completed"

    (root / "postrun_report.json").write_text(json.dumps({"schema_version": "x"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = build_status(root)
    assert status["status"] == "postrun_completed"
