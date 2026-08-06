#!/usr/bin/env python3
"""Build or verify the deterministic KG_v2 debug terminology layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from debug_agent_system.knowledge_v2.terminology import (  # noqa: E402
    build_terminology_layer,
    write_terminology_layer,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg-v2-root",
        type=Path,
        default=REPO_ROOT / "data/kg_v2",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build in memory and fail if generated revision differs.",
    )
    args = parser.parse_args()
    root = args.kg_v2_root.resolve()
    if args.check:
        built = build_terminology_layer(root)
        expected_path = root / "terminology/terminology_manifest.json"
        expected = (
            json.loads(expected_path.read_text(encoding="utf-8"))
            if expected_path.exists()
            else {}
        )
        report = built["report"]
        result = {
            "status": (
                "ok"
                if expected.get("revision") == report.get("revision")
                else "drift"
            ),
            "expected_revision": expected.get("revision"),
            "actual_revision": report.get("revision"),
            "report": report,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ok" else 1
    manifest = write_terminology_layer(root)
    print(json.dumps(
        {"status": "written", "manifest": manifest},
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
