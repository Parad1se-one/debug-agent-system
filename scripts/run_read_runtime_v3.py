"""Run Read Runtime v3 independently of the frozen official entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from debug_agent_system import DebugAgentSystem
from debug_agent_system.read_runtime_v3 import ReadRuntimeV3, load_options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--system-config", default="config/debug_agent_system.yaml")
    parser.add_argument("--v3-config", default="config/read_runtime_v3.yaml")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    system = DebugAgentSystem.from_config(REPO_ROOT / args.system_config)
    runtime = ReadRuntimeV3.from_system(
        system,
        options=load_options(REPO_ROOT / args.v3_config),
        workspace=REPO_ROOT,
    )
    resources = [
        {
            "resource_id": f"cli:{index}",
            "kind": "auto",
            "name": Path(value).name,
            "path": str(Path(value).expanduser().resolve()),
        }
        for index, value in enumerate(args.evidence, start=1)
    ]
    result = runtime.run({
        "query": args.query,
        "interactive": False,
        "evidence_resources": resources,
    })
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = REPO_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

