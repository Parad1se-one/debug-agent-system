from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from debug_agent_system import DebugAgentSystem
from debug_agent_system.read_runtime_v4 import ReadRuntimeV4, load_options


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Read Runtime v4 shadow/active investigation runtime")
    parser.add_argument("query")
    parser.add_argument("--system-config", default="config/debug_agent_system.yaml")
    parser.add_argument("--v4-config", default="config/read_runtime_v4.yaml")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--active", action="store_true", help="Use v4 answer after verifier; default remains shadow")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    system = DebugAgentSystem.from_config(REPO_ROOT / args.system_config)
    options = load_options(REPO_ROOT / args.v4_config)
    if args.active:
        options.shadow_mode = False
    runtime = ReadRuntimeV4.from_system(system, options=options, workspace=REPO_ROOT)
    resources = [{"resource_id": f"cli:{index}", "kind": "auto", "name": Path(value).name, "path": str(Path(value).expanduser().resolve())} for index, value in enumerate(args.evidence, 1)]
    result = runtime.run({"query": args.query, "interactive": False, "evidence_resources": resources})
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
