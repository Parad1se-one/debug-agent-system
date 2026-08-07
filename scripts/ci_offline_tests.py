"""CI helper: run the offline-safe test subset.

The full test suite (`tests/run_tests.py`) includes tests that require
proprietary data (raw field documents, gold annotations, offline Jira exports)
which is intentionally **not** distributed with this public repo. Those tests
fail on a clean clone by design.

This script runs exactly the files listed in `tests/offline_manifest.txt` —
the subset that passes with only the shipped (sanitized) data.

Usage:
    PYTHONPATH=src python3 scripts/ci_offline_tests.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "offline_manifest.txt"


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 2
    files = [line.strip() for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not files:
        print("empty manifest", file=sys.stderr)
        return 2
    print(f"running {len(files)} offline test files from tests/offline_manifest.txt")
    cmd = [sys.executable, str(ROOT / "tests" / "run_tests.py"), *files]
    proc = subprocess.run(cmd, cwd=ROOT, env={"PYTHONPATH": "src"})
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
