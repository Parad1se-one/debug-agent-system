"""CLI wrapper for the independent KG_v2+raw Codex read pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from debug_agent_system.kg_raw_codex.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
