from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path) -> Path:
    """Locate this repository without depending on a fixed package depth."""

    path = Path(start).resolve()
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "debug_agent_system").is_dir():
            return candidate
    raise RuntimeError(f"debug_agent_system project root not found from {start}")
