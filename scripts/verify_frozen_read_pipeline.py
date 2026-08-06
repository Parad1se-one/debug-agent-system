"""Verify that an explicitly frozen read-side baseline has not drifted.

The v1 manifest only tracked individual files.  Read Runtime v3 freezes the
current production baseline with a v2 manifest that can additionally pin
Python package trees and immutable data snapshots.  The default remains v1 so
the historical 2026-07-30 audit is not silently redefined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config/read_side_frozen_manifest_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path, pattern: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def verify(manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    drift: list[dict[str, Any]] = []
    checked = 0

    for path_value, expected in (payload.get("files") or {}).items():
        checked += 1
        path = REPO_ROOT / path_value
        actual = _sha256(path) if path.is_file() else "missing"
        if actual != expected:
            drift.append({
                "kind": "file",
                "path": path_value,
                "expected": str(expected),
                "actual": actual,
            })

    for tree in payload.get("trees") or []:
        checked += 1
        path_value = str(tree.get("path") or "")
        root = REPO_ROOT / path_value
        if not root.is_dir():
            actual, count = "missing", 0
        else:
            actual, count = _tree_digest(
                root,
                str(tree.get("pattern") or "**/*.py"),
            )
        expected = str(tree.get("sha256") or "")
        expected_count = int(tree.get("file_count") or 0)
        if actual != expected or count != expected_count:
            drift.append({
                "kind": "tree",
                "path": path_value,
                "pattern": str(tree.get("pattern") or "**/*.py"),
                "expected": expected,
                "actual": actual,
                "expected_file_count": expected_count,
                "actual_file_count": count,
            })

    return {
        "schema_version": payload.get("schema_version"),
        "manifest": manifest.relative_to(REPO_ROOT).as_posix(),
        "checked": checked,
        "frozen": not drift,
        "drift": drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
        help="Manifest path relative to the repository root.",
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    result = verify(manifest)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
