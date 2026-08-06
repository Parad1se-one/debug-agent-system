"""Versioned, immutable gold-set loading and integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_GOLD_ROOT = Path("data/annotations/goldcases/gold-v1")
DEFAULT_MANIFEST = "gold-v1.manifest.json"


class GoldSetIntegrityError(ValueError):
    """Raised when a frozen gold set no longer matches its manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_gold_set(
    root: str | Path = DEFAULT_GOLD_ROOT,
    manifest_name: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    base = Path(root)
    manifest_path = base / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    rows = [*(manifest.get("cases") or []), *(manifest.get("supporting_files") or [])]
    for row in rows:
        if not isinstance(row, dict):
            continue
        relative = str(row.get("file") or "")
        expected = str(row.get("sha256") or "")
        path = base / relative
        actual = _sha256(path) if path.is_file() else ""
        item = {"file": relative, "expected_sha256": expected, "actual_sha256": actual}
        checked.append(item)
        if not relative or not expected or actual != expected:
            failures.append(item)
    case_ids = [str(row.get("case_id") or "") for row in manifest.get("cases") or [] if isinstance(row, dict)]
    if case_ids != [f"goldcase-{index:03d}" for index in range(1, 11)]:
        failures.append({
            "file": manifest_name,
            "expected_sha256": "ordered case ids goldcase-001..goldcase-010",
            "actual_sha256": ",".join(case_ids),
        })
    report = {
        "gold_set_id": str(manifest.get("gold_set_id") or ""),
        "manifest": str(manifest_path),
        "immutable": bool((manifest.get("policy") or {}).get("immutable")),
        "case_count": len(case_ids),
        "ok": not failures,
        "checked": checked,
        "failures": failures,
    }
    if failures:
        raise GoldSetIntegrityError(json.dumps(report, ensure_ascii=False))
    return report


def main() -> int:
    report = verify_gold_set()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
