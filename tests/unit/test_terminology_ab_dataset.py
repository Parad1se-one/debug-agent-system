from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _validator_module():
    path = REPO_ROOT / "scripts/validate_terminology_ab_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "validate_terminology_ab_dataset", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminology_ab_dataset_contract() -> None:
    module = _validator_module()
    report = module.validate_dataset(
        REPO_ROOT / "data/eval/terminology_ab_v1"
    )
    assert report["status"] == "passed", report["issues"]
    assert report["case_count"] == 60
    assert report["category_counts"] == module.EXPECTED_QUOTAS
