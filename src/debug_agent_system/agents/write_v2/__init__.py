from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "IncrementalIngestV2Agent",
    "WriteSideV2Pipeline",
    "build_expert_corrected_candidate",
    "build_v2_review_item",
]

_MODULE_BY_NAME = {
    "IncrementalIngestV2Agent": ".ingest",
    "build_v2_review_item": ".ingest",
    "WriteSideV2Pipeline": ".pipeline",
    "build_expert_corrected_candidate": ".expert_review",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
