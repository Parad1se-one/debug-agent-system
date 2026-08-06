"""Low-coupling adapter around KG v2 materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2 import JsonKGV2Store, KGV2Materializer


class KGv2Adapter:
    def __init__(self, root: str | Path = "data/kg_v2") -> None:
        self.store = JsonKGV2Store(root)
        self.materializer = KGV2Materializer(self.store)

    def preview(self) -> dict[str, Any]:
        materialized = self.materializer.materialize()
        return {
            "errors": materialized["errors"],
            "checks": materialized["checks"],
            "solutions": materialized["solutions"],
            "policies": materialized["policies"],
            "edges": materialized["edges"],
        }

    def materialize_execution_view(self, out_root: str | Path | None = None) -> dict[str, Any]:
        target = Path(out_root) if out_root is not None else self.store.materialized_root
        materialized = self.materializer.materialize(target)
        return {
            "status": "materialized",
            "out_root": str(target),
            "counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
        }
