"""Independent configuration for Read Runtime v3."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReadRuntimeV3Options:
    enabled: bool = True
    shadow_mode: bool = True
    planner: str = "evidence_first_bootstrap"
    baseline_enabled: bool = True
    kg_sag_enabled: bool = True
    raw_enabled: bool = True
    incident_enabled: bool = True
    fail_open_to_baseline: bool = True
    budgets: dict[str, int] = field(default_factory=dict)
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    timeout_seconds: int = 600
    max_tool_rounds: int = 8
    max_tool_calls: int = 48


def load_options(path: str | Path) -> ReadRuntimeV3Options:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime = dict(payload.get("read_runtime_v3") or payload)
    providers = dict(runtime.get("providers") or {})
    return ReadRuntimeV3Options(
        enabled=bool(runtime.get("enabled", True)),
        shadow_mode=bool(runtime.get("shadow_mode", True)),
        planner=str(runtime.get("planner") or "evidence_first_bootstrap"),
        baseline_enabled=bool(providers.get("baseline", True)),
        kg_sag_enabled=bool(providers.get("kg_sag", True)),
        raw_enabled=bool(providers.get("raw", True)),
        incident_enabled=bool(providers.get("incident", True)),
        fail_open_to_baseline=bool(runtime.get("fail_open_to_baseline", True)),
        budgets={
            str(key): max(1, int(value))
            for key, value in (runtime.get("budgets") or {}).items()
        },
        model=str(runtime.get("model") or "gpt-5.4"),
        reasoning_effort=str(runtime.get("reasoning_effort") or "medium"),
        timeout_seconds=max(30, int(runtime.get("timeout_seconds") or 600)),
        max_tool_rounds=max(1, int(runtime.get("max_tool_rounds") or 8)),
        max_tool_calls=max(1, int(runtime.get("max_tool_calls") or 48)),
    )
