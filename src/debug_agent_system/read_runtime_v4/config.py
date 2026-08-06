from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass(slots=True)
class ReadRuntimeV4Options:
    enabled: bool = True
    shadow_mode: bool = True
    baseline_enabled: bool = True
    kg_sag_enabled: bool = True
    raw_enabled: bool = True
    incident_enabled: bool = True
    planner: str = "deterministic_investigation"
    fail_open_to_v3: bool = True
    budgets: dict[str, int] = field(default_factory=dict)
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    timeout_seconds: int = 600
    max_tool_rounds: int = 8
    max_tool_calls: int = 48


def load_options(path: str | Path) -> ReadRuntimeV4Options:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime = dict(payload.get("read_runtime_v4") or payload)
    providers = dict(runtime.get("providers") or {})
    return ReadRuntimeV4Options(
        enabled=bool(runtime.get("enabled", True)),
        shadow_mode=bool(runtime.get("shadow_mode", True)),
        baseline_enabled=bool(providers.get("baseline", True)),
        kg_sag_enabled=bool(providers.get("kg_sag", True)),
        raw_enabled=bool(providers.get("raw", True)),
        incident_enabled=bool(providers.get("incident", True)),
        planner=str(runtime.get("planner") or "deterministic_investigation"),
        fail_open_to_v3=bool(runtime.get("fail_open_to_v3", True)),
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
