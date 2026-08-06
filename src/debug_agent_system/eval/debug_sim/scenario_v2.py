"""Scenario v2 contracts for real diagnosis quality evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

ExpectedStatus = Literal["ask_info", "step", "resolved", "escalate"]


@dataclass(slots=True)
class RequiredCheck:
    id: str = ""
    text: str = ""
    required: bool = True


@dataclass(slots=True)
class UserTurn:
    when_check_contains: str = ""
    reply: str = ""
    expected_next: str = ""


@dataclass(slots=True)
class ScenarioV2:
    case_id: str
    query: str
    source: str = "kg_curated"
    difficulty: str = ""
    query_type: str = "debug"
    target_error_id: str = ""
    acceptable_error_ids: list[str] = field(default_factory=list)
    expected_status: ExpectedStatus = "step"
    required_checks: list[RequiredCheck] = field(default_factory=list)
    expected_resolution_facts: list[str] = field(default_factory=list)
    evidence_key_facts: list[str] = field(default_factory=list)
    required_info: list[str] = field(default_factory=list)
    user_turns: list[UserTurn] = field(default_factory=list)
    escalation_target: str = ""
    safety_flags: list[str] = field(default_factory=list)
    max_turns: int = 6
    metadata: dict[str, Any] = field(default_factory=dict)


def from_dict(raw: dict[str, Any]) -> ScenarioV2:
    checks = []
    for item in raw.get("required_checks") or []:
        if isinstance(item, str):
            checks.append(RequiredCheck(text=item, required=True))
        elif isinstance(item, dict):
            checks.append(RequiredCheck(
                id=str(item.get("id") or ""),
                text=str(item.get("text") or ""),
                required=bool(item.get("required", True)),
            ))
    turns = []
    for item in raw.get("user_turns") or []:
        if isinstance(item, dict):
            turns.append(UserTurn(
                when_check_contains=str(item.get("when_check_contains") or ""),
                reply=str(item.get("reply") or ""),
                expected_next=str(item.get("expected_next") or ""),
            ))
    return ScenarioV2(
        case_id=str(raw.get("case_id") or ""),
        query=str(raw.get("query") or ""),
        source=str(raw.get("source") or "kg_curated"),
        difficulty=str(raw.get("difficulty") or ""),
        query_type=str(raw.get("query_type") or "debug"),
        target_error_id=str(raw.get("target_error_id") or ""),
        acceptable_error_ids=[str(x) for x in raw.get("acceptable_error_ids") or []],
        expected_status=_status(str(raw.get("expected_status") or "step")),
        required_checks=checks,
        expected_resolution_facts=[str(x) for x in raw.get("expected_resolution_facts") or [] if str(x).strip()],
        evidence_key_facts=[str(x) for x in raw.get("evidence_key_facts") or [] if str(x).strip()],
        required_info=[str(x) for x in raw.get("required_info") or [] if str(x).strip()],
        user_turns=turns,
        escalation_target=str(raw.get("escalation_target") or ""),
        safety_flags=[str(x) for x in raw.get("safety_flags") or []],
        max_turns=int(raw.get("max_turns") or 6),
        metadata=dict(raw.get("metadata") or {}),
    )


def _status(value: str) -> ExpectedStatus:
    if value in {"ask_info", "step", "resolved", "escalate"}:
        return value  # type: ignore[return-value]
    return "step"


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def load_scenarios(path: str | Path, limit: int | None = None) -> list[ScenarioV2]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("scenarios") or []
    else:
        rows = data
    scenarios = [from_dict(x) for x in rows if isinstance(x, dict)]
    if limit is not None and limit > 0:
        return scenarios[:limit]
    return scenarios


def write_scenarios(path: str | Path, scenarios: list[ScenarioV2]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([to_jsonable(s) for s in scenarios], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
