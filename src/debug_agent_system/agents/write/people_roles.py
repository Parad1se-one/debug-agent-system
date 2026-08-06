"""Shared people-role registry contracts for W1 observations and W7 resolution."""

from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_ROLE_REGISTRY = Path(__file__).resolve().parents[4] / "config" / "people_role_registry.json"
DEFAULT_FAE_ROSTER = Path(__file__).resolve().parents[4] / "data" / "annotations" / "fae_engineers_2026-07-21.csv"


def normalize_person_name(value: Any) -> str:
    return " ".join(str(value or "").replace("@", "").replace("<br>", " ").split()).strip()


@lru_cache(maxsize=4)
def load_people_role_registry(path: str | Path = DEFAULT_ROLE_REGISTRY) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.exists():
        return {"schema_version": "debug_agent_system.people_roles.v1", "people": [], "teams": []}
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", "debug_agent_system.people_roles.v1")
    payload.setdefault("people", [])
    payload.setdefault("teams", [])
    return payload


def people_index(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in registry.get("people") or []:
        if not isinstance(item, dict):
            continue
        name = normalize_person_name(item.get("name"))
        if not name:
            continue
        target = out.setdefault(name, {
            "name": name,
            "organization_roles": [],
            "responsibility_scopes": [],
            "status": str(item.get("status") or "confirmed"),
            "sources": [],
            "departments": [],
            "emails": [],
            "open_ids": [],
            "account_statuses": [],
        })
        target["organization_roles"] = _unique([
            *target.get("organization_roles", []),
            *item.get("organization_roles", []),
        ])
        target["responsibility_scopes"] = _unique([
            *target.get("responsibility_scopes", []),
            *item.get("responsibility_scopes", []),
        ])
        target["sources"] = _unique([*target.get("sources", []), str(item.get("source") or "")])
        target["departments"] = _unique([*target.get("departments", []), str(item.get("department") or "")])
        target["emails"] = _unique([*target.get("emails", []), str(item.get("email") or "")])
        target["open_ids"] = _unique([*target.get("open_ids", []), str(item.get("open_id") or "")])
        target["account_statuses"] = _unique([
            *target.get("account_statuses", []),
            str(item.get("account_status") or ""),
        ])
    return out


def load_fae_roster(path: str | Path = DEFAULT_FAE_ROSTER) -> list[dict[str, str]]:
    """Load and strictly validate the explicit FAE roster snapshot."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"姓名", "FAE标注", "所属部门", "企业邮箱", "open_id", "账号状态"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"fae_roster_missing_columns:{','.join(sorted(missing))}")
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]

    names: set[str] = set()
    open_ids: set[str] = set()
    for row in rows:
        name = normalize_person_name(row["姓名"])
        open_id = row["open_id"]
        if not name or row["FAE标注"] != "FAE工程师" or not open_id:
            raise ValueError(f"fae_roster_invalid_row:{name or '<empty>'}")
        if name in names:
            raise ValueError(f"fae_roster_duplicate_name:{name}")
        if open_id in open_ids:
            raise ValueError(f"fae_roster_duplicate_open_id:{open_id}")
        names.add(name)
        open_ids.add(open_id)
        row["姓名"] = name
    return rows


def fae_roster_provenance(path: str | Path = DEFAULT_FAE_ROSTER) -> dict[str, Any]:
    source = Path(path).resolve()
    rows = load_fae_roster(source)
    return {
        "path": source.relative_to(DEFAULT_ROLE_REGISTRY.parents[1]).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "snapshot_date": "2026-07-21",
        "record_count": len(rows),
        "active_count": sum(row["账号状态"] == "已激活" for row in rows),
    }


def sync_fae_roster(
    registry_path: str | Path = DEFAULT_ROLE_REGISTRY,
    roster_path: str | Path = DEFAULT_FAE_ROSTER,
) -> dict[str, Any]:
    """Replace the explicit FAE team from a hash-bound roster snapshot."""

    registry_path = Path(registry_path)
    registry = load_people_role_registry(registry_path)
    rows = load_fae_roster(roster_path)
    provenance = fae_roster_provenance(roster_path)
    source_label = provenance["path"]
    active_rows = [row for row in rows if row["账号状态"] == "已激活"]
    active_names = [row["姓名"] for row in active_rows]

    teams = [item for item in registry.get("teams") or [] if item.get("role") != "fae"]
    teams.insert(0, {
        "role": "fae",
        "confirmed_members": active_names,
        "source": source_label,
        "source_sha256": provenance["sha256"],
    })

    by_name = {
        normalize_person_name(item.get("name")): dict(item)
        for item in registry.get("people") or []
        if isinstance(item, dict) and normalize_person_name(item.get("name"))
    }
    for row in active_rows:
        name = row["姓名"]
        person = by_name.get(name, {"name": name})
        person["organization_roles"] = _unique([*person.get("organization_roles", []), "fae"])
        person.setdefault("responsibility_scopes", [])
        person.update({
            "department": row["所属部门"],
            "email": row["企业邮箱"],
            "open_id": row["open_id"],
            "account_status": row["账号状态"],
            "status": "confirmed",
            "source": source_label,
        })
        by_name[name] = person

    registry["fae_roster"] = provenance
    registry["teams"] = teams
    registry["people"] = list(by_name.values())
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_people_role_registry.cache_clear()
    return registry


def _unique(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
