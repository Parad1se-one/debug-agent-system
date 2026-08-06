"""Synchronize the explicit FAE roster snapshot into the people-role registry."""

from __future__ import annotations

import argparse
import json

from debug_agent_system.agents.write.people_roles import sync_fae_roster


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sync-fae-roster")
    parser.add_argument("--registry", default="config/people_role_registry.json")
    parser.add_argument("--roster", default="data/annotations/fae_engineers_2026-07-21.csv")
    args = parser.parse_args(argv)
    registry = sync_fae_roster(args.registry, args.roster)
    print(json.dumps({
        "registry": args.registry,
        "roster": registry["fae_roster"],
        "confirmed_fae_count": len(registry["teams"][0]["confirmed_members"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
