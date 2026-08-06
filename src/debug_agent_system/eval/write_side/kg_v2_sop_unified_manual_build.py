"""Compatibility CLI for the curated KG v2 write-side builder."""

from debug_agent_system.agents.write_v2.sop_manual_build import build_graph, main

__all__ = ["build_graph", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
