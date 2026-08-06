"""Deterministic execution of terminology search obligations.

The harness is intentionally a discovery step, not a retriever: it searches
each required source/canonical term before the model starts, records the
search trace even when there are zero hits, and hands bounded excerpts to the
model.  The results never become answer evidence until the model reads and
cites the original corpus file.
"""

from __future__ import annotations

from typing import Any


def execute_terminology_search_contract(
    contract: dict[str, Any],
    tools: Any,
    *,
    path_glob: str = "data/**/*",
    max_matches: int = 20,
    context_lines: int = 1,
) -> dict[str, Any]:
    """Execute every required term through the read-only search primitive."""

    tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(
        contract.get("required_search_groups") or [],
        start=1,
    ):
        if not isinstance(raw_group, dict):
            continue
        for term_index, raw_term in enumerate(
            raw_group.get("required_terms") or [],
            start=1,
        ):
            term = str(raw_term or "").strip()
            if not term:
                continue
            arguments = {
                "query": term,
                "path_glob": path_glob,
                "regex": False,
                "case_sensitive": False,
                "max_matches": max(1, min(500, int(max_matches))),
                "context_lines": max(0, min(5, int(context_lines))),
            }
            result, audit = tools.execute("search_text", arguments)
            task = {
                "task_id": f"equivalence:{group_index}:{term_index}",
                "group_index": group_index,
                "term": term,
                "source_surface_form": str(
                    raw_group.get("source_surface_form") or ""
                ),
                "canonical_name": str(raw_group.get("canonical_name") or ""),
                "resolution_status": str(
                    raw_group.get("resolution_status") or "resolved"
                ),
                "can_lock_variant": bool(
                    raw_group.get("can_lock_variant", False)
                ),
            }
            tasks.append(task)
            results.append({
                **task,
                "matches": list(result.get("matches") or [])
                if isinstance(result, dict)
                else [],
                "returned": int(result.get("returned") or 0)
                if isinstance(result, dict)
                else 0,
                "truncated": bool(result.get("truncated"))
                if isinstance(result, dict)
                else False,
            })
            trace.append({
                **audit,
                "origin": "deterministic_terminology_harness",
                "task_id": task["task_id"],
                "term": term,
            })

    # This object is a prompt/audit artifact.  It is not included in
    # files_read and therefore cannot silently turn a search excerpt into
    # answer evidence.
    return {
        "schema_version": "debug_agent_system.terminology_search_execution.v1",
        "path_glob": path_glob,
        "task_count": len(tasks),
        "tasks": tasks,
        "results": results,
        "tool_trace": trace,
    }


__all__ = ["execute_terminology_search_contract"]
