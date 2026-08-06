"""Auditable terminology obligations for the model-directed read pipeline.

This module does not retrieve, rank or select evidence.  It projects the
deterministic terminology resolver output into a small search contract and
checks the model's existing tool trace after the investigation.  Codex still
decides where and how to search.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

from debug_agent_system.kg_raw_codex.coverage import AnswerScope
from debug_agent_system.knowledge_v2.terminology import normalize_term


_CONTRACT_SCHEMA_VERSION = (
    "debug_agent_system.terminology_search_contract.v1"
)
_GOVERNANCE_ONLY_PATHS = (
    "data/kg_v2/terminology/noun_terminology_inventory.json",
    "data/kg_v2/terminology/noun_terminology_inventory.md",
    "data/kg_v2/review_queue/terminology_candidates.json",
    "docs/archive/snapshots/20260803/KG_v2名词候选与变体人工审核建议.md",
)
_RUNTIME_AUTHORITY_PATHS = (
    "data/kg_v2/terminology/entity_ontology.json",
    "data/kg_v2/terminology/curated_terms.json",
    "data/kg_v2/objects/debug_concepts.json",
    "data/kg_v2/objects/term_expressions.json",
    "data/kg_v2/objects/term_senses.json",
    "data/kg_v2/relations/edges.json",
)
_CLI_SEARCH_COMMAND = re.compile(
    r"(?:^|[;&|()\s])(?:rg|grep|find|fd)(?:\s|$)",
    flags=re.I,
)


def _is_governance_path(path: str) -> bool:
    """Recognize the whole governance namespace, not just one snapshot."""

    normalized = str(path or "").replace("\\", "/").lstrip("./")
    return (
        normalized.startswith("data/kg_v2/terminology/noun_")
        or normalized.startswith("data/kg_v2/review_queue/")
        or normalized.startswith("docs/archive/snapshots/")
        or normalized.endswith("/KG_v2名词候选与变体人工审核建议.md")
    )


def build_resolver_context(answer_scope: AnswerScope) -> dict[str, list[str]]:
    """Project query-task dimensions that safely strengthen disambiguation.

    Operations are supplied as possible phases, while explicit branch
    conditions are supplied as signals.  Requested objects are deliberately
    not guessed into equipment/subsystem fields because doing so could create
    false context rather than clarify it.
    """

    phases: list[str] = []
    for operation in (
        *answer_scope.context_operations,
        *answer_scope.requested_operations,
    ):
        value = str(operation).strip()
        if not value:
            continue
        phases.extend((value, f"{value}阶段"))
    return {
        key: list(dict.fromkeys(values))
        for key, values in {
            "phases": phases,
            "signals": [
                str(value).strip()
                for value in answer_scope.branch_conditions
                if str(value).strip()
            ],
        }.items()
        if values
    }


def build_terminology_search_contract(
    query: str,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """Build bounded search obligations without making retrieval decisions.

    Consumes the new ``query_expansions`` structured field from the resolver
    when available, falling back to the legacy ``resolved_mentions`` loop.
    """

    # ── Prefer the new structured query_expansions when present ──
    qe = resolution.get("query_expansions") or {}
    search_obligations = qe.get("search_obligations") or {}
    required_pairs = search_obligations.get("required_pairs") or []
    optional_expansions = search_obligations.get("optional_expansions") or []
    blocked = qe.get("blocked_expansions") or []
    ambiguous = qe.get("ambiguous_surfaces") or []

    # Build required search groups from structured pairs
    required_groups: list[dict[str, Any]] = []
    seen_groups: set[tuple[str, str]] = set()

    for pair in required_pairs:
        source = str(pair.get("source") or "").strip()
        canonical = str(pair.get("canonical") or "").strip()
        source_key = normalize_term(source)
        canonical_key = normalize_term(canonical)
        if not source_key or not canonical_key or source_key == canonical_key:
            continue
        identity = (source_key, canonical_key)
        if identity in seen_groups:
            continue
        seen_groups.add(identity)
        required_groups.append({
            "source_surface_form": source,
            "canonical_name": canonical,
            "required_terms": [source, canonical],
            "obligation": "search_all",
            "reason": "approved_equivalence",
            "resolution_status": "resolved",
            "can_lock_variant": False,
        })

    # Fallback to legacy resolved_mentions loop when query_expansions absent
    if not required_pairs:

        def append_group(
            *,
            source: str,
            canonical: str,
            concept: dict[str, Any],
            relation_types: Iterable[Any],
            reason: str,
            resolution_status: str = "resolved",
        ) -> None:
            source = str(source or "").strip()
            canonical = str(canonical or "").strip()
            source_key = normalize_term(source)
            canonical_key = normalize_term(canonical)
            if not source_key or not canonical_key or source_key == canonical_key:
                return
            identity = (source_key, canonical_key)
            if identity in seen_groups:
                return
            seen_groups.add(identity)
            group = {
                "source_surface_form": source,
                "canonical_name": canonical,
                "concept_id": str(concept.get("concept_id") or ""),
                "relation_types": sorted({
                    str(value)
                    for value in relation_types
                    if str(value)
                }),
                "required_terms": [source, canonical],
                "obligation": "search_all",
                "reason": reason,
            }
            if reason != "approved_equivalence":
                group.update({
                    "resolution_status": resolution_status,
                    "can_lock_variant": False,
                })
            required_groups.append(group)

        resolved_mentions = [
            mention
            for mention in resolution.get("resolved_mentions") or []
            if isinstance(mention, dict)
        ]
        for mention in resolved_mentions:
            if not isinstance(mention, dict):
                continue
            source = str(mention.get("surface_form") or "").strip()
            concept = mention.get("concept") or {}
            canonical = str(concept.get("canonical_name") or "").strip()
            append_group(
                source=source,
                canonical=canonical,
                concept=concept,
                relation_types=mention.get("relation_types") or [],
                reason="approved_equivalence",
            )

        # Some approved expansions are emitted as a surface form without a
        # source_surface_form (for example ``DL`` is an approved expansion of
        # the canonical ``DL算法``).  When that surface is present in the query,
        # make the source/canonical pair mandatory instead of leaving it as a
        # model choice.  This is generic and does not encode a query name.
        query_key = normalize_term(query)
        for raw in resolution.get("retrieval_expansions") or []:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("authority") or "") != "approved_equivalence":
                continue
            source = str(raw.get("source_surface_form") or raw.get("text") or "")
            if not source or normalize_term(source) not in query_key:
                continue
            source_key = normalize_term(source)
            for mention in resolved_mentions:
                concept = mention.get("concept") or {}
                canonical = str(concept.get("canonical_name") or "")
                if source_key == normalize_term(canonical):
                    continue
                if normalize_term(canonical) not in query_key:
                    continue
                append_group(
                    source=source,
                    canonical=canonical,
                    concept=concept,
                    relation_types=mention.get("relation_types") or [],
                    reason="approved_equivalence_expansion",
                )
                break

        # Ambiguous approved senses must still be searched so the model can see
        # the candidate evidence, but they are explicitly non-locking.
        for raw_mention in resolution.get("ambiguous_mentions") or []:
            if not isinstance(raw_mention, dict):
                continue
            source = str(raw_mention.get("surface_form") or "").strip()
            for ranked in raw_mention.get("candidate_concepts") or []:
                if not isinstance(ranked, dict):
                    continue
                concept = ranked.get("concept") or {}
                append_group(
                    source=source,
                    canonical=str(concept.get("canonical_name") or ""),
                    concept=concept,
                    relation_types=[],
                    reason="ambiguous_candidate",
                    resolution_status="ambiguous",
                )

    # ── Optional expansions: use structured data when available ──
    optional_expansions_out: list[dict[str, Any]] = []
    seen_optional: set[tuple[str, str]] = set()
    required_keys = {
        normalize_term(term)
        for group in required_groups
        for term in group["required_terms"]
    }

    if optional_expansions:
        # Use structured optional expansions from query_expansions
        for raw in optional_expansions:
            text = str(raw.get("text") or "").strip()
            authority = str(raw.get("authority") or "search_hint").strip()
            key = (normalize_term(text), authority)
            if not key[0] or key in seen_optional:
                continue
            if key[0] in required_keys:
                continue
            seen_optional.add(key)
            optional_expansions_out.append({
                "text": text,
                "authority": authority,
                "source_surface_form": "",
                "can_lock_variant": False,
            })
    else:
        # Legacy fallback
        for raw in resolution.get("retrieval_expansions") or []:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            authority = str(raw.get("authority") or "").strip()
            key = (normalize_term(text), authority)
            if not key[0] or key in seen_optional:
                continue
            if authority == "approved_equivalence" and key[0] in required_keys:
                continue
            seen_optional.add(key)
            optional_expansions_out.append({
                "text": text,
                "authority": authority,
                "source_surface_form": str(
                    raw.get("source_surface_form") or ""
                ),
                "can_lock_variant": False,
            })

    unresolved: list[dict[str, Any]] = []
    for field, authority in (
        ("ambiguous_mentions", "approved_equivalence_ambiguous"),
        ("ambiguous_supporting_mentions", "search_hint_ambiguous"),
    ):
        for raw in resolution.get(field) or []:
            if not isinstance(raw, dict):
                continue
            unresolved.append({
                "surface_form": str(raw.get("surface_form") or ""),
                "authority": authority,
                "reason": str(raw.get("reason") or "context_required"),
                "required_context": list(raw.get("required_context") or []),
                "can_lock_variant": False,
            })

    return {
        "schema_version": _CONTRACT_SCHEMA_VERSION,
        "query": str(query),
        "required_search_groups": required_groups,
        "optional_expansions": optional_expansions_out,
        "unresolved_terms": unresolved,
        "authority_policy": {
            "approved_equivalence": (
                "原始表达和规范名均须实际用于搜索；只扩展检索，不证明根因"
            ),
            "search_hint": "可选宽召回；不能锁定 Variant 或证明根因",
            "entity_relation": "可选范围扩展；不能替换 Query 对象",
            "governance_material": (
                "inventory、review queue、人工审核建议只用于治理，"
                "pending/rejected/needs_re_review 不具备运行时解析权限"
            ),
        },
        "runtime_authority_paths": list(_RUNTIME_AUTHORITY_PATHS),
        "governance_only_paths": list(_GOVERNANCE_ONLY_PATHS),
    }


def audit_terminology_search_contract(
    contract: dict[str, Any],
    tool_trace: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Check required equivalent terms against model-directed search calls."""

    searchable_inputs = _searchable_trace_inputs(tool_trace)
    normalized_inputs = [normalize_term(value) for value in searchable_inputs]
    groups: list[dict[str, Any]] = []
    missing_terms: list[str] = []
    for raw_group in contract.get("required_search_groups") or []:
        if not isinstance(raw_group, dict):
            continue
        terms = [
            str(value).strip()
            for value in raw_group.get("required_terms") or []
            if str(value).strip()
        ]
        statuses: list[dict[str, Any]] = []
        for term in terms:
            normalized = normalize_term(term)
            matched = bool(normalized) and any(
                normalized in candidate for candidate in normalized_inputs
            )
            statuses.append({"term": term, "searched": matched})
            if not matched and term not in missing_terms:
                missing_terms.append(term)
        groups.append({
            "source_surface_form": str(
                raw_group.get("source_surface_form") or ""
            ),
            "canonical_name": str(raw_group.get("canonical_name") or ""),
            "terms": statuses,
            "complete": all(item["searched"] for item in statuses),
        })
    return {
        "schema_version": "debug_agent_system.terminology_search_audit.v1",
        "complete": not missing_terms,
        "required_group_count": len(groups),
        "groups": groups,
        "missing_terms": missing_terms,
        "search_input_count": len(searchable_inputs),
    }


def terminology_search_errors(audit: dict[str, Any]) -> list[str]:
    return [
        f"terminology_required_search_missing:{term}"
        for term in audit.get("missing_terms") or []
        if str(term).strip()
    ]


def terminology_governance_authority_errors(
    draft: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    """Reject governance views when they are cited as answer evidence.

    Codex may read these files to understand review state, but pending review
    material cannot close a facet or support an answer claim.
    """

    governance_paths = {
        str(path).strip()
        for path in contract.get("governance_only_paths") or []
        if str(path).strip()
    }
    used: set[str] = set()
    for raw in draft.get("coverage_ledger") or []:
        if not isinstance(raw, dict):
            continue
        used.update(
            str(path).strip()
            for path in raw.get("source_paths") or []
            if (
                str(path).strip() in governance_paths
                or _is_governance_path(str(path).strip())
            )
        )
    answer = str(draft.get("answer_markdown") or "")
    for cited in re.findall(r"【来源：([^】]+)】", answer):
        # The native answer commonly groups several citations in one marker
        # with ``；``.  Inspect each path instead of comparing the whole
        # marker, otherwise a governance path can hide beside a valid source.
        for raw_path in re.split(r"[；;\n,，]+", str(cited)):
            path = str(raw_path).strip().strip("` ")
            if path in governance_paths or _is_governance_path(path):
                used.add(path)
    return [
        f"terminology_governance_material_used_as_evidence:{path}"
        for path in sorted(used)
    ]


def load_terminology_manifest(root: Path) -> dict[str, Any]:
    """Load the reproducibility fields without treating the report as facts."""

    path = root / "terminology/terminology_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing"}
    if not isinstance(value, dict):
        return {"status": "invalid"}
    return {
        "status": "loaded",
        "schema_version": str(value.get("schema_version") or ""),
        "terminology_version": str(value.get("terminology_version") or ""),
        "revision": str(value.get("revision") or ""),
        "concept_count": _safe_count(value.get("concept_count")),
        "expression_count": _safe_count(value.get("expression_count")),
        "sense_count": _safe_count(value.get("sense_count")),
    }


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _searchable_trace_inputs(
    tool_trace: Iterable[dict[str, Any]],
) -> list[str]:
    values: list[str] = []
    for raw in tool_trace:
        if not isinstance(raw, dict) or raw.get("status") == "error":
            continue
        name = str(raw.get("name") or "")
        if name in {"search_text", "list_files"}:
            arguments = raw.get("arguments") or {}
            if isinstance(arguments, dict):
                values.extend(
                    str(arguments.get(field) or "")
                    for field in ("query", "glob", "path_glob")
                    if arguments.get(field)
                )
            continue
        if raw.get("type") == "command_execution":
            command = str(raw.get("command") or "")
            if command and _CLI_SEARCH_COMMAND.search(command):
                values.append(command)
    return list(dict.fromkeys(value for value in values if value))


__all__ = [
    "audit_terminology_search_contract",
    "build_resolver_context",
    "build_terminology_search_contract",
    "load_terminology_manifest",
    "terminology_governance_authority_errors",
    "terminology_search_errors",
]
