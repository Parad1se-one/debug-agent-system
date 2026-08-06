"""Deterministic component builder for reviewed must-link decisions."""

from __future__ import annotations

from typing import Any

from .contracts import canonical_hash


def _edge_pair(edge: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((
        str(edge.get("left_case_ref") or ""),
        str(edge.get("right_case_ref") or ""),
    )))


def build_component_conflicts(
    link_decision: dict[str, Any],
) -> dict[str, Any]:
    """Find cannot edges contradicted by a transitive must-link path."""

    must_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    cannot_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    adjacency: dict[str, set[str]] = {}
    for edge in link_decision.get("edge_decisions") or []:
        if not isinstance(edge, dict):
            continue
        pair = _edge_pair(edge)
        if not all(pair) or pair[0] == pair[1]:
            continue
        decision = str(edge.get("decision") or "")
        if decision == "must_link":
            must_by_pair[pair] = edge
            adjacency.setdefault(pair[0], set()).add(pair[1])
            adjacency.setdefault(pair[1], set()).add(pair[0])
        elif decision == "cannot_link":
            cannot_by_pair[pair] = edge

    conflicts: list[dict[str, Any]] = []
    for index, (pair, cannot_edge) in enumerate(
        sorted(cannot_by_pair.items()),
        1,
    ):
        left, right = pair
        queue: list[str] = [left]
        previous: dict[str, str | None] = {left: None}
        while queue and right not in previous:
            current = queue.pop(0)
            for neighbor in sorted(adjacency.get(current) or []):
                if neighbor in previous:
                    continue
                previous[neighbor] = current
                queue.append(neighbor)
        if right not in previous:
            continue
        path_nodes: list[str] = []
        current: str | None = right
        while current is not None:
            path_nodes.append(current)
            current = previous[current]
        path_nodes.reverse()
        must_path = [
            must_by_pair[tuple(sorted((path_nodes[offset], path_nodes[offset + 1])))]
            for offset in range(len(path_nodes) - 1)
        ]
        conflicts.append({
            "conflict_ref": f"conflict:{index:03d}",
            "cannot_link_edge": cannot_edge,
            "must_link_path": must_path,
            "case_refs": sorted(set(path_nodes)),
        })
    output = {
        "schema_version": "w7.component_conflicts.v1",
        "conflicts": conflicts,
    }
    output["conflicts_hash"] = canonical_hash(output)
    return output


def apply_component_consistency_decision(
    link_decision: dict[str, Any],
    consistency_decision: dict[str, Any],
) -> dict[str, Any]:
    """Apply evidence-bounded cannot→possible overrides."""

    weak_pairs = {
        tuple(sorted((
            str(item.get("left_case_ref") or ""),
            str(item.get("right_case_ref") or ""),
        )))
        for item in consistency_decision.get("conflict_decisions") or []
        if isinstance(item, dict)
        and str(item.get("decision") or "") == "weak_cannot"
    }
    decisions: list[dict[str, Any]] = []
    for edge in link_decision.get("edge_decisions") or []:
        if not isinstance(edge, dict):
            continue
        pair = _edge_pair(edge)
        if (
            str(edge.get("decision") or "") == "cannot_link"
            and pair in weak_pairs
        ):
            decisions.append({
                **edge,
                "decision": "possible_link",
                "local_override_reason": (
                    "component_consistency_weak_cannot"
                ),
                "original_decision": "cannot_link",
            })
        else:
            decisions.append(edge)
    output = {
        "schema_version": "w7.trace_link_decision.v1",
        "edge_decisions": decisions,
        "uncertainties": [
            *list(link_decision.get("uncertainties") or []),
            *[
                f"component_consistency:{value}"
                for value in (
                    consistency_decision.get("uncertainties") or []
                )
            ],
        ],
    }
    output["decision_hash"] = canonical_hash(output)
    return output


def apply_candidate_edge_safety_guards(
    link_decision: dict[str, Any],
    candidate_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prevent semantic decisions from crossing explicit identity shifts.

    A phrase such as "another device" is a boundary signal, but not enough to
    assert a hard cannot-link: a shared Jira or attachment may later prove a
    replacement hand-off. Therefore only automatic must-link materialization
    is blocked and the edge remains available for human review.
    """

    blockers_by_pair = {
        _edge_pair(edge): [
            str(value)
            for value in edge.get("auto_merge_blockers") or []
            if str(value)
        ]
        for edge in candidate_edges
        if isinstance(edge, dict)
        and str(edge.get("edge_class") or "") != "identity_edge"
        and edge.get("auto_merge_blockers")
    }
    decisions: list[dict[str, Any]] = []
    for edge in link_decision.get("edge_decisions") or []:
        if not isinstance(edge, dict):
            continue
        blockers = blockers_by_pair.get(_edge_pair(edge), [])
        if (
            str(edge.get("decision") or "") == "must_link"
            and blockers
        ):
            decisions.append({
                **edge,
                "decision": "possible_link",
                "relation_hint": "",
                "original_decision": "must_link",
                "local_override_reason": (
                    "candidate_identity_discontinuity_guard"
                ),
                "auto_merge_blockers": blockers,
            })
        else:
            decisions.append(edge)
    output = {
        "schema_version": "w7.trace_link_decision.v1",
        "edge_decisions": decisions,
        "uncertainties": list(link_decision.get("uncertainties") or []),
    }
    output["decision_hash"] = canonical_hash(output)
    return output


def build_component_bridge_candidates(
    components: dict[str, Any],
    link_decision: dict[str, Any],
    *,
    max_component_size: int = 12,
) -> dict[str, Any]:
    """Select safe cross-component possible links for semantic re-review."""

    component_by_case: dict[str, dict[str, Any]] = {}
    for component in components.get("components") or []:
        if not isinstance(component, dict):
            continue
        for case_ref in component.get("case_refs") or []:
            component_by_case[str(case_ref)] = component
    cannot_pairs = {
        _edge_pair(edge)
        for edge in link_decision.get("edge_decisions") or []
        if isinstance(edge, dict)
        and str(edge.get("decision") or "") == "cannot_link"
    }
    candidates: list[dict[str, Any]] = []
    for edge in link_decision.get("edge_decisions") or []:
        if (
            not isinstance(edge, dict)
            or str(edge.get("decision") or "") != "possible_link"
            or bool(edge.get("auto_merge_blockers"))
        ):
            continue
        left, right = _edge_pair(edge)
        left_component = component_by_case.get(left)
        right_component = component_by_case.get(right)
        if (
            not left_component
            or not right_component
            or left_component is right_component
        ):
            continue
        left_refs = {
            str(value)
            for value in left_component.get("case_refs") or []
            if str(value)
        }
        right_refs = {
            str(value)
            for value in right_component.get("case_refs") or []
            if str(value)
        }
        blockers = sorted(
            pair for pair in cannot_pairs
            if (
                pair[0] in left_refs and pair[1] in right_refs
            ) or (
                pair[1] in left_refs and pair[0] in right_refs
            )
        )
        if blockers:
            continue
        if len(left_refs) + len(right_refs) > max_component_size:
            continue
        candidates.append({
            "left_case_ref": left,
            "right_case_ref": right,
            "left_component_ref": str(
                left_component.get("component_ref") or ""
            ),
            "right_component_ref": str(
                right_component.get("component_ref") or ""
            ),
            "left_component_case_refs": sorted(left_refs),
            "right_component_case_refs": sorted(right_refs),
            "candidate_edge": edge,
        })
    candidates.sort(
        key=lambda item: (
            item["left_case_ref"],
            item["right_case_ref"],
        )
    )
    output = {
        "schema_version": "w7.component_bridge_candidates.v1",
        "candidates": candidates,
        "max_component_size": max(1, int(max_component_size)),
    }
    output["candidates_hash"] = canonical_hash(output)
    return output


def apply_component_bridge_decision(
    link_decision: dict[str, Any],
    bridge_decision: dict[str, Any],
) -> dict[str, Any]:
    """Apply bounded possible→must/cannot component bridge decisions."""

    patches = {
        tuple(sorted((
            str(item.get("left_case_ref") or ""),
            str(item.get("right_case_ref") or ""),
        ))): item
        for item in bridge_decision.get("bridge_decisions") or []
        if isinstance(item, dict)
    }
    values: list[dict[str, Any]] = []
    for edge in link_decision.get("edge_decisions") or []:
        if not isinstance(edge, dict):
            continue
        pair = _edge_pair(edge)
        patch = patches.get(pair)
        if (
            str(edge.get("decision") or "") != "possible_link"
            or patch is None
        ):
            values.append(edge)
            continue
        bridge_value = str(patch.get("decision") or "")
        if bridge_value == "promote_must":
            decision = "must_link"
        elif bridge_value == "confirm_cannot":
            decision = "cannot_link"
        else:
            values.append(edge)
            continue
        values.append({
            **edge,
            "decision": decision,
            "original_decision": "possible_link",
            "local_override_reason": (
                f"component_bridge_{bridge_value}"
            ),
            "bridge_evidence_message_ids": list(
                patch.get("evidence_message_ids") or []
            ),
            "bridge_reasons": list(patch.get("reasons") or []),
        })
    output = {
        "schema_version": "w7.trace_link_decision.v1",
        "edge_decisions": values,
        "uncertainties": [
            *list(link_decision.get("uncertainties") or []),
            *[
                f"component_bridge:{value}"
                for value in bridge_decision.get("uncertainties") or []
            ],
        ],
    }
    output["decision_hash"] = canonical_hash(output)
    return output


def build_trace_components(
    graph: dict[str, Any],
    link_decision: dict[str, Any],
    *,
    core_case_refs: set[str] | None = None,
    max_component_size: int = 12,
) -> tuple[dict[str, Any], list[str]]:
    nodes = [
        str(value) for value in graph.get("node_refs") or [] if str(value)
    ]
    parent = {node: node for node in nodes}
    sizes = {node: 1 for node in nodes}
    members = {node: {node} for node in nodes}
    limit = max(1, int(max_component_size))

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep = min(left_root, right_root)
        merge = max(left_root, right_root)
        parent[merge] = keep
        sizes[keep] += sizes[merge]
        members[keep].update(members[merge])
        del members[merge]

    possible_edges: list[dict[str, Any]] = []
    cannot_edges: list[dict[str, Any]] = []
    downgraded_edges: list[dict[str, Any]] = []
    issues: list[str] = []
    valid_edges: list[dict[str, Any]] = []
    for edge in link_decision.get("edge_decisions") or []:
        if not isinstance(edge, dict):
            continue
        left = str(edge.get("left_case_ref") or "")
        right = str(edge.get("right_case_ref") or "")
        if left not in parent or right not in parent:
            issues.append(f"component_unknown_edge:{left}:{right}")
            continue
        decision = str(edge.get("decision") or "")
        valid_edges.append(edge)
        if decision == "possible_link":
            possible_edges.append(edge)
        elif decision == "cannot_link":
            cannot_edges.append(edge)

    cannot_pairs = {
        tuple(sorted((
            str(edge.get("left_case_ref") or ""),
            str(edge.get("right_case_ref") or ""),
        )))
        for edge in cannot_edges
    }

    def blocking_cannot_pairs(
        left_root: str,
        right_root: str,
    ) -> list[list[str]]:
        return [
            [left_member, right_member]
            for left_member in sorted(members[left_root])
            for right_member in sorted(members[right_root])
            if tuple(sorted((left_member, right_member))) in cannot_pairs
        ]

    must_edges = sorted(
        (
            edge for edge in valid_edges
            if str(edge.get("decision") or "") == "must_link"
        ),
        key=lambda edge: (
            min(
                str(edge.get("left_case_ref") or ""),
                str(edge.get("right_case_ref") or ""),
            ),
            max(
                str(edge.get("left_case_ref") or ""),
                str(edge.get("right_case_ref") or ""),
            ),
            canonical_hash(edge),
        ),
    )
    for edge in must_edges:
        left = str(edge.get("left_case_ref") or "")
        right = str(edge.get("right_case_ref") or "")
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        blockers = blocking_cannot_pairs(left_root, right_root)
        downgrade_reason = ""
        if blockers:
            downgrade_reason = "cannot_link_conflict"
        elif sizes[left_root] + sizes[right_root] > limit:
            downgrade_reason = "component_size_limit"
        if downgrade_reason:
            downgraded = {
                **edge,
                "decision": "possible_link",
                "local_downgrade_reason": downgrade_reason,
            }
            if blockers:
                downgraded["blocking_cannot_pairs"] = blockers
            possible_edges.append(downgraded)
            downgraded_edges.append(downgraded)
            continue
        union(left, right)

    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(find(node), []).append(node)
    components = [
        {
            "component_ref": f"component:{index:03d}",
            "case_refs": sorted(case_refs),
        }
        for index, case_refs in enumerate(
            sorted(grouped.values(), key=lambda values: tuple(sorted(values))),
            1,
        )
    ]
    for component in components:
        if len(component["case_refs"]) > limit:
            issues.append(
                "component_too_large:"
                f"{component['component_ref']}:{len(component['case_refs'])}:{limit}"
            )
        component_members = set(component["case_refs"])
        internal_conflicts = sorted(
            pair for pair in cannot_pairs
            if set(pair).issubset(component_members)
        )
        if internal_conflicts:
            issues.extend(
                "component_cannot_link_conflict:"
                f"{component['component_ref']}:{left}:{right}"
                for left, right in internal_conflicts
            )
    core = set(core_case_refs or nodes)
    if core_case_refs is not None:
        for component in components:
            if not core.intersection(component["case_refs"]):
                issues.append(
                    f"component_without_core:{component['component_ref']}"
                )
    edge_sort_key = lambda edge: (
        min(
            str(edge.get("left_case_ref") or ""),
            str(edge.get("right_case_ref") or ""),
        ),
        max(
            str(edge.get("left_case_ref") or ""),
            str(edge.get("right_case_ref") or ""),
        ),
        str(edge.get("decision") or ""),
        str(edge.get("local_downgrade_reason") or ""),
        canonical_hash(edge),
    )
    possible_edges.sort(key=edge_sort_key)
    cannot_edges.sort(key=edge_sort_key)
    downgraded_edges.sort(key=edge_sort_key)
    output = {
        "schema_version": "w7.trace_components.v1",
        "components": components,
        "possible_edges": possible_edges,
        "cannot_link_edges": cannot_edges,
        "downgraded_edges": downgraded_edges,
        "conflict_edges": [
            edge for edge in downgraded_edges
            if edge.get("local_downgrade_reason")
            == "cannot_link_conflict"
        ],
        "overflow_edges": [
            edge for edge in downgraded_edges
            if edge.get("local_downgrade_reason")
            == "component_size_limit"
        ],
        "max_component_size": limit,
    }
    output["components_hash"] = canonical_hash(output)
    return output, sorted(set(issues))
