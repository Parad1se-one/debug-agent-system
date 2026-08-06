"""Sparse, deterministic candidate graph for W7b model adjudication."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from .contracts import canonical_hash, dedupe_strings


_ANCHOR_RE = re.compile(
    r"\b(?:SMTAOITS|TEST)-\d+\b|"
    r"\b(?:SI|AOI)[A-Z0-9_-]{4,}\b|"
    r"\b[A-Za-z0-9_-]{4,}\.(?:proj|dmp|dlog|log)\b",
    re.IGNORECASE,
)
_DEVICE_RE = re.compile(
    r"\b(?:SI|AOI)[A-Z0-9_-]{4,}\b|"
    r"(?<![A-Za-z0-9])\d{3,5}T(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DEVICE_IDENTITY_SHIFT_RE = re.compile(
    r"(?:另|另外|其它|其他)一台(?:设备|机器|机台)?"
    r"|新交付的?一台(?:设备|机器|机台|\d{3,5}T)?"
    r"|(?:不同|另一)(?:设备|机器|机台)"
)


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except ValueError:
        try:
            return datetime.strptime(text[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            return None


def _text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "title",
            "fault_summary",
            "problem_summary",
            "device_scope",
            "site_scope",
            "channel_site_scopes",
            "site_scope_provenance",
        )
    )


def _bigrams(value: str) -> set[str]:
    compact = re.sub(r"[\s，。；：、,.!?！？:;()（）\[\]【】]+", "", value.lower())
    return {
        compact[index:index + 2]
        for index in range(max(0, len(compact) - 1))
        if len(compact[index:index + 2]) == 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _prepare(item: dict[str, Any], index: int) -> dict[str, Any]:
    text = _text(item)
    jira = {
        value.upper()
        for value in dedupe_strings(item.get("jira_keys") or [])
    } | {
        value.upper()
        for value in re.findall(r"\b(?:SMTAOITS|TEST)-\d+\b", text, re.I)
    }
    attachments = set(dedupe_strings(
        item.get("attachment_ids")
        or item.get("attachment_message_ids")
        or []
    ))
    messages = set(dedupe_strings([
        *(item.get("source_message_ids") or []),
        *(item.get("evidence_message_ids") or []),
    ]))
    families = set(dedupe_strings(
        item.get("family_ids")
        or [item.get("family_id")]
    ))
    anchors = {value.upper() for value in _ANCHOR_RE.findall(text)}
    devices = {value.upper() for value in _DEVICE_RE.findall(text)}
    explicit_device = str(item.get("device_scope") or "").strip()
    if explicit_device:
        devices.add(explicit_device.upper())
    sites = {
        str(value).strip().upper()
        for value in [
            *(item.get("site_scopes") or []),
            item.get("site_scope"),
        ]
        if str(value or "").strip()
    }
    return {
        "index": index,
        "ref": str(
            item.get("case_ref")
            or item.get("case_item_ref")
            or item.get("fragment_ref")
            or f"case-{index + 1}"
        ),
        "chat_id": str(item.get("chat_id") or ""),
        "text": text,
        "bigrams": _bigrams(text),
        "jira": jira,
        "attachments": attachments,
        "messages": messages,
        "families": families,
        "anchors": anchors,
        "devices": devices,
        "device_identity_shift": bool(
            _DEVICE_IDENTITY_SHIFT_RE.search(text)
        ),
        "sites": sites,
        "parent_episode_id": str(item.get("parent_episode_id") or ""),
        "time": _parse_time(
            item.get("start_time")
            or item.get("create_time")
            or item.get("time_span")
        ),
    }


def build_sparse_candidate_graph(
    case_items: list[dict[str, Any]],
    *,
    top_k: int = 4,
    strong_threshold: float = 6.0,
    weak_threshold: float = 1.0,
) -> dict[str, Any]:
    prepared = [
        _prepare(item, index)
        for index, item in enumerate(case_items)
        if isinstance(item, dict)
    ]
    identity_edges: list[dict[str, Any]] = []
    scored_by_node: dict[str, list[dict[str, Any]]] = {
        item["ref"]: [] for item in prepared
    }
    for left_index, left in enumerate(prepared):
        for right in prepared[left_index + 1:]:
            reasons: list[str] = []
            shared_jira = left["jira"] & right["jira"]
            shared_attachments = left["attachments"] & right["attachments"]
            shared_messages = left["messages"] & right["messages"]
            shared_families = left["families"] & right["families"]
            shared_devices = left["devices"] & right["devices"]
            shared_sites = left["sites"] & right["sites"]
            shared_anchors = left["anchors"] & right["anchors"]
            semantic = _jaccard(left["bigrams"], right["bigrams"])
            score = semantic * 4.0
            if shared_devices:
                score += 4.0
                reasons.append(
                    "shared_device:" + ",".join(sorted(shared_devices))
                )
            if shared_families:
                score += 4.0
                reasons.append(
                    "shared_family:" + ",".join(sorted(shared_families))
                )
            if shared_sites:
                score += 1.0
                reasons.append(
                    "shared_site:" + ",".join(sorted(shared_sites))
                )
            if shared_messages:
                # One chat message may contain several independently
                # extracted atomic cases (for example a daily report listing
                # camera recovery and an unrelated missed defect). Shared
                # provenance improves recall but is not trace identity.
                score += 1.5
                reasons.append(
                    "shared_message:" + ",".join(sorted(shared_messages))
                )
            if (
                left["parent_episode_id"]
                and left["parent_episode_id"]
                == right["parent_episode_id"]
            ):
                # A legacy episode is also allowed to contain multiple atomic
                # cases. Keep it as a weak retrieval hint only.
                score += 0.5
                reasons.append("shared_parent_episode")
            if shared_anchors:
                score += 3.0
                reasons.append(
                    "shared_anchor:" + ",".join(sorted(shared_anchors))
                )
            if left["chat_id"] and left["chat_id"] == right["chat_id"]:
                score += 0.5
            if left["time"] and right["time"]:
                hours = abs((right["time"] - left["time"]).total_seconds()) / 3600
                if hours <= 72:
                    score += 1.5
                    reasons.append(f"within_hours:{round(hours, 1)}")
                elif hours <= 24 * 14:
                    score += 0.5
            if semantic >= 0.20:
                reasons.append(f"semantic_jaccard:{semantic:.3f}")
            edge = {
                "left_case_ref": left["ref"],
                "right_case_ref": right["ref"],
                "score": round(score, 4),
                "reasons": reasons,
                "auto_merge_blockers": (
                    ["device_identity_shift_without_shared_device"]
                    if (
                        (
                            left["device_identity_shift"]
                            or right["device_identity_shift"]
                        )
                        and not shared_devices
                    )
                    else []
                ),
            }
            if shared_jira or shared_attachments:
                identity_reasons = list(reasons)
                if shared_jira:
                    identity_reasons.append(
                        "shared_jira:" + ",".join(sorted(shared_jira))
                    )
                if shared_attachments:
                    identity_reasons.append(
                        "shared_attachment:" + ",".join(
                            sorted(shared_attachments)
                        )
                    )
                identity_edges.append({
                    **edge,
                    "edge_class": "identity_edge",
                    "score": max(edge["score"], 10.0),
                    "reasons": identity_reasons,
                    "requires_adjudication": True,
                })
                continue
            if score < weak_threshold:
                continue
            edge["edge_class"] = (
                "strong_semantic_edge"
                if score >= strong_threshold
                else "weak_retrieval_edge"
            )
            # Every selected semantic edge is only a retrieval candidate.
            # Weak edges must still be adjudicated; otherwise lowering recall
            # thresholds has no effect on the assembled graph.
            edge["requires_adjudication"] = True
            scored_by_node[left["ref"]].append(edge)
            scored_by_node[right["ref"]].append(edge)
    selected_keys: set[tuple[str, str]] = set()
    for values in scored_by_node.values():
        for edge in sorted(
            values,
            key=lambda item: (-float(item["score"]), item["right_case_ref"]),
        )[:max(0, int(top_k))]:
            selected_keys.add((
                str(edge["left_case_ref"]),
                str(edge["right_case_ref"]),
            ))
    semantic_edges = [
        edge
        for values in scored_by_node.values()
        for edge in values
        if (
            str(edge["left_case_ref"]),
            str(edge["right_case_ref"]),
        ) in selected_keys
    ]
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in [*identity_edges, *semantic_edges]:
        key = (str(edge["left_case_ref"]), str(edge["right_case_ref"]))
        current = deduped.get(key)
        if current is None or float(edge["score"]) > float(current["score"]):
            deduped[key] = edge
    edges = sorted(
        deduped.values(),
        key=lambda item: (
            item["left_case_ref"],
            item["right_case_ref"],
        ),
    )
    output = {
        "schema_version": "w7.sparse_candidate_graph.v4",
        "node_refs": [item["ref"] for item in prepared],
        "edges": edges,
        "config": {
            "top_k": int(top_k),
            "strong_threshold": strong_threshold,
            "weak_threshold": weak_threshold,
            "shared_message_weight": 1.5,
            "shared_parent_episode_weight": 0.5,
            "shared_site_weight": 1.0,
        },
        "stats": {
            "nodes": len(prepared),
            "edges": len(edges),
            "identity_edges": sum(
                item["edge_class"] == "identity_edge" for item in edges
            ),
            "strong_edges": sum(
                item["edge_class"] == "strong_semantic_edge"
                for item in edges
            ),
            "weak_edges": sum(
                item["edge_class"] == "weak_retrieval_edge"
                for item in edges
            ),
        },
    }
    output["graph_hash"] = canonical_hash(output)
    return output
