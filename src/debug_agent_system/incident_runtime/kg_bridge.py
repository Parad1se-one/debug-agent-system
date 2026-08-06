"""Stable-anchor bridge from incident observations to immutable KG_v2."""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.read_model import KGV2ReadModel, V2Candidate

from .contracts import DiagnosticEvent, StackTrace

_ADDRESS = re.compile(r"^(?:0x)?[0-9a-fA-F]{8,16}$")
_BUILD_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[^\s]+[/\\]){2,}")
_WEAK_ANCHORS = {
    "in", "at", "error", "exception", "diagnostic_event", "unknown",
    "symv", "cv", "cuda", "file", "line",
}
_GENERIC_EVENT_KINDS = {
    "diagnostic_event", "timeout", "reset", "crash", "exception",
    "process_start", "process_exit",
}


class IncidentKGBridge:
    schema_version = "debug_agent_system.incident_kg_retrieval.v1"

    def __init__(self, read_model: KGV2ReadModel) -> None:
        self.read_model = read_model

    def retrieve(
        self,
        query: str,
        events: Iterable[DiagnosticEvent],
        stack_traces: Iterable[StackTrace],
        *,
        limit: int = 10,
    ) -> dict[str, Any]:
        anchors = self.extract_anchors(events, stack_traces)
        retrieval_query = " ".join(
            _dedupe([
                _sanitize_query(query),
                *[str(item["value"]) for item in anchors if item["stability"] != "volatile"],
            ])
        )[:6000]
        candidates = self.read_model.search_variants(retrieval_query, limit=max(1, min(limit, 20)))
        retrieval = self.read_model.last_retrieval or {}
        return {
            "schema_version": self.schema_version,
            "query": query,
            "retrieval_query": retrieval_query,
            "anchors": anchors,
            "candidates": [self._candidate(item, anchors) for item in candidates],
            "supporting_chunks": [
                _bounded_chunk(item)
                for item in retrieval.get("chunks") or []
                if isinstance(item, dict)
            ][:64],
            "paths": list(retrieval.get("paths") or [])[:80],
            "trace": dict(retrieval.get("trace") or {}),
            "policy": {
                "volatile_anchors_are_required_facets": False,
                "canonical_kg_mutated": False,
                "similar_cases_are_formal_knowledge": False,
            },
        }

    @staticmethod
    def extract_anchors(
        events: Iterable[DiagnosticEvent],
        stack_traces: Iterable[StackTrace],
    ) -> list[dict[str, Any]]:
        by_key: dict[tuple[str, str], dict[str, Any]] = {}

        def add(kind: str, value: str, stability: str, evidence_ids: Iterable[str]) -> None:
            normalized = str(value or "").strip()
            if not normalized or normalized.lower() in _WEAK_ANCHORS:
                return
            if _ADDRESS.fullmatch(normalized) or _BUILD_PATH.search(normalized):
                stability = "volatile"
            key = (kind, normalized.lower())
            item = by_key.setdefault(key, {
                "kind": kind,
                "value": normalized,
                "stability": stability,
                "evidence_ids": [],
            })
            item["evidence_ids"] = _dedupe([*item["evidence_ids"], *evidence_ids])

        for event in events:
            for code in event.error_codes:
                add("error_code", code, "stable", event.evidence_ids)
            add("event_kind", event.event_kind, "stable", event.evidence_ids)
            add("component", event.component, "contextual", event.evidence_ids)
            add("module", event.module, "contextual", event.evidence_ids)
            add("function", event.function, "stable", event.evidence_ids)
        for trace in stack_traces:
            for frame in trace.frames:
                add("module", frame.module, "contextual", frame.evidence_ids)
                add("function", frame.function, "stable", frame.evidence_ids)
                add("source_file", frame.source_file, "volatile", frame.evidence_ids)
                add("source_line", str(frame.line or ""), "volatile", frame.evidence_ids)
                add("address", frame.address, "volatile", frame.evidence_ids)
        rank = {"stable": 0, "contextual": 1, "volatile": 2}
        # Python's sort is stable: within the same stability/kind retain first
        # observation order so the primary incident signature is not displaced
        # alphabetically by a later secondary warning code.
        return sorted(
            by_key.values(),
            key=lambda item: (rank[item["stability"]], item["kind"]),
        )

    def _candidate(self, candidate: V2Candidate, anchors: list[dict[str, Any]]) -> dict[str, Any]:
        searchable = " ".join([
            candidate.family_label,
            candidate.variant_label,
            *candidate.matched_fields,
            *[str(chunk.get("text") or "") for chunk in candidate.supporting_chunks[:12]],
        ]).lower()
        matched_anchors = [
            item
            for item in anchors
            if item["stability"] != "volatile"
            and len(str(item["value"])) >= 3
            and str(item["value"]).lower() not in _WEAK_ANCHORS
            and str(item["value"]).lower() in searchable
        ]
        strong = [
            item for item in matched_anchors
            if item["kind"] in {"error_code", "function"}
        ]
        discriminative_context = {
            (str(item["kind"]), str(item["value"]).lower())
            for item in matched_anchors
            if item["kind"] in {"component", "module"}
            or (
                item["kind"] == "event_kind"
                and str(item["value"]).lower() not in _GENERIC_EVENT_KINDS
            )
        }
        if not strong and len(discriminative_context) < 2:
            # One generic overlap (for example "timeout") is insufficient to
            # turn a broad retrieval result into an incident hypothesis.
            matched_anchors = []
        support_evidence_ids = _dedupe(
            evidence_id
            for item in matched_anchors
            for evidence_id in item["evidence_ids"]
        )
        required_info: list[dict[str, Any]] = []
        try:
            plan = self.read_model.compile_plan(candidate.family_id, candidate.variant_id)
            required_info = self.read_model.required_info(plan.required_info_ids)
        except (KeyError, ValueError):
            required_info = []
        return {
            **asdict(candidate),
            "matched_incident_anchors": matched_anchors,
            "support_evidence_ids": support_evidence_ids,
            "required_info": required_info,
            "source_ids": _dedupe(candidate.evidence_ids),
        }


def _bounded_chunk(value: dict[str, Any]) -> dict[str, Any]:
    item = dict(value)
    text = str(item.get("text") or "")
    if len(text) > 4000:
        item["text"] = text[:4000]
        item["text_truncated"] = True
    return item


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _sanitize_query(value: str) -> str:
    """Keep user intent and stable signatures; discard volatile trace noise."""

    lines: list[str] = []
    for raw in str(value or "").splitlines():
        line = _BUILD_PATH.sub(" ", raw)
        line = re.sub(
            r"\b(?:0x)?(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,64}\b",
            " ",
            line,
        )
        line = re.sub(r"^\s*\d+#.*$", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and len(line) >= 2:
            lines.append(line[:500])
    return " ".join(lines[:20])


__all__ = ["IncidentKGBridge"]
