"""Content-addressed evidence graph used by Read Runtime v3."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Iterable

from .contracts import EvidenceLink, EvidenceRecord, SourceAnchor, to_jsonable


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EvidenceFabric:
    """Append-only evidence records plus typed provenance links."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._links: dict[str, EvidenceLink] = {}
        self._provider_index: dict[str, list[str]] = {}

    def create_record(
        self,
        *,
        kind: str,
        provider: str,
        source_ref: str,
        assertion: str,
        summary: str,
        content: Any = None,
        anchors: Iterable[SourceAnchor] = (),
        confidence: float = 1.0,
        source_revision: str = "",
        parser_version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        digest = content_hash(content if content is not None else summary)
        identity = {
            "kind": kind,
            "provider": provider,
            "source_ref": source_ref,
            "content_sha256": digest,
            "anchors": [asdict(anchor) for anchor in anchors],
            "source_revision": source_revision,
            "parser_version": parser_version,
        }
        evidence_id = "ev3:" + content_hash(identity)[:24]
        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,  # type: ignore[arg-type]
            provider=provider,
            source_ref=source_ref,
            assertion=assertion,  # type: ignore[arg-type]
            summary=str(summary or "").strip(),
            content=content,
            content_sha256=digest,
            anchors=list(anchors),
            confidence=max(0.0, min(1.0, float(confidence))),
            source_revision=source_revision,
            parser_version=parser_version,
            metadata=dict(metadata or {}),
        )
        return self.add_record(record)

    def add_record(self, record: EvidenceRecord) -> EvidenceRecord:
        previous = self._records.get(record.evidence_id)
        if previous is not None:
            if canonical_json(previous) != canonical_json(record):
                raise ValueError(f"evidence_identity_collision:{record.evidence_id}")
            return previous
        self._records[record.evidence_id] = record
        self._provider_index.setdefault(record.provider, []).append(record.evidence_id)
        return record

    def link(
        self,
        relation: str,
        from_evidence_id: str,
        to_evidence_id: str,
        *,
        explanation: str = "",
        confidence: float = 1.0,
    ) -> EvidenceLink:
        for evidence_id in (from_evidence_id, to_evidence_id):
            if evidence_id not in self._records:
                raise KeyError(f"unknown_evidence:{evidence_id}")
        identity = {
            "relation": relation,
            "from": from_evidence_id,
            "to": to_evidence_id,
            "explanation": explanation,
        }
        link = EvidenceLink(
            link_id="el3:" + content_hash(identity)[:24],
            relation=relation,  # type: ignore[arg-type]
            from_evidence_id=from_evidence_id,
            to_evidence_id=to_evidence_id,
            explanation=explanation,
            confidence=max(0.0, min(1.0, float(confidence))),
        )
        self._links.setdefault(link.link_id, link)
        return self._links[link.link_id]

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def records(
        self,
        *,
        provider: str = "",
        kind: str = "",
    ) -> list[EvidenceRecord]:
        values = list(self._records.values())
        if provider:
            values = [record for record in values if record.provider == provider]
        if kind:
            values = [record for record in values if record.kind == kind]
        return values

    def links(self) -> list[EvidenceLink]:
        return list(self._links.values())

    def snapshot(self) -> dict[str, Any]:
        records = [to_jsonable(item) for item in self._records.values()]
        links = [to_jsonable(item) for item in self._links.values()]
        payload = {"records": records, "links": links}
        return {
            "schema_version": "debug_agent_system.evidence_fabric_snapshot.v3",
            "record_count": len(records),
            "link_count": len(links),
            "providers": {
                provider: len(ids)
                for provider, ids in sorted(self._provider_index.items())
            },
            "fingerprint": content_hash(payload),
            **payload,
        }

