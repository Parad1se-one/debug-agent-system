"""Read-only provider adapters for the frozen read-side capabilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from debug_agent_system.kg_raw_codex.pipeline import CORPUS_ROOTS, CorpusReadTools
from debug_agent_system.knowledge_v2.source_chunk_builder import _read_blocks
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import (
    kg_v2_graph_revision,
    kg_v2_source_revision,
)

from .contracts import (
    HypothesisRecord,
    ReadRequest,
    ReadTask,
    SourceAnchor,
    to_jsonable,
)
from .fabric import EvidenceFabric, content_hash


class Provider(Protocol):
    name: str

    def collect(
        self,
        request: ReadRequest,
        task: ReadTask,
        fabric: EvidenceFabric,
    ) -> dict[str, Any]: ...


class RequestContextProvider:
    """Normalize caller-supplied context into the common evidence fabric.

    This provider does not infer a diagnosis or trace boundary.  It only makes
    chat turns, log summaries and explicitly supplied source-only records
    available through the same auditable evidence contract as KG, raw files
    and incident packages.
    """

    name = "request_context"

    def collect(
        self,
        request: ReadRequest,
        task: ReadTask,
        fabric: EvidenceFabric,
    ) -> dict[str, Any]:
        record_ids: list[str] = []
        message_ids: list[str] = []
        attachment_ids: list[str] = []

        if request.log_summary:
            record = fabric.create_record(
                kind="diagnostic_event",
                provider=self.name,
                source_ref="request:log_summary",
                assertion="observed",
                summary=str(
                    request.log_summary.get("summary")
                    or request.log_summary.get("message")
                    or "Caller-supplied log summary"
                )[:500],
                content=request.log_summary,
                metadata={"request_context_kind": "log_summary"},
            )
            record_ids.append(record.evidence_id)

        for index, turn in enumerate(request.chat_history, start=1):
            if not isinstance(turn, dict):
                continue
            native_id = str(turn.get("message_id") or turn.get("id") or f"turn:{index}")
            text = str(turn.get("content") or turn.get("text") or "")
            record = fabric.create_record(
                kind="diagnostic_event",
                provider=self.name,
                source_ref=native_id,
                assertion="observed",
                summary=text[:500] or f"Conversation turn {index}",
                content=turn,
                anchors=[SourceAnchor(
                    source_id=native_id,
                    timestamp=str(turn.get("create_time") or turn.get("timestamp") or ""),
                )],
                metadata={
                    "request_context_kind": "chat_turn",
                    "role": str(turn.get("role") or ""),
                },
            )
            record_ids.append(record.evidence_id)
            message_ids.append(native_id)

        source_context = request.routing_context.get("source_only_context")
        if isinstance(source_context, dict):
            context_meta = dict(
                request.routing_context.get("source_only_context_ref") or {}
            )
            context_id = str(
                source_context.get("case_id")
                or source_context.get("input_evidence_sha256")
                or context_meta.get("sha256")
                or "source-only-context"
            )
            root = fabric.create_record(
                kind="source_artifact",
                provider=self.name,
                source_ref=context_id,
                assertion="observed",
                summary=f"Source-only context {context_id}",
                content={
                    key: source_context.get(key)
                    for key in (
                        "schema_version", "case_id", "source_kind", "source_file",
                        "label_visibility", "session_start", "session_end_exclusive",
                        "analysis_window", "input_evidence_sha256", "messages_sha256",
                    )
                    if source_context.get(key) is not None
                },
                anchors=[SourceAnchor(path=str(context_meta.get("path") or ""))],
                metadata={
                    "request_context_kind": "source_only_context",
                    "source_only": True,
                    "label_visibility": str(source_context.get("label_visibility") or ""),
                    "sha256": str(context_meta.get("sha256") or ""),
                },
            )
            record_ids.append(root.evidence_id)
            for index, message in enumerate(source_context.get("messages") or [], start=1):
                if not isinstance(message, dict):
                    continue
                native_id = str(message.get("message_id") or f"source-message:{index}")
                text = str(message.get("text") or message.get("content") or "")
                record = fabric.create_record(
                    kind="diagnostic_event",
                    provider=self.name,
                    source_ref=native_id,
                    assertion="observed",
                    summary=text[:500] or f"Source record {index}",
                    content=message,
                    anchors=[SourceAnchor(
                        source_id=native_id,
                        timestamp=str(message.get("create_time") or ""),
                    )],
                    metadata={
                        "request_context_kind": "source_message",
                        "source_only": True,
                        "thread_id": str(message.get("thread_id") or ""),
                        "chat_id": str(message.get("chat_id") or ""),
                    },
                )
                record_ids.append(record.evidence_id)
                message_ids.append(native_id)
                fabric.link("contains", root.evidence_id, record.evidence_id)
                for attachment_index, attachment in enumerate(
                    message.get("attachments") or [], start=1
                ):
                    if not isinstance(attachment, dict):
                        continue
                    attachment_ref = str(
                        attachment.get("file_key") or attachment.get("path")
                        or attachment.get("name")
                        or f"{native_id}:attachment:{attachment_index}"
                    )
                    media = fabric.create_record(
                        kind="media_asset",
                        provider=self.name,
                        source_ref=attachment_ref,
                        assertion="observed",
                        summary=str(attachment.get("name") or attachment_ref),
                        content=attachment,
                        anchors=[SourceAnchor(
                            path=str(attachment.get("path") or ""),
                            source_id=native_id,
                            artifact_id=attachment_ref,
                        )],
                        metadata={
                            "request_context_kind": "source_attachment",
                            "source_only": True,
                        },
                    )
                    record_ids.append(media.evidence_id)
                    attachment_ids.append(attachment_ref)
                    fabric.link("contains", record.evidence_id, media.evidence_id)

        return {
            "record_evidence_ids": record_ids,
            "source_message_ids": list(dict.fromkeys(message_ids)),
            "source_attachment_ids": list(dict.fromkeys(attachment_ids)),
            "source_only": isinstance(source_context, dict),
        }


class FrozenPipelineProvider:
    """Adapt the frozen official response into v3 evidence records."""

    name = "frozen_read_pipeline"

    def __init__(self, runner: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.runner = runner

    def collect(
        self,
        request: ReadRequest,
        task: ReadTask,
        fabric: EvidenceFabric,
    ) -> dict[str, Any]:
        response = dict(self.runner(request.to_baseline_payload()) or {})
        source_record_ids: list[str] = []
        for source in response.get("sources") or []:
            record = fabric.create_record(
                kind="source_artifact",
                provider=self.name,
                source_ref=str(source),
                assertion="source_asserted",
                summary=str(source),
                content={"source": source},
                anchors=[SourceAnchor(path=str(source))],
            )
            source_record_ids.append(record.evidence_id)
        kg_record_ids: list[str] = []
        for object_id in response.get("evidence_ids") or []:
            record = fabric.create_record(
                kind="kg_object",
                provider=self.name,
                source_ref=str(object_id),
                assertion="source_asserted",
                summary=f"Frozen runtime evidence object {object_id}",
                content={"object_id": object_id},
                anchors=[SourceAnchor(object_id=str(object_id))],
            )
            kg_record_ids.append(record.evidence_id)
        decision = fabric.create_record(
            kind="runtime_decision",
            provider=self.name,
            source_ref=str(response.get("session_id") or "baseline"),
            assertion="derived",
            summary=(
                f"status={response.get('status')}; family={response.get('family_id')}; "
                f"variant={response.get('variant_id')}"
            ),
            content={
                key: response.get(key)
                for key in (
                    "schema_version", "session_id", "status", "failure_type",
                    "confidence", "family_id", "variant_id", "plan_id",
                    "current_action_id", "required_data", "metadata",
                )
            },
        )
        answer = fabric.create_record(
            kind="answer_fragment",
            provider=self.name,
            source_ref=str(response.get("session_id") or "baseline"),
            assertion="derived",
            summary=str(response.get("answer") or "")[:500],
            content={
                "answer": str(response.get("answer") or ""),
                "answer_sections": response.get("answer_sections") or [],
            },
        )
        fabric.link("derived_from", answer.evidence_id, decision.evidence_id)
        for evidence_id in [*source_record_ids, *kg_record_ids]:
            fabric.link("derived_from", answer.evidence_id, evidence_id)
        return {
            "response": response,
            "decision_evidence_id": decision.evidence_id,
            "answer_evidence_id": answer.evidence_id,
            "source_evidence_ids": source_record_ids,
            "kg_evidence_ids": kg_record_ids,
        }


class KGSAGProvider:
    """Expose KG/SAG candidates and chunks without changing lock semantics."""

    name = "kg_v2_sag"

    def __init__(self, read_model: Any, *, kg_root: str | Path) -> None:
        self.read_model = read_model
        self.kg_root = Path(kg_root)

    def collect(
        self,
        request: ReadRequest,
        task: ReadTask,
        fabric: EvidenceFabric,
    ) -> dict[str, Any]:
        limit = min(20, max(1, int(task.budgets.get("kg_candidates", 10))))
        return self.search(request.query, fabric=fabric, limit=limit)

    def search(
        self,
        query: str,
        *,
        fabric: EvidenceFabric,
        limit: int = 10,
    ) -> dict[str, Any]:
        limit = min(20, max(1, int(limit)))
        candidates = self.read_model.search_variants(query, limit=limit)
        revision = kg_v2_graph_revision(self.kg_root)
        source_revision = kg_v2_source_revision(self.kg_root)
        candidate_ids: list[str] = []
        candidate_by_variant: dict[str, str] = {}
        for candidate in candidates:
            payload = to_jsonable(candidate)
            record = fabric.create_record(
                kind="kg_object",
                provider=self.name,
                source_ref=str(candidate.variant_id),
                assertion="source_asserted",
                summary=(
                    f"{candidate.family_label} / {candidate.variant_label}; "
                    f"score={candidate.score}"
                ),
                content=payload,
                anchors=[
                    SourceAnchor(object_id=str(candidate.family_id)),
                    SourceAnchor(object_id=str(candidate.variant_id)),
                ],
                source_revision=revision,
                metadata={"source_revision": source_revision, "route": candidate.route},
            )
            candidate_ids.append(record.evidence_id)
            candidate_by_variant[str(candidate.variant_id)] = record.evidence_id
        chunk_ids: list[str] = []
        for chunk in (self.read_model.last_retrieval or {}).get("chunks") or []:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("object_id") or "")
            record = fabric.create_record(
                kind="document_chunk",
                provider=self.name,
                source_ref=chunk_id,
                assertion="source_asserted",
                summary=str(
                    chunk.get("heading") or chunk.get("title")
                    or chunk.get("text") or chunk_id
                )[:500],
                content=chunk,
                anchors=[
                    SourceAnchor(
                        path=str(chunk.get("source_path") or ""),
                        object_id=str(chunk.get("object_id") or ""),
                        chunk_id=chunk_id,
                    )
                ],
                source_revision=revision,
            )
            chunk_ids.append(record.evidence_id)
            for variant_id in chunk.get("variant_ids") or []:
                candidate_id = candidate_by_variant.get(str(variant_id))
                if candidate_id:
                    fabric.link("supports", record.evidence_id, candidate_id, confidence=0.6)
        return {
            "query": query,
            "candidate_evidence_ids": candidate_ids,
            "chunk_evidence_ids": chunk_ids,
            "retrieval_trace": dict((self.read_model.last_retrieval or {}).get("trace") or {}),
            "graph_revision": revision,
            "source_revision": source_revision,
        }

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        value = self.read_model.get(str(object_id or ""))
        return dict(value) if isinstance(value, dict) else None


class IncidentProvider:
    """Adapt Incident Evidence Runtime output into the common fabric."""

    name = "incident_evidence_runtime"

    def __init__(self, runner: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.runner = runner

    def should_collect(self, request: ReadRequest, task: ReadTask) -> bool:
        return bool(
            request.evidence_resources
            or request.log_summary
            or task.complexity == "incident"
        )

    def collect(
        self,
        request: ReadRequest,
        task: ReadTask,
        fabric: EvidenceFabric,
    ) -> dict[str, Any]:
        if not self.should_collect(request, task):
            return {"skipped": True, "reason": "no_incident_scope"}
        result = dict(self.runner(request.to_baseline_payload()) or {})
        evidence_map: dict[str, str] = {}
        for link in result.get("evidence_links") or []:
            source_id = str(link.get("evidence_id") or "")
            record = fabric.create_record(
                kind="source_artifact",
                provider=self.name,
                source_ref=source_id or str(link.get("source_name") or ""),
                assertion="observed",
                summary=str(link.get("source_name") or source_id),
                content=link,
                anchors=[SourceAnchor(
                    path=str(link.get("source_name") or ""),
                    line_start=link.get("line_start"),
                    line_end=link.get("line_end"),
                    byte_start=link.get("byte_start"),
                    byte_end=link.get("byte_end"),
                    timestamp=str(link.get("timestamp") or ""),
                    artifact_id=str(link.get("artifact_id") or ""),
                )],
                parser_version=str(link.get("parser_version") or ""),
            )
            if source_id:
                evidence_map[source_id] = record.evidence_id
        event_ids: list[str] = []
        for event in result.get("events") or []:
            record = fabric.create_record(
                kind="diagnostic_event",
                provider=self.name,
                source_ref=str(event.get("event_id") or ""),
                assertion="observed",
                summary=str(event.get("message") or "")[:500],
                content=event,
                anchors=[SourceAnchor(
                    timestamp=str(event.get("timestamp_utc") or event.get("timestamp_raw") or ""),
                    artifact_id=str(event.get("artifact_id") or ""),
                )],
                metadata={"polarity": event.get("polarity"), "severity": event.get("severity")},
            )
            event_ids.append(record.evidence_id)
            for native_id in event.get("evidence_ids") or []:
                if native_id in evidence_map:
                    fabric.link("derived_from", record.evidence_id, evidence_map[native_id])
        stack_ids: list[str] = []
        for trace in result.get("stack_traces") or []:
            record = fabric.create_record(
                kind="stack_trace",
                provider=self.name,
                source_ref=str(trace.get("trace_id") or ""),
                assertion="observed",
                summary=f"stack trace {trace.get('trace_id')} ({len(trace.get('frames') or [])} frames)",
                content=trace,
                anchors=[SourceAnchor(artifact_id=str(trace.get("artifact_id") or ""))],
            )
            stack_ids.append(record.evidence_id)
        environment_ids: list[str] = []
        environment = result.get("environment") or {}
        for field, values in (environment.get("values") or {}).items():
            record = fabric.create_record(
                kind="environment_fact",
                provider=self.name,
                source_ref=str(field),
                assertion="observed",
                summary=f"{field}: {', '.join(str(item) for item in values)}",
                content={"field": field, "values": values},
            )
            environment_ids.append(record.evidence_id)
            for native_id in (environment.get("evidence_ids") or {}).get(field) or []:
                if native_id in evidence_map:
                    fabric.link("derived_from", record.evidence_id, evidence_map[native_id])
        hypotheses: list[HypothesisRecord] = []
        for value in result.get("hypotheses") or []:
            support = [
                evidence_map.get(item, item)
                for item in value.get("support_evidence_ids") or []
                if evidence_map.get(item, item) in {record.evidence_id for record in fabric.records()}
            ]
            contradict = [
                evidence_map.get(item, item)
                for item in value.get("contradict_evidence_ids") or []
                if evidence_map.get(item, item) in {record.evidence_id for record in fabric.records()}
            ]
            status = str(value.get("status") or "candidate")
            state = {
                "supported": "observed_support",
                "observed_support": "observed_support",
                "locked": "locked_root_cause",
                "ruled_out": "contradicted",
                "inconclusive": "needs_evidence",
            }.get(status, status if status in {"candidate", "needs_evidence", "observed_support"} else "candidate")
            hypotheses.append(HypothesisRecord(
                hypothesis_id=str(value.get("hypothesis_id") or ""),
                label=str(value.get("label") or ""),
                mechanism=str(value.get("failure_mechanism") or ""),
                suspected_component=str(value.get("suspected_component") or ""),
                state=state,  # type: ignore[arg-type]
                confidence=float(value.get("confidence") or 0.0),
                support_evidence_ids=support,
                contradict_evidence_ids=contradict,
                missing_evidence=[str(item) for item in value.get("missing_evidence") or []],
                family_id=str(value.get("family_id") or ""),
                variant_id=str(value.get("variant_id") or ""),
                source_provider=self.name,
            ))
        report = fabric.create_record(
            kind="answer_fragment",
            provider=self.name,
            source_ref=str((result.get("case") or {}).get("case_id") or "incident"),
            assertion="derived",
            summary=str(result.get("report") or "")[:500],
            content={"report": result.get("report"), "next_tests": result.get("next_tests") or []},
        )
        for evidence_id in [*event_ids, *stack_ids, *environment_ids]:
            fabric.link("derived_from", report.evidence_id, evidence_id)
        return {
            "result": result,
            "report_evidence_id": report.evidence_id,
            "event_evidence_ids": event_ids,
            "stack_evidence_ids": stack_ids,
            "environment_evidence_ids": environment_ids,
            "hypotheses": hypotheses,
        }


class RawCorpusProvider:
    """Generic bounded primitives over raw and KG_v2 source files.

    This adapter intentionally contains no ranker or query-specific routing.
    An agent/planner chooses the glob, search expression and exact range.
    """

    name = "raw_corpus"

    def __init__(self, *, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.tools = CorpusReadTools(self.workspace)

    def collect(
        self,
        request: ReadRequest,
        task: ReadTask,
        fabric: EvidenceFabric,
    ) -> dict[str, Any]:
        ids: list[str] = []
        for resource in request.evidence_resources:
            if not isinstance(resource, dict):
                continue
            source_ref = str(
                resource.get("path") or resource.get("url")
                or resource.get("name") or resource.get("resource_id") or ""
            )
            record = fabric.create_record(
                kind="source_artifact",
                provider=self.name,
                source_ref=source_ref,
                assertion="observed",
                summary=str(resource.get("name") or source_ref),
                content={
                    key: resource.get(key)
                    for key in ("resource_id", "kind", "name", "path", "url", "mime", "size", "sha256")
                },
                anchors=[SourceAnchor(path=str(resource.get("path") or ""))],
                metadata={"intake_only": True},
            )
            ids.append(record.evidence_id)
        return {"resource_evidence_ids": ids, "agent_tools_available": True}

    def list_files(self, *, glob: str, limit: int = 200) -> dict[str, Any]:
        return self.tools.execute("list_files", {"glob": glob, "limit": limit})[0]

    def search_text(
        self,
        *,
        query: str,
        path_glob: str,
        regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 100,
        context_lines: int = 1,
    ) -> dict[str, Any]:
        return self.tools.execute("search_text", {
            "query": query,
            "path_glob": path_glob,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "max_matches": max_matches,
            "context_lines": context_lines,
        })[0]

    def read_text(self, *, path: str, start_line: int, end_line: int) -> dict[str, Any]:
        return self.tools.execute("read_text", {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
        })[0]

    def read_document(self, *, path: str, start_block: int = 1, end_block: int = 200) -> dict[str, Any]:
        resolved = self._resolve_corpus_path(path)
        blocks = _read_blocks(resolved)
        start = max(1, int(start_block))
        end = min(len(blocks), max(start, int(end_block)))
        selected = blocks[start - 1:end]
        return {
            "path": path,
            "start_block": start,
            "end_block": start + len(selected) - 1,
            "total_blocks": len(blocks),
            "blocks": [
                {
                    "index": index,
                    "text": block.text,
                    "kind": block.kind,
                    "heading_level": block.heading_level,
                    "list_level": block.list_level,
                    "list_style": block.list_style,
                    "list_marker": block.list_marker,
                    "media_refs": list(block.media_refs),
                }
                for index, block in enumerate(selected, start=start)
            ],
        }

    @staticmethod
    def _resolve_corpus_path(logical: str) -> Path:
        value = Path(str(logical or ""))
        parts = value.parts
        if len(parts) >= 3 and parts[:2] == ("data", "raw"):
            root = CORPUS_ROOTS["raw"]
            path = root.joinpath(*parts[2:]).resolve()
        elif len(parts) >= 3 and parts[:2] == ("data", "kg_v2"):
            root = CORPUS_ROOTS["kg_v2"]
            path = root.joinpath(*parts[2:]).resolve()
        else:
            raise ValueError("path_outside_corpus")
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("file_not_found_or_outside_corpus")
        return path


class ReadToolRegistry:
    """Small typed registry that an agentic planner can call iteratively."""

    def __init__(
        self,
        *,
        kg: KGSAGProvider | None = None,
        raw: RawCorpusProvider | None = None,
        incident: IncidentProvider | None = None,
        fabric: EvidenceFabric | None = None,
    ) -> None:
        self.kg = kg
        self.raw = raw
        self.incident = incident
        self.fabric = fabric

    def schemas(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if self.raw:
            tools.extend([
                {
                    **schema,
                    "name": "raw_" + str(schema["name"]),
                }
                for schema in CorpusReadTools.schemas()
            ])
            tools.append({
                "type": "function",
                "name": "raw_read_document",
                "description": "Read an exact semantic block range from a corpus DOCX/XLSX/PPTX/Markdown document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_block": {"type": "integer", "minimum": 1},
                        "end_block": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_block", "end_block"],
                    "additionalProperties": False,
                },
                "strict": True,
            })
        if self.kg:
            tools.extend([{
                "type": "function",
                "name": "kg_search_candidates",
                "description": "Search KG_v2/SAG using a planner-selected query and add candidates/chunks to the evidence fabric.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                    },
                    "required": ["query", "limit"],
                    "additionalProperties": False,
                },
                "strict": True,
            }, {
                "type": "function",
                "name": "kg_get_object",
                "description": "Read one exact canonical KG_v2 object by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"object_id": {"type": "string"}},
                    "required": ["object_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            }])
        if self.fabric:
            tools.extend([{
                "type": "function",
                "name": "evidence_get_snapshot",
                "description": "Return the current immutable evidence graph snapshot.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            }, {
                "type": "function",
                "name": "evidence_query",
                "description": "Read a bounded set of evidence records by provider and/or kind.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "kind": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200}
                    },
                    "required": ["provider", "kind", "limit"],
                    "additionalProperties": False,
                },
                "strict": True,
            }])
        return tools

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        evidence_ids: list[str] = []
        if self.raw and name == "raw_list_files":
            payload = self.raw.list_files(**arguments)
        elif self.raw and name == "raw_search_text":
            payload = self.raw.search_text(**arguments)
            evidence_ids = self._ingest_raw_matches(payload)
        elif self.raw and name == "raw_read_text":
            payload = self.raw.read_text(**arguments)
            evidence_ids = self._ingest_raw_text(payload)
        elif self.raw and name == "raw_read_document":
            payload = self.raw.read_document(**arguments)
            evidence_ids = self._ingest_raw_document(payload)
        elif self.kg and name == "kg_search_candidates":
            if self.fabric is None:
                raise ValueError("evidence_fabric_unavailable")
            payload = self.kg.search(
                str(arguments.get("query") or ""),
                fabric=self.fabric,
                limit=int(arguments.get("limit") or 10),
            )
            evidence_ids = [
                *payload.get("candidate_evidence_ids", []),
                *payload.get("chunk_evidence_ids", []),
            ]
        elif self.kg and name == "kg_get_object":
            object_id = str(arguments.get("object_id") or "")
            value = self.kg.get_object(object_id)
            payload = {"object": value}
            if value is not None and self.fabric is not None:
                record = self.fabric.create_record(
                    kind="kg_object",
                    provider="kg_v2_sag",
                    source_ref=object_id,
                    assertion="source_asserted",
                    summary=str(value.get("label") or value.get("title") or object_id),
                    content=value,
                    anchors=[SourceAnchor(object_id=object_id)],
                )
                evidence_ids = [record.evidence_id]
        elif self.fabric and name == "evidence_get_snapshot":
            payload = self.fabric.snapshot()
        elif self.fabric and name == "evidence_query":
            limit = min(200, max(1, int(arguments.get("limit") or 50)))
            records = self.fabric.records(
                provider=str(arguments.get("provider") or ""),
                kind=str(arguments.get("kind") or ""),
            )[:limit]
            payload = {
                "records": [to_jsonable(record) for record in records],
                "returned": len(records),
                "truncated": len(self.fabric.records(
                    provider=str(arguments.get("provider") or ""),
                    kind=str(arguments.get("kind") or ""),
                )) > limit,
            }
            evidence_ids = [record.evidence_id for record in records]
        else:
            raise ValueError(f"unknown_read_runtime_v3_tool:{name}")
        return {
            "schema_version": "debug_agent_system.read_tool_result.v3",
            "tool": name,
            "status": "ok",
            "payload": payload,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "warnings": [],
            "exclusions": [],
            "truncated": bool(payload.get("truncated", False)) if isinstance(payload, dict) else False,
            "capability": {"read_only": True, "side_effect": False, "approval_required": False},
            "observability": {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "result_sha256": content_hash(payload),
            },
        }

    def _ingest_raw_matches(self, payload: dict[str, Any]) -> list[str]:
        if self.fabric is None:
            return []
        result: list[str] = []
        for match in payload.get("matches") or []:
            record = self.fabric.create_record(
                kind="document_chunk",
                provider="raw_corpus",
                source_ref=f"{match.get('path')}:{match.get('line')}",
                assertion="observed",
                summary=str(match.get("excerpt") or "")[:500],
                content=match,
                anchors=[SourceAnchor(
                    path=str(match.get("path") or ""),
                    line_start=int(match.get("line") or 0) or None,
                )],
                metadata={"retrieval_operation": "search_text"},
            )
            result.append(record.evidence_id)
        return result

    def _ingest_raw_text(self, payload: dict[str, Any]) -> list[str]:
        if self.fabric is None or payload.get("error"):
            return []
        record = self.fabric.create_record(
            kind="document_chunk",
            provider="raw_corpus",
            source_ref=(
                f"{payload.get('path')}:{payload.get('start_line')}-{payload.get('end_line')}"
            ),
            assertion="observed",
            summary=str(payload.get("text") or "")[:500],
            content=payload,
            anchors=[SourceAnchor(
                path=str(payload.get("path") or ""),
                line_start=int(payload.get("start_line") or 0) or None,
                line_end=int(payload.get("end_line") or 0) or None,
            )],
            metadata={"retrieval_operation": "read_text"},
        )
        return [record.evidence_id]

    def _ingest_raw_document(self, payload: dict[str, Any]) -> list[str]:
        if self.fabric is None or payload.get("error"):
            return []
        record = self.fabric.create_record(
            kind="document_chunk",
            provider="raw_corpus",
            source_ref=(
                f"{payload.get('path')}#blocks={payload.get('start_block')}-{payload.get('end_block')}"
            ),
            assertion="observed",
            summary="\n".join(
                str(item.get("text") or "") for item in payload.get("blocks") or []
            )[:500],
            content=payload,
            anchors=[SourceAnchor(path=str(payload.get("path") or ""))],
            metadata={"retrieval_operation": "read_document"},
        )
        return [record.evidence_id]
