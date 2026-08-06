"""KG_v2 + raw answers produced by a model-directed Codex investigation.

The pipeline intentionally has no pre-ranked retrieval result and no
domain-specific query rules.  It exposes bounded, read-only filesystem
primitives over the evidence corpus and lets Codex choose search terms, files,
iterations and answer organization through the Responses API.  Local code
retains the corpus boundary, tool execution, structured-output contract,
evidence verification, media materialization and audit artifact.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterator, Protocol

from debug_agent_system.adapters.codex_read.client import (
    CodexReadClientError,
    CodexResponsesClient,
    _read_local_env,
)
from debug_agent_system.kg_raw_codex.coverage import (
    AnswerScope,
    ProcedureVariantRequirement,
    RequiredFacet,
    build_answer_scope,
    build_required_facets,
    verify_answer_draft,
)
from debug_agent_system.kg_raw_codex.prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    SYSTEM_PROMPT_VERSION,
)
from debug_agent_system.kg_raw_codex.terminology_contract import (
    audit_terminology_search_contract,
    build_resolver_context,
    build_terminology_search_contract,
    load_terminology_manifest,
    terminology_governance_authority_errors,
    terminology_search_errors,
)
from debug_agent_system.kg_raw_codex.terminology_harness import (
    execute_terminology_search_contract,
)
from debug_agent_system.knowledge_v2.document_links import (
    extract_docx_hyperlinks,
)
from debug_agent_system.knowledge_v2.source_chunk_builder import (
    _read_docx_blocks,
)
from debug_agent_system.knowledge_v2.terminology import TerminologyResolver
from debug_agent_system.kg_raw_codex.deepseek_runner import (
    DeepSeekChatAgentRunner,
    DeepSeekChatCompletionsClient,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOTS = {
    "raw": (REPO_ROOT / "data/raw").resolve(),
    "kg_v2": (REPO_ROOT / "data/kg_v2").resolve(),
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/results/read_side_codex_comparison_20260730"
    / "kg_v2_raw_codex_first_query.json"
)
_DRAFT_SCHEMA_VERSION = "debug_agent_system.kg_raw_codex_draft.v5"
_ANSWER_SCHEMA_VERSION = "debug_agent_system.kg_raw_codex_answer.v5"
_SOURCE_PATH = re.compile(r"^data/(?:raw|kg_v2)/")
_PROCEDURE_VARIANT_LABEL = re.compile(
    r"^\s*(?:\[[^\]\n]+\]\s*)?(?P<label>"
    r"(?:方案|方法)\s*(?:[0-9]+|[零一二两三四五六七八九十]+)"
    r"|第\s*(?:[0-9]+|[零一二两三四五六七八九十]+)\s*种"
    r"(?:操作)?方法)\s*(?:[：:、.．]|$)",
    flags=re.M,
)
_EXTERNAL_ARTIFACT_WORDS = re.compile(
    r"(?:脚本|批处理|可执行文件|安装包|下载后|以管理员(?:权限|身份)运行)",
    flags=re.I,
)
_INTEGRITY_MARKERS = re.compile(
    r"(?:sha(?:1|256|512)?|md5|checksum|哈希|校验值)",
    flags=re.I,
)
_URL = re.compile(r"https?://[^\s<>()\]】]+", flags=re.I)


class CodexResponsesAgentError(RuntimeError):
    """A sanitized failure from the Responses API agent loop."""


class CodexCliAgentError(RuntimeError):
    """A sanitized failure from a non-interactive local Codex CLI run."""


class AgentRunner(Protocol):
    model: str
    runtime_metadata: dict[str, Any]

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        output_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the structured final response and runtime audit."""


@dataclass(slots=True)
class CodexResponsesAgentRunner:
    """Let Codex investigate through generic read-only corpus functions."""

    client: CodexResponsesClient
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    max_tool_rounds: int = 24
    max_tool_calls: int = 80
    runtime_metadata: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.max_tool_rounds = max(1, int(self.max_tool_rounds))
        self.max_tool_calls = max(1, int(self.max_tool_calls))
        self.runtime_metadata = {
            "engine": "responses_api",
            "transport": "non_streaming",
            "agent_loop": "model_directed_function_calls",
            "sandbox": "corpus_read_only",
            "store": False,
            "credential_source": ".env.local",
        }

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        output_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.client.timeout_seconds = max(30, int(timeout_seconds))
        tools = CorpusReadTools(workspace)
        input_items: list[dict[str, Any]] = [{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }]
        trace: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        request_ids: list[str] = []
        call_count = 0

        for round_index in range(1, self.max_tool_rounds + 2):
            body = {
                "model": self.model,
                "instructions": (
                    "Follow the supplied AOI evidence investigation contract. "
                    "Use the corpus tools iteratively before answering."
                ),
                "input": input_items,
                "tools": tools.schemas(),
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "reasoning": {"effort": self.reasoning_effort},
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "kg_raw_codex_answer",
                        "strict": True,
                        "schema": output_schema,
                    }
                },
                # Preserve reasoning locally between calls without retaining
                # the response server-side.
                "store": False,
                "include": ["reasoning.encrypted_content"],
            }
            try:
                response = self.client.create(body)
            except CodexReadClientError as exc:
                raise CodexResponsesAgentError(str(exc)) from exc
            _add_usage(usage, self.client.last_usage)
            if self.client.last_request_id:
                request_ids.append(self.client.last_request_id)
            output = response.get("output") or []
            calls = [
                item for item in output
                if isinstance(item, dict)
                and item.get("type") == "function_call"
            ]
            if not calls:
                draft = _structured_response_payload(output)
                return draft, {
                    "thread_id": str(response.get("id") or ""),
                    "request_ids": request_ids,
                    "usage": usage,
                    "tool_trace": trace,
                    "files_read": sorted(tools.files_read),
                    "tool_rounds": round_index - 1,
                    "tool_calls": call_count,
                    "stderr_warnings": [],
                }
            if round_index > self.max_tool_rounds:
                raise CodexResponsesAgentError("codex_tool_round_limit")

            # The Responses API requires every output item, including
            # encrypted reasoning, to be carried into a store:false turn.
            input_items.extend(output)
            for call in calls:
                call_count += 1
                if call_count > self.max_tool_calls:
                    raise CodexResponsesAgentError("codex_tool_call_limit")
                result, audit = tools.execute(
                    str(call.get("name") or ""),
                    call.get("arguments"),
                )
                audit.update({
                    "round": round_index,
                    "call_id": str(call.get("call_id") or ""),
                })
                trace.append(audit)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": str(call.get("call_id") or ""),
                    "output": json.dumps(result, ensure_ascii=False),
                })
        raise CodexResponsesAgentError("codex_agent_no_final_answer")


@dataclass(slots=True)
class CodexCliAgentRunner:
    """Let the locally authenticated Codex CLI plan, search and compose."""

    model: str
    reasoning_effort: str = "medium"
    codex_binary: str = "codex"
    runtime_metadata: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        binary = shutil.which(self.codex_binary)
        if binary is None:
            raise CodexCliAgentError("codex_cli_not_found")
        self.codex_binary = binary
        version = subprocess.run(
            [binary, "--version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        self.runtime_metadata = {
            "engine": "codex_cli",
            "transport": "non_interactive_exec",
            "agent_loop": "codex_native_planning_and_shell",
            "sandbox": "read_only",
            "approval_policy": "never",
            "ephemeral": True,
            "credential_source": "local_login",
            "codex_version": version,
            "reasoning_effort": self.reasoning_effort,
        }

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        output_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="kg-raw-codex-cli-") as raw:
            temp = Path(raw)
            schema_path = temp / "output.schema.json"
            answer_path = temp / "answer.json"
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            command = [
                self.codex_binary,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--cd",
                str(workspace),
                "--config",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(answer_path),
                "--json",
                "-",
            ]
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=max(30, int(timeout_seconds)),
                    env=os.environ.copy(),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexCliAgentError("codex_cli_timeout") from exc

            events = _parse_codex_cli_events(completed.stdout)
            trace = _codex_cli_tool_trace(events)
            if completed.returncode != 0 or not answer_path.is_file():
                raise CodexCliAgentError(
                    "codex_cli_failed:" + _codex_cli_failure(events)
                )
            try:
                draft = json.loads(answer_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CodexCliAgentError(
                    "codex_cli_invalid_structured_output"
                ) from exc
            if not isinstance(draft, dict):
                raise CodexCliAgentError("codex_cli_output_not_object")
            draft = _normalize_cli_draft_paths(draft, workspace=workspace)
            audited_files = _codex_cli_audited_files(
                draft,
                events=events,
                workspace=workspace,
            )
            usage = _codex_cli_usage(events)
            warnings = [
                line.strip()
                for line in completed.stderr.splitlines()
                if line.strip()
            ]
            return draft, {
                "thread_id": _codex_cli_thread_id(events),
                "usage": usage,
                "tool_trace": trace,
                "files_read": audited_files,
                "tool_rounds": 1,
                "tool_calls": len(trace),
                "stderr_warnings": warnings[-20:],
                "process_returncode": completed.returncode,
            }


class CorpusReadTools:
    """Generic file access over the two evidence roots and DOCX views.

    There is deliberately no relevance scoring, query rewriting or ranking
    here.  Results are deterministic path/match listings; Codex decides what
    to search and what to read next.
    """

    _TEXT_SUFFIXES = {
        ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv",
        ".xml", ".html", ".htm", ".log", ".ini", ".cfg", ".toml",
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.files_read: set[str] = set()

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "list_files",
                "description": (
                    "List corpus files by relative glob. Returns paths only; "
                    "it does not rank relevance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "glob": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                    },
                    "required": ["glob", "limit"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "search_text",
                "description": (
                    "Search text files for a literal or regular expression. "
                    "Choose the query and path glob yourself; results are in "
                    "path/line order with no relevance score."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path_glob": {"type": "string"},
                        "regex": {"type": "boolean"},
                        "case_sensitive": {"type": "boolean"},
                        "max_matches": {
                            "type": "integer", "minimum": 1, "maximum": 500
                        },
                        "context_lines": {
                            "type": "integer", "minimum": 0, "maximum": 5
                        },
                    },
                    "required": [
                        "query", "path_glob", "regex", "case_sensitive",
                        "max_matches", "context_lines",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "read_text",
                "description": (
                    "Read an exact inclusive line range from one corpus text "
                    "file. Use repeated calls for long documents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    def execute(
        self,
        name: str,
        raw_arguments: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )
            if not isinstance(arguments, dict):
                raise ValueError("arguments_not_object")
            if name == "list_files":
                result = self._list_files(arguments)
            elif name == "search_text":
                result = self._search_text(arguments)
            elif name == "read_text":
                result = self._read_text(arguments)
            else:
                raise ValueError("unknown_tool")
            status = "ok"
        except (ValueError, OSError, re.error, UnicodeError) as exc:
            result = {"error": str(exc)[:200]}
            status = "error"
        encoded = json.dumps(result, ensure_ascii=False)
        audit = {
            "type": "function_call",
            "name": name,
            "status": status,
            "arguments": _safe_tool_arguments(arguments if 'arguments' in locals() else {}),
            "output_chars": len(encoded),
            "output_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }
        return result, audit

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = _safe_glob(arguments.get("glob"))
        limit = min(500, max(1, int(arguments.get("limit") or 200)))
        paths = [
            path for path in self._logical_files()
            if fnmatch.fnmatchcase(path, pattern)
        ]
        return {
            "paths": paths[:limit],
            "returned": min(len(paths), limit),
            "truncated": len(paths) > limit,
        }

    def _search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        if not query or len(query) > 1000:
            raise ValueError("invalid_query")
        pattern = _safe_glob(arguments.get("path_glob"))
        use_regex = bool(arguments.get("regex"))
        case_sensitive = bool(arguments.get("case_sensitive"))
        max_matches = min(500, max(1, int(arguments.get("max_matches") or 100)))
        context = min(5, max(0, int(arguments.get("context_lines") or 0)))
        flags = 0 if case_sensitive else re.IGNORECASE
        expression = re.compile(query if use_regex else re.escape(query), flags)
        matches: list[dict[str, Any]] = []
        for logical in self._logical_files():
            if not fnmatch.fnmatchcase(logical, pattern):
                continue
            path = self._resolve(logical)
            if path.suffix.lower() not in self._TEXT_SUFFIXES:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            file_used = False
            for index, line in enumerate(lines):
                if not expression.search(line):
                    continue
                start = max(0, index - context)
                end = min(len(lines), index + context + 1)
                matches.append({
                    "path": logical,
                    "line": index + 1,
                    "excerpt": "\n".join(
                        f"{line_no + 1}:{lines[line_no]}"
                        for line_no in range(start, end)
                    ),
                })
                file_used = True
                if len(matches) >= max_matches:
                    break
            if file_used:
                self.files_read.add(self._canonical_source(logical, lines))
            if len(matches) >= max_matches:
                break
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": len(matches) >= max_matches,
        }

    def _read_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        logical = _safe_logical_path(arguments.get("path"))
        path = self._resolve(logical)
        if path.suffix.lower() not in self._TEXT_SUFFIXES:
            raise ValueError("not_a_supported_text_file")
        start = max(1, int(arguments.get("start_line") or 1))
        end = max(start, int(arguments.get("end_line") or start))
        if end - start + 1 > 500:
            raise ValueError("line_range_exceeds_500")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        self.files_read.add(self._canonical_source(logical, lines))
        selected = lines[start - 1:min(end, len(lines))]
        return {
            "path": logical,
            "start_line": start,
            "end_line": start + len(selected) - 1,
            "total_lines": len(lines),
            "text": "\n".join(
                f"{number}:{line}"
                for number, line in enumerate(selected, start=start)
            ),
        }

    def _logical_files(self) -> list[str]:
        paths: list[str] = []
        for scope, root in CORPUS_ROOTS.items():
            for path in root.rglob("*"):
                if path.is_file():
                    paths.append(
                        (Path("data") / scope / path.relative_to(root)).as_posix()
                    )
        extracted = self.workspace / "data/extracted_docx"
        if extracted.is_dir():
            for path in extracted.rglob("*"):
                if path.is_file():
                    paths.append(path.relative_to(self.workspace).as_posix())
        return sorted(dict.fromkeys(paths))

    def _resolve(self, logical: str) -> Path:
        logical = _safe_logical_path(logical)
        parts = Path(logical).parts
        if len(parts) >= 3 and parts[:2] == ("data", "raw"):
            path = CORPUS_ROOTS["raw"].joinpath(*parts[2:]).resolve()
            root = CORPUS_ROOTS["raw"]
        elif len(parts) >= 3 and parts[:2] == ("data", "kg_v2"):
            path = CORPUS_ROOTS["kg_v2"].joinpath(*parts[2:]).resolve()
            root = CORPUS_ROOTS["kg_v2"]
        elif len(parts) >= 3 and parts[:2] == ("data", "extracted_docx"):
            path = self.workspace.joinpath(*parts).resolve()
            root = (self.workspace / "data/extracted_docx").resolve()
        else:
            raise ValueError("path_outside_corpus")
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("file_not_found_or_outside_corpus")
        return path

    @staticmethod
    def _canonical_source(logical: str, lines: list[str]) -> str:
        if logical.startswith("data/extracted_docx/") and lines:
            marker = "SOURCE_PATH:"
            if lines[0].startswith(marker):
                value = lines[0][len(marker):].strip()
                if _SOURCE_PATH.match(value):
                    return value
        return logical


@dataclass(slots=True)
class CorpusWorkspace:
    root: Path
    media_inventory: list[dict[str, Any]]
    docx_count: int


@contextmanager
def prepared_corpus_workspace(*, asset_root: Path) -> Iterator[CorpusWorkspace]:
    """Expose the corpus and deterministic DOCX text, without ranking it."""

    asset_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kg-raw-codex-corpus-") as raw:
        root = Path(raw)
        data_dir = root / "data"
        data_dir.mkdir()
        for scope, source_root in CORPUS_ROOTS.items():
            (data_dir / scope).symlink_to(source_root, target_is_directory=True)

        extracted_root = data_dir / "extracted_docx"
        extracted_root.mkdir()
        media_inventory: list[dict[str, Any]] = []
        docx_count = 0
        for scope, source_root in CORPUS_ROOTS.items():
            for source in sorted(source_root.rglob("*.docx")):
                if not source.is_file():
                    continue
                docx_count += 1
                portable = _portable_path(source)
                destination = (
                    extracted_root
                    / scope
                    / source.relative_to(source_root)
                ).with_suffix(source.suffix + ".md")
                destination.parent.mkdir(parents=True, exist_ok=True)
                text, media = _extract_docx_for_agent(
                    source,
                    portable=portable,
                    asset_root=asset_root,
                )
                destination.write_text(text, encoding="utf-8")
                media_inventory.extend(media)

        (root / "README.md").write_text(
            "\n".join([
                "# KG_v2 + raw corpus",
                "",
                "- `data/raw` and `data/kg_v2` are the only evidence roots.",
                "- `data/extracted_docx` contains deterministic Markdown "
                "views of every DOCX.",
                "- Each extracted view starts with its canonical "
                "`SOURCE_PATH`; cite that path, not the temporary view path.",
                "- Codex can use `list_files`, `search_text` and `read_text` "
                "iteratively. There is no pre-ranked result set.",
                "",
            ]),
            encoding="utf-8",
        )
        yield CorpusWorkspace(
            root=root,
            media_inventory=_deduplicate_media(media_inventory),
            docx_count=docx_count,
        )


class KGRawCodexPipeline:
    """Codex-directed investigation followed by deterministic release checks."""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        verification_attempts: int = 2,
        timeout_seconds: int = 600,
        terminology_enabled: bool = True,
    ) -> None:
        self.runner = runner
        self.verification_attempts = max(1, int(verification_attempts))
        self.timeout_seconds = max(30, int(timeout_seconds))
        self.terminology_enabled = bool(terminology_enabled)

    def run(self, query: str, output: Path) -> dict[str, Any]:
        required_facets = build_required_facets(query)
        answer_scope = build_answer_scope(query)
        facet_payload = [facet.to_dict() for facet in required_facets]
        terminology_context = (
            build_resolver_context(answer_scope)
            if self.terminology_enabled
            else {}
        )
        if self.terminology_enabled:
            terminology = TerminologyResolver.from_root(
                CORPUS_ROOTS["kg_v2"]
            )
            terminology_resolution = terminology.resolve(
                query,
                limit=30,
                context=terminology_context,
            )
        else:
            terminology_resolution = {
                "schema_version": (
                    "debug_agent_system.terminology_resolution.disabled.v1"
                ),
                "query": query,
                "status": "disabled_for_ab_control",
                "resolved_mentions": [],
                "ambiguous_mentions": [],
                "ambiguous_supporting_mentions": [],
                "supporting_concepts": [],
                "safe_expansions": [],
                "retrieval_expansions": [],
                "entity_relations": [],
            }
        terminology_search_contract = build_terminology_search_contract(
            query,
            terminology_resolution,
        )
        terminology_manifest = load_terminology_manifest(
            CORPUS_ROOTS["kg_v2"]
        )
        terminology_manifest = {
            **terminology_manifest,
            "enabled": self.terminology_enabled,
        }
        attempts: list[dict[str, Any]] = []
        aggregate_trace: list[dict[str, Any]] = []
        total_usage: dict[str, int] = {}
        accepted: dict[str, Any] | None = None
        accepted_terminology_audit: dict[str, Any] | None = None
        accepted_procedure_variants: list[
            ProcedureVariantRequirement
        ] = []
        terminology_search_execution: dict[str, Any] = {
            "schema_version": (
                "debug_agent_system.terminology_search_execution.v1"
            ),
            "task_count": 0,
            "tasks": [],
            "results": [],
            "tool_trace": [],
        }
        deterministic_terminology_trace: list[dict[str, Any]] = []
        last_errors: list[str] = []

        with prepared_corpus_workspace(
            asset_root=output.parent / "kg_v2_raw_assets"
        ) as workspace:
            # Execute approved source/canonical obligations before Codex
            # starts.  This is a deterministic discovery/audit step; its
            # excerpts are not counted as evidence or files_read.
            terminology_tools = CorpusReadTools(workspace.root)
            terminology_search_execution = execute_terminology_search_contract(
                terminology_search_contract,
                terminology_tools,
            )
            deterministic_terminology_trace = list(
                terminology_search_execution.get("tool_trace") or []
            )
            aggregate_trace.extend(deterministic_terminology_trace)
            for attempt_index in range(self.verification_attempts):
                prompt = _agent_prompt(
                    query=query,
                    facets=facet_payload,
                    answer_scope=answer_scope.to_dict(),
                    terminology_resolution=terminology_resolution,
                    terminology_search_contract=terminology_search_contract,
                    terminology_search_execution=terminology_search_execution,
                    previous_errors=last_errors,
                )
                draft, audit = self.runner.run(
                    prompt=prompt,
                    workspace=workspace.root,
                    output_schema=_draft_output_schema(),
                    timeout_seconds=self.timeout_seconds,
                )
                draft = _finalize_draft_contract(
                    draft,
                    actual_files_read=audit.get("files_read") or [],
                )
                model_trace = list(audit.get("tool_trace") or [])
                aggregate_trace.extend(model_trace)
                combined_trace = [
                    *deterministic_terminology_trace,
                    *model_trace,
                ]
                _add_usage(total_usage, audit.get("usage") or {})
                required_procedure_variants = (
                    _discover_required_procedure_variants(
                        query,
                        draft,
                        required_facets=required_facets,
                        answer_scope=answer_scope,
                        workspace=workspace.root,
                    )
                )
                errors = _verify_native_draft(
                    draft,
                    required_facets=required_facets,
                    answer_scope=answer_scope,
                    media_inventory=workspace.media_inventory,
                    actual_files_read=audit.get("files_read") or [],
                    tool_trace=combined_trace,
                    terminology_search_contract=(
                        terminology_search_contract
                    ),
                    required_procedure_variants=(
                        required_procedure_variants
                    ),
                )
                attempts.append({
                    "attempt": attempt_index + 1,
                    "accepted": not errors,
                    "errors": errors,
                    "thread_id": str(audit.get("thread_id") or ""),
                    "usage": dict(audit.get("usage") or {}),
                    "stderr_warnings": list(
                        audit.get("stderr_warnings") or []
                    ),
                })
                if not errors:
                    accepted = draft
                    accepted_terminology_audit = (
                        audit_terminology_search_contract(
                            terminology_search_contract,
                            combined_trace,
                        )
                    )
                    accepted_procedure_variants = (
                        required_procedure_variants
                    )
                    break
                last_errors = errors

        if accepted is None:
            raise RuntimeError(
                "kg_raw_codex_verification_failed:"
                + ",".join(last_errors or ["codex_returned_no_final_answer"])
            )

        files_read = list(dict.fromkeys(
            str(path) for path in accepted.get("files_read") or []
            if str(path).strip()
        ))
        answer = str(accepted["answer_markdown"]).strip()
        payload = {
            "schema_version": _ANSWER_SCHEMA_VERSION,
            "query": query,
            "answer": answer,
            "answer_draft": accepted,
            "required_facets": facet_payload,
            "required_procedure_variants": [
                requirement.to_dict()
                for requirement in accepted_procedure_variants
            ],
            "answer_scope": answer_scope.to_dict(),
            "terminology_enabled": self.terminology_enabled,
            "terminology_context": terminology_context,
            "terminology_resolution": terminology_resolution,
            "terminology_search_contract": terminology_search_contract,
            "terminology_search_execution": terminology_search_execution,
            "terminology_search_audit": accepted_terminology_audit or {},
            "terminology_manifest": terminology_manifest,
            "model": self.runner.model,
            "prompt": {
                "system_version": SYSTEM_PROMPT_VERSION,
                "system_sha256": SYSTEM_PROMPT_SHA256,
            },
            "runtime": dict(self.runner.runtime_metadata),
            "usage": total_usage,
            "allowed_roots": [
                _portable_path(root) for root in CORPUS_ROOTS.values()
            ],
            "files_read": files_read,
            "media_exposed": _media_used_by_answer(
                answer,
                media_inventory=workspace.media_inventory,
            ),
            "tool_trace": aggregate_trace,
            "verification": {
                "passed": True,
                "attempt_count": len(attempts),
                "attempts": attempts,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload


def run(
    query: str,
    output: Path,
    *,
    max_rounds: int | None = None,
    runtime: str | None = None,
    model: str | None = None,
    terminology_enabled: bool = True,
) -> dict[str, Any]:
    """Run with either the Responses harness or local Codex CLI."""

    config = _load_pipeline_config()
    selected_runtime = str(
        runtime or config.get("runtime") or "responses_api"
    ).strip()
    if selected_runtime == "deepseek":
        default_model = (
            os.environ.get("DEEPSEEK_W2_TOOL_MODEL", "").strip()
            or os.environ.get("DEEPSEEK_W2_MODEL", "").strip()
            or str(config.get("deepseek_model") or "deepseek-v4-flash")
        )
    else:
        default_model = (
            config.get("cli_model")
            if selected_runtime == "codex_cli"
            else config.get("model")
        ) or (
            "gpt-5.3-codex"
            if selected_runtime == "codex_cli"
            else "gpt-5.4"
        )
    selected_model = str(model or default_model).strip()
    if selected_runtime == "codex_cli":
        runner: AgentRunner = CodexCliAgentRunner(
            model=selected_model,
            reasoning_effort=str(
                config.get("cli_reasoning_effort")
                or config.get("reasoning_effort")
                or "medium"
            ),
            codex_binary=str(config.get("codex_binary") or "codex"),
        )
    elif selected_runtime == "responses_api":
        local_env = _read_local_env(REPO_ROOT / ".env.local")
        client = CodexResponsesClient(
            api_key=local_env.get("OPENAI_API_KEY", ""),
            base_url=local_env.get("OPENAI_BASE_URL", ""),
            timeout_seconds=int(config.get("timeout_seconds") or 600),
        )
        runner = CodexResponsesAgentRunner(
            client=client,
            model=selected_model,
            reasoning_effort=str(config.get("reasoning_effort") or "medium"),
            max_tool_rounds=int(config.get("max_tool_rounds") or 24),
            max_tool_calls=int(config.get("max_tool_calls") or 80),
        )
    elif selected_runtime == "deepseek":
        local_env = _read_local_env(REPO_ROOT / ".env.local")
        client = DeepSeekChatCompletionsClient(
            api_key=local_env.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", ""),
            timeout_seconds=int(config.get("timeout_seconds") or 600),
        )
        runner = DeepSeekChatAgentRunner(
            client=client,
            model=selected_model,
            max_tool_rounds=int(config.get("max_tool_rounds") or 24),
            max_tool_calls=int(config.get("max_tool_calls") or 80),
        )
    else:
        raise ValueError(f"unsupported_kg_raw_codex_runtime:{selected_runtime}")
    pipeline = KGRawCodexPipeline(
        runner=runner,
        verification_attempts=int(
            max_rounds
            or config.get("verification_attempts")
            or config.get("max_rounds")
            or 2
        ),
        timeout_seconds=int(config.get("timeout_seconds") or 600),
        terminology_enabled=terminology_enabled,
    )
    return pipeline.run(query, output)


def _extract_docx_for_agent(
    path: Path,
    *,
    portable: str,
    asset_root: Path,
) -> tuple[str, list[dict[str, Any]]]:
    lines = [
        f"SOURCE_PATH: {portable}",
        f"DOCUMENT_TITLE: {path.stem}",
        "",
    ]
    media_inventory: list[dict[str, Any]] = []
    for block in _read_docx_blocks(path, asset_root=asset_root):
        text = str(block.text or "").strip()
        if text:
            lines.append(f"[{block.kind}] {text}")
        for raw_media in block.media_refs:
            asset_path = str(raw_media.get("asset_path") or "").strip()
            if not asset_path:
                continue
            media = {
                "media_id": str(raw_media.get("media_id") or ""),
                "media_kind": str(raw_media.get("media_kind") or ""),
                "context_label": str(
                    raw_media.get("context_label")
                    or raw_media.get("caption")
                    or raw_media.get("label")
                    or "源文档图片"
                ),
                "asset_path": asset_path,
                "mime_type": str(raw_media.get("mime_type") or ""),
                "relationship_id": str(
                    raw_media.get("relationship_id") or ""
                ),
                "source_path": portable,
            }
            media_inventory.append(media)
            lines.append(
                "[source_media] "
                f"kind={media['media_kind']}; "
                f"context_label={media['context_label']}; "
                f"asset_path={media['asset_path']}; "
                f"relationship_id={media['relationship_id']}"
            )
    for link in extract_docx_hyperlinks(path):
        lines.append(
            "[navigation_link] "
            f"relationship_id={link.get('relationship_id')}; "
            f"text={link.get('link_text')}; "
            f"target_url={link.get('target_url')}; "
            f"wiki_token={link.get('wiki_token')}"
        )
    lines.append("")
    return "\n".join(lines), media_inventory


def _agent_prompt(
    *,
    query: str,
    facets: list[dict[str, Any]],
    answer_scope: dict[str, Any],
    terminology_resolution: dict[str, Any],
    terminology_search_contract: dict[str, Any],
    terminology_search_execution: dict[str, Any],
    previous_errors: list[str],
) -> str:
    # ── Build actionable terminology instructions ──
    qe = terminology_resolution.get("query_expansions") or {}
    search_obligations = qe.get("search_obligations") or {}
    required_pairs = search_obligations.get("required_pairs") or []
    blocked = qe.get("blocked_expansions") or []
    ambiguous = qe.get("ambiguous_surfaces") or []

    term_lines: list[str] = []
    if required_pairs:
        term_lines.append(
            "## 术语搜索义务（必须执行，不得跳过）"
        )
        term_lines.append(
            "以下每对术语，你必须分别用 search_text 搜索原文形式和规范名，"
            "两者都必须出现在 tool_trace 中："
        )
        for pair in required_pairs:
            source = pair.get("source", "")
            canonical = pair.get("canonical", "")
            term_lines.append(
                f"- 原文「{source}」→ 规范名「{canonical}」"
                f"（用 search_text 搜索这两个词）"
            )
        term_lines.append("")
    if blocked:
        term_lines.append("## 已阻止的术语扩展（不得用于搜索或锁定 Variant）")
        for item in blocked:
            term_lines.append(
                f"- 「{item.get('surface_form','')}」→「{item.get('canonical_name','')}」"
                f"：{item.get('reason','')}"
            )
        term_lines.append("")
    if ambiguous:
        term_lines.append("## 未消歧的术语（只能作为候选宽召回，不能锁定 Variant）")
        for item in ambiguous:
            term_lines.append(
                f"- 「{item.get('surface_form','')}」"
                f"（需要上下文：{item.get('required_context',[])}）"
            )
        term_lines.append("")

    required_search_groups = terminology_search_contract.get(
        "required_search_groups"
    ) or []
    sections = [
        SYSTEM_PROMPT,
        "",
        "## 当前任务",
        f"用户问题：{query}",
        "",
        *term_lines,
        "TERM_RESOLUTION（仅用于消歧和搜索扩展，不证明根因）：",
        json.dumps(terminology_resolution, ensure_ascii=False, indent=2),
        "",
        "TERMINOLOGY_SEARCH_CONTRACT（搜索义务与权限，不是预排序候选）：",
        json.dumps(
            terminology_search_contract,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "DETERMINISTIC_TERMINOLOGY_SEARCH_RESULTS（仅为发现线索；必须继续 read_text 原文后才能引用）：",
        json.dumps(
            terminology_search_execution,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "REQUIRED_FACETS（最终 ledger 必须逐项闭包）：",
        json.dumps(facets, ensure_ascii=False, indent=2),
        "",
        "ANSWER_SCOPE（限制回答展开边界；不得把相邻任务升级为主任务）：",
        json.dumps(answer_scope, ensure_ascii=False, indent=2),
    ]
    if previous_errors:
        sections.extend([
            "",
            "上一次候选答案未通过本地发布校验。重新独立调查并修复这些问题：",
            *[f"- {error}" for error in previous_errors],
        ])
    return "\n".join(sections)


def _draft_output_schema() -> dict[str, Any]:
    ledger_item = {
        "type": "object",
        "properties": {
            "facet_id": {"type": "string"},
            "label": {"type": "string"},
            "kind": {"type": "string"},
            "status": {"type": "string", "enum": ["covered", "gap"]},
            "source_paths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason": {"type": "string"},
        },
        "required": [
            "facet_id",
            "label",
            "kind",
            "status",
            "source_paths",
            "reason",
        ],
        "additionalProperties": False,
    }
    procedure_variant_item = {
        "type": "object",
        "properties": {
            "source_path": {"type": "string"},
            "source_label": {"type": "string"},
            "answer_label": {"type": "string"},
            "status": {
                "type": "string",
                "enum": [
                    "expanded",
                    "guarded",
                    "omitted_evidence_gap",
                ],
            },
            "reason": {"type": "string"},
        },
        "required": [
            "source_path",
            "source_label",
            "answer_label",
            "status",
            "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "const": _DRAFT_SCHEMA_VERSION,
            },
            "answer_markdown": {"type": "string"},
            "coverage_ledger": {
                "type": "array",
                "items": ledger_item,
            },
            "procedure_variant_ledger": {
                "type": "array",
                "items": procedure_variant_item,
            },
            "files_read": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "schema_version",
            "answer_markdown",
            "coverage_ledger",
            "procedure_variant_ledger",
            "files_read",
        ],
        "additionalProperties": False,
    }


def _discover_required_procedure_variants(
    query: str,
    draft: dict[str, Any],
    *,
    required_facets: list[RequiredFacet],
    answer_scope: AnswerScope,
    workspace: Path,
) -> list[ProcedureVariantRequirement]:
    """Discover peer procedures from sources that close requested tasks.

    This deliberately runs after the agent has selected and read evidence. It
    does not retrieve or rank documents. It only prevents composition from
    silently dropping explicit sibling paths present in the selected source.
    """

    if not _should_enforce_procedure_variants(
        query,
        answer_scope=answer_scope,
        required_facets=required_facets,
    ):
        return []

    task_facet_ids = {
        facet.facet_id
        for facet in required_facets
        if facet.kind == "query_task"
    }
    source_paths: list[str] = []
    for raw in draft.get("coverage_ledger") or []:
        if (
            not isinstance(raw, dict)
            or str(raw.get("status") or "") != "covered"
            or str(raw.get("facet_id") or "") not in task_facet_ids
        ):
            continue
        source_paths.extend(
            str(path).strip()
            for path in raw.get("source_paths") or []
            if str(path).strip().startswith("data/raw/")
        )

    # Wide recall that closes facets with many sources signals a broad
    # knowledge query, not a targeted procedure lookup.  Enforcing sibling
    # method coverage on every incidental source creates noise.
    unique_sources = list(dict.fromkeys(source_paths))
    if len(unique_sources) > 5:
        return []

    result: list[ProcedureVariantRequirement] = []
    seen: set[tuple[str, str]] = set()
    for source_path in dict.fromkeys(source_paths):
        text = _procedure_source_text(source_path, workspace=workspace)
        if not text:
            continue
        matches = list(_PROCEDURE_VARIANT_LABEL.finditer(text))
        if len(matches) < 2:
            continue
        for index, match in enumerate(matches):
            source_label = re.sub(
                r"\s+", "", str(match.group("label") or "")
            )
            key = (source_path, source_label)
            if not source_label or key in seen:
                continue
            seen.add(key)
            block_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )
            block = text[match.start():block_end]
            urls = tuple(dict.fromkeys(_URL.findall(block)))
            unverified_external = bool(
                urls
                and _EXTERNAL_ARTIFACT_WORDS.search(block)
                and not _INTEGRITY_MARKERS.search(block)
            )
            result.append(ProcedureVariantRequirement(
                source_path=source_path,
                source_label=source_label,
                external_urls=urls if unverified_external else (),
                external_artifact_unverified=unverified_external,
            ))
    return result


def _should_enforce_procedure_variants(
    query: str,
    *,
    answer_scope: AnswerScope,
    required_facets: list[RequiredFacet],
) -> bool:
    """Return whether sibling方案/方法 closure is part of this query.

    Many troubleshooting sources contain method ledgers as document structure.
    Requiring that ledger for every incidental source turns ordinary symptom
    questions into full SOP rewrites.  Keep the obligation for procedure-like
    requests and explicit方案/方法 questions only.
    """

    if answer_scope.request_kind in {"procedure_lookup", "comparison_lookup"}:
        return True
    intent_text = " ".join([
        query,
        *(facet.label for facet in required_facets),
        *(term for facet in required_facets for term in facet.match_terms),
    ])
    return bool(re.search(
        r"(?:方案|方法|步骤|流程|操作方法|处理路径)",
        intent_text,
    ))


def _procedure_source_text(source_path: str, *, workspace: Path) -> str:
    """Read the deterministic view of one already-selected source."""

    if not _SOURCE_PATH.match(source_path):
        return ""
    parts = Path(source_path).parts
    if len(parts) < 3:
        return ""
    scope = parts[1]
    relative = Path(*parts[2:])
    if relative.suffix.casefold() == ".docx":
        candidate = (
            workspace / "data/extracted_docx" / scope / relative
        ).with_suffix(".docx.md")
    else:
        candidate = workspace / "data" / scope / relative
    try:
        resolved = candidate.resolve()
        if not candidate.is_file():
            return ""
        return resolved.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _verify_native_draft(
    draft: dict[str, Any],
    *,
    required_facets: list[RequiredFacet],
    answer_scope: AnswerScope,
    media_inventory: list[dict[str, Any]],
    actual_files_read: list[str],
    tool_trace: list[dict[str, Any]],
    terminology_search_contract: dict[str, Any],
    required_procedure_variants: list[ProcedureVariantRequirement],
) -> list[str]:
    errors: list[str] = []
    if draft.get("schema_version") != _DRAFT_SCHEMA_VERSION:
        errors.append("invalid_draft_schema_version")
    files = draft.get("files_read")
    if not isinstance(files, list):
        return [*errors, "files_read_not_list"]
    files_read = [str(path).strip() for path in files if str(path).strip()]
    if len(files_read) != len(set(files_read)):
        errors.append("duplicate_files_read")
    actual = {
        str(path).strip()
        for path in actual_files_read
        if str(path).strip()
    }
    for path in files_read:
        try:
            _resolve_source_path(path)
        except ValueError as exc:
            errors.append(f"invalid_files_read_path:{path}:{exc}")
        if path not in actual:
            errors.append(f"claimed_unread_file:{path}")
    scopes = {
        path.split("/", 2)[1]
        for path in files_read
        if _SOURCE_PATH.match(path)
    }
    for scope in sorted(set(CORPUS_ROOTS) - scopes):
        errors.append(f"missing_read_scope:{scope}")
    errors.extend(terminology_search_errors(
        audit_terminology_search_contract(
            terminology_search_contract,
            tool_trace,
        )
    ))
    errors.extend(terminology_governance_authority_errors(
        draft,
        terminology_search_contract,
    ))
    errors.extend(verify_answer_draft(
        draft,
        required_facets=required_facets,
        files_read=files_read,
        media_exposed=media_inventory,
        answer_scope=answer_scope,
        required_procedure_variants=required_procedure_variants,
    ))
    return list(dict.fromkeys(errors))


def _resolve_source_path(value: str) -> Path:
    if not _SOURCE_PATH.match(value):
        raise ValueError("path_outside_allowed_corpus")
    candidate = (REPO_ROOT / value).resolve()
    if not any(
        candidate == root or root in candidate.parents
        for root in CORPUS_ROOTS.values()
    ):
        raise ValueError("path_outside_allowed_corpus")
    if not candidate.is_file():
        raise ValueError("corpus_file_not_found")
    return candidate


def _portable_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _load_pipeline_config() -> dict[str, Any]:
    path = REPO_ROOT / "config/kg_v2_raw_codex.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _structured_response_payload(output: list[Any]) -> dict[str, Any]:
    texts = [
        str(content.get("text") or "")
        for item in output
        if isinstance(item, dict)
        for content in (item.get("content") or [])
        if isinstance(content, dict)
        and content.get("type") == "output_text"
    ]
    if not texts:
        raise CodexResponsesAgentError("codex_missing_structured_output")
    try:
        payload = json.loads(texts[-1])
    except json.JSONDecodeError as exc:
        raise CodexResponsesAgentError(
            "codex_invalid_structured_output"
        ) from exc
    if not isinstance(payload, dict):
        raise CodexResponsesAgentError("codex_output_not_object")
    return payload


def _parse_codex_cli_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _codex_cli_tool_trace(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type not in {
            "command_execution", "mcp_tool_call", "web_search",
        }:
            continue
        output = str(
            item.get("aggregated_output")
            or item.get("output")
            or item.get("result")
            or ""
        )
        trace.append({
            "type": item_type,
            "status": str(item.get("status") or "completed"),
            "command": str(item.get("command") or "")[:2000],
            "exit_code": item.get("exit_code"),
            "output_chars": len(output),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        })
    return trace


def _codex_cli_audited_files(
    draft: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    workspace: Path,
) -> list[str]:
    """Cross-check declared evidence against the CLI's emitted tool trace.

    Codex CLI does not expose an OS-level read ledger. Its JSONL stream does
    include executed commands and their outputs, so a source is accepted only
    when its canonical path (or deterministic DOCX view path) appears there.
    """

    serialized = json.dumps(events, ensure_ascii=False)
    result: list[str] = []
    for raw in draft.get("files_read") or []:
        source = str(raw or "").strip()
        try:
            source_path = _resolve_source_path(source)
        except ValueError:
            continue
        candidates = {source, str(workspace / source)}
        if source_path.suffix.lower() == ".docx":
            parts = Path(source).parts
            if len(parts) >= 3:
                extracted = (
                    Path("data/extracted_docx")
                    / parts[1]
                    / Path(*parts[2:])
                ).with_suffix(".docx.md").as_posix()
                candidates.update({extracted, str(workspace / extracted)})
        if any(candidate in serialized for candidate in candidates):
            result.append(source)
    return list(dict.fromkeys(result))


def _normalize_cli_draft_paths(
    draft: dict[str, Any],
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize CLI workspace paths and deterministic document views."""

    mapping: dict[str, str] = {}
    extracted = workspace / "data/extracted_docx"
    for path in extracted.rglob("*.md"):
        first_line = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[:1]
        if not first_line or not first_line[0].startswith("SOURCE_PATH:"):
            continue
        source = first_line[0].split(":", 1)[1].strip()
        if _SOURCE_PATH.match(source):
            mapping[path.relative_to(workspace).as_posix()] = source

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            result = value
            workspace_prefix = workspace.as_posix().rstrip("/")
            for root in ("raw", "kg_v2"):
                result = result.replace(
                    f"{workspace_prefix}/data/{root}/",
                    f"data/{root}/",
                )
                source_prefix = (
                    CORPUS_ROOTS[root].resolve().as_posix().rstrip("/")
                )
                result = result.replace(
                    f"{source_prefix}/",
                    f"data/{root}/",
                )
            # Codex can execute through an internal read-only mount whose
            # temporary directory name differs from the host workspace.  The
            # evidence-root suffix is stable, so strip only an absolute temp
            # prefix immediately before ``data/raw`` or ``data/kg_v2``.
            result = re.sub(
                r"(?<![A-Za-z0-9_.-])/(?:tmp|private/tmp)/"
                r"[^/\s)\]】]+/data/(raw|kg_v2)/",
                r"data/\1/",
                result,
            )
            for temporary, source in mapping.items():
                result = result.replace(temporary, source)
            # Codex may put harmless padding inside Markdown link targets.
            # Strip only target-edge whitespace; do not alter labels or paths.
            result = re.sub(
                r"(!?\[[^\]\n]*\]\()\s+([^)]+?)\s*(\))",
                r"\1\2\3",
                result,
            )
            seen_images: set[str] = set()

            def keep_first_image(match: re.Match[str]) -> str:
                path = match.group(2).strip()
                if path in seen_images:
                    return ""
                seen_images.add(path)
                return match.group(1)

            result = re.sub(
                r"(!\[[^\]\n]*\]\(([^)\n]+)\))",
                keep_first_image,
                result,
            )
            return result
        return value

    normalized = normalize(draft)
    return normalized if isinstance(normalized, dict) else draft


def _finalize_draft_contract(
    draft: dict[str, Any],
    *,
    actual_files_read: list[str],
) -> dict[str, Any]:
    """Apply runtime-neutral, evidence-preserving output normalization."""

    result = dict(draft)
    answer = str(result.get("answer_markdown") or "").strip()

    seen_images: set[str] = set()

    def keep_first_image(match: re.Match[str]) -> str:
        path = match.group(2).strip()
        if path in seen_images:
            return ""
        seen_images.add(path)
        return match.group(1)

    answer = re.sub(
        r"(!\[[^\]\n]*\]\(([^)\n]+)\))",
        keep_first_image,
        answer,
    )
    answer = re.sub(
        r"`((?:【来源：[^】\n]+】)+)`",
        r"\1",
        answer,
    )

    read_set = {
        str(path).strip()
        for path in actual_files_read
        if str(path).strip()
    }
    missing_markers: list[str] = []
    for entry in result.get("coverage_ledger") or []:
        if (
            not isinstance(entry, dict)
            or str(entry.get("status") or "") != "covered"
        ):
            continue
        for raw_path in entry.get("source_paths") or []:
            source = str(raw_path or "").strip()
            marker = f"【来源：{source}】"
            if (
                source
                and source in read_set
                and marker not in answer
                and source not in missing_markers
            ):
                missing_markers.append(source)
    if missing_markers:
        source_lines = "\n".join(
            f"- 【来源：{source}】"
            for source in missing_markers
        )
        answer = (
            f"{answer}\n\n## 证据来源补充\n{source_lines}"
        ).strip()

    result["answer_markdown"] = answer
    return result


def _codex_cli_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        _add_usage(total, usage)
    return total


def _codex_cli_thread_id(events: list[dict[str, Any]]) -> str:
    return next(
        (
            str(event.get("thread_id") or "")
            for event in events
            if event.get("type") == "thread.started"
        ),
        "",
    )


def _codex_cli_failure(events: list[dict[str, Any]]) -> str:
    messages: list[str] = []
    for event in events:
        if event.get("type") in {"error", "turn.failed"}:
            value = event.get("message") or event.get("error") or ""
            if isinstance(value, dict):
                value = value.get("message") or value
            messages.append(str(value))
    detail = messages[-1] if messages else "nonzero_exit_without_final_answer"
    detail = re.sub(r"\s+", " ", detail).strip()
    return detail[:500]


def _safe_logical_path(value: Any) -> str:
    logical = str(value or "").strip().replace("\\", "/")
    path = Path(logical)
    if (
        not logical.startswith("data/")
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in logical
    ):
        raise ValueError("path_outside_corpus")
    return path.as_posix()


def _safe_glob(value: Any) -> str:
    pattern = str(value or "").strip().replace("\\", "/")
    if not pattern:
        pattern = "data/**/*"
    if (
        not pattern.startswith("data/")
        or Path(pattern).is_absolute()
        or ".." in Path(pattern).parts
        or "\x00" in pattern
    ):
        raise ValueError("glob_outside_corpus")
    return pattern


def _safe_tool_arguments(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "glob", "limit", "query", "path_glob", "regex", "case_sensitive",
        "max_matches", "context_lines", "path", "start_line", "end_line",
    }
    return {
        key: (item[:500] if isinstance(item, str) else item)
        for key, item in value.items()
        if key in allowed
    }


def _add_usage(total: dict[str, int], current: dict[str, Any]) -> None:
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = current.get(candidate)
            if isinstance(value, (int, float)):
                normalized[target] = int(value)
                break
    if "total_tokens" not in normalized and {
        "prompt_tokens", "completion_tokens"
    } <= set(normalized):
        normalized["total_tokens"] = (
            normalized["prompt_tokens"]
            + normalized["completion_tokens"]
        )
    for key, value in normalized.items():
        total[key] = total.get(key, 0) + value


def _deduplicate_media(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        identity = (
            str(value.get("media_kind") or ""),
            str(value.get("asset_path") or ""),
        )
        if not identity[1] or identity in seen:
            continue
        seen.add(identity)
        result.append(value)
    return result


def _media_used_by_answer(
    answer: str,
    *,
    media_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cited = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", answer))
    cited.update(re.findall(r"\[[^\]]+\]\(([^)]+)\)", answer))
    return [
        item
        for item in media_inventory
        if str(item.get("asset_path") or "") in cited
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runtime",
        choices=("responses_api", "codex_cli"),
        help="Override the configured agent transport.",
    )
    parser.add_argument(
        "--model",
        help="Override the model for the selected transport.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        help="Compatibility alias for maximum verification attempts.",
    )
    parser.add_argument(
        "--no-terminology",
        action="store_true",
        help="Disable only the terminology layer for controlled A/B runs.",
    )
    args = parser.parse_args()
    payload = run(
        args.query,
        args.output,
        max_rounds=args.max_rounds,
        runtime=args.runtime,
        model=args.model,
        terminology_enabled=not args.no_terminology,
    )
    print(json.dumps({
        "output": str(args.output),
        "answer_length": len(payload["answer"]),
        "files_read": payload["files_read"],
        "tool_calls": len(payload["tool_trace"]),
        "usage": payload["usage"],
        "runtime": payload["runtime"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
