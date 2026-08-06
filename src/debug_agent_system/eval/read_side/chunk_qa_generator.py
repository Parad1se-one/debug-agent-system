"""Model-read, source-grounded QA generation from individual document chunks.

Unlike the legacy feature-selftest builder, this module never turns a title
into a question with string templates.  One model request receives exactly one
chunk, decides whether that chunk can support a benchmark case, and either
returns a grounded query-answer pair or rejects the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Protocol

from debug_agent_system.adapters.codex_read.client import CodexResponsesClient


SCHEMA_VERSION = "debug_agent_system.chunk_qa_benchmark.v1"
GENERATOR_VERSION = "model_read_per_chunk.v1"

_SPACE = re.compile(r"\s+")
_NORMALIZE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)
_BAD_QUERY_PHRASES = (
    "根据以上",
    "根据文档",
    "根据chunk",
    "根据 chunk",
    "这段内容",
    "上述资料",
    "原文",
    "按哪个处理动作",
    "按什么现场处理",
    "按什么方式处理",
    "执行什么处理",
    "这个模块",
    "报报警",
)
_BAD_ANSWER_PHRASES = ("处理结果说明", "根据原文", "上述内容")
_ANSWER_ACTION = re.compile(
    r"检查|进入|打开|关闭|保存|复制|覆盖|替换|删除|修改|设置|调整|"
    r"拔插|插拔|断电|重启|升级|安装|下载|运行|清理|紧固|更换"
)
_TECHNICAL_ANCHOR = re.compile(
    r"\d|[/\\]|\.(?:json|toml|bat|exe|log|cfg|dll)\b|"
    r"error|timeout|ip|端口|日志|版本|参数|阈值",
    re.IGNORECASE,
)


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "rejection_reason",
        "query",
        "reference_answer",
        "evidence_excerpts",
        "query_type",
        "product",
        "module",
        "quality_notes",
    ],
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "rejection_reason": {"type": "string"},
        "query": {"type": "string"},
        "reference_answer": {"type": "string"},
        "evidence_excerpts": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "query_type": {
            "type": "string",
            "enum": [
                "diagnosis",
                "procedure",
                "configuration",
                "mechanism",
                "comparison",
            ],
        },
        "product": {"type": "string"},
        "module": {"type": "string"},
        "quality_notes": {"type": "string"},
    },
}


SYSTEM_INSTRUCTIONS = """你是 AOI Debug Benchmark 的资深标注员。每次只阅读一个 chunk，并独立判断它能否支持一对高质量 Query-Answer。

核心规则：
1. 不准用标题拼接“怎么处理”；必须理解正文后，将信息改写为真实现场工程师会提出的单一问题。
2. Query 要包含足以检索和区分故障的现场现象、对象、版本、报错或条件，但不能泄露答案；不要提“文档、chunk、上述内容”。
   metadata 仅用于来源审计；不要把 site、date、handler 填入 Query，也不要用 metadata 补写 chunk 正文没有的故障事实。
   使用自然、直接的现场问法；禁止“原文提供的是什么”“按哪个处理动作”“这类场景如何操作”等考试腔或占位表达。
3. 一条 Query 只考一个连贯意图。原文混有多个无关问题时，只能选择其中证据闭合的一项，否则 reject。
4. reference_answer 只能使用当前 chunk 明确提供的信息。保留原文中的版本、路径、命令、日志字段、文件名、参数值和操作顺序；禁止补充常识、KG 内容或通用安全话术。
5. evidence_excerpts 必须是当前 chunk 中逐字连续出现的原文片段，并足以直接支撑答案。不得改写摘录。
6. 若 chunk 只有问题/标题、处理结果过短且含义不清、关键答案只在缺失图片中、指代无法解析、或不能形成可判分答案，必须 reject。宁缺毋滥。
   不要为了凑数量修补语病或猜测标题含义；标题本身无法确定故障现象时也应 reject。
7. accept 时 query、reference_answer、evidence_excerpts 都必须非空；reject 时三者必须为空，并给出具体 rejection_reason。
8. 不要把“收集日志/远程看看/升级最新版本”单独当作高质量答案，除非原文还明确说明检查对象、判断信号或操作细节。
9. quality_notes 只写一句简短的可审计说明，不写思维过程。
"""


REVIEW_INSTRUCTIONS = """你是 AOI Debug Benchmark 的独立终审员。你会收到一个且仅一个原始 chunk，以及初审生成的一对候选 Query-Answer。逐项审查并直接输出修订后的最终结果或 reject。

终审标准：
1. Query 必须像现场工程师自然提出的问题，独立可读、对象明确、语法通顺；不得有模糊指代（如“这个模块/这个问题”）、重复词（如“报报警”）、考试腔（如“按哪个处理动作/原文提供什么”）或标题机械加尾缀。
2. Query 只能使用 chunk 正文里的故障事实；不得加入 site、date、handler，不得通过答案反向泄露解决步骤。
3. Answer 必须直接回答 Query，并且所有事实都由当前 chunk 支撑；删除排期、人员、培训等与 Query 无关的旁支信息。版本、路径、命令、日志、参数和文件名必须准确保留。
4. evidence_excerpts 必须是 chunk 中逐字连续出现的片段，并充分支撑最终 Answer。
5. 只有“清理/调整/升级/重启/远程处理/反馈”等泛化动作、却没有具体对象、方法、判断信号或技术锚点的候选，不足以成为高质量 benchmark，应 reject。
6. 原始标题含义不清、关键图片缺失、答案无法判分时必须 reject；不要猜测或修补原始资料。
7. 若问题可在不引入新事实的前提下改好，直接重写并返回 accept；否则返回 reject。reject 时 query、reference_answer、evidence_excerpts 必须为空。
8. quality_notes 只写一句终审结论，不写思维过程。
"""


class ResponsesClient(Protocol):
    last_usage: dict[str, Any]
    last_request_id: str

    def create(self, body_payload: dict[str, Any]) -> dict[str, Any]: ...


class CodexCliStructuredClient:
    """Adapt ``codex exec`` structured output to the small Responses protocol.

    This fallback uses the local Codex login and is useful when the repository
    intentionally does not store an API key.  Each invocation is ephemeral,
    read-only, and receives only the one chunk embedded in the prompt.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = 600,
        codex_binary: str = "codex",
    ) -> None:
        binary = shutil.which(codex_binary)
        if not binary:
            raise RuntimeError("codex_cli_not_found")
        self.binary = binary
        self.timeout_seconds = max(30, int(timeout_seconds))
        self.last_usage: dict[str, Any] = {}
        self.last_request_id = ""

    def create(self, body_payload: dict[str, Any]) -> dict[str, Any]:
        model = str(body_payload.get("model") or "").strip()
        instructions = str(body_payload.get("instructions") or "").strip()
        input_texts = [
            str(content.get("text") or "")
            for item in body_payload.get("input") or []
            if isinstance(item, dict)
            for content in item.get("content") or []
            if isinstance(content, dict) and content.get("type") == "input_text"
        ]
        schema = (
            ((body_payload.get("text") or {}).get("format") or {}).get("schema")
        )
        if not isinstance(schema, dict):
            raise ValueError("codex_cli_missing_output_schema")
        prompt = instructions + "\n\n" + "\n\n".join(input_texts)
        reasoning = str(
            ((body_payload.get("reasoning") or {}).get("effort") or "medium")
        )

        with tempfile.TemporaryDirectory(prefix="chunk-qa-codex-") as raw_tmp:
            tmp = Path(raw_tmp)
            schema_path = tmp / "schema.json"
            output_path = tmp / "output.json"
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                self.binary,
                "exec",
                "--ephemeral",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(tmp),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-c",
                f'model_reasoning_effort="{reasoning}"',
            ]
            if model:
                command.extend(["--model", model])
            command.append("-")
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"codex_cli_failed:{completed.returncode}")
            if not output_path.is_file():
                raise RuntimeError("codex_cli_missing_output")
            raw_output = output_path.read_text(encoding="utf-8").strip()
            payload = json.loads(raw_output)
            if not isinstance(payload, dict):
                raise ValueError("codex_cli_output_not_object")
            self.last_usage = _codex_cli_usage(completed.stdout)
        return {
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }],
            }],
        }


def _codex_cli_usage(stdout: str) -> dict[str, int]:
    usage: dict[str, int] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        raw = event.get("usage")
        if not isinstance(raw, dict):
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = raw.get(key)
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
    return usage


@dataclass(slots=True)
class GeneratedChunkCase:
    payload: dict[str, Any]
    audit: dict[str, Any]


def clean_text(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def normalized(value: Any) -> str:
    return _NORMALIZE.sub("", clean_text(value).lower())


def trigrams(value: Any) -> set[str]:
    text = normalized(value)
    if len(text) < 3:
        return {text} if text else set()
    return {text[index : index + 3] for index in range(len(text) - 2)}


def similarity(left: Any, right: Any) -> float:
    left_grams = trigrams(left)
    right_grams = trigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _without_question_suffix(value: Any) -> str:
    text = normalized(value)
    for suffix in ("该怎么处理", "怎么处理", "如何处理", "怎么办"):
        normalized_suffix = normalized(suffix)
        if text.endswith(normalized_suffix):
            return text[: -len(normalized_suffix)]
    return text


def chunk_sha256(chunk: dict[str, Any]) -> str:
    canonical = json.dumps(chunk, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _structured_payload(response: dict[str, Any]) -> dict[str, Any]:
    texts = [
        str(content.get("text") or "")
        for item in response.get("output") or []
        if isinstance(item, dict)
        for content in item.get("content") or []
        if isinstance(content, dict) and content.get("type") == "output_text"
    ]
    if not texts:
        raise ValueError("missing_structured_output")
    payload = json.loads(texts[-1])
    if not isinstance(payload, dict):
        raise ValueError("structured_output_not_object")
    return payload


def _excerpt_in_source(excerpt: str, source_text: str) -> bool:
    return clean_text(excerpt) in clean_text(source_text)


def validate_model_payload(
    payload: dict[str, Any],
    *,
    chunk: dict[str, Any],
    existing_queries: list[str],
    duplicate_limit: float = 0.84,
) -> list[str]:
    """Return deterministic quality failures for a model decision."""

    errors: list[str] = []
    decision = payload.get("decision")
    if decision not in {"accept", "reject"}:
        return ["invalid_decision"]
    if decision == "reject":
        if not clean_text(payload.get("rejection_reason")):
            errors.append("reject_without_reason")
        if any(
            clean_text(payload.get(key))
            for key in ("query", "reference_answer")
        ) or payload.get("evidence_excerpts"):
            errors.append("reject_contains_case_content")
        return errors

    source_text = str(chunk.get("text") or "")
    metadata = chunk.get("metadata") or {}
    title = clean_text(metadata.get("title") if isinstance(metadata, dict) else "")
    query = clean_text(payload.get("query"))
    answer = clean_text(payload.get("reference_answer"))
    excerpts = payload.get("evidence_excerpts")

    if len(query) < 18 or len(query) > 180:
        errors.append("query_length_out_of_range")
    if query and not query.endswith(("？", "?")):
        errors.append("query_not_question")
    if any(phrase.lower() in query.lower() for phrase in _BAD_QUERY_PHRASES):
        errors.append("query_mentions_source_context")
    if "怎么处理怎么处理" in query or "该怎么处理怎么处理" in query:
        errors.append("query_template_artifact")
    if title and _without_question_suffix(query) == normalized(title):
        errors.append("query_is_title_copy")
    if len(answer) < 12:
        errors.append("answer_too_short")
    elif len(answer) < 18:
        actions = set(_ANSWER_ACTION.findall(answer))
        if len(actions) < 2 and not _TECHNICAL_ANCHOR.search(answer):
            errors.append("answer_too_thin")
    if any(phrase in answer for phrase in _BAD_ANSWER_PHRASES):
        errors.append("answer_contains_source_meta_language")
    if not isinstance(excerpts, list) or not excerpts:
        errors.append("missing_evidence_excerpts")
    else:
        for index, excerpt in enumerate(excerpts):
            if len(clean_text(excerpt)) < 6:
                errors.append(f"evidence_{index}_too_short")
            elif not _excerpt_in_source(str(excerpt), source_text):
                errors.append(f"evidence_{index}_not_verbatim")

    for prior in existing_queries:
        score = similarity(query, prior)
        if score >= duplicate_limit:
            errors.append(f"query_near_duplicate:{score:.3f}")
            break
    return errors


def _prompt_for_chunk(
    chunk: dict[str, Any],
    *,
    chunk_index: int,
    correction_errors: list[str] | None = None,
) -> str:
    metadata = chunk.get("metadata")
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    prompt = {
        "task": "read_one_chunk_and_propose_or_reject_one_qa_pair",
        "chunk_index": chunk_index,
        "metadata": safe_metadata,
        "chunk_text": str(chunk.get("text") or ""),
    }
    if correction_errors:
        prompt["previous_validation_errors"] = correction_errors
        prompt["instruction"] = (
            "上一次输出未通过确定性校验。重新阅读同一 chunk 并修订；"
            "若无法忠实修订，返回 reject。"
        )
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _prompt_for_review(
    chunk: dict[str, Any],
    *,
    chunk_index: int,
    candidate: dict[str, Any],
) -> str:
    metadata = chunk.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return json.dumps(
        {
            "task": "adversarially_review_and_rewrite_or_reject_candidate",
            "chunk_index": chunk_index,
            "metadata_for_audit_only": metadata,
            "chunk_text": str(chunk.get("text") or ""),
            "candidate": candidate,
        },
        ensure_ascii=False,
        indent=2,
    )


def _request_structured(
    client: ResponsesClient,
    *,
    model: str,
    instructions: str,
    prompt: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    response = client.create({
        "model": model,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "chunk_qa_decision",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "store": False,
    })
    return _structured_payload(response)


def generate_one_chunk(
    client: ResponsesClient,
    *,
    model: str,
    chunk: dict[str, Any],
    chunk_index: int,
    existing_queries: list[str],
    reasoning_effort: str = "medium",
    max_attempts: int = 2,
    review: bool = True,
    review_model: str | None = None,
) -> GeneratedChunkCase:
    """Have the model read one chunk, then independently review accepted work."""

    attempts: list[dict[str, Any]] = []
    correction_errors: list[str] | None = None
    final_payload: dict[str, Any] = {}
    final_errors: list[str] = []
    for attempt in range(1, max(1, max_attempts) + 1):
        final_payload = _request_structured(
            client,
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            prompt=_prompt_for_chunk(
                chunk,
                chunk_index=chunk_index,
                correction_errors=correction_errors,
            ),
            reasoning_effort=reasoning_effort,
        )
        final_errors = validate_model_payload(
            final_payload,
            chunk=chunk,
            existing_queries=existing_queries,
        )
        attempts.append({
            "attempt": attempt,
            "stage": "author",
            "request_id": client.last_request_id,
            "usage": dict(client.last_usage),
            "decision": final_payload.get("decision"),
            "validation_errors": final_errors,
        })
        if not final_errors:
            break
        correction_errors = final_errors

    if not final_errors and final_payload.get("decision") == "accept" and review:
        reviewed = _request_structured(
            client,
            model=review_model or model,
            instructions=REVIEW_INSTRUCTIONS,
            prompt=_prompt_for_review(
                chunk,
                chunk_index=chunk_index,
                candidate=final_payload,
            ),
            reasoning_effort=reasoning_effort,
        )
        review_errors = validate_model_payload(
            reviewed,
            chunk=chunk,
            existing_queries=existing_queries,
        )
        attempts.append({
            "attempt": 1,
            "stage": "review",
            "request_id": client.last_request_id,
            "usage": dict(client.last_usage),
            "decision": reviewed.get("decision"),
            "validation_errors": review_errors,
        })
        if review_errors:
            final_errors = [f"review:{error}" for error in review_errors]
        else:
            final_payload = reviewed

    if final_errors:
        final_payload = {
            "decision": "reject",
            "rejection_reason": "model_output_failed_validation:"
            + ",".join(final_errors),
            "query": "",
            "reference_answer": "",
            "evidence_excerpts": [],
            "query_type": "diagnosis",
            "product": "",
            "module": "",
            "quality_notes": "确定性校验未通过，未纳入 benchmark。",
        }

    return GeneratedChunkCase(
        payload=final_payload,
        audit={
            "generator_version": GENERATOR_VERSION,
            "model": model,
            "review_model": review_model or model if review else "",
            "chunk_index": chunk_index,
            "chunk_sha256": chunk_sha256(chunk),
            "attempts": attempts,
        },
    )


def build_case_record(
    generated: GeneratedChunkCase,
    *,
    chunk: dict[str, Any],
    case_number: int,
) -> dict[str, Any]:
    payload = generated.payload
    metadata = chunk.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": f"chunk-qa-{case_number:04d}",
        "query": clean_text(payload.get("query")),
        "answer_gold": {
            "reference_answer": clean_text(payload.get("reference_answer")),
            "evidence_excerpts": payload.get("evidence_excerpts") or [],
            "answer_mode": "single_chunk_source_grounded",
        },
        "classification": {
            "query_type": payload.get("query_type"),
            "product": clean_text(payload.get("product")),
            "module": clean_text(payload.get("module")),
        },
        "source": {
            "origin": metadata.get("source"),
            "title": metadata.get("title"),
            "section_num": metadata.get("section_num"),
            "date": metadata.get("date"),
            "site": metadata.get("site"),
            "handler": metadata.get("handler"),
            "chunk_index": generated.audit["chunk_index"],
            "chunk_sha256": generated.audit["chunk_sha256"],
            "chunk_text": str(chunk.get("text") or ""),
        },
        "quality": {
            "generation_method": GENERATOR_VERSION,
            "source_grounded": True,
            "model_quality_notes": clean_text(payload.get("quality_notes")),
        },
        "generation_audit": generated.audit,
    }


def make_client(
    *,
    env_file: Path,
    timeout_seconds: int,
) -> CodexResponsesClient:
    return CodexResponsesClient(
        env_file=env_file,
        timeout_seconds=timeout_seconds,
    )


def make_cli_client(*, timeout_seconds: int) -> CodexCliStructuredClient:
    return CodexCliStructuredClient(timeout_seconds=timeout_seconds)


__all__ = [
    "GENERATOR_VERSION",
    "OUTPUT_SCHEMA",
    "SCHEMA_VERSION",
    "GeneratedChunkCase",
    "CodexCliStructuredClient",
    "build_case_record",
    "chunk_sha256",
    "generate_one_chunk",
    "make_client",
    "make_cli_client",
    "similarity",
    "validate_model_payload",
]
