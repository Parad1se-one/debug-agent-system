"""DeepSeek Chat Completions agent runner for the KG_v2+raw read path.

The independent bypass is model-directed: DeepSeek decides which corpus tool
to call, what to search, and when to stop.  Local code keeps the same corpus
boundary, deterministic tool execution, tool trace and structured-output
verification as the Responses and Codex CLI runners, so a batch can switch
runtimes without changing the verifier contract.

The runtime uses the official OpenAI-compatible Chat Completions endpoint
(``https://api.deepseek.com/chat/completions``).  It deliberately does not
reuse the Responses API transport because DeepSeek does not expose
``/v1/responses``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekChatCompletionsError(RuntimeError):
    """A sanitized DeepSeek chat-completions failure safe for audit logs."""


def _corpus_tools(workspace: Path):
    """Late import avoids a circular import with ``kg_raw_codex.pipeline``."""

    from debug_agent_system.kg_raw_codex.pipeline import CorpusReadTools

    return CorpusReadTools(workspace)


def _read_local_env(path: str | Path | None) -> dict[str, str]:
    """Read simple KEY=VALUE entries without mutating process environment."""

    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _chat_completions_url(base_url: str) -> str:
    """Accept an API root, a /v1 root, or a full chat-completions endpoint."""

    value = str(base_url or "").strip().rstrip("/")
    if not value:
        return DEEPSEEK_CHAT_COMPLETIONS_URL
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


class DeepSeekChatCompletionsClient:
    """Minimal non-streaming Chat Completions transport for DeepSeek."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "",
        timeout_seconds: int = 600,
        env_file: str | Path | None = None,
        max_attempts: int = 3,
    ) -> None:
        local = _read_local_env(env_file)
        self.api_key = api_key or local.get("DEEPSEEK_API_KEY", "")
        self.base_url = _chat_completions_url(
            base_url or os.environ.get("DEEPSEEK_BASE_URL", "")
        )
        self.timeout_seconds = max(30, int(timeout_seconds))
        self.max_attempts = max(1, min(4, int(max_attempts)))
        self.last_usage: dict[str, Any] = {}
        self.last_request_id = ""

    def create(self, body_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise DeepSeekChatCompletionsError("missing_DEEPSEEK_API_KEY")
        body = json.dumps(body_payload, ensure_ascii=False).encode("utf-8")
        request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            req = request.Request(
                self.base_url,
                data=body,
                method="POST",
                headers=request_headers,
            )
            try:
                with request.urlopen(
                    req, timeout=self.timeout_seconds
                ) as response:
                    self.last_request_id = str(
                        response.headers.get("x-request-id") or ""
                    )
                    payload = json.loads(
                        response.read().decode("utf-8")
                    )
                self.last_usage = (
                    payload.get("usage")
                    if isinstance(payload.get("usage"), dict)
                    else {}
                )
                return payload
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = DeepSeekChatCompletionsError(
                    f"deepseek_http_{exc.code}:{detail}"
                )
                if (
                    exc.code not in RETRYABLE_HTTP_CODES
                    or attempt >= self.max_attempts
                ):
                    raise last_error from exc
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = DeepSeekChatCompletionsError(
                    f"deepseek_transport:{type(exc).__name__}:{exc}"
                )
                if attempt >= self.max_attempts:
                    raise last_error from exc
            time.sleep(min(4.0, float(2 ** (attempt - 1))))
        raise DeepSeekChatCompletionsError(
            str(last_error or "deepseek_unknown_transport_failure")
        )


def _to_chat_completions_tools(
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Responses-style tool schemas to Chat Completions format."""

    converted: list[dict[str, Any]] = []
    for item in schemas:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(item.get("description") or ""),
                "parameters": item.get("parameters") or {
                    "type": "object",
                    "properties": {},
                },
            },
        })
    return converted


def _extract_draft_json(content: str) -> dict[str, Any]:
    """Parse the final answer payload with markdown-fence and prose tolerance.

    DeepSeek often prepends a short reasoning note before the JSON object and
    wraps the payload in a ```json fence.  Try, in order: whole-text parse,
    fenced parse, then first-``{``-to-last-``}`` extraction.
    """

    text = str(content or "").strip()
    if not text:
        raise DeepSeekChatCompletionsError("deepseek_empty_final_content")

    def _as_dict(value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    # 1. Whole-text parse.
    try:
        payload = _as_dict(json.loads(text))
        if payload is not None:
            return payload
    except json.JSONDecodeError:
        pass

    # 2. Markdown fence (with or without a language tag).
    fenced = re.search(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        text,
        flags=re.S,
    )
    if fenced:
        try:
            payload = _as_dict(json.loads(fenced.group(1).strip()))
            if payload is not None:
                return payload
        except json.JSONDecodeError:
            pass

    # 3. First ``{`` to last ``}`` (covers prose + JSON without a fence).
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = _as_dict(json.loads(text[start : end + 1]))
            if payload is not None:
                return payload
        except json.JSONDecodeError as exc:
            raise DeepSeekChatCompletionsError(
                "deepseek_invalid_structured_output"
            ) from exc

    raise DeepSeekChatCompletionsError("deepseek_invalid_structured_output")


@dataclass(slots=True)
class DeepSeekChatAgentRunner:
    """Let DeepSeek investigate through the same generic corpus tools."""

    client: DeepSeekChatCompletionsClient
    model: str = _DEFAULT_MODEL
    max_tool_rounds: int = 24
    max_tool_calls: int = 80
    max_tokens: int = 16000
    runtime_metadata: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.max_tool_rounds = max(1, int(self.max_tool_rounds))
        self.max_tool_calls = max(1, int(self.max_tool_calls))
        self.max_tokens = max(4096, int(self.max_tokens))
        self.runtime_metadata = {
            "engine": "deepseek",
            "transport": "non_streaming",
            "agent_loop": "model_directed_function_calls",
            "sandbox": "corpus_read_only",
            "api": "chat_completions",
            "credential_source": ".env.local",
            "model": self.model,
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
        tools = _corpus_tools(workspace)
        chat_tools = _to_chat_completions_tools(tools.schemas())
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Follow the supplied AOI evidence investigation contract. "
                    "Use the corpus tools iteratively before answering. "
                    "收敛规则（必须遵守）：\n"
                    "1. 优先用 search_text 精确搜索术语义务中的词，不要反复"
                    "list_files 枚举目录；\n"
                    "2. 一次搜索命中关键来源后，用 read_text 读取证据；\n"
                    "3. 当 required facets 的 covered 来源都已读到、证据足以"
                    "组织答案时，立即停止调用工具，直接输出最终 JSON；\n"
                    "4. 不要重复已经执行过的搜索词或已读取的文件；\n"
                    "5. 最多再调用 12 次工具就必须输出最终 JSON。\n"
                    "图片引用规则（必须遵守）：\n"
                    "1. 只引用 read_text 视图里 [source_media] 行给出的"
                    "asset_path 完整字符串（如 ![...](kg_v2_raw_assets/...))；\n"
                    "2. 绝不引用 embedded_media、data/raw/.../embedded_media/"
                    "或任何未在 [source_media] 行出现的路径；\n"
                    "3. 没有对应 asset_path 时不要引用图片，宁可省略；\n"
                    "4. 每个引用路径必须在文档视图中逐字出现，不能改写或拼接。\n"
                    "最终必须只输出符合给定 JSON Schema 的 JSON 对象，"
                    "不要输出 Markdown 代码块、解释或额外文本。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        trace: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        request_ids: list[str] = []
        call_count = 0

        for round_index in range(1, self.max_tool_rounds + 2):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": chat_tools,
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
            if str(self.model).startswith("deepseek-v4-"):
                payload["thinking"] = {"type": "disabled"}
            raw = self.client.create(payload)
            if self.client.last_request_id:
                request_ids.append(self.client.last_request_id)
            _accumulate_usage(usage, self.client.last_usage)

            choices = raw.get("choices") or []
            if not choices:
                raise DeepSeekChatCompletionsError(
                    "deepseek_no_choices"
                )
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise DeepSeekChatCompletionsError(
                    "deepseek_no_message"
                )
            finish_reason = str(choices[0].get("finish_reason") or "")
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                content = str(message.get("content") or "")
                if not content.strip():
                    raise DeepSeekChatCompletionsError(
                        "deepseek_empty_final_content"
                    )
                draft = _extract_draft_json(content)
                draft = _normalize_draft_paths(
                    draft,
                    _build_path_aliases(workspace),
                )
                return draft, {
                    "thread_id": str(raw.get("id") or ""),
                    "request_ids": request_ids,
                    "usage": usage,
                    "tool_trace": trace,
                    "files_read": sorted(tools.files_read),
                    "tool_rounds": round_index - 1,
                    "tool_calls": call_count,
                    "stderr_warnings": [],
                    "process_returncode": 0,
                }

            if round_index > self.max_tool_rounds:
                raise DeepSeekChatCompletionsError(
                    "deepseek_tool_round_limit"
                )
            # Carry the assistant tool_calls message into the next turn.
            messages.append(message)
            for call in tool_calls:
                call_count += 1
                if call_count > self.max_tool_calls:
                    raise DeepSeekChatCompletionsError(
                        "deepseek_tool_call_limit"
                    )
                function = (
                    call.get("function")
                    if isinstance(call.get("function"), dict)
                    else {}
                )
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or "{}"
                result, audit = tools.execute(name, arguments)
                audit.update({
                    "round": round_index,
                    "call_id": str(call.get("id") or ""),
                    "finish_reason": finish_reason,
                })
                trace.append(audit)
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": json.dumps(result, ensure_ascii=False),
                })
        raise DeepSeekChatCompletionsError("deepseek_agent_no_final_answer")


def _accumulate_usage(total: dict[str, int], current: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = current.get(key)
        if isinstance(value, (int, float)):
            total[key] = int(total.get(key, 0)) + int(value)


__all__ = [
    "DeepSeekChatAgentRunner",
    "DeepSeekChatCompletionsClient",
    "DeepSeekChatCompletionsError",
]

_SOURCE_PATH_RE = re.compile(r"^data/(?:raw|kg_v2)/")


def _build_path_aliases(workspace: Path) -> dict[str, str]:
    """Map extracted DOCX view paths back to canonical ``data/raw`` sources.

    The corpus tools expose read-only Markdown views of DOCX files as
    ``data/extracted_docx/raw/...`` so the model can search and read them.
    Release verification only accepts canonical ``data/raw`` / ``data/kg_v2``
    source paths.  Every view records its canonical ``SOURCE_PATH`` on the
    first line, so the runner can normalize the draft before verification.
    """

    aliases: dict[str, str] = {}
    extracted = workspace / "data/extracted_docx"
    if not extracted.is_dir():
        return aliases
    for path in extracted.rglob("*"):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        if not lines or not lines[0].startswith("SOURCE_PATH:"):
            continue
        canonical = lines[0][len("SOURCE_PATH:"):].strip()
        if _SOURCE_PATH_RE.match(canonical):
            aliases[path.relative_to(workspace).as_posix()] = canonical
    return aliases


def _normalize_draft_paths(
    draft: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    """Replace extracted view paths with canonical sources in the draft."""

    if not aliases:
        return dict(draft)

    def _map(value: str) -> str:
        return aliases.get(str(value or ""), str(value or ""))

    result = dict(draft)
    result["files_read"] = [
        _map(path) for path in draft.get("files_read") or []
    ]
    ledger: list[dict[str, Any]] = []
    for raw in draft.get("coverage_ledger") or []:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        entry["source_paths"] = [
            _map(path) for path in raw.get("source_paths") or []
        ]
        ledger.append(entry)
    result["coverage_ledger"] = ledger
    variants: list[dict[str, Any]] = []
    for raw in draft.get("procedure_variant_ledger") or []:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        entry["source_path"] = _map(raw.get("source_path"))
        variants.append(entry)
    result["procedure_variant_ledger"] = variants
    answer = str(draft.get("answer_markdown") or "")
    for logical, canonical in aliases.items():
        answer = answer.replace(
            f"【来源：{logical}】",
            f"【来源：{canonical}】",
        )
    result["answer_markdown"] = answer
    return result
