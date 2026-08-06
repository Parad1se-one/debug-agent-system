"""Dependency-free OpenAI-compatible client for the read-side Codex path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request


class CodexReadClientError(RuntimeError):
    """A sanitized read-side API failure safe to expose in audit metadata."""


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
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _responses_url(base_url: str) -> str:
    """Accept an API root, a /v1 root, or a full Responses endpoint."""

    value = str(base_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/responses"):
        return value
    if value.endswith("/v1"):
        return f"{value}/responses"
    return f"{value}/v1/responses"


class CodexResponsesClient:
    """Minimal non-streaming Responses API transport.

    The agent loop and tool execution deliberately live in the caller.  This
    class only handles authentication, transport, safe errors and usage
    metadata.  Explicit constructor values take precedence so callers can
    pin credentials to a repository-local ``.env.local`` without depending
    on a CLI login or the parent process environment.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "",
        timeout_seconds: int = 600,
        env_file: str | Path | None = None,
    ) -> None:
        local = _read_local_env(env_file)
        self.api_key = api_key or local.get("OPENAI_API_KEY", "")
        configured_base = base_url or local.get("OPENAI_BASE_URL", "")
        self.base_url = _responses_url(configured_base)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.last_usage: dict[str, Any] = {}
        self.last_request_id = ""

    def create(self, body_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise CodexReadClientError("missing_OPENAI_API_KEY")
        if not self.base_url:
            raise CodexReadClientError("missing_OPENAI_BASE_URL")
        body = json.dumps(body_payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                self.last_request_id = str(
                    response.headers.get("x-request-id") or ""
                )
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = _safe_http_error_detail(exc)
            suffix = f":{detail}" if detail else ""
            raise CodexReadClientError(
                f"codex_responses_http_{exc.code}{suffix}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CodexReadClientError(
                f"codex_responses_transport:{type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise CodexReadClientError("codex_responses_not_object")
        if payload.get("status") not in {None, "completed", "in_progress"}:
            raise CodexReadClientError(
                f"codex_responses_status:{payload.get('status')}"
            )
        output = payload.get("output")
        if not isinstance(output, list):
            raise CodexReadClientError("codex_responses_missing_output")
        self.last_usage = _normalized_usage(payload.get("usage"))
        return payload


class CodexReadClient:
    """Return OpenAI-compatible assistant messages and strict JSON objects."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.3-codex",
        base_url: str = "",
        timeout_seconds: int = 60,
        env_file: str | Path | None = None,
    ) -> None:
        local = _read_local_env(env_file)
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or local.get("OPENAI_API_KEY", "")
        )
        configured_base = (
            base_url
            or os.environ.get("OPENAI_BASE_URL", "")
            or local.get("OPENAI_BASE_URL", "")
        )
        self.base_url = _chat_completions_url(configured_base)
        self.model = model
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.last_usage: dict[str, Any] = {}
        self.last_request_id = ""

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._request({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        })

    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        message = self._request({
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        })
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CodexReadClientError("codex_missing_json_content")
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```")
            raw = raw.removesuffix("```").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CodexReadClientError("codex_invalid_json_content") from exc
        if not isinstance(payload, dict):
            raise CodexReadClientError("codex_json_not_object")
        return payload

    def _request(self, body_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise CodexReadClientError("missing_OPENAI_API_KEY")
        if not self.base_url:
            raise CodexReadClientError("missing_OPENAI_BASE_URL")
        body = json.dumps(body_payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                self.last_request_id = str(
                    response.headers.get("x-request-id") or ""
                )
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            # Parse only the standard error envelope.  Never retain headers or
            # the full gateway body because either may contain credentials.
            detail = _safe_http_error_detail(exc)
            suffix = f":{detail}" if detail else ""
            raise CodexReadClientError(
                f"codex_http_{exc.code}{suffix}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CodexReadClientError(
                f"codex_transport:{type(exc).__name__}"
            ) from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise CodexReadClientError("codex_empty_choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise CodexReadClientError("codex_missing_message")
        usage = payload.get("usage")
        self.last_usage = _normalized_usage(usage)
        return message


def _safe_http_error_detail(exc: error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return ""
    raw = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        values = [
            str(raw.get("type") or ""),
            str(raw.get("code") or ""),
            str(raw.get("message") or ""),
        ]
    else:
        values = [str(raw or "")]
    # Header-like/token-like content is omitted instead of redacted so no
    # credential-shaped value reaches logs or response metadata.
    safe: list[str] = []
    for value in values:
        compact = " ".join(value.split())
        lower = compact.lower()
        if not compact or any(
            marker in lower
            for marker in ("authorization", "bearer ", "api_key", "api key")
        ):
            continue
        safe.append(compact[:240])
    return "|".join(safe)


def _normalized_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    result: dict[str, int] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            raw = value.get(candidate)
            if isinstance(raw, (int, float)):
                result[target] = int(raw)
                break
    if "total_tokens" not in result and {
        "prompt_tokens", "completion_tokens"
    } <= set(result):
        result["total_tokens"] = (
            result["prompt_tokens"] + result["completion_tokens"]
        )
    return result


__all__ = [
    "CodexReadClient",
    "CodexReadClientError",
    "CodexResponsesClient",
    "_chat_completions_url",
    "_responses_url",
]
