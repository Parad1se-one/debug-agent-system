"""Small DeepSeek Tool Calls client with explicit v4 limits and diagnostics."""

from __future__ import annotations

import json
import os
import time
from typing import Any
import urllib.error
import urllib.request


DEEPSEEK_BETA_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/beta/chat/completions"
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class DeepSeekToolCallError(RuntimeError):
    """Raised when a Tool Call cannot be safely consumed."""


def _decode_json_frames(text: str, *, label: str) -> tuple[Any, int]:
    """Decode keep-alive-adjacent JSON and dedupe only identical frames."""

    remaining = text.strip()
    values: list[Any] = []
    decoder = json.JSONDecoder()
    while remaining:
        try:
            value, end = decoder.raw_decode(remaining)
        except json.JSONDecodeError as exc:
            prefix_hex = remaining[:24].encode("utf-8", errors="replace").hex()
            raise DeepSeekToolCallError(
                f"{label}_json_decode:{exc.msg}:pos={exc.pos}:prefix_hex={prefix_hex}"
            ) from exc
        values.append(value)
        remaining = remaining[end:].strip()
    if not values:
        raise DeepSeekToolCallError(f"{label}_empty")
    canonical = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in values
    }
    if len(canonical) != 1:
        raise DeepSeekToolCallError(f"{label}_multiple_nonidentical_json_values:{len(values)}")
    return values[-1], len(values)


def configured_model() -> str:
    return (
        os.environ.get("DEEPSEEK_W2_TOOL_MODEL", "").strip()
        or os.environ.get("DEEPSEEK_W2_MODEL", "").strip()
        or "deepseek-v4-pro"
    )


def model_output_limit(model: str) -> int:
    return 384_000 if model.startswith("deepseek-v4-") else 8_192


def call_strict_tool(
    *,
    api_key: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    tool: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float | None = None,
    max_attempts: int = 3,
    user_id: str = "debug_agent_write_side",
) -> dict[str, Any]:
    """Call one strict function and return arguments plus transport metadata."""

    model = configured_model()
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    function_name = str(function.get("name") or "")
    if not function_name:
        raise ValueError("deepseek_tool_name_missing")
    output_budget = max(1_024, min(model_output_limit(model), int(max_tokens)))
    timeout = float(timeout_seconds or os.environ.get("DEEPSEEK_W2_TIMEOUT", "240"))
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": output_budget,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": function_name}},
        "user_id": user_id,
    }
    if model.startswith("deepseek-v4-"):
        payload["thinking"] = {"type": "disabled"}

    attempts = max(1, min(4, int(max_attempts)))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            DEEPSEEK_BETA_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=min(120.0, timeout)) as response:  # noqa: S310
                chunks: list[bytes] = []
                while True:
                    if time.monotonic() - started > timeout:
                        raise DeepSeekToolCallError(
                            f"deepseek_wall_clock_timeout>{timeout}s"
                        )
                    line = response.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    if stripped and not stripped.startswith(b":"):
                        chunks.append(line)
                response_text = b"".join(chunks).decode("utf-8")
                raw, response_json_frame_count = _decode_json_frames(
                    response_text, label="deepseek_response"
                )
            choice = (raw.get("choices") or [{}])[0]
            finish_reason = str(choice.get("finish_reason") or "")
            if finish_reason == "length":
                raise DeepSeekToolCallError(
                    f"deepseek_output_truncated:max_tokens={output_budget}"
                )
            if finish_reason in {"content_filter", "insufficient_system_resource"}:
                raise DeepSeekToolCallError(f"deepseek_finish_reason:{finish_reason}")
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = str(message.get("content") or "")
                raise DeepSeekToolCallError(
                    f"deepseek_missing_tool_call:finish_reason={finish_reason}:content={content[:240]}"
                )
            selected = next(
                (
                    item for item in tool_calls
                    if str(((item or {}).get("function") or {}).get("name") or "") == function_name
                ),
                None,
            )
            if not isinstance(selected, dict):
                raise DeepSeekToolCallError("deepseek_wrong_tool_call")
            arguments_text = str((selected.get("function") or {}).get("arguments") or "{}")
            arguments, argument_json_frame_count = _decode_json_frames(
                arguments_text, label="deepseek_tool_arguments"
            )
            if not isinstance(arguments, dict):
                raise DeepSeekToolCallError("deepseek_tool_arguments_not_object")
            return {
                "arguments": arguments,
                "model": str(raw.get("model") or model),
                "requested_model": model,
                "finish_reason": finish_reason,
                "usage": raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
                "system_fingerprint": str(raw.get("system_fingerprint") or ""),
                "attempt_count": attempt,
                "max_tokens": output_budget,
                "response_json_frame_count": response_json_frame_count,
                "argument_json_frame_count": argument_json_frame_count,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1_000]
            last_error = DeepSeekToolCallError(f"deepseek_http_{exc.code}:{body}")
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= attempts:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = DeepSeekToolCallError(f"deepseek_transport:{type(exc).__name__}:{exc}")
            if attempt >= attempts:
                raise last_error from exc
        except (json.JSONDecodeError, DeepSeekToolCallError) as exc:
            last_error = exc
            retryable = isinstance(exc, DeepSeekToolCallError) and str(exc).startswith((
                "deepseek_finish_reason:insufficient_system_resource",
                "deepseek_wall_clock_timeout",
                # The beta Tool Calls endpoint can occasionally return a
                # syntactically truncated ``function.arguments`` string even
                # when ``finish_reason`` is not ``length``.  Never repair or
                # guess that JSON locally; retry the whole deterministic call.
                "deepseek_tool_arguments_json_decode:",
            ))
            if attempt >= attempts or not retryable:
                raise
        time.sleep(min(4.0, float(2 ** (attempt - 1))))
    raise DeepSeekToolCallError(str(last_error or "deepseek_unknown_failure"))


def call_json_object(
    *,
    api_key: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float | None = None,
    max_attempts: int = 3,
    user_id: str = "debug_agent_write_side",
) -> dict[str, Any]:
    """Call DeepSeek JSON Output and fail closed on empty/truncated content."""

    model = configured_model()
    output_budget = max(1_024, min(model_output_limit(model), int(max_tokens)))
    timeout = float(timeout_seconds or os.environ.get("DEEPSEEK_W2_TIMEOUT", "240"))
    attempts = max(1, min(4, int(max_attempts)))
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": output_budget,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "user_id": user_id,
    }
    if model.startswith("deepseek-v4-"):
        payload["thinking"] = {"type": "disabled"}

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            DEEPSEEK_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=min(120.0, timeout)) as response:  # noqa: S310
                chunks: list[bytes] = []
                while True:
                    if time.monotonic() - started > timeout:
                        raise DeepSeekToolCallError(f"deepseek_wall_clock_timeout>{timeout}s")
                    line = response.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    if stripped and not stripped.startswith(b":"):
                        chunks.append(line)
                raw, response_json_frame_count = _decode_json_frames(
                    b"".join(chunks).decode("utf-8"), label="deepseek_response"
                )
            choice = (raw.get("choices") or [{}])[0]
            finish_reason = str(choice.get("finish_reason") or "")
            if finish_reason == "length":
                raise DeepSeekToolCallError(f"deepseek_output_truncated:max_tokens={output_budget}")
            if finish_reason in {"content_filter", "insufficient_system_resource"}:
                raise DeepSeekToolCallError(f"deepseek_finish_reason:{finish_reason}")
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = str(message.get("content") or "")
            arguments, content_json_frame_count = _decode_json_frames(
                content, label="deepseek_json_content"
            )
            if not isinstance(arguments, dict):
                raise DeepSeekToolCallError("deepseek_json_content_not_object")
            return {
                "arguments": arguments,
                "model": str(raw.get("model") or model),
                "requested_model": model,
                "finish_reason": finish_reason,
                "usage": raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
                "system_fingerprint": str(raw.get("system_fingerprint") or ""),
                "attempt_count": attempt,
                "max_tokens": output_budget,
                "response_json_frame_count": response_json_frame_count,
                "content_json_frame_count": content_json_frame_count,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1_000]
            last_error = DeepSeekToolCallError(f"deepseek_http_{exc.code}:{body}")
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= attempts:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = DeepSeekToolCallError(f"deepseek_transport:{type(exc).__name__}:{exc}")
            if attempt >= attempts:
                raise last_error from exc
        except DeepSeekToolCallError as exc:
            last_error = exc
            retryable = str(exc).startswith((
                "deepseek_finish_reason:insufficient_system_resource",
                "deepseek_wall_clock_timeout",
                "deepseek_json_content_empty",
                "deepseek_json_content_json_decode:",
            ))
            if attempt >= attempts or not retryable:
                raise
        time.sleep(min(4.0, float(2 ** (attempt - 1))))
    raise DeepSeekToolCallError(str(last_error or "deepseek_unknown_failure"))
