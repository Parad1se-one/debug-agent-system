"""Small OpenAI-compatible DeepSeek client for read-side tool selection."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


class DeepSeekReadClientError(RuntimeError):
    pass


class DeepSeekReadClient:
    """Return one assistant message; orchestration remains in the Harness."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/beta/chat/completions",
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.api_key:
            raise DeepSeekReadClientError("missing_DEEPSEEK_API_KEY")
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0,
            },
            ensure_ascii=False,
        ).encode("utf-8")
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
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise DeepSeekReadClientError(
                f"deepseek_http_{exc.code}:{detail}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DeepSeekReadClientError(
                f"deepseek_transport:{type(exc).__name__}:{exc}"
            ) from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise DeepSeekReadClientError("deepseek_empty_choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise DeepSeekReadClientError("deepseek_missing_message")
        return message

    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return one strict JSON object for evidence-constrained composition."""

        message = self._request({
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        })
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekReadClientError("deepseek_missing_json_content")
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```")
            raw = raw.removesuffix("```").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DeepSeekReadClientError("deepseek_invalid_json_content") from exc
        if not isinstance(payload, dict):
            raise DeepSeekReadClientError("deepseek_json_not_object")
        return payload

    def _request(self, body_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise DeepSeekReadClientError("missing_DEEPSEEK_API_KEY")
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
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise DeepSeekReadClientError(
                f"deepseek_http_{exc.code}:{detail}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DeepSeekReadClientError(
                f"deepseek_transport:{type(exc).__name__}:{exc}"
            ) from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise DeepSeekReadClientError("deepseek_empty_choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise DeepSeekReadClientError("deepseek_missing_message")
        return message


__all__ = ["DeepSeekReadClient", "DeepSeekReadClientError"]
