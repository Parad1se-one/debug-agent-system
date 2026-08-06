from __future__ import annotations

from pathlib import Path

import pytest

from debug_agent_system.adapters.codex_read.client import (
    CodexReadClient,
    CodexReadClientError,
    CodexResponsesClient,
    _chat_completions_url,
    _responses_url,
)


def test_codex_endpoint_normalizes_api_root_and_v1_root() -> None:
    assert (
        _chat_completions_url("https://gateway.example")
        == "https://gateway.example/v1/chat/completions"
    )
    assert (
        _responses_url("https://gateway.example")
        == "https://gateway.example/v1/responses"
    )
    assert (
        _responses_url("https://gateway.example/v1/")
        == "https://gateway.example/v1/responses"
    )
    assert (
        _chat_completions_url("https://gateway.example/v1/")
        == "https://gateway.example/v1/chat/completions"
    )
    assert (
        _chat_completions_url(
            "https://gateway.example/v1/chat/completions"
        )
        == "https://gateway.example/v1/chat/completions"
    )


def test_codex_client_reads_local_env_without_mutating_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "OPENAI_BASE_URL=https://gateway.example/v1\n"
        "OPENAI_API_KEY=test-only-secret\n",
        encoding="utf-8",
    )

    client = CodexReadClient(env_file=env_file)

    assert client.api_key == "test-only-secret"
    assert client.base_url == (
        "https://gateway.example/v1/chat/completions"
    )


def test_codex_client_reports_missing_key_before_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = CodexReadClient(
        base_url="https://gateway.example",
        env_file=tmp_path / "missing.env",
    )

    with pytest.raises(CodexReadClientError, match="missing_OPENAI_API_KEY"):
        client.complete_json(messages=[])


def test_responses_client_is_pinned_to_explicit_local_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example")
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "OPENAI_BASE_URL=https://gateway.example/v1\n"
        "OPENAI_API_KEY=local-only-secret\n",
        encoding="utf-8",
    )

    client = CodexResponsesClient(env_file=env_file)

    assert client.api_key == "local-only-secret"
    assert client.base_url == "https://gateway.example/v1/responses"
