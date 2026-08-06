"""Deterministic source-ledger adapters for W7a."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import canonical_hash, dedupe_strings


_ATTACHMENT_ID_KEYS = (
    "attachment_id",
    "file_key",
    "file_token",
    "image_key",
    "media_id",
    "payload_sha256",
    "sha256",
)


def _attachment_refs(
    message: dict[str, Any], message_id: str
) -> list[dict[str, Any]]:
    values: list[Any] = []
    for key in ("attachments", "attachment_metadata", "files", "media"):
        raw = message.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif isinstance(raw, dict):
            values.append(raw)
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values, 1):
        if isinstance(value, dict):
            attachment_id = next((
                str(value.get(key) or "")
                for key in _ATTACHMENT_ID_KEYS
                if str(value.get(key) or "")
            ), "")
            normalized = deepcopy(value)
        else:
            attachment_id = str(value or "")
            normalized = {"value": attachment_id}
        attachment_id = attachment_id or (
            f"{message_id}:attachment:{index}"
        )
        if attachment_id in seen:
            continue
        seen.add(attachment_id)
        normalized["attachment_id"] = attachment_id
        refs.append(normalized)
    return refs


def _episode_messages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        *(episode.get("messages") or []),
        *(episode.get("fault_description_messages") or []),
        *(episode.get("diagnostic_chain_messages") or []),
        *(episode.get("action_messages") or []),
        *(episode.get("resolution_messages") or []),
        *(episode.get("noise_messages") or []),
        *(episode.get("case_evidence_messages") or []),
        *(episode.get("case_context_messages") or []),
        *(episode.get("context_messages") or []),
    ]
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for index, value in enumerate(candidates):
        if not isinstance(value, dict):
            continue
        message_id = str(
            value.get("message_id")
            or value.get("id")
            or f"episode-message-{index + 1}"
        )
        if message_id in seen:
            continue
        seen.add(message_id)
        message = deepcopy(value)
        message["message_id"] = message_id
        message.setdefault(
            "text",
            value.get("content")
            or value.get("message")
            or value.get("summary")
            or "",
        )
        message["attachment_refs"] = _attachment_refs(message, message_id)
        output.append(message)
    return output


def build_episode_source_ledger(episode: dict[str, Any]) -> dict[str, Any]:
    rows = _episode_messages(episode)
    allowed_ids = dedupe_strings(
        row.get("message_id") for row in rows if isinstance(row, dict)
    )
    allowed_attachment_ids = dedupe_strings(
        attachment.get("attachment_id")
        for row in rows
        for attachment in row.get("attachment_refs") or []
        if isinstance(attachment, dict)
    )
    ledger = {
        "schema_version": "w7.source_ledger.v2",
        "source_thread_id": str(
            episode.get("thread_id")
            or episode.get("source_thread_id")
            or ""
        ),
        "episode_id": str(episode.get("episode_id") or ""),
        "rows": rows,
        "allowed_message_ids": allowed_ids,
        "allowed_attachment_ids": allowed_attachment_ids,
        "core_message_ids": allowed_ids,
        "stats": {
            "rows": len(rows),
            "attachments": sum(
                len(row.get("attachment_refs") or [])
                for row in rows
                if isinstance(row, dict)
            ),
        },
    }
    ledger["ledger_hash"] = canonical_hash(ledger)
    return ledger


def evidence_anchor_candidates(
    ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only rows that may carry attachment/media evidence."""

    output: list[dict[str, Any]] = []
    for row in ledger.get("rows") or []:
        if not isinstance(row, dict):
            continue
        attachments = [
            item for item in row.get("attachment_refs") or []
            if isinstance(item, dict)
        ]
        message_type = str(
            row.get("message_type")
            or row.get("msg_type")
            or row.get("type")
            or ""
        ).lower()
        text = str(row.get("text") or "").strip()
        media_without_text = (
            not text
            and message_type in {
                "file", "image", "video", "media", "audio", "post"
            }
        )
        if not attachments and not media_without_text:
            continue
        output.append({
            "message_id": str(row.get("message_id") or ""),
            "create_time": row.get("create_time") or "",
            "sender": row.get("sender") or {},
            "text": text,
            "message_type": message_type,
            "attachment_refs": attachments,
        })
    return output


def attach_case_source_context(
    case_cards: list[dict[str, Any]],
    ledger: dict[str, Any],
    *,
    max_text_chars: int = 4000,
) -> list[dict[str, Any]]:
    """Attach bounded verbatim source rows for semantic decision agents."""

    rows_by_id = {
        str(row.get("message_id") or ""): row
        for row in ledger.get("rows") or []
        if isinstance(row, dict)
        and str(row.get("message_id") or "")
    }
    output: list[dict[str, Any]] = []
    limit = max(1, int(max_text_chars))
    for card in case_cards:
        if not isinstance(card, dict):
            continue
        message_ids = dedupe_strings([
            *(card.get("source_message_ids") or []),
            *(card.get("evidence_message_ids") or []),
        ])
        source_rows: list[dict[str, Any]] = []
        for message_id in message_ids:
            row = rows_by_id.get(message_id)
            if row is None:
                continue
            sender = row.get("sender")
            sender_name = (
                str(sender.get("name") or sender.get("id") or "")
                if isinstance(sender, dict)
                else str(sender or "")
            )
            source_rows.append({
                "message_id": message_id,
                "create_time": str(
                    row.get("create_time")
                    or row.get("sent_at")
                    or row.get("timestamp")
                    or ""
                ),
                "sender": sender_name,
                "message_type": str(
                    row.get("message_type")
                    or row.get("msg_type")
                    or row.get("type")
                    or ""
                ),
                "text": str(
                    row.get("text")
                    or row.get("content_summary")
                    or ""
                )[:limit],
                "attachments": [{
                    "attachment_id": str(
                        value.get("attachment_id") or ""
                    ),
                    "name": str(
                        value.get("name")
                        or value.get("file_name")
                        or value.get("filename")
                        or ""
                    ),
                    "mime_type": str(
                        value.get("mime_type")
                        or value.get("type")
                        or ""
                    ),
                } for value in row.get("attachment_refs") or []
                if isinstance(value, dict)],
            })
        current = deepcopy(card)
        current["source_context_rows"] = source_rows
        output.append(current)
    return output
