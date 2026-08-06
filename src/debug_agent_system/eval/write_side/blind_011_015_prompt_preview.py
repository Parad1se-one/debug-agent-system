"""Build auditable source-only Prompt-A requests for blind cases 011--015.

No file below ``ground_truth/`` is opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write import review_context as review_ctx
from debug_agent_system.agents.write.people_roles import load_people_role_registry, people_index
from debug_agent_system.agents.write.w2_extract.case_understanding_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    _alignment_style_example,
    _redact_prompt_text,
    tool_schema,
)
from debug_agent_system.knowledge_v2.contracts import APPROVED_FAMILY_LABELS


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sender_alias(value: Any, aliases: dict[str, str]) -> str:
    if isinstance(value, dict):
        raw = str(value.get("id") or value.get("name") or "").strip()
    else:
        raw = str(value or "").strip()
    if not raw:
        return ""
    if raw not in aliases:
        aliases[raw] = f"participant_{len(aliases) + 1}"
    return aliases[raw]


def _role_indexes() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_name = people_index(load_people_role_registry())
    by_open_id = {
        open_id: row
        for row in by_name.values()
        for open_id in row.get("open_ids") or []
    }
    return by_name, by_open_id


def _sender_organization_roles(
    sender: Any,
    *,
    by_name: dict[str, dict[str, Any]],
    by_open_id: dict[str, dict[str, Any]],
) -> list[str]:
    if not isinstance(sender, dict):
        return []
    sender_id = str(sender.get("id") or "").strip()
    sender_name = str(sender.get("name") or "").strip()
    row = by_open_id.get(sender_id) or by_open_id.get(sender_name) or by_name.get(sender_name) or {}
    return [str(value) for value in row.get("organization_roles") or [] if str(value)]


def _alignment_examples() -> list[dict[str, Any]]:
    examples = []
    for example in review_ctx.load_reviewed_examples(gold_root="data/annotations/goldcases/gold-v1"):
        if not isinstance(example, dict):
            continue
        examples.append(_alignment_style_example(example))
        if len(examples) >= 4:
            break
    return examples


def _jira_evidence_text(item: dict[str, Any]) -> str:
    comments = [
        {
            "author": str(comment.get("author") or ""),
            "created": str(comment.get("created") or ""),
            "body": str(comment.get("body") or "")[:1200],
        }
        for comment in item.get("comments") or []
        if isinstance(comment, dict)
    ][:20]
    payload = {
        "key": str(item.get("key") or ""),
        "summary": str(item.get("summary") or ""),
        "description": str(item.get("description") or "")[:4000],
        "status": str(item.get("status") or ""),
        "resolution": str(item.get("resolution") or ""),
        "created": str(item.get("created") or ""),
        "updated": str(item.get("updated") or ""),
        "comments": comments,
    }
    return _redact_prompt_text(json.dumps(payload, ensure_ascii=False))[:9000]


def _artifact_evidence_text(item: dict[str, Any]) -> str:
    excluded = {"path", "file_sha256", "source_message_ids"}
    payload = {
        key: value
        for key, value in item.items()
        if key not in excluded
    }
    return _redact_prompt_text(json.dumps(payload, ensure_ascii=False))[:5000]


def build_request(input_payload: dict[str, Any], *, alignment_examples: list[dict[str, Any]]) -> dict[str, Any]:
    aliases: dict[str, str] = {}
    roles_by_name, roles_by_open_id = _role_indexes()
    current_messages = []
    promoted_evidence = []
    allowed_ids = []
    for message in input_payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        message_id = str(message.get("message_id") or "")
        if not message_id:
            continue
        allowed_ids.append(message_id)
        current_messages.append({
            "message_id": message_id,
            "time": str(message.get("create_time") or ""),
            "sender": _sender_alias(message.get("sender"), aliases),
            "sender_organization_roles": _sender_organization_roles(
                message.get("sender"),
                by_name=roles_by_name,
                by_open_id=roles_by_open_id,
            ),
            "w1_role": "unsegmented_source",
            "text": _redact_prompt_text(message.get("text") or "")[:1600],
            "root_id": str(message.get("root_id") or ""),
            "parent_id": str(message.get("parent_id") or ""),
            "attachment_names": [
                _redact_prompt_text(item.get("name") or item.get("file_key") or "")[:240]
                for item in message.get("attachments") or []
                if isinstance(item, dict)
            ],
        })
    for item in input_payload.get("linked_jira_issues") or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id:
            continue
        allowed_ids.append(evidence_id)
        promoted_evidence.append({
            "message_id": evidence_id,
            "time": str(item.get("updated") or item.get("created") or ""),
            "sender": "jira_snapshot",
            "sender_organization_roles": [],
            "text": _jira_evidence_text(item),
            "promotion_reason": "source_bundle_linked_jira_snapshot",
        })
    for item in input_payload.get("external_artifacts") or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("artifact_ref") or "")
        if not evidence_id:
            continue
        allowed_ids.append(evidence_id)
        promoted_evidence.append({
            "message_id": evidence_id,
            "time": "",
            "sender": "attachment_inspector",
            "sender_organization_roles": [],
            "text": _artifact_evidence_text(item),
            "promotion_reason": "source_bundle_attachment_inspection_or_availability",
        })
    prompt_input = {
        "prompt_version": PROMPT_VERSION,
        "source_episode_id": str(input_payload.get("case_id") or ""),
        "source_thread_id": f"blind-source-session:{input_payload.get('case_id') or ''}",
        "input_mode": "unsegmented_source_session",
        "current_episode_messages": current_messages,
        "promoted_case_evidence": promoted_evidence,
        "allowed_evidence_ids": sorted(set(allowed_ids)),
        "family_ontology": sorted(APPROVED_FAMILY_LABELS),
        "w1_hints": {"symptom": "", "actions": [], "conclusion": ""},
        "alignment_examples": alignment_examples,
    }
    request_envelope = {
        "system_prompt": SYSTEM_PROMPT,
        "tool_schema": tool_schema(),
        "prompt_input": prompt_input,
    }
    return {
        "request_id": str(input_payload.get("case_id") or ""),
        "source_episode_id": str(input_payload.get("case_id") or ""),
        "input_messages_sha256": str(input_payload.get("messages_sha256") or ""),
        "payload_sha256": _canonical_hash(request_envelope),
        "request": request_envelope,
        "disclosure": disclosure_summary(prompt_input),
    }


def disclosure_summary(prompt_input: dict[str, Any]) -> dict[str, Any]:
    messages = [
        item
        for key in ("current_episode_messages", "promoted_case_evidence")
        for item in prompt_input.get(key) or []
        if isinstance(item, dict)
    ]
    text = " ".join(str(item.get("text") or "") for item in messages if isinstance(item, dict))
    # A 1600-character message cap can truncate ``@participant`` itself.
    without_aliases = re.sub(r"@?participan(?:t_?\d*)?", "", text, flags=re.IGNORECASE)
    return {
        "message_count": len(messages),
        "character_count": len(text),
        "contains_ip_address": bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)),
        "contains_software_version": bool(re.search(r"(?<!\d)\d+\.\d+(?:\.\d+){0,2}(?!\d)", text)),
        "contains_log_or_error_detail": any(marker in text.lower() for marker in ("日志", "报错", "错误", "http", "dmp", "jira")),
        "contains_unredacted_personal_marker": bool(
            re.search(r"\b1[3-9]\d{9}\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", without_aliases)
            or re.search(r"@[\u4e00-\u9fffA-Za-z][\w\-·（）()\u4e00-\u9fff]+", without_aliases)
        ),
        "sender_alias_count": len({str(item.get("sender") or "") for item in messages if isinstance(item, dict) and item.get("sender")}),
    }


def build_preview(input_root: str | Path) -> dict[str, Any]:
    root = Path(input_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    auxiliary_inputs = []
    for item in manifest.get("allowed_auxiliary_inputs") or []:
        path = Path(str(item.get("path") or ""))
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if not actual_hash or actual_hash != str(item.get("sha256") or ""):
            raise ValueError(f"blind_auxiliary_input_hash_mismatch:{path}")
        auxiliary_inputs.append(dict(item))
    examples = _alignment_examples()
    requests = []
    for row in manifest.get("cases") or []:
        path = root / str(row.get("file") or "")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != str(row.get("file_sha256") or ""):
            raise ValueError(f"blind_input_file_hash_mismatch:{path.name}")
        payload = json.loads(body.decode("utf-8"))
        if payload.get("messages_sha256") != row.get("messages_sha256"):
            raise ValueError(f"blind_input_message_hash_mismatch:{path.name}")
        requests.append(build_request(payload, alignment_examples=examples))
    return {
        "schema_version": "kg_v2.blind_prompt_preview.v1",
        "batch_id": str(manifest.get("batch_id") or ""),
        "prompt_version": PROMPT_VERSION,
        "source_only": True,
        "ground_truth_accessed": False,
        "network_io_performed": False,
        "allowed_auxiliary_inputs": auxiliary_inputs,
        "request_count": len(requests),
        "requests": requests,
    }


def response_template(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "kg_v2.blind_prompt_responses.v1",
        "batch_id": preview.get("batch_id"),
        "prompt_version": preview.get("prompt_version"),
        "responses": [
            {
                "request_id": item.get("request_id"),
                "source_episode_id": item.get("source_episode_id"),
                "input_messages_sha256": item.get("input_messages_sha256"),
                "payload_sha256": item.get("payload_sha256"),
                "tool_arguments": None,
            }
            for item in preview.get("requests") or []
        ],
    }


def preview_markdown(preview: dict[str, Any]) -> str:
    lines = [
        "# goldcase-011–015 DeepSeek Prompt 外发预览",
        "",
        f"- batch: `{preview.get('batch_id')}`",
        f"- prompt version: `{preview.get('prompt_version')}`",
        f"- source only: `{str(bool(preview.get('source_only'))).lower()}`",
        f"- ground truth accessed: `{str(bool(preview.get('ground_truth_accessed'))).lower()}`",
        f"- network I/O performed: `{str(bool(preview.get('network_io_performed'))).lower()}`",
        "",
        "| case | messages | chars | senders | IP | version | log/error | personal marker | payload hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in preview.get("requests") or []:
        disclosure = item.get("disclosure") or {}
        yn = lambda value: "yes" if value else "no"
        lines.append(
            f"| {item.get('request_id')} | {disclosure.get('message_count')} | {disclosure.get('character_count')} | "
            f"{disclosure.get('sender_alias_count')} | {yn(disclosure.get('contains_ip_address'))} | "
            f"{yn(disclosure.get('contains_software_version'))} | {yn(disclosure.get('contains_log_or_error_detail'))} | "
            f"{yn(disclosure.get('contains_unredacted_personal_marker'))} | `{str(item.get('payload_sha256') or '')[:12]}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blind-011-015-prompt-preview")
    parser.add_argument("--input-root", default="data/annotations/goldcases/review-v3/inputs")
    parser.add_argument("--out", default="data/results/gold-011-015-review-v3-prompt-preview.json")
    parser.add_argument("--md-out", default="data/results/gold-011-015-review-v3-prompt-preview.md")
    parser.add_argument("--response-template-out", default="data/results/gold-011-015-review-v3-prompt-responses.template.json")
    args = parser.parse_args(argv)
    preview = build_preview(args.input_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(preview_markdown(preview), encoding="utf-8")
    template_out = Path(args.response_template_out)
    template_out.parent.mkdir(parents=True, exist_ok=True)
    template_out.write_text(json.dumps(response_template(preview), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "md_out": str(md_out), "response_template_out": str(template_out), "request_count": preview["request_count"], "network_io_performed": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
