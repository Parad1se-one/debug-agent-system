"""Offline, reviewable export for the gold-v1 Prompt-A evaluation.

This module never performs network I/O.  It materialises exactly the system
prompt, strict tool schema, and per-case anonymised payload that an approved
external runner may use, plus a disclosure summary for human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import debug_agent_system.agents.write.review_context as review_ctx
from debug_agent_system.agents.write import KnowledgeExtractionAgent
from debug_agent_system.agents.write.w2_extract.case_understanding_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt_input,
    tool_schema,
)
from debug_agent_system.eval.write_side.gold_set import verify_gold_set
from debug_agent_system.eval.write_side.kg_v2_gold_compare import load_gold_cases
from debug_agent_system.knowledge.json_store import JsonKGStore


_IP = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_VERSION = re.compile(r"(?<!\d)(?:v)?\d{1,2}\.\d+(?:\.\d+){0,2}(?!\d)", re.IGNORECASE)
_LOG_HINT = re.compile(r"\b(?:dmp|dlog|dump|bugcheck|http\s*status|error|exception|0x[0-9a-f]+)\b", re.IGNORECASE)
_PERSONAL = re.compile(r"(?:\b1[3-9]\d{9}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|@[\w\-·（）()\u4e00-\u9fff]+)")


def build_preview(
    *,
    gold_root: str | Path = "data/annotations/goldcases/gold-v1",
    kg_root: str | Path = "data/kg",
) -> dict[str, Any]:
    integrity = verify_gold_set(gold_root)
    cases = load_gold_cases(gold_root)
    sop = review_ctx.load_sop_seed_background()
    reviewed_examples = review_ctx.load_reviewed_examples(gold_root=gold_root)
    extractor = KnowledgeExtractionAgent(JsonKGStore(kg_root), deepseek_enabled=False, w2_mode="prompt_first")
    requests: list[dict[str, Any]] = []
    for case in cases:
        episode = case.payload.get("episode_input") if isinstance(case.payload.get("episode_input"), dict) else {}
        source_episode_id = str(case.payload.get("source_episode_id") or "")
        loo_examples = [
            example for example in reviewed_examples
            if str(example.get("case_id") or "") != case.case_id
            and str(example.get("source_episode_id") or "") != source_episode_id
        ]
        background = review_ctx.build_sop_background_for_episode(episode, sop, loo_examples)
        prepared = review_ctx.inject_review_context(episode, background, review_case_id=case.case_id)
        semantics = extractor.extract_semantics(prepared, deepseek_enrich=False)
        prompt_input = build_prompt_input(semantics)
        payload_text = json.dumps(prompt_input, ensure_ascii=False, sort_keys=True)
        requests.append({
            "request_id": case.case_id,
            "source_episode_id": str(prompt_input.get("source_episode_id") or ""),
            "prompt_input": prompt_input,
            "payload_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
            "disclosure": disclosure_summary(prompt_input),
            "loo_audit": {
                "current_gold_case_excluded": all(
                    str(item.get("case_id") or "") != case.case_id
                    for item in background.get("reviewed_case_examples") or []
                    if isinstance(item, dict)
                ),
                "alignment_example_count": len(prompt_input.get("alignment_examples") or []),
            },
        })
    return {
        "schema_version": "kg_v2.gold_prompt_export.v1",
        "gold_set_id": integrity.get("gold_set_id") or "gold-v1",
        "gold_set_integrity_ok": bool(integrity.get("ok")),
        "external_destination": "https://api.deepseek.com/beta/chat/completions",
        "network_io_performed": False,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "tool_schema": tool_schema(),
        "request_count": len(requests),
        "redactions": ["sender_alias", "at_mention", "phone_number", "email_address", "exact_source_gold_example"],
        "requests": requests,
    }


def disclosure_summary(prompt_input: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(prompt_input, ensure_ascii=False)
    personal_scan_text = text.replace("@participant", "")
    messages = [
        item
        for key in ("current_episode_messages", "promoted_case_evidence")
        for item in prompt_input.get(key) or []
        if isinstance(item, dict)
    ]
    return {
        "message_count": len(messages),
        "text_character_count": sum(len(str(item.get("text") or "")) for item in messages),
        "contains_ip_address": bool(_IP.search(text)),
        "contains_software_version": bool(_VERSION.search(text)),
        "contains_log_or_error_detail": bool(_LOG_HINT.search(text)),
        "contains_unredacted_personal_marker": bool(_PERSONAL.search(personal_scan_text)),
        "contains_internal_fault_text": bool(messages),
    }


def preview_markdown(preview: dict[str, Any]) -> str:
    lines = [
        "# gold-v1 DeepSeek Prompt 外发预览",
        "",
        f"- gold set: `{preview.get('gold_set_id')}`",
        f"- integrity: `{'pass' if preview.get('gold_set_integrity_ok') else 'fail'}`",
        f"- prompt version: `{preview.get('prompt_version')}`",
        f"- destination: `{preview.get('external_destination')}`",
        f"- network I/O performed: `{str(bool(preview.get('network_io_performed'))).lower()}`",
        f"- requests: `{preview.get('request_count', 0)}`",
        "",
        "> 本文件只汇总披露范围；完整逐 Case payload 位于同名 JSON。",
        "",
        "| case | messages | chars | IP | version | log/error | personal marker | exact gold excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in preview.get("requests") or []:
        disclosure = item.get("disclosure") or {}
        loo = item.get("loo_audit") or {}
        lines.append(
            f"| {item.get('request_id')} | {disclosure.get('message_count', 0)} | "
            f"{disclosure.get('text_character_count', 0)} | "
            f"{'yes' if disclosure.get('contains_ip_address') else 'no'} | "
            f"{'yes' if disclosure.get('contains_software_version') else 'no'} | "
            f"{'yes' if disclosure.get('contains_log_or_error_detail') else 'no'} | "
            f"{'yes' if disclosure.get('contains_unredacted_personal_marker') else 'no'} | "
            f"{'yes' if loo.get('current_gold_case_excluded') else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def response_template(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "kg_v2.gold_prompt_responses.v1",
        "gold_set_id": str(preview.get("gold_set_id") or ""),
        "prompt_version": str(preview.get("prompt_version") or ""),
        "responses": [
            {
                "request_id": str(item.get("request_id") or ""),
                "source_episode_id": str(item.get("source_episode_id") or ""),
                "payload_sha256": str(item.get("payload_sha256") or ""),
                "tool_arguments": None,
            }
            for item in preview.get("requests") or []
            if isinstance(item, dict)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gold-prompt-preview")
    parser.add_argument("--gold-root", default="data/annotations/goldcases/gold-v1")
    parser.add_argument("--kg-root", default="data/kg")
    parser.add_argument("--out", default="data/results/gold-v1-prompt-preview.json")
    parser.add_argument("--md-out", default="data/results/gold-v1-prompt-preview.md")
    parser.add_argument("--response-template-out", default="data/results/gold-v1-prompt-responses.template.json")
    args = parser.parse_args(argv)
    preview = build_preview(gold_root=args.gold_root, kg_root=args.kg_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(preview_markdown(preview), encoding="utf-8")
    response_out = Path(args.response_template_out)
    response_out.parent.mkdir(parents=True, exist_ok=True)
    response_out.write_text(json.dumps(response_template(preview), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "md_out": str(md_out),
        "response_template_out": str(response_out),
        "request_count": preview.get("request_count"),
        "network_io_performed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
