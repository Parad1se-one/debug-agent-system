"""Validate and replay externally produced Prompt-A tool arguments offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import debug_agent_system.agents.write.w2_extract as w2_module
from debug_agent_system.eval.write_side.gold_prompt_preview import build_preview
from debug_agent_system.eval.write_side.kg_v2_gold_compare import (
    baseline_markdown,
    run_legacy_bridge_baseline,
)


def validate_response_payload(
    preview: dict[str, Any],
    response_payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    issues: list[str] = []
    if str(response_payload.get("schema_version") or "") != "kg_v2.gold_prompt_responses.v1":
        issues.append("invalid_response_schema_version")
    if str(response_payload.get("gold_set_id") or "") != str(preview.get("gold_set_id") or ""):
        issues.append("gold_set_id_mismatch")
    if str(response_payload.get("prompt_version") or "") != str(preview.get("prompt_version") or ""):
        issues.append("prompt_version_mismatch")
    expected = {
        str(item.get("source_episode_id") or ""): item
        for item in preview.get("requests") or []
        if isinstance(item, dict) and str(item.get("source_episode_id") or "")
    }
    responses: dict[str, dict[str, Any]] = {}
    request_ids: set[str] = set()
    for index, item in enumerate(response_payload.get("responses") or []):
        if not isinstance(item, dict):
            issues.append(f"response_not_object:{index}")
            continue
        request_id = str(item.get("request_id") or "")
        source_episode_id = str(item.get("source_episode_id") or "")
        if request_id in request_ids:
            issues.append(f"duplicate_request_id:{request_id}")
        request_ids.add(request_id)
        expected_item = expected.get(source_episode_id)
        if expected_item is None:
            issues.append(f"unknown_source_episode_id:{source_episode_id}")
            continue
        if request_id != str(expected_item.get("request_id") or ""):
            issues.append(f"request_id_mismatch:{source_episode_id}")
        if str(item.get("payload_sha256") or "") != str(expected_item.get("payload_sha256") or ""):
            issues.append(f"payload_sha256_mismatch:{request_id}")
        tool_arguments = item.get("tool_arguments")
        if not isinstance(tool_arguments, dict):
            issues.append(f"missing_tool_arguments:{request_id}")
            continue
        responses[source_episode_id] = tool_arguments
    for source_episode_id, item in expected.items():
        if source_episode_id not in responses:
            issues.append(f"missing_response:{item.get('request_id')}")
    return responses, sorted(set(issues))


def replay_responses(
    *,
    response_path: str | Path,
    gold_root: str | Path = "data/annotations/goldcases/gold-v1",
    kg_root: str | Path = "data/kg",
) -> dict[str, Any]:
    preview = build_preview(gold_root=gold_root, kg_root=kg_root)
    response_payload = json.loads(Path(response_path).read_text(encoding="utf-8"))
    responses, issues = validate_response_payload(preview, response_payload)
    if issues:
        raise ValueError("invalid_gold_prompt_responses:" + ",".join(issues))

    original_call = w2_module._call_deepseek_case_understanding_with_hard_timeout
    old_key = os.environ.get("DEEPSEEK_API_KEY")

    def offline_call(prompt_input: dict[str, Any], *, api_key: str, repair_issues: list[str]) -> dict[str, Any]:
        source_episode_id = str(prompt_input.get("source_episode_id") or "")
        if source_episode_id not in responses:
            raise ValueError(f"offline_response_missing:{source_episode_id}")
        return responses[source_episode_id]

    try:
        os.environ["DEEPSEEK_API_KEY"] = "offline-replay-no-network"
        w2_module._call_deepseek_case_understanding_with_hard_timeout = offline_call
        report = run_legacy_bridge_baseline(
            gold_root=gold_root,
            kg_root=kg_root,
            deepseek=True,
            runner_mode="prompt_first",
            with_w7_loo=True,
        )
    finally:
        w2_module._call_deepseek_case_understanding_with_hard_timeout = original_call
        if old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_key
    report["response_import"] = {
        "source": str(response_path),
        "network_io_performed": False,
        "payload_hashes_verified": True,
        "response_count": len(responses),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gold-prompt-replay")
    parser.add_argument("responses")
    parser.add_argument("--gold-root", default="data/annotations/goldcases/gold-v1")
    parser.add_argument("--kg-root", default="data/kg")
    parser.add_argument("--out", default="data/results/gold-v1-w1-w7-prompt-replayed.json")
    parser.add_argument("--md-out", default="data/results/gold-v1-w1-w7-prompt-replayed.md")
    args = parser.parse_args(argv)
    report = replay_responses(response_path=args.responses, gold_root=args.gold_root, kg_root=args.kg_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(baseline_markdown(report), encoding="utf-8")
    print(json.dumps({"out": str(out), "md_out": str(md_out), "summary": report.get("summary")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
