"""Source-only W1 baseline and delayed ground-truth scoring for 011--015."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent
from debug_agent_system.agents.write.w1_message_relations import (
    annotate_semantic_fragments,
    assign_reference_aware_segments,
    infer_context_continuation_edges,
    infer_cross_window_trace_edges,
)


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def predict_source_only(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Run W1 without accepting or loading any label-bearing object."""

    collector = ChatCollectAgent()
    normalized = collector.normalize_messages(input_payload.get("messages") or [])
    fragmented, fragment_report = annotate_semantic_fragments(normalized)
    context_edges = [
        *infer_context_continuation_edges(fragmented),
        *infer_cross_window_trace_edges(fragmented),
    ]
    segmented, segment_report = assign_reference_aware_segments(
        fragmented,
        context_edges=context_edges,
    )
    summaries = collector.aggregate_threads(segmented)
    episodes = [
        episode
        for summary in summaries
        for episode in summary.get("episodes") or []
        if isinstance(episode, dict)
    ]
    projected_episodes = []
    for episode in episodes:
        extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
        projected_episodes.append({
            "episode_id": str(episode.get("episode_id") or ""),
            "thread_id": str(episode.get("thread_id") or ""),
            "completeness": str(episode.get("completeness") or ""),
            "fault_focus_text": str(extracted.get("fault_focus_text") or extracted.get("symptom_raw") or ""),
            "debug_actions": [str(value) for value in extracted.get("debug_actions") or []],
            "conclusion": str(extracted.get("conclusion") or extracted.get("key_conclusion") or ""),
            "evidence_message_ids": [str(value) for value in episode.get("evidence_message_ids") or []],
            "fault_message_ids": [str(item.get("message_id") or "") for item in episode.get("fault_description_messages") or [] if isinstance(item, dict)],
            "diagnostic_message_ids": [str(item.get("message_id") or "") for item in episode.get("diagnostic_chain_messages") or [] if isinstance(item, dict)],
            "resolution_message_ids": [str(item.get("message_id") or "") for item in episode.get("resolution_messages") or [] if isinstance(item, dict)],
            "noise_message_ids": [str(item.get("message_id") or "") for item in episode.get("noise_messages") or [] if isinstance(item, dict)],
        })
    active = [episode for episode in projected_episodes if episode.get("completeness") != "noise"]
    prediction = {
        "schema_version": "kg_v2.blind_w1_prediction.v1",
        "case_id": str(input_payload.get("case_id") or ""),
        "input_messages_sha256": str(input_payload.get("messages_sha256") or ""),
        "source_only": True,
        "ground_truth_accessed": False,
        "message_count": len(normalized),
        "context_edge_count": len(context_edges),
        "thread_count": len(summaries),
        "episode_count": len(projected_episodes),
        "active_episode_count": len(active),
        "noise_episode_count": len(projected_episodes) - len(active),
        "fragment_report": fragment_report,
        "segmentation_report": segment_report,
        "episodes": projected_episodes,
    }
    prediction["prediction_sha256"] = _canonical_hash(prediction)
    return prediction


def score_prediction(prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Load labels only after prediction and compare boundary-level invariants."""

    issues: list[dict[str, Any]] = []
    expected_count = int(truth.get("case_count") or 0)
    actual_count = int(prediction.get("active_episode_count") or 0)
    expected_split = bool(truth.get("split_required"))
    actual_split = actual_count > 1
    if expected_count != actual_count:
        issues.append({"code": "wrong_case_count", "expected": expected_count, "actual": actual_count})
    if expected_split != actual_split:
        issues.append({"code": "wrong_split_decision", "expected": expected_split, "actual": actual_split})
    expected_evidence = {
        str(evidence_id)
        for case in truth.get("cases") or []
        if isinstance(case, dict)
        for evidence_id in case.get("evidence_anchor_ids") or []
    }
    active_evidence = {
        str(evidence_id)
        for episode in prediction.get("episodes") or []
        if isinstance(episode, dict) and episode.get("completeness") != "noise"
        for evidence_id in episode.get("evidence_message_ids") or []
    }
    missing_evidence = sorted(expected_evidence - active_evidence)
    if missing_evidence:
        issues.append({"code": "gold_evidence_missing_from_active_episodes", "expected": sorted(expected_evidence), "actual": sorted(active_evidence)})
    return {
        "case_id": truth.get("case_id"),
        "input_hash_match": truth.get("input_messages_sha256") == prediction.get("input_messages_sha256"),
        "prediction_sha256": prediction.get("prediction_sha256"),
        "expected_case_count": expected_count,
        "predicted_active_episode_count": actual_count,
        "expected_split_required": expected_split,
        "predicted_split_required": actual_split,
        "critical_errors": issues,
    }


def baseline_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# goldcase-011–015 当前 W1 盲测基线",
        "",
        f"- prediction frozen before ground truth load: `{str(bool(report.get('prediction_frozen_before_ground_truth_load'))).lower()}`",
        f"- cases: `{summary.get('case_count')}`",
        f"- critical error cases: `{summary.get('critical_error_cases')}`",
        f"- exact case-count matches: `{summary.get('exact_case_count_matches')}`",
        "",
        "| case | expected traces | W1 active episodes | expected split | W1 split | critical errors | prediction hash |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    by_case = {item.get("case_id"): item for item in report.get("predictions") or []}
    for score in report.get("scores") or []:
        prediction = by_case.get(score.get("case_id")) or {}
        codes = ", ".join(item.get("code") or "" for item in score.get("critical_errors") or []) or "—"
        lines.append(
            f"| {score.get('case_id')} | {score.get('expected_case_count')} | "
            f"{score.get('predicted_active_episode_count')} | {str(score.get('expected_split_required')).lower()} | "
            f"{str(score.get('predicted_split_required')).lower()} | {codes} | `{str(prediction.get('prediction_sha256') or '')[:12]}` |"
        )
    for prediction in report.get("predictions") or []:
        lines.extend(["", f"## {prediction.get('case_id')}", ""])
        for episode in prediction.get("episodes") or []:
            lines.append(
                f"- `{episode.get('completeness')}` `{episode.get('episode_id')}`: "
                f"{episode.get('fault_focus_text') or '（无 fault focus）'}"
            )
    return "\n".join(lines) + "\n"


def run(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    input_paths = sorted((root / "inputs").glob("goldcase-*.json"))
    predictions = [
        predict_source_only(json.loads(path.read_text(encoding="utf-8")))
        for path in input_paths
    ]
    # Serialize and hash the whole source-only phase before opening any truth
    # file.  This makes the phase boundary explicit and auditable.
    prediction_batch_sha256 = _canonical_hash(predictions)
    truth_by_case = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "ground_truth").glob("goldcase-*.json"))
    }
    scores = [score_prediction(prediction, truth_by_case[prediction["case_id"]]) for prediction in predictions]
    return {
        "schema_version": "kg_v2.blind_w1_baseline.v1",
        "batch_id": "gold-011-015-review-v3",
        "prediction_frozen_before_ground_truth_load": True,
        "prediction_batch_sha256": prediction_batch_sha256,
        "predictions": predictions,
        "scores": scores,
        "summary": {
            "case_count": len(scores),
            "critical_error_cases": sum(bool(item.get("critical_errors")) for item in scores),
            "exact_case_count_matches": sum(item.get("expected_case_count") == item.get("predicted_active_episode_count") for item in scores),
            "critical_error_counts": {
                code: sum(
                    any(error.get("code") == code for error in item.get("critical_errors") or [])
                    for item in scores
                )
                for code in sorted({error.get("code") for item in scores for error in item.get("critical_errors") or []})
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blind-011-015-w1-baseline")
    parser.add_argument("--root", default="data/annotations/goldcases/review-v3")
    parser.add_argument("--out", default="data/results/gold-011-015-review-v3-w1-baseline.json")
    parser.add_argument("--md-out", default="data/results/gold-011-015-review-v3-w1-baseline.md")
    args = parser.parse_args(argv)
    report = run(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(baseline_markdown(report), encoding="utf-8")
    print(json.dumps({"out": str(out), "md_out": str(md_out), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
