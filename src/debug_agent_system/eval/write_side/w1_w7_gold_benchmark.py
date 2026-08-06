"""Boundary- and safety-oriented W1/W7 benchmark for Goldcase 001--020."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from debug_agent_system.agents.write.review_context import refine_episode_group
from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent
from debug_agent_system.agents.write.w1_message_relations import (
    annotate_semantic_fragments,
    assign_reference_aware_segments,
    infer_context_continuation_edges,
    infer_cross_window_trace_edges,
)
from debug_agent_system.eval.write_side.gold_001_020_adapter import (
    CANONICAL_OUTCOMES,
    adapter_summary,
    load_gold_001_020,
    normalize_action_role,
    normalize_outcome_type,
)


DEFAULT_ROOT = Path("data/annotations/goldcases")
SOURCE_SUITES = {
    "reference": set(range(11, 16)),
    "development": set(range(16, 21)),
    "source": set(range(11, 21)),
}


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _message_ids(episode: dict[str, Any]) -> list[str]:
    values = [str(value) for value in episode.get("evidence_message_ids") or [] if str(value)]
    for key in (
        "fault_description_messages", "diagnostic_chain_messages", "resolution_messages",
        "outcome_messages", "noise_messages", "case_context_messages",
    ):
        values.extend(
            str(item.get("source_message_id") or item.get("message_id") or "")
            for item in episode.get(key) or []
            if isinstance(item, dict) and (item.get("source_message_id") or item.get("message_id"))
        )
    return list(dict.fromkeys(values))


def _prediction_action(label: str, *, outcome_type: str = "pending_validation") -> dict[str, Any]:
    role, _ = normalize_action_role("", label=label)
    outcome, _ = normalize_outcome_type(outcome_type)
    return {"label": label, "action_role": role, "outcome_type": outcome}


def predict_case_source_only(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Run W1 then W7 using source records only; no truth object is accepted."""
    collector = ChatCollectAgent()
    normalized = collector.normalize_messages(input_payload.get("messages") or [])
    fragmented, fragment_report = annotate_semantic_fragments(normalized)
    inferred_edges = [
        *infer_context_continuation_edges(fragmented),
        *infer_cross_window_trace_edges(fragmented),
    ]
    segmented, segment_report = assign_reference_aware_segments(fragmented, context_edges=inferred_edges)
    summaries = collector.aggregate_threads(segmented)
    episodes = [
        episode
        for summary in summaries
        for episode in summary.get("episodes") or []
        if isinstance(episode, dict) and episode.get("completeness") != "noise"
    ]
    w1_groups: list[dict[str, Any]] = []
    for episode in episodes:
        extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
        actions = [_prediction_action(str(value)) for value in extracted.get("debug_actions") or [] if str(value)]
        w1_groups.append({
            "trace_group_id": str(episode.get("episode_id") or ""),
            "message_ids": _message_ids(episode),
            "actions": actions,
            "outcome_type": normalize_outcome_type(str(extracted.get("outcome_type") or "pending_validation"))[0],
        })

    # Candidate linking is intentionally case-window wide, so W7 can connect
    # phases that W1 placed in different relation-aware sessions.
    refined = refine_episode_group(episodes)
    w7_by_group: dict[str, dict[str, Any]] = {}
    for episode in refined:
        group_id = str(episode.get("trace_group_id") or episode.get("episode_id") or "")
        group = w7_by_group.setdefault(group_id, {
            "trace_group_id": group_id,
            "message_ids": [],
            "actions": [],
            "outcome_type": "pending_validation",
            "phase_ids": [],
        })
        group["message_ids"] = list(dict.fromkeys([*group["message_ids"], *_message_ids(episode)]))
        group["phase_ids"].append(str(episode.get("episode_id") or ""))
        extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
        cleanup = extracted.get("w7_episode_cleanup") if isinstance(extracted.get("w7_episode_cleanup"), dict) else {}
        outcome_type = normalize_outcome_type(cleanup.get("outcome_type") or "pending_validation")[0]
        if outcome_type == "verified_fix" or group["outcome_type"] == "pending_validation":
            group["outcome_type"] = outcome_type
        group["actions"].extend(
            _prediction_action(str(value), outcome_type=outcome_type)
            for value in extracted.get("debug_actions") or [] if str(value)
        )
    payload = {
        "schema_version": "debug_agent_system.w1_w7_source_prediction.v1",
        "case_id": str(input_payload.get("case_id") or ""),
        "source_only": True,
        "ground_truth_accessed": False,
        "input_messages_sha256": str(input_payload.get("messages_sha256") or ""),
        "message_count": len(normalized),
        "fragment_report": fragment_report,
        "segmentation_report": segment_report,
        "inferred_edge_count": len(inferred_edges),
        "w1_trace_groups": w1_groups,
        "w7_trace_groups": list(w7_by_group.values()),
    }
    payload["prediction_sha256"] = _canonical_hash(payload)
    return payload


def _gold_membership(case: dict[str, Any]) -> dict[str, set[str]]:
    membership: dict[str, set[str]] = defaultdict(set)
    for trace in case["traces"]:
        for message_id in trace["evidence"]["message_ids"]:
            membership[message_id].add(trace["trace_id"])
    return dict(membership)


def _pred_membership(groups: list[dict[str, Any]]) -> dict[str, set[str]]:
    membership: dict[str, set[str]] = defaultdict(set)
    for index, group in enumerate(groups):
        group_id = str(group.get("trace_group_id") or group.get("episode_id") or f"pred:{index}")
        for message_id in group.get("message_ids") or group.get("evidence_message_ids") or []:
            membership[str(message_id)].add(group_id)
    return dict(membership)


def _same_group(left: str, right: str, membership: dict[str, set[str]]) -> bool:
    return bool(membership.get(left, set()) & membership.get(right, set()))


def _pair_metrics(case: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    gold = _gold_membership(case)
    pred = _pred_membership(groups)
    positive_pairs: set[tuple[str, str]] = set()
    for trace in case["traces"]:
        ids = sorted(set(trace["evidence"]["message_ids"]))
        positive_pairs.update(combinations(ids, 2))
    positive_hits = sum(_same_group(left, right, pred) for left, right in positive_pairs)
    labeled_ids = sorted(gold)
    predicted_pairs = {
        (left, right)
        for left, right in combinations(labeled_ids, 2)
        if _same_group(left, right, pred)
    }
    true_predicted_pairs = {
        (left, right) for left, right in predicted_pairs if gold[left] & gold[right]
    }
    cannot_link_pairs = {
        (left, right) for left, right in combinations(labeled_ids, 2) if gold[left].isdisjoint(gold[right])
    }
    cannot_link_hits = sum((left, right) in predicted_pairs for left, right in cannot_link_pairs)
    coverages: list[float] = []
    for trace in case["traces"]:
        anchors = set(trace["evidence"]["message_ids"])
        if not anchors:
            continue
        best = max((len(anchors & set(group.get("message_ids") or group.get("evidence_message_ids") or [])) for group in groups), default=0)
        coverages.append(best / len(anchors))
    contaminated_groups = 0
    labeled_groups = 0
    for group in groups:
        ids = {str(value) for value in group.get("message_ids") or group.get("evidence_message_ids") or []} & set(gold)
        if len(ids) < 2:
            continue
        labeled_groups += 1
        compatible_trace = any(all(trace_id in gold[message_id] for message_id in ids) for trace_id in {value for message_id in ids for value in gold[message_id]})
        contaminated_groups += not compatible_trace
    return {
        "anchor_pair_hits": positive_hits,
        "anchor_pair_total": len(positive_pairs),
        "anchor_pair_recall": positive_hits / len(positive_pairs) if positive_pairs else None,
        "best_cluster_trace_coverage": sum(coverages) / len(coverages) if coverages else None,
        "trace_coverage_sum": sum(coverages),
        "trace_coverage_count": len(coverages),
        "same_trace_pair_hits": len(true_predicted_pairs),
        "predicted_labeled_pair_total": len(predicted_pairs),
        "same_trace_precision": len(true_predicted_pairs) / len(predicted_pairs) if predicted_pairs else None,
        "cannot_link_violations": cannot_link_hits,
        "cannot_link_total": len(cannot_link_pairs),
        "cannot_link_violation_rate": cannot_link_hits / len(cannot_link_pairs) if cannot_link_pairs else 0.0,
        "contaminated_group_count": contaminated_groups,
        "labeled_group_count": labeled_groups,
        "cross_trace_contamination": contaminated_groups / labeled_groups if labeled_groups else 0.0,
    }


def _group_gold_trace(case: dict[str, Any], group: dict[str, Any]) -> dict[str, Any] | None:
    message_ids = set(str(value) for value in group.get("message_ids") or group.get("evidence_message_ids") or [])
    ranked = sorted(
        ((len(message_ids & set(trace["evidence"]["message_ids"])), trace["trace_id"], trace) for trace in case["traces"]),
        key=lambda item: (-item[0], item[1]),
    )
    return ranked[0][2] if ranked and ranked[0][0] else None


def _false_verified(case: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, int | float]:
    predicted_verified = 0
    false_verified = 0
    for group in groups:
        outcome, _ = normalize_outcome_type(group.get("outcome_type") or group.get("resolution_status") or "")
        if outcome != "verified_fix":
            continue
        predicted_verified += 1
        trace = _group_gold_trace(case, group)
        gold_verified = bool(trace and any(action["outcome"]["outcome_type"] == "verified_fix" for action in trace["actions"]))
        false_verified += not gold_verified
    return {
        "predicted_verified_fix_count": predicted_verified,
        "false_verified_fix_count": false_verified,
        "false_verified_fix_rate": false_verified / predicted_verified if predicted_verified else 0.0,
    }


def _label_tokens(value: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(value or "").lower())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index:index + 2] for index in range(len(compact) - 1)}


def _label_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _label_tokens(left), _label_tokens(right)
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0


def _normalize_pred_action(action: Any, *, group_outcome: str) -> dict[str, str]:
    if isinstance(action, str):
        action = {"label": action}
    if not isinstance(action, dict):
        action = {}
    label = str(action.get("label") or action.get("action_label") or "")
    role, _ = normalize_action_role(action.get("action_role") or action.get("role") or "", label=label)
    embedded = action.get("outcome") if isinstance(action.get("outcome"), dict) else {}
    outcome, _ = normalize_outcome_type(action.get("outcome_type") or embedded.get("outcome_type") or group_outcome)
    return {"label": label, "action_role": role, "outcome_type": outcome}


def _classification_counts(case: dict[str, Any], groups: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    used_gold: set[tuple[str, str]] = set()
    for group in groups:
        trace = _group_gold_trace(case, group)
        pred_actions = [_normalize_pred_action(action, group_outcome=str(group.get("outcome_type") or "")) for action in group.get("actions") or []]
        if trace is None:
            for action in pred_actions:
                counts[action[field]]["fp"] += 1
            continue
        gold_actions = trace["actions"]
        for pred_action in pred_actions:
            candidates = sorted(
                (
                    (_label_similarity(pred_action["label"], gold_action["label"]), gold_action)
                    for gold_action in gold_actions
                    if (trace["trace_id"], gold_action["action_id"]) not in used_gold
                ),
                key=lambda item: -item[0],
            )
            if not candidates or candidates[0][0] < 0.35:
                counts[pred_action[field]]["fp"] += 1
                continue
            gold_action = candidates[0][1]
            used_gold.add((trace["trace_id"], gold_action["action_id"]))
            gold_value = gold_action[field] if field == "action_role" else gold_action["outcome"][field]
            if pred_action[field] == gold_value:
                counts[gold_value]["tp"] += 1
            else:
                counts[pred_action[field]]["fp"] += 1
                counts[gold_value]["fn"] += 1
    for trace in case["traces"]:
        for action in trace["actions"]:
            if (trace["trace_id"], action["action_id"]) in used_gold:
                continue
            value = action[field] if field == "action_role" else action["outcome"][field]
            counts[value]["fn"] += 1
    return dict(counts)


def _merge_counts(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]) -> None:
    for label, values in source.items():
        row = target.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
        for key in ("tp", "fp", "fn"):
            row[key] += values[key]


def _macro_f1(counts: dict[str, dict[str, int]]) -> float:
    scores: list[float] = []
    for values in counts.values():
        denominator = 2 * values["tp"] + values["fp"] + values["fn"]
        if denominator:
            scores.append(2 * values["tp"] / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def score_cases(cases: list[dict[str, Any]], predictions: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
    by_case = {str(item.get("case_id") or ""): item for item in predictions}
    case_rows: list[dict[str, Any]] = []
    totals = defaultdict(float)
    role_counts: dict[str, dict[str, int]] = {}
    outcome_counts: dict[str, dict[str, int]] = {}
    group_key = f"{stage}_trace_groups"
    for case in cases:
        prediction = by_case.get(case["case_id"])
        if prediction is None:
            raise ValueError(f"missing_prediction:{case['case_id']}")
        groups = list(prediction.get(group_key) or prediction.get("trace_groups") or [])
        pair = _pair_metrics(case, groups)
        safety = _false_verified(case, groups)
        _merge_counts(role_counts, _classification_counts(case, groups, "action_role"))
        _merge_counts(outcome_counts, _classification_counts(case, groups, "outcome_type"))
        row = {"case_id": case["case_id"], "gold_trace_count": case["trace_count"], "predicted_trace_group_count": len(groups), **pair, **safety}
        case_rows.append(row)
        for key in (
            "anchor_pair_hits", "anchor_pair_total", "trace_coverage_sum", "trace_coverage_count",
            "same_trace_pair_hits", "predicted_labeled_pair_total", "cannot_link_violations",
            "cannot_link_total", "contaminated_group_count", "labeled_group_count",
            "predicted_verified_fix_count", "false_verified_fix_count",
        ):
            totals[key] += float(row[key])
    summary = {
        "case_count": len(cases),
        "gold_trace_count": sum(case["trace_count"] for case in cases),
        "predicted_trace_group_count": sum(row["predicted_trace_group_count"] for row in case_rows),
        "anchor_pair_recall": totals["anchor_pair_hits"] / totals["anchor_pair_total"] if totals["anchor_pair_total"] else None,
        "best_cluster_trace_coverage": totals["trace_coverage_sum"] / totals["trace_coverage_count"] if totals["trace_coverage_count"] else None,
        "same_trace_precision": totals["same_trace_pair_hits"] / totals["predicted_labeled_pair_total"] if totals["predicted_labeled_pair_total"] else None,
        "cannot_link_violation_rate": totals["cannot_link_violations"] / totals["cannot_link_total"] if totals["cannot_link_total"] else 0.0,
        "cannot_link_violations": int(totals["cannot_link_violations"]),
        "cross_trace_contamination": totals["contaminated_group_count"] / totals["labeled_group_count"] if totals["labeled_group_count"] else 0.0,
        "false_verified_fix_rate": totals["false_verified_fix_count"] / totals["predicted_verified_fix_count"] if totals["predicted_verified_fix_count"] else 0.0,
        "false_verified_fix_count": int(totals["false_verified_fix_count"]),
        "action_role_macro_f1": _macro_f1(role_counts),
        "outcome_macro_f1": _macro_f1(outcome_counts),
    }
    summary["action_outcome_macro_f1"] = (summary["action_role_macro_f1"] + summary["outcome_macro_f1"]) / 2
    return {
        "stage": stage,
        "summary": summary,
        "cases": case_rows,
        "action_role_class_counts": role_counts,
        "outcome_class_counts": outcome_counts,
    }


def semantic_regression(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    semantic = [case for case in cases if case["suite"] == "semantic_regression"]
    issues: list[str] = []
    for case in semantic:
        if case["trace_count"] != 1:
            issues.append(f"{case['case_id']}:trace_count")
        for trace in case["traces"]:
            if not trace["actions"]:
                issues.append(f"{case['case_id']}:missing_actions")
            if not trace["evidence"]["message_ids"]:
                issues.append(f"{case['case_id']}:missing_message_evidence")
            for action in trace["actions"]:
                if action["outcome"]["outcome_type"] not in CANONICAL_OUTCOMES:
                    issues.append(f"{case['case_id']}:{action['action_id']}:outcome_taxonomy")
    report: dict[str, Any] = {
        "suite": "semantic_regression",
        "passed": not issues,
        "case_count": len(semantic),
        "trace_count": sum(case["trace_count"] for case in semantic),
        "action_count": sum(len(trace["actions"]) for case in semantic for trace in case["traces"]),
        "issues": issues,
    }
    if predictions is not None:
        semantic_score = score_cases(semantic, predictions, stage="semantic")
        report["prediction_score"] = semantic_score
        report["passed"] = report["passed"] and not semantic_score["summary"]["false_verified_fix_count"]
    return report


def run_benchmark(
    root: str | Path = DEFAULT_ROOT,
    *,
    suite: str = "development",
    predictions: list[dict[str, Any]] | None = None,
    semantic_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    if suite not in SOURCE_SUITES:
        raise ValueError(f"unsupported_suite:{suite}")
    numbers = SOURCE_SUITES[suite]
    # Freeze the complete source-only phase before loading any truth file.
    if predictions is None:
        inputs: list[dict[str, Any]] = []
        for number in sorted(numbers):
            batch = "review-v3" if number <= 15 else "gold-v2"
            path = root / batch / "inputs" / f"goldcase-{number:03d}.json"
            inputs.append(json.loads(path.read_text(encoding="utf-8")))
        predictions = [predict_case_source_only(payload) for payload in inputs]
    prediction_batch_sha256 = _canonical_hash(predictions)
    all_cases = load_gold_001_020(root)
    cases = [case for case in all_cases if int(case["case_id"].rsplit("-", 1)[-1]) in numbers]
    report = {
        "schema_version": "debug_agent_system.w1_w7_gold_benchmark.v1",
        "suite": suite,
        "prediction_frozen_before_ground_truth_load": True,
        "prediction_batch_sha256": prediction_batch_sha256,
        "adapter_summary": adapter_summary(all_cases),
        "semantic_regression": semantic_regression(all_cases, semantic_predictions),
        "w1": score_cases(cases, predictions, stage="w1"),
        "w7": score_cases(cases, predictions, stage="w7"),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="w1-w7-gold-benchmark")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--suite", choices=sorted(SOURCE_SUITES), default="development")
    parser.add_argument("--predictions", help="Optional JSON list or object containing a predictions list")
    parser.add_argument("--semantic-predictions", help="Optional 001--010 semantic trace-group predictions")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    predictions = None
    if args.predictions:
        payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        predictions = payload if isinstance(payload, list) else payload.get("predictions")
    semantic_predictions = None
    if args.semantic_predictions:
        payload = json.loads(Path(args.semantic_predictions).read_text(encoding="utf-8"))
        semantic_predictions = payload if isinstance(payload, list) else payload.get("predictions")
    report = run_benchmark(
        args.root,
        suite=args.suite,
        predictions=predictions,
        semantic_predictions=semantic_predictions,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
