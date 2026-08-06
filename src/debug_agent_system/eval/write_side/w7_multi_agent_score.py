"""Score W7 batch shadow predictions against human episode corrections."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


def _compact(value: Any) -> str:
    return re.sub(
        r"[\s，。；：、,.!?！？:;()（）\[\]【】]+",
        "",
        str(value or "").lower(),
    )


def _bigrams(value: Any) -> set[str]:
    text = _compact(value)
    return {
        text[index:index + 2]
        for index in range(max(0, len(text) - 1))
        if len(text[index:index + 2]) == 2
    }


def _similarity(left: Any, right: Any) -> float:
    left_values = _bigrams(left)
    right_values = _bigrams(right)
    union = left_values | right_values
    return (
        len(left_values & right_values) / len(union)
        if union else float(not left_values and not right_values)
    )


def _expected(
    episode: dict[str, Any],
    *,
    boolean_field: str,
    correction_field: str,
    snapshot_field: str,
) -> Any:
    corrected = episode.get(correction_field)
    if (
        episode.get(boolean_field) is False
        and corrected not in (None, "", [], {})
    ):
        return corrected
    snapshot = (
        episode.get("w7_snapshot")
        if isinstance(episode.get("w7_snapshot"), dict)
        else {}
    )
    return snapshot.get(snapshot_field)


def _prediction_index(
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    cards = [
        item for item in result.get("case_cards") or []
        if isinstance(item, dict)
    ]
    cards_by_ref = {
        str(item.get("case_ref") or ""): item
        for item in cards
        if str(item.get("case_ref") or "")
    }
    traces = (
        ((result.get("trace_compiler") or {}).get("bundle") or {}).get(
            "traces"
        )
        if isinstance(result.get("trace_compiler"), dict)
        else []
    ) or []
    trace_by_case: dict[str, dict[str, Any]] = {}
    phase_by_case: dict[str, dict[str, Any]] = {}
    phase_status_by_case: dict[str, str] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        phases = [
            item for item in trace.get("phases") or []
            if isinstance(item, dict)
        ]
        for phase_index, phase in enumerate(phases):
            case_ref = str(phase.get("case_ref") or "")
            if case_ref:
                trace_by_case[case_ref] = trace
                phase_by_case[case_ref] = phase
                event_type = str(phase.get("event_type") or "")
                later_events = {
                    str(item.get("event_type") or "")
                    for item in phases[phase_index + 1:]
                }
                if not event_type:
                    # Compatibility for older shadow fixtures that only
                    # carried order and trace terminal status.
                    phase_status = str(
                        trace.get("resolution_status") or "unknown"
                    )
                elif event_type == "recurrence":
                    phase_status = "recurrence"
                elif event_type == "short_term_recovery":
                    phase_status = (
                        "ineffective"
                        if "recurrence" in later_events
                        else "provisionally_resolved"
                    )
                elif event_type in {"resolution", "validation"}:
                    phase_status = (
                        "ineffective"
                        if "recurrence" in later_events
                        else "verified"
                    )
                elif event_type in {"report", "diagnosis", "action"}:
                    phase_status = "pending"
                else:
                    phase_status = "unknown"
                if (
                    phase_index == len(phases) - 1
                    and event_type in {
                        "resolution",
                        "validation",
                        "recurrence",
                        "short_term_recovery",
                    }
                ):
                    phase_status = str(
                        trace.get("resolution_status")
                        or phase_status
                    )
                phase_status_by_case[case_ref] = phase_status
    by_episode: dict[str, dict[str, Any]] = {}

    def episode_prediction(episode_id: str) -> dict[str, Any]:
        return by_episode.setdefault(episode_id, {
            "case_refs": [],
            "case_predictions": [],
            "fault_summaries": [],
            "trace_refs": [],
            "phase_indices": [],
            "phase_counts": [],
            "resolution_statuses": [],
            "resolution_evidence_message_ids": [],
            "w2_schema_valid": [],
            "case_boundary_schema_valid": False,
            "evidence_anchor_schema_valid": False,
            "atomic_adapter_schema_valid": False,
            "boundary_fragment_count": 0,
            "w2_candidate_count": 0,
        })

    for unit in result.get("units") or []:
        if not isinstance(unit, dict):
            continue
        episode_id = str(unit.get("episode_id") or "")
        if not episode_id:
            continue
        item = episode_prediction(episode_id)
        w7a = unit.get("w7a") if isinstance(unit.get("w7a"), dict) else {}
        boundary = (
            w7a.get("case_boundary")
            if isinstance(w7a.get("case_boundary"), dict)
            else {}
        )
        anchor = (
            w7a.get("evidence_anchor")
            if isinstance(w7a.get("evidence_anchor"), dict)
            else {}
        )
        atomic = (
            w7a.get("atomic_case_adapter")
            if isinstance(w7a.get("atomic_case_adapter"), dict)
            else {}
        )
        item["case_boundary_schema_valid"] = bool(
            boundary.get("schema_valid")
        )
        item["evidence_anchor_schema_valid"] = bool(
            anchor.get("schema_valid")
        )
        item["atomic_adapter_schema_valid"] = bool(
            atomic.get("schema_valid")
        )
        item["boundary_fragment_count"] = len(
            (boundary.get("decision") or {}).get("case_fragments") or []
        )
        item["w2_candidate_count"] = len(
            (unit.get("w2") or {}).get("candidates") or []
        )

    for case_ref, card in cards_by_ref.items():
        episode_id = str(card.get("parent_episode_id") or "")
        if not episode_id:
            continue
        item = episode_prediction(episode_id)
        item["case_refs"].append(case_ref)
        item["fault_summaries"].append(
            str(
                card.get("fault_summary")
                or card.get("title")
                or ""
            )
        )
        item["w2_schema_valid"].append(
            bool(card.get("production_schema_valid"))
        )
        trace = trace_by_case.get(case_ref)
        phase = phase_by_case.get(case_ref)
        case_prediction = {
            "case_ref": case_ref,
            "case_kind": str(card.get("case_kind") or ""),
            "fault_summary": str(
                card.get("fault_summary")
                or card.get("title")
                or ""
            ),
            "trace_ref": "",
            "phase_index": 0,
            "phase_count": 0,
            "resolution_status": "",
            "trace_resolution_status": "",
            "resolution_evidence_message_ids": [],
            "w2_schema_valid": bool(
                card.get("production_schema_valid")
            ),
        }
        if trace:
            trace_ref = str(trace.get("compiled_trace_id") or "")
            item["trace_refs"].append(trace_ref)
            item["phase_counts"].append(
                len(trace.get("phases") or [])
            )
            resolution_status = str(
                trace.get("resolution_status") or "unknown"
            )
            phase_resolution_status = str(
                phase_status_by_case.get(case_ref)
                or resolution_status
            )
            resolution_evidence = dedupe_strings_for_score(
                trace.get("resolution_evidence_message_ids") or []
            )
            item["resolution_statuses"].append(
                phase_resolution_status
            )
            item["resolution_evidence_message_ids"].extend(
                resolution_evidence
            )
            case_prediction.update({
                "trace_ref": trace_ref,
                "phase_count": len(trace.get("phases") or []),
                "resolution_status": phase_resolution_status,
                "trace_resolution_status": resolution_status,
                "resolution_evidence_message_ids": resolution_evidence,
            })
        if phase:
            phase_index = int(phase.get("phase_index") or 0)
            item["phase_indices"].append(phase_index)
            case_prediction["phase_index"] = phase_index
        item["case_predictions"].append(case_prediction)
    return by_episode


def _select_case_prediction(
    predicted: dict[str, Any],
    expected_focus: Any,
) -> tuple[dict[str, Any] | None, float]:
    """Align an episode-level human label to one atomic predicted case.

    W7a may correctly split one legacy episode into several cases.  Evaluation
    therefore selects the case whose fault summary best matches the human
    focus instead of treating multiple trace refs as an automatic miss.
    """

    candidates = [
        item for item in predicted.get("case_predictions") or []
        if isinstance(item, dict)
    ]
    if not candidates:
        return None, 0.0
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_similarity(
                expected_focus,
                item.get("fault_summary") or "",
            ),
            not bool(item.get("w2_schema_valid")),
            str(item.get("case_kind") or "") != "diagnostic_case",
            str(item.get("case_ref") or ""),
        ),
    )
    selected = ranked[0]
    return selected, _similarity(
        expected_focus,
        selected.get("fault_summary") or "",
    )


def _load_results(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, Any]] = {}
    for row in manifest.get("results") or []:
        if not isinstance(row, dict):
            continue
        result_path = Path(str(row.get("result") or ""))
        if not result_path.is_absolute():
            candidate = manifest_path.parent / result_path
            result_path = candidate if candidate.is_file() else result_path
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for unit in result.get("units") or []:
            if not isinstance(unit, dict):
                continue
            thread_id = str(unit.get("source_thread_id") or "")
            if thread_id:
                output[thread_id] = result
    return output


def _pairwise_trace_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    comparable = [
        row for row in rows
        if row.get("expected_trace_group")
    ]
    for index, left in enumerate(comparable):
        for right in comparable[index + 1:]:
            expected_same = (
                left["expected_trace_group"]
                == right["expected_trace_group"]
            )
            predicted_same = bool(
                left.get("predicted_trace_ref")
                and left.get("predicted_trace_ref")
                == right.get("predicted_trace_ref")
            )
            if expected_same and predicted_same:
                tp += 1
            elif not expected_same and predicted_same:
                fp += 1
            elif expected_same and not predicted_same:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _pairwise_trace_overlap_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score episode-level labels by overlap of predicted atomic traces."""

    tp = fp = fn = tn = 0
    comparable = [
        row for row in rows
        if row.get("expected_trace_group")
    ]
    for index, left in enumerate(comparable):
        left_refs = set(
            dedupe_strings_for_score(
                left.get("episode_predicted_trace_refs") or []
            )
        )
        for right in comparable[index + 1:]:
            expected_same = (
                left["expected_trace_group"]
                == right["expected_trace_group"]
            )
            right_refs = set(
                dedupe_strings_for_score(
                    right.get("episode_predicted_trace_refs") or []
                )
            )
            predicted_same = bool(left_refs & right_refs)
            if expected_same and predicted_same:
                tp += 1
            elif not expected_same and predicted_same:
                fp += 1
            elif expected_same:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "label_granularity": "legacy_episode",
        "prediction_projection": "any_atomic_trace_overlap",
    }


def score(
    *,
    manifest_path: Path,
    annotations_path: Path,
    session_limit: int = 5,
) -> dict[str, Any]:
    annotations = json.loads(
        annotations_path.read_text(encoding="utf-8")
    )
    results = _load_results(manifest_path)
    details: list[dict[str, Any]] = []
    reviewed_sessions = 0
    missing_sessions: list[str] = []
    for session in annotations.get("sessions") or []:
        if not isinstance(session, dict):
            continue
        if (
            not str(session.get("reviewer") or "").strip()
            or str(session.get("session_verdict") or "") == "exclude"
        ):
            continue
        if session_limit > 0 and reviewed_sessions >= session_limit:
            break
        reviewed_sessions += 1
        thread_id = str(session.get("thread_id") or "")
        prediction = results.get(thread_id)
        if prediction is None:
            missing_sessions.append(thread_id)
            continue
        by_episode = _prediction_index(prediction)
        for episode in session.get("episodes") or []:
            if not isinstance(episode, dict):
                continue
            episode_id = str(episode.get("episode_id") or "")
            predicted = by_episode.get(episode_id) or {}
            expected_focus = _expected(
                episode,
                boolean_field="fault_focus_correct",
                correction_field="corrected_fault_focus",
                snapshot_field="fault_focus",
            )
            expected_status = _expected(
                episode,
                boolean_field="resolution_status_correct",
                correction_field="corrected_resolution_status",
                snapshot_field="resolution_status",
            )
            expected_group = _expected(
                episode,
                boolean_field="trace_group_correct",
                correction_field="corrected_trace_group_id",
                snapshot_field="trace_group_id",
            )
            expected_phase = _expected(
                episode,
                boolean_field="trace_phase_correct",
                correction_field="corrected_trace_phase_index",
                snapshot_field="trace_phase_index",
            )
            expected_phase_count = _expected(
                episode,
                boolean_field="trace_phase_correct",
                correction_field="corrected_trace_phase_count",
                snapshot_field="trace_phase_count",
            )
            expected_w2 = _expected(
                episode,
                boolean_field="w2_readiness_correct",
                correction_field="corrected_w2_readiness",
                snapshot_field="w2_ready",
            )
            expected_resolution_evidence = dedupe_strings_for_score(
                _expected(
                    episode,
                    boolean_field="resolution_evidence_correct",
                    correction_field=(
                        "corrected_resolution_evidence_message_ids"
                    ),
                    snapshot_field="resolution_evidence_message_ids",
                )
            )
            selected_case, selection_similarity = (
                _select_case_prediction(predicted, expected_focus)
            )
            if selected_case is not None:
                predicted_focus = str(
                    selected_case.get("fault_summary") or ""
                )
                predicted_trace_refs = dedupe_strings_for_score(
                    selected_case.get("trace_ref") or ""
                )
                predicted_phases = [
                    int(selected_case.get("phase_index") or 0)
                ]
                predicted_phases = [
                    value for value in predicted_phases if value > 0
                ]
                predicted_phase_counts = [
                    int(selected_case.get("phase_count") or 0)
                ]
                predicted_phase_counts = [
                    value for value in predicted_phase_counts
                    if value > 0
                ]
                predicted_statuses = dedupe_strings_for_score(
                    selected_case.get("resolution_status") or ""
                )
                predicted_trace_statuses = dedupe_strings_for_score(
                    selected_case.get("trace_resolution_status") or ""
                )
                predicted_evidence = set(dedupe_strings_for_score(
                    selected_case.get(
                        "resolution_evidence_message_ids"
                    ) or []
                ))
                predicted_w2_ready = bool(
                    selected_case.get("w2_schema_valid")
                )
            else:
                predicted_focus = ""
                predicted_trace_refs = []
                predicted_phases = []
                predicted_phase_counts = []
                predicted_statuses = []
                predicted_trace_statuses = []
                predicted_evidence = set()
                predicted_w2_ready = False
            focus_similarity = selection_similarity
            evidence_recall = (
                len(
                    set(expected_resolution_evidence)
                    & predicted_evidence
                )
                / len(set(expected_resolution_evidence))
                if expected_resolution_evidence else 1.0
            )
            row = {
                "thread_id": thread_id,
                "episode_id": episode_id,
                "annotation_source_context_gap": (
                    "source_context"
                    in {
                        str(value)
                        for value in episode.get("issue_tags") or []
                    }
                ),
                "expected_fault_focus": expected_focus or "",
                "predicted_fault_focus": predicted_focus,
                "selected_case_ref": (
                    str(selected_case.get("case_ref") or "")
                    if selected_case else ""
                ),
                "case_selection_similarity": round(
                    selection_similarity, 4
                ),
                "predicted_trace_refs": predicted_trace_refs,
                "episode_predicted_trace_refs": (
                    dedupe_strings_for_score(
                        predicted.get("trace_refs") or []
                    )
                ),
                "fault_focus_similarity": round(
                    focus_similarity, 4
                ),
                "fault_focus_match": focus_similarity >= 0.5,
                "expected_trace_group": str(expected_group or ""),
                "predicted_trace_ref": (
                    predicted_trace_refs[0]
                    if len(predicted_trace_refs) == 1 else ""
                ),
                "expected_phase_index": int(expected_phase or 0),
                "predicted_phase_indices": predicted_phases,
                "phase_index_match": (
                    not expected_phase
                    or predicted_phases == [int(expected_phase)]
                ),
                "expected_phase_count": int(
                    expected_phase_count or 0
                ),
                "predicted_phase_counts": predicted_phase_counts,
                "phase_count_match": (
                    not expected_phase_count
                    or predicted_phase_counts
                    == [int(expected_phase_count)]
                ),
                "expected_resolution_status": str(
                    expected_status or ""
                ),
                "predicted_resolution_statuses": predicted_statuses,
                "predicted_trace_resolution_statuses": (
                    predicted_trace_statuses
                ),
                "resolution_status_match": (
                    not expected_status
                    or predicted_statuses == [str(expected_status)]
                ),
                "trace_terminal_projection_match": (
                    not expected_status
                    or predicted_trace_statuses
                    == [str(expected_status)]
                ),
                "expected_resolution_evidence_message_ids": (
                    expected_resolution_evidence
                ),
                "predicted_resolution_evidence_message_ids": sorted(
                    predicted_evidence
                ),
                "resolution_evidence_recall": round(
                    evidence_recall, 4
                ),
                "resolution_evidence_match": evidence_recall == 1.0,
                "expected_w2_ready": bool(expected_w2),
                "predicted_w2_ready": predicted_w2_ready,
                "w2_readiness_match": (
                    bool(expected_w2)
                    == predicted_w2_ready
                ),
                "predicted_case_count": len(
                    predicted.get("case_refs") or []
                ),
            }
            row["strict_episode_match"] = all((
                row["fault_focus_match"],
                row["phase_index_match"],
                row["phase_count_match"],
                row["resolution_status_match"],
                row["resolution_evidence_match"],
                row["w2_readiness_match"],
                bool(row["predicted_case_count"]),
            ))
            attribution: list[str] = []
            if row["annotation_source_context_gap"]:
                attribution.append("upstream_source_context_gap")
            if not predicted.get("case_boundary_schema_valid"):
                attribution.append("case_boundary")
            elif not predicted.get("evidence_anchor_schema_valid"):
                attribution.append("evidence_anchor")
            elif not predicted.get("atomic_adapter_schema_valid"):
                attribution.append("atomic_case_adapter")
            if (
                bool(expected_w2)
                and not int(predicted.get("w2_candidate_count") or 0)
            ):
                attribution.append("w2_atomic_extraction")
            if (
                not row["fault_focus_match"]
                and "case_boundary" not in attribution
            ):
                attribution.append("case_boundary_or_w2_semantics")
            if (
                expected_group
                and not row["predicted_trace_ref"]
                and row["predicted_case_count"]
            ):
                attribution.append("neighbor_link_or_trace_compiler")
            if not row["phase_index_match"] or not row["phase_count_match"]:
                attribution.append("trace_phase")
            if not row["resolution_status_match"]:
                attribution.append("outcome_reconciler")
            if not row["resolution_evidence_match"]:
                attribution.append("evidence_anchor_or_outcome_reconciler")
            if not row["w2_readiness_match"]:
                attribution.append("w2_atomic_extraction")
            row["error_attribution"] = list(dict.fromkeys(attribution))
            details.append(row)
    selected_trace_metrics = _pairwise_trace_metrics(details)
    trace_metrics = _pairwise_trace_overlap_metrics(details)
    input_observable_details = [
        row for row in details
        if not row.get("annotation_source_context_gap")
    ]
    input_observable_trace_metrics = _pairwise_trace_overlap_metrics(
        input_observable_details
    )
    input_observable_trace_metrics.update({
        "excluded_source_context_gap_episodes": (
            len(details) - len(input_observable_details)
        ),
        "observable_episodes": len(input_observable_details),
    })
    fields = (
        "fault_focus_match",
        "phase_index_match",
        "phase_count_match",
        "resolution_status_match",
        "resolution_evidence_match",
        "w2_readiness_match",
        "strict_episode_match",
    )
    metrics = {
        field: {
            "passed": sum(bool(row[field]) for row in details),
            "total": len(details),
            "rate": round(
                sum(bool(row[field]) for row in details)
                / max(1, len(details)),
                4,
            ),
        }
        for field in fields
    }
    metrics["trace_pairwise"] = trace_metrics
    metrics["trace_terminal_projection_match"] = {
        "passed": sum(
            bool(row["trace_terminal_projection_match"])
            for row in details
        ),
        "total": len(details),
        "rate": round(
            sum(
                bool(row["trace_terminal_projection_match"])
                for row in details
            ) / max(1, len(details)),
            4,
        ),
        "diagnostic_only": True,
    }
    metrics["trace_pairwise_input_observable"] = (
        input_observable_trace_metrics
    )
    metrics["trace_pairwise_selected_case"] = (
        selected_trace_metrics
    )
    attribution_counts: dict[str, int] = dict(sorted(
        (
            stage,
            sum(
                stage in row.get("error_attribution", [])
                for row in details
            ),
        )
        for stage in {
            value
            for row in details
            for value in row.get("error_attribution", [])
        }
    ))
    return {
        "schema_version": "w7.multi_agent_score.v4",
        "manifest": str(manifest_path),
        "annotations": str(annotations_path),
        "reviewed_sessions": reviewed_sessions,
        "scored_sessions": reviewed_sessions - len(missing_sessions),
        "missing_session_predictions": missing_sessions,
        "episodes": len(details),
        "metrics": metrics,
        "error_attribution_counts": attribution_counts,
        "details": details,
        "gate": {
            "status": (
                "PASS"
                if (
                    not missing_sessions
                    and metrics["strict_episode_match"]["rate"] == 1.0
                    and trace_metrics["f1"] >= 0.9
                )
                else "FAIL"
            ),
            "requirements": {
                "missing_session_predictions": 0,
                "strict_episode_match": 1.0,
                "trace_pairwise_f1": 0.9,
            },
        },
    }


def dedupe_strings_for_score(value: Any) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[,，;；\s]+", value)
    elif isinstance(value, list):
        values = value
    else:
        values = [] if value in (None, "") else [value]
    return list(dict.fromkeys(
        str(item) for item in values if str(item or "")
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="w7-multi-agent-score")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--session-limit", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = score(
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        session_limit=max(0, int(args.session_limit)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "episodes": report["episodes"],
        "metrics": report["metrics"],
        "gate": report["gate"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
