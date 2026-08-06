"""Human-review template and completion gate for W7 targeted regression packs."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SESSION_VERDICTS = {"pass", "needs_fix", "exclude"}
ISSUE_TAGS = {
    "fault_focus",
    "multi_fault_split",
    "resolution_status",
    "resolution_evidence",
    "short_term_recovery",
    "recurrence",
    "trace_group",
    "trace_phase",
    "w2_readiness",
    "source_context",
    "other",
}
BOOLEAN_FIELDS = (
    "fault_focus_correct",
    "multi_fault_split_correct",
    "resolution_status_correct",
    "resolution_evidence_correct",
    "trace_group_correct",
    "trace_phase_correct",
    "w2_readiness_correct",
)
STRUCTURED_CORRECTION_FIELDS = (
    "corrected_fault_focus",
    "corrected_episode_scope",
    "corrected_case_items",
    "corrected_resolution_status",
    "corrected_resolution_evidence_message_ids",
    "corrected_trace_group_id",
    "corrected_trace_phase_index",
    "corrected_trace_phase_count",
    "corrected_w2_readiness",
)
RELEASE_MIN_ACCURACY = {
    "fault_focus_correct": 0.90,
    "multi_fault_split_correct": 0.95,
    "resolution_status_correct": 0.95,
    "resolution_evidence_correct": 0.95,
    "trace_group_correct": 0.90,
    "trace_phase_correct": 0.90,
    "w2_readiness_correct": 0.95,
}
FALSE_FIELD_CORRECTIONS = {
    "fault_focus_correct": ("corrected_fault_focus",),
    "multi_fault_split_correct": ("corrected_episode_scope", "corrected_case_items"),
    "resolution_status_correct": ("corrected_resolution_status",),
    "resolution_evidence_correct": ("corrected_resolution_evidence_message_ids",),
    "trace_group_correct": ("corrected_trace_group_id",),
    "trace_phase_correct": ("corrected_trace_phase_index", "corrected_trace_phase_count"),
    "w2_readiness_correct": ("corrected_w2_readiness",),
}


def build_template(review_pack: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    old_rows = {
        str(row.get("thread_id") or ""): row
        for row in (existing or {}).get("sessions") or []
        if isinstance(row, dict)
    }
    sessions: list[dict[str, Any]] = []
    cases = [case for case in review_pack.get("cases") or [] if isinstance(case, dict)]
    cases.sort(key=lambda case: (
        -int(case.get("review_priority_score") or 0),
        str(case.get("thread_id") or ""),
    ))
    for case in cases:
        if not isinstance(case, dict):
            continue
        thread_id = str(case.get("thread_id") or "")
        previous = old_rows.get(thread_id) or {}
        old_episodes = {
            str(row.get("episode_id") or ""): row
            for row in previous.get("episodes") or []
            if isinstance(row, dict)
        }
        episodes = []
        for after in case.get("after") or []:
            if not isinstance(after, dict):
                continue
            episode_id = str(after.get("episode_id") or "")
            prior = old_episodes.get(episode_id) or {}
            episodes.append({
                "episode_id": episode_id,
                "w7_snapshot": {
                    key: after.get(key)
                    for key in (
                        "episode_scope",
                        "continuation",
                        "trace_group_id",
                        "trace_phase_index",
                        "trace_phase_count",
                        "trace_relation_type",
                        "trace_link_strength",
                        "trace_link_reasons",
                        "trace_link_candidates",
                        "resolution_status",
                        "w2_ready",
                        "w2_block_reasons",
                        "fault_focus",
                    )
                },
                **{field: prior.get(field) for field in BOOLEAN_FIELDS},
                "issue_tags": list(prior.get("issue_tags") or []),
                "corrected_fault_focus": str(prior.get("corrected_fault_focus") or ""),
                "corrected_episode_scope": str(prior.get("corrected_episode_scope") or ""),
                "corrected_case_items": str(prior.get("corrected_case_items") or ""),
                "corrected_resolution_status": str(prior.get("corrected_resolution_status") or ""),
                "corrected_resolution_evidence_message_ids": str(prior.get("corrected_resolution_evidence_message_ids") or ""),
                "corrected_trace_group_id": str(prior.get("corrected_trace_group_id") or ""),
                "corrected_trace_phase_index": prior.get("corrected_trace_phase_index"),
                "corrected_trace_phase_count": prior.get("corrected_trace_phase_count"),
                "corrected_w2_readiness": prior.get("corrected_w2_readiness"),
                "notes": str(prior.get("notes") or ""),
            })
        sessions.append({
            "thread_id": thread_id,
            "full_context_markdown": case.get("full_context_markdown") or "",
            "full_context_json": case.get("full_context_json") or "",
            "review_priority_score": int(case.get("review_priority_score") or 0),
            "review_priority_reasons": list(case.get("review_priority_reasons") or []),
            "weak_trace_link_candidate_count": int(case.get("weak_trace_link_candidate_count") or 0),
            "accepted_trace_link_count": int(case.get("accepted_trace_link_count") or 0),
            "reviewer": str(previous.get("reviewer") or ""),
            "reviewed_at": str(previous.get("reviewed_at") or ""),
            "session_verdict": str(previous.get("session_verdict") or ""),
            "session_issue_tags": list(previous.get("session_issue_tags") or []),
            "session_notes": str(previous.get("session_notes") or ""),
            "episodes": episodes,
        })
    return {
        "schema_version": "debug_agent_system.w7_human_review.v1",
        "review_pack_input": str((review_pack.get("summary") or {}).get("input") or ""),
        "required_min_sessions": int((existing or {}).get("required_min_sessions") or 50),
        "target_sessions": len(sessions),
        "allowed_session_verdicts": sorted(SESSION_VERDICTS),
        "allowed_issue_tags": sorted(ISSUE_TAGS),
        "boolean_fields": list(BOOLEAN_FIELDS),
        "structured_correction_fields": list(STRUCTURED_CORRECTION_FIELDS),
        "instructions": [
            "Open each full_context_markdown/json before deciding; review_pack summaries are insufficient.",
            "Fill reviewer, reviewed_at, session_verdict, and every boolean field for every episode.",
            "Use issue_tags and notes whenever a field is false or the session verdict is needs_fix/exclude.",
            "Do not use an automated agent identity as the reviewer for the human completion gate.",
            "Priority order puts fixed-173 trace calibration and weak-link cases first; priority is triage metadata, not a model verdict.",
            "When a boolean is false, fill its structured correction whenever possible; notes remain accepted for migrated v1 annotations.",
        ],
        "sessions": sessions,
    }


def validate_annotations(payload: dict[str, Any], *, min_sessions: int | None = None) -> dict[str, Any]:
    rows = [row for row in payload.get("sessions") or [] if isinstance(row, dict)]
    required = int(min_sessions if min_sessions is not None else payload.get("required_min_sessions") or 50)
    completed = 0
    issues: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    issue_tags: Counter[str] = Counter()
    boolean_failures: Counter[str] = Counter()
    boolean_totals: Counter[str] = Counter()
    boolean_successes: Counter[str] = Counter()
    completed_episodes = 0
    structured_correction_counts: Counter[str] = Counter()
    false_fields_with_structured_correction = 0
    false_field_count = 0
    structured_trace_rows: list[dict[str, Any]] = []
    for row in rows:
        thread_id = str(row.get("thread_id") or "")
        row_issues: list[str] = []
        reviewer = str(row.get("reviewer") or "").strip()
        reviewed_at = str(row.get("reviewed_at") or "").strip()
        verdict = str(row.get("session_verdict") or "").strip()
        if not reviewer:
            row_issues.append("missing_reviewer")
        if not reviewed_at:
            row_issues.append("missing_reviewed_at")
        if verdict not in SESSION_VERDICTS:
            row_issues.append("invalid_session_verdict")
        tags = [str(tag) for tag in row.get("session_issue_tags") or []]
        unknown_tags = sorted(set(tags) - ISSUE_TAGS)
        if unknown_tags:
            row_issues.append("unknown_session_issue_tags:" + ",".join(unknown_tags))
        episode_rows = [item for item in row.get("episodes") or [] if isinstance(item, dict)]
        if not episode_rows:
            row_issues.append("missing_episode_annotations")
        for episode in episode_rows:
            episode_id = str(episode.get("episode_id") or "")
            episode_tags = [str(tag) for tag in episode.get("issue_tags") or []]
            unknown_episode_tags = sorted(set(episode_tags) - ISSUE_TAGS)
            if unknown_episode_tags:
                row_issues.append(f"{episode_id}:unknown_issue_tags:" + ",".join(unknown_episode_tags))
            false_fields = []
            for field in BOOLEAN_FIELDS:
                value = episode.get(field)
                if not isinstance(value, bool):
                    row_issues.append(f"{episode_id}:{field}:not_boolean")
                elif value is False:
                    false_fields.append(field)
            if false_fields and not str(episode.get("notes") or "").strip():
                missing_structured = [
                    field for field in false_fields
                    if not all(
                        episode.get(name) not in (None, "", [], {})
                        for name in FALSE_FIELD_CORRECTIONS[field]
                    )
                ]
                if missing_structured:
                    row_issues.append(f"{episode_id}:false_field_requires_structured_correction_or_notes")
        if verdict in {"needs_fix", "exclude"} and not str(row.get("session_notes") or "").strip():
            row_issues.append("non_pass_verdict_requires_session_notes")
        if row_issues:
            issues.append({"thread_id": thread_id, "issues": row_issues})
            continue
        completed += 1
        completed_episodes += len(episode_rows)
        verdicts[verdict] += 1
        issue_tags.update(tags)
        for episode in episode_rows:
            issue_tags.update(str(tag) for tag in episode.get("issue_tags") or [])
            for field in BOOLEAN_FIELDS:
                boolean_totals[field] += 1
                if episode.get(field) is True:
                    boolean_successes[field] += 1
                else:
                    boolean_failures[field] += 1
                    correction_fields = FALSE_FIELD_CORRECTIONS[field]
                    false_field_count += 1
                    if all(
                        episode.get(name) not in (None, "", [], {})
                        for name in correction_fields
                    ):
                        false_fields_with_structured_correction += 1
                        structured_correction_counts.update(correction_fields)
            corrected_group = str(episode.get("corrected_trace_group_id") or "")
            if corrected_group:
                snapshot = episode.get("w7_snapshot") if isinstance(episode.get("w7_snapshot"), dict) else {}
                structured_trace_rows.append({
                    "episode_id": str(episode.get("episode_id") or ""),
                    "corrected_trace_group_id": corrected_group,
                    "predicted_trace_group_id": str(snapshot.get("trace_group_id") or ""),
                    "corrected_trace_phase_index": episode.get("corrected_trace_phase_index"),
                    "corrected_trace_phase_count": episode.get("corrected_trace_phase_count"),
                    "predicted_trace_phase_index": snapshot.get("trace_phase_index"),
                    "predicted_trace_phase_count": snapshot.get("trace_phase_count"),
                })
    completion_status = "PASS" if completed >= required else "INCOMPLETE"
    field_quality = {
        field: {
            "correct": boolean_successes[field],
            "incorrect": boolean_failures[field],
            "total": boolean_totals[field],
            "accuracy": round(boolean_successes[field] / boolean_totals[field], 4) if boolean_totals[field] else 0.0,
            "required_min_accuracy": RELEASE_MIN_ACCURACY[field],
        }
        for field in BOOLEAN_FIELDS
    }
    quality_failures = [
        field for field, row in field_quality.items()
        if row["total"] and row["accuracy"] < row["required_min_accuracy"]
    ]
    quality_status = (
        "INSUFFICIENT" if completion_status != "PASS"
        else ("FAIL" if quality_failures else "PASS")
    )
    release_status = (
        "INCOMPLETE" if completion_status != "PASS"
        else ("FAIL" if quality_status != "PASS" else "PASS")
    )
    gold_same_pairs = 0
    predicted_same_pairs = 0
    true_positive_pairs = 0
    cross_trace_contamination_pairs = 0
    for left, right in combinations(structured_trace_rows, 2):
        gold_same = left["corrected_trace_group_id"] == right["corrected_trace_group_id"]
        predicted_same = bool(
            left["predicted_trace_group_id"]
            and left["predicted_trace_group_id"] == right["predicted_trace_group_id"]
        )
        gold_same_pairs += int(gold_same)
        predicted_same_pairs += int(predicted_same)
        true_positive_pairs += int(gold_same and predicted_same)
        cross_trace_contamination_pairs += int(not gold_same and predicted_same)
    phase_rows = [
        row for row in structured_trace_rows
        if isinstance(row.get("corrected_trace_phase_index"), int)
        and isinstance(row.get("corrected_trace_phase_count"), int)
    ]
    phase_exact = sum(
        row["corrected_trace_phase_index"] == row["predicted_trace_phase_index"]
        and row["corrected_trace_phase_count"] == row["predicted_trace_phase_count"]
        for row in phase_rows
    )
    structured_trace_metrics = {
        "target_episode_count": len(structured_trace_rows),
        "gold_same_trace_pair_count": gold_same_pairs,
        "predicted_same_trace_pair_count": predicted_same_pairs,
        "true_positive_same_trace_pair_count": true_positive_pairs,
        "same_trace_pair_recall": round(true_positive_pairs / gold_same_pairs, 4) if gold_same_pairs else None,
        "same_trace_pair_precision": round(true_positive_pairs / predicted_same_pairs, 4) if predicted_same_pairs else None,
        "cross_trace_contamination_pair_count": cross_trace_contamination_pairs,
        "phase_target_count": len(phase_rows),
        "phase_exact_count": phase_exact,
        "phase_exact_accuracy": round(phase_exact / len(phase_rows), 4) if phase_rows else None,
    }
    return {
        "schema_version": "debug_agent_system.w7_human_review_report.v1",
        # ``status`` remains the historical completion-only contract.  Release
        # automation must use ``release_status`` below.
        "status": completion_status,
        "completion_status": completion_status,
        "quality_status": quality_status,
        "release_status": release_status,
        "required_min_sessions": required,
        "total_sessions": len(rows),
        "completed_sessions": completed,
        "completed_episodes": completed_episodes,
        "remaining_sessions": max(0, required - completed),
        "completion_rate": round(completed / len(rows), 4) if rows else 0.0,
        "verdict_counts": dict(sorted(verdicts.items())),
        "issue_tag_counts": dict(sorted(issue_tags.items())),
        "boolean_failure_counts": dict(sorted(boolean_failures.items())),
        "field_quality": field_quality,
        "quality_failures": quality_failures,
        "structured_correction_counts": dict(sorted(structured_correction_counts.items())),
        "false_field_count": false_field_count,
        "false_fields_with_structured_correction": false_fields_with_structured_correction,
        "structured_correction_coverage": round(
            false_fields_with_structured_correction / false_field_count, 4
        ) if false_field_count else 1.0,
        "structured_trace_metrics": structured_trace_metrics,
        "incomplete_or_invalid": issues,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def build_adjudicated_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Export completed human rows as immutable rule-learning/eval examples."""
    report = validate_annotations(payload, min_sessions=0)
    invalid_ids = {
        str(item.get("thread_id") or "")
        for item in report.get("incomplete_or_invalid") or []
        if isinstance(item, dict)
    }
    sessions = []
    for row in payload.get("sessions") or []:
        if not isinstance(row, dict) or str(row.get("thread_id") or "") in invalid_ids:
            continue
        episodes = []
        for episode in row.get("episodes") or []:
            if not isinstance(episode, dict):
                continue
            judgments = {field: episode.get(field) for field in BOOLEAN_FIELDS}
            corrections = {
                field: episode.get(field)
                for field in STRUCTURED_CORRECTION_FIELDS
                if episode.get(field) not in (None, "", [], {})
            }
            false_fields = [field for field, value in judgments.items() if value is False]
            structured_targets_complete = all(
                all(name in corrections for name in FALSE_FIELD_CORRECTIONS[field])
                for field in false_fields
            )
            episodes.append({
                "episode_id": str(episode.get("episode_id") or ""),
                "prediction": dict(episode.get("w7_snapshot") or {}),
                "judgments": judgments,
                "corrections": corrections,
                "structured_targets_complete": structured_targets_complete,
                "issue_tags": list(episode.get("issue_tags") or []),
                "notes": str(episode.get("notes") or ""),
            })
        sessions.append({
            "thread_id": str(row.get("thread_id") or ""),
            "session_verdict": str(row.get("session_verdict") or ""),
            "issue_tags": list(row.get("session_issue_tags") or []),
            "session_notes": str(row.get("session_notes") or ""),
            "reviewer": str(row.get("reviewer") or ""),
            "reviewed_at": str(row.get("reviewed_at") or ""),
            "episodes": episodes,
        })
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "debug_agent_system.w7_adjudicated_dataset.v1",
        "source_schema_version": str(payload.get("schema_version") or ""),
        "source_annotation_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "source_review_pack": str(payload.get("review_pack_input") or ""),
        "completed_sessions": len(sessions),
        "completed_episodes": sum(len(row["episodes"]) for row in sessions),
        "sessions": sessions,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def merge_structured_corrections(
    payload: dict[str, Any],
    correction_payload: dict[str, Any],
) -> dict[str, Any]:
    """Merge explicit human targets without touching judgments or verdicts."""
    out = json.loads(json.dumps(payload, ensure_ascii=False))
    sessions = {
        str(row.get("thread_id") or ""): row
        for row in out.get("sessions") or []
        if isinstance(row, dict)
    }
    applied = 0
    for correction_session in correction_payload.get("sessions") or []:
        if not isinstance(correction_session, dict):
            continue
        thread_id = str(correction_session.get("thread_id") or "")
        session = sessions.get(thread_id)
        if session is None:
            raise ValueError(f"unknown correction thread_id: {thread_id}")
        episodes = [item for item in session.get("episodes") or [] if isinstance(item, dict)]
        for correction in correction_session.get("episodes") or []:
            if not isinstance(correction, dict):
                continue
            explicit_id = str(correction.get("episode_id") or "")
            suffix = str(correction.get("episode_suffix") or "")
            matches = [
                episode for episode in episodes
                if (explicit_id and str(episode.get("episode_id") or "") == explicit_id)
                or (suffix and str(episode.get("episode_id") or "").endswith(suffix))
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"correction episode selector must match exactly once: {thread_id}:{explicit_id or suffix}"
                )
            target = matches[0]
            fields = {
                key: value
                for key, value in correction.items()
                if key in STRUCTURED_CORRECTION_FIELDS and value not in (None, "", [], {})
            }
            unknown = sorted(
                key for key in correction
                if key not in {*STRUCTURED_CORRECTION_FIELDS, "episode_id", "episode_suffix"}
            )
            if unknown:
                raise ValueError("unknown structured correction fields: " + ",".join(unknown))
            target.update(fields)
            applied += len(fields)
    source_hash = hashlib.sha256(
        json.dumps(correction_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sources = list(out.get("structured_correction_sources") or [])
    source_row = {
        "schema_version": str(correction_payload.get("schema_version") or ""),
        "sha256": source_hash,
        "applied_field_count": applied,
    }
    if source_row not in sources:
        sources.append(source_row)
    out["structured_correction_sources"] = sources
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init")
    init.add_argument("review_pack")
    init.add_argument("--out", required=True)
    init.add_argument(
        "--existing",
        default="",
        help="Optional prior annotation JSON whose completed or partial human fields must be preserved by thread/episode id.",
    )
    validate = sub.add_parser("validate")
    validate.add_argument("annotations")
    validate.add_argument("--min-sessions", type=int, default=0)
    validate.add_argument("--out", required=True)
    export = sub.add_parser("export")
    export.add_argument("annotations")
    export.add_argument("--out", required=True)
    apply_corrections = sub.add_parser("apply-corrections")
    apply_corrections.add_argument("annotations")
    apply_corrections.add_argument("corrections")
    apply_corrections.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "init":
        pack = json.loads(Path(args.review_pack).read_text(encoding="utf-8"))
        out = Path(args.out)
        existing_path = Path(args.existing) if args.existing else out
        existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else None
        payload = build_template(pack, existing)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "initialized", "sessions": len(payload["sessions"]), "out": str(out)}, ensure_ascii=False))
        return 0
    payload = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if args.cmd == "apply-corrections":
        corrections = json.loads(Path(args.corrections).read_text(encoding="utf-8"))
        merged = merge_structured_corrections(payload, corrections)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "corrections_applied",
            "sources": merged.get("structured_correction_sources") or [],
            "out": str(out),
        }, ensure_ascii=False))
        return 0
    if args.cmd == "export":
        dataset = build_adjudicated_dataset(payload)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "exported",
            "completed_sessions": dataset["completed_sessions"],
            "completed_episodes": dataset["completed_episodes"],
            "out": str(out),
        }, ensure_ascii=False))
        return 0
    report = validate_annotations(payload, min_sessions=args.min_sessions or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key]
        for key in (
            "status",
            "completion_status",
            "quality_status",
            "release_status",
            "required_min_sessions",
            "total_sessions",
            "completed_sessions",
            "remaining_sessions",
            "completion_rate",
            "verdict_counts",
            "issue_tag_counts",
            "boolean_failure_counts",
            "field_quality",
            "quality_failures",
        )
    } | {"report_out": str(out)}, ensure_ascii=False, indent=2))
    return 0 if report["release_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
