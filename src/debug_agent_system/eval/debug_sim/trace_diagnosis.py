from __future__ import annotations

from typing import Any

from .scenario_v2 import ScenarioV2

_LOW_SCORE_THRESHOLD = 0.5


def build_trace_digest(transcript: dict[str, Any]) -> dict[str, Any]:
    trace = _trace(transcript)
    summary = trace.get("summary") or {}
    candidate_paths = [row for row in trace.get("candidate_paths") or [] if isinstance(row, dict)]
    store_candidate_paths = [row for row in trace.get("store_candidate_paths") or [] if isinstance(row, dict)]
    filtered_entities = [row for row in summary.get("filtered_query_entities") or [] if isinstance(row, dict)]
    selected_check_trace = _selected_check_trace(transcript)
    return {
        "candidate_scores": [_candidate_digest(row) for row in candidate_paths[:3]],
        "store_candidate_scores": [_candidate_digest(row) for row in store_candidate_paths[:3]],
        "candidate_ids": [str(row.get("error_id") or "") for row in candidate_paths[:8]],
        "store_candidate_ids": [str(row.get("error_id") or "") for row in store_candidate_paths[:8]],
        "filtered_entity_skip_counts": _count_by_key(filtered_entities, "skip_reason"),
        "top_retrieval_entities": [_entity_digest(row) for row in (trace.get("retrieval_entity_roles") or [])[:6] if isinstance(row, dict)],
        "selected_check_trace": selected_check_trace,
        "presented_check_trace": [row for row in transcript.get("presented_check_trace") or [] if isinstance(row, dict)][:6],
        "branch_trace": [row for row in transcript.get("branch_trace") or [] if isinstance(row, dict)][:6],
        "branch_targets": [str(row.get("to_check_id") or "") for row in transcript.get("branch_options") or [] if isinstance(row, dict)][:8],
        "source_mismatch_first_check": bool(summary.get("source_mismatch_first_check")),
        "d_only_top_candidate": bool(summary.get("d_only_top_candidate")),
        "final_trace_aligned": bool(summary.get("final_trace_aligned")),
    }


def diagnose_failure(
    scenario: ScenarioV2,
    transcript: dict[str, Any],
    detail: dict[str, Any],
) -> dict[str, Any]:
    expected_errors = {scenario.target_error_id, *scenario.acceptable_error_ids} - {""}
    final_ids = [str(x) for x in (detail.get("trace_digest") or {}).get("candidate_ids") or [] if str(x)]
    store_ids = [str(x) for x in (detail.get("trace_digest") or {}).get("store_candidate_ids") or [] if str(x)]
    top_error_id = str(detail.get("top_error_id") or transcript.get("top_error_id") or "")
    first_check_id = str(detail.get("first_check_id") or transcript.get("first_check_id") or "")
    first_check_text = str(detail.get("first_check_text") or transcript.get("first_check_text") or "")
    current_check_id = str(transcript.get("current_check_id") or "")
    final_status = str(detail.get("final_status") or transcript.get("final_status") or "")
    target_acc = detail.get("target_error_acc")
    first_check_acc = detail.get("first_check_acc")
    replay_notes = [str(x) for x in detail.get("chat_replay_notes") or []]
    expected_status = str(scenario.expected_status or "")
    trace_digest = detail.get("trace_digest") or {}

    if detail.get("status") == "simulator_gap":
        return _diag(
            "replay",
            "simulator_gap",
            0.7,
            "simulator should have been able to continue",
            "simulator_gap",
            {"replay_events": transcript.get("replay_events") or []},
            "补 replay truth 或调整模拟器匹配规则",
        )

    if target_acc == 0.0 and expected_errors:
        if not any(error_id in final_ids for error_id in expected_errors) and not any(error_id in store_ids for error_id in expected_errors):
            return _diag(
                "retrieval",
                "target_absent_from_candidates",
                1.0,
                sorted(expected_errors),
                {"final_candidate_ids": final_ids, "store_candidate_ids": store_ids},
                {"candidate_scores": trace_digest.get("candidate_scores") or []},
                "检查 query entity、FTS seed 和 hyperedge 扩展是否覆盖目标 error",
            )
        if any(error_id in store_ids for error_id in expected_errors) and not any(error_id in final_ids[:1] for error_id in expected_errors):
            return _diag(
                "ranking",
                "target_lost_after_rerank",
                1.0 if not trace_digest.get("final_trace_aligned", True) else 0.8,
                sorted(expected_errors),
                {"final_candidate_ids": final_ids, "store_candidate_ids": store_ids},
                {
                    "candidate_scores": trace_digest.get("candidate_scores") or [],
                    "store_candidate_scores": trace_digest.get("store_candidate_scores") or [],
                },
                "检查 rerank boost、family canonicalization 和 fault prior 是否把正确目标压下去",
            )
        return _diag(
            "ranking",
            "target_not_top",
            0.9,
            sorted(expected_errors),
            {"final_candidate_ids": final_ids},
            {"candidate_scores": trace_digest.get("candidate_scores") or []},
            "比较 top candidates 的 score components，确认 fault prior 或 rerank 是否需要调整",
        )

    if expected_errors and top_error_id in expected_errors and not first_check_id and not first_check_text:
        return _diag(
            "lock",
            "top_correct_but_no_first_check",
            1.0,
            "non-empty locked subgraph and first/current check",
            {"first_check_id": first_check_id, "current_check_id": current_check_id},
            {"selected_check_trace": trace_digest.get("selected_check_trace") or {}},
            "检查 lock/load_locked_subgraph 是否返回空 checks，或 runtime 是否漏写 current check",
        )

    required_branch_targets = _required_branch_targets(scenario)
    actual_branch_targets = set(str(x) for x in trace_digest.get("branch_targets") or [] if str(x))
    if required_branch_targets and not required_branch_targets <= actual_branch_targets:
        return _diag(
            "branch",
            "branch_option_missing",
            0.9,
            sorted(required_branch_targets),
            sorted(actual_branch_targets),
            {"branch_trace": trace_digest.get("branch_trace") or []},
            "检查 branch_options 生成与条件分支建模是否漏了目标 branch",
        )
    if required_branch_targets and required_branch_targets & actual_branch_targets and first_check_acc == 0.0:
        return _diag(
            "branch",
            "branch_option_not_selected",
            0.9,
            sorted(required_branch_targets),
            {"first_check_id": first_check_id, "first_check_text": first_check_text},
            {
                "branch_trace": trace_digest.get("branch_trace") or [],
                "selected_check_trace": trace_digest.get("selected_check_trace") or {},
            },
            "检查 branch match score 和 non-interactive branch 选择是否偏到错误分支",
        )

    if expected_errors and top_error_id in expected_errors and first_check_acc == 0.0:
        cause = "top_correct_first_check_wrong"
        next_action = "检查 traversal relevance、branch 选择和 source mismatch penalty"
        if trace_digest.get("source_mismatch_first_check"):
            cause = "source_mismatch_first_check"
            next_action = "检查 supplemental candidate 的 check 是否抢到了 primary subgraph 的首检"
        return _diag(
            "traversal",
            cause,
            1.0 if cause == "source_mismatch_first_check" else 0.9,
            {"required_checks": [item.text or item.id for item in scenario.required_checks if item.required]},
            {"first_check_id": first_check_id, "first_check_text": first_check_text},
            {
                "selected_check_trace": trace_digest.get("selected_check_trace") or {},
                "presented_check_trace": trace_digest.get("presented_check_trace") or [],
            },
            next_action,
        )

    if any(note.startswith("replay_unmatched_step") for note in replay_notes):
        return _diag(
            "replay",
            "replay_unmatched_step",
            1.0,
            "a replay truth check should match the rendered step",
            {"first_check_text": first_check_text},
            {"replay_events": transcript.get("replay_events") or []},
            "检查 check text 匹配规则、rendered check wording 和 replay truth 对齐",
        )
    if any(note.startswith("replay_exhausted") for note in replay_notes):
        return _diag(
            "replay",
            "replay_exhausted",
            1.0,
            "replay should still have a matching user turn or truth row",
            {"replay_events": transcript.get("replay_events") or []},
            {"replay_events": transcript.get("replay_events") or []},
            "补 user_turn 或 replay_truth，或缩小 ask-info/check 文本匹配条件",
        )

    if detail.get("terminal_ok") == 0.0:
        return _diag(
            "terminal",
            "terminal_status_mismatch",
            0.8,
            expected_status,
            final_status,
            {
                "effective_result_covered": detail.get("effective_result_covered"),
                "chat_replay_notes": replay_notes,
            },
            "检查 pending_validation、resolved/escalate 判定和 replay effective result 处理",
        )

    if detail.get("evidence_recall") == 0.0 and (detail.get("target_error_acc") == 1.0 or detail.get("first_check_acc") == 1.0):
        return _diag(
            "render",
            "evidence_text_missing",
            0.7,
            list(scenario.evidence_key_facts or []),
            {"output_text_present": True},
            {
                "candidate_scores": trace_digest.get("candidate_scores") or [],
                "selected_check_trace": trace_digest.get("selected_check_trace") or {},
            },
            "检查 generator 文案是否遗漏关键证据词，或 scorer evidence fact 是否过严",
        )

    composite = float(detail.get("composite_gated") or detail.get("chat_replay_composite") or 1.0)
    if composite < _LOW_SCORE_THRESHOLD:
        return _diag(
            "unknown",
            "low_score_without_clear_stage",
            0.4,
            "high composite score",
            composite,
            {
                "candidate_scores": trace_digest.get("candidate_scores") or [],
                "selected_check_trace": trace_digest.get("selected_check_trace") or {},
            },
            "人工检查该 case 的 candidate paths、首检和 replay truth",
        )

    return _diag(
        "ok",
        "no_failure_detected",
        1.0,
        "",
        "",
        {
            "candidate_scores": trace_digest.get("candidate_scores") or [],
            "selected_check_trace": trace_digest.get("selected_check_trace") or {},
        },
        "无需额外定位",
    )


def _diag(
    stage: str,
    cause: str,
    confidence: float,
    expected: Any,
    observed: Any,
    evidence: Any,
    next_debug_action: str,
) -> dict[str, Any]:
    return {
        "primary_stage": stage,
        "primary_cause": cause,
        "confidence": round(float(confidence), 4),
        "expected": expected,
        "observed": observed,
        "evidence": evidence,
        "next_debug_action": next_debug_action,
    }


def _trace(transcript: dict[str, Any]) -> dict[str, Any]:
    turns = [row for row in transcript.get("turns") or [] if isinstance(row, dict)]
    for turn in reversed(turns):
        if str(turn.get("actor") or "") != "agent":
            continue
        response = turn.get("response") or {}
        trace = (response.get("metadata") or {}).get("retrieval_trace") or {}
        if isinstance(trace, dict) and trace:
            return trace
    return {}


def _selected_check_trace(transcript: dict[str, Any]) -> dict[str, Any]:
    selected = transcript.get("selected_check_trace")
    if isinstance(selected, dict):
        return selected
    current_id = str(transcript.get("current_check_id") or transcript.get("first_check_id") or "")
    for row in transcript.get("presented_check_trace") or []:
        if isinstance(row, dict) and str(row.get("check_id") or "") == current_id:
            return row
    return {}


def _candidate_digest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "error_id": str(row.get("error_id") or ""),
        "final_rank": row.get("final_rank"),
        "final_score": row.get("final_score", row.get("score")),
        "score_components": row.get("score_components") or {},
        "rerank_boost": row.get("rerank_boost"),
        "source_tiers": sorted({str(path.get("source_tier") or "") for path in row.get("paths") or [] if isinstance(path, dict)}),
        "path_event_ids": [str(path.get("event_id") or "") for path in row.get("paths") or [] if isinstance(path, dict)][:4],
    }


def _entity_digest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": str(row.get("entity") or ""),
        "role": str(row.get("role") or ""),
        "degree": int(row.get("degree") or 0),
        "adjusted_weight": row.get("adjusted_weight"),
    }


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _required_branch_targets(scenario: ScenarioV2) -> set[str]:
    value = (scenario.metadata or {}).get("required_branch_targets") or []
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}
