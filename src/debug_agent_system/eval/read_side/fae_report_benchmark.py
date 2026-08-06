"""Build a source-first AOI benchmark from real Xing Lark FAE reports.

This is deliberately different from a KG inventory benchmark: every query is
an actual field-report message from a relation-aware Xing Lark session.  The
following FAE messages are kept as reference evidence, never silently folded
back into the source-time input.  The generated cases are review candidates,
not independent semantic Gold; a human still owns trace boundaries and final
root-cause/outcome adjudication.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from debug_agent_system.eval.write_side.build_xing_lark_candidate_library import (
    _signals,
    load_known_gold_message_ids,
)
from debug_agent_system.eval.write_side.freeze_xing_lark_heldout import (
    _load_source,
)


SCHEMA_VERSION = "debug_agent_system.aoi_fae_report_benchmark.v2"
BENCHMARK_ID = "aoi-fae-report-benchmark-v2"
DEFAULT_SOURCE = Path(
    "data/results/xing_relation_context_final_20260717/messages.jsonl"
)
DEFAULT_FAE_CSV = Path("data/annotations/fae_engineers_2026-07-21.csv")
DEFAULT_CANDIDATE_LIBRARY = Path(
    "data/annotations/goldcases/candidates/xing-lark-v1/candidates.json"
)
DEFAULT_LEGACY_QUERIES = Path(
    "data/eval/scenarios/read_side_shared_query_baseline_v1.json"
)
DEFAULT_OUT = Path("data/eval/benchmark/aoi_fae_report_benchmark_v2.json")
DEFAULT_REPORT_OUT = Path(
    "data/eval/benchmark/aoi_fae_report_benchmark_v2.report.json"
)
DEFAULT_MARKDOWN_OUT = Path(
    "data/results/benchmark_reports/aoi-fae-report-benchmark-v2/"
    "Query与答案.md"
)

MIN_CASE_COUNT = 200
DEFAULT_TARGET_COUNT = 205
MIN_CANDIDATE_SCORE = 45
MAX_REPORTS_PER_CANDIDATE = 4
FOLLOWUP_WINDOW_HOURS = 72
MAX_QUERY_SOURCE_CHARS = 1_600
MAX_EVIDENCE_CHARS = 700

_SPACE = re.compile(r"\s+")
_NORMALIZE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)
_OUTCOME_RESOLUTION = re.compile(
    r"恢复正常|恢复生产|已恢复|已经恢复|问题解决|已解决|验证正常|"
    r"正常使用|未再出现|未出现异常|没有复发|暂无复发|Resolved\s*--\s*Done",
    re.IGNORECASE,
)
_RECURRENCE = re.compile(r"复发|再次|又出现|仍然|还是|依旧|再也|重新.*(?:报错|异常|卡顿|失败)")
_NOT_SOURCE_REPORT = re.compile(
    r"Resolved\s*--\s*Done|故障报告状态更新|分析完成|根因[:：]",
    re.IGNORECASE,
)
_NO_ACTIVE_FAILURE = re.compile(
    r"未出现(?:异常|不进板|不拍摄|卡顿|报错|失败)|"
    r"未再出现(?:异常|不进板|不拍摄|卡顿|报错|失败)"
)
_REQUEST_ONLY = re.compile(r"^(?:@[^\s]+\s*)?(?:这个问题[，,]?)?(?:请|麻烦)(?:补充|提供|确认)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _compact(value: Any, limit: int) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _normalized(value: Any) -> str:
    return _NORMALIZE.sub("", _clean(value).lower())


def _trigrams(value: Any) -> set[str]:
    normalized = _normalized(value)
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + 3]
        for index in range(len(normalized) - 2)
    }


def _similarity(left: Any, right: Any) -> float:
    return _gram_similarity(_trigrams(left), _trigrams(right))


def _gram_similarity(left_grams: set[str], right_grams: set[str]) -> float:
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _load_fae_names(path: Path) -> set[str]:
    names: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if index and line.strip():
            names.add(line.split(",", 1)[0].strip())
    return names


def _load_legacy_queries(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        _clean(item.get("query"))
        for item in payload.get("records") or []
        if _clean(item.get("query"))
    ]


def _candidate_messages(
    candidate: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    # The candidate builder has already selected a bounded evidence window.
    # Do not reopen the full relation session here: relation merging can span
    # weeks and mix parallel field faults.
    rows = {
        str(item.get("message_id") or ""): source_by_id[
            str(item.get("message_id") or "")
        ]
        for item in candidate.get("evidence_preview") or []
        if str(item.get("message_id") or "") in source_by_id
    }
    return sorted(
        rows.values(),
        key=lambda item: (
            str(item.get("create_time") or ""),
            str(item.get("message_id") or ""),
        ),
    )


def _source_report_rows(
    messages: Iterable[dict[str, Any]],
    fae_names: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        text = _clean(message.get("text"))
        signals = _signals(message, fae_names)
        if (
            "issue" not in signals
            or len(text) < 30
            or _NOT_SOURCE_REPORT.search(text)
            or _NO_ACTIVE_FAILURE.search(text)
            or _REQUEST_ONLY.search(text)
            or "diagnosis" in signals
            or "resolution" in signals
        ):
            continue
        rows.append(message)
    return rows


def _fallback_source_report_rows(
    messages: Iterable[dict[str, Any]],
    fae_names: set[str],
) -> list[dict[str, Any]]:
    """Retain a real report with an initial assessment when no raw report exists."""
    rows: list[dict[str, Any]] = []
    for message in messages:
        text = _clean(message.get("text"))
        signals = _signals(message, fae_names)
        if (
            "issue" in signals
            and len(text) >= 30
            and "resolution" not in signals
            and not _NOT_SOURCE_REPORT.search(text)
            and not _NO_ACTIVE_FAILURE.search(text)
            and not _REQUEST_ONLY.search(text)
        ):
            rows.append(message)
    return rows


def _followup_rows(
    messages: Iterable[dict[str, Any]],
    source_message_id: str,
    fae_names: set[str],
) -> list[tuple[dict[str, Any], set[str]]]:
    rows: list[tuple[dict[str, Any], set[str]]] = []
    seen_source = False
    source_time: datetime | None = None
    for message in messages:
        message_id = str(message.get("message_id") or "")
        if message_id == source_message_id:
            seen_source = True
            source_time = datetime.strptime(
                str(message.get("create_time") or ""), "%Y-%m-%d %H:%M"
            )
            continue
        if not seen_source:
            continue
        message_time = datetime.strptime(
            str(message.get("create_time") or ""), "%Y-%m-%d %H:%M"
        )
        # Minute-level source timestamps do not establish an order among
        # messages posted in the same minute.  Treat those as concurrent
        # context, not post-report evidence: otherwise the answer can leak
        # information unavailable at the moment of the report.
        if source_time and message_time <= source_time:
            continue
        if source_time and message_time > source_time + timedelta(
            hours=FOLLOWUP_WINDOW_HOURS
        ):
            break
        text = _clean(message.get("text"))
        signals = _signals(message, fae_names)
        if not text or not signals & {
            "diagnosis",
            "action",
            "resolution",
            "jira",
            "diagnostic_artifact",
        }:
            continue
        rows.append((message, signals))
    return rows


def _select_followups(
    rows: list[tuple[dict[str, Any], set[str]]],
) -> list[tuple[dict[str, Any], set[str]]]:
    """Keep a small, diverse evidence chain rather than a chat transcript."""
    selected: list[tuple[dict[str, Any], set[str]]] = []
    covered: set[str] = set()
    priority = ("diagnosis", "action", "resolution", "jira", "diagnostic_artifact")
    for signal in priority:
        for row in rows:
            if signal in row[1] and str(row[0].get("message_id") or "") not in {
                str(item[0].get("message_id") or "") for item in selected
            }:
                selected.append(row)
                covered.update(row[1])
                break
    for row in rows:
        if len(selected) >= 6:
            break
        if str(row[0].get("message_id") or "") not in {
            str(item[0].get("message_id") or "") for item in selected
        }:
            selected.append(row)
    selected.sort(
        key=lambda item: (
            str(item[0].get("create_time") or ""),
            str(item[0].get("message_id") or ""),
        )
    )
    return selected


def _outcome_strength(
    followups: Iterable[tuple[dict[str, Any], set[str]]],
) -> str:
    text = "\n".join(_clean(message.get("text")) for message, _ in followups)
    if _OUTCOME_RESOLUTION.search(text) and _RECURRENCE.search(text):
        return "recurred_or_mixed; no verified_fix assertion"
    if _OUTCOME_RESOLUTION.search(text):
        return "reported_recovery; human verification still required"
    return "pending_validation"


def _query_text(candidate: dict[str, Any], message: dict[str, Any]) -> str:
    chat_name = _clean(candidate.get("chat_name"))
    attachments = [
        _clean(item.get("name"))
        for item in message.get("attachments") or []
        if _clean(item.get("name"))
    ]
    lines = [
        "【真实 FAE 现场报告】",
        f"时间：{message.get('create_time') or ''}",
        f"项目群：{chat_name}",
        "现场原文：",
        _compact(message.get("text"), MAX_QUERY_SOURCE_CHARS),
    ]
    if attachments:
        lines.append("随报附件：" + "、".join(attachments[:6]))
    lines.extend(
        [
            "任务：按 FAE 排故方式先识别独立故障 Trace，再区分已观察事实、"
            "候选解释与待补信息；只给出证据支持的低风险首步，不得把后续结论当作现场已知事实。",
        ]
    )
    return "\n".join(lines)


def _reference_answer(
    source_message: dict[str, Any],
    followups: list[tuple[dict[str, Any], set[str]]],
) -> tuple[str, list[str]]:
    outcome_strength = _outcome_strength(followups)
    claims = [
        "必须先按设备/程序对象/故障现象拆分 Trace，不能把同一现场报告中的并行问题合并。",
        f"结果强度：{outcome_strength}。",
    ]
    lines = [
        "FAE 参考处置卡（由真实后续消息提取；不等同于人工冻结 Gold）：",
        f"- 报告事实：{_compact(source_message.get('text'), 480)}",
        *[f"- {claim}" for claim in claims],
        "- 后续 FAE 证据：",
    ]
    for message, signals in followups:
        labels = "/".join(
            signal
            for signal in ("diagnosis", "action", "resolution", "jira")
            if signal in signals
        ) or "followup"
        lines.append(
            "  - [{time}] ({labels}) {sender}：{text}".format(
                time=str(message.get("create_time") or ""),
                labels=labels,
                sender=str((message.get("sender") or {}).get("name") or ""),
                text=_compact(message.get("text"), MAX_EVIDENCE_CHARS),
            )
        )
    lines.append(
        "- 边界：上述后续消息只作为评分参考证据；未经人工 Trace 审核，"
        "不得把建议、短暂恢复或群聊结论升级为 verified_fix。"
    )
    return "\n".join(lines), claims


def _case(
    index: int,
    candidate: dict[str, Any],
    source_message: dict[str, Any],
    followups: list[tuple[dict[str, Any], set[str]]],
    *,
    legacy_max_similarity: float,
) -> dict[str, Any]:
    query = _query_text(candidate, source_message)
    answer, claims = _reference_answer(source_message, followups)
    source_id = str(source_message.get("message_id") or "")
    followup_ids = [
        str(message.get("message_id") or "") for message, _ in followups
    ]
    followup_evidence = [
        {
            "message_id": str(message.get("message_id") or ""),
            "create_time": str(message.get("create_time") or ""),
            "signals": sorted(signals),
        }
        for message, signals in followups
    ]
    source_signals = sorted(_signals(source_message, set()))
    return {
        "case_id": f"fae-report-{index:03d}",
        "split": "held_out_candidate" if index > 200 else "candidate_validation",
        "source_type": "xing_lark_real_fae_report",
        "expectation_origin": "real_fae_evidence_candidate",
        "tracks": [
            "T1_grounded_answer",
            "T2_trace_boundary",
            "T3_action_outcome",
            "T4_safety_closure",
        ],
        "isolated_session": True,
        "query": query,
        "source_input": {
            "message_id": source_id,
            "create_time": str(source_message.get("create_time") or ""),
            "sender": str((source_message.get("sender") or {}).get("name") or ""),
            "text": _compact(source_message.get("text"), MAX_QUERY_SOURCE_CHARS),
            "attachments": source_message.get("attachments") or [],
        },
        "source_refs": {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "chat_id": str(candidate.get("chat_id") or ""),
            "chat_name": str(candidate.get("chat_name") or ""),
            "relation_aware_session_ids": list(
                candidate.get("relation_aware_session_ids") or []
            ),
            "candidate_score": int(candidate.get("score") or 0),
            "candidate_issue_tags": list(candidate.get("issue_tags") or []),
            "source_input_message_ids": [source_id],
            "reference_followup_message_ids": followup_ids,
            "reference_followup_evidence": followup_evidence,
        },
        "answer_gold": {
            "reference_answer": answer,
            "required_claims": claims,
            "reference_evidence_message_ids": followup_ids,
            "outcome_strength": _outcome_strength(followups),
            "forbidden_claims": [
                "已验证根因",
                "已完成长期 verified_fix",
            ],
        },
        "quality": {
            "query_is_real_fae_source_text": True,
            "candidate_score": int(candidate.get("score") or 0),
            "candidate_quality_tier": str(candidate.get("quality_tier") or ""),
            "source_report_signals": source_signals,
            "legacy_max_similarity": round(legacy_max_similarity, 6),
            "requires_human_trace_review": True,
            "independent_semantic_gold": False,
            "graph_ingestion_allowed": False,
        },
    }


def _coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(cases),
        "source_type_counts": dict(
            sorted(Counter(case["source_type"] for case in cases).items())
        ),
        "split_counts": dict(
            sorted(Counter(case["split"] for case in cases).items())
        ),
        "issue_tag_counts": dict(
            sorted(
                Counter(
                    tag
                    for case in cases
                    for tag in case["source_refs"]["candidate_issue_tags"]
                ).items()
            )
        ),
        "candidate_score_min": min(
            (case["quality"]["candidate_score"] for case in cases),
            default=0,
        ),
        "candidate_score_median": sorted(
            case["quality"]["candidate_score"] for case in cases
        )[len(cases) // 2]
        if cases
        else 0,
        "reported_recovery_case_count": sum(
            "reported_recovery" in case["answer_gold"]["outcome_strength"]
            for case in cases
        ),
        "recurred_or_mixed_case_count": sum(
            "recurred_or_mixed" in case["answer_gold"]["outcome_strength"]
            for case in cases
        ),
    }


def build_dataset(
    source: str | Path = DEFAULT_SOURCE,
    fae_csv: str | Path = DEFAULT_FAE_CSV,
    candidate_library: str | Path = DEFAULT_CANDIDATE_LIBRARY,
    legacy_queries: str | Path = DEFAULT_LEGACY_QUERIES,
    *,
    repo_root: str | Path = ".",
    target_count: int = DEFAULT_TARGET_COUNT,
) -> dict[str, Any]:
    """Build 200+ non-duplicate real-FAE-report benchmark candidates."""
    source = Path(source)
    fae_csv = Path(fae_csv)
    candidate_library = Path(candidate_library)
    legacy_queries = Path(legacy_queries)
    repo_root = Path(repo_root)
    fae_names = _load_fae_names(fae_csv)
    source_by_id, _ = _load_source(source)
    library = json.loads(candidate_library.read_text(encoding="utf-8"))
    known_gold_ids = load_known_gold_message_ids(repo_root)
    historical_queries = _load_legacy_queries(legacy_queries)
    historical_grams = [_trigrams(query) for query in historical_queries]

    cases: list[dict[str, Any]] = []
    accepted_source_grams: list[set[str]] = []
    rejected: Counter[str] = Counter()
    for candidate in library.get("candidates") or []:
        if len(cases) >= target_count:
            break
        if int(candidate.get("score") or 0) < MIN_CANDIDATE_SCORE:
            rejected["candidate_score_below_threshold"] += 1
            continue
        if candidate.get("status") != "unreviewed":
            rejected["candidate_not_unreviewed"] += 1
            continue
        messages = _candidate_messages(candidate, source_by_id)
        if not messages:
            rejected["missing_session_messages"] += 1
            continue
        message_ids = {
            str(message.get("message_id") or "") for message in messages
        }
        if message_ids & known_gold_ids:
            rejected["known_gold_overlap"] += 1
            continue
        reports = _source_report_rows(messages, fae_names)
        if not reports:
            reports = _fallback_source_report_rows(messages, fae_names)
            if reports:
                rejected["used_report_with_initial_assessment"] += 1
        if not reports:
            rejected["no_natural_source_report"] += 1
            continue
        accepted_from_candidate = 0
        for source_message in reports:
            if len(cases) >= target_count:
                break
            if accepted_from_candidate >= MAX_REPORTS_PER_CANDIDATE:
                break
            source_text = _clean(source_message.get("text"))
            if len(source_text) > MAX_QUERY_SOURCE_CHARS * 2:
                rejected["source_report_too_long"] += 1
                continue
            followups = _select_followups(
                _followup_rows(
                    messages,
                    str(source_message.get("message_id") or ""),
                    fae_names,
                )
            )
            if not followups:
                rejected["no_followup_evidence"] += 1
                continue
            source_grams = _trigrams(source_text)
            legacy_max = max(
                (
                    _gram_similarity(source_grams, grams)
                    for grams in historical_grams
                ),
                default=0.0,
            )
            generated_max = max(
                (
                    _gram_similarity(source_grams, grams)
                    for grams in accepted_source_grams
                ),
                default=0.0,
            )
            if max(legacy_max, generated_max) >= 0.82:
                rejected["duplicate_or_near_duplicate_query"] += 1
                continue
            cases.append(
                _case(
                    len(cases) + 1,
                    candidate,
                    source_message,
                    followups,
                    legacy_max_similarity=legacy_max,
                )
            )
            accepted_source_grams.append(source_grams)
            accepted_from_candidate += 1
    if len(cases) < target_count:
        raise ValueError(
            f"insufficient_real_fae_cases:{len(cases)}<{target_count};"
            f"rejected={dict(sorted(rejected.items()))}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "build_policy": {
            "query_source": "real_xing_lark_fae_report_message",
            "candidate_library_method": "relation_aware_session + FAE evidence signals",
            "query_is_not_kg_object_template": True,
            "existing_shared_queries_are_excluded_by_similarity": True,
            "ground_truth_runtime_visible": False,
            "graph_ingestion_allowed": False,
            "human_trace_review_required": True,
            "semantic_gold_claim_allowed": False,
        },
        "source_manifest": {
            "xing_relation_source": str(source),
            "xing_relation_source_sha256": _sha256(source),
            "fae_roster": str(fae_csv),
            "fae_roster_sha256": _sha256(fae_csv),
            "legacy_query_seed": str(legacy_queries),
            "legacy_query_seed_sha256": _sha256(legacy_queries),
            "candidate_library": str(candidate_library),
            "candidate_library_sha256": _sha256(candidate_library),
            "max_reports_per_candidate": MAX_REPORTS_PER_CANDIDATE,
            "followup_window_hours": FOLLOWUP_WINDOW_HOURS,
            "candidate_min_score": MIN_CANDIDATE_SCORE,
        },
        "cases": cases,
        "coverage": _coverage(cases),
    }


def validate_dataset(
    dataset: dict[str, Any],
    legacy_queries: str | Path = DEFAULT_LEGACY_QUERIES,
) -> dict[str, Any]:
    issues: list[str] = []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version")
    if dataset.get("benchmark_id") != BENCHMARK_ID:
        issues.append("benchmark_id")
    cases = dataset.get("cases") or []
    if len(cases) < MIN_CASE_COUNT:
        issues.append("case_count")
    legacy = _load_legacy_queries(Path(legacy_queries))
    legacy_grams = [_trigrams(item) for item in legacy]
    seen_case_ids: set[str] = set()
    seen_query_norms: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id") or "")
        prefix = f"{case_id}:"
        if not case_id or case_id in seen_case_ids:
            issues.append(prefix + "duplicate_case_id")
        seen_case_ids.add(case_id)
        if case.get("source_type") != "xing_lark_real_fae_report":
            issues.append(prefix + "source_type")
        if case.get("expectation_origin") != "real_fae_evidence_candidate":
            issues.append(prefix + "expectation_origin")
        if not case.get("isolated_session"):
            issues.append(prefix + "session_not_isolated")
        if not case.get("quality", {}).get("query_is_real_fae_source_text"):
            issues.append(prefix + "query_not_real_fae_source")
        if case.get("quality", {}).get("graph_ingestion_allowed") is not False:
            issues.append(prefix + "graph_ingestion_allowed")
        source_input = case.get("source_input") or {}
        source_refs = case.get("source_refs") or {}
        source_id = str(source_input.get("message_id") or "")
        if not source_id or source_id not in set(
            source_refs.get("source_input_message_ids") or []
        ):
            issues.append(prefix + "source_input_message")
        query = str(case.get("query") or "")
        if not query or _compact(source_input.get("text"), MAX_QUERY_SOURCE_CHARS) not in query:
            issues.append(prefix + "query_source_text")
        if not source_refs.get("reference_followup_message_ids"):
            issues.append(prefix + "followup_evidence")
        source_time_text = str(source_input.get("create_time") or "")
        try:
            source_time = datetime.strptime(source_time_text, "%Y-%m-%d %H:%M")
        except ValueError:
            issues.append(prefix + "source_time")
            source_time = None
        for evidence in source_refs.get("reference_followup_evidence") or []:
            try:
                evidence_time = datetime.strptime(
                    str(evidence.get("create_time") or ""), "%Y-%m-%d %H:%M"
                )
            except ValueError:
                issues.append(prefix + "followup_time")
                continue
            if source_time and not (
                source_time < evidence_time
                <= source_time + timedelta(hours=FOLLOWUP_WINDOW_HOURS)
            ):
                issues.append(prefix + "followup_outside_window")
        answer = str((case.get("answer_gold") or {}).get("reference_answer") or "")
        if not answer:
            issues.append(prefix + "reference_answer")
        if "verified_fix" in answer and "不得把" not in answer:
            issues.append(prefix + "false_verified_fix")
        if int(case.get("quality", {}).get("candidate_score") or 0) < MIN_CANDIDATE_SCORE:
            issues.append(prefix + "candidate_score")
        if max(
            (
                _gram_similarity(_trigrams(source_input.get("text")), grams)
                for grams in legacy_grams
            ),
            default=0.0,
        ) >= 0.82:
            issues.append(prefix + "duplicates_legacy_query")
        query_norm = _normalized(source_input.get("text"))
        if query_norm in seen_query_norms:
            issues.append(prefix + "duplicates_generated_query")
        seen_query_norms.add(query_norm)
    if dataset.get("coverage") != _coverage(cases):
        issues.append("coverage")
    return {
        "schema_version": "debug_agent_system.aoi_fae_report_benchmark.validation.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
        "coverage": _coverage(cases),
    }


def render_markdown(dataset: dict[str, Any]) -> str:
    cases = dataset.get("cases") or []
    coverage = dataset.get("coverage") or {}
    lines = [
        "# AOI FAE Report Benchmark v2：真实 Query 与参考答案",
        "",
        "> 本文件由真实 Xing Lark FAE 报告构建；每个 Query 的核心现场原文"
        "来自独立 relation-aware session，而不是 KG 对象模板。",
        "> 后续 FAE 消息只作为参考证据，不可在实际推理时倒灌为初始已知事实；"
        "全体 case 仍需人工 Trace 审核，不能作为独立 semantic Gold。",
        "",
        f"- Case 总数：{len(cases)}",
        f"- 议题标签：`{json.dumps(coverage.get('issue_tag_counts') or {}, ensure_ascii=False)}`",
        f"- 候选分数：最低 {coverage.get('candidate_score_min')}，"
        f"中位 {coverage.get('candidate_score_median')}",
        "",
    ]
    for case in cases:
        refs = case["source_refs"]
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- 来源候选：`{refs['candidate_id']}`；分数 `{refs['candidate_score']}`",
                f"- 议题标签：`{', '.join(refs['candidate_issue_tags'])}`",
                f"- 输入消息：`{case['source_input']['message_id']}`",
                f"- 参考后续消息：`{', '.join(refs['reference_followup_message_ids'])}`",
                "",
                "**Query**",
                "",
                *[f"    {line}" for line in case["query"].splitlines()],
                "",
                "**参考答案（后续 FAE 证据卡）**",
                "",
                *[
                    f"    {line}"
                    for line in case["answer_gold"]["reference_answer"].splitlines()
                ],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(prog="aoi-fae-report-benchmark")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fae-csv", type=Path, default=DEFAULT_FAE_CSV)
    parser.add_argument("--candidate-library", type=Path, default=DEFAULT_CANDIDATE_LIBRARY)
    parser.add_argument("--legacy-queries", type=Path, default=DEFAULT_LEGACY_QUERIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        dataset = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        dataset = build_dataset(
            args.source,
            args.fae_csv,
            args.candidate_library,
            args.legacy_queries,
            target_count=args.target_count,
        )
        write_json(args.out, dataset)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(dataset), encoding="utf-8")
    report = validate_dataset(dataset, args.legacy_queries)
    write_json(args.report_out, report)
    print(
        json.dumps(
            {
                "dataset": str(args.out),
                "report": str(args.report_out),
                "status": report["status"],
                "coverage": report["coverage"],
                **({"markdown": str(args.markdown_out)} if not args.validate_only else {}),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
