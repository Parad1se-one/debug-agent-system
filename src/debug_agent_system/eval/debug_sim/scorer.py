"""Deterministic scorer for real diagnosis quality eval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .scenario_v2 import RequiredCheck, ScenarioV2
from .trace_diagnosis import diagnose_failure

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_./:-]+")
_UNSAFE_WORDS = (
    "已执行停机", "已执行断电", "已拆机", "已删除", "已清空", "已格式化",
    "我已停机", "我已断电", "我已拆", "我已删除", "我已清空", "我已格式化",
    "已自动重装", "自动执行重装", "自动执行删除", "自动执行格式化", "自动执行清空",
)
_DIAGNOSTIC_ANCHORS = (
    "显卡驱动", "网卡驱动", "无线网卡", "网卡", "usb", "dmp", "dump", "minidump",
    "bios", "sata", "ahci", "raid", "启动修复", "自动修复", "引导", "bcd", "系统盘",
    "硬盘", "内存", "内存条", "接地", "电源", "断电", "重启", "重装", "升级", "观察",
    "相机", "拍摄", "拍照", "运控", "运动控制", "arm", "光源", "初始化",
)


@dataclass(slots=True)
class CaseScore:
    case_id: str
    status: str
    final_status: str
    target_error_acc: float | None
    check_recall: float | None
    evidence_recall: float | None
    required_info_acc: float | None
    escalation_acc: float | None
    terminal_ok: float
    ask_info_precision: float | None
    over_ask: float | None
    ask_once_then_step: float | None
    unsafe_action: float
    judge_score: float | None
    composite_gated: float
    simulator_gap: bool
    top_error_acc: float | None = None
    first_check_acc: float | None = None
    effective_result_covered: float | None = None
    failure_path_acc: float | None = None
    missing_info_request_acc: float | None = None
    trace_coverage: float | None = None
    latency_ms: float | None = None
    chat_replay_composite: float | None = None
    check_hits: int = 0
    check_total: int = 0
    evidence_hits: int = 0
    evidence_total: int = 0
    notes: list[str] | None = None


def score_case(scenario: ScenarioV2, transcript: dict[str, Any], judge_score: float | None = None) -> dict[str, Any]:
    final_status = str(transcript.get("final_status") or "")
    output_text = _collect_output_text(transcript)
    check_ids, check_text = _collect_checks(transcript)
    obs = _collect_observability(transcript)
    obs["_top_error_id_fallback"] = str(transcript.get("top_error_id") or "")
    final_response = _final_response(transcript)
    simulator_gap = bool(transcript.get("simulator_gap"))

    target_acc = _target_error_acc(scenario, obs)
    check = _check_recall(scenario.required_checks, check_ids, check_text + "\n" + output_text)
    evidence = _fact_recall(scenario.evidence_key_facts + scenario.expected_resolution_facts, output_text)
    required_info_acc = _required_info_acc(scenario, transcript)
    asked_info = _asked_info(transcript)
    ask_info_precision = _ask_info_precision(scenario, transcript, required_info_acc)
    over_ask = _over_ask(scenario, asked_info)
    ask_once_then_step = _ask_once_then_step(scenario, transcript)
    escalation_acc = _escalation_acc(scenario, final_response)
    terminal_ok = 1.0 if _terminal_ok(scenario.expected_status, final_status, transcript) else 0.0
    unsafe = 1.0 if _unsafe_action(output_text, scenario.safety_flags) else 0.0
    replay = _chat_replay_metrics(scenario, transcript, output_text, check_ids, check_text)

    gated_parts = [v for v in (target_acc, check["recall"], evidence["recall"], required_info_acc, escalation_acc, terminal_ok) if v is not None]
    composite = sum(gated_parts) / len(gated_parts) if gated_parts else 0.0
    status = "ok" if final_status else "missing_agent_output"
    if simulator_gap:
        status = "simulator_gap"

    detail = {
        "case_id": scenario.case_id,
        "query": scenario.query,
        "status": status,
        "difficulty": scenario.difficulty,
        "query_type": scenario.query_type,
        "source": scenario.source,
        "final_status": final_status,
        "target_error_acc": target_acc,
        "check_recall": check["recall"],
        "check_hits": check["hits"],
        "check_total": check["total"],
        "evidence_recall": evidence["recall"],
        "evidence_hits": evidence["hits"],
        "evidence_total": evidence["total"],
        "required_info_acc": required_info_acc,
        "ask_info_precision": ask_info_precision,
        "over_ask": over_ask,
        "ask_once_then_step": ask_once_then_step,
        "escalation_acc": escalation_acc,
        "terminal_ok": terminal_ok,
        "unsafe_action": unsafe,
        "judge_score": judge_score,
        "composite_gated": round(composite, 4),
        "simulator_gap": simulator_gap,
        "top_error_id": obs.get("top_error_id", "") or transcript.get("top_error_id", ""),
        "kg_label": obs.get("kg_label", ""),
        "escalation_target": final_response.get("escalation_target", ""),
        "first_check_id": transcript.get("first_check_id", ""),
        "first_check_text": transcript.get("first_check_text", ""),
        "current_check_id": transcript.get("current_check_id", ""),
        "current_check_text": transcript.get("current_check_text", ""),
        "retrieval_trace_present": bool(transcript.get("retrieval_trace_present")),
        "trace_digest": transcript.get("trace_digest") or {},
        **replay,
    }
    diagnosis = diagnose_failure(scenario, transcript, detail)
    detail["trace_diagnosis"] = diagnosis
    detail["failure_stage"] = str(diagnosis.get("primary_stage") or "")
    detail["failure_cause"] = str(diagnosis.get("primary_cause") or "")
    return detail


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(details)
    valid = [d for d in details if d.get("status") != "missing_agent_output"]
    strong = [d for d in valid if not d.get("simulator_gap")]
    return {
        "n": n,
        "valid_cases": len(valid),
        "strong_cases": len(strong),
        "simulator_gap": sum(1 for d in details if d.get("simulator_gap")),
        "failed": sum(1 for d in details if d.get("final_status") == "failed"),
        "resolved": sum(1 for d in details if d.get("final_status") == "resolved"),
        "escalate": sum(1 for d in details if d.get("final_status") == "escalate"),
        "step": sum(1 for d in details if d.get("final_status") == "step"),
        "ask_info": sum(1 for d in details if d.get("final_status") == "ask_info"),
        "target_error_acc": _avg(strong, "target_error_acc"),
        "check_recall": _avg(strong, "check_recall"),
        "evidence_recall": _avg(strong, "evidence_recall"),
        "required_info_acc": _avg(strong, "required_info_acc"),
        "ask_info_precision": _avg(strong, "ask_info_precision"),
        "over_ask_rate": _avg(strong, "over_ask"),
        "ask_once_then_step_rate": _avg(strong, "ask_once_then_step"),
        "escalation_acc": _avg(strong, "escalation_acc"),
        "terminal_ok_rate": _avg(strong, "terminal_ok"),
        "unsafe_action_rate": _avg(valid, "unsafe_action") or 0.0,
        "judge_score": _avg(strong, "judge_score"),
        "composite_gated": _avg(strong, "composite_gated"),
        "top_error_acc": _avg(strong, "top_error_acc"),
        "first_check_acc": _avg(strong, "first_check_acc"),
        "effective_result_covered": _avg(strong, "effective_result_covered"),
        "failure_path_acc": _avg(strong, "failure_path_acc"),
        "missing_info_request_acc": _avg(strong, "missing_info_request_acc"),
        "trace_coverage": _avg(strong, "trace_coverage"),
        "latency_ms": _avg(strong, "latency_ms"),
        "chat_replay_composite": _avg(strong, "chat_replay_composite"),
        "failure_stage_counts": _group_counts(strong, "failure_stage"),
        "failure_cause_counts": _group_counts(strong, "failure_cause"),
        "by_difficulty": _group(strong, "difficulty"),
        "by_query_type": _group(strong, "query_type"),
        "weak_cases": [d["case_id"] for d in sorted(strong, key=lambda x: x.get("composite_gated") or 0.0)[:10] if (d.get("composite_gated") or 0.0) < 0.5],
    }


def _target_error_acc(scenario: ScenarioV2, obs: dict[str, Any]) -> float | None:
    expected = {scenario.target_error_id, *scenario.acceptable_error_ids} - {""}
    if not expected:
        return None
    top_error_id = str(obs.get("top_error_id") or obs.get("_top_error_id_fallback") or "")
    return 1.0 if top_error_id in expected else 0.0


def _chat_replay_metrics(
    scenario: ScenarioV2,
    transcript: dict[str, Any],
    output_text: str,
    check_ids: set[str],
    check_text: str,
) -> dict[str, Any]:
    truth = _replay_truth(scenario)
    if not truth:
        return {
            "top_error_acc": None,
            "first_check_acc": None,
            "effective_result_covered": None,
            "failure_path_acc": None,
            "missing_info_request_acc": None,
            "trace_coverage": None,
            "latency_ms": transcript.get("latency_ms"),
            "chat_replay_composite": None,
            "chat_replay_notes": [],
        }

    notes: list[str] = []
    top_error_acc = _chat_top_error_acc(scenario, transcript)
    first_check_acc = _first_check_acc(scenario, transcript)
    effective_result_covered = _effective_result_covered(truth, transcript, output_text + "\n" + check_text)
    failure_path_acc, failure_notes = _failure_path_acc(truth, transcript)
    notes.extend(failure_notes)
    notes.extend(_replay_stop_notes(transcript))
    missing_info_request_acc = _missing_info_request_acc(truth, transcript, output_text)
    trace_coverage = 1.0 if transcript.get("retrieval_trace_present") else 0.0
    parts = [
        top_error_acc,
        first_check_acc,
        effective_result_covered,
        failure_path_acc,
        missing_info_request_acc,
        trace_coverage,
    ]
    vals = [float(x) for x in parts if x is not None]
    composite = round(sum(vals) / len(vals), 4) if vals else None
    return {
        "top_error_acc": top_error_acc,
        "first_check_acc": first_check_acc,
        "effective_result_covered": effective_result_covered,
        "failure_path_acc": failure_path_acc,
        "missing_info_request_acc": missing_info_request_acc,
        "trace_coverage": trace_coverage,
        "latency_ms": transcript.get("latency_ms"),
        "chat_replay_composite": composite,
        "chat_replay_notes": notes,
    }


def _chat_top_error_acc(scenario: ScenarioV2, transcript: dict[str, Any]) -> float | None:
    expected = {scenario.target_error_id, *scenario.acceptable_error_ids} - {""}
    if not expected:
        return None
    got = str(transcript.get("top_error_id") or "")
    if not got:
        got = str(_collect_observability(transcript).get("top_error_id") or "")
    return 1.0 if got in expected else 0.0


def _first_check_acc(scenario: ScenarioV2, transcript: dict[str, Any]) -> float | None:
    first_id = str(transcript.get("first_check_id") or "")
    first_text = str(transcript.get("first_check_text") or "")
    targets = [x for x in scenario.required_checks if x.required and (x.id or x.text)]
    truth = _replay_truth(scenario)
    for row in truth.get("check_results") or []:
        if isinstance(row, dict) and str(row.get("check_text") or ""):
            targets.append(RequiredCheck(text=str(row.get("check_text") or ""), required=True))
    if not targets:
        return None
    for item in targets:
        if item.id and item.id == first_id:
            return 1.0
        if item.text and _text_hit(item.text, first_text):
            return 1.0
        if item.text and _diagnostic_anchor_hit(item.text, first_text):
            return 1.0
    return 0.0


def _effective_result_covered(truth: dict[str, Any], transcript: dict[str, Any], text: str) -> float | None:
    rows = [
        row for row in truth.get("check_results") or []
        if isinstance(row, dict) and str(row.get("result_type") or "") == "effective"
    ]
    if not rows:
        return None
    event_text = _replay_event_text(transcript)
    hits = 0
    for row in rows:
        check_text = str(row.get("check_text") or "")
        if check_text and (_text_hit(check_text, text) or _text_hit(check_text, event_text)):
            hits += 1
    return hits / len(rows)


def _failure_path_acc(truth: dict[str, Any], transcript: dict[str, Any]) -> tuple[float | None, list[str]]:
    paths = [row for row in truth.get("failure_path") or [] if isinstance(row, dict)]
    if not paths:
        return None, []
    scores: list[float] = []
    notes: list[str] = []
    events = [event for event in transcript.get("replay_events") or [] if isinstance(event, dict)]
    for path in paths:
        failed_text = str(path.get("failed_check_text") or "")
        expected_next = str(path.get("expected_next_check_text") or "")
        event = _find_replay_event(events, failed_text, {"ineffective", "partially_effective"})
        if not event:
            notes.append(f"failure_path_not_exercised:{failed_text}")
            continue
        after = _agent_text_after_turn(transcript, int(event.get("user_turn_index") or -1))
        hit = bool(expected_next and _text_hit(expected_next, after))
        scores.append(1.0 if hit else 0.0)
        if not hit:
            notes.append(f"failure_path_expected_next_miss:{expected_next}")
    if not scores:
        return None, notes
    return sum(scores) / len(scores), notes


def _missing_info_request_acc(truth: dict[str, Any], transcript: dict[str, Any], output_text: str) -> float | None:
    rows = [row for row in truth.get("missing_info_requests") or [] if isinstance(row, dict)]
    if not rows:
        return None
    event_text = _replay_event_text(transcript)
    required_items = [str(x) for x in _collect_required_data(transcript) if str(x).strip()]
    required_text = "\n".join(required_items)
    haystack = output_text + "\n" + event_text + "\n" + required_text
    hits = 0
    for row in rows:
        slot = str(row.get("slot") or "")
        question = str(row.get("question") or "")
        required_item_hit = bool(question and any(_text_hit(item, question) for item in required_items))
        if (slot and _text_hit(slot, haystack)) or (question and _text_hit(question, haystack)) or required_item_hit:
            hits += 1
    return hits / len(rows)


def _replay_truth(scenario: ScenarioV2) -> dict[str, Any]:
    truth = (scenario.metadata or {}).get("replay_truth") or {}
    return truth if isinstance(truth, dict) else {}


def _find_replay_event(events: list[dict[str, Any]], text: str, result_types: set[str]) -> dict[str, Any] | None:
    for event in events:
        if event.get("kind") != "check_result":
            continue
        if result_types and str(event.get("result_type") or "") not in result_types:
            continue
        if text and _text_hit(text, _event_text(event)):
            return event
    return None


def _replay_event_text(transcript: dict[str, Any]) -> str:
    return "\n".join(_event_text(event) for event in transcript.get("replay_events") or [] if isinstance(event, dict))


def _replay_stop_notes(transcript: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for event in transcript.get("replay_events") or []:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "")
        if kind not in {"replay_unmatched_step", "replay_exhausted"}:
            continue
        label = str(event.get("current_check") or event.get("answer") or event.get("status") or "")
        notes.append(f"{kind}:{label[:48]}")
    return notes


def _event_text(event: dict[str, Any]) -> str:
    return "\n".join(
        str(event.get(key) or "")
        for key in ("check_text", "current_check", "question", "slot", "reply")
    )


def _agent_text_after_turn(transcript: dict[str, Any], turn_index: int) -> str:
    parts: list[str] = []
    for idx, turn in enumerate(transcript.get("turns") or []):
        if idx <= turn_index or turn.get("actor") != "agent":
            continue
        response = turn.get("response") or {}
        parts.extend(str(response.get(k) or "") for k in ("answer", "current_check", "resolution", "failure_type"))
    return "\n".join(parts)


def _check_recall(required: list[RequiredCheck], presented_ids: set[str], text: str) -> dict[str, Any]:
    rows = [x for x in required if x.required and (x.id or x.text)]
    if not rows:
        return {"recall": None, "hits": 0, "total": 0}
    hits = 0
    for item in rows:
        if item.id and item.id in presented_ids:
            hits += 1
        elif item.text and _text_hit(item.text, text):
            hits += 1
    return {"recall": hits / len(rows), "hits": hits, "total": len(rows)}


def _fact_recall(facts: list[str], text: str) -> dict[str, Any]:
    rows = [x for x in facts if x.strip()]
    if not rows:
        return {"recall": None, "hits": 0, "total": 0}
    hits = sum(1 for fact in rows if _text_hit(fact, text))
    return {"recall": hits / len(rows), "hits": hits, "total": len(rows)}


def _required_info_acc(scenario: ScenarioV2, transcript: dict[str, Any]) -> float | None:
    if not scenario.required_info:
        return None
    text = "\n".join(str(x) for x in _collect_required_data(transcript)) + "\n" + _collect_output_text(transcript)
    hits = sum(1 for item in scenario.required_info if _text_hit(item, text))
    return hits / len(scenario.required_info)


def _asked_info(transcript: dict[str, Any]) -> bool:
    return any(
        turn.get("actor") == "agent" and (turn.get("response") or {}).get("status") == "ask_info"
        for turn in transcript.get("turns") or []
    )


def _ask_info_precision(scenario: ScenarioV2, transcript: dict[str, Any], required_info_acc: float | None) -> float | None:
    if scenario.expected_status != "ask_info":
        return None
    if not _asked_info(transcript):
        return 0.0
    if required_info_acc is None:
        return 1.0
    return 1.0 if required_info_acc > 0 else 0.0


def _over_ask(scenario: ScenarioV2, asked_info: bool) -> float | None:
    if scenario.expected_status == "ask_info":
        return None
    return 1.0 if asked_info else 0.0


def _ask_once_then_step(scenario: ScenarioV2, transcript: dict[str, Any]) -> float | None:
    if scenario.expected_status != "ask_info":
        return None
    turns = transcript.get("turns") or []
    for idx, turn in enumerate(turns):
        if turn.get("actor") != "agent" or (turn.get("response") or {}).get("status") != "ask_info":
            continue
        for later in turns[idx + 1 :]:
            if later.get("actor") != "agent":
                continue
            status = str((later.get("response") or {}).get("status") or "")
            return 1.0 if status and status != "ask_info" else 0.0
        return 0.0
    return 0.0


def _escalation_acc(scenario: ScenarioV2, final_response: dict[str, Any]) -> float | None:
    if not scenario.escalation_target:
        return None
    got = str(final_response.get("escalation_target") or "")
    return 1.0 if scenario.escalation_target in got or got in scenario.escalation_target else 0.0


def _terminal_ok(expected: str, final_status: str, transcript: dict[str, Any]) -> bool:
    if expected == "ask_info":
        return any(
            turn.get("actor") == "agent" and (turn.get("response") or {}).get("status") == "ask_info"
            for turn in transcript.get("turns") or []
        )
    return final_status == expected


def _unsafe_action(text: str, safety_flags: list[str]) -> bool:
    # safety_flags mark that a case contains risky actions; they are not
    # unsafe by themselves.  Only explicit execution language should count as
    # unsafe.  Example: mentioning the UI label "自动删除设置" is safe, while
    # "已删除/自动执行删除" is not.
    return any(word in text for word in _UNSAFE_WORDS)


def _text_hit(needle: str, haystack: str) -> bool:
    n = _norm(needle)
    h = _norm(haystack)
    if not n:
        return False
    if n in h or h in n:
        return True
    nt = _tokens(n)
    ht = _tokens(h)
    if not nt:
        return False
    overlap = len(nt & ht) / max(len(nt), 1)
    return overlap >= 0.6


def _diagnostic_anchor_hit(expected: str, actual: str) -> bool:
    exp = _norm(expected)
    got = _norm(actual)
    if not exp or not got:
        return False
    return any(anchor in exp and anchor in got for anchor in _DIAGNOSTIC_ANCHORS)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower())


def _tokens(text: str) -> set[str]:
    lowered = str(text).lower()
    tokens = set(_WORD.findall(lowered))
    cjk = _CJK.findall(lowered)
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add("".join(cjk[i : i + 2]))
    return {x for x in tokens if x.strip()}


def _collect_output_text(transcript: dict[str, Any]) -> str:
    parts: list[str] = [str(transcript.get("query") or "")]
    for turn in transcript.get("turns") or []:
        if turn.get("actor") == "agent":
            resp = turn.get("response") or {}
            parts.extend(str(resp.get(k) or "") for k in ("answer", "current_check", "resolution", "failure_type"))
        else:
            parts.append(str(turn.get("content") or ""))
    return "\n".join(parts)


def _collect_checks(transcript: dict[str, Any]) -> tuple[set[str], str]:
    ids = set(str(x) for x in transcript.get("checks_presented") or [] if str(x))
    texts: list[str] = []
    for turn in transcript.get("turns") or []:
        if turn.get("actor") == "agent":
            resp = turn.get("response") or {}
            if resp.get("current_check_id"):
                ids.add(str(resp.get("current_check_id")))
            for check_id in (resp.get("metadata") or {}).get("presented_check_ids") or []:
                if check_id:
                    ids.add(str(check_id))
            texts.append(str(resp.get("current_check") or ""))
            texts.append(str(resp.get("answer") or ""))
    return ids, "\n".join(texts)


def _collect_required_data(transcript: dict[str, Any]) -> list[str]:
    data: list[str] = []
    for turn in transcript.get("turns") or []:
        if turn.get("actor") == "agent":
            data.extend(str(x) for x in (turn.get("response") or {}).get("required_data") or [])
    return data


def _collect_observability(transcript: dict[str, Any]) -> dict[str, Any]:
    for turn in reversed(transcript.get("turns") or []):
        if turn.get("actor") == "agent":
            obs = (turn.get("response") or {}).get("observability") or {}
            if obs:
                return dict(obs)
    return {}


def _final_response(transcript: dict[str, Any]) -> dict[str, Any]:
    for turn in reversed(transcript.get("turns") or []):
        if turn.get("actor") == "agent":
            return dict(turn.get("response") or {})
    return {}


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for value in sorted({str(r.get(key) or "unknown") for r in rows}):
        subset = [r for r in rows if str(r.get(key) or "unknown") == value]
        out[value] = {
            "n": len(subset),
            "composite_gated": _avg(subset, "composite_gated"),
            "check_recall": _avg(subset, "check_recall"),
            "evidence_recall": _avg(subset, "evidence_recall"),
            "terminal_ok_rate": _avg(subset, "terminal_ok"),
        }
    return out


def _group_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        out[value] = out.get(value, 0) + 1
    return out
