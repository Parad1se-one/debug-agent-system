from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.runtime import DebugAgentSystem

from .scenario_v2 import RequiredCheck, ScenarioV2, UserTurn, load_scenarios, to_jsonable
from .scorer import score_case, summarize as summarize_scores
from .trace_diagnosis import build_trace_digest

TERMINAL = {"resolved", "escalate", "failed"}
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_./:-]+")


def build_smoke_scenarios(system: DebugAgentSystem, limit: int) -> list[ScenarioV2]:
    scenarios: list[ScenarioV2] = []
    for error in system.store.errors:
        error_id = str(error.get("error_id") or "")
        if not error_id:
            continue
        try:
            subgraph = system.store.load_locked_subgraph(error_id)
        except Exception:
            continue
        if not subgraph.checks:
            continue
        label = str(error.get("label") or error.get("symptom") or error_id)
        symptom = str(error.get("symptom") or label)
        scenarios.append(ScenarioV2(
            case_id=error_id,
            query=f"现场问题：{label}。{symptom}",
            source="kg_curated",
            query_type="debug",
            target_error_id=error_id,
            acceptable_error_ids=[],
            expected_status="resolved",
            required_checks=[RequiredCheck(id=subgraph.checks[0].check_id, text=subgraph.checks[0].label)],
            evidence_key_facts=[label],
            user_turns=[UserTurn(when_check_contains="", reply="已解决，恢复正常。")],
            max_turns=6,
        ))
        if len(scenarios) >= limit:
            break
    return scenarios


def run_one(system: DebugAgentSystem, scenario: ScenarioV2) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    replay_events: list[dict[str, Any]] = []
    session_id = _safe_session_id(f"eval-{scenario.case_id}")
    interactive = not (scenario.expected_status == "step" and not scenario.user_turns)
    started = time.perf_counter()
    first = system.start({"query": scenario.query, "interactive": interactive, "session": {"session_id": session_id}})
    turns.append({"actor": "agent", "response": first})
    current = first
    checks_presented: list[str] = []
    first_check_id, first_check_text = _first_check("", "", current)
    top_error_id = _response_top_error_id(current)
    retrieval_trace_present = _response_has_retrieval_trace(current)
    simulator_gap = False
    _extend_presented_checks(checks_presented, current)

    # Single-turn expected step scenarios intentionally evaluate the initial answer.
    if scenario.expected_status == "step" and not scenario.user_turns:
        return _transcript(
            scenario,
            current,
            turns,
            checks_presented,
            simulator_gap,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            first_check_id=first_check_id,
            first_check_text=first_check_text or _first_agent_text(turns),
            top_error_id=top_error_id,
            retrieval_trace_present=retrieval_trace_present,
            replay_events=replay_events,
        )

    for _ in range(scenario.max_turns):
        status = str(current.get("status") or "")
        if status in TERMINAL:
            break
        user, event = _simulator_reply_event(scenario, current, replay_events, len(turns))
        if user is None:
            if event:
                replay_events.append(event)
            simulator_gap = not bool(_replay_truth(scenario))
            break
        if event:
            replay_events.append(event)
        turns.append({"actor": "user", "content": user})
        previous_status = status
        current = system.step(str(current.get("session_id") or session_id), user)
        turns.append({"actor": "agent", "response": current})
        first_check_id, first_check_text = _first_check(first_check_id, first_check_text, current)
        top_error_id = top_error_id or _response_top_error_id(current)
        retrieval_trace_present = retrieval_trace_present or _response_has_retrieval_trace(current)
        _extend_presented_checks(checks_presented, current)
        if scenario.expected_status == "ask_info" and previous_status == "ask_info":
            break
        if scenario.expected_status == "step" and not scenario.user_turns:
            break
    return _transcript(
        scenario,
        current,
        turns,
        checks_presented,
        simulator_gap,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        first_check_id=first_check_id,
        first_check_text=first_check_text or _first_agent_text(turns),
        top_error_id=top_error_id,
        retrieval_trace_present=retrieval_trace_present,
        replay_events=replay_events,
    )


def _transcript(
    scenario: ScenarioV2,
    current: dict[str, Any],
    turns: list[dict[str, Any]],
    checks_presented: list[str],
    simulator_gap: bool,
    *,
    latency_ms: float | None = None,
    first_check_id: str = "",
    first_check_text: str = "",
    top_error_id: str = "",
    retrieval_trace_present: bool = False,
    replay_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": scenario.case_id,
        "query": scenario.query,
        "expected_status": scenario.expected_status,
        "final_status": current.get("status"),
        "checks_presented": checks_presented,
        "required_checks": [to_jsonable(x) for x in scenario.required_checks],
        "terminal_ok": current.get("status") in {"resolved", "escalate"},
        "simulator_gap": simulator_gap,
        "latency_ms": latency_ms,
        "first_check_id": first_check_id,
        "first_check_text": first_check_text,
        "top_error_id": top_error_id,
        "retrieval_trace_present": bool(retrieval_trace_present),
        "current_check_id": str(current.get("current_check_id") or ""),
        "current_check_text": str(current.get("current_check") or ""),
        "presented_check_trace": list((current.get("metadata") or {}).get("presented_check_trace") or []),
        "selected_check_trace": dict((current.get("metadata") or {}).get("selected_check_trace") or {}),
        "branch_trace": list((current.get("metadata") or {}).get("branch_trace") or []),
        "branch_options": list((current.get("metadata") or {}).get("branch_options") or []),
        "trace_digest": build_trace_digest({
            "turns": turns,
            "current_check_id": str(current.get("current_check_id") or ""),
            "first_check_id": first_check_id,
            "presented_check_trace": list((current.get("metadata") or {}).get("presented_check_trace") or []),
            "selected_check_trace": dict((current.get("metadata") or {}).get("selected_check_trace") or {}),
            "branch_trace": list((current.get("metadata") or {}).get("branch_trace") or []),
            "branch_options": list((current.get("metadata") or {}).get("branch_options") or []),
        }),
        "replay_events": replay_events or [],
        "turns": turns,
    }


def _simulator_reply(scenario: ScenarioV2, response: dict[str, Any]) -> str | None:
    reply, _ = _simulator_reply_event(scenario, response, [], 0)
    return reply


def _simulator_reply_event(
    scenario: ScenarioV2,
    response: dict[str, Any],
    replay_events: list[dict[str, Any]],
    agent_turn_index: int,
) -> tuple[str | None, dict[str, Any] | None]:
    status = str(response.get("status") or "")
    if status == "ask_info":
        reply, event = _replay_missing_info_reply(scenario, response, replay_events, agent_turn_index)
        if reply:
            return reply, event
        if scenario.user_turns:
            turn = _next_user_turn(scenario.user_turns, replay_events, "")
            if turn is None:
                return None, _replay_stop_event("replay_exhausted", response, agent_turn_index)
            return turn.reply or None, {
                "kind": "user_turn",
                "status": status,
                "reply": turn.reply,
                "agent_turn_index": agent_turn_index,
                "user_turn_index": agent_turn_index + 1,
            }
        if scenario.required_info:
            return "补充：" + "；".join(f"{x}=未知" for x in scenario.required_info), None
        required = response.get("required_data") or []
        if required:
            return "补充：" + "；".join(f"{x}=未知" for x in required), None
        return "补充：未知。", None

    if status == "step":
        reply, event = _replay_check_result_reply(scenario, response, replay_events, agent_turn_index)
        if reply:
            return reply, event
        text = "\n".join(str(response.get(k) or "") for k in ("current_check", "answer"))
        for turn in scenario.user_turns:
            if not turn.when_check_contains or turn.when_check_contains in text:
                if _user_turn_used(replay_events, turn.reply, turn.when_check_contains):
                    continue
                return turn.reply or None, {
                    "kind": "user_turn",
                    "status": status,
                    "matched_text": turn.when_check_contains,
                    "reply": turn.reply,
                    "agent_turn_index": agent_turn_index,
                    "user_turn_index": agent_turn_index + 1,
                }
        if _replay_truth(scenario):
            return None, _replay_stop_event("replay_unmatched_step", response, agent_turn_index)
        return None, None
    return None, None


def _extend_presented_checks(out: list[str], response: dict[str, Any]) -> None:
    ids = list((response.get("metadata") or {}).get("presented_check_ids") or [])
    if response.get("current_check_id"):
        ids.append(str(response["current_check_id"]))
    for check_id in ids:
        if check_id and check_id not in out:
            out.append(str(check_id))


def _replay_check_result_reply(
    scenario: ScenarioV2,
    response: dict[str, Any],
    replay_events: list[dict[str, Any]],
    agent_turn_index: int,
) -> tuple[str | None, dict[str, Any] | None]:
    rows = _replay_truth(scenario).get("check_results") or []
    if not rows:
        return None, None
    used = {
        int(event.get("check_result_index"))
        for event in replay_events
        if event.get("kind") == "check_result" and str(event.get("check_result_index") or "").isdigit()
    }
    response_text = _response_match_text(response)
    for idx, row in enumerate(rows):
        if idx in used or not isinstance(row, dict):
            continue
        check_text = str(row.get("check_text") or "")
        reply = str(row.get("user_reply") or "")
        if check_text and reply and _replay_text_hit(check_text, response_text):
            return reply, {
                "kind": "check_result",
                "check_result_index": idx,
                "check_text": check_text,
                "result_type": str(row.get("result_type") or ""),
                "reply": reply,
                "evidence_message_ids": [str(x) for x in row.get("evidence_message_ids") or []],
                "agent_turn_index": agent_turn_index,
                "user_turn_index": agent_turn_index + 1,
                "current_check_id": str(response.get("current_check_id") or ""),
                "current_check": str(response.get("current_check") or ""),
            }
    return None, None


def _replay_missing_info_reply(
    scenario: ScenarioV2,
    response: dict[str, Any],
    replay_events: list[dict[str, Any]],
    agent_turn_index: int,
) -> tuple[str | None, dict[str, Any] | None]:
    rows = _replay_truth(scenario).get("missing_info_requests") or []
    if not rows:
        return None, None
    used = {
        str(event.get("slot") or "")
        for event in replay_events
        if event.get("kind") == "missing_info_request"
    }
    required_items = [str(x) for x in response.get("required_data") or [] if str(x).strip()]
    response_text = _response_match_text(response) + "\n" + "\n".join(required_items)
    for row in rows:
        if not isinstance(row, dict):
            continue
        slot = str(row.get("slot") or "")
        question = str(row.get("question") or "")
        if slot in used:
            continue
        required_item_hit = bool(question and any(_replay_text_hit(item, question) for item in required_items))
        if (
            (slot and _replay_text_hit(slot, response_text))
            or (question and _replay_text_hit(question, response_text))
            or required_item_hit
        ):
            reply = f"补充：{question or slot}。现场已有对应反馈。"
            return reply, {
                "kind": "missing_info_request",
                "slot": slot,
                "question": question,
                "reply": reply,
                "provided_later": bool(row.get("provided_later")),
                "evidence_message_ids": [str(x) for x in row.get("evidence_message_ids") or []],
                "agent_turn_index": agent_turn_index,
                "user_turn_index": agent_turn_index + 1,
            }
    return None, None


def _replay_truth(scenario: ScenarioV2) -> dict[str, Any]:
    truth = (scenario.metadata or {}).get("replay_truth") or {}
    return truth if isinstance(truth, dict) else {}


def _replay_stop_event(kind: str, response: dict[str, Any], agent_turn_index: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": str(response.get("status") or ""),
        "current_check_id": str(response.get("current_check_id") or ""),
        "current_check": str(response.get("current_check") or ""),
        "answer": str(response.get("answer") or ""),
        "required_data": [str(x) for x in response.get("required_data") or []],
        "agent_turn_index": agent_turn_index,
    }


def _response_match_text(response: dict[str, Any]) -> str:
    parts = [
        str(response.get("current_check_id") or ""),
        str(response.get("current_check") or ""),
        str(response.get("answer") or ""),
        str(response.get("resolution") or ""),
        "\n".join(str(x) for x in response.get("required_data") or []),
    ]
    return "\n".join(parts)


def _first_check(current_id: str, current_text: str, response: dict[str, Any]) -> tuple[str, str]:
    if current_id or current_text:
        return current_id, current_text
    if str(response.get("status") or "") != "step":
        return "", ""
    return str(response.get("current_check_id") or ""), str(response.get("current_check") or "")


def _response_top_error_id(response: dict[str, Any]) -> str:
    obs = response.get("observability") or {}
    return str(obs.get("top_error_id") or "")


def _response_has_retrieval_trace(response: dict[str, Any]) -> bool:
    trace = (response.get("metadata") or {}).get("retrieval_trace") or {}
    if not isinstance(trace, dict):
        return False
    return bool(trace.get("candidate_paths") or trace.get("seed_events") or trace.get("expanded_events"))


def _first_agent_text(turns: list[dict[str, Any]]) -> str:
    for turn in turns:
        if turn.get("actor") != "agent":
            continue
        response = turn.get("response") or {}
        text = str(response.get("current_check") or response.get("answer") or "")
        if text:
            return text
    return ""


def _next_user_turn(turns: list[UserTurn], replay_events: list[dict[str, Any]], match_text: str) -> UserTurn | None:
    for turn in turns:
        if match_text and turn.when_check_contains and turn.when_check_contains not in match_text:
            continue
        if _user_turn_used(replay_events, turn.reply, turn.when_check_contains):
            continue
        return turn
    return None


def _user_turn_used(replay_events: list[dict[str, Any]], reply: str, matched_text: str) -> bool:
    for event in replay_events:
        if event.get("kind") != "user_turn":
            continue
        if str(event.get("reply") or "") != str(reply or ""):
            continue
        event_match = str(event.get("matched_text") or "")
        if not event_match or not matched_text or event_match == str(matched_text or ""):
            return True
    return False


def _replay_text_hit(needle: str, haystack: str) -> bool:
    n = _norm(needle)
    h = _norm(haystack)
    if not n or not h:
        return False
    if n in h or h in n:
        return True
    nt = _tokens(n)
    ht = _tokens(h)
    if not nt:
        return False
    return len(nt & ht) / max(len(nt), 1) >= 0.6


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


def score_transcripts(scenarios: list[ScenarioV2], transcripts: list[dict[str, Any]], judge_policy: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {s.case_id: s for s in scenarios}
    details: list[dict[str, Any]] = []
    for transcript in transcripts:
        scenario = by_id[str(transcript.get("case_id") or "")]
        # v1 keeps LLM judge report-only and absent unless a future judge file is provided.
        judge_score = None if judge_policy in {"none", "report-only"} else None
        details.append(score_case(scenario, transcript, judge_score=judge_score))
    return details, summarize_scores(details)


def legacy_summary(transcripts: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(transcripts)
    if n == 0:
        return {"n": 0, "terminal_ok_rate": 0.0, "avg_check_recall": 0.0}
    check_recalls = []
    for t in transcripts:
        required = [x for x in t.get("required_checks") or [] if x]
        req_ids = {str(x.get("id") or x.get("text") or x) for x in required}
        presented = set(str(x) for x in t.get("checks_presented") or [])
        check_recalls.append(len(req_ids & presented) / len(req_ids) if req_ids else 1.0)
    return {
        "n": n,
        "resolved": sum(1 for t in transcripts if t["final_status"] == "resolved"),
        "escalate": sum(1 for t in transcripts if t["final_status"] == "escalate"),
        "failed": sum(1 for t in transcripts if t["final_status"] == "failed"),
        "terminal_ok_rate": round(sum(1 for t in transcripts if t["terminal_ok"]) / n, 4),
        "avg_check_recall": round(sum(check_recalls) / n, 4),
    }


def build_meta(args: argparse.Namespace, n: int) -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.eval_run.v1",
        "config": str(args.config),
        "scenario_file": str(args.scenario_file or ""),
        "judge": args.judge,
        "n": n,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
    }


def write_run(run: dict[str, Any], out_dir: str | Path, real: bool, latest_name: str = "") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "latest.txt").write_text(str(path) + "\n", encoding="utf-8")
    if real and not latest_name:
        (out / "latest_real.txt").write_text(str(path) + "\n", encoding="utf-8")
    if latest_name:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in latest_name)
        if not safe_name.endswith(".txt"):
            safe_name += ".txt"
        (out / safe_name).write_text(str(path) + "\n", encoding="utf-8")
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _safe_session_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/debug_agent_system.yaml")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out-dir", default="data/results/runs")
    parser.add_argument("--scenario-file", default="")
    parser.add_argument("--judge", choices=["none", "report-only"], default="none")
    parser.add_argument("--latest-name", default="", help="also write an isolated latest pointer, e.g. latest_ask_info.txt")
    args = parser.parse_args(argv)

    system = DebugAgentSystem(load_config(args.config))
    real = bool(args.scenario_file)
    scenarios = load_scenarios(args.scenario_file, args.limit) if real else build_smoke_scenarios(system, args.limit)
    transcripts = [run_one(system, scenario) for scenario in scenarios]
    if real:
        details, summary = score_transcripts(scenarios, transcripts, args.judge)
    else:
        details, summary = [], legacy_summary(transcripts)
    run = {
        "run_id": f"debug_sim_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "meta": build_meta(args, len(scenarios)),
        "summary": summary,
        "details": details,
        "transcripts": transcripts,
    }
    write_run(run, args.out_dir, real=real, latest_name=args.latest_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if real:
        return 0 if (summary.get("failed") or 0) == 0 else 1
    return 0 if summary.get("terminal_ok_rate") == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
