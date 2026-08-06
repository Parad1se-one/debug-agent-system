from __future__ import annotations

from dataclasses import dataclass
import math
import re

from debug_agent_system.core.contracts import CheckNode, LockedSubgraph, SessionState, SolutionNode

_SOLVED = ("已解决", "解决了", "恢复", "好了", "正常", "通过", "ok", "OK")
_BAD = ("未解决", "没有", "不行", "失败", "异常", "仍然", "还是", "未恢复")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_./:-]+")
_RENDER_LIMIT = 8


@dataclass(slots=True)
class TraversalDecision:
    status: str
    check: CheckNode | None = None
    solution: SolutionNode | None = None
    reason: str = ""


class TopologyTraversalAgent:
    """B/D: deterministic check traversal over locked subgraph."""

    def first_step(self, state: SessionState, subgraph: LockedSubgraph, skip_check_ids: list[str] | None = None) -> TraversalDecision:
        state.current_index = 0
        return self._next(state, subgraph, set(skip_check_ids or []), "first_step")

    def select_check(self, state: SessionState, subgraph: LockedSubgraph, check_id: str, reason: str) -> TraversalDecision:
        ordered = self._rank_checks(state.query, subgraph)
        by_id = {check.check_id: check for check in ordered}
        check = by_id.get(check_id)
        if check is None:
            return TraversalDecision("escalate", reason=f"unknown_branch_check:{check_id}")
        subgraph.checks[:] = ordered
        state.current_index = ordered.index(check)
        self._present(state, subgraph, ordered, check, set(state.ruled_out), reason)
        return TraversalDecision("step", check=check, reason=reason)

    def after_user_result(self, state: SessionState, subgraph: LockedSubgraph, user_message: str) -> TraversalDecision:
        if state.current_check_id:
            state.check_results[state.current_check_id] = user_message
            if self._is_solved(user_message) or self._is_root_cause_feedback(user_message):
                solved_check_id = self._match_resolution_check(state, subgraph, user_message) or state.current_check_id
                if solved_check_id != state.current_check_id:
                    state.check_results[solved_check_id] = user_message
                solution = self._solution_for(subgraph, solved_check_id)
                state.which_check_solved = solved_check_id
                return TraversalDecision("resolved", solution=solution, reason="user_marked_solved")
            state.ruled_out.append(state.current_check_id)
            state.current_index += 1
        return self._next(state, subgraph, set(), "next_after_result")

    def _next(self, state: SessionState, subgraph: LockedSubgraph, skip: set[str], reason: str) -> TraversalDecision:
        ordered = self._rank_checks(state.query, subgraph)
        subgraph.checks[:] = ordered
        while state.current_index < len(ordered):
            check = ordered[state.current_index]
            if check.check_id in skip or check.check_id in state.ruled_out:
                state.current_index += 1
                state.ruled_out.append(check.check_id)
                continue
            self._present(state, subgraph, ordered, check, skip | set(state.ruled_out), reason)
            return TraversalDecision("step", check=check, reason=reason)
        return TraversalDecision("escalate", reason="no_more_checks")

    def _present(self, state: SessionState, subgraph: LockedSubgraph, ordered: list[CheckNode], check: CheckNode, skip: set[str], reason: str) -> None:
        state.current_check_id = check.check_id
        state.current_check = check.label
        interactive = bool((state.metadata.get("input") or {}).get("interactive", True))
        recommended = [check] if interactive else self._recommended_checks(ordered, check, skip)
        presented_ids = [c.check_id for c in recommended]
        if interactive:
            branch_ids = _branch_context_ids(state)
            presented_ids = _dedupe_ids([check.check_id, *branch_ids])
            by_id = {item.check_id: item for item in ordered}
            recommended = [by_id[check_id] for check_id in presented_ids if check_id in by_id]
        open_hypothesis_ids = self._open_hypothesis_ids(state, subgraph, ordered, check, skip)
        presented_trace = [_check_trace(item, subgraph) for item in recommended]
        selected_trace = _check_trace(check, subgraph)
        state.metadata["traversal"] = {
            "reason": reason,
            "ordered_check_ids": [c.check_id for c in ordered],
            "current_check_id": check.check_id,
            "presented_check_ids": presented_ids,
            "open_hypothesis_check_ids": open_hypothesis_ids,
            "presented_check_trace": presented_trace,
            "selected_check_trace": selected_trace,
            "source_mismatch_first_check": _source_mismatch(check, subgraph),
        }
        state.metadata["open_hypothesis_check_ids"] = open_hypothesis_ids
        state.metadata["presented_check_ids"] = presented_ids
        state.metadata["presented_check_trace"] = presented_trace
        state.metadata["selected_check_trace"] = selected_trace
        state.metadata["source_mismatch_first_check"] = _source_mismatch(check, subgraph)
        for check_id in presented_ids:
            if check_id and check_id not in state.checks_presented:
                state.checks_presented.append(check_id)


    def _open_hypothesis_ids(
        self,
        state: SessionState,
        subgraph: LockedSubgraph,
        ordered: list[CheckNode],
        current: CheckNode,
        skip: set[str],
        limit: int = 6,
    ) -> list[str]:
        """Keep plausible alternatives visible without rendering a long checklist.

        B/D must locate the next best check, but selecting one branch must not
        erase sibling/child hypotheses.  This frontier is metadata for later
        attribution and UI display; it is not an instruction to execute all
        checks immediately.
        """
        blocked = set(skip) | set(state.ruled_out)
        ids: list[str] = []
        for edge in subgraph.next_edges_by_check.get(current.check_id) or []:
            to_id = str(edge.get("to_check_id") or "")
            if to_id and to_id != current.check_id and to_id not in blocked:
                ids.append(to_id)
        for option in state.metadata.get("branch_options") or []:
            if isinstance(option, dict):
                to_id = str(option.get("to_check_id") or "")
                if to_id and to_id != current.check_id and to_id not in blocked:
                    ids.append(to_id)
        for check in ordered:
            if check.check_id != current.check_id and check.check_id not in blocked:
                ids.append(check.check_id)
            if len(_dedupe_ids(ids)) >= limit:
                break
        return _dedupe_ids(ids)[:limit]

    def _match_resolution_check(self, state: SessionState, subgraph: LockedSubgraph, user_message: str) -> str | None:
        by_id = {check.check_id: check for check in subgraph.checks}
        candidate_ids = _dedupe_ids([
            state.current_check_id,
            *[str(x) for x in state.metadata.get("open_hypothesis_check_ids") or []],
            *[str(x) for x in (state.metadata.get("traversal") or {}).get("open_hypothesis_check_ids") or []],
            *[str(x) for x in state.metadata.get("presented_check_ids") or []],
        ])
        candidates = [by_id[x] for x in candidate_ids if x in by_id]
        if not candidates:
            candidates = list(subgraph.checks)
        scored = [(_resolution_match_score(user_message, check, subgraph), check.check_id) for check in candidates]
        scored = [(score, check_id) for score, check_id in scored if score > 0]
        if not scored:
            return state.current_check_id
        scored.sort(reverse=True)
        best_score, best_id = scored[0]
        current_score = next((score for score, check_id in scored if check_id == state.current_check_id), 0)
        if best_id != state.current_check_id and best_score >= max(3, current_score + 2):
            return best_id
        return state.current_check_id

    def _solution_for(self, subgraph: LockedSubgraph, check_id: str) -> SolutionNode | None:
        sols = subgraph.solutions_by_check.get(check_id) or []
        return sols[0] if sols else None

    def _is_solved(self, text: str) -> bool:
        t = text.strip()
        if any(bad in t for bad in _BAD) and not any(ok in t for ok in _SOLVED):
            return False
        if any(pending in t.lower() for pending in ("还需要", "还需", "需要进一步", "进一步确认", "继续观察", "观察中", "待验证", "未确认")):
            return False
        return any(ok in t for ok in _SOLVED)

    def _is_root_cause_feedback(self, text: str) -> bool:
        t = text.strip().lower()
        if any(bad in t for bad in ("不是", "并非", "未确认", "不确定")):
            return False
        return any(token in t for token in ("实际是", "实际上是", "现场反馈是", "确认是", "定位到", "问题是", "故障是", "更换"))

    def _rank_checks(self, query: str, subgraph: LockedSubgraph) -> list[CheckNode]:
        checks = list(subgraph.checks)
        if not checks:
            return checks
        scores = {c.check_id: self._relevance(query, c, subgraph) for c in checks}
        # Keep related branch checks together: relevance first, then topology depth
        # and declared order as tie-breakers.
        return sorted(
            checks,
            key=lambda c: (
                -scores[c.check_id],
                int(c.payload.get("_graph_depth") or 0),
                c.step_order or 9999,
                c.check_id,
            ),
        )

    def _recommended_checks(self, ordered: list[CheckNode], current: CheckNode, skip: set[str]) -> list[CheckNode]:
        out = [current]
        for check in ordered:
            if check.check_id == current.check_id or check.check_id in skip:
                continue
            out.append(check)
            if len(out) >= _RENDER_LIMIT:
                break
        return out

    def _relevance(self, query: str, check: CheckNode, subgraph: LockedSubgraph) -> float:
        text = _check_text(check)
        q_tokens = _tokens(query)
        c_tokens = _tokens(text)
        if not q_tokens or not c_tokens:
            lexical = 0.0
        else:
            lexical = len(q_tokens & c_tokens) / max(math.sqrt(len(q_tokens)), 1.0)
        exact = 0.0
        t = text.lower()
        for token in _salient_phrases(query):
            if token in t:
                exact += min(4.0, max(1.0, len(token) / 3.0))
        # Prefer concrete branch checks over generic entry checks when the query
        # already contains enough symptom detail.
        depth = int(check.payload.get("_graph_depth") or 0)
        depth_bonus = min(depth, 3) * 0.15
        root_penalty = 0.35 if depth == 0 and len(subgraph.checks) > 1 else 0.0
        source_mismatch_penalty = _source_mismatch_penalty(check, subgraph)
        policy_bonus = _policy_prior(check, subgraph)
        already_tried_penalty = _already_tried_penalty(query, check, subgraph)
        safety_penalty = _safety_penalty(check)
        return lexical + exact + depth_bonus + policy_bonus - root_penalty - source_mismatch_penalty - already_tried_penalty - safety_penalty


def _check_text(check: CheckNode) -> str:
    parts = [
        check.check_id,
        check.label,
        check.how_to_check,
        str(check.payload.get("_incoming_condition") or ""),
        str(check.payload.get("_solution_text") or ""),
    ]
    for key in ("condition_tags", "keywords"):
        value = check.payload.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value)
    return " ".join(x for x in parts if x)


def _source_mismatch_penalty(check: CheckNode, subgraph: LockedSubgraph) -> float:
    source_error_id = str(check.payload.get("_source_error_id") or subgraph.error_id)
    if not source_error_id or source_error_id == subgraph.error_id:
        return 0.0
    return 3.0


def _source_mismatch(check: CheckNode, subgraph: LockedSubgraph) -> bool:
    source_error_id = str(check.payload.get("_source_error_id") or subgraph.error_id)
    return bool(source_error_id and source_error_id != subgraph.error_id)


def _check_trace(check: CheckNode, subgraph: LockedSubgraph) -> dict[str, str]:
    source_error_id = str(check.payload.get("_source_error_id") or subgraph.error_id)
    introduced_by = str(check.payload.get("_introduced_by") or ("primary_subgraph" if source_error_id == subgraph.error_id else "supplemental_candidate"))
    return {
        "check_id": check.check_id,
        "label": check.label,
        "source_error_id": source_error_id,
        "source_tier": str(check.payload.get("_source_tier") or ""),
        "introduced_by": introduced_by,
    }


def _policy_prior(check: CheckNode, subgraph: LockedSubgraph) -> float:
    policy = subgraph.payload.get("_diagnostic_policy") if isinstance(subgraph.payload, dict) else {}
    if not isinstance(policy, dict):
        return 0.0
    for item in policy.get("ordered_checks") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("check_id") or "") == check.check_id or str(item.get("label") or "") == check.label:
            return float(item.get("policy_prior") or 0.0)
    return 0.0


def _already_tried_penalty(query: str, check: CheckNode, subgraph: LockedSubgraph) -> float:
    if not any(marker in query for marker in ("已", "已经", "试过", "更换", "检查", "无效", "失败", "排除", "没用")):
        return 0.0
    text = _check_text(check).lower()
    penalty = 0.0
    tried_terms = _tried_action_terms(query)
    negative_context = any(marker in query for marker in ("无效", "失败", "没用", "排除", "未解决", "不行"))
    for term in tried_terms:
        if term and term in text and negative_context:
            penalty += 5.0
    for outcome in check.payload.get("_historical_outcomes") or []:
        if isinstance(outcome, dict) and str(outcome.get("outcome_type") or "") == "ineffective":
            action = str(outcome.get("action_label") or "").lower()
            if action and any(term in action or action in term for term in tried_terms):
                penalty += 3.0
    return min(penalty, 9.0)


def _tried_action_terms(query: str) -> list[str]:
    terms: list[str] = []
    for phrase in _salient_phrases(query):
        cleaned = phrase
        for marker in ("已经", "已", "试过", "无效", "失败", "没用", "排除", "未解决", "不行", "还是", "仍然"):
            cleaned = cleaned.replace(marker, "")
        cleaned = cleaned.strip("，。；、:： ")
        if len(cleaned) >= 2:
            terms.append(cleaned)
        core = cleaned
        for verb in ("更换", "检查", "排查", "替换", "验证", "测试", "确认", "分析"):
            core = core.replace(verb, "")
        core = core.strip("，。；、:： ")
        if len(core) >= 2:
            terms.append(core)
    return _dedupe_ids(terms)


def _safety_penalty(check: CheckNode) -> float:
    if check.destructive:
        return 1.0
    for outcome in check.payload.get("_historical_outcomes") or []:
        if isinstance(outcome, dict) and (outcome.get("high_cost") or outcome.get("destructive")):
            return 0.8
    return 0.0


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(_WORD.findall(lowered))
    cjk = _CJK.findall(lowered)
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add("".join(cjk[i : i + 2]))
    return {x for x in tokens if x.strip()}


def _salient_phrases(text: str) -> list[str]:
    raw = re.split(r"[，。；、\s/()（）]+", text.lower())
    return [x for x in raw if len(x) >= 2]


def _branch_context_ids(state: SessionState) -> list[str]:
    ids: list[str] = []
    pending = state.metadata.get("pending_branch") or {}
    if isinstance(pending, dict) and pending.get("parent_check_id"):
        ids.append(str(pending["parent_check_id"]))
    for option in state.metadata.get("branch_options") or []:
        if isinstance(option, dict) and option.get("to_check_id"):
            ids.append(str(option["to_check_id"]))
    return ids


def _dedupe_ids(ids: list[str]) -> list[str]:
    out: list[str] = []
    for item in ids:
        if item and item not in out:
            out.append(item)
    return out


def _resolution_match_score(text: str, check: CheckNode, subgraph: LockedSubgraph) -> int:
    lowered = text.lower()
    haystack = _check_text(check).lower()
    score = 0
    for token in _salient_phrases(text):
        if token in haystack:
            score += max(1, min(4, len(token) // 2))
    # Domain-significant cause words should carry more weight than generic
    # branch wording such as "fixed code" or "system".  These are still
    # check-text matches, not case-specific memorization.
    weighted = {
        "内存": 8,
        "内存条": 10,
        "memtest": 8,
        "memory": 8,
        "硬盘": 7,
        "smart": 7,
        "磁盘": 6,
        "bcd": 7,
        "启动修复": 6,
        "系统文件": 6,
        "显卡": 6,
        "网卡": 6,
        "电源": 6,
    }
    for token, weight in weighted.items():
        if token in lowered and token in haystack:
            score += weight
    solutions = " ".join(sol.content for sol in subgraph.solutions_by_check.get(check.check_id) or [])
    if solutions:
        sol_text = solutions.lower()
        for token, weight in weighted.items():
            if token in lowered and token in sol_text:
                score += weight
    return score
