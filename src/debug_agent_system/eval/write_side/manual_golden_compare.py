"""Compare W2 auto extraction against manual KG write annotations.

This evaluator is intentionally deterministic and lightweight.  It does not
write the main KG.  It loads manual review examples, obtains the corresponding
W1 episodes, runs W2, and reports review-grade overlap metrics for the fields
that matter before batch candidate generation: Error labels, Trace order,
Outcome action/type modeling, and RequiredInfo slots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write import ChatCollectAgent, KnowledgeExtractionAgent, WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_.:-]+")

REQUIRED_INFO_SLOT_ALIASES = {
    "dmp_package": "log_package",
    "dump_package": "log_package",
    "diagnostic_data": "log_package",
    "diagnostic_log": "log_package",
    "driver_allocation_trace": "log_package",
    "wpr_trace": "log_package",
    "poolmon_trace": "log_package",
    "blue_screen_code": "error_message",
    "bugcheck_code": "error_message",
    "error_code": "error_message",
    "pte_exhaustion_signals": "error_message",
    "graphics_driver_version": "software_version",
    "driver_version": "software_version",
    "version_and_memory_context": "environment",
    "memory_config": "environment",
    "memory_cpu_test": "environment",
    "driver_context": "environment",
    "production_constraint": "environment",
    "recurrence_after_driver_change": "repro_steps",
    "recurrence_after_mitigation": "repro_steps",
    "capture_behavior_after_toggle": "repro_steps",
    "nic_role_map": "ip_config",
    "filter_driver_binding": "ip_config",
}


def _normalize_required_info_slot(slot: str) -> str:
    value = str(slot or "").strip()
    return REQUIRED_INFO_SLOT_ALIASES.get(value, value)


@dataclass(slots=True)
class ManualCase:
    case_id: str
    sample_id: str
    source_episode_id: str
    source_thread_id: str
    file: str
    candidate: dict[str, Any]


def load_manual_cases(root: str | Path) -> list[ManualCase]:
    base = Path(root)
    index_path = base / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"manual index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    out: list[ManualCase] = []
    for row in index.get("cases") or []:
        if not isinstance(row, dict):
            continue
        path = base / str(row.get("file") or "")
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        proposal = payload.get("refined_merge_proposal") if isinstance(payload.get("refined_merge_proposal"), dict) else {}
        proposals: list[dict[str, Any]] = []
        if proposal.get("nodes"):
            proposals.append(proposal)
        for key in ("primary_candidate", "secondary_candidate"):
            if isinstance(proposal.get(key), dict):
                proposals.append(proposal[key])
        for idx, candidate in enumerate(proposals, start=1):
            clean = _normalise_manual_candidate(candidate, payload)
            case_id = str(row.get("case_id") or path.stem)
            if len(proposals) > 1:
                case_id = f"{case_id}:{idx}"
            out.append(ManualCase(
                case_id=case_id,
                sample_id=str(row.get("sample_id") or payload.get("sample_id") or ""),
                source_episode_id=str(row.get("source_episode_id") or payload.get("source_episode_id") or ""),
                source_thread_id=str(row.get("source_thread_id") or payload.get("source_thread_id") or ""),
                file=path.name,
                candidate=clean,
            ))
    return out


def _normalise_manual_candidate(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    edges = []
    for edge in candidate.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        clean = dict(edge)
        if "relation" not in clean and clean.get("type"):
            clean["relation"] = clean["type"]
        edges.append(clean)
    nodes = [dict(node) for node in candidate.get("nodes") or [] if isinstance(node, dict)]
    outcomes = [node for node in nodes if node.get("type") == "DiagnosticOutcome"]
    return {
        "candidate_id": candidate.get("candidate_id") or f"manual:{payload.get('sample_id') or payload.get('review_id') or 'unknown'}",
        "source_episode_id": candidate.get("source_episode_id") or payload.get("source_episode_id") or "",
        "source_thread_id": candidate.get("source_thread_id") or payload.get("source_thread_id") or "",
        "nodes": nodes,
        "edges": edges,
        "diagnostic_outcomes": outcomes,
        "diagnostic_trace": next((node for node in nodes if node.get("type") == "DiagnosticTrace"), {}),
        "case_variant_candidate": next((node for node in nodes if node.get("type") == "Error"), {}),
        "evidence_ids": candidate.get("evidence_message_ids") or payload.get("evidence_message_ids") or [],
    }


def load_episodes(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def collect_episodes(import_root: str | Path, *, hits_only: bool = False, limit: int = 0) -> list[dict[str, Any]]:
    run = ChatCollectAgent().import_xing_upload(import_root, limit=limit, hits_only=hits_only, out_dir=None)
    return WriteSidePipeline._episodes_from_summaries(run["thread_summaries"])


def compare_manual_cases(
    *,
    manual_root: str | Path,
    kg_root: str | Path = "data/kg",
    episodes_path: str | Path | None = None,
    import_root: str | Path | None = None,
    hits_only: bool = False,
    limit: int = 0,
    deepseek: bool | None = None,
) -> dict[str, Any]:
    manual_cases = load_manual_cases(manual_root)
    episodes = load_episodes(episodes_path)
    found = {str(ep.get("episode_id") or ""): ep for ep in episodes if isinstance(ep, dict)}
    missing_ids = [case.source_episode_id for case in manual_cases if case.source_episode_id not in found]
    if missing_ids and import_root:
        for ep in collect_episodes(import_root, hits_only=hits_only, limit=limit):
            if isinstance(ep, dict):
                found[str(ep.get("episode_id") or "")] = ep
    episodes_by_thread: dict[str, list[dict[str, Any]]] = {}
    for episode in found.values():
        episodes_by_thread.setdefault(str(episode.get("thread_id") or episode.get("source_thread_id") or ""), []).append(episode)
    store = JsonKGStore(kg_root)
    extractor = KnowledgeExtractionAgent(store, deepseek_enabled=deepseek)
    details: list[dict[str, Any]] = []
    for case in manual_cases:
        candidate_episodes = _candidate_episodes_for_case(case, found, episodes_by_thread.get(case.source_thread_id, []))
        if not candidate_episodes:
            details.append({
                "case_id": case.case_id,
                "sample_id": case.sample_id,
                "source_episode_id": case.source_episode_id,
                "status": "missing_episode",
                "scores": _empty_scores(),
                "missing": ["episode"],
            })
            continue
        best_detail: dict[str, Any] | None = None
        considered: list[dict[str, Any]] = []
        for episode, matched_by in candidate_episodes:
            auto = extractor.extract(episode)
            detail = _compare_case(case, auto)
            detail["matched_episode_id"] = str(episode.get("episode_id") or "")
            detail["matched_episode_by"] = matched_by
            considered.append({
                "episode_id": detail["matched_episode_id"],
                "matched_by": matched_by,
                "score": round(_detail_rank(detail), 4),
                "passed_minimum_overlap": bool(detail.get("passed_minimum_overlap")),
                "auto_error_label": (detail.get("auto") or {}).get("error_label"),
            })
            if best_detail is None or _detail_rank(detail) > _detail_rank(best_detail):
                best_detail = detail
        assert best_detail is not None
        best_detail["episode_candidates_considered"] = considered
        details.append(best_detail)
    summary = _summarise(details)
    return {
        "schema_version": "debug_agent_system.write_manual_golden_compare.v1",
        "manual_root": str(manual_root),
        "episodes_path": str(episodes_path or ""),
        "import_root": str(import_root or ""),
        "deepseek_enabled": bool(deepseek),
        "summary": summary,
        "details": details,
    }


def _candidate_episodes_for_case(
    case: ManualCase,
    found: dict[str, dict[str, Any]],
    thread_episodes: list[dict[str, Any]],
    *,
    top_k: int = 4,
) -> list[tuple[dict[str, Any], str]]:
    """Return bounded current-W1 episodes to compare against one manual case.

    Manual examples were authored against an earlier W1 segmentation.  The
    exact episode id can still exist but point at a thin ask-info turn while the
    diagnosis evidence moved to a nearby current episode.  The golden gate
    should therefore answer "did W1/W2 produce a matching candidate in this
    thread?", not overfit to a stale episode ordinal.
    """

    out: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    exact = found.get(case.source_episode_id)
    if exact:
        exact_score = _similar(_manual_case_text(case), _episode_compare_text(exact, include_context=False))
        out.append((exact, "exact_source_episode_id" if exact_score >= 0.05 else "exact_source_episode_id_low_direct_similarity"))
        seen.add(str(exact.get("episode_id") or ""))
    scored: list[tuple[float, dict[str, Any]]] = []
    manual_text = _manual_case_text(case)
    for episode in thread_episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id or episode_id in seen:
            continue
        direct = _episode_compare_text(episode, include_context=False)
        score = _similar(manual_text, direct)
        if score >= 0.02:
            scored.append((score, episode))
    for _, episode in sorted(scored, key=lambda item: item[0], reverse=True)[: max(0, top_k - len(out))]:
        out.append((episode, "thread_direct_text_similarity"))
        seen.add(str(episode.get("episode_id") or ""))
    if out:
        return out
    fallback = _fallback_episode_for_case(case, thread_episodes)
    return [fallback] if fallback else []


def _fallback_episode_for_case(case: ManualCase, episodes: list[dict[str, Any]]) -> tuple[dict[str, Any], str] | None:
    """Find the best current W1 episode when deterministic splitting changed IDs."""

    if not episodes:
        return None
    manual_text = _manual_case_text(case)
    best_direct: tuple[float, dict[str, Any] | None] = (0.0, None)
    for episode in episodes:
        episode_text = _episode_compare_text(episode, include_context=False)
        score = _similar(manual_text, episode_text)
        if score > best_direct[0]:
            best_direct = (score, episode)
    if best_direct[0] >= 0.02 and best_direct[1]:
        return best_direct[1], "thread_direct_text_similarity"

    best_context: tuple[float, dict[str, Any] | None] = (0.0, None)
    for episode in episodes:
        episode_text = _episode_compare_text(episode, include_context=True)
        score = _similar(manual_text, episode_text)
        if score > best_context[0]:
            best_context = (score, episode)
    return (best_context[1], "thread_context_text_similarity") if best_context[0] >= 0.02 and best_context[1] else None


def _manual_case_text(case: ManualCase) -> str:
    manual_texts = [
        _node_label(_first_node(case.candidate, "Error")),
        *_trace_labels(case.candidate),
        *(_clean_text(item.get("action_label") or item.get("label") or item.get("content") or "") for item in _outcomes(case.candidate)),
    ]
    return " ".join(text for text in manual_texts if text)


def _episode_compare_text(episode: dict[str, Any], *, include_context: bool = True) -> str:
    texts: list[str] = []
    keys = ["fault_description_messages", "diagnostic_chain_messages", "resolution_messages"]
    if include_context:
        keys.append("case_context_messages")
    for key in keys:
        for msg in episode.get(key) or []:
            if isinstance(msg, dict):
                texts.append(str(msg.get("text") or msg.get("content_summary") or ""))
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    for key in ("symptom_raw", "conclusion", "key_conclusion"):
        texts.append(str(extracted.get(key) or ""))
    texts.extend(str(x) for x in extracted.get("debug_actions") or [] if x)
    return " ".join(texts)


def _detail_rank(detail: dict[str, Any]) -> float:
    scores = detail.get("scores") if isinstance(detail.get("scores"), dict) else {}
    return (
        (1.0 if detail.get("passed_minimum_overlap") else 0.0) * 3.0
        + float(scores.get("error_label_similarity") or 0.0)
        + float(scores.get("check_label_recall") or 0.0)
        + float(scores.get("trace_order_recall") or 0.0)
        + float(scores.get("outcome_action_recall") or 0.0)
        + float(scores.get("outcome_type_precision") or 0.0) * 0.5
        + float(scores.get("required_info_slot_recall") or 0.0) * 0.25
    )


def _compare_case(case: ManualCase, auto: dict[str, Any]) -> dict[str, Any]:
    manual = case.candidate
    manual_error = _first_node(manual, "Error")
    auto_error = _first_node(auto, "Error")
    manual_checks = _nodes(manual, "DiagnosticCheck")
    auto_checks = _nodes(auto, "DiagnosticCheck")
    manual_outcomes = _outcomes(manual)
    auto_outcomes = _outcomes(auto)
    manual_required_slots = _manual_required_slots(manual)
    auto_required_slots = {
        _normalize_required_info_slot(str(item.get("slot") or ""))
        for item in auto.get("required_info_candidates") or []
        if isinstance(item, dict) and item.get("slot")
    }
    trace_order = _trace_order_recall(_trace_labels(manual), _trace_labels(auto))
    outcome_compare = _outcome_compare(manual_outcomes, auto_outcomes)
    check_recall = _label_recall([_node_label(x) for x in manual_checks], [_node_label(x) for x in auto_checks])
    required_slot_recall = _set_recall(manual_required_slots, auto_required_slots)
    error_similarity = _best_similarity(_node_label(manual_error), [_node_label(auto_error), str(auto.get("label") or "")])
    scores = {
        "error_label_similarity": round(error_similarity, 4),
        "check_label_recall": round(check_recall, 4),
        "trace_order_recall": round(trace_order, 4),
        "outcome_action_recall": round(outcome_compare["action_recall"], 4),
        "outcome_type_precision": round(outcome_compare["type_precision"], 4),
        "required_info_slot_recall": round(required_slot_recall, 4),
        "schema_valid": 1.0 if auto.get("schema_valid") else 0.0,
    }
    strong_structure_hit = scores["check_label_recall"] >= 0.2 or scores["outcome_action_recall"] >= 0.2 or scores["trace_order_recall"] >= 0.2
    weak_structure_hit = scores["check_label_recall"] >= 0.15 or scores["outcome_action_recall"] >= 0.15 or scores["trace_order_recall"] >= 0.15
    passed = bool(auto.get("schema_valid")) and (
        (scores["error_label_similarity"] >= 0.15 and strong_structure_hit)
        or (scores["error_label_similarity"] >= 0.4 and weak_structure_hit)
    )
    return {
        "case_id": case.case_id,
        "sample_id": case.sample_id,
        "source_episode_id": case.source_episode_id,
        "status": "compared",
        "passed_minimum_overlap": passed,
        "manual": {
            "error_label": _node_label(manual_error),
            "check_labels": [_node_label(x) for x in manual_checks],
            "trace_labels": _trace_labels(manual),
            "outcomes": [_outcome_signature(x) for x in manual_outcomes],
            "required_info_slots": sorted(manual_required_slots),
        },
        "auto": {
            "candidate_id": auto.get("candidate_id"),
            "schema_valid": bool(auto.get("schema_valid")),
            "schema_issues": auto.get("schema_issues") or [],
            "deepseek_used": bool((auto.get("observability") or {}).get("deepseek_used")),
            "error_label": _node_label(auto_error) or str(auto.get("label") or ""),
            "check_labels": [_node_label(x) for x in auto_checks],
            "trace_labels": _trace_labels(auto),
            "outcomes": [_outcome_signature(x) for x in auto_outcomes],
            "required_info_slots": sorted(auto_required_slots),
        },
        "scores": scores,
        "matched_outcomes": outcome_compare["matched"],
    }


def _summarise(details: list[dict[str, Any]]) -> dict[str, Any]:
    compared = [d for d in details if d.get("status") == "compared"]
    def avg(key: str) -> float:
        vals = [float((d.get("scores") or {}).get(key) or 0.0) for d in compared]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
    schema_valid_rate = avg("schema_valid")
    minimum_overlap_pass_rate = round(sum(1 for d in compared if d.get("passed_minimum_overlap")) / len(compared), 4) if compared else 0.0
    # This is deliberately conservative: expanding real chat batch candidates is
    # only safe when every manual case is found, all auto candidates are schema
    # valid, and at least most cases show non-trivial overlap with the manual
    # refined proposal.  A false value is not a test failure; it is the gate
    # doing its job by preventing noisy batch expansion.
    ready_for_batch = bool(compared) and len(compared) == len(details) and schema_valid_rate >= 1.0 and minimum_overlap_pass_rate >= 0.8
    return {
        "cases": len(details),
        "compared": len(compared),
        "missing_episode": sum(1 for d in details if d.get("status") == "missing_episode"),
        "schema_valid_rate": schema_valid_rate,
        "minimum_overlap_pass_rate": minimum_overlap_pass_rate,
        "avg_error_label_similarity": avg("error_label_similarity"),
        "avg_check_label_recall": avg("check_label_recall"),
        "avg_trace_order_recall": avg("trace_order_recall"),
        "avg_outcome_action_recall": avg("outcome_action_recall"),
        "avg_outcome_type_precision": avg("outcome_type_precision"),
        "avg_required_info_slot_recall": avg("required_info_slot_recall"),
        "ready_for_batch_candidates": ready_for_batch,
        "readiness_reason": "ready" if ready_for_batch else "manual_golden_overlap_below_batch_threshold",
    }


def _empty_scores() -> dict[str, float]:
    return {
        "error_label_similarity": 0.0,
        "check_label_recall": 0.0,
        "trace_order_recall": 0.0,
        "outcome_action_recall": 0.0,
        "outcome_type_precision": 0.0,
        "required_info_slot_recall": 0.0,
        "schema_valid": 0.0,
    }


def _nodes(candidate: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [node for node in candidate.get("nodes") or [] if isinstance(node, dict) and node.get("type") == node_type]


def _first_node(candidate: dict[str, Any], node_type: str) -> dict[str, Any]:
    nodes = _nodes(candidate, node_type)
    return nodes[0] if nodes else {}


def _outcomes(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    out = _nodes(candidate, "DiagnosticOutcome")
    for item in candidate.get("diagnostic_outcomes") or []:
        if isinstance(item, dict) and item not in out:
            out.append(item)
    return out


def _node_label(node: dict[str, Any]) -> str:
    return _clean_text(node.get("label") or node.get("action_label") or node.get("content") or node.get("symptom") or "")


def _trace_labels(candidate: dict[str, Any]) -> list[str]:
    trace = candidate.get("diagnostic_trace") if isinstance(candidate.get("diagnostic_trace"), dict) else _first_node(candidate, "DiagnosticTrace")
    labels: list[str] = []
    for key in ("recommended_order", "actual_order"):
        for item in trace.get(key) or []:
            if isinstance(item, dict):
                label = _clean_text(item.get("label") or item.get("action_label") or item.get("check_id") or "")
            else:
                label = _clean_text(item)
            if label:
                labels.append(label)
        if labels:
            break
    return labels


def _manual_required_slots(candidate: dict[str, Any]) -> set[str]:
    slots: set[str] = set()
    for node in _nodes(candidate, "Error"):
        for item in node.get("required_info_schema") or []:
            if isinstance(item, dict) and item.get("slot"):
                slots.add(_normalize_required_info_slot(str(item["slot"])))
    return slots


def _trace_order_recall(manual: list[str], auto: list[str]) -> float:
    if not manual:
        return 1.0
    if not auto:
        return 0.0
    hits = 0
    cursor = 0
    for label in manual:
        for idx in range(cursor, len(auto)):
            if _similar(label, auto[idx]) >= 0.18:
                hits += 1
                cursor = idx + 1
                break
    return hits / len(manual)


def _outcome_compare(manual: list[dict[str, Any]], auto: list[dict[str, Any]]) -> dict[str, Any]:
    if not manual:
        return {"action_recall": 1.0, "type_precision": 1.0, "matched": []}
    matched: list[dict[str, Any]] = []
    used: set[int] = set()
    type_hits = 0
    for m in manual:
        m_label = _clean_text(m.get("action_label") or m.get("label") or m.get("content") or "")
        best_idx = -1
        best_score = 0.0
        for idx, a in enumerate(auto):
            if idx in used:
                continue
            a_label = _clean_text(a.get("action_label") or a.get("label") or a.get("content") or "")
            score = _similar(m_label, a_label)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= 0.18:
            used.add(best_idx)
            a = auto[best_idx]
            same_type = str(m.get("outcome_type") or "") == str(a.get("outcome_type") or "")
            type_hits += 1 if same_type else 0
            matched.append({
                "manual_action": m_label,
                "auto_action": _clean_text(a.get("action_label") or a.get("label") or a.get("content") or ""),
                "similarity": round(best_score, 4),
                "manual_type": str(m.get("outcome_type") or ""),
                "auto_type": str(a.get("outcome_type") or ""),
                "type_match": same_type,
            })
    return {
        "action_recall": len(matched) / len(manual),
        "type_precision": type_hits / len(matched) if matched else 0.0,
        "matched": matched,
    }


def _label_recall(manual: list[str], auto: list[str]) -> float:
    if not manual:
        return 1.0
    if not auto:
        return 0.0
    return sum(1 for label in manual if _best_similarity(label, auto) >= 0.18) / len(manual)


def _set_recall(manual: set[str], auto: set[str]) -> float:
    if not manual:
        return 1.0
    return len(manual & auto) / len(manual)


def _best_similarity(value: str, candidates: list[str]) -> float:
    return max((_similar(value, item) for item in candidates if item), default=0.0)


def _similar(left: str, right: str) -> float:
    ltok = _tokens(left)
    rtok = _tokens(right)
    if not ltok or not rtok:
        return 0.0
    overlap = ltok & rtok
    base = len(overlap) / len(ltok | rtok)
    salient = {tok for tok in overlap if len(tok) >= 4 or tok in {"蓝屏", "黑屏", "闪退", "卡顿", "拍照", "拍摄", "内存", "驱动", "网卡", "相机", "dmp", "pte", "pfn"}}
    if salient:
        containment = len(overlap) / max(1, min(len(ltok), len(rtok)))
        return max(base, (base + containment) / 2)
    return base


def _tokens(text: str) -> set[str]:
    lowered = _clean_text(text).lower()
    tokens = set(_WORD.findall(lowered))
    cjk = _CJK.findall(lowered)
    tokens.update(cjk)
    for size in (2, 3, 4):
        for i in range(len(cjk) - size + 1):
            tokens.add("".join(cjk[i:i+size]))
    return {x for x in tokens if x.strip()}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("/", " ").split()).strip()


def _outcome_signature(item: dict[str, Any]) -> dict[str, str]:
    return {
        "action_label": _clean_text(item.get("action_label") or item.get("label") or item.get("content") or ""),
        "outcome_type": str(item.get("outcome_type") or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-root", default="data/kg/review_queue/manual_review_examples")
    parser.add_argument("--kg-root", default="data/kg")
    parser.add_argument("--episodes", default="")
    parser.add_argument("--import-root", default="")
    parser.add_argument("--hits-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = compare_manual_cases(
        manual_root=args.manual_root,
        kg_root=args.kg_root,
        episodes_path=args.episodes or None,
        import_root=args.import_root or None,
        hits_only=args.hits_only,
        limit=args.limit,
        deepseek=True if args.deepseek else False,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["summary"].get("ready_for_batch_candidates") else 1


if __name__ == "__main__":
    raise SystemExit(main())
