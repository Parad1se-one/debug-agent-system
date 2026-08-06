from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write import KnowledgeExtractionAgent
import debug_agent_system.agents.write.review_context as review_ctx
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2.compat import _canonicalize_family_label, _family_candidates
from debug_agent_system.knowledge_v2.builders import infer_required_info_slot

_WORD = re.compile(r"[A-Za-z0-9_.:-]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(slots=True)
class ReviewCase:
    case_id: str
    source_kind: str
    source_episode_id: str
    reference_file: str
    reference_payload: dict[str, Any]


DEFAULT_CASES: list[ReviewCase] = []


def _norm(text: Any) -> str:
    value = str(text or "").lower()
    return " ".join(_WORD.findall(value) + _CJK.findall(value))


def _episode_text(episode: dict[str, Any]) -> str:
    ext = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    parts: list[str] = []
    for key in ("symptom_raw", "conclusion", "key_conclusion"):
        parts.append(str(ext.get(key) or ""))
    parts.extend(str(x) for x in ext.get("debug_actions") or [])
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_context_messages"):
        for msg in episode.get(key) or []:
            if isinstance(msg, dict):
                parts.append(str(msg.get("text") or msg.get("content_summary") or ""))
    return " ".join(parts)


def _load_review_cases() -> list[ReviewCase]:
    cases: list[ReviewCase] = []
    gold_root = Path("data/annotations/goldcases/gold-v1")
    for name in (
        "goldcase-001.json", "goldcase-002.json", "goldcase-003.json", "goldcase-004.json",
        "goldcase-005.json", "goldcase-006.json", "goldcase-007.json", "goldcase-008.json",
        "goldcase-009.json", "goldcase-010.json",
    ):
        payload = json.loads((gold_root / name).read_text(encoding="utf-8"))
        cases.append(ReviewCase(
            case_id=str(payload.get("case_id") or name[:-5]),
            source_kind=str(payload.get("source_kind") or "gold_case"),
            source_episode_id=str(payload.get("source_episode_id") or ""),
            reference_file=str(gold_root / name),
            reference_payload=payload,
        ))
    manual_root = Path("data/kg/review_queue/manual_review_examples")
    for name in ("chat-rank-aa7f9f81327e.json", "chat-rank-b8f3c02dbdaf.json", "chat-rank-240b3ff8f1e9.json", "chat-rank-68b3b3d0da80.json"):
        payload = json.loads((manual_root / name).read_text(encoding="utf-8"))
        cases.append(ReviewCase(
            case_id=str(payload.get("sample_id") or name[:-5]),
            source_kind="manual_review",
            source_episode_id=str(payload.get("source_episode_id") or ""),
            reference_file=str(manual_root / name),
            reference_payload=payload,
        ))
    return cases


def _manual_example_to_reviewed(payload: dict[str, Any], source_file: str) -> dict[str, Any]:
    refined = payload.get("refined_merge_proposal") if isinstance(payload.get("refined_merge_proposal"), dict) else {}
    human = payload.get("human_correction") if isinstance(payload.get("human_correction"), dict) else {}
    nodes = [item for item in refined.get("nodes") or [] if isinstance(item, dict)]
    error = next((item for item in nodes if item.get("type") == "Error"), {})
    checks = [item for item in nodes if item.get("type") == "DiagnosticCheck"]
    solutions = [item for item in nodes if item.get("type") == "Solution"]
    evidence_findings = [item for item in payload.get("evidence_findings") or [] if isinstance(item, dict)]
    canonical_error_id = str(refined.get("canonical_error_id") or payload.get("manual_decision", {}).get("canonical_error_id") or "")
    family_label = _manual_family_label(canonical_error_id, error, payload)
    exact_reuse_allowed = bool(error and checks)
    gold_structure = _manual_gold_structure(payload, family_label, error, checks, solutions)
    if gold_structure.get("cases"):
        exact_reuse_allowed = True
    return {
        "case_id": str(payload.get("sample_id") or Path(source_file).stem),
        "source_episode_id": str(payload.get("source_episode_id") or ""),
        "family_label": family_label,
        "variant_label": str(error.get("label") or ""),
        "source_excerpt": [str(item.get("summary") or item.get("finding") or "") for item in evidence_findings[:6]],
        "evidence_anchor_map": {str(item.get("message_id") or f"m{idx+1}"): str(item.get("summary") or item.get("finding") or "") for idx, item in enumerate(evidence_findings[:12])},
        "gold": gold_structure,
        "review_type": "manual_review",
        "exact_reuse_allowed": exact_reuse_allowed,
        "source_file": source_file,
    }


def _manual_family_label(canonical_error_id: str, error: dict[str, Any], payload: dict[str, Any]) -> str:
    canonical_map = {
        "err:camera-capture-failure": "相机拍摄失败",
        "err:industrial-pc-freeze-black-screen": "工控机蓝屏",
    }
    if canonical_error_id in canonical_map:
        return canonical_map[canonical_error_id]
    return _canonicalize_family_label(
        str(error.get("label") or canonical_error_id or ""),
        str(error.get("subsystem") or ""),
        str(error.get("category") or ""),
        " ".join([
            str(error.get("label") or ""),
            str(error.get("symptom") or ""),
            str(payload.get("review_summary") or ""),
        ]),
    )


def _manual_solution_outcome_type(text: str) -> str:
    raw = str(text or "")
    if any(k in raw for k in ("无效", "验证失败", "未解决", "仍出现", "仍复发")):
        return "ineffective"
    if any(k in raw for k in ("短时正常", "一度未出现", "暂未复发", "临时恢复")):
        return "partial_temporary"
    if any(k in raw for k in ("用于", "抓取", "定位", "排查", "分析")) and not any(k in raw for k in ("最终", "解决")):
        return "diagnostic_method"
    if any(k in raw for k in ("需人工确认", "待验证", "pending", "需人工")):
        return "pending_validation"
    if any(k in raw for k in ("最终判断", "根因", "最终解决")):
        return "pending_validation"
    return "pending_validation"


def _manual_outcome_type(value: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "ineffective": "ineffective",
        "partial_temporary": "partial_temporary",
        "candidate_final_fix_high_cost": "pending_validation",
        "temporary_recovery": "partial_temporary",
        "partial_then_recurred": "recurred",
        "mitigation_observed": "mitigation_observed",
        "workaround": "mitigation_observed",
        "pending_validation": "pending_validation",
        "pending_rnd_investigation": "pending_validation",
        "cleared_not_root_cause": "context_not_root_cause",
        "temporary_then_recurred": "recurred",
        "mitigation_uncertain": "mitigation_observed",
        "case_verified_fix": "verified_fix",
        "mitigation_observed_then_recurred": "recurred",
        "recommended_pending_validation": "pending_validation",
        "context_not_root_cause": "context_not_root_cause",
        "diagnostic_method": "diagnostic_method",
    }
    return mapping.get(raw, _manual_solution_outcome_type(raw))


def _manual_gold_structure(payload: dict[str, Any], family_label: str, error: dict[str, Any], checks: list[dict[str, Any]], solutions: list[dict[str, Any]]) -> dict[str, Any]:
    human = payload.get("human_correction") if isinstance(payload.get("human_correction"), dict) else {}
    required_info = [{"slot": infer_required_info_slot(str(text)), "question": str(text)} for text in (error.get("required_info") or [])[:8]]
    if human.get("correct_modeling") == "split_episode_into_two_candidates":
        primary_variant = str(human.get("primary_error_label") or "")
        secondary_variant = str(human.get("secondary_error_label") or "")
        primary_family = "相机拍摄失败" if any(k in primary_variant for k in ("拍摄失败", "不拍照", "相机")) else _canonicalize_family_label(primary_variant, "", "", primary_variant)
        secondary_family = "工控机蓝屏" if any(k in secondary_variant for k in ("蓝屏", "igdkmdn64", "驱动")) else _canonicalize_family_label(secondary_variant, "", "", secondary_variant)
        primary_actions = [{"label": str(x)} for x in human.get("primary_check_nodes") or [] if str(x).strip()]
        secondary_actions = [{"label": str(x)} for x in human.get("secondary_check_nodes") or [] if str(x).strip()]
        raw_outcomes = [x for x in human.get("solution_or_outcome_nodes") or [] if isinstance(x, dict)]
        primary_outcomes = []
        secondary_outcomes = []
        for item in raw_outcomes:
            entry = {
                "action_label": str(item.get("label") or ""),
                "outcome_type": _manual_outcome_type(str(item.get("outcome") or "")),
                "summary": str(item.get("note") or ""),
            }
            label = str(item.get("label") or "")
            if "过滤驱动" in label or "拍照" in label:
                primary_outcomes.append(entry)
            else:
                secondary_outcomes.append(entry)
        primary_required = [
            {"slot": "ip_config", "question": "请提供相机网卡与非相机网卡的区分、过滤驱动勾选状态和网口截图。"},
            {"slot": "log_package", "question": "请提供拍摄失败时的诊断日志和报错截图。"},
        ]
        secondary_required = [
            {"slot": "dmp_package", "question": "请提供蓝屏对应的DMP/minidump文件。"},
            {"slot": "software_version", "question": "请提供Intel核显驱动版本、系统版本和近期驱动变更记录。"},
        ]
        return {
            "cases": [
                {
                    "family": {"label": primary_family},
                    "variant": {"label": primary_variant},
                    "actions": primary_actions,
                    "outcomes": primary_outcomes,
                    "required_info": primary_required,
                    "trace": {"recommended_action_labels": [x["label"] for x in primary_actions]},
                },
                {
                    "family": {"label": secondary_family},
                    "variant": {"label": secondary_variant},
                    "actions": secondary_actions,
                    "outcomes": secondary_outcomes,
                    "required_info": secondary_required,
                    "trace": {"recommended_action_labels": [x["label"] for x in secondary_actions]},
                },
            ]
        }

    if human:
        clean_variant = str(human.get("correct_error_label") or error.get("label") or "")
        clean_actions = [{"label": str(x)} for x in human.get("check_nodes") or [] if str(x).strip()]
        clean_outcomes = []
        for item in human.get("solution_or_outcome_nodes") or []:
            if not isinstance(item, dict):
                continue
            clean_outcomes.append({
                "action_label": str(item.get("label") or ""),
                "outcome_type": _manual_outcome_type(str(item.get("outcome") or "")),
                "summary": str(item.get("note") or ""),
            })
        return {
            "family": {"label": family_label},
            "variant": {"label": clean_variant},
            "actions": clean_actions,
            "outcomes": clean_outcomes,
            "required_info": required_info,
            "trace": {"recommended_action_labels": [x["label"] for x in clean_actions]},
        }

    return {
        "family": {"label": family_label},
        "variant": {"label": str(error.get("label") or "")},
        "actions": [{"label": str(item.get("label") or "")} for item in checks[:12]],
        "outcomes": [{"action_label": str(item.get("content") or ""), "outcome_type": _manual_solution_outcome_type(str(item.get("content") or ""))} for item in solutions[:12]],
        "required_info": required_info,
        "trace": {"recommended_action_labels": [str(item.get("label") or "") for item in checks[:12]]},
    }


def _load_reviewed_gold_examples() -> list[dict[str, Any]]:
    return review_ctx.load_reviewed_examples()


def _load_episode_index() -> dict[str, dict[str, Any]]:
    return review_ctx.load_episode_index()


def _manual_review_fallback_episode(case: ReviewCase) -> dict[str, Any]:
    return review_ctx.manual_review_fallback_episode(case.case_id, case.reference_payload)


def _gold_review_fallback_episode(case: ReviewCase) -> dict[str, Any]:
    return review_ctx.gold_review_fallback_episode(case.case_id, case.reference_payload)


def _load_sop_background() -> dict[str, Any]:
    return review_ctx.load_sop_seed_background()


def _score_family(episode_text: str, family_view: dict[str, Any]) -> float:
    return review_ctx.score_family(episode_text, family_view)


def _score_gold_example(episode_text: str, example: dict[str, Any], top_family_labels: set[str]) -> float:
    return review_ctx.score_reviewed_example(episode_text, example, top_family_labels)


def _reviewed_examples_for_episode(episode: dict[str, Any], examples: list[dict[str, Any]], top_families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return review_ctx.reviewed_examples_for_episode(episode, examples, top_families)


def _sop_background_for_episode(episode: dict[str, Any], sop: dict[str, Any], reviewed_examples: list[dict[str, Any]]) -> dict[str, Any]:
    return review_ctx.build_sop_background_for_episode(episode, sop, reviewed_examples)


def _inject_sop_background(episode: dict[str, Any], background: dict[str, Any]) -> dict[str, Any]:
    return review_ctx.inject_review_context(episode, background)


def _reference_summary(case: ReviewCase) -> dict[str, Any]:
    payload = case.reference_payload
    if case.source_kind == "manual_review":
        manual = payload.get("manual_decision") or {}
        refined = payload.get("refined_merge_proposal") or {}
        nodes = refined.get("nodes") or []
        if isinstance(refined.get("primary_candidate"), dict):
            nodes = refined["primary_candidate"].get("nodes") or []
        nodes = [x for x in nodes if isinstance(x, dict)]
        return {
            "type": "manual_review",
            "sample_id": payload.get("sample_id"),
            "target_error_id": manual.get("target_error_id"),
            "canonical_error_id": manual.get("canonical_error_id"),
            "reason_codes": manual.get("reason_codes") or [],
            "check_labels": [x.get("label") for x in nodes if x.get("type") == "DiagnosticCheck"][:12],
            "solution_summaries": [x.get("content") for x in nodes if x.get("type") == "Solution"][:12],
        }
    gold = payload.get("gold") if isinstance(payload.get("gold"), dict) else {}
    return {
        "type": "gold_case",
        "family": (gold.get("family") or {}).get("label"),
        "variant": (gold.get("variant") or {}).get("label"),
        "actions": [x.get("label") for x in gold.get("actions") or [] if isinstance(x, dict)],
        "outcomes": [
            {"action_label": x.get("action_label"), "outcome_type": x.get("outcome_type")}
            for x in gold.get("outcomes") or []
            if isinstance(x, dict)
        ],
        "required_info": [
            {"slot": x.get("slot"), "question": x.get("question")}
            for x in gold.get("required_info") or []
            if isinstance(x, dict)
        ],
        "trace": gold.get("trace") or {},
    }


def _episode_brief(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id") or "",
        "thread_id": episode.get("thread_id") or "",
        "completeness": episode.get("completeness") or "",
        "fault_description_messages": (episode.get("fault_description_messages") or [])[:8],
        "diagnostic_chain_messages": (episode.get("diagnostic_chain_messages") or [])[:16],
        "resolution_messages": (episode.get("resolution_messages") or [])[:8],
        "case_context_messages": (episode.get("case_context_messages") or [])[:24],
        "attachments": (episode.get("attachments") or [])[:24],
        "extracted": episode.get("extracted") or {},
    }


def _select_additional_cases(
    *,
    limit: int,
    episodes: dict[str, dict[str, Any]],
    reviewed_examples: list[dict[str, Any]],
    sop: dict[str, Any],
    excluded_episode_ids: set[str],
) -> list[ReviewCase]:
    scored: list[tuple[float, str, str]] = []
    for episode_id, episode in episodes.items():
        if not episode_id or episode_id in excluded_episode_ids:
            continue
        if not review_ctx.is_review_ready_episode(episode):
            continue
        fault_focus = review_ctx.primary_fault_text(episode)
        background = review_ctx.build_sop_background_for_episode(episode, sop, reviewed_examples)
        family_candidates = _family_candidates(fault_focus, "", "")
        if not family_candidates:
            continue
        top = background.get("top_family_background") or []
        top_label = str(family_candidates[0] or "")
        top_score = float(top[0].get("score") or 0.0) if top else 0.0
        if not top_label or top_label in {"算法/程序调优异常", "主程序/系统异常", "硬件/运控异常"}:
            continue
        richness = (
            top_score * 10
            + review_ctx.review_ready_episode_score(episode)
        )
        scored.append((richness, episode_id, top_label))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: list[ReviewCase] = []
    by_family: dict[str, int] = {}
    family_cap = max(2, math.ceil(limit / 10))
    for _, episode_id, family_label in scored:
        if len(chosen) >= limit:
            break
        if by_family.get(family_label, 0) >= family_cap:
            continue
        by_family[family_label] = by_family.get(family_label, 0) + 1
        chosen.append(ReviewCase(
            case_id=f"additional:{len(chosen)+1:02d}:{family_label}",
            source_kind="chat_case",
            source_episode_id=episode_id,
            reference_file="",
            reference_payload={},
        ))
    return chosen


def build_review_pack(
    *,
    out_json: str | Path = "data/results/w2_sop_background_ten_pack.json",
    case_mode: str = "default",
    limit: int = 10,
    episodes_json: str | Path = review_ctx.DEFAULT_EPISODES_JSON,
) -> dict[str, Any]:
    default_cases = _load_review_cases()
    episodes = review_ctx.load_episode_index(episodes_json)
    sop = review_ctx.load_sop_seed_background()
    reviewed_examples = review_ctx.load_reviewed_examples()
    if case_mode == "default":
        cases = default_cases
    elif case_mode == "additional10":
        excluded = {str(case.source_episode_id or "") for case in default_cases if str(case.source_episode_id or "")}
        cases = _select_additional_cases(
            limit=max(limit * 5, limit),
            episodes=episodes,
            reviewed_examples=reviewed_examples,
            sop=sop,
            excluded_episode_ids=excluded,
        )
    else:
        raise ValueError(f"unsupported_case_mode:{case_mode}")
    extractor = KnowledgeExtractionAgent(JsonKGStore("data/kg"), deepseek_enabled=bool(os.environ.get("DEBUG_AGENT_SYSTEM_W2_DEEPSEEK") == "1"), w2_mode="native_v2")
    details = []
    missing = []
    skipped_weak: list[dict[str, Any]] = []
    for case in cases:
        if case_mode == "additional10" and len(details) >= limit:
            break
        episode = episodes.get(case.source_episode_id)
        if not episode:
            episode = review_ctx.gold_review_fallback_episode(case.case_id, case.reference_payload) or review_ctx.manual_review_fallback_episode(case.case_id, case.reference_payload)
        if not episode:
            missing.append({"case_id": case.case_id, "source_episode_id": case.source_episode_id})
            continue
        background = review_ctx.build_sop_background_for_episode(episode, sop, reviewed_examples)
        enriched_episode = review_ctx.inject_review_context(episode, background, review_case_id=case.case_id)
        auto = extractor.extract(enriched_episode, w2_mode="native_v2")
        effective_schema_valid = bool(auto.get("production_schema_valid", auto.get("schema_valid")))
        if case_mode == "additional10":
            split_cases = ((auto.get("candidate_draft_v2") or {}).get("split_cases") or [])
            total_actions = sum(len(c.get("actions") or []) for c in split_cases if isinstance(c, dict))
            first = split_cases[0] if split_cases and isinstance(split_cases[0], dict) else {}
            first_family = str((first.get("family") or {}).get("label") or "")
            if (not effective_schema_valid) or (not split_cases) or (not first_family) or total_actions < 2:
                skipped_weak.append({
                    "case_id": case.case_id,
                    "source_episode_id": case.source_episode_id,
                    "schema_valid": effective_schema_valid,
                    "schema_issues": auto.get("production_schema_issues") or auto.get("schema_issues") or [],
                    "split_case_count": len(split_cases),
                    "first_family": first_family,
                    "total_actions": total_actions,
                })
                continue
        details.append({
            "case_id": case.case_id,
            "source_kind": case.source_kind,
            "source_episode_id": case.source_episode_id,
            "reference_file": case.reference_file,
            "reference_summary": _reference_summary(case),
            "episode_input_full": _episode_brief(episode),
            "sop_background": background,
            "w2_output": {
                "candidate_id": auto.get("candidate_id"),
                "schema_valid": effective_schema_valid,
                "schema_issues": auto.get("production_schema_issues") or auto.get("schema_issues") or [],
                "deepseek_used": bool((auto.get("observability") or {}).get("deepseek_used")),
                "deepseek_error": str((auto.get("observability") or {}).get("deepseek_error") or ""),
                "production_schema_valid": bool(auto.get("production_schema_valid", auto.get("schema_valid"))),
                "production_schema_issues": auto.get("production_schema_issues") or [],
                "case_understanding_card": auto.get("case_understanding_card") or {},
                "candidate_draft_v2": auto.get("candidate_draft_v2") or {},
                "candidate_draft_v2_normalized_bundle": auto.get("candidate_draft_v2_normalized_bundle") or {},
            },
        })
    pack = {
        "schema_version": "debug_agent_system.w2_sop_background_ten_pack.v1",
        "sop_background_root": sop.get("root") or "data/results/kg_v2_sop_seed_draft_manual.json",
        "case_mode": case_mode,
        "episodes_json": str(episodes_json),
        "count": len(details),
        "missing": missing,
        "skipped_weak": skipped_weak,
        "details": details,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="data/results/w2_sop_background_ten_pack.json")
    parser.add_argument("--case-mode", choices=["default", "additional10"], default="default")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--episodes-json", default=review_ctx.DEFAULT_EPISODES_JSON)
    args = parser.parse_args(argv)
    out = build_review_pack(out_json=args.out_json, case_mode=args.case_mode, limit=args.limit, episodes_json=args.episodes_json)
    print(json.dumps({"out_json": args.out_json, "case_mode": args.case_mode, "episodes_json": args.episodes_json, "count": out["count"], "missing": out["missing"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
