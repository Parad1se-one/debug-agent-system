"""KG v2 gold-case comparator and dry-run baseline runner."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write import KnowledgeExtractionAgent
from debug_agent_system.agents.write import review_context as review_ctx
from debug_agent_system.agents.write.w4_quality_gate import QualityGateAgent
from debug_agent_system.eval.write_side.gold_set import verify_gold_set
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2 import build_v2_bundle_from_legacy_candidate

_WORD = re.compile(r"[A-Za-z0-9_.:-]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(slots=True)
class GoldCase:
    case_id: str
    status: str
    payload: dict[str, Any]


def load_gold_cases(root: str | Path) -> list[GoldCase]:
    base = Path(root)
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    cases: list[GoldCase] = []
    for row in index.get("cases") or []:
        if not isinstance(row, dict):
            continue
        path = base / str(row.get("file") or "")
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(GoldCase(
            case_id=str(payload.get("case_id") or row.get("case_id") or path.stem),
            status=str(payload.get("status") or "draft"),
            payload=payload,
        ))
    return cases


def build_prompt_a_input(case: GoldCase) -> dict[str, Any]:
    payload = case.payload
    episode = payload.get("episode_input") if isinstance(payload.get("episode_input"), dict) else {}
    excerpt = [str(x) for x in payload.get("source_excerpt") or [] if str(x).strip()]
    source_case_draft = {
        "source_episode_id": payload.get("source_episode_id") or "",
        "source_thread_id": payload.get("source_thread_id") or "",
        "source_kind": payload.get("source_kind") or "",
        "source_excerpt": excerpt,
    }
    evidence_bundle = {
        "fault_description_messages": episode.get("fault_description_messages") or [],
        "diagnostic_chain_messages": episode.get("diagnostic_chain_messages") or [],
        "resolution_messages": episode.get("resolution_messages") or [],
        "evidence_message_ids": episode.get("evidence_message_ids") or [],
        "attachments": episode.get("attachments") or [],
        "evidence_anchor_map": payload.get("evidence_anchor_map") or {},
    }
    return {
        "schema_version": "kg_v2.prompt_a_input.v1",
        "case_id": case.case_id,
        "source_case_draft": source_case_draft,
        "evidence_bundle": evidence_bundle,
    }


def build_prompt_b_input(case: GoldCase) -> dict[str, Any]:
    gold = payload_gold(case)
    return {
        "schema_version": "kg_v2.prompt_b_input.v1",
        "case_id": case.case_id,
        "case_understanding_card": {
            "case_ref": case.case_id,
            "family_hypothesis": gold.get("family") or {},
            "variant_hypothesis": gold.get("variant") or {},
            "actions": [
                {
                    "action_ref": f"act_{idx}",
                    "label": item.get("label") or "",
                    "summary": item.get("summary") or "",
                    "action_role": item.get("action_role") or "",
                    "atomicity_ok": True,
                    "source_evidence_ids": item.get("evidence_anchor_ids") or [],
                }
                for idx, item in enumerate(gold.get("actions") or [], start=1)
                if isinstance(item, dict)
            ],
            "outcomes": [
                {
                    "action_ref": _action_ref_for_label(gold.get("actions") or [], item.get("action_label") or ""),
                    "outcome_type": item.get("outcome_type") or "",
                    "summary": item.get("summary") or "",
                    "why_not_other_types": "",
                    "source_evidence_ids": item.get("evidence_anchor_ids") or [],
                    "high_cost": bool(item.get("high_cost")),
                    "destructive": bool(item.get("destructive")),
                }
                for item in gold.get("outcomes") or []
                if isinstance(item, dict)
            ],
            "required_info": [
                {
                    "slot_hint": item.get("slot") or "",
                    "question": item.get("question") or "",
                    "why_required": item.get("why_required") or "",
                    "blocks": item.get("blocks") or [],
                    "source_evidence_ids": item.get("evidence_anchor_ids") or [],
                    "generic_risk": "low" if item.get("slot") != "other" else "high",
                }
                for item in gold.get("required_info") or []
                if isinstance(item, dict)
            ],
            "evidence_anchor_ids": list(dict.fromkeys(
                [*case.payload.get("evidence_anchor_map", {}).keys(), *[x for item in gold.get("required_info") or [] if isinstance(item, dict) for x in item.get("evidence_anchor_ids") or []]]
            )),
            "uncertainties": gold.get("uncertainties") or [],
        },
    }


def payload_gold(case: GoldCase) -> dict[str, Any]:
    gold = case.payload.get("gold")
    return gold if isinstance(gold, dict) else {}


def run_legacy_bridge_baseline(
    *,
    gold_root: str | Path,
    kg_root: str | Path = "data/kg",
    deepseek: bool | None = None,
    emit_prompt_inputs: bool = False,
    runner_mode: str = "legacy_bridge",
    with_w7_loo: bool = False,
) -> dict[str, Any]:
    cases = load_gold_cases(gold_root)
    integrity = verify_gold_set(gold_root) if (Path(gold_root) / "gold-v1.manifest.json").exists() else {}
    w2_mode = runner_mode if runner_mode in {"native_v2", "prompt_first"} else ("native_v2" if runner_mode == "compare" else "legacy_only")
    extractor = KnowledgeExtractionAgent(JsonKGStore(kg_root), deepseek_enabled=deepseek, w2_mode=w2_mode)
    quality_gate = QualityGateAgent()
    sop = review_ctx.load_sop_seed_background() if with_w7_loo else {}
    reviewed_examples = review_ctx.load_reviewed_examples(gold_root=gold_root) if with_w7_loo else []
    details: list[dict[str, Any]] = []
    for case in cases:
        episode = case.payload.get("episode_input") if isinstance(case.payload.get("episode_input"), dict) else {}
        w7_background: dict[str, Any] = {}
        if with_w7_loo:
            # A gold case may guide future cases, but it must never reveal its
            # own answer while being evaluated.
            case_source_episode_id = str(case.payload.get("source_episode_id") or "")
            loo_examples = [
                example for example in reviewed_examples
                if str(example.get("case_id") or "") != case.case_id
                and str(example.get("source_episode_id") or "") != case_source_episode_id
            ]
            w7_background = review_ctx.build_sop_background_for_episode(episode, sop, loo_examples)
            episode = review_ctx.inject_review_context(episode, w7_background, review_case_id=case.case_id)
        auto = extractor.extract(episode, w2_mode=w2_mode)
        bridge_bundle = build_v2_bundle_from_legacy_candidate(auto, episode)
        native_bundle = auto.get("candidate_draft_v2_normalized_bundle") if isinstance(auto.get("candidate_draft_v2_normalized_bundle"), dict) else {}
        if not native_bundle:
            native_bundle = bridge_bundle
        chosen_bundle = native_bundle if runner_mode in {"native_v2", "prompt_first", "compare"} else bridge_bundle
        detail = compare_gold_case(case, chosen_bundle)
        detail["legacy_candidate"] = {
            "candidate_id": auto.get("candidate_id") or "",
            "schema_valid": bool(auto.get("schema_valid")),
            "schema_issues": auto.get("schema_issues") or [],
            "deepseek_used": bool((auto.get("observability") or {}).get("deepseek_used")),
            "deepseek_error": str((auto.get("observability") or {}).get("deepseek_error") or ""),
        }
        detail["bridge_v2_bundle"] = {
            "schema_valid": bool(bridge_bundle.get("schema_valid")),
            "schema_issues": bridge_bundle.get("schema_issues") or [],
        }
        detail["native_v2_bundle"] = {
            "schema_valid": bool(native_bundle.get("schema_valid")),
            "schema_issues": native_bundle.get("schema_issues") or [],
        }
        detail["case_understanding_card"] = auto.get("case_understanding_card") or {}
        detail["case_understanding_extraction"] = auto.get("case_understanding_extraction") or {}
        detail["candidate_draft_v2"] = auto.get("candidate_draft_v2") or {}
        detail["w4_v2_gate"] = quality_gate.score_v2_bundle(chosen_bundle)
        detail["critical_errors"] = _critical_errors(case, chosen_bundle, detail["w4_v2_gate"])
        if runner_mode == "prompt_first" and str((detail["case_understanding_extraction"] or {}).get("case_understanding_source") or "") != "deepseek_prompt_a":
            detail["critical_errors"].append({
                "code": "prompt_first_fallback",
                "expected": "deepseek_prompt_a",
                "actual": str((detail["case_understanding_extraction"] or {}).get("case_understanding_source") or "missing"),
            })
        if runner_mode == "prompt_first" and not bool(auto.get("production_schema_valid")):
            detail["critical_errors"].append({
                "code": "prompt_first_production_schema_invalid",
                "expected": True,
                "actual": False,
            })
        detail["w7_leave_one_out"] = {
            "enabled": with_w7_loo,
            "selected_case_ids": [
                str(item.get("case_id") or "")
                for item in w7_background.get("reviewed_case_examples") or []
                if isinstance(item, dict)
            ],
        }
        if runner_mode == "compare":
            detail["legacy_bridge_compare"] = compare_gold_case(case, bridge_bundle)
            detail["native_v2_compare"] = compare_gold_case(case, native_bundle)
            detail["composite"] = detail["native_v2_compare"]["composite"]
        if emit_prompt_inputs:
            detail["prompt_a_input"] = build_prompt_a_input(case)
            detail["prompt_b_input"] = build_prompt_b_input(case)
        details.append(detail)
    return {
        "schema_version": "kg_v2.gold_compare.v1",
        "runner_mode": runner_mode,
        "gold_root": str(gold_root),
        "kg_root": str(kg_root),
        "deepseek_enabled": bool(deepseek),
        "gold_set_integrity": integrity,
        "stage_coverage": {
            "w1": "gold episode evidence projection; raw-history boundary scored separately",
            "w7": "leave-one-case-out" if with_w7_loo else "disabled",
            "w2": w2_mode,
            "w4": "v2 semantic quality gate",
        },
        "summary": summarise(details),
        "details": details,
    }


def _critical_errors(case: GoldCase, bundle: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return actionable, per-case regression failures rather than one score."""
    gold = payload_gold(case)
    objects = bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {}
    family = _first(objects.get("FaultFamily") or [])
    variant = _first(objects.get("FaultVariant") or [])
    actions = [item for item in objects.get("DiagnosticAction") or [] if isinstance(item, dict)]
    outcomes = [item for item in objects.get("ActionOutcome") or [] if isinstance(item, dict)]
    traces = [item for item in objects.get("DiagnosticTrace") or [] if isinstance(item, dict)]
    errors: list[dict[str, Any]] = []

    def add(code: str, expected: Any, actual: Any) -> None:
        errors.append({"code": code, "expected": expected, "actual": actual})

    if not _eq_text((gold.get("family") or {}).get("label"), family.get("label") or ""):
        add("family_mismatch", (gold.get("family") or {}).get("label") or "", family.get("label") or "")
    if not _semantic_label_match(
        (gold.get("variant") or {}).get("label"), variant.get("label") or "", kind="variant"
    ):
        add("variant_mismatch", (gold.get("variant") or {}).get("label") or "", variant.get("label") or "")
    expected_actions = [_norm_text(item.get("label") or "") for item in gold.get("actions") or [] if isinstance(item, dict)]
    actual_actions = [_norm_text(item.get("label") or "") for item in actions]
    missing_actions = [
        label for label in expected_actions
        if label and not any(_semantic_label_match(label, candidate, kind="action") for candidate in actual_actions)
    ]
    unexpected_actions = [
        label for label in actual_actions
        if label and not any(_semantic_label_match(candidate, label, kind="action") for candidate in expected_actions)
    ]
    if missing_actions:
        add("missing_actions", missing_actions, actual_actions)
    if unexpected_actions:
        add("unsupported_actions", expected_actions, unexpected_actions)

    action_by_id = {str(item.get("action_id") or ""): _norm_text(item.get("label") or "") for item in actions}
    first_trace = _first(traces)
    expected_recommended = [_norm_text(value) for value in (gold.get("trace") or {}).get("recommended_action_labels") or []]
    expected_actual = [_norm_text(value) for value in (gold.get("trace") or {}).get("actual_action_labels") or []]
    actual_recommended = [action_by_id.get(str(value), _norm_text(value)) for value in first_trace.get("recommended_action_ids") or []]
    actual_executed = [action_by_id.get(str(value), _norm_text(value)) for value in first_trace.get("actual_action_ids") or []]
    if not _semantic_sequence_equal(expected_recommended, actual_recommended):
        add("recommended_trace_mismatch", expected_recommended, actual_recommended)
    if not _semantic_sequence_equal(expected_actual, actual_executed):
        add("actual_trace_mismatch", expected_actual, actual_executed)
    if (
        not _semantic_sequence_equal(expected_recommended, expected_actual)
        and _semantic_sequence_equal(actual_recommended, actual_executed)
    ):
        add("recommended_actual_collapsed", {"recommended": expected_recommended, "actual": expected_actual}, actual_recommended)

    action_ids = set(action_by_id)
    outcome_action_ids = {str(item.get("action_id") or "") for item in outcomes}
    actions_without_outcome = sorted(action_by_id[action_id] for action_id in action_ids - outcome_action_ids)
    if actions_without_outcome:
        add("actions_without_outcome", "every action has an outcome", actions_without_outcome)
    ungrounded_outcomes = [
        str(item.get("outcome_id") or item.get("summary") or "") for item in outcomes
        if str(item.get("action_id") or "") not in action_ids or not (item.get("evidence_ids") or item.get("evidence_message_ids"))
    ]
    if ungrounded_outcomes:
        add("unbound_or_ungrounded_outcomes", [], ungrounded_outcomes)
    # Overall outcome accuracy is a graded metric, but promoting an action the
    # reviewer explicitly marked temporary/unverified to ``verified_fix`` is
    # a safety-critical semantic error and must fail the zero-critical gate.
    gold_outcomes = [item for item in gold.get("outcomes") or [] if isinstance(item, dict)]
    unsafe_fix_promotions: list[dict[str, str]] = []
    for outcome in outcomes:
        if str(outcome.get("outcome_type") or "") != "verified_fix":
            continue
        actual_label = action_by_id.get(str(outcome.get("action_id") or ""), "")
        ranked_expected = sorted(
            gold_outcomes,
            key=lambda item: _semantic_label_score(
                item.get("action_label") or "", actual_label, kind="action"
            ),
            reverse=True,
        )
        expected = ranked_expected[0] if ranked_expected and _semantic_label_match(
            ranked_expected[0].get("action_label") or "", actual_label, kind="action"
        ) else None
        if expected is None or str(expected.get("outcome_type") or "") == "verified_fix":
            continue
        unsafe_fix_promotions.append({
            "action": actual_label,
            "expected": str(expected.get("outcome_type") or ""),
            "actual": "verified_fix",
        })
    if unsafe_fix_promotions:
        add("temporary_or_unverified_promoted_to_verified_fix", [], unsafe_fix_promotions)
    if "kg_v2_unsubstantiated_verified_fix" in (gate.get("issues") or []):
        add("unsubstantiated_verified_fix", False, True)
    return errors


def compare_gold_case(case: GoldCase, bundle: dict[str, Any]) -> dict[str, Any]:
    gold = payload_gold(case)
    objects = bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {}
    family = _first(objects.get("FaultFamily") or [])
    variant = _first(objects.get("FaultVariant") or [])
    actions = [item for item in objects.get("DiagnosticAction") or [] if isinstance(item, dict)]
    outcomes = [item for item in objects.get("ActionOutcome") or [] if isinstance(item, dict)]
    required = [item for item in objects.get("RequiredInfoSpec") or [] if isinstance(item, dict)]
    traces = [item for item in objects.get("DiagnosticTrace") or [] if isinstance(item, dict)]

    family_match = _eq_text((gold.get("family") or {}).get("label"), family.get("label") if isinstance(family, dict) else "")
    variant_exact_match = _eq_text((gold.get("variant") or {}).get("label"), variant.get("label") if isinstance(variant, dict) else "")
    variant_match = _semantic_label_match(
        (gold.get("variant") or {}).get("label"),
        variant.get("label") if isinstance(variant, dict) else "",
        kind="variant",
    )
    gold_action_labels = [_norm_text(item.get("label") or "") for item in gold.get("actions") or [] if isinstance(item, dict)]
    auto_action_labels = [_norm_text(item.get("label") or "") for item in actions]
    action_overlap = _set_overlap(gold_action_labels, auto_action_labels)
    outcome_accuracy = _outcome_accuracy(gold.get("outcomes") or [], outcomes, actions)
    required_overlap = _required_info_overlap(gold.get("required_info") or [], required)
    trace_order_recall = _trace_order_recall((gold.get("trace") or {}).get("recommended_action_labels") or [], traces, actions)
    composite = round((int(family_match) + int(variant_match) + action_overlap["f1"] + outcome_accuracy + required_overlap["f1"] + trace_order_recall) / 6.0, 4)

    return {
        "case_id": case.case_id,
        "status": case.status,
        "source_kind": case.payload.get("source_kind") or "",
        "source_episode_id": case.payload.get("source_episode_id") or "",
        "family_match": family_match,
        "variant_match": variant_match,
        "variant_exact_match": variant_exact_match,
        "action_metrics": action_overlap,
        "outcome_type_acc": round(outcome_accuracy, 4),
        "required_info_metrics": required_overlap,
        "trace_order_recall": round(trace_order_recall, 4),
        "composite": composite,
        "passed_minimum_overlap": bool(family_match and variant_match and action_overlap["recall"] >= 0.5),
        "gold": {
            "family_label": (gold.get("family") or {}).get("label") or "",
            "variant_label": (gold.get("variant") or {}).get("label") or "",
            "action_labels": [item.get("label") or "" for item in gold.get("actions") or [] if isinstance(item, dict)],
            "required_slots": [item.get("slot") or "" for item in gold.get("required_info") or [] if isinstance(item, dict)],
        },
        "auto": {
            "family_label": family.get("label") if isinstance(family, dict) else "",
            "variant_label": variant.get("label") if isinstance(variant, dict) else "",
            "action_labels": [item.get("label") or "" for item in actions],
            "required_slots": [item.get("slot") or "" for item in required],
        },
    }


def summarise(details: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(details)
    if n == 0:
        return {"n": 0}
    error_counts: dict[str, int] = {}
    for detail in details:
        for error in detail.get("critical_errors") or []:
            code = str(error.get("code") or "unknown")
            error_counts[code] = error_counts.get(code, 0) + 1
    extraction_source_counts: dict[str, int] = {}
    for detail in details:
        source = str((detail.get("case_understanding_extraction") or {}).get("case_understanding_source") or "unknown")
        extraction_source_counts[source] = extraction_source_counts.get(source, 0) + 1
    return {
        "n": n,
        "family_exact_rate": round(sum(1 for d in details if d.get("family_match")) / n, 4),
        "variant_exact_rate": round(sum(1 for d in details if d.get("variant_exact_match")) / n, 4),
        "variant_semantic_rate": round(sum(1 for d in details if d.get("variant_match")) / n, 4),
        "action_recall_avg": round(sum(float((d.get("action_metrics") or {}).get("recall") or 0.0) for d in details) / n, 4),
        "action_precision_avg": round(sum(float((d.get("action_metrics") or {}).get("precision") or 0.0) for d in details) / n, 4),
        "outcome_type_acc_avg": round(sum(float(d.get("outcome_type_acc") or 0.0) for d in details) / n, 4),
        "required_info_recall_avg": round(sum(float((d.get("required_info_metrics") or {}).get("recall") or 0.0) for d in details) / n, 4),
        "trace_order_recall_avg": round(sum(float(d.get("trace_order_recall") or 0.0) for d in details) / n, 4),
        "composite_avg": round(sum(float(d.get("composite") or 0.0) for d in details) / n, 4),
        "passed_minimum_overlap": sum(1 for d in details if d.get("passed_minimum_overlap")),
        "critical_error_cases": sum(1 for d in details if d.get("critical_errors")),
        "critical_error_counts": dict(sorted(error_counts.items())),
        "case_understanding_source_counts": dict(sorted(extraction_source_counts.items())),
    }


def baseline_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# gold-v1 写侧基线报告",
        "",
        f"- gold set: `{(report.get('gold_set_integrity') or {}).get('gold_set_id') or 'unversioned'}`",
        f"- W7: `{(report.get('stage_coverage') or {}).get('w7') or 'disabled'}`",
        f"- family exact: `{summary.get('family_exact_rate', 0)}`",
        f"- variant exact/semantic: `{summary.get('variant_exact_rate', 0)}` / `{summary.get('variant_semantic_rate', 0)}`",
        f"- action recall: `{summary.get('action_recall_avg', 0)}`",
        f"- outcome accuracy: `{summary.get('outcome_type_acc_avg', 0)}`",
        f"- extraction sources: `{summary.get('case_understanding_source_counts') or {}}`",
        f"- critical error cases: `{summary.get('critical_error_cases', 0)}/{summary.get('n', 0)}`",
        "",
        "> W1 边界列当前使用 gold evidence projection；完整原始群聊的边界/跨窗回归单独运行，不能把本报告当成 W1 边界满分。",
        "",
        "| case | source | family | actions P/R | outcome | trace | W4 | critical errors |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for detail in report.get("details") or []:
        action = detail.get("action_metrics") or {}
        codes = ", ".join(str(item.get("code") or "") for item in detail.get("critical_errors") or []) or "—"
        lines.append(
            f"| {detail.get('case_id')} | {(detail.get('case_understanding_extraction') or {}).get('case_understanding_source') or 'unknown'} | "
            f"{'✓' if detail.get('family_match') else '✗'} | "
            f"{action.get('precision', 0):.2f}/{action.get('recall', 0):.2f} | "
            f"{float(detail.get('outcome_type_acc') or 0):.2f} | "
            f"{float(detail.get('trace_order_recall') or 0):.2f} | "
            f"{'pass' if (detail.get('w4_v2_gate') or {}).get('passed') else 'fail'} | {codes} |"
        )
    lines.extend(["", "## 错误计数", ""])
    for code, count in (summary.get("critical_error_counts") or {}).items():
        lines.append(f"- `{code}`: {count}")
    return "\n".join(lines) + "\n"


def _action_ref_for_label(actions: list[dict[str, Any]], label: str) -> str:
    want = _norm_text(label)
    for idx, item in enumerate(actions, start=1):
        if _norm_text(item.get("label") or "") == want:
            return f"act_{idx}"
    return ""


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    tokens = _WORD.findall(text)
    cjk = _CJK.findall(text)
    return " ".join(tokens + cjk)


def _eq_text(a: Any, b: Any) -> bool:
    return bool(_norm_text(a) and _norm_text(a) == _norm_text(b))


_ACTION_VERB_GROUPS = (
    ("收集", "导出", "提供", "上传", "记录", "抓取"),
    ("检查", "确认", "核对", "查询", "查看", "判断"),
    ("分析",),
    ("重启",),
    ("升级", "更新"),
    ("更换", "替换", "换"),
    ("拔插", "重插", "插回", "拔除", "拔掉"),
    ("修复",),
    ("设置", "调整", "切换", "还原", "恢复"),
    ("卸载", "清除", "删除", "清理"),
    ("观察", "监控"),
    ("验证", "复验", "测试"),
    ("进入",),
    ("等待",),
    ("规范",),
    ("点胶",),
    ("增加",),
)
_DOMAIN_TERMS = (
    "user.cfg.toml", "dmp", "pfn", "pte", "cpu", "nvidia", "driververifier", "wpr", "poolmon",
    "defender", "ddu", "bios", "sata", "ahci", "raid", "smart", "buddy", "usb", "pci", "2.5g",
    "诊断数据", "转储", "内存", "网卡", "驱动", "相机", "网线", "端口", "磁环", "残帧", "事件包",
    "接地", "关机", "主程序", "配置", "备份", "日志", "光源", "固态", "放电", "引导", "d盘",
    "u盘", "硬盘", "电源", "供电", "端子", "外设", "生产", "报错", "系统", "软件",
)


def _semantic_compact(value: Any) -> str:
    text = "".join(ch for ch in str(value or "").lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    for old, new in (
        ("内存条", "内存"), ("转储文件", "转储"), ("大恒相机固件", "大恒固件"),
        ("重新", ""), ("执行", ""), ("后仍无法进入系统", ""), ("无效", ""),
        ("首次诊断数据", "诊断数据"), ("完整诊断数据", "诊断数据"),
        ("系统文件修复", "系统修复"), ("金手指异物", "异物"),
    ):
        text = text.replace(old, new)
    return text


def _action_groups(value: Any) -> set[int]:
    text = _semantic_compact(value)
    return {index for index, aliases in enumerate(_ACTION_VERB_GROUPS) if any(alias in text for alias in aliases)}


def _domain_terms(value: Any) -> set[str]:
    text = _semantic_compact(value)
    return {term for term in _DOMAIN_TERMS if _semantic_compact(term) in text}


def _semantic_label_score(expected: Any, actual: Any, *, kind: str = "action") -> float:
    left = _semantic_compact(expected)
    right = _semantic_compact(actual)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    ratio = SequenceMatcher(None, left, right).ratio()
    left_terms, right_terms = _domain_terms(left), _domain_terms(right)
    term_union = left_terms | right_terms
    term_score = len(left_terms & right_terms) / len(term_union) if term_union else 0.0
    if kind == "variant":
        containment = min(len(left), len(right)) >= 6 and (left in right or right in left)
        return max(0.88 if containment else 0.0, 0.58 * ratio + 0.42 * term_score)
    left_groups, right_groups = _action_groups(left), _action_groups(right)
    if left_groups and right_groups and not (left_groups & right_groups):
        return 0.0
    verb_score = 1.0 if left_groups & right_groups else 0.0
    if len(left_groups & right_groups) >= 2:
        return max(0.75, ratio)
    containment = min(len(left), len(right)) >= 4 and (left in right or right in left)
    # A compound gold action can be represented by multiple atomic predicted
    # nodes; either node is supported when it shares the operation and target.
    containment_score = 0.9 if containment and (verb_score or term_score) else 0.0
    return max(containment_score, 0.5 * ratio + 0.25 * term_score + 0.25 * verb_score)


def _semantic_label_match(expected: Any, actual: Any, *, kind: str = "action") -> bool:
    threshold = 0.48 if kind == "variant" else 0.57
    return _semantic_label_score(expected, actual, kind=kind) >= threshold


def _semantic_sequence_recall(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 1.0
    cursor = 0
    hit = 0
    for want in expected:
        for index in range(cursor, len(actual)):
            if _semantic_label_match(want, actual[index], kind="action"):
                hit += 1
                cursor = index + 1
                break
    return hit / len(expected)


def _semantic_sequence_equal(expected: list[str], actual: list[str]) -> bool:
    if not expected and not actual:
        return True
    if not expected or not actual:
        return False
    expected_supported = all(any(_semantic_label_match(item, candidate, kind="action") for candidate in actual) for item in expected)
    actual_supported = all(any(_semantic_label_match(item, candidate, kind="action") for candidate in expected) for item in actual)
    # Critical trace errors concern membership and recommended-vs-actual
    # status.  Ordering remains visible in ``trace_order_recall`` as a
    # non-critical quality metric so a harmless equivalent order does not
    # obscure a real missing/executed action regression.
    return expected_supported and actual_supported


def _set_overlap(expected: list[str], actual: list[str]) -> dict[str, float]:
    e = [item for item in dict.fromkeys(expected) if item]
    a = [item for item in dict.fromkeys(actual) if item]
    if not e and not a:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not e or not a:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    expected_hit = sum(any(_semantic_label_match(item, candidate, kind="action") for candidate in a) for item in e)
    actual_hit = sum(any(_semantic_label_match(item, candidate, kind="action") for candidate in e) for item in a)
    precision = actual_hit / len(a)
    recall = expected_hit / len(e)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _outcome_accuracy(expected: list[dict[str, Any]], actual: list[dict[str, Any]], actions: list[dict[str, Any]]) -> float:
    gold_map = {
        _norm_text(item.get("action_label") or ""): str(item.get("outcome_type") or "")
        for item in expected
        if isinstance(item, dict) and _norm_text(item.get("action_label") or "")
    }
    if not gold_map:
        return 1.0
    action_labels_by_id = {
        str(item.get("action_id") or ""): _norm_text(item.get("label") or "")
        for item in actions
        if isinstance(item, dict) and item.get("action_id")
    }
    auto_map = {
        (action_labels_by_id.get(str(item.get("action_id") or "")) or _norm_text(item.get("action_label") or item.get("summary") or "")): str(item.get("outcome_type") or "")
        for item in actual
        if isinstance(item, dict)
    }
    matched = 0
    total = 0
    for action_label, outcome_type in gold_map.items():
        total += 1
        for auto_label, auto_type in auto_map.items():
            if _semantic_label_match(action_label, auto_label, kind="action") and auto_type == outcome_type:
                matched += 1
                break
    return matched / max(total, 1)


def _required_info_overlap(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, float]:
    expected_slots = [_norm_text(item.get("slot") or "") for item in expected if isinstance(item, dict)]
    actual_slots = [_norm_text(item.get("slot") or "") for item in actual if isinstance(item, dict)]
    return _set_overlap(expected_slots, actual_slots)


def _trace_order_recall(expected_labels: list[str], traces: list[dict[str, Any]], actions: list[dict[str, Any]]) -> float:
    expected = [_norm_text(x) for x in expected_labels if _norm_text(x)]
    if not expected:
        return 1.0
    action_labels_by_id = {
        _norm_text(item.get("action_id") or ""): _norm_text(item.get("label") or "")
        for item in actions
        if isinstance(item, dict) and (item.get("action_id") or item.get("label"))
    }
    auto: list[str] = []
    for trace in traces:
        for item in trace.get("recommended_action_ids") or []:
            norm = _norm_text(item)
            auto.append(action_labels_by_id.get(norm, norm))
    if not auto:
        return 0.0
    hit = 0
    idx = 0
    for item in auto:
        if idx < len(expected) and _semantic_label_match(expected[idx], item, kind="action"):
            hit += 1
            idx += 1
    return hit / len(expected)


def _first(items: list[dict[str, Any]]) -> dict[str, Any]:
    return items[0] if items else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kg-v2-gold-compare")
    parser.add_argument("--gold-root", default="data/annotations/goldcases/gold-v1")
    parser.add_argument("--kg-root", default="data/kg")
    parser.add_argument("--runner-mode", choices=["legacy_bridge", "native_v2", "prompt_first", "compare"], default="legacy_bridge")
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--emit-prompt-inputs", action="store_true")
    parser.add_argument("--with-w7-loo", action="store_true", help="evaluate W7 context without exposing the current gold case")
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    out = run_legacy_bridge_baseline(
        gold_root=args.gold_root,
        kg_root=args.kg_root,
        deepseek=True if args.deepseek else False,
        emit_prompt_inputs=args.emit_prompt_inputs,
        runner_mode=args.runner_mode,
        with_w7_loo=args.with_w7_loo,
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        path = Path(args.md_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(baseline_markdown(out), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
