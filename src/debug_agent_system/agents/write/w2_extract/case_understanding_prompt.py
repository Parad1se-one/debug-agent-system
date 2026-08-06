"""Prompt-first W2 case-understanding contract.

This module deliberately contains no fault-specific lexical routing.  DeepSeek
does the semantic interpretation; local code only supplies the ontology,
current-case evidence boundary, and generic safety invariants.
"""

from __future__ import annotations

import re
from typing import Any

from debug_agent_system.knowledge_v2.contracts import (
    ACTION_ROLES,
    APPROVED_FAMILY_LABELS,
    INTERNAL_REQUIRED_INFO_SLOTS,
    OUTCOME_TYPES,
)
from debug_agent_system.knowledge_v2.validator import validate_case_understanding_card


PROMPT_VERSION = "w2.case_understanding.deepseek.v1"

CATEGORIES = ("系统与软件异常", "硬件与运控", "算法与程序调优")
EXECUTION_STATUSES = ("actual", "recommended")
HYPOTHESIS_STATES = ("proposed", "supported", "revised", "rejected", "final")
CAUSAL_ROLES = ("candidate", "root", "coexisting", "secondary")


SYSTEM_PROMPT = """\
你是工业现场故障群聊的写侧案例理解器。你的输出不是解决建议，而是对当前证据中已经发生的事实、明确提出的建议和诊断状态演化进行结构化标注。

证据边界：
1. 只有 current_episode_messages 和 promoted_case_evidence 中带 message_id 的内容可以成为本轮事实证据。
2. alignment_examples 只能学习命名风格、动作粒度、字段判定标准；禁止复制其中的案例事实、动作、结果、根因或 evidence id。
3. 每个 case、action、outcome、required_info 和 hypothesis 都必须引用当前证据 message_id。不得编造 message_id。

Trace 与 Case：
1. 一个输入中存在互不依赖的故障现象、设备或排查链时拆成多个 case；时间不连续但设备、故障和复发链一致时保持同一 case。
2. 日报或汇总消息按语义片段判断，不要把协调、培训、排期、致谢和状态播报当成故障动作。
3. family 是稳定的故障现象类别，variant 是本次案例的区分条件；不要把产品名、模块名或动作写成 family。
4. family_ontology 是当前活动 KG 的规范 family 名称；语义匹配时优先复用。确实没有合适类别时可以提出新 family，但必须在 uncertainties 中标明 new_family_candidate。

Action 与 Outcome：
1. action 必须原子化，一条 action 只包含一个可执行或可观察步骤。
2. actual 表示证据明确说明已经执行；recommended 表示仅建议、计划、追问或条件性方案。不得凭语气推断已执行。
3. 每个 action 必须恰好对应至少一个 outcome。推荐动作通常是 pending_validation；诊断/收集动作可为 diagnostic_method 或 context_not_root_cause。
4. ineffective、partial_temporary、mitigation_observed、recurred 和 verified_fix 必须绑定明确的结果证据。
5. 只有证据同时支持问题恢复以及稳定性验证、未复发或恢复生产时，才允许 verified_fix。短时恢复、重启后正常、仍需观察均不是 verified_fix。

诊断演化：
1. 后续结论可以把早期假设标成 revised、rejected 或 final，但不得删除早期假设。
2. root 是最终主根因；coexisting 是并存但非主因；secondary 是由主故障导致的次生问题；证据不足时使用 candidate。
3. 不确定就写入 uncertainties，不要补全聊天中不存在的事实。

严格按工具 schema 输出，不输出解释文字。"""


def tool_schema() -> dict[str, Any]:
    """Return the strict DeepSeek tool schema for Prompt A."""

    evidence_ids = {"type": "array", "items": {"type": "string"}}
    strings = {"type": "array", "items": {"type": "string"}}
    family = _strict_object({
        "label": {"type": "string"},
        "summary": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "subsystem": {"type": "string"},
        "scenario": {"type": "string"},
        "why_family_not_variant": {"type": "string"},
        "confidence": {"type": "number"},
    })
    variant = _strict_object({
        "label": {"type": "string"},
        "summary": {"type": "string"},
        "distinguishing_conditions": strings,
        "confidence": {"type": "number"},
    })
    action = _strict_object({
        "action_ref": {"type": "string"},
        "label": {"type": "string"},
        "summary": {"type": "string"},
        "action_role": {"type": "string", "enum": sorted(ACTION_ROLES)},
        "execution_status": {"type": "string", "enum": list(EXECUTION_STATUSES)},
        "atomicity_ok": {"type": "boolean"},
        "source_evidence_ids": evidence_ids,
        "high_cost": {"type": "boolean"},
        "destructive": {"type": "boolean"},
    })
    outcome = _strict_object({
        "action_ref": {"type": "string"},
        "outcome_type": {"type": "string", "enum": sorted(OUTCOME_TYPES)},
        "summary": {"type": "string"},
        "why_not_other_types": {"type": "string"},
        "source_evidence_ids": evidence_ids,
        "high_cost": {"type": "boolean"},
        "destructive": {"type": "boolean"},
        "root_cause_summary": {"type": "string"},
    })
    required_info = _strict_object({
        "slot_hint": {"type": "string", "enum": sorted(INTERNAL_REQUIRED_INFO_SLOTS)},
        "question": {"type": "string"},
        "why_required": {"type": "string"},
        "blocks": strings,
        "source_evidence_ids": evidence_ids,
        "generic_risk": {"type": "string", "enum": ["low", "medium", "high"]},
    })
    hypothesis = _strict_object({
        "order": {"type": "integer"},
        "state": {"type": "string", "enum": list(HYPOTHESIS_STATES)},
        "causal_role": {"type": "string", "enum": list(CAUSAL_ROLES)},
        "summary": {"type": "string"},
        "source_evidence_ids": evidence_ids,
    })
    case = _strict_object({
        "case_ref": {"type": "string"},
        "candidate_scope": {"type": "string", "enum": ["fault_only", "fault_execution"]},
        "family_hypothesis": family,
        "variant_hypothesis": variant,
        "symptom_summary": {"type": "string"},
        "evidence_anchor_ids": evidence_ids,
        "actions": {"type": "array", "items": action},
        "outcomes": {"type": "array", "items": outcome},
        "required_info": {"type": "array", "items": required_info},
        "hypothesis_timeline": {"type": "array", "items": hypothesis},
        "uncertainties": strings,
    })
    return {
        "type": "function",
        "function": {
            "name": "extract_case_understanding_card",
            "description": "Extract evidence-grounded fault cases from one chat episode.",
            "strict": True,
            "parameters": _strict_object({
                "split_required": {"type": "boolean"},
                "split_reason": {"type": "string"},
                "cases": {"type": "array", "items": case},
                "global_uncertainties": strings,
            }),
        },
    }


def build_prompt_input(semantics: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded, evidence-first prompt payload from W1/W2 semantics."""

    episode = semantics.get("episode") if isinstance(semantics.get("episode"), dict) else {}
    allowed_ids = {str(value) for value in semantics.get("evidence_ids") or [] if str(value)}
    current_messages: list[dict[str, Any]] = []
    seen: set[str] = set()
    sender_aliases: dict[str, str] = {}
    for role, key in (
        ("fault", "fault_description_messages"),
        ("diagnostic", "diagnostic_chain_messages"),
        ("resolution", "resolution_messages"),
    ):
        for message in episode.get(key) or []:
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            if not message_id or message_id not in allowed_ids or message_id in seen:
                continue
            seen.add(message_id)
            current_messages.append({
                "message_id": message_id,
                "time": message.get("create_time") or message.get("time") or "",
                "sender": _sender_alias(message.get("sender"), sender_aliases),
                "w1_role": role,
                "text": _redact_prompt_text(message.get("text") or message.get("content_summary") or "")[:1200],
            })
    promoted: list[dict[str, Any]] = []
    for message in episode.get("case_evidence_messages") or []:
        if not isinstance(message, dict):
            continue
        message_id = str(message.get("message_id") or message.get("source_message_id") or "")
        if not message_id or message_id not in allowed_ids or message_id in seen:
            continue
        seen.add(message_id)
        promoted.append({
            "message_id": message_id,
            "time": message.get("create_time") or message.get("time") or "",
            "sender": _sender_alias(message.get("sender"), sender_aliases),
            "text": _redact_prompt_text(message.get("text") or message.get("content_summary") or "")[:1200],
            "promotion_reason": str(message.get("promotion_reason") or ""),
        })
    review_context = semantics.get("review_context") or semantics.get("sop_background") or {}
    examples = []
    if isinstance(review_context, dict):
        for example in review_context.get("reviewed_case_examples") or []:
            if not isinstance(example, dict) or example.get("exact_source_match"):
                continue
            examples.append(_alignment_style_example(example))
            if len(examples) >= 4:
                break
    return {
        "prompt_version": PROMPT_VERSION,
        "source_episode_id": str(semantics.get("source_episode_id") or ""),
        "source_thread_id": str(semantics.get("source_thread_id") or ""),
        "current_episode_messages": current_messages,
        "promoted_case_evidence": promoted,
        "allowed_evidence_ids": sorted(allowed_ids),
        "family_ontology": sorted(APPROVED_FAMILY_LABELS),
        "w1_hints": {
            "symptom": _redact_prompt_text(semantics.get("symptom_raw") or "")[:500],
            "actions": [_redact_prompt_text(value)[:300] for value in semantics.get("debug_actions") or []][:30],
            "conclusion": _redact_prompt_text(semantics.get("conclusion") or "")[:500],
        },
        "alignment_examples": examples,
    }


def _alignment_style_example(example: dict[str, Any]) -> dict[str, Any]:
    """Remove raw source/provenance while retaining annotation style.

    Few-shot input should demonstrate ontology and field granularity, not act
    as a second evidence source for the current case.
    """

    gold = example.get("gold_structure") if isinstance(example.get("gold_structure"), dict) else {}
    raw_cases = gold.get("cases") if isinstance(gold.get("cases"), list) else [gold]
    cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        family = raw_case.get("family") if isinstance(raw_case.get("family"), dict) else {}
        variant = raw_case.get("variant") if isinstance(raw_case.get("variant"), dict) else {}
        cases.append({
            "family": {
                "label": _redact_prompt_text(family.get("label") or ""),
                "category": _redact_prompt_text(family.get("category") or ""),
                "subsystem": _redact_prompt_text(family.get("subsystem") or ""),
            },
            "variant": {"label": _redact_prompt_text(variant.get("label") or "")},
            "actions": [
                {
                    "label": _redact_prompt_text(item.get("label") or item.get("action_label") or ""),
                    "action_role": str(item.get("action_role") or ""),
                    "execution_status": str(item.get("execution_status") or ""),
                }
                for item in raw_case.get("actions") or []
                if isinstance(item, dict)
            ],
            "outcomes": [
                {
                    "action_label": _redact_prompt_text(item.get("action_label") or item.get("label") or ""),
                    "outcome_type": str(item.get("outcome_type") or ""),
                }
                for item in raw_case.get("outcomes") or []
                if isinstance(item, dict)
            ],
            "required_info_slots": [
                str(item.get("slot") or item.get("slot_hint") or "")
                for item in raw_case.get("required_info") or []
                if isinstance(item, dict)
            ],
        })
    return {
        "example_ref": "alignment_example",
        "review_type": str(example.get("review_type") or ""),
        "annotation_style": cases,
    }


def _sender_alias(value: Any, aliases: dict[str, str]) -> str:
    sender = str(value or "").strip()
    if not sender:
        return ""
    if sender not in aliases:
        aliases[sender] = f"participant_{len(aliases) + 1}"
    return aliases[sender]


def _redact_prompt_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"@[\w\-·（）()\u4e00-\u9fff]+", "@participant", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "<phone>", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<email>", text)
    return text


def normalize_card(
    raw: dict[str, Any],
    semantics: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Wrap a model result and enforce generic evidence/safety constraints.

    Returns ``(card, blocking_issues, corrections)``.  Corrections are safe
    monotonic downgrades (for example verified_fix -> mitigation_observed),
    never new domain facts.
    """

    allowed_map = evidence_anchor_map(semantics)
    allowed = set(allowed_map)
    blocking: list[str] = []
    corrections: list[str] = []
    cases: list[dict[str, Any]] = []
    raw_cases = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(raw_cases, list):
        raw_cases = []
        blocking.append("prompt_cases_not_list")
    for case_index, source_case in enumerate(raw_cases):
        if not isinstance(source_case, dict):
            blocking.append(f"cases[{case_index}]:not_object")
            continue
        case = dict(source_case)
        case["case_ref"] = str(case.get("case_ref") or f"case_{case_index + 1}")
        case["evidence_anchor_ids"] = _grounded_ids(
            case.get("evidence_anchor_ids"), allowed, blocking, f"cases[{case_index}].evidence_anchor_ids"
        )
        actions: list[dict[str, Any]] = []
        action_refs: set[str] = set()
        for action_index, source_action in enumerate(case.get("actions") or []):
            if not isinstance(source_action, dict):
                blocking.append(f"cases[{case_index}].actions[{action_index}]:not_object")
                continue
            action = dict(source_action)
            ref = str(action.get("action_ref") or f"act_{action_index + 1}")
            if ref in action_refs:
                blocking.append(f"cases[{case_index}]:duplicate_action_ref:{ref}")
                continue
            action_refs.add(ref)
            action["action_ref"] = ref
            action["execution_status"] = str(action.get("execution_status") or "recommended")
            action["source_evidence_ids"] = _grounded_ids(
                action.get("source_evidence_ids"), allowed, blocking,
                f"cases[{case_index}].actions[{action_index}].source_evidence_ids",
            )
            if not action["source_evidence_ids"]:
                blocking.append(f"cases[{case_index}].actions[{action_index}]:missing_current_evidence")
            # This field is derived locally from the evidence boundary; model
            # output cannot promote neighbouring W7 context into direct proof.
            action["evidence_scope"] = _action_evidence_scope(
                action["source_evidence_ids"], semantics
            )
            actions.append(action)
        case["actions"] = actions

        outcomes: list[dict[str, Any]] = []
        outcome_refs: set[str] = set()
        action_by_ref = {str(action.get("action_ref") or ""): action for action in actions}
        for outcome_index, source_outcome in enumerate(case.get("outcomes") or []):
            if not isinstance(source_outcome, dict):
                blocking.append(f"cases[{case_index}].outcomes[{outcome_index}]:not_object")
                continue
            outcome = dict(source_outcome)
            ref = str(outcome.get("action_ref") or "")
            if ref not in action_by_ref:
                blocking.append(f"cases[{case_index}].outcomes[{outcome_index}]:unknown_action_ref:{ref}")
                continue
            outcome_refs.add(ref)
            outcome["source_evidence_ids"] = _grounded_ids(
                outcome.get("source_evidence_ids"), allowed, blocking,
                f"cases[{case_index}].outcomes[{outcome_index}].source_evidence_ids",
            )
            if not outcome["source_evidence_ids"]:
                blocking.append(f"cases[{case_index}].outcomes[{outcome_index}]:missing_current_evidence")
            _downgrade_unsafe_verified_fix(outcome, action_by_ref[ref], allowed_map, corrections)
            outcomes.append(outcome)
        for ref in sorted(action_refs - outcome_refs):
            blocking.append(f"cases[{case_index}]:action_without_outcome:{ref}")
        case["outcomes"] = outcomes

        required_info = []
        for req_index, source_req in enumerate(case.get("required_info") or []):
            if not isinstance(source_req, dict):
                blocking.append(f"cases[{case_index}].required_info[{req_index}]:not_object")
                continue
            req = dict(source_req)
            req["source_evidence_ids"] = _grounded_ids(
                req.get("source_evidence_ids"), allowed, blocking,
                f"cases[{case_index}].required_info[{req_index}].source_evidence_ids",
            )
            if not req["source_evidence_ids"]:
                blocking.append(f"cases[{case_index}].required_info[{req_index}]:missing_current_evidence")
            required_info.append(req)
        case["required_info"] = required_info

        timeline = []
        for hypothesis_index, source_hypothesis in enumerate(case.get("hypothesis_timeline") or []):
            if not isinstance(source_hypothesis, dict):
                blocking.append(f"cases[{case_index}].hypothesis_timeline[{hypothesis_index}]:not_object")
                continue
            hypothesis = dict(source_hypothesis)
            hypothesis["source_evidence_ids"] = _grounded_ids(
                hypothesis.get("source_evidence_ids"), allowed, blocking,
                f"cases[{case_index}].hypothesis_timeline[{hypothesis_index}].source_evidence_ids",
            )
            if not hypothesis["source_evidence_ids"]:
                blocking.append(f"cases[{case_index}].hypothesis_timeline[{hypothesis_index}]:missing_current_evidence")
            timeline.append(hypothesis)
        case["hypothesis_timeline"] = timeline
        cases.append(case)

    card = {
        "schema_version": "kg_v2.case_understanding.v1",
        "source_episode_id": str(semantics.get("source_episode_id") or ""),
        "source_thread_id": str(semantics.get("source_thread_id") or ""),
        "case_count": len(cases),
        "split_required": bool(raw.get("split_required")) if isinstance(raw, dict) else False,
        "split_reason": str(raw.get("split_reason") or "") if isinstance(raw, dict) else "",
        "cases": cases,
        "evidence_anchor_map": allowed_map,
        "global_uncertainties": [str(value) for value in (raw.get("global_uncertainties") or []) if str(value)] if isinstance(raw, dict) else [],
        "prompt_version": PROMPT_VERSION,
        "extraction_source": "deepseek_prompt_a",
    }
    if card["split_required"] != (len(cases) > 1):
        blocking.append("split_flag_case_count_mismatch")
    blocking.extend(validate_case_understanding_card(card))
    card["schema_issues"] = sorted(set(blocking))
    card["schema_valid"] = not card["schema_issues"]
    return card, card["schema_issues"], corrections


def _action_evidence_scope(evidence_ids: list[str], semantics: dict[str, Any]) -> str:
    selected = {str(value) for value in evidence_ids if str(value)}
    roles = {
        str(item.get("source_role") or "")
        for item in semantics.get("sentence_roles") or []
        if isinstance(item, dict)
        and selected & {str(value) for value in item.get("evidence_message_ids") or [] if str(value)}
    }
    direct = bool(roles & {"current_fault", "current_diagnostic", "current_resolution"})
    promoted = "w7_promoted" in roles
    if direct and promoted:
        return "mixed_current_and_promoted"
    if direct:
        return "current_episode_direct"
    if promoted:
        return "w7_promoted_only"
    return "legacy_unspecified"


def evidence_anchor_map(semantics: dict[str, Any]) -> dict[str, str]:
    """Return only current-case, whitelisted message evidence."""

    episode = semantics.get("episode") if isinstance(semantics.get("episode"), dict) else {}
    allowed = {str(value) for value in semantics.get("evidence_ids") or [] if str(value)}
    out: dict[str, str] = {}
    for key in (
        "fault_description_messages",
        "diagnostic_chain_messages",
        "resolution_messages",
        "case_evidence_messages",
    ):
        for message in episode.get(key) or []:
            if not isinstance(message, dict):
                continue
            message_id = str(message.get("message_id") or message.get("source_message_id") or "")
            text = str(message.get("text") or message.get("content_summary") or "").strip()
            if not text and message.get("attachment_names"):
                text = "[attachments] " + ", ".join(
                    str(value) for value in message.get("attachment_names") or [] if str(value)
                )
            if message_id in allowed and text and message_id not in out:
                out[message_id] = text[:1200]
    return out


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _grounded_ids(
    values: Any,
    allowed: set[str],
    issues: list[str],
    path: str,
) -> list[str]:
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        evidence_id = str(value or "")
        if not evidence_id:
            continue
        if evidence_id not in allowed:
            issues.append(f"{path}:unknown_evidence_id:{evidence_id}")
            continue
        if evidence_id not in result:
            result.append(evidence_id)
    return result


def _downgrade_unsafe_verified_fix(
    outcome: dict[str, Any],
    action: dict[str, Any],
    evidence_map: dict[str, str],
    corrections: list[str],
) -> None:
    if str(outcome.get("outcome_type") or "") != "verified_fix":
        return
    ref = str(action.get("action_ref") or "")
    if str(action.get("execution_status") or "") != "actual":
        outcome["outcome_type"] = "pending_validation"
        outcome["why_not_other_types"] = "recommended_action_cannot_be_verified_fix"
        corrections.append(f"{ref}:verified_fix_to_pending_validation:recommended")
        return
    evidence_ids = list(dict.fromkeys([
        *(action.get("source_evidence_ids") or []),
        *(outcome.get("source_evidence_ids") or []),
    ]))
    text = " ".join(evidence_map.get(str(evidence_id), "") for evidence_id in evidence_ids)
    resolution_markers = (
        "已解决", "解决了", "已恢复", "恢复正常", "正常启动", "验证正常",
        "测试正常", "运行正常", "恢复生产", "resolved", "fixed",
    )
    durable_markers = (
        "恢复生产", "未再出现", "不再出现", "未复发", "没有复发", "持续稳定",
        "观察后", "至今正常", "后续正常", "多日", "数日", "小时未", "小时无",
    )
    temporary_markers = (
        "临时恢复", "暂时恢复", "短时恢复", "短暂恢复", "一度恢复",
        "仍需观察", "待观察", "可能复发", "复发风险", "仅短期", "短期可用",
    )
    # A durable clause can follow the concise engineering expression
    # ``处理后恢复`` without the adverbs used by the original marker list.
    # Accept that generic form only when the same evidence does not explicitly
    # qualify the recovery as temporary or recurrence-prone.
    has_temporary_qualifier = any(marker in text for marker in temporary_markers)
    has_observed_recovery = any(marker in text for marker in resolution_markers) or "恢复" in text
    if has_observed_recovery and has_temporary_qualifier:
        outcome["outcome_type"] = "partial_temporary"
        outcome["why_not_other_types"] = "recovery_is_explicitly_temporary_or_recurrence_prone"
        corrections.append(f"{ref}:verified_fix_to_partial_temporary:temporary_recovery_evidence")
        return
    has_resolution = has_observed_recovery
    has_durable = any(marker in text for marker in durable_markers)
    if has_resolution and has_durable:
        return
    outcome["outcome_type"] = "mitigation_observed" if has_resolution else "pending_validation"
    outcome["why_not_other_types"] = "verified_fix_requires_recovery_and_durable_validation"
    corrections.append(f"{ref}:verified_fix_to_{outcome['outcome_type']}:insufficient_durable_evidence")
