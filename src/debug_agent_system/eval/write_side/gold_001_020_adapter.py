"""Read-only, auditable adapter for the three Goldcase 001--020 schemas.

The source annotations are frozen review artefacts.  This module never writes
them; it projects all three schemas into one evaluation contract while keeping
the raw labels and source hashes needed to audit every normalization decision.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_ROOT = Path("data/annotations/goldcases")
MESSAGE_ID_RE = re.compile(r"om_x[0-9a-z]{24,}", re.IGNORECASE)
JIRA_ID_RE = re.compile(r"(?:jira:)?(?:SMTAOITS|TEST)-\d+", re.IGNORECASE)
TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]*$")
BACKTICK_TOKEN_RE = re.compile(r"`([a-z][a-z0-9_]*)`")
SYNTHETIC_MESSAGE_ID_RE = re.compile(r"m\d+")

CANONICAL_OUTCOMES = {
    "diagnostic_method",
    "ineffective",
    "execution_failed",
    "partial_temporary",
    "mitigation_observed",
    "pending_validation",
    "verified_fix",
    "context_not_root_cause",
}

_OUTCOME_MAP = {
    "verified": "verified_fix",
    "verified_fix": "verified_fix",
    "targeted_fix_verified": "verified_fix",
    "fix_verified": "verified_fix",
    "no_effect": "ineffective",
    "ineffective": "ineffective",
    "still_failing": "ineffective",
    "failed": "ineffective",
    "execution_failed": "execution_failed",
    "update_failed": "execution_failed",
    "installation_failed": "execution_failed",
    "partial_temporary": "partial_temporary",
    "temporary_recovery": "partial_temporary",
    "immediate_temporary_recovery": "partial_temporary",
    "temporary_mitigation_then_recurred": "partial_temporary",
    "short_term_recovery": "partial_temporary",
    "mitigation_observed": "mitigation_observed",
    "production_observed": "mitigation_observed",
    "recovery_observed": "mitigation_observed",
    "diagnostic_method": "diagnostic_method",
    "diagnostic_evidence_available": "diagnostic_method",
    "direct_log_evidence": "diagnostic_method",
    "hypothesis_generated": "diagnostic_method",
    "reported_symptom_reclassified": "diagnostic_method",
    "evidence_supports_hypothesis": "diagnostic_method",
    "context_not_root_cause": "context_not_root_cause",
    "not_root_cause": "context_not_root_cause",
}

_PENDING_TOKENS = (
    "pending", "missing", "unknown", "unverified", "not_independently",
    "not_evaluated", "recommended", "metadata_only", "payload_missing",
    "not_recorded", "inconclusive", "remote_access_ready",
)
_TEMPORARY_TOKENS = ("temporary", "recurred", "short_observation", "immediate_recovery")
_DIAGNOSTIC_TOKENS = ("diagnostic", "evidence", "hypothesis", "reclassified", "supported")

_ROLE_MAP = {
    "collect": "collect",
    "inspect": "inspect",
    "observe": "observe",
    "verify": "verify",
    "change": "change",
    "mitigate": "mitigate",
    "escalate": "escalate",
    "act": "act",
    "safety_precondition": "safety_precondition",
}


class GoldAdapterError(ValueError):
    """Raised when a frozen annotation cannot be projected safely."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _strings(child)


def evidence_ids(value: Any) -> list[str]:
    """Extract stable message/Jira evidence ids from an arbitrary annotation subtree."""
    result: list[str] = []
    for text in _strings(value):
        result.extend(MESSAGE_ID_RE.findall(text))
        result.extend(
            match if match.lower().startswith("jira:") else f"jira:{match}"
            for match in JIRA_ID_RE.findall(text)
        )
    return list(dict.fromkeys(result))


def _is_message_evidence_id(value: str) -> bool:
    return bool(MESSAGE_ID_RE.fullmatch(value) or SYNTHETIC_MESSAGE_ID_RE.fullmatch(value))


def _clean_token(value: Any) -> str:
    return str(value or "").strip().strip("`").lower().replace("-", "_")


def normalize_outcome_type(value: Any, *, summary: str = "", assessment: str = "") -> tuple[str, str]:
    """Return a canonical outcome and an auditable normalization reason."""
    raw = _clean_token(value)
    candidate = raw
    if not TOKEN_RE.fullmatch(candidate):
        tokens = BACKTICK_TOKEN_RE.findall(str(assessment or ""))
        if tokens:
            candidate = _clean_token(tokens[0])
            reason = "categorical_token_recovered_from_assessment"
        else:
            reason = "narrative_outcome_inferred"
    else:
        reason = "direct_or_mapped_token"
    if candidate in CANONICAL_OUTCOMES:
        return candidate, reason
    if candidate in _OUTCOME_MAP:
        return _OUTCOME_MAP[candidate], reason
    lowered = " ".join((candidate, str(value or ""), summary, assessment)).lower()
    if any(token in lowered for token in ("execution_failed", "安装失败", "执行失败", "自身报错")):
        return "execution_failed", reason
    if any(token in lowered for token in ("no_effect", "无效", "故障仍", "还是一样", "未解决")):
        return "ineffective", reason
    if any(token in lowered for token in _TEMPORARY_TOKENS) or any(
        token in lowered for token in ("暂时", "短期", "随后复发", "再次报", "当下可")
    ):
        return "partial_temporary", reason
    if any(token in lowered for token in ("mitigation", "恢复生产", "观察到恢复")):
        return "mitigation_observed", reason
    if any(token in lowered for token in _DIAGNOSTIC_TOKENS):
        return "diagnostic_method", reason
    if any(token in lowered for token in ("not_root_cause", "排除", "反驳", "不是根因")):
        return "context_not_root_cause", reason
    if any(token in lowered for token in ("verified_fix", "验证通过", "长期未复发", "最终解决")):
        return "verified_fix", reason
    if any(token in candidate for token in _PENDING_TOKENS) or any(
        token in lowered for token in ("待验证", "未见", "没有记录", "不能确认", "未解析", "本地不能复算")
    ):
        return "pending_validation", reason
    return "pending_validation", f"{reason}:unknown_token"


def normalize_action_role(value: Any, *, label: str = "") -> tuple[str, str]:
    raw = _clean_token(value)
    if raw in _ROLE_MAP:
        return _ROLE_MAP[raw], "direct"
    for token, role in (
        ("collect", "collect"), ("log", "collect"), ("inspect", "inspect"),
        ("check", "inspect"), ("verify", "verify"), ("retest", "verify"),
        ("observe", "observe"), ("reinstall", "change"), ("replace", "change"),
        ("change", "change"), ("reconnect", "change"), ("restart", "mitigate"),
        ("mitigate", "mitigate"), ("escalate", "escalate"),
    ):
        if token in raw:
            return role, "mapped_raw_role"
    for token, role in (
        ("收集", "collect"), ("日志", "collect"), ("检查", "inspect"), ("核查", "inspect"),
        ("验证", "verify"), ("复测", "verify"), ("观察", "observe"), ("更换", "change"),
        ("重装", "change"), ("安装", "change"), ("拔插", "change"), ("升级", "change"),
        ("重启", "mitigate"), ("提交", "escalate"),
    ):
        if token in label:
            return role, "inferred_from_label"
    return "act", "fallback"


def _canonical_action(action: dict[str, Any], *, fallback_ref: str, separate_outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    label = str(action.get("label") or action.get("action_label") or "").strip()
    raw_role = action.get("action_role") or action.get("role") or ""
    role, role_reason = normalize_action_role(raw_role, label=label)
    embedded = action.get("outcome") if isinstance(action.get("outcome"), dict) else {}
    outcome = embedded or separate_outcome or {}
    raw_outcome = outcome.get("outcome_type") or "pending_validation"
    summary = str(outcome.get("summary") or "")
    reviewed_row = action.get("reviewed_source_row") if isinstance(action.get("reviewed_source_row"), dict) else {}
    assessment = str(reviewed_row.get("判定") or "")
    canonical_outcome, outcome_reason = normalize_outcome_type(raw_outcome, summary=summary, assessment=assessment)
    source_ids = list(dict.fromkeys([
        *evidence_ids(action.get("source_evidence_ids") or []),
        *evidence_ids(action.get("evidence_anchor_ids") or []),
        *evidence_ids(outcome.get("source_evidence_ids") or []),
    ]))
    return {
        "action_id": str(action.get("action_ref") or fallback_ref),
        "occurrence_id": action.get("occurrence_ref"),
        "label": label,
        "summary": str(action.get("summary") or ""),
        "action_role": role,
        "raw_action_role": str(raw_role or ""),
        "action_role_normalization_reason": role_reason,
        "execution_status": str(action.get("execution_status") or "reviewed_unspecified"),
        "evidence_ids": source_ids,
        "outcome": {
            "outcome_type": canonical_outcome,
            "raw_outcome_type": str(raw_outcome or ""),
            "summary": summary or str(raw_outcome or ""),
            "normalization_reason": outcome_reason,
            "evidence_ids": evidence_ids(outcome.get("source_evidence_ids") or []),
        },
    }


def _identity_rows(payload: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    maps = [payload.get("device_identity_map"), payload.get("affected_device_map")]
    for mapping in maps:
        if isinstance(mapping, dict):
            mapping = [{"identity": key, "description": value} for key, value in mapping.items()]
        for item in mapping or []:
            if isinstance(item, dict):
                rows.append({
                    "identity": str(item.get("identity") or item.get("device") or item.get("device_id") or item.get("label") or ""),
                    "site": str(item.get("site") or item.get("location") or ""),
                    "line": str(item.get("line") or item.get("line_id") or ""),
                    "version": str(item.get("version") or item.get("software_version") or ""),
                    "description": " ".join(str(value) for value in item.values() if value is not None),
                })
    variant = trace.get("variant") if isinstance(trace.get("variant"), dict) else {}
    if any(variant.get(key) for key in ("equipment_type", "site", "software_version")):
        rows.append({
            "identity": str(variant.get("equipment_type") or ""),
            "site": str(variant.get("site") or ""),
            "line": "",
            "version": str(variant.get("software_version") or ""),
            "description": str(variant.get("owner_context") or ""),
        })
    for occurrence in trace.get("occurrences") or []:
        if isinstance(occurrence, dict) and occurrence.get("device_scope"):
            rows.append({"identity": str(occurrence["device_scope"]), "site": "", "line": "", "version": "", "description": "occurrence_scope"})
    unique: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "").strip().lower() for field in ("identity", "site", "line", "version"))
        if any(key):
            unique.setdefault(key, row)
    return list(unique.values())


def _canonical_trace(
    payload: dict[str, Any],
    raw_trace: dict[str, Any],
    *,
    trace_id: str,
    actions: list[dict[str, Any]],
    standalone_outcomes: list[dict[str, Any]] | None = None,
    fallback_evidence: Any = None,
) -> dict[str, Any]:
    outcome_by_action = {
        str(item.get("action_label") or ""): item
        for item in standalone_outcomes or []
        if isinstance(item, dict)
    }
    canonical_actions = [
        _canonical_action(
            action,
            fallback_ref=f"{trace_id}:action:{index}",
            separate_outcome=outcome_by_action.get(str(action.get("label") or "")),
        )
        for index, action in enumerate(actions, start=1)
        if isinstance(action, dict)
    ]
    raw_anchors = raw_trace.get("evidence_anchor_ids") or []
    anchor_ids = evidence_ids([raw_anchors, fallback_evidence or []])
    anchor_ids.extend(
        str(value) for value in fallback_evidence or []
        if isinstance(value, str) and _is_message_evidence_id(value)
    )
    anchor_ids = list(dict.fromkeys(anchor_ids))
    if not anchor_ids:
        anchor_ids = list(dict.fromkeys(value for action in canonical_actions for value in action["evidence_ids"]))
    message_ids = [value for value in anchor_ids if _is_message_evidence_id(value)]
    external_ids = [value for value in anchor_ids if value not in message_ids]
    family = raw_trace.get("family") if isinstance(raw_trace.get("family"), dict) else {}
    variant = raw_trace.get("variant") if isinstance(raw_trace.get("variant"), dict) else {}
    summary = str(raw_trace.get("symptom_summary") or raw_trace.get("summary") or raw_trace.get("label") or "")
    return {
        "trace_id": trace_id,
        "summary": summary,
        "family": {
            "label": str(family.get("label") or ""),
            "category": str(family.get("category") or ""),
            "subsystem": str(family.get("subsystem") or ""),
        },
        "variant": deepcopy(variant),
        "identities": _identity_rows(payload, raw_trace),
        "occurrences": deepcopy(raw_trace.get("occurrences") or []),
        "actions": canonical_actions,
        "outcomes": [action["outcome"] for action in canonical_actions],
        "outcome_state": str(raw_trace.get("outcome_state") or ""),
        "root_cause_state": str(raw_trace.get("root_cause_state") or ""),
        "evidence": {
            "anchor_ids": anchor_ids,
            "message_ids": message_ids,
            "external_ids": external_ids,
            "context_only_ids": [],
            "excluded_ids": [],
        },
        "uncertainties": deepcopy(raw_trace.get("uncertainties") or []),
        "raw_trace": deepcopy(raw_trace),
    }


def _adapt_v1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    gold = payload.get("gold") if isinstance(payload.get("gold"), dict) else {}
    raw_trace = gold.get("trace") if isinstance(gold.get("trace"), dict) else {}
    raw_trace = {**raw_trace, "family": gold.get("family") or {}, "variant": gold.get("variant") or {}}
    fallback = [
        *(payload.get("episode_input") or {}).get("evidence_message_ids", []),
        *evidence_ids(payload.get("evidence_anchor_map") or {}),
    ]
    return [_canonical_trace(
        payload,
        raw_trace,
        trace_id=f"{payload['case_id']}-trace-1",
        actions=list(gold.get("actions") or []),
        standalone_outcomes=list(gold.get("outcomes") or []),
        fallback_evidence=fallback,
    )]


def _adapt_multi(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_traces = payload.get("cases") or payload.get("traces") or []
    return [
        _canonical_trace(
            payload,
            trace,
            trace_id=str(trace.get("case_ref") or trace.get("trace_ref") or f"{payload['case_id']}-trace-{index}"),
            actions=list(trace.get("actions") or []),
        )
        for index, trace in enumerate(raw_traces, start=1)
        if isinstance(trace, dict)
    ]


def _suite_for(case_id: str) -> str:
    number = int(case_id.rsplit("-", 1)[-1])
    if number <= 10:
        return "semantic_regression"
    if number <= 15:
        return "reference_regression"
    return "development_regression"


def adapt_gold_file(path: str | Path, *, input_path: str | Path | None = None) -> dict[str, Any]:
    """Adapt one frozen truth file without modifying it."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    case_id = str(payload.get("case_id") or path.stem)
    schema = str(payload.get("schema_version") or "")
    if schema == "kg_v2.gold_case.v1":
        traces = _adapt_v1(payload)
    elif schema in {"kg_v2.blind_ground_truth.v3", "kg_v2.gold_ground_truth.v1"}:
        traces = _adapt_multi(payload)
    else:
        raise GoldAdapterError(f"{case_id}:unsupported_schema:{schema}")
    expected = int(payload.get("case_count") or payload.get("trace_count") or len(traces))
    if len(traces) != expected:
        raise GoldAdapterError(f"{case_id}:trace_count:{len(traces)}!={expected}")
    excluded = evidence_ids(payload.get("excluded_fragments") or payload.get("excluded_parallel_faults") or [])
    trace_anchor_ids = {value for trace in traces for value in trace["evidence"]["anchor_ids"]}
    for trace in traces:
        trace["evidence"]["excluded_ids"] = sorted(set(excluded) - set(trace["evidence"]["anchor_ids"]))
    input_payload: dict[str, Any] | None = None
    input_source: dict[str, Any] = {"available": False}
    if input_path is not None and Path(input_path).is_file():
        resolved = Path(input_path)
        input_payload = json.loads(resolved.read_text(encoding="utf-8"))
        all_input_ids = {
            str(item.get("message_id") or "")
            for item in input_payload.get("messages") or []
            if isinstance(item, dict) and item.get("message_id")
        }
        context_only = sorted(all_input_ids - trace_anchor_ids - set(excluded))
        for trace in traces:
            trace["evidence"]["context_only_ids"] = context_only
        input_source = {
            "available": True,
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "message_count": len(input_payload.get("messages") or []),
            "messages_sha256": str(input_payload.get("messages_sha256") or ""),
        }
    return {
        "schema_version": "debug_agent_system.gold_trace.v1",
        "case_id": case_id,
        "suite": _suite_for(case_id),
        "source_schema_version": schema,
        "review_status": str(payload.get("review_status") or payload.get("status") or ""),
        "graph_ingestion": bool(payload.get("graph_ingestion")),
        "split_required": bool(payload.get("split_required")),
        "source": {"truth_path": str(path), "truth_sha256": _sha256(path), "input": input_source},
        "trace_count": len(traces),
        "traces": traces,
        "excluded_evidence_ids": excluded,
        "input_payload": input_payload,
    }


def load_gold_001_020(root: str | Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    """Load all 20 approved Goldcases through the common read-only contract."""
    root = Path(root)
    truth_paths = [
        *sorted((root / "gold-v1").glob("goldcase-*.json")),
        *sorted((root / "review-v3" / "ground_truth").glob("goldcase-*.json")),
        *sorted((root / "gold-v2" / "ground_truth").glob("goldcase-*.json")),
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for truth_path in truth_paths:
        case_id = truth_path.stem
        number = int(case_id.rsplit("-", 1)[-1])
        if not 1 <= number <= 20:
            continue
        input_path = None
        if number >= 11:
            batch = "review-v3" if number <= 15 else "gold-v2"
            input_path = root / batch / "inputs" / truth_path.name
        by_id[case_id] = adapt_gold_file(truth_path, input_path=input_path)
    expected = [f"goldcase-{number:03d}" for number in range(1, 21)]
    if sorted(by_id) != expected:
        raise GoldAdapterError(f"case_ids:{','.join(sorted(by_id))}")
    return [by_id[case_id] for case_id in expected]


def adapter_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.gold_adapter_summary.v1",
        "case_count": len(cases),
        "trace_count": sum(case["trace_count"] for case in cases),
        "action_count": sum(len(trace["actions"]) for case in cases for trace in case["traces"]),
        "source_message_count": sum(int(case["source"]["input"].get("message_count") or 0) for case in cases),
        "suite_counts": {
            suite: sum(case["suite"] == suite for case in cases)
            for suite in ("semantic_regression", "reference_regression", "development_regression")
        },
        "outcome_counts": {
            outcome: sum(
                action["outcome"]["outcome_type"] == outcome
                for case in cases for trace in case["traces"] for action in trace["actions"]
            )
            for outcome in sorted(CANONICAL_OUTCOMES)
        },
    }
