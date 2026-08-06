"""KG_v2-native read model used by the production diagnosis runtime.

This module deliberately does not import the legacy ``knowledge`` package or
materialized Error/Check/Solution views.  Every identifier returned here is a
primary key of an object in KG_v2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import SqliteSAGV2

_CJK = re.compile(r"[\u4e00-\u9fff]{2,}")
_WORD = re.compile(r"[A-Za-z0-9_.:/+-]+")
_OX_CODE = re.compile(r"\b[oO]x([0-9a-fA-F]{6,8})\b")
_GENERIC_TOKENS = {
    "问题", "异常", "设备", "系统", "程序", "用户", "检查", "正常", "失败", "故障",
    "现场", "情况", "处理", "相关", "当前", "出现", "进行", "结果", "测试", "资料",
}


@dataclass(slots=True)
class V2Candidate:
    family_id: str
    variant_id: str
    family_label: str
    variant_label: str
    score: float
    route: str = "kg_v2_native_lexical"
    matched_fields: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    retrieval_paths: list[dict[str, Any]] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    supporting_chunks: list[dict[str, Any]] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False


@dataclass(slots=True)
class V2PlanStep:
    action_id: str
    label: str
    instruction: str
    action_role: str
    ordinal: int
    trace_step_id: str = ""
    destructive: bool = False
    high_cost: bool = False
    stage: str = ""
    safety_level: str = "safe"
    applicability_condition: str = ""
    expected_result: str = ""
    media_refs: list[dict[str, Any]] = field(default_factory=list)
    outcome_ids: list[str] = field(default_factory=list)
    branch_rule_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V2DiagnosticPlan:
    plan_id: str
    source_type: str
    family_id: str
    variant_id: str
    trace_id: str = ""
    policy_id: str = ""
    steps: list[V2PlanStep] = field(default_factory=list)
    required_info_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


class KGV2ReadModel:
    """Indexed immutable view over the canonical KG_v2 JSON graph."""

    def __init__(self, root: str, sag: SqliteSAGV2 | None = None) -> None:
        self.store = JsonKGV2Store(root)
        self.sag = sag
        self.by_type = {
            object_type: self.store.object_index(object_type)
            for object_type in V2_PRIMARY_KEYS
        }
        self.object_type_by_id: dict[str, str] = {}
        for object_type, objects in self.by_type.items():
            for object_id in objects:
                self.object_type_by_id[object_id] = object_type
        self.outgoing: dict[str, list[dict[str, Any]]] = {}
        self.incoming: dict[str, list[dict[str, Any]]] = {}
        for edge in self.store.relations:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src and dst:
                self.outgoing.setdefault(src, []).append(edge)
                self.incoming.setdefault(dst, []).append(edge)
        self.actions_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
        for action_id, action in self.by_type["DiagnosticAction"].items():
            variant_id = str(action.get("variant_id") or "")
            if variant_id:
                self.actions_by_variant.setdefault(variant_id, {})[action_id] = action
        self._evidence_cache: dict[tuple[str, ...], list[str]] = {}
        self.last_retrieval: dict[str, Any] = {"chunks": [], "trace": {}}
        self._variant_token_df: Counter[str] = Counter()
        for variant in self.by_type["FaultVariant"].values():
            text = " ".join([
                str(variant.get("label") or ""),
                str(variant.get("summary") or ""),
                *[str(item) for item in variant.get("keywords") or []],
            ])
            self._variant_token_df.update(_tokens(text))

    def has_object(self, object_id: str, object_type: str | None = None) -> bool:
        found_type = self.object_type_by_id.get(str(object_id or ""))
        return bool(found_type and (object_type is None or found_type == object_type))

    def get(self, object_id: str) -> dict[str, Any] | None:
        object_type = self.object_type_by_id.get(str(object_id or ""))
        return self.by_type.get(object_type, {}).get(object_id) if object_type else None

    def search_variants(self, query: str, limit: int = 10) -> list[V2Candidate]:
        normalized = _normalize(query)
        query_tokens = _tokens(normalized)
        recalled_ids: list[str] = []
        retrieval_paths: list[dict[str, Any]] = []
        recall_by_variant: dict[str, dict[str, Any]] = {}
        chunks: list[dict[str, Any]] = []
        trace: dict[str, Any] = {}
        if self.sag is not None:
            # Keep the original spelling for retrieval scope analysis.  The
            # scorer below intentionally uses a lowercase normalization, but
            # upper-case tool names such as SFC, DISM, DDU and SMART carry
            # entity meaning.  Passing ``normalized`` here erased that signal
            # before ``strong_identifiers`` could see it and allowed a generic
            # "修复系统" section to replace the named-tool document.
            retrieval = self.sag.retrieve(query, variant_limit=240, chunk_limit=24)
            recall_by_variant = {
                str(item.get("variant_id") or ""): item
                for item in retrieval.get("variant_rows") or []
                if str(item.get("variant_id") or "")
            }
            recalled_ids = list(recall_by_variant)
            retrieval_paths = list(retrieval.get("paths") or [])
            chunks = list(retrieval.get("chunks") or [])
            trace = dict(retrieval.get("trace") or {})
        variants = self.by_type["FaultVariant"]
        # The graph is intentionally small enough to fuse SAG recall with a
        # canonical scan.  This prevents an FTS seed miss from becoming a hard
        # recall miss while keeping SAG paths and rank evidence auditable.
        candidate_ids = _dedupe([*recalled_ids, *variants])
        candidates: list[V2Candidate] = []
        for variant_id in candidate_ids:
            variant = variants.get(variant_id)
            if variant is None:
                continue
            if not self.is_runtime_variant(variant_id):
                continue
            family_id = str(variant.get("family_id") or "")
            family = self.by_type["FaultFamily"].get(family_id) or {}
            recall = recall_by_variant.get(variant_id) or {}
            score, matched, components = _variant_score(
                normalized,
                query_tokens,
                variant,
                family,
                token_df=self._variant_token_df,
                variant_count=max(len(variants), 1),
            )
            # Variant FTS is the primary candidate engine.  Its index document
            # already aggregates typed KG_v2 fields and discriminative n-grams,
            # so preserve enough of its rank to beat broad SOP keyword matches.
            sag_recall = min(50.0, float(recall.get("recall_score") or 0.0) * 0.5)
            chunk_support = min(4.0, float(recall.get("chunk_support_score") or 0.0))
            score += sag_recall + chunk_support
            components.update({"sag_recall": round(sag_recall, 4), "chunk_support": round(chunk_support, 4)})
            if score <= 0:
                continue
            variant_chunks = [item for item in chunks if variant_id in set(item.get("variant_ids") or [])]
            chunk_evidence_ids = [
                str(item.get("object_id") or "") for item in variant_chunks
                if str(item.get("object_type") or "") == "EvidenceItem"
            ]
            evidence_ids = _dedupe([*self.evidence_ids_for([variant_id]), *chunk_evidence_ids])
            variant_paths = [item for item in retrieval_paths if item.get("variant_id") == variant_id][:12]
            candidates.append(V2Candidate(
                family_id=family_id,
                variant_id=variant_id,
                family_label=str(family.get("label") or family_id),
                variant_label=str(variant.get("label") or variant_id),
                score=round(score, 4),
                route="sag_v2_native" if variant_paths else "kg_v2_native_scan_fallback",
                matched_fields=matched,
                evidence_ids=evidence_ids,
                retrieval_paths=variant_paths,
                matched_entities=_dedupe(recall.get("matched_terms") or []),
                supporting_chunks=variant_chunks[:12],
                score_components={key: round(value, 4) for key, value in components.items()},
                fallback_used=not bool(variant_paths),
            ))
        candidates.sort(key=lambda item: (-item.score, item.variant_id))
        selected = candidates[: max(0, int(limit))]
        trace.update({
            "candidate_count": len(candidates),
            "top_score": selected[0].score if selected else 0.0,
            "second_score": selected[1].score if len(selected) > 1 else 0.0,
            "top_margin": round(selected[0].score - selected[1].score, 4) if len(selected) > 1 else 0.0,
            "fallback_used": bool(selected and selected[0].fallback_used),
        })
        self.last_retrieval = {"chunks": chunks, "paths": retrieval_paths, "trace": trace}
        return selected

    def is_runtime_variant(self, variant_id: str) -> bool:
        variant = self.by_type["FaultVariant"].get(variant_id) or {}
        if variant.get("execution_materialize_allowed") is False:
            return False
        actions = [
            action for action in self.actions_by_variant.get(variant_id, {}).values()
            if action.get("execution_materialize_allowed") is not False
        ]
        return bool(actions and self.evidence_ids_for([variant_id, str(variant.get("family_id") or "")]))

    def compile_plan(self, family_id: str, variant_id: str) -> V2DiagnosticPlan:
        if not self.has_object(variant_id, "FaultVariant"):
            raise KeyError(f"unknown KG_v2 FaultVariant: {variant_id}")
        variant_actions = {
            action_id: action
            for action_id, action in self.actions_by_variant.get(variant_id, {}).items()
            if action.get("execution_materialize_allowed") is not False
        }
        traces = [
            trace for trace in self.by_type["DiagnosticTrace"].values()
            if str(trace.get("variant_id") or "") == variant_id
        ]
        trace = max(traces, key=self._trace_quality, default=None)
        trace_id = str((trace or {}).get("trace_id") or "")
        policies = [
            policy for policy in self.by_type["DecisionPolicy"].values()
            if str(policy.get("family_id") or "") == family_id
        ]
        policy = max(policies, key=lambda item: len(item.get("ordered_action_ids") or []), default=None)
        policy_id = str((policy or {}).get("policy_id") or "")

        ordered_action_ids: list[str] = []
        if trace:
            ordered_action_ids.extend(str(x) for x in trace.get("actual_action_ids") or [])
            ordered_action_ids.extend(str(x) for x in trace.get("recommended_action_ids") or [])
        if policy:
            ordered_action_ids.extend(
                str(x) for x in policy.get("ordered_action_ids") or []
                if str(x) in variant_actions
            )
        ordered_action_ids.extend(
            action_id for action_id, _ in sorted(
                variant_actions.items(),
                key=lambda item: (int(item[1].get("step_order") or 9999), item[0]),
            )
        )
        ordered_action_ids = _dedupe(x for x in ordered_action_ids if x in variant_actions)

        trace_steps = [
            item for item in self.by_type["TraceStep"].values()
            if trace_id and str(item.get("trace_id") or "") == trace_id
        ]
        trace_steps.sort(key=lambda item: (int(item.get("ordinal") or 9999), str(item.get("trace_step_id") or "")))
        step_by_action: dict[str, dict[str, Any]] = {}
        for item in trace_steps:
            step_by_action.setdefault(str(item.get("action_id") or ""), item)
        outcomes_by_action: dict[str, list[dict[str, Any]]] = {}
        for outcome in self.by_type["ActionOutcome"].values():
            action_id = str(outcome.get("action_id") or "")
            if action_id in variant_actions:
                outcomes_by_action.setdefault(action_id, []).append(outcome)
        branch_by_trace_step: dict[str, list[dict[str, Any]]] = {}
        for rule in self.by_type["BranchRule"].values():
            if trace_id and str(rule.get("trace_id") or "") == trace_id:
                branch_by_trace_step.setdefault(str(rule.get("from_trace_step_id") or ""), []).append(rule)

        plan_evidence = self.evidence_ids_for([variant_id, trace_id, policy_id])
        steps: list[V2PlanStep] = []
        for ordinal, action_id in enumerate(ordered_action_ids, start=1):
            action = variant_actions[action_id]
            trace_step = step_by_action.get(action_id) or {}
            trace_step_id = str(trace_step.get("trace_step_id") or "")
            outcomes = outcomes_by_action.get(action_id) or []
            rules = branch_by_trace_step.get(trace_step_id) or []
            evidence_ids = self.evidence_ids_for([
                action_id,
                trace_step_id,
                *[str(item.get("outcome_id") or "") for item in outcomes],
                *[str(item.get("branch_rule_id") or "") for item in rules],
            ])
            if not evidence_ids:
                evidence_ids = list(plan_evidence)
            steps.append(V2PlanStep(
                action_id=action_id,
                label=str(action.get("label") or action_id),
                instruction=str(action.get("summary") or action.get("label") or action_id),
                action_role=str(action.get("action_role") or "inspect"),
                ordinal=ordinal,
                trace_step_id=trace_step_id,
                destructive=bool(action.get("destructive")),
                high_cost=bool(action.get("high_cost")),
                stage=str(action.get("stage") or ""),
                safety_level=str(action.get("safety_level") or "safe"),
                applicability_condition=str(action.get("applicability_condition") or ""),
                expected_result=str(action.get("expected_result") or ""),
                media_refs=[
                    dict(item)
                    for item in action.get("curated_image_refs") or []
                    if isinstance(item, dict)
                ],
                outcome_ids=[str(item.get("outcome_id") or "") for item in outcomes],
                branch_rule_ids=[str(item.get("branch_rule_id") or "") for item in rules],
                evidence_ids=evidence_ids,
            ))
        required_info_ids = [
            required_id for required_id, item in self.by_type["RequiredInfoSpec"].items()
            if str(item.get("variant_id") or "") == variant_id
        ]
        source_type = "DiagnosticTrace" if trace_id else ("DecisionPolicy" if policy_id else "FaultVariant")
        plan_id = trace_id or policy_id or variant_id
        return V2DiagnosticPlan(
            plan_id=plan_id,
            source_type=source_type,
            family_id=family_id,
            variant_id=variant_id,
            trace_id=trace_id,
            policy_id=policy_id,
            steps=steps,
            required_info_ids=required_info_ids,
            evidence_ids=_dedupe([*plan_evidence, *[e for step in steps for e in step.evidence_ids]]),
        )

    def required_info(self, required_info_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [
            self.by_type["RequiredInfoSpec"][item_id]
            for item_id in required_info_ids
            if item_id in self.by_type["RequiredInfoSpec"]
        ]

    def evidence(self, evidence_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [
            self.by_type["EvidenceItem"][item_id]
            for item_id in _dedupe(evidence_ids)
            if item_id in self.by_type["EvidenceItem"]
        ]

    def evidence_ids_for(self, object_ids: Iterable[str]) -> list[str]:
        cache_key = tuple(sorted({str(x) for x in object_ids if x}))
        cached = self._evidence_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        evidence_ids: list[str] = []
        frontier = list(cache_key)
        seen: set[str] = set()
        # Evidence is allowed to travel through SourceCase and DiagnosticTrace,
        # but not through arbitrary semantic neighbors that could contaminate a
        # candidate with another variant's support.
        for _ in range(3):
            next_frontier: list[str] = []
            for object_id in frontier:
                if object_id in seen:
                    continue
                seen.add(object_id)
                obj = self.get(object_id) or {}
                evidence_ids.extend(str(x) for x in obj.get("evidence_ids") or [])
                for edge in self.incoming.get(object_id) or []:
                    src = str(edge.get("from") or "")
                    rel = str(edge.get("relation") or "")
                    if rel == "evidences" and self.has_object(src, "EvidenceItem"):
                        evidence_ids.append(src)
                    elif rel == "supports" and self.has_object(src, "SourceCase"):
                        next_frontier.append(src)
                for edge in self.outgoing.get(object_id) or []:
                    dst = str(edge.get("to") or "")
                    rel = str(edge.get("relation") or "")
                    if rel in {"has_trace", "has_outcome", "has_observation"}:
                        next_frontier.append(dst)
            frontier = next_frontier
        result = [x for x in _dedupe(evidence_ids) if self.has_object(x, "EvidenceItem")]
        self._evidence_cache[cache_key] = result
        return list(result)

    def branch_rules_for_step(self, step: V2PlanStep) -> list[dict[str, Any]]:
        return [
            self.by_type["BranchRule"][rule_id]
            for rule_id in step.branch_rule_ids
            if rule_id in self.by_type["BranchRule"]
        ]

    def outcomes_for_step(self, step: V2PlanStep) -> list[dict[str, Any]]:
        return [
            self.by_type["ActionOutcome"][outcome_id]
            for outcome_id in step.outcome_ids
            if outcome_id in self.by_type["ActionOutcome"]
        ]

    def _trace_quality(self, trace: dict[str, Any]) -> tuple[int, int, int]:
        trace_id = str(trace.get("trace_id") or "")
        trace_steps = sum(
            1 for item in self.by_type["TraceStep"].values()
            if str(item.get("trace_id") or "") == trace_id
        )
        return (
            trace_steps,
            len(trace.get("actual_action_ids") or []),
            len(trace.get("recommended_action_ids") or []),
        )


def _normalize(text: str) -> str:
    return _OX_CODE.sub(lambda match: f"0x{match.group(1).lower()}", str(text or "")).lower()


def _tokens(text: str) -> set[str]:
    normalized = str(text or "").lower()
    tokens = set(_WORD.findall(normalized))
    for run in _CJK.findall(normalized):
        if len(run) <= 8:
            tokens.add(run)
        for size in (2, 3, 4):
            tokens.update(run[index:index + size] for index in range(len(run) - size + 1))
    return {token for token in tokens if token.strip() and token not in _GENERIC_TOKENS}


def _variant_score(
    query: str,
    query_tokens: set[str],
    variant: dict[str, Any],
    family: dict[str, Any],
    *,
    token_df: Counter[str] | None = None,
    variant_count: int = 1,
) -> tuple[float, list[str], dict[str, float]]:
    matched: list[str] = []
    score = 0.0
    components: dict[str, float] = {}
    fields = {
        "variant_label": str(variant.get("label") or "").lower(),
        "variant_summary": str(variant.get("summary") or "").lower(),
        "family_label": str(family.get("label") or "").lower(),
        "family_summary": str(family.get("summary") or "").lower(),
        "subsystem": str(family.get("subsystem") or "").lower(),
        "error_phase": str(variant.get("error_phase") or "").lower(),
    }
    label = fields["variant_label"]
    family_label = fields["family_label"]
    label_codes = re.findall(r"0x[0-9a-f]{6,8}", label)
    if label_codes and any(code in query for code in label_codes):
        score += 24.0
        components["exact_error_code"] = 24.0
        matched.append(f"variant.label:error_code:{label_codes[0]}")
    if label and len(label) >= 4 and label in query:
        score += 18.0
        components["exact_variant_label"] = 18.0
        matched.append("variant.label:exact")
    if family_label and family_label in query:
        score += 8.0
        components["exact_family_label"] = 8.0
        matched.append("family.label:exact")
    for keyword in variant.get("keywords") or []:
        normalized_keyword = str(keyword or "").lower().strip()
        if len(normalized_keyword) >= 2 and normalized_keyword in query:
            exact_code = bool(re.fullmatch(r"0x[0-9a-f]{6,8}|[a-z_]+(?:[._-][a-z0-9_]+)+", normalized_keyword))
            contribution = 15.0 if exact_code else min(7.0, 2.5 + len(normalized_keyword) / 3.0)
            score += contribution
            components["variant_keywords"] = components.get("variant_keywords", 0.0) + contribution
            matched.append(f"variant.keyword:{keyword}")
    for keyword in family.get("keywords") or []:
        normalized_keyword = str(keyword or "").lower().strip()
        if len(normalized_keyword) >= 2 and normalized_keyword in query:
            contribution = min(4.0, 1.0 + len(normalized_keyword) / 4.0)
            score += contribution
            components["family_keywords"] = components.get("family_keywords", 0.0) + contribution
            matched.append(f"family.keyword:{keyword}")
    weights = {
        "variant_label": 9.0,
        "variant_summary": 6.0,
        "family_label": 4.0,
        "family_summary": 2.0,
        "subsystem": 3.0,
        "error_phase": 2.0,
    }
    for name, value in fields.items():
        value_tokens = _tokens(value)
        if not value_tokens or not query_tokens:
            continue
        overlap = query_tokens & value_tokens
        if not overlap:
            continue
        similarity = len(overlap) / max(math.sqrt(len(query_tokens) * len(value_tokens)), 1.0)
        contribution = weights[name] * similarity
        score += contribution
        components[name] = components.get(name, 0.0) + contribution
        if contribution >= 0.75:
            matched.append(name)
    # Rare 3/4-character phrases distinguish a precise incident from a generic
    # SOP sibling without requiring hand-maintained per-variant keywords.
    variant_tokens = _tokens(" ".join([fields["variant_label"], fields["variant_summary"]]))
    rare_overlap = [
        token for token in query_tokens & variant_tokens
        if len(token) >= 3 and not re.fullmatch(r"[a-z0-9]{2,}", token)
    ]
    if rare_overlap and token_df is not None:
        rare_score = sum(
            min(2.0, math.log((variant_count + 1) / (int(token_df.get(token, 0)) + 1)) + 1.0)
            for token in rare_overlap
        )
        rare_score = min(16.0, rare_score * 0.75)
        score += rare_score
        components["rare_phrase_overlap"] = rare_score
        matched.extend(f"rare:{token}" for token in sorted(rare_overlap, key=lambda item: (-len(item), item))[:8])
    return score, _dedupe(matched), components


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "")
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
