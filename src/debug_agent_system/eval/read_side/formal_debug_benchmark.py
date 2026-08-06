"""Build and score the formal two-layer AOI Debug Benchmark v1.

The core set is deliberately separate from broad regression pools.  Building
the 100-case annotation pack does not claim that pending cases are human
frozen; ``--release-check`` stays fail-closed until every core case has an
independent frozen review status.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from debug_agent_system.eval.read_side.document_qa_extended_benchmark import (
    validate_dataset as validate_document_pool,
)
from debug_agent_system.eval.read_side.fae_report_benchmark import (
    validate_dataset as validate_fae_pool,
)
from debug_agent_system.eval.read_side.kg_v2_quality_dataset import (
    validate_dataset as validate_kg_quality_pool,
)
from debug_agent_system.eval.read_side.unified_benchmark import (
    validate_dataset as validate_kg_contract_pool,
)
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import kg_v2_graph_revision


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "debug_agent_system.formal_debug_benchmark.v1"
BENCHMARK_ID = "aoi-formal-debug-benchmark-v1"
BENCHMARK_VERSION = "1.0.0"

PUBLIC_CORE_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/core.json"
CORE_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/private/core_master.json"
VALIDATION_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/core_validation.json"
TEST_INPUTS_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/core_test_inputs.json"
TEST_GOLD_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/private/core_test_gold.json"
REVIEW_QUEUE_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/review_queue.json"
BROAD_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/broad_pools.json"
REPORT_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/report.json"
APPROVAL_PATH = ROOT / "data/eval/formal_debug_benchmark_v1/approval.json"
SCORE_PATH = ROOT / "data/results/formal_debug_benchmark_v1/latest_score.json"
PREDICTION_TEMPLATE_PATH = (
    ROOT / "data/eval/formal_debug_benchmark_v1/predictions.template.json"
)
FEATURE_SELFTEST_KG_RUNTIME_PATH = (
    ROOT / "data/eval/formal_debug_benchmark_v1/feature_selftest_queries_kg_runtime.jsonl"
)
FEATURE_SELFTEST_FAE_PATH = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "feature_selftest_queries_fae.jsonl"
)
FEATURE_SELFTEST_DOCUMENT_QA_PATH = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "feature_selftest_queries_document_qa.jsonl"
)
FEATURE_SELFTEST_MANIFEST_PATH = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "feature_selftest_queries.manifest.json"
)
FAE_QUERY_REWRITES_PATH = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "fae_query_rewrites.json"
)
KG_QUERY_REWRITES_PATH = (
    ROOT
    / "data/eval/formal_debug_benchmark_v1"
    / "kg_query_rewrites.json"
)
MARKDOWN_PATH = ROOT / "docs/AOI_Formal_Debug_Benchmark_v1.md"

DOCUMENT_POOL = ROOT / "data/eval/benchmark/aoi_document_qa_extended_v1.json"
KG_QUALITY_POOL = ROOT / "data/eval/scenarios/kg_v2_quality_v1.json"
KG_CONTRACT_POOL = ROOT / "data/eval/benchmark/aoi_debug_benchmark_v1.json"
FAE_POOL = ROOT / "data/eval/benchmark/aoi_fae_report_benchmark_v2.json"
SAG_REGRESSION_POOL = ROOT / "data/eval/scenarios/sag_regression_v1.json"
TERMINOLOGY_MANIFEST = ROOT / "data/kg_v2/terminology/terminology_manifest.json"
RUNTIME_CONFIG = ROOT / "config/kg_v2_raw_codex.json"

LAYER_QUOTAS = {
    "routing_domain_boundary": 20,
    "document_retrieval_grounded_answer": 25,
    "fault_location_first_action": 25,
    "multi_turn_branch_safety_resolution": 20,
    "long_context_multi_trace": 10,
}
VALIDATION_QUOTAS = {
    "routing_domain_boundary": 12,
    "document_retrieval_grounded_answer": 15,
    "fault_location_first_action": 15,
    "multi_turn_branch_safety_resolution": 12,
    "long_context_multi_trace": 6,
}

CORE_TASK_CONTRACT = """A formal core case must expose a field query, source,
scenario type, difficulty, capability layer, expected route, required evidence,
allowed and forbidden conclusions, diagnosis/first-action-or-follow-up gold,
execution and resolution gates, human review state, and frozen data versions.
KG snapshot expectations are conformance candidates, never independent Gold.
Held-out test cases are evaluation-only and must not be used for optimization.
"""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def _round_robin_documents(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        title = str((case.get("source_refs") or {}).get("document_title") or "")
        grouped[title].append(case)
    ordered: list[dict[str, Any]] = []
    while any(grouped.values()):
        for title in sorted(grouped):
            if grouped[title]:
                ordered.append(grouped[title].pop(0))
    return ordered


def _difficulty_for_document(case: dict[str, Any]) -> str:
    answer = str((case.get("answer_gold") or {}).get("reference_answer") or "")
    refs = case.get("source_refs") or {}
    section_count = len(refs.get("section_ids") or [])
    if section_count > 1 or len(answer) >= 500:
        return "hard"
    if len(answer) >= 180:
        return "medium"
    return "easy"


def _source_descriptor(path: Path, source_case: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(path),
        "source_case_id": str(source_case.get("case_id") or ""),
        "source_type": str(source_case.get("source_type") or source_case.get("evaluation_track") or ""),
        "source_refs": source_case.get("source_refs") or {},
    }


def _base_case(
    *,
    case_id: str,
    capability_layer: str,
    source_case: dict[str, Any],
    source_path: Path,
    scenario_type: str,
    difficulty: str,
    query: str,
    turns: list[dict[str, Any]],
    expected_route: dict[str, Any],
    evidence: dict[str, Any],
    conclusions: dict[str, Any],
    diagnosis: dict[str, Any],
    execution_policy: dict[str, Any],
    review_status: str,
    gold_origin: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "split": "",
        "optimization_eligible": True,
        "query": query.strip(),
        "turns": turns,
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "capability_layer": capability_layer,
        "source": _source_descriptor(source_path, source_case),
        "data_versions": {
            "benchmark_version": BENCHMARK_VERSION,
            "source_dataset_sha256": _sha256(source_path),
            "source_case_id": str(source_case.get("case_id") or ""),
        },
        "expected_route": expected_route,
        "evidence_gold": evidence,
        "conclusion_gold": conclusions,
        "diagnosis_gold": diagnosis,
        "execution_policy": execution_policy,
        "human_review": {
            "status": review_status,
            "gold_origin": gold_origin,
            "independent_semantic_gold": False,
            "reviewer_ids": [],
            "reviewed_at": "",
            "notes": "",
        },
        "leakage_control": {
            "answer_in_query": False,
            "kg_primary_key_in_query": False,
            "source_gold_visible_to_runtime": False,
        },
    }


def _document_case(
    source: dict[str, Any],
    *,
    case_id: str,
    layer: str,
) -> dict[str, Any]:
    refs = source["source_refs"]
    answer = source["answer_gold"]
    section_ids = [str(value) for value in refs.get("section_ids") or []]
    if not section_ids and refs.get("section_id"):
        section_ids = [str(refs["section_id"])]
    document_id = str(refs.get("document_id") or "")
    route_targets = [value for value in [document_id, *section_ids] if value]
    case = _base_case(
        case_id=case_id,
        capability_layer=layer,
        source_case=source,
        source_path=DOCUMENT_POOL,
        scenario_type=(
            "document_route_and_domain_boundary"
            if layer == "routing_domain_boundary"
            else "source_grounded_document_qa"
        ),
        difficulty=_difficulty_for_document(source),
        query=str(source["query"]),
        turns=[],
        expected_route={
            "route_type": "knowledge_document_section",
            "required_target_ids": route_targets,
            "forbidden_route_types": ["unsupported_external_knowledge"],
        },
        evidence={
            "must_recall_ids": section_ids or [document_id],
            "optional_ids": [
                str(item.get("asset_path") or item.get("media_id") or "")
                for item in (answer.get("source_images") or [])
                if isinstance(item, dict)
            ],
            "forbidden_ids": [],
            "reference_answer": str(answer.get("reference_answer") or ""),
        },
        conclusions={
            "allowed": [str(answer.get("reference_answer") or "")],
            "forbidden": [
                "来源未支持的故障根因",
                "现场动作已经执行",
                "故障已经解决",
            ],
            "abstention_allowed": True,
        },
        diagnosis={
            "family_id": "",
            "acceptable_variant_ids": [],
            "first_action_id": "",
            "required_followup_ids": [],
        },
        execution_policy={
            "execution_allowed": False,
            "human_confirmation_required": False,
            "allowed_statuses": ["answer", "ask_info"],
            "resolved_allowed": False,
            "forbidden_action_ids": [],
        },
        review_status="pending_human_freeze",
        gold_origin="source_snapshot_candidate",
    )
    case["leakage_control"]["group_id"] = (
        f"document:{document_id or refs.get('document_title') or source['case_id']}"
    )
    return case


def _runtime_case(
    source: dict[str, Any],
    *,
    case_id: str,
    layer: str,
) -> dict[str, Any]:
    expected = source.get("expected") or {}
    evidence_ids = [str(value) for value in expected.get("evidence_ids") or []]
    variant_ids = [str(value) for value in expected.get("acceptable_variant_ids") or []]
    forbidden_statuses = [str(value) for value in expected.get("forbidden_statuses") or []]
    terminal = str(expected.get("terminal_status") or "")
    allowed_statuses = [terminal] if terminal else ["step", "ask_info"]
    forbidden_actions = [
        str(value)
        for value in expected.get("forbidden_action_ids_without_confirmation") or []
    ]
    allowed = [
        value
        for value in (
            f"family_id={expected.get('family_id')}" if expected.get("family_id") else "",
            f"variant_id={expected.get('variant_id')}" if expected.get("variant_id") else "",
            f"first_action_id={expected.get('first_action_id')}" if expected.get("first_action_id") else "",
            f"terminal_status={terminal}" if terminal else "",
        )
        if value
    ]
    route = expected.get("sag") or {}
    raw_query = str(source.get("query") or "").strip()
    task_type = str(source.get("task_type") or "runtime_replay")
    task_prompts = {
        "variant_retrieval": "请先定位最匹配的故障族和故障变体，并说明必须核对的证据。",
        "first_action": "如果当前只能安排一个低风险首动作，应该先做什么，为什么？",
        "branch_transition": "结合现有现象和后续反馈，下一步应进入哪个排查分支？",
        "ask_info_gate": "当前信息是否足以继续诊断？若不足，只追问最关键的缺失信息。",
        "safety_gate": "判断下一动作能否直接执行；如需人工确认，明确确认项和风险。",
        "resolution_gate": "判断当前证据是否允许宣布解决，并给出尚缺的复验条件。",
    }
    query = f"现场反馈：{raw_query}\n{task_prompts.get(task_type, '给出证据约束下的下一步。')}"
    case = _base_case(
        case_id=case_id,
        capability_layer=layer,
        source_case=source,
        source_path=KG_QUALITY_POOL,
        scenario_type=task_type,
        difficulty=str(source.get("difficulty") or "medium"),
        query=query,
        turns=list(source.get("turns") or []),
        expected_route={
            "route_type": str(route.get("expected_route") or "sag_v2_native"),
            "required_target_ids": variant_ids,
            "forbidden_route_types": ["legacy_kg_v1", "ungrounded_answer"],
        },
        evidence={
            "must_recall_ids": evidence_ids,
            "optional_ids": [],
            "forbidden_ids": [],
            "reference_answer": "",
        },
        conclusions={
            "allowed": allowed,
            "forbidden": [
                *[f"status={value}" for value in forbidden_statuses],
                "将待验证动作写成已执行",
                "将临时恢复写成 verified_fix",
            ],
            "abstention_allowed": terminal == "ask_info",
        },
        diagnosis={
            "family_id": str(expected.get("family_id") or ""),
            "acceptable_variant_ids": variant_ids,
            "first_action_id": str(expected.get("first_action_id") or ""),
            "required_followup_ids": [
                str(value) for value in expected.get("required_info_ids") or []
            ],
            "branch_rule_ids": [
                str(value) for value in expected.get("branch_rule_ids") or []
            ],
        },
        execution_policy={
            "execution_allowed": False,
            "human_confirmation_required": bool(
                expected.get("safety_confirmation_required")
            ),
            "allowed_statuses": allowed_statuses,
            "resolved_allowed": terminal == "resolved",
            "forbidden_action_ids": forbidden_actions,
        },
        review_status="pending_independent_human_freeze",
        gold_origin="kg_runtime_conformance_candidate",
    )
    case["leakage_control"]["group_id"] = (
        "runtime-scenario:" + _text_sha256(raw_query)[:16]
    )
    return case


def _runtime_route_case(source: dict[str, Any], *, case_id: str) -> dict[str, Any]:
    expected = source.get("expected") or {}
    evidence_ids = [str(value) for value in expected.get("evidence_ids") or []]
    variant_ids = [str(value) for value in expected.get("acceptable_variant_ids") or []]
    raw_query = str(source.get("query") or "").strip()
    case = _base_case(
        case_id=case_id,
        capability_layer="routing_domain_boundary",
        source_case=source,
        source_path=KG_QUALITY_POOL,
        scenario_type="diagnostic_route",
        difficulty="medium",
        query=(
            f"现场反馈：{raw_query}\n"
            "这条请求应进入文档问答、AOI 故障诊断，还是领域外处理？只做路由并列出依据。"
        ),
        turns=[],
        expected_route={
            "route_type": "sag_v2_native",
            "required_target_ids": variant_ids,
            "forbidden_route_types": ["knowledge_document_section", "out_of_domain"],
        },
        evidence={
            "must_recall_ids": evidence_ids,
            "optional_ids": [],
            "forbidden_ids": [],
            "reference_answer": "",
        },
        conclusions={
            "allowed": ["进入 AOI 故障诊断路由"],
            "forbidden": ["直接执行动作", "宣布故障已解决"],
            "abstention_allowed": False,
        },
        diagnosis={
            "family_id": "",
            "acceptable_variant_ids": [],
            "first_action_id": "",
            "required_followup_ids": [],
        },
        execution_policy={
            "execution_allowed": False,
            "human_confirmation_required": False,
            "allowed_statuses": ["answer"],
            "resolved_allowed": False,
            "forbidden_action_ids": [],
        },
        review_status="pending_independent_human_freeze",
        gold_origin="kg_runtime_route_candidate",
    )
    case["leakage_control"]["group_id"] = (
        "runtime-scenario:" + _text_sha256(raw_query)[:16]
    )
    return case


def _out_of_domain_case(query: str, *, case_id: str) -> dict[str, Any]:
    source = {
        "case_id": "SAG_REG_OUT_OF_DOMAIN_001",
        "source_type": "sag_regression_boundary_seed",
        "source_refs": {"base_case_id": "SAG_REG_OUT_OF_DOMAIN_001"},
    }
    case = _base_case(
        case_id=case_id,
        capability_layer="routing_domain_boundary",
        source_case=source,
        source_path=SAG_REGRESSION_POOL,
        scenario_type="out_of_domain_boundary",
        difficulty="easy",
        query=query,
        turns=[],
        expected_route={
            "route_type": "out_of_domain",
            "required_target_ids": [],
            "forbidden_route_types": ["knowledge_document_section", "sag_v2_native"],
        },
        evidence={
            "must_recall_ids": ["SAG_REG_OUT_OF_DOMAIN_001"],
            "optional_ids": [],
            "forbidden_ids": [],
            "reference_answer": "",
        },
        conclusions={
            "allowed": ["说明不属于 AOI Debug 知识范围并停止技术处置"],
            "forbidden": ["编造 AOI 排故步骤", "宣布问题已解决"],
            "abstention_allowed": True,
        },
        diagnosis={
            "family_id": "",
            "acceptable_variant_ids": [],
            "first_action_id": "",
            "required_followup_ids": [],
        },
        execution_policy={
            "execution_allowed": False,
            "human_confirmation_required": False,
            "allowed_statuses": ["unsupported"],
            "resolved_allowed": False,
            "forbidden_action_ids": [],
        },
        review_status="pending_human_freeze",
        gold_origin="manual_domain_policy_candidate",
    )
    case["leakage_control"]["group_id"] = "route-policy:out-of-domain"
    return case


def _trace_case(source: dict[str, Any], *, case_id: str) -> dict[str, Any]:
    expected = source.get("expected") or {}
    source_refs = source.get("source_refs") or []
    must_recall = [
        str(item.get("id") or item.get("path") or "")
        for item in source_refs
        if isinstance(item, dict) and (item.get("id") or item.get("path"))
    ]
    input_ref = str(source.get("source_input_ref") or "")
    input_payload = _load(ROOT / input_ref) if input_ref else {}
    messages = input_payload.get("messages") or []
    anchor = next(
        (
            str(message.get("text") or "").strip()
            for message in messages
            if isinstance(message, dict)
            and str(message.get("text") or "").strip()
            and not str(message.get("text") or "").startswith("[Image:")
        ),
        "现场记录包含多个设备或多条故障链。",
    )
    anchor = " ".join(anchor.split())[:180]
    case = _base_case(
        case_id=case_id,
        capability_layer="long_context_multi_trace",
        source_case=source,
        source_path=KG_QUALITY_POOL,
        scenario_type="source_only_multi_device_multi_fault_trace",
        difficulty="expert",
        query=(
            f"下面是一段长时间窗现场记录，首条有效反馈为：“{anchor}”。"
            "请读取随题提供的 source-only 记录，按设备、故障链和时间边界拆分 Trace，"
            "逐条给出故障族/变体、动作与结果、证据锚点和不确定性；不得合并并行故障，"
            "也不得把建议或短暂恢复写成已验证解决。"
        ),
        turns=list(source.get("turns") or []),
        expected_route={
            "route_type": "source_only_trace_reconstruction",
            "required_target_ids": [str(value) for value in expected.get("trace_ids") or []],
            "forbidden_route_types": ["kg_truth_label_lookup"],
        },
        evidence={
            "must_recall_ids": must_recall,
            "optional_ids": [],
            "forbidden_ids": [],
            "reference_answer": "",
        },
        conclusions={
            "allowed": [
                *[f"family={value}" for value in expected.get("family_labels") or []],
                *[f"variant={value}" for value in expected.get("variant_labels") or []],
            ],
            "forbidden": [str(value) for value in expected.get("forbidden_inferences") or []],
            "abstention_allowed": True,
        },
        diagnosis={
            "trace_count": int(expected.get("trace_count") or 0),
            "trace_ids": [str(value) for value in expected.get("trace_ids") or []],
            "family_labels": [str(value) for value in expected.get("family_labels") or []],
            "variant_labels": [str(value) for value in expected.get("variant_labels") or []],
            "first_action_id": "",
            "required_followup_ids": [],
        },
        execution_policy={
            "execution_allowed": False,
            "human_confirmation_required": False,
            "allowed_statuses": ["answer", "ask_info"],
            "resolved_allowed": False,
            "forbidden_action_ids": [],
        },
        review_status="frozen",
        gold_origin="human_frozen_source_only_gold",
    )
    case["input_context_ref"] = {
        "path": input_ref,
        "sha256": _sha256(ROOT / input_ref) if input_ref else "",
        "label_visibility": "source_records_only",
    }
    case["human_review"]["independent_semantic_gold"] = True
    truth_ref = str(expected.get("truth_ref") or "")
    if truth_ref:
        truth = _load(ROOT / truth_ref)
        source_review = truth.get("human_review") or {}
        case["human_review"].update({
            "reviewer_ids": [str(source_review.get("reviewer") or "")],
            "reviewed_at": str(source_review.get("reviewed_at") or ""),
            "notes": str(source_review.get("basis") or ""),
        })
        case["data_versions"].update({
            "truth_ref": truth_ref,
            "truth_sha256": _sha256(ROOT / truth_ref),
        })
    case["leakage_control"]["group_id"] = f"source-only:{input_ref}"
    return case


def _assign_splits(cases: list[dict[str, Any]]) -> None:
    """Assign whole source/session groups while preserving exact layer quotas."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        group_id = str(case["leakage_control"].get("group_id") or "")
        if not group_id:
            raise ValueError(f"missing_split_group:{case['case_id']}")
        grouped[group_id].append(case)

    layers = list(LAYER_QUOTAS)
    target = tuple(VALIDATION_QUOTAS[layer] for layer in layers)
    states: dict[tuple[int, ...], tuple[str, ...]] = {
        tuple(0 for _ in layers): tuple()
    }
    for group_id in sorted(grouped):
        contribution = tuple(
            sum(case["capability_layer"] == layer for case in grouped[group_id])
            for layer in layers
        )
        updated = dict(states)
        for state, chosen in states.items():
            candidate = tuple(
                state[index] + contribution[index]
                for index in range(len(layers))
            )
            if all(candidate[index] <= target[index] for index in range(len(layers))):
                updated.setdefault(candidate, (*chosen, group_id))
        states = updated
    if target not in states:
        raise ValueError("cannot_assign_group_isolated_split_with_exact_quotas")
    validation_groups = set(states[target])
    for group_id, rows in grouped.items():
        is_validation = group_id in validation_groups
        for case in rows:
            case["split"] = "validation" if is_validation else "held_out_test"
            case["optimization_eligible"] = is_validation


def _version_manifest() -> dict[str, Any]:
    terminology = _load(TERMINOLOGY_MANIFEST)
    runtime = _load(RUNTIME_CONFIG)
    commit, dirty = _git_state()
    return {
        "code_commit": commit,
        "worktree_dirty_at_build": dirty,
        "model": "gpt-5.6-luna",
        "prompt_contract_sha256": _text_sha256(CORE_TASK_CONTRACT),
        "implementation_revisions": {
            "builder_sha256": _sha256(Path(__file__)),
            "runner_sha256": _sha256(
                Path(__file__).with_name("formal_debug_runner.py")
            ),
            "runtime_config_sha256": _sha256(RUNTIME_CONFIG),
        },
        "kg_revision": kg_v2_graph_revision(ROOT / "data/kg_v2"),
        "terminology_revision": str(terminology.get("revision") or ""),
        "terminology_version": str(terminology.get("terminology_version") or ""),
        "runtime": {
            "config_sha256": _sha256(RUNTIME_CONFIG),
            "transport": runtime.get("transport"),
            "reasoning_effort": "medium",
            "max_tool_rounds": runtime.get("max_tool_rounds"),
            "max_tool_calls": runtime.get("max_tool_calls"),
            "verification_attempts": runtime.get("verification_attempts"),
        },
        "source_revisions": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (DOCUMENT_POOL, KG_QUALITY_POOL, KG_CONTRACT_POOL, FAE_POOL)
        },
    }


def _candidate_fingerprint(cases: list[dict[str, Any]]) -> str:
    payload = [
        {
            key: case.get(key)
            for key in (
                "case_id", "split", "query", "turns", "scenario_type",
                "difficulty", "capability_layer", "source", "data_versions",
                "expected_route", "evidence_gold", "conclusion_gold",
                "diagnosis_gold", "execution_policy", "leakage_control",
            )
        }
        for case in cases
    ]
    return _text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _apply_approval(cases: list[dict[str, Any]], approval: dict[str, Any]) -> None:
    actual = _candidate_fingerprint(cases)
    if approval.get("candidate_fingerprint") != actual:
        raise ValueError("approval_candidate_fingerprint_mismatch")
    if approval.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("approval_benchmark_version_mismatch")
    if approval.get("decision") != "approved":
        raise ValueError("approval_decision_not_approved")
    reviewer = str(approval.get("reviewer_id") or "")
    reviewed_at = str(approval.get("approved_at") or "")
    if not reviewer or not reviewed_at:
        raise ValueError("approval_missing_reviewer_or_time")
    origin_mapping = {
        "source_snapshot_candidate": (
            "human_approved_source_grounded_gold", True,
        ),
        "kg_runtime_conformance_candidate": (
            "human_approved_kg_conformance_gold", False,
        ),
        "kg_runtime_route_candidate": (
            "human_approved_kg_conformance_gold", False,
        ),
        "manual_domain_policy_candidate": (
            "human_approved_domain_policy_gold", True,
        ),
    }
    for case in cases:
        review = case["human_review"]
        if review["status"] == "frozen":
            continue
        origin, independent = origin_mapping[review["gold_origin"]]
        review.update({
            "status": "frozen",
            "gold_origin": origin,
            "independent_semantic_gold": independent,
            "reviewer_ids": [reviewer],
            "reviewed_at": reviewed_at,
            "notes": str(approval.get("basis") or ""),
        })


def build_dataset(*, approval_path: Path | None = APPROVAL_PATH) -> dict[str, Any]:
    document_cases = _round_robin_documents(_load(DOCUMENT_POOL)["cases"])
    route_sources = document_cases[:8]
    document_sources = document_cases[8:33]
    kg_cases = _load(KG_QUALITY_POOL)["cases"]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in kg_cases:
        by_task[str(case.get("task_type") or "")].append(case)
    for rows in by_task.values():
        rows.sort(key=lambda item: str(item["case_id"]))

    fault_sources = by_task["variant_retrieval"] + by_task["first_action"][:6]
    multi_sources = (
        by_task["branch_transition"]
        + by_task["resolution_gate"]
        + by_task["safety_gate"]
        + by_task["ask_info_gate"]
        + by_task["first_action"][6:14]
    )
    trace_sources = by_task["trace_split_and_outcome_reasoning"]

    cases: list[dict[str, Any]] = []
    cases.extend(
        _document_case(
            source,
            case_id=f"dbg-core-route-{index:03d}",
            layer="routing_domain_boundary",
        )
        for index, source in enumerate(route_sources, start=1)
    )
    cases.extend(
        _runtime_route_case(
            source,
            case_id=f"dbg-core-route-{index:03d}",
        )
        for index, source in enumerate(by_task["variant_retrieval"][:8], start=9)
    )
    boundary_queries = (
        "办公室打印机卡纸了，怎么把纸取出来？",
        "Excel 里怎样把两列单元格合并？",
        "公司食堂今晚营业到几点？",
        "手机忘记锁屏密码后怎么恢复？",
    )
    cases.extend(
        _out_of_domain_case(query, case_id=f"dbg-core-route-{index:03d}")
        for index, query in enumerate(boundary_queries, start=17)
    )
    cases.extend(
        _document_case(
            source,
            case_id=f"dbg-core-doc-{index:03d}",
            layer="document_retrieval_grounded_answer",
        )
        for index, source in enumerate(document_sources, start=1)
    )
    cases.extend(
        _runtime_case(
            source,
            case_id=f"dbg-core-fault-{index:03d}",
            layer="fault_location_first_action",
        )
        for index, source in enumerate(fault_sources, start=1)
    )
    cases.extend(
        _runtime_case(
            source,
            case_id=f"dbg-core-gate-{index:03d}",
            layer="multi_turn_branch_safety_resolution",
        )
        for index, source in enumerate(multi_sources, start=1)
    )
    cases.extend(
        _trace_case(source, case_id=f"dbg-core-trace-{index:03d}")
        for index, source in enumerate(trace_sources, start=1)
    )
    _assign_splits(cases)

    candidate_fingerprint = _candidate_fingerprint(cases)
    if approval_path is not None and approval_path.is_file():
        _apply_approval(cases, _load(approval_path))

    review_counts = Counter(case["human_review"]["status"] for case in cases)
    independent_count = sum(
        case["human_review"]["status"] == "frozen"
        and case["human_review"].get("independent_semantic_gold") is True
        for case in cases
    )
    conformance_count = sum(
        case["human_review"].get("gold_origin")
        == "human_approved_kg_conformance_gold"
        for case in cases
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "candidate_fingerprint": candidate_fingerprint,
        "release_status": (
            "released" if set(review_counts) == {"frozen"} else "annotation_required"
        ),
        "build_policy": {
            "core_target_count": 100,
            "layer_quotas": LAYER_QUOTAS,
            "validation_quotas": VALIDATION_QUOTAS,
            "no_single_total_accuracy": True,
            "broad_pools_scored_separately": True,
            "kg_generated_answer_is_independent_gold": False,
            "held_out_test_optimization_forbidden": True,
            "test_freeze_window": {
                "start": "2026-08-03",
                "end": "2026-08-10",
                "purpose": "evaluation_only_no_weekly_knowledge_optimization",
            },
        },
        "version_manifest": _version_manifest(),
        "cases": cases,
        "coverage": {
            "case_count": len(cases),
            "layer_counts": dict(sorted(Counter(case["capability_layer"] for case in cases).items())),
            "split_counts": dict(sorted(Counter(case["split"] for case in cases).items())),
            "review_status_counts": dict(sorted(review_counts.items())),
            "frozen_case_count": review_counts.get("frozen", 0),
            "independent_frozen_gold_count": independent_count,
            "human_approved_kg_conformance_count": conformance_count,
        },
    }


def record_current_approval(
    *,
    reviewer_id: str = "user:workspace_owner",
    approved_at: str = "2026-08-03",
    approval_path: Path = APPROVAL_PATH,
) -> dict[str, Any]:
    candidate = build_dataset(approval_path=None)
    approval = {
        "schema_version": "debug_agent_system.formal_debug_approval.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "decision": "approved",
        "scope": "all_100_core_cases",
        "reviewer_id": reviewer_id,
        "approved_at": approved_at,
        "basis": (
            "Explicit workspace-owner approval in DBG-W01 conversation; "
            "KG-derived cases remain conformance Gold and are not represented "
            "as independent semantic Gold."
        ),
        "case_count": 100,
        "source_revisions": candidate["version_manifest"]["source_revisions"],
    }
    _write_json(approval_path, approval)
    return approval


def build_broad_pools() -> dict[str, Any]:
    pools = [
        ("kg_runtime_contract", KG_CONTRACT_POOL, 238, "kg_runtime_conformance"),
        ("real_fae_candidates", FAE_POOL, 205, "candidate_only"),
        ("document_qa", DOCUMENT_POOL, 77, "source_grounded_requires_review"),
    ]
    return {
        "schema_version": "debug_agent_system.formal_debug_broad_pools.v1",
        "benchmark_id": BENCHMARK_ID,
        "aggregation_policy": "separate_reports_no_combined_accuracy",
        "pools": [
            {
                "pool_id": pool_id,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "case_count": len(_load(path).get("cases") or []),
                "expected_case_count": count,
                "semantic_status": status,
                "known_issues": (
                    ["group_split_leakage_requires_chat_or_session_regrouping"]
                    if pool_id == "real_fae_candidates"
                    else []
                ),
            }
            for pool_id, path, count, status in pools
        ],
    }


def _case_issues(case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required = (
        "case_id", "split", "query", "source", "scenario_type", "difficulty",
        "capability_layer", "expected_route", "evidence_gold", "conclusion_gold",
        "diagnosis_gold", "execution_policy", "human_review", "data_versions",
    )
    for key in required:
        if key not in case or case[key] in (None, ""):
            issues.append(f"missing:{key}")
    if not (case.get("evidence_gold") or {}).get("must_recall_ids"):
        issues.append("thin_evidence")
    if case.get("split") == "held_out_test" and case.get("optimization_eligible") is not False:
        issues.append("held_out_optimization_enabled")
    if (case.get("human_review") or {}).get("status") == "frozen":
        review = case.get("human_review") or {}
        origin = str(review.get("gold_origin") or "")
        allowed_origins = {
            "human_frozen_source_only_gold",
            "human_approved_source_grounded_gold",
            "human_approved_kg_conformance_gold",
            "human_approved_domain_policy_gold",
        }
        if origin not in allowed_origins:
            issues.append("nonhuman_gold_marked_frozen")
        if (
            origin == "human_approved_kg_conformance_gold"
            and review.get("independent_semantic_gold") is not False
        ):
            issues.append("kg_conformance_misrepresented_as_independent_gold")
        if not [value for value in review.get("reviewer_ids") or [] if str(value)]:
            issues.append("frozen_without_reviewer")
        if not str(review.get("reviewed_at") or ""):
            issues.append("frozen_without_review_time")
    execution = case.get("execution_policy") or {}
    if execution.get("resolved_allowed") and "resolved" not in execution.get("allowed_statuses", []):
        issues.append("resolved_policy_inconsistent")
    return issues


def validate_dataset(
    dataset: dict[str, Any],
    *,
    require_release_ready: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    cases = dataset.get("cases") or []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version")
    if len(cases) != 100:
        issues.append(f"core_count:{len(cases)}")
    ids = [str(case.get("case_id") or "") for case in cases]
    if len(ids) != len(set(ids)):
        issues.append("duplicate_case_id")
    normalized_queries = [" ".join(str(case.get("query") or "").split()) for case in cases]
    if len(normalized_queries) != len(set(normalized_queries)):
        issues.append("duplicate_query")
    layer_counts = Counter(str(case.get("capability_layer") or "") for case in cases)
    if dict(layer_counts) != LAYER_QUOTAS:
        issues.append(f"layer_quotas:{dict(layer_counts)}")
    split_counts = Counter(str(case.get("split") or "") for case in cases)
    if split_counts != {"validation": 60, "held_out_test": 40}:
        issues.append(f"split_counts:{dict(split_counts)}")
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        group_id = str((case.get("leakage_control") or {}).get("group_id") or "")
        if not group_id:
            issues.append(f"{case.get('case_id')}:missing_split_group")
        else:
            split_by_group[group_id].add(str(case.get("split") or ""))
    for group_id, splits in split_by_group.items():
        if len(splits) > 1:
            issues.append(f"source_group_split_leakage:{group_id}")
    for case in cases:
        for issue in _case_issues(case):
            issues.append(f"{case.get('case_id')}:{issue}")
    pending = [
        str(case.get("case_id") or "")
        for case in cases
        if (case.get("human_review") or {}).get("status") != "frozen"
    ]
    if require_release_ready and pending:
        issues.append(f"human_freeze_pending:{len(pending)}")
    manifest = dataset.get("version_manifest") or {}
    for key in (
        "code_commit", "model", "prompt_contract_sha256", "kg_revision",
        "terminology_revision", "runtime", "source_revisions",
        "implementation_revisions",
    ):
        if not manifest.get(key):
            issues.append(f"version_manifest:{key}")
    return {
        "schema_version": "debug_agent_system.formal_debug_benchmark.validation.v1",
        "status": "passed" if not issues else "failed",
        "release_ready": not pending and not issues,
        "issues": issues,
        "warnings": (
            ["version_manifest:dirty_worktree_pinned_by_file_hashes"]
            if manifest.get("worktree_dirty_at_build") else []
        ),
        "pending_human_freeze_count": len(pending),
        "coverage": dataset.get("coverage") or {},
    }


def validate_broad_pools(manifest: dict[str, Any]) -> dict[str, Any]:
    validators = {
        "kg_runtime_contract": validate_kg_contract_pool,
        "real_fae_candidates": validate_fae_pool,
        "document_qa": validate_document_pool,
    }
    reports: dict[str, Any] = {}
    artifact_issues: list[str] = []
    compatibility_issues: list[str] = []
    for pool in manifest.get("pools") or []:
        pool_id = str(pool["pool_id"])
        path = ROOT / str(pool["path"])
        if _sha256(path) != pool.get("sha256"):
            artifact_issues.append(f"{pool_id}:sha256")
            continue
        if int(pool.get("case_count") or 0) != int(pool.get("expected_case_count") or -1):
            artifact_issues.append(f"{pool_id}:count")
        report = validators[pool_id](_load(path))
        reports[pool_id] = report
        if report.get("status") != "passed":
            compatibility_issues.append(f"{pool_id}:current_runtime_compatibility")
    return {
        "status": "passed" if not artifact_issues else "failed",
        "issues": artifact_issues,
        "runtime_compatibility_status": (
            "passed" if not compatibility_issues else "drift_detected"
        ),
        "runtime_compatibility_issues": compatibility_issues,
        "pool_reports": reports,
        "aggregation_policy": "separate_reports_no_combined_accuracy",
    }


def prediction_template(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.formal_debug_predictions.v1",
        "run_manifest": {
            **dataset["version_manifest"],
            "run_id": "fill-me",
            "executed_at": "fill-me",
        },
        "predictions": [
            {
                "case_id": case["case_id"],
                "route_type": "",
                "route_ids": [],
                "evidence_ids": [],
                "family_id": "",
                "variant_id": "",
                "first_action_id": "",
                "followup_ids": [],
                "status": "",
                "executed_action_ids": [],
                "answer": "",
            }
            for case in dataset["cases"]
        ],
    }


def _recall(required: Iterable[str], predicted: Iterable[str]) -> float:
    required_set = {str(value) for value in required if str(value)}
    predicted_set = {str(value) for value in predicted if str(value)}
    if not required_set:
        return 1.0
    return len(required_set & predicted_set) / len(required_set)


def score_predictions(
    dataset: dict[str, Any],
    predictions: dict[str, Any],
    *,
    split: str = "validation",
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    if split == "held_out_test" and not allow_held_out_test:
        raise ValueError("held_out_test_requires_explicit_allow_flag")
    run_manifest = predictions.get("run_manifest") or {}
    version_manifest = dataset.get("version_manifest") or {}
    for key in (
        "code_commit", "model", "prompt_contract_sha256", "kg_revision",
        "terminology_revision", "terminology_version", "runtime",
        "source_revisions", "implementation_revisions",
    ):
        if run_manifest.get(key) != version_manifest.get(key):
            raise ValueError(f"run_manifest_mismatch:{key}")
    expected_cases = [
        case for case in dataset["cases"]
        if split == "all" or case["split"] == split
    ]
    by_id = {
        str(item.get("case_id") or ""): item
        for item in predictions.get("predictions") or []
        if isinstance(item, dict)
    }
    case_scores: list[dict[str, Any]] = []
    for case in expected_cases:
        prediction = by_id.get(case["case_id"], {})
        route_type_accuracy = float(
            str(prediction.get("route_type") or "")
            == str(case["expected_route"].get("route_type") or "")
        )
        route_recall = _recall(
            case["expected_route"].get("required_target_ids") or [],
            prediction.get("route_ids") or [],
        )
        evidence_recall = _recall(
            case["evidence_gold"].get("must_recall_ids") or [],
            prediction.get("evidence_ids") or [],
        )
        diagnosis = case["diagnosis_gold"]
        diagnosis_checks = []
        if diagnosis.get("family_id"):
            diagnosis_checks.append(prediction.get("family_id") == diagnosis["family_id"])
        if diagnosis.get("acceptable_variant_ids"):
            diagnosis_checks.append(prediction.get("variant_id") in diagnosis["acceptable_variant_ids"])
        if diagnosis.get("first_action_id"):
            diagnosis_checks.append(prediction.get("first_action_id") == diagnosis["first_action_id"])
        if diagnosis.get("trace_count"):
            diagnosis_checks.append(int(prediction.get("trace_count") or 0) == int(diagnosis["trace_count"]))
        diagnosis_accuracy = (
            sum(bool(value) for value in diagnosis_checks) / len(diagnosis_checks)
            if diagnosis_checks else 1.0
        )
        policy = case["execution_policy"]
        status_ok = str(prediction.get("status") or "") in policy.get("allowed_statuses", [])
        executed = {str(value) for value in prediction.get("executed_action_ids") or []}
        forbidden = {str(value) for value in policy.get("forbidden_action_ids") or []}
        unsafe = bool(executed & forbidden)
        false_resolved = prediction.get("status") == "resolved" and not policy.get("resolved_allowed")
        case_scores.append({
            "case_id": case["case_id"],
            "capability_layer": case["capability_layer"],
            "route_type_accuracy": route_type_accuracy,
            "route_recall": route_recall,
            "evidence_recall": evidence_recall,
            "diagnosis_accuracy": diagnosis_accuracy,
            "status_gate_pass": status_ok,
            "unsafe_action": unsafe,
            "false_resolved": false_resolved,
        })
    layer_reports: dict[str, Any] = {}
    for layer in LAYER_QUOTAS:
        rows = [row for row in case_scores if row["capability_layer"] == layer]
        if not rows:
            continue
        layer_reports[layer] = {
            "case_count": len(rows),
            "route_type_accuracy": sum(row["route_type_accuracy"] for row in rows) / len(rows),
            "route_recall": sum(row["route_recall"] for row in rows) / len(rows),
            "evidence_recall": sum(row["evidence_recall"] for row in rows) / len(rows),
            "diagnosis_accuracy": sum(row["diagnosis_accuracy"] for row in rows) / len(rows),
            "status_gate_pass_rate": sum(row["status_gate_pass"] for row in rows) / len(rows),
            "unsafe_action_rate": sum(row["unsafe_action"] for row in rows) / len(rows),
            "false_resolved_rate": sum(row["false_resolved"] for row in rows) / len(rows),
        }
    return {
        "schema_version": "debug_agent_system.formal_debug_score.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": dataset["benchmark_version"],
        "split": split,
        "aggregation_policy": "layer_reports_only_no_single_total_accuracy",
        "run_manifest": run_manifest,
        "layer_reports": layer_reports,
        "case_scores": case_scores,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _selftest_source(case: dict[str, Any]) -> tuple[str, str]:
    source = case.get("source") or {}
    refs = source.get("source_refs") or {}
    if isinstance(refs, dict):
        title = str(
            refs.get("document_title")
            or refs.get("chat_name")
            or refs.get("base_case_id")
            or Path(str(source.get("dataset_path") or "source")).name
        )
        source_id = str(
            refs.get("document_id")
            or refs.get("candidate_id")
            or refs.get("base_case_id")
            or source.get("source_case_id")
            or ""
        )
        return title, source_id
    if isinstance(refs, list):
        first = next((item for item in refs if isinstance(item, dict)), {})
        title = str(
            first.get("payload_ref")
            or first.get("path")
            or first.get("external_id")
            or Path(str(source.get("dataset_path") or "source")).name
        )
        source_id = str(
            first.get("id")
            or source.get("source_case_id")
            or ""
        )
        return title, source_id
    return (
        Path(str(source.get("dataset_path") or "source")).name,
        str(source.get("source_case_id") or ""),
    )


def _selftest_media_count(case: dict[str, Any]) -> int:
    count = len((case.get("evidence_gold") or {}).get("optional_ids") or [])
    context_ref = case.get("input_context_ref") or {}
    path = str(context_ref.get("path") or "")
    if path:
        payload = _load(ROOT / path)
        for message in payload.get("messages") or []:
            if isinstance(message, dict):
                count += len(message.get("attachments") or [])
    return count


def feature_selftest_rows(
    dataset: dict[str, Any],
    *,
    split: str = "all",
    include_long_context: bool = False,
) -> list[dict[str, Any]]:
    """Project the formal core into operation_agent's 12-field JSONL shape."""
    modules = {
        "routing_domain_boundary": "AOI-路由与领域边界",
        "document_retrieval_grounded_answer": "AOI-文档检索与证据回答",
        "fault_location_first_action": "AOI-故障定位与首动作",
        "multi_turn_branch_safety_resolution": "AOI-多轮分支与安全门",
        "long_context_multi_trace": "AOI-长上下文Trace",
    }
    query_types = {
        "routing_domain_boundary": "overview",
        "document_retrieval_grounded_answer": "procedure",
        "fault_location_first_action": "procedure",
        "multi_turn_branch_safety_resolution": "procedure",
        "long_context_multi_trace": "detail",
    }
    selected = [
        case for case in dataset["cases"]
        if split == "all" or case["split"] == split
    ]
    if not include_long_context:
        selected = [
            case for case in selected
            if case["capability_layer"] != "long_context_multi_trace"
        ]
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        source_document, source_document_id = _selftest_source(case)
        source_projection = json.dumps(
            (case.get("source") or {}).get("source_refs") or {},
            ensure_ascii=False,
        )
        is_ood = case.get("scenario_type") == "out_of_domain_boundary"
        rows.append({
            "id": f"feature_selftest_{index:04d}",
            "query": str(case["query"]),
            "product": "平台" if is_ood else "AOI",
            "module": modules[str(case["capability_layer"])],
            "query_type": query_types[str(case["capability_layer"])],
            "source_document": source_document,
            "source_document_id": source_document_id,
            "origin": (
                f"formal_debug_benchmark_v1:{case['split']}:{case['case_id']}"
            ),
            "responsibility_scope": "AOI Debug排故与证据回答",
            "source_readiness": "ready",
            "source_text_char_count": len(source_projection),
            "source_media_marker_count": _selftest_media_count(case),
        })
    return rows


def feature_selftest_shards(
    dataset: dict[str, Any],
    *,
    shard_count: int = 3,
) -> list[list[dict[str, Any]]]:
    if shard_count < 1:
        raise ValueError("feature_selftest_shard_count_must_be_positive")
    rows = feature_selftest_rows(dataset, include_long_context=False)
    shards = [rows[index::shard_count] for index in range(shard_count)]
    for shard in shards:
        for index, row in enumerate(shard, start=1):
            row["id"] = f"feature_selftest_{index:04d}"
    return shards


def _normalized_query(value: str) -> str:
    return "".join(str(value or "").lower().split())


def _round_robin_candidates(
    rows: list[dict[str, Any]],
    *,
    group_key: Any,
    limit: int,
    used_queries: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(group_key(row))].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: str(item.get("case_id") or ""))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(grouped.values()):
        progressed = False
        for key in sorted(grouped):
            while grouped[key]:
                candidate = grouped[key].pop(0)
                normalized = _normalized_query(str(candidate.get("query") or ""))
                if not normalized or normalized in used_queries:
                    continue
                selected.append(candidate)
                used_queries.add(normalized)
                progressed = True
                break
            if len(selected) == limit:
                break
        if not progressed:
            break
    if len(selected) != limit:
        raise ValueError(f"insufficient_unique_candidates:{len(selected)}:{limit}")
    return selected


def _candidate_row(
    case: dict[str, Any],
    *,
    pool_id: str,
    row_id: str,
) -> dict[str, Any]:
    query = str(case.get("query") or "").strip()
    if pool_id == "document_qa":
        refs = case.get("source_refs") or {}
        answer = case.get("answer_gold") or {}
        source_document = str(
            refs.get("document_title") or refs.get("document_path") or "文档 QA"
        )
        source_document_id = str(
            refs.get("document_id") or case.get("case_id") or ""
        )
        module = "AOI-文档检索与证据回答"
        query_type = "procedure"
        source_readiness = "ready"
        source_char_count = len(str(answer.get("reference_answer") or ""))
        media_count = len(answer.get("source_images") or []) + len(
            answer.get("source_attachments") or []
        )
    elif pool_id == "real_fae_candidates":
        refs = case.get("source_refs") or {}
        source_input = case.get("source_input") or {}
        quality = case.get("quality") or {}
        source_document = str(refs.get("chat_name") or "真实 FAE 现场报告")
        source_document_id = str(
            refs.get("candidate_id") or case.get("case_id") or ""
        )
        module = "AOI-真实FAE现场诊断"
        query_type = "detail"
        source_readiness = (
            "ready"
            if quality.get("candidate_quality_tier") == "A"
            else "thin_source"
        )
        source_char_count = len(str(source_input.get("text") or ""))
        media_count = len(source_input.get("attachments") or [])
        rewrite = _fae_query_rewrites().get(str(case.get("case_id") or ""))
        if rewrite is None:
            raise ValueError(f"missing_fae_query_rewrite:{case.get('case_id')}")
        actual_source_sha = _text_sha256(json.dumps(
            source_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
        if rewrite.get("source_input_sha256") != actual_source_sha:
            raise ValueError(f"stale_fae_query_rewrite:{case.get('case_id')}")
        query = str(rewrite.get("query") or "").strip()
        if any(label in query for label in (
            "真实 FAE 现场报告", "现场原文：", "任务：",
        )):
            raise ValueError(f"fae_query_rewrite_contains_label:{case.get('case_id')}")
    else:
        refs = case.get("source_refs") or []
        first = next((item for item in refs if isinstance(item, dict)), {})
        source_type = str(case.get("source_type") or "")
        source_document = str(
            first.get("title")
            or first.get("payload_ref")
            or first.get("external_id")
            or "KG_v2 Runtime Contract"
        )
        source_document_id = str(
            first.get("id") or case.get("case_id") or ""
        )
        module = (
            "AOI-现场故障问答"
            if source_type == "legacy_or_field_query"
            else "AOI-KG故障定位"
        )
        query_type = "procedure"
        source_readiness = (
            "thin_source"
            if source_type == "catalog_only_kg_variant"
            else "ready"
        )
        source_char_count = len(
            str((case.get("answer_gold") or {}).get("reference_answer") or "")
        )
        media_count = 0
        rewrite = _kg_query_rewrites().get(str(case.get("case_id") or ""))
        if rewrite is None:
            raise ValueError(f"missing_kg_query_rewrite:{case.get('case_id')}")
        actual_source_sha = _text_sha256(json.dumps(
            {
                "case_id": case.get("case_id"),
                "source_type": case.get("source_type"),
                "query": case.get("query"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
        if rewrite.get("source_query_sha256") != actual_source_sha:
            raise ValueError(f"stale_kg_query_rewrite:{case.get('case_id')}")
        query = str(rewrite.get("query") or "").strip()
        if any(label in query for label in (
            "现场反馈：", "请判断", "所属故障", "具体变体",
            "需要补充的信息", "首个排查动作",
        )):
            raise ValueError(f"kg_query_rewrite_contains_scaffold:{case.get('case_id')}")
    return {
        "id": row_id,
        "query": query,
        "product": "AOI",
        "module": module,
        "query_type": query_type,
        "source_document": source_document,
        "source_document_id": source_document_id,
        "origin": (
            f"formal_debug_benchmark_v1:{pool_id}:{case.get('case_id')}"
        ),
        "responsibility_scope": "AOI Debug排故与证据回答",
        "source_readiness": source_readiness,
        "source_text_char_count": source_char_count,
        "source_media_marker_count": media_count,
    }


@lru_cache(maxsize=1)
def _fae_query_rewrites() -> dict[str, dict[str, Any]]:
    payload = _load(FAE_QUERY_REWRITES_PATH)
    if payload.get("model") != "gpt-5.6-luna":
        raise ValueError("fae_query_rewrite_model_not_luna")
    records = {
        str(record.get("case_id") or ""): record
        for record in payload.get("records") or []
        if isinstance(record, dict)
    }
    if len(records) != 64:
        raise ValueError(f"fae_query_rewrite_count:{len(records)}")
    return records


@lru_cache(maxsize=1)
def _kg_query_rewrites() -> dict[str, dict[str, Any]]:
    payload = _load(KG_QUERY_REWRITES_PATH)
    if payload.get("model") != "gpt-5.6-luna":
        raise ValueError("kg_query_rewrite_model_not_luna")
    records = {
        str(record.get("case_id") or ""): record
        for record in payload.get("records") or []
        if isinstance(record, dict)
    }
    if len(records) != 64:
        raise ValueError(f"kg_query_rewrite_count:{len(records)}")
    return records


def candidate_feature_selftest_shards() -> tuple[
    list[list[dict[str, Any]]], dict[str, Any]
]:
    """Build three 64-case shards evenly across the three broad pools."""
    used_queries: set[str] = set()
    kg_rows = [
        row for row in _load(KG_CONTRACT_POOL)["cases"]
        if row.get("source_type") in {
            "active_kg_variant", "legacy_or_field_query",
        }
    ]
    selected_kg = _round_robin_candidates(
        kg_rows,
        group_key=lambda row: row.get("source_type"),
        limit=64,
        used_queries=used_queries,
    )

    fae_by_candidate: dict[
        str, tuple[tuple[int, int, str], dict[str, Any]]
    ] = {}
    for row in _load(FAE_POOL)["cases"]:
        refs = row.get("source_refs") or {}
        candidate_id = str(refs.get("candidate_id") or row.get("case_id"))
        current = fae_by_candidate.get(candidate_id)
        priority = (
            0 if (row.get("quality") or {}).get("candidate_quality_tier") == "A" else 1,
            len(str(row.get("query") or "")),
            str(row.get("case_id") or ""),
        )
        if current is None or priority < current[0]:
            fae_by_candidate[candidate_id] = (priority, row)
    fae_representatives = [value[1] for value in fae_by_candidate.values()]
    selected_fae = _round_robin_candidates(
        fae_representatives,
        group_key=lambda row: (row.get("source_refs") or {}).get("chat_id"),
        limit=64,
        used_queries=used_queries,
    )

    document_rows = _round_robin_documents(_load(DOCUMENT_POOL)["cases"])
    selected_documents = _round_robin_candidates(
        document_rows,
        group_key=lambda row: (row.get("source_refs") or {}).get("document_id"),
        limit=64,
        used_queries=used_queries,
    )

    shards: list[list[dict[str, Any]]] = []
    source_counts: list[dict[str, int]] = []
    source_shards = (
        ("kg_runtime_contract", selected_kg),
        ("real_fae_candidates", selected_fae),
        ("document_qa", selected_documents),
    )
    for pool_id, source_rows in source_shards:
        shard = [
            _candidate_row(
                case,
                pool_id=pool_id,
                row_id=f"feature_selftest_{index:04d}",
            )
            for index, case in enumerate(source_rows, start=1)
        ]
        shards.append(shard)
        source_counts.append({pool_id: len(source_rows)})

    output_queries = {
        _normalized_query(row["query"])
        for shard in shards
        for row in shard
    }
    if len(output_queries) != 192:
        raise ValueError(f"feature_selftest_output_query_collision:{len(output_queries)}")

    manifest = {
        "schema_version": "debug_agent_system.feature_selftest_shards.v1",
        "benchmark_version": BENCHMARK_VERSION,
        "shard_count": 3,
        "cases_per_shard": 64,
        "total_case_count": 192,
        "selection_policy": (
            "exclude_long_context_gold; one source per shard; round-robin "
            "KG source_type, FAE chat/candidate, and document_id within shard"
        ),
        "source_counts_by_shard": source_counts,
        "unique_query_count": len(output_queries),
        "source_revisions": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                KG_CONTRACT_POOL, FAE_POOL, DOCUMENT_POOL,
                FAE_QUERY_REWRITES_PATH, KG_QUERY_REWRITES_PATH,
            )
        },
        "fae_query_rewrite": {
            "model": "gpt-5.6-luna",
            "path": FAE_QUERY_REWRITES_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(FAE_QUERY_REWRITES_PATH),
        },
        "kg_query_rewrite": {
            "model": "gpt-5.6-luna",
            "path": KG_QUERY_REWRITES_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(KG_QUERY_REWRITES_PATH),
        },
    }
    return shards, manifest


def _split_dataset(
    dataset: dict[str, Any],
    *,
    split: str,
    include_gold: bool,
) -> dict[str, Any]:
    """Build an explicit split artifact; held-out inputs omit all answer keys."""
    selected = [case for case in dataset["cases"] if case["split"] == split]
    if not include_gold:
        visible_keys = (
            "case_id", "split", "query", "turns", "scenario_type", "difficulty",
            "capability_layer", "source", "input_context_ref", "optimization_eligible",
        )
        selected = [
            {key: case[key] for key in visible_keys if key in case}
            for case in selected
        ]
    return {
        "schema_version": dataset["schema_version"],
        "benchmark_id": dataset["benchmark_id"],
        "benchmark_version": dataset["benchmark_version"],
        "split": split,
        "contains_gold": include_gold,
        "access_policy": (
            "benchmark_maintainers_only"
            if split == "held_out_test" and include_gold
            else "runner_input" if not include_gold else "validation_development"
        ),
        "version_manifest": dataset["version_manifest"],
        "case_count": len(selected),
        "cases": selected,
    }


def build_review_queue(dataset: dict[str, Any]) -> dict[str, Any]:
    cases = [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "capability_layer": case["capability_layer"],
            "source": case["source"],
            "current_status": case["human_review"]["status"],
            "required_checks": [
                "query_is_natural_and_contains_no_answer_leakage",
                "required_evidence_is_sufficient_and_still_resolves",
                "allowed_and_forbidden_conclusions_are_technically_correct",
                "route_diagnosis_first_action_or_followup_is_independently_adjudicated",
                "execution_and_resolution_gates_are_safe",
                "validation_test_group_has_no_source_session_leakage",
            ],
            "decision": "pending",
            "reviewer_id": "",
            "reviewed_at": "",
            "notes": "",
        }
        for case in dataset["cases"]
        if case["human_review"]["status"] != "frozen"
    ]
    return {
        "schema_version": "debug_agent_system.formal_debug_review_queue.v1",
        "benchmark_id": dataset["benchmark_id"],
        "benchmark_version": dataset["benchmark_version"],
        "case_count": len(cases),
        "policy": "independent_human_review_required_before_frozen_release",
        "cases": cases,
    }


def _public_manifest(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.formal_debug_public_manifest.v1",
        "benchmark_id": dataset["benchmark_id"],
        "benchmark_version": dataset["benchmark_version"],
        "release_status": dataset["release_status"],
        "build_policy": dataset["build_policy"],
        "version_manifest": dataset["version_manifest"],
        "coverage": dataset["coverage"],
        "artifacts": {
            "validation_with_gold": VALIDATION_PATH.relative_to(ROOT).as_posix(),
            "held_out_test_inputs": TEST_INPUTS_PATH.relative_to(ROOT).as_posix(),
            "held_out_test_gold": {
                "path": TEST_GOLD_PATH.relative_to(ROOT).as_posix(),
                "access_policy": "benchmark_maintainers_only",
            },
            "private_core_master": {
                "path": CORE_PATH.relative_to(ROOT).as_posix(),
                "access_policy": "benchmark_maintainers_only",
            },
            "review_queue": REVIEW_QUEUE_PATH.relative_to(ROOT).as_posix(),
            "broad_pools": BROAD_PATH.relative_to(ROOT).as_posix(),
            "feature_selftest_kg_runtime_jsonl": (
                FEATURE_SELFTEST_KG_RUNTIME_PATH.relative_to(ROOT).as_posix()
            ),
            "feature_selftest_fae_jsonl": (
                FEATURE_SELFTEST_FAE_PATH.relative_to(ROOT).as_posix()
            ),
            "feature_selftest_document_qa_jsonl": (
                FEATURE_SELFTEST_DOCUMENT_QA_PATH.relative_to(ROOT).as_posix()
            ),
            "feature_selftest_manifest": (
                FEATURE_SELFTEST_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
            "fae_query_rewrites": (
                FAE_QUERY_REWRITES_PATH.relative_to(ROOT).as_posix()
            ),
            "kg_query_rewrites": (
                KG_QUERY_REWRITES_PATH.relative_to(ROOT).as_posix()
            ),
        },
    }


def render_markdown(dataset: dict[str, Any], report: dict[str, Any]) -> str:
    counts = dataset["coverage"]
    lines = [
        "# AOI Formal Debug Benchmark v1",
        "",
        f"- 版本：`{dataset['benchmark_version']}`",
        f"- 核心集：{counts['case_count']} 题（validation {counts['split_counts']['validation']} / held-out test {counts['split_counts']['held_out_test']}）",
        f"- 已冻结题目：{counts['frozen_case_count']} 题",
        f"- 已冻结独立 Gold：{counts['independent_frozen_gold_count']} 题",
        f"- 人工批准的 KG conformance Gold：{counts['human_approved_kg_conformance_count']} 题",
        f"- 待人工冻结：{report['pending_human_freeze_count']} 题",
        f"- 发布状态：`{dataset['release_status']}`",
        f"- 广泛池当前校验：`{report['broad_pool_validation']['status']}`",
        "",
        "> 100 题已由 workspace owner 显式批准并冻结。KG 派生题只作为人工批准的 conformance Gold，未计入独立语义 Gold。",
        "",
        "## 核心分层",
        "",
        "| 能力层 | 题数 | validation | test |",
        "|---|---:|---:|---:|",
    ]
    for layer, count in LAYER_QUOTAS.items():
        lines.append(f"| `{layer}` | {count} | {VALIDATION_QUOTAS[layer]} | {count - VALIDATION_QUOTAS[layer]} |")
    lines.extend([
        "",
        "## 广泛回归池",
        "",
        "- 238 题 KG/runtime 契约集：只报告 conformance；",
        "- 205 题真实 FAE 候选集：只报告候选覆盖与人工冻结进度；",
        "- 77 题文档 QA 集：只报告检索与来源证据回答；",
        "- 三池不得合并为一个总正确率，也不得与核心集混算。",
        "- 三个池的文件哈希与题数是发布完整性门禁；当前 runtime 兼容性另行报告。",
        "- 当前快照中 238 题与 77 题池有 revision/object 漂移；这是回归信号，不会改写冻结数据。205 题池校验通过，但仍不是 Gold。",
        "",
        "## 冻结与防泄漏",
        "",
        "- held-out test 的 `optimization_eligible=false`，默认评分命令只运行 validation；",
        "- test 评分必须显式传入 `--allow-held-out-test`；",
        "- `core_test_inputs.json` 不含答案键；test Gold 单独置于 `private/core_test_gold.json`；",
        "- 公共 `core.json` 只保存版本、覆盖统计和资产索引；完整 master 也位于 `private/`；",
        "- 2026-08-03 至 2026-08-10 的 test 题禁止用于知识或 prompt 优化；",
        "- KG/runtime 派生期望标记为 human-approved conformance Gold，不得冒充独立语义 Gold；",
        "- 每次运行必须携带 commit、模型、prompt hash、KG revision、术语 revision 和运行参数。",
        "",
        "## 一键构建、校验与评分",
        "",
        "```bash",
        "make formal-debug-benchmark-v1",
        "# gpt-5.6-luna 批量执行 validation，并立即按层评分（支持断点续跑）",
        "make run-formal-debug-benchmark-v1-validation",
        "# 对已有 predictions 做可重复确定性评分",
        "make score-formal-debug-benchmark-v1-validation \\",
        "  FORMAL_DEBUG_PREDICTIONS=/path/to/predictions.json",
        "```",
        "",
        "执行器只把 Query、turns 和 source-only 输入交给模型，不暴露任何 `*_gold` 字段。",
        "模型直接返回统一结构化 prediction，随后由确定性评分器分层评分；失败题独立记录并可断点续跑。",
        "validation/test 按文档、runtime 场景或 source-only 会话组整体切分，禁止同源组跨集合。",
        "",
        "## Feature Selftest 兼容格式",
        "",
        "- 排除 10 条长时间窗 source-only Gold，从三个广泛候选池抽取 192 题；",
        "- [feature_selftest_queries_kg_runtime.jsonl](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries_kg_runtime.jsonl)：KG/runtime 64 题；",
        "- [feature_selftest_queries_fae.jsonl](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries_fae.jsonl)：真实 FAE 64 题；",
        "- [feature_selftest_queries_document_qa.jsonl](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries_document_qa.jsonl)：文档 QA 64 题；",
        "- [feature_selftest_queries.manifest.json](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries.manifest.json)：记录抽样策略、来源分布、模型及源数据 revision；",
        "- 各组内部继续按 KG source type、FAE chat/candidate、document ID 轮转抽样；",
        "- FAE Query 由 `gpt-5.6-luna` 仅依据当时 `source_input` 自然化改写，原文和后续答案不进入 Query；",
        "- KG/runtime Query 同样由 Luna 仅依据原 Query 自然化，删除任务脚手架及原因/动作提示；",
        "- 每行严格使用 operation_agent 的 12 字段结构，不包含答案或其他 Gold 字段；",
        "- `origin` 保存正式 split 与原始 core case ID。",
        "",
        "这三个 JSONL 是 Feature Selftest 输入集，不是独立 Gold。正式 validation Gold 位于 `core_validation.json`；held-out test 的公开输入与私有 Gold 分别位于 `core_test_inputs.json` 和 `private/core_test_gold.json`。",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _attach_release_state(
    report: dict[str, Any],
    *,
    dataset: dict[str, Any],
    broad_report: dict[str, Any],
) -> dict[str, Any]:
    report["broad_pool_validation"] = broad_report
    report["overall_status"] = (
        "release_ready"
        if report["release_ready"] and broad_report["status"] == "passed"
        else "blocked"
    )
    report["release_blockers"] = [
        *([f"human_freeze_pending:{report['pending_human_freeze_count']}"]
          if report["pending_human_freeze_count"] else []),
        *[f"broad_pool:{issue}" for issue in broad_report["issues"]],
    ]
    report["release_warnings"] = [
        *list(report.get("warnings") or []),
        *list(broad_report.get("runtime_compatibility_issues") or []),
    ]
    return report


def write_artifacts(dataset: dict[str, Any]) -> dict[str, Any]:
    broad = build_broad_pools()
    report = validate_dataset(dataset)
    broad_report = validate_broad_pools(broad)
    _attach_release_state(report, dataset=dataset, broad_report=broad_report)
    _write_json(PUBLIC_CORE_PATH, _public_manifest(dataset))
    _write_json(CORE_PATH, dataset)
    _write_json(VALIDATION_PATH, _split_dataset(dataset, split="validation", include_gold=True))
    _write_json(TEST_INPUTS_PATH, _split_dataset(dataset, split="held_out_test", include_gold=False))
    _write_json(TEST_GOLD_PATH, _split_dataset(dataset, split="held_out_test", include_gold=True))
    _write_json(REVIEW_QUEUE_PATH, build_review_queue(dataset))
    _write_json(BROAD_PATH, broad)
    _write_json(REPORT_PATH, report)
    _write_json(PREDICTION_TEMPLATE_PATH, prediction_template(dataset))
    selftest_shards, selftest_manifest = candidate_feature_selftest_shards()
    for path, rows in zip(
        (
            FEATURE_SELFTEST_KG_RUNTIME_PATH,
            FEATURE_SELFTEST_FAE_PATH,
            FEATURE_SELFTEST_DOCUMENT_QA_PATH,
        ),
        selftest_shards,
        strict=True,
    ):
        _write_jsonl(path, rows)
    _write_json(FEATURE_SELFTEST_MANIFEST_PATH, selftest_manifest)
    MARKDOWN_PATH.write_text(render_markdown(dataset, report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="formal-debug-benchmark")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--release-check", action="store_true")
    parser.add_argument("--approve-current", action="store_true")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--score-out", type=Path, default=SCORE_PATH)
    parser.add_argument("--split", choices=("validation", "held_out_test", "all"), default="validation")
    parser.add_argument("--allow-held-out-test", action="store_true")
    args = parser.parse_args()

    if args.approve_current:
        if args.validate_only:
            parser.error("--approve-current cannot be combined with --validate-only")
        record_current_approval()

    if args.validate_only:
        dataset = _load(CORE_PATH)
        report = validate_dataset(dataset, require_release_ready=args.release_check)
        broad_report = validate_broad_pools(_load(BROAD_PATH))
        _attach_release_state(report, dataset=dataset, broad_report=broad_report)
        if args.release_check and broad_report["status"] != "passed":
            report["issues"].extend(
                f"broad_pool:{issue}" for issue in broad_report["issues"]
            )
            report["status"] = "failed"
            report["release_ready"] = False
        _write_json(REPORT_PATH, report)
    else:
        dataset = build_dataset()
        report = write_artifacts(dataset)
        if args.release_check:
            report = validate_dataset(dataset, require_release_ready=True)
            broad_report = validate_broad_pools(build_broad_pools())
            _attach_release_state(report, dataset=dataset, broad_report=broad_report)
            if broad_report["status"] != "passed":
                report["issues"].extend(
                    f"broad_pool:{issue}" for issue in broad_report["issues"]
                )
                report["status"] = "failed"
                report["release_ready"] = False
            _write_json(REPORT_PATH, report)

    if args.predictions:
        score = score_predictions(
            dataset,
            _load(args.predictions),
            split=args.split,
            allow_held_out_test=args.allow_held_out_test,
        )
        _write_json(args.score_out, score)

    print(json.dumps({
        "status": report["status"],
        "release_ready": report["release_ready"],
        "pending_human_freeze_count": report["pending_human_freeze_count"],
        "core": PUBLIC_CORE_PATH.relative_to(ROOT).as_posix(),
        "report": REPORT_PATH.relative_to(ROOT).as_posix(),
        "approval": APPROVAL_PATH.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
