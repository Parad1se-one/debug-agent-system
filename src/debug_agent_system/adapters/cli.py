from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from debug_agent_system.adapters.kg_v2_adapter import KGv2Adapter
from debug_agent_system.adapters.codex_read import CodexReadToolHarness
from debug_agent_system.runtime import DebugAgentSystem
from debug_agent_system.agents.write import (
    ChatCollectAgent,
    ConflictResolutionAgent,
    QualityGateAgent,
    RawDocIngestAgent,
    SectionCaseBundleAgent,
    WriteSidePipeline,
)
from debug_agent_system.agents.write_v2 import WriteSideV2Pipeline
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.agents.tools import EvidenceContextParserAgent, EvidenceToolAgent, parse_json_payload
from debug_agent_system.eval.write_side.kg_v2_gold_compare import main as kg_v2_gold_compare_main
from debug_agent_system.eval.write_side.manual_golden_compare import compare_manual_cases
from debug_agent_system.eval.write_side.w2_postrun_compare import main as w2_postrun_compare_main
from debug_agent_system.eval.write_side.w2_live_report import main as w2_live_report_main
from debug_agent_system.eval.write_side.w2_live_compare import main as w2_live_compare_main
from debug_agent_system.eval.write_side.w2_family_diagnostics import main as w2_family_diagnostics_main
from debug_agent_system.eval.write_side.w2_postrun_report import main as w2_postrun_report_main
from debug_agent_system.eval.write_side.w2_quality_gate import main as w2_quality_gate_main
from debug_agent_system.eval.write_side.w2_quality_diagnostics import main as w2_quality_diagnostics_main
from debug_agent_system.eval.write_side.w2_run_status import main as w2_run_status_main
from debug_agent_system.eval.write_side.w2_split_diagnostics import main as w2_split_diagnostics_main
from debug_agent_system.eval.write_side.kg_v2_overview import main as kg_v2_overview_main
from debug_agent_system.knowledge.sqlite_sag import build_sqlite_sag
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2 import JsonKGV2Store
from debug_agent_system.core.config import load_config


def _legacy_write_context(config_path: str | Path) -> SimpleNamespace:
    """Load write-side legacy storage without initializing the v2 read runtime."""

    config = load_config(config_path)
    return SimpleNamespace(config=config, store=JsonKGStore(config.kg_root))


def _tool_category(schema: dict[str, Any]) -> str:
    """Best-effort extract a tool category from a rendered schema."""

    name = str(schema.get("name") or (schema.get("function") or {}).get("name") or "")
    if not name:
        return ""
    category_hints = {
        "parse_evidence": "evidence",
        "parse_evidence_context": "evidence",
        "parse_evtx_window": "incident",
        "read_kernel_dump": "incident",
        "read_log_window": "incident",
        "search_diagnostic_events": "incident",
        "build_incident_timeline": "incident",
        "kg_": "kg",
        "list_files": "corpus",
        "search_text": "corpus",
        "read_text": "corpus",
    }
    for prefix, category in category_hints.items():
        if name.startswith(prefix):
            return category
    return "misc"


def _cli_evidence_resources(values: list[str]) -> list[dict[str, object]]:
    """Accept either a strict resource JSON object or a local file path."""

    resources: list[dict[str, object]] = []
    for value in values:
        stripped = str(value or "").strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError("evidence_resource_json_must_be_object")
            resources.append(parsed)
            continue
        path = Path(stripped)
        resources.append(
            {
                "resource_id": "",
                "kind": "auto",
                "name": path.name,
                "path": str(path),
                "url": "",
                "text": "",
                "mime": "",
                "size": None,
                "sha256": "",
                "source_message_id": "",
                "metadata": {},
            }
        )
    return resources


def _kg_v2_sag_publish_required(results: list[dict[str, object]], kg_version: str) -> bool:
    """Return whether an approved batch changed canonical KG_v2 state."""

    return kg_version in {"v2", "both"} and any(
        bool(item.get("requires_sag_publish"))
        for item in results
        if isinstance(item, dict)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="debug-agent-system")
    parser.add_argument("--config", default="config/debug_agent_system.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    diag = sub.add_parser("diagnose")
    diag.add_argument("query")
    diag.add_argument("--non-interactive", action="store_true")
    diag.add_argument("--evidence-resource", action="append", default=[])
    incident = sub.add_parser("analyze-incident")
    incident.add_argument("query")
    incident.add_argument("--evidence-resource", action="append", default=[])
    incident.add_argument(
        "--log-summary-json",
        default="",
        help="JSON object or a path to a JSON file supplied by an upstream log collector.",
    )
    incident.add_argument("--out", default="")
    step = sub.add_parser("step")
    step.add_argument("session_id")
    step.add_argument("message")
    step.add_argument("--evidence-resource", action="append", default=[])
    read_harness = sub.add_parser("read-tool-harness")
    read_harness.add_argument("query")
    read_harness.add_argument("--non-interactive", action="store_true")
    read_harness.add_argument("--evidence-resource", action="append", default=[])
    read_harness.add_argument(
        "--codex",
        action="store_true",
        help="Enable the optional Codex tool controller for this invocation.",
    )
    read_harness.add_argument(
        "--deepseek",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    extract_w1w2 = sub.add_parser("extract-xing-w1w2")
    extract_w1w2.add_argument("import_root")
    extract_w1w2.add_argument("--limit", type=int, default=0)
    extract_w1w2.add_argument("--hits-only", action="store_true")
    extract_w1w2.add_argument("--out-dir", default=None)
    extract_w1w2.add_argument("--sample-limit", type=int, default=5)
    extract_w1w2.add_argument("--emit-candidates", action="store_true")
    extract_w1w2.add_argument("--w2-workers", type=int, default=1)
    extract_w1w2.add_argument("--w2-mode", choices=["legacy_only", "legacy_bridge", "native_v2", "prompt_first", "compare"], default="legacy_only")
    extract_text_w1 = sub.add_parser("extract-text-w1")
    extract_text_w1.add_argument("import_root")
    extract_text_w1.add_argument("--limit", type=int, default=0)
    extract_text_w1.add_argument("--out-dir", default=None)
    extract_xing_relations = sub.add_parser("extract-xing-relations-w1")
    extract_xing_relations.add_argument("xing_import_root")
    extract_xing_relations.add_argument("relation_import_root")
    extract_xing_relations.add_argument("--limit", type=int, default=0)
    extract_xing_relations.add_argument("--hits-only", action="store_true")
    extract_xing_relations.add_argument("--quiet-gap-hours", type=float, default=12.0)
    extract_xing_relations.add_argument("--max-messages", type=int, default=120)
    extract_xing_relations.add_argument("--context-attach-minutes", type=float, default=60.0)
    extract_xing_relations.add_argument("--out-dir", required=True)
    w9_doc_strategy = sub.add_parser("w9-doc-strategy")
    w9_doc_strategy.add_argument("path")
    w9_doc_strategy_batch = sub.add_parser("w9-doc-strategy-batch")
    w9_doc_strategy_batch.add_argument("root")
    w9_doc_strategy_batch.add_argument("--include-sop", action="store_true")
    w9_doc_sections = sub.add_parser("w9-doc-structured-sections")
    w9_doc_sections.add_argument("path")
    w9_doc_section_cases = sub.add_parser("w9-doc-section-cases")
    w9_doc_section_cases.add_argument("path")
    w9_doc_build = sub.add_parser("w9-doc-build")
    w9_doc_build.add_argument("path")
    w9_doc_build.add_argument("--out-dir", required=True)
    w9_doc_build_not_entered = sub.add_parser("w9-doc-build-not-entered")
    w9_doc_build_not_entered.add_argument("root")
    w9_doc_build_not_entered.add_argument("--manifest-path", default="data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json")
    w9_doc_build_not_entered.add_argument("--out-root", required=True)
    w9_doc_build_not_entered.add_argument("--include-sop", action="store_true")
    w10_bundle = sub.add_parser("w10-section-case-bundle")
    w10_bundle.add_argument("section_cases_json")
    w10_bundle.add_argument("--out", default="")
    w10_bundle_tree = sub.add_parser("w10-section-case-bundle-tree")
    w10_bundle_tree.add_argument("root")
    w10_bundle_tree.add_argument("--out-root", required=True)
    extract_summaries_w2 = sub.add_parser("extract-summaries-w2")
    extract_summaries_w2.add_argument("summaries_path")
    extract_summaries_w2.add_argument("--out-dir", default=None)
    extract_summaries_w2.add_argument("--sample-limit", type=int, default=5)
    extract_summaries_w2.add_argument("--emit-candidates", action="store_true")
    extract_summaries_w2.add_argument("--w2-workers", type=int, default=1)
    extract_summaries_w2.add_argument("--w2-mode", choices=["legacy_only", "legacy_bridge", "native_v2", "prompt_first", "compare"], default="legacy_only")
    process_w2_candidates = sub.add_parser("process-w2-candidates")
    process_w2_candidates.add_argument("candidates_jsonl")
    process_w2_candidates.add_argument("--queue-dir", default=None)
    process_w2_candidates.add_argument("--apply-approved", action="store_true")
    process_w2_candidates.add_argument("--emit-episodes", action="store_true")
    process_w2_candidates.add_argument("--dry-run-merge", action="store_true")
    process_w2_candidates.add_argument("--kg-mode", choices=["legacy", "v2", "both"], default="legacy")
    process_w2_candidates.add_argument("--kg-v2-root", default=None)
    process_w2_candidates.add_argument("--kg-v2-queue-dir", default=None)
    manual_compare = sub.add_parser("manual-golden-compare")
    manual_compare.add_argument("--manual-root", default="data/kg/review_queue/manual_review_examples")
    manual_compare.add_argument("--episodes", default="")
    manual_compare.add_argument("--import-root", default="")
    manual_compare.add_argument("--hits-only", action="store_true")
    manual_compare.add_argument("--limit", type=int, default=0)
    manual_compare.add_argument("--deepseek", action="store_true")
    manual_compare.add_argument("--out", default="")
    ingest = sub.add_parser("ingest-xing")
    ingest.add_argument("import_root")
    ingest.add_argument("--limit", type=int, default=0)
    ingest.add_argument("--hits-only", action="store_true")
    ingest.add_argument("--out-dir", default=None)
    ingest.add_argument("--queue-dir", default=None)
    ingest.add_argument("--emit-episodes", action="store_true")
    ingest.add_argument("--dry-run-merge", action="store_true")
    ingest.add_argument("--apply-approved", action="store_true")
    ingest.add_argument("--w2-workers", type=int, default=1)
    ingest.add_argument("--kg-mode", choices=["legacy", "v2", "both"], default="legacy")
    ingest.add_argument("--kg-v2-root", default=None)
    ingest.add_argument("--kg-v2-queue-dir", default=None)
    ingest.add_argument("--w2-mode", choices=["legacy_only", "legacy_bridge", "native_v2", "prompt_first", "compare"], default="legacy_only")
    ingest_text = sub.add_parser("ingest-text-history")
    ingest_text.add_argument("import_root")
    ingest_text.add_argument("--limit", type=int, default=0)
    ingest_text.add_argument("--out-dir", default=None)
    ingest_text.add_argument("--queue-dir", default=None)
    ingest_text.add_argument("--emit-episodes", action="store_true")
    ingest_text.add_argument("--no-dry-run-merge", action="store_true")
    ingest_text.add_argument("--apply-approved", action="store_true")
    ingest_text.add_argument("--w2-workers", type=int, default=1)
    ingest_text.add_argument("--kg-v2-root", default="data/kg_v2")
    ingest_text.add_argument("--kg-v2-queue-dir", default=None)
    ingest_text.add_argument("--w2-mode", choices=["legacy_bridge", "native_v2", "prompt_first", "compare"], default="native_v2")
    ingest_doc = sub.add_parser("ingest-non-sop-doc")
    ingest_doc.add_argument("path")
    ingest_doc.add_argument("--queue-dir", default=None)
    ingest_doc.add_argument("--kg-v2-root", default="data/kg_v2")
    ingest_doc.add_argument("--kg-v2-queue-dir", default=None)
    ingest_doc.add_argument("--no-dry-run-merge", action="store_true")
    ingest_docs = sub.add_parser("ingest-non-sop-docs")
    ingest_docs.add_argument("root")
    ingest_docs.add_argument("--manifest-path", default="data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json")
    ingest_docs.add_argument("--limit", type=int, default=0)
    ingest_docs.add_argument("--out", default=None)
    ingest_docs.add_argument("--queue-dir", default=None)
    ingest_docs.add_argument("--kg-v2-root", default="data/kg_v2")
    ingest_docs.add_argument("--kg-v2-queue-dir", default=None)
    ingest_docs.add_argument("--no-dry-run-merge", action="store_true")
    ingest_sop_doc = sub.add_parser("ingest-sop-doc")
    ingest_sop_doc.add_argument("path")
    ingest_sop_doc.add_argument("--out", default=None)
    ingest_sop_doc.add_argument("--queue-dir", default=None)
    ingest_sop_doc.add_argument("--kg-v2-root", default="data/kg_v2")
    ingest_sop_doc.add_argument("--kg-v2-queue-dir", default=None)
    ingest_sop_doc.add_argument("--no-dry-run-merge", action="store_true")
    sync_sop_docs = sub.add_parser("sync-sop-docs")
    sync_sop_docs.add_argument("root")
    sync_sop_docs.add_argument("--limit", type=int, default=0)
    sync_sop_docs.add_argument("--out", default=None)
    sync_sop_docs.add_argument("--queue-dir", default=None)
    sync_sop_docs.add_argument("--kg-v2-root", default="data/kg_v2")
    sync_sop_docs.add_argument("--kg-v2-queue-dir", default=None)
    sync_sop_docs.add_argument("--no-dry-run-merge", action="store_true")
    ingest_evidence = sub.add_parser("ingest-evidence-context")
    ingest_evidence.add_argument("root")
    ingest_evidence.add_argument("--max-bytes", type=int, default=65536)
    ingest_evidence.add_argument("--limit", type=int, default=0)
    ingest_evidence.add_argument("--queue-dir", default=None)
    ingest_evidence.add_argument("--kg-v2-root", default="data/kg_v2")
    ingest_evidence.add_argument("--kg-v2-queue-dir", default=None)
    ingest_evidence.add_argument("--no-dry-run-merge", action="store_true")
    expert_correction = sub.add_parser("ingest-expert-correction")
    expert_correction.add_argument("review_item_json")
    expert_correction.add_argument("correction_json")
    expert_correction.add_argument("--queue-dir", default=None)
    expert_correction.add_argument("--kg-v2-root", default="data/kg_v2")
    expert_correction.add_argument("--kg-v2-queue-dir", default=None)
    expert_correction.add_argument("--no-dry-run-merge", action="store_true")
    diagnostic_feedback = sub.add_parser("ingest-diagnostic-feedback")
    diagnostic_feedback.add_argument("transcript_json")
    diagnostic_feedback.add_argument("--queue-dir", default=None)
    diagnostic_feedback.add_argument("--kg-v2-root", default="data/kg_v2")
    diagnostic_feedback.add_argument("--kg-v2-queue-dir", default=None)
    diagnostic_feedback.add_argument("--no-dry-run-merge", action="store_true")
    log_pattern = sub.add_parser("ingest-log-pattern")
    log_pattern.add_argument("log_summary_json")
    log_pattern.add_argument("--queue-dir", default=None)
    log_pattern.add_argument("--kg-v2-root", default="data/kg_v2")
    log_pattern.add_argument("--kg-v2-queue-dir", default=None)
    log_pattern.add_argument("--no-dry-run-merge", action="store_true")
    atr_proposal = sub.add_parser("ingest-atr-weight-proposal")
    atr_proposal.add_argument("feedback_json")
    atr_proposal.add_argument("--queue-dir", default=None)
    atr_proposal.add_argument("--kg-v2-root", default="data/kg_v2")
    atr_proposal.add_argument("--kg-v2-queue-dir", default=None)
    parse_proj = sub.add_parser("parse-proj")
    parse_proj.add_argument("path")
    parse_proj.add_argument("--max-bytes", type=int, default=65536)
    parse_jira = sub.add_parser("parse-jira")
    parse_jira.add_argument("value")
    parse_attachment = sub.add_parser("parse-attachment")
    parse_attachment.add_argument("path")
    parse_document = sub.add_parser("parse-document")
    parse_document.add_argument("path")
    parse_document.add_argument("--max-bytes", type=int, default=65536)
    parse_image = sub.add_parser("parse-image")
    parse_image.add_argument("path")
    parse_image.add_argument("--max-bytes", type=int, default=65536)
    parse_evidence = sub.add_parser("parse-evidence")
    parse_evidence.add_argument("tool", choices=["attachment", "document", "dmp", "image", "jira", "proj", "log_package"])
    parse_evidence.add_argument("payload")
    parse_evidence.add_argument("--max-bytes", type=int, default=65536)
    parse_evidence_context = sub.add_parser("parse-evidence-context")
    parse_evidence_context.add_argument("root")
    parse_evidence_context.add_argument("--max-bytes", type=int, default=65536)
    parse_evidence_context.add_argument("--limit", type=int, default=0)
    parse_evidence_context.add_argument("--out", default="")
    list_tools = sub.add_parser("list-tools")
    list_tools.add_argument("--style", choices=["responses", "chat_completions"], default="responses")
    list_tools.add_argument("--category", default="")
    list_tools.add_argument("--out", default="")
    run_tool = sub.add_parser("run-tool")
    run_tool.add_argument("tool")
    run_tool.add_argument("arguments_json", default="{}", nargs="?")
    sag_build = sub.add_parser("sag-build")
    sag_build.add_argument("--out", default="data/kg_sag/debug_agent.sqlite")
    sag_build.add_argument("--raw-root", default="data/raw/aoi_debug_agent_sources")
    sag_build.add_argument("--kg-root", default="data/kg")
    sag_build.add_argument("--kg-v2-root", default="data/kg_v2")
    sag_build.add_argument("--w1-root", default="data/results/w1_full_20260703_061455")
    sag_build.add_argument("--no-w1", action="store_true")
    sag_build.add_argument("--report-out", default="data/kg_sag/build_report.json")
    review_decision = sub.add_parser("review-decision")
    review_decision.add_argument("queue", choices=["candidates", "merge_candidates", "noise_candidates", "ask_info_candidates", "v2_typed_candidates", "atr_weight_proposals"])
    review_decision.add_argument("item_id")
    review_decision.add_argument("action")
    review_decision.add_argument("--queue-dir", default=None)
    review_decision.add_argument("--kg-version", choices=["legacy", "v2"], default="legacy")
    review_decision.add_argument("--kg-v2-root", default="data/kg_v2")
    review_decision.add_argument("--reviewer", default="")
    review_decision.add_argument("--note", default="")
    review_correction = sub.add_parser("review-correction")
    review_correction.add_argument(
        "queue",
        choices=["v2_typed_candidates"],
    )
    review_correction.add_argument("item_id")
    review_correction.add_argument("operation")
    review_correction.add_argument("target_ref")
    review_correction.add_argument("--payload-json", default="{}")
    review_correction.add_argument(
        "--evidence-message-id", action="append", default=[]
    )
    review_correction.add_argument("--queue-dir", default=None)
    review_correction.add_argument("--kg-v2-root", default="data/kg_v2")
    review_correction.add_argument("--reviewer", default="")
    review_correction.add_argument("--note", default="")
    review_compile_corrections = sub.add_parser(
        "review-compile-corrections"
    )
    review_compile_corrections.add_argument(
        "queue",
        choices=["v2_typed_candidates"],
    )
    review_compile_corrections.add_argument("item_id")
    review_compile_corrections.add_argument("--queue-dir", default=None)
    review_compile_corrections.add_argument(
        "--kg-v2-root", default="data/kg_v2"
    )
    apply_queue = sub.add_parser("apply-approved-queue")
    apply_queue.add_argument("--queue-dir", default=None)
    apply_queue.add_argument("--kg-version", choices=["legacy", "v2", "both"], default="legacy")
    apply_queue.add_argument("--kg-v2-root", default="data/kg_v2")
    apply_queue.add_argument("--kg-v2-queue-dir", default=None)
    apply_queue.add_argument("--kg-v2-sag-out", default="data/kg_v2_sag/debug_agent_v2.sqlite")
    apply_queue.add_argument("--skip-sag-build", action="store_true")
    apply_queue.add_argument("--out", default=None)
    kg_v2_seed_sop = sub.add_parser("kg-v2-seed-sop")
    kg_v2_seed_sop.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_seed_sop.add_argument("--chunks", default="data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json")
    kg_v2_seed_sop.add_argument("--limit", type=int, default=0)
    kg_v2_seed_sop.add_argument("--replace", action="store_true")
    kg_v2_seed_manual = sub.add_parser("kg-v2-seed-manual")
    kg_v2_seed_manual.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_seed_manual.add_argument("--manual-root", default="data/kg/review_queue/manual_review_examples")
    kg_v2_seed_manual.add_argument("--limit", type=int, default=0)
    kg_v2_seed_manual.add_argument("--replace", action="store_true")
    kg_v2_seed_all = sub.add_parser("kg-v2-seed-all")
    kg_v2_seed_all.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_seed_all.add_argument("--chunks", default="data/raw/aoi_debug_agent_sources/chunks/debug_chunks.json")
    kg_v2_seed_all.add_argument("--manual-root", default="data/kg/review_queue/manual_review_examples")
    kg_v2_seed_all.add_argument("--sop-limit", type=int, default=0)
    kg_v2_seed_all.add_argument("--manual-limit", type=int, default=0)
    kg_v2_seed_all.add_argument("--no-replace", action="store_true")
    kg_v2_build_curated = sub.add_parser("kg-v2-build-curated")
    kg_v2_build_curated.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_build_curated.add_argument("--build-root", default="data/kg_v2_sop_draft_build")
    kg_v2_build_curated.add_argument("--gold-root", default="data/kg_v2_sop_draft_build/gold_cases")
    kg_v2_build_curated.add_argument("--summary-out", default="data/results/kg_v2_write_side_build_summary.json")
    kg_v2_build_curated.add_argument(
        "--allow-active-rebuild",
        action="store_true",
        help="bootstrap/rollback only; bypasses the active KG rebuild guard",
    )
    kg_v2_validate = sub.add_parser("kg-v2-validate")
    kg_v2_validate.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_materialize = sub.add_parser("kg-v2-materialize")
    kg_v2_materialize.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_materialize.add_argument("--out", default="data/kg_v2/materialized_execution")
    kg_v2_sag_build = sub.add_parser("kg-v2-sag-build")
    kg_v2_sag_build.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_sag_build.add_argument("--out", default="data/kg_v2_sag/debug_agent_v2.sqlite")
    kg_v2_sag_build.add_argument("--no-reset", action="store_true")
    kg_v2_gold_compare = sub.add_parser("kg-v2-gold-compare")
    kg_v2_gold_compare.add_argument("--gold-root", default="data/annotations/goldcases/gold-v1")
    kg_v2_gold_compare.add_argument("--kg-root", default="data/kg")
    kg_v2_gold_compare.add_argument("--runner-mode", choices=["legacy_bridge", "native_v2", "prompt_first", "compare"], default="legacy_bridge")
    kg_v2_gold_compare.add_argument("--deepseek", action="store_true")
    kg_v2_gold_compare.add_argument("--emit-prompt-inputs", action="store_true")
    kg_v2_gold_compare.add_argument("--out", default="")
    kg_v2_overview = sub.add_parser("kg-v2-overview")
    kg_v2_overview.add_argument("--kg-v2-root", default="data/kg_v2")
    kg_v2_overview.add_argument("--pinned-run-dir", default="data/results/w2_native_v2_full_pinned_20260708_010455")
    kg_v2_overview.add_argument("--snapshot-out", default="data/results/kg_v2_overview_snapshot.json")
    kg_v2_overview.add_argument("--html-out", default="data/results/kg_v2_overview.html")
    w2_family_diag = sub.add_parser("w2-family-diagnostics")
    w2_family_diag.add_argument("input_jsonl")
    w2_family_diag.add_argument("--out", default="")
    w2_family_diag.add_argument("--sample-limit", type=int, default=20)
    w2_quality_diag = sub.add_parser("w2-quality-diagnostics")
    w2_quality_diag.add_argument("input_jsonl")
    w2_quality_diag.add_argument("--out", default="")
    w2_quality_diag.add_argument("--sample-limit", type=int, default=25)
    w2_quality_gate = sub.add_parser("w2-quality-gate")
    w2_quality_gate.add_argument("--diagnostics", required=True)
    w2_quality_gate.add_argument("--max-noncanonical-family-rate", type=float, default=0.02)
    w2_quality_gate.add_argument("--max-pseudo-family-rate", type=float, default=0.0)
    w2_quality_gate.add_argument("--max-long-variant-rate", type=float, default=0.03)
    w2_quality_gate.add_argument("--max-questionish-variant-rate", type=float, default=0.0)
    w2_quality_gate.add_argument("--max-empty-case-rate", type=float, default=0.12)
    w2_quality_gate.add_argument("--max-report-noise-rate", type=float, default=0.08)
    w2_quality_gate.add_argument("--max-positive-status-rate", type=float, default=0.12)
    w2_quality_gate.add_argument("--max-split-required-rate", type=float, default=0.12)
    w2_quality_gate.add_argument("--max-action-duplicates-rate", type=float, default=0.02)
    w2_postrun = sub.add_parser("w2-postrun-report")
    w2_postrun.add_argument("--run-dir", required=True)
    w2_postrun.add_argument("--out", default="")
    w2_postrun_compare = sub.add_parser("w2-postrun-compare")
    w2_postrun_compare.add_argument("--base", required=True)
    w2_postrun_compare.add_argument("--candidate", required=True)
    w2_postrun_compare.add_argument("--out", default="")
    w2_live = sub.add_parser("w2-live-report")
    w2_live.add_argument("--run-dir", required=True)
    w2_live.add_argument("--out", default="")
    w2_live_compare = sub.add_parser("w2-live-compare")
    w2_live_compare.add_argument("--base", required=True)
    w2_live_compare.add_argument("--candidate", required=True)
    w2_live_compare.add_argument("--out", default="")
    w2_run_status = sub.add_parser("w2-run-status")
    w2_run_status.add_argument("--run-dir", required=True)
    w2_split_diag = sub.add_parser("w2-split-diagnostics")
    w2_split_diag.add_argument("input_jsonl")
    w2_split_diag.add_argument("--out", default="")
    w2_split_diag.add_argument("--sample-limit", type=int, default=25)
    return parser


def _refine_and_gate_v2_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    refined = ConflictResolutionAgent().normalize_v2_bundle(bundle)
    refined["quality_gate"] = QualityGateAgent().score_v2_bundle(refined)
    return refined


def _write_refined_v2_bundle(bundle: dict[str, Any], out_path: str | Path) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "type": "W3W4V2BundleWriteResult",
        "bundle_path": str(out),
        "schema_valid": bool(bundle.get("schema_valid")),
        "schema_issues": list(bundle.get("schema_issues") or []),
        "quality_gate": bundle.get("quality_gate") or {},
        "report": bundle.get("report") or {},
    }


def _refine_w10_bundle_tree(root: str | Path, out_root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    out_base = Path(out_root)
    results: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("section_cases.json")):
        raw = SectionCaseBundleAgent().build_bundle_from_file(path)
        refined = _refine_and_gate_v2_bundle(raw)
        out_path = out_base / path.relative_to(root_path).parent / "kg_v2_draft_bundle.json"
        result = _write_refined_v2_bundle(refined, out_path)
        results.append({"section_cases_json": str(path), **result})
    return {
        "type": "W10W3W4SectionCaseBundleBatchResult",
        "root": str(root_path),
        "out_root": str(out_base),
        "bundle_count": len(results),
        "schema_valid_count": sum(1 for item in results if item["schema_valid"]),
        "schema_invalid_count": sum(1 for item in results if not item["schema_valid"]),
        "quality_gate_passed_count": sum(1 for item in results if (item.get("quality_gate") or {}).get("passed")),
        "quality_gate_failed_count": sum(1 for item in results if not (item.get("quality_gate") or {}).get("passed")),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "diagnose":
        system = DebugAgentSystem.from_config(args.config)
        out = system.diagnose({
            "query": args.query,
            "interactive": not args.non_interactive,
            "evidence_resources": _cli_evidence_resources(args.evidence_resource),
        })
    elif args.cmd == "analyze-incident":
        system = DebugAgentSystem.from_config(args.config)
        log_summary: dict[str, object] = {}
        if args.log_summary_json:
            raw = str(args.log_summary_json)
            summary_path = Path(raw)
            if summary_path.exists() and summary_path.is_file():
                decoded = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                decoded = json.loads(raw)
            if not isinstance(decoded, dict):
                raise ValueError("log_summary_json_must_be_object")
            log_summary = decoded
        out = system.analyze_incident({
            "query": args.query,
            "evidence_resources": _cli_evidence_resources(args.evidence_resource),
            "log_summary": log_summary,
        })
        if args.out:
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.cmd == "step":
        system = DebugAgentSystem.from_config(args.config)
        out = system.step(
            args.session_id,
            args.message,
            evidence_resources=_cli_evidence_resources(args.evidence_resource),
        )
    elif args.cmd == "read-tool-harness":
        config = load_config(args.config)
        if args.codex or args.deepseek:
            config.read_llm.enabled = True
        system = DebugAgentSystem(config)
        out = CodexReadToolHarness(system).run(
            args.query,
            evidence_resources=_cli_evidence_resources(args.evidence_resource),
            interactive=not args.non_interactive,
        )
    elif args.cmd == "extract-xing-w1w2":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
        ).run_w1_w2_xing_upload(
            args.import_root,
            limit=args.limit,
            hits_only=args.hits_only,
            out_dir=args.out_dir,
            sample_limit=args.sample_limit,
            emit_candidates=args.emit_candidates,
            w2_workers=args.w2_workers,
            w2_mode=args.w2_mode,
        )
    elif args.cmd == "extract-text-w1":
        run = ChatCollectAgent().import_text_history(
            args.import_root,
            limit=args.limit,
            out_dir=args.out_dir,
        )
        output_files = {}
        if args.out_dir:
            output_files = {
                "messages": str(Path(args.out_dir) / "messages.jsonl"),
                "thread_summaries": str(Path(args.out_dir) / "thread_summaries.json"),
                "episodes": str(Path(args.out_dir) / "episodes.json"),
                "field_report_anchors": str(Path(args.out_dir) / "field_report_anchors.json"),
                "observed_people": str(Path(args.out_dir) / "observed_people.json"),
                "run_manifest": str(Path(args.out_dir) / "run_manifest.json"),
            }
        out = {
            "run_manifest": run["run_manifest"],
            "counts": run["run_manifest"].get("counts", {}),
            "output_files": output_files,
        }
    elif args.cmd == "extract-xing-relations-w1":
        run = ChatCollectAgent().import_xing_with_relations(
            args.xing_import_root,
            args.relation_import_root,
            limit=args.limit,
            hits_only=args.hits_only,
            quiet_gap_hours=args.quiet_gap_hours,
            max_messages=args.max_messages,
            context_attach_minutes=args.context_attach_minutes,
            out_dir=args.out_dir,
        )
        out = {
            "run_manifest": run["run_manifest"],
            "counts": run["run_manifest"].get("counts", {}),
            "output_files": {
                "messages": str(Path(args.out_dir) / "messages.jsonl"),
                "thread_summaries": str(Path(args.out_dir) / "thread_summaries.json"),
                "episodes": str(Path(args.out_dir) / "episodes.json"),
                "message_reference_graph": str(Path(args.out_dir) / "message_reference_graph.json"),
                "run_manifest": str(Path(args.out_dir) / "run_manifest.json"),
            },
        }
    elif args.cmd == "w9-doc-strategy":
        out = RawDocIngestAgent().inspect_document(args.path)
    elif args.cmd == "w9-doc-strategy-batch":
        out = RawDocIngestAgent().build_root_checklist(args.root, include_sop=args.include_sop)
    elif args.cmd == "w9-doc-structured-sections":
        out = RawDocIngestAgent().build_structured_sections(args.path)
    elif args.cmd == "w9-doc-section-cases":
        out = RawDocIngestAgent().build_section_cases(args.path)
    elif args.cmd == "w9-doc-build":
        out = RawDocIngestAgent().write_doc_outputs(args.path, args.out_dir)
    elif args.cmd == "w9-doc-build-not-entered":
        out = RawDocIngestAgent().build_not_entered_docs(
            args.root,
            manifest_path=args.manifest_path,
            include_sop=args.include_sop,
            out_root=args.out_root,
        )
    elif args.cmd == "w10-section-case-bundle":
        raw_bundle = SectionCaseBundleAgent().build_bundle_from_file(args.section_cases_json)
        refined_bundle = _refine_and_gate_v2_bundle(raw_bundle)
        out = _write_refined_v2_bundle(refined_bundle, args.out) if args.out else refined_bundle
    elif args.cmd == "w10-section-case-bundle-tree":
        out = _refine_w10_bundle_tree(args.root, args.out_root)
    elif args.cmd == "extract-summaries-w2":
        system = _legacy_write_context(args.config)
        summaries = json.loads(Path(args.summaries_path).read_text(encoding="utf-8"))
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
        ).run_w2_summaries(
            summaries,
            out_dir=args.out_dir,
            sample_limit=args.sample_limit,
            emit_candidates=args.emit_candidates,
            w2_workers=args.w2_workers,
            w2_mode=args.w2_mode,
        )
    elif args.cmd == "process-w2-candidates":
        system = _legacy_write_context(args.config)
        candidates: list[dict[str, Any]] = []
        for line in Path(args.candidates_jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                candidates.append(row)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_candidates(
            candidates,
            apply_approved=args.apply_approved,
            emit_episodes=args.emit_episodes,
            dry_run_merge=args.dry_run_merge,
            kg_mode=args.kg_mode,
        )
    elif args.cmd == "manual-golden-compare":
        system = _legacy_write_context(args.config)
        out = compare_manual_cases(
            manual_root=args.manual_root,
            kg_root=system.store.root if hasattr(system.store, "root") else "data/kg",
            episodes_path=args.episodes or None,
            import_root=args.import_root or None,
            hits_only=args.hits_only,
            limit=args.limit,
            deepseek=True if args.deepseek else False,
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.cmd == "ingest-xing":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_xing_upload(
            args.import_root,
            limit=args.limit,
            hits_only=args.hits_only,
            out_dir=args.out_dir,
            apply_approved=args.apply_approved,
            emit_episodes=args.emit_episodes,
            dry_run_merge=args.dry_run_merge,
            w2_workers=args.w2_workers,
            kg_mode=args.kg_mode,
            w2_mode=args.w2_mode,
        )
    elif args.cmd == "ingest-text-history":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_text_history(
            args.import_root,
            limit=args.limit,
            out_dir=args.out_dir,
            apply_approved=args.apply_approved,
            emit_episodes=args.emit_episodes,
            dry_run_merge=not args.no_dry_run_merge,
            w2_workers=args.w2_workers,
            kg_mode="v2",
            w2_mode=args.w2_mode,
        )
    elif args.cmd == "ingest-non-sop-doc":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_non_sop_document(args.path, dry_run_merge=not args.no_dry_run_merge)
    elif args.cmd == "ingest-non-sop-docs":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_non_sop_documents(
            args.root,
            manifest_path=args.manifest_path,
            limit=args.limit,
            dry_run_merge=not args.no_dry_run_merge,
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.cmd == "ingest-sop-doc":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_sop_document(
            args.path,
            dry_run_merge=not args.no_dry_run_merge,
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.cmd == "sync-sop-docs":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_sop_documents(
            args.root,
            limit=args.limit,
            dry_run_merge=not args.no_dry_run_merge,
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.cmd == "ingest-evidence-context":
        system = _legacy_write_context(args.config)
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_evidence_context(
            args.root,
            max_bytes=args.max_bytes,
            limit=args.limit,
            dry_run_merge=not args.no_dry_run_merge,
        )
    elif args.cmd == "ingest-expert-correction":
        system = _legacy_write_context(args.config)
        review_item = json.loads(Path(args.review_item_json).read_text(encoding="utf-8"))
        correction = json.loads(Path(args.correction_json).read_text(encoding="utf-8"))
        if not isinstance(review_item, dict) or not isinstance(correction, dict):
            raise ValueError("expert correction inputs must both be JSON objects")
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_expert_correction(
            review_item,
            correction,
            dry_run_merge=not args.no_dry_run_merge,
        )
    elif args.cmd == "ingest-diagnostic-feedback":
        system = _legacy_write_context(args.config)
        transcript = json.loads(Path(args.transcript_json).read_text(encoding="utf-8"))
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_diagnostic_feedback(
            transcript,
            dry_run_merge=not args.no_dry_run_merge,
        )
    elif args.cmd == "ingest-log-pattern":
        system = _legacy_write_context(args.config)
        log_summary = json.loads(Path(args.log_summary_json).read_text(encoding="utf-8"))
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_log_pattern(
            log_summary,
            dry_run_merge=not args.no_dry_run_merge,
        )
    elif args.cmd == "ingest-atr-weight-proposal":
        system = _legacy_write_context(args.config)
        feedback = json.loads(Path(args.feedback_json).read_text(encoding="utf-8"))
        out = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).run_atr_weight_proposal(feedback)
    elif args.cmd == "parse-proj":
        out = EvidenceToolAgent().parse_proj(args.path, max_bytes=args.max_bytes)
    elif args.cmd == "parse-jira":
        out = EvidenceToolAgent().parse_jira(args.value)
    elif args.cmd == "parse-attachment":
        out = EvidenceToolAgent().parse_attachment(args.path)
    elif args.cmd == "parse-document":
        out = EvidenceToolAgent().parse_document(args.path, max_bytes=args.max_bytes)
    elif args.cmd == "parse-image":
        out = EvidenceToolAgent().parse_image(args.path, max_bytes=args.max_bytes)
    elif args.cmd == "parse-evidence":
        out = EvidenceToolAgent().parse(args.tool, parse_json_payload(args.payload), max_bytes=args.max_bytes)
    elif args.cmd == "parse-evidence-context":
        out = EvidenceContextParserAgent().parse_context(args.root, max_bytes=args.max_bytes, limit=args.limit)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.cmd == "list-tools":
        from debug_agent_system.agents.tools import ToolRegistry, build_default_registry

        registry = build_default_registry()
        schemas = registry.schemas(args.style)
        if args.category:
            schemas = [item for item in schemas if _tool_category(item) == args.category]
        out = {
            "schema_version": registry.schema_version,
            "style": args.style,
            "count": len(schemas),
            "tools": schemas,
        }
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.cmd == "run-tool":
        from debug_agent_system.agents.tools import build_default_registry

        try:
            arguments = json.loads(args.arguments_json) if args.arguments_json.strip() else {}
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid arguments JSON: {exc.msg}") from exc
        if not isinstance(arguments, dict):
            raise SystemExit("arguments must decode to an object")
        out = build_default_registry().execute(args.tool, arguments)
    elif args.cmd == "sag-build":
        out = build_sqlite_sag(
            args.out,
            raw_root=args.raw_root,
            kg_root=args.kg_root,
            kg_v2_root=args.kg_v2_root,
            w1_root=None if args.no_w1 else args.w1_root,
            reset=True,
        )
        if args.report_out:
            Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report_out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.cmd == "review-decision":
        system = _legacy_write_context(args.config)
        v2_queue = args.queue in {"v2_typed_candidates", "atr_weight_proposals"}
        queue_store = JsonKGV2Store(args.kg_v2_root) if args.kg_version == "v2" or v2_queue else system.store
        out = ReviewQueueAgent(queue_store, queue_dir=args.queue_dir).mark_decision(
            args.queue,
            args.item_id,
            args.action,
            reviewer=args.reviewer,
            note=args.note,
        )
    elif args.cmd == "review-correction":
        try:
            correction_payload = json.loads(args.payload_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"invalid --payload-json: {exc.msg}"
            ) from exc
        if not isinstance(correction_payload, dict):
            raise SystemExit("--payload-json must decode to an object")
        out = ReviewQueueAgent(
            JsonKGV2Store(args.kg_v2_root),
            queue_dir=args.queue_dir,
        ).append_trace_correction(
            args.queue,
            args.item_id,
            args.operation,
            target_ref=args.target_ref,
            payload=correction_payload,
            evidence_message_ids=list(args.evidence_message_id),
            reviewer=args.reviewer,
            note=args.note,
        )
    elif args.cmd == "review-compile-corrections":
        out = ReviewQueueAgent(
            JsonKGV2Store(args.kg_v2_root),
            queue_dir=args.queue_dir,
        ).compile_trace_corrections(
            args.queue,
            args.item_id,
            quality_gate_scorer=(
                QualityGateAgent().score_typed_candidate
            ),
        )
    elif args.cmd == "apply-approved-queue":
        system = _legacy_write_context(args.config)
        apply_results = WriteSidePipeline(
            system.store,
            match_threshold=system.config.thresholds.graph_match_min_score,
            queue_dir=args.queue_dir,
            kg_v2_root=args.kg_v2_root,
            kg_v2_queue_dir=args.kg_v2_queue_dir,
        ).apply_approved_review_queue(kg_mode=args.kg_version)
        sag_publish_required = _kg_v2_sag_publish_required(apply_results, args.kg_version)
        if sag_publish_required:
            v2_pipeline = WriteSideV2Pipeline(args.kg_v2_root)
            out = {
                "apply_results": apply_results,
                "graph_validation": v2_pipeline.validate_current_graph(),
                "sag_build": {
                    "status": "skipped",
                    "reason": "skip_sag_build",
                    "index_stale": True,
                    "target": args.kg_v2_sag_out,
                } if args.skip_sag_build else v2_pipeline.build_sqlite_sag(args.kg_v2_sag_out, reset=True),
            }
        else:
            out = apply_results
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.cmd == "kg-v2-seed-sop":
        out = WriteSideV2Pipeline(args.kg_v2_root).seed_sop(args.chunks, limit=args.limit, replace=args.replace)
    elif args.cmd == "kg-v2-seed-manual":
        out = WriteSideV2Pipeline(args.kg_v2_root).seed_manual_cases(args.manual_root, limit=args.limit, replace=args.replace)
    elif args.cmd == "kg-v2-seed-all":
        out = WriteSideV2Pipeline(args.kg_v2_root).seed_all(
            chunks_path=args.chunks,
            manual_root=args.manual_root,
            sop_limit=args.sop_limit,
            manual_limit=args.manual_limit,
            replace=not args.no_replace,
        )
    elif args.cmd == "kg-v2-build-curated":
        out = WriteSideV2Pipeline(args.kg_v2_root).build_curated_sop(
            build_root=args.build_root,
            gold_root=args.gold_root,
            summary_out=args.summary_out,
            allow_active_rebuild=args.allow_active_rebuild,
        )
    elif args.cmd == "kg-v2-validate":
        out = WriteSideV2Pipeline(args.kg_v2_root).validate_current_graph()
    elif args.cmd == "kg-v2-materialize":
        out = KGv2Adapter(args.kg_v2_root).materialize_execution_view(args.out)
    elif args.cmd == "kg-v2-sag-build":
        out = WriteSideV2Pipeline(args.kg_v2_root).build_sqlite_sag(args.out, reset=not args.no_reset)
    elif args.cmd == "kg-v2-gold-compare":
        return kg_v2_gold_compare_main([
            "--gold-root", args.gold_root,
            "--kg-root", args.kg_root,
            "--runner-mode", args.runner_mode,
            *(["--deepseek"] if args.deepseek else []),
            *(["--emit-prompt-inputs"] if args.emit_prompt_inputs else []),
            *(["--out", args.out] if args.out else []),
        ])
    elif args.cmd == "kg-v2-overview":
        return kg_v2_overview_main([
            "--kg-v2-root", args.kg_v2_root,
            "--pinned-run-dir", args.pinned_run_dir,
            "--snapshot-out", args.snapshot_out,
            "--html-out", args.html_out,
        ])
    elif args.cmd == "w2-family-diagnostics":
        return w2_family_diagnostics_main([
            args.input_jsonl,
            *(["--out", args.out] if args.out else []),
            "--sample-limit", str(args.sample_limit),
        ])
    elif args.cmd == "w2-quality-diagnostics":
        return w2_quality_diagnostics_main([
            args.input_jsonl,
            *(["--out", args.out] if args.out else []),
            "--sample-limit", str(args.sample_limit),
        ])
    elif args.cmd == "w2-quality-gate":
        return w2_quality_gate_main([
            "--diagnostics", args.diagnostics,
            "--max-noncanonical-family-rate", str(args.max_noncanonical_family_rate),
            "--max-pseudo-family-rate", str(args.max_pseudo_family_rate),
            "--max-long-variant-rate", str(args.max_long_variant_rate),
            "--max-questionish-variant-rate", str(args.max_questionish_variant_rate),
            "--max-empty-case-rate", str(args.max_empty_case_rate),
            "--max-report-noise-rate", str(args.max_report_noise_rate),
            "--max-positive-status-rate", str(args.max_positive_status_rate),
            "--max-split-required-rate", str(args.max_split_required_rate),
            "--max-action-duplicates-rate", str(args.max_action_duplicates_rate),
        ])
    elif args.cmd == "w2-postrun-report":
        return w2_postrun_report_main([
            "--run-dir", args.run_dir,
            *(["--out", args.out] if args.out else []),
        ])
    elif args.cmd == "w2-postrun-compare":
        return w2_postrun_compare_main([
            "--base", args.base,
            "--candidate", args.candidate,
            *(["--out", args.out] if args.out else []),
        ])
    elif args.cmd == "w2-live-report":
        return w2_live_report_main([
            "--run-dir", args.run_dir,
            *(["--out", args.out] if args.out else []),
        ])
    elif args.cmd == "w2-live-compare":
        return w2_live_compare_main([
            "--base", args.base,
            "--candidate", args.candidate,
            *(["--out", args.out] if args.out else []),
        ])
    elif args.cmd == "w2-run-status":
        return w2_run_status_main([
            "--run-dir", args.run_dir,
        ])
    elif args.cmd == "w2-split-diagnostics":
        return w2_split_diagnostics_main([
            args.input_jsonl,
            *(["--out", args.out] if args.out else []),
            "--sample-limit", str(args.sample_limit),
        ])
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"unknown command: {args.cmd}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
