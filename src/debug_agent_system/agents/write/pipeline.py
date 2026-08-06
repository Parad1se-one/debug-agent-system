"""Write-side W1→W6 orchestration for real chat archives.

The pipeline is conservative by construction: it emits episodes, schema-valid
candidate drafts, review items, and dry-run merge plans.  It never mutates the
main KG unless a candidate/review item has already been human-approved and the
caller explicitly enables apply.
"""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any
from datetime import UTC, datetime

import debug_agent_system.agents.write.review_context as review_ctx
from debug_agent_system.agents.tools import EvidenceContextParserAgent
from debug_agent_system.agents.loop import ATRWeightingAgent, DiagnosticFeedbackAgent, LogPatternAgent
from debug_agent_system.agents.write.non_sop_intake import (
    NonSopIntakeError,
    SOP_INCREMENTAL_CONTRACT,
    build_write_intake_envelope,
    compute_kg_v2_graph_hash,
    is_sop_source_reference,
    load_alignment_context_index,
)
from debug_agent_system.agents.write.w1_chat_collect import ChatCollectAgent
from debug_agent_system.agents.write.w2_extract import KnowledgeExtractionAgent
from debug_agent_system.agents.write.w3_conflict import ConflictResolutionAgent
from debug_agent_system.agents.write.w4_quality_gate import QualityGateAgent
from debug_agent_system.agents.write.w5_incremental_ingest import IncrementalIngestAgent
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.agents.write.w9_raw_doc_ingest import RawDocIngestAgent
from debug_agent_system.agents.write.w10_section_case_bundle import SectionCaseBundleAgent
from debug_agent_system.agents.write_v2.expert_review import build_expert_corrected_candidate
from debug_agent_system.knowledge.store import KGStore
from debug_agent_system.knowledge_v2 import JsonKGV2Store, build_v2_bundle_from_legacy_candidate, validate_graph
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.agents.write_v2 import IncrementalIngestV2Agent
from debug_agent_system.agents.write.w7_trace.contracts import resolve_w7_mode
from debug_agent_system.agents.write.w7_trace.atomic_case_adapter import (
    w2_atomic_episodes,
    w7_case_cards_from_w2_candidates,
)
from debug_agent_system.agents.write.w7_trace.batch_orchestrator import (
    W7BatchShadowOrchestrator,
)
from debug_agent_system.agents.write.w7_trace.contracts import canonical_hash
from debug_agent_system.agents.write.w7_trace.orchestrator import (
    W7ShadowOrchestrator,
)


class WriteSidePipeline:
    """Conservative W1-W6 pipeline over W1 fault episodes."""

    def __init__(
        self,
        store: KGStore,
        *,
        match_threshold: float = 4.0,
        queue_dir: str | Path | None = None,
        kg_v2_root: str | Path | None = None,
        kg_v2_queue_dir: str | Path | None = None,
        w2_mode: str = "legacy_only",
        review_context_enabled: bool = True,
        w7_mode: str | None = None,
        review_context_sop_seed_json: str | Path = review_ctx.DEFAULT_SOP_SEED_JSON,
        review_context_gold_root: str | Path | None = None,
        review_context_manual_root: str | Path = review_ctx.DEFAULT_MANUAL_ROOT,
        w7_decision_client: Any | None = None,
        w7_shadow_out_dir: str | Path | None = None,
        w7_decision_workers: int = 1,
        w7_atomic_workers: int = 1,
        w7_batch_scope: str = "thread",
    ) -> None:
        self.store = store
        self.w1 = ChatCollectAgent()
        self.w2 = KnowledgeExtractionAgent(store, match_threshold=match_threshold, w2_mode=w2_mode)
        self.w3 = ConflictResolutionAgent()
        self.w4 = QualityGateAgent()
        self.w5 = IncrementalIngestAgent(store)
        self.w6 = ReviewQueueAgent(store, queue_dir=queue_dir)
        self.w7 = review_ctx.ReviewContextAgent()
        self.w7_mode = resolve_w7_mode(w7_mode)
        if self.w7_mode in {"assisted", "multi_agent"}:
            raise ValueError(
                f"w7_mode_not_yet_promotable:{self.w7_mode};"
                "use legacy or shadow_multi_agent"
            )
        self.w7_shadow_enabled = self.w7_mode == "shadow_multi_agent"
        self.w7_legacy_authoritative = True
        # The model client is injected by the caller so normal legacy runs do
        # not unexpectedly make network calls.  Shadow output is always kept
        # in a separate evaluation directory and never enters W6/W5.
        self.w7_decision_client = w7_decision_client
        self.w7_shadow_out_dir = (
            Path(w7_shadow_out_dir)
            if w7_shadow_out_dir is not None
            else None
        )
        self.w7_decision_workers = max(1, int(w7_decision_workers))
        self.w7_atomic_workers = max(1, int(w7_atomic_workers))
        if w7_batch_scope not in {"thread", "chat", "all"}:
            raise ValueError(f"unsupported_w7_batch_scope:{w7_batch_scope}")
        self.w7_batch_scope = w7_batch_scope
        self._last_w7_shadow_manifest: dict[str, Any] | None = None
        self.kg_v2_store = JsonKGV2Store(kg_v2_root) if kg_v2_root is not None else None
        self.w5_v2 = IncrementalIngestV2Agent(self.kg_v2_store) if self.kg_v2_store is not None else None
        self.w6_v2 = ReviewQueueAgent(self.kg_v2_store, queue_dir=kg_v2_queue_dir) if self.kg_v2_store is not None else None
        self.review_context_enabled = bool(review_context_enabled)
        self.review_context_sop_seed_json = Path(review_context_sop_seed_json)
        self.review_context_gold_root = (
            Path(review_context_gold_root)
            if review_context_gold_root is not None
            else (Path(kg_v2_root) / "gold_cases" if kg_v2_root is not None else Path(review_ctx.DEFAULT_GOLD_ROOT))
        )
        self.review_context_manual_root = Path(review_context_manual_root)
        default_alignment_root = Path("data/kg_v2")
        self.review_context_kg_v2_root = (
            Path(kg_v2_root)
            if kg_v2_root is not None
            else (default_alignment_root if default_alignment_root.exists() else None)
        )
        self._review_context_sop_cache: dict[str, Any] | None = None
        self._review_context_examples_cache: list[dict[str, Any]] | None = None
        self._review_context_alignment_index_cache: dict[str, Any] | None = None

    def run_w1_w2_xing_upload(
        self,
        import_root: str | Path,
        *,
        limit: int = 0,
        hits_only: bool = False,
        out_dir: str | Path | None = None,
        sample_limit: int = 5,
        emit_candidates: bool = False,
        w2_workers: int = 1,
        w2_mode: str | None = None,
    ) -> dict[str, Any]:
        """Run only W1 collection and W2 extraction over a Xing upload archive.

        This is intentionally narrower than ``run_xing_upload``: it does not run
        W3-W6, does not enqueue review items, and never mutates the main KG.  It
        exists so the write-side can inspect real archive output before tuning
        conflict/gate/merge behavior.
        """

        run = self.w1.import_xing_upload(import_root, limit=limit, hits_only=hits_only, out_dir=None)
        episodes = self._episodes_from_summaries(
            run["thread_summaries"],
            refine_trace=self.review_context_enabled,
        )
        candidates = self._extract_w2_candidates(episodes, workers=w2_workers, w2_mode=w2_mode)
        summary = self._w1_w2_summary(run, episodes, candidates)
        samples = self._w1_w2_samples(episodes, candidates, limit=sample_limit)
        output_files: dict[str, str] = {}
        if out_dir is not None:
            output_files = self._write_w1_w2_run(out_dir, run, candidates, summary, samples)
        return {
            "run_manifest": run["run_manifest"],
            "summary": summary,
            "samples": samples,
            "output_files": output_files,
            "candidates": candidates if emit_candidates else [],
        }

    def run_w2_summaries(
        self,
        summaries: list[dict[str, Any]],
        *,
        out_dir: str | Path | None = None,
        sample_limit: int = 5,
        emit_candidates: bool = False,
        w2_workers: int = 1,
        w2_mode: str | None = None,
    ) -> dict[str, Any]:
        episodes = self._episodes_from_summaries(
            summaries,
            refine_trace=self.review_context_enabled,
        )
        out = Path(out_dir) if out_dir is not None else None
        progress_path = out / "progress.json" if out is not None else None
        partial_candidates_path = out / "w2_candidates.partial.jsonl" if out is not None else None
        resumed_candidates = self._load_partial_candidates(partial_candidates_path) if partial_candidates_path is not None else {}
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        if progress_path is not None:
            progress_path.write_text(json.dumps({
                "status": "running",
                "thread_summaries": len(summaries),
                "episodes_total": len(episodes),
                "episodes_completed": len(resumed_candidates),
                "resumed_from_existing": len(resumed_candidates),
                "w2_mode": w2_mode or getattr(self.w2, "w2_mode", "legacy_only"),
                "partial_candidates_path": str(partial_candidates_path) if partial_candidates_path is not None else "",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        def progress_callback(completed: int, total: int) -> None:
            if progress_path is None:
                return
            progress_path.write_text(json.dumps({
                "status": "running",
                "thread_summaries": len(summaries),
                "episodes_total": total,
                "episodes_completed": completed,
                "resumed_from_existing": len(resumed_candidates),
                "w2_mode": w2_mode or getattr(self.w2, "w2_mode", "legacy_only"),
                "partial_candidates_path": str(partial_candidates_path) if partial_candidates_path is not None else "",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        candidates = self._extract_w2_candidates(
            episodes,
            workers=w2_workers,
            w2_mode=w2_mode,
            progress_callback=progress_callback,
            partial_candidates_path=partial_candidates_path,
            resumed_candidates=resumed_candidates,
        )
        case_issues: Counter[str] = Counter()
        draft_issues: Counter[str] = Counter()
        bundle_issues: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        subsystem_counts: Counter[str] = Counter()
        noncanonical_family_count = 0
        pseudo_family_count = 0
        long_variant_count = 0
        questionish_variant_count = 0
        extraction_error_counts: Counter[str] = Counter()
        approved_families = {
            "工控机蓝屏", "工控机异常重启", "用户配置加载失败", "运控初始化失败", "光源初始化失败",
            "主程序初始化卡住无明确报错", "相机拍摄失败", "相机初始化失败", "CAD 导入失败", "Mark 点对齐失败",
            "扫码识别失败", "界面显示异常", "误报调优异常", "漏检调优异常", "CT 时间异常增加", "复判站出图慢",
            "主程序/系统异常", "算法/程序调优异常",
        }
        pseudo_families = {
            "AOI_复判站", "AOI检测软件", "display", "camera", "software", "算法/检测逻辑",
            "显示/分辨率/缩放", "显示/界面", "复判流程",
        }
        for cand in candidates:
            extraction_error = cand.get("extraction_error") if isinstance(cand.get("extraction_error"), dict) else {}
            if extraction_error.get("code"):
                extraction_error_counts[str(extraction_error["code"])] += 1
            for issue in cand.get("case_understanding_card_schema_issues") or []:
                case_issues[str(issue)] += 1
            for issue in cand.get("candidate_draft_v2_schema_issues") or []:
                draft_issues[str(issue)] += 1
            for issue in cand.get("candidate_draft_v2_bundle_schema_issues") or []:
                bundle_issues[str(issue)] += 1
            split_cases = ((cand.get("candidate_draft_v2") or {}).get("split_cases") or [])
            first = split_cases[0] if split_cases and isinstance(split_cases[0], dict) else {}
            family = first.get("family") if isinstance(first.get("family"), dict) else {}
            variant = first.get("variant") if isinstance(first.get("variant"), dict) else {}
            family_label = str(family.get("label") or "")
            subsystem = str(family.get("subsystem") or "")
            variant_label = str(variant.get("label") or "")
            if family_label:
                family_counts[family_label] += 1
            if subsystem:
                subsystem_counts[subsystem] += 1
            if family_label and family_label not in approved_families:
                noncanonical_family_count += 1
            if family_label in pseudo_families:
                pseudo_family_count += 1
            if len(variant_label) > 40:
                long_variant_count += 1
            if variant_label.startswith(("我这个现场", "现场反馈", "客户反馈")) or variant_label.endswith(("是什么问题", "怎么处理", "怎么办", "如何处理", "吗", "么")):
                questionish_variant_count += 1
        summary = {
            "schema_version": "w2_summaries_run.v1",
            "w2_mode": w2_mode or getattr(self.w2, "w2_mode", "legacy_only"),
            "thread_summaries": len(summaries),
            "episodes": len(episodes),
            "legacy_schema_valid_candidates": sum(1 for c in candidates if c.get("schema_valid")),
            "production_schema_valid_candidates": sum(1 for c in candidates if c.get("production_schema_valid", c.get("schema_valid"))),
            "native_case_understanding_valid": sum(1 for c in candidates if c.get("case_understanding_card_schema_valid")),
            "native_candidate_draft_valid": sum(1 for c in candidates if c.get("candidate_draft_v2_schema_valid")),
            "native_bundle_valid": sum(1 for c in candidates if c.get("candidate_draft_v2_bundle_schema_valid")),
            "split_required": sum(1 for c in candidates if ((c.get("case_understanding_card") or {}).get("split_required"))),
            "deepseek_used": sum(1 for c in candidates if ((c.get("observability") or {}).get("deepseek_used"))),
            "resumed_from_existing": len(resumed_candidates),
            "top_case_understanding_issues": case_issues.most_common(10),
            "top_candidate_draft_issues": draft_issues.most_common(10),
            "top_bundle_issues": bundle_issues.most_common(10),
            "top_families": family_counts.most_common(20),
            "top_family_subsystems": subsystem_counts.most_common(20),
            "noncanonical_family_count": noncanonical_family_count,
            "pseudo_family_count": pseudo_family_count,
            "long_variant_count": long_variant_count,
            "questionish_variant_count": questionish_variant_count,
            "extraction_error_count": sum(extraction_error_counts.values()),
            "extraction_errors": extraction_error_counts.most_common(10),
        }
        samples = []
        for cand in candidates:
            if len(samples) >= max(0, sample_limit):
                break
            card = cand.get("case_understanding_card") or {}
            draft = cand.get("candidate_draft_v2") or {}
            split_cases = draft.get("split_cases") if isinstance(draft.get("split_cases"), list) else []
            first = split_cases[0] if split_cases and isinstance(split_cases[0], dict) else {}
            family = first.get("family") if isinstance(first.get("family"), dict) else {}
            variant = first.get("variant") if isinstance(first.get("variant"), dict) else {}
            actions = first.get("actions") if isinstance(first.get("actions"), list) else []
            reqs = first.get("required_info") if isinstance(first.get("required_info"), list) else []
            samples.append({
                "candidate_id": cand.get("candidate_id"),
                "label": cand.get("label"),
                "case_understanding_valid": cand.get("case_understanding_card_schema_valid"),
                "candidate_draft_valid": cand.get("candidate_draft_v2_schema_valid"),
                "production_schema_valid": cand.get("production_schema_valid", cand.get("schema_valid")),
                "split_required": card.get("split_required"),
                "family": family.get("label", ""),
                "variant": variant.get("label", ""),
                "action_count": len(actions),
                "required_info_count": len(reqs),
            })
        output_files: dict[str, str] = {}
        if out is not None:
            cand_path = out / "w2_candidates.jsonl"
            with cand_path.open("w", encoding="utf-8") as f:
                for cand in candidates:
                    f.write(json.dumps(cand, ensure_ascii=False) + "\n")
            (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if progress_path is not None:
                progress_path.write_text(json.dumps({
                    "status": "completed",
                    "thread_summaries": len(summaries),
                    "episodes_total": len(episodes),
                    "episodes_completed": len(episodes),
                    "resumed_from_existing": len(resumed_candidates),
                    "w2_mode": w2_mode or getattr(self.w2, "w2_mode", "legacy_only"),
                    "summary_path": str(out / "summary.json"),
                    "candidates_path": str(cand_path),
                    "partial_candidates_path": str(partial_candidates_path) if partial_candidates_path is not None else "",
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output_files = {
                "candidates": str(cand_path),
                "summary": str(out / "summary.json"),
                "samples": str(out / "samples.json"),
                "progress": str(progress_path) if progress_path is not None else "",
                "partial_candidates": str(partial_candidates_path) if partial_candidates_path is not None else "",
            }
        return {
            "summary": summary,
            "samples": samples,
            "output_files": output_files,
            "candidates": candidates if emit_candidates else [],
        }

    def run_xing_upload(
        self,
        import_root: str | Path,
        *,
        limit: int = 0,
        hits_only: bool = False,
        out_dir: str | Path | None = None,
        apply_approved: bool = False,
        emit_episodes: bool = False,
        dry_run_merge: bool = True,
        w2_workers: int = 1,
        kg_mode: str = "legacy",
        w2_mode: str | None = None,
    ) -> dict[str, Any]:
        non_sop_manifest = self._preflight_non_sop_kg_v2(kg_mode) if kg_mode in {"v2", "both"} else {}
        run = self.w1.import_xing_upload(import_root, limit=limit, hits_only=hits_only, out_dir=out_dir)
        progress_path = Path(out_dir) / "pipeline_progress.json" if out_dir is not None else None
        summary_path = Path(out_dir) / "pipeline_summary.json" if out_dir is not None else None
        partial_candidates_path = Path(out_dir) / "w2_candidates.partial.jsonl" if out_dir is not None else None
        self._write_progress(
            progress_path,
            stage="w1_completed",
            payload={
                "messages": run["run_manifest"]["counts"]["messages"],
                "threads": run["run_manifest"]["counts"]["threads"],
                "episodes": run["run_manifest"]["counts"]["episodes"],
                "attachments": run["run_manifest"]["counts"]["attachments"],
            },
        )
        pipeline = self.run_summaries(
            run["thread_summaries"],
            apply_approved=apply_approved,
            emit_episodes=emit_episodes,
            dry_run_merge=dry_run_merge,
            w2_workers=w2_workers,
            kg_mode=kg_mode,
            w2_mode=w2_mode,
            source_type="chat",
            progress_path=progress_path,
            partial_candidates_path=partial_candidates_path,
        )
        run_manifest = dict(run["run_manifest"])
        if non_sop_manifest:
            run_manifest["non_sop_incremental"] = non_sop_manifest
        result = {"run_manifest": run_manifest, **pipeline}
        self._write_progress(progress_path, stage="completed", payload={"summary": pipeline.get("summary") or {}, "review_summary": pipeline.get("review_summary") or {}})
        if summary_path is not None:
            summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def run_text_history(
        self,
        import_root: str | Path,
        *,
        limit: int = 0,
        out_dir: str | Path | None = None,
        apply_approved: bool = False,
        emit_episodes: bool = False,
        dry_run_merge: bool = True,
        w2_workers: int = 1,
        kg_mode: str = "v2",
        w2_mode: str | None = "native_v2",
    ) -> dict[str, Any]:
        """Run text-only history through the same non-SOP W1→W7→W2→W6 path."""

        non_sop_manifest = self._preflight_non_sop_kg_v2(kg_mode) if kg_mode in {"v2", "both"} else {}
        run = self.w1.import_text_history(import_root, limit=limit, out_dir=out_dir)
        progress_path = Path(out_dir) / "pipeline_progress.json" if out_dir is not None else None
        summary_path = Path(out_dir) / "pipeline_summary.json" if out_dir is not None else None
        partial_candidates_path = Path(out_dir) / "w2_candidates.partial.jsonl" if out_dir is not None else None
        pipeline = self.run_summaries(
            run["thread_summaries"],
            apply_approved=apply_approved,
            emit_episodes=emit_episodes,
            dry_run_merge=dry_run_merge,
            w2_workers=w2_workers,
            kg_mode=kg_mode,
            w2_mode=w2_mode,
            source_type="text_history",
            progress_path=progress_path,
            partial_candidates_path=partial_candidates_path,
        )
        run_manifest = dict(run["run_manifest"])
        if non_sop_manifest:
            run_manifest["non_sop_incremental"] = non_sop_manifest
        result = {"run_manifest": run_manifest, **pipeline}
        self._write_progress(progress_path, stage="completed", payload={"summary": pipeline.get("summary") or {}, "review_summary": pipeline.get("review_summary") or {}})
        if summary_path is not None:
            summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def run_non_sop_document(
        self,
        path: str | Path,
        *,
        dry_run_merge: bool = True,
        split_review_scopes: bool = False,
    ) -> dict[str, Any]:
        """Run one non-SOP raw document through W9→W10→W3→W4→W6."""

        return self._run_versioned_document(
            path,
            source_type="raw_doc",
            source_kind="raw_doc",
            allow_sop=False,
            dry_run_merge=dry_run_merge,
            split_review_scopes=split_review_scopes,
        )

    def run_sop_document(
        self,
        path: str | Path,
        *,
        dry_run_merge: bool = True,
        split_review_scopes: bool = True,
    ) -> dict[str, Any]:
        """Stage one SOP source as a hash-bound, approved-only update."""

        return self._run_versioned_document(
            path,
            source_type="sop_doc",
            source_kind="sop",
            allow_sop=True,
            dry_run_merge=dry_run_merge,
            split_review_scopes=split_review_scopes,
        )

    def _run_versioned_document(
        self,
        path: str | Path,
        *,
        source_type: str,
        source_kind: str,
        allow_sop: bool,
        dry_run_merge: bool,
        split_review_scopes: bool,
    ) -> dict[str, Any]:
        """Shared W9/W10 document path with an explicit source contract."""

        doc_path = Path(path)
        name_lower = doc_path.name.lower()
        normalized_path = doc_path.as_posix().lower()
        source_is_sop = (
            is_sop_source_reference(normalized_path)
            or is_sop_source_reference(name_lower)
        )
        if source_is_sop and not allow_sop:
            raise NonSopIntakeError("sop_source_rejected", "SOP documents are excluded from non-SOP incremental ingestion.", {"path": str(doc_path)})
        if allow_sop and "data/kg_v2_sop_draft_build" in normalized_path:
            raise NonSopIntakeError(
                "sop_build_path_rejected",
                "SOP incremental sources must be raw documents, not curated build artifacts.",
                {"path": str(doc_path)},
            )
        manifest = self._preflight_non_sop_kg_v2("v2")
        inspection = RawDocIngestAgent().inspect_document(doc_path)
        if inspection.get("excluded_from_w9") and not allow_sop:
            raise NonSopIntakeError("sop_source_rejected", "SOP documents are excluded from non-SOP incremental ingestion.", {"path": str(doc_path)})
        if allow_sop and not (
            source_is_sop or bool(inspection.get("excluded_from_w9"))
        ):
            raise NonSopIntakeError(
                "sop_source_required",
                "The SOP incremental entrypoint only accepts explicitly identified SOP documents.",
                {"path": str(doc_path)},
            )
        section_payload = RawDocIngestAgent().build_section_cases(doc_path)
        strategy = section_payload.get("strategy") if isinstance(section_payload.get("strategy"), dict) else {}
        output_mode = str(strategy.get("kg_output_mode") or "review_only")
        knowledge_kind, admission_target = self._document_route(output_mode)
        bundle_agent = SectionCaseBundleAgent()
        raw_bundle = bundle_agent.build_bundle(section_payload)
        bundle = self._mark_bundle_as_document_source(
            self.w3.normalize_v2_bundle(raw_bundle),
            doc_path,
            source_kind=source_kind,
        )
        atomic_mapping_bundles = [
            self._mark_bundle_as_document_source(
                self.w3.normalize_v2_bundle(atomic_bundle),
                doc_path,
                source_kind=source_kind,
            )
            for atomic_bundle in (
                bundle_agent.build_atomic_case_bundles(section_payload)
                if allow_sop
                else []
            )
        ]
        chunk_manifest = bundle.get("chunk_manifest") if isinstance(bundle.get("chunk_manifest"), dict) else {}
        chunk_manifest_ref = self._chunk_manifest_ref(chunk_manifest)
        text = self._document_text(section_payload)
        envelope = build_write_intake_envelope(
            source_type=source_type,
            source_kind=source_kind,
            text=text,
            source_ref={"path": str(doc_path), "name": doc_path.name},
            knowledge_kind=knowledge_kind,
            payload={
                "text": text,
                "candidate_id": str(bundle.get("candidate_id") or bundle.get("bundle_id") or ""),
                "objects": bundle.get("objects") or {},
                "relations": bundle.get("relations") or [],
                "schema_valid": bool(bundle.get("schema_valid")),
                "schema_issues": list(bundle.get("schema_issues") or []),
                "strategy": bundle.get("strategy") if isinstance(bundle.get("strategy"), dict) else {},
                "section_cases": section_payload.get("section_cases") or [],
                "chunk_manifest": chunk_manifest,
            },
            evidence_pack={
                "source_path": str(doc_path),
                "structured_sections": section_payload.get("structured_sections") or [],
                "chunk_manifest_ref": chunk_manifest_ref,
            },
            lineage={"agent_id": "W9/W10", "output_mode": output_mode},
            metadata={
                "incremental_source_contract": (
                    SOP_INCREMENTAL_CONTRACT if allow_sop else "raw_document_incremental.v1"
                )
            },
        )
        envelope.update({
            "candidate_id": str(bundle.get("candidate_id") or bundle.get("bundle_id") or envelope["intake_id"]),
            "schema_valid": bool(bundle.get("schema_valid")),
            "schema_issues": list(bundle.get("schema_issues") or []),
            "admission_target": admission_target,
            "operation": "merge_graph",
        })
        if split_review_scopes:
            scoped_results: list[dict[str, Any]] = []
            review_items: list[dict[str, Any]] = []
            scope_specs: list[tuple[str, dict[str, Any]]] = [
                ("document_layer", self._document_layer_bundle(bundle))
            ]
            if allow_sop:
                scope_specs.extend(
                    ("fault_mapping", atomic_bundle)
                    for atomic_bundle in atomic_mapping_bundles
                    if self._has_fault_mapping(atomic_bundle)
                )
            elif self._has_fault_mapping(bundle):
                scope_specs.append(("fault_mapping", bundle))
            for review_scope, scoped_bundle in scope_specs:
                atomic_case = (
                    scoped_bundle.get("atomic_case")
                    if isinstance(scoped_bundle.get("atomic_case"), dict)
                    else {}
                )
                atomic_case_id = str(atomic_case.get("atomic_case_id") or "")
                scope_identity = (
                    str(scoped_bundle.get("bundle_id") or atomic_case_id)
                    if atomic_case_id
                    else review_scope
                )
                scoped_candidate_id = (
                    f"{str(bundle.get('candidate_id') or bundle.get('bundle_id') or envelope['intake_id'])}:{review_scope}:{scope_identity}"
                )
                scoped_text = text
                if atomic_case:
                    action_labels = [
                        str(item.get("label") or "")
                        for item in (scoped_bundle.get("objects") or {}).get(
                            "DiagnosticAction"
                        ) or []
                        if isinstance(item, dict)
                    ]
                    scoped_text = "\n".join(
                        item
                        for item in [
                            str(atomic_case.get("section_title") or ""),
                            str(atomic_case.get("family_label") or ""),
                            str(atomic_case.get("variant_label") or ""),
                            *action_labels,
                        ]
                        if item
                    )
                scoped_envelope = build_write_intake_envelope(
                    source_type=source_type,
                    source_kind=source_kind,
                    text=scoped_text,
                    source_ref={"path": str(doc_path), "name": doc_path.name},
                    knowledge_kind=knowledge_kind,
                    payload={
                        "text": scoped_text,
                        "candidate_id": scoped_candidate_id,
                        "objects": scoped_bundle.get("objects") or {},
                        "relations": scoped_bundle.get("relations") or [],
                        "schema_valid": bool(scoped_bundle.get("schema_valid")),
                        "schema_issues": list(scoped_bundle.get("schema_issues") or []),
                        "strategy": scoped_bundle.get("strategy") if isinstance(scoped_bundle.get("strategy"), dict) else {},
                        "section_cases": (
                            [atomic_case] if atomic_case else section_payload.get("section_cases") or []
                        ),
                        "atomic_case": atomic_case,
                        "chunk_manifest": chunk_manifest if review_scope == "document_layer" else {},
                        "review_scope": review_scope,
                    },
                    evidence_pack={
                        "source_path": str(doc_path),
                        "structured_sections": (
                            [
                                item
                                for item in section_payload.get("structured_sections") or []
                                if isinstance(item, dict)
                                and str(item.get("section_id") or "")
                                == str(atomic_case.get("section_id") or "")
                            ]
                            if atomic_case
                            else section_payload.get("structured_sections") or []
                        ),
                        "chunk_manifest_ref": chunk_manifest_ref,
                        "atomic_case": atomic_case,
                    },
                    lineage={
                        "agent_id": "W9/W10",
                        "output_mode": output_mode,
                        "review_scope": review_scope,
                        "atomic_case_id": atomic_case_id,
                    },
                    metadata={
                        "review_scope": review_scope,
                        "atomic_case_id": atomic_case_id,
                        "incremental_source_contract": (
                            SOP_INCREMENTAL_CONTRACT if allow_sop else "raw_document_incremental.v1"
                        ),
                    },
                )
                scoped_envelope.update({
                    "candidate_id": scoped_candidate_id,
                    "schema_valid": bool(scoped_bundle.get("schema_valid")),
                    "schema_issues": list(scoped_bundle.get("schema_issues") or []),
                    "admission_target": admission_target,
                    "operation": "merge_graph",
                    "review_scope": review_scope,
                })
                prepared = self._prepare_typed_envelope(scoped_envelope, dry_run_merge=dry_run_merge)
                prepared["review_item"]["review_scope"] = review_scope
                prepared["review_item"]["observability"]["review_scope"] = review_scope
                if atomic_case_id:
                    prepared["review_item"]["atomic_case"] = atomic_case
                    prepared["review_item"]["observability"]["atomic_case_id"] = atomic_case_id
                review_items.append(prepared["review_item"])
                scoped_results.append(prepared)
            queue_write = self.w6_v2.enqueue_many("v2_typed_candidates", review_items)
            queued = next(
                (item for item in scoped_results if item["review_item"].get("review_scope") == "fault_mapping"),
                scoped_results[0],
            )
            queued = {**queued, "review_scopes": scoped_results, "queue_write": queue_write}
        else:
            queued = self._queue_typed_envelope(envelope, dry_run_merge=dry_run_merge)
        return {
            "run_manifest": {**manifest, "source_type": source_type, "source_path": str(doc_path), "w9_output_mode": output_mode},
            "section_case_count": len(section_payload.get("section_cases") or []),
            "bundle": bundle,
            **queued,
        }

    def run_sop_documents(
        self,
        root: str | Path,
        *,
        limit: int = 0,
        dry_run_merge: bool = True,
    ) -> dict[str, Any]:
        """Detect SOP source revisions and queue only changed documents.

        This is an automatic *update detector and candidate producer*.  It
        deliberately stops at W6; approval and W5 apply remain separate.
        """

        if self.kg_v2_store is None:
            raise NonSopIntakeError(
                "kg_v2_pipeline_unavailable",
                "SOP document sync requires an active KG v2 root.",
            )
        checklist = RawDocIngestAgent().build_root_checklist(
            root,
            include_sop=True,
        )
        documents = [
            item
            for item in checklist.get("documents") or []
            if isinstance(item, dict) and item.get("excluded_from_w9")
        ]
        if limit > 0:
            documents = documents[:limit]
        current_documents = {
            str(item.get("source_path") or ""): item
            for item in self.kg_v2_store.objects_by_type.get(
                "KnowledgeDocument"
            ) or []
            if isinstance(item, dict)
            and str(item.get("source_path") or "")
        }
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for document in documents:
            path = Path(str(document.get("path") or ""))
            try:
                source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                counts["error"] += 1
                rows.append({
                    "path": str(path),
                    "status": "error",
                    "error": {
                        "code": type(exc).__name__,
                        "message": str(exc),
                    },
                })
                continue
            current = current_documents.get(str(path)) or {}
            if (
                current.get("approved") is True
                and str(current.get("content_hash") or "") == source_hash
            ):
                counts["unchanged"] += 1
                rows.append({
                    "path": str(path),
                    "status": "unchanged",
                    "content_hash": source_hash,
                    "document_id": str(current.get("document_id") or ""),
                })
                continue
            try:
                result = self.run_sop_document(
                    path,
                    dry_run_merge=dry_run_merge,
                    split_review_scopes=True,
                )
            except Exception as exc:  # noqa: BLE001 - retain per-source audit
                counts["error"] += 1
                rows.append({
                    "path": str(path),
                    "status": "error",
                    "content_hash": source_hash,
                    "error": {
                        "code": type(exc).__name__,
                        "message": str(exc),
                    },
                })
                continue
            counts["queued_update" if current else "queued_new"] += 1
            rows.append({
                "path": str(path),
                "status": "queued_update" if current else "queued_new",
                "content_hash": source_hash,
                "review_scopes": [
                    {
                        "review_scope": str(
                            scoped.get("review_item", {}).get(
                                "review_scope"
                            ) or ""
                        ),
                        "review_id": str(
                            scoped.get("review_item", {}).get("review_id")
                            or ""
                        ),
                    }
                    for scoped in result.get("review_scopes") or []
                    if isinstance(scoped, dict)
                ],
            })
        return {
            "schema_version": "sop_document_sync.v1",
            "source_root": str(root),
            "incremental_source_contract": SOP_INCREMENTAL_CONTRACT,
            "summary": {
                "source_count": len(documents),
                **dict(sorted(counts.items())),
            },
            "documents": rows,
            "apply_requires_human_approval": True,
        }

    def run_non_sop_documents(
        self,
        root: str | Path,
        *,
        manifest_path: str | Path = "data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json",
        limit: int = 0,
        dry_run_merge: bool = True,
    ) -> dict[str, Any]:
        """Batch W9→W10→W3→W4→W6 for documents not already in the SOP seed.

        This is a review-queue producer only.  It never calls W5 apply and it
        excludes SOP sources before parsing.
        """

        # The SOP source manifest also contains several non-SOP manuals used by
        # an older seed builder.  Excluding by manifest membership would drop
        # those manuals from the new document layer.  This batch therefore
        # scans every raw document and excludes by source semantics only.
        checklist = RawDocIngestAgent().build_root_checklist(root, include_sop=False)
        documents = [item for item in checklist.get("documents") or [] if isinstance(item, dict)]
        if limit > 0:
            documents = documents[:limit]
        rows: list[dict[str, Any]] = []
        totals: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        targets: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        relation_count = 0
        schema_invalid = 0
        materialize_allowed = 0
        seen_semantic_documents: dict[str, dict[str, Any]] = {}
        for document in documents:
            path = str(document.get("path") or "")
            try:
                section_payload = RawDocIngestAgent().build_section_cases(path)
                semantic_hash = self._semantic_document_hash(section_payload)
                if semantic_hash in seen_semantic_documents:
                    canonical = seen_semantic_documents[semantic_hash]
                    alias_bundle = self._semantic_duplicate_evidence_bundle(
                        duplicate_path=path,
                        semantic_hash=semantic_hash,
                        canonical_path=str(canonical["path"]),
                        canonical_document=dict(canonical["document"]),
                    )
                    alias_text = (
                        f"{Path(path).name} 与 {Path(str(canonical['path'])).name} "
                        "语义内容相同；本条只登记来源别名，不复制文档语义节点。"
                    )
                    alias_envelope = build_write_intake_envelope(
                        source_type="raw_doc",
                        text=alias_text,
                        source_ref={"path": path, "name": Path(path).name},
                        knowledge_kind="evidence_only",
                        payload={
                            "text": alias_text,
                            "candidate_id": alias_bundle["candidate_id"],
                            "objects": alias_bundle["objects"],
                            "relations": alias_bundle["relations"],
                            "schema_valid": alias_bundle["schema_valid"],
                            "schema_issues": alias_bundle["schema_issues"],
                            "review_scope": "duplicate_source_alias",
                        },
                        evidence_pack={
                            "source_path": path,
                            "semantic_content_hash": semantic_hash,
                            "duplicate_of": str(canonical["path"]),
                        },
                        lineage={"agent_id": "W9", "output_mode": "duplicate_source_alias"},
                        metadata={"review_scope": "duplicate_source_alias"},
                    )
                    alias_envelope.update({
                        "candidate_id": alias_bundle["candidate_id"],
                        "schema_valid": alias_bundle["schema_valid"],
                        "schema_issues": alias_bundle["schema_issues"],
                        "admission_target": "evidence_only",
                        "operation": "merge_graph",
                        "review_scope": "duplicate_source_alias",
                    })
                    prepared = self._prepare_typed_envelope(alias_envelope, dry_run_merge=dry_run_merge)
                    prepared["review_item"]["review_scope"] = "duplicate_source_alias"
                    prepared["review_item"]["observability"]["review_scope"] = "duplicate_source_alias"
                    queue_write = self.w6_v2.enqueue("v2_typed_candidates", prepared["review_item"])
                    statuses["duplicate"] += 1
                    totals["KnowledgeDocument"] += 0
                    totals["EvidenceItem"] += 1
                    relation_count += 1
                    decisions[str(prepared["quality_gate"].get("decision") or "unknown")] += 1
                    targets["evidence_only"] += 1
                    rows.append({
                        "path": path,
                        "name": Path(path).name,
                        "status": "duplicate",
                        "semantic_content_hash": semantic_hash,
                        "duplicate_of": str(canonical["path"]),
                        "source_alias_queued": True,
                        "review_id": str(prepared["review_item"].get("review_id") or ""),
                        "review_scope": "duplicate_source_alias",
                        "decision": str(prepared["quality_gate"].get("decision") or ""),
                        "queue_write": queue_write,
                    })
                    continue
                result = self.run_non_sop_document(
                    path,
                    dry_run_merge=dry_run_merge,
                    split_review_scopes=True,
                )
                bundle = result.get("bundle") if isinstance(result.get("bundle"), dict) else {}
                objects = bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {}
                gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
                review_item = result.get("review_item") if isinstance(result.get("review_item"), dict) else {}
                object_counts = {
                    object_type: len([item for item in items or [] if isinstance(item, dict)])
                    for object_type, items in sorted(objects.items())
                }
                for object_type, count in object_counts.items():
                    totals[object_type] += count
                relations = bundle.get("relations") if isinstance(bundle.get("relations"), list) else []
                canonical_documents = [
                    item for item in objects.get("KnowledgeDocument") or []
                    if isinstance(item, dict) and item.get("document_id")
                ]
                if canonical_documents:
                    seen_semantic_documents[semantic_hash] = {
                        "path": path,
                        "document": canonical_documents[0],
                    }
                relation_count += len(relations)
                decision = str(gate.get("decision") or "unknown")
                target = str(gate.get("admission_target") or result.get("admission_target") or "unknown")
                valid = bool(bundle.get("schema_valid"))
                decisions[decision] += 1
                targets[target] += 1
                statuses["ok"] += 1
                schema_invalid += int(not valid)
                materialize_allowed += int(bool(gate.get("materialize_allowed")))
                rows.append({
                    "path": path,
                    "name": Path(path).name,
                    "status": "ok",
                    "semantic_content_hash": semantic_hash,
                    "strategy": str(((document.get("strategy") or {}).get("strategy_id")) or ""),
                    "decision": decision,
                    "target": target,
                    "materialize_allowed": bool(gate.get("materialize_allowed")),
                    "schema_valid": valid,
                    "schema_issues": list(bundle.get("schema_issues") or []),
                    "gate_issues": list(gate.get("issues") or []),
                    "semantic_issues": list(((gate.get("kg_v2_semantic_gate") or {}).get("issues")) or []),
                    "review_id": str(review_item.get("review_id") or ""),
                    "review_scopes": [
                        {
                            "review_scope": str(item.get("review_item", {}).get("review_scope") or ""),
                            "review_id": str(item.get("review_item", {}).get("review_id") or ""),
                            "decision": str(item.get("quality_gate", {}).get("decision") or ""),
                        }
                        for item in result.get("review_scopes") or []
                        if isinstance(item, dict)
                    ],
                    "object_counts": object_counts,
                    "relation_count": len(relations),
                })
            except NonSopIntakeError as exc:
                statuses["excluded"] += 1
                rows.append({
                    "path": path,
                    "name": Path(path).name,
                    "status": "excluded",
                    "error": exc.to_dict(),
                })
            except Exception as exc:  # noqa: BLE001 - batch must preserve per-document failures
                statuses["error"] += 1
                rows.append({
                    "path": path,
                    "name": Path(path).name,
                    "status": "error",
                    "error": {"code": type(exc).__name__, "message": str(exc)},
                })
        return {
            "schema_version": "non_sop_document_batch.v1",
            "run_manifest": {
                "build_mode": "non_sop_incremental",
                "source_root": str(root),
                "manifest_path": str(manifest_path),
                "manifest_used_for_exclusion": False,
                "sop_excluded": True,
                "dry_run_merge": dry_run_merge,
            },
            "summary": {
                "documents": len(documents),
                "status": dict(sorted(statuses.items())),
                "decisions": dict(sorted(decisions.items())),
                "targets": dict(sorted(targets.items())),
                "schema_invalid": schema_invalid,
                "materialize_allowed": materialize_allowed,
                "object_totals": dict(sorted(totals.items())),
                "relations": relation_count,
            },
            "rows": rows,
        }

    @staticmethod
    def _semantic_duplicate_evidence_bundle(
        *,
        duplicate_path: str,
        semantic_hash: str,
        canonical_path: str,
        canonical_document: dict[str, Any],
    ) -> dict[str, Any]:
        """Represent a duplicate source without duplicating its semantic graph."""

        document_id = str(canonical_document.get("document_id") or "")
        digest = hashlib.sha256(
            f"{duplicate_path}|{semantic_hash}|{document_id}".encode("utf-8")
        ).hexdigest()[:20]
        evidence_id = f"evidence:duplicate-document:{digest}"
        evidence = {
            "evidence_id": evidence_id,
            "source_kind": "tool_parse",
            "external_id": semantic_hash[:32],
            "title": Path(duplicate_path).name[:80],
            "summary": (
                f"语义内容与 {Path(canonical_path).name} 相同；保留该来源路径作为证据别名，"
                "不重复生成 KnowledgeSection、ProcedureStep 或故障语义节点。"
            )[:500],
            "payload_ref": duplicate_path[:200],
        }
        objects = {
            "KnowledgeDocument": [canonical_document],
            "EvidenceItem": [evidence],
        }
        relations = [{"from": evidence_id, "to": document_id, "relation": "evidences"}]
        issues = validate_graph(objects, relations)
        return {
            "candidate_id": f"w9:duplicate-source:{digest}",
            "objects": objects,
            "relations": relations,
            "schema_valid": not issues,
            "schema_issues": issues,
        }

    @staticmethod
    def _has_fault_mapping(bundle: dict[str, Any]) -> bool:
        objects = bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {}
        return any(objects.get(key) for key in (
            "FaultFamily", "FaultVariant", "DiagnosticAction", "RequiredInfoSpec", "ActionOutcome", "DiagnosticTrace"
        ))

    @staticmethod
    def _semantic_document_hash(payload: dict[str, Any]) -> str:
        ignored = {"section_id", "case_id", "source_doc_id", "source_doc_title", "path", "name"}

        def canonical(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    str(key): canonical(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                    if str(key) not in ignored
                }
            if isinstance(value, list):
                return [canonical(item) for item in value]
            return value

        section_cases = payload.get("section_cases") or []
        semantic = {
            "strategy": str(((payload.get("strategy") or {}).get("strategy_id")) or ""),
            # section_case is the normalized semantic representation and strips
            # DOCX-vs-Markdown bullet syntax.  Fall back to raw sections only
            # for reference documents that intentionally have no cases.
            "section_cases": section_cases,
            "structured_sections": [] if section_cases else (payload.get("structured_sections") or []),
        }
        encoded = json.dumps(canonical(semantic), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_manifest_ref(manifest: dict[str, Any]) -> dict[str, Any]:
        if not manifest:
            return {}
        stats = manifest.get("stats") if isinstance(manifest.get("stats"), dict) else {}
        return {
            "manifest_id": str(manifest.get("manifest_id") or ""),
            "manifest_hash": str(manifest.get("manifest_hash") or ""),
            "schema_version": str(manifest.get("schema_version") or ""),
            "chunker_version": str(manifest.get("chunker_version") or ""),
            "binding_status": str(manifest.get("binding_status") or ""),
            "source_path": str(manifest.get("source_path") or ""),
            "source_file_hash": str(manifest.get("source_file_hash") or ""),
            "chunk_count": int(stats.get("chunk_count") or len(manifest.get("chunks") or [])),
            "bound_section_count": int(stats.get("bound_section_count") or 0),
        }

    @staticmethod
    def _document_layer_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
        allowed_types = {"KnowledgeDocument", "KnowledgeSection", "ProcedureStep", "EvidenceItem"}
        source_objects = bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {}
        objects = {
            object_type: list(items or []) if object_type in allowed_types else []
            for object_type, items in source_objects.items()
        }
        valid_ids = {
            str(item.get(V2_PRIMARY_KEYS[object_type]) or "")
            for object_type in allowed_types
            for item in objects.get(object_type) or []
            if isinstance(item, dict) and str(item.get(V2_PRIMARY_KEYS[object_type]) or "")
        }
        relations = [
            relation
            for relation in bundle.get("relations") or []
            if isinstance(relation, dict)
            and str(relation.get("from") or "") in valid_ids
            and str(relation.get("to") or "") in valid_ids
        ]
        issues = validate_graph(objects, relations)
        return {
            **bundle,
            "candidate_id": f"{str(bundle.get('candidate_id') or bundle.get('bundle_id') or 'document')}:document_layer",
            "objects": objects,
            "relations": relations,
            "schema_valid": not issues,
            "schema_issues": issues,
        }

    def run_evidence_context(
        self,
        root: str | Path,
        *,
        max_bytes: int = 65536,
        limit: int = 0,
        dry_run_merge: bool = True,
    ) -> dict[str, Any]:
        """Parse Jira/attachment contexts and queue evidence-only KG v2 candidates."""

        manifest = self._preflight_non_sop_kg_v2("v2")
        parsed = EvidenceContextParserAgent().parse_context(root, max_bytes=max_bytes, limit=limit)
        results: list[dict[str, Any]] = []
        review_items: list[dict[str, Any]] = []
        for context in parsed.get("contexts") or []:
            if not isinstance(context, dict):
                continue
            envelope = self._evidence_context_envelope(context)
            prepared = self._prepare_typed_envelope(envelope, dry_run_merge=dry_run_merge)
            review_items.append(prepared["review_item"])
            results.append(prepared)
        queue_write = self.w6_v2.enqueue_many("v2_typed_candidates", review_items) if review_items else {
            "status": "batch_written", "queue": "v2_typed_candidates.json", "queued": 0, "updated": 0
        }
        return {
            "run_manifest": {**manifest, "source_type": "evidence_context", "source_root": str(root)},
            "parse_result": parsed,
            "queued_contexts": len(results),
            "queue_write": queue_write,
            "results": results,
        }

    def run_expert_correction(
        self,
        review_item: dict[str, Any],
        correction: dict[str, Any],
        *,
        dry_run_merge: bool = True,
    ) -> dict[str, Any]:
        """Re-enter an expert correction through W4/W6 as a fresh candidate.

        Expert review is an evidence-producing source adapter, not an approval
        shortcut.  Even when the reviewer supplied a complete corrected graph,
        the new or rebound candidate must receive a new identity and pass the
        same typed/semantic gates as every other non-SOP source.
        """

        manifest = self._preflight_non_sop_kg_v2("v2")
        # Expert correction supplies semantics, but W3 still owns canonical
        # IDs, duplicate removal and reference rewriting.  Keeping this hop
        # explicit prevents the human-review path from becoming a shortcut
        # around the normal W2 -> W3 -> W4 -> W6 -> W5 control plane.
        candidate = self.w3.normalize_v2_bundle(
            build_expert_corrected_candidate(review_item, correction)
        )
        source_messages = [
            dict(item)
            for item in candidate.get("source_messages") or []
            if isinstance(item, dict)
        ]
        message_ids = list(dict.fromkeys(
            str(item.get("message_id") or "")
            for item in source_messages
            if str(item.get("message_id") or "")
        ))
        typed = review_item.get("typed_candidate") if isinstance(review_item.get("typed_candidate"), dict) else {}
        original_payload = typed.get("payload") if isinstance(typed.get("payload"), dict) else {}
        original_episode = original_payload.get("episode") if isinstance(original_payload.get("episode"), dict) else {}
        # The original episode may contain W7 alignment context and attachment
        # paths whose filenames mention "SOP".  They are useful background for
        # the initial W2 proposal but are not the source of an expert
        # correction.  Carry only the current message evidence into the fresh
        # manual-review envelope so W4 does not misclassify a field case as an
        # SOP document or let alignment data become candidate provenance.
        correction_episode = {
            key: original_episode.get(key)
            for key in (
                "episode_id", "thread_id", "completeness",
                "fault_description_messages", "diagnostic_chain_messages",
                "resolution_messages", "case_evidence_messages",
                "case_context_messages", "noise_messages",
                "evidence_message_ids", "source_offsets",
            )
            if key in original_episode
        }
        parent_review_id = str(review_item.get("review_id") or correction.get("review_id") or "")
        source_episode_id = str(
            correction.get("source_episode_id_original")
            or original_episode.get("episode_id")
            or ""
        )
        source_text = str(candidate.get("source_text") or correction.get("variant") or "").strip()
        envelope = build_write_intake_envelope(
            source_type="manual_review",
            text=source_text,
            source_ref={
                "review_id": parent_review_id,
                "source_episode_id": source_episode_id,
                "message_ids": message_ids,
            },
            knowledge_kind="fault_case",
            payload={
                "text": source_text,
                "objects": candidate.get("objects") or {},
                "relations": candidate.get("relations") or [],
                "schema_valid": bool(candidate.get("schema_valid")),
                "schema_issues": list(candidate.get("schema_issues") or []),
                "source_messages": source_messages,
                "episode": correction_episode,
                "expert_correction": candidate.get("expert_correction") or {},
                "supersedes_review_id": candidate.get("supersedes_review_id") or parent_review_id,
                "provenance_rebound": bool(candidate.get("provenance_rebound")),
            },
            evidence_pack={
                "message_ids": message_ids,
                "source_messages": source_messages,
                "expert_correction": candidate.get("expert_correction") or {},
            },
            lineage={
                "agent_id": "W6-EXPERT-REVIEW",
                "parent_review_id": parent_review_id,
                "source_episode_id": source_episode_id,
                "provenance_rebound": bool(candidate.get("provenance_rebound")),
            },
            dedupe_key=str(candidate.get("dedupe_key") or ""),
        )
        envelope.update({
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "schema_valid": bool(candidate.get("schema_valid")),
            "schema_issues": list(candidate.get("schema_issues") or []),
            "admission_target": "fault_execution",
            "operation": "merge_graph",
            "review_id": f"review:typed:{candidate.get('dedupe_key') or candidate.get('candidate_id') or 'expert-correction'}",
        })
        queued = self._queue_typed_envelope(envelope, dry_run_merge=dry_run_merge)
        return {
            "run_manifest": {
                **manifest,
                "source_type": "manual_review",
                "parent_review_id": parent_review_id,
                "provenance_rebound": bool(candidate.get("provenance_rebound")),
            },
            "candidate": candidate,
            **queued,
        }

    def run_diagnostic_feedback(
        self,
        transcript: dict[str, Any],
        *,
        dry_run_merge: bool = True,
    ) -> dict[str, Any]:
        """Queue one read-side transcript as reviewable current evidence."""

        if not isinstance(transcript, dict):
            raise ValueError("diagnostic transcript must be a JSON object")
        proposal = DiagnosticFeedbackAgent().build_candidate(transcript)
        session_id = str(transcript.get("session_id") or transcript.get("case_id") or "diagnostic-session")
        return self._queue_loop_evidence(
            source_type="diagnostic_feedback",
            source_id=session_id,
            proposal=proposal,
            agent_id="D1",
            text=self._loop_evidence_text(transcript, fallback=session_id),
            dry_run_merge=dry_run_merge,
        )

    def run_log_pattern(
        self,
        log_summary: dict[str, Any],
        *,
        dry_run_merge: bool = True,
    ) -> dict[str, Any]:
        """Queue an unmatched log pattern as evidence, never as a KG pattern."""

        if not isinstance(log_summary, dict):
            raise ValueError("log summary must be a JSON object")
        proposal = LogPatternAgent().propose(log_summary)
        signature = str(
            log_summary.get("signature_id")
            or log_summary.get("pattern")
            or log_summary.get("signature")
            or "unmatched-log"
        )
        return self._queue_loop_evidence(
            source_type="log_pattern",
            source_id=signature,
            proposal=proposal,
            agent_id="D3",
            text=self._loop_evidence_text(log_summary, fallback=signature),
            dry_run_merge=dry_run_merge,
        )

    def run_atr_weight_proposal(self, feedback: dict[str, Any]) -> dict[str, Any]:
        """Queue D2 for human review without pretending weights are KG data.

        The current repository has no versioned ATR weight store or rollback
        applier.  D2 therefore receives a durable, idempotent review boundary
        of its own; approval records intent but cannot be consumed by W5.
        """

        if not isinstance(feedback, dict):
            raise ValueError("ATR feedback must be a JSON object")
        if self.w6_v2 is None:
            raise NonSopIntakeError("kg_v2_pipeline_unavailable", "ATR proposal review requires a KG v2 review store.")
        manifest = self._preflight_non_sop_kg_v2("v2")
        proposal = ATRWeightingAgent().propose(feedback)
        source_id = str(
            feedback.get("session_id")
            or feedback.get("feedback_id")
            or feedback.get("case_id")
            or "atr-feedback"
        )
        identity_basis = {
            "source_id": source_id,
            "top_error_id": feedback.get("top_error_id") or "",
            "solved_check_id": feedback.get("which_check_solved") or feedback.get("solved_check_id") or "",
        }
        dedupe_digest = hashlib.sha256(
            json.dumps(identity_basis, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        content_hash = "content:" + hashlib.sha256(
            json.dumps(proposal, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        proposal_id = f"atr-weight-proposal:{dedupe_digest}"
        item = {
            "review_id": f"review:{proposal_id}",
            "proposal_id": proposal_id,
            "candidate_id": proposal_id,
            "dedupe_key": proposal_id,
            "content_hash": content_hash,
            "queue": "atr_weight_proposals",
            "proposal": proposal,
            "review_actions": ["accept", "reject", "request_more_info"],
            "review_status": "pending",
            "application_boundary": {
                "status": "not_implemented",
                "operation": "none",
                "reason": "versioned_atr_weight_store_and_rollback_applier_not_defined",
                "w5_eligible": False,
            },
            "observability": {
                "agent_id": "D2/W6",
                "source_id": source_id,
                "proposal_id": proposal_id,
            },
        }
        queue_write = self.w6_v2.enqueue("atr_weight_proposals", item)
        return {
            "run_manifest": {**manifest, "source_type": "atr_feedback", "source_id": source_id},
            "proposal": proposal,
            "review_item": item,
            "queue_write": queue_write,
        }

    def _queue_loop_evidence(
        self,
        *,
        source_type: str,
        source_id: str,
        proposal: dict[str, Any],
        agent_id: str,
        text: str,
        dry_run_merge: bool,
    ) -> dict[str, Any]:
        manifest = self._preflight_non_sop_kg_v2("v2")
        canonical = json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{source_type}|{source_id}|{canonical}".encode("utf-8")).hexdigest()[:20]
        case_id = f"case:{source_type}:{digest}"
        evidence_id = f"evidence:{source_type}:{digest}"
        objects = {
            "SourceCase": [{
                "case_id": case_id,
                # KG v2 keeps loop proposals in the existing manual-review
                # provenance class; the precise D1/D3 origin remains on the
                # intake source_type and lineage instead of expanding the
                # curated object schema from the non-SOP path.
                "source_kind": "manual_review",
                "title": str(source_id)[:80],
                "summary": text[:240],
                "source_ref": str(source_id)[:200],
                "approved": False,
            }],
            "EvidenceItem": [{
                "evidence_id": evidence_id,
                "source_kind": "manual_review",
                "external_id": str(source_id)[:120],
                "title": str(source_id)[:80],
                "summary": text[:500],
                "payload_ref": str(source_id)[:200],
            }],
        }
        relations = [{"from": evidence_id, "to": case_id, "relation": "evidences"}]
        issues = validate_graph(objects, relations)
        envelope = build_write_intake_envelope(
            source_type=source_type,
            text=text,
            source_ref={"source_id": source_id},
            knowledge_kind="evidence_only",
            payload={
                "text": text,
                "objects": objects,
                "relations": relations,
                "schema_valid": not issues,
                "schema_issues": issues,
                "evidence_disposition": "evidence_only",
                "loop_proposal": proposal,
            },
            evidence_pack={"proposal": proposal},
            lineage={"agent_id": agent_id, "source_id": source_id},
        )
        envelope.update({
            "candidate_id": f"{source_type}:{digest}",
            "schema_valid": not issues,
            "schema_issues": issues,
            "admission_target": "evidence_only",
            "evidence_disposition": "evidence_only",
            "operation": "merge_graph",
        })
        queued = self._queue_typed_envelope(envelope, dry_run_merge=dry_run_merge)
        return {
            "run_manifest": {**manifest, "source_type": source_type, "source_id": source_id},
            "proposal": proposal,
            **queued,
        }

    @staticmethod
    def _loop_evidence_text(payload: dict[str, Any], *, fallback: str) -> str:
        preferred = [
            payload.get("query"),
            payload.get("final_status"),
            payload.get("top_error_id"),
            payload.get("pattern"),
            payload.get("signature"),
            payload.get("summary"),
        ]
        compact = "；".join(str(item).strip() for item in preferred if str(item or "").strip())
        if compact:
            return compact
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return serialized if serialized not in {"{}", ""} else fallback

    def _queue_typed_envelope(self, envelope: dict[str, Any], *, dry_run_merge: bool) -> dict[str, Any]:
        prepared = self._prepare_typed_envelope(envelope, dry_run_merge=dry_run_merge)
        prepared["queue_write"] = self.w6_v2.enqueue("v2_typed_candidates", prepared["review_item"])
        return prepared

    def _prepare_typed_envelope(self, envelope: dict[str, Any], *, dry_run_merge: bool) -> dict[str, Any]:
        """Prepare one typed review item without writing the queue.

        Large Jira/attachment archives use this boundary so W6 can upsert the
        complete batch with one queue read and one queue write.
        """
        if self.w4 is None or self.w5_v2 is None or self.w6_v2 is None:
            raise NonSopIntakeError("kg_v2_pipeline_unavailable", "Typed non-SOP ingestion requires W4, W5 v2 and W6 v2.")
        gate = self._score_typed_envelope(envelope)
        envelope["quality_gate"] = gate
        envelope["admission_target"] = gate.get("admission_target") or envelope.get("admission_target") or "evidence_only"
        envelope["materialize_allowed"] = bool(gate.get("materialize_allowed"))
        envelope["mapping_version"] = gate.get("mapping_version") or ""
        dry_plan = self.w5_v2.dry_run_merge_plan(envelope) if dry_run_merge else {}
        item = self.w6_v2.build_typed_review_item(envelope, gate, dry_run_plan=dry_plan)
        return {
            "intake_id": envelope.get("intake_id") or "",
            "dedupe_key": envelope.get("dedupe_key") or "",
            "quality_gate": gate,
            "dry_run_merge_plan": dry_plan,
            "review_item": item,
        }

    def _score_typed_envelope(self, envelope: dict[str, Any]) -> dict[str, Any]:
        gate = self.w4.score_typed_candidate(envelope)
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        objects = payload.get("objects") if isinstance(payload.get("objects"), dict) else {}
        has_fault_semantics = any(objects.get(key) for key in (
            "FaultFamily", "FaultVariant", "DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace"
        ))
        if not has_fault_semantics:
            return gate
        semantic_bundle = {
            "candidate_id": envelope.get("candidate_id") or envelope.get("intake_id") or "",
            "objects": objects,
            "relations": payload.get("relations") if isinstance(payload.get("relations"), list) else [],
            "schema_valid": bool(payload.get("schema_valid", envelope.get("schema_valid", True))),
            "schema_issues": list(payload.get("schema_issues") or envelope.get("schema_issues") or []),
            "strategy": payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {},
            "source_text": str(payload.get("text") or envelope.get("text") or ""),
            "source_message_ids": [
                str(value)
                for value in (
                    (envelope.get("source_ref") or {}).get("message_ids")
                    if isinstance(envelope.get("source_ref"), dict)
                    else []
                ) or []
                if str(value)
            ],
            "source_messages": list(payload.get("source_messages") or []),
        }
        semantic_gate = self.w4.score_v2_bundle(semantic_bundle)
        gate = {**gate, "kg_v2_semantic_gate": semantic_gate}
        if not semantic_gate.get("passed"):
            gate.update({
                "decision": "route_review" if gate.get("decision") == "admit" else gate.get("decision"),
                "materialize_allowed": False,
                "issues": sorted(set([*gate.get("issues", []), *semantic_gate.get("issues", []), "kg_v2_semantic_gate_failed"])),
                "passed": gate.get("decision") != "reject",
            })
        return gate

    @staticmethod
    def _document_route(output_mode: str) -> tuple[str, str]:
        routes = {
            # Raw manuals describe recommended diagnosis, not observed case
            # outcomes.  Their variants/actions enter support review and must
            # not be treated as executable historical evidence.
            "variant_case_bundle": ("support", "fault_support"),
            "family_support_bundle": ("support", "fault_support"),
            "atomic_case_bundle": ("support", "fault_support"),
            "playbook_bundle": ("playbook", "playbook"),
            "procedure_library_only": ("procedure", "procedure_library"),
            "reference_constraint_only": ("reference", "reference_constraint"),
            "policy_template_only": ("policy", "policy_template"),
            "faq_support_bundle": ("support", "fault_support"),
            "overlay_only": ("overlay", "overlay"),
            "review_only": ("evidence_only", "evidence_only"),
        }
        return routes.get(output_mode, ("evidence_only", "evidence_only"))

    @staticmethod
    def _document_text(payload: dict[str, Any]) -> str:
        parts: list[str] = [str(payload.get("name") or "")]
        for row in payload.get("section_cases") or []:
            if not isinstance(row, dict):
                continue
            parts.extend(str(row.get(key) or "") for key in ("section_title", "variant_candidate", "summary"))
            parts.extend(str(item) for item in row.get("actions") or [])
            parts.extend(str(item) for item in row.get("cause_notes") or [])
        text = "\n".join(part.strip() for part in parts if part and part.strip())
        return text or str(payload.get("path") or "non-SOP document")

    @staticmethod
    def _mark_bundle_as_non_sop_raw_doc(bundle: dict[str, Any], path: Path) -> dict[str, Any]:
        return WriteSidePipeline._mark_bundle_as_document_source(
            bundle,
            path,
            source_kind="raw_doc",
        )

    @staticmethod
    def _mark_bundle_as_document_source(
        bundle: dict[str, Any],
        path: Path,
        *,
        source_kind: str,
    ) -> dict[str, Any]:
        out = json.loads(json.dumps(bundle, ensure_ascii=False))
        objects = out.get("objects") if isinstance(out.get("objects"), dict) else {}
        for object_type in (
            "KnowledgeDocument",
            "ProcedureStep",
            "FaultFamily",
            "DiagnosticAction",
        ):
            for item in objects.get(object_type) or []:
                if isinstance(item, dict):
                    item["source_kind"] = source_kind
                    item["source_ref"] = str(path)
        for item in objects.get("SourceCase") or []:
            if isinstance(item, dict):
                item["source_kind"] = source_kind
                item["source_ref"] = str(path)[:200]
                item["approved"] = False
        for item in objects.get("EvidenceItem") or []:
            if isinstance(item, dict):
                item["source_kind"] = (
                    "sop" if source_kind == "sop" else "tool_parse"
                )
                item["payload_ref"] = str(path)[:200]
        issues = validate_graph(objects, out.get("relations") or [])
        out["schema_valid"] = not issues
        out["schema_issues"] = issues
        out["source_type"] = (
            "sop_doc" if source_kind == "sop" else "raw_doc"
        )
        out["source_ref"] = str(path)
        return out

    @staticmethod
    def _document_evidence_bundle(payload: dict[str, Any], *, knowledge_kind: str, admission_target: str) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        path = str(payload.get("path") or "")
        document_name = str(payload.get("name") or Path(path).name or "non-SOP document")
        case_digest = hashlib.sha256(f"{path}|{knowledge_kind}|source-case".encode("utf-8")).hexdigest()[:20]
        case_id = f"case:raw-doc:{case_digest}"
        relations: list[dict[str, Any]] = []
        case_summary_parts: list[str] = []
        for index, row in enumerate(payload.get("section_cases") or [], start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("section_title") or row.get("variant_candidate") or f"section {index}")[:80]
            summary_parts = [
                str(row.get("variant_candidate") or ""),
                *[str(item) for item in row.get("actions") or []],
                *[str(item) for item in row.get("cause_notes") or []],
            ]
            summary = "；".join(part.strip() for part in summary_parts if part.strip())[:500] or title
            case_summary_parts.append(f"{title}：{summary}")
            digest = hashlib.sha256(f"{path}|{index}|{title}|{summary}".encode("utf-8")).hexdigest()[:20]
            evidence_id = f"evidence:raw-doc:{digest}"
            items.append({
                "evidence_id": evidence_id,
                "source_kind": "tool_parse",
                "external_id": str(row.get("case_id") or f"section:{index}")[:120],
                "title": title,
                "summary": summary,
                "payload_ref": path[:200],
                "knowledge_kind": knowledge_kind,
                "_admission_target": admission_target,
                "execution_materialize_allowed": False,
            })
            relations.append({"from": evidence_id, "to": case_id, "relation": "evidences"})
        source_cases = [{
            "case_id": case_id,
            "source_kind": "raw_doc",
            "title": document_name[:80],
            "summary": "；".join(case_summary_parts)[:240] or document_name[:240],
            "source_ref": path[:200],
            "approved": False,
        }]
        objects = {
            "FaultFamily": [], "FaultVariant": [], "DiagnosticAction": [], "ActionOutcome": [],
            "RequiredInfoSpec": [], "DiagnosticTrace": [], "DecisionPolicy": [],
            "EvidenceItem": items, "SourceCase": source_cases,
        }
        issues = validate_graph(objects, relations)
        bundle_id = hashlib.sha256(f"{path}|{knowledge_kind}".encode("utf-8")).hexdigest()[:20]
        return {
            "bundle_id": f"bundle:raw-doc:{bundle_id}",
            "objects": objects,
            "relations": relations,
            "schema_valid": not issues,
            "schema_issues": issues,
        }

    def _evidence_context_envelope(self, context: dict[str, Any]) -> dict[str, Any]:
        context_id = str(context.get("context_id") or "evidence-context")
        source_context = context.get("source_context") if isinstance(context.get("source_context"), dict) else {}
        source_manifest = context.get("source_manifest") if isinstance(context.get("source_manifest"), dict) else {}
        anchor_messages = [str(item) for item in source_context.get("anchor_messages") or [] if str(item).strip()]
        files = [item for item in context.get("files") or [] if isinstance(item, dict)]
        text_parts = [*anchor_messages, *[str(item.get("name") or item.get("path") or "") for item in files]]
        text = "\n".join(part for part in text_parts if part) or context_id
        evidence_items: list[dict[str, Any]] = []
        for index, item in enumerate(files, start=1):
            tool = str(item.get("tool") or "attachment")
            parsed = item.get("parse_result") if isinstance(item.get("parse_result"), dict) else {}
            summary = self._evidence_summary(parsed, fallback=str(item.get("name") or "evidence"))
            digest = hashlib.sha256(f"{context_id}|{index}|{item.get('path') or item.get('name')}".encode("utf-8")).hexdigest()[:20]
            evidence_items.append({
                "evidence_id": f"evidence:context:{digest}",
                "source_kind": "jira" if tool == "jira" else "tool_parse",
                "external_id": str(item.get("name") or context_id)[:120],
                "title": str(item.get("name") or tool)[:80],
                "summary": summary,
                "payload_ref": str(item.get("path") or context.get("context_root") or "")[:200],
                "knowledge_kind": "evidence_only",
                "_admission_target": "evidence_only",
                "execution_materialize_allowed": False,
                "evidence_origin": str(item.get("evidence_origin") or ""),
                "binding_status": str(item.get("binding_status") or ""),
                "source_message_id": str(item.get("source_message_id") or ""),
                "source_create_time": str(item.get("source_create_time") or ""),
            })
        disposition, linked_case = self._evidence_context_disposition(
            context_id=context_id,
            source_context=source_context,
            source_manifest=source_manifest,
            files=files,
            evidence_items=evidence_items,
        )
        source_cases: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        if linked_case:
            source_cases.append(linked_case)
            relations.extend(
                {"from": item["evidence_id"], "to": linked_case["case_id"], "relation": "evidences"}
                for item in evidence_items
            )
        graph_evidence_items = evidence_items if linked_case else []
        objects = {
            "FaultFamily": [], "FaultVariant": [], "DiagnosticAction": [], "ActionOutcome": [],
            "RequiredInfoSpec": [], "DiagnosticTrace": [], "DecisionPolicy": [],
            "EvidenceItem": graph_evidence_items, "SourceCase": source_cases,
        }
        source_type = "jira" if any(str(item.get("tool") or "") == "jira" for item in files) else "attachment"
        issues = validate_graph(objects, relations)
        envelope = build_write_intake_envelope(
            source_type=source_type,
            text=text,
            source_ref={"context_id": context_id, "path": str(context.get("context_root") or "")},
            knowledge_kind="evidence_only",
            payload={
                "text": text,
                "objects": objects,
                "relations": relations,
                "schema_valid": not issues,
                "schema_issues": issues,
                "evidence_disposition": disposition,
            },
            evidence_pack={
                "tool_evidence": context.get("tool_evidence") or {},
                "files": files,
                "source_context": source_context,
                "evidence_items": evidence_items,
            },
            lineage={"agent_id": "TOOL-EVIDENCE-CONTEXT", "context_id": context_id, "evidence_disposition": disposition},
        )
        envelope.update({
            "candidate_id": f"evidence-context:{context_id}",
            "schema_valid": not issues,
            "schema_issues": issues,
            "admission_target": "evidence_only",
            "evidence_disposition": disposition,
            "operation": "merge_graph",
        })
        return envelope

    def _evidence_context_disposition(
        self,
        *,
        context_id: str,
        source_context: dict[str, Any],
        source_manifest: dict[str, Any],
        files: list[dict[str, Any]],
        evidence_items: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | None]:
        if not evidence_items or not any(self._usable_parse_result(item.get("parse_result")) for item in files):
            return "reject_review_only", None

        linked_case_id = str(
            source_manifest.get("source_case_id")
            or source_manifest.get("linked_source_case_id")
            or source_context.get("source_case_id")
            or ""
        )
        if linked_case_id and self.kg_v2_store is not None:
            existing = self.kg_v2_store.object_index("SourceCase").get(linked_case_id)
            if existing:
                return "merge_evidence", dict(existing)

        jira_details = [
            detail
            for item in files
            if str(item.get("tool") or "") == "jira"
            for detail in ((item.get("parse_result") or {}).get("offline_details") or [])
            if isinstance(detail, dict)
        ]
        anchors = [str(item) for item in source_context.get("anchor_messages") or [] if str(item).strip()]
        complete_case = bool(jira_details or (anchors and files))
        if not complete_case:
            return "evidence_only", None

        if jira_details:
            detail = jira_details[0]
            issue_key = str(detail.get("issue_key") or context_id)
            title = str(detail.get("summary") or issue_key)[:80]
            summary = "；".join(
                part.strip()
                for part in (
                    str(detail.get("summary") or ""),
                    str(detail.get("description_preview") or ""),
                    str(detail.get("comment_preview_text") or ""),
                )
                if part.strip()
            )[:240] or title
            case_id = f"case:jira:{issue_key}"
            source_kind = "jira_case"
        else:
            digest = hashlib.sha256(f"{context_id}|{'|'.join(anchors)}".encode("utf-8")).hexdigest()[:20]
            title = (anchors[0] if anchors else context_id)[:80]
            summary = "；".join(anchors)[:240] or title
            case_id = f"case:attachment:{digest}"
            source_kind = "chat_case"
        return "new_source_case", {
            "case_id": case_id,
            "source_kind": source_kind,
            "title": title,
            "summary": summary,
            "source_ref": str(source_manifest.get("segment_id") or source_context.get("segment_id") or context_id)[:200],
            "approved": False,
        }

    @staticmethod
    def _usable_parse_result(value: Any) -> bool:
        if not isinstance(value, dict) or not value:
            return False
        if value.get("exists") is False:
            return False
        status = str(value.get("status") or "").lower()
        return status not in {"missing", "error", "parse_error", "unsupported", "not_found"}

    @staticmethod
    def _evidence_summary(parsed: dict[str, Any], *, fallback: str) -> str:
        details = [item for item in parsed.get("offline_details") or [] if isinstance(item, dict)]
        if details:
            detail = details[0]
            text = "；".join(
                part.strip()
                for part in (
                    str(detail.get("summary") or ""),
                    str(detail.get("description_preview") or ""),
                    str(detail.get("comment_preview_text") or ""),
                )
                if part.strip()
            )
            if text:
                return text[:500]
        for key in ("summary_hint", "text_preview", "dump_kind", "status"):
            if str(parsed.get(key) or "").strip():
                return str(parsed.get(key))[:500]
        return (json.dumps(parsed, ensure_ascii=False, sort_keys=True)[:500] or fallback)[:500]

    def run_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        apply_approved: bool = False,
        emit_episodes: bool = False,
        dry_run_merge: bool = True,
        kg_mode: str = "legacy",
    ) -> dict[str, Any]:
        episodes = [self._episode_from_candidate(candidate) for candidate in candidates]
        return self._run_candidate_episode_pairs(
            episodes,
            candidates,
            apply_approved=apply_approved,
            emit_episodes=emit_episodes,
            dry_run_merge=dry_run_merge,
            kg_mode=kg_mode,
            summary_counts={"summaries": 0, "episodes": len(episodes)},
        )

    def _run_w7_shadow_batches(
        self,
        episodes: list[dict[str, Any]],
        *,
        source_type: str,
        out_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run the split W7 chain as an isolated, content-addressed shadow.

        This is intentionally called alongside the legacy extraction path.
        The legacy candidate remains authoritative, while W7a→atomic W2→W7b
        results are written only under ``w7_shadow`` for calibration and human
        comparison.  Keeping this boundary in the main pipeline (rather than
        only in an evaluation script) ensures production callers exercise the
        same source preparation and W2 adapter as the eventual replacement.
        """

        if not self.w7_shadow_enabled:
            return {"status": "disabled", "mode": self.w7_mode}
        if self.w7_decision_client is None:
            return {
                "status": "not_run",
                "mode": self.w7_mode,
                "reason": "w7_decision_client_not_configured",
                "promotion_allowed": False,
                "legacy_authoritative": True,
            }
        grouped: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for index, episode in enumerate(episodes, 1):
            if self.w7_batch_scope == "all":
                key = "all"
            elif self.w7_batch_scope == "chat":
                raw = str(
                    episode.get("chat_id")
                    or episode.get("source_thread_id")
                    or episode.get("thread_id")
                    or episode.get("episode_id")
                    or f"episode-{index}"
                )
                key = raw.split(":", 1)[0].split("_20", 1)[0]
            else:
                key = str(
                    episode.get("source_thread_id")
                    or episode.get("thread_id")
                    or episode.get("episode_id")
                    or f"episode-{index}"
                )
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(episode)

        root = (
            Path(out_dir) / "w7_shadow"
            if out_dir is not None
            else self.w7_shadow_out_dir
        )
        if root is None:
            # A caller that did not request an output directory still gets a
            # returned manifest, with checkpointing disabled as well.
            persist = False
        else:
            persist = True
            root.mkdir(parents=True, exist_ok=True)
        batch_results: list[dict[str, Any]] = []
        for batch_id in order:
            batch_episodes = grouped[batch_id]
            run_key = canonical_hash({
                "source_type": source_type,
                "batch_id": batch_id,
                "episode_ids": [
                    str(item.get("episode_id") or "")
                    for item in batch_episodes
                ],
            })[:12]
            orchestrator = W7ShadowOrchestrator(
                client=self.w7_decision_client,
                checkpoint_root=(
                    root / "checkpoints" / run_key if root is not None else None
                ),
                component_workers=self.w7_decision_workers,
            )

            def extract_atomic(manifest: dict[str, Any]) -> dict[str, Any]:
                return self.extract_w7_atomic_cases(
                    manifest,
                    w2_mode="native_v2",
                    source_type=source_type,
                    workers=self.w7_atomic_workers,
                )

            result = W7BatchShadowOrchestrator(
                orchestrator,
                decision_workers=self.w7_decision_workers,
                atomic_workers=self.w7_atomic_workers,
            ).run(
                batch_id=batch_id,
                episodes=batch_episodes,
                atomic_extractor=extract_atomic,
            )
            result["shadow_run_key"] = run_key
            result["source_type"] = source_type
            batch_path = (
                root / "batches" / f"{run_key}.json"
                if root is not None
                else Path()
            )
            if persist:
                batch_path.parent.mkdir(parents=True, exist_ok=True)
                batch_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            batch_results.append({
                "batch_id": batch_id,
                "run_key": run_key,
                "result": str(batch_path) if persist else "",
                "schema_valid": bool(result.get("schema_valid")),
                "promotion_allowed": bool(
                    result.get("promotion_allowed")
                ),
                "legacy_authoritative": bool(
                    result.get("legacy_authoritative")
                ),
                "result_hash": str(result.get("result_hash") or ""),
            })
        manifest: dict[str, Any] = {
            "schema_version": "w7.pipeline_shadow_manifest.v1",
            "mode": self.w7_mode,
            "source_type": source_type,
            "batch_scope": self.w7_batch_scope,
            "status": "completed",
            "promotion_allowed": False,
            "legacy_authoritative": True,
            "batch_count": len(batch_results),
            "episode_count": len(episodes),
            "batches": batch_results,
        }
        if persist:
            manifest_path = root / "manifest.json"
            manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_hash"] = canonical_hash(manifest)
        if persist:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return manifest

    def run_summaries(
        self,
        summaries: list[dict[str, Any]],
        *,
        apply_approved: bool = False,
        emit_episodes: bool = False,
        dry_run_merge: bool = True,
        w2_workers: int = 1,
        kg_mode: str = "legacy",
        w2_mode: str | None = None,
        source_type: str = "chat",
        progress_path: Path | None = None,
        partial_candidates_path: Path | None = None,
    ) -> dict[str, Any]:
        # Keep an untouched W1 projection for the new W7 shadow.  The legacy
        # path may refine/split episodes using ReviewContextAgent; feeding that
        # post-processed result into the shadow would leak legacy decisions
        # into the comparison and make the calibration optimistic.
        raw_episodes: list[dict[str, Any]] = []
        if self.w7_shadow_enabled and self.w7_decision_client is not None:
            raw_episodes = self._episodes_from_summaries(
                summaries,
                refine_trace=False,
            )
            for episode in raw_episodes:
                episode["_write_source_type"] = source_type
        episodes = self._episodes_from_summaries(
            summaries,
            refine_trace=self.review_context_enabled,
        )
        for episode in episodes:
            episode["_write_source_type"] = source_type
        total = len(episodes)

        def progress_callback(completed: int, total_count: int) -> None:
            self._write_progress(
                progress_path,
                stage="w2_extract",
                payload={
                    "episodes_total": total_count,
                    "episodes_completed": completed,
                    "w2_mode": w2_mode or getattr(self.w2, "w2_mode", "legacy_only"),
                },
            )

        candidates = self._extract_w2_candidates(
            episodes,
            workers=w2_workers,
            w2_mode=w2_mode,
            source_type=source_type,
            progress_callback=progress_callback if progress_path is not None else None,
            partial_candidates_path=partial_candidates_path,
            resumed_candidates=self._load_partial_candidates(partial_candidates_path) if partial_candidates_path is not None else None,
        )
        self._write_progress(
            progress_path,
            stage="w2_completed",
            payload={
                "episodes_total": total,
                "candidates": len(candidates),
            },
        )
        shadow_manifest = self._run_w7_shadow_batches(
            raw_episodes,
            source_type=source_type,
            out_dir=(progress_path.parent if progress_path is not None else None),
        )
        result = self._run_candidate_episode_pairs(
            episodes,
            candidates,
            apply_approved=apply_approved,
            emit_episodes=emit_episodes,
            dry_run_merge=dry_run_merge,
            kg_mode=kg_mode,
            summary_counts={"summaries": len(summaries), "episodes": len(episodes)},
            progress_path=progress_path,
        )
        result["w7_shadow"] = shadow_manifest
        self._last_w7_shadow_manifest = shadow_manifest
        return result

    def _run_candidate_episode_pairs(
        self,
        episodes: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        *,
        apply_approved: bool,
        emit_episodes: bool,
        dry_run_merge: bool,
        kg_mode: str,
        summary_counts: dict[str, int],
        progress_path: Path | None = None,
    ) -> dict[str, Any]:
        details: list[dict[str, Any]] = []
        queue_writes: list[dict[str, Any]] = []
        approved_apply_results: list[dict[str, Any]] = []
        pending_queue_items: dict[str, list[dict[str, Any]]] = {"candidates": [], "merge_candidates": [], "noise_candidates": [], "ask_info_candidates": []}
        dry_run_merge_plans: list[dict[str, Any]] = []
        dry_run_required_info_plans: list[dict[str, Any]] = []
        v2_queue_writes: list[dict[str, Any]] = []
        v2_dry_run_merge_plans: list[dict[str, Any]] = []
        v2_approved_apply_results: list[dict[str, Any]] = []
        pending_v2_queue_items: dict[str, list[dict[str, Any]]] = {"v2_typed_candidates": []}
        use_legacy = kg_mode in {"legacy", "both"}
        use_v2 = kg_mode in {"v2", "both"} and self.kg_v2_store is not None and self.w5_v2 is not None and self.w6_v2 is not None
        counts = {
            "summaries": int(summary_counts.get("summaries") or 0),
            "episodes": int(summary_counts.get("episodes") or len(episodes)),
            "candidates": 0,
            "required_info_candidates": 0,
            "gate_passed": 0,
            "ask_info_gate_passed": 0,
            "queued_candidates": 0,
            "queued_merges": 0,
            "queued_noise": 0,
            "queued_ask_info": 0,
            "dry_run": 0,
            "required_info_dry_run": 0,
            "applied": 0,
            "required_info_applied": 0,
            "v2_candidates": 0,
            "v2_gate_passed": 0,
            "v2_queued_candidates": 0,
            "v2_queued_merges": 0,
            "v2_queued_noise": 0,
            "v2_queued_typed": 0,
            "v2_dry_run": 0,
            "v2_applied": 0,
        }
        for episode, candidate in zip(episodes, candidates):
            candidate_episode = (
                candidate.get("episode")
                if isinstance(candidate.get("episode"), dict)
                and str((candidate.get("episode") or {}).get("episode_id") or "")
                == str(episode.get("episode_id") or "")
                else episode
            )
            conflict = self.w3.resolve(candidate)
            gate = self.w4.score(candidate)
            raw_v2_bundle = candidate.get("candidate_draft_v2_normalized_bundle") if isinstance(candidate.get("candidate_draft_v2_normalized_bundle"), dict) else {}
            v2_bundle = {}
            if raw_v2_bundle:
                hydrated = self._hydrate_v2_bundle_identity(candidate, candidate_episode, raw_v2_bundle)
                v2_bundle = self.w3.normalize_v2_bundle(hydrated)
                candidate["candidate_draft_v2_w3_bundle"] = v2_bundle
            v2_semantic_gate = {}
            if v2_bundle:
                v2_semantic_gate = self.w4.score_v2_bundle(v2_bundle)
                extraction_source = str((candidate.get("case_understanding_extraction") or {}).get("case_understanding_source") or "")
                prompt_card_authoritative = (
                    str(candidate.get("w2_mode") or "") in {"native_v2", "prompt_first"}
                    and extraction_source == "deepseek_prompt_a"
                )
                if prompt_card_authoritative:
                    production_valid = bool(candidate.get("production_schema_valid"))
                    gate = {
                        **v2_semantic_gate,
                        "passed": bool(v2_semantic_gate.get("passed")) and production_valid,
                        "issues": sorted(set([
                            *v2_semantic_gate.get("issues", []),
                            *([] if production_valid else ["w2_production_schema_invalid"]),
                        ])),
                        "kg_v2_semantic_gate": v2_semantic_gate,
                        "gate_source": "prompt_case_understanding_v2",
                    }
                else:
                    gate = {
                        **gate,
                        "confidence": min(float(gate.get("confidence") or 0.0), float(v2_semantic_gate.get("confidence") or 0.0)),
                        "clarity": min(float(gate.get("clarity") or 0.0), float(v2_semantic_gate.get("clarity") or 0.0)),
                        "relevance": min(float(gate.get("relevance") or 0.0), float(v2_semantic_gate.get("relevance") or 0.0)),
                        "schema_validity": min(float(gate.get("schema_validity") or 0.0), float(v2_semantic_gate.get("schema_validity") or 0.0)),
                        "weighted_sum": min(float(gate.get("weighted_sum") or 0.0), float(v2_semantic_gate.get("weighted_sum") or 0.0)),
                        "passed": bool(gate.get("passed")) and bool(v2_semantic_gate.get("passed")),
                        "issues": sorted(set([*gate.get("issues", []), *v2_semantic_gate.get("issues", [])])),
                        "kg_v2_semantic_gate": v2_semantic_gate,
                        "gate_source": "legacy_and_v2_minimum",
                    }
            counts["candidates"] += 1
            if gate["passed"]:
                counts["gate_passed"] += 1
            dry_plan = self.w5.dry_run_merge_plan(candidate) if dry_run_merge and use_legacy else {}
            if dry_plan:
                dry_run_merge_plans.append(dry_plan)
                counts["dry_run"] += 1
            logical_queue = self._queue_for(conflict, gate)
            ingest = {"status": "not_applied", "reason": "pending_review_item_not_auto_applied"}
            if use_legacy:
                review_item = self.w6.build_review_item(logical_queue, candidate, candidate_episode, conflict, gate, dry_plan)
                pending_queue_items.setdefault(logical_queue, []).append(review_item)
                if logical_queue == "merge_candidates":
                    counts["queued_merges"] += 1
                elif logical_queue == "noise_candidates":
                    counts["queued_noise"] += 1
                else:
                    counts["queued_candidates"] += 1
            v2_dry_plan = {}
            v2_queue = ""
            if use_v2:
                if not v2_bundle:
                    raw_v2_bundle = build_v2_bundle_from_legacy_candidate(candidate, candidate_episode)
                    hydrated = self._hydrate_v2_bundle_identity(candidate, candidate_episode, raw_v2_bundle)
                    v2_bundle = self.w3.normalize_v2_bundle(hydrated)
                    candidate["candidate_draft_v2_w3_bundle"] = v2_bundle
                counts["v2_candidates"] += 1
                if not v2_semantic_gate:
                    v2_semantic_gate = self.w4.score_v2_bundle(v2_bundle)
                v2_gate = gate if gate.get("kg_v2_semantic_gate") else {
                    **gate,
                    "confidence": min(float(gate.get("confidence") or 0.0), float(v2_semantic_gate.get("confidence") or 0.0)),
                    "clarity": min(float(gate.get("clarity") or 0.0), float(v2_semantic_gate.get("clarity") or 0.0)),
                    "relevance": min(float(gate.get("relevance") or 0.0), float(v2_semantic_gate.get("relevance") or 0.0)),
                    "schema_validity": min(float(gate.get("schema_validity") or 0.0), float(v2_semantic_gate.get("schema_validity") or 0.0)),
                    "weighted_sum": min(float(gate.get("weighted_sum") or 0.0), float(v2_semantic_gate.get("weighted_sum") or 0.0)),
                    "passed": bool(gate.get("passed")) and bool(v2_semantic_gate.get("passed")),
                    "issues": sorted(set([*gate.get("issues", []), *v2_semantic_gate.get("issues", [])])),
                    "kg_v2_semantic_gate": v2_semantic_gate,
                }
                source_type = str(candidate_episode.get("_write_source_type") or episode.get("_write_source_type") or "chat")
                # W7 may promote same-case nearby messages (or Jira-linked
                # evidence) into the prepared W2 episode.  The candidate's
                # episode is the authoritative evidence scope for the typed
                # envelope; using the pre-W7 episode here makes provenance
                # validation reject valid promoted evidence as "outside
                # intake" and silently turns otherwise reviewable cases into
                # hard rejects.
                typed_envelope = self._typed_envelope_for_v2_bundle(
                    candidate_episode,
                    v2_bundle,
                    source_type=source_type,
                )
                typed_gate = self.w4.score_typed_candidate(typed_envelope)
                w2_production_valid = bool(candidate.get("production_schema_valid", candidate.get("schema_valid")))
                if not bool(v2_semantic_gate.get("passed")) or not w2_production_valid:
                    upstream_issues = [*v2_semantic_gate.get("issues", [])]
                    if not w2_production_valid:
                        upstream_issues.append("w2_production_schema_invalid")
                    typed_gate = {
                        **typed_gate,
                        "decision": "route_review" if typed_gate.get("decision") == "admit" else typed_gate.get("decision"),
                        "materialize_allowed": False,
                        "merge_allowed": False,
                        "issues": sorted(set([*typed_gate.get("issues", []), *upstream_issues, "kg_v2_semantic_gate_failed"])),
                        "passed": typed_gate.get("decision") != "reject",
                    }
                typed_envelope["quality_gate"] = typed_gate
                typed_envelope["admission_target"] = typed_gate.get("admission_target") or "fault_execution"
                typed_envelope["admission_readiness"] = typed_gate.get("admission_readiness") or "not_ready"
                typed_envelope["merge_allowed"] = bool(typed_gate.get("merge_allowed"))
                typed_envelope["materialize_allowed"] = bool(typed_gate.get("materialize_allowed"))
                typed_envelope["mapping_version"] = typed_gate.get("mapping_version") or ""
                if typed_gate.get("decision") == "admit":
                    counts["v2_gate_passed"] += 1
                v2_queue = "v2_typed_candidates"
                v2_dry_plan = self.w5_v2.dry_run_merge_plan(typed_envelope) if dry_run_merge else {}
                if v2_dry_plan:
                    v2_dry_run_merge_plans.append(v2_dry_plan)
                    counts["v2_dry_run"] += 1
                v2_review_item = self.w6_v2.build_typed_review_item(typed_envelope, typed_gate, dry_run_plan=v2_dry_plan)
                pending_v2_queue_items.setdefault(v2_queue, []).append(v2_review_item)
                counts["v2_queued_typed"] += 1
                counts["v2_queued_candidates"] += 1
            details.append({
                "candidate_id": candidate.get("candidate_id"),
                "episode_id": episode.get("episode_id"),
                "thread_id": candidate.get("source_thread_id"),
                "queue": logical_queue if use_legacy else "",
                "v2_queue": v2_queue,
                "gate": gate,
                "conflict": conflict,
                "ingest": ingest,
                "dry_run_merge_plan": dry_plan,
                "v2_candidate_id": v2_bundle.get("candidate_id") if isinstance(v2_bundle, dict) else "",
                "v2_schema_valid": bool(v2_bundle.get("schema_valid")) if isinstance(v2_bundle, dict) else False,
                "v2_schema_issues": list(v2_bundle.get("schema_issues") or []) if isinstance(v2_bundle, dict) else [],
                "v2_typed_decision": typed_gate if use_v2 else {},
                "dry_run_merge_plan_v2": v2_dry_plan,
            })
            for required_info in candidate.get("required_info_candidates") or []:
                if not isinstance(required_info, dict):
                    continue
                counts["required_info_candidates"] += 1
                ask_conflict = self.w3.resolve_required_info(required_info)
                normalized_required_info = ask_conflict.get("candidate") if isinstance(ask_conflict.get("candidate"), dict) else required_info
                ask_gate = self.w4.score_required_info(normalized_required_info)
                if ask_gate["passed"]:
                    counts["ask_info_gate_passed"] += 1
                ask_plan = self.w5.dry_run_required_info_merge(normalized_required_info) if dry_run_merge else {}
                if ask_plan:
                    dry_run_required_info_plans.append(ask_plan)
                    counts["required_info_dry_run"] += 1
                ask_item = self.w6.build_ask_info_review_item(normalized_required_info, episode, ask_gate, ask_conflict)
                if ask_plan:
                    ask_item["dry_run_required_info_merge_plan"] = ask_plan
                ask_ingest = {"status": "not_applied", "reason": "pending_review_item_not_auto_applied"}
                pending_queue_items["ask_info_candidates"].append(ask_item)
                counts["queued_ask_info"] += 1
                details.append({
                    "candidate_id": normalized_required_info.get("candidate_id"),
                    "episode_id": episode.get("episode_id"),
                    "thread_id": normalized_required_info.get("source_thread_id"),
                    "queue": "ask_info_candidates" if use_legacy else "",
                    "gate": ask_gate,
                    "conflict": ask_conflict,
                    "ingest": ask_ingest,
                    "dry_run_required_info_merge_plan": ask_plan,
                })
        self._write_progress(
            progress_path,
            stage="queues_built",
            payload={
                "episodes": counts["episodes"],
                "candidates": counts["candidates"],
                "gate_passed": counts["gate_passed"],
                "queued_merges": counts["queued_merges"],
                "queued_noise": counts["queued_noise"],
                "queued_ask_info": counts["queued_ask_info"],
            },
        )
        if use_legacy:
            for queue_name, items in pending_queue_items.items():
                if items:
                    queue_writes.append(self.w6.enqueue_many(queue_name, items))
        if use_v2:
            for queue_name, items in pending_v2_queue_items.items():
                if items:
                    v2_queue_writes.append(self.w6_v2.enqueue_many(queue_name, items))
        if apply_approved:
            if use_legacy:
                approved_apply_results = self.apply_approved_review_queue(kg_mode="legacy")
                counts["applied"] = sum(1 for item in approved_apply_results if str(item.get("status") or "") not in {"skipped", "not_applied"} and not str(item.get("status") or "").startswith("required_info_"))
                counts["required_info_applied"] = sum(1 for item in approved_apply_results if str(item.get("status") or "").startswith("required_info_"))
            if use_v2:
                v2_approved_apply_results = self.apply_approved_review_queue(kg_mode="v2")
                counts["v2_applied"] = sum(1 for item in v2_approved_apply_results if str(item.get("status") or "") not in {"skipped", "not_applied"})
        review_summary = {
            "candidates": counts["queued_candidates"],
            "merge_candidates": counts["queued_merges"],
            "noise_candidates": counts["queued_noise"],
            "ask_info_candidates": counts["queued_ask_info"],
            "gate_passed": counts["gate_passed"],
            "gate_failed": counts["candidates"] - counts["gate_passed"],
            "ask_info_gate_passed": counts["ask_info_gate_passed"],
            "ask_info_gate_failed": counts["required_info_candidates"] - counts["ask_info_gate_passed"],
            "dry_run_merge_plans": len(dry_run_merge_plans),
            "dry_run_required_info_merge_plans": len(dry_run_required_info_plans),
        }
        v2_review_summary = {
            "v2_typed_candidates": counts["v2_queued_typed"],
            "candidates": counts["v2_queued_candidates"],
            "merge_candidates": counts["v2_queued_merges"],
            "noise_candidates": counts["v2_queued_noise"],
            "gate_passed": counts["v2_gate_passed"],
            "gate_failed": counts["v2_candidates"] - counts["v2_gate_passed"],
            "dry_run_merge_plans": len(v2_dry_run_merge_plans),
        }
        return {
            "summary": counts,
            "kg_mode": kg_mode,
            "review_summary": review_summary,
            "v2_review_summary": v2_review_summary,
            "episodes": episodes if emit_episodes else [],
            "queue_writes": queue_writes,
            "v2_queue_writes": v2_queue_writes,
            "dry_run_merge_plans": dry_run_merge_plans,
            "v2_dry_run_merge_plans": v2_dry_run_merge_plans,
            "dry_run_required_info_merge_plans": dry_run_required_info_plans,
            "approved_apply_results": approved_apply_results,
            "v2_approved_apply_results": v2_approved_apply_results,
            "details": details,
        }

    def _typed_envelope_for_v2_bundle(
        self,
        episode: dict[str, Any],
        bundle: dict[str, Any],
        *,
        source_type: str,
    ) -> dict[str, Any]:
        base = self._episode_intake_envelope(episode, source_type=source_type)
        payload = {
            "text": base["text"],
            "candidate_id": str(bundle.get("candidate_id") or bundle.get("bundle_id") or ""),
            "objects": bundle.get("objects") if isinstance(bundle.get("objects"), dict) else {},
            "relations": bundle.get("relations") if isinstance(bundle.get("relations"), list) else [],
            "schema_valid": bool(bundle.get("schema_valid")),
            "schema_issues": list(bundle.get("schema_issues") or []),
            "context_evidence_policy": str(
                ((bundle.get("extraction_metadata") or {}).get("context_evidence_policy"))
                if isinstance(bundle.get("extraction_metadata"), dict)
                else ""
            ) or ("current_episode_only.v1" if source_type in {"chat", "text_history"} else ""),
            "episode": episode,
            "source_messages": list(bundle.get("source_messages") or []),
        }
        envelope = build_write_intake_envelope(
            source_type=source_type,
            source_ref=base["source_ref"],
            knowledge_kind="fault_case",
            payload=payload,
            evidence_pack=base["evidence_pack"],
            lineage=base["lineage"],
            metadata={
                "review_context": ((episode.get("extracted") or {}).get("review_context") if isinstance(episode.get("extracted"), dict) else {}),
                "alignment_context": ((episode.get("extracted") or {}).get("review_context") if isinstance(episode.get("extracted"), dict) else {}),
                "source_mode": "non_sop_incremental",
            },
        )
        envelope.update({
            "candidate_id": payload["candidate_id"],
            "schema_valid": payload["schema_valid"],
            "schema_issues": payload["schema_issues"],
            "operation": "merge_graph",
        })
        return envelope

    @staticmethod
    def _write_progress(path: Path | None, *, stage: str, payload: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "stage": stage,
                    **payload,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def apply_approved_review_queue(self, *, kg_mode: str = "legacy") -> list[dict[str, Any]]:
        """Apply only review items that were approved before this run.

        This is the production write boundary: freshly generated pending review
        items are never auto-applied. Human approval is represented by queue
        items whose `review_status/status/selected_action` marks approval.
        """

        if kg_mode == "v2":
            if self.w5_v2 is None or self.w6_v2 is None:
                return []
            return self.w5_v2.apply_approved_review_queue(self.w6_v2)
        if kg_mode == "both":
            return [
                *self.apply_approved_review_queue(kg_mode="legacy"),
                *self.apply_approved_review_queue(kg_mode="v2"),
            ]
        results: list[dict[str, Any]] = []
        for queue_name in ("candidates.json", "merge_candidates.json"):
            for item in self.w6.read_queue(queue_name):
                if not isinstance(item, dict):
                    continue
                if not self._is_approved_review_item(item):
                    continue
                results.append(self.w5.apply_approved(item))
        for item in self.w6.read_queue("ask_info_candidates.json"):
            if not isinstance(item, dict):
                continue
            if not self._is_approved_review_item(item):
                continue
            results.append(self.w5.apply_approved_required_info(item))
        return results

    @staticmethod
    def _is_approved_review_item(item: dict[str, Any]) -> bool:
        return bool(item.get("human_approved")) or str(item.get("review_status") or item.get("status") or "") in {"approved", "human_approved", "accepted"} or str(item.get("selected_action") or "") in {"approve", "accept", "merge"}

    def _extract_w2_candidates(
        self,
        episodes: list[dict[str, Any]],
        *,
        workers: int = 1,
        w2_mode: str | None = None,
        source_type: str = "chat",
        progress_callback: Any | None = None,
        partial_candidates_path: Path | None = None,
        resumed_candidates: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        workers = max(1, int(workers or 1))
        if workers > 1 and hasattr(self.store, "conn"):
            workers = 1
        resumed_candidates = resumed_candidates or {}
        total = len(episodes)
        results: list[dict[str, Any] | None] = [None] * total
        completed = 0
        remaining: list[tuple[int, dict[str, Any]]] = []
        for idx, episode in enumerate(episodes):
            episode_id = str(episode.get("episode_id") or "")
            resumed = resumed_candidates.get(episode_id)
            if resumed is not None:
                results[idx] = resumed
                completed += 1
            else:
                remaining.append((idx, episode))
        if progress_callback is not None and completed:
            progress_callback(completed, total)
        if workers == 1 or len(remaining) <= 1:
            for idx, episode in remaining:
                candidate = self._extract_one_w2_candidate(
                    episode,
                    w2_mode=w2_mode,
                    source_type=source_type,
                )
                results[idx] = candidate
                self._append_partial_candidate(partial_candidates_path, candidate)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
            return [item for item in results if isinstance(item, dict)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._extract_one_w2_candidate,
                    episode,
                    w2_mode=w2_mode,
                    source_type=source_type,
                ): idx
                for idx, episode in remaining
            }
            for future in as_completed(futures):
                idx = futures[future]
                candidate = future.result()
                results[idx] = candidate
                self._append_partial_candidate(partial_candidates_path, candidate)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
        return [item for item in results if isinstance(item, dict)]

    def extract_w7_atomic_cases(
        self,
        atomic_case_manifest: dict[str, Any],
        *,
        w2_mode: str | None = "native_v2",
        source_type: str = "chat",
        workers: int = 1,
    ) -> dict[str, Any]:
        """Run W2 on W7a-isolated cases without queueing or applying them.

        This is the explicit W7a->W2 adapter boundary used by shadow/assisted
        evaluation.  The normal legacy episode path remains unchanged until
        the multi-agent release gates pass.
        """

        episodes = w2_atomic_episodes(atomic_case_manifest)
        for episode in episodes:
            episode["_write_source_type"] = source_type
        candidates = self._extract_w2_candidates(
            episodes,
            workers=workers,
            w2_mode=w2_mode,
            source_type=source_type,
        )
        return {
            "schema_version": "w7.atomic_w2_result.v1",
            "manifest_hash": str(
                atomic_case_manifest.get("manifest_hash") or ""
            ),
            "parent_episode_id": str(
                atomic_case_manifest.get("parent_episode_id") or ""
            ),
            "atomic_episode_ids": [
                str(item.get("episode_id") or "") for item in episodes
            ],
            "candidates": candidates,
            "w7b_case_cards": w7_case_cards_from_w2_candidates(candidates),
            "summary": {
                "atomic_cases": len(episodes),
                "candidates": len(candidates),
                "schema_valid": sum(
                    bool(item.get("production_schema_valid"))
                    for item in candidates
                ),
            },
            "queue_written": False,
            "kg_mutated": False,
        }

    def _extract_one_w2_candidate(
        self,
        episode: dict[str, Any],
        *,
        w2_mode: str | None,
        source_type: str,
    ) -> dict[str, Any]:
        """Isolate malformed source records without aborting a batch.

        W2 runs over imported history where an occasional empty/no-text episode
        is a data-quality result, not a pipeline-level failure.  The failed row
        remains visible and resumable as a schema-invalid candidate; W3/W4 can
        then deterministically reject it.
        """

        try:
            prepared = self._prepare_episode_for_w2(episode, source_type=source_type)
            return self.w2.extract(prepared, w2_mode=w2_mode)
        except Exception as exc:
            return self._failed_w2_candidate(episode, w2_mode=w2_mode, error=exc)

    @staticmethod
    def _failed_w2_candidate(
        episode: dict[str, Any],
        *,
        w2_mode: str | None,
        error: Exception,
    ) -> dict[str, Any]:
        episode_id = str(episode.get("episode_id") or episode.get("thread_id") or "unknown")
        thread_id = str(episode.get("thread_id") or episode.get("source_thread_id") or "")
        digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()[:16]
        if isinstance(error, NonSopIntakeError):
            error_code = error.code
            error_detail = error.to_dict()
        else:
            error_code = "extraction_exception"
            error_detail = {
                "code": error_code,
                "message": str(error),
                "details": {"exception_type": type(error).__name__},
            }
        issue = f"extraction_invalid:{error_code}"
        mode = str(w2_mode or "native_v2")
        return {
            "type": "SchemaValidCandidate",
            "candidate_type": "ChatKnowledgeCandidate",
            "candidate_id": f"chatcand:failed-{digest}",
            "id": f"chatcand:failed-{digest}",
            "status": "extraction_invalid",
            "auto_ingest": False,
            "proposal_only": True,
            "source": episode_id,
            "source_episode_id": episode_id,
            "source_thread_id": thread_id,
            "label": "空文本或无法提取的历史消息",
            "w2_mode": mode,
            "schema_valid": False,
            "schema_issues": [issue],
            "production_schema_valid": False,
            "production_schema_issues": [issue],
            "case_understanding_card": {},
            "case_understanding_card_schema_valid": False,
            "case_understanding_card_schema_issues": [issue],
            "candidate_draft_v2": {"split_cases": []},
            "candidate_draft_v2_schema_valid": False,
            "candidate_draft_v2_schema_issues": [issue],
            "candidate_draft_v2_normalized_bundle": {
                "schema_version": "kg_v2.candidate_bundle.v1",
                "schema_valid": False,
                "schema_issues": [issue],
                "objects": {},
                "relations": [],
            },
            "candidate_draft_v2_bundle_schema_valid": False,
            "candidate_draft_v2_bundle_schema_issues": [issue],
            "nodes": [],
            "edges": [],
            "required_info_candidates": [],
            "diagnostic_outcomes": [],
            "episode": episode,
            "extraction_error": error_detail,
            "observability": {
                "agent_id": "W2",
                "episode_id": episode_id,
                "thread_id": thread_id,
                "schema_valid": False,
                "deepseek_used": False,
                "context_evidence_policy": "current_episode_only.v1",
                "error": error_detail,
            },
        }

    def _prepare_episode_for_w2(self, episode: dict[str, Any], *, source_type: str = "chat") -> dict[str, Any]:
        if not self.review_context_enabled:
            return episode
        if self.review_context_kg_v2_root is not None:
            envelope = self._episode_intake_envelope(episode, source_type=source_type)
            background = review_ctx.build_non_sop_alignment_background(
                envelope,
                kg_v2_root=self.review_context_kg_v2_root,
                gold_root=self.review_context_gold_root,
                alignment_index=self._review_context_alignment_index(),
            )
            return self.w7.prepare_episode(episode, background)
        try:
            background = review_ctx.build_sop_background_for_episode(
                episode,
                self._review_context_sop(),
                self._review_context_examples(),
            )
            return self.w7.prepare_episode(episode, background)
        except Exception:
            return episode

    def _episode_intake_envelope(self, episode: dict[str, Any], *, source_type: str) -> dict[str, Any]:
        text = review_ctx.episode_text(episode)
        extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
        message_ids = sorted({str(item) for item in episode.get("evidence_message_ids") or [] if str(item)})
        return build_write_intake_envelope(
            source_type=source_type,
            text=text,
            source_ref={
                "episode_id": str(episode.get("episode_id") or ""),
                "thread_id": str(episode.get("thread_id") or ""),
                "message_ids": message_ids,
            },
            knowledge_kind="fault_case",
            payload={"text": text, "episode": episode},
            evidence_pack={
                "message_ids": message_ids,
                "case_evidence_messages": list(episode.get("case_evidence_messages") or []),
                "linked_jira_evidence": list(extracted.get("linked_jira_evidence") or []),
                "source_offsets": list(episode.get("source_offsets") or []),
                "attachments": list(episode.get("attachments") or []),
                "tool_evidence": extracted.get("tool_evidence") if isinstance(extracted.get("tool_evidence"), dict) else {},
            },
            lineage={
                "source_episode_id": str(episode.get("episode_id") or ""),
                "source_thread_id": str(episode.get("thread_id") or ""),
            },
        )

    def _preflight_non_sop_kg_v2(self, kg_mode: str) -> dict[str, Any]:
        if kg_mode not in {"v2", "both"}:
            return {}
        if self.review_context_kg_v2_root is None or self.kg_v2_store is None:
            raise NonSopIntakeError("kg_v2_baseline_missing", "Non-SOP v2 ingestion requires an explicit active data/kg_v2 root.")
        baseline_hash = compute_kg_v2_graph_hash(self.review_context_kg_v2_root)
        return {
            "build_mode": "non_sop_incremental",
            "baseline_graph_hash": baseline_hash,
            "source_policy": {
                "include": [
                    "chat", "text_history", "raw_doc", "jira", "attachment", "manual_review",
                    "diagnostic_feedback", "log_pattern",
                ],
                "exclude": ["sop"],
            },
        }

    def _review_context_sop(self) -> dict[str, Any]:
        if self._review_context_sop_cache is None:
            self._review_context_sop_cache = review_ctx.load_sop_seed_background(self.review_context_sop_seed_json)
        return self._review_context_sop_cache

    def _review_context_alignment_index(self) -> dict[str, Any]:
        if self._review_context_alignment_index_cache is None:
            if self.review_context_kg_v2_root is None:
                raise NonSopIntakeError("kg_v2_baseline_missing", "W7 alignment requires an active KG v2 root.")
            self._review_context_alignment_index_cache = load_alignment_context_index(
                kg_v2_root=self.review_context_kg_v2_root,
                gold_root=self.review_context_gold_root,
            )
        return self._review_context_alignment_index_cache

    def _review_context_examples(self) -> list[dict[str, Any]]:
        if self._review_context_examples_cache is None:
            self._review_context_examples_cache = review_ctx.load_reviewed_examples(
                gold_root=self.review_context_gold_root,
                manual_root=self.review_context_manual_root,
            )
        return self._review_context_examples_cache

    @staticmethod
    def _load_partial_candidates(path: Path | None) -> dict[str, dict[str, Any]]:
        if path is None or not path.exists():
            return {}
        out: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            episode_id = str(candidate.get("source_episode_id") or (candidate.get("episode") or {}).get("episode_id") or "")
            if episode_id:
                out[episode_id] = candidate
        return out

    @staticmethod
    def _append_partial_candidate(path: Path | None, candidate: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    @staticmethod
    def _episode_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        episode = candidate.get("episode") if isinstance(candidate.get("episode"), dict) else {}
        if episode:
            return episode
        return {
            "episode_id": str(candidate.get("source_episode_id") or candidate.get("candidate_id") or ""),
            "thread_id": str(candidate.get("source_thread_id") or ""),
            "completeness": "partial",
            "fault_description_messages": [],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "noise_messages": [],
            "attachments": [],
            "evidence_message_ids": list(candidate.get("evidence_ids") or []),
            "source_offsets": list(candidate.get("source_offsets") or []),
        }

    @staticmethod
    def _hydrate_v2_bundle_identity(candidate: dict[str, Any], episode: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(bundle, dict):
            return {}
        out = dict(bundle)
        objects = out.get("objects") if isinstance(out.get("objects"), dict) else {}
        families = [item for item in objects.get("FaultFamily") or [] if isinstance(item, dict)]
        variants = [item for item in objects.get("FaultVariant") or [] if isinstance(item, dict)]
        legacy_candidate_id = str(out.get("legacy_candidate_id") or candidate.get("candidate_id") or candidate.get("id") or "")
        source_episode_id = str(candidate.get("source_episode_id") or episode.get("episode_id") or "")
        source_thread_id = str(candidate.get("source_thread_id") or episode.get("thread_id") or "")
        if not out.get("candidate_id"):
            fallback = legacy_candidate_id or source_episode_id or source_thread_id or "unknown"
            out["candidate_id"] = f"v2:{fallback}"
        if not out.get("legacy_candidate_id"):
            out["legacy_candidate_id"] = legacy_candidate_id
        if not out.get("family_id") and families:
            out["family_id"] = str(families[0].get("family_id") or "")
        if not out.get("variant_id") and variants:
            out["variant_id"] = str(variants[0].get("variant_id") or "")
        if not out.get("source_text"):
            parts: list[str] = []
            for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_evidence_messages"):
                for message in episode.get(key) or []:
                    if isinstance(message, dict):
                        parts.append(str(message.get("text") or message.get("content_summary") or ""))
            out["source_text"] = " ".join(part for part in parts if part).strip()
        if not out.get("source_message_ids"):
            message_ids: list[str] = [
                str(value) for value in episode.get("evidence_message_ids") or [] if str(value)
            ]
            for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages", "case_evidence_messages"):
                for message in episode.get(key) or []:
                    if isinstance(message, dict) and str(message.get("message_id") or ""):
                        message_ids.append(str(message.get("message_id")))
            out["source_message_ids"] = list(dict.fromkeys(message_ids))
        if not out.get("source_messages"):
            source_messages: list[dict[str, str]] = []
            for role, key in (
                ("fault", "fault_description_messages"),
                ("diagnostic", "diagnostic_chain_messages"),
                ("resolution", "resolution_messages"),
                ("w7_promoted", "case_evidence_messages"),
            ):
                for message in episode.get(key) or []:
                    if not isinstance(message, dict) or not str(message.get("message_id") or ""):
                        continue
                    source_messages.append({
                        "message_id": str(message.get("message_id") or ""),
                        "role": role,
                        "text": str(message.get("text") or message.get("content_summary") or ""),
                    })
            out["source_messages"] = source_messages
        return out

    @staticmethod
    def _w1_w2_summary(run: dict[str, Any], episodes: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        outcome_types: Counter[str] = Counter()
        required_slots: Counter[str] = Counter()
        schema_issues: Counter[str] = Counter()
        candidate_roles: Counter[str] = Counter()
        completeness: Counter[str] = Counter(str(ep.get("completeness") or "") for ep in episodes)
        attachment_roles: Counter[str] = Counter()
        for ep in episodes:
            for att in ep.get("attachments") or []:
                if isinstance(att, dict):
                    attachment_roles[str(att.get("evidence_role") or att.get("kind") or "attachment")] += 1
        for candidate in candidates:
            variant = candidate.get("case_variant_candidate") if isinstance(candidate.get("case_variant_candidate"), dict) else {}
            candidate_roles[str(variant.get("entry_role") or "unknown")] += 1
            for outcome in candidate.get("diagnostic_outcomes") or []:
                if isinstance(outcome, dict):
                    outcome_types[str(outcome.get("outcome_type") or "unknown")] += 1
            for req in candidate.get("required_info_candidates") or []:
                if isinstance(req, dict):
                    required_slots[str(req.get("slot") or "unknown")] += 1
            for issue in candidate.get("schema_issues") or []:
                schema_issues[str(issue)] += 1
        return {
            "messages": len(run.get("messages") or []),
            "thread_summaries": len(run.get("thread_summaries") or []),
            "episodes": len(episodes),
            "episode_completeness": dict(completeness),
            "attachments": sum(len(ep.get("attachments") or []) for ep in episodes),
            "attachment_roles": dict(attachment_roles),
            "w2_candidates": len(candidates),
            "schema_valid_candidates": sum(1 for c in candidates if c.get("schema_valid")),
            "schema_invalid_candidates": sum(1 for c in candidates if not c.get("schema_valid")),
            "production_schema_valid_candidates": sum(1 for c in candidates if c.get("production_schema_valid", c.get("schema_valid"))),
            "production_schema_invalid_candidates": sum(1 for c in candidates if not c.get("production_schema_valid", c.get("schema_valid"))),
            "candidate_roles": dict(candidate_roles),
            "diagnostic_traces": sum(1 for c in candidates if c.get("diagnostic_trace")),
            "diagnostic_outcomes": sum(len(c.get("diagnostic_outcomes") or []) for c in candidates),
            "outcome_types": dict(outcome_types),
            "required_info_candidates": sum(len(c.get("required_info_candidates") or []) for c in candidates),
            "required_info_slots": dict(required_slots),
            "schema_issue_counts": dict(schema_issues),
            "deepseek_enabled_candidates": sum(1 for c in candidates if ((c.get("observability") or {}).get("deepseek_enabled"))),
            "deepseek_used_candidates": sum(1 for c in candidates if ((c.get("observability") or {}).get("deepseek_used"))),
            "deepseek_error_candidates": sum(1 for c in candidates if ((c.get("observability") or {}).get("deepseek_error"))),
        }

    @staticmethod
    def _w1_w2_samples(episodes: list[dict[str, Any]], candidates: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for episode, candidate in zip(episodes, candidates):
            outcomes = [x for x in candidate.get("diagnostic_outcomes") or [] if isinstance(x, dict)]
            required = [x for x in candidate.get("required_info_candidates") or [] if isinstance(x, dict)]
            if not outcomes and not required and episode.get("completeness") == "noise":
                continue
            samples.append({
                "episode_id": episode.get("episode_id"),
                "thread_id": episode.get("thread_id"),
                "completeness": episode.get("completeness"),
                "candidate_id": candidate.get("candidate_id"),
                "label": candidate.get("label"),
                "schema_valid": candidate.get("schema_valid"),
                "schema_issues": candidate.get("schema_issues") or [],
                "case_variant_candidate": candidate.get("case_variant_candidate") or {},
                "diagnostic_trace": candidate.get("diagnostic_trace") or {},
                "diagnostic_outcomes": outcomes[:8],
                "required_info_candidates": required[:8],
                "evidence_ids": candidate.get("evidence_ids") or [],
                "attachment_evidence": (candidate.get("attachment_evidence") or [])[:8],
                "source_offsets": (candidate.get("source_offsets") or [])[:8],
            })
            if len(samples) >= max(0, limit):
                break
        return samples

    def _write_w1_w2_run(
        self,
        out_dir: str | Path,
        run: dict[str, Any],
        candidates: list[dict[str, Any]],
        summary: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_files = self.w1.write_run(out / "w1", run)
        candidate_path = out / "w2_candidates.jsonl"
        with candidate_path.open("w", encoding="utf-8") as f:
            for candidate in candidates:
                f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        (out / "w1_w2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "w1_w2_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output_files.update({
            "w2_candidates": str(candidate_path),
            "w1_w2_summary": str(out / "w1_w2_summary.json"),
            "w1_w2_samples": str(out / "w1_w2_samples.json"),
        })
        return output_files

    @staticmethod
    def _episodes_from_summaries(
        summaries: list[dict[str, Any]],
        *,
        refine_trace: bool = False,
    ) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        for summary in summaries:
            raw = [x for x in summary.get("episodes") or [] if isinstance(x, dict)]
            if raw:
                # W7 trace inference is review metadata only.  Each episode
                # remains an independent W2 candidate and keeps its own
                # evidence/outcomes; no message concatenation happens here.
                episodes.extend(review_ctx.refine_episode_group(raw) if refine_trace else raw)
            else:
                episodes.append(summary)
        return episodes

    @staticmethod
    def _queue_for(conflict: dict[str, Any], gate: dict[str, Any]) -> str:
        issues = set(gate.get("issues") or [])
        if not gate.get("passed") or conflict.get("decision") == "Insufficient" or issues & {"noise_episode", "review_only_noise", "missing_check_or_solution", "missing_evidence", "schema_invalid"}:
            return "noise_candidates"
        if conflict.get("existing_error_id") or conflict.get("decision") in {"Refine", "Contradict"}:
            return "merge_candidates"
        return "candidates"
