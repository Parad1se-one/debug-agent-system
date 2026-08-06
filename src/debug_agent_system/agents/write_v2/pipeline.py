"""Isolated write-side pipeline for building and materializing KG v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from debug_agent_system.core.paths import project_root
from debug_agent_system.knowledge_v2 import (
    JsonKGV2Store,
    KGV2Materializer,
    build_doc_source_seed,
    build_manual_case_seed,
    build_media_asset_graph,
    build_sop_seed,
    build_sqlite_sag_v2,
    merge_bundles,
    validate_graph,
)

DEFAULT_SOP_SOURCE_MANIFEST = "data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json"
DEFAULT_MANUAL_ROOT = "data/kg/review_queue/manual_review_examples"
DEFAULT_CURATED_BUILD_ROOT = "data/kg_v2_sop_draft_build"
DEFAULT_GOLD_ROOT = "data/annotations/goldcases/gold-v1"
DEFAULT_CURATED_SUMMARY = "data/results/kg_v2_write_side_build_summary.json"
DEFAULT_SQLITE_SAG_V2 = "data/kg_v2_sag/debug_agent_v2.sqlite"


class WriteSideV2Pipeline:
    def __init__(self, root: str | Path = "data/kg_v2") -> None:
        self.store = JsonKGV2Store(root)

    def seed_sop(
        self,
        chunks_path: str | Path = DEFAULT_SOP_SOURCE_MANIFEST,
        *,
        limit: int = 0,
        replace: bool = False,
    ) -> dict[str, Any]:
        path = Path(chunks_path)
        if path.name.endswith("kg_v2_source_manifest.json"):
            bundle = build_doc_source_seed(chunks_path, limit=limit)
        else:
            bundle = build_sop_seed(chunks_path, limit=limit)
        return self._write_bundle(bundle, replace=replace)

    def seed_manual_cases(
        self,
        manual_root: str | Path = DEFAULT_MANUAL_ROOT,
        *,
        limit: int = 0,
        replace: bool = False,
    ) -> dict[str, Any]:
        bundle = build_manual_case_seed(manual_root, limit=limit)
        return self._write_bundle(bundle, replace=replace)

    def seed_all(
        self,
        *,
        chunks_path: str | Path = DEFAULT_SOP_SOURCE_MANIFEST,
        manual_root: str | Path = DEFAULT_MANUAL_ROOT,
        sop_limit: int = 0,
        manual_limit: int = 0,
        replace: bool = True,
    ) -> dict[str, Any]:
        path = Path(chunks_path)
        bundle = merge_bundles(
            build_doc_source_seed(chunks_path, limit=sop_limit)
            if path.name.endswith("kg_v2_source_manifest.json")
            else build_sop_seed(chunks_path, limit=sop_limit),
            build_manual_case_seed(manual_root, limit=manual_limit),
        )
        return self._write_bundle(bundle, replace=replace)

    def build_curated_sop(
        self,
        *,
        build_root: str | Path = DEFAULT_CURATED_BUILD_ROOT,
        gold_root: str | Path = DEFAULT_GOLD_ROOT,
        summary_out: str | Path = DEFAULT_CURATED_SUMMARY,
        allow_active_rebuild: bool = False,
    ) -> dict[str, Any]:
        """Bootstrap a graph from reviewed cards; never update the active KG by default."""

        active_root = project_root(__file__) / "data/kg_v2"
        if (
            self.store.root.resolve() == active_root.resolve()
            and not allow_active_rebuild
        ):
            raise RuntimeError(
                "active_curated_rebuild_disabled: use sync-sop-docs for "
                "versioned W4/W6/W5 updates; pass allow_active_rebuild only "
                "for an explicitly backed-up bootstrap or rollback"
            )

        from debug_agent_system.agents.write_v2.sop_manual_build import build_graph

        result = build_graph(
            target_root=self.store.root,
            build_root=build_root,
            gold_root=gold_root,
            summary_out=summary_out,
        )
        self.store = JsonKGV2Store(self.store.root)
        return {
            "status": "built",
            "target_root": result["target_root"],
            "build_root": result["build_root"],
            "raw_gold_case_ids": result["raw_gold_case_ids"],
            "gold_cases_ingested": result["gold_cases_ingested"],
            "counts": result["counts"],
            "relation_count": result["relation_count"],
            "materialized": result["materialized"],
            "summary_out": str(summary_out),
        }

    def validate_current_graph(self) -> dict[str, Any]:
        schema_root = self.store.root / "schema"
        issues = validate_graph(
            self.store.objects_by_type,
            self.store.relations,
            schema_root=schema_root if (schema_root / "object-types.json").exists() else None,
        )
        return {
            "status": "valid" if not issues else "invalid",
            "issues": issues,
            "object_counts": {key: len(value) for key, value in self.store.objects_by_type.items()},
            "relation_count": len(self.store.relations),
        }

    def materialize_execution(self, out_root: str | Path | None = None) -> dict[str, Any]:
        policy_result = self.recompute_policies()
        self.store = JsonKGV2Store(self.store.root)
        target = Path(out_root) if out_root is not None else self.store.materialized_root
        materialized = KGV2Materializer(self.store).materialize(target)
        return {
            "status": "materialized",
            "out_root": str(target),
            "counts": {key: len(value) for key, value in materialized.items() if isinstance(value, list)},
            "policy_recompute": policy_result,
        }

    def build_sqlite_sag(self, out: str | Path = DEFAULT_SQLITE_SAG_V2, *, reset: bool = True) -> dict[str, Any]:
        return build_sqlite_sag_v2(self.store.root, out, reset=reset)

    def rebuild_media_assets(
        self,
        *,
        asset_root: str | Path = "data/kg_v2_sag/assets",
    ) -> dict[str, Any]:
        """Reparse every canonical source document and publish its media layer."""

        repo_root = project_root(__file__)
        resolved_asset_root = Path(asset_root)
        if not resolved_asset_root.is_absolute():
            resolved_asset_root = repo_root / resolved_asset_root
        media_assets, media_relations, stats = build_media_asset_graph(
            repo_root,
            self.store.objects_by_type.get("KnowledgeDocument") or [],
            self.store.objects_by_type.get("KnowledgeSection") or [],
            procedure_steps=self.store.objects_by_type.get("ProcedureStep") or [],
            actions=self.store.objects_by_type.get("DiagnosticAction") or [],
            relations=self.store.relations,
            asset_root=resolved_asset_root,
        )
        old_media_ids = {
            str(item.get("media_id") or "")
            for item in self.store.objects_by_type.get("MediaAsset") or []
            if str(item.get("media_id") or "")
        }
        objects = {
            key: list(value) for key, value in self.store.objects_by_type.items()
        }
        objects["MediaAsset"] = media_assets
        relations = [
            dict(item)
            for item in self.store.relations
            if isinstance(item, dict)
            and str(item.get("from") or "") not in old_media_ids
            and str(item.get("to") or "") not in old_media_ids
            and str(item.get("relation") or "") not in {
                "has_media",
                "section_media",
                "step_media",
                "action_media",
            }
        ]
        relations.extend(media_relations)
        result = self.store.replace_graph(objects, relations, validate=True)
        if result.get("status") == "replaced":
            self.store = JsonKGV2Store(self.store.root)
        return {**result, "media_stats": stats}

    def _write_bundle(self, bundle: dict[str, Any], *, replace: bool) -> dict[str, Any]:
        writer = self.store.replace_graph if replace else self.store.merge_graph
        result = writer(bundle.get("objects") or {}, bundle.get("relations") or [])
        if result.get("status") in {"replaced", "merged"}:
            self.store = JsonKGV2Store(self.store.root)
        return {
            **result,
            "report": bundle.get("report") or {},
        }

    def recompute_policies(self) -> dict[str, Any]:
        materializer = KGV2Materializer(self.store)
        policies = materializer.build_policy_objects()
        objects = {key: list(value) for key, value in self.store.objects_by_type.items()}
        old_policy_ids = {
            str(item.get("policy_id") or "")
            for item in objects.get("DecisionPolicy") or []
            if isinstance(item, dict)
        }
        objects["DecisionPolicy"] = policies
        relations = [
            dict(item)
            for item in self.store.relations
            if isinstance(item, dict)
            and str(item.get("from") or "") not in old_policy_ids
            and str(item.get("to") or "") not in old_policy_ids
            and str(item.get("relation") or "") != "for_family"
        ]
        relations.extend(
            {
                "from": str(policy.get("policy_id") or ""),
                "to": str(policy.get("family_id") or ""),
                "relation": "for_family",
            }
            for policy in policies
        )
        result = self.store.replace_graph(objects, relations, validate=True)
        self.store = JsonKGV2Store(self.store.root)
        return {
            **result,
            "policy_count": len(policies),
        }
