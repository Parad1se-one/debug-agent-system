from __future__ import annotations

import json
from pathlib import Path
import tempfile

from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline


def test_curated_build_copies_gold_cases_without_ingesting_them_and_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target = tmp_path / "kg_v2"
        summary = tmp_path / "summary.json"
        pipeline = WriteSideV2Pipeline(target)

        first = pipeline.build_curated_sop(summary_out=summary)
        second = pipeline.build_curated_sop(summary_out=summary)

        assert first["counts"] == second["counts"]
        assert first["raw_gold_case_ids"] == [f"goldcase-{idx:03d}" for idx in range(1, 11)]
        assert first["gold_cases_ingested"] is False
        assert first["counts"]["FaultVariant"] == 88
        # These two outcomes are curated SOP runtime-validation templates,
        # not copied gold-case outcomes.
        assert first["counts"]["ActionOutcome"] == 2
        assert (target / "gold_cases" / "index.json").exists()
        raw_sources = json.loads((target / "gold_cases" / "raw_source_texts.json").read_text(encoding="utf-8"))
        assert raw_sources["graph_ingestion"] is False
        assert len(raw_sources["cases"]) == 10

        source_cases = json.loads((target / "objects" / "source_cases.json").read_text(encoding="utf-8"))
        gold_cases_in_graph = [
            item
            for item in source_cases
            if item.get("source_kind") == "manual_review" and item.get("approved") is True
        ]
        assert gold_cases_in_graph == []
        assert pipeline.validate_current_graph()["status"] == "valid"


def test_curated_builder_cannot_replace_active_kg_without_explicit_bootstrap_flag() -> None:
    pipeline = WriteSideV2Pipeline("data/kg_v2")
    try:
        pipeline.build_curated_sop()
    except RuntimeError as exc:
        assert "active_curated_rebuild_disabled" in str(exc)
    else:
        raise AssertionError("expected active curated rebuild guard")
