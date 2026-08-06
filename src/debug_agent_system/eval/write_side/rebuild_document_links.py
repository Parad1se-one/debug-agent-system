"""Rebuild canonical parent/child and cross-reference document relations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline
from debug_agent_system.core.paths import project_root
from debug_agent_system.knowledge_v2.document_links import (
    DOCUMENT_LINK_RELATIONS,
    build_document_link_graph,
    extract_docx_hyperlinks,
)
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


DEFAULT_REPORT = Path("data/results/document_link_rebuild.json")


def rebuild(
    *,
    kg_root: Path,
    report_path: Path,
    sag_paths: list[Path],
    apply: bool,
) -> dict[str, Any]:
    repo_root = project_root(__file__)
    store = JsonKGV2Store(kg_root)
    objects = {
        object_type: [dict(item) for item in items]
        for object_type, items in store.objects_by_type.items()
    }
    documents = objects.get("KnowledgeDocument") or []
    for document in documents:
        source_path = Path(str(document.get("source_path") or ""))
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        document["source_links"] = extract_docx_hyperlinks(source_path)

    document_relations, link_report = build_document_link_graph(
        repo_root,
        documents,
    )
    relations = [
        dict(item)
        for item in store.relations
        if isinstance(item, dict)
        and str(item.get("relation") or "") not in DOCUMENT_LINK_RELATIONS
    ]
    relations.extend(document_relations)
    issues = validate_graph(objects, relations, schema_root=kg_root / "schema")
    if issues:
        raise RuntimeError(
            "proposed document-link graph invalid: " + "; ".join(issues[:40])
        )

    result: dict[str, Any] = {
        "schema_version": "debug_agent_system.document_link_rebuild.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "kg_root": str(kg_root),
        "document_count": len(documents),
        "link_report": link_report,
        "proposed_relation_count": len(relations),
        "validation": {"status": "valid", "issues": []},
    }
    if apply:
        write_result = store.replace_graph(objects, relations, validate=True)
        pipeline = WriteSideV2Pipeline(kg_root)
        materialized = pipeline.materialize_execution()
        sag_results = [
            pipeline.build_sqlite_sag(path, reset=True) for path in sag_paths
        ]
        final_validation = pipeline.validate_current_graph()
        if final_validation.get("status") != "valid":
            raise RuntimeError(
                "published graph invalid: "
                + "; ".join(final_validation.get("issues") or [])
            )
        result.update({
            "write_result": write_result,
            "materialized": materialized,
            "sag_results": sag_results,
            "final_validation": final_validation,
        })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-root", type=Path, default=Path("data/kg_v2"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--sag-path", action="append", type=Path, default=None)
    args = parser.parse_args(argv)
    result = rebuild(
        kg_root=args.kg_root,
        report_path=args.report,
        apply=args.apply,
        sag_paths=args.sag_path or [
            Path("data/kg_v2_sag/debug_agent_v2.sqlite"),
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
