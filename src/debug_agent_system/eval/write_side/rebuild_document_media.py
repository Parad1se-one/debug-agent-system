"""Rebuild first-class media assets for every canonical KG_v2 document."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import zipfile

from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline
from debug_agent_system.core.paths import project_root
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.source_chunk_builder import build_media_asset_graph
from debug_agent_system.knowledge_v2.validator import validate_graph


DEFAULT_REPORT = Path("data/results/document_media_rebuild.json")
MEDIA_RELATIONS = {"has_media", "section_media", "step_media", "action_media"}


def _source_archive_images(
    root: Path,
    documents: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for document in documents:
        source_path = str(document.get("source_path") or "")
        path = root / source_path
        if path.suffix.lower() != ".docx" or not path.is_file():
            continue
        with zipfile.ZipFile(path) as archive:
            result.update(
                (str(document.get("document_id") or ""), name)
                for name in archive.namelist()
                if name.startswith("word/media/") and not name.endswith("/")
            )
    return result


def _sag_media_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT media_refs_json FROM source_chunks WHERE media_refs_json != '[]'"
        ).fetchall()
        object_count = int(connection.execute(
            "SELECT COUNT(*) FROM objects WHERE object_type='MediaAsset'"
        ).fetchone()[0])
    refs = [
        item
        for (payload,) in rows
        for item in json.loads(str(payload or "[]"))
        if isinstance(item, dict)
    ]
    return {
        "path": str(path),
        "status": "ready",
        "media_asset_object_count": object_count,
        "media_chunk_count": len(rows),
        "media_reference_count": len(refs),
        "unique_source_occurrence_count": len({
            (str(item.get("source_path") or ""), str(item.get("archive_path") or ""))
            for item in refs
        }),
        "unique_content_hash_count": len({
            str(item.get("content_hash") or "") for item in refs
            if str(item.get("content_hash") or "")
        }),
    }


def rebuild(
    *,
    kg_root: Path,
    asset_root: Path,
    report_path: Path,
    sag_paths: list[Path],
    apply: bool,
) -> dict[str, Any]:
    repo_root = project_root(__file__)
    if not asset_root.is_absolute():
        asset_root = repo_root / asset_root
    store = JsonKGV2Store(kg_root)
    objects = {key: list(value) for key, value in store.objects_by_type.items()}
    documents = objects.get("KnowledgeDocument") or []
    media_assets, media_relations, media_stats = build_media_asset_graph(
        repo_root,
        documents,
        objects.get("KnowledgeSection") or [],
        procedure_steps=objects.get("ProcedureStep") or [],
        actions=objects.get("DiagnosticAction") or [],
        relations=store.relations,
        asset_root=asset_root,
    )
    old_media_ids = {
        str(item.get("media_id") or "")
        for item in objects.get("MediaAsset") or []
        if str(item.get("media_id") or "")
    }
    objects["MediaAsset"] = media_assets
    relations = [
        dict(item)
        for item in store.relations
        if isinstance(item, dict)
        and str(item.get("from") or "") not in old_media_ids
        and str(item.get("to") or "") not in old_media_ids
        and str(item.get("relation") or "") not in MEDIA_RELATIONS
    ]
    relations.extend(media_relations)
    issues = validate_graph(objects, relations, schema_root=kg_root / "schema")
    if issues:
        raise RuntimeError("proposed media graph invalid: " + "; ".join(issues[:40]))

    source_archive_images = _source_archive_images(repo_root, documents)
    graph_image_occurrences = {
        (
            str(occurrence.get("document_id") or ""),
            str(occurrence.get("archive_path") or ""),
        )
        for media in media_assets
        if str(media.get("media_kind") or "") == "image"
        for occurrence in media.get("source_occurrences") or []
    }
    missing_images = sorted(source_archive_images - graph_image_occurrences)
    unexpected_images = sorted(graph_image_occurrences - source_archive_images)
    if missing_images or unexpected_images:
        raise RuntimeError(
            f"source image coverage mismatch: missing={len(missing_images)}, "
            f"unexpected={len(unexpected_images)}"
        )

    by_document: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "image_occurrence_count": 0,
        "attachment_occurrence_count": 0,
        "media_ids": set(),
    })
    for media in media_assets:
        media_id = str(media.get("media_id") or "")
        media_kind = str(media.get("media_kind") or "")
        for occurrence in media.get("source_occurrences") or []:
            document_id = str(occurrence.get("document_id") or "")
            row = by_document[document_id]
            row[f"{media_kind}_occurrence_count"] += 1
            row["media_ids"].add(media_id)
    document_by_id = {
        str(item.get("document_id") or ""): item for item in documents
    }
    per_document = [
        {
            "document_id": document_id,
            "title": str((document_by_id.get(document_id) or {}).get("title") or ""),
            "image_occurrence_count": int(values["image_occurrence_count"]),
            "attachment_occurrence_count": int(values["attachment_occurrence_count"]),
            "unique_media_asset_count": len(values["media_ids"]),
        }
        for document_id, values in sorted(by_document.items())
    ]

    result: dict[str, Any] = {
        "schema_version": "debug_agent_system.document_media_rebuild.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "kg_root": str(kg_root),
        "asset_root": str(asset_root),
        "document_count": len(documents),
        "media_stats": media_stats,
        "source_archive_image_count": len(source_archive_images),
        "graph_image_occurrence_count": len(graph_image_occurrences),
        "missing_source_images": missing_images,
        "unexpected_graph_images": unexpected_images,
        "per_document": per_document,
        "proposed_object_counts": {
            key: len(value) for key, value in objects.items()
        },
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
            "sag_media_stats": [_sag_media_stats(path) for path in sag_paths],
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
    parser.add_argument(
        "--asset-root", type=Path, default=Path("data/kg_v2_sag/assets")
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--sag-path", action="append", type=Path, default=None)
    args = parser.parse_args(argv)
    result = rebuild(
        kg_root=args.kg_root,
        asset_root=args.asset_root,
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
