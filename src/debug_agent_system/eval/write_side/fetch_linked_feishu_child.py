"""Fetch one Feishu child document by its DOCX relationship anchor.

The command is deliberately source-driven: callers identify a local parent
DOCX and an ``rId``.  The target URL, visible label and wiki token are read
from the parent package rather than repeated in query-specific code.

An authenticated ``lark-cli`` is required for remote access.  A failure before
publication writes an auditable report without mutating KG_v2.  ``--apply``
publishes the exact exported document layer, records lineage, rebuilds document
links, and then rebuilds SAG.  The source DOCX remains the hash-pinned source
of truth; embedded images are materialized by the standard SAG source chunk
builder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from debug_agent_system.agents.write.w9_raw_doc_ingest import RawDocIngestAgent
from debug_agent_system.agents.write.w10_section_case_bundle import (
    SectionCaseBundleAgent,
)
from debug_agent_system.core.paths import project_root
from debug_agent_system.eval.write_side.rebuild_document_links import rebuild
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.document_links import extract_docx_hyperlinks
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


DEFAULT_PARENT = Path("data/raw/aoi_debug_agent_sources/Dism++软件使用教程.docx")
DEFAULT_OUTPUT_ROOT = Path("data/raw/aoi_debug_agent_sources")
DEFAULT_REPORT = Path("data/results/linked_feishu_child_fetch.json")
_SAFE_NAME = re.compile(r'[<>:"/\\\\|?*\\x00-\\x1f]+')


def fetch_linked_child(
    *,
    parent_source: Path,
    relationship_id: str,
    output_root: Path,
    report_path: Path,
    kg_root: Path,
    sag_paths: list[Path],
    lark_cli: str,
    apply: bool,
) -> dict[str, Any]:
    root = project_root(__file__)
    parent_path = _resolve(root, parent_source)
    output_dir = _resolve(root, output_root)
    report_file = _resolve(root, report_path)
    kg_dir = _resolve(root, kg_root)
    links = extract_docx_hyperlinks(parent_path)
    link = next(
        (
            item
            for item in links
            if str(item.get("relationship_id") or "") == relationship_id
        ),
        None,
    )
    if link is None:
        raise ValueError(
            f"relationship anchor not found: {parent_path}#{relationship_id}"
        )
    result: dict[str, Any] = {
        "schema_version": "debug_agent_system.linked_feishu_child_fetch.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "fetch_only",
        "status": "fetching",
        "parent": {
            "source_path": _relative(root, parent_path),
            "content_hash": _sha256(parent_path),
        },
        "source_anchor": dict(link),
        "child": {},
        "commands": [],
    }
    kg_mutated = False
    try:
        executable = _find_lark_cli(lark_cli)
        inspect = _run_lark(
            executable,
            [
                "drive", "+inspect", "--url", str(link["target_url"]),
                "--format", "json", "--as", "user",
            ],
            cwd=root,
            timeout=300,
        )
        result["commands"].append(_command_audit(inspect))
        data = inspect.get("json") or {}
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = data["data"]
        token = str(data.get("token") or data.get("obj_token") or "").strip()
        doc_type = str(data.get("type") or data.get("obj_type") or "").strip()
        remote_title = str(
            data.get("title") or link.get("link_text") or token or "飞书子文档"
        ).strip()
        # The visible parent-link label is the canonical graph title.  Remote
        # titles can drift independently and are retained in lineage instead
        # of weakening deterministic parent/child resolution.
        title = str(link.get("link_text") or remote_title).strip()
        if not token or doc_type not in {"doc", "docx"}:
            raise RuntimeError(
                f"linked target is not an exportable Feishu document: "
                f"type={doc_type or 'unknown'} token={token or 'missing'}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        before = {item.resolve() for item in output_dir.glob("*.docx")}
        export = _run_lark(
            executable,
            [
                "drive", "+export", "--token", token, "--doc-type", doc_type,
                "--file-extension", "docx", "--file-name", _safe_name(title),
                "--output-dir", ".", "--overwrite", "--as", "user",
            ],
            cwd=output_dir,
            timeout=900,
        )
        result["commands"].append(_command_audit(export))
        child_path = _locate_export(output_dir, before, title)
        fetch = _run_lark(
            executable,
            [
                "docs", "+fetch", "--api-version", "v2", "--doc",
                str(link["target_url"]), "--doc-format", "xml", "--detail",
                "full", "--format", "json", "--as", "user",
            ],
            cwd=root,
            timeout=600,
        )
        result["commands"].append(_command_audit(fetch))
        snapshot_dir = output_dir / "_feishu_source_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{link.get('wiki_token') or token}.json"
        snapshot_path.write_text(
            json.dumps(fetch.get("json") or {}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        lineage = {
            "schema_version": "debug_agent_system.feishu_source_lineage.v1",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "parent_source_path": _relative(root, parent_path),
            "parent_content_hash": _sha256(parent_path),
            "relationship_id": relationship_id,
            "link_text": str(link.get("link_text") or ""),
            "target_url": str(link.get("target_url") or ""),
            "wiki_token": str(link.get("wiki_token") or ""),
            "child_title": title,
            "child_remote_title": remote_title,
            "child_token": token,
            "child_type": doc_type,
            "child_source_path": _relative(root, child_path),
            "child_content_hash": _sha256(child_path),
            "fetch_snapshot_path": _relative(root, snapshot_path),
        }
        lineage_path = snapshot_dir / (
            f"{link.get('wiki_token') or token}.lineage.json"
        )
        lineage_path.write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["child"] = {
            **lineage,
            "lineage_path": _relative(root, lineage_path),
        }
        if apply:
            publication = _publish_document_layer(
                root=root,
                kg_root=kg_dir,
                child_path=child_path,
                title=title,
            )
            kg_mutated = publication.get("write_result", {}).get(
                "status"
            ) == "merged"
            rebuild_result = rebuild(
                kg_root=kg_dir,
                report_path=report_file.with_name(
                    report_file.stem + "_document_links.json"
                ),
                sag_paths=[_resolve(root, item) for item in sag_paths],
                apply=True,
            )
            result["publication"] = publication
            result["rebuild"] = {
                "validation": rebuild_result.get("final_validation"),
                "sag_results": rebuild_result.get("sag_results"),
                "link_report": {
                    key: rebuild_result["link_report"].get(key)
                    for key in (
                        "resolved_relation_count",
                        "unresolved_count",
                        "child_relation_count",
                    )
                },
            }
        result["status"] = "published" if apply else "fetched"
    except Exception as exc:
        result["status"] = "fetch_failed"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "kg_mutated": kg_mutated,
        }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _publish_document_layer(
    *,
    root: Path,
    kg_root: Path,
    child_path: Path,
    title: str,
) -> dict[str, Any]:
    relative_path = Path(_relative(root, child_path))
    payload = RawDocIngestAgent().build_section_cases(relative_path)
    bundle = SectionCaseBundleAgent().build_bundle(payload)
    allowed = {
        "KnowledgeDocument", "KnowledgeSection", "ProcedureStep", "EvidenceItem"
    }
    objects = {
        object_type: [
            dict(item)
            for item in (bundle.get("objects") or {}).get(object_type) or []
            if isinstance(item, dict)
        ]
        for object_type in allowed
    }
    for document in objects["KnowledgeDocument"]:
        document.update({
            "title": title,
            "approved": True,
            "source_kind": "raw_doc",
        })
    valid_ids = {
        str(item.get(V2_PRIMARY_KEYS[object_type]) or "")
        for object_type in allowed
        for item in objects.get(object_type) or []
    }
    relations = [
        dict(item)
        for item in bundle.get("relations") or []
        if isinstance(item, dict)
        and str(item.get("from") or "") in valid_ids
        and str(item.get("to") or "") in valid_ids
    ]
    issues = validate_graph(objects, relations, schema_root=kg_root / "schema")
    if issues:
        raise RuntimeError("fetched document layer invalid: " + "; ".join(issues))
    store = JsonKGV2Store(kg_root)
    write_result = store.merge_graph(objects, relations, validate=True)
    return {
        "write_result": write_result,
        "object_counts": {key: len(value) for key, value in objects.items()},
        "relation_count": len(relations),
    }


def _run_lark(
    executable: str,
    args: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    command = [executable, *args]
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    parsed: Any = None
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
        "json": parsed,
    }
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"lark-cli exited with {completed.returncode}"
        )
    return result


def _command_audit(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": list(result.get("command") or []),
        "returncode": int(result.get("returncode") or 0),
    }


def _find_lark_cli(value: str) -> str:
    candidate = str(value or "").strip()
    if candidate:
        resolved = shutil.which(candidate) or (
            candidate if Path(candidate).is_file() else ""
        )
    else:
        resolved = shutil.which("lark-cli") or shutil.which("lark") or ""
    if not resolved:
        raise FileNotFoundError(
            "authenticated lark-cli not found; provide --lark-cli or install "
            "the user-authenticated Feishu CLI"
        )
    return resolved


def _locate_export(output_dir: Path, before: set[Path], title: str) -> Path:
    candidates = [
        item
        for item in output_dir.glob("*.docx")
        if item.resolve() not in before
    ]
    if not candidates:
        expected = output_dir / f"{_safe_name(title)}.docx"
        if expected.is_file():
            return expected
        raise RuntimeError("Feishu export completed without a DOCX output")
    source = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    target = output_dir / f"{_safe_name(title)}.docx"
    if source != target:
        source.replace(target)
    return target


def _safe_name(value: str) -> str:
    return _SAFE_NAME.sub("_", str(value or "")).strip(" .")[:120] or "飞书子文档"


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _relative(root: Path, value: Path) -> str:
    try:
        return str(value.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(value.resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-source", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--relationship-id", default="rId5")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--kg-root", type=Path, default=Path("data/kg_v2"))
    parser.add_argument(
        "--sag-path",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--lark-cli", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = fetch_linked_child(
        parent_source=args.parent_source,
        relationship_id=args.relationship_id,
        output_root=args.output_root,
        report_path=args.report,
        kg_root=args.kg_root,
        sag_paths=args.sag_path or [
            Path("data/kg_v2_sag/debug_agent_v2.sqlite")
        ],
        lark_cli=args.lark_cli,
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"fetched", "published"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
