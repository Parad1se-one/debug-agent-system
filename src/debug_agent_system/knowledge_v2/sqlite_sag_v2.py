"""SQLite SAG_v2 indexes for variant decisions and source-grounded answers."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.query_scope import (
    analyze_query_scope,
    matched_strong_identifiers,
    scope_polarity_compatible,
    source_document_title,
    subject_domain_compatible,
    title_match_signals,
)
from debug_agent_system.knowledge_v2.source_chunk_builder import rebuild_source_chunks


SAG_V2_INDEX_SCHEMA = "debug_agent_system.sqlite_sag_v2.v15"
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
_IDENTIFIER = re.compile(r"0x[0-9a-f]{6,8}|[a-z0-9_]+(?:[._+-][a-z0-9_]+)+|[a-z0-9_]{2,}")
_GENERIC_TERMS = {
    "问题", "异常", "设备", "系统", "程序", "用户", "检查", "正常", "失败", "故障",
    "现场", "情况", "处理", "相关", "当前", "出现", "进行", "结果", "测试", "资料",
}


class SqliteSAGV2:
    """Read-only facade over a revision-pinned dual-channel SAG_v2 index."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.last_retrieval_trace: dict[str, Any] = {}

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Compatibility object search used by diagnostics and older tests."""

        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = _fts_rows(
                conn,
                table="object_fts",
                select_sql="""
                    SELECT o.object_id, o.object_type, o.label, o.summary, o.payload_json,
                           bm25(object_fts) AS rank
                    FROM object_fts f JOIN objects o ON o.object_id = f.object_id
                """,
                query=query,
                limit=limit,
            )
            if not rows:
                rows = self._like_search(conn, _search_terms(query) or [query.strip()], limit)
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def retrieve(
        self,
        query: str,
        *,
        variant_limit: int = 200,
        chunk_limit: int = 24,
    ) -> dict[str, Any]:
        """Recall variants and answer chunks while retaining every native path."""

        empty = {"variant_rows": [], "chunks": [], "paths": [], "trace": {}}
        if not self.sqlite_path.exists():
            self.last_retrieval_trace = {"mode": "sqlite_sag_v2", "index_missing": True}
            return empty | {"trace": dict(self.last_retrieval_trace)}
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            variant_rows = _fts_rows(
                conn,
                table="variant_fts",
                select_sql="""
                    SELECT v.variant_id, v.family_id, v.label, v.search_text,
                           bm25(variant_fts, 0.0, 1.0) AS rank
                    FROM variant_fts f JOIN variant_documents v ON v.variant_id = f.variant_id
                """,
                query=query,
                limit=variant_limit,
            )
            # Retrieve a wider raw-BM25 pool before applying the bounded
            # relevance score below. Overlapping Chinese n-grams can otherwise
            # let a very short title/reference crowd useful body chunks out of
            # the final window.
            chunk_select_sql = """
                    SELECT c.chunk_id, c.object_id, c.object_type, c.source_kind, c.source_label,
                           c.text, c.source_offsets_json, c.media_refs_json,
                           c.content_hash, c.source_file_hash, c.approved,
                           c.variant_ids_json, c.paths_json, bm25(chunk_fts, 0.0, 1.0) AS rank
                    FROM chunk_fts f JOIN source_chunks c ON c.chunk_id = f.chunk_id
                    WHERE c.approved = 1
                """
            chunk_rows = _fts_rows(
                conn,
                table="chunk_fts",
                select_sql=chunk_select_sql,
                query=query,
                limit=max(int(chunk_limit) * 3, 72),
            )
            # Verbose natural-language questions can let many generic symptom
            # n-grams fill the BM25 pool before the one source paragraph that
            # contains the explicitly named tool.  Give every strong entity a
            # small independent FTS recall lane, then score all rows with the
            # complete query below.  This is bounded indexed retrieval, not a
            # full-table content scan.
            row_by_id = {
                str(row["chunk_id"]): row
                for row in chunk_rows
            }
            for identifier in analyze_query_scope(query).strong_identifiers:
                for row in _fts_rows(
                    conn,
                    table="chunk_fts",
                    select_sql=chunk_select_sql,
                    query=identifier,
                    limit=12,
                ):
                    row_by_id.setdefault(str(row["chunk_id"]), row)
            chunk_rows = list(row_by_id.values())

            variant_by_id: dict[str, dict[str, Any]] = {}
            paths: list[dict[str, Any]] = []
            for row in variant_rows:
                item = dict(row)
                variant_id = str(item.get("variant_id") or "")
                item["recall_score"] = _rank_score(item.get("rank"))
                item["matched_terms"] = _matched_terms(query, str(item.get("search_text") or ""))
                item.pop("search_text", None)
                variant_by_id[variant_id] = item
                paths.append({
                    "seed_object_id": f"variant-document:{variant_id}",
                    "seed_object_type": "VariantSearchDocument",
                    "relation": "indexes_variant",
                    "variant_id": variant_id,
                    "method": "variant_fts",
                    "rank": item.get("rank"),
                })

            chunks: list[dict[str, Any]] = []
            for row in chunk_rows:
                raw = dict(row)
                variant_ids = _json_list(raw.pop("variant_ids_json", "[]"))
                chunk_paths = _json_list(raw.pop("paths_json", "[]"))
                text = str(raw.get("text") or "")
                score, score_components = _chunk_relevance_score(
                    query,
                    text,
                    source_label=str(raw.get("source_label") or ""),
                    raw_rank=raw.get("rank"),
                )
                chunk = {
                    **raw,
                    "approved": bool(raw.get("approved")),
                    "source_offsets": _json_list(raw.pop("source_offsets_json", "[]")),
                    "media_refs": self._portable_media_refs(
                        _json_list(raw.pop("media_refs_json", "[]"))
                    ),
                    "variant_ids": variant_ids,
                    "raw_retrieval_score": _rank_score(raw.get("rank")),
                    "retrieval_score": score,
                    "score_components": score_components,
                    "retrieval_paths": chunk_paths,
                    "matched_terms": _matched_terms(query, text),
                }
                chunk["document_id"] = _document_id_for_chunk(conn, chunk)
                chunks.append(chunk)
                distinctive_matches = [
                    term for term in chunk.get("matched_terms") or []
                    if len(str(term)) >= 3
                ]
                variant_support_eligible = bool(
                    float(score_components.get("query_coverage") or 0.0) >= 0.2
                    and score >= 2.0
                    and distinctive_matches
                )
                if not variant_support_eligible:
                    continue
                for variant_id in variant_ids:
                    for path in chunk_paths or [{
                        "seed_object_id": chunk["object_id"],
                        "seed_object_type": chunk["object_type"],
                        "relation": "supports_variant",
                        "variant_id": variant_id,
                    }]:
                        paths.append({
                            **path,
                            "chunk_id": chunk["chunk_id"],
                            "method": "chunk_fts",
                            "rank": chunk.get("rank"),
                        })
                    current = variant_by_id.setdefault(variant_id, {
                        "variant_id": variant_id,
                        "family_id": "",
                        "label": variant_id,
                        "rank": chunk.get("rank"),
                        "recall_score": 0.0,
                        "matched_terms": [],
                    })
                    current["chunk_support_score"] = round(
                        float(current.get("chunk_support_score") or 0.0)
                        + min(float(chunk.get("retrieval_score") or 0.0), 8.0) * 0.25,
                        4,
                    )

            chunks.sort(key=lambda item: (
                -float(item.get("retrieval_score") or 0.0),
                -len(str(item.get("text") or "")),
                str(item.get("chunk_id") or ""),
            ))
            direct_documents = _direct_document_matches(
                conn,
                query,
                [
                    *chunks,
                    *_document_title_probe_chunks(conn, query),
                ],
            )
            navigation_documents, navigation_excluded = _navigation_document_matches(
                conn,
                query,
                direct_documents,
            )
            navigation_parent_ids = {
                str(item.get("parent_document_id") or "")
                for item in navigation_documents
                if str(item.get("parent_document_id") or "")
            }
            direct_document_ids = {
                str(item.get("document_id") or "")
                for item in direct_documents
                if str(item.get("document_id") or "")
            }
            navigation_document_ids = {
                str(item.get("document_id") or "")
                for item in navigation_documents
                if str(item.get("document_id") or "")
            }
            section_scopes: dict[str, dict[str, set[str]]] = {}
            for item in direct_documents:
                if str(item.get("expansion_scope") or "") != "section":
                    continue
                document_id = str(item.get("document_id") or "")
                if not document_id or document_id in navigation_document_ids:
                    continue
                scope = section_scopes.setdefault(
                    document_id,
                    {"chunk_ids": set(), "source_labels": set()},
                )
                chunk_id = str(item.get("chunk_id") or "")
                source_label = str(item.get("entry_source_label") or "")
                if chunk_id and not chunk_id.startswith("title-probe:"):
                    scope["chunk_ids"].add(chunk_id)
                if source_label:
                    scope["source_labels"].add(_semantic_key(source_label))
            expanded = _expand_document_chunks(
                conn,
                query,
                [
                    *sorted(direct_document_ids - navigation_parent_ids),
                    *[
                        str(item.get("document_id") or "")
                        for item in navigation_documents
                    ],
                ],
                max_documents=12,
                asset_root=self.sqlite_path.parent / "assets",
                section_scopes=section_scopes,
            )
            by_chunk_id = {
                str(item.get("chunk_id") or ""): item
                for item in chunks
                if str(item.get("chunk_id") or "")
                and (
                    str(item.get("document_id") or "") not in navigation_parent_ids
                    # A navigation parent can also contain query-relevant
                    # source facts of its own (for example the Dism++ owner
                    # table).  Keep its raw, traceable source chunks while
                    # suppressing synthetic parent summaries that would
                    # duplicate the expanded child documents.
                    or (
                        str(item.get("chunk_id") or "").startswith("chunk:source:")
                        and _keep_navigation_parent_source_chunk(query, item)
                    )
                )
            }
            for item in expanded:
                by_chunk_id.setdefault(str(item.get("chunk_id") or ""), item)
            chunks = list(by_chunk_id.values())
            navigation_by_document = {
                str(item.get("document_id") or ""): item
                for item in navigation_documents
            }
            for chunk in chunks:
                chunk_document_id = str(chunk.get("document_id") or "")
                section_scope = section_scopes.get(chunk_document_id)
                inside_direct_scope = (
                    section_scope is None
                    or str(chunk.get("chunk_id") or "")
                    in section_scope["chunk_ids"]
                    or _semantic_key(str(chunk.get("source_label") or ""))
                    in section_scope["source_labels"]
                )
                if (
                    chunk_document_id
                    in direct_document_ids | navigation_document_ids
                    and inside_direct_scope
                ):
                    chunk["direct_document_match"] = True
                    chunk["direct_expansion_scope"] = (
                        "section" if section_scope is not None else "document"
                    )
                if chunk_document_id in navigation_document_ids:
                    navigation = navigation_by_document.get(chunk_document_id) or {}
                    chunk["navigation_document_match"] = True
                    chunk["navigation_order"] = int(
                        navigation.get("navigation_order") or 999999
                    )
                    chunk["navigation_depth"] = int(
                        navigation.get("navigation_depth") or 1
                    )
                    chunk["navigation_path"] = list(
                        navigation.get("navigation_path") or []
                    )
                    chunk["navigation_document_path"] = list(
                        navigation.get("navigation_document_path") or []
                    )
                    chunk["navigation_order_path"] = list(
                        navigation.get("navigation_order_path") or []
                    )
                    chunk["navigation_paths"] = list(
                        navigation.get("navigation_paths") or []
                    )
                    chunk["navigation_branch_score"] = float(
                        navigation.get("branch_score") or 0.0
                    )
                    chunk["navigation_selection_reason"] = str(
                        navigation.get("selection_reason") or ""
                    )
            chunks.sort(key=lambda item: (
                0 if item.get("direct_document_match") and str(item.get("chunk_id") or "").startswith("chunk:source:") else 1,
                int(item.get("navigation_order") or 999999),
                -float(item.get("retrieval_score") or 0.0),
                -len(str(item.get("text") or "")),
                str(item.get("chunk_id") or ""),
            ))
            chunks = chunks[: max(int(chunk_limit), 1)]

            # Generic object seeds remain a secondary graph recall channel.
            object_paths = self._candidate_paths_from_object_seeds(conn, query, limit=min(variant_limit, 240))
            paths.extend(object_paths)
            for path in object_paths:
                variant_id = str(path.get("variant_id") or "")
                if variant_id and variant_id not in variant_by_id:
                    variant_by_id[variant_id] = {
                        "variant_id": variant_id,
                        "family_id": "",
                        "label": variant_id,
                        "rank": 0.0,
                        "recall_score": 0.0,
                        "matched_terms": [],
                    }

            ordered = list(variant_by_id.values())
            ordered.sort(key=lambda item: (
                -float(item.get("recall_score") or 0.0),
                -float(item.get("chunk_support_score") or 0.0),
                str(item.get("variant_id") or ""),
            ))
            trace = {
                "mode": "sqlite_sag_v2_dual_channel",
                "index_schema": self.index_schema(),
                "query_terms": _search_terms(query),
                "variant_candidate_count": len(ordered),
                "supporting_chunk_count": len(chunks),
                "path_count": len(paths),
                "orphan_chunk_count": sum(not item.get("variant_ids") for item in chunks),
                "direct_document_matches": direct_documents,
                "navigation_document_matches": navigation_documents,
                "navigation_parent_document_ids": sorted(navigation_parent_ids),
                "navigation_excluded": navigation_excluded,
                "navigation_max_depth": 2,
                "fallback_used": not bool(variant_rows),
            }
            self.last_retrieval_trace = trace
            return {"variant_rows": ordered, "chunks": chunks, "paths": paths, "trace": trace}
        finally:
            conn.close()

    def _portable_media_refs(self, values: Iterable[Any]) -> list[Any]:
        """Rebase materialized media to the asset tree beside the SAG index.

        Older indexes intentionally retain the absolute build-machine path for
        auditability.  A packaged runtime must not depend on that machine, so
        prefer ``<sqlite parent>/assets/<source>/<file>`` whenever the
        corresponding materialized asset exists.
        """

        return _portable_media_refs(values, self.sqlite_path.parent / "assets")

    def candidate_variant_ids(self, query: str, limit: int = 200) -> tuple[list[str], list[dict[str, Any]]]:
        result = self.retrieve(query, variant_limit=limit)
        ids = [str(item.get("variant_id") or "") for item in result["variant_rows"]]
        return [item for item in ids if item], list(result["paths"])

    def expand_source_document_chunks(
        self,
        query: str,
        document_ids: Iterable[str],
    ) -> list[dict[str, Any]]:
        """Return the complete parsed source outline for selected documents."""

        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            return [
                item
                for item in _expand_document_chunks(
                    conn,
                    query,
                    [str(value) for value in document_ids if str(value)],
                    asset_root=self.sqlite_path.parent / "assets",
                )
                if str(item.get("chunk_id") or "").startswith("chunk:source:")
            ]
        finally:
            conn.close()

    def _candidate_paths_from_object_seeds(
        self,
        conn: sqlite3.Connection,
        query: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        seeds = self.search(query, limit=limit)
        paths: list[dict[str, Any]] = []
        for seed in seeds:
            object_id = str(seed.get("object_id") or "")
            object_type = str(seed.get("object_type") or "")
            payload = json.loads(str(seed.get("payload_json") or "{}"))
            found: list[tuple[str, str]] = []
            if object_type == "FaultVariant":
                found.append((object_id, "seed_is_variant"))
            variant_id = str(payload.get("variant_id") or "")
            if variant_id and variant_id != object_id:
                found.append((variant_id, f"{object_type}.variant_id"))
            family_id = object_id if object_type == "FaultFamily" else str(payload.get("family_id") or "")
            if family_id:
                rows = conn.execute(
                    "SELECT dst FROM relations WHERE src=? AND relation='has_variant' LIMIT 80",
                    (family_id,),
                ).fetchall()
                found.extend((str(row["dst"]), "has_variant") for row in rows)
            for found_variant_id, relation in found:
                if found_variant_id.startswith("variant:"):
                    paths.append({
                        "seed_object_id": object_id,
                        "seed_object_type": object_type,
                        "relation": relation,
                        "variant_id": found_variant_id,
                        "method": "object_fts",
                        "rank": seed.get("rank"),
                    })
        return paths

    def graph_revision(self) -> str:
        return self._metadata("graph_revision")

    def index_schema(self) -> str:
        return self._metadata("index_schema")

    def source_revision(self) -> str:
        return self._metadata("source_revision")

    def _metadata(self, key: str) -> str:
        if not self.sqlite_path.exists():
            return ""
        conn = sqlite3.connect(self.sqlite_path)
        try:
            row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            return str(row[0]) if row else ""
        except sqlite3.OperationalError:
            return ""
        finally:
            conn.close()

    @staticmethod
    def _like_search(conn: sqlite3.Connection, terms: list[str], limit: int) -> list[sqlite3.Row]:
        usable = [term for term in terms if len(term) >= 2][:12]
        if not usable:
            return []
        clauses = " OR ".join("label LIKE ? OR summary LIKE ? OR payload_json LIKE ?" for _ in usable)
        params: list[Any] = []
        for term in usable:
            like = f"%{term}%"
            params.extend([like, like, like])
        params.append(int(limit))
        return conn.execute(
            f"SELECT object_id, object_type, label, summary, payload_json, 0.0 AS rank FROM objects WHERE {clauses} LIMIT ?",
            params,
        ).fetchall()


def build_sqlite_sag_v2(kg_v2_root: str | Path, sqlite_path: str | Path, *, reset: bool = True) -> dict[str, Any]:
    store = JsonKGV2Store(kg_v2_root)
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp") if reset else path
    if not reset and path.exists():
        raise FileExistsError(f"SAG_v2 already exists and reset=False: {path}")
    conn = sqlite3.connect(build_path)
    report: dict[str, Any] | None = None
    try:
        _create_schema(conn)
        revision = kg_v2_graph_revision(kg_v2_root)
        source_revision = kg_v2_source_revision(kg_v2_root)
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            [
                ("graph_revision", revision),
                ("source_revision", source_revision),
                ("index_schema", SAG_V2_INDEX_SCHEMA),
            ],
        )
        object_index, object_type = _object_indexes(store)
        object_count = _insert_objects(conn, store)
        relation_count = _insert_relations(conn, store.relations)
        variant_count = _insert_variant_documents(conn, store, object_index, object_type)
        chunk_count, source_stats = _insert_source_chunks(
            conn,
            store,
            object_index,
            object_type,
            kg_v2_root=Path(kg_v2_root),
            asset_root=path.parent / "assets",
        )
        conn.commit()
        report = {
            "status": "built",
            "sqlite_path": str(path),
            "index_schema": SAG_V2_INDEX_SCHEMA,
            "object_count": object_count,
            "relation_count": relation_count,
            "variant_document_count": variant_count,
            "source_chunk_count": chunk_count,
            **source_stats,
            "graph_revision": revision,
            "source_revision": source_revision,
        }
    except Exception:
        conn.close()
        if reset and build_path.exists():
            build_path.unlink()
        raise
    finally:
        conn.close()
    if reset:
        build_path.replace(path)
    if report is None:
        raise RuntimeError(f"SAG_v2 build did not produce a report: {path}")
    return report


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE objects(object_id TEXT PRIMARY KEY, object_type TEXT NOT NULL, label TEXT NOT NULL, summary TEXT NOT NULL, payload_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE relations(src TEXT NOT NULL, dst TEXT NOT NULL, relation TEXT NOT NULL, payload_json TEXT NOT NULL)")
    conn.execute("CREATE INDEX idx_relations_src_relation ON relations(src, relation)")
    conn.execute("CREATE INDEX idx_relations_dst_relation ON relations(dst, relation)")
    conn.execute("CREATE INDEX idx_objects_type ON objects(object_type)")
    conn.execute("CREATE VIRTUAL TABLE object_fts USING fts5(object_id UNINDEXED, object_type UNINDEXED, text)")
    conn.execute("CREATE TABLE variant_documents(variant_id TEXT PRIMARY KEY, family_id TEXT NOT NULL, label TEXT NOT NULL, search_text TEXT NOT NULL)")
    conn.execute("CREATE VIRTUAL TABLE variant_fts USING fts5(variant_id UNINDEXED, text)")
    conn.execute("""CREATE TABLE source_chunks(
        chunk_id TEXT PRIMARY KEY, object_id TEXT NOT NULL, object_type TEXT NOT NULL,
        source_kind TEXT NOT NULL, source_label TEXT NOT NULL, text TEXT NOT NULL,
        source_offsets_json TEXT NOT NULL, media_refs_json TEXT NOT NULL,
        content_hash TEXT NOT NULL, source_file_hash TEXT NOT NULL,
        approved INTEGER NOT NULL,
        variant_ids_json TEXT NOT NULL, paths_json TEXT NOT NULL
    )""")
    conn.execute("CREATE INDEX idx_source_chunks_object ON source_chunks(object_id)")
    conn.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED, text)")


def _object_indexes(store: JsonKGV2Store) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    objects: dict[str, dict[str, Any]] = {}
    types: dict[str, str] = {}
    for obj_type, items in store.objects_by_type.items():
        pk = V2_PRIMARY_KEYS[obj_type]
        for item in items or []:
            object_id = str(item.get(pk) or "") if isinstance(item, dict) else ""
            if object_id:
                objects[object_id] = item
                types[object_id] = obj_type
    return objects, types


def _insert_objects(conn: sqlite3.Connection, store: JsonKGV2Store) -> int:
    count = 0
    for obj_type, items in store.objects_by_type.items():
        pk = V2_PRIMARY_KEYS[obj_type]
        for item in items or []:
            if not isinstance(item, dict):
                continue
            obj_id = str(item.get(pk) or "")
            if not obj_id:
                continue
            label = str(item.get("label") or item.get("title") or obj_id)
            summary = str(item.get("summary") or item.get("question") or item.get("why_required") or label)
            payload = json.dumps(item, ensure_ascii=False)
            conn.execute("INSERT INTO objects VALUES(?,?,?,?,?)", (obj_id, obj_type, label, summary, payload))
            conn.execute("INSERT INTO object_fts VALUES(?,?,?)", (obj_id, obj_type, _index_text(" ".join([label, summary, payload]))))
            count += 1
    return count


def _insert_relations(conn: sqlite3.Connection, relations: Iterable[dict[str, Any]]) -> int:
    count = 0
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        conn.execute("INSERT INTO relations VALUES(?,?,?,?)", (
            str(rel.get("from") or ""), str(rel.get("to") or ""),
            str(rel.get("relation") or ""), json.dumps(rel, ensure_ascii=False),
        ))
        count += 1
    return count


def _insert_variant_documents(
    conn: sqlite3.Connection,
    store: JsonKGV2Store,
    objects: dict[str, dict[str, Any]],
    types: dict[str, str],
) -> tuple[int, dict[str, int]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for object_id, item in objects.items():
        variant_id = str(item.get("variant_id") or "")
        family_id = str(item.get("family_id") or "")
        if variant_id and types.get(object_id) != "FaultVariant":
            by_variant[variant_id].append(item)
        if family_id and types.get(object_id) != "FaultFamily":
            by_family[family_id].append(item)
    families = {str(item.get("family_id") or ""): item for item in store.objects_by_type.get("FaultFamily", [])}
    count = 0
    for variant in store.objects_by_type.get("FaultVariant", []):
        variant_id = str(variant.get("variant_id") or "")
        family_id = str(variant.get("family_id") or "")
        family = families.get(family_id) or {}
        values: list[str] = []
        for item in [variant, family, *by_variant.get(variant_id, [])]:
            values.extend(_searchable_values(item))
        # Family-scoped information is deliberately capped and lower signal;
        # it helps recall without turning every sibling into the same document.
        for item in by_family.get(family_id, [])[:8]:
            if not str(item.get("variant_id") or ""):
                values.extend(_searchable_values(item)[:2])
        text = " ".join(_dedupe(values))
        label = str(variant.get("label") or variant_id)
        conn.execute("INSERT INTO variant_documents VALUES(?,?,?,?)", (variant_id, family_id, label, text))
        conn.execute("INSERT INTO variant_fts VALUES(?,?)", (variant_id, _index_text(text)))
        count += 1
    return count


def _insert_source_chunks(
    conn: sqlite3.Connection,
    store: JsonKGV2Store,
    objects: dict[str, dict[str, Any]],
    types: dict[str, str],
    *,
    kg_v2_root: Path,
    asset_root: Path,
) -> tuple[int, dict[str, int]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in store.relations:
        if isinstance(edge, dict):
            outgoing[str(edge.get("from") or "")].append(edge)
    documents = {str(item.get("document_id") or ""): item for item in store.objects_by_type.get("KnowledgeDocument", [])}
    count = 0
    for obj_type in ("EvidenceItem", "KnowledgeSection", "SourceCase"):
        pk = V2_PRIMARY_KEYS[obj_type]
        for item in store.objects_by_type.get(obj_type, []):
            object_id = str(item.get(pk) or "")
            text = str(item.get("summary") or "").strip()
            if not object_id or not text:
                continue
            document_id = str(item.get("document_id") or "")
            document = documents.get(document_id) or {}
            approved = item.get("approved") is not False and document.get("approved") is not False
            variant_paths = _variant_paths(object_id, objects, types, outgoing)
            variant_ids = _dedupe(str(path.get("variant_id") or "") for path in variant_paths)
            offsets = item.get("source_offsets") or []
            if not isinstance(offsets, list):
                offsets = [offsets]
            source_label = str(
                item.get("title") or item.get("heading") or document.get("title")
                or item.get("external_id") or object_id
            )
            source_kind = str(item.get("source_kind") or document.get("source_kind") or obj_type)
            chunk_id = f"chunk:{object_id}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            conn.execute("INSERT INTO source_chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                chunk_id, object_id, obj_type, source_kind, source_label, text,
                json.dumps(offsets, ensure_ascii=False), "[]", content_hash,
                str(document.get("content_hash") or ""), int(approved),
                json.dumps(variant_ids, ensure_ascii=False), json.dumps(variant_paths, ensure_ascii=False),
            ))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (chunk_id, _index_text(" ".join([source_label, text]))))
            count += 1
    source_count, source_stats = _insert_current_source_chunks(
        conn,
        kg_v2_root=kg_v2_root,
        store=store,
        objects=objects,
        types=types,
        outgoing=outgoing,
        asset_root=asset_root,
    )
    return count + source_count, source_stats


def _insert_current_source_chunks(
    conn: sqlite3.Connection,
    *,
    kg_v2_root: Path,
    store: JsonKGV2Store,
    objects: dict[str, dict[str, Any]],
    types: dict[str, str],
    outgoing: dict[str, list[dict[str, Any]]],
    asset_root: Path,
) -> tuple[int, dict[str, int]]:
    """Index chunks rebuilt from the current, hash-pinned source documents."""

    rows, stats = rebuild_source_chunks(
        kg_v2_root.parent.parent,
        store.objects_by_type.get("KnowledgeDocument", []),
        store.objects_by_type.get("KnowledgeSection", []),
        asset_root=asset_root,
        media_assets=store.objects_by_type.get("MediaAsset", []),
    )
    variant_section_ids = {
        object_id
        for object_id, edges in outgoing.items()
        if any(str(edge.get("relation") or "") == "describes_variant" for edge in edges)
    }
    directly_aligned_variant_sections: set[str] = set()
    count = 0
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        section_id = str(row.get("section_id") or "")
        direct_section_ids = [str(item) for item in row.get("direct_section_ids") or [] if str(item)]
        directly_aligned_variant_sections.update(variant_section_ids.intersection(direct_section_ids))
        # Only an explicitly aligned KnowledgeSection may affect Variant rank.
        variant_paths: list[dict[str, Any]] = []
        for direct_section_id in direct_section_ids:
            variant_paths.extend(_variant_paths(direct_section_id, objects, types, outgoing))
        variant_paths = _dedupe_path_rows(variant_paths)
        variant_ids = _dedupe(str(path.get("variant_id") or "") for path in variant_paths)
        object_id = section_id or str(row.get("document_id") or "")
        object_type = "KnowledgeSection" if section_id else "KnowledgeDocument"
        conn.execute("INSERT INTO source_chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            row["chunk_id"], object_id, object_type, row["source_kind"], row["source_label"], text,
            json.dumps(row["source_offsets"], ensure_ascii=False),
            json.dumps(row.get("media_refs") or [], ensure_ascii=False), row["content_hash"],
            row["source_file_hash"], int(bool(row["approved"])),
            json.dumps(variant_ids, ensure_ascii=False), json.dumps(variant_paths, ensure_ascii=False),
        ))
        conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (
            row["chunk_id"], _index_text(" ".join([row["source_path"], row["source_label"], text])),
        ))
        count += 1
    stats.update({
        "source_variant_section_count": len(variant_section_ids),
        "source_directly_aligned_variant_section_count": len(directly_aligned_variant_sections),
    })
    return count, stats


def _variant_paths(
    start_id: str,
    objects: dict[str, dict[str, Any]],
    types: dict[str, str],
    outgoing: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    queue: list[tuple[str, list[dict[str, str]]]] = [(start_id, [])]
    visited = {start_id}
    for _ in range(5):
        next_queue: list[tuple[str, list[dict[str, str]]]] = []
        for object_id, chain in queue:
            item = objects.get(object_id) or {}
            object_type = types.get(object_id, "")
            candidates: list[tuple[str, str]] = []
            if object_type == "FaultVariant":
                candidates.append((object_id, "object_is_variant"))
            variant_id = str(item.get("variant_id") or "")
            if variant_id:
                candidates.append((variant_id, f"{object_type}.variant_id"))
            family_id = str(item.get("family_id") or "")
            if family_id and not variant_id:
                for edge in outgoing.get(family_id, []):
                    if str(edge.get("relation") or "") == "has_variant":
                        candidates.append((str(edge.get("to") or ""), "has_variant"))
            for variant_id, relation in candidates:
                if variant_id.startswith("variant:"):
                    paths.append({
                        "seed_object_id": start_id,
                        "seed_object_type": types.get(start_id, ""),
                        "relation": relation,
                        "relation_chain": chain,
                        "variant_id": variant_id,
                    })
            for edge in outgoing.get(object_id, []):
                dst = str(edge.get("to") or "")
                relation = str(edge.get("relation") or "")
                if dst and dst not in visited and relation in {
                    "evidences", "supports", "describes_variant", "applicable_to", "has_section",
                    "has_step", "has_trace", "has_trace_step", "used_action", "outcome_of",
                    "branch_from", "branch_to", "has_outcome",
                }:
                    visited.add(dst)
                    next_queue.append((dst, [*chain, {"from": object_id, "relation": relation, "to": dst}]))
            for key in ("action_id", "trace_id", "from_trace_step_id", "to_trace_step_id", "source_case_id"):
                ref = str(item.get(key) or "")
                if ref and ref in objects and ref not in visited:
                    visited.add(ref)
                    next_queue.append((ref, [*chain, {"from": object_id, "relation": key, "to": ref}]))
        queue = next_queue
        if not queue:
            break
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        key = (str(path.get("variant_id") or ""), json.dumps(path.get("relation_chain") or [], sort_keys=True))
        unique[key] = path
    return list(unique.values())[:40]


def _searchable_values(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "label", "title", "summary", "question", "why_required", "instruction",
        "root_cause_summary", "error_phase", "subsystem", "equipment_type", "slot",
    ):
        value = str(item.get(key) or "").strip()
        if value:
            values.append(value)
    values.extend(str(value).strip() for value in item.get("keywords") or [] if str(value).strip())
    return values


def _chunk_relevance_score(
    query: str,
    text: str,
    *,
    source_label: str,
    raw_rank: Any,
) -> tuple[float, dict[str, Any]]:
    """Bound raw FTS rank and reward useful body coverage.

    FTS indexes overlapping Chinese 2/3/4-grams. A short exact reference can
    therefore accumulate a very large negative BM25 rank even though it has
    little answer content. This score keeps BM25 as a capped recall signal and
    makes query coverage plus information density the primary ranking inputs.
    """

    normalized_query = _semantic_key(query)
    normalized_text = _semantic_key(text)
    coverage = _query_coverage(query, text)
    exact_phrase = bool(normalized_query and normalized_query in normalized_text)
    raw_score = _rank_score(raw_rank)
    raw_component = min(3.0, math.log1p(raw_score))
    text_length = len(str(text or "").strip())
    information_factor = 0.65 + 0.35 * min(1.0, text_length / 180.0)
    reference_only = bool(
        text_length <= 100
        and re.search(r"(?m)^\s*(?:参考|参见|详见|见)\s*[：:]?", str(text or ""))
    )
    label_key = _semantic_key(re.sub(r"\.[a-z0-9]+$", "", source_label, flags=re.IGNORECASE))
    title_only = bool(text_length <= 50 and normalized_text and normalized_text == label_key)
    navigation_factor = 0.42 if reference_only else 0.82 if title_only else 1.0
    identifiers = list(analyze_query_scope(query).strong_identifiers)
    matched_identifiers = matched_strong_identifiers(
        query,
        f"{source_label} {text}",
    )
    identifier_ratio = (
        len(matched_identifiers) / len(identifiers) if identifiers else 0.0
    )
    score = (
        12.0 * coverage
        + (5.0 if exact_phrase else 0.0)
        + raw_component
        + 4.0 * identifier_ratio
    )
    score *= information_factor * navigation_factor
    return round(score, 4), {
        "query_coverage": round(coverage, 4),
        "exact_phrase": exact_phrase,
        "raw_bm25_score": raw_score,
        "raw_component": round(raw_component, 4),
        "information_factor": round(information_factor, 4),
        "reference_only": reference_only,
        "title_only": title_only,
        "matched_identifiers": matched_identifiers,
        "identifier_ratio": round(identifier_ratio, 4),
    }


def _query_coverage(query: str, text: str) -> float:
    units = _query_units(query)
    if not units:
        return 0.0
    normalized_text = _semantic_key(text)
    matched = sum(weight for unit, weight in units if unit in normalized_text)
    total = sum(weight for _unit, weight in units)
    return min(1.0, matched / total) if total else 0.0


def _query_units(value: str) -> list[tuple[str, float]]:
    normalized = _semantic_key(value)
    units: dict[str, float] = {}
    for identifier in _IDENTIFIER.findall(str(value or "").lower()):
        units[identifier] = max(units.get(identifier, 0.0), 2.0)
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            units[run] = 1.0
            continue
        for index in range(len(run) - 1):
            units[run[index:index + 2]] = 1.0
    return list(units.items())


def _semantic_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _document_id_for_chunk(conn: sqlite3.Connection, chunk: dict[str, Any]) -> str:
    object_id = str(chunk.get("object_id") or "")
    object_type = str(chunk.get("object_type") or "")
    if object_type == "KnowledgeDocument":
        return object_id
    if object_type == "KnowledgeSection":
        row = conn.execute(
            "SELECT payload_json FROM objects WHERE object_id=? AND object_type='KnowledgeSection'",
            (object_id,),
        ).fetchone()
        if row:
            try:
                return str(json.loads(str(row[0] or "{}")).get("document_id") or "")
            except (TypeError, ValueError, json.JSONDecodeError):
                return ""
    return ""


def _direct_document_matches(
    conn: sqlite3.Connection,
    query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    document_query = _document_lookup_query(query)
    query_scope = analyze_query_scope(query)
    document_identifier_matches: dict[str, set[str]] = {}
    for candidate in chunks:
        candidate_document_id = str(
            candidate.get("document_id")
            or candidate.get("object_id")
            or ""
        )
        if not candidate_document_id.startswith("knowledge-document:"):
            continue
        corpus = " ".join([
            str(candidate.get("source_label") or ""),
            str(candidate.get("source_document_title") or ""),
            str(candidate.get("text") or ""),
        ])
        document_identifier_matches.setdefault(
            candidate_document_id,
            set(),
        ).update(matched_strong_identifiers(query, corpus))
    matches: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or chunk.get("object_id") or "")
        # Some source manifests represent the document title as the first
        # KnowledgeSection instead of a standalone KnowledgeDocument chunk.
        # A section can therefore be the direct entry point to its document.
        # This remains true after the reviewed document has been linked to a
        # formal variant: ``variant_ids`` describe graph applicability, not
        # whether the user directly named the source. EvidenceItem/SourceCase
        # labels still cannot implicitly claim document ownership below.
        if not document_id or not document_id.startswith("knowledge-document:"):
            continue
        label = str(chunk.get("source_label") or "")
        source_title = (
            str(chunk.get("source_document_title") or "")
            or source_document_title(chunk)
        )
        if source_title and not _document_title_polarity_compatible(
            query,
            source_title,
        ):
            # Applicability belongs to the whole source domain, not only to
            # the internal heading that happened to match the query.  A
            # neutral section such as “工控机授权” must not make a source
            # explicitly scoped to a mutually exclusive branch eligible.
            continue
        if source_title and not subject_domain_compatible(query, source_title):
            continue
        candidate_titles = _dedupe([
            re.sub(r"\.[a-z0-9]+$", "", label, flags=re.IGNORECASE),
            source_title,
        ])
        compatible_titles = [
            title for title in candidate_titles
            if (
                _document_title_polarity_compatible(query, title)
                and subject_domain_compatible(query, title)
            )
        ]
        if not compatible_titles:
            continue
        scored_titles = []
        for title in compatible_titles:
            coverage = _query_coverage(document_query, title)
            reverse_coverage = _query_coverage(title, query)
            signals = title_match_signals(query, title)
            scored_titles.append((
                max(
                    float(signals.get("scope_score") or 0.0),
                    0.35 * coverage + 0.15 * reverse_coverage,
                ),
                title,
                coverage,
                reverse_coverage,
                signals,
            ))
        (
            title_strength,
            matched_title,
            coverage,
            reverse_coverage,
            match_signals,
        ) = max(scored_titles, key=lambda item: (item[0], len(item[1])))
        source_title_signals = (
            title_match_signals(query, source_title)
            if source_title
            else {}
        )
        source_title_is_primary_anchor = bool(
            source_title
            and source_title_signals.get("safe")
            and (
                float(source_title_signals.get("scope_score") or 0.0) >= 0.35
                or float(
                    source_title_signals.get("identifier_ratio") or 0.0
                ) > 0.0
            )
        )
        if (
            str(chunk.get("object_type") or "") == "KnowledgeSection"
            and "named_identifier" not in set(
                match_signals.get("match_reasons") or []
            )
            and (
                (
                    len(_semantic_key(matched_title)) <= 2
                    and coverage < 0.2
                )
                or (
                    float(match_signals.get("topic_strength") or 0.0) < 0.25
                    and float(match_signals.get("subject_strength") or 0.0) < 0.25
                    and coverage < 0.5
                )
            )
        ):
            # Structural headings such as “判断/操作/测试” describe a role
            # inside a document, not the document's subject.  They may remain
            # supporting chunks but cannot activate the whole source domain.
            continue
        # The document/section title must explain most of the user's query.
        # Reverse coverage alone is unsafe: a one-word section such as
        # "显卡" naturally occurs in a detailed CUDA incident, but that does
        # not make the whole hardware guide the user's directly named source.
        if coverage < 0.6 and not bool(match_signals.get("safe")):
            continue
        if (
            str(chunk.get("object_type") or "") != "KnowledgeDocument"
            and reverse_coverage < 0.5
            and not bool(match_signals.get("safe"))
        ):
            # A section heading may be a direct document entry only when the
            # query and heading substantially explain each other.  This keeps
            # exact headings such as “电脑卡顿” and “USB设备问题解决方案”,
            # while preventing a long internal heading that merely contains
            # “无法进入系统” from pulling in an entire unrelated handbook.
            continue
        item = {
            "document_id": document_id,
            # Trace/document scope should identify the source document rather
            # than whichever high-scoring internal heading happened to admit
            # it.  Keep that heading separately for audit.
            "source_label": source_title or label,
            "entry_source_label": label,
            "source_document_title": source_title,
            "matched_title": matched_title,
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "entry_object_id": str(chunk.get("object_id") or ""),
            "query_coverage": round(coverage, 4),
            "title_coverage": round(reverse_coverage, 4),
            "match_strength": round(title_strength, 4),
            "match_reasons": list(match_signals.get("reasons") or []),
            "retrieval_score": float(chunk.get("retrieval_score") or 0.0),
            "entry_object_type": str(chunk.get("object_type") or ""),
            "expansion_scope": (
                "section"
                if (
                    str(chunk.get("object_type") or "") == "KnowledgeSection"
                    and _semantic_key(matched_title)
                    != _semantic_key(source_title or label)
                    and not source_title_is_primary_anchor
                )
                else "document"
            ),
            "source_title_primary_anchor": source_title_is_primary_anchor,
            "document_identifier_ratio": round(
                len(document_identifier_matches.get(document_id, set()))
                / len(query_scope.strong_identifiers),
                4,
            ) if query_scope.strong_identifiers else 0.0,
        }
        current = matches.get(document_id)
        if current is None or (
            float(item["match_strength"]),
            float(item["query_coverage"]),
            float(item["title_coverage"]),
            float(item["retrieval_score"]),
        ) > (
            float(current["match_strength"]),
            float(current["query_coverage"]),
            float(current["title_coverage"]),
            float(current["retrieval_score"]),
        ):
            matches[document_id] = item
    ordered = sorted(
        matches.values(),
        key=lambda item: (
            -float(item.get("match_strength") or 0.0),
            -max(float(item["query_coverage"]), float(item["title_coverage"])),
            -float(item["retrieval_score"]),
        ),
    )
    exact = [
        item for item in ordered
        if float(item.get("query_coverage") or 0.0) >= 0.95
        and float(item.get("title_coverage") or 0.0) >= 0.8
    ]
    # Once the user has directly named a document/section, weaker partial
    # title matches must not turn unrelated handbooks into additional direct
    # documents.  Preserve genuinely duplicated titles, but collapse the
    # common export-copy pattern ``标题`` + ``标题 (1)``.  Those files are two
    # physical imports of one logical document and otherwise duplicate every
    # answer item and source attribution.
    selected = exact or ordered
    selected = [
        item for item in selected
        if (
            float(item.get("match_strength") or 0.0) >= 0.35
            or "named_identifier" in set(item.get("match_reasons") or [])
        )
    ]
    if query_scope.request_kind == "comparison_lookup":
        selected = [
            item for item in selected
            if (
                (
                    bool(query_scope.strong_identifiers)
                    and float(
                        title_match_signals(
                            query,
                            str(item.get("matched_title") or ""),
                        ).get("identifier_ratio") or 0.0
                    ) > 0.0
                )
                or (
                    # A comparison operand may be a descriptive product/tool
                    # name rather than a strong ASCII identifier.  Requiring
                    # every source title to repeat the one strong identifier
                    # discards the other side of comparisons such as a
                    # built-in diagnostic versus a named external tool.
                    float(item.get("match_strength") or 0.0) >= 0.35
                    and float(
                        title_match_signals(
                            query,
                            str(item.get("matched_title") or ""),
                        ).get("distinctive_subject_strength") or 0.0
                    ) > 0.0
                    and bool(
                        set(item.get("match_reasons") or [])
                        - {"named_identifier"}
                    )
                )
            )
        ]
    # A named tool/model lookup has a primary entity scope.  Once at least one
    # source title satisfies that scope, do not mix in documents admitted only
    # through generic words such as “系统”, “BIOS” or “设置”.
    if (
        query_scope.strong_identifiers
        and query_scope.request_kind != "comparison_lookup"
    ):
        descriptive = [
            item for item in selected
            if (
                bool(
                    set(item.get("match_reasons") or [])
                    - {"named_identifier"}
                )
                and float(item.get("match_strength") or 0.0) >= 0.35
                and float(
                    item.get("document_identifier_ratio") or 0.0
                ) > 0.0
                and (
                    float(
                        title_match_signals(
                            query,
                            str(item.get("matched_title") or ""),
                        ).get("distinctive_subject_strength") or 0.0
                    ) > 0.0
                    or float(
                        title_match_signals(
                            query,
                            str(item.get("matched_title") or ""),
                        ).get("requested_operation_coverage") or 0.0
                    ) > 0.0
                )
            )
        ]
        named = [
            item for item in selected
            if "named_identifier" in set(item.get("match_reasons") or [])
        ]
        # Concrete identifiers are hard evidence-scope constraints, but they
        # are not automatically the user's topic.  A source title that
        # explains the surrounding procedure (“键盘随机按键/无响应”,
        # “磁盘文件系统检测和修复”, “彻底卸载显卡驱动”) is more specific
        # than a generic handbook admitted only because its body/title
        # mentions USB, CHKDSK or DDU.
        selected = descriptive or named
    # A single coherent primary document is safer than expanding several
    # merely adjacent handbooks.  Keep near-ties (duplicate exports and true
    # sibling procedures), but drop weaker partial-title matches.
    if selected and query_scope.request_kind != "comparison_lookup":
        coherent_candidates = list(selected)
        if query_scope.strong_identifiers:
            # A broad but explicit identifier such as USB can occur in
            # several handbooks.  Named-identifier equality alone must not
            # turn every one of those handbooks into a primary document.
            # Prefer the title that also explains the surrounding topic,
            # while retaining genuine duplicate exports of that title.
            def topical_strength(item: dict[str, Any]) -> float:
                signals = title_match_signals(
                    query,
                    str(item.get("matched_title") or ""),
                )
                # Forward coverage answers “how much of the user's request
                # does this title explain?”.  Reverse coverage over-rewards a
                # tiny internal heading that only repeats the identifier.
                return (
                    float(item.get("query_coverage") or 0.0)
                    + 0.25 * float(signals.get("topic_strength") or 0.0)
                )

            best_topical = max(topical_strength(item) for item in selected)
            selected = [
                item for item in selected
                if best_topical - topical_strength(item) <= 0.02
            ]
        else:
            best_strength = max(
                float(item.get("match_strength") or 0.0) for item in selected
            )
            selected = [
                item for item in selected
                if best_strength - float(item.get("match_strength") or 0.0) <= 0.02
            ]
            # A procedure can explicitly request several ordered operations
            # whose authoritative instructions live in sibling documents.
            # Keep the strongest source for every uncovered operation instead
            # of collapsing “uninstall then install” to whichever title has
            # the higher lexical score.
            if query_scope.request_kind == "procedure_lookup":
                query_signals = title_match_signals(query, query)
                requested_operations = set(
                    query_signals.get("requested_operations") or []
                )
                if len(requested_operations) > 1:
                    covered_operations: set[str] = set()
                    for item in selected:
                        signals = title_match_signals(
                            query,
                            str(item.get("matched_title") or ""),
                        )
                        covered_operations.update(
                            requested_operations.intersection(
                                signals.get("title_operations") or []
                            )
                        )
                    for item in coherent_candidates:
                        if item in selected:
                            continue
                        signals = title_match_signals(
                            query,
                            str(item.get("matched_title") or ""),
                        )
                        item_operations = requested_operations.intersection(
                            signals.get("title_operations") or []
                        )
                        if (
                            item_operations - covered_operations
                            and float(item.get("match_strength") or 0.0) >= 0.35
                            and float(
                                signals.get("distinctive_subject_strength")
                                or 0.0
                            ) > 0.0
                        ):
                            selected.append(item)
                            covered_operations.update(item_operations)
                        if covered_operations >= requested_operations:
                            break
    return _collapse_export_alias_documents(conn, selected)[:3]


def _document_title_probe_chunks(
    conn: sqlite3.Connection,
    query: str,
) -> list[dict[str, Any]]:
    """Scan the small document/section title catalog independently of FTS.

    Long natural-language questions can fill the raw BM25 window with body
    paragraphs before an exact short title such as ``电脑卡顿`` is seen.  Title
    discovery is a bounded metadata lookup, not a JSON/full-content fallback,
    and the normal title safety rules still decide whether a probe is usable.
    """

    rows = conn.execute(
        """
        SELECT object_id, object_type, label, payload_json
        FROM objects
        WHERE object_type IN ('KnowledgeDocument', 'KnowledgeSection')
        """
    ).fetchall()
    document_labels: dict[str, str] = {}
    for row in rows:
        if str(row[1]) != "KnowledgeDocument":
            continue
        try:
            payload = json.loads(str(row[3] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        document_labels[str(row[0])] = str(
            payload.get("title")
            or payload.get("source_label")
            or row[2]
            or ""
        )
    probes: list[dict[str, Any]] = []
    for row in rows:
        object_id = str(row[0] or "")
        object_type = str(row[1] or "")
        label = str(row[2] or "")
        try:
            payload = json.loads(str(row[3] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        # Some overlays intentionally use the stable object id as the object
        # table label.  The source heading/title in the payload is the
        # human-readable retrieval anchor and must be used by the bounded
        # title catalog.
        payload_label = str(
            payload.get("heading")
            or payload.get("title")
            or payload.get("source_label")
            or ""
        ).strip()
        if payload_label:
            label = payload_label
        if not object_id or not label:
            continue
        if object_type == "KnowledgeDocument":
            document_id = object_id
        else:
            document_id = str(payload.get("document_id") or "")
        if not document_id.startswith("knowledge-document:"):
            continue
        score, components = _chunk_relevance_score(
            query,
            label,
            source_label=label,
            raw_rank=0.0,
        )
        probes.append({
            "chunk_id": f"title-probe:{object_id}",
            "object_id": object_id,
            "object_type": object_type,
            "source_kind": "document_title_catalog",
            "source_label": label,
            "source_document_title": document_labels.get(document_id, label),
            "text": label,
            "approved": True,
            "retrieval_score": score,
            "score_components": components,
            "document_id": document_id,
        })
    return probes


def _document_lookup_query(query: str) -> str:
    """Remove conversational request wording before matching a document title.

    A title such as ``电脑不开机排查`` fully explains the intent of
    ``电脑不开机，应该怎么排查？``.  Treating ``应该怎么`` as title content
    lowers query coverage and incorrectly routes the request through mixed
    retrieval.  This normalization is deliberately limited to request
    scaffolding; fault terms and requested operations remain untouched.
    """

    value = str(query or "")
    for phrase in (
        "我想知道", "麻烦帮我", "请帮我", "怎么才能", "应该怎么",
        "应当怎么", "该怎么", "要怎么", "怎么办", "如何", "怎样",
        "怎么", "请问", "麻烦", "帮我",
    ):
        value = value.replace(phrase, "")
    return value if _semantic_key(value) else str(query or "")


def _keep_navigation_parent_source_chunk(
    query: str,
    chunk: dict[str, Any],
) -> bool:
    """Keep parent facts without leaking a directory table into every answer."""

    text = str(chunk.get("text") or "")
    normalized = re.sub(r"\s+", "", text)
    if (
        len(normalized) <= 160
        and any(
            token in normalized
            for token in (
                "选择以下对应的方式",
                "根据情况选择以下方式",
                "参考以下文档",
                "以下子目录",
            )
        )
        and not re.search(r"https?://|下载地址|负责人|所有者|修改时间|创建时间", text)
    ):
        return False
    is_directory_table = bool(
        re.search(
            r"名称\s*\|\s*所有者\s*\|\s*修改时间(?:\s*\|\s*创建时间)?",
            text,
        )
    )
    if not is_directory_table:
        return True
    # Directory metadata is evidence only when the user requested ownership,
    # maintenance, chronology or the directory listing itself.  A procedure
    # lookup such as “如何进入安全模式” should contain the selected child
    # instructions, not the parent table.
    return any(
        token in str(query or "").lower()
        for token in (
            "谁维护", "谁负责", "负责人", "所有者", "维护人", "创建时间",
            "修改时间", "什么时候", "目录", "有哪些", "列表",
        )
    )


def _collapse_export_alias_documents(
    conn: sqlite3.Connection,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse numbered export copies only when an unsuffixed peer exists."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for document in documents:
        label = re.sub(
            r"\.[a-z0-9]+$",
            "",
            str(
                document.get("source_document_title")
                or document.get("matched_title")
                or document.get("source_label")
                or ""
            ),
            flags=re.IGNORECASE,
        ).strip()
        base = re.sub(r"\s*[（(]\d+[）)]\s*$", "", label).strip()
        key = _semantic_key(base)
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(document)

    hash_sets = _document_content_hash_sets(
        conn,
        {
            str(item.get("document_id") or "")
            for item in documents
            if str(item.get("document_id") or "")
        },
    )
    collapsed: list[dict[str, Any]] = []
    for key in group_order:
        peers = grouped[key]
        labels = [
            re.sub(
                r"\.[a-z0-9]+$",
                "",
                str(
                    item.get("source_document_title")
                    or item.get("matched_title")
                    or item.get("source_label")
                    or ""
                ),
                flags=re.IGNORECASE,
            ).strip()
            for item in peers
        ]
        has_unsuffixed = any(
            not re.search(r"\s*[（(]\d+[）)]\s*$", label)
            for label in labels
        )
        # Identical labels may still be independently maintained sources.
        # Collapse only when a numbered alias and its base title coexist.
        peer_hashes = [
            hash_sets.get(str(item.get("document_id") or ""), set())
            for item in peers
        ]
        similarities = [
            len(left & right) / min(len(left), len(right))
            for index, left in enumerate(peer_hashes)
            for right in peer_hashes[index + 1:]
            if left and right
        ]
        is_content_duplicate = bool(similarities) and min(similarities) >= 0.8
        if has_unsuffixed and len(set(labels)) > 1 and is_content_duplicate:
            selected = dict(max(
                peers,
                key=lambda item: (
                    float(item.get("retrieval_score") or 0.0),
                    float(item.get("query_coverage") or 0.0),
                    float(item.get("title_coverage") or 0.0),
                ),
            ))
            selected["original_source_label"] = selected.get("source_label")
            selected["source_label"] = re.sub(
                r"\s*[（(]\d+[）)](?=\.[a-z0-9]+$|$)",
                "",
                str(selected.get("source_label") or ""),
                flags=re.IGNORECASE,
            )
            collapsed.append(selected)
        else:
            collapsed.extend(peers)
    collapsed.sort(
        key=lambda item: (
            -max(
                float(item.get("query_coverage") or 0.0),
                float(item.get("title_coverage") or 0.0),
            ),
            -float(item.get("retrieval_score") or 0.0),
        ),
    )
    return collapsed


def _document_content_hash_sets(
    conn: sqlite3.Connection,
    document_ids: set[str],
) -> dict[str, set[str]]:
    result = {document_id: set() for document_id in document_ids}
    if not document_ids:
        return result
    rows = conn.execute(
        """
        SELECT s.object_id, s.object_type, s.content_hash, o.payload_json
        FROM source_chunks AS s
        LEFT JOIN objects AS o ON o.object_id = s.object_id
        WHERE s.approved = 1
        """
    ).fetchall()
    for row in rows:
        object_id = str(row[0] or "")
        object_type = str(row[1] or "")
        document_id = object_id if object_type == "KnowledgeDocument" else ""
        if object_type == "KnowledgeSection":
            try:
                document_id = str(
                    json.loads(str(row[3] or "{}")).get("document_id") or ""
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                document_id = ""
        if document_id in result and str(row[2] or ""):
            result[document_id].add(str(row[2]))
    return result


def _navigation_document_matches(
    conn: sqlite3.Connection,
    query: str,
    direct_documents: list[dict[str, Any]],
    *,
    max_automatic_children: int = 8,
    max_depth: int = 2,
    max_total_documents: int = 12,
    descendant_score_threshold: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve a bounded, query-guided navigation subgraph.

    The first hop of a complete, small directory is expanded in source order.
    Deeper hops must match the requested branch, preventing a navigation page
    from recursively pulling every linked handbook into the answer.
    """

    queue: list[dict[str, Any]] = []
    for root_order, parent in enumerate(direct_documents, start=1):
        parent_id = str(parent.get("document_id") or "")
        if not parent_id:
            continue
        navigation_label = str(
            parent.get("entry_source_label")
            or parent.get("source_label")
            or parent_id
        )
        queue.append({
            "document_id": parent_id,
            "source_label": str(parent.get("source_label") or parent_id),
            "navigation_depth": 0,
            "navigation_document_path": [parent_id],
            # The friendly source title is useful in the direct-match trace,
            # while a navigation path should retain the concrete source file
            # label so the path remains auditable and stable.
            "navigation_path": [navigation_label],
            "navigation_order_path": [root_order],
        })

    matches_by_document: dict[str, dict[str, Any]] = {}
    expanded_parents: set[str] = set()
    excluded: list[dict[str, Any]] = []
    while queue:
        parent = queue.pop(0)
        parent_id = str(parent.get("document_id") or "")
        parent_depth = int(parent.get("navigation_depth") or 0)
        if (
            not parent_id
            or parent_id in expanded_parents
            or parent_depth >= max(int(max_depth), 1)
        ):
            continue
        expanded_parents.add(parent_id)
        children, completeness = _navigation_children(conn, parent)
        navigation_complete = bool(completeness["complete"])
        if not navigation_complete:
            excluded.append({
                "document_id": parent_id,
                "source_label": str(parent.get("source_label") or ""),
                "navigation_depth": parent_depth,
                "navigation_path": list(parent.get("navigation_path") or []),
                "reason": "partial_navigation_document",
                "declared_link_count": completeness["declared_link_count"],
                "resolved_link_count": completeness["resolved_link_count"],
            })
            # A partially resolved directory is still useful: select only the
            # resolved children whose labels match the requested branch.  The
            # old all-or-nothing rule discarded valid child documents such as
            # “修复系统/修复引导” merely because unrelated links were absent.
            if not children:
                continue

        automatic_first_hop = (
            navigation_complete
            and parent_depth == 0
            and len(children) <= max_automatic_children
        )
        branch_scores = [
            _navigation_branch_score(
                query,
                [
                    str(child.get("source_label") or ""),
                    *[str(item) for item in child.get("link_texts") or []],
                ],
            )
            for child in children
        ]
        best_branch_score = max(branch_scores, default=0.0)
        requested_operations = set(
            title_match_signals(query, query).get("requested_operations") or []
        )
        child_operations = [
            requested_operations.intersection(
                title_match_signals(
                    query,
                    " ".join([
                        str(child.get("source_label") or ""),
                        *[
                            str(item)
                            for item in child.get("link_texts") or []
                        ],
                    ]),
                ).get("title_operations") or []
            )
            for child in children
        ]
        best_score_by_operation = {
            operation: max(
                (
                    score
                    for score, operations in zip(
                        branch_scores,
                        child_operations,
                    )
                    if operation in operations
                ),
                default=0.0,
            )
            for operation in requested_operations
        }
        facet_selected_indexes = {
            index
            for index, (score, operations) in enumerate(
                zip(branch_scores, child_operations)
            )
            if any(
                operation in operations
                and best_score_by_operation.get(operation, 0.0)
                >= descendant_score_threshold
                and best_score_by_operation[operation] - score <= 0.05
                for operation in requested_operations
            )
        }
        facet_guided_first_hop = bool(
            automatic_first_hop
            and len(requested_operations) > 1
            and len({
                operation
                for index in facet_selected_indexes
                for operation in child_operations[index]
            }) >= 2
        )
        query_guided_first_hop = bool(
            automatic_first_hop
            and analyze_query_scope(query).request_kind != "comparison_lookup"
            and best_branch_score >= descendant_score_threshold
        )
        for child_index, (child, branch_score) in enumerate(
            zip(children, branch_scores)
        ):
            labels = [
                str(child.get("source_label") or ""),
                *[str(item) for item in child.get("link_texts") or []],
            ]
            if (
                facet_guided_first_hop
                and child_index in facet_selected_indexes
            ):
                selection_reason = "query_facet_first_hop"
            elif facet_guided_first_hop:
                excluded.append({
                    "document_id": str(child.get("document_id") or ""),
                    "source_label": str(child.get("source_label") or ""),
                    "parent_document_id": parent_id,
                    "navigation_depth": parent_depth + 1,
                    "reason": "uncovered_first_hop_facet",
                    "branch_score": branch_score,
                })
                continue
            elif (
                query_guided_first_hop
                and best_branch_score - branch_score <= 0.05
            ):
                selection_reason = "query_guided_first_hop"
            elif query_guided_first_hop:
                excluded.append({
                    "document_id": str(child.get("document_id") or ""),
                    "source_label": str(child.get("source_label") or ""),
                    "parent_document_id": parent_id,
                    "navigation_depth": parent_depth + 1,
                    "reason": "weaker_first_hop_branch",
                    "branch_score": branch_score,
                    "best_branch_score": best_branch_score,
                })
                continue
            elif automatic_first_hop:
                selection_reason = "complete_first_hop_directory"
            elif branch_score >= descendant_score_threshold:
                selection_reason = (
                    "query_branch_match"
                    if navigation_complete
                    else "partial_directory_query_branch_match"
                )
            else:
                excluded.append({
                    "document_id": str(child.get("document_id") or ""),
                    "source_label": str(child.get("source_label") or ""),
                    "parent_document_id": parent_id,
                    "navigation_depth": parent_depth + 1,
                    "reason": "query_branch_mismatch",
                    "branch_score": branch_score,
                })
                continue

            document_id = str(child.get("document_id") or "")
            if not document_id:
                continue
            link_order = int(child.pop("link_order", 999999) or 999999)
            path = [
                *list(parent.get("navigation_path") or []),
                str(child.get("source_label") or document_id),
            ]
            document_path = [
                *list(parent.get("navigation_document_path") or []),
                document_id,
            ]
            order_path = [
                *list(parent.get("navigation_order_path") or []),
                link_order,
            ]
            path_item = {
                "document_ids": document_path,
                "source_labels": path,
                "link_orders": order_path,
            }
            candidate = {
                **child,
                "navigation_depth": parent_depth + 1,
                "navigation_path": path,
                "navigation_document_path": document_path,
                "navigation_order_path": order_path,
                "navigation_paths": [path_item],
                "branch_score": branch_score,
                "selection_reason": selection_reason,
            }
            current = matches_by_document.get(document_id)
            if current is None:
                if len(matches_by_document) >= max_total_documents:
                    excluded.append({
                        "document_id": document_id,
                        "source_label": str(child.get("source_label") or ""),
                        "reason": "navigation_document_budget",
                    })
                    continue
                matches_by_document[document_id] = candidate
                current = candidate
            else:
                current["navigation_paths"] = _dedupe_navigation_paths([
                    *list(current.get("navigation_paths") or []),
                    path_item,
                ])
                current["branch_score"] = max(
                    float(current.get("branch_score") or 0.0),
                    branch_score,
                )
                if tuple(order_path) < tuple(
                    current.get("navigation_order_path") or [999999]
                ):
                    preserved_paths = current["navigation_paths"]
                    preserved_score = current["branch_score"]
                    current.update(candidate)
                    current["navigation_paths"] = preserved_paths
                    current["branch_score"] = preserved_score
            if parent_depth + 1 < max_depth:
                queue.append(current)

    matches = sorted(
        matches_by_document.values(),
        key=lambda item: (
            tuple(item.get("navigation_order_path") or [999999]),
            int(item.get("navigation_depth") or 999999),
            str(item.get("document_id") or ""),
        ),
    )
    for navigation_order, item in enumerate(matches, start=1):
        item["navigation_order"] = navigation_order
    return matches, excluded


def _navigation_children(
    conn: sqlite3.Connection,
    parent: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_id = str(parent.get("document_id") or "")
    parent_row = conn.execute(
        "SELECT payload_json FROM objects WHERE object_id=? "
        "AND object_type='KnowledgeDocument'",
        (parent_id,),
    ).fetchone()
    parent_payload = json.loads(str(parent_row[0] or "{}")) if parent_row else {}
    declared_wiki_links = sum(
        bool(str(item.get("wiki_token") or ""))
        for item in parent_payload.get("source_links") or []
        if isinstance(item, dict)
    )
    rows = conn.execute(
        """
            SELECT r.dst, r.payload_json, o.label, o.payload_json
            FROM relations r
            JOIN objects o ON o.object_id = r.dst
            WHERE r.src=? AND r.relation='has_child_document'
              AND o.object_type='KnowledgeDocument'
        """,
        (parent_id,),
    ).fetchall()
    children: list[dict[str, Any]] = []
    resolved_link_occurrences = 0
    for row in rows:
        relation_payload = json.loads(str(row[1] or "{}"))
        document_payload = json.loads(str(row[3] or "{}"))
        link_orders = [
            int(value)
            for value in relation_payload.get("link_orders") or []
            if str(value).isdigit()
        ]
        resolved_link_occurrences += len(link_orders)
        children.append({
            "document_id": str(row[0] or ""),
            "source_label": str(row[2] or document_payload.get("title") or ""),
            "parent_document_id": parent_id,
            "parent_source_label": str(parent.get("source_label") or ""),
            "relation": "has_child_document",
            "link_texts": [
                str(value)
                for value in relation_payload.get("link_texts") or []
                if str(value).strip()
            ],
            "target_urls": list(relation_payload.get("target_urls") or []),
            "wiki_tokens": list(relation_payload.get("wiki_tokens") or []),
            "link_order": min(link_orders) if link_orders else 999999,
            "document_kind": str(document_payload.get("document_kind") or ""),
        })
    children.sort(key=lambda item: (
        int(item.get("link_order") or 999999),
        str(item.get("document_id") or ""),
    ))
    return children, {
        "complete": not (
            declared_wiki_links
            and resolved_link_occurrences < declared_wiki_links
        ),
        "declared_link_count": declared_wiki_links,
        "resolved_link_count": resolved_link_occurrences,
    }


def _navigation_branch_score(query: str, labels: Iterable[str]) -> float:
    query_key = _semantic_key(query)
    query_intents = {
        token for token in ("系统", "引导") if token in query_key
    }
    best = 0.0
    for raw_label in labels:
        label = re.sub(
            r"\.(?:docx?|pdf|md|txt|xlsx?|pptx?)$",
            "",
            str(raw_label or "").strip(),
            flags=re.IGNORECASE,
        )
        label_key = _semantic_key(label)
        if not label_key:
            continue
        label_intents = {
            token for token in ("系统", "引导") if token in label_key
        }
        if query_intents and label_intents and not query_intents.intersection(label_intents):
            continue
        label_bigrams = {
            label_key[index:index + 2]
            for index in range(max(len(label_key) - 1, 0))
        }
        query_bigrams = {
            query_key[index:index + 2]
            for index in range(max(len(query_key) - 1, 0))
        }
        bigram_recall = (
            len(label_bigrams.intersection(query_bigrams)) / len(label_bigrams)
            if label_bigrams else 0.0
        )
        score = max(
            bigram_recall,
            _query_coverage(label, query),
            1.0 if label_key in query_key else 0.0,
        )
        best = max(best, score)
    return round(min(best, 1.0), 4)


def _dedupe_navigation_paths(
    paths: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        if not isinstance(path, dict):
            continue
        key = tuple(str(value) for value in path.get("document_ids") or [])
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(path))
    return result


def _document_title_polarity_compatible(query: str, title: str) -> bool:
    query_key = _semantic_key(query)
    title_key = _semantic_key(title)
    negative = ("无法", "不能", "不可", "进不去", "未能")
    positive = ("可以进入", "能够进入", "能进入")
    query_negative = any(token in query_key for token in negative)
    title_negative = any(token in title_key for token in negative)
    query_positive = any(token in query_key for token in positive)
    title_positive = any(token in title_key for token in positive)
    if query_negative and title_positive and not title_negative:
        return False
    if query_positive and title_negative and not query_negative:
        return False
    return scope_polarity_compatible(query, title)


def _expand_document_chunks(
    conn: sqlite3.Connection,
    query: str,
    document_ids: list[str],
    *,
    max_documents: int = 12,
    asset_root: Path | None = None,
    section_scopes: dict[str, dict[str, set[str]]] | None = None,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    section_scopes = section_scopes or {}
    for document_id in _dedupe(document_ids)[:max(int(max_documents), 1)]:
        section_rows = conn.execute(
            "SELECT dst FROM relations WHERE src=? AND relation='has_section'",
            (document_id,),
        ).fetchall()
        object_ids = [document_id, *[str(row[0]) for row in section_rows if str(row[0])]]
        placeholders = ",".join("?" for _value in object_ids)
        if not placeholders:
            continue
        rows = conn.execute(
            f"""
                SELECT chunk_id, object_id, object_type, source_kind, source_label,
                       text, source_offsets_json, media_refs_json,
                       content_hash, source_file_hash, approved,
                       variant_ids_json, paths_json, 0.0 AS rank
                FROM source_chunks
                WHERE approved=1 AND object_id IN ({placeholders})
            """,
            object_ids,
        ).fetchall()
        for row in rows:
            raw = dict(row)
            section_scope = section_scopes.get(document_id)
            if (
                section_scope is not None
                and str(raw.get("chunk_id") or "")
                not in section_scope.get("chunk_ids", set())
                and _semantic_key(str(raw.get("source_label") or ""))
                not in section_scope.get("source_labels", set())
            ):
                # A direct hit on an internal FAQ/handbook heading owns only
                # that semantic section.  Expanding every sibling would turn
                # one relevant fact into an unrelated whole-book dump.  A
                # real document-title hit and an explicitly navigated child
                # still expand the complete document.
                continue
            text = str(raw.get("text") or "")
            score, components = _chunk_relevance_score(
                query,
                text,
                source_label=str(raw.get("source_label") or ""),
                raw_rank=raw.get("rank"),
            )
            expanded.append({
                **raw,
                "approved": bool(raw.get("approved")),
                "source_offsets": _json_list(raw.get("source_offsets_json", "[]")),
                "media_refs": _portable_media_refs(
                    _json_list(raw.get("media_refs_json", "[]")),
                    asset_root,
                ),
                "variant_ids": _json_list(raw.get("variant_ids_json", "[]")),
                "retrieval_paths": _json_list(raw.get("paths_json", "[]")),
                "raw_retrieval_score": 0.0,
                "retrieval_score": score,
                "score_components": components,
                "matched_terms": _matched_terms(query, text),
                "document_id": document_id,
                "document_expansion": True,
                "direct_expansion_scope": (
                    "section" if section_scope is not None else "document"
                ),
            })
    return expanded


def _portable_media_refs(
    values: Iterable[Any],
    asset_root: Path | None,
) -> list[Any]:
    if asset_root is None:
        return [dict(value) if isinstance(value, dict) else value for value in values]
    rebased: list[Any] = []
    for value in values:
        if not isinstance(value, dict):
            rebased.append(value)
            continue
        item = dict(value)
        source_path = str(item.get("source_path") or "").strip()
        if source_path:
            source_parts = Path(source_path).parts
            try:
                data_index = source_parts.index("data")
            except ValueError:
                data_index = -1
            if data_index >= 0:
                item["source_path"] = Path(
                    *source_parts[data_index:]
                ).as_posix()
        current = str(item.get("asset_path") or "").strip()
        if current:
            parts = Path(current).parts
            try:
                assets_index = len(parts) - 1 - list(reversed(parts)).index("assets")
            except ValueError:
                assets_index = -1
            if assets_index >= 0 and assets_index + 1 < len(parts):
                candidate = asset_root.joinpath(*parts[assets_index + 1 :])
                if candidate.exists():
                    item["asset_path"] = str(candidate.resolve())
        rebased.append(item)
    return rebased


def _fts_rows(
    conn: sqlite3.Connection,
    *,
    table: str,
    select_sql: str,
    query: str,
    limit: int,
) -> list[sqlite3.Row]:
    terms = _search_terms(query)
    if not terms:
        return []
    fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
    connector = "AND" if re.search(r"\bWHERE\b", select_sql, flags=re.IGNORECASE) else "WHERE"
    try:
        return conn.execute(
            f"{select_sql} {connector} {table} MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, int(limit)),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def kg_v2_graph_revision(kg_v2_root: str | Path) -> str:
    root = Path(kg_v2_root)
    digest = hashlib.sha256()
    paths = sorted((root / "objects").glob("*.json"))
    paths.extend(sorted((root / "relations").glob("*.json")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def kg_v2_source_revision(kg_v2_root: str | Path) -> str:
    """Hash the current bytes of every KnowledgeDocument source file."""

    root = Path(kg_v2_root)
    document_path = root / "objects" / "knowledge_documents.json"
    try:
        documents = json.loads(document_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        documents = []
    if not isinstance(documents, list):
        documents = []
    project_root = root.parent.parent
    digest = hashlib.sha256()
    for document in sorted(
        (item for item in documents if isinstance(item, dict)),
        key=lambda item: str(item.get("source_path") or ""),
    ):
        source_path = str(document.get("source_path") or "")
        if not source_path:
            continue
        digest.update(source_path.encode("utf-8"))
        path = project_root / source_path
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"MISSING")
    return digest.hexdigest()


def _search_terms(query: str) -> list[str]:
    text = str(query or "").strip().lower()
    terms: list[str] = list(_IDENTIFIER.findall(text))
    for run in _CJK_RUN.findall(text):
        if len(run) <= 6:
            terms.append(run)
        for size in (4, 3, 2):
            terms.extend(run[index:index + size] for index in range(len(run) - size + 1))
    # Long and distinctive terms first; two-character terms are a recall net.
    return [term for term in dict.fromkeys(terms) if term not in _GENERIC_TERMS][:96]


def _index_text(text: str) -> str:
    normalized = str(text or "").lower()
    return " ".join(_dedupe([normalized, *_search_terms(normalized)]))


def _matched_terms(query: str, text: str) -> list[str]:
    lowered = str(text or "").lower()
    return [term for term in _search_terms(query) if term in lowered][:40]


def _rank_score(value: Any) -> float:
    try:
        rank = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, -rank), 4)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _lookup_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _dedupe_path_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result[:80]


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
