"""SQLite-backed SAG shadow store.

This module keeps the SAG experiment behind the existing KGStore boundary.  The
SQLite schema stores event/entity incidence rows; query-time joins across shared
entities form the dynamic hyperedges used for expansion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from debug_agent_system.core.contracts import Candidate, CheckNode, LockedSubgraph, SolutionNode
from debug_agent_system.knowledge.json_store import JsonKGStore

_WORD = re.compile(r"[a-zA-Z0-9_.:+#-]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_OX_CODE = re.compile(r"\b[oO]x([0-9a-fA-F]{6,8})\b")
_TIME_TOKEN = re.compile(r"^\d{1,2}:\d{2}$")
_SENTENCE_SPLIT = re.compile(r"[\n。；;]+")
_DOMAIN_PHRASES = (
    "CAD导入", "解析失败", "尺寸过大", "导入后没显示", "角度识别", "自动对齐",
    "BIOS", "CMOS", "CLR_CMOS", "纽扣电池", "显示输出", "蓝屏", "自动修复",
    "开机页面", "启动修复", "Windows启动", "Windows 启动", "停留开机页面",
    "MEMORY.DMP", "DMP", "BugCheck", "复判站", "AOI", "DLOG", "相机", "光源",
    "工控机", "内存", "硬盘", "主板", "电源", "网卡", "采集卡", "误报", "漏检",
    "自动退出", "软件自动退出", "异常退出", "主程序退出", "闪退",
    "初始化失败", "相机连接异常", "相机IP", "检查相机IP",
)
_DEFAULT_QUERY_STOPWORDS = {
    "客户", "客户反馈", "反馈", "问题", "之前", "目前", "目前问题", "当前", "当前问题",
    "已经", "已经做过", "做过", "做过操作", "操作", "怎么", "怎么办", "怎么排查",
    "排查", "处理", "需要", "请给出", "这个", "那个", "现象", "一会", "一直",
    "题之前", "之前是", "目前问", "问题之", "问题之前", "过操作", "发生", "发生时间",
    "时间", "左右", "现场", "现场反馈", "设备运行", "运行中",
}
_WEAK_QUERY_MARKERS = ("问题之前", "目前问题", "已经做过", "做过操作", "客户反馈")
_ACTION_TRIED_MARKERS = ("已", "已经", "做过", "试过", "尝试", "操作", "断电", "放电", "长按", "重启", "更换", "修复")
_ACTION_CONTEXT_MARKERS = ("已", "已经", "做过", "试过", "尝试")
_TEMPORAL_MARKERS = ("之前", "目前", "当前", "最近", "曾经", "一会", "偶发", "持续")
_DEVICE_TERMS = ("工控机", "相机", "显示器", "主机", "主板", "内存", "硬盘", "电源", "网卡", "CAD", "BIOS", "CMOS", "复判站")
_GENERIC_DEVICE_TERMS = {"设备", "AOI", "aoi"}
_HIGH_DEGREE_DEVICE_TERMS = {"设备", "aoi"}
_SYMPTOM_TERMS = (
    "蓝屏", "自动修复", "开机页面", "启动", "不显示", "无显示", "显示修复",
    "停留", "循环", "失败", "异常", "报错", "无法", "不能", "黑屏", "卡死",
    "闪退", "自动退出", "异常退出", "突然关闭", "主程序退出", "软件自动退出",
    "漏检", "误报", "尺寸过大", "解析失败", "风扇不转", "无通电",
    "不亮", "无反应", "无通电反应", "不开机", "无法开机", "开机无反应",
)
_EXACT_SYMPTOM_TERMS = {item.lower() for item in _SYMPTOM_TERMS}
_WEAK_NGRAM_FRAGMENTS = (
    "客户", "反馈", "问题", "排查", "怎么", "应该", "处理", "发生", "时间",
    "左右", "现场", "目前", "之前", "已经", "做过", "操作", "需要", "补充",
)

_FAULT_FAMILIES = {
    "industrial_pc_blue_screen": {
        "canonical": "err:industrial-pc-blue-screen",
        "members": {"err:industrial-pc-blue-screen", "err:industrial-pc-blue-screen-crash", "err:system-bsod-restart"},
    },
    "industrial_pc_no_boot": {
        "canonical": "err:industrial-pc-no-boot",
        "members": {
            "err:industrial-pc-no-boot",
            "err:industrial-pc-power-failure",
            "err:industrial-pc-power-on-no-response",
            "err:review-pc-power-on-fan-not-spin",
            "err:device-power-off",
        },
    },
    "cad_import": {
        "canonical": "err:cad-import-failure",
        "members": {
            "err:cad-import-failure",
            "err:cad-import-error",
            "err:cad-angle-recognition-failure",
            "err:cad-import-failure-insufficient-component-boxes",
        },
    },
    "camera_capture": {
        "canonical": "err:camera-capture-failure",
        "members": {
            "err:camera-capture-failure",
            "err:2d-camera-intermittent-capture-failure",
            "err:camera-capture-timeout-failure",
            "err:camera-capture-failure-frequent",
            "err:camera-trigger-timeout-no-capture",
            "err:camera-capture-returned-empty-image",
            "err:camera-capture-failure-device-paused",
            "err:frequent-capture-failure",
        },
    },
}
_FAMILY_BY_ERROR_ID = {
    member: family
    for family, config in _FAULT_FAMILIES.items()
    for member in config["members"]
}

TIER_TRUST = {"A": 0.95, "B": 0.75, "C": 0.45, "D": 0.2}
TIER_SCORE = {"A": 2.0, "B": 1.2, "C": 0.55, "D": 0.15}
QUERY_ROLE_WEIGHT = {"error_code": 3.0, "symptom": 2.2, "device": 1.5, "domain": 1.4, "temporal_context": 0.25, "action_tried": 0.2}
EXECUTABLE_RELATIONS = {"has_check", "next", "resolved_by", "requires_info"}


def build_sqlite_sag(
    out_path: str | Path,
    *,
    raw_root: str | Path = "data/raw/aoi_debug_agent_sources",
    kg_root: str | Path = "data/kg",
    kg_v2_root: str | Path | None = None,
    w1_root: str | Path | None = "data/results/w1_full_20260703_061455",
    reset: bool = True,
) -> dict[str, Any]:
    """Build the SQLite SAG shadow store and return a report."""

    builder = SqliteSAGBuilder(out_path, reset=reset)
    builder.initialize()
    builder.import_raw(raw_root)
    builder.import_json_kg(kg_root)
    if kg_v2_root:
        builder.import_kg_v2(kg_v2_root, legacy_kg_root=kg_root)
    if w1_root:
        builder.import_w1(w1_root)
    return builder.report()


class SqliteSAGBuilder:
    def __init__(self, out_path: str | Path, *, reset: bool = True) -> None:
        self.path = Path(out_path)
        self.reset = reset
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._stable_to_event: dict[str, str] = {}
        self._kg_v2_stats = {
            "materialized_root": "",
            "mapped_variants": 0,
            "imported_variant_events": 0,
            "imported_check_events": 0,
            "imported_solution_events": 0,
            "mapping_review_links": 0,
        }
        self._raw_manifest_stats = {
            "manifest_path": "",
            "source_documents": 0,
        }

    def initialize(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def import_raw(self, raw_root: str | Path) -> None:
        root = Path(raw_root)
        if not root.exists():
            return
        manifest_path = root / "kg_v2_source_manifest.json"
        if manifest_path.exists():
            self._import_raw_manifest(manifest_path)
        seen_text: set[str] = set()
        for path in sorted((root / "chunks").glob("debug_chunks*.json")):
            rows = _read_json(path, [])
            if not isinstance(rows, list):
                continue
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                text = _clean_text(row.get("text"))
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                source = str(meta.get("source") or "chunks")
                title = str(meta.get("title") or meta.get("section_num") or f"{path.stem}:{idx}")
                tier = _source_tier(source)
                doc_id = _id("doc", source, title)
                self._insert_document(
                    doc_id,
                    source_type=source,
                    title=title,
                    path=str(path),
                    source_tier=tier,
                    payload=meta,
                )
                chunk_id = _id("chunk", str(path), str(idx), title, text[:80])
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO source_chunks
                    (chunk_id, doc_id, source_type, title, text, source_tier, trust_level, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, source, title, text, tier, TIER_TRUST[tier], _json(meta)),
                )
                event_id = f"event:source_chunk:{chunk_id}"
                self._insert_event(
                    event_id,
                    "source_chunk",
                    stable_id=chunk_id,
                    label=title,
                    text=text,
                    source_tier=tier,
                    needs_review=0 if tier == "A" else 1,
                    source_ref=chunk_id,
                    payload={"metadata": meta},
                )
        self.conn.commit()

    def _import_raw_manifest(self, manifest_path: Path) -> None:
        payload = _read_json(manifest_path, {})
        if not isinstance(payload, dict):
            return
        rows = payload.get("sources") or []
        if not isinstance(rows, list):
            return
        self._raw_manifest_stats["manifest_path"] = str(manifest_path)
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("source_id") or "").strip()
            if not title:
                continue
            source_type = str(row.get("source_kind") or row.get("parser") or "manifest")
            source_ref = str(
                row.get("fetch_json_path")
                or row.get("fallback_docx_path")
                or row.get("markdown_path")
                or row.get("source_path")
                or row.get("source_id")
                or title
            )
            tier = _source_tier(source_type)
            doc_id = _id("manifest_doc", str(row.get("source_id") or title), title)
            self._insert_document(
                doc_id,
                source_type=source_type,
                title=title,
                path=source_ref,
                source_tier=tier,
                payload=row,
            )
            event_id = f"event:source_manifest:{doc_id}"
            text = _clean_text(
                " ".join(
                    part for part in (
                        title,
                        source_type,
                        str(row.get("parser") or ""),
                        str(row.get("section_strategy") or ""),
                        Path(source_ref).name if source_ref else "",
                    )
                    if part
                )
            )
            self._insert_event(
                event_id,
                "source_document",
                doc_id,
                title,
                text or title,
                tier,
                0 if bool(row.get("enabled", True)) else 1,
                source_ref,
                row,
            )
            self._raw_manifest_stats["source_documents"] += 1
        self.conn.commit()

    def import_json_kg(self, kg_root: str | Path) -> None:
        root = Path(kg_root)
        if not root.exists():
            return
        store = JsonKGStore(root)
        for node in store.errors:
            error_id = str(node.get("error_id") or "")
            if not error_id:
                continue
            event_id = f"event:error:{error_id}"
            tier = _source_tier(str(node.get("source") or node.get("source_type") or "KG"))
            text = _node_text(node)
            self._insert_event(event_id, "fault_target", error_id, _label(node, error_id), text, tier, 0, error_id, node)
            self._insert_alias(error_id, "Error", event_id, "stable")
        for node in store.checks:
            check_id = str(node.get("check_id") or "")
            if not check_id:
                continue
            event_id = f"event:check:{check_id}"
            tier = _source_tier(str(node.get("source") or node.get("source_type") or "KG"))
            self._insert_event(event_id, "check", check_id, _label(node, check_id), _node_text(node), tier, 0, check_id, node)
            self._insert_alias(check_id, "DiagnosticCheck", event_id, "stable")
        for node in store.solutions:
            solution_id = str(node.get("solution_id") or "")
            if not solution_id:
                continue
            event_id = f"event:solution:{solution_id}"
            tier = _source_tier(str(node.get("source") or node.get("source_type") or "KG"))
            self._insert_event(event_id, "solution", solution_id, _label(node, solution_id), _node_text(node), tier, 0, solution_id, node)
            self._insert_alias(solution_id, "Solution", event_id, "stable")
        for event_type, rows, key, stable_type in (
            ("diagnostic_trace", store.traces, "trace_id", "DiagnosticTrace"),
            ("diagnostic_outcome", store.outcomes, "outcome_id", "DiagnosticOutcome"),
            ("diagnostic_policy", store.policies, "policy_id", "DiagnosticPolicy"),
        ):
            for node in rows:
                stable_id = str(node.get(key) or "")
                if not stable_id:
                    continue
                event_id = f"event:{event_type}:{stable_id}"
                self._insert_event(event_id, event_type, stable_id, _label(node, stable_id), _node_text(node), "B", 0, stable_id, node)
                self._insert_alias(stable_id, stable_type, event_id, "stable")
        for edge in store.edges if isinstance(store.edges, list) else []:
            if not isinstance(edge, dict):
                continue
            from_event = self._stable_to_event.get(str(edge.get("from") or ""))
            to_event = self._stable_to_event.get(str(edge.get("to") or ""))
            relation = str(edge.get("relation") or "")
            if not from_event or not to_event or not relation:
                continue
            self._insert_link(
                from_event,
                to_event,
                relation,
                condition=str(edge.get("condition") or ""),
                source_tier=_edge_tier(relation),
                confidence=0.95,
                needs_review=0,
                payload=edge,
            )
        self.conn.commit()

    def import_kg_v2(self, kg_v2_root: str | Path, *, legacy_kg_root: str | Path) -> None:
        root = Path(kg_v2_root)
        materialized_root = root / "materialized_execution" if (root / "materialized_execution").exists() else root
        if not (materialized_root / "instances" / "errors" / "errors.json").exists():
            return
        legacy_root = Path(legacy_kg_root)
        if not legacy_root.exists():
            return
        legacy_store = JsonKGStore(legacy_root)
        v2_store = JsonKGStore(materialized_root)
        self._kg_v2_stats["materialized_root"] = str(materialized_root)
        for node in v2_store.errors:
            if not isinstance(node, dict) or str(node.get("entry_role") or "") != "case_variant":
                continue
            mapping = self._map_v2_error_to_legacy(node, legacy_store)
            if mapping is None:
                continue
            legacy_error_id = str(mapping["error_id"])
            legacy_event_id = self._stable_to_event.get(legacy_error_id)
            if not legacy_event_id:
                continue
            v2_error_id = str(node.get("error_id") or "")
            if not v2_error_id:
                continue
            payload = dict(node)
            payload["_mapped_legacy_error_id"] = legacy_error_id
            payload["_mapping_score"] = float(mapping["score"])
            payload["_mapping_margin"] = float(mapping["margin"])
            payload["_mapping_confidence"] = float(mapping["confidence"])
            needs_review = 0 if float(mapping["confidence"]) >= 0.8 else 1
            error_event_id = f"event:kgv2_error:{v2_error_id}"
            self._insert_event(
                error_event_id,
                "v2_fault_variant",
                v2_error_id,
                _label(node, v2_error_id),
                _node_text(node),
                "B",
                needs_review,
                v2_error_id,
                payload,
            )
            self._insert_alias(v2_error_id, "V2Error", error_event_id, "stable")
            self._insert_link(
                error_event_id,
                legacy_event_id,
                "maps_to_error",
                condition="",
                source_tier="B",
                confidence=float(mapping["confidence"]),
                needs_review=needs_review,
                payload=payload,
            )
            self._kg_v2_stats["mapped_variants"] += 1
            self._kg_v2_stats["imported_variant_events"] += 1
            if needs_review:
                self._kg_v2_stats["mapping_review_links"] += 1
            self._import_kg_v2_subgraph(v2_store, v2_error_id, error_event_id, float(mapping["confidence"]))
        self.conn.commit()

    def _import_kg_v2_subgraph(self, v2_store: JsonKGStore, v2_error_id: str, parent_event_id: str, confidence: float) -> None:
        try:
            subgraph = v2_store.load_locked_subgraph(v2_error_id)
        except KeyError:
            return
        needs_review = 0 if confidence >= 0.8 else 1
        check_event_ids: dict[str, str] = {}
        for check in subgraph.checks:
            event_id = f"event:kgv2_check:{check.check_id}"
            payload = dict(check.payload)
            payload.update({
                "check_id": check.check_id,
                "label": check.label,
                "how_to_check": check.how_to_check,
                "step_order": check.step_order,
                "destructive": check.destructive,
            })
            self._insert_event(
                event_id,
                "v2_check_hint",
                check.check_id,
                check.label,
                check.how_to_check,
                "B",
                needs_review,
                check.check_id,
                payload,
            )
            self._insert_alias(check.check_id, "V2Check", event_id, "stable")
            check_event_ids[check.check_id] = event_id
            self._kg_v2_stats["imported_check_events"] += 1
            if int(check.payload.get("_graph_depth") or 0) == 0:
                self._insert_link(
                    parent_event_id,
                    event_id,
                    "has_check",
                    condition=str(check.payload.get("_incoming_condition") or ""),
                    source_tier="B",
                    confidence=confidence,
                    needs_review=needs_review,
                    payload=payload,
                )
        for from_check_id, edges in subgraph.next_edges_by_check.items():
            from_event_id = check_event_ids.get(from_check_id)
            if not from_event_id:
                continue
            for edge in edges or []:
                to_event_id = check_event_ids.get(str(edge.get("to_check_id") or ""))
                if not to_event_id:
                    continue
                self._insert_link(
                    from_event_id,
                    to_event_id,
                    "next",
                    condition=str(edge.get("condition") or ""),
                    source_tier="B",
                    confidence=confidence,
                    needs_review=needs_review,
                    payload=edge if isinstance(edge, dict) else {},
                )
        for check_id, solutions in subgraph.solutions_by_check.items():
            from_event_id = check_event_ids.get(check_id)
            if not from_event_id:
                continue
            for solution in solutions:
                event_id = f"event:kgv2_solution:{solution.solution_id}"
                payload = dict(solution.payload)
                payload.update({
                    "solution_id": solution.solution_id,
                    "content": solution.content,
                    "evidence_level": solution.evidence_level,
                    "destructive": solution.destructive,
                })
                self._insert_event(
                    event_id,
                    "v2_solution_hint",
                    solution.solution_id,
                    solution.solution_id,
                    solution.content,
                    "B",
                    needs_review,
                    solution.solution_id,
                    payload,
                )
                self._insert_alias(solution.solution_id, "V2Solution", event_id, "stable")
                self._insert_link(
                    from_event_id,
                    event_id,
                    "resolved_by",
                    condition=str(solution.payload.get("_edge_condition") or ""),
                    source_tier="B",
                    confidence=confidence,
                    needs_review=needs_review,
                    payload=payload,
                )
                self._kg_v2_stats["imported_solution_events"] += 1

    def _map_v2_error_to_legacy(self, node: dict[str, Any], legacy_store: JsonKGStore) -> dict[str, Any] | None:
        query = " ".join(
            str(node.get(key) or "")
            for key in ("label", "symptom", "category", "subsystem", "scenario")
        ).strip()
        if not query:
            return None
        candidates = legacy_store.search_errors(query, limit=3)
        if not candidates:
            return None
        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        top_score = float(top.score)
        margin = top_score - float(second.score if second is not None else 0.0)
        if top_score < 6.0 or margin < 1.0:
            return None
        confidence = 0.75
        if top_score >= 14.0 and margin >= 3.0:
            confidence = 0.95
        elif top_score >= 9.0 and margin >= 2.0:
            confidence = 0.85
        return {
            "error_id": top.error_id,
            "score": round(top_score, 4),
            "margin": round(margin, 4),
            "confidence": confidence,
        }

    def import_w1(self, w1_root: str | Path) -> None:
        root = Path(w1_root)
        w1_dir = root / "w1" if (root / "w1").exists() else root
        manifest = _read_json(w1_dir / "run_manifest.json", {})
        if not isinstance(manifest, dict):
            manifest = {}
        run_id = str(manifest.get("run_id") or root.name)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO source_runs
            (run_id, source, import_root, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, str(manifest.get("source") or "w1_full"), str(manifest.get("import_root") or ""), _json(manifest)),
        )
        messages_path = w1_dir / "messages.jsonl"
        if messages_path.exists():
            with messages_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._insert_w1_message(run_id, msg)
        summaries_path = w1_dir / "thread_summaries.json"
        summaries = _read_json(summaries_path, [])
        if isinstance(summaries, list):
            for summary in summaries:
                if isinstance(summary, dict):
                    self._insert_thread(run_id, summary)
        episodes_path = w1_dir / "episodes.json"
        episodes = _read_json(episodes_path, [])
        if isinstance(episodes, list):
            for episode in episodes:
                if isinstance(episode, dict):
                    self._insert_episode(run_id, episode)
        self.conn.commit()

    def report(self) -> dict[str, Any]:
        self.conn.commit()
        counts = {
            name: int(self.conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in (
                "source_runs", "source_documents", "source_chunks", "source_threads",
                "source_messages", "source_episodes", "events", "entities",
                "event_entities", "event_links", "id_aliases",
            )
        }
        tier_counts = {
            row["source_tier"]: row["n"]
            for row in self.conn.execute("SELECT source_tier, COUNT(*) AS n FROM events GROUP BY source_tier")
        }
        manifest_counts, manifest_completeness = self._manifest_counts()
        w1_counts = {
            "messages": int(manifest_counts.get("messages", counts["source_messages"])),
            "episodes": int(manifest_counts.get("episodes", counts["source_episodes"])),
            "complete": int(manifest_completeness.get("complete", self.conn.execute("SELECT COUNT(*) FROM source_episodes WHERE completeness = 'complete'").fetchone()[0])),
            "partial": int(manifest_completeness.get("partial", self.conn.execute("SELECT COUNT(*) FROM source_episodes WHERE completeness = 'partial'").fetchone()[0])),
        }
        low_confidence_links = int(
            self.conn.execute("SELECT COUNT(*) FROM event_links WHERE needs_review = 1 OR confidence < 0.7").fetchone()[0]
        )
        old_id_coverage = int(
            self.conn.execute("SELECT COUNT(*) FROM id_aliases WHERE alias_type = 'stable'").fetchone()[0]
        )
        return {
            "schema_version": "debug_agent_system.kg_sag.build_report.v1",
            "sqlite_path": str(self.path),
            "counts": counts,
            "tier_counts": tier_counts,
            "w1_counts": w1_counts,
            "raw_manifest": dict(self._raw_manifest_stats),
            "kg_v2_stats": dict(self._kg_v2_stats),
            "old_id_coverage": old_id_coverage,
            "low_confidence_links": low_confidence_links,
            "llm_extraction": {"enabled": False, "reason": "v1 deterministic shadow builder"},
        }

    def _manifest_counts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        for row in self.conn.execute("SELECT payload_json FROM source_runs LIMIT 1"):
            payload = _loads(row["payload_json"])
            counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
            completeness = payload.get("episode_completeness") if isinstance(payload.get("episode_completeness"), dict) else {}
            return counts, completeness
        return {}, {}

    def _insert_w1_message(self, run_id: str, msg: dict[str, Any]) -> None:
        message_id = str(msg.get("message_id") or "")
        if not message_id:
            return
        text = _clean_text(msg.get("text") or (msg.get("raw") or {}).get("content"))
        thread_id = str(msg.get("thread_id") or "")
        self.conn.execute(
            """
            INSERT OR IGNORE INTO source_messages
            (message_id, source_run_id, thread_id, create_time, msg_type, text, source_tier, trust_level, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, 'D', ?, ?)
            """,
            (message_id, run_id, thread_id, str(msg.get("create_time") or ""), str(msg.get("msg_type") or ""), text, TIER_TRUST["D"], _json(msg)),
        )
        if text and _is_debug_like(text):
            event_id = f"event:w1_message:{message_id}"
            self._insert_event(event_id, "retrieval_evidence", message_id, text[:80], text, "D", 1, message_id, msg)

    def _insert_thread(self, run_id: str, summary: dict[str, Any]) -> None:
        thread_id = str(summary.get("thread_id") or "")
        if not thread_id:
            return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO source_threads
            (thread_id, source_run_id, start_time, end_time, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, run_id, str(summary.get("start_time") or ""), str(summary.get("end_time") or ""), _json(summary)),
        )

    def _insert_episode(self, run_id: str, episode: dict[str, Any]) -> None:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            return
        completeness = str(episode.get("completeness") or "partial")
        tier = "C" if completeness == "complete" else "D"
        self.conn.execute(
            """
            INSERT OR IGNORE INTO source_episodes
            (episode_id, source_run_id, thread_id, completeness, source_tier, trust_level, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (episode_id, run_id, str(episode.get("thread_id") or ""), completeness, tier, TIER_TRUST[tier], _json(episode)),
        )
        text = _episode_text(episode)
        if not text:
            return
        event_type = "diagnostic_trace" if completeness == "complete" else "symptom_variant"
        event_id = f"event:w1_episode:{episode_id}"
        self._insert_event(event_id, event_type, episode_id, text[:100], text, tier, 1, episode_id, episode)
        if completeness == "complete" and episode.get("resolution_messages"):
            out_id = f"event:w1_outcome:{episode_id}"
            resolution_text = " ".join(_message_texts(episode.get("resolution_messages") or []))
            self._insert_event(out_id, "diagnostic_outcome", episode_id + ":outcome", resolution_text[:100], resolution_text, tier, 1, episode_id, episode)

    def _insert_document(
        self,
        doc_id: str,
        *,
        source_type: str,
        title: str,
        path: str,
        source_tier: str,
        payload: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO source_documents
            (doc_id, source_type, title, path, source_tier, trust_level, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, source_type, title, path, source_tier, TIER_TRUST[source_tier], _json(payload)),
        )

    def _insert_event(
        self,
        event_id: str,
        event_type: str,
        stable_id: str,
        label: str,
        text: str,
        source_tier: str,
        needs_review: int,
        source_ref: str,
        payload: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO events
            (event_id, event_type, stable_id, label, text, source_tier, trust_level, needs_review, source_ref, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, event_type, stable_id, label, text, source_tier, TIER_TRUST[source_tier], needs_review, source_ref, _json(payload)),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO event_fts(event_id, label, text) VALUES (?, ?, ?)",
            (event_id, label, text),
        )
        entities = sorted(_entities_for(label + " " + text), key=lambda item: (item[2], len(item[0])), reverse=True)[:120]
        for name, role, weight in entities:
            entity_id = _entity_id(name)
            self.conn.execute(
                "INSERT OR IGNORE INTO entities(entity_id, name, normalized, entity_type, payload_json) VALUES (?, ?, ?, ?, ?)",
                (entity_id, name, _normalize_entity(name), _entity_type(name), "{}"),
            )
            self.conn.execute(
                """
                INSERT OR IGNORE INTO event_entities(event_id, entity_id, role, weight, source_tier)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, entity_id, role, weight, source_tier),
            )

    def _insert_alias(self, stable_id: str, stable_type: str, event_id: str, alias_type: str) -> None:
        alias_id = _id("alias", stable_id, stable_type, event_id, alias_type)
        self._stable_to_event[stable_id] = event_id
        self.conn.execute(
            """
            INSERT OR IGNORE INTO id_aliases(alias_id, stable_id, stable_type, event_id, alias_type, payload_json)
            VALUES (?, ?, ?, ?, ?, '{}')
            """,
            (alias_id, stable_id, stable_type, event_id, alias_type),
        )

    def _insert_link(
        self,
        from_event: str,
        to_event: str,
        relation: str,
        *,
        condition: str,
        source_tier: str,
        confidence: float,
        needs_review: int,
        payload: dict[str, Any],
    ) -> None:
        if source_tier == "D" and relation in EXECUTABLE_RELATIONS:
            needs_review = 1
        link_id = _id("link", from_event, to_event, relation, condition)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO event_links
            (link_id, from_event_id, to_event_id, relation, condition, source_tier, trust_level, confidence, needs_review, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (link_id, from_event, to_event, relation, condition, source_tier, TIER_TRUST[source_tier], confidence, needs_review, _json(payload)),
        )


class SqliteSAGStore:
    """KGStore implementation backed by SQLite SAG tables."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        kg_root: str | Path = "data/kg",
        trace_enabled: bool = True,
        max_hops: int = 1,
        event_budget: int = 150,
        entity_stopwords_path: str | Path | None = None,
        degree_penalty: bool = True,
        max_entity_degree: int = 240,
        family_canonicalization: bool = True,
        llm_rerank: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.root = Path(kg_root)
        self.trace_enabled = trace_enabled
        self.max_hops = max(0, int(max_hops))
        self.event_budget = max(25, int(event_budget))
        self.entity_stopwords = _load_stopwords(entity_stopwords_path)
        self.degree_penalty = bool(degree_penalty)
        self.max_entity_degree = max(25, int(max_entity_degree))
        self.family_canonicalization = bool(family_canonicalization)
        self.llm_rerank = bool(llm_rerank)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.fallback = JsonKGStore(self.root) if self.root.exists() else None
        self.last_retrieval_trace: dict[str, Any] = {}
        self._events_by_id: dict[str, sqlite3.Row] = {}
        self._fault_targets: list[sqlite3.Row] = []
        self._stable_by_event_type: dict[tuple[str, str], str] = {}
        self._event_by_stable_type: dict[tuple[str, str], str] = {}
        self._target_ids_by_event: dict[str, set[str]] = {}
        self._target_tokens_by_event: dict[str, set[str]] = {}
        self._entity_degree_by_id: dict[str, int] = {}
        self._entity_degree_by_norm: dict[str, int] = {}
        self._entity_tiers_by_id: dict[str, Counter[str]] = {}
        self._entity_tiers_by_norm: dict[str, Counter[str]] = {}
        self._entity_idf_by_norm: dict[str, float] = {}
        self._load_caches()

    @property
    def errors(self) -> list[dict[str, Any]]:
        """Enumerate fault records from the SAG itself for store-agnostic eval."""

        out: list[dict[str, Any]] = []
        for row in self._fault_targets:
            error_id = self._stable_id_for_event(str(row["event_id"]), "Error")
            if not error_id:
                continue
            payload = _loads(row["payload_json"])
            item = dict(payload) if isinstance(payload, dict) else {}
            item["error_id"] = error_id
            item.setdefault("label", str(row["label"] or error_id))
            item.setdefault("symptom", str(row["text"] or item["label"]))
            out.append(item)
        return out

    def search_errors(self, query: str, limit: int = 5) -> list[Candidate]:
        normalized = _normalize_codes(query)
        q_entity_rows = _query_entity_rows(normalized, self.entity_stopwords)
        retrieval_rows, filtered_entity_trace = self._retrieval_entity_rows(q_entity_rows)
        q_entities = [str(row["entity"]) for row in retrieval_rows][:120]
        q_entity_weights = {
            str(row["entity"]): float(row["adjusted_weight"])
            for row in retrieval_rows
        }
        seed_event_ids, seed_trace = self._seed_events(normalized, retrieval_rows)
        seed_methods = _seed_method_map(seed_trace)
        expanded_event_ids, hyperedges, expand_summary = self._expand(seed_event_ids)
        scored = self._score_targets(normalized, q_entities, q_entity_weights, seed_methods, seed_event_ids, expanded_event_ids)
        role_counts = Counter(str(row["role"]) for row in q_entity_rows)
        high_degree_query_entities = [row for row in filtered_entity_trace if row.get("skip_reason") == "high_degree"][:20]
        expand_summary.update({
            "query_entity_count": len(q_entity_rows),
            "retrieval_entity_count": len(q_entities),
            "query_entity_role_counts": dict(role_counts),
            "filtered_query_entity_count": len(filtered_entity_trace),
            "high_degree_query_entity_count": len(high_degree_query_entities),
            "high_degree_query_entities": high_degree_query_entities,
            "filtered_query_entities": filtered_entity_trace[:40],
        })
        trace = {
            "mode": "sqlite_sag",
            "query_entities": q_entities[:80],
            "query_entity_roles": q_entity_rows[:120],
            "retrieval_entity_roles": retrieval_rows[:120],
            "seed_events": seed_trace[:80],
            "shared_entity_hyperedges": hyperedges[:80],
            "expanded_event_count": len(expanded_event_ids),
            "summary": expand_summary,
            "candidate_paths": [],
        }
        candidates: list[Candidate] = []
        for error_id, score, path_rows, score_components in scored[:limit]:
            target = self._event_for_stable(error_id, "Error")
            if target is None:
                continue
            payload = _loads(target["payload_json"])
            path = [_path_row(row) for row in path_rows[:8]]
            trace["candidate_paths"].append({
                "error_id": error_id,
                "score": round(score, 4),
                "score_components": score_components,
                "paths": path,
            })
            candidates.append(Candidate(
                error_id=error_id,
                label=str(target["label"] or payload.get("label") or error_id),
                score=round(score, 4),
                route="sqlite_sag",
                evidence=[f"sag_path:{p['event_id']}:{p['source_tier']}" for p in path[:3]],
                payload=payload,
            ))
        candidate_paths = trace["candidate_paths"]
        d_only_count = sum(
            1 for item in candidate_paths
            if item.get("paths") and all(str(path.get("source_tier")) == "D" for path in item.get("paths") or [])
        )
        trace["summary"].update({
            "candidate_count": len(candidates),
            "d_only_candidate_count": d_only_count,
            "d_only_top_candidate": bool(candidate_paths and candidate_paths[0].get("paths") and all(str(path.get("source_tier")) == "D" for path in candidate_paths[0].get("paths") or [])),
            "family_canonicalization": self.family_canonicalization,
            "family_canonicalized_candidate_count": sum(
                1 for item in candidate_paths
                if float((item.get("score_components") or {}).get("family_boost") or 0.0) > 0.0
            ),
            "llm_rerank": self.llm_rerank,
        })
        self.last_retrieval_trace = trace
        return candidates

    def load_locked_subgraph(self, error_id: str) -> LockedSubgraph:
        target = self._event_for_stable(error_id, "Error")
        if target is None:
            if self.fallback is not None:
                return self.fallback.load_locked_subgraph(error_id)
            raise KeyError(f"unknown error_id: {error_id}")
        payload = _loads(target["payload_json"])
        check_refs = self._reachable_check_events(str(target["event_id"]))
        checks: list[CheckNode] = []
        solutions_by_check: dict[str, list[SolutionNode]] = {}
        next_edges_by_check: dict[str, list[dict[str, Any]]] = {}
        for check_event, depth, incoming in check_refs:
            check_payload = _loads(check_event["payload_json"])
            check_id = str(check_payload.get("check_id") or check_event["stable_id"])
            if not check_id:
                continue
            check_payload["_graph_depth"] = depth
            check_payload["_incoming_relation"] = str(incoming["relation"] or "")
            check_payload["_incoming_condition"] = str(incoming["condition"] or "")
            check_payload["_source_error_id"] = error_id
            check_payload["_source_error_label"] = str(target["label"] or error_id)
            check_payload["_source_tier"] = str(check_event["source_tier"])
            check_payload["_introduced_by"] = "primary_subgraph"
            check = CheckNode(
                check_id=check_id,
                label=str(check_payload.get("label") or check_event["label"] or check_id),
                how_to_check=str(check_payload.get("how_to_check") or check_event["text"] or check_event["label"] or check_id),
                step_order=int(check_payload.get("step_order") or 0),
                destructive=_is_destructive(check_payload),
                payload=check_payload,
            )
            checks.append(check)
            sols = self._solutions_for_check(str(check_event["event_id"]))
            solutions_by_check[check_id] = sols
        check_event_by_id = {str(row["event_id"]): row for row, _, _ in check_refs}
        check_id_by_event = {
            event_id: str(_loads(row["payload_json"]).get("check_id") or row["stable_id"])
            for event_id, row in check_event_by_id.items()
        }
        for from_event_id, from_check_id in check_id_by_event.items():
            for link in self.conn.execute(
                """
                SELECT l.*, e.label AS to_label, e.payload_json AS to_payload
                FROM event_links l
                JOIN events e ON e.event_id = l.to_event_id
                WHERE l.from_event_id = ? AND l.relation = 'next'
                  AND l.needs_review = 0 AND l.source_tier != 'D'
                """,
                (from_event_id,),
            ):
                to_event_id = str(link["to_event_id"])
                if to_event_id not in check_id_by_event:
                    continue
                next_edges_by_check.setdefault(from_check_id, []).append({
                    "from_check_id": from_check_id,
                    "to_check_id": check_id_by_event[to_event_id],
                    "to_label": str(link["to_label"] or check_id_by_event[to_event_id]),
                    "condition": str(link["condition"] or ""),
                    "relation": "next",
                })
        checks.sort(key=lambda c: (int(c.payload.get("_graph_depth") or 0), c.step_order or 9999, c.check_id))
        sources = sorted({str(payload.get("source_title") or payload.get("source") or "SQLite SAG")})
        return LockedSubgraph(
            error_id=error_id,
            label=str(payload.get("label") or target["label"] or error_id),
            symptom=str(payload.get("symptom") or ""),
            category=str(payload.get("category") or ""),
            escalation_target=str(payload.get("escalation_target") or ""),
            required_info=_required_info_labels(payload),
            checks=checks,
            solutions_by_check=solutions_by_check,
            next_edges_by_check=next_edges_by_check,
            sources=sources,
            payload=payload,
        )

    def read_review_queue(self, name: str) -> list[dict]:
        return self.fallback.read_review_queue(name) if self.fallback is not None else []

    def write_review_queue(self, name: str, data: list[dict]) -> None:
        if self.fallback is not None:
            self.fallback.write_review_queue(name, data)

    def dry_run_apply(self, candidate: dict) -> dict:
        return self.fallback.dry_run_apply(candidate) if self.fallback is not None else {"status": "unsupported"}

    def apply_approved(self, candidate: dict) -> dict:
        return self.fallback.apply_approved(candidate) if self.fallback is not None else {"status": "unsupported"}

    def apply_required_info_approved(self, candidate: dict) -> dict:
        return self.fallback.apply_required_info_approved(candidate) if self.fallback is not None else {"status": "unsupported"}

    def _seed_events(self, query: str, q_entity_rows: list[dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
        seeds: set[str] = set()
        trace: list[dict[str, Any]] = []
        fts_query = _fts_query_from_rows(q_entity_rows) or _fts_query(query)
        if fts_query:
            try:
                rows = self.conn.execute(
                    """
                    SELECT e.event_id, e.event_type, e.label, e.source_tier, bm25(event_fts) AS rank
                    FROM event_fts
                    JOIN events e ON e.event_id = event_fts.event_id
                    WHERE event_fts MATCH ?
                    ORDER BY rank
                    LIMIT 80
                    """,
                    (fts_query,),
                ).fetchall()
                for row in rows:
                    seeds.add(str(row["event_id"]))
                    trace.append({"event_id": row["event_id"], "method": "fts", "label": row["label"], "source_tier": row["source_tier"]})
            except sqlite3.OperationalError:
                pass
        seed_entities = [
            row for row in q_entity_rows
            if row["role"] not in {"noise", "action_tried", "temporal_context"}
            and not self._is_high_frequency_entity(str(row["normalized"]), str(row["role"]))
        ][:80]
        for entity_row in seed_entities:
            entity = str(entity_row["entity"])
            for row in self.conn.execute(
                """
                SELECT ee.event_id, e.event_type, e.label, e.source_tier
                FROM entities ent
                JOIN event_entities ee ON ee.entity_id = ent.entity_id
                JOIN events e ON e.event_id = ee.event_id
                WHERE ent.normalized = ?
                LIMIT 60
                """,
                (_normalize_entity(entity),),
            ):
                seeds.add(str(row["event_id"]))
                trace.append({
                    "event_id": row["event_id"],
                    "method": "entity",
                    "entity": entity,
                    "entity_role": entity_row["role"],
                    "entity_degree": self._entity_degree_by_norm.get(str(entity_row["normalized"]), 0),
                    "label": row["label"],
                    "source_tier": row["source_tier"],
                })
        if len(seeds) < 10:
            like_terms = [str(row["entity"]) for row in seed_entities if len(str(row["entity"])) >= 3][:3]
            for term in like_terms:
                for row in self.conn.execute(
                    """
                    SELECT event_id, event_type, label, source_tier
                    FROM events
                    WHERE source_tier != 'D' AND (label LIKE ? OR text LIKE ?)
                    LIMIT 20
                    """,
                    (f"%{term}%", f"%{term}%"),
                ):
                    seeds.add(str(row["event_id"]))
                    trace.append({"event_id": row["event_id"], "method": "like", "entity": term, "label": row["label"], "source_tier": row["source_tier"]})
        return seeds, trace

    def _retrieval_entity_rows(self, q_entity_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for row in q_entity_rows:
            normalized = str(row.get("normalized") or "")
            role = str(row.get("role") or "")
            degree = self._entity_degree_by_norm.get(normalized, 0)
            idf = self._entity_idf_by_norm.get(normalized, 1.0)
            base_weight = float(row.get("weight") or 0.0)
            raw_role = str(row.get("raw_role") or "")
            skip_reason = self._query_entity_skip_reason(normalized, role, degree, raw_role=raw_role)
            trace_row = {
                "entity": str(row.get("entity") or ""),
                "normalized": normalized,
                "role": role,
                "raw_role": raw_role,
                "degree": degree,
                "idf": round(idf, 4),
                "tier_distribution": dict(self._entity_tiers_by_norm.get(normalized, Counter())),
            }
            if skip_reason:
                trace_row["skip_reason"] = skip_reason
                skipped.append(trace_row)
                continue
            adjusted = base_weight * idf * _degree_weight(degree, self.max_entity_degree)
            if raw_role == "cjk_ngram" and role == "domain":
                adjusted = min(adjusted, base_weight)
            if role == "domain" and degree > max(40, self.max_entity_degree // 2):
                adjusted *= 0.55
            next_row = dict(row)
            next_row["degree"] = degree
            next_row["idf"] = round(idf, 4)
            next_row["adjusted_weight"] = round(max(adjusted, 0.05), 4)
            rows.append(next_row)
        rows.sort(
            key=lambda item: (
                float(item.get("adjusted_weight") or 0.0),
                -int(item.get("degree") or 0),
                len(str(item.get("entity") or "")),
            ),
            reverse=True,
        )
        return rows[:120], skipped

    def _query_entity_skip_reason(self, normalized: str, role: str, degree: int, *, raw_role: str = "") -> str:
        if role in {"noise", "action_tried", "temporal_context"}:
            return role
        if not normalized:
            return "empty"
        if not self.degree_penalty or role == "error_code":
            return ""
        if normalized in {_normalize_entity(item) for item in _HIGH_DEGREE_DEVICE_TERMS}:
            if degree > max(40, self.max_entity_degree // 3):
                return "generic_device_high_degree"
        if raw_role == "cjk_ngram" and role == "domain":
            return "weak_cjk_ngram" if degree else "zero_degree_cjk_ngram"
        if raw_role == "cjk_ngram" and role == "symptom":
            if normalized not in _EXACT_SYMPTOM_TERMS:
                return "partial_symptom_cjk_ngram"
            if normalized == "启动":
                return "weak_symptom_cjk_ngram"
            if degree == 0:
                return "partial_symptom_cjk_ngram"
        if raw_role == "cjk_ngram" and role == "device":
            device_terms = {_normalize_entity(item) for item in _DEVICE_TERMS}
            if normalized not in device_terms:
                return "partial_device_cjk_ngram"
        if degree > self.max_entity_degree and role not in {"symptom", "device"}:
            return "high_degree"
        if degree > self.max_entity_degree * 2 and role == "symptom":
            return "high_degree"
        return ""

    def _expand(self, seed_event_ids: set[str]) -> tuple[set[str], list[dict[str, Any]], dict[str, Any]]:
        expanded = set(seed_event_ids)
        frontier = set(seed_event_ids)
        hyperedges: list[dict[str, Any]] = []
        summary = {
            "skipped_high_degree_entities": 0,
            "skipped_dominant_d_entities": 0,
            "d_tier_expanded_events": 0,
            "d_tier_budget": max(8, self.event_budget // 10),
        }
        for _ in range(self.max_hops):
            if not frontier or len(expanded) >= self.event_budget:
                break
            placeholders = ",".join("?" for _ in frontier)
            entity_rows = self.conn.execute(
                f"""
                SELECT ee.entity_id, ent.name, COUNT(*) AS seed_count
                FROM event_entities ee
                JOIN entities ent ON ent.entity_id = ee.entity_id
                WHERE ee.event_id IN ({placeholders})
                GROUP BY ee.entity_id, ent.name
                ORDER BY seed_count DESC
                LIMIT 80
                """,
                tuple(frontier),
            ).fetchall()
            next_frontier: set[str] = set()
            for entity in entity_rows:
                entity_id = str(entity["entity_id"])
                degree = self._entity_degree_by_id.get(entity_id, 0)
                normalized = _normalize_entity(str(entity["name"] or ""))
                norm_degree = self._entity_degree_by_norm.get(normalized, degree)
                tier_distribution = self._entity_tiers_by_id.get(entity_id, Counter())
                if self.degree_penalty and (degree > self.max_entity_degree or norm_degree > self.max_entity_degree):
                    summary["skipped_high_degree_entities"] += 1
                    continue
                if _d_tier_dominant(tier_distribution):
                    summary["skipped_dominant_d_entities"] += 1
                    continue
                if len(expanded) >= self.event_budget:
                    break
                rows = self.conn.execute(
                    """
                    SELECT ee.event_id, e.source_tier, e.event_type
                    FROM event_entities ee
                    JOIN events e ON e.event_id = ee.event_id
                    WHERE ee.entity_id = ?
                    LIMIT 80
                    """,
                    (entity["entity_id"],),
                ).fetchall()
                added = []
                for row in rows:
                    event_id = str(row["event_id"])
                    if str(row["source_tier"]) == "D" and summary["d_tier_expanded_events"] >= summary["d_tier_budget"]:
                        continue
                    if event_id not in expanded:
                        expanded.add(event_id)
                        next_frontier.add(event_id)
                        added.append(event_id)
                        if str(row["source_tier"]) == "D":
                            summary["d_tier_expanded_events"] += 1
                    if len(expanded) >= self.event_budget:
                        break
                if added:
                    hyperedges.append({
                        "entity": entity["name"],
                        "entity_degree": degree,
                        "normalized_degree": norm_degree,
                        "tier_distribution": dict(tier_distribution),
                        "added_event_ids": added[:20],
                        "added_count": len(added),
                    })
            frontier = next_frontier
        return expanded, hyperedges, summary

    def _score_targets(
        self,
        query: str,
        q_entities: list[str],
        q_entity_weights: dict[str, float],
        seed_methods: dict[str, set[str]],
        seed_event_ids: set[str],
        expanded_event_ids: set[str],
    ) -> list[tuple[str, float, list[sqlite3.Row], dict[str, float]]]:
        q_json_tokens = _json_like_tokens(query)
        rows = []
        if expanded_event_ids:
            placeholders = ",".join("?" for _ in expanded_event_ids)
            rows = self.conn.execute(
                f"SELECT * FROM events WHERE event_id IN ({placeholders})",
                tuple(expanded_event_ids),
            ).fetchall()
        by_error: dict[str, list[tuple[float, sqlite3.Row, dict[str, float]]]] = {}
        for row in rows:
            target_ids = self._target_ids_for_event(str(row["event_id"]))
            if not target_ids:
                continue
            event_score, components = self._event_score(query, q_entities, q_entity_weights, seed_methods, row, seed_event_ids)
            for target_id in target_ids:
                by_error.setdefault(target_id, []).append((event_score, row, components))
        for row in self._fault_targets:
            target_id = self._stable_id_for_event(str(row["event_id"]), "Error")
            if not target_id:
                continue
            target_score, components = self._target_event_score(query, q_entities, q_entity_weights, q_json_tokens, row)
            if target_score <= 0:
                continue
            by_error.setdefault(target_id, []).append((target_score, row, components))
        if self.family_canonicalization:
            by_error = self._canonicalize_families(by_error)
        ranked: list[tuple[str, float, list[sqlite3.Row], dict[str, float]]] = []
        for error_id, items in by_error.items():
            positive = [(score, row, components) for score, row, components in items if score > 0]
            if not positive:
                continue
            positive.sort(key=lambda item: item[0], reverse=True)
            score = positive[0][0] + sum(item[0] for item in positive[1:4]) * 0.25
            score += min(0.35, len(positive) * 0.03)
            if not any(str(row["source_tier"]) != "D" for _, row, _ in positive):
                score *= 0.25
            ranked.append((error_id, score, [row for _, row, _ in positive], _merge_score_components(positive[:4])))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _event_score(
        self,
        query: str,
        q_entities: list[str],
        q_entity_weights: dict[str, float],
        seed_methods: dict[str, set[str]],
        row: sqlite3.Row,
        seed_event_ids: set[str],
    ) -> tuple[float, dict[str, float]]:
        text = str(row["label"] or "") + " " + str(row["text"] or "")
        entity_score = _lexical_score(query, text, q_entities, q_entity_weights)
        components = {
            "fts_score": 2.0 if "fts" in seed_methods.get(str(row["event_id"]), set()) else 0.0,
            "entity_score": entity_score,
            "hyperedge_score": 0.0,
            "tier_score": TIER_SCORE.get(str(row["source_tier"]), 0.0),
            "review_penalty": 0.0,
            "family_boost": 0.0,
            "fault_prior": 0.0,
        }
        score = entity_score
        event_id = str(row["event_id"])
        if event_id in seed_event_ids:
            score += 2.0
        else:
            components["hyperedge_score"] = 0.45
            score += components["hyperedge_score"]
        score += components["tier_score"]
        if int(row["needs_review"] or 0):
            components["review_penalty"] = round(score * 0.55, 4)
            score *= 0.45
        if str(row["event_type"]) == "fault_target":
            score += 1.5
        if str(row["source_tier"]) == "D":
            components["review_penalty"] += round(score * 0.65, 4)
            score *= 0.35
        return score, components

    def _target_event_score(
        self,
        query: str,
        q_entities: list[str],
        q_entity_weights: dict[str, float],
        q_json_tokens: set[str],
        row: sqlite3.Row,
    ) -> tuple[float, dict[str, float]]:
        payload = _loads(row["payload_json"])
        event_id = str(row["event_id"])
        text = str(row["label"] or "") + " " + str(row["text"] or "")
        entity_score = _lexical_score(query, text, q_entities, q_entity_weights)
        components = {
            "fts_score": 0.0,
            "entity_score": entity_score,
            "hyperedge_score": 0.0,
            "tier_score": TIER_SCORE.get(str(row["source_tier"]), 0.0),
            "review_penalty": 0.0,
            "family_boost": 0.0,
            "fault_prior": 0.0,
        }
        score = entity_score + components["tier_score"]
        stable_id = self._stable_id_for_event(event_id, "Error")
        prior = _fault_prior_score(query, stable_id)
        if prior:
            components["fault_prior"] = prior
            score += prior
        overlap = q_json_tokens & self._target_tokens_by_event.get(event_id, set())
        if overlap:
            overlap_score = len(overlap) / max(math.sqrt(len(q_json_tokens) or 1), 1.0) * 2.0
            components["entity_score"] += overlap_score
            score += overlap_score
        q = query.lower()
        for key, weight in (("label", 7.0), ("symptom", 5.0), ("source_title", 3.0)):
            value = str(payload.get(key) or (row["label"] if key == "label" else "") or "").lower().strip()
            if value and (value in q or q in value):
                components["entity_score"] += weight
                score += weight
        for keyword in payload.get("keywords") or []:
            value = str(keyword).lower().strip()
            if value and value in q:
                keyword_score = min(4.0, max(1.0, len(value) / 2.5))
                components["entity_score"] += keyword_score
                score += keyword_score
        for phrase in ("初始化", "相机", "光源", "复判", "ct", "闪退", "运控", "漏检", "误报", "拍照", "ip"):
            if phrase in q and phrase in text.lower():
                components["entity_score"] += 1.5
                score += 1.5
        if int(row["needs_review"] or 0):
            components["review_penalty"] = round(score * 0.55, 4)
            score *= 0.45
        if str(row["source_tier"]) == "D":
            components["review_penalty"] += round(score * 0.65, 4)
            score *= 0.35
        return score, components

    def _canonicalize_families(
        self,
        by_error: dict[str, list[tuple[float, sqlite3.Row, dict[str, float]]]],
    ) -> dict[str, list[tuple[float, sqlite3.Row, dict[str, float]]]]:
        out: dict[str, list[tuple[float, sqlite3.Row, dict[str, float]]]] = {}
        for error_id, items in by_error.items():
            family = _FAMILY_BY_ERROR_ID.get(error_id)
            if not family:
                out.setdefault(error_id, []).extend(items)
                continue
            canonical = str(_FAULT_FAMILIES[family]["canonical"])
            if self._event_for_stable(canonical, "Error") is None:
                out.setdefault(error_id, []).extend(items)
                continue
            boosted: list[tuple[float, sqlite3.Row, dict[str, float]]] = []
            for score, row, components in items:
                next_components = dict(components)
                if error_id != canonical:
                    next_components["family_boost"] = max(float(next_components.get("family_boost") or 0.0), 0.35)
                    score += 0.35
                boosted.append((score, row, next_components))
            out.setdefault(canonical, []).extend(boosted)
        return out

    def _is_high_frequency_entity(self, normalized: str, role: str) -> bool:
        if not self.degree_penalty or role == "error_code":
            return False
        degree = self._entity_degree_by_norm.get(normalized, 0)
        if normalized in {_normalize_entity(item) for item in _HIGH_DEGREE_DEVICE_TERMS}:
            return degree > max(40, self.max_entity_degree // 3)
        return degree > self.max_entity_degree and role != "device"

    def _target_ids_for_event(self, event_id: str) -> list[str]:
        return sorted(self._target_ids_by_event.get(event_id) or [])

    def _stable_id_for_event(self, event_id: str, stable_type: str) -> str:
        return self._stable_by_event_type.get((event_id, stable_type), "")

    def _event_for_stable(self, stable_id: str, stable_type: str) -> sqlite3.Row | None:
        event_id = self._event_by_stable_type.get((stable_id, stable_type), "")
        return self._events_by_id.get(event_id)

    def _load_caches(self) -> None:
        try:
            rows = self.conn.execute("SELECT * FROM events").fetchall()
        except sqlite3.OperationalError:
            return
        self._events_by_id = {str(row["event_id"]): row for row in rows}
        self._fault_targets = [row for row in rows if str(row["event_type"]) == "fault_target"]
        self._entity_degree_by_id = {}
        self._entity_degree_by_norm = {}
        self._entity_tiers_by_id = {}
        self._entity_tiers_by_norm = {}
        for row in self.conn.execute(
            """
            SELECT ent.entity_id, ent.normalized, ee.source_tier, COUNT(DISTINCT ee.event_id) AS n
            FROM entities ent
            JOIN event_entities ee ON ee.entity_id = ent.entity_id
            GROUP BY ent.entity_id, ent.normalized, ee.source_tier
            """
        ):
            entity_id = str(row["entity_id"])
            normalized = str(row["normalized"])
            n = int(row["n"] or 0)
            self._entity_degree_by_id[entity_id] = self._entity_degree_by_id.get(entity_id, 0) + n
            self._entity_degree_by_norm[normalized] = self._entity_degree_by_norm.get(normalized, 0) + n
            self._entity_tiers_by_id.setdefault(entity_id, Counter())[str(row["source_tier"])] += n
            self._entity_tiers_by_norm.setdefault(normalized, Counter())[str(row["source_tier"])] += n
        event_count = max(len(rows), 1)
        self._entity_idf_by_norm = {
            normalized: max(0.2, min(3.0, math.log((event_count + 1) / (degree + 1))))
            for normalized, degree in self._entity_degree_by_norm.items()
        }
        self._target_tokens_by_event = {}
        for row in self._fault_targets:
            payload = _loads(row["payload_json"])
            text = _node_text(payload) or f"{row['label'] or ''} {row['text'] or ''}"
            event_id = str(row["event_id"])
            self._target_tokens_by_event[event_id] = _json_like_tokens(text)
        for row in self.conn.execute("SELECT stable_id, stable_type, event_id FROM id_aliases ORDER BY alias_type"):
            key = (str(row["event_id"]), str(row["stable_type"]))
            stable_key = (str(row["stable_id"]), str(row["stable_type"]))
            self._stable_by_event_type.setdefault(key, str(row["stable_id"]))
            self._event_by_stable_type.setdefault(stable_key, str(row["event_id"]))
        target_map: dict[str, set[str]] = {}
        for row in self._fault_targets:
            event_id = str(row["event_id"])
            stable = self._stable_id_for_event(event_id, "Error")
            if stable:
                target_map.setdefault(event_id, set()).add(stable)
        executable_links = self.conn.execute(
            """
            SELECT from_event_id, to_event_id, relation
            FROM event_links
            WHERE needs_review = 0 AND source_tier != 'D'
            """
        ).fetchall()
        for link in executable_links:
            if str(link["relation"]) != "maps_to_error":
                continue
            from_id = str(link["from_event_id"])
            to_id = str(link["to_event_id"])
            target = self._stable_id_for_event(to_id, "Error")
            if target:
                target_map.setdefault(from_id, set()).add(target)
        for link in executable_links:
            relation = str(link["relation"])
            from_id = str(link["from_event_id"])
            to_id = str(link["to_event_id"])
            if relation in {"has_check", "has_trace", "has_outcome"}:
                target = self._stable_id_for_event(from_id, "Error")
                if target:
                    target_map.setdefault(to_id, set()).add(target)
        for link in executable_links:
            if str(link["relation"]) != "resolved_by":
                continue
            from_id = str(link["from_event_id"])
            to_id = str(link["to_event_id"])
            for target in target_map.get(from_id, set()):
                target_map.setdefault(to_id, set()).add(target)
        self._target_ids_by_event = target_map

    def _reachable_check_events(self, target_event_id: str) -> list[tuple[sqlite3.Row, int, sqlite3.Row]]:
        out: list[tuple[sqlite3.Row, int, sqlite3.Row]] = []
        seen: set[str] = set()
        frontier: list[tuple[str, int, sqlite3.Row]] = []
        for link in self.conn.execute(
            """
            SELECT * FROM event_links
            WHERE from_event_id = ? AND relation = 'has_check'
              AND needs_review = 0 AND source_tier != 'D'
            """,
            (target_event_id,),
        ):
            frontier.append((str(link["to_event_id"]), 0, link))
        while frontier:
            event_id, depth, incoming = frontier.pop(0)
            if event_id in seen:
                continue
            seen.add(event_id)
            row = self.conn.execute("SELECT * FROM events WHERE event_id = ? AND event_type = 'check'", (event_id,)).fetchone()
            if row is None:
                continue
            out.append((row, depth, incoming))
            for link in self.conn.execute(
                """
                SELECT * FROM event_links
                WHERE from_event_id = ? AND relation = 'next'
                  AND needs_review = 0 AND source_tier != 'D'
                """,
                (event_id,),
            ):
                frontier.append((str(link["to_event_id"]), depth + 1, link))
        return out

    def _solutions_for_check(self, check_event_id: str) -> list[SolutionNode]:
        sols: list[SolutionNode] = []
        for row in self.conn.execute(
            """
            SELECT e.*, l.condition AS edge_condition
            FROM event_links l
            JOIN events e ON e.event_id = l.to_event_id
            WHERE l.from_event_id = ? AND l.relation = 'resolved_by'
              AND l.needs_review = 0 AND l.source_tier != 'D'
            """,
            (check_event_id,),
        ):
            payload = _loads(row["payload_json"])
            payload["_edge_condition"] = str(row["edge_condition"] or "")
            sols.append(SolutionNode(
                solution_id=str(payload.get("solution_id") or row["stable_id"] or row["event_id"]),
                content=str(payload.get("content") or row["text"] or ""),
                evidence_level=str(payload.get("evidence_level") or ""),
                destructive=_is_destructive(payload),
                payload=payload,
            ))
        return sols


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS source_runs (
  run_id TEXT PRIMARY KEY,
  source TEXT,
  import_root TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS source_documents (
  doc_id TEXT PRIMARY KEY,
  source_type TEXT,
  title TEXT,
  path TEXT,
  source_tier TEXT NOT NULL,
  trust_level REAL NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS source_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT,
  source_type TEXT,
  title TEXT,
  text TEXT,
  source_tier TEXT NOT NULL,
  trust_level REAL NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS source_threads (
  thread_id TEXT PRIMARY KEY,
  source_run_id TEXT,
  start_time TEXT,
  end_time TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS source_messages (
  message_id TEXT PRIMARY KEY,
  source_run_id TEXT,
  thread_id TEXT,
  create_time TEXT,
  msg_type TEXT,
  text TEXT,
  source_tier TEXT NOT NULL,
  trust_level REAL NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS source_episodes (
  episode_id TEXT PRIMARY KEY,
  source_run_id TEXT,
  thread_id TEXT,
  completeness TEXT,
  source_tier TEXT NOT NULL,
  trust_level REAL NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  stable_id TEXT,
  label TEXT,
  text TEXT,
  source_tier TEXT NOT NULL,
  trust_level REAL NOT NULL,
  needs_review INTEGER NOT NULL DEFAULT 0,
  source_ref TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  normalized TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS event_entities (
  event_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  role TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  source_tier TEXT NOT NULL,
  PRIMARY KEY (event_id, entity_id, role)
);
CREATE TABLE IF NOT EXISTS event_links (
  link_id TEXT PRIMARY KEY,
  from_event_id TEXT NOT NULL,
  to_event_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  condition TEXT NOT NULL DEFAULT '',
  source_tier TEXT NOT NULL,
  trust_level REAL NOT NULL,
  confidence REAL NOT NULL,
  needs_review INTEGER NOT NULL DEFAULT 0,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS id_aliases (
  alias_id TEXT PRIMARY KEY,
  stable_id TEXT NOT NULL,
  stable_type TEXT NOT NULL,
  event_id TEXT NOT NULL,
  alias_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(event_id UNINDEXED, label, text);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized);
CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_event_links_from ON event_links(from_event_id, relation);
CREATE INDEX IF NOT EXISTS idx_event_links_to ON event_links(to_event_id, relation);
CREATE INDEX IF NOT EXISTS idx_id_aliases_stable ON id_aliases(stable_id, stable_type);
"""


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(str(value or "{}"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _entity_id(name: str) -> str:
    return _id("entity", _normalize_entity(name))


def _normalize_codes(text: str) -> str:
    return _OX_CODE.sub(lambda m: f"0x{m.group(1).lower()}", str(text or ""))


def _normalize_entity(text: str) -> str:
    return _normalize_codes(text).lower().strip(" \t\r\n，,。；;:：()（）[]【】\"'")


def _clean_text(value: Any, limit: int = 2000) -> str:
    return " ".join(str(value or "").split())[:limit]


def _node_text(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "label", "symptom", "category", "subsystem", "scenario", "source_title",
        "content", "how_to_check", "summary", "action_label", "outcome_type",
        "root_cause_summary", "condition", "section",
    ):
        value = node.get(key)
        if value:
            parts.append(str(value))
    for key in ("keywords", "required_info", "condition_tags", "source_trace_ids", "source_outcome_ids"):
        value = node.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value if x)
    return _clean_text(" ".join(parts), 4000)


def _label(node: dict[str, Any], fallback: str) -> str:
    return str(node.get("label") or node.get("symptom") or node.get("content") or node.get("how_to_check") or fallback)


def _source_tier(source: str) -> str:
    lowered = str(source or "").lower()
    if any(x in lowered for x in ("sop", "faq", "manual", "手册", "标准操作")):
        return "A"
    if any(x in lowered for x in ("tech_support", "技术支持", "chunk", "kg")):
        return "B"
    return "B"


def _edge_tier(relation: str) -> str:
    return "A" if relation in EXECUTABLE_RELATIONS else "B"


def _entities_for(text: str) -> list[tuple[str, str, float]]:
    found: dict[str, tuple[str, float]] = {}

    def remember(name: str, role: str, weight: float) -> None:
        normalized = _normalize_entity(name)
        if not normalized:
            return
        existing = found.get(name)
        if existing is None or weight > existing[1]:
            found[name] = (role, weight)

    lowered = _normalize_codes(text)
    for phrase in _DOMAIN_PHRASES:
        if phrase.lower() in lowered.lower():
            remember(phrase, "domain_phrase", 2.5)
    for token in _WORD.findall(lowered):
        clean = token.strip("._-")
        if len(clean) >= 2:
            remember(clean, "token", 1.4 if any(ch.isdigit() for ch in clean) else 1.0)
    cjk = _CJK.findall(lowered)
    for size, weight in ((2, 0.55), (3, 0.75), (4, 0.85)):
        for i in range(len(cjk) - size + 1):
            remember("".join(cjk[i : i + size]), "cjk_ngram", weight)
    return [(name, role, weight) for name, (role, weight) in found.items() if _normalize_entity(name)]


def _query_entities(text: str) -> set[str]:
    return {name for name, _, _ in _entities_for(text)}


def _query_entity_rows(text: str, stopwords: set[str] | None = None) -> list[dict[str, Any]]:
    stopwords = stopwords or _DEFAULT_QUERY_STOPWORDS
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, raw_role, raw_weight in _entities_for(text):
        normalized = _normalize_entity(name)
        if not normalized or normalized in seen:
            continue
        role = _query_entity_role(name, text, stopwords)
        if role == "noise":
            weight = 0.0
        elif role == "domain":
            weight = max(float(raw_weight), QUERY_ROLE_WEIGHT["domain"])
        else:
            weight = QUERY_ROLE_WEIGHT.get(role, 1.0)
        rows.append({
            "entity": name,
            "normalized": normalized,
            "role": role,
            "weight": round(weight, 4),
            "raw_role": raw_role,
        })
        seen.add(normalized)
    rows.sort(key=lambda row: (float(row["weight"]), len(str(row["entity"])), str(row["entity"])), reverse=True)
    return rows


def _query_entity_role(name: str, query: str, stopwords: set[str]) -> str:
    normalized = _normalize_entity(name)
    if _is_noise_entity(normalized, stopwords):
        return "noise"
    if re.fullmatch(r"0x[0-9a-f]{6,8}", normalized):
        return "error_code"
    if _TIME_TOKEN.fullmatch(normalized):
        return "temporal_context"
    lowered = name.lower()
    if any(term.lower() in lowered for term in _SYMPTOM_TERMS):
        return "symptom"
    if name in _GENERIC_DEVICE_TERMS or normalized in {_normalize_entity(x) for x in _GENERIC_DEVICE_TERMS}:
        return "domain"
    if any(term.lower() in lowered for term in _DEVICE_TERMS):
        return "device"
    if any(marker in normalized for marker in _ACTION_TRIED_MARKERS) and any(marker in query for marker in _ACTION_CONTEXT_MARKERS):
        return "action_tried"
    if any(marker in normalized for marker in _TEMPORAL_MARKERS):
        return "temporal_context"
    if len(name) >= 3:
        return "domain"
    return "noise"


def _is_noise_entity(normalized: str, stopwords: set[str]) -> bool:
    if normalized in {_normalize_entity(item) for item in stopwords}:
        return True
    if len(normalized) <= 1:
        return True
    if len(normalized) <= 4 and any(fragment in normalized for fragment in _WEAK_NGRAM_FRAGMENTS):
        return True
    if len(normalized) <= 5 and any(marker in normalized for marker in _TEMPORAL_MARKERS + ("问题", "客户", "反馈", "操作", "怎么")):
        return True
    return any(marker in normalized for marker in _WEAK_QUERY_MARKERS)


def _fault_prior_score(query: str, stable_id: str) -> float:
    if not stable_id:
        return 0.0
    q = query.lower()
    compact = q.replace(" ", "")

    def has_any(*phrases: str) -> bool:
        return any(phrase.lower().replace(" ", "") in compact for phrase in phrases)

    no_power = has_any(
        "完全无电",
        "无通电",
        "无通电反应",
        "风扇不转",
        "风扇转一下就停",
        "循环掉电",
        "开机键指示灯不亮",
        "键盘灯和鼠标灯也不亮",
        "电源灯不亮",
        "点不亮",
    )
    os_boot = has_any("无启动设备", "提示无启动", "进不了windows", "无法进入windows", "品牌logo", "自检信息")
    running_freeze = has_any("运行一段时间后", "黑屏死机", "风扇仍在转", "有时正常有时无显示")
    environment_reboot = has_any("频繁无故重启", "无故重启", "大功率设备", "市电", "工业环境干扰", "烧焦味")
    board_contact = has_any("板卡氧化", "氧化接触不良", "接口或板卡", "板卡", "潮湿", "粉尘", "振动", "接触不良")
    explicit_board_contact = has_any("板卡氧化", "氧化接触不良", "接口或板卡", "板卡接触不良")
    cad_auto_align = has_any("自动对齐", "精确对齐", "位置存在偏差", "焊盘框", "手动裁剪cad", "重新整理cad")
    cad_angle_mismatch = has_any(
        "cad器件角度不匹配",
        "器件角度不匹配",
        "角度与器件真实角度不匹配",
        "器件框的角度",
    )
    init_camera_ip = has_any("初始化失败", "相机连接异常", "相机ip", "检查相机ip") and has_any("相机", "camera")
    software_auto_exit = has_any("软件会自动退出", "软件自动退出", "应用自动退出", "主程序自动退出", "自动关闭", "突然关闭", "自动退出")
    software_context = has_any("软件", "应用", "主程序", "aoi")
    reset_context = has_any("重启报错", "报错需要重置", "需要重置", "硬复位", "重新开启软件", "测板", "转圈")

    if stable_id == "err:industrial-pc-no-boot":
        if no_power:
            return 4.0
        if os_boot or running_freeze or environment_reboot or board_contact:
            return -5.0 if running_freeze else -3.0
    if stable_id == "err:os-boot-failure-stuck-at-bios" and os_boot:
        return 5.0
    if stable_id == "err:industrial-pc-freeze-black-screen" and running_freeze:
        return 4.0 if explicit_board_contact else 10.0
    if stable_id == "err:industrial-pc-unexpected-reboot" and environment_reboot:
        return 5.0
    if stable_id == "err:pcie-board-not-detected-by-system" and board_contact:
        return 12.0 if explicit_board_contact else 6.0
    if stable_id == "err:cad-angle-mismatch" and cad_angle_mismatch:
        return 12.0
    if stable_id == "err:cad-auto-alignment-failure-ts" and cad_auto_align:
        return 3.0 if cad_angle_mismatch else 12.0
    if stable_id == "err:cad-import-failure" and cad_auto_align:
        return -3.0
    if init_camera_ip:
        if stable_id == "err:init-camera-ip":
            return 8.0
        if stable_id == "err:system-initialization-failure":
            return -3.0
    if software_auto_exit and software_context and not reset_context:
        if stable_id == "err:app-crash-version-0-23-9":
            return 8.0
        if stable_id == "err:software-freeze-crash-restart-error-reset-needed":
            return -4.0
    return 0.0


def _entity_type(name: str) -> str:
    if re.fullmatch(r"0x[0-9a-fA-F]{6,8}", _normalize_entity(name)):
        return "error_code"
    if any(ch.isdigit() for ch in name):
        return "identifier"
    return "term"


def _fts_query(text: str) -> str:
    terms = []
    for term in _WORD.findall(_normalize_codes(text).lower()):
        clean = term.strip("._-")
        if len(clean) >= 2:
            terms.append(clean.replace('"', ""))
    return " OR ".join(f'"{x}"' for x in terms[:12])


def _fts_query_from_rows(rows: list[dict[str, Any]]) -> str:
    terms: list[str] = []
    for row in rows:
        role = str(row.get("role") or "")
        if role not in {"error_code", "symptom", "device", "domain"}:
            continue
        if role == "domain" and float(row.get("adjusted_weight") or row.get("weight") or 0.0) < 0.8:
            continue
        entity = str(row.get("entity") or "").strip().lower().replace('"', "")
        normalized = str(row.get("normalized") or "").strip().lower().replace('"', "")
        for term in (entity, normalized):
            if len(term) >= 2 and term not in terms:
                terms.append(term)
        if len(terms) >= 12:
            break
    return " OR ".join(f'"{x}"' for x in terms[:12])


def _lexical_score(
    query: str,
    text: str,
    q_entities: Iterable[str] | None = None,
    q_entity_weights: dict[str, float] | None = None,
) -> float:
    entities = list(q_entities) if q_entities is not None else sorted(_query_entities(query), key=lambda item: (len(item), item), reverse=True)[:120]
    if not entities:
        return 0.0
    text_norm = _normalize_entity(text)
    normalized_entities = [(_normalize_entity(entity), entity) for entity in entities[:120]]
    hits = [(normalized, entity) for normalized, entity in normalized_entities if normalized and normalized in text_norm]
    if not hits:
        return 0.0
    if not q_entity_weights:
        return len(hits) / max(math.sqrt(len(entities)), 1.0) * 2.0
    hit_weight = sum(float(q_entity_weights.get(entity, q_entity_weights.get(normalized, 1.0))) for normalized, entity in hits)
    total_weight = sum(float(q_entity_weights.get(entity, 1.0)) for entity in entities[:120])
    return hit_weight / max(math.sqrt(total_weight), 1.0) * 2.0


def _degree_weight(degree: int, max_entity_degree: int) -> float:
    if degree <= 0:
        return 1.0
    if degree <= max(1, max_entity_degree // 4):
        return 1.0
    return max(0.25, 1.0 / math.log(degree + 2))


def _d_tier_dominant(tiers: Counter[str]) -> bool:
    total = sum(int(v) for v in tiers.values())
    if total < 8:
        return False
    return int(tiers.get("D", 0)) / max(total, 1) >= 0.85


def _seed_method_map(seed_trace: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in seed_trace:
        event_id = str(row.get("event_id") or "")
        method = str(row.get("method") or "")
        if event_id and method:
            out.setdefault(event_id, set()).add(method)
    return out


def _merge_score_components(items: list[tuple[float, sqlite3.Row, dict[str, float]]]) -> dict[str, float]:
    merged = {
        "fts_score": 0.0,
        "entity_score": 0.0,
        "hyperedge_score": 0.0,
        "tier_score": 0.0,
        "review_penalty": 0.0,
        "family_boost": 0.0,
        "fault_prior": 0.0,
    }
    for idx, (_, _, components) in enumerate(items):
        factor = 1.0 if idx == 0 else 0.25
        for key in merged:
            merged[key] += float(components.get(key) or 0.0) * factor
    return {key: round(value, 4) for key, value in merged.items()}


def _load_stopwords(path: str | Path | None) -> set[str]:
    stopwords = set(_DEFAULT_QUERY_STOPWORDS)
    if not path:
        return stopwords
    p = Path(path)
    if not p.exists():
        return stopwords
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stopwords.add(line)
    return stopwords


def _json_like_tokens(text: str) -> set[str]:
    lowered = _normalize_codes(text).lower()
    tokens = set(_WORD.findall(lowered))
    cjk = _CJK.findall(lowered)
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add("".join(cjk[i : i + 2]))
    return {token for token in tokens if token.strip()}


def _path_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "event_id": str(row["event_id"]),
        "event_type": str(row["event_type"]),
        "label": str(row["label"] or ""),
        "source_tier": str(row["source_tier"]),
        "needs_review": int(row["needs_review"] or 0),
    }


def _is_debug_like(text: str) -> bool:
    markers = ("故障", "异常", "报错", "失败", "无法", "不能", "蓝屏", "重启", "黑屏", "卡死", "闪退", "漏检", "误报", "CAD", "BIOS", "DMP", "日志")
    return any(marker.lower() in text.lower() for marker in markers)


def _message_texts(rows: Iterable[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows:
        if isinstance(row, dict):
            text = _clean_text(row.get("text") or row.get("content_summary"))
            if text:
                out.append(text)
    return out


def _episode_text(episode: dict[str, Any]) -> str:
    parts = []
    for key in ("fault_description_messages", "diagnostic_chain_messages", "resolution_messages"):
        parts.extend(_message_texts(episode.get(key) or []))
    extracted = episode.get("extracted") if isinstance(episode.get("extracted"), dict) else {}
    for key in ("symptom_raw", "conclusion", "key_conclusion"):
        if extracted.get(key):
            parts.append(str(extracted[key]))
    for action in extracted.get("debug_actions") or []:
        parts.append(str(action))
    return _clean_text(" ".join(parts), 4000)


def _required_info_labels(error: dict[str, Any]) -> list[str]:
    structured = error.get("required_info_schema")
    if isinstance(structured, list) and structured:
        out = []
        for item in structured:
            if isinstance(item, dict):
                out.append(str(item.get("question") or item.get("slot") or ""))
            else:
                out.append(str(item))
        return [x for x in out if x]
    raw = error.get("required_info") or []
    return [str(x) for x in raw if str(x or "").strip()]


def _is_destructive(node: dict[str, Any]) -> bool:
    text = " ".join(str(node.get(k) or "") for k in ("label", "how_to_check", "content", "method"))
    return any(word in text for word in ("删除", "格式化", "重装", "短接", "断电", "更换", "清 CMOS"))
