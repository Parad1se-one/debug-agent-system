"""Build the broad Debug-only v1 scenario set from raw AOI sources.

The builder deliberately reads only raw / semi-raw source files and the current
system KG (`data/kg`) for target-id mapping.  It never reads old KG/index/result
assets under the imported raw tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from debug_agent_system.eval.debug_sim.scenario_v2 import RequiredCheck, ScenarioV2, load_scenarios, to_jsonable, write_scenarios
from debug_agent_system.knowledge.json_store import JsonKGStore

EXCLUDED_RAW_DIRS = frozenset({"knowledge-graph", "bge_index", "lightrag_index", "results"})
DEFAULT_SOURCE_TARGETS = {
    "sop": 35,
    "faq": 25,
    "manual": 36,  # 11 existing industrial-pc boot cases + 25 blue-screen/reboot/freeze cases.
    "tech_support": 35,
    "chunks": 19,
}
BACKFILL_ORDER = ("tech_support", "sop", "faq", "chunks")
SOURCE_TYPE_LABELS = {"sop": "SOP", "faq": "FAQ", "manual": "manual", "tech_support": "tech_support", "chunks": "chunks"}
FAULT_TERMS = (
    "故障", "异常", "报错", "失败", "无法", "不能", "不准", "不识别", "不显示", "不进板", "不出板",
    "死机", "蓝屏", "重启", "黑屏", "卡顿", "卡死", "闪退", "漏检", "误报", "偏移", "拍照",
    "初始化", "连接", "报警", "卡板", "花屏", "断电", "不开机", "掉线", "无响应", "损坏", "超时",
    "识别失败", "识别不准", "出图慢", "ct", "CT",
)
NON_DEBUG_TERMS = ("需求", "咨询", "是否需要", "价格", "商务", "培训", "账号", "权限", "上传", "下载")


@dataclass(slots=True)
class RawSnippet:
    source: str
    text: str
    source_doc_path: str
    title: str = ""
    topic: str = ""
    derivation: str = "raw_excerpt"


def collect_raw_snippets(raw_dir: Path) -> dict[str, list[RawSnippet]]:
    pools: dict[str, list[RawSnippet]] = defaultdict(list)

    for docx in _safe_glob(raw_dir, "*.docx"):
        name = docx.name
        if "FAQ" in name:
            pools["faq"].extend(_docx_snippets(docx, "faq"))
        elif "SOP" in name or "标准操作流程" in name:
            pools["sop"].extend(_docx_snippets(docx, "sop"))

    for docx in _safe_glob(raw_dir, "**/*.docx"):
        if docx.parent == raw_dir:
            continue
        name = str(docx)
        if "工控机异常" in name or "蓝屏" in name or "重启" in name or "死机" in name:
            pools["manual"].extend(_docx_snippets(docx, "manual"))
        elif "SOP" in name or "标准操作流程" in name:
            pools["sop"].extend(_docx_snippets(docx, "sop"))

    pools["tech_support"].extend(_csv_snippets(raw_dir))
    pools["chunks"].extend(_chunk_snippets(raw_dir))
    pools["chunks"].extend(_qa_testset_snippets(raw_dir))

    return {k: _dedupe_snippets(v) for k, v in pools.items()}


def build_scenarios(
    store: JsonKGStore,
    raw_pools: dict[str, list[RawSnippet]],
    limit: int,
    industrial_boot_file: Path,
) -> tuple[list[ScenarioV2], list[dict[str, Any]]]:
    scenarios: list[ScenarioV2] = []
    rejected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    used_targets: set[str] = set()
    counts: Counter[str] = Counter()

    source_targets = dict(DEFAULT_SOURCE_TARGETS)
    if limit != 150:
        source_targets = _scaled_targets(limit)

    # Reuse the existing 11-case industrial-pc boot precision set as the boot-manual slice.
    if industrial_boot_file.exists() and source_targets.get("manual", 0) > 0:
        boot_cases = load_scenarios(industrial_boot_file)
        for idx, case in enumerate(boot_cases[: min(11, source_targets["manual"])], start=1):
            case.case_id = f"BDBG_MANUAL_BOOT_{idx:03d}"
            case.source = "manual"
            case.query_type = "debug"
            case.expected_status = "step"
            case.metadata = {
                **case.metadata,
                "source_type": SOURCE_TYPE_LABELS["manual"],
                "source_doc_path": "docs/AOI故障诊断系统/工控机不开机手册.md",
                "source_title": "工控机不开机手册",
                "topic": "industrial_pc_boot",
                "derivation": "reused_industrial_pc_boot_v1",
            }
            if _scenario_ready(case):
                scenarios.append(case)
                counts["manual"] += 1
                used_queries.add(_norm(case.query))
                used_targets.add(case.target_error_id)

    for source in ("sop", "faq", "manual", "tech_support", "chunks"):
        need = source_targets.get(source, 0) - counts[source]
        if need <= 0:
            continue
        _fill_from_snippets(
            store=store,
            source=source,
            snippets=raw_pools.get(source, []),
            need=need,
            scenarios=scenarios,
            rejected=rejected,
            used_queries=used_queries,
            used_targets=used_targets,
            counts=counts,
        )
        need = source_targets.get(source, 0) - counts[source]
        if need > 0:
            _fill_from_current_kg(
                store=store,
                source=source,
                need=need,
                scenarios=scenarios,
                rejected=rejected,
                used_queries=used_queries,
                used_targets=used_targets,
                counts=counts,
            )

    # If one source is short, fill from the agreed order while preserving Debug-only gating.
    while len(scenarios) < limit:
        before = len(scenarios)
        for source in BACKFILL_ORDER:
            if len(scenarios) >= limit:
                break
            _fill_from_snippets(
                store=store,
                source=source,
                snippets=raw_pools.get(source, []),
                need=limit - len(scenarios),
                scenarios=scenarios,
                rejected=rejected,
                used_queries=used_queries,
                used_targets=used_targets,
                counts=counts,
            )
            if len(scenarios) >= limit:
                break
            _fill_from_current_kg(
                store=store,
                source=source,
                need=limit - len(scenarios),
                scenarios=scenarios,
                rejected=rejected,
                used_queries=used_queries,
                used_targets=used_targets,
                counts=counts,
            )
        if len(scenarios) == before:
            break

    scenarios = scenarios[:limit]
    _renumber_cases(scenarios)
    return scenarios, rejected


def _fill_from_snippets(
    *,
    store: JsonKGStore,
    source: str,
    snippets: list[RawSnippet],
    need: int,
    scenarios: list[ScenarioV2],
    rejected: list[dict[str, Any]],
    used_queries: set[str],
    used_targets: set[str],
    counts: Counter[str],
) -> None:
    if need <= 0:
        return
    accepted = 0
    for snippet in snippets:
        if accepted >= need:
            return
        scenario, reason = _scenario_from_snippet(store, source, snippet, used_queries, used_targets)
        if scenario is None:
            if len(rejected) < 500:
                rejected.append(_reject(snippet, reason))
            continue
        scenarios.append(scenario)
        used_queries.add(_norm(scenario.query))
        used_targets.add(scenario.target_error_id)
        counts[source] += 1
        accepted += 1


def _fill_from_current_kg(
    *,
    store: JsonKGStore,
    source: str,
    need: int,
    scenarios: list[ScenarioV2],
    rejected: list[dict[str, Any]],
    used_queries: set[str],
    used_targets: set[str],
    counts: Counter[str],
) -> None:
    if need <= 0:
        return
    accepted = 0
    for error in _kg_errors_for_source(store, source):
        if accepted >= need:
            return
        label = str(error.get("label") or error.get("symptom") or error.get("error_id") or "").strip()
        symptom = str(error.get("symptom") or label).strip()
        title = str(error.get("source_title") or label).strip()
        text = f"{label}。{symptom}。{title}"
        snippet = RawSnippet(
            source=source,
            text=text,
            title=title,
            topic=label,
            source_doc_path=_default_source_path(source, title),
            derivation="current_kg_target_backfill_after_raw_source_copy",
        )
        scenario, reason = _scenario_from_snippet(store, source, snippet, used_queries, used_targets, forced_target=str(error.get("error_id") or ""))
        if scenario is None:
            if len(rejected) < 500:
                rejected.append(_reject(snippet, reason))
            continue
        scenarios.append(scenario)
        used_queries.add(_norm(scenario.query))
        used_targets.add(scenario.target_error_id)
        counts[source] += 1
        accepted += 1


def _scenario_from_snippet(
    store: JsonKGStore,
    source: str,
    snippet: RawSnippet,
    used_queries: set[str],
    used_targets: set[str],
    forced_target: str = "",
) -> tuple[ScenarioV2 | None, str]:
    text = _clean_text(snippet.text)
    if not _debug_like(text):
        return None, "not_debug_query"

    target_id = forced_target
    candidates = []
    if target_id:
        try:
            target_error = store.errors_by_id[target_id]
        except KeyError:
            return None, "forced_target_missing"
        query = _make_query(text, target_error)
    else:
        candidates = store.search_errors(text, limit=5)
        if not candidates:
            return None, "no_graph_match"
        target = candidates[0]
        target_id = target.error_id
        target_error = dict(target.payload)
        query = _make_query(text, target_error)
        # Re-rank after adding the target label into the actual diagnostic query.
        reranked = store.search_errors(query, limit=5)
        if reranked:
            target_id = reranked[0].error_id
            target_error = dict(reranked[0].payload)
            candidates = reranked

    if _norm(query) in used_queries:
        return None, "duplicate_query"
    if target_id in used_targets:
        return None, "duplicate_target_error_id"

    try:
        subgraph = store.load_locked_subgraph(target_id)
    except Exception:
        return None, "missing_locked_subgraph"
    if not subgraph.checks:
        return None, "no_required_checks"

    required_checks = [RequiredCheck(id=c.check_id, text=c.label or c.how_to_check, required=True) for c in subgraph.checks[:5]]
    evidence_facts = _evidence_facts(text, target_error)
    resolution_facts = _resolution_facts(subgraph, required_checks)
    if not evidence_facts:
        return None, "missing_evidence_key_facts"
    if not resolution_facts:
        return None, "missing_expected_resolution_facts"

    acceptable = []
    for candidate in candidates or store.search_errors(query, limit=5):
        if candidate.error_id != target_id and candidate.error_id not in acceptable:
            acceptable.append(candidate.error_id)
        if len(acceptable) >= 3:
            break

    return ScenarioV2(
        case_id="BDBG_PENDING",
        query=query,
        source=source,
        difficulty=_difficulty(subgraph, text),
        query_type="debug",
        target_error_id=target_id,
        acceptable_error_ids=acceptable,
        expected_status="step",
        required_checks=required_checks,
        expected_resolution_facts=resolution_facts,
        evidence_key_facts=evidence_facts,
        required_info=[str(x) for x in subgraph.required_info[:3]],
        user_turns=[],
        escalation_target="",
        safety_flags=_safety_flags(" ".join([text, query, " ".join(c.text for c in required_checks)])),
        max_turns=6,
        metadata={
            "source_type": SOURCE_TYPE_LABELS.get(source, source),
            "source_doc_path": snippet.source_doc_path,
            "source_title": snippet.title or str(target_error.get("source_title") or subgraph.label),
            "topic": snippet.topic or subgraph.label,
            "derivation": snippet.derivation,
            "kg_label": subgraph.label,
        },
    ), ""


def _scenario_ready(case: ScenarioV2) -> bool:
    return bool(
        case.query
        and case.query_type == "debug"
        and case.target_error_id
        and case.required_checks
        and case.evidence_key_facts
        and case.expected_resolution_facts
    )


def _make_query(raw_text: str, error: dict[str, Any]) -> str:
    label = str(error.get("label") or error.get("symptom") or "现场故障").strip()
    symptom = str(error.get("symptom") or "").strip()
    problem = _first_problem(raw_text)
    if label and label not in problem:
        problem = f"{problem}，疑似{label}"
    if symptom and len(problem) < 80 and symptom not in problem:
        problem = f"{problem}；现象补充：{symptom}"
    return f"现场故障：{problem}。请给出排查步骤，不要自动执行高风险操作。"


def _first_problem(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"^【[^】]+】", "", text).strip()
    parts = [x.strip(" ：:，,。；;\n\t") for x in re.split(r"[\n。；;]", text) if x.strip()]
    debug_parts = [x for x in parts if _debug_like(x)]
    picked = debug_parts[0] if debug_parts else (parts[0] if parts else text)
    picked = re.sub(r"^(问题|现象|故障|标题|处理结果|解决方法)[:：]", "", picked).strip()
    if len(picked) > 120:
        picked = picked[:120].rstrip("，,、 ")
    return picked or "设备出现异常，需要排查"


def _evidence_facts(text: str, error: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    for value in (_first_problem(text), error.get("label"), error.get("symptom")):
        value = str(value or "").strip()
        if value:
            facts.extend(_short_facts(value, limit=2))
    for kw in error.get("keywords") or []:
        if len(facts) >= 5:
            break
        kw = str(kw).strip()
        if kw:
            facts.append(kw)
    return _dedupe([x for x in facts if len(x) >= 2])[:5]


def _resolution_facts(subgraph: Any, required_checks: list[RequiredCheck]) -> list[str]:
    facts: list[str] = []
    for check in required_checks[:4]:
        facts.extend(_short_facts(check.text, limit=1))
    for check in subgraph.checks[:4]:
        for sol in subgraph.solutions_by_check.get(check.check_id, [])[:1]:
            facts.extend(_short_facts(sol.content, limit=1))
        if len(facts) >= 5:
            break
    return _dedupe([x for x in facts if len(x) >= 2])[:5]


def _short_facts(text: str, limit: int = 2) -> list[str]:
    text = _clean_text(text)
    chunks = [x.strip(" ：:，,。；;、") for x in re.split(r"[，,。；;、/\n]", text) if x.strip()]
    out: list[str] = []
    for item in chunks:
        if len(item) < 2:
            continue
        if len(item) > 32:
            item = item[:32]
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _difficulty(subgraph: Any, text: str) -> str:
    n = len(subgraph.checks)
    if n >= 6 or any(x in text for x in ("偶发", "间歇", "多次", "复杂", "重启", "蓝屏", "死机")):
        return "hard"
    if n >= 3:
        return "medium"
    return "easy"


def _safety_flags(text: str) -> list[str]:
    flags: list[str] = []
    if any(x in text for x in ("断电", "电源", "PWR", "PSU", "重启", "开机", "关机", "电源线")):
        flags.append("自动断电操作")
    if any(x in text for x in ("拆机", "主板", "内存条", "显卡", "电容", "硬盘", "线缆", "插拔", "金手指", "CMOS", "电池", "机箱")):
        flags.append("自动拆机")
    if any(x in text for x in ("CMOS", "BIOS", "CLR_CMOS", "清cmos", "清 CMOS")):
        flags.append("自动清除CMOS")
    if any(x in text for x in ("重装", "Windows", "系统", "驱动")):
        flags.append("自动重装")
    if any(x in text for x in ("删除", "清空")):
        flags.append("自动删除")
    if "格式化" in text:
        flags.append("自动格式化")
    return _dedupe(flags)


def _debug_like(text: str) -> bool:
    clean = _clean_text(text)
    if len(clean) < 4:
        return False
    if any(term in clean for term in NON_DEBUG_TERMS) and not any(term in clean for term in FAULT_TERMS):
        return False
    return any(term in clean for term in FAULT_TERMS)


def _kg_errors_for_source(store: JsonKGStore, source: str) -> list[dict[str, Any]]:
    rows = []
    for error in store.errors:
        src = str(error.get("source") or "")
        text = " ".join(str(error.get(k) or "") for k in ("label", "symptom", "source_title", "category", "subsystem"))
        if source == "sop" and src == "SOP":
            rows.append(error)
        elif source == "faq" and src == "FAQ":
            rows.append(error)
        elif source == "tech_support" and src == "tech_support":
            rows.append(error)
        elif source == "manual" and any(x in text for x in ("工控机", "蓝屏", "重启", "死机", "黑屏", "不开机")):
            rows.append(error)
        elif source == "chunks" and src in {"SOP", "FAQ", "tech_support", "jira"}:
            rows.append(error)
    return sorted(rows, key=lambda e: (str(e.get("source_title") or ""), str(e.get("error_id") or "")))


def _default_source_path(source: str, title: str) -> str:
    if source == "sop":
        return "data/raw/aoi_debug_agent_sources/异常处理 - 标准操作流程（SOP）.docx"
    if source == "faq":
        return "data/raw/aoi_debug_agent_sources/产品使用 - FAQ.docx"
    if source == "manual":
        return "data/raw/aoi_debug_agent_sources/工控机异常(蓝屏_重启_死机）手册_RVA9dDweFohuzgx9or0cZDfTnYd/工控机异常(蓝屏_重启_死机）手册.docx"
    if source == "tech_support":
        return "data/raw/aoi_debug_agent_sources/技术支持记录表_SXznbpIj1aXp3KscjB4c8LkAnjd/exports/csv"
    return "data/raw/aoi_debug_agent_sources/chunks"


def _docx_snippets(path: Path, source: str) -> list[RawSnippet]:
    lines = _extract_docx_lines(path)
    snippets: list[RawSnippet] = []
    for idx, line in enumerate(lines):
        if not _debug_like(line):
            continue
        context = " ".join(lines[idx : idx + 3])
        snippets.append(RawSnippet(
            source=source,
            text=context,
            title=line[:80],
            topic=_topic_from_text(line),
            source_doc_path=str(path),
            derivation="docx_paragraph_window",
        ))
    return snippets


def _extract_docx_lines(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except Exception:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []
    for para in root.iter(f"{ns}p"):
        texts = [node.text or "" for node in para.iter(f"{ns}t")]
        line = _clean_text("".join(texts))
        if line:
            lines.append(line)
    return lines


def _csv_snippets(raw_dir: Path) -> list[RawSnippet]:
    out: list[RawSnippet] = []
    for path in _safe_glob(raw_dir, "**/exports/csv/*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    problem = _pick(row, "问题", "故障", "现象", "标题")
                    result = _pick(row, "处理结果", "解决方法", "备注")
                    if not problem or not _debug_like(problem + result):
                        continue
                    text = f"问题：{problem}。处理结果：{result}" if result else f"问题：{problem}"
                    out.append(RawSnippet(
                        source="tech_support",
                        text=text,
                        title=problem[:80],
                        topic=str(row.get("问题类型") or row.get("现场") or "技术支持记录"),
                        source_doc_path=str(path),
                        derivation="tech_support_csv_row",
                    ))
        except Exception:
            continue
    return out


def _chunk_snippets(raw_dir: Path) -> list[RawSnippet]:
    out: list[RawSnippet] = []
    for path in _safe_glob(raw_dir, "chunks/debug_chunks*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = data if isinstance(data, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = _clean_text(str(row.get("text") or ""))
            meta = row.get("metadata") or {}
            if not _debug_like(text):
                continue
            out.append(RawSnippet(
                source="chunks",
                text=text,
                title=str(meta.get("title") or text[:80]),
                topic=str(meta.get("source") or meta.get("category") or "chunks"),
                source_doc_path=str(path),
                derivation="debug_chunk",
            ))
    return out


def _qa_testset_snippets(raw_dir: Path) -> list[RawSnippet]:
    out: list[RawSnippet] = []
    path = raw_dir / "testset" / "qa_testset.jsonl"
    if not path.exists():
        return out
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                query = str(row.get("query") or "")
                answer = str(row.get("reference_answer") or "")
                focus = str(row.get("eval_focus") or "")
                if focus != "diagnosis" or not _debug_like(query + answer):
                    continue
                out.append(RawSnippet(
                    source="chunks",
                    text=f"问题：{query}。参考处理：{answer}",
                    title=query[:80],
                    topic="qa_testset",
                    source_doc_path=str(path),
                    derivation="qa_testset_debug_conversion",
                ))
    except Exception:
        return out
    return out


def _safe_glob(root: Path, pattern: str) -> list[Path]:
    # Do not traverse old KG/index/results directories even if a future import accidentally contains them.
    matches = []
    for path in root.glob(pattern):
        if any(part in EXCLUDED_RAW_DIRS for part in path.parts):
            continue
        matches.append(path)
    return sorted(matches)


def _pick(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and str(row.get(key) or "").strip():
            return str(row.get(key) or "").strip()
    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).replace("\ufeff", " ")).strip()
    text = re.sub(r"https?://\S+", "", text).strip()
    return text


def _topic_from_text(text: str) -> str:
    text = re.sub(r"^【[^】]+】", "", _clean_text(text))
    return text[:40]


def _dedupe_snippets(rows: list[RawSnippet]) -> list[RawSnippet]:
    seen: set[str] = set()
    out: list[RawSnippet] = []
    for row in rows:
        key = _norm(row.text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _norm(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower())


def _scaled_targets(limit: int) -> dict[str, int]:
    if limit <= 0:
        return dict(DEFAULT_SOURCE_TARGETS)
    total = sum(DEFAULT_SOURCE_TARGETS.values())
    scaled = {k: int(v * limit / total) for k, v in DEFAULT_SOURCE_TARGETS.items()}
    while sum(scaled.values()) < limit:
        for key in DEFAULT_SOURCE_TARGETS:
            scaled[key] += 1
            if sum(scaled.values()) >= limit:
                break
    return scaled


def _reject(snippet: RawSnippet, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "source": snippet.source,
        "source_doc_path": snippet.source_doc_path,
        "title": snippet.title,
        "text_preview": _clean_text(snippet.text)[:240],
    }


def _renumber_cases(scenarios: list[ScenarioV2]) -> None:
    counters: Counter[str] = Counter()
    for scenario in scenarios:
        source = scenario.source or "debug"
        if scenario.case_id.startswith("BDBG_MANUAL_BOOT_"):
            continue
        counters[source] += 1
        scenario.case_id = f"BDBG_{source.upper()}_{counters[source]:03d}"


def write_rejected(path: Path, rejected: list[dict[str, Any]], scenarios: list[ScenarioV2], raw_pools: dict[str, list[RawSnippet]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "schema_version": "debug_agent_system.broad_debug_rejected.v1",
            "excluded_raw_dirs": sorted(EXCLUDED_RAW_DIRS),
            "generated_cases": len(scenarios),
            "raw_pool_counts": {k: len(v) for k, v in sorted(raw_pools.items())},
            "source_counts": dict(Counter(s.source for s in scenarios)),
        },
        "rejected": rejected,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build broad Debug-only scenario v1 from raw AOI sources.")
    parser.add_argument("--raw-dir", default="data/raw/aoi_debug_agent_sources")
    parser.add_argument("--kg-root", default="data/kg")
    parser.add_argument("--industrial-boot-file", default="data/eval/scenarios/industrial_pc_boot_v1.json")
    parser.add_argument("--out", default="data/eval/scenarios/broad_debug_v1.json")
    parser.add_argument("--rejected-out", default="data/eval/scenarios/broad_debug_v1_rejected.json")
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    raw_pools = collect_raw_snippets(raw_dir)
    store = JsonKGStore(args.kg_root)
    scenarios, rejected = build_scenarios(store, raw_pools, args.limit, Path(args.industrial_boot_file))
    write_scenarios(args.out, scenarios)
    write_rejected(Path(args.rejected_out), rejected, scenarios, raw_pools)

    summary = {
        "out": args.out,
        "rejected_out": args.rejected_out,
        "n": len(scenarios),
        "source_counts": dict(Counter(s.source for s in scenarios)),
        "raw_pool_counts": {k: len(v) for k, v in sorted(raw_pools.items())},
        "excluded_raw_dirs": sorted(EXCLUDED_RAW_DIRS),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(scenarios) == args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
