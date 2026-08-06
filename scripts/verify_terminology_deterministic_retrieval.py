#!/usr/bin/env python3
"""Low-cost deterministic terminology retrieval verification (016-020 focus).

This is the model-free half of the terminology layer.  For each case it
replays the deterministic resolver + search-contract pipeline and checks
whether the layer locates the *correct canonical terms* (对照 gold
``required_search_pairs`` / ``must_resolve`` / ``must_not_resolve``), then
runs a cheap deterministic keyword scan over the text corpus to show what a
non-agentic retrieval mode would locate.  No LLM call is made.

Usage:
    PYTHONPATH=src python3 scripts/verify_terminology_deterministic_retrieval.py \
        [--start 16 --limit 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from debug_agent_system.kg_raw_codex.coverage import build_answer_scope
from debug_agent_system.kg_raw_codex.terminology_contract import (
    build_resolver_context,
    build_terminology_search_contract,
)
from debug_agent_system.knowledge_v2.terminology import (
    TerminologyResolver,
    normalize_term,
)

SUITE = REPO_ROOT / "data/eval/terminology_ab_v1"
_TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv",
                  ".xml", ".html", ".htm", ".log", ".ini", ".cfg", ".toml"}


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _iter_text_files() -> list[Path]:
    """All searchable text files under data/kg_v2 and data/raw (docx skipped)."""

    files: list[Path] = []
    for scope in ("kg_v2", "raw"):
        root = REPO_ROOT / "data" / scope
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in _TEXT_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                files.append(path)
    return sorted(files)


def _deterministic_hits(terms: list[str], files: list[Path]) -> list[dict]:
    """Cheap keyword scan: for each term, which corpus files contain it."""

    hits: list[dict] = []
    for term in terms:
        if not term:
            continue
        needle = term.casefold()
        matched: list[str] = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if needle in text.casefold():
                matched.append(
                    path.relative_to(REPO_ROOT).as_posix()
                )
        hits.append({"term": term, "hit_files": matched[:8],
                     "hit_count": len(matched)})
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=16)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    cases = _load_cases(SUITE / "cases.jsonl")
    selected = cases[args.start - 1 : args.start - 1 + args.limit]
    resolver = TerminologyResolver.from_root(REPO_ROOT / "data/kg_v2")
    text_files = _iter_text_files()

    totals = {
        "must_resolve": 0, "must_resolve_hit": 0,
        "pairs": 0, "pairs_hit": 0,
        "must_not_resolve": 0, "must_not_resolve_ok": 0,
    }
    print("=" * 70)
    print("低成本术语检索模式验证（确定性，无 LLM）")
    print(f"题段：{selected[0]['id']} ~ {selected[-1]['id']}（{len(selected)} 题）")
    print(f"文本语料文件数：{len(text_files)}（data/kg_v2 + data/raw 可读文本）")
    print("=" * 70)

    for case in selected:
        case_id = str(case["id"])
        query = str(case["query"])
        expected = case.get("expected") or {}
        scope = build_answer_scope(query)
        res = resolver.resolve(
            query, limit=30, context=build_resolver_context(scope)
        )
        qe = res.get("query_expansions") or {}
        so = qe.get("search_obligations") or {}
        pairs = so.get("required_pairs") or []
        blocked = qe.get("blocked_expansions") or []
        ambiguous = qe.get("ambiguous_surfaces") or []
        matched = qe.get("matched_terms") or []
        entities = qe.get("canonical_entities") or []

        print(f"\n▶ {case_id} [{case.get('category')}]")
        print(f"  query: {query}")

        # 1) 术语定位（matched_terms）
        print("  ── 术语定位 ──")
        if matched:
            for m in matched:
                print(
                    f"    「{m['surface_form']}」 → 规范名「{m['canonical_name']}」"
                    f" ({', '.join(m.get('relation_types') or [])})"
                )
        else:
            print("    （无已消歧匹配）")
        if blocked:
            for b in blocked:
                print(f"    ⛔ 拦截: 「{b['surface_form']}」→「{b['canonical_name']}」({b.get('reason','')})")
        if ambiguous:
            for a in ambiguous:
                print(f"    ⚠️ 未消歧: 「{a['surface_form']}」")

        # 2) 必搜对 vs gold
        gold_pairs = [
            (p[0], p[1]) for p in expected.get("required_search_pairs") or []
        ]
        gen_pairs = {
            (normalize_term(p.get("source") or ""),
             normalize_term(p.get("canonical") or ""))
            for p in pairs
        }
        print("  ── 必搜对（vs gold required_search_pairs）──")
        if gold_pairs:
            for src, can in gold_pairs:
                key = (normalize_term(src), normalize_term(can))
                ok = key in gen_pairs
                totals["pairs"] += 1
                if ok:
                    totals["pairs_hit"] += 1
                print(f"    {('✓' if ok else '✗')} {src} → {can}")
        else:
            print("    （本题 gold 无必搜对要求）")
            for p in pairs:
                print(f"    (extra) {p['source']} → {p['canonical']}")

        # 3) must_resolve
        print("  ── must_resolve ──")
        entities_norm = {
            normalize_term(e.get("canonical_name") or "") for e in entities
        }
        for term in expected.get("must_resolve") or []:
            totals["must_resolve"] += 1
            ok = normalize_term(term) in entities_norm
            if ok:
                totals["must_resolve_hit"] += 1
            print(f"    {('✓' if ok else '✗')} {term}")

        # 4) must_not_resolve
        for term in expected.get("must_not_resolve") or []:
            totals["must_not_resolve"] += 1
            ok = normalize_term(term) not in entities_norm
            if ok:
                totals["must_not_resolve_ok"] += 1
            print(
                f"  ── must_not_resolve: {('✓ 未解析' if ok else '✗ 被错误解析')} {term}"
            )

        # 5) 低成本确定性检索演示
        if pairs:
            search_terms = sorted({
                term
                for p in pairs
                for term in (p["source"], p["canonical"])
                if term
            })
            hits = _deterministic_hits(search_terms, text_files)
            print("  ── 确定性检索（按必搜对扫描，无 LLM）──")
            for h in hits:
                print(
                    f"    「{h['term']}」命中 {h['hit_count']} 个文件"
                    + (f"：{', '.join(h['hit_files'][:3])}" if h["hit_files"] else "")
                )

    def rate(h, t):
        return "N/A" if not t else f"{h}/{t} ({h / t:.0%})"

    print("\n" + "=" * 70)
    print("汇总（确定性，无 LLM）：")
    print(f"  术语定位 must_resolve : {rate(totals['must_resolve_hit'], totals['must_resolve'])}")
    print(f"  必搜对生成            : {rate(totals['pairs_hit'], totals['pairs'])}")
    print(f"  must_not_resolve 正确 : {rate(totals['must_not_resolve_ok'], totals['must_not_resolve'])}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
