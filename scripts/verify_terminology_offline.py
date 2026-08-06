#!/usr/bin/env python3
"""Offline terminology-layer verification over the 60-case AB dataset.

This is the deterministic, model-independent half of the terminology AB
evaluation.  It replays the resolver + search-contract pipeline for every
case and measures:

- must_resolve 命中率：resolver 是否解析出要求的规范概念；
- required_search_pairs 生成率：搜索契约是否生成了原词→规范名的必搜对
  （对应 AB 报告里 required_search 2/35 的修复目标）；
- must_not_resolve 正确率：安全负例是否没有被错误解析/锁定；
- blocked/ambiguous 统计：错误扩展是否被上下文门控拦截。

它不调用任何 LLM，因此结果与模型无关，可作为术语层修复的确定性门禁。
"""

from __future__ import annotations

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


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    cases = _load_cases(SUITE / "cases.jsonl")
    resolver = TerminologyResolver.from_root(REPO_ROOT / "data/kg_v2")

    stats = {
        "case_count": len(cases),
        "must_resolve_hits": 0,
        "must_resolve_total": 0,
        "pair_generated": 0,
        "pair_total": 0,
        "must_not_resolve_ok": 0,
        "must_not_resolve_total": 0,
        "blocked_expansions": 0,
        "ambiguous_surfaces": 0,
        "cases_with_all_pairs": 0,
    }
    per_case: list[dict] = []
    pair_failures: list[dict] = []
    not_resolve_failures: list[dict] = []
    resolve_failures: list[dict] = []

    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        expected = case.get("expected") or {}
        scope = build_answer_scope(query)
        res = resolver.resolve(
            query,
            limit=30,
            context=build_resolver_context(scope),
        )
        qe = res.get("query_expansions") or {}
        so = qe.get("search_obligations") or {}
        pairs = so.get("required_pairs") or []
        blocked = qe.get("blocked_expansions") or []
        ambiguous = qe.get("ambiguous_surfaces") or []
        matched_terms = qe.get("matched_terms") or []
        canonical_entities = qe.get("canonical_entities") or []

        normalized_pairs = {
            (normalize_term(p.get("source") or ""),
             normalize_term(p.get("canonical") or ""))
            for p in pairs
        }
        matched_canonicals = {
            normalize_term(m.get("canonical_name") or "")
            for m in matched_terms
        }

        row = {"case_id": case_id, "category": case.get("category")}
        stats["blocked_expansions"] += len(blocked)
        stats["ambiguous_surfaces"] += len(ambiguous)

        # 1) must_resolve
        resolved_all = True
        for term in expected.get("must_resolve") or []:
            stats["must_resolve_total"] += 1
            hit = any(
                normalize_term(term) == c
                or normalize_term(term) in {
                    normalize_term(a) for a in c.get("aliases") or []
                }
                for c in canonical_entities
            ) or normalize_term(term) in matched_canonicals
            if hit:
                stats["must_resolve_hits"] += 1
            else:
                resolved_all = False
                resolve_failures.append({
                    "case_id": case_id,
                    "query": query,
                    "missing": term,
                    "matched": [
                        m.get("canonical_name")
                        for m in matched_terms
                    ],
                })
        row["must_resolve_ok"] = resolved_all

        # 2) required_search_pairs generation
        expected_pairs = expected.get("required_search_pairs") or []
        pair_all = True
        if expected_pairs:
            for pair in expected_pairs:
                stats["pair_total"] += 1
                key = (
                    normalize_term(pair[0]),
                    normalize_term(pair[1]),
                )
                if key in normalized_pairs:
                    stats["pair_generated"] += 1
                else:
                    pair_all = False
                    pair_failures.append({
                        "case_id": case_id,
                        "query": query,
                        "pair": pair,
                        "generated": sorted(
                            f"{a}->{b}" for a, b in normalized_pairs
                        ),
                    })
        row["pairs_ok"] = pair_all
        if pair_all:
            stats["cases_with_all_pairs"] += 1

        # 3) must_not_resolve
        not_resolved_ok = True
        for term in expected.get("must_not_resolve") or []:
            stats["must_not_resolve_total"] += 1
            key = normalize_term(term)
            resolved = key in matched_canonicals
            if not resolved:
                stats["must_not_resolve_ok"] += 1
            else:
                not_resolved_ok = False
                not_resolve_failures.append({
                    "case_id": case_id,
                    "query": query,
                    "must_not_resolve": term,
                })
        row["must_not_resolve_ok"] = not_resolved_ok

        per_case.append(row)

    # ── Report ──
    def rate(hit: int, total: int) -> str:
        if not total:
            return "N/A"
        return f"{hit}/{total} ({hit / total:.1%})"

    print("=" * 64)
    print("KG_v2 术语层离线验证报告")
    print(f"测试集：{SUITE / 'cases.jsonl'}（{stats['case_count']} 题）")
    print(f"解析器：TerminologyResolver（确定性，无 LLM 调用）")
    print("=" * 64)
    print(f"must_resolve 命中       : {rate(stats['must_resolve_hits'], stats['must_resolve_total'])}")
    print(f"required_search_pairs   : {rate(stats['pair_generated'], stats['pair_total'])}")
    print(f"must_not_resolve 正确   : {rate(stats['must_not_resolve_ok'], stats['must_not_resolve_total'])}")
    print(f"blocked_expansions 总数 : {stats['blocked_expansions']}")
    print(f"ambiguous_surfaces 总数 : {stats['ambiguous_surfaces']}")
    print(f"所有必搜对都生成的题    : {stats['cases_with_all_pairs']}/{stats['case_count']}")
    print()

    if resolve_failures:
        print("-- must_resolve 失败 --")
        for f in resolve_failures:
            print(f"  {f['case_id']}: 缺 {f['missing']} | matched={f['matched']}")
    if pair_failures:
        print("-- required_search_pairs 缺失 --")
        for f in pair_failures:
            print(f"  {f['case_id']}: {f['pair']} | generated={f['generated']}")
    if not_resolve_failures:
        print("-- must_not_resolve 误解析 --")
        for f in not_resolve_failures:
            print(f"  {f['case_id']}: {f['must_not_resolve']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
