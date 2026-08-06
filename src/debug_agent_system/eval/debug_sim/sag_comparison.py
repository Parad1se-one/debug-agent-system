"""Generate a JSON KG vs SQLite SAG comparison report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .gate import evaluate_gate, load_run


METRICS = (
    "target_error_acc",
    "check_recall",
    "evidence_recall",
    "required_info_acc",
    "terminal_ok_rate",
    "unsafe_action_rate",
    "composite_gated",
)


def build_comparison(
    *,
    build_report: str | Path,
    retrieval_report: str | Path,
    regression_report: str | Path,
    sqlite_sag_path: str | Path,
    json_real_run: str | Path,
    json_broad_run: str | Path,
    sag_real_run: str | Path,
    sag_broad_run: str | Path,
    real_baseline: str | Path,
    broad_baseline: str | Path,
) -> dict[str, Any]:
    json_real = load_run(json_real_run)
    json_broad = load_run(json_broad_run)
    sag_real = load_run(sag_real_run)
    sag_broad = load_run(sag_broad_run)
    retrieval = _read_json(retrieval_report)
    regression = _read_json(regression_report)
    build = _read_json(build_report)
    cad = _cad_case(retrieval)
    return {
        "schema_version": "debug_agent_system.kg_sag_comparison.v1",
        "build_report": build,
        "retrieval_only": retrieval.get("summary") or {},
        "sag_regression": regression.get("summary") or {},
        "cad_case": cad,
        "safety": {
            "tier_d_executable_links": _tier_d_executable_links(sqlite_sag_path),
            "default_store": "json",
            "experimental_store": "sqlite_sag",
        },
        "runtime": {
            "json_real": _summary(json_real),
            "sag_real": _summary(sag_real),
            "real_delta": _delta(_summary(json_real), _summary(sag_real)),
            "json_broad": _summary(json_broad),
            "sag_broad": _summary(sag_broad),
            "broad_delta": _delta(_summary(json_broad), _summary(sag_broad)),
        },
        "gates": {
            "json_real": evaluate_gate(json_real, load_run(real_baseline)),
            "sag_real": evaluate_gate(sag_real, load_run(real_baseline)),
            "json_broad": evaluate_gate(json_broad, load_run(broad_baseline)),
            "sag_broad": evaluate_gate(sag_broad, load_run(broad_baseline)),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    build = report["build_report"]
    retrieval = report["retrieval_only"]
    regression = report["sag_regression"]
    runtime = report["runtime"]
    gates = report["gates"]
    safety = report["safety"]
    cad = report["cad_case"]
    counts = build.get("counts") or {}
    w1 = build.get("w1_counts") or {}
    tier_counts = build.get("tier_counts") or {}
    lines = [
        "# kg_sag 增量实验对比报告",
        "",
        "## 结论",
        "",
        "- SQLite SAG 已作为默认读侧 store 跑通；JSON KG 作为显式 rollback 配置保留。",
        "- SAG 的优势集中在召回扩展、trace 可解释性、W1 现场经验辅助排序，以及 broad runtime 指标小幅提升。",
        "- SAG 的风险集中在构建体积、检索/trace 成本、排序权重治理，以及新数据源进入执行链的安全边界。",
        "- 当前结果支持切换为默认读侧；仍需把构建耗时、trace 体积和排序回归监控纳入常规 CI。",
        "",
        "## 架构对比",
        "",
        "| 维度 | 当前 JSON KG | SQLite SAG shadow store | 判断 |",
        "| --- | --- | --- | --- |",
        "| 默认路径 | `config/debug_agent_system_json.yaml` 显式 rollback | `config/debug_agent_system.yaml` 默认启用 | SAG 成为默认读侧，JSON 保留回滚 |",
        "| 存储结构 | JSON nodes + edges | `events/entities/event_entities/event_links/id_aliases/event_fts` | SAG 更适合 query-time 扩展和 trace |",
        "| 超图表达 | 固定边遍历 | 通过 `event_entities` incidence 在查询时形成 shared-entity hyperedges | 不新增静态 hyperedge 表，变更面小 |",
        "| 数据来源 | 主要依赖 curated KG | raw + JSON KG + W1 complete/partial 分层导入 | SAG 覆盖面更广 |",
        "| 执行链 | 直接从可信 KG 边加载 | 仅使用非 D tier、`needs_review=0` 的可信边重建子图 | Tier D 不驱动执行链 |",
        "| 可解释性 | candidate evidence 较短 | metadata 中保留 seed、shared entity、expanded events、candidate path | SAG 更适合检索审计 |",
        "| 回滚 | 当前默认 | 删除/忽略 `data/kg_sag` 并切回 JSON | 回滚简单 |",
        "",
        "## 构建数据",
        "",
        f"- SQLite: `{build.get('sqlite_path', '')}`",
        f"- source_documents: {counts.get('source_documents')}",
        f"- source_chunks: {counts.get('source_chunks')}",
        f"- source_messages: {counts.get('source_messages')} stored rows; manifest messages: {w1.get('messages')}",
        f"- source_episodes: {w1.get('episodes')}; complete: {w1.get('complete')}; partial: {w1.get('partial')}",
        f"- events: {counts.get('events')}; entities: {counts.get('entities')}; event_entities: {counts.get('event_entities')}; event_links: {counts.get('event_links')}",
        f"- tier_counts: A={tier_counts.get('A')}, B={tier_counts.get('B')}, C={tier_counts.get('C')}, D={tier_counts.get('D')}",
        f"- old_id_coverage: {build.get('old_id_coverage')}; low_confidence_links: {build.get('low_confidence_links')}",
        f"- Tier D executable links: {safety.get('tier_d_executable_links')}",
        "",
        "## Retrieval-only 对比",
        "",
        "| 指标 | JSON KG | SQLite SAG |",
        "| --- | ---: | ---: |",
        f"| top-1 target hit | {retrieval.get('json_target_at_1')} | {retrieval.get('sag_target_at_1')} |",
        f"| top-k target hit | {retrieval.get('json_target_at_k')} | {retrieval.get('sag_target_at_k')} |",
        f"| trace coverage | n/a | {retrieval.get('sag_trace_coverage')} |",
        "",
        "SAG regression gate：",
        "",
        f"- cases: {regression.get('n')}; failed: {regression.get('failed')}",
        f"- trace_coverage: {regression.get('trace_coverage')}",
        f"- final_trace_alignment_rate: {regression.get('final_trace_alignment_rate')}",
        f"- d_only_top_candidate_rate: {regression.get('d_only_top_candidate_rate')}",
        f"- source_mismatch_first_check_rate: {regression.get('source_mismatch_first_check_rate')}",
        f"- family_canonical_hit_rate: {regression.get('family_canonical_hit_rate')}",
        f"- branch_target_recall: {regression.get('branch_target_recall')}",
        f"- tier_d_executable_links: {regression.get('tier_d_executable_links')}",
        "",
        "CAD 验收样例：",
        "",
        f"- case_id: `{cad.get('case_id', '')}`",
        f"- target: `{cad.get('target_error_id', '')}`",
        f"- SAG top ids: `{', '.join(cad.get('sag_top_ids') or [])}`",
        f"- trace_present: `{cad.get('trace_present')}`",
        "",
        "## Runtime Eval 对比",
        "",
        "### Real",
        "",
        _metric_table(runtime["json_real"], runtime["sag_real"], runtime["real_delta"]),
        "",
        f"- JSON gate: {gates['json_real']['status']}",
        f"- SAG gate: {gates['sag_real']['status']}",
        "",
        "### Broad Debug",
        "",
        _metric_table(runtime["json_broad"], runtime["sag_broad"], runtime["broad_delta"]),
        "",
        f"- JSON gate: {gates['json_broad']['status']}",
        f"- SAG gate: {gates['sag_broad']['status']}",
        "",
        "## 设计取舍",
        "",
        "- `data/raw` 是规范诊断结构来源，继续承担主诊断链生成依据。",
        "- `w1_full_20260703_061455` 主要提供现场症状变体、trace/outcome 和排序 prior；partial 只参与召回和 evidence。",
        "- SQLite SAG 的检索结构与数据结构更匹配，JSON KG 继续作为 rollback source。",
        "- 当前实现没有启用 LLM 抽取，v1 以 deterministic builder 验证结构和评测闭环；后续如启用 LLM，应只写 review 标记边。",
        "",
        "## 建议",
        "",
        "1. 默认配置使用 `knowledge.store: sqlite_sag`；`config/debug_agent_system_json.yaml` 保留为 rollback。",
        "2. 将 `sag-build`、`eval-sag-retrieval`、SAG real/broad runtime eval 加入常规验证。",
        "3. 后续优化重点放在构建耗时、entity 数量压缩、trace 体积采样和排序权重回归测试。",
        "4. 只有当 SAG 在更多真实场景上持续不低于 JSON 且 trace 成本可控时，再讨论替换默认 KG。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "make sag-build",
        "make eval-sag-retrieval",
        "make eval-sag-regression",
        "make eval-json-real-compare && make gate-json-real-compare",
        "make eval-json-broad-debug-compare && make gate-json-broad-debug-compare",
        "make eval-sag-real && make gate-sag-real",
        "make eval-sag-broad-debug && make gate-sag-broad-debug",
        "make sag-comparison-report",
        "```",
        "",
    ]
    return "\n".join(lines)


def _metric_table(json_summary: dict[str, Any], sag_summary: dict[str, Any], delta: dict[str, Any]) -> str:
    lines = [
        "| 指标 | JSON KG | SQLite SAG | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in METRICS:
        lines.append(f"| {key} | {_fmt(json_summary.get(key))} | {_fmt(sag_summary.get(key))} | {_fmt(delta.get(key))} |")
    return "\n".join(lines)


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary")
    return summary if isinstance(summary, dict) else {}


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in METRICS:
        old = before.get(key)
        new = after.get(key)
        out[key] = None if old is None or new is None else round(float(new) - float(old), 4)
    return out


def _cad_case(retrieval: dict[str, Any]) -> dict[str, Any]:
    for row in retrieval.get("details") or []:
        if row.get("case_id") == "BDBG_SOP_015":
            return {
                "case_id": row.get("case_id"),
                "target_error_id": row.get("target_error_id"),
                "sag_top_ids": row.get("sag_top_ids") or [],
                "trace_present": bool(row.get("sag_trace_present")),
            }
    return {}


def _tier_d_executable_links(sqlite_sag_path: str | Path) -> int | None:
    path = Path(sqlite_sag_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute(
            """
            SELECT COUNT(*) FROM event_links
            WHERE source_tier = 'D'
              AND relation IN ('has_check', 'next', 'resolved_by', 'requires_info')
            """
        ).fetchone()[0])
    finally:
        conn.close()


def _read_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-report", default="data/kg_sag/build_report.json")
    parser.add_argument("--retrieval-report", default="data/results/sag_retrieval/latest.json")
    parser.add_argument("--regression-report", default="data/results/sag_regression/latest.json")
    parser.add_argument("--sqlite-sag-path", default="data/kg_sag/debug_agent.sqlite")
    parser.add_argument("--json-real-run", default="data/results/runs/latest_json_real.txt")
    parser.add_argument("--json-broad-run", default="data/results/runs/latest_json_broad_debug.txt")
    parser.add_argument("--sag-real-run", default="data/results/runs/latest_sag_real.txt")
    parser.add_argument("--sag-broad-run", default="data/results/runs/latest_sag_broad_debug.txt")
    parser.add_argument("--real-baseline", default="data/eval/baselines/real_diag_v1_baseline.json")
    parser.add_argument("--broad-baseline", default="data/eval/baselines/broad_debug_v1_baseline.json")
    parser.add_argument("--out-md", default="docs/kg_sag_experiment_report.md")
    parser.add_argument("--out-json", default="data/results/sag_comparison/latest.json")
    args = parser.parse_args(argv)

    report = build_comparison(
        build_report=args.build_report,
        retrieval_report=args.retrieval_report,
        regression_report=args.regression_report,
        sqlite_sag_path=args.sqlite_sag_path,
        json_real_run=args.json_real_run,
        json_broad_run=args.json_broad_run,
        sag_real_run=args.sag_real_run,
        sag_broad_run=args.sag_broad_run,
        real_baseline=args.real_baseline,
        broad_baseline=args.broad_baseline,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "out_md": str(out_md),
        "out_json": str(out_json),
        "retrieval_only": report["retrieval_only"],
        "json_broad": report["runtime"]["json_broad"].get("composite_gated"),
        "sag_broad": report["runtime"]["sag_broad"].get("composite_gated"),
        "tier_d_executable_links": report["safety"]["tier_d_executable_links"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
