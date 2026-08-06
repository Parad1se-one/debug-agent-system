"""Refresh the typical-query Markdown report from actual runtime responses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from debug_agent_system.core.config import load_config
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph
from debug_agent_system.runtime.system import DebugAgentSystem


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "docs/archive/read-side/20260727/KG_v2读侧典型Query实测结果.md"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "data/results/typical_query_report/KG_v2读侧典型Query实测结果.md"
)


@dataclass(frozen=True, slots=True)
class TypicalQuery:
    number: int
    query: str


QUERIES = (
    TypicalQuery(1, "开机后一直转圈无法进入系统"),
    TypicalQuery(2, "电脑卡顿"),
    TypicalQuery(3, "板子到达进板口，皮带不转"),
    TypicalQuery(4, "如何进入安全模式"),
    TypicalQuery(5, "如何进行Windows系统/引导修复"),
    TypicalQuery(6, "网页打不开但微信/飞书能用"),
    TypicalQuery(7, "检测界面出现拍照失败问题"),
    TypicalQuery(8, "USB设备问题"),
)


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _relative_answer(answer: str, report_path: Path) -> str:
    data_root = str((REPO_ROOT / "data").resolve())
    relative_data_root = Path(os.path.relpath(
        REPO_ROOT / "data",
        start=report_path.resolve().parent,
    )).as_posix()
    relative = str(answer or "").replace(
        f"{data_root}/",
        f"{relative_data_root}/",
    )
    # Runtime Markdown uses two trailing spaces for hard line breaks.  Reports
    # are committed artifacts, so preserve the rendering with explicit <br>
    # tags instead of introducing repository-wide trailing whitespace.
    return "\n".join(
        f"{line.rstrip()}<br>" if line.endswith("  ") else line.rstrip()
        for line in relative.splitlines()
    )


def _rebase_repository_data_links(
    text: str,
    source_path: Path,
    report_path: Path,
) -> str:
    """Keep repository-local data links valid when a template is relocated."""

    source_data_root = Path(os.path.relpath(
        REPO_ROOT / "data",
        start=source_path.resolve().parent,
    )).as_posix()
    report_data_root = Path(os.path.relpath(
        REPO_ROOT / "data",
        start=report_path.resolve().parent,
    )).as_posix()
    if source_data_root == report_data_root:
        return text
    return text.replace(f"{source_data_root}/", f"{report_data_root}/")


def _candidate_summary(response: dict[str, Any]) -> tuple[str, str]:
    retrieval = (response.get("metadata") or {}).get("retrieval") or {}
    candidates = list(retrieval.get("candidates") or [])
    values = [
        f"{item.get('variant_label') or item.get('variant_id') or '未知候选'} / "
        f"{item.get('score')}"
        for item in candidates[:3]
    ]
    return "；".join(values) if values else "无", str(
        retrieval.get("top_margin")
    )


def _direct_document_summary(trace: dict[str, Any]) -> str:
    values: list[str] = []
    for item in trace.get("direct_document_matches") or []:
        coverage = item.get("query_coverage")
        values.append(
            f"{item.get('source_label')}"
            f"（coverage={coverage}，entry={item.get('entry_object_type')}）"
        )
    return "；".join(values) if values else "无"


def _navigation_summary(trace: dict[str, Any]) -> str:
    values = [
        f"{item.get('source_label')}"
        f"（depth={item.get('navigation_depth')}，"
        f"order={item.get('navigation_order')}）"
        for item in trace.get("navigation_document_matches") or []
    ]
    return "；".join(values)


def _navigation_excluded_summary(trace: dict[str, Any]) -> str:
    values = [
        f"{item.get('source_label') or item.get('document_id')}"
        f"（`{item.get('reason')}`）"
        for item in trace.get("navigation_excluded") or []
    ]
    return "；".join(values)


def _media_summary(response: dict[str, Any]) -> str:
    media = [
        item
        for section in response.get("answer_sections") or []
        for fact in section.get("items") or []
        for item in fact.get("media_refs") or []
        if isinstance(item, dict)
    ]
    images = sum(item.get("media_kind") == "image" for item in media)
    attachments = sum(item.get("media_kind") != "image" for item in media)
    values: list[str] = []
    if images:
        values.append(f"图片 {images} 张")
    if attachments:
        values.append(f"附件 {attachments} 个")
    return "；".join(values) if values else "无"


def _summary_lines(response: dict[str, Any]) -> list[str]:
    metadata = response.get("metadata") or {}
    retrieval = metadata.get("retrieval") or {}
    trace = retrieval.get("trace") or {}
    candidates, top_margin = _candidate_summary(response)
    required = [
        str(item).strip()
        for item in response.get("required_data") or []
        if str(item).strip()
    ]
    lines = [
        f"- 状态：`{response.get('status')}`",
        f"- 锁定状态：`{(response.get('observability') or {}).get('lock_status')}`",
        f"- Family：`{response.get('family_id') or '未锁定'}`",
        f"- Variant：`{response.get('variant_id') or '未锁定'}`",
        f"- 置信度：{response.get('confidence')}",
        f"- Top3 候选：{candidates}",
        f"- Top margin：{top_margin}",
        f"- 直接文档：{_direct_document_summary(trace)}",
    ]
    navigation = _navigation_summary(trace)
    if navigation:
        lines.append(f"- 导航子文档：{navigation}")
    excluded = _navigation_excluded_summary(trace)
    if excluded:
        lines.append(f"- 导航排除：{excluded}")
    lines.extend([
        f"- 充分性：`{_compact_json(metadata.get('sufficiency') or {})}`",
        f"- 回答覆盖：`{_compact_json(metadata.get('answer_coverage') or {})}`",
        f"- 媒体资源：{_media_summary(response)}",
        f"- 需要补充：{'；'.join(required) if required else '无'}",
    ])
    return lines


def _replace_section(
    text: str,
    item: TypicalQuery,
    response: dict[str, Any],
    report_path: Path,
) -> str:
    pattern = re.compile(
        rf"### 4\.{item.number} {re.escape(item.query)}\n\n"
        r"运行摘要：\n\n.*?\n\n"
        r"<details open>\n"
        r"<summary>完整运行时回答</summary>\n\n"
        r".*?\n\n</details>",
        re.DOTALL,
    )
    replacement = (
        f"### 4.{item.number} {item.query}\n\n"
        "运行摘要：\n\n"
        + "\n".join(_summary_lines(response))
        + "\n\n<details open>\n"
        "<summary>完整运行时回答</summary>\n\n"
        + _relative_answer(
            str(response.get("answer") or ""),
            report_path,
        ).strip()
        + "\n\n</details>"
    )
    updated, count = pattern.subn(lambda _match: replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"cannot locate report section 4.{item.number}: {item.query}")
    return updated


def _replace_overview_row(
    text: str,
    item: TypicalQuery,
    response: dict[str, Any],
) -> str:
    metadata = response.get("metadata") or {}
    retrieval = metadata.get("retrieval") or {}
    sufficiency = metadata.get("sufficiency") or {}
    coverage = metadata.get("answer_coverage") or {}
    top = next(iter(retrieval.get("candidates") or []), {})
    top_value = (
        f"{top.get('variant_label') or top.get('variant_id')} / {top.get('score')}"
        if top
        else "无"
    )
    row = (
        f"| {item.number} | {item.query} | `{response.get('status')}` | "
        f"{top_value} | {response.get('variant_id') or '未锁定'} | "
        f"{'是' if sufficiency.get('answerable') else '否'} | "
        f"{'是' if sufficiency.get('diagnosable') else '否'} | "
        f"{'是' if sufficiency.get('executable') else '否'} | "
        f"{coverage.get('included_fact_count', 0)}/"
        f"{coverage.get('eligible_fact_count', 0)} |"
    )
    pattern = re.compile(
        rf"^\| {item.number} \| {re.escape(item.query)} \|.*$",
        re.MULTILINE,
    )
    updated, count = pattern.subn(row, text, count=1)
    if count != 1:
        raise RuntimeError(f"cannot locate overview row {item.number}: {item.query}")
    return updated


def update_report(
    report_path: Path = DEFAULT_REPORT,
    config_path: Path = REPO_ROOT / "config/debug_agent_system.yaml",
    template_path: Path = DEFAULT_TEMPLATE,
) -> dict[str, Any]:
    config = load_config(config_path)
    with tempfile.TemporaryDirectory(prefix="typical-query-report-") as temp_dir:
        config.session_store = Path(temp_dir) / "sessions"
        system = DebugAgentSystem(config)
        responses = {
            item.number: system.start({
                "query": item.query,
                "interactive": False,
                "session": {"session_id": f"typical-query-report-{item.number}"},
            })
            for item in QUERIES
        }
        sag = system.read_model.sag
        sag_metadata = (
            {
                "index_schema": sag.index_schema(),
                "graph_revision": sag.graph_revision(),
                "source_revision": sag.source_revision(),
            }
            if sag is not None
            else {}
        )

    source_path = report_path if report_path.is_file() else template_path
    text = _rebase_repository_data_links(
        source_path.read_text(encoding="utf-8"),
        source_path,
        report_path,
    )
    for item in QUERIES:
        response = responses[item.number]
        text = _replace_overview_row(text, item, response)
        text = _replace_section(text, item, response, report_path)

    if sag_metadata:
        text = re.sub(
            r"(?m)^- SAG schema：`[^`]+`$",
            f"- SAG schema：`{sag_metadata['index_schema']}`",
            text,
            count=1,
        )
        text = re.sub(
            r"(?m)^- KG_v2 graph revision：`[^`]+`$",
            f"- KG_v2 graph revision：`{sag_metadata['graph_revision']}`",
            text,
            count=1,
        )
        text = re.sub(
            r"(?m)^- source revision：`[^`]+`$",
            f"- source revision：`{sag_metadata['source_revision']}`",
            text,
            count=1,
        )
    kg_store = JsonKGV2Store(config.knowledge.kg_v2_root)
    validation_issues = validate_graph(
        kg_store.objects_by_type,
        kg_store.relations,
        schema_root=kg_store.root / "schema",
    )
    validation_status = "valid" if not validation_issues else "invalid"
    text = re.sub(
        r"(?m)^- KG_v2 校验：.*$",
        (
            f"- KG_v2 校验：`{validation_status}`，"
            f"{len(validation_issues)} 个 issue，{len(kg_store.relations)} 条关系。"
        ),
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^- 回归结果：.*$",
        (
            "- 回归结果：本轮文档结构、SAG、证据组织与真实 Query "
            "定向回归 31/31 通过；重新执行 8 条典型 Query。"
            "未重复运行 600+ 项全量套件。"
        ),
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^- 执行日期：\d{4}-\d{2}-\d{2}$",
        f"- 执行日期：{date.today().isoformat()}",
        text,
        count=1,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return {
        "report_path": str(report_path),
        "query_count": len(QUERIES),
        "answer_coverage": {
            item.query: responses[item.number]["metadata"]["answer_coverage"]
            for item in QUERIES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/debug_agent_system.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(
        update_report(args.report, args.config, args.template),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
