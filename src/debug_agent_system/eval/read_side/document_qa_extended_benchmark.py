"""Build a task-level Document QA benchmark from non-SOP KG v2 documents.

The benchmark unit is a user intent, not a document chunk.  Consecutive
sections that jointly describe one operation are merged into one case, while
independent diagnostic branches or FAQ questions remain separate cases.
Answers are assembled only from approved KnowledgeSection snapshots and their
bound MediaAssets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

from debug_agent_system.eval.read_side.document_qa_benchmark import (
    LEGACY_SIMILARITY_LIMIT,
    _clean,
    _load_query_records,
    _normalized,
    _sha256,
    _similarity,
)


SCHEMA_VERSION = "debug_agent_system.aoi_document_qa_extended.v1"
VALIDATION_SCHEMA_VERSION = (
    "debug_agent_system.aoi_document_qa_extended.validation.v1"
)
BENCHMARK_ID = "aoi-document-qa-extended-v1"
MIN_TASK_CASE_COUNT = 70

DEFAULT_DOCUMENTS = Path("data/kg_v2/objects/knowledge_documents.json")
DEFAULT_SECTIONS = Path("data/kg_v2/objects/knowledge_sections.json")
DEFAULT_STEPS = Path("data/kg_v2/objects/procedure_steps.json")
DEFAULT_MEDIA = Path("data/kg_v2/objects/media_assets.json")
DEFAULT_OUT = Path("data/eval/benchmark/aoi_document_qa_extended_v1.json")
DEFAULT_REPORT_OUT = Path(
    "data/eval/benchmark/aoi_document_qa_extended_v1.report.json"
)
DEFAULT_MARKDOWN_OUT = Path(
    "data/results/benchmark_reports/aoi-document-qa-extended-v1/"
    "Query与答案.md"
)
DEFAULT_HISTORICAL_QUERY_PATHS = (
    Path("data/eval/scenarios/read_side_shared_query_baseline_v1.json"),
    Path("data/eval/scenarios/read_side_pure_codex_baseline_v1.json"),
    Path("data/eval/benchmark/aoi_debug_benchmark_v1.json"),
    Path("data/eval/benchmark/aoi_fae_report_benchmark_v2.json"),
    Path("data/eval/benchmark/aoi_document_qa_pilot_v1.json"),
)

_DOC_SUFFIX = re.compile(r"\.(?:docx|md)$", re.IGNORECASE)
_DUPLICATE_SUFFIX = re.compile(r"\s*\(1\)(?=\.(?:docx|md)$)", re.IGNORECASE)
_QUESTION_SUFFIX = ("？", "?")
_GENERIC_HEADINGS = {
    "问题",
    "问题：",
    "现象",
    "排查步骤",
    "排查步骤：",
    "可能原因:",
    "可能原因：",
    "注意事项：",
    "目标",
    "操作",
    "判断",
    "判断：",
    "文档目的",
    "解决方案",
    "诊断步骤",
    "主要原因深度排查",
}

# These are navigation/index documents whose content is represented by their
# child documents.  Keeping them would create duplicate intents without adding
# an independently answerable procedure.
_INDEX_DOCUMENTS = {
    "Windows常见问题解决方案.docx",
    "Windows系统_引导修复.docx",
    "关闭快速启动.docx",
    "内存检测.docx",
    "卸载并重装显卡驱动.docx",
    "如何进入安全模式.docx",
}


def _spec(
    query: str,
    orders: Iterable[int],
    label: str,
) -> dict[str, Any]:
    return {"query": query, "orders": tuple(orders), "label": label}


# Explicit grouping is used where a document contains several sections.  A
# group represents one realistic user intent.  Sequential M.2 operations,
# keyboard checks and ingress checks are deliberately kept as one task.
_TASK_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "CPU温度过高问题处理指南.docx": (
        _spec(
            "工控机 CPU 温度多少算正常，超过多少需要处理？",
            (3,),
            "CPU 温度范围",
        ),
        _spec(
            "工控机 CPU 温度过高时，应该如何排查和处理？",
            (5, 6, 8, 9, 10, 12, 13, 14, 15, 16),
            "CPU 温度过高排查",
        ),
    ),
    "D盘扩容方法（软件操作）.docx": (
        _spec(
            "如何打开磁盘管理工具？",
            (1,),
            "打开磁盘管理工具",
        ),
        _spec(
            "如何修改磁盘盘符，并把新的机械硬盘设置为 D 盘？",
            (2, 3),
            "修改磁盘盘符",
        ),
    ),
    "M.2SSD硬盘更换和数据迁移.docx": (
        _spec(
            "如何更换 M.2 SSD 并迁移系统？",
            range(2, 10),
            "更换 M.2 SSD 并迁移系统",
        ),
    ),
    "P95使用文档.docx": (
        _spec(
            "如何使用 P95 对 CPU 进行压力测试，并查看测试中是否报错？",
            range(1, 8),
            "使用 P95 进行 CPU 压力测试",
        ),
    ),
    "产品使用 - FAQ.docx": (
        _spec(
            "编程时部分器件框角度与器件真实角度不匹配，应该如何处理？",
            (17, 18),
            "器件框角度不匹配",
        ),
        _spec(
            "多种 CAD 编程时，同一料号下的器件角度不同应该如何处理？",
            (19, 20),
            "多 CAD 器件角度不一致",
        ),
    ),
    "工控机不开机手册.docx": tuple(
        _spec(query, (order,), label)
        for order, query, label in (
            (2, "工控机完全没有通电反应时应该如何排查？", "完全无通电反应"),
            (3, "工控机已经通电但显示器没有画面时应该如何排查？", "通电但无显示"),
            (4, "工控机卡在 POST 自检阶段时应该如何定位故障？", "POST 自检异常"),
            (5, "工控机风扇转一下就停或循环重启时应该如何排查？", "电源保护或短路"),
            (6, "工控机能通过自检但无法进入操作系统时应该如何排查？", "自检后无法进系统"),
            (7, "工控机间歇性黑屏或死机时应该如何排查？", "间歇性黑屏或死机"),
            (8, "修改 BIOS 或更换硬件后工控机无法开机，应该如何恢复？", "BIOS 或硬件变更后不开机"),
            (9, "工控机受到市电或工业环境干扰而频繁重启时应该如何排查？", "市电或环境干扰"),
            (10, "工控机因接口或板卡接触不良而运行不稳定时应该如何排查？", "接口或板卡接触不良"),
            (11, "工控机断电后 BIOS 配置和时间总是丢失，应该如何处理？", "BIOS 电池耗尽"),
            (12, "工控机连接多个硬盘后无法从系统盘启动，应该如何排查？", "多硬盘启动冲突"),
        )
    ),
    "工控机异常(蓝屏&重启&死机）手册.docx": tuple(
        _spec(query, (order,), label)
        for order, query, label in (
            (2, "工控机出现蓝屏时应该收集什么信息并如何排查？", "蓝屏"),
            (3, "工控机陷入无限蓝屏重启循环时应该如何处理？", "无限蓝屏循环"),
            (4, "工控机运行过程中无提示自动重启时应该如何排查？", "无提示重启"),
            (5, "工控机完全卡死且键盘鼠标无响应时应该如何排查？", "完全卡死"),
            (7, "蓝屏、重启或死机疑似由 CPU 引起时应该如何排查？", "CPU 排查"),
            (8, "蓝屏、重启或死机疑似由内存引起时应该如何排查？", "内存排查"),
            (9, "蓝屏或死机伴随硬盘异常时应该如何排查？", "硬盘排查"),
            (10, "黑屏、花屏或蓝屏疑似由显卡引起时应该如何排查？", "显卡排查"),
            (11, "大量网络传输时出现死机、重启或断网，应该如何排查网卡？", "网卡排查"),
            (12, "工控机高负载或无规律重启时应该如何检查电源？", "电源排查"),
            (13, "安装软件、驱动或系统更新后工控机变得不稳定，应该如何恢复？", "软件和驱动冲突"),
        )
    ),
    "数据采集.docx": (
        _spec(
            "不同主程序版本下，如何导出 Ctrl+X 整图？",
            (1, 2, 3),
            "导出 Ctrl+X 整图",
        ),
        _spec("如何导出 SPC 数据？", (1, 4), "导出 SPC 数据"),
        _spec("现场问题截图应该如何标注？", (5,), "现场问题截图"),
        _spec(
            "如何采集并回放 AOI 设备的 FOV 碎图？",
            (6, 7, 8, 9),
            "AOI FOV 碎图采集与回放",
        ),
        _spec("如何导出 SPI 设备的 FOV 碎图？", (10,), "SPI FOV 碎图导出"),
        _spec("如何导出 Buddy 日志和配置？", (11,), "Buddy 日志导出"),
    ),
    "检测界面出现拍照失败问题处理.docx": (
        _spec("拍照失败时如何升级相机 SDK？", (7,), "升级相机 SDK"),
        _spec("拍照失败且日志提示相机丢失时如何升级相机固件？", (8,), "升级相机固件"),
        _spec(
            "拍照失败时，相机网口和其他网卡参数应该如何配置？",
            (11, 13, 15),
            "相机和网卡配置",
        ),
        _spec(
            "拍照失败且系统日志出现网卡过热或 network link is lost 时应该如何处理？",
            (17,),
            "网卡日志异常",
        ),
        _spec("拍照失败需要更换 M.2 网卡时应该如何操作？", (20,), "更换 M.2 网卡"),
        _spec("拍照失败需要更换 PCI 网卡时应该如何操作？", (21,), "更换 PCI 网卡"),
        _spec(
            "拍照失败时如何检查、布置和更换相机网线？",
            (23, 24, 25),
            "相机网线处理",
        ),
        _spec(
            "拍照失败时设备地线和工控机机箱应该如何接地？",
            (27, 28),
            "设备接地",
        ),
        _spec(
            "拍照失败时如何清理系统中的旧驱动和无效设备？",
            (31, 32, 33, 34),
            "清理旧驱动和无效设备",
        ),
        _spec(
            "双轨设备检测大板时报拍照失败、小板正常时，如何设置进程绑核？",
            (36,),
            "双轨进程绑核",
        ),
        _spec(
            "拍照失败处理完成后，如何进行相机老化验证？",
            (40,),
            "相机老化验证",
        ),
    ),
    "键盘随机按键 _ 无响应.docx": (
        _spec(
            "键盘出现随机按键或无响应时，如何判断是键盘、USB 链路还是系统问题？",
            range(1, 17),
            "键盘随机按键或无响应",
        ),
    ),
    "进板失败SOP--20250521.docx": (
        _spec(
            "板卡到达进板口但皮带不转时应该如何排查？",
            range(1, 11),
            "进板失败排查",
        ),
    ),
    "磁盘的数据清理.docx": (
        _spec(
            "AOI 或 SPI 工控机磁盘空间不足时，应该如何清理和迁移数据？",
            range(1, 10),
            "磁盘数据清理",
        ),
    ),
}


_DEFAULT_QUERIES = {
    "Dism++软件使用教程.docx": "Dism++ 应该从哪里下载，下载后如何选择并运行正确版本？",
    "USB设备问题解决方案.docx": "USB 设备疑似受到静电干扰时如何断电释放并恢复？",
    "Windows内存检测方法.docx": "如何使用 Windows 内存诊断检查内存？",
    "memtest86使用方法.docx": "如何使用 MemTest86 检测内存？",
    "windows系统里关闭.docx": "如何在 Windows 系统中关闭快速启动？",
    "主板bios里关闭.docx": "如何在主板 BIOS 中关闭快速启动？",
    "修复引导.docx": "如何使用 Dism++ 修复 Windows 引导？",
    "修复系统.docx": "如何使用 Dism++ 修复受损的 Windows 系统？",
    "分析内存转储文件 (MEMORY.DMP).docx": "如何使用 WinDbg 打开并初步分析 MEMORY.DMP 文件？",
    "加密狗软狗授权步骤.docx": "如何为复判站或 AOI 工控机完成软狗授权？",
    "卸载软件.docx": "如何使用卸载工具完整卸载 Windows 软件？",
    "可以进入系统.docx": "电脑可以进入 Windows 时，如何进入安全模式？",
    "可以进系统.docx": "电脑可以进入系统时，系统损坏或引导异常应该如何修复？",
    "复盘站连接方法与连接不成功异常处理.docx": "如何连接复判站，连接不成功时应该如何排查？",
    "如何进行MEMORY.DMP文件的分析.docx": "如何收集、配置并分析 MEMORY.DMP 文件？",
    "安装显卡驱动.docx": "如何安装 NVIDIA 显卡驱动？",
    "工控机主板接线规范-v1.0-20250403.docx": "工控机主板应该如何规范接线并排查光源参数设置失败？",
    "开机后一直转圈无法进去系统.docx": "Windows 开机一直转圈、无法进入系统时应该如何处理？",
    "彻底卸载显卡驱动.docx": "如何在安全模式下使用 DDU 彻底卸载显卡驱动？",
    "快速系统文件修复.docx": "如何使用 SFC 和 DISM 快速修复 Windows 系统文件？",
    "新硬件稳定性测试（讨论）.docx": "更换工控机硬件后应该如何验证兼容性和稳定性？",
    "无法上网_显示_无Internet_.docx": "Windows 显示“无 Internet”时应该如何排查？",
    "无法进入系统.docx": "Windows 无法进入系统时应该如何修复启动项？",
    "更换_加装内存教程.docx": "如何安全地更换或加装工控机内存？",
    "更新驱动（除显卡驱动）.docx": "除显卡外的设备驱动应该从哪里获取并如何更新？",
    "机械硬盘技术要求.docx": "AOI 或 SPI 工控机选配机械硬盘时需要满足哪些技术要求？",
    "现场问题反馈流程.md": "FAE 遇到现场问题时应该如何反馈、升级和闭环？",
    "电脑不开机排查.docx": "电脑无法开机时应该按照什么顺序排查？",
    "电脑卡顿.docx": "Windows 电脑卡顿时应该如何定位并处理？",
    "磁盘分区与合并.docx": "如何对 Windows 磁盘进行分区或合并？",
    "磁盘文件系统检测和修复.docx": "如何检测并修复 Windows 磁盘文件系统？",
    "禁止Windows更新.docx": "如何禁用 Windows 自动更新并确认设置已经生效？",
    "网页打不开但微信_飞书能用.docx": "微信和飞书能联网但网页打不开时应该如何修复？",
    "软狗更新教程.docx": "已经授权过的软狗应该如何更新授权？",
}


def _load_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected_json_list:{path}")
    return [item for item in payload if isinstance(item, dict)]


def _canonical_document_title(title: str) -> str:
    return _DUPLICATE_SUFFIX.sub("", title)


def _is_duplicate_document(title: str, available_titles: set[str]) -> bool:
    canonical = _canonical_document_title(title)
    return canonical != title and canonical in available_titles


def _default_query(title: str) -> str:
    if title in _DEFAULT_QUERIES:
        return _DEFAULT_QUERIES[title]
    topic = _DOC_SUFFIX.sub("", title).replace("_", " ")
    return f"{topic}应该如何操作？"


def _usable_default_sections(
    document: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    title = str(document.get("title") or "")
    title_stem = _DOC_SUFFIX.sub("", title)
    selected: list[dict[str, Any]] = []
    for section in rows:
        heading = _clean(section.get("heading")).strip("：:")
        summary = _clean(section.get("summary"))
        if not summary:
            continue
        if len(rows) > 1 and (
            _normalized(heading) == _normalized(title_stem)
            or heading in _GENERIC_HEADINGS
            or (
                _normalized(summary) == _normalized(heading)
                and len(summary) < 80
            )
        ):
            continue
        selected.append(section)
    return selected


def _step_text(step: dict[str, Any]) -> str:
    values = [_clean(step.get("instruction"))]
    values.extend(_clean(item) for item in step.get("details") or [])
    if _clean(step.get("expected_result")):
        values.append("预期结果：" + _clean(step.get("expected_result")))
    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized(value)
        if value and normalized not in seen:
            deduplicated.append(value)
            seen.add(normalized)
    return "；".join(deduplicated)


def _section_answer(
    section: dict[str, Any],
    steps_by_section: dict[str, list[dict[str, Any]]],
) -> str:
    steps = steps_by_section.get(str(section.get("section_id") or ""), [])
    if not steps:
        return _clean(section.get("summary"))
    parts: list[str] = []
    section_heading = _normalized(section.get("heading"))
    for step in steps:
        text = _step_text(step)
        if not text:
            continue
        label = _clean(step.get("label")).strip("：:")
        if (
            len(steps) > 1
            and label
            and _normalized(label) != section_heading
        ):
            parts.append(f"#### {label}\n\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts) or _clean(section.get("summary"))


def _task_answer(
    sections: list[dict[str, Any]],
    steps_by_section: dict[str, list[dict[str, Any]]],
) -> str:
    if len(sections) == 1:
        return _section_answer(sections[0], steps_by_section)
    parts: list[str] = []
    for section in sections:
        heading = _clean(section.get("heading")).strip("：:")
        content = _section_answer(section, steps_by_section)
        if not content:
            continue
        parts.append(f"### {heading}\n\n{content}")
    return "\n\n".join(parts)


def _historical_queries(
    paths: Iterable[str | Path],
) -> tuple[list[str], list[Path]]:
    queries: list[str] = []
    existing_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        existing_paths.append(path)
        queries.extend(_load_query_records(path))
    return queries, existing_paths


def _media_index(
    media: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in media:
        if item.get("approved") is not True:
            continue
        for section_id in item.get("section_ids") or []:
            by_section[str(section_id)].append(item)
    return by_section


def _media_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(items, key=lambda value: str(value.get("media_id") or "")):
        media_id = str(item.get("media_id") or "")
        if not media_id or media_id in seen:
            continue
        path = Path(str(item.get("relative_path") or ""))
        if not path.exists():
            raise FileNotFoundError(path)
        evidence.append(
            {
                "media_id": media_id,
                "media_kind": str(item.get("media_kind") or ""),
                "label": _clean(item.get("label")) or path.name,
                "mime_type": str(item.get("mime_type") or ""),
                "path": str(path),
                "sha256": _sha256(path),
                "content_hash": str(item.get("content_hash") or ""),
            }
        )
        seen.add(media_id)
    return evidence


def _filter_task_media(
    document_title: str,
    task_label: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # The current KG snapshot binds every image in 产品使用 - FAQ to the
    # answer sections.  None of those document-wide images can be attributed
    # safely to these two text-only FAQ answers.
    if document_title == "产品使用 - FAQ.docx":
        return [
            item for item in items if item.get("media_kind") != "image"
        ]
    # D盘文档的首个 Section 在写侧误绑了后续新建卷/改盘符截图。
    if (
        document_title == "D盘扩容方法（软件操作）.docx"
        and task_label == "打开磁盘管理工具"
    ):
        return [
            item
            for item in items
            if item.get("media_kind") != "image"
            or any(
                marker in _clean(item.get("label"))
                for marker in ("Win+X", "diskmgmt.msc")
            )
        ]
    return items


def _build_case(
    *,
    case_id: str,
    document: dict[str, Any],
    sections: list[dict[str, Any]],
    query: str,
    task_label: str,
    media: list[dict[str, Any]],
    historical_queries: list[str],
    steps_by_section: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_path = Path(str(document.get("source_path") or ""))
    legacy_max = max(
        (_similarity(query, existing) for existing in historical_queries),
        default=0.0,
    )
    images = [item for item in media if item["media_kind"] == "image"]
    attachments = [item for item in media if item["media_kind"] != "image"]
    section_refs = [
        {
            "section_id": str(section.get("section_id") or ""),
            "section_heading": _clean(section.get("heading")),
            "section_order": int(section.get("section_order") or 0),
            "section_level": int(section.get("level") or 0),
            "source_offsets": list(section.get("source_offsets") or []),
        }
        for section in sections
    ]
    procedure_step_ids = [
        str(step.get("procedure_step_id") or "")
        for section in sections
        for step in steps_by_section.get(
            str(section.get("section_id") or ""),
            [],
        )
        if step.get("procedure_step_id")
    ]
    primary = section_refs[0]
    return {
        "case_id": case_id,
        "split": "extended_review",
        "source_type": "approved_kg_v2_document_snapshot",
        "source_granularity": "task",
        "expectation_origin": "source_document_snapshot",
        "tracks": ["T0_evidence_retrieval", "T1_grounded_answer"],
        "isolated_session": True,
        "query": query,
        "task_label": task_label,
        "source_refs": {
            "document_id": str(document.get("document_id") or ""),
            "document_title": str(document.get("title") or ""),
            "document_path": str(source_path),
            "document_sha256": _sha256(source_path),
            "document_content_hash": str(document.get("content_hash") or ""),
            "section_id": primary["section_id"],
            "section_heading": primary["section_heading"],
            "section_order": primary["section_order"],
            "section_level": primary["section_level"],
            "source_offsets": primary["source_offsets"],
            "section_ids": [item["section_id"] for item in section_refs],
            "section_headings": [
                item["section_heading"] for item in section_refs
            ],
            "sections": section_refs,
            "procedure_step_ids": procedure_step_ids,
        },
        "answer_gold": {
            "answer_mode":
                "knowledge_section_group_with_procedure_steps_and_media",
            "reference_answer": _task_answer(sections, steps_by_section),
            "source_images": images,
            "source_attachments": attachments,
            "generic_governance_text_added": False,
        },
        "quality": {
            "query_generation": "manual_intent_curated",
            "query_requires_human_review": True,
            "answer_source_snapshot_grounded": True,
            "independent_expert_gold": False,
            "graph_ingestion_allowed": False,
            "legacy_max_similarity": round(legacy_max, 6),
        },
    }


def build_dataset(
    documents_path: str | Path = DEFAULT_DOCUMENTS,
    sections_path: str | Path = DEFAULT_SECTIONS,
    steps_path: str | Path = DEFAULT_STEPS,
    media_path: str | Path = DEFAULT_MEDIA,
    *,
    historical_query_paths: Iterable[
        str | Path
    ] = DEFAULT_HISTORICAL_QUERY_PATHS,
) -> dict[str, Any]:
    documents_path = Path(documents_path)
    sections_path = Path(sections_path)
    steps_path = Path(steps_path)
    media_path = Path(media_path)
    documents = _load_list(documents_path)
    all_sections = _load_list(sections_path)
    all_steps = _load_list(steps_path)
    media = _load_list(media_path)
    historical_queries, historical_paths = _historical_queries(
        historical_query_paths
    )
    by_section_media = _media_index(media)

    approved_documents = [
        document for document in documents if document.get("approved") is True
    ]
    available_titles = {
        str(document.get("title") or "") for document in approved_documents
    }
    sections_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in all_sections:
        sections_by_document[str(section.get("document_id") or "")].append(
            section
        )
    for rows in sections_by_document.values():
        rows.sort(
            key=lambda item: (
                int(item.get("section_order") or 0),
                str(item.get("section_id") or ""),
            )
        )
    steps_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in all_steps:
        steps_by_section[str(step.get("section_id") or "")].append(step)
    for rows in steps_by_section.values():
        rows.sort(
            key=lambda item: (
                int(item.get("step_order") or 0),
                str(item.get("procedure_step_id") or ""),
            )
        )

    task_rows: list[
        tuple[dict[str, Any], list[dict[str, Any]], str, str]
    ] = []
    for document in sorted(
        approved_documents,
        key=lambda item: str(item.get("title") or ""),
    ):
        title = str(document.get("title") or "")
        if (
            "异常处理 - 标准操作流程" in title
            or title in _INDEX_DOCUMENTS
            or _is_duplicate_document(title, available_titles)
        ):
            continue
        rows = sections_by_document.get(
            str(document.get("document_id") or ""),
            [],
        )
        by_order = {
            int(section.get("section_order") or 0): section
            for section in rows
        }
        specs = _TASK_GROUPS.get(title)
        if specs:
            for spec in specs:
                selected = [
                    by_order[order]
                    for order in spec["orders"]
                    if order in by_order
                ]
                if selected:
                    task_rows.append(
                        (
                            document,
                            selected,
                            str(spec["query"]),
                            str(spec["label"]),
                        )
                    )
            continue
        selected = _usable_default_sections(document, rows)
        if selected:
            task_rows.append(
                (
                    document,
                    selected,
                    _default_query(title),
                    _DOC_SUFFIX.sub("", title),
                )
            )

    cases: list[dict[str, Any]] = []
    truncated_task_count = 0
    for document, selected, query, task_label in task_rows:
        if "…" in _task_answer(selected, steps_by_section):
            truncated_task_count += 1
            continue
        media_items: list[dict[str, Any]] = []
        for section in selected:
            media_items.extend(
                by_section_media.get(str(section.get("section_id") or ""), [])
            )
        media_items = _filter_task_media(
            str(document.get("title") or ""),
            task_label,
            media_items,
        )
        cases.append(
            _build_case(
                case_id=f"ext-doc-qa-{len(cases) + 1:03d}",
                document=document,
                sections=selected,
                query=query,
                task_label=task_label,
                media=_media_evidence(media_items),
                historical_queries=historical_queries,
                steps_by_section=steps_by_section,
            )
        )

    document_counts = Counter(
        case["source_refs"]["document_title"] for case in cases
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "build_policy": {
            "scope": "approved_non_sop_documents",
            "benchmark_unit": "independent_user_intent",
            "query_style": "fae_task_oriented_manual_curated",
            "sequential_sections_must_be_merged": True,
            "fixed_case_quota_allowed": False,
            "truncated_source_evidence_allowed": False,
            "answer_source": "approved_knowledge_section_group_snapshot",
            "original_media_must_be_preserved": True,
            "fault_card_actions_allowed_in_answer": False,
            "required_info_allowed_in_answer": False,
            "generic_governance_allowed_in_answer": False,
            "duplicate_document_copies_allowed": False,
            "independent_expert_gold_claim_allowed": False,
            "graph_ingestion_allowed": False,
        },
        "source_manifest": {
            "documents": str(documents_path),
            "documents_sha256": _sha256(documents_path),
            "sections": str(sections_path),
            "sections_sha256": _sha256(sections_path),
            "procedure_steps": str(steps_path),
            "procedure_steps_sha256": _sha256(steps_path),
            "media_assets": str(media_path),
            "media_assets_sha256": _sha256(media_path),
            "historical_query_sources": [
                str(path) for path in historical_paths
            ],
        },
        "cases": cases,
        "coverage": {
            "case_count": len(cases),
            "document_count": len(document_counts),
            "document_counts": dict(sorted(document_counts.items())),
            "source_granularity_counts": {"task": len(cases)},
            "section_reference_count": sum(
                len(case["source_refs"]["section_ids"]) for case in cases
            ),
            "image_reference_count": sum(
                len(case["answer_gold"]["source_images"])
                for case in cases
            ),
            "attachment_reference_count": sum(
                len(case["answer_gold"]["source_attachments"])
                for case in cases
            ),
            "excluded_truncated_task_count": truncated_task_count,
            "legacy_similarity_max": max(
                (
                    float(case["quality"]["legacy_max_similarity"])
                    for case in cases
                ),
                default=0.0,
            ),
        },
    }


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version")
    if dataset.get("benchmark_id") != BENCHMARK_ID:
        issues.append("benchmark_id")
    cases = list(dataset.get("cases") or [])
    if len(cases) < MIN_TASK_CASE_COUNT:
        issues.append("case_count_below_task_floor")

    documents = {
        str(item.get("document_id") or ""): item
        for item in _load_list(DEFAULT_DOCUMENTS)
    }
    sections = {
        str(item.get("section_id") or ""): item
        for item in _load_list(DEFAULT_SECTIONS)
    }
    steps_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in _load_list(DEFAULT_STEPS):
        steps_by_section[str(step.get("section_id") or "")].append(step)
    for rows in steps_by_section.values():
        rows.sort(
            key=lambda item: (
                int(item.get("step_order") or 0),
                str(item.get("procedure_step_id") or ""),
            )
        )
    seen_queries: set[str] = set()
    seen_case_ids: set[str] = set()
    banned_query_phrases = (
        "文档给出",
        "文档要求",
        "哪三种",
        "哪几种",
        "分别怎样",
        "针对“",
        "步骤中",
    )
    for case in cases:
        case_id = str(case.get("case_id") or "")
        prefix = case_id + ":"
        if not case_id or case_id in seen_case_ids:
            issues.append(prefix + "case_id")
        seen_case_ids.add(case_id)
        query = _clean(case.get("query"))
        query_norm = _normalized(query)
        if len(query) < 8 or not query.endswith(_QUESTION_SUFFIX):
            issues.append(prefix + "query_quality")
        if any(phrase in query for phrase in banned_query_phrases):
            issues.append(prefix + "query_exam_style")
        if "…" in str((case.get("answer_gold") or {}).get(
            "reference_answer"
        ) or ""):
            issues.append(prefix + "truncated_answer")
        if query_norm in seen_queries:
            issues.append(prefix + "duplicate_query")
        seen_queries.add(query_norm)
        legacy_similarity = float(
            (case.get("quality") or {}).get("legacy_max_similarity") or 0
        )
        if legacy_similarity >= LEGACY_SIMILARITY_LIMIT:
            issues.append(prefix + "legacy_duplicate")

        refs = case.get("source_refs") or {}
        answer_gold = case.get("answer_gold") or {}
        document = documents.get(str(refs.get("document_id") or ""))
        section_ids = [str(value) for value in refs.get("section_ids") or []]
        selected = [sections.get(section_id) for section_id in section_ids]
        if not document or not selected or any(item is None for item in selected):
            issues.append(prefix + "source_object")
            continue
        source_path = Path(str(document.get("source_path") or ""))
        if not source_path.exists():
            issues.append(prefix + "source_path")
        elif _sha256(source_path) != refs.get("document_sha256"):
            issues.append(prefix + "document_sha256")
        expected_answer = _task_answer(
            [item for item in selected if item is not None],
            steps_by_section,
        )
        if answer_gold.get("reference_answer") != expected_answer:
            issues.append(prefix + "reference_answer")
        if (
            answer_gold.get("answer_mode")
            != "knowledge_section_group_with_procedure_steps_and_media"
        ):
            issues.append(prefix + "answer_mode")
        if answer_gold.get("generic_governance_text_added") is not False:
            issues.append(prefix + "generic_governance")
        for media_item in (
            *(answer_gold.get("source_images") or []),
            *(answer_gold.get("source_attachments") or []),
        ):
            path = Path(str(media_item.get("path") or ""))
            if not path.exists():
                issues.append(prefix + "media_path:" + str(path))
            elif _sha256(path) != media_item.get("sha256"):
                issues.append(prefix + "media_sha256:" + str(path))

    coverage = dataset.get("coverage") or {}
    if int(coverage.get("case_count") or 0) != len(cases):
        issues.append("coverage_case_count")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
        "coverage": coverage,
    }


def render_markdown(dataset: dict[str, Any]) -> str:
    coverage = dataset.get("coverage") or {}
    lines = [
        "# AOI Document QA Extended v1：任务级 Query 与参考答案",
        "",
        "> Benchmark 以 FAE 可独立提出的任务为单位，不再按 Section 或",
        "> ProcedureStep 机械出题；同一操作流程的多个章节会合并为一题。",
        "> 答案优先使用完整 ProcedureStep，Section summary 仅作回退，",
        "> 并保留可归属到该任务的原图和附件。",
        "> Query 已做任务化改写，但仍需 FAE 人工复核后才能冻结为 Gold。",
        "",
        f"- Case 总数：{coverage.get('case_count', 0)}",
        f"- 文档总数：{coverage.get('document_count', 0)}",
        f"- Section 引用：{coverage.get('section_reference_count', 0)}",
        f"- 图片引用：{coverage.get('image_reference_count', 0)}",
        f"- 附件引用：{coverage.get('attachment_reference_count', 0)}",
        f"- 因源证据截断而排除："
        f"{coverage.get('excluded_truncated_task_count', 0)}",
        f"- 与历史 Query 最高相似度："
        f"{coverage.get('legacy_similarity_max', 0)}",
        "",
    ]
    for case in dataset.get("cases") or []:
        refs = case["source_refs"]
        headings = " / ".join(refs.get("section_headings") or [])
        lines.extend(
            [
                f"## {case['case_id']} · {case['task_label']}",
                "",
                f"- 原始文档：`{refs['document_path']}`",
                f"- 文档标题：`{refs['document_title']}`",
                f"- 聚合 Section：`{len(refs.get('section_ids') or [])}` 个",
                f"- 聚合 ProcedureStep："
                f"`{len(refs.get('procedure_step_ids') or [])}` 个",
                f"- Section 标题：{headings}",
                "",
                "**Query**",
                "",
                case["query"],
                "",
                "**参考答案**",
                "",
                case["answer_gold"]["reference_answer"],
                "",
            ]
        )
        images = case["answer_gold"].get("source_images") or []
        if images:
            lines.extend(["**原文图片证据**", ""])
            for image in images:
                label = _clean(image.get("label")).replace(
                    "[", "（"
                ).replace("]", "）")
                lines.extend([f"![{label}](<../{image['path']}>)", ""])
        attachments = case["answer_gold"].get("source_attachments") or []
        if attachments:
            lines.extend(["**原文附件**", ""])
            for attachment in attachments:
                label = _clean(attachment.get("label"))
                lines.append(f"- [{label}](<../{attachment['path']}>)")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="aoi-document-qa-extended")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=DEFAULT_MARKDOWN_OUT,
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        dataset = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        dataset = build_dataset(
            args.documents,
            args.sections,
            args.steps,
            args.media,
        )
        _write_json(args.out, dataset)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(
            render_markdown(dataset),
            encoding="utf-8",
        )
    report = validate_dataset(dataset)
    _write_json(args.report_out, report)
    print(
        json.dumps(
            {
                "dataset": str(args.out),
                "report": str(args.report_out),
                "markdown": str(args.markdown_out),
                "status": report["status"],
                "coverage": report["coverage"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
