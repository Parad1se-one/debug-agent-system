"""W9 raw knowledge document intake.

W9 is intentionally separate from W1:

- W1 ingests field evidence from chats / Jira / attachments.
- W9 classifies and stages raw knowledge documents such as manuals, guides,
  procedures, specs, and FAQ sources before they are converted into KG v2
  bundles.

This module does not write KG data yet. It provides an executable checklist:

- inspect one document and decide its `doc_strategy`
- scan a root directory and build a per-document processing checklist
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

from debug_agent_system.agents.tools import parse_document_evidence
from debug_agent_system.knowledge_v2.document_links import extract_docx_hyperlinks
from debug_agent_system.knowledge_v2.source_chunk_builder import (
    build_staged_chunk_manifest,
    read_source_text_lines,
)

W9_EXCLUDED_SOP_NAMES = ("异常处理 - 标准操作流程（SOP）.docx", "异常处理_-_标准操作流程（SOP）.docx")
_TEXT_EXTS = {".md", ".txt"}
_DOC_EXTS = {".doc", ".docx", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx"}
_FULL_STRUCTURE_EXTS = _TEXT_EXTS | {".docx", ".xlsx", ".pptx"}
_OOXML_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_SECTION_HEADING_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)(?:(?<=\.\d)(?=[^\d\s.])|\s+|[．、：:\-]|\.(?!\d))\s*(?P<title>\S.*)$"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.．、:：)]?)\s*")
_PLAYBOOK_HEADING_RE = re.compile(r"^方案(?P<num>[一二三四五六七八九十0-9]+)[：:]\s*(?P<title>\S.*)$")
_FAULT_MANUAL_CASE_RE = re.compile(r"^情况\s*(?P<num>\d+)[：:]\s*(?P<title>\S.*)$")
_FAULT_MANUAL_NUMBERED_CASE_RE = re.compile(
    r"^(?P<num>\d+)[.．、]?\s*(?P<title>蓝屏|无限蓝屏重启循环|重启|死机(?:（[^）]+）)?)$"
)
_FAULT_MANUAL_COMPONENT_RE = re.compile(
    r"^(?P<num>\d+)[.．、]?\s*(?P<title>CPU|内存|硬盘|显卡|网卡|电源|软件、驱动与系统配置冲突)$",
    re.IGNORECASE,
)
_SOP_NAME_RE = re.compile(r"(?<![a-z0-9])sop(?![a-z0-9])", re.IGNORECASE)
_INLINE_STEP_SPLIT_RE = re.compile(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])")
_PROCEDURE_ACTION_PREFIXES = (
    "检查", "确认", "分析", "收集", "导出", "提供", "升级", "回退", "重装", "更换", "排查", "观察", "验证",
    "启用", "卸载", "重启", "截图", "抓取", "记录", "修复", "关闭", "打开", "设置", "测试", "拔插", "安装",
    "使用", "查看", "对比", "比较", "测量", "监控", "清理", "清洁", "恢复", "执行", "联系", "触摸", "优化",
    "限制", "拆下", "拆卸", "涂抹", "整理", "送修", "点击", "选择", "输入", "按下", "按住", "右键", "鼠标右击",
    "进入", "定位", "找到", "双击", "勾选", "取消", "下载", "解压", "关机", "拍照", "取出", "将", "从", "等待",
    "拔掉", "插入", "连续", "逐行", "运行",
    "手摸", "尝试", "复现", "win+r",
)
_NON_ACTION_HEADING_SUFFIXES = ("原因", "场景", "注意事项", "结果解读", "异常情况", "正常情况", "名称", "步骤")
_TROUBLESHOOTING_SEMANTIC_HEADINGS = (
    "问题现象", "现象", "目标", "排查步骤", "诊断步骤", "解决方案", "操作", "判断", "可能原因", "原因",
    "正常情况", "异常情况", "怎么处理", "怎么看结果", "注意事项", "方法一", "方法二", "方法三",
    "命令诊断", "物理检查", "软件层面", "硬件层面",
)
_CHINESE_SECTION_RE = re.compile(r"^(?P<num>[一二三四五六七八九十]+)[、.．]\s*(?P<title>\S.*)$")
_PROCEDURE_STEP_RE = re.compile(r"^(?:(第(?P<num>[一二三四五六七八九十0-9]+)步)|(?P<arabic>\d+[.．、:：)]))\s*(?P<title>\S.*)$")
DEFAULT_KG_V2_SOURCE_MANIFEST = "data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json"


@dataclass(frozen=True)
class DocStrategySpec:
    strategy_id: str
    summary: str
    section_unit: str
    kg_output_mode: str
    kg_background_mode: str
    llm_mode: str
    parser_hint: str


DOC_STRATEGY_SPECS: dict[str, DocStrategySpec] = {
    "sop_fault_catalog_doc": DocStrategySpec(
        strategy_id="sop_fault_catalog_doc",
        summary="多主题 SOP；按叶子故障章节生成独立原子 case，禁止整篇文档共用一个 Family/Variant。",
        section_unit="故障叶子章节 / 带操作的异常章节",
        kg_output_mode="atomic_case_bundle",
        kg_background_mode="existing_family_catalog_with_section_local_mapping",
        llm_mode="optional_atomic_case_semantic_review",
        parser_hint="sop_numbered_fault_catalog",
    ),
    "fault_manual_numbered": DocStrategySpec(
        strategy_id="fault_manual_numbered",
        summary="按“情况 N”或编号故障场景切 section，每个 section 生成 fault section_case。",
        section_unit="情况 N / 故障编号项",
        kg_output_mode="variant_case_bundle",
        kg_background_mode="light_family_catalog_for_mapping_only",
        llm_mode="optional_leaf_mapping_only",
        parser_hint="numbered_fault_manual",
    ),
    "troubleshooting_topic_doc": DocStrategySpec(
        strategy_id="troubleshooting_topic_doc",
        summary="单专题故障指南；保留专题结构，把诊断/解决/预防拆成不同 section_kind。",
        section_unit="功能段 / 子标题段",
        kg_output_mode="family_support_bundle",
        kg_background_mode="light_family_and_action_catalog",
        llm_mode="optional_section_semantic_split",
        parser_hint="troubleshooting_numbered_doc",
    ),
    "repair_playbook_doc": DocStrategySpec(
        strategy_id="repair_playbook_doc",
        summary="按“方案一/方案二”或 playbook 步骤切分，生成 action playbook。",
        section_unit="方案 N / playbook block",
        kg_output_mode="playbook_bundle",
        kg_background_mode="light_action_role_catalog",
        llm_mode="optional_applicability_mapping",
        parser_hint="solution_playbook_doc",
    ),
    "procedure_doc": DocStrategySpec(
        strategy_id="procedure_doc",
        summary="安装/迁移/工具使用教程，进入 procedure library，不直接生成 fault variant。",
        section_unit="步骤块",
        kg_output_mode="procedure_library_only",
        kg_background_mode="none_for_structure_minimal_for_tags",
        llm_mode="not_required",
        parser_hint="procedure_steps_doc",
    ),
    "spec_doc": DocStrategySpec(
        strategy_id="spec_doc",
        summary="规格/技术要求文档，进入 reference constraints，不直接生成故障 case。",
        section_unit="约束块 / 参数表",
        kg_output_mode="reference_constraint_only",
        kg_background_mode="none",
        llm_mode="not_required",
        parser_hint="spec_constraints_doc",
    ),
    "document_index_doc": DocStrategySpec(
        strategy_id="document_index_doc",
        summary="目录/导航型文档；保留引用说明，不伪造为可执行步骤。",
        section_unit="索引项 / 引用说明",
        kg_output_mode="reference_constraint_only",
        kg_background_mode="none",
        llm_mode="not_required",
        parser_hint="document_index",
    ),
    "validation_checklist_doc": DocStrategySpec(
        strategy_id="validation_checklist_doc",
        summary="稳定性测试/验收清单文档，生成 trace template / validation policy。",
        section_unit="检查项 / 测试块",
        kg_output_mode="policy_template_only",
        kg_background_mode="light_action_catalog",
        llm_mode="optional_policy_grouping",
        parser_hint="validation_checklist_doc",
    ),
    "faq_doc": DocStrategySpec(
        strategy_id="faq_doc",
        summary="FAQ/QA 文档，优先生成问答 support snippets，不直接生成主 fault graph。",
        section_unit="问答对",
        kg_output_mode="faq_support_bundle",
        kg_background_mode="light_family_catalog_if_needed",
        llm_mode="optional_qa_normalization",
        parser_hint="faq_pairs_doc",
    ),
    "overlay_process_doc": DocStrategySpec(
        strategy_id="overlay_process_doc",
        summary="流程/职责/负责人文档，只生成 overlay，不直接生成故障节点。",
        section_unit="流程段 / 角色表",
        kg_output_mode="overlay_only",
        kg_background_mode="none",
        llm_mode="not_required",
        parser_hint="process_overlay_doc",
    ),
    "unclassified_doc": DocStrategySpec(
        strategy_id="unclassified_doc",
        summary="结构不稳定或文本不足，先做人工确认，不自动入图。",
        section_unit="manual_review_needed",
        kg_output_mode="review_only",
        kg_background_mode="none",
        llm_mode="optional_after_manual_scoping",
        parser_hint="manual_triage",
    ),
}


def _read_text_preview(path: Path, *, max_chars: int = 4000) -> str:
    ext = path.suffix.lower()
    if ext in _TEXT_EXTS and path.exists():
        return path.read_text(encoding="utf-8")[:max_chars]
    if ext in _DOC_EXTS:
        out = parse_document_evidence(path)
        return str(out.get("text_preview") or "")[:max_chars]
    return ""


def _read_doc_lines(path: Path) -> list[str]:
    ext = path.suffix.lower()
    if ext in _TEXT_EXTS and path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ext in {".xlsx", ".pptx"} and path.exists():
        return [
            line.strip()
            for line in read_source_text_lines(path)
            if line.strip()
        ]
    if ext != ".docx" or not path.exists():
        preview = _read_text_preview(path, max_chars=4000)
        return [line.strip() for line in re.split(r"[。！？!?]\s*|\n+", preview) if line.strip()]
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except Exception:
        preview = _read_text_preview(path, max_chars=4000)
        return [line.strip() for line in re.split(r"[。！？!?]\s*|\n+", preview) if line.strip()]
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        preview = _read_text_preview(path, max_chars=4000)
        return [line.strip() for line in re.split(r"[。！？!?]\s*|\n+", preview) if line.strip()]
    lines: list[str] = []
    for para in root.iter(f"{_OOXML_NS}p"):
        texts = [node.text or "" for node in para.iter(f"{_OOXML_NS}t")]
        line = _normalize("".join(texts))
        if line:
            lines.append(line)
    return lines


def _is_sop_doc(path: Path) -> bool:
    name = path.name
    lowered = name.lower()
    if re.search(r"non[-_\s]?sop", lowered, re.IGNORECASE) or "非sop" in lowered:
        return False
    return (
        name in W9_EXCLUDED_SOP_NAMES
        or "标准操作流程" in name
        or bool(_SOP_NAME_RE.search(lowered))
    )


def _normalize(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    value = str(text or "")
    return any(marker in value for marker in markers)


def _clean_md_inline(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value)
    value = value.replace("**", "").replace("__", "")
    value = value.replace("\\", "")
    value = value.lstrip("#").strip()
    return _normalize(value)


def _entered_doc_names_from_manifest(path: str | Path = DEFAULT_KG_V2_SOURCE_MANIFEST) -> set[str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for item in payload.get("sources", []):
        if not isinstance(item, dict):
            continue
        for key in ("source_path", "fallback_docx_path"):
            raw = str(item.get(key) or "").strip()
            if raw:
                out.add(Path(raw).name)
        title = str(item.get("title") or "").strip()
        if title:
            out.add(f"{title}.docx")
            out.add(f"{title}.md")
    return out


def _strategy_for(path: Path, preview: str) -> str:
    name = path.name
    text = _normalize(preview)
    if path.name == "现场问题反馈流程.md":
        return "overlay_process_doc"
    if _is_sop_doc(path):
        return "sop_fault_catalog_doc"
    if _contains_any(name, ("手册",)) and _contains_any(text, ("情况 1", "情况 2", "排查步骤", "一般原因")):
        return "fault_manual_numbered"
    if _contains_any(name, ("Windows系统_引导修复", "关闭快速启动", "内存检测.docx", "可以进系统.docx", "如何进入安全模式")):
        return "document_index_doc"
    if (
        _contains_any(name.upper(), ("MEMORY.DMP", "DMP")) and _contains_any(name, ("分析", "方法"))
    ):
        return "procedure_doc"
    if "FAQ" in name or _contains_any(text, ("常见问题", "FAQ", "Q&A", "问题：", "回答：")):
        return "faq_doc"
    if _contains_any(text, ("问题现象", "可能原因", "排查步骤")):
        return "troubleshooting_topic_doc"
    if _contains_any(name, ("技术要求", "规范")) or _contains_any(text, ("容量要求", "接口类型", "尺寸规格", "可靠性指标", "使用限制", "技术要求", "接线规范")):
        return "spec_doc"
    if _contains_any(name, ("解决方案",)) or _contains_any(text, ("方案一", "方案二", "方案三", "方案四", "方案五")):
        return "repair_playbook_doc"
    if _contains_any(name, ("问题处理指南", "指南")) or _contains_any(text, ("文档目的", "常见原因分类", "诊断步骤", "解决方案", "预防措施", "常见误区")):
        return "troubleshooting_topic_doc"
    if _contains_any(name, ("稳定性测试", "讨论")) or _contains_any(text, ("理想结果", "压力测试", "连续运行", "稳定性验证")):
        return "validation_checklist_doc"
    if _contains_any(name, ("使用文档", "教程", "迁移", "加装内存", "更新教程", "方法", "修复", "关闭", "安装", "卸载", "进入", "数据采集")) or _contains_any(text, ("准备工作", "第一步", "第二步", "打开机箱", "进入BIOS", "下载", "解压", "重启电脑", "点击“", "步骤1", "磁盘管理器", "安全模式", "Windows更新", "驱动器号", "导出", "收集日志")):
        return "procedure_doc"
    if _contains_any(name, ("无法", "不开机", "卡顿", "无Internet", "无响应", "打不开")) or _contains_any(text, ("无法进入系统", "系统文件修复", "引导修复", "内存检测", "转储文件", "重装显卡驱动", "随机按键", "无法上网")):
        return "troubleshooting_topic_doc"
    return "unclassified_doc"


class RawDocIngestAgent:
    """W9: build an executable processing checklist for raw knowledge docs."""

    agent_id = "W9"

    def inspect_document(self, path: str | Path) -> dict[str, Any]:
        doc_path = Path(path)
        preview = _read_text_preview(doc_path)
        structure_supported = doc_path.suffix.lower() in _FULL_STRUCTURE_EXTS
        if structure_supported and doc_path.suffix.lower() in {
            ".xlsx", ".pptx"
        }:
            preview = "\n".join(_read_doc_lines(doc_path))[:4000]
        strategy_id = (
            _strategy_for(doc_path, preview)
            if structure_supported
            else "unclassified_doc"
        )
        spec = DOC_STRATEGY_SPECS[strategy_id]
        return {
            "type": "W9DocInspection",
            "agent_id": self.agent_id,
            "path": str(doc_path),
            "name": doc_path.name,
            "extension": doc_path.suffix.lower(),
            "excluded_from_w9": _is_sop_doc(doc_path),
            "structure_parse_status": (
                "supported" if structure_supported else "review_only"
            ),
            "strategy": asdict(spec),
            "text_preview": preview[:2000],
            "recommended_steps": self._steps_for_strategy(strategy_id, doc_path),
        }

    def build_structured_sections(self, path: str | Path) -> dict[str, Any]:
        doc_path = Path(path)
        lines = _read_doc_lines(doc_path)
        preview = "\n".join(lines)[:4000] or _read_text_preview(doc_path)
        strategy_id = (
            _strategy_for(doc_path, preview)
            if doc_path.suffix.lower() in _FULL_STRUCTURE_EXTS
            else "unclassified_doc"
        )
        spec = DOC_STRATEGY_SPECS[strategy_id]
        if strategy_id in {"troubleshooting_topic_doc", "sop_fault_catalog_doc"}:
            sections = self._structured_sections_troubleshooting(doc_path, lines)
        elif strategy_id == "repair_playbook_doc":
            sections = self._structured_sections_repair_playbook(doc_path, lines)
        elif strategy_id == "fault_manual_numbered":
            sections = self._structured_sections_fault_manual(doc_path, lines)
        elif strategy_id == "procedure_doc":
            sections = self._structured_sections_procedure(doc_path, lines)
        elif strategy_id == "spec_doc":
            sections = self._structured_sections_spec(doc_path, lines)
        elif strategy_id == "document_index_doc":
            sections = self._structured_sections_spec(doc_path, lines)
        elif strategy_id == "validation_checklist_doc":
            sections = self._structured_sections_validation(doc_path, lines)
        elif strategy_id == "faq_doc":
            sections = self._structured_sections_faq(doc_path, lines)
        elif strategy_id == "overlay_process_doc":
            sections = self._structured_sections_overlay(doc_path, lines)
        else:
            sections = self._structured_sections_unclassified(doc_path, lines)
        return {
            "type": "W9StructuredSections",
            "agent_id": self.agent_id,
            "path": str(doc_path),
            "name": doc_path.name,
            "strategy": asdict(spec),
            "line_count": len(lines),
            "document_links": extract_docx_hyperlinks(doc_path),
            "structured_sections": sections,
        }

    def build_section_cases(self, path: str | Path) -> dict[str, Any]:
        structured = self.build_structured_sections(path)
        strategy_id = str(((structured.get("strategy") or {}).get("strategy_id")) or "")
        sections = [row for row in structured.get("structured_sections") or [] if isinstance(row, dict)]
        if strategy_id == "sop_fault_catalog_doc":
            section_cases = self._section_cases_sop(Path(path), sections)
        elif strategy_id == "troubleshooting_topic_doc":
            section_cases = self._section_cases_troubleshooting(Path(path), sections)
        elif strategy_id == "repair_playbook_doc":
            section_cases = self._section_cases_repair_playbook(Path(path), sections)
        elif strategy_id == "fault_manual_numbered":
            section_cases = self._section_cases_fault_manual(Path(path), sections)
        elif strategy_id == "procedure_doc":
            section_cases = self._section_cases_procedure(Path(path), sections)
        elif strategy_id == "spec_doc":
            section_cases = self._section_cases_spec(Path(path), sections)
        elif strategy_id == "document_index_doc":
            section_cases = self._section_cases_spec(Path(path), sections)
        elif strategy_id == "validation_checklist_doc":
            section_cases = self._section_cases_validation(Path(path), sections)
        elif strategy_id == "faq_doc":
            section_cases = self._section_cases_faq(Path(path), sections)
        elif strategy_id == "overlay_process_doc":
            section_cases = self._section_cases_overlay(Path(path), sections)
        else:
            section_cases = self._section_cases_unclassified(Path(path), sections)
        chunk_manifest = build_staged_chunk_manifest(
            path,
            sections,
            source_doc_title=str(structured.get("name") or Path(path).name),
        )
        return {
            "type": "W9SectionCases",
            "agent_id": self.agent_id,
            "path": str(path),
            "name": Path(path).name,
            "strategy": structured.get("strategy") or {},
            "document_links": list(structured.get("document_links") or []),
            "structured_sections": sections,
            "section_cases": section_cases,
            "chunk_manifest": chunk_manifest,
        }

    def write_doc_outputs(self, path: str | Path, out_dir: str | Path) -> dict[str, Any]:
        payload = self.build_section_cases(path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        structured_path = out / "structured_sections.json"
        cases_path = out / "section_cases.json"
        chunk_manifest_path = out / "chunk_manifest.json"
        structured_path.write_text(json.dumps({
            "type": payload["type"],
            "agent_id": payload["agent_id"],
            "path": payload["path"],
            "name": payload["name"],
            "strategy": payload["strategy"],
            "document_links": payload.get("document_links") or [],
            "structured_sections": payload["structured_sections"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cases_path.write_text(json.dumps({
            "type": payload["type"],
            "agent_id": payload["agent_id"],
            "path": payload["path"],
            "name": payload["name"],
            "strategy": payload["strategy"],
            "document_links": payload.get("document_links") or [],
            "structured_sections": payload["structured_sections"],
            "section_cases": payload["section_cases"],
            "chunk_manifest": payload["chunk_manifest"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        chunk_manifest_path.write_text(
            json.dumps(payload["chunk_manifest"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "type": "W9DocBuildResult",
            "agent_id": self.agent_id,
            "path": str(path),
            "strategy": payload["strategy"],
            "counts": {
                "structured_sections": len(payload["structured_sections"]),
                "section_cases": len(payload["section_cases"]),
                "chunks": len(payload["chunk_manifest"].get("chunks") or []),
            },
            "output_files": {
                "structured_sections": str(structured_path),
                "section_cases": str(cases_path),
                "chunk_manifest": str(chunk_manifest_path),
            },
        }

    def build_root_checklist(self, root: str | Path, *, include_sop: bool = False) -> dict[str, Any]:
        root_path = Path(root)
        rows: list[dict[str, Any]] = []
        for path in sorted(root_path.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _TEXT_EXTS | _DOC_EXTS:
                continue
            if not include_sop and _is_sop_doc(path):
                continue
            rows.append(self.inspect_document(path))
        counts: dict[str, int] = {}
        for row in rows:
            strategy_id = str(((row.get("strategy") or {}).get("strategy_id")) or "unknown")
            counts[strategy_id] = counts.get(strategy_id, 0) + 1
        return {
            "type": "W9DocStrategyChecklist",
            "agent_id": self.agent_id,
            "root": str(root_path),
            "include_sop": include_sop,
            "counts_by_strategy": counts,
            "documents": rows,
        }

    def build_not_entered_docs(
        self,
        root: str | Path,
        *,
        manifest_path: str | Path = DEFAULT_KG_V2_SOURCE_MANIFEST,
        include_sop: bool = False,
        out_root: str | Path | None = None,
    ) -> dict[str, Any]:
        root_path = Path(root)
        entered = _entered_doc_names_from_manifest(manifest_path)
        documents: list[dict[str, Any]] = []
        counts_by_strategy: dict[str, int] = {}
        total_cases = 0
        total_sections = 0
        output_dirs: dict[str, str] = {}
        for path in sorted(root_path.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _TEXT_EXTS | _DOC_EXTS:
                continue
            if not include_sop and _is_sop_doc(path):
                continue
            if path.name in entered:
                continue
            inspection = self.inspect_document(path)
            strategy_id = str(((inspection.get("strategy") or {}).get("strategy_id")) or "unknown")
            counts_by_strategy[strategy_id] = counts_by_strategy.get(strategy_id, 0) + 1
            record = dict(inspection)
            if out_root is not None:
                slug_base = re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem).strip("_") or "doc"
                slug = f"{slug_base}_{hashlib.sha1(path.name.encode('utf-8')).hexdigest()[:8]}"
                doc_out = Path(out_root) / slug
                build = self.write_doc_outputs(path, doc_out)
                record["build"] = build
                output_dirs[path.name] = str(doc_out)
                total_sections += int(build["counts"]["structured_sections"])
                total_cases += int(build["counts"]["section_cases"])
            documents.append(record)
        return {
            "type": "W9NotEnteredDocsBuild",
            "agent_id": self.agent_id,
            "root": str(root_path),
            "manifest_path": str(manifest_path),
            "entered_doc_names": sorted(entered),
            "counts_by_strategy": counts_by_strategy,
            "documents": documents,
            "summary": {
                "doc_count": len(documents),
                "structured_sections": total_sections,
                "section_cases": total_cases,
            },
            "output_dirs": output_dirs,
        }

    def _steps_for_strategy(self, strategy_id: str, path: Path) -> list[str]:
        common = [
            "Phase A: safe parse raw document into bounded preview / lines",
            "Phase B: build structured_sections with document-specific section_strategy",
            "Phase C: map structured_sections into section_case objects",
            "Phase D: map section_case into kg_v2 bundle",
        ]
        specific = {
            "fault_manual_numbered": [
                f"split `{path.name}` by `情况 N` blocks",
                "extract `表现 / 排查步骤 / 一般原因` fields",
                "emit fault variants and ordered diagnostic actions",
            ],
            "troubleshooting_topic_doc": [
                f"split `{path.name}` into topic sections such as purpose / causes / diagnosis / solutions / cautions",
                "treat thresholds, cautions, and preventive notes as support knowledge, not fault variants",
                "emit family-scoped support bundle plus action templates",
            ],
            "repair_playbook_doc": [
                f"split `{path.name}` by `方案 N` or equivalent playbook blocks",
                "emit action playbooks and applicability conditions",
                "do not auto-create multiple fault variants unless the document explicitly separates fault modes",
            ],
            "procedure_doc": [
                f"treat `{path.name}` as procedure library entry",
                "extract ordered steps, preconditions, and tool references",
                "do not auto-write main fault KG nodes",
            ],
            "spec_doc": [
                f"treat `{path.name}` as reference constraint doc",
                "extract thresholds / compatibility / limits",
                "emit reference-only support objects",
            ],
            "document_index_doc": [
                f"treat `{path.name}` as a navigation/reference index",
                "preserve referenced document names and guidance as reference evidence",
                "do not manufacture procedure steps from directory rows",
            ],
            "validation_checklist_doc": [
                f"treat `{path.name}` as validation checklist",
                "extract test items and expected pass conditions",
                "emit policy / trace templates instead of fault variants",
            ],
            "faq_doc": [
                f"split `{path.name}` into QA pairs if the raw source is usable",
                "fallback to manual review if preview text is too weak",
                "emit support snippets only",
            ],
            "overlay_process_doc": [
                f"treat `{path.name}` as overlay source",
                "extract process roles / escalation ownership only",
                "do not emit fault graph objects",
            ],
            "unclassified_doc": [
                f"manual triage required for `{path.name}`",
                "decide whether it is procedure/spec/playbook/faq before KG mapping",
            ],
        }.get(strategy_id, [])
        return [*common, *specific]

    def _structured_sections_troubleshooting(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        path_stack: list[tuple[int, str, str]] = []
        preamble: list[str] = []
        for line in lines:
            match = _SECTION_HEADING_RE.match(line)
            if match:
                if current is not None:
                    sections.append(current)
                num = match.group("num")
                heading_title = match.group("title").strip().lstrip("-—– ")
                level = num.count(".") + 1
                path_stack = [item for item in path_stack if item[0] < level]
                path_stack.append((level, num, heading_title))
                current = {
                    "section_id": f"doc:{path.stem}:sec:{num}",
                    "heading_number": num,
                    "level": level,
                    "section_title": heading_title,
                    "path_titles": [item[2] for item in path_stack],
                    "section_kind": self._troubleshooting_section_kind(
                        " / ".join([item[2] for item in path_stack]),
                        num,
                    ),
                    "body_lines": [],
                }
                continue
            clean = _normalize(line)
            if not clean:
                continue
            if current is None:
                preamble.append(clean)
            else:
                current["body_lines"].append(clean)
        if current is not None:
            sections.append(current)
        if not sections:
            semantic = self._structured_sections_troubleshooting_semantic(path, lines)
            if semantic:
                return semantic
        if preamble:
            sections.insert(0, {
                "section_id": f"doc:{path.stem}:sec:intro",
                "heading_number": "0",
                "level": 0,
                "section_title": title,
                "path_titles": [title],
                "section_kind": "doc_intro",
                "body_lines": preamble,
            })
        return sections

    def _structured_sections_troubleshooting_semantic(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        intro: list[str] = []
        current: dict[str, Any] | None = None
        for raw_line in lines:
            line = _normalize(raw_line)
            if not line or line.replace("_", "/") == title.replace("_", "/"):
                continue
            heading = next((
                marker
                for marker in _TROUBLESHOOTING_SEMANTIC_HEADINGS
                if line == marker
                or line.startswith(f"{marker}：")
                or line.startswith(f"{marker}:")
                or line.startswith(f"{marker}（")
                or (marker in {"现象", "目标"} and line.startswith(marker))
            ), "")
            if heading:
                if current is not None:
                    sections.append(current)
                tail = line[len(heading):].lstrip("：: ")
                current = {
                    "section_id": f"doc:{path.stem}:semantic:{len(sections)+1}",
                    "heading_number": str(len(sections) + 1),
                    "level": 1,
                    "section_title": heading,
                    "path_titles": [title, heading],
                    "section_kind": self._troubleshooting_section_kind(heading, str(len(sections) + 1)),
                    "body_lines": [tail] if tail else [],
                }
                continue
            if current is None:
                intro.append(line)
            else:
                current["body_lines"].append(line)
        if current is not None:
            sections.append(current)
        if intro:
            sections.insert(0, {
                "section_id": f"doc:{path.stem}:semantic:intro",
                "heading_number": "0",
                "level": 0,
                "section_title": title,
                "path_titles": [title],
                "section_kind": "doc_intro",
                "body_lines": intro,
            })
        return sections

    def _structured_sections_repair_playbook(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        preamble: list[str] = []
        current: dict[str, Any] | None = None
        for raw_line in lines:
            line = _normalize(raw_line)
            if not line:
                continue
            match = _PLAYBOOK_HEADING_RE.match(line)
            if match:
                if current is not None:
                    sections.append(current)
                current = {
                    "section_id": f"doc:{path.stem}:scheme:{match.group('num')}",
                    "heading_number": str(match.group("num")),
                    "level": 1,
                    "section_title": str(match.group("title")).strip(),
                    "path_titles": [title, str(match.group("title")).strip()],
                    "section_kind": "solution_playbook",
                    "body_lines": [],
                }
                continue
            if current is None:
                preamble.append(line)
            else:
                current["body_lines"].append(line)
        if current is not None:
            sections.append(current)
        if preamble:
            sections.insert(0, {
                "section_id": f"doc:{path.stem}:intro",
                "heading_number": "0",
                "level": 0,
                "section_title": title,
                "path_titles": [title],
                "section_kind": "doc_intro",
                "body_lines": preamble,
            })
        return sections

    def _structured_sections_fault_manual(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        mode = ""
        deep_dive = False
        intro_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current
            if current is not None:
                sections.append(current)
                current = None

        def new_section(*, number: str, section_title: str, section_kind: str, level: int) -> dict[str, Any]:
            return {
                "section_id": f"doc:{path.stem}:{'component' if section_kind == 'cause_support' else 'case'}:{number}",
                "heading_number": number,
                "level": level,
                "section_title": section_title,
                "path_titles": [title, "主要原因深度排查", section_title]
                if section_kind == "cause_support"
                else [title, section_title],
                "section_kind": section_kind,
                "body_lines": [],
                "symptom_lines": [],
                "action_lines": [],
                "cause_lines": [],
                "reasoning_lines": [],
            }

        for raw_line in lines:
            line = _clean_md_inline(raw_line)
            if not line:
                continue
            if line == "主要原因深度排查":
                flush_current()
                deep_dive = True
                mode = ""
                sections.append({
                    "section_id": f"doc:{path.stem}:deep-dive",
                    "heading_number": "deep-dive",
                    "level": 1,
                    "section_title": line,
                    "path_titles": [title, line],
                    "section_kind": "cause_overview",
                    "body_lines": [],
                })
                continue

            match = _FAULT_MANUAL_CASE_RE.match(line)
            if match is None and not deep_dive:
                match = _FAULT_MANUAL_NUMBERED_CASE_RE.match(line)
            if match:
                flush_current()
                current = new_section(
                    number=str(match.group("num")),
                    section_title=str(match.group("title")).strip(),
                    section_kind="fault_case",
                    level=1,
                )
                mode = ""
                continue

            component = _FAULT_MANUAL_COMPONENT_RE.match(line) if deep_dive else None
            if component:
                flush_current()
                current = new_section(
                    number=str(component.group("num")),
                    section_title=str(component.group("title")).strip(),
                    section_kind="cause_support",
                    level=2,
                )
                mode = ""
                continue
            if current is None:
                intro_lines.append(line)
                continue
            current["body_lines"].append(line)
            marker_line = _BULLET_RE.sub("", line).strip()
            if marker_line.startswith("表现："):
                mode = "symptom"
                tail = marker_line.split("：", 1)[1].strip()
                if tail:
                    current["symptom_lines"].append(tail)
                continue
            if marker_line.startswith("排查思路："):
                mode = "reasoning"
                tail = marker_line.split("：", 1)[1].strip()
                if tail:
                    current["reasoning_lines"].append(tail)
                continue
            if marker_line.startswith("排查步骤：") or marker_line == "排查：":
                mode = "action"
                tail = marker_line.split("：", 1)[1].strip()
                if tail:
                    current["action_lines"].append(tail)
                continue
            if marker_line.startswith("一般原因："):
                mode = "cause"
                tail = marker_line.split("：", 1)[1].strip()
                if tail:
                    current["cause_lines"].append(tail)
                continue
            clean = _BULLET_RE.sub("", marker_line).strip()
            if not clean:
                continue
            if mode == "symptom":
                current["symptom_lines"].append(clean)
            elif mode == "action":
                current["action_lines"].append(clean)
            elif mode == "cause":
                current["cause_lines"].append(clean)
            elif mode == "reasoning":
                current["reasoning_lines"].append(clean)
        flush_current()
        if intro_lines:
            sections.insert(0, {
                "section_id": f"doc:{path.stem}:intro",
                "heading_number": "0",
                "level": 0,
                "section_title": title,
                "path_titles": [title],
                "section_kind": "doc_intro",
                "body_lines": intro_lines,
            })
        return sections

    def _structured_sections_procedure(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = {
            "section_id": f"doc:{path.stem}:intro",
            "heading_number": "0",
            "level": 0,
            "section_title": title,
            "path_titles": [title],
            "section_kind": "procedure_intro",
            "body_lines": [],
        }
        step_index = 0
        for raw_line in lines:
            line = _clean_md_inline(raw_line)
            if not line:
                continue
            if (
                line.replace("_", "/") == title.replace("_", "/")
                and current is not None
                and not current["body_lines"]
            ):
                continue
            match = _PROCEDURE_STEP_RE.match(line)
            if match:
                if current is not None and current["body_lines"]:
                    sections.append(current)
                step_index += 1
                current = {
                    "section_id": f"doc:{path.stem}:step:{step_index}",
                    "heading_number": str(step_index),
                    "level": 1,
                    "section_title": match.group("title").strip(),
                    "path_titles": [title, match.group("title").strip()],
                    "section_kind": "procedure_step",
                    "body_lines": [],
                }
                continue
            if current is None:
                continue
            current["body_lines"].append(line)
        if current is not None and current["body_lines"]:
            sections.append(current)
        return sections

    def _structured_sections_spec(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        body = [_clean_md_inline(line) for line in lines if _clean_md_inline(line)]
        if not body:
            return []
        return [{
            "section_id": f"doc:{path.stem}:spec:1",
            "heading_number": "1",
            "level": 1,
            "section_title": title,
            "path_titles": [title],
            "section_kind": "spec_constraints",
            "body_lines": body[1:] if len(body) > 1 else body,
        }]

    def _structured_sections_validation(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for raw_line in lines:
            line = _clean_md_inline(raw_line)
            if not line:
                continue
            match = _CHINESE_SECTION_RE.match(line)
            if match:
                if current is not None:
                    sections.append(current)
                current = {
                    "section_id": f"doc:{path.stem}:check:{match.group('num')}",
                    "heading_number": str(match.group("num")),
                    "level": 1,
                    "section_title": match.group("title").strip(),
                    "path_titles": [title, match.group("title").strip()],
                    "section_kind": "validation_block",
                    "body_lines": [],
                }
                continue
            if current is None:
                current = {
                    "section_id": f"doc:{path.stem}:intro",
                    "heading_number": "0",
                    "level": 0,
                    "section_title": title,
                    "path_titles": [title],
                    "section_kind": "validation_block",
                    "body_lines": [],
                }
            current["body_lines"].append(line)
        if current is not None and current["body_lines"]:
            sections.append(current)
        return sections

    def _structured_sections_faq(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        body = [_clean_md_inline(line) for line in lines if _clean_md_inline(line)]
        if not body:
            return []
        questions = [line for line in body if ("？" in line or "?" in line or "问题" in line)]
        if not questions:
            return [{
                "section_id": f"doc:{path.stem}:faq:review",
                "heading_number": "0",
                "level": 0,
                "section_title": title,
                "path_titles": [title],
                "section_kind": "faq_review_needed",
                "body_lines": body[:20],
            }]
        sections: list[dict[str, Any]] = []
        for idx, line in enumerate(questions[:20], start=1):
            sections.append({
                "section_id": f"doc:{path.stem}:faq:{idx}",
                "heading_number": str(idx),
                "level": 1,
                "section_title": line,
                "path_titles": [title, line],
                "section_kind": "faq_pair",
                "body_lines": [],
            })
        return sections

    def _structured_sections_overlay(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        title = path.stem
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for raw_line in lines:
            line = _clean_md_inline(raw_line)
            if not line:
                continue
            if line.startswith("# "):
                if current is not None:
                    sections.append(current)
                heading = line.lstrip("#").strip()
                current = {
                    "section_id": f"doc:{path.stem}:overlay:{len(sections)+1}",
                    "heading_number": str(len(sections) + 1),
                    "level": 1,
                    "section_title": heading,
                    "path_titles": [title, heading],
                    "section_kind": "overlay_block",
                    "body_lines": [],
                }
                continue
            if current is None:
                current = {
                    "section_id": f"doc:{path.stem}:overlay:intro",
                    "heading_number": "0",
                    "level": 0,
                    "section_title": title,
                    "path_titles": [title],
                    "section_kind": "overlay_block",
                    "body_lines": [],
                }
            current["body_lines"].append(line)
        if current is not None and current["body_lines"]:
            sections.append(current)
        return sections

    def _structured_sections_unclassified(self, path: Path, lines: list[str]) -> list[dict[str, Any]]:
        body = [_clean_md_inline(line) for line in lines if _clean_md_inline(line)]
        if not body:
            return []
        return [{
            "section_id": f"doc:{path.stem}:triage:1",
            "heading_number": "1",
            "level": 1,
            "section_title": path.stem,
            "path_titles": [path.stem],
            "section_kind": "manual_review_needed",
            "body_lines": body[:30],
        }]

    def _troubleshooting_section_kind(self, title: str, heading_number: str) -> str:
        text = f"{heading_number} {title}"
        if _contains_any(text, ("文档目的",)):
            return "doc_intro"
        if _contains_any(text, ("正常CPU温度范围", "温度范围参考")):
            return "threshold_reference"
        if _contains_any(text, ("常见原因分类", "硬件原因", "环境原因")):
            return "cause_cluster"
        if _contains_any(text, ("解决方案", "软件/系统层面", "硬件操作", "极端情况")):
            return "solution_playbook"
        if _contains_any(text, ("诊断步骤", "确认温度读数", "观察现象", "排查", "物理检查", "命令诊断", "操作")):
            return "diagnostic_actions"
        if _contains_any(text, ("预防措施",)):
            return "preventive_note"
        if _contains_any(text, ("误区", "澄清")):
            return "operator_caution"
        return "support_note"

    def _section_cases_troubleshooting(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        family_scope = self._troubleshooting_family_scope(path, sections)
        out: list[dict[str, Any]] = []
        for section in sections:
            kind = str(section.get("section_kind") or "")
            title = str(section.get("section_title") or "")
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            if kind == "doc_intro":
                continue
            if not lines:
                continue
            case = {
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": title,
                "section_case_kind": kind,
                "family_scope_candidates": family_scope,
                "variant_candidate": "",
                "actions": [],
                "procedure_steps": [],
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": [],
                "output_mode": "family_support_bundle",
                "fault_mapping_allowed": False,
            }
            if kind == "threshold_reference":
                case["thresholds"] = lines
            elif kind == "cause_cluster":
                case["cause_notes"] = lines
            elif kind == "diagnostic_actions":
                action_payload = self._action_payload(lines)
                if action_payload.get("actions"):
                    case.update(action_payload)
                elif self._is_action_heading(title):
                    case.update({
                        "actions": [title],
                        "procedure_steps": [{
                            "step_order": 1,
                            "label": title.rstrip("：:"),
                            "instruction": lines[0] if lines else title,
                            "details": lines[1:],
                            "source_lines": [title, *lines],
                        }],
                    })
                case["required_info"] = self._diagnostic_required_info(lines)
                case["variant_candidate"] = self._troubleshooting_variant(path, title)
                case["fault_mapping_allowed"] = True
            elif kind == "solution_playbook":
                case.update(self._action_payload(lines))
                case["required_info"] = self._solution_required_info(lines)
                case["variant_candidate"] = self._troubleshooting_variant(path, title)
                case["fault_mapping_allowed"] = True
            elif kind == "preventive_note":
                case["support_notes"] = lines
                case["output_mode"] = "reference_constraint_only"
            elif kind == "operator_caution":
                case["support_notes"] = lines
                case["output_mode"] = "reference_constraint_only"
            else:
                case["support_notes"] = lines
                case["output_mode"] = "reference_constraint_only"
            out.append(case)
        if not out:
            intro_lines: list[str] = []
            for section in sections:
                if str(section.get("section_kind") or "") == "doc_intro":
                    intro_lines.extend(str(x) for x in section.get("body_lines") or [] if str(x).strip())
            if intro_lines:
                out.append({
                    "case_id": f"doc:{path.stem}:fallback:case",
                    "source_doc_id": f"doc:{path.stem}",
                    "source_doc_title": path.stem,
                    "section_id": f"doc:{path.stem}:fallback",
                    "section_title": path.stem,
                    "section_case_kind": "troubleshooting_fallback",
                    "family_scope_candidates": family_scope,
                    "variant_candidate": path.stem,
                    "actions": self._to_actions(intro_lines),
                    "procedure_steps": self._to_procedure_steps(intro_lines),
                    "required_info": [],
                    "thresholds": [],
                    "cause_notes": [],
                    "support_notes": intro_lines[:20],
                    "output_mode": "review_only",
                })
        return out

    def _section_cases_sop(
        self,
        path: Path,
        sections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Turn a multi-topic SOP into section-local atomic fault cases.

        A SOP is a catalogue, not one troubleshooting topic.  The generic
        troubleshooting adapter used to scan the whole document for a family
        marker and consequently attached every action to the first matching
        family.  Here family and variant identity are derived exclusively from
        the current heading path, and only leaf sections with both a fault
        signal and at least one review-grade operation may enter fault mapping.
        """

        # DOCX tables of contents repeat heading ids before the real body.  A
        # content-bearing occurrence is authoritative for semantic extraction.
        by_section_id: dict[str, dict[str, Any]] = {}
        section_order: list[str] = []
        for section in sections:
            section_id = str(section.get("section_id") or "")
            if not section_id:
                continue
            if section_id not in by_section_id:
                section_order.append(section_id)
            current = by_section_id.get(section_id)
            if current is None or (
                not current.get("body_lines") and section.get("body_lines")
            ):
                by_section_id[section_id] = section

        out: list[dict[str, Any]] = []
        previous_context_lines: list[str] = []
        for section_id in section_order:
            section = by_section_id[section_id]
            title = str(section.get("section_title") or "").strip()
            lines = [
                str(value).strip()
                for value in section.get("body_lines") or []
                if str(value).strip()
            ]
            if not lines:
                continue
            path_titles = [
                str(value).strip()
                for value in section.get("path_titles") or []
                if str(value).strip()
            ]
            action_payload = self._sop_action_payload(lines)
            fault_like = self._sop_fault_like(title, path_titles)
            mapping_allowed = bool(
                fault_like and action_payload.get("procedure_steps")
            )
            contextual_variant = self._sop_context_variant(
                title,
                path_titles,
                previous_context_lines,
            )
            semantic_path = [*path_titles, contextual_variant]
            family_scope = self._sop_family_scope(title, semantic_path)
            variant = (
                contextual_variant or self._sop_variant(title, semantic_path)
                if mapping_allowed
                else ""
            )
            out.append({
                "case_id": f"{section_id}:atomic-case",
                "atomic_case_id": f"{section_id}:atomic-case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section_id,
                "section_title": title,
                "source_path_titles": path_titles,
                "section_case_kind": (
                    "sop_atomic_fault_case" if mapping_allowed else "sop_support_section"
                ),
                "family_scope_candidates": [family_scope] if mapping_allowed else [],
                "variant_candidate": variant,
                "actions": list(action_payload.get("actions") or []) if mapping_allowed else [],
                "procedure_steps": list(action_payload.get("procedure_steps") or []) if mapping_allowed else [],
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": [] if mapping_allowed else lines,
                "output_mode": (
                    "atomic_case_bundle" if mapping_allowed else "reference_constraint_only"
                ),
                "fault_mapping_allowed": mapping_allowed,
            })
            if not mapping_allowed:
                previous_context_lines = lines
        return out

    def _sop_action_payload(self, lines: list[str]) -> dict[str, Any]:
        """Extract only explicit operations from one SOP leaf section."""

        normalized: list[str] = []
        explicit_inline_steps: list[str] = []
        for raw in lines:
            text = str(raw or "").strip()
            if not text:
                continue
            match = re.match(r"^(?:处理方法|操作方法|排查方法|排查步骤)[：:]\s*(.+)$", text)
            if match:
                text = str(match.group(1) or "").strip()
            segments = [item.strip() for item in re.split(r"[；;]+", text) if item.strip()]
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                if not self._looks_like_action_text(segment):
                    should = re.search(
                        r"(?:应|需要|建议|可尝试|可以尝试)\s*(检查|确认|清理|清洁|关闭|打开|设置|重启|升级|更换|删除|收集|查看|调整|安装|拔插|恢复|执行|联系)(.+)",
                        segment,
                    )
                    if should:
                        segment = f"{should.group(1)}{should.group(2)}".strip()
                normalized.append(segment)
                if len(segments) > 1 and self._looks_like_action_text(segment):
                    explicit_inline_steps.append(segment)
        if len(explicit_inline_steps) > 1:
            steps = [
                {
                    "step_order": index,
                    "label": value.rstrip("。；;"),
                    "instruction": value,
                    "details": [],
                    "source_lines": [value],
                }
                for index, value in enumerate(explicit_inline_steps, start=1)
            ]
            return self._finalize_sop_action_payload({
                "actions": [str(step["label"]) for step in steps],
                "procedure_steps": steps,
            })
        return self._finalize_sop_action_payload(self._action_payload(normalized))

    def _finalize_sop_action_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Split compound operations, discard incomplete fragments and dedupe."""

        split_pattern = re.compile(
            r"(?:，|,)?(?:然后|并)(?=(?:检查|查看|确认|打开|关闭|排除|验证|拔插|"
            r"清洁|收集|点击|设置|安装|卸载|更新|重装|联系|定位|观察|输入|使用|选择|运动))"
        )
        steps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_step in payload.get("procedure_steps") or []:
            if not isinstance(raw_step, dict):
                continue
            label = str(raw_step.get("label") or "").strip().rstrip("。；;")
            instruction = str(raw_step.get("instruction") or "").strip().rstrip("。；;")
            bracket_pairs = (("(", ")"), ("（", "）"), ("[", "]"), ("【", "】"))
            if (
                instruction
                and any(label.count(left) != label.count(right) for left, right in bracket_pairs)
                and not any(instruction.count(left) != instruction.count(right) for left, right in bracket_pairs)
            ):
                label = instruction
            if label.startswith("尝试") and len(label) > 2:
                label = label[2:].strip()
            if not label or re.search(r"(?:后|然后|并)\s*$", label):
                continue
            parts = [item.strip() for item in split_pattern.split(label) if item.strip()]
            for part in parts:
                key = "".join(part.lower().split())
                if not key or key in seen:
                    continue
                seen.add(key)
                item = dict(raw_step)
                item.update({
                    "step_order": len(steps) + 1,
                    "label": part,
                    "instruction": (
                        instruction or part
                        if len(parts) == 1
                        else part
                    ),
                    "details": list(raw_step.get("details") or []) if len(parts) == 1 else [],
                    "source_lines": list(raw_step.get("source_lines") or [part]),
                })
                steps.append(item)
        return {
            "actions": [str(step.get("label") or "") for step in steps],
            "procedure_steps": steps,
        }

    @staticmethod
    def _sop_context_variant(
        title: str,
        path_titles: list[str],
        previous_context_lines: list[str],
    ) -> str:
        """Recover a fault label when a parser emits a generic action heading."""

        generic = {
            "排查", "排查步骤", "处理方法", "操作方法", "处理思路",
            "解决方案", "操作步骤", "诊断步骤",
        }
        if str(title or "").strip().rstrip("：:") not in generic:
            return ""
        for raw in reversed(previous_context_lines):
            value = str(raw or "").strip().lstrip("# ")
            match = re.match(r"^(?:情况\s*\d+|问题现象|现象)[：:]\s*(.+)$", value)
            if match:
                candidate = str(match.group(1) or "").strip().rstrip("。；;：:")
                if candidate:
                    return candidate
        for candidate in reversed(path_titles[:-1]):
            value = str(candidate or "").strip().rstrip("：:")
            if value and value not in generic and "标准操作流程" not in value:
                return value
        return ""

    @staticmethod
    def _sop_fault_like(title: str, path_titles: list[str]) -> bool:
        semantic_path = [
            item
            for item in path_titles
            if "标准操作流程" not in item and item.strip() != "异常处理"
        ]
        text = " ".join([*semantic_path, title])
        generic_reference_markers = (
            "关于用户认证的一些知识",
            "不同器件如何做替代料",
            "发生问题时，不同的情况下如何收集数据",
            "工具使用方法",
            "接线方法",
            "清理流程",
        )
        if any(marker in title for marker in generic_reference_markers):
            return False
        fault_markers = (
            "失败", "异常", "报错", "错误", "无法", "打不开", "不允许",
            "卡死", "卡顿", "卡住", "缓慢", "超时", "丢帧", "残帧", "缺失",
            "模糊", "错位", "报警", "漏气", "黑屏", "蓝屏", "重启", "闪退",
            "不进板", "不到位", "未触发", "频闪", "爆满", "占用100%", "排查",
            "问题", "不流畅", "不准确", "不一致", "没有参考图",
            "断连",
        )
        return any(marker in text for marker in fault_markers)

    @staticmethod
    def _sop_family_scope(title: str, path_titles: list[str]) -> str:
        text = " ".join([
            *[
                item
                for item in path_titles
                if "标准操作流程" not in item and item.strip() != "异常处理"
            ],
            title,
        ])
        upper = text.upper()
        if "没有参考图" in title or "没图" in title:
            return "Buddy 模板缺失"
        if "2D" in upper and any(marker in title for marker in ("成像差异", "成像不一致")):
            return "2D成像一致性异常"
        if any(marker in title for marker in ("拼图出现错位", "拼图错位")):
            return "图像拼接错位"
        if "复判" in text or "复盘站" in text:
            if "加载板卡" in title or "板卡加载" in title:
                return "复判站加载板卡异常"
            if "保存结果失败" in title:
                return "复判保存结果失败"
            if any(marker in title for marker in ("加载数据慢", "出图慢", "TX时间超过")):
                return "复判站出图慢"
            if "BADMARK" in upper or "跳叉板" in title:
                return "坏板标记异常"
        if "加载板卡失败" in title:
            return "程序板卡加载失败"
        if "CAD" in upper and any(marker in title for marker in ("导入", "解析失败", "尺寸过大")):
            return "CAD 导入失败"
        if any(marker in title for marker in ("角度不统一", "角度不一致")):
            return "CAD 角度不一致"
        if "MARK" in upper and any(marker in title for marker in ("对齐", "跑偏", "遮挡")):
            return "Mark 点对齐失败"
        if "主程序和工厂程序均无法打开" in title or "主程序无法打开" in title:
            return "主程序无法打开"
        if "工厂程序无法打开" in title:
            return "工厂程序无法打开"
        if "模型加载失败" in title:
            return "用户配置加载失败"
        if "3D相机初始化失败" in title:
            return "相机初始化失败"
        if "初始化运动控制卡" in title:
            return "运控卡初始化异常"
        if "运控打不开" in title or "运动控制程序错误" in title:
            return "运控程序无法打开"
        if "拍摄失败" in title or "相机事件超时" in title or "残帧" in title:
            return "相机拍摄失败"
        if "拍照速度越来越慢" in title:
            return "CT 时间异常增加"
        if "成像模糊" in title:
            return "相机成像模糊"
        if "扫码枪" in title:
            return "扫码枪异常"
        if "皮带" in title:
            return "皮带运行异常"
        if "轨道宽度无法调节" in title:
            return "轨道宽度无法调节"
        if "轨道有板" in title or "进到轨道内一半" in title:
            return "进板失败"
        if "挡块" in title:
            return "挡块异常"
        if "出板失败" in title:
            return "出板失败"
        if any(marker in title for marker in ("气压过低", "漏气")):
            return "气压异常"
        if any(marker in title for marker in ("D盘", "C盘空间")):
            return "磁盘 I/O 异常"
        if "自动重启" in title:
            return "工控机异常重启"
        if any(marker in title for marker in ("卡死迟钝", "软件卡死")):
            return "软件卡死无响应"
        if "黑屏" in title:
            return "工控机黑屏无显示"
        if any(marker in title for marker in ("cuda", "CUDA", "Windows核心驱动")):
            return "CUDA 计算设备不可用"
        if "Press F1" in title or "SETUF" in title:
            return "BIOS 启动配置异常"
        if "光控通信" in title or "跟光控通信" in title:
            return "光控通信异常"
        if "光源问题" in title:
            return "光源异常"
        if any(marker in title for marker in ("运控", "运动控制", "伺服")):
            return "运控问题"
        if any(marker in title for marker in ("自动重启", "黑屏", "蓝屏", "任务管理器", "C盘", "D盘", "磁盘", "不进入系统", "cuda")):
            return "工控机/复判站/编程站及操作系统问题"
        if "模型加载失败" in title:
            return "主程序软件问题"
        if "光" in title and any(marker in title for marker in ("初始化", "通信", "光源问题", "光控")):
            return "光源初始化失败"
        if any(marker in text for marker in ("相机", "拍摄", "拍照", "残帧", "CXP", "成像", "拼图")):
            return "相机拍摄失败"
        if any(marker in text for marker in ("复判站", "复盘站", "复判报")):
            return "复判站软件问题"
        if "BUDDY" in text.upper() or "buddy" in text:
            return "Buddy问题"
        if any(marker in text for marker in ("CAD", "Mark", "mark", "OCR", "OCV", "检测框", "缺陷", "编程优化", "料号角度", "角度不统一")):
            return "模型优化问题"
        if any(marker in text for marker in ("扫码枪", "上下道", "MES")):
            return "外部对接设备"
        if any(marker in text for marker in ("轨道", "皮带", "传感器", "挡块", "顶升", "伺服", "气压", "运控", "进板", "出板", "卡板")):
            return "运控问题"
        if "标定" in text or "原点" in text:
            return "标定问题"
        if "SPC" in text.upper():
            return "SPC问题"
        if any(marker in text for marker in ("系统", "工控机", "电脑", "磁盘", "蓝屏", "黑屏", "显卡", "DMP")):
            return "工控机/复判站/编程站及操作系统问题"
        if any(marker in text for marker in ("主程序", "用户配置", "用户认证", "初始化失败")):
            return "主程序软件问题"
        return "软件使用及调试问题"

    @staticmethod
    def _sop_variant(title: str, path_titles: list[str]) -> str:
        value = str(title or "").strip().rstrip("：:。？?")
        generic = {
            "排查", "排查步骤", "处理方法", "操作方法", "处理思路",
            "解决方案", "操作步骤", "诊断步骤", "初始化失败", "设置",
            "检测", "系统", "相机",
        }
        if value in generic:
            parent = next(
                (
                    item.strip().rstrip("：:。？?")
                    for item in reversed(path_titles[:-1])
                    if item.strip() and item.strip() not in generic
                ),
                "",
            )
            if parent:
                value = f"{parent}-{value}"
        replacements = (
            ("复判站检测页面没有参考图，点击加载最新模板后也没图", "复判站参考图缺失"),
            ("2D设备同一光源下，器件成像差异较大", "2D同光源器件成像一致性异常"),
            ("拼图出现错位需要检查什么", "远近轨像素差异导致图像拼接错位"),
            ("光源问题", "光源初始化或连接异常"),
            ("拍摄大板时，相机拍照速度越来越慢", "大板拍摄速度异常下降"),
            ("设备成像模糊", "顶升不到位导致相机成像模糊"),
            ("扫码枪频闪", "扫码枪异常频闪"),
            ("黄色报警，气压过低", "气压过低报警"),
            ("跟光控通信相关的问题", "日志出现光控通信异常"),
            ("气压表位置漏气", "气压表漏气异常"),
            ("原点排查", "业务原点标定异常"),
            ("轨道宽度无法调节", "传感器或异物导致轨道宽度无法调节"),
            ("出板失败", "镂空或过长板卡出板失败"),
            (
                "编程测试页面出现卡死迟钝时，软件方面排查：设备正常测试中设备不进板，人工去操作鼠标时反应迟钝，软件卡死，任务管理器也弹不出来无法操作，断电重启之后正常测试",
                "软件卡死且任务管理器无响应",
            ),
        )
        for source, target in replacements:
            if value == source:
                return target
        return value

    @staticmethod
    def _troubleshooting_family_scope(path: Path, sections: list[dict[str, Any]]) -> list[str]:
        text = " ".join([
            path.stem,
            *[str(section.get("section_title") or "") for section in sections],
            *[
                str(line)
                for section in sections
                for line in section.get("body_lines") or []
            ],
        ])
        if any(marker in text for marker in ("拍照失败", "拍摄失败", "相机配置", "相机SDK")):
            return ["相机拍摄失败"]
        if "CPU" in text and any(marker in text for marker in ("温度", "过热", "散热")):
            return ["CPU温度异常"]
        if any(marker in text for marker in ("无法上网", "无Internet", "DNS故障", "网络中断")):
            return ["网络连接异常"]
        if any(marker in text for marker in ("键盘随机按键", "键盘无响应", "转接链路")):
            return ["键盘输入异常"]
        return ["主程序/系统异常"]

    @staticmethod
    def _is_action_heading(title: str) -> bool:
        text = str(title or "").strip()
        if text in {"排查步骤", "诊断步骤", "操作", "判断", "观察现象"}:
            return False
        return any(verb in text for verb in _PROCEDURE_ACTION_PREFIXES)

    @staticmethod
    def _troubleshooting_variant(path: Path, section_title: str) -> str:
        text = f"{path.stem} {section_title}"
        if "CPU" in text and any(marker in text for marker in ("温度", "过热", "散热", "硬件操作", "观察现象")):
            return "CPU温度异常升高"
        return path.stem

    def _section_cases_repair_playbook(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            if str(section.get("section_kind") or "") != "solution_playbook":
                continue
            title = str(section.get("section_title") or "")
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            if not lines:
                continue
            text = " ".join(lines)
            required: list[str] = []
            if _contains_any(text, ("设备管理器", "USB Root Hub", "USB主控制器")):
                required.append("请提供设备管理器中 USB 控制器列表和异常项截图。")
            if _contains_any(text, ("电源管理", "USB选择性暂停", "电源选项")):
                required.append("请提供当前 Windows 电源管理与 USB 节能设置截图。")
            if _contains_any(text, ("bios", "BIOS", "Load Optimized Defaults", "恢复出厂设置")):
                required.append("请提供 BIOS 当前关键设置和恢复默认后的复现结果。")
            if _contains_any(title, ("静电",)):
                required.append("请说明断电、释放余电后问题是否恢复。")
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": title,
                "section_case_kind": "solution_playbook",
                "family_scope_candidates": ["USB设备异常"],
                "variant_candidate": f"USB设备异常/{title}",
                **self._action_payload(lines),
                "required_info": required,
                "thresholds": [],
                "cause_notes": [],
                "support_notes": [line for line in lines if "非常常见" in title][:1],
                "playbook_condition": title,
                "output_mode": "playbook_bundle",
            })
        return out

    def _section_cases_fault_manual(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            section_kind = str(section.get("section_kind") or "")
            if section_kind not in {"fault_case", "cause_support"}:
                continue
            title = str(section.get("section_title") or "")
            symptom_lines = [str(x) for x in section.get("symptom_lines") or [] if str(x).strip()]
            action_lines = [str(x) for x in section.get("action_lines") or [] if str(x).strip()]
            cause_lines = [str(x) for x in section.get("cause_lines") or [] if str(x).strip()]
            reasoning_lines = [str(x) for x in section.get("reasoning_lines") or [] if str(x).strip()]
            if not symptom_lines and not action_lines and not cause_lines:
                continue
            text = " ".join([*symptom_lines, *action_lines, *cause_lines, *[str(x) for x in section.get("body_lines") or []]])
            required: list[str] = []
            if _contains_any(text, ("Debug 灯", "CPU 灯", "DRAM 灯", "VGA 灯", "BOOT 灯")):
                required.append("请提供主板 Debug 灯状态、亮灯顺序或现场照片。")
            if _contains_any(text, ("BIOS", "启动顺序", "硬盘")):
                required.append("请说明是否可进入 BIOS，以及 BIOS 中硬盘识别和启动顺序状态。")
            if _contains_any(text, ("显示器", "视频线", "DP", "HDMI", "VGA")):
                required.append("请提供显示器电源状态、视频线接口类型和连接位置。")
            if _contains_any(text, ("电源", "无通电", "风扇不转")):
                required.append("请说明电源线、插座供电、指示灯和风扇是否有反应。")
            if _contains_any(text, ("内存", "DRAM")):
                required.append("请说明内存条插槽位置、单条测试结果和 DRAM 灯状态。")
            if _contains_any(text, ("错误代码", ".sys", "蓝屏")):
                required.append("请提供蓝屏错误代码、失败模块文件名和现场照片。")
            if _contains_any(text, ("Minidump", ".dmp", "WinDbg")):
                required.append("请提供故障时间点对应的 Minidump / DMP 转储文件。")
            if _contains_any(text, ("事件查看器", "事件ID", "6008")):
                required.append("请提供故障时间点前后的 Windows 系统事件日志。")
            if section_kind == "cause_support":
                out.append({
                    "case_id": f"{section.get('section_id')}:support",
                    "source_doc_id": f"doc:{path.stem}",
                    "source_doc_title": path.stem,
                    "section_id": section.get("section_id"),
                    "section_title": title,
                    "section_case_kind": "component_support",
                    "family_scope_candidates": [],
                    "variant_candidate": "",
                    **self._fault_manual_action_payload(action_lines),
                    "required_info": [],
                    "thresholds": [],
                    "cause_notes": cause_lines,
                    "support_notes": [*symptom_lines, *reasoning_lines],
                    "symptom_signals": symptom_lines,
                    "fault_mapping_allowed": False,
                    "output_mode": "procedure_library_only",
                })
                continue
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": title,
                "section_case_kind": "fault_case",
                "family_scope_candidates": self._boot_manual_family_scope(title, text),
                "variant_candidate": title,
                **self._fault_manual_action_payload(action_lines),
                "required_info": required,
                "thresholds": [],
                "cause_notes": cause_lines,
                "support_notes": [*symptom_lines, *reasoning_lines],
                "symptom_signals": symptom_lines,
                "fault_mapping_allowed": True,
                "output_mode": "variant_case_bundle",
            })
        return out

    def _section_cases_procedure(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            kind = str(section.get("section_kind") or "")
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            if kind not in {"procedure_step", "procedure_intro"} or not lines:
                continue
            if kind == "procedure_step":
                title = str(section.get("section_title") or "").strip()
                procedure_steps = [{
                    "step_order": 1,
                    "label": title,
                    "instruction": lines[0] if lines else title,
                    "details": lines[1:],
                    "source_lines": [title, *lines],
                }] if title else []
                action_payload = {
                    "actions": [title] if title else [],
                    "procedure_steps": procedure_steps,
                }
            else:
                action_payload = self._action_payload(lines)
            if not action_payload.get("actions"):
                continue
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_case_kind": "procedure_step",
                "family_scope_candidates": [],
                "variant_candidate": "",
                **action_payload,
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": [],
                "output_mode": "procedure_library_only",
            })
        return out

    def _section_cases_spec(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            if not lines:
                continue
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_case_kind": "spec_constraint",
                "family_scope_candidates": [],
                "variant_candidate": "",
                "actions": [],
                "required_info": [],
                "thresholds": lines,
                "cause_notes": [],
                "support_notes": [],
                "output_mode": "reference_constraint_only",
            })
        return out

    def _section_cases_validation(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            if not lines:
                continue
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_case_kind": "validation_block",
                "family_scope_candidates": [],
                "variant_candidate": "",
                **self._action_payload(lines),
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": [],
                "expected_results": [line for line in lines if _contains_any(line, ("无", "正常", "稳定", "通过"))],
                "output_mode": "policy_template_only",
            })
        return out

    def _section_cases_faq(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_case_kind": str(section.get("section_kind") or ""),
                "family_scope_candidates": [],
                "variant_candidate": "",
                "actions": [],
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": lines,
                "output_mode": "faq_support_bundle" if section.get("section_kind") == "faq_pair" else "review_only",
            })
        return out

    def _section_cases_overlay(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            if not lines:
                continue
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_case_kind": "overlay_block",
                "family_scope_candidates": [],
                "variant_candidate": "",
                "actions": [],
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": lines,
                "output_mode": "overlay_only",
            })
        return out

    def _section_cases_unclassified(self, path: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for section in sections:
            lines = [str(x) for x in section.get("body_lines") or [] if str(x).strip()]
            out.append({
                "case_id": f"{section.get('section_id')}:case",
                "source_doc_id": f"doc:{path.stem}",
                "source_doc_title": path.stem,
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_case_kind": "manual_review_needed",
                "family_scope_candidates": [],
                "variant_candidate": "",
                "actions": [],
                "required_info": [],
                "thresholds": [],
                "cause_notes": [],
                "support_notes": lines,
                "output_mode": "review_only",
            })
        return out

    def _to_actions(self, lines: list[str]) -> list[str]:
        actions: list[str] = []
        for line in lines:
            clean = _BULLET_RE.sub("", line).strip()
            if clean:
                actions.append(clean)
        return actions

    def _action_payload(self, lines: list[str]) -> dict[str, Any]:
        procedure_steps = self._to_procedure_steps(lines)
        fallback_actions = [
            action
            for action in self._to_actions(lines)
            if self._looks_like_action_text(action)
        ]
        return {
            "actions": [str(step.get("label") or "") for step in procedure_steps]
            if procedure_steps
            else fallback_actions,
            "procedure_steps": procedure_steps,
        }

    @staticmethod
    def _looks_like_action_text(text: str) -> bool:
        value = str(text or "").strip()
        if not value or value.endswith(("?", "？")):
            return False
        core = value
        for prefix in ("先", "再", "同时", "逐行", "彻底", "手动", "临时"):
            if core.startswith(prefix):
                core = core[len(prefix):].lstrip()
                break
        heading = re.split(r"[：:]", core, maxsplit=1)[0].strip()
        return (
            core.startswith(_PROCEDURE_ACTION_PREFIXES)
            or heading.endswith(("检查", "分析", "诊断", "测试", "验证", "修复", "排查", "确认"))
        )

    def _fault_manual_action_payload(self, lines: list[str]) -> dict[str, Any]:
        label_map = {
            "首要操作": "记录屏幕关键信息",
            "信息收集": "收集故障时间点系统日志",
            "初步分析": "分析错误代码或失败文件名",
            "高级分析": "分析 DMP 转储文件",
            "进入安全模式": "进入安全模式",
            "模式分析": "根据错误代码稳定性选择排查路径",
            "让错误显现": "关闭自动重新启动以显示蓝屏代码",
            "查阅日志": "查看故障时间点系统事件日志",
            "排除软件冲突": "执行干净启动排除软件冲突",
            "重点硬件排查": "检查温度并替换测试电源",
            "初步判断": "判断系统是否能调出安全界面",
            "软件层面": "检查高占用进程并进入安全模式排查软件",
            "硬件层面": "检查散热、内存、主板和电源",
        }
        condition_prefixes = ("若错误代码", "能调出安全界面", "完全无响应")
        steps: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for raw in lines:
            text = str(raw or "").strip()
            if not text:
                continue
            head, separator, tail = text.partition("：")
            head = head.strip()
            tail = tail.strip() if separator else ""
            if head.startswith(condition_prefixes) and current is not None:
                current["details"].append(text)
                current["source_lines"].append(text)
                continue
            label = label_map.get(head)
            if label:
                if current is not None:
                    steps.append(current)
                current = {
                    "step_order": len(steps) + 1,
                    "label": label,
                    "instruction": tail or text,
                    "details": [],
                    "source_lines": [text],
                }
                continue
            if current is not None and not head.startswith(_PROCEDURE_ACTION_PREFIXES):
                current["details"].append(text)
                current["source_lines"].append(text)
                continue
            if current is not None:
                steps.append(current)
                current = None
            fallback = self._to_procedure_steps([text])
            if fallback:
                step = dict(fallback[0])
                step["step_order"] = len(steps) + 1
                steps.append(step)
        if current is not None:
            steps.append(current)
        if not steps:
            return self._action_payload(lines)
        return {
            "actions": [str(step.get("label") or "") for step in steps],
            "procedure_steps": steps,
        }

    def _to_procedure_steps(self, lines: list[str]) -> list[dict[str, Any]]:
        """Preserve explicit parent-step hierarchy instead of flattening bullets.

        A block such as ``第一步：清洁除尘`` followed by one instruction and
        several target bullets represents one operation.  The child lines are
        retained as instruction/details and must not become peer actions.
        """

        steps: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        expanded_lines: list[str] = []
        for raw_line in lines:
            chunks = _INLINE_STEP_SPLIT_RE.split(str(raw_line or ""))
            expanded_lines.extend(chunk.lstrip("①②③④⑤⑥⑦⑧⑨⑩ ") for chunk in chunks if chunk.strip())
        for raw_line in expanded_lines:
            clean = _BULLET_RE.sub("", str(raw_line or "")).strip()
            if not clean:
                continue
            match = _PROCEDURE_STEP_RE.match(clean)
            if match:
                if current is not None:
                    steps.append(current)
                label = str(match.group("title") or "").strip(" ：:、.-")
                current = {
                    "step_order": len(steps) + 1,
                    "label": label or clean,
                    "instruction": "",
                    "details": [],
                    "source_lines": [clean],
                }
                continue
            if current is None:
                continue
            current["source_lines"].append(clean)
            if not current["instruction"]:
                current["instruction"] = clean.rstrip("：:")
            else:
                current["details"].append(clean)
        if current is not None:
            steps.append(current)
        if not steps:
            cleaned = [
                re.sub(
                    r"^[一二三四五六七八九十]+[、：:]\s*",
                    "",
                    _BULLET_RE.sub("", str(line or "")).strip(),
                )
                for line in lines
                if _BULLET_RE.sub("", str(line or "")).strip()
            ]
            first_heading = cleaned[0].rstrip("：:") if cleaned else ""
            if (
                cleaned
                and cleaned[0].endswith(("：", ":"))
                and first_heading.startswith(_PROCEDURE_ACTION_PREFIXES)
                and not first_heading.endswith(_NON_ACTION_HEADING_SUFFIXES)
            ):
                parent = cleaned[0].rstrip("：:")
                steps.append({
                    "step_order": 1,
                    "label": parent,
                    "instruction": parent,
                    "details": cleaned[1:],
                    "source_lines": cleaned,
                })
            else:
                current = None
                pending_context = ""
                for clean in cleaned:
                    plain = clean.strip()
                    heading = plain.rstrip("：:")
                    heading_prefix = re.split(r"[：:]", heading, maxsplit=1)[0].strip()
                    is_action = (
                        heading.startswith(_PROCEDURE_ACTION_PREFIXES)
                        and not heading_prefix.endswith(_NON_ACTION_HEADING_SUFFIXES)
                    )
                    if plain.endswith(("：", ":")) and not is_action:
                        pending_context = heading_prefix
                        continue
                    if not is_action and heading_prefix.endswith(_NON_ACTION_HEADING_SUFFIXES):
                        pending_context = heading_prefix
                        continue
                    if is_action:
                        if current is not None:
                            steps.append(current)
                        label = heading
                        for sep in ("：", ":", "，", "。", "；"):
                            if sep in label:
                                label = label.split(sep, 1)[0].strip()
                                break
                        if label.startswith("尝试") and len(label) > 2:
                            label = label[2:].strip()
                        current = {
                            "step_order": len(steps) + 1,
                            "label": label[:60] or heading[:60],
                            "instruction": plain,
                            "details": [pending_context] if pending_context else [],
                            "source_lines": [plain],
                        }
                        pending_context = ""
                    elif current is not None:
                        current["details"].append(plain)
                        current["source_lines"].append(plain)
                if current is not None:
                    steps.append(current)
        return steps

    def _diagnostic_required_info(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        text = " ".join(lines)
        if _contains_any(text, ("温度", "核心温度", "封装温度")):
            out.append("请提供 CPU 核心温度、CPU 封装温度和测试时温度曲线。")
        if _contains_any(text, ("降频", "卡顿", "关机", "重启")):
            out.append("请说明是否出现降频、卡顿、自动重启以及触发时机。")
        if _contains_any(text, ("风扇", "散热器", "螺丝", "积尘")):
            out.append("请提供风扇转速、散热器固定情况和积尘状态。")
        if _contains_any(text, ("OCCT", "AIDA64")):
            out.append("请提供 OCCT 或 AIDA64 的监控结果截图。")
        return out

    def _solution_required_info(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        text = " ".join(lines)
        if _contains_any(text, ("BIOS", "风扇策略")):
            out.append("请提供当前 BIOS 风扇策略或风扇曲线配置。")
        if _contains_any(text, ("硅脂", "散热器")):
            out.append("请提供散热器安装状态和硅脂处理结果。")
        if _contains_any(text, ("主板", "供电")):
            out.append("请说明主板 CPU 供电插头状态以及是否做过替换测试。")
        return out

    def _boot_manual_family_scope(self, title: str, text: str) -> list[str]:
        normalized_title = re.sub(r"\s+", "", title)
        if normalized_title == "蓝屏" or "无限蓝屏重启循环" in normalized_title:
            return ["工控机蓝屏"]
        if normalized_title == "重启":
            return ["工控机异常重启"]
        if normalized_title.startswith("死机"):
            return ["工控机死机"]
        joined = f"{title} {text}"
        if _contains_any(joined, ("无通电", "无法开机", "电源保护", "短路")):
            return ["工控机无法开机"]
        if _contains_any(joined, ("无显示", "黑屏", "显示器无显示")):
            return ["工控机黑屏无显示"]
        if _contains_any(joined, ("间歇性黑屏", "死机")):
            return ["工控机死机"]
        if _contains_any(joined, ("重启", "市电", "工业环境干扰")):
            return ["工控机异常重启"]
        if _contains_any(joined, ("BOOT", "自检", "BIOS")):
            return ["工控机无法开机", "工控机黑屏无显示"]
        return ["工控机无法开机"]


__all__ = ["DOC_STRATEGY_SPECS", "RawDocIngestAgent"]
