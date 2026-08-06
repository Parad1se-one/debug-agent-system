"""Document-first KG v2 base builder from explicit source manifest."""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.builders import (
    _action_label,
    _canonical_family_for_seed,
    _dedupe_objects,
    _dedupe_relations,
    _family_category,
    _family_summary_for_seed,
    _is_destructive,
    _is_high_cost,
    _short_title,
    _subsystem_for_seed,
    _variant_label_for_sop,
    infer_action_role,
    infer_required_info_slot,
)
from debug_agent_system.knowledge_v2.contracts import make_id, trim_text

DEFAULT_SOURCE_MANIFEST = "data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json"
_TITLE_HEADING_RE = re.compile(r"^\s*(?:[0-9]+[.．、)]\s*)?(.+?)\s*$")
_MANUAL_SECTION_RE = re.compile(r"^\s*([0-9]+)[.．、]?\s*([^：:]{0,40})\s*$")
_BOOT_SECTION_RE = re.compile(r"^\*\*情况\s*([0-9]+)[：:]\s*(.+?)\*\*$")


class _FeishuHTMLBlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[dict[str, str]] = []
        self._stack: list[str] = []
        self._text_parts: list[str] = []
        self._current_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._stack.append(tag)
        if tag in {"h1", "h2", "h3", "h4", "h5", "p", "li"} and not self._current_tag:
            self._current_tag = tag
            self._text_parts = []
        elif self._current_tag and tag in {"p", "li"} and self._text_parts:
            self._text_parts.append("\n")
        elif tag == "br":
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == self._current_tag and self._text_parts:
            text = _clean_block_text("".join(self._text_parts))
            if text:
                self.blocks.append({"tag": tag, "text": text})
            self._current_tag = ""
            self._text_parts = []
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._current_tag:
            self._text_parts.append(data)


def _clean_block_text(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _load_manifest(path: str | Path = DEFAULT_SOURCE_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError(f"invalid_source_manifest:{manifest_path}")
    data["_manifest_path"] = str(manifest_path)
    return data


def build_doc_source_seed(manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST, *, limit: int = 0) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    objects = _empty_objects()
    relations: list[dict[str, Any]] = []
    section_count = 0
    source_reports: list[dict[str, Any]] = []
    family_seen: set[str] = set()
    overlays: list[dict[str, Any]] = []

    for source in manifest.get("sources") or []:
        if not isinstance(source, dict):
            continue
        if not bool(source.get("enabled", True)):
            source_reports.append({
                "source_id": source.get("source_id"),
                "status": "disabled",
                "reason": source.get("reason") or "",
            })
            continue
        if str(source.get("mode") or "sections") == "overlay":
            current_overlays = _load_source_overlays(source)
            overlays.extend(current_overlays)
            source_reports.append({
                "source_id": source.get("source_id"),
                "title": source.get("title") or "",
                "status": "loaded" if current_overlays else "empty",
                "overlay_count": len(current_overlays),
            })
            continue
        sections = _load_source_sections(source)
        source_reports.append({
            "source_id": source.get("source_id"),
            "title": source.get("title") or "",
            "status": "loaded" if sections else "empty",
            "section_count": len(sections),
        })
        for section in sections:
            if limit and section_count >= limit:
                break
            title = str(section.get("title") or "").strip()
            body = str(section.get("body") or "").strip()
            if not title or not body:
                continue
            section_count += 1

            case_ref = str(section.get("section_id") or f"{source.get('source_id')}:{section_count}")
            case_id = make_id("case", case_ref)
            evidence_id = make_id("evidence", case_ref)
            title_short = _short_title(title)
            family_label = _canonical_family_for_seed(title_short, title, body)
            family_id = make_id("family", family_label)
            family_summary = trim_text(_family_summary_for_seed(family_label, title), 80)
            category = _family_category(title, body)
            subsystem = _subsystem_for_seed(family_label, title, body)
            variant_label = _variant_label_for_sop(title, family_label, case_ref)
            variant_id = make_id("variant", f"{case_ref}:{variant_label}")

            if family_id not in family_seen:
                objects["FaultFamily"].append({
                    "family_id": family_id,
                    "label": family_label,
                    "summary": family_summary,
                    "category": category,
                    "subsystem": subsystem,
                    "scenario": trim_text(title, 60),
                    "keywords": _keywords_from_section(title, body),
                    "source_kind": "sop",
                    "escalation_target": "",
                })
                family_seen.add(family_id)

            objects["FaultVariant"].append({
                "variant_id": variant_id,
                "family_id": family_id,
                "label": trim_text(variant_label, 60),
                "summary": trim_text(title, 180),
                "equipment_type": "",
                "site": "",
                "software_version": "",
                "error_phase": trim_text(title, 40),
                "owner_context": f"{source.get('source_id')}:{case_ref}",
                "escalation_target": "",
                "keywords": _keywords_from_section(title, body),
            })
            relations.append({"from": family_id, "to": variant_id, "relation": "has_variant"})

            lines = _section_action_lines(title, body)
            action_ids: list[str] = []
            for idx, line in enumerate(lines, start=1):
                action_id = make_id("action", f"{case_ref}:{idx}:{line}")
                action_ids.append(action_id)
                objects["DiagnosticAction"].append({
                    "action_id": action_id,
                    "family_id": family_id,
                    "label": trim_text(_action_label(line), 60),
                    "summary": trim_text(line, 180),
                    "action_role": infer_action_role(line),
                    "step_order": idx,
                    "destructive": _is_destructive(line),
                    "high_cost": _is_high_cost(line),
                    "source_kind": "sop",
                })

            required_specs = _required_info_specs_for_section(family_id, variant_id, case_ref, title, body)
            objects["RequiredInfoSpec"].extend(required_specs)
            for spec in required_specs:
                rid = str(spec.get("required_info_id") or "")
                relations.append({"from": family_id, "to": rid, "relation": "has_required_info"})
                relations.append({"from": case_id, "to": rid, "relation": "supports"})
                relations.append({"from": evidence_id, "to": rid, "relation": "evidences"})

            objects["SourceCase"].append({
                "case_id": case_id,
                "source_kind": "sop",
                "title": trim_text(title, 80),
                "summary": trim_text(body, 240),
                "source_ref": case_ref,
                "approved": True,
            })
            objects["EvidenceItem"].append({
                "evidence_id": evidence_id,
                "source_kind": "sop",
                "external_id": case_ref,
                "title": trim_text(title, 80),
                "summary": trim_text(body, 500),
                "payload_ref": str(source.get("source_path") or source.get("fetch_json_path") or ""),
            })
            relations.append({"from": case_id, "to": variant_id, "relation": "supports"})
            relations.append({"from": evidence_id, "to": case_id, "relation": "evidences"})

            if action_ids:
                trace_id = make_id("trace", case_ref)
                objects["DiagnosticTrace"].append({
                    "trace_id": trace_id,
                    "family_id": family_id,
                    "variant_id": variant_id,
                    "source_case_id": case_id,
                    "summary": trim_text(f"{family_label} 的文档标准排查链", 160),
                    "recommended_action_ids": action_ids,
                    "actual_action_ids": action_ids,
                    "evidence_ids": [evidence_id],
                })
                relations.append({"from": family_id, "to": trace_id, "relation": "has_trace"})
                relations.append({"from": case_id, "to": trace_id, "relation": "supports"})
                for action_id in action_ids:
                    relations.append({"from": trace_id, "to": action_id, "relation": "used_action"})

    if overlays:
        _apply_family_overlays(objects["FaultFamily"], overlays)

    return {
        "objects": _dedupe_objects(objects),
        "relations": _dedupe_relations(relations),
        "report": {
            "source": "document_first_manifest",
            "manifest_path": manifest["_manifest_path"],
            "sources": source_reports,
            "sections": section_count,
            "overlay_count": len(overlays),
        },
    }


def _load_source_sections(source: dict[str, Any]) -> list[dict[str, Any]]:
    parser = str(source.get("parser") or "").strip()
    if parser == "feishu_fetch_html":
        return _load_fetch_html_sections(source)
    if parser == "docx_lines":
        return _load_docx_line_sections(source)
    if parser == "markdown_sections":
        return _load_markdown_sections(source)
    raise ValueError(f"unsupported_source_parser:{parser}")


def _load_source_overlays(source: dict[str, Any]) -> list[dict[str, Any]]:
    parser = str(source.get("parser") or "").strip()
    strategy = str(source.get("section_strategy") or "")
    if parser == "markdown_sections" and strategy == "process_roles_markdown":
        return _load_process_role_overlays(source)
    raise ValueError(f"unsupported_overlay_source:{parser}:{strategy}")


def _load_fetch_html_sections(source: dict[str, Any]) -> list[dict[str, Any]]:
    fetch_path = Path(str(source.get("fetch_json_path") or ""))
    if not fetch_path.exists():
        return _load_docx_line_sections(source)
    raw = json.loads(fetch_path.read_text(encoding="utf-8"))
    content = str((((raw.get("data") or {}).get("document") or {}).get("content")) or "")
    if not content:
        return _load_docx_line_sections(source)
    parser = _FeishuHTMLBlockParser()
    parser.feed(content)
    blocks = parser.blocks
    mode = str(source.get("section_strategy") or "")
    if mode == "sop_headings":
        sections = _sections_from_sop_blocks(blocks, source)
        supplements = _supplement_sop_h5_sections(content, source, existing_titles={str(x.get("title") or "") for x in sections})
        return sections + supplements
    if mode == "manual_numbered":
        return _sections_from_manual_blocks(blocks, source)
    raise ValueError(f"unsupported_section_strategy:{mode}")


def _load_docx_line_sections(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(source.get("fallback_docx_path") or source.get("source_path") or ""))
    if not path.exists():
        return []
    lines = _extract_docx_lines(path)
    title = str(source.get("title") or path.stem)
    blocks = [{"tag": "p", "text": line} for line in lines]
    mode = str(source.get("section_strategy") or "")
    if mode == "sop_headings":
        return _sections_from_sop_blocks(blocks, source, title=title)
    if mode == "manual_numbered":
        return _sections_from_manual_blocks(blocks, source, title=title)
    return []


def _load_markdown_sections(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(source.get("source_path") or source.get("fallback_docx_path") or ""))
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    mode = str(source.get("section_strategy") or "")
    if mode == "boot_manual_markdown":
        return _sections_from_boot_manual_markdown(text, source)
    if mode == "manual_numbered":
        blocks = [{"tag": "p", "text": line} for line in text.splitlines()]
        return _sections_from_manual_blocks(blocks, source, title=str(source.get("title") or path.stem))
    return []


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
        line = _clean_block_text("".join(texts))
        if line:
            lines.append(line)
    return lines


def _sections_from_sop_blocks(blocks: list[dict[str, str]], source: dict[str, Any], *, title: str | None = None) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    h1 = h2 = h3 = ""
    current_title = ""
    body_parts: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal current_title, body_parts, counter
        if not current_title or not body_parts:
            current_title = ""
            body_parts = []
            return
        counter += 1
        section_id = f"{source.get('source_id')}:{counter}"
        path_bits = [x for x in (h1, h2, h3, current_title) if x]
        sections.append({
            "source_id": source.get("source_id"),
            "section_id": section_id,
            "path": path_bits,
            "title": current_title,
            "body": "\n".join(body_parts),
        })
        current_title = ""
        body_parts = []

    for block in blocks:
        tag = block.get("tag") or ""
        text = _clean_block_text(block.get("text") or "")
        if not text:
            continue
        if tag == "h1":
            flush()
            h1, h2, h3 = text, "", ""
            continue
        if tag == "h2":
            flush()
            h2, h3 = text, ""
            continue
        if tag == "h3":
            flush()
            h3 = text
            continue
        if tag in {"h4", "h5"}:
            flush()
            current_title = text
            continue
        if not current_title and h3 and _issue_like_title(h3):
            current_title = h3
        if current_title:
            body_parts.append(text)
    flush()

    if not sections and title:
        lines = [block["text"] for block in blocks if block.get("text")]
        if lines:
            sections.append({
                "source_id": source.get("source_id"),
                "section_id": f"{source.get('source_id')}:1",
                "path": [title],
                "title": title,
                "body": "\n".join(lines[:80]),
            })
    return sections


def _sections_from_manual_blocks(blocks: list[dict[str, str]], source: dict[str, Any], *, title: str | None = None) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title = ""
    body_parts: list[str] = []
    counter = 0
    prefix = title or str(source.get("title") or "")

    def flush() -> None:
        nonlocal current_title, body_parts, counter
        if not current_title or not body_parts:
            current_title = ""
            body_parts = []
            return
        counter += 1
        sections.append({
            "source_id": source.get("source_id"),
            "section_id": f"{source.get('source_id')}:{counter}",
            "path": [prefix, current_title] if prefix else [current_title],
            "title": current_title,
            "body": "\n".join(body_parts),
        })
        current_title = ""
        body_parts = []

    for block in blocks:
        text = _clean_block_text(block.get("text") or "")
        if not text:
            continue
        m = _MANUAL_SECTION_RE.match(text)
        if m and any(k in text for k in ("蓝屏", "重启", "死机", "黑屏", "不开机", "无法开机")):
            flush()
            current_title = text
            continue
        if current_title:
            body_parts.append(text)
    flush()
    return sections


def _sections_from_boot_manual_markdown(text: str, source: dict[str, Any]) -> list[dict[str, Any]]:
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    current_title = ""
    body_parts: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal current_title, body_parts, counter
        if not current_title or not body_parts:
            current_title = ""
            body_parts = []
            return
        counter += 1
        sections.append({
            "source_id": source.get("source_id"),
            "section_id": f"{source.get('source_id')}:{counter}",
            "path": [str(source.get("title") or ""), current_title],
            "title": current_title,
            "body": "\n".join(body_parts),
        })
        current_title = ""
        body_parts = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _BOOT_SECTION_RE.match(line)
        if m:
            flush()
            current_title = f"情况 {m.group(1)}：{m.group(2).strip()}"
            continue
        if current_title:
            body_parts.append(_clean_markdown_line(line))
    flush()
    return sections


def _supplement_sop_h5_sections(content: str, source: dict[str, Any], *, existing_titles: set[str]) -> list[dict[str, Any]]:
    pattern = re.compile(r"<h5[^>]*>(?P<title>.*?)</h5>(?P<body>.*?)(?=<h5|<h4|<h3|<h2|<h1|$)", re.S | re.I)
    sections: list[dict[str, Any]] = []
    counter = 0
    for m in pattern.finditer(content):
        title = _clean_html_inline_text(m.group("title"))
        if not title or title in existing_titles:
            continue
        body_html = str(m.group("body") or "")
        body_blocks = _html_fragment_blocks(body_html)
        body_lines = [b.get("text") or "" for b in body_blocks if b.get("text")]
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        counter += 1
        sections.append({
            "source_id": source.get("source_id"),
            "section_id": f"{source.get('source_id')}:h5:{counter}",
            "path": [str(source.get("title") or ""), title],
            "title": title,
            "body": body,
        })
    return sections


def _html_fragment_blocks(fragment: str) -> list[dict[str, str]]:
    parser = _FeishuHTMLBlockParser()
    parser.feed(fragment)
    return parser.blocks


def _clean_html_inline_text(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", str(text or ""))
    return _clean_block_text(value)


def _clean_markdown_line(line: str) -> str:
    text = str(line or "").strip()
    text = re.sub(r"^\s*[-*+]\s*", "", text)
    text = text.replace("**", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return _clean_block_text(text)


def _issue_like_title(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    if len(clean) > 50:
        return False
    return any(k in clean for k in ("失败", "异常", "报错", "无法", "卡住", "闪退", "蓝屏", "黑屏", "重启", "死机"))


def _load_process_role_overlays(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(source.get("source_path") or ""))
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    in_table = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("|**负责人**|") or line.startswith("|负责人|"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table:
            if not line.startswith("|"):
                break
            parts = [part.strip().strip("*") for part in line.strip("|").split("|")]
            if len(parts) >= 3:
                rows.append({
                    "owner": parts[0],
                    "category": parts[1],
                    "description": parts[2],
                })
    overlays: list[dict[str, Any]] = []
    for row in rows:
        owner = str(row.get("owner") or "")
        category = str(row.get("category") or "")
        families: list[str] = []
        if "@工程师乙" in owner:
            families.extend(["工控机蓝屏", "工控机异常重启", "工控机无法开机", "工控机黑屏无显示", "主程序/系统异常", "软件卡死无响应", "程序运行卡顿", "BIOS 启动配置异常", "磁盘 I/O 异常"])
        if "@工程师丑" in owner:
            families.extend(["相机拍摄失败", "相机初始化失败", "运控初始化失败", "挡块异常", "传感器感应异常", "轨道宽度无法调节", "进板失败", "皮带运行异常", "顶升机构异常", "气压异常", "程序板卡加载失败", "控制器网络配置异常", "出板失败"])
        if "@工程师丙" in owner:
            families.extend(["误报调优异常", "漏检调优异常", "识别框大小不准确", "焊盘框不对齐", "器件框角度不匹配", "框选识别不准", "坏板标记异常"])
        if "@工程师H" in owner:
            families.extend(["MES 过站异常"])
        if "@工程师J" in owner:
            families.extend(["SPC 页面无法打开"])
        if "@工程师C" in owner:
            families.extend(["Buddy 模板缺失", "复判保存结果失败"])
        if "@工程师A" in owner:
            families.extend(["扫码识别失败"])
        if not families:
            continue
        overlays.append({
            "source_id": source.get("source_id"),
            "owner": owner,
            "category": category,
            "description": row.get("description") or "",
            "families": sorted(set(families)),
        })
    return overlays


def _apply_family_overlays(families: list[dict[str, Any]], overlays: list[dict[str, Any]]) -> None:
    by_label = {str(item.get("label") or ""): item for item in families if isinstance(item, dict)}
    for overlay in overlays:
        owner = str(overlay.get("owner") or "")
        category = str(overlay.get("category") or "")
        for label in overlay.get("families") or []:
            family = by_label.get(str(label))
            if not family:
                continue
            if owner:
                family["escalation_target"] = owner
            kws = family.get("keywords")
            if not isinstance(kws, list):
                kws = []
            for extra in [owner, category]:
                if extra and extra not in kws:
                    kws.append(extra)
            family["keywords"] = kws[:24]


def _section_action_lines(title: str, body: str) -> list[str]:
    raw_lines = []
    for line in str(body or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|[0-9]+[.．、):：]?|[一二三四五六七八九十]+[、.．:：])\s*", "", line).strip()
        if not line:
            continue
        parts = [x.strip() for x in re.split(r"[。]\s*|\n+", line) if x.strip()]
        raw_lines.extend(parts or [line])
    lines = []
    for line in raw_lines:
        if len(line) < 3:
            continue
        if any(k in line for k in ("排查步骤", "一般原因", "表现", "排查思路", "信息收集", "初步分析", "高级分析", "重要提示", "核心概念")):
            continue
        if not any(k in line for k in ("检查", "确认", "分析", "导出", "进入", "卸载", "安装", "运行", "更换", "观察", "监控", "查看", "记录", "拍照", "收集", "提供", "测试", "修复", "恢复", "关闭", "打开", "复位", "重启", "清理", "使用", "核对", "区分", "尝试", "检测", "升级", "排查", "联系", "开启")):
            continue
        lines.append(trim_text(line, 180))
    if not lines:
        fallback = trim_text(title, 180)
        if fallback:
            lines = [fallback]
    deduped = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped[:16]


def _required_info_specs_for_section(family_id: str, variant_id: str, case_ref: str, title: str, body: str) -> list[dict[str, Any]]:
    lines = []
    for line in str(body or "").splitlines():
        text = _clean_block_text(line)
        if not text:
            continue
        if any(k in text for k in ("请提供", "记录", "拍照", "导出", "信息收集", "日志", "版本", "报错", "DMP", "Minidump", "MEMORY.DMP", "相机IP", "复现")):
            lines.append(text)
    if not lines:
        lines = _default_required_info_lines(title, body)
    specs = []
    seen = set()
    for line in lines:
        slot = infer_required_info_slot(line)
        question = trim_text(_required_info_question(slot, line), 100)
        key = (slot, question)
        if key in seen:
            continue
        seen.add(key)
        rid = make_id("required-info", f"{case_ref}:{slot}:{question}")
        specs.append({
            "required_info_id": rid,
            "family_id": family_id,
            "variant_id": variant_id,
            "slot": slot,
            "question": question,
            "why_required": trim_text(_required_info_why(slot, title), 160),
            "condition": "",
            "blocks": [question],
            "priority": "high" if slot in {"log_package", "error_message", "software_version", "ip_config", "dmp_package"} else "medium",
            "evidence_ids": [make_id("evidence", case_ref)],
        })
    return specs[:10]


def _required_info_question(slot: str, line: str) -> str:
    slot_map = {
        "dmp_package": "请提供蓝屏或重启对应的 dmp/转储文件。",
        "log_package": "请提供诊断日志 / DLOG / 诊断数据包。",
        "software_version": "请提供软件版本 / SDK 版本 / machine 版本。",
        "error_message": "请提供完整报错信息或错误代码。",
        "error_phase": "请说明故障发生阶段和复现时机。",
        "ip_config": "请提供 IP / 网卡 / 控制器网络配置。",
        "repro_steps": "请提供稳定复现步骤。",
        "program_file": "请提供程序文件 / CAD / 配方 / 配置文件。",
        "environment": "请提供环境信息，如内存、磁盘、温度、供电、接地等。",
        "owner_context": "请提供现场上下文和责任归属信息。",
    }
    return slot_map.get(slot, trim_text(line, 100))


def _required_info_why(slot: str, title: str) -> str:
    why_map = {
        "dmp_package": "需要从转储文件中定位蓝屏/重启的内核错误来源。",
        "log_package": "需要从日志里确认报错点、阶段和模块边界。",
        "software_version": "需要判断是否命中特定版本行为或已知版本差异。",
        "error_message": "需要用错误码 / 报错文本收敛故障分支。",
        "error_phase": "需要用故障发生阶段缩小诊断路径。",
        "ip_config": "需要判断是否是网络配置或链路稳定性问题。",
        "repro_steps": "需要稳定复现路径才能验证修复和定位分支。",
        "program_file": "需要核对程序 / CAD / 配方 / 配置是否本身异常。",
        "environment": "需要排查供电、温度、磁盘、内存等环境因素。",
        "owner_context": "需要确认责任边界和现场处理限制。",
    }
    return why_map.get(slot, f"缺少该信息会影响 {trim_text(title, 40)} 的诊断分流。")


def _default_required_info_lines(title: str, body: str) -> list[str]:
    text = f"{title}\n{body}"
    lines = []
    if any(k in text for k in ("蓝屏", "BugCheck", "Minidump", "MEMORY.DMP", "重启")):
        lines.extend(["请提供蓝屏或重启对应的 dmp/转储文件。", "请提供诊断日志 / DLOG / 诊断数据包。", "请提供软件版本 / SDK 版本 / machine 版本。"])
    if any(k in text for k in ("相机", "拍摄失败", "拍照失败", "IP")):
        lines.extend(["请提供诊断日志 / DLOG / 诊断数据包。", "请提供 IP / 网卡 / 控制器网络配置。", "请说明故障发生阶段和复现时机。"])
    if any(k in text for k in ("无法开机", "不开机", "黑屏")):
        lines.extend(["请拍摄开机现象并记录指示灯/风扇状态。", "请提供软件版本 / SDK 版本 / machine 版本。"])
    return lines


def _keywords_from_section(title: str, body: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]{2,8}", f"{title} {body}")
    out = []
    seen = set()
    for word in words:
        w = word.strip()
        if len(w) < 2:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= 16:
            break
    return out


def _empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {
        "FaultFamily": [],
        "FaultVariant": [],
        "DiagnosticAction": [],
        "ActionOutcome": [],
        "RequiredInfoSpec": [],
        "DiagnosticTrace": [],
        "DecisionPolicy": [],
        "EvidenceItem": [],
        "SourceCase": [],
    }
