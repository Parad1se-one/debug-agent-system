from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.materializer import KGV2Materializer
from debug_agent_system.knowledge_v2.validator import validate_graph


BUILD_ROOT = Path("data/kg_v2_sop_draft_build")
TARGET_ROOT = Path("data/kg_v2")
SUMMARY_OUT = Path("data/results/kg_v2_sop_draft_main_program_phase1_summary.json")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hid(prefix: str, *parts: str) -> str:
    raw = " | ".join(str(x or "") for x in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _family_id(label: str) -> str:
    return _hid("family", label)


def _variant_id(family_label: str, variant_label: str) -> str:
    return _hid("variant", family_label, variant_label)


def _action_id(variant_id: str, step_order: int, label: str) -> str:
    return _hid("action", variant_id, str(step_order), label)


def _required_id(variant_id: str, slot: str, question: str) -> str:
    return _hid("required-info", variant_id, slot, question)


def _trace_id(variant_id: str, summary: str) -> str:
    return _hid("trace", variant_id, summary)


def _policy_id(family_id: str) -> str:
    return _hid("policy", family_id)


def _case_id(section_id: str, family_label: str, variant_label: str) -> str:
    return _hid("case", section_id, family_label, variant_label)


def _evidence_id(section_id: str, family_label: str) -> str:
    return _hid("evidence", section_id, family_label)


APPENDIX1_TAXONOMY = [
    {"owner": "工程师K", "family_label": "软件功能咨询/需求类"},
    {"owner": "工程师庚", "family_label": "主程序软件问题"},
    {"owner": "工程师A", "family_label": "复判站软件问题"},
    {"owner": "工程师丁", "family_label": "软件使用及调试问题"},
    {"owner": "工程师乙", "family_label": "工控机/复判站/编程站及操作系统问题"},
    {"owner": "工程师丑", "family_label": "运控问题"},
    {"owner": "工程师F", "family_label": "标定问题"},
    {"owner": "工程师甲", "family_label": "硬件问题"},
    {"owner": "工程师丙", "family_label": "模型优化问题"},
    {"owner": "工程师I", "family_label": "3D成像问题"},
    {"owner": "工程师H", "family_label": "MES问题"},
    {"owner": "工程师J", "family_label": "SPC问题"},
    {"owner": "工程师C", "family_label": "Buddy问题"},
    {"owner": "工程师G", "family_label": "迁移工具"},
    {"owner": "工程师A", "family_label": "外部对接设备"},
    {"owner": "工程师午", "family_label": "其他问题及无法分类问题"},
]


SECTION_MAP = [
    {
        "canonical_section_id": "1.1.1.1.1",
        "title": "闪退导致模板文件损坏，如何修复？",
        "source_title_path": ["主程序", "主页面", "模板管理", "闪退导致模板文件损坏，如何修复？"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 主页面 > 模板管理 > 闪退导致模板文件损坏，如何修复？",
        },
        "appendix1_family_labels": ["Buddy问题"],
        "notes": "按 docx 本体标题编号对齐；当前先建模为 Buddy 模板元数据损坏修复。",
    },
    {
        "canonical_section_id": "1.1.2.1.1",
        "title": "CT时间长/复判站出图慢",
        "source_title_path": ["主程序", "主页面", "设置", "CT时间长/复判站出图慢"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 主页面 > 设置 > CT时间长/复判站出图慢",
        },
        "appendix1_family_labels": ["软件使用及调试问题", "复判站软件问题"],
        "notes": "真实标图编号由人工确认；按 docx 本体标题编号对齐。",
    },
    {
        "canonical_section_id": "1.1.1.1.2",
        "title": "buddy正常开启，模板管理刷新后也没有模板，创建模板失败。",
        "source_title_path": ["主程序", "主页面", "模板管理", "buddy正常开启，模板管理刷新后也没有模板，创建模板失败。"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 主页面 > 模板管理 > buddy正常开启，模板管理刷新后也没有模板，创建模板失败。",
        },
        "appendix1_family_labels": ["Buddy问题"],
        "notes": "当前按附表1 主分类裁决为 Buddy问题；具体变体先落为 Buddy 模板创建失败。",
    },
    {
        "canonical_section_id": "1.1.3",
        "title": "设备主程序无法打开",
        "source_title_path": ["主程序", "主页面", "设备主程序无法打开"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 主页面 > 设备主程序无法打开",
        },
        "appendix1_family_labels": ["运控问题"],
        "notes": "这是主页面下的父级条目；当前 build 用其子条目 1.1.3.1.1 入图。",
    },
    {
        "canonical_section_id": "1.1.3.1.1",
        "title": "设备主程序和工厂程序均无法打开",
        "source_title_path": ["主程序", "主页面", "设备主程序无法打开", "设备主程序和工厂程序均无法打开"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 主页面 > 设备主程序无法打开 > 设备主程序和工厂程序均无法打开",
        },
        "appendix1_family_labels": ["运控问题"],
        "notes": "当前按附表1 主分类裁决为运控问题；动作围绕运控日志和 MotionPanel 检查。",
    },
    {
        "canonical_section_id": "1.1.3.1.2",
        "title": "AOI主程序或者复判站主程序报:程序初始化失败:加载用户配置失败，处理思路",
        "source_title_path": ["主程序", "主页面", "设备主程序无法打开", "AOI主程序或者复判站主程序报:程序初始化失败:加载用户配置失败，处理思路"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 主页面 > 设备主程序无法打开 > AOI主程序或者复判站主程序报:程序初始化失败:加载用户配置失败，处理思路",
        },
        "appendix1_family_labels": ["主程序软件问题", "复判站软件问题"],
        "notes": "当前按附表1 主分类拆成主程序软件问题与复判站软件问题两个 family；具体故障语义下沉到 variant。",
    },
    {
        "canonical_section_id": "1.2.1.1.1",
        "title": "板卡弯曲/板卡加工误差时如何编程",
        "source_title_path": ["主程序", "编程&编程优化", "整版", "板卡弯曲/板卡加工误差时如何编程"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > 整版 > 板卡弯曲/板卡加工误差时如何编程",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前按附表1 主分类裁决为软件使用及调试问题；具体变体聚焦板弯补偿编程。",
    },
    {
        "canonical_section_id": "1.2.2.1.1",
        "title": "CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。",
        "source_title_path": ["主程序", "编程&编程优化", "CAD", "CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > CAD > CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前按附表1 主分类裁决为软件使用及调试问题；具体变体聚焦 CAD 导入失败。",
    },
    {
        "canonical_section_id": "1.2.3",
        "title": "板卡存在部分特殊斜角度位置，导入CAD同一料号有不一样角度，处理思路",
        "source_title_path": ["主程序", "编程&编程优化", "板卡存在部分特殊斜角度位置，导入CAD同一料号有不一样角度，处理思路"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > 板卡存在部分特殊斜角度位置，导入CAD同一料号有不一样角度，处理思路",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前按附表1 主分类裁决为软件使用及调试问题；具体变体聚焦 CAD 角度不一致。",
    },
    {
        "canonical_section_id": "1.2.4.1.1",
        "title": "希望一套程序用在多台机器上，但是Mark点在不同机器上会轻微跑偏而且还可能被遮挡。",
        "source_title_path": ["主程序", "编程&编程优化", "Mark点", "希望一套程序用在多台机器上，但是Mark点在不同机器上会轻微跑偏而且还可能被遮挡。"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > Mark点 > 希望一套程序用在多台机器上，但是Mark点在不同机器上会轻微跑偏而且还可能被遮挡。",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前按附表1 主分类裁决为软件使用及调试问题；具体变体聚焦 Mark 多机复用轻微跑偏/遮挡。",
    },
    {
        "canonical_section_id": "1.2.4.1.2",
        "title": "mark点对齐失败",
        "source_title_path": ["主程序", "编程&编程优化", "Mark点", "mark点对齐失败"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > Mark点 > mark点对齐失败",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前按附表1 主分类裁决为软件使用及调试问题；具体变体聚焦 Mark 点对齐失败。",
    },
    {
        "canonical_section_id": "1.2.4.1.3",
        "title": "双轨设备，近端程序拷贝到远端报mark点错误",
        "source_title_path": ["主程序", "编程&编程优化", "Mark点", "双轨设备，近端程序拷贝到远端报mark点错误"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > Mark点 > 双轨设备，近端程序拷贝到远端报mark点错误",
        },
        "appendix1_family_labels": ["硬件问题"],
        "notes": "当前按附表1 主分类裁决为硬件问题；具体变体聚焦远近轨夹边挡块位置差异导致的 Mark 遮挡。",
    },
    {
        "canonical_section_id": "1.2.6.1.1",
        "title": "识别框大小不准确",
        "source_title_path": ["主程序", "编程&编程优化", "检测框", "识别框大小不准确"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > 检测框 > 识别框大小不准确",
        },
        "appendix1_family_labels": ["模型优化问题"],
        "notes": "当前按动作与升级路径先裁决为模型优化问题；若后续案例证明只是现场参数误配，再在案例层修正。",
    },
    {
        "canonical_section_id": "1.2.6.1.2",
        "title": "侧立和翘脚漏报(客户06)",
        "source_title_path": ["主程序", "编程&编程优化", "检测框", "侧立和翘脚漏报(客户06)"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 编程&编程优化 > 检测框 > 侧立和翘脚漏报(客户06)",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前先按阈值调试问题建模；若后续历史案例显示核心是模型泛化缺陷，再补模型优化归因。",
    },
    {
        "canonical_section_id": "1.3.2",
        "title": "复判站检测页面没有参考图，点击加载最新模板后也没图",
        "source_title_path": ["主程序", "检测", "复判站检测页面没有参考图，点击加载最新模板后也没图"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 检测 > 复判站检测页面没有参考图，点击加载最新模板后也没图",
        },
        "appendix1_family_labels": ["复判站软件问题"],
        "notes": "当前按复判站页面与日志处理动作裁决为复判站软件问题。",
    },
    {
        "canonical_section_id": "1.3.3",
        "title": "2D设备同一光源下，器件成像差异较大",
        "source_title_path": ["主程序", "检测", "2D设备同一光源下，器件成像差异较大"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 检测 > 2D设备同一光源下，器件成像差异较大",
        },
        "appendix1_family_labels": ["硬件问题"],
        "notes": "当前动作全部围绕板平整度、轨道、顶升、反光与器件实物差异，先按硬件问题建模。",
    },
    {
        "canonical_section_id": "1.3.4",
        "title": "复盘站出现板卡加载时间长和加载板卡失败同时出现",
        "source_title_path": ["主程序", "检测", "复盘站出现板卡加载时间长和加载板卡失败同时出现"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 检测 > 复盘站出现板卡加载时间长和加载板卡失败同时出现",
        },
        "appendix1_family_labels": ["复判站软件问题"],
        "notes": "当前按日志、网络速度与 IP 检查链路，先裁决为复判站软件问题。",
    },
    {
        "canonical_section_id": "1.3.5",
        "title": "不同器件如何做替代料",
        "source_title_path": ["主程序", "检测", "不同器件如何做替代料"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 检测 > 不同器件如何做替代料",
        },
        "appendix1_family_labels": ["软件使用及调试问题"],
        "notes": "当前是模板库配置/编程规则知识，按软件使用及调试问题建模。",
    },
    {
        "canonical_section_id": "1.4.1.1.1",
        "title": "相机ip问题",
        "source_title_path": ["主程序", "初始化", "初始化失败", "相机ip问题"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 初始化 > 初始化失败 > 相机ip问题",
        },
        "appendix1_family_labels": ["硬件问题"],
        "notes": "当前按相机网络配置异常先归到硬件问题。",
    },
    {
        "canonical_section_id": "1.4.1.1.2",
        "title": "光源问题",
        "source_title_path": ["主程序", "初始化", "初始化失败", "光源问题"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 初始化 > 初始化失败 > 光源问题",
        },
        "appendix1_family_labels": ["硬件问题"],
        "notes": "当前动作集中在断电重启、插拔光控、检查 ARM/IP、防火墙与收集日志，先归到硬件问题。",
    },
    {
        "canonical_section_id": "1.4.1.1.4",
        "title": "卡在初始化运动控制卡，运控闪退",
        "source_title_path": ["主程序", "初始化", "初始化失败", "卡在初始化运动控制卡，运控闪退"],
        "source_doc": "异常处理 - 标准操作流程（SOP）",
        "source_locator": {
            "fetch_json_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json",
            "path_hint": "主程序 > 初始化 > 初始化失败 > 卡在初始化运动控制卡，运控闪退",
        },
        "appendix1_family_labels": ["运控问题"],
        "notes": "当前按运控日志与网卡速率调整链路，归到运控问题。",
    },
]


SECTION_RAW_TEXTS = {
    "1.1.1.1.1": (
        "1. 从 Buddy 的数据库读取项目元数据，可使用浏览器访问本地接口 "
        "http://localhost:8899/api/host/v1/projects/<project_id>。\n"
        "2. 拷贝返回 json 中 .data.meta 字段的值。\n"
        "3. 将该字段值覆盖粘贴到对应 project 目录下的 meta.json 中。\n"
        "4. 完成覆盖后重新打开/刷新模板管理，确认模板是否恢复。"
    ),
    "1.1.2.1.1": (
    "CT=max(capture time，detection time)\n"
    "1. 开启大量“模板匹配”增加时长；\n"
    "2. 开启“检测阶段使用算法生成焊盘框”增加时长；\n"
    "3. 生产环境开启保存ok小图增加时长；\n"
    "4. 板卡越大，器件越多耗时越久；\n"
    "5. 开启整板异物检测增加时长；\n"
    "6. 查看内存和显存占用情况，查看内存性能是否符合预期；\n"
    "7. 检查模板库是否存在大量替代料；\n"
    "8. 双轨设备同时检测会存在资源竞争，0.26.7之前的版本只缓存一个程序，切换调用模板消耗时间，0.26.7之后的版本缓存两个程序；\n"
    "9. 复判站smt-buddy-term中put res时间与主程序smt-buddy-host文件get res时间差为复判站与主程序之间数据的传输时长，"
    "主程序T(saving result) + 传输时长=T(review station)。"
    ),
    "1.1.3.1.1": (
        "1. 收集日志，查看运控日志，是否有日志 ACME.sdk.msdk.global ERROR - invalid map<K, T> key。\n"
        "2. 用MotionPanel.exe（地址：D:/AOI装机软件/运控卡SDK/PCI-9014/MotionPanel）查一下运控卡状态，"
        "如果是9014卡，新的网口运控卡是e450demo。"
    ),
    "1.1.3.1.2": (
        "1. 原因是系统安装包内可能存有残留文件没有清除；\n"
        "2. 处理方法为打开文档>ACME>review-station>conf，建议操作前先备份一个conf文件，然后删除conf文件内所有文件，重新开启软件；\n"
        "3. 删除conf文件夹里面所有东西就好，然后重新开软件，这几个文件软件会自动重新生成刚刚删除那几个文件；\n"
        "4. 客户08项目案例：现场更换工控机后打开主程序报警加载用户配置失败，疑是user.cfg.toml文件是空白文件导致。"
        "使用最近一次诊断日志的配置文件user.cfg.toml替换当前文件，可正常进入软件。之后修改第一项配置（如CAD旋转方向），再改回并重启软件，确认是否正常。"
    ),
    "1.1.1.1.2": (
        "1. buddy正常开启，但模板管理刷新后也没有模板，创建模板失败；\n"
        "2. 如果设备首次安装 buddy 版本为 0.11.2~0.11.4，升级到 0.14.x 后，可能会遇到权限问题导致 buddy 无法正常启动；\n"
        "3. 下载 useraccess_restore_使用管理员身份运行.bat 到电脑，关闭 buddy，右键使用管理员身份运行后恢复权限。"
    ),
    "1.2.1.1.1": (
        "1. 板弯补偿（3%以内）；\n"
        "2. 标记点区域。"
    ),
    "1.2.2.1.1": (
        "1. 检查导后的CAD格式是否正确，目前支持多种编码格式，若自动识别的编码格式错误，需手动选择正确的编码格式。\n"
        "2. 检查导后XY及角度位置是否正确。\n"
        "3. 检查导入的CAD有没有多余文字、空格符等。\n"
        "4. 检查CAD内是否有特殊符号。\n"
        "5. 检查CAD坐标是否为拼版坐标（位号不能有相同的，如果有做去重处理）。\n"
        "6. 检查坐标数值是否超出板卡尺寸。"
    ),
    "1.2.3": (
        "1. 正常没有这种特殊角度时，使用顺时针导CAD再开始做程序。\n"
        "2. 如果板上很多这种斜角度物料，先改逆时针再导CAD开始做程序。\n"
        "3. 做完程序后把设置改回顺时针，因为大部分都用顺时针。\n"
        "4. 如果按逆时针还是无法解决同一个料号角度不统一问题，继续检查 CAD 视图行列是否设置正确、是否设置反。\n"
        "5. 行列设置反时，同一个料号下角度可能出现 0/180 度。\n"
        "6. 客户版本 0.26 的场景中，展厅用最新版本没有复现行列设置反导致的方向不一致问题。"
    ),
    "1.2.4.1.1": (
        "1. 尝试使用模板匹配算法。\n"
        "2. 选择非圆点的特征部位作为匹配对象。"
    ),
    "1.2.4.1.2": (
        "1. 检查标记点选择的位置是否合理，参数是否合理。\n"
        "2. 通常情况下要选择形状匹配算法，通过调试模板强/弱阈值，使特征点分布能够充分体现 mark 点形状，同时尽量减少干扰的特征点（不在轮廓上的特征点）。\n"
        "3. 特殊情况下可以使用模板匹配算法。\n"
        "4. 进板方向错误时，检查板卡方向是否正确（尤其是鸳鸯板）。\n"
        "5. 进板不到位时，参考 2.4.1.7。"
    ),
    "1.2.4.1.3": (
        "1. 两轨夹边挡块不在同一位置时，远轨使用时可能盖住 mark 点导致报错；如果同一程序，需要将两条轨的夹边挡块固定的位置统一。\n"
        "2. 板边 mark 点因为挡块遮挡时，需要改成板内 mark 点。"
    ),
    "1.2.6.1.1": (
        "1. 远程回流数据，用最新版本验证问题。\n"
        "2. 若识别框准确，指导现场调试并更新模型/版本。\n"
        "3. 若识别框不准确，需要给标注团队提 jira，参考 jira 标准格式，提供数据。"
    ),
    "1.2.6.1.2": (
        "1. 调整 XY 轴和角度阈值。"
    ),
    "1.3.2": (
        "1. 收集日志。\n"
        "2. 查看日志是否有“500”报错。\n"
        "3. 在复判站中删除报错日志中的 json 文件。"
    ),
    "1.3.3": (
        "1. 检查板卡是否是弯板。\n"
        "2. 检查轨道大小是否合适（板子是否可以在轨道滑动）。\n"
        "3. 检查顶升是否把板子夹平（板子是否被夹弯）。\n"
        "4. 查看是否是挡块（器件）反光导致的（器件是否在挡块附近）。\n"
        "5. 检查器件本身是否存在差异、脏污、破损。"
    ),
    "1.3.4": (
        "1. 收集日志，查看日志加载板卡失败的原因是否为网络超时。\n"
        "2. 测试网络速度是否正常。\n"
        "3. 查看复盘站的 IP 是否正常。"
    ),
    "1.3.5": (
        "1. 将封装、OCR、极性同时加入模板库。"
    ),
    "1.4.1.1.1": (
        "1. 相机 IP 为自动获取方式识别不到时，改为 192.168.0.101，0 网段，后缀可随意改。"
    ),
    "1.4.1.1.2": (
        "1. 将软件全部退出，断电 1 分钟后重启。\n"
        "2. 2D 设备插拔光控（前面屏幕下面开门）。\n"
        "3. 查看系统 IP 连接（网络连接、ARM 连接），防火墙是否关闭。\n"
        "4. 收集日志，反馈到项目群，联系硬件。"
    ),
    "1.4.1.1.4": (
        "1. 查看运控日志是否有异常，无异常按照下方步骤操作。\n"
        "2. 打开网络适配器，查看网速是否正常，运控卡需要 100M。\n"
        "3. 网速异常时，点击相机的网口，右键点击属性。\n"
        "4. 点击配置后，找到 Speed & Duplex/连接速度和双工模式进行更改。"
    ),
}


def write_appendix_taxonomy() -> Path:
    path = BUILD_ROOT / "appendix1_owner_taxonomy.json"
    payload = {
        "schema_version": "debug_agent_system.appendix1_owner_taxonomy.v1",
        "source": "data/raw/现场问题反馈流程.md",
        "items": APPENDIX1_TAXONOMY,
    }
    _write_json(path, payload)
    return path


def write_section_id_map() -> Path:
    path = BUILD_ROOT / "section_id_map_main_program.json"
    payload = {
        "schema_version": "debug_agent_system.main_program_section_id_map.v1",
        "scope": "SOP/1.主程序/phase1-batch1",
        "items": SECTION_MAP,
    }
    _write_json(path, payload)
    return path


def write_section_inventory() -> Path:
    path = BUILD_ROOT / "section_inventory_main_program.json"
    payload = {
        "schema_version": "debug_agent_system.main_program_section_inventory.v1",
        "scope": "SOP/1.主程序/phase1-batch1",
        "items": [
            {
                "canonical_section_id": "1.1.1.1.1",
                "title": "闪退导致模板文件损坏，如何修复？",
                "source_title_path": ["主程序", "主页面", "模板管理", "闪退导致模板文件损坏，如何修复？"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.1.1.1.1"],
                "doc_order": 1,
                "image_ids": [],
                "appendix1_family_labels": ["Buddy问题"],
            },
            {
                "canonical_section_id": "1.1.2.1.1",
                "title": "CT时间长/复判站出图慢",
                "source_title_path": ["主程序", "主页面", "设置", "CT时间长/复判站出图慢"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.1.2.1.1"],
                "doc_order": 3,
                "image_ids": [],
                "appendix1_family_labels": ["软件使用及调试问题", "复判站软件问题"],
            },
            {
                "canonical_section_id": "1.1.1.1.2",
                "title": "buddy正常开启，模板管理刷新后也没有模板，创建模板失败。",
                "source_title_path": ["主程序", "主页面", "模板管理", "buddy正常开启，模板管理刷新后也没有模板，创建模板失败。"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.1.1.1.2"],
                "doc_order": 2,
                "image_ids": ["img:sop:1.1.1.1.2:buddy-create-timeout"],
                "appendix1_family_labels": ["Buddy问题"],
            },
            {
                "canonical_section_id": "1.1.3",
                "title": "设备主程序无法打开",
                "source_title_path": ["主程序", "主页面", "设备主程序无法打开"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": "父级条目；当前实际入图内容来自子条目 1.1.3.1.1。",
                "doc_order": 4,
                "image_ids": [],
                "appendix1_family_labels": ["运控问题"],
            },
            {
                "canonical_section_id": "1.1.3.1.1",
                "title": "设备主程序和工厂程序均无法打开",
                "source_title_path": ["主程序", "主页面", "设备主程序无法打开", "设备主程序和工厂程序均无法打开"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.1.3.1.1"],
                "doc_order": 5,
                "image_ids": ["img:sop:1.1.3.1.1:motionpanel"],
                "appendix1_family_labels": ["运控问题"],
            },
            {
                "canonical_section_id": "1.1.3.1.2",
                "title": "AOI主程序或者复判站主程序报:程序初始化失败:加载用户配置失败，处理思路",
                "source_title_path": ["主程序", "主页面", "设备主程序无法打开", "AOI主程序或者复判站主程序报:程序初始化失败:加载用户配置失败，处理思路"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.1.3.1.2"],
                "doc_order": 6,
                "image_ids": [
                    "img:sop:1.1.3.1.2:error-popup",
                    "img:sop:1.1.3.1.2:conf-delete-guide",
                    "img:sop:1.1.3.1.2:log-key-parse",
                    "img:sop:1.1.3.1.2:usercfg-empty",
                ],
                "appendix1_family_labels": ["主程序软件问题", "复判站软件问题"],
            },
            {
                "canonical_section_id": "1.2.1.1.1",
                "title": "板卡弯曲/板卡加工误差时如何编程",
                "source_title_path": ["主程序", "编程&编程优化", "整版", "板卡弯曲/板卡加工误差时如何编程"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.1.1.1"],
                "doc_order": 7,
                "image_ids": [],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.2.2.1.1",
                "title": "CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。",
                "source_title_path": ["主程序", "编程&编程优化", "CAD", "CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.2.1.1"],
                "doc_order": 8,
                "image_ids": [],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.2.3",
                "title": "板卡存在部分特殊斜角度位置，导入CAD同一料号有不一样角度，处理思路",
                "source_title_path": ["主程序", "编程&编程优化", "板卡存在部分特殊斜角度位置，导入CAD同一料号有不一样角度，处理思路"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.3"],
                "doc_order": 9,
                "image_ids": [],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.2.4.1.1",
                "title": "希望一套程序用在多台机器上，但是Mark点在不同机器上会轻微跑偏而且还可能被遮挡。",
                "source_title_path": ["主程序", "编程&编程优化", "Mark点", "希望一套程序用在多台机器上，但是Mark点在不同机器上会轻微跑偏而且还可能被遮挡。"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.4.1.1"],
                "doc_order": 10,
                "image_ids": ["img:sop:1.2.4.1.1:template-match-overview", "img:sop:1.2.4.1.1:non-round-feature"],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.2.4.1.2",
                "title": "mark点对齐失败",
                "source_title_path": ["主程序", "编程&编程优化", "Mark点", "mark点对齐失败"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.4.1.2"],
                "doc_order": 11,
                "image_ids": ["img:sop:1.2.4.1.2:shape-match", "img:sop:1.2.4.1.2:template-match-fallback"],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.2.4.1.3",
                "title": "双轨设备，近端程序拷贝到远端报mark点错误",
                "source_title_path": ["主程序", "编程&编程优化", "Mark点", "双轨设备，近端程序拷贝到远端报mark点错误"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.4.1.3"],
                "doc_order": 12,
                "image_ids": [],
                "appendix1_family_labels": ["硬件问题"],
            },
            {
                "canonical_section_id": "1.2.6.1.1",
                "title": "识别框大小不准确",
                "source_title_path": ["主程序", "编程&编程优化", "检测框", "识别框大小不准确"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.6.1.1"],
                "doc_order": 13,
                "image_ids": [],
                "appendix1_family_labels": ["模型优化问题"],
            },
            {
                "canonical_section_id": "1.2.6.1.2",
                "title": "侧立和翘脚漏报(客户06)",
                "source_title_path": ["主程序", "编程&编程优化", "检测框", "侧立和翘脚漏报(客户06)"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.2.6.1.2"],
                "doc_order": 14,
                "image_ids": ["img:sop:1.2.6.1.2:upright-miss-1", "img:sop:1.2.6.1.2:upright-miss-2"],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.3.2",
                "title": "复判站检测页面没有参考图，点击加载最新模板后也没图",
                "source_title_path": ["主程序", "检测", "复判站检测页面没有参考图，点击加载最新模板后也没图"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.3.2"],
                "doc_order": 15,
                "image_ids": ["img:sop:1.3.2:log-500", "img:sop:1.3.2:delete-json"],
                "appendix1_family_labels": ["复判站软件问题"],
            },
            {
                "canonical_section_id": "1.3.3",
                "title": "2D设备同一光源下，器件成像差异较大",
                "source_title_path": ["主程序", "检测", "2D设备同一光源下，器件成像差异较大"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.3.3"],
                "doc_order": 16,
                "image_ids": [],
                "appendix1_family_labels": ["硬件问题"],
            },
            {
                "canonical_section_id": "1.3.4",
                "title": "复盘站出现板卡加载时间长和加载板卡失败同时出现",
                "source_title_path": ["主程序", "检测", "复盘站出现板卡加载时间长和加载板卡失败同时出现"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.3.4"],
                "doc_order": 17,
                "image_ids": [],
                "appendix1_family_labels": ["复判站软件问题"],
            },
            {
                "canonical_section_id": "1.3.5",
                "title": "不同器件如何做替代料",
                "source_title_path": ["主程序", "检测", "不同器件如何做替代料"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.3.5"],
                "doc_order": 18,
                "image_ids": [],
                "appendix1_family_labels": ["软件使用及调试问题"],
            },
            {
                "canonical_section_id": "1.4.1.1.1",
                "title": "相机ip问题",
                "source_title_path": ["主程序", "初始化", "初始化失败", "相机ip问题"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.4.1.1.1"],
                "doc_order": 19,
                "image_ids": ["img:sop:1.4.1.1.1:camera-ip"],
                "appendix1_family_labels": ["硬件问题"],
            },
            {
                "canonical_section_id": "1.4.1.1.2",
                "title": "光源问题",
                "source_title_path": ["主程序", "初始化", "初始化失败", "光源问题"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.4.1.1.2"],
                "doc_order": 20,
                "image_ids": ["img:sop:1.4.1.1.2:light-control"],
                "appendix1_family_labels": ["硬件问题"],
            },
            {
                "canonical_section_id": "1.4.1.1.4",
                "title": "卡在初始化运动控制卡，运控闪退",
                "source_title_path": ["主程序", "初始化", "初始化失败", "卡在初始化运动控制卡，运控闪退"],
                "source_doc": "异常处理 - 标准操作流程（SOP）",
                "raw_text": SECTION_RAW_TEXTS["1.4.1.1.4"],
                "doc_order": 21,
                "image_ids": ["img:sop:1.4.1.1.4:adapter-speed", "img:sop:1.4.1.1.4:speed-duplex"],
                "appendix1_family_labels": ["运控问题"],
            },
        ],
    }
    _write_json(path, payload)
    return path


def write_family_map() -> Path:
    path = BUILD_ROOT / "family_map_main_program.json"
    payload = {
        "schema_version": "debug_agent_system.main_program_family_map.v1",
        "scope": "SOP/1.主程序/phase1-batch1",
        "items": [
            {
                "canonical_section_id": "1.1.1.1.1",
                "title": "闪退导致模板文件损坏，如何修复？",
                "family_assignments": [
                    {
                        "family_label": "Buddy问题",
                        "variant_label": "模板文件损坏导致模板无法加载",
                        "owner": "工程师C",
                        "rationale": "该 section 的修复方法直接围绕 Buddy 项目元数据与 meta.json 恢复，归属 Buddy 模板管理链路。",
                    }
                ],
                "split_required": False,
                "notes": "当前不拆成“闪退”和“模板损坏”两个 variant，保留为模板元数据损坏修复场景。",
            },
            {
                "canonical_section_id": "1.1.2.1.1",
                "title": "CT时间长/复判站出图慢",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "CT 时间异常增加",
                        "owner": "工程师丁",
                        "rationale": "该 section 中关于模板匹配、焊盘框、替代料、双轨缓存与资源竞争的内容属于软件使用与调试范围。",
                    },
                    {
                        "family_label": "复判站软件问题",
                        "variant_label": "复判站出图慢",
                        "owner": "工程师A",
                        "rationale": "该 section 中关于 put res / get res / review station 的内容属于复判站软件链路与主程序传输链路问题。",
                    },
                ],
                "split_required": True,
                "notes": "不允许保留单 family `CT时间长/复判站出图慢`。",
            },
            {
                "canonical_section_id": "1.1.1.1.2",
                "title": "buddy正常开启，模板管理刷新后也没有模板，创建模板失败。",
                "family_assignments": [
                    {
                        "family_label": "Buddy问题",
                        "variant_label": "Buddy 模板创建失败",
                        "owner": "工程师C",
                        "rationale": "该 section 明确描述 Buddy 模板管理无模板且创建模板失败，并给出版本升级后的权限修复动作。",
                    }
                ],
                "split_required": False,
                "notes": "当前先统一收敛为 Buddy 模板创建失败，不再拆成模板缺失/模板创建失败两个 variant。",
            },
            {
                "canonical_section_id": "1.1.3.1.1",
                "title": "设备主程序和工厂程序均无法打开",
                "family_assignments": [
                    {
                        "family_label": "运控问题",
                        "variant_label": "设备主程序和工厂程序均无法打开",
                        "owner": "工程师丑",
                        "rationale": "该 section 的动作全部围绕运控日志、运控卡状态、MotionPanel 和 9014/e450demo 卡状态检查，责任更偏运控链路而非主程序 UI 本身。",
                    }
                ],
                "split_required": False,
                "notes": "当前先不拆成“主程序无法打开/工厂程序无法打开”两个 variant，保持为同一运控故障变体。",
            },
            {
                "canonical_section_id": "1.1.3.1.2",
                "title": "AOI主程序或者复判站主程序报:程序初始化失败:加载用户配置失败，处理思路",
                "family_assignments": [
                    {
                        "family_label": "主程序软件问题",
                        "variant_label": "主程序加载用户配置失败",
                        "owner": "工程师庚",
                        "rationale": "标题明确覆盖 AOI 主程序；该类配置加载失败在附表1 归主程序软件问题。",
                    },
                    {
                        "family_label": "复判站软件问题",
                        "variant_label": "复判站加载用户配置失败",
                        "owner": "工程师A",
                        "rationale": "标题明确覆盖复判站主程序；同类配置加载失败在附表1 归复判站软件问题。",
                    },
                ],
                "split_required": True,
                "notes": "同一 section 同时覆盖主程序与复判站，两者共享动作链，但 family 归属不同。",
            },
            {
                "canonical_section_id": "1.2.1.1.1",
                "title": "板卡弯曲/板卡加工误差时如何编程",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "板卡弯曲/加工误差编程",
                        "owner": "工程师丁",
                        "rationale": "该 section 给出的是编程调试策略（板弯补偿、标记点区域），更接近软件使用与调试而不是硬件故障本体。",
                    }
                ],
                "split_required": False,
                "notes": "当前先收敛为一个 variant，不扩成独立硬件 family。",
            },
            {
                "canonical_section_id": "1.2.2.1.1",
                "title": "CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "CAD 导入失败",
                        "owner": "工程师丁",
                        "rationale": "该 section 给出的是 CAD 导入与编程参数检查链路，更接近软件使用及调试问题；按既有 KG_v2 约束收敛到 CAD 导入失败。",
                    }
                ],
                "split_required": False,
                "notes": "当前不再细拆为“尺寸过大”“不显示”等多个 variant，统一先收敛为 CAD 导入失败。",
            },
            {
                "canonical_section_id": "1.2.3",
                "title": "板卡存在部分特殊斜角度位置，导入CAD同一料号有不一样角度，处理思路",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "CAD 角度不一致",
                        "owner": "工程师丁",
                        "rationale": "该 section 的核心是导入 CAD 后同一料号角度不统一，且主要通过顺时针/逆时针与行列设置修正，因此应归到软件使用及调试问题下的 CAD 角度不一致。",
                    }
                ],
                "split_required": False,
                "notes": "当前先不再拆成更多子 variant，统一收敛为 CAD 角度不一致。",
            },
            {
                "canonical_section_id": "1.2.4.1.1",
                "title": "希望一套程序用在多台机器上，但是Mark点在不同机器上会轻微跑偏而且还可能被遮挡。",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "Mark 多机复用轻微跑偏/遮挡",
                        "owner": "工程师丁",
                        "rationale": "该 section 的核心是同一程序跨多机复用时 Mark 点轻微跑偏或被遮挡，解决策略是模板匹配与非圆点特征选择，因此归到软件使用及调试问题。",
                    }
                ],
                "split_required": False,
                "notes": "当前先收敛为单一 variant，不再拆成跑偏与遮挡两个子 variant。",
            },
            {
                "canonical_section_id": "1.2.4.1.2",
                "title": "mark点对齐失败",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "Mark 点对齐失败",
                        "owner": "工程师丁",
                        "rationale": "该 section 的核心是 Mark 点选择/参数、形状匹配算法与模板阈值调试，以及进板方向/进板不到位引起的对齐失败。",
                    }
                ],
                "split_required": False,
                "notes": "当前先收敛为单一 variant，不再拆成参数问题/进板方向问题/进板不到位多个子 variant。",
            },
            {
                "canonical_section_id": "1.2.6.1.1",
                "title": "识别框大小不准确",
                "family_assignments": [
                    {
                        "family_label": "模型优化问题",
                        "variant_label": "识别框大小不准确",
                        "owner": "工程师丙",
                        "rationale": "该 section 明确要求回流数据、用最新版本验证，并在识别框不准确时提 jira 给标注团队，责任更接近模型/标注优化链路。",
                    }
                ],
                "split_required": False,
                "notes": "当前先按模型优化问题建模；若后续积累到稳定现场调参闭环，再在案例层分流。",
            },
            {
                "canonical_section_id": "1.2.6.1.2",
                "title": "侧立和翘脚漏报(客户06)",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "侧立和翘脚漏报",
                        "owner": "工程师丁",
                        "rationale": "当前 SOP 只给出调节 XY 轴和角度阈值的处理动作，更接近现场调试与参数优化问题。",
                    }
                ],
                "split_required": False,
                "notes": "当前先不扩成模型问题，保留为阈值调试型 variant。",
            },
            {
                "canonical_section_id": "1.3.2",
                "title": "复判站检测页面没有参考图，点击加载最新模板后也没图",
                "family_assignments": [
                    {
                        "family_label": "复判站软件问题",
                        "variant_label": "复判站检测页面无参考图",
                        "owner": "工程师A",
                        "rationale": "该 section 的动作围绕复判站日志、500 报错与删除异常 json 文件，责任归属更接近复判站软件链路。",
                    }
                ],
                "split_required": False,
                "notes": "当前先收敛为复判站页面资源加载异常，不再拆成日志异常与模板刷新异常两个 variant。",
            },
            {
                "canonical_section_id": "1.3.3",
                "title": "2D设备同一光源下，器件成像差异较大",
                "family_assignments": [
                    {
                        "family_label": "硬件问题",
                        "variant_label": "同光源器件成像差异大",
                        "owner": "工程师甲",
                        "rationale": "当前排查链全部围绕板弯、轨道、顶升、挡块反光与器件实物差异，优先归属硬件侧。",
                    }
                ],
                "split_required": False,
                "notes": "当前不额外拆成板弯/轨道/反光多个子 variant，先保留为同光源成像差异问题。",
            },
            {
                "canonical_section_id": "1.3.4",
                "title": "复盘站出现板卡加载时间长和加载板卡失败同时出现",
                "family_assignments": [
                    {
                        "family_label": "复判站软件问题",
                        "variant_label": "复判站加载板卡超时/失败",
                        "owner": "工程师A",
                        "rationale": "当前动作围绕日志中的网络超时、网络速度与复判站 IP 配置，先归到复判站软件链路。",
                    }
                ],
                "split_required": False,
                "notes": "当前先不扩成系统网络问题 family，保留在复判站软件问题下。",
            },
            {
                "canonical_section_id": "1.3.5",
                "title": "不同器件如何做替代料",
                "family_assignments": [
                    {
                        "family_label": "软件使用及调试问题",
                        "variant_label": "不同器件替代料配置",
                        "owner": "工程师丁",
                        "rationale": "该 section 给出的是模板库配置方法，属于编程与调试知识，不是硬件故障。",
                    }
                ],
                "split_required": False,
                "notes": "当前按程序配置知识入图，不单独建流程知识外层。",
            },
            {
                "canonical_section_id": "1.4.1.1.1",
                "title": "相机ip问题",
                "family_assignments": [
                    {
                        "family_label": "硬件问题",
                        "variant_label": "相机IP自动获取识别不到",
                        "owner": "工程师甲",
                        "rationale": "当前处理动作是固定相机网段与 IP 配置，属于相机链路初始化侧问题。",
                    }
                ],
                "split_required": False,
                "notes": "当前不拆成网络配置与相机发现失败两个子 variant。",
            },
            {
                "canonical_section_id": "1.4.1.1.2",
                "title": "光源问题",
                "family_assignments": [
                    {
                        "family_label": "硬件问题",
                        "variant_label": "光源初始化异常",
                        "owner": "工程师甲",
                        "rationale": "当前动作全部围绕光控、ARM/IP连接、防火墙与日志，优先归到硬件链路。",
                    }
                ],
                "split_required": False,
                "notes": "当前先不拆成光控掉线和网络连接异常两个子 variant。",
            },
            {
                "canonical_section_id": "1.4.1.1.4",
                "title": "卡在初始化运动控制卡，运控闪退",
                "family_assignments": [
                    {
                        "family_label": "运控问题",
                        "variant_label": "初始化运动控制卡卡住/运控闪退",
                        "owner": "工程师丑",
                        "rationale": "当前排查链直接围绕运控日志与网卡 100M 速率设置，责任归属明确偏运控。",
                    }
                ],
                "split_required": False,
                "notes": "当前先保留为单一运控初始化变体。",
            }
        ],
    }
    _write_json(path, payload)
    return path


def _card_software_usage() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "软件使用/调试",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.1.2.1.1"],
        "variants": [
            {
                "label": "CT 时间异常增加",
                "summary": "检测节拍变长，CT 主要体现在 capture time 或 detection time 增大。",
                "error_phase": "检测/运行阶段",
                "owner_context": "SOP:1.1.2.1.1 | 附表1:软件使用及调试问题",
                "keywords": ["CT", "capture time", "detection time", "模板匹配", "焊盘框", "替代料", "双轨缓存"],
            }
        ],
        "actions": [
            {
                "label": "分解 CT 构成",
                "summary": "先确认 CT=max(capture time, detection time)，判断是拍摄耗时还是检测耗时主导。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查模板匹配开关是否过多",
                "summary": "排查是否开启大量模板匹配导致检测链耗时上升。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查检测阶段算法生成焊盘框是否开启",
                "summary": "确认是否启用了检测阶段使用算法生成焊盘框的功能，该功能会增加时长。",
                "action_role": "inspect",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查保存 OK 小图是否开启",
                "summary": "生产环境保存 OK 小图会增加处理耗时，需要确认是否开启。",
                "action_role": "inspect",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查整板异物检测是否开启",
                "summary": "确认是否开启整板异物检测，该能力会增加检测耗时。",
                "action_role": "inspect",
                "step_order": 5,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "查看内存和显存占用",
                "summary": "检查内存与显存占用情况，判断当前硬件性能是否符合预期。",
                "action_role": "observe",
                "step_order": 6,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查模板库是否存在大量替代料",
                "summary": "替代料过多会增加匹配与调度开销，需要确认模板库规模是否异常。",
                "action_role": "inspect",
                "step_order": 7,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查双轨资源竞争与缓存机制",
                "summary": "确认是否为双轨同时检测带来的资源竞争，或是否命中 0.26.7 前后缓存机制差异。",
                "action_role": "inspect",
                "step_order": 8,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "software_version",
                "question": "请提供主程序、算法包及 machine 版本。",
                "why_required": "需要确认是否命中 0.26.7 前后缓存机制与已知版本行为差异。",
                "condition": "检测节拍变长",
                "blocks": ["检查双轨资源竞争与缓存机制"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请提供当前设备的内存、显存、板卡尺寸、器件规模以及是否开启整板异物检测、保存 OK 小图等环境信息。",
                "why_required": "需要判断时长增加是由配置开关、板卡规模还是硬件资源瓶颈导致。",
                "condition": "CT 时间异常增加",
                "blocks": ["查看内存和显存占用", "检查整板异物检测是否开启", "检查保存 OK 小图是否开启"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明 CT 变长出现的工步、板型和复现步骤。",
                "why_required": "需要确认问题是否稳定复现，并把节拍增长落到具体工序。",
                "condition": "CT 时间异常增加",
                "blocks": ["分解 CT 构成"],
                "priority": "medium",
            },
            {
                "slot": "owner_context",
                "question": "请确认当前问题由调试负责人继续跟进，还是需要转交复判站负责人并行排查。",
                "why_required": "该 section 同时拆到调试链与复判站链，需要明确当前责任边界。",
                "condition": "CT 时间异常增加 / 复判站出图慢混合表述",
                "blocks": ["检查双轨资源竞争与缓存机制"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先分解 CT 构成，再依次排查模板匹配、焊盘框、OK 小图、异物检测、资源占用、替代料与双轨缓存机制。",
            "recommended_action_labels": [
                "分解 CT 构成",
                "检查模板匹配开关是否过多",
                "检查检测阶段算法生成焊盘框是否开启",
                "检查保存 OK 小图是否开启",
                "检查整板异物检测是否开启",
                "查看内存和显存占用",
                "检查模板库是否存在大量替代料",
                "检查双轨资源竞争与缓存机制",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 未发现静态图片可绑定到 action，暂不挂图。"],
    }


def _card_review_station() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "复判站软件问题",
            "summary": "附表1 中由复判站软件负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "复判站/软件",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师A",
            "owner_domain": "附表1:复判站软件问题",
        },
        "source_sections": ["1.1.2.1.1"],
        "variants": [
            {
                "label": "复判站出图慢",
                "summary": "复判站结果显示或主程序到复判站的数据传输链路变慢。",
                "error_phase": "复判站检测/显示阶段",
                "owner_context": "SOP:1.1.2.1.1 | 附表1:复判站软件问题",
                "keywords": ["put res", "get res", "review station", "saving result", "传输时长"],
            }
        ],
        "actions": [
            {
                "label": "对比 smt-buddy-term put res 与主程序 get res 时间差",
                "summary": "对比复判站 put res 和主程序 get res 的时间差，估算复判站与主程序之间的数据传输时长。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "判断主程序 saving result 与传输时长占比",
                "summary": "根据 T(saving result) + 传输时长 = T(review station) 判断瓶颈是否在主程序保存结果还是复判链路传输。",
                "action_role": "compare",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查复判站与主程序之间的数据传输链路",
                "summary": "确认复判站到主程序的网络/进程通信是否异常，是否存在链路级延迟。",
                "action_role": "inspect",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查是否属于双轨竞争或缓存切换",
                "summary": "确认是否因双轨同时检测、程序缓存切换或资源竞争导致复判站出图变慢。",
                "action_role": "inspect",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.2.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供复判站日志、主程序日志，以及包含 put res / get res 的时间片段。",
                "why_required": "需要直接定位复判站与主程序之间的时间差与瓶颈位置。",
                "condition": "复判站出图慢",
                "blocks": ["对比 smt-buddy-term put res 与主程序 get res 时间差", "判断主程序 saving result 与传输时长占比"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供复判站、主程序与 machine 版本。",
                "why_required": "需要判断是否命中版本级缓存/传输行为差异。",
                "condition": "复判站出图慢",
                "blocks": ["检查是否属于双轨竞争或缓存切换"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请提供双轨/单轨模式、板型规模、当前设备负载和资源使用情况。",
                "why_required": "需要判断是否属于双轨竞争、资源瓶颈或大板负载影响。",
                "condition": "复判站出图慢",
                "blocks": ["检查是否属于双轨竞争或缓存切换"],
                "priority": "medium",
            },
            {
                "slot": "ip_config",
                "question": "请提供复判站与主程序之间的网络配置和链路信息。",
                "why_required": "复判站出图慢可能来自主程序到复判站的数据传输链路异常。",
                "condition": "复判站出图慢",
                "blocks": ["检查复判站与主程序之间的数据传输链路"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先取 put res / get res 时间差，再判断 saving result 与传输时长占比，之后排查链路和双轨缓存竞争。",
            "recommended_action_labels": [
                "对比 smt-buddy-term put res 与主程序 get res 时间差",
                "判断主程序 saving result 与传输时长占比",
                "检查复判站与主程序之间的数据传输链路",
                "检查是否属于双轨竞争或缓存切换",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 未发现静态图片可绑定到 action，暂不挂图。"],
    }


def _card_buddy_template_create_fail() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "Buddy问题",
            "summary": "附表1 中由 Buddy 负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "Buddy/模板管理",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师C",
            "owner_domain": "附表1:Buddy问题",
        },
        "source_sections": ["1.1.1.1.2"],
        "variants": [
            {
                "label": "Buddy 模板创建失败",
                "summary": "Buddy 正常开启，但模板管理刷新后没有模板，创建模板时出现网络请求失败/超时。",
                "error_phase": "模板管理阶段",
                "owner_context": "SOP:1.1.1.1.2 | 附表1:Buddy问题",
                "keywords": ["Buddy", "模板管理", "创建模板失败", "Timeout", "0.11.2~0.11.4", "0.14.x"],
            }
        ],
        "actions": [
            {
                "label": "确认模板管理刷新后无模板且创建模板失败",
                "summary": "先确认当前表现为模板管理刷新后无模板，并且创建模板时报网络请求失败/超时。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.1.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.1.1.2:buddy-create-timeout",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0188_img_img_v3_02e4_519142d0-0767-485d-bf0c-718888465e1g.jpg_Ur8UbU9oyop1eXxWuV6cUad1nHh.jpg",
                        "source_section_id": "1.1.1.1.2",
                        "caption": "创建模板失败：网络请求失败/超时",
                        "why_this_image": "辅助确认当前 Buddy 模板创建失败的实际界面表现。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "检查 Buddy 版本升级路径是否命中 0.11.2~0.11.4 升级到 0.14.x",
                "summary": "确认设备首次安装的 Buddy 版本是否在 0.11.2~0.11.4，且当前已升级到 0.14.x，以判断是否命中已知权限问题。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.1.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "关闭 Buddy 并以管理员身份运行 useraccess_restore 脚本",
                "summary": "下载 useraccess_restore_使用管理员身份运行.bat 到电脑，关闭 Buddy 后以管理员身份运行，用于恢复权限。",
                "action_role": "change",
                "step_order": 3,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.1.1.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "恢复权限后重新打开 Buddy 并刷新模板管理验证",
                "summary": "执行权限恢复后，重新打开 Buddy，刷新模板管理并再次验证模板是否出现、创建是否恢复正常。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.1.1.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "software_version",
                "question": "请提供设备首次安装的 Buddy 版本，以及当前升级后的 Buddy 版本。",
                "why_required": "需要确认是否命中 0.11.2~0.11.4 升级到 0.14.x 的已知权限问题。",
                "condition": "Buddy 模板管理刷新后无模板且创建模板失败",
                "blocks": ["检查 Buddy 版本升级路径是否命中 0.11.2~0.11.4 升级到 0.14.x"],
                "priority": "high",
            },
            {
                "slot": "log_package",
                "question": "请提供 Buddy 日志和主程序日志，尤其是模板管理刷新与创建模板时段的日志。",
                "why_required": "需要确认是权限问题、网络请求超时，还是 Buddy 服务本身异常。",
                "condition": "Buddy 模板创建失败",
                "blocks": ["确认模板管理刷新后无模板且创建模板失败", "恢复权限后重新打开 Buddy 并刷新模板管理验证"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明刷新模板管理与创建模板失败的具体操作步骤和复现频率。",
                "why_required": "需要确认问题是否稳定复现，并在权限修复后进行同路径验证。",
                "condition": "Buddy 模板创建失败",
                "blocks": ["恢复权限后重新打开 Buddy 并刷新模板管理验证"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认模板管理无模板且创建失败，再核对版本升级路径，随后运行 useraccess_restore 恢复权限，最后重开 Buddy 验证。",
            "recommended_action_labels": [
                "确认模板管理刷新后无模板且创建模板失败",
                "检查 Buddy 版本升级路径是否命中 0.11.2~0.11.4 升级到 0.14.x",
                "关闭 Buddy 并以管理员身份运行 useraccess_restore 脚本",
                "恢复权限后重新打开 Buddy 并刷新模板管理验证",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "确认模板管理刷新后无模板且创建模板失败", "image_id": "img:sop:1.1.1.1.2:buddy-create-timeout", "source_section_id": "1.1.1.1.2"},
        ],
        "review_notes": ["当前 section 已绑定 1 张静态图，用于确认模板创建失败/超时界面。"],
    }


def _card_buddy_template_meta_repair() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "Buddy问题",
            "summary": "附表1 中由 Buddy 负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "Buddy/模板管理",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师C",
            "owner_domain": "附表1:Buddy问题",
        },
        "source_sections": ["1.1.1.1.1"],
        "variants": [
            {
                "label": "模板文件损坏导致模板无法加载",
                "summary": "主程序/模板管理闪退后，Buddy 项目 meta.json 损坏，导致模板信息无法正常读取。",
                "error_phase": "模板管理阶段",
                "owner_context": "SOP:1.1.1.1.1 | 附表1:Buddy问题",
                "keywords": ["闪退", "模板文件损坏", "meta.json", "Buddy", "模板管理"],
            }
        ],
        "actions": [
            {
                "label": "确认故障项目存在模板文件损坏现象",
                "summary": "先确认当前问题表现为闪退后模板无法正常加载，且需要针对具体项目目录进行修复。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.1.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "通过 Buddy 本地接口读取项目元数据",
                "summary": "使用浏览器访问 localhost:8899/api/host/v1/projects/<project_id>，读取目标项目的返回 json。",
                "action_role": "collect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.1.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "提取 .data.meta 并覆盖项目目录 meta.json",
                "summary": "从接口返回 json 中拷贝 .data.meta 字段值，覆盖粘贴到对应 project 目录下的 meta.json 中。",
                "action_role": "change",
                "step_order": 3,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.1.1.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "重新打开模板管理验证模板是否恢复",
                "summary": "完成 meta.json 覆盖后，重新打开或刷新模板管理，确认模板是否可以正常显示与读取。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.1.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "owner_context",
                "question": "请提供出现模板损坏的具体项目 ID 或对应 project 目录位置。",
                "why_required": "需要定位要访问的 Buddy 项目接口和本地 project 目录，才能修复目标 meta.json。",
                "condition": "模板文件损坏导致模板无法加载",
                "blocks": ["通过 Buddy 本地接口读取项目元数据", "提取 .data.meta 并覆盖项目目录 meta.json"],
                "priority": "high",
            },
            {
                "slot": "program_file",
                "question": "请提供当前损坏项目目录下的 meta.json 文件或其路径。",
                "why_required": "需要确认目标 meta.json 是否存在、是否可覆盖，以及修复前后的具体文件位置。",
                "condition": "模板文件损坏导致模板无法加载",
                "blocks": ["提取 .data.meta 并覆盖项目目录 meta.json"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明闪退发生时的操作步骤，以及修复后模板管理是否恢复正常。",
                "why_required": "需要确认这是闪退后的模板元数据损坏，而不是其他模板管理异常，并验证修复结果。",
                "condition": "模板文件损坏导致模板无法加载",
                "blocks": ["确认故障项目存在模板文件损坏现象", "重新打开模板管理验证模板是否恢复"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认模板损坏项目，再读取 Buddy 本地接口中的项目元数据，覆盖本地 meta.json，最后重新打开模板管理验证恢复情况。",
            "recommended_action_labels": [
                "确认故障项目存在模板文件损坏现象",
                "通过 Buddy 本地接口读取项目元数据",
                "提取 .data.meta 并覆盖项目目录 meta.json",
                "重新打开模板管理验证模板是否恢复",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 未发现静态图片，先不挂图。"],
    }


def _card_board_warp_programming() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "软件使用/调试",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.2.1.1.1"],
        "variants": [
            {
                "label": "板卡弯曲/加工误差编程",
                "summary": "板卡存在弯曲或加工误差时，编程侧需优先考虑板弯补偿与标记点区域策略。",
                "error_phase": "编程阶段",
                "owner_context": "SOP:1.2.1.1.1 | 附表1:软件使用及调试问题",
                "keywords": ["板弯补偿", "标记点区域", "整版编程", "加工误差"],
            }
        ],
        "actions": [
            {
                "label": "评估是否适用板弯补偿（3%以内）",
                "summary": "先判断板卡弯曲/加工误差是否在板弯补偿适用范围内，并优先考虑使用板弯补偿。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.1.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "设置标记点区域",
                "summary": "根据板弯或加工误差情况，结合标记点区域进行编程补偿与定位约束。",
                "action_role": "change",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.1.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "repro_steps",
                "question": "请说明板弯/加工误差在编程时的具体表现，以及当前板型和工步。",
                "why_required": "需要确认是整板编程阶段的几何补偿问题，而不是后续检测或运输问题。",
                "condition": "板卡弯曲/加工误差时编程异常",
                "blocks": ["评估是否适用板弯补偿（3%以内）"],
                "priority": "high",
            },
            {
                "slot": "sample_image",
                "question": "请提供板卡弯曲或标记点区域相关的示意图/拍照图。",
                "why_required": "需要结合实际板形与标记点位置判断板弯补偿和标记点区域如何设置。",
                "condition": "板卡弯曲/加工误差时编程异常",
                "blocks": ["设置标记点区域"],
                "priority": "medium",
            },
            {
                "slot": "program_file",
                "question": "请提供当前程序/CAD/标记点设置文件。",
                "why_required": "需要确认当前编程配置是否已经包含板弯补偿与合理的标记点区域。",
                "condition": "板卡弯曲/加工误差时编程异常",
                "blocks": ["设置标记点区域"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先评估板弯补偿是否适用，再结合标记点区域进行编程补偿。",
            "recommended_action_labels": [
                "评估是否适用板弯补偿（3%以内）",
                "设置标记点区域",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 未发现静态图，先不挂图。"],
    }


def _card_motion_program_open_fail() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "运控问题",
            "summary": "附表1 中由运控负责人处理的主程序专题问题。",
            "category": "硬件与运控",
            "subsystem": "运控/启动链路",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丑",
            "owner_domain": "附表1:运控问题",
        },
        "source_sections": ["1.1.3.1.1"],
        "variants": [
            {
                "label": "设备主程序和工厂程序均无法打开",
                "summary": "设备主程序和工厂程序均无法打开，需优先检查运控日志与运控卡状态。",
                "error_phase": "启动/打开阶段",
                "owner_context": "SOP:1.1.3.1.1 | 附表1:运控问题",
                "keywords": ["invalid map<K, T> key", "MotionPanel", "PCI-9014", "e450demo", "运控卡状态"],
            }
        ],
        "actions": [
            {
                "label": "收集日志并检查运控日志是否有 invalid map<K, T> key",
                "summary": "先收集日志，重点检查运控日志中是否出现 ACME.sdk.msdk.global ERROR - invalid map<K, T> key。",
                "action_role": "collect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "使用 MotionPanel 检查运控卡状态",
                "summary": "使用 D:/AOI装机软件/运控卡SDK/PCI-9014/MotionPanel 检查运控卡当前状态。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.1",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.1:motionpanel",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0190_img_image.png_Ib5HbibGlomEBcxKFgwco8EGnYc.png",
                        "source_section_id": "1.1.3.1.1",
                        "caption": "SOP 附图：设备主程序和工厂程序均无法打开",
                        "why_this_image": "辅助人工确认该条目对应的运控卡状态检查示意图。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "确认 9014 运控卡与 e450demo 网口卡对应关系",
                "summary": "如果是 9014 卡，进一步确认新的网口运控卡是否为 e450demo，并核对卡状态是否匹配。",
                "action_role": "compare",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供主程序打不开时的完整日志，尤其是运控日志。",
                "why_required": "需要先确认是否出现 invalid map<K, T> key 等明确运控异常信号。",
                "condition": "设备主程序和工厂程序均无法打开",
                "blocks": ["收集日志并检查运控日志是否有 invalid map<K, T> key"],
                "priority": "high",
            },
            {
                "slot": "device_model",
                "question": "请确认当前运控卡型号，以及是否为 9014 / e450demo 组合。",
                "why_required": "需要判断是否命中运控卡型号与网口卡对应关系异常。",
                "condition": "设备主程序和工厂程序均无法打开",
                "blocks": ["使用 MotionPanel 检查运控卡状态", "确认 9014 运控卡与 e450demo 网口卡对应关系"],
                "priority": "high",
            },
            {
                "slot": "owner_context",
                "question": "请确认当前问题是否由运控负责人继续跟进，或是否已转交工控机/系统负责人并行排查。",
                "why_required": "需要锁定当前责任边界，避免“主程序打不开”被误判成纯软件问题。",
                "condition": "设备主程序和工厂程序均无法打开",
                "blocks": ["确认 9014 运控卡与 e450demo 网口卡对应关系"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先收集日志并查运控报错，再用 MotionPanel 看运控卡状态，最后确认 9014 与 e450demo 的对应关系。",
            "recommended_action_labels": [
                "收集日志并检查运控日志是否有 invalid map<K, T> key",
                "使用 MotionPanel 检查运控卡状态",
                "确认 9014 运控卡与 e450demo 网口卡对应关系",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {
                "action_label": "使用 MotionPanel 检查运控卡状态",
                "image_id": "img:sop:1.1.3.1.1:motionpanel",
                "source_section_id": "1.1.3.1.1",
            }
        ],
        "review_notes": ["当前 section 已绑定 1 张静态图片到 MotionPanel 检查动作。"],
    }


def _card_main_program_config_load_fail() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "主程序软件问题",
            "summary": "附表1 中由主程序软件负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "主程序/配置链路",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师庚",
            "owner_domain": "附表1:主程序软件问题",
        },
        "source_sections": ["1.1.3.1.2"],
        "variants": [
            {
                "label": "主程序加载用户配置失败",
                "summary": "AOI 主程序启动时报程序初始化失败：加载用户配置失败，需优先排查 conf 与 user.cfg.toml。",
                "error_phase": "启动/初始化阶段",
                "owner_context": "SOP:1.1.3.1.2 | 附表1:主程序软件问题",
                "keywords": ["程序初始化失败", "加载用户配置失败", "conf", "user.cfg.toml", "主程序"],
            }
        ],
        "actions": [
            {
                "label": "确认弹窗报错为加载用户配置失败",
                "summary": "先确认当前主程序弹窗确认为“程序初始化失败：加载用户配置失败”。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.2:error-popup",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0191_img_img_v3_02rc_fa9669f1-ad80-488f-bee8-960039822afg.jpg_YiHFbeuYgomaKUxsrCGc8vLnnag.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "主程序弹窗：程序初始化失败-加载用户配置失败",
                        "why_this_image": "辅助确认该 section 的具体报错界面。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "备份并清空 conf 目录",
                "summary": "打开文档>ACME>review-station>conf，建议先备份 conf，再删除 conf 文件内所有文件后重新启动。",
                "action_role": "change",
                "step_order": 2,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.2:conf-delete-guide",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0192_img_img_v3_02rc_7f688500-a88e-472b-8a80-da75ed44cf0g.jpg_QQgUb8a1NocvPCxy6ercgAnPnZe.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "conf 目录清理示意图",
                        "why_this_image": "辅助人工定位并清空 conf 目录。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "检查日志与 user.cfg.toml 是否异常",
                "summary": "查看日志是否出现 key parse 相关异常，并检查 user.cfg.toml 是否为空白或损坏。",
                "action_role": "inspect",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.2:log-key-parse",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0193_img_img_v3_02rc_973d82df-f603-43c1-8125-07e6236f699g.jpg_ALrcbxsiwoI92cxgYDlcYjasnzh.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "日志与报错窗口叠加示意图",
                        "why_this_image": "辅助识别日志中的配置解析异常。",
                        "priority": "medium",
                    },
                    {
                        "image_id": "img:sop:1.1.3.1.2:usercfg-empty",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0194_img_img_v3_02rc_dad7a1de-3d79-43aa-beb3-86ebf88a301g.jpg_UzMIb5jhMouJuPxtCsecoKGwnLc.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "user.cfg.toml 为空白文件示意图",
                        "why_this_image": "辅助确认 user.cfg.toml 异常/空白这一类根因线索。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "用最近一次正常诊断日志中的 user.cfg.toml 替换验证",
                "summary": "使用最近一次诊断日志中的 user.cfg.toml 替换当前文件，再修改一项配置后改回并重启验证是否恢复正常。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "program_file",
                "question": "请提供当前 conf 目录内容以及 user.cfg.toml 文件。",
                "why_required": "需要确认是 conf 残留、配置损坏，还是 user.cfg.toml 空白异常。",
                "condition": "主程序加载用户配置失败",
                "blocks": ["备份并清空 conf 目录", "检查日志与 user.cfg.toml 是否异常"],
                "priority": "high",
            },
            {
                "slot": "log_package",
                "question": "请提供最近一次诊断日志及其中可用的配置文件。",
                "why_required": "需要用历史正常配置进行替换验证，并对比当前日志中的配置解析异常。",
                "condition": "主程序加载用户配置失败",
                "blocks": ["检查日志与 user.cfg.toml 是否异常", "用最近一次正常诊断日志中的 user.cfg.toml 替换验证"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供当前主程序版本及最近是否发生工控机更换或安装包重装。",
                "why_required": "需要判断是否为安装包残留与版本/环境变化引起的配置恢复问题。",
                "condition": "主程序加载用户配置失败",
                "blocks": ["备份并清空 conf 目录"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认错误提示，再备份并清空 conf，随后检查日志与 user.cfg.toml，最后用历史正常配置替换验证。",
            "recommended_action_labels": [
                "确认弹窗报错为加载用户配置失败",
                "备份并清空 conf 目录",
                "检查日志与 user.cfg.toml 是否异常",
                "用最近一次正常诊断日志中的 user.cfg.toml 替换验证",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "确认弹窗报错为加载用户配置失败", "image_id": "img:sop:1.1.3.1.2:error-popup", "source_section_id": "1.1.3.1.2"},
            {"action_label": "备份并清空 conf 目录", "image_id": "img:sop:1.1.3.1.2:conf-delete-guide", "source_section_id": "1.1.3.1.2"},
            {"action_label": "检查日志与 user.cfg.toml 是否异常", "image_id": "img:sop:1.1.3.1.2:log-key-parse", "source_section_id": "1.1.3.1.2"},
            {"action_label": "检查日志与 user.cfg.toml 是否异常", "image_id": "img:sop:1.1.3.1.2:usercfg-empty", "source_section_id": "1.1.3.1.2"},
        ],
        "review_notes": ["当前 section 已绑定 4 张静态图，分别辅助报错确认、conf 清理、日志判断与 user.cfg 异常确认。"],
    }


def _card_review_station_config_load_fail() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "复判站软件问题",
            "summary": "附表1 中由复判站软件负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "复判站/软件",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师A",
            "owner_domain": "附表1:复判站软件问题",
        },
        "source_sections": ["1.1.3.1.2"],
        "variants": [
            {
                "label": "复判站加载用户配置失败",
                "summary": "复判站主程序启动时报程序初始化失败：加载用户配置失败，需优先排查 conf 与 user.cfg.toml。",
                "error_phase": "启动/初始化阶段",
                "owner_context": "SOP:1.1.3.1.2 | 附表1:复判站软件问题",
                "keywords": ["程序初始化失败", "加载用户配置失败", "conf", "user.cfg.toml", "复判站"],
            }
        ],
        "actions": [
            {
                "label": "确认弹窗报错为加载用户配置失败",
                "summary": "先确认当前复判站弹窗确认为“程序初始化失败：加载用户配置失败”。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.2:error-popup",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0191_img_img_v3_02rc_fa9669f1-ad80-488f-bee8-960039822afg.jpg_YiHFbeuYgomaKUxsrCGc8vLnnag.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "复判站/主程序通用报错界面",
                        "why_this_image": "辅助确认该 section 的共同错误提示。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "备份并清空 conf 目录",
                "summary": "打开文档>ACME>review-station>conf，先备份再删除 conf 文件内所有文件，随后重新启动复判站软件。",
                "action_role": "change",
                "step_order": 2,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.2:conf-delete-guide",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0192_img_img_v3_02rc_7f688500-a88e-472b-8a80-da75ed44cf0g.jpg_QQgUb8a1NocvPCxy6ercgAnPnZe.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "review-station/conf 清理示意图",
                        "why_this_image": "该图直接展示了复判站 conf 路径与删除位置。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "检查日志与 user.cfg.toml 是否异常",
                "summary": "查看日志是否出现配置解析异常，并检查 user.cfg.toml 是否为空白或损坏。",
                "action_role": "inspect",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.1.3.1.2:log-key-parse",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0193_img_img_v3_02rc_973d82df-f603-43c1-8125-07e6236f699g.jpg_ALrcbxsiwoI92cxgYDlcYjasnzh.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "日志/错误窗口示意图",
                        "why_this_image": "辅助检查配置解析异常信号。",
                        "priority": "medium",
                    },
                    {
                        "image_id": "img:sop:1.1.3.1.2:usercfg-empty",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0194_img_img_v3_02rc_dad7a1de-3d79-43aa-beb3-86ebf88a301g.jpg_UzMIb5jhMouJuPxtCsecoKGwnLc.jpg",
                        "source_section_id": "1.1.3.1.2",
                        "caption": "空白 user.cfg.toml 示意图",
                        "why_this_image": "辅助确认复判站配置文件异常。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "用最近一次正常诊断日志中的 user.cfg.toml 替换验证",
                "summary": "使用最近一次正常诊断日志中的 user.cfg.toml 替换当前文件，再修改一项配置并改回后重启验证。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.1.3.1.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "program_file",
                "question": "请提供当前 review-station/conf 目录内容以及 user.cfg.toml 文件。",
                "why_required": "需要确认复判站配置目录是否残留异常文件，或 user.cfg.toml 是否空白损坏。",
                "condition": "复判站加载用户配置失败",
                "blocks": ["备份并清空 conf 目录", "检查日志与 user.cfg.toml 是否异常"],
                "priority": "high",
            },
            {
                "slot": "log_package",
                "question": "请提供最近一次正常诊断日志及其中可用的配置文件。",
                "why_required": "需要回填正常配置并对比当前日志中的配置异常信号。",
                "condition": "复判站加载用户配置失败",
                "blocks": ["检查日志与 user.cfg.toml 是否异常", "用最近一次正常诊断日志中的 user.cfg.toml 替换验证"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供当前复判站版本及最近是否发生工控机更换或安装包重装。",
                "why_required": "需要判断是否因安装残留或环境变化触发复判站配置问题。",
                "condition": "复判站加载用户配置失败",
                "blocks": ["备份并清空 conf 目录"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认错误提示，再备份并清空 conf，随后检查日志与 user.cfg.toml，最后用历史正常配置替换验证。",
            "recommended_action_labels": [
                "确认弹窗报错为加载用户配置失败",
                "备份并清空 conf 目录",
                "检查日志与 user.cfg.toml 是否异常",
                "用最近一次正常诊断日志中的 user.cfg.toml 替换验证",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "确认弹窗报错为加载用户配置失败", "image_id": "img:sop:1.1.3.1.2:error-popup", "source_section_id": "1.1.3.1.2"},
            {"action_label": "备份并清空 conf 目录", "image_id": "img:sop:1.1.3.1.2:conf-delete-guide", "source_section_id": "1.1.3.1.2"},
            {"action_label": "检查日志与 user.cfg.toml 是否异常", "image_id": "img:sop:1.1.3.1.2:log-key-parse", "source_section_id": "1.1.3.1.2"},
            {"action_label": "检查日志与 user.cfg.toml 是否异常", "image_id": "img:sop:1.1.3.1.2:usercfg-empty", "source_section_id": "1.1.3.1.2"},
        ],
        "review_notes": ["当前 section 已绑定 4 张静态图，分别辅助报错确认、conf 清理、日志判断与 user.cfg 异常确认。"],
    }


def _card_cad_import_failure() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "软件使用/调试",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.2.2.1.1"],
        "variants": [
            {
                "label": "CAD 导入失败",
                "summary": "导入 CAD 时出现解析失败、尺寸过大或导入后不显示。",
                "error_phase": "编程阶段",
                "owner_context": "SOP:1.2.2.1.1 | 附表1:软件使用及调试问题",
                "keywords": ["CAD", "解析失败", "尺寸过大", "导入后没显示", "编码格式", "拼版坐标"],
            }
        ],
        "actions": [
            {
                "label": "检查导入后的 CAD 编码格式是否正确",
                "summary": "确认导入后的 CAD 编码格式是否正确，若自动识别错误则手动选择正确编码格式。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查导后 XY 及角度位置是否正确",
                "summary": "确认导入后 XY 坐标与角度位置是否正常。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查 CAD 是否包含多余文字或空格符",
                "summary": "确认 CAD 中是否存在多余文字、空格符等导致解析异常。",
                "action_role": "inspect",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查 CAD 内是否存在特殊符号",
                "summary": "确认 CAD 内是否有异常特殊符号导致导入失败。",
                "action_role": "inspect",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查 CAD 坐标是否为拼版坐标",
                "summary": "确认 CAD 使用的是拼版坐标，且位号没有重复；如有重复需先去重。",
                "action_role": "inspect",
                "step_order": 5,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.2.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "检查坐标数值是否超出板卡尺寸",
                "summary": "确认 CAD 坐标数值没有超出板卡实际尺寸范围。",
                "action_role": "inspect",
                "step_order": 6,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.2.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "program_file",
                "question": "请提供当前导入的 CAD 文件。",
                "why_required": "需要直接检查 CAD 文件本体的编码、文字、特殊符号和坐标内容。",
                "condition": "CAD 导入失败",
                "blocks": ["检查导入后的 CAD 编码格式是否正确", "检查 CAD 是否包含多余文字或空格符", "检查 CAD 内是否存在特殊符号"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明导入 CAD 时的具体步骤，以及是在解析失败、尺寸过大还是导入后不显示。",
                "why_required": "需要确认当前命中的失败分支，以避免把解析失败、显示异常和尺寸异常混为一类。",
                "condition": "CAD 导入失败",
                "blocks": ["检查导后 XY 及角度位置是否正确", "检查坐标数值是否超出板卡尺寸"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供当前主程序版本。",
                "why_required": "需要确认是否命中特定版本的 CAD 导入行为差异。",
                "condition": "CAD 导入失败",
                "blocks": ["检查导入后的 CAD 编码格式是否正确"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认编码格式，再检查 XY/角度、文字空格、特殊符号、拼版坐标与尺寸范围。",
            "recommended_action_labels": [
                "检查导入后的 CAD 编码格式是否正确",
                "检查导后 XY 及角度位置是否正确",
                "检查 CAD 是否包含多余文字或空格符",
                "检查 CAD 内是否存在特殊符号",
                "检查 CAD 坐标是否为拼版坐标",
                "检查坐标数值是否超出板卡尺寸",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 未发现静态图，先不挂图。"],
    }


def _card_cad_angle_inconsistency() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "软件使用/调试",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.2.3"],
        "variants": [
            {
                "label": "CAD 角度不一致",
                "summary": "导入 CAD 后，同一料号在板上特殊斜角度位置出现角度不一致。",
                "error_phase": "编程阶段",
                "owner_context": "SOP:1.2.3 | 附表1:软件使用及调试问题",
                "keywords": ["顺时针", "逆时针", "角度不一致", "行列设置反", "0/180度"],
            }
        ],
        "actions": [
            {
                "label": "正常场景按顺时针导入 CAD",
                "summary": "当板上没有这种特殊斜角度物料时，先按顺时针导入 CAD 再开始做程序。",
                "action_role": "change",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.3",
                "curated_image_refs": [],
            },
            {
                "label": "特殊斜角物料场景改为逆时针导入 CAD",
                "summary": "如果板上有很多这种斜角度物料，先改逆时针再导 CAD 开始做程序。",
                "action_role": "change",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.3",
                "curated_image_refs": [],
            },
            {
                "label": "做完程序后将设置改回顺时针",
                "summary": "程序完成后把设置改回顺时针，因为大部分场景仍使用顺时针。",
                "action_role": "change",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.3",
                "curated_image_refs": [],
            },
            {
                "label": "检查 CAD 视图行列是否设置反",
                "summary": "如果按逆时针仍无法解决，则检查 CAD 视图中的行列设置是否反了。",
                "action_role": "inspect",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.3",
                "curated_image_refs": [],
            },
            {
                "label": "确认是否存在 0/180 度角度分裂",
                "summary": "行列设置反时，同一料号下角度可能表现为 0/180 度分裂，需要确认这一特征。",
                "action_role": "compare",
                "step_order": 5,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.3",
                "curated_image_refs": [],
            },
            {
                "label": "对比 0.26 客户版本与展厅新版本表现",
                "summary": "确认客户 0.26 版本与展厅新版本在行列设置反导致方向不一致上的差异。",
                "action_role": "compare",
                "step_order": 6,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.3",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "program_file",
                "question": "请提供当前程序文件和 CAD 文件。",
                "why_required": "需要确认当前导入 CAD 的方向设置、视图行列配置以及同料号角度分布。",
                "condition": "CAD 角度不一致",
                "blocks": ["检查 CAD 视图行列是否设置反", "确认是否存在 0/180 度角度分裂"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明哪些料号、哪些位置出现特殊斜角度，以及当前导 CAD 时使用的是顺时针还是逆时针。",
                "why_required": "需要确认问题是否只在特定斜角物料与导入方向组合下触发。",
                "condition": "CAD 角度不一致",
                "blocks": ["正常场景按顺时针导入 CAD", "特殊斜角物料场景改为逆时针导入 CAD"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供客户现场版本与展厅验证版本。",
                "why_required": "需要判断是否存在版本差异导致的行列设置行为不一致。",
                "condition": "CAD 角度不一致",
                "blocks": ["对比 0.26 客户版本与展厅新版本表现"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先按顺时针/逆时针切换导 CAD，再检查行列设置是否反，最后确认是否存在版本差异。",
            "recommended_action_labels": [
                "正常场景按顺时针导入 CAD",
                "特殊斜角物料场景改为逆时针导入 CAD",
                "做完程序后将设置改回顺时针",
                "检查 CAD 视图行列是否设置反",
                "确认是否存在 0/180 度角度分裂",
                "对比 0.26 客户版本与展厅新版本表现",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 的现有图片更像检测界面截图，不适合作为本条 action 的静态说明图，暂不挂图。"],
    }


def _card_mark_multi_machine_reuse() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "软件使用/调试",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.2.4.1.1"],
        "variants": [
            {
                "label": "Mark 多机复用轻微跑偏/遮挡",
                "summary": "同一套程序复用于多台机器时，Mark 点会轻微跑偏，且可能被遮挡。",
                "error_phase": "编程/调试阶段",
                "owner_context": "SOP:1.2.4.1.1 | 附表1:软件使用及调试问题",
                "keywords": ["Mark点", "多机复用", "轻微跑偏", "被遮挡", "模板匹配", "非圆点特征"],
            }
        ],
        "actions": [
            {
                "label": "尝试使用模板匹配算法",
                "summary": "面对多机复用导致的 Mark 点轻微跑偏或遮挡时，优先尝试模板匹配算法。",
                "action_role": "change",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.1",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.2.4.1.1:template-match-overview",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0196_img_img_v3_02ed_5a2adeec-2261-4459-acb6-2463a08add4g.jpg_VhHEbOLFTo43BgxWpc9cRzRwnub.jpg",
                        "source_section_id": "1.2.4.1.1",
                        "caption": "Mark 多机复用场景下的整体示意图",
                        "why_this_image": "辅助理解多机复用时 Mark 点与板边结构的整体位置关系。",
                        "priority": "medium",
                    }
                ],
            },
            {
                "label": "选择非圆点的特征部位作为匹配对象",
                "summary": "不要只盯圆形 Mark 点，优先选择非圆点的稳定特征部位作为模板匹配对象。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.1",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.2.4.1.1:non-round-feature",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0197_img_img_v3_02ed_c60f6e9c-604d-49c0-b3e7-cddeac20a65g.jpg_SKlTbJF3AobWQ6xd4ExcTuqTnnf.jpg",
                        "source_section_id": "1.2.4.1.1",
                        "caption": "非圆点特征部位模板匹配示意图",
                        "why_this_image": "直接展示本条 SOP 想表达的“非圆点特征部位”选取方式。",
                        "priority": "high",
                    }
                ],
            },
        ],
        "required_info": [
            {
                "slot": "sample_image",
                "question": "请提供当前设备上的 Mark 点成像截图，以及被遮挡或跑偏的实际示例图。",
                "why_required": "需要确认是特征选择问题、轻微偏移问题，还是实际遮挡问题。",
                "condition": "Mark 点在不同机器上轻微跑偏或被遮挡",
                "blocks": ["选择非圆点的特征部位作为匹配对象"],
                "priority": "high",
            },
            {
                "slot": "program_file",
                "question": "请提供当前程序和 Mark 点相关配置。",
                "why_required": "需要确认当前程序是否已经采用模板匹配算法，以及 Mark 点选取策略是否一致。",
                "condition": "Mark 多机复用异常",
                "blocks": ["尝试使用模板匹配算法"],
                "priority": "medium",
            },
            {
                "slot": "repro_steps",
                "question": "请说明是哪几台机器之间复用同一套程序，以及跑偏/遮挡出现的具体位置和频率。",
                "why_required": "需要确认异常是否与机器差异、板边结构或特定位置相关。",
                "condition": "Mark 多机复用异常",
                "blocks": ["尝试使用模板匹配算法", "选择非圆点的特征部位作为匹配对象"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先切到模板匹配算法，再选择非圆点特征部位作为 Mark 匹配对象。",
            "recommended_action_labels": [
                "尝试使用模板匹配算法",
                "选择非圆点的特征部位作为匹配对象",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "尝试使用模板匹配算法", "image_id": "img:sop:1.2.4.1.1:template-match-overview", "source_section_id": "1.2.4.1.1"},
            {"action_label": "选择非圆点的特征部位作为匹配对象", "image_id": "img:sop:1.2.4.1.1:non-round-feature", "source_section_id": "1.2.4.1.1"},
        ],
        "review_notes": ["当前 section 已绑定 2 张静态图，分别辅助整体理解与非圆点特征选择。"],
    }


def _card_mark_alignment_failure() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "软件使用/调试",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.2.4.1.2"],
        "variants": [
            {
                "label": "Mark 点对齐失败",
                "summary": "Mark 点选择/参数不合理，或进板方向/进板不到位导致 Mark 对齐失败。",
                "error_phase": "编程/调试阶段",
                "owner_context": "SOP:1.2.4.1.2 | 附表1:软件使用及调试问题",
                "keywords": ["Mark点对齐失败", "形状匹配算法", "模板强阈值", "模板弱阈值", "进板方向错误", "进板不到位"],
            }
        ],
        "actions": [
            {
                "label": "检查 Mark 点选择位置与参数是否合理",
                "summary": "先确认 Mark 点选择位置是否合理，参数是否合理。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "优先使用形状匹配算法并调试模板强弱阈值",
                "summary": "通常情况下优先选择形状匹配算法，通过调试模板强/弱阈值，使特征点充分体现 Mark 点形状并减少轮廓外干扰点。",
                "action_role": "change",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.2.4.1.2:shape-match",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0196_img_img_v3_02ed_5a2adeec-2261-4459-acb6-2463a08add4g.jpg_VhHEbOLFTo43BgxWpc9cRzRwnub.jpg",
                        "source_section_id": "1.2.4.1.2",
                        "caption": "形状匹配算法在板上 Mark 相关区域的示意图",
                        "why_this_image": "辅助理解 Mark 点轮廓与整板位置关系。",
                        "priority": "medium",
                    }
                ],
            },
            {
                "label": "特殊情况下切换到模板匹配算法",
                "summary": "若形状匹配不适用，则切换到模板匹配算法。",
                "action_role": "change",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.2.4.1.2:template-match-fallback",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0197_img_img_v3_02ed_c60f6e9c-604d-49c0-b3e7-cddeac20a65g.jpg_SKlTbJF3AobWQ6xd4ExcTuqTnnf.jpg",
                        "source_section_id": "1.2.4.1.2",
                        "caption": "模板匹配算法参数区示意图",
                        "why_this_image": "辅助理解“特殊情况下使用模板匹配算法”的界面位置。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "检查板卡方向是否正确",
                "summary": "若怀疑进板方向错误，先检查板卡方向是否正确，尤其关注鸳鸯板。",
                "action_role": "inspect",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "检查是否进板不到位",
                "summary": "若怀疑进板不到位导致 Mark 对齐失败，则转入 2.4.1.7 相关进板链路继续排查。",
                "action_role": "inspect",
                "step_order": 5,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "sample_image",
                "question": "请提供当前 Mark 点成像截图、模板特征点分布截图。",
                "why_required": "需要确认特征点是否真正落在 Mark 点轮廓上，以及是否有大量干扰点。",
                "condition": "Mark 点对齐失败",
                "blocks": ["检查 Mark 点选择位置与参数是否合理", "优先使用形状匹配算法并调试模板强弱阈值"],
                "priority": "high",
            },
            {
                "slot": "program_file",
                "question": "请提供当前程序和 Mark 点相关参数配置。",
                "why_required": "需要确认当前算法选择、参数阈值和 Mark 点配置是否合理。",
                "condition": "Mark 点对齐失败",
                "blocks": ["优先使用形状匹配算法并调试模板强弱阈值", "特殊情况下切换到模板匹配算法"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明是所有板都失败，还是特定方向/特定板型/特定进板场景失败。",
                "why_required": "需要区分参数问题、进板方向问题和进板不到位问题。",
                "condition": "Mark 点对齐失败",
                "blocks": ["检查板卡方向是否正确", "检查是否进板不到位"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先看 Mark 点选择与参数，再优先用形状匹配算法调阈值，必要时切模板匹配，最后排板卡方向和进板不到位。",
            "recommended_action_labels": [
                "检查 Mark 点选择位置与参数是否合理",
                "优先使用形状匹配算法并调试模板强弱阈值",
                "特殊情况下切换到模板匹配算法",
                "检查板卡方向是否正确",
                "检查是否进板不到位",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "优先使用形状匹配算法并调试模板强弱阈值", "image_id": "img:sop:1.2.4.1.2:shape-match", "source_section_id": "1.2.4.1.2"},
            {"action_label": "特殊情况下切换到模板匹配算法", "image_id": "img:sop:1.2.4.1.2:template-match-fallback", "source_section_id": "1.2.4.1.2"},
        ],
        "review_notes": ["当前 section 复用了 1.2.4.1.1 的两张图，但绑定到不同的算法选择动作，语义上仍然成立。"],
    }


def _card_far_track_mark_error() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "硬件问题",
            "summary": "附表1 中由硬件负责人处理的主程序专题问题。",
            "category": "硬件与运控",
            "subsystem": "轨道/挡块机构",
            "scenario": "主程序专题",
            "source_kind": "sop",
            "escalation_target": "工程师甲",
            "owner_domain": "附表1:硬件问题",
        },
        "source_sections": ["1.2.4.1.3"],
        "variants": [
            {
                "label": "远端报Mark点错误",
                "summary": "双轨设备中，近端程序拷贝到远端后，因夹边挡块位置不一致遮挡板边 Mark 点，导致远端报 mark 点错误。",
                "error_phase": "编程/调试阶段",
                "owner_context": "SOP:1.2.4.1.3 | 附表1:硬件问题",
                "keywords": ["双轨设备", "远端报mark点错误", "夹边挡块", "挡块遮挡", "板内mark点"],
            }
        ],
        "actions": [
            {
                "label": "检查近远轨夹边挡块位置是否一致",
                "summary": "先确认近轨与远轨夹边挡块的固定位置是否一致。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.3",
                "curated_image_refs": [],
            },
            {
                "label": "统一两条轨的夹边挡块位置",
                "summary": "如果同一程序需要在近远轨通用，则将两条轨的夹边挡块固定到统一位置。",
                "action_role": "change",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.3",
                "curated_image_refs": [],
            },
            {
                "label": "确认板边 Mark 点是否被挡块遮挡",
                "summary": "观察远轨运行时，板边 Mark 点是否被挡块实际遮挡。",
                "action_role": "observe",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.3",
                "curated_image_refs": [],
            },
            {
                "label": "将板边 Mark 点改成板内 Mark 点",
                "summary": "若板边 Mark 点被挡块遮挡，则改成板内 Mark 点。",
                "action_role": "change",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.4.1.3",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "sample_image",
                "question": "请提供近轨与远轨夹边挡块位置及 Mark 点区域的现场照片。",
                "why_required": "需要确认错误是否来自挡块位置不一致以及板边 Mark 点被遮挡。",
                "condition": "双轨设备近端程序拷贝到远端报 mark 点错误",
                "blocks": ["检查近远轨夹边挡块位置是否一致", "确认板边 Mark 点是否被挡块遮挡"],
                "priority": "high",
            },
            {
                "slot": "program_file",
                "question": "请提供当前近端/远端共用的程序文件与 Mark 点设置。",
                "why_required": "需要确认是否真的是同一程序跨轨复用，以及当前 Mark 点仍配置为板边。",
                "condition": "远端报Mark点错误",
                "blocks": ["统一两条轨的夹边挡块位置", "将板边 Mark 点改成板内 Mark 点"],
                "priority": "medium",
            },
            {
                "slot": "repro_steps",
                "question": "请说明错误是否只在远轨复用时出现，以及近端是否正常。",
                "why_required": "需要确认问题是轨间位置差异还是程序本身错误。",
                "condition": "远端报Mark点错误",
                "blocks": ["检查近远轨夹边挡块位置是否一致"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先查近远轨挡块位置是否一致，再确认板边 Mark 点是否被遮挡，最后统一挡块位置或改成板内 Mark 点。",
            "recommended_action_labels": [
                "检查近远轨夹边挡块位置是否一致",
                "统一两条轨的夹边挡块位置",
                "确认板边 Mark 点是否被挡块遮挡",
                "将板边 Mark 点改成板内 Mark 点",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 没有专门的示意图，先不挂图。"],
    }


def _card_detection_box_inaccurate() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "模型优化问题",
            "summary": "附表1 中由模型优化负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "检测框/模型输出",
            "scenario": "编程与检测框验证阶段",
            "source_kind": "sop",
            "escalation_target": "工程师丙",
            "owner_domain": "附表1:模型优化问题",
        },
        "source_sections": ["1.2.6.1.1"],
        "variants": [
            {
                "label": "识别框大小不准确",
                "summary": "检测框大小异常，需要通过数据回流与最新版本验证，区分现场调试问题与标注/模型问题。",
                "error_phase": "检测框验证阶段",
                "owner_context": "SOP:1.2.6.1.1 | 附表1:模型优化问题",
                "keywords": ["识别框", "检测框", "大小不准确", "回流数据", "jira", "标注团队"],
            }
        ],
        "actions": [
            {
                "label": "回流问题数据并在最新版本验证识别框",
                "summary": "先远程回流问题数据，在最新版本中复现并验证识别框是否仍然异常。",
                "action_role": "collect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "判断识别框是否实际准确",
                "summary": "基于最新版本验证结果，区分识别框本身是否准确，避免把现场调试问题误判成模型问题。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "识别框准确时指导现场调试并更新模型或版本",
                "summary": "若识别框本身准确，则指导现场做调试，并按需要更新模型或版本。",
                "action_role": "change",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "识别框不准确时按标准格式提交标注团队 Jira",
                "summary": "若识别框本身不准确，则按 jira 标准格式提交标注团队，并附带问题数据。",
                "action_role": "escalate",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供可回流的问题数据包，用于在最新版本中验证识别框。",
                "why_required": "需要先拿到原始问题数据，才能判断识别框异常是否稳定复现。",
                "condition": "识别框大小不准确",
                "blocks": ["回流问题数据并在最新版本验证识别框"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供当前现场版本与用于复测的最新版本信息。",
                "why_required": "需要区分是否已经被新版本修复，避免重复进入标注/Jira 流程。",
                "condition": "识别框大小不准确",
                "blocks": ["回流问题数据并在最新版本验证识别框", "识别框准确时指导现场调试并更新模型或版本"],
                "priority": "high",
            },
            {
                "slot": "sample_image",
                "question": "请提供识别框异常的典型截图或样本图。",
                "why_required": "需要把识别框异常表现与回流数据绑定，支持调试和 Jira 证据提交。",
                "condition": "识别框大小不准确",
                "blocks": ["判断识别框是否实际准确", "识别框不准确时按标准格式提交标注团队 Jira"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先回流数据并在最新版本验证，再判断识别框是否准确；准确则做现场调试，不准确则提交标注团队 Jira。",
            "recommended_action_labels": [
                "回流问题数据并在最新版本验证识别框",
                "判断识别框是否实际准确",
                "识别框准确时指导现场调试并更新模型或版本",
                "识别框不准确时按标准格式提交标注团队 Jira",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 没有独立静态图，先不挂图。"],
    }


def _card_upright_lead_miss() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "检测框/参数调试",
            "scenario": "检测框阈值调试阶段",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.2.6.1.2"],
        "variants": [
            {
                "label": "侧立和翘脚漏报",
                "summary": "侧立或翘脚场景存在漏报，需要优先调试 XY 轴与角度阈值。",
                "error_phase": "检测调试阶段",
                "owner_context": "SOP:1.2.6.1.2 | 附表1:软件使用及调试问题",
                "keywords": ["侧立", "翘脚", "漏报", "XY阈值", "角度阈值"],
            }
        ],
        "actions": [
            {
                "label": "确认问题表现为侧立或翘脚漏报",
                "summary": "先确认当前问题是侧立/翘脚场景下的漏报，而不是其它缺陷类型。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.2.6.1.2:upright-miss-1",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0198_img_115cd998997db74ab01f8e580c17e9c.jpg_MnSwbGdS9o7f75xiD1FcZqi0njf.jpg",
                        "source_section_id": "1.2.6.1.2",
                        "caption": "侧立/翘脚漏报示例图 1",
                        "why_this_image": "辅助确认漏报表现及阈值调试目标。",
                        "priority": "high",
                    },
                    {
                        "image_id": "img:sop:1.2.6.1.2:upright-miss-2",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0199_img_6772949dd6957e268207cf33bc01e1a.jpg_ZprLbys4WoT1Rvxzlk2cw36Qnwb.jpg",
                        "source_section_id": "1.2.6.1.2",
                        "caption": "侧立/翘脚漏报示例图 2",
                        "why_this_image": "辅助确认阈值调整前后的漏报形态。",
                        "priority": "high",
                    },
                ],
            },
            {
                "label": "调整 XY 轴阈值",
                "summary": "根据漏报表现先调节 XY 轴相关阈值，观察识别框与检出结果变化。",
                "action_role": "change",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "调整角度阈值",
                "summary": "继续调节角度阈值，改善侧立与翘脚场景下的漏报问题。",
                "action_role": "change",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "复测并确认漏报是否收敛",
                "summary": "调整阈值后结合样本图与现场板卡复测，确认漏报是否明显收敛。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.2.6.1.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "sample_image",
                "question": "请提供侧立或翘脚漏报的典型样本图和对应元件位置。",
                "why_required": "需要基于具体漏报样本做 XY/角度阈值调试，而不是盲调参数。",
                "condition": "侧立和翘脚漏报",
                "blocks": ["确认问题表现为侧立或翘脚漏报", "复测并确认漏报是否收敛"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供当前主程序/模型版本信息。",
                "why_required": "需要确认当前阈值调试是否受版本差异影响。",
                "condition": "侧立和翘脚漏报",
                "blocks": ["调整 XY 轴阈值", "调整角度阈值"],
                "priority": "medium",
            },
            {
                "slot": "repro_steps",
                "question": "请说明漏报发生的检测步骤、板型和复现频率。",
                "why_required": "需要确认阈值调试后的复测路径与现场触发条件一致。",
                "condition": "侧立和翘脚漏报",
                "blocks": ["复测并确认漏报是否收敛"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认侧立/翘脚漏报样本，再调 XY 阈值与角度阈值，最后复测确认是否收敛。",
            "recommended_action_labels": [
                "确认问题表现为侧立或翘脚漏报",
                "调整 XY 轴阈值",
                "调整角度阈值",
                "复测并确认漏报是否收敛",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "确认问题表现为侧立或翘脚漏报", "image_id": "img:sop:1.2.6.1.2:upright-miss-1", "source_section_id": "1.2.6.1.2"},
            {"action_label": "确认问题表现为侧立或翘脚漏报", "image_id": "img:sop:1.2.6.1.2:upright-miss-2", "source_section_id": "1.2.6.1.2"},
        ],
        "review_notes": ["当前 section 已绑定 2 张示例图，用于辅助阈值调试与复测。"],
    }


def _card_review_station_no_reference_images() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "复判站软件问题",
            "summary": "附表1 中由复判站软件负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "复判站/参考图加载",
            "scenario": "检测页面展示阶段",
            "source_kind": "sop",
            "escalation_target": "工程师A",
            "owner_domain": "附表1:复判站软件问题",
        },
        "source_sections": ["1.3.2"],
        "variants": [
            {
                "label": "复判站检测页面无参考图",
                "summary": "复判站检测页面没有参考图，点击加载最新模板后仍无图，需优先排查日志中的 500 报错与异常 json 文件。",
                "error_phase": "复判站检测页面",
                "owner_context": "SOP:1.3.2 | 附表1:复判站软件问题",
                "keywords": ["复判站", "没有参考图", "加载最新模板", "500报错", "json文件"],
            }
        ],
        "actions": [
            {
                "label": "收集复判站日志",
                "summary": "先收集复判站相关日志，保留当前无参考图问题发生时段的完整日志。",
                "action_role": "collect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.3.2",
                "curated_image_refs": [],
            },
            {
                "label": "检查日志是否存在 500 报错",
                "summary": "重点查看日志中是否存在 500 报错，用于判断是否为复判站侧资源加载异常。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.3.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.3.2:log-500",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0200_img_image.png_PQEGbUfKqo2xV5xGJ3QcaKqMnIf.png",
                        "source_section_id": "1.3.2",
                        "caption": "日志 500 报错示例",
                        "why_this_image": "辅助识别应关注的日志报错样式。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "删除复判站报错日志中的异常 json 文件",
                "summary": "根据报错日志定位异常 json 文件，并在复判站中删除对应文件后重试。",
                "action_role": "change",
                "step_order": 3,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.3.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.3.2:delete-json",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0201_img_image.png_WVCZbOAcMoxdHXxt1hQcY4Vdn9e.png",
                        "source_section_id": "1.3.2",
                        "caption": "删除异常 json 文件示意",
                        "why_this_image": "辅助确认需删除的异常 json 文件位置。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "重新加载最新模板验证参考图是否恢复",
                "summary": "删除异常 json 文件后，再次点击加载最新模板，确认参考图是否恢复显示。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.3.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供复判站日志，尤其是无参考图时段的完整日志。",
                "why_required": "需要先确认是否存在 500 报错以及对应的异常资源文件。",
                "condition": "复判站检测页面无参考图",
                "blocks": ["收集复判站日志", "检查日志是否存在 500 报错"],
                "priority": "high",
            },
            {
                "slot": "program_file",
                "question": "请提供日志中提到的异常 json 文件名或路径。",
                "why_required": "需要准确定位应删除的异常 json 文件，避免误删其它资源。",
                "condition": "复判站检测页面无参考图",
                "blocks": ["删除复判站报错日志中的异常 json 文件"],
                "priority": "high",
            },
            {
                "slot": "software_version",
                "question": "请提供当前复判站版本和主程序版本。",
                "why_required": "需要区分是否为版本相关的资源加载异常。",
                "condition": "复判站检测页面无参考图",
                "blocks": ["重新加载最新模板验证参考图是否恢复"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先收集日志并检查 500 报错，再删除异常 json 文件，最后重载模板确认参考图是否恢复。",
            "recommended_action_labels": [
                "收集复判站日志",
                "检查日志是否存在 500 报错",
                "删除复判站报错日志中的异常 json 文件",
                "重新加载最新模板验证参考图是否恢复",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "检查日志是否存在 500 报错", "image_id": "img:sop:1.3.2:log-500", "source_section_id": "1.3.2"},
            {"action_label": "删除复判站报错日志中的异常 json 文件", "image_id": "img:sop:1.3.2:delete-json", "source_section_id": "1.3.2"},
        ],
        "review_notes": ["当前 section 已绑定 2 张静态图，分别对应 500 报错示例与异常 json 文件删除示意。"],
    }


def _card_same_light_imaging_difference() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "硬件问题",
            "summary": "附表1 中由硬件负责人处理的主程序专题问题。",
            "category": "硬件与运控",
            "subsystem": "相机/轨道/顶升/器件实物",
            "scenario": "检测成像阶段",
            "source_kind": "sop",
            "escalation_target": "工程师甲",
            "owner_domain": "附表1:硬件问题",
        },
        "source_sections": ["1.3.3"],
        "variants": [
            {
                "label": "同光源器件成像差异大",
                "summary": "2D 设备同一光源下器件成像差异明显，需要优先排查板弯、轨道、顶升、反光和器件本体状态。",
                "error_phase": "检测成像阶段",
                "owner_context": "SOP:1.3.3 | 附表1:硬件问题",
                "keywords": ["2D设备", "同一光源", "成像差异", "弯板", "顶升", "反光", "器件脏污"],
            }
        ],
        "actions": [
            {"label": "检查板卡是否为弯板", "summary": "先检查板卡是否存在弯板问题。", "action_role": "inspect", "step_order": 1, "destructive": False, "high_cost": False, "source_section_id": "1.3.3", "curated_image_refs": []},
            {"label": "检查轨道宽度是否合适", "summary": "确认板子在轨道中是否可以滑动，避免轨道挤压导致成像差异。", "action_role": "inspect", "step_order": 2, "destructive": False, "high_cost": False, "source_section_id": "1.3.3", "curated_image_refs": []},
            {"label": "检查顶升是否把板子夹平", "summary": "确认顶升是否把板子夹弯或未夹平。", "action_role": "inspect", "step_order": 3, "destructive": False, "high_cost": False, "source_section_id": "1.3.3", "curated_image_refs": []},
            {"label": "检查挡块或邻近器件是否反光", "summary": "确认是否因挡块或邻近器件反光导致成像异常。", "action_role": "inspect", "step_order": 4, "destructive": False, "high_cost": False, "source_section_id": "1.3.3", "curated_image_refs": []},
            {"label": "检查器件本身是否存在差异脏污破损", "summary": "检查器件是否存在本体差异、脏污或破损。", "action_role": "inspect", "step_order": 5, "destructive": False, "high_cost": False, "source_section_id": "1.3.3", "curated_image_refs": []},
        ],
        "required_info": [
            {
                "slot": "sample_image",
                "question": "请提供同一光源下成像差异明显的器件截图或对比图。",
                "why_required": "需要确认是整体平整度问题、局部反光问题，还是器件本体差异问题。",
                "condition": "同光源器件成像差异大",
                "blocks": ["检查挡块或邻近器件是否反光", "检查器件本身是否存在差异脏污破损"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请提供当前板型、轨道宽度、顶升状态和是否存在弯板情况。",
                "why_required": "需要把成像差异与机械夹持和平整度条件绑定起来。",
                "condition": "同光源器件成像差异大",
                "blocks": ["检查板卡是否为弯板", "检查轨道宽度是否合适", "检查顶升是否把板子夹平"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明是哪些器件、哪些位置、什么批次或哪些板上出现同光源成像差异。",
                "why_required": "需要区分个别器件异常与整板/整机机械条件异常。",
                "condition": "同光源器件成像差异大",
                "blocks": ["检查器件本身是否存在差异脏污破损"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先查板弯、轨道和顶升，再查挡块反光和器件本体差异。",
            "recommended_action_labels": [
                "检查板卡是否为弯板",
                "检查轨道宽度是否合适",
                "检查顶升是否把板子夹平",
                "检查挡块或邻近器件是否反光",
                "检查器件本身是否存在差异脏污破损",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 没有独立静态图，先不挂图。"],
    }


def _card_review_station_board_load_timeout() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "复判站软件问题",
            "summary": "附表1 中由复判站软件负责人处理的主程序专题问题。",
            "category": "系统与软件异常",
            "subsystem": "复判站/板卡加载链路",
            "scenario": "检测页面加载阶段",
            "source_kind": "sop",
            "escalation_target": "工程师A",
            "owner_domain": "附表1:复判站软件问题",
        },
        "source_sections": ["1.3.4"],
        "variants": [
            {
                "label": "复判站加载板卡超时/失败",
                "summary": "复判站出现板卡加载时间长和加载失败同时出现，需要优先排查网络超时、网络速度与 IP 配置。",
                "error_phase": "复判站板卡加载阶段",
                "owner_context": "SOP:1.3.4 | 附表1:复判站软件问题",
                "keywords": ["复判站", "加载板卡失败", "加载时间长", "网络超时", "网络速度", "IP"],
            }
        ],
        "actions": [
            {"label": "收集日志并确认是否为网络超时", "summary": "先收集日志，确认加载板卡失败是否由网络超时触发。", "action_role": "collect", "step_order": 1, "destructive": False, "high_cost": False, "source_section_id": "1.3.4", "curated_image_refs": []},
            {"label": "测试网络速度是否正常", "summary": "检查复判站与主机之间的网络速度是否异常。", "action_role": "inspect", "step_order": 2, "destructive": False, "high_cost": False, "source_section_id": "1.3.4", "curated_image_refs": []},
            {"label": "检查复判站 IP 配置是否正常", "summary": "确认复判站网络配置和 IP 地址是否正确。", "action_role": "inspect", "step_order": 3, "destructive": False, "high_cost": False, "source_section_id": "1.3.4", "curated_image_refs": []},
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供复判站加载板卡失败时段的完整日志。",
                "why_required": "需要确认失败根因是否是网络超时，而不是其它资源加载异常。",
                "condition": "复判站加载板卡超时/失败",
                "blocks": ["收集日志并确认是否为网络超时"],
                "priority": "high",
            },
            {
                "slot": "ip_config",
                "question": "请提供复判站当前 IP 配置和网络拓扑信息。",
                "why_required": "需要直接验证 IP 配置是否正确，并判断网络链路是否异常。",
                "condition": "复判站加载板卡超时/失败",
                "blocks": ["检查复判站 IP 配置是否正常"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请提供网络测速结果、交换机/网线环境以及复现频率。",
                "why_required": "需要判断是偶发链路抖动还是稳定的网络性能问题。",
                "condition": "复判站加载板卡超时/失败",
                "blocks": ["测试网络速度是否正常"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先看日志是否网络超时，再测网络速度，最后查复判站 IP 配置。",
            "recommended_action_labels": [
                "收集日志并确认是否为网络超时",
                "测试网络速度是否正常",
                "检查复判站 IP 配置是否正常",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 没有独立静态图，先不挂图。"],
    }


def _card_substitute_material_configuration() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "软件使用及调试问题",
            "summary": "附表1 中由软件使用及调试负责人处理的主程序专题问题。",
            "category": "算法与程序调优",
            "subsystem": "模板库/编程配置",
            "scenario": "替代料配置阶段",
            "source_kind": "sop",
            "escalation_target": "工程师丁",
            "owner_domain": "附表1:软件使用及调试问题",
        },
        "source_sections": ["1.3.5"],
        "variants": [
            {
                "label": "不同器件替代料配置",
                "summary": "不同器件做替代料时，需要将封装、OCR、极性同时加入模板库。",
                "error_phase": "模板库配置阶段",
                "owner_context": "SOP:1.3.5 | 附表1:软件使用及调试问题",
                "keywords": ["替代料", "封装", "OCR", "极性", "模板库"],
            }
        ],
        "actions": [
            {
                "label": "将封装加入模板库",
                "summary": "先把替代料相关封装加入模板库。",
                "action_role": "change",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.3.5",
                "curated_image_refs": [],
            },
            {
                "label": "将 OCR 加入模板库",
                "summary": "再把 OCR 相关特征加入模板库。",
                "action_role": "change",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.3.5",
                "curated_image_refs": [],
            },
            {
                "label": "将极性加入模板库",
                "summary": "最后把极性一并加入模板库，形成完整替代料配置。",
                "action_role": "change",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.3.5",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "program_file",
                "question": "请提供当前程序或模板库配置，以确认替代料要写入的位置。",
                "why_required": "需要定位当前模板库配置上下文，避免把替代料写错位置。",
                "condition": "不同器件替代料配置",
                "blocks": ["将封装加入模板库", "将 OCR 加入模板库", "将极性加入模板库"],
                "priority": "high",
            },
            {
                "slot": "sample_image",
                "question": "请提供原器件与替代器件的封装、OCR、极性对比图。",
                "why_required": "需要明确哪些特征应同步加入模板库。",
                "condition": "不同器件替代料配置",
                "blocks": ["将封装加入模板库", "将 OCR 加入模板库", "将极性加入模板库"],
                "priority": "high",
            },
            {
                "slot": "repro_steps",
                "question": "请说明当前替代料配置的目标器件、使用场景和预期识别差异。",
                "why_required": "需要确认替代料配置的适用范围和验证路径。",
                "condition": "不同器件替代料配置",
                "blocks": ["将极性加入模板库"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "替代料配置时，封装、OCR、极性需要同时进入模板库。",
            "recommended_action_labels": [
                "将封装加入模板库",
                "将 OCR 加入模板库",
                "将极性加入模板库",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [],
        "review_notes": ["当前 section 没有独立静态图，先不挂图。"],
    }


def _card_camera_ip_problem() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "硬件问题",
            "summary": "附表1 中由硬件负责人处理的主程序专题问题。",
            "category": "硬件与运控",
            "subsystem": "相机/网络配置",
            "scenario": "初始化阶段",
            "source_kind": "sop",
            "escalation_target": "工程师甲",
            "owner_domain": "附表1:硬件问题",
        },
        "source_sections": ["1.4.1.1.1"],
        "variants": [
            {
                "label": "相机IP自动获取识别不到",
                "summary": "相机使用自动获取 IP 时无法被识别，需要改为固定 192.168.0.x 网段地址。",
                "error_phase": "初始化阶段",
                "owner_context": "SOP:1.4.1.1.1 | 附表1:硬件问题",
                "keywords": ["相机IP", "自动获取", "识别不到", "192.168.0.101", "0网段"],
            }
        ],
        "actions": [
            {
                "label": "确认相机使用自动获取 IP 且当前识别不到",
                "summary": "先确认当前相机是自动获取 IP 模式，并且软件侧无法识别。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.1",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.4.1.1.1:camera-ip",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0202_img_image.png_Bo9tbPsHNogesLxoUZncK1f9nDb.png",
                        "source_section_id": "1.4.1.1.1",
                        "caption": "相机 IP 配置示意",
                        "why_this_image": "辅助确认相机 IP 配置位置和目标网段。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "将相机 IP 改为 192.168.0.101 同网段固定地址",
                "summary": "把相机 IP 从自动获取改为 192.168.0.101，0 网段，后缀可按现场调整。",
                "action_role": "change",
                "step_order": 2,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.4.1.1.1",
                "curated_image_refs": [],
            },
            {
                "label": "重新初始化并验证相机是否恢复识别",
                "summary": "修改为固定 IP 后重新初始化，确认相机是否可以被正常识别。",
                "action_role": "verify",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.1",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "ip_config",
                "question": "请提供当前相机 IP 配置、主机网卡网段和相机发现结果。",
                "why_required": "需要确认是自动获取导致的发现失败，还是主机与相机不在同一网段。",
                "condition": "相机IP自动获取识别不到",
                "blocks": ["确认相机使用自动获取 IP 且当前识别不到", "将相机 IP 改为 192.168.0.101 同网段固定地址"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请提供当前是 2D 还是 3D 设备，以及涉及的相机数量。",
                "why_required": "需要确认现场相机拓扑，避免固定 IP 与其它相机冲突。",
                "condition": "相机IP自动获取识别不到",
                "blocks": ["将相机 IP 改为 192.168.0.101 同网段固定地址"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先确认相机处于自动获取 IP 且无法识别，再改成固定 0 网段地址，最后复测初始化。",
            "recommended_action_labels": [
                "确认相机使用自动获取 IP 且当前识别不到",
                "将相机 IP 改为 192.168.0.101 同网段固定地址",
                "重新初始化并验证相机是否恢复识别",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "确认相机使用自动获取 IP 且当前识别不到", "image_id": "img:sop:1.4.1.1.1:camera-ip", "source_section_id": "1.4.1.1.1"},
        ],
        "review_notes": ["当前 section 已绑定 1 张配置示意图。"],
    }


def _card_light_source_problem() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "硬件问题",
            "summary": "附表1 中由硬件负责人处理的主程序专题问题。",
            "category": "硬件与运控",
            "subsystem": "光源/光控/ARM连接",
            "scenario": "初始化阶段",
            "source_kind": "sop",
            "escalation_target": "工程师甲",
            "owner_domain": "附表1:硬件问题",
        },
        "source_sections": ["1.4.1.1.2"],
        "variants": [
            {
                "label": "光源初始化异常",
                "summary": "初始化时光源链路异常，需要优先断电重启、插拔光控，并检查 ARM/IP 连接与防火墙。",
                "error_phase": "初始化阶段",
                "owner_context": "SOP:1.4.1.1.2 | 附表1:硬件问题",
                "keywords": ["光源问题", "光控", "断电重启", "ARM连接", "防火墙"],
            }
        ],
        "actions": [
            {
                "label": "退出软件并断电 1 分钟后重启",
                "summary": "先将软件全部退出，断电 1 分钟后重新上电启动。",
                "action_role": "change",
                "step_order": 1,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.4.1.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "2D设备插拔光控",
                "summary": "对 2D 设备插拔光控（前面屏幕下面开门处）以恢复光源控制链路。",
                "action_role": "change",
                "step_order": 2,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.4.1.1.2",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.4.1.1.2:light-control",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0203_img_image.png_TFPgbgxbroJY5VxqKJscmPInnlb.png",
                        "source_section_id": "1.4.1.1.2",
                        "caption": "光控位置示意",
                        "why_this_image": "辅助定位 2D 设备光控插拔位置。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "检查系统 IP 连接和 ARM 连接及防火墙",
                "summary": "检查网络连接、ARM 连接状态，并确认防火墙是否关闭。",
                "action_role": "inspect",
                "step_order": 3,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.2",
                "curated_image_refs": [],
            },
            {
                "label": "收集日志并反馈项目群联系硬件",
                "summary": "若仍异常，则收集日志，反馈到项目群并联系硬件同事继续排查。",
                "action_role": "escalate",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.2",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供光源初始化异常时段的完整日志。",
                "why_required": "需要把重启、插拔光控后的结果与日志信号对应起来。",
                "condition": "光源初始化异常",
                "blocks": ["收集日志并反馈项目群联系硬件"],
                "priority": "high",
            },
            {
                "slot": "ip_config",
                "question": "请提供系统网络连接、ARM 连接状态和相关 IP 配置。",
                "why_required": "需要确认是否为网络或 ARM 连接异常导致的光源初始化问题。",
                "condition": "光源初始化异常",
                "blocks": ["检查系统 IP 连接和 ARM 连接及防火墙"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请说明当前设备类型（2D/3D）、光控状态和是否已做过断电重启。",
                "why_required": "需要区分不同设备类型的光源链路与已尝试动作。",
                "condition": "光源初始化异常",
                "blocks": ["退出软件并断电 1 分钟后重启", "2D设备插拔光控"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先断电重启，再插拔光控，之后检查 ARM/IP/防火墙，最后收集日志升级硬件。",
            "recommended_action_labels": [
                "退出软件并断电 1 分钟后重启",
                "2D设备插拔光控",
                "检查系统 IP 连接和 ARM 连接及防火墙",
                "收集日志并反馈项目群联系硬件",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "2D设备插拔光控", "image_id": "img:sop:1.4.1.1.2:light-control", "source_section_id": "1.4.1.1.2"},
        ],
        "review_notes": ["当前 section 已绑定 1 张光控位置示意图。"],
    }


def _card_motion_control_init_stuck() -> dict[str, Any]:
    return {
        "schema_version": "debug_agent_system.main_program_manual_card.v1",
        "status": "approved_for_phase1_build",
        "family": {
            "label": "运控问题",
            "summary": "附表1 中由运控负责人处理的主程序专题问题。",
            "category": "硬件与运控",
            "subsystem": "运控卡/网卡速率",
            "scenario": "初始化阶段",
            "source_kind": "sop",
            "escalation_target": "工程师丑",
            "owner_domain": "附表1:运控问题",
        },
        "source_sections": ["1.4.1.1.4"],
        "variants": [
            {
                "label": "初始化运动控制卡卡住/运控闪退",
                "summary": "初始化阶段卡在运动控制卡，或运控程序闪退，需要优先看运控日志和网卡 100M 速率设置。",
                "error_phase": "初始化阶段",
                "owner_context": "SOP:1.4.1.1.4 | 附表1:运控问题",
                "keywords": ["初始化运动控制卡", "运控闪退", "网速100M", "Speed & Duplex", "运控日志"],
            }
        ],
        "actions": [
            {
                "label": "检查运控日志是否有异常",
                "summary": "先查看运控日志是否有明确异常；若无异常，再继续检查网卡速率链路。",
                "action_role": "inspect",
                "step_order": 1,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.4",
                "curated_image_refs": [],
            },
            {
                "label": "检查网络适配器速率是否为 100M",
                "summary": "打开网络适配器查看网速是否正常，运控卡需要 100M。",
                "action_role": "inspect",
                "step_order": 2,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.4",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.4.1.1.4:adapter-speed",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0204_img_image.png_LIkxbOPhTonsvpxMidFcQ2xxn0j.png",
                        "source_section_id": "1.4.1.1.4",
                        "caption": "网络适配器速率检查示意",
                        "why_this_image": "辅助定位网卡速率检查入口。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "网速异常时修改 Speed & Duplex",
                "summary": "若速率异常，则进入网口配置并修改 Speed & Duplex/连接速度和双工模式。",
                "action_role": "change",
                "step_order": 3,
                "destructive": True,
                "high_cost": False,
                "source_section_id": "1.4.1.1.4",
                "curated_image_refs": [
                    {
                        "image_id": "img:sop:1.4.1.1.4:speed-duplex",
                        "relative_path": "data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/embedded_media/0205_img_image.png_NLmQb0mpLoRxsNxlKEscIlNenrf.png",
                        "source_section_id": "1.4.1.1.4",
                        "caption": "Speed & Duplex 设置示意",
                        "why_this_image": "辅助定位需要修改的网卡速率配置项。",
                        "priority": "high",
                    }
                ],
            },
            {
                "label": "重新初始化验证运控是否恢复",
                "summary": "修改速率设置后重新初始化，确认是否不再卡在运动控制卡且运控程序不再闪退。",
                "action_role": "verify",
                "step_order": 4,
                "destructive": False,
                "high_cost": False,
                "source_section_id": "1.4.1.1.4",
                "curated_image_refs": [],
            },
        ],
        "required_info": [
            {
                "slot": "log_package",
                "question": "请提供运控日志和初始化失败时段的系统日志。",
                "why_required": "需要先确认日志是否已有明确异常，避免盲目调整网卡速率。",
                "condition": "初始化运动控制卡卡住/运控闪退",
                "blocks": ["检查运控日志是否有异常"],
                "priority": "high",
            },
            {
                "slot": "ip_config",
                "question": "请提供当前网卡速率、网口配置和相机/运控相关网卡信息。",
                "why_required": "需要确认是否命中网卡速率不为 100M 的问题。",
                "condition": "初始化运动控制卡卡住/运控闪退",
                "blocks": ["检查网络适配器速率是否为 100M", "网速异常时修改 Speed & Duplex"],
                "priority": "high",
            },
            {
                "slot": "environment",
                "question": "请说明当前运控卡型号、网口环境和问题复现频率。",
                "why_required": "需要确认是否为特定运控卡/网口环境下的初始化异常。",
                "condition": "初始化运动控制卡卡住/运控闪退",
                "blocks": ["重新初始化验证运控是否恢复"],
                "priority": "medium",
            },
        ],
        "trace": {
            "summary": "先看运控日志，再查网卡是否 100M，必要时改 Speed & Duplex，最后复测初始化。",
            "recommended_action_labels": [
                "检查运控日志是否有异常",
                "检查网络适配器速率是否为 100M",
                "网速异常时修改 Speed & Duplex",
                "重新初始化验证运控是否恢复",
            ],
            "actual_action_labels": [],
        },
        "image_bindings": [
            {"action_label": "检查网络适配器速率是否为 100M", "image_id": "img:sop:1.4.1.1.4:adapter-speed", "source_section_id": "1.4.1.1.4"},
            {"action_label": "网速异常时修改 Speed & Duplex", "image_id": "img:sop:1.4.1.1.4:speed-duplex", "source_section_id": "1.4.1.1.4"},
        ],
        "review_notes": ["当前 section 已绑定 2 张静态图，分别对应网卡速率检查与 Speed & Duplex 设置。"],
    }


def write_manual_cards() -> list[Path]:
    cards = [
        (BUILD_ROOT / "manual_cards/main_program/family_Buddy问题_模板文件损坏修复.json", _card_buddy_template_meta_repair()),
        (BUILD_ROOT / "manual_cards/main_program/family_Buddy问题_模板创建失败.json", _card_buddy_template_create_fail()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题.json", _card_software_usage()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_板卡弯曲加工误差编程.json", _card_board_warp_programming()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_CAD导入失败.json", _card_cad_import_failure()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_CAD角度不一致.json", _card_cad_angle_inconsistency()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_Mark多机复用轻微跑偏遮挡.json", _card_mark_multi_machine_reuse()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_Mark点对齐失败.json", _card_mark_alignment_failure()),
        (BUILD_ROOT / "manual_cards/main_program/family_硬件问题_远端报Mark点错误.json", _card_far_track_mark_error()),
        (BUILD_ROOT / "manual_cards/main_program/family_模型优化问题_识别框大小不准确.json", _card_detection_box_inaccurate()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_侧立和翘脚漏报.json", _card_upright_lead_miss()),
        (BUILD_ROOT / "manual_cards/main_program/family_复判站软件问题_检测页面无参考图.json", _card_review_station_no_reference_images()),
        (BUILD_ROOT / "manual_cards/main_program/family_硬件问题_同光源器件成像差异大.json", _card_same_light_imaging_difference()),
        (BUILD_ROOT / "manual_cards/main_program/family_复判站软件问题_加载板卡超时失败.json", _card_review_station_board_load_timeout()),
        (BUILD_ROOT / "manual_cards/main_program/family_软件使用及调试问题_不同器件替代料配置.json", _card_substitute_material_configuration()),
        (BUILD_ROOT / "manual_cards/main_program/family_硬件问题_相机IP自动获取识别不到.json", _card_camera_ip_problem()),
        (BUILD_ROOT / "manual_cards/main_program/family_硬件问题_光源初始化异常.json", _card_light_source_problem()),
        (BUILD_ROOT / "manual_cards/main_program/family_运控问题_初始化运动控制卡卡住运控闪退.json", _card_motion_control_init_stuck()),
        (BUILD_ROOT / "manual_cards/main_program/family_复判站软件问题.json", _card_review_station()),
        (BUILD_ROOT / "manual_cards/main_program/family_运控问题.json", _card_motion_program_open_fail()),
        (BUILD_ROOT / "manual_cards/main_program/family_主程序软件问题_加载用户配置失败.json", _card_main_program_config_load_fail()),
        (BUILD_ROOT / "manual_cards/main_program/family_复判站软件问题_加载用户配置失败.json", _card_review_station_config_load_fail()),
    ]
    out = []
    for path, payload in cards:
        _write_json(path, payload)
        out.append(path)
    return out


def _load_cards() -> list[dict[str, Any]]:
    cards = []
    card_root = BUILD_ROOT / "manual_cards/main_program"
    for path in sorted(card_root.glob("*.json")):
        cards.append(json.loads(path.read_text(encoding="utf-8")))
    return cards


def build_graph() -> dict[str, Any]:
    cards = _load_cards()
    objects: dict[str, list[dict[str, Any]]] = {
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
    relations: list[dict[str, Any]] = []

    def add_rel(src: str, dst: str, rel: str) -> None:
        relations.append({"from": src, "to": dst, "relation": rel})

    for card in cards:
        family_raw = card["family"]
        family = {
            "family_id": _family_id(family_raw["label"]),
            "label": family_raw["label"],
            "summary": family_raw["summary"],
            "category": family_raw["category"],
            "subsystem": family_raw["subsystem"],
            "scenario": family_raw["scenario"],
            "keywords": [],
            "source_kind": family_raw["source_kind"],
            "escalation_target": family_raw["escalation_target"],
            "owner_domain": family_raw["owner_domain"],
        }
        if not any(x["family_id"] == family["family_id"] for x in objects["FaultFamily"]):
            objects["FaultFamily"].append(family)

        for variant_raw in card["variants"]:
            variant = {
                "variant_id": _variant_id(family["label"], variant_raw["label"]),
                "family_id": family["family_id"],
                "label": variant_raw["label"],
                "summary": variant_raw["summary"],
                "equipment_type": "",
                "site": "",
                "software_version": "",
                "error_phase": variant_raw["error_phase"],
                "owner_context": variant_raw["owner_context"],
                "escalation_target": family["escalation_target"],
                "keywords": variant_raw["keywords"],
            }
            objects["FaultVariant"].append(variant)
            add_rel(family["family_id"], variant["variant_id"], "has_variant")

            source_section = card["source_sections"][0]
            case = {
                "case_id": _case_id(source_section, family["label"], variant["label"]),
                "source_kind": "sop",
                "title": variant["label"],
                "summary": variant["summary"],
                "source_ref": source_section,
                "approved": True,
            }
            evidence = {
                "evidence_id": _evidence_id(source_section, family["label"]),
                "source_kind": "sop",
                "external_id": source_section,
                "title": f"SOP {source_section}",
                "summary": SECTION_RAW_TEXTS.get(source_section, source_section),
                "payload_ref": "异常处理 - 标准操作流程（SOP）",
            }
            objects["SourceCase"].append(case)
            objects["EvidenceItem"].append(evidence)
            add_rel(case["case_id"], variant["variant_id"], "supports")
            add_rel(evidence["evidence_id"], case["case_id"], "evidences")

            action_ids: list[str] = []
            action_label_to_id: dict[str, str] = {}
            for action_raw in card["actions"]:
                action = {
                    "action_id": _action_id(variant["variant_id"], int(action_raw["step_order"]), action_raw["label"]),
                    "family_id": family["family_id"],
                    "variant_id": variant["variant_id"],
                    "label": action_raw["label"],
                    "summary": action_raw["summary"],
                    "action_role": action_raw["action_role"],
                    "step_order": int(action_raw["step_order"]),
                    "destructive": bool(action_raw["destructive"]),
                    "high_cost": bool(action_raw["high_cost"]),
                    "source_kind": "sop",
                    "source_section_id": action_raw["source_section_id"],
                    "curated_image_refs": list(action_raw.get("curated_image_refs") or []),
                }
                objects["DiagnosticAction"].append(action)
                action_ids.append(action["action_id"])
                action_label_to_id[action["label"]] = action["action_id"]
            req_ids: list[str] = []
            for req_raw in card["required_info"]:
                req = {
                    "required_info_id": _required_id(variant["variant_id"], req_raw["slot"], req_raw["question"]),
                    "family_id": family["family_id"],
                    "variant_id": variant["variant_id"],
                    "slot": req_raw["slot"],
                    "question": req_raw["question"],
                    "why_required": req_raw["why_required"],
                    "condition": req_raw["condition"],
                    "blocks": req_raw["blocks"],
                    "priority": req_raw["priority"],
                    "evidence_ids": [evidence["evidence_id"]],
                }
                objects["RequiredInfoSpec"].append(req)
                req_ids.append(req["required_info_id"])
                add_rel(variant["variant_id"], req["required_info_id"], "has_required_info")
                add_rel(case["case_id"], req["required_info_id"], "supports")
                add_rel(evidence["evidence_id"], req["required_info_id"], "evidences")

            trace = {
                "trace_id": _trace_id(variant["variant_id"], card["trace"]["summary"]),
                "family_id": family["family_id"],
                "variant_id": variant["variant_id"],
                "source_case_id": case["case_id"],
                "summary": card["trace"]["summary"],
                "recommended_action_ids": [action_label_to_id[x] for x in card["trace"]["recommended_action_labels"]],
                "actual_action_ids": [action_label_to_id[x] for x in card["trace"]["actual_action_labels"] if x in action_label_to_id],
                "evidence_ids": [evidence["evidence_id"]],
            }
            objects["DiagnosticTrace"].append(trace)
            add_rel(variant["variant_id"], trace["trace_id"], "has_trace")
            add_rel(case["case_id"], trace["trace_id"], "supports")
            for action_id in trace["recommended_action_ids"]:
                add_rel(trace["trace_id"], action_id, "used_action")

    for family in objects["FaultFamily"]:
        family_id = family["family_id"]
        fam_actions = [a for a in objects["DiagnosticAction"] if a["family_id"] == family_id]
        fam_outcomes = [o for o in objects["ActionOutcome"] if o["family_id"] == family_id]
        fam_traces = [t for t in objects["DiagnosticTrace"] if t["family_id"] == family_id]
        policy = {
            "policy_id": _policy_id(family_id),
            "family_id": family_id,
            "source_trace_ids": [t["trace_id"] for t in fam_traces],
            "source_outcome_ids": [o["outcome_id"] for o in fam_outcomes],
            "ordered_action_ids": [a["action_id"] for a in sorted(fam_actions, key=lambda x: (int(x.get("step_order") or 999), x.get("label") or ""))],
            "ineffective_action_ids": [],
            "high_cost_action_ids": [a["action_id"] for a in fam_actions if a.get("high_cost") or a.get("destructive")],
            "deterministic_recompute": True,
        }
        objects["DecisionPolicy"].append(policy)
        add_rel(policy["policy_id"], family_id, "for_family")

    store = JsonKGV2Store(TARGET_ROOT)
    issues = validate_graph(objects, relations)
    if issues:
        raise RuntimeError("schema validation failed: " + "; ".join(issues[:20]))
    replace = store.replace_graph(objects, relations, validate=True)
    if replace.get("status") != "replaced":
        raise RuntimeError(f"replace_graph failed: {replace}")
    materialized = KGV2Materializer(store).materialize(store.materialized_root)
    summary = {
        "schema_version": "debug_agent_system.main_program_manual_build_summary.v1",
        "scope": "SOP/1.主程序/phase1-batch1",
        "build_root": str(BUILD_ROOT),
        "target_root": str(TARGET_ROOT),
        "section_ids": sorted({x for card in cards for x in card.get("source_sections", [])}),
        "counts": {k: len(v) for k, v in objects.items()},
        "relation_count": len(relations),
        "replace": replace,
        "materialized": materialized,
        "families": [x["label"] for x in objects["FaultFamily"]],
        "variants": [x["label"] for x in objects["FaultVariant"]],
    }
    _write_json(SUMMARY_OUT, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["appendix-taxonomy", "section-map", "inventory", "family-map", "manual-cards", "build", "all"],
        default="all",
    )
    args = parser.parse_args(argv)
    out: dict[str, Any] = {"step": args.step}
    if args.step in {"appendix-taxonomy", "all"}:
        out["appendix_taxonomy"] = str(write_appendix_taxonomy())
    if args.step in {"section-map", "all"}:
        out["section_id_map"] = str(write_section_id_map())
    if args.step in {"inventory", "all"}:
        out["section_inventory"] = str(write_section_inventory())
    if args.step in {"family-map", "all"}:
        out["family_map"] = str(write_family_map())
    if args.step in {"manual-cards", "all"}:
        out["manual_cards"] = [str(x) for x in write_manual_cards()]
    if args.step in {"build", "all"}:
        out["build_summary"] = build_graph()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
