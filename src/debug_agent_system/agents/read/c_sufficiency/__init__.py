from __future__ import annotations

from dataclasses import dataclass

from debug_agent_system.core.contracts import Candidate, LockedSubgraph

GENERIC_QUERY_MARKERS = ("缺少完整现象", "缺少完整", "信息不完整", "资料不完整", "需要诊断", "需要排查")
SPECIFIC_SIGNAL_MARKERS = (
    "报错", "错误码", "错误代码", "初始化", "相机", "光源", "工控", "蓝屏", "闪退", "卡死", "漏检", "误报",
    "IP", "ip", "版本", "日志", "DLOG", "dmp", "无法", "失败", "连接", "导出", "SPC", "Mark",
)


@dataclass(slots=True)
class SufficiencyDecision:
    sufficient: bool
    required_info: list[str]
    reason: str
    confidence: float


class SufficiencyGate:
    """C: deterministic minimum-data gate; no hard answer on absent KG match."""

    def __init__(self, graph_match_min_score: float = 4.0, max_required_items: int = 3) -> None:
        self.graph_match_min_score = graph_match_min_score
        self.max_required_items = max_required_items

    def decide(self, query: str, candidates: list[Candidate], subgraph: LockedSubgraph | None = None) -> SufficiencyDecision:
        if not query.strip():
            return SufficiencyDecision(False, ["故障现象/报错文本"], "empty_query", 0.0)
        explicit_required = _explicit_missing_required_info(query)
        if explicit_required:
            required = _subgraph_required_info_for_explicit_missing(explicit_required, subgraph)
            return SufficiencyDecision(False, required[: self.max_required_items], "explicit_missing_required_info", 0.0)
        if _under_specified(query):
            return SufficiencyDecision(False, ["故障现象/完整报错文本", "软件版本", "诊断数据包/日志"], "missing_fault_context", 0.0)
        if not candidates:
            return SufficiencyDecision(False, ["报错截图或完整报错文本", "软件版本", "诊断数据包/日志"], "no_graph_match", 0.0)
        top = candidates[0]
        if top.score < self.graph_match_min_score:
            required = (subgraph.required_info if subgraph else [])[: self.max_required_items]
            if not required:
                required = ["报错截图或完整报错文本", "软件版本", "诊断数据包/日志"]
            return SufficiencyDecision(False, required, "low_graph_score", min(top.score / self.graph_match_min_score, 0.5))
        return SufficiencyDecision(True, [], "sufficient", min(0.95, 0.4 + top.score / 20.0))



def _subgraph_required_info_for_explicit_missing(explicit_required: list[str], subgraph: LockedSubgraph | None) -> list[str]:
    if subgraph is None or not subgraph.required_info:
        return explicit_required
    out: list[str] = []
    for label in explicit_required:
        keywords = _required_label_keywords(label)
        matched = [item for item in subgraph.required_info if any(k and k in item for k in keywords)]
        matched.sort(key=lambda item: _required_match_score(label, item), reverse=True)
        for item in matched:
            if item not in out:
                out.append(item)
        if not matched and label not in out:
            out.append(label)
    return out or explicit_required


def _required_match_score(label: str, item: str) -> int:
    label_text = str(label or "")
    item_text = str(item or "")
    score = 0
    if "错误" in label_text or "报错" in label_text:
        if any(k in item_text for k in ("错误代码", "错误码")):
            score += 12
        if any(k in item_text for k in ("Stop Code", "stop code")):
            score += 8
        if any(k in item_text for k in ("报错", "故障模块", "错误文件")):
            score += 2
        if "蓝屏" in item_text:
            score += 1
    if "日志" in label_text or "诊断数据" in label_text:
        if any(k in item_text for k in ("系统日志", "事件日志", "DLOG", "诊断数据", "Minidump", "MEMORY.DMP", "dmp", "DMP")):
            score += 8
    if "版本" in label_text:
        if any(k in item_text for k in ("主程序版本", "软件版本", "算法包版本", "驱动版本", "版本号")):
            score += 8
    return score


def _required_label_keywords(label: str) -> tuple[str, ...]:
    text = str(label or "")
    if "版本" in text:
        return ("版本", "主程序", "算法包", "驱动")
    if "日志" in text or "诊断数据" in text:
        return ("日志", "DLOG", "诊断数据", "dmp", "DMP", "系统日志")
    if "报错" in text or "错误" in text:
        return ("报错", "错误", "错误码", "错误代码", "代码", "蓝屏", "DMP", "dmp")
    if "截图" in text or "样本" in text:
        return ("截图", "图片", "样本", "照片")
    if "IP" in text or "网络" in text:
        return ("IP", "ip", "网络", "网段")
    if "现场" in text or "客户" in text:
        return ("现场", "客户", "设备编号", "线体")
    if "设备型号" in text:
        return ("型号", "设备", "相机", "工控机", "控制器")
    if "复现" in text:
        return ("复现", "操作", "步骤", "频率")
    if "程序" in text or "配方" in text:
        return ("程序", "配方", "模板", "板型")
    return tuple(x for x in (text,) if x)

def _under_specified(query: str) -> bool:
    text = query.strip()
    if not text:
        return True
    if any(marker in text for marker in GENERIC_QUERY_MARKERS) and sum(1 for marker in SPECIFIC_SIGNAL_MARKERS if marker in text) <= 1:
        return True
    compact = "".join(text.split())
    if len(compact) < 12 and not any(marker in compact for marker in SPECIFIC_SIGNAL_MARKERS):
        return True
    return False


def _explicit_missing_required_info(query: str) -> list[str]:
    text = query.strip()
    if not any(k in text for k in ("缺少", "未提供", "没有提供", "需要补充", "需要提供", "请补充")):
        return []
    # "缺少D盘分区/缺少文件/缺少配置" can be the fault symptom itself, not a
    # statement that the user omitted diagnostic information.  Only explicit
    # request/omission phrases should trigger ask-info.
    if "缺少" in text and not any(k in text for k in ("当前缺少", "仍缺少", "还缺少", "未提供", "没有提供", "需要补充", "需要提供", "请补充", "缺少完整", "缺少故障", "缺少报错", "缺少日志", "缺少诊断数据")):
        return []
    pairs = (
        ("诊断数据包/日志", ("日志", "DLOG", "dlog", "诊断数据", "数据包", "dmp", "DMP", "系统日志")),
        ("故障现象/完整报错文本", ("报错", "错误码", "错误代码", "错误文本", "报错文本", "报错截图", "完整现象", "完整报错")),
        ("软件版本", ("版本", "主程序版本", "算法包版本", "软件版本")),
        ("IP/网络配置", ("IP", "ip", "网段", "网络配置")),
        ("故障发生阶段", ("故障发生阶段", "发生阶段", "启动", "初始化", "扫码", "检测", "复判")),
        ("设备型号", ("设备型号", "硬件对象", "相机型号", "控制器型号", "工控机型号")),
        ("现场/客户信息", ("现场", "客户", "线体", "设备编号")),
        ("复现步骤", ("复现", "操作步骤", "必现", "偶发", "复现频率")),
        ("程序/配方文件", ("程序", "配方", "模板", "板型")),
        ("样本/截图", ("样本", "原图", "图片", "截图", "样本图")),
        ("运行环境", ("系统环境", "电源", "磁盘", "内存", "运行环境")),
        ("责任归属上下文", ("责任归属", "归属上下文", "责任模块")),
    )
    out: list[str] = []
    for label, keywords in pairs:
        if any(k in text for k in keywords) and label not in out:
            out.append(label)
    return out
