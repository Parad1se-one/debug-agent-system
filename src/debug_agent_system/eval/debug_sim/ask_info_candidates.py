"""Build ask-info eval scenarios from W6 ask-info review queue."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_RESOURCE_PLACEHOLDER_RE = re.compile(r"\[(Image|Media|File|Video):[^\]]+\]", re.IGNORECASE)
_DANGLING_RESOURCE_PLACEHOLDER_RE = re.compile(r"\[(Image|Media|File|Video):\s*[^\s，,。；;]*$", re.IGNORECASE)
_RESOURCE_ID_RE = re.compile(r"\b(?:img|image|media|file|video)_v?[A-Za-z0-9][A-Za-z0-9_.-]{8,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def load_queue(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def build_scenarios(items: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        candidate = item.get("required_info_candidate") if isinstance(item.get("required_info_candidate"), dict) else {}
        if not candidate:
            continue
        gate = item.get("quality_gate") if isinstance(item.get("quality_gate"), dict) else {}
        if gate and not gate.get("passed"):
            continue
        slot = str(candidate.get("slot") or "")
        label = str(candidate.get("label") or slot or "补充信息")
        case_id = str(candidate.get("candidate_id") or item.get("review_id") or "")
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        source_query = _query_from_item(item, candidate)
        supplement = _supplement_reply(item, candidate, source_query, label)
        if not _is_gate_grade_supplement(supplement):
            continue
        scenarios.append({
            "case_id": f"ask_info_{len(scenarios) + 1:03d}_{case_id.replace(':', '_')}",
            "query": _ask_query_from_candidate(candidate, source_query),
            "source": "w6_ask_info_review_queue",
            "difficulty": "missing_info",
            "query_type": "debug",
            "expected_status": "ask_info",
            "target_error_id": str(candidate.get("target_error_id") or ""),
            "acceptable_error_ids": list(candidate.get("acceptable_error_ids") or []),
            "required_info": _required_info_terms(slot, label, str(candidate.get("question") or "")),
            "user_turns": [{
                "when_check_contains": "",
                "reply": supplement,
                "expected_next": "step",
            }],
            "evidence_key_facts": _facts(item, candidate),
            "max_turns": 3,
            "metadata": {
                "source_episode_id": candidate.get("source_episode_id") or "",
                "source_thread_id": candidate.get("source_thread_id") or "",
                "source_query": source_query,
                "slot": slot,
                "condition": candidate.get("condition") or "",
            },
        })
        if len(scenarios) >= limit:
            break
    return scenarios


def write_scenarios(path: str | Path, scenarios: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _query_from_item(item: dict[str, Any], candidate: dict[str, Any]) -> str:
    episode = item.get("episode") if isinstance(item.get("episode"), dict) else {}
    texts: list[str] = []
    for msg in episode.get("fault_description_messages") or []:
        if isinstance(msg, dict):
            text = _clean_eval_text(msg.get("content_summary") or msg.get("text") or "")
            if text:
                texts.append(text)
    if texts:
        return "现场反馈：" + " ".join(texts[:2])
    request = candidate.get("source_request") if isinstance(candidate.get("source_request"), dict) else {}
    if request.get("context_before"):
        before = request["context_before"][0] if isinstance(request["context_before"][0], dict) else {}
        text = _clean_eval_text(before.get("content_summary") or before.get("text") or "")
        if text:
            return "现场反馈：" + text
    return "现场反馈：设备出现故障，需要诊断。"


def _supplement_reply(item: dict[str, Any], candidate: dict[str, Any], source_query: str, label: str) -> str:
    """Generate a concrete user reply after the first ask-info turn.

    `label=已提供` is too weak for stateful eval: the read side cannot consume
    an opaque file marker, so it may correctly ask a second branch question.
    This reply keeps the original slot label but adds deterministic key facts
    that represent what a parsed log/proj/Jira/attachment would expose.
    """

    slot = str(candidate.get("slot") or "")
    condition = str(candidate.get("condition") or "")
    text = f"{source_query} {candidate.get('question') or ''} {condition} " + " ".join(_facts(item, candidate))
    lowered = text.lower()
    facts: list[str] = []
    if slot == "log_package" and condition == "dmp":
        facts.append("已提供 MEMORY.DMP、系统事件日志和 BugCheck 固定错误代码 0x00000139")
    elif slot == "log_package":
        facts.append("已提供 DLOG 诊断数据包和系统日志")
    elif slot == "error_message":
        if condition == "dmp" or any(k in text for k in ("蓝屏", "BugCheck", "bugcheck", "dmp", "DMP", "dump", "Dump", "转存储", "转储")):
            facts.append("完整报错为 BugCheck 0x00000139，重启前有蓝屏并已保留事件日志")
        elif any(k in text for k in ("拍照", "相机", "camera", "trigger", "timeout")) or "timeout" in lowered:
            facts.append("完整报错为 camera trigger timeout，拍照超时且没有正常采图")
        else:
            facts.append("已提供完整报错文本和报错截图")
    elif slot == "error_phase":
        if condition == "dmp" or any(k in text for k in ("蓝屏", "BugCheck", "bugcheck", "dmp", "DMP", "dump", "Dump", "转存储", "转储")):
            facts.append("故障发生在 Windows 运行中，重启前出现蓝屏 BugCheck 0x00000139")
        elif any(k in text for k in ("初始化", "启动", "startup", "init")):
            facts.append("故障发生在启动/初始化阶段")
        elif any(k in text for k in ("拍照", "相机", "camera")):
            facts.append("故障发生在拍照触发阶段，日志提示 trigger/camera timeout")
        else:
            facts.append("故障发生阶段已确认")
    elif slot == "ip_config":
        facts.append("已提供相机/控制器 IP、网段和 ping 连通性结果")
    elif slot == "software_version":
        facts.append("已提供主程序版本 0.27.44 和算法包版本")
    elif slot == "program_file":
        facts.append("已提供配方/程序文件，并确认其中相机 IP 配置可读取")
    elif slot == "sample_image":
        facts.append("已提供原图、缺陷截图和报错截图")
    elif slot == "repro_steps":
        facts.append("已提供复现步骤：拍照触发后超时，现象可复现")
    elif slot == "environment":
        if any(k in text for k in ("硬盘", "磁盘", "分区", "disk", "Disk")) and not any(k in text for k in ("黑屏", "重启", "蓝屏", "Kernel-Power", "kernel-power")):
            facts.append("已提供磁盘管理器截图，第三块物理硬盘的第三个分区异常，系统盘剩余空间不足")
        elif any(k in text for k in ("重启", "黑屏", "电源", "市电", "工业环境", "Kernel-Power", "kernel-power")):
            facts.append("现场为黑屏后直接复位，事件日志只有 Kernel-Power 41/6008，怀疑市电或工业环境干扰")
        else:
            facts.append("已提供系统环境、电源、磁盘和内存状态")
    elif slot == "device_model":
        facts.append("已提供相机、控制器和工控机型号")
    elif slot == "site":
        facts.append("已提供现场、客户、线体和设备编号")
    elif slot == "owner_context":
        facts.append("已提供责任归属上下文和当前处理人")
    else:
        facts.append("已提供现场排查所需补充信息")

    if slot in {"log_package", "error_message", "error_phase", "repro_steps"} and any(k in text for k in ("拍照", "相机", "camera")) and not any("trigger" in fact.lower() or "拍照" in fact for fact in facts):
        facts.append("日志提示 trigger/camera timeout，属于拍照超时分支")
    explicit_dump = any(k in text for k in ("蓝屏", "BugCheck", "bugcheck", "dmp", "DMP", "dump", "Dump", "转存储", "转储"))
    if (condition == "dmp" or explicit_dump) and slot in {"log_package", "error_message", "error_phase"} and not any("0x" in fact or "BugCheck" in fact for fact in facts):
        facts.append("重启前有蓝屏 BugCheck 固定错误代码 0x00000139")

    return f"补充：{_compact_source_query(source_query)}；{label}=已提供；{'；'.join(_dedupe(facts))}。"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value or "").split())
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _is_gate_grade_supplement(reply: str) -> bool:
    """Keep ask-info eval focused on scenarios with consumable follow-up facts."""

    text = str(reply or "").lower()
    markers = (
        "bugcheck",
        "0x",
        "memory.dmp",
        "kernel-power",
        "41/6008",
        "磁盘管理器截图",
        "系统盘剩余空间不足",
        "完整报错文本",
        "报错截图",
        "trigger/camera timeout",
        "启动/初始化阶段",
        "相机 ip",
        "ip、网段",
        "已提供主程序版本",
        "配方/程序文件",
    )
    return any(marker in text for marker in markers)


def _required_info_terms(slot: str, label: str, question: str) -> list[str]:
    terms: list[str] = []
    if slot:
        terms.append(f"slot:{slot}")
    if question:
        terms.append(f"question:{question}")
    terms.extend(_slot_synonyms(slot))
    if label:
        terms.append(label)
    deduped: list[str] = []
    for item in terms:
        if item and item not in deduped:
            deduped.append(item)
    return deduped or ["补充信息"]


def _slot_synonyms(slot: str) -> list[str]:
    return {
        "log_package": ["诊断数据包/日志", "诊断数据包", "日志"],
        "error_message": ["故障现象/完整报错文本", "完整报错文本", "报错截图"],
        "software_version": ["软件版本"],
        "error_phase": ["故障发生阶段", "发生阶段"],
        "device_model": ["设备型号", "硬件对象"],
        "site": ["现场/客户信息", "设备编号"],
        "ip_config": ["IP/网络配置", "网络配置"],
        "repro_steps": ["复现步骤", "复现频率"],
        "sample_image": ["样本/截图", "样本图", "截图"],
        "program_file": ["程序/配方文件", "配方", "板型"],
        "environment": ["运行环境", "系统环境"],
        "owner_context": ["责任归属上下文", "责任归属"],
    }.get(slot, [])


def _ask_query_from_candidate(candidate: dict[str, Any], source_query: str) -> str:
    slot = str(candidate.get("slot") or "")
    condition = str(candidate.get("condition") or "")
    prefix = _compact_source_query(source_query)
    missing = {
        "log_package": "蓝屏或重启对应的 dmp/系统日志" if condition == "dmp" else "诊断数据包或日志",
        "error_message": "完整报错文本或报错截图",
        "software_version": "软件版本",
        "error_phase": "故障发生阶段",
        "device_model": "设备型号和硬件对象",
        "site": "现场、客户、线体或设备编号信息",
        "ip_config": "IP/网络配置",
        "repro_steps": "复现步骤和复现频率",
        "sample_image": "样本图、原图或截图",
        "program_file": "程序、模板、配方或板型文件",
        "environment": "系统环境、电源、磁盘、内存或运行环境信息",
        "owner_context": "责任归属上下文",
    }.get(slot, "现场排查所需补充信息")
    if prefix:
        return f"{prefix}；当前缺少{missing}。"
    return f"现场反馈：设备出现故障，需要诊断；当前缺少{missing}。"


def _compact_source_query(source_query: str) -> str:
    text = _clean_eval_text(source_query)
    if not text:
        return ""
    return text[:160]


def _facts(item: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    facts = [_clean_eval_text(candidate.get("why_required") or ""), _clean_eval_text(candidate.get("condition") or "")]
    episode = item.get("episode") if isinstance(item.get("episode"), dict) else {}
    for msg in episode.get("diagnostic_chain_messages") or []:
        if isinstance(msg, dict):
            facts.append(_clean_eval_text(msg.get("content_summary") or msg.get("text") or ""))
    return [x for x in facts if x.strip()][:3]


def _clean_eval_text(value: Any) -> str:
    text = str(value or "")
    text = _RESOURCE_PLACEHOLDER_RE.sub(lambda m: {"image": "截图", "media": "媒体附件", "file": "附件", "video": "视频附件"}.get(m.group(1).lower(), "附件"), text)
    text = _DANGLING_RESOURCE_PLACEHOLDER_RE.sub(lambda m: {"image": "截图", "media": "媒体附件", "file": "附件", "video": "视频附件"}.get(m.group(1).lower(), "附件"), text)
    text = _RESOURCE_ID_RE.sub("附件", text)
    text = _URL_RE.sub("链接", text)
    return " ".join(text.split()).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="data/kg/review_queue/ask_info_candidates.json")
    parser.add_argument("--out", default="data/eval/scenarios/ask_info_candidates_v1.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--allow-empty", action="store_true", help="write an empty scenario file when no scenarios are generated")
    args = parser.parse_args(argv)
    scenarios = build_scenarios(load_queue(args.queue), limit=args.limit)
    if not scenarios and not args.allow_empty:
        print(json.dumps({"written": 0, "out": args.out, "error": "no ask-info scenarios generated; output not modified"}, ensure_ascii=False, indent=2))
        return 1
    write_scenarios(args.out, scenarios)
    print(json.dumps({"written": len(scenarios), "out": args.out}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
