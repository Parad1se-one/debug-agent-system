"""Build a review-first candidate library from the Xing Lark export.

The library is deliberately not a gold-set generator.  It ranks evidence-rich
chat windows and preserves the signals needed for a human to decide the real
trace boundary, split parallel faults, and author ground truth afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


SOURCE = Path("data/results/xing_relation_context_final_20260717/messages.jsonl")
DEFAULT_OUTPUT = Path("data/annotations/goldcases/candidates/xing-lark-v1")
DEFAULT_SINCE = "2025-12-01 00:00"

ISSUE_RE = re.compile(
    r"报错|失败|异常|蓝屏|黑屏|花屏|卡死|崩溃|闪退|断连|掉线|不拍|"
    r"无法|不能|未响应|超时|丢失|丢包|漏报|误报|不进板|不出板|停留"
)
DAILY_RE = re.compile(r"今日.*(?:汇报|现场)|现场(?:问题|情况)|工作汇报|每日总结|问题总结")
DIAGNOSIS_RE = re.compile(r"从日志|日志来看|定位到|判断为|原因[:：]|根因|错误类型|调用链|排查方向")
ACTION_RE = re.compile(
    r"重启|重装|升级|回退|更换|替换|拔插|清除|修改|调整|修复|检查|排查|"
    r"收集|导出|抓取|验证|观察|提交jira|提交JIRA"
)
RESOLUTION_RE = re.compile(
    r"恢复生产|恢复正常|已恢复|已经恢复|问题解决|已解决|解决了|验证正常|"
    r"正常使用|未再出现|未出现异常|没有复发|暂无复发|Resolved\s*--\s*Done"
)
WEAK_ONLY_RE = re.compile(r"需求|培训|交付|进度|计划")
JIRA_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
ISSUE_TAG_PATTERNS = {
    "storage": re.compile(r"硬盘|磁盘|DiskGenius|SMART|坏块|掉盘|盘符|C盘|D盘|克隆"),
    "system_crash": re.compile(r"蓝屏|黑屏|自动重启|死机|系统崩溃|进不去系统"),
    "performance": re.compile(r"卡顿|卡死|无响应|后台进程|CPU|内存占用"),
    "camera_capture": re.compile(r"相机|拍摄|少收图|丢帧|CXP|采集卡|曝光"),
    "motion_control": re.compile(r"运控|运动控制|进板|出板|卡板|轨道|皮带|挡板|顶升"),
    "buddy": re.compile(r"buddy|Buddy|BUDDY|HTTP\s*500|HTTP\s*502"),
    "review_station": re.compile(r"复判站|复盘站|加载板卡|复判数据"),
    "algorithm_quality": re.compile(r"误报|漏报|漏检|检出|模型|算法|置信度"),
    "template_program": re.compile(r"模板|程序|proj|编程|标记点|mark点|Mark点"),
    "network_io": re.compile(r"网卡|网络|掉线|断连|IP|IO信号|要板信号"),
}


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def _compact(text: str, limit: int = 180) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_message_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_message_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_message_ids(item)
    elif isinstance(value, str):
        yield from re.findall(r"\bom_[a-zA-Z0-9]+\b", value)


def load_known_gold_message_ids(root: Path) -> set[str]:
    paths = sorted((root / "data/annotations/goldcases/gold-v1").glob("goldcase-*.json"))
    paths += sorted((root / "data/annotations/goldcases/review-v3/ground_truth").glob("goldcase-*.json"))
    paths += sorted((root / "data/annotations/goldcases/review-v3/inputs").glob("goldcase-*.json"))
    paths += sorted((root / "data/annotations/goldcases/gold-v2/ground_truth").glob("goldcase-*.json"))
    paths += sorted((root / "data/annotations/goldcases/gold-v2/inputs").glob("goldcase-*.json"))
    known: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        known.update(_walk_message_ids(payload))
    return known


def _attachment_rows(message: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in message.get("attachments") or []:
        name = str(item.get("name") or "")
        path = str(item.get("path") or "")
        key = str(item.get("file_key") or "")
        identity = (key, name)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({
            "message_id": str(message.get("message_id") or ""),
            "name": name,
            "kind": str(item.get("kind") or ""),
            "evidence_role": str(item.get("evidence_role") or ""),
            "file_key": key,
            "local_path": path or None,
            "payload_available": bool(path and Path(path).is_file()),
        })
    return rows


def _signals(message: dict[str, Any], fae_names: set[str]) -> set[str]:
    text = str(message.get("text") or "")
    sender = str((message.get("sender") or {}).get("name") or "")
    attachments = _attachment_rows(message)
    links = message.get("links") or []
    signals: set[str] = set()
    if ISSUE_RE.search(text):
        signals.add("issue")
    if DAILY_RE.search(text):
        signals.add("daily_report")
    if DIAGNOSIS_RE.search(text):
        signals.add("diagnosis")
    if ACTION_RE.search(text):
        signals.add("action")
    if RESOLUTION_RE.search(text):
        signals.add("resolution")
    if sender in fae_names:
        signals.add("fae_sender")
    if any(link.get("type") == "jira" or "jira." in str(link.get("url") or "") for link in links) or JIRA_KEY_RE.search(text):
        signals.add("jira")
    if attachments:
        signals.add("attachment")
    if any(
        item["evidence_role"] == "log_package"
        or "诊断数据" in item["name"]
        or item["name"].lower().endswith((".zip", ".dmp", ".evtx"))
        for item in attachments
    ):
        signals.add("diagnostic_artifact")
    return signals


def _issue_tags(texts: Iterable[str]) -> list[str]:
    body = "\n".join(texts)
    return sorted(tag for tag, pattern in ISSUE_TAG_PATTERNS.items() if pattern.search(body))


def _is_anchor(signals: set[str], text: str) -> bool:
    if signals & {"diagnostic_artifact", "jira", "resolution", "diagnosis", "daily_report"}:
        return "issue" in signals or len(signals & {"diagnostic_artifact", "jira", "resolution", "diagnosis"}) >= 1
    return "issue" in signals and bool(signals & {"attachment", "action", "fae_sender"}) and len(text.strip()) >= 12


@dataclass
class Anchor:
    message: dict[str, Any]
    time: datetime
    signals: set[str]


@dataclass
class Cluster:
    chat_id: str
    anchors: list[Anchor] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return self.anchors[0].time

    @property
    def end(self) -> datetime:
        return self.anchors[-1].time


def _cluster_anchors(anchors: list[Anchor]) -> list[Cluster]:
    # A candidate is a relation-aware conversation unit, not an arbitrary
    # rolling time bucket.  Wider same-chat time windows are attached later as
    # search hints so longitudinal evidence can be recovered without merging
    # every parallel fault in a busy project group.
    by_unit: dict[tuple[str, str], list[Anchor]] = defaultdict(list)
    for anchor in anchors:
        chat_id = str(anchor.message.get("chat_id") or "")
        session_id = str(anchor.message.get("relation_aware_session_id") or "")
        if not session_id:
            session_id = f"fallback:{anchor.time:%Y-%m-%d}"
        by_unit[(chat_id, session_id)].append(anchor)
    clusters: list[Cluster] = []
    for (chat_id, _session_id), items in by_unit.items():
        items.sort(key=lambda item: (item.time, str(item.message.get("message_id") or "")))
        clusters.append(Cluster(chat_id=chat_id, anchors=items))
    return clusters


def _score(signals: set[str], message_count: int, known_overlap: bool, weak_only: bool) -> int:
    weights = {
        "diagnostic_artifact": 20,
        "jira": 15,
        "resolution": 15,
        "diagnosis": 12,
        "action": 7,
        "daily_report": 4,
        "attachment": 4,
        "fae_sender": 2,
        "issue": 3,
    }
    score = sum(weights.get(item, 0) for item in signals)
    score += min(5, max(0, message_count - 2))
    if known_overlap:
        score -= 55
    if weak_only:
        score -= 18
    return max(0, min(100, score))


def build_library(
    source: Path,
    repo_root: Path,
    fae_csv: Path,
    since: str = DEFAULT_SINCE,
    limit: int = 120,
) -> dict[str, Any]:
    fae_names: set[str] = set()
    for index, line in enumerate(fae_csv.read_text(encoding="utf-8-sig").splitlines()):
        if index == 0 or not line.strip():
            continue
        fae_names.add(line.split(",", 1)[0].strip())
    known_ids = load_known_gold_message_ids(repo_root)
    since_dt = _parse_time(since)

    messages_by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchors: list[Anchor] = []
    source_count = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            message = json.loads(line)
            create_time = str(message.get("create_time") or "")
            if not create_time:
                continue
            when = _parse_time(create_time)
            if when < since_dt:
                continue
            source_count += 1
            chat_id = str(message.get("chat_id") or "")
            messages_by_chat[chat_id].append(message)
            signals = _signals(message, fae_names)
            if _is_anchor(signals, str(message.get("text") or "")):
                anchors.append(Anchor(message=message, time=when, signals=signals))

    clusters = _cluster_anchors(anchors)
    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        # Keep a review window wider than the automatic anchor cluster.  The
        # window is a search hint, never an asserted trace boundary.
        review_start = cluster.start - timedelta(days=2)
        review_end = cluster.end + timedelta(days=7)
        wide_context = [
            message for message in messages_by_chat[cluster.chat_id]
            if review_start <= _parse_time(str(message.get("create_time") or "")) <= review_end
        ]
        core_session_ids = {
            str(item.message.get("relation_aware_session_id") or "") for item in cluster.anchors
        }
        core_session_ids.discard("")
        nearby = [
            message for message in wide_context
            if (
                str(message.get("relation_aware_session_id") or "") in core_session_ids
                or str(message.get("message_id") or "") in {
                    str(item.message.get("message_id") or "") for item in cluster.anchors
                }
            )
        ]
        evidence: list[dict[str, Any]] = []
        all_signals: set[str] = set()
        attachments: list[dict[str, Any]] = []
        jira_urls: set[str] = set()
        overlap_ids: set[str] = set()
        for message in nearby:
            signals = _signals(message, fae_names)
            if not signals & {"issue", "diagnosis", "action", "resolution", "jira", "diagnostic_artifact", "daily_report"}:
                continue
            all_signals.update(signals)
            message_id = str(message.get("message_id") or "")
            if message_id in known_ids:
                overlap_ids.add(message_id)
            attachments.extend(_attachment_rows(message))
            for link in message.get("links") or []:
                url = str(link.get("url") or "")
                if link.get("type") == "jira" or "jira." in url:
                    jira_urls.add(url)
            evidence.append({
                "create_time": str(message.get("create_time") or ""),
                "message_id": message_id,
                "sender": str((message.get("sender") or {}).get("name") or ""),
                "signals": sorted(signals),
                "text": _compact(str(message.get("text") or "")),
                "root_id": str(message.get("root_id") or ""),
                "parent_id": str(message.get("parent_id") or ""),
                "relation_aware_session_id": str(message.get("relation_aware_session_id") or ""),
            })
        if not evidence:
            continue
        evidence.sort(key=lambda item: (item["create_time"], item["message_id"]))
        weak_only = not (all_signals & {"diagnostic_artifact", "diagnosis", "resolution"}) and all(
            WEAK_ONLY_RE.search(item["text"]) for item in evidence if item["text"]
        )
        score = _score(all_signals, len(evidence), bool(overlap_ids), weak_only)
        chat_name = str((cluster.anchors[0].message.get("raw") or {}).get("chat_name") or "")
        stable_key = "|".join([
            cluster.chat_id,
            *sorted(core_session_ids),
            f"{cluster.start:%Y-%m-%d %H:%M}",
            f"{cluster.end:%Y-%m-%d %H:%M}",
        ])
        first_issue = next((item["text"] for item in evidence if "issue" in item["signals"] and item["text"]), evidence[0]["text"])
        candidates.append({
            "candidate_id": "xlc-" + hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:12],
            "status": "known_gold_overlap" if overlap_ids else "unreviewed",
            "quality_tier": "A" if score >= 70 else "B" if score >= 50 else "C",
            "score": score,
            "title_hint": _compact(first_issue, 100),
            "chat_id": cluster.chat_id,
            "chat_name": chat_name,
            "anchor_time_range": {"start": cluster.start.strftime("%Y-%m-%d %H:%M"), "end": cluster.end.strftime("%Y-%m-%d %H:%M")},
            "recommended_review_window": {"start": review_start.strftime("%Y-%m-%d %H:%M"), "end": review_end.strftime("%Y-%m-%d %H:%M")},
            "signals": sorted(all_signals),
            "issue_tags": _issue_tags(item["text"] for item in evidence),
            "anchor_message_ids": [str(item.message.get("message_id") or "") for item in cluster.anchors],
            "relation_aware_session_ids": sorted({item["relation_aware_session_id"] for item in evidence if item["relation_aware_session_id"]}),
            "nearby_relation_aware_session_ids": sorted({
                str(message.get("relation_aware_session_id") or "")
                for message in wide_context
                if str(message.get("relation_aware_session_id") or "")
            }),
            "jira_urls": sorted(jira_urls),
            "attachments": attachments,
            "payload_available_count": sum(bool(item["payload_available"]) for item in attachments),
            "known_gold_overlap_message_ids": sorted(overlap_ids),
            "evidence_message_count": len(evidence),
            "evidence_preview": evidence[:40],
            "review_notes": [
                "候选窗口只是检索范围，不是最终 trace 边界。",
                "审核时必须拆分同群并行 fault family，并追踪临时恢复与 verified fix 的区别。",
            ],
        })

    candidates.sort(key=lambda item: (-item["score"], item["status"] != "unreviewed", item["anchor_time_range"]["start"], item["candidate_id"]))
    selected = candidates[:limit]
    by_chat_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_chat_candidates[item["chat_id"]].append(item)
    for item in selected:
        center = _parse_time(item["anchor_time_range"]["start"])
        own_tags = set(item["issue_tags"])
        nearby_rows: list[dict[str, Any]] = []
        for other in by_chat_candidates[item["chat_id"]]:
            if other["candidate_id"] == item["candidate_id"]:
                continue
            other_time = _parse_time(other["anchor_time_range"]["start"])
            distance_days = abs((other_time - center).total_seconds()) / 86400
            if distance_days > 60:
                continue
            shared_tags = sorted(own_tags & set(other["issue_tags"]))
            nearby_rows.append({
                "candidate_id": other["candidate_id"],
                "score": other["score"],
                "anchor_time_range": other["anchor_time_range"],
                "shared_issue_tags": shared_tags,
                "distance_days": round(distance_days, 2),
                "title_hint": other["title_hint"],
            })
        nearby_rows.sort(key=lambda row: (not bool(row["shared_issue_tags"]), row["distance_days"], -row["score"], row["candidate_id"]))
        item["related_candidates"] = nearby_rows[:12]
    return {
        "schema_version": "debug_agent_system.xing_lark_candidate_library.v1",
        "library_id": "xing-lark-v1",
        "source_file": str(source),
        "source_sha256": _file_sha256(source),
        "scan_parameters_sha256": _canonical_hash({
            "source": str(source),
            "since": since,
            "limit": limit,
            "candidate_unit": "relation_aware_session",
        }),
        "generated_at": datetime.now().astimezone().isoformat(),
        "policy": {
            "review_first": True,
            "graph_ingestion": False,
            "candidate_is_not_ground_truth": True,
            "since": since,
            "candidate_unit": "relation_aware_session",
            "review_window_days_before": 2,
            "review_window_days_after": 7,
        },
        "statistics": {
            "messages_scanned": source_count,
            "anchors_found": len(anchors),
            "clusters_found": len(candidates),
            "candidates_written": len(selected),
            "tier_a": sum(item["quality_tier"] == "A" for item in selected),
            "tier_b": sum(item["quality_tier"] == "B" for item in selected),
            "known_gold_overlap": sum(item["status"] == "known_gold_overlap" for item in selected),
        },
        "candidates": selected,
    }


def render_readme(payload: dict[str, Any]) -> str:
    stats = payload["statistics"]
    lines = [
        "# Xing Lark Goldcase 候选库 v1",
        "",
        "> 这是人工审核入口，不是 ground truth，也不会直接写入 KG。候选窗口用于找证据，最终 trace 必须人工重划边界。",
        "",
        "人工选择与审核状态见 [`selections/README.md`](selections/README.md)。",
        "",
        "## 统计",
        "",
        f"- 扫描消息：{stats['messages_scanned']}",
        f"- 原始候选簇：{stats['clusters_found']}",
        f"- 本次保留：{stats['candidates_written']}（A 级 {stats['tier_a']}，B 级 {stats['tier_b']}）",
        f"- 本次保留中与 001–015 已知证据重叠：{stats['known_gold_overlap']}",
        "",
        "## 优先审核清单",
        "",
        "| 排名 | candidate | 分数 | 时间 | 群聊 | 证据特征 | payload | Jira | 状态 | 标题提示 |",
        "|---:|---|---:|---|---|---|---:|---:|---|---|",
    ]
    for rank, item in enumerate(payload["candidates"][:50], 1):
        lines.append(
            "| {rank} | `{cid}` | {score} | {start} ~ {end} | {chat} | {signals} | {payloads} | {jira} | {status} | {title} |".format(
                rank=rank,
                cid=item["candidate_id"],
                score=item["score"],
                start=item["anchor_time_range"]["start"],
                end=item["anchor_time_range"]["end"],
                chat=str(item["chat_name"]).replace("|", "/"),
                signals=", ".join(item["signals"]),
                payloads=item["payload_available_count"],
                jira=len(item["jira_urls"]),
                status=item["status"],
                title=str(item["title_hint"]).replace("|", "/"),
            )
        )
    lines += [
        "",
        "## 评分用途",
        "",
        "分数只衡量可审核性：诊断包、Jira、诊断文字、动作和恢复/复测信号越完整，排名越高。它不代表结论正确，也不代表单个候选只含一个业务 trace。",
        "",
        "## 重建",
        "",
        "```bash",
        "PYTHONPATH=src python -m debug_agent_system.eval.write_side.build_xing_lark_candidate_library",
        "```",
        "",
        "详细证据、附件路径和关系字段见 `candidates.json`。",
    ]
    return "\n".join(lines) + "\n"


def write_library(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (output / "candidates.json").write_text(body, encoding="utf-8")
    (output / "README.md").write_text(render_readme(payload), encoding="utf-8")
    manifest = {
        "schema_version": "debug_agent_system.xing_lark_candidate_library_manifest.v1",
        "library_id": payload["library_id"],
        "candidate_file": "candidates.json",
        "candidate_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "candidate_count": len(payload["candidates"]),
        "graph_ingestion": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build-xing-lark-candidate-library")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--since", default=DEFAULT_SINCE)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    payload = build_library(
        source=args.source,
        repo_root=repo_root,
        fae_csv=repo_root / "data/annotations/fae_engineers_2026-07-21.csv",
        since=args.since,
        limit=args.limit,
    )
    write_library(payload, args.out)
    print(json.dumps(payload["statistics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
