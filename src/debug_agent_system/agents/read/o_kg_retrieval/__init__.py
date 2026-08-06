from __future__ import annotations

import re

from debug_agent_system.core.contracts import Candidate
from debug_agent_system.knowledge.store import KGStore

_OX_CODE = re.compile(r"\b[oO]x([0-9a-fA-F]{6,8})\b")
_HEX_CODE = re.compile(r"\b0x[0-9a-fA-F]{6,8}\b")
_BLUE_SIGNALS = (
    "蓝屏",
    "stop code",
    "bugcheck",
    "自动修复",
    "安全模式",
    "minidump",
    "memory.dmp",
    "dmp",
)
_BLUE_ERROR_IDS = {
    "err:industrial-pc-blue-screen": 4.0,
    "err:industrial-pc-blue-screen-crash": 3.5,
    "err:industrial-pc-unexpected-reboot": 2.0,
    "err:system-bsod-restart": 2.0,
}
_STRONG_BLUE_SIGNALS = ("stop code", "bugcheck", "minidump", "memory.dmp", "dmp")
_OS_BOOT_SIGNALS = ("无启动设备", "提示无启动", "进不了windows", "进不了 windows", "无法进入windows", "无法进入 windows")
_SOFTWARE_CRASH_SIGNALS = (
    "软件会自动退出",
    "软件自动退出",
    "自动退出软件",
    "自动退出",
    "程序自动退出",
    "应用自动退出",
    "软件闪退",
    "程序闪退",
    "主程序闪退",
    "闪退",
    "异常退出",
    "突然关闭",
    "主程序退出",
    "主程序关闭",
    "程序崩溃",
    "应用异常",
    "crash",
)
_SOFTWARE_CRASH_ERROR_IDS = {
    "err:app-crash-version-0-23-9": 9.0,
    "err:software-freeze-crash-restart-error-reset-needed": 5.5,
    "err:main-app-frequent-freeze-crash": 4.5,
    "err:app-unexpected-crash-no-error": 4.5,
    "err:software-crash-restart-failure": 4.0,
    "err:aoi-software-frequent-crash": 4.0,
    "err:3d-software-crash": 3.5,
}
_SOFTWARE_CRASH_PENALTY_ERROR_IDS = {
    "err:industrial-pc-blue-screen",
    "err:industrial-pc-blue-screen-crash",
    "err:system-bsod-restart",
}
_SOFTWARE_CRASH_EXCLUSIONS = ("相机", "拍照", "拍摄", "cad", "导入cad", "启动页面", "开机页面")


class KGRetrievalAgent:
    """O-KG: deterministic KG candidate recall/ranking."""

    def __init__(self, store: KGStore) -> None:
        self.store = store

    def retrieve(self, query: str, limit: int = 5) -> list[Candidate]:
        normalized = normalize_error_codes(query)
        candidates = self.store.search_errors(normalized, limit=max(limit, 80))
        candidates = rerank_blue_screen_candidates(normalized, candidates)
        candidates = rerank_software_crash_candidates(normalized, candidates)
        return candidates[:limit]


def normalize_error_codes(text: str) -> str:
    """Normalize common OCR/user typo `Oxc...` to Windows style `0xc...`."""

    return _OX_CODE.sub(lambda m: f"0x{m.group(1).lower()}", text)


def has_blue_screen_signal(text: str) -> bool:
    lowered = normalize_error_codes(text).lower()
    return bool(_HEX_CODE.search(lowered)) or any(signal in lowered for signal in _BLUE_SIGNALS)


def has_software_crash_signal(text: str) -> bool:
    lowered = normalize_error_codes(text).lower()
    if not any(signal in lowered for signal in _SOFTWARE_CRASH_SIGNALS):
        return False
    if has_blue_screen_signal(lowered):
        return False
    return not any(signal in lowered for signal in _SOFTWARE_CRASH_EXCLUSIONS)


def rerank_blue_screen_candidates(query: str, candidates: list[Candidate]) -> list[Candidate]:
    if not candidates or not has_blue_screen_signal(query):
        return candidates
    lowered = normalize_error_codes(query).lower()
    strong_blue = bool(_HEX_CODE.search(lowered)) or any(signal in lowered for signal in _STRONG_BLUE_SIGNALS)
    if not strong_blue and any(signal in lowered for signal in _OS_BOOT_SIGNALS):
        return candidates
    top_score = max(float(candidates[0].score), 0.01)
    adjusted: list[Candidate] = []
    for candidate in candidates:
        boost = _BLUE_ERROR_IDS.get(candidate.error_id, 0.0)
        score = float(candidate.score) + boost
        route = candidate.route
        evidence = list(candidate.evidence)
        payload = dict(candidate.payload)
        if boost:
            route = "lexical_kg+blue_screen_rerank"
            evidence.append(f"blue_screen_rerank:+{boost:g}")
            payload["_rerank_boost"] = boost
        adjusted.append(Candidate(
            error_id=candidate.error_id,
            label=candidate.label,
            score=round(score, 4),
            route=route,
            evidence=evidence,
            payload=payload,
        ))
    adjusted.sort(key=lambda c: c.score, reverse=True)

    # Canonical blue-screen tree owns the richer SOP topology.  If no-boot won
    # lexically but the blue-screen tree was already a plausible candidate,
    # prefer the richer tree instead of letting generic no-boot swallow it.
    canonical = next((c for c in adjusted if c.error_id == "err:industrial-pc-blue-screen"), None)
    if canonical and adjusted[0].error_id == "err:industrial-pc-no-boot":
        original_canonical = next((c for c in candidates if c.error_id == canonical.error_id), canonical)
        if float(original_canonical.score) >= top_score * 0.55:
            adjusted = [canonical] + [c for c in adjusted if c.error_id != canonical.error_id]
    return adjusted


def rerank_software_crash_candidates(query: str, candidates: list[Candidate]) -> list[Candidate]:
    if not candidates or not has_software_crash_signal(query):
        return candidates
    adjusted: list[Candidate] = []
    for candidate in candidates:
        boost = _software_crash_boost(candidate)
        score = float(candidate.score) + boost
        route = candidate.route
        evidence = list(candidate.evidence)
        payload = dict(candidate.payload)
        if boost:
            route = _append_route(route, "software_crash_rerank")
            evidence.append(f"software_crash_rerank:{boost:+g}")
            payload["_rerank_boost"] = float(payload.get("_rerank_boost") or 0.0) + boost
        adjusted.append(Candidate(
            error_id=candidate.error_id,
            label=candidate.label,
            score=round(score, 4),
            route=route,
            evidence=evidence,
            payload=payload,
        ))
    adjusted.sort(key=lambda c: c.score, reverse=True)
    return adjusted


def _software_crash_boost(candidate: Candidate) -> float:
    if candidate.error_id in _SOFTWARE_CRASH_PENALTY_ERROR_IDS:
        return -4.0
    boost = _SOFTWARE_CRASH_ERROR_IDS.get(candidate.error_id, 0.0)
    if boost:
        return boost
    payload = candidate.payload or {}
    text = " ".join(
        str(x)
        for x in (
            candidate.error_id,
            candidate.label,
            payload.get("label", ""),
            payload.get("symptom", ""),
            payload.get("scenario", ""),
            payload.get("source_title", ""),
            " ".join(str(k) for k in payload.get("keywords") or []),
        )
    ).lower()
    if "闪退" in text or "自动退出" in text or "异常退出" in text or "crash" in text:
        return 2.5
    return 0.0


def _append_route(route: str, suffix: str) -> str:
    if not route:
        return suffix
    if suffix in route.split("+"):
        return route
    return f"{route}+{suffix}"
