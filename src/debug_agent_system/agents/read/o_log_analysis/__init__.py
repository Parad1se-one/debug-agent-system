from __future__ import annotations

import re


class LogAnalysisAgent:
    """O-LOG: extract lightweight hints from query/log text; never owns diagnosis."""

    def analyze(self, query: str, log_summary: dict | None = None) -> dict:
        log_summary = dict(log_summary or {})
        text = query + " " + " ".join(str(v) for v in log_summary.values())
        branch_hints = []
        suggested_check_ids = []
        for key, hint in (("相机", "branch:camera"), ("光源", "branch:light_source"), ("运控", "branch:motion_control"), ("配置", "branch:config_load")):
            if key in text:
                branch_hints.append(hint)
        versions = sorted(set(re.findall(r"\b(?:v|V)?\d+\.\d+(?:\.\d+)?\b", text)))
        return {
            "log_available": bool(log_summary),
            "branch_hints": branch_hints,
            "suggested_check_ids": suggested_check_ids,
            "version_hints": versions,
            "device_hints": [x for x in ("相机", "光源", "运控", "工控机") if x in text],
            "parse_errors": [],
        }
