"""Jira evidence parser.

Default mode is offline: parse Jira issue keys and enrich them from a local
export. Network fetching remains outside this tool entry; the explicit Jira
sync adapter refreshes the export separately.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

JIRA_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}-\d+\b")
URL_RE = re.compile(r"https?://[^\s\]）)，,。；;]+", re.IGNORECASE)
VERSION_RE = re.compile(r"(?<![\d.])(?:v)?\d{1,2}\.\d+(?:\.\d+){0,2}(?![\d.])", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
FAULT_SPLIT_RE = re.compile(
    r"(设备|现场|软件|程序|客户|AOI|SPI|DIP|3D|Ai脚|测试|运行|初始化|报错|异常|失败|闪退|卡死|漏检|误报|不能|无法)"
)
DEFAULT_OFFLINE_ROOTS = (
    Path("data/imports/jira_offline/raw"),
)


class JiraParserAgent:
    """Tool entry for parsing Jira links/text as evidence metadata."""

    schema_version = "debug_agent_system.tool.jira_parse.v1"

    def __init__(
        self,
        offline_root: str | Path | None = None,
        *,
        max_description_chars: int = 2400,
        max_comment_chars: int = 1200,
        max_comments: int = 3,
    ) -> None:
        self.offline_root = Path(offline_root) if offline_root is not None else None
        self.max_description_chars = max_description_chars
        self.max_comment_chars = max_comment_chars
        self.max_comments = max_comments

    def parse(self, value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict):
            text = " ".join(str(value.get(k) or "") for k in ("url", "label", "text", "content"))
            source = dict(value)
        else:
            text = str(value or "")
            source = {"text": text}
        urls = URL_RE.findall(text)
        keys = sorted(set(JIRA_KEY_RE.findall(text)))
        parsed_urls = [self._url_meta(url) for url in urls]
        for item in parsed_urls:
            keys.extend(JIRA_KEY_RE.findall(item.get("path", "")))
        keys = sorted(set(keys))
        issue_summaries = [self._issue_summary(key, text) for key in keys]
        offline_details = [detail for key in keys if (detail := self._offline_issue_detail(key))]
        title_hints = self._unique([
            *(item["title"] for item in issue_summaries if item.get("title")),
            *(item.get("summary") for item in offline_details if item.get("summary")),
        ])
        version_hints = self._unique(
            [
                *(
                    value
                    for item in issue_summaries
                    for value in item.get("versions", [])
                    if not IP_RE.fullmatch(str(value))
                ),
                *(
                    value
                    for item in offline_details
                    for value in VERSION_RE.findall(" ".join(str(item.get(k) or "") for k in ("summary", "description_preview", "comment_preview_text")))
                    if not IP_RE.fullmatch(str(value))
                ),
            ]
        )
        site_hints = self._unique([
            *(value for item in issue_summaries for value in item.get("site_hints", [])),
            *(value for item in offline_details for value in self._site_hints(str(item.get("summary") or ""))),
        ])
        description_hints = self._unique(item.get("description_preview") for item in offline_details if item.get("description_preview"))
        comment_hints = self._unique(
            comment.get("body_preview")
            for item in offline_details
            for comment in item.get("comments_preview", [])
            if isinstance(comment, dict) and comment.get("body_preview")
        )
        return {
            "schema_version": self.schema_version,
            "type": "JiraParseResult",
            "issue_keys": keys,
            "urls": parsed_urls,
            "issue_summaries": issue_summaries,
            "offline_details": offline_details,
            "title_hints": title_hints,
            "version_hints": version_hints,
            "site_hints": site_hints,
            "description_hints": description_hints,
            "comment_hints": comment_hints,
            "summary_hint": self._summary_hint(text),
            "status": "offline_detail_found" if offline_details else "metadata_only",
            "fetched": False,
            "offline_detail_found": bool(offline_details),
            "source": source,
            "observability": {
                "agent_id": "TOOL-JIRA",
                "boundary": "offline_local_detail" if offline_details else "offline_metadata_only",
            },
        }

    def _url_meta(self, url: str) -> dict[str, str]:
        parsed = urlparse(url)
        return {
            "url": url,
            "host": parsed.netloc,
            "path": parsed.path,
            "type": "jira" if "jira" in parsed.netloc.lower() or "/browse/" in parsed.path else "other",
        }

    def _summary_hint(self, text: str) -> str:
        no_url = URL_RE.sub(" ", str(text or ""))
        compact = " ".join(no_url.split())
        return compact[:240]

    def _issue_summary(self, issue_key: str, text: str) -> dict[str, Any]:
        title = self._title_for_issue(issue_key, text)
        versions = self._unique(value for value in VERSION_RE.findall(title) if not IP_RE.fullmatch(value))
        return {
            "issue_key": issue_key,
            "title": title,
            "versions": versions,
            "site_hints": self._site_hints(title),
            "source": "jira_link_label_or_text",
        }

    def _title_for_issue(self, issue_key: str, text: str) -> str:
        clean = URL_RE.sub(" ", str(text or ""))
        clean = re.sub(r"\s+", " ", clean).strip()
        pattern = re.compile(rf"(?:\[{re.escape(issue_key)}\]|{re.escape(issue_key)})\s*(?P<title>.*?)(?:\s*-\s*Jira\b|$)", re.IGNORECASE)
        match = pattern.search(clean)
        title = match.group("title") if match else clean
        title = title.strip(" \t\r\n-—:：[]【】()（）")
        title = re.sub(r"\s*-\s*Jira\b.*$", "", title, flags=re.IGNORECASE).strip()
        title = JIRA_KEY_RE.sub(" ", title)
        title = re.sub(r"\s+", " ", title).strip(" -—:：[]【】()（）")
        return title[:240]

    def _site_hints(self, title: str) -> list[str]:
        stripped = VERSION_RE.sub(" ", str(title or ""))
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped:
            return []
        prefix = FAULT_SPLIT_RE.split(stripped, maxsplit=1)[0].strip(" -—:：,，。")
        if not prefix:
            return []
        # Real Jira titles usually look like "1.3.5 客户02 设备报错...".
        # Keep only short human-readable prefixes; avoid turning long fault text
        # into a Site candidate.
        if 2 <= len(prefix) <= 16 and re.search(r"[\u4e00-\u9fff]", prefix):
            return [prefix]
        return []

    def _unique(self, values: Any) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def _offline_roots(self) -> list[Path]:
        if self.offline_root is not None:
            return [self.offline_root]
        env_root = os.environ.get("DEBUG_AGENT_SYSTEM_JIRA_OFFLINE_ROOT")
        if env_root:
            return [Path(env_root)]
        return list(DEFAULT_OFFLINE_ROOTS)

    def _offline_issue_detail(self, issue_key: str) -> dict[str, Any] | None:
        key = str(issue_key or "").strip()
        if not key:
            return None
        for root in self._offline_roots():
            candidates = [root / "fault_details" / f"{key}.json", root / f"{key}.json"]
            for path in candidates:
                if not path.exists() or not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    return self._offline_detail_payload(data, path)
        return None

    def _offline_detail_payload(self, data: dict[str, Any], path: Path) -> dict[str, Any]:
        comments = data.get("comments") if isinstance(data.get("comments"), list) else []
        issue_links = data.get("issue_links") if isinstance(data.get("issue_links"), list) else []
        attachments = data.get("attachments") if isinstance(data.get("attachments"), list) else []
        remote_links = data.get("remote_links") if isinstance(data.get("remote_links"), list) else []
        changelog = data.get("changelog") if isinstance(data.get("changelog"), list) else []
        worklogs = data.get("worklogs") if isinstance(data.get("worklogs"), list) else []
        comments_preview: list[dict[str, str]] = []
        for comment in comments[: self.max_comments]:
            if not isinstance(comment, dict):
                continue
            comments_preview.append({
                "author": str(comment.get("author") or ""),
                "created": str(comment.get("created") or ""),
                "body_preview": self._clean_preview(comment.get("body"), self.max_comment_chars),
            })
        comment_preview_text = "\n".join(item["body_preview"] for item in comments_preview if item.get("body_preview"))
        return {
            "issue_key": str(data.get("key") or path.stem),
            "summary": str(data.get("summary") or ""),
            "status": str(data.get("status") or ""),
            "resolution": str(data.get("resolution") or ""),
            "assignee": str(data.get("assignee") or ""),
            "reporter": str(data.get("reporter") or ""),
            "created": str(data.get("created") or ""),
            "updated": str(data.get("updated") or ""),
            "issue_type": str(data.get("issue_type") or ""),
            "priority": str(data.get("priority") or ""),
            "description_preview": self._clean_preview(data.get("description"), self.max_description_chars),
            "comments_preview": comments_preview,
            "comment_preview_text": comment_preview_text[: self.max_comment_chars * self.max_comments],
            "comments_count": len(comments),
            "issue_links": issue_links,
            "linked_issue_keys": self._unique(
                item.get("issue_key") for item in issue_links if isinstance(item, dict)
            ),
            "issue_links_count": len(issue_links),
            "attachments": attachments,
            "attachments_count": len(attachments),
            "remote_links": remote_links,
            "remote_links_count": len(remote_links),
            "changelog_count": len(changelog),
            "worklogs_count": len(worklogs),
            "endpoint_errors": data.get("endpoint_errors") if isinstance(data.get("endpoint_errors"), dict) else {},
            "schema_version": str(data.get("schema_version") or "legacy"),
            "source": "local_jira_offline_export",
            "source_path": str(path),
        }

    def _clean_preview(self, value: Any, limit: int) -> str:
        text = str(value or "")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[: max(0, limit)]


__all__ = ["JiraParserAgent"]
