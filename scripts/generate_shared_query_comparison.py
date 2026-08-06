"""Generate a reproducible full before/after report for a shared conversation.

The baseline can be imported from a public-share JSON endpoint or an offline
MHTML snapshot.  Every user message is paired with the following assistant
message, and every fixed answer is produced by the current KG_v2 runtime.  A
compact JSON baseline is persisted so the embedded QA snapshot can rerun the
comparison without the original page or share service.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import hashlib
from html import escape
from html import unescape
from html.parser import HTMLParser
import ipaddress
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

# Keep this release/evaluation helper directly runnable from a source checkout;
# an editable package install remains supported but is not required.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from debug_agent_system.core.config import load_config
from debug_agent_system.runtime.system import DebugAgentSystem


def _resolve_layout() -> tuple[Path, Path, Path, str]:
    """Support both the canonical repository and the embedded QA snapshot."""

    script_path = Path(__file__).resolve()
    for candidate in script_path.parents:
        canonical_config = candidate / "config/debug_agent_system.yaml"
        if canonical_config.is_file():
            return (
                candidate,
                canonical_config,
                candidate / "docs",
                "../data/",
            )
        embedded_config = (
            candidate
            / "data/runtime/config/debug_agent_system_sag.yaml"
        )
        if embedded_config.is_file():
            return (
                candidate / "data/runtime",
                embedded_config,
                candidate / "docs/debug_agent_system",
                "../../data/runtime/data/",
            )
    raise RuntimeError("cannot locate canonical or embedded debug_agent_system layout")


REPO_ROOT, CONFIG_PATH, _DOCS_ROOT, REPORT_DATA_PREFIX = _resolve_layout()
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data/results/read_side_codex_comparison_20260730"
)
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "KG_v2读侧Codex升级与分享Query对比.md"
DEFAULT_BASELINE = (
    REPO_ROOT
    / "data/eval/scenarios/read_side_shared_query_baseline_v1.json"
)
DEFAULT_PURE_CODEX_BASELINE = (
    REPO_ROOT
    / "data/eval/scenarios/read_side_pure_codex_baseline_v1.json"
)
DEFAULT_KG_RAW_CODEX_ANSWER = (
    REPO_ROOT
    / "data/results/read_side_codex_comparison_20260730"
    / "kg_v2_raw_codex_first_query.json"
)
DEFAULT_MHTML = (
    REPO_ROOT
    / "tmp/Windows 启动异常、系统文件损坏或引导报错时，... · ACME.mhtml"
)
SHARE_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>]+/share/[A-Za-z0-9_-]+"
)


class _PlainTextParser(HTMLParser):
    """Convert the small shared-answer HTML vocabulary to readable text."""

    BLOCK_TAGS = {
        "p", "div", "article", "section", "ul", "ol", "li", "h1", "h2",
        "h3", "h4", "h5", "h6", "table", "tr", "pre", "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in self.BLOCK_TAGS or tag == "br":
            self._newline()
        if tag == "li":
            self.parts.append("- ")
        elif tag == "img":
            values = dict(attrs)
            self.parts.append(f"[图片：{values.get('alt') or '未命名'}]")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def value(self) -> str:
        text = unescape("".join(self.parts))
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    return parser.value()


def _portable_original_html(value: str) -> str:
    """Keep full answer structure while replacing unavailable remote images."""

    def replace_image(match: re.Match[str]) -> str:
        tag = match.group(0)
        alt_match = re.search(r'\balt="([^"]*)"', tag)
        alt = unescape(alt_match.group(1)) if alt_match else "未命名"
        return f"<em>[原回答图片：{alt}]</em>"

    result = re.sub(r"<img\b[^>]*>", replace_image, value, flags=re.I)
    result = re.sub(r"\s(?:class|loading)=\"[^\"]*\"", "", result)
    return result.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_mhtml(mhtml_path: Path, baseline_path: Path) -> dict[str, Any]:
    """Import all paired user/assistant messages from the shared-page MHTML."""

    raw_mhtml = mhtml_path.read_bytes()
    message = BytesParser(policy=policy.default).parsebytes(
        raw_mhtml
    )
    html_payload = next(
        part.get_payload(decode=True).decode("utf-8")
        for part in message.walk()
        if part.get_content_type() == "text/html"
    )
    articles = re.findall(
        r'<article class="message ([^"]*)">(.*?)</article>',
        html_payload,
        flags=re.S,
    )
    messages: list[dict[str, str]] = []
    for css_class, body in articles:
        content_match = re.search(
            r'<div class="content">(.*)</div>\s*$',
            body,
            flags=re.S,
        )
        if not content_match:
            continue
        role = "user" if "user" in css_class.split() else "assistant"
        content_html = content_match.group(1).strip()
        messages.append({
            "role": role,
            "html": content_html,
            "text": _html_to_text(content_html),
        })

    records: list[dict[str, Any]] = []
    for index, current in enumerate(messages):
        if current["role"] != "user":
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        if not following or following["role"] != "assistant":
            raise ValueError(
                f"user message at position {index} has no assistant response"
            )
        records.append({
            "query": current["text"],
            "original_answer_html": _portable_original_html(following["html"]),
            "original_answer_text": following["text"],
        })

    image_count = sum(
        1
        for part in message.walk()
        if part.get_content_maintype() == "image"
    )
    decoded_mhtml = raw_mhtml.decode("utf-8", errors="ignore")
    share_match = SHARE_URL_PATTERN.search(decoded_mhtml)
    share_url = share_match.group(0) if share_match else ""
    payload: dict[str, Any] = {
        "schema_version": "debug_agent_system.shared_query_baseline.v1",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "mhtml",
            "name": mhtml_path.name,
            "sha256": _sha256(mhtml_path),
            "message_count": len(messages),
            "query_answer_count": len(records),
            "embedded_image_count": image_count,
            "share_url": share_url,
        },
        "records": records,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _public_share_api_url(share_url: str) -> str:
    parsed = url_parse.urlsplit(str(share_url or "").strip())
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[-2] != "share":
        raise ValueError("share URL must end with /share/<token>")
    return url_parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        f"/api/public/shares/{parts[-1]}",
        "",
        "",
    ))


def _public_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            value = block.get("text") or block.get("content")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts).strip()


def import_public_share(
    share_url: str,
    baseline_path: Path,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Import a sanitized public session payload using only the stdlib."""

    api_url = _public_share_api_url(share_url)
    req = url_request.Request(
        api_url,
        headers={"Accept": "application/json"},
    )
    parsed_api_url = url_parse.urlsplit(api_url)
    try:
        address = ipaddress.ip_address(parsed_api_url.hostname or "")
    except ValueError:
        address = None
    opener = (
        url_request.build_opener(url_request.ProxyHandler({}))
        if address and (address.is_private or address.is_loopback)
        else url_request.build_opener()
    )
    try:
        with opener.open(req, timeout=timeout_seconds) as response:
            raw = response.read()
    except url_error.HTTPError as exc:
        exc.read()
        raise RuntimeError(f"public_share_http_{exc.code}") from exc
    except (url_error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"public_share_transport:{type(exc).__name__}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("public_share_invalid_json") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("public_share_payload_not_object")
    share_payload = payload.get("share", payload)
    if not isinstance(share_payload, dict):
        raise RuntimeError("public_share_payload_not_object")
    raw_messages = share_payload.get("messages")
    if not isinstance(raw_messages, list):
        raise RuntimeError("public_share_missing_messages")

    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        text = _public_message_text(item)
        if text:
            messages.append({"role": role, "text": text})

    records: list[dict[str, Any]] = []
    for index, current in enumerate(messages):
        if current["role"] != "user":
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        if not following or following["role"] != "assistant":
            raise ValueError(
                f"user message at position {index} has no assistant response"
            )
        records.append({
            "query": current["text"],
            "original_answer_html": (
                "<pre>" + escape(following["text"]) + "</pre>"
            ),
            "original_answer_text": following["text"],
        })
    if not records:
        raise RuntimeError("public_share_contains_no_query_answer_pairs")

    baseline: dict[str, Any] = {
        "schema_version": "debug_agent_system.shared_query_baseline.v1",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "public_share",
            "title": str(share_payload.get("title") or ""),
            "share_url": share_url,
            "api_url": api_url,
            "message_count": len(messages),
            "query_answer_count": len(records),
            "asset_count": int(share_payload.get("asset_count") or 0),
            "created_at": str(share_payload.get("created_at") or ""),
            "expires_at": str(share_payload.get("expires_at") or ""),
        },
        "records": records,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return baseline


def _load_baseline(
    baseline_path: Path,
    mhtml_path: Path | None,
    share_url: str | None,
) -> dict[str, Any]:
    if share_url:
        return import_public_share(share_url, baseline_path)
    if mhtml_path is not None:
        return import_mhtml(mhtml_path, baseline_path)
    if baseline_path.is_file():
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    if DEFAULT_MHTML.is_file():
        return import_mhtml(DEFAULT_MHTML, baseline_path)
    raise FileNotFoundError(
        f"baseline not found: {baseline_path}; pass --mhtml to import it"
    )


def _load_pure_codex_baseline(
    baseline_path: Path,
    share_url: str | None,
) -> dict[str, Any] | None:
    if share_url:
        baseline = import_public_share(share_url, baseline_path)
        baseline.setdefault("source", {})["baseline_role"] = "pure_codex"
        baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return baseline
    if baseline_path.is_file():
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    return None


def _attach_pure_codex_answers(
    records: list[dict[str, Any]],
    pure_codex_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    if not pure_codex_baseline:
        return {
            "available": 0,
            "matched": 0,
            "extra": 0,
            "extra_queries": [],
        }
    pure_records = pure_codex_baseline.get("records") or []
    source = pure_codex_baseline.get("source") or {}
    parsed_share_url = url_parse.urlsplit(
        str(source.get("share_url") or "")
    )
    share_origin = url_parse.urlunsplit((
        parsed_share_url.scheme,
        parsed_share_url.netloc,
        "",
        "",
        "",
    )).rstrip("/")
    by_query = {
        str(item.get("query") or "").strip(): item
        for item in pure_records
        if str(item.get("query") or "").strip()
    }
    matched = 0
    record_queries = {
        str(record.get("query") or "").strip() for record in records
    }
    for record in records:
        item = by_query.get(str(record.get("query") or "").strip())
        if not item:
            continue
        answer_text = str(item.get("original_answer_text") or "")
        if share_origin:
            answer_text = answer_text.replace(
                "](/api/public/shares/",
                f"]({share_origin}/api/public/shares/",
            )
        record["pure_codex_answer_html"] = str(
            item.get("original_answer_html")
            or ("<pre>" + escape(answer_text) + "</pre>")
        )
        record["pure_codex_answer_text"] = answer_text
        record["pure_codex_metrics"] = _answer_metrics(answer_text)
        matched += 1
    missing = [
        str(record.get("query") or "")
        for record in records
        if "pure_codex_metrics" not in record
    ]
    if missing:
        raise ValueError(
            "pure Codex baseline does not cover current queries: "
            + " | ".join(missing[:5])
        )
    extra_queries = list(dict.fromkeys(
        str(item.get("query") or "").strip()
        for item in pure_records
        if (
            str(item.get("query") or "").strip()
            and str(item.get("query") or "").strip() not in record_queries
        )
    ))
    return {
        "available": len(pure_records),
        "matched": matched,
        "extra": len(extra_queries),
        "extra_queries": extra_queries,
    }


def _attach_kg_raw_codex_answer(
    records: list[dict[str, Any]],
    answer_path: Path | None,
) -> bool:
    """Attach the corpus-direct Codex result to its exact Query only."""

    if answer_path is None or not answer_path.is_file():
        return False
    payload = json.loads(answer_path.read_text(encoding="utf-8"))
    query = str(payload.get("query") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    if not query or not answer:
        raise ValueError("KG_v2+raw Codex artifact misses query or answer")
    matched = [
        record
        for record in records
        if str(record.get("query") or "").strip() == query
    ]
    if len(matched) != 1:
        raise ValueError(
            "KG_v2+raw Codex artifact must match exactly one baseline Query"
        )
    record = matched[0]
    record.pop("kg_v2_raw_codex_failure", None)
    record["kg_v2_raw_codex_answer_text"] = answer
    record["kg_v2_raw_codex_metrics"] = _answer_metrics(answer)
    record["kg_v2_raw_codex_metadata"] = {
        "schema_version": str(payload.get("schema_version") or ""),
        "model": str(payload.get("model") or ""),
        "runtime": dict(payload.get("runtime") or {}),
        "prompt": dict(payload.get("prompt") or {}),
        "usage": dict(payload.get("usage") or {}),
        "allowed_roots": list(payload.get("allowed_roots") or []),
        "files_read": list(payload.get("files_read") or []),
        "media_exposed": list(payload.get("media_exposed") or []),
        "tool_call_count": len(payload.get("tool_trace") or []),
        "required_facets": list(payload.get("required_facets") or []),
        "verification": dict(payload.get("verification") or {}),
        "artifact_path": (
            answer_path.resolve().relative_to(REPO_ROOT).as_posix()
        ),
    }
    return True


def _attach_kg_raw_codex_answer_dir(
    records: list[dict[str, Any]],
    answer_dir: Path | None,
) -> int:
    """Attach every valid per-query artifact from a resumable batch."""

    if answer_dir is None or not answer_dir.is_dir():
        return 0
    attached = 0
    for path in sorted(answer_dir.glob("*.json")):
        if (
            path.name == "batch_manifest.json"
            or path.name.endswith(".failure.json")
        ):
            continue
        attached += int(_attach_kg_raw_codex_answer(records, path))
    for path in sorted(answer_dir.glob("*.failure.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        query = str(payload.get("query") or "").strip()
        matched = [
            record
            for record in records
            if str(record.get("query") or "").strip() == query
        ]
        if len(matched) == 1:
            record = matched[0]
            # A failure artifact from the current overlay must not leave a
            # successful answer from an older batch attached to the record.
            # Otherwise the report would silently present stale output as the
            # result of the current rerun.
            record.pop("kg_v2_raw_codex_answer_text", None)
            record.pop("kg_v2_raw_codex_metrics", None)
            record.pop("kg_v2_raw_codex_metadata", None)
            record["kg_v2_raw_codex_failure"] = {
                "error": str(payload.get("error") or "unknown_failure"),
                "attempts": int(payload.get("attempts") or 0),
                "artifact_path": (
                    path.resolve().relative_to(REPO_ROOT).as_posix()
                ),
            }
    return attached


def _count_raw_disk_commands(answer: str) -> int:
    pattern = re.compile(
        r"(?im)^\s*(?:diskpart|list\s+(?:disk|partition|volume)|"
        r"sel(?:ect)?\s+(?:disk|partition|volume)|assign\s+letter|"
        r"bootrec\b|bcdboot\b)"
    )
    return len(pattern.findall(str(answer or "")))


def _public_failure_detail(error: Any) -> str:
    """Keep reports useful without publishing gateway billing/request details."""
    detail = str(error or "未产生通过校验的产物")
    if "insufficient_user_quota" in detail:
        return (
            "Codex Responses API HTTP 403："
            "insufficient_user_quota（用户额度不足）"
        )
    return detail


def _answer_metrics(answer: str) -> dict[str, Any]:
    answer = str(answer or "")
    return {
        "answer_length": len(answer),
        "generic_followup_count": answer.count("为了进一步精确判断"),
        "raw_disk_command_count": _count_raw_disk_commands(answer),
        "safety_detail_red_flags": sum(
            answer.lower().count(value.lower())
            for value in (
                "短接主板", "clr_cmos", "jbat1", "重新涂抹硅脂",
                "只插入一条内存", "断开所有非系统硬盘",
            )
        ),
        "source_marker_count": answer.count("【来源："),
        "image_marker_count": answer.count("[图片："),
        "markdown_image_count": len(re.findall(r"!\[[^\]]*\]\([^)]+\)", answer)),
    }


def _response_metrics(response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer") or "")
    metadata = response.get("metadata") or {}
    retrieval = metadata.get("retrieval") or {}
    trace = retrieval.get("trace") or {}
    coverage = metadata.get("answer_coverage") or {}
    evidence_pack = metadata.get("evidence_pack") or {}
    answer_composer = metadata.get("answer_composer") or {}
    metrics = _answer_metrics(answer)
    metrics.update({
        "status": response.get("status"),
        "lock_status": (response.get("observability") or {}).get("lock_status"),
        "family_id": response.get("family_id") or "",
        "variant_id": response.get("variant_id") or "",
        "section_titles": [
            section.get("title")
            for section in response.get("answer_sections") or []
        ],
        "required_data_count": len(response.get("required_data") or []),
        "eligible_fact_count": (
            coverage.get("eligible_fact_count", 0)
        ),
        "included_fact_count": (
            coverage.get("included_fact_count", 0)
        ),
        "coverage_complete": bool(coverage.get("complete")),
        "query_facets_complete": bool(
            coverage.get("query_facets_complete")
        ),
        "evidence_floor_met": bool(
            coverage.get("evidence_floor_met")
        ),
        "grounded_item_count": int(
            coverage.get("grounded_item_count") or 0
        ),
        "supported_query_facets": list(
            coverage.get("supported_query_facets") or []
        ),
        "uncovered_query_facets": list(
            coverage.get("uncovered_query_facets") or []
        ),
        "evidence_pack_built": (
            evidence_pack.get("schema_version")
            in {
                "debug_agent_system.answer_evidence_pack.v1",
                "debug_agent_system.answer_evidence_pack.v2",
            }
        ),
        "answer_composer_enabled": bool(answer_composer.get("enabled")),
        "answer_composer_provider": str(
            answer_composer.get("provider") or ""
        ),
        "answer_composer_attempted": bool(answer_composer.get("attempted")),
        "answer_composer_used": bool(answer_composer.get("used")),
        "answer_composer_fallback": bool(
            answer_composer.get("fallback_used")
        ),
        "answer_composer_fallback_reason": str(
            answer_composer.get("fallback_reason") or ""
        ),
        "direct_documents": [
            item.get("source_label")
            for item in trace.get("direct_document_matches") or []
            if item.get("source_label")
        ],
        "navigation_documents": [
            item.get("source_label")
            for item in trace.get("navigation_document_matches") or []
            if item.get("source_label")
        ],
    })
    return metrics


def _relative_answer(answer: str) -> str:
    data_root = str((REPO_ROOT / "data").resolve())
    return str(answer or "").replace(f"{data_root}/", REPORT_DATA_PREFIX)


def _automatic_gate(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    """Apply only objective runtime invariants; semantic quality remains visible."""

    reasons: list[str] = []
    if metrics["generic_followup_count"]:
        reasons.append("仍含通用追加追问")
    if metrics["raw_disk_command_count"]:
        reasons.append("仍直接输出磁盘/引导命令")
    if metrics["safety_detail_red_flags"]:
        reasons.append("仍含被禁止的高风险执行细节")
    if not metrics["coverage_complete"] and metrics["eligible_fact_count"]:
        reasons.append("合格事实覆盖不完整")
    if not metrics["evidence_pack_built"]:
        reasons.append("未生成 Evidence Pack")
    if not metrics["evidence_floor_met"]:
        reasons.append("没有经过批准且可追溯的正文证据")
    if not metrics["query_facets_complete"]:
        reasons.append(
            "Query 子任务证据未闭包："
            + "、".join(metrics["uncovered_query_facets"])
        )
    return ("通过" if not reasons else "需复核"), reasons


def _metric_total(
    records: list[dict[str, Any]],
    side: str,
    metric: str,
) -> int:
    return sum(int(record[f"{side}_metrics"][metric]) for record in records)


def _render_report(
    records: list[dict[str, Any]],
    generated_at: str,
    baseline_source: dict[str, Any],
    pure_codex_source: dict[str, Any] | None = None,
    pure_codex_alignment: dict[str, Any] | None = None,
    fixed_results_reused: bool = False,
    refreshed_fixed_queries: list[str] | None = None,
) -> str:
    refreshed_fixed_queries = refreshed_fixed_queries or []
    pure_codex_source = pure_codex_source or {}
    pure_codex_alignment = pure_codex_alignment or {
        "available": 0,
        "matched": 0,
        "extra": 0,
        "extra_queries": [],
    }
    passed = sum(record["automatic_gate"] == "通过" for record in records)
    fixed_locked = sum(
        bool(record["fixed_metrics"]["variant_id"]) for record in records
    )
    original_long = sum(
        record["original_metrics"]["answer_length"] > 5000
        for record in records
    )
    fixed_long = sum(
        record["fixed_metrics"]["answer_length"] > 5000
        for record in records
    )
    pure_codex_long = sum(
        record["pure_codex_metrics"]["answer_length"] > 5000
        for record in records
    )
    packs_built = sum(
        record["fixed_metrics"]["evidence_pack_built"]
        for record in records
    )
    facets_complete = sum(
        record["fixed_metrics"]["query_facets_complete"]
        for record in records
    )
    evidence_floor_met = sum(
        record["fixed_metrics"]["evidence_floor_met"]
        for record in records
    )
    composer_attempted = sum(
        record["fixed_metrics"]["answer_composer_attempted"]
        for record in records
    )
    composer_used = sum(
        record["fixed_metrics"]["answer_composer_used"]
        for record in records
    )
    composer_fallback = sum(
        record["fixed_metrics"]["answer_composer_fallback"]
        for record in records
    )
    composer_transport_fallback = sum(
        "transport" in str(
            record["fixed_metrics"]["answer_composer_fallback_reason"] or ""
        )
        for record in records
    )
    no_grounded_content = sum(
        record["fixed_metrics"]["answer_composer_fallback_reason"]
        == "no_approved_grounded_content"
        for record in records
    )
    record_count = len(records)
    kg_raw_records = [
        record
        for record in records
        if str(record.get("kg_v2_raw_codex_answer_text") or "").strip()
    ]
    kg_raw_missing = [
        (index, str(record.get("query") or ""))
        for index, record in enumerate(records, start=1)
        if not str(record.get("kg_v2_raw_codex_answer_text") or "").strip()
    ]
    kg_raw_total_tokens = sum(
        int(
            (
                (record.get("kg_v2_raw_codex_metadata") or {})
                .get("usage", {})
            ).get("total_tokens")
            or 0
        )
        for record in kg_raw_records
    )
    kg_raw_tool_calls = sum(
        int(
            (record.get("kg_v2_raw_codex_metadata") or {})
            .get("tool_call_count")
            or 0
        )
        for record in kg_raw_records
    )
    kg_raw_media = sum(
        len(
            (record.get("kg_v2_raw_codex_metadata") or {})
            .get("media_exposed")
            or []
        )
        for record in kg_raw_records
    )
    kg_raw_prompt_versions: dict[str, int] = {}
    for record in kg_raw_records:
        version = str(
            (
                (record.get("kg_v2_raw_codex_metadata") or {})
                .get("prompt")
                or {}
            ).get("system_version")
            or "未记录"
        )
        kg_raw_prompt_versions[version] = (
            kg_raw_prompt_versions.get(version, 0) + 1
        )
    kg_raw_prompt_version_text = "；".join(
        f"`{version}`={count}"
        for version, count in sorted(kg_raw_prompt_versions.items())
    )
    baseline_type = baseline_source.get("type", "unknown")
    baseline_url = str(baseline_source.get("share_url") or "")
    baseline_origin = (
        f"[DeepSeek 读侧管线分享页]({baseline_url})"
        if baseline_url
        else "用户提供的 DeepSeek 读侧管线结果"
    )
    pure_codex_url = str(pure_codex_source.get("share_url") or "")
    pure_codex_origin = (
        f"[纯 Codex 分享页]({pure_codex_url})"
        if pure_codex_url
        else "冻结的纯 Codex 分享结果"
    )
    original_detail_label = (
        "DeepSeek 读侧管线（离线快照全量正文）"
        if baseline_type == "mhtml"
        else "DeepSeek 读侧管线（分享页全量正文）"
    )
    lines = [
        "# KG_v2 读侧 Codex 升级与分享 Query 全量对比",
        "",
        f"- 生成时间：{generated_at}",
        f"- DeepSeek 读侧管线来源：{baseline_origin}；当前报告使用冻结的"
        f"`{baseline_type}` 基线，避免分享服务状态导致结果漂移。",
        f"- 基线类型：`{baseline_source.get('type', 'unknown')}`；"
        f"名称：`{baseline_source.get('name', '')}`",
        f"- 基线 SHA-256：`{baseline_source.get('sha256', 'n/a')}`",
        f"- 基线消息：{baseline_source.get('message_count', 0)} 条；"
        f"有效 Query/Answer：{len(records)} 组；"
        f"图片/附件："
        f"{baseline_source.get('embedded_image_count', baseline_source.get('asset_count', 0))} 个。",
        f"- 纯 Codex 来源：{pure_codex_origin}；分享页共 "
        f"{pure_codex_alignment['available']} 组 Query/Answer，其中 "
        f"{pure_codex_alignment['matched']} 组与当前基线精确同题，"
        f"额外 {pure_codex_alignment['extra']} 组不纳入本次同题门禁对比。",
        "- Codex 读侧管线结果来源："
        + (
            (
                "复用已保存的全量结果，并仅重新运行 "
                f"{len(refreshed_fixed_queries)} 条指定 Query："
                + "；".join(
                    f"`{query}`" for query in refreshed_fixed_queries
                )
                + "。"
            )
            if refreshed_fixed_queries
            else (
                "复用已保存的 "
                "`DebugAgentSystem.start(..., interactive=false)` 逐条实际运行响应；"
                "本轮未重复运行 47 条当前管线。"
            )
            if fixed_results_reused
            else "当前 `DebugAgentSystem.start(..., interactive=false)` "
            "逐条实际运行。"
        ),
        f"- 本次答案组织：Codex 显式启用；调用 {composer_attempted}/{len(records)}，"
        f"通过本地 verifier 并采用 {composer_used}/{len(records)}，"
        f"确定性降级 {composer_fallback}/{len(records)}。",
        "- 正文展示规则：DeepSeek 读侧管线、纯 Codex 与 Codex 读侧管线"
        "均保留全量正文，"
        "不做摘要或人工改写；"
        "MHTML 中只能由分享服务访问的图片替换为同位置图片说明。",
        f"- `KG_v2+raw Codex读侧管线` 已实测并附入 "
        f"{len(kg_raw_records)}/{record_count} 条 Query：Codex 只能通过"
        "只读工具检索并读取 `data/raw` 与 `data/kg_v2`，不经过 SAG、"
        "Evidence Pack 或被冻结的当前回答 Composer；最终草稿必须通过"
        "本地 Query facet、来源和媒体校验。",
        "",
        "## 1. 全量总体变化",
        "",
        f"| 指标 | DeepSeek 读侧管线（全 {record_count} 项） | "
        f"纯 Codex（同 {record_count} 项） | "
        f"Codex 读侧管线（同 {record_count} 项） |",
        "|---|---:|---:|---:|",
        "| 通用“为了进一步精确判断”追加追问 | "
        f"{_metric_total(records, 'original', 'generic_followup_count')} | "
        f"{_metric_total(records, 'pure_codex', 'generic_followup_count')} | "
        f"{_metric_total(records, 'fixed', 'generic_followup_count')} |",
        "| 超过 5000 字符的回答 | "
        f"{original_long} | {pure_codex_long} | {fixed_long} |",
        "| 直接暴露磁盘/引导命令 | "
        f"{_metric_total(records, 'original', 'raw_disk_command_count')} | "
        f"{_metric_total(records, 'pure_codex', 'raw_disk_command_count')} | "
        f"{_metric_total(records, 'fixed', 'raw_disk_command_count')} |",
        "| 高风险执行细节命中 | "
        f"{_metric_total(records, 'original', 'safety_detail_red_flags')} | "
        f"{_metric_total(records, 'pure_codex', 'safety_detail_red_flags')} | "
        f"{_metric_total(records, 'fixed', 'safety_detail_red_flags')} |",
        "| 锁定故障 Variant | 未输出结构化字段 | 未输出结构化字段 | "
        f"{fixed_locked}/{len(records)} |",
        "| Evidence Pack 生成 | 未输出结构化字段 | 未输出结构化字段 | "
        f"{packs_built}/{len(records)} |",
        "| 可追溯正文证据门槛 | 未输出结构化字段 | 未输出结构化字段 | "
        f"{evidence_floor_met}/{len(records)} |",
        "| Query 子任务证据闭包 | 未输出结构化字段 | 未输出结构化字段 | "
        f"{facets_complete}/{len(records)} |",
        "| Codex Composer 调用/采用/降级 | 未输出结构化字段 | 不适用 | "
        f"{composer_attempted}/{composer_used}/{composer_fallback} |",
        "| KG_v2 客观运行门禁通过 | 未设置 | 未设置 | "
        f"{passed}/{len(records)} |",
        "",
        "> “客观运行门禁”只检查通用追问、危险命令/细节和证据覆盖，"
        "不把它冒充为语义正确率。每条回答的召回文档和完整正文均在下方保留，"
        "用于审阅主题是否正确。",
        "",
        "### 1.1 对比结论",
        "",
        f"- 当前管线不是对纯 Codex 的全面替代结论：{facets_complete}/{record_count} "
        "达到可追溯正文与 Query 子任务闭包，可在这些问题上提供更稳定的来源、安全门、"
        "诊断状态和确定性降级；其余问题仍需补齐写侧/索引证据。",
        f"- 新增的纯 Codex 分享页与 DeepSeek 读侧管线基线有 "
        f"{pure_codex_alignment['matched']} "
        "个精确同题。常规条目按“DeepSeek 读侧管线 → 纯 Codex → "
        "Codex 读侧管线”展示同题全量正文；已通过旁路校验的条目再加入"
        "“KG_v2+raw Codex读侧管线”。纯 Codex 的优势主要是开放知识编织与表达完整；"
        "它没有输出"
        "Evidence Pack、Query facet、Variant 锁定和安全门等可机器验证状态。",
        f"- Codex Composer 在 {composer_attempted} 条有正文证据的 Query 上采用 "
        f"{composer_used} 条；{composer_transport_fallback} 条因瞬时传输错误降级，"
        f"{no_grounded_content} 条因无批准正文证据而不调用模型。",
        f"- 相比 DeepSeek 读侧管线基线，通用追加追问由 "
        f"{_metric_total(records, 'original', 'generic_followup_count')} 降为 "
        f"{_metric_total(records, 'fixed', 'generic_followup_count')}，直接暴露的磁盘/"
        f"引导命令命中由 {_metric_total(records, 'original', 'raw_disk_command_count')} "
        f"降为 {_metric_total(records, 'fixed', 'raw_disk_command_count')}；这些是"
        "可自动验证的组织和安全改善，不等于语义正确率。",
        f"- {record_count - facets_complete} 条未通过项中，"
        f"{no_grounded_content} 条没有批准且可追溯的正文证据；此时纯 Codex 的通用知识"
        "可能更完整，但当前管线选择明确暴露知识边界，不让模型绕开 KG_v2 证据闭包。",
    ]
    failed_records = [
        (index, record)
        for index, record in enumerate(records, start=1)
        if record["automatic_gate"] != "通过"
    ]
    if failed_records:
        lines.extend([
            "",
            "### 1.2 未通过门禁项",
            "",
            "| # | Query | 原因 |",
            "|---:|---|---|",
        ])
        for index, record in failed_records:
            lines.append(
                f"| {index} | {record['query']} | "
                f"{'；'.join(record['gate_reasons'])} |"
            )
        lines.extend([
            "",
            "> 未通过表示 Evidence Pack 没有批准且可追溯的正文证据，或没有为某个 "
            "Query 子任务建立证据闭包。无正文证据时不调用 Codex；有证据但子任务 "
            "未闭包时也不能让模型在候选池外补写。",
        ])
    extra_queries = list(pure_codex_alignment.get("extra_queries") or [])
    if extra_queries:
        lines.extend([
            "",
            "### 1.3 纯 Codex 分享页的额外 Query",
            "",
            "以下 Query 不在冻结的 47 项同题基线中，因此不计入三方指标，"
            "也没有在本报告中伪造对应的 KG_v2 对照结果：",
            "",
        ])
        lines.extend(
            f"{index}. {query}"
            for index, query in enumerate(extra_queries, start=1)
        )
    lines.extend([
        "",
        "### 1.4 KG_v2+raw Codex 旁路全量实测",
        "",
        f"- 通过本地证据校验并写入完整回答："
        f"{len(kg_raw_records)}/{record_count}。",
        f"- 已通过结果合计工具调用：{kg_raw_tool_calls}；"
        f"合计 token：{kg_raw_total_tokens}；"
        f"暴露可引用媒体：{kg_raw_media}。",
        f"- System Prompt 版本分布：{kg_raw_prompt_version_text}。"
        "本节是管线演进过程中的真实批次结果，不把混合版本伪称为同版本横评；"
        "批处理恢复逻辑会只复用当前 prompt 版本，严格同版复测需重新调用旧版本项。",
        "- 这里的“通过”只表示 Query facet、来源、文件读取范围和媒体引用"
        "满足本地 verifier，不等同于人工语义评分。",
    ])
    if kg_raw_missing:
        lines.extend([
            "- 以下 Query 未产生通过 verifier 的旁路回答，正文不被写入，"
            "避免把失败草稿伪装成正式结果：",
            "",
        ])
        for index, query in kg_raw_missing:
            failure = (
                records[index - 1].get("kg_v2_raw_codex_failure") or {}
            )
            detail = _public_failure_detail(failure.get("error"))
            lines.append(f"  - #{index} {query}：`{detail}`")
    current_cli_batch = [
        (index, record)
        for index, record in enumerate(records, start=1)
        if (
            str(
                (
                    (record.get("kg_v2_raw_codex_metadata") or {})
                    .get("runtime")
                    or {}
                ).get("engine")
                or ""
            ) == "codex_cli"
            or str(
                (record.get("kg_v2_raw_codex_failure") or {}).get("error")
                or ""
            ).startswith("CodexCliAgentError:")
        )
    ]
    current_batch_size = len(current_cli_batch)
    current_pass_count = sum(
        bool(record.get("kg_v2_raw_codex_metadata"))
        for _, record in current_cli_batch
    )
    current_engines = sorted({
        str(
            (
                (record.get("kg_v2_raw_codex_metadata") or {})
                .get("runtime")
                or {}
            ).get("engine")
            or "unknown"
        )
        for _, record in current_cli_batch
        if record.get("kg_v2_raw_codex_metadata")
    })
    current_failure_count = sum(
        bool(record.get("kg_v2_raw_codex_failure"))
        for _, record in current_cli_batch
    )
    if current_batch_size:
        current_source_markers = sum(
            int(
                (record.get("kg_v2_raw_codex_metrics") or {})
                .get("source_marker_count")
                or 0
            )
            for _, record in current_cli_batch
        )
        current_images = sum(
            int(
                (record.get("kg_v2_raw_codex_metrics") or {})
                .get("markdown_image_count")
                or 0
            )
            for _, record in current_cli_batch
        )
        pure_images = sum(
            int(record["pure_codex_metrics"]["markdown_image_count"])
            for _, record in current_cli_batch
        )
        current_prompt_versions = sorted({
            str(
                (
                    (record.get("kg_v2_raw_codex_metadata") or {})
                    .get("prompt")
                    or {}
                ).get("system_version")
                or "未记录"
            )
            for _, record in current_cli_batch
            if record.get("kg_v2_raw_codex_metadata")
        })
        lines.extend([
            "",
            "### 1.5 当前 KG_v2+raw Codex 本地登录态复测批次",
            "",
            f"- 本批次覆盖 {current_batch_size} 条：通过并写入完整回答 "
            f"{current_pass_count}/{current_batch_size}；调用失败 "
            f"{current_failure_count}/{current_batch_size}。",
            f"- 运行引擎：`{' / '.join(current_engines)}`。本轮由本机 "
            "Codex 登录态执行，模型自主规划只读 shell 检索、迭代读取 "
            "`data/kg_v2` 与 `data/raw` 并组织回答；不读取 `.env.local`，"
            "不使用预制 TopK 或 Query 定向文件规则。",
            f"- Prompt 版本：`{' / '.join(current_prompt_versions)}`。同一批次包含"
            "演进中的多个 Prompt 版本时，本节会明确标出，不把它称为同版横评。",
            f"- 与纯 Codex 同题回答相比，本批次增加 {current_source_markers} 个"
            "逐段来源标记；纯 Codex / 当前旁路分别引用 "
            f"{pure_images}/{current_images} 张图片。图片数减少不自动等于质量下降："
            "当前旁路只保留已读来源中与闭包步骤相关且去重后的媒体，但仍需人工检查"
            "是否漏掉有解释价值的不同界面。",
            "",
            "| # | 状态 | 模型 / Prompt | 回答长度 | 只读文件 | "
            "工具调用 | 媒体 | token | 纯 Codex / 旁路长度 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for index, record in current_cli_batch:
            metadata = record.get("kg_v2_raw_codex_metadata") or {}
            prompt = metadata.get("prompt") or {}
            usage = metadata.get("usage") or {}
            failure = record.get("kg_v2_raw_codex_failure") or {}
            if metadata:
                lines.append(
                    f"| {index} | `passed` | "
                    f"`{metadata.get('model', '')}` / "
                    f"`{prompt.get('system_version', '未记录')}` | "
                    f"{len(str(record.get('kg_v2_raw_codex_answer_text') or ''))} | "
                    f"{len(metadata.get('files_read') or [])} | "
                    f"{metadata.get('tool_call_count', 0)} | "
                    f"{len(metadata.get('media_exposed') or [])} | "
                    f"{int(usage.get('total_tokens') or 0)} | "
                    f"{record['pure_codex_metrics']['answer_length']} / "
                    f"{len(str(record.get('kg_v2_raw_codex_answer_text') or ''))} |"
                )
            else:
                lines.append(
                    f"| {index} | `failed`："
                    f"{_public_failure_detail(failure.get('error'))} | "
                    "— | 0 | 0 | 0 | 0 | 0 | — |"
                )
        lines.extend([
            "",
            "#### 与纯 Codex 的阶段性评价",
            "",
            "- 做得更好的部分：通过项均有可审计的 KG/raw 双域读取记录和逐项"
            "facet ledger；Dism++、安全模式及能进/不能进系统分支可回到原始文档，"
            "流程图片只从实际读取资料中物化。v14 还要求多条件答案使用“分支标题 + "
            "分支内独立编号”。",
            "- 仍需优化的部分：旁路回答通常比纯 Codex 长，来源标记密度和调查 token"
            "偏高；发布重试会完整重跑调查。调用失败项只保留失败审计，不用旧答案或"
            "其他模型静默填充。",
            "- 当前结论：可追溯性、文档覆盖和安全控制更强，简洁性和运行成本更弱。"
            "后续应优化来源聚合、工具输出压缩和可恢复重试，而不是增加 Query 字符串或"
            "文件名规则。",
        ])
    lines.extend([
        "",
        "## 2. 全量逐项索引",
        "",
        "| # | Query | DeepSeek 读侧管线长度 | 纯 Codex 长度 | "
        "Codex 读侧管线长度 | 当前路由 | 主文档/导航文档 | 门禁 |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ])
    for index, record in enumerate(records, start=1):
        metrics = record["fixed_metrics"]
        sources = list(dict.fromkeys([
            *metrics["direct_documents"],
            *metrics["navigation_documents"],
        ]))
        route = (
            f"`{metrics['status']}` / `{metrics['lock_status']}` / "
            f"Variant={metrics['variant_id'] or '未锁定'}"
        )
        lines.append(
            f"| {index} | {record['query']} | "
            f"{record['original_metrics']['answer_length']} | "
            f"{record['pure_codex_metrics']['answer_length']} | "
            f"{metrics['answer_length']} | {route} | "
            f"{'；'.join(sources) or '无'} | "
            f"**{record['automatic_gate']}** |"
        )

    lines.extend([
        "",
        "## 3. 通用机制修复说明",
        "",
        f"> 生产代码没有加入这 {record_count} 条 Query 的字符串特判。具体 Query 只作为"
        "回归 goldcase；修复作用于所有输入共同经过的意图、证据域和回答组织层。",
        "",
        "1. Query task v2 将输入解析为实体、对象、操作、条件、顺序、比较、版本、"
        "交付物和故障对象域，再决定知识回答、故障诊断或规格/配置/升级流程等证据角色。",
        "2. 文档标题、章节标题、导航子文档和正文 Chunk 共用一套"
        "“主题—任务—实体—条件—对象域”适用性评分；不再由某个短词或原始 BM25"
        "单独决定主文档。",
        "3. 工具名、型号、厂商和授权介质作为证据域约束；附件名会在剥离已知"
        "扩展名后做严格实体匹配，但不会把 `DISM` 与 `Dism++`、短型号与长 SKU"
        "混为一体。",
        "4. 文档导航只展开与当前意图兼容的已解析子文档；父目录元数据仅在用户"
        "询问负责人、时间或目录时进入正文。",
        "5. 复合 Query 使用同一 operation facet 解析驱动导航首跳选择和答案覆盖；"
        "每个有证据的子任务必须进入 Evidence Pack，不能只保留最高分单分支。",
        "6. Evidence Pack v2 将条目分为 required、optional 和 excluded，并要求至少"
        "一条批准且可追溯的正文证据；弱候选名称和检索分数不能充当回答证据。",
        "7. `CodexEvidenceAnswerComposer` 可以从 optional 中选择、合并和省略，只"
        "返回本地条目 ID 的章节与顺序；本地 verifier 通过后才从原条目渲染正文。",
        "8. Codex 默认关闭；缺 key、超时、非法 JSON、未知 ID、漏 facet 或漏"
        "事实时只调用一次并回退确定性 `EvidenceAnswerComposer`。Supervisor 不"
        "追加通用追问或二次改写。",
        "9. DOCX 导航链接保留原始 relationship ID、父文档、目标 URL 和 wiki "
        "token；写侧可按这些锚点抓取并入图子文档，随后统一重建文档关系与 SAG。",
        "10. 对“Query facet 未闭包、但召回 Chunk 已包含对应导航入口”的情况，"
        "运行时在 Codex 编织和本地校验之后确定性加入“资料缺口”，明确已看到的"
        "子任务、来源锚点和缺失原因，且不补写未取得的步骤。",
        "",
        "### 3.1 本次 rId5 写侧状态",
        "",
        "- 已从 `Dism++软件使用教程.docx` 解析并索引 "
        "`rId5 → 制作镜像/备份镜像 → XuDgwZpkjiFtKQkinnJc46dGnPh`，"
        "冻结基线仍保留该父文档来源锚点和既有 SAG 状态。",
        "- 用户新上传的 `data/raw/aoi_debug_agent_sources/制作镜像_备份镜像.docx` "
        "已包含备份正文、1 张源图和来源路径。为保持冻结边界，本轮没有把它写回"
        "既有 KG/SAG；新 `KG_v2+raw Codex读侧管线` 直接只读该 raw 文件，"
        "因此第一条 Query 的 `operation:备份` 已闭包。被冻结的 Codex 读侧管线"
        "仍会按旧证据状态显示导航入口正文缺失，两者差异是预期的对照结果。",
        "",
        f"## 4. {record_count} 项对比完整回答",
        "",
    ])
    for index, record in enumerate(records, start=1):
        original = record["original_metrics"]
        pure_codex = record["pure_codex_metrics"]
        fixed = record["fixed_metrics"]
        sources = list(dict.fromkeys([
            *fixed["direct_documents"],
            *fixed["navigation_documents"],
        ]))
        lines.extend([
            f"### 4.{index} {record['query']}",
            "",
            f"- DeepSeek 读侧管线回答长度：{original['answer_length']}",
            f"- 纯 Codex 回答长度：{pure_codex['answer_length']}",
            f"- Codex 读侧管线状态：`{fixed['status']}`；"
            f"锁定状态：`{fixed['lock_status']}`；"
            f"Variant：`{fixed['variant_id'] or '未锁定'}`",
            f"- Codex 读侧管线召回文档：{'；'.join(sources) or '无'}",
            f"- Codex 读侧管线事实覆盖：{fixed['included_fact_count']}/"
            f"{fixed['eligible_fact_count']}，"
            f"complete={str(fixed['coverage_complete']).lower()}",
            f"- Query facet：支持="
            f"{'、'.join(fixed['supported_query_facets']) or '无'}；"
            f"未覆盖={'、'.join(fixed['uncovered_query_facets']) or '无'}；"
            f"complete={str(fixed['query_facets_complete']).lower()}",
            f"- 正文证据门槛："
            f"evidence_floor_met={str(fixed['evidence_floor_met']).lower()}；"
            f"grounded_item_count={fixed['grounded_item_count']}",
            f"- Evidence Pack："
            f"{'已生成' if fixed['evidence_pack_built'] else '未生成'}；"
            f"Codex Composer（provider="
            f"{fixed['answer_composer_provider'] or '未调用'}）：attempted="
            f"{str(fixed['answer_composer_attempted']).lower()}，"
            f"used={str(fixed['answer_composer_used']).lower()}，"
            f"fallback={str(fixed['answer_composer_fallback']).lower()}，"
            f"reason={fixed['answer_composer_fallback_reason'] or '无'}",
            f"- 客观运行门禁：`{record['automatic_gate']}`",
        ])
        kg_raw_answer = str(
            record.get("kg_v2_raw_codex_answer_text") or ""
        ).strip()
        if kg_raw_answer:
            lines.append(
                f"- KG_v2+raw Codex读侧管线回答长度：{len(kg_raw_answer)}"
            )
        kg_raw_failure = record.get("kg_v2_raw_codex_failure") or {}
        if kg_raw_failure:
            failure_error = _public_failure_detail(
                kg_raw_failure.get("error")
            )
            lines.append(
                "- KG_v2+raw Codex读侧管线本轮状态：`failed`；"
                f"尝试次数：{kg_raw_failure.get('attempts', 0)}；"
                f"原因：{failure_error}"
            )
        if record["gate_reasons"]:
            lines.append("- 门禁原因：" + "；".join(record["gate_reasons"]))
        lines.extend([
            "",
            "<details>",
            f"<summary>{original_detail_label}</summary>",
            "",
            record["original_answer_html"].strip().replace(
                "[原回答图片",
                "[DeepSeek 读侧管线图片",
            ),
            "",
            "</details>",
            "",
            "<details>",
            "<summary>纯 Codex 回答（新增分享页全量正文）</summary>",
            "",
            record["pure_codex_answer_text"].strip(),
            "",
            "</details>",
            "",
        ])
        if kg_raw_answer:
            kg_raw_meta = record.get("kg_v2_raw_codex_metadata") or {}
            kg_raw_prompt = kg_raw_meta.get("prompt") or {}
            kg_raw_files = list(
                dict.fromkeys(kg_raw_meta.get("files_read") or [])
            )
            lines.extend([
                "<details open>",
                "<summary>KG_v2+raw Codex读侧管线（独立旁路全量正文）</summary>",
                "",
                f"> 模型：`{kg_raw_meta.get('model', '')}`；"
                f"System Prompt："
                f"`{kg_raw_prompt.get('system_version', '未记录')}`；"
                f"只读文件：{len(kg_raw_meta.get('files_read') or [])}；"
                f"工具调用：{kg_raw_meta.get('tool_call_count', 0)}；"
                f"可用媒体：{len(kg_raw_meta.get('media_exposed') or [])}；"
                f"facet：{len(kg_raw_meta.get('required_facets') or [])}；"
                f"本地校验："
                f"{'通过' if (kg_raw_meta.get('verification') or {}).get('passed') else '未通过'}。",
                "",
                "> 实际读取源文件："
                + (
                    "；".join(f"`{path}`" for path in kg_raw_files)
                    if kg_raw_files
                    else "无"
                ),
                "",
                _relative_answer(kg_raw_answer),
                "",
                "</details>",
                "",
            ])
        elif kg_raw_failure:
            lines.extend([
                "<details open>",
                "<summary>KG_v2+raw Codex读侧管线（本轮未生成回答）</summary>",
                "",
                "本轮没有可作为正式结果写入的回答。"
                f"调用状态：`failed`；原因：{failure_error}。",
                "",
                "该失败仅表示当前 Agent 运行未完成，不代表本地资料缺失，"
                "也不代表旧批次回答通过了当前版本复测。额度恢复后可按本条 "
                "Query 断点重跑。",
                "",
                "</details>",
                "",
            ])
        lines.extend([
            "<details open>",
            "<summary>Codex 读侧管线（当前管线全量正文）</summary>",
            "",
            _relative_answer(record["response"]["answer"]).strip(),
            "",
            "</details>",
            "",
        ])

    lines.extend([
        "## 5. 已知边界",
        "",
        "- 当前环境未配置 PDF 原文解析器，技嘉、铭瑄、研华部分主板手册尚未进入 "
        "Chunk 索引。相关 Query 应返回 `knowledge_scope_not_covered`，不能用光源"
        "控制器等无关材料代答。",
        "- 离线快照封装了原页面图片，本报告保留图片原位置和说明，但没有把二进制图片"
        "复制进 Markdown；原始图片仍可在同一快照中离线查看。",
        "",
    ])
    return "\n".join(lines)


def generate(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT,
    baseline_path: Path = DEFAULT_BASELINE,
    mhtml_path: Path | None = None,
    share_url: str | None = None,
    enable_answer_composer: bool = False,
    pure_codex_baseline_path: Path = DEFAULT_PURE_CODEX_BASELINE,
    pure_codex_share_url: str | None = None,
    kg_raw_codex_answer_path: Path | None = DEFAULT_KG_RAW_CODEX_ANSWER,
    kg_raw_codex_answer_dir: Path | None = None,
    reuse_fixed_results: bool = False,
    refresh_fixed_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    refresh_fixed_queries = list(dict.fromkeys(
        query.strip()
        for query in (refresh_fixed_queries or [])
        if query.strip()
    ))
    baseline = _load_baseline(baseline_path, mhtml_path, share_url)
    baseline_records = baseline.get("records") or []
    if not baseline_records:
        raise ValueError("baseline contains no query/answer records")
    pure_codex_baseline = _load_pure_codex_baseline(
        pure_codex_baseline_path,
        pure_codex_share_url,
    )
    if not pure_codex_baseline:
        raise FileNotFoundError(
            "pure Codex baseline not found; pass --pure-codex-share-url "
            "to import it"
        )

    result_path = output_dir / "comparison_results.json"
    if reuse_fixed_results or refresh_fixed_queries:
        if not result_path.is_file():
            raise FileNotFoundError(
                f"cannot reuse missing fixed results: {result_path}"
            )
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        records = list(existing.get("records") or [])
        expected_queries = [
            str(item.get("query") or "") for item in baseline_records
        ]
        actual_queries = [
            str(item.get("query") or "") for item in records
        ]
        if actual_queries != expected_queries:
            raise ValueError(
                "reused fixed results do not match the current baseline "
                "query order"
            )
        if refresh_fixed_queries:
            missing = [
                query
                for query in refresh_fixed_queries
                if query not in actual_queries
            ]
            if missing:
                raise ValueError(
                    "cannot refresh queries absent from the baseline: "
                    + "；".join(missing)
                )
            config = load_config(CONFIG_PATH)
            if enable_answer_composer:
                config.read_llm.enabled = True
                config.read_llm.answer_composer_enabled = True
            with tempfile.TemporaryDirectory(
                prefix="shared-query-partial-refresh-"
            ) as temp:
                config.session_store = Path(temp) / "sessions"
                system = DebugAgentSystem(config)
                for index, record in enumerate(records, start=1):
                    if record["query"] not in refresh_fixed_queries:
                        continue
                    response = system.start({
                        "query": record["query"],
                        "interactive": False,
                        "session": {
                            "session_id": f"shared-query-refresh-{index}"
                        },
                    })
                    fixed_metrics = _response_metrics(response)
                    automatic_gate, gate_reasons = _automatic_gate(
                        fixed_metrics
                    )
                    record["fixed_metrics"] = fixed_metrics
                    record["automatic_gate"] = automatic_gate
                    record["gate_reasons"] = gate_reasons
                    record["response"] = response
    else:
        config = load_config(CONFIG_PATH)
        if enable_answer_composer:
            config.read_llm.enabled = True
            config.read_llm.answer_composer_enabled = True
        with tempfile.TemporaryDirectory(
            prefix="shared-query-comparison-"
        ) as temp:
            config.session_store = Path(temp) / "sessions"
            system = DebugAgentSystem(config)
            records = []
            for index, item in enumerate(baseline_records, start=1):
                response = system.start({
                    "query": item["query"],
                    "interactive": False,
                    "session": {
                        "session_id": f"shared-query-fixed-{index}"
                    },
                })
                fixed_metrics = _response_metrics(response)
                automatic_gate, gate_reasons = _automatic_gate(fixed_metrics)
                records.append({
                    "query": item["query"],
                    "original_answer_html": item["original_answer_html"],
                    "original_answer_text": item["original_answer_text"],
                    "original_metrics": _answer_metrics(
                        item["original_answer_text"]
                    ),
                    "fixed_metrics": fixed_metrics,
                    "automatic_gate": automatic_gate,
                    "gate_reasons": gate_reasons,
                    "response": response,
                })

    pure_codex_alignment = _attach_pure_codex_answers(
        records,
        pure_codex_baseline,
    )
    _attach_kg_raw_codex_answer(records, kg_raw_codex_answer_path)
    _attach_kg_raw_codex_answer_dir(records, kg_raw_codex_answer_dir)

    generated_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "baseline_source": baseline.get("source") or {},
                "pure_codex_source": (
                    pure_codex_baseline.get("source") or {}
                ),
                "pure_codex_alignment": pure_codex_alignment,
                "fixed_results_reused": reuse_fixed_results,
                "refreshed_fixed_queries": refresh_fixed_queries,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rendered_report = _render_report(
        records,
        generated_at,
        baseline.get("source") or {},
        pure_codex_baseline.get("source") or {},
        pure_codex_alignment,
        reuse_fixed_results,
        refresh_fixed_queries,
    )
    # Model-authored Markdown may contain accidental end-of-line padding.
    # Normalize it at the report boundary so regenerated artifacts stay clean
    # and diffable without changing any visible answer text.
    rendered_report = "\n".join(
        line.rstrip() for line in rendered_report.splitlines()
    ).rstrip() + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered_report, encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--mhtml",
        type=Path,
        default=None,
        help="import/refresh the JSON baseline from an offline MHTML snapshot",
    )
    parser.add_argument(
        "--share-url",
        default="",
        help=(
            "import/refresh the JSON baseline from a public /share/<token> URL"
        ),
    )
    parser.add_argument(
        "--enable-answer-composer",
        action="store_true",
        help=(
            "attempt the single-call Codex composer; missing key or "
            "verification failure remains deterministic fail-open"
        ),
    )
    parser.add_argument(
        "--pure-codex-baseline",
        type=Path,
        default=DEFAULT_PURE_CODEX_BASELINE,
        help="sanitized JSON baseline imported from the pure Codex share",
    )
    parser.add_argument(
        "--pure-codex-share-url",
        default="",
        help=(
            "import/refresh the pure Codex baseline from a "
            "public /share/<token> URL"
        ),
    )
    parser.add_argument(
        "--kg-raw-codex-answer",
        type=Path,
        default=DEFAULT_KG_RAW_CODEX_ANSWER,
        help=(
            "corpus-direct Codex answer artifact; attached only to its exact "
            "baseline Query"
        ),
    )
    parser.add_argument(
        "--kg-raw-codex-answer-dir",
        type=Path,
        default=None,
        help=(
            "directory containing independent per-query KG_v2+raw Codex "
            "answer artifacts from the resumable batch runner"
        ),
    )
    parser.add_argument(
        "--reuse-fixed-results",
        action="store_true",
        help=(
            "reuse comparison_results.json instead of rerunning the current "
            "runtime; the baseline query order must match exactly"
        ),
    )
    parser.add_argument(
        "--refresh-fixed-query",
        action="append",
        default=[],
        help=(
            "reuse the saved full result set but rerun this exact Query; "
            "repeat the option to refresh more than one Query"
        ),
    )
    args = parser.parse_args()
    records = generate(
        output_dir=args.output_dir,
        report_path=args.report,
        baseline_path=args.baseline,
        mhtml_path=args.mhtml,
        share_url=args.share_url or None,
        enable_answer_composer=args.enable_answer_composer,
        pure_codex_baseline_path=args.pure_codex_baseline,
        pure_codex_share_url=args.pure_codex_share_url or None,
        kg_raw_codex_answer_path=args.kg_raw_codex_answer,
        kg_raw_codex_answer_dir=args.kg_raw_codex_answer_dir,
        reuse_fixed_results=args.reuse_fixed_results,
        refresh_fixed_queries=args.refresh_fixed_query,
    )
    print(json.dumps(
        {
            "records": len(records),
            "passed": sum(
                record["automatic_gate"] == "通过" for record in records
            ),
            "report": str(args.report),
            "baseline": str(args.baseline),
            "pure_codex_baseline": str(args.pure_codex_baseline),
            "output_dir": str(args.output_dir),
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
