"""Local, resumable browser workbench for the W7 human-review gate.

The server intentionally binds to loopback by default.  It renders the full
context and writes annotations back with an atomic replace, so reviewers do
not need to edit the schema by hand.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import secrets
import tempfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlencode, urlparse

from markdown_it import MarkdownIt

from .w7_human_review import BOOLEAN_FIELDS, ISSUE_TAGS, SESSION_VERDICTS, validate_annotations


_MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
_DETAILS_OPEN = re.compile(r"^\s*<details><summary>(.*?)</summary>\s*$")


def strip_embedded_w2_json(source: str) -> str:
    """Remove generated W2 JSON blocks from the main review page.

    A high-risk session can contain dozens of episodes and megabytes of JSON.
    Collapsed ``details`` nodes still make the browser parse all of that text,
    so the workbench exposes the machine JSON through a separate endpoint.
    """
    lines: list[str] = []
    skipping = False
    for line in source.splitlines():
        match = _DETAILS_OPEN.fullmatch(line)
        if not skipping and match and "完整 W2 input JSON" in match.group(1):
            skipping = True
            lines.extend(("", "> 完整 W2 input JSON 已移至页面顶部的按需链接，避免阻塞浏览器渲染。", ""))
            continue
        if skipping:
            if line.strip() == "</details>":
                skipping = False
            continue
        lines.append(line)
    return "\n".join(lines)


def render_markdown_safe(source: str) -> str:
    """Render review Markdown while allowing only generated details wrappers.

    Raw HTML remains disabled because the Markdown contains untrusted chat text.
    The W7 report generator emits exact one-line ``details`` wrappers, so those
    two tags are restored after CommonMark has safely escaped everything else.
    """
    prefix = "W7SAFEDETAILSTOKEN"
    while prefix in source:
        prefix += "X"
    replacements: dict[str, str] = {}
    lines: list[str] = []
    details_index = 0
    for line in source.splitlines():
        match = _DETAILS_OPEN.fullmatch(line)
        if match:
            token = f"{prefix}OPEN{details_index}"
            replacements[f"<p>{token}</p>"] = (
                "<details><summary>" + html.escape(match.group(1)) + "</summary>"
            )
            details_index += 1
            lines.extend(("", token, ""))
        elif line.strip() == "</details>":
            token = f"{prefix}CLOSE{details_index}"
            replacements[f"<p>{token}</p>"] = "</details>"
            details_index += 1
            lines.extend(("", token, ""))
        else:
            lines.append(line)
    rendered = _MARKDOWN.render("\n".join(lines))
    for escaped_token, safe_html in replacements.items():
        rendered = rendered.replace(escaped_token, safe_html)
    return rendered


def _one(form: Mapping[str, list[str]], key: str) -> str:
    values = form.get(key) or []
    return str(values[-1]) if values else ""


def _boolean(value: str, field: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "":
        return None
    raise ValueError(f"invalid boolean value for {field}: {value!r}")


def _optional_positive_int(value: str, field: str) -> int | None:
    if not value.strip():
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def apply_form_update(
    payload: dict[str, Any],
    thread_id: str,
    form: Mapping[str, list[str]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply one browser form submission while preserving all other sessions."""
    session = next(
        (row for row in payload.get("sessions") or [] if str(row.get("thread_id") or "") == thread_id),
        None,
    )
    if session is None:
        raise ValueError(f"unknown thread_id: {thread_id}")

    verdict = _one(form, "session_verdict").strip()
    if verdict and verdict not in SESSION_VERDICTS:
        raise ValueError(f"invalid session verdict: {verdict!r}")
    session_tags = [str(tag) for tag in form.get("session_issue_tags") or []]
    unknown = sorted(set(session_tags) - ISSUE_TAGS)
    if unknown:
        raise ValueError("unknown session issue tags: " + ",".join(unknown))

    reviewer = _one(form, "reviewer").strip()
    reviewed_at = _one(form, "reviewed_at").strip()
    if reviewer and verdict and not reviewed_at:
        reviewed_at = (now or datetime.now(UTC)).isoformat()
    session.update({
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "session_verdict": verdict,
        "session_issue_tags": session_tags,
        "session_notes": _one(form, "session_notes").strip(),
    })

    episodes = [row for row in session.get("episodes") or [] if isinstance(row, dict)]
    for index, episode in enumerate(episodes):
        prefix = f"episode:{index}:"
        for field in BOOLEAN_FIELDS:
            episode[field] = _boolean(_one(form, prefix + field), prefix + field)
        tags = [str(tag) for tag in form.get(prefix + "issue_tags") or []]
        unknown = sorted(set(tags) - ISSUE_TAGS)
        if unknown:
            raise ValueError(f"unknown episode issue tags at {index}: " + ",".join(unknown))
        episode["issue_tags"] = tags
        episode["corrected_fault_focus"] = _one(form, prefix + "corrected_fault_focus").strip()
        episode["corrected_episode_scope"] = _one(form, prefix + "corrected_episode_scope").strip()
        episode["corrected_case_items"] = _one(form, prefix + "corrected_case_items").strip()
        episode["corrected_resolution_status"] = _one(form, prefix + "corrected_resolution_status").strip()
        episode["corrected_resolution_evidence_message_ids"] = _one(
            form, prefix + "corrected_resolution_evidence_message_ids"
        ).strip()
        episode["corrected_trace_group_id"] = _one(form, prefix + "corrected_trace_group_id").strip()
        episode["corrected_trace_phase_index"] = _optional_positive_int(
            _one(form, prefix + "corrected_trace_phase_index"),
            prefix + "corrected_trace_phase_index",
        )
        episode["corrected_trace_phase_count"] = _optional_positive_int(
            _one(form, prefix + "corrected_trace_phase_count"),
            prefix + "corrected_trace_phase_count",
        )
        episode["corrected_w2_readiness"] = _boolean(
            _one(form, prefix + "corrected_w2_readiness"),
            prefix + "corrected_w2_readiness",
        )
        episode["notes"] = _one(form, prefix + "notes").strip()
    payload["last_saved_at"] = (now or datetime.now(UTC)).isoformat()
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON file without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _session_complete(session: dict[str, Any]) -> bool:
    probe = {
        "required_min_sessions": 1,
        "sessions": [session],
    }
    return validate_annotations(probe, min_sessions=1)["completed_sessions"] == 1


def _input(name: str, value: Any, *, size: int = 48) -> str:
    return (
        f'<input name="{html.escape(name)}" value="{html.escape(str(value or ""))}" '
        f'size="{size}">'
    )


def _select_boolean(name: str, value: Any) -> str:
    choices = [("", "未判断"), ("true", "正确"), ("false", "不正确")]
    current = "true" if value is True else "false" if value is False else ""
    options = "".join(
        f'<option value="{raw}"{" selected" if raw == current else ""}>{label}</option>'
        for raw, label in choices
    )
    return f'<select name="{html.escape(name)}">{options}</select>'


def _tag_controls(name: str, selected: list[str]) -> str:
    selected_set = set(selected)
    return " ".join(
        '<label class="tag">'
        f'<input type="checkbox" name="{html.escape(name)}" value="{html.escape(tag)}"'
        f'{" checked" if tag in selected_set else ""}>{html.escape(tag)}</label>'
        for tag in sorted(ISSUE_TAGS)
    )


def render_page(payload: dict[str, Any], index: int, token: str, notice: str = "") -> str:
    sessions = [row for row in payload.get("sessions") or [] if isinstance(row, dict)]
    if not sessions:
        raise ValueError("annotation file contains no sessions")
    index = max(0, min(index, len(sessions) - 1))
    session = sessions[index]
    completed = sum(_session_complete(row) for row in sessions)
    required = int(payload.get("required_min_sessions") or 50)
    incomplete = [i for i, row in enumerate(sessions) if not _session_complete(row)]
    next_incomplete = next((i for i in incomplete if i > index), incomplete[0] if incomplete else index)
    context_path = Path(str(session.get("full_context_markdown") or ""))
    root = Path(str(payload.get("_review_root") or "."))
    context_file = context_path if context_path.is_absolute() else root / context_path
    try:
        context = context_file.read_text(encoding="utf-8")
    except OSError as exc:
        context = f"无法读取完整上下文：{exc}"
    rendered_context = render_markdown_safe(strip_embedded_w2_json(context))

    episode_blocks = []
    for episode_index, episode in enumerate(session.get("episodes") or []):
        prefix = f"episode:{episode_index}:"
        snapshot = episode.get("w7_snapshot") or {}
        judgments = "".join(
            '<tr><td>' + html.escape(field) + '</td><td>'
            + _select_boolean(prefix + field, episode.get(field)) + '</td></tr>'
            for field in BOOLEAN_FIELDS
        )
        episode_blocks.append(f"""
        <section class="episode">
          <h3>Episode {episode_index + 1}: <code>{html.escape(str(episode.get('episode_id') or ''))}</code></h3>
          <pre>{html.escape(json.dumps(snapshot, ensure_ascii=False, indent=2))}</pre>
          <table><tbody>{judgments}</tbody></table>
          <p><b>问题标签</b><br>{_tag_controls(prefix + 'issue_tags', list(episode.get('issue_tags') or []))}</p>
          <p><b>修正后的 fault focus（如需要）</b><br>{_input(prefix + 'corrected_fault_focus', episode.get('corrected_fault_focus'), size=100)}</p>
          <details><summary>结构化正确答案（有“不正确”字段时尽量填写）</summary>
          <p><b>episode scope</b><br>{_input(prefix + 'corrected_episode_scope', episode.get('corrected_episode_scope'), size=60)}</p>
          <p><b>拆分后的 case items</b>（每行一个，或填写 JSON）<br><textarea name="{prefix}corrected_case_items" rows="3">{html.escape(str(episode.get('corrected_case_items') or ''))}</textarea></p>
          <p><b>resolution status</b>（verified / pending / ineffective / unknown）<br>{_input(prefix + 'corrected_resolution_status', episode.get('corrected_resolution_status'), size=60)}</p>
          <p><b>resolution evidence message IDs</b>（逗号分隔）<br>{_input(prefix + 'corrected_resolution_evidence_message_ids', episode.get('corrected_resolution_evidence_message_ids'), size=100)}</p>
          <p><b>trace group label</b><br>{_input(prefix + 'corrected_trace_group_id', episode.get('corrected_trace_group_id'), size=60)}</p>
          <p><b>trace phase index / count</b><br>
          {_input(prefix + 'corrected_trace_phase_index', episode.get('corrected_trace_phase_index'), size=8)} /
          {_input(prefix + 'corrected_trace_phase_count', episode.get('corrected_trace_phase_count'), size=8)}</p>
          <p><b>W2 readiness 正确值</b><br>{_select_boolean(prefix + 'corrected_w2_readiness', episode.get('corrected_w2_readiness'))}</p>
          </details>
          <p><b>Episode 说明</b><br><textarea name="{prefix}notes" rows="3">{html.escape(str(episode.get('notes') or ''))}</textarea></p>
        </section>""")

    verdict_options = '<option value="">未判断</option>' + "".join(
        f'<option value="{value}"{" selected" if session.get("session_verdict") == value else ""}>{value}</option>'
        for value in sorted(SESSION_VERDICTS)
    )
    query_prev = urlencode({"index": max(0, index - 1)})
    query_next = urlencode({"index": min(len(sessions) - 1, index + 1)})
    query_incomplete = urlencode({"index": next_incomplete})
    notice_html = f'<div class="notice">{html.escape(notice)}</div>' if notice else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>W7 人工审核</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#202124}}
header{{position:sticky;top:0;background:#172033;color:white;padding:12px 24px;z-index:2}}
main{{max-width:1500px;margin:auto;padding:18px}} .grid{{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(520px,1fr);gap:18px}}
.card,.episode{{background:white;border:1px solid #d8dce3;border-radius:8px;padding:16px;margin-bottom:14px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:12px;border-radius:5px;max-height:62vh;overflow:auto}}
textarea{{width:100%;box-sizing:border-box}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #dfe2e6;padding:6px;text-align:left;vertical-align:top}}
.markdown-body{{overflow-wrap:anywhere}} .markdown-body h1{{font-size:1.5rem}} .markdown-body h2{{font-size:1.3rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
.markdown-body h3{{font-size:1.1rem}} .markdown-body code{{background:#f1f3f5;padding:.1em .3em;border-radius:3px}}
.markdown-body pre code{{background:transparent;padding:0}} .markdown-body details{{border:1px solid #dfe2e6;border-radius:5px;padding:8px;margin:10px 0}}
.markdown-body summary{{cursor:pointer;font-weight:600}} .markdown-body blockquote{{border-left:4px solid #ccd2da;margin-left:0;padding-left:12px;color:#555}}
.tag{{display:inline-block;margin:3px 8px 3px 0}} nav a,button{{margin-right:8px;padding:7px 11px}} .notice{{background:#e7f6e9;padding:9px;margin-bottom:12px}}
code{{overflow-wrap:anywhere}} @media(max-width:1050px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><b>W7 人工审核</b>　已完成 {completed}/{required}（模板 {len(sessions)} 条）　当前 {index + 1}/{len(sessions)}</header>
<main>{notice_html}
<nav><a href="/?{query_prev}">上一条</a><a href="/?{query_next}">下一条</a><a href="/?{query_incomplete}">下一条未完成</a></nav>
<h2><code>{html.escape(str(session.get('thread_id') or ''))}</code></h2>
<p><b>审核优先级：</b>{int(session.get('review_priority_score') or 0)}；
依据：{html.escape(', '.join(str(item) for item in session.get('review_priority_reasons') or []) or '常规风险抽样')}；
弱 Trace 建议：{int(session.get('weak_trace_link_candidate_count') or 0)}；
已接受的强/中连接：{int(session.get('accepted_trace_link_count') or 0)}。
优先级仅用于排序，不代表判定结论。</p>
<div class="grid"><div class="card"><h3>完整上下文（必须阅读）</h3>
<p><a href="/context-json?index={index}" target="_blank" rel="noopener">按需打开完整机器 JSON / W2 input</a></p>
<article class="markdown-body">{rendered_context}</article></div>
<form method="post" action="/save" class="card">
<input type="hidden" name="csrf" value="{html.escape(token)}"><input type="hidden" name="thread_id" value="{html.escape(str(session.get('thread_id') or ''))}"><input type="hidden" name="index" value="{index}">
<p><b>审核人</b><br>{_input('reviewer', session.get('reviewer'))}</p>
<p><b>审核时间</b>（留空时在提交 verdict 后自动记录）<br>{_input('reviewed_at', session.get('reviewed_at'))}</p>
<p><b>Session verdict</b><br><select name="session_verdict">{verdict_options}</select></p>
<p><b>Session 问题标签</b><br>{_tag_controls('session_issue_tags', list(session.get('session_issue_tags') or []))}</p>
<p><b>Session 说明</b><br><textarea name="session_notes" rows="3">{html.escape(str(session.get('session_notes') or ''))}</textarea></p>
{''.join(episode_blocks)}
<button name="action" value="stay">保存草稿</button><button name="action" value="next">保存并进入下一条未完成</button>
</form></div></main></body></html>"""


def serve(annotations: Path, host: str, port: int) -> None:
    token = secrets.token_urlsafe(24)

    class ReviewHandler(BaseHTTPRequestHandler):
        def _load(self) -> dict[str, Any]:
            payload = json.loads(annotations.read_text(encoding="utf-8"))
            payload["_review_root"] = str(annotations.parent)
            return payload

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/context-json":
                try:
                    query = parse_qs(parsed.query)
                    index = int((query.get("index") or ["0"])[-1])
                    payload = self._load()
                    sessions = [row for row in payload.get("sessions") or [] if isinstance(row, dict)]
                    if not 0 <= index < len(sessions):
                        raise ValueError("context index out of range")
                    root = Path(str(payload.get("_review_root") or ".")).resolve()
                    context_path = Path(str(sessions[index].get("full_context_json") or ""))
                    context_file = (context_path if context_path.is_absolute() else root / context_path).resolve()
                    if not context_file.is_relative_to(root):
                        raise ValueError("context path escapes review root")
                    body = context_file.read_bytes()
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    self.send_error(400, str(exc))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path != "/":
                self.send_error(404)
                return
            query = parse_qs(parsed.query)
            try:
                index = int((query.get("index") or ["0"])[-1])
                body = render_page(self._load(), index, token).encode("utf-8")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(500, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 2_000_000:
                    raise ValueError("form is too large")
                form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                if _one(form, "csrf") != token:
                    self.send_error(403, "invalid CSRF token")
                    return
                payload = self._load()
                payload.pop("_review_root", None)
                thread_id = _one(form, "thread_id")
                apply_form_update(payload, thread_id, form)
                atomic_write_json(annotations, payload)
                current = int(_one(form, "index") or 0)
                if _one(form, "action") == "next":
                    sessions = payload.get("sessions") or []
                    incomplete = [i for i, row in enumerate(sessions) if not _session_complete(row)]
                    current = next((i for i in incomplete if i > current), incomplete[0] if incomplete else current)
                location = "/?" + urlencode({"index": current})
                self.send_response(303)
                self.send_header("Location", location)
                self.end_headers()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[w7-review] {self.address_string()} {format % args}")

    server = HTTPServer((host, port), ReviewHandler)
    print(f"W7 review workbench: http://{host}:{server.server_port}/")
    print(f"Annotations: {annotations}")
    print("Press Ctrl-C to stop; progress is saved after every form submission.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.annotations.exists():
        parser.error(f"annotation file does not exist: {args.annotations}; run init first")
    serve(args.annotations, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
