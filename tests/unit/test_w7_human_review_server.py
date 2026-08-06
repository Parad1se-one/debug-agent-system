from __future__ import annotations

import json
from datetime import UTC, datetime

from debug_agent_system.eval.write_side.w7_human_review import BOOLEAN_FIELDS, build_template
from debug_agent_system.eval.write_side.w7_human_review_server import (
    apply_form_update,
    atomic_write_json,
    render_markdown_safe,
    render_page,
    strip_embedded_w2_json,
)


def _payload() -> dict:
    return build_template({
        "cases": [{
            "thread_id": "thread:1",
            "full_context_markdown": "context.md",
            "after": [{"episode_id": "episode:1", "fault_focus": "camera failed"}],
        }],
    })


def test_apply_form_update_records_a_complete_human_annotation() -> None:
    payload = _payload()
    form = {
        "reviewer": ["alice"],
        "reviewed_at": [""],
        "session_verdict": ["pass"],
        "session_issue_tags": [],
        "session_notes": [""],
        **{f"episode:0:{field}": ["true"] for field in BOOLEAN_FIELDS},
        "episode:0:issue_tags": [],
        "episode:0:corrected_fault_focus": [""],
        "episode:0:corrected_episode_scope": ["single_fault"],
        "episode:0:corrected_case_items": ["相机拍摄失败"],
        "episode:0:corrected_resolution_status": ["pending"],
        "episode:0:corrected_resolution_evidence_message_ids": ["m2,m3"],
        "episode:0:corrected_trace_group_id": ["trace:A"],
        "episode:0:corrected_trace_phase_index": ["1"],
        "episode:0:corrected_trace_phase_count": ["2"],
        "episode:0:corrected_w2_readiness": ["false"],
        "episode:0:notes": [""],
    }
    now = datetime(2026, 7, 17, tzinfo=UTC)

    updated = apply_form_update(payload, "thread:1", form, now=now)

    session = updated["sessions"][0]
    assert session["reviewer"] == "alice"
    assert session["reviewed_at"] == "2026-07-17T00:00:00+00:00"
    assert session["session_verdict"] == "pass"
    assert all(session["episodes"][0][field] is True for field in BOOLEAN_FIELDS)
    assert session["episodes"][0]["corrected_trace_group_id"] == "trace:A"
    assert session["episodes"][0]["corrected_trace_phase_index"] == 1
    assert session["episodes"][0]["corrected_trace_phase_count"] == 2
    assert session["episodes"][0]["corrected_w2_readiness"] is False


def test_atomic_write_and_page_render_keep_context_and_existing_values(tmp_path) -> None:
    payload = _payload()
    context = tmp_path / "context.md"
    context.write_text("<script>unsafe</script> full evidence", encoding="utf-8")
    payload["_review_root"] = str(tmp_path)
    payload["sessions"][0]["reviewer"] = "alice"
    page = render_page(payload, 0, "token")

    assert "&lt;script&gt;unsafe&lt;/script&gt; full evidence" in page
    assert 'value="alice"' in page
    assert 'value="token"' in page

    payload.pop("_review_root")
    output = tmp_path / "annotations.json"
    atomic_write_json(output, payload)
    assert json.loads(output.read_text(encoding="utf-8"))["sessions"][0]["thread_id"] == "thread:1"


def test_markdown_renderer_formats_review_content_without_allowing_raw_html() -> None:
    rendered = render_markdown_safe("""# 标题

| 字段 | 结果 |
|---|---|
| fault | `camera` |

<details><summary>展开 JSON</summary>

```json
{"ok": true}
```

</details>

<script>alert('unsafe')</script>
""")

    assert "<h1>标题</h1>" in rendered
    assert "<table>" in rendered
    assert "<code>camera</code>" in rendered
    assert "<details><summary>展开 JSON</summary>" in rendered
    assert '<code class="language-json">' in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;alert('unsafe')&lt;/script&gt;" in rendered


def test_main_page_strips_generated_w2_json_but_keeps_other_details() -> None:
    source = """# Context

<details><summary>展开完整 W2 input JSON</summary>

```json
{"very_large": "payload"}
```

</details>

<details><summary>人工说明</summary>

keep this

</details>
"""

    compact = strip_embedded_w2_json(source)

    assert "very_large" not in compact
    assert "按需链接" in compact
    assert "人工说明" in compact
    assert "keep this" in compact
