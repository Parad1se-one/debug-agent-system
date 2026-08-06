"""Render concise per-case Markdown views for frozen goldcase-001--010."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _text(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|")


def render(payload: dict[str, Any]) -> str:
    gold = payload["gold"]
    family = gold["family"]
    variant = gold["variant"]
    lines = [
        f"# {payload['case_id']}：{family['label']}",
        "",
        f"- 状态：`{payload.get('status')}`",
        f"- Source episode：`{payload.get('source_episode_id')}`",
        f"- Family：`{family.get('label')}`",
        f"- Variant：`{variant.get('label')}`",
        f"- 症状/范围：{_text(variant.get('summary'))}",
        "",
        "## 原子动作与结果",
        "",
        "| # | role | 动作 | 已标结果 |",
        "|---:|---|---|---|",
    ]
    outcomes: dict[str, list[str]] = {}
    for item in gold.get("outcomes") or []:
        outcomes.setdefault(str(item.get("action_label") or ""), []).append(str(item.get("outcome_type") or ""))
    for index, action in enumerate(gold.get("actions") or [], start=1):
        label = str(action.get("label") or "")
        lines.append(
            f"| {index} | `{action.get('action_role')}` | {_text(label)} | "
            f"{', '.join(f'`{value}`' for value in outcomes.get(label, [])) or '—'} |"
        )
    lines.extend(["", "## Trace", "", _text((gold.get("trace") or {}).get("summary")), "", "## 不确定项", ""])
    lines.extend(f"- {_text(item)}" for item in gold.get("uncertainties") or [])
    lines.extend(["", "## 证据锚点", ""])
    lines.extend(f"- `{_text(key)}`：{_text(value)}" for key, value in (payload.get("evidence_anchor_map") or {}).items())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-gold-v1-annotations")
    parser.add_argument("--root", default="data/annotations/goldcases/gold-v1")
    parser.add_argument("--out", default="data/annotations/goldcases/gold-v1/reviews")
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.glob("goldcase-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = render(payload)
        target = out / f"{path.stem}.md"
        target.write_text(body, encoding="utf-8")
        rows.append((path.stem, payload["gold"]["family"]["label"], target.name))
    index = [
        "# gold-v1 人工标注审阅索引",
        "",
        "| Case | Family | 文档 |",
        "|---|---|---|",
        *(f"| {case_id} | {family} | [{file}]({file}) |" for case_id, family, file in rows),
        "",
        "> 结构化 JSON 与 `gold-v1.manifest.json` 才是冻结 Ground Truth；本目录 Markdown 由 JSON 自动渲染。",
        "",
    ]
    (out / "README.md").write_text("\n".join(index), encoding="utf-8")
    print(json.dumps({"documents": len(rows), "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
