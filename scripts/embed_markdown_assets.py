#!/usr/bin/env python3
"""Build a single-file Markdown report from an asset-linked Markdown source.

The companion self-contained HTML is used as the data-URI source, so image
conversion and attachment encoding stay identical across both deliverables.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


DATA_URI_BY_SOURCE = re.compile(
    r"""(?:src|href)="(?P<data>data:[^"]+)"[^>]*"""
    r'data-source-path="(?P<source>[^"]+)"'
)
MARKDOWN_ASSET = re.compile(
    r"""(?P<prefix>!?\[[^\]]*]\()"""
    r"""(?P<source>\.\./data/kg_v2_sag/assets/[^)\s]+)"""
    r"""(?P<suffix>\))"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = args.source.read_text(encoding="utf-8")
    rendered_html = args.html.read_text(encoding="utf-8")
    data_by_source = {
        html.unescape(match.group("source")): html.unescape(match.group("data"))
        for match in DATA_URI_BY_SOURCE.finditer(rendered_html)
    }

    replaced = 0

    def embed(match: re.Match[str]) -> str:
        nonlocal replaced
        source = match.group("source")
        data_uri = data_by_source.get(source)
        if data_uri is None:
            raise SystemExit(f"missing embedded data for {source}")
        replaced += 1
        return f"{match.group('prefix')}<{data_uri}>{match.group('suffix')}"

    result = MARKDOWN_ASSET.sub(embed, markdown)
    header = (
        "> **单文件图片内嵌版**：图片和附件使用 Base64 `data:` URI 写入本文件，"
        "不依赖外部资源目录。部分在线 Markdown 导入器会出于安全原因过滤 "
        "`data:` URI；如飞书导入后图片缺失，请使用同目录的 HTML 版复制粘贴，"
        "或改用带内嵌图片的 Word 文档导入。\n\n"
    )
    args.output.write_text(header + result, encoding="utf-8")
    print(f"embedded {replaced} Markdown asset references")
    print(f"unique embedded resources: {len(data_by_source)}")
    print(f"output: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
