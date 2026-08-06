#!/usr/bin/env python3
"""Build a self-contained HTML copy of the typical-query Markdown report.

Images and downloadable attachments referenced by the Markdown are converted
to data URIs. Images are resized and encoded as WebP when Pillow is available;
otherwise their original bytes and MIME types are embedded.

Markdown rendering requires Python-Markdown:

    python -m pip install Markdown Pillow
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


DEFAULT_SOURCE = Path(
    "data/results/typical_query_report/KG_v2读侧典型Query实测结果.md"
)
DEFAULT_OUTPUT = Path(
    "data/results/typical_query_report/"
    "KG_v2读侧典型Query实测结果_图片内嵌版.html"
)


def _import_markdown():
    try:
        import markdown  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "缺少 Python-Markdown。请先执行："
            "python -m pip install Markdown Pillow"
        ) from exc
    return markdown


def _import_pillow():
    try:
        from PIL import Image, ImageOps  # type: ignore
    except ImportError:
        return None, None
    return Image, ImageOps


def _local_path(raw_url: str, source_dir: Path) -> Path | None:
    parsed = urlsplit(html.unescape(raw_url))
    if parsed.scheme or parsed.netloc or raw_url.startswith(("#", "data:")):
        return None
    value = unquote(parsed.path)
    if not value:
        return None
    return (source_dir / value).resolve()


def _image_data_uri(
    path: Path,
    *,
    max_edge: int,
    webp_quality: int,
    image_module,
    image_ops_module,
) -> tuple[str, str]:
    if image_module is not None and image_ops_module is not None:
        try:
            with image_module.open(path) as image:
                image = image_ops_module.exif_transpose(image)
                image.thumbnail((max_edge, max_edge), image_module.Resampling.LANCZOS)
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert(
                        "RGBA" if "transparency" in image.info else "RGB"
                    )
                output = io.BytesIO()
                image.save(
                    output,
                    format="WEBP",
                    quality=webp_quality,
                    method=6,
                )
                payload = base64.b64encode(output.getvalue()).decode("ascii")
                return f"data:image/webp;base64,{payload}", "image/webp"
        except Exception:
            pass

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}", mime


def _attachment_data_uri(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}", mime


def _replace_html_assets(
    rendered: str,
    *,
    source_dir: Path,
    max_edge: int,
    webp_quality: int,
) -> tuple[str, int, int, int]:
    image_module, image_ops_module = _import_pillow()
    image_cache: dict[Path, tuple[str, str]] = {}
    attachment_cache: dict[Path, tuple[str, str]] = {}
    image_occurrences = 0

    image_pattern = re.compile(
        r"<img(?P<before>[^>]*?)\s+src=(?P<quote>[\"'])(?P<src>.*?)(?P=quote)"
        r"(?P<after>[^>]*)>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_image(match: re.Match[str]) -> str:
        nonlocal image_occurrences
        path = _local_path(match.group("src"), source_dir)
        if path is None:
            return match.group(0)
        if not path.is_file():
            raise FileNotFoundError(f"Markdown 引用的图片不存在：{path}")
        cached = image_cache.get(path)
        if cached is None:
            cached = _image_data_uri(
                path,
                max_edge=max_edge,
                webp_quality=webp_quality,
                image_module=image_module,
                image_ops_module=image_ops_module,
            )
            image_cache[path] = cached
        data_uri, _mime = cached
        image_occurrences += 1
        source_path = html.escape(match.group("src"), quote=True)
        attributes = match.group("before") + match.group("after")
        alt_match = re.search(
            r"\balt=([\"'])(.*?)\1",
            attributes,
            flags=re.IGNORECASE | re.DOTALL,
        )
        alt = alt_match.group(2) if alt_match else "源文档图片"
        return (
            f'<img src="{data_uri}" data-source-path="{source_path}" '
            f'loading="lazy" decoding="async" alt="{html.escape(alt, quote=True)}">'
        )

    rendered = image_pattern.sub(replace_image, rendered)

    link_pattern = re.compile(
        r"<a(?P<before>[^>]*?)\s+href=(?P<quote>[\"'])(?P<href>.*?)(?P=quote)"
        r"(?P<after>[^>]*)>(?P<label>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_link(match: re.Match[str]) -> str:
        path = _local_path(match.group("href"), source_dir)
        if path is None:
            return match.group(0)
        if not path.is_file():
            raise FileNotFoundError(f"Markdown 引用的附件不存在：{path}")
        cached = attachment_cache.get(path)
        if cached is None:
            cached = _attachment_data_uri(path)
            attachment_cache[path] = cached
        data_uri, _mime = cached
        source_path = html.escape(match.group("href"), quote=True)
        filename = html.escape(path.name, quote=True)
        return (
            f'<a href="{data_uri}" download="{filename}" '
            f'data-source-path="{source_path}">{match.group("label")}</a>'
        )

    rendered = link_pattern.sub(replace_link, rendered)
    return rendered, len(image_cache), image_occurrences, len(attachment_cache)


def _extract_title(markdown_text: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown_text)
    return match.group(1) if match else "KG_v2 读侧典型 Query 实测结果"


def _page(
    *,
    title: str,
    body: str,
    image_count: int,
    image_occurrences: int,
    attachment_count: int,
    optimized: bool,
    max_edge: int,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    optimization_note = (
        f"图片展示副本统一编码为 WebP（最长边不超过 {max_edge}px）"
        if optimized
        else "当前环境没有 Pillow，图片按原始格式内嵌"
    )
    meta = (
        f"共嵌入 {image_count} 个唯一图片资源（正文中出现 {image_occurrences} 次）"
        f"和 {attachment_count} 个附件；{optimization_note}。"
        f"生成时间：{generated_at}。该文件不依赖外部图片或附件目录。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme:light; --ink:#17202a; --muted:#5f6b76; --line:#d9e0e7; --accent:#185abc; --panel:#f7f9fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#eef2f6; font:15px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:24px auto; padding:36px 48px 64px; background:#fff; box-shadow:0 6px 28px rgba(27,39,51,.10); border-radius:12px; }}
h1 {{ font-size:30px; margin-top:0; border-bottom:2px solid var(--accent); padding-bottom:12px; }}
h2 {{ margin-top:34px; padding-bottom:7px; border-bottom:1px solid var(--line); }}
h3 {{ margin-top:26px; color:#243b53; }}
a {{ color:var(--accent); overflow-wrap:anywhere; }}
table {{ border-collapse:collapse; width:100%; display:block; overflow-x:auto; }}
th,td {{ border:1px solid var(--line); padding:8px 10px; vertical-align:top; }}
th {{ background:var(--panel); }}
code {{ background:#edf2f7; padding:.12em .35em; border-radius:4px; }}
pre {{ background:#17202a; color:#f8fafc; padding:14px; border-radius:8px; overflow:auto; }}
pre code {{ background:transparent; padding:0; }}
img {{ display:block; max-width:100%; height:auto; margin:12px auto 24px; border:1px solid var(--line); border-radius:6px; box-shadow:0 2px 10px rgba(27,39,51,.08); }}
details {{ margin:14px 0 28px; border:1px solid var(--line); border-radius:8px; padding:0 18px 18px; }}
summary {{ cursor:pointer; font-weight:700; padding:14px 0; color:#243b53; }}
.meta {{ margin:-10px 0 28px; padding:12px 14px; background:#eef6ff; border-left:4px solid var(--accent); color:var(--muted); }}
.meta strong {{ color:var(--ink); }}
li {{ margin:.32em 0; }}
@media (max-width:720px) {{ main {{ width:100%; margin:0; padding:24px 18px 48px; border-radius:0; }} h1 {{ font-size:25px; }} }}
@media print {{ body {{ background:#fff; }} main {{ box-shadow:none; width:100%; margin:0; padding:0; }} details {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<main>
<div class="meta"><strong>图片与附件内嵌版</strong>：{html.escape(meta)}</div>
{body}
</main>
</body>
</html>
"""


def build(
    source: Path,
    output: Path,
    *,
    max_edge: int,
    webp_quality: int,
) -> tuple[int, int, int]:
    markdown_module = _import_markdown()
    source = source.resolve()
    output = output.resolve()
    markdown_text = source.read_text(encoding="utf-8")
    markdown_text = markdown_text.replace(
        "<details open>",
        '<details open markdown="1">',
    )
    rendered = markdown_module.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    rendered, image_count, image_occurrences, attachment_count = (
        _replace_html_assets(
            rendered,
            source_dir=source.parent,
            max_edge=max_edge,
            webp_quality=webp_quality,
        )
    )
    image_module, _image_ops_module = _import_pillow()
    document = _page(
        title=_extract_title(markdown_text),
        body=rendered,
        image_count=image_count,
        image_occurrences=image_occurrences,
        attachment_count=attachment_count,
        optimized=image_module is not None,
        max_edge=max_edge,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return image_count, image_occurrences, attachment_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-edge", type=int, default=2400)
    parser.add_argument("--webp-quality", type=int, default=82)
    args = parser.parse_args()
    images, occurrences, attachments = build(
        args.source,
        args.output,
        max_edge=args.max_edge,
        webp_quality=args.webp_quality,
    )
    print(
        f"已生成 {args.output}："
        f"{images} 个唯一图片（{occurrences} 次引用），{attachments} 个附件。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
