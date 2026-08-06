from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.knowledge_v2.source_chunk_builder import rebuild_source_chunks


PROJECT_ROOT = Path(__file__).parents[2]


def _source_chunks() -> tuple[list[dict], dict[str, int]]:
    object_root = PROJECT_ROOT / "data" / "kg_v2" / "objects"
    documents = json.loads((object_root / "knowledge_documents.json").read_text(encoding="utf-8"))
    sections = json.loads((object_root / "knowledge_sections.json").read_text(encoding="utf-8"))
    return rebuild_source_chunks(PROJECT_ROOT, documents, sections)


def test_semantic_chunks_cover_every_section_and_keep_bounded_context() -> None:
    chunks, stats = _source_chunks()

    assert stats["source_document_count"] == 55
    assert stats["source_parse_failure_count"] == 0
    assert stats["source_hash_mismatch_count"] == 0
    assert stats["source_aligned_section_count"] == stats["source_section_count"] == 228
    assert stats["source_directly_aligned_section_count"] == 228
    assert 250 <= len(chunks) <= 400
    assert max(len(chunk["text"]) for chunk in chunks) <= 650
    assert all(
        offset["block_end"] - offset["block_start"] + 1 <= 24
        for chunk in chunks
        for offset in chunk["source_offsets"]
    )


def test_faq_question_and_answer_are_one_chunk_without_next_question() -> None:
    chunks, _stats = _source_chunks()
    faq_chunk = next(chunk for chunk in chunks if "CAD自动对齐失败时如何处理?" in chunk["text"])

    assert "导入CAD后下一步执行编程" in faq_chunk["text"]
    assert "若自动对齐仍无法将所有拼板对齐" in faq_chunk["text"]
    assert "编程时部分器件框的角度" not in faq_chunk["text"]
    assert "heading" in faq_chunk["source_offsets"][0]["block_types"]
    assert faq_chunk["direct_section_ids"]


def test_docx_table_rows_preserve_cell_boundaries() -> None:
    chunks, stats = _source_chunks()
    table_chunk = next(
        chunk for chunk in chunks
        if "制作镜像/备份镜像 | 工程师乙" in chunk["text"]
    )

    assert "名称 | 所有者 | 修改时间 | 创建时间" in table_chunk["text"]
    assert "table_row" in table_chunk["source_offsets"][0]["block_types"]
    assert stats["source_table_chunk_count"] > 0


def test_docx_local_outline_headings_and_embedded_media_are_preserved() -> None:
    chunks, stats = _source_chunks()
    computer_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("source_path") or "").endswith("/电脑卡顿.docx")
    ]

    assert {
        "强制关闭卡死程序：",
        "一键深度清理（每周1次）：",
        "自动清理：",
        "防卡顿黄金法则",
    }.issubset({str(chunk.get("source_label") or "") for chunk in computer_chunks})
    for heading in {
        "强制关闭卡死程序：",
        "一键深度清理（每周1次）：",
        "自动清理：",
        "防卡顿黄金法则",
    }:
        assert any(
            chunk.get("source_label") == heading
            and "heading" in chunk["source_offsets"][0]["block_types"]
            for chunk in computer_chunks
        )

    media_refs = [
        media
        for chunk in computer_chunks
        for media in chunk.get("media_refs") or []
    ]
    image_paths = {
        str(item.get("archive_path") or "")
        for item in media_refs
        if item.get("media_kind") == "image"
    }
    assert image_paths == {"word/media/image1.png", "word/media/image2.png"}
    assert any(
        item.get("media_kind") == "attachment"
        and str(item.get("archive_path") or "").endswith("Microsoft_Excel_Worksheet1.xlsx")
        for item in media_refs
    )
    assert stats["source_image_count"] >= 2
    assert stats["source_attachment_count"] >= 1


def test_unstyled_numbered_solutions_become_semantic_chunks() -> None:
    chunks, _stats = _source_chunks()
    usb_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("source_path") or "").endswith("/USB设备问题解决方案.docx")
    ]

    expected_headings = ["方案一", "方案二", "方案三", "方案四", "方案五"]
    for index, heading in enumerate(expected_headings):
        chunk = next(
            item for item in usb_chunks
            if str(item.get("source_label") or "").startswith(heading)
        )
        assert "heading" in chunk["source_offsets"][0]["block_types"]
        content_blocks = chunk["source_offsets"][0]["content_blocks"]
        list_items = [
            block for block in content_blocks
            if block.get("kind") == "list_item"
        ]
        assert list_items
        assert list_items[0]["list_style"] == "ordered"
        assert list_items[0]["list_marker"] == "1."
        if index + 1 < len(expected_headings):
            assert expected_headings[index + 1] not in chunk["text"]


def test_single_cell_command_tables_stay_with_their_method() -> None:
    chunks, _stats = _source_chunks()
    web_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("source_path") or "").endswith("/网页打不开但微信_飞书能用.docx")
    ]
    method_one = next(
        chunk for chunk in web_chunks
        if str(chunk.get("source_label") or "").startswith("方法一")
    )

    assert "ipconfig /flushdns" in method_one["text"]
    assert "netsh int ip reset" in method_one["text"]
    assert "netsh winsock reset" in method_one["text"]
    assert '成功会显示"重置Winsock"' in method_one["text"]
    assert "方法二" not in method_one["text"]
    assert "code_block" in method_one["source_offsets"][0]["block_types"]


def test_camera_failure_document_keeps_full_outline_and_contextual_images() -> None:
    chunks, _stats = _source_chunks()
    camera_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("source_path") or "").endswith("/检测界面出现拍照失败问题处理.docx")
    ]

    assert len(camera_chunks) >= 20
    labels = {str(chunk.get("source_label") or "") for chunk in camera_chunks}
    for heading in ("2.1.1.1", "2.2.1", "2.5.2", "2.7", "2.9"):
        assert any(label.startswith(heading) for label in labels), heading

    images = [
        media
        for chunk in camera_chunks
        for media in chunk.get("media_refs") or []
        if media.get("media_kind") == "image"
    ]
    assert len({item["content_hash"] for item in images}) >= 40
    assert all(str(item.get("context_label") or "").strip() for item in images)
    assert all(not str(item["context_label"]).startswith("Drawing ") for item in images)


def test_boarding_failure_figures_are_anchored_to_following_captions() -> None:
    chunks, _stats = _source_chunks()
    boarding_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("source_path") or "").endswith(
            "/进板失败SOP--20250521.docx"
        )
    ]
    first_generation = next(
        chunk for chunk in boarding_chunks
        if str(chunk.get("source_label") or "").startswith("3.2.1")
    )
    blocks = first_generation["source_offsets"][0]["content_blocks"]
    images = [block for block in blocks if block.get("kind") == "image"]
    captions = [
        str(block.get("text") or "")
        for block in blocks
        if block.get("kind") == "figure_caption"
    ]

    assert len(images) == 3
    assert all(block.get("media_keys") for block in images)
    assert captions == ["图1", "图2", "图3"]
    assert [
        str(item.get("context_label") or "")
        for item in first_generation["media_refs"]
    ] == ["图1", "图2", "图3"]
