from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sqlite3

from debug_agent_system.core.config import load_config
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.runtime.system import DebugAgentSystem


KG_ROOT = Path("data/kg_v2")
SOURCE_PATH = Path(
    "data/raw/aoi_debug_agent_sources/进板失败SOP--20250521.docx"
)
DOCUMENT_ID = (
    "knowledge-document:data-raw-aoi_debug_agent_sources-sop--20250521.docx:"
    "b2ba1827c-1d4b75be82"
)
VARIANT_ID = "variant:0338d170a1d3"


def _section_code(item: dict) -> str:
    value = str((item.get("source_offsets") or [""])[0])
    return value.split(":sec:", 1)[1] if ":sec:" in value else ""


def test_boarding_failure_source_is_a_complete_approved_document_graph() -> None:
    store = JsonKGV2Store(KG_ROOT)
    document = store.object_index("KnowledgeDocument")[DOCUMENT_ID]
    sections = [
        item
        for item in store.objects_by_type["KnowledgeSection"]
        if item.get("document_id") == DOCUMENT_ID
    ]
    section_ids = {str(item["section_id"]) for item in sections}
    steps = [
        item
        for item in store.objects_by_type["ProcedureStep"]
        if str(item.get("section_id") or "") in section_ids
    ]
    media = [
        item
        for item in store.objects_by_type["MediaAsset"]
        if DOCUMENT_ID in (item.get("document_ids") or [])
    ]

    assert document["approved"] is True
    assert document["source_path"] == str(SOURCE_PATH)
    assert document["content_hash"] == hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    assert {_section_code(item) for item in sections} == {
        "intro", "1", "2", "3", "3.1", "3.2", "3.2.1", "3.3", "3.4", "3.5"
    }
    assert len(steps) == 5
    assert len(media) == 7
    assert all(item["section_ids"] for item in media)
    assert all(item["asset_path"] and Path(item["asset_path"]).is_file() for item in media)


def test_boarding_failure_formal_evidence_actions_and_sag_are_published() -> None:
    store = JsonKGV2Store(KG_ROOT)
    evidence = store.object_index("EvidenceItem")["evidence:ddccf90609cc"]
    source_case = store.object_index("SourceCase")["case:5e0c0e1c9661"]
    actions = [
        item
        for item in store.objects_by_type["DiagnosticAction"]
        if item.get("variant_id") == VARIANT_ID
    ]

    assert evidence["payload_ref"] == str(SOURCE_PATH)
    assert not evidence["summary"].endswith("…")
    assert all(f"3.{number}" in evidence["summary"] for number in range(1, 6))
    assert source_case["approved"] is True
    assert source_case["source_ref"] == str(SOURCE_PATH)
    assert len(actions) == 5
    assert sum(len(item.get("curated_image_refs") or []) for item in actions) == 7
    assert next(item for item in actions if item["stage"] == "3.5")[
        "safety_level"
    ] == "human_confirmation"

    with sqlite3.connect("data/kg_v2_sag/debug_agent_v2.sqlite") as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, media_refs_json
            FROM source_chunks
            WHERE chunk_id LIKE 'chunk:source:%'
              AND source_offsets_json LIKE ?
            """,
            ("%进板失败SOP--20250521.docx%",),
        ).fetchall()
    assert len(rows) == 5


def test_boarding_failure_real_query_renders_full_3_1_to_3_5_and_images(
    tmp_path: Path,
) -> None:
    config = load_config("config/debug_agent_system.yaml")
    config.session_store = tmp_path / "sessions"
    out = DebugAgentSystem(config).start({
        "query": "板子到达进板口，皮带不转，导致板子停在入口，应该从哪里开始排查？",
        "interactive": False,
        "session": {"session_id": "boarding-failure-full-document"},
    })
    answer = out["answer"]

    assert out["family_id"] == "family:bc03a0555f1c"
    assert out["variant_id"] == VARIANT_ID
    assert len(answer) > 3000
    assert answer.count("![") == 7
    assert answer.count("图片说明：") == 7
    assert "[图片：Drawing" not in answer
    assert not re.search(r"(?m)^\s*\d+\.\s*图\d+\s*$", answer)
    for caption in (
        "图1：进板传感器指示灯及触发状态",
        "图2：进板传感器灵敏度调节位置",
        "图3：移除板子后的未触发状态",
        "图4：工厂软件所在路径",
        "图5：工厂软件皮带正转/反转操作",
        "图6：工厂软件进板 IO 信号检查",
        "图8：设备后部皮带电机线位置",
    ):
        assert answer.count(caption) == 2
    assert "3.1 检查出板口板子是否已出板" in answer
    assert "3.2.1 第一代产品" in answer
    assert "3.3 检查皮带运转情况" in answer
    assert "打开工厂软件时，运控和主程序一定要退掉" in answer
    assert "3.4 检查IO点位信号" in answer
    assert "若两个信号相反" in answer
    assert "3.5 检查皮带电机" in answer
    assert "设备后部" in answer
    assert out["metadata"]["answer_coverage"]["complete"] is True
