from __future__ import annotations

from pathlib import Path

from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


KG_ROOT = Path("data/kg_v2")
CAMERA_VARIANT_ID = "variant:505989010b74"


def test_canonical_media_graph_covers_every_document_image_occurrence() -> None:
    store = JsonKGV2Store(KG_ROOT)
    media = store.objects_by_type["MediaAsset"]

    assert len(media) == 294
    assert sum(item["media_kind"] == "image" for item in media) == 293
    assert sum(item["media_kind"] == "attachment" for item in media) == 1
    assert sum(len(item["source_occurrences"]) for item in media) == 298
    assert sum(
        len(item["source_occurrences"])
        for item in media
        if item["media_kind"] == "image"
    ) == 297
    assert len({
        document_id
        for item in media
        for document_id in item["document_ids"]
    }) == 25
    assert all(item["document_ids"] for item in media)
    assert all(item["section_ids"] for item in media)
    assert all(item["source_chunk_ids"] for item in media)
    assert all(
        item["asset_path"] and Path(item["asset_path"]).is_file()
        for item in media
    )


def test_media_graph_has_document_section_step_and_action_links() -> None:
    store = JsonKGV2Store(KG_ROOT)
    media = store.objects_by_type["MediaAsset"]
    relation_keys = {
        (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("relation") or ""),
        )
        for item in store.relations
    }

    for item in media:
        media_id = item["media_id"]
        assert all(
            (document_id, media_id, "has_media") in relation_keys
            for document_id in item["document_ids"]
        )
        assert all(
            (section_id, media_id, "section_media") in relation_keys
            for section_id in item["section_ids"]
        )
        assert all(
            (step_id, media_id, "step_media") in relation_keys
            for step_id in item["procedure_step_ids"]
        )
        assert all(
            (action_id, media_id, "action_media") in relation_keys
            for action_id in item["action_ids"]
        )


def test_every_curated_camera_action_image_is_a_first_class_media_link() -> None:
    store = JsonKGV2Store(KG_ROOT)
    media_ids = {
        str(item.get("media_id") or "")
        for item in store.objects_by_type["MediaAsset"]
    }
    relation_keys = {
        (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("relation") or ""),
        )
        for item in store.relations
    }
    actions = [
        item
        for item in store.objects_by_type["DiagnosticAction"]
        if item.get("variant_id") == CAMERA_VARIANT_ID
    ]
    refs = [
        (str(action["action_id"]), str(ref.get("media_id") or ""))
        for action in actions
        for ref in action.get("curated_image_refs") or []
    ]

    assert len(refs) == 47
    assert len({media_id for _action_id, media_id in refs}) == 47
    assert all(media_id in media_ids for _action_id, media_id in refs)
    assert all(
        (action_id, media_id, "action_media") in relation_keys
        for action_id, media_id in refs
    )
