from debug_agent_system.eval.read_side.document_qa_benchmark import (
    LEGACY_SIMILARITY_LIMIT,
    MIN_CASE_COUNT,
    build_dataset,
    render_markdown,
    validate_dataset,
)


def test_document_qa_pilot_uses_original_section_text_for_answers():
    dataset = build_dataset()
    report = validate_dataset(dataset)

    assert report["status"] == "passed", report["issues"]
    assert len(dataset["cases"]) == 22
    assert len(dataset["cases"]) >= MIN_CASE_COUNT
    assert dataset["coverage"]["unique_manual_card_count"] == 22
    assert dataset["coverage"]["unique_source_section_count"] == 20
    assert (
        dataset["coverage"]["legacy_similarity_max"]
        < LEGACY_SIMILARITY_LIMIT
    )

    cards = set()
    queries = set()
    for case in dataset["cases"]:
        refs = case["source_refs"]
        quality = case["quality"]
        answer_gold = case["answer_gold"]
        answer = answer_gold["reference_answer"]
        assert (
            case["source_type"]
            == "original_sop_document_section_multimodal"
        )
        assert refs["manual_card_status"] == "approved_for_phase1_build"
        assert refs["manual_card_role"] == (
            "section_routing_and_source_media_binding"
        )
        assert refs["canonical_section_id"]
        assert refs["source_heading_path"]
        assert refs["source_fragment_sha256"]
        assert refs["manual_card_path"] not in cards
        cards.add(refs["manual_card_path"])
        assert case["query"] not in queries
        queries.add(case["query"])
        assert case["query"].endswith("？")
        assert "遇到“" not in case["query"]
        assert "根据批准资料应确认" not in case["query"]
        assert quality["query_manual_curated"] is True
        assert quality["answer_source_section_grounded"] is True
        assert quality["source_media_preserved"] is True
        assert quality["answer_has_non_source_claims"] is False
        assert quality["independent_expert_gold"] is False
        assert quality["graph_ingestion_allowed"] is False
        assert quality["legacy_max_similarity"] < LEGACY_SIMILARITY_LIMIT
        assert answer_gold["answer_mode"] == (
            "extractive_source_text_with_original_media"
        )
        assert answer_gold["evidence_excerpts"]
        assert answer_gold["card_action_text_used_in_answer"] is False
        assert len(answer_gold["source_images"]) >= refs["source_image_count"]
        assert "执行前需要确认：" not in answer
        assert "验证与边界：" not in answer
        assert "不代表现场已经执行" not in answer
        assert "verified_fix" not in answer


def test_doc_qa_001_preserves_image_and_attachment_display_name():
    dataset = build_dataset()
    case = dataset["cases"][0]
    answer_gold = case["answer_gold"]
    filename = "useraccess_restore_使用管理员身份运行.bat"

    assert case["case_id"] == "doc-qa-001"
    assert filename in answer_gold["reference_answer"]
    assert len(answer_gold["source_images"]) == 1
    assert len(answer_gold["source_attachments"]) == 1
    assert (
        answer_gold["source_attachments"][0]["display_name"]
        == filename
    )

    markdown = render_markdown(dataset)
    assert "![创建模板失败：网络请求失败/超时]" in markdown
    assert f"[{filename}]" in markdown


def test_document_qa_markdown_contains_every_query_and_answer():
    dataset = build_dataset()
    markdown = render_markdown(dataset)

    assert markdown.count("\n## doc-qa-") == len(dataset["cases"])
    assert markdown.count("\n**Query**\n") == len(dataset["cases"])
    assert markdown.count("\n**参考答案**\n") == len(dataset["cases"])
