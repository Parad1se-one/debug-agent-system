from debug_agent_system.eval.read_side.fae_report_benchmark import (
    MIN_CASE_COUNT,
    build_dataset,
    render_markdown,
    validate_dataset,
)


def test_fae_report_benchmark_uses_200_plus_real_reports_without_legacy_repeats():
    dataset = build_dataset()
    report = validate_dataset(dataset)

    assert report["status"] == "passed", report["issues"]
    assert len(dataset["cases"]) >= MIN_CASE_COUNT
    assert dataset["coverage"]["case_count"] == 205
    assert dataset["coverage"]["candidate_score_min"] >= 45
    assert dataset["coverage"]["issue_tag_counts"]

    source_message_ids = set()
    for case in dataset["cases"]:
        source = case["source_input"]
        refs = case["source_refs"]
        quality = case["quality"]
        assert case["source_type"] == "xing_lark_real_fae_report"
        assert quality["query_is_real_fae_source_text"] is True
        assert quality["graph_ingestion_allowed"] is False
        assert quality["legacy_max_similarity"] < 0.82
        assert source["message_id"] not in source_message_ids
        source_message_ids.add(source["message_id"])
        assert source["text"] in case["query"]
        assert refs["reference_followup_message_ids"]
        assert refs["reference_followup_evidence"]
        assert len(refs["reference_followup_message_ids"]) == len(
            refs["reference_followup_evidence"]
        )
        assert all(
            item["message_id"] in refs["reference_followup_message_ids"]
            for item in refs["reference_followup_evidence"]
        )
        assert "KG_v2" not in case["query"]
        assert "遇到“" not in case["query"]
        assert "不得把建议、短暂恢复或群聊结论升级为 verified_fix" in case[
            "answer_gold"
        ]["reference_answer"]


def test_fae_report_markdown_lists_every_query_and_reference_answer():
    dataset = build_dataset()
    markdown = render_markdown(dataset)

    assert markdown.count("\n## fae-report-") == len(dataset["cases"])
    assert markdown.count("\n**Query**\n") == len(dataset["cases"])
    assert markdown.count("\n**参考答案（后续 FAE 证据卡）**\n") == len(
        dataset["cases"]
    )
