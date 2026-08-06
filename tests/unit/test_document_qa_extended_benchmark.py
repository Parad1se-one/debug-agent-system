from debug_agent_system.eval.read_side.document_qa_extended_benchmark import (
    MIN_TASK_CASE_COUNT,
    build_dataset,
    render_markdown,
    validate_dataset,
)


def test_extended_document_qa_has_task_level_grounded_cases():
    dataset = build_dataset()
    report = validate_dataset(dataset)

    assert report["status"] == "passed", report["issues"]
    assert len(dataset["cases"]) >= MIN_TASK_CASE_COUNT
    assert dataset["coverage"]["document_count"] >= 40
    assert dataset["coverage"]["source_granularity_counts"] == {
        "task": len(dataset["cases"])
    }
    assert dataset["coverage"]["section_reference_count"] > len(
        dataset["cases"]
    )
    assert dataset["coverage"]["image_reference_count"] > 150

    queries = set()
    for case in dataset["cases"]:
        refs = case["source_refs"]
        answer_gold = case["answer_gold"]
        quality = case["quality"]
        assert "异常处理 - 标准操作流程" not in refs["document_title"]
        assert case["query"].endswith("？")
        assert case["query"] not in queries
        assert "文档给出" not in case["query"]
        assert "步骤中" not in case["query"]
        queries.add(case["query"])
        assert case["source_granularity"] == "task"
        assert refs["section_ids"]
        assert answer_gold["reference_answer"]
        assert "…" not in answer_gold["reference_answer"]
        assert answer_gold["generic_governance_text_added"] is False
        assert quality["answer_source_snapshot_grounded"] is True
        assert quality["independent_expert_gold"] is False
        assert quality["graph_ingestion_allowed"] is False


def test_extended_document_qa_markdown_preserves_cases_and_media():
    dataset = build_dataset()
    markdown = render_markdown(dataset)
    case_count = len(dataset["cases"])

    assert markdown.count("\n## ext-doc-qa-") == case_count
    assert markdown.count("\n**Query**\n") == case_count
    assert markdown.count("\n**参考答案**\n") == case_count
    assert markdown.count("![") >= 150


def test_sequential_document_sections_are_merged_into_one_task():
    dataset = build_dataset()
    cases = dataset["cases"]

    disk_management = [
        case for case in cases
        if case["query"] == "如何打开磁盘管理工具？"
    ]
    assert len(disk_management) == 1

    m2_cases = [
        case for case in cases
        if case["source_refs"]["document_title"]
        == "M.2SSD硬盘更换和数据迁移.docx"
    ]
    assert len(m2_cases) == 1
    assert m2_cases[0]["query"] == "如何更换 M.2 SSD 并迁移系统？"
    assert len(m2_cases[0]["source_refs"]["section_ids"]) == 8
    answer = m2_cases[0]["answer_gold"]["reference_answer"]
    assert "准备工作" in answer
    assert "在PE系统中克隆硬盘" in answer
    assert "最终组装与收尾" in answer
