from __future__ import annotations

import json
from pathlib import Path

import debug_agent_system.kg_raw_codex.pipeline as pipeline_module
from debug_agent_system.kg_raw_codex.coverage import (
    ProcedureVariantRequirement,
    RequiredFacet,
    build_answer_scope,
    build_required_facets,
    verify_answer_draft,
)
from debug_agent_system.kg_raw_codex.pipeline import (
    CodexResponsesAgentRunner,
    CorpusReadTools,
    KGRawCodexPipeline,
)


class FakeResponsesClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict] = []
        self.last_usage: dict = {}
        self.last_request_id = ""
        self.timeout_seconds = 0

    def create(self, body: dict) -> dict:
        self.requests.append(body)
        response = next(self.responses)
        self.last_usage = response.get("usage") or {}
        self.last_request_id = f"request-{len(self.requests)}"
        return response


def test_responses_runner_lets_model_drive_generic_corpus_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "raw"
    kg = tmp_path / "kg"
    workspace = tmp_path / "workspace"
    raw.mkdir()
    kg.mkdir()
    workspace.mkdir()
    (raw / "guide.md").write_text("alpha procedure", encoding="utf-8")
    (kg / "graph.json").write_text('{"name":"alpha"}', encoding="utf-8")
    monkeypatch.setattr(
        pipeline_module,
        "CORPUS_ROOTS",
        {"raw": raw.resolve(), "kg_v2": kg.resolve()},
    )
    final = {
        "schema_version": "debug_agent_system.kg_raw_codex_draft.v5",
        "answer_markdown": "answer",
        "coverage_ledger": [],
        "procedure_variant_ledger": [],
        "files_read": ["data/raw/guide.md"],
    }
    client = FakeResponsesClient([
        {
            "id": "response-1",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 3},
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "encrypted_content": "opaque",
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "id": "call-item-1",
                    "call_id": "call-1",
                    "name": "read_text",
                    "arguments": json.dumps({
                        "path": "data/raw/guide.md",
                        "start_line": 1,
                        "end_line": 20,
                    }),
                    "status": "completed",
                },
            ],
        },
        {
            "id": "response-2",
            "status": "completed",
            "usage": {"input_tokens": 20, "output_tokens": 5},
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": json.dumps(final),
                }],
            }],
        },
    ])
    runner = CodexResponsesAgentRunner(client=client, model="gpt-5.4")
    draft, audit = runner.run(
        prompt="investigate",
        workspace=workspace,
        output_schema={"type": "object"},
        timeout_seconds=60,
    )

    assert draft == final
    assert audit["files_read"] == ["data/raw/guide.md"]
    assert audit["usage"]["total_tokens"] == 38
    assert audit["tool_trace"][0]["name"] == "read_text"
    assert client.requests[0]["store"] is False
    assert client.requests[0]["include"] == ["reasoning.encrypted_content"]
    assert [
        tool["name"] for tool in client.requests[0]["tools"]
    ] == ["list_files", "search_text", "read_text"]
    second_input = client.requests[1]["input"]
    assert any(
        item.get("type") == "function_call_output"
        for item in second_input
    )
    assert not any(
        key in json.dumps(client.requests)
        for key in ("codex exec", "codex_binary", "auth_mode")
    )


def test_corpus_tools_reject_paths_outside_evidence_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "raw"
    kg = tmp_path / "kg"
    workspace = tmp_path / "workspace"
    raw.mkdir()
    kg.mkdir()
    workspace.mkdir()
    (raw / "guide.md").write_text("procedure", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_module,
        "CORPUS_ROOTS",
        {"raw": raw.resolve(), "kg_v2": kg.resolve()},
    )
    tools = CorpusReadTools(workspace)

    result, audit = tools.execute("read_text", {
        "path": "../../etc/passwd",
        "start_line": 1,
        "end_line": 5,
    })

    assert audit["status"] == "error"
    assert "outside_corpus" in result["error"]
    assert not tools.files_read


def test_cli_draft_normalization_is_workspace_and_query_agnostic(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "isolated-corpus"
    extracted = workspace / "data/extracted_docx/raw/manual.docx.md"
    extracted.parent.mkdir(parents=True)
    extracted.write_text(
        "SOURCE_PATH: data/raw/manual.docx\ncontent",
        encoding="utf-8",
    )
    draft = {
        "files_read": [
            f"{workspace}/data/raw/manual.docx",
            f"{workspace}/data/kg_v2/objects/items.json",
            "data/extracted_docx/raw/manual.docx.md",
        ],
        "answer_markdown": (
            f"【来源：{workspace}/data/raw/manual.docx】\n"
            "[沙箱路径](/tmp/kg-raw-corpus-abc/data/raw/带 空格.docx)\n"
            f"![步骤图]( {workspace}/data/raw/media/step.png )\n"
            f"重复：![同一资源]( {workspace}/data/raw/media/step.png )"
        ),
    }

    normalized = pipeline_module._normalize_cli_draft_paths(
        draft,
        workspace=workspace,
    )

    assert normalized["files_read"] == [
        "data/raw/manual.docx",
        "data/kg_v2/objects/items.json",
        "data/raw/manual.docx",
    ]
    assert (
        "【来源：data/raw/manual.docx】"
        in normalized["answer_markdown"]
    )
    assert (
        "[沙箱路径](data/raw/带 空格.docx)"
        in normalized["answer_markdown"]
    )
    assert (
        "![步骤图](data/raw/media/step.png)"
        in normalized["answer_markdown"]
    )
    assert "![同一资源]" not in normalized["answer_markdown"]


def test_draft_finalization_only_adds_audited_covered_sources() -> None:
    draft = {
        "answer_markdown": (
            "结论\n"
            "![图一](data/raw/media/a.png)\n"
            "![重复图](data/raw/media/a.png)\n"
            "`【来源：data/raw/a.md】`"
        ),
        "coverage_ledger": [
            {
                "status": "covered",
                "source_paths": [
                    "data/raw/a.md",
                    "data/raw/not-read.md",
                ],
            },
            {
                "status": "gap",
                "source_paths": ["data/raw/gap.md"],
            },
        ],
    }

    finalized = pipeline_module._finalize_draft_contract(
        draft,
        actual_files_read=["data/raw/a.md"],
    )

    answer = finalized["answer_markdown"]
    assert answer.count("data/raw/media/a.png") == 1
    assert "【来源：data/raw/a.md】" in answer
    assert "`【来源：" not in answer
    assert "data/raw/not-read.md" not in answer
    assert "data/raw/gap.md" not in answer


def test_answer_scope_only_allows_fallback_when_explicitly_requested() -> None:
    direct = build_answer_scope("如何进入某个诊断环境？")
    fallback = build_answer_scope("如果进入该环境仍然失败，下一步怎么办？")

    assert direct.max_fallback_depth == 0
    assert fallback.max_fallback_depth == 1


def test_pipeline_cross_checks_claimed_files_against_tool_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    raw = repo / "data/raw"
    kg = repo / "data/kg_v2"
    raw.mkdir(parents=True)
    kg.mkdir(parents=True)
    (raw / "guide.md").write_text("procedure", encoding="utf-8")
    (kg / "graph.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline_module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        pipeline_module,
        "CORPUS_ROOTS",
        {"raw": raw.resolve(), "kg_v2": kg.resolve()},
    )

    class FakeRunner:
        model = "gpt-5.4"
        runtime_metadata = {
            "engine": "responses_api",
            "agent_loop": "model_directed_function_calls",
        }

        def run(self, **kwargs):
            return ({
                "schema_version": (
                    "debug_agent_system.kg_raw_codex_draft.v5"
                ),
                "answer_markdown": "根据两类资料组织回答。",
                "coverage_ledger": [],
                "procedure_variant_ledger": [],
                "files_read": [
                    "data/raw/guide.md",
                    "data/kg_v2/graph.json",
                ],
            }, {
                "thread_id": "response-2",
                "usage": {"input_tokens": 7, "output_tokens": 2},
                "tool_trace": [],
                "files_read": [
                    "data/raw/guide.md",
                    "data/kg_v2/graph.json",
                ],
            })

    output = repo / "results/answer.json"
    payload = KGRawCodexPipeline(
        runner=FakeRunner(),
        verification_attempts=1,
    ).run("整理现有资料", output)

    assert payload["runtime"]["agent_loop"] == (
        "model_directed_function_calls"
    )
    assert payload["files_read"] == [
        "data/raw/guide.md",
        "data/kg_v2/graph.json",
    ]
    assert payload["terminology_search_contract"][
        "required_search_groups"
    ] == []
    assert payload["terminology_search_audit"]["complete"] is True
    assert payload["terminology_manifest"]["status"] == "missing"
    assert payload["verification"]["passed"] is True
    assert output.is_file()


def test_release_gate_does_not_reject_valid_semantics_by_exact_wording() -> None:
    source = "data/raw/guide.md"
    draft = {
        "answer_markdown": (
            "先进入 Windows 的安全启动环境，再卸载最近的驱动。"
            f"【来源：{source}】"
        ),
        "coverage_ledger": [{
            "facet_id": "operation_object:进入:安全模式排查",
            "label": "进入安全模式排查",
            "kind": "query_task",
            "status": "covered",
            "source_paths": [source],
            "reason": "文档给出进入方式和排查动作。",
        }],
    }
    facet = RequiredFacet(
        facet_id="operation_object:进入:安全模式排查",
        kind="query_task",
        label="进入安全模式排查",
        match_terms=("进入", "安全模式排查"),
    )

    errors = verify_answer_draft(
        draft,
        required_facets=[facet],
        files_read=[source],
        media_exposed=[],
    )

    assert errors == []


def test_scope_keeps_purpose_action_out_of_primary_object() -> None:
    query = "安装软件后出现异常，需要进入安全模式排查时应该怎么进入？"

    facets = build_required_facets(query)
    scope = build_answer_scope(query)

    assert any(
        facet.facet_id == "operation_object:进入:安全模式"
        for facet in facets
    )
    assert not any("安全模式排查" in facet.facet_id for facet in facets)
    assert scope.requested_operations == ("进入",)
    assert scope.allow_system_repair_commands is False
    assert scope.allow_boot_repair_commands is False
    assert not any(
        facet.facet_id == "safety:execution_preconditions"
        for facet in facets
    )


def test_release_gate_rejects_unrequested_downstream_commands() -> None:
    source = "data/raw/guide.md"
    draft = {
        "answer_markdown": (
            "按资料进入目标环境。"
            f"【来源：{source}】\n\n"
            "```cmd\nsfc /scannow\nbootrec /fixmbr\n```\n"
        ),
        "coverage_ledger": [{
            "facet_id": "operation:进入",
            "label": "进入",
            "kind": "query_task",
            "status": "covered",
            "source_paths": [source],
            "reason": "资料给出进入方式。",
        }],
    }
    facet = RequiredFacet(
        facet_id="operation:进入",
        kind="query_task",
        label="进入",
        match_terms=("进入",),
    )

    errors = verify_answer_draft(
        draft,
        required_facets=[facet],
        files_read=[source],
        media_exposed=[],
        answer_scope=build_answer_scope("如何进入安全模式？"),
    )

    assert "out_of_scope_system_repair_command" in errors
    assert "out_of_scope_boot_repair_command" in errors


def test_release_gate_allows_grounded_non_destructive_command() -> None:
    source = "data/raw/power.md"
    draft = {
        "answer_markdown": (
            "按原文关闭休眠。"
            f"【来源：{source}】\n\n```cmd\npowercfg /h off\n```"
        ),
        "coverage_ledger": [{
            "facet_id": "operation:关闭",
            "label": "关闭",
            "kind": "query_task",
            "status": "covered",
            "source_paths": [source],
            "reason": "原文给出非破坏性系统配置命令。",
        }],
    }
    facet = RequiredFacet(
        facet_id="operation:关闭",
        kind="query_task",
        label="关闭",
        match_terms=("关闭",),
    )

    errors = verify_answer_draft(
        draft,
        required_facets=[facet],
        files_read=[source],
        media_exposed=[],
        answer_scope=build_answer_scope("如何关闭休眠？"),
    )

    assert errors == []


def test_parallel_procedure_gate_rejects_silently_dropped_method() -> None:
    source = "data/raw/guide.docx"
    requirements = [
        ProcedureVariantRequirement(source, "方案一"),
        ProcedureVariantRequirement(source, "方案二"),
        ProcedureVariantRequirement(source, "方案三"),
    ]
    draft = {
        "answer_markdown": "### 方案一（已展开）\n1. 执行操作。",
        "coverage_ledger": [],
        "procedure_variant_ledger": [{
            "source_path": source,
            "source_label": "方案一",
            "answer_label": "方案一",
            "status": "expanded",
            "reason": "原文步骤完整。",
        }],
    }

    errors = verify_answer_draft(
        draft,
        required_facets=[],
        files_read=[source],
        media_exposed=[],
        required_procedure_variants=requirements,
    )

    assert f"missing_procedure_variant:{source}:方案二" in errors
    assert f"missing_procedure_variant:{source}:方案三" in errors


def test_parallel_procedure_gate_accepts_expanded_and_guarded_methods() -> None:
    source = "data/raw/guide.docx"
    url = "https://example.invalid/script"
    requirements = [
        ProcedureVariantRequirement(source, "第一种操作方法"),
        ProcedureVariantRequirement(source, "第二种操作方法"),
        ProcedureVariantRequirement(
            source,
            "第三种操作方法",
            external_urls=(url,),
            external_artifact_unverified=True,
        ),
    ]
    draft = {
        "answer_markdown": (
            "### 控制面板（已展开）\n1. 执行操作。\n\n"
            "### 系统命令（已展开）\n```cmd\npowercfg /h off\n```\n\n"
            "### 外部脚本（风险受控地展示）\n"
            f"原文链接：{url}。脚本内容、版本、哈希未核验，"
            "优先使用可审计的系统内置方法。"
        ),
        "coverage_ledger": [],
        "procedure_variant_ledger": [
            {
                "source_path": source,
                "source_label": "第一种操作方法",
                "answer_label": "控制面板",
                "status": "expanded",
                "reason": "原文步骤完整。",
            },
            {
                "source_path": source,
                "source_label": "第二种操作方法：控制面板没有该选项",
                "answer_label": "系统命令",
                "status": "expanded",
                "reason": "原文命令完整且非破坏性。",
            },
            {
                "source_path": source,
                "source_label": "第三种操作方法：使用外部脚本",
                "answer_label": "外部脚本",
                "status": "guarded",
                "reason": "仅保留原文链接并披露审计缺口。",
            },
        ],
    }

    errors = verify_answer_draft(
        draft,
        required_facets=[],
        files_read=[source],
        media_exposed=[],
        required_procedure_variants=requirements,
    )

    assert errors == []


def test_parallel_procedure_discovery_is_numbering_style_agnostic(
    tmp_path: Path,
) -> None:
    source = "data/raw/guide.docx"
    extracted = tmp_path / "data/extracted_docx/raw/guide.docx.md"
    extracted.parent.mkdir(parents=True)
    extracted.write_text(
        "\n".join([
            f"SOURCE_PATH: {source}",
            "[list_item] 第一种操作方法：",
            "[list_item] 使用图形界面。",
            "[list_item] 第二种操作方法：",
            "[list_item] 输入 powercfg /h off。",
            "[list_item] 第三种操作方法：使用脚本",
            "[list_item] 下载脚本：https://example.invalid/script",
            "[list_item] 下载后以管理员权限运行。",
        ]),
        encoding="utf-8",
    )
    facet = RequiredFacet(
        facet_id="operation:关闭",
        kind="query_task",
        label="关闭",
        match_terms=("关闭",),
    )
    draft = {"coverage_ledger": [{
        "facet_id": facet.facet_id,
        "status": "covered",
        "source_paths": [source],
    }]}

    requirements = pipeline_module._discover_required_procedure_variants(
        "如何关闭休眠，有哪些操作方法？",
        draft,
        required_facets=[facet],
        answer_scope=build_answer_scope("如何关闭休眠，有哪些操作方法？"),
        workspace=tmp_path,
    )

    assert [item.source_label for item in requirements] == [
        "第一种操作方法",
        "第二种操作方法",
        "第三种操作方法",
    ]
    assert requirements[-1].external_artifact_unverified is True
    assert requirements[-1].external_urls == (
        "https://example.invalid/script",
    )


def test_fault_query_does_not_force_procedure_variant_ledger(
    tmp_path: Path,
) -> None:
    source = "data/raw/cases/power.docx"
    extracted = tmp_path / "data/extracted_docx/raw/cases/power.docx.md"
    extracted.parent.mkdir(parents=True)
    extracted.write_text(
        "\n".join([
            f"SOURCE_PATH: {source}",
            "[list_item] 方法1：",
            "[list_item] 检查供电。",
            "[list_item] 方法2：",
            "[list_item] 更换电源。",
        ]),
        encoding="utf-8",
    )
    facet = RequiredFacet(
        facet_id="entity:ipc",
        kind="query_task",
        label="ipc",
        match_terms=("ipc",),
    )
    draft = {"coverage_ledger": [{
        "facet_id": facet.facet_id,
        "status": "covered",
        "source_paths": [source],
    }]}

    requirements = pipeline_module._discover_required_procedure_variants(
        "IPC 按电源键没有反应，怎么排查？",
        draft,
        required_facets=[facet],
        answer_scope=build_answer_scope("IPC 按电源键没有反应，怎么排查？"),
        workspace=tmp_path,
    )

    assert requirements == []


def test_release_gate_rejects_flat_multi_branch_numbering() -> None:
    scope = build_answer_scope(
        "电脑能进入系统和不能进入系统时，分别如何进入安全模式？"
    )
    draft = {
        "answer_markdown": (
            "## 两种分支\n"
            "1. 可以进入系统\n"
            "1. 执行第一步。\n"
            "2. 执行第二步。\n\n"
            "2. 无法进入系统\n"
            "1. 执行第一步。"
        ),
        "coverage_ledger": [],
    }

    errors = verify_answer_draft(
        draft,
        required_facets=[],
        files_read=[],
        media_exposed=[],
        answer_scope=scope,
    )

    assert "branch_heading_missing:可以进入系统" in errors
    assert "branch_heading_missing:无法进入系统" in errors


def test_release_gate_accepts_branch_headings_with_independent_steps() -> None:
    scope = build_answer_scope(
        "电脑能进入系统和不能进入系统时，分别如何进入安全模式？"
    )
    draft = {
        "answer_markdown": (
            "## 两种分支\n\n"
            "### 可以进入系统\n\n"
            "1. 执行第一步。\n"
            "2. 执行第二步。\n\n"
            "### 无法进入系统\n\n"
            "1. 执行第一步。\n"
            "2. 执行第二步。"
        ),
        "coverage_ledger": [],
    }

    errors = verify_answer_draft(
        draft,
        required_facets=[],
        files_read=[],
        media_exposed=[],
        answer_scope=scope,
    )

    assert errors == []
