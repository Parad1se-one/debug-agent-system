from __future__ import annotations

from debug_agent_system.read_runtime_v3.contracts import ReadRequest
from debug_agent_system.read_runtime_v3.tasking import normalize_task


def test_task_normalizer_preserves_query_scope_time_and_resources():
    request = ReadRequest(
        query="设备闪退，参考时间：2026-08-01 21:30",
        evidence_resources=[{
            "resource_id": "pkg:1",
            "kind": "log_package",
            "name": "diagnostics_20260801.zip",
            "path": "/tmp/diagnostics_20260801.zip",
        }],
    )
    task = normalize_task(request)
    assert task.complexity == "incident"
    assert task.resource_ids == ["pkg:1"]
    assert task.time_windows[0]["reference_time"] == "2026-08-01T21:30:00"
    assert task.budgets["provider_calls"] == 12
    assert any(item == "time_windows:1" for item in task.normalization_trace)


def test_task_normalizer_uses_existing_generic_query_scope():
    task = normalize_task(ReadRequest(query="如何进入安全模式"))
    assert task.mode == "knowledge_lookup"
    assert task.request_kind == "procedure_lookup"
    assert task.facets == ["进入"]
    assert task.facet_details == [{
        "facet_id": "operation:进入",
        "kind": "operation",
        "label": "进入",
        "match_terms": ["进入", "启动到", "打开"],
        "required_for_closure": True,
    }]
    assert task.complexity in {"simple", "standard"}


def test_task_normalizer_keeps_multiple_facets_as_structured_contracts():
    task = normalize_task(ReadRequest(
        query="Windows 启动异常时，怎样用 Dism++ 备份并修复系统或引导？"
    ))
    assert task.facets == ["备份", "修复", "dism++"]
    assert all(isinstance(item, dict) for item in task.facet_details)
    assert all(item["required_for_closure"] for item in task.facet_details)
    assert not any(label.startswith("{") for label in task.facets)


def test_task_normalizer_reconciles_unseen_information_question():
    task = normalize_task(ReadRequest(query="如何打开系统管理工具？"))
    assert task.mode == "knowledge_lookup"
    assert task.request_kind == "procedure_lookup"
    assert (
        "v3_scope_reconciled:information_question_without_fault_observation"
        in task.normalization_trace
    )


def test_task_normalizer_keeps_observed_fault_as_diagnosis():
    task = normalize_task(ReadRequest(query="电脑不开机，应该怎么排查？"))
    assert task.mode == "fault_diagnosis"
    assert not any(item.startswith("v3_scope_reconciled:") for item in task.normalization_trace)


def test_task_normalizer_keeps_explicit_field_report_as_diagnosis():
    task = normalize_task(ReadRequest(
        query=(
            "现场反馈：3D拍摄日志提示某个 FOV 图片不足 42 张，采图不完整。"
            "请先定位最匹配的故障族和故障变体，并说明必须核对什么证据？"
        )
    ))
    assert task.mode == "fault_diagnosis"
    assert task.request_kind == "fault_diagnosis"
    assert "v3_scope_reconciled:explicit_incident_observation" in task.normalization_trace


def test_task_normalizer_recognizes_conditional_procedure_question():
    task = normalize_task(ReadRequest(query="拍照失败时如何升级相机 SDK？"))
    assert task.mode == "knowledge_lookup"
    assert task.request_kind == "procedure_lookup"
    assert "v3_scope_reconciled:conditional_procedure_question" in task.normalization_trace

    task = normalize_task(ReadRequest(query="电脑无法开机时应该按照什么顺序排查？"))
    assert task.mode == "knowledge_lookup"
    assert "v3_scope_reconciled:conditional_procedure_question" in task.normalization_trace


def test_task_normalizer_merges_duplicate_semantic_labels():
    task = normalize_task(ReadRequest(
        query="现场记录包含 2.5G 网卡和 NDIS 10400，请定位故障"
    ))
    assert task.facets.count("2.5g") == 1
    assert [item["label"] for item in task.facet_details].count("2.5g") == 1
    assert len(task.facets) == len(task.facet_details)


def test_task_normalizer_uses_explicit_source_only_context_contract():
    task = normalize_task(ReadRequest(
        query="请按设备、故障链和时间边界拆分 Trace",
        routing_context={
            "source_only_context": {
                "label_visibility": "source_records_only",
                "messages": [],
            },
        },
    ))
    assert task.mode == "source_only_trace_reconstruction"
    assert task.request_kind == "trace_reconstruction"
    assert "v3_scope_reconciled:explicit_source_only_context" in task.normalization_trace
