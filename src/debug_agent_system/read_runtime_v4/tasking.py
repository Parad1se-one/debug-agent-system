from __future__ import annotations

from debug_agent_system.read_runtime_v3.contracts import ReadRequest
from debug_agent_system.read_runtime_v3.tasking import normalize_task

from .contracts import InvestigationTask


def compile_task(request: ReadRequest, budgets: dict[str, int] | None = None) -> InvestigationTask:
    task = normalize_task(request, budgets=budgets)
    incident = task.complexity == "incident" or bool(request.evidence_resources)
    if incident:
        goal = "建立时间约束的事故证据链，区分观察、故障域、候选根因和验证条件"
        output = "incident_report"
        sections = [
            "案件摘要", "时间对齐", "直接观察", "综合判断", "候选与反证",
            "建议立即采取", "下一步验证", "候选修复动作", "修复后验证",
            "仍需补充的证据", "来源",
        ]
        hints = ["incident_scope", "log_window", "evtx", "minidump", "timeline"]
    elif task.request_kind in {"procedure_lookup", "knowledge_lookup"}:
        goal = "从原文和 KG 证据中完整回答用户请求，并保留并列方案和媒体"
        output = "procedure_answer"
        sections = ["结论", "前置条件", "方案与步骤", "风险", "成功标志", "来源"]
        hints = ["document_closure", "parallel_variants", "media"]
    else:
        goal = "根据可追溯证据回答问题，并显式标记未知和证据缺口"
        output = "evidence_answer"
        sections = ["根据资料可知", "尚不能确认", "需要补充的信息", "资料来源"]
        hints = ["facet_closure", "source_closure"]
    return InvestigationTask(
        task=task,
        goal=goal,
        output_contract=output,
        risk_scope="controlled" if incident else "safe",
        requested_sections=sections,
        parser_hints=hints,
    )
