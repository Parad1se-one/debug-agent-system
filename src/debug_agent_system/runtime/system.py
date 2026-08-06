"""KG_v2-native production read-side orchestrator.

The runtime invariant is intentionally strict: primary identity, candidates,
plan steps, branch transitions, resolution outcomes, and evidence are KG_v2
objects.  No legacy Error/Check/Solution store is imported or consulted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from debug_agent_system.agents.read.codex_answer import CodexEvidenceAnswerComposer
from debug_agent_system.agents.read.evidence_answer import (
    EvidenceAnswerComposer,
    render_answer_sections,
)
from debug_agent_system.agents.read.evidence_pack import EvidencePackBuilder
from debug_agent_system.agents.read.mem_session import DiagnosticSessionStore
from debug_agent_system.agents.read.navigation_evidence_gap import (
    insert_navigation_evidence_gap,
    navigation_evidence_gap_section,
)
from debug_agent_system.agents.read.o_evidence_gap import EvidenceGapResolver
from debug_agent_system.core.config import SystemConfig, load_config
from debug_agent_system.core.contracts import AgentResponse, DebugAgentInput, SessionState, to_jsonable
from debug_agent_system.incident_runtime import IncidentEvidenceRuntime
from debug_agent_system.incident_runtime.artifacts import ArtifactLimits
from debug_agent_system.knowledge_v2.read_model import (
    KGV2ReadModel,
    V2Candidate,
    V2DiagnosticPlan,
    V2PlanStep,
)
from debug_agent_system.knowledge_v2.query_scope import (
    analyze_query_scope,
    chunk_matches_named_scope,
    diagnostic_subject_compatible,
    scope_polarity_compatible,
    source_document_title,
    subject_domain_compatible,
    title_match_signals,
)
from debug_agent_system.knowledge_v2.sqlite_sag_v2 import (
    SAG_V2_INDEX_SCHEMA,
    SqliteSAGV2,
    build_sqlite_sag_v2,
    kg_v2_graph_revision,
    kg_v2_source_revision,
)

_SOLVED = ("已解决", "解决了", "恢复正常", "恢复生产", "修复", "好了", "不再复现")
_NEGATIVE = ("未解决", "没有解决", "仍然", "还是", "无效", "失败", "不行", "继续复现")
_PENDING = ("观察中", "继续观察", "待验证", "还需要", "进一步确认", "暂时恢复", "临时恢复")
_CONFIRM = ("确认", "同意", "可以执行", "人工已批准", "继续执行", "yes", "ok")
_REJECT = ("拒绝", "不执行", "不要执行", "取消", "否")


class DebugAgentSystem:
    """O0 supervisor for the KG_v2-only read pipeline."""

    schema_version = "debug_agent_system.response.v2"

    def __init__(
        self,
        config: SystemConfig | None = None,
        *,
        answer_model_client: Any | None = None,
    ) -> None:
        self.config = config or load_config()
        self.read_model = self._build_read_model()
        # Public compatibility attribute.  It is a KG_v2 read model, never a
        # legacy KGStore.
        self.store = self.read_model
        self.answer_composer = EvidenceAnswerComposer(self.read_model)
        self.evidence_pack_builder = EvidencePackBuilder(
            max_documents=self.config.read_llm.max_answer_documents,
            max_chunks=self.config.read_llm.max_answer_chunks,
            max_input_chars=self.config.read_llm.max_answer_input_chars,
        )
        self.llm_answer_composer = CodexEvidenceAnswerComposer(
            client=answer_model_client,
            model=self.config.read_llm.model,
            base_url=self.config.read_llm.base_url,
            timeout_seconds=self.config.read_llm.timeout_seconds,
            env_file=self.config.root / ".env.local",
        )
        self.evidence_gap_resolver = EvidenceGapResolver()
        incident = self.config.incident_runtime
        self.incident_runtime = IncidentEvidenceRuntime(
            self.read_model,
            limits=ArtifactLimits(
                max_package_bytes=incident.max_package_bytes,
                max_member_bytes=incident.max_member_bytes,
                max_dump_member_bytes=incident.max_dump_member_bytes,
                max_total_uncompressed_bytes=incident.max_total_uncompressed_bytes,
                max_members=incident.max_members,
                max_nesting=incident.max_nesting,
                max_compression_ratio=incident.max_compression_ratio,
            ),
            allow_dump_analysis=incident.allow_dump_analysis,
            allow_ocr_analysis=incident.allow_ocr_analysis,
        )
        self.sessions = DiagnosticSessionStore(self.config.session_store)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "DebugAgentSystem":
        return cls(load_config(path))

    def _build_read_model(self) -> KGV2ReadModel:
        root = self.config.knowledge.kg_v2_root
        if not (root / "objects" / "fault_variants.json").exists():
            raise FileNotFoundError(f"KG_v2 root is unavailable: {root}")
        sag: SqliteSAGV2 | None = None
        if self.config.knowledge.store == "sqlite_sag_v2":
            sag_path = self.config.knowledge.kg_v2_sqlite_path
            sag = SqliteSAGV2(sag_path)
            revision = kg_v2_graph_revision(root)
            source_revision = kg_v2_source_revision(root)
            graph_or_schema_mismatch = (
                sag.graph_revision() != revision
                or sag.index_schema() != SAG_V2_INDEX_SCHEMA
            )
            if self.config.knowledge.sag_snapshot_mode and graph_or_schema_mismatch:
                raise RuntimeError(
                    "packaged SAG_v2 snapshot does not match its canonical KG_v2 "
                    f"(graph={sag.graph_revision()!r}/{revision!r}, "
                    f"schema={sag.index_schema()!r}/{SAG_V2_INDEX_SCHEMA!r})"
                )
            if not self.config.knowledge.sag_snapshot_mode and (
                graph_or_schema_mismatch
                or sag.source_revision() != source_revision
            ):
                build_sqlite_sag_v2(root, sag_path, reset=True)
                sag = SqliteSAGV2(sag_path)
        elif self.config.knowledge.store != "kg_v2_json":
            raise ValueError(
                "debug_agent_system runtime supports only kg_v2_json or sqlite_sag_v2; "
                f"got {self.config.knowledge.store!r}"
            )
        return KGV2ReadModel(str(root), sag=sag)

    def start(self, payload: dict[str, Any] | DebugAgentInput) -> dict[str, Any]:
        inp = self._input(payload)
        requested_sid = str(inp.session.get("session_id") or "") or None
        state = self.sessions.create(inp.query, requested_sid)
        state.metadata["input"] = {
            "interactive": inp.interactive,
            "routing_context": inp.routing_context,
            "evidence_resources": list(inp.evidence_resources),
            "log_summary": dict(inp.log_summary),
        }
        if (
            self.config.incident_runtime.enabled
            and (
                inp.evidence_resources
                or inp.log_summary
                or _looks_like_incident_query(inp.query)
            )
        ):
            incident_result = self.incident_runtime.analyze(
                inp.query,
                inp.evidence_resources,
                log_summary=inp.log_summary,
            )
            state.metadata["incident_runtime"] = incident_result.to_dict()
        response = self._start_state(state, inp.query, inp.interactive)
        response = self._maybe_complete_evidence_gap(
            state,
            response,
            query=inp.query,
            interactive=inp.interactive,
            evidence_resources=inp.evidence_resources,
        )
        if (
            self.config.incident_runtime.enabled
            and not self.config.incident_runtime.shadow_mode
            and state.metadata.get("incident_runtime")
        ):
            incident_payload = dict(state.metadata["incident_runtime"])
            response.answer = str(incident_payload.get("report") or response.answer)
            response.metadata["incident_runtime"] = incident_payload
            response.metadata["incident_runtime"]["active_answer"] = True
        return to_jsonable(response)

    def diagnose(self, payload: dict[str, Any] | DebugAgentInput) -> dict[str, Any]:
        return self.start(payload)

    def analyze_incident(
        self,
        payload: dict[str, Any] | DebugAgentInput,
    ) -> dict[str, Any]:
        """Run the structured incident side path without changing KG state."""

        inp = self._input(payload)
        result = self.incident_runtime.analyze(
            inp.query,
            inp.evidence_resources,
            log_summary=inp.log_summary,
        )
        return result.to_dict()

    def step(
        self,
        session_id: str,
        user_message: str,
        evidence_resources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if state is None:
            return to_jsonable(AgentResponse(
                schema_version=self.schema_version,
                session_id=session_id,
                status="failed",
                answer="unknown session_id",
                failure_type="unknown_session",
                observability={"session_id": session_id, "agent_id": "O0", "status": "failed"},
            ))
        state.turn_count += 1
        if state.status == "resolved":
            return to_jsonable(self._response(
                state,
                "resolved",
                f"该会话已由 KG_v2 verified_fix 证据闭环：{state.resolution}",
                resolution=state.resolution,
            ))
        if state.status == "escalate":
            return to_jsonable(self._response(
                state,
                "escalate",
                "该会话已经结束并进入人工升级流程。",
                escalation_target=state.escalation_target,
            ))
        if state.turn_count >= self.config.runtime.max_turns:
            return to_jsonable(self._escalate(state, None, "max_turns_reached"))

        if (
            state.status == "ask_info"
            and state.required_data
            and evidence_resources
            and self.config.evidence_gap.enabled
            and self._evidence_gap_allowed(state)
        ):
            if self.config.incident_runtime.enabled:
                previous = list(
                    ((state.metadata.get("input") or {}).get("evidence_resources") or [])
                )
                combined = [*previous, *evidence_resources]
                incident_result = self.incident_runtime.analyze(
                    _strip_missing_statements(state.query),
                    combined,
                    log_summary=dict(
                        ((state.metadata.get("input") or {}).get("log_summary") or {})
                    ),
                )
                state.metadata["incident_runtime"] = incident_result.to_dict()
                self.sessions.save(state)
                response = self._response(
                    state,
                    "ask_info",
                    incident_result.report,
                    required_data=[
                        item
                        for hypothesis in incident_result.hypotheses
                        for item in hypothesis.missing_evidence
                    ][: self.config.thresholds.ask_info_max_required_items],
                    failure_type="incident_evidence_incomplete",
                )
                return to_jsonable(response)
            resolution = self.evidence_gap_resolver.resolve(
                state.required_data,
                evidence_resources,
                max_resources=self.config.evidence_gap.max_resources,
                max_bytes=self.config.evidence_gap.max_bytes_per_resource,
                max_rounds=self.config.evidence_gap.max_rounds,
                processed_fingerprints=state.metadata.get(
                    "processed_tool_fingerprints"
                )
                or [],
            )
            self._record_evidence_gap(state, resolution)
            if resolution.retrieval_context:
                state.query = "\n".join(
                    item
                    for item in (
                        _strip_missing_statements(state.query),
                        f"用户补充：{user_message}" if user_message else "",
                        "以下为只读工具提取、尚未等同于根因的现场证据：",
                        resolution.retrieval_context,
                    )
                    if str(item).strip()
                )
                interactive = bool(
                    (state.metadata.get("input") or {}).get("interactive", True)
                )
                return to_jsonable(
                    self._start_state(state, state.query, interactive)
                )
            if not str(user_message or "").strip():
                return to_jsonable(
                    self._response(
                        state,
                        "ask_info",
                        "已检查补充材料，但其中没有足以回答当前缺口的可追溯观察。",
                        required_data=state.required_data,
                        failure_type=state.failure_type,
                    )
                )

        if state.lock_status == "document_answer_only" and not state.top_variant_id:
            # A source-complete knowledge answer has no diagnostic plan to
            # restore. Treat a later message as another read request instead
            # of failing the session with ``invalid_kg_v2_session``.
            state.query = str(user_message or "").strip()
            interactive = bool(
                (state.metadata.get("input") or {}).get("interactive", True)
            )
            return to_jsonable(
                self._start_state(state, state.query, interactive)
            )

        plan = self._restore_plan(state)
        if plan is None:
            state.status = "failed"
            state.failure_type = "invalid_kg_v2_session"
            self.sessions.save(state)
            return to_jsonable(self._response(state, "failed", "会话引用的 KG_v2 诊断上下文已失效。"))

        pending_confirmation = str(state.metadata.get("pending_confirmation_action_id") or "")
        if pending_confirmation:
            if _contains_any(user_message, _REJECT):
                state.ruled_out.append(pending_confirmation)
                state.metadata.pop("pending_confirmation_action_id", None)
                return to_jsonable(self._advance(state, plan, "unsafe_action_rejected"))
            if _contains_any(user_message, _CONFIRM):
                state.metadata.pop("pending_confirmation_action_id", None)
                step = self._step_by_action(plan, pending_confirmation)
                if step is None:
                    return to_jsonable(self._escalate(state, plan, "confirmed_action_not_in_plan"))
                state.metadata.setdefault("confirmed_action_ids", []).append(pending_confirmation)
                return to_jsonable(self._present_step(state, plan, step, safety_confirmed=True))
            state.required_data = [f"请明确是否由人工确认执行高成本/高风险动作：{pending_confirmation}"]
            self.sessions.save(state)
            return to_jsonable(self._response(
                state,
                "ask_info",
                "该动作需要人工明确确认；未确认前不会进入执行步骤。",
                required_data=state.required_data,
            ))

        pending_branch_action = str(
            state.metadata.get("pending_branch_condition_action_id") or ""
        )
        if pending_branch_action:
            current = self._step_by_action(plan, pending_branch_action)
            if current is None:
                return to_jsonable(
                    self._escalate(state, plan, "pending_branch_action_not_in_plan")
                )
            outcome_type = str(
                state.metadata.get("pending_branch_outcome_type")
                or "diagnostic_method"
            )
            branch = self._matching_branch(
                current,
                outcome_type,
                user_message,
                (state.metadata.get("input") or {}).get("routing_context") or {},
            )
            if branch is not None and branch.get("_selection_status") == "needs_condition":
                return to_jsonable(
                    self._ask_branch_condition(
                        state, current, outcome_type, branch
                    )
                )
            state.metadata.pop("pending_branch_condition_action_id", None)
            state.metadata.pop("pending_branch_outcome_type", None)
            state.action_results[current.action_id] = user_message
            state.check_results[current.action_id] = user_message
            if branch is not None:
                return to_jsonable(
                    self._follow_branch(
                        state, plan, current, outcome_type, branch
                    )
                )
            return to_jsonable(
                self._ask_branch_condition(
                    state,
                    current,
                    outcome_type,
                    {
                        "_selection_status": "needs_condition",
                        "conditions": ["请提供与分支条件对应的明确现场信号"],
                        "candidate_branch_rule_ids": [],
                    },
                )
            )

        if state.status == "ask_info" and state.required_data:
            state.query = f"{_strip_missing_statements(state.query)}\n用户补充：{user_message}".strip()
            state.required_data = []
            state.failure_type = ""
            interactive = bool((state.metadata.get("input") or {}).get("interactive", True))
            return to_jsonable(self._start_state(state, state.query, interactive))

        current = self._step_by_action(plan, state.current_action_id)
        if current is None:
            return to_jsonable(self._escalate(state, plan, "current_action_not_in_plan"))

        outcome_type = _classify_outcome(user_message)
        state.action_results[current.action_id] = user_message
        state.check_results[current.action_id] = user_message
        state.metadata.setdefault("observed_outcomes", []).append({
            "action_id": current.action_id,
            "trace_step_id": current.trace_step_id,
            "classified_outcome_type": outcome_type,
            "user_message": user_message,
        })

        branch = self._matching_branch(
            current,
            outcome_type,
            user_message,
            (state.metadata.get("input") or {}).get("routing_context") or {},
        )
        if branch is not None:
            if branch.get("_selection_status") == "needs_condition":
                return to_jsonable(
                    self._ask_branch_condition(
                        state, current, outcome_type, branch
                    )
                )
            return to_jsonable(
                self._follow_branch(
                    state, plan, current, outcome_type, branch
                )
            )

        if outcome_type == "verified_fix":
            verified = self._verified_fix_outcome(current, user_message)
            if verified is not None:
                return to_jsonable(self._resolve(state, plan, current, verified, None))
            return to_jsonable(self._pending_validation(
                state,
                plan,
                current,
                "user_reported_fix_without_kg_verified_fix",
            ))
        return to_jsonable(self._advance(state, plan, f"outcome:{outcome_type}"))

    def _maybe_complete_evidence_gap(
        self,
        state: SessionState,
        response: AgentResponse,
        *,
        query: str,
        interactive: bool,
        evidence_resources: list[dict[str, Any]],
    ) -> AgentResponse:
        if (
            response.status != "ask_info"
            or not self.config.evidence_gap.enabled
            or not evidence_resources
            or not self._evidence_gap_allowed(state)
        ):
            return response
        # The incident runtime keeps parser observations as typed evidence.
        # Do not flatten them back into query text and lose time/source/stack
        # relationships.  Legacy EvidenceGapResolver remains the rollback path.
        if state.metadata.get("incident_runtime"):
            return response
        resolution = self.evidence_gap_resolver.resolve(
            response.required_data,
            evidence_resources,
            max_resources=self.config.evidence_gap.max_resources,
            max_bytes=self.config.evidence_gap.max_bytes_per_resource,
            max_rounds=self.config.evidence_gap.max_rounds,
            processed_fingerprints=state.metadata.get(
                "processed_tool_fingerprints"
            )
            or [],
        )
        self._record_evidence_gap(state, resolution)
        if resolution.retrieval_context:
            state.query = (
                f"{query}\n以下为只读工具提取、尚未等同于根因的现场证据：\n"
                f"{resolution.retrieval_context}"
            )
            return self._start_state(state, state.query, interactive)
        # Re-compose once so parser observations and exclusions are visible in
        # the answer/metadata even when they cannot improve KG retrieval.
        return self._response(
            state,
            response.status,
            response.answer,
            required_data=response.required_data,
            failure_type=response.failure_type,
        )

    @staticmethod
    def _evidence_gap_allowed(state: SessionState) -> bool:
        if state.metadata.get("pending_confirmation_action_id"):
            return False
        if state.metadata.get("pending_branch_condition_action_id"):
            return False
        return state.failure_type not in {
            "pending_validation",
            "branch_condition_required",
        }

    def _record_evidence_gap(self, state: SessionState, resolution: Any) -> None:
        serialized = to_jsonable(resolution)
        state.metadata["evidence_gap_resolution"] = serialized
        fingerprints = [
            str(item.get("call_fingerprint") or "")
            for item in serialized.get("tool_results") or []
            if str(item.get("call_fingerprint") or "")
        ]
        state.metadata["processed_tool_fingerprints"] = list(
            dict.fromkeys(
                [
                    *(state.metadata.get("processed_tool_fingerprints") or []),
                    *fingerprints,
                ]
            )
        )
        self.sessions.save(state)

    def _start_state(self, state: SessionState, query: str, interactive: bool) -> AgentResponse:
        self._reset_diagnosis_state(state)
        query_scope = analyze_query_scope(query)
        candidates = self.read_model.search_variants(query, limit=10)
        retrieval_context = self.read_model.last_retrieval or {}
        state.metadata["retrieval"] = {
            "backend": "sqlite_sag_v2" if self.read_model.sag is not None else "kg_v2_json_scan",
            "graph_revision": self._current_graph_revision(),
            "candidates": [to_jsonable(item) for item in candidates],
            "supporting_chunks": list(retrieval_context.get("chunks") or []),
            "trace": dict(retrieval_context.get("trace") or {}),
            "top_margin": float((retrieval_context.get("trace") or {}).get("top_margin") or 0.0),
            "fallback_used": bool((retrieval_context.get("trace") or {}).get("fallback_used")),
        }
        state.metadata["graph_revision"] = self._current_graph_revision()
        state.metadata["query_scope"] = query_scope.to_dict()
        retrieval_trace = retrieval_context.get("trace") or {}
        direct_documents = list(retrieval_trace.get("direct_document_matches") or [])
        navigation_documents = list(
            retrieval_trace.get("navigation_document_matches") or []
        )
        if direct_documents and self._should_answer_from_document(
            candidates,
            query_scope.mode,
        ):
            navigation_answer_complete = bool(navigation_documents)
            response_status = (
                "step"
                if navigation_answer_complete or query_scope.mode == "knowledge_lookup"
                else "ask_info"
            )
            state.status = response_status
            state.failure_type = (
                ""
                if navigation_answer_complete or query_scope.mode == "knowledge_lookup"
                else "document_answer_variant_unverified"
            )
            state.lock_status = "document_answer_only"
            # A fully resolved navigation page is already answerable from its
            # child documents.  Do not append a generic fault-diagnosis
            # question that is unrelated to the requested procedure.
            state.required_data = (
                []
                if navigation_documents or query_scope.mode == "knowledge_lookup"
                else [self._document_followup_question(query)]
            )
            state.metadata["document_answer_mode"] = {
                "active": True,
                "reason": "direct_document_match_without_reliable_variant_support",
                "documents": direct_documents,
                "navigation_documents": navigation_documents,
                "candidate_variant_ids": [item.variant_id for item in candidates[:3]],
            }
            self.sessions.save(state)
            return self._response(
                state,
                response_status,
                "已直接命中相关文档；先按文档组织现有信息。由于文档尚未绑定到可靠的 KG_v2 Variant，"
                "候选故障只作为适用条件展示，不据此替换文档处理路径。",
                required_data=state.required_data,
            )
        if query_scope.mode == "knowledge_lookup":
            primary_chunks, primary_documents, excluded_chunks = (
                self._primary_knowledge_scope(
                    query,
                    list(retrieval_context.get("chunks") or []),
                )
            )
            state.metadata["retrieval"]["supporting_chunks"] = primary_chunks
            state.metadata["retrieval"]["primary_evidence_scope"] = {
                "mode": "knowledge_lookup",
                "document_ids": [
                    str(item.get("document_id") or "")
                    for item in primary_documents
                    if str(item.get("document_id") or "")
                ],
                "chunk_ids": [
                    str(item.get("chunk_id") or "")
                    for item in primary_chunks
                    if str(item.get("chunk_id") or "")
                ],
                "excluded": excluded_chunks,
            }
            if primary_chunks:
                state.status = "step"
                state.failure_type = ""
                state.lock_status = "document_answer_only"
                state.required_data = []
                state.metadata["document_answer_mode"] = {
                    "active": True,
                    "reason": "knowledge_intent_primary_evidence_scope",
                    "documents": primary_documents,
                    "navigation_documents": [],
                    "candidate_variant_ids": [
                        item.variant_id for item in candidates[:3]
                    ],
                }
                self.sessions.save(state)
                return self._response(
                    state,
                    "step",
                    "已按知识查询意图限定主证据域；回答只使用与所指工具、型号或主题相符的可追溯资料。",
                )
            state.status = "ask_info"
            state.failure_type = "knowledge_scope_not_covered"
            state.lock_status = "document_answer_only"
            state.required_data = ["请提供所指文档、工具或型号的准确名称，或补充相关资料。"]
            self.sessions.save(state)
            return self._response(
                state,
                "ask_info",
                "当前索引没有找到与所指知识主题一致的可追溯原文，未使用相邻故障 Variant 代替回答。",
                required_data=state.required_data,
            )
        if not candidates:
            state.status = "ask_info"
            state.failure_type = "no_kg_v2_variant_match"
            state.required_data = ["请补充完整故障现象或报错文本"]
            self.sessions.save(state)
            return self._response(
                state,
                "ask_info",
                "当前信息无法命中 KG_v2 FaultVariant，请补充更完整的故障现象。",
                required_data=state.required_data,
            )
        top = candidates[0]
        if not self._candidate_named_scope_compatible(query, top):
            state.status = "ask_info"
            state.failure_type = "candidate_entity_scope_mismatch"
            state.required_data = ["请补充完整故障现象，或确认所指工具、型号和报错文本。"]
            self.sessions.save(state)
            return self._response(
                state,
                "ask_info",
                "候选 FaultVariant 未覆盖 Query 中的关键工具或型号，已阻止错误锁定。",
                required_data=state.required_data,
            )
        if not diagnostic_subject_compatible(
            query,
            " ".join((top.family_label, top.variant_label)),
        ):
            state.status = "ask_info"
            state.failure_type = "candidate_subject_scope_mismatch"
            state.required_data = [
                "请补充发生误报/漏报的具体对象、所在模块和现场表现。"
            ]
            self.sessions.save(state)
            return self._response(
                state,
                "ask_info",
                "候选 FaultVariant 的复合故障对象与 Query 不一致，已阻止仅凭通用故障词错误锁定。",
                required_data=state.required_data,
            )
        if top.score < self.config.thresholds.graph_match_min_score:
            state.status = "ask_info"
            state.failure_type = "low_kg_v2_candidate_score"
            state.required_data = ["请补充完整故障现象、明确报错文本或所指文档名称"]
            self.sessions.save(state)
            return self._response(
                state,
                "ask_info",
                "当前检索结果不足以锁定 KG_v2 FaultVariant；如有相关资料，先提供可引用内容，再请求补充现场信息。",
                required_data=state.required_data,
            )
        self._bind_candidate(state, top)
        plan = self.read_model.compile_plan(top.family_id, top.variant_id)
        self._bind_plan(state, plan)
        top_margin = float(state.metadata["retrieval"].get("top_margin") or 0.0)
        if len(candidates) > 1 and top_margin < self.config.thresholds.graph_match_min_margin:
            state.lock_status = "kg_v2_tentative_ambiguous"
            state.metadata["ambiguous_candidate_ids"] = [item.variant_id for item in candidates[:3]]
            return self._ask_required_info(state, plan, "ambiguous_kg_v2_candidates")
        missing = self._initial_missing_required_info(query, plan)
        if missing:
            return self._ask_required_info(state, plan, "missing_branch_context", missing)
        if not plan.steps:
            return self._escalate(state, plan, "kg_v2_plan_has_no_actions")
        state.current_index = 0
        step = plan.steps[0]
        if not interactive:
            state.metadata["recommended_action_ids"] = [item.action_id for item in plan.steps[:8]]
        return self._present_step(state, plan, step)

    @staticmethod
    def _should_answer_from_document(
        candidates: list[V2Candidate],
        query_mode: str = "fault_diagnosis",
    ) -> bool:
        if query_mode == "knowledge_lookup":
            return True
        if not candidates:
            return True
        top = candidates[0]
        if not top.supporting_chunks:
            return True
        strong_variant_match = "variant_label" in set(top.matched_fields)
        return not strong_variant_match and top.score < 12.0

    def _primary_knowledge_scope(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Select one coherent source domain for a non-diagnostic question."""

        eligible: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not bool(chunk.get("approved", True)):
                excluded.append({"id": chunk_id, "reason": "not_approved"})
                continue
            corpus = " ".join([
                str(chunk.get("source_label") or ""),
                source_document_title(chunk),
                str(chunk.get("text") or ""),
            ])
            if not scope_polarity_compatible(query, corpus):
                excluded.append({"id": chunk_id, "reason": "applicability_branch_conflict"})
                continue
            if not subject_domain_compatible(query, corpus):
                excluded.append({"id": chunk_id, "reason": "subject_domain_mismatch"})
                continue
            if not chunk_matches_named_scope(query, chunk):
                excluded.append({"id": chunk_id, "reason": "named_entity_scope_mismatch"})
                continue
            components = (
                chunk.get("score_components")
                if isinstance(chunk.get("score_components"), dict)
                else {}
            )
            coverage = float(components.get("query_coverage") or 0.0)
            score = float(chunk.get("retrieval_score") or 0.0)
            title_signals = title_match_signals(
                query,
                source_document_title(chunk) or str(chunk.get("source_label") or ""),
            )
            if not bool(title_signals.get("safe")) and (
                coverage < 0.5 or score < 4.0
            ):
                excluded.append({"id": chunk_id, "reason": "weak_primary_scope"})
                continue
            eligible.append({
                **chunk,
                "primary_evidence_match": True,
            })

        if not eligible:
            return [], [], excluded
        query_scope = analyze_query_scope(query)
        document_scores: dict[str, dict[str, Any]] = {}
        for chunk in eligible:
            document_id = str(chunk.get("document_id") or "")
            if document_id:
                components = (
                    chunk.get("score_components")
                    if isinstance(chunk.get("score_components"), dict)
                    else {}
                )
                title_signals = title_match_signals(
                    query,
                    source_document_title(chunk)
                    or str(chunk.get("source_label") or ""),
                )
                current = document_scores.setdefault(document_id, {
                    "max_score": 0.0,
                    "max_coverage": 0.0,
                    "identifier_ratio": 0.0,
                    "title_safe": False,
                })
                current["max_score"] = max(
                    float(current["max_score"]),
                    float(chunk.get("retrieval_score") or 0.0),
                )
                current["max_coverage"] = max(
                    float(current["max_coverage"]),
                    float(components.get("query_coverage") or 0.0),
                )
                current["identifier_ratio"] = max(
                    float(current["identifier_ratio"]),
                    float(components.get("identifier_ratio") or 0.0),
                )
                current["title_safe"] = bool(
                    current["title_safe"] or title_signals.get("safe")
                )
        ranked_documents = sorted(
            document_scores.items(),
            key=lambda item: (
                -(
                    float(item[1]["max_score"])
                    + (4.0 if item[1]["title_safe"] else 0.0)
                    + 3.0 * float(item[1]["identifier_ratio"])
                ),
                -float(item[1]["max_coverage"]),
                item[0],
            ),
        )
        # A procedure lookup expands one coherent handbook instead of the
        # three documents that merely mention the same command.  A comparison
        # may intentionally need multiple sources, but title anchors take
        # precedence over incidental body hits.
        if query_scope.request_kind == "comparison_lookup":
            title_anchored = [
                item for item in ranked_documents if item[1]["title_safe"]
            ]
            selected_document_ids = [
                document_id
                for document_id, _stats in (title_anchored or ranked_documents)[:3]
            ]
        else:
            selected_document_ids = [
                document_id for document_id, _stats in ranked_documents[:1]
            ]
        primary_documents = [
            {
                "document_id": document_id,
                "source_label": next(
                    (
                        source_document_title(item)
                        or str(item.get("source_label") or document_id)
                        for item in eligible
                        if str(item.get("document_id") or "") == document_id
                    ),
                    document_id,
                ),
                "match_reasons": ["knowledge_intent_primary_scope"],
            }
            for document_id in selected_document_ids
        ]
        if selected_document_ids and self.read_model.sag is not None:
            expanded = self.read_model.sag.expand_source_document_chunks(
                query,
                selected_document_ids,
            )
            primary = [
                {
                    **item,
                    "direct_document_match": True,
                    "primary_evidence_match": True,
                }
                for item in expanded
                if str(item.get("document_id") or "") in selected_document_ids
            ]
            if primary:
                return primary, primary_documents, excluded
        selected_ids = set(selected_document_ids)
        primary = [
            {
                **item,
                "direct_document_match": bool(
                    str(item.get("document_id") or "") in selected_ids
                ),
            }
            for item in eligible
            if not selected_ids or str(item.get("document_id") or "") in selected_ids
        ]
        return primary, primary_documents, excluded

    @staticmethod
    def _candidate_named_scope_compatible(
        query: str,
        candidate: V2Candidate,
    ) -> bool:
        scope = analyze_query_scope(query)
        if not scope.strong_identifiers:
            return True
        corpus = " ".join([
            candidate.family_label,
            candidate.variant_label,
            *candidate.matched_entities,
            *[
                str(chunk.get("text") or "")
                for chunk in candidate.supporting_chunks
            ],
        ])
        probe = {
            "source_label": candidate.variant_label,
            "text": corpus,
            "source_offsets": [],
        }
        return chunk_matches_named_scope(query, probe)

    @staticmethod
    def _document_followup_question(query: str) -> str:
        text = str(query or "").lower()
        if "如何进入安全模式" in text or text.strip() == "安全模式":
            return "当前是否还能进入 Windows 系统？"
        if "转圈" in text or "无法进入系统" in text:
            return "请补充当前停留界面、是否出现报错，以及能否进入自动修复（WinRE）或安全模式。"
        if "usb" in text:
            return "请补充具体是哪类USB设备、系统是否识别，以及设备管理器中的状态或报错。"
        if "卡顿" in text:
            return "请补充卡顿发生阶段、持续或偶发情况，以及CPU、内存、磁盘占用和报错表现。"
        if any(token in text for token in ("不开机", "无法开机", "无通电")):
            return (
                "请补充按下电源键后的现场表现：风扇和指示灯是否亮、屏幕是否显示、"
                "是否有蜂鸣声或 Debug 灯，以及当前已完成哪些无需拆机的检查。"
            )
        return "请补充当前现场表现和报错，以区分文档适用路径与候选故障分支。"

    @staticmethod
    def _reset_diagnosis_state(state: SessionState) -> None:
        state.status = "step"
        state.top_error_id = ""
        state.top_error_label = ""
        state.retrieval_route = ""
        state.lock_status = ""
        state.current_check_id = ""
        state.current_check = ""
        state.current_index = 0
        state.checks_presented = []
        state.check_results = {}
        state.ruled_out = []
        state.which_check_solved = ""
        state.required_data = []
        state.resolution = ""
        state.escalation_target = ""
        state.failure_type = ""
        state.top_family_id = ""
        state.top_variant_id = ""
        state.top_family_label = ""
        state.top_variant_label = ""
        state.plan_id = ""
        state.plan_source_type = ""
        state.current_action_id = ""
        state.current_trace_step_id = ""
        state.actions_presented = []
        state.action_results = {}
        state.resolved_action_id = ""
        state.evidence_ids = []
        preserved = {
            key: state.metadata.get(key)
            for key in (
                "input",
                "evidence_gap_resolution",
                "processed_tool_fingerprints",
                "codex_tool_harness",
                # Kept only so sessions created by the pre-Codex adapter can
                # still be resumed during the migration window.
                "deepseek_tool_harness",
                "incident_runtime",
            )
            if key in state.metadata
        }
        state.metadata = preserved

    def _bind_candidate(self, state: SessionState, candidate: V2Candidate) -> None:
        state.top_family_id = candidate.family_id
        state.top_variant_id = candidate.variant_id
        state.top_family_label = candidate.family_label
        state.top_variant_label = candidate.variant_label
        state.retrieval_route = candidate.route
        state.confidence = min(0.98, 0.35 + candidate.score / 30.0)
        state.lock_status = "kg_v2_locked"
        # Legacy-named API aliases contain KG_v2 IDs and labels.
        state.top_error_id = candidate.variant_id
        state.top_error_label = candidate.variant_label
        self._add_evidence(state, candidate.evidence_ids)

    def _bind_plan(self, state: SessionState, plan: V2DiagnosticPlan) -> None:
        state.plan_id = plan.plan_id
        state.plan_source_type = plan.source_type
        state.metadata["diagnostic_plan"] = to_jsonable(plan)
        state.metadata["plan_action_ids"] = [step.action_id for step in plan.steps]
        self._add_evidence(state, plan.evidence_ids)

    def _restore_plan(self, state: SessionState) -> V2DiagnosticPlan | None:
        pinned_revision = str(state.metadata.get("graph_revision") or "")
        if pinned_revision and pinned_revision != self._current_graph_revision():
            return None
        if not (
            self.read_model.has_object(state.top_family_id, "FaultFamily")
            and self.read_model.has_object(state.top_variant_id, "FaultVariant")
        ):
            return None
        plan = self.read_model.compile_plan(state.top_family_id, state.top_variant_id)
        if plan.plan_id != state.plan_id:
            return None
        return plan

    def _current_graph_revision(self) -> str:
        if self.read_model.sag is not None:
            return self.read_model.sag.graph_revision()
        return kg_v2_graph_revision(self.config.knowledge.kg_v2_root)

    def _present_step(
        self,
        state: SessionState,
        plan: V2DiagnosticPlan,
        step: V2PlanStep,
        *,
        safety_confirmed: bool = False,
    ) -> AgentResponse:
        if (
            (step.destructive or step.high_cost)
            and self.config.runtime.destructive_action_requires_human_confirm
            and not safety_confirmed
            and step.action_id not in set(state.metadata.get("confirmed_action_ids") or [])
        ):
            state.status = "ask_info"
            state.current_action_id = step.action_id
            state.current_trace_step_id = step.trace_step_id
            state.current_check_id = step.action_id
            state.current_check = step.label
            state.metadata["pending_confirmation_action_id"] = step.action_id
            state.required_data = [f"请由人工确认是否执行：{step.label}"]
            self._add_evidence(state, step.evidence_ids)
            self.sessions.save(state)
            return self._response(
                state,
                "ask_info",
                f"KG_v2 计划的下一动作属于高成本或高风险动作：{step.label}。未获得人工确认前不会执行。",
                required_data=state.required_data,
            )
        state.status = "step"
        state.failure_type = ""
        state.required_data = []
        state.current_index = max(0, step.ordinal - 1)
        state.current_action_id = step.action_id
        state.current_trace_step_id = step.trace_step_id
        state.current_check_id = step.action_id
        state.current_check = step.label
        if step.action_id not in state.actions_presented:
            state.actions_presented.append(step.action_id)
        if step.action_id not in state.checks_presented:
            state.checks_presented.append(step.action_id)
        self._add_evidence(state, step.evidence_ids)
        state.metadata["current_plan_step"] = to_jsonable(step)
        self.sessions.save(state)
        return self._response(state, "step", self._render_step(state, plan, step))

    def _advance(self, state: SessionState, plan: V2DiagnosticPlan, reason: str) -> AgentResponse:
        state.ruled_out.append(state.current_action_id)
        start = max(0, state.current_index + 1)
        for step in plan.steps[start:]:
            if step.action_id not in set(state.ruled_out):
                state.metadata["advance_reason"] = reason
                return self._present_step(state, plan, step)
        return self._escalate(state, plan, "kg_v2_plan_exhausted")

    def _pending_validation(
        self,
        state: SessionState,
        plan: V2DiagnosticPlan,
        step: V2PlanStep,
        reason: str,
    ) -> AgentResponse:
        state.status = "step"
        state.failure_type = "pending_validation"
        state.metadata["verification"] = {
            "supported": False,
            "reason": reason,
            "required_outcome_type": "verified_fix",
            "action_id": step.action_id,
        }
        self.sessions.save(state)
        answer = (
            f"现场反馈表明动作“{step.label}”可能有效，但 KG_v2 中没有该动作对应的 verified_fix 证据，"
            "因此当前只能标记为待验证，不能宣布已解决。请继续验证是否复现，或进入后续诊断动作。"
        )
        return self._response(state, "step", answer)

    def _resolve(
        self,
        state: SessionState,
        plan: V2DiagnosticPlan,
        step: V2PlanStep,
        outcome: dict[str, Any],
        branch: dict[str, Any] | None,
    ) -> AgentResponse:
        outcome_id = str(outcome.get("outcome_id") or "")
        evidence_ids = self.read_model.evidence_ids_for([
            outcome_id,
            str((branch or {}).get("branch_rule_id") or ""),
            step.action_id,
        ])
        if not evidence_ids or str(outcome.get("outcome_type") or "") != "verified_fix":
            return self._pending_validation(state, plan, step, "verified_fix_missing_evidence")
        state.status = "resolved"
        state.resolved_action_id = step.action_id
        state.which_check_solved = step.action_id
        state.resolution = str(outcome.get("summary") or outcome.get("root_cause_summary") or step.instruction)
        state.confidence = 0.9
        self._add_evidence(state, evidence_ids)
        state.metadata["verification"] = {
            "supported": True,
            "outcome_id": outcome_id,
            "outcome_type": "verified_fix",
            "activation_mode": str(outcome.get("activation_mode") or ""),
            "action_id": step.action_id,
            "branch_rule_id": str((branch or {}).get("branch_rule_id") or ""),
            "evidence_ids": evidence_ids,
            "runtime_confirmation": str(
                state.action_results.get(step.action_id) or ""
            ),
        }
        self.sessions.save(state)
        return self._response(
            state,
            "resolved",
            f"诊断闭环：{state.top_variant_label}\nKG_v2 已验证处理：{state.resolution}",
            resolution=state.resolution,
            confidence=state.confidence,
        )

    def _escalate(
        self,
        state: SessionState,
        plan: V2DiagnosticPlan | None,
        reason: str,
    ) -> AgentResponse:
        variant = self.read_model.get(state.top_variant_id) or {}
        family = self.read_model.get(state.top_family_id) or {}
        target = str(variant.get("escalation_target") or family.get("escalation_target") or "人工技术支持")
        state.status = "escalate"
        state.failure_type = reason
        state.escalation_target = target
        if plan is not None:
            self._add_evidence(state, plan.evidence_ids)
        state.metadata["evidence_pack"] = {
            "family_id": state.top_family_id,
            "variant_id": state.top_variant_id,
            "plan_id": state.plan_id,
            "completed_action_ids": list(state.action_results),
            "evidence_ids": list(state.evidence_ids),
        }
        self.sessions.save(state)
        return self._response(
            state,
            "escalate",
            f"KG_v2 诊断计划未能闭环：{state.top_variant_label}。建议携带已执行动作和证据升级至 {target}。",
            escalation_target=target,
            failure_type=reason,
        )

    def _ask_required_info(
        self,
        state: SessionState,
        plan: V2DiagnosticPlan,
        reason: str,
        selected: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        required = selected or self.read_model.required_info(plan.required_info_ids)
        required = sorted(required, key=lambda item: 0 if item.get("priority") == "high" else 1)
        required = required[: self.config.thresholds.ask_info_max_required_items]
        questions = [str(item.get("question") or item.get("why_required") or item.get("slot") or "") for item in required]
        questions = [item for item in questions if item]
        if not questions:
            questions = ["请补充更完整的故障现象或报错文本"]
        evidence_ids = [str(evidence_id) for item in required for evidence_id in item.get("evidence_ids") or []]
        self._add_evidence(state, evidence_ids)
        state.status = "ask_info"
        state.failure_type = reason
        state.required_data = questions
        state.metadata["required_info_ids"] = [str(item.get("required_info_id") or "") for item in required]
        self.sessions.save(state)
        return self._response(
            state,
            "ask_info",
            "KG_v2 计划需要补充以下信息后才能继续：" + "；".join(questions),
            required_data=questions,
        )

    def _initial_missing_required_info(
        self,
        query: str,
        plan: V2DiagnosticPlan,
    ) -> list[dict[str, Any]]:
        required = self.read_model.required_info(plan.required_info_ids)
        explicitly_missing = _contains_any(
            query,
            ("当前缺少", "仍缺少", "未提供", "需要补充", "没有提供"),
        )
        matched: list[dict[str, Any]] = []
        query_lower = query.lower()
        for item in required:
            priority = str(item.get("priority") or "").lower()
            blocks = list(item.get("blocks") or [])
            slot = str(item.get("slot") or "")
            signals = _slot_signals(slot)
            if explicitly_missing and any(signal in query_lower for signal in signals):
                matched.append(item)
                continue
            # High-priority information that selects two or more downstream
            # actions is branch context, not optional enrichment.  Enforce it
            # before execution regardless of whether the Variant was locked
            # by an exact keyword, a document, or another retrieval route.
            if (
                priority == "high"
                and len(blocks) >= 2
                and _required_info_selects_branch(item)
                and not _required_info_satisfied(query_lower, item)
            ):
                matched.append(item)
        if explicitly_missing and not matched:
            return required[: self.config.thresholds.ask_info_max_required_items]
        return matched[: self.config.thresholds.ask_info_max_required_items]

    def _matching_branch(
        self,
        step: V2PlanStep,
        outcome_type: str,
        user_message: str = "",
        routing_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        rules = self.read_model.branch_rules_for_step(step)
        matches = [
            rule for rule in rules
            if outcome_type in {str(x) for x in rule.get("trigger_outcome_types") or []}
        ]
        if not matches:
            return None
        signal_text = " ".join([
            str(user_message or ""),
            " ".join(
                f"{key}={value}"
                for key, value in sorted((routing_context or {}).items())
            ),
        ]).lower()
        signal_key = _branch_signal_key(signal_text)
        configured = [
            rule for rule in matches
            if (
                rule.get("match_any")
                or rule.get("match_all")
                or rule.get("match_all_groups")
                or rule.get("exclude_any")
            )
        ]
        scored: list[tuple[int, dict[str, Any]]] = []
        for rule in matches:
            excluded = any(
                _branch_signal_present(signal, signal_text, signal_key)
                for signal in rule.get("exclude_any") or []
                if str(signal)
            )
            if excluded:
                continue
            match_all = [
                str(signal).lower()
                for signal in rule.get("match_all") or []
                if str(signal)
            ]
            match_any = [
                str(signal).lower()
                for signal in rule.get("match_any") or []
                if str(signal)
            ]
            match_all_groups = [
                [str(signal).lower() for signal in group if str(signal)]
                for group in rule.get("match_all_groups") or []
                if isinstance(group, list) and group
            ]
            all_matched = not match_all or all(
                _branch_signal_present(signal, signal_text, signal_key)
                for signal in match_all
            )
            all_groups_matched = not match_all_groups or all(
                any(
                    _branch_signal_present(signal, signal_text, signal_key)
                    for signal in group
                )
                for group in match_all_groups
            )
            any_count = sum(
                _branch_signal_present(signal, signal_text, signal_key)
                for signal in match_any
            )
            code_matched = bool(
                str(rule.get("condition_code") or "").lower()
                and _branch_signal_present(
                    str(rule.get("condition_code") or ""),
                    signal_text,
                    signal_key,
                )
            )
            if not all_matched or not all_groups_matched:
                continue
            if match_any and not any_count and not code_matched:
                continue
            if match_all or match_all_groups or match_any or code_matched:
                scored.append((
                    (100 if code_matched else 0)
                    + 20 * len(match_all)
                    + 30 * len(match_all_groups)
                    + 10 * any_count,
                    rule,
                ))
        if scored:
            scored.sort(
                key=lambda item: (
                    -item[0],
                    int(item[1].get("priority") or 9999),
                    str(item[1].get("branch_rule_id") or ""),
                )
            )
            return scored[0][1]
        if configured:
            return {
                "_selection_status": "needs_condition",
                "candidate_branch_rule_ids": [
                    str(rule.get("branch_rule_id") or "") for rule in matches
                ],
                "conditions": [
                    str(rule.get("condition") or rule.get("condition_code") or "")
                    for rule in matches
                    if str(rule.get("condition") or rule.get("condition_code") or "")
                ],
            }
        matches.sort(key=lambda item: (int(item.get("priority") or 9999), str(item.get("branch_rule_id") or "")))
        return matches[0] if matches else None

    def _verified_fix_outcome(
        self, step: V2PlanStep, user_message: str = ""
    ) -> dict[str, Any] | None:
        for outcome in self.read_model.outcomes_for_step(step):
            if str(outcome.get("outcome_type") or "") == "verified_fix":
                if (
                    str(outcome.get("activation_mode") or "")
                    == "human_confirmed_runtime"
                    and not _activation_requirements_met(outcome, user_message)
                ):
                    continue
                evidence = self.read_model.evidence_ids_for([str(outcome.get("outcome_id") or "")])
                if evidence:
                    return outcome
        return None

    def _ask_branch_condition(
        self,
        state: SessionState,
        step: V2PlanStep,
        outcome_type: str,
        selection: dict[str, Any],
    ) -> AgentResponse:
        conditions = [
            str(item) for item in selection.get("conditions") or [] if str(item)
        ]
        question = (
            "请明确现场符合哪一种条件：" + "；".join(conditions)
            if conditions
            else "请补充能够区分下一诊断分支的明确现场信号。"
        )
        state.status = "ask_info"
        state.failure_type = "branch_condition_required"
        state.required_data = [question]
        state.metadata["pending_branch_condition_action_id"] = step.action_id
        state.metadata["pending_branch_outcome_type"] = outcome_type
        state.metadata["candidate_branch_rule_ids"] = list(
            selection.get("candidate_branch_rule_ids") or []
        )
        self.sessions.save(state)
        return self._response(
            state,
            "ask_info",
            "当前结果可进入多个互斥分支，不能仅按优先级猜测。",
            required_data=state.required_data,
        )

    def _follow_branch(
        self,
        state: SessionState,
        plan: V2DiagnosticPlan,
        current: V2PlanStep,
        outcome_type: str,
        branch: dict[str, Any],
    ) -> AgentResponse:
        state.metadata.setdefault("applied_branch_rule_ids", []).append(
            branch["branch_rule_id"]
        )
        self._add_evidence(state, branch.get("evidence_ids") or [])
        terminal = str(branch.get("terminal_status") or "continue")
        if terminal == "resolved":
            user_message = str(state.action_results.get(current.action_id) or "")
            verified = self._verified_fix_outcome(current, user_message)
            if verified is not None and outcome_type == "verified_fix":
                return self._resolve(state, plan, current, verified, branch)
            return self._pending_validation(
                state,
                plan,
                current,
                "branch_resolution_without_activated_verified_fix",
            )
        if terminal in {"unresolved", "escalated"}:
            return self._escalate(
                state, plan, f"branch_terminal:{terminal}"
            )
        target_trace_step_id = str(branch.get("to_trace_step_id") or "")
        target = self._step_by_trace_step(plan, target_trace_step_id)
        if target is None:
            return self._escalate(state, plan, "branch_target_not_in_plan")
        return self._present_step(state, plan, target)

    @staticmethod
    def _step_by_action(plan: V2DiagnosticPlan, action_id: str) -> V2PlanStep | None:
        return next((step for step in plan.steps if step.action_id == action_id), None)

    @staticmethod
    def _step_by_trace_step(plan: V2DiagnosticPlan, trace_step_id: str) -> V2PlanStep | None:
        return next((step for step in plan.steps if step.trace_step_id == trace_step_id), None)

    def _render_step(self, state: SessionState, plan: V2DiagnosticPlan, step: V2PlanStep) -> str:
        lines = [
            f"命中 KG_v2 故障：{state.top_family_label} / {state.top_variant_label}",
            f"诊断计划：{plan.source_type} {plan.plan_id}",
            f"当前动作（{step.ordinal}/{len(plan.steps)}）：{step.label}",
            step.instruction,
        ]
        if step.destructive or step.high_cost:
            lines.append("安全提示：该动作被 KG_v2 标记为高成本或高风险，仅在人工确认后执行。")
        if step.evidence_ids:
            lines.append("证据：" + "、".join(step.evidence_ids))
        return "\n".join(lines)

    def _response(self, state: SessionState, status: str, answer: str, **overrides: Any) -> AgentResponse:
        evidence_ids = [
            evidence_id for evidence_id in state.evidence_ids
            if self.read_model.has_object(evidence_id, "EvidenceItem")
        ]
        plan: V2DiagnosticPlan | None = None
        if state.top_family_id and state.top_variant_id:
            try:
                candidate_plan = self.read_model.compile_plan(state.top_family_id, state.top_variant_id)
                if not state.plan_id or candidate_plan.plan_id == state.plan_id:
                    plan = candidate_plan
            except KeyError:
                plan = None
        required_data = overrides.get("required_data", list(state.required_data))
        composed = self.answer_composer.compose(
            state=state,
            status=status,
            base_answer=answer,
            plan=plan,
            required_data=required_data,
        )
        evidence_pack = self.evidence_pack_builder.build(
            state=state,
            status=status,
            composed=composed,
            plan=plan,
            required_data=required_data,
        )
        facets = evidence_pack.payload["query_scope"]["facets"]
        composed.coverage["query_facets"] = facets
        composed.coverage["supported_query_facets"] = list(
            evidence_pack.payload["query_scope"]["supported_facets"]
        )
        composed.coverage["uncovered_query_facets"] = list(
            evidence_pack.payload["query_scope"]["unsupported_facets"]
        )
        composed.coverage["evidence_floor_met"] = bool(
            evidence_pack.payload["query_scope"]["evidence_floor_met"]
        )
        composed.coverage["grounded_item_count"] = len(
            evidence_pack.payload["query_scope"]["grounded_item_ids"]
        )
        composed.coverage["query_facets_complete"] = bool(
            composed.coverage["evidence_floor_met"]
            and not composed.coverage["uncovered_query_facets"]
        )
        composer_enabled = bool(
            self.config.read_llm.enabled
            and self.config.read_llm.answer_composer_enabled
        )
        if composer_enabled:
            llm_result = self.llm_answer_composer.compose(
                evidence_pack,
                deterministic_answer=composed.answer,
                deterministic_sections=composed.sections,
            )
            composed.answer = llm_result.answer
            composed.sections = llm_result.sections
            composer_metadata = llm_result.metadata
        else:
            composer_metadata = {
                "provider": self.config.read_llm.provider,
                "enabled": False,
                "attempted": False,
                "used": False,
                "fallback_used": False,
                "fallback_reason": (
                    "read_llm_disabled"
                    if not self.config.read_llm.enabled
                    else "answer_composer_disabled"
                ),
                "model": self.config.read_llm.model,
                "call_count": 0,
                "verification_errors": [],
            }
        gap_section, navigation_gaps = navigation_evidence_gap_section(
            model=self.read_model,
            state=state,
            evidence_pack=evidence_pack.payload,
        )
        composed.sections = insert_navigation_evidence_gap(
            composed.sections,
            gap_section,
        )
        if gap_section is not None:
            # This pass runs after the optional LLM organizer so the model
            # cannot accidentally hide a recalled-but-unclosed subtask.
            composed.answer = render_answer_sections(composed.sections)
        composed.coverage["navigation_evidence_gaps"] = navigation_gaps
        composed.coverage["navigation_evidence_gap_count"] = len(navigation_gaps)
        state.metadata["evidence_pack"] = evidence_pack.payload
        state.metadata["answer_composer"] = composer_metadata
        state.metadata["answer_coverage"] = composed.coverage
        state.metadata["sufficiency"] = composed.sufficiency
        evidence = self.read_model.evidence(evidence_ids)
        metadata = {key: value for key, value in state.metadata.items() if key != "input"}
        metadata["evidence"] = evidence
        metadata["runtime_invariants"] = {
            "identity_source": "KG_v2.FaultVariant",
            "candidate_source": "KG_v2.FaultFamily+FaultVariant",
            "plan_source": f"KG_v2.{state.plan_source_type}" if state.plan_source_type else "KG_v2",
            "evidence_source": "KG_v2.EvidenceItem",
            "legacy_graph_used": False,
        }
        observability = {
            "session_id": state.session_id,
            "agent_id": "O0",
            "status": status,
            "family_id": state.top_family_id,
            "variant_id": state.top_variant_id,
            "plan_id": state.plan_id,
            "current_action_id": state.current_action_id,
            "current_trace_step_id": state.current_trace_step_id,
            "top_error_id": state.top_variant_id,
            "which_check_solved": state.resolved_action_id,
            "retrieval_route": state.retrieval_route,
            "lock_status": state.lock_status,
            "failure_type": overrides.get("failure_type", state.failure_type),
        }
        self.sessions.save(state)
        chunk_ids = [
            chunk_id
            for section in composed.sections
            for chunk_id in section.chunk_ids
        ]
        section_evidence_ids = [
            evidence_id
            for section in composed.sections
            for evidence_id in section.evidence_ids
        ]
        return AgentResponse(
            schema_version=self.schema_version,
            session_id=state.session_id,
            status=status,  # type: ignore[arg-type]
            answer=composed.answer,
            required_data=required_data,
            current_check_id=state.current_action_id,
            current_check=state.current_check,
            resolution=overrides.get("resolution", state.resolution),
            confidence=float(overrides.get("confidence", state.confidence)),
            escalation_target=overrides.get("escalation_target", state.escalation_target),
            sources=list(
                dict.fromkeys(
                    [*evidence_ids, *section_evidence_ids, *chunk_ids]
                )
            ),
            failure_type=overrides.get("failure_type", state.failure_type),
            observability=observability,
            metadata=metadata,
            family_id=state.top_family_id,
            variant_id=state.top_variant_id,
            plan_id=state.plan_id,
            current_action_id=state.current_action_id,
            evidence_ids=list(evidence_ids),
            answer_sections=composed.sections,
        )

    def _add_evidence(self, state: SessionState, evidence_ids: Iterable[str]) -> None:
        for evidence_id in evidence_ids:
            item = str(evidence_id or "")
            if self.read_model.has_object(item, "EvidenceItem") and item not in state.evidence_ids:
                state.evidence_ids.append(item)

    @staticmethod
    def _input(payload: dict[str, Any] | DebugAgentInput) -> DebugAgentInput:
        if isinstance(payload, DebugAgentInput):
            return payload
        return DebugAgentInput(
            query=str(payload.get("query") or payload.get("original_query") or ""),
            interactive=bool(payload.get("interactive", True)),
            session=dict(payload.get("session") or {}),
            chat_history=list(payload.get("chat_history") or []),
            log_summary=dict(payload.get("log_summary") or {}),
            routing_context=dict(payload.get("routing_context") or {}),
            evidence_resources=list(payload.get("evidence_resources") or []),
        )


def _classify_outcome(text: str) -> str:
    lowered = str(text or "").lower()
    if _contains_any(lowered, ("不是根因", "并非根因", "排除该", "可以排除", "与此无关", "未证明是根因")):
        return "context_not_root_cause"
    if _contains_any(lowered, ("再次复现", "又复现", "继续复现", "重新复现", "复发")):
        return "recurred"
    if _contains_any(lowered, _NEGATIVE):
        return "ineffective"
    if _contains_any(lowered, ("暂时恢复", "临时恢复", "短暂恢复", "恢复后又")):
        return "partial_temporary"
    if _contains_any(lowered, ("缓解", "好转")):
        return "mitigation_observed"
    if _contains_any(lowered, _PENDING):
        return "pending_validation"
    if _contains_any(lowered, _SOLVED):
        return "verified_fix"
    if "复现" in lowered:
        return "recurred"
    return "diagnostic_method"


def _contains_any(text: str, values: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(value).lower() in lowered for value in values)


def _looks_like_incident_query(value: str) -> bool:
    text = str(value or "")
    lowered = text.lower()
    return bool(
        len(text.splitlines()) >= 3
        and (
            "trace:" in lowered
            or "调用栈" in text
            or " exception" in lowered
            or " error" in lowered
            or "0x" in lowered
        )
    )


def _activation_requirements_met(
    outcome: dict[str, Any], user_message: str
) -> bool:
    requirements = outcome.get("activation_requirements")
    groups = (
        requirements.get("all_of_groups")
        if isinstance(requirements, dict)
        else None
    )
    if not isinstance(groups, list) or not groups:
        return False
    lowered = str(user_message or "").lower()
    signal_key = _branch_signal_key(lowered)
    return all(
        isinstance(group, list)
        and bool(group)
        and any(
            _branch_signal_present(signal, lowered, signal_key)
            for signal in group
            if str(signal)
        )
        for group in groups
    )


def _branch_signal_key(value: Any) -> str:
    return "".join(
        character
        for character in str(value or "").lower()
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )


def _branch_signal_present(
    signal: Any, raw_text: str, normalized_text: str
) -> bool:
    raw_signal = str(signal or "").lower().strip()
    if not raw_signal:
        return False
    return (
        raw_signal in raw_text
        or _branch_signal_key(raw_signal) in normalized_text
    )


def _slot_signals(slot: str) -> tuple[str, ...]:
    mapping = {
        "log_package": ("日志", "dlog", "数据包"),
        "dmp_package": ("dmp", "dump"),
        "software_version": ("版本",),
        "error_phase": ("阶段", "发生时间"),
        "error_message": ("报错", "错误码", "错误代码"),
        "device_model": ("型号", "设备"),
        "ip_config": ("ip", "网段", "网络配置"),
        "repro_steps": ("复现", "操作步骤"),
        "sample_image": ("图片", "截图", "样本"),
        "program_file": ("程序", "配方", "模板"),
        "environment": ("环境", "电源", "内存", "磁盘"),
    }
    return mapping.get(slot, (slot.lower(),))


def _required_info_satisfied(query: str, item: dict[str, Any]) -> bool:
    """Whether the initial query already contains branch-selecting context.

    A Variant-label match only identifies the fault family.  It does not
    answer questions such as "fans off or running", "which version", or
    "which IP topology".  RequiredInfoSpec remains the source of truth; this
    helper only recognizes concrete observations already present in the query.
    """

    lowered = str(query or "").lower()
    slot = str(item.get("slot") or "").lower()
    question = str(item.get("question") or "").lower()
    signals = set(_slot_signals(slot))
    if slot == "environment":
        signals.update((
            "风扇",
            "指示灯",
            "debug灯",
            "debug 灯",
            "屏幕",
            "显示",
            "蜂鸣",
            "无反应",
            "黑屏",
            "重启",
            "死机",
            "供电",
            "电源",
            "皮带",
            "传感器",
            "感应灯",
            "出板口",
            "接地",
            "走线",
            "负载",
        ))
    elif slot == "driver_context":
        signals.update(("驱动", "设备管理器"))
    elif slot == "production_constraint":
        signals.update(("停机", "备件", "审批", "允许更换"))
    elif slot == "other":
        # "other" is only satisfiable through a meaningful phrase that is
        # actually present in the RequiredInfo question.
        signals.update(
            token
            for token in ("双轨", "大板", "负载", "现场")
            if token in question
        )
    return any(
        signal
        and signal in question
        and signal in lowered
        for signal in signals
    )


def _required_info_selects_branch(item: dict[str, Any]) -> bool:
    """Identify RequiredInfo that changes the first diagnostic branch.

    Priority and blocked-action count alone are insufficient: a log bundle or
    source file can improve later diagnosis without preventing a safe first
    check.  Initial execution is paused only when the KG explicitly describes
    the information as selecting a path/branch and asks for a distinguishing
    observation.
    """

    why = str(item.get("why_required") or "").lower()
    question = str(item.get("question") or "").lower()
    selects_path = _contains_any(
        why,
        ("分支", "路径", "优先走", "定位是", "区分"),
    )
    asks_observation = _contains_any(
        question,
        (
            "是否",
            "状态",
            "表现",
            "现象",
            "类型",
            "接口",
            "拓扑",
            "灯",
            "屏幕",
            "蜂鸣",
        ),
    )
    return selects_path and asks_observation


def _strip_missing_statements(text: str) -> str:
    import re

    cleaned = re.sub(
        r"(?:当前|仍|还)?(?:缺少|未提供|没有提供)[^。；\n]*[。；]?",
        "",
        str(text or ""),
    )
    return cleaned.strip(" \n，。；")
