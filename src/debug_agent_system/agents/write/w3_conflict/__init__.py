"""W3 deterministic conflict classification for schema-valid write candidates."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any

from debug_agent_system.knowledge_v2.builders import infer_action_role, infer_required_info_slot
from debug_agent_system.knowledge_v2.compat import (
    _canonicalize_action_candidate,
    _canonicalize_family_label,
    _drop_action_candidate,
    _summary_for_family,
)
from debug_agent_system.knowledge_v2.contracts import (
    ACTION_ROLES,
    APPROVED_FAMILY_LABELS,
    FAMILY_SUBSYSTEM_EXPECTED,
    INTERNAL_REQUIRED_INFO_SLOTS,
    V2_PRIMARY_KEYS,
    make_family_id,
    make_id,
    trim_text,
)
from debug_agent_system.knowledge_v2.validator import validate_graph
from debug_agent_system.agents.write.w7_trace.trace_compiler import TraceCompiler

SLOT_SYNONYMS = {
    "DLOG": "log_package",
    "dlog": "log_package",
    "dmp": "log_package",
    "DMP": "log_package",
    "dump": "log_package",
    "诊断数据包": "log_package",
    "诊断数据": "log_package",
    "日志包": "log_package",
    "日志": "log_package",
    "WPR": "log_package",
    "PoolMon": "log_package",
    "软件版本": "software_version",
    "主程序版本": "software_version",
    "算法包版本": "software_version",
    "驱动版本": "software_version",
    "显卡驱动版本": "software_version",
    "版本": "software_version",
    "蓝屏代码": "error_message",
    "BugCheck": "error_message",
    "PTE": "error_message",
    "PFN": "error_message",
    "网卡角色": "ip_config",
    "过滤驱动": "ip_config",
    "相机网卡": "ip_config",
    "复发": "repro_steps",
    "复现": "repro_steps",
    "生产约束": "environment",
    "内存测试": "environment",
}

_DOCUMENT_ACTION_VERBS = (
    "检查", "查看", "确认", "观察", "测试", "验证", "导出", "收集", "记录", "分析",
    "比较", "尝试", "设置", "调整", "优化", "恢复", "限制", "清理", "清洁", "拆下",
    "涂抹", "安装", "更换", "拔插", "重启", "关闭", "开启", "联系", "送修", "测量",
    "使用", "运行", "进入", "删除", "卸载", "修复", "更新", "整理",
)
_DOCUMENT_META_LABELS = {"注意", "提示", "说明", "备注", "标准建议", "第一步", "第二步", "第三步", "第四步", "第五步"}


def _document_action_label(label: str, summary: str) -> str:
    """Recover the executable clause from document headings.

    W9 intentionally preserves source structure, so W10 can emit labels such
    as ``第一步`` or ``注意``.  W3 is the first semantic-normalization stage;
    it must turn those wrappers into an action without inventing content.
    """

    clean_label = str(label or "").strip()
    clean_summary = str(summary or clean_label).strip()
    if clean_label.startswith("手摸"):
        return trim_text(f"触摸检查{clean_label[2:]}", 60)
    if clean_label in _DOCUMENT_META_LABELS or re.fullmatch(r"第[一二三四五六七八九十\d]+步", clean_label):
        body = re.sub(r"^(?:第[一二三四五六七八九十\d]+步|注意|提示|说明|备注|标准建议)\s*[：:]\s*", "", clean_summary).strip()
        if "建议" in body:
            body = body.split("建议", 1)[1].strip(" ：:，,")
        body = re.sub(r"^(?:建议|请)\s*", "", body).strip()
        if clean_label == "标准建议" and body and not any(verb in body for verb in _DOCUMENT_ACTION_VERBS):
            body = f"检查机箱风向为{body}"
        if body:
            for sep in ("。", "；", ";"):
                if sep in body:
                    body = body.split(sep, 1)[0].strip()
            return trim_text(body, 60)
    return clean_label


def _document_non_action_fragment(label: str, summary: str) -> bool:
    """Drop tool names/list items while keeping genuine atomic actions."""

    clean_label = str(label or "").strip()
    clean_summary = str(summary or clean_label).strip()
    if not clean_label:
        return True
    if clean_label in _DOCUMENT_META_LABELS:
        return True
    has_verb = any(verb in clean_label for verb in _DOCUMENT_ACTION_VERBS)
    if has_verb:
        return False
    # A standalone product/tool token (OCCT, AIDA64, DDU...) is evidence for
    # the surrounding action, not an action of its own.
    if clean_label == clean_summary and re.fullmatch(r"[A-Za-z0-9_.+\-/（）()等 ]{2,32}", clean_label):
        return True
    # W9 list children such as "散热器鳍片" inherit the parent verb in the
    # source document.  Without that parent verb they are not executable.
    if clean_label == clean_summary and len(clean_label) <= 24 and not any(mark in clean_label for mark in ("是否", "？", "?")):
        return True
    return False


def _nodes(candidate: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [node for node in candidate.get("nodes") or [] if node.get("type") == node_type]


def _semantic_id(prefix: str, value: str) -> str:
    raw = str(value or "")
    # Families are canonical ontology objects shared across all write paths.
    # Keep exactly the same id convention as W2/W10 and the active graph.
    if prefix == "family":
        return make_family_id(raw)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return make_id(prefix, f"{raw}-{digest}")


def _clean_v2_variant_label(value: str) -> str:
    label = trim_text(value, 80).strip(" ：:，,。？?")
    for prefix in ("我这个现场", "现场反馈", "客户反馈", "问题反馈"):
        if label.startswith(prefix):
            label = label[len(prefix):].strip(" ：:，,。？?")
    for suffix in ("是什么问题", "怎么处理", "怎么办", "如何处理", "咋回事"):
        if label.endswith(suffix):
            label = label[:-len(suffix)].strip(" ：:，,。？?")
    return trim_text(label, 60)


def _strip_action_result_suffix(label: str) -> str:
    """Keep the atomic action in the action node; outcomes live separately."""

    clean = str(label or "").strip()
    if "后" not in clean:
        return clean
    action, result = clean.split("后", 1)
    if action and any(marker in result for marker in (
        "恢复正常", "问题解决", "故障消除", "未再出现", "不再出现", "仍然", "依然", "无效", "失败", "闪退", "报错",
    )):
        return action.strip(" ，,。；;")
    return clean


def _more_specific_approved_family(family_label: str, context: str) -> str:
    """Refine an approved umbrella family using strong variant-level signals."""

    text = str(context or "")
    lowered = text.lower()
    if (
        family_label in {"用户配置加载失败", "主程序/系统异常", "算法/程序调优异常"}
        and re.search(r"(?<![a-z])mes(?![a-z])", lowered)
        and any(marker in text for marker in ("报错", "上传", "过站", "工单", "站位"))
    ):
        return "MES 过站异常"
    if (
        family_label in {"软件卡死无响应", "主程序/系统异常", "用户配置加载失败"}
        and "spc" in lowered
        and any(marker in text for marker in ("打不开", "无法打开", "单板分析", "页面"))
    ):
        return "SPC 页面无法打开"
    if "黑屏" in text and any(marker in text for marker in ("鼠标", "光标", "键盘", "显示器")):
        return "工控机黑屏无显示"
    if "复判站" in text and "出图" in text and any(marker in text for marker in ("慢", "卡", "延迟", "秒")):
        return "复判站出图慢"
    if family_label != "软件卡死无响应" and any(marker in text for marker in ("不进板", "进板失败", "进版口不进去")):
        return "进板失败"
    if family_label != "软件卡死无响应" and any(marker in text for marker in ("不出板", "出板失败", "没有送出动作")):
        return "出板失败"
    if "复判站" in text and "加载" in text:
        return "复判站加载板卡异常"
    if any(marker in lowered for marker in ("显存不足", "爆显存", "cuda out of memory", "cuda oom")):
        return "CUDA 计算设备不可用"
    if any(marker in text for marker in ("计算时间变长", "CT过长", "CT 太长", "出结果CT")):
        return "CT 时间异常增加"
    if any(marker in lowered for marker in ("com口", "com7", "com8", "串口")) and any(
        marker in text for marker in ("拒绝访问", "断联", "无法发送", "连接失败")
    ):
        return "外设连接不稳定"
    if "导入插件" in text and any(marker in lowered for marker in ("json", "解析", "导入")):
        return "程序板卡加载失败"
    if "气压" in text and any(marker in text for marker in ("顶板", "顶升", "不拍照", "拍照流程")):
        return "气压异常"
    return family_label


def _node_ids(candidate: dict[str, Any], node_type: str) -> set[str]:
    pk = {"Error": "error_id", "DiagnosticCheck": "check_id", "Solution": "solution_id", "Site": "site_id", "SoftwareVersion": "version_id"}.get(node_type, "id")
    return {str(node.get(pk) or node.get("id") or "") for node in _nodes(candidate, node_type) if str(node.get(pk) or node.get("id") or "")}


def _has_condition_context(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("sites") or candidate.get("versions") or candidate.get("devices") or _nodes(candidate, "Site") or _nodes(candidate, "SoftwareVersion"))


def _diagnostic_outcomes(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node for node in _nodes(candidate, "DiagnosticOutcome")]
    for item in candidate.get("diagnostic_outcomes") or []:
        if isinstance(item, dict):
            out.append(item)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in out:
        key = str(item.get("outcome_id") or item.get("action_label") or repr(item))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _outcome_reason_codes(candidate: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for outcome in _diagnostic_outcomes(candidate):
        outcome_type = str(outcome.get("outcome_type") or "")
        if outcome_type:
            codes.append(f"outcome:{outcome_type}")
        if outcome_type and outcome_type != "verified_fix":
            codes.append("non_verified_outcome_not_resolved_by")
        if outcome.get("high_cost"):
            codes.append("high_cost_requires_human")
        if outcome.get("destructive"):
            codes.append("destructive_requires_human")
    return codes


def _base(candidate: dict[str, Any], decision: str, conflict_type: str, reason_codes: list[str], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "decision": decision,
        "conflict_type": conflict_type,
        "reason_codes": sorted(set(reason_codes)),
        "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
        "existing_error_id": str((existing or {}).get("error_id") or ""),
        "requires_human": True,
        "candidate_category": candidate.get("category") or "",
        "node_diff": {
            "diagnostic_checks": sorted(_node_ids(candidate, "DiagnosticCheck")),
            "solutions": sorted(_node_ids(candidate, "Solution")),
            "sites": sorted(_node_ids(candidate, "Site")),
            "software_versions": sorted(_node_ids(candidate, "SoftwareVersion")),
            "edge_count": len(candidate.get("edges") or []),
        },
    }
    if existing:
        out["existing_match"] = {
            "error_id": existing.get("error_id"),
            "label": existing.get("label"),
            "score": existing.get("score"),
            "route": existing.get("route"),
            "evidence": existing.get("evidence") or [],
        }
    return out


class ConflictResolutionAgent:
    """W3: classify new-vs-existing KG conflicts into review-grade buckets."""

    def resolve(self, candidate: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
        existing = existing or candidate.get("matched_existing_error")
        schema_valid = bool(candidate.get("schema_valid"))
        has_evidence = bool(candidate.get("evidence_ids") or candidate.get("source_offsets"))
        has_check = bool(_nodes(candidate, "DiagnosticCheck"))
        has_solution = bool(_nodes(candidate, "Solution"))
        reason_codes: list[str] = []
        if schema_valid:
            reason_codes.append("schema_valid")
        else:
            reason_codes.extend(["schema_invalid", *[str(x) for x in candidate.get("schema_issues") or []]])
        if has_evidence:
            reason_codes.append("has_evidence")
        else:
            reason_codes.append("missing_evidence")
        if has_check:
            reason_codes.append("has_diagnostic_check")
        if has_solution:
            reason_codes.append("has_solution")
        if _has_condition_context(candidate):
            reason_codes.append("has_condition_context")
        reason_codes.extend(_outcome_reason_codes(candidate))

        if not schema_valid or not has_evidence or (not has_check and not has_solution):
            return _base(candidate, "Insufficient", "insufficient_evidence", reason_codes, existing if isinstance(existing, dict) else None)

        if not existing:
            return _base(candidate, "Agree", "new_error", [*reason_codes, "no_existing_match"], None)

        score = float(existing.get("score") or 0.0)
        if score:
            reason_codes.append(f"kg_match_score:{score:.2f}")
        else:
            reason_codes.append("matched_existing_error")

        check_count = len(_nodes(candidate, "DiagnosticCheck"))
        solution_count = len(_nodes(candidate, "Solution"))
        has_conditions = _has_condition_context(candidate)
        if has_conditions:
            return _base(candidate, "Refine", "condition_branch", [*reason_codes, "condition_specific_candidate"], existing)
        if check_count > 1:
            return _base(candidate, "Refine", "check_order", [*reason_codes, "multi_step_check_chain"], existing)
        if solution_count > 1:
            return _base(candidate, "Contradict", "solution_conflict", [*reason_codes, "multiple_candidate_solutions"], existing)
        return _base(candidate, "Agree", "condition_branch", [*reason_codes, "compatible_with_existing_error"], existing)

    def resolve_required_info(self, required_info_candidate: dict[str, Any]) -> dict[str, Any]:
        candidate = dict(required_info_candidate)
        text = " ".join(str(candidate.get(k) or "") for k in ("label", "question", "why_required", "condition"))
        request = candidate.get("source_request") if isinstance(candidate.get("source_request"), dict) else {}
        text = f"{text} {request.get('text') or ''}"
        original_slot = str(candidate.get("slot") or "other")
        slot = original_slot
        reason_codes: list[str] = []
        for token, normalized in SLOT_SYNONYMS.items():
            if token in text and (original_slot == "other" or original_slot == normalized):
                slot = normalized
                reason_codes.append(f"slot_synonym:{token}->{normalized}")
                break
        condition = str(candidate.get("condition") or "")
        lowered = text.lower()
        service_restart = any(k in text for k in (
            "重启相机服务", "重启服务", "重启主程序", "重启软件", "重启程序",
            "相机服务重启", "服务重启", "主程序重启", "软件重启", "程序重启",
        ))
        manual_reboot_action = any(k in text for k in ("重启设备", "设备重启", "重启机器", "机器重启", "重启电脑", "电脑重启")) and not any(
            k in text for k in ("自动重启", "突然重启", "无故重启", "异常重启", "莫名重启")
        )
        explicit_reboot_failure = any(k in text for k in ("自动重启", "突然重启", "无故重启", "异常重启", "莫名重启"))
        reboot_failure = "重启" in text and not service_restart and not manual_reboot_action and (
            explicit_reboot_failure or any(k in text for k in ("蓝屏", "工控机", "系统", "死机", "黑屏", "BugCheck", "bugcheck"))
        )
        if slot in {"other", "log_package", "error_message", "error_phase"} and (
            "蓝屏" in text or reboot_failure or "dmp" in lowered or "dump" in lowered
        ):
            if slot == "other":
                slot = "log_package"
            condition = "dmp"
            reason_codes.append("condition_branch:dmp")
        elif slot in {"other", "log_package", "error_message", "error_phase"} and ("初始化" in text or "启动" in text or "startup" in lowered or "init" in lowered):
            if slot == "log_package":
                condition = "startup/init log"
            elif slot == "other":
                slot = "log_package"
                condition = "startup/init log"
            else:
                condition = condition or "startup/init phase"
            reason_codes.append("condition_branch:startup_init")
        elif any(k in text for k in ("相机", "控制器", "IP", "ip", "网段", "ping", "Ping")):
            if slot == "other":
                slot = "ip_config"
            if slot == "ip_config":
                condition = "camera/controller network config"
            reason_codes.append("condition_branch:network_config")
        candidate["slot"] = slot
        candidate["condition"] = condition
        if not candidate.get("evidence_message_ids"):
            reason_codes.append("missing_evidence")
        if not candidate.get("target_error_id"):
            reason_codes.append("review_only:no_target_error")
        if slot == "other":
            reason_codes.append("slot_other")
        if slot != original_slot:
            reason_codes.append(f"slot_changed:{original_slot}->{slot}")

        if not candidate.get("evidence_message_ids") or slot == "other":
            decision = "Insufficient"
            conflict_type = "insufficient_evidence"
        elif slot != original_slot or condition:
            decision = "Refine"
            conflict_type = "condition_branch" if condition else "required_info_slot"
        else:
            decision = "Agree"
            conflict_type = "required_info_slot"
        return {
            "decision": decision,
            "existing_error_id": str(candidate.get("target_error_id") or ""),
            "conflict_type": conflict_type,
            "requires_human": True,
            "reason_codes": sorted(set(reason_codes or ["deterministic_required_info_check"])),
            "candidate": candidate,
        }

    def normalize_v2_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """Normalize a W2/W10 KG v2 bundle without inventing new evidence.

        W3 owns deterministic structural refinement.  It canonicalizes family
        and variant labels, removes exact duplicate actions/required-info
        records, rewrites all affected references, and then re-validates the
        graph.  Semantic quality and admission remain W4 responsibilities.
        """

        out = deepcopy(bundle) if isinstance(bundle, dict) else {}
        objects = out.get("objects") if isinstance(out.get("objects"), dict) else {}
        normalized_objects = {
            obj_type: [deepcopy(item) for item in objects.get(obj_type) or [] if isinstance(item, dict)]
            for obj_type in V2_PRIMARY_KEYS
        }
        relations = [deepcopy(item) for item in out.get("relations") or [] if isinstance(item, dict)]
        changes: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        review_flags: list[str] = []
        family_scope_candidates = [
            str(item) for item in out.get("family_scope_candidates") or [] if str(item).strip()
        ]
        approved_scopes = sorted(set(item for item in family_scope_candidates if item in APPROVED_FAMILY_LABELS))
        if len(approved_scopes) > 1:
            review_flags.append("ambiguous_family_scope")

        normalized_objects["FaultVariant"], support_variant_map = self._drop_support_only_variants(normalized_objects, changes)
        id_map.update(support_variant_map)

        family_context = self._v2_bundle_text(out, normalized_objects)
        normalized_objects["FaultFamily"], family_map = self._normalize_v2_families(
            normalized_objects["FaultFamily"], family_context, changes
        )
        id_map.update(family_map)
        self._rewrite_object_references(normalized_objects, family_map)

        normalized_objects["FaultVariant"], variant_map = self._normalize_v2_variants(
            normalized_objects["FaultVariant"], normalized_objects["FaultFamily"], changes
        )
        id_map.update(variant_map)
        self._rewrite_object_references(normalized_objects, variant_map)

        (
            normalized_objects["DiagnosticAction"],
            promoted_required_info,
            promoted_action_map,
            promoted_relations,
        ) = self._promote_document_questions_to_required_info(
            normalized_objects["DiagnosticAction"],
            normalized_objects["FaultVariant"],
            normalized_objects["DiagnosticTrace"],
            normalized_objects["SourceCase"],
            normalized_objects["EvidenceItem"],
            relations,
            changes,
        )
        normalized_objects["RequiredInfoSpec"].extend(promoted_required_info)
        relations.extend(promoted_relations)
        id_map.update(promoted_action_map)
        self._rewrite_object_references(normalized_objects, promoted_action_map)

        normalized_objects["DiagnosticAction"], action_map = self._normalize_v2_actions(
            normalized_objects["DiagnosticAction"], normalized_objects["FaultFamily"], changes
        )
        id_map.update(action_map)
        self._rewrite_object_references(normalized_objects, action_map)

        normalized_objects["ActionOutcome"], outcome_map = self._normalize_v2_outcomes(
            normalized_objects["ActionOutcome"], changes
        )
        id_map.update(outcome_map)
        self._rewrite_object_references(normalized_objects, outcome_map)

        normalized_objects["RequiredInfoSpec"], required_map = self._normalize_v2_required_info(
            normalized_objects["RequiredInfoSpec"], changes
        )
        id_map.update(required_map)
        self._rewrite_object_references(normalized_objects, required_map)

        orphan_map = self._drop_orphan_document_fault_mappings(normalized_objects, changes)
        id_map.update(orphan_map)
        self._rewrite_object_references(normalized_objects, orphan_map)

        self._rebuild_trace_execution_semantics(normalized_objects, changes)

        relations = self._rewrite_relations(relations, id_map)
        relations = self._drop_relations_with_missing_objects(normalized_objects, relations, changes)
        relations = self._replace_trace_execution_relations(normalized_objects, relations)
        issues = validate_graph(normalized_objects, relations)
        output_mode = str((out.get("strategy") or {}).get("kg_output_mode") or "") if isinstance(out.get("strategy"), dict) else ""
        has_fault_candidate = any(normalized_objects.get(key) for key in (
            "FaultFamily", "FaultVariant", "DiagnosticAction", "RequiredInfoSpec"
        ))
        has_document_layer = bool(normalized_objects.get("KnowledgeDocument")) and bool(normalized_objects.get("KnowledgeSection"))
        if has_fault_candidate and (not normalized_objects["FaultFamily"] or not normalized_objects["FaultVariant"]):
            issues = sorted(set([*issues, "empty_v2_fault_bundle"]))
        elif not has_fault_candidate and not has_document_layer:
            issues = sorted(set([*issues, "empty_v2_document_bundle"]))
        source_report = deepcopy(out.get("report")) if isinstance(out.get("report"), dict) else {}
        report = {
            **source_report,
            "family_count": len(normalized_objects["FaultFamily"]),
            "variant_count": len(normalized_objects["FaultVariant"]),
            "action_count": len(normalized_objects["DiagnosticAction"]),
            "required_info_count": len(normalized_objects["RequiredInfoSpec"]),
            "trace_count": len(normalized_objects["DiagnosticTrace"]),
            "outcome_count": len(normalized_objects["ActionOutcome"]),
            "evidence_count": len(normalized_objects["EvidenceItem"]),
            "source_case_count": len(normalized_objects["SourceCase"]),
            "document_count": len(normalized_objects.get("KnowledgeDocument") or []),
            "section_count": len(normalized_objects.get("KnowledgeSection") or []),
            "procedure_step_count": len(normalized_objects.get("ProcedureStep") or []),
        }
        out.update({
            "type": "W3NormalizedKGV2Bundle",
            "candidate_id": str(out.get("candidate_id") or out.get("bundle_id") or ""),
            "family_ids": [str(item.get("family_id") or "") for item in normalized_objects["FaultFamily"]],
            "variant_ids": [str(item.get("variant_id") or "") for item in normalized_objects["FaultVariant"]],
            "objects": normalized_objects,
            "relations": relations,
            "schema_valid": not issues,
            "schema_issues": issues,
            "report": report,
            "w3_refinement": {
                "agent_id": "W3",
                "source_type": str(bundle.get("type") or "") if isinstance(bundle, dict) else "",
                "source_schema_valid": bool(bundle.get("schema_valid")) if isinstance(bundle, dict) else False,
                "change_count": len(changes),
                "changes": changes,
                "review_flags": review_flags,
                "family_scope_candidates": family_scope_candidates,
                "source_report": source_report,
            },
        })
        return out

    @staticmethod
    def _rebuild_trace_execution_semantics(
        objects: dict[str, list[dict[str, Any]]], changes: list[dict[str, Any]]
    ) -> None:
        """Compatibility wrapper around the shared W7 TraceCompiler."""

        TraceCompiler.rebuild_execution_objects(objects, changes)

    @staticmethod
    def _branch_destination(outcome_type: str, next_step_id: str) -> tuple[str, str]:
        if outcome_type == "verified_fix":
            return "", "resolved"
        if next_step_id:
            return next_step_id, "continue"
        if outcome_type in {"partial_temporary", "mitigation_observed"}:
            return "", "monitoring"
        return "", "unresolved"

    @staticmethod
    def _replace_trace_execution_relations(
        objects: dict[str, list[dict[str, Any]]], relations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return TraceCompiler.replace_execution_relations(objects, relations)

    @staticmethod
    def _drop_orphan_document_fault_mappings(
        objects: dict[str, list[dict[str, Any]]], changes: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Remove document mappings whose executable/support payload vanished.

        W3 may correctly remove headings or tool names from DiagnosticAction.
        The corresponding document section must remain, but an empty Variant or
        Family must not survive as a fake fault node.
        """

        if not objects.get("KnowledgeDocument"):
            return {}
        active_variant_ids = {
            str(item.get("variant_id") or "")
            for obj_type in ("DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace")
            for item in objects.get(obj_type) or []
            if str(item.get("variant_id") or "")
        }
        id_map: dict[str, str] = {}
        kept_variants: list[dict[str, Any]] = []
        for variant in objects.get("FaultVariant") or []:
            variant_id = str(variant.get("variant_id") or "")
            if variant_id and variant_id not in active_variant_ids:
                id_map[variant_id] = ""
                changes.append({"kind": "orphan_document_variant_removed", "variant_id": variant_id, "label": variant.get("label") or ""})
            else:
                kept_variants.append(variant)
        objects["FaultVariant"] = kept_variants

        active_family_ids = {
            str(item.get("family_id") or "")
            for obj_type in ("FaultVariant", "DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace")
            for item in objects.get(obj_type) or []
            if str(item.get("family_id") or "")
        }
        kept_families: list[dict[str, Any]] = []
        for family in objects.get("FaultFamily") or []:
            family_id = str(family.get("family_id") or "")
            if family_id and family_id not in active_family_ids:
                id_map[family_id] = ""
                changes.append({"kind": "orphan_document_family_removed", "family_id": family_id, "label": family.get("label") or ""})
            else:
                kept_families.append(family)
        objects["FaultFamily"] = kept_families
        return id_map

    @staticmethod
    def _drop_relations_with_missing_objects(
        objects: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        changes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        known_ids = {
            str(item.get(V2_PRIMARY_KEYS.get(obj_type, "id")) or "")
            for obj_type, items in objects.items()
            for item in items or []
            if isinstance(item, dict) and str(item.get(V2_PRIMARY_KEYS.get(obj_type, "id")) or "")
        }
        kept: list[dict[str, Any]] = []
        for relation in relations:
            source = str(relation.get("from") or "")
            target = str(relation.get("to") or "")
            if source not in known_ids or target not in known_ids:
                changes.append({"kind": "orphan_relation_removed", "from": source, "to": target, "relation": relation.get("relation") or ""})
                continue
            kept.append(relation)
        return kept

    @staticmethod
    def _drop_support_only_variants(
        objects: dict[str, list[dict[str, Any]]], changes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        active_variant_ids = {
            str(item.get("variant_id") or "")
            for obj_type in ("DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace")
            for item in objects.get(obj_type) or []
            if str(item.get("variant_id") or "")
        }
        support_markers = ("参考", "原因", "预防", "误区", "背景", "文档目的", "注意事项", "术语")
        kept: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        for variant in objects.get("FaultVariant") or []:
            variant_id = str(variant.get("variant_id") or "")
            label = str(variant.get("label") or "")
            if variant_id not in active_variant_ids and any(marker in label for marker in support_markers):
                id_map[variant_id] = ""
                changes.append({"kind": "support_only_variant_removed", "variant_id": variant_id, "label": label})
                continue
            kept.append(variant)
        return kept, id_map

    @staticmethod
    def _v2_bundle_text(bundle: dict[str, Any], objects: dict[str, list[dict[str, Any]]]) -> str:
        parts = [str(bundle.get("source_doc_title") or "")]
        for item in objects.get("FaultFamily") or []:
            parts.extend(str(item.get(key) or "") for key in ("label", "summary", "scenario"))
        for item in objects.get("FaultVariant") or []:
            parts.extend(str(item.get(key) or "") for key in ("label", "summary", "scenario"))
        # SourceCase/Trace summaries are the focused current-case context.  Do
        # not use the full W1 episode here: long chat windows often contain
        # several unrelated faults and would make deterministic refinement
        # copy context from a neighbouring case.
        for item in objects.get("SourceCase") or []:
            parts.extend(str(item.get(key) or "") for key in ("title", "summary"))
        for item in objects.get("DiagnosticTrace") or []:
            parts.append(str(item.get("summary") or ""))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _normalize_v2_families(
        families: list[dict[str, Any]], context: str, changes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        out: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        seen: dict[str, dict[str, Any]] = {}
        for family in families:
            item = deepcopy(family)
            old_id = str(item.get("family_id") or "")
            old_label = str(item.get("label") or "").strip()
            if old_label in APPROVED_FAMILY_LABELS:
                canonical = _more_specific_approved_family(old_label, context)
            else:
                canonical = _canonicalize_family_label(
                    old_label, str(item.get("subsystem") or ""), str(item.get("category") or ""), context
                )
            canonical = trim_text(canonical or old_label, 40)
            if canonical and canonical != old_label:
                changes.append({"kind": "family_canonicalized", "from": old_label, "to": canonical})
            item["label"] = canonical
            canonical_summary = _summary_for_family(canonical)
            if canonical_summary and canonical_summary != str(item.get("summary") or ""):
                changes.append({
                    "kind": "family_summary_aligned",
                    "family": canonical,
                    "from": str(item.get("summary") or ""),
                    "to": canonical_summary,
                })
                item["summary"] = canonical_summary
            expected_subsystem = FAMILY_SUBSYSTEM_EXPECTED.get(canonical)
            if expected_subsystem and item.get("subsystem") != expected_subsystem:
                changes.append({
                    "kind": "family_subsystem_aligned",
                    "family": canonical,
                    "from": str(item.get("subsystem") or ""),
                    "to": expected_subsystem,
                })
                item["subsystem"] = expected_subsystem
            new_id = _semantic_id("family", canonical or old_id)
            if old_id:
                id_map[old_id] = new_id
            item["family_id"] = new_id
            if new_id in seen:
                changes.append({"kind": "family_exact_merged", "from_id": old_id, "to_id": new_id})
                continue
            seen[new_id] = item
            out.append(item)
        return out, id_map

    @staticmethod
    def _normalize_v2_variants(
        variants: list[dict[str, Any]],
        families: list[dict[str, Any]],
        changes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        family_labels = {str(item.get("family_id") or ""): str(item.get("label") or "") for item in families}
        out: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        seen: dict[tuple[str, str], str] = {}
        for variant in variants:
            item = deepcopy(variant)
            old_id = str(item.get("variant_id") or "")
            family_id = str(item.get("family_id") or "")
            old_label = str(item.get("label") or "").strip()
            label = _clean_v2_variant_label(old_label)
            family_label = family_labels.get(family_id, "")
            if label == family_label:
                summary_label = _clean_v2_variant_label(str(item.get("summary") or ""))
                if summary_label and summary_label != family_label:
                    label = summary_label
            if label != old_label:
                changes.append({"kind": "variant_label_normalized", "from": old_label, "to": label})
            item["label"] = label
            new_id = _semantic_id("variant", f"{family_id}:{label or old_id}")
            key = (family_id, " ".join(label.lower().split()))
            if old_id:
                id_map[old_id] = seen.get(key, new_id)
            if key in seen:
                changes.append({"kind": "variant_exact_merged", "from_id": old_id, "to_id": seen[key]})
                continue
            seen[key] = new_id
            item["variant_id"] = new_id
            out.append(item)
        return out, id_map

    @staticmethod
    def _normalize_v2_actions(
        actions: list[dict[str, Any]], families: list[dict[str, Any]], changes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        family_labels = {str(item.get("family_id") or ""): str(item.get("label") or "") for item in families}
        out: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        seen: dict[tuple[str, str, str], str] = {}
        order_by_variant: dict[str, int] = {}
        for action in actions:
            item = deepcopy(action)
            old_id = str(item.get("action_id") or "")
            family_id = str(item.get("family_id") or "")
            variant_id = str(item.get("variant_id") or "")
            old_label = str(item.get("label") or "").strip()
            summary = str(item.get("summary") or old_label).strip()
            atomic_label = _strip_action_result_suffix(old_label)
            if atomic_label != old_label:
                changes.append({"kind": "action_result_suffix_removed", "from": old_label, "to": atomic_label})
                old_label = atomic_label
            if str(item.get("source_kind") or "") in {
                "raw_doc", "sop", "hybrid"
            }:
                recovered_label = _document_action_label(old_label, summary)
                if recovered_label != old_label:
                    changes.append({"kind": "document_action_heading_unwrapped", "from": old_label, "to": recovered_label})
                    old_label = recovered_label
                if _document_non_action_fragment(old_label, summary):
                    if old_id:
                        id_map[old_id] = ""
                    changes.append({"kind": "document_non_action_fragment_removed", "action_id": old_id, "label": old_label})
                    continue
            if len(old_label) < 4 or old_label in set(_DOCUMENT_ACTION_VERBS):
                if old_id:
                    id_map[old_id] = ""
                changes.append({"kind": "underspecified_action_removed", "action_id": old_id, "label": old_label})
                continue
            role = str(item.get("action_role") or "")
            if role not in ACTION_ROLES:
                role = infer_action_role(f"{old_label} {summary}")
            label, summary, role = _canonicalize_action_candidate(
                old_label, summary, role, family_labels.get(family_id, "")
            )
            label = trim_text(label, 60)
            summary = trim_text(summary or label, 180)
            if _drop_action_candidate(label, summary, family_labels.get(family_id, "")):
                if old_id:
                    id_map[old_id] = ""
                changes.append({"kind": "action_noise_removed", "action_id": old_id, "label": old_label})
                continue
            key = (variant_id, " ".join(label.lower().split()), role)
            new_id = seen.get(key) or _semantic_id("action", f"{variant_id}:{role}:{label}")
            if old_id:
                id_map[old_id] = new_id
            if key in seen:
                changes.append({"kind": "action_exact_merged", "from_id": old_id, "to_id": new_id})
                continue
            seen[key] = new_id
            if label != old_label:
                changes.append({"kind": "action_label_normalized", "from": old_label, "to": label})
            order_by_variant[variant_id] = order_by_variant.get(variant_id, 0) + 1
            item.update({
                "action_id": new_id,
                "label": label,
                "summary": summary,
                "action_role": role,
                "step_order": order_by_variant[variant_id],
            })
            out.append(item)
        return out, id_map

    @staticmethod
    def _promote_document_questions_to_required_info(
        actions: list[dict[str, Any]],
        variants: list[dict[str, Any]],
        traces: list[dict[str, Any]],
        source_cases: list[dict[str, Any]],
        evidence_items: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        changes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        """Move document questions out of the executable action chain.

        A sentence such as ``是否刚开机温度就偏高？`` is valuable, but it is
        a C-gate information request rather than a DiagnosticAction.  Evidence
        remains bound to the same source case and variant.
        """

        variant_by_id = {str(item.get("variant_id") or ""): item for item in variants}
        case_by_variant: dict[str, list[str]] = {}
        evidence_by_case: dict[str, list[str]] = {}
        for relation in relations:
            source = str(relation.get("from") or "")
            target = str(relation.get("to") or "")
            rel_type = str(relation.get("relation") or "")
            if rel_type == "supports" and source.startswith("case:") and target in variant_by_id:
                case_by_variant.setdefault(target, []).append(source)
            elif rel_type == "evidences" and source.startswith("evidence:") and target.startswith("case:"):
                evidence_by_case.setdefault(target, []).append(source)
        trace_evidence_by_variant: dict[str, list[str]] = {}
        for trace in traces:
            variant_id = str(trace.get("variant_id") or "")
            trace_evidence_by_variant.setdefault(variant_id, []).extend(
                str(value) for value in trace.get("evidence_ids") or [] if str(value)
            )
        known_case_ids = {str(item.get("case_id") or "") for item in source_cases}
        known_evidence_ids = {str(item.get("evidence_id") or "") for item in evidence_items}

        kept: list[dict[str, Any]] = []
        promoted: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        new_relations: list[dict[str, Any]] = []
        for action in actions:
            item = deepcopy(action)
            old_id = str(item.get("action_id") or "")
            label = str(item.get("label") or "").strip()
            summary = str(item.get("summary") or label).strip()
            source_kind = str(item.get("source_kind") or "")
            is_question = (
                source_kind in {"raw_doc", "sop", "hybrid"}
                and (label.startswith(("是否", "有无", "能否", "请问")) or label.endswith(("？", "?")))
            )
            if not is_question:
                kept.append(item)
                continue
            variant_id = str(item.get("variant_id") or "")
            family_id = str(item.get("family_id") or "")
            variant = variant_by_id.get(variant_id) or {}
            variant_label = str(variant.get("label") or "当前故障")
            question = trim_text(summary if summary.endswith(("？", "?")) else label, 100)
            slot = infer_required_info_slot(question)
            if slot == "other" and any(token in question for token in ("温度", "°C", "风扇", "散热", "硅脂", "积尘", "环境", "压力测试")):
                slot = "environment"
            if slot == "other" and any(token in question for token in ("降频", "卡顿", "关机", "重启", "触发时机")):
                slot = "error_phase"
            case_ids = [value for value in case_by_variant.get(variant_id, []) if value in known_case_ids]
            evidence_ids = list(dict.fromkeys([
                *[value for value in trace_evidence_by_variant.get(variant_id, []) if value in known_evidence_ids],
                *[
                    evidence_id
                    for case_id in case_ids
                    for evidence_id in evidence_by_case.get(case_id, [])
                    if evidence_id in known_evidence_ids
                ],
            ]))
            required_id = _semantic_id("required-info", f"{variant_id}:{slot}:{question}")
            promoted.append({
                "required_info_id": required_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "slot": slot,
                "question": question,
                "why_required": trim_text(f"用于确认{variant_label}的发生条件并缩小诊断范围。", 160),
                "condition": "",
                "blocks": [trim_text(variant_label, 60)],
                "priority": "medium",
                "evidence_ids": evidence_ids,
            })
            new_relations.append({"from": variant_id, "to": required_id, "relation": "has_required_info"})
            for case_id in case_ids:
                new_relations.append({"from": case_id, "to": required_id, "relation": "supports"})
            for evidence_id in evidence_ids:
                new_relations.append({"from": evidence_id, "to": required_id, "relation": "evidences"})
            if old_id:
                id_map[old_id] = ""
            changes.append({
                "kind": "document_question_promoted_to_required_info",
                "action_id": old_id,
                "required_info_id": required_id,
                "question": question,
            })
        return kept, promoted, id_map, new_relations

    @staticmethod
    def _normalize_v2_required_info(
        required_info: list[dict[str, Any]], changes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        out: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        seen: dict[tuple[str, str, str], str] = {}
        for required in required_info:
            item = deepcopy(required)
            old_id = str(item.get("required_info_id") or "")
            old_slot = str(item.get("slot") or "other")
            question = trim_text(item.get("question") or "", 100)
            slot = old_slot if old_slot in INTERNAL_REQUIRED_INFO_SLOTS else infer_required_info_slot(
                f"{old_slot} {question} {item.get('why_required') or ''}"
            )
            if slot == "other" and any(token in f"{question} {item.get('why_required') or ''}" for token in ("温度", "°C", "风扇", "散热", "硅脂", "积尘", "BIOS", "压力测试")):
                slot = "environment"
            if slot == "other" and any(token in f"{question} {item.get('why_required') or ''}" for token in ("时机", "发生时间", "触发阶段", "降频", "卡顿", "关机", "重启")):
                slot = "error_phase"
            if slot not in INTERNAL_REQUIRED_INFO_SLOTS:
                slot = "other"
            if slot != old_slot:
                changes.append({"kind": "required_info_slot_normalized", "from": old_slot, "to": slot})
            family_id = str(item.get("family_id") or "")
            variant_id = str(item.get("variant_id") or "")
            key = (variant_id or family_id, slot, " ".join(question.lower().split()))
            new_id = seen.get(key) or _semantic_id("required-info", f"{variant_id or family_id}:{slot}:{question}")
            if old_id:
                id_map[old_id] = new_id
            if key in seen:
                existing = next(row for row in out if row.get("required_info_id") == new_id)
                existing["evidence_ids"] = list(dict.fromkeys([
                    *[str(value) for value in existing.get("evidence_ids") or [] if str(value)],
                    *[str(value) for value in item.get("evidence_ids") or [] if str(value)],
                ]))
                changes.append({"kind": "required_info_exact_merged", "from_id": old_id, "to_id": new_id})
                continue
            seen[key] = new_id
            item.update({"required_info_id": new_id, "slot": slot, "question": question})
            out.append(item)
        return out, id_map

    @staticmethod
    def _normalize_v2_outcomes(
        outcomes: list[dict[str, Any]], changes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        out: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        seen: dict[tuple[str, str, str, str], str] = {}
        for outcome in outcomes:
            item = deepcopy(outcome)
            old_id = str(item.get("outcome_id") or "")
            action_id = str(item.get("action_id") or "")
            outcome_type = str(item.get("outcome_type") or "")
            source_case_id = str(item.get("source_case_id") or "")
            summary = trim_text(item.get("summary") or "", 200)
            if outcome_type == "verified_fix":
                lowered = summary.lower().strip()
                normalized_type = outcome_type
                if lowered in {
                    "camera_capture_chain", "software_version_change", "startup/init", "startup/init phase",
                    "root_cause", "verified", "fixed", "resolved",
                }:
                    normalized_type = "pending_validation"
                elif any(marker in summary for marker in ("未复现", "无法复现", "未发现异常", "仍需验证", "待验证", "应该能解决", "可能解决")):
                    normalized_type = "pending_validation"
                elif any(marker in summary for marker in ("非根因", "不是根因", "非直接原因", "无关")):
                    normalized_type = "context_not_root_cause"
                elif summary.strip() in {"就重启了", "又重启了", "仍然重启", "再次出现", "问题复发"}:
                    normalized_type = "recurred"
                if normalized_type != outcome_type:
                    changes.append({
                        "kind": "outcome_type_evidence_normalized",
                        "outcome_id": old_id,
                        "from": outcome_type,
                        "to": normalized_type,
                        "summary": summary,
                    })
                    outcome_type = normalized_type
            key = (source_case_id, action_id, outcome_type, " ".join(summary.lower().split()))
            new_id = seen.get(key) or _semantic_id(
                "outcome", f"{source_case_id}:{action_id}:{outcome_type}:{summary}"
            )
            if old_id:
                id_map[old_id] = new_id
            if key in seen:
                existing = next(row for row in out if row.get("outcome_id") == new_id)
                existing["evidence_ids"] = list(dict.fromkeys([
                    *[str(value) for value in existing.get("evidence_ids") or [] if str(value)],
                    *[str(value) for value in item.get("evidence_ids") or [] if str(value)],
                ]))
                changes.append({"kind": "outcome_exact_merged", "from_id": old_id, "to_id": new_id})
                continue
            seen[key] = new_id
            item.update({"outcome_id": new_id, "outcome_type": outcome_type, "summary": summary})
            out.append(item)
        return out, id_map

    @staticmethod
    def _rewrite_object_references(objects: dict[str, list[dict[str, Any]]], id_map: dict[str, str]) -> None:
        if not id_map:
            return
        scalar_keys = (
            "family_id", "variant_id", "action_id", "source_case_id", "trace_id",
            "trace_step_id", "from_trace_step_id", "to_trace_step_id",
        )
        list_keys = (
            "recommended_action_ids",
            "actual_action_ids",
            "ordered_action_ids",
            "ineffective_action_ids",
            "high_cost_action_ids",
            "source_trace_ids",
            "source_outcome_ids",
            "outcome_ids",
            "observation_ids",
            "branch_rule_ids",
            "evidence_ids",
        )
        for items in objects.values():
            for item in items:
                for key in scalar_keys:
                    value = str(item.get(key) or "")
                    if value in id_map:
                        item[key] = id_map[value]
                for key in list_keys:
                    if not isinstance(item.get(key), list):
                        continue
                    rewritten: list[str] = []
                    for value in item[key]:
                        mapped = id_map.get(str(value), str(value))
                        if mapped and mapped not in rewritten:
                            rewritten.append(mapped)
                    item[key] = rewritten

    @staticmethod
    def _rewrite_relations(relations: list[dict[str, Any]], id_map: dict[str, str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in relations:
            source = id_map.get(str(relation.get("from") or ""), str(relation.get("from") or ""))
            target = id_map.get(str(relation.get("to") or ""), str(relation.get("to") or ""))
            rel_type = str(relation.get("relation") or "")
            key = (source, target, rel_type)
            if not source or not target or not rel_type or key in seen:
                continue
            seen.add(key)
            out.append({**relation, "from": source, "to": target})
        return out
