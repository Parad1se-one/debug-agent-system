"""Evidence-constrained, fail-open answer organization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from debug_agent_system.agents.read.evidence_answer import render_answer_sections
from debug_agent_system.agents.read.evidence_pack import EvidencePack
from debug_agent_system.core.contracts import AnswerSection


class JSONAnswerClient(Protocol):
    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class LLMCompositionResult:
    used: bool
    answer: str
    sections: list[AnswerSection]
    metadata: dict[str, Any]


class EvidenceAnswerVerifier:
    """Verify references and coverage before any model plan is rendered."""

    schema_version = "debug_agent_system.llm_answer_composition.v2"
    allowed_titles = {
        "known": "根据资料可知",
        "diagnostic_steps": "建议排查顺序",
        "document_guidance": "文档建议的处理路径",
        "conditions": "适用条件与不同情况",
        "uncertainty": "尚不能确认",
        "required_info": "需要补充的信息",
    }

    def verify(
        self,
        output: dict[str, Any],
        pack: EvidencePack,
    ) -> tuple[list[AnswerSection] | None, list[str]]:
        errors: list[str] = []
        if output.get("schema_version") != self.schema_version:
            errors.append("invalid_schema_version")
        raw_sections = output.get("answer_sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            errors.append("missing_answer_sections")
            return None, errors
        allowed_ids = set(pack.source_items)
        selected: list[str] = []
        sections: list[AnswerSection] = []
        seen_required = False
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                errors.append("invalid_section")
                continue
            section_type = str(raw_section.get("section_type") or "")
            if section_type not in self.allowed_titles:
                errors.append(f"invalid_section_type:{section_type}")
                continue
            if seen_required and section_type != "required_info":
                errors.append("section_after_required_info")
            if section_type == "required_info":
                seen_required = True
            item_ids = raw_section.get("source_item_ids")
            if not isinstance(item_ids, list) or not item_ids:
                errors.append(f"missing_source_item_ids:{section_type}")
                continue
            normalized_ids = [str(item_id or "") for item_id in item_ids]
            unknown = [item_id for item_id in normalized_ids if item_id not in allowed_ids]
            if unknown:
                errors.append("unknown_source_item_ids:" + ",".join(unknown))
                continue
            incompatible = [
                item_id
                for item_id in normalized_ids
                if str(
                    pack.source_items[item_id].get(
                        "original_section_type"
                    )
                    or ""
                )
                != section_type
            ]
            if incompatible:
                errors.append(
                    "section_type_mismatch:" + ",".join(incompatible)
                )
                continue
            duplicate = [item_id for item_id in normalized_ids if item_id in selected]
            if duplicate:
                errors.append("duplicate_source_item_ids:" + ",".join(duplicate))
                continue
            selected.extend(normalized_ids)
            canonical_items = [
                _canonical_item(pack.source_items[item_id])
                for item_id in normalized_ids
            ]
            sections.append(_section(
                section_type,
                self.allowed_titles[section_type],
                canonical_items,
            ))
        required = {
            item_id
            for item_id, item in pack.source_items.items()
            if str(item.get("selection_class") or "") == "required"
        }
        missing = sorted(required - set(selected))
        if missing:
            errors.append("missing_required_source_items:" + ",".join(missing))
        grounded_selected = set(
            pack.payload["query_scope"].get("grounded_item_ids") or []
        ) & set(selected)
        if (
            pack.payload["query_scope"].get("evidence_floor_met")
            and not grounded_selected
        ):
            errors.append("no_grounded_content_selected")
        actual_facets = {
            str(facet.get("facet_id") or "")
            for facet in pack.payload["query_scope"]["facets"]
            if set(facet.get("supported_item_ids") or []) & set(selected)
        }
        expected_facets = set(pack.payload["query_scope"]["supported_facets"])
        if expected_facets - actual_facets:
            errors.append(
                "missing_supported_facets:"
                + ",".join(sorted(expected_facets - actual_facets))
            )
        claimed_facets = {
            str(value) for value in output.get("covered_query_facets") or []
        }
        if claimed_facets != actual_facets:
            errors.append("covered_query_facets_mismatch")
        claimed_uncovered = {
            str(value) for value in output.get("uncovered_query_facets") or []
        }
        expected_uncovered = set(
            pack.payload["query_scope"]["unsupported_facets"]
        )
        if claimed_uncovered != expected_uncovered:
            errors.append("uncovered_query_facets_mismatch")
        if errors:
            return None, errors
        if pack.source_section is not None:
            filtered_source_section = _filter_source_section(
                pack.source_section,
                [pack.source_items[item_id] for item_id in selected],
            )
            if filtered_source_section is not None:
                sections.append(filtered_source_section)
        return sections, []


class DeepSeekEvidenceAnswerComposer:
    """Ask DeepSeek for grouping/order only; render canonical local facts."""

    system_prompt = """
你是 KG_v2 读侧答案编排器。你的唯一任务是把 Evidence Pack.source_items 中已有的
item_id 按原 section 类型分组和排序。不能改写事实，不能添加结论、命令、来源、图片或
执行结果。

必须只返回一个 JSON 对象，严格使用以下结构：
{
  "schema_version": "debug_agent_system.llm_answer_composition.v2",
  "answer_sections": [
    {
      "section_type": "known|diagnostic_steps|document_guidance|conditions|uncertainty|required_info",
      "source_item_ids": ["answer-item:..."]
    }
  ],
  "covered_query_facets": ["Evidence Pack 中已支持的 facet_id"],
  "uncovered_query_facets": ["Evidence Pack 中未支持的 facet_id"]
}

规则：
1. 每个 selection_class=required 的 source_item 必须且只能出现一次，不能虚构 item_id。
2. selection_class=optional 的条目只在直接支持用户任务、补充分支或必要上下文时选择；
   弱相关、重复或会让主线偏移的 optional 条目应省略。
3. source_item 只能放入其 original_section_type 对应的 section_type。
4. covered_query_facets 必须逐字复制 query_scope.supported_facets；
   uncovered_query_facets 必须逐字复制 query_scope.unsupported_facets。
5. required_info section 必须在所有已有回答之后；不要输出空 section。
6. 按 query_scope.task_model.deliverable 选择组织方式：procedure 强调顺序，
   comparison 强调同维度对比，configuration/specification 强调实体和参数范围。
7. 不要输出 Markdown、解释、标题、items、text、evidence_ids、chunk_ids 或其他字段。
""".strip()

    def __init__(
        self,
        *,
        client: JSONAnswerClient | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/beta/chat/completions",
        timeout_seconds: int = 60,
    ) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.verifier = EvidenceAnswerVerifier()

    def compose(
        self,
        pack: EvidencePack,
        *,
        deterministic_answer: str,
        deterministic_sections: list[AnswerSection],
    ) -> LLMCompositionResult:
        base_metadata = {
            "enabled": True,
            "attempted": False,
            "used": False,
            "fallback_used": True,
            "fallback_reason": "",
            "model": self.model,
            "call_count": 0,
            "verification_errors": [],
        }
        if not pack.eligible_for_llm:
            base_metadata["fallback_reason"] = pack.fallback_reason
            return LLMCompositionResult(
                False, deterministic_answer, deterministic_sections, base_metadata
            )
        client = self.client
        if client is None:
            # Lazy import avoids a runtime/adapters package initialization
            # cycle: the Tool Harness imports DebugAgentSystem.
            from debug_agent_system.adapters.deepseek_read.client import (
                DeepSeekReadClient,
            )

            client = DeepSeekReadClient(
                model=self.model,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(pack.payload, ensure_ascii=False),
            },
        ]
        base_metadata["attempted"] = True
        base_metadata["call_count"] = 1
        try:
            output = client.complete_json(messages=messages)
            sections, errors = self.verifier.verify(output, pack)
            if errors or sections is None:
                base_metadata["fallback_reason"] = "verification_failed"
                base_metadata["verification_errors"] = errors
                return LLMCompositionResult(
                    False,
                    deterministic_answer,
                    deterministic_sections,
                    base_metadata,
                )
            answer = render_answer_sections(sections)
            base_metadata.update({
                "used": True,
                "fallback_used": False,
                "fallback_reason": "",
                "covered_query_facets": output.get("covered_query_facets") or [],
                "selected_item_count": sum(
                    len(section.get("source_item_ids") or [])
                    for section in output.get("answer_sections") or []
                ),
                "required_item_count": len(
                    pack.payload.get("selection_policy", {}).get(
                        "required_item_ids", []
                    )
                ),
                "optional_item_count": len(
                    pack.payload.get("selection_policy", {}).get(
                        "optional_item_ids", []
                    )
                ),
            })
            return LLMCompositionResult(True, answer, sections, base_metadata)
        except Exception as exc:  # fail-open is the explicit runtime contract
            base_metadata["fallback_reason"] = (
                f"{type(exc).__name__}:{str(exc)[:300]}"
            )
            return LLMCompositionResult(
                False, deterministic_answer, deterministic_sections, base_metadata
            )


def _canonical_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {
            "item_id",
            "mandatory",
            "selection_class",
            "original_section_type",
            "original_section_title",
            "original_section_order",
            "original_item_order",
        }
    }


def _filter_source_section(
    source_section: AnswerSection,
    selected_items: list[dict[str, Any]],
) -> AnswerSection | None:
    selected_evidence = {
        str(value)
        for item in selected_items
        for value in item.get("evidence_ids") or []
    }
    selected_chunks = {
        str(value)
        for item in selected_items
        for value in item.get("chunk_ids") or []
    }
    selected_sources = {
        str(value)
        for item in selected_items
        for value in item.get("sources") or []
    }
    items = [
        dict(item)
        for item in source_section.items
        if (
            set(str(value) for value in item.get("evidence_ids") or [])
            & selected_evidence
            or set(str(value) for value in item.get("chunk_ids") or [])
            & selected_chunks
            or str(item.get("text") or "") in selected_sources
        )
    ]
    if not items:
        return None
    return _section(
        source_section.section_type,
        source_section.title,
        items,
    )


def _section(
    section_type: str,
    title: str,
    items: list[dict[str, Any]],
) -> AnswerSection:
    return AnswerSection(
        section_type=section_type,
        title=title,
        items=items,
        evidence_ids=_dedupe(
            evidence_id
            for item in items
            for evidence_id in item.get("evidence_ids") or []
        ),
        chunk_ids=_dedupe(
            chunk_id
            for item in items
            for chunk_id in item.get("chunk_ids") or []
        ),
    )


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "")
        if item and item not in result:
            result.append(item)
    return result
