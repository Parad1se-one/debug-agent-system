"""Evidence-constrained Codex organization with deterministic verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from debug_agent_system.agents.read.evidence_answer import (
    render_answer_sections,
)
from debug_agent_system.agents.read.evidence_pack import EvidencePack
from debug_agent_system.core.contracts import AnswerSection


_CONTENT_SECTION_TYPES = {
    "known",
    "diagnostic_steps",
    "document_guidance",
    "conditions",
}
_FIXED_SECTION_TYPES = {"uncertainty", "required_info"}


class JSONAnswerClient(Protocol):
    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class CodexCompositionResult:
    used: bool
    answer: str
    sections: list[AnswerSection]
    metadata: dict[str, Any]


class CodexEvidenceAnswerVerifier:
    """Verify source closure, section compatibility and facet coverage."""

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
            return None, [*errors, "missing_answer_sections"]
        allowed_ids = set(pack.source_items)
        selected: list[str] = []
        sections: list[AnswerSection] = []
        seen_required_info = False
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                errors.append("invalid_section")
                continue
            section_type = str(raw_section.get("section_type") or "")
            if section_type not in self.allowed_titles:
                errors.append(f"invalid_section_type:{section_type}")
                continue
            if seen_required_info and section_type != "required_info":
                errors.append("section_after_required_info")
            if section_type == "required_info":
                seen_required_info = True
            item_ids = raw_section.get("source_item_ids")
            if not isinstance(item_ids, list) or not item_ids:
                errors.append(f"missing_source_item_ids:{section_type}")
                continue
            normalized_ids = [str(item_id or "") for item_id in item_ids]
            unknown = [
                item_id for item_id in normalized_ids
                if item_id not in allowed_ids
            ]
            if unknown:
                errors.append(
                    "unknown_source_item_ids:" + ",".join(unknown)
                )
                continue
            incompatible = [
                item_id
                for item_id in normalized_ids
                if not _section_compatible(
                    pack.source_items[item_id],
                    section_type,
                )
            ]
            if incompatible:
                errors.append(
                    "section_type_mismatch:" + ",".join(incompatible)
                )
                continue
            duplicate = [
                item_id for item_id in normalized_ids if item_id in selected
            ]
            if duplicate:
                errors.append(
                    "duplicate_source_item_ids:" + ",".join(duplicate)
                )
                continue
            selected.extend(normalized_ids)
            sections.append(_section(
                section_type,
                self.allowed_titles[section_type],
                [
                    _canonical_item(pack.source_items[item_id])
                    for item_id in normalized_ids
                ],
            ))
        required = {
            item_id
            for item_id, item in pack.source_items.items()
            if str(item.get("selection_class") or "") == "required"
        }
        missing = sorted(required - set(selected))
        if missing:
            errors.append(
                "missing_required_source_items:" + ",".join(missing)
            )
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
        expected_facets = set(
            pack.payload["query_scope"]["supported_facets"]
        )
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
            str(value)
            for value in output.get("uncovered_query_facets") or []
        }
        expected_uncovered = set(
            pack.payload["query_scope"]["unsupported_facets"]
        )
        if claimed_uncovered != expected_uncovered:
            errors.append("uncovered_query_facets_mismatch")
        if errors:
            return None, errors
        if pack.source_section is not None:
            source_section = _filter_source_section(
                pack.source_section,
                [pack.source_items[item_id] for item_id in selected],
            )
            if source_section is not None:
                sections.append(source_section)
        return sections, []


class CodexEvidenceAnswerComposer:
    """Let Codex select/order evidence; render only canonical local facts."""

    system_prompt = """
你是 KG_v2 读侧的 Codex 证据编排器。你收到的是一个封闭 Evidence Pack。
你的工作不是自由回答，而是为了让答案更清楚地完成以下任务：
- 按用户 query_scope 的任务类型组织主线；
- 覆盖每一个已有证据支持的必要 facet；
- 保留重要前置条件、分支差异、风险说明、图片和来源；
- 删除与问题无关或重复的 optional 条目；
- 把 required_info 放在已有可回答内容之后。

你只能选择并排序 source_items 中已有的 item_id。不能改写事实，不能新增命令、结论、
来源、图片、执行结果或“已解决”声明。正文由本地确定性渲染器生成。

只返回一个严格的 json 对象（不要使用代码块）：
{
  "schema_version": "debug_agent_system.llm_answer_composition.v2",
  "answer_sections": [
    {
      "section_type": "known|diagnostic_steps|document_guidance|conditions|uncertainty|required_info",
      "source_item_ids": ["answer-item:..."]
    }
  ],
  "covered_query_facets": ["逐字复制 Evidence Pack.query_scope.supported_facets"],
  "uncovered_query_facets": ["逐字复制 Evidence Pack.query_scope.unsupported_facets"]
}

硬约束：
1. selection_class=required 的条目必须且只能出现一次。
2. optional 仅在直接支持任务、补充分支/安全前置条件或媒体说明时选择。
3. uncertainty 和 required_info 必须留在 original_section_type；其余有证据正文可以在
   known、diagnostic_steps、document_guidance、conditions 之间重新分组。
4. procedure 按实际操作依赖排序；comparison 先共同前提再按条件分开；
   configuration/specification 先对象范围再参数和验证方式。
5. 不输出空 section；required_info 必须最后；不输出任何额外字段或 Markdown。
""".strip()

    def __init__(
        self,
        *,
        client: JSONAnswerClient | None = None,
        model: str = "gpt-5.3-codex",
        base_url: str = "",
        timeout_seconds: int = 60,
        env_file: str | Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.env_file = env_file
        self.verifier = CodexEvidenceAnswerVerifier()

    def compose(
        self,
        pack: EvidencePack,
        *,
        deterministic_answer: str,
        deterministic_sections: list[AnswerSection],
    ) -> CodexCompositionResult:
        metadata: dict[str, Any] = {
            "provider": "codex",
            "enabled": True,
            "attempted": False,
            "used": False,
            "fallback_used": True,
            "fallback_reason": "",
            "model": self.model,
            "call_count": 0,
            "verification_errors": [],
            "evidence_mode": "closed_pack_canonical_render",
        }
        if not pack.eligible_for_llm:
            metadata["fallback_reason"] = pack.fallback_reason
            return CodexCompositionResult(
                False,
                deterministic_answer,
                deterministic_sections,
                metadata,
            )
        client = self.client
        if client is None:
            # Lazy import avoids runtime -> adapter -> runtime initialization.
            from debug_agent_system.adapters.codex_read.client import (
                CodexReadClient,
            )

            client = CodexReadClient(
                model=self.model,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
                env_file=self.env_file,
            )
        request_payload = {
            **pack.payload,
            "output_format": "json",
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(request_payload, ensure_ascii=False),
            },
        ]
        metadata["attempted"] = True
        metadata["call_count"] = 1
        try:
            output = client.complete_json(messages=messages)
            sections, errors = self.verifier.verify(output, pack)
            if errors or sections is None:
                metadata["fallback_reason"] = "verification_failed"
                metadata["verification_errors"] = errors
                return CodexCompositionResult(
                    False,
                    deterministic_answer,
                    deterministic_sections,
                    metadata,
                )
            answer = render_answer_sections(sections)
            metadata.update({
                "used": True,
                "fallback_used": False,
                "covered_query_facets": (
                    output.get("covered_query_facets") or []
                ),
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
            usage = getattr(client, "last_usage", {})
            if isinstance(usage, dict) and usage:
                metadata["usage"] = dict(usage)
            request_id = str(getattr(client, "last_request_id", "") or "")
            if request_id:
                metadata["request_id"] = request_id
            return CodexCompositionResult(True, answer, sections, metadata)
        except Exception as exc:  # fail-open is the explicit runtime contract
            metadata["fallback_reason"] = (
                f"{type(exc).__name__}:{str(exc)[:200]}"
            )
            return CodexCompositionResult(
                False,
                deterministic_answer,
                deterministic_sections,
                metadata,
            )


def _section_compatible(item: dict[str, Any], section_type: str) -> bool:
    original = str(item.get("original_section_type") or "")
    if original in _FIXED_SECTION_TYPES:
        return original == section_type
    if original in _CONTENT_SECTION_TYPES:
        return section_type in _CONTENT_SECTION_TYPES
    return original == section_type


def _canonical_item(item: dict[str, Any]) -> dict[str, Any]:
    internal = {
        "item_id",
        "mandatory",
        "selection_class",
        "original_section_type",
        "original_section_title",
        "original_section_order",
        "original_item_order",
    }
    return {key: value for key, value in item.items() if key not in internal}


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
            value
            for item in items
            for value in item.get("evidence_ids") or []
        ),
        chunk_ids=_dedupe(
            value
            for item in items
            for value in item.get("chunk_ids") or []
        ),
    )


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "")
        if item and item not in result:
            result.append(item)
    return result


__all__ = [
    "CodexCompositionResult",
    "CodexEvidenceAnswerComposer",
    "CodexEvidenceAnswerVerifier",
]
