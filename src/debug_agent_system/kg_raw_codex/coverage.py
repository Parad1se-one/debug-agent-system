"""Deterministic closure contract for the independent KG_v2+raw pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from debug_agent_system.knowledge_v2.query_scope import analyze_query_scope


_OBJECT_SPLIT = re.compile(r"(?:或者|或是|或|以及|并且|并|和|与|、|/)")
_OBJECT_NOISE = re.compile(
    r"^(?:用|使用|进行|执行|如何|怎样|怎么|应该|应当|需要|先|再|然后|"
    r"同时|分别|相关|对应|问题|操作|步骤|方法|流程|的|对)+"
)
_PURPOSE_BOUNDARY = re.compile(
    r"(?:用于|以便|从而|来)?(?:排查|诊断|检查|确认|验证|测试|观察)"
)
_SAFETY_OPERATIONS = {
    "备份", "还原", "恢复", "重装", "格式化", "删除", "卸载", "更换",
    "刷写", "升级",
}
_STORAGE_OBJECTS = ("磁盘", "硬盘", "分区", "盘符", "卷", "U盘", "镜像")
_DESTRUCTIVE_STORAGE_TERMS = (
    "格式化", "清空", "删除分区", "重分区", "重新分区", "转换分区",
    "初始化磁盘", "擦除",
)
_EXPLICIT_FALLBACK_REQUEST = re.compile(
    r"(?:失败后|仍(?:然)?(?:不|无法)|还是(?:不|无法)|"
    r"(?:若|如果|当).{0,24}(?:失败|无法|不能).{0,24}(?:怎么办|如何|怎样|下一步))"
)


@dataclass(frozen=True, slots=True)
class RequiredFacet:
    """One auditable answer-closure obligation."""

    facet_id: str
    kind: str
    label: str
    match_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "kind": self.kind,
            "label": self.label,
            "match_terms": list(self.match_terms),
            "required_for_closure": True,
        }


@dataclass(frozen=True, slots=True)
class ProcedureVariantRequirement:
    """One sibling procedure path discovered in an answer-closing source."""

    source_path: str
    source_label: str
    external_urls: tuple[str, ...] = ()
    external_artifact_unverified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_label": self.source_label,
            "external_urls": list(self.external_urls),
            "external_artifact_unverified": (
                self.external_artifact_unverified
            ),
        }


@dataclass(frozen=True, slots=True)
class AnswerScope:
    """A query-derived contract that limits how far an answer may expand."""

    request_kind: str
    requested_operations: tuple[str, ...]
    context_operations: tuple[str, ...]
    requested_objects: tuple[str, ...]
    branch_conditions: tuple[str, ...]
    named_tools: tuple[str, ...]
    detail_policy: str
    max_fallback_depth: int
    allow_system_repair_commands: bool
    allow_boot_repair_commands: bool
    allow_destructive_storage_commands: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_kind": self.request_kind,
            "requested_operations": list(self.requested_operations),
            "context_operations": list(self.context_operations),
            "requested_objects": list(self.requested_objects),
            "branch_conditions": list(self.branch_conditions),
            "named_tools": list(self.named_tools),
            "detail_policy": self.detail_policy,
            "max_fallback_depth": self.max_fallback_depth,
            "named_tool_is_primary_constraint": bool(self.named_tools),
            "allow_system_repair_commands": self.allow_system_repair_commands,
            "allow_boot_repair_commands": self.allow_boot_repair_commands,
            "allow_destructive_storage_commands": (
                self.allow_destructive_storage_commands
            ),
            "command_control_scope": {
                "allow_system_repair_commands": "仅控制 SFC/DISM 类系统修复命令",
                "allow_boot_repair_commands": "仅控制 bootrec/bcdboot 类引导修复命令",
                "allow_destructive_storage_commands": (
                    "仅控制格式化、清盘、删分区等破坏性磁盘命令"
                ),
                "other_source_grounded_non_destructive_commands": "允许展示",
            },
            "boundary_rules": [
                "只完整展开用户明确请求的操作、对象和条件分支",
                "目的动作和后续维修任务不自动升级为主任务",
                (
                    "用户明确请求失败后路径时，只可简述一层兜底"
                    if self.max_fallback_depth
                    else "用户未请求失败后路径，不展开下游兜底任务"
                ),
                "具名工具存在时，以该工具的流程为主线",
                "allow_* 不是所有命令的总开关",
            ],
        }


def build_answer_scope(query: str) -> AnswerScope:
    """Derive a general task boundary without query-specific answer rules."""

    scope = analyze_query_scope(query)
    task_model = scope.task_model
    operations = tuple(
        str(value) for value in task_model.get("operations") or []
        if str(value).strip()
    )
    context_operations = tuple(
        str(value) for value in task_model.get("context_operations") or []
        if str(value).strip()
    )
    objects = tuple(
        str(value) for value in task_model.get("objects") or []
        if str(value).strip()
    )
    conditions = tuple(
        str(value.get("label") or "")
        for value in task_model.get("conditions") or []
        if isinstance(value, dict) and str(value.get("label") or "").strip()
    )
    operation_objects = _operation_objects(query, operations)
    requested_object_text = " ".join(
        [*objects, *(object_name for _, object_name in operation_objects)]
    )
    repair_requested = "修复" in operations
    destructive_storage = any(
        term in query for term in _DESTRUCTIVE_STORAGE_TERMS
    ) or (
        any(operation in _SAFETY_OPERATIONS for operation in operations)
        and any(term in requested_object_text for term in _STORAGE_OBJECTS)
        and any(term in query for term in ("删除", "格式化", "重建", "转换"))
    )
    return AnswerScope(
        request_kind=scope.request_kind,
        requested_operations=operations,
        context_operations=context_operations,
        requested_objects=tuple(
            object_name for _, object_name in operation_objects
        ),
        branch_conditions=conditions,
        named_tools=tuple(scope.strong_identifiers),
        detail_policy="requested_tasks_only",
        max_fallback_depth=(
            1 if _EXPLICIT_FALLBACK_REQUEST.search(query) else 0
        ),
        allow_system_repair_commands=(
            repair_requested
            and any(term in query for term in ("系统文件", "系统修复"))
        ),
        allow_boot_repair_commands=(
            repair_requested and "引导" in query
        ),
        allow_destructive_storage_commands=destructive_storage,
    )


def build_required_facets(query: str) -> list[RequiredFacet]:
    """Build reusable task, prerequisite, object and safety obligations."""

    scope = analyze_query_scope(query)
    facets: list[RequiredFacet] = []
    for raw in scope.task_model.get("facets") or []:
        if not raw.get("required_for_closure"):
            continue
        facets.append(RequiredFacet(
            facet_id=str(raw.get("facet_id") or ""),
            kind="query_task",
            label=str(raw.get("label") or ""),
            match_terms=tuple(
                str(value) for value in raw.get("match_terms") or []
                if str(value).strip()
            ),
        ))

    if scope.request_kind == "procedure_lookup":
        for identifier in scope.strong_identifiers:
            facets.append(RequiredFacet(
                facet_id=f"prerequisite:{identifier}",
                kind="prerequisite",
                label=f"进入并使用 {identifier} 所需的工具或环境",
                match_terms=(
                    identifier, "准备", "启动", "打开", "管理员", "环境",
                ),
            ))

    operation_objects = _operation_objects(
        query,
        tuple(scope.task_model.get("operations") or ()),
    )
    for operation, object_name in operation_objects:
        facets.append(RequiredFacet(
            facet_id=f"operation_object:{operation}:{object_name}",
            kind="query_task",
            label=f"{operation}{object_name}",
            match_terms=(operation, object_name),
        ))

    scope_contract = build_answer_scope(query)
    if (
        any(
            operation in _SAFETY_OPERATIONS
            for operation in scope_contract.requested_operations
        )
        or scope_contract.allow_boot_repair_commands
        or scope_contract.allow_destructive_storage_commands
    ):
        facets.append(RequiredFacet(
            facet_id="safety:execution_preconditions",
            kind="safety",
            label="执行前核对、数据风险与安全前置条件",
            match_terms=("核对", "不能选错", "数据", "风险", "备份"),
        ))

    return _deduplicate_facets(facets)


def verify_answer_draft(
    draft: dict[str, Any],
    *,
    required_facets: Iterable[RequiredFacet],
    files_read: Iterable[str],
    media_exposed: Iterable[dict[str, Any]],
    answer_scope: AnswerScope | None = None,
    required_procedure_variants: Iterable[
        ProcedureVariantRequirement
    ] = (),
) -> list[str]:
    """Reject ungrounded or incomplete LLM composition before rendering."""

    errors: list[str] = []
    answer = str(draft.get("answer_markdown") or "").strip()
    ledger = draft.get("coverage_ledger")
    if not answer:
        errors.append("missing_answer_markdown")
    if not isinstance(ledger, list):
        return [*errors, "coverage_ledger_not_list"]

    read_set = set(files_read)
    entries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(ledger):
        if not isinstance(raw, dict):
            errors.append(f"ledger_entry_not_object:{index}")
            continue
        facet_id = str(raw.get("facet_id") or "")
        if not facet_id:
            errors.append(f"ledger_entry_missing_facet_id:{index}")
            continue
        if facet_id in entries:
            errors.append(f"duplicate_facet:{facet_id}")
            continue
        entries[facet_id] = raw

    cited_sources: set[str] = set()
    has_gap = False
    for facet in required_facets:
        entry = entries.get(facet.facet_id)
        if entry is None:
            errors.append(f"missing_facet:{facet.facet_id}")
            continue
        status = str(entry.get("status") or "")
        if status not in {"covered", "gap"}:
            errors.append(f"invalid_facet_status:{facet.facet_id}:{status}")
            continue
        sources = [
            str(path) for path in entry.get("source_paths") or []
            if str(path).strip()
        ]
        unknown = sorted(set(sources) - read_set)
        if unknown:
            errors.append(
                f"facet_uses_unread_source:{facet.facet_id}:"
                + ",".join(unknown)
            )
        if status == "covered":
            if not sources:
                errors.append(f"covered_facet_without_source:{facet.facet_id}")
            cited_sources.update(sources)
        else:
            has_gap = True

    for source in cited_sources:
        if f"【来源：{source}】" not in answer:
            errors.append(f"source_marker_missing:{source}")

    if has_gap and "资料缺口" not in answer:
        errors.append("gap_section_missing")

    errors.extend(_verify_procedure_variant_ledger(
        draft,
        answer=answer,
        files_read=read_set,
        requirements=required_procedure_variants,
    ))

    image_paths = {
        str(item.get("asset_path") or "")
        for item in media_exposed
        if str(item.get("media_kind") or "") == "image"
        and str(item.get("asset_path") or "")
        and str(item.get("source_path") or "") in read_set
    }
    cited_image_sequence = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", answer)
    cited_images = set(cited_image_sequence)
    if len(cited_image_sequence) != len(cited_images):
        errors.append("duplicate_image_path")
    for path in sorted(cited_images - image_paths):
        errors.append(f"unknown_image_path:{path}")
    if (
        answer_scope is not None
        and answer_scope.request_kind
        in {"procedure_lookup", "comparison_lookup"}
    ):
        relevant_images = {
            str(item.get("asset_path") or "")
            for item in media_exposed
            if str(item.get("media_kind") or "") == "image"
            and str(item.get("source_path") or "") in cited_sources
            and str(item.get("asset_path") or "")
        }
        if relevant_images and not (cited_images & relevant_images):
            errors.append("relevant_procedural_media_not_cited")

    if answer_scope is not None:
        errors.extend(_verify_markdown_branch_structure(
            answer,
            answer_scope=answer_scope,
        ))
        command_text = "\n".join(
            re.findall(r"```(?:[^\n]*)\n(.*?)```", answer, flags=re.S)
        )
        if (
            not answer_scope.allow_system_repair_commands
            and re.search(r"(?im)^\s*(?:sfc|dism)(?:\.exe)?\b", command_text)
        ):
            errors.append("out_of_scope_system_repair_command")
        if (
            not answer_scope.allow_boot_repair_commands
            and re.search(r"(?im)^\s*(?:bootrec|bcdboot)\b", command_text)
        ):
            errors.append("out_of_scope_boot_repair_command")
        if (
            not answer_scope.allow_destructive_storage_commands
            and re.search(
                r"(?im)^\s*(?:clean(?:\s+all)?|"
                r"format\s+fs=|delete\s+partition|convert\s+(?:gpt|mbr)|"
                r"create\s+partition)\b",
                command_text,
            )
        ):
            errors.append("out_of_scope_destructive_storage_command")
        if (
            not (
                answer_scope.allow_boot_repair_commands
                or answer_scope.allow_destructive_storage_commands
            )
            and re.search(
                r"(?im)^\s*(?:diskpart|list\s+(?:disk|partition|volume)|"
                r"sel(?:ect)?\s+(?:disk|partition|volume)|"
                r"assign\s+letter)\b",
                command_text,
            )
        ):
            errors.append("out_of_scope_disk_management_command")

    # A document may contain pictures for an unrelated section.  Requiring an
    # arbitrary image whenever any fact from that document is cited produces
    # misleading answers.  Relevance and placement remain an agent obligation;
    # the deterministic gate only ensures every image actually used was
    # materialized from the allowed corpus.

    return list(dict.fromkeys(errors))


def _verify_procedure_variant_ledger(
    draft: dict[str, Any],
    *,
    answer: str,
    files_read: set[str],
    requirements: Iterable[ProcedureVariantRequirement],
) -> list[str]:
    """Require every source-declared sibling path to remain auditable.

    Facet closure answers whether the requested task has some evidence.  This
    second ledger answers a different question: if an answer-closing source
    declares several peer solutions, did composition account for each one?
    """

    required = list(requirements)
    if not required:
        return []
    raw_ledger = draft.get("procedure_variant_ledger")
    if not isinstance(raw_ledger, list):
        return ["procedure_variant_ledger_not_list"]

    errors: list[str] = []
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_ledger):
        if not isinstance(raw, dict):
            errors.append(f"procedure_variant_not_object:{index}")
            continue
        source_path = str(raw.get("source_path") or "").strip()
        source_label = str(raw.get("source_label") or "").strip()
        key = (source_path, _normalize_procedure_label(source_label))
        if not source_path or not source_label:
            errors.append(f"procedure_variant_missing_identity:{index}")
            continue
        if source_path not in files_read:
            errors.append(f"procedure_variant_uses_unread_source:{source_path}")
        if key in entries:
            errors.append(
                f"duplicate_procedure_variant:{source_path}:{source_label}"
            )
            continue
        entries[key] = raw

    headings = _markdown_atx_headings(answer)
    expected_keys = {
        (
            requirement.source_path,
            _normalize_procedure_label(requirement.source_label),
        )
        for requirement in required
    }
    for key, raw in entries.items():
        if key not in expected_keys:
            errors.append(f"unexpected_procedure_variant:{key[0]}:{key[1]}")

    status_markers = {
        "expanded": "已展开",
        "guarded": "风险受控地展示",
        "omitted_evidence_gap": "因证据缺失而省略",
    }
    for requirement in required:
        key = (
            requirement.source_path,
            _normalize_procedure_label(requirement.source_label),
        )
        entry = entries.get(key)
        if entry is None:
            errors.append(
                "missing_procedure_variant:"
                f"{requirement.source_path}:{requirement.source_label}"
            )
            continue
        status = str(entry.get("status") or "")
        if status not in status_markers:
            errors.append(
                "invalid_procedure_variant_status:"
                f"{requirement.source_path}:{requirement.source_label}:{status}"
            )
            continue
        answer_label = str(entry.get("answer_label") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if not answer_label:
            errors.append(
                f"procedure_variant_answer_label_missing:{requirement.source_label}"
            )
        if not reason:
            errors.append(
                f"procedure_variant_reason_missing:{requirement.source_label}"
            )
        marker = status_markers[status]
        normalized_answer_label = _normalize_markdown_label(answer_label)
        if not any(
            normalized_answer_label
            and normalized_answer_label in _normalize_markdown_label(title)
            and marker in title
            for _, level, title in headings
            if level >= 2
        ):
            errors.append(
                "procedure_variant_heading_missing:"
                f"{requirement.source_label}:{marker}"
            )

        if status == "omitted_evidence_gap" and "资料缺口" not in answer:
            errors.append(
                f"procedure_variant_gap_section_missing:{requirement.source_label}"
            )

        if requirement.external_artifact_unverified:
            if status != "guarded":
                errors.append(
                    "unverified_external_artifact_not_guarded:"
                    f"{requirement.source_label}"
                )
                continue
            if "脚本内容、版本、哈希未核验" not in answer:
                errors.append("external_artifact_audit_warning_missing")
            if "优先使用可审计的系统内置方法" not in answer:
                errors.append("external_artifact_builtin_preference_missing")
            for url in requirement.external_urls:
                if url not in answer:
                    errors.append(f"external_artifact_url_missing:{url}")
    return errors


def _verify_markdown_branch_structure(
    answer: str,
    *,
    answer_scope: AnswerScope,
) -> list[str]:
    """Require explicit title/step hierarchy for multi-branch procedures.

    Branch labels come from the generic query-scope analyzer.  The check is
    deliberately independent of document names, products and query wording:
    every canonical branch condition must own a level-three Markdown section,
    and procedural branches must contain their own ordered steps.
    """

    conditions = tuple(dict.fromkeys(
        condition.strip()
        for condition in answer_scope.branch_conditions
        if condition.strip()
    ))
    if len(conditions) < 2 or not answer:
        return []

    headings = _markdown_atx_headings(answer)
    errors: list[str] = []
    for condition in conditions:
        normalized_condition = _normalize_markdown_label(condition)
        matches = [
            (line_index, title)
            for line_index, level, title in headings
            if level == 3
            and normalized_condition in _normalize_markdown_label(title)
        ]
        if not matches:
            errors.append(f"branch_heading_missing:{condition}")
            continue
        if answer_scope.requested_operations:
            branch_start = matches[0][0]
            branch_end = next(
                (
                    line_index
                    for line_index, level, _ in headings
                    if line_index > branch_start and level <= 3
                ),
                len(answer.splitlines()),
            )
            branch_lines = answer.splitlines()[branch_start + 1:branch_end]
            if not any(
                re.match(r"^\s*\d+[.)]\s+\S", line)
                for line in branch_lines
            ):
                errors.append(f"branch_steps_missing:{condition}")
    return errors


def _markdown_atx_headings(answer: str) -> list[tuple[int, int, str]]:
    """Return ATX headings outside fenced code as line/level/title tuples."""

    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_marker = ""
    for line_index, line in enumerate(answer.splitlines()):
        fence = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            headings.append((
                line_index,
                len(match.group(1)),
                match.group(2).strip(),
            ))
    return headings


def _normalize_markdown_label(value: str) -> str:
    value = re.sub(r"[`*_~\[\](){}<>《》【】]", "", value.casefold())
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def _normalize_procedure_label(value: str) -> str:
    """Normalize numbering style without collapsing different peer paths."""

    identifier = re.search(
        r"(?:方案|方法)\s*(?:[0-9]+|[零一二两三四五六七八九十]+)"
        r"|第\s*(?:[0-9]+|[零一二两三四五六七八九十]+)\s*种"
        r"(?:操作)?方法",
        value,
    )
    if identifier:
        value = identifier.group(0)
    numerals = {
        "零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
        "四": "4", "五": "5", "六": "6", "七": "7", "八": "8",
        "九": "9", "十": "10",
    }
    normalized = _normalize_markdown_label(value)
    for chinese, arabic in numerals.items():
        normalized = normalized.replace(chinese, arabic)
    return normalized


def _operation_objects(
    query: str,
    operations: tuple[str, ...],
) -> list[tuple[str, str]]:
    positions = sorted(
        (match.start(), match.end(), operation)
        for operation in operations
        for match in [re.search(re.escape(operation), query)]
        if match is not None
    )
    result: list[tuple[str, str]] = []
    for index, (_, end, operation) in enumerate(positions):
        next_start = positions[index + 1][0] if index + 1 < len(positions) else len(query)
        segment = query[end:next_start]
        segment = re.split(r"[，。；;？！?]|(?:时|后|前)", segment, maxsplit=1)[0]
        segment = _PURPOSE_BOUNDARY.split(segment, maxsplit=1)[0]
        segment = _OBJECT_NOISE.sub("", segment).strip()
        for value in _OBJECT_SPLIT.split(segment):
            candidate = _OBJECT_NOISE.sub("", value).strip(" 的：:")
            if 1 <= len(candidate) <= 12 and candidate not in operations:
                result.append((operation, candidate))
    return result


def _deduplicate_facets(
    facets: Iterable[RequiredFacet],
) -> list[RequiredFacet]:
    result: list[RequiredFacet] = []
    seen: set[str] = set()
    for facet in facets:
        if not facet.facet_id or facet.facet_id in seen:
            continue
        seen.add(facet.facet_id)
        result.append(facet)
    return result
