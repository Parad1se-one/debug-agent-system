"""Deterministic query intent and evidence-scope helpers.

The read side serves two different jobs:

* diagnose a reported incident through a ``FaultVariant``;
* answer a documentation/procedure/specification question from source chunks.

Lexical similarity alone cannot safely choose between them.  A query naming
``MemTest86`` or a concrete motherboard model must not be converted into the
highest-scoring generic fault merely because both mention memory or BIOS.
This module keeps that distinction explicit without relying on an online LLM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


_ASCII_TOKEN = re.compile(
    r"[A-Za-z][A-Za-z0-9_.+-]{1,}|\d+(?:\.\d+)+(?:[A-Za-z]+)?"
)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
_REQUEST_SCAFFOLDING = (
    "我想知道", "麻烦帮我", "请帮我", "怎么才能", "应该怎么", "应当怎么",
    "该怎么", "要怎么", "怎么办", "如何", "怎样", "怎么", "请问", "麻烦",
    "帮我", "请",
)
_KNOWLEDGE_PATTERNS = (
    "如何使用", "怎样使用", "怎么使用", "使用方法", "操作步骤", "教程",
    "如何进入", "怎样进入", "怎么进入", "如何关闭", "怎样关闭", "怎么关闭",
    "如何禁用", "怎样禁用", "怎么禁用", "如何分析", "怎样分析", "怎么分析",
    "如何采集", "怎样采集", "怎么采集", "如何收集", "怎样收集", "怎么收集",
    "如何清理", "怎样清理", "怎么清理",
    "如何检测", "怎样检测", "怎么检测", "如何判断", "怎样判断", "怎么判断",
    "如何下载", "怎样下载", "怎么下载", "选哪个", "选择哪个",
    "如何核对", "怎样核对", "怎么核对", "如何处理", "怎样处理",
    "如何进行", "怎样进行", "应该怎么选", "应当怎么选", "怎么选",
    "技术要求", "规格要求", "型号要求", "配置要求", "有什么区别", "区别是什么",
    "适用场景", "流程是什么", "需要哪些信息", "需要什么信息",
)
_SPECIFICATION_PATTERNS = (
    "技术要求", "规格", "型号", "容量要求", "配置要求", "不合规", "怎么选",
    "选型", "参数要求",
)
_CONFIGURATION_PATTERNS = (
    "接线", "接线规范", "接口", "跳线", "针脚", "端子",
    "串口", "端口", "端口号", "bios设置", "bios 设置", "核对",
)
_OPERATION_TOPICS = (
    "授权", "修复", "检测", "清理", "安装", "卸载", "更新", "备份", "还原",
    "分析", "诊断", "检查", "定位", "测试", "采集", "收集", "接线", "设置",
    "进入", "关闭", "禁用", "处理", "判断", "更换", "加装", "迁移", "制作",
    "配置", "导出",
)
_OPERATION_EQUIVALENCE = {
    "安装": ("安装", "重新安装", "重装", "装驱动", "部署"),
    "卸载": ("卸载", "移除", "删除", "清除", "清理旧驱动"),
    "修复": ("修复", "恢复", "重建", "解决", "处理"),
    "检测": (
        "检测", "检查", "测试", "诊断", "排查", "定位", "验证",
        "查看", "观察", "确认",
    ),
    "清理": ("清理", "清除", "释放", "删除"),
    "备份": ("备份", "保存", "拷贝", "复制"),
    "更换": ("更换", "替换", "换成", "换用"),
    "设置": ("设置", "配置", "调整", "修改"),
    "进入": ("进入", "启动到", "打开"),
    "导出": ("导出", "生成", "保存"),
    "采集": ("采集", "收集", "抓取", "导出"),
    "授权": ("授权", "激活", "许可", "许可证"),
}
_SUBJECT_DOMAIN_PROFILES = {
    # Runtime resource pressure and physical memory integrity both contain
    # “内存”, but they are different diagnostic objects and require different
    # procedures.  Keep the distinction as a reusable subject ontology rather
    # than a query-specific title blacklist.
    "runtime_resource_usage": (
        "资源占用", "cpu占用", "内存占用", "磁盘占用", "程序无响应",
        "进程", "任务管理器", "卡顿", "卡死程序",
    ),
    "memory_integrity": (
        "内存检测", "内存诊断", "内存测试", "内存条", "memtest",
        "memorydiagnostic", "pass", "fail",
    ),
}
_DIAGNOSTIC_PATTERNS = (
    "报错", "失败", "异常", "无法", "不能", "不转", "不开机", "蓝屏", "黑屏",
    "卡顿", "死机", "重启", "掉线", "误报", "漏报", "打不开", "不识别",
)
_WEAK_ASCII_IDENTIFIERS = {
    "windows", "system", "software", "hardware", "device", "driver", "error",
    "debug", "manual", "setup", "setting",
    # These are broad platform/component nouns, not concrete tool or model
    # identifiers.  Treating them as a hard scope used to discard the right
    # document merely because its title omitted "AOI" or "CPU".
    "aoi", "bios", "cpu", "ssd", "spc", "fov", "ddr4", "ddr5",
    # Result words are not named entities.  Treating PASS/FAIL as two
    # mandatory identifiers made a perfectly grounded MemTest86 query look
    # uncovered when a section described the result in Chinese.
    "pass", "fail",
}
_VENDOR_ALIASES = {
    "gigabyte": "技嘉",
    "maxsun": "铭瑄",
    "advantech": "研华",
    "软件许可证": "软狗",
    "软件授权": "软授权",
    "硬件加密狗": "硬狗",
}
_IDENTIFIER_ALIASES = {
    "prime95": ("p95",),
    "p95": ("prime95",),
    "smart": ("s.m.a.r.t",),
    "s.m.a.r.t.": ("smart",),
}


@dataclass(frozen=True, slots=True)
class QueryScope:
    mode: str
    request_kind: str
    strong_identifiers: tuple[str, ...]
    diagnostic_signals: tuple[str, ...]
    reasons: tuple[str, ...]
    task_model: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_query_scope(query: str) -> QueryScope:
    """Classify the query before any variant can become executable."""

    text = str(query or "")
    lowered = text.lower()
    knowledge_signals = list(
        phrase for phrase in _KNOWLEDGE_PATTERNS if phrase in lowered
    )
    # Natural questions often insert an adverb between “如何/怎么” and the
    # requested operation (for example “如何彻底清理”).  Match this bounded
    # form without treating “故障如何排查/怎么解决” as a knowledge lookup.
    if re.search(
        r"(?:如何|怎样|怎么).{0,12}(?:使用|用|进入|关闭|禁用|禁止|分析|"
        r"采集|收集|清理|检测|判断|核对|处理|进行|执行|选择|安装|卸载|"
        r"授权|设置|迁移|修复|定位|检查|测试|做)",
        lowered,
    ):
        knowledge_signals.append("bounded_how_operation")
    if re.search(
        r"(?:应该|应当|需要|先).{0,10}(?:做|进行|执行).{0,16}"
        r"(?:哪些|什么|修复|定位|检查|检测|测试|分析|采集|收集)",
        lowered,
    ):
        knowledge_signals.append("bounded_should_operation")
    knowledge_signals = tuple(dict.fromkeys(knowledge_signals))
    diagnostic_signals = tuple(
        phrase for phrase in _DIAGNOSTIC_PATTERNS if phrase in lowered
    )
    comparison_lookup = bool(
        any(token in lowered for token in ("分别", "区别", "对比", "怎么选", "如何选"))
        and any(
            token in lowered
            for token in (
                "进入", "处理", "修复", "检测", "诊断", "系统", "工具",
                "方案", "方法",
            )
        )
    )
    if comparison_lookup:
        request_kind = "comparison_lookup"
    elif any(phrase in lowered for phrase in _SPECIFICATION_PATTERNS):
        request_kind = "specification_lookup"
    elif any(phrase in lowered for phrase in _CONFIGURATION_PATTERNS):
        request_kind = "configuration_lookup"
    elif (
        "授权" in lowered
        and any(
            phrase in lowered
            for phrase in ("更新", "续期", "延期", "到期", "失效")
        )
    ):
        # Exporting a request/fingerprint file is only a supporting step in a
        # licence-renewal workflow.  The state-changing lifecycle task must
        # define the evidence role, otherwise any “导出…并更新授权” question
        # is incorrectly routed to generic log/data-export documentation.
        request_kind = "authorization_update_lookup"
    elif any(phrase in lowered for phrase in ("jira", "升级处理")):
        request_kind = "evidence_or_escalation_lookup"
    elif (
        any(phrase in lowered for phrase in ("采集", "收集", "导出"))
        and not any(
            phrase in lowered
            for phrase in (
                "授权", "安装", "卸载", "修复", "迁移", "清理", "更换",
            )
        )
    ):
        request_kind = "evidence_or_escalation_lookup"
    elif knowledge_signals:
        request_kind = "procedure_lookup"
    else:
        request_kind = "fault_diagnosis"

    # Explicit tutorial/configuration/specification wording wins even if the
    # document topic itself contains words such as “失败” or “修复”.
    knowledge_mode = bool(knowledge_signals) or request_kind != "fault_diagnosis"
    mode = "knowledge_lookup" if knowledge_mode else "fault_diagnosis"
    reasons = [
        f"request_kind:{request_kind}",
        *(f"knowledge_signal:{value}" for value in knowledge_signals[:3]),
    ]
    if not knowledge_mode and diagnostic_signals:
        reasons.append(f"diagnostic_signal:{diagnostic_signals[0]}")
    task_model = _build_query_task_model(
        text,
        request_kind=request_kind,
        strong_ids=strong_identifiers(text),
    )
    return QueryScope(
        mode=mode,
        request_kind=request_kind,
        strong_identifiers=tuple(strong_identifiers(text)),
        diagnostic_signals=diagnostic_signals,
        reasons=tuple(reasons),
        task_model=task_model,
    )


def strong_identifiers(value: str) -> list[str]:
    """Return named tools, codes and model-like tokens from user text."""

    result: list[str] = []
    for raw in _ASCII_TOKEN.findall(str(value or "")):
        normalized = raw.lower().strip("._-")
        if len(normalized) < 2 or normalized in _WEAK_ASCII_IDENTIFIERS:
            continue
        is_named = bool(
            re.search(r"\d|[.+_-]", raw)
            or (raw.isupper() and len(raw) >= 3)
        )
        if is_named and normalized not in result:
            result.append(normalized)
    return result


def matched_strong_identifiers(query: str, text: str) -> list[str]:
    """Return concrete query identifiers represented by a source string."""

    return [
        identifier
        for identifier in strong_identifiers(query)
        if _identifier_in_text(identifier, text)
    ]


def title_match_signals(query: str, title: str) -> dict[str, Any]:
    """Explain whether a document title is a safe primary-evidence anchor."""

    scope = analyze_query_scope(query)
    query_key = _canonical_match_key(with_vendor_aliases(query))
    title_key = _canonical_match_key(with_vendor_aliases(title))
    query_ids = list(scope.strong_identifiers)
    matched_ids = matched_strong_identifiers(query, title)
    identifier_ratio = (
        len(matched_ids) / len(query_ids) if query_ids else 0.0
    )
    common_cjk = longest_common_cjk_phrase(query_key, title_key)
    topic_strength = (
        min(1.0, len(common_cjk) / max(len(title_key), 1))
        if common_cjk
        else 0.0
    )
    reverse_units = _unit_coverage(title, query)
    query_operations = {
        _canonical_match_key(token)
        for token in _OPERATION_TOPICS
        if _canonical_match_key(token) in query_key
    }
    title_operations = {
        _canonical_match_key(token)
        for token in _OPERATION_TOPICS
        if _canonical_match_key(token) in title_key
    }
    requested_operations, context_operations = _query_operation_facets(query)
    requested_operation_coverage = _set_coverage(
        requested_operations,
        title_operations,
    )
    context_operation_coverage = _set_coverage(
        context_operations,
        title_operations,
    )
    operation_aligned = (
        not query_operations
        or bool(query_operations & title_operations)
    )
    shared_topics = _shared_title_topics(query_key, title_key)
    subject_topics: list[str] = []
    for topic in shared_topics:
        subject = topic
        for operation in _OPERATION_TOPICS:
            subject = subject.replace(_canonical_match_key(operation), "")
        if len(subject) >= 2 and subject not in subject_topics:
            subject_topics.append(subject)
    subject_strength = min(
        1.0,
        sum(len(topic) for topic in subject_topics) / 8.0,
    )
    distinctive_subject_topics = [
        topic for topic in subject_topics
        if not any(
            _canonical_match_key(signal) in topic
            or topic in _canonical_match_key(signal)
            for signal in _DIAGNOSTIC_PATTERNS
        )
    ]
    distinctive_subject_strength = min(
        1.0,
        sum(len(topic) for topic in distinctive_subject_topics) / 8.0,
    )
    effective_topic_strength = topic_strength if subject_topics else 0.0
    intent_alignment = _request_kind_title_alignment(
        scope.request_kind,
        title_key,
        shared_topics,
        subject_topics,
        scope.diagnostic_signals,
    )
    # One compatibility score is shared by document-title discovery,
    # coherent document selection and navigation.  Named identifiers remain
    # an evidence-scope constraint, while the score measures how much of the
    # actual task and subject the title explains.
    scope_score = min(
        1.0,
        0.30 * effective_topic_strength
        + 0.15 * reverse_units
        + 0.20 * requested_operation_coverage
        + 0.10 * context_operation_coverage
        + 0.20 * subject_strength
        + 0.15 * intent_alignment,
    )
    reasons: list[str] = []
    safe = False

    if query_ids:
        required_ratio = 1.0 if len(query_ids) == 1 else 0.5
        if identifier_ratio >= required_ratio:
            safe = True
            reasons.append("named_identifier")
    if (
        scope.mode == "knowledge_lookup"
        and reverse_units >= 0.75
        and len(title_key) >= 4
        # A short noun/action title must not swallow a much longer procedure
        # question whose requested operation is absent from that title.  For
        # example, “更换工控机” is related to a new-PC stability-test query,
        # but it is not itself the requested compatibility test procedure.
        and (operation_aligned or _unit_coverage(query, title) >= 0.6)
        and (
            scope.request_kind != "specification_lookup"
            or any(
                token in title_key
                for token in ("要求", "规格", "规范", "选型", "配置", "参数")
            )
        )
    ):
        safe = True
        reasons.append("knowledge_title_covered")
    if (
        scope.request_kind == "specification_lookup"
        and len(common_cjk) >= 4
        and any(token in title_key for token in ("要求", "规格", "规范", "选型", "配置"))
    ):
        safe = True
        reasons.append("specification_topic")
    if (
        scope.mode == "knowledge_lookup"
        and len(common_cjk) >= 5
        and reverse_units >= 0.5
    ):
        safe = True
        reasons.append("procedure_topic")
    if (
        scope.mode == "knowledge_lookup"
        and bool(query_operations & title_operations)
        and (
            len(common_cjk) >= 4
            or reverse_units >= 0.6
            or bool(shared_topics)
        )
    ):
        safe = True
        reasons.append("operation_topic")
    if (
        scope.mode == "knowledge_lookup"
        and bool(set(scope.diagnostic_signals) & {
            signal for signal in _DIAGNOSTIC_PATTERNS if signal in title_key
        })
        and bool([
            topic for topic in shared_topics
            if topic not in _DIAGNOSTIC_PATTERNS
        ])
        and reverse_units >= 0.6
    ):
        safe = True
        reasons.append("diagnostic_topic")
    if (
        scope.mode == "knowledge_lookup"
        and scope_score >= 0.35
        and topic_strength >= 0.2
        and bool(subject_topics)
    ):
        safe = True
        reasons.append("scope_compatibility")

    return {
        "safe": safe,
        "matched_identifiers": matched_ids,
        "identifier_ratio": round(identifier_ratio, 4),
        "longest_common_cjk": common_cjk,
        "topic_strength": round(topic_strength, 4),
        "reverse_unit_coverage": round(reverse_units, 4),
        "operation_aligned": operation_aligned,
        "requested_operation_coverage": round(
            requested_operation_coverage,
            4,
        ),
        "requested_operations": sorted(requested_operations),
        "title_operations": sorted(title_operations),
        "context_operation_coverage": round(
            context_operation_coverage,
            4,
        ),
        "subject_strength": round(subject_strength, 4),
        "distinctive_subject_strength": round(
            distinctive_subject_strength,
            4,
        ),
        "distinctive_subject_topics": distinctive_subject_topics,
        "intent_alignment": round(intent_alignment, 4),
        "scope_score": round(scope_score, 4),
        "shared_topics": shared_topics,
        "subject_topics": subject_topics,
        "reasons": reasons,
    }


def _query_operation_facets(query: str) -> tuple[set[str], set[str]]:
    """Separate the requested task from operations that describe context."""

    text = str(query or "")
    canonical_operations = {
        operation: _canonical_match_key(operation)
        for operation in _OPERATION_TOPICS
    }
    request_match = re.search(
        r"(?:如何|怎样|怎么|应该|应当|需要).{0,4}",
        text,
    )
    request_segment = text[request_match.start():] if request_match else text
    requested = {
        canonical
        for operation, canonical in canonical_operations.items()
        if operation in request_segment
    }
    context_segment = re.split(r"(?:之后|以后|后)", text, maxsplit=1)[0]
    context = {
        canonical
        for operation, canonical in canonical_operations.items()
        if operation in context_segment
    }
    return requested, context - requested


def query_operation_facets(query: str) -> tuple[list[str], list[str]]:
    """Return stable, public operation facets for retrieval and answer closure.

    The private set-based helper is used by title scoring.  Evidence Pack
    construction also needs the same interpretation of a compound request,
    but must not duplicate or subtly drift from that logic.
    """

    requested, context = _query_operation_facets(query)
    return sorted(requested), sorted(context)


def analyze_query_task(query: str) -> dict[str, Any]:
    """Return the semantic task model used by evidence-closure checks.

    This is deliberately deterministic and query-agnostic.  It models the
    dimensions that a useful answer must preserve instead of treating every
    question as an unordered bag of operation words.
    """

    return dict(analyze_query_scope(query).task_model)


def operation_match_terms(operation: str) -> tuple[str, ...]:
    """Return centralized semantic equivalents for an operation facet."""

    canonical = _canonical_match_key(operation)
    terms: list[str] = [str(operation), canonical]
    for key, values in _OPERATION_EQUIVALENCE.items():
        if _canonical_match_key(key) != canonical:
            continue
        terms.extend(values)
    return tuple(dict.fromkeys(
        re.sub(r"\s+", "", value).lower()
        for value in terms
        if str(value).strip()
    ))


def task_phrase_match_terms(value: str) -> tuple[str, ...]:
    """Expand an object phrase through the centralized operation ontology.

    Comparison operands often embed an operation noun, for example
    ``Windows 内存诊断`` while the source heading says ``Windows 内存检测``.
    Expanding the embedded term keeps closure semantic without introducing
    query- or document-title special cases.
    """

    seeds = {str(value or "").strip()}
    for _, equivalents in _OPERATION_EQUIVALENCE.items():
        for source in equivalents:
            for seed in tuple(seeds):
                if source not in seed:
                    continue
                seeds.update(
                    seed.replace(source, target)
                    for target in equivalents
                )
    terms: list[str] = []
    for seed in seeds:
        if not seed:
            continue
        terms.extend((seed, semantic_key(seed)))
    return tuple(dict.fromkeys(
        re.sub(r"\s+", "", with_vendor_aliases(term)).lower()
        for term in terms
        if term
    ))


def task_facet_matches_text(facet: dict[str, Any], text: str) -> bool:
    """Check whether attributable source text supports one task facet."""

    raw_corpus = with_vendor_aliases(text).lower()
    corpus = re.sub(r"\s+", "", raw_corpus)
    if not corpus:
        return False
    kind = str(facet.get("kind") or "")
    label = str(facet.get("label") or "")
    if kind == "entity":
        # Keep token boundaries for ASCII tools/models.  Removing all
        # whitespace first can glue an attachment such as ``DDU.zip`` to the
        # preceding URL and make a present entity look uncovered.
        return _identifier_in_text(label, raw_corpus)
    terms = [
        re.sub(r"\s+", "", with_vendor_aliases(str(value))).lower()
        for value in facet.get("match_terms") or []
        if str(value).strip()
    ]
    if kind == "operation":
        terms = list(operation_match_terms(label))
    if kind in {"condition", "version", "comparison_object", "object"}:
        return any(term and term in corpus for term in terms)
    return any(term and term in corpus for term in terms)


def _build_query_task_model(
    query: str,
    *,
    request_kind: str,
    strong_ids: Iterable[str],
) -> dict[str, Any]:
    text = str(query or "")
    lowered = with_vendor_aliases(text)
    requested_operations, context_operations = _query_operation_facets(text)
    ordered_operations = _ordered_requested_operations(
        text,
        requested_operations,
    )
    entities = list(dict.fromkeys(str(value) for value in strong_ids))
    versions = list(dict.fromkeys(
        value.lower()
        for value in re.findall(
            r"(?<![A-Za-z0-9])\d+(?:\.\d+)+(?:[A-Za-z]+)?",
            text,
            flags=re.IGNORECASE,
        )
    ))
    conditions = _query_conditions(lowered)
    comparison_objects = _comparison_objects(
        text,
        request_kind=request_kind,
        entities=entities,
        conditions=conditions,
        versions=versions,
    )
    objects = _query_objects(
        text,
        operations=ordered_operations,
        conditions=conditions,
    )
    sequence = _query_sequence(text, ordered_operations)
    deliverable = {
        "comparison_lookup": "comparison",
        "specification_lookup": "specification",
        "configuration_lookup": "configuration",
        "evidence_or_escalation_lookup": "evidence_checklist",
        "authorization_update_lookup": "procedure",
        "procedure_lookup": "procedure",
        "fault_diagnosis": "diagnosis",
    }.get(request_kind, "knowledge_answer")

    facets: list[dict[str, Any]] = []
    for operation in ordered_operations:
        facets.append({
            "facet_id": f"operation:{operation}",
            "kind": "operation",
            "label": operation,
            "match_terms": list(operation_match_terms(operation)),
            "required_for_closure": True,
        })
    for entity in entities:
        facets.append({
            "facet_id": f"entity:{entity}",
            "kind": "entity",
            "label": entity,
            "match_terms": [entity],
            "required_for_closure": True,
        })
    for condition in conditions:
        if not condition["required_for_closure"]:
            continue
        facets.append({
            "facet_id": f"condition:{condition['condition_id']}",
            "kind": "condition",
            "label": condition["label"],
            "match_terms": condition["match_terms"],
            "required_for_closure": True,
        })
    for version in versions:
        facets.append({
            "facet_id": f"version:{version}",
            "kind": "version",
            "label": version,
            "match_terms": [version],
            "required_for_closure": True,
        })
    if request_kind == "comparison_lookup":
        for value in comparison_objects:
            facets.append({
                "facet_id": f"comparison_object:{semantic_key(value)}",
                "kind": "comparison_object",
                "label": value,
                "match_terms": list(task_phrase_match_terms(value)),
                "required_for_closure": True,
            })

    return {
        "schema_version": "debug_agent_system.query_task.v2",
        "entities": entities,
        "objects": objects,
        "operations": ordered_operations,
        "context_operations": sorted(context_operations),
        "conditions": conditions,
        "sequence": sequence,
        "subject_domains": subject_domains(text),
        "comparison": {
            "active": request_kind == "comparison_lookup",
            "objects": comparison_objects,
            "dimensions": _comparison_dimensions(text),
        },
        "versions": versions,
        "deliverable": deliverable,
        "facets": _dedupe_facets(facets),
    }


def _ordered_requested_operations(
    query: str,
    requested: set[str],
) -> list[str]:
    positions: dict[str, int] = {}
    for operation in _OPERATION_TOPICS:
        position = str(query).find(operation)
        canonical = _canonical_match_key(operation)
        if position >= 0 and canonical in requested:
            positions[canonical] = min(positions.get(canonical, position), position)
    return [
        operation
        for operation, _position in sorted(
            positions.items(),
            key=lambda item: (item[1], item[0]),
        )
    ] or sorted(requested)


def _query_conditions(query: str) -> list[dict[str, Any]]:
    condition_specs = (
        (
            "can_enter_system",
            "可以进入系统",
            (
                "可以进入系统", "能进入系统", "能够进入系统", "可以正常进入系统",
                "可以进系统", "能进系统", "能够进系统",
            ),
        ),
        (
            "cannot_enter_system",
            "无法进入系统",
            (
                "无法进入系统", "不能进入系统", "进不去系统", "无法正常进入系统",
                "无法进系统", "不能进系统", "进不了系统",
            ),
        ),
    )
    conditions: list[dict[str, Any]] = []
    compact = re.sub(r"\s+", "", query).lower()
    for condition_id, label, terms in condition_specs:
        if any(term in compact for term in terms):
            conditions.append({
                "condition_id": condition_id,
                "label": label,
                "match_terms": list(terms),
                "required_for_closure": True,
            })
    present = {item["condition_id"] for item in conditions}
    # Resolve an omitted repeated object in constructions such as
    # “能进系统，也可能进不去”.  The explicit “系统” anchor prevents a
    # generic “网页进不去” from becoming an OS applicability branch.
    if (
        "can_enter_system" in present
        and "cannot_enter_system" not in present
        and "系统" in compact
        and any(term in compact for term in ("进不去", "进不了", "无法进入", "不能进入"))
    ):
        conditions.append({
            "condition_id": "cannot_enter_system",
            "label": "无法进入系统",
            "match_terms": list(condition_specs[1][2]),
            "required_for_closure": True,
        })
    for marker in ("如果", "若", "当", "前提", "情况下"):
        if marker not in query:
            continue
        clause = next((
            part.strip()
            for part in re.split(r"[，,；;。?？]", query)
            if marker in part and part.strip()
        ), "")
        if clause:
            conditions.append({
                "condition_id": f"clause:{semantic_key(clause)[:24]}",
                "label": clause,
                "match_terms": [strip_request_scaffolding(clause)],
                "required_for_closure": False,
            })
    return conditions


def _comparison_objects(
    query: str,
    *,
    request_kind: str,
    entities: list[str],
    conditions: list[dict[str, Any]],
    versions: list[str],
) -> list[str]:
    if request_kind != "comparison_lookup":
        return []
    # Mutually exclusive applicability branches and version comparisons are
    # already first-class facets.  Re-encoding the whole surrounding clause
    # as a comparison operand creates impossible closure requirements.
    if len(conditions) >= 2 or len(versions) >= 2:
        return []
    if len(entities) >= 2:
        return entities
    prefix = re.split(
        r"(?:分别|有什么区别|区别是什么|怎么选|如何选|对比)",
        str(query),
        maxsplit=1,
    )[0]
    # Drop incident context before the actual operands:
    # “……重启时，Windows 内存诊断和 MemTest86 怎么选”.
    contextual_parts = re.split(r"(?:时|情况下)[，,]", prefix)
    if len(contextual_parts) > 1:
        prefix = contextual_parts[-1]
    if not re.search(r"(?:与|和|还是|vs\.?)", prefix, flags=re.IGNORECASE):
        return []
    values = [
        _clean_task_phrase(value)
        for value in re.split(r"(?:与|和|还是|vs\.?|VS\.?)", prefix)
    ]
    return [
        value for value in dict.fromkeys(values)
        if len(semantic_key(value)) >= 2
    ][:4]


def _query_objects(
    query: str,
    *,
    operations: list[str],
    conditions: list[dict[str, Any]],
) -> list[str]:
    value = str(query)
    for scaffold in _REQUEST_SCAFFOLDING:
        value = value.replace(scaffold, "")
    for operation in _OPERATION_TOPICS:
        value = value.replace(operation, "")
    for condition in conditions:
        for term in condition.get("match_terms") or []:
            value = value.replace(str(term), "")
    values = [
        _clean_task_phrase(part)
        for part in re.split(r"[，,；;。?？]|(?:之后|然后|再|并且|以及)", value)
    ]
    return [
        item for item in dict.fromkeys(values)
        if 2 <= len(semantic_key(item)) <= 32
    ][:6]


def _clean_task_phrase(value: str) -> str:
    result = str(value or "").strip(" ：:，,；;。?？")
    result = re.sub(
        r"(?:时|后|前|情况下|分别|哪些|什么|怎么|如何|怎样|应该|应当|需要)+$",
        "",
        result,
    )
    return result.strip()


def _has_sequence_signal(query: str) -> bool:
    return bool(
        re.search(r"(?:先).{0,30}(?:再|然后|之后)", query)
        or re.search(r"(?:之后|以后|完成后)", query)
    )


def _query_sequence(
    query: str,
    ordered_operations: list[str],
) -> list[dict[str, Any]]:
    if not _has_sequence_signal(query):
        return []
    compact = re.sub(r"\s+", "", str(query or ""))
    match = re.search(
        r"先(?P<before>[^，,；;。?？]{1,40}?)(?:，|,)?"
        r"(?:再|然后|之后)(?P<after>[^，,；;。?？]{1,40})",
        compact,
    )
    if match:
        before = _clean_task_phrase(match.group("before"))
        after = _clean_task_phrase(match.group("after"))
        if before and after:
            return [{
                "before": before,
                "after": after,
                "relation": "before",
            }]
    return [
        {
            "before": ordered_operations[index],
            "after": ordered_operations[index + 1],
            "relation": "before",
        }
        for index in range(len(ordered_operations) - 1)
    ]


def _comparison_dimensions(query: str) -> list[str]:
    dimensions = [
        value
        for value in ("适用场景", "耗时", "风险", "步骤", "结果", "规格", "性能")
        if value in query
    ]
    return dimensions or (["差异", "适用条件"] if any(
        value in query for value in ("区别", "对比", "分别", "怎么选", "如何选")
    ) else [])


def _dedupe_facets(facets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for facet in facets:
        facet_id = str(facet.get("facet_id") or "")
        if not facet_id or facet_id in seen:
            continue
        seen.add(facet_id)
        result.append(facet)
    return result


def _set_coverage(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)


def _canonical_match_key(value: str) -> str:
    """Normalize operation synonyms only for title/topic comparison."""

    result = semantic_key(strip_request_scaffolding(value))
    # Source handbooks commonly say “磁盘”, while users often name the
    # concrete drive letter (“D盘/C盘”).  This alias is only for title/topic
    # comparison; body recall still keeps the exact drive and version.
    result = re.sub(r"[a-z]盘", "磁盘", result)
    result = result.replace("磁盘的数据", "磁盘")
    for source, target in (
        ("重新安装", "安装"),
        ("重装", "安装"),
        ("诊断", "检测"),
        ("检查", "检测"),
        ("排查", "检测"),
        ("定位", "检测"),
        ("处理", "修复"),
        ("解决", "修复"),
        ("使用方法", "使用"),
        ("操作方法", "使用"),
        ("教程", "使用"),
    ):
        result = result.replace(source, target)
    return result


def _shared_title_topics(query_key: str, title_key: str) -> list[str]:
    """Return salient title compounds also present in the query.

    This lets a long question such as “怎样用 Windows 内存诊断检查” anchor
    ``Windows内存检测方法`` without lowering the global title threshold.
    Operation and document-scaffolding compounds are excluded, so a shared
    word such as “方法” or “修复” alone cannot select a document.
    """

    ignored = {
        "系统", "方法", "文档", "使用", "检测", "修复", "设置", "异常",
        "问题", "故障", "操作", "步骤", "教程", "处理", "进行", "应该",
        "怎么", "如何", "怎样", "什么", "哪些", "清理", "设备", "出现",
        # Platform/category headings are useful navigation labels, but they
        # do not by themselves identify the evidence subject.  The concrete
        # symptom, component, procedure or named identifier must carry the
        # match.
        "工控机", "电脑", "windows", "主程序", "aoi", "进入", "模式",
    }
    request_operation_chars = set(
        "如何怎样怎么什么哪些应该应当需要进行执行使用操作方法教程"
        "处理解决修复检测检查诊断排查定位设置"
    )
    topics: list[str] = []
    for run in _CJK_RUN.findall(title_key):
        for size in (4, 3, 2):
            for index in range(len(run) - size + 1):
                token = run[index:index + size]
                if (
                    any(
                        value in token or token in value
                        for value in ignored
                    )
                    or set(token) <= request_operation_chars
                    or token not in query_key
                ):
                    continue
                if any(token in value or value in token for value in topics):
                    continue
                topics.append(token)
    return topics[:6]


def _request_kind_title_alignment(
    request_kind: str,
    title_key: str,
    shared_topics: list[str],
    subject_topics: list[str],
    diagnostic_signals: Iterable[str],
) -> float:
    """Measure whether a title represents the requested evidence role.

    This is intentionally query-agnostic: it maps the already classified
    request kind to document-role vocabulary.  It prevents a generic entity
    heading from winning only because it is short, while allowing a concise
    specification, escalation flow or diagnostic title to be a primary
    evidence-domain anchor.
    """

    role_markers = {
        "specification_lookup": ("要求", "规格", "规范", "选型", "参数", "配置"),
        "configuration_lookup": ("接线", "接口", "跳线", "针脚", "端口", "配置", "设置"),
        "authorization_update_lookup": ("更新", "续期", "延期"),
        "evidence_or_escalation_lookup": (
            "反馈", "流程", "追踪", "jira", "升级", "采集", "收集", "导出",
        ),
    }
    markers = role_markers.get(request_kind, ())
    if markers and any(marker in title_key for marker in markers):
        return 1.0
    if (
        request_kind in {"procedure_lookup", "fault_diagnosis"}
        and subject_topics
        and any(
            _canonical_match_key(signal) in title_key
            for signal in diagnostic_signals
        )
    ):
        return 1.0
    if shared_topics and subject_topics:
        return 0.25
    return 0.0


def chunk_matches_named_scope(query: str, chunk: dict[str, Any]) -> bool:
    """Reject chunks that miss the concrete named entity in a lookup query."""

    identifiers = strong_identifiers(query)
    if not identifiers:
        return True
    corpus = " ".join([
        str(chunk.get("source_label") or ""),
        str(chunk.get("text") or ""),
        _source_path_text(chunk),
    ])
    matched = sum(
        _identifier_in_text(identifier, corpus)
        for identifier in identifiers
    )
    scope = analyze_query_scope(query)
    if scope.request_kind == "comparison_lookup":
        return True
    # A configuration lookup that names a concrete board/model must match the
    # complete identifier tuple.  Matching only generic "BIOS" or "B760" is
    # not enough to answer a DS3H/AIMB-788/B760M-D4-specific question.
    required = (
        len(identifiers)
        if scope.request_kind == "configuration_lookup"
        else 1 if len(identifiers) == 1
        else max(1, (len(identifiers) + 1) // 2)
    )
    return matched >= required


def scope_polarity_compatible(query: str, candidate: str) -> bool:
    """Reject evidence belonging to a mutually exclusive applicability branch."""

    query_key = semantic_key(query)
    candidate_key = semantic_key(candidate)
    exclusive_pairs = (
        (("硬件加密狗", "硬狗"), ("软件许可证", "软件授权", "软狗", "软授权")),
        (("可以进入系统", "能进入系统"), ("无法进入系统", "不能进入系统", "进不去系统")),
    )
    for left_tokens, right_tokens in exclusive_pairs:
        query_left = any(token in query_key for token in left_tokens)
        query_right = any(token in query_key for token in right_tokens)
        candidate_left = any(token in candidate_key for token in left_tokens)
        candidate_right = any(token in candidate_key for token in right_tokens)
        if query_left and candidate_right and not candidate_left:
            return False
        if query_right and candidate_left and not candidate_right:
            return False
    return True


def subject_domains(value: str) -> list[str]:
    """Return the semantic diagnostic/knowledge object domains in ``value``."""

    corpus = semantic_key(value).lower()
    domains: list[str] = []
    for domain, signals in _SUBJECT_DOMAIN_PROFILES.items():
        if any(semantic_key(signal).lower() in corpus for signal in signals):
            domains.append(domain)
    return domains


def subject_domain_compatible(query: str, candidate: str) -> bool:
    """Reject evidence for a different object that shares a surface noun.

    A runtime-resource question mentioning “内存占用” must not expand a
    physical-memory integrity manual merely because both contain “内存”.
    Candidates that also contain runtime-resource signals remain eligible,
    and comparison questions may intentionally span both domains.
    """

    scope = analyze_query_scope(query)
    if scope.request_kind == "comparison_lookup":
        return True
    query_domains = set(subject_domains(query))
    candidate_domains = set(subject_domains(candidate))
    mutually_exclusive = (
        ("runtime_resource_usage", "memory_integrity"),
    )
    for left, right in mutually_exclusive:
        if left in query_domains and right in candidate_domains:
            if left not in candidate_domains:
                return False
        if right in query_domains and left in candidate_domains:
            if right not in candidate_domains:
                return False
    return True


def diagnostic_subject_compatible(query: str, candidate: str) -> bool:
    """Check the object immediately qualified by an overloaded fault term.

    ``维修板误报`` and ``轨道有板误报`` share the high-IDF string ``板误报``,
    but they describe different objects.  Character n-gram retrieval should
    surface the candidate; it must not be allowed to lock it.  The check is
    intentionally limited to overloaded diagnostic relations so ordinary
    paraphrases remain unaffected.
    """

    query_key = semantic_key(query)
    candidate_key = semantic_key(candidate)
    for marker in ("误报", "漏报", "错报", "不识别"):
        if marker not in query_key or marker not in candidate_key:
            continue
        query_subject = _subject_before_marker(query_key, marker)
        candidate_subject = _subject_before_marker(candidate_key, marker)
        # A one-character subject such as “板误报” is genuinely broad and
        # should be handled by confidence/margin and RequiredInfo instead.
        if len(query_subject) < 2 or len(candidate_subject) < 2:
            continue
        if len(longest_common_cjk_phrase(query_subject, candidate_subject)) >= 2:
            continue
        return False
    return True


def _subject_before_marker(value: str, marker: str) -> str:
    prefix = str(value or "").split(marker, 1)[0]
    for scaffold in _REQUEST_SCAFFOLDING:
        prefix = prefix.replace(semantic_key(scaffold), "")
    # Keep the nearest bounded compound.  Leading conversational context is
    # not part of the object being diagnosed.
    return prefix[-8:]


def source_document_title(chunk: dict[str, Any]) -> str:
    """Return the hash-pinned source filename when it is available."""

    source_path = _source_path_text(chunk)
    if source_path:
        name = source_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return re.sub(r"\.[A-Za-z0-9]+$", "", name)
    return str(chunk.get("source_label") or "")


def with_vendor_aliases(value: str) -> str:
    result = str(value or "").lower()
    for english, chinese in _VENDOR_ALIASES.items():
        result = result.replace(english, chinese)
    return result


def _identifier_in_text(identifier: str, text: str) -> bool:
    raw_identifier = str(identifier or "").lower()
    raw_text = with_vendor_aliases(text)
    candidates = (raw_identifier, *_IDENTIFIER_ALIASES.get(raw_identifier, ()))
    raw_text_identifiers = {
        value.lower().strip("._-")
        for value in _ASCII_TOKEN.findall(raw_text)
    }
    text_identifiers = set(raw_text_identifiers)
    # A tool/model is often present only as a source attachment or download
    # package (``DDU.zip``, ``Prime95.exe``).  Treat the basename of a known
    # file type as the same strict identifier without generally splitting
    # punctuation: ``DISM`` must still not match the distinct tool
    # ``Dism++`` and a short model prefix must not match a longer SKU.
    known_file_extensions = {
        "7z", "bat", "bin", "cab", "cmd", "csv", "doc", "docx", "exe",
        "gz", "img", "iso", "json", "log", "md", "msi", "pdf", "ps1",
        "rar", "tar", "txt", "xls", "xlsx", "xml", "zip",
    }
    for token in raw_text_identifiers:
        basename, separator, extension = token.rpartition(".")
        if separator and basename and extension in known_file_extensions:
            text_identifiers.add(basename)
    for candidate in candidates:
        normalized_candidate = str(candidate).lower().strip("._-")
        if re.search(r"[.+_-]", candidate):
            if normalized_candidate in text_identifiers:
                return True
            continue
        # Use token equality for ASCII identifiers.  A command named DISM is
        # not the same entity as the Dism++ application, just as a short model
        # prefix is not sufficient for a longer board SKU.
        if normalized_candidate in text_identifiers:
            return True
    return False


def semantic_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def longest_common_cjk_phrase(left: str, right: str) -> str:
    left_runs = _CJK_RUN.findall(str(left or ""))
    right_key = semantic_key(right)
    best = ""
    for run in left_runs:
        for start in range(len(run)):
            for end in range(start + len(best) + 1, len(run) + 1):
                value = run[start:end]
                if value in right_key and len(value) > len(best):
                    best = value
    return best


def strip_request_scaffolding(value: str) -> str:
    result = str(value or "")
    for phrase in _REQUEST_SCAFFOLDING:
        result = result.replace(phrase, "")
    return result


def _unit_coverage(source: str, target: str) -> float:
    source_key = semantic_key(strip_request_scaffolding(source))
    target_key = semantic_key(strip_request_scaffolding(target))
    units: set[str] = set()
    for identifier in strong_identifiers(source):
        units.add(identifier)
    for run in _CJK_RUN.findall(source_key):
        if len(run) == 1:
            units.add(run)
        else:
            units.update(run[index:index + 2] for index in range(len(run) - 1))
    if not units:
        return 0.0
    return sum(semantic_key(with_vendor_aliases(unit)) in target_key for unit in units) / len(units)


def _source_path_text(chunk: dict[str, Any]) -> str:
    for item in chunk.get("source_offsets") or []:
        if isinstance(item, dict) and str(item.get("source_path") or ""):
            return str(item.get("source_path") or "")
    return ""


def dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "")
        if item and item not in result:
            result.append(item)
    return result
