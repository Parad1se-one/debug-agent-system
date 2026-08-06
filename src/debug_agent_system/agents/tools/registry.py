"""统一只读工具注册表（Tool Registry）。

本模块是 debug_agent_system 的唯一工具事实源。它把散落在
``CorpusReadTools.schemas()``、``ReadToolRegistry.schemas()``、
``adapters/codex_read/executor.py read_side_tool_schemas()`` 和
``adapters/deepseek_read/executor.py`` 的工具定义统一为一份
``ToolDefinition`` 清单，并对外提供两种输出形态：

- ``schemas(style="responses")``：Responses API 风格（Codex/Copilot 旁路）。
- ``schemas(style="chat_completions")``：OpenAI Chat Completions 风格
  （DeepSeek 旁路）。

所有工具都是**只读**的：不写 canonical KG_v2、不执行附件脚本、不触发设备
动作。每个执行结果统一包装为 ``ToolResultEnvelope``
（``debug_agent_system.read_tool_result.v1``），并携带来源绑定的
observations/evidence_ids/safety 字段。

设计约束
--------
- 执行器（backend）延迟绑定：注册表只声明工具与参数契约，具体实现通过
  ``ToolBackend`` 协议注入，便于复用现有 parser/runtime 且可测试。
- 永不抛出：``execute`` 对未知工具、参数错误、后端异常一律返回结构化失败，
  保证 Agent 循环不会因单个工具崩溃。
- 来源闭合：凡能产生证据的工具都会在 result 中带上 evidence_ids，供上层
  Evidence Fabric / verifier 使用。
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

TOOL_REGISTRY_SCHEMA = "debug_agent_system.tool_registry.v1"
TOOL_RESULT_SCHEMA = "debug_agent_system.read_tool_result.v1"


# ---------------------------------------------------------------------------
# 工具契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """单个工具参数的定义（JSON Schema 兼容子集）。"""

    name: str
    type: str = "string"  # string|integer|boolean|array|object|number
    description: str = ""
    required: bool = False
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None  # array 的元素 schema
    properties: dict[str, Any] | None = None  # object 的字段 schema
    minimum: int | None = None
    maximum: int | None = None
    default: Any = None

    def to_schema(self) -> dict[str, Any]:
        """转为 JSON Schema 属性片段（不含 required 标记）。"""

        node: dict[str, Any] = {"type": self.type}
        if self.description:
            node["description"] = self.description
        if self.enum is not None:
            node["enum"] = self.enum
        if self.items is not None:
            node["items"] = self.items
        if self.properties is not None:
            node["properties"] = self.properties
        if self.minimum is not None:
            node["minimum"] = self.minimum
        if self.maximum is not None:
            node["maximum"] = self.maximum
        if self.default is not None:
            node["default"] = self.default
        return node


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """一个只读工具的完整契约。"""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = "misc"  # evidence|incident|kg|corpus|diagnose|write|misc
    backend: str = ""  # 后端标识：router|incident|kg_v2|corpus|system|custom
    read_only: bool = True
    extra_schema: dict[str, Any] = field(default_factory=dict)  # 附加 schema 字段

    @property
    def required(self) -> list[str]:
        return [item.name for item in self.parameters if item.required]

    def parameters_schema(self) -> dict[str, Any]:
        """构造工具的 parameters 对象（含 required）。"""

        properties = {
            item.name: item.to_schema() for item in self.parameters
        }
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": self.required,
            "additionalProperties": False,
        }
        schema.update(self.extra_schema)
        return schema

    def responses_schema(self) -> dict[str, Any]:
        """Responses API 风格工具 schema。"""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema(),
            "strict": True,
        }

    def chat_completions_schema(self) -> dict[str, Any]:
        """OpenAI Chat Completions 风格工具 schema（DeepSeek）。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


class ToolBackend(Protocol):
    """工具执行后端协议：输入参数字典，输出结构化 payload。"""

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 执行结果
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolExecutionResult:
    """一次工具执行的统一结果（envelope 数据载体）。"""

    schema_version: str = TOOL_RESULT_SCHEMA
    tool: str = ""
    call_id: str = ""
    status: str = "ok"  # ok|parse_failed|error
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    safety: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": self.tool,
            "call_id": self.call_id,
            "status": self.status,
            "payload": self.payload,
            "evidence_ids": self.evidence_ids,
            "source_ids": self.source_ids,
            "errors": self.errors,
            "excluded": self.excluded,
            "safety": self.safety,
            "observability": self.observability,
        }


def _call_fingerprint(tool: str, arguments: dict[str, Any]) -> str:
    raw = f"{tool}:{sorted(str(k)+'='+str(v) for k, v in (arguments or {}).items())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# 统一注册表
# ---------------------------------------------------------------------------


class ToolRegistry:
    """只读工具的唯一注册表与执行入口。

    用法::

        registry = ToolRegistry()
        registry.register(tool_definition, backend)
        schemas = registry.schemas("chat_completions")   # 给 DeepSeek
        result = registry.execute("parse_evidence", {...})
    """

    schema_version = TOOL_REGISTRY_SCHEMA

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._backends: dict[str, ToolBackend] = {}

    # -- 注册 --------------------------------------------------------------

    def register(
        self,
        definition: ToolDefinition,
        backend: ToolBackend | Callable[[dict[str, Any]], dict[str, Any]],
    ) -> "ToolRegistry":
        if not definition.name or not definition.name.strip():
            raise ValueError("tool_definition_requires_name")
        if definition.name in self._definitions:
            raise ValueError(f"duplicate_tool_definition:{definition.name}")
        self._definitions[definition.name] = definition
        if callable(backend) and not isinstance(backend, ToolBackend):
            self._backends[definition.name] = _CallableBackend(backend)
        else:
            self._backends[definition.name] = backend  # type: ignore[assignment]
        return self

    # -- 查询 --------------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._definitions)

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def by_category(self, category: str) -> list[ToolDefinition]:
        return [
            item
            for item in self._definitions.values()
            if item.category == category
        ]

    def definitions(self) -> list[ToolDefinition]:
        return sorted(self._definitions.values(), key=lambda item: item.name)

    def schemas(self, style: str = "responses") -> list[dict[str, Any]]:
        """输出全部工具 schema。

        style:
          - ``responses``：Responses API（Codex/Copilot 旁路）。
          - ``chat_completions``：Chat Completions（DeepSeek 旁路）。
        """

        if style not in {"responses", "chat_completions"}:
            raise ValueError(f"unsupported_tool_schema_style:{style}")
        return [
            (
                item.responses_schema()
                if style == "responses"
                else item.chat_completions_schema()
            )
            for item in self.definitions()
        ]

    # -- 执行 --------------------------------------------------------------

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str = "",
    ) -> dict[str, Any]:
        """执行一个工具并返回统一 envelope（绝不抛出）。"""

        definition = self._definitions.get(name)
        if definition is None:
            return ToolExecutionResult(
                tool=name,
                status="error",
                errors=[{"code": "unknown_tool", "message": f"unknown tool: {name}"}],
            ).to_dict()
        backend = self._backends.get(name)
        if backend is None:
            return ToolExecutionResult(
                tool=name,
                status="error",
                errors=[{"code": "backend_unavailable", "message": f"backend not registered: {name}"}],
            ).to_dict()
        arguments = dict(arguments or {})
        validation = self._validate(definition, arguments)
        if validation:
            return ToolExecutionResult(
                tool=name,
                status="error",
                errors=[{"code": "invalid_arguments", "message": validation}],
            ).to_dict()
        fingerprint = _call_fingerprint(name, arguments)
        try:
            payload = backend.execute(arguments)
        except Exception as exc:  # noqa: BLE001 - 结构化失败，绝不抛出
            return ToolExecutionResult(
                tool=name,
                call_id=call_id or f"call-{fingerprint[:12]}",
                status="error",
                errors=[{
                    "code": "backend_error",
                    "message": f"{type(exc).__name__}:{str(exc)[:400]}",
                }],
                observability={"elapsed_ms": 0, "status": "error"},
            ).to_dict()
        evidence_ids = _extract_evidence_ids(payload)
        result = ToolExecutionResult(
            tool=name,
            call_id=call_id or f"call-{fingerprint[:12]}",
            status="ok",
            payload=payload,
            evidence_ids=evidence_ids,
            safety={
                "read_only": definition.read_only,
                "side_effect": False,
                "approval_required": False,
            },
            observability={
                "backend": definition.backend,
                "category": definition.category,
                "status": "ok",
            },
        )
        return result.to_dict()

    @staticmethod
    def _validate(definition: ToolDefinition, arguments: dict[str, Any]) -> str:
        """轻量参数校验：required 存在性 + 基本类型。"""

        allowed = {item.name for item in definition.parameters}
        for key in arguments:
            if key not in allowed:
                return f"unexpected_argument:{key}"
        for param in definition.parameters:
            if param.required and param.name not in arguments:
                return f"missing_required_argument:{param.name}"
            if param.name in arguments and arguments[param.name] is not None:
                value = arguments[param.name]
                if param.type == "integer" and not isinstance(value, int):
                    return f"argument_type_mismatch:{param.name}"
                if param.type == "boolean" and not isinstance(value, bool):
                    return f"argument_type_mismatch:{param.name}"
                if param.type == "string" and not isinstance(value, str):
                    return f"argument_type_mismatch:{param.name}"
                if (
                    param.enum is not None
                    and isinstance(value, str)
                    and value not in param.enum
                ):
                    return f"argument_not_in_enum:{param.name}"
        return ""


class _CallableBackend:
    """把普通函数包装为 ToolBackend。"""

    def __init__(self, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._fn = fn

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._fn(arguments)


def _extract_evidence_ids(payload: dict[str, Any]) -> list[str]:
    """从后端 payload 中尽力提取 evidence_ids（来源闭合）。"""

    if not isinstance(payload, dict):
        return []
    ids: list[str] = []
    for key in ("evidence_ids", "evidence_id"):
        value = payload.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item))
        elif isinstance(value, str) and value:
            ids.append(value)
    nested = payload.get("payload")
    if isinstance(nested, dict):
        ids.extend(_extract_evidence_ids(nested))
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------------------
# 工具定义工厂（内置全套只读工具）
# ---------------------------------------------------------------------------

_EVIDENCE_TOOLS = ["attachment", "document", "dmp", "image", "jira", "log_package", "proj", "auto"]


def build_default_registry(*, backend_provider: Callable[[str], Any] | None = None) -> ToolRegistry:
    """构建带全套默认只读工具的注册表。

    ``backend_provider`` 可选：给定工具名返回实际执行后端；为 None 时使用
    内置的 ``_default_backend``（基于 agents/tools 的 EvidenceToolAgent 等）。
    后续接入 incident_runtime / KGV2ReadModel / CorpusReadTools 时，只需在该
    函数中把对应工具挂到对应 backend 即可。
    """

    registry = ToolRegistry()
    provider = backend_provider or _default_backend

    # -- 证据解析组（category=evidence） ------------------------------------
    registry.register(
        ToolDefinition(
            name="parse_evidence",
            description=(
                "用有界只读解析器检查一个调用方提供的证据资源。"
                "自动识别或显式指定资源类型（附件/文档/转储/图片/Jira/日志包/工程文件）。"
                "图片仅返回头部元数据，不做 OCR；日志包不解压、不执行。"
                "返回结构化的工具观察与 evidence_ids，供证据闭包使用。"
            ),
            parameters=[
                ToolParameter("tool", "string", "资源类型；auto 表示按内容推断", required=True,
                              enum=_EVIDENCE_TOOLS),
                ToolParameter("resource", "object", "资源描述对象（path/url/text/name/kind 等）",
                              required=True),
                ToolParameter("max_bytes", "integer", "最大读取字节数（1-1048576）", maximum=1048576,
                              minimum=1, default=65536),
            ],
            category="evidence",
            backend="router",
        ),
        provider("parse_evidence"),
    )

    registry.register(
        ToolDefinition(
            name="parse_evidence_context",
            description=(
                "读取本地样本目录 / source_manifest.json，批量路由其中原始文件到"
                "对应的安全解析器，返回分组工具证据包。只读、不解压、不执行。"
            ),
            parameters=[
                ToolParameter("root", "string", "样本目录或 source_manifest.json 路径", required=True),
                ToolParameter("max_bytes", "integer", "单文件最大读取字节数", maximum=1048576,
                              minimum=1, default=65536),
                ToolParameter("limit", "integer", "最多处理文件数；0 表示不限制", minimum=0, default=0),
            ],
            category="evidence",
            backend="router_context",
        ),
        provider("parse_evidence_context"),
    )

    # -- 诊断包/事件分析组（category=incident） -----------------------------
    registry.register(
        ToolDefinition(
            name="parse_evtx_window",
            description=(
                "解析 Windows EVTX 事件日志，可按 query 参考时间窗对齐并只保留命中"
                "窗口的记录。返回 provider/event_id/severity/message/timestamp。"
                "用于还原蓝屏、异常关机、驱动复位等系统事件。"
            ),
            parameters=[
                ToolParameter("path", "string", "EVTX 文件路径或压缩包内成员路径", required=True),
                ToolParameter("time_scope", "object", "时间窗对象（reference_windows），可选",
                              properties={"reference_windows": {"type": "array"}}),
                ToolParameter("max_records", "integer", "最多扫描记录数", minimum=100, default=100000),
                ToolParameter("max_selected", "integer", "最多保留记录数", minimum=10, default=5000),
            ],
            category="incident",
            backend="incident_evtx",
        ),
        provider("parse_evtx_window"),
    )

    registry.register(
        ToolDefinition(
            name="read_kernel_dump",
            description=(
                "只读解析 Windows 内核转储（Minidump MDMP 或 PAGEDU64 完整内核转储）"
                "的头部元数据：bugcheck code、名称、参数、OS 版本、处理器数。"
                "不读全量文件、不做符号化、不执行调试器。用于把蓝屏终止代码对齐到"
                "具体崩溃签名。"
            ),
            parameters=[
                ToolParameter("path", "string", "DMP 文件路径", required=True),
            ],
            category="incident",
            backend="incident_dump",
        ),
        provider("read_kernel_dump"),
    )

    registry.register(
        ToolDefinition(
            name="read_log_window",
            description=(
                "读取某个已解析日志文件在指定行号附近的上下文窗口（before/after 行）。"
                "用于查看事件发生前后的原始日志细节。"
            ),
            parameters=[
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("artifact_id", "string", "日志工件 ID", required=True),
                ToolParameter("line", "integer", "中心行号（1 起）", required=True, minimum=1),
                ToolParameter("before", "integer", "向前行数", minimum=0, default=10),
                ToolParameter("after", "integer", "向后行数", minimum=0, default=20),
            ],
            category="incident",
            backend="incident_log_window",
        ),
        provider("read_log_window"),
    )

    registry.register(
        ToolDefinition(
            name="search_diagnostic_events",
            description=(
                "在已解析的 Incident 案件中按 provider/event_id/关键字搜索诊断事件，"
                "返回命中的事件列表（含时间戳与消息）。用于在大量 EVTX/日志事件中"
                "定位与 query 相关的关键证据。"
            ),
            parameters=[
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("query", "string", "搜索关键字（provider/event_id/文本）", required=True),
                ToolParameter("limit", "integer", "最大返回条数", minimum=1, maximum=200, default=50),
            ],
            category="incident",
            backend="incident_events",
        ),
        provider("search_diagnostic_events"),
    )

    registry.register(
        ToolDefinition(
            name="build_incident_timeline",
            description=(
                "把已解析事件按时间排序并生成时间线，标注异常/重启/进程启动等关键"
                "信号。用于还原故障发生顺序与因果关系。"
            ),
            parameters=[
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
            ],
            category="incident",
            backend="incident_timeline",
        ),
        provider("build_incident_timeline"),
    )

    # -- KG 检索组（category=kg） -------------------------------------------
    registry.register(
        ToolDefinition(
            name="kg_search_candidates",
            description=(
                "用一条 query 检索 KG_v2/SAG 故障候选（FaultVariant），返回候选、"
                "支持 chunk 与检索路径。用于把现场证据映射到已知故障知识。"
                "只读，不修改候选排序。"
            ),
            parameters=[
                ToolParameter("query", "string", "检索用 query", required=True),
                ToolParameter("limit", "integer", "候选数（1-20）", minimum=1, maximum=20, default=10),
            ],
            category="kg",
            backend="kg_v2",
        ),
        provider("kg_search_candidates"),
    )

    registry.register(
        ToolDefinition(
            name="kg_get_object",
            description=(
                "按 object_id 读取单个规范 KG_v2 对象（family/variant/action/trace/"
                "required-info 等）。用于查看候选的完整定义。"
            ),
            parameters=[
                ToolParameter("object_id", "string", "KG_v2 对象 ID", required=True),
            ],
            category="kg",
            backend="kg_v2",
        ),
        provider("kg_get_object"),
    )

    registry.register(
        ToolDefinition(
            name="kg_compile_plan",
            description=(
                "把一个 family+variant 编译为可执行的 V2DiagnosticPlan（步骤/分支/"
                "结果/required-info/证据）。用于查看已锁定的诊断路径。只读，不执行分支。"
            ),
            parameters=[
                ToolParameter("family_id", "string", "Family 对象 ID", required=True),
                ToolParameter("variant_id", "string", "Variant 对象 ID", required=True),
            ],
            category="kg",
            backend="kg_v2",
        ),
        provider("kg_compile_plan"),
    )

    # -- 语料阅读组（category=corpus） --------------------------------------
    registry.register(
        ToolDefinition(
            name="list_files",
            description=(
                "按相对 glob 列出语料库（data/raw 与 data/kg_v2）中的文件路径。"
                "仅返回路径，不做相关性排序。用于让模型自行决定下一步读取哪些文件。"
            ),
            parameters=[
                ToolParameter("glob", "string", "相对 glob 模式（如 **/*.log）", required=True),
                ToolParameter("limit", "integer", "最大返回条数", minimum=1, maximum=500, default=100),
            ],
            category="corpus",
            backend="corpus",
        ),
        provider("list_files"),
    )

    registry.register(
        ToolDefinition(
            name="search_text",
            description=(
                "在语料库文本文件中搜索字面量或正则表达式。结果按 path/line 顺序返回，"
                "无相关性评分。用于定位术语、错误码、关键词出现的位置。"
            ),
            parameters=[
                ToolParameter("query", "string", "搜索文本或正则", required=True),
                ToolParameter("path_glob", "string", "限定路径 glob（默认全库）", default="**/*"),
                ToolParameter("regex", "boolean", "是否按正则解析", default=False),
                ToolParameter("case_sensitive", "boolean", "是否区分大小写", default=False),
                ToolParameter("max_matches", "integer", "最大命中数", minimum=1, maximum=500, default=100),
                ToolParameter("context_lines", "integer", "上下文行数（0-5）", minimum=0, maximum=5, default=0),
            ],
            category="corpus",
            backend="corpus",
        ),
        provider("search_text"),
    )

    registry.register(
        ToolDefinition(
            name="read_text",
            description=(
                "读取语料库中某个文本文件的精确行区间。长文档请分段多次调用。"
                "用于阅读证据原文，作为答案的来源依据。"
            ),
            parameters=[
                ToolParameter("path", "string", "语料库内相对路径", required=True),
                ToolParameter("start_line", "integer", "起始行（1 起）", required=True, minimum=1),
                ToolParameter("end_line", "integer", "结束行（含）", required=True, minimum=1),
            ],
            category="corpus",
            backend="corpus",
        ),
        provider("read_text"),
    )

    # -- 扩展工具（全量只读面） --------------------------------------------
    _register_extended_tools(registry, provider)

    return registry


# ---------------------------------------------------------------------------
# 扩展工具：确定性诊断、文档展开、Incident 全量、写侧只读面
# ---------------------------------------------------------------------------


def _register_extended_tools(
    registry: ToolRegistry,
    provider: Callable[[str], Any],
) -> None:
    """注册全量只读工具面（在基础 13 个之上扩展）。

    这些工具与 ``adapters/codex_read/executor.py`` 的 32 工具白名单对齐，
    并额外包含写侧 W9 只读面。后端由 ``backend_provider`` 注入。
    """

    # -- 确定性诊断组（category=diagnose） --------------------------------
    registry.register(
        ToolDefinition(
            name="diagnose_start",
            description=(
                "启动确定性 KG_v2 诊断运行时。只有该运行时可以锁定 Variant 或编译"
                "诊断动作。用于把现场 query 交给冻结诊断状态机，得到官方 answer/status。"
            ),
            parameters=[
                ToolParameter("query", "string", "诊断 query", required=True),
                ToolParameter("interactive", "boolean", "是否交互式", default=True),
                ToolParameter("session_id", "string", "可选会话 ID"),
                ToolParameter("routing_context", "object", "路由上下文（stage/query_type/interface/side）"),
                ToolParameter("evidence_resources", "array", "证据资源数组", items={"type": "object"}),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("diagnose_start"),
    )

    registry.register(
        ToolDefinition(
            name="diagnose_step",
            description=(
                "用用户文本和调用方提供的证据资源继续一个已有的确定性诊断会话。"
                "推进冻结状态机，返回下一步 answer/status/required_data。"
            ),
            parameters=[
                ToolParameter("session_id", "string", "会话 ID", required=True),
                ToolParameter("user_message", "string", "用户反馈文本", required=True),
                ToolParameter("evidence_resources", "array", "证据资源数组", items={"type": "object"}),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("diagnose_step"),
    )

    registry.register(
        ToolDefinition(
            name="retrieve_evidence",
            description=(
                "从两个 SAG 通道召回 KG_v2 FaultVariant 候选与已批准来源 chunk，"
                "不创建会话。用于调查早期快速对齐已知故障知识。"
            ),
            parameters=[
                ToolParameter("query", "string", "检索 query", required=True),
                ToolParameter("limit", "integer", "候选数（1-20）", minimum=1, maximum=20, default=10),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("retrieve_evidence"),
    )

    registry.register(
        ToolDefinition(
            name="expand_document_context",
            description=(
                "把显式选中的来源文档展开为完整的已批准语义大纲，保持 chunk 顺序与"
                "媒体引用。用于阅读知识文档正文作为答案依据。"
            ),
            parameters=[
                ToolParameter("query", "string", "检索 query", required=True),
                ToolParameter("document_ids", "array", "文档 ID 列表（1-8）", required=True,
                              items={"type": "string"}),
                ToolParameter("max_chunks", "integer", "最大 chunk 数（1-64）", minimum=1, maximum=64, default=64),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("expand_document_context"),
    )

    registry.register(
        ToolDefinition(
            name="inspect_kg_path",
            description=(
                "检查一个 KG_v2 Family/Variant 诊断路径，包括 Trace 动作、结果、分支"
                "条件、证据与风险标志。此工具绝不选择分支或执行动作。"
            ),
            parameters=[
                ToolParameter("family_id", "string", "Family ID", required=True),
                ToolParameter("variant_id", "string", "Variant ID", required=True),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("inspect_kg_path"),
    )

    registry.register(
        ToolDefinition(
            name="inspect_source_assets",
            description=(
                "列出选中已批准文档 chunk 绑定的来源图片与附件，包括标题与出处。"
                "用于确认答案引用的媒体是否真实存在。"
            ),
            parameters=[
                ToolParameter("query", "string", "检索 query", required=True),
                ToolParameter("document_ids", "array", "文档 ID 列表（1-8）", required=True,
                              items={"type": "string"}),
                ToolParameter("max_items", "integer", "最大条目数（1-100）", minimum=1, maximum=100, default=50),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("inspect_source_assets"),
    )

    registry.register(
        ToolDefinition(
            name="render_evidence_answer",
            description=(
                "调查后提交来源闭合的答案计划。本地 verifier 检查每个 item ID、必需"
                "事实、query facet 与章节类型后渲染规范本地文本。此工具不能改变诊断"
                "或安全状态。"
            ),
            parameters=[
                ToolParameter("session_id", "string", "会话 ID", required=True),
                ToolParameter("answer_sections", "array", "答案章节（section_type+source_item_ids）",
                              required=True, items={"type": "object"}),
                ToolParameter("covered_query_facets", "array", "已覆盖 query facet", items={"type": "string"}),
                ToolParameter("uncovered_query_facets", "array", "未覆盖 query facet", items={"type": "string"}),
            ],
            category="diagnose",
            backend="codex_executor",
        ),
        provider("render_evidence_answer"),
    )

    # -- Incident 全量组（category=incident） ------------------------------
    registry.register(
        ToolDefinition(
            name="analyze_incident",
            description=(
                "创建一个不可变 Incident 快照，解析诊断工件，查询 KG_v2 假设并构建"
                "本地验证的 Jira 报告。返回 case_id 供后续 incident 工具使用。"
            ),
            parameters=[
                ToolParameter("query", "string", "案件 query", required=True),
                ToolParameter("evidence_resources", "array", "证据资源数组（最多 24）",
                              required=True, items={"type": "object"}),
                ToolParameter("log_summary", "string", "可选日志摘要（JSON 字符串）"),
            ],
            category="incident",
            backend="codex_executor",
        ),
        provider("analyze_incident"),
    )

    registry.register(
        ToolDefinition(
            name="index_log_package",
            description=(
                "当调用方显式需要一个不可变诊断包索引以进行更深检查时，作为 incident "
                "摄取入口。返回 case_id、工件数与事件数。"
            ),
            parameters=[
                ToolParameter("query", "string", "案件 query", required=True),
                ToolParameter("evidence_resources", "array", "证据资源数组（最多 24）",
                              required=True, items={"type": "object"}),
                ToolParameter("log_summary", "string", "可选日志摘要（JSON 字符串）"),
            ],
            category="incident",
            backend="codex_executor",
        ),
        provider("index_log_package"),
    )

    registry.register(
        ToolDefinition(
            name="parse_incident_scope",
            description=(
                "把 Jira/query 中的参考时间规范化为独立的本地时间窗，用于解析前筛选"
                "诊断包。"
            ),
            parameters=[
                ToolParameter("query", "string", "query 文本", required=True),
                ToolParameter("resource_hints", "array", "资源名提示（最多 48）", items={"type": "string"}),
            ],
            category="incident",
            backend="codex_executor",
        ),
        provider("parse_incident_scope"),
    )

    for tool_name, description, params in [
        (
            "get_jira_snapshot",
            "读取不可变 Jira/案件快照。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "get_incident_scope",
            "读取规范化 query 时间 scope（工件选择期间使用）。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "get_incident_evidence_pack",
            "返回来源封闭的 Incident Evidence Pack v3，供本地或模型侧合成。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "list_artifacts",
            "列出来源与压缩包成员工件（含 hash 与解析状态）。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "inspect_archive_manifest",
            "检查压缩包祖先与成员，不执行/不向调用方路径解压附件。",
            [
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("artifact_id", "string", "工件 ID", required=True),
            ],
        ),
        (
            "search_diagnostic_events_by_time",
            "按 ISO 本地时间区间在已抽取事件中搜索。",
            [
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("start_time", "string", "起始 ISO 时间", required=True),
                ToolParameter("end_time", "string", "结束 ISO 时间", required=True),
                ToolParameter("query", "string", "关键字", required=True),
                ToolParameter("limit", "integer", "最大条数", minimum=1, maximum=200, default=50),
            ],
        ),
        (
            "extract_log_time_windows",
            "返回从压缩包成员按 query 时间 scope 流式抽取的不可变、来源行绑定窗口。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "inspect_stacktrace",
            "检查规范化栈帧，同时保留检测点与根因边界。",
            [
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("trace_id", "string", "栈 trace ID"),
            ],
        ),
        (
            "inspect_environment",
            "检查来源绑定的软件、驱动、OS 与硬件版本观察。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "inspect_evtx",
            "返回选定 EVTX 工件的来源绑定 Windows provider/event/time/data 记录（含时间对齐）。",
            [
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("artifact_id", "string", "EVTX 工件 ID", required=True),
            ],
        ),
        (
            "inspect_dump",
            "返回有界 minidump 进程、异常、OS 与加载模块元数据；符号化栈单独报告。",
            [
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("artifact_id", "string", "DMP 工件 ID", required=True),
            ],
        ),
        (
            "query_kg_hypotheses",
            "返回 KG_v2 候选的支持/反驳/缺失证据假设矩阵。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "retrieve_similar_cases",
            "返回词法相关 SourceCase 记录作为非权威线索，绝不作为正式知识或证明。",
            [
                ToolParameter("case_id", "string", "Incident 案件 ID", required=True),
                ToolParameter("limit", "integer", "最大条数", minimum=1, maximum=20, default=10),
            ],
        ),
        (
            "propose_next_tests",
            "返回按信息增益、风险与成本排序的安全下一步测试。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "plan_reproduction",
            "构建不执行的受控复现与观察计划。绝不控制生产设备或运行附件脚本。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
        (
            "compare_reproduction_runs",
            "比较两个不可变 incident run 的重复签名。命中证明复发，不代表受控复现或已验证修复。",
            [
                ToolParameter("baseline_case_id", "string", "基线案件 ID", required=True),
                ToolParameter("candidate_case_id", "string", "候选案件 ID", required=True),
            ],
        ),
        (
            "compare_incident_environments",
            "比较两个已解析 incident 环境快照，不把差异当作根因。",
            [
                ToolParameter("left_case_id", "string", "左案件 ID", required=True),
                ToolParameter("right_case_id", "string", "右案件 ID", required=True),
            ],
        ),
        (
            "render_incident_report",
            "返回本地验证的 Jira 友好报告；此工具不能改变 Jira 状态。",
            [ToolParameter("case_id", "string", "Incident 案件 ID", required=True)],
        ),
    ]:
        registry.register(
            ToolDefinition(
                name=tool_name,
                description=description,
                parameters=params,
                category="incident",
                backend="codex_executor",
            ),
            provider(tool_name),
        )

    # -- 写侧只读面（category=write） --------------------------------------
    registry.register(
        ToolDefinition(
            name="w9_inspect_document",
            description=(
                "检查一个原始知识文档并决定其处理策略（doc_strategy）与推荐步骤。"
                "只读，不写 KG、不修改文档。用于把手册/指南/流程/SOP 分类为知识源。"
            ),
            parameters=[
                ToolParameter("path", "string", "文档路径", required=True),
            ],
            category="write",
            backend="w9",
        ),
        provider("w9_inspect_document"),
    )

    registry.register(
        ToolDefinition(
            name="w9_build_structured_sections",
            description=(
                "把文档解析为结构化语义分块（Section/FAQ/table 等）。只读，返回分块"
                "结果，不落盘。用于为知识入库准备结构化内容。"
            ),
            parameters=[
                ToolParameter("path", "string", "文档路径", required=True),
            ],
            category="write",
            backend="w9",
        ),
        provider("w9_build_structured_sections"),
    )

    registry.register(
        ToolDefinition(
            name="w9_build_section_cases",
            description=(
                "把文档章节转换为 section-case 与内容寻址 chunk_manifest（approved=false）。"
                "只读，返回草案，不写 KG。"
            ),
            parameters=[
                ToolParameter("path", "string", "文档路径", required=True),
            ],
            category="write",
            backend="w9",
        ),
        provider("w9_build_section_cases"),
    )

    registry.register(
        ToolDefinition(
            name="w9_build_root_checklist",
            description=(
                "扫描一个根目录，为其中每个文档生成处理清单（含是否已入库判断）。"
                "只读，不修改任何文档或 KG。"
            ),
            parameters=[
                ToolParameter("root", "string", "根目录", required=True),
                ToolParameter("include_sop", "boolean", "是否包含 SOP 文档", default=False),
            ],
            category="write",
            backend="w9",
        ),
        provider("w9_build_root_checklist"),
    )


# ---------------------------------------------------------------------------
# 默认后端（可被 backend_provider 覆盖）
# ---------------------------------------------------------------------------


def _default_backend(name: str) -> ToolBackend | None:
    """内置默认后端：基于 agents/tools 的 EvidenceToolAgent 与基础能力。

    完整 incident_runtime / KGV2ReadModel / CorpusReadTools 后端由调用方通过
    ``build_default_registry(backend_provider=...)`` 注入；此处提供
    parse_evidence 等不依赖系统实例的最小实现。
    """

    from .router import EvidenceToolAgent
    from .executor import ReadEvidenceToolExecutor

    executor = ReadEvidenceToolExecutor()

    if name == "parse_evidence":
        def _run(arguments: dict[str, Any]) -> dict[str, Any]:
            envelope = executor.execute(
                arguments.get("resource") or {},
                tool=str(arguments.get("tool") or "auto"),
                max_bytes=int(arguments.get("max_bytes") or 65536),
            )
            return envelope.to_dict() if hasattr(envelope, "to_dict") else _envelope_to_dict(envelope)
        return _CallableBackend(_run)

    if name == "parse_evidence_context":
        from .context_parser import EvidenceContextParserAgent

        agent = EvidenceContextParserAgent()

        def _run_context(arguments: dict[str, Any]) -> dict[str, Any]:
            return agent.parse_context(
                arguments.get("root") or "",
                max_bytes=int(arguments.get("max_bytes") or 65536),
                limit=int(arguments.get("limit") or 0),
            )
        return _CallableBackend(_run_context)

    return None


def _envelope_to_dict(envelope: Any) -> dict[str, Any]:
    """把 dataclass envelope 序列化为 dict（兼容现有 ToolResultEnvelope）。"""

    return {
        "schema_version": getattr(envelope, "schema_version", TOOL_RESULT_SCHEMA),
        "tool": getattr(envelope, "tool", ""),
        "call_id": getattr(envelope, "call_id", ""),
        "call_fingerprint": getattr(envelope, "call_fingerprint", ""),
        "status": getattr(envelope, "status", "ok"),
        "resource_id": getattr(envelope, "resource_id", ""),
        "payload": getattr(envelope, "payload", {}),
        "observations": [
            {
                "observation_id": getattr(item, "observation_id", ""),
                "field": getattr(item, "field", ""),
                "value": getattr(item, "value", None),
                "confidence": getattr(item, "confidence", 0.0),
                "evidence_ids": list(getattr(item, "evidence_ids", [])),
                "source_ids": list(getattr(item, "source_ids", [])),
            }
            for item in getattr(envelope, "observations", [])
        ],
        "evidence_ids": list(getattr(envelope, "evidence_ids", [])),
        "source_ids": list(getattr(envelope, "source_ids", [])),
        "errors": list(getattr(envelope, "errors", [])),
        "excluded": list(getattr(envelope, "excluded", [])),
        "safety": getattr(envelope, "safety", {}),
        "observability": getattr(envelope, "observability", {}),
    }


__all__ = [
    "ToolDefinition",
    "ToolParameter",
    "ToolRegistry",
    "ToolExecutionResult",
    "ToolBackend",
    "build_default_registry",
]
