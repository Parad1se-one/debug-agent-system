# Tool agents: evidence parsing entries

这些工具是给 W1/W6、读侧 O-LOG/O-ESC、QA adapter 或后续 MCP 使用的**证据解析入口**。它们只把外部证据转换为结构化元数据，不直接写 KG、不改变会话状态、不执行现场动作。

## Stable entry

其他 agent 默认只依赖统一入口。函数入口适合普通 agent 调用；`EvidenceToolAgent` 适合批量路由或需要 `infer_and_parse()` 的场景：

```python
from debug_agent_system.agents.tools import (
    EvidenceContextParserAgent,
    EvidenceToolAgent,
    parse_attachment_evidence,
    parse_document_evidence,
    parse_dmp_evidence,
    parse_evidence_context,
    parse_image_evidence,
    parse_jira_evidence,
    parse_log_package_evidence,
    parse_proj_evidence,
)

proj = parse_proj_evidence("/path/to/recipe.proj")
jira = parse_jira_evidence("https://jira.example.com/browse/SMTAOITS-1234")
attachment = parse_attachment_evidence({"name": "DLOG_init.zip", "path": "/tmp/DLOG_init.zip"})
document = parse_document_evidence("/path/to/report.pdf")
image = parse_image_evidence({"name": "capture.png", "path": "/tmp/capture.png"})
log_package = parse_log_package_evidence({"name": "DLOG_init.zip", "path": "/tmp/DLOG_init.zip"})
dmp = parse_dmp_evidence("/path/to/MEMORY.DMP")
context = parse_evidence_context("/path/to/tool_sample_dir")

agent = EvidenceToolAgent()
proj2 = agent.parse("proj", "/path/to/recipe.proj")

# 读侧统一 envelope；用于 Evidence Gap Resolver 和 Tool Harness。
from debug_agent_system.agents.tools import ReadEvidenceToolExecutor
envelope = ReadEvidenceToolExecutor().execute(
    {"kind": "log_package", "name": "startup.log", "path": "/tmp/startup.log"}
)
```

CLI 等价入口：

```bash
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence jira \
  'https://jira.example.com/browse/SMTAOITS-1234'
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence attachment \
  '{"name":"DLOG_init.zip","path":"/tmp/DLOG_init.zip"}'
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence document /path/to/report.pdf
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence image /path/to/capture.png
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence log_package \
  '{"name":"DLOG_init.zip","path":"/tmp/DLOG_init.zip"}'
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence dmp /path/to/MEMORY.DMP
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence proj /path/to/recipe.proj
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli parse-evidence-context \
  data/imports/tool_samples --out data/results/tool_samples_context_parse.json
```

保留兼容入口：`parse-proj` / `parse-jira` / `parse-attachment` / `parse-document` / `parse-image`。

## Contract

- type: Worker / Tool
- owner: `src/debug_agent_system/agents/tools`
- responsibility: 解析 proj/Jira/附件/文档/图片/日志包证据为 JSON-like 结构化证据。
- non_goals:
  - 不写 `data/kg` 主图或 review queue。
  - 不执行 `.proj`、脚本、二进制或压缩包内容；`.proj` tar 只读 manifest 和 bounded text hints，不落盘解包。
  - 不解压日志包，不读取二进制日志正文；只允许对白名单文本日志/附件/Office XML 做有界 preview/hints；PDF 只读 bounded bytes；旧 Office 只读 OLE header；图片只读 bounded header metadata，不 OCR、不读像素；不联网抓 Jira。
  - 不把 evidence 直接裁决为 KG 事实；裁决仍归 W2-W6 或读侧 verifier。
- upstream_inputs:
  - `tool`: `attachment|document|image|jira|proj|log_package|dmp`
  - `payload`: string/path/dict metadata
- outputs:
  - `AttachmentParseResult` / `DocumentParseResult` / `DmpParseResult` / `ImageParseResult` / `JiraParseResult` / `ProjParseResult` / `LogPackageParseResult`
  - 所有输出都带 `tool_entry.schema_version/tool/agent_id`
  - 读侧统一包装输出 `debug_agent_system.read_tool_result.v1`，包含
    `call_fingerprint/observations/evidence_ids/source_ids/excluded/safety/observability`
- failure_modes:
  - unknown tool -> `EvidenceToolError(status=parse_failed)`
  - missing/unreadable file -> structured result or `parse_failed`，不抛出到上游编排
  - Jira network unavailable -> 不联网，输出 `metadata_only` + `fetched=false`
- observability:
  - `observability.agent_id`: `TOOL-ATTACHMENT|TOOL-DOCUMENT|TOOL-IMAGE|TOOL-JIRA|TOOL-PROJ|TOOL-LOG-PACKAGE|TOOL-ROUTER`
  - `tool_entry`: 记录统一入口路由信息

## Tool boundaries

| Tool | Agent | Safe behavior | Main output |
|---|---|---|---|
| attachment | `AttachmentParserAgent` | 分类附件元数据；仅对白名单文本附件做 bounded preview；不解压、不 OCR | `evidence_role`, `mime_guess`, `text_preview_read`, `key_hints`, `archive_extracted=false` |
| document | `DocumentParserAgent` | PDF 只读 bounded bytes；docx/xlsx/pptx 只读 zip central directory 和白名单 XML preview；doc/xls/ppt 只读 OLE header；不渲染、不 OCR、不执行宏/公式 | `document_format`, `pdf_version`, `page_count_hint`, `entries`, `text_preview_read`, `macros_executed=false`, `formulas_evaluated=false`, `ocr_performed=false` |
| image | `ImageParserAgent` | 只读 PNG/JPEG/GIF/WebP/BMP header；不读像素、不 OCR、不调用视觉模型 | `image_format`, `width`, `height`, `megapixels`, `pixels_read=false`, `ocr_performed=false` |
| jira | `JiraParserAgent` | 离线解析 URL/issue key/Markdown 标题；可读取本地 `data/imports/jira_offline/raw/fault_details/*.json`；不联网 | `issue_keys`, `urls`, `title_hints`, `version_hints`, `site_hints`, `description_hints`, `comment_hints`, `offline_details`, `summary_hint`, `fetched=false` |
| proj | `ProjParserAgent` | 有界读取文本预览；tar 型 `.proj` 只读 manifest + 白名单文本条目 preview；不执行、不修改、不落盘解包 | `entries`, `archive_format`, `key_hints.project_names/app_versions/ip_addresses/file_roles/model_types`, `executed=false`, `mutated=false` |
| log_package | `LogPackageParserAgent` | zip 读 central directory + 白名单文本 hints；7z/rar 通过 `bsdtar -tf` 只读 manifest；`.log/.txt/.csv` 可做 bounded text hints；不解压 | `entries`, `text_hints`, `detected_roles`, `has_dmp/has_evtx/has_startup_log/has_dlog` |
| dmp | `DmpParserAgent` | 只读 `.dmp/.mdmp` bounded header；不执行 WinDbg、不扫描完整内存、不提取内存正文 | `header_signature`, `dump_kind`, `architecture_hint`, `windbg_ready`, `debugger_executed=false`, `full_content_read=false` |
| evidence_context | `EvidenceContextParserAgent` | 读取 `source_manifest.json` 和所在目录上下文，批量路由 raw 文件到上述安全解析器；不解压、不执行、不联网 | `contexts[]`, `tool_evidence`, `summary_hints`, `safety` |

## Effective strategy

- 写侧 review evidence 使用这些工具补充 `evidence_pack.tool_evidence`，但是否入图仍由 W6 人工批准。
- 读侧可把工具输出当作 O-LOG/O-ESC 的辅助证据，但不能绕过 C 门控和 EA 校验。
- `ReadEvidenceToolExecutor` 将 parser-specific 字典归一为来源绑定 observation；只有
  `supports_retrieval=true` 的 observation 可补全知识/诊断上下文。
- 图片只读 header，`supports_retrieval=false`，明确记录 `ocr_not_supported`。
- 如果证据解析失败，上游应降级为 metadata-only evidence，而不是丢弃整条 episode/session。
## Agent contract summary

- owner: `src/debug_agent_system/agents/tools/router.py` (`EvidenceToolAgent`) plus parser subagents; context-level packaging lives in `src/debug_agent_system/agents/tools/context_parser`.
- inputs:
  - `tool: attachment|document|image|jira|proj|log_package|dmp` plus string/dict/path payload, or `infer_and_parse(payload)`.
  - `parse_evidence_context(root)` accepts a `source_manifest.json`, a sample directory, or a Jira offline root.
  - `parse_many(items)` accepts list entries with explicit `tool/payload` or raw payloads for inference.
- outputs:
  - Tool-specific parse result dicts: `AttachmentParseResult`, `DocumentParseResult`, `DmpParseResult`, `ImageParseResult`, `JiraParseResult`, `ProjParseResult`, `LogPackageParseResult`, or structured `EvidenceToolError`.
  - Context parse result dict: `EvidenceContextParseResult`.
  - Every result includes `tool_entry` and `observability.agent_id` where parser supports it.
- failure_modes:
  - Unknown tool -> `EvidenceToolError` with `status=parse_failed`.
  - Missing/bad local file -> parser-specific non-content or parse-failed result; no mutation.
- non_goals:
  - No network fetch.
  - No archive extraction.
  - No executable project-file evaluation.
