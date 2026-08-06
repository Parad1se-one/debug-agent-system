# KG_v2 读侧主动证据补全与 Codex Tool Harness

> 更新时间：2026-07-30
> 本文描述当前读侧模型层。旧文件
> [`KG_v2读侧主动证据补全与DeepSeek Tool Harness.md`](archive/read-side/20260727/KG_v2读侧主动证据补全与DeepSeek%20Tool%20Harness.md)
> 保留迁移前设计和历史实测记录，
> 不再代表 active runtime。

## 1. 目标形态

当前读侧不是“把 Query 和一堆文档直接交给 Codex”，而是：

```text
Query / 会话 / 调用方资源
  → Query task v2
  → SAG_v2 Variant + Chunk 双通道宽召回
  → KG_v2 Variant / Trace / Branch / Safety 确定性裁决
  → Evidence Gap 只读解析
  → Evidence Pack v2
  → Codex 调查、选择和排序本地条目 ID
  → 本地 Verifier
  → canonical 事实、来源和媒体确定性渲染
```

职责边界：

- KG_v2 runtime 独占 Variant 锁定、计划编译、BranchRule、安全确认和 `verified_fix`；
- Codex 可以调查候选、整篇文档、KG 路径、图片、附件和调用方资源；
- Codex 不能直接写事实正文，只能选择 Evidence Pack 中已有的 `source_item_id`；
- 本地 verifier 失败、模型超时或配置缺失时，回退 `EvidenceAnswerComposer`；
- 模型输出永远不是 KG_v2 的第二事实源，也不会回写 KG。

## 2. 配置与密钥

```yaml
read_llm:
  provider: codex
  enabled: false
  answer_composer_enabled: false
  model: gpt-5.3-codex
  base_url: ""
  timeout_seconds: 60
  max_tool_rounds: 3
  max_answer_documents: 8
  max_answer_chunks: 64
  max_answer_input_chars: 60000
  answer_fallback: deterministic
```

`base_url` 留空时，客户端按以下优先级读取：

1. 显式构造参数；
2. 进程环境变量；
3. 项目根目录 `.env.local`。

使用的变量为：

```text
OPENAI_BASE_URL
OPENAI_API_KEY
```

客户端只使用 Python 标准库，不修改进程环境，不记录 Authorization header、完整网关错误
正文或密钥。配置默认关闭，因此离线测试和不具备模型配置的部署仍可运行。

## 3. Codex Tool Surface

`CodexReadToolHarness` 暴露 8 个 strict、只读或纯渲染 Tool：

| Tool | 输入 | 输出 | 权限边界 |
|---|---|---|---|
| `diagnose_start` | Query、路由上下文、资源 | 标准 `AgentResponse`、Evidence Pack | 诊断由本地 runtime 完成 |
| `diagnose_step` | session、用户反馈、资源 | 下一轮标准响应 | 不允许 Codex 选分支 |
| `retrieve_evidence` | Query、limit | Variant 候选、supporting chunks、paths、trace | 最多 20 个候选 |
| `expand_document_context` | Query、document IDs | 按源顺序的批准 Chunk | 最多 8 文档、64 Chunk |
| `inspect_kg_path` | Family/Variant ID | TraceStep、Action、Outcome、BranchRule、Evidence | 不执行 Action |
| `inspect_source_assets` | Query、document IDs | 图片/附件、caption、Chunk 和来源锚点 | 最多 100 个资源 |
| `parse_evidence` | 调用方 resource | source-bound observation | 有界只读 parser |
| `render_evidence_answer` | session、章节和 item IDs、facet 声明 | 本地校验和渲染后的响应 | 不改变诊断或安全状态 |

明确没有暴露：

- `select_branch`
- `execute_action`
- `confirm_risk`
- `mark_resolved`
- KG/review queue 写入

启用 `IncidentEvidenceRuntime` 后，同一 Harness 另有一组案件级 strict Tool，用于安全索引
Jira/日志包、读取 Incident Evidence Pack v3、查询事件/栈/环境/日志窗口、检查 EVTX/DMP、
查看 KG 假设与下一步测试并提交本地验证报告。它们不改变上述八个冻结基线工具的语义。
完整清单与输入边界见
[读侧 Jira 诊断数据包与 Incident Evidence Runtime](读侧-Jira诊断数据包与Incident-Evidence-Runtime.md)。

## 4. 调查如何进入最终回答

仅有检索工具而没有收口工具，会出现“Codex 看到了更多证据，但最终仍返回首次确定性回答”
的问题。当前通过 `render_evidence_answer` 闭环：

1. Codex 先调用 `diagnose_start`，获得本地运行时响应及 Evidence Pack；
2. 必要时使用 retrieve/expand/KG/assets/parse 工具核对证据；
3. Codex 提交章节类型、`source_item_ids`、covered facets 和 uncovered facets；
4. 本地 `CodexEvidenceAnswerVerifier` 校验；
5. 通过后从 canonical item 恢复正文、命令安全摘要、来源和媒体；
6. 拒绝时保留首次确定性回答，不接受 Codex 自由文本代替。

Verifier 至少检查：

- schema version；
- item ID 必须属于当前 Pack；
- `required` 条目不得遗漏；
- item 不得重复；
- `uncertainty`、`required_info` 必须保留原章节；有来源正文可在四个内容章节间重组；
- 已支持 Query facet 不得遗漏；
- unsupported facet 声明必须与 Pack 完全一致；
- `required_info` 必须最后；
- evidence floor 必须由批准且可追溯的正文证据满足。

## 5. 两种启用方式

### 5.1 Tool Harness

```bash
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli \
  read-tool-harness "检测界面出现拍照失败问题" \
  --codex
```

旧 `--deepseek` 参数仅作为命令行迁移别名，内部仍调用 Codex；新代码和文档应使用
`--codex`。

### 5.2 单次答案 Composer

设置：

```yaml
read_llm:
  enabled: true
  answer_composer_enabled: true
```

普通 `DebugAgentSystem.start/step` 会在确定性 Evidence Pack 之后调用一次
`CodexEvidenceAnswerComposer`。模型仍只返回 item ID 计划；失败时确定性降级。

Tool Harness 适合需要主动调查和资源解析的入口；单次 Composer 适合已有宽召回证据充分、
只需改善组织的入口。两者共享同一个 verifier 和安全边界。

## 6. 当前验证

截至 2026-07-30：

- Codex client、8 个 Tool、closed-pack renderer、未知 ID 拒绝和 fail-open 定向测试通过；
- 真实 `gpt-5.3-codex` 单 Query 已通过 `.env.local` 配置调用；
- 纯 Codex 分享页 47 条全量复跑完成：Evidence Pack 47/47，正文证据门槛 42/47，
  Query 子任务闭包 38/47，Codex Composer 采用 40/47；
- 7 条确定性降级中，2 条为瞬时传输错误，5 条为无批准正文证据；
- 原回答通用追加追问命中 44、新管线为 0；原回答磁盘/引导命令直接暴露命中 36、
  新管线为 0；
- 模型采用结果由本地 verifier 通过后才进入 `answer_sections`；
- 缺少模型配置时不会发起网络请求；
- 读侧仍不依赖第三方 Python 库。

纯 Codex 分享基线为：

```text
http://intranet-host/share/dzlSbu1VMCOG4F2nFPXhdcSWT7vepgDJ2YvLckdjHPoV-mxs
```

仓库中的离线 MHTML 正是该 token 的快照，包含 47 组 Query/Answer；分享服务当前返回
空的 HTTP 502 时，可使用该快照和冻结 JSON 重放。对比必须同时报告召回/证据闭包、
安全门、来源、Codex 是否采用和完整回答，不把字符长度或客观门禁冒充语义正确率。
完整结果见
[KG_v2读侧Codex升级与分享Query对比](archive/read-side/20260803/KG_v2读侧Codex升级与分享Query对比.md)。

## 7. 后续优化

优先级从高到低：

1. 为公开分享 JSON 导入器增加固定 schema fixture，避免依赖在线服务；
2. 将 Tool 调用的 token、延迟、轮数和命中收益写入统一观测；
3. 评测“无 Harness、单次 Composer、Harness + render”三条路径的增益；
4. 为未解析 PDF/附件补 source parser，使关键证据先进入 Pack；
5. 对大文档增加分层目录 Tool，降低一次返回 64 Chunk 的上下文成本；
6. 在独立 shadow gate 达标前保持默认关闭。
