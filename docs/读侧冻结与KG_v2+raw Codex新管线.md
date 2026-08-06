# 读侧冻结与 KG_v2+raw Codex Agent 管线

> 更新时间：2026-08-04
> 当前 System Prompt：`debug_agent_system.kg_raw_codex.system.v16`
> 当前回答契约：`debug_agent_system.kg_raw_codex_answer.v5`
> 文档导航与状态分层见 [读侧文档索引](读侧文档索引.md)。

## 1. 决策与边界

2026-07-30 起，现有 KG_v2/SAG 读侧作为对照基线冻结：

`DebugAgentSystem.start → SAG → Evidence Pack → Composer`

冻结表示本轮不再修改上述路径的召回、诊断裁决、回答组织器和运行配置，也不因新增
raw 文档重建其 SAG。2026-07-30 的历史冻结文件及 SHA-256 记录仍保留在
`config/read_side_frozen_manifest_v1.json`，可执行：

```bash
python scripts/verify_frozen_read_pipeline.py
```

2026-08-04 在不重跑既有冻结基线的前提下，进一步对当前代码快照建立 v2 冻结边界。
冻结范围覆盖当前 KG_v2/SAG 读侧、KG_v2+raw Codex 旁路、Incident Runtime、关键配置、
SAG 数据快照及既有正式评测输入。v2 同时支持逐文件哈希和 Python 包目录树哈希，清单为：

`config/read_side_frozen_manifest_v2.json`

验证命令为：

```bash
python scripts/verify_frozen_read_pipeline.py \
  --manifest config/read_side_frozen_manifest_v2.json
```

当前清单共检查 16 个文件快照和 4 个代码目录树。任何文件内容、目录内 Python 文件数量
或包级聚合哈希发生变化都会报告 drift 并返回非零状态。Read Runtime v3 只能在新的
`src/debug_agent_system/read_runtime_v3/`、独立配置、独立入口和独立测试中演进，不能修改
v2 清单中的冻结路径。这里的“冻结”是代码与数据边界冻结；既有基线测试结论继续引用
本文档已有存档，不为建立 v2 清单而重复运行。

2026-08-04 Read Runtime v3 全量 shadow validation 前后都显式使用 v2 清单校验，
两次均为 `checked=20`、`frozen=true`、`drift=[]`。不带 `--manifest` 的历史命令
仍指向 v1，不应用它判断 2026-08-04 的 v2 冻结状态。

独立旁路调整为：

```text
Query
  → 本地生成 required facets、术语上下文与术语搜索契约
  → 构造只读 KG_v2+raw 语料工作区
  → 选择 Codex 运行载体
       ├─ Responses API：.env.local + 通用只读函数工具
       └─ Codex CLI：本地登录态 + 只读 shell
  → Codex Agent
       ├─ 自主规划调查
       ├─ 反复搜索、读取 KG_v2 与 raw
       ├─ 按导航入口继续追查正文与媒体
       └─ 组织结构化完整回答
  → 本地来源/facet/术语搜索/并列方案/媒体/安全发布校验
  → 审计产物
```

它位于 `src/debug_agent_system/kg_raw_codex/`，不注册到
`DebugAgentSystem.start()`，也不修改 QA 项目的 embedded snapshot。

## 2. 已删除的自制检索层

该旁路不再包含以下实现：

- `KGRawCorpusTools`；
- `search_kg_raw_corpus`；
- `read_kg_raw_file`；
- 手写中文分词、候选文件打分和 Top-K 排名；
- 带自制召回、重写或排序决策的 function-calling 循环；
- 本地代码替 Codex 决定“下一轮应该搜索什么”的逻辑。

本地代码不会先给 Codex 一个预排序候选集。Codex 面对的是受限但完整的语料工作区，
可以根据问题自行决定搜索词、文件范围、读取顺序、是否沿导航链接继续查找，以及何时
已经获得足够证据。

这解决了旧实现的结构性问题：自制 ranker 一旦没有把“可以进入系统.docx”、某个飞书
子文档或复合 Query 的第二个任务排进 Top-K，后续模型再强也看不到缺失资料。

## 3. Codex Agent 怎样运行

同一条管线实现了两个可显式选择、不会相互静默回退的 `AgentRunner`：

| 运行载体 | 执行器 | 认证 | 模型怎样访问证据 |
|---|---|---|---|
| Responses API | `CodexResponsesAgentRunner` | 仓库 `.env.local` | `list_files`、`search_text`、`read_text` |
| 本地 Codex | `CodexCliAgentRunner` | 本机 Codex 登录态 | `rg --files`、`rg`、`sed` 等只读 shell |

Responses API 每轮都由模型决定是否以及如何调用三个通用只读工具：

- `list_files`：按模型给出的 glob 列出路径，不返回相关性分数；
- `search_text`：按模型给出的 literal/regex 和路径范围返回行号与有界上下文；
- `read_text`：读取模型指定文件的精确行区间。

这些是证据目录的文件访问层，不做中文分词、Query 改写、候选评分、Top-K 或下一步
决策。Codex 决定搜索词、范围、迭代和停止条件。本地只执行函数、限制路径不能逃出
`data/raw` / `data/kg_v2` / 临时 DOCX 视图，并记录参数、输出 hash 和实际读取来源。

多轮使用 `store:false`，同时回传 Responses API 提供的 encrypted reasoning 与函数
结果，因此服务端不保留该响应仍能延续同一推理链。

本地 Codex 运行时使用 `codex exec --ephemeral --ignore-user-config --ignore-rules`，工作
目录固定为临时语料工作区，sandbox 为 `read-only`，approval policy 为 `never`。它不读
`.env.local`，也不会继承仓库内的定向规则；模型直接用通用只读 shell 自主规划文件名搜索、
正文检索和分段阅读。两种运行载体最终都必须返回同一份严格 JSON Schema，并经过同一
verifier；批次不能因为某个载体额度耗尽而静默换模型或换认证方式。

唯一证据域为：

- `data/raw`；
- `data/kg_v2`。

为了让 Agent 能可靠读取 DOCX，管线会确定性展开所有 DOCX 为临时 Markdown 视图：

- 展开视图只用于搜索和阅读，不成为新事实源；
- 每个视图首行保存规范 `SOURCE_PATH`；
- 段落、表格、导航链接和媒体引用一并保留；
- 图片与附件物化到结果目录旁的 `kg_v2_raw_assets/`；
- 最终引用必须回到真实 `data/raw/...` 或 `data/kg_v2/...` 路径。

这一步是格式适配，不是检索或排名。

## 4. System Prompt 与职责划分

独立 System Prompt 当前版本为：

`debug_agent_system.kg_raw_codex.system.v16`

源码位于：

`src/debug_agent_system/kg_raw_codex/prompt.py`

Prompt 要求 Codex：

- 先拆解问题和制定调查计划；
- 对每个 required facet 分别搜索，不停在第一个看似相关的文件；
- 先用有界文件定位，再读取命中上下文，避免把全语料命中一次性灌入上下文；
- 把 KG_v2 对象/关系与 raw 原文步骤分开使用；
- 遇到目录、超链接、relationship ID 或 wiki token 时继续追查本地正文；
- 按前置准备、用户任务顺序、分支、验证、风险和资料缺口组织答案；
- 多条件回答使用“分支标题 + 分支内独立编号”，不能把分支名与步骤铺成同级列表；
- 同一答案证据中明确列出的并列方案必须逐项登记并说明是已展开、风险受控地展示，
  还是因证据缺失而省略；
- `allow_*` 只控制系统修复、引导修复和破坏性磁盘命令，不是普通原文命令的总开关；
- 外部脚本链接不静默删除；缺少内容、版本、哈希审计时必须披露并优先推荐系统内置方法；
- 只引用实际读取的来源；
- 只引用语料展开时暴露的真实图片；
- 不用常识补写参数、命令、根因或成功状态。

本地代码仍保留少量不可交给生成模型的发布职责：

| 职责 | 执行方 |
|---|---|
| 调查计划、搜索词、搜索迭代、跨文档编织 | Codex |
| 文件相关性和读取顺序 | Codex |
| 回答章节、步骤、分支与措辞 | Codex |
| 证据目录边界 | 本地代码 |
| Query required facets | 本地确定性任务契约 |
| 已批准术语解析、搜索义务与术语版本 | 本地确定性术语契约 |
| 原词/规范词搜索、其他检索词与迭代顺序 | Codex |
| 来源路径存在性与越界检查 | 本地 verifier |
| facet 覆盖、来源 ledger 和图片路径校验 | 本地 verifier |
| 从答案实际采用的 raw 原文独立发现并列方案 | 本地 verifier |
| 每个并列方案的展开/受控/缺口状态与答案标题一致性 | 本地 verifier |
| 函数参数、状态、输出 hash、token 审计 | 本地产物 |

因此，这是一条“Codex 负责智能调查，本地负责不可绕过的发布约束”的管线，不是把另一套
手写检索器包装成 Tool 交给模型调用。

## 5. 通用闭包机制

管线不会为 Dism++、安全模式、某个文件名或某条 benchmark Query 写专用回答规则。
每次运行先从 Query 生成 `required_facets`：

- 用户明确请求的 operation、entity 等任务；
- 并列操作对象；
- 具名工具或环境的准备工作；
- 系统、引导、磁盘、镜像、恢复等操作的安全前置条件。

Codex 最终返回：

- `answer_markdown`；
- `coverage_ledger`；
- `procedure_variant_ledger`；
- `files_read`。

本地 verifier 会拒绝：

- 漏掉 required facet；
- 把 inventory、review queue 或人工审核建议当作 facet 证据或答案事实来源；
- 漏掉答案证据中明确声明的同级方案/方法；
- `covered` 没有实际读取来源；
- 引用不存在、未读取或越界的路径；
- 模型声称读取、但工具审计中没有实际返回过内容的文件；
- 引用不是从语料物化出的媒体；
- 把资料说明写成本次现场已确认根因；
- 输出不符合固定 schema。

### 5.1 并列方案覆盖与命令边界

facet 闭包只证明“至少有一种做法回答了用户任务”，不能证明同一份原文中的替代路径
都已组织进答案。v5 契约新增 `procedure_variant_ledger`：本地代码在 Codex 选定并实际
读取来源之后，从用于闭包 `query_task` 的 raw 正文中识别“方案一/二/三”、
“方法一/二/三”和“第一/二/三种操作方法”。若同一来源存在两个或更多同级路径，
每个路径都必须同时出现在 ledger 和独立 Markdown 标题中，并标记为：

- `expanded` / 已展开；
- `guarded` / 风险受控地展示；
- `omitted_evidence_gap` / 因证据缺失而省略。

发现未核验的外部脚本链接时，不能以“安全”为由无声删除整条方案，也不能把它写成
已验证首选。答案可以保留原文 URL，但必须明确写明“脚本内容、版本、哈希未核验，
优先使用可审计的系统内置方法”。本地门禁会核对状态、警示语和原文 URL。

命令门禁仍采用类别白名单：`allow_system_repair_commands` 只控制 SFC/DISM 类系统修复，
`allow_boot_repair_commands` 只控制 bootrec/bcdboot 类引导修复，
`allow_destructive_storage_commands` 只控制清盘、格式化、删除分区等破坏性磁盘操作。
这些字段不是所有命令的总开关；直接完成 Query、原文可追溯且非破坏性的命令不会因
三个字段为 false 而被拒绝。

校验失败时，Codex 会看到错误列表并重新调查或重写，最多执行配置的发布校验轮数。
本地 verifier 不替 Codex补搜候选，也不再运行自制 ranker。

### 5.2 术语表怎样进入读侧

术语层在 Codex 调查前由本地确定性 resolver 读取一次，但不产生 Top-K 文档列表：

```text
Query + Answer Scope
  → TerminologyResolver.resolve(context=...)
  → TERM_RESOLUTION
       ├─ resolved_mentions：已批准且已消歧
       ├─ ambiguous_mentions：保留歧义和所需上下文
       ├─ search_hint：只做可选宽召回
       └─ entity_relation：只做可选范围扩展
  → TERMINOLOGY_SEARCH_CONTRACT
       ├─ approved_equivalence：原始表达与规范名都必须实际搜索
       ├─ optional_expansions：是否使用由 Codex 自主决定
       └─ governance_only_paths：不得提升为运行时事实
  → Codex 自主搜索和读取
  → terminology_search_audit
```

这里的强制项仅是“搜索过原词和规范词”，不会指定文件、分数、Top-K、搜索顺序或停止
条件。命中为空允许继续调查，但不能记为已读来源。这样既避免已批准别名导致漏召回，
又不重新引入手写检索器。

运行时权限来自：

- `terminology/entity_ontology.json` 中正式概念、已批准别名和正式关系；
- `terminology/curated_terms.json` 中有效批准；
- 由上述输入确定性生成的 `DebugConcept`、`TermExpression`、`TermSense` 和关系。

`noun_terminology_inventory`、`noun_discovery_report`、review queue 和人工审核建议是治理
视图，不是证据真值；`pending`、`rejected`、`needs_re_review` 不会自动参与等价解析。
结果产物同时保存 `terminology_manifest` 的版本、revision 和计数，以及解析上下文、
搜索契约和搜索审计，便于复现“本次回答使用了哪版术语表”。

## 6. 认证与网关

配置文件为 `config/kg_v2_raw_codex.json`。仓库默认仍是 Responses API，同时声明可由
命令行显式选择的本地 Codex 模型：

```json
{
  "runtime": "responses_api",
  "model": "gpt-5.4",
  "cli_model": "gpt-5.3-codex-spark",
  "credential_source": ".env.local",
  "transport": "non_streaming",
  "reasoning_effort": "medium"
}
```

选择 `responses_api` 时，`.env.local` 必须包含 `OPENAI_BASE_URL` 与
`OPENAI_API_KEY`；选择 `codex_cli` 时，必须已有本地 Codex 登录态，不读取上述变量。
当前内部网关已验证 `gpt-5.4` 的非流式 Responses API；本地登录态已验证
`gpt-5.3-codex-spark` 与 `gpt-5.6-luna`。模型由命令行或配置显式选择；两个载体都不回退到
旧 Chat Completions，也不在失败时自动互换模型或认证方式。

## 7. 使用方式

单条 Query：

```bash
PYTHONPATH=src python scripts/run_kg_raw_codex_answer.py \
  "安装软件或驱动后 Windows 出现异常，需要进入安全模式排查时应该怎么进入？" \
  --runtime codex_cli --model gpt-5.3-codex-spark \
  --output data/results/kg_raw_codex_cli/safe_mode.json
```

批量运行：

```bash
PYTHONPATH=src python scripts/run_kg_raw_codex_batch.py \
  --input data/results/read_side_codex_comparison_20260730/comparison_results.json \
  --output-dir data/results/kg_raw_codex_cli_batch \
  --runtime codex_cli --model gpt-5.6-luna
```

结果记录完整回答、required facets、coverage ledger、术语版本/解析/搜索审计、实际读取
文件、暴露媒体、工具轨迹、token 使用和 verifier 尝试。批处理的 `.failure.json` 与
`batch_manifest.json` 会保留模型
额度、超时或发布门禁失败，报告生成器不会用旧成功产物覆盖当前失败项。

## 8. 当前真实验证

Query：

> 安装软件或驱动后 Windows 出现异常，需要进入安全模式排查时应该怎么进入？

Responses Agent 的早期验证产物为：

`data/results/kg_raw_codex_responses_smoke_20260731/safe_mode_v8.json`

Codex 自主执行 15 次工具调用，读取了 7 份 raw 文档和 1 份 KG_v2 文件，其中包括
旧自制检索曾漏掉的：

- `data/raw/aoi_debug_agent_sources/可以进入系统.docx`；
- `data/raw/aoi_debug_agent_sources/无法进入系统.docx`。

最终回答正确区分：

- 可以进入 Windows 时通过 Shift + 重启进入 WinRE；
- 不能进入系统时通过连续中断启动进入自动修复；
- 进入安全模式后再处理最近安装的软件或驱动；
- 资料能说明的操作与本次现场尚未确认的根因。

Responses transport 与通用 Agent 工具定向测试 7/7 通过。该结果证明搜索词、
文件选择和迭代来自 Codex，而不是为这条 Query 新增文件名特判。

2026-08-03 以本地登录态和 v14 Prompt 运行第 6–10 条。第 6–8 条使用
`gpt-5.3-codex-spark`；Spark 额度耗尽后，经人工明确允许，第 9–10 条使用
`gpt-5.6-luna` 重新运行。模型切换是显式批次参数，不是运行时静默回退：

| # | 模型 | 状态 | 读取文件 | 工具调用 | 媒体 | token | 说明 |
|---:|---|---|---:|---:|---:|---:|---|
| 6 | `gpt-5.3-codex-spark` | 通过 | 7 | 16 | 4 | 279402 | Dism++ 引导修复 |
| 7 | `gpt-5.3-codex-spark` | 通过 | 5 | 18 | 3 | 642817 | Dism++ 系统修复 |
| 8 | `gpt-5.3-codex-spark` | 通过 | 10 | 36 | 1 | 1015609 | 第二轮通过，正确使用分支标题与分支内编号 |
| 9 | `gpt-5.6-luna` | 通过 | 8 | 7 | 1 | 312235 | 同时覆盖 Windows 与 BIOS 快速启动 |
| 10 | `gpt-5.6-luna` | 通过 | 6 | 6 | 0 | 按 Query 范围仅展开 Windows 快速启动 |

五条回答均通过相同的 required facet、来源、媒体与结构 verifier。该组结果适合验证同一管线
在两个本地登录态模型上的可运行性，但由于模型不同，不能把第 6–10 条表述为严格的同模型
质量或成本横评。第 9、10 条成功后，批处理已清理对应的旧 `.failure.json`。

## 9. 已知限制与下一步

当前实现优先完成能力边界纠正，仍有两个需要继续优化的工程问题：

- 每次运行都会展开完整 DOCX 集合，后续应做基于源文件 hash 的只读缓存；
- Codex Agent 的 token 和时延显著高于自制 Top-K 管线；Responses `store:false` 下每轮
  需要续传历史与加密 reasoning，本地 CLI 的多次验证重试同样会放大 token 成本。第 8 条
  因首稿存在重复 `files_read` 被拒绝并完整重跑，累计超过 100 万 token，说明发布约束
  已 fail-closed，但重试成本仍需优化。
  后续应通过工具输出压缩、DOCX 缓存和可配置调查预算降低成本，但不能
  重新引入由本地 ranker 决定模型能看到哪些资料的单点瓶颈。

后续评测应比较 required facet closure、来源可追溯率、图片相关性、unsupported claim、
资料缺口准确性、token/时延，以及相对冻结 SAG 基线和纯 Codex 的答案质量。
