# Read Runtime v3：设计原理、收益、取舍与相关工作

> 文档性质：目标架构、已落地影子实现与技术决策说明（Architecture Decision Record / Technical Report）  
> 更新时间：2026-08-04  
> 状态：**v3 已实现独立 shadow runtime；尚未接管冻结生产回答**  
> 适用范围：`debug_agent_system` 读侧的 KG_v2、SAG、raw 文档、Jira/诊断数据包、Codex 调查工具、答案生成与本地门禁  
> 不包含：写侧抽取、审核、canonical KG 写入，以及生产设备上的自动执行器

## 1. 技术摘要

Read Runtime v3 不是“再写一套检索器”，也不是“把更多原文塞给 Codex”。它要解决的是当前几条读侧路径已经分别具备能力、但缺少统一调查语义的问题：

- KG_v2 原生运行时能够锁定 Variant、编译 Trace、选择 Branch、控制风险并验证 `verified_fix`；
- SAG 能够快速定位 Variant、Chunk 和文档候选；
- raw 文档保留最完整、最可审计的原始资料、图片和附件；
- Incident Evidence Runtime 能够解析 Jira、日志、EVTX、DMP、环境和时间线；
- Codex 能够根据证据缺口规划下一次搜索，并把跨文档、跨工具结果组织成可读回答。

这些能力目前拥有不同入口、Scope、Evidence Pack、状态和 verifier。v3 的核心不是把它们重写成一个巨型模块，而是建立一套共同的请求、证据、假设、策略和响应协议，让不同证据源在同一次调查中协作。

一句话定位：

> **Read Runtime v3 是一个 evidence-bounded、graph-constrained、policy-gated 的 agentic investigation runtime：允许 Agent 自主规划调查，但用统一证据闭包、KG 诊断结构和本地策略门禁约束结论与动作。**

它采用五项关键分工：

1. **SAG 找候选，不成为第二事实源；**
2. **raw 提供原文事实，不自行决定诊断状态；**
3. **KG_v2 提供诊断结构、分支和安全语义，不冒充当前现场事实；**
4. **Incident 提供本次案件的观察事实，不自动写回正式知识；**
5. **Codex 负责规划、迭代检索、候选假设和表达，本地代码负责证据准入、风险、状态和最终校验。**

这套设计优先优化的不是“回答看起来更聪明”，而是以下工程性质：

- 同一个结论能回到原始文件、KG 对象或现场事件；
- 检测点、故障域、候选根因、确认根因和已验证修复不再混为一谈；
- 简单 Query 不付出完整 Incident 调查成本，复杂案件又不受固定 Top-K 限制；
- 模型不可用时仍能确定性降级；
- 新增一种证据源或 parser 时，不必再新增一条孤立回答管线。

## 2. 先澄清：这里的 “v3” 指什么

截至 2026-08-04，仓库已有独立的 `ReadRuntimeV3` 类、`ReadRequest` / `ReadTask` /
`EvidenceRecord` / `AnswerPlan` / `ReadResponse` 契约、命令行入口和影子评测入口。它位于
`src/debug_agent_system/read_runtime_v3/`，通过 `config/read_runtime_v3.yaml` 配置，默认仍是
`shadow_mode=true`。

因此，“已实现 v3” 和“v3 已发布接管”是两件事：

- **已实现**：新契约、Evidence Fabric、provider adapter、工具注册表、Codex 调查
  planner、确定性 fallback、Policy、verifier、Trace 契约和影子 diff 都有独立代码；
- **尚未接管**：影子模式仍返回冻结基线的正式 `answer/status`，v3 产出位于
  `shadow.proposed_answer/proposed_status`，本地 verifier 不通过或 planner 失败时回退到基线；
- **不能混淆**：`debug_agent_system.incident_evidence_pack.v3` 是 Incident provider 的证据包
  schema，不是整个 Read Runtime v3。

被 v3 统一适配、但仍可独立回滚的原有运行形态如下：

| 运行形态 | 主要能力 | 当前边界 |
|---|---|---|
| KG_v2 + SAG 原生运行时 | Variant、Trace、Branch、状态推进、安全门、`verified_fix` | 主入口较重；对大型案件材料的证据建模有限 |
| Evidence Pack + Codex Tool Harness | 在闭合条目集合中扩展、选择和编排 | 工具执行器集中；仍继承主链的证据结构 |
| KG_v2+raw Codex 独立旁路 | Codex 自主搜索 raw/KG、阅读原文、组织答案 | 与主状态机隔离；拥有独立 Scope、coverage 和 verifier |
| Incident Evidence Runtime | 时间窗口、artifact、事件、栈、环境、相关性、假设和 next-best test | 当前为可选旁路；默认关闭，shadow mode 不改变基线答案 |

本文后续的“v3”既表示长期架构，也表示当前与该架构一致的 shadow
实现。涉及生产行为时会明确写“冻结基线”或“v3 shadow”，不用同一个名称暗示
已切流。

### 2.1 已落地的实现边界

| 实现 | 代码 | 当前责任 |
|---|---|---|
| v3 契约 | `read_runtime_v3/contracts.py` | Request、Task、Evidence、Hypothesis、Trace、Plan、Policy、Response |
| Evidence Fabric | `read_runtime_v3/fabric.py` | 内容寻址 Evidence ID、连接、去重、snapshot |
| Task Normalizer | `read_runtime_v3/tasking.py` | 区分知识流程、已观测现场故障和 source-only Trace 重建 |
| Provider adapter | `read_runtime_v3/providers.py` | 冻结基线、KG/SAG、raw、Incident 和请求附带上下文转为证据 |
| Planner | `read_runtime_v3/planner.py`、`agentic.py` | 确定性 bootstrap/fallback；可选 Codex 只读工具调查与 strict Answer Plan |
| Policy / verifier | `read_runtime_v3/policy.py`、`verifier.py` | 模型外状态、风险、引用、Trace 和未支持 claim 检查 |
| Orchestrator | `read_runtime_v3/runtime.py` | 协调 provider/planner/policy/verifier，产出影子 diff，安全回退 |
| 评测映射 | `read_runtime_v3/evaluation.py` | 将 v3 路由、Evidence ID、Hypothesis/Trace 投影到正式 Benchmark |

两个 CLI 分开“跑一条 Query”和“跑 shadow Benchmark”：

```bash
PYTHONPATH=src python scripts/run_read_runtime_v3.py "Query"
PYTHONPATH=src python scripts/run_read_runtime_v3_benchmark.py --output-dir /path/to/run
```

默认 `evidence_first_bootstrap` 只做可重现证据组装，不会假装已经做了 Codex 的语义裁决。
将 `planner` 显式设为 `codex_agentic` 后，Codex 才会在轮次和调用预算内执行受限调查。

## 3. 为什么现在需要 v3

### 3.1 问题已经从“检索答案”变成“组织调查”

普通文档问答可以近似为：

```text
Query → 找到相关段落 → 按来源组织回答
```

但典型 Jira/现场诊断更接近：

```text
问题描述 + 参考时间 + 诊断包
  → 判断哪些文件与哪些时间窗口有关
  → 抽取事件、调用栈、环境和生命周期信号
  → 与 KG 中的故障机制、分支条件和历史结果对照
  → 形成多个候选假设及反证
  → 选择最能区分候选的下一步测试
  → 在风险边界内给出回答或继续追问
```

后者不是一次 Top-K 检索，也不是一次 Prompt 编写问题。它需要一个可持续更新的调查状态。

### 3.2 现有数据契约彼此分离

当前至少同时存在：

- `debug_agent_system.answer_evidence_pack.v2`；
- `debug_agent_system.incident_evidence_pack.v3`；
- `debug_agent_system.kg_raw_codex_answer.v5`；
- Query Task、AnswerScope、IncidentScope 等不同任务表达；
- 文档 Chunk、KG 对象、artifact/event/stack/environment 等不同证据表达。

这些 schema 在各自路径内是合理的，但跨路径组合时会遇到以下问题：

- 一个 raw 原文段落和一个 SAG Chunk 是否是同一事实，没有共同 Identity；
- 一个 Incident Event 支持哪个 KG 假设，缺少统一 Evidence Link；
- 同一 facet 在三个 coverage ledger 中可能有不同名称和状态；
- 每条路径都要单独实现 source closure、排除理由、预算和 verifier；
- Codex 需要知道自己正在调用哪一套工具、能相信哪些字段、何时停止。

v3 首先统一这些**边界契约**，而不是先统一所有内部实现。

### 3.3 Incident 不能继续只是“答案覆盖层”

当前 `DebugAgentSystem.start()` 可以先运行 Incident 分析，将结果写入 metadata；在非 shadow 模式下，Incident 报告还可以成为 active answer。这种集成适合快速验证，但长期存在语义问题：

- KG 主链和 Incident 层分别得出判断，缺少共同假设状态；
- Incident 可能改变最终回答，却没有参与 Variant/Branch 的共同裁决；
- raw 中已有但未进入 KG/SAG 的关键文档，Incident 路径不能自然补查；
- 两边的 evidence ID 和 source closure 不相通。

v3 中 Incident 应成为第一类 evidence provider，而不是独立的答案替换器。

### 3.4 Agent 能力提升后，更需要模型外边界

固定 Prompt 只能处理预期内的顺序。让 Codex 自己选择搜索词、工具和下一步证据后，系统能处理更多未见过的问题；同时也引入了新风险：

- 在证据不足时提前收敛；
- 把日志中的检测位置写成唯一根因；
- 忽略反证或并列方案；
- 被附件或文档中的指令性文本影响；
- 展示或执行不应授权的动作；
- 模型自评“足够”但来源并未闭包。

因此 Agent 自主性越强，Evidence Fabric、Policy Adjudicator 和 verifier 越不能依赖同一个模型自我约束。

### 3.5 数据规模不允许每次全量交给模型

当前工作区的 `data/raw` 约 5.2 GB、包含 3400 余个文件；`data/kg_v2` 和 `data/kg_v2_sag` 也分别是数百 MB 量级。即使模型能够使用本地搜索，也不应对每个 Query 全量遍历全部资料。

v3 必须利用多层证据访问：

1. SAG/KG/术语先建立候选域；
2. raw 只对候选来源做原文闭包；
3. Incident 根据参考时间、artifact 类型和当前证据缺口有界解析；
4. 只有在当前假设无法区分时才扩窗、换源或增加工具调用。

## 4. 设计约束与不变量

v3 的架构选择受以下不变量约束：

1. **来源不闭包，不形成正式事实。** 每个正式 claim 必须能回到合法 Evidence ID。
2. **观察不等于解释。** 日志错误、栈顶函数和异常模块首先是 detection point 或 observed fact。
3. **候选不等于锁定。** KG 召回结果、相似案例和模型提出的解释都不能直接成为唯一根因。
4. **恢复不等于修复。** 一次重启恢复、短时未复发和 Jira 关闭都不能自动成为 `verified_fix`。
5. **答案许可与执行许可分离。** 能展示原文命令，不代表系统可以在设备上执行该命令。
6. **破坏性动作默认拒绝。** 格式化、固件、磁盘/分区、生产设备动作等必须经过独立策略和人工确认。
7. **模型失败不使安全门失效。** 内容组织可以降级，执行授权必须 fail-closed。
8. **SAG 不是第二事实源。** 它是 KG/raw 的服务索引，revision 不一致必须拒绝或重建。
9. **现场证据不直接污染 canonical KG。** Incident 和历史案例先进入候选/审核链。
10. **迁移可回滚。** v3 不能以一次性重写方式替换已验证的 KG_v2 状态机。

## 5. 总体架构

### 5.1 控制面

```text
ReadRequest v3
      │
      ▼
Task Normalizer
Scope / Facet / Entity / Time / Risk / Resource
      │
      ▼
Codex Planner ─────────────────────────────────────────────┐
      │                                                    │
      │ 选择只读 provider、搜索词、扩窗和下一步证据         │
      ▼                                                    │
Tool Registry                                              │
  ├─ sag.*                                                 │
  ├─ kg.*                                                  │
  ├─ raw.*                                                 │
  ├─ incident.*                                            │
  ├─ evidence.*                                            │
  └─ hypothesis.*                                          │
      │                                                    │
      ▼                                                    │
Unified Evidence Fabric ── coverage / contradiction / gap ─┘
      │                         达到停止条件
      ▼
Hypothesis State + Answer Plan + Next Best Test
      │
      ▼
Local Policy Adjudicator
      │
      ├─ 状态、Variant、Branch、风险、verified_fix
      ▼
Codex Renderer 或 Deterministic Renderer
      │
      ▼
Claim / Citation / Coverage / Safety Verifier
      │
      ▼
ReadResponse v3
```

### 5.2 证据面

```text
raw file / KG object / SAG chunk / ZIP member / EVTX record / DMP stream
                         │
                         ▼
       parser / index / graph adapter / artifact intake
                         │
                         ▼
                    EvidenceRecord
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
   EvidenceLink     DerivationLink    ContradictionLink
        │                │                 │
        └────────────────┴─────────────────┘
                         ▼
            Claim / Hypothesis / Answer Section
```

控制面回答“下一步调查什么、当前能下什么结论”；证据面回答“这个结论来自哪里、经过了什么解析、是否存在反证”。两者分离后，既能更换模型，也能增加 parser，而不重写诊断真值语义。

## 6. 核心设计决策

### 6.1 统一契约，不强行统一存储

v3 首先定义共同对象：

- `ReadRequest`：Query、会话、资源、环境、时间、调用方策略；
- `ReadTask`：任务类型、facets、实体、范围、风险、预算；
- `EvidenceRecord`：类型、内容摘要、来源、hash、时间、parser/version、观察/推断属性；
- `EvidenceLink`：支持、反证、派生、同源、同实体、时间邻近；
- `HypothesisRecord`：状态、置信度、支持、反证、缺口、下一步测试；
- `AnswerPlan`：回答章节、claim、来源、风险和缺失信息；
- `ReadResponse`：答案、状态、来源、调查轨迹、预算、排除项和降级说明。

但 KG 图、SQLite SAG、raw 文件、artifact manifest 不必迁入一个物理数据库。强行统一存储会带来大规模迁移和语义损失；统一 Identity、Link 和查询接口已经足以实现跨源调查。

**收益：** 低迁移风险；保留各存储优势；新增 provider 成本下降。  
**代价：** 需要稳定 ID、revision 和缓存协议；跨源去重比单库查询更复杂。

### 6.2 Evidence Fabric 不是“大 Prompt”

Evidence Fabric 保存当前调查允许使用的闭合证据集合及其关系，而不是把所有正文拼成字符串。大文件默认只保留引用、hash、成员和有界摘要；需要原文时再通过 provider 读取指定范围。

每条 Evidence 至少记录：

- 稳定 Evidence ID；
- source artifact / KG object / raw path；
- 行号、字节、Chunk、record 或 frame anchor；
- 内容 hash 与数据 revision；
- parser 和 schema version；
- 观察事实还是派生判断；
- 时间范围和时区语义；
- 支持、反证、排除或截断原因；
- 读取成本、缓存和 continuation token。

**收益：** 引用可验证；可重复查询；模型上下文受控；能做 claim-level closure。  
**代价：** schema 和 provenance 管理成本上升；需要处理同一事实的多个表示。

### 6.3 明确 KG、SAG、raw、Incident 的职责

| 组件 | v3 中的角色 | 不能承担的角色 |
|---|---|---|
| SAG | 低成本候选定位、过滤、排序 | 最终事实、唯一根因、执行授权 |
| KG_v2 | 诊断语义、Variant/Trace/Branch、RequiredInfo、安全和 Outcome | 当前现场是否真的发生、raw 原文替代品 |
| raw | 权威原文、图片、附件和来源闭包 | Variant 锁定、Branch 选择、风险裁决 |
| Incident | 本次案件的 event/stack/environment/timeline/correlation | canonical KG、自动确认根因或修复 |
| 历史案例 | 弱先验、相似签名和候选测试 | 正式知识或当前案件直接证据 |

这种分工使不同证据可以互相制约：KG 告诉系统“什么机制值得检查”，Incident 告诉系统“现场观察到了什么”，raw 告诉系统“正式资料具体写了什么”，SAG 负责快速找到它们。

### 6.4 Agent 负责调查规划，本地代码负责裁决

Codex 适合负责：

- 将复合 Query 分解为 facets；
- 选择先查 KG/SAG、raw 还是 Incident；
- 根据新观察调整搜索词和工具顺序；
- 提出候选假设和可区分候选的测试；
- 对证据充分的内容做排序、比较和可读表达。

本地确定性代码继续负责：

- Evidence 准入与路径边界；
- source ID、hash、revision 和引用合法性；
- 重复、派生和矛盾关系；
- Variant/Branch 锁定；
- 高风险动作和人工确认；
- `verified_fix`；
- coverage、并列方案和最终 verifier。

这不是因为模型“不能推理”，而是因为这些决策需要可复现、可审计、可测试，并且不能随模型版本或措辞变化。

### 6.5 假设必须有显式状态机

建议的状态不是单一 confidence，而是：

```text
candidate
  ├─ observed_support
  ├─ kg_supported
  ├─ contradicted
  ├─ needs_evidence
  ├─ locked_root_cause
  └─ verified_fix
```

其中：

- `observed_support`：案件证据支持某个故障域；
- `kg_supported`：KG 中存在可追溯机制或处理路径；
- `contradicted`：存在明确反证；
- `locked_root_cause`：本地门禁确认当前证据足以锁定；
- `verified_fix`：实施动作并完成规定的恢复/耐久验证。

“GPU API illegal memory access” 可以支持 GPU/显示执行链异常，但不能仅凭该字符串区分驱动软件、GPU 硬件、电源或 PCIe 稳定性。状态机让系统能够给出有用的故障域判断，同时不假装已经确认唯一根因。

### 6.6 检索由证据缺口驱动，而不是固定 Top-K

每轮调查更新：

- required / optional / excluded facets；
- 已覆盖事实；
- 当前假设的支持和反证；
- 尚未区分的候选；
- 可能改变结论的 next-best evidence；
- 剩余时间、token、I/O 和工具预算。

停止条件包括：

1. Query facets 已闭包且答案状态明确；
2. 当前证据足以锁定 Variant 或明确保持 `needs_evidence`；
3. 下一工具的预期信息增益低于成本；
4. 需要用户、现场或外部权限才能继续；
5. 达到预算或安全边界。

这样既避免一次召回不足，也避免 Agent 无限制搜索。

### 6.7 工具注册表按 provider 分域

建议将集中式 if/elif executor 演进为带版本和能力声明的 provider registry：

```text
sag.search_candidates
kg.inspect_variant
kg.expand_trace
raw.search_text
raw.read_range
raw.list_media
incident.inspect_manifest
incident.search_events
incident.read_time_window
incident.inspect_stack
incident.compare_environment
evidence.link
evidence.check_closure
hypothesis.update
answer.submit_plan
```

每个工具统一返回：

- schema/version；
- Evidence IDs；
- exclusions/warnings；
- truncated/continuation；
- cost/latency；
- read-only / side-effect / approval 分类。

这能让 Codex 使用一致协议，也允许后续对接 MCP 风格工具而不把内部架构绑定到某个协议实现。

### 6.8 Answer Plan 与渲染分离

Codex 不直接提交一段无法验证的最终 prose，而是提交 `AnswerPlan`：

- 章节和顺序；
- 每个 claim；
- claim 使用的 Evidence IDs；
- 当前结论等级；
- 风险提示；
- 需要用户补充的信息；
- next-best test；
- 并列方案的展开/风险受控/证据缺失状态。

通过 verifier 后，可以由 Codex renderer 生成自然表达；模型不可用时，也可以由确定性 renderer 使用同一 Answer Plan 输出。这样“内容正确性边界”和“表达质量”不再绑定。

### 6.9 策略裁决独立于模型

`allow_*` 不应是一个控制所有命令的总开关。策略至少应区分：

- 信息是否可以展示；
- 非破坏性内置命令是否可以引用；
- 高风险命令是否只能说明风险而不展开；
- 工具是否可以实际执行；
- 是否需要人工批准；
- 当前会话是否满足前置条件。

例如，来源明确且直接完成用户请求的非破坏性系统命令可以展示；格式化磁盘、固件、生产设备动作则属于另一策略域。策略结果与执行点分离，便于审计和回放。

### 6.10 用 shadow 迁移，而不是大爆炸重写

建议迁移顺序：

1. 先定义 v3 contract，并为当前三类 Evidence Pack 写适配器；
2. 建立稳定 Evidence ID、source closure 和 provider registry；
3. 在不改变答案的情况下 shadow 生成 v3 Evidence Fabric；
4. 对文档 QA、KG locked、SOP、Jira/诊断包分别做双跑；
5. 先切换 Answer Plan/verifier，再切换 planner；
6. 稳定后才删除重复路径和兼容规则。

这样可以逐项归因差异：是 Scope、检索、证据、假设、策略还是表达发生了变化。

## 7. 设计收益

| 收益 | 具体表现 |
|---|---|
| 统一调查语义 | 文档 QA、KG 诊断和 Jira 案件共享 facet、evidence、hypothesis、gap 和 response 状态 |
| 更强来源审计 | claim 可回到 raw 行、KG 对象、ZIP 成员、EVTX record 或 DMP stream |
| 降低误诊 | detection point、correlation、fault domain、root cause、verified fix 分层 |
| 更好处理复杂 Query | Agent 可根据证据缺口跨 KG/SAG/raw/Incident 迭代，而非受一次 Top-K 限制 |
| 更低上下文成本 | 大文件引用化、时间窗口化、按需原文读取和 continuation |
| 安全边界稳定 | 模型变化不影响 Variant、Branch、风险和 `verified_fix` 语义 |
| 更易扩展 | 新 parser/provider 接入统一 EvidenceRecord，而不是复制一条回答管线 |
| 更易评测 | 可分开定位 task、retrieval、evidence、diagnosis、policy、rendering 的失败 |
| 更可靠退化 | 无 LLM 时使用同一 Answer Plan 与确定性 renderer；执行许可始终 fail-closed |
| 支持多轮调查 | 新证据可以更新假设和 gap，而不是每轮重新拼接 Query 后丢失历史语义 |

## 8. 主要取舍与代价

| 取舍 | 得到什么 | 付出什么 / 缓解方式 |
|---|---|---|
| 统一契约而非统一数据库 | 低迁移风险、保留专用存储 | 跨源 Identity 和去重复杂；用稳定 ID、hash、revision 缓解 |
| 类型化 Evidence 而非全文上下文 | 可验证、可关联、低 token | parser/schema 演进成本；保持原始 source anchor 和 fallback |
| 时间窗口优先 | 降 I/O、内存和噪声 | 可能漏掉早期诱因；保留完整 manifest、支持扩窗与 sidecar 索引 |
| Agent 自主调查 | 适应未见过问题、跨源补证 | 成本和非确定性增加；使用预算、停止条件、trace 和 verifier |
| 本地策略裁决 | 安全、可回放、跨模型稳定 | 规则/策略维护成本；只固化不变量，不写 Query 定向答案规则 |
| 保守假设状态 | 少把相关性写成因果 | 回答显得不够“果断”；同步给出 next-best test 提高可行动性 |
| KG 宽召回后证据相交 | 避免相似知识越级归因 | 当前案件缺证据时只能 `needs_evidence`；通过 raw/Incident 补查 |
| DMP/EVTX 默认只读有界解析 | 安全、无需外部调试器也可运行 | 深层符号归因有限；把符号化作为受控可选 provider |
| Answer Plan + 双 renderer | 表达与事实边界分离、可降级 | 多一道协议和校验；以共享 contract 抵消重复实现 |
| shadow 分阶段迁移 | 可比较、可回滚 | 一段时间内双路径并存；设明确退出标准和删除窗口 |

## 9. 为什么不选其他方案

### 9.1 只让 Codex 全量搜索 raw

优点是实现快、表达自然，也能发现尚未进入 KG/SAG 的资料。问题是：

- 5 GB 级目录每次调查成本不可控；
- 无统一 source/evidence identity；
- Variant、Branch、风险和 `verified_fix` 无法由自由搜索保证；
- 复杂诊断包仍需要专用 parser 和时间关联；
- 难以区分“没找到”和“资料不存在”。

它适合作为能力旁路和上限对照，不适合作为唯一生产运行时。

### 9.2 继续向 `runtime/system.py` 增加分支

这能保持单入口，但会让 Query、KG、Incident、LLM、回答和迁移策略进一步耦合。短期修复容易，长期无法解释一次结论究竟由哪个证据面和哪个裁决面产生。v3 应保留单一外部入口，但通过 orchestrator、provider 和 policy 分层，而不是单文件集中。

### 9.3 用 Incident 报告覆盖原回答

这能快速上线案件能力，却让 KG 主链和 Incident 各自完成一次推理。长期会产生状态冲突、重复答案和 source closure 分裂。v3 将 Incident 降为 evidence provider，让统一假设和策略决定最终输出。

### 9.4 全部改成确定性规则

确定性规则适合安全、schema、状态和已知结构，不适合开放式 facet 分解、跨文档搜索词调整和复杂回答编织。全规则方案会回到不断追加领域/Query 特判的问题。

### 9.5 全部交给 LLM，包括根因和风险

它减少代码，却牺牲复现、审计和安全。模型自评不能成为 `verified_fix` 或破坏性动作授权。v3 选择“探索与表达由模型负责，事实和授权由本地系统裁决”。

### 9.6 一次性替换现有运行时

KG_v2 原生运行时已经沉淀了 Variant、Trace、Branch 和 `verified_fix` 不变量。大爆炸重写会同时改变检索、状态、答案和安全，任何回归都难以定位。v3 必须是契约适配和 shadow 双跑驱动的渐进迁移。

## 10. 一个 CUDA 闪退案件在 v3 中如何运行

假设用户提交：软件版本 1.4.9，设备在 8 月 1 日 21:30 和 8 月 3 日 06:04 正常检测时闪退，并附带诊断包。

### 10.1 Task Normalizer

解析出：

- 任务：incident diagnosis；
- 对象：AOI 设备、业务应用、GPU/CUDA 图像处理链；
- 两个独立参考时间点，不扩成两天连续窗口；
- 资源：ZIP、可能的 EVTX/DMP/日志；
- facets：发生了什么、是否复发、故障域、是否可锁定根因、下一步验证。

### 10.2 Incident provider

- 安全枚举 archive，记录所有成员和 exclusion；
- 只对目标日期日志流式取窗；
- 从 EVTX 抽取 Display/TDR/WER 等事件；
- 从 DMP 抽取异常、进程、模块和版本；
- 建立应用 fatal、驱动复位、进程重启和跨日期相同签名的时间关系。

此时可形成 observed facts，但还不能宣布驱动是唯一根因。

### 10.3 KG/SAG provider

使用稳定锚点宽召回：

- 错误 `-217`；
- `illegal memory access`；
- `cv::cuda::GpuMat::upload`；
- GPU/显示驱动执行链；
- 业务组件和版本。

地址、构建路径和长 hash 只作审计字段，不成为强召回 facet。

### 10.4 raw provider

对 KG/SAG 候选引用的源文档做原文闭包；若 KG 中只有导航或摘要，则读取 raw 中对应正文、版本适用条件、图片和附件。

### 10.5 Hypothesis update

可能形成：

| 假设 | 支持 | 反证/缺口 | 状态 |
|---|---|---|---|
| GPU/显示驱动执行链异常 | CUDA fatal、Display/TDR、DMP 时间与进程闭环 | 尚不能区分驱动软件、GPU、供电或 PCIe | `observed_support` |
| 业务算法单点越界 | 调用点位于模板匹配/GpuMat upload | 若同时发生系统级 TDR，则单纯业务越界解释不足 | `candidate` |
| GPU 硬件/供电/PCIe 不稳 | 跨日期复发、驱动复位可能支持 | 缺硬件对换、温度、电源和单变量实验 | `needs_evidence` |

### 10.6 Policy 与 Answer Plan

系统可以回答“当前证据更支持 GPU/显示执行链故障域”，但不能写成“已确认显卡驱动版本是唯一根因”。下一步应选择能最大区分候选的测试，例如：

- 固定业务版本，仅对比驱动版本；
- 固定驱动，仅做 GPU/设备对换；
- 采集同窗温度、供电、PCIe/WHEA、TDR 和应用事件；
- 在隔离环境执行受控复现，并预先定义停止和回滚条件。

最终回答同时给出事实、综合判断、未确认项、下一步测试和来源，而不是只输出一个根因标签。

## 11. 可以借鉴的相关工作

v3 不是照搬某个 RAG/Agent 框架，而是组合多类成熟思想。借鉴时必须同时说明“采用什么”和“不采用什么”。

### 11.1 Agentic retrieval 与自适应调查

| 工作 | 可借鉴 | 不直接照搬 |
|---|---|---|
| [ReAct](https://openreview.net/pdf?id=WE_vluYUL-X) | reasoning/action/observation 交错；新观察驱动下一工具 | 自由文本 thought/action 难审计；v3 使用 strict tools、预算、停止条件和 Evidence Pack |
| [Self-RAG](https://openreview.net/pdf?id=hSyW5go0v8) | 按需检索，显式检查证据是否支持回答 | 同一 LLM 的自评不能成为生产通过条件；最终由本地 closure/coverage gate 裁决 |
| [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) | 按 Query 复杂度选择零/单步/多步检索 | 通用复杂度分类不能替代 `answerable/diagnosable/executable` 和风险等级 |

### 11.2 GraphRAG 与知识图谱诊断

| 工作 | 可借鉴 | 不直接照搬 |
|---|---|---|
| [From Local to Global / GraphRAG](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/) | 图索引、跨文档 overview、local/global 查询模式 | 社区摘要不能承担现场根因、Branch 和执行授权 |
| [HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html) | 从 Query/Incident anchor 做局部图扩展和多跳整合 | 图连通度不等于因果强度，高连接节点不能自动成为根因 |
| [IndustryAssetEQA](https://aclanthology.org/2026.acl-industry.49/) | episode/incident 中心建模、telemetry + FMEA-KG、provenance 和模型外 safety gate | 资产 schema 不能直接套用；需要保留 AOI 的 Variant/Trace、日志、版本和媒体结构 |

### 11.3 Incident diagnosis、observability 与复现

| 工作/规范 | 可借鉴 | 不直接照搬 |
|---|---|---|
| [LLM-based Agents for Root Cause Analysis](https://arxiv.org/abs/2403.04123) | Agent 动态选择日志、指标等证据；原始讨论需先结构化 | 云服务 RCA 不能覆盖 AOI 的驱动、硬件、图像和现场动作语义 |
| [Nissist](https://arxiv.org/abs/2402.17531) | 把 troubleshooting guide 和历史处置组织为连续计划 | 不因“多 Agent”而拆分；v3 优先单 orchestrator + typed providers |
| [TSGuard](https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/34/TSGuard-Automated-User-Centric-Incident-Diagnosis-for-AI-Workloads-in-the-Cloud) | 历史 on-call 知识和结构化迭代试错 | 历史案例只作弱先验，不能覆盖本次直接证据 |
| [StepFly](https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/76/StepFly-Agentic-Troubleshooting-Guide-Automation-for-Incident-Diagnosis) | 将并列步骤、证据依赖和停止条件组织成 DAG | 物理设备动作不能默认并行，必须考虑风险、互斥和回滚 |
| [OpenTelemetry Logs](https://opentelemetry.io/docs/specs/otel/logs/data-model/) 与 [W3C Trace Context](https://www.w3.org/TR/trace-context/) | 统一时间、Resource、Trace/Span 和跨源关联字段 | 旧 AOI 日志常无 trace ID，需要保留时间/线程/文件/栈等弱关联及置信度 |
| [Google SRE Incident Management](https://sre.google/sre-book/managing-incidents/) 与 [Emergency Response](https://sre.google/sre-book/emergency-response/) | 活的 incident state、受控演练、退出/回滚和成功判据 | 不必照搬组织角色；重点借鉴事实/假设/行动日志和复现控制面 |
| [AIOpsLab](https://arxiv.org/abs/2501.06706) | 在隔离环境评估工具轨迹、fault injection 和 next-best test | 不在生产 AOI 上自动注入故障；执行必须隔离并批准 |

### 11.4 Provenance、约束和引用质量

| 工作/规范 | 可借鉴 | 不直接照搬 |
|---|---|---|
| [W3C PROV-O](https://www.w3.org/TR/prov-o/) | Entity/Activity/Agent 和派生关系；统一 source/parser/claim provenance | 不必在热路径完整实现本体；内部轻量 schema 保留映射即可 |
| [W3C SHACL](https://www.w3.org/TR/shacl/) | 把结构约束与数据分离，输出机器可读验证报告 | 结构合法不证明现实根因正确，仍需实验和人工判断 |
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | 分开评估 citation correctness、completeness 和回答质量 | 通用 QA 引用指标不足以覆盖诊断因果、风险和分支 |
| [RAGAS](https://aclanthology.org/2024.eacl-demo.16/) | 分开定位检索相关性、faithfulness 和生成质量 | LLM-as-judge 只作辅助，关键安全门保持确定性 |

### 11.5 Tool orchestration 与模型外策略

| 工作/规范 | 可借鉴 | 不直接照搬 |
|---|---|---|
| [Model Context Protocol Tools](https://modelcontextprotocol.io/specification/draft/server/tools) | 类型化输入输出、工具发现和结构化结果 | 协议不是授权系统；服务端仍需校验路径、预算和副作用 |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) 的 [guardrails](https://openai.github.io/openai-agents-python/guardrails/) 与 [tracing](https://openai.github.io/openai-agents-python/tracing/) | 标准 agent loop、工具 guardrail、trace/span | 框架 guardrail 不能替代 Evidence Pack、策略和本地 verifier |
| [Open Policy Agent](https://www.openpolicyagent.org/docs) 与 [Cedar](https://docs.cedarpolicy.com/auth/authorization.html) | policy decision 与 enforcement 分离、默认拒绝、禁止优先 | 第一阶段可先在 Python 中实现 policy-as-data，不必立刻引入新运行时依赖 |

### 11.6 Agent 安全与系统评测

| 工作 | 可借鉴 | 不直接照搬 |
|---|---|---|
| [ToolEmu](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html) | 模拟危险工具长尾失败，构造路径穿越、误删和附件执行红队集 | 模拟结果不能替代服务端只读和批准控制 |
| [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html) | 把日志、Jira、文档内指令视为不可信数据，测 prompt injection | 通用办公攻击集不覆盖 AOI 设备风险，需要项目内安全集 |
| [SafeToolBench](https://aclanthology.org/2025.findings-emnlp.958/) | 工具调用前联合评估用户意图和工具风险；区分展示、建议和执行 | 领域风险仍需覆盖磁盘、驱动、固件、网络和设备动作 |
| [AgentBench](https://openreview.net/pdf/6eee0bd1fd98c135372baedb2a5644233a013bb2.pdf) | 评估工具选择、步数、恢复、停止、预算和轨迹，而非只看最终文本 | 通用成功率不能替代事实覆盖、反证、Branch、安全和 `verified_fix` 指标 |

## 12. 评测与发布门槛

v3 不应只比较“回答好不好看”。至少需要六层评测：

### 12.1 Task/Scope

- 复合 Query facet recall；
- 实体、时间、资源和风险识别；
- required/optional/excluded 分类；
- Query 中导航入口但正文缺失时的 evidence gap。

### 12.2 Retrieval/Evidence

- evidence recall 与 source closure；
- raw/KG/SAG/Incident 跨源去重；
- citation correctness / completeness；
- parser exclusion 是否完整；
- 时间窗口命中率和窗口外噪声率。

### 12.3 Diagnosis

- Variant/root-cause false lock rate；
- detection point 误写根因率；
- 反证遗漏率；
- 假设置信度校准；
- next-best test 对候选的区分能力；
- 复发、复现、恢复和 verified fix 混淆率。

### 12.4 Answer/Procedure

- facet coverage；
- 并列方案覆盖；
- 条件、步骤、结果和退出条件完整性；
- 图片/附件引用闭包；
- “证据缺失而省略”是否明确。

### 12.5 Safety/Policy

- 破坏性动作越权率；
- 非破坏性命令被误删率；
- prompt injection / untrusted content；
- `verified_fix` false positive；
- 模型失败时执行许可是否仍 fail-closed。

### 12.6 Efficiency/Operations

- P50/P95 latency；
- token、工具轮次、I/O 和解析成本；
- cache hit、continuation 和重复解析率；
- no-LLM degradation；
- shadow 与 active 的逐 Query diff 及原因；
- schema/revision 不一致的拒绝和恢复。

发布顺序应覆盖：普通文档 QA、长 SOP、已锁定 KG Variant、多轮分支、Jira/诊断包、混合文档+现场证据，以及无模型/超时/解析器失败场景。

## 13. 实施阶段与当前进度

### Phase 0：冻结基线（已完成）

- 既有冻结测试结果继续引用 `读侧冻结与KG_v2+raw Codex新管线.md`，本轮未重跑旧基线套件；
- `config/read_side_frozen_manifest_v2.json` 内容锁定 16 个文件和 4 个 Python 目录树；
- `scripts/verify_frozen_read_pipeline.py --manifest config/read_side_frozen_manifest_v2.json`
  当前结果为 20/20、`frozen=true`、无 drift；
- v1 清单仍保留 2026-07-30 历史快照语义，不被 v2 静默重定义。

### Phase 1：统一 contract（已完成 shadow 版）

- 定义 `ReadRequest v3`、`EvidenceRecord`、`EvidenceLink`、`HypothesisRecord`、`AnswerPlan`、`ReadResponse v3`；
- 为现有 Evidence Pack 编写只读 adapter；
- 不改变当前答案。

### Phase 2：Evidence Fabric（已完成内存版）

- 内容寻址 Evidence ID；
- raw/KG/SAG/Incident source adapter；
- claim-level provenance、exclusion 和 derivation；
- 待完成：持久化、大文件引用化和跨请求缓存。

### Phase 3：Provider Tool Registry（已完成只读版）

- provider namespace；
- 统一 schema/version、warning、truncation、continuation、cost；
- 工具 trace 和只读/副作用分类。

### Phase 4：统一 Planner（bootstrap 与 Codex 契约已完成）

- facet/gap 驱动的多轮调查；
- 已完成：全局工具轮次、调用次数和 Evidence 数量预算；
- 待完成：按复杂度动态分配预算，以及可计算的信息增益/停止准则；
- 先 shadow，不接管状态。

### Phase 5：Hypothesis 与 Policy（无状态影子版已完成）

- 已完成：每次请求内的候选/需证据/否定/支持状态、支持与反证 ID；
- 已完成：Python Policy 对 shadow 状态、破坏性风险和错误 `verified_fix`
  升级做模型外限制；
- 待完成：跨会话假设持久化，以及脱离冻结基线状态后的 Variant/Branch 接管裁决；
- 待决策：是否将 Python 策略迁移到 OPA/Cedar 类 policy-as-data 引擎。

### Phase 6：Answer Plan 与 verifier（已完成 shadow 版）

- 已完成：统一章节、claim、Hypothesis、Trace、风险和缺口；
- Codex/确定性双 renderer；
- claim/citation/coverage/safety verifier。
- 待完成：将并列方案和媒体从通用 section 字段升级为独立结构契约。

### Phase 7：shadow 发布与清理（影子评测已完成，尚未切流）

- 分场景双跑；
- 差异归因；
- 达标后逐步接管；
- 最后删除 DeepSeek 重复 Harness、独立 answer override 和被替代的兼容规则。

### 13.1 2026-08-04 影子验证结果

本轮只跑 v3 shadow 正式验证集，没有重跑冻结基线测试。产物位于：

`data/results/read_runtime_v3/formal_validation/iteration_007_full60_contract_trace/`

| 指标 | 结果 | 解读 |
|---|---:|---|
| 完成 | 60/60 | 全部请求产出 v3 结构产物 |
| contract failure | 0 | 无 schema/结构失败 |
| verifier pass | 100% | 当前 bootstrap Answer Plan 均通过本地门禁 |
| 正式 answer/status 与冻结基线一致 | 100% | shadow 未改写生产输出 |
| proposed answer 变化 | 25% | v3 已对部分 Query 形成不同的受引用组织 |
| unsafe action / false resolved | 0 | 未观测到越权动作或错误解决声明 |
| 冻结校验 | 前 20/20，后 20/20 | 评测前后都无 drift |

这些数字证明的是**契约、证据接入、影子隔离和安全回退已跑通**，不是
“v3 诊断语义已全面超过基线”。默认 bootstrap 不执行 Codex 的语义选择，所以：

- 已召回的 Family/Variant 仅作为候选证据，不自动升级为已锁定根因；
- source-only 记录已完成请求上下文入 Fabric 和正确路由，但 bootstrap 不会生成
  多 Trace 语义编织；该能力需由 `codex_agentic` 产生 `TraceCandidate`后再经 verifier；
- Benchmark 审计发现 1 条 CPU 文档真值 ID 过期：源文件 SHA-256 已变更，运行时
  当前 ID 与旧标注不一致。该项被标记为 `stale_source_ids`，不为提高分数硬编码旧 ID。

单元测覆盖 Evidence Fabric、Task Normalizer、runtime/fallback、Agent strict contract 和
formal evaluation 投影，当前为 **27 passed**。

## 14. 仍需回答的开放问题

1. Evidence Fabric 首版只做内存+结果文件，还是直接提供持久化事件存储？
2. 稳定 Evidence ID 如何同时编码内容 hash、source revision 和 range，而不让 ID 过长？
3. claim-level provenance 应在 Answer Plan 生成前还是 renderer 后二次抽取？
4. contradiction 是结构化枚举、规则推断，还是允许 Agent 提议后由本地验证？
5. planner 的 token、工具、I/O 和 wall-clock 预算如何按任务类型动态分配？
6. 多轮会话中，新文件替换旧路径时如何检测内容变化并使旧假设失效？
7. policy 首版继续用 Python，还是引入 OPA/Cedar 风格的独立策略层？
8. DMP 符号化、远端 Jira 和隔离复现环境如何做 capability-scoped 授权？
9. 历史案例在成为 KG 正式知识前，如何表达新鲜度、适用版本和置信度衰减？
10. 如何用 held-out 跨领域事故验证当前 GPU/CUDA 经验没有演变为新一轮硬编码规则？

## 15. 结论

Read Runtime v3 的必要性来自一个事实：读侧已经不只是“从知识库找一段答案”，而是在同时处理正式知识、原始资料、现场事件、调用栈、环境、风险和多轮验证。

合理的长期形态不是让某一层吞掉其他层，而是：

- 用统一 contract 描述任务；
- 用 Evidence Fabric 连接异构证据；
- 用 Codex 进行自适应调查和回答组织；
- 用 KG_v2 保留诊断结构；
- 用本地 Policy Adjudicator 和 verifier 保证状态、安全和来源；
- 用 shadow 双跑渐进迁移。

它最重要的取舍是：接受更多契约、provenance 和策略复杂度，换取跨证据源的一致性、可审计性、安全性和长期可扩展性。对于 AOI 这类既要回答知识问题、又要推进现场诊断、还可能触及生产设备的系统，这个取舍是值得的。

## 16. 与现有文档的关系

- [读侧管线组织方式总览](读侧-管线组织方式总览.md)：说明**当前**几条读侧路径如何运行；
- [读侧 KG_v2 原生运行时](读侧-KG_v2原生运行时.md)：说明冻结基线的状态和安全不变量；
- [KG_v2 主动证据补全与 Codex Tool Harness](KG_v2读侧主动证据补全与Codex%20Tool%20Harness.md)：说明当前 Evidence Pack 和模型工具边界；
- [读侧 Jira 诊断数据包与 Incident Evidence Runtime](读侧-Jira诊断数据包与Incident-Evidence-Runtime.md)：说明当前案件证据扩展的实现；
- [读侧冻结与 KG_v2+raw Codex 新管线](读侧冻结与KG_v2+raw%20Codex新管线.md)：说明独立 Codex 旁路；
- 本文：解释这些能力为什么需要在 Read Runtime v3 中统一，以及统一时应保留哪些边界。
