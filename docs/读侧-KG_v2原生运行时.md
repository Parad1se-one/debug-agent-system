# 读侧 KG_v2 原生运行时

> 状态：冻结基线运行契约。当前读侧文档导航见 [读侧文档索引](读侧文档索引.md)。

## 不变量

生产读运行时遵守以下强制不变量：

1. 会话主故障 ID 是 `FaultVariant.variant_id`，同时保留 `family_id`。
2. 主候选仅包含 KG_v2 `FaultFamily` 和 `FaultVariant`。
3. 主计划仅由 `DiagnosticTrace`、`DecisionPolicy`、`TraceStep`、`DiagnosticAction` 和 `BranchRule` 编译。
4. 顶层 `AgentResponse.evidence_ids` 仅引用当前 KG revision 的
   `EvidenceItem.evidence_id`；调用方提交的日志/文档等只读观察使用独立
   `tool-evidence:*` ID，并在 `answer_sections`/`sources` 中绑定原资源，不伪装成
   canonical KG 证据。
5. legacy `Error/Check/Solution` 不得参与运行时检索、排序、推进或解决判定。

旧 API 字段 `top_error_id/current_check_id/which_check_solved` 暂时保留，但值分别是
`variant_id/action_id/resolved_action_id`，仅作为序列化别名。

## SAG_v2

SAG_v2 是从 KG_v2 构建的 SQLite serving index，不是第二事实源。索引记录
`graph_revision`，运行时发现 revision 不一致时从 KG_v2 重建。SAG 命中任意对象后，
只能通过对象自带的 `variant_id/family_id` 或原生 `has_variant` 关系召回 Variant，
并在响应中保留 seed 到 Variant 的检索路径。

若使用 `kg_v2_json` 配置，运行时直接扫描同一 KG_v2 图。任何模式都不会回退
`data/kg` 或 `materialized_execution`。

## 计划和闭环

锁定 Variant 后优先选择质量最高的 `DiagnosticTrace`，按
`actual_action_ids/recommended_action_ids` 和 `TraceStep.ordinal` 编译计划。用户反馈被
归一为 Outcome 类型，再匹配当前 TraceStep 的 BranchRule。无匹配分支时才按计划顺序
推进。

运行时只接纳有原生 Trace、可执行 Action 和 EvidenceItem 支持的 Variant。标记为
`execution_materialize_allowed=false` 的支持性文档不会被提升为执行计划。

`resolved` 是 fail-closed 状态：必须同时存在当前 Action 的 `verified_fix` Outcome、
有效 EvidenceItem，以及用户的解决反馈。`pending_validation`、临时缓解和无证据的口头
解决反馈都不能闭环。高成本或破坏性 Action 在人工明确确认前保持 `ask_info`。

## 证据回答、充分性与安全门

回答行为不依赖检索路由。无论 Query 是直接命中文档、由 Variant 关系带回文档，还是从
EvidenceItem 到达文档，只要召回了已批准、带 hash 和来源锚点的原文 Chunk，都按源文件
顺序进入同一证据组织流程。长文档的 Section/情况/方法必须保持完整；图级聚合摘要若已被
至少三个原文 Chunk 覆盖，则以
`aggregate_summary_superseded_by_source_chunks` 记录排除，避免重复正文和聚合摘要绕过
安全处理。

充分性继续拆成三个轴：

- `answerable`：有可追溯资料即可为真，不要求锁定 Variant；
- `diagnosable`：必须实际处于 `kg_v2_locked` 且已编译出计划；
- `executable`：除可诊断外，还必须满足首步分支上下文和安全确认。

首轮 RequiredInfo 门不能简单按 priority 或 blocked-action 数量触发。只有
RequiredInfo 的 `why_required` 明确表示它用于选择“分支/路径”，且 question 要求现场
表现、状态、接口或拓扑等区分性观察时，缺失信息才阻断首步执行。日志包、配置文件、版本
等后续诊断增强材料不会无条件阻断安全的第一步。若 Query 已包含对应观察，例如“风扇持续
转动但屏幕无显示”或“皮带不转”，则直接使用该观察选路。

安全处理同样覆盖原文 Chunk 和 EvidenceItem 聚合摘要，并在渲染 content blocks 前完成。
拆机/部件拆装、BIOS/CMOS、磁盘/引导、市电或带电测量分别转换为保留排查目的和前置条件
的安全摘要；原始命令、跳线、短接和拆装步骤不进入正文。判定采用“动作动词 × 内部硬件
对象”的通用规则，不依赖某一篇文档或某一个 Query 的固定句子。

## Query 意图、实体作用域与最终编织

读侧在 Variant 排序前先生成 `metadata.query_scope`。`knowledge_lookup` 覆盖明确的
使用方法、流程、规格、配置、授权、采集和文档导航问题；这类 Query 的目标是找到原文
证据，不得因为“系统、内存、BIOS、授权”等通用词与某个故障相似就锁定相邻 Variant。
`fault_diagnosis` 才允许在满足原有置信度、margin、Trace 和 Evidence 门后锁定 Variant。

Query 中的错误码、工具名、产品型号和厂商名属于强实体约束。存在强实体时，主证据必须
覆盖该实体；若索引没有对应材料，返回 `knowledge_scope_not_covered`，不得用别的主板、
软件或故障案例填充答案。软狗/硬狗、可以进入系统/无法进入系统等互斥适用条件同样在
文档与 Chunk 层隔离。

直接命中文档后，以“源文档 + Query 命中的导航子文档”为主证据域。部分目录中只要已有
可解析子文档，就按 Query 选择相关分支，同时在 trace 中保留未解析链接和排除原因；父
目录自身的原文表格、负责人等有效事实不能因展开子文档而丢失。导航路径保留具体源文件
名，回答标题使用可读的文档标题。

`EvidenceAnswerComposer` 是最终回答所有者。QA Supervisor 必须保留
`answer_sections/items/evidence_ids/chunk_ids`，不得再次让在线 Composer 改写已绑定
来源的正文，也不得在结构化回答后追加通用“请补充版本、日志、图片”。知识型问题在证据
充分时返回 `step` 且 `required_data=[]`；诊断型问题才追问真正影响分支选择或安全执行的
现场信息。高风险提示按类别在整篇回答中完整说明一次，后续同类段落只引用该安全前置
条件，避免重复警告淹没排查主线。

## 主动证据补全

`start` 和 `step` 都接受可选 `evidence_resources`。运行时先完成原有 KG_v2 检索和
充分性判断；若状态为 `ask_info`，`EvidenceGapResolver` 才会选择与
`required_data` 相关的资源，调用有界只读 parser，并把来源绑定 observation 作为新的
检索/充分性上下文再次运行。该层位于安全确认、BranchRule 和 verified-fix 门外，不能
替代任何确定性裁决。

统一结果写入 `metadata.evidence_gap_resolution`，包括 resolved/unresolved items、
observations、tool results、排除原因、停止原因和调用轮次。图片只提供 header metadata，
不做 OCR，不能满足截图文字类缺口。

## 可选 Incident Evidence Runtime

Jira 问题描述、粘贴的长日志和诊断数据包不再只作为普通 Query 文本处理。启用
`incident_runtime.enabled` 后，系统建立独立 `IncidentCase`，先解析 Query 参考时间和资源年份，
安全枚举附件，只对相关日志窗口抽取事件、调用栈与环境，构造时间线、相关性和案件图，再用
稳定错误码/组件/函数查询只读 KG_v2。KG 宽召回
候选会完整保留供审计，但只有与案件 evidence ID 实际相交的候选才能成为正式假设。

结果写入 `metadata.incident_runtime`，包括 `incident_evidence_pack.v3`、假设矩阵、下一步
测试、排除原因和 verifier 结果。默认 `enabled=false, shadow_mode=true`；关闭开关即可回到
本节其余冻结行为。完整契约见
[读侧 Jira 诊断数据包与 Incident Evidence Runtime](读侧-Jira诊断数据包与Incident-Evidence-Runtime.md)。

## 可选 Codex Tool Harness

`src/debug_agent_system/adapters/codex_read` 已实现 OpenAI-compatible Codex 控制器。
Codex 可选择 `diagnose_start`、`diagnose_step`、`retrieve_evidence`、
`expand_document_context`、`inspect_kg_path`、`inspect_source_assets`、
`parse_evidence` 和 `render_evidence_answer` 等 KG 原生 strict Tool。最后一个 Tool 只提交
Evidence Pack 条目 ID 和章节顺序，由本地 verifier 校验并渲染；模型不能直接写入事实。
本地 runtime 始终独占 Variant 锁定、计划编译、分支、安全门和解决判定。

启用案件证据层后，同一 Harness 还可使用 `analyze_incident`、`index_log_package`、
`parse_incident_scope`、`get_incident_scope`、`get_incident_evidence_pack`、按参考时间的事件/栈/
日志窗口查询、EVTX/DMP 检查、案件 KG 查询、相似案例、假设矩阵、下一步测试、
`plan_reproduction`、`compare_reproduction_runs` 和 `render_incident_report`。参考时间在附件解析前
规范化，压缩包窗口外成员仅保留清单；“跨日期复发”“受控复现”“修复验证”分别建模，不因一次
未出现目标签名就宣布修复。工具共享同一个 case ID，且仍由本地
executor/parser 执行；Codex 不能执行附件、联网修改 Jira 或写 canonical KG。

默认 `read_llm.enabled=false`、`answer_composer_enabled=false`。Codex 缺少配置、超时、
返回非法 JSON 或校验失败时，系统 fail-open 到确定性 runtime。完整输入输出契约、CLI
和测试说明见
[KG_v2读侧主动证据补全与Codex Tool Harness](KG_v2读侧主动证据补全与Codex%20Tool%20Harness.md)。
