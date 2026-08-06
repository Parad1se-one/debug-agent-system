# Read Runtime v4：证据调查管线

> 状态：已实现独立旁路，默认 `shadow_mode=true`；冻结基线和 Read Runtime v3 不变。
> 更新时间：2026-08-04。

## 1. 定位

Read Runtime v4 是在冻结读侧和 v3 Evidence Fabric 之上的新增调查管线。它的中心对象不再是“冻结答案”，而是 `InvestigationState`：事实、假设、证据缺口、矛盾、下一步验证和排除理由。

```text
Query / 会话 / 诊断包
        ↓
任务编译（Query 类型、时间、资源、风险、输出契约）
        ↓
Incident / Request Context / KG-SAG / raw / baseline Provider
        ↓
Evidence Fabric（统一证据 ID、来源锚点和关系）
        ↓
InvestigationState（facts / hypotheses / gaps / next_tests）
        ↓
Answer Compiler（案件优先或流程优先）
        ↓
Policy + Verifier
        ↓
ReadResponse v4
```

## 2. 与 v3 的区别

v3 默认是 `evidence_first_bootstrap`：先取得冻结回答，再把其它 Provider 追加到 shadow answer。v4 先建立案件证据状态；有诊断包时，事故证据优先进入回答，冻结答案只作为参考证据，不再把无关的相机、网卡或其它文档步骤拼进事故结论。

v4 当前的默认 Planner 是确定性的 `deterministic_investigation`，这样可以在 Codex 不可用时稳定运行。后续 Codex 只读 Planner 只需输出同一份 `InvestigationState`/`V4AnswerPlan`，不改变本地策略、答案编译和校验。

## 3. Query 类型与输出契约

任务编译器复用通用 Query scope 和时间 scope，但只根据结构判断，不绑定具体 Query：

- 有诊断资源、日志摘要或参考时间：`incident_report`；
- 条件式“如何/怎么”问题：`procedure_answer`；
- 其它知识问答：`evidence_answer`。

事故回答按“案件摘要、时间对齐、直接观察、综合判断、建议立即采取、下一步验证、候选修复动作、修复后验证、证据缺口、来源”组织。流程回答按“结论、前置条件、方案与步骤、风险、成功标志、来源”组织。

### 3.1 行动计划不新增顶层对象

v4 复用 `InvestigationState.next_tests` 承载行动计划，不修改冻结 v3 的
`DiagnosticTest` 结构，也不新增 `ActionCandidate` 顶层对象。每条记录在 v4
规范化时补充以下字段：

- `kind`：`containment`、`diagnosis`、`remediation` 或 `verification`；
- `priority` / `status`：动作排序和 `recommended`、`conditional`、`blocked`、`omitted`；
- `preconditions` / `rollback`：执行前提和回滚边界；
- `expected_observations` / `distinguishes_hypothesis_ids`：预期观察及区分的假设；
- `risk` / `cost` / `requires_confirmation`：风险、成本和人工确认要求；
- `evidence_ids` / `source_ids`：案件证据与知识来源的追溯锚点。

因此“补证据”和“先采取什么行动”使用同一份可校验记录，但在答案中按
`kind` 分组。证据不足时，仍可输出低风险的证据保全或现场隔离动作；这类
动作不等于根因确认，也不授予设备执行权限。破坏性动作始终标记为
`blocked`，等待显式授权。

## 4. 证据和状态边界

- 日志、EVTX、DMP 和调用栈首先是 `observed` 事实，不自动等于根因；
- KG/SAG 候选只提供机制、分支和验证路径，不自动锁定现场根因；
- `locked_root_cause` 和 `verified_fix` 不由 v4 自动升级；
- `executable` 默认关闭；只读分析不执行附件脚本或设备动作；
- 破坏性章节必须进入 `risk_controlled`，并等待显式授权；
- 每个事实、假设和答案章节都必须能回到 Evidence Fabric 中的证据 ID。

## 5. 运行方式

默认 shadow：

```bash
PYTHONPATH=src python scripts/run_read_runtime_v4.py \
  "设备在 2026-08-01 21:30 闪退" \
  --evidence /path/to/diagnostic.zip \
  --output data/results/read_runtime_v4/result.json
```

默认正式 `answer/status` 仍来自冻结基线；v4 编译结果在 `shadow.proposed_answer`。经过验证后才可以显式使用：

```bash
PYTHONPATH=src python scripts/run_read_runtime_v4.py \
  "设备在 2026-08-01 21:30 闪退" \
  --evidence /path/to/diagnostic.zip \
  --active
```

`--active` 只让通过本地 verifier 的 v4 答案生效，不授予设备执行权限。

## 6. 当前实现和后续工作

已实现：v4 契约、任务编译、案件优先 Provider 顺序、Evidence State、确定性答案编译、行动计划字段规范化与优先级排序、策略门禁、来源校验、shadow diff、CLI 和单元测试。

尚未接入：Codex agentic Planner、DMP 符号化调试器、持久化日志时间索引、跨轮次会话状态和自动化复现执行器。它们应作为只读、可审计、可回滚的 Provider/Planner 增量接入，而不是修改 v4 的本地安全边界。
