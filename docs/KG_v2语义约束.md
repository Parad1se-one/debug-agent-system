# KG v2 语义约束

> 本文定义 `data/kg_v2` 的语义边界。它不是代码实现说明，而是 **W2 抽取、W3 归一化、W4 门控、W6 人工审核** 的共同判尺。
>
> 目标不是“先把图搭起来”，而是先保证：**写进去的对象粒度对、语义对、可审核、可 materialize 到读侧执行视图。**

---

## 1. 总原则

### 1.1 v2 的主语义不再是旧 `Error / DiagnosticCheck / Solution`

KG v2 的主对象是：

- `FaultFamily`
- `FaultVariant`
- `DiagnosticAction`
- `ActionOutcome`
- `RequiredInfoSpec`
- `DiagnosticTrace`
- `DecisionPolicy`
- `EvidenceItem`
- `SourceCase`

旧读侧仍然消费 `Error -> has_check -> next -> resolved_by`，但这只是 **materialized execution view**，不是 v2 原图主存结构。

### 1.2 一切新增语义先写 v2 原图，再投影给读侧

写侧的首要目标不是“直接产出旧 KG 节点”，而是：

1. 先产出正确的 `family / variant / action / outcome / required-info / trace / evidence`
2. 再由 deterministic materializer 投影成旧读侧能消费的执行视图

### 1.3 原子化优先于覆盖率

v2 的首要目标是粒度正确，而不是一次写进尽可能多的信息。

如果一句话里同时包含：

- 故障现象
- 排查动作
- 动作结果
- 根因推断

则必须拆开，不能直接塞进一个对象。

### 1.4 证据层和诊断层必须分离

长文本、聊天原话、Jira 描述、日志片段、附件摘要，属于 `EvidenceItem / SourceCase`。

`FaultFamily / FaultVariant / DiagnosticAction / RequiredInfoSpec` 不允许承载长段原文。

### 1.5 materialized view 只允许从 approved v2 对象生成

读侧可执行视图只允许来自：

- approved `FaultFamily / FaultVariant`
- approved `DiagnosticAction`
- approved `ActionOutcome`
- approved `RequiredInfoSpec`
- deterministic recompute 的 `DecisionPolicy`

W2/LLM 不得直接写 `resolved_by` 事实。

---

## 2. 对象定义与边界

## 2.1 FaultFamily

### 定义

`FaultFamily` 表示 **一类可稳定复现的故障家族**，是 v2 的主入口对象。

它回答的问题是：

- “这到底是哪一类故障？”
- “这类故障属于哪个大类/子系统/场景？”

### 必须包含

- 家族级短名称
- 家族级核心症状摘要
- 分类标签：`category / subsystem / scenario`

### 不允许包含

- 某个具体站点
- 某个具体版本
- 某次具体聊天结论
- 排查顺序
- 某次动作是否成功
- “更换内存条后恢复”这种 episode 级结果

### 命名规则

- `label`：短名称，优先 6 到 20 个汉字
- `summary`：家族级核心故障描述，80 字内
- 不出现“现场反馈”“客户说”“某某项目”“某版本”“某设备编号”之类 episode 上下文
- 默认优先写 **故障类/症状类名称**，而不是具体条件或具体根因
- 只有当“根因本身已经稳定到足以成为一类独立故障家族”时，family 名称里才允许出现根因词
- 如果两个现象对应的 required-info、首轮诊断动作、后续分支明显不同，则必须拆成两个 family，而不是沿用旧专题并写成一个大类

### 正例

- `工控机蓝屏`
- `工控机异常重启`
- `相机拍摄失败`
- `CAD 导入失败`

### 反例

- `客户07现场设备操作很卡然后蓝屏，内存128G，3200频率，邢工建议先看DMP`
- `相机网卡过滤驱动配置错误导致拍摄失败`
- `客户12 machine 8.0.2 回退 7.2.3 后 2 小时再次卡顿`

---

## 2.2 FaultVariant

### 定义

`FaultVariant` 表示 **FaultFamily 下某个具体现场变体**。

它回答的问题是：

- “这次故障属于哪个家族下的哪个条件化分支？”

### 必须包含

- `family_id`
- 变体短名称
- 变体摘要

### 可以包含

- `equipment_type`
- `site`
- `software_version`
- `error_phase`
- `owner_context`
- `escalation_target`
- 变体级关键词

### 不允许包含

- 整段排查过程
- 多个动作结果混写
- 大段证据原文

### 命名规则

- `label`：要体现“区别于家族的关键条件”
- `summary`：只保留变体专有条件，不复述完整排查过程
- variant 允许带条件、环境、版本、特定错误签名、局部根因线索
- variant 不要求一定已经证明最终根因，但必须比 family 更具体

### 正例

- `编程拍照速度延迟现象`
- `MEMORY_MANAGEMENT/PFN 不同步蓝屏`
- `相机网卡过滤驱动取消勾选导致拍摄失败`
- `导入后尺寸过大且不显示的 CAD 导入失败`

### 反例

- `客户反馈很卡，后来蓝屏，更换内存条后正常，但还不确定`
- `相机不拍照，勾选恢复，之后又蓝屏，怀疑核显驱动`

> 上面第二个反例本质上是 **两个 variant / 两个 family 混在一起**，必须 split-case。

---

## 2.3 DiagnosticAction

### 定义

`DiagnosticAction` 表示一个 **原子诊断动作**。

它回答的问题是：

- “工程师实际建议或执行的单步动作是什么？”

### 核心约束

一个 `DiagnosticAction` 只能表达 **一件事**。

### 合法 `action_role`

- `inspect`
- `collect`
- `compare`
- `change`
- `verify`
- `observe`
- `escalate`

### 原子化规则

一个 action 里不允许同时出现：

- “检查 A 并且更换 B”
- “收集日志然后回退版本再观察”
- “分析 DMP、卸载驱动、更新显卡驱动”

这种多动作描述必须拆成多个 action。

### 命名规则

- `label`：动作短名称，只保留一个动作
- `summary`：描述怎么做，180 字内
- `label` 优先使用动宾结构

### 推荐模板

- `检查采集卡`
- `分析 DMP 是否指向 igdkmdn64.sys`
- `卸载无线网卡驱动`
- `回退 machine 版本验证`
- `观察 48 小时是否复发`

### 反例

- `检查采集卡并更换工控机验证`
- `收集 DMP 后更新显卡驱动并通知客户每天重启`

---

## 2.4 ActionOutcome

### 定义

`ActionOutcome` 表示 **某个 action 在某个 family/variant/condition 下的一次结果**。

它回答的问题是：

- “这个动作在这个场景下到底有没有效？”

### 核心约束

一个 `ActionOutcome` 只能表达：

- 一个 action
- 一个结果类型
- 一组证据

不能把多次试验合并成一句模糊自然语言。

### outcome_type 判定表

#### `verified_fix`

定义：

- 有明确证据表明问题已解决
- 且不是短时观察后立即复发

常见信号：

- `更换内存条后至今未再出现蓝屏`
- `恢复勾选后拍照立即恢复正常，且该问题的直接原因已明确`

禁止误判：

- 只有“今天没再复现”
- 只有“暂时恢复”

#### `ineffective`

定义：

- 动作已执行，明确无效

例子：

- `更换采集卡无效`
- `更换 CXP 线失败`

#### `partial_temporary`

定义：

- 短时恢复，但随后复发

例子：

- `版本回退后正常 2h，随后再次延迟卡顿`

#### `mitigation_observed`

定义：

- 观察到缓解，但尚不能证明根因闭环

例子：

- `卸载无线网卡驱动后暂未再蓝屏，仍待观察`
- `每日断电重启后暂未复发`

#### `recurred`

定义：

- 之前看似恢复，后来明确再次复发

#### `pending_validation`

定义：

- 动作方向被提出，但尚未完成验证
- 尤其用于高成本、返厂、重标、停线动作

例子：

- `更换相机需返厂且重标 3D，待验证`

#### `diagnostic_method`

定义：

- 这是诊断手段，不是修复动作

例子：

- `开启 Driver Verifier`
- `运行 WPR / PoolMon`
- `导出 DMP`

#### `context_not_root_cause`

定义：

- 提供了背景上下文，但并不是根因动作结果

### 关键硬约束

- 只有 `verified_fix` 才有资格 materialize 成 `resolved_by`
- `mitigation_observed / partial_temporary / pending_validation / diagnostic_method` 一律不能 materialize 成 `resolved_by`

---

## 2.5 RequiredInfoSpec

### 定义

`RequiredInfoSpec` 表示 **读侧真正值得问的一条结构化追问信息**。

它回答的问题是：

- “为了缩小诊断空间，我们具体还缺什么？”

### 核心约束

一个 `RequiredInfoSpec` 只允许对应：

- 一个 `slot`
- 一个 `question`
- 一个 `why_required`
- 一个 `blocks`

### 不允许

- `请发日志/给资料` 这种泛化请求直接入图
- 一个 spec 同时要日志、版本、截图、现场信息

### 命名与内容规则

- `slot`：必须来自受控词表
- `question`：直接给读侧/用户的话术
- `why_required`：必须说明为什么这个信息能缩小路径
- `blocks`：缺它会卡住哪类 action / 分支

### 正例

- `slot=log_package`
  - `question=请提供该故障对应的 DLOG 或诊断数据包。`
  - `why_required=需要确认故障发生阶段和底层模块。`

- `slot=software_version`
  - `question=请提供主程序和算法包版本。`
  - `why_required=需要判断是否命中已知版本缺陷。`

### 反例

- `请把所有资料都发一下`
- `麻烦给下日志和截图和版本还有现场说明`

---

## 2.6 DiagnosticTrace

### 定义

`DiagnosticTrace` 记录 **一个案例中的动作顺序**。

它回答的问题是：

- “工程师建议怎么查？”
- “现场实际上怎么做了？”

### 核心约束

- `recommended_action_ids`：建议顺序
- `actual_action_ids`：实际执行顺序
- 必须关联 `SourceCase` 和 `EvidenceItem`

### 不允许

- 把 trace 当成 policy
- 把单例 trace 当成通用最优路径

---

## 2.7 DecisionPolicy

### 定义

`DecisionPolicy` 是从 approved trace/outcome 聚合出来的 **读侧策略层**。

### 核心约束

- 只能 deterministic recompute
- 不能由 W2/LLM 直接生成

### 负责表达

- 哪些 action 排序更靠前
- 哪些 action 经常无效
- 哪些 action 高成本/高风险

### 不负责表达

- 新的事实
- 新的根因

---

## 2.8 EvidenceItem 与 SourceCase

### EvidenceItem

`EvidenceItem` 承载原始或解析后的证据片段：

- 消息
- Jira 描述
- 评论
- 附件摘要
- DMP header
- log hint
- proj 摘要

### SourceCase

`SourceCase` 是案例容器，连接：

- Evidence
- Variant
- Trace
- Outcome
- RequiredInfo

### 关键约束

- 长文本优先放 `EvidenceItem.summary`
- `SourceCase.summary` 只做压缩摘要
- 不允许把这些长文本塞回 `FaultFamily / FaultVariant / DiagnosticAction`
- 来源语义必须显式区分：`sop` 仅用于 SOP；群聊/Jira 历史案例用 `case/chat_case/jira_case`；非 SOP 原始知识文档用 `raw_doc`。
- W9/W10 处理非 SOP 文档时，不得沿用 builder 的 `source_kind=sop` 默认值；进入 W4 前必须重标为 `raw_doc`，原文证据使用 `EvidenceItem.source_kind=tool_parse`。
- `raw_doc` 产生的 `SourceCase.approved` 初始必须为 `false`，只有 W6 人工批准并由 W5 合并时才能改为 `true`。

---

## 3. 命名规范

## 3.1 FaultFamily 命名模板

模板：

- `核心故障类`
- `核心故障现象`

默认不推荐：

- `具体条件 + 故障`
- `具体根因 + 故障`

除非该根因本身已经是可独立复用的稳定专题

例子：

- `工控机蓝屏/异常重启`
- `相机拍摄失败`
- `CAD 导入失败`

## 3.2 FaultVariant 命名模板

模板：

- `条件 + 故障现象`
- `动作/环境约束 + 故障现象`
- `错误签名 + 故障现象`
- `局部根因线索 + 故障现象`

例子：

- `编程拍照速度延迟现象`
- `MEMORY_MANAGEMENT/PFN 不同步蓝屏`
- `相机网卡过滤驱动取消勾选导致拍摄失败`
- `更换内存后仍复发的 MEMORY_MANAGEMENT 蓝屏`

## 3.3 DiagnosticAction 命名模板

模板：

- `动词 + 对象`

例子：

- `检查采集卡`
- `分析 DMP`
- `卸载无线网卡驱动`
- `回退 machine 版本验证`
- `观察 48 小时是否复发`

## 3.4 ActionOutcome 摘要模板

模板：

- `动作 + 结果`

例子：

- `更换采集卡无效`
- `回退版本后短时正常 2h 后复发`
- `更换内存条后未再出现蓝屏`
- `更换相机需返厂重标，待验证`

---

## 4. split-case 规则

一个 episode 必须拆成多个 case 的信号：

- query 和结论属于不同 family
- 同一对话里出现两个独立故障链
- 某个动作结果明显属于另一个问题

### 正例

`chat-rank:240b3ff8f1e9`

必须拆成：

- `相机网卡过滤驱动取消勾选导致拍摄失败`
- `Intel 核显驱动 igdkmdn64.sys 导致蓝屏`

### 规则

- split-case 后，每个 case 各自有 family / variant / actions / outcomes / required-info / trace
- 不允许把一个 case 的 outcome 挂到另一个 case 上

---

## 5. materialize 约束

v2 原图和读侧执行视图之间必须满足以下约束：

1. `FaultFamily / FaultVariant` 才能投影为读侧 `Error`
2. `DiagnosticAction(action_role in inspect/collect/compare/change/verify)` 才能投影为 `DiagnosticCheck`
3. 只有 `ActionOutcome(verified_fix)` 才能投影出 `resolved_by`
4. `RequiredInfoSpec` 必须投影成结构化 `required_info_schema`
5. `DecisionPolicy` 只能由 approved trace/outcome 重新计算

---

## 6. W2/W3/W4/W6 的门禁含义

## 6.1 W2 抽取阶段

必须做到：

- 抽出 v2 原生对象
- 不把长文本塞进 family/variant/action 主字段
- 能识别 split-case

## 6.2 W3 归一化阶段

必须做到：

- family/variant 归一化
- slot alias 归一化
- outcome 类型归一化
- 多动作拆分

## 6.3 W4 门控阶段

必须拦截：

- family/variant 混淆
- action 非原子
- outcome 类型不清
- required-info 泛化
- split-case 失败
- 非 `verified_fix` 却企图 materialize `resolved_by`

## 6.4 W6 审核阶段

review item 必须让 reviewer 看见：

- family
- variant
- actions
- outcomes
- required-info
- trace
- evidence pack
- materialized execution preview

---

## 7. 两个案例模板

## 7.1 case001：编程拍照速度延迟

推荐建模：

- `FaultFamily`：`相机拍摄失败`
- `FaultVariant`：`编程拍照速度延迟现象`
- `DiagnosticAction`
  - `检查光机控制板`
  - `检查采集卡`
  - `检查驱动`
  - `检查内存条`
  - `检查 CXP 线`
  - `回退 machine 版本验证`
  - `更换工控机验证`
- `ActionOutcome`
  - `更换采集卡无效` -> `ineffective`
  - `排查驱动无效` -> `ineffective`
  - `版本回退后 2h 再复发` -> `partial_temporary`
  - `更换工控机失败` -> `ineffective`
  - `更换 CXP 线失败` -> `ineffective`
  - `更换相机需返厂重标` -> `pending_validation`

禁止建模：

- 把 `更换采集卡无效` 当 `resolved_by`
- 把 `更换相机` 在未完成闭环前当 `verified_fix`

## 7.2 蓝屏 MEMORY_MANAGEMENT

推荐建模：

- `FaultFamily`：`工控机蓝屏`
- `FaultVariant`：`更换内存后仍复发的 MEMORY_MANAGEMENT 蓝屏`
- `DiagnosticAction`
  - `分析 DMP`
  - `测试内存和 CPU 稳定性`
  - `卸载无线网卡驱动`
  - `更新 NVIDIA 驱动`
  - `开启 Driver Verifier`
  - `观察是否复发`
- `ActionOutcome`
  - `卸载无线网卡驱动后暂未复发` -> `mitigation_observed`
  - `更新显卡驱动后暂未复发` -> `mitigation_observed`
  - `Driver Verifier` -> `diagnostic_method`

禁止建模：

- 把 `暂未复发` 直接当 `verified_fix`
- 把 `Driver Verifier` 当 solution

说明：

- `蓝屏` 和 `异常重启` 不是默认同一个 family。
- 只有在案例语义明确表达“重启只是蓝屏后的伴随现象”时，`重启` 才作为蓝屏 case 的伴随症状保留。
- 如果现场现象是“无蓝屏直接重启 / 掉电 / 黑屏后自动恢复 / watchdog reset”，应建成另一个 family，例如 `工控机异常重启`。

---

## 8. 通过标准

当这份文档被认为“定版”时，应满足：

1. 拿一条人工案例，至少两个人能拆出高度一致的 family / variant / action / outcome / required-info
2. 对 `verified_fix / mitigation_observed / partial_temporary / pending_validation / diagnostic_method` 的判定没有原则性分歧
3. reviewer 能用本文直接判断一个 v2 candidate 是“语义正确”还是“只是结构合法”

一句话版本：

**KG v2 的难点不是 schema，而是“哪句话该落成哪个对象”。这份文档就是那把尺。**

---

## 9. 案例理解与对象抽取方法

## 9.1 不直接“看到聊天就抽对象”

W2 不应该把原始群聊/Jira/附件文本直接硬映射成 v2 对象。

正确顺序应是：

1. **先理解案例**
2. **再拆语义层次**
3. **最后输出结构化对象**

否则最常见的问题就是：

- 把 variant 抽成 family
- 把整段排查话术抽成一个 action
- 把观察性缓解抽成 verified fix
- 把 ask-info 直接泛化成“发日志/给资料”

## 9.2 推荐的抽取流程

### Stage A：证据预处理（deterministic）

先由非 LLM 逻辑完成：

- episode 切分
- split-case 粗筛信号
- message / Jira / attachment / tool_evidence 归并
- 时间线排序
- 去噪（寒暄、项目汇报、转发噪声）

输出一个 `EvidenceBundle + SourceCase draft`。

### Stage B：案例理解卡（LLM 或规则 + LLM）

在真正抽对象前，先生成一张 **案例理解卡**，内容至少包括：

- 这条案例里有几个故障？
- 每个故障的主症状是什么？
- 哪个是故障家族，哪个只是现场条件？
- 工程师做了哪些动作？
- 每个动作的结果是什么？
- 哪些信息只是证据，哪些信息应该成为结构化 required-info？
- 哪些地方不确定，必须进入 review？

这一步的输出不直接入图，只作为下一步结构化抽取的中间语义层。

### Stage C：结构化对象抽取

从“案例理解卡”生成：

- `FaultFamilyCandidate`
- `FaultVariantCandidate`
- `DiagnosticActionCandidate[]`
- `ActionOutcomeCandidate[]`
- `RequiredInfoSpecCandidate[]`
- `DiagnosticTraceCandidate`
- `EvidenceItemCandidate[]`
- `SourceCaseCandidate`

### Stage D：deterministic 归一化

W3 再做：

- family / variant 命名归一化
- slot 归并
- outcome 类型归一化
- split-case 修正
- action 原子化修正

### Stage E：W4 语义门控

W4 必须拦截：

- family/variant 混淆
- action 非原子
- outcome 类型不清
- required-info 泛化
- 非 verified_fix 却企图 materialize 成 resolved_by

## 9.3 需要 prompt 模板吗？

需要，但 **不是只写一个 prompt 就够了**。

至少要有两套模板：

### Prompt A：案例理解模板

目标：

- 先判断这条案例到底在讲几个故障
- 先区分 family / variant / evidence / action / outcome / ask-info

输出应该是“理解卡”，不是直接出图。

### Prompt B：结构化抽取模板

目标：

- 在理解卡基础上输出严格 JSON
- 只允许输出受控字段
- 不允许自由发挥新字段

输出是 v2 candidate。

## 9.4 Prompt 里必须显式写的约束

无论是理解模板还是抽取模板，都必须写清楚：

1. `FaultFamily` 默认是故障类/症状类，不是具体条件或具体根因
2. `FaultVariant` 才允许带版本、现场条件、局部根因线索
3. 一个 `DiagnosticAction` 只能包含一个动作
4. `ActionOutcome` 不能把多次结果合并成一句模糊描述
5. `mitigation_observed` 不能写成 `verified_fix`
6. `diagnostic_method` 不能 materialize 成 `resolved_by`
7. ask-info 必须是“能缩小诊断空间的信息”，不是泛化索要资料
8. 如果 query、动作、结论跨两个故障，必须 split-case

## 9.5 推荐的 prompt 产物形状

### 案例理解卡

最少字段：

- `case_count`
- `split_required`
- `family_hypotheses[]`
- `variant_hypotheses[]`
- `action_candidates[]`
- `outcome_candidates[]`
- `required_info_candidates[]`
- `uncertainties[]`
- `evidence_anchor_ids[]`

### v2 candidate

最少字段：

- `family`
- `variant`
- `actions`
- `outcomes`
- `required_info`
- `trace`
- `source_case`
- `evidence`
- `uncertainties`

## 9.6 对 case001 和过滤驱动案例的启发

### case001

抽取时不能直接把：

- `编程拍照速度延迟现象`

当成 family。

正确做法是先理解：

- 家族级故障：`相机拍摄失败`
- 现场变体：`编程拍照速度延迟现象`
- 多个 action 和多个 negative outcome

### 过滤驱动案例

抽取时不能直接把：

- `相机网卡过滤驱动取消勾选导致拍摄失败`

当成 family。

正确做法是先理解：

- 家族级故障：`相机拍摄失败`
- 变体：`相机网卡过滤驱动取消勾选导致拍摄失败`

这也是为什么 **先做案例理解卡，再做结构化抽取** 是必须的。
