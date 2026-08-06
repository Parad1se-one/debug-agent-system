# KG_v2 读侧术语层 A/B 测试设计

更新时间：2026-08-04

## 1. 目标与边界

本测试用于回答一个明确问题：在代码、模型、Prompt、知识、运行参数和 Query 完全相同时，开启经过审核的术语层，是否能提高现场表达的路由、证据召回或回答质量，同时不引入错误扩展与错误故障锁定。

术语命中只允许：

1. 生成规范化表达；
2. 扩展检索词；
3. 提供实体关系和消歧提示；
4. 输出可审计的解析与检索契约。

术语命中不允许直接锁定 Family/Variant，也不能单独证明根因。

本套件独立于正式 Debug Benchmark 的 gold，不修改正式集答案。它专门测量术语层的边际收益；通过后还需跑正式全量集确认没有跨任务回归。

## 2. 60 题构成

| 子集 | 数量 | 主要目的 |
|---|---:|---|
| 规范名 | 15 | 验证开启术语层不会破坏已经清楚的表达 |
| 现场别称、缩写、错拼 | 20 | 测量“复盘站、IPC、CXP线、码枪、运控卡”等表达的增益 |
| 英文名和中英文混写 | 15 | 测量 WinPE、DDU、CXP camera、SMEMA handshake、DL model 等表达 |
| 歧义词、无关词和禁止错误扩展 | 10 | 防止“主机、板子、PE、SPI、Mark、驱动、回流焊、USB”等脱离 Debug 语境后被强行映射 |

逐题数据在 [cases.jsonl](../data/eval/terminology_ab_v1/cases.jsonl)，每题包含：

- `must_resolve`：B 组应识别的规范概念；
- `must_not_resolve`：禁止进入已解析概念、检索契约或答案证据范围的概念；
- `required_search_pairs`：B 组必须同时检索的“原词 + 规范名”；
- `ambiguous_surfaces`：应保持歧义、不应直接归一的表面词；
- `must_not_lock_variant`：全部 60 题均为 `true`；
- `gold_source_documents`：用于确定性检查来源是否被读取并用于回答。

## 3. 唯一变量

两组都调用同一个 `KGRawCodexPipeline`：

| 实验臂 | `terminology_enabled` | 其他输入 |
|---|---:|---|
| A：control | `false` | 完全相同 |
| B：treatment | `true` | 完全相同 |

运行器在同一进程内从一份公共参数构造两个实验臂，唯一按臂写入的参数是 `terminology_enabled`。默认按题成对交错运行，并用固定种子决定每题先跑 A 还是 B，降低时间顺序和服务波动的影响。

运行前固化并记录：

- Git commit、全仓 dirty patch 审计 hash，以及实际运行依赖代码的独立 hash；
- system prompt version/hash；
- Query 集、实验声明、运行配置、raw 来源清单、KG 关系和术语清单 hash；
- `data/raw` 与 `data/kg_v2` 全树的路径、大小和 mtime 指纹；
- 术语 revision；
- runtime 与 model。

运行结束再次计算实际运行依赖与证据语料指纹；中途发生变化即判本次实验无效。全仓 dirty hash 仅用于审计，不让无关工作流的并行开发制造假失效。

## 4. 运行方式

先做结构校验和当前解析器兼容性预检：

```bash
python scripts/validate_terminology_ab_dataset.py
python scripts/validate_terminology_ab_dataset.py --check-current-resolver
```

只查看 120 次调用的调度与冻结指纹，不调用模型：

```bash
python scripts/run_terminology_ab.py
```

正式运行：

```bash
python scripts/run_terminology_ab.py --execute
```

小批验证可显式指定 `--start` 与 `--limit`；最终验收必须移除这些参数，完整运行 60 题、120 个实验臂，不得用代表集替代。

## 5. 指标

### 5.1 主指标

1. `route_accuracy`：是否进入正确对象、故障域和资料域，0/1；术语命中本身不算正确路由。
2. `evidence_recall`：有 gold 来源的题，至少一个 gold 来源被实际读取且用于回答，0/1。
3. `answer_quality`：事实正确性、问题覆盖、结构可执行性、证据约束各 0/1，总分 0～4；最终用除以 4 的标准化分比较。

答案盲评时隐藏 A/B 标签、术语开关和术语解析 metadata，题内两份答案顺序随机化。来源读取、检索契约与 Variant 锁定由程序确定性评分，不能交给主观评审替代。

### 5.2 安全与机制指标

1. `required_search_compliance`：B 组是否对每个 required pair 同时执行原词和规范名检索。
2. `unsafe_expansion_count`：禁止概念进入解析、检索或答案证据范围的总次数。
3. `wrong_variant_lock_count`：仅因术语命中造成错误 Family/Variant 锁定或确定性根因的次数。
4. 发布失败数、工具失败数和超时数：作为运行健康度护栏。

## 6. 验收门槛

必须同时满足：

1. 别称/错拼 20 题中，路由准确率、证据召回率或标准化回答质量至少一项 `B-A >= 0.10`；其他可用主指标不得下降超过 `0.02`，且逐题 B 胜数大于 A 胜数。
2. 所有非安全题中，每项主指标 `B-A >= -0.02`；规范名和中英文混写子集任一指标不得下降超过 `0.05`。
3. 10 个安全题的 `wrong_variant_lock_count == 0`。
4. 10 个安全题的 `unsafe_expansion_count == 0`。
5. 有 required pair 的 B 组 `required_search_compliance == 1.0`。
6. B 组发布失败数不得高于 A 组。

“接入成功”“命中了术语”或少数示例变好都不构成验收通过。

## 7. 首轮预检暴露的问题

数据契约校验为 60/60 通过；在不调用模型的情况下，用当前解析器预检发现 5 个失败：

| 类型 | Query 表达 | 当前问题 | 应进入的失败桶 |
|---|---|---|---|
| 正向漏解析 | 运控卡 | 未闭包到“运动控制卡” | `nested_or_overlapping_mention` 或检索契约生成 |
| 正向漏解析 | DL model | 未闭包到“DL算法” | `insufficient_context_disambiguation` |
| 安全误扩展 | 驱动业务增长 | 错映射到“设备驱动程序” | `out_of_domain_false_expansion` |
| 安全误扩展 | 回流焊工艺 | 错映射到“回流焊设备” | `insufficient_context_disambiguation` |
| 安全误扩展 | USB 协议栈 | 同时映射到“USB接口” | `nested_or_overlapping_mention` |

这些题应保留为回归门，不应通过删除难题或降低期望来制造收益。

## 8. 未达标后的迭代规则

每个失败必须保存 A/B 工具轨迹、术语解析、required search、实际来源与评分差异，并归入一个主失败桶：

- 缺少别称或错拼；
- 嵌套/重叠 mention；
- 上下文消歧不足；
- 实体关系扩展过宽；
- required search 未执行；
- 已召回但未用于回答；
- 回答编织回归；
- 域外误扩展。

只允许修改通用解析机制、上下文约束、检索契约或已经审核的词义，不增加单 Query 特判。修改后生成新的 terminology revision 和 run id，再完整重跑相同 60 题。首轮若无收益，结论应是“当前策略未通过”，随后按失败桶迭代，而不是只说明术语库已经接入。
