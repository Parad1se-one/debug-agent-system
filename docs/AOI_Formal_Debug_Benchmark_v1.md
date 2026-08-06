# AOI Formal Debug Benchmark v1

- 版本：`1.0.0`
- 核心集：100 题（validation 60 / held-out test 40）
- 已冻结题目：100 题
- 已冻结独立 Gold：47 题
- 人工批准的 KG conformance Gold：53 题
- 待人工冻结：0 题
- 发布状态：`released`
- 广泛池当前校验：`passed`

> 100 题已由 workspace owner 显式批准并冻结。KG 派生题只作为人工批准的 conformance Gold，未计入独立语义 Gold。

## 核心分层

| 能力层 | 题数 | validation | test |
|---|---:|---:|---:|
| `routing_domain_boundary` | 20 | 12 | 8 |
| `document_retrieval_grounded_answer` | 25 | 15 | 10 |
| `fault_location_first_action` | 25 | 15 | 10 |
| `multi_turn_branch_safety_resolution` | 20 | 12 | 8 |
| `long_context_multi_trace` | 10 | 6 | 4 |

## 广泛回归池

- 238 题 KG/runtime 契约集：只报告 conformance；
- 205 题真实 FAE 候选集：只报告候选覆盖与人工冻结进度；
- 77 题文档 QA 集：只报告检索与来源证据回答；
- 三池不得合并为一个总正确率，也不得与核心集混算。
- 三个池的文件哈希与题数是发布完整性门禁；当前 runtime 兼容性另行报告。
- 当前快照中 238 题与 77 题池有 revision/object 漂移；这是回归信号，不会改写冻结数据。205 题池校验通过，但仍不是 Gold。

## Incident Package 结构回归层

`incident_package_validation.json` 是新增的公开结构验证集，用于检查诊断数据包安全解析、
事件/调用栈/环境抽取、稳定锚点、Evidence Pack 来源闭包和 canonical KG 不变性。

它不属于核心 100 题，不计入语义准确率，也不与 238/205/77 广泛池混算。运行器只读取
公开 validation 样例，不读取 held-out test 或 private Gold：

```bash
PYTHONPATH=src python scripts/run_incident_package_benchmark.py
```

## 冻结与防泄漏

- held-out test 的 `optimization_eligible=false`，默认评分命令只运行 validation；
- test 评分必须显式传入 `--allow-held-out-test`；
- `core_test_inputs.json` 不含答案键；test Gold 单独置于 `private/core_test_gold.json`；
- 公共 `core.json` 只保存版本、覆盖统计和资产索引；完整 master 也位于 `private/`；
- 2026-08-03 至 2026-08-10 的 test 题禁止用于知识或 prompt 优化；
- KG/runtime 派生期望标记为 human-approved conformance Gold，不得冒充独立语义 Gold；
- 每次运行必须携带 commit、模型、prompt hash、KG revision、术语 revision 和运行参数。

## 一键构建、校验与评分

```bash
make formal-debug-benchmark-v1
# gpt-5.6-luna 批量执行 validation，并立即按层评分（支持断点续跑）
make run-formal-debug-benchmark-v1-validation
# 对已有 predictions 做可重复确定性评分
make score-formal-debug-benchmark-v1-validation \
  FORMAL_DEBUG_PREDICTIONS=/path/to/predictions.json
# 独立的诊断数据包结构回归，不计入上述评分
PYTHONPATH=src python scripts/run_incident_package_benchmark.py
```

执行器只把 Query、turns 和 source-only 输入交给模型，不暴露任何 `*_gold` 字段。
模型直接返回统一结构化 prediction，随后由确定性评分器分层评分；失败题独立记录并可断点续跑。
validation/test 按文档、runtime 场景或 source-only 会话组整体切分，禁止同源组跨集合。

## Feature Selftest 兼容格式

- 排除 10 条长时间窗 source-only Gold，从三个广泛候选池抽取 192 题；
- [feature_selftest_queries_kg_runtime.jsonl](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries_kg_runtime.jsonl)：KG/runtime 64 题；
- [feature_selftest_queries_fae.jsonl](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries_fae.jsonl)：真实 FAE 64 题；
- [feature_selftest_queries_document_qa.jsonl](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries_document_qa.jsonl)：文档 QA 64 题；
- [feature_selftest_queries.manifest.json](../data/eval/formal_debug_benchmark_v1/feature_selftest_queries.manifest.json)：记录抽样策略、来源分布、模型及源数据 revision；
- 各组内部继续按 KG source type、FAE chat/candidate、document ID 轮转抽样；
- FAE Query 由 `gpt-5.6-luna` 仅依据当时 `source_input` 自然化改写，原文和后续答案不进入 Query；
- KG/runtime Query 同样由 Luna 仅依据原 Query 自然化，删除任务脚手架及原因/动作提示；
- 每行严格使用 operation_agent 的 12 字段结构，不包含答案或其他 Gold 字段；
- `origin` 保存正式 split 与原始 core case ID。

这三个 JSONL 是 Feature Selftest 输入集，不是独立 Gold。正式 validation Gold 位于 `core_validation.json`；held-out test 的公开输入与私有 Gold 分别位于 `core_test_inputs.json` 和 `private/core_test_gold.json`。
