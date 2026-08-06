# AOI Benchmark：逐 Chunk 阅读生成方式

旧版 `feature_selftest` 生成器已停用。旧逻辑从标题或正文片段抽取 seed，再统一拼接“怎么处理”，并没有理解 chunk；因此会产生标题复述、多个故障混在一个 Query、答案缺失以及薄证据强行出题等问题。

当前生成方式由 Codex 对 chunk 逐条处理。一次生成任务只能看到一个 chunk：首轮模型先决定 `accept` 或 `reject`；首轮通过后，同一 chunk 和候选问答会交给独立终审调用，专门检查自然度、模糊指代、答案相关性和薄证据。两轮都通过才允许收录。模型输出失败时可在同一 chunk 上修订一次，但不允许跨 chunk 补答案。

每条收录记录包含：

- 自然现场 Query；
- 仅由当前 chunk 支撑的 `reference_answer`；
- 可在当前 chunk 中逐字匹配的 `evidence_excerpts`；
- chunk 原始索引、SHA-256、来源、标题和 Section；
- 模型、调用次数、校验错误和 token usage 审计信息。

确定性质量门会拒绝以下结果：Query 太短或不是问句、标题机械加尾缀、提到“文档/chunk”、答案过短、证据为空或不能在原文逐字匹配、以及与已收录 Query 高度近似。并行执行时还会在写入前串行复查批内重复。

仅比较 Query 不足以发现“同一处理正文换了多个标题”的重复记录。因此写入前还会比较去掉标题后的来源正文；正文 trigram Jaccard 相似度达到 `0.90` 时只保留首条。`--resume` 加载旧结果时会执行同一压缩规则并重新连续编号。

## 运行

仓库未保存 `OPENAI_API_KEY` 时，使用本机 Codex 登录态：

```bash
PYTHONPATH=src python scripts/build_chunk_qa_benchmark.py \
  --runtime codex_cli \
  --model gpt-5.6-luna \
  --count 240 \
  --workers 6
```

需要中断后续跑时增加 `--resume`。每个已处理 chunk 都会立即写入 audit JSONL，因此程序中断不会丢失已完成的判定；标记为 `error` 的临时失败不会被视为已处理，续跑时可重试。

任一并行批次中若至少一半调用失败（且失败数不少于 2），生成器会立即熔断并输出未完成状态，防止服务限流或鉴权异常时快速扫过全部候选。降低 `--workers` 或待服务恢复后使用 `--resume` 继续。

若 `.env.local` 中同时配置了 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`，可改用：

```bash
PYTHONPATH=src python scripts/build_chunk_qa_benchmark.py \
  --runtime responses_api \
  --model gpt-5.4 \
  --count 240
```

默认产物仍写入原 `feature_selftest_queries_from_raw_and_fae.*` 路径，以便已有调用方读取；但数据结构已升级为包含 Query、Gold Answer 和证据链的 `debug_agent_system.chunk_qa_benchmark.v1`。

默认候选顺序为 `source_richness`：SOP、FAQ 保持文档顺序，`tech_support` 按 chunk 正文长度从高到低送审。这只减少先处理大量薄记录造成的无效调用，不替代模型阅读或质量判定；每条收录项仍对应一次独立 chunk 生成和一次独立终审。需要严格按原文件顺序时可使用 `--candidate-order input`。

现有三套资产不会被直接拼成新 Gold。生成器会加载 `aoi_debug_benchmark_v1.json`、`aoi_fae_report_benchmark_v2.json` 和 `kg_v2_quality_v1.json` 中的历史 Query，作为全局重复检查池；其 KG/runtime 断言、FAE 候选答案和流程期望不会写入当前 chunk 的参考答案。三套资产的适用方式如下：

- AOI Debug Benchmark v1：复用覆盖标签、历史 Query 和 runtime/KG 契约，答案不转为问答 Gold；
- AOI FAE Report Benchmark v2：复用真实表达、故障主题和人工复审候选，未冻结答案不转 Gold；
- KG_v2 分层测试集：复用 diagnosis/procedure/configuration 等能力维度，继续独立评估定位、首动作、分支、安全与解决门；
- 旧 Feature Selftest：充分证据项可重新送入逐 chunk 双审，薄证据项保留为 reject/负样本来源。

## 边界

- 当前 `debug_chunks.json` 有 4,711 个 chunk，其中大量 `tech_support` 记录只有标题或“远程处理/已解决”等结果，不能支撑 Gold Answer。生成数量是目标上限，不是必须牺牲质量完成的配额。
- chunk 中没有实际图片资源标记。遇到“如下图/图中”且答案依赖图片时必须拒绝；不能用模型猜测图片内容。
- `site`、`date`、`handler` 只用于来源审计，不允许为了让问题看起来更具体而写进 Query。
- 该产物是单 chunk 文档依据的 QA Gold，不等同于跨 chunk/KG 的专家诊断 Gold。
