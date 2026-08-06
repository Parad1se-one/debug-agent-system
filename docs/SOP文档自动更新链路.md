# SOP 文档自动更新链路

> 状态日期：2026-08-03

## 1. 结论

SOP 更新不再以 `data/kg_v2_sop_draft_build` 对活动 `data/kg_v2` 做默认
全量覆盖。正常入口改为：

```text
SOP 原始文件变更检测
  -> W9 结构解析与 source-hash chunk manifest
  -> W10 draft KnowledgeDocument/Section/Step + 逐章节 atomic fault mapping
  -> W3 规范化
  -> W4 typed/semantic gate
  -> W6 人工审核（document_layer / fault_mapping 分权）
  -> W5 hash-bound 原子替换
  -> KG v2 validator
  -> terminology / materialized execution / SAG 批次原子刷新
```

“自动更新”表示自动发现新文件和内容变化、自动产生新版本候选并撤销过期
审批，不表示绕过 W6 自动批准。

## 2. 入口

扫描目录中的 SOP 文档：

```bash
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli sync-sop-docs \
  data/raw/aoi_debug_agent_sources \
  --kg-v2-root data/kg_v2 \
  --out data/results/sop_document_sync_latest.json
```

单文件：

```bash
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli ingest-sop-doc \
  'data/raw/aoi_debug_agent_sources/异常处理 - 标准操作流程（SOP）.docx' \
  --kg-v2-root data/kg_v2
```

兼容 Makefile 入口：

```bash
make kg-v2-build-curated
```

该 target 现在等价于 `sop-doc-sync`，只生成/更新 W6 候选。旧 builder
改名为：

```bash
make kg-v2-build-curated-legacy
```

旧命令带有 `--allow-active-rebuild`，只允许在已经建立活动图备份且明确执行
bootstrap/rollback 时使用。

## 3. 信任契约

SOP 增量 envelope 必须同时满足：

- `source_type=sop_doc`；
- `source_kind=sop`；
- `metadata.incremental_source_contract=sop_document_incremental.v1`；
- source path 是原始文件，不能位于 `data/kg_v2_sop_draft_build`。

缺少任意条件，W4/W5 仍按伪造 SOP 来源拒绝。该契约不会放宽群聊、附件、
人工修正或普通 raw document 的 provenance 边界。

## 4. 新增、重放与换版

### 新增

不存在相同 `source_path` 的已批准 `KnowledgeDocument` 时，生成
`queued_new`。W9/W10 chunk 固定为 `pending_review`。

### 无变化

已批准文档的 `source_path + content_hash` 与源文件相同时返回
`unchanged`，不重复排队。

### 换版

同路径 bytes 变化后：

1. document/bundle/content hash 改变；
2. W6 旧决定不继承，新版本进入 pending/needs_re_review；
3. 每个原子 mapping 与 document layer 使用相同的 source path/hash；两类 scope 必须分别批准；
4. W5 在换版时将 document layer 与首个原子 mapping 组合成一次 `document_mapping_pair` graph commit，其余同版本 mapping 随后增量合并；若完全相同的已批准文档版本已在图中，mapping 可直接复用该文档依赖；
5. 旧 Document/Section/Step/Evidence 和仅由旧来源支撑的 Action/RequiredInfo
   被清理，共享 Family/Variant/Action 保留；
6. terminology 与 SAG 在最终图验证成功后重建。

审核后、应用前源文件再次变化时，W5 返回
`source_content_changed_since_review` 并撤销批准。

## 5. 格式能力

| 格式 | 当前处理 |
|---|---|
| DOCX | 段落、标题、列表、表格、媒体、链接、offset |
| Markdown/TXT | 标题、段落、表格、offset |
| XLSX | 全 worksheet 文本行，保留 table-row 边界，不计算公式 |
| PPTX | 全 slide 文本，不只读取第一页 |
| PDF/DOC/XLS/PPT | 无可靠完整正文解析时 `review_only` |
| 扫描件/图片 | 不做 OCR，走附件人工复核 |

W9 不执行宏、不计算公式、不运行嵌入对象，也不根据无法解析的文件名生成
执行动作。

## 6. 多主题 SOP 的原子故障映射

`sop_fault_catalog_doc` 不再沿用整篇文档只选一个 Family/Variant 的规则：

1. W9 去除目录与正文的重复 section，按带操作的故障叶子章节生成 `atomic_case_id`；
2. Family 只使用当前标题路径和当前章节语义，不扫描整份 SOP 的首个关键词；
3. W10 为每个 case 生成独立的 FaultFamily、FaultVariant、DiagnosticAction、EvidenceItem 和 SourceCase；
4. mapping bundle 不重复携带 KnowledgeDocument/Section/ProcedureStep，避免每个 case 触发 source-scoped document replacement；
5. EvidenceItem 通过 source section anchor 绑定推荐动作；SOP action 不要求现场 ActionOutcome；
6. W4 对 45 个 bundle 分别门控，W6 对每个 bundle 分别记录审核决定和 content hash；
7. `approve_support_only` 不生成 DiagnosticTrace、ActionOutcome 或 DecisionPolicy，也不进入自动执行图。

Family ID 对纯中文标签沿用历史 hash 形式；对 `CAD 导入失败`、`CAD 角度不一致` 等 ASCII/CJK 混合标签追加完整语义 hash，防止 slug 碰撞。W3 只在强证据下改写宽泛 Family，不能因 action 详情提及 SPC/MES/进板而覆盖已经明确的故障 Family。

## 7. 旧 curated build 的角色

`data/kg_v2_sop_draft_build` 继续作为历史 reviewed manual cards 和回滚输入，
不再是日常更新权威入口。`WriteSideV2Pipeline.build_curated_sop()` 对活动
`data/kg_v2` 默认 fail-closed；临时 KG 测试和显式
`allow_active_rebuild=True` 的恢复操作仍可使用。

旧 builder 在目标已经存在新版 schema 时不会再用 build root 的 v2.0 schema
覆盖它，避免恢复基础图后破坏文档增量对象类型。

## 8. 2026-08-03 验证

- 文档/SOP 定向回归：103 项通过；
- 临时 KG 上完成 SOP 新增、无变化跳过、换版重新审核、document/mapping
  原子替换；
- 使用真实《异常处理 - 标准操作流程（SOP）》生成 131 个 Section、37 个
  ProcedureStep 和 102 个 source-aligned chunk；document layer 为 W4
  `admit`；
- 真实 SOP 拆出 45 个原子 fault mapping，覆盖 36 个 Family、45 个 Variant、87 个 DiagnosticAction；45 项均通过 W4，并由 W6 以 `approve_support_only` 逐项批准；
- W5 45/45 成功合并，KG validator 为 `valid`，SAG v15 重建成功；最终 7,534 个对象、14,068 条关系；
- DiagnosticTrace、ActionOutcome、DecisionPolicy 数量保持不变，确认 SOP 推荐动作未被误当成现场已执行事实；
- 临时批准后的完整图 validator 通过并完成 SAG build；
- 本轮 SOP/W3/W4/W5 定向回归：120 项通过。
