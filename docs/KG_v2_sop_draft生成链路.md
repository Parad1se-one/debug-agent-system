# KG_v2 生成链路（当前真实状态）

> 状态日期：2026-08-03
>
> 本文主体保留 2026-07-10 curated bootstrap 的历史结构。日常 SOP 更新已经
> 转为 `sync-sop-docs -> W9/W10 -> W4 -> W6 -> W5` 版本化增量链路，详见
> `docs/SOP文档自动更新链路.md`。curated builder 不再允许默认覆盖活动图。

---

## 1. 当前目标

历史 bootstrap 中，`data/kg_v2` 的基础图来自手工 SOP draft；当前活动图还
包含后续 approved-only 增量，已经不是手工 draft 的原样副本。

当前工作方式已经明确为：

1. 继续按 `section map -> inventory -> family map -> manual cards -> unified build` 的流程推进
2. 主输出目录始终只有一个：`data/kg_v2`
3. `data/kg_v2_sop_draft_build` 是写侧 curated build 输入层，不是图输出目录
4. SOP draft builder 不消费 goldcase；10 条 reviewed gold cases 保存在 `data/annotations/goldcases/gold-v1`，另由显式授权的独立映射器写入活动 KG_v2
5. 历史 `data/kg_v2_sop_draft` 已归档到 `data/archive/kg_v2_sop_draft_20260710`
6. 当前已将 `1.主程序`、`2.*` 主体、`3 标定`、`4 BUDDY`、`5 运控`、`6 SPC` 的本地可恢复内容并入统一主图

---

## 2. 当前 authoritative 原始来源

当前阶段使用的原始资料：

1. `data/raw/aoi_debug_agent_sources/异常处理 - 标准操作流程（SOP）.docx`
2. `data/raw/aoi_debug_agent_sources/异常处理_-_标准操作流程（SOP）_MMc7dg3J2o2kZFxz4mOcQytmnAf/fetch.json`
3. `data/raw/现场问题反馈流程.md`
4. `data/raw/aoi_debug_agent_sources/` 下补入的外链 docx / md / pdf

补充说明：

- `SOP.docx`：编号真相源
- `fetch.json`：正文、图片、引用真相源
- `现场问题反馈流程.md`：`FaultFamily` / `escalation_target` 主分类真相源（附表1）
- `aoi_debug_agent_sources` 里的补充 docx/md：用于回填 cite-only section

---

## 3. 当前 authoritative 中间输入

当前 `data/kg_v2` 不直接从原始文档一步生成，而是先落人工结构化输入。

当前已存在的输入层目录：

- `data/kg_v2_sop_draft_build/main_program/*`
- `data/kg_v2_sop_draft_build/hardware_camera/*`
- `data/kg_v2_sop_draft_build/updown_connection/*`
- `data/kg_v2_sop_draft_build/track_belt/*`
- `data/kg_v2_sop_draft_build/sensors/*`
- `data/kg_v2_sop_draft_build/stoppers_lift/*`
- `data/kg_v2_sop_draft_build/system_ops/*`
- `data/kg_v2_sop_draft_build/ipc/*`
- `data/kg_v2_sop_draft_build/review_station/*`
- `data/kg_v2_sop_draft_build/gas_pressure/*`
- `data/kg_v2_sop_draft_build/calibration/*`
- `data/kg_v2_sop_draft_build/buddy/*`
- `data/kg_v2_sop_draft_build/motion_control/*`
- `data/kg_v2_sop_draft_build/spc/*`
- `data/kg_v2_sop_draft_build/gold_cases/*`

`gold_cases/` 是 SOP draft builder 的历史 reviewed 原始材料，不参与该 builder 的 `FaultFamily/FaultVariant/ActionOutcome/DecisionPolicy` 构建。当前 canonical 标注已迁移到 `data/annotations/goldcases/`。
`data/annotations/goldcases/gold-v1/raw_source_texts.json` 汇总 10 条案例的 `source_excerpt` 与 evidence anchors。冻结文件仍保留原始 `graph_ingestion=false`；001–010 后续依据显式人工授权，通过 `ingest_gold_v1_to_kg_v2.py` 独立入图，授权与图哈希记录在同目录 ingestion manifest 中。

辅助产物（节选）：

- `data/results/kg_v2_main_program_gap_audit_20260710.json`
- `data/results/sop_hardware_system_docx_outline_20260710.json`
- `data/results/sop_hardware_camera_subitem_audit_20260710.json`
- `data/results/sop_sensors_section_audit_20260710.json`
- `data/results/sop_stoppers_lift_section_audit_20260710.json`
- `data/results/sop_system_ops_section_audit_20260710.json`
- `data/results/sop_ipc_section_audit_20260710.json`
- `data/results/sop_review_station_section_audit_20260710.json`
- `data/results/sop_gas_pressure_section_audit_20260710.json`
- `data/results/sop_calibration_section_audit_20260710.json`
- `data/results/sop_buddy_section_audit_20260710.json`
- `data/results/sop_motion_control_section_audit_20260710.json`
- `data/results/sop_spc_section_audit_20260710.json`

---

## 4. 当前 canonical builder 入口

> 以下入口现在只用于 bootstrap/rollback。日常更新不得调用它覆盖活动图；
> 正常入口是 `make kg-v2-build-curated`，该 target 已改为版本化 SOP sync。

当前写侧 canonical builder：

- `src/debug_agent_system/agents/write_v2/sop_manual_build.py`
- `WriteSideV2Pipeline.build_curated_sop()`

执行命令：

```bash
cd <repo-root>
PYTHONPATH=src python3 -m debug_agent_system.agents.write_v2.sop_manual_build \
  --target-root data/kg_v2 \
  --build-root data/kg_v2_sop_draft_build \
  --gold-root data/kg_v2_sop_draft_build/gold_cases
```

它会从 manual cards 重建 SOP 图，并把 gold JSON 原样复制到目标目录，但不会把 gold 内容转成图对象：

- `data/kg_v2`

当前统一 summary：

- `data/results/kg_v2_write_side_build_summary.json`

---

## 5. 当前唯一主输出目录

当前唯一 canonical 图输出目录：

- `data/kg_v2`

因此当前目录分工为：

- `data/kg_v2_sop_draft_build`：写侧 curated 输入层
- `data/kg_v2`：当前活动图；基础层来自历史 curated bootstrap，后续由
  approved-only 增量维护
- `data/archive/kg_v2_sop_draft_20260710`：历史手工 SOP 图快照，只用于回溯
- `data/archive/kg_v2_pre_sop_promotion_20260710`：提升前旧自动抽取图，只用于回滚

---

## 6. 当前已入统一主图的覆盖范围

当前已经并入统一主图的章/专题：

- `1. 主程序`（主体）
- `2.1 相机`
- `2.3 上下道连接`
- `2.4 轨道及皮带`
- `2.5 传感器`（部分）
- `2.6 挡块及顶升`（部分）
- `2.7 系统`（多数本地正文 + 多个外链回填）
- `2.8 工控机`（部分外链回填）
- `2.9 复判站`
- `2.10 气压装置`
- `3 标定`
- `4 BUDDY`
- `5 运控`
- `6 SPC`（含外链回填）

具体 section 以：

- `data/results/kg_v2_write_side_build_summary.json`

中的 `section_ids` 为准。

---

## 7. 当前仍未完成的项（真实缺口）

### 7.1 `1. 主程序`
- `1.2.5 SN`：`skip_no_body`
- `1.2.7 OCR&OCV`：`skip_no_body`
- `1.2.8 缺陷`：`skip_no_body`
- `1.2.9 3D相关功能`：`skip_no_body`
- `1.3.1 直通模式`：`skip_no_body`
- `1.4.1.1.3 3D相机初始化失败排查`：`deferred_external_only`
- `1.5.1 关于用户认证的一些知识`：`excluded_process_knowledge`

### 7.2 `2.1 相机`
- `2.1.2.2.7 拍摄失败`：`deferred_docx_fetch_mismatch`

### 7.3 `2.5 传感器`
- `2.5.1.1.6`：`deferred_external_doc_only`
- `2.5.1.1.7`：`deferred_external_doc_only`
- `2.5.1.1.8`：`deferred_external_doc_only`

### 7.4 `2.6 挡块及顶升`
- `2.6.1.1.3`：`deferred_external_doc_only`

### 7.5 `2.7 系统`
- `2.7.1.1.7`：`excluded_process_knowledge`

### 7.6 仍待后续确认的外链/缺源情况
- 任何当前仍未在 `section_ids` 中出现、且 audit 中标为 `deferred_*` 的条目
- `5.1.1.1.2 原始条纹图采集SOP`：`missing_source`

---

## 8. 当前统一主图统计

来自：

- `data/results/kg_v2_write_side_build_summary.json`

当前统计：

- `FaultFamily = 12`
- `FaultVariant = 88`
- `DiagnosticAction = 282`
- `ActionOutcome = 0`
- `RequiredInfoSpec = 163`
- `DiagnosticTrace = 88`
- `DecisionPolicy = 12`
- `EvidenceItem = 88`
- `SourceCase = 88`
- `relations = 1223`

materialized：

- `errors = 100`
- `checks = 272`
- `solutions = 0`
- `traces = 88`
- `outcomes = 0`
- `policies = 100`
- `edges = 548`

---

## 9. 当前已覆盖的 family / variant 概览

当前 families：

- `Buddy问题`
- `主程序软件问题`
- `复判站软件问题`
- `模型优化问题`
- `硬件问题`
- `软件使用及调试问题`
- `运控问题`
- `3D成像问题`
- `工控机/复判站/编程站及操作系统问题`
- `标定问题`
- `外部对接设备`
- `SPC问题`

当前已覆盖的代表性 variants（节选）：

- `Buddy安装报错`
- `运控打不开-运动控制程序错误`
- `SPC页面打不开-浏览器被360劫持`
- `SPC数据采集与导出`
- `SPC联动磁盘数据清理`
- `工控机USB口识别不到设备`
- `工控机无法正常开机`
- `复判站IP与相机IP冲突`
- `BADMARK跳叉板显示异常`
- `复判站复判数据加载板卡失败`
- `区域拍照功能报错-y轴限位异常`
- `气缸/气路积水污染`
- `气压表位置漏气`
- 以及此前各章已经入图的所有 variants

完整名单以 unified summary 为准。

---

## 10. 当前建模纪律

1. 主输出目录只有一个：`data/kg_v2`
2. `kg_v2_sop_draft_build` 只放 reviewed 输入，不放输出图
3. reviewed gold case 只进入 `gold_cases/`；未另行批准前禁止转成图对象、关系或 policy
4. 没有正文的标题不强行建卡
5. 只有外链 `<cite>`、没有本地正文的 section 先 defer，不伪造内容
6. 高密度 section 必须按子项拆，不能整体压成一个大 variant
7. `2D` / `3D` / `运控` / `系统驱动` / `外部对接设备` / `SPC` 责任线不同，不能为省事合并

---

## 11. 下一步

当前本地可恢复的主干内容已经基本并入统一主图。

当前剩余缺源清单：

- `data/results/kg_v2_remaining_missing_source_manifest_20260710.json`

下一步应转入：

### A. 优先回填剩余真正缺源项
- `1.4.1.1.3`
- `2.1.2.2.7`
- `2.5.1.1.6 / 1.1.7 / 1.1.8`
- `2.6.1.1.3`
- `5.1.1.1.2`

### B. 明确哪些条目长期保持 process-knowledge / skip
- `1.5.1`
- `2.7.1.1.7`
- 其他只有流程指引、但不适合进入 fault graph 的条目

### C. 若外链文档继续补充到本地，再继续回填
尤其是 remaining missing-source manifest 中列出的 7 个 section。

如果按“本地已有可结构化 source”为完成标准，当前主图已经覆盖了绝大多数主干章节。
