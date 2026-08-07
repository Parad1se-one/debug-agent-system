# debug-agent-system

**面向 AOI 设备的、以知识图谱驱动的多 Agent 故障诊断系统（含训练/评测闭环）。**

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Parad1se-one/debug-agent-system/actions/workflows/ci.yml/badge.svg)
[![English](https://img.shields.io/badge/README-English-blue)](README.md)

本项目把 AOI（自动光学检测）设备现场排故工作流——工程师从群聊、日志和历史文档中快速定位问题并沉淀为标准化检查流程——重构为独立、确定性的多 Agent Python 包。

系统接收故障描述、证据包（日志、转储、EVTX）与发生时间，输出**带建议顺序的分步排查计划**，并动态规划：解析日志并检索知识图谱、依据检查结果调整后续步骤、信息不足时补问、知识缺口时升级负责人。

> English version: [`README.md`](README.md)。

---

## 亮点

- **确定性、KG 原生运行时** —— `DebugAgentSystem` 提供精简公共 API：`start` / `step` / `diagnose` / `analyze_incident`，覆盖信息充分性判断、诊断规划、分支执行、证据校验与跨轮会话状态；核心回路**不依赖 LLM**。
- **执行图知识治理** —— 将 Error/Check/Solution 图谱升级为执行图：**57 个故障族 · 162 个故障变体 · 585 个诊断动作 · 524 个证据项 · 98 条诊断轨迹 · 131 条分支规则 · 1,548 条物化边**。工单/群聊/Jira/文档等写入统一经 typed 质量门、人工审核队列与 `approved-only apply` 入图。
- **离线回归与安全门** —— **11 例精准集**与**150 例广泛集**衡量故障定位准确率、检查链召回率、证据召回率与不安全动作率（当前为 **0%**）。
- **读写闭环** —— 读侧基于 KG 诊断；写侧（W1–W10）把群聊、文档、Jira、专家修正与诊断反馈版本化、门禁化地写回图谱。

> 完整专有 KG（基于内部现场数据构建）**不随仓库分发**。本仓库提供 schema、**脱敏图谱子集**与公开评测场景，可端到端跑通管线。

---

## 公共 API

```python
from debug_agent_system import DebugAgentSystem

system = DebugAgentSystem.from_config("config/debug_agent_system.yaml")

# 开启诊断会话
first = system.start({"query": "主程序加载用户配置失败，user.cfg.toml异常"})

# 反馈检查结果并推进计划
next_turn = system.step(first["session_id"], "已检查但仍未解决")

# 信息不足时可提交来源绑定的只读证据资源
with_log = system.start({
    "query": "初始化失败，请结合启动日志判断",
    "evidence_resources": [
        {"kind": "log_package", "name": "startup.log", "path": "/tmp/startup.log"}
    ],
})
```

标准响应结构：

```json
{
  "schema_version": "debug_agent_system.response.v2",
  "session_id": "...",
  "status": "ask_info|step|resolved|escalate|failed",
  "answer": "...",
  "required_data": [],
  "family_id": "family:...",
  "variant_id": "variant:...",
  "plan_id": "trace:...",
  "current_action_id": "action:...",
  "evidence_ids": ["evidence:..."],
  "current_check": "...",
  "resolution": "...",
  "confidence": 0.0,
  "escalation_target": "...",
  "sources": [],
  "observability": { "family_id": "...", "variant_id": "...", "retrieval_route": "...", "lock_status": "...", "which_check_solved": "..." }
}
```

---

## 架构

```mermaid
flowchart LR
    subgraph 读侧（诊断）
        Q[故障描述 + 证据] --> C{充分性门禁}
        C -- 不足 --> ASK[ask_info 补问]
        C -- 充分 --> KG[KG_v2 检索 / 子图锁定]
        KG --> P[规划：轨迹 + 分支规则]
        P --> EX[执行检查/动作]
        EX --> V{证据校验}
        V -- 通过 --> RES[resolved + 证据包]
        V -- 未通过 --> P
        P -- 知识缺口 --> ESC[升级负责人]
    end

    subgraph 写侧（知识治理）
        SRC[群聊/文档/Jira/工单] --> W1[W1 采集]
        W1 --> W2[W2 抽取]
        W2 --> W4[W4 质量门]
        W4 --> W6[W6 审核队列]
        W6 -- approved-only apply --> KG
    end

    KG --> 读侧
    读侧 -- 诊断反馈/日志模式 --> 写侧
```

读侧子代理（确定性、由 O0 编排）：`MEM` 会话存储 · `C` 充分性门禁 · `O-LOG` 日志分析 · `O-KG` 图谱检索 · `A` 子图锁定 · `B-D` 拓扑遍历/分支执行 · `O-GEN` 答案生成 · `EA` 证据校验 · `O-ESC` 升级 · `O-EvidenceGap` 证据缺口补全。

写侧子代理：W1 群聊采集 → W2 抽取 → W3 冲突 → W4 质量门 → W5 增量入图 → W6 审核队列 → W7 轨迹组装 → W9 原始文档注入 → W10 章节/用例打包。

---

## 知识图谱（KG_v2）

- Schema：19 类实体（`FaultFamily`、`FaultVariant`、`DiagnosticAction`、`ActionOutcome`、`RequiredInfoSpec`、`DiagnosticTrace`、`TraceStep`、`BranchRule`、`DecisionPolicy`、`EvidenceItem`、`SourceCase`、术语实体等）——见 `data/kg_v2/schema/`。
- 执行视图：物化的 `branches / checks / errors / observations / outcomes / policies / solutions / trace_steps / traces` 及规范边集。
- 术语层：名词概念、表达式、义项与上下文策略，用于查询扩展与别名消解。

当前图谱快照（2026-08-04）：

| 实体 | 数量 |
|---|---|
| 故障族 | 57 |
| 故障变体 | 162 |
| 诊断动作 | 585 |
| 证据项 | 524 |
| 诊断轨迹 | 98 |
| 分支规则 | 131 |
| 物化边 | 1,548 |

---

## 评测

两套无需模型的离线回归，衡量故障定位准确率、检查链召回率、证据召回率与不安全动作率。指标由 `src/debug_agent_system/eval/debug_sim/scorer.py` 计算。

**历史内部快照结果**（完整内部 KG；图谱此后持续增长）：

| 套件 | 配置 | 日期 | 故障定位准确率 | 检查链召回率 | 证据召回率 | 不安全动作率 | 综合 |
|---|---|---|---|---|---|---|---|
| 精准集（11） | baseline | 2026-06-29 | 100% | 100% | 99.17% | 0% | 0.9979 |
| 精准集（11） | SAG | 2026-07-17 | 100% | 100% | 100% | 0% | 1.0000 |
| 广泛集（150） | baseline | 2026-06-29 | 100% | 95.82% | 94.53% | 0% | 0.8593 |
| 广泛集（150） | SAG | 2026-07-06 | 98.67% | 98.93% | 95.33% | 0% | 0.8641 |

> **可复现性说明**：本仓库提供 schema、**脱敏图谱子集**与公开场景，可端到端跑通管线；精确得分取决于完整内部图谱快照与代码版本，上表为冻结的内部基线。可用下方命令自行运行评测，观察在随附子集上的管线行为。

```bash
# 跑 11 例精准集
PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
  --scenario-file data/eval/scenarios/industrial_pc_boot_v1.json --limit 11 --out-dir /tmp/eval

# 跑 150 例广泛集
PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
  --scenario-file data/eval/scenarios/broad_debug_v1.json --limit 150 --out-dir /tmp/eval
```

---

## 快速开始

要求：Python ≥ 3.11。可选 extras：`incident`（python-evtx），以及自备 Codex/DeepSeek 的密钥。

```bash
# 安装（核心运行时，无重型依赖）
pip install -e .

# 运行测试套件（stdlib runner；无需网络、无需 API key）
PYTHONPATH=src python3 tests/run_tests.py

# CLI 冒烟：诊断一条故障
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli diagnose \
  "AOI主程序初始化失败，相机连接异常，请检查相机IP"
```

详见 `docs/` 下的读侧管线、KG 设计、写侧管线与评测方法。

---

## 目录结构

```text
config/                系统配置（存储类型、阈值、路径）
data/kg_v2/            KG_v2 schema、脱敏图谱子集、术语层
data/eval/scenarios/   公开评测场景（11 + 150 例）
docs/                  架构、契约、评测文档
src/debug_agent_system/
  core/                dataclass 契约、配置、可观测性
  knowledge_v2/        KG_v2 存储、SQLite SAG 索引、术语
  agents/read/         读侧诊断子代理（O0..O-ESC）
  agents/write/        写侧知识入图子代理（W1-W10）
  agents/tools/        只读工具注册表与执行器
  runtime/system.py    O0 编排器 / 公共 API
  eval/                debug_sim runner、评分、门禁、基准
  adapters/            CLI 与 QA 适配器、Codex/DeepSeek 读侧 harness
tests/                 stdlib 测试套件（离线、mock）
```

## 数据与出处说明

- `data/kg_v2/` 中的脱敏图谱子集保留结构、移除内部标识/人名/链接/路径；可用 `src/debug_agent_system/knowledge_v2/sqlite_sag_v2.py` 从这些文件重建 SQLite serving 索引。
- 原始专有语料（现场群聊、工单、内部文档）**不随仓库分发**；上文数字来自内部快照。
- 可选 LLM 路径（Codex/DeepSeek）只从本地环境文件读取密钥；仓库不提交任何密钥。

## License

MIT —— 见 [LICENSE](LICENSE)。
