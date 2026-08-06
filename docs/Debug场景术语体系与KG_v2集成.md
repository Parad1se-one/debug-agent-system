# Debug 场景术语体系与 KG_v2 集成

> 实施日期：2026-07-30；名词语料发现层更新：2026-07-31
>
> 当前版本：`kg_v2.debug_terminology.v4`
> 目标：优先打通 Debug 场景中的设备、型号、站点、部件、软件、接口和工件名词，
> 把复合 KG 字段拆成原子实体和有类型关系；现场别称进入审核闭环。Action/Operation
> 只保留既有兼容投影，不再作为本轮候选治理重点。

## 1. 已实现的总体结构

术语层不是 KG_v2 之外的一份孤立词典，而是 KG_v2 的概念投影层：

```text
FaultFamily / FaultVariant ── primary_concept ──> 故障 DebugConcept

DiagnosticAction 实例 ── primary_concept ──────> Operation DebugConcept
        N 个 Trace 实例                         1 个可复用语义操作

equipment_type / subsystem 复合字段
        │
        ├─ 原子化 ──> 型号 / 设备 / 站点 / 工作站 / 部件 / 软件 /
        │               接口 / 连接 / 协议 / 工件
        └─ 关系化 ──> model_of / is_a / part_of / runs_on /
                       deployed_at / has_component / has_interface /
                       connected_via / endpoint_of / uses_protocol /
                       installed_in / powered_by / signals_to

               DebugConcept
                    ▲
                    │ sense_denotes
                TermSense
                    ▲
                    │ expression_has_sense
              TermExpression

DebugConcept ── broader_concept ──> 上位故障概念
DebugConcept ── concept_context ──> 类别 / 子系统 / 设备 / 阶段
Document / Section / Step ── mentions_concept ──> DebugConcept

群聊历史 / 文档 Chunk / 技术支持记录
        │
        ├─ 配置候选：设备、部件、软件、接口、工件、数据对象
        ├─ 开放发现：设备型号、程序/配置/数据库文件、显式简称
        ├─ 真实前缀归一：SI-252T / SI252T / 252T
        ├─ 记录级共现：只生成 associated_with 高风险候选
        ├─ 名词概念候选 / 变体叫法候选 / 结构关系候选 / 关联候选
        ▼
 noun_discovery_candidates.json（无诊断权限）
        │ 逐项审核，显式选择规范名、类型、目标和关系
        ▼
 entity_ontology.json（正式名词图）── 确定性重建 KG_v2 术语层
```

三类对象职责如下：

| 对象 | 作用 |
|---|---|
| `DebugConcept` | 稳定概念身份；故障概念对应唯一 Family/Variant，操作概念可由多个 Action 实例共同实现 |
| `TermExpression` | 用户和资料中实际出现的字符串，包括规范名、别名、简称、英文名和错拼 |
| `TermSense` | 某个表达在特定上下文中指向哪个概念，并记录关系类型、设备、子系统和阶段 |

Family、Variant 和 Action 仍是诊断真值。名词关系只表达结构、类别、承载或连接关系，
不证明现场根因。共享 Operation 只保留为向后兼容能力；各 Trace 中的条件、顺序、
参数、安全要求和验证标准仍保留在原 Action 上。术语层不能替代 Variant 锁定、
BranchRule、安全确认或 `verified_fix`。

### 1.1 “工作站”在本体系中的含义

`station` 与 `workstation` 是两个不同层级：

- `station` 是业务职责或功能工位，例如 AOI主站、复判站、编程站；
- `workstation` 是承载该职责的物理计算节点，例如 AOI主站工控机、复判工作站、
  编程工作站；
- 工控机是计算设备类别，工作站是“计算设备 + 部署角色 + 运行环境”的节点概念，
  因而工作站不等同于某个固定品牌或固定 BOM。

工作站通常包含六层内容：

1. 计算硬件：主板、CPU、内存、系统盘、网卡，以及按设备配置安装的图像采集卡、
   运动控制卡等；
2. 操作系统与驱动：Windows、网卡/显卡/采集卡驱动及设备服务；
3. 业务软件：AOI主程序、复判站软件或编程软件；
4. 网络与设备通信：以太网、CXP、串口、SMEMA 等连接；
5. 人机外设：键盘、鼠标、显示器、加密狗和 U 盘等；
6. 本地数据与配置：程序、模板、日志、授权和站点配置。

这些内容不是全部用 `part_of` 表示。硬件组成使用 `has_component`，软件使用
`runs_on`，物理节点与功能工位使用 `deployed_at`，外设和设备链路使用
`has_interface / connected_via / endpoint_of / uses_protocol`。类型级典型拓扑均标记
`scope=type_pattern`；只有补齐具体设备实例、端点和来源锚点后，才能成为现场连接事实。

### 1.2 典型连接的中间层

连接不再压缩成一条含义模糊的“设备 A `connected_to` 设备 B”，而按以下层次表达：

```text
设备或部件 ── has_interface ──> 接口
接口       ── endpoint_of ────> 连接
线缆       ── part_of ────────> 连接
连接       ── uses_protocol ──> 协议
设备或部件 ── connected_via ──> 连接
```

当前批准的典型连接模式包括：

| 端点 A | 接口/介质/协议 | 端点 B | 关键关系 |
|---|---|---|---|
| 键盘、鼠标、加密狗、U盘 | USB接口、可选 USB线缆、USB协议 | 工作站 | `connected_via USB外设连接` |
| 显示器 | HDMI/DisplayPort 接口、线缆和协议 | 工作站 | `connected_via HDMI/DisplayPort显示连接` |
| 相机 | CXP接口、CXP线缆、CoaXPress协议 | 图像采集卡 | `connected_via CXP采集连接` |
| 相机、网卡或工作站 | 以太网接口、以太网线、以太网协议 | 网络对端 | `connected_via 以太网连接` |
| 光源控制器 | 串行接口、串口线缆、串行通信协议 | 工作站 | `connected_via 串口控制连接` |
| 上游设备、AOI、下游设备 | SMEMA接口、信号线缆、握手协议 | 相邻产线设备 | `connected_via`；有方向证据时使用 `signals_to` |
| 系统盘 | SATA接口、SATA数据线、SATA协议 | 主板 | `connected_via SATA存储连接` |
| 图像采集卡、网卡、运动控制卡 | PCIe接口、PCIe协议 | 主板 | `installed_in`；M.2 设备单独关联 M.2接口 |

这些都是可复用的类型模板，不声明每台工作站都有全部接口，也不声明每台相机都同时
支持 CXP 和以太网。实例化时仍必须依据 BOM、设备型号、现场照片、日志或原文连接说明。

## 2. 当前构建结果

本次从现有 KG_v2 确定性构建：

| 项目 | 数量 |
|---|---:|
| 概念 `DebugConcept` | 884 |
| 表达 `TermExpression` | 1257 |
| 词义 `TermSense` | 1334 |
| Family/Variant/Action 原生投影 | 642 |
| DiagnosticAction 实例 | 501 |
| Operation 规范概念 | 486 |
| 被多个 Action 复用的 Operation | 12 |
| 合并掉的重复 Operation 概念 | 15 |
| 类别概念 | 3 |
| 名词层概念 | 187 |
| 设备 / 型号 / 站点 / 工作站 | 11 / 9 / 3 / 4 |
| 部件 / 软件 / 接口 / 连接 / 协议 / 工件 | 29 / 22 / 14 / 10 / 9 / 1 |
| 驱动 / 固件 / SDK | 5 / 3 / 3 |
| 软件产物 / 运行进程 / 配置 / 数据库 / 日志 / 诊断文件 | 8 / 1 / 9 / 4 / 1 / 2 |
| 子系统概念 | 39 |
| 故障阶段概念 | 67 |
| 语义实体关系 | 196 |
| `context_member` | 55 |
| 待审核名词别称 / 已批准名词别名 | 9 / 2 |
| `concept_context` 关系 | 1392 |
| 文档概念提及 `mentions_concept` | 167 |
| 待消歧表达 | 59 |

当前 revision：

`dbf2c31b995bb5757433283c7d3d664740aa8d794a078a67b1e7cc993193f6d2`

完整计数、歧义表达和 legacy 概念清单保存在
`data/kg_v2/terminology/terminology_manifest.json`。

在正式图之外，本轮对 137,043 条去重群聊、4,711 个文档 Chunk 和 4,563 条去重
技术支持记录执行了名词发现：

| 发现项 | 数量 |
|---|---:|
| 新名词概念 | 127 |
| 变体叫法 | 58 |
| 结构关系候选 | 82 |
| 记录级关联候选 | 97 |
| 合计 | 364 |

候选覆盖部件 24 项、接口 10 项、软件 25 项、数据对象 23 项、产品型号 15 项、
检测对象 10 项、工件 6 项、设备 6 项、标识符 4 项、外部系统 3 项，以及 1 种材料。
除板卡、器件、焊盘、MES、SDK、Mark 点、加密狗、轨道、料号、条码、显卡、网卡、
传感器、皮带和接驳台外，本轮还从真实语料补出了 `Buddy`、`SPC`、`Jira`、`CUDA`、
`Qt`、`MVS`、`DDU`、`Microsoft Defender`、`EAP`、`SMEMA`、`PLC`、`CAD`、
`RGB图`、`OCR/DL/ODA/ODB`、`IC/LED/BGA/QFN/QFP/CHIP`，以及
`SI1020T/SI252T/SY2600D` 等机型和 `machined.exe/user.cfg.toml/host.db` 等文件专名。

型号归一只在语料中观察到唯一真实前缀时，才把无前缀写法并入同一候选：
`SI-252T`、`SI252T` 和上下文中的 `252T` 可以进入同一审核项；`SY-2600D` 不会被
臆造为 `SI2600D`，验证码式的 `SY2023` 也会因缺少设备/型号上下文而被过滤。上述所有
结果均带来源证据，在审核通过前不会成为正式词义或诊断事实。

## 3. 名词实体如何与 KG_v2 打通

### 3.1 结构化字段原子化

构建器读取 `FaultVariant.equipment_type` 和 `FaultFamily.subsystem`，按 `/` 或 `／`
拆解复合字段，并保留原字段作为 `legacy` 上下文锚点。例如：

| KG 原值 | 原子实体 | 关系 |
|---|---|---|
| `SI2020T/工控机` | `SI2020T`、`工控机` | `SI2020T model_of 工控机` |
| `复判站/软件` | `复判站`、`复判站软件` | `复判站软件 runs_on 复判站` |
| `3D相机/CXP链路` | `3D相机`、`CXP链路` | 通过复合锚点保留同一上下文 |
| `工控机/启动链路` | `工控机`、`工控机启动链路` | `工控机启动链路 part_of 工控机` |

“软件”“启动链路”这类离开父对象就含义不完整的通用子名会被限定为“复判站软件”
“工控机启动链路”，不会在全局创建一个含义不明的“软件”概念。单值字段如“工控机”
直接复用原子实体，不再另外创建同名上下文概念。

复合锚点只存在于图中，通过 `context_member` 连接原子实体；它不创建
`TermExpression/TermSense`。这是因为 `复判站/软件` 和 `复判站软件` 归一化后相同，
如果两者都作为规范词义会制造假歧义。

### 3.2 关系语义

| 关系 | 方向 | 例子 | 是否可证明根因 |
|---|---|---|---|
| `model_of` | 型号 → 设备类别 | `SI2020T → 工控机` | 否 |
| `is_a` | 子类 → 上位类 | `3D相机 → 相机` | 否 |
| `part_of` | 部件/子系统 → 所属对象 | `工控机启动链路 → 工控机` | 否 |
| `runs_on` | 软件 → 运行载体 | `Windows → 工控机` | 否 |
| `connected_to` | 站点/设备 → 相连对象 | `复判站 → AOI设备` | 否 |
| `processed_by` | 工件 → 处理设备 | `PCB → AOI设备` | 否 |
| `driver_of` | 驱动 → 硬件 | `显卡驱动 → 显卡` | 否 |
| `firmware_of` | 固件 → 嵌入设备/部件 | `BIOS固件 → 主板` | 否 |
| `sdk_for` | SDK → 支持的设备/能力 | `相机 SDK → 相机` | 否 |
| `artifact_of` | 部署产物 → 逻辑软件 | `smt-aoi.exe → AOI主程序` | 否 |
| `configuration_of` | 配置文件 → 配置对象 | `user.cfg.toml → AOI主程序` | 否 |
| `associated_with` | 实体 ↔ 语料关联实体 | `扫码枪 ↔ MES` | 否 |
| `context_member` | 复合锚点 → 原子实体 | `SI2020T/工控机 → SI2020T` | 否 |

正式关系来源只有两种：KG 结构化字段的确定性投影，或
`data/kg_v2/terminology/entity_ontology.json` 中明确标记 `approved=true` 的人工
基线。raw 语料可以生成带证据的关系候选，但不能自行生成已批准关系。特别是
`associated_with` 只表达“在多条独立记录中共同出现”，不等价于 `part_of`、
`runs_on`、`driver_of`、`firmware_of` 等结构语义。`communicates_with`、
`identifies`、`input_of` 和 `output_of` 虽已由 schema 预留，但当前正式本体没有
批准实例；只有补齐对象边界、方向和来源后才能进入结构图。

### 3.3 关系进入检索

resolver 的 `entity_relations` 返回命中名词的一跳语义邻域，但隐藏内部兼容用的
`context_member`。经审核的 `associated_with` 可以返回给 Codex 作为关联上下文，
但固定标记 `can_expand_retrieval=false`、`can_lock_variant=false`。对于有方向的
上位结构关系，resolver 还会生成 `authority=entity_relation` 的低权限检索扩展：

```text
Query: SI2020T不开机
resolved: SI2020T
entity_relation: SI2020T model_of 工控机
retrieval expansion: 工控机
```

因此该 Query 能检索到《工控机不开机手册》。只允许从已提及的子实体向上位类、整体或
运行载体扩展；不会从“工控机”反向展开全部子系统，以免宽召回失控。关系扩展和
`search_hint` 一样均为 `can_lock_variant=false`。

## 4. 规范名、同义词和检索提示必须分开

现有 Family/Variant 的 `keywords` 有 465 个。它们最初用于宽召回，不足以证明概念
等价，因此全部投影为 `search_hint`：

- 可以扩展检索；
- 可以让候选进入比较；
- 不允许直接锁定 Variant；
- 不允许生成未经 KG 支持的执行动作；
- 多个概念共享同一关键词时必须返回歧义候选。

只有审核通过并写入 `curated_terms.json`（故障术语）或
`entity_ontology.json.aliases`（名词术语）的表达，才可以成为安全等价关系。支持：

- `exact_synonym`
- `colloquial_alias`
- `abbreviation`
- `english_equivalent`
- `historical_name`
- `typo_variant`

示例结构：

```json
{
  "surface_form": "复盘站",
  "relation_type": "typo_variant",
  "concept_key": "station:复判站",
  "approved": true
}
```

故障术语也可继续使用 `canonical_target_type + canonical_target_id` 指向
Family/Variant。构建器会拒绝未知关系类型、空表达和不存在的目标概念。`approved`
不是 `true` 的条目不会生效。这样可以逐步积累术语，而不是把所有相似词一次性自动
合并。

最初只从 raw 快照人工整理了 8 个种子候选；现在多源发现队列已用统一统计口径重新
扫描群聊、文档 Chunk 和技术支持记录。部分代表性变体为：

| 现场表达 | 建议规范概念 | 建议关系 | 多源出现次数 | 当前权限 |
|---|---|---|---:|---|
| 板子 | PCB | `colloquial_alias` | 4362 | 待审核 |
| Mark | Mark点 | `colloquial_alias` | 3057 | 待审核 |
| 码枪 | 扫码枪 | `colloquial_alias` | 1191 | 待审核 |
| 元件 | 元器件 | `colloquial_alias` | 1178 | 待审核 |
| 复盘站 | 复判站 | `typo_variant` | 800 | 待审核 |
| 感应器 | 传感器 | `colloquial_alias` | 769 | 高风险待审核 |
| 主机 | 工控机 | `colloquial_alias` | 534 | 高风险待审核 |
| 二维码 | 二维码 `is_a` 条码 | 子类关系 | 465 | 待审核 |
| PCB板 | PCB | `exact_synonym` | 131 | 待审核 |
| Barcode | 条码 | `english_equivalent` | 77 | 待审核 |
| USB口 | USB | `colloquial_alias` | 70 | 待审核 |
| SSD | 固态硬盘 | `abbreviation` | 39 | 待审核 |
| IPC | 工控机 | `abbreviation` | 23 | 高风险待审核 |
| 工业相机 | 工业相机 `is_a` 相机 | 子类关系 | 3 | 待审核 |

出现次数只表示去重语料中字符串出现频率，不等于同义关系正确。例如“主机”可能不是
工控机，“二维码”是条码子类而非无条件同义词，“工业相机”也可能需要保留为相机子类。
因此这些项只进入有证据的候选图；人工可把建议关系改为更准确的
`abbreviation`、`exact_synonym` 或拒绝合并。

## 5. Action/Operation 兼容投影

本轮不再扩展动作名称治理。已有能力继续保留：KG_v2 中的 `DiagnosticAction` 是
Trace 内的执行实例，不天然等于“唯一术语概念”；相同动作可投影到共享 Operation，
但不会删除或合并 Action 本身。

Operation 的身份为：

```text
normalize(label) + action_role
```

这是一条通用的语义投影规则：

- 相同规范名称、相同动作角色的 Action 复用一个 Operation；
- 每个 Action 仍通过 `primary_concept` 指向该 Operation，来源实例不丢失；
- Operation 保存全部 `source_object_ids`；
- 相同名称但角色不同不会自动合并，例如“检查电源线”的 `inspect` 与 `change`
  仍是两个概念；
- 具体 Trace 条件和安全约束不提升到共享概念，避免跨场景误用步骤。

因此，概念去重不会删除 KG 对象，也不会把两个故障 Variant 合并，只消除术语层中由
Action 实例化方式造成的重复规范概念。

## 6. 同词多义与上下文消歧

术语 resolver 输出三个不同集合：

- `resolved_mentions`：当前表达只对应一个已批准词义；
- `ambiguous_mentions`：同一表达对应多个概念，且当前上下文不足或候选差距不足；
- `supporting_concepts`：只由 `search_hint` 命中，可辅助召回但不能诊断锁定。

例如，一个动作名称可能在多个 Trace 中重复出现；“蓝屏”“闪退”“相机 IP”等宽泛表达
也可能对应多个 Variant。resolver 会从 Query 中自动识别类别、设备、子系统和阶段，
也接受调用方显式传入已观测到的 `signals` 等结构化上下文。候选评分权重为：

| 上下文字段 | 单项匹配分 |
|---|---:|
| 设备 `equipment_types` | 3.0 |
| 子系统 `subsystems` | 3.0 |
| 阶段 `phases` | 2.0 |
| 信号 `signals` | 2.0 |
| 类别 `categories` | 1.5 |

仅当第一候选至少命中一个上下文、总分不低于 2.0，且与第二候选的分差不低于 2.0 时，
才输出 `resolution_method=context_disambiguation`。否则保留全部候选，返回
`context_required` 或 `context_margin_insufficient`，并给出 `required_context`。
排除信号命中会额外扣分。系统不会按数组顺序、Top1 默认值或单个 `search_hint`
选择概念。

调用方可显式传入上下文：

```json
{
  "text": "初始化异常",
  "context": {
    "equipment_types": ["相机"],
    "subsystems": ["采集链路"],
    "phases": ["检测"],
    "signals": ["拍照失败"]
  }
}
```

## 7. 候选术语审核工作流

候选生成与术语生效完全分离。目前有两个职责不同的队列：

- `terminology_candidates.json`：已有 KG 概念的歧义和 `search_hint` 晋升；
- `noun_discovery_candidates.json`：从群聊、文档和支持记录发现的新名词、变体和关系。

第一类队列共 430 项：

| 候选类型 | 数量 | 含义 |
|---|---:|---|
| 名词候选 `noun_*` | 8 | 设备、站点、工件的错字、简称或现场别称 |
| `ambiguous_expression` | 47 | 同一表达关联多个概念，需要选择、保留歧义或拒绝 |
| `alias_promotion` | 375 | 当前只有 `search_hint`，可评估是否升级为人工等价表达 |

生成队列：

```bash
make kg-v2-terminology-review-build
```

生成器会为每项保存候选概念、类别/设备/子系统/阶段上下文、KG 来源对象、风险级别和
内容 hash。名词候选额外保存 `review_domain=noun_entity`、raw 出现次数和最多 5 个
语料来源路径，并优先排在队列前面。纯 Operation 候选不再进入新增审核队列。新候选
全部为 `pending`；重新生成时，内容未变化的人工决策会保留，已审核候选内容发生变化
时转为 `needs_re_review`，不会沿用旧批准。

审核人需要填写：

```json
{
  "review_status": "approved",
  "selected_action": "approve",
  "selected_concept_id": "concept:...",
  "approved_relation_type": "colloquial_alias",
  "reviewed_by": "领域审核人",
  "reviewed_at": "2026-07-30T00:00:00Z",
  "review_note": "审核依据"
}
```

也可以设置 `selected_action=reject|defer`。只有目标属于该项候选、关系类型合法、
审核人非空且决策不冲突的批准项才能导入：

```bash
make kg-v2-terminology-review-apply
```

导入器只把有效批准追加到 `curated_terms.json`，然后确定性重建术语层。无效目标、
非法关系、缺审核人，以及“状态为 rejected 但 action 为 approve”等冲突记录都会
拒绝并返回非零状态。当前 430 项均保持 `pending`，没有自动批准。

### 7.1 多源名词发现队列

生成：

```bash
make kg-v2-noun-discovery-build
```

发现器使用 `data/kg_v2/terminology/noun_discovery_config.json`，以可重跑的线性阶段
扫描三种异构来源。群聊按 `message_id` 去重，文档按 Chunk 身份与正文去重，技术支持
记录按 `record_id` 去重；ASCII 术语使用词边界，避免把 `PE` 计入任意英文子串。
除了配置候选，开放发现器还识别设备型号、程序/配置/数据库/日志文件名和显式简称声明。
每项候选保存：

- 各来源出现次数、去重记录数、覆盖来源种类；
- 最多 5 个带来源路径和来源 ID 的上下文片段；
- 建议规范名、概念类型、风险和稳定 `content_hash`；
- 对变体保存建议目标和别名关系；
- 对结构关系保存建议起点、终点和有向关系；
- 对共现关联保存共同记录数、Jaccard、来源覆盖、样例和 `associated_with` 建议。

当前队列 364 项全部是名词域，不包含 Operation/Action。高风险表达不会因为频率高而
自动合并，例如：

- `主机` 可能是工控机，也可能是其他主机；
- `板卡` 可能是 PCB 工件，也可能是硬件扩展卡；
- `感应器/传感器` 在现场常混用，但具体类别可能不同；
- `DP`、`SN`、`大图`、`日志` 都依赖上下文；
- `二维码` 是条码子类，不应无条件等同全部条码。
- `SI-252T/SI252T/252T` 只有在真实前缀唯一时才建议归一；
- `user.cfg.toml`、`machined.exe`、`host.db` 按文件身份发现，不从任意英文串猜专名；
- `Buddy ↔ 主程序`、`扫码枪 ↔ MES` 一类共现只进入关联审核，不直接变成结构边。

审核时必须显式填写选择字段，不能只把状态改为 approved：

```json
{
  "selected_action": "approve",
  "selected_canonical_name": "轨道传感器",
  "selected_concept_type": "component",
  "reviewed_by": "领域审核人"
}
```

变体必须填写 `selected_concept_key + approved_relation_type`；结构关系和关联候选
都必须填写 `selected_relation + selected_target_key`，不能把建议关系直接当成审核
结论。导入命令：

```bash
make kg-v2-noun-discovery-apply
```

导入器把新概念和关系写入 `entity_ontology.json`，把已批准名词变体写入独立的
`aliases`，不会混回仍为低权限的 `alias_candidates`。随后确定性重建术语层。缺少
审核人、缺少显式选择、未知概念类型、非法关系或不存在目标都会被拒绝。

队列再次生成时，内容 hash 未变化的审核决定会保留；证据或建议发生变化的已审核项
会转为 `needs_re_review`。汇总报告保存在
`data/kg_v2/terminology/noun_discovery_report.json`；本次候选报告生成到
`data/kg_v2/terminology/noun_discovery_report.md`。正式概念、已批准别名、正式关系和
全部待审核候选合并后的统一大表生成到
`data/kg_v2/terminology/noun_terminology_inventory.json` 和
`data/kg_v2/terminology/noun_terminology_inventory.md`。

## 8. 与读侧的结合

独立 `KG_v2+raw Codex` 旁路在每次模型调查前执行确定性术语解析。它不是暴露给模型
自行调用的检索 Tool，也不生成预排序文档。真实顺序为：

```text
Query
  → Answer Scope 投影阶段/信号上下文
  → TerminologyResolver.resolve(query, context)
  → TERM_RESOLUTION
  → TERMINOLOGY_SEARCH_CONTRACT
  → Codex 使用通用只读文件能力自主调查
       ├─ Responses API：list_files / search_text / read_text
       └─ Codex CLI：rg --files / rg / sed 等只读 shell
  → Codex 组织答案与 coverage ledger
  → facet/source/terminology/media/safety verifier
```

`approved_equivalence` 命中时，契约要求原始表达和规范名都真实出现在模型的搜索调用
中；本地只审计是否搜索，不指定命中文档、评分、Top-K、顺序或停止条件。`search_hint`
和 `authority=entity_relation` 是可选附加检索词，均保持 `can_lock_variant=false`。
`ambiguous_mentions` 不会默认选择第一项；模型需要使用 `required_context` 继续调查或做
最小追问。最终诊断仍必须由 KG_v2 对象、关系、raw 原文和现场证据共同支持。

运行时权限边界为：

- 正式输入：`entity_ontology.json`、`curated_terms.json` 中有效批准内容；
- 运行投影：确定性生成的 `DebugConcept`、`TermExpression`、`TermSense` 与关系；
- 仅治理：inventory、discovery report、review queue、人工审核建议；其中
  `pending/rejected/needs_re_review` 不能成为自动等价、诊断事实或执行依据。

每个回答产物保存 `terminology_manifest` 的 `terminology_version`、`revision` 和计数，
以及 `terminology_context`、`terminology_resolution`、`terminology_search_contract`、
`terminology_search_audit`。因此可以审计某条回答具体使用了哪版术语层，以及批准别名的
原词和规范词是否都被实际搜索。

冻结的 `DebugAgentSystem.start → SAG → Evidence Pack → Composer` 基线没有被修改。
术语能力只接入新的 `src/debug_agent_system/kg_raw_codex/` 旁路。

## 9. 与写侧的结合

写侧在“候选已审批、图合并成功、文档关系刷新成功”之后自动重建术语层：

```text
审核通过
  → merge_graph
  → refresh_document_links
  → write_terminology_layer
  → 记录 terminology_version / revision / counts
  → 合图事务完成后，独立重建 pending/needs_re_review 术语候选
  → 后续 SAG 或新读侧使用同一版概念层
```

未批准、schema 不合法或合图失败时不会触发术语刷新。术语候选生成是合图后的独立、
可重跑治理任务，不放进 W5 原子事务，避免第二次全图扫描拖慢提交或扩大回滚边界。
术语层刷新和候选生成都不会自动批准候选；候选审核仍通过独立 review queue 完成。
写侧返回值和审计记录包含 `terminology_refresh`，便于定位“KG 已更新但术语层未更新”
的版本漂移。

## 10. 构建、检查和数据文件

构建：

```bash
make kg-v2-terminology-build
```

只检查当前产物是否与 KG 和人工术语表一致：

```bash
make kg-v2-terminology-check
```

主要实现与产物：

- `src/debug_agent_system/knowledge_v2/terminology.py`
- `src/debug_agent_system/knowledge_v2/entity_terminology.py`
- `src/debug_agent_system/knowledge_v2/terminology_review.py`
- `src/debug_agent_system/knowledge_v2/noun_discovery.py`
- `scripts/build_debug_terminology.py`
- `scripts/manage_debug_terminology_review.py`
- `data/kg_v2/objects/debug_concepts.json`
- `data/kg_v2/objects/term_expressions.json`
- `data/kg_v2/objects/term_senses.json`
- `data/kg_v2/terminology/entity_ontology.json`
- `data/kg_v2/terminology/noun_discovery_config.json`
- `data/kg_v2/terminology/noun_discovery_report.json`
- `data/kg_v2/terminology/noun_discovery_report.md`
- `data/kg_v2/terminology/noun_terminology_inventory.json`
- `data/kg_v2/terminology/noun_terminology_inventory.md`
- `data/kg_v2/terminology/curated_terms.json`
- `data/kg_v2/terminology/terminology_manifest.json`
- `data/kg_v2/review_queue/terminology_candidates.json`
- `data/kg_v2/review_queue/noun_discovery_candidates.json`

## 11. 当前边界与后续治理

当前版本已建立可运行的名词实体图、复合字段原子化、关系型检索扩展、上下文评分与
保守弃权、歧义审计、候选审核队列、读侧 Tool 和写侧自动刷新，但没有把历史关键词
或语料发现出的 127 个新概念、58 个变体、82 条结构关系候选和 97 条记录级关联候选
自动升级为正式图事实。这是有意的安全边界：语料能证明“这个名字被使用过”或
“两个名字经常一起出现”，不能单独证明“两个名字必然等价”或“两个实体必然存在某种
结构关系”。

后续应按以下顺序治理：

1. 按 Query 失败日志、语料统计和当前 `search_hint` 生成候选，不直接生效；
2. 由领域人员确认候选概念、关系类型、设备、子系统、阶段和排除条件；
3. 通过审核导入器把故障别名写入 `curated_terms.json`，把名词概念、已批准别名和
   实体关系写入 `entity_ontology.json`；
4. 重建术语层并跑歧义、召回、安全和 benchmark 回归；
5. 对长期未使用、冲突或过时表达标记 deprecated，而不是物理删除历史。

衡量指标应同时包含 term resolution coverage、歧义正确暴露率、Variant 误锁率、召回
提升、unsupported claim 和人工术语审核成本，不能只看 Top1 命中率。
