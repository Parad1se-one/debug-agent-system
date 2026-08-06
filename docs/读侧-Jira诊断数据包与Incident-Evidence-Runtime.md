# 读侧 Jira 诊断数据包与 Incident Evidence Runtime

> 状态：已实现，默认关闭；更新时间：2026-08-04。
> 目标：把 Jira 问题描述、日志包、调用栈、环境信息和 KG_v2 知识组织成可审计的案件证据，而不是把整包文本直接交给模型猜根因。

## 1. 适用范围与边界

该运行时处理通过 Jira、工单或对话提交的诊断材料：日志、ZIP/TAR、DMP、EVTX、JSON、CSV、图片和补充说明。它是一条接在现有读侧旁边的案件证据路径，不替换 KG_v2 原生诊断状态机。

它保证：

- 所有事实保留 artifact、行号/字节位置、hash 和解析方式；
- 压缩包有成员数、大小、嵌套、压缩比和路径穿越限制；
- 原始事件、调用栈、环境、KG 候选、假设和测试分层保存；
- 检测位置不自动等同于根因，Jira 状态不自动等同于已验证修复；
- 相似案例只作弱证据，不能冒充正式 KG 知识；
- 只读分析不执行附件，不写 canonical KG_v2；
- 默认关闭，开启后也可先以 shadow mode 旁路观察；
- Query 中的参考时间先被规范成独立时间点和有界窗口，再决定读取哪些日志；窗口外材料不会被静默解释成案件事实；
- “相同签名再次出现”“受控复现成功”“修复后验证通过”是三个不同结论，工具和报告分别标记。

它暂不保证：

- 自动访问远端 Jira。当前只解析输入中的 Jira key，并读取本地离线快照；
- 任意 EVTX 都能在没有可选依赖时解析。安装 `incident` extra 后可结构化读取；缺少依赖、文件损坏或格式不支持时会明确记录 exclusion；
- DMP 的符号化调用栈、OCR、符号服务器、驱动调试或附件脚本执行。DMP 默认只读解析标准 Minidump 流；符号化深析仍需受控调试器与符号源，附件始终不执行；
- KG 中没有的根因由模型补写。证据不足时输出待定位假设和下一步采集项；
- 第一次读取普通 ZIP 时对入选的压缩日志做真正的顺序解压；当前还没有持久化稀疏时间索引，因此它避免的是整包物化和全量事件解析，不承诺对压缩流随机跳转。重复分析可在后续版本用 sidecar 时间索引继续降时延。

## 2. 总体流程

```text
Jira 描述 / Query / evidence_resources / log_summary
                         │
                         ▼
       Incident Scope：参考时间、年份、窗口、时区语义
                         │
                         ▼
       压缩包中央目录枚举 + 候选成员日期预筛
                         │
                         ▼
       Artifact Intake + 安全清单、流式 hash/窗口抽取
                         │
          ┌──────────────┼────────────────┐
          ▼              ▼                ▼
       文本日志      ZIP/TAR 成员     EVTX/DMP/图片
                                      结构化只读解析
          │              │                │
          └────── Diagnostic Parser Registry ──────┘
                         │
        Event / Stack / Environment + Correlations
                         │
              Timeline + Incident Case Graph
                         │
                         ▼
           稳定错误码/组件/函数 → KG_v2 宽召回
                         │
               候选与案件证据相交校验
                         │
                         ▼
     Hypothesis Matrix + Next Best Tests + Exclusions
                         │
                         ▼
                Incident Evidence Pack v3
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      确定性诊断报告          Codex strict tools 调查
             └───────────┬───────────┘
                         ▼
                  本地 verifier
```

## 3. 数据契约

### 3.1 IncidentCase 与 ArtifactManifest

`IncidentCase` 保存 Query、Jira key、设备/站点/版本/复现信息、用户提交资源和日志摘要。`ArtifactManifest` 对每个根资源和压缩包成员记录：

- `artifact_id`、文件名、kind、MIME、大小、SHA-256；
- 父 artifact、archive member、嵌套层级；
- `available/rejected/metadata_only` 状态；
- safety flags、来源和调用方 metadata。

不安全路径、链接、超限成员或异常压缩比不会静默消失，而会形成 rejected manifest 和 exclusion。
根压缩包的 SHA-256 采用流式计算，不再把数百 MB 文件一次性读入内存。存在 Query 时间范围时，成员清单仍完整保留，但不相关成员记录为 `scope_skipped`；入选日志生成只包含命中窗口的来源绑定副本，并保存原文件行号、CRC、原大小和命中窗口。

### 3.2 规范化诊断对象

解析器输出四类对象：

| 对象 | 内容 |
|---|---|
| `DiagnosticEvent` | 时间、级别、错误码、组件、模块、函数、原始消息与 evidence IDs |
| `StackTrace` / `StackFrame` | 线程、帧序、模块、函数、地址、源码位置与 evidence IDs |
| `EnvironmentSnapshot` | 产品/系统/GPU/驱动/CUDA/OpenCV/业务库版本及来源 |
| `EvidenceLink` | artifact 到行号、字节范围、解析器和摘要的统一来源索引 |

`IncidentScope` 保存 Query 中提取的参考时间点、窗口、年份推断和时区语义。多个参考时间默认解释为多个独立观察点，不擅自把“8 月 1 日 21:30、8 月 3 日 06:04”扩成两天的连续范围。默认每个时间点读取前 120 秒、后 180 秒。

`correlations` 保存异常与后续进程启动的时间闭环、EVTX 驱动复位/LiveKernelEvent 与应用异常的时间邻接，以及跨日期重复签名。相关性只提高“确实发生过闪退、驱动复位、重启或复发”的可信度，不直接宣布根因。

调用栈地址、构建目录、commit/hash 等易变字段只用于审计，不作为 KG 必须命中的 Query facet。

### 3.3 Incident Evidence Pack v3

`debug_agent_system.incident_evidence_pack.v3` 包含：

- case 与 artifact manifest；
- diagnostic events、stack traces、environment；
- query scope、timeline、correlations 与 case graph；
- KG retrieval 的 anchors、候选、Chunk、路径和 trace；
- hypothesis matrix 与 next-best tests；
- source index、exclusions 和 claim policy。

本地 verifier 要求假设的支持/反证 evidence ID 必须在 source index 闭包内，并检查“检测点不等于根因”等关键边界。

## 4. 解析器策略

| 输入 | 默认行为 | 可选增强 | 失败行为 |
|---|---|---|---|
| LOG/TXT/JSON/CSV | 有界读取、事件/版本/栈抽取；保留原始行号 | Query 有时间时只事件化命中窗口 | 保留解析 exclusion |
| ZIP | 安全枚举；按成员日期预筛；入选日志流式解压并抽取时间窗口 | 后续可增加持久化稀疏时间索引 | 超限/危险成员形成 rejected manifest；窗口外成员记 `scope_skipped` |
| TAR | 安全枚举、逐成员有界解析 | 无 | 超限/危险成员形成 rejected manifest |
| EVTX | 保留 header、hash 和来源；安装 `incident` extra 后结构化读取 Provider、EventID、级别、UTC 时间和 EventData/UserData | 按 Query 时间窗校准到设备本地时间并只事件化命中记录 | 明确 `evtx_parser_unavailable/failed` |
| DMP | 无外部依赖地解析标准 Minidump 头、异常、进程/线程、系统版本、模块及版本；不执行转储内容 | 受控环境中用 `cdb`/`windbg` 和固定符号源补充符号化栈 | 格式不支持时明确 `minidump_parse_failed`；无符号时明确边界，不把模块地址归属当根因 |
| 图片 | header、尺寸/MIME | 显式开启且有 tesseract 时 OCR | 明确 `ocr_disabled/unavailable` |
| 附件脚本 | 只记录、只读取 | 无 | 永不执行 |

日志解析不会把所有包含 `failed`、`timeout` 或数字的文本都当成故障。当前通用降噪包括：忽略 INFO/DEBUG 中的普通资产名和坐标数字、排除无时间戳配置项、把栈帧挂到调用栈而非重复生成事件，并单独识别异常头、进程启动 PID 和生命周期信号。该策略按日志结构和证据强度工作，不绑定某个 Query 或错误码。

### 4.1 参考时间快路径

对于“参考时间：8月1日21：30，8月3日6：04”一类输入，运行顺序是：

1. 从 Query 解析时间点；缺年份时只能从诊断包名或资源日期中推断，并记录 warning；
2. 先读 ZIP 中央目录，通过文件名日期筛掉明显无关的日志；
3. 对入选的压缩日志顺序解压，只保留落入参考窗口的行和必要上下文；
4. 只对窗口副本做事件、调用栈、环境和生命周期解析；
5. 用原成员路径、原行号、CRC 和根包 hash 回指来源。

因此它避免了整包落盘、窗口外事件化和无关日志进入模型上下文。由于 deflate 压缩流不能可靠随机跳到任意时间，首次分析仍可能顺序解压目标日期日志；若同一包需要多次变更时间窗口，下一步应建立按成员 hash 缓存的稀疏时间 sidecar，而不是再增加 Query 定向规则。

### 4.2 Windows 二进制证据解析

EVTX 和 DMP 的文件名经常不包含日期，DMP 还常用 UUID 命名，不能沿用普通日志的文件名日期预筛。当前实现将它们视为“内部带时间戳的高价值二进制证据”：在安全上限内保留并解析内部记录，再按案件时间窗选择证据。

EVTX 的处理步骤为：

1. 用 `python-evtx` 只读遍历记录并规范化 Provider、EventID、级别、UTC 时间、进程/线程和事件字段；
2. 优先读取 Windows `CurrentBias` 推断本地偏移；缺少该事件时，使用 Query 参考窗口与高信号事件做通用偏移匹配，并把推断方式和得分留在元数据中；
3. 只把命中窗口的驱动异常、Display 复位、WER/LiveKernelEvent 等记录转成 `DiagnosticEvent`；
4. 原始 EVTX、记录号和 XML 摘要继续进入来源索引。

DMP 的默认解析不依赖本机调试器，读取 Minidump 的 MiscInfo、SystemInfo、ModuleList、ThreadList 和 Exception 流，得到创建时间、进程/线程、异常码、异常地址归属模块、Windows 版本和已加载模块版本。该结果可用于回答“哪个进程退出、何时退出、当时加载了什么版本”，但没有符号化栈时不能根据“地址落在某模块范围”直接断言该模块是根因。

安装 EVTX 可选依赖：

```bash
python -m pip install -e '.[incident]'
```

## 5. KG 检索与综合判断

检索桥从错误码、异常类型、组件、模块和函数构造稳定锚点，同时剔除构建路径、地址和长 hex 标识。KG_v2 仍执行宽召回，所有候选原样留在 Evidence Pack 供审计。

正式假设更严格：只有候选文本与案件稳定锚点相交，并且能回指案件 evidence ID，才进入 hypothesis matrix。若没有 KG 候选满足条件，但日志、EVTX 和 DMP 在同一时间窗内形成一致的结构化故障链，运行时可以生成“证据融合假设”，明确它来自案件证据而非 KG 根因；该假设最多收敛到证据支持的故障域，不越级区分驱动软件缺陷、GPU 硬件、电源或 PCIe 稳定性。若连跨源闭环也不足，则仍生成低置信度的“案件错误签名待定位”占位假设。

每个假设包含：

- 支持证据与反证；
- 缺失证据；
- 置信度与状态；
- KG source IDs（若有）；
- 可区分候选的下一步最小测试。

这避免“召回到了某条知识”被错误解释为“已经定位根因”。

## 6. Codex 只读工具

现有 Codex Harness 除 KG 原生工具外增加：

- `parse_incident_scope`：在建立案件前预览 Query 的参考时间窗口；
- `analyze_incident` / `index_log_package`：建立同一个案件与 Evidence Pack；
- `get_incident_scope`：读取本次案件实际采用的时间范围、年份推断和 warning；
- `get_incident_evidence_pack`：读取 v3 证据包；
- `search_diagnostic_events`：按错误码、组件、函数或消息检索规范事件；
- `search_diagnostic_events_by_time`：只检索指定窗口内的规范事件；
- `extract_log_time_windows`：按案件 scope 读取来源绑定的日志窗口；
- `inspect_stacktrace`：读取规范调用栈；
- `read_log_window`：读取来源绑定的上下文窗口；
- `inspect_evtx`：返回命中案件窗口的 Provider、EventID、驱动复位/LiveKernelEvent 及时间对齐元数据，或明确 exclusion；
- `inspect_dump`：返回进程、线程、异常、系统和模块版本；若无符号化栈，明确标记其不能独立证明根因；
- `query_kg_hypotheses`：查看稳定锚点、KG 候选及其支持/反证/缺口矩阵；
- `retrieve_similar_cases`：只读返回弱案例证据；
- `build_incident_timeline` / `propose_next_tests`：读取确定性时间线和下一步测试；
- `plan_reproduction`：根据现有证据生成受控复现计划，但不执行设备动作、脚本或故障注入；
- `compare_reproduction_runs`：比较两个不可变运行结果中的故障签名、环境和时间闭环；
- `render_incident_report`：提交规范报告并触发本地校验。

Codex 负责调查顺序和解释组织，不得绕过来源闭包、安全边界或自行宣布已验证修复。

Read Runtime v3 中，这些能力通过 `read_runtime_v3.providers.IncidentProvider` 以及统一
`ReadToolRegistry` 接入；Incident Evidence Pack 被归一为 EvidenceRecord，不再以“覆盖原回答”
的方式变成另一套事实源。当前 v3 仍为 shadow，Incident Runtime 本身也保留独立
开关和回滚入口。统一方案见
[读侧 Read Runtime v3 设计原理与取舍](读侧-Read-Runtime-v3设计原理与取舍.md)。

### 6.1 “复现”在本运行时中的含义

- **观察到复发**：历史包中相同稳定签名在不同日期出现。它证明问题重复发生，但现场变量没有被控制。
- **受控复现**：固定版本、配置、硬件、输入和负载，仅改变一个待验证因素；同时保存操作时间、基线和候选运行的 Evidence Pack。`plan_reproduction` 只产出这个实验协议。
- **修复验证**：在同等条件和足够循环/观察时长下，候选运行不再出现目标签名，并满足业务成功信号和人工确认。`compare_reproduction_runs` 只给差异，不会因一次“未出现”自动判定已修复。

如果以后需要自动执行复现，必须另建受权限控制的执行 Tool，显式定义设备范围、动作白名单、回滚、急停和人工批准。它不能借用当前只读 Harness，也不能执行诊断包中的任意脚本。

## 7. 启用、CLI 与回滚

默认配置：

```yaml
incident_runtime:
  enabled: false
  shadow_mode: true
  allow_dump_analysis: false
  allow_ocr_analysis: false
```

独立运行：

```bash
PYTHONPATH=src python -m debug_agent_system.adapters.cli \
  --config config/debug_agent_system.yaml \
  analyze-incident "Jira 问题描述" \
  --evidence-resource /path/to/package.zip \
  --out /tmp/incident-result.json
```

集成运行时先以 `enabled=true, shadow_mode=true` 记录 `metadata.incident_runtime`，但继续返回原 KG 答案。验证稳定后改为 `shadow_mode=false`，诊断数据包类请求才使用 Incident report 作为回答。任何时候关闭 `enabled` 即恢复冻结基线。

## 8. 验收方式

结构回归集不计入 Formal Debug Benchmark 100 题准确率，也不读取 held-out Gold：

```bash
PYTHONPATH=src python scripts/run_incident_package_benchmark.py
PYTHONPATH=src pytest -q \
  tests/unit/test_windows_binary_evidence_parsers.py \
  tests/unit/test_incident_evidence_runtime.py \
  tests/integration/test_incident_runtime_integration.py
```

当前公开样例验证：CUDA `-217 illegal memory access` 能抽取事件、SYMV 版本和调用栈；地址、构建路径和 commit 不进入 KG retrieval query；无直接支持的 KG 候选不进入正式根因假设；Evidence Pack 来源闭包与报告 verifier 通过。

2026-08-04 使用客户03 AOI-0014 真实诊断包复测：根 ZIP 约 500 MB，Query 的两个参考时间形成两个独立窗口。先完成普通日志窗口化，再加入 EVTX 与 DMP 结构化解析，结果如下：

| 指标 | 仅文本窗口化 | 加入 EVTX + DMP |
|---|---:|---:|
| 规范事件 | 112 | 137 |
| `illegal_memory_access` | 4 | 4 |
| GPU 驱动图形异常 | 0 | 8 |
| Display 驱动复位 | 0 | 2 |
| LiveKernelEvent/WATCHDOG | 0 | 4 |
| DMP 应用异常 | 0 | 2 |
| correlations | 4 | 10 |
| 环境 | SYMV 1.1.0 | Windows 10.0.19044、SYMV 1.1.0.0、NVIDIA 模块版本 32.0.15.6070、`cudart64_12.dll` 文件版本 6.14.11.12080 |
| 主假设 | 错误签名待定位，`needs_evidence`，0.25 | GPU/显示驱动执行链异常触发 CUDA 失败与应用退出，`supported`，0.74 |

两个参考窗口都形成同构证据链：

- 2026-08-01 21:29:59，`nvlddmkm` Event 13 报告 `Graphics Exception: MISSING_INLINE_DATA` 和 `ESR 0x404600=0x80000002`，随后 Event 153 报告 `GPUID: 100`；21:30:00 出现 Display Event 4101 驱动复位；21:30:02 出现 `LiveKernelEvent 141`，指向 `WATCHDOG-20260801-2129.dmp`。
- 2026-08-03 06:03:39 至 06:03:43，再次出现相同的 Event 13、Event 153、Display 4101 和 `LiveKernelEvent 141`，指向 `WATCHDOG-20260803-0603.dmp`。
- 包内两个应用 DMP 分别创建于本地 2026-08-01 21:30:00 和 2026-08-03 06:03:40，均记录 `smt-aoi.exe` 的 `0x40000015 fatal_app_exit`，异常地址和模块偏移相同；二者都加载 SYMV、NVIDIA CUDA driver 和 CUDA runtime 模块。

因此诊断结论有更新：现在可以把近端故障域从笼统的“CUDA -217 待定位”收敛到“GPU/显示驱动执行链发生异常和 TDR/复位，继而出现 CUDA illegal memory access 与应用退出”。但还不能写成“NVIDIA 驱动软件就是唯一根因”：同样的 Windows 证据也可能由 GPU 硬件、电源或 PCIe 不稳定触发。诊断包只引用了两个 Windows WATCHDOG 内核转储路径，实际包含的是应用 DMP；仍需取得并符号化 WATCHDOG DMP，以及做驱动版本与硬件/供电/插槽的单变量对照，才能继续区分。

窗口化版本峰值 RSS 约 170 MB，旧全量路径约 909 MB；报告从约 97k 字符收敛到约 14k 字符。当前仍需顺序解压入选日志，所以这是内存和证据范围优化，不应误写成 ZIP 内随机访问。

相关结构与 Harness 回归：

```bash
PYTHONPATH=src pytest -q \
  tests/unit/test_windows_binary_evidence_parsers.py \
  tests/unit/test_incident_evidence_runtime.py \
  tests/integration/test_incident_runtime_integration.py
# 19 passed
```

## 9. 分阶段上线计划

| 阶段 | 工作 | 验收门 | 回滚 |
|---|---|---|---|
| P0 契约冻结 | 冻结 Artifact/Event/Stack/Environment/Evidence Pack schema | schema、来源闭包和安全测试通过 | 不接主链 |
| P1 离线解析 | 用公开及脱敏 Jira 数据包验证日志、压缩包、EVTX 结构化事件和 DMP 标准流 | 不静默丢成员；所有失败有 exclusion；无符号结果不越级归因 | 仅保留 CLI |
| P2 Shadow | `enabled=true, shadow_mode=true` 接入真实入口 | 原回答不变；metadata 可审计；时延/失败率可接受 | `enabled=false` |
| P3 Codex 调查 | 开放案件级 strict Tool，验证多轮检索与规范收口 | 必须调用 `render_incident_report`；自由文本不能越过 verifier | 关闭 read LLM/Harness |
| P4 Active | 对明确的 Jira/诊断包请求返回 Incident report | 人工评审定位边界、下一步测试与来源完整性 | `shadow_mode=true` |
| P5 闭环治理 | 将人工确认的修复结果送入既有写侧审核，不由读侧直接入 KG | 审核、版本、证据和回归门全部通过 | 停止写侧提交 |

优先补充的样本族为：GPU/CUDA 异常、相机/采集失败、网络掉线、Windows 崩溃和性能退化。
扩样按“输入格式 × 故障域 × 是否有调用栈 × 是否有环境矩阵”分层，不围绕单一错误码增加
专用规则。
