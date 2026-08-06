# C Sufficiency Gate Agent

- id: `C`
- type: Verifier
- owner: `src/debug_agent_system/agents/read/c_sufficiency`
- responsibility: decide whether the query plus KG candidates are sufficient to enter diagnosis; if not, return concrete missing information instead of hard-answering.
- entrypoint: `SufficiencyGate.decide(query, candidates, subgraph=None)`.
- inputs:
  - `query: str` current user query, including user supplements for ask-info retries.
  - `candidates: list[Candidate]` from O-KG.
  - Optional `subgraph: LockedSubgraph` to reuse `required_info` on low graph score.
  - Config thresholds: `graph_match_min_score`, `max_required_items`.
- outputs:
  - `SufficiencyDecision`: `sufficient: bool`, `required_info: list[str]`, `reason`, `confidence`.
  - Reasons include `empty_query`, `explicit_missing_required_info`, `missing_fault_context`, `no_graph_match`, `low_graph_score`, `sufficient`.
- failure_modes:
  - Empty query -> ask for fault symptom/error text.
  - Generic query -> ask for symptom, version, logs.
  - No candidates -> ask for screenshot/error text, version, logs.
  - Low score -> ask subgraph-specific `required_info` if available.
- observability:
  - O0 stores decision in `metadata.sufficiency` and response confidence.
- non_goals:
  - Does not select `next.condition` branches; O0 branch gate owns `branch_condition_missing`.
  - Does not call tools/LLM.
  - Does not mutate session except through O0.

## 与条件树追问的边界

C 只回答一个问题：当前 query 是否足够进入 KG 诊断。它不负责在同一个 Error 子图内选择 `next.condition` 分支。

例如“客户开机设备一直停留开机页面，一会显示一会不显示；之前有蓝屏；已做过断电放电”这个 query：

1. O-KG 因“蓝屏”信号命中 `err:industrial-pc-blue-screen`；
2. C 判断召回和基础诊断信息足够，因此返回 sufficient；
3. O0 branch gate 在 `check:industrial-pc-blue-screen-step2` 发现固定码、随机码、重装后蓝屏、启动/自动修复固定码等多个条件分支无法唯一判定；
4. O0 返回 `ask_info`，追问页面类型、BIOS/系统盘、固定错误码/dump。

所以本例的 `ask_info` 不是 C 的 `missing_info`，而是条件树分支的 `branch_condition_missing`。
