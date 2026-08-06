# B-D Topology Traversal Agent

- id: `B-D`
- type: Orchestrator
- owner: `src/debug_agent_system/agents/read/bd_traversal`
- responsibility: choose and advance diagnostic checks over a locked subgraph; expose compact current-step output for interactive mode while preserving branch context in metadata.
- entrypoints:
  - `first_step(state, subgraph, skip_check_ids=None)`.
  - `select_check(state, subgraph, check_id, reason)`.
  - `after_user_result(state, subgraph, user_message)`.
- inputs:
  - `SessionState` from MEM/O0.
  - `LockedSubgraph` with `checks`, `solutions_by_check`, `next_edges_by_check`.
  - Optional confirmed `skip_check_ids`; only confirmed skips belong here.
  - User result text for step continuation.
- outputs:
  - `TraversalDecision(status, check, solution, reason)` where status is `step|resolved|escalate`.
  - Mutates session fields through MEM-owned object: `current_check_id`, `current_check`, `current_index`, `checks_presented`, `check_results`, `ruled_out`, `which_check_solved`, `metadata.presented_check_ids`, `metadata.traversal`.
- failure_modes:
  - User says solved -> `resolved` with first solution for current check.
  - User says not solved/unclear -> current check is ruled out and traversal advances.
  - No remaining checks -> `escalate/no_more_checks`.
  - Unknown selected branch check -> `escalate/unknown_branch_check`.
- observability:
  - `metadata.traversal.reason`, `ordered_check_ids`, `current_check_id`, `presented_check_ids`.
  - O0 branch gate adds `branch_options` and `branch_trace`.
- interactive_policy:
  - interactive answer renders only current check.
  - branch candidates remain in metadata for agent/UI consumption.
  - non-interactive can render a compact checklist.
- non_goals:
  - Does not decide sufficiency.
  - Does not own branch ambiguity ask-info; O0 branch gate does.
  - Does not execute field actions.


## 开放假设与闭环归因

B/D 不能因为当前选择了一个最优 check，就把其他仍可能成立的分支从诊断状态里删除。交互模式下用户只看到当前一步，但 B/D 会在 metadata 中保留：

- `metadata.open_hypothesis_check_ids`
- `metadata.traversal.open_hypothesis_check_ids`

这些 id 表示“暂未排除的候选前沿”，用于两件事：

1. UI/上层 agent 可以知道还有哪些可能方向没有被排除；
2. 用户后续反馈“实际是 X / 更换 X 后已解决”时，B/D 可以把 `which_check_solved` 归因到匹配的开放候选，而不是机械归因到当前 check。

例如“开机页面循环 + 既往自动修复蓝屏 + 0xc0000001”会先进入启动/BCD/系统盘当前步骤；但内存稳定性 check 仍保留在开放假设中。如果现场最终反馈“内存条故障，更换内存后解决”，B/D 应归因到 `check:industrial-pc-blue-screen-step3b1`。这不是把内存写死到 O0 路由，而是 B/D 对真实反馈的归因能力。
