# debug-agent-system

**A knowledge-graph-driven multi-agent system for AOI equipment fault diagnosis, with a closed training/evaluation loop.**

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Parad1se-one/debug-agent-system/actions/workflows/ci.yml/badge.svg)
[![中文](https://img.shields.io/badge/README-中文-blue)](README.zh-CN.md)

`debug-agent-system` reconstructs an AOI (Automated Optical Inspection) field-service diagnosis workflow — where engineers locate faults from group-chat records, logs, and historical documents, and turn them into standardized checklists — as a standalone, deterministic multi-agent Python package.

The system ingests fault descriptions, evidence packages (logs, dumps, EVTX), and occurrence time, then produces **step-by-step troubleshooting plans with suggested ordering**. It plans dynamically: it parses logs and queries a knowledge graph, adapts later steps based on check results, asks for missing information, and escalates to the right owner when knowledge is insufficient.

> 中文版说明见 [`README.zh-CN.md`](README.zh-CN.md)。

---

## Highlights

- **Deterministic, KG-native runtime** — `DebugAgentSystem` exposes a small public API: `start` / `step` / `diagnose` / `analyze_incident`, covering sufficiency judgment, diagnostic planning, branch execution, evidence verification, and multi-turn session state — **no LLM required** for the core loop.
- **Execution-graph knowledge governance** — the error/check/solution graph is upgraded to an *execution graph*: **57 fault families · 162 variants · 585 diagnostic actions · 524 evidence items · 98 traces · 131 branch rules · 1,548 materialized edges**. All writes (tickets, group chats, Jira, documents) pass through a typed quality gate, a human review queue, and `approved-only apply`.
- **Offline regression & safety gates** — a **11-case precision set** and a **150-case broad set** measure fault-localization accuracy, check-chain recall, evidence recall, and unsafe-action rate (which is **0%**).
- **Read / write closed loop** — the read side consumes the KG for diagnosis; the write side (W1–W10 agents) ingests group chats, docs, Jira, expert corrections, and diagnostic feedback back into the graph, versioned and gated.

> The full proprietary KG (built from internal field data) is **not distributed**. This repo ships the schema, a **sanitized graph subset**, and the public evaluation scenarios so the pipeline can be exercised end-to-end.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language / runtime | Python 3.11+ · stdlib + `sqlite3` · `markdown-it-py` |
| Knowledge graph | JSON object store + SQLite serving index (SAG) + schema-validated execution view |
| Orchestration | Deterministic O0 supervisor + 10 read-side agents (no LLM in core loop) |
| Data pipeline | W1–W10 write-side agents (chat / doc / Jira ingestion, quality gate, review queue) |
| Evaluation | Offline scenario runner + scorer + regression gates (no model needed) |
| Optional LLM paths | Codex / DeepSeek read harnesses (opt-in, keys from local env only) |
| Tests | `tests/run_tests.py` stdlib runner (offline, mocked, no network) |

---

## Public API

```python
from debug_agent_system import DebugAgentSystem

system = DebugAgentSystem.from_config("config/debug_agent_system.yaml")

# start a diagnostic session
first = system.start({"query": "主程序加载用户配置失败，user.cfg.toml异常"})

# feed back a check result and advance the plan
next_turn = system.step(first["session_id"], "已检查但仍未解决")

# attach read-only evidence when information is insufficient
with_log = system.start({
    "query": "初始化失败，请结合启动日志判断",
    "evidence_resources": [
        {"kind": "log_package", "name": "startup.log", "path": "/tmp/startup.log"}
    ],
})
```

Standard response schema:

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

## Demo (real output, this repo)

```bash
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli diagnose \
  "AOI主程序初始化失败，相机连接异常，请检查相机IP" --non-interactive
```

The runtime retrieves the fault variant, locks a subgraph, plans a trace, and
returns a deterministic first step with its evidence sources:

```text
status: step · confidence: 0.98 · family: 相机/光源类 · variant: 相机初始化失败
current_check: 确认弹窗报错为加载用户配置失败
next (human-confirm): 备份并清空 conf 目录
next: 检查日志与 user.cfg.toml 是否异常
next: 用最近一次正常诊断日志中的 user.cfg.toml 替换验证
sources: [evidence:9846bca31ef8, SOP 1.1.3.1.2, ...]
```

---

## Architecture

```mermaid
flowchart LR
    subgraph Read side (diagnosis)
        Q[Fault query + evidence] --> C{Sufficiency gate}
        C -- insufficient --> ASK[ask_info]
        C -- sufficient --> KG[KG_v2 retrieval / subgraph lock]
        KG --> P[Plan: trace + branch rules]
        P --> EX[Execute check/action]
        EX --> V{Evidence verifier}
        V -- verified --> RES[resolved + evidence pack]
        V -- not verified --> P
        P -- knowledge gap --> ESC[escalate owner]
    end

    subgraph Write side (knowledge governance)
        SRC[Chats / docs / Jira / tickets] --> W1[W1 collect]
        W1 --> W2[W2 extract]
        W2 --> W4[W4 quality gate]
        W4 --> W6[W6 review queue]
        W6 -- approved-only apply --> KG
    end

    KG --> Read
    Read -- diagnostic feedback / log patterns --> Write
```

Read-side agents (deterministic, O0 orchestrated): `MEM` session store · `C` sufficiency gate · `O-LOG` log analysis · `O-KG` KG retrieval · `A` subgraph lock · `B-D` topology traversal / branch execution · `O-GEN` answer generation · `EA` evidence verifier · `O-ESC` escalation · `O-EvidenceGap` evidence gap resolution.

Write-side agents: W1 chat collect → W2 extract → W3 conflict → W4 quality gate → W5 incremental ingest → W6 review queue → W7 trace assembly → W9 raw-doc ingest → W10 section/case bundling.

---

## Knowledge graph (KG_v2)

- Schema: 19 entity types (`FaultFamily`, `FaultVariant`, `DiagnosticAction`, `ActionOutcome`, `RequiredInfoSpec`, `DiagnosticTrace`, `TraceStep`, `BranchRule`, `DecisionPolicy`, `EvidenceItem`, `SourceCase`, terminology entities, …) — see `data/kg_v2/schema/`.
- Execution view: materialized `branches / checks / errors / observations / outcomes / policies / solutions / trace_steps / traces` with a canonical edge set.
- Terminology layer: noun concepts, expressions, senses, context policies for query expansion and alias resolution.

Current graph snapshot (2026-08-04):

| Entity | Count |
|---|---|
| Fault families | 57 |
| Fault variants | 162 |
| Diagnostic actions | 585 |
| Evidence items | 524 |
| Diagnostic traces | 98 |
| Branch rules | 131 |
| Materialized edges | 1,548 |

---

## Evaluation

Two offline regression suites (no model required) measure fault-localization accuracy, check-chain recall, evidence recall, and unsafe-action rate. Metrics are computed by `src/debug_agent_system/eval/debug_sim/scorer.py`.

**Historical internal snapshot results** (full internal KG; graph has grown since):

| Suite | Config | Date | Fault-loc. accuracy | Check-chain recall | Evidence recall | Unsafe-action rate | Composite |
|---|---|---|---|---|---|---|---|
| Precision set (11) | baseline | 2026-06-29 | 100% | 100% | 99.17% | 0% | 0.9979 |
| Precision set (11) | SAG | 2026-07-17 | 100% | 100% | 100% | 0% | 1.0000 |
| Broad set (150) | baseline | 2026-06-29 | 100% | 95.82% | 94.53% | 0% | 0.8593 |
| Broad set (150) | SAG | 2026-07-06 | 98.67% | 98.93% | 95.33% | 0% | 0.8641 |

> **Reproducibility note**: this repo ships the schema, a **sanitized graph subset**, and the public scenarios so the pipeline can be exercised end-to-end. Exact scores depend on the full internal graph snapshot and the code revision; the numbers above are the frozen internal baselines. Run the eval yourself with the commands below to observe the pipeline behavior on the shipped subset.

```bash
# run the 11-case precision set
PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
  --scenario-file data/eval/scenarios/industrial_pc_boot_v1.json --limit 11 --out-dir /tmp/eval

# run the 150-case broad set
PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
  --scenario-file data/eval/scenarios/broad_debug_v1.json --limit 150 --out-dir /tmp/eval
```

---

## Quickstart

Requirements: Python ≥ 3.11. Optional extras: `incident` (python-evtx), plus your own LLM keys for Codex/DeepSeek paths.

```bash
# install (core runtime only; no heavy deps)
pip install -e .

# run the test suite (stdlib runner; no network, no API keys required)
PYTHONPATH=src python3 tests/run_tests.py

# CLI smoke: diagnose a fault
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli diagnose \
  "AOI主程序初始化失败，相机连接异常，请检查相机IP"
```

See `docs/` for the read-side pipeline, KG design, write-side pipeline, and evaluation methodology.

---

## Repository layout

```text
config/                 system configs (store type, thresholds, paths)
data/kg_v2/             KG_v2 schema, sanitized graph subset, terminology
data/eval/scenarios/    public evaluation scenarios (11 + 150 cases)
docs/                   architecture, contracts, evaluation docs
src/debug_agent_system/
  core/                 dataclass contracts, config, observability
  knowledge_v2/         KG_v2 store, SQLite SAG index, terminology
  agents/read/          read-side diagnostic agents (O0..O-ESC)
  agents/write/         write-side knowledge-ingestion agents (W1-W10)
  agents/tools/         read-only tool registry & executors
  runtime/system.py     O0 supervisor / public API
  eval/                 debug_sim runner, scorers, gates, benchmarks
  adapters/             CLI & QA adapters, Codex/DeepSeek read harnesses
tests/                  stdlib test suite (offline, mocked)
```

## Notes on data & provenance

- The sanitized graph subset in `data/kg_v2/` preserves structure while removing internal identifiers, names, links, and paths; a rebuild of the SQLite serving index from these files is supported by `src/debug_agent_system/knowledge_v2/sqlite_sag_v2.py`.
- The raw proprietary corpus (field chat logs, tickets, internal docs) is intentionally **not** included; numbers quoted above come from internal snapshots.
- Optional LLM-backed paths (Codex / DeepSeek) read keys from local env files only; no key is committed.

## License

MIT — see [LICENSE](LICENSE).
