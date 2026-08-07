# Architecture

This document describes the runtime architecture of `debug-agent-system`. The
project is a **deterministic, knowledge-graph-driven multi-agent system** for
AOI equipment fault diagnosis, organized as a **read side** (diagnosis) and a
**write side** (knowledge governance) closed loop.

## System overview

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

## Read side (diagnosis)

The entry point is `DebugAgentSystem` (`src/debug_agent_system/runtime/system.py`),
an **O0 supervisor** that owns the deterministic ordering of a diagnosis session.
Public surface: `start` / `step` / `diagnose` / `analyze_incident`.

| Agent | Responsibility | Deterministic? |
|---|---|---|
| `MEM` | Session persistence (the single session store) | yes |
| `C` | Sufficiency gate — decide `ask_info` vs proceed | yes |
| `O-LOG` | Log / branch-hint analysis | yes |
| `O-KG` | KG_v2 retrieval & ranking (blue-screen / software-crash re-rank) | yes |
| `A` | Lock a candidate error's causal subgraph | yes |
| `B-D` | Topology traversal — first step, next check, branch on user result | yes |
| `O-GEN` | Render the answer (checks / conclusion / escalation) | yes |
| `EA` | Evidence verifier — align resolution against `EvidenceItem` | yes |
| `O-ESC` | Escalate to the right owner with an evidence pack | yes |
| `O-EvidenceGap` | Resolve missing evidence (ask or accept read-only resources) | yes |

Key properties:

- **No LLM in the core loop.** The whole read-side pipeline is deterministic —
  retrieval, locking, planning, branch execution, and verification run on
  `sqlite3` + the JSON KG, so behavior is reproducible and testable offline.
- **Session state** is persisted and resumed via `step(session_id, feedback)`.
- **Evidence resources** can be attached at any turn (`evidence_resources`),
  subject to a size/type gate.
- **Safety**: destructive/high-cost actions require human confirmation;
  `resolved` must be backed by a `verified_fix` outcome **and** a valid
  `EvidenceItem`.

Optional LLM-backed paths exist as **opt-in harnesses only** (Codex / DeepSeek
read harnesses in `adapters/`), and never write to the KG or decide branches.

## Write side (knowledge governance)

All knowledge enters the graph through a gated, versioned pipeline:

```
W1 collect → W2 extract → W3 conflict → W4 quality gate → W5 incremental ingest
     → W6 review queue (human) → approved-only apply → KG_v2
W9 raw-doc ingest → W10 section/case bundling (SOP atomicity)
```

- **Typed/readiness checks** normalize each input (group chat, text history,
  non-SOP/SOP documents, Jira + attachments, expert corrections, diagnostic
  feedback, log patterns) before it can enter the pipeline.
- **`approved-only apply`**: write side never mutates the active graph
  automatically — it produces candidates into a review queue; only explicitly
  approved items are applied (idempotent, replayable, snapshot-audited).
- **Terminology governance**: noun concepts, expressions, and senses are
  versioned; query expansion and alias resolution use structured context
  policies with deterministic verification.

## Knowledge graph (KG_v2)

- **Schema**: 19 entity types (`FaultFamily`, `FaultVariant`, `DiagnosticAction`,
  `ActionOutcome`, `RequiredInfoSpec`, `DiagnosticTrace`, `TraceStep`,
  `BranchRule`, `DecisionPolicy`, `EvidenceItem`, `SourceCase`, terminology
  entities, …). Validated against `data/kg_v2/schema/object-types.json`
  (required fields, enums, `max_length`).
- **Execution view**: a materialized projection
  (`branches / checks / errors / observations / outcomes / policies /
  solutions / trace_steps / traces` + canonical edges) used by the runtime.
- **Serving index**: a SQLite SAG index (`SqliteSAGV2`) rebuilt deterministically
  from the JSON store when the KG revision changes.
- **Current snapshot** (2026-08-04): 57 families · 162 variants · 585 actions ·
  524 evidence items · 98 traces · 131 branch rules · 1,548 materialized edges.

## Evaluation

Two offline regression suites run **without any model**:

- **11-case precision set** (`industrial_pc_boot_v1.json`) — industrial-PC boot
  faults.
- **150-case broad set** (`broad_debug_v1.json`) — cross-domain Debug faults.

Metrics (`eval/debug_sim/scorer.py`): fault-localization accuracy, check-chain
recall, evidence recall, required-info accuracy, unsafe-action rate,
terminal-ok rate, and a gated composite. `gate.py` enforces regression
thresholds (composite drop, recall drop, target drop, flip limit).

Historical internal results are listed in the README; the shipped graph is a
sanitized subset, so exact scores are snapshot-dependent.

## Runtime & data flow (typical `diagnose`)

1. `start(query)` → session created; sufficiency gate decides.
2. If sufficient → `O-KG` retrieves candidate family/variant → `A` locks a
   subgraph → `B-D` produces the first check from a trace + branch rules.
3. `O-GEN` renders the answer; `EA` verifies any claimed resolution against
   evidence; unresolved or unsafe paths are escalated.
4. `step(session_id, feedback)` continues the trace, applies branch rules, and
   re-verifies.

## Repository map

```text
src/debug_agent_system/
  core/                 dataclass contracts, config, observability, paths
  knowledge/            legacy KG store (compat; not consumed by runtime)
  knowledge_v2/         KG_v2 store, SQLite SAG, terminology, validator
  agents/read/          O0..O-ESC read-side agents + evidence pack/answer
  agents/write/         W1–W10 write-side agents + pipeline
  agents/tools/         read-only tool registry & executors (attachment, dmp,
                        evtx, jira, log package, image, proj, context)
  runtime/system.py     O0 supervisor / public API
  eval/                 debug_sim runner + scorer + gates; read/write benchmarks
  adapters/             CLI & QA adapters, Codex/DeepSeek read harnesses
  incident_runtime/     structured Jira/diagnostic-package evidence parsing
```

## Design notes

- **Determinism first**: the core must be reproducible offline; LLM features are
  add-ons, never the control path.
- **Evidence is subordinate**: every claim can be traced to `EvidenceItem` /
  `SourceCase`; `resolved` requires both a verified outcome and evidence.
- **Knowledge is gated**: human approval is the boundary between candidate and
  active graph; nothing auto-writes to production knowledge.
- **Eval is a release gate**: the scorer + gate protect against regressions
  across KG revisions and code changes.
