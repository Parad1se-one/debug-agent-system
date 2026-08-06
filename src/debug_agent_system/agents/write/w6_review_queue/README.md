# W6 Review Queue Agent

- id: `W6`
- type: Human-in-loop boundary
- owner: `src/debug_agent_system/agents/write/w6_review_queue`
- responsibility: build and persist idempotent human review items for candidates, merge candidates, noise, and ask-info candidates.
- entrypoints:
  - `build_review_item(queue, candidate, episode, conflict, quality_gate, dry_run_merge_plan=None)`.
  - `build_ask_info_review_item(required_info_candidate, episode, quality_gate, conflict=None)`.
  - `enqueue(name, item)`, `enqueue_many(name, items)`, `read_queue(name)`.
  - `mark_decision(name, item_id, action, reviewer="", note="")`.
- inputs:
  - W2 candidate, W1 episode, W3 conflict, W4 gate, optional W5 dry-run plan.
  - Required-info candidate with gate/conflict.
  - Queue name: `candidates`, `merge_candidates`, `noise_candidates`, `ask_info_candidates`.
- outputs:
  - Review item with `review_id`, `candidate_id`, `queue`, candidate/episode/conflict/gate/dry-run, `evidence_pack`, `review_actions`, `review_status=pending`, `observability`.
  - Queue write result: `queued|updated|batch_written`, queue file, size, queued/updated counts.
  - Review decision result: `decision_recorded|invalid_action|not_found`; approval only marks queue metadata and does not apply graph changes.
- failure_modes:
  - Unknown queue name -> normalized to `candidates.json`.
  - Duplicate review id -> update existing item idempotently.
  - Missing attachment/document/image/Jira/proj/log parse -> evidence pack still written with parse results/errors.
- observability:
  - `observability.agent_id=W6`, queue, candidate_id; evidence pack includes messages/source offsets/tool evidence.
- non_goals:
  - Does not mutate main KG.
  - Does not approve candidates automatically.
  - Does not apply approved items; W5 owns approved-only graph mutation.
  - Does not fetch remote Jira/files; evidence tools are metadata/bounded local only; documents are bounded metadata/text only; images are header-only, no OCR.

## CLI decision flow

```bash
# 1. Mark a review item after human inspection. This only edits the queue item.
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli review-decision \
  candidates review:abc123 approve --queue-dir /tmp/review_queue --reviewer alice --note "evidence checked"

# 2. Apply approved queue items explicitly through W5; no archive import needed.
PYTHONPATH=src python3 -m debug_agent_system.adapters.cli apply-approved-queue \
  --queue-dir /tmp/review_queue
```
