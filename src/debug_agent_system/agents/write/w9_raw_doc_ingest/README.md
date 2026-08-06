# W9 Raw Doc Ingest Agent

- id: `W9`
- type: Worker
- owner: `src/debug_agent_system/agents/write/w9_raw_doc_ingest`
- responsibility: classify non-SOP raw knowledge documents into stable `doc_strategy` buckets and produce an executable staging checklist before KG v2 ingestion.

## Why W9 exists

W9 is intentionally separate from W1.

- `W1` handles field evidence from chats / Jira / attachments.
- `W9` handles raw knowledge documents such as manuals, guides, tutorials, specs, FAQ files, and process overlays.

## Current scope

Current W9 supports:

- processing checklist for all non-SOP raw docs
- `structured_sections`
- `section_case`
- review-only `chunk_manifest` built with the shared Section/FAQ/table semantic chunker

Implemented section extraction / section-case staging:

- `troubleshooting_topic_doc`
- `repair_playbook_doc`
- `fault_manual_numbered`
- `procedure_doc`
- `spec_doc`
- `validation_checklist_doc`
- `faq_doc`
- `overlay_process_doc`
- `unclassified_doc` as review-only fallback

W9 still does **not** write KG objects yet.

Supported strategy buckets:

- `fault_manual_numbered`
- `troubleshooting_topic_doc`
- `repair_playbook_doc`
- `procedure_doc`
- `spec_doc`
- `validation_checklist_doc`
- `faq_doc`
- `overlay_process_doc`
- `unclassified_doc`

## Entrypoints

- `inspect_document(path)`
- `build_root_checklist(root, include_sop=False)`
- `build_structured_sections(path)`
- `build_section_cases(path)`
- `write_doc_outputs(path, out_dir)`

`build_section_cases()` returns a content-addressed `chunk_manifest` alongside
the sections/cases. Chunks retain source text, block offsets and source hash,
but remain `approved=false` and cannot be queried by the online SAG at W9.
`write_doc_outputs()` also writes `chunk_manifest.json`.

## Output contract

- single file inspection:
  - `type=W9DocInspection`
  - `strategy`
  - `text_preview`
  - `recommended_steps[]`

- batch root checklist:
  - `type=W9DocStrategyChecklist`
  - `counts_by_strategy`
  - `documents[]`

## Boundary

- W9 excludes SOP by default.
- W9 does not execute macros, OCR, archive extraction, or full-content rendering.
- W9 does not call LLM in the current skeleton.
- W9 never publishes staged chunks; W10 binds draft KG ids and W5 controls the approved publish boundary.
