# TOOL-ATTACHMENT attachment_parser

- type: Worker / Tool
- responsibility: classify attachment metadata into evidence roles and read bounded previews from whitelisted text attachments.
- inputs: path string or W1 attachment metadata dict.
- outputs: `AttachmentParseResult` with `evidence_role`, `mime_guess`, `text_preview_read`, `key_hints`, `archive_extracted=false`.
- non_goals: no archive extraction, no OCR, no binary/PDF/Office/image parsing, no KG writes.
- failure_modes: malformed metadata degrades to best-effort metadata result.
- observability: `observability.agent_id=TOOL-ATTACHMENT`.
- strategy_validity: safe if text reading remains bounded, deterministic, and limited to whitelisted text extensions.

## Text preview boundary

Only these extensions may be read as bounded text preview:

- `.txt`
- `.csv`
- `.json`
- `.ini`
- `.toml`
- `.cfg`
- `.yaml`
- `.yml`
- `.reg`

Still metadata-only:

- images (`.jpg/.png/.webp/...`)
- PDF / Office / model files
- archives and log packages, which are routed to `TOOL-LOG-PACKAGE`
- `.proj`, which is routed to `TOOL-PROJ`

When preview is read, `key_hints` may contain:

- `versions`
- `ip_addresses`
- `jira_ids`
- `urls`
- `error_codes`
- `error_lines`
- `phase_hints`
