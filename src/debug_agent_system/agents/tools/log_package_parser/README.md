# TOOL-LOG-PACKAGE log_package_parser

- type: Worker / Tool
- responsibility: safely inspect log package metadata, zip central-directory manifests, and bounded hints from whitelisted text logs.
- inputs: path string/pathlike or W1 attachment metadata dict.
- outputs: `LogPackageParseResult` with `entries`, `detected_roles`, `text_hints`, `has_dmp`, `has_evtx`, `has_startup_log`, `has_dlog`.
- non_goals: no archive extraction, no execution, no OCR, no binary log body reading, no full log parsing, no KG writes.
- failure_modes: missing/unreadable/non-zip archive returns metadata-only or structured parse_failed through `EvidenceToolAgent`.
- observability: `observability.agent_id=TOOL-LOG-PACKAGE`.
- strategy_validity: safe if zip reading is limited to central-directory metadata plus bounded preview of `.log/.txt/.csv` entries only; `.dmp/.evtx` remain metadata-only.
