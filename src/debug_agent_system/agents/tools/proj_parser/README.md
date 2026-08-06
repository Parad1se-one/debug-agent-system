# TOOL-PROJ proj_parser

- type: Worker / Tool
- responsibility: provide bounded text preview, tar manifest metadata, and key hints from AOI `.proj`/program files.
- inputs: local path string/pathlike.
- outputs: `ProjParseResult` with `text_preview`, `entries`, `archive_format`, `archive_manifest_read`, `key_hints`, `executed=false`, `mutated=false`, `archive_extracted=false`.
- non_goals: no execution, no import, no mutation, no archive extraction to disk, no binary image/detail body parsing, no full-file mandatory read, no KG writes.
- failure_modes: missing path returns structured non-content result; parse errors should be caught by `EvidenceToolAgent` as `parse_failed`.
- observability: `observability.agent_id=TOOL-PROJ`.
- strategy_validity: safe if reads stay bounded by `max_bytes`, tar handling is manifest plus whitelisted text-entry preview only, and project content is never executed.
