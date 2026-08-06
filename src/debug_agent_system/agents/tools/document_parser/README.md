# TOOL-DOCUMENT document_parser

- type: Worker / Tool
- responsibility: parse bounded PDF/Office document metadata for review evidence.
- inputs: path string or W1 attachment metadata dict.
- outputs: `DocumentParseResult` with document format, bounded preview/manifest fields, and safety flags.
- non_goals: no macro execution, no formula evaluation, no page rendering, no OCR, no KG writes.
- failure_modes: missing/unsupported/corrupt documents degrade to metadata-only result.
- observability: `observability.agent_id=TOOL-DOCUMENT`.
- strategy_validity: safe if PDF reads remain bounded bytes and OOXML reads stay manifest/whitelisted XML only.

Supported:

- PDF: bounded header/body bytes for version/page-count hints and printable preview.
- OOXML (`.docx/.xlsx/.pptx`): zip central directory + whitelisted XML text preview.
- Legacy Office (`.doc/.xls/.ppt`): OLE header metadata only.
