# TOOL-IMAGE image_parser

- type: Worker / Tool
- responsibility: parse safe image header metadata for screenshot/sample-image evidence.
- inputs: path string or W1 attachment metadata dict.
- outputs: `ImageParseResult` with `image_format`, `width`, `height`, `megapixels`, `aspect_ratio`, `header_read`.
- non_goals: no OCR, no pixel decoding, no vision model, no archive extraction, no KG writes.
- failure_modes: missing/unsupported images degrade to metadata-only result.
- observability: `observability.agent_id=TOOL-IMAGE`.
- strategy_validity: safe if it remains bounded header metadata only.

Supported header formats: PNG, JPEG, GIF, WebP, BMP.
