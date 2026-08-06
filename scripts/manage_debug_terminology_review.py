#!/usr/bin/env python3
"""Generate or apply the KG_v2 terminology human-review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from debug_agent_system.knowledge_v2.terminology_review import (  # noqa: E402
    apply_approved_terminology_reviews,
    write_terminology_review_queue,
)
from debug_agent_system.knowledge_v2.noun_discovery import (  # noqa: E402
    apply_approved_noun_discovery,
    write_noun_discovery_queue,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg-v2-root",
        type=Path,
        default=REPO_ROOT / "data/kg_v2",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply-approved",
        action="store_true",
        help=(
            "Import only explicitly approved, fully specified review items "
            "into curated_terms.json and rebuild the terminology layer."
        ),
    )
    mode.add_argument(
        "--discover-nouns",
        action="store_true",
        help=(
            "Scan configured group-chat, document and support-record "
            "corpora and refresh the non-authoritative noun review queue."
        ),
    )
    mode.add_argument(
        "--apply-approved-nouns",
        action="store_true",
        help=(
            "Import only explicitly approved noun concepts, aliases and "
            "relations into entity_ontology.json, then rebuild terminology."
        ),
    )
    args = parser.parse_args()
    root = args.kg_v2_root.resolve()
    if args.apply_approved:
        result = apply_approved_terminology_reviews(root)
    elif args.discover_nouns:
        result = write_noun_discovery_queue(root)
    elif args.apply_approved_nouns:
        result = apply_approved_noun_discovery(root)
    else:
        result = write_terminology_review_queue(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("rejected_approval_count") else 2


if __name__ == "__main__":
    raise SystemExit(main())
