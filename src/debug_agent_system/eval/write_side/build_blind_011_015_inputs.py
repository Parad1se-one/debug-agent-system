"""Freeze source-only inputs for the 011--015 blind evaluation batch.

The curated identifiers select evidence boundaries only.  This module does
not contain labels, expected cases, fault families, actions, or outcomes, so
running the extractor cannot leak the later ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kg_v2.blind_input.v1"
SOURCE = Path("data/results/xing_relation_context_final_20260717/messages.jsonl")

MESSAGE_IDS: dict[str, tuple[str, ...]] = {
    "goldcase-011": (
        "om_x100b5c5c7ba288f0c351002a0f2756f",
        "om_x100b5c5ceb91b8acb4bcbf5aa85046f",
        "om_x100b5c5ce74cbcb0c36f9ce43f89c91",
        "om_x100b5c5ce170d0b4c42366c951f2594",
        "om_x100b5c5cf9fb7478c2cb381e1012882",
        "om_x100b5c5cf5a6a0a4c4d92333ecb23b4",
        "om_x100b5c5dbcb1a880b291d3d0d644923",
    ),
    "goldcase-012": (
        "om_x100b513b38e0f0a0c2c411ae05bf6f0",
        "om_x100b513bcfdae0a0b2883010e144878",
        "om_x100b5124499b24a0c4a60f50131d2a9",
        "om_x100b5169231980bcc4f2dfc59ad8d5a",
        "om_x100b516933f95090c44df30cd91b86f",
        "om_x100b5169d23c9890c4d98187a127d0a",
        "om_x100b5169ecbb34acc441e81a712a7a7",
        "om_x100b5169ed8ef8b0c244fd627544ed0",
        "om_x100b516a970c94bcc4f9ea71e2cc881",
    ),
    "goldcase-013": (
        "om_x100b51c1d61968b8b2182731e26fc93",
    ),
    "goldcase-014": (
        "om_x100b50e1cc92f0b8c221d34887bca2d",
        "om_x100b50e3ffa31ca4c34c0a93936569a",
        "om_x100b50e3f6505484b240705c34c0655",
        "om_x100b50e3f18314a4b2e5a9a38e825eb",
        "om_x100b50ec66aad4fcb3ffb82c2294f21",
        "om_x100b50ec058db484b39b8e6942d4358",
        "om_x100b50ed2c9ba8b0c22cb739d93a057",
        "om_x100b50edca61e884b2eebaeec70b181",
        "om_x100b50ee14a7ec90c33d1216e0d5b91",
        "om_x100b50ee13a09ce0b15574e6817c51b",
        "om_x100b50ee243ab08cb37666da07d78ca",
        "om_x100b50ee265488a8b152f85e477c2c0",
        "om_x100b50eef65b50ecc2d8da57cf25fa9",
        "om_x100b50eef21a58bcb2f7e18f31c990a",
        "om_x100b50eef01390a0b124515019af452",
        "om_x100b50ea690b1ca8c27391a880449f5",
        "om_x100b50c0146e2cb4c2aec445ab75bd1",
        "om_x100b6ff70bf9f568c3ab56119541fec",
        "om_x100b6ff727646884b29e12062de44b2",
        "om_x100b6ff7a62f9ca4c3c459d795b139a",
        "om_x100b6ff07244a8b4b28bbafc959953e",
        "om_x100b6ff073038ca8c10637000d3b873",
        "om_x100b6ff23581e4a4c2d7200061a1b50",
        "om_x100b6ff2c2f9a8a0c4e0879ff69d102",
        "om_x100b6ff2c5ab9134c4d4067dbf25c8f",
        "om_x100b6ff2c0bb80b0c2d61765e37164c",
        "om_x100b6e5faa2184a0c3dfb28ce27ee86",
        "om_x100b6e584c98088cb270bce4ae17bf9",
        "om_x100b6e584ee4f4a8b281cba69f8d544",
        "om_x100b6e5fb1d4aca8b3c2a6bcd65cf9b",
    ),
    "goldcase-015": (
        "om_x100b6e4eaac4e91cc45eee3bb40f0c5",
        "om_x100b6e4f773cfcb4c44382c27a1d766",
        "om_x100b6e4f8e42ececc3969caaa0beada",
        "om_x100b6e482331f4b8c451a061cf080df",
        "om_x100b6e48c22daca0b303e4bde3f1485",
        "om_x100b6e48eb7ce8b4c25d6d5c57ce16a",
        "om_x100b6eb7ba060ca4b11fc13dcbbc4e0",
        "om_x100b6e498a2a7c8cb27a48286119350",
        "om_x100b6e49b35470f8c104303ecf636f5",
        "om_x100b6e4a45f9b4b4c34a5db903fc0bd",
    ),
}


def _message_projection(message: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "message_id": str(message.get("message_id") or ""),
        "chat_id": str(message.get("chat_id") or ""),
        "thread_id": str(message.get("thread_id") or ""),
        "relation_aware_session_id": str(message.get("relation_aware_session_id") or ""),
        "root_id": str(message.get("root_id") or ""),
        "parent_id": str(message.get("parent_id") or ""),
        "create_time": str(message.get("create_time") or ""),
        "sender": message.get("sender") or {},
        "msg_type": str(message.get("msg_type") or ""),
        "text": str(message.get("text") or ""),
        "mentions": message.get("mentions") or [],
        "attachments": message.get("attachments") or [],
        "links": message.get("links") or [],
    }
    canonical = json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    projected["source_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return projected


def build_inputs(source: str | Path = SOURCE) -> dict[str, dict[str, Any]]:
    wanted = {message_id for ids in MESSAGE_IDS.values() for message_id in ids}
    found: dict[str, dict[str, Any]] = {}
    with Path(source).open(encoding="utf-8") as handle:
        for line in handle:
            message = json.loads(line)
            message_id = str(message.get("message_id") or "")
            if message_id in wanted:
                if message_id in found:
                    raise ValueError(f"duplicate_source_message:{message_id}")
                found[message_id] = _message_projection(message)
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError("missing_source_messages:" + ",".join(missing))

    output: dict[str, dict[str, Any]] = {}
    for case_id, ids in MESSAGE_IDS.items():
        messages = [found[message_id] for message_id in ids]
        messages.sort(key=lambda item: (item["create_time"], item["message_id"]))
        canonical_messages = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        output[case_id] = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "batch_id": "gold-011-015-blind-v1",
            "source_kind": "xing_lark_raw_messages",
            "source_file": str(source),
            "label_visibility": "source_only",
            "graph_ingestion": False,
            "messages_sha256": hashlib.sha256(canonical_messages.encode("utf-8")).hexdigest(),
            "messages": messages,
        }
    return output


def write_inputs(output_root: str | Path, source: str | Path = SOURCE) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(source)
    rows = []
    for case_id, payload in sorted(inputs.items()):
        path = root / f"{case_id}.json"
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        path.write_text(body, encoding="utf-8")
        rows.append({
            "case_id": case_id,
            "file": path.name,
            "message_count": len(payload["messages"]),
            "messages_sha256": payload["messages_sha256"],
            "file_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        })
    manifest = {
        "schema_version": "kg_v2.blind_input_manifest.v1",
        "batch_id": "gold-011-015-blind-v1",
        "immutable": True,
        "contains_ground_truth": False,
        "source": str(source),
        "cases": rows,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build-blind-011-015-inputs")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default="data/kg_v2/blind_cases/gold-011-015-blind-v1/inputs")
    args = parser.parse_args(argv)
    report = write_inputs(args.out, args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
