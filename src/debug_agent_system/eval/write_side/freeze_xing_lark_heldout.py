"""Freeze source-only Xing Lark inputs for a chronological held-out batch.

This module intentionally cannot author ground truth.  It selects only
relation-aware sessions whose complete message membership is later than every
message already referenced by Goldcases 001--020, writes immutable source
inputs and hashes, and fails closed when the source export does not contain
enough eligible cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from debug_agent_system.eval.write_side.build_xing_lark_candidate_library import (
    load_known_gold_message_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LIBRARY = REPO_ROOT / "data/annotations/goldcases/candidates/xing-lark-v1/candidates.json"
DEFAULT_SOURCE = REPO_ROOT / "data/results/xing_relation_context_final_20260717/messages.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/annotations/goldcases/heldout-021-025"
TIME_FORMAT = "%Y-%m-%d %H:%M"
JIRA_RE = re.compile(r"\b(?:SMTAOITS|TEST)-\d+\b")


class HeldoutFreezeError(ValueError):
    """Raised when a chronological held-out batch cannot be frozen safely."""


def _parse_time(value: str) -> datetime:
    for pattern in (TIME_FORMAT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp: {value}")


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _message_projection(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: message.get(key)
        for key in (
            "message_id",
            "thread_id",
            "chat_id",
            "sender",
            "create_time",
            "msg_type",
            "text",
            "attachments",
            "links",
            "root_id",
            "parent_id",
            "upper_message_id",
            "relation_thread_id",
            "relation_source",
            "relation_aware_session_id",
        )
    }


def _linked_jira_issues(messages: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    keys = sorted(set(JIRA_RE.findall(json.dumps(messages, ensure_ascii=False))))
    rows: list[dict[str, Any]] = []
    for key in keys:
        path = repo_root / f"data/imports/jira_offline/raw/fault_details/{key}.json"
        if not path.is_file():
            rows.append({
                "evidence_id": f"jira:{key}",
                "key": key,
                "retrieval_status": "not_available_at_freeze",
            })
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "evidence_id": f"jira:{key}",
            "key": key,
            "summary": payload.get("summary"),
            "description": payload.get("description"),
            "status": payload.get("status"),
            "resolution": payload.get("resolution"),
            "created": payload.get("created"),
            "updated": payload.get("updated"),
            "components": payload.get("components") or [],
            "fix_versions": payload.get("fix_versions") or [],
            "issue_links": payload.get("issue_links") or [],
            "comments": payload.get("comments") or [],
            "retrieval_status": "frozen_local_snapshot",
            "source_file": str(path.relative_to(repo_root)),
            "source_file_sha256": _sha256(path),
        })
    return rows


def _external_artifacts(messages: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        message_id = str(message.get("message_id") or "")
        for index, value in enumerate(message.get("attachments") or [], start=1):
            attachment = value if isinstance(value, dict) else {"value": value}
            row: dict[str, Any] = {
                "artifact_ref": f"{message_id}:attachment:{index}",
                "source_message_ids": [message_id],
                "retrieval_status": "metadata_only",
                "content_used_for_selection": False,
                "source_metadata": attachment,
            }
            local_value = next(
                (
                    attachment.get(key)
                    for key in ("path", "local_path", "file_path")
                    if attachment.get(key)
                ),
                None,
            )
            if local_value:
                local_path = Path(str(local_value))
                if not local_path.is_absolute():
                    local_path = repo_root / local_path
                if local_path.is_file():
                    row.update({
                        "retrieval_status": "local_file_available",
                        "local_path": (
                            str(local_path.relative_to(repo_root))
                            if local_path.is_relative_to(repo_root)
                            else str(local_path)
                        ),
                        "file_sha256": _sha256(local_path),
                        "size_bytes": local_path.stat().st_size,
                    })
            rows.append(row)
    return rows


def _load_source(path: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_session: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            message = json.loads(line)
            message_id = str(message.get("message_id") or "")
            chat_id = str(message.get("chat_id") or "")
            session_id = str(message.get("relation_aware_session_id") or "")
            if message_id:
                by_id[message_id] = message
            if chat_id and session_id:
                by_session.setdefault((chat_id, session_id), []).append(message)
    for messages in by_session.values():
        messages.sort(key=lambda item: (str(item.get("create_time") or ""), str(item.get("message_id") or "")))
    return by_id, by_session


def load_embedded_gold_time_bounds(
    root: Path,
    known_gold_ids: set[str],
) -> tuple[list[str], set[str]]:
    """Read exact or relation-safe time bounds from canonical source-only Gold.

    A root/parent/upper message must precede the child that references it, so a
    child's ``create_time`` is a conservative upper bound when the parent is
    absent from the current relation-context export.
    """

    goldcases = root / "data/annotations/goldcases"
    paths = sorted((goldcases / "gold-v1").glob("goldcase-*.json"))
    paths += sorted((goldcases / "review-v3/inputs").glob("goldcase-*.json"))
    paths += sorted((goldcases / "gold-v2/inputs").glob("goldcase-*.json"))
    times: list[str] = []
    bounded_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            create_time = str(message.get("create_time") or "")
            if not create_time:
                continue
            # Validate now so malformed canonical times fail before selection.
            _parse_time(create_time)
            times.append(create_time)
            message_id = str(message.get("message_id") or "")
            if message_id in known_gold_ids:
                bounded_ids.add(message_id)
            for key in ("root_id", "parent_id", "upper_message_id"):
                relation_id = str(message.get(key) or "")
                if relation_id in known_gold_ids:
                    bounded_ids.add(relation_id)
        for value in (
            payload.get("analysis_end_inclusive"),
            (payload.get("analysis_window") or {}).get("end_inclusive")
            if isinstance(payload.get("analysis_window"), dict)
            else None,
        ):
            if value:
                _parse_time(str(value))
                times.append(str(value))
    return times, bounded_ids


def derive_gold_cutoff(
    source_by_id: dict[str, dict[str, Any]],
    known_gold_ids: set[str],
    *,
    additional_time_bounds: list[str] | None = None,
) -> tuple[str, list[str]]:
    unresolved = sorted(message_id for message_id in known_gold_ids if message_id not in source_by_id)
    times = [
        str(source_by_id[message_id].get("create_time") or "")
        for message_id in known_gold_ids
        if message_id in source_by_id and source_by_id[message_id].get("create_time")
    ]
    times.extend(str(value) for value in additional_time_bounds or [] if value)
    if not times:
        raise HeldoutFreezeError("cannot_derive_gold_cutoff:no_resolved_gold_messages")
    for value in times:
        _parse_time(value)
    return max(times, key=_parse_time), unresolved


def select_eligible_candidates(
    library: dict[str, Any],
    source_by_session: dict[tuple[str, str], list[dict[str, Any]]],
    known_gold_ids: set[str],
    cutoff: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff_time = _parse_time(cutoff)
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in library.get("candidates") or []:
        candidate_id = str(candidate.get("candidate_id") or "")
        chat_id = str(candidate.get("chat_id") or "")
        session_ids = [str(value) for value in candidate.get("relation_aware_session_ids") or [] if value]
        messages: list[dict[str, Any]] = []
        for session_id in session_ids:
            messages.extend(source_by_session.get((chat_id, session_id), []))
        deduped = {
            str(message.get("message_id") or ""): message
            for message in messages
            if message.get("message_id") and message.get("create_time")
        }
        messages = sorted(
            deduped.values(),
            key=lambda item: (str(item.get("create_time") or ""), str(item.get("message_id") or "")),
        )
        reasons: list[str] = []
        if not messages:
            reasons.append("source_session_empty")
        overlap = sorted(set(deduped) & known_gold_ids)
        if overlap:
            reasons.append("known_gold_message_overlap")
        times = [_parse_time(str(message["create_time"])) for message in messages]
        if times and min(times) <= cutoff_time:
            reasons.append("session_not_strictly_after_cutoff")
        if candidate.get("status") == "known_gold_overlap":
            reasons.append("library_marks_known_gold_overlap")
        row = {
            "candidate_id": candidate_id,
            "score": int(candidate.get("score") or 0),
            "chat_id": chat_id,
            "chat_name": candidate.get("chat_name"),
            "relation_aware_session_ids": session_ids,
            "message_count": len(messages),
            "time_range": {
                "start": str(messages[0].get("create_time") or "") if messages else None,
                "end": str(messages[-1].get("create_time") or "") if messages else None,
            },
            "known_gold_overlap_message_ids": overlap,
            "messages": [_message_projection(message) for message in messages],
            "candidate_metadata": {
                "quality_tier": candidate.get("quality_tier"),
                "score": candidate.get("score"),
                "signals": candidate.get("signals") or [],
                "issue_tags": candidate.get("issue_tags") or [],
                "anchor_message_ids": candidate.get("anchor_message_ids") or [],
            },
        }
        if reasons:
            rejected.append({key: value for key, value in row.items() if key != "messages"} | {"reasons": reasons})
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: (-row["score"], row["time_range"]["start"], row["candidate_id"]))
    return eligible, rejected


def probe(
    library_path: Path = DEFAULT_LIBRARY,
    source_path: Path = DEFAULT_SOURCE,
    count: int = 5,
    cutoff: str | None = None,
) -> dict[str, Any]:
    library = json.loads(library_path.read_text(encoding="utf-8"))
    source_by_id, source_by_session = _load_source(source_path)
    known_gold_ids = load_known_gold_message_ids(REPO_ROOT)
    embedded_time_bounds, embedded_bounded_ids = load_embedded_gold_time_bounds(
        REPO_ROOT,
        known_gold_ids,
    )
    derived_cutoff, unresolved = derive_gold_cutoff(
        source_by_id,
        known_gold_ids,
        additional_time_bounds=embedded_time_bounds,
    )
    unbounded = sorted(set(unresolved) - embedded_bounded_ids)
    effective_cutoff = cutoff or derived_cutoff
    eligible, rejected = select_eligible_candidates(
        library,
        source_by_session,
        known_gold_ids,
        effective_cutoff,
    )
    return {
        "schema_version": "debug_agent_system.xing_lark_heldout_probe.v1",
        "source_file": str(source_path.relative_to(REPO_ROOT) if source_path.is_relative_to(REPO_ROOT) else source_path),
        "source_sha256": _sha256(source_path),
        "library_file": str(library_path.relative_to(REPO_ROOT) if library_path.is_relative_to(REPO_ROOT) else library_path),
        "library_sha256": _sha256(library_path),
        "derived_gold_cutoff": derived_cutoff,
        "effective_cutoff": effective_cutoff,
        "selection_policy": "complete_relation_aware_session_strictly_after_gold_001_020_cutoff",
        "required_count": count,
        "eligible_count": len(eligible),
        "cutoff_integrity": "valid" if not unbounded else "unbounded_known_gold_messages",
        "enough_candidates": len(eligible) >= count and not unbounded,
        "unresolved_known_gold_message_ids": unresolved,
        "unbounded_known_gold_message_ids": unbounded,
        "eligible": [{key: value for key, value in row.items() if key != "messages"} for row in eligible],
        "rejection_counts": {
            reason: sum(reason in row["reasons"] for row in rejected)
            for reason in sorted({reason for row in rejected for reason in row["reasons"]})
        },
        "_eligible_with_messages": eligible,
    }


def freeze(
    output: Path = DEFAULT_OUTPUT,
    library_path: Path = DEFAULT_LIBRARY,
    source_path: Path = DEFAULT_SOURCE,
    count: int = 5,
    start_number: int = 21,
    cutoff: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise HeldoutFreezeError(f"output_already_exists:{output}")
    report = probe(library_path=library_path, source_path=source_path, count=count, cutoff=cutoff)
    eligible = report.pop("_eligible_with_messages")
    if report["unbounded_known_gold_message_ids"]:
        raise HeldoutFreezeError(
            "unbounded_known_gold_messages:"
            + ",".join(report["unbounded_known_gold_message_ids"])
        )
    if len(eligible) < count:
        raise HeldoutFreezeError(
            f"insufficient_chronological_candidates:required={count}:eligible={len(eligible)}:"
            f"cutoff={report['effective_cutoff']}"
        )

    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for offset, candidate in enumerate(eligible[:count]):
        case_id = f"goldcase-{start_number + offset:03d}"
        messages = candidate["messages"]
        linked_jira_issues = _linked_jira_issues(messages, REPO_ROOT)
        external_artifacts = _external_artifacts(messages, REPO_ROOT)
        evidence_hash = _canonical_hash({
            "messages": messages,
            "linked_jira_issues": linked_jira_issues,
            "external_artifacts": external_artifacts,
        })
        payload = {
            "schema_version": "kg_v2.chronological_heldout_input.v1",
            "case_id": case_id,
            "batch_id": f"heldout-{start_number:03d}-{start_number + count - 1:03d}",
            "source_kind": "xing_lark_relation_aware_session",
            "label_visibility": "source_only_no_ground_truth",
            "ground_truth_status": "not_authored",
            "graph_ingestion": False,
            "chronological_cutoff_exclusive": report["effective_cutoff"],
            "candidate_id": candidate["candidate_id"],
            "chat_id": candidate["chat_id"],
            "chat_name": candidate["chat_name"],
            "relation_aware_session_ids": candidate["relation_aware_session_ids"],
            "analysis_window": candidate["time_range"],
            "message_count": len(messages),
            "messages_sha256": _canonical_hash(messages),
            "input_evidence_sha256": evidence_hash,
            "messages": messages,
            "linked_jira_issues": linked_jira_issues,
            "external_artifacts": external_artifacts,
        }
        path = inputs / f"{case_id}.json"
        _write_json(path, payload)
        rows.append({
            "case_id": case_id,
            "file": path.name,
            "file_sha256": _sha256(path),
            "messages_sha256": payload["messages_sha256"],
            "input_evidence_sha256": payload["input_evidence_sha256"],
            "message_count": len(messages),
            "linked_jira_count": len(linked_jira_issues),
            "external_artifact_count": len(external_artifacts),
            "candidate_id": candidate["candidate_id"],
            "time_range": candidate["time_range"],
            "selection_audit": candidate["candidate_metadata"],
        })

    manifest = {
        "schema_version": "kg_v2.chronological_heldout_input_manifest.v1",
        "batch_id": f"heldout-{start_number:03d}-{start_number + count - 1:03d}",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "contains_ground_truth": False,
        "ground_truth_status": "not_authored",
        "graph_ingestion": False,
        "chronological_cutoff_exclusive": report["effective_cutoff"],
        "selection_policy": report["selection_policy"],
        "source_file": report["source_file"],
        "source_sha256": report["source_sha256"],
        "library_file": report["library_file"],
        "library_sha256": report["library_sha256"],
        "cases": rows,
    }
    manifest_path = inputs / "manifest.json"
    _write_json(manifest_path, manifest)
    (output / "README.md").write_text(
        "# Xing Lark chronological held-out inputs\n\n"
        "本目录只包含在既有 Gold 证据时间边界之后冻结的 source-only 输入与哈希。"
        "冻结时没有生成或查看 Ground Truth，且 `graph_ingestion=false`。\n",
        encoding="utf-8",
    )
    return {key: value for key, value in report.items() if not key.startswith("_")} | {
        "output": str(output),
        "manifest": str(manifest_path),
        "frozen_case_count": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freeze-xing-lark-heldout")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--start-number", type=int, default=21)
    parser.add_argument("--cutoff")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    if args.probe:
        report = probe(
            library_path=args.library,
            source_path=args.source,
            count=args.count,
            cutoff=args.cutoff,
        )
        report.pop("_eligible_with_messages", None)
    else:
        report = freeze(
            output=args.out,
            library_path=args.library,
            source_path=args.source,
            count=args.count,
            start_number=args.start_number,
            cutoff=args.cutoff,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
