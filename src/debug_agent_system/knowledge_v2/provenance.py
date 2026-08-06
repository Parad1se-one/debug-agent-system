from __future__ import annotations

from typing import Any


GOLD_ORIGIN_VALUES = {"gold_case", "reviewed_gold_case", "alignment_only"}
GOLD_PATH_MARKERS = ("/gold_cases/", "data/annotations/goldcases/gold-v1", "reviewed_case_examples")
EPISODE_MESSAGE_GROUPS = (
    "fault_description_messages",
    "diagnostic_chain_messages",
    "resolution_messages",
    "case_context_messages",
    "noise_messages",
)


def alignment_provenance_issues(envelope: dict[str, Any]) -> list[str]:
    """Reject candidate evidence copied from alignment/gold context.

    Gold cases remain valid W2 few-shot and regression references.  Candidate
    evidence must still resolve to the current intake's messages or tool
    evidence, never to the alignment payload itself.
    """

    containers = _candidate_containers(envelope)
    objects = _objects(containers)
    issues: list[str] = []

    for item in [*objects.get("EvidenceItem", []), *objects.get("SourceCase", [])]:
        for key in ("source_kind", "review_type", "context_role"):
            value = str(item.get(key) or "").strip().lower()
            if value in GOLD_ORIGIN_VALUES:
                issues.append(f"alignment_evidence_origin:{key}:{value}")
        for key in ("source_ref", "payload_ref", "external_id", "case_id", "evidence_id"):
            value = str(item.get(key) or "").replace("\\", "/")
            lowered = value.lower()
            if any(marker in lowered for marker in GOLD_PATH_MARKERS) or lowered.startswith("goldcase-"):
                issues.append(f"alignment_evidence_reference:{key}")

    allowed_message_ids = _current_message_ids(containers)
    source_type = _source_type(containers)
    chat_evidence = [
        item for item in objects.get("EvidenceItem", [])
        if str(item.get("source_kind") or "").strip().lower() in {"chat_message", "text_message", "text_history_message"}
    ]
    if source_type in {"chat", "text_history", "manual_review"} and chat_evidence:
        if not allowed_message_ids:
            issues.append("alignment_evidence_missing_current_message_ids")
        for item in chat_evidence:
            external_id = str(item.get("external_id") or "").strip()
            if allowed_message_ids and external_id and external_id not in allowed_message_ids:
                issues.append(f"alignment_evidence_message_outside_intake:{external_id}")

    current_paths = _current_source_paths(containers)
    if source_type == "raw_doc":
        for item in objects.get("EvidenceItem", []):
            if str(item.get("source_kind") or "").strip().lower() != "tool_parse":
                issues.append("alignment_evidence_kind_mismatch:raw_doc")
            if not _path_belongs_to_intake(str(item.get("payload_ref") or ""), current_paths):
                issues.append("alignment_evidence_path_outside_intake:raw_doc")
    if source_type == "sop_doc":
        for item in objects.get("EvidenceItem", []):
            if str(item.get("source_kind") or "").strip().lower() != "sop":
                issues.append("alignment_evidence_kind_mismatch:sop_doc")
            if not _path_belongs_to_intake(
                str(item.get("payload_ref") or ""), current_paths
            ):
                issues.append(
                    "alignment_evidence_path_outside_intake:sop_doc"
                )
    if source_type in {"jira", "attachment"}:
        for item in objects.get("EvidenceItem", []):
            if str(item.get("source_kind") or "").strip().lower() not in {"jira", "tool_parse"}:
                issues.append(f"alignment_evidence_kind_mismatch:{source_type}")
            if not _path_belongs_to_intake(str(item.get("payload_ref") or ""), current_paths):
                issues.append(f"alignment_evidence_path_outside_intake:{source_type}")

    return sorted(set(issues))


def _candidate_containers(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending = [envelope]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        out.append(item)
        for key in ("typed_candidate", "candidate", "payload", "graph", "evidence_pack"):
            child = item.get(key)
            if isinstance(child, dict):
                pending.append(child)
    return out


def _objects(containers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for container in containers:
        raw = container.get("objects")
        if not isinstance(raw, dict):
            continue
        for object_type, items in raw.items():
            if not isinstance(items, list):
                continue
            out.setdefault(str(object_type), []).extend(item for item in items if isinstance(item, dict))
    return out


def _current_message_ids(containers: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for container in containers:
        for key in ("message_ids", "evidence_message_ids"):
            value = container.get(key)
            if isinstance(value, list):
                out.update(str(item) for item in value if str(item))
        source_ref = container.get("source_ref")
        if isinstance(source_ref, dict):
            out.update(str(item) for item in source_ref.get("message_ids") or [] if str(item))
        episode = container.get("episode")
        if not isinstance(episode, dict):
            continue
        out.update(str(item) for item in episode.get("evidence_message_ids") or [] if str(item))
        for group in EPISODE_MESSAGE_GROUPS:
            for message in episode.get(group) or []:
                if isinstance(message, dict) and str(message.get("message_id") or ""):
                    out.add(str(message["message_id"]))
    return out


def _source_type(containers: list[dict[str, Any]]) -> str:
    for container in containers:
        value = str(container.get("source_type") or "").strip().lower()
        if value:
            return value
    return ""


def _current_source_paths(containers: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for container in containers:
        source_ref = container.get("source_ref")
        if isinstance(source_ref, str) and source_ref.strip():
            out.add(_normalized_path(source_ref))
        elif isinstance(source_ref, dict):
            for key in ("path", "source_path", "file", "file_path"):
                if str(source_ref.get(key) or "").strip():
                    out.add(_normalized_path(source_ref[key]))
        for item in container.get("files") or []:
            if isinstance(item, dict) and str(item.get("path") or "").strip():
                out.add(_normalized_path(item["path"]))
    return {item for item in out if item}


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def _path_belongs_to_intake(value: str, current_paths: set[str]) -> bool:
    path = _normalized_path(value)
    if not path or not current_paths:
        return False
    return any(path == root or path.startswith(f"{root}/") for root in current_paths)
