from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_SOURCE_TYPES = {
    "chat",
    "text_history",
    "raw_doc",
    "sop_doc",
    "jira",
    "attachment",
    "manual_review",
    "diagnostic_feedback",
    "log_pattern",
}
SOP_INCREMENTAL_CONTRACT = "sop_document_incremental.v1"
ALLOWED_KNOWLEDGE_KINDS = {
    "fault_case",
    "support",
    "playbook",
    "procedure",
    "reference",
    "policy",
    "overlay",
    "evidence_only",
}
DEFAULT_KG_V2_ROOT = "data/kg_v2"
DEFAULT_GOLD_ROOT = "data/annotations/goldcases/gold-v1"
FORBIDDEN_SOP_BUILD_PART = "data/kg_v2_sop_draft_build"
GRAPH_HASH_PARTS = ("schema", "objects", "relations", "materialized_execution")
UNORDERED_SCALAR_LIST_KEYS = {
    "message_ids",
    "evidence_message_ids",
    "provided_evidence_message_ids",
    "acceptable_error_ids",
}

_WORD = re.compile(r"[A-Za-z0-9_.:-]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_SOP_TOKEN = re.compile(r"(?<![a-z0-9])sop(?![a-z0-9])", re.IGNORECASE)
_NON_SOP_TOKEN = re.compile(r"non[-_\s]?sop", re.IGNORECASE)


@dataclass(frozen=True)
class NonSopIntakeError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details or {},
        }


def structured_failure(error: NonSopIntakeError) -> dict[str, Any]:
    return {"ok": False, "error": error.to_dict()}


def _forbidden_path_match(value: Any) -> bool:
    text = str(value or "").replace("\\", "/").lower()
    return FORBIDDEN_SOP_BUILD_PART.lower() in text


def is_sop_source_reference(value: Any) -> bool:
    text = str(value or "").replace("\\", "/")
    lowered = text.lower()
    if _forbidden_path_match(lowered):
        return True
    masked = _NON_SOP_TOKEN.sub("", lowered).replace("非sop", "")
    return "标准操作流程" in masked or bool(_SOP_TOKEN.search(masked))


def _source_ref_paths(source_ref: Any) -> list[str]:
    if source_ref is None:
        return []
    if isinstance(source_ref, (str, Path)):
        return [str(source_ref)]
    if isinstance(source_ref, dict):
        paths: list[str] = []
        for key, value in source_ref.items():
            key_text = str(key).lower()
            if isinstance(value, (str, Path)) and (
                "path" in key_text or "file" in key_text or "source" in key_text or _forbidden_path_match(value)
            ):
                paths.append(str(value))
            elif isinstance(value, (dict, list, tuple)):
                paths.extend(_source_ref_paths(value))
        return paths
    if isinstance(source_ref, (list, tuple)):
        paths: list[str] = []
        for item in source_ref:
            paths.extend(_source_ref_paths(item))
        return paths
    return []


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonicalize(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _canonicalize(item, key=str(child_key))
            for child_key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        items = [_canonicalize(item) for item in value]
        if key in UNORDERED_SCALAR_LIST_KEYS and all(not isinstance(item, (dict, list)) for item in items):
            return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return items
    return value


def _stable_hash(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _as_payload(payload: Any, text: Any, content: Any) -> dict[str, Any]:
    if payload is None:
        value: dict[str, Any] = {}
    elif isinstance(payload, dict):
        value = dict(payload)
    else:
        raise NonSopIntakeError("invalid_payload", "payload must be a dict when provided.")
    if not str(value.get("text") or "").strip():
        fallback_text = str(text or content or "").strip()
        if fallback_text:
            value["text"] = fallback_text
    if not str(value.get("text") or "").strip():
        raise NonSopIntakeError("missing_intake_text", "Write intake payload requires payload.text or non-empty text/content.")
    return value


def _as_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise NonSopIntakeError(f"invalid_{field}", f"{field} must be a dict when provided.")


def validate_write_intake_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise NonSopIntakeError("invalid_intake_payload", "Write intake payload must be a dict.")

    source_kind = str(payload.get("source_kind") or "").strip().lower()
    source_type = str(payload.get("source_type") or "").strip().lower()
    knowledge_kind = str(payload.get("knowledge_kind") or "fault_case").strip().lower()
    metadata = _as_mapping(payload.get("metadata"), field="metadata")
    sop_incremental = (
        source_type == "sop_doc"
        and source_kind == "sop"
        and str(metadata.get("incremental_source_contract") or "")
        == SOP_INCREMENTAL_CONTRACT
    )
    if source_type == "sop":
        raise NonSopIntakeError(
            "sop_source_rejected",
            "SOP sources require the explicit versioned SOP document contract.",
            {"source_kind": source_kind, "source_type": source_type},
        )
    if source_type == "sop_doc" and not sop_incremental:
        raise NonSopIntakeError(
            "invalid_sop_incremental_contract",
            "sop_doc requires source_kind=sop and the versioned SOP incremental contract.",
            {
                "source_kind": source_kind,
                "source_type": source_type,
                "required_contract": SOP_INCREMENTAL_CONTRACT,
            },
        )
    if source_kind == "sop" and not sop_incremental:
        raise NonSopIntakeError(
            "sop_source_rejected",
            "SOP sources require the explicit versioned SOP document contract.",
            {"source_kind": source_kind, "source_type": source_type},
        )
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise NonSopIntakeError(
            "invalid_source_type",
            "source_type must be one of the supported write intake source types.",
            {"source_type": source_type, "allowed_source_types": sorted(ALLOWED_SOURCE_TYPES)},
        )
    if knowledge_kind not in ALLOWED_KNOWLEDGE_KINDS:
        raise NonSopIntakeError(
            "invalid_knowledge_kind",
            "knowledge_kind must be one of the allowed non-SOP knowledge kinds.",
            {"knowledge_kind": knowledge_kind, "allowed_knowledge_kinds": sorted(ALLOWED_KNOWLEDGE_KINDS)},
        )

    source_ref_paths = _source_ref_paths(payload.get("source_ref"))
    forbidden = [path for path in source_ref_paths if _forbidden_path_match(path)]
    if forbidden:
        raise NonSopIntakeError(
            "sop_build_path_rejected",
            "source_ref may not point into data/kg_v2_sop_draft_build.",
            {"forbidden_paths": forbidden},
        )

    payload_body = _as_payload(payload.get("payload"), payload.get("text"), payload.get("content"))
    evidence_pack = _as_mapping(payload.get("evidence_pack"), field="evidence_pack")
    lineage = _as_mapping(payload.get("lineage"), field="lineage")
    text = str(payload_body.get("text") or "").strip()
    source_identity = {
        "source_type": source_type,
        "source_ref": payload.get("source_ref"),
        "knowledge_kind": knowledge_kind,
        "lineage": lineage,
    }
    if payload.get("source_ref") in (None, "", [], {}) and not lineage:
        source_identity["text"] = text
    content_basis = {"payload": payload_body, "evidence_pack": evidence_pack}
    intake_id = str(payload.get("intake_id") or "").strip() or _stable_hash("intake", source_identity)
    dedupe_key = str(payload.get("dedupe_key") or "").strip() or _stable_hash("dedupe", source_identity)

    return {
        "schema_version": "debug_agent_system.write_intake_envelope.v1",
        "intake_id": intake_id,
        "source_type": source_type,
        "source_kind": source_kind,
        "source_ref": payload.get("source_ref"),
        "knowledge_kind": knowledge_kind,
        "payload": payload_body,
        "evidence_pack": evidence_pack,
        "lineage": lineage,
        "dedupe_key": dedupe_key,
        "content_hash": _stable_hash("content", content_basis),
        "text": text,
        "metadata": metadata,
    }


def build_write_intake_envelope(
    *,
    source_type: str,
    text: str = "",
    source_ref: Any = None,
    knowledge_kind: str = "fault_case",
    payload: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
    intake_id: str = "",
    dedupe_key: str = "",
    metadata: dict[str, Any] | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "source_type": source_type,
        "source_ref": source_ref,
        "knowledge_kind": knowledge_kind,
        "payload": payload or {"text": text},
        "evidence_pack": evidence_pack or {},
        "lineage": lineage or {},
        "intake_id": intake_id,
        "dedupe_key": dedupe_key,
        "text": text,
        "metadata": metadata or {},
    }
    if source_kind is not None:
        raw["source_kind"] = source_kind
    return validate_write_intake_envelope(raw)


def try_build_write_intake_envelope(**kwargs: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "envelope": build_write_intake_envelope(**kwargs)}
    except NonSopIntakeError as exc:
        return structured_failure(exc)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NonSopIntakeError(
            "missing_kg_v2_file",
            "Required KG v2 file is missing.",
            {"path": str(path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise NonSopIntakeError(
            "invalid_kg_v2_json",
            "KG v2 JSON file could not be parsed.",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc


def _load_json_tree(root: Path, part: str) -> dict[str, Any]:
    part_root = root / part
    if not part_root.exists():
        raise NonSopIntakeError("missing_kg_v2_part", "Required KG v2 graph part is missing.", {"part": part, "path": str(part_root)})
    out: dict[str, Any] = {}
    for path in sorted(part_root.rglob("*.json")):
        if _forbidden_path_match(path):
            raise NonSopIntakeError("sop_build_path_rejected", "KG v2 graph read attempted forbidden SOP build path.", {"path": str(path)})
        out[path.relative_to(root).as_posix()] = _read_json(path)
    return out


def load_kg_v2_graph(root: str | Path = DEFAULT_KG_V2_ROOT) -> dict[str, Any]:
    kg_root = Path(root)
    if _forbidden_path_match(kg_root):
        raise NonSopIntakeError("sop_build_path_rejected", "KG v2 root may not be data/kg_v2_sop_draft_build.", {"root": str(kg_root)})
    return {part: _load_json_tree(kg_root, part) for part in GRAPH_HASH_PARTS}


def compute_kg_v2_graph_hash(root: str | Path = DEFAULT_KG_V2_ROOT) -> str:
    graph = load_kg_v2_graph(root)
    canonical = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _norm(text: Any) -> str:
    value = str(text or "").lower()
    return " ".join(_WORD.findall(value) + _CJK.findall(value))


def _score_text(query: str, *features: Any) -> float:
    query_norm = _norm(query)
    if not query_norm:
        return 0.0
    query_tokens = set(query_norm.split())
    score = 0.0
    for feature in features:
        feature_norm = _norm(feature)
        if not feature_norm:
            continue
        feature_tokens = feature_norm.split()
        score += sum(1.0 for token in feature_tokens if token in query_tokens)
        if feature_norm in query_norm:
            score += 2.0
    return score


def _by_id(items: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    return {str(item.get(key) or ""): item for item in items if isinstance(item, dict) and item.get(key)}


def _graph_objects(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    objects = graph.get("objects") if isinstance(graph.get("objects"), dict) else {}
    by_name = {
        "families": "objects/fault_families.json",
        "variants": "objects/fault_variants.json",
        "actions": "objects/diagnostic_actions.json",
        "required_info": "objects/required_info_specs.json",
    }
    return {
        out_key: [item for item in objects.get(path, []) if isinstance(item, dict)]
        for out_key, path in by_name.items()
    }


def _recall_alignment_rows(query_text: str, graph: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    objects = _graph_objects(graph)
    families = _by_id(objects["families"], "family_id")
    variants = objects["variants"]
    actions_by_variant: dict[str, list[dict[str, Any]]] = {}
    required_by_variant: dict[str, list[dict[str, Any]]] = {}
    for action in objects["actions"]:
        actions_by_variant.setdefault(str(action.get("variant_id") or ""), []).append(action)
    for req in objects["required_info"]:
        required_by_variant.setdefault(str(req.get("variant_id") or ""), []).append(req)

    ranked: list[tuple[float, dict[str, Any]]] = []
    for variant in variants:
        family = families.get(str(variant.get("family_id") or ""), {})
        actions = sorted(actions_by_variant.get(str(variant.get("variant_id") or ""), []), key=lambda item: (item.get("step_order") or 999, item.get("label") or ""))
        reqs = sorted(required_by_variant.get(str(variant.get("variant_id") or ""), []), key=lambda item: (item.get("priority") or "", item.get("slot") or ""))
        score = _score_text(
            query_text,
            family.get("label"),
            family.get("summary"),
            family.get("category"),
            family.get("subsystem"),
            variant.get("label"),
            variant.get("summary"),
            " ".join(str(x) for x in variant.get("keywords") or []),
            *[action.get("label") for action in actions[:8]],
            *[req.get("question") for req in reqs[:8]],
        )
        if score <= 0:
            continue
        ranked.append((score, {
            "score": round(score, 3),
            "family": {
                "family_id": family.get("family_id") or "",
                "label": family.get("label") or "",
                "summary": family.get("summary") or "",
                "category": family.get("category") or "",
                "subsystem": family.get("subsystem") or "",
            },
            "variant": {
                "variant_id": variant.get("variant_id") or "",
                "family_id": variant.get("family_id") or "",
                "label": variant.get("label") or "",
                "summary": variant.get("summary") or "",
                "error_phase": variant.get("error_phase") or "",
            },
            "actions": [
                {
                    "action_id": action.get("action_id") or "",
                    "label": action.get("label") or "",
                    "summary": action.get("summary") or "",
                    "action_role": action.get("action_role") or "",
                    "step_order": action.get("step_order") or 0,
                }
                for action in actions[:8]
            ],
            "required_info": [
                {
                    "required_info_id": req.get("required_info_id") or "",
                    "slot": req.get("slot") or "",
                    "question": req.get("question") or "",
                    "why_required": req.get("why_required") or "",
                    "priority": req.get("priority") or "",
                }
                for req in reqs[:8]
            ],
        }))
    ranked.sort(key=lambda item: (-item[0], item[1]["family"]["label"], item[1]["variant"]["label"]))
    return [item for _, item in ranked[:limit]]


def load_reviewed_gold_examples(root: str | Path = DEFAULT_GOLD_ROOT) -> list[dict[str, Any]]:
    gold_root = Path(root)
    if _forbidden_path_match(gold_root):
        raise NonSopIntakeError("sop_build_path_rejected", "Gold cases must come from active data/kg_v2, not SOP draft build.", {"root": str(gold_root)})
    rows: list[dict[str, Any]] = []
    for path in sorted(gold_root.glob("goldcase-*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or str(payload.get("status") or "") != "reviewed":
            continue
        gold = payload.get("gold") if isinstance(payload.get("gold"), dict) else {}
        family = gold.get("family") if isinstance(gold.get("family"), dict) else {}
        variant = gold.get("variant") if isinstance(gold.get("variant"), dict) else {}
        rows.append({
            "case_id": str(payload.get("case_id") or path.stem),
            "review_type": "gold_case",
            "graph_ingestion": False,
            "source_episode_id": str(payload.get("source_episode_id") or ""),
            "family_label": str(family.get("label") or ""),
            "variant_label": str(variant.get("label") or ""),
            "source_excerpt": payload.get("source_excerpt") or [],
            "gold_structure": {
                "cases": gold.get("cases") or [],
                "family": family,
                "variant": variant,
                "actions": gold.get("actions") or [],
                "outcomes": gold.get("outcomes") or [],
                "required_info": gold.get("required_info") or [],
                "trace": gold.get("trace") or {},
            },
        })
    return rows


def _rank_gold_examples(
    query_text: str,
    examples: list[dict[str, Any]],
    family_labels: set[str],
    *,
    source_episode_id: str = "",
    limit: int,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for example in examples:
        score = _score_text(
            query_text,
            example.get("family_label"),
            example.get("variant_label"),
            " ".join(str(x) for x in example.get("source_excerpt") or []),
        )
        if example.get("family_label") in family_labels:
            score += 8.0
        exact_source_match = bool(source_episode_id and str(example.get("source_episode_id") or "") == source_episode_id)
        if exact_source_match:
            score += 100.0
        row = dict(example)
        row["score"] = round(score, 3)
        row["exact_source_match"] = exact_source_match
        if exact_source_match:
            row["selection_reason"] = "exact_source_match"
        elif example.get("family_label") in family_labels:
            row["selection_reason"] = "family_alignment"
        elif score > 0:
            row["selection_reason"] = "lexical_match"
        else:
            row["selection_reason"] = "fallback_style_reference"
        ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], item[1].get("case_id") or ""))
    return [item for _, item in ranked[:limit]]


def build_alignment_only_background(
    envelope: dict[str, Any],
    *,
    kg_v2_root: str | Path = DEFAULT_KG_V2_ROOT,
    gold_root: str | Path = DEFAULT_GOLD_ROOT,
    limit: int = 4,
    alignment_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_write_intake_envelope(envelope)
    index = alignment_index or load_alignment_context_index(kg_v2_root=kg_v2_root, gold_root=gold_root)
    graph = index["graph"]
    baseline_hash = str(index["baseline_graph_hash"])
    recalled = _recall_alignment_rows(validated["text"], graph, limit=limit)
    family_labels = {str(row.get("family", {}).get("label") or "") for row in recalled}
    source_ref = validated.get("source_ref") if isinstance(validated.get("source_ref"), dict) else {}
    source_episode_id = str(source_ref.get("episode_id") or validated.get("lineage", {}).get("source_episode_id") or "")
    gold_examples = _rank_gold_examples(
        validated["text"],
        index["reviewed_gold_examples"],
        family_labels,
        source_episode_id=source_episode_id,
        limit=3,
    )
    return {
        "schema_version": "debug_agent_system.non_sop_alignment_background.v1",
        "context_role": "alignment_only",
        "baseline_graph_hash": baseline_hash,
        "facts_may_not_be_copied_as_new_evidence": True,
        "graph_ingestion": False,
        "source_type": validated["source_type"],
        "intake_id": validated["intake_id"],
        "dedupe_key": validated["dedupe_key"],
        "allows_new_family": not recalled,
        "recalled_background": recalled,
        "reviewed_case_examples": gold_examples,
    }


def load_alignment_context_index(
    *,
    kg_v2_root: str | Path = DEFAULT_KG_V2_ROOT,
    gold_root: str | Path = DEFAULT_GOLD_ROOT,
) -> dict[str, Any]:
    """Load immutable W7 alignment data once per write-side run.

    Full text-history ingestion can contain tens of thousands of episodes.
    Reloading and hashing the graph for every episode multiplies memory and IO,
    especially when W2 workers run concurrently.  Callers may safely share the
    returned read-only dictionaries across worker threads.
    """

    graph = load_kg_v2_graph(kg_v2_root)
    canonical = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "debug_agent_system.non_sop_alignment_index.v1",
        "kg_v2_root": str(kg_v2_root),
        "gold_root": str(gold_root),
        "baseline_graph_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "graph": graph,
        "reviewed_gold_examples": load_reviewed_gold_examples(gold_root),
    }


def try_build_alignment_only_background(envelope: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "background": build_alignment_only_background(envelope, **kwargs)}
    except NonSopIntakeError as exc:
        return structured_failure(exc)
