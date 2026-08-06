"""JSON-backed isolated store for KG v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.entity_terminology import (
    ENTITY_RELATION_TYPES,
)
from debug_agent_system.knowledge_v2.validator import validate_graph

V2_OBJECT_FILES = {
    "KnowledgeDocument": "knowledge_documents.json",
    "MediaAsset": "media_assets.json",
    "KnowledgeSection": "knowledge_sections.json",
    "ProcedureStep": "procedure_steps.json",
    "FaultFamily": "fault_families.json",
    "FaultVariant": "fault_variants.json",
    "DiagnosticAction": "diagnostic_actions.json",
    "ActionOutcome": "action_outcomes.json",
    "RequiredInfoSpec": "required_info_specs.json",
    "DiagnosticTrace": "diagnostic_traces.json",
    "TraceStep": "trace_steps.json",
    "ExecutionObservation": "execution_observations.json",
    "BranchRule": "branch_rules.json",
    "DecisionPolicy": "decision_policies.json",
    "EvidenceItem": "evidence_items.json",
    "SourceCase": "source_cases.json",
    "DebugConcept": "debug_concepts.json",
    "TermExpression": "term_expressions.json",
    "TermSense": "term_senses.json",
}
DERIVED_TERMINOLOGY_OBJECT_TYPES = {
    "DebugConcept",
    "TermExpression",
    "TermSense",
}
DERIVED_TERMINOLOGY_RELATIONS = {
    "primary_concept",
    "expression_has_sense",
    "sense_denotes",
    "broader_concept",
    "concept_context",
    "mentions_concept",
} | ENTITY_RELATION_TYPES


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class JsonKGV2Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects_root = self.root / "objects"
        self.relations_root = self.root / "relations"
        self.review_root = self.root / "review_queue"
        self.materialized_root = self.root / "materialized_execution"
        self.ensure_layout()
        self.objects_by_type = {
            obj_type: _load_json(self.objects_root / file_name, [])
            for obj_type, file_name in V2_OBJECT_FILES.items()
        }
        self.relations = _load_json(self.relations_root / "edges.json", [])

    def ensure_layout(self) -> None:
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.relations_root.mkdir(parents=True, exist_ok=True)
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.materialized_root.mkdir(parents=True, exist_ok=True)

    def all_objects(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for items in self.objects_by_type.values():
            out.extend(item for item in items if isinstance(item, dict))
        return out

    def object_index(self, obj_type: str) -> dict[str, dict[str, Any]]:
        pk = V2_PRIMARY_KEYS[obj_type]
        return {
            str(item.get(pk) or ""): item
            for item in self.objects_by_type.get(obj_type) or []
            if isinstance(item, dict) and item.get(pk)
        }

    def replace_graph(
        self,
        objects_by_type: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        return self._write_graph(objects_by_type, relations, validate=validate, status="replaced")

    def _write_graph(
        self,
        objects_by_type: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        *,
        validate: bool,
        status: str,
    ) -> dict[str, Any]:
        schema_root = self.root / "schema"
        local_schema = schema_root if (schema_root / "object-types.json").exists() else None
        issues = validate_graph(objects_by_type, relations, schema_root=local_schema) if validate else []
        if issues:
            return {"status": "schema_invalid", "issues": issues}
        for obj_type, file_name in V2_OBJECT_FILES.items():
            data = [item for item in objects_by_type.get(obj_type) or [] if isinstance(item, dict)]
            (self.objects_root / file_name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.relations_root / "edges.json").write_text(json.dumps(relations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.__init__(self.root)
        return {
            "status": status,
            "object_counts": {k: len(v) for k, v in self.objects_by_type.items()},
            "relation_count": len(self.relations),
        }

    def merge_graph(
        self,
        objects_by_type: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        *,
        validate: bool = True,
        replace_document_sources: bool = False,
    ) -> dict[str, Any]:
        merged_objects = {obj_type: list(self.objects_by_type.get(obj_type) or []) for obj_type in V2_OBJECT_FILES}
        merged_relations = list(self.relations)
        replacement_report: dict[str, Any] = {}
        if replace_document_sources:
            merged_objects, merged_relations, replacement_report = self._purge_replaced_document_sources(
                merged_objects,
                merged_relations,
                objects_by_type,
            )
        for obj_type, incoming in objects_by_type.items():
            if obj_type not in V2_OBJECT_FILES:
                continue
            pk = V2_PRIMARY_KEYS[obj_type]
            index = {
                str(item.get(pk) or ""): item
                for item in merged_objects[obj_type]
                if isinstance(item, dict) and item.get(pk)
            }
            for item in incoming or []:
                if not isinstance(item, dict):
                    continue
                obj_id = str(item.get(pk) or "")
                if not obj_id:
                    continue
                if obj_id in index:
                    index[obj_id].update({k: v for k, v in item.items() if v not in (None, "", [])})
                else:
                    merged_objects[obj_type].append(dict(item))
                    index[obj_id] = merged_objects[obj_type][-1]
        seen = {
            (
                str(rel.get("from") or ""),
                str(rel.get("to") or ""),
                str(rel.get("relation") or ""),
            )
            for rel in merged_relations
            if isinstance(rel, dict)
        }
        for rel in relations or []:
            if not isinstance(rel, dict):
                continue
            key = (str(rel.get("from") or ""), str(rel.get("to") or ""), str(rel.get("relation") or ""))
            if not all(key) or key in seen:
                continue
            merged_relations.append(dict(rel))
            seen.add(key)
        result = self._write_graph(merged_objects, merged_relations, validate=validate, status="merged")
        if replacement_report and result.get("status") == "merged":
            result["document_source_replacement"] = replacement_report
        return result

    @staticmethod
    def _purge_replaced_document_sources(
        current_objects: dict[str, list[dict[str, Any]]],
        current_relations: list[dict[str, Any]],
        incoming_objects: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Remove the old document layer for source paths replaced by W9/W10.

        Replacement is enabled only when the incoming candidate contains both
        KnowledgeDocument and KnowledgeSection objects. Evidence aliases and
        approval-only document updates therefore keep merge semantics.
        """

        incoming_documents = [
            item for item in incoming_objects.get("KnowledgeDocument") or []
            if isinstance(item, dict) and str(item.get("source_path") or "")
        ]
        incoming_sections = [
            item for item in incoming_objects.get("KnowledgeSection") or []
            if isinstance(item, dict)
        ]
        if not incoming_documents or not incoming_sections:
            return current_objects, current_relations, {}
        source_paths = {str(item.get("source_path") or "") for item in incoming_documents}
        old_documents = [
            item for item in current_objects.get("KnowledgeDocument") or []
            if isinstance(item, dict) and str(item.get("source_path") or "") in source_paths
        ]
        old_document_ids = {
            str(item.get("document_id") or "") for item in old_documents if str(item.get("document_id") or "")
        }
        if not old_document_ids:
            return current_objects, current_relations, {}
        old_section_ids = {
            str(item.get("section_id") or "")
            for item in current_objects.get("KnowledgeSection") or []
            if isinstance(item, dict) and str(item.get("document_id") or "") in old_document_ids
        }
        old_step_ids = {
            str(item.get("procedure_step_id") or "")
            for item in current_objects.get("ProcedureStep") or []
            if isinstance(item, dict) and str(item.get("section_id") or "") in old_section_ids
        }
        old_layer_targets = old_document_ids | old_section_ids | old_step_ids
        media_touching_replaced_documents = [
            item
            for item in current_objects.get("MediaAsset") or []
            if isinstance(item, dict)
            and any(
                str(document_id or "") in old_document_ids
                for document_id in item.get("document_ids") or []
            )
        ]
        old_media_ids = {
            str(item.get("media_id") or "")
            for item in media_touching_replaced_documents
            if {
                str(document_id or "")
                for document_id in item.get("document_ids") or []
                if str(document_id or "")
            }.issubset(old_document_ids)
        }
        old_layer_targets |= old_media_ids
        old_evidence_ids = {
            str(relation.get("from") or "")
            for relation in current_relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "evidences"
            and str(relation.get("to") or "") in old_layer_targets
        }
        old_action_ids = {
            str(relation.get("to") or "")
            for relation in current_relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "candidate_action"
            and str(relation.get("from") or "") in old_step_ids
        }
        source_layer_ids = old_layer_targets | old_evidence_ids
        removable_action_ids = {
            str(item.get("action_id") or "")
            for item in current_objects.get("DiagnosticAction") or []
            if isinstance(item, dict)
            and str(item.get("action_id") or "") in old_action_ids
            and str(item.get("source_kind") or "") in {"raw_doc", "sop"}
            and not JsonKGV2Store._has_external_relation(
                str(item.get("action_id") or ""),
                current_relations,
                source_layer_ids,
            )
        }
        old_required_info_ids = {
            str(relation.get("to") or "")
            for relation in current_relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "") == "evidences"
            and str(relation.get("from") or "") in old_evidence_ids
        }
        removable_required_info_ids = {
            str(item.get("required_info_id") or "")
            for item in current_objects.get("RequiredInfoSpec") or []
            if isinstance(item, dict)
            and str(item.get("required_info_id") or "") in old_required_info_ids
            and {
                str(value)
                for value in item.get("evidence_ids") or []
                if str(value)
            }.issubset(old_evidence_ids)
            and not any(
                isinstance(relation, dict)
                and str(relation.get("relation") or "") == "evidences"
                and str(relation.get("to") or "")
                == str(item.get("required_info_id") or "")
                and str(relation.get("from") or "") not in old_evidence_ids
                for relation in current_relations
            )
        }
        removed_ids = (
            source_layer_ids
            | removable_action_ids
            | removable_required_info_ids
        )
        filtered = {key: list(value) for key, value in current_objects.items()}
        filtered["KnowledgeDocument"] = [
            item for item in filtered.get("KnowledgeDocument") or []
            if str(item.get("document_id") or "") not in old_document_ids
        ]
        filtered["KnowledgeSection"] = [
            item for item in filtered.get("KnowledgeSection") or []
            if str(item.get("section_id") or "") not in old_section_ids
        ]
        filtered["ProcedureStep"] = [
            item for item in filtered.get("ProcedureStep") or []
            if str(item.get("procedure_step_id") or "") not in old_step_ids
        ]
        retained_media: list[dict[str, Any]] = []
        for item in filtered.get("MediaAsset") or []:
            media_id = str(item.get("media_id") or "")
            if media_id in old_media_ids:
                continue
            if item not in media_touching_replaced_documents:
                retained_media.append(item)
                continue
            cloned = dict(item)
            cloned["document_ids"] = [
                value for value in item.get("document_ids") or []
                if str(value or "") not in old_document_ids
            ]
            cloned["section_ids"] = [
                value for value in item.get("section_ids") or []
                if str(value or "") not in old_section_ids
            ]
            cloned["procedure_step_ids"] = [
                value for value in item.get("procedure_step_ids") or []
                if str(value or "") not in old_step_ids
            ]
            cloned["source_occurrences"] = [
                value for value in item.get("source_occurrences") or []
                if isinstance(value, dict)
                and str(value.get("document_id") or "") not in old_document_ids
            ]
            cloned["source_chunk_ids"] = sorted({
                str(chunk_id or "")
                for occurrence in cloned["source_occurrences"]
                for chunk_id in occurrence.get("source_chunk_ids") or []
                if str(chunk_id or "")
            })
            retained_media.append(cloned)
        filtered["MediaAsset"] = retained_media
        filtered["EvidenceItem"] = [
            item for item in filtered.get("EvidenceItem") or []
            if str(item.get("evidence_id") or "") not in old_evidence_ids
        ]
        filtered["DiagnosticAction"] = [
            item for item in filtered.get("DiagnosticAction") or []
            if str(item.get("action_id") or "") not in removable_action_ids
        ]
        filtered["RequiredInfoSpec"] = [
            item for item in filtered.get("RequiredInfoSpec") or []
            if str(item.get("required_info_id") or "") not in removable_required_info_ids
        ]
        # Terminology is a fully derived projection. Keeping it during a
        # source-scoped replacement would make primary_concept look like an
        # external business reference and would leave stale concepts pointing
        # at removed source-only actions. W5 rebuilds the complete layer after
        # the business graph replacement succeeds.
        for object_type in DERIVED_TERMINOLOGY_OBJECT_TYPES:
            filtered[object_type] = []
        relations = [
            relation for relation in current_relations
            if isinstance(relation, dict)
            and str(relation.get("relation") or "")
            not in DERIVED_TERMINOLOGY_RELATIONS
            and str(relation.get("from") or "") not in removed_ids
            and str(relation.get("to") or "") not in removed_ids
        ]
        return filtered, relations, {
            "source_paths": sorted(source_paths),
            "removed_document_count": len(old_document_ids),
            "removed_section_count": len(old_section_ids),
            "removed_procedure_step_count": len(old_step_ids),
            "removed_media_count": len(old_media_ids),
            "updated_shared_media_count": (
                len(media_touching_replaced_documents) - len(old_media_ids)
            ),
            "removed_evidence_count": len(old_evidence_ids),
            "removed_source_only_action_count": len(removable_action_ids),
            "removed_source_only_required_info_count": len(removable_required_info_ids),
        }

    @staticmethod
    def _has_external_relation(
        object_id: str,
        relations: list[dict[str, Any]],
        source_layer_ids: set[str],
    ) -> bool:
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            if (
                str(relation.get("relation") or "")
                in DERIVED_TERMINOLOGY_RELATIONS
            ):
                continue
            source = str(relation.get("from") or "")
            target = str(relation.get("to") or "")
            if source == object_id and target not in source_layer_ids:
                return True
            if target == object_id and source not in source_layer_ids:
                return True
        return False

    def write_review_queue(self, name: str, items: list[dict[str, Any]]) -> None:
        path = self.review_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def read_review_queue(self, name: str) -> list[dict[str, Any]]:
        data = _load_json(self.review_root / name, [])
        return data if isinstance(data, list) else []
