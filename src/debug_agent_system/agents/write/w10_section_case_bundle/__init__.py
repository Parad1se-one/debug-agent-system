"""W10 section-case to KG v2 draft bundle."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from debug_agent_system.knowledge_v2.contracts import FAMILY_SUBSYSTEM_EXPECTED, V2_PRIMARY_KEYS, make_family_id, make_id, trim_text
from debug_agent_system.knowledge_v2.validator import validate_graph
from debug_agent_system.knowledge_v2.builders import infer_action_role, infer_required_info_slot


def _empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in V2_PRIMARY_KEYS}


def _dedupe_objects(objects: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {key: [] for key in V2_PRIMARY_KEYS}
    for obj_type, items in objects.items():
        pk = V2_PRIMARY_KEYS.get(obj_type)
        seen: set[str] = set()
        for item in items or []:
            if not isinstance(item, dict):
                continue
            obj_id = str(item.get(pk) or "")
            if not obj_id or obj_id in seen:
                continue
            seen.add(obj_id)
            out[obj_type].append(item)
    return out


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        key = (str(rel.get("from") or ""), str(rel.get("to") or ""), str(rel.get("relation") or ""))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


def _family_label(section_case: dict[str, Any], source_doc_title: str) -> str:
    scopes = [str(x) for x in section_case.get("family_scope_candidates") or [] if str(x).strip()]
    if scopes:
        return scopes[0]
    return trim_text(source_doc_title, 40)


def _family_category(label: str, section_case: dict[str, Any], source_doc_title: str) -> str:
    text = " ".join([
        str(label or ""),
        str(source_doc_title or ""),
        str(section_case.get("section_title") or ""),
        str(section_case.get("variant_candidate") or ""),
        " ".join(str(x) for x in section_case.get("actions") or []),
        " ".join(str(x) for x in section_case.get("cause_notes") or []),
    ])
    if any(token in text for token in ("蓝屏", "重启", "黑屏", "死机", "系统", "驱动", "Windows", "BIOS", "DMP", "启动")):
        return "系统与软件异常"
    if any(token in text for token in ("相机", "运控", "进板", "出板", "光源", "USB", "硬盘", "内存", "主板", "CPU", "风扇", "散热")):
        return "硬件与运控"
    return "系统与软件异常"


def _family_subsystem(label: str, source_doc_title: str) -> str:
    return FAMILY_SUBSYSTEM_EXPECTED.get(label) or trim_text(source_doc_title, 40)


def _variant_id(family_id: str, variant_label: str) -> str:
    variant_id = make_id("variant", f"{family_id}:{variant_label}")
    # ``make_id`` intentionally strips non-ASCII characters.  A mixed label
    # such as ``(仅2D)残帧`` would otherwise collapse with ``(仅2D)事件超时``.
    # Keep the readable ASCII prefix and always bind non-ASCII labels to their
    # full semantic text with a stable digest.
    if any(ord(char) > 127 for char in variant_label):
        suffix = hashlib.sha1(variant_label.encode("utf-8")).hexdigest()[:12]
        variant_id = f"{variant_id.rstrip(':')}:{suffix}"
    return variant_id


class SectionCaseBundleAgent:
    """W10: convert W9 section_case outputs into KG v2 draft bundles."""

    agent_id = "W10"

    def build_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        section_cases = [row for row in payload.get("section_cases") or [] if isinstance(row, dict)]
        source_doc_title = str(payload.get("name") or payload.get("source_doc_title") or "")
        source_doc_path = str(payload.get("path") or "")
        strategy = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}
        family_scope_candidates = list(dict.fromkeys(
            str(value)
            for row in section_cases
            for value in row.get("family_scope_candidates") or []
            if str(value).strip()
        ))
        structured_sections = [row for row in payload.get("structured_sections") or [] if isinstance(row, dict)]
        if not section_cases and not structured_sections:
            return {
                "type": "W10SectionCaseBundleDraft",
                "agent_id": self.agent_id,
                "bundle_id": make_id("w10", f"{source_doc_title}:empty"),
                "source_doc_title": source_doc_title,
                "source_doc_path": source_doc_path,
                "strategy": strategy,
                "family_scope_candidates": family_scope_candidates,
                "family_ids": [],
                "variant_ids": [],
                "objects": _empty_objects(),
                "relations": [],
                "schema_valid": False,
                "schema_issues": ["empty_section_cases"],
                "report": {
                    "section_case_count": 0,
                    "family_count": 0,
                    "variant_count": 0,
                    "action_count": 0,
                    "required_info_count": 0,
                    "trace_count": 0,
                    "outcome_count": 0,
                    "evidence_count": 0,
                    "source_case_count": 0,
                },
            }
        objects = _empty_objects()
        relations: list[dict[str, Any]] = []
        document_maps = self._add_document_layer(
            objects,
            relations,
            payload,
            source_doc_title,
            source_doc_path,
        )
        chunk_manifest = self._bind_chunk_manifest(payload.get("chunk_manifest"), document_maps)
        output_mode = str(strategy.get("kg_output_mode") or "review_only")
        for row in section_cases:
            mapping_allowed = row.get("fault_mapping_allowed")
            if mapping_allowed is None:
                mapping_allowed = output_mode in {"family_support_bundle", "variant_case_bundle"}
            mapping_has_payload = bool(row.get("actions") or row.get("required_info"))
            if mapping_allowed and mapping_has_payload:
                self._add_section_case(
                    objects,
                    relations,
                    row,
                    source_doc_title,
                    source_doc_path,
                    document_maps,
                )
        objects = _dedupe_objects(objects)
        relations = _dedupe_relations(relations)
        issues = validate_graph(objects, relations)
        family_ids = [item["family_id"] for item in objects["FaultFamily"]]
        variant_ids = [item["variant_id"] for item in objects["FaultVariant"]]
        document_id = str(((document_maps.get("document") or {}).get("id")) or "")
        # The document id includes the source content hash.  Including it in
        # the bundle identity makes a new file version a fresh review item,
        # while byte-identical reruns remain idempotent.
        bundle_id = make_id("w10", f"{source_doc_title}:{document_id}:{len(section_cases)}")
        return {
            "type": "W10SectionCaseBundleDraft",
            "agent_id": self.agent_id,
            "bundle_id": bundle_id,
            "source_doc_title": source_doc_title,
            "source_doc_path": source_doc_path,
            "strategy": strategy,
            "family_scope_candidates": family_scope_candidates,
            "family_ids": family_ids,
            "variant_ids": variant_ids,
            "objects": objects,
            "relations": relations,
            "chunk_manifest": chunk_manifest,
            "schema_valid": not issues,
            "schema_issues": issues,
            "report": {
                "section_case_count": len(section_cases),
                "family_count": len(objects["FaultFamily"]),
                "variant_count": len(objects["FaultVariant"]),
                "action_count": len(objects["DiagnosticAction"]),
                "required_info_count": len(objects["RequiredInfoSpec"]),
                "trace_count": len(objects["DiagnosticTrace"]),
                "outcome_count": len(objects["ActionOutcome"]),
                "evidence_count": len(objects["EvidenceItem"]),
                "source_case_count": len(objects["SourceCase"]),
                "document_count": len(objects["KnowledgeDocument"]),
                "section_count": len(objects["KnowledgeSection"]),
                "procedure_step_count": len(objects["ProcedureStep"]),
            },
        }

    def build_atomic_case_bundles(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build one mapping-only draft per approved-looking SOP leaf case.

        Document objects are deliberately absent from these drafts.  W5 can
        therefore merge independently approved case mappings without invoking
        source-scoped replacement for the already-approved document layer.
        The relevant existing Section/ProcedureStep/Evidence objects remain in
        each bundle so source lineage and candidate_action edges stay intact.
        """

        full = self.build_bundle(payload)
        source_objects = (
            full.get("objects") if isinstance(full.get("objects"), dict) else {}
        )
        source_relations = [
            relation
            for relation in full.get("relations") or []
            if isinstance(relation, dict)
        ]
        bundles: list[dict[str, Any]] = []
        for section_case in payload.get("section_cases") or []:
            if not isinstance(section_case, dict):
                continue
            if not bool(section_case.get("fault_mapping_allowed")):
                continue
            if not section_case.get("procedure_steps"):
                continue
            source_section_id = str(section_case.get("section_id") or "")
            family_label = _family_label(
                section_case,
                str(payload.get("name") or payload.get("source_doc_title") or ""),
            )
            variant_label = str(
                section_case.get("variant_candidate") or ""
            ).strip()
            if not source_section_id or not family_label or not variant_label:
                continue
            family_id = make_family_id(family_label)
            variant_id = _variant_id(family_id, variant_label)

            objects = _empty_objects()
            objects["FaultFamily"] = [
                item
                for item in source_objects.get("FaultFamily") or []
                if isinstance(item, dict) and item.get("family_id") == family_id
            ]
            objects["FaultVariant"] = [
                item
                for item in source_objects.get("FaultVariant") or []
                if isinstance(item, dict) and item.get("variant_id") == variant_id
            ]
            # The canonical document/section/procedure layer is approved in a
            # separate review scope.  Repeating those objects here would make
            # W5 treat every atomic case as a document-version replacement.
            # Mapping drafts therefore carry lineage through source_ref and
            # source_hierarchy on each action, but contain only fault objects.
            objects["DiagnosticAction"] = [
                item
                for item in source_objects.get("DiagnosticAction") or []
                if isinstance(item, dict)
                and item.get("family_id") == family_id
                and item.get("variant_id") == variant_id
                and str((item.get("source_hierarchy") or {}).get("section_id") or "")
                == source_section_id
            ]
            objects["RequiredInfoSpec"] = [
                item
                for item in source_objects.get("RequiredInfoSpec") or []
                if isinstance(item, dict)
                and item.get("family_id") == family_id
                and item.get("variant_id") == variant_id
            ]
            objects["EvidenceItem"] = [
                item
                for item in source_objects.get("EvidenceItem") or []
                if isinstance(item, dict)
                and str(item.get("external_id") or "") == source_section_id
            ]
            evidence_ids = [
                str(item.get("evidence_id") or "")
                for item in objects["EvidenceItem"]
                if str(item.get("evidence_id") or "")
            ]
            for action in objects["DiagnosticAction"]:
                action["evidence_ids"] = list(evidence_ids)
            source_case_id = make_id(
                "case",
                f"{payload.get('path')}:{source_section_id}:{variant_label}",
            )
            action_summaries = [
                str(item.get("summary") or item.get("label") or "")
                for item in objects["DiagnosticAction"]
                if isinstance(item, dict)
            ]
            objects["SourceCase"] = [{
                "case_id": source_case_id,
                "source_kind": "raw_doc",
                "title": trim_text(variant_label, 80),
                "summary": trim_text(
                    "；".join(action_summaries) or variant_label,
                    240,
                ),
                "source_ref": trim_text(source_section_id, 200),
                "approved": False,
                "trust_tier": "extracted",
            }]
            objects = _dedupe_objects(objects)
            selected_ids = {
                str(item.get(V2_PRIMARY_KEYS[object_type]) or "")
                for object_type, values in objects.items()
                for item in values or []
                if isinstance(item, dict)
                and str(item.get(V2_PRIMARY_KEYS[object_type]) or "")
            }
            relations = _dedupe_relations([
                relation
                for relation in source_relations
                if str(relation.get("from") or "") in selected_ids
                and str(relation.get("to") or "") in selected_ids
            ])
            relations.append({
                "from": source_case_id,
                "to": variant_id,
                "relation": "supports",
            })
            relations.extend(
                {
                    "from": evidence_id,
                    "to": source_case_id,
                    "relation": "evidences",
                }
                for evidence_id in evidence_ids
            )
            relations = _dedupe_relations(relations)
            issues = validate_graph(objects, relations)
            atomic_case_id = str(
                section_case.get("atomic_case_id")
                or section_case.get("case_id")
                or f"{source_section_id}:atomic-case"
            )
            bundle_id = make_id(
                "w10-atomic",
                f"{payload.get('path')}:{atomic_case_id}:{family_id}:{variant_id}",
            )
            bundles.append({
                "type": "W10AtomicSectionCaseBundleDraft",
                "agent_id": self.agent_id,
                "bundle_id": bundle_id,
                "source_doc_title": str(payload.get("name") or ""),
                "source_doc_path": str(payload.get("path") or ""),
                "strategy": payload.get("strategy") or {},
                "family_scope_candidates": [family_label],
                "family_ids": [family_id],
                "variant_ids": [variant_id],
                "atomic_case": {
                    "atomic_case_id": atomic_case_id,
                    "section_id": source_section_id,
                    "section_title": str(section_case.get("section_title") or ""),
                    "family_label": family_label,
                    "family_id": family_id,
                    "variant_label": variant_label,
                    "variant_id": variant_id,
                    "action_count": len(objects["DiagnosticAction"]),
                },
                "objects": objects,
                "relations": relations,
                "schema_valid": not issues,
                "schema_issues": issues,
                "report": {
                    "section_case_count": 1,
                    "family_count": len(objects["FaultFamily"]),
                    "variant_count": len(objects["FaultVariant"]),
                    "action_count": len(objects["DiagnosticAction"]),
                    "required_info_count": len(objects["RequiredInfoSpec"]),
                    "evidence_count": 0,
                    "source_case_count": 1,
                    "section_count": 0,
                    "procedure_step_count": 0,
                },
            })
        return bundles

    def build_bundle_from_file(self, path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.build_bundle(payload)

    def write_bundle(self, path: str | Path, out_path: str | Path) -> dict[str, Any]:
        bundle = self.build_bundle_from_file(path)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "type": "W10SectionCaseBundleWriteResult",
            "agent_id": self.agent_id,
            "bundle_path": str(out),
            "schema_valid": bundle["schema_valid"],
            "schema_issues": bundle["schema_issues"],
            "report": bundle["report"],
        }

    def write_bundle_tree(self, root: str | Path, out_root: str | Path) -> dict[str, Any]:
        root_path = Path(root)
        out_base = Path(out_root)
        out_base.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        valid = 0
        invalid = 0
        for path in sorted(root_path.rglob("section_cases.json")):
            rel = path.relative_to(root_path)
            out_path = out_base / rel.parent / "kg_v2_draft_bundle.json"
            result = self.write_bundle(path, out_path)
            results.append({
                "section_cases_json": str(path),
                "bundle_path": str(out_path),
                **result,
            })
            if result.get("schema_valid"):
                valid += 1
            else:
                invalid += 1
        return {
            "type": "W10SectionCaseBundleBatchResult",
            "agent_id": self.agent_id,
            "root": str(root_path),
            "out_root": str(out_base),
            "bundle_count": len(results),
            "schema_valid_count": valid,
            "schema_invalid_count": invalid,
            "results": results,
        }

    def _add_document_layer(
        self,
        objects: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        payload: dict[str, Any],
        source_doc_title: str,
        source_doc_path: str,
    ) -> dict[str, dict[str, Any]]:
        strategy = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}
        strategy_id = str(strategy.get("strategy_id") or "unclassified_doc")
        path = Path(source_doc_path) if source_doc_path else None
        if path is not None and path.exists() and path.is_file():
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            content_hash = hashlib.sha256(
                json.dumps(payload.get("structured_sections") or [], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
        document_id = make_id("knowledge-document", f"{source_doc_path}:{content_hash}")
        objects["KnowledgeDocument"].append({
            "document_id": document_id,
            "title": trim_text(source_doc_title or (path.name if path else "未命名文档"), 80),
            "document_kind": strategy_id,
            "source_path": trim_text(source_doc_path or source_doc_title, 200),
            "content_hash": content_hash,
            "version": "",
            "owner": "",
            "approved": False,
            "source_kind": "raw_doc",
            "source_links": [
                {
                    "relationship_id": trim_text(
                        str(item.get("relationship_id") or ""), 80
                    ),
                    "link_order": int(item.get("link_order") or 0),
                    "paragraph_order": int(item.get("paragraph_order") or 0),
                    "link_text": trim_text(str(item.get("link_text") or ""), 160),
                    "source_context": trim_text(str(item.get("source_context") or ""), 500),
                    "target_url": trim_text(str(item.get("target_url") or ""), 500),
                    "wiki_token": trim_text(str(item.get("wiki_token") or ""), 80),
                    "standalone": bool(item.get("standalone")),
                }
                for item in payload.get("document_links") or []
                if isinstance(item, dict) and str(item.get("target_url") or "")
            ],
        })
        section_by_source: dict[str, str] = {}
        evidence_by_source: dict[str, str] = {}
        step_ids_by_source: dict[str, list[str]] = {}
        cases_by_section = {
            str(row.get("section_id") or ""): row
            for row in payload.get("section_cases") or []
            if isinstance(row, dict) and row.get("section_id")
        }
        previous_steps_by_section: dict[str, str] = {}
        for index, section in enumerate(payload.get("structured_sections") or [], start=1):
            if not isinstance(section, dict):
                continue
            source_section_id = str(section.get("section_id") or f"section:{index}")
            section_id = make_id("knowledge-section", f"{document_id}:{source_section_id}")
            section_by_source[source_section_id] = section_id
            body_lines = [str(item).strip() for item in section.get("body_lines") or [] if str(item).strip()]
            heading = str(section.get("section_title") or f"章节 {index}")
            role = self._knowledge_role(str(section.get("section_kind") or ""), heading)
            objects["KnowledgeSection"].append({
                "section_id": section_id,
                "document_id": document_id,
                "heading": trim_text(heading, 100),
                "section_order": index,
                "level": int(section.get("level") or 0),
                "knowledge_role": role,
                "summary": trim_text("；".join(body_lines) or heading, 240),
                "source_offsets": [source_section_id],
            })
            relations.append({"from": document_id, "to": section_id, "relation": "has_section"})
            evidence_id = make_id("evidence", f"{document_id}:{source_section_id}")
            evidence_by_source[source_section_id] = evidence_id
            objects["EvidenceItem"].append({
                "evidence_id": evidence_id,
                "source_kind": "tool_parse",
                "external_id": trim_text(source_section_id, 120),
                "title": trim_text(heading, 80),
                "summary": trim_text("；".join(body_lines) or heading, 500),
                "payload_ref": trim_text(source_doc_path or source_doc_title, 200),
            })
            relations.append({"from": evidence_id, "to": section_id, "relation": "evidences"})

            section_case = cases_by_section.get(source_section_id) or {}
            procedure_steps = [
                item for item in section_case.get("procedure_steps") or [] if isinstance(item, dict)
            ]
            if not procedure_steps:
                procedure_steps = [
                    {
                        "step_order": step_index,
                        "label": self._action_label(str(action or "").strip()),
                        "instruction": str(action or "").strip(),
                        "details": [],
                    }
                    for step_index, action in enumerate(section_case.get("actions") or [], start=1)
                    if str(action or "").strip() and not str(action or "").strip().endswith(("?", "？"))
                ]
            step_ids: list[str] = []
            previous_step_id = ""
            for step_index, step in enumerate(procedure_steps, start=1):
                label = str(step.get("label") or "").strip()
                instruction = str(step.get("instruction") or label).strip()
                if not label or label.endswith(("?", "？")):
                    continue
                details = [str(item).strip() for item in step.get("details") or [] if str(item).strip()]
                procedure_step_id = make_id("procedure-step", f"{section_id}:{step_index}:{label}:{instruction}")
                destructive = any(token in f"{label} {instruction}" for token in ("删除", "清空", "格式化", "拆下", "更换", "送修"))
                high_cost = any(token in f"{label} {instruction}" for token in ("更换主板", "返厂", "送修"))
                objects["ProcedureStep"].append({
                    "procedure_step_id": procedure_step_id,
                    "section_id": section_id,
                    "label": trim_text(label, 80),
                    "instruction": trim_text(instruction or label, 240),
                    "details": [trim_text(item, 160) for item in details],
                    "step_order": int(step.get("step_order") or step_index),
                    "expected_result": "",
                    "prerequisites": [],
                    "safety_level": "human_confirmation" if destructive or high_cost else "safe",
                    "high_cost": high_cost,
                    "destructive": destructive,
                    "source_kind": "raw_doc",
                })
                relations.append({"from": section_id, "to": procedure_step_id, "relation": "has_step"})
                relations.append({"from": evidence_id, "to": procedure_step_id, "relation": "evidences"})
                if previous_step_id:
                    relations.append({"from": previous_step_id, "to": procedure_step_id, "relation": "next_step"})
                previous_step_id = procedure_step_id
                step_ids.append(procedure_step_id)
            if step_ids:
                step_ids_by_source[source_section_id] = step_ids
                previous_steps_by_section[source_section_id] = step_ids[-1]
        return {
            "document": {"id": document_id},
            "section_by_source": section_by_source,
            "evidence_by_source": evidence_by_source,
            "step_ids_by_source": step_ids_by_source,
        }

    @staticmethod
    def _bind_chunk_manifest(
        raw_manifest: Any,
        document_maps: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind W9 source section ids to W10 draft KG ids for review."""

        if not isinstance(raw_manifest, dict):
            return {}
        source_manifest_id = str(raw_manifest.get("manifest_id") or "")
        section_map = {
            str(key): str(value)
            for key, value in (document_maps.get("section_by_source") or {}).items()
            if str(key) and str(value)
        }
        document_id = str(((document_maps.get("document") or {}).get("id")) or "")
        bound_chunks: list[dict[str, Any]] = []
        bound_section_ids: set[str] = set()
        for raw_chunk in raw_manifest.get("chunks") or []:
            if not isinstance(raw_chunk, dict):
                continue
            chunk = deepcopy(raw_chunk)
            source_section_ids = [str(value) for value in chunk.get("section_ids") or [] if str(value)]
            source_direct_ids = [str(value) for value in chunk.get("direct_section_ids") or [] if str(value)]
            source_primary_id = str(chunk.get("section_id") or "")
            section_ids = list(dict.fromkeys(
                section_map[value] for value in source_section_ids if value in section_map
            ))
            direct_section_ids = list(dict.fromkeys(
                section_map[value] for value in source_direct_ids if value in section_map
            ))
            primary_section_id = section_map.get(source_primary_id, "")
            if not primary_section_id and section_ids:
                primary_section_id = section_ids[-1]
            bound_section_ids.update(section_ids)
            source_offsets = chunk.get("source_offsets") or []
            offset = source_offsets[0] if source_offsets and isinstance(source_offsets[0], dict) else {}
            stable_key = "|".join((
                document_id,
                str(chunk.get("source_file_hash") or raw_manifest.get("source_file_hash") or ""),
                str(offset.get("block_start") or offset.get("paragraph_start") or ""),
                str(offset.get("block_end") or offset.get("paragraph_end") or ""),
                ",".join(section_ids),
                str(chunk.get("content_hash") or ""),
            ))
            chunk.update({
                "staged_chunk_id": str(chunk.get("chunk_id") or ""),
                "chunk_id": f"chunk:source:{hashlib.sha256(stable_key.encode('utf-8')).hexdigest()[:24]}",
                "document_id": document_id,
                "section_id": primary_section_id,
                "section_ids": section_ids,
                "direct_section_ids": direct_section_ids,
                "source_section_id": source_primary_id,
                "source_section_ids": source_section_ids,
                "source_direct_section_ids": source_direct_ids,
                "approved": False,
                "staging_status": "pending_review",
            })
            bound_chunks.append(chunk)
        content = {
            key: deepcopy(value)
            for key, value in raw_manifest.items()
            if key not in {"manifest_id", "manifest_hash", "chunks", "binding_status", "stats"}
        }
        content.update({
            "binding_status": "draft_kg_sections",
            "source_manifest_id": source_manifest_id,
            "document_id": document_id,
            "chunks": bound_chunks,
            "stats": {
                **(deepcopy(raw_manifest.get("stats")) if isinstance(raw_manifest.get("stats"), dict) else {}),
                "chunk_count": len(bound_chunks),
                "bound_section_count": len(bound_section_ids),
                "unbound_chunk_count": sum(1 for chunk in bound_chunks if not chunk.get("section_ids")),
            },
        })
        manifest_hash = hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            **content,
            "manifest_id": f"chunk-manifest:{manifest_hash[:24]}",
            "manifest_hash": manifest_hash,
        }

    @staticmethod
    def _knowledge_role(section_kind: str, heading: str) -> str:
        text = f"{section_kind} {heading}"
        if section_kind == "doc_intro":
            return "intro"
        if "cause" in section_kind or "原因" in heading:
            return "cause"
        if section_kind in {"diagnostic_actions", "solution_playbook", "procedure_step"}:
            return "procedure"
        if section_kind in {"threshold_reference", "preventive_note"}:
            return "reference"
        if section_kind == "operator_caution" or "注意" in heading or "误区" in heading:
            return "warning"
        if "faq" in section_kind:
            return "faq"
        if "policy" in section_kind or "validation" in section_kind:
            return "policy"
        return "support"

    def _add_section_case(
        self,
        objects: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        section_case: dict[str, Any],
        source_doc_title: str,
        source_doc_path: str,
        document_maps: dict[str, dict[str, Any]],
    ) -> None:
        source_section_id = str(section_case.get("section_id") or "")
        if not source_section_id:
            return
        section_title = str(section_case.get("section_title") or source_doc_title or "未命名章节")
        family_label = _family_label(section_case, source_doc_title)
        if not family_label:
            return
        family_id = make_family_id(family_label)
        variant_label = str(section_case.get("variant_candidate") or "").strip()
        variant_id = _variant_id(family_id, variant_label) if variant_label else ""
        category = _family_category(family_label, section_case, source_doc_title)
        subsystem = _family_subsystem(family_label, source_doc_title)
        source_kind = "raw_doc"
        section_id = str((document_maps.get("section_by_source") or {}).get(source_section_id) or "")
        evidence_id = str((document_maps.get("evidence_by_source") or {}).get(source_section_id) or "")
        evidence_ids = [evidence_id] if evidence_id else []

        objects["FaultFamily"].append({
            "family_id": family_id,
            "label": trim_text(family_label, 40),
            "summary": trim_text(family_label, 80),
            "category": category,
            "subsystem": trim_text(subsystem, 40),
            "scenario": "",
            "keywords": [trim_text(family_label, 40)],
            "source_kind": source_kind,
            "escalation_target": "",
        })
        if section_id:
            relations.append({"from": section_id, "to": family_id, "relation": "applicable_to"})
        if variant_id:
            objects["FaultVariant"].append({
                "variant_id": variant_id,
                "family_id": family_id,
                "label": trim_text(variant_label, 60),
                "summary": trim_text(variant_label, 180),
                "equipment_type": "工控机",
                "site": "",
                "software_version": "",
                "error_phase": "",
                "owner_context": trim_text(source_doc_title, 80),
                "escalation_target": "",
                "keywords": [trim_text(variant_label, 40)],
            })
            relations.append({"from": family_id, "to": variant_id, "relation": "has_variant"})
            if section_id:
                relations.append({"from": section_id, "to": variant_id, "relation": "describes_variant"})

        procedure_index = {
            str(item.get("procedure_step_id") or ""): item
            for item in objects.get("ProcedureStep") or []
            if isinstance(item, dict)
        }
        procedure_step_ids = list((document_maps.get("step_ids_by_source") or {}).get(source_section_id) or [])
        for idx, procedure_step_id in enumerate(procedure_step_ids, start=1):
            step = procedure_index.get(str(procedure_step_id)) or {}
            label = str(step.get("label") or "").strip()
            instruction = str(step.get("instruction") or label).strip()
            details = [str(item).strip() for item in step.get("details") or [] if str(item).strip()]
            text = "；".join(part for part in [instruction, *details] if part) or label
            action_id = make_id("action", f"{procedure_step_id}:{idx}")
            objects["DiagnosticAction"].append({
                "action_id": action_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "label": trim_text(label or self._action_label(text), 60),
                "summary": trim_text(text, 180),
                "action_role": infer_action_role(f"{label} {text}"),
                "step_order": int(step.get("step_order") or idx),
                "destructive": any(token in text for token in ("删除", "清空", "重装", "更换", "送修", "拔掉", "拆下", "拆卸")),
                "high_cost": any(token in text for token in ("更换主板", "送修", "返厂")),
                "source_kind": source_kind,
                "procedure_instruction": trim_text(instruction, 180),
                "procedure_details": [trim_text(item, 120) for item in details],
                "source_hierarchy": {
                    "section_id": source_section_id,
                    "section_title": section_title,
                    "parent_step_label": label,
                },
            })
            relations.append({"from": procedure_step_id, "to": action_id, "relation": "candidate_action"})

        for idx, question in enumerate(section_case.get("required_info") or [], start=1):
            q = str(question or "").strip()
            if not q:
                continue
            req_id = make_id("required-info", f"{source_section_id}:{idx}:{q}")
            objects["RequiredInfoSpec"].append({
                "required_info_id": req_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "slot": infer_required_info_slot(q),
                "question": trim_text(q, 100),
                "why_required": trim_text(f"该信息用于缩小 {variant_label or family_label} 的诊断范围。", 160),
                "condition": "",
                "blocks": [trim_text(variant_label or family_label, 60)],
                "priority": "medium",
                "evidence_ids": evidence_ids,
            })
            relations.append({"from": variant_id or family_id, "to": req_id, "relation": "has_required_info"})
            if evidence_id:
                relations.append({"from": evidence_id, "to": req_id, "relation": "evidences"})

    def _variant_summary(self, section_case: dict[str, Any], section_title: str) -> str:
        for key in ("support_notes", "cause_notes", "thresholds", "actions"):
            values = [str(x) for x in section_case.get(key) or [] if str(x).strip()]
            if values:
                return f"{section_title}：{values[0]}"
        return section_title

    def _source_case_summary(self, section_case: dict[str, Any]) -> str:
        chunks: list[str] = []
        for key in ("support_notes", "cause_notes", "thresholds", "actions", "required_info"):
            chunks.extend(str(x) for x in section_case.get(key) or [] if str(x).strip())
        return "；".join(chunks[:8]) or str(section_case.get("section_title") or "")

    def _evidence_summary(self, section_case: dict[str, Any]) -> str:
        return self._source_case_summary(section_case)

    def _action_label(self, text: str) -> str:
        value = str(text or "").strip()
        for sep in ("：", ":", "，", ",", "。"):
            if sep in value:
                value = value.split(sep, 1)[0]
                break
        value = value.split("（", 1)[0].split("(", 1)[0].strip()
        if value.startswith("尝试") and len(value) > 2:
            value = value[2:].strip()
        return trim_text(value, 60)


__all__ = ["SectionCaseBundleAgent"]
