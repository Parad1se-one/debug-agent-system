"""Publish the reviewed boarding-failure SOP into the canonical KG_v2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write import RawDocIngestAgent, SectionCaseBundleAgent
from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline
from debug_agent_system.core.paths import project_root
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS, make_id
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store


SOURCE_PATH = "data/raw/aoi_debug_agent_sources/进板失败SOP--20250521.docx"
FAMILY_ID = "family:bc03a0555f1c"
VARIANT_ID = "variant:0338d170a1d3"
TRACE_ID = "trace:23e54f807bea"
SOURCE_CASE_ID = "case:5e0c0e1c9661"
CORE_EVIDENCE_ID = "evidence:ddccf90609cc"
DEFAULT_REPORT = Path("data/results/boarding_failure_document_rebuild.json")

ACTION_BY_SECTION = {
    "3.1": "action:43189f7c1f19",
    "3.2.1": "action:db905b5af65f",
    "3.3": "action:067f7227cb5e",
    "3.4": "action:54a1a9ff1629",
    "3.5": "action:bbe2107bff74",
}

STEP_SPECS: dict[str, dict[str, Any]] = {
    "3.1": {
        "label": "检查出板口是否仍有板未出板",
        "instruction": "检查出板口是否仍有板；若出板口有板，先将其安全取出，再观察设备能否正常进板。",
        "details": [
            "进板口和出板口同时感应到板属于异常状态，设备此时不会继续进板。",
        ],
        "expected_result": "移出出板口滞留板后，进板流程恢复或可继续定位下一项。",
        "safety_level": "safe",
        "destructive": False,
    },
    "3.2.1": {
        "label": "检查进板传感器是否被触发并调整灵敏度",
        "instruction": "将板放在进板口流道上，观察传感器电源灯和感应灯；未触发时调整传感器灵敏度。",
        "details": [
            "绿色灯为电源灯，常亮表示传感器工作状态正常。",
            "红色灯为感应灯：感应到板时亮起，未感应到板时熄灭。",
            "最佳状态是放板时红灯正好亮起，拿掉板时红灯正好熄灭。",
        ],
        "expected_result": "放板时红色感应灯亮起，移除板时红色感应灯熄灭。",
        "safety_level": "safe",
        "destructive": False,
    },
    "3.3": {
        "label": "在工厂软件中检查皮带正反转",
        "instruction": "确认进板传感器已触发后，退出运控和主程序，再进入工厂软件检查皮带正转和反转。",
        "details": [
            "工厂软件所在路径见图4，皮带正转/反转操作见图5。",
            "打开工厂软件前必须退出运控和主程序，避免控制冲突。",
        ],
        "expected_result": "工厂软件中皮带正转和反转均可正常执行。",
        "safety_level": "caution",
        "destructive": False,
    },
    "3.4": {
        "label": "检查IO点位信号是否正确",
        "instruction": "皮带运行正常时，在工厂软件中将板放到对应流道进板口，观察进板 IO 信号是否触发。",
        "details": [
            "IO 信号界面见图6。",
            "若两个信号相反，说明运控卡 0 卡和 1 卡可能插反；调换前需停止设备并由人工确认。",
            "原文注明双轨示意图7暂缺，不能把缺图当作已验证证据。",
        ],
        "expected_result": "板到达对应进板口时，正确流道的 IO 信号被触发且点位方向一致。",
        "safety_level": "caution",
        "destructive": False,
    },
    "3.5": {
        "label": "重新拔插皮带电机线",
        "instruction": "停止设备并由人工确认后，在设备后部检查皮带电机线接触情况并重新拔插。",
        "details": [
            "电机线所在位置见图8。",
        ],
        "expected_result": "电机线连接牢固，重新上电验证后皮带可正常运行。",
        "safety_level": "human_confirmation",
        "destructive": True,
    },
}

IMAGE_BINDINGS = {
    "3.2.1": {
        "word/media/image1.png": "图1：进板传感器指示灯及触发状态",
        "word/media/image2.png": "图2：进板传感器灵敏度调节位置",
        "word/media/image3.png": "图3：移除板子后的未触发状态",
    },
    "3.3": {
        "word/media/image4.png": "图4：工厂软件所在路径",
        "word/media/image5.png": "图5：工厂软件皮带正转/反转操作",
    },
    "3.4": {
        "word/media/image6.png": "图6：工厂软件进板 IO 信号检查",
    },
    "3.5": {
        "word/media/image7.jpeg": "图8：设备后部皮带电机线位置",
    },
}


def _source_code(item: dict[str, Any]) -> str:
    for value in item.get("source_offsets") or []:
        marker = ":sec:"
        if marker in str(value):
            return str(value).split(marker, 1)[1]
    return ""


def _media_refs(
    staged_payload: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    source = repo_root / SOURCE_PATH
    document_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    by_archive: dict[str, dict[str, Any]] = {}
    for chunk in (staged_payload.get("chunk_manifest") or {}).get("chunks") or []:
        for ref in chunk.get("media_refs") or []:
            if isinstance(ref, dict) and str(ref.get("archive_path") or ""):
                by_archive[str(ref["archive_path"])] = dict(ref)
    result: dict[str, list[dict[str, Any]]] = {}
    for code, bindings in IMAGE_BINDINGS.items():
        refs: list[dict[str, Any]] = []
        for archive_path, caption in bindings.items():
            raw = by_archive.get(archive_path)
            if raw is None:
                raise RuntimeError(f"missing staged source image: {archive_path}")
            content_hash = str(raw.get("content_hash") or "")
            suffix = Path(archive_path).suffix.lower()
            relative_path = (
                f"data/kg_v2_sag/assets/{document_hash[:16]}/"
                f"{content_hash[:24]}{suffix}"
            )
            refs.append({
                **raw,
                "media_id": f"media:{content_hash[:24]}",
                "source_path": SOURCE_PATH,
                "relative_path": relative_path,
                "asset_path": str((repo_root / relative_path).resolve()),
                "context_label": caption,
                "caption": caption,
                "role": (
                    "diagnostic_evidence"
                    if code in {"3.2.1", "3.4"}
                    else "procedure_operation"
                ),
                "source_section_id": code,
                "primary": True,
            })
        result[code] = refs
    return result


def _build_document_fragment(
    active: JsonKGV2Store,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    repo_root = project_root(__file__)
    source = repo_root / SOURCE_PATH
    staged = RawDocIngestAgent().build_section_cases(SOURCE_PATH)
    draft = SectionCaseBundleAgent().build_bundle(staged)
    if not draft.get("schema_valid"):
        raise RuntimeError(
            "W10 document draft invalid: "
            + "; ".join(draft.get("schema_issues") or [])
        )

    objects = {key: [] for key in V2_PRIMARY_KEYS}
    for object_type in ("KnowledgeDocument", "KnowledgeSection", "EvidenceItem"):
        objects[object_type] = [
            deepcopy(item) for item in draft["objects"].get(object_type) or []
        ]
    document = objects["KnowledgeDocument"][0]
    document["approved"] = True
    document_id = str(document["document_id"])
    sections = objects["KnowledgeSection"]
    section_by_code = {_source_code(item): item for item in sections}
    evidence_by_code = {
        str(item.get("external_id") or "").split(":sec:", 1)[1]: item
        for item in objects["EvidenceItem"]
        if ":sec:" in str(item.get("external_id") or "")
    }

    curated_summaries = {
        code: "；".join([spec["instruction"], *spec["details"]])
        for code, spec in STEP_SPECS.items()
    }
    for code, summary in curated_summaries.items():
        if code in section_by_code:
            section_by_code[code]["summary"] = summary[:240]
        if code in evidence_by_code:
            evidence_by_code[code]["summary"] = summary[:500]
            evidence_by_code[code]["payload_ref"] = SOURCE_PATH

    relations = [
        deepcopy(item)
        for item in draft.get("relations") or []
        if str(item.get("relation") or "") in {"has_section", "evidences"}
        and str(item.get("to") or "") in {
            document_id,
            *[str(section.get("section_id") or "") for section in sections],
        }
    ]
    step_ids: list[str] = []
    previous_step_id = ""
    for ordinal, (code, action_id) in enumerate(ACTION_BY_SECTION.items(), start=1):
        section = section_by_code[code]
        spec = STEP_SPECS[code]
        section_id = str(section["section_id"])
        step_id = make_id(
            "procedure-step",
            f"{section_id}:{ordinal}:{spec['label']}:{spec['instruction']}",
        )
        step_ids.append(step_id)
        objects["ProcedureStep"].append({
            "procedure_step_id": step_id,
            "section_id": section_id,
            "label": spec["label"],
            "instruction": spec["instruction"],
            "details": list(spec["details"]),
            "step_order": ordinal,
            "expected_result": spec["expected_result"],
            "prerequisites": [],
            "safety_level": spec["safety_level"],
            "high_cost": False,
            "destructive": spec["destructive"],
            "source_kind": "raw_doc",
        })
        relations.extend([
            {"from": section_id, "to": step_id, "relation": "has_step"},
            {"from": step_id, "to": action_id, "relation": "candidate_action"},
        ])
        evidence = evidence_by_code.get(code) or {}
        if evidence:
            relations.append({
                "from": str(evidence["evidence_id"]),
                "to": step_id,
                "relation": "evidences",
            })
        if previous_step_id:
            relations.append({
                "from": previous_step_id,
                "to": step_id,
                "relation": "next_step",
            })
        previous_step_id = step_id

    for code, section in section_by_code.items():
        section_id = str(section["section_id"])
        parent_code = code.rsplit(".", 1)[0] if "." in code else ""
        if parent_code in section_by_code:
            parent_id = str(section_by_code[parent_code]["section_id"])
            section["parent_section_id"] = parent_id
            relations.append({
                "from": parent_id,
                "to": section_id,
                "relation": "has_subsection",
            })
        relations.extend([
            {"from": section_id, "to": FAMILY_ID, "relation": "applicable_to"},
            {"from": section_id, "to": VARIANT_ID, "relation": "describes_variant"},
        ])

    image_refs = _media_refs(staged, repo_root=repo_root)
    active_actions = {
        str(item.get("action_id") or ""): item
        for item in active.objects_by_type.get("DiagnosticAction") or []
    }
    for code, action_id in ACTION_BY_SECTION.items():
        current = active_actions.get(action_id)
        if current is None:
            raise RuntimeError(f"missing formal action: {action_id}")
        spec = STEP_SPECS[code]
        action = deepcopy(current)
        action.update({
            "summary": spec["instruction"],
            "source_section_id": code,
            "stage": code,
            "safety_level": spec["safety_level"],
            "applicability_condition": (
                "按 3.1→3.5 顺序执行；仅当前一步未恢复时继续下一步。"
            ),
            "expected_result": spec["expected_result"],
            "destructive": spec["destructive"],
            "curated_image_refs": image_refs.get(code) or [],
        })
        objects["DiagnosticAction"].append(action)

    core_evidence = deepcopy(active.object_index("EvidenceItem")[CORE_EVIDENCE_ID])
    core_evidence.update({
        "summary": (
            "正式来源《进板失败SOP--20250521》规定按顺序排查："
            "3.1 检查出板口是否仍有板；"
            "3.2 检查进板传感器指示灯并调整灵敏度；"
            "3.3 退出运控和主程序后在工厂软件检查皮带正反转；"
            "3.4 检查进板 IO 点位，信号相反时人工确认运控卡顺序；"
            "3.5 停机并人工确认后重新拔插设备后部皮带电机线。"
        ),
        "payload_ref": SOURCE_PATH,
    })
    objects["EvidenceItem"].append(core_evidence)
    relations.append({
        "from": CORE_EVIDENCE_ID,
        "to": document_id,
        "relation": "evidences",
    })

    source_case = deepcopy(active.object_index("SourceCase")[SOURCE_CASE_ID])
    source_case.update({
        "source_ref": SOURCE_PATH,
        "approved": True,
    })
    objects["SourceCase"].append(source_case)

    seen: set[tuple[str, str, str]] = set()
    deduped_relations: list[dict[str, Any]] = []
    for relation in relations:
        key = (
            str(relation.get("from") or ""),
            str(relation.get("to") or ""),
            str(relation.get("relation") or ""),
        )
        if all(key) and key not in seen:
            seen.add(key)
            deduped_relations.append(relation)
    return objects, deduped_relations, {
        "document_id": document_id,
        "source_file_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "section_count": len(sections),
        "procedure_step_count": len(step_ids),
        "source_chunk_count": len(
            (staged.get("chunk_manifest") or {}).get("chunks") or []
        ),
        "source_image_count": sum(len(value) for value in image_refs.values()),
        "section_ids_by_code": {
            code: str(item.get("section_id") or "")
            for code, item in section_by_code.items()
        },
    }


def rebuild(
    *,
    kg_root: Path,
    report_path: Path,
    apply: bool,
    sag_path: Path,
) -> dict[str, Any]:
    active = JsonKGV2Store(kg_root)
    objects, relations, fragment_report = _build_document_fragment(active)
    result: dict[str, Any] = {
        "schema_version": "debug_agent_system.boarding_failure_document_rebuild.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "kg_root": str(kg_root),
        "family_id": FAMILY_ID,
        "variant_id": VARIANT_ID,
        "trace_id": TRACE_ID,
        "source_case_id": SOURCE_CASE_ID,
        "core_evidence_id": CORE_EVIDENCE_ID,
        "fragment_object_counts": {
            key: len(value) for key, value in objects.items() if value
        },
        "fragment_relation_count": len(relations),
        **fragment_report,
    }
    if apply:
        merge_result = active.merge_graph(
            objects,
            relations,
            validate=True,
            replace_document_sources=True,
        )
        if merge_result.get("status") != "merged":
            raise RuntimeError(
                "document merge failed: "
                + "; ".join(merge_result.get("issues") or [])
            )
        pipeline = WriteSideV2Pipeline(kg_root)
        media_result = pipeline.rebuild_media_assets()
        materialized = pipeline.materialize_execution()
        sag_result = pipeline.build_sqlite_sag(sag_path, reset=True)
        validation = pipeline.validate_current_graph()
        if validation.get("status") != "valid":
            raise RuntimeError(
                "published graph invalid: "
                + "; ".join(validation.get("issues") or [])
            )
        result.update({
            "merge_result": merge_result,
            "media_result": media_result,
            "materialized": materialized,
            "sag_result": sag_result,
            "final_validation": validation,
        })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-root", type=Path, default=Path("data/kg_v2"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--sag-path",
        type=Path,
        default=Path("data/kg_v2_sag/debug_agent_v2.sqlite"),
    )
    args = parser.parse_args(argv)
    result = rebuild(
        kg_root=args.kg_root,
        report_path=args.report,
        apply=args.apply,
        sag_path=args.sag_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
