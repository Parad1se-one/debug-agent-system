"""Rebuild and selectively publish the camera-capture-failure family.

The curated SOP graph is first built in an isolated temporary directory.  Only
the reviewed camera family fragment is then merged into the active KG_v2 graph;
raw document nodes and reviewed field cases remain in place.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

from debug_agent_system.agents.write_v2.pipeline import WriteSideV2Pipeline
from debug_agent_system.agents.write_v2.sop_manual_build import build_graph
from debug_agent_system.knowledge_v2.contracts import V2_PRIMARY_KEYS
from debug_agent_system.knowledge_v2.json_store import JsonKGV2Store
from debug_agent_system.knowledge_v2.validator import validate_graph


FAMILY_ID = "family:5274d74078aa"
LEGACY_CASE_FAMILY_ID = "family::5274d74078aa"
LEGACY_COARSE_FAMILY_ID = "family:bc03a0555f1c"
VARIANT_ID = "variant:505989010b74"
LEGACY_COARSE_VARIANT_ID = "variant:e994b10c2f5b"
DOCUMENT_ID = (
    "knowledge-document:data-raw-aoi_debug_agent_sources-.docx:"
    "277c4fd897c8d393463cbb-9370c1d2e2"
)
DEFAULT_REPORT = Path("data/results/camera_capture_failure_rebuild.json")


ACTION_LABELS_BY_SECTION = {
    "2": ["确认拍照失败现象并收集故障时段日志"],
    "2.1.1.1": ["检查相机SDK版本"],
    "2.1.1.2": [
        "检查日志中的相机丢事件或事件ACK异常",
        "按相机型号匹配工具并升级固件",
    ],
    "2.2.1": ["检查相机网口参数", "按基线配置相机网口参数"],
    "2.2.2.1": ["取消其他网卡的GigE Vision Filter"],
    "2.3.1": ["检查工控机电源模式", "设置卓越性能并关闭网卡节能"],
    "2.4.1": ["检查系统日志中的网卡过热和网络断链"],
    "2.4.2": ["检查网卡驱动和系统驱动状态"],
    "2.4.3.1": [
        "确认网卡类型和更换条件",
        "更换M.2网卡",
        "恢复相机及运控IP并复查网络日志",
    ],
    "2.4.3.2": [
        "确认网卡类型和更换条件",
        "更换PCI接口网卡",
        "恢复相机及运控IP并复查网络日志",
    ],
    "2.5.2": ["检查网线弯折走线和表面损伤"],
    "2.5.3": ["更换相机网线并按标准重新走线"],
    "2.7.1.1": ["使用DriverStoreExplorer清理旧驱动"],
    "2.7.1.2.2": ["手动删除隐藏或异常设备驱动"],
    "2.7.3": [
        "确认是否为双轨大板特定场景",
        "临时设置主程序和运控进程绑核",
    ],
    "2.9": [
        "执行相机老化至少一小时",
        "执行拍照老化并记录丢帧结果",
        "收集复发日志并升级研发",
    ],
}


def _section_code(item: dict[str, Any]) -> str:
    for offset in item.get("source_offsets") or []:
        match = re.search(r":sec:([^:]+)$", str(offset))
        if match:
            return match.group(1)
    return ""


def _fragment(store: JsonKGV2Store) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    objects = store.objects_by_type
    selected: dict[str, list[dict[str, Any]]] = {key: [] for key in objects}
    selected["FaultFamily"] = [
        deepcopy(item)
        for item in objects["FaultFamily"]
        if str(item.get("family_id") or "") == FAMILY_ID
    ]
    selected["FaultVariant"] = [
        deepcopy(item)
        for item in objects["FaultVariant"]
        if str(item.get("variant_id") or "") == VARIANT_ID
    ]
    for object_type in ("DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace"):
        selected[object_type] = [
            deepcopy(item)
            for item in objects[object_type]
            if str(item.get("variant_id") or "") == VARIANT_ID
        ]
    trace_ids = {
        str(item.get("trace_id") or "") for item in selected["DiagnosticTrace"]
    }
    selected["TraceStep"] = [
        deepcopy(item)
        for item in objects["TraceStep"]
        if str(item.get("trace_id") or "") in trace_ids
    ]
    selected["BranchRule"] = [
        deepcopy(item)
        for item in objects["BranchRule"]
        if str(item.get("trace_id") or "") in trace_ids
    ]
    trace_step_ids = {
        str(item.get("trace_step_id") or "") for item in selected["TraceStep"]
    }
    selected["ExecutionObservation"] = [
        deepcopy(item)
        for item in objects["ExecutionObservation"]
        if str(item.get("trace_step_id") or "") in trace_step_ids
    ]
    case_ids = {
        str(item.get("source_case_id") or "")
        for object_type in ("DiagnosticTrace", "TraceStep", "BranchRule", "ActionOutcome")
        for item in selected[object_type]
        if str(item.get("source_case_id") or "")
    }
    selected["SourceCase"] = [
        deepcopy(item)
        for item in objects["SourceCase"]
        if str(item.get("case_id") or "") in case_ids
    ]
    evidence_ids = {
        str(evidence_id)
        for object_type in (
            "DiagnosticTrace",
            "TraceStep",
            "BranchRule",
            "ActionOutcome",
            "RequiredInfoSpec",
        )
        for item in selected[object_type]
        for evidence_id in item.get("evidence_ids") or []
        if str(evidence_id)
    }
    selected["EvidenceItem"] = [
        deepcopy(item)
        for item in objects["EvidenceItem"]
        if str(item.get("evidence_id") or "") in evidence_ids
    ]
    # Policies are recomputed from the merged graph after publication; carrying
    # the isolated policy would make its inputs incomplete once field cases are
    # migrated into the same family.
    selected["DecisionPolicy"] = []
    selected_ids = {
        str(item.get(V2_PRIMARY_KEYS[object_type]) or "")
        for object_type, items in selected.items()
        for item in items
    }
    relations = [
        deepcopy(item)
        for item in store.relations
        if str(item.get("from") or "") in selected_ids
        and str(item.get("to") or "") in selected_ids
    ]
    return selected, relations


def _legacy_coarse_ids(store: JsonKGV2Store) -> set[str]:
    objects = store.objects_by_type
    removed = {LEGACY_COARSE_VARIANT_ID}
    for object_type in ("DiagnosticAction", "ActionOutcome", "RequiredInfoSpec", "DiagnosticTrace"):
        pk = V2_PRIMARY_KEYS[object_type]
        removed.update(
            str(item.get(pk) or "")
            for item in objects[object_type]
            if str(item.get("variant_id") or "") == LEGACY_COARSE_VARIANT_ID
        )
    trace_ids = {
        str(item.get("trace_id") or "")
        for item in objects["DiagnosticTrace"]
        if str(item.get("variant_id") or "") == LEGACY_COARSE_VARIANT_ID
    }
    for object_type in ("TraceStep", "BranchRule"):
        pk = V2_PRIMARY_KEYS[object_type]
        removed.update(
            str(item.get(pk) or "")
            for item in objects[object_type]
            if str(item.get("trace_id") or "") in trace_ids
        )
    trace_step_ids = {
        str(item.get("trace_step_id") or "")
        for item in objects["TraceStep"]
        if str(item.get("trace_id") or "") in trace_ids
    }
    removed.update(
        str(item.get("observation_id") or "")
        for item in objects["ExecutionObservation"]
        if str(item.get("trace_step_id") or "") in trace_step_ids
    )

    candidate_cases = {
        str(edge.get("from") or "")
        for edge in store.relations
        if str(edge.get("relation") or "") == "supports"
        and str(edge.get("to") or "") in removed
    }
    for case_id in candidate_cases:
        retained_supports = [
            edge
            for edge in store.relations
            if str(edge.get("from") or "") == case_id
            and str(edge.get("relation") or "") == "supports"
            and str(edge.get("to") or "") not in removed
        ]
        if not retained_supports:
            removed.add(case_id)
    candidate_evidence = {
        str(edge.get("from") or "")
        for edge in store.relations
        if str(edge.get("relation") or "") == "evidences"
        and str(edge.get("to") or "") in removed
    }
    for evidence_id in candidate_evidence:
        retained_evidence = [
            edge
            for edge in store.relations
            if str(edge.get("from") or "") == evidence_id
            and str(edge.get("relation") or "") == "evidences"
            and str(edge.get("to") or "") not in removed
        ]
        if not retained_evidence:
            removed.add(evidence_id)
    return {item for item in removed if item}


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in relations:
        key = (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("relation") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        result.append({"from": key[0], "to": key[1], "relation": key[2]})
    return result


def _merge(
    active: JsonKGV2Store,
    fragment_objects: dict[str, list[dict[str, Any]]],
    fragment_relations: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    objects = {
        key: [deepcopy(item) for item in value]
        for key, value in active.objects_by_type.items()
    }
    relations = [deepcopy(item) for item in active.relations]
    removed_ids = _legacy_coarse_ids(active)

    for object_type, items in objects.items():
        pk = V2_PRIMARY_KEYS[object_type]
        migrated: list[dict[str, Any]] = []
        for item in items:
            object_id = str(item.get(pk) or "")
            if object_id in removed_ids:
                continue
            if object_type == "FaultFamily" and object_id == LEGACY_CASE_FAMILY_ID:
                continue
            cloned = deepcopy(item)
            if str(cloned.get("family_id") or "") == LEGACY_CASE_FAMILY_ID:
                cloned["family_id"] = FAMILY_ID
            migrated.append(cloned)
        objects[object_type] = migrated

    old_policy_ids = {
        str(item.get("policy_id") or "")
        for item in objects["DecisionPolicy"]
        if str(item.get("policy_id") or "")
    }
    objects["DecisionPolicy"] = []
    relations = [
        {
            **item,
            "from": FAMILY_ID
            if str(item.get("from") or "") == LEGACY_CASE_FAMILY_ID
            else str(item.get("from") or ""),
            "to": FAMILY_ID
            if str(item.get("to") or "") == LEGACY_CASE_FAMILY_ID
            else str(item.get("to") or ""),
        }
        for item in relations
        if str(item.get("from") or "") not in removed_ids
        and str(item.get("to") or "") not in removed_ids
        and str(item.get("from") or "") not in old_policy_ids
        and str(item.get("to") or "") not in old_policy_ids
        and str(item.get("relation") or "") != "for_family"
    ]

    for object_type, incoming in fragment_objects.items():
        if object_type == "DecisionPolicy":
            continue
        pk = V2_PRIMARY_KEYS[object_type]
        incoming_by_id = {
            str(item.get(pk) or ""): deepcopy(item)
            for item in incoming
            if str(item.get(pk) or "")
        }
        objects[object_type] = [
            item
            for item in objects[object_type]
            if str(item.get(pk) or "") not in incoming_by_id
        ]
        objects[object_type].extend(incoming_by_id.values())
    relations.extend(fragment_relations)

    sections = [
        item
        for item in objects["KnowledgeSection"]
        if str(item.get("document_id") or "") == DOCUMENT_ID
    ]
    section_by_code = {
        _section_code(item): item for item in sections if _section_code(item)
    }
    for code, section in section_by_code.items():
        parent_code = code.rsplit(".", 1)[0] if "." in code else ""
        parent = section_by_code.get(parent_code)
        if parent is not None:
            section["parent_section_id"] = str(parent.get("section_id") or "")
            relations.append({
                "from": str(parent.get("section_id") or ""),
                "to": str(section.get("section_id") or ""),
                "relation": "has_subsection",
            })
        relations.extend([
            {
                "from": str(section.get("section_id") or ""),
                "to": FAMILY_ID,
                "relation": "applicable_to",
            },
            {
                "from": str(section.get("section_id") or ""),
                "to": VARIANT_ID,
                "relation": "describes_variant",
            },
        ])

    action_id_by_label = {
        str(item.get("label") or ""): str(item.get("action_id") or "")
        for item in objects["DiagnosticAction"]
        if str(item.get("variant_id") or "") == VARIANT_ID
    }
    section_code_by_id = {
        str(item.get("section_id") or ""): _section_code(item) for item in sections
    }
    for step in objects["ProcedureStep"]:
        code = section_code_by_id.get(str(step.get("section_id") or ""), "")
        for label in ACTION_LABELS_BY_SECTION.get(code, []):
            action_id = action_id_by_label.get(label)
            if action_id:
                relations.append({
                    "from": str(step.get("procedure_step_id") or ""),
                    "to": action_id,
                    "relation": "candidate_action",
                })

    relations = _dedupe_relations(relations)
    report = {
        "removed_legacy_coarse_ids": sorted(removed_ids),
        "migrated_family_id": {
            "from": LEGACY_CASE_FAMILY_ID,
            "to": FAMILY_ID,
        },
        "document_section_count": len(sections),
        "subsection_relation_count": sum(
            1
            for item in relations
            if str(item.get("relation") or "") == "has_subsection"
            and str(item.get("to") or "") in {
                str(section.get("section_id") or "") for section in sections
            }
        ),
        "procedure_action_relation_count": sum(
            1
            for item in relations
            if str(item.get("relation") or "") == "candidate_action"
            and str(item.get("to") or "") in set(action_id_by_label.values())
        ),
    }
    return objects, relations, report


def rebuild(
    *,
    kg_root: Path,
    build_root: Path,
    gold_root: Path,
    report_path: Path,
    apply: bool,
    sag_paths: list[Path],
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="camera-capture-failure-") as tmp:
        isolated_root = Path(tmp) / "kg_v2"
        isolated_summary = Path(tmp) / "summary.json"
        build_graph(
            target_root=isolated_root,
            build_root=build_root,
            gold_root=gold_root,
            summary_out=isolated_summary,
        )
        fragment_objects, fragment_relations = _fragment(JsonKGV2Store(isolated_root))

    active = JsonKGV2Store(kg_root)
    before_counts = {
        key: len(value) for key, value in active.objects_by_type.items()
    }
    objects, relations, merge_report = _merge(
        active, fragment_objects, fragment_relations
    )
    issues = validate_graph(objects, relations, schema_root=kg_root / "schema")
    if issues:
        raise RuntimeError("proposed graph invalid: " + "; ".join(issues[:30]))

    result: dict[str, Any] = {
        "schema_version": "debug_agent_system.camera_capture_failure_rebuild.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "kg_root": str(kg_root),
        "family_id": FAMILY_ID,
        "variant_id": VARIANT_ID,
        "before_counts": before_counts,
        "proposed_counts": {key: len(value) for key, value in objects.items()},
        "proposed_relation_count": len(relations),
        "validation": {"status": "valid", "issues": []},
        "fragment_counts": {
            key: len(value) for key, value in fragment_objects.items() if value
        },
        "fragment_relation_count": len(fragment_relations),
        **merge_report,
    }
    if apply:
        write_result = active.replace_graph(objects, relations, validate=True)
        pipeline = WriteSideV2Pipeline(kg_root)
        materialized = pipeline.materialize_execution()
        sag_results = [
            pipeline.build_sqlite_sag(path, reset=True) for path in sag_paths
        ]
        final_validation = pipeline.validate_current_graph()
        if final_validation.get("status") != "valid":
            raise RuntimeError(
                "published graph invalid: "
                + "; ".join(final_validation.get("issues") or [])
            )
        result.update({
            "write_result": write_result,
            "materialized": materialized,
            "sag_results": sag_results,
            "final_validation": final_validation,
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
    parser.add_argument(
        "--build-root", type=Path, default=Path("data/kg_v2_sop_draft_build")
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=Path("data/annotations/goldcases/gold-v1"),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--sag-path",
        action="append",
        type=Path,
        default=None,
        help="SAG path to publish; may be supplied more than once.",
    )
    args = parser.parse_args(argv)
    result = rebuild(
        kg_root=args.kg_root,
        build_root=args.build_root,
        gold_root=args.gold_root,
        report_path=args.report,
        apply=args.apply,
        sag_paths=args.sag_path
        or [
            Path("data/kg_v2_sag/debug_agent_v2.sqlite"),
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
