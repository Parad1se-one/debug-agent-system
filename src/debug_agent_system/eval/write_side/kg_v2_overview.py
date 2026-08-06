from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_snapshot(
    *,
    kg_v2_root: str | Path = "data/kg_v2",
    pinned_run_dir: str | Path = "data/results/w2_native_v2_full_pinned_20260708_010455",
) -> dict[str, Any]:
    kg_root = Path(kg_v2_root)
    run_root = Path(pinned_run_dir)
    objects_root = kg_root / "objects"
    mat_root = kg_root / "materialized_execution"

    object_files = {
        "FaultFamily": "fault_families.json",
        "FaultVariant": "fault_variants.json",
        "DiagnosticAction": "diagnostic_actions.json",
        "ActionOutcome": "action_outcomes.json",
        "RequiredInfoSpec": "required_info_specs.json",
        "DiagnosticTrace": "diagnostic_traces.json",
        "DecisionPolicy": "decision_policies.json",
        "EvidenceItem": "evidence_items.json",
        "SourceCase": "source_cases.json",
    }
    objects = {k: _load_json(objects_root / v, []) for k, v in object_files.items()}

    materialized_counts: dict[str, int] = {}
    for p in sorted((mat_root / "instances").rglob("*.json")):
        data = _load_json(p, [])
        materialized_counts[str(p.relative_to(mat_root))] = len(data) if isinstance(data, list) else 0
    materialized_edges = _load_json(mat_root / "edges.json", [])

    families = [x for x in objects["FaultFamily"] if isinstance(x, dict)]
    variants = [x for x in objects["FaultVariant"] if isinstance(x, dict)]
    actions = [x for x in objects["DiagnosticAction"] if isinstance(x, dict)]
    outcomes = [x for x in objects["ActionOutcome"] if isinstance(x, dict)]
    reqs = [x for x in objects["RequiredInfoSpec"] if isinstance(x, dict)]
    traces = [x for x in objects["DiagnosticTrace"] if isinstance(x, dict)]
    policies = [x for x in objects["DecisionPolicy"] if isinstance(x, dict)]
    source_cases = [x for x in objects["SourceCase"] if isinstance(x, dict)]

    families_by_id = {str(x.get("family_id") or ""): x for x in families if x.get("family_id")}
    actions_by_id = {str(x.get("action_id") or ""): x for x in actions if x.get("action_id")}
    source_cases_by_id = {str(x.get("case_id") or ""): x for x in source_cases if x.get("case_id")}
    evidence_by_id = {str(x.get("evidence_id") or ""): x for x in objects["EvidenceItem"] if isinstance(x, dict) and x.get("evidence_id")}

    variants_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcomes_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reqs_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    policies_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcomes_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reqs_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in variants:
        variants_by_family[str(item.get("family_id") or "")].append(item)
    for item in actions:
        actions_by_family[str(item.get("family_id") or "")].append(item)
        actions_by_variant[str(item.get("variant_id") or "")].append(item)
    for item in outcomes:
        outcomes_by_family[str(item.get("family_id") or "")].append(item)
        outcomes_by_variant[str(item.get("variant_id") or "")].append(item)
    for item in reqs:
        reqs_by_family[str(item.get("family_id") or "")].append(item)
        reqs_by_variant[str(item.get("variant_id") or "")].append(item)
    for item in traces:
        traces_by_family[str(item.get("family_id") or "")].append(item)
        traces_by_variant[str(item.get("variant_id") or "")].append(item)
    for item in policies:
        policies_by_family[str(item.get("family_id") or "")].append(item)

    queue_counts_by_label: dict[str, dict[str, int]] = defaultdict(lambda: {"candidates": 0, "merge_candidates": 0, "noise_candidates": 0})
    review_root = run_root / "review_queue_v2"
    for queue_name in ("candidates.json", "merge_candidates.json", "noise_candidates.json"):
        queue = _load_json(review_root / queue_name, [])
        logical = queue_name[:-5]
        for item in queue:
            if not isinstance(item, dict):
                continue
            cand = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
            bundle_objs = cand.get("objects") if isinstance(cand.get("objects"), dict) else {}
            family_objs = [x for x in bundle_objs.get("FaultFamily") or [] if isinstance(x, dict)]
            if family_objs:
                queue_counts_by_label[str(family_objs[0].get("label") or "")][logical] += 1

    pinned_summary = _load_json(run_root / "summary.json", {})
    postrun = _load_json(run_root / "postrun_report.json", {})
    downstream_summary = _load_json(run_root / "downstream_summary.json", {})

    family_rows = []
    for family in sorted(families, key=lambda x: (-len(variants_by_family.get(str(x.get("family_id") or ""), [])), str(x.get("label") or ""))):
        family_id = str(family.get("family_id") or "")
        family_variants = variants_by_family.get(family_id, [])
        family_actions = actions_by_family.get(family_id, [])
        family_outcomes = outcomes_by_family.get(family_id, [])
        family_reqs = reqs_by_family.get(family_id, [])
        family_traces = traces_by_family.get(family_id, [])
        family_policies = policies_by_family.get(family_id, [])
        queue_counts = dict(queue_counts_by_label.get(str(family.get("label") or ""), {"candidates": 0, "merge_candidates": 0, "noise_candidates": 0}))

        quality_status = "usable"
        if family.get("label") in {"主程序/系统异常", "算法/程序调优异常"} and len(family_variants) >= 150:
            quality_status = "needs_normalization"
        elif len(family_variants) >= 80 or queue_counts["noise_candidates"] > 0:
            quality_status = "mixed"

        outcome_type_counts = Counter(str(x.get("outcome_type") or "") for x in family_outcomes)

        def build_action_rows(
            scoped_actions: list[dict[str, Any]],
            scoped_outcomes: list[dict[str, Any]],
            scoped_traces: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            action_rows_by_label: dict[str, dict[str, Any]] = {}
            for action in scoped_actions:
                label = str(action.get("label") or "")
                if not label:
                    continue
                row = action_rows_by_label.setdefault(label, {
                    "label": label,
                    "action_role": str(action.get("action_role") or ""),
                    "action_count": 0,
                    "trace_hits": 0,
                    "high_cost": False,
                    "destructive": False,
                    "sample_summaries": [],
                    "outcome_counts": Counter(),
                    "source_section_id": str(action.get("source_section_id") or ""),
                })
                row["action_count"] += 1
                row["high_cost"] = row["high_cost"] or bool(action.get("high_cost"))
                row["destructive"] = row["destructive"] or bool(action.get("destructive"))
                summary = str(action.get("summary") or "")
                if summary and summary not in row["sample_summaries"] and len(row["sample_summaries"]) < 3:
                    row["sample_summaries"].append(summary)
            for outcome in scoped_outcomes:
                action = actions_by_id.get(str(outcome.get("action_id") or ""), {})
                label = str(action.get("label") or outcome.get("summary") or "")
                if not label:
                    continue
                row = action_rows_by_label.setdefault(label, {
                    "label": label,
                    "action_role": str(action.get("action_role") or ""),
                    "action_count": 0,
                    "trace_hits": 0,
                    "high_cost": False,
                    "destructive": False,
                    "sample_summaries": [],
                    "outcome_counts": Counter(),
                    "source_section_id": str(action.get("source_section_id") or ""),
                })
                row["outcome_counts"][str(outcome.get("outcome_type") or "")] += 1
                row["high_cost"] = row["high_cost"] or bool(outcome.get("high_cost"))
                row["destructive"] = row["destructive"] or bool(outcome.get("destructive"))
            for trace in scoped_traces:
                for action_id in trace.get("recommended_action_ids") or []:
                    action = actions_by_id.get(str(action_id), {})
                    label = str(action.get("label") or "")
                    if label in action_rows_by_label:
                        action_rows_by_label[label]["trace_hits"] += 1

            action_rows = []
            for row in action_rows_by_label.values():
                action_rows.append({
                    "label": row["label"],
                    "action_role": row["action_role"],
                    "action_count": row["action_count"],
                    "trace_hits": row["trace_hits"],
                    "high_cost": row["high_cost"],
                    "destructive": row["destructive"],
                    "outcome_counts": dict(row["outcome_counts"]),
                    "sample_summaries": row["sample_summaries"],
                    "source_section_id": row["source_section_id"],
                })
            action_rows.sort(key=lambda x: (-sum(x["outcome_counts"].values()), -x["trace_hits"], x["label"]))
            return action_rows

        action_rows = build_action_rows(family_actions, family_outcomes, family_traces)

        seen_trace_signatures = set()
        trace_rows = []
        for trace in family_traces:
            labels = [str((actions_by_id.get(str(aid), {}) or {}).get("label") or aid) for aid in trace.get("recommended_action_ids") or []]
            labels = [x for x in labels if x]
            sig = tuple(labels[:10])
            if sig in seen_trace_signatures:
                continue
            seen_trace_signatures.add(sig)
            case_obj = source_cases_by_id.get(str(trace.get("source_case_id") or ""), {})
            evidence_rows = []
            for evidence_id in trace.get("evidence_ids") or []:
                evidence = evidence_by_id.get(str(evidence_id), {})
                evidence_rows.append({
                    "evidence_id": str(evidence.get("evidence_id") or evidence_id),
                    "title": str(evidence.get("title") or ""),
                    "source_kind": str(evidence.get("source_kind") or ""),
                    "external_id": str(evidence.get("external_id") or ""),
                    "payload_ref": str(evidence.get("payload_ref") or ""),
                })
            trace_rows.append({
                "trace_id": str(trace.get("trace_id") or ""),
                "summary": str(trace.get("summary") or ""),
                "source_case_title": str(case_obj.get("title") or ""),
                "source_kind": str(case_obj.get("source_kind") or ""),
                "source_ref": str(case_obj.get("source_ref") or ""),
                "source_doc": str(case_obj.get("source_doc") or case_obj.get("payload_ref") or ""),
                "recommended_actions": labels[:12],
                "actual_actions": [str((actions_by_id.get(str(aid), {}) or {}).get("label") or aid) for aid in trace.get("actual_action_ids") or []][:12],
                "evidence_count": len(trace.get("evidence_ids") or []),
                "evidences": evidence_rows[:8],
            })
            if len(trace_rows) >= 12:
                break

        req_counter = Counter((str(x.get("slot") or ""), str(x.get("question") or "")) for x in family_reqs)
        req_rows = []
        for (slot, question), c in req_counter.most_common(20):
            sample = next((x for x in family_reqs if str(x.get("slot") or "") == slot and str(x.get("question") or "") == question), {})
            req_rows.append({
                "slot": slot,
                "question": question,
                "priority": str(sample.get("priority") or ""),
                "why_required": str(sample.get("why_required") or ""),
                "count": c,
            })

        policy_rows = []
        for policy in family_policies[:4]:
            policy_rows.append({
                "policy_id": str(policy.get("policy_id") or ""),
                "ordered_action_ids": [str(x) for x in policy.get("ordered_action_ids") or []][:12],
                "ineffective_action_ids": [str(x) for x in policy.get("ineffective_action_ids") or []][:12],
                "high_cost_action_ids": [str(x) for x in policy.get("high_cost_action_ids") or []][:12],
                "source_trace_count": len(policy.get("source_trace_ids") or []),
                "source_outcome_count": len(policy.get("source_outcome_ids") or []),
            })

        variant_rows = []
        for variant in family_variants:
            variant_id = str(variant.get("variant_id") or "")
            variant_actions = actions_by_variant.get(variant_id, [])
            variant_outcomes = outcomes_by_variant.get(variant_id, [])
            variant_traces = traces_by_variant.get(variant_id, [])
            variant_reqs = reqs_by_variant.get(variant_id, [])
            variant_action_rows = build_action_rows(variant_actions, variant_outcomes, variant_traces)
            variant_outcome_counts = Counter(str(x.get("outcome_type") or "") for x in variant_outcomes)
            variant_req_counter = Counter((str(x.get("slot") or ""), str(x.get("question") or "")) for x in variant_reqs)
            variant_req_rows = []
            for (slot, question), c in variant_req_counter.most_common(12):
                sample = next((x for x in variant_reqs if str(x.get("slot") or "") == slot and str(x.get("question") or "") == question), {})
                variant_req_rows.append({
                    "slot": slot,
                    "question": question,
                    "priority": str(sample.get("priority") or ""),
                    "why_required": str(sample.get("why_required") or ""),
                    "count": c,
                })
            variant_trace_rows = [t for t in trace_rows if str(t.get("trace_id") or "") in {str(x.get("trace_id") or "") for x in variant_traces}]
            variant_rows.append({
                "variant_id": variant_id,
                "label": str(variant.get("label") or ""),
                "summary": str(variant.get("summary") or ""),
                "error_phase": str(variant.get("error_phase") or ""),
                "owner_context": str(variant.get("owner_context") or ""),
                "escalation_target": str(variant.get("escalation_target") or ""),
                "keywords": [str(x) for x in (variant.get("keywords") or [])[:12]],
                "action_count": len(variant_actions),
                "outcome_count": len(variant_outcomes),
                "trace_count": len(variant_traces),
                "required_info_count": len(variant_reqs),
                "outcome_type_counts": dict(variant_outcome_counts),
                "actions": variant_action_rows,
                "required_info": variant_req_rows,
                "representative_traces": variant_trace_rows[:6],
            })
        variant_rows.sort(key=lambda x: (-x["outcome_count"], -x["trace_count"], -x["action_count"], x["label"]))

        family_rows.append({
            "family_id": family_id,
            "label": str(family.get("label") or ""),
            "summary": str(family.get("summary") or ""),
            "category": str(family.get("category") or ""),
            "subsystem": str(family.get("subsystem") or "(empty)"),
            "scenario": str(family.get("scenario") or ""),
            "source_kind": str(family.get("source_kind") or ""),
            "variant_count": len(family_variants),
            "trace_count": len(family_traces),
            "outcome_count": len(family_outcomes),
            "action_count": len(family_actions),
            "required_info_count": len(family_reqs),
            "policy_count": len(family_policies),
            "queue_counts": queue_counts,
            "quality_status": quality_status,
            "outcome_type_counts": dict(outcome_type_counts),
            "actions": action_rows,
            "variants": variant_rows,
            "representative_traces": trace_rows,
            "required_info": req_rows,
            "policies": policy_rows,
            "sample_variants": [
                {"label": str(v.get("label") or ""), "summary": str(v.get("summary") or ""), "error_phase": str(v.get("error_phase") or "")}
                for v in family_variants[:10]
            ],
        })

    return {
        "title": "KG v2 Diagnostic Overview",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kg_v2_root": str(kg_root),
        "pinned_run_dir": str(run_root),
        "stats": {
            "object_counts": {k: len(v) for k, v in objects.items()},
            "materialized_counts": materialized_counts,
            "materialized_edge_count": len(materialized_edges) if isinstance(materialized_edges, list) else 0,
            "pinned_w2_summary": pinned_summary,
            "pinned_quality_gate": postrun.get("quality_gate") or {},
            "downstream_summary": downstream_summary,
            "family_count": len(families),
            "variant_count": len(variants),
        },
        "categories": sorted({str(x.get("category") or "") for x in families if str(x.get("category") or "")}),
        "subsystems": sorted({str(x.get("subsystem") or "(empty)") for x in family_rows}),
        "families": family_rows,
    }


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KG v2 Diagnostic Overview</title>
  <style>
    :root {
      --bg:#0b1020; --bg-soft:#10182b; --panel:#121c31; --panel-raised:#17243d;
      --line:#263754; --line-soft:#1c2a44; --text:#eef4ff; --muted:#94a5c3;
      --blue:#6ea8fe; --blue-soft:#203d6d; --cyan:#54d6d2; --green:#65d391;
      --orange:#f0b35b; --red:#ff7d83; --shadow:0 18px 45px rgba(2,8,23,.24);
    }
    * { box-sizing:border-box; }
    html { background:var(--bg); }
    body { margin:0; background:radial-gradient(circle at 85% -20%,#203a66 0,transparent 38%),var(--bg); color:var(--text); font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif; letter-spacing:.01em; }
    header { position:sticky; top:0; z-index:30; background:rgba(11,16,32,.92); backdrop-filter:blur(18px); border-bottom:1px solid var(--line); padding:18px clamp(16px,3vw,36px) 16px; }
    .eyebrow { display:flex; align-items:center; gap:8px; color:var(--cyan); text-transform:uppercase; letter-spacing:.14em; font-size:10px; font-weight:800; }
    .eyebrow::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 0 4px rgba(84,214,210,.12); }
    h1 { margin:7px 0 5px; font-size:clamp(21px,2.5vw,30px); letter-spacing:-.03em; }
    h2,h3 { margin-top:0; letter-spacing:-.02em; }
    h2 { font-size:18px; } h3 { font-size:15px; }
    .sub { color:var(--muted); font-size:13px; line-height:1.65; }
    .toolbar { display:grid; grid-template-columns:minmax(240px,2fr) repeat(4,minmax(130px,1fr)) auto; gap:9px; margin-top:17px; }
    input,select,button { width:100%; background:rgba(18,28,49,.86); color:var(--text); border:1px solid var(--line); border-radius:10px; padding:10px 12px; font-size:12px; outline:none; transition:border-color .18s,box-shadow .18s,background .18s; }
    input::placeholder { color:#6f809d; }
    input:focus,select:focus,button:focus-visible { border-color:var(--blue); box-shadow:0 0 0 3px rgba(110,168,254,.14); }
    button { cursor:pointer; width:auto; color:#dce9ff; background:var(--blue-soft); border-color:#315996; font-weight:700; white-space:nowrap; }
    button:hover { background:#2a528f; }
    main { display:grid; grid-template-columns:minmax(315px,360px) minmax(0,1fr); min-height:calc(100vh - 157px); }
    aside { border-right:1px solid var(--line); background:rgba(10,17,33,.72); overflow:hidden; display:flex; flex-direction:column; }
    .side-top { padding:16px; border-bottom:1px solid var(--line); background:rgba(16,24,43,.65); }
    .side-label { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
    .metric-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }
    .metric { min-height:70px; background:linear-gradient(150deg,rgba(23,36,61,.9),rgba(17,27,47,.9)); border:1px solid var(--line-soft); border-radius:11px; padding:10px; }
    .metric .k { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
    .metric .v { font-size:20px; line-height:1; font-weight:800; margin-top:10px; color:#fff; }
    .family-list { overflow:auto; padding:12px; scrollbar-color:#304a71 transparent; }
    .family-item { position:relative; border:1px solid var(--line-soft); background:rgba(18,28,49,.74); border-radius:12px; padding:13px 13px 12px 16px; margin-bottom:9px; cursor:pointer; transition:transform .16s,border-color .16s,background .16s; }
    .family-item::before { content:""; position:absolute; left:0; top:12px; bottom:12px; width:3px; border-radius:3px; background:#395783; }
    .family-item:hover { transform:translateY(-1px); border-color:#3c5e8d; background:#172742; }
    .family-item.active { border-color:var(--blue); background:linear-gradient(135deg,#1b3155,#15233c); box-shadow:0 8px 24px rgba(25,69,130,.18); }
    .family-item.active::before { background:var(--cyan); }
    .family-title { font-size:14px; font-weight:800; line-height:1.4; margin-bottom:9px; color:#f6f9ff; }
    .meta-row { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:7px; }
    .badge { display:inline-flex; align-items:center; gap:5px; border:1px solid #2a3d5d; border-radius:999px; padding:3px 8px; font-size:10px; color:#a7b8d2; background:rgba(7,14,29,.45); }
    .badge.ok { color:var(--green); border-color:rgba(101,211,145,.28); background:rgba(46,119,76,.12); }
    .badge.warn { color:var(--orange); border-color:rgba(240,179,91,.28); background:rgba(133,87,24,.12); }
    .badge.bad { color:var(--red); border-color:rgba(255,125,131,.3); background:rgba(127,39,47,.14); }
    section { min-width:0; overflow:auto; padding:clamp(18px,3vw,36px); background:linear-gradient(180deg,rgba(14,23,42,.22),transparent 300px); }
    .panel { background:rgba(18,28,49,.84); border:1px solid var(--line); border-radius:16px; padding:20px; margin-bottom:16px; box-shadow:0 8px 28px rgba(2,8,23,.1); }
    .panel > h2,.panel > h3 { display:flex; align-items:center; gap:9px; margin-bottom:15px; }
    .panel > h2::before,.panel > h3::before { content:""; width:4px; height:18px; border-radius:4px; background:var(--blue); }
    .kv { display:grid; grid-template-columns:145px 1fr; gap:10px 18px; font-size:12px; line-height:1.55; }
    .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
    .trace-card,.req-card { border:1px solid var(--line-soft); border-radius:12px; background:rgba(9,16,31,.52); padding:13px; margin-bottom:12px; }
    .variant-card { position:relative; border:1px solid #2c456a; border-radius:15px; background:linear-gradient(145deg,rgba(24,42,70,.72),rgba(13,23,42,.82)); padding:18px; margin-bottom:14px; overflow:hidden; }
    .variant-card::after { content:"VARIANT"; position:absolute; top:13px; right:16px; color:rgba(120,160,220,.35); font-size:9px; font-weight:900; letter-spacing:.16em; }
    .action-card { border:1px solid var(--line-soft); border-radius:11px; background:rgba(9,16,31,.62); padding:13px 14px; margin:9px 0; }
    .action-card:hover { border-color:#385b89; }
    .title-sm { font-size:13px; font-weight:800; line-height:1.5; margin-bottom:7px; color:#edf4ff; }
    .summary-sm { color:#b5c4da; font-size:12px; line-height:1.65; margin-bottom:10px; }
    .action-heading { display:flex; align-items:flex-start; gap:10px; }
    .action-index { display:grid; place-items:center; flex:0 0 24px; height:24px; border-radius:8px; color:#bfe0ff; background:#1d3d69; font-size:11px; font-weight:800; }
    .mini-list { margin:0; padding-left:18px; font-size:12px; line-height:1.7; color:#c7d4e7; }
    .empty { color:var(--muted); font-size:13px; padding:18px; border:1px dashed var(--line); border-radius:10px; text-align:center; }
    @media (max-width:1200px){ .toolbar{grid-template-columns:1fr 1fr;} main{grid-template-columns:1fr;} aside{border-right:0;border-bottom:1px solid var(--line);max-height:48vh;} section{padding:20px;} }
    @media (max-width:650px){ .toolbar{grid-template-columns:1fr;} .metric-grid{grid-template-columns:repeat(3,1fr);} .kv{grid-template-columns:1fr;gap:3px;} .kv .k:not(:first-child){margin-top:8px;} }
  </style>
</head>
<body>
<header>
  <div class="eyebrow">Knowledge graph / diagnostic workbench</div>
  <h1>KG v2 Diagnostic Overview</h1>
  <div class="sub">按 <b>FaultFamily → FaultVariant → Action</b> 的层级展示当前 kg_v2；底层判定字段保留在数据中，本视图暂不展开。</div>
  <div class="toolbar">
    <input id="searchInput" placeholder="搜索 family / variant / action / required info 关键词" />
    <select id="categoryFilter"></select>
    <select id="subsystemFilter"></select>
    <select id="qualityFilter"></select>
    <select id="sortFilter"></select>
    <button id="resetBtn">重置筛选</button>
  </div>
</header>
<main>
  <aside>
    <div class="side-top"><div class="side-label"><span>Graph navigator</span><span id="familyCountLabel"></span></div><div class="metric-grid" id="metrics"></div></div>
    <div class="family-list" id="familyList"></div>
  </aside>
  <section>
    <div class="panel"><h2>当前快照摘要</h2><div id="summaryText" class="sub">加载中…</div></div>
    <div id="detailRoot"></div>
  </section>
</main>
<script>
let RAW=null, filteredFamilies=[], selectedFamily=null;
fetch('kg_v2_overview_snapshot.json').then(r=>r.json()).then(data=>{RAW=data;initControls();applyFilters();}).catch(err=>{document.getElementById('summaryText').textContent='加载失败：'+err;});
function fillSelect(id, options){const el=document.getElementById(id); el.innerHTML=options.map(opt=>`<option value="${escapeHtml(opt[0])}">${escapeHtml(opt[1])}</option>`).join('');}
function initControls(){
  fillSelect('categoryFilter', [['ALL','全部 category'], ...RAW.categories.map(x=>[x,x])]);
  fillSelect('subsystemFilter', [['ALL','全部 subsystem'], ...RAW.subsystems.map(x=>[x,x])]);
  fillSelect('qualityFilter', [['ALL','全部质量状态'], ['usable','usable'], ['mixed','mixed'], ['needs_normalization','needs_normalization']]);
  fillSelect('sortFilter', [['review_desc','按 review backlog 降序'], ['trace_desc','按 trace 数降序'], ['family_asc','按 family A→Z'], ['variant_desc','按 variant 数降序']]);
  ['searchInput','categoryFilter','subsystemFilter','qualityFilter','sortFilter'].forEach(id=>{ document.getElementById(id).addEventListener('input', applyFilters); document.getElementById(id).addEventListener('change', applyFilters); });
  document.getElementById('resetBtn').onclick=()=>{ document.getElementById('searchInput').value=''; document.getElementById('categoryFilter').value='ALL'; document.getElementById('subsystemFilter').value='ALL'; document.getElementById('qualityFilter').value='ALL'; document.getElementById('sortFilter').value='review_desc'; applyFilters(); };
}
function applyFilters(){
  const q=document.getElementById('searchInput').value.trim().toLowerCase();
  const category=document.getElementById('categoryFilter').value;
  const subsystem=document.getElementById('subsystemFilter').value;
  const quality=document.getElementById('qualityFilter').value;
  const sortMode=document.getElementById('sortFilter').value;
  filteredFamilies = RAW.families.filter(f=>{
    if(category!=='ALL' && f.category!==category) return false;
    if(subsystem!=='ALL' && f.subsystem!==subsystem) return false;
    if(quality!=='ALL' && f.quality_status!==quality) return false;
    if(!q) return true;
    const hay=[f.label,f.summary,f.category,f.subsystem,...(f.actions||[]).flatMap(a=>[a.label,...(a.sample_summaries||[])]),...(f.required_info||[]).flatMap(r=>[r.slot,r.question,r.why_required]),...(f.representative_traces||[]).flatMap(t=>[t.summary,t.source_case_title,...(t.recommended_actions||[])]),...(f.variants||[]).flatMap(v=>[v.label,v.summary,v.error_phase,...(v.actions||[]).flatMap(a=>[a.label,...(a.sample_summaries||[])]),...(v.required_info||[]).flatMap(r=>[r.slot,r.question,r.why_required])])].join(' | ').toLowerCase();
    return hay.includes(q);
  });
  filteredFamilies.sort((a,b)=>{
    const aReview=(a.queue_counts.merge_candidates||0)+(a.queue_counts.noise_candidates||0)+(a.queue_counts.candidates||0);
    const bReview=(b.queue_counts.merge_candidates||0)+(b.queue_counts.noise_candidates||0)+(b.queue_counts.candidates||0);
    if(sortMode==='review_desc') return bReview-aReview || b.trace_count-a.trace_count;
    if(sortMode==='trace_desc') return b.trace_count-a.trace_count || b.variant_count-a.variant_count;
    if(sortMode==='variant_desc') return b.variant_count-a.variant_count || a.label.localeCompare(b.label,'zh-CN');
    if(sortMode==='family_asc') return a.label.localeCompare(b.label,'zh-CN');
    return 0;
  });
  if(!selectedFamily || !filteredFamilies.find(f=>f.family_id===selectedFamily.family_id)) selectedFamily=filteredFamilies[0]||null;
  renderMetrics(); renderSummary(); renderFamilyList(); renderDetail();
}
function renderMetrics(){
  const s=RAW.stats;
  document.getElementById('metrics').innerHTML=[['families',s.family_count],['variants',s.variant_count],['traces',s.object_counts.DiagnosticTrace],['policies',s.object_counts.DecisionPolicy],['solutions',s.materialized_counts['instances/solutions/solutions.json']||0]].map(([k,v])=>`<div class="metric"><div class="k">${escapeHtml(String(k))}</div><div class="v">${escapeHtml(String(v))}</div></div>`).join('');
  document.getElementById('familyCountLabel').textContent=`${filteredFamilies.length} / ${s.family_count}`;
}
function renderSummary(){
  const s=RAW.stats, gate=s.pinned_quality_gate||{};
  document.getElementById('summaryText').innerHTML=`当前主图共有 <b>${s.family_count}</b> 个 family、<b>${s.variant_count}</b> 个 variant、<b>${s.object_counts.DiagnosticTrace}</b> 条 trace、<b>${s.object_counts.DecisionPolicy}</b> 个 policy、<b>${s.materialized_counts['instances/solutions/solutions.json']||0}</b> 个 solution。<br>pinned W2 quality gate: <b>${escapeHtml(String(gate.status||'unknown'))}</b>；当前 v2 review backlog：merge <b>${s.downstream_summary.v2_review_summary.merge_candidates}</b> / noise <b>${s.downstream_summary.v2_review_summary.noise_candidates}</b> / candidate <b>${s.downstream_summary.v2_review_summary.candidates}</b>。`;
}
function renderFamilyList(){
  const root=document.getElementById('familyList');
  if(!filteredFamilies.length){ root.innerHTML='<div class="empty">没有符合筛选条件的 family。</div>'; return; }
  root.innerHTML=filteredFamilies.map(f=>{
    const active=selectedFamily&&selectedFamily.family_id===f.family_id?'active':'';
    const reviewBacklog=(f.queue_counts.merge_candidates||0)+(f.queue_counts.noise_candidates||0)+(f.queue_counts.candidates||0);
    const qualityCls=f.quality_status==='usable'?'ok':(f.quality_status==='mixed'?'warn':'bad');
    return `<div class="family-item ${active}" data-id="${escapeHtml(f.family_id)}"><div class="family-title">${escapeHtml(f.label)}</div><div class="meta-row"><span class="badge">${escapeHtml(f.category)}</span><span class="badge">${escapeHtml(f.subsystem)}</span><span class="badge ${qualityCls}">${escapeHtml(f.quality_status)}</span></div><div class="meta-row"><span class="badge">trace ${f.trace_count}</span><span class="badge">req ${f.required_info_count}</span><span class="badge">review ${reviewBacklog}</span></div><div class="sub">${escapeHtml(f.summary||'')}</div></div>`;
  }).join('');
  root.querySelectorAll('.family-item').forEach(el=>el.onclick=()=>{ selectedFamily=filteredFamilies.find(f=>f.family_id===el.dataset.id)||null; renderFamilyList(); renderDetail(); });
}
function renderDetail(){
  const root=document.getElementById('detailRoot');
  if(!selectedFamily){ root.innerHTML='<div class="empty">请选择一个 family。</div>'; return; }
  const f=selectedFamily;
  root.innerHTML=`
    <div class="panel"><h2>${escapeHtml(f.label)}</h2><div class="kv">
      <div class="k">family_id</div><div>${escapeHtml(f.family_id)}</div>
      <div class="k">category</div><div>${escapeHtml(f.category)}</div>
      <div class="k">subsystem</div><div>${escapeHtml(f.subsystem)}</div>
      <div class="k">summary</div><div>${escapeHtml(f.summary)}</div>
      <div class="k">trace_count</div><div>${f.trace_count}</div>
      <div class="k">policy_count</div><div>${f.policy_count}</div>
      <div class="k">required_info_count</div><div>${f.required_info_count}</div>
      <div class="k">variant_count</div><div>${f.variant_count}</div>
      <div class="k">queue_counts</div><div>candidate ${f.queue_counts.candidates||0} / merge ${f.queue_counts.merge_candidates||0} / noise ${f.queue_counts.noise_candidates||0}</div>
    </div></div>
    <div class="panel"><h3>FaultVariant 视图</h3>${f.variants.length ? f.variants.map(v=>`<div class="variant-card"><div class="title-sm">${escapeHtml(v.label)}</div><div class="summary-sm">${escapeHtml(v.summary||'')}</div><div class="meta-row"><span class="badge">phase ${escapeHtml(v.error_phase||'(empty)')}</span>${v.escalation_target ? `<span class="badge">owner ${escapeHtml(v.escalation_target)}</span>` : ''}<span class="badge">${v.action_count} actions</span></div>${v.keywords && v.keywords.length ? `<div class="summary-sm">关键词：${v.keywords.map(escapeHtml).join(' / ')}</div>` : ''}<div><div class="k">actions</div>${v.actions.length ? v.actions.map(a=>`<div class="action-card"><div class="title-sm">${escapeHtml(a.label)}</div><div class="meta-row"><span class="badge">${escapeHtml(a.action_role||'')}</span>${a.source_section_id ? `<span class="badge">section ${escapeHtml(a.source_section_id)}</span>` : ''}<span class="badge">${a.action_count} instances</span>${a.high_cost?'<span class="badge bad">high_cost</span>':''}${a.destructive?'<span class="badge bad">destructive</span>':''}</div><div class="summary-sm">${(a.sample_summaries||[]).map(escapeHtml).join('； ')}</div></div>`).join('') : '<div class="empty">暂无 action。</div>'}</div></div>`).join('') : '<div class="empty">暂无 variant。</div>'}</div>
    <div class="panel"><h3>Family 级聚合视图</h3>${f.actions.length ? f.actions.slice(0,30).map(a=>`<div class="action-card"><div class="title-sm">${escapeHtml(a.label)}</div><div class="meta-row"><span class="badge">${escapeHtml(a.action_role||'')}</span><span class="badge">trace_hits ${a.trace_hits}</span><span class="badge">${a.action_count} instances</span>${a.high_cost?'<span class="badge bad">high_cost</span>':''}${a.destructive?'<span class="badge bad">destructive</span>':''}</div><div class="summary-sm">${(a.sample_summaries||[]).map(escapeHtml).join('； ')}</div></div>`).join('') : '<div class="empty">暂无 family 级 action 聚合。</div>'}</div>
  `;
}
function escapeHtml(text){return String(text??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
</script>
</body>
</html>
'''


def write_overview(*, kg_v2_root: str | Path = "data/kg_v2", pinned_run_dir: str | Path = "data/results/w2_native_v2_full_pinned_20260708_010455", snapshot_out: str | Path = "data/results/kg_v2_overview_snapshot.json", html_out: str | Path = "data/results/kg_v2_overview.html") -> dict[str, Any]:
    snapshot = build_snapshot(kg_v2_root=kg_v2_root, pinned_run_dir=pinned_run_dir)
    snapshot_path = Path(snapshot_out)
    html_path = Path(html_out)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(HTML_TEMPLATE, encoding="utf-8")
    return {"snapshot_out": str(snapshot_path), "html_out": str(html_path), "family_count": snapshot["stats"]["family_count"], "variant_count": snapshot["stats"]["variant_count"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-v2-root", default="data/kg_v2")
    parser.add_argument("--pinned-run-dir", default="data/results/w2_native_v2_full_pinned_20260708_010455")
    parser.add_argument("--snapshot-out", default="data/results/kg_v2_overview_snapshot.json")
    parser.add_argument("--html-out", default="data/results/kg_v2_overview.html")
    args = parser.parse_args(argv)
    out = write_overview(kg_v2_root=args.kg_v2_root, pinned_run_dir=args.pinned_run_dir, snapshot_out=args.snapshot_out, html_out=args.html_out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
