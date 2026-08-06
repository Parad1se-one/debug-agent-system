from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_snapshot(kg_root: str | Path = "data/kg_v2") -> dict[str, Any]:
    root = Path(kg_root)
    obj_root = root / "objects"
    rel_root = root / "relations"

    families = [x for x in _load_json(obj_root / "fault_families.json", []) if isinstance(x, dict)]
    variants = [x for x in _load_json(obj_root / "fault_variants.json", []) if isinstance(x, dict)]
    actions = [x for x in _load_json(obj_root / "diagnostic_actions.json", []) if isinstance(x, dict)]
    outcomes = [x for x in _load_json(obj_root / "action_outcomes.json", []) if isinstance(x, dict)]
    reqs = [x for x in _load_json(obj_root / "required_info_specs.json", []) if isinstance(x, dict)]
    traces = [x for x in _load_json(obj_root / "diagnostic_traces.json", []) if isinstance(x, dict)]
    policies = [x for x in _load_json(obj_root / "decision_policies.json", []) if isinstance(x, dict)]
    evidences = [x for x in _load_json(obj_root / "evidence_items.json", []) if isinstance(x, dict)]
    cases = [x for x in _load_json(obj_root / "source_cases.json", []) if isinstance(x, dict)]
    edges = [x for x in _load_json(rel_root / "edges.json", []) if isinstance(x, dict)]

    actions_by_id = {str(x.get("action_id") or ""): x for x in actions if x.get("action_id")}
    cases_by_id = {str(x.get("case_id") or ""): x for x in cases if x.get("case_id")}
    evidence_by_id = {str(x.get("evidence_id") or ""): x for x in evidences if x.get("evidence_id")}

    variants_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcomes_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reqs_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    policies_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_ids_by_family: dict[str, set[str]] = defaultdict(set)

    for item in variants:
        variants_by_family[str(item.get("family_id") or "")].append(item)
    for item in actions:
        actions_by_family[str(item.get("family_id") or "")].append(item)
    for item in outcomes:
        outcomes_by_family[str(item.get("family_id") or "")].append(item)
    for item in reqs:
        reqs_by_family[str(item.get("family_id") or "")].append(item)
    for item in traces:
        traces_by_family[str(item.get("family_id") or "")].append(item)
        if item.get("source_case_id"):
            case_ids_by_family[str(item.get("family_id") or "")].add(str(item.get("source_case_id")))
    for item in policies:
        policies_by_family[str(item.get("family_id") or "")].append(item)
    for edge in edges:
        if edge.get("relation") == "supports" and edge.get("to"):
            case_id = str(edge.get("from") or "")
            to_id = str(edge.get("to") or "")
            if to_id.startswith("variant:"):
                variant = next((v for v in variants if str(v.get("variant_id") or "") == to_id), None)
                if variant:
                    case_ids_by_family[str(variant.get("family_id") or "")].add(case_id)

    family_rows: list[dict[str, Any]] = []
    for family in sorted(families, key=lambda x: (-len(variants_by_family.get(str(x.get("family_id") or ""), [])), str(x.get("label") or ""))):
        family_id = str(family.get("family_id") or "")
        family_variants = variants_by_family.get(family_id, [])
        family_actions = actions_by_family.get(family_id, [])
        family_outcomes = outcomes_by_family.get(family_id, [])
        family_reqs = reqs_by_family.get(family_id, [])
        family_traces = traces_by_family.get(family_id, [])
        family_policies = policies_by_family.get(family_id, [])
        family_case_ids = case_ids_by_family.get(family_id, set())
        family_cases = [cases_by_id[cid] for cid in family_case_ids if cid in cases_by_id]

        action_counter = Counter(str(x.get("label") or "") for x in family_actions if str(x.get("label") or ""))
        outcome_counter = Counter(str(x.get("outcome_type") or "") for x in family_outcomes if str(x.get("outcome_type") or ""))
        req_counter = Counter((str(x.get("slot") or ""), str(x.get("question") or "")) for x in family_reqs)

        representative_traces = []
        seen_trace_signatures = set()
        for trace in family_traces:
            labels = [str((actions_by_id.get(str(aid), {}) or {}).get("label") or aid) for aid in trace.get("recommended_action_ids") or []]
            labels = [x for x in labels if x]
            sig = tuple(labels[:10])
            if sig in seen_trace_signatures:
                continue
            seen_trace_signatures.add(sig)
            case_obj = cases_by_id.get(str(trace.get("source_case_id") or ""), {})
            representative_traces.append({
                "trace_id": str(trace.get("trace_id") or ""),
                "summary": str(trace.get("summary") or ""),
                "source_case_title": str(case_obj.get("title") or ""),
                "source_kind": str(case_obj.get("source_kind") or ""),
                "recommended_actions": labels[:12],
                "actual_actions": [str((actions_by_id.get(str(aid), {}) or {}).get("label") or aid) for aid in trace.get("actual_action_ids") or []][:12],
                "evidence_count": len(trace.get("evidence_ids") or []),
            })
            if len(representative_traces) >= 8:
                break

        top_actions = []
        for label, count in action_counter.most_common(16):
            action = next((x for x in family_actions if str(x.get("label") or "") == label), {})
            top_actions.append({
                "label": label,
                "count": count,
                "action_role": str(action.get("action_role") or ""),
                "summary": str(action.get("summary") or ""),
                "high_cost": bool(action.get("high_cost")),
                "destructive": bool(action.get("destructive")),
            })

        required_info_rows = []
        for (slot, question), count in req_counter.most_common(16):
            sample = next((x for x in family_reqs if str(x.get("slot") or "") == slot and str(x.get("question") or "") == question), {})
            required_info_rows.append({
                "slot": slot,
                "question": question,
                "count": count,
                "priority": str(sample.get("priority") or ""),
                "why_required": str(sample.get("why_required") or ""),
            })

        sample_cases = []
        for case in sorted(family_cases, key=lambda x: str(x.get("title") or ""))[:12]:
            sample_cases.append({
                "case_id": str(case.get("case_id") or ""),
                "title": str(case.get("title") or ""),
                "summary": str(case.get("summary") or ""),
                "source_ref": str(case.get("source_ref") or ""),
                "source_kind": str(case.get("source_kind") or ""),
            })

        policy_rows = []
        for policy in family_policies[:4]:
            policy_rows.append({
                "policy_id": str(policy.get("policy_id") or ""),
                "target_error_id": str(policy.get("target_error_id") or ""),
                "ordered_checks": policy.get("ordered_checks") or [],
                "solution_stats": policy.get("solution_stats") or [],
                "unsafe_actions": policy.get("unsafe_actions") or [],
            })

        evidence_titles = []
        for case in family_cases[:8]:
            case_id = str(case.get("case_id") or "")
            supporting = [edge for edge in edges if str(edge.get("from") or "") == case_id and edge.get("relation") == "supports"]
            if supporting:
                evidence_titles.append(str(case.get("title") or case_id))

        family_rows.append({
            "family_id": family_id,
            "label": str(family.get("label") or ""),
            "summary": str(family.get("summary") or ""),
            "category": str(family.get("category") or ""),
            "subsystem": str(family.get("subsystem") or "(empty)"),
            "scenario": str(family.get("scenario") or ""),
            "source_kind": str(family.get("source_kind") or ""),
            "variant_count": len(family_variants),
            "case_count": len(family_cases),
            "action_count": len(family_actions),
            "outcome_count": len(family_outcomes),
            "required_info_count": len(family_reqs),
            "trace_count": len(family_traces),
            "policy_count": len(family_policies),
            "top_actions": top_actions,
            "outcome_type_counts": dict(outcome_counter),
            "required_info": required_info_rows,
            "representative_traces": representative_traces,
            "sample_cases": sample_cases,
            "policies": policy_rows,
            "evidence_titles": evidence_titles,
        })

    return {
        "title": "KG v2 SOP Draft Overview",
        "kg_root": str(root),
        "stats": {
            "family_count": len(families),
            "variant_count": len(variants),
            "case_count": len(cases),
            "action_count": len(actions),
            "outcome_count": len(outcomes),
            "required_info_count": len(reqs),
            "trace_count": len(traces),
            "policy_count": len(policies),
            "relation_count": len(edges),
        },
        "categories": sorted({str(x.get("category") or "") for x in families if str(x.get("category") or "")}),
        "families": family_rows,
    }


def render_html(snapshot: dict[str, Any]) -> str:
    data_json = json.dumps(snapshot, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KG v2 SOP Draft Overview</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6d0c4;
      --nav: #efe8da;
      --accent: #5b4b8a;
      --accent-2: #2f6f5e;
      --accent-3: #8a5b4b;
      --warn: #9f3a38;
      --shadow: 0 8px 24px rgba(41, 37, 36, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: Georgia, "Noto Serif SC", "Source Han Serif SC", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f2ede2 0%, #f8f5ee 100%);
      line-height: 1.7;
    }}
    .layout {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    aside {{
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      overflow: auto;
      padding: 24px 20px 40px;
      background: var(--nav);
      border-right: 1px solid var(--line);
    }}
    main {{ padding: 32px 40px 80px; max-width: 1180px; }}
    h1, h2, h3, h4 {{ margin: 0; font-weight: 700; line-height: 1.3; }}
    h1 {{ font-size: 30px; margin-bottom: 10px; }}
    h2 {{ font-size: 24px; margin-bottom: 18px; }}
    h3 {{ font-size: 19px; margin-bottom: 12px; }}
    h4 {{ font-size: 15px; margin-bottom: 8px; }}
    p {{ margin: 0 0 12px; }}
    .lead {{ color: var(--muted); font-size: 14px; margin-bottom: 22px; }}
    .note {{ padding: 14px 16px; border: 1px solid var(--line); background: #faf7f0; border-radius: 12px; margin-bottom: 18px; color: #4b5563; font-size: 14px; }}
    .toolbar {{ display:flex; flex-direction:column; gap:10px; margin-bottom:18px; }}
    .toolbar input, .toolbar select {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 14px;
      color: var(--ink);
    }}
    .toc-group {{ margin-bottom: 24px; }}
    .toc-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}
    .toc-link {{
      display: block;
      padding: 9px 10px;
      margin-bottom: 6px;
      border-radius: 10px;
      color: var(--ink);
      text-decoration: none;
      background: rgba(255,255,255,0.35);
      border: 1px solid transparent;
      transition: 0.15s ease;
      font-size: 14px;
    }}
    .toc-link:hover {{ background: #fff; border-color: var(--line); }}
    .toc-sub {{ display:block; color: var(--muted); font-size: 12px; margin-top:4px; }}
    .page {{
      background: var(--paper);
      border: 1px solid #e7e0d4;
      box-shadow: var(--shadow);
      border-radius: 18px;
      padding: 28px 28px 32px;
      margin-bottom: 28px;
    }}
    .chip-row {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 18px; }}
    .chip {{
      display:inline-flex;
      align-items:center;
      border:1px solid var(--line);
      background:#fbf8f2;
      border-radius:999px;
      padding:4px 10px;
      font-size:12px;
      color:var(--muted);
    }}
    .chip.family {{ color: var(--accent); }}
    .chip.variant {{ color: var(--accent-2); }}
    .chip.action {{ color: var(--accent-3); }}
    .chip.warn {{ color: var(--warn); }}
    .meta-grid {{ display:grid; grid-template-columns: 160px 1fr; gap:8px 14px; margin-bottom:18px; font-size:14px; }}
    .meta-grid .k {{ color: var(--muted); }}
    .section-gap {{ margin-top: 18px; }}
    .variant-block {{ border-top:1px dashed var(--line); padding-top:18px; margin-top:18px; }}
    .source-quote {{ border-left:3px solid #cbbca3; padding:10px 14px; background:#fbf8f2; color:#4b5563; margin:12px 0; font-size:14px; }}
    .timeline {{ display:grid; gap:12px; }}
    .timeline-item {{
      display:grid;
      grid-template-columns:70px 1fr;
      gap:12px;
      align-items:start;
      padding:12px 14px;
      border:1px solid var(--line);
      border-radius:12px;
      background:#fcfaf5;
    }}
    .step {{ font-size:12px; color:var(--muted); font-weight:700; letter-spacing:0.04em; text-transform:uppercase; padding-top:2px; }}
    .item-title {{ font-size:15px; font-weight:700; margin-bottom:6px; }}
    .item-body {{ font-size:14px; color:#4b5563; }}
    .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .box {{ border:1px solid var(--line); background:#fcfaf5; border-radius:14px; padding:14px 16px; }}
    .list {{ margin:0; padding-left:18px; font-size:14px; }}
    .list li {{ margin:6px 0; }}
    .muted {{ color: var(--muted); }}
    .empty {{ color: var(--muted); font-style: italic; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ position: relative; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      main {{ padding: 24px 18px 60px; }}
      .cols {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="toc-group">
        <div class="toc-title">Draft</div>
        <div class="note">
          这份页面直接读取 canonical <code>data/kg_v2</code> 的结构化内容，内容来自当前写侧 curated build。
        </div>
      </div>
      <div class="toolbar">
        <input id="search" placeholder="搜索 family / action / trace / required info" />
        <select id="category"></select>
      </div>
      <div class="toc-group">
        <div class="toc-title">Families</div>
        <div id="familyToc"></div>
      </div>
    </aside>
    <main>
      <article class="page">
        <h1>KG v2（data/kg_v2）</h1>
        <p class="lead">
          这份可视化展示的是当前 canonical <code>data/kg_v2</code> 目录里的结构化对象。
        </p>
        <div class="meta-grid" id="topMeta"></div>
      </article>
      <div id="familyPages"></div>
    </main>
  </div>
  <script>
    const RAW = {data_json};
    const familyPages = document.getElementById('familyPages');
    const familyToc = document.getElementById('familyToc');
    const topMeta = document.getElementById('topMeta');
    const searchEl = document.getElementById('search');
    const categoryEl = document.getElementById('category');

    function esc(v) {{
      return String(v ?? '').replace(/[&<>\"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}}[ch]));
    }}

    function topMetaHtml() {{
      const s = RAW.stats || {{}};
      const rows = [
        ['family_count', s.family_count],
        ['variant_count', s.variant_count],
        ['case_count', s.case_count],
        ['action_count', s.action_count],
        ['outcome_count', s.outcome_count],
        ['required_info_count', s.required_info_count],
        ['trace_count', s.trace_count],
        ['policy_count', s.policy_count],
        ['relation_count', s.relation_count],
      ];
      return rows.map(([k,v]) => `<div class="k">${{esc(k)}}</div><div>${{esc(v)}}</div>`).join('');
    }}

    function fillCategory() {{
      const options = ['ALL', ...(RAW.categories || [])];
      categoryEl.innerHTML = options.map(x => `<option value="${{esc(x)}}">${{esc(x === 'ALL' ? '全部 category' : x)}}</option>`).join('');
    }}

    function familyHaystack(f) {{
      return [
        f.label, f.summary, f.category, f.subsystem, f.scenario,
        ...(f.top_actions || []).flatMap(x => [x.label, x.summary]),
        ...(f.required_info || []).flatMap(x => [x.slot, x.question, x.why_required]),
        ...(f.representative_traces || []).flatMap(x => [x.summary, ...(x.recommended_actions || [])]),
        ...(f.sample_cases || []).flatMap(x => [x.title, x.summary]),
      ].join(' ').toLowerCase();
    }}

    function filteredFamilies() {{
      const q = (searchEl.value || '').trim().toLowerCase();
      const cat = categoryEl.value || 'ALL';
      return (RAW.families || []).filter(f => {{
        if (cat !== 'ALL' && f.category !== cat) return false;
        if (!q) return true;
        return familyHaystack(f).includes(q);
      }});
    }}

    function renderToc(families) {{
      familyToc.innerHTML = families.map(f => `
        <a class="toc-link" href="#${{esc(f.family_id)}}">
          ${{esc(f.label)}}
          <span class="toc-sub">${{esc(f.category)}} · actions ${{esc(f.action_count)}} · traces ${{esc(f.trace_count)}}</span>
        </a>
      `).join('');
    }}

    function actionList(rows) {{
      if (!rows.length) return '<div class="empty">暂无 action。</div>';
      return rows.map(x => `
        <div class="box">
          <h4>${{esc(x.label)}}</h4>
          <div class="chip-row">
            <span class="chip action">${{esc(x.action_role || 'inspect')}}</span>
            <span class="chip">count ${{esc(x.count)}}</span>
            ${{x.high_cost ? '<span class="chip warn">high_cost</span>' : ''}}
            ${{x.destructive ? '<span class="chip warn">destructive</span>' : ''}}
          </div>
          <p class="muted">${{esc(x.summary || '')}}</p>
        </div>
      `).join('');
    }}

    function reqList(rows) {{
      if (!rows.length) return '<div class="empty">暂无 required info。</div>';
      return rows.map(x => `
        <div class="box">
          <h4>${{esc(x.slot)}}</h4>
          <p>${{esc(x.question)}}</p>
          <p class="muted">${{esc(x.why_required || '')}}</p>
          <div class="chip-row">
            <span class="chip">${{esc(x.priority || '')}}</span>
            <span class="chip">count ${{esc(x.count)}}</span>
          </div>
        </div>
      `).join('');
    }}

    function traceList(rows) {{
      if (!rows.length) return '<div class="empty">暂无 trace。</div>';
      return `<div class="timeline">` + rows.map((t, idx) => `
        <div class="timeline-item">
          <div class="step">Trace ${{idx + 1}}</div>
          <div>
            <div class="item-title">${{esc(t.source_case_title || t.trace_id)}}</div>
            <div class="item-body">${{esc(t.summary || '')}}</div>
            <div class="chip-row">
              <span class="chip">recommended ${{esc((t.recommended_actions || []).length)}}</span>
              <span class="chip">actual ${{esc((t.actual_actions || []).length)}}</span>
              <span class="chip">evidence ${{esc(t.evidence_count)}}</span>
            </div>
            <div class="cols">
              <div class="box">
                <h4>recommended actions</h4>
                <ul class="list">${{(t.recommended_actions || []).map(x => `<li>${{esc(x)}}</li>`).join('')}}</ul>
              </div>
              <div class="box">
                <h4>actual actions</h4>
                <ul class="list">${{(t.actual_actions || []).map(x => `<li>${{esc(x)}}</li>`).join('')}}</ul>
              </div>
            </div>
          </div>
        </div>
      `).join('') + `</div>`;
    }}

    function sampleCaseList(rows) {{
      if (!rows.length) return '<div class="empty">暂无 source cases。</div>';
      return rows.map(x => `
        <div class="box">
          <h4>${{esc(x.title || x.case_id)}}</h4>
          <p class="muted">${{esc(x.summary || '')}}</p>
          <div class="chip-row">
            <span class="chip">${{esc(x.source_kind || '')}}</span>
            <span class="chip">${{esc(x.source_ref || '')}}</span>
          </div>
        </div>
      `).join('');
    }}

    function policyList(rows) {{
      if (!rows.length) return '<div class="empty">暂无 decision policy。</div>';
      return rows.map(x => `
        <div class="box">
          <h4>${{esc(x.policy_id)}}</h4>
          <p class="muted">target_error_id: ${{esc(x.target_error_id)}}</p>
          <div class="source-quote">ordered_checks: ${{esc((x.ordered_checks || []).length)}} · solution_stats: ${{esc((x.solution_stats || []).length)}} · unsafe_actions: ${{esc((x.unsafe_actions || []).length)}}</div>
        </div>
      `).join('');
    }}

    function renderFamilies(families) {{
      familyPages.innerHTML = families.map(f => `
        <article class="page" id="${{esc(f.family_id)}}">
          <h2>FaultFamily：${{esc(f.label)}}</h2>
          <p class="lead">${{esc(f.summary || '')}}</p>
          <div class="chip-row">
            <span class="chip family">${{esc(f.category)}}</span>
            <span class="chip variant">${{esc(f.subsystem)}}</span>
            <span class="chip">variants ${{esc(f.variant_count)}}</span>
            <span class="chip">cases ${{esc(f.case_count)}}</span>
            <span class="chip">actions ${{esc(f.action_count)}}</span>
            <span class="chip">outcomes ${{esc(f.outcome_count)}}</span>
            <span class="chip">req ${{esc(f.required_info_count)}}</span>
            <span class="chip">traces ${{esc(f.trace_count)}}</span>
            <span class="chip">policies ${{esc(f.policy_count)}}</span>
          </div>
          <div class="meta-grid">
            <div class="k">family_id</div><div>${{esc(f.family_id)}}</div>
            <div class="k">scenario</div><div>${{esc(f.scenario || '')}}</div>
            <div class="k">source_kind</div><div>${{esc(f.source_kind || '')}}</div>
            <div class="k">evidence titles</div><div>${{esc((f.evidence_titles || []).join('；'))}}</div>
          </div>

          <div class="section-gap">
            <h3>代表性排查链</h3>
            ${{traceList(f.representative_traces || [])}}
          </div>

          <div class="section-gap cols">
            <div>
              <h3>常见动作</h3>
              ${{actionList(f.top_actions || [])}}
            </div>
            <div>
              <h3>常见 Required Info</h3>
              ${{reqList(f.required_info || [])}}
            </div>
          </div>

          <div class="section-gap">
            <h3>Source Cases</h3>
            ${{sampleCaseList(f.sample_cases || [])}}
          </div>

          <div class="section-gap">
            <h3>Decision Policies</h3>
            ${{policyList(f.policies || [])}}
          </div>
        </article>
      `).join('');
    }}

    function render() {{
      const families = filteredFamilies();
      topMeta.innerHTML = topMetaHtml();
      renderToc(families);
      renderFamilies(families);
    }}

    fillCategory();
    searchEl.addEventListener('input', render);
    categoryEl.addEventListener('change', render);
    render();
  </script>
</body>
</html>"""


def write_overview(
    *,
    kg_root: str | Path = "data/kg_v2",
    snapshot_out: str | Path = "tmp/kg_v2_sop_draft_book_snapshot.json",
    html_out: str | Path = "tmp/kg_v2_sop_draft_book_overview.html",
) -> dict[str, Any]:
    snapshot = build_snapshot(kg_root)
    snapshot_path = Path(snapshot_out)
    html_path = Path(html_out)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(snapshot), encoding="utf-8")
    return {
        "status": "written",
        "kg_root": str(kg_root),
        "snapshot_out": str(snapshot_path),
        "html_out": str(html_path),
        "family_count": snapshot["stats"]["family_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-root", default="data/kg_v2")
    parser.add_argument("--snapshot-out", default="tmp/kg_v2_sop_draft_book_snapshot.json")
    parser.add_argument("--html-out", default="tmp/kg_v2_sop_draft_book_overview.html")
    args = parser.parse_args(argv)
    out = write_overview(kg_root=args.kg_root, snapshot_out=args.snapshot_out, html_out=args.html_out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
