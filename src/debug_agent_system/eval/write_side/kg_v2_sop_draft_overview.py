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
    families = [x for x in _load_json(obj_root / "fault_families.json", []) if isinstance(x, dict)]
    variants = [x for x in _load_json(obj_root / "fault_variants.json", []) if isinstance(x, dict)]
    actions = [x for x in _load_json(obj_root / "diagnostic_actions.json", []) if isinstance(x, dict)]
    reqs = [x for x in _load_json(obj_root / "required_info_specs.json", []) if isinstance(x, dict)]
    traces = [x for x in _load_json(obj_root / "diagnostic_traces.json", []) if isinstance(x, dict)]
    cases = [x for x in _load_json(obj_root / "source_cases.json", []) if isinstance(x, dict)]

    actions_by_id = {str(x.get("action_id") or ""): x for x in actions if x.get("action_id")}
    cases_by_id = {str(x.get("case_id") or ""): x for x in cases if x.get("case_id")}
    variants_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reqs_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in variants:
        variants_by_family[str(item.get("family_id") or "")].append(item)
    for item in actions:
        actions_by_family[str(item.get("family_id") or "")].append(item)
    for item in reqs:
        reqs_by_family[str(item.get("family_id") or "")].append(item)
    for item in traces:
        traces_by_family[str(item.get("family_id") or "")].append(item)

    family_rows = []
    for family in sorted(families, key=lambda x: (-len(variants_by_family.get(str(x.get("family_id") or ""), [])), str(x.get("label") or ""))):
        family_id = str(family.get("family_id") or "")
        family_variants = variants_by_family.get(family_id, [])
        family_actions = actions_by_family.get(family_id, [])
        family_reqs = reqs_by_family.get(family_id, [])
        family_traces = traces_by_family.get(family_id, [])

        action_counter = Counter(str(x.get("label") or "") for x in family_actions if str(x.get("label") or ""))
        req_counter = Counter((str(x.get("slot") or ""), str(x.get("question") or "")) for x in family_reqs)
        sample_cases = []
        for variant in family_variants[:12]:
            owner_context = str(variant.get("owner_context") or "")
            case_obj = next((x for x in cases if owner_context and owner_context.endswith(str(x.get("source_ref") or ""))), None)
            sample_cases.append({
                "label": str(variant.get("label") or ""),
                "summary": str(variant.get("summary") or ""),
                "error_phase": str(variant.get("error_phase") or ""),
                "source_case_title": str((case_obj or {}).get("title") or ""),
            })

        rep_traces = []
        seen_signatures = set()
        for trace in family_traces:
            labels = [str((actions_by_id.get(str(aid), {}) or {}).get("label") or aid) for aid in trace.get("recommended_action_ids") or []]
            labels = [x for x in labels if x]
            sig = tuple(labels[:8])
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            case_obj = cases_by_id.get(str(trace.get("source_case_id") or ""), {})
            rep_traces.append({
                "trace_id": str(trace.get("trace_id") or ""),
                "summary": str(trace.get("summary") or ""),
                "source_case_title": str(case_obj.get("title") or ""),
                "recommended_actions": labels[:10],
                "actual_actions": [str((actions_by_id.get(str(aid), {}) or {}).get("label") or aid) for aid in trace.get("actual_action_ids") or []][:10],
            })
            if len(rep_traces) >= 10:
                break

        family_rows.append({
            "family_id": family_id,
            "label": str(family.get("label") or ""),
            "summary": str(family.get("summary") or ""),
            "category": str(family.get("category") or ""),
            "subsystem": str(family.get("subsystem") or "(empty)"),
            "scenario": str(family.get("scenario") or ""),
            "sop_case_count": len(family_variants),
            "action_count": len(family_actions),
            "required_info_count": len(family_reqs),
            "trace_count": len(family_traces),
            "top_actions": [{"label": k, "count": v} for k, v in action_counter.most_common(20)],
            "required_info": [
                {"slot": slot, "question": question, "count": count}
                for (slot, question), count in req_counter.most_common(20)
            ],
            "representative_traces": rep_traces,
            "sample_cases": sample_cases,
        })

    return {
        "title": "KG v2 SOP Draft Overview",
        "kg_root": str(root),
        "stats": {
            "family_count": len(families),
            "sop_case_count": len(variants),
            "action_count": len(actions),
            "required_info_count": len(reqs),
            "trace_count": len(traces),
        },
        "categories": sorted({str(x.get("category") or "") for x in families if str(x.get("category") or "")}),
        "subsystems": sorted({str(x.get("subsystem") or "(empty)") for x in family_rows}),
        "families": family_rows,
    }


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KG v2 SOP Draft Overview</title>
  <style>
    :root { --bg:#0d1117; --panel:#161b22; --panel2:#0f141b; --line:#30363d; --text:#e6edf3; --muted:#8b949e; --blue:#58a6ff; --green:#3fb950; --orange:#d29922; }
    *{box-sizing:border-box;} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
    header{position:sticky;top:0;z-index:30;background:rgba(13,17,23,.96);border-bottom:1px solid var(--line);padding:16px 18px 14px;}
    h1{margin:0 0 6px;font-size:22px;} .sub{color:var(--muted);font-size:13px;line-height:1.5;}
    .toolbar{display:grid;grid-template-columns:2fr 1fr 1fr 1fr auto;gap:10px;margin-top:14px;}
    input,select,button{width:100%;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13px;outline:none;} button{cursor:pointer;width:auto;}
    main{display:grid;grid-template-columns:420px 1fr;min-height:calc(100vh - 104px);} aside{border-right:1px solid var(--line);background:var(--panel2);overflow:hidden;display:flex;flex-direction:column;}
    .side-top{padding:14px;border-bottom:1px solid var(--line);} .metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
    .metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px;} .metric .k{font-size:12px;color:var(--muted);} .metric .v{font-size:18px;font-weight:700;margin-top:4px;}
    .family-list{overflow:auto;padding:8px;} .family-item{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:12px;margin-bottom:8px;cursor:pointer;}
    .family-item:hover,.family-item.active{border-color:var(--blue);background:#17263d;} .family-title{font-size:14px;font-weight:700;line-height:1.4;margin-bottom:8px;}
    .meta-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px;} .badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:3px 9px;font-size:11px;color:var(--muted);background:#0d1117;}
    section{overflow:auto;padding:18px;} .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px;}
    .kv{display:grid;grid-template-columns:150px 1fr;gap:8px 12px;font-size:13px;} .k{color:var(--muted);}
    .trace-card,.action-card,.req-card,.case-card{border:1px solid var(--line);border-radius:12px;background:#0d1117;padding:12px;margin-bottom:12px;}
    .title-sm{font-size:14px;font-weight:700;line-height:1.5;margin-bottom:6px;} .summary-sm{color:#c9d1d9;font-size:12px;line-height:1.55;margin-bottom:10px;}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;} .mini-list{margin:0;padding-left:18px;font-size:12px;line-height:1.6;color:#c9d1d9;} .empty{color:var(--muted);font-size:13px;padding:18px;}
    @media (max-width:1200px){ .toolbar{grid-template-columns:1fr 1fr;} main{grid-template-columns:1fr;} aside{border-right:0;border-bottom:1px solid var(--line);max-height:46vh;} }
  </style>
</head>
<body>
<header>
  <h1>KG v2 SOP Draft Overview</h1>
  <div class="sub">只展示 <b>SOP 底稿提取结果</b>。不包含 W2 结果、review queue、outcome 结果层。主视角是：<b>family → SOP 条目 → 标准排查链 → required info</b>。</div>
  <div class="toolbar">
    <input id="searchInput" placeholder="搜索 family / SOP 条目 / action / required info 关键词" />
    <select id="categoryFilter"></select>
    <select id="subsystemFilter"></select>
    <select id="sortFilter"></select>
    <button id="resetBtn">重置筛选</button>
  </div>
</header>
<main>
  <aside>
    <div class="side-top"><div class="metric-grid" id="metrics"></div></div>
    <div class="family-list" id="familyList"></div>
  </aside>
  <section>
    <div class="panel"><h2>当前快照摘要</h2><div id="summaryText" class="sub">加载中…</div></div>
    <div id="detailRoot"></div>
  </section>
</main>
<script>
let RAW=null, filteredFamilies=[], selectedFamily=null;
fetch('kg_v2_sop_draft_overview_snapshot.json').then(r=>r.json()).then(data=>{RAW=data;initControls();applyFilters();}).catch(err=>{document.getElementById('summaryText').textContent='加载失败：'+err;});
function fillSelect(id, options){const el=document.getElementById(id);el.innerHTML=options.map(opt=>`<option value="${escapeHtml(opt[0])}">${escapeHtml(opt[1])}</option>`).join('');}
function initControls(){
  fillSelect('categoryFilter', [['ALL','全部 category'], ...RAW.categories.map(x=>[x,x])]);
  fillSelect('subsystemFilter', [['ALL','全部 subsystem'], ...RAW.subsystems.map(x=>[x,x])]);
  fillSelect('sortFilter', [['case_desc','按 SOP 条目数降序'], ['trace_desc','按 trace 数降序'], ['family_asc','按 family A→Z']]);
  ['searchInput','categoryFilter','subsystemFilter','sortFilter'].forEach(id=>{document.getElementById(id).addEventListener('input',applyFilters);document.getElementById(id).addEventListener('change',applyFilters);});
  document.getElementById('resetBtn').onclick=()=>{document.getElementById('searchInput').value='';document.getElementById('categoryFilter').value='ALL';document.getElementById('subsystemFilter').value='ALL';document.getElementById('sortFilter').value='case_desc';applyFilters();};
}
function applyFilters(){
  const q=document.getElementById('searchInput').value.trim().toLowerCase();
  const category=document.getElementById('categoryFilter').value;
  const subsystem=document.getElementById('subsystemFilter').value;
  const sortMode=document.getElementById('sortFilter').value;
  filteredFamilies = RAW.families.filter(f=>{
    if(category!=='ALL' && f.category!==category) return false;
    if(subsystem!=='ALL' && f.subsystem!==subsystem) return false;
    if(!q) return true;
    const hay=[f.label,f.summary,f.category,f.subsystem,...(f.sample_cases||[]).flatMap(c=>[c.label,c.summary,c.error_phase]),...(f.top_actions||[]).flatMap(a=>[a.label]),...(f.required_info||[]).flatMap(r=>[r.slot,r.question]),...(f.representative_traces||[]).flatMap(t=>[t.summary,...(t.recommended_actions||[])])].join(' | ').toLowerCase();
    return hay.includes(q);
  });
  filteredFamilies.sort((a,b)=>{
    if(sortMode==='case_desc') return b.sop_case_count-a.sop_case_count || a.label.localeCompare(b.label,'zh-CN');
    if(sortMode==='trace_desc') return b.trace_count-a.trace_count || b.sop_case_count-a.sop_case_count;
    if(sortMode==='family_asc') return a.label.localeCompare(b.label,'zh-CN');
    return 0;
  });
  if(!selectedFamily || !filteredFamilies.find(f=>f.family_id===selectedFamily.family_id)) selectedFamily=filteredFamilies[0]||null;
  renderMetrics(); renderSummary(); renderFamilyList(); renderDetail();
}
function renderMetrics(){
  const s=RAW.stats;
  document.getElementById('metrics').innerHTML=[['families',s.family_count],['SOP cases',s.sop_case_count],['actions',s.action_count],['required_info',s.required_info_count],['traces',s.trace_count]].map(([k,v])=>`<div class="metric"><div class="k">${escapeHtml(String(k))}</div><div class="v">${escapeHtml(String(v))}</div></div>`).join('');
}
function renderSummary(){
  const s=RAW.stats;
  document.getElementById('summaryText').innerHTML=`当前 SOP 底稿共有 <b>${s.family_count}</b> 个 family、<b>${s.sop_case_count}</b> 个 SOP 条目、<b>${s.action_count}</b> 个 action、<b>${s.required_info_count}</b> 个 required_info、<b>${s.trace_count}</b> 条标准排查链。`;
}
function renderFamilyList(){
  const root=document.getElementById('familyList');
  if(!filteredFamilies.length){root.innerHTML='<div class="empty">没有符合筛选条件的 family。</div>';return;}
  root.innerHTML=filteredFamilies.map(f=>{const active=selectedFamily&&selectedFamily.family_id===f.family_id?'active':'';return `<div class="family-item ${active}" data-id="${escapeHtml(f.family_id)}"><div class="family-title">${escapeHtml(f.label)}</div><div class="meta-row"><span class="badge">${escapeHtml(f.category)}</span><span class="badge">${escapeHtml(f.subsystem)}</span></div><div class="meta-row"><span class="badge">cases ${f.sop_case_count}</span><span class="badge">actions ${f.action_count}</span><span class="badge">req ${f.required_info_count}</span><span class="badge">trace ${f.trace_count}</span></div><div class="sub">${escapeHtml(f.summary||'')}</div></div>`;}).join('');
  root.querySelectorAll('.family-item').forEach(el=>el.onclick=()=>{selectedFamily=filteredFamilies.find(f=>f.family_id===el.dataset.id)||null;renderFamilyList();renderDetail();});
}
function renderDetail(){
  const root=document.getElementById('detailRoot');
  if(!selectedFamily){root.innerHTML='<div class="empty">请选择一个 family。</div>';return;}
  const f=selectedFamily;
  root.innerHTML=`
    <div class="panel"><h2>${escapeHtml(f.label)}</h2><div class="kv">
      <div class="k">family_id</div><div>${escapeHtml(f.family_id)}</div>
      <div class="k">category</div><div>${escapeHtml(f.category)}</div>
      <div class="k">subsystem</div><div>${escapeHtml(f.subsystem)}</div>
      <div class="k">summary</div><div>${escapeHtml(f.summary)}</div>
      <div class="k">SOP case count</div><div>${f.sop_case_count}</div>
      <div class="k">action_count</div><div>${f.action_count}</div>
      <div class="k">required_info_count</div><div>${f.required_info_count}</div>
      <div class="k">trace_count</div><div>${f.trace_count}</div>
    </div></div>
    <div class="panel"><h3>代表性 SOP 排查链</h3>${f.representative_traces.length ? f.representative_traces.map(t=>`<div class="trace-card"><div class="title-sm">${escapeHtml(t.source_case_title || t.trace_id)}</div><div class="summary-sm">${escapeHtml(t.summary || '')}</div><div class="grid2"><div><div class="k">recommended</div><ul class="mini-list">${(t.recommended_actions||[]).map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ul></div><div><div class="k">actual</div><ul class="mini-list">${(t.actual_actions||[]).map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ul></div></div></div>`).join('') : '<div class="empty">暂无 trace。</div>'}</div>
    <div class="panel"><h3>常见 SOP 动作</h3>${f.top_actions.length ? f.top_actions.map(a=>`<div class="action-card"><div class="title-sm">${escapeHtml(a.label)}</div><div class="meta-row"><span class="badge">count ${a.count}</span></div></div>`).join('') : '<div class="empty">暂无 action。</div>'}</div>
    <div class="panel"><h3>常见 Required Info</h3>${f.required_info.length ? f.required_info.map(r=>`<div class="req-card"><div class="title-sm">${escapeHtml(r.slot)}</div><div class="summary-sm">${escapeHtml(r.question)}</div><div class="meta-row"><span class="badge">count ${r.count}</span></div></div>`).join('') : '<div class="empty">暂无 required info。</div>'}</div>
    <div class="panel"><h3>SOP 条目样本</h3>${f.sample_cases.length ? f.sample_cases.map(v=>`<div class="case-card"><div class="title-sm">${escapeHtml(v.label)}</div><div class="summary-sm">${escapeHtml(v.summary||'')}</div><div class="meta-row"><span class="badge">phase ${escapeHtml(v.error_phase||'(empty)')}</span></div></div>`).join('') : '<div class="empty">暂无样本。</div>'}</div>
  `;
}
function escapeHtml(text){return String(text??'').replace(/[&<>\"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch]));}
</script>
</body>
</html>
"""


def write_overview(
    *,
    kg_root: str | Path = "data/kg_v2",
    snapshot_out: str | Path = "data/results/kg_v2_sop_draft_overview_snapshot.json",
    html_out: str | Path = "data/results/kg_v2_sop_draft_overview.html",
) -> dict[str, Any]:
    snapshot = build_snapshot(kg_root)
    snapshot_path = Path(snapshot_out)
    html_path = Path(html_out)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(HTML, encoding="utf-8")
    return {
        "snapshot_out": str(snapshot_path),
        "html_out": str(html_path),
        "family_count": snapshot["stats"]["family_count"],
        "sop_case_count": snapshot["stats"]["sop_case_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-root", default="data/kg_v2")
    parser.add_argument("--snapshot-out", default="data/results/kg_v2_sop_draft_overview_snapshot.json")
    parser.add_argument("--html-out", default="data/results/kg_v2_sop_draft_overview.html")
    args = parser.parse_args(argv)
    out = write_overview(kg_root=args.kg_root, snapshot_out=args.snapshot_out, html_out=args.html_out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
