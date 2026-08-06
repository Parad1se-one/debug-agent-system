# KG v2 Gold Case 正式标注模板 v1

> 目标：给人工标注提供统一模板。后续所有 `gold case` 都按这个模板沉淀，避免每条样本各写各的。

---

## 1. 使用原则

1. 一条 `gold case` 只描述一个可审核 case。
2. 如果原始 episode 是混合故障，先 split-case，再分别标注。
3. 标注时优先写：
   - `family`
   - `variant`
   - `actions`
   - `outcomes`
   - `required_info`
   - `trace`
   - `evidence anchors`
4. 不要把长原文直接塞进结构字段；长原文只放在 `source_excerpt`。
5. `verified_fix / mitigation_observed / partial_temporary / pending_validation / diagnostic_method` 必须明确区分。

---

## 2. 推荐文件命名

推荐命名：

- `goldcase-001-用户配置加载失败.md`
- `goldcase-002-相机拍摄失败-网口切换.md`
- `goldcase-003-memory-management-蓝屏.md`

也可以先集中写在一份总表里，再拆文件。

---

## 3. Markdown 标注模板

```md
# goldcase-xxx: 标题

## Meta

- `status`: `draft|reviewed|approved`
- `source_kind`: `chat_case|manual_review|sop`
- `source_episode_id`: `...`
- `source_thread_id`: `...`
- `source_file`: `...`
- `annotator`: `...`
- `reviewer`: `...`

## Source Excerpt

> 原文摘录 1

> 原文摘录 2

## Family

- `label`: `...`
- `summary`: `...`
- `category`: `硬件与运控|算法与程序调优|系统与软件异常`
- `subsystem`: `...`
- `scenario`: `...`
- `why_family_not_variant`: `...`

## Variant

- `label`: `...`
- `summary`: `...`
- `equipment_type`: `...`
- `site`: `...`
- `software_version`: `...`
- `error_phase`: `...`
- `owner_context`: `...`
- `why_variant_not_family`: `...`

## Actions

1. `label`: `...`
   - `action_role`: `inspect|collect|compare|change|verify|observe|escalate`
   - `summary`: `...`
   - `step_order`: `...`
   - `destructive`: `true|false`
   - `high_cost`: `true|false`
   - `evidence_anchor_ids`: `[...]`

## Outcomes

1. `action_label`: `...`
   - `outcome_type`: `verified_fix|ineffective|partial_temporary|mitigation_observed|recurred|pending_validation|diagnostic_method|context_not_root_cause`
   - `summary`: `...`
   - `root_cause_summary`: `...`
   - `high_cost`: `true|false`
   - `destructive`: `true|false`
   - `evidence_anchor_ids`: `[...]`

## Required Info

1. `slot`: `...`
   - `question`: `...`
   - `why_required`: `...`
   - `condition`: `...`
   - `blocks`: `[...]`
   - `priority`: `high|medium|low`
   - `evidence_anchor_ids`: `[...]`

## Trace

- `summary`: `...`
- `recommended_action_labels`: `[...]`
- `actual_action_labels`: `[...]`
- `evidence_anchor_ids`: `[...]`

## Uncertainties

- `...`

## Decision Notes

- `should_enter_gold_set`: `true|false`
- `why`: `...`
```

---

## 4. JSON 契约模板

```json
{
  "schema_version": "kg_v2.gold_case.v1",
  "status": "draft",
  "source_kind": "chat_case",
  "source_episode_id": "",
  "source_thread_id": "",
  "source_file": "",
  "annotator": "",
  "reviewer": "",
  "source_excerpt": [],
  "family": {
    "label": "",
    "summary": "",
    "category": "",
    "subsystem": "",
    "scenario": "",
    "why_family_not_variant": ""
  },
  "variant": {
    "label": "",
    "summary": "",
    "equipment_type": "",
    "site": "",
    "software_version": "",
    "error_phase": "",
    "owner_context": "",
    "why_variant_not_family": ""
  },
  "actions": [],
  "outcomes": [],
  "required_info": [],
  "trace": {
    "summary": "",
    "recommended_action_labels": [],
    "actual_action_labels": [],
    "evidence_anchor_ids": []
  },
  "uncertainties": [],
  "decision_notes": {
    "should_enter_gold_set": true,
    "why": ""
  }
}
```

---

## 5. 标注通过线

一条 case 能进入正式 gold set，最少满足：

1. `family / variant` 层级明确
2. `actions[]` 原子化
3. `outcomes[]` 类型明确
4. `required_info[]` 不泛化
5. `trace` 能从原文中对得上
6. 有明确 `evidence_anchor_ids`

不满足任一条，就继续停留在 `draft`。
