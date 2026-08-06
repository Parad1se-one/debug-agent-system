# TOOL-JIRA jira_parser

- type: Worker / Tool
- responsibility: parse Jira issue keys, URL metadata, and offline title hints from text or link dicts.
- inputs: text string or dict containing `url/label/text/content`.
- outputs: `JiraParseResult` with `issue_keys`, `urls`, `issue_summaries`, `title_hints`, `version_hints`, `site_hints`, `summary_hint`, `fetched=false`.
- non_goals: no network fetch, no credential handling, no Jira state mutation, no KG writes.
- failure_modes: non-Jira text returns empty `issue_keys`, still metadata-only.
- observability: `observability.agent_id=TOOL-JIRA`.
- strategy_validity: safe if it remains offline unless a future approved adapter explicitly provides credentials/API policy.

## Offline title parsing

真实飞书样本中的 Jira 信息多为 Markdown 链接标题，例如：

```text
[[SMTAOITS-1234] 1.3.5 客户02 设备报错“应用异常”，之后闪退 - Jira](https://jira.example.com/browse/SMTAOITS-1234)
```

`TOOL-JIRA` 不抓取 Jira 页面，但会从链接标题中提取：

- `issue_summaries[].title`: `1.3.5 客户02 设备报错“应用异常”，之后闪退`
- `version_hints`: `["1.3.5"]`
- `site_hints`: `["客户02"]`

这些只是 evidence hints；W2 可用它们补充候选语义，是否入图仍走 W4/W6。
