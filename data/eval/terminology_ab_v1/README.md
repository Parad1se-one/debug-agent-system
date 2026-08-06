# KG_v2 读侧术语层 A/B v1

本目录保存术语层的 60 题配对 A/B 测试集。

- `cases.jsonl`：逐题 Query、分层、来源金标和术语安全预期。
- `experiment.json`：唯一变量、冻结项、指标、门禁和失败迭代政策。
- `scripts/run_terminology_ab.py`：成对交错、冻结指纹的执行器。

测试臂：

- A：`terminology_enabled=false`
- B：`terminology_enabled=true`

两组必须使用同一代码快照、模型、Prompt、知识、raw、参数和题序。术语关闭并不关闭
KG_v2 或 raw 检索，只关闭 `TerminologyResolver → search contract → search audit` 这一层。

结构检查：

```bash
python scripts/validate_terminology_ab_dataset.py
python scripts/validate_terminology_ab_dataset.py --check-current-resolver
```

只预览实验计划、不调用模型：

```bash
python scripts/run_terminology_ab.py
```

该套件是术语专项开发集，不替代正式 Debug Benchmark，也不能用它的安全负例反向写入
单题规则。任何术语修改后必须全量重跑 60 题。
