# O-EG：Evidence Gap Resolver

O-EG 只在确定性 KG_v2 运行时返回 `ask_info` 且调用方已经提供证据资源时工作。

处理顺序为：

`required_data → 有界只读 Parser → source-bound observations → 相关性过滤 → 追加检索上下文`

边界：

- Tool observation 必须绑定 `resource_id`，有消息锚点时同时绑定 `source_message_id`。
- Parser 失败、截断和排除原因保留在统一 Tool Envelope 中。
- 图片 Parser 只读取文件头和尺寸，不做 OCR，不得生成截图文字。
- O-EG 不选择 BranchRule，不生成 DiagnosticAction，不确认高风险动作，也不能产生 `verified_fix`。
- 相同 Tool 与参数用 fingerprint 去重；本地 Resolver 不重复调用。
- 默认最多接收 12 个资源、每个资源读取 64 KiB；配置上限可调整。
