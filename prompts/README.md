# Prompt 快照与调用口径

这些文件由本地源码 AST 提取/原样复制，未调用模型。`manifest.json` 保存来源 commit、源文件、行号和 SHA-256。原始系统 prompt 前后的换行也保留。

| 文件 | 实际作用 |
|---|---|
| `active_additive_system.txt` | 当前普通 `Memory.add(infer=True)` 的系统 prompt：`ADDITIVE_EXTRACTION_PROMPT` |
| `conditional_agent_suffix.txt` | 仅 `agent_id` 存在且 `user_id` 不存在时附加；本次 LoCoMo 按 user_id 分 conversation，不触发 |
| `mem0_prompts_source.py` | 完整原样源码；包含当前 user prompt builder、辅助函数和历史模板 |
| `upstream_locomo_answer_template.txt` | 配套 benchmark 的回答模板，非写入模板 |
| `upstream_locomo_judge_system.txt` | 配套 benchmark 的 judge system prompt |
| `upstream_locomo_judge_template.txt` | 配套 benchmark 的判分模板；宽松规则必须单独披露 |
| `locomo_prompts_source.py` | 完整原样 benchmark prompt 源码，含实际渲染、排序及 gold 预处理逻辑 |

`mem0_prompts_source.py` 的 `FACT_RETRIEVAL_PROMPT`、`USER_MEMORY_EXTRACTION_PROMPT`、`AGENT_MEMORY_EXTRACTION_PROMPT` 与 `DEFAULT_UPDATE_MEMORY_PROMPT` 属于仍保留的旧模板，**不在此版本默认普通 add 调用链中**。不能只按常量名称推断其生效。

## 当前默认写入

核心代码在一次 add 中取最近 10 条历史消息及 top-10 相关旧记忆；把新输入消息整体交给一次生成调用。系统模板中有“up to 20”描述，实际当前调用参数/数据库保留是 10，报告以执行代码为准。历史消息在 user prompt 中每条截断至 300 字符后附省略号；新消息不通过这一历史截断函数。

User prompt 由 `generate_additive_extraction_prompt` 构造，依次含：

```text
## Summary
[当前默认调用未提供，空字符串]
## Last k Messages
[最近 10 条历史消息，各自截断]
## Recently Extracted Memories
[]
## Existing Memories
[当前输入相关的 top-10 旧记忆，ID 在调用内重映射]
## New Messages
[本次新消息]
## Observation Date
[观察日期]
## Current Date
[系统日期]
[可选 Custom Instructions / Language Requirement]
# Output:
```

上面是结构解释，不是新写的替代 prompt；真正原文及构造函数见源码快照。当前 SDK 调用未传日期，两个日期都默认运行当天；不能把解释中的“观察日期”当作 LoCoMo 日期已经成功传入。

系统模板要求只 ADD、从 user 和 assistant 两种角色提取、自足表述、去重、链接相关旧记忆、保留归因与时间。它允许“不值得提取”返回空 memory 数组。但模板要求的 `linked_memory_ids` 未被当前 add 流水线保存/消费，不能称为已实现的 LLM 跨记忆链接；实际关联来自 NLP 实体索引。JEV 的额外作用需要与这些已有能力及实现缺口分别比较。

## 复现边界

当前 benchmark 的 answerer/judge CLI 默认均为 `gpt-5`。模板保留原样不意味着本项目已有该模型访问权限，也不意味着将采用其默认模型。模型配置须在启动前显式锁定。

回答模板含 LoCoMo 特定年份与推断指令；judge 含列表命中一项可判对等宽松规则。将上游口径作为单独复现轨道，不将其命名为严格完整证据正确率；研究轨道需使用固定的严格口径。

Mem0 LICENSE 随快照保留；所有上游材料归原作者。快照文件仅用于审计，不能当作可直接运行的本地 Mem0 替代模块。
