# 续工状态 — 2026-10-12

最新用户决定优先于旧 handoff：HyperMem 已放弃，其专属运行/研究/测试缓存及 Qwen 模型已经删除；不得从旧启动器恢复或将它标成完成。定时器保持关闭。下一步待完成的 baseline 是 **V-Mem、PRISM**，当前提交不启动它们。

A-Mem 和 SimpleMem 均有 1540 个有效 Reader/Judge 结果，accepted summary、逐题数据和独审记录已选入仓库；完整原始证据压缩包仍在原 D 盘。主论文还使用 Gemini 3.8 主实验及 Gemini 2.5 三臂累计消融，分别保留独立来源，不能混成一个实验队列。

## 继续实验前的必要工作

- 仓库续工以本文件和结果索引为准。原工作站 baseline 的 `execution_plan.md` / `execution_handoff.md` 是本地历史参考，本次不整份上传；旧文中的“继续 HyperMem”已被取消决定覆盖。
- 旧顺序 dispatcher 可能仍硬性要求 HyperMem 的完成证明。开始 V-Mem 前应单独更新并核查“取消 HyperMem 后转到 V-Mem”的队列逻辑，不能伪造前序完成标记。
- V-Mem、PRISM 的模型、适配、检索上下文、每会话并发和资源计量仍需沿已审配置检查；不因仓库存在源码而声称已经运行。
- 禁止复用已成功 QA 来择优重抽；保留初遍、技术恢复、继承、真实物理请求和缺失的不同口径。

## 方法与公平性限制

SimpleMem 实际采用 40 条单说话者 utterance 窗口、overlap=2、step=38；155/158 个窗口跨 session，最多包含四个 session，148 个提示还带前窗最多三条记忆摘要。它不是逐 session 隔离写入的结果。用户曾讨论 session 对齐，但没有执行新版本重测。不得悄悄更改旧结果或推断跨 session 必然提高了多少分。

原 A/V/PRISM 配置采用 MiniLM 384 维，SimpleMem 采用 Gemini embedding 768 维。不同原生粒度、上下文预算、encoder/reranker 配置必须随比较披露；不能宣称所有方法处理等量 token。Gemini 2.5 与记录为 Gemini 3.8 的条件也不能直接作为只改变单个组件的因果实验。

API usage 缺失时使用 null，不把未返回的 thoughts 或费用填为零。实验物理调用、传输重试、技术替换、本地 exact-cache 与继承分别统计；provider 的 cachedContentTokenCount 不是本地 cache 次数。并发请求耗时之和不是总墙钟时间，采样 RSS 峰不等于完整运行峰值。历史本地模型 token 缺少实测时只能标注可重建子集，不能补写成全程实测。

## 论文当前进度

`paper_acl2027_2026-10-07/main.tex` 与 PDF 为保留的八页工作稿。已完成 A/Simple 的结果还未填入该稿的 baseline 主表，六个计划小节也仍待实验。论文源文与数值来源保持原样；后续可据本次结果索引更新，不要自行填充 V-Mem/PRISM/HyperMem 分数。
