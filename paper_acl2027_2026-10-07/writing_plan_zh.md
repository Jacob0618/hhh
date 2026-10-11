# 第一版写作说明与待补实验

## 本版范围

这是英文 ACL 论文初稿，不是已完成全部验证的投稿版本。参考用户指定的 MemGAS v2 论文，仅迁移段落功能、论证顺序和章节组织，未复用其措辞、算法、理论或实验结论。暂不放任何图。

正文仅以 **原 R03 + 新 R05 + C** 为最终方法。论文命名为两个模块：

- **Query-Conditioned Fact Selection (QFS)**：原 R03，按问题逐条评价候选事实的明确回答价值，再选 20 条并恢复池内顺序。
- **Source Evidence Recovery (SER)**：新 R05 + C 整体，来源锚点、部分支持备选、旧/新窗口比较、邻近证据补全、最终最多两条来源附件。C 是 SER 内部步骤，不是早先的切分 C1/C2。

暂用中性标题 **JEV-Guided Fact Selection and Source Evidence Recovery for Conversational Memory**。项目可继续叫 JEV-Mem，但已有同期论文 *Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents*，本稿 Related Work 正面引用并区分；不主张首次将 JEV 用于记忆。正式品牌标题仍可由作者决定。

## 两轮子 agent 讨论

第一轮 A/B 读参考原文与实际 PDF 页面，分别拆解 Abstract/Introduction 与 Method/Experiments/Related Work，并直接讨论。第二轮 A/B 结合原始代码与冻结结果讨论科学故事、模块边界和证据缺口。四份分析及两组共识在 `discussion/`。

参考论文的主线是：**背景与任务约束 → 现有应对方式 → 两个具体缺陷及后果 → 依同样顺序提出两项机制 → 实验验证**。本稿迁移为：

1. 跨会话问答需要从记忆构造当前答案的证据。
2. 现有外部记忆组织和检索使历史可访问，但“选中哪条事实”与“暴露该事实背后的哪些细节”是两件事。
3. 缺陷一：话题/人名相关不等于包含所问关系、值或列表项。缺陷二：压缩事实和窄窗口可能使原文中的条件、归属、指代或时间关系不可见。
4. QFS 回应第一个问题；SER 回应第二个问题。
5. 主结果只展示最终组合；三行移除链说明整体与阶段差值；未证实机制的实验只留标题。

不把逐事实评分写成集合充分性认证，不把局部来源检索写成全对话独立召回，不把模型提出的“保留证据”判据写成已证明保证。

## 每节如何写

| 部分 | 段落功能与顺序 | 本稿落实 |
|---|---|---|
| Abstract | 背景 → 两个问题 → 总方法 → 模块 1 → 模块 2 → 结果 | 一段闭合问题—方法—证据链，不罗列研发历史 |
| Introduction 第 1 段 | 任务 → 长历史约束 → 外部记忆 → 证据构造需求 | 从跨会话问题进入，避免泛泛夸 LLM |
| Introduction 第 2 段 | 有组织地介绍已有路线 | Mem0、MemGPT、A-MEM、MemGAS，收束到事实与来源两个决策 |
| Introduction 第 3 段 | 每个缺陷紧接其后果 | explicit answer usefulness；source detail omission |
| Introduction 第 4 段 | 解法与缺陷同序 → 有限实验结论 | QFS → SER → 固定 1540 题结果，无额外重复 contributions 列表 |
| Methodology | 定义输入 → 各模块按执行顺序 → 输出 | Preliminaries → QFS → SER，SER 内部按 5 步解释 |
| Experiments | 设置 → 主结果 → 组件移除 → 分析 | 一个 ours 行；四个基线数字空；三行累计消融；六个空小节 |
| Related Work | 与方法相关的路线 → 本文的具体位置 | 记忆管理、粒度/来源、JEV 控制，不复述图网络贡献 |
| Conclusion | 回扣两个问题与解法 → 已测结论 | 不提出未证实 SOTA 或新实验结果 |
| Limitations / Ethics | 研究边界、数据与隐私 | 同批探索、Judge、候选/Token 不同、来源范围、费用未知等集中说明 |
| Appendix | 精确复现口径 | 候选池构造、来源边界、失败规则、F1 与 Judge 版本 |

## 主表与消融数值

主表只填最终方法：官方 F1 **60.238**、Judge **84.545**。Mem0、MemGPT、A-MEM、MemGAS 的主表数值为空，等待用户提供，并在填入前核对相同数据/模型/Reader/Judge/分母，不能把原论文异协议值直接混成受控实验。

| 配置 | QFS | SER | 官方 F1 | Judge (%) | Judge 对/总 |
|---|---:|---:|---:|---:|---:|
| Full | 有 | 有 | 60.238 | 84.545 | 1302/1540 |
| w/o SER | 有 | 无 | 52.547 | 76.818 | 1183/1540 |
| w/o SER, QFS | 无 | 无 | 51.044 | 74.870 | 1153/1540 |

74.870 是两个模块均无的 Mem0，**不能写成只去掉 QFS 而仍保留 SER 的独立消融**。标题使用 *Progressive Component Removal*。两个累计步骤的 Judge 差值分别为 7.727pp 和 1.948pp；整套差值 9.675pp，不等于两个组件各自独立大于 3pp。

## 六个待补实验：正文仅有标题，内容空

以下是补验证计划，不是本次已授权执行的实验；本次仅写稿。

| 空标题 | 要回答的问题 | 最少需要的验证 |
|---|---|---|
| Selector Controls | 提升来自候选池、JEV 排序，还是简单排序也可以？ | 同一冻结候选池与最终 20 条预算下，BM25/原顺序/JEV；来源选择也需同候选的廉价选择器对照。避免把扩大候选池算成纯 JEV 作用。 |
| Module Decomposition | QFS、SER、窗口比较、C 的贡献能否拆开？ | 补真正 QFS 关闭、SER 保留；在同 QFS 下补去掉窗口比较、去掉邻近补全 C。先冻结规则，再生成，不以正确性择优。 |
| Matched-Token Comparison | 原文增加的 token 解释了多少收益？ | 按实际 Reader 输入 token 控制上下文预算，对照同等篇幅的简单原文附加；字节与 20 facts 不等于 token。 |
| Generalization | 同批开发后的效果能否迁移？ | 独立未用会话/数据集；必要时另一固定 Reader。现 10 conv 已接触，不重标盲测。 |
| Evidence Analysis | 改善与退化在哪一环：候选池、最终 20、附件、Reader、Judge？ | 利用 LoCoMo evidence IDs，先定义 fact→turn 的可靠映射；按阶段统计 evidence coverage，并核读完整答案证据。对翻转样本做原文审阅和独立人审/Judge 一致性；不要把 scope 命中当语义支持。 |
| Hyperparameter Analysis | 20 facts、4 anchors、8 windows、2 attachments、阈值 0.5 是否稳健？ | 每次变一个因素，小范围事先固定网格；在独立开发集选择，报告各点而非只挑最好。维持其他 Reader/Judge/数据配置。 |

正文不填写这些小节的实验设计、预期收益或空壳结论。这里只记录故事需要什么证据，待运行并核验后再写。

## 公式清单与代码对应

| 公式 | 实际含义 | 证据 |
|---|---|---|
| (1) | 原 R03 逐事实回答价值评分 | `jev_mem/evidence_selection_r03.py` + `R03_selection_protocol_v2.json` |
| (2) | 按分数选成员、池序呈现 | `choose_jev` / `_decision` |
| (3) | 至少一个原子主张的部分来源支持筛选 | `source_support_stack_v1.py:build_support_plan/resolve_support` |
| (4) | 新 R05 的旧窗口与备选窗口比较后替换 | `r05_refinement_v1.py:build_candidate_variants/resolve_multi` |
| (5) | 在不改事实前缀、ID、日期及顺序的前提下附加最多两条来源 | `source_append_r05.py:selected_facts` |

这些式子是实现的紧凑描述，不是新理论或最优性证明。使用同样的顺序介绍问题、变量、运算与输出；未迁移 MemGAS 的 GMM、熵路由或图检索公式。

## 证据与 claim 检查表

- [x] 最终配置确实是原 R03 + 新 R05 + C，未写入未采用的新 R03 算法。
- [x] 所有三行固定 1540，Reader/Judge 全有效，未伪造未完成实验。
- [x] 完整方法 F1/Judge 与摘要、结果、消融一致。
- [x] 池组成披露：baseline20 + 额外 native≤40 + native80 外全库 BM25≤20 + native 补足；非简单 native80。
- [x] SER 只覆盖原 top4 来源锚点，没有池外全原文独立召回。
- [x] >0.5 是操作门限，非 calibrated truth guarantee；最终 top2 没有绝对门限。
- [x] 同 20 facts 不代表同 token；未声称节省补充次数或成本。
- [x] 原同批数据用于探索，继承请求不是独立重复。
- [x] 主表空白基线不支撑 SOTA 结论；逐步删除不冒充完整 factorial。
- [x] 同期 Jev-Mem 引用并定位，未宣称该方向禁止重叠或首次使用 JEV。
- [ ] 其他 baseline 数值与协议由用户补充后核对。
- [ ] 六个待补验证未执行；目前正文仍为空。
- [ ] 最终作者、署名、资源与人工复核由作者在投稿前确认。

## 正式数据入口

冻结最终审计：`D:/jev mem/research/r03_r05_refinement_2026-10-06/heartbeat_20261006T1052/live_qa_v2/final_summary_v1.json`，SHA256 `01aa05237a252832135f9c03122d16ee8d6f1aab0b0b2e97d16f2836533567e5`。

行映射：Full=`original_multi_source_closure`；w/o SER=`r03_original`；w/o SER,QFS=`baseline`。其他历史结果未加入论文主表。本稿不改变运行配置、原数据或任何已完成实验。
