# Draft Story Review — B

2026-10-07。只读审查 `main.tex`、`references.bib`，对照最终结果、已核源码、两轮故事共识及 MemGAS 范文。未修改正文，未增加实验、模型调用或实验结果。下列行号对应本次初读版本；协作中如正文已经更新，请按所引短语定位。

## 结论

科学故事成立，且已完成参考论文的修辞迁移：先交代 conversational memory 的问题，再把 gap 分为逐条事实的 explicit answer usefulness 与压缩后缺失的来源细节，随后按 QFS → SER 执行顺序回应。数值、累计删除逻辑、实验占位约束均通过。没有需要重构主线或增加主实验组数的问题。

只建议先修三处小的准确性问题，再删少量重复句。六个待补实验小节应继续保持空白。

## 必要修复

| 位置与定位短语 | 问题 | 最小修复 |
|---|---|---|
| `main.tex:101`: “if at most two remain, both are retained” | `at most two` 包含 0、1、2，代码保留全部剩余 packets；`both` 仅适用于恰好两个。 | 改为 “if at most two remain, all remaining packets are retained without this ranking call.” |
| `main.tex:125`: “Each condition has one accepted outcome per question.” | 全文 `accepted answers` 指 Judge 为 yes，此处实际上表示有效且已接受的实验产物；容易被读成每个条件每题均正确，与 1,302/1,540 冲突。 | 改为 “Each condition has one valid evaluated outcome per question.” 或 “one valid Reader response and Judge label per question.” |
| `main.tex:122`: “the fixed hybrid pool described in Section~\ref{sec:qfs}” | 混合池构造在 Preliminaries/Appendix Implementation Details，QFS 小节描述的是评分与选取。 | 指向 `Appendix~\ref{app:implementation}`。 |

以上均为文字层面修正，不改变方法或结果。

## 数字与约束核对

来源：`D:/jev mem/research/r03_r05_refinement_2026-10-06/final_result_digest_v1.json`。分别绑定 `original_multi_source_closure`、`r03_original`、`baseline`，先按完整精度相减，再保留三位小数。

| 状态 | Official F1 | Judge (%) | Accepted / N | 初稿 |
|---|---:|---:|---:|---|
| QFS +, SER + | 60.238 | 84.545 | 1302 / 1540 | 正确 |
| QFS +, SER − | 52.547 | 76.818 | 1183 / 1540 | 正确 |
| QFS −, SER − | 51.044 | 74.870 | 1153 / 1540 | 正确 |

| 对比 | F1 差 | Judge 百分点差 | 净答对差 | 初稿 |
|---|---:|---:|---:|---|
| full − QFS only | 7.691 | 7.727 | 119 | 所报数值正确 |
| QFS only − base | 1.503 | 1.948 | 30 | 所报 Judge 差正确；不必为完整性添加未写数值 |
| full − base | 9.194 | 9.675 | 149 | 正确 |

- 主表：恰好四个空 baseline（Mem0、MemGPT、A-MEM、MemGAS），一个有数值的 ours。没有把旧七臂探索搬入主实验。
- 删除表：明确 `(+,+) → (+,−) → (−,−)`，末行 `w/o SER, QFS`；没有把末行误作独立 `w/o QFS`。
- 明确当前 QFS 对比也包含 candidate-pool configuration，SER 增加 source tokens；未声称纯 JEV 因果、同 token 优势或正向协同。
- `Selector Controls`、`Module Decomposition`、`Matched-Token Comparison`、`Generalization`、`Evidence Analysis`、`Hyperparameter Analysis` 六节均只有标题和 label，内容为空。
- `figure` 环境数为 0。
- 固定 1,540 题、十个 conversations、探索后选定、完整请求验证继承与复用均已说明；未误称 blind test 或独立重复。

## 引用核验

9 个正文引用键均存在于 BibTeX，且没有未被正文使用的条目。已打开第一手论文页面或官方文档核对目前正文的概括。当前引用没有支撑超出原文范围的性能结论，也没有把范文性能转移给本方法。

| 引用 | 当前正文用途 | 核验 |
|---|---|---|
| Maharana et al., LoCoMo | 长对话中的时序/分散信息困难；数据集来源 | 支持。[ACL Anthology](https://aclanthology.org/2024.acl-long.747/) |
| Liu et al., Lost in the Middle | 更长输入不保证可靠利用相关信息 | 支持。[ACL Anthology](https://aclanthology.org/2024.tacl-1.9/) |
| Chhikara et al., Mem0 | 抽取、整合与检索长期显著信息 | 支持。[作者预印本](https://arxiv.org/abs/2504.19413) |
| Packer et al., MemGPT | 管理不同 memory tiers | 支持。[作者预印本](https://arxiv.org/abs/2310.08560) |
| Xu et al., A-MEM | 结构化 notes、links、新记忆引发旧记忆表示更新 | 支持。[作者预印本](https://arxiv.org/abs/2502.12110) |
| Xu et al., MemGAS | 多粒度关联、query-dependent routing/filtering | 支持，且正文清楚区分自己的 anchored source scopes。[范文 v2](https://arxiv.org/abs/2505.19549v2) |
| Lewis et al., RAG | 将检索信息与答案生成结合 | 支持。[作者预印本](https://arxiv.org/abs/2005.11401) |
| Jiang et al., Jev-Mem | System-One controller、结构化多关系记忆、adaptive retrieval | 支持。正文已正面承认相近同期工作，并作方法区分。[作者预印本](https://arxiv.org/abs/2609.23986) |
| TypeSafe AI, Jev | typed questions/shared state/structured values/Noul | 支持。官方页介绍 Noul 的 0–1 输出；本论文仍独立限定其为本任务未校准的 operational score，合理。[官方文档](https://docs.typesafe.ai/introduction) |

这次引用核验限于正文所用的技术概括和主要书目信息。所有实验分数仍以本地存档为依据，没有拿文献成绩填入空 baseline。

## 修辞迁移与必要去重

迁移已经完成，体现为：紧凑摘要的 problem → two decisions → two operations → measured result；Introduction 四段依次完成场景、相关方法、两项 gap、方法与证据；Method 用 purpose → input → operation/formula → explanation → downstream output；Experiments 按 settings → main result → component removal 排列；Related Work 在实验后。没有照搬 GMM、entropy router、PPR、跨四 benchmark 或 SOTA 结论。

对 LaTeX 命令作简单清理后，与本地 `exemplar_extracted.txt` 作连续 8 个英文单词的重复检查，未发现匹配。该检查只是辅助证据，范文抽取存在粘连词；人工检查也未发现独特论证内容或句式照搬。

建议只删以下重复内容，保留准确性边界的必要落点：

1. **Main Results 第 128 行末两句**关于 baseline cells 和“不推断排名”的说明，可删除或压成一句；表注已经准确说明空值不是零。不要改动空表格本身。
2. **Related Work 第 197 行**可删 “We neither claim the first use …” 和末句 “Their downstream effects require …”。前文已经引用同期 Jev-Mem 并正面对比其控制/图检索与本方法，足以避免优先权误导；控制实验缺口已经在 Results 和 Limitations 表述。留下方法关系本身会更自然。
3. **Results 第 168 行末句** “These distinctions determine which component claims the current results support.” 可删，前两句已经逐项讲清累计链、候选池和 source-token 差异。这句不再提供新信息。
4. **Introduction 第 38 行末句** “The evaluation thus measures …” 可删；其前一句已给出明确 removal 顺序与两次下降。

不建议继续大幅删除 Method 的 scope、local-judgment 和阈值边界；它们直接限定算法操作，而不只是文风 disclaimer。Limitations 可保持为独立集中说明，不需再把它们复制进空实验节或摘要。

总体建议：按三处必要小修和上述少量去重完成本轮，无需改故事、数字、表格结构或增加结果。
