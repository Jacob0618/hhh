# R11 实际执行记录

2026-10-05，北京时间。工作区D:\jev mem。用户授权准确度优先的一轮，未授权自动无限扩展。新一轮沿用原Writer产物、SDK20、纯文本Gemini3.8 Reader/Judge；没有写入重建。

## 首遍评分已完整结束

- selector：`runs/source_set_r11_v1`。
- manifest SHA：`566a540f6100a93fe0e4447c2f9264fe19f8f4285a74b98566b127afb872236d`。
- protocol SHA：`aa8b8ec9820ef92c7c35629f1064090ef241a01c5a19a9db8199f089b1fdfe47`。
- G 1540 complete；P 1501 complete / 39 failed；全部3080终态，0local errors。
- 初始编排已停止，绝不重启。它在评分结束后保存失败列表时把list传给要求dict的save_new，TypeError只发生于编排报告步骤，未开始QA，评分响应及intent/ledger完整保留。后续入口版本独立，不修改原记录。
- 原严格选择已全部封存：`selected_inputs.json` SHA `0d4070bd52e05a6c5cb4a2cb5b7d1cb430cd2e874c568d8cc1271d16d96890c6`，4620槽。G1540题附两段；P1484题附两段，17个合法最高平局及39失败均保留N0。

## 已审格式兼容派生，未改变原科学主表

Astra与Sol在不读QA/gold的情况下讨论并核查全部1540 P原响应。38个失败满足“两位百分数格点舍入相容”的数学区间，其中37个唯一最高选择可得到原pair，另1个最高平局仍N0；剩余c7_q133的返回choice并非最大概率，拒绝且不替模型选择。

原值（包括0.5700000000000001浮点尾巴）、原choice和原raw SHA均不改变，不归一化，不新调用JEV。这是提供方精度未声明情况下的显式兼容假设，不能认定真实成因已获提供方确认。

- 新纯模块：`jev_mem/source_set_rounding_r11.py`。原strict parser不改。
- 新来源：`runs/source_set_r11_rounding_view_v1`，由 `prepare_source_set_rounding_r11.py` 全量重放派生。
- 严格主结果与兼容敏感性分开。只执行兼容实际输入：严格失败原本N0，直接继承真实N0完整请求响应；严格成功和兼容未改变输入者共享真实同臂响应。`project_source_set_strict_r11.py` 保留真实producer artifact和请求签名证明，绝非另一轮模型运行。
- 各种计数分开：strict_selection_counts不变，effective selection_counts重算，兼容最多新增74个原文附件。
- 25项core/dispatcher/舍入离线tests通过；额外接线测试与独审材料同目录。mock不是效果实验。

## 当前接续

21:16:23启动Hidden `tools/resume_source_set_r11_reviewed_v1.py`，历史launcher34440，实际PID需检查 `R11_live/reviewed_status.json`。旧PID已终止，不能重放旧pipeline。

顺序为：已封存strict来源校验 → 新兼容视图prepare → 新QA prepare → `runs/source_set_rounding_qa_r11_v1`三臂4620槽（baseline/g/p各1540） → 全部结束后统一一次已有技术恢复 → 原raw审计 → strict exact投影 → 严格/兼容五臂比较 → 两位agent真实结果讨论。不再发任何JEV，新增只有必要的G/P Reader/Judge。

日志：`R11_live/reviewed.stdout.log`、`.stderr.log`。健康快照工具 `tools/snapshot_source_set_r11.py`，核真实PID/创建时间/命令、逐conv Reader/Judge/QA与实际API增量。run的来源校验可能几分钟没有API；看实际CPU/文件读取进展，不能凭静态百分比假称卡死或恢复。原阶段意图保护仍启用，失败不可隐式重发。

QA全部沿原纯文本参数、旧函数与统一一次技术恢复规则；合法NO不重判。源校验失败/未知请求/局部IO问题先停审查。凭据仅DPAPI vault，不输出密钥。旧R05及归档保持不变。

结果尚未产生；不宣称G/P已提升，也不算作两个独立成功组件。报告必须包含同完整Reader输入的新旧波动子群和原文候选局限。

## 终态追加：本轮完成，以本节覆盖上述运行中状态

2026-10-05晚，reviewed controller已正常结束：status=complete、stage=awaiting_result_discussion。全部4620槽Reader/Judge/QA首遍complete，0local errors；无需QA技术恢复，未新建恢复运行。两套raw summary均no_invalid_artifacts。严格投影37题使用真实同请求N0，其余同臂生产响应，未再次生成。正式strict/rounded五臂comparison均发布。

严格主表：N0 F1 51.044000/Judge74.870130%；cheap57.675337/81.298701%；旧local58.394349/82.142857%；G58.080987/81.688312%；P58.032092/81.428571%。P舍入兼容敏感性58.279589/81.623377%，不覆盖strict。

G相对local16改善23退化；P strict18/29、compatible17/25。G同完整Reader签名860题中的翻转10/12；P strict737题7/10、compatible754题7/11。全部退化语义审查已由Astra和Sol分别完成，并实际交叉讨论。正式comparison与两位独立数字一致，crosscheck材料同目录。并行只读归档SHA复验71,450个唯一输入all_verified；没有改实验来源或追加API。

结论与下一提案见R11_result_readout.md、R11_results_astra.md/json、R11_results_sol.md/json、R11_comparison_crosscheck_sol_v1.json。保留旧local默认；下一探索为query-independent支持来源定位×query-conditioned跨度补全，保持旧local排名规则及Reader。正贡献需靠后续消融证明，不能把本轮G/P或未验证M1/M2算两个独立成功组件。

本次只授权一轮的实际执行及结果讨论现已完成；未启动第二轮，未重启旧周期自动化。原10轮结束后的jev-10暂停状态未更改。不要重放任何R11 intent或controller，也不要为了负向结果择优重跑。最终closure记录另存R11_result_discussion_closure_v1.json。
