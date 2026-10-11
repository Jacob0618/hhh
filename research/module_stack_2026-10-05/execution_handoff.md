# 本轮状态与接续说明

当前是用户已授权的一轮固定1540模块堆叠实验。主链：N0→R03 JEV→R03+R08→+R05；4个无原文因子臂，4个共同R08原20底座的S×C原文臂，外加N0，共9×1540。详细科学方案见 execution_plan.md 与 protocol_v1.json。

新的源计划/调用入口为 tools/prepare_module_stack_v1.py、tools/score_module_stack_v1.py；新QA/汇总薄适配 tools/run_module_stack_qa_v1.py、tools/summarize_module_stack_v1.py。真实模型/Reader/Judge及其durable stage核心仍原实现；原源码未修改。三个阶段support→closure→rank，完整前驱实际源重建、SHA、producer/consumer/raw核验；已知失败阶段结束后统一一个恢复阶段（原provider每阶段retries2），未知或不完整intent阻断，不清guard。

旧 runs/module_stack_v1_20261005 只完成0API的prepare预检，后经独审增强了新代码的失败和完整性门禁，该预检namespace不用于实验、不重启、不覆写。没有其模型调用/QA结果。正式运行指定全新 runs/module_stack_v1_final_20261005；QA runs/module_stack_qa_v1_20261005。不能把预检存在视为正式实验已启动。

归档 R03_R04_R06_R08_review_v3.zip SHA b7466b44fe60174dfc6cc5acb2f8851f238afb6905fdc4941b1911da7fd57eaa 已真实全量核对：6160选择输入、39成员、10conv；只继承输入，不继承删除的API缓存/旧QA。N0真实12289绑定和1540当前请求签名复核通过，合法完整exact请求继承。原R05缓存/结果未删除，D盘当前约55.7GiB可用；不复制向量库。

pipeline tools/run_module_stack_pipeline_v1.py 只按冻结门禁顺序推进一次：prepare→3个JEV阶段及各自最多一次统一技术恢复→seal→QA首遍→一次已知失败恢复→原raw离线汇总。需要正式code/SPEC哈希review ready后启动，只一个controller。全部科学源码在正式prepare后冻结，健康活跃不改。pipeline completed_awaiting_research_discussion表示技术实验完成，仍需要ROOT与Astra/Sol分析结果，不自动扩大实验或挑正确性重试。

观察工具 tools/snapshot_module_stack_v1.py；至少两快照约30秒及真实PID/命令/创建时间，阶段产物/HTTP/cache/错误增量。阶段计划重建属于纯离线计算，可在开始真实HTTP前耗时，不误判但须结合CPU/实物；10分钟无进展排查而不直接报健康。凭据仅现有DPAPI vault，不输出。

最新启动/完成/异常信息在本文件后追加，research/module_stack_2026-10-05/live_v1/status.json为本轮pipeline状态。旧jev-10当前仍PAUSED，不能用旧R10/七臂prompt恢复。若需恢复每小时跟进应更新成此新有限任务，不启动旧实验。


## 正式启动与实际健康确认（2026-10-05 23:54北京时间）

正式pipeline已于23:47:30启动；launcher6128/actual4544，命令、创建时间、父子关系均通过snapshot实核。finalreview_v1.json SHA6a56d9f9431e409eb6e1af635867e7e8eb4b0602f66e411692d86e372e3336cd，正式source manifest位于runs/module_stack_v1_final_20261005。58项冻结前测试通过；后仅只读snapshot追加manifest前units/requests计数测试，不改科学CODE。

support已真实构建1637事实、3274请求并启动。health_dispatch_a(15:52:56 UTC)至health_dispatch_b(15:53:42 UTC)间隔45.89秒，完整score响应1481→2163（+682），均complete；实际HTTP增695，其中694HTTP200、1运输错误，物理日志持续增长。两个已见运输错误分别按原request_hash+同context证实attempt2 HTTP200恢复；非用其他任务的成功替代。现无终态评分失败、无local error、无读错；c0/c1已完成，c8/c9自动补位，其余均真实新增。以上仅support阶段，不是QA完成率、支持语义准确率或最后科学效果。

小时心跳jev-10已用本轮有限任务替换旧十轮prompt并ACTIVE（每小时，正常静默），不会重启旧十轮。pipeline后续自动按已审门禁推进；完成后仍需读取final_summary_v1及result_analysis_contract.md，与Astra/Sol分析差值/失误/交互再交付、暂停。不要重复启动pipeline；若needs_review查实际失败stage和产物，仅另版本必要技术修复。result_analysis_contract.md特别强调R03旧池本身含R01独立BM25候选，N0→R03不是纯JEV单因素，独立JEV作用用同池BM25/JEV比较。


## 小时核查 2026-10-06 00:50–00:57北京

独立Sol只读核对support/closure终态与运输错误链；support3274/3274、closure12320/12320 complete，controller全raw/producer-consumer审计已通过，无local errors、无unknown、无额外恢复phase。真实PID4544/父6128及原创建时间/命令再次核对；冻结40代码文件SHA全一致，未修改。

00:51前后两个快照无新API是rank输入重建/验证（CPU持续增加，随后证实结束），不是卡死。rank真实score于00:53:53启动，00:55:18→00:56:11两快照约52.91秒，终态有效1356→2229（+873），8活跃conv均新产物；窗口物理518=517HTTP200+1运输错误，exact缓存358。00:57按rank全部观测4条错误链各自hash+context核实attempt2已成功，不用别的调用冒充恢复。D盘尚余约58.0GB十进制。详细证据research/module_stack_2026-10-05/heartbeat_20261006_0050/root_audit.json及health_a..d。snapshot多文件非原子，status/files的1条采样差异不当成失败。

原controller健康继续，不重启、不改科学代码/缓存/配置。QA尚未开始，不能报提升；等待rank终态→封存13860源槽→统一QA/一次技术恢复→最终raw总结。小时心跳继续ACTIVE。Sol还在编写仅源几何机制计数（不看QA），若文件到达可用于最终机制分析，但非准确率。

## 2026-10-06 01:51–02:21北京：QA全部终态，离线原始响应审计仍在运行

此追加覆盖上一段“QA尚未开始”。同一controller4544/父6128完成全部源阶段和QA，未重启。support3274、closure12320、rank6160全部complete；source13860槽已封存。QA正式run_outcome完成时间17:53:48.171112 UTC：9×1540=13860，Reader/Judge/QA全complete、compat全strict_valid、10conv终态、local_errors0；原统一recovery_jobs为空，不需要额外恢复phase。

pipeline从17:55:38.848155 UTC进入summary，仍在有限的离线raw replay/来源与缓存审计。约18:19 UTC controller ReadOperationCount已4,780,841、ReadTransferCount36.50GB、CPU持续增长，stderr无报错、无新模型流量。QA/API静止是终态，不是QA挂住。summary旧复用代码含继承stage逐ledger重算fingerprint、普通stage扫描与intent×ledger/多次SHA，独立QA adapter静态检查确认有限循环、未发现未知请求重发或无限重试；因此不能重启pipeline、清guard或为了提速改活跃冻结代码。最终summary未发布前不把轻审或完整槽数当raw重放已通过。

独立技术审查已完成：technical_terminal_review_v1.md/json，JSON SHA c176da6273039d0d7751edb2424cff56eeba92e344a5ea35b1483e61bb8afdae。0 production issues；9×1540唯一job、55,440 stage bytes、17,811 stage intents、29,628 API intents均核过，40 finalreview源码/67 QA绑定源码/27原始来源绑定匹配。全阶段29,707物理尝试=29,628成功+79运输错误，79均同hash/context在原retries2内恢复，非额外recoveryphase。QA27720 stage=12061新producer+5750本轮cache+1556继承Reader+8353继承Judge；Judge继承可能由新Reader产生相同答案触发，不能当独立Judge重判。Rank6160=3705producer+2455cache，物理3711包含6运输错误。

两份终态健康快照与说明在heartbeat_20261006_0151/health_a.json、health_b.json、terminal_transition.md。这些多文件快照非原子，health_a开始时间早于读取到run_outcome的时间，不能用snapshot开始时间作为QA完成时刻。D剩余约56.77GB（十进制），未清缓存或磁盘。measured_ledger_usage_v1.json记录本次新物理usage，旧N0继承独立，不把79未知usage算0。JEV成功input32107120/output1081652；QA成功prompt15527922/visible87180/thinking1335367/total16950469；无已采用新价目，总费用未知。

Sol已存qa_ledger_sol_light_audit_v1.json与source_input_groups_sol_v1.json，前者32条QA transport链逐项恢复，后者仅输入变化计数，不是分数。前一检查的mechanism_counts_v1.json仍为S/C机制几何计数，不是语义准确率。

下一步：等待live_v1/final_summary_v1.json稳定发布且pipeline completed_awaiting_research_discussion；必须audit_status=no_invalid_artifacts。不要另起第二个summary抢跑同一重放。ROOT已新增纯离线research/module_stack_2026-10-05/analyze_final_rows_v1.py，尚未执行；它只从正式summary派生9臂指标、主链/类别/conv gain-loss、actual完整Reader request hash同异分层、两套2×2简单效应/交互及固定seed2000次描述bootstrap/LOCO，不改原summary。执行前确认正式summary已通过，输出derived_results_v1.json一次性新文件。

然后followup现有module_merge_astra（Astra）与module_merge_sol（6.1Sol）：已约定Astra主链/pool×selector，Sol主审S/C与source病例，双方交叉核对并共同讨论。二者尚未读QA正确性/未发布结果报告，等待正式summary；不要把等待消息当科学分析完成。需要复盘得失的检索20/附件/源原文/Reader/Judge，区分已有证据遗漏、locator换弱来源、closure噪声和评分不确定；按result_analysis_contract.md报告单次/共享请求/已见1540限制。全部交付后才暂停jev-10，目前保持ACTIVE，不扩实验、不归档、不删除原R05或ZIP。


## 2026-10-06 03:10北京：本轮全部完成、原raw复核通过、Astra/Sol讨论完成

本段覆盖此前等待summary/讨论状态。pipeline于18:25:28.772394 UTC终态completed_awaiting_research_discussion；原controller已退出。正式final_summary_v1.json已通过no_invalid_artifacts、errors空，SHA ee1039384934b84b6da183dbe4c13904ca9355fdcdf3d2b633c8813b80123686。全部9×1540=13860 Reader/Judge有效，无missing，原统一恢复队列为空；79条运输错误均已在原provider重试内恢复。未另起QA或summary、未改冻结模型/prompt/选择、未按正确性重试。

本次heartbeat_20261006_0253/health_a.json和health_b.json两快照确认全终态、无API新增、无活跃pipeline、0读错，D盘约56.58GB十进制可用。保持原R05、ZIP、缓存与结果，不清盘不复制库。

ROOT派生derived_results_v1.json，Astra与6.1Sol独立核全部九臂三指标、12边gain/loss IDs与类别/conversation分层、完整Reader请求同异和两因子交互，全部一致。双方已多轮讨论并明确同意报告建议。Astra主链12病例+2来源病例；Sol全部C15翻转及指定S9病例，未读全部S翻转，未当人工gold。完整结果在result_report_2026-10-06.md、results_astra_v1.md/json、results_sol_v1.md/json、case_material_sol_v1.json；research_completion_v1.json绑定最终证据SHA。

主链Judge：N0 74.870→R03 76.818→R08 76.688→R05 83.117→C 83.831；F1分别51.044/52.547/52.594/58.702/59.063。R03 +1.948pp(49gain/19loss)，R08 −0.130pp(15/17)，R05 +6.429pp(122/23)，C +0.714pp(13/2)；S对原R05 −2.532pp(28/67)，S+C81.429低于原R05。最佳vsN0 +8.961pp，158gain/20loss。仅原文总组件主链>3pp，不是两独立组件各>3；同池JEV相对弱BM25的+8～10点不能冒充强N0组件贡献。所有四个source臂每题都2附件，未证明按需少补充。

共识：保留JEV选择和原文/C小正候选，当前S单中心替换拒升级。S同时可能丢问题所需原子关系或因center BM25/top4准入丢候选；C两loss目标证据仍在，新Reader遗漏而非删除证据。Judge仍有近同义/标点翻转，不改gold不挑重判。旧池+source/C未测，不能推出删除R08后最佳成绩不变。后续建议保留旧locator、多原子来源+问题定向选择，以及OJ/NJ×source/C；只建议，未自动启动。

此次有限比较、核查和讨论已全部完成，报告已发布；现在按用户约定暂停jev-10（工具确认另存automation_pause_receipt_v1.json），不归档、不恢复任何旧批次、不扩大实验。若未来新用户授权，先读此完成记录而非重跑旧pipeline。
