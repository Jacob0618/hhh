# SimpleMem Gemini 2.5 独立结果审查

结论：**passed，结果可接受**。此结论只涉及结果审查，不授权归档、删除、银行证书或下一方法启动。

实际使用独立审查进程执行了原终态完整 collect，并与93MB终态proof逐字段相等；随后实际调用冻结 seal_simplemem_native_v2.reconstruct(manifest,proof,created_utc=原summary时间)，1540题的原首遍、补齐首遍、有效逐题结果，以及完整QA请求审计、方法请求审计和summary六份输出全部逐字段一致。不是仅数文件或采信内部passed。全重放143.09秒，网络/凭据入口禁止，模型调用0，数据库引擎未打开。原44287、新3446文件完整canonical路径SHA映射前后相同；原库存还严格等proof库存。冻结代码和上游hash在重放中逐项核验。

最终summary SHA：`fcd728e9610227a4e0141cd174a6f6e25f4be0a08cd3d087f3c959397cce3f12`，audit_status=no_invalid_artifacts。原status仍needs_review，新status为complete_awaiting_result_review，未修改。

原首遍1540槽位有1470有效Reader、1469有效Judge：70检索失败不伪造分数，另1旧Judge格式失败。新namespace仅补70检索及70首遍QA，全部成功；随后统一一次QA技术恢复只处理c3_q24 Judge，原成功Reader字节内容强制继承，最终Judge为NO，未因NO重试。有效Reader/Judge均1540，缺失0、补值0。原1470 QA、失败响应及raw全部保留，继承和缓存不是新的独立样本。

|指标|SimpleMem|已接受同2.5 Mem0|配对差值|
|---|---:|---:|---:|
|Judge正确|1263/1540，82.012987%|1164/1540，75.584416%|+6.428571pp|
|官方LoCoMo F1|56.696199%|53.221636%|+3.474563pp|
|Custom token F1|53.728361%|51.138502%|+2.589859pp|

Judge逐题198胜、99负、净99。全部1540题题目/gold/类别/date/顺序匹配，Mem0 canonical和独立聚合逐题数字相同。Mem0范围仅四份固定path+SHA的已接受numeric leaves，不追旧删除依赖，也不声称重放Mem0全raw。

|类别编号|N|胜/负|Judge净提升pp|官方F1差pp|
|---|---:|---:|---:|---:|
|1|282|40/30|+3.546099|+4.195549|
|2|321|32/30|+0.623053|-7.774382|
|3|96|9/5|+4.166667|+3.215152|
|4|841|117/34|+9.869203|+7.556011|

类别4贡献83/99净Judge胜。类别2虽然Judge略升，词面F1明显下降，不能将总体提高解读成每类、每指标均改善。全部翻转query列表及每题差值已保存，不选择性删去负例。配对差值是该已曝光cohort上的描述性比较，不宣称盲测泛化或独立因果优势。

方法保真：固定上游text路径、40/2窗口、时序对话、speaker/date/caption、native规划与两轮reflection、25/5/5检索及完整context均保留。Gemini embedding001768替换须披露，外部共享Reader/Judge取代方法自己的答案生成。v2仅在已证明的确切失败点恢复上游native兼容分支；77等具体应用量以method_request_audit原记录为准，不扩成任意异常兜底。没有writer重建、原请求重新付费或gold进入native检索。完整请求/来源/producer/consumer/intent/raw顺序和嵌入双重归一化实际重放。来源关联是window级，不等于人工逐事实真实性审判。数据库只做完整字节绑定和导出一致性，未新跑DB搜索，也未替代后续银行证书。

Native结果25–137条，中位35、均41.5877条；context字符中位10511.5、均12228.18。以单个native_context容器交Reader不等于只给1条原事实。与Mem0 top20不等信息量或token预算；字符也不是token。不能把收益孤立归因于某个组件。

费用复核：原method11920物理/176cache，原QA2925物理/22cache；新method319物理/352cache、新QA140物理/0cache、唯一恢复1物理/0cache。合计15305物理、550exact cache。新method唯一HTTP503在同request/context/stage attempt2返回200，完整链已核。原所有运输/429链也在原fullproof重放中闭合。原14845和新460物理分开计费，cache旧usage不加为新调用量。4883物理请求usage缺失/unknown（主要embedding及错误）；thoughts字段全缺，美元cost为null，均不能当0。已报告prompt17848833、candidate2000542、total19849375仅是有字段响应的部分合计，不是全部端点已知总成本。

本报告的JSON包含完整双根库存及冻结code/proof/实际独立重放、费用、数值和方法保真证据SHA。未修改任何实验run、冻结源码或配置，未发API、访问凭据、打开原DB、启动pipeline/archive/delete。
