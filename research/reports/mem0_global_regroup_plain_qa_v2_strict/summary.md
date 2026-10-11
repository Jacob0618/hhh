# Mem0两臂纯文本QA离线审核

各臂计划1540题；Reader/F1与Judge有效分母分别报告。

|臂|Reader有效|Judge有效|官方F1（有效题）|custom F1（有效题）|Judge（有效题）|
|---|---:|---:|---:|---:|---:|
|dated_unmerged385|1540/1540|1430/1540|54.375|51.856|68.671|
|dated_merged|1540/1540|1432/1540|55.047|52.335|68.575|

Previously exposed10-conversation cohort; each metric uses its own common-valid subset. Missingness can bias pairs. Bootstrap is exploratory, not blind validation or a causal claim.

零填充仅是缺失敏感度分析，不是实际生成评分。费用未知项不会填零，原缓存不重复计费。
