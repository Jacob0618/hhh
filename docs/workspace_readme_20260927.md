# Mem0 × JEV × LoCoMo：准备工作区

状态：**2026-09-27 用户已授权真实实验。四项 API 探测、Mem0 与独立架构端到端检查已通过；完整事实库正在构建。** 当前主协议：**完整 session 一次写入，每个问题最终 top20 事实**。top20 按读取阶段解释，写入事实不截断。最新状态看 [真实运行记录](research/live_execution_2026-09-27.md) 与各 runs 目录的 queue_status.json。

## 已固定的材料

- 普通 Mem0 Python OSS 2.2.1：`upstream/mem0`，commit `94c3fe9f238f3dbf29c9ce98643bd71eb13077cd`。
- 配套 memory-benchmarks submodule：commit `4b61c5d31b9c668a12b4f5e78064248a02c82d2b`。
- 本地 LoCoMo 原始副本：`data/locomo10.json`；1986 题，仅剔除 category 5 的 446 题，最终 1540 题。
- 数据 SHA-256：`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`。
- 10 段 conversation、272 个 session、5882 个 turn；源数据所有 turn 都有非空文本。本项目按用户要求，每个完整 session 一次 add，预计272次提交；上游原始 CHUNK_SIZE=1 仅作为审计记录。内部调用与重试另计。

## 阅读顺序

1. [当前具体执行计划](research/session_top20_design_zh.md)：session/top20、标签 schema、补证、选择器与对照。
2. [Mem0 写入审计](research/mem0_oss_write_audit.md)：实际调用路径及源码行号。
3. [Benchmark 协议审计](research/mem0_benchmark_protocol_audit.md)：写入、日期、接口、评分兼容性。
4. [Prompt 索引](prompts/README.md)：原样快照，区分当前 active 与历史未调用模板。
5. [数据审计](research/locomo_dataset_audit.md)及[逐题清单](research/locomo_1540_manifest.json)。
6. [当前实验规格](configs/experiment_plan.json)：已启用真实调用；tools/run_experiment_queue.py 负责可恢复的分阶段实验。
7. [环境与启动检查](research/local_environment.md)及[实际检查结果](research/local_startup_report.json)。
8. [逐 session 静态清单](research/session_ingestion_manifest.json)：272个完整写入边界与源顺序尾部10条。

## 当前完成范围

Git 源码和 evaluation 子模块已拉取，固定 checkout 已安装到独立 CPython3.11.14 的 `.venv`。真实 Mem0、Gemini providers、本地 Qdrant、SQLite、BM25 与 spaCy 已验证可用；不需要 Docker 或图数据库。凭据从原有 Windows DPAPI vault 读取，未复制明文到本项目；上游源码未修改。开发阶段8题的旧reader输出已归档，正式比较统一使用简短答案与局部证据编号的 reader v2。

后续直接调用 Python Memory，使用 D 盘本地 Qdrant 路径和 SQLite。**离线初始化通过，不等于生成、embedding、BM25或真实session写入已验证。** 实验前仍须补完日期/历史顺序适配、原文来源追踪、真实模型版本配置和检索能力检查；上游 HTTP benchmark wrapper 不能直接视为与当前 OSS 兼容。

静态快照可以用 `.venv/Scripts/python.exe tools/prepare_sources.py` 重建。只做离线启动检查则运行 `.venv/Scripts/python.exe tools/check_mem0_startup.py`。这些脚本不启动评测。原始数据与既有研究目录均保留。
