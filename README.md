# JEV memory research — continuation snapshot

本仓库保存当前论文、核心实现、公开基准数据、已接受的逐题结果和续工记录。整理日期：2026-10-12（北京时间）。来源工作区是 `D:\jev mem`；桌面旧仓库的历史没有混入本次提交。

## 从这里开始

1. [当前论文与 PDF](paper_acl2027_2026-10-07/README.md)：`main.tex`、`main.pdf`、参考文献和已验证的来源说明。
2. [续工说明](docs/CONTINUATION.md)：当前实验状态、已知限制、下一步边界。
3. [结果索引](docs/RESULTS.md)：主实验、消融、A-Mem/SimpleMem 的数值与逐题数据。
4. [复现与环境](REPRODUCIBILITY.md)：离线校验、固定上游版本、本地路径和凭据要求。
5. [提交文件清单](docs/repository_snapshot.json)：逐文件 SHA-256；[本地大文件清单](docs/local_only_artifacts.json)记录未上传的证据压缩包。

## 当前状态

| 工作 | 状态 |
|---|---|
| Gemini 3.8 主实验及 R03/R05 条件比较 | 已完成；原始端点标识按实验记录保留 |
| Gemini 2.5 的 Mem0 → QFS → QFS+SER 累计消融 | 已完成，三个条件各 1540 题 |
| A-Mem | 已完成 1540 题并完成结果审查与缓存清理 |
| SimpleMem | 已完成 1540 题并完成结果审查与缓存清理 |
| HyperMem | 用户已取消，专属实验与模型已清理；没有最终准度 |
| V-Mem、PRISM | 尚未运行，共剩两项 |

定时检查已暂停。本仓库不包含自动启动实验的 CI；克隆或校验不会调用模型。论文当前版本仍有待补的 baseline 表格和实验小节，不能将占位内容视为已完成结果。

## 克隆与离线检查

```bash
git clone --recurse-submodules https://github.com/Jacob0618/hhh.git
cd hhh
python tools/verify_repository_snapshot.py
```

Python 3.11 可运行这个标准库校验器。它检查已提交文件的字节及上游 revision，不重新推理，也不声称重放所有原始审计。

## 上传范围

包含论文源文件/PDF、`jev_mem`、`tools`、`tests`、配置、提示词、LoCoMo 数据及来源、选定结果和研究记录。Mem0、A-Mem、V-Mem、PRISM 使用固定 Git submodule；SimpleMem 保留带来源与许可的代码快照。通用历史源码可能包含未启用方法的接口，不代表该方法有完成结果。

不包含 API 密钥、DPAPI vault、Python 环境、下载模型、运行缓存、活动数据库或已取消的 HyperMem 专属实验。大型原始证据 ZIP 留在原工作区，只登记大小和 SHA；因此 GitHub 是论文/代码/数值数据的续工快照，**不是全部原始证据的异地备份**。

保留证据和冻结源码不为迁移而改字节；一些历史脚本含 `D:\jev mem` 绝对路径。换机器先阅读复现说明，不直接运行旧 launcher 或恢复旧配额。
