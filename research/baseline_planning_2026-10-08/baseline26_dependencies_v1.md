# 2026 候选基线的模型部署依赖

核验日：2026-10-08。仅检查官方固定版本源代码、既有缓存目录清单与论文；未安装、下载模型权重、加载模型或调用模型 API。以下选择为研究计划，未修改实验实现。

## 结论

| 基线 / 分支 | 当前官方实现的本地模型 | 生成模型 | 若检索 embedding 也统一成 Gemini | 本机新增权重需求 |
|---|---|---|---|---|
| V-Mem，LoCoMo 文本入口 | all-MiniLM-L6-v2，384 维 | Reader 经 OpenAI-compatible API；默认无生成式建库/检索 | 需 embedding adapter；可去掉本地 MiniLM | 原生 MiniLM 已缓存，通常无须再下载 |
| PRISM | all-MiniLM-L6-v2，384 维，默认 CPU | 抽取、因果维护、N3 证据重排、Reader 可用 API | 需 embedding adapter、维度/索引重建；无其他本地神经网络是该文本入口必需 | 原生 MiniLM 已缓存，通常无须再下载 |
| StructMem，官方 LoCoMo 构建例程 | 检索 encoder + LLMLingua-2 压缩器；分段共享压缩器 | 双视角事件抽取、总结、离线更新、Reader 可用 API | 检索 encoder 可以替换；LLMLingua-2 仍需要保留或另做远程同模型服务 | 本机已缓存允许使用的 XLM-R-large 版；尚未验证加载/运行 |

三者都不要求本地部署生成大模型，也不需要训练。V-Mem 与 PRISM 可直接复用 A-Mem 的同一份 MiniLM 磁盘缓存；多进程同时运行仍会分别占用内存。改用 Gemini embedding 是可实现的适配，不是仅改模型字符串就自动可用。

## 固定的官方源码

- V-Mem：`36916b14dc5241e04acbf5cd0f3c40799bc09550`。
  - [LoCoMo 入口](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/run_locomo.py) 创建 `Memory(..., vvisual=None)`，只加载文本 encoder。
  - [embeddings.py](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/embeddings.py) 使用 SentenceTransformer、FP32 NumPy 输出、L2 normalization。
  - [README](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/README.md) 的 SigLIP 服务属于图像分支，DINOv2 是视觉 encoder 消融；不要把这些计入本轮 LoCoMo 文本部署。
- PRISM：`fdc09e6bfe31f9c2852eb830ee07de1d750472f4`。
  - [配置](https://github.com/jingyip-cat/PRISM/blob/fdc09e6bfe31f9c2852eb830ee07de1d750472f4/config.example.py) 指定 MiniLM、384 维、CPU；生成组件使用 API 模型。
  - [embedding wrapper](https://github.com/jingyip-cat/PRISM/blob/fdc09e6bfe31f9c2852eb830ee07de1d750472f4/mmem/utils/embedding.py) 有独立维度属性及单条/批量接口，替换后须与索引维度一致。
  - [N3 reranker](https://github.com/jingyip-cat/PRISM/blob/fdc09e6bfe31f9c2852eb830ee07de1d750472f4/mmem/retrieval/reranker.py) 调用 `LLMClient.chat_json`，用 `qa_model` 给候选 bundle 打分，不是另部署 CrossEncoder 或 Qwen reranker。
  - [LLM client](https://github.com/jingyip-cat/PRISM/blob/fdc09e6bfe31f9c2852eb830ee07de1d750472f4/mmem/utils/llm.py) 使用 OpenAI-compatible API；统一 Gemini 需要复用项目 adapter 以控制 thinking/usage/JSON/retry。
- StructMem / LightMem：`8449d574df6bae1bdf3314a1564da65e2f37e046`。
  - [LoCoMo 构建脚本](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/experiments/locomo/add_locomo.py) 开启 `pre_compress`、`topic_segment`、`precomp_topic_shared`；压缩器与 embedding 的路径均为占位配置，并默认选 CUDA。
  - [LLMLingua 配置](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/src/lightmem/configs/pre_compressor/llmlingua_2.py) 的类默认是 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`，允许列表同时包含 `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`，也允许本地路径。
  - [topic segmenter](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/src/lightmem/factory/topic_segmenter/llmlingua_2.py) 共享压缩器 model/tokenizer 并读取 `output_attentions`；它不是检索 encoder，不能由 Gemini embedding 的向量结果替代。
  - [OpenAI embedding provider](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/src/lightmem/factory/text_embedder/openai.py) 已有远程 embedding 接口，仍应使用本项目 Gemini adapter 统一维度、归一化和统计。

## 现存缓存与容量边界

主代理本轮只读盘点保存在 `local_model_cache_inventory_v1.json`：

- MiniLM 完整 SentenceTransformer snapshot 为 `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`，单个权重文件 **90,868,376 bytes，约 90.9 MB**。配置、tokenizer、modules、pooling 文件齐全，SentenceTransformer 最大输入 256 wordpieces。该事实证明文件存在，不等同运行兼容性验证。另一 snapshot 缺 modules/pooling，不作为完整可复用版本。
- LLMLingua-2 XLM-R-large snapshot 为 `ebaba9b0e874dadd3003ffcff828e4397e568089`，单个权重文件 **2,235,829,648 bytes，约 2.236 GB / 2.082 GiB**；目录另有 config、tokenizer 与 special-token 配置。它是 StructMem 配置明确允许的型号，但不是默认 BERT 型号；使用时应明确记录选型，不能称已验证与论文具体压缩 checkpoint 一致。

容量规划：MiniLM 小批量 CPU 推理可先给每个编排+encoder 进程留约 **1–3 GiB RAM**、无需 GPU；这是规划而非实测。StructMem 的现存 XLM-R-large 不能用 2.236 GB 文件大小直接代表运行峰值；应在单实例、小 batch、限制输入长度条件下测量。其官方 LoCoMo 脚本默认 `MAX_WORKERS=5`、`USE_PROCESS_POOL=True`，每个进程可复制压缩器和 encoder，因此不可按一份模型的内存直接启动默认五进程。前处理压缩与 topic segmentation 在单实例内部共享同一 LLMLingua 模型。

## 适配需要记录

1. 用户本轮已指定 A-Mem 保留本地 MiniLM；SimpleMem 与 HyperMem 改 `gemini-embedding-001`、768 维；HyperMem 去除独立 reranker。不能称六种方法 embedding 全统一，因为 A-Mem 是保留原生 MiniLM 的显式例外。
2. 上述变更不自动决定 V-Mem / PRISM / StructMem 的 encoder；表中给出其原生部署与 Gemini 适配选项。若后续统一，标注 adapted embedding，重建所有索引并记录 task type、归一化、维度、revision、截断策略。
3. HyperMem 的无重排决定针对其独立 reranker。PRISM N3 是其方法的 API 证据压缩模块，保留它不增加本地模型；删除它应标为 N3 消融，不应作为省本地显存的必要步骤。
4. StructMem 若为了完全不加载本地模型而关闭 LLMLingua，不仅删除压缩，还影响基于 attention 的分段，需要明确改版；不能以“检索 embedding 换 Gemini”概括。
5. 本轮不应计入原生 benchmark 中可选 BERTScore 等额外评分模型；共同 Reader/Judge 路径需按实际入口确定，不能由 requirements 名称推断必加载模型。

本目录 `sources_baseline26/` 保存固定 SHA 的官方文本源码供复核，未执行这些源码。未给出新的模型/API价格估算。
