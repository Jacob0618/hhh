# 结果与数据索引

下表汇总 Gemini 2.5 条件；数值来自保留的 accepted summaries，未重新推理。F1 和 Judge 单位均为百分比。

| 条件 | Official F1 | Judge YES / 1540 | Judge % |
|---|---:|---:|---:|
| Mem0 base | 53.221636 | 1164 | 75.584416 |
| + QFS | 54.886885 | 1213 | 78.766234 |
| Full QFS + SER（含 C） | 59.884397 | 1295 | 84.090909 |
| A-Mem | 43.176122 | 969 | 62.922078 |
| SimpleMem | 56.696199 | 1263 | 82.012987 |

Gemini 3.8 主实验另有七臂，完整条件 F1 60.238005、Judge 1302/1540；与 Gemini 2.5 分开解读。以下保留全部相关条件和探索历史，避免只发布最佳结果。

A/Simple 虽已完成，原论文 baseline 表仍待更新。HyperMem 已取消且无最终1540结果；V-Mem/PRISM尚未运行。

大型原始 summary、审计 inventory 和三个原raw ZIP仅在 [本地证据清单](local_only_artifacts.json)登记。逐题数值可查，但本仓库不保证完整原raw审计链的离线重放。

## paper_current

Authoritative current draft and its versioned validation; baseline table is still blank in this draft.

- [paper_acl2027_2026-10-07/main.tex](../paper_acl2027_2026-10-07/main.tex)
- [paper_acl2027_2026-10-07/main.pdf](../paper_acl2027_2026-10-07/main.pdf)
- [paper_acl2027_2026-10-07/references.bib](../paper_acl2027_2026-10-07/references.bib)
- [paper_acl2027_2026-10-07/acl.sty](../paper_acl2027_2026-10-07/acl.sty)
- [paper_acl2027_2026-10-07/acl_natbib.bst](../paper_acl2027_2026-10-07/acl_natbib.bst)
- [paper_acl2027_2026-10-07/README.md](../paper_acl2027_2026-10-07/README.md)
- [paper_acl2027_2026-10-07/writing_plan_zh.md](../paper_acl2027_2026-10-07/writing_plan_zh.md)
- [paper_acl2027_2026-10-07/revision_gemini25_2026-10-08.md](../paper_acl2027_2026-10-07/revision_gemini25_2026-10-08.md)
- [paper_acl2027_2026-10-07/validate_document_v2.py](../paper_acl2027_2026-10-07/validate_document_v2.py)
- [paper_acl2027_2026-10-07/document_validation_v2.json](../paper_acl2027_2026-10-07/document_validation_v2.json)
- [paper_acl2027_2026-10-07/JEV_Mem_ACL2027_source_v2.zip](../paper_acl2027_2026-10-07/JEV_Mem_ACL2027_source_v2.zip)
- [paper_acl2027_2026-10-07/template/provenance.json](../paper_acl2027_2026-10-07/template/provenance.json)

## dataset_and_protocol

Fixed dataset/question set and original scoring/reader protocol, not credentials or machine environment.

- [data/locomo10.json](../data/locomo10.json)
- [data/local_copy_manifest.json](../data/local_copy_manifest.json)
- [configs/sdk20_full1540_locked_v1.json](../configs/sdk20_full1540_locked_v1.json)
- [configs/canonical_sdk20_protocol_v1.json](../configs/canonical_sdk20_protocol_v1.json)
- [configs/plain_text_qa_v1.json](../configs/plain_text_qa_v1.json)
- [configs/gemini38_seven_arm_full1540_v1.json](../configs/gemini38_seven_arm_full1540_v1.json)
- [prompts/README.md](../prompts/README.md)

## gemini38_mem0_r03_r05_complete_seven_arms

Contains all seven tested arms, including unchanged or negative results; source for current manuscript main result.

- [research/r03_r05_refinement_2026-10-06/final_readout_v1.md](../research/r03_r05_refinement_2026-10-06/final_readout_v1.md)
- [research/r03_r05_refinement_2026-10-06/final_result_digest_v1.json](../research/r03_r05_refinement_2026-10-06/final_result_digest_v1.json)
- [research/r03_r05_refinement_2026-10-06/completion_record_v1.json](../research/r03_r05_refinement_2026-10-06/completion_record_v1.json)
- [research/r03_r05_refinement_2026-10-06/finalreview_v1.json](../research/r03_r05_refinement_2026-10-06/finalreview_v1.json)
- [research/r03_r05_refinement_2026-10-06/protocol_v1.json](../research/r03_r05_refinement_2026-10-06/protocol_v1.json)
- [research/r03_r05_refinement_2026-10-06/r03_protocol_v1.json](../research/r03_r05_refinement_2026-10-06/r03_protocol_v1.json)
- [research/r03_r05_refinement_2026-10-06/execution_plan.md](../research/r03_r05_refinement_2026-10-06/execution_plan.md)
- [research/r03_r05_refinement_2026-10-06/execution_handoff.md](../research/r03_r05_refinement_2026-10-06/execution_handoff.md)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/compact_records_sol_v1.csv](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/compact_records_sol_v1.csv)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/provisional_metrics_sol_v1.json](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/provisional_metrics_sol_v1.json)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/adoption_mechanisms_sol_v1.json](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/adoption_mechanisms_sol_v1.json)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/case_review_astra_v1.json](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/case_review_astra_v1.json)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_acceptance_sol_v1.json](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_acceptance_sol_v1.json)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_acceptance_astra_v1.json](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_acceptance_astra_v1.json)
- [research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_discussion_consensus_v1.json](../research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_discussion_consensus_v1.json)

## gemini25_mem0_r03_r05_complete_three_arms

All 4620 slots and independent numerical/usage comparison, including denominator and missing-usage limits.

- [research/gemini25_ablation_2026-10-07/final_results_2026-10-08.md](../research/gemini25_ablation_2026-10-07/final_results_2026-10-08.md)
- [research/gemini25_ablation_2026-10-07/execution_plan.md](../research/gemini25_ablation_2026-10-07/execution_plan.md)
- [research/gemini25_ablation_2026-10-07/execution_handoff.md](../research/gemini25_ablation_2026-10-07/execution_handoff.md)
- [research/gemini25_ablation_2026-10-07/full_structural_v1.json](../research/gemini25_ablation_2026-10-07/full_structural_v1.json)
- [research/gemini25_ablation_2026-10-07/full_semantic_v1.json](../research/gemini25_ablation_2026-10-07/full_semantic_v1.json)
- [research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.json](../research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.json)
- [research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.md](../research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.md)
- [research/gemini25_ablation_2026-10-07/result_review_v1/independent_aggregation_v1.json](../research/gemini25_ablation_2026-10-07/result_review_v1/independent_aggregation_v1.json)
- [research/gemini25_ablation_2026-10-07/result_review_v1/independent_aggregation_v1.md](../research/gemini25_ablation_2026-10-07/result_review_v1/independent_aggregation_v1.md)
- [research/gemini25_ablation_2026-10-07/result_review_v1/independent_usage_v1.json](../research/gemini25_ablation_2026-10-07/result_review_v1/independent_usage_v1.json)
- [research/gemini25_ablation_2026-10-07/result_review_v1/canonical_usage_comparison_v1.json](../research/gemini25_ablation_2026-10-07/result_review_v1/canonical_usage_comparison_v1.json)
- [research/gemini25_ablation_2026-10-07/result_review_v1/usage_binding_v1.json](../research/gemini25_ablation_2026-10-07/result_review_v1/usage_binding_v1.json)

## module_stack_selection_history

Preserves component-selection context and unsuccessful combinations instead of publishing only the winner.

- [research/module_stack_2026-10-05/result_report_2026-10-06.md](../research/module_stack_2026-10-05/result_report_2026-10-06.md)
- [research/module_stack_2026-10-05/derived_results_v1.json](../research/module_stack_2026-10-05/derived_results_v1.json)
- [research/module_stack_2026-10-05/results_sol_v1.json](../research/module_stack_2026-10-05/results_sol_v1.json)
- [research/module_stack_2026-10-05/results_astra_v1.json](../research/module_stack_2026-10-05/results_astra_v1.json)
- [research/module_stack_2026-10-05/research_completion_v1.json](../research/module_stack_2026-10-05/research_completion_v1.json)
- [research/module_stack_2026-10-05/finalreview_v1.json](../research/module_stack_2026-10-05/finalreview_v1.json)
- [research/module_stack_2026-10-05/protocol_v1.json](../research/module_stack_2026-10-05/protocol_v1.json)
- [research/module_stack_2026-10-05/execution_handoff.md](../research/module_stack_2026-10-05/execution_handoff.md)
- [research/module_stack_2026-10-05/logical_reader_token_sizes_v1.json](../research/module_stack_2026-10-05/logical_reader_token_sizes_v1.json)

## r01_to_r10_exploration_history

Compact historical numerical reviews, explicitly exploratory and not independent test cohorts.

- [research/jev_autoresearch_2026-10-04/execution_handoff.md](../research/jev_autoresearch_2026-10-04/execution_handoff.md)
- [research/jev_autoresearch_2026-10-04/r01_result_compact.json](../research/jev_autoresearch_2026-10-04/r01_result_compact.json)
- [research/jev_autoresearch_2026-10-04/r01_result_readout.md](../research/jev_autoresearch_2026-10-04/r01_result_readout.md)
- [research/jev_autoresearch_2026-10-04/R02_result_astra.json](../research/jev_autoresearch_2026-10-04/R02_result_astra.json)
- [research/jev_autoresearch_2026-10-04/r02_result_readout.md](../research/jev_autoresearch_2026-10-04/r02_result_readout.md)
- [research/jev_autoresearch_2026-10-04/R03_result_astra.json](../research/jev_autoresearch_2026-10-04/R03_result_astra.json)
- [research/jev_autoresearch_2026-10-04/r03_result_readout.md](../research/jev_autoresearch_2026-10-04/r03_result_readout.md)
- [research/jev_autoresearch_2026-10-04/R04_result_astra.json](../research/jev_autoresearch_2026-10-04/R04_result_astra.json)
- [research/jev_autoresearch_2026-10-04/r04_result_readout.md](../research/jev_autoresearch_2026-10-04/r04_result_readout.md)
- [research/jev_autoresearch_2026-10-04/r05_root_compact_results.json](../research/jev_autoresearch_2026-10-04/r05_root_compact_results.json)
- [research/jev_autoresearch_2026-10-04/r05_result_readout.md](../research/jev_autoresearch_2026-10-04/r05_result_readout.md)
- [research/jev_autoresearch_2026-10-04/R05_result_write61.json](../research/jev_autoresearch_2026-10-04/R05_result_write61.json)
- [research/jev_autoresearch_2026-10-04/R05_source_append_protocol_final_v1.json](../research/jev_autoresearch_2026-10-04/R05_source_append_protocol_final_v1.json)
- [research/jev_autoresearch_2026-10-04/R06_result_astra.json](../research/jev_autoresearch_2026-10-04/R06_result_astra.json)
- [research/jev_autoresearch_2026-10-04/R07_result_astra.json](../research/jev_autoresearch_2026-10-04/R07_result_astra.json)
- [research/jev_autoresearch_2026-10-04/R08_result_astra.json](../research/jev_autoresearch_2026-10-04/R08_result_astra.json)
- [research/jev_autoresearch_2026-10-04/R09_result_astra.json](../research/jev_autoresearch_2026-10-04/R09_result_astra.json)
- [research/jev_autoresearch_2026-10-04/r09_result_readout.md](../research/jev_autoresearch_2026-10-04/r09_result_readout.md)
- [research/jev_autoresearch_2026-10-04/R10_result_astra_v1.json](../research/jev_autoresearch_2026-10-04/R10_result_astra_v1.json)

## r11_and_earlier_mem0_context

Compact prior baselines and negative controls; not substitutes for the final matched cohorts.

- [research/module_consolidation_2026-10-05/R11_result_readout.md](../research/module_consolidation_2026-10-05/R11_result_readout.md)
- [research/module_consolidation_2026-10-05/R11_execution_handoff.md](../research/module_consolidation_2026-10-05/R11_execution_handoff.md)
- [research/gemini38_retest_2026-10-03/seven_arm_results_2026-10-04.md](../research/gemini38_retest_2026-10-03/seven_arm_results_2026-10-04.md)
- [research/gemini38_retest_2026-10-03/summary_independent_review_61.json](../research/gemini38_retest_2026-10-03/summary_independent_review_61.json)
- [research/reports/regroup_refinement_2026-09-30/results_readout.md](../research/reports/regroup_refinement_2026-09-30/results_readout.md)
- [research/reports/mem0_global_regroup_plain_qa_v2_strict/summary.md](../research/reports/mem0_global_regroup_plain_qa_v2_strict/summary.md)
- [research/mem0_sparse1540_final_summary_2026-09-28/summary.md](../research/mem0_sparse1540_final_summary_2026-09-28/summary.md)

## accepted_amem_1540

Final accepted effective rows; includes firstpass and exact request audits so failures/recovery are not erased.

- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/summary.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/summary.json)
- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/per_query.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/per_query.json)
- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/firstpass_per_query.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/firstpass_per_query.json)
- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/method_request_audit.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/method_request_audit.json)
- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/raw_request_audit.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/raw_request_audit.json)
- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/cleanup_receipt.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/cleanup_receipt.json)
- [runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/predecessor_completion.json](../runs/five_baselines_gemini25_v1_20261008/amem_native_v4/retrieval_continuation_v1/results/predecessor_completion.json)
- [research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_result_review_v2.md](../research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_result_review_v2.md)
- [research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_result_review_v2.json](../research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_result_review_v2.json)
- [research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_comparison_v1.json](../research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_comparison_v1.json)
- [research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_usage_chain_v2.json](../research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_usage_chain_v2.json)
- [research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_usage_telemetry_correction_v2.json](../research/baseline_execution_2026-10-08/amem_retrieval_continuation_v1/result_review_v1/independent_usage_telemetry_correction_v2.json)

## accepted_simplemem_1540

Final accepted effective rows plus initial/completed firstpass and full-context method fidelity and paired numerical checks.

- [runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/summary.json](../runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/summary.json)
- [runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/per_query.json](../runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/per_query.json)
- [runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/firstpass_per_query.json](../runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/firstpass_per_query.json)
- [runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/completed_firstpass_per_query.json](../runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/completed_firstpass_per_query.json)
- [runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/method_request_audit.json](../runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/method_request_audit.json)
- [runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/raw_request_audit.json](../runs/five_baselines_gemini25_v1_20261008/simplemem_native_v2/results/raw_request_audit.json)
- [research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/independent_result_review_v1.md](../research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/independent_result_review_v1.md)
- [research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/independent_usage_review_v1.json](../research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/independent_usage_review_v1.json)
- [research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/method_fidelity_review_v1.json](../research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/method_fidelity_review_v1.json)
- [research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/paired_mem0_numeric_review_v1.json](../research/baseline_execution_2026-10-08/simplemem_native_v2/result_review_v1/paired_mem0_numeric_review_v1.json)
- [research/baseline_execution_2026-10-08/simplemem_native_v2_lifecycle/cleanup_receipt.json](../research/baseline_execution_2026-10-08/simplemem_native_v2_lifecycle/cleanup_receipt.json)
- [research/baseline_execution_2026-10-08/simplemem_native_v2_lifecycle/predecessor_completion.json](../research/baseline_execution_2026-10-08/simplemem_native_v2_lifecycle/predecessor_completion.json)

