# Independent aggregation bound to completed QA raw audit

Completed 2026-10-07T22:20:04.434985+00:00. All 4620 answer metrics were recomputed and all 4620 saved Judge texts reparsed; no mismatches. All 18480 leaf hashes match the completed mandatory raw seal (`no_invalid_artifacts`). The zero-job recovery is terminal.

| Arm | Official F1 (%) | Custom F1 (%) | Judge correct / valid / fixed | Judge (%) |
|---|---:|---:|---:|---:|
| baseline | 53.221636 | 51.138502 | 1164 / 1540 / 1540 | 75.584416 |
| r03_original | 54.886885 | 52.499132 | 1213 / 1540 / 1540 | 78.766234 |
| original_multi_source_closure | 59.884397 | 57.408205 | 1295 / 1540 / 1540 | 84.090909 |

| Cumulative contrast | Judge gain / loss | Judge delta (pp) | Official F1 delta (pp) | Custom F1 delta (pp) |
|---|---:|---:|---:|---:|
| r03_original minus baseline | 92 / 43 | +3.181818 | +1.665249 | +1.360631 |
| original_multi_source_closure minus baseline | 163 / 32 | +8.506494 | +6.662761 | +6.269703 |
| original_multi_source_closure minus r03_original | 117 / 35 | +5.324675 | +4.997512 | +4.909072 |

## By category

| Arm | Category | N | Official F1 (%) | Custom F1 (%) | Judge correct | Judge (%) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 282 | 48.045669 | 42.790635 | 202 | 71.631206 |
| baseline | 2 | 321 | 57.060917 | 56.731664 | 253 | 78.816199 |
| baseline | 3 | 96 | 30.081495 | 27.114148 | 42 | 43.750000 |
| baseline | 4 | 841 | 56.133249 | 54.545197 | 667 | 79.310345 |
| r03_original | 1 | 282 | 53.103950 | 45.947158 | 214 | 75.886525 |
| r03_original | 2 | 321 | 59.523347 | 59.130973 | 256 | 79.750779 |
| r03_original | 3 | 96 | 32.771900 | 30.004516 | 46 | 47.916667 |
| r03_original | 4 | 841 | 56.239467 | 54.732568 | 697 | 82.877527 |
| original_multi_source_closure | 1 | 282 | 54.270901 | 47.074545 | 225 | 79.787234 |
| original_multi_source_closure | 2 | 321 | 61.752354 | 61.367706 | 266 | 82.866044 |
| original_multi_source_closure | 3 | 96 | 29.144070 | 26.414355 | 47 | 48.958333 |
| original_multi_source_closure | 4 | 841 | 64.562712 | 62.899883 | 757 | 90.011891 |

Paired gains/losses, changed query IDs, F1 differences and per-conversation/category detail are in the JSON. This report is descriptive; it does not claim statistical significance.

The comparison is cumulative: baseline -> original R03 QFS -> original R03 + new R05+C. The final increment is conditional on R03 and combines the added source mechanism and closure; there is no independent source-only arm or separate causal estimate for each added component. Gemini2.5 thinkingBudget=0 with a new Writer bank is not equivalent to Gemini3.8 low thinking or a Reader-only model change. Do not claim two independently isolated components each exceed 3pp.

Usage is handled in the separate read-only `independent_usage_v1.json`; exact physical counts exclude cache reuse and retain unknown failure usage/cost. That parallel review was not yet present at this report observation; bind it in the subsequent comparison report.

Canonical summary observation: exists=False, compared=False. This aggregation binds only the already completed QA seal; the canonical summary must be compared and bound in a separate version when available.

No API/vault/credential access, scientific-source replay, full raw API replay, source change or production mutation occurred. The paired sealed JSON binds this Markdown SHA.
