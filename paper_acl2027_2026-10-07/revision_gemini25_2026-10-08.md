# Gemini 2.5 result integration — 2026-10-08

Mode: `academic-paper` revision. This is a focused update to the approved ACL draft using completed local evidence. It does not change the method, run an experiment, or alter sealed scientific records.

## Revision record

| Item | Status | Manuscript change |
|---|---|---|
| Preserve the Gemini 3.8 results | ACCEPTED | Original main-result and component-removal numbers remain unchanged; captions and prose now name the Gemini 3.8 cohort. |
| Add the completed Gemini 2.5 comparison | ACCEPTED | Added “Gemini 2.5 Cumulative Ablation,” Table `tab:gemini25`, within-cohort increments, and accepted-answer counts. |
| Separate model and bank protocols | ACCEPTED | Experimental Settings now distinguishes the original 1,637-fact bank from ten new Gemini 2.5 banks with 2,301 facts, each written from the same 272 sessions. Writer, Reader, and Judge all change; `thinkingBudget=0` is not equated with Gemini 3.8 low thinking. |
| Bound the interpretation | ACCEPTED | The new chain is cumulative; SER includes neighboring completion C. No source-only arm, isolated SER/C effect, Judge-only model effect, unseen-test generalization, repeated-run stability, or statistical-significance claim is made. |
| Update the manuscript's high-level account | ACCEPTED | Abstract, Introduction, Conclusion, Limitations, and provenance appendix identify the second cohort and preserve the original result. |
| Preserve the approved draft gaps | ACCEPTED | The main table still contains only the full proposed-method row. All four external-baseline cells and all six planned experiment subsections remain blank. No figures or citations were added. |

## Values and rounding

Tables and prose use three decimal places, consistent with the original draft. The evidence records retain full precision. Display values below use six decimals on a 0–100 scale.

| Gemini 2.5 condition | Canonical arm | Official F1 | Custom F1 | Judge % | Judge yes / valid |
|---|---|---:|---:|---:|---:|
| Mem0 base | `baseline` | 53.221636 | 51.138502 | 75.584416 | 1164 / 1540 |
| + QFS | `r03_original` | 54.886885 | 52.499132 | 78.766234 | 1213 / 1540 |
| Full QFS + SER (including C) | `original_multi_source_closure` | 59.884397 | 57.408205 | 84.090909 | 1295 / 1540 |

| Cumulative contrast | Official F1 change | Judge change (percentage points) | Judge gains / losses |
|---|---:|---:|---:|
| QFS minus base | +1.665249 | +3.181818 | 92 / 43 |
| Full minus QFS | +4.997512 | +5.324675 | 117 / 35 |
| Full minus base | +6.662761 | +8.506494 | 163 / 32 |

Every arm has 1,540 valid Reader and Judge outcomes. All 4,620 QA slots are complete in the accepted first-pass view; the QA recovery set is empty. Bounded transport retries and earlier precomputation failures/recovery are separate records. The manuscript does not label the entire run as retry-free or failure-free. No old Gemini 3.8 fact identifiers, selection outputs, JEV scores, or QA answers are inherited in the Gemini 2.5 cohort; exact-request caching within that cohort does not constitute an independent repeat.

## Evidence pointers

- [Completed Gemini 2.5 report](<D:/jev mem/research/gemini25_ablation_2026-10-07/final_results_2026-10-08.md>).
- [Canonical Gemini 2.5 summary](<D:/jev mem/research/gemini25_ablation_2026-10-07/heartbeat_20261007T1814/billing_resume_v1/live_v1/final_summary_v1.json>), SHA256 `a73ac20650ac3f135ae07404ca4042a1ccb42fe1574ca9109c5edd63fb2d15d3`.
- [Final canonical comparison](<D:/jev mem/research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.json>) and [human-readable comparison](<D:/jev mem/research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.md>), including matching per-question metrics and labels, denominators, and cumulative-design limitations.
- [Independent Gemini 2.5 aggregation](<D:/jev mem/research/gemini25_ablation_2026-10-07/result_review_v1/independent_aggregation_v1.json>).
- [Original Gemini 3.8 summary](<D:/jev mem/research/r03_r05_refinement_2026-10-06/heartbeat_20261006T1052/live_qa_v2/final_summary_v1.json>), SHA256 `01aa05237a252832135f9c03122d16ee8d6f1aab0b0b2e97d16f2836533567e5`.

## Review and build scope

The revision was checked for cohort-specific bank and request-reuse language, correct cumulative attribution, unchanged Gemini 3.8 table cells, all new Gemini 2.5 table values, accepted counts, blank baseline cells, six empty subsections, citation-key equality, and absence of figure commands. No new literature claim or reference was introduced.

The pre-existing `README.md`, `document_validation.json`, and `validate_document.py` describe the 2026-10-07 build. In particular, that validator hard-codes two tables and seven PDF pages, so it must be updated before it can certify this three-table revision. Compilation, PDF visual review, refreshed build metadata, and source packaging are handled separately by the coordinating agent. This note does not claim those steps have passed.
