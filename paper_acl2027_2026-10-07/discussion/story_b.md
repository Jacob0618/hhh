# Round 2 — Evidence, Experimental Story, and Verification Plan

Prepared 2026-10-07. Planning notes outside the manuscript; no experiments, model/API calls, scientific-code edits, or cache changes were performed. Numerical checks below only recompute the three authorized rows from the existing final digest. Target: ACL 2027, no figures.

## 1. Agreed scientific story

**One-sentence argument.** For question answering over conversational memory, query-conditioned selection determines which stored facts reach the reader, and bounded source recovery restores question-relevant detail around selected anchors; their final combination reaches official LoCoMo F1 60.238 and Judge 84.545% on the fixed 1,540-question evaluation.

The gap sequence should be concrete: **which facts to show → which source details to restore**. Retrieval overlap may prioritize facts without the details requested by a question, motivating per-fact answer-usefulness selection. Even a useful selected fact may compress away wording or relations retained in its source conversation, motivating source evidence recovery. These are design motivations, not measured estimates of how often each failure occurs.

Do not make set-level sufficiency, complementarity, multi-hop reasoning, or verified provenance the central story. Original R03 independently scores facts and selects membership; it does not optimize a joint set objective. SER uses bounded input scopes and partial-support judgments; it does not certify the entire fact or search all conversational history.

| Canonical paper term | Exact experiment mapping | Wording decision |
|---|---|---|
| Query-Conditioned Fact Selection (QFS) | Original R03 | Per-fact usefulness selection; preserve frozen-pool order after choosing members |
| Source Evidence Recovery (SER) | New R05 + C | Window refinement, adjacent-turn completion, and local attachment selection form one module |
| Full method | Original R03 + new R05 + C | The only proposed-method row populated in the main table |
| Official LoCoMo F1 | `official_locomo_f1 × 100` | Report on a 0–100 scale; omit custom token F1 from the principal story |
| Judge | Binary accepted-answer count / 1,540 × 100 | Percentage, distinct from human adjudication |

Provisional neutral title, as directed by root: **Query-Conditioned Fact Selection and Source Evidence Recovery for Conversational Memory**. The project remains JEV-Mem; the paper name awaits the authors. Root has verified a same-name contemporary Jev-Mem paper, arXiv:2609.23986 (Jiang, Li, Li). Related Work must distinguish the actual methods directly using root's verified source notes; do not claim the first use of JEV for memory. No facts about that paper are inferred here from its name.

## 2. Exact method-to-claim boundary

**QFS.** The verified implementation accepts an ordered 20–80 candidate pool; the reported run uses 80. It ranks complete per-fact JEV scores, takes 20 unique members with stable pool-index ties, and restores their original pool order for the reader. There is no absolute score threshold. If scores are unavailable, it preserves the original baseline 20. The candidate-pool composition belongs to inherited setup, not a claim that QFS newly retrieves facts.

**SER.** Original lexical admission yields at most four distinct anchor packets from the selected facts' current input scopes. Each base packet contains the center and eligible immediate neighbors, within its session and scope. For each admitted anchor, new R05 considers up to eight support windows, filters distinct alternatives at a raw score strictly above 0.5, and proposes one by whole-window query BM25, then support score, then original center position. A query-conditioned original-versus-alternative judgment above 0.5 controls replacement. Replacement retains the consuming anchor; packets are deduplicated without admitting outside anchors or refilling the top four.

C then considers left, right, and both-side adjacent full-turn extensions; the chosen extension must score above 0.5 and contain at most five turns. Final local incremental-usefulness ranking selects at most two packets; when two or fewer remain, all are selected without a ranking call. Attachments are appended to the original records, preserving fact prefixes, dates, identities, and total fact count. The algorithm permits at most two attachments; this evaluation observed exactly two per question. The 4,096-byte packet cap is a UTF-8 byte cap, not a token cap.

This establishes an implemented procedure, not a guarantee that each attachment improves an answer. JEV values are uncalibrated decision/ranking signals. The reported full-system difference combines the chosen facts, added source text, window policy, C, local ranking, and reader response; it cannot identify a pure JEV-comparator effect.

## 3. Numerical ledger and experiment order

The main table has at most four requested baseline rows with empty numerical cells, followed by one populated full-method row. Do not populate baseline cells from other papers or insert historical R03/R05 variants. Keep the three measured configurations in the separate **Progressive Component Removal** table:

| Configuration | QFS | SER | Official F1 | Judge (%) | Accepted / total |
|---|:---:|:---:|---:|---:|---:|
| Full method | + | + | 60.238 | 84.545 | 1,302 / 1,540 |
| Without source recovery | + | − | 52.547 | 76.818 | 1,183 / 1,540 |
| Without both modules | − | − | 51.044 | 74.870 | 1,153 / 1,540 |

Numbers were checked directly against `final_result_digest_v1.json`, using keys `original_multi_source_closure`, `r03_original`, and `baseline`. F1 and Judge are scaled by 100 and rounded only at display time.

| Existing contrast | Official F1 difference | Judge difference (percentage points) | Net accepted answers | Valid reading |
|---|---:|---:|---:|---|
| Full − QFS only | +7.691 | +7.727 | +119 | SER added with QFS present |
| QFS only − base pipeline | +1.503 | +1.948 | +30 | QFS added with SER absent |
| Full − base pipeline | +9.194 | +9.675 | +149 | Entire final pipeline difference |

These are conditional steps in one cumulative sequence. The last row of the first table is not independent `w/o R03`: SER is also absent. The unmeasured `(−,+)` configuration is omitted from this table. Do not claim independent module effects, positive synergy, statistical significance, or matched-token superiority.

Recommended Experiments sequence:

1. **Experimental Settings:** fixed 1,540 questions over ten conversations; metric definitions; shared candidate/fact setup; frozen reader/judge protocol; exact-request inheritance/reuse disclosed once.
2. **Main Results:** one populated full-method row; report the absolute result in one short paragraph. Blank baseline cells support no SOTA conclusion.
3. **Progressive Component Removal:** explicit states and the measured downward sequence, with one sentence identifying the conditional interpretation.
4. The six short unperformed-analysis headings below, each with an empty body. All plans remain in these external notes.
5. **Related Work**, then **Conclusion**, following the exemplar's functional sequence without copying wording or importing its results.

### Abstract-safe result sentence

“On a fixed evaluation of 1,540 LoCoMo questions, the complete pipeline achieves an official F1 of 60.238 and a Judge acceptance rate of 84.545%. Progressively removing source evidence recovery and then fact selection reduces Judge acceptance to 76.818% and 74.870%, respectively.”

Use the first sentence alone if the abstract is tight. Do not turn the second into two independent ablation claims. The abstract can explain the two operations and end with this result; it need not enumerate every caveat.

### Conclusion-safe interpretation

“The measured removal sequence supports combining query-conditioned fact selection with bounded source evidence recovery in this conversational-memory pipeline.”

This is narrower than claiming a general memory architecture advantage, cross-model robustness, efficiency, or a causal benefit of JEV itself.

## 4. Six manuscript placeholders

These are the exact short English titles to insert, with **no paragraph, table, number, or plan underneath** in the manuscript:

- Selector Controls
- Module Decomposition
- Matched-Token Comparison
- Generalization
- Evidence Analysis
- Hyperparameter Analysis

## 5. External verification plan — not manuscript prose

No item below is executed or presented as a result. These would extend analysis sections, not add multiple proposed methods to the current main experiment. Generalization is the highest evidential priority because the existing cohort has been used during exploration.

| Empty manuscript title | Essential question | Minimal future verification design | Claim unlocked only after verification |
|---|---|---|---|
| Selector Controls | Is the JEV decision better than a cheaper rule on identical candidates? | Freeze candidate pools, support alternatives, downstream C/ranking, reader, and judge. Compare QFS JEV with same-pool lexical selection. Separately replace only the R05 comparative decision with retain-original and a prespecified lexical rule on the same displayed alternatives. If studying final rank, freeze the post-C packets and vary only that ranking. Keep the contrasts distinct; do not bundle them into one causal estimate. | A specified selector's conditional advantage over its actual same-candidate control; not a general advantage of every JEV call. |
| Module Decomposition | What does each module contribute in the final context? | Add the missing QFS-off/SER-on condition under a prespecified native-20 input rule to complete the QFS×SER design. Separately hold original QFS and new R05 fixed while removing C; compare this with QFS only and full SER. Fix scope/admission and count policies and report their realized changes. | QFS with SER present, QFS×SER interaction, and C's conditional contribution. Do not describe C as meaningful without a source window. |
| Matched-Token Comparison | Does source evidence help beyond using a larger reader input? | Predeclare common reader-token ceilings using the actual tokenizer/provider accounting. Allocate whole facts and complete source turns under those ceilings; compare source augmentation with additional fact context from the same available memory pool. Report actual tokens, answer quality, and latency separately; use no padding to manufacture equal lengths and no UTF-8-byte proxy. | Quality under a stated common token ceiling; efficiency only if full measurement supports it. |
| Generalization | Does the frozen policy hold beyond the explored cohort? | Freeze prompts, thresholds, candidate policy, and naming before evaluating untouched conversations; split at conversation level. Keep the main reader/judge protocol first. A different backbone or benchmark is a separate subsequent transfer check, not a substitute for unseen conversations. Use conversation-level paired uncertainty and publish the exact population. | Unseen-conversation performance; broader transfer only for the specific additional protocol actually tested. |
| Evidence Analysis | Are gains associated with recovered requested detail, and are labels reliable? | Prespecify a conversation-stratified sample of gains, regressions, and unchanged cases. Inspect original facts, final attachments, and answers; record detail coverage, explicitness, distractor competition, and retained answer evidence. Have human assessors review answers without arm identifiers and compare their decisions with Judge. Use sampling weights for prevalence estimates; directed cases remain illustrations. | Supported evidence pathways and human/Judge agreement on the sampled population, not universal explanations from a few chosen cases. |
| Hyperparameter Analysis | Is the result dependent on the fixed budgets or operational thresholds? | Freeze the final policy and vary one factor at a time in a compact prespecified range: selected-fact budget, source-candidate/attachment cap, support/comparison/C threshold, and window extent. Prioritize the attachment budget and 0.5 decision thresholds; keep raw scores uncalibrated. Tune only on a separate development split and reserve the holdout for the frozen choice. | Local sensitivity to tested controls and budgets; no claim of global robustness or optimality. |

Prioritize a small number of contrasts for a later authorized study. This table is a reviewable plan, not an instruction to run every possible arm or to expand the current results table.

## 6. Place each boundary once

- **Method:** scopes are input-exposure boundaries, raw judgments are uncalibrated, and attachments preserve fact count but add source text. Explain these where the algorithm operates.
- **Settings/reproducibility:** this is a fixed, repeatedly explored cohort; inherited references were validated through complete requests and raw responses, and identical requests could reuse outputs. These are aligned paired comparisons, not independent complete reruns.
- **Removal caption/prose:** states are cumulative and conditional; new R05 and C form SER.
- **Limitations:** ten conversation clusters and no untouched evaluation; no same-candidate cheap-selector or token-matched control; Judge acceptance is not human truth. Avoid duplicating this list in every result paragraph.
- **External notes only:** old/new policy search history, retry ledgers, exploratory case details, and these future plans. Do not import seven experimental arms into the paper's principal story.

### Source anchors

- `D:/jev mem/research/r03_r05_refinement_2026-10-06/final_readout_v1.md`
- `D:/jev mem/research/r03_r05_refinement_2026-10-06/final_result_digest_v1.json`
- `D:/jev mem/research/r03_r05_refinement_2026-10-06/result_discussion_v1/final_discussion_consensus_v1.json`
- `D:/jev mem/jev_mem/evidence_selection_r03.py`: original QFS membership/order/fallback policy.
- `D:/jev mem/jev_mem/source_append_r05.py`: lexical admission, local top-two selection, exact record append.
- `D:/jev mem/jev_mem/r05_refinement_v1.py`: same-anchor alternative proposal and question-conditioned replacement.
- `D:/jev mem/jev_mem/source_support_stack_v1.py`: S8 support windows, adjacent completion, bounded final ranking.

Direct discussion with agent A agreed on the per-fact-usefulness → source-detail sequence and rejected set-sufficiency language. Root subsequently fixed the six placeholder titles, the main-table limit, and the provisional neutral title. The method names and experimental states above follow those decisions.
