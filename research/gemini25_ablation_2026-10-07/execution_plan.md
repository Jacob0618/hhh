# Gemini 2.5 full-rebuild ablation — 2026-10-07

The user explicitly chose Writer, Reader and Judge all Gemini 2.5 Flash, with new memory banks and new retrieval. This is one finite three-condition comparison, not another optimization loop. The old Gemini 3.8 results and paper tables remain unchanged until the new experiment is audited and reported separately.

## Fixed conditions

1. `baseline`: native Mem0 SDK top20 from the new original-session bank.
2. `r03_original`: original R03 Query-Conditioned Fact Selection over the original hybrid80 construction, twenty facts in original pool order.
3. `original_multi_source_closure`: the same R03 selection plus new R05 Source Evidence Recovery and the existing neighboring completion C.

All use the same 1540 original category1–4 questions, gold answers and last-session reference dates; 4620 report slots. This is progressive removal, not an independent R03-off/source-on condition. Only ten physical session banks are needed, shared by the three conditions; all 272 original sessions are written. No grouped writer or new R03 refinement is included.

## Configuration and reuse

- Writer/Reader/Judge: `gemini-2.5-flash`; embedding `gemini-embedding-001`, 768 dimensions; JEV `jev-1.13.0`.
- Fixed `thinkingBudget=0` for all three Gemini roles, no temperature field. Gemini 2.5 does not support the old `thinkingLevel=low`; this explicit compatibility change is not claimed to be equivalent to low thinking. [Official parameter documentation](https://ai.google.dev/gemini-api/docs/generate-content/thinking).
- Existing native writer prompt and JSON memory contract, writer output cap16384. Existing Reader and paper-era Mem0 Judge prompts, plaintext and output cap8192, no15-word limit. Judge parsing and official/custom F1 unchanged.
- Reuse original scientific functions via thin isolated adapters. Do not edit frozen historical source or inject old facts, UUID selections, JEV scores, Reader answers or Judge labels into the new generation run.
- Original R03 pointwise predicate and top20 rule, original pool constructor, source support, multi-window comparison, C and local ranking predicates/thresholds remain unchanged. New bank facts require new JEV inputs and scoring.
- Eight conversation workers maximum; four QA/JEV threads per conversation; existing shared JEV15HTTP/s and embedding3-second physical gate. Source selection sees question/date but no gold/category/answer.

## Execution and integrity

Fresh writer namespace `runs/gemini25_ablation_writer_v1_20261007`; source `runs/gemini25_ablation_source_v1_20261007`; QA `runs/gemini25_ablation_qa_v1_20261007`. Only declared stages dispatch; prepare/check/summary are offline. Credentials only through the existing DPAPI vault, never messages or logs.

Write and review complete development conversations0/1 first, then remaining2–9. Before retrieval, audit all ten terminal banks for exact source/date/raw-generation-to-ADD/history/index bindings and required source semantics. A failed/partial bank does not silently pass. Copy only terminal native search/history files, verify inventory, and open only the copies. New retrieval uses unchanged native top20/capture and original pool builder. Historical deleted caches are not reconstructed; any fresh embedding calls are explicitly accounted for.

Source phases: R03 selection; query-independent support for the new bank facts; multi-window comparison; adjacent completion; local ranking; sealed three-arm inputs. Raw producer/consumer/intent/ledger bindings and derived scores are audited. At the end of each full phase, all known technical failures may receive one uniform recovery with the existing bounded transport retries. Unknown or incomplete intents, local errors and billing failures stop for review; no blind restart or second recovery namespace. Successful NO is never retried.

QA persists Reader/F1 before Judge. All4620 firstpass slots terminate before one uniform technical recovery; successful Reader stages are inherited only within this run's technical recovery. Same exact requests in this run may legitimately share cache. Successful old3.8 scores are not references for new2.5 dispatch. Report firstpass/recovery, valid and fixed1540 denominators, paired flips/categories, official/customF1, Judge, actual usage/cache/unknown costs separately.

Two health snapshots around30seconds apart must show actual stage artifacts, process identity, API activity and disk space. Investigate ten-minute lack of artifacts rather than treating HTTP200 or process existence as success. Long offline checks need CPU/IO or stack evidence. Publish the finite result after independent review; do not tune from its scores or launch new variants automatically.
