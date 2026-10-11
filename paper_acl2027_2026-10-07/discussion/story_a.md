# Round 2 — Scientific story A

Planning notes only, 2026-10-07. The final method is original R03 + new R05 + C. No experiments, experimental APIs, or scientific code edits were performed.

## Recommended central argument

A conversational memory pipeline must make two different decisions before answering: which stored facts contain information requested by the question, and which original source details should accompany those facts. The paper should follow this actual execution order. Its contribution is query-conditioned fact selection followed by bounded source evidence recovery. It does not introduce a memory writer, guarantee an answer-complete evidence set, or retrieve independently from the whole dialogue.

Use two exact module names throughout: **Query-Conditioned Fact Selection (QFS)** and **Source Evidence Recovery (SER)**. QFS denotes original R03. SER denotes new R05 together with neighboring completion C. Historical variant labels belong only in internal notes or reproducibility mappings, not the main conceptual story.

## Two limitations → two modules → bounded evidence

| Limitation / motivation | Implemented response | Evidence available | Boundary |
|---|---|---|---|
| Candidate retrieval can prioritize topical or name overlap although a fact does not explicitly supply the requested actor–event–relation–value information. With a fixed number of presented facts, this makes fact membership consequential. | QFS scores each candidate's explicit usefulness for the question and selects the top 20 facts from a fixed pool, then restores pool order. | QFS condition: F1 52.547 and Judge 76.818; base condition: 51.044 and 74.870. | This is a whole QFS-stage contrast to the base pipeline, including the candidate pool configuration. It does not isolate a JEV advantage against a same-pool selector, prove redundancy reduction, or establish set complementarity. |
| Even a useful compressed fact may omit an explicit detail, condition, referent, or temporal relation retained in its exposed source dialogue. A locally retrieved window can also center on a less useful part of that source. | SER preserves the admitted anchors, proposes partially supported alternative windows, compares them against the original windows for the query, optionally adds useful adjacent turns, and appends at most two ranked source packets. | Full condition: F1 60.238 and Judge 84.545; QFS-only: 52.547 and 76.818. | This is joint evidence for source recovery plus C. It does not isolate C, the window comparator, or source selection, and is not a matched-token comparison. |

These are operational motivations, not measured claims about the frequency of errors. Avoid making “insufficient evidence” the first gap: it invites an unsupported promise that independent pointwise QFS scores certify the selected set. Prefer **explicit answer usefulness** for QFS and **question-relevant source detail** for SER.

## Abstract outline: one paragraph, seven or eight moves

1. Establish question answering over accumulated conversation and the role of compact fact memories.
2. Explain the two decision problems compactly: retrieval relevance can diverge from explicit answer usefulness; useful fact text can omit requested source detail.
3. Introduce the final framework as QFS followed by SER, operating on an existing fact memory.
4. State QFS's concrete action: score candidate facts against the question, retain 20, preserve their original content and pool order.
5. State SER's concrete action: recover source windows for admitted fact anchors, accept guarded window refinements, and complete local context when adjacent turns add a requested relation.
6. Close the method chain: rank the resulting source packets for information less explicit in their own anchor facts and append at most two before answer generation.
7. Report evaluation scope and absolute results: 1,540 fixed LoCoMo questions, official F1 60.238 (0–100) and Judge 84.545%.
8. Optional if space permits: the measured progressive removal lowers Judge to 76.818 and 74.870. Name it progressive/cumulative, not two independent ablations.

Keep detailed caps and thresholds in Method, except the 20-fact/two-attachment distinction if it clarifies what the reader receives. Do not end with an unsupported superiority statement.

## Introduction paragraph outline

**Paragraph 1 — Task and evidence bottleneck.** Start with answering a question about earlier conversations. Explain that compact external facts let the system retain useful history while keeping answer context bounded. End on the specific obligation to construct useful evidence for the current question. No general claims about a new writer or unlimited memory.

**Paragraph 2 — Existing response and two decisions.** Give a short, citation-supported account of fact-based memory, retrieval or reranking, and access to source dialogue. Organize it around the distinction between selecting a fact and exposing the detail behind it. This paragraph prepares the two gaps; it should not become an inventory of loosely related architectures.

**Paragraph 3 — Two operational limitations.** First, similarity or topical match alone need not identify an explicit requested relation; use actor, event, relation, and value as concrete examples rather than the vague word “sufficiency.” Second, compression and narrow source localization can leave a requested condition or reference implicit even when the fact is relevant. Explain each consequence immediately. Do not assert that either issue dominates measured failures unless a future analysis establishes that.

**Paragraph 4 — Matched response and evidence.** Name QFS and SER in the same order. Describe QFS's independent usefulness ranking and SER's bounded window comparison, neighboring completion, and attachment ranking in two connected sentences. Close by reporting the fixed evaluation and absolute full-method result. A short final sentence may give the measured cumulative removal chain. This paragraph already carries the contribution content; a repeated bullet list is optional and probably unnecessary.

## Method details the story must preserve

### Preliminaries / inherited candidate preparation

- Define a canonical fact record by its original id, text, and observation date. Define the query and optional reference date separately. Dates of observation are not automatically dates of events.
- Define each fact's **current input scope** as the source turns exposed during its associated memory operation. This is an input boundary, not a certified semantic provenance mapping.
- The original R03 pool is not simply the native top80: concatenate baseline20, up to40 additional native candidates, up to20 positive BM25 facts found by one full-bank scan outside the entire original native80, then unused native candidates to fill remaining capacity. All measured pools contain 80 records; the implementation accepts 20–80.
- State that this pool is fixed across the reported refinement comparisons. Do not imply that the native baseline itself chose from the same hybrid pool.

### QFS

- The exact original predicate asks whether an individual record explicitly supplies at least one requested item with the relevant actor, event, relation, or value. A requested list member or explicit step of a multipart answer can count; topical/name match alone cannot. This is a memory-relative usefulness judgment.
- One score per candidate is obtained with JEV 1.13.0; raw scores rank records and are not calibrated truth probabilities.
- Select top20 with frozen pool index as the tie-break, then restore selected members to frozen pool order. There is no score threshold and no set-level comparison, sequential complement test, new-R03 replacement proposal, or learned diversity objective.
- One compact equation can express top20 membership followed by ordering. Do not obscure the ordering operation inside an unexplained set notation.
- If all required scores are unavailable, retain the baseline20 rather than inventing scores; implementation details can place this in reproducibility material.

### SER

1. **Admit anchors.** For each selected fact, search only its current input scope. Original lexical centers require positive question–turn and fact–turn BM25. Build center±1 complete-turn packets within the same scope/session and the 4096-byte packet cap. Choose one per fact by question BM25, fact BM25, then center order; deduplicate packets; retain top4 by the same priorities with fact-slot tie-breaking.
2. **Propose alternative windows.** For each admitted anchor, construct at most8 same-scope center±1 windows ordered by fact BM25 and source position. Query-independent support judgments ask whether a window supports at least one substantive atomic claim in the fact. This is partial support, not whole-fact certification. Among distinct windows with support score>0.5, propose one by whole-window query BM25, support score, then center position. Query BM25's corpus contains all eligible S8 windows before support filtering.
3. **Compare original and alternative.** Replace only when a query-conditioned comparison score>0.5 indicates the alternative adds or clarifies requested explicit detail while retaining the original window's question-relevant support. This is the comparator's criterion, not a guaranteed property of every accepted window. Preserve the top4 admitted anchor universe; duplicate packets merge and no new outside anchor fills a collision.
4. **Complete adjacent evidence (C).** Consider adding one complete turn left, right, or both within the same scope/session. The maximum completed packet contains five turns. The predicate asks whether added turns resolve a concrete requested answer, referent, attribution, condition, or temporal relation missing from the base window. Choose the highest score if>0.5; tie-break by fewer added turns and then variant order. Otherwise keep the base packet.
5. **Rank and append.** Rank the resulting candidates by local information gain relative to each candidate's own anchor fact for the question. It does not assess all other19 facts or full-conversation completeness. Take top2 with no absolute acceptance threshold; if ≤2 candidates remain, attach all without ranking. In the measured source conditions exactly two records per question were augmented. Preserve fact text as the prefix, observation date, identities, record count, and order.

Present the five SER steps as concise internal paragraphs within one module, not five separately proven contributions. A second equation can express guarded window replacement; a third can express final context assembly. The original and refined windows need distinct notation, and C should not simultaneously denote the overall answer context.

## Results and claim language

The main table contains the full method's F1 60.238 and Judge 84.545; all other baseline numerical cells remain blank as requested. That table supports an absolute measured outcome, not a comparative superiority conclusion.

The measured component chain is (QFS on, SER on) → (on, off) → (off, off), with F1 60.238 → 52.547 → 51.044 and Judge 84.545 → 76.818 → 74.870. Label it **Progressive Component Removal**. Removing SER changes Judge by −7.727 percentage points; subsequently removing QFS changes it by −1.948. There is no QFS-off/SER-on measurement. Do not label the last row simply “w/o QFS” if that implies SER remains.

The evidence supports usefulness of the full selection-and-recovery pipeline under this fixed protocol. It does not show SOTA, blind-test generalization, savings in total cost or attachments, universal multi-hop gains, guaranteed support preservation, a new writing mechanism, or independent whole-dialogue retrieval. Full gain over base is +9.675 Judge points and +9.194 F1 points; it cannot all be attributed to the new R05 refinement or to JEV.

## Discussion with story B

Direct exchange reached agreement on (i) the two concrete decision problems, (ii) replacing the sufficiency/complementarity motif with pointwise answer usefulness, (iii) the exact SER sequence and preserved top4 admission, and (iv) the cumulative interpretation of the measured component chain. B and root finalized six empty analysis headings: Selector Controls; Module Decomposition; Matched-Token Comparison; Generalization; Evidence Analysis; Hyperparameter Analysis. Detailed proposals belong in planning notes; manuscript bodies under these headings remain empty.

## Source anchors

- `jev_mem/evidence_exchange_r01.py`, `build_pool`: hybrid frozen pool composition.
- `jev_mem/evidence_selection_r03.py`, `build_request`, `choose_jev`, `_decision`: original QFS scoring, membership, and order.
- `research/jev_autoresearch_2026-10-04/R03_selection_protocol_v2.json`: exact original pointwise predicate.
- `jev_mem/source_append_r05.py`, `build_candidates`, `choose_jev`, `selected_facts`: original anchor admission and final append.
- `jev_mem/r05_refinement_v1.py`, `build_candidate_variants`, `resolve_multi`: partial-support alternatives and guarded comparison.
- `jev_mem/source_support_stack_v1.py`, `build_support_plan`, `build_closure_plan`, `resolve_closure`: support locator and C.
- `research/r03_r05_refinement_2026-10-06/final_readout_v1.md`: frozen results and evidence limits.

