# Pattern Transfer Discussion B: Method, Experiments, and Appendices

First-round planning only; not manuscript prose. Prepared on 2026-10-07.

## Sources and scope

- Skill: `C:/Users/86139/.codex/skills/paper-pattern-transfer/SKILL.md`.
- Exemplar: MemGAS, `C:/Users/86139/Downloads/2505.19549v2.pdf`; extracted full text in `reference/exemplar_extracted.txt`. Actual page images 3 and 5 were inspected to check paragraph and equation pacing.
- Target evidence: `D:/jev mem/research/r03_r05_refinement_2026-10-06/final_readout_v1.md`.
- The target implementation is the original R03 plus the new R05 and shared neighbor completion C. Exact scientific operations remain subject to root's source-code verification.
- No experiments were run, no experimental API was called, and no scientific code was edited.

## 1. What the exemplar actually does

### Method subsection order and paragraph roles

The introduction ends on page 3, followed immediately by Section 2, Methodology. A one-paragraph roadmap precedes four subsections.

| Exemplar location | Paragraph function and order | Transferable function | Excluded scientific content |
|---|---|---|---|
| 2.1 Preliminary | Define dialogue session, memory bank, query, retrieved memory, and answer; then state the two central method challenges | Define the objects consumed and produced by the target before detailing modules | Four memory granularities and a learned/adaptive granularity mapping |
| 2.2 Multi-Granularity Association Construction, opening | State the limitation of a prior operation and the purpose of this construction | One short purpose statement linked to the introduction's first gap | A claim that our system constructs deeper cross-memory associations |
| 2.2, metadata paragraph | Input session -> metadata generation and segmentation -> displayed transformation -> symbol definitions -> bundled memory representation | Define an existing representation before its transformation; follow each formula with definitions and its output's role | Generated summaries/keywords/turn decomposition as new contributions unless verified |
| 2.2, dynamic association paragraphs | Introduce update event -> compute similarity -> identify accept/reject groups -> update an association structure | Explain operation in execution order and hand the output to the next stage | GMM, accept/reject clustering, graph construction, biological consolidation claims |
| 2.3 Multi-Granularity Router | Motivate query adaptation -> define scores -> normalize -> calculate uncertainty -> turn uncertainty into weights -> explain why weights alter later processing | Explain why a decision is needed before presenting the quantity used to make it | Entropy, granularity routing, inverse-entropy weights, learned policy claims |
| 2.4 Memory Retrieval and Filter | Recall preceding outputs -> combine relevance -> retrieve -> limit candidate context -> filter -> supply answer generator | Begin with explicit inputs from preceding stages; end with exact reader input | Weighted multi-granularity scores, PPR propagation, graph traversal, new LLM filter |

The visible layout is compact: descriptive subsection headings, occasional bold run-in labels for internal steps, purpose stated before a formula, and explanatory prose immediately after. The target can borrow this rhythm without copying the exemplar's four-granularity structure or formulas. The target user requested no figures, so no architecture or motivating figure is transferred.

### Exemplar equation inventory

| Equation | Actual function | Target treatment |
|---|---|---|
| (1) Metadata generation and session segmentation | Defines two construction operations from one input | Borrow the input -> transformation -> output presentation pattern only |
| (2) Memory chunk containing session, turns, summary, keywords | Packages the objects just defined into a unit used downstream | A compact target memory-record definition is optional, but only after its real fields are verified |
| (3) Softmax relevance distribution and Shannon entropy | Defines the router's evidence and uncertainty statistic | Exclude entirely: no entropy mechanism is established for the target |
| (4) Normalized inverse-entropy weights | Turns the uncertainty measure into a module decision | Exclude the formula; transfer only the rhetorical role of explaining how a verified score controls selection |
| (5) Weighted query-memory similarity | Connects routing decisions to retrieval initialization | Use a target context-assembly relation only if it expresses the implemented handoff |

Five displayed equations do not imply the target needs five. Two or three concise structural equations are enough until exact JEV selection and source-window operations are verified. Never replace missing evidence with decorative scoring formulas or an unimplemented maximization objective.

### Experiments, related work, and conclusion

The exact main-text order is:

1. **3.1 Experimental Settings:** datasets/metrics -> baseline families -> implementation details and shared conditions.
2. **3.2 Overall Results:** a short introduction -> QA results -> retrieval results. Each paragraph moves from aggregate outcome to interpretation of the method.
3. **3.3 Ablation Study:** component removals -> performance changes -> resource interpretation.
4. **3.4 Detailed Comparison Analysis:** query types -> retrieval top-K.
5. **4 Related Work:** retrieval methods -> long-term memory management -> structural memory management. Each family description ends with positioning relative to the proposed mechanism.
6. **5 Conclusion:** restate the problem addressed and method -> summarize demonstrated empirical scope.

Thus the source's actual order is settings -> main results -> ablation -> detailed analysis, not settings -> main results -> detailed analysis -> ablation. The target may preserve this order while omitting unsupported analysis sections. Do not copy the source's broad claims of superiority, robustness, efficiency, or multi-hop reasoning merely because those claims appear beside its tables.

## 2. Target scientific spine and names

The two target concerns can be phrased cautiously as:

1. **Query-conditioned evidence selection:** facts related to the same topic need not be the facts that best support the current question. This motivates selection from an existing candidate pool. It does not establish a learned selector, a complementary-evidence objective, or a new retriever.
2. **Detail loss in compressed facts:** selecting a memory fact may still leave relevant wording or context outside the answer model's input. This motivates restoring bounded source evidence and nearby context. The inspected report does not support a general claim that these omissions are specifically multi-hop failures.

Recommended canonical module names:

| Manuscript name | Short name | Actual experimental configuration | Boundary |
|---|---|---|---|
| Query-Conditioned Fact Selection | QFS | Original R03 | JEV-conditioned selection over the fixed candidate pool; exact score, gating, ordering, and fallback rules need code verification |
| Source Evidence Recovery | SER | New R05 + C | Source-window selection/replacement plus the shared neighboring-evidence completion; C is an internal step, not an independently demonstrated innovation |

Use the eventual system name for the single full method. Keep R03/R05/C as internal traceability labels or appendix mappings rather than unexplained paper branding. Avoid "multi-hop memory reasoning," "graph closure," "router," "learned policy," "optimal selection," and "proven complementarity" unless the implementation and a corresponding claim are independently verified.

## 3. Proposed target Method paragraph skeleton

**2 Method**

One short roadmap paragraph: the input is an existing fact-memory candidate pool plus access to its source conversations; the output is a fixed-size ordered set of facts with source evidence attached to selected records, supplied to the reader. Name QFS then SER in execution order.

**2.1 Preliminaries**

- Paragraph 1: define conversation history, query, fact memory, and the existing candidate retriever. State that memory writing and vector indexing are inherited components, after root verifies their actual implementation.
- Paragraph 2, only if needed: define the candidate pool and selected fact budget; distinguish selecting facts from adding source text. The reported experiment uses 80 candidates and 20 final unique facts; numerical budgets belong naturally in experimental settings unless integral to the method.

**2.2 Query-Conditioned Fact Selection**

- Paragraph 1: purpose and inputs; define precisely what the original R03 receives and what it returns.
- Paragraph 2: verified JEV operation and selection policy in execution order, including any actual fallback or order-preservation behavior. Explain the output as an ordered selection of existing fact records.
- Optional paragraph 3 only if the verified operation needs it: how the final fact set respects uniqueness and a fixed size. Do not expand unrelated experimental branches into additional methods.

**2.3 Source Evidence Recovery**

- Paragraph 1: explain why the selected facts may need source context; identify the implemented eligible facts and source anchors.
- Paragraph 2: the new R05 source-window comparison/replacement mechanism, written from code rather than inferred from the name. The report establishes a constraint to the original top-four anchor range; it does not establish independent retrieval outside that range.
- Paragraph 3: shared neighbor completion C and final source-text assembly. Describe this as part of SER. The report establishes two source-augmented records per question, preservation of fact prefixes and dates, and unchanged final fact count.
- Closing paragraph: define the reader input and generation call. This can remain the last paragraph of 2.3, avoiding a thin extra subsection. If the verified assembly needs substantial explanation, use **2.4 Context Assembly**.

## 4. Target notation and equation proposals

These are interfaces, not replacements for implementation verification.

| Symbol | Intended meaning | Caveat |
|---|---|---|
| q | Current question | Define once |
| H | Source conversation history available to the pipeline | Do not imply every historical passage is independently searched |
| P_q | Ordered fact candidate pool for q | Use an ordered sequence, not an unordered set if order matters |
| F_q | Ordered selected fact records | Record fields and ordering must match actual code |
| L, B | Candidate count and final fact count | Reported settings: 80 and 20 |
| e_i | Source attachment for the i-th selected fact; empty when absent | This is an attachment, not a new memory fact |
| F_tilde_q | Ordered source-augmented reader records | Fact text prefixes and dates are preserved in this experiment |
| y_hat | Reader answer | Do not use a trained-parameter subscript unless describing an actual model |

Recommended conceptual interfaces, expressed here in plain notation pending code verification:

1. `F_q = Select_JEV(q, P_q), |F_q| = B`.
   Purpose: defines the selector's input/output and fixed fact budget. It must be accompanied by actual selection prose; this alone does not explain the algorithm. Do not add an argmax, pairwise interaction term, or redundancy penalty without code evidence.
2. `(e_1, ..., e_B) = Recover(q, F_q, H)` followed by `F_tilde_q = (Attach(f_i, e_i))_{i=1}^B`.
   Purpose: makes explicit that source evidence is attached to selected records and does not become extra facts. Collapse these into one aligned display if retained. Empty attachments preserve the original record.
3. `y_hat = Reader(q, F_tilde_q)`.
   Purpose: closes the execution chain. Prefer inline text if it would otherwise be a decorative standalone equation.

Do not introduce a structured tuple such as `(text, time, pointer)` until the actual record schema is checked. Do not reuse C for a mathematical context object because C already denotes the completion condition in the experimental record. Avoid a separate equation for every implementation constant.

## 5. Minimal target Experiments skeleton

**3.1 Experimental Settings**

- **Data and Metrics.** Define the fixed 1,540-question LoCoMo evaluation, official LoCoMo F1 on a 0-100 scale, and the binary judge acceptance rate. Ten conversations are represented; the exact selection/exclusion rule still needs its artifact citation. Do not call this the full public benchmark without checking that rule.
- **Comparisons.** Main-table baseline names may be grouped briefly, but all baseline numerical cells remain blank under the user's request. The three measured configurations belong in the progressive-removal table. Do not insert literature scores under a different generator/judge protocol.
- **Implementation.** State the frozen 80-candidate pool and order, 20 unique facts, reader/judge Gemini 3.8 Flash with low thinking and no temperature parameter, 8,192 plain-text output setting, and JEV 1.13.0, subject to artifact verification. Explain once that reference conditions were inherited after exact-request/raw-response validation and equal requests could reuse outputs; this is a paired comparison, not an independent complete rerun. Same fact count does not mean same token budget.

**3.2 Main Results**

One concise table with one populated method row: original R03 + new R05 + C has official F1 **60.238** and Judge **84.545%** (**1302/1540**). Other baseline cells are intentionally blank. A short prose paragraph reports the observed full-method outcome without making SOTA or broad relative-performance claims from an incomplete table. No extra main experiment is required.

**3.3 Progressive Component Removal**

Use explicit component state columns rather than ambiguous shorthand:

| Configuration | QFS | SER | Official F1 | Judge (%) | Correct / 1540 |
|---|---|---|---:|---:|---:|
| Full method | On | On | 60.238 | 84.545 | 1302 |
| Without source recovery | On | Off | 52.547 | 76.818 | 1183 |
| Without both modules / base memory pipeline | Off | Off | 51.044 | 74.870 | 1153 |

The sequence is **84.545 -> 76.818 -> 74.870**. The last row is not an independent `w/o R03` result with SER retained. Do not add an unmeasured QFS-off/SER-on row. First removing SER changes Judge by -7.727 percentage points; subsequently removing QFS changes it by -1.948 points. These are conditional steps in a cumulative removal sequence, not two independent marginal effects or proof of synergy. The full configuration is +9.675 points relative to the base condition.

Paragraph move sequence: question addressed -> configurations and controls -> selective score/count evidence -> interpretation limited to the observed pipeline. The evidence supports the usefulness of source augmentation together with neighboring completion under this protocol. It does not separate R05 from C, does not isolate the JEV comparator from a cheaper selector, and does not show a matched-token advantage.

**3.4 Analysis** is optional. Under the user's minimal-experiment preference, keep only explicitly requested unperformed experiments as short English headings with empty bodies. Examples, if retained: **Matched-Budget Comparison**, **Source Recovery Analysis**, **Generalization**. Do not write speculative paragraphs under them, invented tables, or future-tense pseudo-results. Existing exploratory reports need not all enter the main manuscript.

## 6. Related Work, Conclusion, and appendix transfer

Keep **Related Work after Experiments and before Conclusion**, matching the exemplar. Two or three short family paragraphs are sufficient: conversational fact memory; retrieval/selection for memory-augmented generation; access to source context or evidence if supported by verified sources. The final sentence of each family should position an actual target operation, not claim a comprehensive deficiency of all preceding systems.

Conclusion should restate selection plus source recovery, give the demonstrated fixed-evaluation scope, and avoid expanding into general claims about learning, multi-hop reasoning, universal retrieval quality, lower cost, or cross-model robustness.

| Exemplar appendix | Writing function that may transfer | Target treatment |
|---|---|---|
| A Dataset Statistics | Make evaluation population and categories inspectable | Fill only from target split/manifest artifacts |
| B Baseline Structure | Keep baseline implementation descriptions out of the main results | Describe actually configured comparators; do not imply unrun methods have results |
| C Single-/Multi-Granularity Comparison | Decompose a central representation claim | No analogous target claim is established; omit or retain only a specifically requested empty heading |
| D Cost and Efficiency | Define measurement scope and matched-budget controls | Existing partial API accounting does not establish total cost or efficiency; bytes are not tokens, unknown usage is not zero; no efficiency conclusion |
| E Generalization | Test retriever, generator, and query-type changes | Cross-backbone generalization is unmeasured; do not transfer its results or claim language |
| F Hyperparameter Sensitivity | Explain sensitivity to actual method controls | No completed sweep is established; leave an English heading only if requested |
| G Error Analysis | Connect observed failures to bounded interpretations | Six directed cases can motivate specific failure possibilities, not population frequencies |
| H Theoretical Analysis | Formalize assumptions behind a real mechanism | Do not import GMM/entropy theory, proofs, or guarantees |
| I Prompt Design | Make calls and model roles reproducible | Use the actual writer/selector/source/reader/judge prompts as applicable; do not copy MemGAS prompts |
| J Case Study | Show the real intermediate inputs and resulting answer | Use text/table examples only if included; user requested no figures; preserve actual outputs without corrective edits |

Operational failures, retries, cache validation, and audit details belong in reproducibility material if needed. They should not displace the scientific argument. Mandatory venue sections and template requirements remain root's separate verification task.

## 7. Claim-to-evidence gate

| Candidate claim | Status from final readout |
|---|---|
| Full method obtains F1 60.238 and Judge 84.545 on the frozen 1540 questions | Supported |
| Progressively removing SER and then QFS gives Judge 76.818 and 74.870 | Supported with explicit cumulative labels |
| Full method is +9.675 Judge points over the inherited base reference under the aligned protocol | Supported as a whole-pipeline paired difference |
| New R05 alone yields the entire gain | Unsupported; the package includes source text and C |
| Both revised modules independently improve by more than 3 points | Contradicted by this report; original R03 is retained and new R03 net gain is zero |
| Source recovery uses fewer attachment calls | Unsupported; all source arms attach sources to two records per question |
| Equal fact count demonstrates equal information/token cost | Unsupported |
| Gains demonstrate a universal multi-hop or cross-model advantage | Unsupported |
| JEV is better than any cheaper selector on the same source candidates | Unsupported; that comparison is absent |
| New R05 and new R03 have positive synergy | Unsupported; the reported Judge interaction is zero |

## 8. First-round consensus with agent A

Direct discussion with `/root/paper_pattern_a` reached agreement on:

1. Transfer paragraph functions and section/equation pacing, not MemGAS's scientific machinery, wording, or claims.
2. Preserve the two-gap -> two-strategy order: query-conditioned fact selection, then source evidence recovery. C remains internal to the second module.
3. Avoid asserting learned selection, complementarity, de-redundancy, or multi-hop mechanisms before code verification.
4. The main table has exactly one populated proposed-method row; no SOTA claim can be inferred from blank baseline rows.
5. Label the measured decomposition **Progressive Component Removal** with QFS/SER states `(on,on) -> (on,off) -> (off,off)`, not independent leave-one-out ablations.
6. Use no figures. Keep unsupported experiments as short English headings with empty bodies, or omit them when not explicitly required.
7. Related Work follows Experiments and precedes Conclusion. Supplemental material inherits only reproducibility and analysis functions supported by target artifacts.

Root should verify the exact R03 and new R05+C operations before drafting the final mathematical expressions. This file supplies a paragraph skeleton and evidence boundaries, not invented algorithmic internals.
