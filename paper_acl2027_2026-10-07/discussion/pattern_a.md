# Round 1 — Abstract and Introduction pattern analysis

Status: analysis only; no manuscript prose written. Analyst A has discussed the transfer with analyst B; agreed points are recorded below.

## Sources and boundary

- Writing skill: `C:/Users/86139/.codex/skills/paper-pattern-transfer/SKILL.md`.
- Exemplar: *From Single to Multi-Granularity: Toward Long-Term Memory Association and Selection of Conversational Agents* (MemGAS), `C:/Users/86139/Downloads/2505.19549v2.pdf`; text inspected through `reference/exemplar_extracted.txt` and actual layout inspected on `reference/exemplar-01.png`, `exemplar-02.png`, and `exemplar-03.png`.
- Target evidence: `D:/jev mem/research/r03_r05_refinement_2026-10-06/final_readout_v1.md`.
- Transfer only rhetorical functions, sequencing, level of detail, and equation-to-prose rhythm. Do not transfer MemGAS wording, equations, GMM, entropy routing, graph operations, benchmark coverage, performance claims, or theory.
- The actual final method is original R03 + new R05 + C. The two paper-level modules should be fact selection (original R03) and source evidence recovery (new R05 with C). C is an internal step of the latter module, pending exact code terminology verification by the parent.

## 1. Abstract: nine sentence-level functions

The abstract is one compact paragraph. It has a complete problem–method–evidence arc, rather than a separate list of contributions.

| Move | Function in MemGAS | Pattern to transfer | Target constraint |
|---|---|---|---|
| 1 | Establishes conversational LLM agents as the task setting | One sentence explaining the deployed task and memory need | Avoid generic praise of LLM capabilities |
| 2 | Increasing dialogue history meets limited context and harms coherent/personalized responses | State the concrete constraint and its consequence | Do not claim context expansion itself is our contribution |
| 3 | Names retrieval-augmented memory as the existing response, then identifies its design restriction | Acknowledge the useful existing pipeline before narrowing the gap | Target is an existing candidate memory pipeline, not a new writer/embedding/store |
| 4 | Connects the restriction to missing information or noise and poorer answers | State the causal reason that the gap matters for answering | Present target failure modes as design motivations unless independently demonstrated |
| 5 | Names the framework and compresses its central design into one sentence | Introduce one final method and two coordinated stages | No history of old/new R03 and R05 in the abstract |
| 6 | Gives the first mechanism with its object and operation | First operation answers first gap | Original R03, after implementation verification; no invented diversity or complementarity guarantee |
| 7 | Gives the second mechanism and the trade-off it addresses | Second operation answers second gap | New R05 + C form one source evidence module; C has no isolated estimate |
| 8 | Briefly names the final refinement before answering | Close the execution chain with what the reader receives | Keep final facts, appended sources, and answer generation consistent with the real implementation |
| 9 | Ends with benchmark scope and achieved result | Close on the measured result, not aspiration | Fixed 1,540 LoCoMo questions; Judge 84.545 and official F1 60.238. No SOTA/generalization claim from an otherwise blank comparison table |

The target need not mechanically keep nine sentences. A compact 7–9 sentence abstract can combine the two target limitations while preserving their order and give one method sentence per module. Target abstract evidence should be absolute measured scores, with the exact metric scale clear. It must not inherit MemGAS's four-benchmark, multiple-top-K, retrieval-metric, or across-type superiority claims.

## 2. Introduction: four physical paragraphs, six rhetorical jobs

The original Introduction consists of four main paragraphs spanning pages 1–3. The first figure interrupts the visual flow at the top of page 2; it is not a new prose paragraph. There is no separate bulleted contribution list.

### Paragraph 1: setting → growth pressure → consequence → external memory response

The first paragraph begins with the task value of conversational assistants, narrows to accumulating interaction history, explains the context limitation, connects that limitation to response quality, and only then introduces retrieval-based external memory as a response. Its ending explains what this response enables.

**Transfer:** begin with the need to answer questions over prior conversations, explain why an external memory pipeline is used, and end with the evidence-selection obligation imposed by that pipeline. Do not linger on broad applications or claim that the target invents external memory.

### Paragraph 2: organize existing approaches along axes needed for the coming gap

The second paragraph surveys two design dimensions: memory segmentation/representation, then structured construction. It names representative families and their actual function. This is a targeted bridge to the next paragraph, not an exhaustive literature catalog.

**Transfer:** organize the target background around compressed facts, retrieval/selection, and access to original dialogue. Only include classifications supported by verified citations. Use this paragraph to prepare the distinction between choosing a fact and recovering the evidence behind it. Do not reuse MemGAS's segmentation taxonomy merely to mimic citation density.

### Paragraph 3: explicit two-gap structure, each with a consequence

The paragraph explicitly labels two limitations. For the first, it identifies the missing relation, explains the information loss, and uses the illustrated query to show how partial retrieval produces an incorrect answer. For the second, it identifies fixed selection, explains the completeness/noise trade-off, and points to a granularity experiment. Each named limitation is followed immediately by why it damages the task.

**Transfer:** preserve `gap → operational cause → answer consequence` for each of two target issues:

1. Query relevance among candidate facts does not itself establish that the selected evidence is sufficient for the current answer. This motivates query-conditioned selection. Treat this as a design rationale; the available result does not by itself prove a specific complementarity or deduplication mechanism.
2. A compact fact can omit question-relevant details that remain in the original dialogue. Selecting that fact alone need not expose those details to the reader. This motivates selective source evidence recovery with adjacent context.

The two issues may be separate paragraphs in the ACL layout, but must remain explicit, ordered, and causally connected to the task. The requested draft has no figures, so it cannot depend on an invented motivational figure or unrun oracle analysis. Existing targeted cases may later support a modest observation, but cannot establish error prevalence.

### Paragraph 4: method identity → first response → second response → execution closure → evidence

The final paragraph names MemGAS, repeats its two strategies in the same order as the two limitations, describes the operation of each, closes the retrieval-to-answer chain, and ends with results. This supplies contribution content without repeating it as a separate list.

**Transfer:** introduce the single final pipeline, describe original R03 first, then new R05 + C, and explain that selected facts retain their existing prefix/date while selected records receive source attachments. Include only code-verified operational details in manuscript prose. End with the fixed evaluation and the measured absolute result, plus carefully described cumulative removal evidence if space allows. Do not promise that every component independently improves every metric or category.

## 3. Rest of the exemplar: order and reusable rhythm

| Exemplar location | Actual order and role | Target recommendation |
|---|---|---|
| §2 Methodology lead | One sentence previews definition, construction, routing, retrieval/filtering | One short input-to-output overview |
| §2.1 Preliminary | Task/data objects and retrieval-to-answer goal | Define query, existing fact candidates, selected facts, source windows, and reader output once |
| §2.2 Construction | Representation before transformation; metadata before association | Do not add a new construction contribution; explain inherited memory input in preliminaries |
| §2.3 Router | Purpose, input scores, compact operation, definitions, resulting decision | Fact Selection: purpose → actual input → verified operation → output |
| §2.4 Retrieval and Filter | Earlier outputs feed ranking, selection, filtering, then answer generation | Source Evidence Recovery: verified window choice → C → attachments → final reader context |
| §3.1 Settings | Datasets/metrics → baseline families → implementation details | Keep model/protocol/candidate and final-fact counts here; distinguish inherited/reused requests from new independent runs |
| §3.2 Overall Results | Main QA table and retrieval table, interpreted by task | Main QA results table only; only the single final method row is populated |
| §3.3 Ablation Study | Component removal table and interpretation | Rename to Progressive Component Removal; show component-state columns |
| §3.4 Detailed Comparison | Query types, then top-K | Include only evidence-backed analysis; required new experiments receive concise empty headings |
| §4 Related Work | Retrieval → long-term memory management → structured memory | Group around relevant actual comparison axes; do not import irrelevant graph emphasis |
| §5 Conclusion | Method contribution then bounded result | State what was built and measured, without broader superiority claims |
| End matter | Ethics, reproducibility, references, appendices | Follow ACL requirements, with only verified facts about artifacts/licenses |

MemGAS uses five numbered display equations and additional inline notation, not a rigid one-formula-per-step template. The reusable move is purpose → compact equation → symbol definition → downstream role. For the target, retain only formulas corresponding to implemented operations; do not invent theoretical guarantees or import entropy/GMM/PPR mathematics. Three conceptual equations may be sufficient, but the exact count depends on the verified code.

## 4. Frozen result and claim boundary

The main experiment must contain only one populated target-method row:

- Final pipeline: original R03 + new R05 + C.
- Official LoCoMo F1: 60.238 on a 0–100 scale.
- Judge: 84.545%, 1,302/1,540.
- Other baseline rows: numerical cells left empty per the user's instruction; this prevents a current overall-comparison or SOTA conclusion.

The measured removal chain is:

| Fact selection | Source recovery (new R05 + C) | Official F1 | Judge | Interpretation |
|---|---|---:|---:|---|
| On | On | 60.238 | 84.545 | Final pipeline |
| On | Off | 52.547 | 76.818 | Remove the source recovery module |
| Off | Off | 51.044 | 74.870 | Remove both modules / base memory pipeline |

The first removal changes 7.727 percentage points; the next cumulative removal changes 1.948 points; together they span 9.675 points. These are conditional steps in a removal chain, not two independent component effects. There is no measured `Fact selection Off / Source recovery On` row. Thus 74.870 must never be labeled simply `w/o R03` while implying the full method's source module remains active. R05 alone and C alone are also not separately identified.

Other limits that directly govern writing: the 1,540 items are repeated exploration data over 10 conversations, not a fresh blind test; existing requests can be inherited/reused, not independent reruns; equal fact count is not equal tokens; UTF-8 bytes are not model tokens; there is no complete dollar-cost claim; the final method's gains cannot all be assigned to the new R05 revision or to JEV alone.

## 5. Agreed transfer decisions with analyst B

1. Match the exemplar's causal sequence, not its wording or scientific content: setting → existing response → two gaps and consequences → two matching operations → evidence.
2. Use two paper-level method modules in execution order: Query-Conditioned Fact Selection (original R03) and Source Evidence Recovery (new R05 + C). Context assembly can close the second subsection. These are provisional descriptive names until parent code verification.
3. Keep the two limitations modest: relevant facts may not be sufficient evidence, and compressed facts may omit useful original detail. Do not assert demonstrated complementarity, noise removal, universal multihop improvement, or a new memory-writing method.
4. Follow Settings → Main Results → Progressive Component Removal → relevant further analyses, with no figures. Empty requested experiment headings contain no invented findings.
5. Present the measured removal chain with explicit module-state columns. Do not claim independent R03, R05, or C effects where the required control is missing.
6. The Introduction's last paragraph can carry the contribution content, as the exemplar does; an extra contribution list would repeat the same mechanism and result unless the target needs it for clarity.
7. Analyst B explicitly confirmed these decisions in direct discussion. Optional short labels QFS and SER may be used if they reduce repetition, with their actual definitions established once. The unmeasured `(Off, On)` condition is omitted, rather than presented as an empty ablation row that might imply a completed independent control.

## 6. Checks for the writing round

- Every method verb must match an actual operation; every formula must have a code or artifact anchor.
- Keep the names and ordering of the two limitations, the two strategies, and Method subsections aligned.
- Give background citations to actual source publications, not the exemplar's bibliography copied wholesale.
- Main results, captions, abstract, and conclusion must use the same final method and metric scale.
- No comparison language can exceed the populated evidence; no independent-ablation language can exceed the component states tested.
- Compare source and target for exact sequences of eight or more words, excluding ordinary headings and necessary technical phrases.
- Visually check ACL column fit, equations, and tables after compilation; the exemplar's single-column page proportions are not transferable to ACL.
