# Story consensus recommendation — A

Story A and Story B discussed the scientific framing directly and agree on the following final writing constraints. This is a separate consensus recommendation to avoid conflicting writes to a shared file.

1. Use the neutral working title **Query-Conditioned Fact Selection and Source Evidence Recovery for Conversational Memory**. A same-name contemporary Jev-Mem paper reported by root requires direct Related Work positioning; do not claim first use of JEV for memory.
2. The only proposed pipeline is original R03 + new R05 + C. Paper modules are QFS and SER, with C internal to SER. No new-R03 precision/complement/set-comparison algorithm may enter the manuscript.
3. Gap 1 is the mismatch between retrieval overlap and explicit requested information in individual facts; QFS ranks pointwise answer usefulness. Gap 2 is omitted or less-explicit source detail behind useful compressed facts; SER recovers bounded source windows, checks alternatives against originals, completes adjacent relations, and ranks local attachments.
4. Do not make set sufficiency, complementarity, certified source truth, or guaranteed no-loss refinement a central claim. Prompts ask for these local conditions where applicable; scores do not prove them.
5. Describe the hybrid 80-pool correctly: baseline 20 + up to 40 other native candidates + up to 20 positive full-bank BM25 results outside complete native 80 + native refill. QFS chooses 20, preserves text/date and pool order, and has no score threshold. Its removal contrast does not isolate the scorer from pool construction.
6. SER preserves original top 4 anchor admission. At most 8 partial-support windows per anchor feed one alternative; support>0.5 gates candidates, whole-window query BM25 selects the alternative, and original-versus-alternative comparison>0.5 gates replacement. C tests complete left/right/both extensions up to 5 turns. Final local ranking chooses at most 2 attachments without a score threshold. Original 20 fact records are preserved with appended text; exactly 2 were augmented per measured question.
7. Full method scores are official F1 60.238 and Judge 84.545 on 1,540 fixed questions. Main-table other baseline values remain blank. Existing results cannot support SOTA or a fresh blind-test claim.
8. Component evidence is explicitly cumulative: (QFS,SER)=(on,on)→(on,off)→(off,off). F1 60.238→52.547→51.044; Judge 84.545→76.818→74.870. Call this Progressive Component Removal, not two independent leave-one-out results.
9. Preserve the exemplar's paragraph logic and purpose→equation→definition→handoff rhythm. Introduction can use four paragraphs: task; relevant existing approaches; two limitations; two responses and bounded evidence. Do not transfer its algorithm or results.
10. No figures. Final analysis headings have empty bodies: Selector Controls; Module Decomposition; Matched-Token Comparison; Generalization; Evidence Analysis; Hyperparameter Analysis. Detailed proposed studies belong only in planning notes.

Full story, abstract moves, Introduction outline, Method checklist, and code anchors are in `story_a.md`. Evidence plans and companion checks are in Story B's separate notes.

