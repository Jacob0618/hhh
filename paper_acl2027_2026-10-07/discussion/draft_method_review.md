# Draft method review — Story A

Reviewed `main.tex` on 2026-10-07 against the original R03, source append, source support stack, new R05 refinement, original R03 predicate, and final readout. Read-only review; no manuscript or scientific files changed. References were not re-audited in this methods pass.

## Findings requiring small corrections

### 1. Specify the actual anchor deduplication and tie order

Location: `main.tex`, Anchor admission paragraph, around line 74; Appendix Lexical source localization, around line 227.

The wording “retain at most four candidates using the same lexical priorities” follows a per-fact order of query score, fact score, source position. The actual global order inserts anchor slot before source position. More importantly, deduplication does not select an arbitrary original anchor: among identical packets, it chooses the canonical consuming anchor by highest fact BM25, then earlier anchor slot, then earlier center position. This choice can change the fact used by later support, comparison, completion, and local ranking predicates. Preserving aliases alone does not specify that behavior.

Minimal fix, preferably in the appendix: “For identical packets, the canonical anchor is chosen by descending fact score, then anchor slot and center position. Unique candidates are ordered by descending query score, descending fact score, anchor slot, and center position before taking the first four.” The main paragraph may refer to these deterministic ties without expanding further.

Code: `jev_mem/source_append_r05.py`, `build_candidates`, lines 157–162. This is an exact reproducibility issue, not a challenge to the conceptual story.

### 2. Complete two mathematical input/definition details

Locations: replacement equation around line 90; assembly equation and explanation around lines 105–111.

- The replacement score is written as J(P_rep; q, m, w0, wa), although its actual shared query state contains both question text and reference date. QFS explicitly lists d_q, so the notation currently looks inconsistent. Add d_q to this equation, or define q globally as the complete question state and consistently remove the separate date argument everywhere.
- The assembly equation uses w_m^* without defining it. Define it as the anchor's final packet after guarded replacement, neighboring completion, and deduplication, selected by the local ranking step. No new equation is needed.

Code: `jev_mem/r05_refinement_v1.py`, `_query` and `build_candidate_variants`; `jev_mem/source_support_stack_v1.py`, `build_closure_plan`, `apply_closures`, `build_rank_request`.

One wording correction can be made at the same location: change “if at most two remain, both are retained” to “if at most two remain, all are retained,” since the algorithm also admits zero or one candidate.

## Verified without a requested change

- QFS matches original R03: pointwise explicit usefulness, top 20, pool-index tie-breaking, restored pool order, no threshold, no new-R03 complement or set comparison.
- The hybrid candidate pool and its distinction from native baseline retrieval are correctly disclosed.
- SER correctly preserves the original admitted anchor universe; it does not claim full-dialogue independent retrieval.
- The support threshold, distinct alternative requirement, whole-window query BM25 ranking before support filtering, original-versus-alternative gate, C variants and score/tie rule, and final threshold-free top two agree with code.
- The scope, session, complete-turn, five-turn, and byte constraints are consistent with implementation. The text distinguishes byte bounds, fact count, and token budget.
- Source text is appended while original fact content remains a prefix; record dates, count, identity, and order remain unchanged.
- The result values and differences are correct. The three-row chain is explicitly cumulative, and the missing independent QFS-off/SER-on condition is not represented as measured.
- Current aggregate claims remain within the readout. I found no claim of a new writer, guaranteed complementarity, certified source truth, whole-dialogue recall, matched-token gain, blind testing, cost saving, or SOTA.

After the two small method/notation corrections above, this methods pass has no substantive unresolved blocker. No extra limitations or repeated caveats are recommended.
