# ACL 2027 working paper — revised 2026-10-08

## Files

- `main.tex`: authoritative editable English manuscript, ACL anonymous-review format.
- `main.pdf`: successfully compiled eight-page paper, three tables, no figures.
- `references.bib`: nine verified primary-source references.
- `acl.sty`, `acl_natbib.bst`: unmodified official ACL files.
- `writing_plan_zh.md`: section-by-section writing transfer, two-module story, exact ablation interpretation, and proposed validation behind each empty heading.
- `discussion/`: first-pair pattern discussion, second-pair story discussion, and subsequent independent draft reviews.
- `revision_gemini25_2026-10-08.md`: focused Gemini 2.5 revision record and evidence pointers; its build-scope paragraph records the state before final compilation.
- `validate_document_v2.py`, `document_validation_v2.json`: current offline checks, both model cohorts' result provenance, compilation and visual-review evidence.
- `JEV_Mem_ACL2027_source_v2.zip`: current source package, with an explicit file allowlist and no experiment caches or raw research outputs.
- `validate_document.py`, `document_validation.json`, `JEV_Mem_ACL2027_source_v1.zip`: preserved historical validation and source package for the 2026-10-07 draft; their two-table/seven-page assumptions do not certify this revision.
- `template/provenance.json`: exact official template source and commit.

The main table contains only one populated proposed-method row. Baseline values remain blank. The six pending experiment subsections intentionally contain no body text. Separate Gemini 3.8 and Gemini 2.5 tables report cumulative component comparisons; neither table treats the Mem0 base row as an independent removal of only QFS. The Methodology section and original Gemini 3.8 result values are unchanged.

## Template

The [ACL 2027 main-conference call](https://2027.aclweb.org/calls/main/) directs submissions to the official ACL style. We use [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`, downloaded 2026-10-07. `acl.sty` SHA256: `19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de`. No dimensions, margins, or official style internals were modified. The review option adds anonymity and line numbers. This is a draft, not a submitted or accepted ACL paper.

## Compile

Upload `JEV_Mem_ACL2027_source_v2.zip` to Overleaf, set `main.tex` as the main document, and use pdfLaTeX. Or run a normal LaTeX installation:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The revised eight-page PDF was successfully compiled on 2026-10-08 with the existing local [Tectonic 0.17.0](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0) binary, which invokes TeX/BibTeX and produces a PDF. Its portable binary is under `build_tools/` in the local working folder, not in the source ZIP:

```powershell
.\build_tools\tectonic.exe --keep-logs --keep-intermediates main.tex
```

The Codex built-in compiler again returned `Unable to find standard directories for platform` during this revision. It did not compile the delivered PDF. The local Tectonic run, `main.log`, `main.blg`, and the exported PDF establish the successful build. The coordinating agent visually inspected all eight rendered pages under `render_gemini25_v2/`; no clipping, overlapping text, broken tables, or missing citations were identified. Offline log checks found no overfull boxes, undefined references, or LaTeX/BibTeX errors. The official ACL style remains unmodified. Benign fontconfig, lineno encoding, and underfull-box warnings remain recorded.

Run the versioned offline validator with the bundled Python runtime, which provides `pypdf`:

```powershell
& 'C:\Users\86139\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\validate_document_v2.py
```

This reads the preserved local result summaries, checks document structure and the existing PDF, writes `document_validation_v2.json`, and builds the v2 source ZIP. It makes no experiment or external API calls. Visual-review attribution is a recorded review by the coordinating agent, not an automated judgment by this script.

## Result provenance

The original Gemini 3.8 cohort retains these three rows:

| Manuscript configuration | Archived arm | Official F1 | Judge % |
|---|---|---:|---:|
| Full | `original_multi_source_closure` | 60.2380052357 | 84.5454545455 |
| w/o SER | `r03_original` | 52.5469107510 | 76.8181818182 |
| w/o SER, QFS | `baseline` | 51.0440001139 | 74.8701298701 |

Source summary: `D:/jev mem/research/r03_r05_refinement_2026-10-06/heartbeat_20261006T1052/live_qa_v2/final_summary_v1.json`, SHA256 `01aa05237a252832135f9c03122d16ee8d6f1aab0b0b2e97d16f2836533567e5`; accepted audit status `no_invalid_artifacts`.

Endpoint identifiers are the values in the preserved experimental configuration: `gemini-3.8-flash`, `gemini-embedding-001` (768 dimensions), and `jev-1.13.0`. The first is the shared Writer/Reader/Judge identifier. Reader/Judge use `thinkingLevel: low`, `maxOutputTokens: 8192`, with no temperature field and no JSON schema. These are recorded runtime settings, not new product-specification claims.

The added Gemini 2.5 cohort reports the following separate cumulative chain:

| Manuscript configuration | Archived arm | Official F1 | Custom F1 | Judge % | Judge yes / valid |
|---|---|---:|---:|---:|---:|
| Mem0 base | `baseline` | 53.221636 | 51.138502 | 75.584416 | 1164 / 1540 |
| + QFS | `r03_original` | 54.886885 | 52.499132 | 78.766234 | 1213 / 1540 |
| Full QFS + SER (including C) | `original_multi_source_closure` | 59.884397 | 57.408205 | 84.090909 | 1295 / 1540 |

Source summary: `D:/jev mem/research/gemini25_ablation_2026-10-07/heartbeat_20261007T1814/billing_resume_v1/live_v1/final_summary_v1.json`, SHA256 `a73ac20650ac3f135ae07404ca4042a1ccb42fe1574ca9109c5edd63fb2d15d3`. The final independent comparison is `research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.json`; its accepted status, summary hash, all three metric means, denominators, and counts are checked by the v2 document validator.

Gemini 2.5 Flash supplies Writer, Reader, and Judge with `thinkingBudget=0`. It rewrites the same 272 sessions into ten new banks containing 2,301 facts, compared with 1,637 facts in the original cohort. No old fact identifiers, selections, JEV scores, or QA answers are inherited. Each new arm has 1,540 valid Reader and Judge outcomes; all 4,620 QA slots appear in the accepted first-pass view, and the subsequent QA recovery set is empty. This does not erase bounded transport retries or separately recorded precomputation recovery.

The cumulative Judge increments are +3.181818 percentage points for QFS, +5.324675 for SER including C conditional on QFS, and +8.506494 for the full pipeline over its own base. The design has no source-only arm or independent SER/C separation. Writer, Reader, Judge, thinking setting, and memory content change across model cohorts, so cross-model score differences cannot isolate a Judge effect. These are single runs on previously examined conversations, without a claim of repeated-run stability or statistical significance. Manuscript tables round scores to three decimals.

No experiment, paid model call, cache deletion, scientific configuration change, or automation reactivation was performed for this writing task.

## Naming and remaining work

The neutral paper title avoids confusion with the distinct contemporaneous Jev-Mem paper (Jiang et al., 2026), cited in Related Work. The project name can remain JEV-Mem internally. Author names, affiliations, final title, the remaining baseline values, and the six planned validation sections still require author input or later evidence. The appendix documents real implementation and evaluation choices; it does not fill the pending analysis headings with unrun results.
