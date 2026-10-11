"""Offline validation and explicit-allowlist source packaging for the 2026-10-08 revision.

No model, experiment, network, or external API calls are made. Visual review is
attributed to the coordinating agent's completed eight-page inspection.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
tex = (ROOT / "main.tex").read_text(encoding="utf-8-sig")
bib = (ROOT / "references.bib").read_text(encoding="utf-8-sig")
clean = re.sub(r"(?<!\\)%[^\n]*", "", tex)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


historical_names = ["validate_document.py", "document_validation.json", "JEV_Mem_ACL2027_source_v1.zip"]
historical_hashes = {name: sha(ROOT / name) for name in historical_names}
pending = ["Selector Controls", "Module Decomposition", "Matched-Token Comparison",
           "Generalization", "Evidence Analysis", "Hyperparameter Analysis"]
empty = {}
for heading in pending:
    tail = clean.split("\\subsection{" + heading + "}", 1)[1]
    body = re.split(r"\\(?:subsection|section)\*?\{", tail, maxsplit=1)[0]
    empty[heading] = not re.sub(r"\\label\{[^}]*\}", "", body).strip()
assert all(empty.values()), empty
assert clean.count(r"\begin{table}") == 3
assert not re.search(r"\\begin\{figure\*?\}|\\includegraphics", clean)
cited = set()
for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}", clean):
    cited.update(x.strip() for x in group.split(","))
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
assert cited == bibkeys and len(cited) == 9, (cited, bibkeys)
tables = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", clean, re.S)
main_table = next(table for table in tables if r"\label{tab:main}" in table)
removal_table = next(table for table in tables if r"\label{tab:removal}" in table)
gemini25_table = next(table for table in tables if r"\label{tab:gemini25}" in table)
for base in ["Mem0", "MemGPT", "A-MEM", "MemGAS"]:
    assert base + r" & & \\" in main_table
assert r"\method{} (ours) & \textbf{60.238} & \textbf{84.545}" in main_table

digest_path = WORKSPACE / "research/r03_r05_refinement_2026-10-06/final_result_digest_v1.json"
digest = read_json(digest_path)
assert digest["audit_status"] == "no_invalid_artifacts"
summary_hash = sha(digest["summary"]["path"])
assert summary_hash == digest["summary"]["sha256"] == "01aa05237a252832135f9c03122d16ee8d6f1aab0b0b2e97d16f2836533567e5"
rows38 = {}
expected38 = {"original_multi_source_closure": ("60.238", "84.545"),
              "r03_original": ("52.547", "76.818"), "baseline": ("51.044", "74.870")}
labels38 = {"original_multi_source_closure": r"Full & + & +",
            "r03_original": r"w/o \ser{} & + & --", "baseline": r"w/o \ser{}, \qfs{} & -- & --"}
for arm, values in expected38.items():
    m = digest["metrics"][arm]
    rounded = (f'{100 * m["official_locomo_f1"]:.3f}', f'{100 * m["judge_correct"]:.3f}')
    assert rounded == values and m["n"] == 1540
    assert labels38[arm] + " & " + " & ".join(rounded) in removal_table
    rows38[arm] = {"official_f1_percent": rounded[0], "judge_percent": rounded[1],
                   "n": m["n"], "judge_yes": m["judge_correct_count"]}

comparison_path = WORKSPACE / "research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.json"
comparison = read_json(comparison_path)
assert comparison["status"] == "passed_final_canonical_comparison_ready_for_qualified_delivery"
assert comparison["delivery_readiness"]["ready_for_qualified_result_delivery"]
assert not comparison["delivery_readiness"]["blocking_mismatches"]
assert all(comparison["checks"].values())
summary25_path = Path(comparison["canonical_summary"]["path"])
summary25_hash = sha(summary25_path)
assert summary25_hash == comparison["canonical_summary"]["sha256"] == "a73ac20650ac3f135ae07404ca4042a1ccb42fe1574ca9109c5edd63fb2d15d3"
summary25 = read_json(summary25_path)
assert summary25["audit_status"] == "no_invalid_artifacts"
assert summary25["planned_per_arm"] == 1540 and summary25["planned_total"] == 4620
assert summary25["repetitions"] == 1
assert comparison["coverage"]["firstpass_total"] == 4620
assert comparison["coverage"]["recovery_total"] == 0
assert summary25["views"]["accepted_firstpass"] == summary25["views"]["final_with_uniform_technical_recovery"]
expected25 = {"baseline": ("53.222", "51.139", "75.584"),
              "r03_original": ("54.887", "52.499", "78.766"),
              "original_multi_source_closure": ("59.884", "57.408", "84.091")}
labels25 = {"baseline": "Mem0 base", "r03_original": r"+ \qfs{}",
            "original_multi_source_closure": r"Full (\qfs{} + \ser{})"}
metric_pairs = [("official_locomo_f1", "official_locomo_f1_mean"),
                ("custom_token_f1", "custom_token_f1_mean"), ("judge_correct", "judge_accuracy_valid")]
rows25 = {}
for arm, values in expected25.items():
    source = summary25["views"]["accepted_firstpass"]["arms"][arm]
    compared = comparison["arms"][arm]
    assert source["planned_denominator"] == 1540
    assert source["statuses"]["reader"]["complete"] == source["statuses"]["judge"]["complete"] == 1540
    assert compared["fixed_denominator"] == compared["reader_valid"] == compared["judge_valid"] == 1540
    rounded = []
    full_precision = {}
    for source_key, compared_key in metric_pairs:
        metric = source["metrics"][source_key]
        value = metric["complete_only_mean"]
        assert metric["valid_denominator"] == metric["planned_denominator"] == 1540
        assert abs(value - compared[compared_key]) < 1e-12
        rounded.append(f"{100 * value:.3f}")
        full_precision[source_key] = value
    assert tuple(rounded) == values
    assert labels25[arm] + " & " + " & ".join(rounded) in gemini25_table
    assert round(full_precision["judge_correct"] * 1540) == compared["judge_correct_count"]
    rows25[arm] = {"official_f1_percent": rounded[0], "custom_f1_percent": rounded[1],
                   "judge_percent": rounded[2], "full_precision_means": full_precision,
                   "n": 1540, "judge_yes": compared["judge_correct_count"]}
assert [rows25[a]["judge_yes"] for a in expected25] == [1164, 1213, 1295]
assert "All reported conditions share this bank" not in clean
assert r"\texttt{thinkingBudget=0}" in clean and "1,637" in clean and "2,301" in clean
assert "no source-only condition" in clean and "cannot isolate a Judge-model effect" in clean

assert sha(ROOT / "acl.sty") == "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de"
with zipfile.ZipFile(ROOT / "JEV_Mem_ACL2027_source_v1.zip") as z:
    original_tex = z.read("main.tex").decode("utf-8-sig")
    assert z.read("acl.sty") == (ROOT / "acl.sty").read_bytes()
    assert z.read("acl_natbib.bst") == (ROOT / "acl_natbib.bst").read_bytes()
    assert z.read("references.bib") == (ROOT / "references.bib").read_bytes()
method = lambda text: text.split(r"\section{Methodology}", 1)[1].split(r"\section{Experiments}", 1)[0]
assert method(original_tex) == method(tex)
reader = PdfReader(ROOT / "main.pdf")
assert len(reader.pages) == 8
page_texts = [p.extract_text() for p in reader.pages]
page_text_lengths = [len(text) for text in page_texts]
assert min(page_text_lengths) > 100
pdf_text = "\n".join(page_texts)
assert all(value in pdf_text for values in expected25.values() for value in values)
log = (ROOT / "main.log").read_text(encoding="utf-8", errors="replace")
bad_log = [line for line in log.splitlines() if re.search(r"Overfull|undefined|^!|LaTeX Error|Package .* Error", line, re.I)]
assert not bad_log, bad_log
assert "Output written on main.xdv (8 pages" in log
blg = (ROOT / "main.blg").read_text(encoding="utf-8", errors="replace")
assert not re.search(r"error|couldn't open|Warning--", blg, re.I)
render_files = [f"render_gemini25_v2/page-{i}.png" for i in range(1, 9)]
assert all((ROOT / name).is_file() for name in render_files)

body = clean.split(r"\begin{abstract}", 1)[1].split(r"\bibliography{", 1)[0]
body = re.sub(r"\\(?:cite\w*|ref|label)\{[^}]*\}", " ", body)
body = re.sub(r"\\[A-Za-z]+", " ", body)
tokens = re.findall(r"[a-z]+", body.lower())
ref = (ROOT / "reference/exemplar_extracted.txt").read_text(encoding="utf-8-sig")
ref_tokens = re.findall(r"[a-z]+", ref.lower())
ref_8 = {tuple(ref_tokens[i:i+8]) for i in range(len(ref_tokens)-7)}
overlap = sorted({" ".join(tokens[i:i+8]) for i in range(len(tokens)-7)
                  if tuple(tokens[i:i+8]) in ref_8})

# Fixed allowlist: no recursive directory inclusion, PDFs, caches, or raw runs.
files = ["main.tex", "references.bib", "acl.sty", "acl_natbib.bst", "main.bbl", "README.md",
         "writing_plan_zh.md", "revision_gemini25_2026-10-08.md", "validate_document_v2.py",
         "document_validation_v2.json", "template/provenance.json", "template/README.md",
         "discussion/consensus_story_a.md", "discussion/draft_method_review.md",
         "discussion/draft_story_review.md", "discussion/pattern_a.md", "discussion/pattern_b.md",
         "discussion/story_a.md", "discussion/story_b.md"]
checks = {
    "schema": "acl-paper-document-validation-v2",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed_with_intentional_draft_gaps",
    "artifact_hashes": {name: sha(ROOT/name) for name in ["main.tex", "main.pdf", "references.bib", "acl.sty", "acl_natbib.bst", "README.md", "validate_document_v2.py", "main.log", "main.blg"]},
    "preserved_v1_artifact_hashes": historical_hashes,
    "template": read_json(ROOT / "template/provenance.json"),
    "result_sources": {
        "gemini38": {"summary_path": digest["summary"]["path"], "sha256": summary_hash,
                     "source_audit_status": digest["audit_status"], "reported_rows": rows38},
        "gemini25": {"summary_path": str(summary25_path), "sha256": summary25_hash,
                     "canonical_comparison_path": str(comparison_path), "canonical_comparison_sha256": sha(comparison_path),
                     "canonical_comparison_status": comparison["status"], "source_audit_status": summary25["audit_status"],
                     "reported_rows": rows25, "firstpass_slots": 4620, "qa_recovery_slots": 0,
                     "comparison_mean_tolerance": 1e-12, "repetitions": 1}},
    "cumulative_design": {"gemini38": "(+,+) -> (+,-) -> (-,-)", "gemini25": "(-,-) -> (+,-) -> (+,+)",
                          "SER_includes_C": True, "source_only_arm": False, "isolated_SER_C_effect": False,
                          "judge_only_cross_model_effect": False, "repeated_run_stability_claim": False},
    "methodology_unchanged_from_v1_source": True,
    "tables": 3, "figures": 0, "blank_main_baselines": ["Mem0", "MemGPT", "A-MEM", "MemGAS"],
    "empty_experiment_sections": empty,
    "citations": {"count": len(cited), "missing": [], "unused": [], "new_citations": 0,
                  "primary_source_review": "discussion/draft_story_review.md"},
    "exemplar_overlap": {"exact_8_word_matches": overlap, "limitation": "Heuristic only: reference extraction contains joined words; manual pattern/content reviews are also saved."},
    "compile": {"engine": "Tectonic 0.17.0", "exit_code": 0, "execution_reported_by": "coordinating agent",
                "log": "main.log", "bibtex_log": "main.blg", "pages": len(reader.pages),
                "page_text_lengths": page_text_lengths, "gemini25_pdf_values_present": True,
                "overfull_or_undefined_or_error_lines": bad_log, "bibtex_errors": 0,
                "benign_warnings": ["fontconfig default configuration", "lineno comment encoding", "underfull boxes"],
                "built_in_compiler": "unavailable: Unable to find standard directories for platform",
                "new_latex_installation": False},
    "visual_review": {"render_directory": "render_gemini25_v2", "reviewed_pages": list(range(1, 9)),
                      "reviewer": "coordinating agent", "review_date": "2026-10-08",
                      "render_hashes": {name: sha(ROOT/name) for name in render_files},
                      "result": "No clipping, overlapping text, broken tables, missing citations, or visible glyph errors identified.",
                      "limitation": "Records the coordinating agent's completed visual inspection; the validator checks render existence and hashes, not appearance."},
    "scope": {"new_experiments": False, "external_API_calls": False, "scientific_source_changes": False,
              "automation_reactivated": False, "v1_artifacts_modified": False},
    "source_package": {"archive": "JEV_Mem_ACL2027_source_v2.zip", "explicit_allowlist": files},
    "remaining_author_work": ["author information and title approval", "matched-protocol baseline results", "six titled validation experiments"],
}
(ROOT / "document_validation_v2.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
archive = ROOT / "JEV_Mem_ACL2027_source_v2.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(ROOT/name, arcname=name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert z.namelist() == files
    assert z.read("main.tex") == (ROOT / "main.tex").read_bytes()
    assert z.read("document_validation_v2.json") == (ROOT / "document_validation_v2.json").read_bytes()
assert historical_hashes == {name: sha(ROOT / name) for name in historical_names}
print(json.dumps({"status": checks["status"], "pages": len(reader.pages), "tables": 3, "figures": 0,
                  "citations": len(cited), "empty_sections": len(empty), "overlap_matches": overlap,
                  "both_model_cohorts_verified": True, "v1_artifacts_unchanged": True,
                  "zip_bytes": archive.stat().st_size, "zip_entries": len(files),
                  "zip_sha256": sha(archive)}, ensure_ascii=False))
