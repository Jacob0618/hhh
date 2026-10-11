"""Offline checks and explicit-allowlist source packaging; no experiment calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
tex = (ROOT / "main.tex").read_text(encoding="utf-8-sig")
bib = (ROOT / "references.bib").read_text(encoding="utf-8-sig")
clean = re.sub(r"(?<!\\)%[^\n]*", "", tex)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

pending = ["Selector Controls", "Module Decomposition", "Matched-Token Comparison",
           "Generalization", "Evidence Analysis", "Hyperparameter Analysis"]
empty = {}
for heading in pending:
    tail = clean.split("\\subsection{" + heading + "}", 1)[1]
    body = re.split(r"\\(?:subsection|section)\*?\{", tail, maxsplit=1)[0]
    body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
    empty[heading] = not body
assert all(empty.values()), empty
assert clean.count(r"\begin{table}") == 2
assert not re.search(r"\\begin\{figure\*?\}", clean)
assert not re.search(r"\\includegraphics", clean)
cited = set()
for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}", clean):
    cited.update(x.strip() for x in group.split(","))
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
assert cited == bibkeys and len(cited) == 9, (cited, bibkeys)
for base in ["Mem0", "MemGPT", "A-MEM", "MemGAS"]:
    assert base + r" & & \\" in clean

digest_path = Path(r"D:\jev mem\research\r03_r05_refinement_2026-10-06\final_result_digest_v1.json")
digest = json.loads(digest_path.read_text(encoding="utf-8-sig"))
assert digest["audit_status"] == "no_invalid_artifacts"
summary_hash = sha(digest["summary"]["path"])
assert summary_hash == digest["summary"]["sha256"]
rows = {}
expected = {"original_multi_source_closure": ("60.238", "84.545"),
            "r03_original": ("52.547", "76.818"), "baseline": ("51.044", "74.870")}
for arm, values in expected.items():
    m = digest["metrics"][arm]
    rounded = (f'{100 * m["official_locomo_f1"]:.3f}', f'{100 * m["judge_correct"]:.3f}')
    assert rounded == values and m["n"] == 1540
    assert all(v in clean for v in values)
    rows[arm] = {"official_f1_percent": rounded[0], "judge_percent": rounded[1],
                 "n": m["n"], "judge_yes": m["judge_correct_count"]}
assert r"w/o \ser{}, \qfs{}" in clean

style_hash = sha(ROOT / "acl.sty")
assert style_hash == "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de"
reader = PdfReader(ROOT / "main.pdf")
assert len(reader.pages) == 7
page_text_lengths = [len(p.extract_text()) for p in reader.pages]
assert min(page_text_lengths) > 100
log = (ROOT / "main.log").read_text(encoding="utf-8", errors="replace")
bad_log = [line for line in log.splitlines() if re.search(
    r"Overfull|undefined|^!|LaTeX Error|Package .* Error", line, re.I)]
assert not bad_log, bad_log
blg = (ROOT / "main.blg").read_text(encoding="utf-8", errors="replace")
assert not re.search(r"error|couldn't open|Warning--", blg, re.I)

# Auxiliary overlap check: reference PDF extraction contains joined words,
# so a zero match does not replace the reviewers' reading of both papers.
body = clean.split(r"\begin{abstract}", 1)[1].split(r"\bibliography{", 1)[0]
body = re.sub(r"\\(?:cite\w*|ref|label)\{[^}]*\}", " ", body)
body = re.sub(r"\\[A-Za-z]+", " ", body)
tokens = re.findall(r"[a-z]+", body.lower())
ref = (ROOT / "reference/exemplar_extracted.txt").read_text(encoding="utf-8-sig")
ref_tokens = re.findall(r"[a-z]+", ref.lower())
ref_8 = {tuple(ref_tokens[i:i+8]) for i in range(len(ref_tokens)-7)}
overlap = sorted({" ".join(tokens[i:i+8]) for i in range(len(tokens)-7)
                  if tuple(tokens[i:i+8]) in ref_8})

checks = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed_with_intentional_draft_gaps",
    "artifact_hashes": {n: sha(ROOT/n) for n in ["main.tex", "main.pdf", "references.bib", "acl.sty", "acl_natbib.bst"]},
    "template": json.loads((ROOT/"template/provenance.json").read_text(encoding="utf-8-sig")),
    "result_source": {"summary_path": digest["summary"]["path"], "sha256": summary_hash,
                      "source_audit_status": digest["audit_status"], "reported_rows": rows},
    "progressive_removal": "(+,+) -> (+,-) -> (-,-); final row removes both QFS and SER",
    "tables": 2, "figures": 0, "blank_main_baselines": ["Mem0", "MemGPT", "A-MEM", "MemGAS"],
    "empty_experiment_sections": empty,
    "citations": {"count": len(cited), "missing": [], "unused": [],
                  "primary_source_review": "discussion/draft_story_review.md"},
    "exemplar_overlap": {"exact_8_word_matches": overlap, "limitation": "Heuristic only: reference extraction contains joined words; manual pattern/content reviews are also saved."},
    "compile": {"engine": "Tectonic 0.17.0", "exit_code": 0, "log": "compile_final.txt",
                "pages": len(reader.pages), "page_text_lengths": page_text_lengths,
                "overfull_or_undefined_or_error_lines": bad_log, "bibtex_errors": 0,
                "benign_warnings": ["fontconfig default configuration", "lineno comment encoding", "underfull boxes"],
                "built_in_compiler": "unavailable: Unable to find standard directories for platform"},
    "visual_review": {"render_directory": "render_final", "reviewed_pages": [1,2,3,4,5,6,7],
                      "result": "No clipping, overlapping text, broken tables, missing citations, or visible glyph errors identified."},
    "agent_review_fixes": ["anchor canonical dedup and tie ordering", "reference date in replacement score", "definition of final source packet", "all rather than both for at-most-two", "valid evaluated outcome wording", "hybrid pool appendix cross-reference", "repetitive sentences removed"],
    "scope": {"new_experiments": False, "scientific_source_changes": False, "automation_reactivated": False},
    "remaining_author_work": ["author information and title approval", "matched-protocol baseline results", "six titled validation experiments"],
}
(ROOT/"document_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

files = ["main.tex", "references.bib", "acl.sty", "acl_natbib.bst", "main.bbl", "README.md",
         "writing_plan_zh.md", "document_validation.json", "template/provenance.json", "template/README.md"]
files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"discussion").glob("*.md"))]
archive = ROOT / "JEV_Mem_ACL2027_source_v1.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(ROOT/name, arcname=name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({"status": checks["status"], "pages": len(reader.pages), "citations": len(cited),
                  "empty_sections": len(empty), "overlap_matches": overlap,
                  "zip_bytes": archive.stat().st_size, "zip_entries": len(files)}, ensure_ascii=False))
