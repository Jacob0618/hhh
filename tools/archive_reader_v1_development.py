"""One-time, non-destructive archival of the pre-v2 eight-query development run."""
from pathlib import Path
from jev_mem.providers import dump_json

ROOT = Path(__file__).resolve().parents[1]
run = (ROOT / "runs/session_top20_v1").resolve()
conv = (run / "conv_0").resolve()
targets = [(conv / "qa", conv / "qa_reader_v1_dev8"),
           (conv / "qa_summary.json", conv / "qa_reader_v1_dev8_summary.json"),
           (run / "last_qa_aggregate.json", run / "reader_v1_dev8_aggregate.json"),
           (run / "last_qa_summary.json", run / "reader_v1_dev8_summary.json")]
for source, destination in targets:
    if not source.resolve().is_relative_to(run) or not destination.resolve().is_relative_to(run):
        raise RuntimeError("Archive path escaped the intended run")
    if source.exists() and destination.exists():
        raise RuntimeError("Archive already exists; refusing overwrite")
for source, destination in targets:
    if source.exists():
        source.rename(destination)
dump_json(run / "reader_v2_protocol_change.json", {
    "stage": "after_8_development_questions_before_full_evaluation",
    "reason": "UUID citation transcription errors and excessively verbose answers",
    "changes": ["local f1..f20 evidence IDs mapped back to real IDs", "shortest complete answers without explanation",
                "method failures isolated so other methods still run"],
    "old_outputs": "conv_0/qa_reader_v1_dev8 and corresponding summaries retained",
    "facts_and_retrieval_changed": False,
    "all_methods_recomputed_with_same_v2_reader": True,
    "comparison_caveat": "v1-to-v2 differences are protocol effects, not JEV gains"
})
print("Preserved the old development artifacts; current QA path ready for v2.")
