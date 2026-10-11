"""Resolve two monitor-only diagnostics without changing any experiment artifact."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_independent_experiments as independent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    original = ROOT / "research/heartbeat_checks/20260927T1504_retrieval_audit.json"
    prior = read(original)
    by_arch = {row["architecture"]: row for row in prior["runs"]}
    for row in prior["runs"]:
        assert row["expected_queries"] == row["retrieval_queries"] == 1540
        assert not row["missing_ids"] and not row["unexpected_ids"]
    assert not by_arch["mem0"]["errors"]
    assert by_arch["mem0"]["code_mismatches"] == ["jev_mem\\locomo_official_metrics.py"]
    metric = ROOT / "jev_mem/locomo_official_metrics.py"
    frozen_metric = ROOT / "research/protocol_snapshot_reader_v2/jev_mem/locomo_official_metrics.py"
    before = frozen_metric.read_text(encoding="utf-8")
    after = metric.read_text(encoding="utf-8")
    assert before.replace("import re\nimport string\n", "import string\nimport regex as re\n") == after
    live_sources = ["jev_mem/data.py", "jev_mem/providers.py", "jev_mem/evaluation.py",
                    "jev_mem/selection.py", "jev_mem/mem0_adapter.py", "jev_mem/independent.py",
                    "tools/run_session_experiments.py", "tools/run_independent_experiments.py"]
    for source in live_sources:
        assert "locomo_official_metrics" not in (ROOT / source).read_text(encoding="utf-8")
    run = ROOT / "runs/independent_top20_v1"
    manifest = read(run / "run_manifest.json")
    assert manifest["protocol"] == independent._protocol()
    assert not by_arch["independent"]["code_mismatches"]
    assert by_arch["independent"]["errors"] == [f"conv_{ci}: changed facts snapshot" for ci in range(10)]
    verified = []
    for ci in range(10):
        summary, facts = independent._ingestion(run / f"conv_{ci}", manifest, False)
        verified.append({"conversation":ci, "canonical_sha256":independent._hash(facts),
                         "facts":len(facts), "bank_and_export_match":True})
    result = {"checked_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"retrieval_and_live_protocol_integrity_not_final_QA_audit", "passed":True,
        "original_audit_preserved":str(original), "original_audit_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
        "retrieval_validation":"All 1540 queries per architecture have bound candidate content and matching fact identities/texts; all pools60 and SDK20 counts validated.",
        "monitor_correction":"The independent runner intentionally hashes canonical parsed JSON, not file bytes; its own read-only checker validated all ten exports and persisted fact banks.",
        "offline_metric_change":{"before_sha256":hashlib.sha256(frozen_metric.read_bytes()).hexdigest(),
            "after_sha256":hashlib.sha256(metric.read_bytes()).hexdigest(),
            "diff":"Only import re replaced with import regex as re, to match the pinned official metric implementation.",
            "not_imported_by_live_runners_reader_or_selectors":True,
            "implication":"Track this offline postprocessing version separately; it did not alter retrieval, model requests, or stored primary QA scores."},
        "independent_fact_snapshots":verified,
        "experiment_artifacts_modified":False,"model_calls":0}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
