"""Pure offline proposal check; stop before Judge and produce no QA scores."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.providers import ModelAPI
from jev_mem.evaluation import answer_and_judge
from jev_mem.answer_format_compat import normalize_reader_prediction, PROTOCOL


class ValidatedReader(Exception):
    """Reached the Judge boundary after the original Reader and citation checks."""


class OfflineReader:
    def __init__(self, signature, response, *, compatible):
        self.signature, self.response, self.compatible = signature, response, compatible
        self.audit = None
        self.requests = 0

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified"):
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        if stage != "qa_reader" or signature != self.signature:
            raise AssertionError("Offline check encountered an unexpected request")
        self.requests += 1
        return deepcopy(self.response)

    def generate(self, *args, **kwargs):
        if kwargs.get("stage") == "strict_judge":
            raise ValidatedReader()
        prediction = ModelAPI.generate(self, *args, **kwargs)
        if self.compatible:
            prediction, self.audit = normalize_reader_prediction(prediction)
        return prediction


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def check(job_run):
    from tools import recover_sdk20_modules as recovery
    run = Path(job_run).resolve()
    manifest = recovery.manifest(run, verify_sources=False)
    # Job and old raw pointers are checked below; no source retrieval scans.
    records = []
    for item in manifest["jobs"]:
        job = recovery.verify_pointer(item)
        recovery.verify(item["path"])
        raw = recovery.verify_pointer(job["old_raw"])
        if raw["request"] != job["expected_reader_signature"]:
            raise ValueError("Pinned primary-table Reader raw signature differs")
        states = {}
        compatible_audit = None
        for mode in (False, True):
            api = OfflineReader(raw["request"], raw["response"], compatible=mode)
            try:
                answer_and_judge(api, job["question"], job["reference"], deepcopy(job["selected_facts"]),
                                 reference_date=job["reference_date"])
            except ValidatedReader:
                states["compatible" if mode else "legacy"] = "reader_and_citations_valid; stopped_before_judge"
            except Exception as exc:
                states["compatible" if mode else "legacy"] = {"type": type(exc).__name__, "message": str(exc)}
            else:
                raise AssertionError("Offline check unexpectedly passed the Judge boundary")
            if api.requests != 1:
                raise AssertionError("Each offline check must decode exactly one pinned Reader response")
            if mode:
                compatible_audit = api.audit
        records.append({"conversation": job["conversation"], "method": job["method"], "query_id": job["query_id"],
                        "job": item, "primary_qa": job["source_qa"], "reader_raw": job["old_raw"],
                        "validation": states, "compatibility_audit": compatible_audit})
    counts = Counter(row["method"] for row in records)
    if dict(counts) != recovery.EXPECTED_FAILURES:
        raise AssertionError("Offline proposal did not cover all55 fixed primary-table failures")
    passed = [r for r in records if r["validation"]["compatible"] == "reader_and_citations_valid; stopped_before_judge"]
    unchanged_old_failure = sum(r["validation"]["legacy"] == {"type": "ValueError", "message": "Reader did not return an answer string"} for r in records)
    return {"schema": "answer-format-compat-offline-proposal-v1", "compatibility_protocol": PROTOCOL,
            "status": "proposal_only_not_applied_to_any_run_or_scores", "model_calls": 0, "judge_requests": 0,
            "qa_scores_generated": 0, "original_rows_modified": 0,
            "fixed_source": "Pinned Reader responses underlying the original sdk20_modules_v1 primary table, not any of the new three recovery attempts.",
            "baseline_source_note": "The inherited baseline in that primary table already includes the previously reviewed technical-recovery sources. This check does not claim all source responses were the first HTTP attempt in project history.",
            "source_manifest": pointer(run / "run_manifest.json"),
            "compatibility_module": pointer(ROOT / "jev_mem/answer_format_compat.py"),
            "targets": len(records), "legacy_reader_failures_reproduced": unchanged_old_failure,
            "compatible_reader_and_citations_valid": len(passed),
            "by_arm": {arm: {"targets": n, "compatible_valid": sum(r["method"] == arm for r in passed)} for arm, n in counts.items()},
            "interpretation": "Passing means only that original Reader/citation validators accept the mechanically normalized output; answer correctness and task-score improvement have not been evaluated.",
            "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-run", type=Path, default=ROOT / "runs/sdk20_modules_recovery_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Proposal check output must be new")
    from unittest.mock import patch
    with patch("jev_mem.providers.requests.Session", side_effect=AssertionError("Model requests forbidden in offline proposal")):
        report = check(args.job_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "targets", "legacy_reader_failures_reproduced", "compatible_reader_and_citations_valid", "by_arm", "model_calls", "judge_requests", "qa_scores_generated")}))


if __name__ == "__main__":
    main()
