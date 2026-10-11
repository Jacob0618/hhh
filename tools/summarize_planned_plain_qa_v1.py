"""Thin offline audit/report adapter for the sealed H0/H1/H2 full1540 arms.

Reuse original raw stage replay, independent Reader/Judge accounting, recovery
priority, usage estimation and paired estimator. No provider/native clients.
"""
from pathlib import Path
import argparse
import json
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_sdk20_plain_qa_v1 as _old
from tools import run_planned_plain_qa_v1 as runner
from tools.run_planned_retrieval_v1 import ARMS
from tools.freeze_sdk20_full1540_v1 import verify_lock

for _name, _value in vars(_old).items():
    if not _name.startswith("__"):
        globals().setdefault(_name, _value)
# Imported modules remain unmodified; all stage functions resolve our runner.
from tools import run_planned_plain_qa_v1 as runner
from tools.run_planned_retrieval_v1 import ARMS
sources = SimpleNamespace(validate=runner.source_validate, historical=_old.sources.historical)
for _name, _function in vars(_old).items():
    if isinstance(_function, FunctionType) and _function.__module__ == _old.__name__:
        _bound = FunctionType(_function.__code__, globals(), _function.__name__,
                              _function.__defaults__, _function.__closure__)
        _bound.__kwdefaults__ = _function.__kwdefaults__
        globals()[_name] = _bound
_original_summarize = summarize
METRICS = _old.METRICS
HISTORICAL_METHOD = "baseline_sdk20"
BASELINE_VIEW = "final_with_uniform_technical_recovery"


def valid_scores(rows, arm, name):
    return {r["query_id"]: {"query_id": r["query_id"], "conversation": r["conversation"],
                             name: r["scores"][name]}
            for r in rows if r["method"] == arm and r["scores"][name] is not None}


def paired(rows, left_arm, right_arm, *, right_rows=None):
    # Original estimator reports right minus left on common-valid IDs.
    return {name: _old.existing.paired_metric(name, valid_scores(rows, left_arm, name),
                valid_scores(rows if right_rows is None else right_rows, right_arm, name),
                samples=2000, denominator=1540) for name in METRICS}


def aggregate(rows):
    original = _old.existing.aggregate
    scoped = FunctionType(original.__code__, {**original.__globals__, "ARMS": ARMS},
                          original.__name__, original.__defaults__, original.__closure__)
    scoped.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    output = scoped(rows, denominator=1540, bootstrap_samples=2000)
    output["paired_comparisons"] = {
        "H1_minus_H0": {"left": ARMS[0], "right": ARMS[1],
                        "metrics": output.pop("paired_merged_minus_unmerged")},
        "H2_minus_H1": {"left": ARMS[1], "right": ARMS[2],
                        "metrics": paired(rows, ARMS[1], ARMS[2])}}
    return output


def summarize(firstpass, recovery=None):
    lock = verify_lock()
    reference = lock["baseline"]["summary"]
    baseline = runner.bound(reference)
    historical = [r for r in baseline["slots_by_view"][BASELINE_VIEW]
                  if r["method"] == HISTORICAL_METHOD]
    if (len(historical) != 1540 or len({r["query_id"] for r in historical}) != 1540 or
        baseline.get("audit_status") != "no_invalid_artifacts" or baseline.get("audit_errors")):
        raise ValueError("Historical canonical N0 summary must remain complete and audited")
    output = _original_summarize(firstpass, recovery)
    population = set(ARMS)
    for view, rows in output["slots_by_view"].items():
        if set(r["method"] for r in rows) != population:
            raise ValueError("Hybrid report must have exactly H0/H1/H2")
        for arm in ARMS:
            selected = [r for r in rows if r["method"] == arm]
            if {r["query_id"] for r in selected} != {r["query_id"] for r in historical}:
                raise ValueError("New arm question IDs differ from canonical N0")
        output["views"][view]["historical_N0_comparisons"] = {
            f"{label}_minus_N0": {"left": HISTORICAL_METHOD, "right": arm,
                "metrics": paired(historical, HISTORICAL_METHOD, arm, right_rows=rows),
                "interpretation": "historical single-run N0; new generation times differ, not an independent repeat"}
            for label, arm in zip(("H0", "H1", "H2"), ARMS)}
    output.update(schema="planned-hybrid-plain-qa-summary-v1", planned_total=4620,
                  declared_arms=list(ARMS), historical_N0={"summary": reference,
                    "view": BASELINE_VIEW, "official_f1_percent": lock["baseline"]["official_f1_percent"],
                    "judge_percent": lock["baseline"]["judge_percent"], "recomputed_or_modified": False})
    output["summary_code_sha256"]["tools/summarize_planned_plain_qa_v1.py"] = runner.sha(Path(__file__))
    output["limitations"] = [line for line in output["limitations"] if "paired_merged_minus_unmerged" not in line]
    output["limitations"].extend([
        "H0 minus N0 is retrieval plus fusion total effect; H1 minus H0 is fixed-pool original-label selection; H2 minus H1 masks only relation weight.",
        "The frozen historical N0 result is reused, never replaced. All10 conversations are exposed development material.",
        "Each new arm has one full1540 pass and at most the unchanged technical recovery; no three-repeat diagnostic is added."])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firstpass", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh report output required")
    with sources.historical.offline():
        output = summarize(args.firstpass, args.recovery)
    runner.save_new(args.output, output)
    print(json.dumps({"output": str(args.output), "audit_status": output["audit_status"],
                      "planned_total": output["planned_total"], "model_calls": 0}))


if __name__ == "__main__":
    main()
