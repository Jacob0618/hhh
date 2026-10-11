"""Offline fixed-cohort QA audit for three pilot400 arms; never calls APIs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, queries
from jev_mem.evaluation import lexical_metrics
from jev_mem.evaluation_pilot import prepare_reader_input
from jev_mem.pilot400 import reader_fact
from jev_mem.selection import deduplicate_candidates
from jev_mem.topic_grouping import raw_turns_from_locomo
from tools.build_pilot400_manifest import build_manifest, ids_sha256

ARMS = ("session_untyped_rank", "session_typed_rank", "regroup_untyped_rank")
METRICS = ("official_locomo_f1", "custom_token_f1", "judge_correct", "judge_completeness")
READER_PROTOCOL = "brief_answer_source_dates_pilot_v1"
METRIC_ADAPTER_SHA256 = "c58b3302cd7b424224f447a5c532c26ad0a3ec71166b9a0d01e40da3919d797d"
OFFICIAL_SOURCE_SHA256 = "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
SCHEMA = "jev-mem-pilot400-qa-v1"
COHORT_SIZE = 400


class ArtifactInvalid(ValueError):
    """Existing output fails protocol validation; never count as complete."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path):
    return json.loads(path.read_bytes())


def _is_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _score(value, name: str) -> float:
    if not _is_number(value) or not 0 <= value <= 1:
        raise ArtifactInvalid(f"{name} must be a finite number in [0,1]")
    return float(value)


def _hash_valid(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verified_official_metric():
    """Check both pinned local adapter and retained upstream source before import."""
    adapter_path = ROOT / "jev_mem/locomo_official_metrics.py"
    source_path = ROOT / "research/sources/locomo_evaluation_cbfbc1.py"
    if _sha(adapter_path.read_bytes()) != METRIC_ADAPTER_SHA256:
        raise ValueError("Official LoCoMo metric adapter hash changed; version and review before scoring")
    if _sha(source_path.read_bytes()) != OFFICIAL_SOURCE_SHA256:
        raise ValueError("Pinned upstream LoCoMo source snapshot hash changed")
    spec = importlib.util.spec_from_file_location("_pilot400_verified_locomo_metric", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SOURCE_SHA256 != OFFICIAL_SOURCE_SHA256:
        raise ValueError("Official metric source declaration does not match verified source")
    return module.official_locomo_f1, {
        "source_commit": module.SOURCE_COMMIT,
        "source_sha256": OFFICIAL_SOURCE_SHA256,
        "adapter_sha256": METRIC_ADAPTER_SHA256,
        "source_url": module.SOURCE_URL,
        "license_status": module.LICENSE_STATUS,
    }


def _expected_inputs(manifest_path: Path):
    manifest = _json(manifest_path)
    if manifest != build_manifest():
        raise ValueError("Manifest differs from the frozen deterministic 400-question cohort")
    data = load_data()
    population = {}
    for ci, conv in enumerate(data):
        # Freeze the same explicit date as the pilot driver: the latest source
        # observation date across the full original conversation, not selected
        # facts, regrouped sessions or the wall clock.
        reference_date = max(turn["observation_date"] for turn in raw_turns_from_locomo(conv["conversation"]))
        for q in queries(conv, ci):
            population[q["id"]] = {**q, "conversation": ci, "reference_date": reference_date}
    expected = {qid: population[qid] for qid in manifest["query_ids"]}
    if len(expected) != COHORT_SIZE or ids_sha256(expected) != manifest["sample"]["query_ids_sha256"]:
        raise ValueError("Unexpected pilot cohort identity or cardinality")
    return manifest, expected


def expected_bank_path(conversation: int, arm: str) -> str:
    name = "regroup" if arm == "regroup_untyped_rank" else "session"
    return f"conversations/conv_{conversation}/banks/{name}/group_fact_bank.json"


def _bank(run_dir: Path, ci: int, arm: str, cache: dict):
    relative = expected_bank_path(ci, arm)
    if relative not in cache:
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ArtifactInvalid("Fact bank resolves outside the run directory")
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get("facts"), list):
            raise ArtifactInvalid("Fact bank requires an object with facts array")
        facts = value["facts"]
        if not isinstance(value.get("schema"), str) or not value["schema"]:
            raise ArtifactInvalid("Fact bank schema is missing")
        if any(not isinstance(f, dict) or not isinstance(f.get("id"), str) or not f["id"] for f in facts):
            raise ArtifactInvalid("Fact bank has missing or invalid fact IDs")
        ids = [f["id"] for f in facts]
        if len(set(ids)) != len(ids):
            raise ArtifactInvalid("Fact bank contains duplicate IDs")
        canonical = deduplicate_candidates(facts)
        cache[relative] = {"relative_path": relative, "sha256": _sha(raw), "ids": set(ids),
                           "schema": value["schema"], "canonical_by_id": {f["id"]: f for f in canonical}}
    return cache[relative]


def validate_artifact(row, expected: dict, arm: str, run_dir: Path, cohort_sha256: str,
                      bank_cache: dict, official_f1) -> tuple[str, dict | None]:
    if not isinstance(row, dict):
        raise ArtifactInvalid("QA artifact must be an object")
    ci, qid = expected["conversation"], expected["id"]
    required_identity = {"schema": SCHEMA, "arm": arm, "query_id": qid,
                         "conversation": ci, "category": expected["category"],
                         "cohort_sha256": cohort_sha256}
    for key, value in required_identity.items():
        if row.get(key) != value or type(row.get(key)) != type(value):
            raise ArtifactInvalid(f"Identity mismatch: {key}")
    status = row.get("status")
    if status not in ("complete", "failed"):
        raise ArtifactInvalid("status must explicitly be complete or failed")
    error = row.get("error")
    if status == "failed":
        if not isinstance(error, dict) or any(not isinstance(error.get(k), str) or not error[k] for k in ("type", "stage", "message")):
            raise ArtifactInvalid("Failed artifact requires nonempty error.type, error.stage, error.message")
        if row.get("result") is not None:
            raise ArtifactInvalid("Failed artifact cannot contain a scored result")
        # Pre-bank technical failures may not have a completed immutable bank.
        # They remain failed and receive no numeric score.
        return "failed", None
    if error is not None:
        raise ArtifactInvalid("Complete artifact carries an error")
    if row.get("bank_path") != expected_bank_path(ci, arm):
        raise ArtifactInvalid("bank_path does not match frozen arm/conversation contract")
    if not _hash_valid(row.get("bank_sha256")):
        raise ArtifactInvalid("bank_sha256 is not a lowercase SHA256")
    bank = _bank(run_dir, ci, arm, bank_cache)
    if row["bank_sha256"] != bank["sha256"]:
        raise ArtifactInvalid("bank_sha256 does not match actual bank bytes")
    selected = row.get("selected_ids")
    if (not isinstance(selected, list) or len(selected) != 20
            or any(not isinstance(x, str) or not x for x in selected)
            or len(set(selected)) != 20 or not set(selected).issubset(bank["ids"])):
        raise ArtifactInvalid("selected_ids must contain exactly 20 unique IDs present in the fact bank")
    result = row.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise ArtifactInvalid("Complete result requires an answer string")
    if type(result.get("selected_count")) is not int or result["selected_count"] != 20:
        raise ArtifactInvalid("result.selected_count must be exactly 20")
    if not set(selected).issubset(bank["canonical_by_id"]):
        raise ArtifactInvalid("selected_ids contains a noncanonical duplicate fact ID")
    prepared = prepare_reader_input(
        expected["question"], [reader_fact(bank["canonical_by_id"][fid]) for fid in selected],
        reference_date=expected["reference_date"])
    if result.get("reader_input") != prepared["payload"]:
        raise ArtifactInvalid("reader_input differs from the exact question, canonical selected facts or frozen reference date")
    if result.get("input_hash") != prepared["input_hash"]:
        raise ArtifactInvalid("input_hash differs from reconstructed reader input")
    if result.get("reader_request_hash") != prepared["reader_request_hash"]:
        raise ArtifactInvalid("reader_request_hash differs from reconstructed reader protocol/settings")
    for key, value in {"reader_protocol": READER_PROTOCOL, "reader_model": "gemini-2.5-flash", "judge_model": "gemini-3.5-flash-lite"}.items():
        if result.get(key) != value:
            raise ArtifactInvalid(f"Unexpected {key}")
    judge = result.get("judge")
    if not isinstance(judge, dict) or type(judge.get("correct")) is not bool or not isinstance(judge.get("reason"), str):
        raise ArtifactInvalid("Judge requires boolean correct and string reason")
    completeness = _score(judge.get("completeness"), "judge.completeness")
    token_f1 = _score(result.get("token_f1"), "result.token_f1")
    recomputed = lexical_metrics(result["answer"], expected["reference"])["token_f1"]
    if not math.isclose(token_f1, recomputed, abs_tol=1e-12, rel_tol=1e-12):
        raise ArtifactInvalid("Stored custom token F1 disagrees with offline recomputation")
    used = result.get("used_fact_ids")
    if (not isinstance(used, list) or any(not isinstance(x, str) for x in used)
            or len(set(used)) != len(used) or not set(used).issubset(selected)):
        raise ArtifactInvalid("Reader used_fact_ids must be unique and contained in selected_ids")
    return "complete", {
        "query_id": qid, "conversation": ci, "category": expected["category"],
        "official_locomo_f1": official_f1(result["answer"], expected["reference"], expected["category"]),
        "custom_token_f1": recomputed,
        "judge_correct": float(judge["correct"]), "judge_completeness": completeness,
    }


def _conditional(records: list[dict]) -> dict:
    return {"scored_questions": len(records), "means": {
        metric: sum(r[metric] for r in records) / len(records) if records else None for metric in METRICS}}


def _percentile(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo) if hi != lo else ordered[lo]


def paired_comparison(baseline: dict, other: dict, *, samples: int, seed: int, denominator: int = COHORT_SIZE):
    common = sorted(set(baseline).intersection(other))
    clusters = defaultdict(list)
    for qid in common:
        if baseline[qid]["conversation"] != other[qid]["conversation"]:
            raise ValueError("Paired conversation identity mismatch")
        clusters[baseline[qid]["conversation"]].append(qid)
    # A partial cluster subset is not silently bootstrapped as the full cohort.
    adequate = set(clusters) == set(range(10)) and min(map(len, clusters.values()), default=0) >= 2
    draw_indices = []
    if adequate:
        rng = random.Random(seed)
        draw_indices = [rng.choices(list(range(10)), k=10) for _ in range(samples)]
    metrics = {}
    for metric in METRICS:
        numerator = {ci: sum(other[qid][metric] - baseline[qid][metric] for qid in ids) for ci, ids in clusters.items()}
        delta = sum(numerator.values()) / len(common) if common else None
        draws = [sum(numerator[ci] for ci in selected) / sum(len(clusters[ci]) for ci in selected)
                 for selected in draw_indices]
        metrics[metric] = {"other_minus_baseline": delta,
                           "cluster_bootstrap_95_percentile_interval": [_percentile(draws, .025), _percentile(draws, .975)] if draws else None}
    return {
        "common_success_count": len(common), "cohort_denominator": denominator,
        "paired_coverage": len(common) / denominator,
        "common_query_ids": common,
        "common_success_by_conversation": {str(ci): len(clusters.get(ci, [])) for ci in range(10)},
        "bootstrap": {"unit": "conversation", "population_clusters": 10, "resampled_clusters_per_draw": 10,
                      "samples": samples if adequate else 0, "seed": seed,
                      "eligible": adequate,
                      "eligibility_rule": "All ten conversations have at least two common-success questions",
                      "reason_if_skipped": None if adequate else "Insufficient paired coverage across all ten clusters",
                      "estimator": "Micro-average paired question difference, with conversation clusters resampled with replacement"},
        "metrics": metrics,
        "interpretation": "Exploratory paired pilot including development conversations, not blind validation or a significance claim. Missingness can bias the common-success subset.",
    }


def summarize_pilot(run_dir, manifest_path=ROOT / "configs/pilot400_manifest_v1.json", *, bootstrap_samples=2000, seed=0):
    run_dir, manifest_path = Path(run_dir), Path(manifest_path)
    if not run_dir.is_dir():
        raise ValueError("Run directory does not exist; refusing to imply an experiment has started")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    manifest, expected = _expected_inputs(manifest_path)
    official_f1, metric_provenance = verified_official_metric()
    cohort_sha = manifest["sample"]["query_ids_sha256"]
    bank_cache, complete, report_arms, unexpected = {}, {}, {}, []
    expected_paths = set()
    for arm in ARMS:
        rows, statuses, failures, missing = {}, {}, [], []
        for qid, q in expected.items():
            relative = f"conversations/conv_{q['conversation']}/qa/{arm}/{qid}.json"
            expected_paths.add(relative)
            path = run_dir / relative
            if not path.exists():
                statuses[qid] = "missing"
                missing.append(qid)
                continue
            try:
                artifact = _json(path)
                status, scored = validate_artifact(artifact, q, arm, run_dir, cohort_sha, bank_cache, official_f1)
                statuses[qid] = status
                if status == "complete":
                    rows[qid] = scored
                else:
                    failures.append({"query_id": qid, "kind": "declared_failure", "error": artifact["error"]})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                statuses[qid] = "failed"
                failures.append({"query_id": qid, "kind": "protocol_invalid", "error_type": type(exc).__name__, "message": str(exc)[:1000]})
        complete[arm] = rows
        counts = {status: sum(value == status for value in statuses.values()) for status in ("complete", "failed", "missing")}
        if sum(counts.values()) != COHORT_SIZE:
            raise AssertionError("Status denominator is not 400")
        records = list(rows.values())
        by_conversation, by_category = {}, {}
        for group, output in (("conversation", by_conversation), ("category", by_category)):
            keys = range(10) if group == "conversation" else (1, 2, 3, 4)
            for value in keys:
                ids = [qid for qid, q in expected.items() if q[group] == value]
                output[str(value)] = {"cohort_count": len(ids),
                    "status_counts": {s: sum(statuses[qid] == s for qid in ids) for s in ("complete", "failed", "missing")},
                    "conditional_metrics": _conditional([rows[qid] for qid in ids if qid in rows])}
        report_arms[arm] = {
            "cohort_denominator": COHORT_SIZE, "status_counts": counts,
            "observed_complete_fraction": counts["complete"] / COHORT_SIZE,
            "all_400_complete": counts["complete"] == COHORT_SIZE,
            "conditional_metrics": _conditional(records),
            "full_cohort_observed_metrics": _conditional(records)["means"] if counts["complete"] == COHORT_SIZE else {m: None for m in METRICS},
            "zero_filled_sensitivity": {
                "is_imputation_not_observed_scoring": True,
                "denominator": COHORT_SIZE,
                "imputed_zero_questions": COHORT_SIZE - len(records),
                "means": {m: sum(row[m] for row in records) / COHORT_SIZE for m in METRICS},
                "note": "Failed/missing questions have no observed score; zeros appear only in this explicitly labelled sensitivity analysis.",
            },
            "by_conversation": by_conversation, "by_category": by_category,
            "missing_query_ids": missing, "failures": failures,
            "scored_questions": records,
        }
    for path in sorted((run_dir / "conversations").glob("conv_*/qa/*/*.json")):
        relative = path.relative_to(run_dir).as_posix()
        if relative not in expected_paths:
            unexpected.append(relative)
    comparisons = {f"{arm}_minus_session_untyped_rank": paired_comparison(
        complete[ARMS[0]], complete[arm], samples=bootstrap_samples, seed=seed) for arm in ARMS[1:]}
    return {
        "schema": "jev-mem-pilot400-summary-v1", "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()), "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": _sha(manifest_path.read_bytes()), "cohort_sha256": cohort_sha,
        "cohort_count_per_arm": COHORT_SIZE, "arms": report_arms,
        "paired_differences": comparisons, "unexpected_artifact_paths": unexpected,
        "all_expected_outputs_valid_and_complete": all(x["all_400_complete"] for x in report_arms.values()) and not unexpected,
        "official_metric_provenance": metric_provenance,
        "verified_banks": [{k: v for k, v in bank.items() if k not in {"ids", "canonical_by_id"}} for bank in bank_cache.values()],
        "reader_protocol": READER_PROTOCOL,
        "reader_integrity": {"version": "reconstruct-canonical-facts-and-source-date-v1",
                             "reference_date": "Maximum observation_date across every original raw turn of the conversation",
                             "validation": "Exact reader payload, input hash, full reader request hash, selected_count=20 and unique subset citations"},
        "interpretation": {
            "is_blind_holdout": False, "development_conversations": [0, 1], "development_selected_count": 60,
            "scope": "Fixed paired 400-question pilot including development questions; not a new blind benchmark.",
            "official_locomo_f1": "Pinned category-aware LoCoMo metric, recomputed offline from prediction and original reference.",
            "custom_token_f1": "Existing unstemmed normalized token overlap, recomputed independently; distinct from official LoCoMo F1.",
            "judge_correct": "Recorded Gemini judge output, schema-validated but not a human ground-truth adjudication.",
            "missingness": "Complete, declared/protocol failures and absent artifacts are separate; absent files are not automatically API failures.",
            "cost": "Not calculated here; use API logs to separate real network usage, cache reuse and unknown failed-request usage.",
            "source_evidence_metrics": "Not reported without validated fact-to-source support annotation.",
        },
    }


def markdown_report(report: dict) -> str:
    def display(value):
        return "N/A" if value is None else f"{value:.4f}"
    lines = ["# Pilot400 offline snapshot", "", f"Snapshot: {report['snapshot_utc']}", "",
             "Fixed 400-question paired pilot; includes 60 development questions from conversations 0/1. Not a blind test.", "",
             "| Arm | Complete / 400 | Failed | Missing | Official LoCoMo F1 (complete only) | Custom token F1 (complete only) | Judge correct (complete only) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, result in report["arms"].items():
        c, m = result["status_counts"], result["conditional_metrics"]["means"]
        lines.append(f"| {arm} | {c['complete']} / 400 | {c['failed']} | {c['missing']} | {display(m['official_locomo_f1'])} | {display(m['custom_token_f1'])} | {display(m['judge_correct'])} |")
    lines += ["", "Failed and missing questions have no observed scores. Full-cohort observed metrics remain null until all 400 succeed. The JSON separately reports explicitly zero-imputed sensitivity values.", "",
              "## Paired differences", "", "Differences use only common successful questions and may be biased by missingness. Intervals resample all ten conversation clusters only when each has at least two common successes.", ""]
    for name, pair in report["paired_differences"].items():
        lines.append(f"- {name}: common success {pair['common_success_count']} / 400 ({pair['paired_coverage']:.1%}).")
        for metric, value in pair["metrics"].items():
            interval = value["cluster_bootstrap_95_percentile_interval"]
            suffix = "not estimated" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
            lines.append(f"  - {metric}: delta {display(value['other_minus_baseline'])}; exploratory cluster 95% interval {suffix}.")
    lines += ["", f"Unexpected artifact paths: {len(report['unexpected_artifact_paths'])}.",
              f"All expected outputs valid and complete: {report['all_expected_outputs_valid_and_complete']}.",
              "", "Judge labels are model assessments, not human adjudication. This report does not assert improved retrieval evidence recall, validated novelty, significance, invoice costs or fully unseen evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/pilot400_manifest_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New offline snapshot directory; existing reports are never overwritten")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = summarize_pilot(args.run_dir, args.manifest, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.output_dir / "summary.json", args.output_dir / "summary.md"
    if json_path.exists() or md_path.exists():
        raise SystemExit("Refusing to overwrite an existing report; choose a new snapshot directory")
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with md_path.open("x", encoding="utf-8") as handle:
        handle.write(markdown_report(report))
    print(json.dumps({"report": str(json_path), "all_expected_outputs_valid_and_complete": report["all_expected_outputs_valid_and_complete"],
                      "status_counts": {arm: value["status_counts"] for arm, value in report["arms"].items()}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
