"""Read-only paired sensitivity under the frozen recorded-input reporting policy.

No provider import, credentials, network, reordered model call or data mutation.
All effects are conditional on original common-success questions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
ARMS = ("session_untyped_rank", "session_typed_rank", "regroup_untyped_rank")
METRICS = ("official_locomo_f1", "custom_token_f1", "judge_correct", "judge_completeness")
POLICY_SHA = "390021383045338c6131389794debe80a3f26ada9f3ad374662113a97baa19eb"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] * (hi - position) + values[hi] * (position - lo)


def paired(baseline, other, denominator, samples=2000, seed=0):
    ids = sorted(set(baseline) & set(other))
    clusters = {}
    for qid in ids:
        ci = baseline[qid]["conversation"]
        if ci != other[qid]["conversation"]:
            raise ValueError("Conversation mismatch")
        clusters.setdefault(ci, []).append(qid)
    adequate = set(clusters) == set(range(10)) and min(map(len, clusters.values()), default=0) >= 2
    draws = []
    if adequate:
        rng = random.Random(seed)
        draws = [rng.choices(list(range(10)), k=10) for _ in range(samples)]
    scores = {}
    for metric in METRICS:
        numerators = {ci: sum(other[qid][metric] - baseline[qid][metric] for qid in qids)
                      for ci, qids in clusters.items()}
        estimate = sum(numerators.values()) / len(ids) if ids else None
        distribution = [sum(numerators[ci] for ci in chosen) / sum(len(clusters[ci]) for ci in chosen) for chosen in draws]
        scores[metric] = {"other_minus_baseline": estimate,
                         "cluster_bootstrap_95_percentile_interval": [percentile(distribution, .025), percentile(distribution, .975)] if draws else None}
    return {"common_success_count": len(ids), "cohort_denominator": denominator,
            "paired_coverage": len(ids) / denominator, "common_query_ids": ids,
            "common_success_by_conversation": {str(ci): len(clusters.get(ci, [])) for ci in range(10)},
            "bootstrap": {"unit": "conversation", "population_clusters": 10, "resampled_clusters_per_draw": 10,
                          "samples": samples if adequate else 0, "seed": seed, "eligible": adequate,
                          "eligibility_rule": "All ten conversations have at least two common-success questions",
                          "reason_if_skipped": None if adequate else "Insufficient paired coverage across all ten clusters",
                          "estimator": "Micro-average paired question difference, with conversation clusters resampled with replacement"},
            "metrics": scores}


def assert_nested_equal(actual, expected, label):
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError(label + " field mismatch")
        for key in actual:
            assert_nested_equal(actual[key], expected[key], label + "." + key)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(label + " length mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_nested_equal(a, b, label + f"[{i}]")
    elif type(actual) is float:
        if not isinstance(expected, (float, int)) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(label + " numerical mismatch")
    elif actual != expected:
        raise ValueError(label + " value mismatch")


def scored_maps(summary, qids):
    result = {}
    for arm in ARMS:
        records = summary["arms"][arm]["scored_questions"]
        if any(not isinstance(r, dict) for r in records):
            raise ValueError("Invalid scored record")
        mapping = {r["query_id"]: r for r in records}
        if len(mapping) != len(records) or not set(mapping) <= qids:
            raise ValueError("Scored IDs duplicate or outside cohort")
        if len(mapping) != summary["arms"][arm]["status_counts"]["complete"]:
            raise ValueError("Completion count mismatch")
        for qid, row in mapping.items():
            if row["conversation"] != int(qid.split("_")[0][1:]) or row["conversation"] not in range(10):
                raise ValueError("Question/conversation identity mismatch")
            for metric in METRICS:
                score = row[metric]
                if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid normalized metric score")
            if row["judge_correct"] not in (0, 1):
                raise ValueError("Judge correctness must be binary")
        result[arm] = mapping
    return result


def population(maps, exclusions, denominator):
    excluded = set(exclusions)
    filtered = {arm: {qid: row for qid, row in maps[arm].items() if qid not in excluded} for arm in ARMS}
    summaries = {}
    for arm, rows in filtered.items():
        removed = sorted(set(maps[arm]) & excluded)
        summaries[arm] = {"scored_successes_retained": len(rows), "scored_successes_removed": len(removed),
                          "removed_success_query_ids": removed,
                          "retained_success_by_conversation": dict(sorted(Counter(str(row["conversation"]) for row in rows.values()).items())),
                          "conditional_metric_means": {m: sum(row[m] for row in rows.values()) / len(rows) if rows else None for m in METRICS}}
    comparisons = {arm + "_minus_" + ARMS[0]: paired(filtered[ARMS[0]], filtered[arm], denominator) for arm in ARMS[1:]}
    return {"excluded_query_ids_all_arms": sorted(excluded), "sensitivity_population_count_per_arm": denominator,
            "arms": summaries, "paired_differences": comparisons}


def analyze(summary_path, policy_path):
    summary_raw, policy_raw = shared_read(summary_path), shared_read(policy_path)
    if digest(policy_raw) != POLICY_SHA:
        raise ValueError("Frozen reporting policy SHA mismatch")
    policy, summary = json.loads(policy_raw), json.loads(summary_raw)
    cohort_path = ROOT / "configs/pilot400_manifest_v1.json"
    cohort_raw = shared_read(cohort_path)
    cohort = json.loads(cohort_raw)
    qids = set(cohort["query_ids"])
    if len(qids) != 400:
        raise ValueError("Expected fixed400 cohort")
    if summary["cohort_count_per_arm"] != 400 or digest(cohort_raw) != summary["manifest_file_sha256"]:
        raise ValueError("Primary summary cohort identity differs")
    bindings = {}
    for key, item in policy["artifacts"].items():
        raw = shared_read(ROOT / item["path"])
        if digest(raw) != item["sha256"]:
            raise ValueError("Policy evidence changed: " + key)
        bindings[key] = item
    maps = scored_maps(summary, qids)
    primary = population(maps, [], 400)
    for name, computed in primary["paired_differences"].items():
        original = summary["paired_differences"][name]
        assert_nested_equal(computed, {key: original[key] for key in computed}, "primary." + name)
    for arm in ARMS:
        means = primary["arms"][arm]["conditional_metric_means"]
        assert_nested_equal(means, summary["arms"][arm]["conditional_metrics"]["means"], "primary." + arm + ".means")
    supplements = {}
    for name in ("primary_reader_order", "secondary_rank_level"):
        rule = policy["sensitivity"][name]
        exclusions = rule["excluded_query_ids"]
        if len(exclusions) != len(set(exclusions)) or not set(exclusions) <= qids:
            raise ValueError("Invalid policy exclusion")
        expected = 400 - len(exclusions)
        if rule["eligibility_count"] != expected or rule["all_arms_same_query_exclusion"] is not True:
            raise ValueError("Policy population mismatch")
        supplements[name] = population(maps, exclusions, expected)
    if digest(shared_read(summary_path)) != digest(summary_raw) or digest(shared_read(policy_path)) != POLICY_SHA:
        raise ValueError("Inputs changed during analysis")
    return {"schema": "jev-mem-pilot400-locked-paired-sensitivity-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "release_status": "pending_actual_recorded_input_gate",
            "strict_deterministic_retrieval_replay": "FAILED_preserved",
            "input_summary": {"path": str(Path(summary_path).resolve()), "sha256": digest(summary_raw),
                              "schema": summary["schema"], "cohort_sha256": summary["cohort_sha256"]},
            "policy": {"path": str(Path(policy_path).resolve()), "sha256": POLICY_SHA,
                       "decision_timing": policy["decision_timing"], "evidence_bindings": bindings},
            "calculation_source_sha256": digest(shared_read(__file__)),
            "original_summary_metric_code_sha256": digest(shared_read(ROOT / "tools/summarize_pilot400.py")),
            "primary_recalculation_agrees_with_original": True,
            "original_primary_status_counts": {arm: summary["arms"][arm]["status_counts"] for arm in ARMS},
            "primary_population_unchanged": primary, "sensitivity": supplements,
            "confidence_interval_protocol": {"seed": 0, "requested_draws": 2000, "unit": "conversation",
                "required_clusters": 10, "minimum_common_successes_each_cluster": 2,
                "estimator": "micro-average paired question difference; resample 10 conversation clusters with replacement",
                "ineligible_interval": None},
            "limits": ["Main 400-question results and original statuses are unchanged.",
                "398/397 are supplemental eligibility populations, not numbers of common successes.",
                "Reported differences use original common-success records, not zero-imputed failures or reordered-reader predictions.",
                "Official LoCoMo F1, custom token F1, Judge correctness and Judge completeness are kept separate.",
                "Sensitivity sets were fixed after diagnostics and before effect inspection, not before the original experiment.",
                "Only observed seed1 differences were diagnosed; no claim of all-seed stability.",
                "These numerical calculations do not pass the pending recorded-input/raw-response gate or the failed strict reconstruction gate.",
                "C missingness in two whole conversation banks prevents the original all-ten-cluster CI; no 8-cluster substitute is used."],
            "model_calls": 0, "source_run_modified": False}


def markdown(report):
    lines = ["# Policy-locked paired sensitivity", "",
             "**Pending recorded-input/raw-response audit. Strict deterministic retrieval replay remains FAILED.**",
             "", "The main 400 questions per arm and every original status remain unchanged. The two supplemental populations remove the policy-fixed IDs symmetrically from all arms. Scores are original observed common-success records; missing outcomes and excluded questions are not reclassified or assigned fabricated scores.",
             "", "| Population | Comparison | Eligible questions | Common successes | All-ten-cluster CI eligible |",
             "| --- | --- | ---: | ---: | --- |"]
    populations = [("primary400", report["primary_population_unchanged"])] + list(report["sensitivity"].items())
    for label, pop in populations:
        for name, pair in pop["paired_differences"].items():
            lines.append(f"| {label} | {name} | {pair['cohort_denominator']} | {pair['common_success_count']} | {pair['bootstrap']['eligible']} |")
    lines += ["", "Differences and intervals below are percentage points (metric scale ×100). Official F1, custom token F1 and Judge metrics are distinct.",
              "", "| Population | Comparison | Metric | Difference (pp) | 95% conversation-cluster interval (pp) |",
              "| --- | --- | --- | ---: | --- |"]
    for label, pop in populations:
        for name, pair in pop["paired_differences"].items():
            for metric, values in pair["metrics"].items():
                delta, interval = values["other_minus_baseline"], values["cluster_bootstrap_95_percentile_interval"]
                text = "Not eligible under original 10-cluster rule" if interval is None else f"[{100*interval[0]:.4f}, {100*interval[1]:.4f}]"
                lines.append(f"| {label} | {name} | {metric} | {100*delta:.4f} | {text} |")
    lines += ["", "Primary reader-order supplement excludes c8_q107 and c9_q71: eligible N=398. Conservative rank-level supplement also excludes c2_q70: eligible N=397. Bootstrap protocol: seed 0, 2,000 draws, ten conversation clusters sampled with replacement; each original cluster must retain at least two common successes. The C comparison has no paired successes for conversations 1 and 7, so its interval remains absent in every population.",
              "", "The primary comparison and conditional means were independently recomputed from the bound summary's scored_questions and matched the original summary within 1e-12. This is a numerical validation, not an integrity gate or an efficacy conclusion.",
              "", "Source summary SHA256: " + report["input_summary"]["sha256"] + ".",
              "Policy SHA256: " + POLICY_SHA + ".",
              "", "The exclusions were decided after diagnostic discovery and before performance-difference inspection. They do not estimate the effect of alternate evidence order, and do not bound behavior under untested hash seeds."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "research/heartbeat_checks/20260928T0014"
    p.add_argument("--summary", type=Path, default=base / "pilot_metrics_pending_integrity/summary.json")
    p.add_argument("--policy", type=Path, default=base / "recorded_input_reporting_policy_v1.json")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.summary, args.policy)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for suffix, text in (("json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"), ("md", markdown(result))):
        (args.output_dir / ("sensitivity_v1." + suffix)).write_text(text, encoding="utf-8")
    print(json.dumps({"release_status": result["release_status"], "primary_recalculation_agrees": True,
                     "common_success_counts": {k: {n: p["common_success_count"] for n, p in v["paired_differences"].items()} for k, v in result["sensitivity"].items()}}))


if __name__ == "__main__":
    main()

