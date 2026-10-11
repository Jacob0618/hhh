"""Offline, snapshot-safe summaries of QA artifacts and recorded API work.

No provider clients are constructed and no network/model calls are made. Missing
embedding billing usage stays unknown. Estimates use explicit configured prices,
not invoices, and actual usage is separate from uncached logical work.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, queries, session_envelopes
from jev_mem.locomo_official_metrics import official_locomo_f1, SOURCE_COMMIT, SOURCE_SHA256, SOURCE_URL, LICENSE_STATUS
from jev_mem.selection import deduplicate_candidates, relevance_state, conditional_state, RELEVANCE_CRITERION, CONDITIONAL_CRITERION

_spec = importlib.util.spec_from_file_location("_jev_summary_runner", ROOT / "tools/run_session_experiments.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
_independent_spec = importlib.util.spec_from_file_location("_jev_summary_independent_runner", ROOT / "tools/run_independent_experiments.py")
independent_runner = importlib.util.module_from_spec(_independent_spec)
_independent_spec.loader.exec_module(independent_runner)

ARCHITECTURE_METHODS = {"mem0": ["sdk20", "rank", "mmr", "jev_single", "conditional"],
                        "independent": ["rank", "mmr", "jev_single", "conditional"]}

PRICES_USD_PER_MILLION = {
    "jev-1.13.0": {"input": 0.042, "output": 0.0},
    "gemini-2.5-flash": {"input": 0.30, "output_including_thinking": 2.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output_including_thinking": 2.50},
}


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def quantiles(values):
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return {"n": len(values), "p50": percentile(values, .5), "p90": percentile(values, .9),
            "p95": percentile(values, .95), "p99": percentile(values, .99)}


def usage_cost(row):
    """Known component estimate; None means no complete billing estimate."""
    usage = row.get("usage") or {}
    if row.get("operation") == "countTokens":
        return {"cost_usd": 0.0, "known_component_usd": 0.0, "unknown": False,
                "reason": "token_count_operation_excluded_from_generation_billing", "input_tokens": None,
                "output_tokens": None, "thinking_tokens": None}
    model = row.get("model")
    if model == "jev-1.13.0":
        input_tokens, output_tokens, thinking = usage.get("input_tokens"), usage.get("output_tokens"), 0
        known = input_tokens * .042 / 1e6 if isinstance(input_tokens, (int, float)) else 0.0
        unknown = not isinstance(input_tokens, (int, float))
    elif model in ("gemini-2.5-flash", "gemini-3.5-flash-lite"):
        input_tokens, output_tokens = usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
        thinking = usage.get("thoughtsTokenCount", 0)
        known = (input_tokens * .30 / 1e6 if isinstance(input_tokens, (int, float)) else 0.0)
        known += (output_tokens * 2.50 / 1e6 if isinstance(output_tokens, (int, float)) else 0.0)
        known += thinking * 2.50 / 1e6 if isinstance(thinking, (int, float)) else 0.0
        unknown = not all(isinstance(value, (int, float)) for value in (input_tokens, output_tokens, thinking))
    else:
        # Neither a missing embed count nor an unspecified embedding tariff is a
        # zero-dollar measurement. Keep both explicitly outside priced subtotal.
        input_tokens = usage.get("promptTokenCount", usage.get("input_tokens"))
        output_tokens, thinking, known, unknown = None, None, 0.0, True
    return {"cost_usd": None if unknown else known, "known_component_usd": known, "unknown": unknown,
            "reason": "missing_usage_or_unpriced_model" if unknown else "configured_unit_price_estimate",
            "input_tokens": input_tokens, "output_tokens": output_tokens, "thinking_tokens": thinking}


def phase_of(row):
    context = row.get("context") or {}
    stage, phase = row.get("stage", ""), context.get("phase", "")
    if stage == "strict_judge" or phase == "evaluation":
        return "evaluation_judge"
    if phase in ("ingestion", "write_typing", "write") or stage in ("mem0_extract", "independent_extract"):
        return "writing"
    if stage.startswith("independent_fact_embedding"):
        return "writing"
    if stage.startswith("independent_query_embedding") or context.get("method") == "shared_retrieval":
        return "reading"
    if phase in ("qa", "retrieval", "reading") or stage == "qa_reader":
        return "reading"
    if context.get("method") in ARCHITECTURE_METHODS["mem0"]:
        return "reading"
    if stage.startswith("mem0_embedding_add"):
        return "writing"
    return "unattributed_phase"


def _request_hash(provider, model, body, operation="generateContent"):
    signature = {"provider": provider, "model": model, "operation": operation, "body": body}
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def current_qa_requests(q, row, retrieval, reference_date):
    """Reconstruct exact primary requests without calling any provider."""
    requests = {}
    def register(provider, model, body, operation, stage):
        key = _request_hash(provider, model, body, operation)
        requests[key] = {"provider": provider, "model": model, "operation": operation, "stage": stage}

    class CaptureReader:
        def generate(self, prompt, *, model="gemini-2.5-flash", system=None, json_output=False,
                     max_output=8192, stage="generation"):
            config = {"temperature": 0, "maxOutputTokens": max_output}
            if model == "gemini-2.5-flash":
                config["thinkingConfig"] = {"thinkingBudget": 0}
            if json_output:
                config["responseMimeType"] = "application/json"
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            register("gemini", model, body, "generateContent", stage)
            if stage == "qa_reader":
                return {"answer": row["prediction"]["answer"],
                        "used_fact_ids": list(row["prediction"].get("used_local_fact_ids", []))}
            return copy.deepcopy(row["judgment"])

    pool = deduplicate_candidates(retrieval["pool60"])
    by_id = {fact["id"]: fact for fact in [*retrieval.get("sdk20", []), *pool]}
    selected = [by_id[fact_id] for fact_id in row["selected_ids"]]
    runner.answer_and_judge(CaptureReader(), q["question"], q["reference"], selected, reference_date=reference_date)
    trace = row.get("selection_trace", {})
    for fact_id in trace.get("single_scores", {}):
        state = relevance_state(q["question"], by_id[fact_id])
        body = {"model": "jev-1.13.0", "state": state,
                "questions": {"relevant": {"type": "noul", "instructions": RELEVANCE_CRITERION}}}
        register("jev", "jev-1.13.0", body, "systemone", "jev_score")
    for round_trace in trace.get("rounds", []):
        selected_before = [by_id[fact_id] for fact_id in round_trace["selected_before"]]
        for fact_id in round_trace["shortlist_ids"]:
            state = conditional_state(q["question"], by_id[fact_id], selected_before)
            body = {"model": "jev-1.13.0", "state": state,
                    "questions": {"new_evidence": {"type": "noul", "instructions": CONDITIONAL_CRITERION}}}
            register("jev", "jev-1.13.0", body, "systemone", "jev_score")
    return requests


def _api_bucket(key):
    return {"phase": key[0], "method": key[1], "provider": key[2], "model": key[3],
            "http_success": 0, "http_failure": 0, "transport_or_unknown_failure": 0, "cache_hits": 0,
            "actual_known_cost_usd": 0.0, "actual_unknown_cost_calls": 0, "actual_latency_ms_values": [],
            "cold_logical_requests": 0, "cold_known_cost_usd": 0.0, "cold_unknown_cost_requests": 0,
            "actual_input_tokens": 0, "actual_output_tokens": 0, "actual_thinking_tokens": 0,
            "actual_usage_missing_calls": 0, "count_tokens_calls": 0, "legacy_unscoped_rows": 0,
            "current_artifact_actual_known_cost_usd": 0.0, "development_overhead_known_cost_usd": 0.0,
            "development_overhead_unknown_cost_calls": 0, "unmatched_or_in_progress_known_cost_usd": 0.0}


def summarize_api_logs(run_dir, conversations, active_qa_requests=None):
    rows, issues = [], []
    for ci in conversations:
        path = Path(run_dir) / f"conv_{ci}/api_calls.jsonl"
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                rows.append({**row, "_conversation": ci})
            except (ValueError, TypeError):
                issues.append({"conversation": ci, "line": line_no, "kind": "invalid_or_incomplete_log_line"})
    # Cache-only rows can recover recorded usage from another log of that exact
    # request, but are never assigned new real billing or zero cold latency.
    canonical = {}
    for row in rows:
        key = (row["_conversation"], row.get("request_hash"))
        score = (not usage_cost(row)["unknown"], not row.get("cache_hit", False), bool(row.get("usage")))
        if key not in canonical or score > canonical[key][0]:
            canonical[key] = (score, row)
    buckets = {}
    cold_seen = set()
    active_hashes = {(key[0], key[3]) for key in active_qa_requests} if active_qa_requests is not None else set()
    archived_answers = collections.defaultdict(set)
    for ci in conversations:
        for path in (Path(run_dir) / f"conv_{ci}/qa_reader_v1_dev8").glob("*/*.json"):
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                archived_answers[ci].add((saved["question"], str(saved["reference"]), saved["prediction"]["answer"]))
            except (ValueError, KeyError, OSError):
                pass
    overhead_cache = {}
    def development_overhead(row):
        ci, request_hash = row["_conversation"], row.get("request_hash")
        if (ci, request_hash) in active_hashes or row.get("stage") not in ("qa_reader", "strict_judge"):
            return False
        key = (ci, request_hash)
        if key not in overhead_cache:
            old = False
            try:
                saved = json.loads((Path(run_dir) / f"conv_{ci}/api_cache/{request_hash}.json").read_text(encoding="utf-8"))
                body = saved["request"]["body"]
                if row["stage"] == "qa_reader":
                    system = "".join(part.get("text", "") for part in body.get("systemInstruction", {}).get("parts", []))
                    old = system != runner.READER_SYSTEM
                else:
                    prompt = json.loads(body["contents"][0]["parts"][0]["text"])
                    old = (prompt["question"], str(prompt["reference_answer"]), prompt["predicted_answer"]) in archived_answers[ci]
            except (OSError, ValueError, KeyError, TypeError):
                pass
            overhead_cache[key] = old
        return overhead_cache[key]
    for row in rows:
        context = row.get("context") or {}
        scoped = bool(context)
        method = context.get("method", "shared" if scoped else "legacy_unscoped")
        phase = phase_of(row)
        key = (phase, method, row.get("provider", "unknown"), row.get("model", "unknown"))
        bucket = buckets.setdefault(key, _api_bucket(key))
        if not scoped:
            bucket["legacy_unscoped_rows"] += 1
        cached = bool(row.get("cache_hit"))
        cost = usage_cost(row)
        if cached:
            bucket["cache_hits"] += 1
        else:
            status = row.get("status")
            if isinstance(status, int) and 200 <= status < 300:
                bucket["http_success"] += 1
            elif isinstance(status, int):
                bucket["http_failure"] += 1
            else:
                bucket["transport_or_unknown_failure"] += 1
            bucket["actual_latency_ms_values"].append(row.get("latency_ms"))
            bucket["actual_known_cost_usd"] += cost["known_component_usd"]
            bucket["actual_unknown_cost_calls"] += int(cost["unknown"])
            if (row["_conversation"], row.get("request_hash")) in active_hashes:
                bucket["current_artifact_actual_known_cost_usd"] += cost["known_component_usd"]
            elif development_overhead(row):
                bucket["development_overhead_known_cost_usd"] += cost["known_component_usd"]
                bucket["development_overhead_unknown_cost_calls"] += int(cost["unknown"])
            else:
                bucket["unmatched_or_in_progress_known_cost_usd"] += cost["known_component_usd"]
            if not row.get("usage") and row.get("operation") != "countTokens":
                bucket["actual_usage_missing_calls"] += 1
            for token_kind in ("input", "output", "thinking"):
                value = cost[token_kind + "_tokens"]
                if isinstance(value, (int, float)):
                    bucket["actual_" + token_kind + "_tokens"] += value
        if row.get("operation") == "countTokens":
            bucket["count_tokens_calls"] += 1
        # Within conversation/phase/method, retries and response-cache reuse of a
        # request for the same question count once for logical cold work.
        cold_key = (row["_conversation"], phase, method, context.get("query_id"), row.get("request_hash"))
        is_qa = context.get("method") in ARCHITECTURE_METHODS["mem0"] or row.get("stage") in ("qa_reader", "strict_judge") or (
            row.get("provider") == "jev" and phase != "writing")
        if row.get("request_hash") and cold_key not in cold_seen and not (active_qa_requests is not None and is_qa):
            cold_seen.add(cold_key)
            source = canonical[(row["_conversation"], row["request_hash"])][1]
            cold = usage_cost(source)
            bucket["cold_logical_requests"] += 1
            bucket["cold_known_cost_usd"] += cold["known_component_usd"]
            bucket["cold_unknown_cost_requests"] += int(cold["unknown"])
    if active_qa_requests is not None:
        for (ci, method, query_id, request_hash), request in active_qa_requests.items():
            phase = "evaluation_judge" if request["stage"] == "strict_judge" else "reading"
            key = (phase, method, request["provider"], request["model"])
            bucket = buckets.setdefault(key, _api_bucket(key))
            source = canonical.get((ci, request_hash), (None, {**request, "usage": {}}))[1]
            cold = usage_cost(source)
            bucket["cold_logical_requests"] += 1
            bucket["cold_known_cost_usd"] += cold["known_component_usd"]
            bucket["cold_unknown_cost_requests"] += int(cold["unknown"])
    groups = []
    for key in sorted(buckets):
        bucket = buckets[key]
        bucket["actual_http_latency_ms"] = quantiles(bucket.pop("actual_latency_ms_values"))
        bucket["actual_cost_complete"] = bucket["actual_unknown_cost_calls"] == 0
        bucket["cold_cost_complete"] = bucket["cold_unknown_cost_requests"] == 0
        groups.append(bucket)
    return {"log_rows": len(rows), "issues": issues, "groups": groups,
            "actual_known_cost_usd": sum(group["actual_known_cost_usd"] for group in groups),
            "actual_unknown_cost_calls": sum(group["actual_unknown_cost_calls"] for group in groups),
            "cold_known_cost_usd": sum(group["cold_known_cost_usd"] for group in groups),
            "cold_unknown_cost_requests": sum(group["cold_unknown_cost_requests"] for group in groups),
            "development_overhead_known_cost_usd": sum(group["development_overhead_known_cost_usd"] for group in groups),
            "development_overhead_unknown_cost_calls": sum(group["development_overhead_unknown_cost_calls"] for group in groups),
            "prices_usd_per_million": PRICES_USD_PER_MILLION,
            "price_basis": "Explicit research estimate parameters; not an invoice; no context-cache/batch discounts applied.",
            "embedding_cost": "unknown when provider usage/tariff is unavailable; never imputed as zero",
            "cold_semantics": "QA: exact request hashes reconstructed from current valid QA artifacts only; per conversation/method/query/hash once. Shared write/retrieval work deduplicated separately. Retries excluded.",
            "development_semantics": "Archived-reader prompts and archived v1 judge answers, unless that request is reused by current artifacts. Unmatched ongoing work is not automatically labeled development.",
            "latency_semantics": "actual non-cache HTTP attempts including failures; excludes backoff, queues and local work",
            "scope": "all existing recorded calls in requested conversation directories, including retries and earlier attempts"}


def metric_group(expected, rows):
    count = len(expected)
    selected = [row["selected_count"] for row in rows]
    fallback = [row.get("selection_trace", {}).get("fallback_count", 0) for row in rows]
    def averages(key):
        values = [float(row["judgment"]["correct"]) if key == "strict_accuracy" else float(row[key]) for row in rows]
        total = sum(values)
        return {"sum": total, "completed_only": total / len(rows) if rows else None,
                "all_requested": total / count if count else None}
    official = ({"sum": sum(row["_official_locomo_f1"] for row in rows),
                 "completed_only": sum(row["_official_locomo_f1"] for row in rows) / len(rows) if rows else None,
                 "all_requested": sum(row["_official_locomo_f1"] for row in rows) / count if count else None}
                if all("_official_locomo_f1" in row for row in rows) else None)
    return {"requested": count, "completed": len(rows), "missing": count - len(rows),
            "strict_accuracy": averages("strict_accuracy"), "token_f1": averages("token_f1"),
            "custom_exact_match": averages("exact_match"), "official_locomo_f1": official,
            "selected_count_distribution": dict(collections.Counter(selected)),
            "exactly_20": sum(value == 20 for value in selected), "fewer_than_20": sum(value < 20 for value in selected),
            "fallback_queries": sum(value > 0 for value in fallback), "fallback_facts": sum(fallback),
            "query_elapsed_seconds": quantiles([row.get("elapsed_seconds") for row in rows]),
            "metric_note": "token_f1/exact_match are current custom lexical metrics, not official category-aware LoCoMo F1"}


def paired_bootstrap(rows_by_method, methods, baseline, samples=2000):
    output = {}
    base = rows_by_method.get(baseline, {})
    for method in methods:
        if method == baseline:
            continue
        other = rows_by_method.get(method, {})
        common = sorted(set(base) & set(other))
        clusters = collections.defaultdict(list)
        for qid in common:
            clusters[base[qid]["conversation"]].append(qid)
        cis = sorted(clusters)
        metrics = {}
        metric_names = ["strict_accuracy", "token_f1", "exact_match"]
        if common and all("_official_locomo_f1" in base[qid] and "_official_locomo_f1" in other[qid] for qid in common):
            metric_names.append("_official_locomo_f1")
        for metric in metric_names:
            def value(row):
                return float(row["judgment"]["correct"]) if metric == "strict_accuracy" else float(row[metric])
            numerators = {ci: sum(value(other[qid]) - value(base[qid]) for qid in clusters[ci]) for ci in cis}
            point = sum(numerators.values()) / len(common) if common else None
            draws = []
            if len(cis) >= 2:
                rng = random.Random(0)
                for _ in range(samples):
                    chosen = rng.choices(cis, k=len(cis))
                    draws.append(sum(numerators[ci] for ci in chosen) / sum(len(clusters[ci]) for ci in chosen))
            metrics[metric.lstrip("_")] = {"difference_vs_baseline": point,
                               "bootstrap_95_percentile_interval": [percentile(draws, .025), percentile(draws, .975)] if draws else None}
        output[method] = {"baseline": baseline, "paired_queries": len(common), "paired_conversations": cis,
                          "seed": 0, "bootstrap_samples": samples, "metrics": metrics,
                          "significance_claim": False,
                          "interpretation": "Exploratory conversation-cluster bootstrap; few clusters and incomplete paired subsets do not support significance claims."}
    return output


def detect_architecture(run_dir):
    manifest_path = Path(run_dir) / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") == "independent-live-run-v1":
            return "independent"
    return "mem0"


def summarize_run(run_dir, conversations, methods=None, *, baseline=None, bootstrap_samples=2000):
    run_dir = Path(run_dir)
    architecture = detect_architecture(run_dir)
    methods = list(methods or ARCHITECTURE_METHODS[architecture])
    if not methods or len(set(methods)) != len(methods) or any(method not in ARCHITECTURE_METHODS[architecture] for method in methods):
        raise ValueError(f"Methods must belong to {architecture}: {ARCHITECTURE_METHODS[architecture]}")
    baseline = baseline or ("sdk20" if "sdk20" in methods else "rank" if "rank" in methods else methods[0])
    if baseline not in methods:
        raise ValueError("The paired baseline must be one of the selected methods")
    data = load_data()
    official_source = ROOT / "research/sources/locomo_evaluation_cbfbc1.py"
    if hashlib.sha256(official_source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned official metric source snapshot failed its hash check")
    expected = {q["id"]: {**q, "conversation": ci} for ci in conversations for q in queries(data[ci], ci)}
    rows_by_method = {method: {} for method in methods}
    active_qa_requests = {}
    issues, ingestion = [], {}
    manifest = None
    independent_protocol_valid = True
    if architecture == "independent":
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("mode") != "full"
                or manifest.get("protocol_sha256") != independent_runner._hash(manifest.get("protocol"))
                or manifest.get("protocol") != independent_runner._protocol()):
            issues.append({"kind": "independent_run_manifest_smoke_or_protocol_code_mismatch"})
            independent_protocol_valid = False
    for ci in conversations:
        conv_dir = run_dir / f"conv_{ci}"
        if not independent_protocol_valid:
            continue
        try:
            if architecture == "independent":
                summary, _ = independent_runner._ingestion(conv_dir, manifest, False)
                expected_sessions = [row["session_id"] for row in session_envelopes(data[ci])]
                bank = json.loads((conv_dir / "independent_store/fact_bank.json").read_text(encoding="utf-8"))
                if summary.get("session_ids") != expected_sessions or [row["session_id"] for row in bank["sessions"]] != expected_sessions:
                    raise ValueError("Independent committed sessions do not match full source session IDs")
                facts_hash = summary["facts_sha256"]
            else:
                summary = json.loads((conv_dir / "ingestion_summary.json").read_text(encoding="utf-8"))
                facts_hash = hashlib.sha256((conv_dir / "facts.json").read_bytes()).hexdigest()
            ingestion[str(ci)] = summary
        except (OSError, ValueError, KeyError, RuntimeError):
            issues.append({"conversation": ci, "kind": "ingestion_snapshot_missing_or_incomplete"})
            continue
        if not summary.get("complete"):
            issues.append({"conversation": ci, "kind": "partial_ingestion_excluded"})
            continue
        reference_date = list(session_envelopes(data[ci]))[-1]["observation_date"]
        for q in queries(data[ci], ci):
            try:
                retrieval = json.loads((conv_dir / "retrieval" / (q["id"] + ".json")).read_text(encoding="utf-8"))
                if architecture == "independent":
                    independent_runner._verify_retrieval(retrieval, independent_runner._retrieval_binding(q, summary, manifest))
                else:
                    runner._verify_retrieval(retrieval, q, summary, facts_hash)
            except (OSError, ValueError, KeyError, RuntimeError):
                if any((conv_dir / "qa" / method / (q["id"] + ".json")).exists() for method in methods):
                    issues.append({"query_id": q["id"], "kind": "qa_has_missing_or_invalid_retrieval"})
                continue
            for method in methods:
                path = conv_dir / "qa" / method / (q["id"] + ".json")
                if not path.exists():
                    continue
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if architecture == "independent":
                        binding = {**independent_runner._retrieval_binding(q, summary, manifest), "method": method,
                                   "reference_sha256": independent_runner._hash(q["reference"]),
                                   "retrieval_sha256": independent_runner._hash(retrieval), "reference_date": reference_date}
                        independent_runner._check_binding(row, binding, "QA")
                        independent_runner._verify_qa_content(row)
                    elif (row.get("cache_schema") != runner.CACHE_SCHEMA
                          or row.get("request_fingerprint") != runner._qa_fingerprint(q, method, retrieval, reference_date)):
                        raise ValueError("Mem0 QA protocol fingerprint mismatch")
                    if (row.get("partial_conversation_smoke") is not False or row.get("method") != method
                            or row.get("query_id") != q["id"] or row.get("conversation") != ci
                            or row.get("category") != q["category"] or not isinstance(row["judgment"]["correct"], bool)):
                        raise ValueError("QA protocol/identity mismatch")
                    for field in ("token_f1", "exact_match"):
                        if not isinstance(row[field], (int, float)) or not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                            raise ValueError("Invalid lexical metric")
                    ids = row["selected_ids"]
                    if row["selected_count"] != len(ids) or len(set(ids)) != len(ids) or not 0 <= len(ids) <= 20:
                        raise ValueError("Invalid reader fact budget")
                    row["_official_locomo_f1"] = official_locomo_f1(row["prediction"]["answer"], q["reference"], q["category"])
                    reconstructed = current_qa_requests(q, row, retrieval, reference_date)
                    rows_by_method[method][q["id"]] = row
                    for request_hash, request in reconstructed.items():
                        active_qa_requests[(ci, method, q["id"], request_hash)] = request
                except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                    issues.append({"query_id": q["id"], "method": method, "kind": "invalid_or_stale_qa_excluded"})
    groups = {}
    for method in methods:
        rows = list(rows_by_method[method].values())
        groups[method] = {"overall": metric_group(expected, rows),
            "by_conversation": {str(ci): metric_group([q for q in expected.values() if q["conversation"] == ci],
                                                     [row for row in rows if row["conversation"] == ci]) for ci in conversations},
            "by_category": {str(category): metric_group([q for q in expected.values() if q["category"] == category],
                                                        [row for row in rows if row["category"] == category]) for category in (1, 2, 3, 4)}}
        category_scores = [group["official_locomo_f1"]["all_requested"] for group in groups[method]["by_category"].values()
                           if group["requested"] and group["official_locomo_f1"] is not None]
        groups[method]["supplementary_equal_category_macro_f1_all_requested"] = (
            sum(category_scores) / len(category_scores) if category_scores else None)
    complete = bool(expected) and all(len(rows_by_method[method]) == len(expected) for method in methods)
    full = set(conversations) == set(range(10)) and len(expected) == 1540
    return {"run_dir": str(run_dir.resolve()), "architecture": architecture, "baseline": baseline,
            "snapshot_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete_full_1540" if complete and full else "complete_requested_cohort" if complete else "in_progress",
            "requested_conversations": conversations, "requested_questions": len(expected), "full_1540": complete and full,
            "methods": groups, "ingestion": ingestion, "issues": issues,
            "api_usage": summarize_api_logs(run_dir, conversations, active_qa_requests),
            "paired_differences": paired_bootstrap(rows_by_method, methods, baseline, bootstrap_samples),
            "per_query_metrics": {method: [{"query_id": qid, "conversation": row["conversation"], "category": row["category"],
                                             "strict_correct": row["judgment"]["correct"], "custom_token_f1": row["token_f1"],
                                             "custom_exact_match": row["exact_match"], "official_locomo_f1": row["_official_locomo_f1"]}
                                            for qid, row in sorted(rows_by_method[method].items())] for method in methods},
            "official_metric_provenance": {"commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
                                            "url": SOURCE_URL, "license_status": LICENSE_STATUS,
                                            "overall_aggregation": "mean over questions, matching official evaluation_stats; equal-category macro is separately labeled supplementary"},
            "development_policy": "Conversations0/1 are development. Full1540 includes development; prior workspace exposure means no clean-holdout claim.",
            "missing_policy": "All requested denominator shared; missing/failed results contribute zero until complete. Completed-only rates can have selection bias."}


def markdown(report):
    lines = ["# JEV Mem 实验快照", "", f"状态：`{report['status']}`；固定 conversation：{report['requested_conversations']}；应评估 **{report['requested_questions']}题/方法**。", "",
             "尚未完成时，全请求分母把失败/缺失计为零贡献，仅表示保守进度；completed-only 可能受完成顺序影响。custom F1/EM与官方LoCoMo分题型F1分列。Judge分数是固定严格提示下的自动评分，尚不等于经过人工验证的正确率；开发集已有评分不一致记录。", "",
             "| 方法 | 完成/应完成 | judge分数（已完成/全请求） | custom F1（已完成/全请求） | 官方F1（已完成/全请求） | EM（已完成/全请求） | 20条/不足20条 | 回退题数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def pair(metric):
        return "/".join("NA" if metric[key] is None else f"{metric[key]:.4f}" for key in ("completed_only", "all_requested"))
    for method, groups in report["methods"].items():
        row = groups["overall"]
        official = pair(row['official_locomo_f1']) if row['official_locomo_f1'] is not None else "NA"
        lines.append(f"| {method} | {row['completed']}/{row['requested']} | {pair(row['strict_accuracy'])} | {pair(row['token_f1'])} | {official} | {pair(row['custom_exact_match'])} | {row['exactly_20']}/{row['fewer_than_20']} | {row['fallback_queries']} |")
    api = report["api_usage"]
    lines += ["", f"实际已知费用分量估计：**${api['actual_known_cost_usd']:.6f}**；另有 **{api['actual_unknown_cost_calls']}次**费用不完整/未知。其中已识别旧reader开发开销：${api['development_overhead_known_cost_usd']:.6f}。冷调用等效已知分量：${api['cold_known_cost_usd']:.6f}，另有{api['cold_unknown_cost_requests']}个逻辑请求未知。", "",
              "以上按配置单价和API usage估计，**不是账单**。Embedding缺失usage不按零费用处理；countTokens不计生成费用。缓存命中不会重复计入实际费用。实际HTTP延迟分位数不包括排队、重试等待和本地计算。", "",
              "| 阶段/方法/模型 | HTTP成功/失败/传输异常 | 缓存命中 | HTTP p50/p95 ms | 实际已知$ | 费用未知调用 | 冷等效已知$ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in api["groups"]:
        latency = group["actual_http_latency_ms"]
        lat = "/".join("NA" if latency[key] is None else f"{latency[key]:.1f}" for key in ("p50", "p95"))
        lines.append(f"| {group['phase']}/{group['method']}/{group['model']} | {group['http_success']}/{group['http_failure']}/{group['transport_or_unknown_failure']} | {group['cache_hits']} | {lat} | {group['actual_known_cost_usd']:.6f} | {group['actual_unknown_cost_calls']} | {group['cold_known_cost_usd']:.6f} |")
    lines += ["", "按 conversation 和题型的完整指标、选中条数分布及配对区间见同名JSON。配对区间使用seed0、conversation簇重采样；仅两段conversation时不作显著性宣称。", "",
              "固定conv0/1为开发；全1540包含开发数据。此前已接触LoCoMo，剩余conversation也不称clean holdout。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--methods", help="Default: five methods for Mem0, four without sdk20 for independent")
    parser.add_argument("--baseline", help="Default: sdk20 when present, otherwise rank")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    conversations = [int(value) for value in args.conversations.split(",")]
    methods = args.methods.split(",") if args.methods else None
    if (not conversations or len(set(conversations)) != len(conversations) or not set(conversations) <= set(range(10))
            or (methods is not None and (not methods or len(set(methods)) != len(methods))) or args.bootstrap_samples < 1):
        parser.error("Invalid conversation/method/bootstrap configuration")
    for path in args.run_dir:
        report = summarize_run(path, conversations, methods, baseline=args.baseline, bootstrap_samples=args.bootstrap_samples)
        output = Path(args.output_dir) if args.output_dir else Path(path) / "reports"
        output.mkdir(parents=True, exist_ok=True)
        stem = Path(path).resolve().name + "_" + ("full1540" if len(conversations) == 10 else "conv_" + "_".join(map(str, conversations)))
        runner.dump_json(output / (stem + ".json"), report)
        (output / (stem + ".md")).write_text(markdown(report), encoding="utf-8")
        print(json.dumps({"status": report["status"], "requested_questions": report["requested_questions"],
                          "json": str(output / (stem + ".json")), "markdown": str(output / (stem + ".md"))}))


if __name__ == "__main__":
    main()
