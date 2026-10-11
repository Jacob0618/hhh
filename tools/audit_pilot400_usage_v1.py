"""Independent offline physical-usage audit; never imports a provider/client.

Prices are the frozen experiment's configured estimates, not invoice evidence.
Only an explicitly named audit output directory is written.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(shared_read(path))


def parse_ledger(raw, *, snapshot=False):
    if raw and not raw.endswith(b"\n"):
        if not snapshot:
            raise ValueError("Terminal ledger has an incomplete trailing line")
        raw = raw[:raw.rfind(b"\n") + 1]
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("request_hash"), str):
            raise ValueError("Malformed API ledger row")
        rows.append(row)
    return raw, rows


def partitions(write, tag, final):
    if not tag.startswith(write) or not final.startswith(tag):
        raise ValueError("Sealed stage ledgers are not byte-exact cumulative prefixes")
    parts = {"write": write, "tag": tag[len(write):], "qa": final[len(tag):]}
    parsed = {}
    for stage, raw in parts.items():
        _, rows = parse_ledger(raw)
        for row in rows:
            if row.get("context", {}).get("driver_stage") != stage:
                raise ValueError("Stage boundary/context disagreement: " + stage)
        parsed[stage] = rows
    return parsed


def prices_from_source(path):
    raw = shared_read(path)
    tree = ast.parse(raw.decode("utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PRICES_USD_PER_MILLION" for t in node.targets):
            return ast.literal_eval(node.value), sha(raw)
    raise ValueError("Configured prices assignment missing")


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def physical_usage(row, prices):
    usage = row.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("Usage must be an object")
    reasons = set()
    tokens = {"input": None, "output": None, "thinking": None}
    cost = Decimal(0)
    status = row.get("status")
    if type(status) is not int:
        reasons.add("transport_or_unknown_usage")
    model, operation = row.get("model"), row.get("operation")
    if operation == "countTokens":
        return cost, reasons, tokens
    if model == "jev-1.13.0":
        tokens.update(input=usage.get("input_tokens"), output=usage.get("output_tokens"), thinking=0)
        rates = {"input": prices[model]["input"]}
    elif model in ("gemini-2.5-flash", "gemini-3.5-flash-lite"):
        tokens.update(input=usage.get("promptTokenCount"), output=usage.get("candidatesTokenCount"), thinking=usage.get("thoughtsTokenCount", 0))
        rates = {"input": prices[model]["input"], "output": prices[model]["output_including_thinking"], "thinking": prices[model]["output_including_thinking"]}
    else:
        tokens["input"] = usage.get("promptTokenCount", usage.get("input_tokens"))
        rates = {}
        reasons.add("unpriced_model")
        if not numeric(tokens["input"]):
            reasons.add("missing_input_usage")
    for name, rate in rates.items():
        if numeric(tokens[name]):
            cost += Decimal(str(tokens[name])) * Decimal(str(rate)) / Decimal(1000000)
        else:
            reasons.add("missing_" + name + "_usage")
    return cost, reasons, tokens


def summarize_rows(rows, prices):
    counts = Counter(rows=len(rows))
    tokens = Counter()
    unknown = Counter()
    cost = Decimal(0)
    hashes = Counter()
    latencies = []
    for row in rows:
        if row.get("cache_hit") is True:
            counts["cache_hits"] += 1
            continue
        if row.get("cache_hit") not in (False, None):
            raise ValueError("Invalid cache_hit")
        counts["physical_attempts"] += 1
        hashes[row["request_hash"]] += 1
        status = row.get("status")
        if type(status) is int and 200 <= status < 300:
            counts["http_successes"] += 1
        elif type(status) is int and 400 <= status < 600:
            counts["http_failures"] += 1
        else:
            counts["transport_or_unknown"] += 1
        component, reasons, tokenrow = physical_usage(row, prices)
        cost += component
        if reasons:
            counts["unknown_cost_attempts"] += 1
            unknown.update(reasons)
        for key, value in tokenrow.items():
            if numeric(value):
                tokens[key] += value
        if numeric(row.get("latency_ms")):
            latencies.append(row["latency_ms"])
        else:
            counts["missing_latency_attempts"] += 1
    for key in ("rows", "physical_attempts", "cache_hits", "http_successes", "http_failures", "transport_or_unknown", "unknown_cost_attempts", "missing_latency_attempts"):
        counts.setdefault(key, 0)
    return {**dict(counts), "known_cost_component_usd": float(cost),
            "known_cost_component_usd_decimal": str(cost),
            "total_cost_usd": None if counts["unknown_cost_attempts"] else float(cost),
            "unknown_cost_reasons": dict(unknown), "recorded_tokens": dict(tokens),
            "unique_physical_request_hashes": len(hashes),
            "extra_physical_attempts_with_repeated_hash": sum(n - 1 for n in hashes.values()),
            "sum_physical_latency_ms": sum(latencies),
            "latency_note": "Sum of request latency across concurrent workers, not wall-clock duration."}


def grouped(rows, key, prices):
    groups = {}
    for row in rows:
        group = key(row)
        groups.setdefault(str(group), []).append(row)
    return {k: summarize_rows(v, prices) for k, v in sorted(groups.items())}


def ledger_report(path, prices, expected_sha=None):
    raw, rows = parse_ledger(shared_read(path))
    if expected_sha is not None and sha(raw) != expected_sha:
        raise ValueError("Lineage ledger digest changed: " + str(path))
    return {"path": str(Path(path).resolve()), "sha256": sha(raw), **summarize_rows(rows, prices)}


def source_lineage(run, all_rows, prices):
    path = run / "cache_seed_lineage.json"
    raw = shared_read(path)
    lineage = json.loads(raw)
    seed_hashes = {x["request_hash"] for x in lineage["copied_cache_files"]}
    if len(seed_hashes) != len(lineage["copied_cache_files"]):
        raise ValueError("Duplicate seed hash")
    # Read every migrated immutable response to validate the seed, not just hits.
    for item in lineage["copied_cache_files"]:
        cached = shared_read(run / "api_cache" / (item["request_hash"] + ".json"))
        if len(cached) != item["bytes"] or sha(cached) != item["sha256"]:
            raise ValueError("Seed response changed: " + item["request_hash"])
    if sha(shared_read(lineage["source_seal_path"])) != lineage["source_seal_sha256"]:
        raise ValueError("Seed source seal changed")
    seed_hits = [r for r in all_rows if r.get("cache_hit") is True and r["request_hash"] in seed_hashes]
    other_hits = [r for r in all_rows if r.get("cache_hit") is True and r["request_hash"] not in seed_hashes]
    journals = []
    # groups/*.json contains committed records; pending_group.json includes each
    # failed transaction exactly once. Do not duplicate group_fact_bank copies.
    paths = sorted(run.glob("conversations/conv_*/banks/*/groups/*.json"))
    paths += sorted(run.glob("conversations/conv_*/banks/*/pending_group.json"))
    for file in paths:
        data = read_json(file)
        for name, journal in data.get("extraction_attempts", {}).items():
            source = journal.get("response_source", "")
            if source.startswith("validated_provider_cache"):
                journals.append({"path": str(file.relative_to(run)).replace("\\", "/"),
                                 "group_id": data.get("group_id"), "attempt": name,
                                 "request_hash": journal["request_hash"], "response_source": source,
                                 "from_v1_seed": journal["request_hash"] in seed_hashes,
                                 "validation": journal.get("validation")})
    return {"lineage_sha256": sha(raw), "copied_response_files": len(seed_hashes),
            "logged_cache_hits_from_v1_seed": len(seed_hits),
            "logged_v1_seed_hashes_reused": len({r["request_hash"] for r in seed_hits}),
            "logged_cache_hits_created_in_v2": len(other_hits),
            "extraction_journal_cache_reuses": journals,
            "extraction_journal_cache_reuse_count": len(journals),
            "extraction_journal_v1_seed_reuse_count": sum(j["from_v1_seed"] for j in journals),
            "v1_source": ledger_report(lineage["source_ledger_path"], prices, lineage["source_ledger_sha256"]),
            "unused_exact_payload_retry_diagnostics": [ledger_report(i["path"], prices, i["sha256"]) for i in lineage.get("unused_diagnostic_ledgers", [])],
            "note": "Ledger cache hits and direct extraction-journal reuse are reuse events, not new HTTP charges. V1 source work and unused diagnostic retry are separately spent; do not treat this seeded run as an unseeded cold start."}


def audit(run, boundary_dir, *, snapshot=False):
    run, boundary_dir = Path(run).resolve(), Path(boundary_dir).resolve()
    status_raw = shared_read(run / "qa_status.json")
    status = json.loads(status_raw)
    if not snapshot and (status.get("status") not in ("complete", "requires_review", "failed") or status.get("done") != status.get("expected") or status.get("expected") != 10):
        raise ValueError("QA is not fully terminal; use --snapshot only for explicitly interim audit")
    prices, price_sha = prices_from_source(ROOT / "tools/summarize_experiments.py")
    write = shared_read(boundary_dir / "write_terminal/api_calls.jsonl")
    tag = shared_read(boundary_dir / "tag_terminal/api_calls.jsonl")
    allraw = shared_read(run / "api_calls.jsonl")
    allraw, allrows = parse_ledger(allraw, snapshot=snapshot)
    parts = partitions(write, tag, allraw)
    stats = {stage: {"boundary_start_bytes": 0 if stage == "write" else len(write) if stage == "tag" else len(tag),
                     "boundary_end_bytes": len(write) if stage == "write" else len(tag) if stage == "tag" else len(allraw),
                     "usage": summarize_rows(rows, prices),
                     "by_stage": grouped(rows, lambda r: r.get("stage", "unscoped"), prices),
                     "by_model": grouped(rows, lambda r: r.get("model", "unspecified"), prices),
                     "by_method": grouped(rows, lambda r: r.get("context", {}).get("method", "unspecified"), prices)}
             for stage, rows in parts.items()}
    legacy = source_lineage(run, allrows, prices)
    total = summarize_rows(allrows, prices)
    separated = [total, legacy["v1_source"], *legacy["unused_exact_payload_retry_diagnostics"]]
    spending = sum(Decimal(x["known_cost_component_usd_decimal"]) for x in separated)
    return {"schema": "jev-mem-pilot400-independent-usage-audit-v1",
            "status": "running_snapshot" if status.get("status") == "running" else "terminal",
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run),
            "qa_status": status["status"], "qa_status_sha256": sha(status_raw),
            "run_manifest_sha256": sha(shared_read(run / "run_manifest.json")),
            "ledger_sha256": sha(allraw), "ledger_bytes": len(allraw),
            "prefix_validation": {"write_sha256": sha(write), "tag_cumulative_sha256": sha(tag), "byte_exact_prefixes": True},
            "configured_prices": prices, "price_source_sha256": price_sha,
            "current_namespace": total, "stages": stats, "seed_lineage": legacy,
            "all_reported_namespaces_known_spend_usd": float(spending),
            "all_reported_namespaces_unknown_cost_attempts": sum(x["unknown_cost_attempts"] for x in separated),
            "caveats": ["Configured estimates, not invoices.", "Unknown embedding or transport usage is not billed zero.", "One physical row is one attempt even for a repeated hash; cache rows carry no new usage.", "HTTP 200 is transport success, not schema or semantic success.", "Shared write/tag/query preparation costs are not silently assigned to one QA arm."]}


def markdown(report):
    rows = [("# Pilot 400 independent usage audit"), "", "Status: " + report["status"],
            "", "| Namespace / stage | Physical attempts | Cache events | Known USD component | Unknown cost attempts |",
            "| --- | ---: | ---: | ---: | ---: |"]
    items = [(k, x["usage"]) for k, x in report["stages"].items()]
    items += [("v2 total", report["current_namespace"]), ("v1 seed source (separate)", report["seed_lineage"]["v1_source"])]
    items += [("unused exact-payload diagnostic (separate)", x) for x in report["seed_lineage"]["unused_exact_payload_retry_diagnostics"]]
    for key, x in items:
        rows.append(f"| {key} | {x['physical_attempts']} | {x['cache_hits']} | {x['known_cost_component_usd']:.9f} | {x['unknown_cost_attempts']} |")
    rows += ["", "Byte-exact write/tag/final ledger prefixes verified. Prices use the frozen configured tariff; these are estimates rather than invoices. Unknown amounts are not zero. The JSON gives per-stage, model, method, usage, cache lineage and physical latency totals.",
             "", "Total known component across v1 seed work, v2 and the unused diagnostic: $" + format(report["all_reported_namespaces_known_spend_usd"], ".9f") + "; unknown attempts: " + str(report["all_reported_namespaces_unknown_cost_attempts"]) + "."]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/pilot400_modules_v2")
    parser.add_argument("--boundary-dir", type=Path, default=ROOT / "research/heartbeat_checks/20260928T0014")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    report = audit(args.run_dir, args.boundary_dir, snapshot=args.snapshot)
    # New output only: never overwrite a sealed or earlier audit snapshot.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "usage_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "usage_audit.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"path": str(args.output_dir), "status": report["status"], "current_namespace": report["current_namespace"]}))


if __name__ == "__main__":
    main()

