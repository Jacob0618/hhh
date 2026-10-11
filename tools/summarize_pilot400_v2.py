"""Offline v2 protocol/bank gate, unchanged fixed400 metrics, and physical usage."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_experiments import usage_cost, PRICES_USD_PER_MILLION


def v2_metrics():
    """Isolate the unchanged metric code while requiring the v2 artifact identity."""
    path = ROOT / "tools/summarize_pilot400.py"
    spec = importlib.util.spec_from_file_location("_pilot400_v2_metric_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCHEMA = "jev-mem-pilot400-qa-v2"
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ledger_usage(path):
    """Count physical attempts, including repeated hashes; cache events are free."""
    path = Path(path)
    if not path.exists():
        raise ValueError("Missing API ledger; absence is not zero cost")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stages = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("request_hash"), str):
            raise ValueError("Malformed ledger row")
        stage = row.get("stage", "unscoped")
        item = stages.setdefault(stage, {"physical_attempts": 0, "cache_hits": 0,
            "http_successes": 0, "http_failures": 0, "transport_or_unknown": 0,
            "known_cost_component_usd": 0.0, "unknown_cost_attempts": 0,
            "recorded_input_tokens": 0, "recorded_output_tokens": 0, "recorded_thinking_tokens": 0})
        if row.get("cache_hit") is True:
            item["cache_hits"] += 1
            continue
        if row.get("cache_hit") not in (False, None):
            raise ValueError("cache_hit must be a boolean")
        item["physical_attempts"] += 1
        status = row.get("status")
        if type(status) is int and 200 <= status < 300:
            item["http_successes"] += 1
        elif type(status) is int and 400 <= status < 600:
            item["http_failures"] += 1
        else:
            item["transport_or_unknown"] += 1
        cost = usage_cost(row)
        unknown = bool(cost["unknown"]) or type(status) is not int
        item["known_cost_component_usd"] += cost["known_component_usd"]
        item["unknown_cost_attempts"] += unknown
        for token in ("input", "output", "thinking"):
            value = cost[token + "_tokens"]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item["recorded_" + token + "_tokens"] += value
    total = {key: sum(stage[key] for stage in stages.values())
             for key in ("physical_attempts", "cache_hits", "http_successes", "http_failures",
                         "transport_or_unknown", "known_cost_component_usd", "unknown_cost_attempts")}
    total["total_cost_usd"] = None if total["unknown_cost_attempts"] else total["known_cost_component_usd"]
    return {"path": str(path.resolve()), "sha256": sha(path), "rows": len(rows),
            **total, "stages": stages, "prices_usd_per_million": PRICES_USD_PER_MILLION,
            "note": "Configured usage-price estimate, not an invoice. Unknown usage is not a billed zero. Cache rows add no new fees; retries are physical attempts, not distinct questions."}


def audit_v2_banks(run_dir):
    from tools import run_pilot400_experiments_v2 as driver
    from jev_mem.group_writer_v2 import validate_generation_provenance
    run_dir = Path(run_dir).resolve()
    manifest = driver.ensure_manifest(run_dir, create=False)
    data = driver.load_data()
    banks = []
    for ci, conv in enumerate(data):
        raw = driver.raw_turns_from_locomo(conv["conversation"])
        for name in ("session", "regroup"):
            bank = driver.load_bank(run_dir, ci, name, raw, manifest)
            records = json.loads((run_dir / bank["path"]).read_text(encoding="utf-8"))["groups"]
            extraction = []
            sources = Counter()
            for record in records:
                checked = validate_generation_provenance(record)
                extraction.append({"group_id": record["group_id"], **checked})
                for journal in record["extraction_attempts"].values():
                    cache = json.loads((run_dir / "api_cache" / (journal["request_hash"] + ".json")).read_text(encoding="utf-8"))
                    if (cache["request"] != journal["request"] or cache["response"] != journal["http_response"]
                            or cache["stage"] != journal["stage"]):
                        raise ValueError("Committed generation differs from its bound HTTP cache")
                    sources[journal["response_source"]] += 1
            banks.append({"conversation": ci, "bank": name, "path": bank["path"],
                          "sha256": bank["sha256"], "facts": len(bank["facts"]),
                          "summary_sha256": bank["summary_sha256"], "generation_provenance": extraction,
                          "generation_response_sources": dict(sources)})
    return {"run_manifest_sha256": sha(run_dir / "run_manifest.json"),
            "protocol_sha256": manifest["protocol_sha256"], "banks": banks,
            "scope": "All twenty complete banks validated against frozen v2 code, source dates and coverage before reporting QA metrics. This does not certify semantic fact support."}


def seed_provenance(run_dir):
    path = Path(run_dir) / "cache_seed_lineage.json"
    if not path.exists():
        return {"seeded": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    source_ledger = Path(data["source_ledger_path"])
    if sha(source_ledger) != data["source_ledger_sha256"]:
        raise ValueError("Seed source ledger changed")
    diagnostics = []
    for item in data.get("unused_diagnostic_ledgers", []):
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Separate diagnostic ledger changed")
        diagnostics.append(ledger_usage(item["path"]))
    return {"seeded": True, "lineage_sha256": sha(path),
            "source_ledger": ledger_usage(source_ledger),
            "unused_diagnostic_ledgers": diagnostics,
            "note": "Source development cost is reported separately. It is not added again for each cache hit; it must not be hidden as a free cold start."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    args = p.parse_args()
    gate = audit_v2_banks(args.run_dir)
    metrics = v2_metrics()
    report = metrics.summarize_pilot(args.run_dir, bootstrap_samples=args.bootstrap_samples)
    report.update(schema="jev-mem-pilot400-summary-v2", writer_v2_bank_audit=gate,
                  current_namespace_usage=ledger_usage(args.run_dir / "api_calls.jsonl"),
                  cache_seed_provenance=seed_provenance(args.run_dir))
    report["interpretation"]["writer"] = "All three arms share the v2 bounded validation-feedback extraction policy; the original v1 development failure and exact-payload failed retry remain separate. No old QA results are mixed in."
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(metrics.markdown_report(report) + "\nWriter v2, full source/date/bank checks passed. See JSON for actual usage, inherited cache cost and protocol hashes.\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "summary.json"),
        "status_counts": {k: v["status_counts"] for k, v in report["arms"].items()}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
