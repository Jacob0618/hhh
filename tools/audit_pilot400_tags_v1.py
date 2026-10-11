"""Read-only pilot400 fact-tag audit; no credentials, API clients or QA/gold reads.

Replays the frozen type_one_fact -> ModelAPI.score_many methods against local
HTTP cache only. Outputs must be outside the run, and are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pilot400_experiments_v2 as driver
from jev_mem.providers import ModelAPI
from jev_mem.pilot400 import type_one_fact, validate_tags
from jev_mem.selection import COARSE_CRITERIA, FINE_CRITERIA

STAGE = "pilot_fact_typing"
DIMENSIONS = tuple({**COARSE_CRITERIA, **FINE_CRITERIA})
SCHEMA = "jev-mem-pilot400-tag-audit-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CacheOnlyJudge:
    """Not a ModelAPI instance; the sole request implementation reads a file."""
    score_many = ModelAPI.score_many

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.proof = None

    def request(self, provider, model, body, *, operation, stage):
        if (provider, model, operation, stage) != ("jev", "jev-1.13.0", "systemone", STAGE):
            raise ValueError("Unexpected fact typing request")
        signature = dict(provider=provider, model=model, operation=operation, body=body)
        digest = sha(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode())
        self.proof = {"request_hash": digest}
        path = self.cache_dir / (digest + ".json")
        raw = path.read_bytes()
        cache = json.loads(raw)
        if cache.get("request") != signature or cache.get("stage") != STAGE:
            raise ValueError("Fact typing cache request/stage differs")
        self.proof.update(cache_sha256=sha(raw), raw_response_sha256=driver.fingerprint(cache["response"]))
        return cache["response"]


def distribution(tags):
    result = {}
    for dimension in DIMENSIONS:
        values = [{**t["coarse_scores"], **t["fine_scores"]}[dimension] for t in tags]
        count = len(values)
        result[dimension] = {"count": count, "min": min(values) if values else None,
            "max": max(values) if values else None, "mean": sum(values) / count if count else None,
            "unique_values": len(set(values)), "zeros": values.count(0), "ones": values.count(1),
            "histogram_10_equal_width_bins": [sum(min(int(v * 10), 9) == i for v in values) for i in range(10)],
            "constant": bool(values) and len(set(values)) == 1,
            "all_zero": bool(values) and all(v == 0 for v in values),
            "all_one": bool(values) and all(v == 1 for v in values)}
    return result


def ledger_audit(raw, expected):
    """Every event remains physical/cache distinct; HTTP success is not a tag."""
    events, issues, by_fact = [], [], {}
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") != STAGE:
            continue
        context = row.get("context", {})
        ci, fid = context.get("conversation"), context.get("fact_id")
        key = (ci, fid)
        good = (type(ci) is int and isinstance(fid, str) and key in expected
                and context.get("method") == "shared_session" and context.get("phase") == "pilot_tag"
                and row.get("provider") == "jev" and row.get("model") == "jev-1.13.0"
                and row.get("operation") == "systemone" and row.get("request_hash") == expected.get(key))
        event = {"line": number, "conversation": ci, "fact_id": fid, "request_hash": row.get("request_hash"),
                 "cache_hit": row.get("cache_hit") is True, "status": row.get("status"),
                 "attempt": row.get("attempt"), "timestamp_utc": row.get("timestamp_utc"), "binding_valid": good}
        events.append(event)
        if not good:
            issues.append({"line": number, "error": "Unscoped or mismatched fact typing event"})
        else:
            by_fact.setdefault(key, []).append(event)
    return events, issues, by_fact


def audit_tags(run_dir):
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run_manifest.json").is_file():
        raise ValueError("Existing run manifest required")
    manifest = driver.ensure_manifest(run_dir, create=False)
    ledger_path = run_dir / "api_calls.jsonl"
    ledger = driver.shared_read(ledger_path)
    conversations, expected, all_tags, issues = [], {}, [], []
    for ci in range(10):
        folder = driver.conv_dir(run_dir, ci)
        result = {"conversation": ci, "facts": [], "errors": []}
        conversations.append(result)
        try:
            # The archived original raw-turn file contains no QA/gold. load_bank
            # verifies its source hash against the sealed stage binding/manifest.
            archive_path = folder / "banks/session/raw_source.json"
            raw = driver.read(archive_path)["raw_turns"]
            bank = driver.load_bank(run_dir, ci, "session", raw, manifest)
            result.update(bank_sha256=bank["sha256"], raw_source_sha256=driver.file_sha(archive_path),
                          expected_facts=len(bank["facts"]))
        except Exception as exc:
            result["errors"].append("Session bank: " + str(exc))
            continue
        try:
            summary_tags, summary_sha = driver.load_tags(run_dir, ci, raw, manifest, bank)
            validate_tags(bank["facts"], summary_tags)
            summary = driver.read(folder / "tag_summary.json")
            if (summary.get("fact_count") != len(bank["facts"])
                    or summary.get("bank_sha256") != bank["sha256"]
                    or summary.get("binding") != driver.tag_binding(manifest, ci, raw, bank)):
                raise ValueError("Tag summary count/bank/explicit binding differs")
            result["tag_summary_sha256"] = summary_sha
        except Exception as exc:
            summary_tags = None
            result["errors"].append("Tag summary: " + str(exc))
        expected_paths, valid_tags = set(), []
        for fact in bank["facts"]:
            fid = fact["id"]
            path = folder / "tags" / (driver.fingerprint(fid) + ".json")
            expected_paths.add(path.name)
            entry = {"fact_id": fid, "fact_sha256": driver.fingerprint(fact),
                     "source_context_sha256": driver.fingerprint(fact.get("source_records")),
                     "artifact_status": "missing", "valid": False, "errors": []}
            result["facts"].append(entry)
            saved = None
            try:
                bound = {**driver.tag_binding(manifest, ci, raw, bank), "fact_sha256": driver.fingerprint(fact), "fact_id": fid}
                row = driver._verified(path, driver.ARTIFACT_SCHEMA, bound)
                if row.get("binding") != bound:
                    raise ValueError("Tag artifact explicit binding differs")
                entry["artifact_status"] = row["status"]
                saved = driver.require(row)["tag"]
                validate_tags([fact], {fid: saved})
                entry["artifact_sha256"] = driver.file_sha(path)
            except Exception as exc:
                entry["errors"].append("Tag artifact: " + str(exc))
            replay = CacheOnlyJudge(run_dir / "api_cache")
            try:
                replayed = type_one_fact(fact, replay)
                if saved != replayed:
                    raise ValueError("Tag differs from frozen raw-response replay, context SHA or metadata")
                if summary_tags is None or summary_tags.get(fid) != saved:
                    raise ValueError("Per-fact tag and complete summary differ")
                entry["valid"] = True
                valid_tags.append(saved)
            except Exception as exc:
                entry["errors"].append("Raw response replay: " + str(exc))
            if replay.proof:
                entry.update(replay.proof)
                expected[(ci, fid)] = replay.proof["request_hash"]
        extras = sorted(p.name for p in (folder / "tags").glob("*.json") if p.name not in expected_paths)
        if extras:
            result["errors"].append("Unexpected tag artifact filenames: " + str(extras))
        result["artifact_counts"] = dict(Counter(f["artifact_status"] for f in result["facts"]))
        result["distribution"] = distribution(valid_tags)
        all_tags.extend(valid_tags)
    events, ledger_issues, by_fact = ledger_audit(ledger, expected)
    issues.extend(ledger_issues)
    for conv in conversations:
        for fact in conv["facts"]:
            matched = by_fact.get((conv["conversation"], fact["fact_id"]), [])
            fact["cache_events"] = sum(e["cache_hit"] for e in matched)
            fact["physical_events"] = sum(not e["cache_hit"] for e in matched)
            evidence = any(e["cache_hit"] or (type(e["status"]) is int and 200 <= e["status"] < 300) for e in matched)
            if fact["valid"] and not evidence:
                fact["valid"] = False
                fact["errors"].append("No attributed successful HTTP/cache event for this fact")
        conv["valid_tags"] = sum(f["valid"] for f in conv["facts"])
        conv["complete"] = not conv["errors"] and conv["valid_tags"] == conv.get("expected_facts")
    stable = driver.shared_read(ledger_path) == ledger
    if not stable:
        issues.append({"error": "Ledger changed during audit; rerun offline after stage completion"})
    return {"schema": SCHEMA, "snapshot_utc": driver.now(), "run_dir": str(run_dir),
        "protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": driver.file_sha(run_dir / "run_manifest.json"),
        "source_basis": "Sealed original raw_source archives validated by frozen load_bank; no dataset QA/gold loaded",
        "all_ten_session_banks_and_tags_valid": all(c["complete"] for c in conversations) and not issues,
        "conversations": conversations, "distribution": distribution(all_tags), "issues": issues,
        "ledger": {"sha256": sha(ledger), "bytes": len(ledger), "stable_during_audit": stable,
                   "events": events, "cache_events": sum(e["cache_hit"] for e in events),
                   "physical_events": sum(not e["cache_hit"] for e in events)},
        "interpretation": "HTTP success alone is insufficient. Distributions describe replay-valid tag artifacts; constant/all-zero/all-one flags are diagnostic, not threshold changes or calibration. Proposed fact support remains unverified."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output == run_dir or run_dir in output.parents or output in run_dir.parents:
        raise ValueError("Output must not overlap the run directory")
    report = audit_tags(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "audit.json"), "valid": report["all_ten_session_banks_and_tags_valid"]}))
    return 0 if report["all_ten_session_banks_and_tags_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
