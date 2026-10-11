"""Offline terminal QA seal, routing and raw-response audit; never creates API clients.

This is an additional reporting gate, not a change to any frozen experiment.
All 1200 terminal artifacts are required. Declared failures retain no scores.
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
from tools.summarize_pilot400_reviewed_v1 import validate_bank_failure_review
from jev_mem import evaluation_pilot as evaluation
from jev_mem.providers import ModelAPI

SCHEMA = "jev-mem-pilot400-qa-integrity-audit-v1"
STAGES = {evaluation.READER_STAGE, evaluation.JUDGE_STAGE}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CacheOnlyAPI:
    """Run the frozen provider decoder using this file-only request method."""
    generate = ModelAPI.generate

    def __init__(self, cache_dir):
        self.cache_dir, self.proofs = Path(cache_dir), []

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
        allowed = {evaluation.READER_STAGE: evaluation.READER_MODEL,
                   evaluation.JUDGE_STAGE: evaluation.JUDGE_MODEL}
        if provider != "gemini" or operation != "generateContent" or allowed.get(stage) != model or cache is not True:
            raise ValueError("Unexpected request in offline QA replay")
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        digest = sha(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode())
        path = self.cache_dir / (digest + ".json")
        raw = driver.shared_read(path)
        row = json.loads(raw)
        if row.get("request") != signature or row.get("stage") != stage:
            raise ValueError("QA HTTP cache signature or stage differs")
        response = row["response"]
        self.proofs.append({"stage": stage, "model": model, "request_hash": digest,
                            "cache_sha256": sha(raw), "response_sha256": driver.fingerprint(response)})
        return response


def verified_qa(path, q, arm, raw, manifest, bank, tags_sha):
    """Bind successful and failed artifacts before any metric code sees them."""
    ci = q["conversation"]
    query_bound = driver.binding(manifest, ci, raw, query_id=q["id"], question_sha256=driver.fingerprint(q["question"]))
    route_bound = {**query_bound, "arm": arm, "bank_sha256": bank["sha256"] if bank else None,
                   "tag_summary_sha256": tags_sha if arm == driver.ARMS[1] else None}
    bound = {**route_bound, "reference_sha256": driver.fingerprint(q["reference"])}
    row = driver._verified(path, driver.QA_SCHEMA, bound)
    identity = {"schema": driver.QA_SCHEMA, "arm": arm, "query_id": q["id"], "conversation": ci,
                "category": q["category"], "cohort_sha256": driver.COHORT_SHA256,
                "protocol_sha256": manifest["protocol_sha256"], "binding": bound,
                "bank_path": f"conversations/conv_{ci}/banks/{'regroup' if arm == driver.ARMS[2] else 'session'}/group_fact_bank.json",
                "bank_sha256": bank["sha256"] if bank else None}
    for key, value in identity.items():
        if type(row.get(key)) is not type(value) or row.get(key) != value:
            raise ValueError("QA identity/binding differs: " + key)
    if row["status"] == "failed":
        error = row.get("error")
        if (row.get("result") is not None or row.get("selected_ids") != [] or not isinstance(error, dict)
                or any(not isinstance(error.get(k), str) or not error[k] for k in ("type", "stage", "message"))):
            raise ValueError("Failed QA must retain its error and no answer or selected facts")
    elif row.get("error") is not None or bank is None:
        raise ValueError("Successful QA has an error or no complete bank")
    return row, query_bound, route_bound


def replay_complete(run_dir, row, q, bank, tags, raw, query_bound, route_bound):
    """Recompute frozen routing and exactly replay both generation responses."""
    folder = driver.conv_dir(run_dir, q["conversation"])
    driver.validate_qa_cache(row, q, bank, tags, row["arm"], raw, folder, query_bound, route_bound, run_dir)
    retrieval_path = folder / "retrieval" / row["arm"] / (q["id"] + ".json")
    retrieval = driver.require(driver._verified(retrieval_path, driver.ARTIFACT_SCHEMA, route_bound))
    cache_api = CacheOnlyAPI(Path(run_dir) / "api_cache")
    result = evaluation.answer_and_judge(cache_api, q["question"], q["reference"], retrieval["selected"],
                                         reference_date=max(t["observation_date"] for t in raw))
    if result != row["result"]:
        raise ValueError("Complete QA result differs from exact raw reader/judge replay")
    if [p["stage"] for p in cache_api.proofs] != [evaluation.READER_STAGE, evaluation.JUDGE_STAGE]:
        raise ValueError("Expected exactly one reader and one judge logical replay")
    return cache_api.proofs


def ledger_index(raw, expected_keys):
    events, by_key = [], {}
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        stage = row.get("stage")
        if stage not in STAGES:
            continue
        context = row.get("context", {})
        ci, arm, qid = context.get("conversation"), context.get("method"), context.get("query_id")
        key = (ci, arm, qid)
        expected_model = evaluation.READER_MODEL if stage == evaluation.READER_STAGE else evaluation.JUDGE_MODEL
        cache = row.get("cache_hit")
        if (type(ci) is not int or not isinstance(arm, str) or not isinstance(qid, str) or key not in expected_keys
                or context.get("architecture") != "pilot400" or context.get("driver_stage") != "qa"
                or context.get("phase") != "pilot_qa" or row.get("provider") != "gemini"
                or row.get("model") != expected_model or row.get("operation") != "generateContent"
                or not isinstance(row.get("request_hash"), str) or len(row["request_hash"]) != 64
                or cache is not None and type(cache) is not bool):
            raise ValueError("Unscoped or mismatched reader/judge event at line " + str(number))
        event = {"line": number, "stage": stage, "request_hash": row["request_hash"], "cache_hit": cache is True,
                 "status": row.get("status"), "attempt": row.get("attempt"), "usage_unknown": row.get("usage_unknown"),
                 "timestamp_utc": row.get("timestamp_utc"), "conversation": ci, "arm": arm, "query_id": qid}
        events.append(event)
        by_key.setdefault(key, []).append(event)
    return events, by_key


def validate_complete_events(proofs, events):
    expected = {p["stage"]: p["request_hash"] for p in proofs}
    for event in events:
        if expected.get(event["stage"]) != event["request_hash"]:
            raise ValueError("Complete QA has an attributed event for a different raw request")
    for stage, digest in expected.items():
        matched = [e for e in events if e["stage"] == stage and e["request_hash"] == digest]
        if not any(e["cache_hit"] or type(e["status"]) is int and 200 <= e["status"] < 300 for e in matched):
            raise ValueError("Complete QA lacks an attributed successful HTTP/cache event for " + stage)


def validate_terminal_status(path, ci, questions, manifest, artifacts):
    row = driver.read(path)
    expected_ids = [q["id"] for q in questions]
    counts = {arm: {"requested": len(questions), "completed": 0, "failed": 0} for arm in driver.ARMS}
    errors = []
    for artifact in artifacts:
        counts[artifact["arm"]]["completed" if artifact["status"] == "complete" else "failed"] += 1
        if artifact["status"] == "failed":
            errors.append({"query_id": artifact["query_id"], "arm": artifact["arm"], "error": artifact["error"]})
    expected = {"schema": "jev-mem-pilot400-conversation-qa-v2", "conversation": ci,
                "cohort_sha256": driver.COHORT_SHA256, "protocol_sha256": manifest["protocol_sha256"],
                "requested_query_ids": expected_ids, "methods": counts, "status": "failed" if errors else "complete"}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Terminal QA status counts/identity differ from sealed artifacts")
    order = lambda values: sorted(values, key=lambda v: (v["query_id"], v["arm"]))
    if order(row.get("errors", [])) != order(errors):
        raise ValueError("Terminal QA status reports errors not represented by artifacts; post-save validation may have failed")
    return counts


def audit_qa(run_dir, review_path):
    run_dir = Path(run_dir).resolve()
    manifest = driver.ensure_manifest(run_dir, create=False)
    admission = validate_bank_failure_review(run_dir, review_path, require_failed_qa=True)
    failed_banks = {(v["conversation"], v["bank"]) for v in admission["failed_banks"]}
    data = driver.load_data()
    _, questions = driver.load_cohort(driver.COHORT_PATH, data)
    if len(questions) != 400 or len({q["id"] for q in questions}) != 400:
        raise ValueError("Expected the frozen 400 unique paired questions")
    expected_keys = {(q["conversation"], arm, q["id"]) for q in questions for arm in driver.ARMS}
    ledger_path = run_dir / "api_calls.jsonl"
    ledger_raw = driver.shared_read(ledger_path)
    events, by_key = ledger_index(ledger_raw, expected_keys)
    reports, conversations, observed_paths = [], [], set()
    for ci in range(10):
        raw = driver.raw_turns_from_locomo(data[ci]["conversation"])
        banks = {name: None if (ci, name) in failed_banks else driver.load_bank(run_dir, ci, name, raw, manifest)
                 for name in ("session", "regroup")}
        tags, tags_sha = driver.load_tags(run_dir, ci, raw, manifest, banks["session"]) if banks["session"] else (None, None)
        qs = [q for q in questions if q["conversation"] == ci]
        artifacts = []
        for q in qs:
            for arm in driver.ARMS:
                path = driver.conv_dir(run_dir, ci) / "qa" / arm / (q["id"] + ".json")
                observed_paths.add(path.relative_to(run_dir).as_posix())
                bank = banks["regroup" if arm == driver.ARMS[2] else "session"]
                row, query_bound, route_bound = verified_qa(path, q, arm, raw, manifest, bank, tags_sha)
                artifacts.append(row)
                matched = by_key.get((ci, arm, q["id"]), [])
                item = {"conversation": ci, "arm": arm, "query_id": q["id"], "status": row["status"],
                        "artifact_sha256": driver.file_sha(path), "error": row.get("error"),
                        "reader_judge_ledger_lines": [e["line"] for e in matched]}
                if row["status"] == "complete":
                    proofs = replay_complete(run_dir, row, q, bank, tags, raw, query_bound, route_bound)
                    validate_complete_events(proofs, matched)
                    item["raw_response_replays"] = proofs
                else:
                    item["failure_scope"] = "Sealed declared failure; no scored result. This audit does not infer unknown network outcomes or reclassify the cause."
                    if bank is None and matched:
                        raise ValueError("Admitted failed bank has downstream reader/judge calls")
                reports.append(item)
        counts = validate_terminal_status(driver.conv_dir(run_dir, ci) / "qa_status.json", ci, qs, manifest, artifacts)
        conversations.append({"conversation": ci, "methods": counts})
    actual_paths = {p.relative_to(run_dir).as_posix() for p in run_dir.glob("conversations/conv_*/qa/*/*.json")}
    if actual_paths != observed_paths or len(reports) != 1200:
        raise ValueError("Exactly 1200 expected terminal QA artifact paths are required")
    if driver.shared_read(ledger_path) != ledger_raw:
        raise ValueError("Ledger changed during final audit; wait for terminal execution")
    return {"schema": SCHEMA, "snapshot_utc": driver.now(), "run_dir": str(run_dir), "valid": True,
            "manifest_sha256": driver.file_sha(run_dir / "run_manifest.json"), "protocol_sha256": manifest["protocol_sha256"],
            "cohort_sha256": driver.COHORT_SHA256, "cohort_per_arm": 400,
            "admission_review_sha256": admission["review_sha256"], "admitted_failed_arm_questions": admission["affected_arm_questions"],
            "terminal_artifacts": len(reports), "status_counts": dict(Counter(r["status"] for r in reports)),
            "conversations": conversations, "artifacts": reports,
            "ledger": {"sha256": sha(ledger_raw), "bytes": len(ledger_raw), "reader_judge_events": events,
                       "cache_events": sum(e["cache_hit"] for e in events), "physical_events": sum(not e["cache_hit"] for e in events)},
            "scope": "No client construction or API calls. All QA seals and exact question/reference/bank/tag/protocol bindings; complete QA frozen retrieval and raw reader/judge replay; terminal status reconciliation. Failures have no observed score. Costs are audited separately, and unknown attempt usage is not zero."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("Output must not overlap the run directory")
    report = audit_qa(run, args.review)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "audit.json"), "valid": report["valid"], "status_counts": report["status_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
