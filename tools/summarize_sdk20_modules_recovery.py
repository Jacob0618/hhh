"""Offline, versioned first-valid recovery overlay; never modifies first-pass rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_sdk20_modules as original
from tools.summarize_mem0_sparse1540 import measure_usage


def sealed(path):
    value = original.read(path)
    checksum = value.get("content_sha256")
    if checksum != original.driver.digest({k: v for k, v in value.items() if k != "content_sha256"}):
        raise original.InvalidArtifact("Recovery content seal differs: " + str(path))
    return value


def checked_pointer(pointer):
    path = Path(pointer["path"]).resolve()
    if original.sha(path) != pointer["sha256"]:
        raise original.InvalidArtifact("Recovery pointer SHA differs: " + str(path))
    return path


def validate_sequence(records, stage):
    """First protocol-valid response is final even when its Judge value is false."""
    if not 1 <= len(records) <= 3:
        raise original.InvalidArtifact("Stage must contain one to three logical attempts")
    for number, record in enumerate(records, 1):
        if record.get("stage") != stage or record.get("logical_attempt") != number:
            raise original.InvalidArtifact("Stage logical attempts are not contiguous from one")
        if record.get("status") not in ("format_failed", "valid", "technical_failed"):
            raise original.InvalidArtifact("Unrecognized recovery attempt status")
        if number < len(records) and record["status"] != "format_failed":
            raise original.InvalidArtifact("Attempt continued after valid response or technical uncertainty")
    if records[-1]["status"] == "format_failed" and len(records) != 3:
        raise original.InvalidArtifact("Terminal format failure did not exhaust the declared uniform bound")
    return records[-1]["status"]


def merge_slots(questions, first_slots, overlays, official):
    """Replace every old failed slot exactly once; successful first-pass slots stay sealed."""
    target = {key for key, value in first_slots.items() if value["status"] == "failed"}
    if set(overlays) != target:
        raise original.InvalidArtifact("Recovery cohort differs from all and only original failures")
    by_query = {q["id"]: q for q in questions}
    result = deepcopy(first_slots)
    for key, overlay in overlays.items():
        arm, qid = key
        q = by_query[qid]
        if overlay["status"] not in ("complete", "failed"):
            raise original.InvalidArtifact("Recovery result must be terminal")
        if overlay["status"] == "failed":
            if overlay.get("result") is not None or not overlay.get("error"):
                raise original.InvalidArtifact("Unrecovered outcome must remain failed/null with its error")
            scores = None
        else:
            if overlay.get("result") is None or overlay.get("error") is not None:
                raise original.InvalidArtifact("Recovered outcome must have a result without error")
            scores = original.score_result(q, overlay["result"], official)
        result[key] = {**deepcopy(first_slots[key]), "status": overlay["status"], "scores": scores,
                       "error": overlay.get("error"), "recovery_artifact": overlay.get("artifact"),
                       "first_pass_status": "failed"}
    return result


class RecoveryReplay(original.ReplayAPI):
    """Recovery transport namespaces are global per stage and logical attempt."""
    def _scope_rows(self, key, stage):
        return [row for (h, st, _), rows in self.ledger.items() if h == key and st == stage for row in rows]


def read_transport(api_dir):
    path = Path(api_dir) / "api_calls.jsonl"
    if not path.exists():
        return [], defaultdict(list), None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ledger = defaultdict(list)
    for row in rows:
        if row.get("resolved_model") not in (None, row.get("model")):
            raise original.InvalidArtifact("Recovery resolved model differs from the fixed model")
        ledger[(row.get("request_hash"), row.get("stage"), original.fingerprint(row.get("context", {})))].append(row)
    return rows, ledger, original.pointer(path)


def audit_attempt(recovery, run, job_path, job, record_path, expected_signature, protocol,
                  reader_response, transport_cache):
    record = sealed(record_path)
    stage, number = record["stage"], record["logical_attempt"]
    expected_dir = (run / "transport" / stage / f"attempt_{number}").resolve()
    if (record.get("protocol_sha256") != protocol or record.get("job_sha256") != original.sha(job_path)
            or any(record.get(k) != job[k] for k in ("conversation", "method", "query_id"))
            or record.get("signature") != expected_signature
            or record.get("request_hash") != original.request_digest(expected_signature)
            or Path(record.get("api_dir", "")).resolve() != expected_dir):
        raise original.InvalidArtifact("Recovery attempt input binding differs")
    if expected_dir not in transport_cache:
        transport_cache[expected_dir] = read_transport(expected_dir)
    _, ledger, _ = transport_cache[expected_dir]
    events = record.get("api_events")
    if not isinstance(events, list):
        raise original.InvalidArtifact("Recovery attempt lacks an event list")
    raw_pointer = record.get("raw")
    key = record["request_hash"]
    raw_path = expected_dir / "api_cache" / f"{key}.json"
    intent_path = expected_dir / "request_intents" / f"{key}.json"
    if (expected_dir / "cache_lineage" / f"{key}.json").exists():
        raise original.InvalidArtifact("Recovery request imported a historical response")
    if not record.get("transport_intent") or checked_pointer(record["transport_intent"]) != intent_path:
        raise original.InvalidArtifact("Recovery dispatch intent is missing or outside its namespace")
    intent = original.read(intent_path)
    if (intent.get("signature") != expected_signature or intent.get("request_hash") != key
            or intent.get("stage") != stage or intent.get("retries") != 2):
        raise original.InvalidArtifact("Recovery intent changed the fixed request or network retry policy")
    if events:
        if len(events) != 1:
            raise original.InvalidArtifact("A recovery logical attempt must have one request event")
        context = events[0].get("context", {})
        expected_context = {"conversation": job["conversation"], "method": job["method"],
                            "query_id": job["query_id"], "phase": "sdk20_recovery",
                            "stage": stage, "logical_attempt": number}
        if context != expected_context:
            raise original.InvalidArtifact("Recovery event consumer context differs")
        api = RecoveryReplay(expected_dir, events, context, ledger)
        try:
            response = api.request(expected_signature["provider"], expected_signature["model"],
                                   expected_signature["body"], operation=expected_signature["operation"], stage=stage)
        except original.DeclaredTransportFailure as exc:
            if (record["status"] != "technical_failed" or raw_pointer is not None
                    or (record.get("error") or {}).get("type") != exc.provider_error_type):
                raise original.InvalidArtifact("Technical failure was reclassified or given a response")
            api.assert_consumed()
            return record, None
        api.assert_consumed()
        if raw_pointer is None or checked_pointer(raw_pointer) != raw_path:
            raise original.InvalidArtifact("Recovery raw response pointer differs")
    else:
        # A previous intent with no durable response is explicitly unresolved,
        # and must stop the consumer rather than acquire another logical round.
        if record["status"] != "technical_failed" or raw_pointer is not None or raw_path.exists():
            raise original.InvalidArtifact("No-event guard cannot claim a parsed response")
        error = record.get("error") or {}
        if error.get("type") != "RuntimeError" or "Prior unresolved logical request is retained; no implicit redispatch:" not in error.get("message", ""):
            raise original.InvalidArtifact("Unrecognized no-event failure requires separate review")
        return record, None
    try:
        derived = (recovery.probe_reader(job, response) if stage == "qa_reader"
                   else recovery.probe_pair(job, reader_response, response))
    except Exception as exc:
        expected_status = "format_failed" if recovery.fmt_error(exc) else "technical_failed"
        if record["status"] != expected_status or record.get("error") != recovery.err(exc):
            raise original.InvalidArtifact("Recovery parsing failure does not replay") from exc
        return record, None
    if record["status"] != "valid" or record.get("error") is not None:
        raise original.InvalidArtifact("A protocol-valid raw response was retried or marked failed")
    if stage == "qa_reader" and record.get("judge_signature") != derived:
        raise original.InvalidArtifact("Reader did not lock its own derived Judge request")
    if stage == "strict_judge" and record.get("result") != derived:
        raise original.InvalidArtifact("Judge result differs from the unchanged evaluator")
    return record, {"response": response, "derived": derived}


def summarize(run_dir, *, bootstrap_samples=2000):
    from tools import recover_sdk20_modules as recovery
    run = Path(run_dir).resolve()
    manifest = recovery.manifest(run)
    protocol = manifest["protocol_sha256"]
    source_path = checked_pointer(manifest["source_summary"])
    first = original.read(source_path)
    source_run = Path(first["run_dir"]).resolve()
    source_protocol = manifest["source_protocol_sha256"]
    source_manifest = original.read(checked_pointer(manifest["source_run_manifest"]))
    if source_manifest["protocol_sha256"] != source_protocol or first["integrity_errors"]:
        raise original.InvalidArtifact("Recovery source protocol is not the validated first pass")
    questions = [{**q, "conversation": ci} for ci, item in enumerate(original.load_data())
                 for q in original.queries(item, ci)]
    if len(questions) != 1540 or first["expected_slots"] != 4620:
        raise original.InvalidArtifact("Recovery summary requires the original fixed1540 cohort")
    by_query = {q["id"]: q for q in questions}
    official, provenance = original.verified_official_metric()
    first_slots, first_rows = {}, {}
    for slot in first["outcomes"]:
        key = slot["method"], slot["query_id"]
        if key in first_slots or key[0] not in original.ARMS or key[1] not in by_query:
            raise original.InvalidArtifact("Unexpected or duplicate original outcome")
        p = checked_pointer(slot["artifact"])
        expected = source_run / f'conv_{slot["conversation"]}' / "qa" / key[0] / (key[1] + ".json")
        if p != expected:
            raise original.InvalidArtifact("Original QA pointer escaped its fixed source run")
        row = original.verified(p, source_protocol)
        result = original.validate_row(row, by_query[key[1]], key[0], source_protocol)
        scores = original.score_result(by_query[key[1]], result, official) if result is not None else None
        if slot["status"] != row["status"] or slot["scores"] != scores or slot["error"] != row["error"]:
            raise original.InvalidArtifact("First-pass result or recomputed score differs from sealed summary")
        first_slots[key], first_rows[key] = slot, row
    expected_slots = {(a, q["id"]) for a in original.ARMS for q in questions}
    if set(first_slots) != expected_slots:
        raise original.InvalidArtifact("First-pass denominator does not contain exactly4620 outcomes")
    targets = {key for key, row in first_rows.items() if row["status"] == "failed"}
    if dict(Counter(arm for arm, _ in targets)) != recovery.EXPECTED_FAILURES:
        raise original.InvalidArtifact("Original recovery target count changed")
    jobs = {}
    for ptr in manifest["jobs"]:
        path = checked_pointer(ptr); job = sealed(path)
        key = job["method"], job["query_id"]
        if key in jobs or key not in targets or path != run / "jobs" / key[0] / (key[1] + ".json"):
            raise original.InvalidArtifact("Recovery job cohort or path differs")
        old_row = first_rows[key]
        if job["old_row"] != old_row or job["source_qa"] != first_slots[key]["artifact"]:
            raise original.InvalidArtifact("Job did not inherit its exact original failed row")
        if job["source_protocol_sha256"] != source_protocol:
            raise original.InvalidArtifact("Job source protocol differs")
        for field in ("conversation", "method", "query_id", "category", "question", "reference", "reference_date", "selected_ids"):
            if job[field] != old_row[field]:
                raise original.InvalidArtifact("Job input differs from its original failed row: " + field)
        selection = sealed(checked_pointer(job["source_selection"]))
        bank = original.read(checked_pointer(job["source_bank"]))
        if selection["selected_ids"] != job["selected_ids"] or selection["status"] != "complete":
            raise original.InvalidArtifact("Recovery changed the original selected20 facts")
        by_id = {f["id"]: f for f in bank["facts"]}
        facts = [{field: by_id[fid].get(field) for field in ("id", "text", "observation_date")}
                 for fid in job["selected_ids"]]
        if job["selected_facts"] != facts or len(facts) != 20:
            raise original.InvalidArtifact("Recovery changed original selected texts or dates")
        signature = recovery.expected_reader(job)
        if signature != job["expected_reader_signature"] or original.request_digest(signature) != job["expected_reader_hash"]:
            raise original.InvalidArtifact("Recovery Reader body does not exactly reconstruct")
        old_raw = original.read(checked_pointer(job["old_raw"]))
        if old_raw.get("request") != signature:
            raise original.InvalidArtifact("Fresh Reader body differs from the original failed request")
        try:
            recovery.probe_reader(job, old_raw["response"])
        except ValueError as exc:
            if type(exc) is not ValueError or str(exc) != "Reader did not return an answer string":
                raise original.InvalidArtifact("Original format failure did not replay") from exc
        else:
            raise original.InvalidArtifact("Original failed response unexpectedly parsed as valid")
        jobs[key] = path, job
    if set(jobs) != targets:
        raise original.InvalidArtifact("Recovery jobs omit an original failed consumer")

    overlays, audited, transport_cache = {}, [], {}
    expected_consumer_files = set()
    claimed_events = set()
    for key, (job_path, job) in jobs.items():
        folder = run / "consumers" / key[0] / key[1]
        final_path = folder / "result.json"
        final = sealed(final_path); expected_consumer_files.add(final_path)
        if (final.get("protocol_sha256") != protocol or final.get("job_sha256") != original.sha(job_path)
                or any(final.get(k) != job[k] for k in ("conversation", "method", "query_id", "selected_ids", "source_qa"))):
            raise original.InvalidArtifact("Recovery final result identity/source changed")
        stage_records = defaultdict(list)
        validated_details = []
        reader_response = None
        judge_signature = None
        seen_judge = False
        for ptr in final["attempt_records"]:
            path = checked_pointer(ptr); expected_consumer_files.add(path)
            raw_record = sealed(path)
            stage, number = raw_record["stage"], raw_record["logical_attempt"]
            if stage not in ("qa_reader", "strict_judge") or path != folder / stage / f"attempt_{number}.json":
                raise original.InvalidArtifact("Attempt record is outside its consumer/stage")
            if stage == "qa_reader" and seen_judge:
                raise original.InvalidArtifact("Reader was generated again after Judge began")
            if stage == "strict_judge":
                seen_judge = True
                if reader_response is None:
                    raise original.InvalidArtifact("Judge was attempted before Reader became valid")
            sig = job["expected_reader_signature"] if stage == "qa_reader" else judge_signature
            record, value = audit_attempt(recovery, run, job_path, job, path, sig, protocol,
                                          reader_response, transport_cache)
            stage_records[stage].append(record)
            validated_details.append({"artifact": ptr, "stage": stage, "logical_attempt": number,
                                      "status": record["status"], "error": record["error"]})
            for event in record["api_events"]:
                claimed_events.add((str(Path(record["api_dir"]).resolve()), record["request_hash"], stage,
                                    original.fingerprint(event["context"])))
            if stage == "qa_reader" and record["status"] == "valid":
                reader_response, judge_signature = value["response"], value["derived"]
        reader_state = validate_sequence(stage_records["qa_reader"], "qa_reader")
        if reader_state == "valid":
            lock_path = checked_pointer(final["reader_lock"])
            expected_consumer_files.add(lock_path)
            if lock_path != folder / "reader_lock.json":
                raise original.InvalidArtifact("Reader lock is outside its consumer")
            lock = sealed(lock_path); first_valid = stage_records["qa_reader"][-1]
            lock_identity = {"schema": recovery.SCHEMA, "protocol_sha256": protocol,
                             "job_sha256": original.sha(job_path), "conversation": job["conversation"],
                             "method": job["method"], "query_id": job["query_id"]}
            lock_expected = {**lock_identity, "logical_attempt": first_valid["logical_attempt"],
                             "attempt": original.pointer(folder / "qa_reader" / f'attempt_{first_valid["logical_attempt"]}.json'),
                             "raw": first_valid["raw"], "expected_reader_signature": job["expected_reader_signature"],
                             "judge_signature": judge_signature}
            if {k: v for k, v in lock.items() if k != "content_sha256"} != lock_expected:
                raise original.InvalidArtifact("Reader lock is not exactly its first valid response")
            judge_state = validate_sequence(stage_records["strict_judge"], "strict_judge")
            terminal = stage_records["strict_judge"][-1]
            if judge_state == "valid":
                raw_judge = original.read(checked_pointer(terminal["raw"]))["response"]
                expected_result = recovery.probe_pair(job, reader_response, raw_judge)
                if final["status"] != "complete" or final["result"] != expected_result or final["error"] is not None or final["status_reason"] != "first_valid_reader_and_judge":
                    raise original.InvalidArtifact("Recovery did not keep its first valid Reader/Judge pair")
            else:
                expected_reason = "judge_technical_failed" if judge_state == "technical_failed" else "judge_format_attempts_exhausted"
                if final["status"] != "failed" or final["result"] is not None or final["error"] != terminal["error"] or final["status_reason"] != expected_reason:
                    raise original.InvalidArtifact("Judge failure final outcome differs")
        else:
            terminal = stage_records["qa_reader"][-1]
            expected_reason = "reader_technical_failed" if reader_state == "technical_failed" else "reader_format_attempts_exhausted"
            if stage_records["strict_judge"] or final["reader_lock"] is not None:
                raise original.InvalidArtifact("Invalid Reader produced a lock or downstream Judge")
            if final["status"] != "failed" or final["result"] is not None or final["error"] != terminal["error"] or final["status_reason"] != expected_reason:
                raise original.InvalidArtifact("Reader failure final outcome differs")
        wrapped = {**first_rows[key], "status": final["status"], "result": final["result"], "error": final["error"]}
        original.validate_row(wrapped, by_query[key[1]], key[0], source_protocol)
        overlays[key] = {**final, "artifact": original.pointer(final_path)}
        audited.append({"method": key[0], "query_id": key[1], "status": final["status"],
                        "status_reason": final["status_reason"], "attempts": validated_details,
                        "artifact": original.pointer(final_path)})
    actual_consumer_files = {p.resolve() for p in (run / "consumers").glob("*/*/**/*.json") if p.name != "status.json"}
    if actual_consumer_files != expected_consumer_files:
        raise original.InvalidArtifact("Unlisted recovery attempt, lock or result file exists")
    if {p.resolve() for p in (run / "jobs").glob("*/*.json")} != {path for path, _ in jobs.values()}:
        raise original.InvalidArtifact("Unlisted recovery job exists")
    all_logs, log_pointers = [], []
    uncertain_guards = 0
    for api_dir in (run / "transport").glob("*/attempt_*"):
        api_dir = api_dir.resolve()
        rows, ledger, ptr = transport_cache.get(api_dir) or read_transport(api_dir)
        if ptr:
            log_pointers.append(ptr)
        for row in rows:
            claim = (str(api_dir), row.get("request_hash"), row.get("stage"), original.fingerprint(row.get("context", {})))
            if claim not in claimed_events:
                raise original.InvalidArtifact("Recovery ledger contains a request outside audited attempt events")
        all_logs.extend(rows)
    for details in audited:
        for attempt in details["attempts"]:
            row = sealed(attempt["artifact"]["path"])
            uncertain_guards += int(row["status"] == "technical_failed" and not row["api_events"])
    merged = merge_slots(questions, first_slots, overlays, official)
    arms, pairs = original.aggregate(questions, merged, bootstrap_samples)
    first_arms, first_pairs = original.aggregate(questions, first_slots, bootstrap_samples)
    usage = measure_usage(all_logs)
    if uncertain_guards:
        usage["total_cost_usd"] = None
    usage.update(source_ledgers=log_pointers, unresolved_no_event_consumers=uncertain_guards,
                 inherited_costs_recomputed=False, full_cold_start_total_cost_usd=None)
    # Finish with content checks over the prepared source and every material output.
    recovery.manifest(run)
    for detail in audited:
        checked_pointer(detail["artifact"])
        for attempt in detail["attempts"]:
            checked_pointer(attempt["artifact"])
    for ptr in log_pointers:
        checked_pointer(ptr)
    return {"schema": recovery.SCHEMA + "-summary-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run), "protocol_sha256": protocol, "source_summary": manifest["source_summary"],
            "cohort_per_arm": 1540, "expected_slots": 4620, "recovery_target_count": 55,
            "recovered_by_arm": {a: sum(v["status"] == "complete" for (arm, _), v in overlays.items() if arm == a) for a in original.ARMS},
            "residual_failures_by_arm": {a: sum(v["status"] == "failed" for (arm, _), v in overlays.items() if arm == a) for a in original.ARMS},
            "official_metric_provenance": provenance, "first_pass": {"arms": first_arms, "paired_differences": first_pairs},
            "arms": arms, "paired_differences": pairs, "outcomes": list(merged.values()),
            "recovery_audit": audited, "new_recovery_usage": usage, "integrity_errors": [], "model_calls": 0,
            "limitations": ["This is a separately versioned bounded format-recovery analysis; original first-pass results and failures remain unchanged.",
                            "Only original failures were retried, under a uniform three-attempt bound for Reader and Judge independently; successful wrong answers were never retried.",
                            "Residual failures remain null; complete-case, full-cohort and missing-zero sensitivity results remain distinct.",
                            "All conversations were previously exposed and only ten clusters are available; recovery does not create a blind test or establish general improvement.",
                            "Fresh recovery costs exclude all inherited original work; recorded cost estimates are not invoices and unknown usage is not zero."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Recovery summary output must be a new directory")
    report = summarize(args.run_dir, bootstrap_samples=args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# SDK20 统一有界补跑结果", "", "首遍指标保留；本表为统一补跑后的独立版本。分数按有效题计算，完整分母与缺失计零敏感性分析见 JSON。", "",
             "| 方法 | 完成/1540 | 本次补回 | 剩余失败 | 官方F1(%) | Judge(%) |", "|---|---:|---:|---:|---:|---:|"]
    for arm, values in report["arms"].items():
        metrics = values["completed_only_metrics"]
        lines.append(f'| {arm} | {values["status_counts"]["complete"]} | {report["recovered_by_arm"][arm]} | {values["status_counts"]["failed"]} | {metrics["official_locomo_f1"]*100:.3f} | {metrics["judge_correct"]*100:.3f} |')
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "recovered": report["recovered_by_arm"],
                      "remaining": report["residual_failures_by_arm"], "integrity_errors": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
