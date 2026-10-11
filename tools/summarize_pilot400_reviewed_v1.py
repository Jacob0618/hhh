"""Offline fixed400 summary with explicitly reviewed terminal writer failures.

This adds a reporting policy; it does not amend or bypass the frozen writer,
driver, old pipeline or old all-complete audit. Unreviewed failures fail closed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pilot400_experiments_v2 as driver
from tools import summarize_pilot400_v2 as base
from jev_mem import group_writer_v2 as writer
from jev_mem.evaluation_pilot import READER_STAGE, JUDGE_STAGE

REVIEW_SCHEMA = "jev-mem-pilot400-bank-failure-review-v1"
SUMMARY_SCHEMA = "jev-mem-pilot400-reviewed-summary-v1"
DISPOSITION = "terminal_failure_no_partial_bank_no_retry"
FILES = {"bank_summary_sha256": "bank_summary.json", "partial_bank_sha256": "group_fact_bank.json",
         "pending_sha256": "pending_group.json", "grouping_sha256": "grouping.json", "raw_source_sha256": "raw_source.json"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Review and artifact times require explicit timezone")
    return dt


def _journal_response(run_dir, journal, prompt, stage, retries):
    signature = writer.generation_signature(prompt)
    h = writer.provider_request_hash(signature)
    if (journal.get("request") != signature or journal.get("request_hash") != h
            or journal.get("stage") != stage or journal.get("max_logical_calls") != 1
            or journal.get("max_http_attempts_in_call") != retries + 1
            or journal.get("writer_protocol") != writer.SCHEMA
            or journal.get("input_payload_schema") != writer.INPUT_PAYLOAD_SCHEMA
            or journal.get("reextraction_policy") != writer.REEXTRACTION_POLICY):
        raise ValueError("Failed extraction request/protocol binding differs")
    cached = read(run_dir / "api_cache" / (h + ".json"))
    if cached.get("request") != signature or cached.get("response") != journal.get("http_response") or cached.get("stage") != stage:
        raise ValueError("Failed extraction HTTP cache differs from journal")
    decoded = writer.decode_http_response(journal["http_response"])
    if decoded != journal.get("decoded_response") or writer._hash(decoded) != journal.get("decoded_response_sha256"):
        raise ValueError("Failed extraction HTTP and decoded response differ")
    return decoded


def _strict_failure(run_dir, folder, raw, manifest, terminal):
    """Only the explicitly supported exhausted-feedback class is admitted."""
    p = read(folder / "pending_group.json")
    if p.get("status") != "terminal_feedback_validation_failure":
        raise ValueError("Unsupported failure class; unknown or other errors require separate review policy")
    if (p.get("protocol") != writer.SCHEMA or p.get("system_prompt") != writer.SYSTEM_PROMPT
            or set(p.get("extraction_attempts", {})) != {"first", "feedback"}
            or "embeddings" in p or p.get("error_type") != "GroupWriteError"):
        raise ValueError("Pending is not the supported strict terminal failure")
    prepared = writer.prepare_group(members=p["members"], raw_turns=raw,
                                    current_date=manifest["protocol"]["parameters"]["current_date"])
    if (p["prompt"] != json.dumps(prepared["payload"], ensure_ascii=False, indent=2)
            or p["source_sha256"] != driver.fingerprint(raw)
            or p["read_only_context"] != prepared["payload"]["READ_ONLY_CONTEXT"]
            or p["member_context_ids"] != prepared["payload"]["member_context_ids"]):
        raise ValueError("Failed pending does not bind original members/context/dates")
    fields = ("user_id", "group_id", "grouping_method", "source_sha256", "members", "read_only_context",
              "member_context_ids", "member_source_turn_ids", "context_source_turn_ids", "member_source_indices",
              "current_date", "protocol", "model", "input_payload_schema", "reextraction_policy",
              "feedback_instructions_sha256", "base_provider_retries", "feedback_provider_retries",
              "system_prompt_sha256", "embedding_dimensions")
    if writer._hash({k:p[k] for k in fields}) != p["request_sha256"]:
        raise ValueError("Failed group transaction fingerprint differs")
    first_journal, second_journal = p["extraction_attempts"]["first"], p["extraction_attempts"]["feedback"]
    first = _journal_response(run_dir, first_journal, p["prompt"], "group_fact_extraction", 2)
    try:
        writer.GroupFactWriter._parse(first, p, p["request_sha256"])
    except writer.GroupWriteError as error:
        reason = str(error)
    else:
        raise ValueError("First extraction did not actually fail strict parsing")
    if (p.get("first_raw_response") != first or p.get("first_validation_error") != reason
            or first_journal.get("validator_error") != reason or first_journal.get("validation") != "failed"):
        raise ValueError("First validation failure was not faithfully preserved")
    second = _journal_response(run_dir, second_journal, writer.feedback_prompt(p["prompt"], first, reason),
                               "group_fact_extraction_feedback_v2", 0)
    try:
        writer.GroupFactWriter._parse(second, p, p["request_sha256"])
    except writer.GroupWriteError as error:
        second_reason = str(error)
    else:
        raise ValueError("Feedback did not actually fail strict parsing")
    if (p.get("feedback_raw_response") != second or second_journal.get("validator_error") != second_reason
            or second_journal.get("validation") != "failed"
            or terminal.get("error") != {"type": "GroupWriteError", "stage": "write", "message": second_reason}):
        raise ValueError("Terminal error does not match the failed feedback parser")
    partial = read(folder / "group_fact_bank.json")
    if (partial.get("schema") != writer.SCHEMA or partial.get("source_sha256") != driver.fingerprint(raw)
            or partial.get("facts") != [f for g in partial["groups"] for f in g["fact_records"]]
            or len(partial["groups"]) != p["sequence"]):
        raise ValueError("Partial bank does not match the exhausted group sequence")
    for record in partial["groups"]:
        writer.validate_generation_provenance(record)
    return {"failure_class": "terminal_feedback_validation_failure", "validated_failure_kind": "exhausted_strict_feedback_validation", "group_id": p["group_id"],
            "first_request_hash": first_journal["request_hash"], "feedback_request_hash": second_journal["request_hash"],
            "partial_groups": len(partial["groups"]), "partial_facts": len(partial["facts"])}


def validate_bank_failure_review(run_dir, review_path, *, require_failed_qa=False):
    """Pure offline gate; pre-QA mode only requires the explicitly failed banks."""
    run_dir, review_path = Path(run_dir).resolve(), Path(review_path).resolve()
    manifest = driver.ensure_manifest(run_dir, create=False)
    review = read(review_path)
    expected = {"schema": REVIEW_SCHEMA, "status": "reviewed", "run_dir": str(run_dir),
                "protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": base.sha(run_dir / "run_manifest.json"),
                "source_dataset_sha256": driver.SOURCE_HASH, "cohort_sha256": driver.COHORT_SHA256}
    if any(review.get(k) != v for k,v in expected.items()):
        raise ValueError("Review identity/protocol/manifest/cohort differs")
    when = timestamp(review["reviewed_utc"])
    if when > datetime.now(timezone.utc) or not isinstance(review.get("reviewed_by"), str) or not review["reviewed_by"].strip():
        raise ValueError("Invalid review time or reviewer attribution")
    report_relative = Path(review["report_path"])
    report = (ROOT / report_relative).resolve()
    if report_relative.is_absolute() or not report.is_relative_to(ROOT) or base.sha(report) != review.get("report_sha256"):
        raise ValueError("Review report path/hash differs")
    entries = review.get("failed_banks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Failure admission requires explicit failed bank entries")
    data = driver.load_data()
    _, questions = driver.load_cohort(driver.COHORT_PATH, data)
    admitted, seen, affected = [], set(), set()
    for item in entries:
        ci, bank = item.get("conversation"), item.get("bank")
        if type(ci) is not int or ci not in range(10) or bank not in {"session", "regroup"} or (ci,bank) in seen:
            raise ValueError("Invalid or duplicate failed bank identity")
        seen.add((ci,bank))
        arms = list(driver.ARMS[:2]) if bank == "session" else [driver.ARMS[2]]
        qs = [q for q in questions if q["conversation"] == ci]
        if (item.get("affected_arms") != arms or item.get("affected_query_ids") != [q["id"] for q in qs]
                or item.get("disposition") != DISPOSITION
                or item.get("failure_class", "terminal_feedback_validation_failure") != "terminal_feedback_validation_failure"):
            raise ValueError("Failure admission changes the fixed arm/question cohort")
        folder = run_dir / "conversations" / f"conv_{ci}" / "banks" / bank
        for key,name in FILES.items():
            if item.get(key) != base.sha(folder / name):
                raise ValueError("Reviewed failed bank bytes changed: " + key)
        raw = driver.raw_turns_from_locomo(data[ci]["conversation"])
        bound = driver._bank_binding(manifest, ci, raw, bank)
        terminal = driver._verified(folder / "bank_summary.json", driver.ARTIFACT_SCHEMA, bound)
        if terminal["status"] != "failed" or terminal.get("error") != item.get("error") or timestamp(terminal["finished_utc"]) > when:
            raise ValueError("Review does not bind an existing terminal failed bank")
        grouping = driver.require(driver._verified(folder / "grouping.json", driver.ARTIFACT_SCHEMA, {**bound, "operation": "grouping"}))
        driver.verify_partition(raw, grouping["groups"])
        if read(folder / "raw_source.json") != {"user_id": "pilot400_conv_" + str(ci), "source_sha256": driver.fingerprint(raw), "raw_turns": raw}:
            raise ValueError("Failed bank raw source archive differs")
        detail = _strict_failure(run_dir, folder, raw, manifest, terminal)
        for q in qs:
            for arm in arms:
                affected.add((ci, arm, q["id"]))
                path = run_dir / "conversations" / f"conv_{ci}" / "qa" / arm / (q["id"] + ".json")
                if path.exists():
                    row = read(path)
                    query_bound = driver.binding(manifest, ci, raw, query_id=q["id"], question_sha256=driver.fingerprint(q["question"]))
                    binding = {**query_bound, "arm": arm, "bank_sha256": None, "tag_summary_sha256": None,
                               "reference_sha256": driver.fingerprint(q["reference"])}
                    row = driver._verified(path, driver.QA_SCHEMA, binding)
                    if (row.get("status") != "failed" or row.get("result") is not None or row.get("selected_ids") != []
                            or row.get("bank_sha256") is not None or row.get("query_id") != q["id"]
                            or row.get("conversation") != ci or row.get("category") != q["category"] or row.get("arm") != arm
                            or row.get("cohort_sha256") != driver.COHORT_SHA256
                            or row.get("error") != {"type": "RecordedFailure", "stage": "qa", "message": "write: " + item["error"]["message"]}
                            or timestamp(row["started_utc"]) < when):
                        raise ValueError("Admitted failed-bank QA was scored, selected facts, or differs from its pre-QA review")
                    if (run_dir / "conversations" / f"conv_{ci}" / "retrieval" / arm / (q["id"] + ".json")).exists():
                        raise ValueError("Failed-bank arm must not have retrieval output")
                elif require_failed_qa:
                    raise ValueError("Missing declared failed QA artifact for reviewed bank: " + q["id"])
        admitted.append({**item, **detail})
    ledger = run_dir / "api_calls.jsonl"
    if not ledger.is_file():
        raise ValueError("Missing physical ledger cannot establish no reader/judge calls")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") in {READER_STAGE, JUDGE_STAGE}:
            context = row.get("context", {})
            key = (context.get("conversation"), context.get("method"), context.get("query_id"))
            if None in key or key in affected:
                raise ValueError("Reader/judge event is unscoped or belongs to an admitted failed bank (including cache hits)")
            if timestamp(row["timestamp_utc"]) < when:
                raise ValueError("Failure admission review was made after pilot reader/judge work started")
    for path in run_dir.glob("conversations/conv_*/qa/*/*.json"):
        if timestamp(read(path)["started_utc"]) < when:
            raise ValueError("Failure admission review must predate every pilot QA artifact")
    return {"schema": REVIEW_SCHEMA, "review_path": str(review_path), "review_sha256": base.sha(review_path),
            "reviewed_utc": review["reviewed_utc"], "reviewed_by": review["reviewed_by"],
            "report_sha256": review["report_sha256"], "protocol_sha256": manifest["protocol_sha256"],
            "failed_banks": admitted, "affected_arm_questions": len(affected), "failed_qa_required": require_failed_qa}


def audit_reviewed_banks(run_dir, review_path):
    run_dir = Path(run_dir).resolve()
    admission = validate_bank_failure_review(run_dir, review_path, require_failed_qa=True)
    failures = {(x["conversation"],x["bank"]) for x in admission["failed_banks"]}
    manifest, data = driver.ensure_manifest(run_dir, create=False), driver.load_data()
    complete = []
    for ci, conv in enumerate(data):
        raw = driver.raw_turns_from_locomo(conv["conversation"])
        for name in ("session", "regroup"):
            if (ci,name) in failures:
                continue
            bank = driver.load_bank(run_dir, ci, name, raw, manifest)  # no catch/skip fallback
            provenance = []
            for group in read(run_dir / bank["path"])["groups"]:
                provenance.append(writer.validate_generation_provenance(group))
                for journal in group["extraction_attempts"].values():
                    cache = read(run_dir / "api_cache" / (journal["request_hash"] + ".json"))
                    if (cache.get("request") != journal["request"] or cache.get("response") != journal["http_response"]
                            or cache.get("stage") != journal["stage"]):
                        raise ValueError("Complete group HTTP cache binding differs")
            complete.append({"conversation": ci, "bank": name, **bank, "generation_provenance": provenance})
            complete[-1].pop("facts")
            complete[-1]["fact_count"] = len(bank["facts"])
    if len(complete) + len(failures) != 20:
        raise ValueError("Exactly twenty terminal bank outcomes must be accounted for")
    return {"review": admission, "complete_banks": complete, "terminal_banks": 20,
            "scope": "Complete banks strictly validated; only explicitly reviewed exhausted failures admitted. Partial banks never feed QA; semantic support remains unverified."}


def summarize_reviewed(run_dir, review_path, *, bootstrap_samples=2000):
    gate = audit_reviewed_banks(run_dir, review_path)
    metrics = base.v2_metrics()
    report = metrics.summarize_pilot(run_dir, bootstrap_samples=bootstrap_samples)
    report.update(schema=SUMMARY_SCHEMA, reviewed_bank_audit=gate,
                  current_namespace_usage=base.ledger_usage(Path(run_dir) / "api_calls.jsonl"),
                  cache_seed_provenance=base.seed_provenance(run_dir))
    report["interpretation"]["writer"] = "Frozen common writer v2. Explicitly reviewed bank failures remain failed in each arm's fixed400 denominator; this is a new reporting/admission policy, not the old all-complete gate passing."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--check-review", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.check_review:
        result = validate_bank_failure_review(args.run_dir, args.review)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required except with --check-review")
    report = summarize_reviewed(args.run_dir, args.review, bootstrap_samples=args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = base.v2_metrics().markdown_report(report) + "\nReviewed terminal bank failures are preserved; the original all-complete bank gate has not passed. See JSON for bindings and costs.\n"
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "summary.json"), "reviewed_failures": report["reviewed_bank_audit"]["review"]["affected_arm_questions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
