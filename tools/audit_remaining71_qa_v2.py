"""Pure replay of the fixed71 recovered QA outcomes using saved HTTP only."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pilot400_remaining71_v2 as driver
from tools import pilot400_recovery_qa_v1 as recovery
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now
from jev_mem.pilot400 import fingerprint


class SavedOnlyAPI(recovery.ResponseSchemaAPI):
    """No ModelAPI initialization, credentials, logs, filesystem writes or network."""
    def __init__(self, run):
        self.run = Path(run)
        self.events = []

    def generate(self, prompt, *, model, system, json_output, max_output, stage):
        schema = recovery.READER_SCHEMA if stage == recovery.evaluation.READER_STAGE else recovery.JUDGE_SCHEMA
        signature = recovery.generation_signature(prompt, model=model, system=system, json_output=json_output,
            max_output=max_output, schema=schema)
        digest = recovery.request_hash(signature)
        path = self.run / "api_cache" / (digest + ".json")
        intent_path = self.run / "qa_request_intents" / (digest + ".json")
        intent, cached = read(intent_path), read(path)
        if (intent["request"] != signature or intent["request_hash"] != digest or intent["policy"] != recovery.POLICY
                or cached["request"] != signature or cached["stage"] != stage):
            raise ValueError("Actual Reader/Judge request does not bind saved HTTP")
        self.events.append({"stage": stage, "request_hash": digest, "cache_sha256": file_sha(path),
            "response_sha256": fingerprint(cached["response"]), "cache_path": str(path)})
        return recovery.decode_response(cached["response"])


def audit():
    run = driver.DEFAULT_OUTPUT
    records = driver.collect(run)
    status = read(run / "qa_remaining71_status.json")
    plan = read(run / "qa_remaining71_plan.json")
    if (status.get("status") not in ("complete", "requires_review") or status.get("done") != 71
            or status.get("plan_sha256") != file_sha(run / "qa_remaining71_plan.json")):
        raise ValueError("QA is not a complete reviewed terminal71 batch")
    expected_paths = {driver._paths(run, r)[1].resolve() for r in records}
    found_paths = {p.resolve() for p in run.glob("conversations/conv_*/qa/*/*.json")}
    if found_paths != expected_paths:
        raise ValueError("New QA files differ from exact fixed71 C-only population")
    plan_entries, reports, physical_hashes = [], [], []
    ledger = [json.loads(line) for line in (run / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    qa_ledger = [row for row in ledger if row.get("context", {}).get("phase") == "pilot_remaining71_qa"]
    seal_path = ROOT / "research/pilot400_remaining71_v2_terminal_seal.json"
    seal = read(seal_path)
    for item in seal["files"]:
        if file_sha(run / item["path"]) != item["sha256"]:
            raise ValueError("A terminal-sealed v2 file changed")
    for record in records:
        retrieval_path, qa_path = driver._paths(run, record)
        retrieved = recovery._verify_sealed(read(retrieval_path))
        if retrieved != driver.retrieval_for(record):
            raise ValueError("Saved selection differs from deterministic new-bank rank20")
        plan_entries.append({"binding": record["binding"], "retrieval_sha256": file_sha(retrieval_path)})
        row = recovery._verify_sealed(read(qa_path))
        selected = retrieved["selected"]
        expected_binding = {**record["binding"], "retrieval_sha256": file_sha(retrieval_path),
                            "selected_sha256": fingerprint(selected)}
        q = record["q"]
        if (row["binding"] != expected_binding or row["selected_ids"] != [f["id"] for f in selected]
                or row["query_id"] != q["id"] or row["conversation"] != q["conversation"]
                or row["arm"] != driver.ARM or row["category"] != q["category"]
                or len(row["selected_ids"]) != 20 or len(set(row["selected_ids"])) != 20):
            raise ValueError("QA identity/selection/budget binding differs")
        api = SavedOnlyAPI(run)
        replayed = None
        error = None
        try:
            replayed = recovery.evaluate_new(q, selected, record["binding"]["reference_date"], api)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        if row["status"] == "complete":
            if error is not None or row["result"] != replayed or row.get("error") is not None:
                raise ValueError("Successful result differs from exact saved Reader/Judge HTTP replay")
            if len(row["api_events"]) != 2 or len(api.events) != 2:
                raise ValueError("Successful QA does not have one Reader and one Judge event")
            for saved, checked in zip(row["api_events"], api.events):
                if any(saved.get(k) != v for k, v in checked.items()):
                    raise ValueError("QA response event bytes/signature differ")
        elif row["status"] == "failed":
            if row.get("result") is not None or not isinstance(row.get("error"), dict):
                raise ValueError("Failed QA must retain null result and explicit error")
            if (q["id"] != "c1_q43" or len(row["api_events"]) != 1
                    or row["error"]["type"] != "RuntimeError" or "SSLEOFError" not in row["error"]["message"]):
                raise ValueError("Unreviewed failure type/question")
            event = row["api_events"][0]
            prepared = recovery.evaluation.prepare_reader_input(q["question"], selected,
                reference_date=record["binding"]["reference_date"])
            signature = recovery.generation_signature(recovery.evaluation._json(prepared["payload"]),
                model=recovery.evaluation.READER_MODEL, system=recovery.evaluation.READER_SYSTEM,
                json_output=True, max_output=recovery.evaluation.MAX_OUTPUT, schema=recovery.READER_SCHEMA)
            digest = recovery.request_hash(signature)
            intent_path = run / "qa_request_intents" / (digest + ".json")
            intent = read(intent_path)
            if (event["request_hash"] != digest or event["source"] != "failed_or_unknown_response"
                    or event["stage"] != recovery.evaluation.READER_STAGE or event["new_physical_attempt"] is not True
                    or intent["request"] != signature or intent["request_hash"] != digest
                    or (run / "api_cache" / (digest + ".json")).exists()):
                raise ValueError("Unknown Reader request intent/event differs")
            matching = [r for r in qa_ledger if r.get("context", {}).get("query_id") == q["id"]]
            if (len(matching) != 1 or matching[0].get("request_hash") != digest
                    or matching[0].get("status") != "transport_error" or matching[0].get("attempt") != 1
                    or matching[0].get("stage") != recovery.evaluation.READER_STAGE
                    or matching[0].get("usage_unknown") is not True
                    or matching[0].get("error") != row["error"]["message"]):
                raise ValueError("Unknown transport attempt lacks exact lone Reader ledger error")
        else:
            raise ValueError("Unknown QA terminal state")
        events = row["api_events"]
        for event in events:
            matching = [line for line in qa_ledger if line.get("request_hash") == event["request_hash"]
                and line.get("context", {}).get("query_id") == q["id"]
                and line.get("context", {}).get("conversation") == q["conversation"]
                and line.get("context", {}).get("method") == driver.ARM
                and line.get("stage") == event["stage"]]
            if not matching:
                raise ValueError("No exact question/context API ledger event")
            physical = [line for line in matching if not line.get("cache_hit")]
            if event["new_physical_attempt"] != bool(physical):
                raise ValueError("Physical-attempt attribution differs")
            physical_hashes.extend(line["request_hash"] for line in physical)
        reports.append({"query_id": q["id"], "conversation": q["conversation"], "status": row["status"],
            "qa_sha256": file_sha(qa_path), "retrieval_sha256": file_sha(retrieval_path), "selected_count": 20,
            "raw_excerpt_selected": retrieved["trace"]["raw_member_excerpt_selected"],
            "reader_request_hash": events[0]["request_hash"],
            "judge_request_hash": events[1]["request_hash"] if row["status"] == "complete" else None,
            "saved_http_exact_replay": row["status"] == "complete",
            "unknown_transport_failure_verified": row["status"] == "failed"})
    if plan.get("entries") != plan_entries or plan.get("expected") != 71 or plan.get("provider_retries") != 0:
        raise ValueError("Executed plan differs from complete frozen input set")
    counts = collections.Counter(physical_hashes)
    if any(count > 1 for count in counts.values()):
        raise ValueError("A request had more than one physical attempt")
    if (len(qa_ledger) != sum(len(read(driver._paths(run, r)[1])["api_events"]) for r in records)
            or status["completed"] != 70 or status["failed"] != 1
            or sum(r["status"] == "complete" for r in reports) != 70):
        raise ValueError("Final status/API population differs from 70 successes and one retained failure")
    return {"schema": "remaining71-new-qa-exact-http-audit-v2", "status": "passed", "created_utc": now(),
        "source_code_sha256": file_sha(__file__), "plan_sha256": file_sha(run / "qa_remaining71_plan.json"),
        "status_sha256": file_sha(run / "qa_remaining71_status.json"), "ledger_sha256": file_sha(run / "api_calls.jsonl"),
        "admission_sha256": file_sha(run / "bank_admission.json"), "questions": len(reports),
        "completed": 70, "failed": 1, "failed_query_id": "c1_q43", "failed_result_is_null": True,
        "terminal_seal_sha256": file_sha(seal_path), "physical_reader_judge_attempts": len(physical_hashes),
        "contextual_cache_events": len(qa_ledger) - len(physical_hashes), "entries": reports,
        "model_calls_during_audit": 0, "credentials_loaded": False,
        "limitation": "Request/result/score lineage proof; independent model judge is not human truth."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Audit output already exists")
    result = audit()
    publish_json(args.out, result)
    print({k: result[k] for k in ("status", "questions", "physical_reader_judge_attempts", "contextual_cache_events")})
