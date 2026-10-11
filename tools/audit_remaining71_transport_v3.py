"""Independently seal and replay the sole separately authorized TLS recovery."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_remaining71_transport_v3 as runner
from tools.audit_remaining71_qa_v2 import SavedOnlyAPI
from tools import pilot400_recovery_qa_v1 as qa
from tools.run_pilot400_experiments_v2 import read, file_sha, publish_json, now


def audit():
    run = runner.OUT
    record, retrieval, expected_plan = runner.prepare()  # Pure, verifies same exact original payload.
    plan = read(run / "plan.json")
    if plan != expected_plan:
        raise ValueError("v3 policy/code/selection plan differs from reviewed original request")
    path = run / "conversations/conv_1/qa/regroup_untyped_rank/c1_q43.json"
    files = list(run.glob("conversations/conv_*/qa/*/*.json"))
    if files != [path]:
        raise ValueError("Only the one reviewed transport question is eligible")
    row = qa._verify_sealed(read(path))
    status = read(run / "status.json")
    if (status["status"] != "complete" or status["completed"] != 1 or status["failed"] != 0
            or status["qa_sha256"] != file_sha(path) or row["status"] != "complete"
            or row["query_id"] != "c1_q43" or row["binding"] != plan["binding"]
            or row["technical_recovery_plan_sha256"] != file_sha(run / "plan.json")
            or row["selected_ids"] != [f["id"] for f in retrieval["selected"]]
            or len(row["selected_ids"]) != 20 or len(set(row["selected_ids"])) != 20):
        raise ValueError("v3 outcome and exact original20 identity/binding differs")
    api = SavedOnlyAPI(run)
    result = qa.evaluate_new(record["q"], retrieval["selected"], plan["binding"]["reference_date"], api)
    if row["result"] != result or row["error"] is not None or len(row["api_events"]) != 2:
        raise ValueError("v3 output differs from exact raw HTTP replay")
    ledger = [json.loads(line) for line in (run / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(ledger) != 2 or len(api.events) != 2:
        raise ValueError("Expected exactly one physical Reader and one physical Judge")
    for stored, checked, http in zip(row["api_events"], api.events, ledger):
        if (any(stored.get(k) != v for k, v in checked.items()) or stored["source"] != "new_response"
                or stored["new_physical_attempt"] is not True or http.get("cache_hit")
                or http.get("status") != 200 or http.get("attempt") != 1
                or http.get("request_hash") != checked["request_hash"] or http.get("stage") != checked["stage"]
                or http["context"].get("query_id") != "c1_q43" or http["context"].get("conversation") != 1
                or http["context"].get("method") != "regroup_untyped_rank"
                or http["context"].get("recovery_version") != 3):
            raise ValueError("v3 raw response or exact consumer-ledger context differs")
    if api.events[0]["request_hash"] != plan["original_request_hash"]:
        raise ValueError("Recovered Reader did not reuse original exact request")
    old_seal_path = ROOT / "research/pilot400_remaining71_v2_terminal_seal.json"
    old_seal = read(old_seal_path)
    for item in old_seal["files"]:
        if file_sha(Path(old_seal["run_dir"]) / item["path"]) != item["sha256"]:
            raise ValueError("The original v2 terminal failure/run was changed")
    entries = [{"path": p.relative_to(run).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)}
               for p in sorted(run.rglob("*")) if p.is_file()]
    seal_path = ROOT / "research/pilot400_remaining71_transport_v3_terminal_seal.json"
    if seal_path.exists():
        raise ValueError("v3 terminal seal already exists")
    publish_json(seal_path, {"schema": "remaining71-transport-v3-terminal-seal", "run_dir": str(run),
        "created_utc": now(), "completed": 1, "failed": 0, "query_id": "c1_q43", "files": entries})
    return {"schema": "remaining71-transport-v3-independent-http-audit", "status": "passed", "created_utc": now(),
        "source_code_sha256": file_sha(__file__), "policy_sha256": plan["policy_sha256"],
        "plan_sha256": file_sha(run / "plan.json"), "qa_sha256": file_sha(path),
        "status_sha256": file_sha(run / "status.json"), "ledger_sha256": file_sha(run / "api_calls.jsonl"),
        "terminal_seal_sha256": file_sha(seal_path), "sealed_files": len(entries),
        "original_v2_terminal_seal_sha256": file_sha(old_seal_path), "original_v2_unchanged": True,
        "query_id": "c1_q43", "completed": 1, "failed": 0, "original_selected20_unchanged": True,
        "identical_original_reader_request": True, "reader_request_hash": api.events[0]["request_hash"],
        "judge_request_hash": api.events[1]["request_hash"], "exact_http_replay": True,
        "physical_attempts": 2, "http200": 2, "cache_events": 0,
        "original_unknown_server_usage_retained": True, "model_calls_during_audit": 0, "credentials_loaded": False}


if __name__ == "__main__":
    out = ROOT / "research/pilot400_remaining71_transport_v3_independent_audit.json"
    if out.exists():
        raise ValueError("Independent audit report already exists")
    report = audit()
    publish_json(out, report)
    print({"status": report["status"], "questions": 1, "sealed_files": report["sealed_files"],
           "physical_attempts": 2, "report_sha256": file_sha(out)})
