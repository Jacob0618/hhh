"""Read-only two-snapshot health evidence for the native Mem0 sparse1540 run.

Reads hot files with Windows READ|WRITE|DELETE sharing. Verifies ProcessPool
spawn children via their actual PID, creation time, parent chain and coordinator.
No ModelAPI imports, credential reads, model requests or run-directory writes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read, parse_rows
from tools.process_control_windows import WinProcess, split_command_line

RUN = ROOT / "runs/mem0_sparse1540_v1"
DRIVER = ROOT / "tools/run_mem0_sparse1540.py"
SCHEMA = "mem0-sparse1540-health-v1"
TERMINAL = {"complete", "failed", "requires_review", "blocked", "interrupted", "worker_error"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(shared_read(path).decode("utf-8-sig"))


def sha(path):
    return hashlib.sha256(shared_read(path)).hexdigest()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def inventory():
    command = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "ForEach-Object { [PSCustomObject]@{pid=[int]$_.ProcessId; parent_pid=[int]$_.ParentProcessId; "
        "created_utc=$_.CreationDate.ToUniversalTime().ToString('o'); command_line=$_.CommandLine; "
        "kernel_100ns=$_.KernelModeTime; user_100ns=$_.UserModeTime} } | ConvertTo-Json -Compress")
    proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True,
        text=True, encoding="utf-8", check=True, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    rows = json.loads(proc.stdout) if proc.stdout.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    return {int(row["pid"]): row for row in rows}


def process_identity(pid, processes):
    row = dict(processes.get(int(pid), {}))
    if not row:
        return {"pid": pid, "alive": False, "identity_verified": False}
    try:
        handle = WinProcess(pid)
        try:
            row.update(alive=handle.is_running(), creation_time_100ns=handle.creation_time_100ns)
        finally:
            handle.close()
        exact_created = row["creation_time_100ns"] / 10_000_000 - 11644473600
        row["cim_handle_creation_match"] = abs(timestamp(row["created_utc"]) - exact_created) < 1
        row["cpu_seconds"] = (int(row.get("kernel_100ns") or 0) + int(row.get("user_100ns") or 0)) / 10_000_000
    except (OSError, ValueError, TypeError) as error:
        row.update(alive=False, identity_error=type(error).__name__, cim_handle_creation_match=False)
    return row


def coordinator_identity(state, processes, stage):
    if not state or "pid" not in state:
        return {"alive": False, "identity_verified": False, "not_started": True}
    row = process_identity(state["pid"], processes)
    if not row.get("alive"):
        return row
    args = split_command_line(row.get("command_line", ""))
    positions = [i for i, arg in enumerate(args) if Path(arg).name.lower() == DRIVER.name.lower()]
    command_ok = len(positions) == 1 and positions[0] + 1 < len(args) and args[positions[0] + 1] == stage
    if command_ok:
        supplied = Path(args[positions[0]])
        command_ok = (supplied if supplied.is_absolute() else ROOT / supplied).resolve() == DRIVER.resolve()
    created = row["creation_time_100ns"] / 10_000_000 - 11644473600
    creation_ok = -1 <= timestamp(state["started_utc"]) - created <= 180
    row.update(command_verified=command_ok, creation_matches_launch=creation_ok,
               identity_verified=bool(command_ok and creation_ok and row.get("cim_handle_creation_match")))
    return row


def worker_identity(state, coordinator, processes):
    row = process_identity(state["pid"], processes)
    if not row.get("alive"):
        return row
    command = row.get("command_line", "")
    spawn = "multiprocessing.spawn" in command and "spawn_main" in command
    explicit_parent = re.search(r"parent_pid\s*=\s*(\d+)", command)
    expected_parent = coordinator.get("pid")
    chain, current = [], row.get("parent_pid")
    for _ in range(4):
        if current is None or current in chain:
            break
        chain.append(current)
        if current == expected_parent:
            break
        current = processes.get(current, {}).get("parent_pid")
    parent_ok = expected_parent in chain and (explicit_parent is None or int(explicit_parent[1]) == expected_parent)
    created = row["creation_time_100ns"] / 10_000_000 - 11644473600
    coordinator_created = coordinator.get("creation_time_100ns", 0) / 10_000_000 - 11644473600
    # A ProcessPool worker can start another conversation long after creation.
    creation_ok = coordinator_created - 1 <= created <= timestamp(state["started_utc"]) + 1
    row.update(spawn_command_verified=spawn, parent_chain=chain, coordinator_parent_verified=parent_ok,
        creation_matches_pool=creation_ok, identity_verified=bool(spawn and parent_ok and creation_ok
        and coordinator.get("identity_verified") and row.get("cim_handle_creation_match")))
    return row


def artifact_counts(folder, stage):
    if stage == "write":
        files = list((folder / "native/session_audit").glob("*/*.json"))
    elif stage == "retrieve":
        files = list((folder / "retrieval/regroup_rank").glob("*.json"))
    else:
        files = list((folder / "qa").glob("*/*.json"))
    statuses, by_arm, facts, malformed = Counter(), {}, 0, 0
    newest, newest_name, byte_count = 0, None, 0
    for path in files:
        stat = path.stat()
        byte_count += stat.st_size
        if stat.st_mtime_ns > newest:
            newest, newest_name = stat.st_mtime_ns, path.name
        try:
            row = read(path)
            statuses[row.get("status", "unknown")] += 1
            if stage == "write" and row.get("status") == "success":
                facts += len(row.get("fact_records", []))
            if stage == "qa":
                key = row.get("arm", path.parent.name)
                by_arm.setdefault(key, Counter())[row.get("status", "unknown")] += 1
        except (OSError, ValueError):
            malformed += 1
    return {"files": len(files), "statuses": dict(statuses), "by_arm": {k: dict(v) for k, v in by_arm.items()},
        "facts_in_successful_writes": facts, "malformed": malformed, "bytes": byte_count,
        "newest_mtime_ns": newest, "latest_artifact": newest_name}


def log_snapshot(path, stage, ci):
    if not path.exists():
        return {"path": str(path), "bytes": 0, "complete_bytes": 0, "last_api_utc": None, "tail_rows": 0}
    size = path.stat().st_size
    start = max(0, size - 131072)
    raw = shared_read(path, offset=start)[:size - start]
    complete = start + raw.rfind(b"\n") + 1 if b"\n" in raw else start
    rows = [r for r in parse_rows(raw) if r.get("context", {}).get("phase") == stage
            and r.get("context", {}).get("conversation") == ci]
    return {"path": str(path), "bytes": size, "complete_bytes": complete,
        "last_api_utc": rows[-1].get("timestamp_utc") if rows else None, "tail_rows": len(rows),
        "tail_stages": dict(Counter(r.get("stage") for r in rows)),
        "last_request_hash": rows[-1].get("request_hash") if rows else None,
        "last_query_id": rows[-1].get("context", {}).get("query_id") if rows else None,
        "last_group_id": rows[-1].get("context", {}).get("group_id") if rows else None}


def snapshot(stage, conversations=None):
    manifest_path = RUN / "manifest.json"
    manifest = read(manifest_path)
    problems = []
    for relative, expected in manifest["code_sha256"].items():
        if sha(ROOT / relative) != expected:
            problems.append("frozen_code_changed:" + relative)
    status_path = RUN / (stage + "_coordinator_status.json")
    coordinator_state = read(status_path) if status_path.exists() else None
    cis = conversations if conversations is not None else (coordinator_state or {}).get("conversations", list(range(10)))
    processes = inventory()
    coordinator = coordinator_identity(coordinator_state, processes, stage)
    if coordinator_state and coordinator_state["status"] == "running" and not coordinator.get("identity_verified"):
        problems.append("running_coordinator_identity_not_verified")
    if coordinator_state and coordinator_state.get("workers", 0) > 8:
        problems.append("coordinator_exceeds_eight_workers")
    output = {}
    for ci in cis:
        folder = RUN / ("conv_" + str(ci))
        path = folder / (stage + "_status.json")
        state = read(path) if path.exists() else {"status": "pending", "done": 0, "failed": 0}
        if state.get("status") == "running":
            worker = worker_identity(state, coordinator, processes)
            if not worker.get("identity_verified"):
                problems.append("conv_" + str(ci) + "_worker_identity_not_verified")
        else:
            worker = {"pid": state.get("pid"), "terminal_or_pending": True}
        log_path = folder / ("qa_api/api_calls.jsonl" if stage == "qa" else "native/api_calls.jsonl")
        output[str(ci)] = {"state": state, "worker": worker, "artifacts": artifact_counts(folder, stage),
                          "api_log": log_snapshot(log_path, stage, ci)}
    return {"schema": SCHEMA, "captured_utc": now(), "run_dir": str(RUN), "stage": stage,
        "manifest_sha256": sha(manifest_path), "coordinator_state": coordinator_state,
        "coordinator": coordinator, "conversations": output, "problems": problems,
        "model_calls": 0, "credentials_loaded": False}


def delta_logs(first, second, stage, ci):
    old, new = first["api_log"], second["api_log"]
    start, stop = old["complete_bytes"], new["complete_bytes"]
    if stop < start:
        return {"log_shrank": True, "bytes_added": new["bytes"] - old["bytes"], "physical_success": 0,
                "physical_errors": 0, "cache_hits": 0}
    rows = parse_rows(shared_read(new["path"], offset=start)[:stop - start]) if stop > start else []
    relevant = [r for r in rows if r.get("context", {}).get("phase") == stage
                and r.get("context", {}).get("conversation") == ci]
    physical = [r for r in relevant if not r.get("cache_hit")]
    success = [r for r in physical if r.get("status") == 200]
    errors = [r for r in physical if r.get("status") != 200]
    key = lambda r: str((r.get("context", {}).get("query_id"), r.get("context", {}).get("group_id"),
                          r.get("stage"), r.get("request_hash")))
    return {"bytes_added": new["bytes"] - old["bytes"], "complete_bytes_added": stop - start,
        "physical_success": len(success), "physical_errors": len(errors),
        "cache_hits": sum(bool(r.get("cache_hit")) for r in relevant), "unrelated_context_rows": len(rows) - len(relevant),
        "physical_statuses": dict(Counter(str(r.get("status")) for r in physical)),
        "successful_stages": dict(Counter(r.get("stage") for r in success)),
        "successful_context_hashes": dict(Counter(key(r) for r in success)),
        "error_context_hashes": dict(Counter(key(r) for r in errors)),
        "last_api_utc": relevant[-1].get("timestamp_utc") if relevant else new.get("last_api_utc")}


def productive_counts(row):
    return {"state_done": row["state"].get("done", 0), "state_completed": row["state"].get("completed", 0),
        "state_facts_added": row["state"].get("facts_added", 0), "artifact_files": row["artifacts"]["files"],
        "artifact_success": sum(row["artifacts"]["statuses"].get(k, 0) for k in ("success", "complete")),
        "artifact_facts": row["artifacts"]["facts_in_successful_writes"]}


def compare(first, second, *, historical=None, log_delta=None):
    problems = list(dict.fromkeys(first["problems"] + second["problems"]))
    if first["manifest_sha256"] != second["manifest_sha256"]:
        problems.append("manifest_changed")
    rows = []
    for ci, new in second["conversations"].items():
        old = first["conversations"][ci]
        delta = (log_delta or delta_logs)(old, new, second["stage"], int(ci))
        counts_old, counts_new = productive_counts(old), productive_counts(new)
        growth = {k: counts_new[k] - counts_old[k] for k in counts_new}
        status = new["state"]["status"]
        if status in TERMINAL:
            verdict = "terminal_requires_integrity_and_failure_review"
        elif status == "pending":
            verdict = "pending_not_started"
        elif not new["worker"].get("identity_verified"):
            verdict = "worker_identity_investigate"
        elif any(v > 0 for k, v in growth.items() if k != "artifact_files"):
            verdict = "new_persisted_progress_observed"
        elif delta["physical_errors"] and not delta["physical_success"]:
            verdict = "only_errors_without_product_investigate"
        elif delta["physical_success"]:
            verdict = "http_work_observed_product_not_yet_finished"
        else:
            verdict = "no_product_or_successful_http_progress_investigate"
        if historical and ci in historical.get("conversations", {}) and status == "running":
            previous = historical["conversations"][ci]
            age = timestamp(second["captured_utc"]) - timestamp(historical["captured_utc"])
            same_worker = all(previous["worker"].get(k) == new["worker"].get(k) for k in ("pid", "creation_time_100ns"))
            if age >= 600 and same_worker and productive_counts(previous) == counts_new:
                verdict = "ten_minutes_without_products_investigate_even_if_http200"
        if delta.get("log_shrank") or any(v < 0 for v in growth.values()) or new["artifacts"]["malformed"]:
            verdict = "artifact_or_ledger_integrity_investigate"
        rows.append({"conversation": int(ci), "status": status, "state_done": new["state"].get("done", 0),
            "state_expected": new["state"].get("expected"), "count_growth": growth, "api": delta,
            "last_worker_update": new["state"].get("updated_utc"), "latest_artifact": new["artifacts"]["latest_artifact"],
            "verdict": verdict})
    return {"schema": SCHEMA, "stage": second["stage"], "first_utc": first["captured_utc"],
        "second_utc": second["captured_utc"], "seconds": timestamp(second["captured_utc"]) - timestamp(first["captured_utc"]),
        "manifest_sha256": second["manifest_sha256"], "problems": problems, "conversations": rows,
        "requires_investigation": bool(problems) or any("investigate" in r["verdict"] for r in rows),
        "terminal_requires_review": any(r["status"] in TERMINAL for r in rows),
        "historical_snapshot_utc": historical.get("captured_utc") if historical else None,
        "model_calls": 0, "credentials_loaded": False,
        "limitations": "HTTP200 indicates liveness, not correct results. Terminal states require separate integrity/failure review; no unchanged-percent health claim."}


def find_history(parent, current, explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = sorted(Path(parent).rglob("second.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:512]
    eligible = []
    for path in candidates:
        try:
            row = read(path)
            if (row.get("schema") == SCHEMA and row.get("stage") == current["stage"]
                    and row.get("manifest_sha256") == current["manifest_sha256"]
                    and timestamp(current["captured_utc"]) - timestamp(row["captured_utc"]) >= 600):
                eligible.append(row)
        except (OSError, ValueError, KeyError):
            continue
    return max(eligible, key=lambda r: timestamp(r["captured_utc"])) if eligible else None


def save(path, row):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("write", "retrieve", "qa"))
    parser.add_argument("--conversations", help="Optional comma-separated subset, such as 0,1")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    cis = [int(x) for x in args.conversations.split(",")] if args.conversations else None
    if cis is not None and (not cis or len(set(cis)) != len(cis) or any(x not in range(10) for x in cis)):
        parser.error("Conversations must be unique values 0..9")
    if not 1 <= args.interval <= 60:
        parser.error("Snapshot interval must be 1..60 seconds")
    output = args.output.resolve()
    if output.is_relative_to(ROOT / "runs"):
        parser.error("Write health evidence to a new audit directory outside experimental runs")
    output.mkdir(parents=True, exist_ok=False)
    first = snapshot(args.stage, cis)
    save(output / "first.json", first)
    print(json.dumps({"first_snapshot": first["captured_utc"], "stage": args.stage, "selected_conversations": list(first["conversations"])}), flush=True)
    time.sleep(args.interval)
    second = snapshot(args.stage, cis)
    save(output / "second.json", second)
    historical = find_history(output.parent, second, args.previous)
    report = compare(first, second, historical=historical)
    save(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
