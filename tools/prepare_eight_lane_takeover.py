"""Explicit user-authorized scheduler handoff; running QA workers are not killed."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from process_control_windows import WinProcess, split_command_line

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def inventory():
    command = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -like '*python*' -and $_.CommandLine -like '*jev mem*' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, check=True, encoding="utf-8")
    data = json.loads(result.stdout or "[]")
    return data if isinstance(data, list) else [data]


def value(args, flag):
    return args[args.index(flag)+1] if flag in args else None


def normalized(path):
    return os.path.normcase(os.path.abspath(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true", help="Stop only the old two scheduling processes after sealing their state")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.exists():
        raise RuntimeError("Handoff manifest already exists; do not repeat a handoff")
    runs = []
    for arch, name, runner, methods in [
        ("mem0", "session_top20_v1", "run_session_experiments.py", ["sdk20", "rank", "mmr", "jev_single", "conditional"]),
        ("independent", "independent_top20_v1", "run_independent_experiments.py", ["rank", "mmr", "jev_single", "conditional"]),
    ]:
        path = ROOT / "runs" / name
        queue = json.loads((path / "queue_status.json").read_text(encoding="utf-8"))
        if queue["status"] != "running" or queue["architecture"] != arch:
            raise RuntimeError("Expected an active original scheduler for " + arch)
        runs.append({"architecture": arch, "run_dir": str(path), "runner": str(ROOT / "tools" / runner), "methods": methods, "original_queue": queue})
    processes = inventory()
    by_pid = {p["ProcessId"]: p for p in processes}
    schedulers = []
    for run in runs:
        pid = run["original_queue"]["pid"]
        proc = by_pid[pid]
        argv = split_command_line(proc["CommandLine"])
        if (not any(Path(x).name == "run_experiment_queue.py" for x in argv)
                or value(argv, "--architecture") != run["architecture"]
                or normalized(value(argv, "--run-dir")) != normalized(run["run_dir"])):
            raise RuntimeError("Scheduler command identity mismatch")
        handle = WinProcess(pid)
        schedulers.append((run, proc, handle))
    if not args.execute:
        print(json.dumps({"check_only": True, "schedulers": [p["ProcessId"] for _,p,_ in schedulers], "target_global_slots": 8}))
        for _, _, handle in schedulers:
            handle.close()
        return
    suspended = []
    worker_handles = []
    termination_started = False
    manifest = None
    try:
        for _, _, handle in schedulers:
            handle.suspend()
            suspended.append(handle)
        processes = inventory()
        by_pid = {p["ProcessId"]: p for p in processes}
        adopted = []
        exited_during_handoff = []
        original_snapshots = []
        for run, proc, scheduler in schedulers:
            current_path = Path(run["run_dir"]) / "queue_status.json"
            current = json.loads(current_path.read_text(encoding="utf-8"))
            if current["pid"] != scheduler.pid:
                raise RuntimeError("Queue identity changed during handoff")
            original_snapshots.append({"architecture": run["architecture"], "state": copy.deepcopy(current), "sha256": hashlib.sha256(current_path.read_bytes()).hexdigest()})
            run["original_queue"] = copy.deepcopy(current)
            matches = []
            for p in processes:
                argv = split_command_line(p["CommandLine"])
                if (any(normalized(x) == normalized(run["runner"]) for x in argv if x.endswith(".py"))
                        and value(argv, "--run-dir")
                        and normalized(value(argv,"--run-dir")) == normalized(run["run_dir"])):
                    if "qa" not in argv:
                        raise RuntimeError("Unexpected non-QA worker at takeover")
                    matches.append((p, argv))
            # A venv launcher and its real interpreter represent one task. Adopt the leaf.
            parents = {p["ParentProcessId"] for p, _ in matches}
            leaves = [(p,a) for p,a in matches if p["ProcessId"] not in parents]
            seen = set()
            for p, argv in leaves:
                ci = int(value(argv, "--conversations"))
                if ci in seen:
                    raise RuntimeError("Duplicate live writer for " + run["architecture"] + str(ci))
                seen.add(ci)
                job = current["jobs"].get("qa/conv_"+str(ci))
                if job is None or job["status"] != "running":
                    raise RuntimeError("Live worker and queue state disagree")
                try:
                    handle = WinProcess(p["ProcessId"])
                except ProcessLookupError:
                    exited_during_handoff.append({"architecture":run["architecture"], "conversation":ci,
                        "pid":p["ProcessId"], "exit_code":None, "reason":"Exited between inventory and handle acquisition; validate final summary"})
                    continue
                worker_handles.append(handle)
                adopted.append({"architecture": run["architecture"], "conversation": ci, "pid": handle.pid,
                    "creation_time_100ns": handle.creation_time_100ns, "command_line": p["CommandLine"], "log": job["log"]})
            # A recorded running job with no remaining PID is never reclassified
            # as pending. The supervisor validates its final summary or preserves
            # an explicit failure; absence of a log is not proof of no attempt.
        if len(adopted) > 8:
            raise RuntimeError("More than eight already-active workers; do not launch another")
        manifest = {"schema":"jev-mem-eight-lane-takeover-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "root":str(ROOT), "global_slots":8, "query_workers":4, "runs":runs, "adopted":adopted,
            "exited_during_handoff":exited_during_handoff,
            "old_schedulers":[{"pid":h.pid,"creation_time_100ns":h.creation_time_100ns,"command_line":p["CommandLine"]} for _,p,h in schedulers],
            "authorization":"User explicitly requested eight parallel conversation lanes; orchestration only; original model and evaluation protocol unchanged",
            "status":"sealed_before_scheduler_stop"}
        dump(destination.parent/"old_queue_snapshots.json",original_snapshots)
        dump(destination,manifest)
        termination_started = True
        for _, _, handle in schedulers:
            handle.terminate()
        for _, _, handle in schedulers:
            end = time.monotonic()+5
            while handle.is_running() and time.monotonic()<end:
                time.sleep(.05)
            if handle.is_running():
                raise RuntimeError("Old scheduler is still alive")
        manifest["status"]="ready_for_eight_lane_supervisor"
        manifest["scheduler_stop_utc"]=datetime.now(timezone.utc).isoformat()
        for item, handle in zip(adopted, worker_handles):
            item["observed_exit_code_at_handoff"] = handle.exit_code()
        dump(destination,manifest)
        print(json.dumps({"handoff":str(destination),"old_schedulers_stopped":[h.pid for _,_,h in schedulers],"adopted_qa_workers":[{"architecture":a["architecture"],"conversation":a["conversation"],"pid":a["pid"]} for a in adopted],"existing_workers_not_terminated":True}))
    except BaseException as error:
        # A failed partial takeover must never leave an old dispatcher suspended.
        recovered = []
        for handle in suspended:
            entry = {"pid":handle.pid}
            try:
                if handle.is_running():
                    handle.resume()
                    entry["resumed"] = True
                else:
                    entry["exit_code"] = handle.exit_code()
            except Exception as recovery_error:
                entry["recovery_error"] = str(recovery_error)
            recovered.append(entry)
        suspended = []
        if manifest is not None:
            manifest["status"] = "handoff_failed_needs_reconcile" if termination_started else "handoff_aborted_old_dispatchers_resumed"
            manifest["failure"] = {"type":type(error).__name__, "message":str(error), "old_dispatchers":recovered}
            dump(destination,manifest)
        raise
    finally:
        if not termination_started:
            for handle in reversed(suspended):
                if handle.is_running():
                    handle.resume()
        for handle in worker_handles + [h for _,_,h in schedulers]:
            handle.close()


if __name__ == "__main__":
    main()
