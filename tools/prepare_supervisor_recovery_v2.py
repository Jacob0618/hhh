"""Seal a new handoff after a dead supervisor; no worker is stopped or restarted."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import run_eight_lane_supervisor as base
from prepare_eight_lane_takeover import inventory, normalized, value
from process_control_windows import WinProcess, split_command_line
from run_eight_lane_supervisor_v2 import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-epoch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    previous = Path(args.previous_epoch).resolve()
    destination = Path(args.output).resolve()
    if destination.parent.exists():
        raise RuntimeError("Recovery epoch must be fresh")
    if previous.parent != base.ROOT / "runs" or destination.parent.parent != base.ROOT / "runs":
        raise ValueError("Epochs must remain directly within this workspace's runs")
    mutex = base.RootMutex(base.ROOT)
    handles = []
    try:
        prior = base.read_json(previous / "global_status.json")
        old_manifest = base.read_json(previous / "handoff.json")
        try:
            old_process = WinProcess(prior["pid"])
        except ProcessLookupError:
            pass
        else:
            old_process.close()
            raise RuntimeError("Previous supervisor PID still exists; no recovery while alive or reused")
        processes = inventory()
        live_schedulers = [p for p in processes if any(
            Path(x).name in {"run_experiment_queue.py", "run_eight_lane_supervisor.py", "run_eight_lane_supervisor_v2.py"}
            for x in split_command_line(p["CommandLine"]))]
        if live_schedulers:
            raise RuntimeError("A scheduling process is still alive")
        runs, adopted, snapshots = [], [], []
        for original in old_manifest["runs"]:
            run = {k: copy.deepcopy(original[k]) for k in ("architecture", "run_dir", "runner", "methods")}
            queue_path = Path(run["run_dir"]) / "queue_status.json"
            queue = base.read_json(queue_path)
            if queue["pid"] != prior["pid"] or queue["methods"] != run["methods"]:
                raise RuntimeError("Current queue identity changed")
            recorded = {key.split("/", 1)[1]: job for key, job in prior["jobs"].items()
                        if key.startswith(run["architecture"] + "/")}
            if recorded != queue["jobs"]:
                raise RuntimeError("Global and per-run jobs disagree; manual reconciliation required")
            # Only scheduling changes. Refuse if any API/selection/reader/runner source changed.
            for relative, expected in queue["source_hashes"].items():
                name = relative.replace("\\", "/")
                if name == "jev_mem/locomo_official_metrics.py":
                    continue  # Documented prior offline scorer import repair; not used by QA.
                if hashlib.sha256((base.ROOT / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Frozen source changed: " + name)
            run["original_queue"] = copy.deepcopy(queue)
            snapshots.append({"architecture": run["architecture"], "state": queue,
                              "sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest()})
            matches = []
            for proc in processes:
                argv = split_command_line(proc["CommandLine"])
                if any(normalized(x) == normalized(run["runner"]) for x in argv if x.endswith(".py")):
                    if normalized(value(argv, "--run-dir") or "") != normalized(run["run_dir"]) or "qa" not in argv:
                        raise RuntimeError("Unexpected experiment worker scope")
                    matches.append((proc, argv))
            parents = {proc["ParentProcessId"] for proc, _ in matches}
            seen = set()
            for proc, argv in matches:
                if proc["ProcessId"] in parents:
                    continue
                ci = int(value(argv, "--conversations"))
                if ci in seen:
                    raise RuntimeError("Duplicate live worker for same architecture/conversation")
                seen.add(ci)
                job = queue["jobs"].get("qa/conv_" + str(ci))
                if not job or job["status"] != "running":
                    raise RuntimeError("Live worker missing from current running state")
                try:
                    handle = WinProcess(proc["ProcessId"])
                except ProcessLookupError:
                    continue  # Keep recorded running job; supervisor validates, never requeues it.
                handles.append(handle)
                record = {"architecture": run["architecture"], "conversation": ci, "pid": handle.pid,
                          "creation_time_100ns": handle.creation_time_100ns,
                          "command_line": proc["CommandLine"], "log": job["log"]}
                base.validate_adopted_command(record, run)
                adopted.append(record)
            runs.append(run)
        if len(adopted) > 8:
            raise RuntimeError("More than eight existing QA jobs")
        manifest = {"schema": base.HANDOFF_SCHEMA, "status": "ready_for_eight_lane_supervisor",
                    "created_utc": base.now(), "root": str(base.ROOT), "global_slots": 8, "query_workers": 4,
                    "runs": runs, "adopted": adopted, "resumed_from_epoch": str(previous),
                    "reason": "Previous supervisor exited on Windows atomic status replace PermissionError",
                    "workers_terminated": False, "failed_jobs_requeued": False,
                    "authorization": "Continue previously authorized fixed experiments with eight global lanes; scheduling repair only"}
        destination.parent.mkdir()
        atomic_json(destination.parent / "old_queue_snapshots.json", snapshots)
        atomic_json(destination.parent / "previous_global_status.json", prior)
        shutil.copy2(previous / "supervisor.stderr.log", destination.parent / "previous_supervisor.stderr.log")
        atomic_json(destination, manifest)
        # All queue states/failed records are copied verbatim; absent jobs alone remain pending.
        print(json.dumps({"handoff": str(destination), "adopted": [{k: row[k] for k in
              ("architecture", "conversation", "pid")} for row in adopted],
              "pending": [run["architecture"] + "/qa/conv_" + str(ci) for run in runs
                          for ci in run["original_queue"]["conversations"]
                          if "qa/conv_" + str(ci) not in run["original_queue"]["jobs"]]}))
    finally:
        for handle in handles:
            handle.close()
        mutex.close()


if __name__ == "__main__":
    main()
