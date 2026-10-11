"""Two-snapshot, read-only liveness checks; no model clients or credential reads.

Windows status/log reads share READ, WRITE and DELETE so an audit does not prevent
atomic publication. A static percentage or a live PID alone cannot pass health.
Only audit files under the requested new output directory are written.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def utcnow():
    return datetime.now(timezone.utc)


def shared_read(path, *, offset=None, tail=None):
    """Close a shared Windows handle before decoding/parsing its bytes."""
    path = Path(path)
    if os.name == "nt":
        import msvcrt
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
            raise
        stream = os.fdopen(fd, "rb")
    else:
        stream = path.open("rb")
    with stream:
        if offset is not None:
            stream.seek(offset)
        elif tail is not None:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - tail))
        return stream.read()


def read_json(path):
    return json.loads(shared_read(path).decode("utf-8-sig"))


def parse_rows(data):
    rows = []
    for line in data.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue  # Tail-start or last in-flight partial row; never infer failure.
        if isinstance(row, dict):
            rows.append(row)
    return rows


def api_delta(path, start, stop):
    if stop < start:
        return {"log_shrank": True}
    raw = shared_read(path, offset=start)[:stop - start]
    rows = parse_rows(raw)
    statuses = Counter(str(row.get("status")) for row in rows if not row.get("cache_hit"))
    succeeded = [row for row in rows if row.get("status") == 200 and not row.get("cache_hit")]
    return {"bytes_added": stop - start, "rows": len(rows),
            "http200": sum(row.get("status") == 200 and not row.get("cache_hit") for row in rows),
            "cache_hits": sum(row.get("cache_hit") is True for row in rows),
            "network_statuses": dict(statuses),
            "unique_successful_request_hashes": len({row.get("request_hash") for row in succeeded}),
            "successful_query_contexts": sorted({str(row.get("context", {}).get("query_id")) for row in succeeded}),
            "exhausted_attempts": sum(row.get("attempt", 0) >= 3 and row.get("status") not in (None, 200) for row in rows)}


def inventory():
    command = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
               "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' "
               "-and $_.CommandLine -match 'run_(eight_lane_supervisor(_v2)?|session_experiments|independent_experiments)\\.py' } "
               "| Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    rows = json.loads(result.stdout or "[]")
    return rows if isinstance(rows, list) else [rows]


def snapshot(epoch, run_dirs):
    from process_control_windows import WinProcess, split_command_line
    from run_eight_lane_supervisor import validate_adopted_command
    now = utcnow()
    state = read_json(epoch / "global_status.json")
    processes = inventory()
    by_pid = {row["ProcessId"]: row for row in processes}
    problems = []
    supervisor = {"pid": state.get("pid"), "alive": False}
    try:
        handle = WinProcess(state["pid"])
        try:
            supervisor.update(alive=handle.is_running(), creation_time_100ns=handle.creation_time_100ns)
        finally:
            handle.close()
        command = by_pid.get(state["pid"], {}).get("CommandLine", "")
        args = split_command_line(command) if command else []
        supervisor["command_verified"] = (any(Path(arg).name in {"run_eight_lane_supervisor.py", "run_eight_lane_supervisor_v2.py"} for arg in args)
                                           and "--handoff" in args and
                                           Path(args[args.index("--handoff") + 1]).resolve().parent == epoch.resolve())
    except (OSError, ValueError, IndexError, KeyError) as error:
        supervisor["error_type"] = type(error).__name__
    state_age = (now - datetime.fromisoformat(state["updated_utc"])).total_seconds()
    if state.get("status") == "running" and (not supervisor["alive"] or not supervisor.get("command_verified") or state_age > 30):
        problems.append("supervisor_missing_wrong_command_or_stale_state")
    runs = {}
    for run_dir in run_dirs:
        queue = read_json(run_dir / "queue_status.json")
        arch = queue["architecture"]
        runner = ROOT / "tools" / ("run_session_experiments.py" if arch == "mem0" else "run_independent_experiments.py")
        if queue["pid"] != state["pid"]:
            problems.append(arch + ":queue_supervisor_pid_mismatch")
        convs = {}
        for folder in sorted(run_dir.glob("conv_*")):
            ci = int(folder.name.split("_")[-1])
            job = queue["jobs"].get("qa/" + folder.name, {})
            counts = {method: len(list((folder / "qa" / method).glob("*.json"))) for method in queue["methods"]}
            api_path = folder / "api_calls.jsonl"
            tail = parse_rows(shared_read(api_path, tail=65536)) if api_path.exists() else []
            log = {"path": str(api_path), "size": api_path.stat().st_size if api_path.exists() else 0,
                   "last_event_utc": tail[-1].get("timestamp_utc") if tail else None}
            record = {"job_status": job.get("status", "pending"), "qa_counts": counts,
                      "qa_total": sum(counts.values()), "api_log": log}
            if job.get("status") == "running":
                matches = []
                for proc in processes:
                    argv = split_command_line(proc["CommandLine"])
                    if (any(Path(arg).name == runner.name for arg in argv) and "--run-dir" in argv
                        and Path(argv[argv.index("--run-dir") + 1]).resolve() == run_dir.resolve()
                        and "--conversations" in argv and argv[argv.index("--conversations") + 1] == str(ci)):
                        matches.append(proc)
                parents = {row["ParentProcessId"] for row in matches}
                leaves = [row for row in matches if row["ProcessId"] not in parents]
                if len(leaves) != 1:
                    problems.append(f"{arch}/{folder.name}:expected_one_worker_found_{len(leaves)}")
                else:
                    proc = leaves[0]
                    try:
                        validate_adopted_command({"command_line": proc["CommandLine"], "conversation": ci},
                                                 {"runner": str(runner), "run_dir": str(run_dir), "methods": queue["methods"]})
                        handle = WinProcess(proc["ProcessId"], expected_creation=job.get("creation_time_100ns"))
                        try:
                            record["worker"] = {"pid": handle.pid, "alive": handle.is_running(),
                                                "creation_time_100ns": handle.creation_time_100ns, "command_verified": True}
                        finally:
                            handle.close()
                    except (OSError, ValueError) as error:
                        problems.append(f"{arch}/{folder.name}:worker_identity_{type(error).__name__}")
            convs[folder.name] = record
        runs[arch] = {"run_dir": str(run_dir), "queue_status": queue["status"], "conversations": convs}
    active = sum(record["job_status"] == "running" for run in runs.values() for record in run["conversations"].values())
    if active > 8 or state.get("active_count", 0) > 8:
        problems.append("global_eight_lane_cap_exceeded")
    return {"captured_utc": now.isoformat(), "supervisor": supervisor, "supervisor_status": state.get("status"),
            "state_age_seconds": state_age, "pending_jobs": state.get("pending_jobs", []),
            "active_jobs": active, "problems": problems, "runs": runs}


def classify_progress(first, second, delta):
    if delta.get("log_shrank") or second["qa_total"] < first["qa_total"]:
        return "artifact_regression_needs_investigation"
    if second["job_status"] != "running":
        return "terminal_requires_summary_review" if second["job_status"] == "failed" else "terminal_or_pending"
    if not second.get("worker", {}).get("alive"):
        return "worker_not_verified_needs_investigation"
    if second["qa_total"] > first["qa_total"] or delta.get("http200", 0) > 0:
        return "confirmed_progress"
    if delta.get("rows", 0) > 0:
        return "activity_without_success_needs_investigation"
    return "no_observed_progress_needs_investigation"


def compare(first, second):
    rows = []
    for arch, run in second["runs"].items():
        for conv, record in run["conversations"].items():
            old = first["runs"][arch]["conversations"][conv]
            if old["job_status"] != "running" and record["job_status"] != "running":
                continue
            delta = api_delta(record["api_log"]["path"], old["api_log"]["size"], record["api_log"]["size"])
            rows.append({"architecture": arch, "conversation": conv, "qa_added": record["qa_total"] - old["qa_total"],
                         "api_delta": delta, "verdict": classify_progress(old, record, delta)})
    unresolved = [row for row in rows if "needs_investigation" in row["verdict"]]
    return {"scope": "Two-snapshot liveness, not a full scientific integrity or failure-recovery audit",
            "first_utc": first["captured_utc"], "second_utc": second["captured_utc"], "jobs": rows,
            "unresolved_jobs": unresolved, "snapshot_problems": second["problems"],
            "healthy_progress_confirmed": bool(rows) and not unresolved and not second["problems"],
            "first_pass_terminal": second["active_jobs"] == 0 and not second["pending_jobs"],
            "model_calls_made": 0}


def long_window_check(previous, current, *, minimum_seconds=600):
    """Success HTTP traffic cannot hide a QA count stuck across long checks."""
    elapsed = (datetime.fromisoformat(current["captured_utc"]) -
               datetime.fromisoformat(previous["captured_utc"])).total_seconds()
    report = {"previous_utc": previous["captured_utc"], "elapsed_seconds": elapsed,
              "minimum_seconds": minimum_seconds, "checked_jobs": [], "unresolved_jobs": []}
    if elapsed < minimum_seconds:
        return report
    for arch, run in current["runs"].items():
        old_convs = previous.get("runs", {}).get(arch, {}).get("conversations", {})
        for name, record in run["conversations"].items():
            old = old_convs.get(name, {})
            if record["job_status"] != "running" or old.get("job_status") != "running":
                continue
            worker, old_worker = record.get("worker", {}), old.get("worker", {})
            if not worker.get("alive") or any(worker.get(key) != old_worker.get(key)
                                               for key in ("pid", "creation_time_100ns")):
                continue
            delta = record["qa_total"] - old["qa_total"]
            row = {"architecture": arch, "conversation": name, "qa_added": delta,
                   "verdict": "qa_progress_confirmed" if delta > 0 else "long_window_QA_stall_needs_investigation"}
            report["checked_jobs"].append(row)
            if delta <= 0:
                report["unresolved_jobs"].append(row)
    return report


def previous_snapshot(current, directory):
    """Use the latest saved health snapshot at least ten minutes old, if any."""
    cutoff = datetime.fromisoformat(current["captured_utc"]).timestamp() - 600
    candidates = []
    for path in Path(directory).glob("*/second.json"):
        try:
            value = read_json(path)
            timestamp = datetime.fromisoformat(value["captured_utc"]).timestamp()
            if timestamp <= cutoff and "runs" in value:
                candidates.append((timestamp, path, value))
        except (OSError, ValueError, KeyError):
            continue
    if not candidates:
        return None
    _, path, value = max(candidates, key=lambda item: item[0])
    return path, value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=Path, default=ROOT / "runs/orchestration_eight_lane_v2")
    parser.add_argument("--run-dir", action="append", type=Path)
    parser.add_argument("--interval", type=float, default=30, help="Seconds between snapshots, 1..60")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    if args.output_dir.exists():
        parser.error("Output directory must be new; audit reports are immutable")
    run_dirs = args.run_dir or [ROOT / "runs/session_top20_v1", ROOT / "runs/independent_top20_v1"]
    args.output_dir.mkdir(parents=True)
    first = snapshot(args.epoch, run_dirs)
    (args.output_dir / "first.json").write_text(json.dumps(first, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"first_snapshot": first["captured_utc"], "active_jobs": first["active_jobs"], "waiting_seconds": args.interval}), flush=True)
    time.sleep(args.interval)
    second = snapshot(args.epoch, run_dirs)
    (args.output_dir / "second.json").write_text(json.dumps(second, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = compare(first, second)
    history = previous_snapshot(second, ROOT / "research/heartbeat_checks")
    if history:
        path, old = history
        report["long_window"] = {"snapshot_path": str(path), **long_window_check(old, second)}
        if report["long_window"]["unresolved_jobs"]:
            report["healthy_progress_confirmed"] = False
            report["unresolved_jobs"].extend(report["long_window"]["unresolved_jobs"])
    else:
        report["long_window"] = {"available": False, "note": "First short-window check only; no long-window liveness claim"}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["healthy_progress_confirmed"] or report["first_pass_terminal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
