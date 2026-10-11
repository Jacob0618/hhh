"""One-shot, gated write -> tag -> qa supervisor for the frozen 400-question pilot.

No provider is constructed. Only the reviewed frozen driver is launched, with its
existing main-batch gate and protocol. A failed stage stops the pipeline; neither
this program nor an automatic restart retries it. --check is entirely read-only.

Development-review schema: jev-mem-pilot400-development-review-v1; status=approved;
absolute run_dir; protocol_sha256; run_manifest_sha256; source_dataset_sha256;
cohort_sha256; development_conversations=[0,1]; source_dates_reviewed=true;
source_coverage_reviewed=true; reviewed_by; timezone-aware reviewed_utc;
workspace-relative review_path and its byte review_sha256; banks is exactly four
rows (conv0/1 x session/regroup), each with conversation, bank, bank_path,
bank_sha256, bank_summary_sha256, grouping_sha256, raw_source_sha256. The final
four hashes are byte hashes of the corresponding bank-directory artifacts.
This tool checks these assertions and artifacts; it does not manufacture review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pilot400_experiments as driver

SCHEMA = "jev-mem-pilot400-pipeline-v1"
REVIEW_SCHEMA = "jev-mem-pilot400-development-review-v1"
STAGES = ("write", "tag", "qa")
CONVERSATIONS = list(range(10))
DRIVER = ROOT / "tools/run_pilot400_experiments.py"
DEFAULT_RUN = ROOT / "runs/pilot400_modules_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Audit timestamps must have a timezone")
    return result


def file_sha(path):
    return driver.file_sha(path)


def validate_development_review(run_dir, review_path):
    run_dir = driver._safe_run_dir(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("An existing, frozen pilot run manifest is required")
    manifest = driver.ensure_manifest(run_dir, create=False)
    review = driver.read(review_path)
    if (review.get("schema") != REVIEW_SCHEMA or review.get("status") != "approved"
            or review.get("run_dir") != str(run_dir)
            or review.get("protocol_sha256") != manifest["protocol_sha256"]
            or review.get("run_manifest_sha256") != file_sha(manifest_path)
            or review.get("source_dataset_sha256") != driver.SOURCE_HASH
            or review.get("cohort_sha256") != driver.COHORT_SHA256
            or review.get("development_conversations") != [0, 1]
            or review.get("source_dates_reviewed") is not True
            or review.get("source_coverage_reviewed") is not True
            or not isinstance(review.get("reviewed_by"), str) or not review["reviewed_by"].strip()):
        raise ValueError("Development raw-source/date review is missing or bound to different inputs")
    reviewed_time = parse_time(review["reviewed_utc"])
    if reviewed_time > datetime.now(timezone.utc):
        raise ValueError("Development review timestamp is in the future")
    report_value = review.get("review_path")
    if not isinstance(report_value, str) or Path(report_value).is_absolute():
        raise ValueError("Development report must be workspace-relative")
    report = (ROOT / report_value).resolve()
    if not report.is_relative_to(ROOT) or file_sha(report) != review.get("review_sha256"):
        raise ValueError("Development review document hash differs")
    rows = review.get("banks")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Exactly four complete development banks must be reviewed")
    keys = [(row.get("conversation"), row.get("bank")) for row in rows]
    expected = {(ci, bank) for ci in (0, 1) for bank in ("session", "regroup")}
    if len(set(keys)) != 4 or set(keys) != expected or any(type(row.get("conversation")) is not int for row in rows):
        raise ValueError("Development review must cover conv0/1 and both fact banks exactly once")
    dataset = driver.load_data()
    validated = []
    for row in sorted(rows, key=lambda x: (x["conversation"], x["bank"])):
        ci, bank_name = row["conversation"], row["bank"]
        raw = driver.raw_turns_from_locomo(dataset[ci]["conversation"])
        # Reuses the frozen complete-source, exactly-once, dates, schema, fact,
        # vector, summary and protocol validation. It makes no model calls.
        bank = driver.load_bank(run_dir, ci, bank_name, raw, manifest)
        folder = run_dir / "conversations" / f"conv_{ci}" / "banks" / bank_name
        expected_row = {"conversation": ci, "bank": bank_name, "bank_path": bank["path"],
                        "bank_sha256": bank["sha256"], "bank_summary_sha256": bank["summary_sha256"],
                        "grouping_sha256": file_sha(folder / "grouping.json"),
                        "raw_source_sha256": file_sha(folder / "raw_source.json")}
        if row != expected_row:
            raise ValueError("Development review does not bind the full current bank: " + str((ci, bank_name)))
        validated.append(expected_row)
    return {"review_path": str(Path(review_path).resolve()), "review_sha256": file_sha(review_path),
            "document_path": str(report), "document_sha256": review["review_sha256"],
            "protocol_sha256": manifest["protocol_sha256"], "run_manifest_sha256": file_sha(manifest_path),
            "source_dataset_sha256": driver.SOURCE_HASH, "cohort_sha256": driver.COHORT_SHA256,
            "banks": validated}


class SupervisorLock:
    """OS releases this lock on process death; driver.lock belongs to the driver."""
    def __init__(self, directory):
        self.path = Path(directory) / "supervisor.lock"
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another pilot pipeline supervisor owns this run") from None
        return self

    def __exit__(self, *_):
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def creation_time(pid):
    from tools.process_control_windows import WinProcess
    handle = WinProcess(pid)
    try:
        return handle.creation_time_100ns
    finally:
        handle.close()


def stage_command(stage, run_dir, closure_path):
    return [str(Path(sys.executable).resolve()), "-u", str(DRIVER.resolve()), stage,
            "--run-dir", str(Path(run_dir).resolve()), "--conversations", "0,1,2,3,4,5,6,7,8,9",
            "--workers", "8", "--closure", str(Path(closure_path).resolve())]


def verify_driver_identity(pid, launcher_pid, launcher_creation, command):
    """Bind a native Python child or Windows venv launcher's direct worker."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid driver PID")
    if pid == launcher_pid:
        return {"pid": pid, "creation_time_100ns": launcher_creation,
                "relationship": "launched_process", "command": command}
    if os.name != "nt":
        raise RuntimeError("Venv worker identity requires Windows process inventory")
    script = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
              "Get-CimInstance Win32_Process -Filter \"ProcessId=" + str(pid) + "\" "
              "| Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    observed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                              capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    value = json.loads(observed.stdout) if observed.stdout.strip() else None
    if not isinstance(value, dict) or value.get("ProcessId") != pid or value.get("ParentProcessId") != launcher_pid:
        raise ValueError("Stage PID is not the launched process or its direct venv worker")
    from tools.process_control_windows import split_command_line
    argv = split_command_line(value.get("CommandLine") or "")
    if not argv or Path(argv[0]).name.lower() not in {"python.exe", "pythonw.exe"} or argv[1:] != command[1:]:
        raise ValueError("Venv worker command does not match the exact frozen invocation")
    created = creation_time(pid)
    now_filetime = int(datetime.now(timezone.utc).timestamp() * 10_000_000) + 116444736000000000
    if not launcher_creation <= created <= now_filetime:
        raise ValueError("Venv worker creation time is outside the launch interval")
    return {"pid": pid, "creation_time_100ns": created, "parent_pid": launcher_pid,
            "relationship": "direct_venv_worker", "command": argv,
            "command_line": value["CommandLine"], "verified_utc": now()}


def validate_stage_status(path, stage, protocol_sha256, started_utc, expected_pid, exited_utc, prior_sha256=None):
    row = driver.read(path)
    current_sha = file_sha(path)
    if prior_sha256 == current_sha:
        raise ValueError("Child exited without publishing a fresh stage status")
    if (row.get("stage") != stage or row.get("protocol_sha256") != protocol_sha256
            or row.get("status") != "complete" or type(row.get("expected")) is not int or row["expected"] != 10
            or type(row.get("done")) is not int or row["done"] != 10
            or row.get("count_unit") != "conversations"
            or set(row.get("conversations", {})) != {str(ci) for ci in CONVERSATIONS}
            or any(value.get("status") != "complete" for value in row["conversations"].values())):
        raise ValueError("Stage is not complete for all ten conversations: " + stage)
    if (parse_time(row["started_utc"]) < parse_time(started_utc)
            or parse_time(row["finished_utc"]) < parse_time(row["started_utc"])
            or parse_time(row["finished_utc"]) > parse_time(exited_utc)
            or type(row.get("pid")) is not int or row["pid"] != expected_pid):
        raise ValueError("Stage status timestamp/PID does not describe this child invocation")
    return {"path": str(Path(path).resolve()), "sha256": current_sha, "driver_pid": row["pid"],
            "started_utc": row["started_utc"], "finished_utc": row["finished_utc"],
            "expected": row["expected"], "done": row["done"]}


def check(run_dir, review_path, closure_path, *, review_checker=None, gate_checker=None):
    run_dir = driver._safe_run_dir(run_dir)
    development = (review_checker or validate_development_review)(run_dir, review_path)
    closure = (gate_checker or driver.main_batch_gate)(closure_path)
    if (run_dir / "driver.lock").exists():
        raise RuntimeError("Existing driver.lock needs its current owner/reviewer; pipeline will not remove it")
    if (run_dir / "orchestration_pipeline_v1/status.json").exists():
        raise RuntimeError("This one-shot pipeline has already been started; no implicit replay is allowed")
    return {"schema": SCHEMA, "status": "ready", "run_dir": str(run_dir),
            "development_review": development, "main_batch_gate": closure,
            "stages": list(STAGES), "conversation_workers": 8, "conversations": CONVERSATIONS,
            "network_calls": 0, "credentials_loaded": False}


def run_pipeline(run_dir, review_path, closure_path, *, popen_factory=None, creation_reader=None,
                 driver_identity_reader=None, review_checker=None, gate_checker=None):
    run_dir = driver._safe_run_dir(run_dir)
    ready = check(run_dir, review_path, closure_path, review_checker=review_checker, gate_checker=gate_checker)
    directory = run_dir / "orchestration_pipeline_v1"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "status.json"
    identify = creation_reader or creation_time
    identify_driver = driver_identity_reader or verify_driver_identity
    with SupervisorLock(directory):
        if path.exists():
            raise RuntimeError("Orchestration status already exists; refuse a second launch")
        state = {**ready, "status": "running", "pid": os.getpid(),
                 "creation_time_100ns": identify(os.getpid()), "started_utc": now(), "updated_utc": now(),
                 "pipeline_code_sha256": file_sha(Path(__file__)),
                 "stage": None, "stages": {s: {"status": "pending"} for s in STAGES},
                 "done": 0, "expected": 3, "count_unit": "stages",
                 "api_log_path": str(run_dir / "api_calls.jsonl"),
                 "driver_status_paths": {s: str(run_dir / (s + "_status.json")) for s in STAGES},
                 "heartbeat_policy": "Observe child identity plus stage/API artifact deltas; never auto-relaunch this pipeline or a failed stage."}

        def save():
            state["updated_utc"] = now()
            driver.publish_json(path, state)

        save()
        child = None
        try:
            for stage in STAGES:
                state["stage"] = stage
                record = state["stages"][stage]
                record.update(status="checking_gates", started_utc=now())
                save()
                # Recheck immutable inputs and main closure before every launch.
                reviewed = (review_checker or validate_development_review)(run_dir, review_path)
                if reviewed != ready["development_review"]:
                    raise ValueError("Development review or frozen input changed during pipeline")
                closure = (gate_checker or driver.main_batch_gate)(closure_path)
                if closure["closure_sha256"] != ready["main_batch_gate"]["closure_sha256"]:
                    raise ValueError("Main-batch closure changed during pipeline")
                if (run_dir / "driver.lock").exists():
                    raise RuntimeError("Another stage owns driver.lock; pipeline refuses to contend")
                stage_path = run_dir / (stage + "_status.json")
                prior_sha = file_sha(stage_path) if stage_path.exists() else None
                command = stage_command(stage, run_dir, closure_path)
                log_path = directory / (stage + ".log")
                record.update(command=command, log=str(log_path), driver_status_path=str(stage_path),
                              prior_status_sha256=prior_sha, status="launching", launch_started_utc=now())
                save()
                with log_path.open("xb") as log:
                    child = (popen_factory or subprocess.Popen)(command, cwd=str(ROOT), stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    # Retain the PID even if opening its identity handle fails.
                    record.update(child_pid=child.pid, status="identity_pending")
                    save()
                    record.update(child_creation_time_100ns=identify(child.pid), status="running")
                    save()
                    def capture_driver_identity():
                        if record.get("driver_identity") is not None or not stage_path.exists():
                            return
                        candidate = driver.read(stage_path)
                        # Ignore a previous development invocation's stale file.
                        if (candidate.get("stage") != stage
                                or candidate.get("protocol_sha256") != ready["development_review"]["protocol_sha256"]
                                or parse_time(candidate["started_utc"]) < parse_time(record["launch_started_utc"])):
                            return
                        record["observed_driver_pid"] = candidate.get("pid")
                        save()
                        record["driver_identity"] = identify_driver(candidate.get("pid"), child.pid,
                            record["child_creation_time_100ns"], command)
                        save()
                    capture_driver_identity()
                    while True:
                        try:
                            code = child.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            capture_driver_identity()
                            record["last_alive_check_utc"] = now()
                            save()
                    record.update(exit_code=code, child_finished_utc=now())
                    save()
                    capture_driver_identity()
                child = None
                if code != 0:
                    raise RuntimeError(stage + " child exited with code " + str(code))
                if not record.get("driver_identity"):
                    raise ValueError("No verified actual driver PID; refusing subsequent stage")
                record["verified_stage_status"] = validate_stage_status(
                    stage_path, stage, ready["development_review"]["protocol_sha256"],
                    record["launch_started_utc"], record["driver_identity"]["pid"], record["child_finished_utc"], prior_sha)
                record.update(status="complete", finished_utc=now())
                state["done"] += 1
                save()
            state.update(status="complete", finished_utc=now(), stage=None)
            save()
        except BaseException as error:
            state.update(status="requires_review", finished_utc=now(),
                         error={"type": type(error).__name__, "message": str(error)[:1800]},
                         active_child_may_still_be_running=child is not None and child.poll() is None,
                         resume_policy="No automatic retry. Preserve this status and driver artifacts; review the recorded PID/creation time before any separate recovery decision.")
            if state["stage"] is not None:
                state["stages"][state["stage"]].update(status="requires_review", error=state["error"], finished_utc=now())
            save()
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--development-review", required=True, type=Path)
    parser.add_argument("--closure", type=Path, default=driver.CLOSURE_PATH)
    parser.add_argument("--check", action="store_true", help="Verify gates without creating state or launching a child")
    args = parser.parse_args(argv)
    result = check(args.run_dir, args.development_review, args.closure) if args.check else run_pipeline(
        args.run_dir, args.development_review, args.closure)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"ready", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
