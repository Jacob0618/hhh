"""Take over frozen QA queues with eight shared conversation slots.

This tool changes scheduling only. It never retries old QA failures, runs tagging,
modifies experiment code, or kills a worker. The handoff epoch is single-use.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_mem.data import load_data, queries

HANDOFF_SCHEMA = "jev-mem-eight-lane-takeover-v1"
STATUS_SCHEMA = "jev-mem-eight-lane-supervisor-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def expected_questions(ci):
    dataset = load_data()
    if isinstance(ci, bool) or not isinstance(ci, int) or not 0 <= ci < len(dataset):
        raise ValueError("Invalid conversation index")
    return list(queries(dataset[ci], ci))


def validate_summary(run_dir, ci, methods, question_loader=expected_questions):
    """Exit zero is insufficient: require the exact full QA cohort and artifacts."""
    errors = []
    expected = question_loader(ci)
    ids = [question["id"] for question in expected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected query cohort must be nonempty and unique")
    folder = Path(run_dir) / f"conv_{ci}"
    try:
        summary = read_json(folder / "qa_summary.json")
    except (OSError, ValueError) as error:
        return {"valid": False, "expected_queries": len(ids), "errors": ["missing_or_invalid_qa_summary: " + type(error).__name__]}
    if summary.get("conversation") != ci:
        errors.append("summary_conversation_mismatch")
    seen = summary.get("requested_query_ids")
    if not isinstance(seen, list) or len(seen) != len(ids) or len(set(seen)) != len(ids) or set(seen) != set(ids):
        errors.append("summary_query_ids_incomplete_or_different")
    if summary.get("requested_queries") != len(ids):
        errors.append("summary_requested_query_count_mismatch")
    for flag in ("smoke_only", "smoke_run", "partial_conversation_smoke"):
        if summary.get(flag) is True:
            errors.append("summary_is_smoke: " + flag)
    if summary.get("full_conversation_qa_scope") is False:
        errors.append("summary_is_partial_query_scope")
    groups = summary.get("methods")
    if not isinstance(groups, dict) or set(groups) != set(methods):
        errors.append("summary_methods_mismatch")
        groups = groups if isinstance(groups, dict) else {}
    for method in methods:
        group = groups.get(method, {})
        if group.get("requested") != len(ids) or group.get("completed") != len(ids):
            errors.append("incomplete_method_counts: " + method)
        method_folder = folder / "qa" / method
        files = {path.stem: path for path in method_folder.glob("*.json")}
        if set(files) != set(ids):
            errors.append("qa_file_cohort_mismatch: " + method)
        for question in expected:
            path = files.get(question["id"])
            if path is None:
                continue
            try:
                result = read_json(path)
                if (result.get("query_id") != question["id"] or result.get("conversation") != ci
                        or result.get("method") != method or result.get("question") != question["question"]):
                    errors.append("qa_file_identity_mismatch: " + method + "/" + question["id"])
                if result.get("partial_conversation_smoke") is not False or result.get("smoke_run") is True:
                    errors.append("qa_file_smoke_or_missing_marker: " + method + "/" + question["id"])
                if not isinstance(result.get("judgment", {}).get("correct"), bool):
                    errors.append("qa_file_missing_judgment: " + method + "/" + question["id"])
            except (OSError, ValueError, TypeError) as error:
                errors.append("invalid_qa_file: " + method + "/" + question["id"] + ":" + type(error).__name__)
    if summary.get("evaluation_complete") is False or summary.get("complete") is False:
        errors.append("summary_declares_incomplete")
    return {"valid": not errors, "expected_queries": len(ids), "expected_methods": list(methods), "errors": errors,
            "summary_path": str(folder / "qa_summary.json"), "checked_utc": now(),
            "scope": "scheduler_cohort_and_identity_validation_not_full_scientific_cache_audit"}


def _argv(command_line):
    if isinstance(command_line, list) and all(isinstance(value, str) for value in command_line):
        return command_line
    if not isinstance(command_line, str) or not command_line.strip():
        raise ValueError("Adopted worker requires its captured command line")
    if os.name != "nt":
        return [value.strip('"') for value in shlex.split(command_line, posix=False)]
    from ctypes import wintypes
    shell32, kernel32 = ctypes.WinDLL("shell32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    count = ctypes.c_int()
    pointer = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not pointer:
        raise OSError(ctypes.get_last_error(), "CommandLineToArgvW failed")
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        kernel32.LocalFree(pointer)


def _path_equal(a, b):
    return str(Path(a).resolve()).casefold() == str(Path(b).resolve()).casefold()


def validate_adopted_command(record, run):
    args = _argv(record["command_line"])
    positions = [i for i, value in enumerate(args) if _path_equal(value, run["runner"])]
    if len(positions) != 1 or positions[0] + 1 >= len(args) or args[positions[0] + 1] != "qa":
        raise ValueError("Adopted command is not the expected QA runner")
    for flag, expected in (("--run-dir", run["run_dir"]), ("--conversations", str(record["conversation"])),
                           ("--methods", ",".join(run["methods"])), ("--workers", "4")):
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            raise ValueError("Adopted command missing/duplicate argument: " + flag)
        actual = args[args.index(flag) + 1]
        matched = _path_equal(actual, expected) if flag == "--run-dir" else actual == expected
        if not matched:
            raise ValueError("Adopted command argument mismatch: " + flag)


class RootMutex:
    """One supervisor per workspace, across Windows sessions and epochs."""
    def __init__(self, root):
        if os.name != "nt":
            raise RuntimeError("Live supervisor requires Windows")
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        suffix = hashlib.sha256(str(Path(root).resolve()).casefold().encode()).hexdigest()[:24]
        self.handle = self.kernel.CreateMutexW(None, True, "Global\\JevMemEightLane_" + suffix)
        error = ctypes.get_last_error()
        if not self.handle:
            raise OSError(error, "Cannot acquire scheduling mutex")
        if error == 183:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise RuntimeError("Another eight-lane supervisor holds this workspace mutex")

    def close(self):
        if self.handle:
            self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class ChildProcess:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    def is_running(self):
        return self.process.poll() is None

    def exit_code(self):
        return self.process.wait()

    def close(self):
        # Never terminate an experiment. Popen's own handle lifetime is managed
        # by its object; wait() is called only after poll reports completion.
        pass


class Supervisor:
    def __init__(self, handoff_path, *, process_factory, popen_factory=subprocess.Popen,
                 summary_validator=validate_summary, writer=atomic_json, sleeper=time.sleep):
        self.path = Path(handoff_path).resolve()
        self.epoch = self.path.parent
        self.handoff = read_json(self.path)
        self.process_factory, self.popen_factory = process_factory, popen_factory
        self.summary_validator, self.writer, self.sleeper = summary_validator, writer, sleeper
        self.active, self.pending, self.states, self.runs = {}, [], {}, {}
        self.started_keys = set()
        self._validate_handoff()
        self.global_state = {"schema": STATUS_SCHEMA, "status": "initializing", "pid": os.getpid(),
            "handoff": str(self.path), "epoch": str(self.epoch), "started_utc": now(),
            "global_slots": 8, "query_workers": 4, "deferred_stages": ["tag"],
            "scheduling_only": True, "runs": {}, "jobs": {}, "max_observed_active": 0}

    def _validate_handoff(self):
        h = self.handoff
        if h.get("schema") != HANDOFF_SCHEMA or h.get("global_slots") != 8 or h.get("query_workers") != 4:
            raise ValueError("Unsupported handoff protocol")
        if h.get("status") != "ready_for_eight_lane_supervisor":
            raise ValueError("Handoff is not ready: old schedulers must be stopped first")
        if not _path_equal(h.get("root", ""), ROOT):
            raise ValueError("Handoff workspace root differs from this tool")
        records = h.get("runs")
        if not isinstance(records, list) or {record.get("architecture") for record in records} != {"mem0", "independent"} or len(records) != 2:
            raise ValueError("Handoff must contain exactly the two distinct architectures")
        for run in records:
            arch = run["architecture"]
            methods = run.get("methods")
            old = run.get("original_queue")
            if (not isinstance(methods, list) or not methods or len(methods) != len(set(methods))
                    or not all(isinstance(value, str) and value for value in methods)
                    or not isinstance(old, dict) or not isinstance(old.get("jobs"), dict)):
                raise ValueError("Invalid methods or original queue")
            cis = old.get("conversations")
            if not isinstance(cis, list) or len(cis) != len(set(cis)) or not cis or any(isinstance(ci, bool) or not isinstance(ci, int) or not 0 <= ci < 10 for ci in cis):
                raise ValueError("Conversation cohort must be a unique list of pinned indices")
            if old.get("architecture") != arch or old.get("methods") != methods:
                raise ValueError("Architecture/methods changed from the original queue")
            if not Path(run["runner"]).is_file() or not Path(run["run_dir"]).is_dir():
                raise ValueError("Runner or run directory does not exist")
            self.runs[arch] = run
            state = copy.deepcopy(old)
            state.update({"status": "initializing", "pid": os.getpid(), "global_slots": 8, "query_workers": 4,
                          "scheduling_epoch": str(self.epoch), "handoff_created_utc": h.get("created_utc"),
                          "original_scheduler_pid": old.get("pid"), "deferred_stages": ["tag"]})
            self.states[arch] = state
        adopted = h.get("adopted", [])
        if not isinstance(adopted, list) or len(adopted) > 8:
            raise ValueError("More adopted workers than available global slots")
        keys, pids = set(), set()
        for record in adopted:
            key = (record.get("architecture"), record.get("conversation"))
            if key in keys or record.get("pid") in pids:
                raise ValueError("Duplicate adopted architecture/conversation or PID")
            keys.add(key)
            pids.add(record.get("pid"))
            run = self.runs.get(key[0])
            if run is None or key[1] not in run["original_queue"]["conversations"]:
                raise ValueError("Adopted worker is outside its run cohort")
            old = run["original_queue"]["jobs"].get(f"qa/conv_{key[1]}")
            if not old or old.get("status") != "running":
                raise ValueError("Only a recorded running QA job may be adopted")
            if (isinstance(record.get("pid"), bool) or not isinstance(record.get("pid"), int) or record["pid"] <= 0
                    or isinstance(record.get("creation_time_100ns"), bool)
                    or not isinstance(record.get("creation_time_100ns"), int) or record["creation_time_100ns"] <= 0):
                raise ValueError("Adopted PID and creation time must be positive integers")
            if str(record.get("log")) != str(old.get("log")):
                raise ValueError("Adopted log differs from the original queue log")
            validate_adopted_command(record, run)

    def _job(self, key):
        return self.states[key[0]]["jobs"].get(f"qa/conv_{key[1]}")

    def _set_job(self, key, value):
        self.states[key[0]]["jobs"][f"qa/conv_{key[1]}"] = value

    def _validate_finished(self, key, exit_code, previous, note):
        run = self.runs[key[0]]
        validation = self.summary_validator(run["run_dir"], key[1], run["methods"])
        success = (exit_code in (None, 0)) and validation["valid"]
        self._set_job(key, {**previous, "status": "complete" if success else "failed", "exit_code": exit_code,
                           "completed_utc": now(), "validation": validation, "completion_note": note,
                           "no_automatic_retry": True})

    def initialize(self):
        # Open/verify every adopted process before any new process can launch.
        adopted = {(record["architecture"], record["conversation"]): record for record in self.handoff.get("adopted", [])}
        try:
            for key, record in adopted.items():
                try:
                    process = self.process_factory(record["pid"], expected_creation=record["creation_time_100ns"])
                except ProcessLookupError as error:
                    if "reused" in str(error).casefold():
                        raise RuntimeError("Adopted PID was reused; refusing to release its slot or launch new work") from error
                    self._validate_finished(key, None, self._job(key), "adopted_pid_already_gone_no_restart")
                    continue
                if process.creation_time_100ns != record["creation_time_100ns"]:
                    process.close()
                    raise RuntimeError("Adopted PID creation time mismatch; possible PID reuse")
                self.active[key] = {"process": process, "kind": "adopted"}
                self._set_job(key, {**self._job(key), "status": "running", "adopted": True, "pid": record["pid"],
                    "creation_time_100ns": record["creation_time_100ns"], "command_line": record["command_line"],
                    "adopted_utc": now()})
            for arch, run in self.runs.items():
                for ci in run["original_queue"]["conversations"]:
                    key = (arch, ci)
                    job = self._job(key)
                    if job is None:
                        self.pending.append(key)
                    elif key in adopted:
                        continue
                    elif job.get("status") == "complete":
                        self._validate_finished(key, job.get("exit_code", 0), job, "prior_complete_validated_no_restart")
                    elif job.get("status") == "running":
                        self._validate_finished(key, None, job, "old_running_without_live_adoption_no_restart")
                    elif job.get("status") != "failed":
                        self._set_job(key, {**job, "status": "failed", "prior_status": job.get("status"),
                            "completion_note": "unrecognized_existing_QA_state_no_restart", "no_automatic_retry": True})
            self.pending.sort(key=lambda key: (key[1], list(self.runs).index(key[0])))
            self.global_state["status"] = "running"
            for state in self.states.values():
                state["status"] = "running"
            self.persist()
        except BaseException:
            self.close()
            raise

    def persist(self):
        if len(self.active) > 8:
            raise AssertionError("Global conversation slot budget exceeded")
        self.global_state.update({"updated_utc": now(), "active_count": len(self.active),
            "active_jobs": [f"{arch}/qa/conv_{ci}" for arch, ci in self.active],
            "pending_jobs": [f"{arch}/qa/conv_{ci}" for arch, ci in self.pending],
            "max_observed_active": max(self.global_state["max_observed_active"], len(self.active)),
            "jobs": {f"{arch}/{key}": value for arch, state in self.states.items() for key, value in state["jobs"].items()},
            "runs": {arch: {"run_dir": run["run_dir"], "status": self.states[arch]["status"]} for arch, run in self.runs.items()}})
        self.writer(self.epoch / "global_status.json", self.global_state)
        for arch, state in self.states.items():
            self.writer(Path(self.runs[arch]["run_dir"]) / "queue_status.json", state)

    def start_job(self, key):
        if key in self.active or key in self.started_keys or self._job(key) is not None or len(self.active) >= 8:
            raise RuntimeError("Refusing duplicate or over-capacity QA launch")
        run = self.runs[key[0]]
        log_path = self.epoch / "logs" / f"{key[0]}_qa_conv_{key[1]}.log"
        command = [sys.executable, "-u", run["runner"], "qa", "--run-dir", run["run_dir"],
                   "--conversations", str(key[1]), "--methods", ",".join(run["methods"]), "--workers", "4"]
        self.started_keys.add(key)
        self._set_job(key, {"status": "launching", "log": str(log_path), "command": command,
                           "claimed_utc": now(), "scheduling_epoch": str(self.epoch)})
        self.persist()  # Durable claim precedes spawn; this handoff cannot replay.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("x", encoding="utf-8") as log:
                process = self.popen_factory(command, cwd=self.handoff["root"], stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.active[key] = {"process": ChildProcess(process), "kind": "new"}
            self._set_job(key, {**self._job(key), "status": "running", "pid": process.pid, "started_utc": now(), "adopted": False})
        except Exception as error:
            self._set_job(key, {**self._job(key), "status": "failed", "error_type": type(error).__name__,
                "error": str(error), "completion_note": "spawn_failed_no_retry", "no_automatic_retry": True})
        self.persist()

    def tick(self):
        for key, active in list(self.active.items()):
            process = active["process"]
            if process.is_running():
                continue
            code = process.exit_code()
            process.close()
            del self.active[key]
            self._validate_finished(key, code, self._job(key), active["kind"] + "_worker_exited")
        while self.pending and len(self.active) < 8:
            key = self.pending.pop(0)
            self.start_job(key)
        self.persist()
        return bool(self.active or self.pending)

    def run(self):
        self.initialize()
        try:
            while self.tick():
                self.sleeper(1.0)
            for arch, state in self.states.items():
                # Preserve old failure gates (including retrieve) and defer tag.
                failed = any(job.get("status") == "failed" for job in state["jobs"].values())
                qa_complete = all(state["jobs"].get(f"qa/conv_{ci}", {}).get("status") == "complete" for ci in state["conversations"])
                state["status"] = "complete" if qa_complete and not failed else "requires_review"
                state["finished_utc"] = now()
            self.global_state["status"] = "complete" if all(state["status"] == "complete" for state in self.states.values()) else "requires_review"
            self.global_state["finished_utc"] = now()
            self.persist()
            return 0 if self.global_state["status"] == "complete" else 1
        except BaseException as error:
            self.global_state.update({"status": "requires_review", "supervisor_error": type(error).__name__ + ": " + str(error),
                                      "workers_not_terminated": True})
            for state in self.states.values():
                state["status"] = "requires_review"
            self.persist()
            raise
        finally:
            self.close()

    def close(self):
        for item in self.active.values():
            item["process"].close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    args = parser.parse_args(argv)
    path = Path(args.handoff).resolve()
    handoff = read_json(path)
    mutex = RootMutex(handoff["root"])
    try:
        if (path.parent / "global_status.json").exists():
            raise RuntimeError("This handoff epoch is already used; prepare a new handoff before restart")
        from process_control_windows import WinProcess
        supervisor = Supervisor(path, process_factory=WinProcess)
        return supervisor.run()
    finally:
        mutex.close()


if __name__ == "__main__":
    raise SystemExit(main())
