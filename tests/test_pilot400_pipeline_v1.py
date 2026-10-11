"""Mock subprocess tests. No real child, provider, credentials, or model request."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_pilot400_pipeline_v1 as pipeline
from jev_mem.providers import ModelAPI


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakePopen:
    def __init__(self, command, owner, **kwargs):
        self.command, self.kwargs, self.owner = command, kwargs, owner
        self.pid = 200 + len(owner.calls)
        self.stage = command[3]
        self.done = False
        self.timeout_emitted = False
        self.returncode = None
        owner.calls.append(self)

    def wait(self, timeout):
        if self.owner.interrupt:
            raise KeyboardInterrupt("mock supervisor interruption")
        if self.owner.wait_once and not self.timeout_emitted:
            self.timeout_emitted = True
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.done = True
        self.returncode = self.owner.exit_codes.get(self.stage, 0)
        if self.owner.write_status:
            row = {"stage": self.stage, "status": "complete", "protocol_sha256": "protocol",
                   "pid": self.pid + 1000, "started_utc": pipeline.now(), "finished_utc": pipeline.now(),
                   "done": 10, "expected": 10, "count_unit": "conversations",
                   "conversations": {str(i): {"status": "complete"} for i in range(10)}}
            row.update(self.owner.status_overrides.get(self.stage, {}))
            write(self.owner.run_dir / (self.stage + "_status.json"), row)
        return self.returncode

    def poll(self):
        return self.returncode


class Factory:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.calls = []
        self.exit_codes = {}
        self.status_overrides = {}
        self.write_status = True
        self.wait_once = False
        self.interrupt = False

    def __call__(self, command, **kwargs):
        if self.calls and not self.calls[-1].done:
            raise AssertionError("Stages overlapped")
        return FakePopen(command, self, **kwargs)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.review = self.root / "review.json"
        self.closure = self.root / "closure.json"
        self.development = {"protocol_sha256": "protocol", "review_sha256": "dev-review"}
        self.gate = {"closure_sha256": "closure", "checked_utc": pipeline.now()}
        self.factory = Factory(self.run_dir)
        self.provider_block = patch.object(ModelAPI, "__init__", side_effect=AssertionError("credentials forbidden"))
        self.provider_block.start()
        self.addCleanup(self.provider_block.stop)

    def invoke(self, **overrides):
        options = {"popen_factory": self.factory, "creation_reader": lambda pid: pid * 100,
                   "driver_identity_reader": lambda pid, launcher, created, command: {
                       "pid": pid, "creation_time_100ns": created + 1, "parent_pid": launcher,
                       "relationship": "mock_direct_venv_worker", "command": command},
                   "review_checker": lambda *_: copy.deepcopy(self.development),
                   "gate_checker": lambda *_: dict(self.gate)}
        options.update(overrides)
        return pipeline.run_pipeline(self.run_dir, self.review, self.closure, **options)

    def test_three_stages_sequential_fixed_argv_and_fresh_status(self):
        self.factory.wait_once = True
        state = self.invoke()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["done"], 3)
        self.assertEqual([p.stage for p in self.factory.calls], ["write", "tag", "qa"])
        for child in self.factory.calls:
            command = child.command
            self.assertEqual(command[0], str(Path(sys.executable).resolve()))
            self.assertEqual(command[2], str(pipeline.DRIVER.resolve()))
            self.assertEqual(command[command.index("--conversations") + 1], "0,1,2,3,4,5,6,7,8,9")
            self.assertEqual(command[command.index("--workers") + 1], "8")
            self.assertEqual(command[command.index("--run-dir") + 1], str(self.run_dir.resolve()))
            self.assertEqual(child.kwargs["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual(child.kwargs["stdin"], subprocess.DEVNULL)
            record = state["stages"][child.stage]
            self.assertEqual(record["child_creation_time_100ns"], child.pid * 100)
            self.assertEqual(record["verified_stage_status"]["done"], 10)
        self.assertFalse((self.run_dir / "driver.lock").exists())

    def test_exit_nonzero_stops_without_tag_or_retry(self):
        self.factory.exit_codes["write"] = 1
        state = self.invoke()
        self.assertEqual(state["status"], "requires_review")
        self.assertEqual(len(self.factory.calls), 1)
        self.assertEqual(state["stages"]["tag"]["status"], "pending")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(len(self.factory.calls), 1)

    def test_exit_zero_but_incomplete_stage_stops(self):
        self.factory.status_overrides["tag"] = {"status": "requires_review", "done": 9}
        state = self.invoke()
        self.assertEqual(state["status"], "requires_review")
        self.assertEqual(state["done"], 1)
        self.assertEqual([p.stage for p in self.factory.calls], ["write", "tag"])

    def test_old_complete_status_cannot_pass_as_new_child(self):
        previous = {"stage": "write", "status": "complete", "protocol_sha256": "protocol",
                    "pid": 123, "started_utc": "2020-01-01T00:00:00+00:00", "finished_utc": "2020-01-01T00:01:00+00:00",
                    "done": 10, "expected": 10, "count_unit": "conversations",
                    "conversations": {str(i): {"status": "complete"} for i in range(10)}}
        write(self.run_dir / "write_status.json", previous)
        self.factory.write_status = False
        state = self.invoke()
        self.assertEqual(state["status"], "requires_review")
        self.assertIn("verified actual driver PID", state["error"]["message"])
        self.assertEqual(len(self.factory.calls), 1)

    def test_gate_rejection_and_driver_lock_never_spawn(self):
        def reject(*_):
            raise ValueError("closure not reviewed")
        with self.assertRaises(ValueError):
            self.invoke(gate_checker=reject)
        self.assertFalse(self.factory.calls)
        (self.run_dir / "driver.lock").write_text("existing owner", encoding="utf8")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual((self.run_dir / "driver.lock").read_text(), "existing owner")
        self.assertFalse(self.factory.calls)

    def test_review_or_closure_change_stops_before_next_stage(self):
        calls = []
        def gate(*_):
            calls.append(1)
            return {"closure_sha256": "closure" if len(calls) <= 2 else "changed"}
        state = self.invoke(gate_checker=gate)
        self.assertEqual(state["status"], "requires_review")
        self.assertEqual(len(self.factory.calls), 1)

    def test_interruption_retains_live_child_identity_for_heartbeat(self):
        self.factory.interrupt = True
        state = self.invoke()
        self.assertEqual(state["status"], "requires_review")
        self.assertTrue(state["active_child_may_still_be_running"])
        self.assertEqual(state["stages"]["write"]["child_pid"], self.factory.calls[0].pid)
        self.assertEqual(state["stages"]["write"]["child_creation_time_100ns"], self.factory.calls[0].pid * 100)
        self.assertEqual(len(self.factory.calls), 1)

    def test_creation_time_error_still_records_child_pid(self):
        def identify(pid):
            if pid != __import__("os").getpid():
                raise ProcessLookupError("mock child identity unavailable")
            return 1234
        state = self.invoke(creation_reader=identify)
        self.assertEqual(state["status"], "requires_review")
        self.assertEqual(state["stages"]["write"]["child_pid"], self.factory.calls[0].pid)
        self.assertTrue(state["active_child_may_still_be_running"])

    def test_check_does_not_create_state_or_spawn(self):
        result = pipeline.check(self.run_dir, self.review, self.closure,
            review_checker=lambda *_: self.development, gate_checker=lambda *_: self.gate)
        self.assertEqual(result["status"], "ready")
        self.assertFalse((self.run_dir / "orchestration_pipeline_v1").exists())
        self.assertFalse(self.factory.calls)

    def test_os_lock_rejects_second_supervisor_and_releases(self):
        with pipeline.SupervisorLock(self.run_dir):
            with self.assertRaises(RuntimeError):
                with pipeline.SupervisorLock(self.run_dir):
                    self.fail("second owner acquired lock")
        with pipeline.SupervisorLock(self.run_dir):
            pass

    def test_unrelated_status_pid_is_rejected_before_following_stage(self):
        def reject(pid, launcher, created, command):
            raise ValueError("PID is unrelated to launcher")
        state = self.invoke(driver_identity_reader=reject)
        self.assertEqual(state["status"], "requires_review")
        self.assertIn("unrelated", state["error"]["message"])
        self.assertEqual(len(self.factory.calls), 1)
        self.assertIn("observed_driver_pid", state["stages"]["write"])

    def test_final_status_pid_and_future_finish_are_rejected(self):
        path = self.run_dir / "write_status.json"
        started = pipeline.now()
        row = {"stage": "write", "status": "complete", "protocol_sha256": "protocol", "pid": 7,
               "started_utc": pipeline.now(), "finished_utc": pipeline.now(), "done": 10, "expected": 10,
               "count_unit": "conversations", "conversations": {str(i): {"status": "complete"} for i in range(10)}}
        write(path, row)
        with self.assertRaises(ValueError):
            pipeline.validate_stage_status(path, "write", "protocol", started, 8, pipeline.now())
        row["finished_utc"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        write(path, row)
        with self.assertRaises(ValueError):
            pipeline.validate_stage_status(path, "write", "protocol", started, 7, pipeline.now())

    def test_direct_pid_identity_uses_launch_creation(self):
        result = pipeline.verify_driver_identity(9, 9, 100, ["python.exe", "-u", "driver.py"])
        self.assertEqual(result["pid"], 9)
        self.assertEqual(result["creation_time_100ns"], 100)

    @unittest.skipUnless(__import__("os").name == "nt", "Windows venv process contract")
    def test_venv_identity_checks_parent_command_and_creation(self):
        command = ["python.exe", "-u", "driver.py", "write"]
        value = {"ProcessId": 11, "ParentProcessId": 10, "CommandLine": "python.exe -u driver.py write"}
        completed = subprocess.CompletedProcess([], 0, json.dumps(value), "")
        with patch.object(pipeline.subprocess, "run", return_value=completed), patch.object(pipeline, "creation_time", return_value=101):
            self.assertEqual(pipeline.verify_driver_identity(11, 10, 100, command)["pid"], 11)
            with self.assertRaises(ValueError):
                pipeline.verify_driver_identity(11, 99, 100, command)
            with self.assertRaises(ValueError):
                pipeline.verify_driver_identity(11, 10, 100, ["python.exe", "-u", "wrong.py", "write"])
        with patch.object(pipeline.subprocess, "run", return_value=completed), patch.object(pipeline, "creation_time", return_value=99):
            with self.assertRaises(ValueError):
                pipeline.verify_driver_identity(11, 10, 100, command)


class DevelopmentReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.manifest = {"protocol_sha256": "protocol", "protocol": {}}
        write(self.run_dir / "run_manifest.json", self.manifest)
        self.report = self.root / "research/review.md"
        self.report.parent.mkdir()
        self.report.write_text("Manual source/date review", encoding="utf8")
        self.path = self.root / "development.json"
        rows = []
        self.banks = {}
        for ci in (0, 1):
            for name in ("session", "regroup"):
                folder = self.run_dir / "conversations" / f"conv_{ci}" / "banks" / name
                for file in ("group_fact_bank.json", "bank_summary.json", "grouping.json", "raw_source.json"):
                    write(folder / file, {"fixture": ci, "bank": name, "file": file})
                bank = {"path": (folder / "group_fact_bank.json").relative_to(self.run_dir).as_posix(),
                        "sha256": pipeline.file_sha(folder / "group_fact_bank.json"),
                        "summary_sha256": pipeline.file_sha(folder / "bank_summary.json")}
                self.banks[ci, name] = bank
                rows.append({"conversation": ci, "bank": name, "bank_path": bank["path"],
                             "bank_sha256": bank["sha256"], "bank_summary_sha256": bank["summary_sha256"],
                             "grouping_sha256": pipeline.file_sha(folder / "grouping.json"),
                             "raw_source_sha256": pipeline.file_sha(folder / "raw_source.json")})
        self.review = {"schema": pipeline.REVIEW_SCHEMA, "status": "approved", "run_dir": str(self.run_dir.resolve()),
                       "protocol_sha256": "protocol", "run_manifest_sha256": pipeline.file_sha(self.run_dir / "run_manifest.json"),
                       "source_dataset_sha256": pipeline.driver.SOURCE_HASH, "cohort_sha256": pipeline.driver.COHORT_SHA256,
                       "development_conversations": [0, 1], "source_dates_reviewed": True, "source_coverage_reviewed": True,
                       "reviewed_by": "mock reviewer", "reviewed_utc": pipeline.now(),
                       "review_path": "research/review.md", "review_sha256": pipeline.file_sha(self.report), "banks": rows}
        write(self.path, self.review)
        self.patches = [patch.object(pipeline, "ROOT", self.root),
                        patch.object(pipeline.driver, "ensure_manifest", return_value=self.manifest),
                        patch.object(pipeline.driver, "load_data", return_value=[{"conversation": {}}] * 2),
                        patch.object(pipeline.driver, "raw_turns_from_locomo", return_value=[{"fixture": "raw"}]),
                        patch.object(pipeline.driver, "load_bank", side_effect=lambda run, ci, name, raw, manifest: self.banks[ci, name]),
                        patch.object(ModelAPI, "__init__", side_effect=AssertionError("credentials forbidden"))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def validate(self):
        return pipeline.validate_development_review(self.run_dir, self.path)

    def test_four_complete_banks_are_bound_without_credentials(self):
        result = self.validate()
        self.assertEqual(len(result["banks"]), 4)
        self.assertEqual(result["protocol_sha256"], "protocol")
        self.assertEqual(pipeline.driver.load_bank.call_count, 4)

    def test_review_flags_and_missing_bank_refused(self):
        for field, value in (("source_dates_reviewed", False), ("source_coverage_reviewed", "true"),
                             ("banks", self.review["banks"][:-1]), ("protocol_sha256", "changed")):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.review)
                changed[field] = value
                write(self.path, changed)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_report_manifest_or_grouping_byte_changes_refused(self):
        for target in (self.report, self.run_dir / "run_manifest.json",
                       self.run_dir / "conversations/conv_0/banks/session/grouping.json"):
            with self.subTest(target=target):
                raw = target.read_bytes()
                target.write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    self.validate()
                target.write_bytes(raw)

    def test_frozen_load_bank_coverage_rejection_propagates(self):
        pipeline.driver.load_bank.side_effect = ValueError("Incomplete exactly-once coverage")
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
