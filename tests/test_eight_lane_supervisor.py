"""Offline supervisor tests: no real processes, model clients, or experiment writes."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eight_lane_supervisor", ROOT / "tools/run_eight_lane_supervisor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeProcess:
    def __init__(self, pid, creation=123):
        self.pid = pid
        self.creation_time_100ns = creation
        self.running = True
        self.closed = False

    def is_running(self):
        return self.running

    def exit_code(self):
        return 0

    def close(self):
        self.closed = True


class FakeChild:
    def __init__(self, pid):
        self.pid = pid
        self.running = True
        self.waited = False

    def poll(self):
        return None if self.running else 0

    def wait(self):
        if self.running:
            raise AssertionError("wait called before the child finished")
        self.waited = True
        return 0


class Fixtures:
    def __init__(self, folder, *, adopted_per_arch=0, count=5, failed=()):
        self.folder = Path(folder)
        self.adopted = {}
        self.children = []
        self.commands = []
        self.open_calls = []
        runs, adopted = [], []
        for arch in ("mem0", "independent"):
            run_dir = self.folder / arch
            run_dir.mkdir()
            methods = ["rank", "conditional"]
            runner = ROOT / "tools" / ("run_session_experiments.py" if arch == "mem0" else "run_independent_experiments.py")
            jobs = {f"retrieve/conv_{ci}": {"status": "complete", "log": "old_retrieval"} for ci in range(count)}
            for ci in failed:
                jobs[f"qa/conv_{ci}"] = {"status": "failed", "exit_code": 1, "log": "old_failed"}
            for ci in range(adopted_per_arch):
                pid = 100 + len(adopted)
                log = run_dir / f"old_conv_{ci}.log"
                log.write_text("original worker owns this log", encoding="utf-8")
                jobs[f"qa/conv_{ci}"] = {"status": "running", "log": str(log)}
                command = ["python.exe", "-u", str(runner), "qa", "--run-dir", str(run_dir), "--conversations", str(ci), "--methods", ",".join(methods), "--workers", "4"]
                adopted.append({"architecture": arch, "conversation": ci, "pid": pid,
                    "creation_time_100ns": 123, "command_line": subprocess.list2cmdline(command), "log": str(log)})
                self.adopted[pid] = FakeProcess(pid)
            original = {"architecture": arch, "pid": 11, "status": "running", "conversations": list(range(count)),
                        "methods": methods, "jobs": jobs, "source_hashes": {"frozen": "same"}, "stages": ["retrieve", "qa", "tag"]}
            runs.append({"architecture": arch, "run_dir": str(run_dir), "runner": str(runner), "methods": methods, "original_queue": original})
        self.handoff = {"schema": module.HANDOFF_SCHEMA, "status": "ready_for_eight_lane_supervisor",
            "root": str(ROOT), "created_utc": "2026-09-28T00:00:00Z", "global_slots": 8, "query_workers": 4,
            "runs": runs, "adopted": adopted}
        self.path = self.folder / "epoch/handoff.json"
        module.atomic_json(self.path, self.handoff)

    def factory(self, pid, expected_creation=None):
        self.open_calls.append((pid, expected_creation))
        return self.adopted[pid]

    def popen(self, command, **kwargs):
        child = FakeChild(1000 + len(self.children))
        self.children.append(child)
        self.commands.append(command)
        self.assert_kwargs = kwargs
        return child

    def supervisor(self, **overrides):
        args = {"process_factory": self.factory, "popen_factory": self.popen,
                "summary_validator": lambda *a: {"valid": True, "errors": []}, "sleeper": lambda _: None}
        args.update(overrides)
        return module.Supervisor(self.path, **args)


class SupervisorTests(unittest.TestCase):
    def test_six_adopted_workers_fill_only_two_of_eight_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixtures(directory, adopted_per_arch=3)
            original = copy.deepcopy(fixture.handoff)
            supervisor = fixture.supervisor()
            supervisor.initialize()
            supervisor.tick()
            self.assertEqual(len(supervisor.active), 8)
            self.assertEqual(len(fixture.children), 2)
            self.assertEqual(supervisor.global_state["max_observed_active"], 8)
            self.assertEqual(supervisor.started_keys, {("mem0", 3), ("independent", 3)})
            self.assertTrue(all(expected == 123 for _, expected in fixture.open_calls))
            self.assertEqual(module.read_json(fixture.path), original)
            for record in fixture.handoff["adopted"]:
                self.assertEqual(Path(record["log"]).read_text(), "original worker owns this log")
            supervisor.close()

    def test_architecture_conversation_dedup_and_no_old_failure_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixtures(directory, count=6, failed=(0,))
            supervisor = fixture.supervisor()
            supervisor.initialize()
            supervisor.tick()
            self.assertEqual(len(fixture.children), 8)
            self.assertFalse(any(key[1] == 0 for key in supervisor.started_keys))
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                supervisor.start_job(("mem0", 1))
            for child in fixture.children:
                child.running = False
            supervisor.tick()
            self.assertEqual(len(fixture.children), 10)
            self.assertTrue(all(child.waited for child in fixture.children[:8]))
            self.assertEqual(len(supervisor.started_keys), 10)
            self.assertEqual(supervisor.states["mem0"]["jobs"]["qa/conv_0"]["status"], "failed")
            self.assertEqual(supervisor.states["mem0"]["source_hashes"], {"frozen": "same"})
            supervisor.close()

    def test_duplicate_adopted_arch_conv_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixtures(directory, adopted_per_arch=1)
            fixture.handoff["adopted"].append(copy.deepcopy(fixture.handoff["adopted"][0]))
            module.atomic_json(fixture.path, fixture.handoff)
            with self.assertRaisesRegex(ValueError, "Duplicate adopted"):
                fixture.supervisor()
            self.assertEqual(fixture.children, [])

    def test_pid_creation_mismatch_cannot_free_a_slot_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixtures(directory, adopted_per_arch=1)
            next(iter(fixture.adopted.values())).creation_time_100ns = 999
            supervisor = fixture.supervisor()
            with self.assertRaisesRegex(RuntimeError, "creation time mismatch"):
                supervisor.initialize()
            self.assertEqual(fixture.children, [])
            with self.assertRaisesRegex(RuntimeError, "reused"):
                fixture.supervisor(process_factory=lambda *a, **kw: (_ for _ in ()).throw(ProcessLookupError("PID was reused: 1"))).initialize()

    def test_gone_adopted_worker_without_summary_fails_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixtures(directory, adopted_per_arch=1)
            supervisor = fixture.supervisor(process_factory=lambda *a, **kw: (_ for _ in ()).throw(ProcessLookupError(1)),
                summary_validator=lambda *a: {"valid": False, "errors": ["missing summary"]})
            supervisor.initialize()
            self.assertEqual(supervisor.states["mem0"]["jobs"]["qa/conv_0"]["status"], "failed")
            self.assertNotIn(("mem0", 0), supervisor.pending)
            supervisor.tick()
            self.assertNotIn(("mem0", 0), supervisor.started_keys)
            supervisor.close()

    def test_ready_gate_and_adopted_command_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixtures(directory, adopted_per_arch=1)
            fixture.handoff["status"] = "sealed_before_scheduler_stop"
            module.atomic_json(fixture.path, fixture.handoff)
            with self.assertRaisesRegex(ValueError, "not ready"):
                fixture.supervisor()
            fixture.handoff["status"] = "ready_for_eight_lane_supervisor"
            fixture.handoff["adopted"][0]["command_line"] = fixture.handoff["adopted"][0]["command_line"].replace("--workers 4", "--workers 8")
            module.atomic_json(fixture.path, fixture.handoff)
            with self.assertRaisesRegex(ValueError, "argument mismatch"):
                fixture.supervisor()

    def test_summary_missing_question_or_file_identity_fails(self):
        questions = [{"id": "c0_q0", "question": "One?"}, {"id": "c0_q1", "question": "Two?"}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "conv_0"
            summary = {"conversation": 0, "requested_queries": 2, "requested_query_ids": ["c0_q0", "c0_q1"],
                       "methods": {"rank": {"requested": 2, "completed": 2}}, "evaluation_complete": True}
            module.atomic_json(folder / "qa_summary.json", summary)
            for question in questions:
                module.atomic_json(folder / "qa/rank" / (question["id"] + ".json"), {"query_id": question["id"],
                    "question": question["question"], "conversation": 0, "method": "rank", "partial_conversation_smoke": False,
                    "judgment": {"correct": True}})
            validate = lambda: module.validate_summary(directory, 0, ["rank"], lambda _: questions)
            self.assertTrue(validate()["valid"])
            summary["requested_query_ids"] = ["c0_q0"]
            module.atomic_json(folder / "qa_summary.json", summary)
            self.assertFalse(validate()["valid"])
            summary["requested_query_ids"] = ["c0_q0", "c0_q1"]
            module.atomic_json(folder / "qa_summary.json", summary)
            path = folder / "qa/rank/c0_q1.json"
            row = module.read_json(path)
            row["query_id"] = "c0_q0"
            module.atomic_json(path, row)
            self.assertFalse(validate()["valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
