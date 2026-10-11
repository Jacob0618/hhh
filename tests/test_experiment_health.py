import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("health", Path(__file__).resolve().parents[1] / "tools/check_experiment_health.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HealthTests(unittest.TestCase):
    def row(self, count=3, alive=True):
        return {"job_status": "running", "qa_total": count, "worker": {"alive": alive}}

    def test_static_percentage_and_live_pid_are_insufficient(self):
        self.assertIn("needs_investigation", module.classify_progress(self.row(), self.row(), {"rows": 0}))

    def test_error_only_or_cache_only_activity_does_not_pass(self):
        for delta in ({"rows": 10, "http200": 0}, {"rows": 10, "cache_hits": 10}):
            self.assertIn("needs_investigation", module.classify_progress(self.row(), self.row(), delta))

    def test_actual_qa_or_success_delta_passes(self):
        self.assertEqual(module.classify_progress(self.row(), self.row(4), {}), "confirmed_progress")
        self.assertEqual(module.classify_progress(self.row(), self.row(), {"http200": 2}), "confirmed_progress")

    def test_unverified_worker_and_regressing_artifacts_fail(self):
        self.assertIn("needs_investigation", module.classify_progress(self.row(), self.row(alive=False), {"http200": 2}))
        self.assertIn("needs_investigation", module.classify_progress(self.row(), self.row(2), {"http200": 2}))

    def test_shared_read_closes_handle_before_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"state": 1}', encoding="utf-8")
            self.assertEqual(module.read_json(path), {"state": 1})
            replacement = Path(directory) / "next.json"
            replacement.write_text('{"state": 2}', encoding="utf-8")
            replacement.replace(path)
            self.assertEqual(module.read_json(path), {"state": 2})

    def test_long_stall_cannot_pass_on_http_activity(self):
        def snap(timestamp, count):
            row = self.row(count)
            row["worker"].update(pid=42, creation_time_100ns=100)
            return {"captured_utc": timestamp, "runs": {"mem0": {"conversations": {"conv_1": row}}}}
        old = snap("2026-09-27T20:00:00+00:00", 68)
        stuck = snap("2026-09-27T20:20:00+00:00", 68)
        good = snap("2026-09-27T20:20:00+00:00", 69)
        self.assertEqual(len(module.long_window_check(old, stuck)["unresolved_jobs"]), 1)
        self.assertEqual(module.long_window_check(old, good)["unresolved_jobs"], [])

    def test_long_stall_not_attributed_to_a_replaced_worker(self):
        def snap(timestamp, pid):
            row = self.row(68)
            row["worker"].update(pid=pid, creation_time_100ns=pid)
            return {"captured_utc": timestamp, "runs": {"mem0": {"conversations": {"conv_1": row}}}}
        self.assertEqual(module.long_window_check(snap("2026-09-27T20:00:00+00:00", 42),
                                                  snap("2026-09-27T20:20:00+00:00", 43))["checked_jobs"], [])


if __name__ == "__main__":
    unittest.main()
