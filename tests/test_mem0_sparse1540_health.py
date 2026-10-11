"""Offline health decisions; a live process/HTTP200 alone is not completion."""
import copy
import unittest
from tools import check_mem0_sparse1540_health as target


def sample(time, done=0, status="running"):
    return {"schema": target.SCHEMA, "captured_utc": time, "stage": "qa", "manifest_sha256": "same",
        "problems": [], "conversations": {"1": {"state": {"status": status, "done": done,
            "completed": done, "expected": 162, "updated_utc": time},
            "worker": {"pid": 22, "creation_time_100ns": 123, "identity_verified": True},
            "artifacts": {"files": done, "statuses": {"complete": done}, "facts_in_successful_writes": 0,
                          "malformed": 0, "latest_artifact": None},
            "api_log": {"path": "unused", "bytes": 0, "complete_bytes": 0}}}}


def delta(success=0, errors=0, cache=0):
    return lambda *args: {"physical_success": success, "physical_errors": errors, "cache_hits": cache, "bytes_added": 0}


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.first = sample("2026-09-28T09:00:00+00:00", 3)
        self.second = sample("2026-09-28T09:00:30+00:00", 3)

    def test_live_pid_without_progress_requires_investigation(self):
        result = target.compare(self.first, self.second, log_delta=delta())
        self.assertTrue(result["requires_investigation"])

    def test_real_artifact_growth_is_progress(self):
        second = sample("2026-09-28T09:00:30+00:00", 4)
        result = target.compare(self.first, second, log_delta=delta(success=2))
        self.assertFalse(result["requires_investigation"])
        self.assertEqual(result["conversations"][0]["count_growth"]["artifact_success"], 1)

    def test_only_error_calls_are_not_healthy(self):
        result = target.compare(self.first, self.second, log_delta=delta(errors=4))
        self.assertTrue(result["requires_investigation"])
        self.assertIn("only_errors", result["conversations"][0]["verdict"])

    def test_http200_does_not_hide_ten_minute_stall(self):
        history = sample("2026-09-28T08:45:00+00:00", 3)
        result = target.compare(self.first, self.second, log_delta=delta(success=20), historical=history)
        self.assertTrue(result["requires_investigation"])
        self.assertIn("ten_minutes", result["conversations"][0]["verdict"])

    def test_terminal_is_not_automatically_valid(self):
        second = sample("2026-09-28T09:00:30+00:00", 4, "complete")
        result = target.compare(self.first, second, log_delta=delta(success=2))
        self.assertTrue(result["terminal_requires_review"])
        self.assertIn("integrity_and_failure_review", result["conversations"][0]["verdict"])

    def test_worker_identity_mismatch_overrides_http_success(self):
        self.second["conversations"]["1"]["worker"]["identity_verified"] = False
        result = target.compare(self.first, self.second, log_delta=delta(success=20))
        self.assertTrue(result["requires_investigation"])

    def test_new_pool_worker_not_assigned_old_worker_stall(self):
        history = sample("2026-09-28T08:45:00+00:00", 3)
        history["conversations"]["1"]["worker"]["creation_time_100ns"] = 0
        result = target.compare(self.first, self.second, log_delta=delta(success=3), historical=history)
        self.assertFalse(result["requires_investigation"])

    def test_manifest_change_flagged(self):
        self.second["manifest_sha256"] = "changed"
        result = target.compare(self.first, self.second, log_delta=delta(success=3))
        self.assertTrue(result["requires_investigation"])


if __name__ == "__main__":
    unittest.main()
