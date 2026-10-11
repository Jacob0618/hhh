import json
from pathlib import Path
import tempfile
import unittest

from tools.summarize_pilot400_v2 import ledger_usage, v2_metrics


class UsageTests(unittest.TestCase):
    def account(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_calls.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            return ledger_usage(path)

    def row(self, **kwargs):
        return {"request_hash": "a" * 64, "stage": "writer", "provider": "gemini",
                "model": "gemini-2.5-flash", "operation": "generateContent", "status": 200,
                "cache_hit": False, "usage": {"promptTokenCount": 100, "candidatesTokenCount": 20,
                                               "thoughtsTokenCount": 0}, **kwargs}

    def test_cache_reuse_not_billed_again(self):
        one = self.account([self.row()])
        reused = self.account([self.row(), self.row(cache_hit=True)])
        self.assertEqual(reused["physical_attempts"], 1)
        self.assertEqual(reused["cache_hits"], 1)
        self.assertEqual(reused["total_cost_usd"], one["total_cost_usd"])

    def test_unknown_transport_never_cost_zero(self):
        result = self.account([self.row(status="transport_error", usage={})])
        self.assertEqual(result["unknown_cost_attempts"], 1)
        self.assertIsNone(result["total_cost_usd"])

    def test_retried_hash_counts_actual_attempts(self):
        result = self.account([self.row(attempt=1), self.row(attempt=2)])
        self.assertEqual(result["physical_attempts"], 2)

    def test_absent_ledger_is_not_free_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ledger_usage(Path(directory) / "missing.jsonl")

    def test_v2_failed_artifact_is_counted_and_v1_mix_rejected(self):
        metrics = v2_metrics()
        expected = {"conversation": 0, "id": "c0_q0", "category": 1}
        row = {"schema": "jev-mem-pilot400-qa-v2", "arm": "session_untyped_rank",
               "query_id": "c0_q0", "conversation": 0, "category": 1, "cohort_sha256": "a" * 64,
               "status": "failed", "result": None,
               "error": {"type": "ValueError", "stage": "reader", "message": "test failure"}}
        self.assertEqual(metrics.validate_artifact(row, expected, row["arm"], Path('.'), 'a'*64, {}, None), ("failed", None))
        row["schema"] = "jev-mem-pilot400-qa-v1"
        with self.assertRaises(metrics.ArtifactInvalid):
            metrics.validate_artifact(row, expected, row["arm"], Path('.'), 'a'*64, {}, None)
        from tools import summarize_pilot400 as original
        self.assertEqual(original.SCHEMA, "jev-mem-pilot400-qa-v1")
