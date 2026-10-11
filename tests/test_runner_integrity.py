"""Offline tests for durable summaries, shared denominators and stale caches."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("session_runner_under_test", ROOT / "tools/run_session_experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeAPI:
    def __init__(self, *args, **kwargs):
        pass

    def scoped(self, **kwargs):
        return self

    def _redact(self, error):
        return str(error)


class RunnerIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev_runner_integrity_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.conv = self.root / "conv_0"
        self.conv.mkdir()
        self.q = {"id": "c0_q0", "question": "Where did Alice move?", "reference": "Paris", "category": 1}
        self.summary = {"complete": True, "sessions_completed": 1, "sessions_total": 1, "user_id": "conv0"}
        runner.dump_json(self.conv / "ingestion_summary.json", self.summary)
        runner.dump_json(self.conv / "facts.json", [])
        self.facts_hash = runner.hashlib.sha256((self.conv / "facts.json").read_bytes()).hexdigest()

    def retrieval(self, *, partial=False):
        summary = {**self.summary, "complete": not partial}
        facts = [{"id": "fact1", "text": "Alice moved to Paris."}]
        row = {"cache_schema": runner.CACHE_SCHEMA, "query_id": self.q["id"], "question": self.q["question"],
               "sdk20": facts, "pool60": facts, "partial_conversation_smoke": partial,
               "request_fingerprint": runner._retrieval_fingerprint(self.q, summary, self.facts_hash)}
        row["content_fingerprint"] = runner._fingerprint({k: row[k] for k in ("sdk20", "pool60")})
        return row

    def test_method_success_before_later_failure_survives_summary(self):
        runner.dump_json(self.conv / "retrieval/c0_q0.json", self.retrieval())
        score = {"judgment": {"correct": True}, "token_f1": 1.0}
        with patch.object(runner, "ModelAPI", FakeAPI), patch.object(runner, "load_data", return_value=[{}]), \
             patch.object(runner, "session_envelopes", return_value=[{"observation_date": "2023-05-08"}]), \
             patch.object(runner, "queries", return_value=[self.q]), \
             patch.object(runner, "answer_and_judge", return_value=score), \
             patch.object(runner, "select", side_effect=RuntimeError("late selector failure")):
            report = runner.qa_one(self.root, 0, ["sdk20", "rank"], workers=1)
        self.assertEqual(report["methods"]["sdk20"]["completed"], 1)
        self.assertEqual(report["methods"]["sdk20"]["accuracy_all_requested"], 1)
        self.assertEqual(report["methods"]["rank"]["completed"], 0)
        self.assertEqual(report["methods"]["rank"]["requested"], 1)
        self.assertEqual(report["methods"]["rank"]["accuracy_all_requested"], 0)
        self.assertFalse(report["evaluation_complete"])
        self.assertEqual(len(report["errors"]), 1)

    def test_partial_retrieval_cannot_be_reused_after_full_ingestion(self):
        with self.assertRaises(runner.CacheProtocolMismatch):
            runner._verify_retrieval(self.retrieval(partial=True), self.q, self.summary, self.facts_hash)

    def test_retrieval_binds_actual_facts_and_candidate_contents(self):
        row = self.retrieval()
        runner._verify_retrieval(row, self.q, self.summary, self.facts_hash)
        with self.assertRaises(runner.CacheProtocolMismatch):
            runner._verify_retrieval(row, self.q, self.summary, "changed_fact_snapshot")
        row["pool60"][0]["text"] = "tampered"
        with self.assertRaises(runner.CacheProtocolMismatch):
            runner._verify_retrieval(row, self.q, self.summary, self.facts_hash)

    def test_summary_rejects_stale_and_partial_rows(self):
        row = {"cache_schema": runner.CACHE_SCHEMA, "request_fingerprint": "old", "conversation": 0,
               "query_id": "c0_q0", "method": "sdk20", "judgment": {"correct": True}, "token_f1": 1.0,
               "partial_conversation_smoke": False}
        runner.dump_json(self.conv / "qa/sdk20/c0_q0.json", row)
        report = runner.summarize_qa_artifacts(self.conv, 0, [self.q], ["sdk20"],
                                             expected_fingerprints={("c0_q0", "sdk20"): "new"})
        self.assertEqual(report["methods"]["sdk20"]["completed"], 0)
        row.update(request_fingerprint="new", partial_conversation_smoke=True)
        runner.dump_json(self.conv / "qa/sdk20/c0_q0.json", row)
        formal = runner.summarize_qa_artifacts(self.conv, 0, [self.q], ["sdk20"],
                                             expected_fingerprints={("c0_q0", "sdk20"): "new"})
        smoke = runner.summarize_qa_artifacts(self.conv, 0, [self.q], ["sdk20"],
                                             expected_fingerprints={("c0_q0", "sdk20"): "new"}, allow_partial=True)
        self.assertEqual(formal["methods"]["sdk20"]["completed"], 0)
        self.assertEqual(smoke["methods"]["sdk20"]["completed"], 1)
        self.assertTrue(smoke["smoke_only"])

    def test_qa_fingerprint_changes_with_reference_prompt_or_pool(self):
        retrieval = self.retrieval()
        baseline = runner._qa_fingerprint(self.q, "sdk20", retrieval, "2023-05-08")
        changed = runner._qa_fingerprint({**self.q, "reference": "London"}, "sdk20", retrieval, "2023-05-08")
        self.assertNotEqual(baseline, changed)
        with patch.object(runner, "READER_SYSTEM", "a different prompt"):
            self.assertNotEqual(baseline, runner._qa_fingerprint(self.q, "sdk20", retrieval, "2023-05-08"))
        self.assertNotEqual(baseline, runner._qa_fingerprint(self.q, "sdk20",
                            {**retrieval, "content_fingerprint": "different pool"}, "2023-05-08"))

    def test_full_1540_denominator_is_shared_even_when_method_fails(self):
        reports = []
        expected = set()
        for ci, count in enumerate((1000, 540)):
            qids = [f"c{ci}_q{i}" for i in range(count)]
            expected.update(qids)
            reports.append({"conversation": ci, "requested_queries": count, "requested_query_ids": qids,
                            "smoke_only": False, "methods": {
                "sdk20": {"completed": count, "requested": count, "strict_correct": count // 2,
                          "token_f1_sum": count * .6},
                "conditional": {"completed": count - 10, "requested": count, "strict_correct": count // 2,
                                "token_f1_sum": count * .6}}})
        aggregate = runner.aggregate_qa_reports(reports, ["sdk20", "conditional"], expected)
        for method in ("sdk20", "conditional"):
            self.assertEqual(aggregate["methods"][method]["requested"], 1540)
            self.assertEqual(aggregate["methods"][method]["accuracy_all_requested"], .5)
        self.assertEqual(aggregate["methods"]["conditional"]["missing"], 20)
        self.assertFalse(aggregate["formal_full_benchmark_ready"])
        reports[0]["smoke_only"] = True
        self.assertEqual(runner.aggregate_qa_reports(reports, ["sdk20"], expected)["report_scope"], "smoke_only")

    def test_no_division_by_zero_and_no_overlapping_cohorts(self):
        summary = runner.summarize_qa_artifacts(self.conv, 0, [], ["sdk20"], expected_fingerprints={})
        self.assertIsNone(summary["methods"]["sdk20"]["accuracy_all_requested"])
        summary["requested_query_ids"] = ["c0_q0"]
        with self.assertRaises(ValueError):
            runner.aggregate_qa_reports([summary, summary], ["sdk20"], {"c0_q0"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
