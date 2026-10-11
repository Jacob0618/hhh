from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests

from tools import summarize_sdk20_modules_recovery as summary
from tests import test_recover_sdk20_modules as fixtures
from tests.test_recover_sdk20_modules import Network, response
from tools import recover_sdk20_modules as recovery


class RecoveryOverlayTests(unittest.TestCase):
    def records(self, *statuses, stage="strict_judge"):
        return [{"stage": stage, "logical_attempt": i, "status": status}
                for i, status in enumerate(statuses, 1)]

    def test_first_valid_or_unknown_cannot_be_followed_by_another_attempt(self):
        for statuses in (("valid", "format_failed"), ("technical_failed", "valid"),
                         ("format_failed", "valid", "valid")):
            with self.assertRaises(summary.original.InvalidArtifact):
                summary.validate_sequence(self.records(*statuses), "strict_judge")
        self.assertEqual(summary.validate_sequence(self.records("format_failed", "valid"), "strict_judge"), "valid")

    def test_uniform_attempt_bound_and_contiguous_sequence(self):
        for statuses in ((), ("format_failed",), ("format_failed", "format_failed"), ("format_failed",) * 4):
            with self.assertRaises(summary.original.InvalidArtifact):
                summary.validate_sequence(self.records(*statuses), "strict_judge")
        self.assertEqual(summary.validate_sequence(self.records(*(["format_failed"] * 3)), "strict_judge"), "format_failed")
        rows = self.records("format_failed", "valid"); rows[1]["logical_attempt"] = 3
        with self.assertRaises(summary.original.InvalidArtifact):
            summary.validate_sequence(rows, "strict_judge")

    def fixture(self):
        questions = [{"id": "c0_q0", "conversation": 0, "category": 4, "reference": "Alice"}]
        slots = {(arm, "c0_q0"): {"status": "failed", "scores": None, "error": {"message": "old"}}
                 for arm in summary.original.ARMS}
        overlays = {key: {"status": "failed", "result": None, "error": {"message": "still failed"}}
                    for key in slots}
        return questions, slots, overlays

    def test_all_original_failed_arms_required_and_success_cannot_be_replaced(self):
        q, slots, overlays = self.fixture()
        incomplete = dict(overlays); incomplete.pop(("baseline_sdk20", "c0_q0"))
        with self.assertRaises(summary.original.InvalidArtifact):
            summary.merge_slots(q, slots, incomplete, lambda *args: 0.)
        slots["baseline_sdk20", "c0_q0"]["status"] = "complete"
        with self.assertRaises(summary.original.InvalidArtifact):
            summary.merge_slots(q, slots, overlays, lambda *args: 0.)

    def test_valid_but_wrong_answer_is_a_completed_recovery_not_a_retry(self):
        q, slots, overlays = self.fixture()
        before = deepcopy(slots)
        overlays["baseline_sdk20", "c0_q0"] = {
            "status": "complete", "error": None,
            "result": {"prediction": {"answer": "Bob"}, "judgment": {"correct": False, "completeness": 0.}, "token_f1": 0.}}
        merged = summary.merge_slots(q, slots, overlays, lambda *args: 0.)
        self.assertEqual(merged["baseline_sdk20", "c0_q0"]["status"], "complete")
        self.assertEqual(merged["baseline_sdk20", "c0_q0"]["scores"]["judge_correct"], 0.)
        self.assertEqual(slots, before)
        self.assertIsNone(merged["preseg_sdk20", "c0_q0"]["scores"])

    def transport_fixture(self, root, *, incomplete=False):
        helper = fixtures.RecoveryTests()
        job = helper.job()
        job.pop("_sha256")
        path = root / "jobs/fact_types_sdk20/c0_q0.json"
        recovery.save_new(path, job)
        job["_sha256"] = recovery.sha(path)
        readers = [response({"answer": "Alice"}, stop="MAX_TOKENS")] if incomplete else [response({"answer": 17}), response({"answer": "Alice"})]
        judges = [] if incomplete else [response({"correct": False, "completeness": 0., "reason": "wrong"})]
        network = Network(readers, judges)
        final = helper.run_case(root, job, network)
        return path, job, final

    def test_real_recorded_mock_transport_replays_format_retry_and_wrong_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path, job, final = self.transport_fixture(root)
            cache = {}; reader = None; judge_sig = None
            records = []
            for pointer in final["attempt_records"]:
                path = Path(pointer["path"])
                saved = recovery.verify(path)
                sig = job["expected_reader_signature"] if saved["stage"] == "qa_reader" else judge_sig
                record, result = summary.audit_attempt(recovery, root, job_path, job, path, sig, "test-protocol", reader, cache)
                records.append(record)
                if record["stage"] == "qa_reader" and result:
                    reader = result["response"]; judge_sig = result["derived"]
            self.assertEqual([r["status"] for r in records], ["format_failed", "valid", "valid"])
            self.assertFalse(records[-1]["result"]["judgment"]["correct"])

    def test_successful_http_incomplete_generation_is_terminal_technical_not_format_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path, job, final = self.transport_fixture(root, incomplete=True)
            path = Path(final["attempt_records"][0]["path"])
            record, result = summary.audit_attempt(recovery, root, job_path, job, path, job["expected_reader_signature"], "test-protocol", None, {})
            self.assertEqual(record["status"], "technical_failed")
            self.assertIsNone(result)
            self.assertIsNotNone(record["raw"])

    def test_forged_attempt_signature_and_model_ledger_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path, job, final = self.transport_fixture(root)
            path = Path(final["attempt_records"][0]["path"])
            sig = deepcopy(job["expected_reader_signature"])
            sig["body"]["generationConfig"]["temperature"] = .7
            with self.assertRaises(summary.original.InvalidArtifact):
                summary.audit_attempt(recovery, root, job_path, job, path, sig, "test-protocol", None, {})
            saved = recovery.verify(path)
            ledger = Path(saved["api_dir"]) / "api_calls.jsonl"
            logs = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
            logs[0]["model"] = "gemini-other"
            ledger.write_text("\n".join(json.dumps(x) for x in logs) + "\n", encoding="utf-8")
            with self.assertRaises(summary.original.InvalidArtifact):
                summary.audit_attempt(recovery, root, job_path, job, path, job["expected_reader_signature"], "test-protocol", None, {})

    def test_network_failure_type_must_match_provider_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); helper = fixtures.RecoveryTests(); job = helper.job()
            job.pop("_sha256")
            job_path = root / "jobs/fact_types_sdk20/c0_q0.json"
            recovery.save_new(job_path, job); job["_sha256"] = recovery.sha(job_path)
            network = Network([requests.Timeout("mock failure")] * 3, [])
            final = helper.run_case(root, job, network)
            path = Path(final["attempt_records"][0]["path"])
            record, value = summary.audit_attempt(recovery, root, job_path, job, path, job["expected_reader_signature"], "test-protocol", None, {})
            self.assertEqual(record["status"], "technical_failed"); self.assertIsNone(value)
            wrong = deepcopy(record); wrong.pop("content_sha256"); wrong["error"]["type"] = "ValueError"
            path.write_text(json.dumps(recovery.old.sealed(wrong)), encoding="utf-8")
            with self.assertRaises(summary.original.InvalidArtifact):
                summary.audit_attempt(recovery, root, job_path, job, path, job["expected_reader_signature"], "test-protocol", None, {})


if __name__ == "__main__":
    unittest.main()
