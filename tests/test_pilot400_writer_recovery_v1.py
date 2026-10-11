"""Temporary-directory tests only: no model, credential, original-run or network use."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import recover_pilot400_writer_v1 as recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeAPI:
    def __init__(self, owner, destination, **kwargs):
        self.owner, self.destination = owner, Path(destination)
        owner.assertEqual(kwargs["retries"], 0)
        owner.assertEqual(kwargs["timeout"], 150)
        owner.factory_calls += 1

    def request(self, provider, model, body, *, operation, stage):
        owner = self.owner
        owner.calls += 1
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        owner.assertEqual(signature, owner.signature)
        owner.assertTrue((self.destination / "recovery_control/attempt_started.json").exists())
        row = {"request_hash": owner.request_hash, "cache_hit": False, "attempt": 1,
               "status": "transport_error" if owner.transport_error else 200}
        (self.destination / "api_calls.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        if owner.transport_error:
            raise RuntimeError("mock transport error")
        raw = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": json.dumps(owner.replacement)}]}}], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
        write(self.destination / "api_cache" / (owner.request_hash + ".json"),
              {"request": signature, "response": raw, "stage": stage})
        return raw


class WriterRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "original"
        self.destination = Path(self.temp.name) / "recovery"
        self.source.mkdir()
        member = {"source_turn_id": "m1", "session_id": "s1", "speaker": "A", "observation_date": "2022-01-01", "text": "A fact"}
        context = {**member, "source_turn_id": "c1", "text": "Unrelated context"}
        self.original_response = {"facts": [{"text": "bad context citation", "source_turn_ids": ["m1", "c1"], "temporal_status": "not_temporal"}]}
        self.pending = {"group_id": "sparse_topic_0004", "user_id": "pilot400_conv_1", "grouping_method": "jev_topics",
                        "members": [member], "read_only_context": [context], "member_context_ids": {"m1": []},
                        "member_source_turn_ids": ["m1"], "context_source_turn_ids": ["c1"],
                        "request_sha256": "groupmock", "prompt": "the exact original prompt",
                        "system_prompt": recovery.SYSTEM_PROMPT, "raw_response": self.original_response,
                        "status": "failed_retry_same_request", "error_type": "GroupWriteError", "context_cost": {"prompt_text_tokens": 20}}
        self.signature = recovery.signature_from_pending(self.pending)
        self.request_hash = recovery.http_hash(self.signature)
        write(self.source / "run_manifest.json", {"protocol": "immutable"})
        write(self.source / recovery.TARGET / "pending_group.json", self.pending)
        write(self.source / recovery.TARGET / "bank_summary.json", {"status": "failed"})
        write(self.source / recovery.TARGET / "group_fact_bank.json", {"groups": ["committed"], "facts": ["fact"]})
        write(self.source / "conversations/conv_0/banks/session/bank_summary.json", {"status": "complete"})
        write(self.source / "api_cache" / (self.request_hash + ".json"), {"request": self.signature, "invalid": True})
        write(self.source / "api_cache/valid.json", {"request": "valid_mock"})
        write(self.source / "write_status.json", {"status": "requires_review"})
        (self.source / "api_calls.jsonl").write_text('{"original":"ledger"}\n', encoding="utf-8")
        self.original_seal = recovery.seal_tree(self.source)
        self.calls = self.factory_calls = 0
        self.transport_error = False
        self.replacement = {"facts": [{"text": "Valid member fact", "source_turn_ids": ["m1"], "temporal_status": "not_temporal"}]}
        self.block = patch.object(recovery.ModelAPI, "__init__", side_effect=AssertionError("Real credentials/network forbidden"))
        self.block.start()
        self.addCleanup(self.block.stop)

    def audited(self, source):
        self.assertEqual(source, self.source)
        return {"files": recovery.seal_tree(source), "protocol_sha256": "protocol", "signature": self.signature,
                "request_hash": self.request_hash, "group_request_hash": "groupmock"}

    def prepare(self):
        return recovery.prepare(self.source, self.destination, source_auditor=self.audited)

    def factory(self, *args, **kwargs):
        return FakeAPI(self, *args, **kwargs)

    def test_prepare_excludes_three_invalid_artifacts_and_old_active_logs(self):
        result = self.prepare()
        self.assertEqual(result["status"], "prepared_blocked")
        self.assertEqual(recovery.seal_tree(self.source), self.original_seal)
        for relative in (recovery.TARGET + "/pending_group.json", recovery.TARGET + "/bank_summary.json",
                         "api_cache/" + self.request_hash + ".json", "api_calls.jsonl", "write_status.json"):
            self.assertFalse((self.destination / relative).exists())
        self.assertTrue((self.destination / "driver.lock").exists())
        lineage = recovery.read(self.destination / "recovery_control/lineage.json")
        self.assertEqual(lineage["source_files"], self.original_seal)
        for relative, record in lineage["copied_files"].items():
            self.assertEqual(recovery.sha(self.destination / relative), record["sha256"])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_one_valid_attempt_preserves_original_and_replaces_only_pending(self):
        self.prepare()
        out = recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(out["status"], "ready_for_frozen_driver")
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.factory_calls, 1)
        self.assertFalse((self.destination / "driver.lock").exists())
        pending = recovery.read(self.destination / recovery.TARGET / "pending_group.json")
        self.assertEqual(pending["raw_response"], self.replacement)
        self.assertEqual(pending["request_sha256"], self.pending["request_sha256"])
        self.assertEqual(pending["prompt"], self.pending["prompt"])
        self.assertEqual(pending["context_cost"], self.pending["context_cost"])
        self.assertNotIn("error_type", pending)
        self.assertEqual(recovery.seal_tree(self.source), self.original_seal)
        with self.assertRaises((ValueError, FileNotFoundError)):
            recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(self.calls, 1)

    def test_invalid_replacement_retains_guard_and_never_retries(self):
        self.prepare()
        self.replacement = copy.deepcopy(self.original_response)
        out = recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(out["status"], "failed_validation")
        self.assertIn("not associated", out["error"])
        self.assertTrue((self.destination / "driver.lock").exists())
        self.assertFalse((self.destination / recovery.TARGET / "pending_group.json").exists())
        with self.assertRaisesRegex(ValueError, "allowance already consumed"):
            recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(self.calls, 1)

    def test_transport_unknown_is_not_zero_cost_or_reattemptable(self):
        self.prepare()
        self.transport_error = True
        out = recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(out["status"], "unknown")
        self.assertTrue(out["usage_unknown"])
        self.assertIsNone(out["cost_total_usd"])
        self.assertTrue((self.destination / "driver.lock").exists())
        with self.assertRaisesRegex(ValueError, "allowance already consumed"):
            recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(self.calls, 1)

    def test_source_mutation_blocks_before_api(self):
        self.prepare()
        write(self.source / "run_manifest.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Original frozen bytes changed"):
            recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(self.factory_calls, 0)

    def test_actual_tag_and_qa_stage_artifacts_reject_write_only_scope(self):
        write(self.source / "write_status.json", {"status": "requires_review", "done": 2,
                                                 "conversations": {"0": {}, "1": {}}})
        for relative in ("conversations/conv_0/tag_summary.json", "conversations/conv_1/qa_status.json"):
            path = self.source / relative
            write(path, {"status": "complete"})
            with self.assertRaisesRegex(ValueError, "pre-QA/pre-tag"):
                recovery.inspect_source(self.source)
            path.unlink()

    def test_copied_success_mutation_blocks_before_api(self):
        self.prepare()
        write(self.destination / recovery.TARGET / "group_fact_bank.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Copied artifact changed"):
            recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(self.factory_calls, 0)

    def test_os_operation_lock_blocks_and_releases_without_deleting_file(self):
        self.prepare()
        lock = self.destination / "recovery_control/operation.lock"
        with recovery.namespace_lock(self.destination):
            with self.assertRaisesRegex(RuntimeError, "owns this namespace"):
                recovery.retry(self.destination, api_factory=self.factory)
        self.assertTrue(lock.exists())
        with recovery.namespace_lock(self.destination):
            pass
        self.assertEqual(self.calls, 0)

    def test_offline_finalize_after_response_persisted_never_calls_api(self):
        self.prepare()
        write(self.destination / "recovery_control/attempt_started.json", {"request_hash": self.request_hash,
              "max_http_attempts": 1, "started_utc": recovery.now()})
        FakeAPI(self, self.destination, retries=0, timeout=150).request(
            self.signature["provider"], self.signature["model"], self.signature["body"],
            operation=self.signature["operation"], stage="group_fact_extraction")
        calls = self.calls
        out = recovery.finalize(self.destination)
        self.assertEqual(out["status"], "ready_for_frozen_driver")
        self.assertEqual(self.calls, calls)

    def test_duplicate_physical_log_rejected_before_driver_release(self):
        self.prepare()
        write(self.destination / "recovery_control/attempt_started.json", {"request_hash": self.request_hash,
              "max_http_attempts": 1, "started_utc": recovery.now()})
        FakeAPI(self, self.destination, retries=0, timeout=150).request(
            self.signature["provider"], self.signature["model"], self.signature["body"],
            operation=self.signature["operation"], stage="group_fact_extraction")
        log = self.destination / "api_calls.jsonl"
        log.write_text(log.read_text(encoding="utf-8") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "at most one"):
            recovery.finalize(self.destination)
        self.assertTrue((self.destination / "driver.lock").exists())

    def test_finalize_after_pending_publication_is_offline_and_idempotent(self):
        self.prepare()
        real_write = recovery._write_new
        def interrupted(path, value):
            if Path(path).name == "outcome.json":
                raise KeyboardInterrupt("mock crash after pending publication")
            return real_write(path, value)
        with patch.object(recovery, "_write_new", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                recovery.retry(self.destination, api_factory=self.factory)
        self.assertEqual(self.calls, 1)
        self.assertTrue((self.destination / recovery.TARGET / "pending_group.json").exists())
        self.assertTrue((self.destination / "driver.lock").exists())
        out = recovery.finalize(self.destination)
        self.assertEqual(out["status"], "ready_for_frozen_driver")
        self.assertEqual(self.calls, 1)
        self.assertEqual(recovery.finalize(self.destination), out)

    def test_finalize_after_outcome_publication_validates_before_release(self):
        self.prepare()
        real_unlink = Path.unlink
        def interrupted(path, *args, **kwargs):
            if path == self.destination / "driver.lock":
                raise KeyboardInterrupt("mock crash before guard release")
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                recovery.retry(self.destination, api_factory=self.factory)
        out = recovery.finalize(self.destination)
        self.assertEqual(out["status"], "ready_for_frozen_driver")
        self.assertEqual(self.calls, 1)
        self.assertFalse((self.destination / "driver.lock").exists())

    def test_corrupt_interrupted_pending_is_not_repaired_or_released(self):
        self.prepare()
        real_write = recovery._write_new
        def interrupted(path, value):
            if Path(path).name == "outcome.json":
                raise KeyboardInterrupt("mock crash")
            return real_write(path, value)
        with patch.object(recovery, "_write_new", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                recovery.retry(self.destination, api_factory=self.factory)
        pending = self.destination / recovery.TARGET / "pending_group.json"
        value = recovery.read(pending)
        value["raw_response"]["facts"][0]["text"] = "tampered"
        write(pending, value)
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            recovery.finalize(self.destination)
        self.assertTrue((self.destination / "driver.lock").exists())
        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
