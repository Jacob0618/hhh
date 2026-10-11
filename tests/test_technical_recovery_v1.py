"""Offline recovery tests: all transports are in-memory mocks."""
import copy
import json
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem import recovery_v1 as r
from jev_mem import evaluation
from jev_mem.providers import ModelAPI
from tools import run_technical_recovery_v1 as cli


def response(payload):
    return {"candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"text": json.dumps(payload)}]}}]}


def sig_generate(payload, stage):
    captured = []
    class Capture:
        def request(self, provider, model, body, *, operation="generateContent", **_):
            captured.append({"provider": provider, "model": model, "body": body, "operation": operation})
            return response({})
    ModelAPI.generate(Capture(), json.dumps(payload, ensure_ascii=False),
        system=evaluation.READER_SYSTEM if stage == "qa_reader" else evaluation.JUDGE_SYSTEM,
        model="gemini-2.5-flash" if stage == "qa_reader" else "gemini-3.5-flash-lite",
        json_output=True, max_output=2048, stage=stage)
    return captured[0]


def reader_signature(fact_id="f1"):
    return sig_generate({"question": "Who?", "conversation_reference_date": "2024-01-01",
                         "facts": [{"id": fact_id, "observation_date": None, "fact": "A is here"}]}, "qa_reader")


def jev_signature():
    return {"provider": "jev", "model": "jev-1.13.0", "operation": "systemone",
            "body": {"model": "jev-1.13.0", "state": "state", "questions": {
                "relevant": {"type": "noul", "instructions": "evidence"}}}}


class FakeStore:
    seal_sha256 = "mock-seal"
    denominators = {"mem0": {"rank": 1540}, "independent": {"rank": 1540}}
    def __init__(self, root):
        self.path = root / "original_seal/seal.json"
        self.runs = {"mem0": {"run_dir": str(root / "old_mem0")},
                     "independent": {"run_dir": str(root / "old_independent")}}
        self.failures = []
        self.logs = defaultdict(lambda: defaultdict(list))
        self.responses = {}
        self.cache_locations = defaultdict(list)
        self.index = defaultdict(dict)
        self.records = {}

    def cached(self, signature, stage):
        values = self.responses.get(r.request_hash(signature), [])
        valid, sources = None, []
        for payload in values:
            try:
                r.validate_response(signature, stage, payload)
                source = {"validation": "valid", "response_sha256": r.digest(payload)}
                if valid is None:
                    valid = (payload, len(sources))
            except Exception as e:
                source = {"validation": "invalid", "error": str(e)}
            sources.append(source)
        return valid, sources

    def inputs(self, failure):
        return ({"id": failure["query_id"], "question": "Who?", "reference": "B", "category": 1},
                {"pool60": [{"id": "actual-" + failure["architecture"], "text": "A is here"}]},
                "2024-01-01", {"retrieval_sha256": "mock"})

    def log(self, failure, signature, stage, status=200):
        self.logs[r.identity(failure)][r.request_hash(signature)].append({
            "stage": stage, "provider": signature["provider"], "model": signature["model"],
            "operation": signature["operation"], "source_line": 1, "status": status, "cache_hit": False})


def failure(architecture="mem0", method="rank", qid="c0_q1"):
    return {"architecture": architecture, "conversation": 0, "query_id": qid,
            "method": method, "error": "technical failure"}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = FakeStore(self.root)
        self.proof = {"consumer": failure(), "kind": "same_request_recovery"}

    def test_exact_original_reader_validator_and_incorrect_judge_is_legal(self):
        signature = reader_signature()
        r.validate_response(signature, "qa_reader", response({"answer": "A", "used_fact_ids": ["f1", "f1"]}))
        for payload in ({"answer": []}, {"answer": "A", "used_fact_ids": ["actual-id"]}):
            with self.assertRaises(ValueError):
                r.validate_response(signature, "qa_reader", response(payload))
        judge = sig_generate({"question": "Who?", "reference_answer": "B", "predicted_answer": "A"}, "strict_judge")
        r.validate_response(judge, "strict_judge", response({"correct": False, "completeness": 0, "reason": "wrong"}))
        with self.assertRaises(ValueError):
            r.validate_response(judge, "strict_judge", response({"correct": False, "completeness": True, "reason": "wrong"}))

    def test_same_hash_concurrent_and_restart_one_new_http(self):
        signature = reader_signature()
        calls = []
        def transport(*args):
            calls.append(args)
            time.sleep(.02)
            return response({"answer": "A", "used_fact_ids": ["f1"]})
        attempts = r.AttemptStore(self.root / "output", "seal", transport)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: attempts.obtain(signature, "qa_reader", self.proof, execute=True), range(16)))
        self.assertEqual(len(calls), 1)
        restarted = r.AttemptStore(self.root / "output", "seal", transport)
        restarted.obtain(signature, "qa_reader", self.proof, execute=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(source == "new_http" for _, source in results), 1)

    def test_attempt_started_unknown_is_never_resent(self):
        signature = reader_signature()
        calls = []
        def transport(*args):
            calls.append(args)
            raise RuntimeError("transport uncertain")
        for _ in range(2):
            with self.assertRaises(r.RecoveryStop) as raised:
                r.AttemptStore(self.root / "output", "seal", transport).obtain(signature, "qa_reader", self.proof, execute=True)
            self.assertEqual(raised.exception.status, "unknown")
        self.assertEqual(len(calls), 1)

    def test_preexisting_exclusive_attempt_blocks_without_call(self):
        signature = reader_signature()
        path = self.root / "output/attempts" / r.request_hash(signature) / "attempt_started.json"
        r.exclusive(path, {"request_hash": r.request_hash(signature)})
        with self.assertRaises(r.RecoveryStop):
            r.AttemptStore(self.root / "output", "seal", lambda *_: self.fail("network")).obtain(
                signature, "qa_reader", self.proof, execute=True)

    def test_known_http_failure_not_unknown_and_never_resent(self):
        calls = []
        def transport(*args):
            calls.append(args)
            raise r.RecoveryStop("failed", "gemini HTTP 503")
        attempts = r.AttemptStore(self.root / "output", "seal", transport)
        for _ in range(2):
            with self.assertRaises(r.RecoveryStop) as raised:
                attempts.obtain(reader_signature(), "qa_reader", self.proof, execute=True)
            self.assertEqual(raised.exception.status, "failed")
        self.assertEqual(len(calls), 1)

    def test_new_malformed_response_consumes_attempt_no_second_format_retry(self):
        signature, calls = reader_signature(), []
        def transport(*args):
            calls.append(args)
            return response({"answer": []})
        attempts = r.AttemptStore(self.root / "output", "seal", transport)
        for _ in range(2):
            with self.assertRaises(ValueError):
                attempts.obtain(signature, "qa_reader", self.proof, execute=True)
        self.assertEqual(len(calls), 1)

    def test_salvage_response_after_crash_zero_http(self):
        signature = reader_signature()
        output = self.root / "output"
        r.exclusive(output / "attempts" / r.request_hash(signature) / "attempt_started.json", {})
        r.publish(output / "transport/api_cache" / (r.request_hash(signature) + ".json"), {
            "request": signature, "response": response({"answer": "A", "used_fact_ids": []})})
        result, source = r.AttemptStore(output, "seal", lambda *_: self.fail("network")).obtain(
            signature, "qa_reader", self.proof, execute=True)
        self.assertEqual(source, "overlay_response")

    def test_valid_cache_priority_and_stable_sources_even_correct_false(self):
        signature = sig_generate({"question": "Who?", "reference_answer": "B", "predicted_answer": "A"}, "strict_judge")
        h = r.request_hash(signature)
        self.store.responses[h] = [response({"correct": "bad"}), response({"correct": False, "completeness": 0, "reason": "wrong"}),
                                   response({"correct": True, "completeness": 1, "reason": "right"})]
        api = r.ReplayAPI(self.store, r.AttemptStore(self.root / "output", "seal", lambda *_: self.fail("network")), failure(), True)
        raw = api.request(**signature, stage="strict_judge")
        self.assertFalse(json.loads(raw["candidates"][0]["content"]["parts"][0]["text"])["correct"])
        self.assertEqual(api.events[0]["chosen_source"], 1)

    def test_cross_arch_reader_ids_remapped_and_judge_only_recovery(self):
        reader = reader_signature()
        judge = sig_generate({"question": "Who?", "reference_answer": "B", "predicted_answer": "A"}, "strict_judge")
        self.store.responses[r.request_hash(reader)] = [response({"answer": "A", "used_fact_ids": ["f1"]})]
        self.store.responses[r.request_hash(judge)] = [response({"correct": "bad"})]
        calls = []
        def transport(signature, stage, _):
            calls.append(stage)
            return response({"correct": False, "completeness": 0, "reason": "wrong"})
        attempts = r.AttemptStore(self.root / "output", self.store.seal_sha256, transport)
        for arch in ("mem0", "independent"):
            f = failure(arch)
            self.store.log(f, judge, "strict_judge")
            row = r.replay_consumer(self.store, attempts, f, execute=True)
            self.assertEqual(row["status"], "complete", row)
            self.assertFalse(row["judgment"]["correct"])
            self.assertEqual(row["prediction"]["used_fact_ids"], ["actual-" + arch])
        self.assertEqual(calls, ["strict_judge"])

    def test_bad_reader_stops_before_judge_and_cache_only_never_calls(self):
        signature = reader_signature()
        f = failure()
        self.store.log(f, signature, "qa_reader")
        self.store.responses[r.request_hash(signature)] = [response({"answer": []})]
        calls = []
        def transport(*args):
            calls.append(args[1])
            return response({"answer": []})
        attempts = r.AttemptStore(self.root / "output", self.store.seal_sha256, transport)
        first = r.replay_consumer(self.store, attempts, f, execute=False)
        self.assertEqual(first["status"], "blocked")
        self.assertFalse(calls)
        second = r.replay_consumer(self.store, attempts, f, execute=True)
        self.assertEqual(second["status"], "failed")
        self.assertEqual(calls, ["qa_reader"])

    def test_network_hash_match_then_first_downstream_dependency(self):
        signature = reader_signature()
        f = failure()
        self.store.log(f, signature, "qa_reader", status="transport_error")
        calls = []
        def transport(sig, stage, proof):
            calls.append((stage, proof["kind"]))
            return response({"answer": "A", "used_fact_ids": []}) if stage == "qa_reader" else response({"correct": False, "completeness": 0, "reason": "wrong"})
        row = r.replay_consumer(self.store, r.AttemptStore(self.root / "output", self.store.seal_sha256, transport), f, execute=True)
        self.assertEqual(row["status"], "complete", row)
        self.assertEqual(calls, [("qa_reader", "same_request_recovery"), ("strict_judge", "first_time_dependency_completion")])

    def test_unbound_request_and_stage_mismatch_refuse(self):
        signature = reader_signature()
        f = failure()
        attempts = r.AttemptStore(self.root / "output", self.store.seal_sha256, lambda *_: self.fail("network"))
        api = r.ReplayAPI(self.store, attempts, f, True)
        with self.assertRaises(r.RecoveryStop) as raised:
            api.request(**signature, stage="qa_reader")
        self.assertEqual(raised.exception.status, "unbound")
        self.store.log(f, signature, "strict_judge")
        with self.assertRaises(r.RecoveryStop):
            r.ReplayAPI(self.store, attempts, f, True).request(**signature, stage="qa_reader")

    def test_original_generation_hash_drift_rejected_even_cache_hit(self):
        f = failure()
        original = reader_signature()
        changed = copy.deepcopy(original)
        changed["body"]["contents"][0]["parts"][0]["text"] = changed["body"]["contents"][0]["parts"][0]["text"].replace("Who?", "Where?")
        self.store.log(f, original, "qa_reader", status="transport_error")
        for cached in (False, True):
            if cached:
                self.store.responses[r.request_hash(changed)] = [response({"answer": "A", "used_fact_ids": []})]
            api = r.ReplayAPI(self.store, r.AttemptStore(self.root / "output", "seal", lambda *_: self.fail("network")), f, True)
            api.events.append({"request_hash": "predecessor", "status": "valid"})
            with self.assertRaises(r.RecoveryStop) as raised:
                api.request(**changed, stage="qa_reader")
            self.assertEqual(raised.exception.status, "unbound")

    def test_conditional_frozen_full_boundary_and_twenty_selected(self):
        f = failure(method="conditional")
        class LegalStore(FakeStore):
            def cached(self, signature, stage):
                if stage == "jev_score":
                    key = next(iter(signature["body"]["questions"]))
                    return ({"answers": {key: {"noul": .8}}}, 0), [{"validation": "valid"}]
                value = {"answer": "A", "used_fact_ids": []} if stage == "qa_reader" else {"correct": False, "completeness": 0, "reason": "wrong"}
                return (response(value), 0), [{"validation": "valid"}]
            def inputs(self, failure):
                q, _, date, binding = super().inputs(failure)
                return q, {"pool60": [{"id": str(i), "text": "Fact " + str(i)} for i in range(60)]}, date, binding
        store = LegalStore(self.root)
        row = r.replay_consumer(store, r.AttemptStore(self.root / "output", store.seal_sha256), f)
        self.assertEqual(row["status"], "complete", row)
        self.assertEqual(len(row["selected_ids"]), 20)
        self.assertEqual(sum(e["stage"] == "jev_score" for e in row["request_events"]), 124)

    def test_run_preserves_failure_and_denominator_and_restart_idempotent(self):
        self.store.failures = [failure()]
        output = self.root / "output"
        with patch.object(ModelAPI, "__init__", side_effect=AssertionError("credentials")):
            first = r.run(self.store, output, "cache-only", 1)
            second = r.run(self.store, output, "cache-only", 1)
        self.assertEqual(first["done"], 1)
        self.assertEqual(second["by_status"], {"unbound": 1})
        self.assertEqual(second["denominators"]["mem0"]["rank"], 1540)
        with self.assertRaises(ValueError):
            r.run(self.store, output, "cache-only", 9)
        row = r.read_json(output / "cache_only/consumers/mem0/conv_0/rank/c0_q1.json")
        self.assertEqual(row["row_sha256"], r.digest({k: v for k, v in row.items() if k != "row_sha256"}))

    def test_review_gate_rejects_code_or_input_mismatch_without_credentials(self):
        output = self.root / "output"
        plan = {"seal_sha256": self.store.seal_sha256, "recovery_code_sha256": r.code_hashes(), "consumers": [], "output_dir": str(output.resolve())}
        r.publish(output / "plan/plan.json", plan)
        review = {"schema": "jev-mem-technical-recovery-review-v1", "approved": True,
                  "seal_sha256": self.store.seal_sha256, "plan_sha256": r.sha((output / "plan/plan.json").read_bytes()),
                  "recovery_code_sha256": r.code_hashes()}
        path = self.root / "review.json"
        r.publish(path, review)
        r.verify_review(path, self.store, output)
        review["recovery_code_sha256"]["jev_mem/recovery_v1.py"] = "wrong"
        r.publish(path, review)
        with self.assertRaises(ValueError):
            r.verify_review(path, self.store, output)

    def test_cli_cache_only_does_not_initialize_provider(self):
        with patch.object(r, "SealStore", return_value=self.store), patch.object(r, "run", return_value={"status": "complete"}) as run, \
             patch.object(ModelAPI, "__init__", side_effect=AssertionError("credentials")):
            cli.main(["cache-only", "--output", str(self.root / "output")])
            self.assertIsNone(run.call_args.kwargs["transport"])

    def test_original_cached_signature_and_byte_mutation_rejected(self):
        signature = reader_signature()
        original = self.root / "original"
        path = original / "conv_0/api_cache" / (r.request_hash(signature) + ".json")
        record = {"request": signature, "response": response({"answer": "A", "used_fact_ids": []})}
        r.publish(path, record)
        store = r.SealStore.__new__(r.SealStore)
        relative = path.relative_to(original).as_posix()
        store.runs = {"mem0": {"run_dir": str(original)}}
        store.index = {"mem0": {relative: {"sha256": r.sha(path.read_bytes()), "bytes": path.stat().st_size}}}
        store.cache_locations = {r.request_hash(signature): [("mem0", relative)]}
        self.assertIsNotNone(store.cached(signature, "qa_reader")[0])
        previous = path.read_bytes()
        path.write_bytes(previous + b" ")
        with self.assertRaises(ValueError):
            store.cached(signature, "qa_reader")
        path.write_bytes(previous)
        changed = copy.deepcopy(signature)
        changed["model"] = "wrong-model"
        store.cache_locations[r.request_hash(changed)] = [("mem0", relative)]
        with self.assertRaises(ValueError):
            store.cached(changed, "qa_reader")
        self.assertEqual(path.read_bytes(), previous)

    def test_namespace_lock_rejects_second_supervisor_then_releases(self):
        with r.NamespaceLock(self.root):
            with self.assertRaises(RuntimeError):
                with r.NamespaceLock(self.root):
                    self.fail("second supervisor entered")
        with r.NamespaceLock(self.root):
            pass


if __name__ == "__main__":
    unittest.main()
