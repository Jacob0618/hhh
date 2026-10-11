"""No real providers/credentials: original evaluator fidelity and durable guards."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from jev_mem import mem0_transfer_qa as target
from tools.run_pilot400_experiments_v2 import publish_json, read, file_sha


def facts():
    return [{"id": "actual_" + str(i), "text": "Fact " + str(i), "observation_date": "2023-01-01"}
            for i in range(20)]


class FakeBase:
    def __init__(self, path, retries=2, shared=None, context=None):
        self.run_dir = Path(path)
        self.retries = retries
        self.context = context or {"conversation": 1, "query_id": "c1_q1", "method": "rank", "phase": "qa"}
        self.shared = shared if shared is not None else {"calls": [], "ledger": [], "transport_failure": False,
            "reader": {"answer": "an answer", "used_fact_ids": ["f1"]},
            "judge": {"correct": True, "completeness": 1, "reason": "Equivalent."}, "finish": "STOP"}

    def scoped(self, **context):
        return FakeBase(self.run_dir, self.retries, self.shared, {**self.context, **context})

    def request(self, provider, model, body, *, stage):
        signature = {"provider": provider, "model": model, "operation": "generateContent", "body": body}
        digest = target.request_hash(signature)
        path = self.run_dir / "api_cache" / (digest + ".json")
        if path.exists():
            self.shared["ledger"].append({"hash": digest, "cache": True})
            return read(path)["response"]
        self.shared["calls"].append({"signature": copy.deepcopy(signature), "stage": stage})
        if self.shared["transport_failure"]:
            raise RuntimeError("Fake unknown transport outcome")
        time.sleep(.002)
        value = self.shared["reader"] if stage == "qa_reader" else self.shared["judge"]
        response = {"candidates": [{"finishReason": self.shared["finish"], "content": {"parts": [
            {"text": json.dumps(value)}]}}]}
        publish_json(path, {"request": signature, "stage": stage, "response": response})
        self.shared["ledger"].append({"hash": digest, "cache": False})
        return response


class StructuredQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = FakeBase(self.root)

    def evaluate(self, api=None):
        return target.evaluate(api or self.base, "QUESTION_MARKER", "GOLD_ONLY_MARKER", facts(), "2023-01-01")

    def test_exact_original_evaluator_result_and_prompts(self):
        output = self.evaluate()
        class Plain:
            def generate(inner, prompt, **kwargs):
                return copy.deepcopy(self.base.shared["reader" if kwargs["stage"] == "qa_reader" else "judge"])
        original = target.evaluation.answer_and_judge(Plain(), "QUESTION_MARKER", "GOLD_ONLY_MARKER", facts(),
                                                      reference_date="2023-01-01")
        self.assertEqual(output["result"], original)
        calls = self.base.shared["calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["signature"]["body"]["systemInstruction"]["parts"][0]["text"], target.evaluation.READER_SYSTEM)
        self.assertEqual(calls[1]["signature"]["body"]["systemInstruction"]["parts"][0]["text"], target.evaluation.JUDGE_SYSTEM)
        self.assertEqual(calls[0]["signature"]["model"], "gemini-2.5-flash")
        self.assertEqual(calls[1]["signature"]["model"], "gemini-3.5-flash-lite")
        self.assertNotIn("GOLD_ONLY_MARKER", json.dumps(calls[0]))
        self.assertIn("GOLD_ONLY_MARKER", json.dumps(calls[1]))

    def test_schema_and_retry_budget_recorded(self):
        output = self.evaluate()
        schemas = [target.READER_SCHEMA, target.JUDGE_SCHEMA]
        for call, schema, event in zip(self.base.shared["calls"], schemas, output["api_events"]):
            self.assertEqual(call["signature"]["body"]["generationConfig"]["responseSchema"], schema)
            self.assertEqual(call["signature"]["body"]["generationConfig"]["maxOutputTokens"], 2048)
            self.assertEqual(event["max_http_attempts"], 3)
            self.assertEqual(event["cache_source"], "new_http")
            self.assertEqual(read(event["event_path"]), event)
            self.assertEqual(event["cache_sha256"], file_sha(event["cache_path"]))

    def test_event_and_intent_do_not_copy_question_gold_or_answers(self):
        self.evaluate()
        for folder in ("mem0_transfer_qa_events", "mem0_transfer_qa_intents"):
            for path in (self.root / folder).rglob("*.json"):
                text = path.read_text(encoding="utf-8")
                for value in ("QUESTION_MARKER", "GOLD_ONLY_MARKER", "an answer", "Equivalent."):
                    self.assertNotIn(value, text)

    def test_cache_reuse_generates_no_new_physical_request(self):
        first = self.evaluate()
        second = self.evaluate()
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(len(self.base.shared["calls"]), 2)
        self.assertTrue(all(e["cache_source"] == "validated_cache" for e in second["api_events"]))

    def test_shared_hash_lock_prevents_concurrent_duplicate_calls(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(lambda i: self.evaluate(self.base.scoped(query_id="copy_" + str(i))), range(8)))
        self.assertEqual(len(self.base.shared["calls"]), 2)
        self.assertTrue(all(row["result"] == rows[0]["result"] for row in rows))
        self.assertEqual(sum(e["new_logical_dispatch"] for row in rows for e in row["api_events"]), 2)

    def test_unknown_attempt_not_resent_by_entry(self):
        self.base.shared["transport_failure"] = True
        with self.assertRaises(target.Mem0TransferQAError) as first:
            self.evaluate()
        self.assertEqual(first.exception.api_events[0]["cache_source"], "failed_or_unknown_response")
        self.base.shared["transport_failure"] = False
        with self.assertRaises(target.Mem0TransferQAError) as second:
            self.evaluate()
        self.assertEqual(second.exception.api_events[0]["cache_source"], "prior_unknown_no_redispatch")
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_invalid_answer_keeps_failure_event_and_no_judge(self):
        self.base.shared["reader"]["answer"] = ["not a string"]
        with self.assertRaises(target.Mem0TransferQAError) as failure:
            self.evaluate()
        self.assertEqual(len(self.base.shared["calls"]), 1)
        self.assertEqual(failure.exception.api_events[0]["cache_source"], "new_http_invalid_response")
        self.assertTrue(Path(failure.exception.api_events[0]["event_path"]).exists())

    def test_nonstop_response_rejected_and_not_retried(self):
        self.base.shared["finish"] = "MAX_TOKENS"
        for _ in range(2):
            with self.assertRaises(target.Mem0TransferQAError):
                self.evaluate()
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_wrong_ids_or_duplicate_ids_rejected(self):
        for ids in (["unknown_id"], ["f1", "f1"]):
            with tempfile.TemporaryDirectory() as folder:
                base = FakeBase(folder)
                base.shared["reader"]["used_fact_ids"] = ids
                with self.assertRaises(target.Mem0TransferQAError):
                    self.evaluate(base)
                self.assertEqual(len(base.shared["calls"]), 1)

    def test_insufficient_or_duplicate_selection_rejected_before_calls(self):
        for selected in (facts()[:19], facts() + [facts()[0]], [facts()[0]] * 20):
            with self.assertRaises(ValueError):
                target.evaluate(self.base, "Q", "R", selected, "2023-01-01")
        self.assertEqual(self.base.shared["calls"], [])

    def test_wrong_model_prompt_or_stage_rejected(self):
        wrapper = target.StructuredMem0QAAPI(self.base)
        valid = {"model": "gemini-2.5-flash", "system": target.evaluation.READER_SYSTEM,
                 "json_output": True, "max_output": 2048, "stage": "qa_reader"}
        for override in ({"model": "gemini-2.0-flash"}, {"system": "other"}, {"stage": "writer"},
                         {"max_output": 8192}, {"json_output": False}):
            with self.assertRaises(ValueError):
                wrapper.generate("{}", **{**valid, **override})
        self.assertEqual(self.base.shared["calls"], [])

    def test_retry_budget_drift_rejected(self):
        self.evaluate()
        self.base.retries = 0
        with self.assertRaises(target.Mem0TransferQAError):
            self.evaluate()
        self.assertEqual(len(self.base.shared["calls"]), 2)
        with self.assertRaises(ValueError):
            target.StructuredMem0QAAPI(FakeBase(self.root, retries=3))

    def test_tampered_cache_request_rejected_without_dispatch(self):
        first = self.evaluate()
        path = Path(first["api_events"][0]["cache_path"])
        changed = read(path)
        changed["request"]["model"] = "other-model"
        publish_json(path, changed)
        with self.assertRaises(target.Mem0TransferQAError):
            self.evaluate()
        self.assertEqual(len(self.base.shared["calls"]), 2)

    def test_cache_without_intent_rejected(self):
        first = self.evaluate()
        Path(first["api_events"][0]["intent_path"]).unlink()
        with self.assertRaises(target.Mem0TransferQAError):
            self.evaluate()
        self.assertEqual(len(self.base.shared["calls"]), 2)

    def test_events_return_only_current_evaluation_on_reused_wrapper(self):
        wrapper = target.StructuredMem0QAAPI(self.base)
        first = self.evaluate(wrapper)
        second = self.evaluate(wrapper)
        self.assertEqual(len(wrapper.events), 4)
        self.assertEqual(len(first["api_events"]), 2)
        self.assertEqual(len(second["api_events"]), 2)


if __name__ == "__main__":
    unittest.main()
