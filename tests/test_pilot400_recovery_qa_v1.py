"""Offline tests: fake transport only; no credentials or real model calls."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from tools import pilot400_recovery_qa_v1 as target


def response(value):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(value)}]}}]}


def facts():
    return [{"id": "fact_" + str(i), "text": "Fact text " + str(i),
             "observation_date": "2023-01-01", "observation_dates": ["2023-01-01"],
             "source_observation_dates": {"D1:" + str(i): "2023-01-01"},
             "temporal_status": "not_temporal", "temporal_status_verified": False,
             "support_status": "proposed_unverified"} for i in range(20)]


class FakeBase:
    def __init__(self, folder, *, shared=None, context=None):
        self.run_dir = Path(folder)
        self.retries = 0
        self.context = context or {}
        self.shared = shared or {"calls": [], "logs": [], "reader": {"answer": "an answer", "used_fact_ids": ["f1"]},
                                 "judge": {"correct": True, "completeness": 1, "reason": "accepted"},
                                 "failure": False}

    def scoped(self, **ctx):
        return FakeBase(self.run_dir, shared=self.shared, context={**self.context, **ctx})

    def _redact(self, value):
        return str(value).replace("fake_secret", "<redacted>")

    def _log(self, value):
        self.shared["logs"].append(value)

    def request(self, provider, model, body, *, stage):
        signature = {"provider": provider, "model": model, "operation": "generateContent", "body": body}
        digest = target.request_hash(signature)
        path = self.run_dir / "api_cache" / (digest + ".json")
        if path.exists():
            return target.read(path)["response"]
        self.shared["calls"].append({"stage": stage, "signature": copy.deepcopy(signature)})
        if self.shared["failure"]:
            raise RuntimeError("fake_secret transport failure")
        time.sleep(.003)
        data = response(self.shared["reader" if stage == target.evaluation.READER_STAGE else "judge"])
        target.publish_json(path, {"request": signature, "response": data})
        return data


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.base = FakeBase(self.path)
        self.api = target.ResponseSchemaAPI(self.base)
        self.q = {"id": "c0_q0", "question": "Question?", "reference": "SECRET_REFERENCE"}

    def reader(self, api=None):
        return target.evaluation.answer_question(api or self.api, self.q["question"], facts(), reference_date="2023-01-01")

    def test_schema_and_same_frozen_prompt(self):
        self.reader()
        request = self.base.shared["calls"][0]["signature"]
        self.assertEqual(request["model"], "gemini-2.5-flash")
        body = request["body"]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], target.evaluation.READER_SYSTEM)
        self.assertEqual(body["generationConfig"]["responseSchema"], target.READER_SCHEMA)
        self.assertEqual(body["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0})
        self.assertNotIn("SECRET_REFERENCE", json.dumps(body))
        self.assertTrue(self.api.events[0]["new_physical_attempt"])

    def test_new_qa_one_reader_one_judge_no_extra_calls(self):
        result = target.evaluate_new(self.q, facts(), "2023-01-01", self.api)
        self.assertEqual(result["answer"], "an answer")
        self.assertEqual(len(self.base.shared["calls"]), 2)
        judge = self.base.shared["calls"][1]["signature"]
        self.assertEqual(judge["model"], "gemini-3.5-flash-lite")
        self.assertEqual(judge["body"]["generationConfig"]["responseSchema"], target.JUDGE_SCHEMA)
        self.assertIn("SECRET_REFERENCE", json.dumps(judge))

    def test_same_hash_threaded_calls_share_one_physical_attempt(self):
        def one(i):
            api = self.api.scoped(method=str(i))
            result = target.evaluate_new(self.q, facts(), "2023-01-01", api)
            return result, api.events
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(one, range(8)))
        self.assertEqual(len(self.base.shared["calls"]), 2)
        self.assertEqual(sum(event["new_physical_attempt"] for _, events in rows for event in events), 2)
        self.assertTrue(all(result["answer"] == "an answer" for result, _ in rows))

    def test_unknown_attempt_is_not_resent(self):
        self.base.shared["failure"] = True
        with self.assertRaises(RuntimeError):
            self.reader()
        self.base.shared["failure"] = False
        with self.assertRaisesRegex(RuntimeError, "unknown outcome is not resent"):
            self.reader()
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_cache_signature_mutation_is_rejected(self):
        self.reader()
        path = next((self.path / "api_cache").glob("*.json"))
        value = target.read(path)
        value["request"]["model"] = "different"
        target.publish_json(path, value)
        with self.assertRaisesRegex(ValueError, "cache signature differs"):
            self.reader()
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_invalid_answer_array_never_coerced_or_judged(self):
        self.base.shared["reader"] = {"answer": ["one", "two"], "used_fact_ids": ["f1"]}
        with self.assertRaisesRegex(ValueError, "answer:string"):
            target.evaluate_new(self.q, facts(), "2023-01-01", self.api)
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_invalid_ids_rejected(self):
        self.base.shared["reader"] = {"answer": "ok", "used_fact_ids": ["f21"]}
        with self.assertRaisesRegex(ValueError, "unique valid"):
            self.reader()

    def test_old_reader_exact_reuse_without_physical_request(self):
        old = self.path / "old"
        old.mkdir()
        prepared = target.evaluation.prepare_reader_input(self.q["question"], facts(), reference_date="2023-01-01")
        sig = target.generation_signature(target._json(prepared["payload"]), model=target.evaluation.READER_MODEL,
            system=target.evaluation.READER_SYSTEM, json_output=True, max_output=target.evaluation.MAX_OUTPUT)
        digest = target.request_hash(sig)
        cache_path = old / "api_cache" / (digest + ".json")
        target.publish_json(cache_path, {"request": sig, "response": response(self.base.shared["reader"])})
        prior = target._original_reader(self.q, facts(), "2023-01-01", old)
        before = cache_path.read_bytes()
        api = target.ResponseSchemaAPI(self.base, original_reader=prior)
        result = target.evaluate_new(self.q, facts(), "2023-01-01", api)
        self.assertTrue(result["judge"]["correct"])
        self.assertEqual(len(self.base.shared["calls"]), 1)
        self.assertEqual(self.base.shared["calls"][0]["stage"], target.evaluation.JUDGE_STAGE)
        self.assertEqual(api.events[0]["source"], "original_valid_reader_cache")
        self.assertEqual(before, cache_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            target.evaluation.answer_question(api, "Other question", facts(), reference_date="2023-01-01")

    def test_legacy_and_structured_signatures_differ(self):
        args = {"model": target.evaluation.READER_MODEL, "system": target.evaluation.READER_SYSTEM,
                "json_output": True, "max_output": 2048}
        first = target.generation_signature("text", **args)
        second = target.generation_signature("text", schema=target.READER_SCHEMA, **args)
        self.assertNotEqual(target.request_hash(first), target.request_hash(second))
        second["body"]["generationConfig"].pop("responseSchema")
        self.assertEqual(first, second)

    def test_frozen_paths_and_bad_retries_rejected(self):
        with self.assertRaises(ValueError):
            target._safe_output(target.ORIGINAL_RUN)
        self.base.retries = 2
        with self.assertRaisesRegex(ValueError, "retries=0"):
            target.ResponseSchemaAPI(self.base)

    def test_other_models_prompts_stages_rejected(self):
        with self.assertRaises(ValueError):
            self.api.generate("hi", stage="other", json_output=True)
        with self.assertRaises(ValueError):
            self.api.generate("hi", stage=target.evaluation.READER_STAGE, json_output=True,
                              model="gemini-other", max_output=2048, system=target.evaluation.READER_SYSTEM)
        self.assertFalse(self.base.shared["calls"])

    def test_twenty_unique_fact_budget(self):
        with self.assertRaises(ValueError):
            target.evaluate_new(self.q, facts()[:19], "2023-01-01", self.api)
        duplicate = facts()
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            target.evaluate_new(self.q, duplicate, "2023-01-01", self.api)
        self.assertFalse(self.base.shared["calls"])

    def test_seals_reject_altered_result(self):
        saved = target._sealed({"status": "failed", "result": None})
        self.assertEqual(target._verify_sealed(saved), saved)
        saved["status"] = "complete"
        with self.assertRaises(ValueError):
            target._verify_sealed(saved)


if __name__ == "__main__":
    unittest.main()
