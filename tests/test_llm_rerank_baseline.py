"""Mock-only controls for the strong generative reranking baseline."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("llm_rerank_baseline", ROOT / "tools/run_llm_rerank_baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def facts(count=60):
    return [{"id": f"actual-{i}", "text": f"Person {i} leads project {i}.", "observation_date": "2023-05-01",
             "gold_evidence": "GOLD_SENTINEL", "reference": "REFERENCE_SENTINEL"} for i in range(count)]


class MockAPI:
    def __init__(self, calls=None, context=None, bad_ids=None):
        self.calls = calls if calls is not None else []
        self.context = dict(context or {})
        self.bad_ids = bad_ids

    def scoped(self, **context):
        return MockAPI(self.calls, {**self.context, **context}, self.bad_ids)

    def generate(self, prompt, **kwargs):
        payload = json.loads(prompt)
        self.calls.append({"payload": payload, "kwargs": kwargs, "context": self.context})
        stage = kwargs["stage"]
        if stage == "gemini_rerank":
            return {"selected_ids": self.bad_ids if self.bad_ids is not None else [row["id"] for row in payload["candidate_facts"][:payload["target_count"]]]}
        if stage == "qa_reader":
            return {"answer": "Ada", "used_fact_ids": [payload["facts"][0]["id"]]}
        if stage == "strict_judge":
            return {"correct": True, "completeness": 1.0, "reason": "matches"}
        raise AssertionError("Unexpected generation stage")


class BaselineTests(unittest.TestCase):
    def test_one_json_call_exact20_and_gold_never_sent_to_reranker(self):
        api = MockAPI()
        selected, trace = runner.rerank("Who leads the project?", facts(), api)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(api.calls[0]["kwargs"]["model"], "gemini-2.5-flash")
        self.assertEqual(trace["selected_local_ids"], [f"f{i}" for i in range(1, 21)])
        self.assertNotIn("GOLD_SENTINEL", json.dumps(api.calls))
        self.assertNotIn("REFERENCE_SENTINEL", json.dumps(api.calls))
        self.assertEqual(trace["local_to_actual"]["f1"], "actual-0")

    def test_duplicate_invented_or_short_selection_fails_without_fallback(self):
        invalid = [["f1"] * 20, [f"f{i}" for i in range(1, 20)], [f"f{i}" for i in range(1, 20)] + ["f999"], ["actual-0"] * 20]
        for values in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "exactly"):
                runner.rerank("Who?", facts(), MockAPI(bad_ids=values))

    def test_small_or_empty_pool_never_invents_twenty(self):
        api = MockAPI()
        selected, _ = runner.rerank("Who?", facts(3) + [{**facts(1)[0], "id": "duplicate"}], api)
        self.assertEqual(len(selected), 3)
        self.assertEqual(api.calls[0]["payload"]["target_count"], 3)
        selected, trace = runner.rerank("Who?", [], api)
        self.assertEqual(selected, [])
        self.assertEqual(trace["generation_calls"], 0)
        self.assertEqual(len(api.calls), 1)

    def test_evaluate_uses_same_reader_judge_and_durable_two_stage_cache(self):
        api = MockAPI()
        question = {"id": "c0_q0", "question": "Who leads the project?", "reference": "SECRET_GOLD_REFERENCE", "category": 1}
        pool = {"pool60": facts(), "request_fingerprint": "frozen_request", "content_fingerprint": "frozen_content"}
        with tempfile.TemporaryDirectory() as directory:
            manifest = runner.ensure_run(Path(directory) / "output", Path(directory) / "source")
            row = runner.evaluate(question, 0, pool, manifest, Path(directory) / "output/conv_0", api, "2023-05-01")
            self.assertEqual(len(api.calls), 3)
            self.assertEqual(row["selected_count"], 20)
            self.assertEqual([call["kwargs"]["stage"] for call in api.calls], ["gemini_rerank", "qa_reader", "strict_judge"])
            self.assertEqual(api.calls[1]["kwargs"]["system"], runner.READER_SYSTEM)
            self.assertEqual(api.calls[2]["kwargs"]["system"], runner.JUDGE_SYSTEM)
            for call in api.calls:
                if "SECRET_GOLD_REFERENCE" in json.dumps(call):
                    self.assertEqual(call["kwargs"]["stage"], "strict_judge")
                self.assertEqual(call["context"]["method"], "gemini_rerank")
            replay = runner.evaluate(question, 0, pool, manifest, Path(directory) / "output/conv_0", api, "2023-05-01")
            self.assertEqual(replay, row)
            self.assertEqual(len(api.calls), 3)

    def test_accounting_separates_cache_usage_and_unknown_retry_usage(self):
        context = {"query_id": "c0_q0", "method": "gemini_rerank"}
        rows = [
            {"context": context, "stage": "gemini_rerank", "cache_hit": False, "latency_ms": 100,
             "usage": {"promptTokenCount": 300, "candidatesTokenCount": 25, "totalTokenCount": 325}},
            {"context": context, "stage": "gemini_rerank", "cache_hit": True, "latency_ms": 0,
             "usage": {"promptTokenCount": 300, "candidatesTokenCount": 25, "totalTokenCount": 325}},
            {"context": context, "stage": "gemini_rerank", "cache_hit": False, "latency_ms": 50, "usage": {}},
            {"context": context, "stage": "qa_reader", "cache_hit": False, "latency_ms": 200,
             "usage": {"promptTokenCount": 100, "candidatesTokenCount": 5, "totalTokenCount": 105}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_calls.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            accounting = runner.api_accounting(path, {"c0_q0"})
        stage = accounting["stages"]["gemini_rerank"]
        self.assertEqual(stage["physical_requests"], 2)
        self.assertEqual(stage["cache_hits"], 1)
        self.assertEqual(stage["physical_usage"]["total_tokens"], 325)
        self.assertEqual(stage["reused_usage"]["total_tokens"], 325)
        self.assertEqual(stage["usage_unknown_requests"], 1)
        self.assertEqual(stage["network_latency_ms"], 150)

    def test_preflight_failure_occurs_before_api_initialization(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("API must not be created before all pools validate")
        question = {"id": "c0_q0", "question": "Who?"}
        with patch.object(runner, "load_data", return_value=[{}]), patch.object(runner, "load_source", return_value=({}, {}, "hash")), patch.object(runner, "queries", return_value=iter([question])), patch.object(runner, "load_pool", side_effect=RuntimeError("bad frozen pool")):
            with self.assertRaisesRegex(RuntimeError, "bad frozen pool"):
                runner.run_conversation("unused", "source", 0, {}, api_factory=forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
