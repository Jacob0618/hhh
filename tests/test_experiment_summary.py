"""No-call accounting and paired-metric regression tests."""
import importlib.util
import ast
import collections
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("summary_under_test", ROOT / "tools/summarize_experiments.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_independent_schema_auto_methods_and_hash_validation(self):
        independent = summary.independent_runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conv = root / "conv_0"
            conv.mkdir()
            protocol = independent._protocol()
            manifest = {"schema": "independent-live-run-v1", "mode": "full", "protocol": protocol,
                        "protocol_sha256": independent._hash(protocol)}
            summary.runner.dump_json(root / "run_manifest.json", manifest)
            facts = [{"id": "fact1", "text": "Alice moved to Paris.", "observation_date": "2023-05-08"}]
            ingestion = {"architecture": "independent", "protocol_sha256": manifest["protocol_sha256"],
                         "smoke_run": False, "facts_sha256": independent._hash(facts), "complete": True,
                         "session_ids": ["session_1"], "user_id": "conv0"}
            summary.runner.dump_json(conv / "ingestion_summary.json", ingestion)
            summary.runner.dump_json(conv / "facts.json", facts)
            summary.runner.dump_json(conv / "independent_store/fact_bank.json", {"facts": facts, "sessions": [{"session_id": "session_1"}]})
            q = {"id": "c0_q0", "question": "Where did Alice move?", "reference": "Paris", "category": 1}
            retrieval = {**independent._retrieval_binding(q, ingestion, manifest), "status": "success", "pool60": facts,
                         "content_sha256": independent._hash({"pool60": facts})}
            summary.runner.dump_json(conv / "retrieval/c0_q0.json", retrieval)
            for method in summary.ARCHITECTURE_METHODS["independent"]:
                row = {**independent._retrieval_binding(q, ingestion, manifest), "method": method,
                       "reference_sha256": independent._hash("Paris"), "retrieval_sha256": independent._hash(retrieval),
                       "reference_date": "2023-05-08", "status": "success", "conversation": 0, "category": 1,
                       "reference": "Paris", "selected_ids": ["fact1"], "selected_count": 1, "selection_trace": {},
                       "prediction": {"answer": "Paris", "used_fact_ids": ["fact1"], "used_local_fact_ids": ["f1"]},
                       "judgment": {"correct": True, "completeness": 1, "reason": "Equivalent"},
                       "token_f1": 1, "exact_match": 1}
                row["result_sha256"] = independent._hash(row)
                summary.runner.dump_json(conv / "qa" / method / "c0_q0.json", row)
            with patch.object(summary, "load_data", return_value=[{}]), \
                 patch.object(summary, "queries", return_value=[q]), \
                 patch.object(summary, "session_envelopes", return_value=[{"session_id": "session_1", "observation_date": "2023-05-08"}]):
                report = summary.summarize_run(root, [0], bootstrap_samples=5)
                self.assertEqual(report["architecture"], "independent")
                self.assertEqual(report["baseline"], "rank")
                self.assertNotIn("sdk20", report["methods"])
                self.assertEqual(report["status"], "complete_requested_cohort")
                self.assertTrue(all(group["overall"]["completed"] == 1 for group in report["methods"].values()))
                row["prediction"]["answer"] = "Tampered answer"
                summary.runner.dump_json(conv / "qa/conditional/c0_q0.json", row)
                changed = summary.summarize_run(root, [0], bootstrap_samples=5)
                self.assertEqual(changed["status"], "in_progress")
                self.assertEqual(changed["methods"]["conditional"]["overall"]["completed"], 0)
                with self.assertRaises(ValueError):
                    summary.summarize_run(root, [0], ["sdk20"])

    def test_independent_context_without_phase_still_attributes_work(self):
        self.assertEqual(summary.phase_of({"stage": "independent_extract", "context": {"architecture": "independent", "session_id": "s1"}}), "writing")
        self.assertEqual(summary.phase_of({"stage": "independent_fact_embedding"}), "writing")
        self.assertEqual(summary.phase_of({"stage": "independent_query_embedding", "context": {"method": "shared_retrieval"}}), "reading")
        self.assertEqual(summary.phase_of({"stage": "jev_score", "context": {"method": "conditional", "architecture": "independent"}}), "reading")
        self.assertEqual(summary.phase_of({"stage": "strict_judge", "context": {"method": "rank"}}), "evaluation_judge")

    def test_official_metric_matches_pinned_functions_including_multi_answer(self):
        import regex
        import string
        import numpy
        from nltk.stem import PorterStemmer

        source = ROOT / "research/sources/locomo_evaluation_cbfbc1.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), summary.SOURCE_SHA256)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"normalize_answer", "f1_score", "f1"}
        kept = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"regex": regex, "string": string, "Counter": collections.Counter,
                     "ps": PorterStemmer(), "np": numpy}
        exec(compile(ast.Module(body=kept, type_ignores=[]), str(source), "exec"), namespace)
        examples = [("Dogs and cats", "dog, cat"), ("Paris, Rome, irrelevant", "Paris, Rome"),
                    ("", ""), ("She was running", "run"), ("2023-05-08", "May 8, 2023")]
        for prediction, reference in examples:
            self.assertAlmostEqual(summary.official_locomo_f1(prediction, reference, 1), namespace["f1"](prediction, reference))
            self.assertAlmostEqual(summary.official_locomo_f1(prediction, reference, 4), namespace["f1_score"](prediction, reference))
        self.assertEqual(summary.official_locomo_f1("Paris", "Paris; also acceptable: France", 3), 1)
        self.assertEqual(summary.official_locomo_f1("Paris, irrelevant", "Paris", 1), 1)

    def test_gemini_thinking_is_billed_as_output(self):
        result = summary.usage_cost({"model": "gemini-3.5-flash-lite", "operation": "generateContent",
                                    "usage": {"promptTokenCount": 1000, "candidatesTokenCount": 100,
                                              "thoughtsTokenCount": 40}})
        self.assertAlmostEqual(result["cost_usd"], (1000*.30 + 140*2.50)/1e6)
        self.assertFalse(result["unknown"])

    def test_embedding_unknown_and_count_tokens_excluded(self):
        embedding = summary.usage_cost({"model": "gemini-embedding-001", "operation": "batchEmbedContents", "usage": {}})
        self.assertTrue(embedding["unknown"])
        self.assertIsNone(embedding["cost_usd"])
        count = summary.usage_cost({"model": "gemini-embedding-001", "operation": "countTokens", "usage": {}})
        self.assertFalse(count["unknown"])
        self.assertEqual(count["cost_usd"], 0)

    def test_cache_retries_cold_dedup_and_real_latencies(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "conv_0"
            path.mkdir()
            base = {"provider": "jev", "model": "jev-1.13.0", "stage": "jev_score", "operation": "systemone",
                    "request_hash": "same", "context": {"phase": "qa", "method": "jev_single", "query_id": "c0_q0"}}
            rows = [{**base, "status": 503, "latency_ms": 100, "cache_hit": False, "usage": {}},
                    {**base, "status": 200, "latency_ms": 300, "cache_hit": False,
                     "usage": {"input_tokens": 1000, "output_tokens": 10}},
                    {**base, "cache_hit": True, "latency_ms": 0,
                     "usage": {"input_tokens": 1000, "output_tokens": 10}},
                    {**base, "context": {"phase": "qa", "method": "conditional", "query_id": "c0_q0"},
                     "cache_hit": True, "latency_ms": 0, "usage": {"input_tokens": 1000, "output_tokens": 10}}]
            (path / "api_calls.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = summary.summarize_api_logs(temp, [0])
            self.assertAlmostEqual(result["actual_known_cost_usd"], .000042)
            self.assertEqual(result["actual_unknown_cost_calls"], 1)
            self.assertAlmostEqual(result["cold_known_cost_usd"], .000084)
            by_method = {group["method"]: group for group in result["groups"]}
            self.assertEqual(by_method["jev_single"]["cold_logical_requests"], 1)
            self.assertEqual(by_method["jev_single"]["actual_http_latency_ms"]["p50"], 200)
            self.assertEqual(by_method["conditional"]["actual_http_latency_ms"]["n"], 0)
            self.assertEqual(by_method["conditional"]["actual_known_cost_usd"], 0)

    def test_unscoped_and_unknown_usage_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "conv_0"
            path.mkdir()
            row = {"provider": "gemini", "model": "gemini-embedding-001", "request_hash": "embed",
                   "operation": "batchEmbedContents", "status": 200, "usage": {}, "latency_ms": 2}
            (path / "api_calls.jsonl").write_text(json.dumps(row) + "\n{partial", encoding="utf-8")
            result = summary.summarize_api_logs(temp, [0])
            self.assertEqual(result["groups"][0]["method"], "legacy_unscoped")
            self.assertEqual(result["actual_unknown_cost_calls"], 1)
            self.assertEqual(len(result["issues"]), 1)

    def test_current_artifact_cold_accounting_excludes_old_reader_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            conv = Path(temp) / "conv_0"
            (conv / "api_cache").mkdir(parents=True)
            base = {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent",
                    "stage": "qa_reader", "status": 200, "latency_ms": 10,
                    "usage": {"promptTokenCount": 1000, "candidatesTokenCount": 100}}
            old = {**base, "request_hash": "old", "context": {"phase": "qa", "method": "sdk20", "query_id": "c0_q0"}}
            new = {**base, "request_hash": "new", "context": old["context"]}
            (conv / "api_calls.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(new), encoding="utf-8")
            (conv / "api_cache/old.json").write_text(json.dumps({"request": {"body": {"systemInstruction": {"parts": [{"text": "old prompt"}]}}}}), encoding="utf-8")
            active = {(0, "sdk20", "c0_q0", "new"): {k: new[k] for k in ("provider", "model", "operation", "stage")}}
            result = summary.summarize_api_logs(temp, [0], active)
            unit = (1000 * .3 + 100 * 2.5) / 1e6
            self.assertAlmostEqual(result["actual_known_cost_usd"], 2 * unit)
            self.assertAlmostEqual(result["cold_known_cost_usd"], unit)
            self.assertAlmostEqual(result["development_overhead_known_cost_usd"], unit)

    def test_shared_denominator_and_seeded_conversation_bootstrap(self):
        row = {"selected_count": 20, "judgment": {"correct": True}, "token_f1": .8,
               "exact_match": 0, "selection_trace": {"fallback_count": 2}}
        group = summary.metric_group(["q0", "q1"], [row])
        self.assertEqual(group["strict_accuracy"]["completed_only"], 1)
        self.assertEqual(group["strict_accuracy"]["all_requested"], .5)
        self.assertEqual(group["fallback_queries"], 1)
        base = {f"q{i}": {**row, "conversation": i, "judgment": {"correct": False}} for i in range(2)}
        other = {f"q{i}": {**row, "conversation": i} for i in range(2)}
        result = summary.paired_bootstrap({"sdk20": base, "conditional": other}, ["sdk20", "conditional"], "sdk20", 50)
        self.assertEqual(result["conditional"]["metrics"]["strict_accuracy"]["bootstrap_95_percentile_interval"], [1, 1])
        self.assertFalse(result["conditional"]["significance_claim"])
        self.assertEqual(result, summary.paired_bootstrap({"sdk20": base, "conditional": other}, ["sdk20", "conditional"], "sdk20", 50))


if __name__ == "__main__":
    unittest.main(verbosity=2)
