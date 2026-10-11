"""Mock-only runner contracts. No credentials, models, network, or paid requests."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("routing_runner", ROOT / "tools/run_routing_experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MockAPI:
    def __init__(self, calls=None, context=None):
        self.calls = calls if calls is not None else []
        self.context = dict(context or {})

    def scoped(self, **context):
        return MockAPI(self.calls, {**self.context, **context})

    def score_many(self, state, criteria, **kwargs):
        self.calls.append({"kind": "score", "state": state, "criteria": criteria, "context": dict(self.context)})
        return {key: .8 for key in criteria}

    def embed(self, texts, **kwargs):
        self.calls.append({"kind": "embed", "texts": list(texts), "kwargs": kwargs, "context": dict(self.context)})
        return [[1.0, 0.0] for _ in texts]

    def generate(self, prompt, **kwargs):
        payload = json.loads(prompt)
        self.calls.append({"kind": "generate", "payload": payload, "kwargs": kwargs, "context": dict(self.context)})
        if kwargs["stage"] == "gap_planner":
            return {"gaps": [{"slot_id": "ignored_slot", "anchors": [{"text": "Project 0", "support_fact_ids": ["f000"]}],
                "required_relation": "leader", "unknown_part": "who leads the project", "constraints": [],
                "status": "partial", "needed_types": {"semantic": .9, "explicit_relation": 1.0}}]}
        if kwargs["stage"] == "qa_reader":
            return {"answer": "Ada", "used_fact_ids": [payload["facts"][0]["id"]]}
        if kwargs["stage"] == "strict_judge":
            return {"correct": True, "completeness": 1.0, "reason": "matches"}
        raise AssertionError("Unexpected generation stage")

    def _redact(self, error):
        return str(error)


def fixture(count=90):
    facts = [{"id": f"f{i:03d}", "text": f"Ada works on Project {i}.", "embedding": [1.0, i / count], "observation_date": "2023-05-01"} for i in range(count)]
    tags = {fact["id"]: {"coarse_scores": {"semantic": .8}, "fine_scores": {"explicit_relation": .8}} for fact in facts}
    binding = {"conversation": 0, "facts_sha256": "frozen", "tags_sha256": "frozen_tags", "partial_conversation_smoke": False}
    question = {"id": "c0_q0", "question": "Who leads Project 0?", "reference": "SECRET_REFERENCE_SENTINEL", "category": 1, "gold_evidence": ["GOLD_EVIDENCE_SENTINEL"]}
    return facts, tags, binding, question


class RoutingRunnerTests(unittest.TestCase):
    def test_preparation_shared_planner_scope_and_no_reference_leak(self):
        facts, tags, binding, question = fixture()
        originals = copy.deepcopy((facts, tags))
        api = MockAPI()
        with tempfile.TemporaryDirectory() as directory:
            manifest = runner.ensure_run(Path(directory) / "output", Path(directory) / "source", "all")
            row = runner.prepare_query(question, facts, tags, binding, manifest, Path(directory) / "output/conv_0", api)
            self.assertEqual({name: len(pool) for name, pool in row["pools"].items()}, {"untyped60": 60, "typed60": 60, "untyped80": 80, "typed80": 80})
            planners = [call for call in api.calls if call["kind"] == "generate"]
            self.assertEqual(len(planners), 1)
            self.assertEqual(planners[0]["context"]["pool"], "expanded_shared")
            self.assertEqual(row["planner_seed_trace"]["selected_count"], 8)
            self.assertEqual(row["untyped_gap_plan"]["gaps"][0]["needed_types"], {})
            self.assertTrue(row["typed_gap_plan"]["gaps"][0]["needed_types"])
            self.assertEqual(row["untyped_gap_plan"]["gaps"][0]["retrieval_query"], row["typed_gap_plan"]["gaps"][0]["retrieval_query"])
            self.assertNotIn("SECRET_REFERENCE_SENTINEL", json.dumps(api.calls))
            self.assertNotIn("GOLD_EVIDENCE_SENTINEL", json.dumps(api.calls))
            self.assertNotIn("reference", row)
            self.assertEqual((facts, tags), originals)
            prior = len(api.calls)
            replay = runner.prepare_query(question, facts, tags, binding, manifest, Path(directory) / "output/conv_0", api)
            self.assertEqual(row, replay)
            self.assertEqual(len(api.calls), prior)

    def test_coarse_scope_excludes_fine_query_and_gap_type_scores(self):
        facts, tags, binding, question = fixture()
        api = MockAPI()
        with tempfile.TemporaryDirectory() as directory:
            manifest = runner.ensure_run(Path(directory) / "output", Path(directory) / "source", "coarse")
            row = runner.prepare_query(question, facts, tags, binding, manifest, Path(directory) / "output/conv_0", api)
            self.assertEqual(set(row["query_type_scores"]), set(runner.COARSE_CRITERIA))
            self.assertEqual(row["query_type_trace"]["criteria_evaluations"], 4)
            self.assertEqual(row["typed_gap_plan"]["gaps"][0]["needed_types"], {"semantic": .9})

    def test_same_frozen_pool_for_rank_and_conditional_and_judge_only_reference(self):
        facts, tags, binding, question = fixture()
        api = MockAPI()
        with tempfile.TemporaryDirectory() as directory:
            manifest = runner.ensure_run(Path(directory) / "output", Path(directory) / "source", "all")
            folder = Path(directory) / "output/conv_0"
            pool = runner.prepare_query(question, facts, tags, binding, manifest, folder, api)
            rank = runner.evaluate_query(question, "typed80_rank", pool, manifest, folder, api, "2023-05-01")
            conditional = runner.evaluate_query(question, "typed80_conditional", pool, manifest, folder, api, "2023-05-01")
            self.assertEqual(rank["pool_fingerprint"], conditional["pool_fingerprint"])
            self.assertEqual(rank["selection_trace"]["pool_hash"], conditional["selection_trace"]["pool_hash"])
            self.assertEqual(rank["selected_count"], 20)
            self.assertEqual(conditional["selected_count"], 20)
            for call in api.calls:
                if "SECRET_REFERENCE_SENTINEL" in json.dumps(call):
                    self.assertEqual(call["kwargs"]["stage"], "strict_judge")
            prior = len(api.calls)
            self.assertEqual(rank, runner.evaluate_query(question, "typed80_rank", pool, manifest, folder, api, "2023-05-01"))
            self.assertEqual(prior, len(api.calls))

    def test_tampered_pool_cache_is_rejected_before_model_calls(self):
        facts, tags, binding, question = fixture()
        api = MockAPI()
        with tempfile.TemporaryDirectory() as directory:
            manifest = runner.ensure_run(Path(directory) / "output", Path(directory) / "source", "all")
            folder = Path(directory) / "output/conv_0"
            runner.prepare_query(question, facts, tags, binding, manifest, folder, api)
            path = folder / "pools/c0_q0.json"
            row = runner.read_json(path)
            row["pools"]["typed60"][0]["text"] = "tampered"
            runner.dump_json(path, row)
            prior = len(api.calls)
            with self.assertRaisesRegex(runner.ProtocolMismatch, "content changed"):
                runner.prepare_query(question, facts, tags, binding, manifest, folder, api)
            self.assertEqual(prior, len(api.calls))

    def test_failures_keep_same_denominator_for_all_methods(self):
        _, _, _, q = fixture()
        requested = [q, {**q, "id": "c0_q1"}]
        rows = [{"query_id": q["id"], "method": "typed60_rank", "judgment": {"correct": True}, "token_f1": 1.0}]
        report = runner.summarize(rows, requested, ["typed60_rank", "typed60_conditional"], [], 0)
        self.assertEqual(report["methods"]["typed60_rank"]["accuracy_all_requested"], .5)
        self.assertEqual(report["methods"]["typed60_conditional"]["missing"], 2)
        self.assertFalse(report["evaluation_complete"])

    def test_check_stage_never_instantiates_api(self):
        facts, tags, binding, question = fixture()
        def forbidden(*args, **kwargs):
            raise AssertionError("Offline check must not create API")
        with patch.object(runner, "load_data", return_value=[{}]), patch.object(runner, "load_source", return_value=(facts, tags, binding)), patch.object(runner, "queries", return_value=iter([question])):
            report = runner.run_conversation("unused", "unused_source", 0, {}, stage="check", api_factory=forbidden)
        self.assertEqual(report["api_calls"], 0)
        self.assertEqual(report["status"], "ready")

    def test_source_requires_complete_ingestion_and_matching_fact_tags(self):
        item = {"conversation": {"speaker_a": "Ada", "speaker_b": "Bob",
                "session_1_date_time": "10:00 AM on 1 May, 2023",
                "session_1": [{"speaker": "Ada", "text": "I work on Project 0.", "dia_id": "D1:1"}]}, "qa": []}
        fact = {"id": "f000", "text": "Ada works on Project 0.", "embedding": [1.0] + [0.0] * 767,
                "input_scope": {"session_id": "session_1"}}
        summary = {"conversation": 0, "complete": True, "sessions_completed": 1, "sessions_total": 1, "facts": 1}
        tag = {"fact_id": "f000", "cache_schema": runner.SOURCE_SCHEMA,
               "coarse_scores": {key: .5 for key in runner.COARSE_CRITERIA},
               "fine_scores": {key: .5 for key in runner.FINE_CRITERIA}}
        tag["request_fingerprint"] = runner.fingerprint({"schema": runner.SOURCE_SCHEMA, "fact": fact,
            "context": "Ada: I work on Project 0.", "prompt_version": runner.PROMPT_VERSION,
            "model": "jev-1.13.0", "selector_code_sha256": runner.source_hash(ROOT / "jev_mem/selection.py")})
        with tempfile.TemporaryDirectory() as directory:
            conv = Path(directory) / "conv_0"
            runner.dump_json(conv / "facts.json", [fact])
            runner.dump_json(conv / "ingestion_summary.json", summary)
            runner.dump_json(conv / "tags/f000.json", tag)
            loaded, _, _ = runner.load_source(directory, 0, item)
            self.assertEqual(loaded, [fact])
            runner.dump_json(conv / "facts.json", [{**fact, "text": "Ada no longer works there."}])
            with self.assertRaisesRegex(runner.ProtocolMismatch, "stale fact"):
                runner.load_source(directory, 0, item)
            runner.dump_json(conv / "facts.json", [fact])
            runner.dump_json(conv / "ingestion_summary.json", {**summary, "complete": False})
            with self.assertRaisesRegex(ValueError, "complete source conversation"):
                runner.load_source(directory, 0, item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
