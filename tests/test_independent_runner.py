"""Offline mocked end-to-end stages and artifact isolation checks."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("independent_runner", ROOT / "tools/run_independent_experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeAPI:
    created = 0
    generations = 0

    def __init__(self, path):
        self.created += 1
        Path(path).mkdir(parents=True, exist_ok=True)

    def generate(self, prompt, **kwargs):
        type(self).generations += 1
        session = json.loads(prompt)["new_session"]
        source = session["turns"][0]["source_turn_id"]
        return {"facts": [{"text": "Alice likes painting " + session["session_id"], "source_turn_ids": [source]}]}

    def embed(self, texts, **kwargs):
        return [[1.0] + [0.0] * 767 for _ in texts]

    def scoped(self, **context):
        return self

    def _redact(self, error):
        return str(error)


def envelopes(item):
    for index in range(2):
        yield {"session_id": f"s{index}", "messages": [{"role": "user", "content": "Alice: I like painting."}],
               "source_turn_ids": [f"t{index}"], "observation_date": f"2023-01-0{index + 1}"}


def questions(item, ci):
    for index in range(2):
        yield {"id": f"c{ci}_q{index}", "question": f"Question {index}", "reference": "painting", "category": 1}


def score(api, question, reference, facts, **kwargs):
    return {"prediction": {"answer": "painting", "used_fact_ids": []}, "judgment": {"correct": True},
            "token_f1": 1.0, "selected_count": len(facts)}


class IndependentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev_independent_runner_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        FakeAPI.generations = 0
        for name, value in {"load_data": lambda: [{}], "session_envelopes": envelopes,
                            "queries": questions, "ModelAPI": FakeAPI, "answer_and_judge": score}.items():
            manager = patch.object(runner, name, value)
            manager.start()
            self.addCleanup(manager.stop)

    def test_full_stages_resume_and_method_failure_keeps_other_successes(self):
        summary = runner.ingest_one(self.path, 0)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["sessions_completed"], 2)
        runner.ingest_one(self.path, 0)
        self.assertEqual(FakeAPI.generations, 2)
        retrieval = runner.retrieve_one(self.path, 0, workers=2)
        self.assertEqual(retrieval["completed"], 2)
        real_select = runner.select

        def selectively_fail(question, facts, api, **kwargs):
            if kwargs["method"] == "mmr" and question == "Question 0":
                raise RuntimeError("offline method failure")
            return real_select(question, facts, api, **kwargs)

        with patch.object(runner, "select", selectively_fail):
            qa = runner.qa_one(self.path, 0, methods=["rank", "mmr"], workers=2)
        self.assertEqual(qa["methods"]["rank"]["completed"], 2)
        self.assertEqual(qa["methods"]["mmr"]["completed"], 1)
        self.assertEqual(qa["methods"]["mmr"]["accuracy_all_requested"], .5)
        self.assertEqual(qa["methods"]["mmr"]["accuracy_successful_only"], 1.)
        self.assertEqual(qa["requested_evaluations"], 4)
        recovered = runner.qa_one(self.path, 0, methods=["rank", "mmr"], workers=2)
        self.assertTrue(recovered["complete"])

    def test_smoke_is_explicit_and_cannot_be_promoted(self):
        with self.assertRaises(ValueError):
            runner.ingest_one(self.path, 0, max_sessions=1)
        summary = runner.ingest_one(self.path, 0, max_sessions=1, allow_partial=True)
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["smoke_run"])
        with self.assertRaises(RuntimeError):
            runner.retrieve_one(self.path, 0)
        runner.retrieve_one(self.path, 0, max_questions=1, allow_partial=True)
        qa = runner.qa_one(self.path, 0, methods=["rank"], max_questions=1, allow_partial=True)
        self.assertEqual(qa["requested_queries"], 1)
        self.assertTrue(qa["partial_conversation_smoke"])
        row = json.loads((self.path / "conv_0/qa/rank/c0_q0.json").read_text())
        self.assertEqual(row["reference_date"], "2023-01-01")

    def test_tampered_snapshot_or_retrieval_not_reused(self):
        runner.ingest_one(self.path, 0)
        runner.retrieve_one(self.path, 0)
        path = self.path / "conv_0/retrieval/c0_q0.json"
        row = json.loads(path.read_text())
        row["partial_conversation_smoke"] = True
        runner.dump_json(path, row)
        report = runner.qa_one(self.path, 0, methods=["rank"])
        self.assertEqual(report["methods"]["rank"]["completed"], 1)
        self.assertEqual(report["methods"]["rank"]["requested"], 2)
        fact_path = self.path / "conv_0/facts.json"
        facts = json.loads(fact_path.read_text())
        facts[0]["text"] = "Changed outside the committed bank"
        runner.dump_json(fact_path, facts)
        with self.assertRaises(RuntimeError):
            runner.retrieve_one(self.path, 0)

    def test_missing_retrieval_is_an_error_for_every_requested_method(self):
        runner.ingest_one(self.path, 0)
        report = runner.qa_one(self.path, 0, methods=["rank", "conditional"])
        self.assertEqual(len(report["errors"]), 4)
        self.assertEqual(report["methods"]["rank"]["accuracy_all_requested"], 0)
        self.assertEqual(report["methods"]["conditional"]["failed"], 2)


if __name__ == "__main__":
    unittest.main()
