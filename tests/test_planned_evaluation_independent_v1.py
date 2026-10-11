"""Independent review of new plain-QA driver guards; zero provider calls."""
import copy
import unittest
from unittest.mock import patch

from tools import run_planned_plain_qa_v1 as new
from tools import run_sdk20_plain_qa_v1 as old
from tools import run_planned_retrieval_v1 as retrieval
from jev_mem.evaluation_plain_text_v1 import reader_input


def job():
    return {"conversation": 0, "method": "hybrid_rank", "query_id": "c0_q0",
            "job_id": "hybrid_rank/c0_q0", "status": "ready", "question": "When?",
            "reference": "unchanged reference", "category": 2, "reference_date": "2023-10-22",
            "selected_ids": [f"f{i}" for i in range(20)],
            "selected_facts": [{"id": f"f{i}", "text": f"original text {i}",
                                "observation_date": "2023-05-08"} for i in range(20)]}


class PlannedEvaluationReviewTests(unittest.TestCase):
    def test_old_module_functions_not_rebound_or_mutated(self):
        self.assertEqual(old.ARMS, ("baseline_sdk20", "fact_types_sdk20"))
        self.assertEqual(old.SCHEMA, "sdk20-plain-qa-run-v1")
        for name in new.REBOUND_FUNCTIONS:
            source, rebound = getattr(old, name), getattr(new, name)
            self.assertIs(source.__globals__, old.__dict__)
            self.assertIs(rebound.__globals__, new.__dict__)
            self.assertIsNot(source, rebound)
            # Only source admission/population are deliberately overridden.
            if name not in {"source_validate", "prepare"}:
                self.assertIs(source.__code__, rebound.__code__)

    def test_original_evaluator_and_models_reused_exactly(self):
        self.assertIs(new.read_answer, old.read_answer)
        self.assertIs(new.judge_answer, old.judge_answer)
        self.assertIs(new.PlainTextEvaluationAPI, old.PlainTextEvaluationAPI)
        for name in ("reader_model", "judge_model", "evaluation_protocol", "transport_retries",
                     "final_priority", "recovery", "valid_yes_and_no", "successful_reader"):
            self.assertEqual(new.POLICY[name], old.POLICY[name])

    def test_sidecar_fields_cannot_enter_reader_and_facts_kept(self):
        original = job(); snapshot = copy.deepcopy(original)
        new.check_jobs([original])
        self.assertEqual(snapshot, original)
        payload = reader_input(original["question"], original["selected_facts"], reference_date=original["reference_date"])
        self.assertEqual(set(payload), {"question", "conversation_reference_date", "facts"})
        self.assertTrue(all(set(f) == {"id", "fact", "observation_date"} for f in payload["facts"]))
        changed = copy.deepcopy(original); changed["selected_facts"][0]["assertion_spans"] = ["hint"]
        with self.assertRaisesRegex(ValueError, "Only frozen Reader"):
            new.check_jobs([changed])

    def test_valid_no_is_never_recovery_and_parent_only_once(self):
        original = job()
        outcome = {"outcomes": [], "results": [{"conversation": 0, "method": original["method"],
            "query_id": original["query_id"], "reader": "r", "judge": "j", "compat": "c", "qa": "q"}]}
        rows = {"r": {"status": "complete"}, "j": {"status": "complete", "result": {"correct": False}},
                "c": {"status": "unavailable", "error": None}}
        with patch.object(new, "bound", side_effect=lambda key: rows[key]):
            self.assertEqual(new.recovery_jobs({"phase_kind": "firstpass", "jobs": [original]}, outcome), [])
        with self.assertRaisesRegex(ValueError, "Only one"):
            new.recovery_jobs({"phase_kind": "technical_recovery"}, outcome)

    def test_unknown_and_local_failure_never_redispatched(self):
        with self.assertRaisesRegex(ValueError, "Local/controller"):
            new.recovery_jobs({"phase_kind": "firstpass"}, {"outcomes": [{"local_errors": ["error"]}]})
        with self.assertRaisesRegex(ValueError, "Missing/unknown"):
            new.recovery_jobs({"phase_kind": "firstpass", "jobs": []}, {"outcomes": [], "results": [
                {"reader": None, "judge": None, "compat": None, "qa": None}]})

    def test_three_repeat_not_implicitly_added(self):
        self.assertNotIn("repeat", new.POLICY)
        self.assertEqual(new.POLICY["slots_per_arm"], 1540)
        self.assertEqual(new.POLICY["facts"], 20)

    def test_profiling_not_equivalence_but_selection_drift_detectable(self):
        original = {"hybrid_rank": {"selected_ids": ["f1"], "trace": {
            "selector_cpu_seconds": .01, "pool_sha256": "same", "effective_query_weights": None}}}
        changed = copy.deepcopy(original); changed["hybrid_rank"]["trace"]["selector_cpu_seconds"] = 8
        self.assertEqual(retrieval.stable_selections(original), retrieval.stable_selections(changed))
        changed["hybrid_rank"]["selected_ids"] = ["f2"]
        self.assertNotEqual(retrieval.stable_selections(original), retrieval.stable_selections(changed))
        self.assertIn("selector_cpu_seconds", original["hybrid_rank"]["trace"])

    def test_original_inventory_drift_is_a_hard_gate(self):
        item = {"original_native": "old-source", "inventory": {"x": "frozen"}}
        with patch.object(retrieval.sources, "inventory", return_value={"x": "changed"}):
            with self.assertRaisesRegex(ValueError, "source integrity gate"):
                retrieval.verify_original_inventory(item)

    def test_missing_embedding_cache_and_counttokens_have_no_fallback(self):
        api = retrieval.CacheOnlyEmbeddings({})
        with self.assertRaisesRegex(RuntimeError, "no network fallback"):
            api.request("gemini", "gemini-embedding-001", {}, operation="embedContent", stage="test")
        with self.assertRaisesRegex(RuntimeError, "prohibited"):
            api.count_tokens("query")
        self.assertEqual(api.events, [])

    def test_fact_projection_excludes_sidecar_and_embedding(self):
        fact = {"id": "x", "text": "original", "observation_date": "2023-01-01", "user_id": "u",
                "embedding": [0], "source_available_at": "fake", "assertion_spans": ["fake"]}
        before = copy.deepcopy(fact)
        projected = retrieval.fact_projection({"facts": [fact]})
        self.assertEqual(set(projected[0]), {"id", "text", "observation_date", "user_id"})
        self.assertEqual(fact, before)


if __name__ == "__main__":
    unittest.main()
