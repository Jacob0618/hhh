"""Offline routing tests using deterministic local fixtures and a mock planner."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_mem.routing import bm25_scores, build_candidate_pool, expand_pool, plan_gaps
from jev_mem.selection import select


def facts(count=100):
    return [
        {"id": f"f{index:03d}", "text": f"Person {index} works on Project {index} and likes tea.",
         "embedding": [1.0, index / max(count, 1)], "observation_date": "2023-05-01"}
        for index in range(count)
    ]


class MockAPI:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((json.loads(prompt), kwargs))
        return self.response


def gap(anchor="Project 1", fact_id="f001", slot="gap_1"):
    return {
        "slot_id": slot, "anchors": [{"text": anchor, "support_fact_ids": [fact_id]}],
        "required_relation": "project leader", "unknown_part": "who leads this project",
        "constraints": [], "status": "partial", "needed_types": {"semantic": 0.9, "explicit_relation": 1.0},
    }


class RoutingTests(unittest.TestCase):
    def test_bm25_has_term_frequency_df_and_length_normalization(self):
        corpus = ["rare rare common", "common common common", "common filler filler filler filler filler"]
        scores = bm25_scores("rare", corpus)
        expected_idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
        expected = expected_idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 4))
        self.assertAlmostEqual(scores[0], expected)
        self.assertEqual(scores[1:], [0.0, 0.0])
        short_long = bm25_scores("common", ["common", "common padding padding padding"])
        self.assertGreater(short_long[0], short_long[1])

    def test_top20_channels_merge_and_generic_backfill_reaches60(self):
        pool = build_candidate_pool("Project tea", facts(), {}, [1.0, 0.0])
        self.assertEqual(len(pool), 60)
        self.assertEqual(len({fact["id"] for fact in pool}), 60)
        channels = [item["channel"] for fact in pool for item in fact["channel_provenance"]]
        self.assertEqual(channels.count("dense"), 20)
        self.assertEqual(channels.count("bm25"), 20)
        self.assertIn("generic_backfill", channels)
        self.assertFalse(any(fact["routing"]["typed_enabled"] for fact in pool))

    def test_soft_types_are_semantic_scores_not_entity_metadata(self):
        rows = facts(70)
        # Keep ordinary retrievers tied. A later fact with high preference score
        # becomes reachable through types while unknown facts remain eligible.
        for row in rows:
            row["embedding"] = [1.0, 0.0]
        tags = {"f025": {"coarse_scores": {"preference": 1.0}, "fine_scores": {}, "entities": ["ignored"]}}
        pool = build_candidate_pool("tea", rows, tags, [1.0, 0.0], query_type_scores={"preference": 1.0})
        typed = [fact for fact in pool if "typed" in fact["routing"]["channels"]]
        self.assertEqual(len(typed), 20)
        self.assertTrue(any(fact["id"] == "f025" for fact in typed))
        self.assertEqual(len(pool), 60)
        self.assertTrue(any(fact["id"] not in tags for fact in pool))
        with self.assertRaisesRegex(ValueError, "unknown semantic type"):
            build_candidate_pool("tea", rows, tags, [1.0, 0.0], query_type_scores={"entities": 1.0})

    def test_missing_types_are_neutral_and_do_not_change_storage_tie_order(self):
        rows = facts(70)
        ordinary = build_candidate_pool("Person 59", rows, {}, [1.0, 0.9])
        missing = build_candidate_pool("Person 59", rows, {}, [1.0, 0.9], query_type_scores={"preference": 1.0})
        # A neutral channel follows generic relevance instead of treating absent
        # labels as hard-negative or using the input position as a type score.
        top_typed = sorted(missing, key=lambda fact: fact["routing"]["channel_ranks"]["typed"])[0]
        base = max(missing, key=lambda fact: fact["routing"]["base_rrf_score"])
        self.assertEqual(top_typed["id"], base["id"])
        self.assertEqual(len(ordinary), len(missing))

    def test_same_pool_can_be_reused_by_selection_arms_without_mutation(self):
        rows = facts(70)
        original = copy.deepcopy(rows)
        pool = build_candidate_pool("Project 2", rows, {}, [1.0, 0.0])
        rank = select("Project 2", pool, method="rank")
        mmr = select("Project 2", pool, method="mmr")
        self.assertEqual(rank.trace["pool_hash"], mmr.trace["pool_hash"])
        self.assertEqual(rows, original)

    def test_exact_text_duplicates_keep_both_ids_and_multiple_channels(self):
        rows = facts(22)
        rows.append({**rows[0], "id": "duplicate"})
        pool = build_candidate_pool("Project", rows, {}, [1.0, 0.0])
        self.assertEqual(len(pool), 22)
        duplicate = next(fact for fact in pool if fact["id"] == "f000")
        self.assertEqual(duplicate["origin_ids"], ["f000", "duplicate"])

    def test_planner_receives_only_question_and_visible_facts(self):
        rows = facts(2)
        rows[1].update({"answer": "LEAK_SENTINEL", "gold_evidence": "LEAK_SENTINEL", "source_context": "LEAK_SENTINEL"})
        api = MockAPI({"gaps": [gap()]})
        plan = plan_gaps("Who leads the project?", rows, api)
        payload, kwargs = api.calls[0]
        self.assertNotIn("LEAK_SENTINEL", json.dumps(payload))
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(kwargs["stage"], "gap_planner")
        self.assertTrue(kwargs["json_output"])
        self.assertFalse(plan["trace"]["fallback"])
        self.assertEqual(plan["gaps"][0]["retrieval_query"], "Project 1 project leader")

    def test_ungrounded_anchor_or_constraint_is_rejected_not_used(self):
        wrong_anchor = gap(anchor="Invented Project")
        wrong_constraint = gap()
        wrong_constraint["constraints"] = ["in 2099"]
        api = MockAPI({"gaps": [wrong_anchor, wrong_constraint]})
        result = plan_gaps("Who leads the project?", facts(2), api)
        self.assertEqual(len(result["trace"]["rejected_gaps"]), 2)
        self.assertTrue(result["trace"]["fallback"])
        self.assertEqual(result["gaps"][0]["retrieval_query"], "Who leads the project?")

    def test_planner_gap_bound_and_api_errors_are_not_hidden(self):
        api = MockAPI({"gaps": [gap() for _ in range(6)]})
        result = plan_gaps("Who leads the project?", facts(2), api)
        self.assertEqual(len(result["gaps"]), 1)
        self.assertEqual(len(result["trace"]["rejected_gaps"]), 5)

        class FailedAPI:
            def generate(self, *args, **kwargs):
                raise RuntimeError("planner failed")

        with self.assertRaisesRegex(RuntimeError, "planner failed"):
            plan_gaps("Who?", facts(2), FailedAPI())

    def test_single_expansion_adds_at_most20_keeps_original_order_and_pool80(self):
        rows = facts(100)
        pool = build_candidate_pool("tea", rows, {}, [1.0, 0.0])
        api = MockAPI({"gaps": [gap()]})
        plan = plan_gaps("Who leads Project 1?", rows[:2], api)
        expanded = expand_pool("tea", rows, {}, [1.0, 0.0], pool, plan)
        self.assertEqual(len(expanded), 80)
        self.assertEqual([f["id"] for f in expanded[:60]], [f["id"] for f in pool])
        self.assertEqual(len({f["id"] for f in expanded}), 80)
        self.assertEqual(sum(f["expansion"]["is_new"] for f in expanded), 20)
        self.assertEqual(expanded[0]["expansion"]["gap_modes"]["gap_1"], "bm25_typed_no_gap_embedding")
        with self.assertRaisesRegex(ValueError, "only one expansion"):
            expand_pool("tea", rows, {}, [1.0, 0.0], expanded, plan)

    def test_expansion_uses_given_gap_vector_not_original_query_vector(self):
        rows = facts(100)
        pool = build_candidate_pool("tea", rows, {}, [1.0, 0.0])
        plan = plan_gaps("Who leads Project 1?", rows[:2], MockAPI({"gaps": [gap()]}))
        expanded = expand_pool("tea", rows, {}, [1.0, 0.0], pool, plan, gap_embeddings={"gap_1": [0.0, 1.0]})
        self.assertEqual(expanded[-1]["expansion"]["gap_modes"]["gap_1"], "bm25_dense_typed")
        self.assertTrue(any(fact["id"] == "f099" for fact in expanded[60:]))

    def test_expansion_bm25_keeps_full_corpus_statistics(self):
        rows = facts(100)
        rows[95]["text"] = "Project 1 project leader rare rare rare."
        pool = build_candidate_pool("tea", rows, {}, [1.0, 0.0])
        plan = plan_gaps("Who leads Project 1?", rows[:2], MockAPI({"gaps": [gap()]}))
        expanded = expand_pool("tea", rows, {}, [1.0, 0.0], pool, plan)
        expected_scores = bm25_scores(plan["gaps"][0]["retrieval_query"], [fact["text"] for fact in rows])
        target = next(fact for fact in expanded[60:] if fact["id"] == "f095")
        self.assertAlmostEqual(target["routing"]["channel_raw_scores"]["bm25"], expected_scores[95])

    def test_cap_small_library_and_no_duplicate_additions(self):
        rows = facts(65)
        pool = build_candidate_pool("tea", rows, {}, [1.0, 0.0])
        plan = plan_gaps("Who?", rows[:2], MockAPI({"gaps": []}))
        expanded = expand_pool("Who?", rows, {}, [1.0, 0.0], pool, plan)
        self.assertEqual(len(expanded), 65)
        self.assertEqual(expanded[0]["expansion"]["new_count"], 5)
        self.assertEqual(expanded[0]["expansion"]["gap_modes"]["query_fallback"], "bm25_dense_typed")
        with self.assertRaisesRegex(ValueError, "frozen expansion limits"):
            expand_pool("Who?", rows, {}, [1.0, 0.0], pool, plan, max_new=21)

    def test_bad_vectors_or_tags_raise_instead_of_silent_routing(self):
        rows = facts(3)
        rows[0]["embedding"] = [0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "nonzero norms"):
            build_candidate_pool("tea", rows, {}, [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "invalid type score"):
            build_candidate_pool("tea", facts(3), {}, [1.0, 0.0], query_type_scores={"semantic": float("nan")})


if __name__ == "__main__":
    unittest.main(verbosity=2)
