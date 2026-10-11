"""Offline contract tests; no provider calls, credentials, downloads, or model use."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_mem.selection import (  # noqa: E402
    COARSE_CRITERIA,
    FINE_CRITERIA,
    ScoreValidationError,
    deduplicate_candidates,
    select,
    tag_fact,
)


def facts(count=60):
    return [{"id": f"f{i:03d}", "text": f"Unique fact {i}", "channel": "dense", "rank": i + 1} for i in range(count)]


class FakeJudge:
    def __init__(self, *, conditional=0.9):
        self.calls = []
        self.conditional = conditional

    def score_many(self, state, criteria):
        data = json.loads(state.split("\n", 1)[1])
        self.calls.append((data, dict(criteria)))
        index = int(data["target_fact"]["id"][1:])
        if "relevant" in criteria:
            return {"relevant": 1 - index / 100}
        if "new_evidence" in criteria:
            return {"new_evidence": self.conditional}
        return {key: 0.75 for key in criteria}


class SelectionContractTests(unittest.TestCase):
    def test_fixed_count_and_bounded_two_rounds(self):
        judge = FakeJudge()
        result = select("What happened?", facts(), judge)
        self.assertEqual(len(result.selected), 20)
        self.assertEqual(len(set(result.trace["selected_ids"])), 20)
        self.assertEqual(result.trace["judge_calls"], 60 + 32 + 32)
        self.assertEqual(result.trace["criteria_evaluations"], 124)
        self.assertEqual([len(row["selected_before"]) for row in result.trace["rounds"]], [8, 14])
        self.assertEqual([len(row["accepted_ids"]) for row in result.trace["rounds"]], [6, 6])
        self.assertEqual(result.trace["fallback_count"], 0)
        # All candidates within a round see an identical selected-set snapshot.
        conditional_calls = [data for data, criteria in judge.calls if "new_evidence" in criteria]
        self.assertTrue(all(len(row["selected_facts"]) == 8 for row in conditional_calls[:32]))
        self.assertTrue(all(len(row["selected_facts"]) == 14 for row in conditional_calls[32:]))

    def test_threshold_rejection_fills_with_rank_and_records_fallback(self):
        result = select("What happened?", facts(), FakeJudge(conditional=0.01))
        self.assertEqual(result.trace["selected_ids"], [f"f{i:03d}" for i in range(20)])
        self.assertEqual(result.trace["fallback_count"], 12)
        self.assertEqual(result.trace["fallback_fraction"], 0.6)
        self.assertTrue(all(len(row["fallback_ids"]) == 6 for row in result.trace["rounds"]))

    def test_small_pool_returns_only_real_unique_facts(self):
        for count in (0, 1, 8, 9, 14, 19, 20):
            with self.subTest(count=count):
                result = select("What happened?", facts(count), FakeJudge())
                self.assertEqual(len(result.selected), count)
                self.assertEqual(len(set(result.trace["selected_ids"])), count)

    def test_exact_dedup_preserves_channels_and_source_variants(self):
        pool = [
            {"id": "a", "text": "Ada moved in May.", "source_context": "first", "channel": "dense", "rank": 2},
            {"id": "b", "text": "Ada  moved in May.", "source_context": "second", "channel_ranks": {"bm25": 1}},
            {"id": "c", "text": "Ada did not move in May.", "channel": "bm25", "rank": 2},
            {"id": "d", "text": "Ada moved in June.", "channel": "dense", "rank": 3},
        ]
        unique = deduplicate_candidates(pool)
        self.assertEqual(len(unique), 3)
        self.assertEqual(unique[0]["origin_ids"], ["a", "b"])
        self.assertEqual(unique[0]["source_contexts"], ["first", "second"])
        self.assertEqual(len(unique[0]["duplicate_variants"]), 2)
        self.assertEqual(unique[0]["channel_provenance"], [{"channel": "dense", "rank": 2}, {"channel": "bm25", "rank": 1}])
        # The input is not mutated while constructing enriched result records.
        self.assertNotIn("origin_ids", pool[0])

    def test_version_collision_is_not_silently_merged(self):
        with self.assertRaisesRegex(ValueError, "conflicting text"):
            deduplicate_candidates([{"id": "same", "text": "x"}, {"id": "same", "text": "not x"}])

    def test_all_arms_have_same_pool_hash_and_ties_are_stable(self):
        pool = facts(25)
        pool.append({**pool[0], "id": "duplicate", "channel": "bm25"})
        results = [select("What happened?", pool, FakeJudge(), method=method) for method in ("rank", "mmr", "jev_single", "conditional")]
        self.assertEqual(len({r.trace["pool_hash"] for r in results}), 1)
        again = select("What happened?", pool, FakeJudge())
        self.assertEqual(results[-1].trace, again.trace)

    def test_typing_is_multilabel_and_preserves_unknown_role_boundary(self):
        judge = FakeJudge()
        result = tag_fact({"id": "f000", "text": "I intend to move in May.", "source_context": "Ada said she intends to move."}, judge)
        self.assertEqual(set(result["coarse_scores"]), set(COARSE_CRITERIA))
        self.assertEqual(set(result["fine_scores"]), set(FINE_CRITERIA))
        self.assertEqual(sum(result["coarse_scores"].values()), 3.0)  # not a softmax
        self.assertEqual(result["criteria_evaluations"], 10)
        self.assertNotIn("bridge", {**result["coarse_scores"], **result["fine_scores"]})
        self.assertEqual(judge.calls[0][0]["target_fact"]["source_context"], ["Ada said she intends to move."])
        self.assertFalse(result["calibrated"])

    def test_same_pool_selector_does_not_consume_hidden_source_context(self):
        judge = FakeJudge()
        pool = facts(21)
        pool[0]["source_context"] = "Extra evidence not present in the fact supplied to QA."
        select("What happened?", pool, judge)
        for data, _ in judge.calls:
            self.assertNotIn("source_context", data["target_fact"])
            self.assertNotIn("Extra evidence", json.dumps(data))

    def test_confidence_missing_nan_or_invalid_scores_raise(self):
        class BadJudge:
            def __init__(self, output):
                self.output = output

            def score_many(self, state, criteria):
                return self.output

        for output in ({"confidence": 0.99}, {"relevant": None}, {"relevant": float("nan")}, {"relevant": True}, {"relevant": 1.1}):
            with self.subTest(output=output), self.assertRaises(ScoreValidationError):
                select("Question?", facts(1), BadJudge(output), method="jev_single")

    def test_provider_failure_cannot_be_recorded_as_selection_fallback(self):
        class FailedJudge:
            def score_many(self, state, criteria):
                raise RuntimeError("provider unavailable")

        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            select("Question?", facts(), FailedJudge())

    def test_mmr_labels_its_actual_similarity_and_rejects_bad_vectors(self):
        lexical = select("Question?", facts(21), method="mmr")
        self.assertEqual(lexical.trace["mmr_similarity"], "lexical_jaccard")
        pool = facts(21)
        for i, fact in enumerate(pool):
            fact["embedding"] = [1.0, float(i)]
        vector = select("Question?", pool, method="mmr")
        self.assertEqual(vector.trace["mmr_similarity"], "embedding_cosine")
        pool[2]["embedding"] = [0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "nonzero"):
            select("Question?", pool, method="mmr")

    def test_impossible_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bounded conditional schedule"):
            select("Question?", facts(), FakeJudge(), top_k=21)

    def test_parallel_completion_preserves_results_and_round_barriers(self):
        class DelayedJudge(FakeJudge):
            def __init__(self):
                super().__init__()
                self.lock = threading.Lock()
                self.active = 0
                self.peak_active = 0
                self.completed = []

            def score_many(self, state, criteria):
                data = json.loads(state.split("\n", 1)[1])
                index = int(data["target_fact"]["id"][1:])
                with self.lock:
                    self.active += 1
                    self.peak_active = max(self.peak_active, self.active)
                time.sleep(0.001 * (1 + (7 - index % 8)))
                result = super().score_many(state, criteria)
                with self.lock:
                    self.active -= 1
                    self.completed.append((data["task"], index, len(data.get("selected_facts", []))))
                return result

        judge = DelayedJudge()
        parallel = select("Question?", facts(45), judge, max_workers=8)
        serial = select("Question?", facts(45), FakeJudge(), max_workers=1)
        self.assertEqual(parallel.trace, serial.trace)
        self.assertGreater(judge.peak_active, 1)
        self.assertLessEqual(judge.peak_active, 8)
        rounds = [size for task, _, size in judge.completed if task == "conditional_new_evidence"]
        # No round-2 work can overtake unfinished round-1 work.
        self.assertEqual(rounds, sorted(rounds))

    def test_parallel_failure_is_raised_without_zero_substitution(self):
        class FailedCandidate(FakeJudge):
            def score_many(self, state, criteria):
                data = json.loads(state.split("\n", 1)[1])
                if data["target_fact"]["id"] == "f005":
                    raise RuntimeError("one candidate failed")
                return super().score_many(state, criteria)

        with self.assertRaisesRegex(RuntimeError, "one candidate failed"):
            select("Question?", facts(), FailedCandidate(), max_workers=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
