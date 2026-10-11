"""Network-free tests using the actual standalone upstream scoring function."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.sdk20_type_adapter import capture_native_pool, typed_select
from jev_mem.selection import COARSE_CRITERIA, FINE_CRITERIA

spec = importlib.util.spec_from_file_location("offline_native_scoring", ROOT / "upstream/mem0/mem0/utils/scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def query_scores(value=0.0):
    return {key: value for key in (*COARSE_CRITERIA, *FINE_CRITERIA)}


def tags_for(rows, value=0.0):
    return {row["id"]: {"coarse_scores": {key: value for key in COARSE_CRITERIA},
                        "fine_scores": {key: value for key in FINE_CRITERIA}} for row in rows}


def candidates(count=80):
    return [{"id": f"f{i:03d}", "score": 0.95 - i * 0.005,
             "payload": {"data": f"fact {i}", "untouched": [i]}} for i in range(count)]


class FakeMemory:
    def __init__(self, rows=None):
        self.rows = candidates() if rows is None else rows
        self.search_calls = []
        self.scorer_calls = []
        self.baseline_result = None

        def original(*args, **kwargs):
            top_k = kwargs["top_k"] if "top_k" in kwargs else args[4]
            self.scorer_calls.append(top_k)
            result = scoring.score_and_rank(*args, **kwargs)
            if top_k == 20:
                self.baseline_result = result
            return result

        self.module = SimpleNamespace(score_and_rank=original)

    def search(self, *, top_k=20, threshold=0.1, bm25=None, entities=None):
        self.search_calls.append(top_k)
        rows = self.rows[:max(top_k * 4, 60)]
        return self.module.score_and_rank(semantic_results=rows, bm25_scores=bm25 or {},
            entity_boosts=entities or {}, threshold=threshold, top_k=top_k, explain=True)


def captured_memory(rows=None):
    memory = FakeMemory(rows)
    with capture_native_pool(_scorer_module=memory.module) as captured:
        result = memory.search()
    return memory, captured, result


class SDK20TypeAdapterTests(unittest.TestCase):
    def test_capture_preserves_native20_and_scores_without_larger_search(self):
        memory = FakeMemory()
        original = memory.module.score_and_rank
        before = copy.deepcopy(memory.rows)
        with capture_native_pool(_scorer_module=memory.module) as captured:
            result = memory.search()
            self.assertIs(result, memory.baseline_result)
        self.assertIs(memory.module.score_and_rank, original)
        self.assertEqual(memory.search_calls, [20])
        self.assertEqual(memory.scorer_calls, [20, 80])
        self.assertEqual(captured["base20"], result)
        self.assertEqual(captured["eligible"][:20], result)
        self.assertEqual(len(captured["eligible"]), 80)
        self.assertEqual(captured["native_scores"], {row["id"]: row["score"] for row in captured["eligible"]})
        self.assertEqual(memory.rows, before)
        result[0]["payload"]["untouched"].append("later mutation")
        self.assertNotEqual(captured["base20"], result)
        self.assertEqual(captured["inputs"]["semantic_results"], before)

    def test_original_semantic_threshold_cannot_be_overridden_by_types_or_keyword(self):
        rows = candidates()
        rows[0]["score"] = 0.099
        rows[1]["score"] = 0.1
        memory = FakeMemory(rows)
        with capture_native_pool(_scorer_module=memory.module) as captured:
            memory.search(bm25={"f000": 1.0, "keyword_only": 1.0}, entities={"f000": 0.5})
        self.assertEqual(len(captured["candidate_ids"]), 80)
        self.assertNotIn("f000", captured["eligible_ids"])
        self.assertIn("f001", captured["eligible_ids"])
        tags = tags_for(rows)
        tags["f000"]["coarse_scores"]["preference"] = 1.0
        query = query_scores()
        query["preference"] = 1.0
        selected = typed_select(captured, tags, query)["selected_ids"]
        self.assertNotIn("f000", selected)
        self.assertNotIn("keyword_only", selected)
        self.assertTrue(set(selected).issubset(captured["eligible_ids"]))

    def test_soft_signal_changes_membership_from_original_eighty_without_mutation(self):
        memory, captured, _ = captured_memory()
        tags = tags_for(memory.rows)
        tags["f020"]["coarse_scores"]["preference"] = 1.0
        query = query_scores()
        query["preference"] = 1.0
        before = copy.deepcopy((captured, tags, query, memory.rows))
        result = typed_select(captured, tags, query)
        self.assertEqual(result["trace"]["entered_ids"], ["f020"])
        self.assertNotIn("f019", result["selected_ids"])
        self.assertEqual(len(result["selected_ids"]), 20)
        self.assertEqual((captured, tags, query, memory.rows), before)
        trace_row = next(row for row in result["trace"]["ranking_in_native_order"] if row["id"] == "f020")
        self.assertEqual(trace_row["native_rank"], 21)
        self.assertEqual(trace_row["type_rank"], 1)
        self.assertAlmostEqual(trace_row["soft_score"], 2 / 81 + 1 / 61)
        low_rows = [row for row in result["trace"]["ranking_in_native_order"] if row["id"] != "f020"]
        self.assertEqual({row["type_rank"] for row in low_rows}, {41.0})

    def test_equal_affinity_and_zero_query_strictly_preserve_native_order(self):
        memory, captured, _ = captured_memory()
        tags = tags_for(memory.rows, 0.37)
        for query in (query_scores(), query_scores(0.4)):
            with self.subTest(query=query):
                result = typed_select(captured, tags, query)
                self.assertEqual(result["selected_ids"], captured["base20_ids"])
                self.assertEqual(result["trace"]["entered_ids"], [])
                self.assertIsNotNone(result["trace"]["fallback_reason"])
        result = typed_select(captured, tags, query_scores(0.4))
        self.assertEqual({row["type_rank"] for row in result["trace"]["ranking_in_native_order"]}, {40.5})

    def test_missing_and_invalid_labels_are_hard_errors_even_for_zero_query(self):
        memory, captured, _ = captured_memory()
        complete = tags_for(memory.rows)
        for bad in (None, True, -0.1, 1.1, float("nan"), float("inf"), "0.5"):
            with self.subTest(bad=bad):
                tags = copy.deepcopy(complete)
                tags["f000"]["fine_scores"]["temporal_constraint"] = bad
                with self.assertRaises(ValueError):
                    typed_select(captured, tags, query_scores())
                query = query_scores()
                query["semantic"] = bad
                with self.assertRaises(ValueError):
                    typed_select(captured, complete, query)
        for remove_fact in (False, True):
            tags = copy.deepcopy(complete)
            if remove_fact:
                del tags["f000"]
            else:
                del tags["f000"]["coarse_scores"]["semantic"]
            with self.assertRaisesRegex(ValueError, "all 10"):
                typed_select(captured, tags, query_scores())
        with self.assertRaisesRegex(ValueError, "all 10"):
            typed_select(captured, complete, {"semantic": 1.0})
        unknown = query_scores()
        unknown["made_up"] = 1.0
        with self.assertRaisesRegex(ValueError, "unknown semantic"):
            typed_select(captured, complete, unknown)

    def test_finally_restores_after_body_or_original_scorer_exception(self):
        memory = FakeMemory()
        original = memory.module.score_and_rank
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with capture_native_pool(_scorer_module=memory.module):
                memory.search()
                raise RuntimeError("body failed")
        self.assertIs(memory.module.score_and_rank, original)

        def failed(*args, **kwargs):
            raise RuntimeError("scorer failed")

        memory.module.score_and_rank = failed
        with self.assertRaisesRegex(RuntimeError, "scorer failed"):
            with capture_native_pool(_scorer_module=memory.module):
                memory.search()
        self.assertIs(memory.module.score_and_rank, failed)
        # The process-level guard was also released after both failures.
        captured_memory()

    def test_rejects_nested_repeated_wrong_topk_and_oversized_capture(self):
        memory = FakeMemory()
        original = memory.module.score_and_rank
        with capture_native_pool(_scorer_module=memory.module):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with capture_native_pool(_scorer_module=memory.module):
                    pass
            memory.search()
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                memory.search()
        self.assertIs(memory.module.score_and_rank, original)
        for top_k, rows in ((60, candidates()), (20, candidates(81))):
            with self.subTest(top_k=top_k, count=len(rows)):
                with self.assertRaises(ValueError):
                    with capture_native_pool(_scorer_module=memory.module):
                        memory.module.score_and_rank(rows, {}, {}, 0.1, top_k)
                self.assertIs(memory.module.score_and_rank, original)

    def test_rejects_cross_thread_capture_and_restores_after_unused_context(self):
        memory = FakeMemory()
        original = memory.module.score_and_rank
        errors = []
        with self.assertRaisesRegex(RuntimeError, "without a successful"):
            with capture_native_pool(_scorer_module=memory.module):
                def other_thread():
                    try:
                        memory.search()
                    except RuntimeError as exc:
                        errors.append(str(exc))
                thread = threading.Thread(target=other_thread)
                thread.start()
                thread.join()
        self.assertTrue(any("owning thread" in error for error in errors))
        self.assertIs(memory.module.score_and_rank, original)

    def test_empty_or_small_pool_and_modified_capture(self):
        for count in (0, 7):
            memory, captured, _ = captured_memory(candidates(count))
            result = typed_select(captured, tags_for(memory.rows), query_scores())
            self.assertEqual(result["selected_ids"], captured["base20_ids"])
            self.assertEqual(len(result["selected_ids"]), count)
        _, captured, _ = captured_memory()
        captured["eligible"][0]["score"] = 999
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            typed_select(captured, {}, query_scores())


if __name__ == "__main__":
    unittest.main()
