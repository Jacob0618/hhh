"""Offline mocks only: no credentials, model clients or networking."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_mem.group_writer import prepare_group
from jev_mem.sparse_topic_pipeline import (
    SparseTopicPipelineError, regroup_atomic_segments,
    run_sparse_topic_pipeline, segment_sparse_sessions,
)
from jev_mem.topic_grouping import render_group, verify_partition


def turns(sizes=(21,), text_size=20):
    result = []
    for session, size in enumerate(sizes):
        for local in range(size):
            result.append({"source_turn_id": f"s{session}_t{local}", "session_id": f"session_{session}",
                           "speaker": "A" if local % 2 == 0 else "B", "observation_date": f"2026-09-{session + 1:02d}",
                           "text": f"turn {session}/{local}: " + "x" * text_size})
    return result


def vectors(count):
    return [[1.] + [0.] * 767 for _ in range(count)]


class Judge:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def score_many(self, state, criteria, *, stage):
        parsed = json.loads(state)
        self.calls.append({"state": parsed, "criteria": criteria, "stage": stage})
        if self.answer:
            return self.answer(parsed, criteria, stage)
        if "topic_shift" in criteria:
            return {"topic_shift": .1, "right_depends_on_left": .1}
        return {key: .9 for key in criteria}


def split(raw, judge=None, **kwargs):
    defaults = {"cut_threshold": .8, "keep_threshold": .3, "dependency_threshold": .6}
    defaults.update(kwargs)
    return segment_sparse_sessions(raw, judge=judge or Judge(), embeddings=vectors(len(raw)), **defaults)


def group(raw, segments, judge=None, **kwargs):
    defaults = {"same_topic_threshold": .7, "token_counter": len}
    defaults.update(kwargs)
    return regroup_atomic_segments(raw, segments, judge=judge or Judge(), embeddings=vectors(len(raw)), **defaults)


class SparseTopicTests(unittest.TestCase):
    def test_four_plus_two_shared_and_second_result_replaces(self):
        raw = turns()
        expanded = []

        def answer(state, criteria, stage):
            if stage.endswith("expanded"):
                expanded.append(state["boundary_before"])
                return {"topic_shift": .95 if len(expanded) == 1 else .05, "right_depends_on_left": .1}
            return {"topic_shift": .5, "right_depends_on_left": .1}

        judge = Judge(answer)
        result = split(raw, judge)
        self.assertEqual(len(judge.calls), 6)
        self.assertEqual(result["trace"]["criterion_evaluations"], 12)
        self.assertEqual([call["stage"] for call in judge.calls],
                         ["sparse_topic_boundary_initial"] * 4 + ["sparse_topic_boundary_expanded"] * 2)
        assessed = [d for d in result["trace"]["decisions"] if d["status"] == "assessed"]
        self.assertEqual(expanded, [d["boundary_before"] for d in assessed[:2]])
        self.assertTrue(assessed[0]["cut"])
        self.assertEqual(assessed[1]["scores"]["topic_shift"], .05)
        self.assertEqual(assessed[1]["decision"], "low_shift_joined")
        self.assertEqual([d["refinement"] for d in assessed[2:]], ["shared_reserve_exhausted"] * 2)

    def test_budgets_reset_per_session_but_never_multiply_refinement(self):
        raw = turns((21, 21))
        judge = Judge(lambda *_: {"topic_shift": .5, "right_depends_on_left": .1})
        result = split(raw, judge)
        self.assertEqual(len(judge.calls), 12)
        self.assertEqual([s["total_logical_calls"] for s in result["trace"]["sessions"]], [6, 6])

    def test_dependency_veto_and_unused_reserve(self):
        judge = Judge(lambda *_: {"topic_shift": .99, "right_depends_on_left": .6})
        result = split(turns(), judge)
        self.assertEqual(len(judge.calls), 4)
        self.assertEqual(result["trace"]["semantic_cut_indices"], [])
        self.assertTrue(all(d["dependency_veto"] for d in result["trace"]["decisions"] if d["status"] == "assessed"))
        self.assertEqual(result["trace"]["sessions"][0]["unused_reserve"], 2)

    def test_unassessed_gaps_are_not_certified_same_topic(self):
        result = split(turns())
        unknown = [d for d in result["trace"]["decisions"] if d["status"] == "unassessed"]
        self.assertEqual(len(unknown), 16)
        self.assertTrue(all(d["scores"] is None and not d["cut"] for d in unknown))
        self.assertEqual(result["trace"]["gap_localization_performed"], False)

    def test_no_rejudgment_if_expansion_adds_nothing(self):
        result = split(turns((2,)), Judge(lambda *_: {"topic_shift": .5, "right_depends_on_left": .1}))
        self.assertEqual(result["trace"]["logical_judge_calls"], 1)
        self.assertEqual(result["trace"]["decisions"][0]["refinement"], "no_additional_context")

    def test_session_boundaries_always_split_preserve_dates_and_originals(self):
        raw = turns((4, 5))
        before = copy.deepcopy(raw)
        result = split(raw)
        self.assertEqual([len(s["turns"]) for s in result["segments"]], [4, 5])
        self.assertEqual([s["observation_date"] for s in result["segments"]], ["2026-09-01", "2026-09-02"])
        self.assertTrue(verify_partition(raw, result["segments"]))
        self.assertEqual(raw, before)

    def test_invalid_scores_fail_with_partial_audit(self):
        judge = Judge(lambda *_: {"topic_shift": True, "right_depends_on_left": .1})
        with self.assertRaises(SparseTopicPipelineError) as caught:
            split(turns(), judge)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(caught.exception.trace["requests"][0]["status"], "error")
        self.assertEqual(caught.exception.trace["logical_judge_calls"], 1)

    def test_explicit_thresholds_and_validation_before_calls(self):
        judge = Judge()
        with self.assertRaises(TypeError):
            segment_sparse_sessions(turns(), judge=judge, embeddings=vectors(21))
        with self.assertRaises(ValueError):
            split(turns(), judge, keep_threshold=.9)
        raw = turns()
        raw[0]["question"] = "must not enter prompts"
        with self.assertRaises(ValueError):
            split(raw, judge)
        self.assertEqual(judge.calls, [])

    def test_merge_keeps_atomic_segments_and_all_raw_members_for_writer(self):
        raw = turns((5, 4))
        segments = split(raw)["segments"]
        before = copy.deepcopy(segments)
        judge = Judge()
        result = group(raw, segments, judge)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(len(result["groups"]), 1)
        merged = result["groups"][0]
        self.assertEqual(merged["segment_ids"], [s["segment_id"] for s in segments])
        self.assertEqual(merged["turns"], raw)
        self.assertIsNone(merged["observation_date"])
        self.assertEqual(merged["observation_dates"], ["2026-09-01", "2026-09-02"])
        self.assertTrue(verify_partition(raw, result["groups"]))
        prepared = prepare_group(members=merged["turns"], raw_turns=raw)
        self.assertEqual(prepared["payload"]["MEMBERS"], raw)
        self.assertEqual(segments, before)
        self.assertEqual(result["writer_grouping_method"], "jev_topics")

    def test_top_three_candidates_in_one_hook_not_three_requests(self):
        raw = turns((1, 1, 1, 1, 1, 1))
        segments = split(raw)["segments"]
        judge = Judge(lambda state, criteria, stage: {key: .1 for key in criteria})
        result = group(raw, segments, judge)
        self.assertEqual(len(judge.calls), 5)
        self.assertEqual([len(call["criteria"]) for call in judge.calls], [1, 2, 3, 3, 3])
        self.assertEqual(result["trace"]["criterion_evaluations"], 12)
        self.assertEqual(len(result["groups"]), 6)
        self.assertEqual([c["group_index"] for c in result["trace"]["decisions"][-1]["proposed_groups"]], [0, 1, 2])

    def test_anchor_segments_bounded_as_group_grows(self):
        raw = turns((1, 1, 1, 1, 1, 1))
        result = group(raw, split(raw)["segments"])
        self.assertEqual(len(result["groups"]), 1)
        for request in result["trace"]["requests"]:
            self.assertLessEqual(len(request["state"]["candidate_groups"][0]["anchors"]), 3)

    def test_excerpt_spans_bounded_and_auditable_writer_never_truncated(self):
        raw = turns((7, 7), text_size=3000)
        raw[0]["caption"] = "z" * 2000
        result = group(raw, split(raw)["segments"])
        excerpts = result["trace"]["excerpt_catalog"]
        for excerpt in excerpts:
            self.assertEqual(excerpt["selected_member_indices"], [0, 3, 6])
            self.assertEqual(excerpt["omitted_member_count"], 4)
            for turn in excerpt["turn_excerpts"]:
                original = next(row for row in raw if row["source_turn_id"] == turn["source_turn_id"])
                for name, field in turn["source_fields"].items():
                    self.assertLessEqual(sum(len(span["text"]) for span in field["spans"]), 512)
                    for span in field["spans"]:
                        self.assertEqual(span["text"], original[name][span["start"]:span["end"]])
        self.assertEqual(result["groups"][0]["turns"], raw)
        self.assertEqual(result["groups"][0]["text"], render_group(raw))

    def test_oversize_singleton_complete_and_isolated(self):
        raw = turns((2, 1))
        raw[0]["text"] = "x" * 5000
        segments = split(raw)["segments"]
        judge = Judge()
        result = group(raw, segments, judge, max_group_tokens=1000)
        self.assertEqual(len(result["groups"]), 2)
        self.assertTrue(result["groups"][0]["oversize_singleton"])
        self.assertEqual(result["groups"][0]["turns"], raw[:2])
        self.assertEqual(result["trace"]["oversize_singleton_segment_ids"], [segments[0]["segment_id"]])
        self.assertEqual(len(judge.calls), 0)

    def test_group_capacity_uses_complete_text_and_blocks_merge(self):
        raw = turns((2, 2), text_size=100)
        segments = split(raw)["segments"]
        cap = max(len(s["text"]) for s in segments)
        judge = Judge()
        result = group(raw, segments, judge, max_group_tokens=cap)
        self.assertEqual(len(result["groups"]), 2)
        self.assertFalse(any(g["oversize_singleton"] for g in result["groups"]))
        self.assertFalse(result["trace"]["decisions"][1]["capacity_checks"][0]["eligible"])
        self.assertEqual(judge.calls, [])

    def test_rejects_rewritten_or_noncontiguous_segments_before_calls(self):
        raw = turns((2, 2))
        segments = split(raw)["segments"]
        segments[0]["turns"][0]["text"] = "rewritten"
        judge = Judge()
        with self.assertRaises(ValueError):
            group(raw, segments, judge)
        with self.assertRaises(ValueError):
            group(raw, list(reversed(split(raw)["segments"])), judge)
        self.assertEqual(judge.calls, [])

    def test_combined_pipeline_totals_and_group_failure_retains_segmentation(self):
        raw = turns((5, 5))
        result = run_sparse_topic_pipeline(raw, judge=Judge(), embeddings=vectors(len(raw)), token_counter=len,
                                          cut_threshold=.8, keep_threshold=.3, dependency_threshold=.6,
                                          same_topic_threshold=.7)
        self.assertEqual(result["trace"]["logical_judge_calls"], 9)
        self.assertEqual(result["trace"]["criterion_evaluations"], 17)
        self.assertFalse(result["trace"]["qa_visible"])
        self.assertTrue(result["trace"]["offline_future_context_used"])

        def answer(state, criteria, stage):
            if "topic_shift" in criteria:
                return {"topic_shift": .1, "right_depends_on_left": .1}
            raise RuntimeError("mock request failed")

        with self.assertRaises(SparseTopicPipelineError) as caught:
            run_sparse_topic_pipeline(raw, judge=Judge(answer), embeddings=vectors(len(raw)), token_counter=len,
                                      cut_threshold=.8, keep_threshold=.3, dependency_threshold=.6, same_topic_threshold=.7)
        self.assertEqual(len(caught.exception.trace["completed_segmentation"]["segments"]), 2)
        self.assertEqual(caught.exception.trace["requests"][-1]["status"], "error")

    def test_empty_and_singleton_conversations_do_not_force_requests(self):
        self.assertEqual(split([])["segments"], [])
        raw = turns((1,))
        judge = Judge()
        result = group(raw, split(raw, judge)["segments"], judge)
        self.assertEqual(judge.calls, [])
        self.assertEqual(result["groups"][0]["turns"], raw)


if __name__ == "__main__":
    unittest.main()
