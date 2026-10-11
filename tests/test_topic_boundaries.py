"""Offline boundary/provenance checks; no API client or credentials."""
import copy
import json
import unittest

from jev_mem.topic_boundaries import boundary_requests, segment_sessions


def turns():
    texts = ["I booked a flight to Paris.", "When will you leave?", "On Friday.",
             "My new project at work starts Monday.", "I will lead its design team.",
             "Our cat has a vet appointment tomorrow."]
    return [{"source_turn_id": "t" + str(i), "session_id": "s1", "speaker": "Alice" if i % 2 == 0 else "Bob",
             "observation_date": "2023-01-01", "text": text} for i, text in enumerate(texts)]


class FakeJudge:
    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def score_many(self, state, criteria, **kwargs):
        payload = json.loads(state)
        self.calls.append((payload, criteria, kwargs))
        key = (payload["boundary_before"], kwargs["stage"])
        return self.values.get(key, {"topic_shift": .1, "right_depends_on_left": .8})


def segment(source, judge, **kwargs):
    return segment_sessions(source, judge=judge, cut_threshold=.75, keep_threshold=.35,
                            dependency_threshold=.5, **kwargs)


class TopicBoundaryTests(unittest.TestCase):
    def test_all_gaps_including_session_internal_changes(self):
        source = turns()
        original = copy.deepcopy(source)
        requests = boundary_requests(source)
        self.assertEqual([row["state"]["boundary_before"] for row in requests], ["t1", "t2", "t3", "t4", "t5"])
        self.assertEqual([t["source_turn_id"] for t in requests[2]["state"]["left_window"]], ["t1", "t2"])
        self.assertEqual([t["source_turn_id"] for t in requests[2]["state"]["right_window"]], ["t3", "t4"])
        judge = FakeJudge({("t3", "topic_boundary_initial"): {"topic_shift": .9, "right_depends_on_left": .1},
                           ("t5", "topic_boundary_initial"): {"topic_shift": .9, "right_depends_on_left": .1}})
        result = segment(source, judge)
        self.assertEqual([row["source_turn_ids"] for row in result["groups"]], [["t0", "t1", "t2"], ["t3", "t4"], ["t5"]])
        self.assertTrue(result["groups"][-1]["singleton_requires_context_review"])
        self.assertEqual(source, original)
        self.assertEqual([t for g in result["groups"] for t in g["turns"]], original)
        self.assertEqual(result["trace"]["jev_score_many_calls"], 5)
        self.assertEqual(result["trace"]["criterion_evaluations"], 10)
        self.assertFalse(result["trace"]["writer_invoked"])

    def test_windows_cannot_cross_original_sessions(self):
        source = turns()
        for turn in source[3:]:
            turn["session_id"] = "s2"
            turn["observation_date"] = "2023-01-02"
        requests = boundary_requests(source)
        self.assertEqual(len(requests), 4)
        self.assertNotIn("t3", [r["state"]["boundary_before"] for r in requests])
        for request in requests:
            state = request["state"]
            self.assertEqual({t["session_id"] for t in state["left_window"] + state["right_window"]}, {state["session_id"]})
        result = segment(source, FakeJudge())
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["groups"][1]["observation_date"], "2023-01-02")

    def test_dependency_veto_and_unresolved_gray_zone(self):
        judge = FakeJudge({("t2", "topic_boundary_initial"): {"topic_shift": .95, "right_depends_on_left": .8},
                           ("t3", "topic_boundary_initial"): {"topic_shift": .5, "right_depends_on_left": .1},
                           ("t3", "topic_boundary_expanded"): {"topic_shift": .5, "right_depends_on_left": .1}})
        result = segment(turns(), judge)
        self.assertEqual(len(result["groups"]), 1)
        self.assertTrue(result["trace"]["decisions"][1]["dependency_veto"])
        self.assertTrue(result["trace"]["decisions"][2]["unresolved_shift"])
        self.assertEqual(result["trace"]["jev_score_many_calls"], 6)

    def test_expansion_uses_second_result_not_optimistic_combination(self):
        judge = FakeJudge({("t3", "topic_boundary_initial"): {"topic_shift": .5, "right_depends_on_left": .05},
                           ("t3", "topic_boundary_expanded"): {"topic_shift": .95, "right_depends_on_left": .95}})
        result = segment(turns(), judge)
        decision = result["trace"]["decisions"][2]
        self.assertEqual(len(decision["evaluations"]), 2)
        self.assertFalse(decision["cut"])
        self.assertTrue(decision["dependency_veto"])

    def test_small_session_does_not_repeat_identical_expanded_request(self):
        judge = FakeJudge({("t1", "topic_boundary_initial"): {"topic_shift": .5, "right_depends_on_left": .1}})
        result = segment(turns()[:2], judge)
        self.assertEqual(len(judge.calls), 1)
        self.assertTrue(result["trace"]["decisions"][0]["unresolved_shift"])

    def test_invalid_score_does_not_become_keep_or_cut(self):
        for bad in [True, float("nan"), -0.1, None]:
            judge = FakeJudge({("t1", "topic_boundary_initial"): {"topic_shift": bad, "right_depends_on_left": .1}})
            with self.assertRaises(ValueError):
                segment(turns(), judge)

    def test_no_qa_duplicate_or_interleaved_session_inputs(self):
        source = turns()
        source[0]["gold"] = "hidden"
        with self.assertRaises(ValueError):
            boundary_requests(source)
        source = turns()
        source[2]["source_turn_id"] = "t1"
        with self.assertRaises(ValueError):
            boundary_requests(source)
        source = turns()
        source[1]["session_id"] = "s2"
        with self.assertRaises(ValueError):
            boundary_requests(source)

    def test_empty_and_single_turn_sessions_have_no_model_calls(self):
        judge = FakeJudge()
        self.assertEqual(segment([], judge)["groups"], [])
        self.assertEqual(len(segment(turns()[:1], judge)["groups"]), 1)
        self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()
