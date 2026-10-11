"""Offline grouping invariants with fake embeddings/JEV/token counts."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem.topic_grouping import GroupCapacityError, group_raw_turns, raw_turns_from_locomo, verify_partition


def vector(axis):
    values = [0.] * 768
    values[axis] = 1.
    return values


def turns():
    return [{"source_turn_id": f"t{i}", "session_id": "s1" if i < 2 else "s2", "speaker": "Alice",
             "observation_date": "2023-01-01" if i < 2 else "2023-01-02", "text": text}
            for i, text in enumerate(["Project Orion launch is delayed.", "My violin lesson is tomorrow.",
                                      "Project Orion launch moved to Monday.", "The violin lesson was cancelled."])]


def fake_count(text):
    # Test double only. Live callers must use the model's exact tokenizer.
    return len(text.split())


class FakeAPI:
    def __init__(self, bad_score=None):
        self.embeddings = []
        self.judgments = []
        self.bad_score = bad_score

    def embed(self, texts, **kwargs):
        self.embeddings.append((texts, kwargs))
        return [vector(0 if "Orion" in text else 1) for text in texts]

    def score_many(self, state, criteria, **kwargs):
        data = json.loads(state)
        self.judgments.append((data, criteria, kwargs))
        result = {}
        for group in data["candidate_groups"]:
            match = all(("Orion" in anchor["text"]) == ("Orion" in data["target"]["text"]) for anchor in group["anchors"])
            result[group["key"]] = self.bad_score if self.bad_score is not None else (.9 if match else .1)
        return result


class TopicGroupingTests(unittest.TestCase):
    def test_jev_preserves_every_turn_and_per_turn_dates(self):
        source = turns()
        untouched = copy.deepcopy(source)
        api = FakeAPI()
        result = group_raw_turns(source, api=api, token_counter=fake_count)
        self.assertEqual(source, untouched)
        self.assertEqual([group["source_turn_ids"] for group in result["groups"]], [["t0", "t2"], ["t1", "t3"]])
        self.assertTrue(verify_partition(source, result["groups"]))
        self.assertIsNone(result["groups"][0]["observation_date"])
        self.assertTrue(result["groups"][0]["cross_date"])
        self.assertIn("2023-01-01", result["groups"][0]["text"])
        self.assertIn("2023-01-02", result["groups"][0]["text"])
        self.assertEqual(result["trace"]["jev_threshold"], .7)
        self.assertTrue(result["trace"]["offline_full_conversation"])
        self.assertEqual(len(api.embeddings), 1)

    def test_shared_speaker_does_not_force_merging(self):
        result = group_raw_turns(turns()[:2], api=FakeAPI(), token_counter=fake_count)
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["trace"]["decisions"][1]["criterion_scores"], {"group_0": .1})

    def test_embedding_control_and_matched_contiguous(self):
        api = FakeAPI()
        topics = group_raw_turns(turns(), method="embedding_topics", api=api, token_counter=fake_count)
        self.assertEqual(api.judgments, [])
        matched = group_raw_turns(turns(), method="matched_contiguous", token_counter=fake_count,
                                 matched_group_sizes=topics["trace"]["group_turn_counts"])
        self.assertEqual([group["source_turn_ids"] for group in matched["groups"]], [["t0", "t1"], ["t2", "t3"]])
        original = group_raw_turns(turns(), method="original_sessions", token_counter=fake_count)
        self.assertEqual(original["trace"]["group_turn_counts"], [2, 2])

    def test_capacity_creates_new_group_without_splitting_turn(self):
        source = turns()

        def size_by_turn_count(text):
            return 1000 * text.count("[SOURCE ")

        result = group_raw_turns(source, api=FakeAPI(), token_counter=size_by_turn_count, max_group_tokens=1500)
        self.assertEqual(len(result["groups"]), 4)
        self.assertTrue(all(len(group["turns"]) == 1 for group in result["groups"]))
        self.assertEqual(result["trace"]["jev_score_many_calls"], 0)

    def test_oversized_turn_fails_before_model_call(self):
        api = FakeAPI()
        with self.assertRaises(GroupCapacityError):
            group_raw_turns(turns(), api=api, token_counter=lambda text: 3000)
        self.assertEqual(api.embeddings, [])
        self.assertEqual(api.judgments, [])

    def test_original_session_cap_cannot_silently_rechunk(self):
        with self.assertRaises(GroupCapacityError):
            group_raw_turns(turns(), method="original_sessions", token_counter=lambda text: 2500)
        result = group_raw_turns(turns(), method="original_sessions", token_counter=lambda text: 2500,
                                 max_group_tokens=None)
        self.assertEqual(result["trace"]["group_turn_counts"], [2, 2])
        self.assertIsNone(result["trace"]["max_group_tokens"])

    def test_candidate_and_anchor_judgment_bounds(self):
        source = []
        for i in range(30):
            source.append({"source_turn_id": str(i), "session_id": "s", "speaker": "Alice",
                           "observation_date": "2023-01-01", "text": "Project Orion step " + str(i)})
        api = FakeAPI()
        result = group_raw_turns(source, api=api, token_counter=fake_count)
        self.assertLessEqual(result["trace"]["jev_criterion_evaluations"], 3 * (len(source) - 1))
        self.assertLessEqual(len(api.judgments), len(source) - 1)
        for state, criteria, _ in api.judgments:
            self.assertLessEqual(len(criteria), 3)
            self.assertTrue(all(len(group["anchors"]) <= 3 for group in state["candidate_groups"]))

    def test_no_gold_or_duplicate_source_ids(self):
        source = turns()
        source[0]["gold_evidence"] = ["t0"]
        with self.assertRaises(ValueError):
            group_raw_turns(source, api=FakeAPI(), token_counter=fake_count)
        source = turns()
        source[1]["source_turn_id"] = source[0]["source_turn_id"]
        with self.assertRaises(ValueError):
            group_raw_turns(source, api=FakeAPI(), token_counter=fake_count)
        with self.assertRaises(ValueError):
            raw_turns_from_locomo({"qa": [], "conversation": {}})

    def test_partition_detects_rewriting_and_duplicate_assignment(self):
        source = turns()
        result = group_raw_turns(source, api=FakeAPI(), token_counter=fake_count)
        modified = copy.deepcopy(result["groups"])
        modified[0]["turns"][0]["text"] = "Invented connecting event"
        with self.assertRaises(ValueError):
            verify_partition(source, modified)
        with self.assertRaises(ValueError):
            verify_partition(source, result["groups"] + [result["groups"][0]])

    def test_invalid_jev_score_is_not_a_negative_fallback(self):
        with self.assertRaises(ValueError):
            group_raw_turns(turns(), api=FakeAPI(bad_score=float("nan")), token_counter=fake_count)

    def test_locomo_adapter_keeps_source_text_speaker_and_caption(self):
        conversation = {"speaker_a": "Alice", "speaker_b": "Bob",
                        "session_1_date_time": "10:00 am on 1 January, 2023",
                        "session_1": [{"dia_id": "D1:1", "speaker": "Bob", "text": "Original\ntext.",
                                       "blip_caption": "A blue boat.", "query": "boat photo"}]}
        units = raw_turns_from_locomo(conversation)
        self.assertEqual(units[0]["text"], "Original\ntext.")
        self.assertEqual(units[0]["speaker"], "Bob")
        self.assertEqual(units[0]["role"], "assistant")
        self.assertEqual(units[0]["caption"], "A blue boat.")
        self.assertEqual(units[0]["observation_date"], "2023-01-01")


if __name__ == "__main__":
    unittest.main()
