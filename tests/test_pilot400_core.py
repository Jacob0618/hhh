import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_mem.data import ROOT, load_data
from jev_mem.pilot400 import (COHORT_SHA256, fingerprint, load_cohort, query_type_scores,
                              reader_fact, retrieve_rank20, type_one_fact, validate_tags)
from jev_mem.selection import COARSE_CRITERIA, FINE_CRITERIA


def fact(index):
    source = {"source_turn_id": f"s{index}", "session_id": "session_1", "speaker": "A",
              "text": f"I have item {index}", "observation_date": "2023-01-01"}
    return {"id": f"item{index}", "text": f"A has item {index}", "embedding": [1.0] + [0.0] * 767,
            "source_records": [source], "proposed_support": [source["source_turn_id"]],
            "source_observation_dates": {source["source_turn_id"]: source["observation_date"]},
            "observation_dates": [source["observation_date"]], "temporal_status": "not_temporal"}


class API:
    def __init__(self):
        self.calls = []

    def score_many(self, state, criteria, **kwargs):
        self.calls.append((state, criteria, kwargs))
        return {key: .5 for key in criteria}


class PilotCoreTests(unittest.TestCase):
    def test_real_manifest_and_original_indices(self):
        manifest, rows = load_cohort(ROOT / "configs/pilot400_manifest_v1.json", load_data())
        self.assertEqual(len(rows), 400)
        self.assertEqual(manifest["sample"]["query_ids_sha256"], COHORT_SHA256)
        self.assertEqual(rows[0]["source_qa_index"], 1)
        self.assertEqual(len({r["conversation"] for r in rows}), 10)

    def test_resampling_rejected(self):
        original = json.loads((ROOT / "configs/pilot400_manifest_v1.json").read_text(encoding="utf-8"))
        original["query_ids"][0] = "c0_q0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cohort(path, load_data())

    def test_one_fact_one_request_ten_scores_no_full_session(self):
        api, f = API(), fact(1)
        f["source_context"] = "UNRELATED_ENTIRE_SESSION"
        row = type_one_fact(f, api)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(len(api.calls[0][1]), 10)
        self.assertNotIn("UNRELATED_ENTIRE_SESSION", api.calls[0][0])
        self.assertIn("I have item 1", api.calls[0][0])
        self.assertEqual(row["fact_sha256"], fingerprint(f))
        validate_tags([f], {f["id"]: row})

    def test_gold_in_source_rejected(self):
        f, api = fact(1), API()
        f["source_records"][0]["answer"] = "gold"
        with self.assertRaises(ValueError):
            type_one_fact(f, api)
        self.assertEqual(api.calls, [])

    def test_query_typing_one_request_ten_criteria(self):
        api = API()
        scores, trace = query_type_scores("What did A plan?", api)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(set(scores), set(COARSE_CRITERIA) | set(FINE_CRITERIA))
        self.assertEqual(trace["criteria_evaluations"], 10)

    def test_local_retrieval_has_no_hidden_planner_or_score_calls(self):
        facts = [fact(i) for i in range(70)]
        with patch("jev_mem.routing.plan_gaps", side_effect=AssertionError("hidden planner")), \
             patch("jev_mem.selection.select", side_effect=AssertionError("hidden selector")):
            result = retrieve_rank20("item 13", facts, facts[0]["embedding"])
        self.assertEqual(len(result["pool"]), 60)
        self.assertEqual(len(result["selected"]), 20)
        self.assertEqual(result["trace"]["model_calls_made_by_retrieval"], 0)

    def test_typed_untyped_use_equal_pool_and_selection_budget(self):
        facts, api = [fact(i) for i in range(62)], API()
        tags = {f["id"]: type_one_fact(f, api) for f in facts}
        needs, _ = query_type_scores("item", api)
        before = len(api.calls)
        untyped = retrieve_rank20("item", facts, facts[0]["embedding"])
        typed = retrieve_rank20("item", facts, facts[0]["embedding"], tags=tags, needs=needs)
        self.assertEqual(len(typed["pool"]), len(untyped["pool"]))
        self.assertEqual(len(api.calls), before)
        self.assertEqual(len(typed["selected"]), 20)
        del tags[facts[0]["id"]]
        with self.assertRaises(ValueError):
            retrieve_rank20("item", facts, facts[0]["embedding"], tags=tags, needs=needs)

    def test_changed_fact_tags_rejected(self):
        f, api = fact(1), API()
        tag = type_one_fact(f, api)
        f["text"] += " changed"
        with self.assertRaises(ValueError):
            validate_tags([f], {f["id"]: tag})

    def test_duplicate_date_provenance_kept_in_one_slot(self):
        f = fact(1)
        other = copy.deepcopy(f)
        other.update(id="other", source_observation_dates={"s2": "2023-02-01"})
        f["duplicate_variants"] = [other]
        combined = reader_fact(f)
        self.assertEqual(combined["observation_dates"], ["2023-01-01", "2023-02-01"])
        self.assertIsNone(combined["observation_date"])
        self.assertEqual(combined["source_observation_dates"]["s2"], "2023-02-01")
        self.assertEqual(f["source_observation_dates"], {"s1": "2023-01-01"})

    def test_duplicate_conflicting_date_rejected(self):
        f = fact(1)
        other = copy.deepcopy(f)
        other["source_observation_dates"]["s1"] = "2023-02-01"
        f["duplicate_variants"] = [other]
        with self.assertRaises(ValueError):
            reader_fact(f)

    def test_null_scalar_not_filled_from_one_source_date(self):
        f = fact(1)
        f["observation_date"] = None
        self.assertIsNone(reader_fact(f)["observation_date"])

    def test_unicode_equivalent_fact_keeps_both_source_dates(self):
        f = fact(1)
        f["text"] = "A likes caf\u00e9."
        other = copy.deepcopy(f)
        other.update(id="other", text="A likes cafe\u0301.",
                     source_observation_dates={"s2": "2023-02-01"})
        f["duplicate_variants"] = [other]
        self.assertEqual(reader_fact(f)["observation_dates"], ["2023-01-01", "2023-02-01"])

    def test_short_bank_does_not_pad_or_resample(self):
        facts = [fact(i) for i in range(19)]
        with self.assertRaises(ValueError):
            retrieve_rank20("item", facts, facts[0]["embedding"])


if __name__ == "__main__":
    unittest.main()
