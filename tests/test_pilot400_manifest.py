"""Offline scientific cohort invariants; no providers or model calls."""

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_pilot400_manifest import (build_manifest, eligible_rows,
    hamilton_allocations, ids_sha256, sampling_key, select_rows)


class Pilot400ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "data" / "locomo10.json").read_text("utf-8"))
        cls.rows = eligible_rows(cls.data)
        cls.manifest = build_manifest()

    def test_exact_400_unique_questions_from_fixed_eligible_parent(self):
        selected = self.manifest["query_ids"]
        parent = json.loads((ROOT / "research" / "locomo_1540_manifest.json").read_text("utf-8"))
        self.assertEqual(len(self.rows), 1540)
        self.assertEqual([row["query_id"] for row in self.rows], parent["included_query_ids_in_source_order"])
        self.assertEqual(len(selected), 400)
        self.assertEqual(len(set(selected)), 400)
        self.assertTrue(set(selected).issubset(parent["included_query_ids_in_source_order"]))
        self.assertEqual({row["conversation"] for row in self.manifest["questions"]}, set(range(10)))
        self.assertEqual({row["category"] for row in self.manifest["questions"]}, {1, 2, 3, 4})

    def test_hamilton_allocation_error_strictly_below_one_in_every_stratum(self):
        strata = self.manifest["strata"]
        self.assertEqual(sum(row["sample_count"] for row in strata), 400)
        for row in strata:
            # Exact arithmetic verifies the inequality, independent of floats.
            self.assertLess(abs(row["sample_count"] * 1540 - 400 * row["population_count"]), 1540)
        empty = next(row for row in strata if (row["conversation"], row["category"]) == (1, 3))
        self.assertEqual(empty["sample_count"], 0)

    def test_selection_deterministic_and_independent_of_input_order(self):
        shuffled = self.rows[:]
        random.Random(77).shuffle(shuffled)
        baseline, _ = select_rows(self.rows)
        repeated, _ = select_rows(shuffled)
        self.assertEqual(baseline, repeated)
        self.assertEqual(self.manifest, build_manifest())

    def test_no_question_gold_output_or_success_dependence(self):
        modified = copy.deepcopy(self.rows)
        for index, row in enumerate(modified):
            row.update(question="different text", question_utf8_sha256="different hash",
                       answer=str(index), evidence=["any"], model_output="wrong",
                       score=index, success=(index % 2 == 0))
        original, _ = select_rows(self.rows)
        altered, _ = select_rows(modified)
        self.assertEqual([row["query_id"] for row in original], [row["query_id"] for row in altered])

    def test_all_arm_pairing_contract_and_source_identity(self):
        expected = self.manifest["query_ids"]
        pairing = self.manifest["pairing"]
        self.assertTrue(pairing["all_arms_use_identical_query_ids"])
        self.assertEqual(pairing["required_query_count_per_arm"], 400)
        self.assertEqual(pairing["required_query_ids_sha256"], ids_sha256(expected))
        self.assertEqual([row["query_id"] for row in self.manifest["questions"]], expected)
        for row in self.manifest["questions"]:
            ci, qi = row["conversation"], row["source_qa_index"]
            source = self.data[ci]["qa"][qi]
            self.assertEqual(row["query_id"], f"c{ci}_q{qi}")
            self.assertEqual(row["category"], int(source["category"]))
            self.assertEqual(row["question_utf8_sha256"], hashlib.sha256(source["question"].encode("utf-8")).hexdigest())
            self.assertEqual(row["development_accessible"], ci in (0, 1))
        self.assertFalse(self.manifest["exposure"]["is_blind_holdout"])

    def test_hash_ranking_selects_exactly_smallest_keys_per_stratum(self):
        ids = set(self.manifest["query_ids"])
        for stratum in self.manifest["strata"]:
            rows = [row for row in self.rows if (row["conversation"], row["category"]) ==
                    (stratum["conversation"], stratum["category"])]
            ranked = sorted(rows, key=lambda row: (sampling_key(self.manifest["sampling"]["seed"], row["query_id"]), row["query_id"]))
            chosen = {row["query_id"] for row in ranked[:stratum["sample_count"]]}
            self.assertEqual(chosen, ids.intersection(row["query_id"] for row in rows))

    def test_saved_manifest_matches_rebuild(self):
        saved = json.loads((ROOT / "configs" / "pilot400_manifest_v1.json").read_text("utf-8"))
        self.assertEqual(saved, self.manifest)
        forbidden = {"answer", "gold", "evidence", "question", "model_output", "score", "success"}
        for row in saved["questions"]:
            self.assertFalse(forbidden.intersection(row))

    def test_invalid_allocations_and_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            hamilton_allocations({(0, 1): 2}, 3)
        with self.assertRaises(ValueError):
            hamilton_allocations({(0, 1): -1}, 0)
        with self.assertRaises(ValueError):
            select_rows(self.rows + [self.rows[0]])


if __name__ == "__main__":
    unittest.main()
