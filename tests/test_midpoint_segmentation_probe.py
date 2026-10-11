"""Pure synthetic checks; never import model providers or read credentials."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_midpoint_segmentation import (CASES, adjacent_probe, build_report,
                                         infer_gaps, midpoint_probe, run_case)


class MidpointProbeTests(unittest.TestCase):
    def test_full_tree_counts_calls_not_depth(self):
        row = midpoint_probe(21, lambda a, b: True)
        self.assertEqual(row["oracle_calls"], 20)
        self.assertEqual(row["max_node_depth"], 4)
        self.assertEqual(row["visited_midpoints"], list(range(21)))

    def test_recursive_children_always_shrink_and_do_not_contain_parent(self):
        for n in range(1, 65):
            row = midpoint_probe(n, lambda a, b: False)
            self.assertEqual(row["oracle_calls"], n - 1)
            self.assertEqual(len(row["visited_midpoints"]), n)
            for call in row["calls"]:
                lo, hi = call["child_span"]
                plo, phi = call["parent_span"]
                self.assertLess(hi - lo, phi - plo)
                self.assertTrue(plo <= lo < hi <= phi)
                self.assertFalse(lo <= call["left_index"] < hi)
                self.assertEqual(call["right_index"], (lo + hi) // 2)

    def test_one_turn_needs_no_oracle_call(self):
        for prune in (False, True):
            row = midpoint_probe(1, lambda a, b: self.fail("oracle not needed"), prune_same=prune)
            self.assertEqual(row["oracle_calls"], 0)
            self.assertTrue(row["terminated"])
            self.assertEqual(infer_gaps(row)["gaps"], [])

    def test_same_topic_prunes_to_two_comparisons_but_records_assumptions(self):
        row = run_case("one", ["A"] * 21)["strategies"][2]
        self.assertEqual(row["oracle_calls"], 2)
        self.assertEqual(row["unvisited_turn_count"], 18)
        self.assertEqual(len(row["interval_assumptions"]), 2)
        self.assertTrue(row["with_pruning_assumptions"]["evaluation"]["all_gaps_identified_correctly"])
        self.assertFalse(row["observed_evidence_only"]["evaluation"]["all_gaps_identified_correctly"])

    def test_hidden_B_is_not_removed_from_report(self):
        row = run_case("return", CASES["short_B_then_return_A"])["strategies"][2]
        self.assertEqual(row["oracle_calls"], 2)
        self.assertEqual(row["with_pruning_assumptions"]["evaluation"]["missed_boundary_after_turns"], [7, 8])
        self.assertEqual(row["with_pruning_assumptions"]["evaluation"]["falsely_inferred_same_gap_after_turns"], [7, 8])
        self.assertEqual(row["incorrect_interval_assumptions"][0]["counterexample_turn_ids"], ["t08"])

    def test_difference_is_not_transitive(self):
        probe = {"num_turns": 3, "calls": [
            {"call": 1, "left_index": 0, "right_index": 2, "same_topic": False},
            {"call": 2, "left_index": 1, "right_index": 2, "same_topic": False}],
                 "interval_assumptions": []}
        inferred = infer_gaps(probe)
        self.assertEqual(inferred["gaps"][0]["inference"], "unknown")
        self.assertEqual(inferred["gaps"][1]["inference"], "different")

    def test_full_midpoint_can_leave_gaps_unidentified(self):
        row = run_case("return", CASES["short_B_then_return_A"])["strategies"][1]
        self.assertEqual(row["oracle_calls"], 20)
        self.assertTrue(row["observed_evidence_only"]["evaluation"]["unknown_gap_after_turns"])
        self.assertFalse(row["observed_evidence_only"]["evaluation"]["all_gaps_identified_correctly"])

    def test_adjacent_scan_is_exact_for_every_constructed_case(self):
        for name, topics in CASES.items():
            row = run_case(name, topics)["strategies"][0]
            self.assertEqual(row["oracle_calls"], 20)
            self.assertTrue(row["observed_evidence_only"]["evaluation"]["all_gaps_identified_correctly"])

    def test_segmentation_keeps_every_turn_once_in_source_order(self):
        for case in build_report()["cases"]:
            for probe in case["strategies"]:
                for view in ("observed_evidence_only", "with_pruning_assumptions"):
                    segments = probe[view]["evaluation"]["segments_if_only_different_gaps_are_cut"]
                    self.assertEqual([x for segment in segments for x in segment["turn_ids"]], case["source_turn_ids"])
                    self.assertTrue(all(segment["turn_ids"] for segment in segments))

    def test_topics_and_case_definitions_are_not_mutated(self):
        source = copy.deepcopy(CASES)
        build_report()
        self.assertEqual(source, CASES)

    def test_invalid_input_and_invalid_oracle_fail(self):
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                midpoint_probe(n, lambda a, b: True)
        with self.assertRaises(TypeError):
            midpoint_probe(3, lambda a, b: .9)
        with self.assertRaises(TypeError):
            adjacent_probe(3, lambda a, b: "same")

    def test_report_explicitly_excludes_real_JEV_claims(self):
        report = build_report()
        self.assertEqual(report["model_api_calls"], 0)
        self.assertEqual(report["experiment_kind"], "synthetic_perfect_oracle_not_real_JEV")
        self.assertEqual(len(report["cases"]), 5)
        self.assertTrue(all(case["num_turns"] == 21 for case in report["cases"]))


if __name__ == "__main__":
    unittest.main()
