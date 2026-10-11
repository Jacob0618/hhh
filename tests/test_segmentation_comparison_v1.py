"""Independent small partitions with hand-computable offline expectations."""
import copy
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem.segmentation_comparison_v1 import (
    INTERPRETATION, aggregate_comparisons, boundary_agreement, clustering_agreement,
    compare_partitions, summarize_call_budget, synthetic_midpoint_counterexample,
    validate_partition,
)
from jev_mem.sparse_topic_pipeline import _group_record


def raw(sizes=(6,)):
    return [{"source_turn_id": f"s{s}_t{i}", "session_id": f"session_{s}",
             "speaker": "A" if i % 2 == 0 else "B", "observation_date": f"2026-09-{s+1:02d}",
             "text": f"original text {s} {i}"}
            for s, size in enumerate(sizes) for i in range(size)]


def groups(units, indices):
    return [_group_record(units, positions, f"g{i}") for i, positions in enumerate(indices)]


def compare(units, cseg, cgroup, rseg, rgroup, **kwargs):
    return compare_partitions(units, groups(units, cseg), groups(units, cgroup),
                              groups(units, rseg), groups(units, rgroup),
                              conversation_id=kwargs.pop("conversation_id", 0), **kwargs)


class PartitionValidationTests(unittest.TestCase):
    def test_global_noncontiguous_groups_are_allowed_but_segments_are_not(self):
        source = raw((3, 3))
        partition = groups(source, [[0, 1, 4], [2, 3, 5]])
        self.assertEqual(validate_partition(source, partition, conversation_id=0), [0, 0, 1, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_partition(source, partition, conversation_id=0, contiguous=True)

    def test_cross_session_contiguous_segment_rejected(self):
        source = raw((3, 3))
        with self.assertRaisesRegex(ValueError, "original session"):
            validate_partition(source, groups(source, [list(range(6))]), conversation_id=0, contiguous=True)

    def test_missing_duplicate_unknown_changed_source_are_rejected(self):
        source = raw()
        variants = [groups(source, [[0, 1], [2, 3, 4]]), groups(source, [[0, 1, 2], [2, 3, 4, 5]])]
        for field, value in (("source_turn_id", "foreign"), ("text", "rewritten"),
                             ("observation_date", "2026-09-09"), ("speaker", "Other")):
            bad = groups(source, [list(range(6))])
            bad[0]["turns"][0][field] = value
            variants.append(bad)
        for variant in variants:
            with self.assertRaises(ValueError):
                validate_partition(source, variant, conversation_id=0)

    def test_redundant_metadata_and_rendering_are_checked(self):
        source = raw()
        for field, value in (("text", "summary"), ("source_order_indices", [2, 1, 0, 3, 4, 5]),
                             ("observation_date", None), ("original_session_ids", ["wrong"]),
                             ("source_turn_ids", ["wrong"]), ("cross_date", True)):
            bad = groups(source, [list(range(6))])
            bad[0][field] = value
            with self.assertRaises(ValueError):
                validate_partition(source, bad, conversation_id=0)

    def test_duplicate_group_identifiers_rejected(self):
        source = raw()
        bad = groups(source, [[0, 1, 2], [3, 4, 5]])
        bad[1]["group_id"] = bad[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_partition(source, bad, conversation_id=0)

    def test_explicit_conversation_identity_cannot_be_crossed(self):
        source = raw()
        bad = groups(source, [list(range(6))])
        bad[0]["conversation"] = 1
        with self.assertRaisesRegex(ValueError, "Cross-conversation"):
            validate_partition(source, bad, conversation_id=0)
        source[0]["conversation_id"] = 1
        with self.assertRaisesRegex(ValueError, "Cross-conversation"):
            validate_partition(source, groups(source, [list(range(6))]), conversation_id=0)

    def test_group_cannot_break_atomic_segment(self):
        source = raw()
        with self.assertRaisesRegex(ValueError, "atomic"):
            compare(source, [[0, 1, 2], [3, 4, 5]], [[0, 1], [2, 3, 4, 5]],
                    [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]])

    def test_inputs_are_not_mutated_and_report_not_accuracy(self):
        source = raw()
        partition = groups(source, [[0, 1, 2], [3, 4, 5]])
        original = copy.deepcopy((source, partition))
        result = compare_partitions(source, partition, partition, partition, partition, conversation_id=0)
        self.assertEqual((source, partition), original)
        self.assertEqual(result["interpretation"], INTERPRETATION)
        self.assertFalse(result["reference_is_human_gold"])
        self.assertEqual(result["new_model_calls"], 0)


class AgreementTests(unittest.TestCase):
    def test_exact_and_one_turn_tolerant_differ(self):
        source = raw()
        result = compare(source, [[0, 1], [2, 3, 4, 5]], [[0, 1], [2, 3, 4, 5]],
                         [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(result["boundary_exact"]["f1"], 0)
        self.assertEqual(result["boundary_tolerant"]["f1"], 1)
        self.assertEqual(result["boundary_tolerant"]["matched_pairs"][0]["displacement_turns"], 1)

    def test_tolerance_is_one_to_one_not_double_credit(self):
        row = boundary_agreement(raw(), [0, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 1], tolerance_turns=1)
        self.assertEqual(row["matched_count"], 1)
        self.assertEqual(row["precision"], .5)
        self.assertEqual(row["recall"], 1)
        self.assertAlmostEqual(row["f1"], 2/3)

    def test_maximum_cardinality_precedes_nearest_individual(self):
        row = boundary_agreement(raw((5,)), [0, 0, 1, 2, 2], [0, 1, 1, 2, 2], tolerance_turns=1)
        self.assertEqual(row["matched_count"], 2)
        self.assertEqual([(r["candidate_right_source_index"], r["reference_right_source_index"])
                          for r in row["matched_pairs"]], [(2, 1), (3, 3)])

    def test_forced_session_boundaries_get_no_credit_or_cross_matches(self):
        source = raw((3, 3))
        exact = boundary_agreement(source, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
        self.assertEqual(exact["matched_count"], 0)
        self.assertIsNone(exact["f1"])
        shifted = boundary_agreement(source, [0, 0, 1, 2, 2, 2], [0, 0, 0, 1, 2, 2], tolerance_turns=2)
        self.assertEqual(shifted["matched_count"], 0)

    def test_pairwise_and_bcubed_hand_computed(self):
        row = clustering_agreement([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(row["pairwise_positive"]["matched_count"], 1)
        self.assertEqual(row["pairwise_positive"]["precision"], .5)
        self.assertAlmostEqual(row["pairwise_positive"]["recall"], 1/3)
        self.assertAlmostEqual(row["pairwise_positive"]["f1"], .4)
        self.assertAlmostEqual(row["bcubed"]["precision"], .75)
        self.assertAlmostEqual(row["bcubed"]["recall"], 2/3)
        self.assertFalse(row["negative_pair_accuracy_reported"])

    def test_pairwise_counts_match_explicit_pairs_for_all_small_binary_labels(self):
        for candidate in itertools.product((0, 1), repeat=4):
            for reference in itertools.product((0, 1), repeat=4):
                a = {(i, j) for i in range(4) for j in range(i+1, 4) if candidate[i] == candidate[j]}
                b = {(i, j) for i in range(4) for j in range(i+1, 4) if reference[i] == reference[j]}
                row = clustering_agreement(list(candidate), list(reference))["pairwise_positive"]
                self.assertEqual((row["matched_count"], row["candidate_positive_count"], row["reference_positive_count"]),
                                 (len(a & b), len(a), len(b)))

    def test_all_singletons_do_not_fabricate_pair_accuracy(self):
        row = clustering_agreement([0, 1, 2], ["a", "b", "c"])
        self.assertIsNone(row["pairwise_positive"]["f1"])
        self.assertEqual(row["bcubed"]["f1"], 1)

    def test_groups_do_not_erase_segmentation_boundary_score(self):
        source = raw()
        result = compare(source, [[0, 1, 2], [3, 4, 5]], [list(range(6))],
                         [[0, 1, 2], [3, 4, 5]], [list(range(6))])
        self.assertEqual(result["boundary_exact"]["matched_count"], 1)
        self.assertEqual(result["event_clustering"]["candidate_group_count"], 1)

    def test_aggregate_excludes_cross_conversation_pairs_and_rejects_duplicates(self):
        source = raw((3,))
        indices = [list(range(3))]
        rows = [compare(source, indices, indices, indices, indices, conversation_id=c) for c in (0, 1)]
        combined = aggregate_comparisons(rows)
        self.assertEqual(combined["event_clustering"]["pairwise_positive"]["matched_count"], 6)
        self.assertEqual(combined["cross_conversation_pairs_counted"], 0)
        self.assertEqual(combined["turn_count"], 6)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_comparisons([rows[0], rows[0]])


class BudgetAndSyntheticTests(unittest.TestCase):
    def test_budget_preserves_unknown_http_and_repeated_expansion(self):
        trace = {"logical_judge_calls": 3, "criterion_evaluations": 6, "requests": [
            {"right_source_index": i, "criteria": {"shift": "x", "dependency": "y"}, "status": "success"}
            for i in [2, 4, 2]]}
        row = summarize_call_budget(trace, session_lengths=[6])
        self.assertEqual(row["logical_judge_calls"], 3)
        self.assertEqual(row["distinct_queried_gaps"], 2)
        self.assertEqual(row["queried_gap_coverage"], .4)
        self.assertIsNone(row["physical_http_attempts"])
        self.assertIsNone(row["known_input_tokens"])
        self.assertEqual(row["unqueried_gaps"], 3)

    def test_budget_rejects_bad_counts_and_session_edge(self):
        with self.assertRaises(ValueError):
            summarize_call_budget({"requests": [], "logical_judge_calls": 1}, session_lengths=[3])
        with self.assertRaises(ValueError):
            summarize_call_budget({"requests": [{"right_source_index": 3}]}, session_lengths=[3, 3])

    def test_supplied_physical_usage_stays_distinct(self):
        row = summarize_call_budget({"requests": []}, session_lengths=[1],
                                    physical_usage={"physical_http_attempts": 5, "cache_hits": 3,
                                                    "known_input_tokens": 100, "unknown_usage_attempts": 1})
        self.assertEqual(row["physical_http_attempts"], 5)
        self.assertEqual(row["logical_judge_calls"], 0)
        self.assertIsNone(row["queried_gap_coverage"])

    def test_synthetic_midpoint_termination_does_not_mean_recall(self):
        row = synthetic_midpoint_counterexample()
        self.assertEqual(row["model_api_calls"], 0)
        self.assertEqual(row["experiment_kind"], "synthetic_perfect_oracle_not_real_JEV")
        pruned = next(r for r in row["strategies"] if r["strategy"] == "midpoint_prune_same")
        self.assertTrue(pruned["terminated"])
        self.assertEqual(pruned["logical_comparisons"], 2)
        self.assertEqual(pruned["missed_boundary_after_turns"], [7, 8])
        full = next(r for r in row["strategies"] if r["strategy"] == "midpoint_full_recursion")
        self.assertEqual(full["logical_comparisons"], 20)
        self.assertFalse(full["all_gaps_identified_correctly"])


if __name__ == "__main__":
    unittest.main()
