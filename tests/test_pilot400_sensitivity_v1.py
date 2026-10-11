import unittest
from tools.analyze_pilot400_sensitivity_v1 import paired, population, METRICS, ARMS


def bank(shift=0):
    return {f"c{ci}_q{j}": {"query_id": f"c{ci}_q{j}", "conversation": ci,
                           **{metric: .25+shift for metric in METRICS}}
            for ci in range(10) for j in range(2)}


class SensitivityTests(unittest.TestCase):
    def test_constant_difference_original_ten_cluster_ci(self):
        p = paired(bank(), bank(.5), 400)
        self.assertEqual(p["common_success_count"], 20)
        self.assertEqual(p["bootstrap"]["samples"], 2000)
        self.assertTrue(p["bootstrap"]["eligible"])
        for value in p["metrics"].values():
            self.assertEqual(value["other_minus_baseline"], .5)
            self.assertEqual(value["cluster_bootstrap_95_percentile_interval"], [.5, .5])

    def test_missing_cluster_never_substitute_nine_clusters(self):
        b = bank(.5)
        b = {k:v for k,v in b.items() if v["conversation"] != 7}
        p = paired(bank(), b, 400)
        self.assertFalse(p["bootstrap"]["eligible"])
        self.assertEqual(p["bootstrap"]["samples"], 0)
        self.assertEqual(p["common_success_by_conversation"]["7"], 0)
        self.assertIsNone(p["metrics"][METRICS[0]]["cluster_bootstrap_95_percentile_interval"])

    def test_cluster_with_one_pair_fails_original_rule(self):
        b = bank(.5)
        del b["c0_q1"]
        p = paired(bank(), b, 399)
        self.assertEqual(p["common_success_count"], 19)
        self.assertFalse(p["bootstrap"]["eligible"])

    def test_symmetric_exclusion_not_zero_imputation(self):
        maps = {arm: bank() for arm in ARMS}
        original = {arm: len(v) for arm,v in maps.items()}
        p = population(maps, ["c0_q0"], 399)
        self.assertEqual(p["sensitivity_population_count_per_arm"], 399)
        for arm in ARMS:
            self.assertEqual(p["arms"][arm]["scored_successes_retained"], 19)
            self.assertEqual(p["arms"][arm]["conditional_metric_means"][METRICS[0]], .25)
            self.assertEqual(len(maps[arm]), original[arm])
        for pair in p["paired_differences"].values():
            self.assertNotIn("c0_q0", pair["common_query_ids"])
            self.assertEqual(pair["common_success_count"], 19)

    def test_pair_identity_mismatch_rejected(self):
        other = bank(.5)
        other["c0_q0"]["conversation"] = 1
        with self.assertRaisesRegex(ValueError, "Conversation"):
            paired(bank(), other, 400)


if __name__ == "__main__":
    unittest.main()

