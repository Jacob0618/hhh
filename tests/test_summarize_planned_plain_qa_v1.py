"""Offline report direction/denominator and original-audit isolation checks."""
import unittest
from unittest.mock import patch

from tools import summarize_planned_plain_qa_v1 as report
from tools import summarize_sdk20_plain_qa_v1 as old


def rows(arm, value):
    return [{"query_id": f"c{i % 10}_q{i}", "conversation": i % 10, "category": 2,
             "method": arm, "selection_status": "complete", "reader_status": "complete",
             "judge_status": "complete", "qa_status": "complete", "errors": [],
             "scores": {key: value for key in report.METRICS}} for i in range(1540)]


class PlannedSummaryTests(unittest.TestCase):
    def test_original_stage_audit_bodies_and_old_globals_preserved(self):
        for name in ("ledger", "audit_stage", "audit_phase", "record_rows"):
            source, copied = getattr(old, name), getattr(report, name)
            self.assertIs(source.__code__, copied.__code__)
            self.assertIs(source.__globals__, old.__dict__)
            self.assertIs(copied.__globals__, report.__dict__)
            self.assertIs(copied.__globals__["runner"], report.runner)
        self.assertEqual(old.ARMS, ("baseline_sdk20", "fact_types_sdk20"))
        self.assertEqual(len(report.ARMS), 3)

    def test_pair_names_and_signs_are_h1_h0_and_h2_h1(self):
        values = rows(report.ARMS[0], .2) + rows(report.ARMS[1], .5) + rows(report.ARMS[2], .4)
        result = report.aggregate(values)
        self.assertNotIn("paired_merged_minus_unmerged", result)
        self.assertEqual(set(result["arms"]), set(report.ARMS))
        pairs = result["paired_comparisons"]
        self.assertAlmostEqual(pairs["H1_minus_H0"]["metrics"]["official_locomo_f1"]["metrics"]["official_locomo_f1"]["other_minus_baseline"], .3)
        self.assertAlmostEqual(pairs["H2_minus_H1"]["metrics"]["official_locomo_f1"]["metrics"]["official_locomo_f1"]["other_minus_baseline"], -.1)
        self.assertEqual(pairs["H1_minus_H0"]["metrics"]["judge_correct"]["common_success_count"], 1540)

    def test_missing_judge_retains_planned1540_independent_reader(self):
        values = rows(report.ARMS[0], .2) + rows(report.ARMS[1], .5) + rows(report.ARMS[2], .4)
        values[0]["scores"]["judge_correct"] = None
        values[0]["judge_status"] = values[0]["qa_status"] = "failed"
        output = report.aggregate(values)["arms"][report.ARMS[0]]["metrics"]
        self.assertEqual(output["judge_correct"]["valid_denominator"], 1539)
        self.assertEqual(output["judge_correct"]["planned_denominator"], 1540)
        self.assertIsNone(output["judge_correct"]["full_cohort_observed_mean"])
        self.assertEqual(output["official_locomo_f1"]["valid_denominator"], 1540)

    def test_total_and_historical_anchor_transformed_without_rescoring(self):
        historical = rows("baseline_sdk20", .1)
        current = rows(report.ARMS[0], .2) + rows(report.ARMS[1], .5) + rows(report.ARMS[2], .4)
        anchor = {"summary": {"path": "frozen", "sha256": "x"},
                  "official_f1_percent": 54.6095082528163, "judge_percent": 66.03896103896105}
        baseline = {"slots_by_view": {report.BASELINE_VIEW: historical},
                    "audit_status": "no_invalid_artifacts", "audit_errors": []}
        original_output = {"slots_by_view": {"strict_firstpass": current},
                           "views": {"strict_firstpass": {}}, "summary_code_sha256": {},
                           "limitations": ["old paired_merged_minus_unmerged alias"]}
        with patch.object(report, "verify_lock", return_value={"baseline": anchor}), \
             patch.object(report.runner, "bound", return_value=baseline), \
             patch.object(report, "_original_summarize", return_value=original_output):
            output = report.summarize("no-model-run")
        self.assertEqual(output["planned_total"], 4620)
        self.assertEqual(output["historical_N0"]["official_f1_percent"], anchor["official_f1_percent"])
        self.assertFalse(output["historical_N0"]["recomputed_or_modified"])
        delta = output["views"]["strict_firstpass"]["historical_N0_comparisons"]["H0_minus_N0"]["metrics"]["official_locomo_f1"]["metrics"]["official_locomo_f1"]["other_minus_baseline"]
        self.assertAlmostEqual(delta, .1)


if __name__ == "__main__":
    unittest.main()
