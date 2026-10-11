"""No-network tests for sparse candidate coverage and explicit request bounds."""
import unittest

from jev_mem.topic_boundary_budget import propose_sparse_boundaries
from tests.test_topic_boundaries import turns


def vector(index):
    value = [0.0] * 768
    value[index] = 1.0
    return value


class BudgetTests(unittest.TestCase):
    def test_budget_and_unassessed_are_not_negative_decisions(self):
        source = turns()
        plan = propose_sparse_boundaries(source, [vector(0)] * 3 + [vector(1)] * 3)
        session = plan["sessions"][0]
        selected = {r["right_source_index"] for r in session["initial_requests"]}
        self.assertTrue({2, 4}.issubset(selected))
        self.assertIn(3, selected)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected | set(session["unassessed_gap_indices"]), set(range(1, 6)))
        self.assertEqual(plan["max_logical_requests_including_refinement"], 6)
        self.assertFalse(plan["cuts_or_groups_emitted"])

    def test_session_locality_and_short_session_budget(self):
        source = turns()
        for row in source[2:]:
            row["session_id"] = "s2"
        plan = propose_sparse_boundaries(source, [vector(0)] * 6)
        self.assertEqual([s["initial_logical_requests"] for s in plan["sessions"]], [1, 3])
        for session in plan["sessions"]:
            for request in session["initial_requests"]:
                state = request["state"]
                self.assertTrue(all(row["session_id"] == session["session_id"] for row in state["left_window"] + state["right_window"]))

    def test_zero_budget_does_not_spend_refinement(self):
        plan = propose_sparse_boundaries(turns(), [vector(0)] * 6, initial_budget=0)
        self.assertEqual(plan["max_logical_requests_including_refinement"], 0)
        self.assertEqual(len(plan["sessions"][0]["unassessed_gap_indices"]), 5)

    def test_invalid_vectors_or_source_count_fail(self):
        with self.assertRaises(ValueError):
            propose_sparse_boundaries(turns(), [vector(0)])
        with self.assertRaises(ValueError):
            propose_sparse_boundaries(turns(), [[0.0] * 768 for _ in range(6)])


if __name__ == "__main__":
    unittest.main()
