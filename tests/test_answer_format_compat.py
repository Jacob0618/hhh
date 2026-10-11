from copy import deepcopy
import unittest

from jev_mem.answer_format_compat import normalize_reader_prediction, PROTOCOL
from jev_mem.evaluation import answer_and_judge


class JudgeBoundary(Exception):
    pass


class ReaderOnly:
    """A deliberate boundary: no model or fake Judge score is produced."""
    def __init__(self, prediction):
        self.prediction = prediction

    def generate(self, *args, **kwargs):
        if kwargs["stage"] == "strict_judge":
            raise JudgeBoundary()
        return normalize_reader_prediction(self.prediction)[0]


class AnswerFormatCompatibilityTests(unittest.TestCase):
    def test_strings_are_identical_without_mutating_the_source(self):
        for answer in ("", " Alice\nBob ", "α, β"):
            original = {"answer": answer, "used_fact_ids": ["f1", "f1"], "extra": {"x": 1}}
            before = deepcopy(original)
            result, audit = normalize_reader_prediction(original)
            self.assertEqual(result, before)
            self.assertEqual(original, before)
            self.assertIsNot(result, original)
            self.assertFalse(audit["transformed"])
            self.assertEqual(audit["protocol"], PROTOCOL)

    def test_lists_use_exact_newline_join_and_retain_original_array(self):
        answer = ["Alice", "  Bob\nCarol  ", "Alice", ""]
        original = {"answer": answer, "used_fact_ids": ["f2"]}
        result, audit = normalize_reader_prediction(original)
        self.assertEqual(result["answer"], "Alice\n  Bob\nCarol  \nAlice\n")
        self.assertEqual(audit["original_answer"], answer)
        self.assertEqual(original["answer"], answer)
        self.assertEqual(result["used_fact_ids"], ["f2"])
        self.assertFalse(audit["answer_quality_evaluated"])

    def test_empty_mixed_nested_numeric_boolean_and_object_values_are_rejected(self):
        for answer in ([], None, 17, True, {"name": "Alice"}, [17], ["Alice", None], [["Alice"]]):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                normalize_reader_prediction({"answer": answer})
        for root in ([], "Alice", None):
            with self.assertRaises(ValueError):
                normalize_reader_prediction(root)

    def test_original_citation_validator_still_rejects_invalid_ids(self):
        facts = [{"id": "real-id", "text": "Alice", "observation_date": "2023-01-01"}]
        for ids in (["f99"], [17], "f1"):
            with self.assertRaisesRegex(ValueError, "Reader cited IDs outside"):
                answer_and_judge(ReaderOnly({"answer": ["Alice"], "used_fact_ids": ids}),
                                 "Who?", "unused", facts, reference_date="2023-01-01")
        with self.assertRaises(JudgeBoundary):
            answer_and_judge(ReaderOnly({"answer": ["Alice"], "used_fact_ids": ["f1", "f1"]}),
                             "Who?", "unused", facts, reference_date="2023-01-01")


if __name__ == "__main__":
    unittest.main()
