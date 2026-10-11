"""Admission, budget, failure immutability, and finite-cohort tests; no real APIs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_pilot400_remaining71_v2 as target
from tests.test_pilot400_recovery_qa_v1 import FakeBase, facts


class Remaining71Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.q = {"id": "c1_q1", "conversation": 1, "category": 2,
                  "question": "Question?", "reference": "an answer"}
        self.record = {"q": self.q, "binding": {"query_id": "c1_q1", "reference_date": "2023-01-01"}}
        self.selected = facts()
        self.retrieval = target._sealed({"schema": target.POLICY + "-retrieval", "status": "complete",
            "binding": self.record["binding"], "pool": self.selected, "selected": self.selected, "trace": {}})
        self.retrieval_path, self.qa_path = target._paths(self.output, self.record)
        target.publish_json(self.retrieval_path, self.retrieval)
        self.base = FakeBase(self.output)

    def test_exact_real_cohort_no_other_questions(self):
        rows = target.fixed_questions(target.load_data())
        self.assertEqual(len(rows), 71)
        self.assertEqual(sum(q["conversation"] == 1 for q in rows), 21)
        self.assertEqual(sum(q["conversation"] == 7 for q in rows), 50)
        self.assertEqual({q["conversation"] for q in rows}, {1, 7})

    def test_incorrect_population_is_rejected(self):
        with patch.object(target, "load_cohort", return_value=({}, [self.q] * 71)):
            with self.assertRaisesRegex(ValueError, "population"):
                target.fixed_questions([])

    def test_denied_admission_precedes_credentials(self):
        with patch.object(target, "collect", side_effect=ValueError("Bank not admitted")), \
                patch.object(target, "ModelAPI") as constructor:
            with self.assertRaisesRegex(ValueError, "not admitted"):
                target.execute()
            constructor.assert_not_called()

    def test_hash_mismatch_blocks_before_bank_read(self):
        review = self.root / "review.md"
        review.write_text("Reviewed", encoding="utf-8")
        admission = {"schema": "pilot400-recovery-v2-bank-admission", "source_dataset_sha256": target.SOURCE_HASH,
            "cohort_sha256": target.COHORT_SHA256, "review_path": "review.md", "review_sha256": target.file_sha(review),
            "banks": {"1": {"admitted": True, "bank_sha256": "wrong"}, "7": {"admitted": True, "bank_sha256": "wrong"}}}
        target.publish_json(self.output / "bank_admission.json", admission)
        target.publish_json(self.output / "banks/conv_1/bank_overlay.json", {"facts": []})
        with patch.object(target, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "hash differs"):
                target.validate_admission(self.output, {1: [], 7: []})

    def test_old_output_cannot_be_used(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            target._output(target.ORIGINAL_RUN)
        with self.assertRaisesRegex(ValueError, "pinned"):
            target._output(target.PREVIOUS_RECOVERY)

    def admission_fixture(self, relative):
        review = self.root / "review.md"
        review.write_text("Reviewed", encoding="utf-8")
        admission = {"schema": "pilot400-recovery-v2-bank-admission", "source_dataset_sha256": target.SOURCE_HASH,
            "cohort_sha256": target.COHORT_SHA256, "review_path": "review.md", "review_sha256": target.file_sha(review),
            "banks": {"1": {"admitted": True, "bank_path": relative, "bank_sha256": "wrong"},
                      "7": {"admitted": True, "bank_sha256": "wrong"}}}
        target.publish_json(self.output / "bank_admission.json", admission)
        return admission

    def test_admitted_bank_path_cannot_escape_conversation(self):
        for path in ("../old/bank_overlay.json", "banks/conv_7/bank_overlay_admitted.json", str(self.root / "old.json")):
            self.admission_fixture(path)
            with patch.object(target, "ROOT", self.root):
                with self.assertRaisesRegex(ValueError, "bank path"):
                    target.validate_admission(self.output, {1: [], 7: []})

    def test_admitted_bank_path_accepts_new_filtered_overlay(self):
        relative = "banks/conv_1/bank_overlay_admitted.json"
        admission = self.admission_fixture(relative)
        bank = {"schema": "admitted-bank", "conversation": 1,
                "source_sha256": target.fingerprint([]), "facts": self.selected}
        for ci in (1, 7):
            path = self.output / "banks" / ("conv_" + str(ci)) / "bank_overlay_admitted.json"
            value = {**bank, "conversation": ci,
                "facts": [{**f, "embedding": [1.0] + [0.0] * 767} for f in self.selected]}
            target.publish_json(path, value)
            admission["banks"][str(ci)].update(bank_path=path.relative_to(self.output).as_posix(),
                                               bank_sha256=target.file_sha(path))
        target.publish_json(self.output / "bank_admission.json", admission)
        with patch.object(target, "ROOT", self.root):
            _, banks = target.validate_admission(self.output, {1: [], 7: []})
        self.assertEqual(Path(banks[1]["path"]).name, "bank_overlay_admitted.json")
        self.assertEqual(len(banks[7]["facts"]), 20)

    def test_duplicate_twenty_selection_rejected(self):
        record = {**self.record, "facts": [], "vector": [1.0]}
        duplicate = {"pool": [self.selected[0]] * 20, "selected": [self.selected[0]] * 20, "trace": {}}
        with patch.object(target, "retrieve", return_value=duplicate):
            with self.assertRaisesRegex(ValueError, "unique"):
                target.retrieval_for(record)

    def test_more_than_sixty_candidates_rejected(self):
        record = {**self.record, "facts": [], "vector": [1.0]}
        excessive = {"pool": self.selected * 4, "selected": self.selected, "trace": {}}
        with patch.object(target, "retrieve", return_value=excessive):
            with self.assertRaisesRegex(ValueError, "at most 60"):
                target.retrieval_for(record)

    def test_failure_is_terminal_not_retried(self):
        self.base.shared["failure"] = True
        row = target.one(self.record, self.output, self.base)
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["result"])
        self.assertEqual(len(self.base.shared["calls"]), 1)
        self.base.shared["failure"] = False
        before = target.file_sha(self.qa_path)
        repeated = target.one(self.record, self.output, self.base)
        self.assertEqual(repeated, row)
        self.assertEqual(target.file_sha(self.qa_path), before)
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_success_reuse_no_extra_calls(self):
        row = target.one(self.record, self.output, self.base)
        self.assertEqual(row["status"], "complete")
        self.assertEqual(len(row["selected_ids"]), 20)
        self.assertEqual(len(self.base.shared["calls"]), 2)
        self.assertEqual(target.one(self.record, self.output, self.base), row)
        self.assertEqual(len(self.base.shared["calls"]), 2)

    def test_terminal_binding_change_hard_fails(self):
        target.one(self.record, self.output, self.base)
        changed = copy.deepcopy(self.record)
        changed["binding"]["reference_date"] = "2023-02-01"
        with self.assertRaisesRegex(ValueError, "binding differs"):
            target.one(changed, self.output, self.base)

    def test_lock_excludes_other_owner(self):
        with target.run_lock(self.output):
            with self.assertRaises(FileExistsError):
                with target.run_lock(self.output):
                    self.fail("Lock should be exclusive")
        self.assertFalse((self.output / "qa_remaining71.lock").exists())

    def test_new_qa_always_uses_recorded_retrieval(self):
        with patch.object(target, "retrieve", side_effect=AssertionError("Do not rerank in QA")):
            result = target.one(self.record, self.output, self.base)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["binding"]["retrieval_sha256"], target.file_sha(self.retrieval_path))


if __name__ == "__main__":
    unittest.main()
