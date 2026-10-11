"""Meaningful anti-fabrication/provenance guards, no model or native-store access."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_source_time_sidecar_v1 as audit


class SourceTimeAuditTests(unittest.TestCase):
    def source(self):
        return {"D1:1": {"text_sha256": audit.text_sha("I plan camping next month."),
                         "source_observation_date": "2023-05-25",
                         "relative_time_expressions": audit.expressions("I plan camping next month.", "2023-05-25")}}

    def fact(self):
        return {"id": "f1", "text": "I planned camping next month.", "observation_date": "2023-05-25",
                "input_scope": {"source_turn_ids": ["D1:1"]}, "verified_support": []}

    def test_scope_never_becomes_assertion_or_date(self):
        original = self.fact(); saved = copy.deepcopy(original)
        row = audit.fact_record(original, "bank-sha", 0, self.source())
        self.assertEqual(original, saved)
        self.assertEqual(row["assertion_spans"], [])
        self.assertEqual(row["context_spans"], [])
        self.assertEqual(row["support_status"], "unknown")
        self.assertIsNone(row["source_available_at"])
        self.assertIsNone(row["event_date"])
        self.assertIsNone(row["fact_relative_time_expressions"][0]["calendar_candidate"])
        self.assertFalse(row["reader_visible"])

    def test_offset_and_candidate_not_resolved(self):
        text = "Next month, then recently."
        rows = audit.expressions(text, "2023-12-25")
        self.assertEqual(text[rows[0]["start"]:rows[0]["end"]], rows[0]["expression"])
        self.assertEqual(rows[0]["calendar_candidate"]["value"], "2024-01")
        self.assertEqual(rows[0]["calendar_candidate"]["granularity"], "month")
        self.assertEqual(rows[0]["resolution_status"], "candidate_only")
        self.assertEqual(rows[0]["event_attachment_status"], "unknown")
        self.assertIsNone(rows[1]["calendar_candidate"])

    def test_ambiguous_weekdays_and_weeks_not_fabricated(self):
        rows = audit.expressions("last Friday, next Saturday, last week, two days ago", "2023-10-22")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["resolution_status"] == "unresolved" for r in rows))
        self.assertTrue(all(r["calendar_candidate"] is None for r in rows))

    def test_unknown_and_duplicate_source_references_fail(self):
        for ids in (["D404:1"], ["D1:1", "D1:1"]):
            fact = self.fact(); fact["input_scope"]["source_turn_ids"] = ids
            with self.assertRaises(ValueError):
                audit.fact_record(fact, "b", 0, self.source())

    def test_raw_api_rejects_gold_and_preserves_source_date(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.raw_turns({"qa": []})
        raw = {"session_1_date_time": "10:00 am on 25 May, 2023", "session_1": [
            {"dia_id": "D1:1", "speaker": "A", "text": "next month"}]}
        rows = audit.raw_turns(raw)
        self.assertEqual(rows[0]["source_observation_date"], "2023-05-25")
        raw["session_1"][0]["gold_evidence"] = []
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.raw_turns(raw)

    def test_immutable_output_refused_before_input_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / "research/planned_execution_2026-10-03"
            out.mkdir(parents=True); (out / "source_time_sidecar_manifest.json").write_text("old")
            with patch.object(audit, "build", side_effect=AssertionError("must not read")):
                with self.assertRaises(FileExistsError):
                    audit.write_outputs(root, out)
            with self.assertRaises(ValueError):
                audit.write_outputs(root, root / "runs/old")


if __name__ == "__main__":
    unittest.main()
