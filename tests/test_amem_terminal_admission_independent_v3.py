"""Read-only certificate counterexamples against actual retained failed raws."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import audit_amem_terminal_admission_v3 as audit

ROOT = Path(__file__).resolve().parents[1]
QUAL = ROOT / "research/baseline_execution_2026-10-08/amem_format_v3/failure_qualification_v4.json"


class IndependentCertificateTests(unittest.TestCase):
    def fixture(self, ci):
        qualification = json.loads(QUAL.read_text(encoding="utf-8"))
        cert = qualification["certificates"][str(ci)]
        entry = {"request_hash": cert["request_hash"], "raw_path": cert["original_raw_path"],
                 "raw_sha256": cert["original_raw_sha256"], "context": cert["context"], "stage": cert["stage"],
                 "native_call_path": cert["native_call_path"], "native_call_sha256": cert["native_call_sha256"]}
        call = json.loads(Path(cert["native_call_path"]).read_text(encoding="utf-8"))
        return qualification, entry, call

    def test_five_actual_model_failure_certificates_have_correct_allowances(self):
        for ci in (1, 2, 5, 7, 9):
            with self.subTest(conversation=ci):
                qualification, entry, call = self.fixture(ci)
                cert = audit.certify_failure(ci, entry, call, qualification, QUAL)
                self.assertEqual(cert["retry_limit"], int(ci in (5, 7)))
                self.assertEqual(entry["mode"], "replace_once" if ci in (5, 7) else "inherit")
                self.assertEqual(cert["review_sha256"], audit.sha(QUAL))

    def test_raw_request_context_call_or_embedded_review_change_rejected(self):
        for name in ("original_raw_sha256", "request_hash", "context", "native_call_sha256", "review_sha256"):
            qualification, entry, call = self.fixture(5)
            cert = qualification["certificates"]["5"]
            cert[name] = {} if name == "context" else "0" * 64
            with self.subTest(field=name), self.assertRaises(ValueError):
                audit.certify_failure(5, entry, call, qualification, QUAL)

    def test_valid_native_action_cannot_be_reclassified_to_regenerate(self):
        qualification, entry, call = self.fixture(2)
        cert = qualification["certificates"]["2"]
        cert["kind"], cert["retry_limit"] = "known_complete_invalid_json", 1
        with self.assertRaises(ValueError):
            audit.certify_failure(2, entry, call, qualification, QUAL)

    def test_global_terminal_guard_precedes_manifest_or_other_source_reads(self):
        with patch.object(audit, "process_evidence", side_effect=ValueError("live controller")), patch.object(audit, "sha", side_effect=AssertionError("must not read source")):
            with self.assertRaisesRegex(ValueError, "live controller"):
                audit.audit(QUAL)


if __name__ == "__main__":
    unittest.main()
