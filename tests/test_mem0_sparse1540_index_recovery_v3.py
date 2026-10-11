"""Versioned pre-mutation continuation guards; no real store/provider opens."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import recover_mem0_sparse1540_index_v3 as target

PRIOR_SHA = "64b61be7376525be60846c27f2d2229795217d7ce6032d43dc507a0d9ef5c985"


class ContinuationGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "recovery"; self.out.mkdir()
        self.arm = self.out / "recovered/session_rank/conv_3"
        for p in (self.arm / "native/qdrant", self.arm / "original_snapshot/qdrant"):
            p.mkdir(parents=True); (p / "db").write_bytes(b"identical")
        self.old = {"helper_sha256": PRIOR_SHA, "source_manifest": self.bind(target.base.OUT / "manifest.json"),
                    "coverage": self.bind(target.COVERAGE), "policy_binding": self.bind(target.POLICY_PATH),
                    "targets": target.TARGETS}
        self.save_old()

    def bind(self, p):
        return {"path": str(Path(p).resolve()), "sha256": "fixture-binding"}

    def save_old(self):
        (self.out / "prepare_intent.json").write_text(json.dumps(self.old), encoding="utf-8")

    def run_rejected(self, message):
        continuation = self.out / "prepare_continuation_intent.json"
        previous = continuation.read_bytes() if continuation.exists() else None
        original_hash = target.ROOT / "tools/recover_mem0_sparse1540_index_v2.py"
        def test_hash(p):
            if Path(p).resolve() == original_hash.resolve():
                return PRIOR_SHA
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        with patch.object(target, "OUT", self.out), \
                patch.object(target, "validate_coverage", return_value=({}, {})), \
                patch.object(target, "binding", side_effect=self.bind), \
                patch.object(target.base, "sha", side_effect=test_hash), \
                patch("qdrant_client.QdrantClient", side_effect=AssertionError("No real store may open")) as client, \
                self.assertRaisesRegex(ValueError, message):
            target.prepare()
        self.assertEqual(client.call_count, 0)
        self.assertEqual(continuation.read_bytes() if continuation.exists() else None, previous)

    def test_unknown_prior_helper_rejected(self):
        self.old["helper_sha256"] = "other"; self.save_old()
        self.run_rejected("Unknown prior")

    def test_prior_policy_or_target_change_rejected(self):
        self.old["targets"] = {"session_rank": "different"}; self.save_old()
        self.run_rejected("context differs")

    def test_committed_repair_intent_cannot_resume(self):
        (self.arm / "repair_intent.json").write_text("{}")
        self.run_rejected("Unexpected prior target")

    def test_prior_completed_manifest_cannot_resume(self):
        (self.out / "recovery_manifest.json").write_text("{}")
        self.run_rejected("Unreviewed continuation")

    def test_second_arm_partially_present_cannot_resume(self):
        (self.out / "recovered/regroup_rank").mkdir()
        self.run_rejected("Prior index mutation")

    def test_existing_native_bytes_different_from_snapshot_rejected(self):
        (self.arm / "native/qdrant/db").write_bytes(b"modified")
        self.run_rejected("Existing copy differs")

    def test_previous_continuation_cannot_be_silently_repeated(self):
        (self.out / "prepare_continuation_intent.json").write_text("{}")
        self.run_rejected("Unreviewed continuation")


if __name__ == "__main__":
    unittest.main()
