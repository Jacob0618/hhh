"""Offline negative controls for the source-only reconstruction audit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import audit_pilot400_members_only_v2 as target
from tests.test_pilot400_members_only_writer_v2 import turns, candidate, assessment


def http(value):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(value)}]}}]}


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.raw = turns()
        self.group = {"group_id": "g1", "turns": self.raw}
        self.sig = target.writer.extraction_request(1, self.group)
        self.request = {"members": self.raw, "read_only_context": [],
            "member_context_ids": {t["source_turn_id"]: [] for t in self.raw},
            "member_source_turn_ids": [t["source_turn_id"] for t in self.raw], "context_source_turn_ids": [],
            "user_id": "pilot400_conv_1", "group_id": "g1", "grouping_method": "jev_topics"}
        self.pending = self.run / "pending.json"
        target.publish_json(self.pending, {"embedding_attempt": {}})

    def make(self, mode="generated_facts"):
        raw = {"facts": [] if mode == "model_empty" else [candidate()]}
        events = [{"request_hash": "extract", "cache_path": "extract.json", "cache_sha256": "extract"}]
        candidates, verdicts, quarantine, accepted = [], {}, [], []
        if mode != "model_empty":
            fact = target.writer.parse_quoted(candidate(), self.request, target.w.provider_request_hash(self.sig), 0)
            candidates = [{"candidate_index": 0, "fact": fact, "proposed": candidate()}]
            review = assessment(supported=mode == "generated_facts")
            verdicts = {"0": review}
            events.append({"request_hash": "review", "cache_path": "review.json", "cache_sha256": "review"})
            if mode == "generated_facts":
                accepted = [fact]
            else:
                quarantine = [{"candidate_index": 0, "stage": "model_support", "proposed": candidate(),
                    "reason": review["reason"], "assessment": review}]
                accepted = target.writer.excerpt_fallback(self.request)
        record = {"schema": target.writer.VERSION, "conversation": 1, "group_id": "g1", "members": self.raw,
            "source_sha256": target.fingerprint(self.raw), "source_only": True, "qa_visible": False,
            "extraction_signature": self.sig, "raw_response": raw, "events": events, "candidates": candidates,
            "assessments": verdicts, "quarantined": quarantine, "failure": None, "mode": mode,
            "facts": [{**f, "embedding": [1.0] + [0.0] * 767} for f in accepted],
            "embedding_path": str(self.pending), "embedding_sha256": target.file_sha(self.pending)}
        def cache(run, sig, stage, **kwargs):
            if stage == "remaining71_members_only_extract":
                return http(raw), events[0]
            return http({"assessments": list(verdicts.values())}), events[1]
        return record, cache

    def audit(self, record, cache):
        with patch.object(target, "checked_cache", side_effect=cache), \
                patch.object(target, "verify_embeddings", return_value={"facts": len(record["facts"])}):
            return target.audit_new_group(1, self.group, self.raw, record, self.run)

    def test_generated_fact_reconstructed(self):
        record, cache = self.make()
        result = self.audit(record, cache)
        self.assertEqual(result["mode"], "generated_facts")
        self.assertFalse(result["semantic_support_verified"])

    def test_genuine_empty_is_not_fallback(self):
        record, cache = self.make("model_empty")
        result = self.audit(record, cache)
        self.assertEqual(result["mode"], "model_empty")
        self.assertEqual(result["facts"], 0)

    def test_rejected_claim_becomes_labeled_excerpt(self):
        record, cache = self.make("raw_member_excerpt_fallback")
        result = self.audit(record, cache)
        self.assertEqual(result["quarantined"], 1)
        self.assertEqual(result["mode"], "raw_member_excerpt_fallback")

    def test_rejected_claim_cannot_be_committed_as_generated(self):
        record, cache = self.make("raw_member_excerpt_fallback")
        record["facts"][0]["text"] = "Unsupported generated claim"
        with self.assertRaisesRegex(ValueError, "Committed facts differ"):
            self.audit(record, cache)

    def test_quarantine_cannot_be_removed(self):
        record, cache = self.make("raw_member_excerpt_fallback")
        record["quarantined"] = []
        with self.assertRaisesRegex(ValueError, "quarantined differs"):
            self.audit(record, cache)

    def test_source_date_cannot_change(self):
        record, cache = self.make()
        record["members"] = copy.deepcopy(record["members"])
        record["members"][0]["observation_date"] = "2030-01-01"
        with self.assertRaisesRegex(ValueError, "source binding"):
            self.audit(record, cache)

    def test_qa_visibility_prohibited(self):
        record, cache = self.make()
        record["qa_visible"] = True
        with self.assertRaisesRegex(ValueError, "source binding"):
            self.audit(record, cache)

    def test_missing_http_cannot_be_pretended_empty(self):
        record, cache = self.make("model_empty")
        with patch.object(target, "checked_cache", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "unknown-attempt review"):
                target.audit_new_group(1, self.group, self.raw, record, self.run)


if __name__ == "__main__":
    unittest.main()
