"""Offline recovery contract tests. Fakes only; no credentials or model traffic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import pilot400_members_only_writer_v2 as target


def turns():
    return [{"source_turn_id": "D1:1", "session_id": "1", "speaker": "Alex",
             "observation_date": "2023-01-01", "text": "I plan to visit Rome.",
             "caption": "A map of Rome."},
            {"source_turn_id": "D2:1", "session_id": "2", "speaker": "Blair",
             "observation_date": "2023-02-01", "text": "I have two cats."}]


def request():
    members = turns()
    return {"members": members, "read_only_context": [],
            "member_context_ids": {t["source_turn_id"]: [] for t in members},
            "member_source_turn_ids": [t["source_turn_id"] for t in members],
            "context_source_turn_ids": [], "user_id": "test", "group_id": "g1",
            "grouping_method": "jev_topics"}


def candidate():
    return {"text": "Alex plans to visit Rome.", "source_turn_ids": ["D1:1"],
            "temporal_status": "uncertain", "source_quotes": [
                {"source_turn_id": "D1:1", "field": "text", "quote": "I plan to visit Rome."}]}


def assessment(index=0, **changes):
    return {"candidate_index": index, "supported": True, "attribution_ok": True,
            "temporal_ok": True, "reason": "Supported in the original source.", **changes}


class FakeAPI:
    def __init__(self, path):
        self.cache_dir = Path(path) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.context = {"conversation": 1}
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Fake transport must not be called in this test")

    def _redact(self, text):
        return str(text)


class FakeWriter:
    _parse = staticmethod(target.w.GroupFactWriter._parse)

    def __init__(self, api, store):
        Path(store).mkdir(parents=True, exist_ok=True)
        self.pending_path = Path(store) / "pending.json"

    def _embed_once(self, facts, pending):
        vectors = [[1.0] + [0.0] * 767 for _ in facts]
        self.pending_path.write_text(json.dumps({"embeddings": vectors}), encoding="utf-8")
        return vectors


class ContractTests(unittest.TestCase):
    def test_valid_quote_keeps_generated_text_and_sources(self):
        fact = target.parse_quoted(candidate(), request(), "request", 0)
        self.assertEqual(fact["text"], candidate()["text"])
        self.assertEqual(fact["member_support_ids"], ["D1:1"])
        self.assertEqual(fact["context_support_ids"], [])
        self.assertFalse(fact["semantic_support_verified"])

    def test_tampered_quote_rejected(self):
        value = candidate()
        value["source_quotes"][0]["quote"] = "I visited Rome."
        with self.assertRaises(ValueError):
            target.parse_quoted(value, request(), "request", 0)

    def test_context_id_cannot_be_smuggled_into_quotes(self):
        value = candidate()
        value["source_turn_ids"] = ["D0:1"]
        value["source_quotes"][0]["source_turn_id"] = "D0:1"
        with self.assertRaises(ValueError):
            target.parse_quoted(value, request(), "request", 0)

    def test_quote_citation_mismatch_rejected(self):
        value = candidate()
        value["source_turn_ids"].append("D2:1")
        with self.assertRaises(ValueError):
            target.parse_quoted(value, request(), "request", 0)

    def test_duplicate_citation_rejected_by_original_parser(self):
        value = candidate()
        value["source_turn_ids"] *= 2
        with self.assertRaises(target.w.GroupWriteError):
            target.parse_quoted(value, request(), "request", 0)

    def test_missing_quote_and_empty_quote_rejected(self):
        for quotes in ([], [{"source_turn_id": "D1:1", "field": "text", "quote": " "}]):
            value = candidate()
            value["source_quotes"] = quotes
            with self.assertRaises(ValueError):
                target.parse_quoted(value, request(), "request", 0)

    def test_quote_match_does_not_claim_semantic_support(self):
        value = candidate()
        value["text"] = "Alex already visited Rome."
        parsed = target.parse_quoted(value, request(), "request", 0)
        self.assertFalse(parsed["semantic_support_verified"])

    def test_each_cited_source_requires_its_own_exact_quote(self):
        value = candidate()
        value["text"] += " Blair has two cats."
        value["source_turn_ids"].append("D2:1")
        value["source_quotes"].append({"source_turn_id": "D2:1", "field": "text", "quote": "two cats"})
        fact = target.parse_quoted(value, request(), "request", 1)
        self.assertEqual(fact["observation_dates"], ["2023-01-01", "2023-02-01"])
        self.assertIsNone(fact["observation_date"])

    def test_validator_omission_and_duplicate_rejected(self):
        for rows in ([assessment()], [assessment(), assessment()]):
            with self.assertRaises(ValueError):
                target.checked_assessments({"assessments": rows}, [0, 1])

    def test_validator_extra_index_rejected(self):
        with self.assertRaises(ValueError):
            target.checked_assessments({"assessments": [assessment(1)]}, [0])

    def test_validator_non_bool_and_bool_index_rejected(self):
        for changes in ({"supported": 1}, {"attribution_ok": "true"}, {"candidate_index": True}):
            with self.assertRaises(ValueError):
                target.checked_assessments({"assessments": [assessment(**changes)]}, [0])

    def test_extraction_payload_contains_only_original_member_scope(self):
        group = {"group_id": "g1", "turns": turns(), "QA": "secret question",
                 "gold": "secret answer", "read_only_context": ["outside"]}
        sig = target.extraction_request(1, group)
        payload = json.loads(sig["body"]["contents"][0]["parts"][0]["text"])
        self.assertEqual(set(payload), {"protocol", "conversation", "group_id", "MEMBERS"})
        self.assertEqual(payload["MEMBERS"], turns())
        self.assertNotIn("secret", json.dumps(sig))
        ids = sig["body"]["generationConfig"]["responseSchema"]["properties"]["facts"]["items"]["properties"]["source_turn_ids"]["items"]["enum"]
        self.assertEqual(ids, ["D1:1", "D2:1"])

    def test_source_whitelist_rejects_embedded_qa_or_gold(self):
        for field in ("question", "gold", "answer", "facts"):
            members = turns()
            members[0][field] = "must not reach API"
            with self.assertRaises(ValueError):
                target.extraction_request(1, {"group_id": "g1", "turns": members})

    def test_raw_fallback_preserves_text_speakers_dates_and_source_ids(self):
        source = request()
        before = copy.deepcopy(source)
        facts = target.excerpt_fallback(source)
        self.assertEqual(source, before)
        text = "\n".join(f["text"] for f in facts)
        for turn in turns():
            self.assertIn(turn["text"], text)
            self.assertIn(turn["speaker"], text)
            self.assertIn(turn["observation_date"], text)
            self.assertIn(turn["source_turn_id"], text)
        self.assertIn("A map of Rome.", text)
        self.assertTrue(all(f["memory_kind"] == "raw_member_excerpt" for f in facts))
        self.assertEqual({sid for f in facts for sid in f["sources"]}, {"D1:1", "D2:1"})
        self.assertTrue(all(not f["semantic_support_verified"] for f in facts))

    def run_group(self, responses):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        base = Path(folder.name)
        api = FakeAPI(base)
        group = {"group_id": "g1", "turns": turns()}
        with patch.object(target, "call_once", side_effect=responses), patch.object(target.w, "GroupFactWriter", FakeWriter):
            return target.new_group(1, group, turns(), base / "group", api)

    def test_model_empty_is_not_manually_created_fallback(self):
        result = self.run_group([({"facts": []}, {"request_hash": "fake"})])
        self.assertEqual(result["mode"], "model_empty")
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["raw_response"], {"facts": []})

    def test_mechanical_rejection_keeps_bad_response_and_uses_raw_fallback(self):
        bad = candidate()
        bad["source_quotes"][0]["quote"] = "not present"
        result = self.run_group([({"facts": [bad]}, {"request_hash": "fake"})])
        self.assertEqual(result["mode"], "raw_member_excerpt_fallback")
        self.assertEqual(result["quarantined"][0]["proposed"], bad)
        self.assertEqual(result["quarantined"][0]["stage"], "mechanical")

    def test_support_rejection_preserved_and_never_silently_admitted(self):
        proposed = candidate()
        verdict = assessment(temporal_ok=False, reason="Plan has been rewritten as completed.")
        result = self.run_group([({"facts": [proposed]}, {"request_hash": "fake_extract"}),
                                 ({"assessments": [verdict]}, {"request_hash": "fake_review"})])
        self.assertEqual(result["mode"], "raw_member_excerpt_fallback")
        self.assertEqual(result["quarantined"][0]["proposed"], proposed)
        self.assertEqual(result["quarantined"][0]["assessment"], verdict)

    def test_missing_validator_item_cannot_admit_candidate(self):
        result = self.run_group([({"facts": [candidate()]}, {"request_hash": "fake_extract"}),
                                 ({"assessments": []}, {"request_hash": "fake_review"})])
        self.assertEqual(result["mode"], "raw_member_excerpt_fallback")
        self.assertEqual(result["failure"]["type"], "ValueError")
        self.assertIn("omits", result["failure"]["message"])

    def test_accepted_candidate_remains_exact_generated_fact(self):
        proposed = candidate()
        result = self.run_group([({"facts": [proposed]}, {"request_hash": "fake_extract"}),
                                 ({"assessments": [assessment()]}, {"request_hash": "fake_review"})])
        self.assertEqual(result["mode"], "generated_facts")
        self.assertEqual(result["facts"][0]["text"], proposed["text"])
        self.assertEqual(result["facts"][0]["sources"], proposed["source_turn_ids"])
        self.assertEqual(result["quarantined"], [])

    def test_integrity_error_never_becomes_raw_fallback(self):
        with self.assertRaises(target.IntegrityError):
            self.run_group([target.IntegrityError("Cache signature was changed")])

    def test_unknown_intent_never_resends(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            api = FakeAPI(base)
            sig = target.extraction_request(1, {"group_id": "g1", "turns": turns()})
            digest = target.w.provider_request_hash(sig)
            intent = base / "writer_request_intents" / (digest + ".json")
            intent.parent.mkdir(parents=True)
            intent.write_text(json.dumps({"request": sig}), encoding="utf-8")
            with patch.object(target, "OUT", base):
                with self.assertRaises(target.w.ExtractionUnknown):
                    target.call_once(api, sig, "offline_test")
            self.assertEqual(api.calls, 0)

    def test_changed_intent_rejected_without_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            api = FakeAPI(base)
            sig = target.extraction_request(1, {"group_id": "g1", "turns": turns()})
            digest = target.w.provider_request_hash(sig)
            intent = base / "writer_request_intents" / (digest + ".json")
            intent.parent.mkdir(parents=True)
            intent.write_text(json.dumps({"request": {"changed": True}}), encoding="utf-8")
            with patch.object(target, "OUT", base):
                with self.assertRaises(ValueError):
                    target.call_once(api, sig, "offline_test")
            self.assertEqual(api.calls, 0)


if __name__ == "__main__":
    unittest.main()
