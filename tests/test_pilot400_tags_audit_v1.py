"""Offline tag audit checks: synthetic facts/cache only, no provider client."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_pilot400_tags_v1 as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")


class Capture(audit.CacheOnlyJudge):
    def request(self, provider, model, body, *, operation, stage):
        self.signature = dict(provider=provider, model=model, operation=operation, body=body)
        return {"answers": {k: {"noul": .4} for k in audit.DIMENSIONS}}


class TagAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.fact = {"id": "f1", "text": "A owns a dog.", "proposed_support": ["D1:1"],
                     "source_records": [{"source_turn_id": "D1:1", "text": "My dog.", "speaker": "A"}]}
        capture = Capture(self.run / "api_cache")
        self.tag = audit.type_one_fact(self.fact, capture)
        signature = capture.signature
        self.digest = audit.sha(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode())
        self.cache = {"request": signature, "response": {"answers": {k: {"noul": .4} for k in audit.DIMENSIONS}}, "stage": audit.STAGE}
        self.cache_path = self.run / "api_cache" / (self.digest + ".json")
        dump(self.cache_path, self.cache)

    def replay(self):
        with patch.object(audit.ModelAPI, "__init__", side_effect=AssertionError("No live client")):
            return audit.type_one_fact(self.fact, audit.CacheOnlyJudge(self.run / "api_cache"))

    def test_exact_frozen_replay_without_api_construction(self):
        self.assertEqual(self.tag, self.replay())
        self.assertEqual(self.tag["context_sha256"], audit.driver.fingerprint(self.fact["source_records"]))

    def test_invalid_http200_payloads_rejected(self):
        for value in (True, float("nan"), -1, 1.1, "0.4"):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.cache)
                changed["response"]["answers"][audit.DIMENSIONS[0]]["noul"] = value
                dump(self.cache_path, changed)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.replay()

    def test_missing_criterion_rejected(self):
        del self.cache["response"]["answers"][audit.DIMENSIONS[0]]
        dump(self.cache_path, self.cache)
        with self.assertRaises(KeyError):
            self.replay()

    def test_cache_binding_tamper_rejected(self):
        self.cache["request"]["body"]["state"] = "Different fact"
        dump(self.cache_path, self.cache)
        with self.assertRaises(ValueError):
            self.replay()

    def event(self, ci=0, cache=False):
        return {"stage": audit.STAGE, "provider": "jev", "model": "jev-1.13.0", "operation": "systemone",
                "request_hash": self.digest, "cache_hit": cache, "status": 200,
                "context": {"conversation": ci, "fact_id": "f1", "method": "shared_session", "phase": "pilot_tag"}}

    def test_ledger_separates_cache_and_physical_and_rejects_wrong_scope(self):
        rows = [self.event(), self.event(cache=True), self.event(ci="0")]
        events, issues, by_fact = audit.ledger_audit("\n".join(map(json.dumps, rows)).encode(), {(0, "f1"): self.digest})
        self.assertEqual(len(events), 3)
        self.assertEqual(len(issues), 1)
        self.assertEqual(len(by_fact[(0, "f1")]), 2)
        self.assertEqual(sum(e["cache_hit"] for e in events), 1)

    def fixture(self):
        dump(self.run / "run_manifest.json", {"protocol_sha256": "p"})
        bank = {"facts": [self.fact], "sha256": "bank"}
        ledger = []
        for ci in range(10):
            folder = self.run / "conversations" / ("conv_" + str(ci))
            dump(folder / "banks/session/raw_source.json", {"raw_turns": []})
            dump(folder / "tag_summary.json", {"fact_count": 1, "bank_sha256": "bank", "binding": {"conversation": ci}})
            bound = {"conversation": ci, "fact_sha256": audit.driver.fingerprint(self.fact), "fact_id": "f1"}
            artifact = audit.driver._seal({"schema": audit.driver.ARTIFACT_SCHEMA,
                "request_fingerprint": audit.driver.fingerprint(bound), "binding": bound,
                "status": "complete", "error": None, "tag": self.tag})
            dump(folder / "tags" / (audit.driver.fingerprint("f1") + ".json"), artifact)
            ledger.append(self.event(ci=ci))
        (self.run / "api_calls.jsonl").write_text("\n".join(map(json.dumps, ledger)), encoding="utf-8")
        stack = ExitStack()
        stack.enter_context(patch.object(audit.driver, "ensure_manifest", return_value={"protocol_sha256": "p"}))
        stack.enter_context(patch.object(audit.driver, "load_bank", return_value=bank))
        stack.enter_context(patch.object(audit.driver, "load_tags", return_value=({"f1": self.tag}, "summary")))
        stack.enter_context(patch.object(audit.driver, "tag_binding", side_effect=lambda manifest, ci, raw, bank: {"conversation": ci}))
        stack.enter_context(patch.object(audit.ModelAPI, "__init__", side_effect=AssertionError("No credentials")))
        stack.enter_context(patch.object(audit.driver, "load_data", side_effect=AssertionError("No QA/gold")))
        self.addCleanup(stack.close)

    def test_ten_conversation_complete_and_constant_distribution(self):
        self.fixture()
        result = audit.audit_tags(self.run)
        self.assertTrue(result["all_ten_session_banks_and_tags_valid"])
        self.assertEqual(result["ledger"]["physical_events"], 10)
        self.assertTrue(result["distribution"]["episodic"]["constant"])
        self.assertFalse(result["distribution"]["episodic"]["all_zero"])

    def test_valid_scores_but_wrong_context_hash_rejected(self):
        self.fixture()
        self.tag["context_sha256"] = "bad"
        result = audit.audit_tags(self.run)
        self.assertFalse(result["all_ten_session_banks_and_tags_valid"])

    def test_missing_artifact_or_missing_event_never_passes(self):
        self.fixture()
        path = self.run / "conversations/conv_0/tags" / (audit.driver.fingerprint("f1") + ".json")
        path.unlink()
        (self.run / "api_calls.jsonl").write_text("", encoding="utf-8")
        result = audit.audit_tags(self.run)
        self.assertFalse(result["all_ten_session_banks_and_tags_valid"])
        self.assertEqual(result["conversations"][0]["artifact_counts"], {"missing": 1})
        self.assertEqual(result["conversations"][1]["valid_tags"], 0)


if __name__ == "__main__":
    unittest.main()
