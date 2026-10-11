from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_mem.writer_eof_compat_v1 import derive_response
from jev_mem.providers import ModelAPI, ExtractionLLM
from tools import recover_mem0_regroup_eof_v1 as recovery


def response(text, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}


class EOFCompatibilityTests(unittest.TestCase):
    def test_exact_single_brace_and_actual_original_validator(self):
        original = response('{"memory":[{"text":"Quoted \\\"brace}\\\" is text", "attributed_to":"user"}]')
        before = copy.deepcopy(original)
        derived, proof = derive_response(original, http_status=200)
        self.assertEqual(original, before)
        old_text = original["candidates"][0]["content"]["parts"][0]["text"]
        self.assertEqual(derived["candidates"][0]["content"]["parts"][0]["text"], old_text + "}")
        self.assertEqual(proof["appended_utf8_hex"], "7d")
        class Probe:
            generate = ModelAPI.generate
            def request(self, *args, **kwargs):
                return derived
        parsed = json.loads(ExtractionLLM(Probe()).generate_response([{"role": "user", "content": "Original parser dry run"}]))
        self.assertEqual(parsed["memory"][0]["text"], 'Quoted "brace}" is text')

    def test_valid_original_returns_same_object(self):
        raw = response('{"memory":[]}')
        derived, proof = derive_response(raw, http_status=200)
        self.assertIs(derived, raw)
        self.assertEqual(proof["action"], "unchanged")

    def test_non_eof_inner_partial_duplicate_extra_root_keys_rejected(self):
        bad = ['{"memory":[{"text":"x"}', '{"memory":[{"text":"x}',
               '{"memory":[{"text":"x","text":"y"}]', '{"x":1,"memory":[]',
               '{"memory":[],"x":1', '{"memory":[{"text":12}]',
               '{"memory": [}', '{"memory":[],', '{"memory":[null]',
               '{"memory":[{"text":""}]']
        for text in bad:
            with self.subTest(text=text), self.assertRaises((ValueError, json.JSONDecodeError)):
                derive_response(response(text), http_status=200)

    def test_http_finish_and_multipart_constraints(self):
        raw = response('{"memory":[]')
        with self.assertRaises(ValueError):
            derive_response(raw, http_status=201)
        with self.assertRaises(ValueError):
            derive_response(response('{"memory":[]', "MAX_TOKENS"), http_status=200)
        two = copy.deepcopy(raw); two["candidates"] *= 2
        with self.assertRaises(ValueError):
            derive_response(two, http_status=200)
        thought = copy.deepcopy(raw); thought["candidates"][0]["content"]["parts"][0]["thought"] = False
        with self.assertRaises(ValueError):
            derive_response(thought, http_status=200)

    def test_real_failed_response_records_byte_preserved(self):
        proof = recovery.read(recovery.CASE_PROOF)
        raw = recovery.frozen.check_pointer(proof["original_raw"])
        derived, audit = derive_response(raw["response"], http_status=200)
        data = json.loads(derived["candidates"][0]["content"]["parts"][0]["text"])
        self.assertEqual(data["memory"], proof["memory_records_unmodified"])
        self.assertEqual(audit["derived_text_sha256"], proof["derivation"]["derived_response_text_sha256"])
        self.assertEqual(recovery.sha(proof["original_raw"]["path"]), proof["original_raw"]["sha256"])


class RecoveryGuardTests(unittest.TestCase):
    def binding(self, directory):
        source = Path(directory) / "source"
        source.mkdir()
        signature = {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent", "body": {"original": True}}
        key = recovery.request_digest(signature)
        raw = source / (key + ".json")
        raw.write_text(json.dumps({"request": signature, "response": response('{"memory":[]'), "stage": "mem0_extract"}), encoding="utf8")
        ledger = source / "api_calls.jsonl"
        ledger.write_text(json.dumps({"request_hash": key, "stage": "mem0_extract", "status": 200,
            "provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent", "usage": {},
            "context": {"conversation": 8, "method": "dated_merged", "group_id": "g"}, "cache_hit": False}) + "\n", encoding="utf8")
        pin = {"group_id": "g", "sequence": 13, "signature": signature, "request_hash": key,
            "raw": recovery.frozen.pointer(raw), "physical_ledger": recovery.frozen.pointer(ledger)}
        return {"cache_root": str(source), "source_inventory": {str(raw.resolve()): recovery.sha(raw)}, "pins": [pin]}, signature

    def test_exact_bad_source_is_reused_without_dispatch_and_original_bytes_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            binding, signature = self.binding(directory)
            api = recovery.PinnedEOFAPI(Path(directory) / "target", binding=binding,
                keys={"GEMINI_API_KEY": "dummy", "TYPESAFE_API_KEY": "dummy2"},
                context={"group_id": "g", "conversation": 8, "method": "dated_merged"})
            with patch("requests.Session.post", side_effect=AssertionError("Network forbidden")) as dispatch:
                result = api.request(**signature, stage="mem0_extract")
            self.assertEqual(dispatch.call_count, 0)
            self.assertEqual(json.loads(result["candidates"][0]["content"]["parts"][0]["text"]), {"memory": []})
            pin = binding["pins"][0]
            self.assertEqual(recovery.sha(api.cache_dir / (pin["request_hash"] + ".json")), pin["raw"]["sha256"])
            self.assertEqual(recovery.sha(pin["raw"]["path"]), pin["raw"]["sha256"])
            self.assertEqual(recovery.verify(api.run_dir / "eof_parse_lineage/g.json")["parse_proof"]["action"], "append_single_root_brace")

    def test_pin_drift_and_offline_miss_stop_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            binding, signature = self.binding(directory)
            api = recovery.PinnedEOFAPI(Path(directory) / "target", binding=binding, offline_stop=True,
                keys={"GEMINI_API_KEY": "dummy", "TYPESAFE_API_KEY": "dummy2"}, context={"group_id": "g"})
            with patch.object(recovery.PacedEmbeddingAPI, "request", side_effect=AssertionError("Backend forbidden")) as backend:
                with self.assertRaisesRegex(ValueError, "chain changed"):
                    api.request(**{**signature, "body": {"changed": True}}, stage="mem0_extract")
                with self.assertRaisesRegex(ValueError, "uncached"):
                    api.request("gemini", "gemini-embedding-001", {}, operation="countTokens", stage="embedding")
                self.assertEqual(backend.call_count, 0)

    def test_http_prefix_survives_append_but_detects_original_event_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            binding, signature = self.binding(directory)
            pin = binding["pins"][0]
            ledger = Path(pin["physical_ledger"]["path"])
            raw = recovery.read(pin["raw"]["path"])
            evidence = recovery.successful_http(ledger, pin["request_hash"], "g", raw["response"])
            with ledger.open("a", encoding="utf8") as handle:
                handle.write(json.dumps({"stage": "other", "request_hash": "later", "status": 200}) + "\n")
            recovery.verify_http_evidence(evidence, pin["request_hash"], "g", raw["response"])
            text = ledger.read_text(encoding="utf8").replace('"status": 200', '"status": 201', 1)
            ledger.write_text(text, encoding="utf8")
            with self.assertRaisesRegex(ValueError, "prefix changed"):
                recovery.verify_http_evidence(evidence, pin["request_hash"], "g", raw["response"])

    def test_namespace_rejected_before_source_checks(self):
        with patch.object(recovery, "source_binding", side_effect=AssertionError("No source scan")):
            with self.assertRaises(FileExistsError):
                recovery.prepare(recovery.SOURCE / "nested_recovery", Path("unused"))

    def test_failed_and_unknown_banks_cannot_reenter(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); folder = run / "conv_8/dated_merged"; folder.mkdir(parents=True)
            (folder / "eof_rebuild_intent.json").write_text("{}")
            with patch.object(recovery, "manifest", return_value={}), patch.object(recovery.frozen, "write_arm") as writer:
                with self.assertRaisesRegex(RuntimeError, "no reentry"):
                    recovery.run_once(run)
                writer.assert_not_called()
            (folder / "eof_recovery_outcome.json").write_text("{}")
            with patch.object(recovery, "manifest", return_value={}), patch.object(recovery, "validate_complete", side_effect=ValueError("failed")), patch.object(recovery.frozen, "write_arm") as writer:
                with self.assertRaisesRegex(ValueError, "failed"):
                    recovery.run_once(run)
                writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
