"""Offline guard/adapter checks. No credentials or model calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import recover_mem0_regroup_sdk20 as r


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")
        return r.frozen.pointer(path)

    def fixture(self):
        folder = self.root / "new/conv_0/dated_unmerged385"
        folder.mkdir(parents=True)
        parsed = [{"id": "0", "text": "fixed original fact"}]
        journal = self.put(self.root / "old/journal.json", {"group_id": "g0", "llm_calls": [{"parsed_memory": parsed}]})
        signature = {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent", "body": {"contents": []}}
        key = r.request_digest(signature)
        raw = self.put(self.root / "old/api_cache" / (key + ".json"), {"request": signature, "response": {"fixed": True}, "stage": "mem0_extract"})
        binding = {"conversation": 0, "arm": "dated_unmerged385", "journal": journal,
            "success_cache_root": str(self.root / "old/api_cache"),
            "raw_first_extraction": {**raw, "request_hash": key, "parsed_memory": parsed}}
        manifest = {"protocol_sha256": "frozen", "recovery": {"source_bindings": [binding], "gate_path": str(self.root / "gate.sqlite")}}
        return folder, binding, manifest, signature

    def test_pinned_raw_reuse_and_exact_body_guard(self):
        folder, binding, _, sig = self.fixture()
        api = object.__new__(r.PinnedRecoveryAPI)
        api.source_binding = binding
        api.context = {"group_id": "g0"}
        api.cache_dir = folder / "native/api_cache"
        api.cache_dir.mkdir(parents=True)
        source = Path(binding["raw_first_extraction"]["path"])
        def consume(*args, **kwargs):
            (api.cache_dir / source.name).write_bytes(source.read_bytes())
            return {"fixed": True}
        with patch.object(r.PacedEmbeddingAPI, "request", side_effect=consume) as mocked:
            self.assertEqual(api.request("gemini", "gemini-2.5-flash", sig["body"], stage="mem0_extract"), {"fixed": True})
            self.assertEqual(mocked.call_count, 1)
            with self.assertRaisesRegex(ValueError, "request changed"):
                api.request("gemini", "gemini-2.5-flash", {"contents": ["changed"]}, stage="mem0_extract")
            self.assertEqual(mocked.call_count, 1)

    def test_pinned_response_cannot_be_regenerated(self):
        folder, binding, _, sig = self.fixture()
        api = object.__new__(r.PinnedRecoveryAPI)
        api.source_binding = binding
        api.context = {"group_id": "g0"}
        api.cache_dir = folder / "native/api_cache"
        api.cache_dir.mkdir(parents=True)
        def changed(*args, **kwargs):
            (api.cache_dir / (binding['raw_first_extraction']['request_hash'] + '.json')).write_text('{}')
            return {"different": True}
        with patch.object(r.PacedEmbeddingAPI, "request", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "exact raw"):
                api.request("gemini", "gemini-2.5-flash", sig["body"], stage="mem0_extract")

    def test_real_cached_provider_own_raw_has_zero_network(self):
        folder, binding, _, sig = self.fixture()
        with patch("jev_mem.providers.load_credentials", side_effect=AssertionError("No credentials")), patch("requests.Session.post", side_effect=AssertionError("No network")) as post:
            api = r.PinnedRecoveryAPI(folder / "native", source_binding=binding,
                keys={"GEMINI_API_KEY": "offline-key", "TYPESAFE_API_KEY": "offline-jev"},
                context={"group_id": "g0"}, gate_path=self.root / "gate.sqlite")
            value = api.request("gemini", "gemini-2.5-flash", sig["body"], stage="mem0_extract")
            self.assertEqual(value, {"fixed": True})
            self.assertEqual(post.call_count, 0)
            self.assertEqual(api.events[0]["source"], "inherited_exact_request")
            key = binding["raw_first_extraction"]["request_hash"]
            self.assertEqual((api.cache_dir / (key + ".json")).read_bytes(), Path(binding["raw_first_extraction"]["path"]).read_bytes())
            self.assertFalse((folder / "native/request_intents" / (key + ".json")).exists())

    def test_own_cache_root_and_scoped_binding(self):
        _, binding, _, _ = self.fixture()
        with patch.object(r.PacedEmbeddingAPI, "__init__", return_value=None) as base:
            r.PinnedRecoveryAPI(self.root, source_binding=binding, keys={})
            self.assertEqual(base.call_args.kwargs["cache_roots"], (Path(binding["success_cache_root"]),))
            with self.assertRaisesRegex(ValueError, "Cross-arm"):
                r.PinnedRecoveryAPI(self.root, source_binding=binding, cache_roots=[self.root / "other"])
        api = object.__new__(r.PinnedRecoveryAPI)
        api.source_binding = binding
        api.run_dir = self.root
        api.keys, api.retries, api.timeout = {}, 2, 120
        api.context = {"conversation": 0}
        api.gate_path = self.root / "gate.sqlite"
        api.interval_seconds, api.cooldown_seconds = 3.0, 30.0
        with patch.object(r.PacedEmbeddingAPI, "__init__", return_value=None) as base:
            scoped = api.scoped(group_id="g1")
            self.assertIs(scoped.source_binding, binding)
            self.assertEqual(base.call_args.kwargs["context"], {"conversation": 0, "group_id": "g1"})

    def test_failed_bank_never_reenters_writer(self):
        folder, binding, manifest, _ = self.fixture()
        calls = []
        def writer(run, ci, arm, value):
            calls.append((ci, arm))
            self.assertIsNot(r.frozen.ModelAPI, original)
            state = {"status": "failed", "error": {"type": "Mock429"}}
            self.put(folder / "write_status.json", state)
            return state
        original = r.frozen.ModelAPI
        with patch.object(r.frozen, "write_arm", side_effect=writer):
            result = r.run_arm(self.root / "new", 0, "dated_unmerged385", manifest)
            self.assertEqual(result["status"], "failed")
            self.assertIs(r.frozen.ModelAPI, original)
            with self.assertRaisesRegex(RuntimeError, "no implicit second"):
                r.run_arm(self.root / "new", 0, "dated_unmerged385", manifest)
        self.assertEqual(len(calls), 1)

    def test_uncertain_bank_intent_blocks_and_restores_constructor(self):
        folder, _, manifest, _ = self.fixture()
        original = r.frozen.ModelAPI
        with patch.object(r.frozen, "write_arm", side_effect=RuntimeError("mock interruption")) as writer:
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                r.run_arm(self.root / "new", 0, "dated_unmerged385", manifest)
            self.assertIs(r.frozen.ModelAPI, original)
            with self.assertRaisesRegex(RuntimeError, "Prior nontransactional"):
                r.run_arm(self.root / "new", 0, "dated_unmerged385", manifest)
            self.assertEqual(writer.call_count, 1)
        self.assertTrue((folder / "rebuild_intent.json").exists())

    def test_complete_bank_resume_is_read_only(self):
        folder, binding, manifest, _ = self.fixture()
        def writer(run, ci, arm, value):
            self.put(folder / "native/group_audit/user/g0.json", {"sequence": 1, "llm_calls": [{"parsed_memory": binding["raw_first_extraction"]["parsed_memory"]}]})
            bank = self.put(folder / "bank.json", {"complete": True})
            state = {"status": "complete", "bank": bank}
            self.put(folder / "write_status.json", state)
            return state
        with patch.object(r.frozen, "write_arm", side_effect=writer) as call:
            result = r.run_arm(self.root / "new", 0, "dated_unmerged385", manifest)
            self.assertTrue(result["first_extraction_verified"])
            again = r.run_arm(self.root / "new", 0, "dated_unmerged385", manifest)
            self.assertEqual(result, {k: v for k, v in again.items() if k != "content_sha256"})
            self.assertEqual(call.call_count, 1)

    def test_dev_only(self):
        with self.assertRaisesRegex(ValueError, "restricted"):
            r.run_conv(self.root, 2)

    def test_prepare_copies_only_inputs_no_api_or_store(self):
        source = self.root / "old"
        input_path = source / "conv_0/dated_unmerged385/group_inputs.json"
        item = {"conversation": 0, "arm": "dated_unmerged385", "input": self.put(input_path, r.sealed({"turns": ["a", "b"]}))}
        self.put(source / "run_manifest.json", {"original": True})
        audit_path = self.root / "audit.json"
        self.put(audit_path, {"audit": True})
        parent = {"protocol_sha256": "parent", "inputs": [item], "schema": r.frozen.SCHEMA}
        with patch.object(r, "source_bindings", return_value=(parent, [])), patch.object(r, "diagnostic_evidence", return_value={"offline_mock": True}), patch.object(r.PacedEmbeddingAPI, "__init__", side_effect=AssertionError("No API")):
            result = r.prepare(self.root / "recovery", source, audit_path)
        self.assertEqual(result["model_calls"], 0)
        output = self.root / "recovery/conv_0/dated_unmerged385/group_inputs.json"
        self.assertEqual(output.read_bytes(), input_path.read_bytes())
        self.assertFalse((output.parent / "native").exists())
        with self.assertRaises(FileExistsError):
            r.prepare(self.root / "recovery", source, audit_path)


if __name__ == "__main__":
    unittest.main()
