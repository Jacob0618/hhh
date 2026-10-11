"""Class API regression and fresh native Mem0 initialization checks, offline."""
import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_mem import local_bm25_context_v2 as bm
from tools import recover_mem0_regroup_bm25_v2 as run


class FakeSparse:
    calls = []
    EMBEDDINGS_REGISTRY = ["original_registry"]

    @classmethod
    def list_supported_models(cls):
        return [{"model": "Qdrant/bm25"}, {"model": "other/model"}]

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.calls.append((args, kwargs))


class ClassContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.snapshot = Path(self.temp.name) / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "english.txt").write_text("the\nand\n")
        self.hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.snapshot.iterdir()}
        FakeSparse.calls = []

    def test_classmethods_registry_and_subclass_identity_preserved(self):
        import fastembed
        with patch.object(fastembed, "SparseTextEmbedding", FakeSparse):
            with bm.local_bm25_context(self.snapshot, self.hashes):
                replacement = fastembed.SparseTextEmbedding
                self.assertIsInstance(replacement, type)
                self.assertTrue(issubclass(replacement, FakeSparse))
                self.assertEqual(replacement.list_supported_models(), FakeSparse.list_supported_models())
                self.assertIs(replacement.EMBEDDINGS_REGISTRY, FakeSparse.EMBEDDINGS_REGISTRY)
                value = replacement("Qdrant/bm25", k=1.2, cache_dir="unchanged")
                self.assertIsInstance(value, FakeSparse)
                self.assertEqual(value.kwargs, {"k": 1.2, "cache_dir": "unchanged", "specific_model_path": str(self.snapshot), "local_files_only": True})
            self.assertIs(fastembed.SparseTextEmbedding, FakeSparse)

    def test_other_model_unchanged(self):
        import fastembed
        with patch.object(fastembed, "SparseTextEmbedding", FakeSparse), bm.local_bm25_context(self.snapshot, self.hashes):
            value = fastembed.SparseTextEmbedding("other/model", foo="bar")
            self.assertEqual(value.args, ("other/model",))
            self.assertEqual(value.kwargs, {"foo": "bar"})

    def test_captured_alias_stops_pinning_after_exit(self):
        import fastembed
        with patch.object(fastembed, "SparseTextEmbedding", FakeSparse):
            with bm.local_bm25_context(self.snapshot, self.hashes):
                alias = fastembed.SparseTextEmbedding
            outside = alias("Qdrant/bm25")
            self.assertEqual(outside.kwargs, {})

    def test_restore_after_exception_and_nested_rejection(self):
        import fastembed
        with patch.object(fastembed, "SparseTextEmbedding", FakeSparse):
            with self.assertRaisesRegex(RuntimeError, "probe error"):
                with bm.local_bm25_context(self.snapshot, self.hashes):
                    with self.assertRaisesRegex(RuntimeError, "already active"):
                        with bm.local_bm25_context(self.snapshot, self.hashes):
                            pass
                    raise RuntimeError("probe error")
            self.assertIs(fastembed.SparseTextEmbedding, FakeSparse)
            self.assertFalse(bm._ACTIVE)

    def test_resource_change_and_network_option_rejected(self):
        import fastembed
        with patch.object(fastembed, "SparseTextEmbedding", FakeSparse), bm.local_bm25_context(self.snapshot, self.hashes):
            with self.assertRaisesRegex(ValueError, "network"):
                fastembed.SparseTextEmbedding("Qdrant/bm25", local_files_only=False)
        (self.snapshot / "english.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "resource changed"):
            with bm.local_bm25_context(self.snapshot, self.hashes):
                pass


class WrapperTests(unittest.TestCase):
    def test_real_native_fresh_process_offline(self):
        probe = run.read(run.prior.PROBE)
        proof = run.fresh_native_smoke({"snapshot": probe["snapshot"], "files_sha256": probe["snapshot_files_sha256"]})
        self.assertTrue(proof["fresh_import_order_verified"])
        self.assertTrue(proof["native_open_memory"])
        self.assertTrue(proof["entity_bm25"])
        self.assertEqual(proof["network_calls"], 0)
        self.assertEqual(proof["model_api_calls"], 0)

    def test_failed_guard_not_retried(self):
        value = {"local_bm25_class_recovery": {"reused_complete_banks": [], "resources": {"snapshot": "fixed", "files_sha256": {}}}}
        with patch.object(run, "manifest", return_value=value), patch.object(run, "acceptance", return_value={}), patch.object(run.frozen, "writer_process_lock", side_effect=lambda *a: contextlib.nullcontext()), patch.object(run, "local_bm25_context", side_effect=lambda *a: contextlib.nullcontext()), patch.object(run.legacy, "run_arm", side_effect=RuntimeError("Prior nontransactional rebuild")) as call:
            with self.assertRaisesRegex(RuntimeError, "Prior nontransactional"):
                run.write(Path("unused"))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[1:3], (1, "dated_merged"))

    def test_prepare_only_target_and_inherited_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source = root / "prior";source.mkdir()
            (source / "run_manifest.json").write_text("{}")
            (source / "offline_bm25_acceptance.json").write_text("{\"inherited\":true}")
            inp = source / "input.json";inp.write_text("{\"ordered\":[1,2,3]}")
            parent = {"protocol_sha256": "old", "inputs": [
                {"conversation": 0, "arm": "dated_merged", "input": run.frozen.pointer(inp)},
                {"conversation": 1, "arm": "dated_merged", "input": run.frozen.pointer(inp)}]}
            audit = root / "audit.json";audit.write_text("{}")
            proof = {"fresh_native": {"native_open_memory": True}, "vectors": {"exact_matches": 518}}
            with patch.object(run, "source_admission", return_value=(parent, [{"old": 1}] * 3, {"fixed": True})), patch.object(run, "acceptance", return_value=proof), patch.object(run, "FAILURE_AUDIT", audit):
                result = run.prepare(root / "new", source)
            self.assertEqual(result["model_calls"], 0)
            self.assertFalse((root / "new/conv_0").exists())
            self.assertEqual((root / "new/conv_1/dated_merged/group_inputs.json").read_bytes(), inp.read_bytes())
            self.assertEqual((root / "new/offline_bm25_acceptance.json").read_bytes(), (source / "offline_bm25_acceptance.json").read_bytes())
            self.assertFalse(list((root / "new").rglob("native")))


if __name__ == "__main__":
    unittest.main()
