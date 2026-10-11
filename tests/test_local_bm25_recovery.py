"""Offline checks of resource resolution and the single-bank recovery wrapper."""
import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from jev_mem import local_bm25_context as bm
from tools import recover_mem0_regroup_bm25_v1 as run


class LocalBM25Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "english.txt").write_text("the\nand\n", encoding="utf8")
        self.hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.snapshot.iterdir()}

    def test_constructor_only_adds_resource_kwargs(self):
        import fastembed
        sentinel = object()
        original = Mock(return_value=sentinel)
        with patch.object(fastembed, "SparseTextEmbedding", original):
            with bm.local_bm25_context(self.snapshot, self.hashes):
                result = fastembed.SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir="existing", k=1.2, b=0.75)
                self.assertIs(result, sentinel)
                original.assert_called_once_with(model_name="Qdrant/bm25", cache_dir="existing", k=1.2, b=0.75,
                    specific_model_path=str(self.snapshot), local_files_only=True)
            self.assertIs(fastembed.SparseTextEmbedding, original)

    def test_other_models_delegated_unchanged(self):
        import fastembed
        original = Mock()
        with patch.object(fastembed, "SparseTextEmbedding", original), bm.local_bm25_context(self.snapshot, self.hashes):
            fastembed.SparseTextEmbedding("other/model", cache_dir="other", custom=True)
            original.assert_called_once_with("other/model", cache_dir="other", custom=True)

    def test_positional_bm25_and_restore_after_exception(self):
        import fastembed
        original = Mock()
        with patch.object(fastembed, "SparseTextEmbedding", original):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                with bm.local_bm25_context(self.snapshot, self.hashes):
                    fastembed.SparseTextEmbedding("Qdrant/bm25")
                    self.assertTrue(original.call_args.kwargs["local_files_only"])
                    raise RuntimeError("interruption")
            self.assertIs(fastembed.SparseTextEmbedding, original)
        self.assertFalse(bm._ACTIVE)

    def test_changed_snapshot_rejected_before_constructor(self):
        import fastembed
        original = Mock()
        (self.snapshot / "english.txt").write_text("tampered")
        with patch.object(fastembed, "SparseTextEmbedding", original):
            with self.assertRaisesRegex(ValueError, "resource changed"):
                with bm.local_bm25_context(self.snapshot, self.hashes):
                    pass
        original.assert_not_called()

    def test_no_fabricated_missing_files(self):
        before = list(self.snapshot.iterdir())
        bm.validate_resources(self.snapshot, self.hashes)
        self.assertEqual(before, list(self.snapshot.iterdir()))
        self.assertFalse((self.snapshot / "mock.file").exists())
        self.assertFalse((self.snapshot / "tamil.txt").exists())

    def test_foreign_snapshot_or_network_opt_in_rejected(self):
        import fastembed
        original = Mock()
        with patch.object(fastembed, "SparseTextEmbedding", original), bm.local_bm25_context(self.snapshot, self.hashes):
            with self.assertRaisesRegex(ValueError, "another"):
                fastembed.SparseTextEmbedding("Qdrant/bm25", specific_model_path=str(self.root))
            with self.assertRaisesRegex(ValueError, "network"):
                fastembed.SparseTextEmbedding("Qdrant/bm25", local_files_only=False)
        original.assert_not_called()

    def test_nested_context_rejected_and_outer_restored(self):
        import fastembed
        original = fastembed.SparseTextEmbedding
        with bm.local_bm25_context(self.snapshot, self.hashes):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with bm.local_bm25_context(self.snapshot, self.hashes):
                    pass
        self.assertIs(fastembed.SparseTextEmbedding, original)


class SingleBankWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_prepare_only_copies_target_input(self):
        source = self.root / "old"
        source.mkdir()
        (source / "run_manifest.json").write_text("{}")
        inp = source / "inputs.json"
        inp.write_text(json.dumps(run.frozen.sealed({"fixed": [1, 2, 3]})))
        retained = {"conversation": 0, "arm": "dated_merged", "input": run.frozen.pointer(inp)}
        target = {"conversation": 1, "arm": "dated_merged", "input": run.frozen.pointer(inp)}
        parent = {"protocol_sha256": "old", "inputs": [retained, target], "schema": run.frozen.SCHEMA}
        reused = [{"bank": "pointer_only"}] * 3
        resources = {"snapshot": "snapshot", "files_sha256": {}}
        audit = self.root / "audit.json"
        audit.write_text("{}")
        report = {"facts_compared": 518, "exact_matches": 518, "network_calls": 0}
        with patch.object(run, "source_admission", return_value=(parent, reused, {"zero_model_calls": True}, resources)), patch.object(run, "offline_acceptance", return_value=report), patch.object(run, "AUDIT", audit):
            result = run.prepare(self.root / "new", source)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["new_writer_banks"], 1)
        self.assertFalse((self.root / "new/conv_0").exists())
        self.assertFalse((self.root / "new/conv_1/dated_merged/native").exists())
        manifest = json.loads((self.root / "new/run_manifest.json").read_text())
        self.assertEqual(manifest["inputs"][0], retained)
        self.assertEqual(manifest["local_bm25_recovery"]["reused_complete_banks"], reused)
        self.assertEqual((self.root / "new/conv_1/dated_merged/group_inputs.json").read_bytes(), inp.read_bytes())
        with self.assertRaises(FileExistsError):
            run.prepare(self.root / "new", source)

    def test_write_delegates_only_one_target_inside_local_context(self):
        active = []
        @contextlib.contextmanager
        def local(*args):
            active.append(True)
            try:
                yield
            finally:
                active.pop()
        value = {"local_bm25_recovery": {"reused_complete_banks": [1, 2, 3], "resources": {"snapshot": "fixed", "files_sha256": {"english": "hash"}}}}
        def delegated(path, ci, arm, manifest):
            self.assertEqual((ci, arm), (1, "dated_merged"))
            self.assertTrue(active)
            return {"status": "complete"}
        with patch.object(run, "manifest", return_value=value), patch.object(run, "offline_acceptance", return_value={}), patch.object(run.frozen, "writer_process_lock", side_effect=lambda *a: contextlib.nullcontext()), patch.object(run, "local_bm25_context", side_effect=local), patch.object(run.legacy, "run_arm", side_effect=delegated) as call:
            self.assertEqual(run.write(self.root), {"status": "complete"})
        self.assertEqual(call.call_count, 1)
        self.assertEqual(active, [])

    def test_new_wrapper_does_not_catch_or_retry_legacy_unknown_guard(self):
        value = {"local_bm25_recovery": {"reused_complete_banks": [], "resources": {"snapshot": "fixed", "files_sha256": {}}}}
        with patch.object(run, "manifest", return_value=value), patch.object(run, "offline_acceptance", return_value={}), patch.object(run.frozen, "writer_process_lock", side_effect=lambda *a: contextlib.nullcontext()), patch.object(run, "local_bm25_context", side_effect=lambda *a: contextlib.nullcontext()), patch.object(run.legacy, "run_arm", side_effect=RuntimeError("Prior nontransactional rebuild")) as call:
            with self.assertRaisesRegex(RuntimeError, "Prior nontransactional"):
                run.write(self.root)
        self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
