from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import recover_mem0_regroup_eof_v2 as entry


class EOFEntryV2Tests(unittest.TestCase):
    def test_markdown_pointer_never_json_decoded_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.md"
            path.write_text("# Markdown policy\n\nOnly one ASCII brace.\n", encoding="utf8")
            pointer = entry.file_pointer(path)
            with patch.object(entry.v1.frozen, "check_pointer", side_effect=AssertionError("No JSON decoder")):
                self.assertEqual(entry.check_file_pointer(pointer), path)
            path.write_text("# Changed\n", encoding="utf8")
            with self.assertRaises(ValueError):
                entry.check_file_pointer(pointer)

    def test_frozen_prior_prepared_v1_has_no_write_attempt(self):
        prior = entry.prior_prepared_binding()
        self.assertEqual(len(prior["files"]), 2)
        self.assertEqual(prior["failure"]["bank_intents"], 0)
        self.assertEqual(prior["failure"]["model_calls"], 0)
        self.assertEqual(prior["protocol_sha256"], entry.PREPARED_V1_PROTOCOL)

    def test_existing_or_nested_namespace_rejected_before_source_reads(self):
        with patch.object(entry.v1, "source_binding", side_effect=AssertionError("No scan")):
            with self.assertRaises(FileExistsError):
                entry.prepare(entry.PREPARED_V1, "unused")
            with self.assertRaises(FileExistsError):
                entry.prepare(entry.PREPARED_V1 / "nested", "unused")

    def test_unknown_once_guard_prevents_writer_reentry(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            folder = run / "conv_8/dated_merged"
            folder.mkdir(parents=True)
            (folder / "eof_rebuild_intent.json").write_text("{}")
            with patch.object(entry, "manifest", return_value={}), patch.object(entry.v1.frozen, "write_arm") as writer:
                with self.assertRaisesRegex(RuntimeError, "no reentry"):
                    entry.run_once(run)
                writer.assert_not_called()

    def test_real_prepare_then_manifest_full_offline(self):
        proof = entry.ROOT / "research/mem0_regroup_eof_prefix_probe_v2_2026-09-29.json"
        self.assertTrue(proof.is_file(), "Run the frozen native zero-network prefix probe first")
        old_before = entry.prior_prepared_binding()
        # A unique temporary direct child meets the real entry namespace rule.
        # Remove only the initially empty directory so production prepare creates it.
        with tempfile.TemporaryDirectory(prefix="_offline_eof_entry_v2_", dir=entry.ROOT / "runs") as directory:
            run = Path(directory).resolve()
            self.assertEqual(run.parent, (entry.ROOT / "runs").resolve())
            run.rmdir()
            with entry.v1.forbid_network() as guards, patch("jev_mem.providers.load_credentials", side_effect=AssertionError("No credentials")):
                prepared = entry.prepare(run, proof)
                value = entry.manifest(run)
                self.assertEqual(prepared["protocol_sha256"], value["protocol_sha256"])
                self.assertEqual(value["eof_recovery"]["policy_document"]["bytes"], entry.v1.POLICY_DOC.stat().st_size)
                self.assertTrue(all(guard.call_count == 0 for guard in guards))
            self.assertEqual(len([p for p in run.rglob("*") if p.is_file()]), 2)
            self.assertFalse((run / "conv_8/dated_merged/native").exists())
            self.assertFalse((run / "conv_8/dated_merged/eof_rebuild_intent.json").exists())
        self.assertEqual(entry.prior_prepared_binding(), old_before)


if __name__ == "__main__":
    unittest.main()
