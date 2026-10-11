"""Recovery QA identity, no-restart and durable transport tests; only fake APIs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_mem0_sparse1540_recovered_qa_v2 as r
from tests.test_mem0_transfer_qa import FakeBase, facts


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return r.base.sha(path)


class InputMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, value in (("OUT", self.root / "new"), ("ROOT", self.root)):
            p = patch.object(r, name, value); p.start(); self.addCleanup(p.stop)
        for name, value in (("OUT", self.root / "old"), ("BASE", self.root / "baseline")):
            p = patch.object(r.base, name, value); p.start(); self.addCleanup(p.stop)
        self.q = {"id": "c0_q0", "conversation": 0, "category": 1, "question": "A question", "reference": "an answer"}
        self.source = {"models": {"writer_reader": "gemini-2.5-flash", "judge": "gemini-3.5-flash-lite"}}
        source = r.base.OUT / "manifest.json"
        recovery = r.OUT / "recovery_policy.json"
        self.questions = [{**self.q, "id": "c%d_q%d" % (i % 10, i), "conversation": i % 10} for i in range(1540)]
        self.manifest = {"policy": r.POLICY, "source_manifest": {"path": str(source), "sha256": dump(source, self.source)},
            "recovery_policy": {"path": str(recovery), "sha256": dump(recovery, {"policy": r.POLICY})},
            "models": self.source["models"], "provider_retries": 2, "conversation_workers": 8,
            "query_workers": 4, "final_unique_facts": 20, "question_ids": [q["id"] for q in self.questions],
            "rows": [{"conversation": i, "arm": a} for i in range(10) for a in r.ARMS],
            "code_sha256": {"new_code.py": dump(self.root / "new_code.py", {"code": 1})}}
        self.write_manifest()

    def write_manifest(self):
        m = self.manifest
        audit = {"policy": r.POLICY, "admitted": True, "source_manifest_sha256": m["source_manifest"]["sha256"],
            "recovery_policy_sha256": m["recovery_policy"]["sha256"], "rows_sha256": r.base.fingerprint(m["rows"]),
            "question_ids_sha256": r.base.fingerprint(m["question_ids"]), "model_calls": 0,
            "audited_files": {m["source_manifest"]["path"]: m["source_manifest"]["sha256"]}}
        path = r.OUT / "audit.json"
        m["input_audit"] = {"path": str(path), "sha256": dump(path, audit)}
        dump(r.OUT / "input_manifest.json", m)

    def verify(self):
        with patch.object(r.base, "verify_manifest", return_value=self.source), \
             patch.object(r.base, "questions", return_value=self.questions), \
             patch.object(r, "REQUIRED_CODE", ("new_code.py",)), \
             patch.object(r, "validate_row", side_effect=lambda row, qs: {"row": row}), \
             patch.object(r.base.ModelAPI, "__init__", side_effect=AssertionError("No provider")):
            return r.verify_inputs()

    def test_twenty_rows_full_cohort_admission_read_only(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        m, digest, loaded = self.verify()
        self.assertEqual(len(loaded), 20)
        self.assertEqual(digest, r.base.sha(r.OUT / "input_manifest.json"))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_duplicate_row_rejected_even_with_resigned_audit(self):
        self.manifest["rows"][-1] = self.manifest["rows"][0]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "twenty"):
            self.verify()

    def test_nonadmitted_or_stale_audit_rejected(self):
        p = Path(self.manifest["input_audit"]["path"])
        value = json.loads(p.read_bytes()); value["admitted"] = False
        self.manifest["input_audit"]["sha256"] = dump(p, value)
        dump(r.OUT / "input_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "audit"):
            self.verify()
        value["admitted"] = True; value["rows_sha256"] = "wrong"
        self.manifest["input_audit"]["sha256"] = dump(p, value)
        dump(r.OUT / "input_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "audit"):
            self.verify()

    def test_code_source_models_cohort_and_retry_are_frozen(self):
        for key, changed in (("models", {}), ("question_ids", []), ("provider_retries", 3), ("query_workers", 8)):
            old = self.manifest[key]; self.manifest[key] = changed; self.write_manifest()
            with self.assertRaises(ValueError, msg=key):
                self.verify()
            self.manifest[key] = old
        self.write_manifest(); (self.root / "new_code.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "code"):
            self.verify()

    def make_row(self, ci=0, arm=None, recovered=False):
        arm = arm or r.ARMS[0]
        self.q["conversation"] = ci; self.q["id"] = "c%d_q0" % ci
        directory = r.OUT / "recovered" / arm / ("conv_" + str(ci)) if recovered else r.base.conv_path(ci)
        bank_path = (directory if arm == r.ARMS[1] else r.base.conv_path(ci)) / ("session_bank.json" if arm == r.ARMS[0] else "regroup_bank.json")
        native = directory / "native" if recovered or arm == r.ARMS[1] else r.base.BASE / ("conv_" + str(ci))
        fs = [{**f, "embedding": [1.] + [0.] * 767} for f in facts()]
        bank = {"schema": r.base.POLICY + "-bank", "conversation": ci, "arm": arm,
            "complete": True, "source_dataset_sha256": r.base.SOURCE_HASH, "facts": fs}
        digest = dump(bank_path, bank)
        retrieval_dir = directory / "retrieval" / arm
        ret = r.base.make_retrieval(self.q, arm, fs, digest, {"synthetic": True})
        row = {"conversation": ci, "arm": arm, "bank_path": str(bank_path), "bank_sha256": digest,
            "retrieval_dir": str(retrieval_dir), "retrieval_files": {self.q["id"]: dump(retrieval_dir / (self.q["id"] + ".json"), ret)},
            "index_recovery": None, "native_ledger_path": str(native / "api_calls.jsonl"), "native_cache_dir": str(native / "api_cache")}
        if recovered:
            p = directory / "index_recovery.json"
            row["index_recovery"] = {"path": str(p), "sha256": dump(p, {"synthetic_recovery": True})}
        return row

    def test_rank_replay_and_bank_canonical_text_date(self):
        row = self.make_row()
        self.assertEqual(len(r.validate_row(row, [self.q])["retrievals"]), 1)
        p = Path(row["retrieval_dir"]) / (self.q["id"] + ".json")
        value = json.loads(p.read_bytes()); value["selected"][0]["text"] = "Fabricated"
        value = r.base._sealed({k: v for k, v in value.items() if k != "content_sha256"})
        row["retrieval_files"][self.q["id"]] = dump(p, value)
        with self.assertRaisesRegex(ValueError, "replay"):
            r.validate_row(row, [self.q])

    def test_changed_bytes_and_extra_retrieval_rejected(self):
        row = self.make_row()
        dump(Path(row["retrieval_dir"]) / "foreign.json", {})
        with self.assertRaisesRegex(ValueError, "extra"):
            r.validate_row(row, [self.q])
        Path(row["bank_path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "bytes"):
            r.validate_row(row, [self.q])

    def test_both_conv3_recoveries_required_and_original_paths_preserved(self):
        for arm in r.ARMS:
            row = self.make_row(3, arm, True)
            self.assertEqual(r.validate_row(row, [self.q])["row"], row)
            row["index_recovery"] = None
            with self.assertRaisesRegex(ValueError, "Both conv3"):
                r.validate_row(row, [self.q])
        row = self.make_row(0, r.ARMS[1], True)
        with self.assertRaisesRegex(ValueError, "Only"):
            r.validate_row(row, [self.q])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p = patch.object(r, "OUT", self.root); p.start(); self.addCleanup(p.stop)
        self.q = {"id": "c0_q0", "conversation": 0, "category": 1, "question": "Question", "reference": "an answer"}
        self.input_hash = dump(self.root / "input_manifest.json", {"synthetic": True})
        self.manifest = {"source_manifest": {"sha256": "frozen-source"}}
        self.loaded = {(0, arm): {"row": {"bank_sha256": "bank", "retrieval_files": {self.q["id"]: "retrieval"}},
            "retrievals": {self.q["id"]: {"selected": facts()}}} for arm in r.ARMS}
        self.provider = None

    def run_worker(self, failure=False):
        def factory(path, **kwargs):
            self.assertTrue((self.root / "conv_0/qa_status.json").exists())
            self.provider = FakeBase(path, **kwargs)
            self.provider._redact = lambda x: x
            self.provider.shared["transport_failure"] = failure
            return self.provider
        with patch.object(r, "verify_inputs", return_value=(self.manifest, self.input_hash, self.loaded)), \
             patch.object(r.base, "questions", return_value=[self.q]), \
             patch.object(r.base, "load_data", return_value=[{}]), \
             patch.object(r.base, "session_envelopes", return_value=[{"observation_date": "2023-01-01"}]), \
             patch.object(r.base, "ModelAPI", side_effect=factory), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
            return r.qa_conv(0, self.input_hash)

    def test_new_namespace_original_evaluator_two_arms_and_binding(self):
        state = self.run_worker()
        self.assertEqual((state["status"], state["done"], state["completed"]), ("complete", 2, 2))
        self.assertEqual(len(self.provider.shared["calls"]), 2)  # Shared identical input cache; four consumers.
        for arm in r.ARMS:
            row = r.base._verify_sealed(r.base.read(self.root / "conv_0/qa" / arm / "c0_q0.json"))
            self.assertEqual(row["binding"]["input_manifest_sha256"], self.input_hash)
            self.assertEqual(row["binding"]["source_manifest_sha256"], "frozen-source")
            self.assertEqual(row["result"]["token_f1"], 1.)
            self.assertEqual(row["api_events"][0]["context"]["architecture"], r.POLICY)
            self.assertEqual(len(row["selected_ids"]), 20)

    def test_failed_unknown_is_terminal_and_cannot_restart(self):
        state = self.run_worker(True)
        self.assertEqual((state["status"], state["done"], state["failed"]), ("requires_review", 2, 2))
        self.assertEqual(len(self.provider.shared["calls"]), 1)
        with self.assertRaisesRegex(ValueError, "no automatic restart"):
            self.run_worker()

    def test_orphan_cache_or_attempt_blocks_client_creation(self):
        (self.root / "conv_0/qa_api").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Prior QA"):
            self.run_worker()
        self.assertIsNone(self.provider)

    def test_manifest_mismatch_blocks_before_provider(self):
        with patch.object(r, "verify_inputs", return_value=(self.manifest, "changed", self.loaded)), \
             patch.object(r.base, "ModelAPI", side_effect=AssertionError("No provider")):
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                r.qa_conv(0, self.input_hash)

    def test_os_lock_releases_but_retains_file(self):
        path = self.root / "guard.lock"
        with r.namespace_lock(path):
            with self.assertRaises((OSError, BlockingIOError)):
                with r.namespace_lock(path):
                    self.fail("Must not acquire overlapping lock")
        with r.namespace_lock(path):
            pass
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
