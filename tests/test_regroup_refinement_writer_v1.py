"""Read-only preparation/gating and mock writer wiring; no model calls."""
import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_regroup_refinement_writer_v1 as w
from jev_mem.topic_grouping import raw_turns_from_locomo, render_group


def item():
    return {"conversation": {"speaker_a": "Alice", "speaker_b": "Bob",
                             "session_1_date_time": "1:00 pm on 01 May, 2023",
                             "session_1": [{"dia_id": "D1:1", "speaker": "Alice", "text": "I started a project."},
                                           {"dia_id": "D1:2", "speaker": "Bob", "text": "Which project?"}]}}


def grouping(ci):
    raw = raw_turns_from_locomo(item()["conversation"])
    segments = [{"segment_id": f"s{i}", "source_turn_ids": [turn["source_turn_id"]], "turns": [turn], "text": render_group([turn])}
                for i, turn in enumerate(raw)]
    return {"conversation": ci, "status": "complete", "qa_visible": False,
            "segments": segments,
            "groups": [{"group_id": "g0", "segment_ids": ["s0", "s1"], "turns": raw, "text": render_group(raw),
                        "source_turn_ids": [t["source_turn_id"] for t in raw]}]}


class WriterDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jev_writer_driver_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.groups = self.root / "groups"
        for ci in range(10):
            w.frozen.save_new(self.groups / f"conv_{ci}" / "grouping.json", grouping(ci))
        self.parent = {"schema": w.frozen.SCHEMA, "protocol_sha256": "old",
                       "protocol_changes": [], "code_sha256": {}}
        self.resource = {"snapshot": "mock-local-only", "files_sha256": {}}
        for target, replacement in (
            ("resources", lambda: self.resource),
            ("validate_resources", lambda *args: None),
            ("PacedEmbeddingAPI", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("API forbidden"))),
        ):
            p = patch.object(w, target, replacement); p.start(); self.addCleanup(p.stop)
        p = patch.object(w.frozen, "load_data", lambda: [copy.deepcopy(item()) for _ in range(10)])
        p.start(); self.addCleanup(p.stop)

    def prepare(self, arm="llm_fullconv"):
        run = self.root / arm
        with patch.object(w.frozen, "manifest", return_value=copy.deepcopy(self.parent)):
            value = w.prepare(run, self.groups, arm, arm == "source_local_context")
        return run, json.loads((run / "run_manifest.json").read_text(encoding="utf8")), value

    def test_all_three_arms_prepare_complete_bound_inputs_without_models(self):
        for arm in w.ARMS:
            with self.subTest(arm=arm):
                run, m, result = self.prepare(arm)
                self.assertEqual(result["calls"], 0)
                self.assertEqual(result["groups"], 10)
                self.assertEqual(len(m["inputs"]), 10)
                self.assertEqual(m["refinement"]["source_local_context"], arm == "source_local_context")
                self.assertEqual(m["refinement"]["policy"]["api_cache_roots"], [])
                with patch.object(w.frozen, "manifest", return_value=m):
                    self.assertEqual(w.manifest(run), m)

    def test_last_incomplete_grouping_leaves_no_partial_namespace(self):
        path = self.groups / "conv_9" / "grouping.json"
        value = grouping(9); value["status"] = "running"
        w.publish_json(path, w.frozen.sealed(value))
        run = self.root / "rejected"
        with patch.object(w.frozen, "manifest", return_value=self.parent):
            with self.assertRaisesRegex(ValueError, "All ten complete"):
                w.prepare(run, self.groups, "llm_fullconv")
        self.assertFalse(run.exists())

    def test_manifest_rejects_changed_group_source_bytes(self):
        run, m, _ = self.prepare()
        path = self.groups / "conv_3" / "grouping.json"
        path.write_text(path.read_text(encoding="utf8") + "\n", encoding="utf8")
        with patch.object(w.frozen, "manifest", return_value=m):
            with self.assertRaisesRegex(ValueError, "Changed source"):
                w.manifest(run)

    def test_arm_and_context_flag_cannot_silently_differ(self):
        with self.assertRaisesRegex(ValueError, "declared context"):
            w.prepare(self.root / "bad", self.groups, "jev_refined", True)
        with self.assertRaisesRegex(ValueError, "declared arm"):
            w.prepare(self.root / "bad", self.groups, "undeclared")

    def review(self, run, m, bank_protocol=None):
        banks = []
        for ci in (0, 1):
            p = run / f"conv_{ci}" / m["arms"][0] / "bank.json"
            w.frozen.save_new(p, {"complete": True, "conversation": ci, "arm": m["arms"][0],
                                  "protocol_sha256": bank_protocol or m["protocol_sha256"]})
            banks.append(w.frozen.pointer(p))
        evidence = run / "review_evidence.json"
        w.frozen.save_new(evidence, {"status": "passed", "protocol_sha256": m["protocol_sha256"]})
        w.frozen.save_new(run / "development_review.json", {
            "status": "passed", "protocol_sha256": m["protocol_sha256"], "banks": banks,
            "source_date_audits": [w.frozen.pointer(evidence)], "index_audits": [w.frozen.pointer(evidence)]})

    def test_expansion_requires_both_reviewed_banks_same_protocol(self):
        run, m, _ = self.prepare()
        with self.assertRaises(FileNotFoundError):
            w.admit_expansion(run, m)
        self.review(run, m, bank_protocol="wrong")
        with self.assertRaisesRegex(ValueError, "mismatched reviewed bank"):
            w.admit_expansion(run, m)

    def test_direct_worker_cannot_bypass_development_gate(self):
        run, m, _ = self.prepare()
        with patch.object(w, "manifest", return_value=m):
            with self.assertRaises(FileNotFoundError):
                w.worker(run, 2)
        self.assertFalse((run / "conv_2" / m["arms"][0] / "write_intent.json").exists())

    def test_valid_review_admits_expansion(self):
        run, m, _ = self.prepare()
        self.review(run, m)
        self.assertIsNone(w.admit_expansion(run, m))

    def test_prior_intent_is_never_reentered(self):
        run, m, _ = self.prepare()
        folder = run / "conv_0" / m["arms"][0]
        (folder / "write_intent.json").write_text("{}", encoding="utf8")
        with patch.object(w, "manifest", return_value=m), patch.object(w.frozen, "write_arm") as call:
            with self.assertRaisesRegex(RuntimeError, "Unresolved write"):
                w.worker(run, 0)
            call.assert_not_called()

    def test_context_factory_receives_materialized_sessions_and_restores_hooks(self):
        run, m, _ = self.prepare("source_local_context")
        arm = m["arms"][0]
        original_api, original_adapter = w.frozen.ModelAPI, w.frozen.Mem0GroupAdapter

        def fake_write(actual_run, ci, actual_arm, actual_manifest):
            self.assertEqual(actual_arm, arm)
            self.assertEqual(actual_manifest["protocol_sha256"], m["protocol_sha256"])
            factory = w.frozen.Mem0GroupAdapter
            self.assertIsInstance(factory.keywords["source_sessions"], list)
            self.assertEqual(len(factory.keywords["source_sessions"]), 1)
            self.assertEqual(len(factory.keywords["segments"]), 2)
            folder = actual_run / f"conv_{ci}" / actual_arm
            bank_path = folder / "bank.json"
            w.frozen.save_new(bank_path, {"complete": True, "conversation": ci, "arm": actual_arm,
                                         "protocol_sha256": m["protocol_sha256"]})
            state = {"status": "complete", "bank": w.frozen.pointer(bank_path)}
            w.publish_json(folder / "write_status.json", state)
            return state

        with patch.object(w, "manifest", return_value=m), \
                patch.object(w, "local_bm25_context", return_value=contextlib.nullcontext()), \
                patch.object(w.frozen, "write_arm", side_effect=fake_write) as write:
            result = w.worker(run, 0)
            replay = w.worker(run, 0)
            self.assertEqual(write.call_count, 1)
        self.assertEqual(result, {k: v for k, v in replay.items() if k != "content_sha256"})
        self.assertEqual(result["status"], "complete")
        self.assertIs(w.frozen.ModelAPI, original_api)
        self.assertIs(w.frozen.Mem0GroupAdapter, original_adapter)


if __name__ == "__main__":
    unittest.main()
