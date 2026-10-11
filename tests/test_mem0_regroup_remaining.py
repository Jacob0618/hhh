"""Offline expansion gate, namespace, scope and one-attempt checks."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_mem0_regroup_remaining_v1 as run


class RemainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, path, value, seal=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.frozen.sealed(value) if seal else value), encoding="utf8")
        return run.frozen.pointer(path)

    def gate_fixture(self):
        old, fourth = self.root / "old", self.root / "fourth"
        parent = {"inputs": [], **{k: "frozen_" + k for k in run.WRITER_FIELDS}}
        rows = []
        for ci in (0, 1):
            for arm in run.frozen.ARMS:
                folder = (fourth if (ci, arm) == (1, "dated_merged") else old) / f"conv_{ci}" / arm
                inp = self.put(folder / "group_inputs.json", {"envelopes": [{"group": "fixed"}]}, True)
                parent["inputs"].append({"conversation": ci, "arm": arm, "input": inp})
                bank = self.put(folder / "bank.json", {"conversation": ci, "arm": arm, "complete": True,
                    "protocol_sha256": "fixture", "facts": [{"id": f"{ci}:{arm}"}], "group_inputs": inp}, True)
                state = self.put(folder / "write_status.json", {"status": "complete", "protocol_sha256": "fixture",
                    "facts": 1, "completed_groups": 1, "expected_groups": 1, "bank": bank})
                outcome = self.put(folder / "recovery_outcome.json", {"status": "complete", "protocol_sha256": "fixture",
                    "bank": bank, "write_status": state}, True)
                rows.append({"conversation": ci, "arm": arm, "bank": bank, "write_status": state, "outcome": outcome})
        audit = self.put(self.root / "offline_fixture_review.json", {"fixture_only": True})
        gate = {"schema": run.GATE_SCHEMA, "status": "passed", "banks": rows,
            "checks": {k: True for k in run.CHECKS}, "audits": {"source_date": [audit], "index": [audit]},
            "limitations": ["Synthetic unit-test fixture, not a real scientific gate."]}
        path = self.root / "fixture_gate.json";self.put(path, gate, True)
        return parent, gate, path, old, fourth

    def test_four_bank_gate_and_known_limitations_admitted(self):
        parent, gate, path, old, fourth = self.gate_fixture()
        source = {**parent, "protocol_sha256": "fixture"}
        with patch.object(run, "DEVELOPMENT", old), patch.object(run, "DEVELOPMENT_FOURTH", fourth), patch.object(run.once, "manifest", return_value=source) as first, patch.object(run.class_recovery, "manifest", return_value=source) as second:
            result = run.development_gate(path, parent)
        self.assertEqual(result["banks"], gate["banks"])
        first.assert_called_once_with(old)
        second.assert_called_once_with(fourth)

    def test_gate_rejects_frozen_source_protocol_or_model_mismatch(self):
        parent, gate, path, old, fourth = self.gate_fixture()
        source = {**parent, "protocol_sha256": "fixture"}
        with patch.object(run, "DEVELOPMENT", old), patch.object(run, "DEVELOPMENT_FOURTH", fourth), patch.object(run.once, "manifest", return_value=source):
            with patch.object(run.class_recovery, "manifest", return_value={**source, "protocol_sha256": "different_run"}):
                with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                    run.development_gate(path, parent)
            with patch.object(run.class_recovery, "manifest", return_value={**source, "models": "other_model"}):
                with self.assertRaisesRegex(ValueError, "writer/model/input protocol"):
                    run.development_gate(path, parent)
            with patch.object(run.class_recovery, "manifest", side_effect=ValueError("Frozen transport changed")):
                with self.assertRaisesRegex(ValueError, "Frozen transport changed"):
                    run.development_gate(path, parent)

    def test_missing_failed_or_unreviewed_development_gate_rejected(self):
        parent, gate, path, old, fourth = self.gate_fixture()
        with patch.object(run, "DEVELOPMENT", old), patch.object(run, "DEVELOPMENT_FOURTH", fourth):
            for invalid in ({**gate, "status": "pending"}, {**gate, "banks": gate["banks"][:3]},
                            {**gate, "checks": {}}, {**gate, "limitations": []},
                            {**gate, "audits": {"source_date": gate["audits"]["source_date"], "index": []}}):
                self.put(path, invalid, True)
                with self.assertRaises(ValueError):
                    run.development_gate(path, parent)

    def test_gate_rejects_changed_review_pointer(self):
        parent, gate, path, old, fourth = self.gate_fixture()
        Path(gate["audits"]["index"][0]["path"]).write_text('{"changed":true}')
        with patch.object(run, "DEVELOPMENT", old), patch.object(run, "DEVELOPMENT_FOURTH", fourth):
            with self.assertRaisesRegex(ValueError, "pointer changed"):
                run.development_gate(path, parent)

    def test_no_gate_no_prepare_side_effects(self):
        destination = self.root / "new"
        with patch.object(run.frozen, "manifest", return_value={}), patch.object(run, "development_gate", side_effect=ValueError("not reviewed")):
            with self.assertRaisesRegex(ValueError, "not reviewed"):
                run.prepare(destination, self.root / "missing_gate.json")
        self.assertFalse(destination.exists())

    def test_prepare_copies_only_sixteen_remaining_inputs(self):
        source = self.root / "source";source.mkdir()
        self.put(source / "run_manifest.json", {})
        parent = {"protocol_sha256": "old", "inputs": []}
        for ci in range(10):
            for arm in run.frozen.ARMS:
                pointer = self.put(source / f"conv_{ci}" / arm / "group_inputs.json", {"conv": ci, "arm": arm}, True)
                parent["inputs"].append({"conversation": ci, "arm": arm, "input": pointer})
        gate_path = self.root / "fixture_gate.json";self.put(gate_path, {})
        with patch.object(run, "SOURCE", source), patch.object(run.frozen, "manifest", return_value=parent), patch.object(run, "development_gate", return_value={"banks": [{"reference": True}] * 4}), patch.object(run, "resources", return_value={"fixed": True}), patch.object(run, "offline_acceptance", return_value={"offline_fixture": True}):
            output = run.prepare(self.root / "new", gate_path, source)
        self.assertEqual(output["new_banks"], 16)
        self.assertEqual(len(list((self.root / "new").rglob("group_inputs.json"))), 16)
        self.assertFalse((self.root / "new/conv_0").exists())
        self.assertFalse((self.root / "new/conv_1").exists())
        self.assertFalse(list((self.root / "new").rglob("native")))

    def arm_fixture(self):
        folder = self.root / "run/conv_2/dated_merged"
        self.put(folder / "group_inputs.json", {"fixed": True})
        value = {"protocol_sha256": "test", "remaining_expansion": {"gate_path": "shared.sqlite"}}
        return folder, value

    def test_failed_attempt_not_reentered_and_api_scope_empty(self):
        folder, value = self.arm_fixture();original = run.frozen.ModelAPI
        def writer(*args):
            run.frozen.ModelAPI(folder / "native", retries=2, context={"conversation": 2})
            state = {"status": "failed", "error": {"type": "MockError"}}
            self.put(folder / "write_status.json", state)
            return state
        with patch.object(run, "PacedEmbeddingAPI", return_value=object()) as api, patch.object(run.frozen, "write_arm", side_effect=writer) as call:
            result = run.write_arm_once(self.root / "run", 2, "dated_merged", value)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(api.call_args.kwargs["cache_roots"], ())
            self.assertEqual(api.call_args.kwargs["gate_path"], "shared.sqlite")
            self.assertIs(run.frozen.ModelAPI, original)
            with self.assertRaisesRegex(RuntimeError, "no implicit retry"):
                run.write_arm_once(self.root / "run", 2, "dated_merged", value)
            self.assertEqual(call.call_count, 1)

    def test_unknown_attempt_durable_intent_blocks_and_restores(self):
        folder, value = self.arm_fixture();original = run.frozen.ModelAPI
        with patch.object(run.frozen, "write_arm", side_effect=RuntimeError("unknown")) as call:
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                run.write_arm_once(self.root / "run", 2, "dated_merged", value)
            self.assertIs(run.frozen.ModelAPI, original)
            with self.assertRaisesRegex(RuntimeError, "Prior nontransactional"):
                run.write_arm_once(self.root / "run", 2, "dated_merged", value)
            self.assertEqual(call.call_count, 1)
        self.assertTrue((folder / "write_intent.json").exists())

    def test_complete_bank_resume_only_validates(self):
        folder, value = self.arm_fixture()
        def writer(*args):
            bank = self.put(folder / "bank.json", {"complete": True})
            state = {"status": "complete", "bank": bank}
            self.put(folder / "write_status.json", state)
            return state
        with patch.object(run.frozen, "write_arm", side_effect=writer) as call:
            run.write_arm_once(self.root / "run", 2, "dated_merged", value)
            result = run.write_arm_once(self.root / "run", 2, "dated_merged", value)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(call.call_count, 1)

    def test_development_never_dispatched_and_arms_sequential_in_context(self):
        with self.assertRaises(ValueError):
            run.write_conv(self.root, 1)
        active = []
        @contextlib.contextmanager
        def context(*args):
            active.append(True)
            try:yield
            finally:active.pop()
        calls = []
        def arm(run_dir, ci, method, value):
            self.assertTrue(active)
            calls.append((ci, method))
            return {"status": "complete"}
        value = {"remaining_expansion": {"resources": {"snapshot": "fixed", "files_sha256": {}}}}
        with patch.object(run, "manifest", return_value=value), patch.object(run, "local_bm25_context", side_effect=context), patch.object(run, "write_arm_once", side_effect=arm):
            run.write_conv(self.root, 2)
        self.assertEqual(calls, [(2, arm) for arm in run.frozen.ARMS])
        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()
