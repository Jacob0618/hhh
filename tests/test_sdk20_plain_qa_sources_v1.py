"""Pure input/lineage guards; no credentials, native client or HTTP."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import sdk20_plain_qa_sources_v1 as s
from jev_mem.evaluation_plain_text_v1 import reader_input


def fixture(reference=2022, failed=False):
    facts = [{"id": f"id{i}", "text": f"fact{i}", "observation_date": "2023-05-08",
              "embedding": [999], "never_send_metadata": True} for i in range(20)]
    ids = [f["id"] for f in facts]
    question = {"id": "c0_q1", "question": "When?", "reference": reference, "category": 2}
    ref = {"path": "bank.json", "sha256": "b" * 64}
    old = {"job_id": "baseline_sdk20/c0_q1", "method": "baseline_sdk20", "conversation": 0,
           "query_id": "c0_q1", "question": "When?", "reference": reference, "category": 2,
           "reference_date": "2023-10-22", "selected_ids": ids,
           "status": "blocked_original_reader" if failed else "ready",
           "answer": None if failed else "NEVER COPY OLD ANSWER", "old_judge": {"correct": True},
           "F1": {"official_locomo_f1": 1}, "pending_original_list": ["OLD ARRAY"] if failed else None}
    payload = reader_input("When?", facts, reference_date=old["reference_date"])
    old["source"] = {"reader_signature": {"body": {"contents": [{"parts": [{"text": json.dumps(payload)}]}]}}}
    selected = {k: copy.deepcopy(old[k]) for k in ("method", "conversation", "query_id", "question", "reference", "category", "selected_ids")}
    selected.update(status="complete", bank_sha256=ref["sha256"])
    return old, {"facts": facts}, ref, selected, {"path": "selection.json", "sha256": "s" * 64}, question


class SourcesTests(unittest.TestCase):
    def setUp(self):
        for name in ("requests.Session", "requests.post", "socket.create_connection", "jev_mem.providers.load_credentials"):
            p = patch(name, side_effect=AssertionError("No API/credentials")); p.start(); self.addCleanup(p.stop)

    def project(self, values):
        return s.job_from_selection(*values, lineage={"retrieval": {"path": "r", "sha256": "r" * 64}})

    def test_fixed_order_exact_input_numeric_gold_and_no_old_answer(self):
        data = fixture(); old = copy.deepcopy(data)
        job = self.project(data)
        self.assertEqual(job["selected_ids"], data[0]["selected_ids"])
        self.assertEqual([f["id"] for f in job["selected_facts"]], job["selected_ids"])
        self.assertTrue(all(set(f) == {"id", "text", "observation_date"} for f in job["selected_facts"]))
        self.assertIs(type(job["reference"]), int)
        self.assertFalse({"answer", "old_judge", "F1", "pending_original_list", "original_reader_raw"} & set(job))
        self.assertNotIn("NEVER COPY OLD ANSWER", json.dumps(job))
        self.assertEqual(data, old)

    def test_old_failed_reader_is_fresh_ready_without_normalizing_array(self):
        job = self.project(fixture(failed=True))
        self.assertEqual(job["status"], "ready")
        self.assertNotIn("OLD ARRAY", json.dumps(job))
        self.assertEqual(len(job["selected_facts"]), 20)

    def test_selection_order_or_budget_or_bank_mismatch_rejected(self):
        for change in ("order", "missing", "duplicate", "bank", "failed"):
            data = fixture(); selection = data[3]
            if change == "order": selection["selected_ids"] = list(reversed(selection["selected_ids"]))
            if change == "missing": selection["selected_ids"] = selection["selected_ids"][:-1]
            if change == "duplicate": selection["selected_ids"][0] = selection["selected_ids"][1]
            if change == "bank": selection["bank_sha256"] = "x" * 64
            if change == "failed": selection["status"] = "failed"
            with self.subTest(change=change), self.assertRaises(ValueError): self.project(data)

    def test_question_gold_or_fact_date_changes_rejected(self):
        for change in ("question", "reference", "category", "fact", "date", "reference_date", "bool"):
            data = fixture()
            if change in ("question", "reference", "category"): data[5][change] = "changed"
            if change == "fact": data[1]["facts"][0]["text"] = "changed"
            if change == "date": data[1]["facts"][0]["observation_date"] = None
            if change == "reference_date": data[0]["reference_date"] = "2000-01-01"
            if change == "bool":
                data = fixture(reference=True)
            with self.subTest(change=change), self.assertRaises(ValueError): self.project(data)

    def test_bound_inputs_first_bad_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "input.json"; p.write_text('{"value":1}', encoding="utf8")
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            loader = s.BoundInputs({"input_hashes": {str(p.resolve()): h}})
            self.assertEqual(loader.get(p), {"value": 1})
            with self.assertRaises(ValueError): loader.get(p, expected="bad")
            p.write_text('{"value":2}', encoding="utf8")
            with self.assertRaises(ValueError): s.BoundInputs({"input_hashes": {str(p.resolve()): h}}).get(p)

    def test_fresh_direct_namespace_and_offline_prepare_validate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "runs").mkdir()
            payload = {"schema": s.SCHEMA, "policy": s.POLICY, "jobs": [self.project(fixture())],
                       "old_failure_slots_included": {"baseline_sdk20": 15, "fact_types_sdk20": 23}}
            with patch.object(s, "ROOT", root), patch.object(s, "build_source_payload", return_value=payload):
                out = s.prepare(root / "runs" / "new"); path = Path(out["source_manifest"]["path"])
                self.assertEqual(s.validate_rebuild(path)["jobs"], payload["jobs"])
                with self.assertRaises(ValueError): s.prepare(root / "runs" / "new")
                with self.assertRaises(ValueError): s.prepare(root / "runs" / "new" / "nested")
                changed = {**payload, "jobs": []}
                with patch.object(s, "build_source_payload", return_value=changed):
                    with self.assertRaises(ValueError): s.validate_rebuild(path)

    def test_fast_validator_rejects_changed_header_without_historical_replay(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "source.json"
            s.save_new(path, {"schema": s.SCHEMA, "policy": s.POLICY, "phase_kind": "wrong"})
            with patch.object(s.historical, "validate", side_effect=AssertionError("No old Reader replay")), \
                 patch.object(s, "build_source_payload", side_effect=AssertionError("No complete rebuild")):
                with self.assertRaisesRegex(ValueError, "protocol/code"):
                    s.validate(path)

    def test_fast_validator_fixed_parent_hash_before_parsing_old_qa(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = root / "parent.json"; parent.write_text('{}', encoding='utf8')
            path = root / "source.json"
            s.save_new(path, {"schema": s.SCHEMA, "policy": s.POLICY,
                "phase_kind": "fresh_plain_reader_and_judge", "code_sha256": {},
                "source_manifest": {"path": str(parent), "sha256": "wrong"}})
            with patch.object(s, "SOURCE", parent), patch.object(s, "SOURCE_SHA", "wrong"), \
                 patch.object(s, "CODE", ()), \
                 patch.object(s.historical, "validate", side_effect=AssertionError("No old Reader replay")), \
                 patch.object(s, "build_source_payload", side_effect=AssertionError("No complete rebuild")):
                with self.assertRaisesRegex(ValueError, "source manifest changed"):
                    s.validate(path)


if __name__ == "__main__":
    unittest.main()
