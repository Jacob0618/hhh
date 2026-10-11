"""Synthetic recorded-input checks; no dataset, model client or network."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import audit_pilot400_recorded_qa_v1 as audit
from tests.test_pilot400_qa_audit_v1 import QAAuditTests, dump


class RecordedAuditTests(unittest.TestCase):
    def setUp(self):
        self.f = QAAuditTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.canonical = {f["id"]: f for f in audit.deduplicate_candidates(self.f.bank["facts"])}
        self.sealed = {p.relative_to(self.f.run).as_posix(): audit.driver.file_sha(p)
                       for p in self.f.run.glob("conversations/**/*.json")}

    def validate(self):
        f = self.f
        return audit.validate_recorded_retrieval(f.run, f.q, f.arm, f.raw, f.bank, self.canonical,
                                                 f.query_bound, f.route_bound, sealed=self.sealed)

    def change_retrieval(self, mutation, *, update_terminal_seal=False):
        row = audit.driver.read(self.f.retrieval_path); mutation(row)
        row = audit.driver._seal({k: v for k, v in row.items() if k != "content_sha256"})
        dump(self.f.retrieval_path, row)
        if update_terminal_seal:
            self.sealed[self.f.retrieval_path.relative_to(self.f.run).as_posix()] = audit.driver.file_sha(self.f.retrieval_path)

    def test_recorded_input_replay_is_offline_and_never_rebuilds_rank(self):
        with patch.object(audit.core.ModelAPI, "__init__", side_effect=AssertionError("No live client")), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("No network")), \
             patch.object(audit.driver, "retrieve_rank20", side_effect=AssertionError("No rank recomputation")):
            saved, proof = self.validate()
            proofs = audit.replay_recorded(self.f.run, self.f.row, self.f.q, self.f.raw, saved, proof)
        self.assertEqual(len(proofs), 2)
        self.assertEqual(proof["selected_count"], 20)

    def test_modified_and_resealed_original_retrieval_is_rejected(self):
        self.change_retrieval(lambda r: r["pool"][0]["routing"].update(missing_type_prior=.6))
        with self.assertRaisesRegex(ValueError, "original terminal sealed"):
            self.validate()

    def test_canonical_text_source_vector_and_origin_are_all_checked(self):
        original = audit.driver.read(self.f.retrieval_path)
        mutations = [lambda f: f.update(text="Changed source fact"),
                     lambda f: f["source_records"][0].update(text="Fabricated citation"),
                     lambda f: f["source_observation_dates"].update({next(iter(f["source_observation_dates"])): "2025-01-01"}),
                     lambda f: f["embedding"].__setitem__(0, .2),
                     lambda f: f["origin_ids"].append("other-origin"),
                     lambda f: f["duplicate_variants"][0].update(text="Lost original provenance")]
        for mutation in mutations:
            with self.subTest(mutation=repr(mutation)):
                dump(self.f.retrieval_path, original)
                self.change_retrieval(lambda r: mutation(r["pool"][0]), update_terminal_seal=True)
                with self.assertRaisesRegex(ValueError, "canonical bank content"):
                    self.validate()

    def test_duplicate_pool_and_changed_trace_fail_even_with_valid_new_seal(self):
        original = audit.driver.read(self.f.retrieval_path)
        self.change_retrieval(lambda r: r["pool"].__setitem__(1, copy.deepcopy(r["pool"][0])), update_terminal_seal=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.validate()
        dump(self.f.retrieval_path, original)
        self.change_retrieval(lambda r: r["trace"].update(model_calls_made_by_retrieval=1), update_terminal_seal=True)
        with self.assertRaisesRegex(ValueError, "selection/order/trace"):
            self.validate()

    def test_selected_must_equal_exact_recorded_pool_prefix_with_full_metadata(self):
        self.change_retrieval(lambda r: r["selected"].reverse(), update_terminal_seal=True)
        with self.assertRaisesRegex(ValueError, "selection/order/trace"):
            self.validate()

    def test_query_routing_metadata_and_rrf_inconsistency_rejected(self):
        original = audit.driver.read(self.f.retrieval_path)
        self.change_retrieval(lambda r: r["pool"][0]["routing"].update(query_hash="different-question"), update_terminal_seal=True)
        with self.assertRaisesRegex(ValueError, "channel/type/query"):
            self.validate()
        dump(self.f.retrieval_path, original)
        self.change_retrieval(lambda r: r["pool"][0]["routing"].update(rrf_score=.7), update_terminal_seal=True)
        with self.assertRaisesRegex(ValueError, "RRF"):
            self.validate()

    def test_original_order_is_part_of_actual_http_request_and_raw_result(self):
        saved, proof = self.validate()
        changed = copy.deepcopy(saved)
        changed["selected"][0], changed["selected"][1] = changed["selected"][1], changed["selected"][0]
        changed_row = copy.deepcopy(self.f.row)
        changed_row["selected_ids"] = [f["id"] for f in changed["selected"]]
        with self.assertRaises(FileNotFoundError):
            audit.replay_recorded(self.f.run, changed_row, self.f.q, self.f.raw, changed, proof)

    def test_result_judge_hash_tamper_is_not_accepted(self):
        saved, proof = self.validate(); row = copy.deepcopy(self.f.row)
        row["result"]["judge_input_hash"] = "bad"
        with self.assertRaisesRegex(ValueError, "recorded-order exact"):
            audit.replay_recorded(self.f.run, row, self.f.q, self.f.raw, saved, proof)

    def test_terminal_seal_detects_any_original_byte_change(self):
        p = self.f.path; b = p.read_bytes()
        seal = {"run_dir": str(self.f.run), "files": [{"path": p.relative_to(self.f.run).as_posix(), "bytes": len(b), "sha256": audit.core.sha(b)}]}
        audit.verify_terminal_seal(self.f.run, seal)
        p.write_bytes(b + b" ")
        with self.assertRaisesRegex(ValueError, "terminal artifact bytes changed"):
            audit.verify_terminal_seal(self.f.run, seal)

    def test_typed_recorded_metadata_matches_frozen_writer_without_reranking(self):
        f = self.f; f.arm = audit.driver.ARMS[1]
        from jev_mem.selection import COARSE_CRITERIA, FINE_CRITERIA
        tags = {fact["id"]: {"fact_id": fact["id"], "fact_sha256": audit.driver.fingerprint(fact),
                            "coarse_scores": {k: .4 for k in COARSE_CRITERIA},
                            "fine_scores": {k: .4 for k in FINE_CRITERIA}} for fact in f.bank["facts"]}
        needs = {k: .6 for k in {**COARSE_CRITERIA, **FINE_CRITERIA}}
        f.route_bound = {**f.query_bound, "arm": f.arm, "bank_sha256": f.bank["sha256"], "tag_summary_sha256": "t" * 64}
        f.retrieval_path = f.folder / "retrieval" / f.arm / "mock_q.json"
        result = audit.driver.retrieve_rank20(f.q["question"], f.bank["facts"], [1.] + [0.] * 767, tags=tags, needs=needs)
        f.save_artifact(f.retrieval_path, f.route_bound, result)
        type_path = f.folder / "query_types/mock_q.json"
        f.save_artifact(type_path, f.query_bound, {"scores": needs})
        for path in [f.retrieval_path, type_path]:
            self.sealed[path.relative_to(f.run).as_posix()] = audit.driver.file_sha(path)
        with patch.object(audit.driver, "retrieve_rank20", side_effect=AssertionError("No rank recomputation")):
            saved, _ = self.validate()
        self.assertTrue(saved["trace"]["typed"])
        self.change_retrieval(lambda r: r["pool"][0]["routing"].update(query_type_scores={}), update_terminal_seal=True)
        with self.assertRaisesRegex(ValueError, "channel/type/query"):
            self.validate()

    def test_policy_is_pinned_not_mutable_post_effect_filtering(self):
        path = self.f.run / "fake_policy.json"; dump(path, {"schema": "jev-mem-pilot400-recorded-input-reporting-policy-v1"})
        with self.assertRaisesRegex(ValueError, "fixed reporting policy"):
            audit.verify_policy(path, self.f.run, self.f.manifest)


if __name__ == "__main__": unittest.main()
