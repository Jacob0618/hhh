"""Offline mocks for strict pilot400 completion, missingness and pairing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, queries
from jev_mem.evaluation import lexical_metrics
from jev_mem.evaluation_pilot import prepare_reader_input
from jev_mem.pilot400 import reader_fact
from jev_mem.selection import deduplicate_candidates
from jev_mem.topic_grouping import raw_turns_from_locomo
from tools.summarize_pilot400 import (ARMS, METRICS, READER_PROTOCOL, SCHEMA,
    ArtifactInvalid, expected_bank_path, paired_comparison, summarize_pilot,
    validate_artifact, verified_official_metric)


class Pilot400SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "configs/pilot400_manifest_v1.json").read_text("utf-8"))
        data = load_data()
        population = {}
        for ci, conv in enumerate(data):
            reference_date = max(t["observation_date"] for t in raw_turns_from_locomo(conv["conversation"]))
            for q in queries(conv, ci):
                population[q["id"]] = {**q, "conversation": ci, "reference_date": reference_date}
        cls.expected = {qid: population[qid] for qid in cls.manifest["query_ids"]}
        cls.cohort_hash = cls.manifest["sample"]["query_ids_sha256"]
        cls.official_f1 = staticmethod(verified_official_metric()[0])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pilot400-offline-test-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)

    def artifact(self, qid=None, arm=ARMS[0], answer=None):
        q = self.expected[qid or self.manifest["query_ids"][0]]
        relative = expected_bank_path(q["conversation"], arm)
        path = self.run / relative
        facts = []
        for i in range(25):
            turn = {"source_turn_id": f"turn-{i}", "session_id": "session_1", "speaker": "Fixture",
                    "observation_date": "2023-01-01", "text": f"Fixture fact {i}"}
            facts.append({"id": f"fact-{i}", "text": f"Fixture fact {i}",
                          "observation_date": "2023-01-01", "observation_dates": ["2023-01-01"],
                          "source_observation_dates": {turn["source_turn_id"]: "2023-01-01"},
                          "temporal_status": "not_temporal", "temporal_status_verified": False,
                          "support_status": "proposed_unverified", "source_records": [turn],
                          "proposed_support": [turn["source_turn_id"]],
                          "embedding": [1.] + [0.] * 767})
        raw = (json.dumps({"schema": "date-aware-raw-group-writer-v1", "facts": facts}, indent=2) + "\n").encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
        answer = str(q["reference"]) if answer is None else answer
        prepared = prepare_reader_input(q["question"], facts[:20], reference_date=q["reference_date"])
        return {
            "schema": SCHEMA, "status": "complete", "arm": arm,
            "query_id": q["id"], "conversation": q["conversation"], "category": q["category"],
            "cohort_sha256": self.cohort_hash, "bank_path": relative,
            "bank_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_ids": [f["id"] for f in facts[:20]],
            "result": {"answer": answer, "judge": {"correct": True, "completeness": 1., "reason": "Mock only"},
                       "token_f1": lexical_metrics(answer, q["reference"])["token_f1"],
                       "selected_count": 20, "reader_input": prepared["payload"],
                       "input_hash": prepared["input_hash"], "reader_request_hash": prepared["reader_request_hash"],
                       "reader_protocol": READER_PROTOCOL, "reader_model": "gemini-2.5-flash",
                       "judge_model": "gemini-3.5-flash-lite", "used_fact_ids": [facts[0]["id"]]},
        }

    def save(self, artifact, *, file_qid=None, file_arm=None):
        path = self.run / f"conversations/conv_{artifact['conversation']}/qa/{file_arm or artifact['arm']}/{file_qid or artifact['query_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact), encoding="utf-8")

    def validate(self, row, expected_qid=None, arm=ARMS[0]):
        return validate_artifact(row, self.expected[expected_qid or self.manifest["query_ids"][0]], arm,
                                 self.run, self.cohort_hash, {}, self.official_f1)

    def test_valid_artifact_recomputes_both_distinct_metrics(self):
        row = self.artifact()
        status, scores = self.validate(row)
        self.assertEqual(status, "complete")
        self.assertEqual(scores["custom_token_f1"], 1.)
        self.assertEqual(scores["official_locomo_f1"], 1.)
        self.assertEqual(scores["judge_correct"], 1.)
        _, provenance = verified_official_metric()
        self.assertEqual(provenance["source_commit"], "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc")

    def test_wrong_id_conversation_category_cohort_and_arm_are_not_complete(self):
        correct = self.artifact()
        for key, bad in (("query_id", "c9_q9999"), ("conversation", 9), ("category", 5),
                         ("cohort_sha256", "0" * 64), ("arm", ARMS[1]), ("conversation", False)):
            row = copy.deepcopy(correct)
            row[key] = bad
            with self.subTest(key=key), self.assertRaises(ArtifactInvalid):
                self.validate(row)

    def test_false_complete_schema_and_old_reader_are_rejected(self):
        correct = self.artifact()
        variants = []
        row = copy.deepcopy(correct); row["result"] = {}; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["answer"] = ["answer"]; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["judge"]["correct"] = 1; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["judge"]["completeness"] = float("nan"); variants.append(row)
        row = copy.deepcopy(correct); row["result"]["token_f1"] = .5; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["reader_protocol"] = "brief_answer_local_evidence_ids_v2"; variants.append(row)
        row = copy.deepcopy(correct); row["status"] = "success"; variants.append(row)
        row = copy.deepcopy(correct); row["error"] = {"message": "failed"}; variants.append(row)
        for i, row in enumerate(variants):
            with self.subTest(variant=i), self.assertRaises(ArtifactInvalid):
                self.validate(row)

    def test_actual_bank_hash_and_exact_twenty_unique_members_required(self):
        correct = self.artifact()
        variants = []
        row = copy.deepcopy(correct); row["bank_sha256"] = "0" * 64; variants.append(row)
        row = copy.deepcopy(correct); row["bank_path"] = "../outside.json"; variants.append(row)
        row = copy.deepcopy(correct); row["selected_ids"] = row["selected_ids"][:19]; variants.append(row)
        row = copy.deepcopy(correct); row["selected_ids"][1] = row["selected_ids"][0]; variants.append(row)
        row = copy.deepcopy(correct); row["selected_ids"][0] = "missing-fact"; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["used_fact_ids"] = ["fact-24"]; variants.append(row)
        for i, row in enumerate(variants):
            with self.subTest(variant=i), self.assertRaises(ArtifactInvalid):
                self.validate(row)

    def test_reader_audit_mismatch_never_counts_complete(self):
        correct = self.artifact()
        variants = []
        row = copy.deepcopy(correct); row["result"]["selected_count"] = 0; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["selected_count"] = 20.; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["reader_input"]["question"] = "Wrong question"; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["reader_input"]["facts"] = []; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["reader_input"]["conversation_reference_date"] = "2000-01-01"; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["input_hash"] = "0" * 64; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["reader_request_hash"] = "0" * 64; variants.append(row)
        row = copy.deepcopy(correct); row["result"]["used_fact_ids"] = ["fact-0", "fact-0"]; variants.append(row)
        row = copy.deepcopy(correct); del row["result"]["used_fact_ids"]; variants.append(row)
        for i, row in enumerate(variants):
            with self.subTest(variant=i), self.assertRaises(ArtifactInvalid):
                self.validate(row)
        broken = copy.deepcopy(correct)
        broken["result"]["reader_input"]["question"] = "Wrong question"
        self.save(broken)
        report = summarize_pilot(self.run, bootstrap_samples=10)
        self.assertEqual(report["arms"][ARMS[0]]["status_counts"], {"complete": 0, "failed": 1, "missing": 399})
        self.assertEqual(report["arms"][ARMS[0]]["failures"][0]["kind"], "protocol_invalid")

    def test_duplicate_source_dates_are_reconstructed_from_full_bank(self):
        row = self.artifact()
        bank_path = self.run / row["bank_path"]
        bank = json.loads(bank_path.read_bytes())
        bank["facts"][0]["text"] = "Fixture caf\u00e9 plan"
        duplicate = copy.deepcopy(bank["facts"][0])
        duplicate.update(id="fact-duplicate", text="Fixture cafe\u0301   plan",
                         observation_date="2023-02-01", observation_dates=["2023-02-01"],
                         source_observation_dates={"extra-turn": "2023-02-01"})
        bank["facts"].append(duplicate)
        raw = json.dumps(bank).encode()
        bank_path.write_bytes(raw)
        row["bank_sha256"] = hashlib.sha256(raw).hexdigest()
        canonical = {f["id"]: f for f in deduplicate_candidates(bank["facts"])}
        facts = [reader_fact(canonical[fid]) for fid in row["selected_ids"]]
        q = self.expected[row["query_id"]]
        prepared = prepare_reader_input(q["question"], facts, reference_date=q["reference_date"])
        row["result"].update(reader_input=prepared["payload"], input_hash=prepared["input_hash"],
                             reader_request_hash=prepared["reader_request_hash"])
        self.assertEqual(self.validate(row)[0], "complete")
        merged = row["result"]["reader_input"]["facts"][0]
        self.assertEqual(merged["observation_dates"], ["2023-01-01", "2023-02-01"])
        self.assertIsNone(merged["observation_date"])
        # Even self-consistent hashes for the truncated provenance are rejected:
        # reconstruction is anchored to the full source bank, not saved payload.
        original_only = [copy.deepcopy(canonical[fid]) for fid in row["selected_ids"]]
        for f in original_only:
            f.pop("duplicate_variants", None)
        tampered = prepare_reader_input(q["question"], original_only, reference_date=q["reference_date"])
        row["result"].update(reader_input=tampered["payload"], input_hash=tampered["input_hash"],
                             reader_request_hash=tampered["reader_request_hash"])
        with self.assertRaises(ArtifactInvalid):
            self.validate(row)

    def test_complete_failed_missing_and_zero_imputation_are_separate(self):
        qids = self.manifest["query_ids"]
        self.save(self.artifact(qids[0]))
        failed = self.artifact(qids[1])
        failed["status"] = "failed"
        failed.pop("result")
        failed["error"] = {"type": "Timeout", "stage": "reader", "message": "Mock timeout"}
        self.save(failed)
        false_complete = self.artifact(qids[2])
        false_complete["query_id"] = "c99_q99"
        self.save(false_complete, file_qid=qids[2])
        report = summarize_pilot(self.run, bootstrap_samples=10)
        arm = report["arms"][ARMS[0]]
        self.assertEqual(arm["status_counts"], {"complete": 1, "failed": 2, "missing": 397})
        self.assertEqual(arm["conditional_metrics"]["means"]["custom_token_f1"], 1.)
        self.assertIsNone(arm["full_cohort_observed_metrics"]["custom_token_f1"])
        self.assertEqual(arm["zero_filled_sensitivity"]["means"]["custom_token_f1"], 1 / 400)
        self.assertTrue(arm["zero_filled_sensitivity"]["is_imputation_not_observed_scoring"])
        self.assertEqual(arm["zero_filled_sensitivity"]["imputed_zero_questions"], 399)
        self.assertFalse(report["all_expected_outputs_valid_and_complete"])
        self.assertEqual(report["arms"][ARMS[1]]["status_counts"], {"complete": 0, "failed": 0, "missing": 400})

    def test_failed_artifact_cannot_hide_scored_result(self):
        row = self.artifact()
        row["status"] = "failed"
        row["error"] = {"type": "Mock", "stage": "reader", "message": "Mock error"}
        with self.assertRaises(ArtifactInvalid):
            self.validate(row)
        row.pop("result")
        row.pop("bank_sha256")
        row.pop("bank_path")
        self.assertEqual(self.validate(row), ("failed", None))

    def test_out_of_cohort_artifact_never_increases_denominator(self):
        extra = self.artifact()
        extra["query_id"] = "c0_q999999"
        self.save(extra)
        report = summarize_pilot(self.run, bootstrap_samples=10)
        self.assertEqual(len(report["unexpected_artifact_paths"]), 1)
        self.assertEqual(report["arms"][ARMS[0]]["status_counts"]["missing"], 400)

    def test_paired_comparison_uses_common_success_not_independent_means(self):
        def record(qid, ci, value):
            return {"query_id": qid, "conversation": ci, **{m: value for m in METRICS}}
        baseline = {"q1": record("q1", 0, .2), "q2": record("q2", 0, 1.)}
        other = {"q1": record("q1", 0, .5), "q3": record("q3", 0, 0.)}
        pair = paired_comparison(baseline, other, samples=100, seed=0)
        self.assertEqual(pair["common_success_count"], 1)
        self.assertEqual(pair["paired_coverage"], 1 / 400)
        self.assertAlmostEqual(pair["metrics"]["official_locomo_f1"]["other_minus_baseline"], .3)
        self.assertIsNone(pair["metrics"]["official_locomo_f1"]["cluster_bootstrap_95_percentile_interval"])

    def test_ten_cluster_bootstrap_is_deterministic_and_requires_all_clusters(self):
        base, other = {}, {}
        for ci in range(10):
            for qi in range(2):
                qid = f"c{ci}_q{qi}"
                base[qid] = {"conversation": ci, **{m: .2 for m in METRICS}}
                other[qid] = {"conversation": ci, **{m: .2 + ci / 20 for m in METRICS}}
        first = paired_comparison(base, other, samples=100, seed=4)
        self.assertEqual(first, paired_comparison(base, other, samples=100, seed=4))
        self.assertTrue(first["bootstrap"]["eligible"])
        self.assertIsNotNone(first["metrics"]["official_locomo_f1"]["cluster_bootstrap_95_percentile_interval"])
        del other["c9_q1"]
        self.assertFalse(paired_comparison(base, other, samples=100, seed=4)["bootstrap"]["eligible"])

    def test_all_expected_complete_requires_1200_valid_artifacts(self):
        for arm in ARMS:
            for qid in self.manifest["query_ids"]:
                self.save(self.artifact(qid, arm))
        report = summarize_pilot(self.run, bootstrap_samples=100)
        self.assertTrue(report["all_expected_outputs_valid_and_complete"])
        for arm in report["arms"].values():
            self.assertEqual(arm["status_counts"], {"complete": 400, "failed": 0, "missing": 0})
            self.assertEqual(arm["full_cohort_observed_metrics"]["custom_token_f1"], 1.)
        for pair in report["paired_differences"].values():
            self.assertEqual(pair["common_success_count"], 400)
            self.assertEqual(pair["metrics"]["official_locomo_f1"]["other_minus_baseline"], 0.)
        self.assertEqual(len(report["verified_banks"]), 20)


if __name__ == "__main__":
    unittest.main()
