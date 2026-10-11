"""Synthetic v2 mapped-input and raw-response summary checks; no real QA/API."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import summarize_mem0_sparse1540_v2 as s
from tests.test_mem0_transfer_qa import FakeBase, facts


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return s.runner.sha(path)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "qa_api"
        self.q = {"id": "c0_q0", "conversation": 0, "category": 1, "question": "A question", "reference": "an answer"}
        self.arm = s.ARMS[0]; self.selected = facts()
        self.binding = {"input_manifest_sha256": "input-map", "source_manifest_sha256": "source",
            "bank_sha256": "bank", "retrieval_sha256": "pool", "selected_sha256": s.runner.fingerprint(self.selected),
            "question_sha256": s.runner.fingerprint(self.q["question"]),
            "reference_sha256": s.runner.fingerprint(self.q["reference"]), "reference_date": "2023-01-01"}
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.build()

    def build(self):
        api = s.transport.StructuredMem0QAAPI(self.base)
        result = error = None
        try:
            result = s.evaluation.answer_and_judge(api, self.q["question"], self.q["reference"], self.selected,
                reference_date=self.binding["reference_date"])
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        self.row = s.runner._sealed({"schema": s.qa.POLICY + "-qa", "conversation": 0, "query_id": self.q["id"],
            "category": 1, "arm": self.arm, "binding": self.binding, "selected_ids": [f["id"] for f in self.selected],
            "result": result, "error": error, "status": "failed" if error else "complete", "api_events": api.events})
        self.ledger = [{"request_hash": e["request_hash"], "stage": e["stage"], "context": e["context"],
            "provider": "gemini", "model": e["model"], "operation": "generateContent",
            "cache_hit": False, "attempt": 1, "status": 200, "usage": {}, "resolved_model": None} for e in api.events]
        if self.base.shared["transport_failure"]:
            self.ledger = [{**r, "status": "transport_error", "error": "Synthetic transport failure", "usage_unknown": True}
                           for r in self.ledger]

    def verify(self):
        with patch.object(s.runner.ModelAPI, "__init__", side_effect=AssertionError("No client")), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
            return s.validate_qa(self.row, self.q, self.arm, self.selected, self.binding, self.folder, self.ledger,
                s.Snapshot(), lambda *args: .5, [self.q["id"]])

    def reseal(self):
        self.row = s.runner._sealed({k: v for k, v in self.row.items() if k != "content_sha256"})

    def test_v2_exact_structured_replay_has_no_global_redirection(self):
        policy, out = s.runner.POLICY, s.runner.OUT
        score, paths = self.verify()
        self.assertEqual(score["official_locomo_f1"], .5)
        self.assertEqual(score["custom_token_f1"], 1.)
        self.assertEqual(len(paths), 2)
        self.assertEqual((policy, out), (s.runner.POLICY, s.runner.OUT))
        self.assertNotEqual(s.runner.POLICY, s.qa.POLICY)

    def test_old_architecture_or_manifest_binding_cannot_be_resealed_into_success(self):
        self.row["schema"] = s.runner.POLICY + "-qa"; self.reseal()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.verify()
        self.row["schema"] = s.qa.POLICY + "-qa"
        self.row["binding"] = {**self.binding, "input_manifest_sha256": "other"}; self.reseal()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.verify()

    def test_raw_response_schema_hash_and_durable_event_are_required(self):
        path = Path(self.row["api_events"][0]["cache_path"])
        raw = json.loads(path.read_bytes()); raw["request"]["body"]["generationConfig"].pop("responseSchema")
        dump(path, raw)
        with self.assertRaises(s.ReplayAuditError):
            self.verify()

    def test_physical_attempt_limit_and_result_metrics_are_replayed(self):
        self.ledger.extend(copy.deepcopy(self.ledger[0]) for _ in range(3))
        with self.assertRaisesRegex(s.ReplayAuditError, "retry budget"):
            self.verify()
        self.ledger = self.ledger[:2]
        self.row["result"]["token_f1"] = .125; self.reseal()
        with self.assertRaisesRegex(ValueError, "metrics"):
            self.verify()

    def test_declared_transport_failure_remains_none(self):
        self.folder = Path(self.tmp.name) / "failed_api"
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.base.shared["transport_failure"] = True; self.build()
        score, paths = self.verify()
        self.assertIsNone(score)
        self.assertIsNone(self.row["result"])
        self.assertEqual(len(paths), 1)

    def test_declared_failure_needs_failed_physical_evidence_not_empty_or200(self):
        self.folder = Path(self.tmp.name) / "proof_api"
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.base.shared["transport_failure"] = True; self.build()
        failed = copy.deepcopy(self.ledger)
        self.ledger = []
        with self.assertRaisesRegex(s.ReplayAuditError, "failed physical evidence"):
            self.verify()
        self.ledger = [{**failed[0], "status": 200, "usage_unknown": False}]
        self.ledger[0].pop("error")
        with self.assertRaisesRegex(s.ReplayAuditError, "failed physical evidence"):
            self.verify()

    def test_cache503_cannot_bypass_physical_failure(self):
        for event in self.ledger:
            event.update(status=503, error="Unavailable")
        self.ledger += [{**event, "cache_hit": True} for event in copy.deepcopy(self.ledger)]
        with self.assertRaisesRegex(s.ReplayAuditError, "Malformed cache consumer"):
            self.verify()
        for event in self.ledger[2:]:
            for field in ("status", "attempt", "error"):
                event.pop(field)
            event.update(latency_ms=0, usage_semantics="reused_response_no_new_model_usage")
        with self.assertRaisesRegex(s.ReplayAuditError, "no successful physical producer"):
            self.verify()

    def test_legitimate_cross_arm_cache_has_one_original_producer_and_own_consumer(self):
        producer = copy.deepcopy(self.ledger)
        self.arm = s.ARMS[1]
        self.base = self.base.scoped(method=self.arm)
        self.build()
        self.ledger = producer + [{"request_hash": e["request_hash"], "stage": e["stage"], "context": e["context"],
            "provider": "gemini", "model": e["model"], "operation": "generateContent", "cache_hit": True,
            "latency_ms": 0, "usage_semantics": "reused_response_no_new_model_usage", "usage": {}, "resolved_model": None}
            for e in self.row["api_events"]]
        score, paths = self.verify()
        self.assertEqual(score["custom_token_f1"], 1.)
        self.assertEqual(len(paths), 2)
        self.ledger = producer
        with self.assertRaisesRegex(s.ReplayAuditError, "attributed successful"):
            self.verify()

    def test_actual_invalid_raw_and_existing_invalid_cache_replay_same_error(self):
        self.folder = Path(self.tmp.name) / "invalid_api"
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.base.shared["reader"]["answer"] = 42; self.build()
        self.assertEqual(self.row["api_events"][0]["cache_source"], "new_http_invalid_response")
        self.assertIsNone(self.verify()[0])
        producer = copy.deepcopy(self.ledger)
        self.arm = s.ARMS[1]; self.base = self.base.scoped(method=self.arm)
        self.build(); self.ledger = producer
        self.assertEqual(self.row["api_events"][0]["cache_source"], "invalid_existing_cache")
        self.assertIsNone(self.verify()[0])

    def test_prior_unknown_guard_has_original_failure_and_no_new_dispatch(self):
        self.folder = Path(self.tmp.name) / "unknown_guard_api"
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.base.shared["transport_failure"] = True; self.build()
        original = copy.deepcopy(self.ledger)
        self.arm = s.ARMS[1]; self.base = self.base.scoped(method=self.arm)
        self.build(); self.ledger = original
        self.assertEqual(self.row["api_events"][0]["cache_source"], "prior_unknown_no_redispatch")
        self.assertIsNone(self.verify()[0])
        self.assertEqual(len(self.base.shared["calls"]), 1)

    def test_raw_usage_metadata_must_equal_producer_ledger(self):
        self.ledger[0]["usage"] = {"promptTokenCount": 99999}
        with self.assertRaisesRegex(s.ReplayAuditError, "usage/model"):
            self.verify()

    def test_foreign_intent_creator_rejected(self):
        path = Path(self.row["api_events"][0]["intent_path"])
        value = json.loads(path.read_bytes()); value["context"]["architecture"] = s.runner.POLICY
        dump(path, value)
        with self.assertRaisesRegex(s.ReplayAuditError, "creator"):
            self.verify()


class CostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ledger = self.root / "new/api_calls.jsonl"
        self.recovery = self.root / "repair_result.json"
        self.entry = {"conversation": 3, "arm": s.ARMS[1], "native_ledger_path": str(self.ledger)}
        self.new = {"stage": "query_embedding", "cache_hit": False, "status": 200,
            "provider": "gemini", "model": "gemini-embedding-001", "usage": {"promptTokenCount": 3},
            "context": {"architecture": s.qa.POLICY, "conversation": 3, "method": s.ARMS[1], "phase": "retrieve"}}

    def set_recovery(self, value):
        self.entry["index_recovery"] = {"path": str(self.recovery), "sha256": dump(self.recovery, value)}

    def ledger_write(self, rows):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_new_cache_and_physical_separate_unknown_not_zero(self):
        self.set_recovery({"inherited_ledger_bytes": 0})
        self.ledger_write([self.new, {**self.new, "cache_hit": True}])
        result = s.recovery_usage(s.Snapshot(), self.entry)
        self.assertEqual((result["physical_attempts"], result["cache_events"]), (1, 1))
        self.assertIsNone(result["total_cost_usd"])

    def test_source_prefix_never_recounted_and_must_match_bytes(self):
        original = self.root / "old.jsonl"
        original.write_text(json.dumps({"cache_hit": False, "old_physical": True}) + "\n", encoding="utf-8")
        self.set_recovery({"inherited_ledger_prefix": {"path": str(original), "sha256": s.runner.sha(original), "bytes": original.stat().st_size}})
        self.ledger_write([json.loads(original.read_bytes()), self.new])
        result = s.recovery_usage(s.Snapshot(), self.entry)
        self.assertEqual(result["physical_attempts"], 1)
        self.ledger_write([self.new])
        with self.assertRaisesRegex(ValueError, "prefix"):
            s.recovery_usage(s.Snapshot(), self.entry)

    def test_undeclared_prefix_unknown_and_unexpected_new_write_rejected(self):
        self.set_recovery({}); self.ledger_write([self.new])
        self.assertFalse(s.recovery_usage(s.Snapshot(), self.entry)["available"])
        self.set_recovery({"inherited_ledger_bytes": 0})
        self.new["context"]["phase"] = "write"; self.ledger_write([self.new])
        with self.assertRaisesRegex(ValueError, "undeclared"):
            s.recovery_usage(s.Snapshot(), self.entry)


class SummaryGateTests(unittest.TestCase):
    def test_complete_input_map_without_qa_reports_all3080_missing_and_preserves_old_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); new = root / "new"; old = root / "old"
            qs = [{"id": "c%d_q%d" % (i // 154, i), "conversation": i // 154, "category": 1,
                   "question": "Question " + str(i), "reference": "answer"} for i in range(1540)]
            source_path = old / "manifest.json"; policy_path = new / "policy.json"
            source = {"source_files": {}}
            manifest = {"source_manifest": {"path": str(source_path), "sha256": dump(source_path, source)},
                "recovery_policy": {"path": str(policy_path), "sha256": dump(policy_path, {"policy": s.qa.POLICY})}, "rows": []}
            loaded = {}
            for ci in range(10):
                dump(old / ("conv_" + str(ci)) / "write_status.json", {"status": "failed" if ci == 3 else "complete"})
                for arm in s.ARMS:
                    folder = root / "inputs" / arm / str(ci)
                    bp = folder / "bank.json"; bank = {"facts": facts()}
                    row = {"conversation": ci, "arm": arm, "bank_path": str(bp), "bank_sha256": dump(bp, bank),
                           "retrieval_dir": str(folder / "retrieval"), "retrieval_files": {}, "index_recovery": None}
                    retrievals = {}
                    for q in [q for q in qs if q["conversation"] == ci]:
                        retrievals[q["id"]] = {"selected": facts()}
                        row["retrieval_files"][q["id"]] = dump(folder / "retrieval" / (q["id"] + ".json"), retrievals[q["id"]])
                    manifest["rows"].append(row)
                    loaded[(ci, arm)] = {"row": row, "bank": bank, "retrievals": retrievals}
            ap = new / "admission.json"
            manifest["input_audit"] = {"path": str(ap), "sha256": dump(ap, {"admitted": True,
                "audited_files": {str(source_path): manifest["source_manifest"]["sha256"]}})}
            digest = dump(new / "input_manifest.json", manifest)
            with patch.object(s.qa, "OUT", new), patch.object(s.runner, "OUT", old), \
                 patch.object(s.qa, "verify_inputs", return_value=(manifest, digest, loaded)), \
                 patch.object(s.runner, "questions", return_value=qs), \
                 patch.object(s.runner, "load_data", return_value=[{}] * 10), \
                 patch.object(s.runner, "session_envelopes", return_value=[{"observation_date": "2023-01-01"}]), \
                 patch.object(s.metric, "verified_official_metric", return_value=(lambda *args: 1., {"synthetic": True})), \
                 patch.object(s, "usage_report", return_value={"synthetic": True}), \
                 patch.object(s.runner.ModelAPI, "__init__", side_effect=AssertionError("No client")):
                report = s.summarize(bootstrap_samples=10)
            self.assertEqual(len(report["outcomes"]), 3080)
            self.assertEqual(len(report["banks"]), 20)
            self.assertFalse(report["all_twenty_banks_and_3080_terminal_slots_verified"])
            self.assertFalse(report["all_3080_answers_complete"])
            self.assertEqual(report["original_v1_write_states_unchanged"]["3"]["status"], "failed")
            for arm in s.ARMS:
                self.assertEqual(report["arms"][arm]["status_counts"]["missing"], 1540)

    def test_unadmitted_map_stops_before_qa_or_metrics(self):
        with patch.object(s.qa, "verify_inputs", side_effect=ValueError("not admitted")), \
             patch.object(s.metric, "verified_official_metric", side_effect=AssertionError("Too early")), \
             patch.object(s.runner.ModelAPI, "__init__", side_effect=AssertionError("No API")):
            with self.assertRaisesRegex(ValueError, "not admitted"):
                s.summarize()

    def test_metric_helper_keeps_missing_failed_invalid_full1540(self):
        qs = [{"id": str(i), "conversation": i % 10, "category": 1} for i in range(1540)]
        slots = {(arm, q["id"]): {**q, "query_id": q["id"], "arm": arm, "status": "missing", "scores": None}
                 for arm in s.ARMS for q in qs}
        slots[(s.ARMS[0], "0")]["status"] = "failed"
        slots[(s.ARMS[1], "1")]["status"] = "invalid"
        arms, paired = s.core.summarize_rows(qs, slots, bootstrap_samples=10)
        for arm in s.ARMS:
            self.assertEqual(arms[arm]["cohort_denominator"], 1540)
            self.assertEqual(sum(arms[arm]["status_counts"].values()), 1540)
            self.assertIsNone(arms[arm]["completed_only_metrics"]["official_locomo_f1"])
            self.assertTrue(arms[arm]["missing_and_failure_zero_sensitivity"]["is_imputation_not_observed_scoring"])


if __name__ == "__main__":
    unittest.main()
