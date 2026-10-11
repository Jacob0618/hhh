"""Synthetic offline summary gates; no experiment QA, credentials or APIs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import summarize_mem0_sparse1540 as s
from tests.test_mem0_transfer_qa import FakeBase, facts


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class QAIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "qa_api"
        self.q = {"id": "c0_q1", "conversation": 0, "category": 1, "question": "Synthetic question", "reference": "an answer"}
        self.arm = s.ARMS[0]
        self.selected = facts()
        self.binding = {"manifest_sha256": "manifest", "bank_sha256": "bank", "retrieval_sha256": "retrieval",
                        "selected_sha256": s.runner.fingerprint(self.selected), "question_sha256": s.runner.fingerprint(self.q["question"]),
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
        self.row = s.runner._sealed({"schema": s.runner.POLICY + "-qa", "query_id": self.q["id"], "conversation": 0,
            "category": 1, "arm": self.arm, "binding": self.binding, "selected_ids": [f["id"] for f in self.selected],
            "api_events": api.events, "result": result, "error": error, "status": "complete" if result else "failed"})
        self.ledger = [{"request_hash": e["request_hash"], "context": e["context"], "stage": e["stage"],
                        "provider": "gemini", "model": e["model"], "operation": "generateContent", "attempt": 1,
                        "status": 200, "cache_hit": e["cache_source"] == "validated_cache"} for e in api.events]

    def verify(self, row=None, snapshot=None):
        return s.validate_qa(row or self.row, self.q, self.arm, self.selected, self.binding, self.folder,
                             self.ledger, snapshot or s.Snapshot(), lambda *args: .75, [self.q["id"]])

    def test_success_exact_replay_without_live_clients(self):
        with patch.object(s.runner.ModelAPI, "__init__", side_effect=AssertionError("No live client")), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
            scores, paths = self.verify()
        self.assertEqual(scores["official_locomo_f1"], .75)
        self.assertEqual(scores["custom_token_f1"], 1.)
        self.assertEqual(len(paths), 2)

    def test_resealed_wrong_reference_binding_rejected(self):
        changed = copy.deepcopy(self.row)
        changed["binding"]["reference_sha256"] = "other"
        changed = s.runner._sealed({k: v for k, v in changed.items() if k != "content_sha256"})
        with self.assertRaisesRegex(ValueError, "identity/input"):
            self.verify(changed)

    def test_http200_and_valid_json_do_not_hide_wrong_raw_body(self):
        p = Path(self.row["api_events"][0]["cache_path"])
        cached = json.loads(p.read_bytes())
        cached["request"]["body"]["generationConfig"].pop("responseSchema")
        dump(p, cached)
        with self.assertRaises(s.ReplayAuditError):
            self.verify()

    def test_durable_consumer_pointer_must_match(self):
        event = self.row["api_events"][0]
        disk = json.loads(Path(event["event_path"]).read_bytes())
        disk["context"]["method"] = s.ARMS[1]
        dump(Path(event["event_path"]), disk)
        with self.assertRaisesRegex(s.ReplayAuditError, "pointer/body"):
            self.verify()

    def test_intent_retry_budget_cannot_change(self):
        p = Path(self.row["api_events"][0]["intent_path"])
        value = json.loads(p.read_bytes()); value["max_http_attempts"] = 4; dump(p, value)
        with self.assertRaisesRegex(s.ReplayAuditError, "intent differs"):
            self.verify()

    def test_missing_success_event_and_extra_request_rejected(self):
        self.ledger[0]["status"] = 403
        with self.assertRaisesRegex(s.ReplayAuditError, "attributed successful"):
            self.verify()
        self.ledger[0]["status"] = 200
        self.ledger.append({**self.ledger[0], "request_hash": "not-the-request"})
        with self.assertRaisesRegex(ValueError, "outside its durable"):
            self.verify()

    def test_four_physical_attempts_rejected(self):
        self.ledger.extend(copy.deepcopy(self.ledger[0]) for _ in range(3))
        with self.assertRaisesRegex(s.ReplayAuditError, "retry budget"):
            self.verify()

    def test_result_tamper_resealed_still_fails_replay(self):
        row = copy.deepcopy(self.row); row["result"]["judgment"]["correct"] = False
        row = s.runner._sealed({k: v for k, v in row.items() if k != "content_sha256"})
        with self.assertRaisesRegex(ValueError, "Result/metrics"):
            self.verify(row)

    def test_native_invalid_local_id_remains_declared_failure(self):
        # A new synthetic namespace/request, so no original cache gets overwritten.
        self.folder = Path(self.temp.name) / "bad_id_api"
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.base.shared["reader"]["used_fact_ids"] = ["nonexistent_local_id"]
        self.build()
        self.assertEqual(self.row["status"], "failed")
        scores, paths = self.verify()
        self.assertIsNone(scores)
        self.assertEqual(len(paths), 1)

    def test_unknown_transport_preserved_without_fabricated_score(self):
        self.folder = Path(self.temp.name) / "unknown_api"
        self.base = FakeBase(self.folder, context=s.scoped_context(0, self.arm, self.q["id"]))
        self.base.shared["transport_failure"] = True
        self.build()
        self.ledger[0].update(status="transport_error", usage_unknown=True)
        scored, _ = self.verify()
        self.assertIsNone(scored)
        self.assertFalse(Path(self.row["api_events"][0]["cache_path"]).exists())

    def test_audit_error_cannot_masquerade_as_declared_failure(self):
        self.row.update(status="failed", result=None, error={"type": "ValueError", "message": "fake"})
        self.row = s.runner._sealed({k: v for k, v in self.row.items() if k != "content_sha256"})
        Path(self.row["api_events"][0]["event_path"]).unlink()
        with self.assertRaises(s.ReplayAuditError):
            self.verify()

    def test_same_request_cache_can_have_distinct_arm_consumers(self):
        original_physical = copy.deepcopy(self.ledger)
        self.arm = s.ARMS[1]
        self.base = self.base.scoped(method=self.arm)
        self.build()
        self.ledger = original_physical + self.ledger
        self.assertTrue(all(e["cache_source"] == "validated_cache" for e in self.row["api_events"]))
        scores, _ = self.verify()
        self.assertEqual(scores["judge_correct"], 1.)

    def test_intent_creator_must_own_physical_dispatch(self):
        self.ledger[0]["context"] = {**self.ledger[0]["context"], "method": s.ARMS[1]}
        with self.assertRaisesRegex(s.ReplayAuditError, "retry budget"):
            self.verify()


class MetricAndCostTests(unittest.TestCase):
    def make(self, count=1540):
        qs = [{"id": "q" + str(i), "conversation": i % 10, "category": 1} for i in range(count)]
        slots = {(arm, q["id"]): {**q, "query_id": q["id"], "status": "missing", "scores": None} for arm in s.ARMS for q in qs}
        return qs, slots

    def test_fixed1540_separates_missing_failure_invalid_and_completed_only(self):
        qs, slots = self.make()
        for arm in s.ARMS:
            slots[(arm, "q0")].update(status="complete", scores={"query_id": "q0", "conversation": 0, "category": 1, **{m: .5 for m in s.METRICS}})
            slots[(arm, "q1")]["status"] = "failed"
            slots[(arm, "q2")]["status"] = "invalid"
        arms, paired = s.summarize_rows(qs, slots, bootstrap_samples=20)
        row = arms[s.ARMS[0]]
        self.assertEqual(row["status_counts"], {"complete": 1, "failed": 1, "missing": 1537, "invalid": 1})
        self.assertEqual(row["completed_only_metrics"][s.METRICS[0]], .5)
        self.assertAlmostEqual(row["missing_and_failure_zero_sensitivity"]["means"][s.METRICS[0]], .5 / 1540)
        self.assertIsNone(row["full_cohort_observed_metrics"][s.METRICS[0]])
        self.assertFalse(paired["bootstrap"]["eligible"])

    def test_paired_bootstrap_uses_ten_clusters_seed0(self):
        qs, slots = self.make(20)
        for a, arm in enumerate(s.ARMS):
            for q in qs:
                slots[(arm, q["id"])].update(status="complete", scores={"query_id": q["id"], "conversation": q["conversation"],
                    "category": 1, **{m: .2 + .1 * a for m in s.METRICS}})
        _, first = s.summarize_rows(qs, slots, bootstrap_samples=100)
        _, second = s.summarize_rows(qs, slots, bootstrap_samples=100)
        self.assertEqual(first, second)
        self.assertTrue(first["bootstrap"]["eligible"])
        self.assertEqual(first["bootstrap"]["seed"], 0)
        self.assertAlmostEqual(first["metrics"][s.METRICS[0]]["other_minus_baseline"], .1)

    def test_cost_counts_physical_retries_once_and_cache_zero_with_thought_tokens(self):
        row = {"request_hash": "same", "model": "gemini-2.5-flash", "operation": "generateContent", "status": 200,
               "cache_hit": False, "usage": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 10}}
        result = s.measure_usage([row, row, {**row, "cache_hit": True}])
        self.assertEqual(result["physical_attempts"], 2)
        self.assertEqual(result["cache_events"], 1)
        self.assertEqual(result["recorded_thinking_tokens"], 20)
        self.assertAlmostEqual(result["total_cost_usd"], 2 * (100 * .3 + 30 * 2.5) / 1e6)

    def test_embedding_unknown_and_missing_ledger_are_not_zero(self):
        rows = [{"model": "gemini-embedding-001", "operation": "batchEmbedContents", "status": 200, "usage": {}},
                {"model": "gemini-2.5-flash", "status": "transport_error", "usage_unknown": True, "usage": {}}]
        result = s.measure_usage(rows)
        self.assertEqual(result["unknown_cost_attempts"], 2)
        self.assertIsNone(result["total_cost_usd"])
        self.assertFalse(s.measure_usage(None)["available"])

    def test_snapshot_rejects_mid_audit_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "a.json"; dump(p, {"a": 1})
            snap = s.Snapshot(); snap.read(p); dump(p, {"a": 2})
            with self.assertRaisesRegex(ValueError, "changed during"):
                snap.read(p)
            self.assertEqual(snap.unchanged(), [str(p.resolve())])

    def test_unadmitted_native_retrieval_prevents_metric_or_qa_reads(self):
        from tools import audit_mem0_sparse1540_retrieval as gate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); qs, _ = self.make()
            manifest = {"arms": list(s.ARMS), "question_ids": [q["id"] for q in qs]}
            dump(root / "manifest.json", manifest)
            with patch.object(s.runner, "OUT", root), patch.object(s.runner, "verify_manifest", return_value=manifest), \
                 patch.object(s.runner, "questions", return_value=qs), patch.object(gate, "audit", return_value={"admitted": False}), \
                 patch.object(s.runner, "load_data", side_effect=AssertionError("No QA on failed admission")):
                with self.assertRaisesRegex(ValueError, "admission has not passed"):
                    s.summarize(root)


if __name__ == "__main__":
    unittest.main()
