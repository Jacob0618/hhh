"""Synthetic-only QA sealing/raw replay checks; no dataset or network access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_pilot400_qa_v1 as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class Capture(audit.CacheOnlyAPI):
    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
        answer = ({"answer": "Fixture", "used_fact_ids": ["f1"]} if stage == audit.evaluation.READER_STAGE
                  else {"correct": True, "completeness": 1.0, "reason": "Synthetic only"})
        response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(answer)}]}}]}
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        digest = audit.sha(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode())
        dump(self.cache_dir / (digest + ".json"), {"request": signature, "response": response, "stage": stage})
        return response


class QAAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.q = {"id": "mock_q", "conversation": 0, "category": 1, "question": "Who owns a dog?", "reference": "Fixture"}
        self.manifest = {"protocol_sha256": "p" * 64}
        self.raw, facts = [], []
        for i in range(25):
            turn = {"source_turn_id": f"D1:{i}", "session_id": "s1", "speaker": "A",
                    "observation_date": "2023-01-01", "text": f"Fact {i}"}
            self.raw.append(turn)
            facts.append({"id": f"fact-{i}", "text": f"A owns dog number {i}.",
                "observation_date": "2023-01-01", "observation_dates": ["2023-01-01"],
                "source_observation_dates": {turn["source_turn_id"]: "2023-01-01"},
                "temporal_status": "not_temporal", "temporal_status_verified": False,
                "support_status": "proposed_unverified", "source_records": [turn],
                "proposed_support": [turn["source_turn_id"]], "embedding": [1.] + [0.] * 767})
        self.bank = {"facts": facts, "sha256": "b" * 64}
        self.arm = audit.driver.ARMS[0]
        self.folder = audit.driver.conv_dir(self.run, 0)
        self.path = self.folder / "qa" / self.arm / "mock_q.json"
        self.query_bound = audit.driver.binding(self.manifest, 0, self.raw, query_id="mock_q",
                                                 question_sha256=audit.driver.fingerprint(self.q["question"]))
        self.route_bound = {**self.query_bound, "arm": self.arm, "bank_sha256": self.bank["sha256"], "tag_summary_sha256": None}
        self.bound = {**self.route_bound, "reference_sha256": audit.driver.fingerprint(self.q["reference"])}
        vector = [1.] + [0.] * 767
        self.save_artifact(self.folder / "query_embeddings/mock_q.json", self.query_bound, {"vector": vector})
        retrieved = audit.driver.retrieve_rank20(self.q["question"], facts, vector)
        self.retrieval_path = self.folder / "retrieval" / self.arm / "mock_q.json"
        self.save_artifact(self.retrieval_path, self.route_bound, retrieved)
        result = audit.evaluation.answer_and_judge(Capture(self.run / "api_cache"), self.q["question"], self.q["reference"],
                                                   retrieved["selected"], reference_date="2023-01-01")
        self.row = {"schema": audit.driver.QA_SCHEMA, "status": "complete", "error": None,
                    "arm": self.arm, "query_id": "mock_q", "conversation": 0, "category": 1,
                    "cohort_sha256": audit.driver.COHORT_SHA256, "protocol_sha256": self.manifest["protocol_sha256"],
                    "bank_path": "conversations/conv_0/banks/session/group_fact_bank.json", "bank_sha256": self.bank["sha256"],
                    "selected_ids": [f["id"] for f in retrieved["selected"]], "result": result,
                    "retrieval_path": self.retrieval_path.relative_to(self.run).as_posix(),
                    "retrieval_sha256": audit.driver.file_sha(self.retrieval_path),
                    "binding": self.bound, "request_fingerprint": audit.driver.fingerprint(self.bound)}
        self.save_qa()

    def save_artifact(self, path, binding, payload):
        dump(path, audit.driver._seal({"schema": audit.driver.ARTIFACT_SCHEMA, "status": "complete", "error": None,
             "binding": binding, "request_fingerprint": audit.driver.fingerprint(binding), **payload}))

    def save_qa(self):
        self.row = audit.driver._seal({k: v for k, v in self.row.items() if k != "content_sha256"})
        dump(self.path, self.row)

    def verify(self):
        return audit.verified_qa(self.path, self.q, self.arm, self.raw, self.manifest, self.bank, None)[0]

    def replay(self):
        return audit.replay_complete(self.run, self.verify(), self.q, self.bank, None, self.raw, self.query_bound, self.route_bound)

    def test_frozen_retrieval_and_complete_raw_replay_without_client_or_network(self):
        with patch.object(audit.ModelAPI, "__init__", side_effect=AssertionError("No API client")), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
            proofs = self.replay()
        self.assertEqual([p["stage"] for p in proofs], [audit.evaluation.READER_STAGE, audit.evaluation.JUDGE_STAGE])
        self.assertTrue(all(len(p["cache_sha256"]) == 64 for p in proofs))

    def test_judge_tamper_with_old_seal_is_rejected(self):
        changed = copy.deepcopy(self.row)
        changed["result"]["judge"]["correct"] = False
        dump(self.path, changed)
        with self.assertRaises(audit.driver.ProtocolMismatch):
            self.verify()

    def test_missing_qa_is_not_a_successful_terminal_artifact(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_reference_identity_is_bound_before_raw_replay(self):
        different = {**self.q, "reference": "A different synthetic reference"}
        with self.assertRaises(audit.driver.ProtocolMismatch):
            audit.verified_qa(self.path, different, self.arm, self.raw, self.manifest, self.bank, None)

    def test_resealed_wrong_protocol_or_explicit_binding_is_rejected(self):
        original = copy.deepcopy(self.row)
        for field, value in [("protocol_sha256", "other"), ("binding", {"wrong": "reference"}), ("category", True)]:
            self.row = {**copy.deepcopy(original), field: value}
            self.save_qa()
            with self.assertRaises(ValueError):
                self.verify()

    def test_resealed_wrong_judge_hash_and_result_fail_raw_replay(self):
        self.row["result"]["judge_input_hash"] = "bad"
        self.save_qa()
        with self.assertRaisesRegex(ValueError, "raw reader/judge replay"):
            self.replay()

    def test_cache_request_and_incomplete_response_rejected(self):
        proofs = self.replay()
        p = self.run / "api_cache" / (proofs[0]["request_hash"] + ".json")
        original = json.loads(p.read_text("utf-8"))
        changed = copy.deepcopy(original); changed["request"]["body"]["contents"] = []
        dump(p, changed)
        with self.assertRaisesRegex(ValueError, "cache signature"):
            self.replay()
        changed = copy.deepcopy(original); changed["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
        dump(p, changed)
        with self.assertRaises(RuntimeError):
            self.replay()

    def test_failed_output_retains_error_but_cannot_hide_answer(self):
        self.row.update(status="failed", result=None, selected_ids=[], error={"type": "RuntimeError", "stage": "qa", "message": "mock transport failure"})
        self.save_qa(); self.assertEqual(self.verify()["status"], "failed")
        self.row["result"] = {"answer": "hidden"}; self.save_qa()
        with self.assertRaisesRegex(ValueError, "no answer"):
            self.verify()

    def event(self, proof, **extra):
        return {"stage": proof["stage"], "provider": "gemini", "model": proof["model"], "operation": "generateContent",
                "request_hash": proof["request_hash"], "status": 200, "cache_hit": False,
                "context": {"conversation": 0, "method": self.arm, "query_id": "mock_q", "architecture": "pilot400",
                            "driver_stage": "qa", "phase": "pilot_qa"}, **extra}

    def test_ledger_scope_raw_request_and_success_evidence(self):
        proofs = self.replay()
        rows = [self.event(p) for p in proofs]
        raw = lambda values: "\n".join(map(json.dumps, values)).encode()
        events, by_key = audit.ledger_index(raw(rows), {(0, self.arm, "mock_q")})
        audit.validate_complete_events(proofs, events)
        rows[0]["status"] = 403
        events, _ = audit.ledger_index(raw(rows), {(0, self.arm, "mock_q")})
        with self.assertRaisesRegex(ValueError, "lacks an attributed"):
            audit.validate_complete_events(proofs, events)
        rows[0]["context"]["method"] = "wrong-arm"
        with self.assertRaisesRegex(ValueError, "Unscoped"):
            audit.ledger_index(raw(rows), {(0, self.arm, "mock_q")})
        changed = copy.deepcopy(by_key[(0, self.arm, "mock_q")]); changed[0]["request_hash"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "different raw request"):
            audit.validate_complete_events(proofs, changed)

    def test_transport_unknown_and_cache_are_separate_physical_events(self):
        proofs = self.replay()
        rows = [self.event(proofs[0], status="transport_error", cache_hit=None, usage_unknown=True),
                self.event(proofs[0], cache_hit=True, status=None), self.event(proofs[1])]
        events, _ = audit.ledger_index("\n".join(map(json.dumps, rows)).encode(), {(0, self.arm, "mock_q")})
        audit.validate_complete_events(proofs, events)
        self.assertEqual(sum(e["cache_hit"] for e in events), 1)
        self.assertTrue(events[0]["usage_unknown"])

    def test_terminal_status_catches_post_save_validation_failure(self):
        artifacts = [{**self.row, "arm": arm} for arm in audit.driver.ARMS]
        row = {"schema": "jev-mem-pilot400-conversation-qa-v2", "conversation": 0,
               "cohort_sha256": audit.driver.COHORT_SHA256, "protocol_sha256": self.manifest["protocol_sha256"],
               "requested_query_ids": ["mock_q"], "methods": {arm: {"requested": 1, "completed": 1, "failed": 0} for arm in audit.driver.ARMS},
               "status": "complete", "errors": []}
        path = self.folder / "qa_status.json"; dump(path, row)
        audit.validate_terminal_status(path, 0, [self.q], self.manifest, artifacts)
        row["errors"] = [{"query_id": "mock_q", "arm": self.arm, "error": {"type": "ProtocolMismatch", "stage": "qa_cache_validation", "message": "stale retrieval"}}]
        dump(path, row)
        with self.assertRaisesRegex(ValueError, "post-save validation"):
            audit.validate_terminal_status(path, 0, [self.q], self.manifest, artifacts)


if __name__ == "__main__":
    unittest.main()
