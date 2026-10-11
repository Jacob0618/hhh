"""Offline report integration; synthetic fixture labels are not real model scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_regroup_refinement_v1 as c
from tools.run_global_segment_grouping import save_once, pointer
from jev_mem.sparse_topic_pipeline import _group_record
from jev_mem.fullconv_llm_segmentation_v1 import request_body, parse_partition


def raw():
    return [{"source_turn_id": f"D1:{i+1}", "session_id": "session_1", "speaker": "A",
             "observation_date": "2023-01-01", "text": f"Turn {i}"} for i in range(4)]


class ComparisonDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="compare_partition_offline_"); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.old = self.root / "old"; self.new = self.root / "new"; self.llm = self.root / "llm"
        for target in ("socket.create_connection", "requests.Session.request", "jev_mem.providers.load_credentials"):
            p = patch(target, side_effect=AssertionError("No network or credentials")); p.start(); self.addCleanup(p.stop)
        for directory in (self.old, self.new, self.llm):
            save_once(directory / "manifest.json", {"model": "gemini-3.8-flash", "jobs": []})
        status = {"status": "complete", "conversations": {str(i): {"status": "complete"} for i in range(10)}}
        for directory, name in ((self.old, "score_status.json"), (self.new, "run_status.json"), (self.llm, "status.json")):
            (directory / name).write_text(json.dumps(status), encoding="utf-8")
        save_once(self.llm / "recovery_review.json", {"inherited_complete": [], "failure_population": []})
        self.parts = [_group_record(raw(), [0, 1], "g0"), _group_record(raw(), [2, 3], "g1")]
        for ci in range(10):
            value = {"conversation": ci, "status": "complete", "segments": self.parts, "groups": self.parts}
            save_once(self.old / f"conversations/conv_{ci}/grouping.json", value)
            save_once(self.llm / f"conv_{ci}/grouping.json", value)
            artifacts = {}
            for arm in c.CANDIDATES[1:]:
                path = self.new / f"conv_{ci}/{arm}/grouping.json"; save_once(path, value); artifacts[arm] = pointer(path)
            save_once(self.new / f"conv_{ci}/result.json", {"status": "complete", "conversation": ci,
                "manifest_sha256": pointer(self.new / "manifest.json")["sha256"], "artifacts": artifacts})

    def report(self, **kwargs):
        with patch.object(c, "original_turns", return_value=[raw() for _ in range(10)]), \
             patch.object(c, "llm_lineage", return_value={"kind": "synthetic_fixture"}):
            return c.build_report(self.old, self.new, self.llm, **kwargs)

    def test_full_population_report_is_agreement_not_accuracy(self):
        result = self.report()
        self.assertEqual(result["interpretation"], "agreement_to_llm_not_accuracy")
        self.assertFalse(result["reference_is_human_gold"]); self.assertEqual(result["new_model_calls"], 0)
        self.assertEqual(result["common_conversations"], list(range(10)))
        for value in result["aggregates"].values():
            self.assertEqual(value["planned_conversations"], 10)
            self.assertEqual(value["compared_conversations"], 10)
            self.assertEqual(value["metrics"]["boundary_exact"]["f1"], 1)
            self.assertEqual(value["metrics"]["event_clustering"]["pairwise_positive"]["f1"], 1)

    def test_missing_reference_preserves_ten_denominator(self):
        (self.llm / "conv_2/grouping.json").unlink()
        result = self.report()
        self.assertEqual(result["source_states"]["llm_reference"][2]["status"], "missing")
        for value in result["aggregates"].values():
            self.assertEqual(value["compared_conversations"], 9)
            self.assertEqual(value["missing_or_invalid_conversations"], [2])
            self.assertEqual(value["planned_source_turns"], 40)
            self.assertEqual(value["compared_source_turns"], 36)

    def test_running_controller_cannot_be_called_final(self):
        p = self.llm / "status.json"
        p.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.report()
        result = self.report(allow_running_snapshot=True)
        self.assertEqual(result["snapshot_kind"], "provisional_running_not_final")
        self.assertFalse(result["all_controllers_terminal"])

    def test_terminal_label_with_running_child_is_not_terminal(self):
        p = self.llm / "status.json"; data = json.loads(p.read_text()); data["conversations"]["0"]["status"] = "requesting"
        p.write_text(json.dumps(data))
        with self.assertRaises(RuntimeError): self.report()

    def test_old_remaining_dispatch_terminal_subset_is_explicitly_supported(self):
        p = self.old / "score_status.json"
        p.write_text(json.dumps({"status": "complete", "requested_conversations": list(range(2, 10)),
                                "conversations": {str(i): {"status": "complete"} for i in range(2, 10)}}))
        result = self.report()
        self.assertEqual(result["snapshot_kind"], "terminal")
        self.assertEqual(result["aggregates"]["original_global"]["compared_conversations"], 10)
        row = json.loads(p.read_text()); row["conversations"]["2"]["status"] = "running"
        self.assertFalse(c.controller_terminal("original_global", row))
        self.assertFalse(c.controller_terminal("llm", json.loads(p.read_text())))

    def test_all_attempt_accounting_includes_transport_failures(self):
        path = self.root / "api_calls.jsonl"
        rows = [{"context": {"conversation": 1}, "request_hash": "h", "stage": "segment",
                 "status": "transport_error", "attempt": i+1} for i in range(3)]
        rows.append({"context": {"conversation": 0}, "request_hash": "j", "stage": "segment", "status": 200,
                     "cache_hit": False, "usage": {"promptTokenCount": 100, "candidatesTokenCount": 20,
                                                   "thoughtsTokenCount": 30, "totalTokenCount": 150}})
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        result = c.ledger_accounting(path)
        self.assertEqual(result["physical_http_attempts"], 4)
        self.assertEqual(result["attempts_with_unknown_usage"], 3)
        self.assertEqual(result["logical_request_count"], 2)
        self.assertEqual(result["reported_usage"]["totalTokenCount"], 150)
        self.assertIsNone(result["usd_total"])

    def test_corrupted_source_is_excluded_not_scored_as_failure_zero(self):
        path = self.old / "conversations/conv_3/grouping.json"; path.unlink()
        bad = copy.deepcopy(self.parts); bad[0]["turns"][0]["text"] = "rewritten"
        save_once(path, {"conversation": 3, "status": "complete", "segments": bad, "groups": bad})
        result = self.report()
        self.assertEqual(result["source_states"]["original_global"][3]["status"], "invalid_artifact")
        self.assertEqual(result["aggregates"]["original_global"]["compared_conversations"], 9)
        self.assertEqual(result["aggregates"]["identity_only"]["compared_conversations"], 10)
        self.assertEqual(len(result["common_conversations"]), 9)

    def test_llm_actual_response_replay_and_budget_lineage(self):
        units = raw(); body = request_body(units); body["generationConfig"]["maxOutputTokens"] = 65536
        body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        inp = self.root / "input.json"; save_once(inp, {"conversation": 0, "raw_turns": units, "request_body": body})
        text = "0 1 E001\n2 3 E002"; response = {"modelVersion": "gemini-3.8-flash",
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
        sig = {"provider": "gemini", "model": "gemini-3.8-flash", "operation": "generateContent", "body": body}
        cache = self.root / "response.json"; save_once(cache, {"request": sig, "response": response})
        key = c.hashlib.sha256(json.dumps(sig, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result = {**parse_partition(text, units), "input": pointer(inp), "response_text": text,
            "api_events": [{"error_type": None, "cache_path": str(cache), "cache_sha256": pointer(cache)["sha256"],
                            "request_hash": key, "context": {"conversation": 0}}]}
        manifest = {"model": "gemini-3.8-flash", "jobs": [{"conversation": 0, "input": pointer(inp)}]}
        review = {"failure_population": [{"conversation": 0}], "inherited_complete": []}
        lineage = c.llm_lineage(result, manifest, review, 0, units)
        self.assertEqual(lineage["effective_generation"]["maxOutputTokens"], 65536)
        self.assertEqual(lineage["kind"], "single_reviewed_recovery")
        result["groups"][0]["turns"][0]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "differs"):
            c.llm_lineage(result, manifest, review, 0, units)


if __name__ == "__main__": unittest.main()
