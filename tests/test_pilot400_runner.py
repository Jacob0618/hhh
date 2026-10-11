"""Mock-only pilot driver integration: no credentials, sockets or live mutations."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_pilot400_experiments as runner


class Monitor:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def update(self, ci=None, **values):
        with self.lock:
            self.events.append((ci, copy.deepcopy(values)))


class FakeAPI:
    def __init__(self, state=None, context=None):
        self.state = state if state is not None else {"calls": [], "lock": threading.Lock(), "fail_reader_arm": None}
        self.context = context or {}

    def scoped(self, **context):
        return FakeAPI(self.state, {**self.context, **context})

    def _record(self, kind, **data):
        with self.state["lock"]:
            self.state["calls"].append({"kind": kind, "context": dict(self.context), **copy.deepcopy(data)})

    def generate(self, prompt, **settings):
        value = json.loads(prompt)
        self._record("generate", payload=value, **settings)
        stage = settings["stage"]
        if stage == "group_fact_extraction":
            assert "question" not in value and "reference" not in value
            first = value["MEMBERS"][0]["source_turn_id"]
            return {"facts": [{"text": f"Source {first}, distinct fact {i}: Sam.", "source_turn_ids": [first],
                               "temporal_status": "not_temporal"} for i in range(24)]}
        if stage == "pilot_qa_reader":
            if self.context.get("method") == self.state["fail_reader_arm"]:
                raise RuntimeError("mock reader failure")
            assert len(value["facts"]) == 20
            assert "reference_answer" not in value
            return {"answer": "Sam", "used_fact_ids": ["f1"]}
        if stage == "pilot_strict_judge":
            return {"correct": True, "completeness": 1., "reason": "mock judge result"}
        raise AssertionError("Unexpected generation/planner stage: " + stage)

    def embed(self, texts, **settings):
        self._record("embed", texts=texts, **settings)
        return [[1.] + [0.] * 767 for _ in texts]

    def count_tokens(self, text, **settings):
        self._record("count_tokens", **settings)
        return max(1, len(text.split()))

    def score_many(self, state, criteria, **settings):
        self._record("score_many", state=state, criteria=criteria, **settings)
        if "topic_shift" in criteria or any(key.startswith("group_") for key in criteria):
            return {key: .1 for key in criteria}
        return {key: .5 for key in criteria}


def raw_fixture():
    return [{"source_turn_id": f"D{i // 2 + 1}:{i % 2 + 1}", "session_id": f"session_{i // 2 + 1}",
             "speaker": "Alex" if i % 2 == 0 else "Sam", "observation_date": f"2023-05-{8 + i // 2:02d}",
             "text": f"Raw source turn {i} mentions a distinct event.", "role": "user" if i % 2 == 0 else "assistant"}
            for i in range(4)]


class PilotDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev_pilot_driver_mock_")
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "pilot"
        self.raw = raw_fixture()
        self.api = FakeAPI()
        self.monitor = Monitor()
        self.manifest = runner.ensure_manifest(self.run_dir, create=True)
        self.questions = [{"id": "c0_q0", "conversation": 0, "category": 1, "source_qa_index": 0,
                           "question": "Who is named?", "reference": "Sam", "gold_evidence": ["GOLD_SENTINEL"]}]

    def prepare_banks_and_tags(self):
        result = runner.write_conversation(self.run_dir, 0, self.raw, self.manifest, self.api, self.monitor)
        self.assertEqual(result["status"], "complete", result)
        result = runner.tag_conversation(self.run_dir, 0, self.raw, self.manifest, self.api, self.monitor)
        self.assertEqual(result["status"], "complete", result)

    def qa(self):
        return runner.qa_conversation(self.run_dir, 0, self.raw, self.questions, self.manifest, self.api, self.monitor)

    def test_mock_full_pipeline_shared_embedding_types_rank20_and_no_hidden_calls(self):
        self.prepare_banks_and_tags()
        bank = runner.load_bank(self.run_dir, 0, "session", self.raw, self.manifest)
        tags, _ = runner.load_tags(self.run_dir, 0, self.raw, self.manifest, bank)
        self.assertEqual(set(tags), {f["id"] for f in bank["facts"]})
        self.assertGreater(len(tags), 20)
        all_previous_calls = list(self.api.state["calls"])
        self.api.state["calls"].clear()
        result = self.qa()
        self.assertEqual(result["status"], "complete", result)
        calls = self.api.state["calls"]
        self.assertEqual(sum(c["kind"] == "embed" for c in calls), 1)
        scores = [c for c in calls if c["kind"] == "score_many"]
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["stage"], "pilot_query_evidence_types")
        self.assertEqual(len(scores[0]["criteria"]), 10)
        self.assertEqual(sum(c.get("stage") == "pilot_qa_reader" for c in calls), 3)
        self.assertEqual(sum(c.get("stage") == "pilot_strict_judge" for c in calls), 3)
        self.assertFalse(any("planner" in c.get("stage", "") or "conditional" in c.get("stage", "") for c in calls))
        for arm in runner.ARMS:
            path = runner.conv_dir(self.run_dir, 0) / "qa" / arm / "c0_q0.json"
            row = runner.read(path)
            self.assertEqual(row["schema"], runner.QA_SCHEMA)
            self.assertEqual(row["status"], "complete")
            self.assertEqual(len(set(row["selected_ids"])), 20)
            self.assertEqual(row["result"]["reader_input"]["conversation_reference_date"], "2023-05-09")
            self.assertEqual(row["cohort_sha256"], runner.COHORT_SHA256)
        for call in all_previous_calls:
            self.assertNotIn("GOLD_SENTINEL", json.dumps({k: v for k, v in call.items() if k != "context"}))

    def test_successful_rerun_has_no_new_model_calls_and_identical_artifacts(self):
        self.prepare_banks_and_tags()
        self.assertEqual(self.qa()["status"], "complete")
        paths = list((runner.conv_dir(self.run_dir, 0) / "qa").glob("*/*.json"))
        before = {p: p.read_bytes() for p in paths}
        count = len(self.api.state["calls"])
        self.assertEqual(runner.write_conversation(self.run_dir, 0, self.raw, self.manifest, self.api, self.monitor)["status"], "complete")
        self.assertEqual(runner.tag_conversation(self.run_dir, 0, self.raw, self.manifest, self.api, self.monitor)["status"], "complete")
        self.assertEqual(self.qa()["status"], "complete")
        self.assertEqual(len(self.api.state["calls"]), count)
        self.assertEqual(before, {p: p.read_bytes() for p in paths})

    def test_failed_arm_is_preserved_without_retry_and_other_arms_complete(self):
        self.prepare_banks_and_tags()
        self.api.state["fail_reader_arm"] = runner.ARMS[1]
        result = self.qa()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["methods"][runner.ARMS[0]]["completed"], 1)
        self.assertEqual(result["methods"][runner.ARMS[2]]["completed"], 1)
        self.assertEqual(result["methods"][runner.ARMS[1]]["failed"], 1)
        path = runner.conv_dir(self.run_dir, 0) / "qa" / runner.ARMS[1] / "c0_q0.json"
        original = path.read_bytes()
        count = len(self.api.state["calls"])
        self.api.state["fail_reader_arm"] = None
        result = self.qa()
        self.assertEqual(result["methods"][runner.ARMS[1]]["failed"], 1)
        self.assertEqual(len(self.api.state["calls"]), count)
        self.assertEqual(original, path.read_bytes())

    def test_cache_reader_input_reconstructed_even_if_content_hash_resealed(self):
        self.prepare_banks_and_tags()
        self.assertEqual(self.qa()["status"], "complete")
        path = runner.conv_dir(self.run_dir, 0) / "qa" / runner.ARMS[0] / "c0_q0.json"
        row = runner.read(path)
        row["result"]["reader_input"]["facts"][0]["observation_date"] = "1999-01-01"
        row = runner._seal({k: v for k, v in row.items() if k != "content_sha256"})
        runner.publish_json(path, row)
        tampered = path.read_bytes()
        count = len(self.api.state["calls"])
        result = self.qa()
        self.assertEqual(result["methods"][runner.ARMS[0]]["failed"], 1)
        self.assertEqual(result["methods"][runner.ARMS[1]]["completed"], 1)
        self.assertEqual(path.read_bytes(), tampered)
        self.assertEqual(len(self.api.state["calls"]), count)

    def test_artifact_success_fingerprint_mismatch_refuses_overwrite(self):
        path = self.run_dir / "test_artifact.json"
        work = Mock(return_value={"value": 1})
        runner.artifact(path, {"version": 1}, work, stage="mock")
        first = path.read_bytes()
        with self.assertRaises(runner.ProtocolMismatch):
            runner.artifact(path, {"version": 2}, work, stage="mock")
        self.assertEqual(path.read_bytes(), first)
        self.assertEqual(work.call_count, 1)

    def test_offline_check_never_constructs_provider_or_creates_run(self):
        fresh = Path(self.temp.name) / "check_only"
        with patch.object(runner, "ModelAPI", side_effect=AssertionError("credentials forbidden")) as factory, \
             patch("jev_mem.providers.load_credentials", side_effect=AssertionError("credentials forbidden")), \
             patch.object(runner, "main_batch_gate", side_effect=FileNotFoundError("no reviewed closure")):
            result = runner.offline_check(fresh)
        factory.assert_not_called()
        self.assertEqual(result["requested_questions_per_arm"], 400)
        self.assertEqual(result["arms"], list(runner.ARMS))
        self.assertFalse(result["ready_for_network"])
        self.assertFalse(result["scientific_live_green"])
        self.assertFalse(fresh.exists())

    def test_main_gate_blocks_before_api_or_run_creation(self):
        fresh = Path(self.temp.name) / "gate_blocked"
        factory = Mock(side_effect=AssertionError("must not initialize"))
        with self.assertRaisesRegex(ValueError, "main still active"):
            runner.run_stage("qa", fresh, conversations=[0], api_factory=factory,
                             gate_checker=Mock(side_effect=ValueError("main still active")))
        factory.assert_not_called()
        self.assertFalse(fresh.exists())

    def test_api_initialization_failure_has_terminal_status_and_releases_lock(self):
        fresh = Path(self.temp.name) / "api_init_failure"
        result = runner.run_stage("write", fresh, conversations=[0],
                                  gate_checker=lambda path: {"mock_gate": True},
                                  api_factory=Mock(side_effect=RuntimeError("mock credential initialization failure")))
        self.assertEqual(result["status"], "requires_review")
        saved = runner.read(fresh / "write_status.json")
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error"]["stage"], "api_initialization")
        self.assertEqual(saved["conversations"]["0"]["status"], "failed")
        self.assertFalse((fresh / "driver.lock").exists())

    def test_stage_scope_is_exact_fixed400_and_worker_cap_is_enforced(self):
        fresh = Path(self.temp.name) / "scope_mock"
        seen = []
        lock = threading.Lock()

        def mock_qa(run_dir, ci, raw, questions, manifest, api, monitor):
            with lock:
                seen.extend(q["id"] for q in questions)
            return {"status": "complete"}

        with patch.object(runner, "qa_conversation", side_effect=mock_qa):
            result = runner.run_stage("qa", fresh, gate_checker=lambda path: {"mock_gate": True},
                                      api_factory=lambda *args, **kwargs: self.api)
        cohort = runner.read(runner.COHORT_PATH)
        self.assertEqual(set(seen), set(cohort["query_ids"]))
        self.assertEqual(len(seen), 400)
        self.assertEqual(result["status"], "complete")
        state = runner.read(fresh / "qa_status.json")
        self.assertEqual((state["done"], state["expected"]), (10, 10))
        self.assertEqual(state["api_log_path"], "api_calls.jsonl")
        with self.assertRaises(ValueError):
            runner.run_stage("qa", fresh, workers=9)

    def test_bounded_status_publication_retries_transient_sharing_errors(self):
        target = self.run_dir / "status_mock.json"
        real_replace = Path.replace
        attempts = []

        def flaky(path, destination):
            attempts.append(path)
            if len(attempts) < 3:
                raise PermissionError("mock Windows sharing violation")
            return real_replace(path, destination)

        with patch.object(Path, "replace", flaky), patch.object(runner.time, "sleep"):
            runner.publish_json(target, {"status": "complete"})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(runner.read(target)["status"], "complete")
        self.assertFalse(list(self.run_dir.glob("*.tmp")))

    def test_closure_requires_review_hash_and_actual_no_running_state(self):
        root = Path(self.temp.name) / "closure_workspace"
        review = root / "research/review.md"
        review.parent.mkdir(parents=True)
        review.write_text("Offline mock review fixture.", encoding="utf-8")
        for relative in runner.MAIN_RUNS:
            runner.publish_json(root / relative / "queue_status.json", {"status": "complete", "jobs": {}})
        closure_path = root / "research/main_batch_closure.json"
        closure = {"schema": runner.CLOSURE_SCHEMA, "status": "completed", "source_dataset_sha256": runner.SOURCE_HASH,
                   "reviewed_utc": "2026-09-28T00:00:00+00:00", "reviewed_by": "offline mock reviewer",
                   "review_path": "research/review.md", "review_sha256": runner.file_sha(review), "live_qa_running": False,
                   "closed_runs": [{"run_dir": relative, "requested_questions": 1540, "status": "complete"} for relative in runner.MAIN_RUNS]}
        runner.publish_json(closure_path, closure)
        with patch.object(runner, "ROOT", root):
            self.assertEqual(runner.main_batch_gate(closure_path, process_probe=lambda: [])["reviewed_status"], "completed")
            with self.assertRaisesRegex(ValueError, "still running"):
                runner.main_batch_gate(closure_path, process_probe=lambda: [{"pid": 111}])
            runner.publish_json(root / runner.MAIN_RUNS[0] / "queue_status.json", {"status": "running"})
            with self.assertRaisesRegex(ValueError, "still reports running"):
                runner.main_batch_gate(closure_path, process_probe=lambda: [])
            closure["review_sha256"] = "0" * 64
            runner.publish_json(closure_path, closure)
            with self.assertRaisesRegex(ValueError, "report path/hash"):
                runner.main_batch_gate(closure_path, process_probe=lambda: [])


if __name__ == "__main__":
    unittest.main()
