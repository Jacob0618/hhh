"""Offline compatibility tests for the separately versioned pilot entrypoints.

The established v1 test cases are loaded in isolated module namespaces and pointed
at v2. Their fake provider supplies HTTP-shaped extraction responses; it never
constructs ModelAPI, reads credentials, spawns a real child, or sends a request.
"""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pilot400_experiments as old_driver
from tools import run_pilot400_experiments_v2 as driver
from tools import run_pilot400_pipeline_v2 as pipeline
from tools import check_pilot400_health_v2 as health
from jev_mem import group_writer, group_writer_v2


def isolated_test_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def http_request_from_fake_generate(self, provider, model, body, *, operation, stage):
    if provider != "gemini" or operation != "generateContent" or stage != "group_fact_extraction":
        raise AssertionError("Unexpected extraction HTTP request in compatibility test")
    if self.retries != 2:
        raise AssertionError("Base extraction must retain the frozen provider retry policy")
    value = self.generate(body["contents"][0]["parts"][0]["text"], model=model, stage=stage)
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(value)}]}}]}


class Version2ContractTests(unittest.TestCase):
    def test_only_writer_policy_and_versions_change(self):
        old, new = old_driver.protocol(), driver.protocol()
        self.assertEqual(driver.RUN_SCHEMA, "jev-mem-pilot400-run-v2")
        self.assertEqual(driver.QA_SCHEMA, "jev-mem-pilot400-qa-v2")
        self.assertEqual(driver.READER_PROTOCOL, old_driver.READER_PROTOCOL)
        self.assertEqual(driver.PARAMETERS, old_driver.PARAMETERS)
        for key in ("arms", "source_dataset_sha256", "cohort_sha256", "cohort_file_sha256",
                    "reader_protocol", "models", "grouping", "raw_embedding_text"):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(new["writer_schema"], "date-aware-raw-group-writer-v2")
        self.assertEqual(new["input_payload_schema"], "date-aware-raw-group-writer-v1")
        self.assertEqual(new["base_extraction_provider_retries"], 2)
        self.assertEqual(new["feedback_provider_retries"], 0)
        self.assertEqual(new["reextraction_policy"], "one-validator-feedback-reextraction-v1")
        self.assertIn("one_validator_feedback_reextraction", new["failure_policy"])
        self.assertEqual(group_writer.SYSTEM_PROMPT, group_writer_v2.SYSTEM_PROMPT)
        self.assertIn("jev_mem/group_writer_v2.py", new["code_sha256"])
        self.assertIn("tools/run_pilot400_experiments_v2.py", new["code_sha256"])
        for name, digest in old["code_sha256"].items():
            if name != "tools/run_pilot400_experiments.py":
                self.assertEqual(new["code_sha256"][name], digest)

    def test_old_pilot_namespaces_rejected_before_provider_or_run_lock(self):
        factory, gate = Mock(), Mock()
        with patch.object(driver, "run_lock", side_effect=AssertionError("old runs must remain untouched")) as lock:
            for relative in driver.PROTECTED_PILOT_RUNS:
                for path in (ROOT / relative, ROOT / relative / "child"):
                    with self.subTest(path=path), self.assertRaisesRegex(ValueError, "preserved version 1"):
                        driver.run_stage("write", path, api_factory=factory, gate_checker=gate)
        lock.assert_not_called()
        factory.assert_not_called()

    def test_old_manifest_rejected_in_new_namespace_without_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "v2"
            driver.publish_json(run / "run_manifest.json", {"schema": old_driver.RUN_SCHEMA})
            before = (run / "run_manifest.json").read_bytes()
            with patch.object(driver, "ModelAPI", side_effect=AssertionError("credentials forbidden")), \
                 self.assertRaises(driver.ProtocolMismatch):
                driver.ensure_manifest(run, create=False)
            self.assertEqual((run / "run_manifest.json").read_bytes(), before)

    def test_pipeline_check_has_v2_schema_and_no_side_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            result = pipeline.check(run, Path(folder) / "review.json", Path(folder) / "closure.json",
                                    review_checker=lambda *_: {"protocol_sha256": "p"},
                                    gate_checker=lambda *_: {"closure_sha256": "c"})
            self.assertEqual(result["schema"], "jev-mem-pilot400-pipeline-v2")
            self.assertFalse(run.exists())
            self.assertEqual(pipeline.REVIEW_SCHEMA, "jev-mem-pilot400-development-review-v2")
            self.assertEqual(pipeline.DRIVER, ROOT / "tools/run_pilot400_experiments_v2.py")

    def test_health_requires_saved_v2_manifest_and_matching_status_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            manifest = driver.ensure_manifest(run, create=True)
            state = {"stage": "write", "status": "complete", "pid": 999,
                     "protocol_sha256": "old-protocol", "conversations": {}}
            driver.publish_json(run / "write_status.json", state)
            with patch.object(health, "WinProcess", side_effect=AssertionError("terminal needs no PID probe")), \
                 patch.object(driver, "ModelAPI", side_effect=AssertionError("credentials forbidden")):
                snap = health.snapshot(run, "write")
            self.assertEqual(snap["schema"], "jev-mem-pilot400-health-v2")
            self.assertIn("stage_protocol_mismatch", snap["problems"])
            changed = copy.deepcopy(manifest)
            changed["schema"] = old_driver.RUN_SCHEMA
            driver.publish_json(run / "run_manifest.json", changed)
            with self.assertRaises(driver.ProtocolMismatch):
                health.snapshot(run, "write")

    def test_health_cannot_accept_a_v1_worker_with_same_run_and_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            manifest = driver.ensure_manifest(run, create=True)
            started = datetime.now(timezone.utc)
            state = {"stage": "write", "status": "running", "pid": 123,
                     "started_utc": started.isoformat(), "protocol_sha256": manifest["protocol_sha256"],
                     "conversations": {"0": {"status": "running"}}}
            driver.publish_json(run / "write_status.json", state)
            handle = Mock()
            handle.is_running.return_value = True
            handle.creation_time_100ns = int((started.timestamp() + 11644473600) * 10_000_000)
            args = [sys.executable, "-u", str(ROOT / "tools/run_pilot400_experiments.py"),
                    "write", "--run-dir", str(run)]
            response = subprocess.CompletedProcess([], 0, json.dumps({"CommandLine": "mock command"}), "")
            with patch.object(health, "WinProcess", return_value=handle), \
                 patch.object(health.subprocess, "run", return_value=response), \
                 patch.object(health, "split_command_line", return_value=args):
                bad = health.snapshot(run, "write")
                args[2] = str(health.DRIVER)
                good = health.snapshot(run, "write")
            self.assertIn("worker_identity_not_verified", bad["problems"])
            self.assertFalse(good["problems"])

    def test_health_different_manifest_between_snapshots_requires_review(self):
        first = {"captured_utc": "2026-09-28T00:00:00+00:00", "run_dir": "r", "protocol_sha256": "p1",
                 "progress": {}, "problems": [], "api_bytes": 0, "state": {"status": "complete"}}
        second = {**first, "protocol_sha256": "p2"}
        report = health.compare(first, second, [])
        self.assertTrue(report["requires_investigation"])
        self.assertIn("snapshot_run_or_protocol_changed", report["problems"])


def load_tests(loader, tests, pattern):
    runner_tests = isolated_test_module("test_pilot400_runner.py", "_pilot_v2_runner_compat")
    runner_tests.runner = driver
    runner_tests.FakeAPI.request = http_request_from_fake_generate
    pipeline_tests = isolated_test_module("test_pilot400_pipeline_v1.py", "_pilot_v2_pipeline_compat")
    pipeline_tests.pipeline = pipeline
    health_tests = isolated_test_module("test_pilot400_health.py", "_pilot_v2_health_compat")
    health_tests.compare, health_tests.progress = health.compare, health.progress
    for module in (runner_tests, pipeline_tests, health_tests):
        tests.addTests(loader.loadTestsFromModule(module))
    return tests


if __name__ == "__main__":
    unittest.main()
