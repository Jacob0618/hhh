"""Independent no-redraw and no-partial-QA continuation counterexamples."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_five_baselines_amem_native_v3 as fixtures
from tools import run_five_baselines_amem_native_v3 as w
from tools.baselines_common_v1 import BaselineFatal, save_new, sha


class IndependentContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ContinuationTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_known_format_replacement_that_is_still_invalid_cannot_be_redrawn(self):
        f = self.fixture
        f.replacement()
        source_bytes = f.raw.read_bytes()
        attempts = []
        def complete_bad_json(api, provider, model, body, **kwargs):
            # The one-use certificate must already be durable at transport entry.
            marker = api.run_dir / "replacements" / (f.key + ".json")
            self.assertTrue(marker.is_file())
            attempts.append(f.key)
            raw = {"request": copy.deepcopy(f.sig), "response": {
                "modelVersion": "gemini-2.5-flash", "candidates": [
                    {"index": 0, "finishReason": "STOP", "content": {
                        "role": "model", "parts": [{"text": '{"context": invalid}'}]}}]}}
            path = api.run_dir / "api_cache" / (f.key + ".json")
            save_new(path, raw)
            api.events.append({"request_hash": f.key, "cache_path": str(path), "cache_sha256": sha(path),
                               "context": copy.deepcopy(api.context), "stage": "amem_analysis", "error_type": None})
            return raw["response"]
        with patch.object(w.BaselineAPI, "request", complete_bad_json):
            for occurrence in range(2):
                expected = BaselineFatal if occurrence == 0 else FileExistsError
                with self.assertRaises(expected):
                    f.api().generate("same original prompt", system="You must respond with a JSON object.",
                                     json_output=True, max_output=16384, stage="amem_analysis")
        self.assertEqual(attempts, [f.key])
        self.assertEqual(f.raw.read_bytes(), source_bytes)
        self.assertTrue((f.new / "work/api_method/api_cache" / (f.key + ".json")).is_file())
        self.assertFalse((f.new / "work/api_method/parser_lineage" / (f.key + ".json")).exists())

    def test_failed_partial_native_build_never_enters_retrieval_or_qa(self):
        f = self.fixture
        def incomplete(conversation, native, api):
            save_new(native / "checkpoint.json", {"completed_notes": 1, "build_complete": False})
            raise BaselineFatal("offline failed second note")
        module = SimpleNamespace(build=incomplete, retrieve=lambda *args: self.fail("partial retrieval forbidden"))
        class OfflineAPI:
            def __init__(self, *args, **kwargs): pass
            def _redact(self, value): return value
        with patch.object(w.base, "BaselineAPI", OfflineAPI), patch.object(w.base, "adapter", return_value=module), patch.object(w.base, "qa_one", side_effect=AssertionError("partial QA forbidden")) as qa:
            outcome = w.base.conv_worker(f.new, "amem", 0, [{"query_id": "never", "question": "never"}], {"conversation": {}})
        self.assertEqual(outcome["status"], "needs_review")
        qa.assert_not_called()
        self.assertFalse((f.new / "results/c0/build.json").exists())
        self.assertFalse((f.new / "results/c0/retrieval").exists())
        self.assertFalse((f.new / "results/firstpass").exists())


if __name__ == "__main__":
    unittest.main()
