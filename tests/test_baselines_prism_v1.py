"""Offline PRISM integration: native graph/FAISS/search with fake model outputs."""
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import baselines_prism_v1 as adapter
from tools import baselines_vmem_v1 as shared
from tools.baselines_common_v1 import BaselineFatal
from tests.test_baselines_vmem_v1 import FakeEncoder


def conversation(count=12):
    raw = {"speaker_a": "Ada", "speaker_b": "Bob", "event_summary": "GOLD_SENTINEL"}
    for index in range(1, count + 1):
        raw[f"session_{index}"] = [{"speaker": "Ada", "dia_id": f"D{index}:0",
            "text": f"Ada visited museum{index} and collected item{index}.",
            "blip_caption": "A red bird." if index == 1 else "", "evidence": "GOLD_SENTINEL"}]
        raw[f"session_{index}_date_time"] = f"2023-05-{index:02d}"
    return raw


class FakeAPI:
    def __init__(self, calls=None, fail_at=None):
        self.calls = [] if calls is None else calls
        self.fail_at = fail_at

    def scoped(self, **context):
        return self

    def generate(self, prompt, *, stage, **kwargs):
        self.calls.append({"prompt": prompt, "stage": stage, "kwargs": kwargs})
        if self.fail_at and len(self.calls) == self.fail_at:
            raise BaselineFatal("mock retained provider failure")
        if stage == "prism_extract":
            text = prompt.split("Text chunk:\n", 1)[1].split("\n\nReturn a single JSON", 1)[0]
            return {"episode_summary": text, "entities": [{"name": "Ada", "entity_type": "person"}],
                "facet_points": [{"content": text, "related_entity_name": "Ada"}],
                "facets": [{"theme": "Ada visits", "facet_point_indices": [0]}], "temporal_info": []}
        if stage == "prism_causal":
            return {"causal_pairs": []}
        if stage == "prism_rerank":
            count = len(re.findall(r"^\[\d+\]", prompt, re.MULTILINE))
            return [{"index": index, "score": 10 - index * 0.5} for index in range(count)]
        if stage == "prism_intent":
            return {"temporal": 0.0, "causal": 0.0, "multi_hop": 0.6, "entity_centric": 0.9}
        raise AssertionError(stage)


class PRISMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.encoder_patch = patch.object(shared, "_ENCODER", FakeEncoder())
        self.encoder_patch.start()
        adapter._RUNTIMES.clear()

    def tearDown(self):
        self.encoder_patch.stop()
        adapter._RUNTIMES.clear()
        self.tmp.cleanup()

    def test_native_graph_n3_export_and_reload_without_regeneration(self):
        api = FakeAPI()
        state = adapter.build(conversation(), self.directory, api)
        self.assertEqual(state["stats"]["episode_count"], 12)
        self.assertEqual(sum(call["stage"] == "prism_extract" for call in api.calls), 12)
        self.assertEqual(sum(call["stage"] == "prism_causal" for call in api.calls), 2)
        self.assertNotIn("GOLD_SENTINEL", json.dumps(api.calls))
        self.assertIn("A red bird.", json.dumps(api.calls))
        result = adapter.retrieve("What museum has Ada visited?", state, api)
        self.assertEqual(len(result["evidence"]), 5)
        self.assertEqual(result["trace"]["reranked_from"], 10)
        self.assertTrue(any(call["stage"] == "prism_rerank" for call in api.calls))
        self.assertTrue(all(item["dia_ids"] for item in result["evidence"]))
        self.assertIn("[Source dialogue]", result["context"])
        self.assertTrue(all(Path(path).is_file() for path in state["export_paths"]))
        adapter._RUNTIMES.clear()
        resumed_api = FakeAPI()
        resumed_state = adapter.build(conversation(), self.directory, resumed_api)
        self.assertEqual(resumed_state, state)
        self.assertEqual(resumed_api.calls, [])
        resumed = adapter.retrieve("What museum has Ada visited?", resumed_state, resumed_api)
        self.assertEqual(resumed["context"], result["context"])

    def test_interrupted_build_resumes_complete_chunks_only(self):
        with self.assertRaises(BaselineFatal):
            adapter.build(conversation(6), self.directory, FakeAPI(fail_at=3))
        self.assertFalse((self.directory / "prism_state.json").exists())
        pointer = json.loads((self.directory / "checkpoint_pointer.json").read_text())
        self.assertEqual(pointer["processed_chunks"], 2)
        api = FakeAPI()
        state = adapter.build(conversation(6), self.directory, api)
        self.assertEqual(state["stats"]["episode_count"], 6)
        self.assertEqual(sum(call["stage"] == "prism_extract" for call in api.calls), 4)

    def test_checkpoint_hash_prevents_silent_partial_graph_load(self):
        state = adapter.build(conversation(2), self.directory, FakeAPI())
        pointer = json.loads((self.directory / "checkpoint_pointer.json").read_text())
        graph = self.directory / pointer["directory"] / "graph.json"
        graph.write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
        adapter._RUNTIMES.clear()
        with self.assertRaises(RuntimeError):
            adapter.retrieve("Where did Ada visit?", state, FakeAPI())

    def test_source_dia_id_mutation_stops_before_query_model_calls(self):
        raw = conversation(2)
        state = adapter.build(raw, self.directory, FakeAPI())
        path = self.directory / "source_chunks.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        for chunk in source.values():
            for occurrence in chunk["occurrences"]:
                occurrence["dia_ids"] = ["CORRUPTED_DIA_ID"]
        path.write_text(json.dumps(source), encoding="utf-8")
        api = FakeAPI()
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            adapter.build(raw, self.directory, api)
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            adapter.retrieve("Where did Ada visit?", state, api)
        adapter._RUNTIMES.clear()
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            adapter.retrieve("Where did Ada visit?", state, api)
        self.assertEqual(api.calls, [])

    def test_completed_checkpoint_provenance_and_exports_are_all_bound(self):
        raw = conversation(2)
        state = adapter.build(raw, self.directory, FakeAPI())
        self.assertTrue(set(adapter._STATE_FILES).issubset(state["files_sha256"]))
        self.assertTrue(any(name.endswith("graph.json") for name in state["files_sha256"]))
        for name in state["files_sha256"]:
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(original + b"CORRUPTED_ARTIFACT")
            api = FakeAPI()
            try:
                with self.subTest(artifact=name):
                    with self.assertRaises((RuntimeError, json.JSONDecodeError)):
                        adapter.build(raw, self.directory, api)
                    with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                        adapter.retrieve("Where did Ada visit?", state, api)
                    adapter._RUNTIMES.clear()
                    with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                        adapter.retrieve("Where did Ada visit?", state, api)
                    self.assertEqual(api.calls, [])
            finally:
                path.write_bytes(original)
        self.assertTrue(adapter.retrieve("Where did Ada visit?", state, FakeAPI())["context"])

    def test_invalid_json_shapes_are_hard_failures(self):
        with self.assertRaises(BaselineFatal):
            adapter._validate_json("prism_extract", {}, "")
        with self.assertRaises(BaselineFatal):
            adapter._validate_json("prism_intent", {"temporal": 0.5}, "")
        with self.assertRaises(BaselineFatal):
            adapter._validate_json("prism_extract", {"episode_summary": "A summary without graph fields"}, "")
        prompt = "Snippets:\n[0] one\n[1] two"
        for response in ({}, [], [{"index": 0, "score": 5}, {"index": 0, "score": 6}]):
            with self.assertRaises(BaselineFatal):
                adapter._validate_json("prism_rerank", response, prompt)

    def test_same_conversation_query_clients_do_not_race(self):
        state = adapter.build(conversation(8), self.directory, FakeAPI())
        guard = threading.Lock()
        counter = {"active": 0, "maximum": 0}

        class ConcurrentAPI(FakeAPI):
            def generate(self, *args, **kwargs):
                with guard:
                    counter["active"] += 1
                    counter["maximum"] = max(counter["maximum"], counter["active"])
                try:
                    time.sleep(0.01)
                    return super().generate(*args, **kwargs)
                finally:
                    with guard:
                        counter["active"] -= 1

        apis = [ConcurrentAPI() for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda api: adapter.retrieve("What museum has Ada visited?", state, api), apis))
        self.assertEqual(counter["maximum"], 1)
        self.assertTrue(all(any(call["stage"] == "prism_rerank" for call in api.calls) for api in apis))
        self.assertTrue(all(result["context"] == results[0]["context"] for result in results))

    def test_prompt_identity_and_full_protocol_are_locked(self):
        native = adapter._SOURCE.load("mmem.retrieval.reranker")
        prompt = native._RERANK_PROMPT.format(question="Where?", snippets="[0] here\n[1] there")
        api = FakeAPI()
        bridge = adapter.NativeLLMBridge(api)
        bridge.chat_json(prompt, system="Return strict JSON only.")
        self.assertEqual(api.calls[0]["prompt"], prompt)
        self.assertEqual(api.calls[0]["kwargs"]["system"], "Return strict JSON only.")
        config = adapter._config()
        self.assertEqual(config.retrieval.intent_classifier_mode, "hybrid")
        self.assertTrue(config.retrieval.enable_reranking)
        self.assertEqual(config.retrieval.top_k, 10)
        self.assertEqual(config.retrieval.rerank_top_n, 5)
        self.assertEqual(config.llm.api_key, "")
        self.assertNotIn("mmem", sys.modules)


if __name__ == "__main__":
    unittest.main()
