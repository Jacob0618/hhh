"""Offline checks against pinned native stages, using fake paid providers."""
import asyncio
import copy
import json
import pickle
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import baselines_hypermem_v1 as hyper


class FakeAPI:
    def __init__(self, bad_stage=None, bad_value=None, boundary_every=None):
        self.calls, self.embeddings = [], []
        self.bad_stage, self.bad_value = bad_stage, bad_value
        self.boundary_every = boundary_every
        self.boundaries = 0

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        stage = kwargs["stage"]
        if stage == self.bad_stage:
            return self.bad_value
        if stage == "hypermem_boundary":
            self.boundaries += 1
            return {"should_end": bool(self.boundary_every and self.boundaries % self.boundary_every == 0),
                    "should_wait": False, "reasoning": "offline fixture", "confidence": 0.9, "topic_summary": "Park meeting"}
        if stage == "hypermem_episode":
            return {"title": "Park meeting", "summary": "A and B visited the park.",
                    "content": "A and B visited the park and discussed their friend and red bicycle."}
        if stage in ("hypermem_topic_create", "hypermem_topic_update"):
            return {"title": "Park visits", "summary": "A and B discussed walking in the park.", "keywords": ["park", "walk"]}
        if stage == "hypermem_topic_match":
            # Only candidate lines, excluding prompt examples.
            ids = re.findall(r"^- (topic_\d+):", prompt, re.MULTILINE)
            return {"results": [{"topic_id": identifier, "match": True, "reasoning": "same subject"} for identifier in ids]}
        if stage == "hypermem_episode_roles":
            # Count the actual formatted episode list from source, not examples.
            ids = re.findall(r"^Episode ID: (episode_\d+)", prompt, re.MULTILINE)
            return {"episode_roles": [{"episode_id": identifier, "role": "developing", "weight": 0.7} for identifier in ids],
                    "coherence_score": 0.8, "reasoning": "offline fixture"}
        if stage == "hypermem_fact_extract":
            return {"facts": [{"fact_id": "fact_1", "content": "A walked in the park with B.", "episode_ids": ["episode_1"],
                               "keywords": ["park", "walk"], "query_patterns": ["Where did A walk?"], "confidence": 0.9}],
                    "reasoning": "offline fixture"}
        if stage == "hypermem_fact_roles":
            return {"fact_roles": [{"fact_id": "fact_1", "role": "core", "weight": 0.9}],
                    "extraction_confidence": 0.9, "reasoning": "offline fixture"}
        raise AssertionError(stage)

    def embed(self, texts, query=False):
        self.embeddings.append({"texts": list(texts), "query": query})
        matrix = np.zeros((len(texts), 768), dtype=np.float64)
        for index, text in enumerate(texts):
            matrix[index, 0] = 1
            matrix[index, 1] = len(text) / 1000
            matrix[index, 2] = sum(map(ord, text)) % 41 / 100
        return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).tolist()


def conversation(count=4):
    return {"speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A" if i % 2 == 0 else "B", "dia_id": f"D1:{i+1}",
                           "text": f"We walked in the park number {i}."} for i in range(count)]}


class HyperMemAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hyper.readiness()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.work = Path(self.directory.name)

    def tearDown(self):
        hyper._INDEXES.pop(str(self.work), None)
        self.directory.cleanup()

    def test_complete_native_build_and_retrieval_offline(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        self.assertEqual(state["graph_counts"], {"episodes": 1, "topics": 1, "facts": 1, "episode_hyperedges": 1, "fact_hyperedges": 1})
        self.assertEqual([c["stage"] for c in api.calls], ["hypermem_boundary", "hypermem_boundary", "hypermem_episode", "hypermem_topic_create", "hypermem_episode_roles", "hypermem_fact_extract", "hypermem_fact_roles"])
        self.assertTrue(all(c["max_output"] == 16384 and "temperature" not in c for c in api.calls))
        self.assertTrue(all(not call["query"] for call in api.embeddings))
        before = copy.deepcopy(state)
        result = hyper.retrieve("Where did A walk?", state, api)
        self.assertIn("[Memory 1]", result["context"])
        self.assertIn("[Fact 1]", result["context"])
        self.assertNotIn("[Topic 1]", result["context"])
        self.assertEqual({entry["layer"] for entry in result["evidence"]}, {"topics", "episodes", "facts"})
        self.assertEqual(before, state)
        self.assertTrue(api.embeddings[-1]["query"])
        self.assertEqual(result["retrieval_log"]["config"]["topic_top_k"], 10)
        self.assertFalse(result["retrieval_log"]["config"]["use_reranker"])

    def test_real_session_date_and_final_episode_date(self):
        data = conversation()
        data["session_10_date_time"] = "2:00 pm on 9 May, 2023"
        data["session_10"] = [{"speaker": "A", "dia_id": "D10:1", "text": "My bike", "img_url": "a.jpg", "blip_caption": "red bicycle"}]
        # Reverse dictionary ordering to expose accidental lexicographic ingestion.
        data = dict(reversed(list(data.items())))
        state = hyper.build(data, self.work, FakeAPI())
        messages = json.loads((self.work / "source_messages.json").read_text())
        self.assertEqual([m["session"] for m in messages], ["session_1"] * 4 + ["session_10"])
        self.assertEqual(len({m["timestamp"] for m in messages[:4]}), 1)
        self.assertIn("A shared an image: red bicycle", messages[-1]["content"])
        episodes = json.loads((self.work / "episodes.json").read_text())
        self.assertTrue(episodes[-1]["timestamp"].startswith("2023-05-09T14:00"))
        self.assertEqual(episodes[-1]["original_data"][-1]["dia_id"], "D10:1")

    def test_boundary_split_update_topic_and_original_overlap(self):
        api = FakeAPI(boundary_every=5)
        state = hyper.build(conversation(10), self.work, api)
        self.assertGreater(state["graph_counts"]["episodes"], 1)
        self.assertIn("hypermem_topic_match", [c["stage"] for c in api.calls])
        self.assertIn("hypermem_topic_update", [c["stage"] for c in api.calls])
        episodes = json.loads((self.work / "episodes.json").read_text())
        first_ids = {m["dia_id"] for m in episodes[0]["original_data"]}
        second_ids = {m["dia_id"] for m in episodes[1]["original_data"]}
        self.assertTrue(first_ids & second_ids)

    def test_schema_failure_does_not_retry_or_create_default_episode(self):
        api = FakeAPI(bad_stage="hypermem_episode", bad_value={"unknown": True})
        with self.assertRaises(hyper.HyperMemError):
            hyper.build(conversation(), self.work, api)
        self.assertEqual(sum(c["stage"] == "hypermem_episode" for c in api.calls), 1)
        self.assertFalse((self.work / "state.json").exists())
        self.assertTrue((self.work / "build_failure.json").is_file())

    def test_original_validator_failure_escapes_while_true(self):
        api = FakeAPI(bad_stage="hypermem_topic_create", bad_value={"title": "", "summary": ""})
        with self.assertRaises(hyper.HyperMemError):
            hyper.build(conversation(), self.work, api)
        self.assertEqual(sum(c["stage"] == "hypermem_topic_create" for c in api.calls), 1)
        self.assertEqual(sum(c["stage"] == "hypermem_episode_roles" for c in api.calls), 0)

    def test_missing_boundary_fields_and_malformed_response_stop(self):
        for index, value in enumerate(("unknown", {}, {"should_end": False})):
            api = FakeAPI(bad_stage="hypermem_boundary", bad_value=value)
            with self.assertRaises(hyper.HyperMemError):
                hyper.build(conversation(), self.work / str(index), api)
            self.assertEqual(len(api.calls), 1)

    def test_working_vectors_reused_when_index_cache_removed(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        original = hyper.retrieve("Where did A walk?", state, api)
        # Delete only exact temporary-directory generated cache files.
        for path in (self.work / "cache").glob("*.pkl"):
            path.unlink()
        hyper._INDEXES.pop(str(self.work))
        calls_before, embeddings_before = len(api.calls), len(api.embeddings)
        # Different question reruns retrieval; documents reuse sealed work vectors.
        result = hyper.retrieve("Who walked in the park?", state, api)
        self.assertEqual(len(api.calls), calls_before)
        self.assertEqual(len(api.embeddings), embeddings_before + 1)
        self.assertEqual(result["context"], original["context"])

    def test_parallel_duplicate_question_and_completed_build_replay(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        calls = len(api.calls)
        self.assertEqual(hyper.build(conversation(), self.work, api), state)
        self.assertEqual(len(api.calls), calls)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: hyper.retrieve("Where did A walk?", state, api), range(4)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(sum(e["query"] for e in api.embeddings), 1)

    def test_bad_vectors_fail_hard(self):
        api = FakeAPI()
        api.embed = lambda *args, **kwargs: [[1.0, 0.0]]
        with self.assertRaises(hyper.HyperMemError):
            hyper.build(conversation(), self.work, api)
        self.assertFalse((self.work / "state.json").exists())

    def test_stage_checkpoint_resume_and_no_vector_export(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        (self.work / "state.json").unlink()
        before = len(api.calls)
        resumed = hyper.build(conversation(), self.work, api)
        self.assertEqual(before, len(api.calls))
        self.assertEqual(state["graph_sha256"], resumed["graph_sha256"])
        self.assertTrue(all("embedding_evidence/" not in path and "cache" not in path and "calls/" not in path
                            for path in resumed["export_paths"]))
        summary = json.loads((self.work / "embedding_evidence_summary.json").read_text())
        self.assertFalse(summary["vector_values_retained_in_results"])
        self.assertTrue(all("vectors" not in row for row in summary["calls"]))

    def test_conversations_are_parallel_and_uuid_isolation_is_deterministic(self):
        def run(index):
            api = FakeAPI()
            directory = self.work / f"conv_{index}"
            state = hyper.build(conversation(), directory, api)
            episodes = json.loads((directory / "episodes.json").read_text())
            return state, episodes[0]["event_id"], len(api.calls)
        with ThreadPoolExecutor(max_workers=3) as pool:
            states = list(pool.map(run, range(3)))
        self.assertEqual(len({item[1] for item in states}), 1)
        self.assertEqual([item[2] for item in states], [7, 7, 7])
        self.assertEqual(len({item[0]["work_dir"] for item in states}), 3)
        for state, _, _ in states:
            hyper._INDEXES.pop(state["work_dir"], None)

    def test_temporal_query_uses_question_only_and_fixed_paper_sizes(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        result = hyper.retrieve("When did A go to the park?", state, api)
        config = result["retrieval_log"]["config"]
        self.assertEqual(config["question_type"], "temporal")
        self.assertEqual((config["initial_candidates"], config["topic_top_k"], config["episode_top_k"], config["fact_top_k"]), (100, 10, 10, 30))
        self.assertEqual(len(result["embedding_evidence_summary"]), 3)
        self.assertEqual([row["cache_hit"] for row in result["embedding_evidence_summary"]], [False, True, True])

    def test_configuration_and_gold_isolation(self):
        data = conversation()
        data["qa"] = [{"question": "secret", "answer": "gold"}]
        with self.assertRaises(BaseException):
            hyper.build(data, self.work, FakeAPI())
        self.assertEqual(hyper.describe()["config"]["retrieval_config"], {"initial_candidates": 100, "topic_top_k": 10, "episode_top_k": 10, "fact_top_k": 30})

    def test_export_integrity_is_checked_even_for_cached_queries(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        question = "Where did A walk?"
        hyper.retrieve(question, state, api)
        before = (len(api.calls), len(api.embeddings))
        self.assertEqual(set(state["artifact_sha256"]), set(state["export_paths"]) - {"state.json"})
        for name in state["export_paths"]:
            path = self.work / name
            original = path.read_bytes()
            with self.subTest(artifact=name):
                try:
                    path.write_bytes(original + b"\n ")
                    with self.assertRaises(hyper.HyperMemError):
                        hyper.retrieve(question, state, api)
                    with self.assertRaises(hyper.HyperMemError):
                        hyper.build(conversation(), self.work, api)
                finally:
                    path.write_bytes(original)
        self.assertEqual(before, (len(api.calls), len(api.embeddings)))

    def test_complete_retrieval_payload_and_missing_seal_are_rejected(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        question = "Where did A walk?"
        hyper.retrieve(question, state, api)
        path = next((self.work / "retrieval").glob("*.json"))
        original = path.read_bytes()
        before = (len(api.calls), len(api.embeddings))
        for field, value in (("context", "corrupt"), ("evidence", []), ("retrieval_log", {}),
                             ("embedding_evidence_summary", []), ("question", "different")):
            with self.subTest(field=field):
                changed = json.loads(original)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(hyper.HyperMemError):
                    hyper.retrieve(question, state, api)
                path.write_bytes(original)
        hyper._seal_path(path).unlink()
        with self.assertRaises(hyper.HyperMemError):
            hyper.retrieve(question, state, api)
        self.assertEqual(before, (len(api.calls), len(api.embeddings)))

    def test_vector_state_binding_cannot_be_replaced_by_a_new_local_seal(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        path = next((self.work / "embedding_evidence").glob("document_*.json"))
        value = json.loads(path.read_text())
        value["vectors"][0] = [0.0, 1.0] + [0.0] * 766
        path.write_text(json.dumps(value), encoding="utf-8")
        # Even replacing the local cache seal cannot change the completed state.
        hyper._seal_path(path).write_text(hyper._file_sha(path), encoding="ascii")
        for cache in (self.work / "cache").glob("*.pkl"):
            cache.unlink()
        hyper._INDEXES.pop(str(self.work), None)
        before = (len(api.calls), len(api.embeddings))
        with self.assertRaises(hyper.HyperMemError):
            hyper.retrieve("Where did A walk?", state, api)
        path.unlink()
        with self.assertRaises(hyper.HyperMemError):
            hyper.retrieve("Where did A walk?", state, api)
        self.assertEqual(before, (len(api.calls), len(api.embeddings)))

    def test_rebuilt_index_must_match_original_hash(self):
        api = FakeAPI()
        state = hyper.build(conversation(), self.work, api)
        stage3 = hyper._load()["main/stage3_hypergraph_index.py"]
        original_build = stage3.build_embedding_index_for_single_conv
        def altered_build(*args, **kwargs):
            result = original_build(*args, **kwargs)
            path = self.work / "cache/hypergraph_embedding_index_conv_0.pkl"
            index = pickle.loads(path.read_bytes())
            index[0]["embedding"][0] += 0.1
            path.write_bytes(pickle.dumps(index))
            return result
        for cache in (self.work / "cache").glob("*.pkl"):
            cache.unlink()
        hyper._INDEXES.pop(str(self.work), None)
        before = (len(api.calls), len(api.embeddings))
        with patch.object(stage3, "build_embedding_index_for_single_conv", altered_build):
            with self.assertRaisesRegex(hyper.HyperMemError, "Rebuilt native index differs"):
                hyper.retrieve("Where did A walk?", state, api)
        self.assertEqual(before, (len(api.calls), len(api.embeddings)))

    def test_finished_graph_checkpoint_binds_source_before_state_rebuild(self):
        api = FakeAPI()
        hyper.build(conversation(), self.work, api)
        (self.work / "state.json").unlink()
        path = self.work / "source_messages.json"
        path.write_bytes(path.read_bytes() + b"\n ")
        before = (len(api.calls), len(api.embeddings))
        with self.assertRaises(hyper.HyperMemError):
            hyper.build(conversation(), self.work, api)
        self.assertEqual(before, (len(api.calls), len(api.embeddings)))


if __name__ == "__main__":
    unittest.main()
