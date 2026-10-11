"""Independent HyperMem preflight: fake providers, real pinned native stages."""
import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from tests.test_baselines_hypermem_v1 import FakeAPI, conversation
from tools import baselines_hypermem_v1 as hyper


class IndependentHyperMemReview(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.work = Path(self.directory.name)
        self.api = FakeAPI()
        self.state = hyper.build(conversation(), self.work, self.api)

    def tearDown(self):
        hyper._INDEXES.pop(str(self.work), None)
        self.directory.cleanup()

    def test_source_provenance_corruption_rejected_before_provider(self):
        path = self.work / "source_messages.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value[0]["dia_id"] = "CORRUPTED_DIA_ID"
        path.write_text(json.dumps(value), encoding="utf-8")
        before = (len(self.api.calls), len(self.api.embeddings))
        with self.assertRaises(hyper.HyperMemError):
            hyper.build(conversation(), self.work, self.api)
        with self.assertRaises(hyper.HyperMemError):
            hyper.retrieve("Where did A walk?", self.state, self.api)
        self.assertEqual(before, (len(self.api.calls), len(self.api.embeddings)))

    def test_each_permanent_export_corruption_rejected(self):
        for relative in self.state["export_paths"]:
            if relative == "state.json":
                continue  # The returned state is the trusted binding root.
            path = self.work / relative
            original = path.read_bytes()
            with self.subTest(relative=relative):
                try:
                    path.write_bytes(original + b"\n ")
                    with self.assertRaises(hyper.HyperMemError):
                        hyper.retrieve("Where did A walk?", self.state, self.api)
                finally:
                    path.write_bytes(original)

    def test_modified_cached_context_rejected(self):
        question = "Where did A walk?"
        hyper.retrieve(question, self.state, self.api)
        path = next((self.work / "retrieval").glob("*.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["context"] = "CORRUPTED_RETRIEVAL_CONTEXT"
        path.write_text(json.dumps(value), encoding="utf-8")
        before = (len(self.api.calls), len(self.api.embeddings))
        with self.assertRaises(hyper.HyperMemError):
            hyper.retrieve(question, self.state, self.api)
        self.assertEqual(before, (len(self.api.calls), len(self.api.embeddings)))

    def test_valid_shaped_modified_document_vector_rejected_on_rebuild(self):
        path = next((self.work / "embedding_evidence").glob("document_*.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["vectors"][0] = [0.0, 1.0] + [0.0] * 766
        path.write_text(json.dumps(value), encoding="utf-8")
        # Delete only generated files in this test's exact temporary directory.
        for path in (self.work / "cache").glob("*.pkl"):
            path.unlink()
        hyper._INDEXES.pop(str(self.work), None)
        before = (len(self.api.calls), len(self.api.embeddings))
        with self.assertRaises(hyper.HyperMemError):
            hyper.retrieve("Who walked in the park?", self.state, self.api)
        self.assertEqual(before, (len(self.api.calls), len(self.api.embeddings)))

    def test_four_distinct_queries_share_readonly_native_graph(self):
        cached_graph = hyper._INDEXES[str(self.work)][1]
        before_graph = copy.deepcopy(cached_graph)
        questions = ["Where did A walk?", "When did A walk?", "Who visited the park?", "What was discussed?"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda question: hyper.retrieve(question, self.state, self.api), questions))
        self.assertEqual(before_graph, cached_graph)
        self.assertEqual([result["question"] for result in results], questions)
        self.assertEqual(sum(row["query"] for row in self.api.embeddings), 4)
        self.assertTrue(all(result["config"]["use_reranker"] is False for result in results))

    def test_complete_context_matches_direct_native_formatter(self):
        question = "Where did A walk?"
        actual = hyper.retrieve(question, self.state, self.api)
        _, graph, bm25, vectors = hyper._INDEXES[str(self.work)]
        stage4 = hyper._load()["main/stage4_hypergraph_retrieval.py"]
        result = stage4.hierarchical_retrieval(
            question, graph, bm25=bm25["bm25"], docs=bm25["docs"],
            emb_index=vectors, embedding_provider=hyper._Embedding(self.api, self.work, query=True),
            reranker_provider=None, config=SimpleNamespace(**copy.deepcopy(hyper.CONFIG)))
        expected = stage4.format_hierarchical_results(result, "A", "B", "011")
        self.assertEqual(expected, actual["context"])
        self.assertEqual(len(actual["evidence"]), sum(len(result[key]) for key in ("topics", "episodes", "facts")))

    def test_unknown_generation_failure_is_terminal_across_build_resume(self):
        calls = []
        class UnknownAPI(FakeAPI):
            def generate(self, *args, **kwargs):
                calls.append(kwargs["stage"])
                raise TimeoutError("unknown transport outcome fixture")
        directory = self.work / "unknown_build"
        for _ in range(2):
            with self.assertRaises(hyper.HyperMemError):
                hyper.build(conversation(), directory, UnknownAPI())
        self.assertEqual(calls, ["hypermem_boundary"])
        self.assertFalse((directory / "state.json").exists())

    def test_unknown_query_embedding_failure_is_terminal_across_resume(self):
        calls = []
        class UnknownAPI(FakeAPI):
            def embed(self, *args, **kwargs):
                calls.append(kwargs["query"])
                raise TimeoutError("unknown embedding outcome fixture")
        for _ in range(2):
            with self.assertRaises(hyper.HyperMemError):
                hyper.retrieve("Unknown query fixture?", self.state, UnknownAPI())
        self.assertEqual(calls, [True])
        self.assertFalse(list((self.work / "retrieval").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
