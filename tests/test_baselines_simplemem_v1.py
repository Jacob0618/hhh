"""Offline pipeline tests: official SimpleMem classes, fake provider/backend only."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import baselines_simplemem_v1 as sm
from tools.baselines_common_v1 import BaselineFatal


def conversation(turns=42):
    result = {"speaker_a": "Alice", "speaker_b": "Bob"}
    boundary = min(20, turns)
    for sid, numbers in ((1, range(boundary)), (2, range(boundary, turns))):
        result[f"session_{sid}_date_time"] = f"2024-01-0{sid}T12:00:00"
        result[f"session_{sid}"] = [
            {"speaker": "Alice" if i % 2 == 0 else "Bob", "text": f"Utterance number {i}.",
             "dia_id": f"D{sid}:{i}"} for i in numbers]
    return result


class FakeAPI:
    def __init__(self, notes=1, incomplete=False):
        self.notes, self.incomplete = notes, incomplete
        self.calls, self.embeddings = [], []
        self.failure = None

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.failure:
            raise self.failure
        stage = kwargs["stage"]
        if stage == "simplemem_write":
            return [{"lossless_restatement": f"Alice memory {i}", "keywords": ["Alice"],
                     "persons": ["Alice"], "entities": [], "timestamp": "2024-01-01T12:00:00"}
                    for i in range(self.notes)]
        if stage == "simplemem_requirements":
            return {"required_info": [{"info_type": "fact", "description": "Alice", "priority": "high"}],
                    "minimal_queries_needed": 1}
        if stage == "simplemem_query_plan":
            return {"queries": ["Alice"]}
        if stage == "simplemem_query_analysis":
            return {"keywords": ["Alice"], "persons": [], "entities": [], "location": None,
                    "time_expression": None}
        if stage == "simplemem_completeness":
            return {"assessment": "incomplete" if self.incomplete else "complete"}
        if stage == "simplemem_missing_queries":
            return {"targeted_queries": ["Alice additional evidence"]}
        raise AssertionError(stage)

    def embed(self, texts, *, query=False):
        self.embeddings.append({"texts": texts, "query": query})
        return [[2.0] + [0.0] * 767 for _ in texts]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.api = FakeAPI()
        self.banks = {}
        self.native = sm._load_native()
        self.stack = []
        self.stack.append(patch.object(sm, "_preflight", lambda: None))
        self.stack.append(patch.object(sm, "_make_store", self.make_store))
        for mocked in self.stack:
            mocked.start()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        for mocked in reversed(self.stack):
            mocked.stop()
        self.temp.cleanup()

    def make_store(self, directory, api, audit):
        rows = self.banks.setdefault(str(directory), [])
        Result = self.native.vector_store_backend.VectorStoreSearchResult

        class Backend:
            _fts_initialized = False

            @property
            def table(self):
                return SimpleNamespace(to_arrow=lambda: SimpleNamespace(to_pylist=lambda: [
                    {"entry_id": row.entry_id, **row.metadata, "vector": list(row.vector)} for row in rows]))

            def _init_fts_index(self):
                self._fts_initialized = True

            def insert(self, records):
                rows.extend(records)

            def count(self):
                return len(rows)

            def get_all(self):
                return [Result(entry_id=r.entry_id, metadata=r.metadata) for r in rows]

            def semantic_search(self, vector, top_k):
                return self.get_all()[:top_k]

            def keyword_search(self, keywords, top_k):
                return self.get_all()[-top_k:]

            def structured_search(self, **kwargs):
                return self.get_all()[:kwargs.get("top_k", 5)]

        return self.native.vector_store.VectorStore(
            db_path=str(directory), embedding_model=sm._EmbeddingBridge(api, audit),
            backend_factory=lambda dimension: Backend())

    def test_native_window_overlap_and_previous_context(self):
        state = sm.build(conversation(), self.directory, self.api)
        writers = [x for x in self.api.calls if x["stage"] == "simplemem_write"]
        self.assertEqual(len(writers), 2)
        self.assertIn("Utterance number 39.", writers[0]["prompt"])
        self.assertNotIn("Utterance number 40.", writers[0]["prompt"])
        self.assertIn("Utterance number 38.", writers[1]["prompt"])
        self.assertIn("Utterance number 41.", writers[1]["prompt"])
        self.assertNotIn("Utterance number 37.", writers[1]["prompt"])
        self.assertIn("Previous Window Memory Entries", writers[1]["prompt"])
        self.assertEqual(state["turn_count"], 42)
        self.assertEqual(state["entry_count"], 2)
        self.assertTrue(all(Path(p).exists() for p in state["export_paths"]))
        for call in self.api.calls:
            self.assertNotIn("temperature", call)
            self.assertEqual(call["max_output"], 16384)
            self.assertTrue(call["stage"].startswith("simplemem_"))
        vectors = next(iter(self.banks.values()))[0].vector
        self.assertEqual(vectors[0], 1.0)

    def test_planning_reflection_and_no_reader_or_top20(self):
        self.api.notes = 30
        state = sm.build(conversation(10), self.directory, self.api)
        before = len(self.api.calls)
        result = sm.retrieve("Alice", state, self.api)
        self.assertEqual([x["stage"] for x in self.api.calls[before:]], [
            "simplemem_requirements", "simplemem_query_plan", "simplemem_query_analysis", "simplemem_completeness"])
        self.assertEqual(result["count"], 30)
        self.assertIn("[Context 30]", result["context"])
        self.assertEqual(len(result["evidence"]), 30)
        self.assertTrue(result["evidence"][0]["source_dia_ids"])
        self.assertEqual([x["query"] for x in self.api.embeddings], [False, True])

    def test_both_reflection_rounds_keep_native_call_count(self):
        self.api.incomplete = True
        state = sm.build(conversation(10), self.directory, self.api)
        before = len(self.api.calls)
        sm.retrieve("Alice", state, self.api)
        calls = [x["stage"] for x in self.api.calls[before:]]
        self.assertEqual(len(calls), 7)
        self.assertEqual(calls.count("simplemem_completeness"), 2)
        self.assertEqual(calls.count("simplemem_missing_queries"), 2)

    def test_source_gold_rejected_before_model_call(self):
        raw = conversation()
        raw["qa"] = [{"answer": "leaked"}]
        with self.assertRaises(BaselineFatal):
            sm.build(raw, self.directory, self.api)
        self.assertEqual(self.api.calls, [])

    def test_schema_error_hard_stops_without_retry_or_empty_state(self):
        self.api.generate = lambda *args, **kwargs: "not json"
        with self.assertRaises(BaselineFatal):
            sm.build(conversation(10), self.directory, self.api)
        self.assertFalse((self.directory / "simplemem/state.json").exists())
        with self.assertRaisesRegex(BaselineFatal, "Partial"):
            sm.build(conversation(10), self.directory, FakeAPI())

    def test_provider_exception_not_swallowed_or_retried(self):
        self.api.failure = BaselineFatal("resource exhausted")
        with self.assertRaisesRegex(BaselineFatal, "resource exhausted"):
            sm.build(conversation(10), self.directory, self.api)
        self.assertEqual(len(self.api.calls), 1)
        self.assertFalse((self.directory / "simplemem/state.json").exists())

    def test_unknown_completeness_hard_stops(self):
        state = sm.build(conversation(10), self.directory, self.api)
        native_generate = self.api.generate
        def bad(prompt, **kwargs):
            if kwargs["stage"] == "simplemem_completeness":
                return {"assessment": "maybe"}
            return native_generate(prompt, **kwargs)
        self.api.generate = bad
        with self.assertRaisesRegex(BaselineFatal, "Unknown completeness"):
            sm.retrieve("Alice", state, self.api)

    def test_concurrent_equal_queries_keep_separate_evidence_files(self):
        from concurrent.futures import ThreadPoolExecutor
        state = sm.build(conversation(10), self.directory, self.api)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: sm.retrieve("Alice", state, self.api), range(4)))
        self.assertEqual(len({x["request_log"] for x in results}), 4)
        self.assertEqual(len({x["context"] for x in results}), 1)

    def test_completed_bank_reuse_has_no_extra_calls_and_tamper_rejected(self):
        state = sm.build(conversation(10), self.directory, self.api)
        before = len(self.api.calls)
        self.assertEqual(sm.build(conversation(10), self.directory, self.api), state)
        self.assertEqual(len(self.api.calls), before)
        Path(state["memory_path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(BaselineFatal, "missing or changed"):
            sm.retrieve("Alice", state, self.api)
        self.assertEqual(len(self.api.calls), before)

    def test_embedding_contract_failure_hard_stops(self):
        self.api.embed = lambda texts, query=False: [[0.0] * 768 for _ in texts]
        with self.assertRaisesRegex(BaselineFatal, "zero embedding"):
            sm.build(conversation(10), self.directory, self.api)
        self.assertFalse((self.directory / "simplemem/state.json").exists())

    def test_pinned_native_prompt_functions_remain_identical(self):
        builder = self.native.memory_builder.MemoryBuilder(None, None, enable_parallel_processing=False)
        prompt = builder._build_extraction_prompt("RAW SOURCE", [1], "PREVIOUS")
        self.assertIn("RAW SOURCE", prompt)
        self.assertIn("PREVIOUS", prompt)
        self.assertIn("Generate enough memory entries", prompt)
        self.assertEqual(sm.SETTINGS["SEMANTIC_TOP_K"], 25)
        self.assertEqual(sm.SETTINGS["KEYWORD_TOP_K"], 5)
        self.assertEqual(sm.SETTINGS["STRUCTURED_TOP_K"], 5)


@unittest.skipUnless(sm.readiness()["ready"], "Isolated native-storage dependencies not available")
class NativeStorageTests(unittest.TestCase):
    def test_real_lancedb_build_and_three_retrieval_paths_without_network(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            api = FakeAPI(notes=4)
            state = sm.build(conversation(10), Path(directory), api)
            result = sm.retrieve("Alice", state, api)
            self.assertEqual(state["entry_count"], 4)
            self.assertEqual(result["count"], 4)
            store = sm._make_store(Path(state["work_dir"]) / "lancedb", api,
                                   sm._Audit(Path(directory) / "offline-check.jsonl"))
            self.assertEqual(len(store.keyword_search(["Alice"], top_k=2)), 2)
            self.assertEqual(len(store.structured_search(persons=["Alice"], top_k=2)), 2)

    def test_same_count_changed_rows_block_build_reuse_without_more_calls(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            api = FakeAPI(notes=4)
            data = conversation(10)
            state = sm.build(data, Path(directory), api)
            store = sm._make_store(Path(state["work_dir"]) / "lancedb", api,
                                   sm._Audit(Path(directory) / "offline-check.jsonl"))
            store.table.update(values={"lossless_restatement": "changed same-count memory"})
            before = (len(api.calls), len(api.embeddings))
            with self.assertRaisesRegex(BaselineFatal, "database content differs"):
                sm.build(data, Path(directory), api)
            self.assertEqual((len(api.calls), len(api.embeddings)), before)

    def test_changed_vectors_block_retrieval_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            api = FakeAPI(notes=4)
            state = sm.build(conversation(10), Path(directory), api)
            store = sm._make_store(Path(state["work_dir"]) / "lancedb", api,
                                   sm._Audit(Path(directory) / "offline-check.jsonl"))
            store.table.update(values={"vector": [0.0, 1.0] + [0.0] * 766})
            before = (len(api.calls), len(api.embeddings))
            with self.assertRaisesRegex(BaselineFatal, "database content differs"):
                sm.retrieve("Alice", state, api)
            self.assertEqual((len(api.calls), len(api.embeddings)), before)

    def test_source_and_plaintext_exports_are_bound_before_retrieval(self):
        for filename in ("source.json", "memory.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                api = FakeAPI(notes=1)
                state = sm.build(conversation(10), Path(directory), api)
                (Path(state["work_dir"]) / filename).write_text("changed", encoding="utf-8")
                before = (len(api.calls), len(api.embeddings))
                with self.assertRaisesRegex(BaselineFatal, "export missing or changed"):
                    sm.retrieve("Alice", state, api)
                self.assertEqual((len(api.calls), len(api.embeddings)), before)


if __name__ == "__main__":
    unittest.main()
