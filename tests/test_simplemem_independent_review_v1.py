"""Independent offline SimpleMem review; native LanceDB/Tantivy, fake models only."""
import ast
import copy
import hashlib
import io
import json
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools import baselines_simplemem_v1 as sm
from tools import run_five_baselines_gemini25_v1 as runner
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal
from tests.test_baselines_simplemem_v1 import FakeAPI, conversation


class DistinctWindowsAPI(FakeAPI):
    def generate(self, prompt, **kwargs):
        result = super().generate(prompt, **kwargs)
        if kwargs["stage"] == "simplemem_write":
            number = sum(x["stage"] == "simplemem_write" for x in self.calls)
            result[0]["lossless_restatement"] = (
                "Alice running aardvark" if number == 1 else "Alice swimming zebra")
        return result


class NativeReview(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("jev_mem.providers.load_credentials", side_effect=AssertionError("credential access forbidden")))
        self.stack.enter_context(patch("requests.Session.request", side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.assertTrue(sm.readiness()["ready"], sm.readiness())

    def tearDown(self):
        self.stack.close()

    def store(self, api):
        return sm._make_store(self.directory / "simplemem/lancedb", api,
                              sm._Audit(self.directory / "independent_audit.jsonl"))

    def test_all_writer_windows_participate_in_native_stemmed_fts(self):
        api = DistinctWindowsAPI()
        state = sm.build(conversation(42), self.directory, api)
        self.assertEqual(state["entry_count"], 2)
        store = self.store(api)
        for word in ("run", "aardvark", "swim", "zebra"):
            with self.subTest(word=word):
                self.assertEqual(len(store.keyword_search([word], top_k=5)), 1)

    def test_same_count_database_content_change_stops_before_query_generation(self):
        api = FakeAPI(notes=4)
        state = sm.build(conversation(10), self.directory, api)
        store = self.store(api)
        store.table.update(values={"lossless_restatement": "CORRUPTED SAMECOUNT"})
        calls = len(api.calls)
        with self.assertRaises(BaselineFatal):
            sm.retrieve("Alice", state, api)
        self.assertEqual(len(api.calls), calls)

    def test_real_storage_four_concurrent_queries_use_separate_providers_and_logs(self):
        state = sm.build(conversation(10), self.directory, FakeAPI(notes=4))
        apis = [FakeAPI() for _ in range(4)]
        def query(index):
            return sm.retrieve(f"Alice independent question {index}", state, apis[index])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(query, range(4)))
        self.assertEqual(len({r["request_log"] for r in results}), 4)
        self.assertEqual(len({r["context"] for r in results}), 1)
        for i, (api, result) in enumerate(zip(apis, results)):
            self.assertEqual(result["count"], 4)
            self.assertEqual([x["stage"] for x in api.calls], [
                "simplemem_requirements", "simplemem_query_plan",
                "simplemem_query_analysis", "simplemem_completeness"])
            self.assertIn(f"Alice independent question {i}", api.calls[0]["prompt"])
            records = [json.loads(x) for x in Path(result["request_log"]).read_text(encoding="utf-8").splitlines()]
            requests = [x for x in records if x["event"] == "generation_request"]
            self.assertEqual(len(requests), 4)
            self.assertIn(f"Alice independent question {i}", requests[0]["prompt"])

    def test_fts_query_failure_is_not_silently_empty_or_retried(self):
        api = FakeAPI(notes=4)
        state = sm.build(conversation(10), self.directory, api)
        backend = sm._load_native().vector_store_backend.LanceDBVectorStoreBackend
        with patch.object(backend, "keyword_search", side_effect=OSError("synthetic FTS failure")) as call:
            with self.assertRaisesRegex(BaselineFatal, "synthetic FTS failure"):
                sm.retrieve("Alice", state, api)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(list((self.directory / "simplemem/retrieval").glob("*.result.json")), [])

    def test_metadata_query_failure_is_not_silently_empty_or_retried(self):
        api = FakeAPI(notes=4)
        state = sm.build(conversation(10), self.directory, api)
        backend = sm._load_native().vector_store_backend.LanceDBVectorStoreBackend
        original = api.generate
        def with_person(prompt, **kwargs):
            result = original(prompt, **kwargs)
            if kwargs["stage"] == "simplemem_query_analysis":
                result["persons"] = ["Alice"]
            return result
        api.generate = with_person
        with patch.object(backend, "structured_search", side_effect=OSError("synthetic SQL failure")) as call:
            with self.assertRaisesRegex(BaselineFatal, "synthetic SQL failure"):
                sm.retrieve("Alice", state, api)
            self.assertEqual(call.call_count, 1)

    def test_memory_exports_survive_runner_copy_with_full_provenance(self):
        state = sm.build(conversation(42), self.directory, DistinctWindowsAPI())
        copied = runner.persist_memory(state, self.directory, self.directory / "retained_results")
        self.assertEqual(len(copied), len(state["export_paths"]))
        for original, item in zip(state["export_paths"], copied):
            self.assertEqual(Path(original).read_bytes(), Path(item["path"]).read_bytes())
            self.assertEqual(hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(), item["sha256"])
        memory = json.loads(Path(copied[0]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(memory["entries"]), 2)
        self.assertEqual(len(memory["sources"]), 42)
        windows = list(memory["window_provenance"].values())
        self.assertEqual(windows, [list(range(1, 41)), list(range(39, 43))])
        self.assertEqual(set(memory["window_provenance"]), {x["entry_id"] for x in memory["entries"]})

    def test_real_shared_transport_keeps_raw_requests_and_scoped_events_with_fake_http(self):
        """The production transport runs; its only HTTP entry point is a local fake."""
        stub = FakeAPI(notes=4)
        systems = {
            "professional information extraction assistant": "simplemem_write",
            "intelligent information requirement analyst": "simplemem_requirements",
            "query generation specialist": "simplemem_query_plan",
            "query analysis assistant": "simplemem_query_analysis",
            "information completeness evaluator": "simplemem_completeness",
        }
        physical = []
        class Response:
            status_code, ok, headers = 200, True, {}
            def __init__(self, value):
                self.value = value
            def json(self):
                return self.value
        def post(url, **kwargs):
            body = kwargs["json"]
            physical.append(copy.deepcopy(body))
            if url.endswith(":batchEmbedContents"):
                self.assertTrue(all(x["outputDimensionality"] == 768 for x in body["requests"]))
                return Response({"embeddings": [{"values": [2.0] + [0.0] * 767} for _ in body["requests"]]})
            self.assertTrue(url.endswith("gemini-2.5-flash:generateContent"))
            self.assertEqual(body["generationConfig"], {"maxOutputTokens": 16384, "thinkingConfig": {"thinkingBudget": 0}})
            system = body["systemInstruction"]["parts"][0]["text"]
            matches = [stage for phrase, stage in systems.items() if phrase in system]
            self.assertEqual(len(matches), 1)
            prompt = body["contents"][0]["parts"][0]["text"]
            generated = stub.generate(prompt, stage=matches[0])
            return Response({"modelVersion": "gemini-2.5-flash", "candidates": [{"finishReason": "STOP",
                "content": {"parts": [{"text": json.dumps(generated)}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}})
        folder = self.directory / "transport"
        api = BaselineAPI(folder / "work/api_method", keys={"GEMINI_API_KEY": "unit-test-key-only"},
                          gate_path=folder / "fake_gate.sqlite",
                          context={"method": "simplemem", "conversation": 0, "phase": "build"})
        with patch("requests.Session.post", side_effect=post):
            state = sm.build(conversation(10), folder / "work/native", api)
            query_api = api.scoped(query_id="independent-q0", phase="retrieval")
            result = sm.retrieve("Alice", state, query_api)
            self.assertEqual(result["count"], 4)
            self.assertEqual(len(api.events), 2)
            self.assertEqual(len(query_api.events), 5)
            self.assertTrue(all(e["context"]["phase"] == "build" for e in api.events))
            self.assertTrue(all(e["context"]["query_id"] == "independent-q0" for e in query_api.events))
            repeated = api.scoped(query_id="independent-q1", phase="retrieval")
            sm.retrieve("Alice", state, repeated)
            self.assertEqual(len(repeated.events), 5)
            self.assertTrue(all(e["source"] == "existing_run_cache" for e in repeated.events))
        self.assertEqual(len(physical), 7)
        audit = runner.audit_requests(folder)
        self.assertEqual(audit["status"], "passed", audit)
        self.assertEqual(len(audit["inventory"]), 7)
        logical = [json.loads(x) for x in (folder / "work/api_method/logical_events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(logical), 12)


class SourceReview(unittest.TestCase):
    def test_normal_function_bodies_equal_pinned_source_except_error_policy(self):
        """Compare every function AST, independently erasing only exception bodies."""
        class EraseExceptionBody(ast.NodeTransformer):
            def visit_ExceptHandler(self, node):
                node.body = [ast.Pass()]
                return node
        checked = 0
        for relative in sm.SOURCE_HASHES:
            if relative.endswith("config_default.py"):
                continue
            path = sm.SOURCE_ROOT / relative
            original = ast.parse(path.read_text(encoding="utf-8"))
            transformed = sm._SourceBridge().visit(copy.deepcopy(original))
            left = [node for node in ast.walk(original) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            right = [node for node in ast.walk(transformed) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                with self.subTest(source=relative, function=a.name):
                    self.assertEqual(ast.dump(EraseExceptionBody().visit(a), include_attributes=False),
                                     ast.dump(EraseExceptionBody().visit(b), include_attributes=False))
                checked += 1
        self.assertGreater(checked, 50)

    def test_locked_cohort_numeric_sessions_equal_chronological_order(self):
        from jev_mem.data import date_value
        source = runner.ROOT / "data/locomo10.json"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), runner.DATA_SHA)
        data = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 10)
        total = 0
        for item in data:
            conv = runner.conversation_only(item)
            keys = [k for k in conv if re.fullmatch(r"session_\d+", k)]
            numeric = sorted(keys, key=lambda k: int(k.split("_")[1]))
            chronological = sorted(keys, key=lambda k: (date_value(conv[k + "_date_time"]), int(k.split("_")[1])))
            self.assertEqual(numeric, chronological)
            dialogues, sources = sm._dialogues(conv)
            total += len(dialogues)
            self.assertEqual(len(dialogues), len(sources))
            self.assertTrue(all("qa" not in x and "answer" not in x for x in sources))
        self.assertEqual(total, 5882)


if __name__ == "__main__":
    unittest.main()
