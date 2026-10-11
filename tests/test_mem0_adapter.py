"""Offline behavioral checks using real Memory.add with in-memory fake providers."""

import hashlib
import json
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
platform.uname()


def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
        raise AssertionError("Offline test attempted external work: " + event)


sys.addaudithook(no_network)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mem0 import Memory
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0.memory import main as module

from jev_mem.mem0_adapter import IncompleteSessionError, Mem0SessionAdapter, SessionRecoveryRequired


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


class FakeEmbedding:
    def embed(self, text, action):
        return [0.0, 1.0]

    def embed_batch(self, texts, action):
        return [[0.0, 1.0] for _ in texts]


class FakeStore:
    def __init__(self):
        self.points = {}

    def search(self, **kwargs):
        return [row for row in self.points.values() if row.payload.get("user_id") == kwargs["filters"]["user_id"]][:10]

    def insert(self, vectors, ids, payloads):
        for memory_id, payload in zip(ids, payloads):
            self.points[memory_id] = SimpleNamespace(id=memory_id, payload=payload)

    def get(self, vector_id):
        return self.points.get(vector_id)


class FakeDB:
    def __init__(self):
        self.saved = []

    def get_last_messages(self, scope, limit):
        # Deliberately wrong data: the adapter must not depend on SQLite's
        # timestamp tie ordering or use this content in the real LLM prompt.
        return [{"role": "assistant", "content": "INCORRECT_DB_HISTORY"}]

    def save_messages(self, messages, scope):
        self.saved.append((messages, scope))

    def batch_add_history(self, rows):
        pass


def fake_memory(responses):
    memory = Memory.__new__(Memory)
    memory.config = SimpleNamespace(llm=SimpleNamespace(config={}))
    memory.api_version = "v1.1"
    memory.custom_instructions = None
    memory.llm = FakeLLM(responses)
    memory.embedding_model = FakeEmbedding()
    memory.vector_store = FakeStore()
    memory.db = FakeDB()
    return memory


class SessionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev_mem_adapter_")
        self.addCleanup(self.temp.cleanup)
        for name, replacement in {
            "lemmatize_for_bm25": lambda text: text.lower(),
            "extract_entities_batch": lambda texts: [[] for _ in texts],
            "capture_event": lambda *args, **kwargs: None,
            "detect_scale_threshold_from_add_result": lambda *args: None,
            "detect_temporal_usage_from_metadata": lambda *args: None,
            "display_first_run_notice": lambda *args: None,
        }.items():
            manager = patch.object(module, name, replacement)
            manager.start()
            self.addCleanup(manager.stop)

    def adapter(self, responses):
        memory = fake_memory(responses)
        return memory, Mem0SessionAdapter(memory, self.temp.name, current_date="2026-09-27")

    def add(self, adapter, session="session_1", messages=None, **kwargs):
        return adapter.add_session(user_id="conv-26", session_id=session,
                                   messages=messages or [{"role": "user", "content": "Alice: I moved to Paris."}],
                                   observation_date=kwargs.pop("observation_date", "2023-05-08"), **kwargs)

    def test_whole_session_original_system_prompt_and_dates(self):
        memory, adapter = self.adapter([{"memory": [{"text": "Alice moved to Paris."}]}])
        turns = [{"role": "user", "content": "Alice: I moved to Paris."},
                 {"role": "assistant", "content": "Bob: When was that?"},
                 {"role": "user", "content": "Alice: Last week."}]
        original_llm, original_store = memory.llm, memory.vector_store
        result = self.add(adapter, messages=turns, source_turn_ids=["D1:1", "D1:2", "D1:3"])
        self.assertEqual(len(original_llm.calls), 1)
        prompts = original_llm.calls[0]["messages"]
        self.assertEqual(prompts[0]["content"], ADDITIVE_EXTRACTION_PROMPT)
        for turn in turns:
            self.assertIn(turn["content"], prompts[1]["content"])
        self.assertIn("## Observation Date\n2023-05-08", prompts[1]["content"])
        self.assertIn("## Current Date\n2026-09-27", prompts[1]["content"])
        self.assertNotIn("INCORRECT_DB_HISTORY", prompts[1]["content"])
        self.assertEqual(memory.db.saved[0][0], turns)
        self.assertEqual(result["fact_records"][0]["verified_support"], [])
        self.assertEqual(result["fact_records"][0]["support_status"], "unverified")
        self.assertEqual(result["fact_records"][0]["input_scope"]["source_turn_ids"], ["D1:1", "D1:2", "D1:3"])
        self.assertIs(memory.llm, original_llm)
        self.assertIs(memory.vector_store, original_store)
        self.assertIs(module.generate_additive_extraction_prompt, generate_additive_extraction_prompt)

    def test_ordered_last_ten_prior_turns_and_original_truncation(self):
        memory, adapter = self.adapter([{"memory": []}, {"memory": []}])
        turns = [{"role": "user", "content": f"turn-{i:02d}: " + "x" * 320} for i in range(12)]
        self.add(adapter, messages=turns)
        second = self.add(adapter, session="session_2", observation_date="2023-05-09")
        prompt = memory.llm.calls[1]["messages"][1]["content"]
        history = prompt.split("## Last k Messages\n", 1)[1].split("## Recently Extracted Memories", 1)[0]
        self.assertNotIn("turn-00", history)
        self.assertNotIn("turn-01", history)
        self.assertIn(turns[-1]["content"][:300] + "...", history)
        self.assertNotIn("x" * 320, history)
        self.assertEqual([row["source_turn_id"] for row in second["history_context"]],
                         [f"session_1:turn:{i}" for i in range(2, 12)])
        self.assertLess(history.index("turn-02"), history.index("turn-11"))

    def test_success_resume_makes_no_call_and_verifies_persisted_store(self):
        memory, adapter = self.adapter([{"memory": [{"text": "Alice moved to Paris."}]}])
        first = self.add(adapter)
        resumed = Mem0SessionAdapter(memory, self.temp.name, current_date="2026-09-27")
        second = self.add(resumed)
        self.assertTrue(second["replayed"])
        self.assertEqual(first["results"], second["results"])
        self.assertEqual(len(memory.llm.calls), 1)
        memory.vector_store.points.clear()
        with self.assertRaises(SessionRecoveryRequired):
            self.add(resumed)

    def test_bad_json_is_failure_not_empty_success_and_does_not_advance(self):
        memory, adapter = self.adapter(["not a JSON response"])
        with self.assertRaises(Exception):
            self.add(adapter)
        rows = list(Path(self.temp.name).glob("*/*.json"))
        record = json.loads(rows[0].read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "requires_recovery")
        self.assertFalse(memory.db.saved)
        self.assertEqual(record["llm_calls"][0]["response"], "not a JSON response")
        with self.assertRaises(SessionRecoveryRequired):
            self.add(adapter, session="session_2")
        with self.assertRaises(SessionRecoveryRequired):
            self.add(adapter)

    def test_partial_embedding_persistence_cannot_mark_success(self):
        memory, adapter = self.adapter([{"memory": [{"text": "fact one"}, {"text": "fact two"}]}])
        memory.embedding_model.embed_batch = lambda texts, action: [[0.0, 1.0]]
        with self.assertRaises(IncompleteSessionError):
            self.add(adapter)
        record = json.loads(next(Path(self.temp.name).glob("*/*.json")).read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "requires_recovery")
        self.assertEqual(record["persistence_check"]["missing_texts"], ["fact two"])
        self.assertEqual(len(record["results"]), 1)

    def test_duplicates_remain_original_sdk_behavior(self):
        memory, adapter = self.adapter([{"memory": [{"text": "a fact"}, {"text": "a fact"}]},
                                       {"memory": [{"text": "a fact"}]}])
        first = self.add(adapter)
        second = self.add(adapter, session="session_2")
        self.assertEqual(len(first["results"]), 1)
        self.assertEqual(second["results"], [])
        self.assertEqual(second["status"], "success")
        self.assertEqual(len(memory.vector_store.points), 1)

    def test_changed_request_and_nonchronological_input_rejected(self):
        _, adapter = self.adapter([{"memory": []}])
        self.add(adapter)
        with self.assertRaises(SessionRecoveryRequired):
            self.add(adapter, observation_date="2023-05-09")
        with self.assertRaises(ValueError):
            self.add(adapter, session="session_2", observation_date="2023-05-07")


if __name__ == "__main__":
    unittest.main(verbosity=2)
