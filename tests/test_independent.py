"""Offline behavior tests for the independent fact bank; no real clients."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_mem.independent import IndependentMemory, IndependentMemoryError


def vector(first=1.0, second=0.0):
    return [first, second] + [0.0] * 766


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.generations = []
        self.embeddings = []
        self.fail_embedding_once = False
        self.documents = {}

    def generate(self, prompt, **kwargs):
        self.generations.append((json.loads(prompt), kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def embed(self, texts, **kwargs):
        self.embeddings.append((list(texts), kwargs))
        if self.fail_embedding_once:
            self.fail_embedding_once = False
            raise RuntimeError("offline simulated embedding failure")
        return [vector() if kwargs.get("query") else self.documents.get(text, vector()) for text in texts]


class IndependentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="jev_independent_test_")
        self.addCleanup(self.directory.cleanup)

    def bank(self, responses):
        api = FakeAPI(responses)
        return api, IndependentMemory(api, self.directory.name)

    def add(self, bank, session="s1", messages=None, source_ids=None, **kwargs):
        return bank.add_session(user_id=kwargs.pop("user_id", "c1"), session_id=session,
                                messages=messages or [{"role": "user", "content": "Alice: I moved to Paris."}],
                                source_turn_ids=source_ids or [session + ":1"],
                                observation_date=kwargs.pop("observation_date", "2023-01-01"), **kwargs)

    def response(self, text="Alice moved to Paris.", sources=None):
        return {"facts": [{"text": text, "source_turn_ids": sources or ["s1:1"]}]}

    def test_complete_session_and_proposed_only_sources(self):
        api, bank = self.bank([self.response(sources=["s1:1", "s1:2"])])
        messages = [{"role": "user", "content": "Alice: I moved to Paris."},
                    {"role": "assistant", "content": "Bob: Was that last week?"}]
        record = self.add(bank, messages=messages, source_ids=["s1:1", "s1:2"])
        self.assertEqual(len(api.generations), 1)
        prompt, kwargs = api.generations[0]
        self.assertEqual([row["message"] for row in prompt["new_session"]["turns"]], messages)
        self.assertEqual(prompt["new_session"]["observation_date"], "2023-01-01")
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(record["status"], "success")
        fact = record["fact_records"][0]
        self.assertEqual(fact["proposed_support"], ["s1:1", "s1:2"])
        self.assertEqual(fact["verified_support"], [])
        self.assertEqual(len(fact["embedding"]), 768)
        self.assertEqual(fact["source_records"][1]["message"]["content"], messages[1]["content"])

    def test_no_twenty_fact_write_cap(self):
        response = {"facts": [{"text": f"Alice owns item {i}.", "source_turn_ids": ["s1:1"]} for i in range(25)]}
        _, bank = self.bank([response])
        self.assertEqual(len(self.add(bank)["fact_records"]), 25)
        self.assertEqual(len(bank.get_facts()), 25)

    def test_success_replay_after_restart_has_no_calls(self):
        api, bank = self.bank([self.response()])
        first = self.add(bank)
        replacement_api = FakeAPI([])
        reopened = IndependentMemory(replacement_api, self.directory.name)
        replay = self.add(reopened)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["fact_records"], first["fact_records"])
        self.assertEqual(replacement_api.generations, [])
        self.assertEqual(replacement_api.embeddings, [])
        with self.assertRaises(IndependentMemoryError):
            self.add(reopened, messages=[{"role": "user", "content": "Different."}])

    def test_embedding_failure_does_not_commit_and_reuses_extraction(self):
        api, bank = self.bank([self.response()])
        api.fail_embedding_once = True
        with self.assertRaises(RuntimeError):
            self.add(bank)
        self.assertEqual(bank.get_facts(), [])
        self.assertTrue((Path(self.directory.name) / "pending_session.json").exists())
        reopened = IndependentMemory(api, self.directory.name)
        result = self.add(reopened)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(api.generations), 1)
        self.assertEqual(len(api.embeddings), 2)
        self.assertFalse((Path(self.directory.name) / "pending_session.json").exists())

    def test_generation_failure_can_retry_but_blocks_other_session(self):
        api, bank = self.bank([RuntimeError("offline failure"), self.response()])
        with self.assertRaises(RuntimeError):
            self.add(bank)
        with self.assertRaises(IndependentMemoryError):
            self.add(bank, session="s2")
        self.assertEqual(self.add(bank)["status"], "success")
        self.assertEqual(len(api.generations), 2)

    def test_commit_survives_audit_cleanup_failure_without_duplicate_calls(self):
        api, bank = self.bank([self.response()])
        with patch.object(bank, "_finish_committed", side_effect=OSError("simulated cleanup crash")):
            with self.assertRaises(OSError):
                self.add(bank)
        # The single bank commit already includes both the fact and checkpoint.
        self.assertEqual(len(bank.get_facts()), 1)
        reopened = IndependentMemory(api, self.directory.name)
        replay = self.add(reopened)
        self.assertTrue(replay["replayed"])
        self.assertEqual(len(api.generations), 1)
        self.assertEqual(len(api.embeddings), 1)
        self.assertEqual(len(reopened.get_facts()), 1)
        self.assertFalse((Path(self.directory.name) / "pending_session.json").exists())

    def test_forged_source_id_is_rejected_and_response_preserved(self):
        _, bank = self.bank([self.response(sources=["future:1"])])
        with self.assertRaises(IndependentMemoryError):
            self.add(bank)
        self.assertEqual(bank.get_facts(), [])
        pending = json.loads((Path(self.directory.name) / "pending_session.json").read_text(encoding="utf-8"))
        self.assertIn("raw_response", pending)
        self.assertEqual(pending["error_type"], "IndependentMemoryError")

    def test_history_last_ten_order_and_dates(self):
        api, bank = self.bank([{"facts": []}, self.response(sources=["s1:3", "s2:1"])])
        messages = [{"role": "user", "content": f"Alice: turn {i} " + "x" * 400} for i in range(12)]
        ids = [f"s1:{i}" for i in range(1, 13)]
        self.add(bank, messages=messages, source_ids=ids)
        self.add(bank, session="s2", observation_date="2023-01-02")
        history = api.generations[1][0]["last_messages"]
        self.assertEqual([row["source_turn_id"] for row in history], ids[-10:])
        self.assertEqual(history[0]["observation_date"], "2023-01-01")
        self.assertEqual(len(history[0]["message"]["content"]), 300)
        self.assertTrue(history[0]["context_truncated"])
        self.assertEqual(api.generations[0][0]["last_messages"], [])

    def test_history_outside_visible_ten_cannot_be_cited(self):
        _, bank = self.bank([{"facts": []}, self.response(sources=["s1:1", "s2:1"])])
        messages = [{"role": "user", "content": f"Alice: turn {i}"} for i in range(12)]
        self.add(bank, messages=messages, source_ids=[f"s1:{i}" for i in range(1, 13)])
        with self.assertRaises(IndependentMemoryError):
            self.add(bank, session="s2", observation_date="2023-01-02")

    def test_isolation_chronology_and_qa_rejection(self):
        _, bank = self.bank([self.response()])
        self.add(bank)
        with self.assertRaises(IndependentMemoryError):
            self.add(bank, user_id="c2", session="s2")
        with self.assertRaises(ValueError):
            self.add(bank, session="s2", observation_date="2022-12-31")
        with self.assertRaises(ValueError):
            self.add(bank, session="s2", messages=[{"role": "user", "content": "Alice: Hi", "gold": "secret"}])

    def test_bm25_can_add_candidate_outside_dense_top_k(self):
        texts = ["Alice enjoys painting.", "Bob enjoys walking.", "Xylophone sold for ten dollars."]
        response = {"facts": [{"text": text, "source_turn_ids": ["s1:1"]} for text in texts]}
        api, bank = self.bank([response])
        api.documents = {texts[0]: vector(1, 0), texts[1]: vector(.8, .6), texts[2]: vector(0, 1)}
        self.add(bank)
        results = bank.search("Xylophone", top_k=2)
        self.assertEqual(len(results), 2)
        lexical = next(row for row in results if row["text"] == texts[2])
        self.assertEqual(lexical["channel_labels"], ["bm25"])
        self.assertGreater(lexical["channel_scores"]["bm25"], 0)
        self.assertEqual(lexical["retrieval_protocol"]["union_size"], 3)
        self.assertEqual(len(lexical["embedding"]), 768)
        self.assertTrue(api.embeddings[-1][1]["query"])

    def test_zero_fact_success_never_calls_embed_or_search(self):
        api, bank = self.bank([{"facts": []}])
        self.assertEqual(self.add(bank)["fact_records"], [])
        self.assertEqual(bank.search("anything"), [])
        self.assertEqual(api.embeddings, [])

    def test_nonfinite_vectors_fail_without_committing(self):
        api, bank = self.bank([self.response()])
        api.documents["Alice moved to Paris."] = vector(float("nan"))
        with self.assertRaises(IndependentMemoryError):
            self.add(bank)
        self.assertEqual(bank.get_facts(), [])


if __name__ == "__main__":
    unittest.main()
