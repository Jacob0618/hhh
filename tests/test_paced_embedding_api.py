"""Offline tests of shared physical pacing and unchanged legacy request bodies."""
import copy
import json
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests

from jev_mem import paced_embedding_api as p
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest


KEYS = {"GEMINI_API_KEY": "test-gemini-placeholder", "TYPESAFE_API_KEY": "test-jev-placeholder"}
BODY = {"requests": [{"model": "models/gemini-embedding-001",
    "content": {"parts": [{"text": "all original text"}]},
    "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 768}]}
EMBEDDING = {"embeddings": [{"values": [1.] + [0.] * 767}]}


class Clock:
    def __init__(self):
        self.current = 1000.0
        self.sleeps = []
        self.hook = None

    def __call__(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += seconds
        if self.hook:
            hook, self.hook = self.hook, None
            hook()


class Response:
    def __init__(self, status=200, data=None, retry_after=None):
        self.status_code, self.ok = status, 200 <= status < 300
        self.data = data if data is not None else (EMBEDDING if self.ok else {"error": "quota"})
        self.headers = {} if retry_after is None else {"retry-after": retry_after}
        self.text = json.dumps(self.data)

    def json(self):
        return copy.deepcopy(self.data)


class Session:
    def __init__(self, responses, clock):
        self.responses, self.clock, self.calls = list(responses), clock, []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": copy.deepcopy(kwargs), "at": self.clock()})
        if not self.responses:
            raise AssertionError("Unexpected extra physical request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _process_gate_admission(path):
    # Pure local SQLite exercise: no API/Session is created in child workers.
    return p.SharedEmbeddingGate(path, interval_seconds=.025).acquire()["admitted_at_unix"]


class PhysicalPacingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jev_embedding_pacing_mock_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.gate = p.SharedEmbeddingGate(self.root / "shared.sqlite", clock=self.clock,
                                          sleeper=self.clock.sleep)
        blocked = patch("requests.Session", side_effect=AssertionError("No network Session allowed"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def api(self, name, responses, **kwargs):
        api = p.PacedEmbeddingAPI(self.root / name, keys=KEYS,
                                  gate_path=self.gate.path, **kwargs)
        api.embedding_gate = self.gate
        session = Session(responses, self.clock)
        api.local.session = session
        return api, session

    def request(self, api):
        return api.request("gemini", p.MODEL, BODY, operation=p.OPERATION,
                           stage="mem0_embedding_search")

    def test_every_physical_retry_obeys_shared_cooldown_and_body_unchanged(self):
        api, session = self.api("paced", [Response(429), Response(429), Response()])
        with patch("jev_mem.providers.time.sleep", self.clock.sleep):
            result = self.request(api)
        self.assertEqual(result, EMBEDDING)
        self.assertEqual([c["at"] for c in session.calls], [1000., 1030., 1060.])
        self.assertTrue(all(c["kwargs"]["json"] == BODY for c in session.calls))
        self.assertTrue(all(c["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
                            for c in session.calls))
        self.assertEqual(api.retries, 2)
        ledger = [json.loads(x) for x in (api.run_dir / "api_calls.jsonl").read_text().splitlines()]
        expected_hash = digest({"provider": "gemini", "model": p.MODEL,
                                "operation": p.OPERATION, "body": BODY})
        self.assertEqual([r["attempt"] for r in ledger], [1, 2, 3])
        self.assertEqual([r["status"] for r in ledger], [429, 429, 200])
        self.assertTrue(all(r["request_hash"] == expected_hash for r in ledger))
        self.assertIs(api.local.session, session)
        self.request(api)
        self.assertEqual(len(session.calls), 3)  # Exact local success is cache-only.

    def test_real_inherited_embedding_construction_matches_original_provider(self):
        api, session = self.api("paced", [Response()])
        old = ModelAPI(self.root / "legacy", keys=KEYS)
        old_session = Session([Response()], self.clock)
        old.local.session = old_session
        self.assertEqual(api.embed(["all original text"], query=True),
                         old.embed(["all original text"], query=True))
        self.assertEqual(session.calls[0]["url"], old_session.calls[0]["url"])
        self.assertEqual(session.calls[0]["kwargs"], old_session.calls[0]["kwargs"])
        self.assertIs(p.PacedEmbeddingAPI.generate, ModelAPI.generate)
        self.assertIs(p.PacedEmbeddingAPI.embed, ModelAPI.embed)
        self.assertIs(p.PacedEmbeddingAPI.score_many, ModelAPI.score_many)

    def test_shared_gate_rechecks_new_cooldown_while_waiting(self):
        other = p.SharedEmbeddingGate(self.gate.path, clock=self.clock, sleeper=self.clock.sleep)
        first = self.gate.acquire()
        self.assertEqual(first["admitted_at_unix"], 1000.)
        self.clock.hook = lambda: other.note_429(45)
        second = self.gate.acquire()
        self.assertEqual(second["admitted_at_unix"], 1046.)
        self.assertEqual(other.acquire()["admitted_at_unix"], 1049.)
        self.assertTrue(all(s <= 1 for s in self.clock.sleeps))

    def test_gate_policy_cannot_silently_diverge_between_workers(self):
        self.gate.acquire()
        other = p.SharedEmbeddingGate(self.gate.path, interval_seconds=1,
                                      clock=self.clock, sleeper=self.clock.sleep)
        with self.assertRaisesRegex(ValueError, "policy differs"):
            other.acquire()

    def test_independent_processes_share_one_physical_start_gate(self):
        path = str(self.root / "process_gate.sqlite")
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
            admissions = sorted(pool.map(_process_gate_admission, [path] * 6))
        self.assertEqual(len(admissions), 6)
        self.assertTrue(all(b - a >= .025 - 1e-6 for a, b in zip(admissions, admissions[1:])))

    def test_inherited_exact_cache_never_paces_or_dispatches(self):
        source = self.root / "source_cache"
        source.mkdir()
        signature = {"provider": "gemini", "model": p.MODEL,
                     "operation": p.OPERATION, "body": BODY}
        (source / (digest(signature) + ".json")).write_text(
            json.dumps({"request": signature, "response": EMBEDDING}), encoding="utf8")
        api, session = self.api("target", [], cache_roots=(source,))
        self.assertEqual(self.request(api), EMBEDDING)
        self.assertEqual(session.calls, [])
        self.assertFalse(self.gate.path.exists())
        self.assertFalse((api.run_dir / "embedding_pacing_events.jsonl").exists())
        self.assertEqual(api.events[0]["source"], "inherited_exact_request")

    def test_three_physical_failures_keep_intent_and_no_new_logical_retry(self):
        api, session = self.api("failed", [Response(429)] * 3)
        with patch("jev_mem.providers.time.sleep", self.clock.sleep):
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                self.request(api)
        with self.assertRaisesRegex(RuntimeError, "no implicit redispatch"):
            self.request(api)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(list((api.run_dir / "request_intents").glob("*.json"))), 1)

    def test_transport_errors_also_pace_each_original_physical_attempt(self):
        api, session = self.api("transport", [requests.ConnectionError("mock offline") for _ in range(3)])
        with patch("jev_mem.providers.time.sleep", self.clock.sleep):
            with self.assertRaisesRegex(RuntimeError, "mock offline"):
                self.request(api)
        self.assertEqual([c["at"] for c in session.calls], [1000., 1003., 1007.])
        with self.assertRaisesRegex(RuntimeError, "no implicit redispatch"):
            self.request(api)
        self.assertEqual(len(session.calls), 3)

    def test_other_operations_keep_original_transport_and_no_embedding_gate(self):
        api, session = self.api("other", [Response(data={"totalTokens": 10}),
            Response(data={"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "ok"}]}}]}),
            Response(data={"answers": {"x": {"noul": .7}}})])
        self.assertEqual(api.count_tokens("some input"), 10)
        self.assertEqual(api.generate("unchanged prompt"), "ok")
        with patch("jev_mem.providers._pace_jev") as jev_pace:
            self.assertEqual(api.score_many("state", {"x": "criterion"}), {"x": .7})
        self.assertEqual(jev_pace.call_count, 1)
        self.assertFalse(self.gate.path.exists())
        self.assertEqual(len(session.calls), 3)

    def test_scoped_api_preserves_shared_policy_and_fixed_retries(self):
        api, _ = self.api("scope", [], context={"conversation": 1})
        child = api.scoped(group_id="group_0002")
        self.assertEqual(child.gate_path, api.gate_path)
        self.assertEqual(child.interval_seconds, 3.)
        self.assertEqual(child.cooldown_seconds, 30.)
        self.assertEqual(child.cache_roots, api.cache_roots)
        self.assertEqual(child.context, {"conversation": 1, "group_id": "group_0002"})
        with self.assertRaisesRegex(ValueError, "retries=2"):
            p.PacedEmbeddingAPI(self.root / "wrong", keys=KEYS, retries=3)

    def test_pacing_log_has_no_credentials_headers_or_request_body(self):
        api, _ = self.api("safe_log", [Response(429, retry_after="35"), Response()])
        with patch("jev_mem.providers.time.sleep", self.clock.sleep):
            self.request(api)
        text = (api.run_dir / "embedding_pacing_events.jsonl").read_text(encoding="utf8")
        self.assertNotIn("all original text", text)
        for secret in KEYS.values():
            self.assertNotIn(secret, text)
        self.assertNotIn("x-goog-api-key", text)
        events = [json.loads(row) for row in text.splitlines()]
        self.assertEqual(events[1]["cooldown_until_unix"], 1035.)
        self.assertEqual(events[2]["admitted_at_unix"], 1035.)


if __name__ == "__main__":
    unittest.main()
