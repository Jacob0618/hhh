"""Meaningful offline transaction tests; all provider effects stay in temp dirs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem import group_writer as v1
from jev_mem import group_writer_v2 as v2
from jev_mem.providers import ModelAPI


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def http(value, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": json.dumps(value)}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}}


class FakeAPI:
    def __init__(self, directory, owner=None, context=None):
        self.cache_dir = Path(directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.owner = owner or self
        self.context = context or {}
        self.retries = 2
        if owner is None:
            self.responses = []
            self.requests = []
            self.count_calls = 0
            self.embed_calls = 0
            self.raise_stage = None
            self.raise_after_cache = False
            self.embed_failure = False
            self.count_failure = False

    def scoped(self, **context):
        return FakeAPI(self.cache_dir, self.owner, {**self.context, **context})

    def request(self, provider, model, body, *, operation, stage):
        signature = {"provider": provider, "model": model, "operation": operation, "body": body}
        row = {"signature": copy.deepcopy(signature), "stage": stage, "retries": self.retries,
               "context": self.context}
        self.owner.requests.append(row)
        if self.owner.raise_stage == stage and not self.owner.raise_after_cache:
            raise RuntimeError("mock uncertain transport")
        response = self.owner.responses.pop(0)
        dump(self.cache_dir / (v2.provider_request_hash(signature) + ".json"),
             {"request": signature, "response": response, "stage": stage})
        if self.owner.raise_stage == stage:
            raise RuntimeError("mock interruption after complete response cache")
        return copy.deepcopy(response)

    def count_tokens(self, text, *, model, stage):
        self.owner.count_calls += 1
        if self.owner.count_failure:
            raise RuntimeError("mock token count transport unknown")
        value = max(1, len(text) // 4)
        signature = {"provider": "gemini", "model": model, "operation": "countTokens",
                     "body": {"contents": [{"parts": [{"text": text}]}]}}
        dump(self.cache_dir / (v2.provider_request_hash(signature) + ".json"),
             {"request": signature, "response": {"totalTokens": value}, "stage": stage})
        return value

    def embed(self, texts, *, query, stage):
        self.owner.embed_calls += 1
        if self.owner.embed_failure:
            raise RuntimeError("mock embedding transport unknown")
        vectors = [[1.] + [0.] * 767 for _ in texts]
        for start in range(0, len(texts), 32):
            signature = {"provider": "gemini", "model": "gemini-embedding-001", "operation": "batchEmbedContents",
                "body": {"requests": [{"model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": text}]}, "taskType": "RETRIEVAL_DOCUMENT", "outputDimensionality": 768}
                    for text in texts[start:start + 32]]}}
            dump(self.cache_dir / (v2.provider_request_hash(signature) + ".json"), {"request": signature,
                 "response": {"embeddings": [{"values": x} for x in vectors[start:start + 32]]}, "stage": stage})
        return vectors


class WriterV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / "bank"
        self.api = FakeAPI(self.root / "cache")
        self.raw = [{"source_turn_id": "D1:" + str(i), "session_id": "s1", "speaker": "A",
                     "observation_date": "2022-01-0" + str(i + 1), "text": "Original source turn " + str(i)} for i in range(5)]
        self.members = [self.raw[2], self.raw[4]]
        self.valid = {"facts": [{"text": "Member's actual fact", "source_turn_ids": ["D1:2"], "temporal_status": "not_temporal"}]}
        self.invalid = {"facts": [{"text": "Context-only information", "source_turn_ids": ["D1:4", "D1:0"], "temporal_status": "not_temporal"}]}
        self.kwargs = {"user_id": "u", "group_id": "g", "members": self.members, "raw_turns": self.raw,
                       "grouping_method": "jev_topics", "current_date": "2026-09-27"}

    def writer(self, directory=None):
        return v2.GroupFactWriter(self.api, directory or self.store)

    def pending(self):
        return json.loads((self.store / "pending_group.json").read_text(encoding="utf-8"))

    def resume_snapshot(self, snapshot, name="resumed"):
        target = self.root / name
        target.mkdir()
        dump(target / "pending_group.json", snapshot)
        return self.writer(target).add_group(**self.kwargs)

    def captured_run(self):
        snapshots = []
        original = v2._atomic
        def capture(path, value):
            result = original(path, value)
            if path.name == "pending_group.json":
                snapshots.append(copy.deepcopy(value))
            return result
        with patch.object(v2, "_atomic", side_effect=capture):
            record = self.writer().add_group(**self.kwargs)
        return record, snapshots

    def test_first_payload_and_system_are_identical_to_v1_and_actual_provider(self):
        old = v1.prepare_group(members=self.members, raw_turns=self.raw)
        new = v2.prepare_group(members=self.members, raw_turns=self.raw)
        self.assertEqual(old, new)
        self.assertEqual(v1.SYSTEM_PROMPT, v2.SYSTEM_PROMPT)
        prompt = json.dumps(new["payload"], ensure_ascii=False, indent=2)
        api = ModelAPI(self.root / "provider", keys={"TYPESAFE_API_KEY": "mock", "GEMINI_API_KEY": "mock"})
        observed = []
        def capture(provider, model, body, **kwargs):
            observed.append({"provider": provider, "model": model, "operation": kwargs.get("operation", "generateContent"), "body": body})
            return http(self.valid)
        with patch.object(api, "request", side_effect=capture):
            api.generate(prompt, model=v1.MODEL, system=v1.SYSTEM_PROMPT, json_output=True, max_output=16384,
                         stage="group_fact_extraction")
        self.assertEqual(observed, [v2.generation_signature(prompt)])

    def test_valid_first_response_is_unchanged_and_has_no_feedback(self):
        self.api.responses = [http(self.valid)]
        result = self.writer().add_group(**self.kwargs)
        self.assertEqual(result["raw_response"], self.valid)
        self.assertEqual(result["selected_extraction"], "first")
        self.assertEqual(set(result["extraction_attempts"]), {"first"})
        self.assertEqual(len(self.api.requests), 1)
        self.assertEqual(self.api.requests[0]["retries"], 2)
        self.assertEqual(self.api.retries, 2)
        self.assertEqual(result["protocol"], v2.SCHEMA)
        self.assertEqual(v2.validate_generation_provenance(result)["feedback_attempts"], 0)
        self.assertTrue(self.writer().add_group(**self.kwargs)["replayed"])
        self.assertEqual(len(self.api.requests), 1)

    def test_invalid_then_valid_reextracts_once_with_full_input_error_and_raw(self):
        self.api.responses = [http(self.invalid), http(self.valid)]
        result = self.writer().add_group(**self.kwargs)
        self.assertEqual([x["retries"] for x in self.api.requests], [2, 0])
        self.assertEqual(self.api.retries, 2)
        self.assertEqual(result["first_raw_response"], self.invalid)
        self.assertEqual(result["feedback_raw_response"], self.valid)
        feedback = self.api.requests[1]["signature"]["body"]["contents"][0]["parts"][0]["text"]
        self.assertTrue(feedback.startswith(result["prompt"]))
        self.assertIn(json.dumps(self.invalid, ensure_ascii=False, indent=2).splitlines()[1].strip(), feedback)
        self.assertIn(result["first_validation_error"], feedback)
        self.assertIn("Never\nattach an irrelevant member", feedback)
        self.assertEqual(result["extraction_attempts"]["first"]["http_response"], http(self.invalid))
        self.assertEqual(v2.validate_generation_provenance(result)["feedback_attempts"], 1)

    def test_two_invalid_outputs_are_preserved_and_never_get_third_call(self):
        self.api.responses = [http(self.invalid), http(self.invalid)]
        for _ in range(2):
            with self.assertRaisesRegex(v2.GroupWriteError, "not associated"):
                self.writer().add_group(**self.kwargs)
        p = self.pending()
        self.assertEqual(p["status"], "terminal_feedback_validation_failure")
        self.assertEqual(p["first_raw_response"], self.invalid)
        self.assertEqual(p["feedback_raw_response"], self.invalid)
        self.assertEqual(len(self.api.requests), 2)
        self.assertFalse((self.store / "group_fact_bank.json").exists())

    def test_unknown_first_intent_cannot_restart_call(self):
        self.api.raise_stage = "group_fact_extraction"
        with self.assertRaises(RuntimeError):
            self.writer().add_group(**self.kwargs)
        self.api.raise_stage = None
        self.api.responses = [http(self.valid)]
        with self.assertRaises(v2.ExtractionUnknown):
            self.writer().add_group(**self.kwargs)
        self.assertEqual(len(self.api.requests), 1)
        self.assertEqual(self.api.embed_calls, 0)

    def test_unknown_feedback_intent_cannot_restart_call(self):
        self.api.responses = [http(self.invalid)]
        self.api.raise_stage = "group_fact_extraction_feedback_v2"
        with self.assertRaises(RuntimeError):
            self.writer().add_group(**self.kwargs)
        self.api.raise_stage = None
        self.api.responses = [http(self.valid)]
        with self.assertRaises(v2.ExtractionUnknown):
            self.writer().add_group(**self.kwargs)
        self.assertEqual(len(self.api.requests), 2)

    def test_response_cache_after_provider_interruption_allows_offline_finish(self):
        self.api.responses = [http(self.valid)]
        self.api.raise_stage = "group_fact_extraction"
        self.api.raise_after_cache = True
        with self.assertRaises(RuntimeError):
            self.writer().add_group(**self.kwargs)
        self.api.raise_stage = None
        result = self.writer().add_group(**self.kwargs)
        self.assertEqual(len(self.api.requests), 1)
        self.assertEqual(result["selected_extraction"], "first")

    def test_seeded_first_cache_reuses_original_body_without_api(self):
        prompt = json.dumps(v2.prepare_group(members=self.members, raw_turns=self.raw)["payload"], ensure_ascii=False, indent=2)
        signature = v2.generation_signature(prompt)
        dump(self.api.cache_dir / (v2.provider_request_hash(signature) + ".json"),
             {"request": signature, "response": http(self.valid), "stage": "group_fact_extraction"})
        result = self.writer().add_group(**self.kwargs)
        self.assertEqual(self.api.requests, [])
        self.assertEqual(result["extraction_attempts"]["first"]["response_source"], "validated_provider_cache_before_intent")

    def test_incomplete_or_nonjson_http_does_not_trigger_parse_feedback(self):
        for name, raw in (("incomplete", http(self.valid, finish="MAX_TOKENS")),
                          ("nonjson", {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not-json"}]}}]})):
            self.api.responses = [raw]
            self.api.cache_dir = self.root / (name + "cache")
            self.api.cache_dir.mkdir()
            before = len(self.api.requests)
            with self.assertRaises((v2.GroupWriteError, json.JSONDecodeError)):
                self.writer(self.root / name).add_group(**self.kwargs)
            self.assertEqual(len(self.api.requests), before + 1)

    def test_request_cache_tampering_blocks_before_any_model_call(self):
        prompt = json.dumps(v2.prepare_group(members=self.members, raw_turns=self.raw)["payload"], ensure_ascii=False, indent=2)
        signature = v2.generation_signature(prompt)
        corrupted = copy.deepcopy(signature)
        corrupted["body"]["contents"][0]["parts"][0]["text"] += "changed"
        dump(self.api.cache_dir / (v2.provider_request_hash(signature) + ".json"),
             {"request": corrupted, "response": http(self.valid), "stage": "group_fact_extraction"})
        with self.assertRaisesRegex(v2.GroupWriteError, "binding differs"):
            self.writer().add_group(**self.kwargs)
        self.assertEqual(self.api.requests, [])

    def test_generation_crash_snapshots_resume_without_new_requests(self):
        self.api.responses = [http(self.invalid), http(self.valid)]
        _, snapshots = self.captured_run()
        candidates = [p for p in snapshots if ("extraction_attempts" in p and "feedback" in p["extraction_attempts"])]
        # Includes feedback-intent-before-response, raw-HTTP-before-decode,
        # decoded-before-parse, and post-parse snapshots. Persisted API caches
        # permit finishing every state without a new extraction request.
        for i, p in enumerate(candidates):
            with self.subTest(snapshot=i):
                result = self.resume_snapshot(p, "snapshot" + str(i))
                self.assertEqual(v2.validate_generation_provenance(result)["feedback_attempts"], 1)
        self.assertEqual(len(self.api.requests), 2)

    def test_missing_embedding_batch_cache_does_not_restart_unknown_embed(self):
        self.api.responses = [http(self.valid)]
        self.api.embed_failure = True
        with self.assertRaises(RuntimeError):
            self.writer().add_group(**self.kwargs)
        self.api.embed_failure = False
        with self.assertRaisesRegex(v2.ExtractionUnknown, "embedding intent"):
            self.writer().add_group(**self.kwargs)
        self.assertEqual(self.api.embed_calls, 1)

    def test_embedding_snapshot_recovers_all_batches_without_embed_call(self):
        many = {"facts": [{**self.valid["facts"][0], "text": "Fact " + str(i)} for i in range(33)]}
        self.api.responses = [http(many)]
        _, snapshots = self.captured_run()
        intent = next(p for p in snapshots if "embedding_attempt" in p and "embeddings" not in p)
        result = self.resume_snapshot(intent)
        self.assertEqual(len(result["fact_records"]), 33)
        self.assertEqual(self.api.embed_calls, 1)

    def test_unknown_context_count_does_not_repeat(self):
        self.api.count_failure = True
        with self.assertRaises(RuntimeError):
            self.writer().add_group(**self.kwargs)
        self.api.count_failure = False
        with self.assertRaisesRegex(v2.ExtractionUnknown, "token count intent"):
            self.writer().add_group(**self.kwargs)
        self.assertEqual(self.api.count_calls, 1)
        self.assertEqual(self.api.requests, [])

    def test_bank_commit_before_audit_crash_replays_without_models(self):
        self.api.responses = [http(self.valid)]
        writer = self.writer()
        with patch.object(writer, "_finish", side_effect=KeyboardInterrupt("mock after bank commit")):
            with self.assertRaises(KeyboardInterrupt):
                writer.add_group(**self.kwargs)
        before = (len(self.api.requests), self.api.count_calls, self.api.embed_calls)
        result = self.writer().add_group(**self.kwargs)
        self.assertTrue(result["replayed"])
        self.assertEqual(before, (len(self.api.requests), self.api.count_calls, self.api.embed_calls))
        self.assertFalse((self.store / "pending_group.json").exists())
        self.assertTrue((self.store / "groups" / (result["request_sha256"] + ".json")).exists())

    def test_provenance_rejects_http_hash_fact_or_reason_tampering(self):
        self.api.responses = [http(self.invalid), http(self.valid)]
        result = self.writer().add_group(**self.kwargs)
        def edits(row):
            yield lambda: row["extraction_attempts"]["feedback"].__setitem__("request_hash", "bad")
            yield lambda: row["extraction_attempts"]["first"].__setitem__("http_response", http(self.valid))
            yield lambda: row["fact_records"][0].__setitem__("text", "changed")
            yield lambda: row.__setitem__("first_validation_error", "invented")
        for index in range(4):
            changed = copy.deepcopy(result)
            list(edits(changed))[index]()
            with self.assertRaises(v2.GroupWriteError):
                v2.validate_generation_provenance(changed)


if __name__ == "__main__":
    unittest.main()
