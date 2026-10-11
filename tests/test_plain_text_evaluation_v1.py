"""Offline protocol tests: dummy credentials and no network or model calls."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from jev_mem import evaluation as old
from jev_mem import evaluation_plain_text_v1 as ev
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI, check_plain_text_request
from jev_mem.paced_embedding_api import PacedEmbeddingAPI
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest


KEYS = {"GEMINI_API_KEY": "offline-gemini-placeholder", "TYPESAFE_API_KEY": "offline-jev-placeholder"}
FACTS = [{"id": "actual-fact-17", "text": "Ada visited Paris.", "observation_date": "2026-09-28"}]
FROZEN = {
    "jev_mem/evaluation.py": "fbfd4f8f5c7b6a8c0f2000696d42e0dfb39b299e0d13c529b5dfeb3ee3299afc",
    "jev_mem/providers.py": "34d0f9cc3c3607e41c8b69b854d65d012e3de78078f6a9d38605f2b760be1dd6",
    "jev_mem/sdk20_cached_api.py": "be4684c0cf035c6dca74bbf418eba166830201001d6872f103d24c0bc4c09b16",
    "jev_mem/paced_embedding_api.py": "148f19fd8018c4ea6d3b855bae88cec64fba30550b2b3362683da1d1330e1582",
    "tools/run_session_experiments.py": "a52eef3f5a4552c3a67546584aedc5c8acd1cd1d04764c22fdb68ab24d1ddf30",
    "tools/run_mem0_regroup_sdk20.py": "93a914fb258066378a21141a7c05a5ac36ec5d34d8b062e6c71a529f9e1ff50d",
}


def response(text, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}


class CaptureAPI:
    generate = ModelAPI.generate

    def __init__(self, outputs, guarded=True):
        self.outputs, self.calls, self.guarded = list(outputs), [], guarded

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified"):
        if self.guarded:
            check_plain_text_request(provider, model, body, operation=operation, stage=stage)
        self.calls.append({"provider": provider, "model": model, "operation": operation,
                           "body": copy.deepcopy(body), "stage": stage})
        return self.outputs.pop(0)


class MockResponse:
    status_code, ok, headers = 200, True, {}

    def __init__(self, text):
        self.data = response(text)

    def json(self):
        return copy.deepcopy(self.data)


class Session:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def post(self, url, **kwargs):
        self.calls.append((url, copy.deepcopy(kwargs)))
        if not self.outputs:
            raise AssertionError("Unexpected extra model dispatch")
        return MockResponse(self.outputs.pop(0))


class PlainTextTests(unittest.TestCase):
    @classmethod
    def frozen_hashes(cls):
        root = Path(__file__).resolve().parents[1]
        return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in FROZEN}

    @classmethod
    def setUpClass(cls):
        if cls.frozen_hashes() != FROZEN:
            raise AssertionError("Frozen legacy source differs before tests")

    @classmethod
    def tearDownClass(cls):
        if cls.frozen_hashes() != FROZEN:
            raise AssertionError("Frozen legacy source changed during tests")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jev_plain_text_mock_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for target in ("requests.Session.request", "requests.Session.post", "socket.create_connection",
                       "jev_mem.providers.load_credentials"):
            blocker = patch(target, side_effect=AssertionError("Offline test: no network or credentials"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def read(self, api):
        return ev.read_answer(api, "Where did Ada visit?", "REFERENCE_SECRET", FACTS,
                              reference_date="2026-09-29")

    def api(self, outputs):
        api = PlainTextEvaluationAPI(self.root / "api", keys=KEYS, gate_path=self.root / "gate.sqlite")
        session = Session(outputs)
        api.local.session = session
        return api, session

    def test_reader_preserves_prose_multiline_and_json_looking_text(self):
        for text in ("Paris", "  Paris\n", "- Paris\n- Rome", '{"answer": ["Paris", "Rome"]}'):
            with self.subTest(text=text):
                api = CaptureAPI([response(text)])
                value = self.read(api)
                self.assertEqual(value["prediction"]["answer"], text)
                self.assertEqual(value["prediction"]["raw"], text)
                self.assertIsNone(value["prediction"]["used_fact_ids"])
                self.assertIsNone(value["prediction"]["used_local_fact_ids"])
                self.assertIsNone(value["source_evidence_metrics"])
                self.assertEqual(value["token_f1"], old.lexical_metrics(text, "REFERENCE_SECRET")["token_f1"])
                self.assertNotIn("REFERENCE_SECRET", json.dumps(api.calls[0]["body"]))

    def test_reader_semantics_and_input_are_preserved(self):
        self.assertEqual(ev.READER_SEMANTICS, old.READER_SYSTEM.split("\nReturn JSON with answer", 1)[0])
        self.assertEqual(ev.READER_SYSTEM, ev.READER_SEMANTICS + "\nReturn only the plain-text answer.")
        api = CaptureAPI([response("Paris")])
        self.read(api)
        self.assertEqual(json.loads(api.calls[0]["body"]["contents"][0]["parts"][0]["text"]),
                         {"question": "Where did Ada visit?", "conversation_reference_date": "2026-09-29",
                          "facts": [{"id": "f1", "observation_date": "2026-09-28", "fact": "Ada visited Paris."}]})

    def test_judge_exact_user_prompt(self):
        expected = (
            "I will give you a question, a reference answer, and a response from a model. "
            "Please answer [[yes]] if the response contains the reference answer. Otherwise, answer [[no]]. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the reference answer, you should also answer [[yes]]. If the response only contains "
            "a subset of the information required by the answer, answer [[no]].\n\n"
            "[User Question]\nQ\n\n[The Start of Reference Answer]\nG\n[The End of Reference Answer]\n\n"
            "[The Start of Model’s Response]\n  R\n\n[The End of Model’s Response]\n\n"
            "Is the model response correct? Answer [[yes]] or [[no]] only.")
        self.assertEqual(ev.make_judge_prompt("Q", "G", "  R\n"), expected)
        api = CaptureAPI([response("[[yes]]")])
        ev.judge_answer(api, "Q", "G", "  R\n")
        body = api.calls[0]["body"]
        self.assertEqual(body["contents"][0]["parts"][0]["text"], expected)
        self.assertNotIn("systemInstruction", body)

    def test_judge_labels_and_only_real_fields(self):
        for raw, value in (("[[yes]]", True), ("\n [[no]] \t", False)):
            answer = ev.judge_answer(CaptureAPI([response(raw)]), "Q", "G", "R")
            self.assertEqual(answer, {"correct": value, "label": raw.strip(), "raw": raw})

    def test_judge_invalid_labels_raise_not_false(self):
        for raw in ("yes", "[[YES]]", "[[yes]] because...", '"[[yes]]"', "[[yes]] [[no]]",
                    '{"correct": true}', "the answer is no"):
            with self.subTest(raw=raw), self.assertRaises(ev.InvalidJudgeLabel) as found:
                ev.judge_answer(CaptureAPI([response(raw)]), "Q", "G", "R")
            self.assertEqual(found.exception.raw_response, raw)

    def test_reader_result_survives_judge_failure(self):
        api = CaptureAPI([response("Paris"), response("yes")])
        result = ev.read_answer(api, "Q", "Paris", FACTS, reference_date="2026-09-29")
        self.assertEqual(result["token_f1"], 1.0)
        with self.assertRaises(ev.InvalidJudgeLabel):
            ev.judge_answer(api, "Q", "Paris", result["prediction"]["answer"])
        self.assertEqual(result["prediction"]["answer"], "Paris")
        self.assertEqual(result["token_f1"], 1.0)
        self.assertNotIn("judgment", result)

    def test_original_generation_failures_are_technical(self):
        for raw in (response(""), response("  \n"), response("Paris", "MAX_TOKENS")):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.read(CaptureAPI([raw]))
        class NonString:
            def generate(self, *args, **kwargs):
                return ["Paris"]
        with self.assertRaises(ValueError):
            self.read(NonString())
        with self.assertRaises(ev.InvalidJudgeLabel):
            ev.judge_answer(NonString(), "Q", "G", "R")

    def test_budget_and_duplicates_reject_before_dispatch(self):
        for facts in (FACTS * 2, [{"id": str(i), "text": "x"} for i in range(21)], [{"id": 1, "text": "x"}]):
            api = CaptureAPI([])
            with self.assertRaises(ValueError):
                ev.read_answer(api, "Q", "G", facts, reference_date=None)
            self.assertEqual(api.calls, [])

    def test_configs_no_json_format_and_generation_is_inherited(self):
        api = CaptureAPI([response("Paris"), response("[[yes]]")])
        value = ev.answer_and_judge(api, "Q", "Paris", FACTS, reference_date=None)
        self.assertEqual(value["evaluation_protocol"], ev.PROTOCOL)
        for call, model, stage in zip(api.calls, (ev.READER_MODEL, ev.JUDGE_MODEL), (ev.READER_STAGE, ev.JUDGE_STAGE)):
            config = {"temperature": 0, "maxOutputTokens": 2048}
            if model == ev.READER_MODEL:
                config["thinkingConfig"] = {"thinkingBudget": 0}
            self.assertEqual(call["body"]["generationConfig"], config)
            self.assertEqual((call["model"], call["stage"]), (model, stage))
        self.assertIs(PlainTextEvaluationAPI.generate, ModelAPI.generate)
        self.assertIs(PlainTextEvaluationAPI.scoped, PacedEmbeddingAPI.scoped)

    def test_guard_rejects_protocol_drift_before_request(self):
        capture = CaptureAPI([response("Paris")])
        self.read(capture)
        valid = capture.calls[0]
        variants = []
        for name, val in (("responseMimeType", "application/json"), ("responseSchema", {"type": "STRING"}),
                          ("maxOutputTokens", 8192), ("temperature", 1), ("thinkingConfig", {"thinkingBudget": 64})):
            bad = copy.deepcopy(valid)
            bad["body"]["generationConfig"][name] = val
            variants.append(bad)
        for name, val in (("model", ev.JUDGE_MODEL), ("provider", "jev"), ("operation", "countTokens"),
                          ("stage", "qa_reader"), ("stage", "strict_judge")):
            variants.append({**copy.deepcopy(valid), name: val})
        bad = copy.deepcopy(valid)
        bad["body"]["systemInstruction"]["parts"][0]["text"] = old.READER_SYSTEM
        variants.append(bad)
        bad = copy.deepcopy(valid)
        bad["body"]["responseSchema"] = {"type": "STRING"}
        variants.append(bad)
        api, session = self.api([])
        for call in variants:
            with self.subTest(call=call), self.assertRaises(ValueError):
                api.request(**call)
        self.assertFalse(session.calls)
        self.assertFalse((api.run_dir / "request_intents").exists())

    def test_raw_cache_is_exact_and_scoped_type_preserved(self):
        api, session = self.api(["Paris", "[[yes]]"])
        first = ev.answer_and_judge(api, "Q", "Paris", FACTS, reference_date=None)
        second = ev.answer_and_judge(api, "Q", "Paris", FACTS, reference_date=None)
        self.assertEqual(first, second)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(list(api.cache_dir.glob("*.json"))), 2)
        self.assertEqual([event["source"] for event in api.events],
                         ["new_request", "new_request", "existing_run_cache", "existing_run_cache"])
        scoped = api.scoped(conversation=2, method="plain")
        self.assertIsInstance(scoped, PlainTextEvaluationAPI)
        self.assertEqual(scoped.context, {"conversation": 2, "method": "plain"})
        self.assertEqual(scoped.cache_roots, api.cache_roots)
        self.assertEqual(scoped.gate_path, api.gate_path)
        self.assertEqual(scoped.retries, 2)
        with self.assertRaises(ValueError):
            scoped.request("gemini", ev.READER_MODEL, {}, stage="qa_reader")
        self.assertFalse(hasattr(scoped.local, "session"))

    def test_unknown_logical_request_is_not_redispatched(self):
        api, _ = self.api([])
        class BrokenSession:
            calls = 0

            def post(self, *args, **kwargs):
                self.calls += 1
                raise requests.ConnectionError("offline simulated interruption")
        session = api.local.session = BrokenSession()
        with patch("jev_mem.providers.time.sleep"), self.assertRaises(RuntimeError):
            self.read(api)
        self.assertEqual(session.calls, 3)  # Existing transport retries=2.
        with self.assertRaisesRegex(RuntimeError, "Prior unresolved logical request"):
            self.read(api)
        self.assertEqual(session.calls, 3)
        self.assertEqual(len(list((api.run_dir / "request_intents").glob("*.json"))), 1)
        self.assertEqual(list(api.cache_dir.glob("*.json")), [])

    def test_judge_guard_rejects_added_instructions_or_schema(self):
        capture = CaptureAPI([response("[[no]]")])
        ev.judge_answer(capture, "Q", "G", "R")
        valid = capture.calls[0]
        variants = []
        bad = copy.deepcopy(valid)
        bad["body"]["systemInstruction"] = {"parts": [{"text": "Be stricter"}]}
        variants.append(bad)
        bad = copy.deepcopy(valid)
        bad["body"]["generationConfig"]["responseSchema"] = {"type": "STRING"}
        variants.append(bad)
        bad = copy.deepcopy(valid)
        bad["body"]["contents"][0]["parts"][0]["text"] = "Changed instructions"
        variants.append(bad)
        api, session = self.api([])
        for call in variants:
            with self.assertRaises(ValueError):
                api.request(**call)
        self.assertFalse(session.calls)

    def test_new_request_hash_differs_from_legacy_not_just_stage(self):
        api = CaptureAPI([response("Paris")])
        self.read(api)
        new = {key: val for key, val in api.calls[0].items() if key != "stage"}
        previous = copy.deepcopy(new)
        previous["body"]["generationConfig"]["responseMimeType"] = "application/json"
        previous["body"]["systemInstruction"] = {"parts": [{"text": old.READER_SYSTEM}]}
        self.assertNotEqual(digest(new), digest(previous))

    def test_nonqa_writer_request_unchanged(self):
        plain = CaptureAPI([response('{"memory": []}')])
        legacy = CaptureAPI([response('{"memory": []}')], guarded=False)
        for api in (plain, legacy):
            self.assertEqual(api.generate("input", system="original writer", json_output=True,
                                          stage="mem0_llm"), {"memory": []})
        self.assertEqual(plain.calls, legacy.calls)


if __name__ == "__main__":
    unittest.main()
