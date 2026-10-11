"""Offline integrity tests for the shared pilot source-date reader."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem.evaluation_pilot import (
    JUDGE_MODEL, JUDGE_STAGE, READER_MODEL, READER_PROTOCOL, READER_STAGE, READER_SYSTEM,
    answer_and_judge, answer_question, prepare_reader_input,
)


def fact(fid="actual-one", text="Alex plans to visit Sam, but the date is uncertain.", *, multi=True):
    dates = {"D1:1": "2023-05-08", "D2:2": "2023-05-15"} if multi else {"D1:1": "2023-05-08"}
    return {"id": fid, "text": text, "observation_date": None if multi else "2023-05-08",
            "observation_dates": sorted(set(dates.values())), "source_observation_dates": dates,
            "temporal_status": "uncertain", "temporal_status_verified": False,
            "support_status": "proposed_unverified", "source_records": [{"text": "RAW_SENTINEL"}],
            "source_context": "CONTEXT_SENTINEL", "embedding": [1.0, 0.0],
            "answer": "FACT_GOLD_SENTINEL", "gold_evidence": "GOLD_EVIDENCE_SENTINEL"}


def prediction(answer="Sam", ids=None):
    return {"answer": answer, "used_fact_ids": ["f1"] if ids is None else ids}


def judgment():
    return {"correct": True, "completeness": 1.0, "reason": "Equivalent person."}


class FakeAPI:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **settings):
        self.calls.append({"prompt": prompt, "payload": json.loads(prompt), "settings": copy.deepcopy(settings)})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class PilotReaderTests(unittest.TestCase):
    def test_multisource_dates_are_preserved_without_raw_or_gold_fields(self):
        original = fact()
        untouched = copy.deepcopy(original)
        api = FakeAPI(prediction(), judgment())
        result = answer_and_judge(api, "Who may Alex visit?", "REFERENCE_ONLY_SENTINEL", [original],
                                  reference_date="2023-12-31")
        evidence = api.calls[0]["payload"]["facts"][0]
        self.assertIsNone(evidence["observation_date"])
        self.assertEqual(evidence["source_observation_dates"], original["source_observation_dates"])
        self.assertEqual(evidence["observation_dates"], ["2023-05-08", "2023-05-15"])
        self.assertFalse(evidence["temporal_status_verified"])
        self.assertEqual(evidence["support_status"], "proposed_unverified")
        serialized = api.calls[0]["prompt"]
        for sentinel in ("RAW_SENTINEL", "CONTEXT_SENTINEL", "FACT_GOLD_SENTINEL",
                         "GOLD_EVIDENCE_SENTINEL", "REFERENCE_ONLY_SENTINEL", "actual-one"):
            self.assertNotIn(sentinel, serialized)
        self.assertEqual(api.calls[1]["payload"]["reference_answer"], "REFERENCE_ONLY_SENTINEL")
        self.assertEqual(api.calls[0]["settings"]["stage"], READER_STAGE)
        self.assertEqual(api.calls[1]["settings"]["stage"], JUDGE_STAGE)
        self.assertEqual(original, untouched)
        self.assertEqual(result["reader_input"], api.calls[0]["payload"])

    def test_null_single_date_and_global_reference_are_not_filled(self):
        row = fact(multi=False)
        row["observation_date"] = None
        result = prepare_reader_input("When?", [row], reference_date=None)
        self.assertIsNone(result["payload"]["facts"][0]["observation_date"])
        self.assertIsNone(result["payload"]["conversation_reference_date"])
        self.assertIn("NOT necessarily", READER_SYSTEM)
        self.assertIn("do not fill it", READER_SYSTEM)

    def test_grounded_annotation_does_not_gain_verification(self):
        row = fact()
        row["temporal_status"] = "grounded"
        evidence = prepare_reader_input("When?", [row], reference_date=None)["payload"]["facts"][0]
        self.assertEqual(evidence["temporal_status"], "grounded")
        self.assertIs(evidence["temporal_status_verified"], False)
        self.assertIn("unverified extraction annotation, even when its value is grounded", READER_SYSTEM)

    def test_reader_signature_has_no_reference_answer_and_only_one_call(self):
        self.assertNotIn("reference", inspect.signature(answer_question).parameters)
        self.assertNotIn("reference", inspect.signature(prepare_reader_input).parameters)
        api = FakeAPI(prediction())
        result = answer_question(api, "Who?", [fact()], reference_date=None)
        self.assertEqual(len(api.calls), 1)
        self.assertNotIn("judge", result)
        with self.assertRaises(TypeError):
            answer_question(api, "Who?", [fact()], reference_date=None, reference="forbidden")

    def test_short_ids_mapping_result_aliases_and_metrics(self):
        api = FakeAPI(prediction("Sam", ["f2", "f1"]), judgment())
        rows = [fact("actual-1", "One distinct fact."), fact("actual-2", "Another distinct fact.")]
        result = answer_and_judge(api, "Who?", "Sam", rows, reference_date="2023-12-31")
        self.assertEqual(result["used_fact_ids"], ["actual-2", "actual-1"])
        self.assertEqual(result["prediction"]["used_local_fact_ids"], ["f2", "f1"])
        self.assertEqual(result["prediction"]["answer"], result["answer"])
        self.assertEqual(result["judgment"], result["judge"])
        self.assertEqual(result["reader_protocol"], READER_PROTOCOL)
        self.assertEqual(result["reader_model"], READER_MODEL)
        self.assertEqual(result["judge_model"], JUDGE_MODEL)
        self.assertEqual(api.calls[0]["settings"]["model"], READER_MODEL)
        self.assertEqual(api.calls[1]["settings"]["model"], JUDGE_MODEL)
        self.assertEqual(result["token_f1"], 1.0)
        self.assertEqual(result["exact_match"], 1.0)

    def test_hash_is_canonical_and_binds_reader_metadata(self):
        row = fact()
        first = prepare_reader_input("Who?", [row], reference_date=None)
        expected = json.dumps(first["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(first["input_hash"], hashlib.sha256(expected.encode("utf-8")).hexdigest())
        reverse = copy.deepcopy(row)
        reverse["source_observation_dates"] = dict(reversed(list(reverse["source_observation_dates"].items())))
        reverse["observation_dates"].reverse()
        reverse["source_context"] = "IGNORED_DIFFERENT_RAW_CONTEXT"
        second = prepare_reader_input("Who?", [reverse], reference_date=None)
        self.assertEqual(first["input_hash"], second["input_hash"])
        self.assertEqual(first["reader_request_hash"], second["reader_request_hash"])
        reverse["temporal_status"] = "grounded"
        changed = prepare_reader_input("Who?", [reverse], reference_date=None)
        self.assertNotEqual(first["input_hash"], changed["input_hash"])
        self.assertNotEqual(first["reader_request_hash"], changed["reader_request_hash"])

    def test_invalid_reader_schema_or_citations_never_reach_judge(self):
        malformed = [[], {}, {"answer": "Sam"}, prediction(12), prediction(ids="f1"),
                     prediction(ids=["f1", "f1"]), prediction(ids=["f2"]),
                     prediction(ids=["actual-one"]), prediction(ids=["D1:1"]),
                     prediction(ids=[1]), {**prediction(), "reason": "extra"}]
        for response in malformed:
            with self.subTest(response=response):
                api = FakeAPI(response, judgment())
                with self.assertRaises(ValueError):
                    answer_and_judge(api, "Who?", "Sam", [fact()], reference_date=None)
                self.assertEqual(len(api.calls), 1)

    def test_invalid_judge_schema_rejected(self):
        malformed = [[], {}, {**judgment(), "correct": 1}, {**judgment(), "completeness": True},
                     {**judgment(), "completeness": -0.1}, {**judgment(), "completeness": 1.1},
                     {**judgment(), "completeness": float("nan")},
                     {**judgment(), "completeness": float("inf")},
                     {**judgment(), "reason": 8}, {**judgment(), "confidence": 1.0}]
        for response in malformed:
            with self.subTest(response=response):
                api = FakeAPI(prediction(), response)
                with self.assertRaises(ValueError):
                    answer_and_judge(api, "Who?", "Sam", [fact()], reference_date=None)
                self.assertEqual(len(api.calls), 2)

    def test_inconsistent_or_invalid_fact_dates_fail_before_api(self):
        changes = [
            {"observation_dates": ["2023-05-08"]},
            {"observation_dates": ["2023-05-08", "2023-05-15", "2023-05-15"]},
            {"observation_date": "2023-05-08"},
            {"observation_date": "2023-02-30"},
            {"observation_dates": []}, {"source_observation_dates": {}},
            {"source_observation_dates": {"": "2023-05-08"}},
            {"source_observation_dates": {"D1:1": None}},
            {"source_observation_dates": {"D1:1": "20230508"}},
            {"temporal_status": "verified"}, {"temporal_status_verified": True},
            {"temporal_status_verified": 0}, {"support_status": "verified"},
        ]
        for change in changes:
            with self.subTest(change=change):
                api = FakeAPI()
                with self.assertRaises(ValueError):
                    answer_question(api, "Who?", [{**fact(), **change}], reference_date=None)
                self.assertEqual(api.calls, [])
        missing = fact()
        del missing["source_observation_dates"]
        with self.assertRaises(ValueError):
            answer_question(FakeAPI(), "Who?", [missing], reference_date=None)

    def test_duplicate_fact_ids_or_text_and_over_budget_are_rejected(self):
        examples = [[fact(), fact()], [fact("a"), fact("b")],
                    [fact(str(i), "Different fact " + str(i)) for i in range(21)]]
        for rows in examples:
            api = FakeAPI()
            with self.assertRaises(ValueError):
                answer_question(api, "Who?", rows, reference_date=None)
            self.assertEqual(api.calls, [])
        api = FakeAPI(prediction(ids=["f20"]))
        rows = [fact(str(i), "Different fact " + str(i)) for i in range(20)]
        self.assertEqual(answer_question(api, "Who?", rows, reference_date=None)["used_fact_ids"], ["19"])

    def test_no_evidence_is_valid_and_cannot_cite_nonexistent_facts(self):
        api = FakeAPI(prediction("Insufficient evidence.", []), judgment())
        result = answer_and_judge(api, "Who?", "Sam", [], reference_date=None)
        self.assertEqual(result["selected_count"], 0)
        self.assertEqual(result["used_fact_ids"], [])
        with self.assertRaises(ValueError):
            answer_question(FakeAPI(prediction()), "Who?", [], reference_date=None)

    def test_text_dedup_matches_case_sensitive_nfc_selector_contract(self):
        rows = [fact("a", "US policy"), fact("b", "us policy")]
        result = answer_question(FakeAPI(prediction(ids=["f1", "f2"])), "What?", rows, reference_date=None)
        self.assertEqual(result["selected_count"], 2)
        rows = [fact("a", "caf\u00e9 plan"), fact("b", "cafe\u0301   plan")]
        with self.assertRaises(ValueError):
            answer_question(FakeAPI(), "What?", rows, reference_date=None)

    def test_invalid_question_reference_date_and_provider_error_fail_closed(self):
        for question, reference_date in [(None, None), ("", None), ("Who?", "2023-05-08T12:00:00"), ("Who?", True)]:
            api = FakeAPI()
            with self.assertRaises(ValueError):
                answer_question(api, question, [fact()], reference_date=reference_date)
            self.assertEqual(api.calls, [])
        api = FakeAPI(RuntimeError("mock network failure"), judgment())
        with self.assertRaisesRegex(RuntimeError, "mock network failure"):
            answer_and_judge(api, "Who?", "Sam", [fact()], reference_date=None)
        self.assertEqual(len(api.calls), 1)


if __name__ == "__main__":
    unittest.main()
