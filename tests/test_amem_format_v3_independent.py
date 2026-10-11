"""Independent offline representation/action tests; never dispatches a request."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import baselines_amem_v3 as v3
from tools import baselines_amem_v1 as v1
from tools import baselines_amem_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "research/baseline_execution_2026-10-08/heartbeat_20261008T1731/amem_failure_chain_review_v1.json"


def fixture(text):
    return {"request": {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent",
                        "body": {"contents": [{"role": "user", "parts": [{"text": "fixture"}]}],
                                 "generationConfig": {"thinkingConfig": {"thinkingBudget": 0},
                                                      "maxOutputTokens": 16384,
                                                      "responseMimeType": "application/json"}}},
            "response": {"modelVersion": "gemini-2.5-flash", "candidates": [
                {"finishReason": "STOP", "index": 0,
                 "content": {"role": "model", "parts": [{"text": text}]}}]}}


def decode(record):
    return v3.decode_generation_json(record, raw_artifact_sha256="f" * 64)


class IndependentBridgeTests(unittest.TestCase):
    def test_local_publication_retries_only_same_serialized_bytes(self):
        original_replace = Path.replace
        identities = (v1._write_json, v2._v1._write_json, v2._runtime._write_json)
        attempts = []
        error = PermissionError("offline transient sharing fixture")
        error.winerror = 32
        def replace(path, target):
            attempts.append(path.read_bytes())
            if len(attempts) < 5:
                raise error
            return original_replace(path, target)
        value = {"unicode": "汉字", "embedded_newline": "first\nsecond"}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            with patch.object(Path, "replace", replace), patch.object(v3.time, "sleep") as sleep, patch.object(v3.json, "dumps", wraps=json.dumps) as serial:
                v3._write_json(target, value)
                self.assertEqual(serial.call_count, 1)
                self.assertEqual(sleep.call_count, 4)
            self.assertEqual(len(attempts), 5)
            self.assertTrue(all(data == attempts[0] for data in attempts))
            self.assertEqual(target.read_bytes(), attempts[0])
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), value)
        self.assertEqual(identities, (v1._write_json, v2._v1._write_json, v2._runtime._write_json))

    def test_local_publication_exhaustion_keeps_old_target_and_final_temp(self):
        error = PermissionError("offline exhausted access fixture")
        error.winerror = 5
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            target.write_bytes(b"old-original-target")
            with patch.object(Path, "replace", side_effect=error) as publication, patch.object(v3.time, "sleep"):
                with self.assertRaises(PermissionError):
                    v3._write_json(target, {"new": "only persisted once"})
                self.assertEqual(publication.call_count, 5)
            self.assertEqual(target.read_bytes(), b"old-original-target")
            self.assertEqual(json.loads(target.with_name(target.name + ".tmp").read_bytes()), {"new": "only persisted once"})

    def test_only_literal_string_tabs_change_and_input_is_immutable(self):
        text = '{"word":"汉字\tvalue","escaped":"a\\tb","slash":"\\\\","quote":"\\\"\tend"}'
        record = fixture(text)
        before = copy.deepcopy(record)
        value, journal = decode(record)
        self.assertEqual(value, {"word": "汉字\tvalue", "escaped": "a\tb", "slash": "\\", "quote": '"\tend'})
        self.assertEqual(record, before)
        self.assertEqual(len(journal["changes"]), 2)
        for change in journal["changes"]:
            offset = change["character_offset"]
            self.assertEqual(text[offset], "\t")
            self.assertEqual(change["utf8_byte_offset"], len(text[:offset].encode()))

    def test_normal_json_and_whitespace_tabs_are_not_derived(self):
        _, journal = decode(fixture('\t{"word":"a\\tb"}\n'))
        self.assertEqual(journal["changes"], [])
        self.assertEqual(journal["original_text_utf8_sha256"], journal["derived_text_utf8_sha256"])

    def test_malformed_and_other_control_inputs_cannot_be_repaired(self):
        bad = ['{"a":"bad\nvalue"}', '{"a":"bad\rvalue"}', '{"a":"bad\x00value"}',
               '{"a":"bad\tvalue",}', '{"a":"bad\tvalue"',
               '{"a":"backslash\\\tvalue"}', '{"a":"bad\tvalue"} EXTRA',
               '{"a":"first","a":"duplicate"}', '{"a": NaN}', '{"a": Infinity}',
               '{"a":"first","a":"dup\tlicate"}', '```json\n{"a":"a\tb"}\n```',
               '["a\tb"]']
        for text in bad:
            with self.subTest(text=repr(text)), self.assertRaises(Exception):
                decode(fixture(text))

    def test_incomplete_or_non_plain_or_changed_contract_rejected(self):
        base = fixture('{"a":"a\tb"}')
        variants = []
        for reason in ("MAX_TOKENS", "SAFETY", None):
            item = copy.deepcopy(base)
            item["response"]["candidates"][0]["finishReason"] = reason
            variants.append(item)
        item = copy.deepcopy(base)
        item["response"]["candidates"] *= 2
        variants.append(item)
        item = copy.deepcopy(base)
        item["response"]["candidates"][0]["content"]["parts"][0]["thought"] = True
        variants.append(item)
        item = copy.deepcopy(base)
        item["request"]["body"]["generationConfig"]["temperature"] = 0
        variants.append(item)
        item = copy.deepcopy(base)
        item["request"]["body"]["generationConfig"]["maxOutputTokens"] = 8192
        variants.append(item)
        item = copy.deepcopy(base)
        item["response"]["candidates"][0]["safetyRatings"] = [{"blocked": True}]
        variants.append(item)
        for index, item in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(v3.AMemV3Error):
                decode(item)

    def test_actual_c1_c9_tab_results_match_independent_payload_hashes(self):
        report = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        actual = {x["conversation"]: x for x in report["failed_native_calls"]}
        expected = {x["conversation"]: x for x in report["native_semantics_followup"]["syntax_representations"]}
        for ci in (1, 9):
            data = Path(actual[ci]["raw_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), actual[ci]["raw_sha256"])
            value, journal = v3.decode_generation_json(json.loads(data), raw_artifact_sha256=actual[ci]["raw_sha256"])
            self.assertEqual(v1._sha(value), expected[ci]["payload_sha256"])
            self.assertEqual(journal["derived_text_utf8_sha256"], expected[ci]["derived_text_sha256"])
            self.assertEqual([c["character_offset"] for c in journal["changes"]], expected[ci]["tab_character_offsets"])

    def test_actual_c5_bad_keywords_remains_rejected(self):
        report = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        row = next(x for x in report["failed_native_calls"] if x["conversation"] == 5)
        raw = Path(row["raw_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), row["raw_sha256"])
        with self.assertRaises(v3.AMemV3Error):
            v3.decode_generation_json(json.loads(raw), raw_artifact_sha256=row["raw_sha256"])

    def test_actual_c2_actions_unchanged_and_original_schema_enforced(self):
        report = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        row = next(x for x in report["failed_native_calls"] if x["conversation"] == 2)
        call = json.loads(Path(row["native_call"]).read_text(encoding="utf-8"))
        controller = v3._Controller(None, Path("unused-independent-never-write"))
        controller.memory_count = 245
        controller.neighbor_indices = [110, 134, 64, 214, 61]
        schema = call["request"]["response_format"]["json_schema"]["schema"]
        original = copy.deepcopy(call["raw_response"])
        value, journal = controller._derive(original, schema, "amem_evolution")
        self.assertEqual(value["actions"], ["strestrengthen", "update_neighbor"])
        self.assertEqual(value["suggested_connections"], [134, 214, 61])
        self.assertEqual(original, call["raw_response"])
        self.assertEqual(journal["native_action_dispatch"]["unknown_actions"], [{"position": 0, "value": "strestrengthen"}])
        invalid = copy.deepcopy(original)
        invalid["actions"] = [5]
        with self.assertRaises(v1.AMemError):
            controller._derive(invalid, schema, "amem_evolution")
        invalid = copy.deepcopy(original)
        invalid["suggested_connections"] = ["1"]
        with self.assertRaises(v1.AMemError):
            controller._derive(invalid, schema, "amem_evolution")


if __name__ == "__main__":
    unittest.main()
