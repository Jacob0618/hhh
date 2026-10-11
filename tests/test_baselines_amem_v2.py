"""Offline regression coverage for actual A-Mem string-index failures."""
import copy
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tools import baselines_amem_v1 as v1
from tools import baselines_amem_v2 as v2
from tests.test_baselines_amem_v1 import FakeAPI, FakeEncoder, conversation, evolution


OLD_WORK = Path(__file__).resolve().parents[1] / "runs/five_baselines_gemini25_v1_20261008/amem/work/native"
C0_SOURCE_ARTIFACT_SHA256 = "b3ad5477a57216f23dcd3da669a93cd8a3d75024307cdb2871ad64cdfe5b0cf6"
# Exact parsed raw payload from c0/note_000002_amem_evolution, not a repaired
# response. Its original source artifact hash is bound above for provenance.
ACTUAL_RAW = {
    "should_evolve": True, "actions": ["strengthen", "update_neighbor"], "suggested_connections": ["0"],
    "tags_to_update": ["Social Interaction", "Informal Dialogue", "Personal Update", "Busy Schedule", "Greeting"],
    "new_context_neighborhood": ["Caroline initiates a friendly exchange with Melanie, inquiring about her well-being to start a conversation, which Melanie then reciprocates with an update on her busy life."],
    "new_tags_neighborhood": [["Social Interaction", "Informal Dialogue", "Personal Check-in", "Reciprocal Greeting"]],
}


class ExactArtifactAPI:
    def __init__(self, records):
        self.records = records
        self.calls = []
        self.events = []

    def generate(self, prompt, **kwargs):
        path, record = self.records[len(self.calls)]
        expected = record["request"]
        assert prompt == expected["prompt"], "Request prompt differs from failed first pass"
        assert kwargs == {key: expected[key] for key in ("system", "json_output", "max_output", "stage")}
        self.calls.append({"prompt": prompt, **kwargs})
        self.events.append({"request_hash": record["request_sha256"], "cache_path": str(path),
                            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "context": {"test_only": True, "conv": path.parents[1].name},
                            "stage": kwargs["stage"], "cache_hit": True})
        return copy.deepcopy(record["raw_response"])


class AMemV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary.name)
        self.encoder = FakeEncoder()
        self.encoder_patch = patch.object(v2._runtime, "_get_encoder", return_value=self.encoder)
        self.encoder_patch.start()

    def tearDown(self):
        self.encoder_patch.stop()
        self.temporary.cleanup()

    def test_actual_raw_is_rejected_by_v1_and_losslessly_derived_by_v2(self):
        original = copy.deepcopy(ACTUAL_RAW)
        schema = {"type": "object", "properties": {"suggested_connections": {"type": "array", "items": {"type": "integer"}}}}
        with self.assertRaises(v1.AMemError):
            v1._validate_json(ACTUAL_RAW, schema)
        derived, lineage = v2.normalize_evolution(ACTUAL_RAW, memory_count=1, neighbor_indices=(0,))
        v1._validate_json(derived, schema)
        self.assertEqual(derived["suggested_connections"], [0])
        self.assertEqual(ACTUAL_RAW, original)
        self.assertTrue(all(derived[key] == ACTUAL_RAW[key] for key in ACTUAL_RAW if key != "suggested_connections"))
        self.assertEqual(lineage["changes"], [{"json_pointer": "/suggested_connections/0", "from": "0", "from_type": "string", "to": 0, "to_type": "integer"}])
        self.assertNotEqual(lineage["original_parsed_sha256"], lineage["derived_parsed_sha256"])

    def test_reject_noncanonical_values_including_unicode_and_other_json_types(self):
        for value in ("00", "01", "-0", "-1", "+0", "0.0", "1e0", " 0", "0 ", "0\n", "０", "٠", "neighbor_0", "", True, False, None, 0.0, [], {}):
            with self.subTest(value=value), self.assertRaises(v2.AMemV2Error):
                v2.normalize_evolution(evolution(suggested_connections=[value]), memory_count=2, neighbor_indices=(0, 1))

    def test_only_actual_retrieved_neighbors_and_existing_global_indexes_allowed(self):
        for value in (0, "0", 2, "2", 10, "10", -1):
            with self.subTest(value=value), self.assertRaises(v2.AMemV2Error):
                v2.normalize_evolution(evolution(suggested_connections=[value]), memory_count=10, neighbor_indices=(1, 3, 5, 7, 9))
        derived, lineage = v2.normalize_evolution(evolution(suggested_connections=["1", 9, "1"]), memory_count=10, neighbor_indices=(1, 3, 5, 7, 9))
        self.assertEqual(derived["suggested_connections"], [1, 9, 1])
        self.assertEqual(len(lineage["changes"]), 2)
        empty, _ = v2.normalize_evolution(evolution(), memory_count=0, neighbor_indices=())
        self.assertEqual(empty["suggested_connections"], [])
        with self.assertRaises(v2.AMemV2Error):
            v2.normalize_evolution(evolution(suggested_connections=["0"]), memory_count=0, neighbor_indices=())

    def test_build_raw_and_derived_separate_with_original_parameters(self):
        raw = copy.deepcopy(ACTUAL_RAW)
        api = FakeAPI([evolution(), raw])
        state = v2.build(conversation(), self.work, api)
        record = json.loads((self.work / "calls/note_000002_amem_evolution.json").read_text())
        self.assertEqual(record["raw_response"], ACTUAL_RAW)
        self.assertEqual(raw, ACTUAL_RAW)
        self.assertEqual(record["normalized_parsed_response"]["suggested_connections"], [0])
        self.assertEqual(record["normalization"]["actual_neighbor_indices"], [0])
        self.assertEqual(record["raw_response_payload_sha256"], v1._sha(ACTUAL_RAW))
        self.assertEqual(state["notes"][1]["links"], [0])
        self.assertEqual(state["seed_k"], 10)
        self.assertTrue(all(call["max_output"] == 16384 and "temperature" not in call for call in api.calls))
        result = v2.retrieve("Who walked in the park?", state, api)
        self.assertEqual(len(result["evidence"]), 3)

    def test_unrelated_bad_fields_are_not_repaired(self):
        for bad in ({"should_evolve": "true"}, {"actions": ["unknown"]}, {"tags_to_update": "a tag"}, {"new_context_neighborhood": [3]}):
            api = FakeAPI([evolution(), {**ACTUAL_RAW, **bad}])
            with self.subTest(bad=bad), self.assertRaises(v2.AMemV2Error):
                v2.build(conversation(), self.work / str(len(list(self.work.iterdir()))), api)

    def test_scientific_outputs_match_v1_given_equivalent_correct_integer_response(self):
        integer_response = copy.deepcopy(ACTUAL_RAW)
        integer_response["suggested_connections"] = [0]
        original_api = FakeAPI([evolution(), integer_response])
        new_api = FakeAPI([evolution(), ACTUAL_RAW])
        with patch.object(v1, "_get_encoder", return_value=self.encoder):
            original_state = v1.build(conversation(), self.work / "original", original_api)
            new_state = v2.build(conversation(), self.work / "new", new_api)
            original_retrieval = v1.retrieve("Who walked?", original_state, original_api)
            new_retrieval = v2.retrieve("Who walked?", new_state, new_api)
        self.assertEqual(original_state["notes"], new_state["notes"])
        self.assertEqual(original_state["corpus"], new_state["corpus"])
        self.assertEqual(original_state["memory_sha256"], new_state["memory_sha256"])
        self.assertEqual(original_retrieval["context"], new_retrieval["context"])
        self.assertEqual(original_retrieval["evidence"], new_retrieval["evidence"])
        self.assertEqual(original_api.calls, new_api.calls)

    def test_private_runtime_does_not_modify_frozen_v1_module(self):
        original_controller, original_classes, original_schema = v1._Controller, v1._classes, v1.SCHEMA
        with ThreadPoolExecutor(max_workers=3) as executor:
            states = list(executor.map(lambda i: v2.build(conversation(), self.work / str(i), FakeAPI([evolution(), ACTUAL_RAW])), range(3)))
        self.assertTrue(all(state["notes"][1]["links"] == [0] for state in states))
        self.assertIs(v1._Controller, original_controller)
        self.assertIs(v1._classes, original_classes)
        self.assertEqual(v1.SCHEMA, original_schema)
        self.assertEqual(hashlib.sha256(v2.BASE_ADAPTER.read_bytes()).hexdigest(), v2.BASE_ADAPTER_SHA256)

    def test_compact_exports_keep_all_derivation_and_update_evidence(self):
        api = FakeAPI([evolution(), ACTUAL_RAW])
        state = v2.build(conversation(), self.work, api)
        self.assertEqual(state["export_paths"], ["state.json", "generation_evidence_summary.json", "memory_updates.jsonl"])
        summary = json.loads((self.work / "generation_evidence_summary.json").read_text())
        self.assertEqual(summary["call_count"], 4)
        self.assertEqual(summary["scope"], "build_writer")
        self.assertTrue(all("prompt" not in row and "raw_response" not in row for row in summary["calls"]))
        evolution_row = next(row for row in summary["calls"] if row["call_artifact"].endswith("note_000002_amem_evolution.json"))
        self.assertEqual(evolution_row["normalization"]["actual_neighbor_indices"], [0])
        self.assertEqual(evolution_row["normalization"]["changes"][0]["from"], "0")
        updates = [json.loads(line) for line in (self.work / "memory_updates.jsonl").read_text().splitlines()]
        self.assertEqual([row["ordinal"] for row in updates], [1, 2])
        self.assertEqual(updates[1]["source"]["dia_id"], "D1:2")
        self.assertTrue((self.work / "sessions/session_1/state.json").is_file())
        self.assertTrue((self.work / "calls/note_000002_amem_evolution.json").is_file())
        summary_bytes_before = (self.work / "generation_evidence_summary.json").read_bytes()
        retrieval = v2.retrieve("Who walked?", state, api)
        summary = json.loads((self.work / "generation_evidence_summary.json").read_text())
        self.assertEqual(summary["call_count"], 4)
        self.assertEqual(sum(row["stage"] == "amem_query_keywords" for row in summary["calls"]), 0)
        self.assertEqual((self.work / "generation_evidence_summary.json").read_bytes(), summary_bytes_before)
        self.assertIn("query_call", retrieval)
        self.assertIn("keywords", retrieval)

    def test_success_replay_rederives_and_verifies_lineage(self):
        api = FakeAPI([evolution(), ACTUAL_RAW])
        v2.build(conversation(), self.work, api)
        count = len(api.calls)
        v2.build(conversation(), self.work, api)
        self.assertEqual(len(api.calls), count)
        path = self.work / "calls/note_000002_amem_evolution.json"
        record = json.loads(path.read_text())
        controller = v2._Controller(api, self.work)
        controller.operation = "note_000002"
        controller.memory_count, controller.neighbor_indices = 1, (0,)
        response = controller.get_completion(record["request"]["prompt"], record["request"]["response_format"])
        self.assertEqual(json.loads(response)["suggested_connections"], [0])
        record["raw_response"]["suggested_connections"] = ["1"]
        path.write_text(json.dumps(record))
        with self.assertRaises(v2.AMemError):
            controller.get_completion(record["request"]["prompt"], record["request"]["response_format"])

    @unittest.skipUnless(OLD_WORK.is_dir(), "Original work artifacts were cleaned after audit; literal actual-raw regression remains above")
    def test_all_ten_real_prefixes_replay_exact_requests_offline(self):
        for conv in range(10):
            directory = OLD_WORK / f"c{conv}"
            paths = [directory / "calls" / filename for filename in ("note_000001_amem_analysis.json", "note_000001_amem_evolution.json", "note_000002_amem_analysis.json", "note_000002_amem_evolution.json")]
            records = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
            if conv == 0:
                self.assertEqual(hashlib.sha256(paths[-1].read_bytes()).hexdigest(), C0_SOURCE_ARTIFACT_SHA256)
                self.assertEqual(records[-1][1]["raw_response"], ACTUAL_RAW)
            original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            source = json.loads((directory / "sessions/session_1/input.json").read_text(encoding="utf-8"))
            turns = source["turns"][:2]
            data = {"speaker_a": turns[0]["speaker"], "speaker_b": turns[1]["speaker"],
                    "session_1_date_time": source["timestamp"], "session_1": turns}
            api = ExactArtifactAPI(records)
            state = v2.build(data, self.work / f"c{conv}", api)
            self.assertEqual(len(api.calls), 4)
            self.assertEqual(state["notes"][1]["links"], [0])
            derived = json.loads((self.work / f"c{conv}" / "calls/note_000002_amem_evolution.json").read_text())
            self.assertEqual(derived["transport_lineage"]["cache_sha256"], original_hashes[paths[-1]])
            self.assertEqual(derived["normalization"]["actual_neighbor_indices"], [0])
            self.assertEqual(original_hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


if __name__ == "__main__":
    unittest.main()
