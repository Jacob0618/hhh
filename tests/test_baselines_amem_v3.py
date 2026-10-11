"""Offline native-action and strict TAB bridge regression; no provider calls."""
import ast
import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from tools import baselines_amem_v1 as v1
from tools import baselines_amem_v2 as v2
from tools import baselines_amem_v3 as v3
from tools.check_experiment_health import shared_read
from jev_mem.gemini25_ablation_provider_v1 import generation_body
from jev_mem.sdk20_cached_api import digest
from tests.test_baselines_amem_v1 import FakeAPI, FakeEncoder, conversation, evolution

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/five_baselines_gemini25_v1_20261008/amem_format_v2"
ACTUAL = {
    1: (352, "a909271e65f9800ea682d3218200680a2b862528e271f43b086c0e1067a3d482", "3739218089b42d2143127e1988e4de909477453baf8118efbcc5eebef0eb0c0f"),
    2: (246, "e58d58b235af06a4f05da23a0e3e8e21d2e82c844d40af2fd664fdb21be905a5", "3a2d493251a82d1a6fe52efda7e13442489d2d03ab0ba8b7be7f8965e34939fc"),
    9: (106, "fc1b05bc318d41bfa2ad0b4b7b72be25615f4bfd16dbd23db7c3537e8c74dfeb", "cec3d0af63eb1ddd206741687259d8c757115a1fa9c82d402c19e76c487758b7"),
}


def read(path):
    return json.loads(shared_read(path))


def raw_fixture(text):
    return {"request": {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent",
                        "body": generation_body("fixture", json_output=True, max_output=16384)},
            "response": {"modelVersion": "gemini-2.5-flash", "candidates": [
                {"finishReason": "STOP", "index": 0, "content": {"role": "model", "parts": [{"text": text}]}}]}}


def decode(raw):
    return v3.decode_generation_json(raw, raw_artifact_sha256=hashlib.sha256(json.dumps(raw).encode()).hexdigest())


class AMemV3Tests(unittest.TestCase):
    def test_local_replace_retries_same_bytes_without_provider(self):
        real_replace = Path.replace
        calls, serializations, payloads = [], [], []
        real_dumps = json.dumps
        def serialize(*args, **kwargs):
            serializations.append(1)
            return real_dumps(*args, **kwargs)
        def replace(path, destination):
            calls.append((path, destination));payloads.append(path.read_bytes())
            if len(calls) < 3:
                error = PermissionError("transient publication fixture");error.winerror = 5 if len(calls) == 1 else 32
                raise error
            return real_replace(path, destination)
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "replace", replace), patch.object(v3.json, "dumps", side_effect=serialize), patch.object(v3.time, "sleep") as sleep:
            output = Path(directory) / "snapshot.json"
            v3._write_json(output, {"x": "\u4e2d\ntext"})
            self.assertEqual(len(serializations), 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual(payloads, [output.read_bytes()] * 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertFalse(output.with_name(output.name + ".tmp").exists())
        self.assertEqual(api.calls, [])

    def test_local_replace_exhaustion_keeps_temp_and_old_target(self):
        for code, expected_calls in ((33, 5), (5, 5), (32, 5), (2, 1), (None, 1)):
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as directory, patch.object(v3.time, "sleep") as sleep:
                output = Path(directory) / "snapshot.json";output.write_bytes(b"old")
                error = PermissionError("fixture")
                if code is not None:error.winerror = code
                with patch.object(Path, "replace", side_effect=error) as replace:
                    with self.assertRaises(PermissionError):v3._write_json(output, {"new": True})
                self.assertEqual(replace.call_count, expected_calls)
                self.assertEqual(sleep.call_count, expected_calls - 1)
                self.assertEqual(output.read_bytes(), b"old")
                self.assertEqual(json.loads(output.with_name(output.name + ".tmp").read_bytes()), {"new": True})

    def test_local_replace_rejects_modified_temp_before_retry(self):
        def replace(path, destination):
            path.write_bytes(b"different serialized bytes")
            error = PermissionError("fixture");error.winerror = 5;raise error
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "replace", replace), patch.object(v3.time, "sleep"):
            output = Path(directory) / "snapshot.json"
            with self.assertRaisesRegex(v3.AMemV3Error, "temporary JSON changed"):
                v3._write_json(output, {"new": True})
            self.assertFalse(output.exists())
            self.assertEqual(output.with_name(output.name + ".tmp").read_bytes(), b"different serialized bytes")

    def test_private_writer_isolation_and_no_generation_retry_on_publication(self):
        before = (v1._write_json, v2._runtime._write_json, v2._v1._write_json)
        real_replace = Path.replace
        failures = []
        def replace(path, destination):
            if (destination.name == "checkpoint.json" and not failures
                    and json.loads(path.read_bytes())["completed_notes"] == 1):
                failures.append(1);error = PermissionError("fixture");error.winerror = 32;raise error
            return real_replace(path, destination)
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory, patch.object(v3._runtime, "_get_encoder", return_value=FakeEncoder()), patch.object(Path, "replace", replace), patch.object(v3.time, "sleep"):
            state = v3.build(conversation(), Path(directory), api)
            self.assertEqual(state["completed_notes"], 2)
            self.assertEqual(len(api.calls), 4)
        self.assertEqual(before, (v1._write_json, v2._runtime._write_json, v2._v1._write_json))
        self.assertIs(v3._runtime._write_json, v3._write_json)
        self.assertIs(v3._private._v1._write_json, v3._write_json)

    def test_tab_escape_preserves_unicode_values_and_offsets(self):
        text = '{"x":"\u4e2d\tend", "escaped":"a\\tb", "quote":"a\\\"b", "slash":"\\\\"}'
        original = raw_fixture(text)
        before = copy.deepcopy(original)
        value, journal = decode(original)
        self.assertEqual(value, {"x": "\u4e2d\tend", "escaped": "a\tb", "quote": 'a"b', "slash": "\\"})
        self.assertEqual(journal["changes"], [{"character_offset": 7, "utf8_byte_offset": 9,
            "from_codepoint": "U+0009", "from_utf8_hex": "09", "to_utf8_hex": "5c74"}])
        self.assertEqual(before, original)

    def test_strict_success_has_no_transform(self):
        value, journal = decode(raw_fixture('{"x":"already\\tescaped"}'))
        self.assertEqual(value["x"], "already\tescaped")
        self.assertEqual(journal["changes"], [])
        self.assertEqual(journal["original_text_utf8_sha256"], journal["derived_text_utf8_sha256"])

    def test_invalid_json_never_receives_generic_repair(self):
        cases = ['{"x":"a\nb"}', '{"x":"a\rb"}', '{"x":"a\x00b"}', '{"x":"a\t',
                 '{"x":"a\tb",}', '{"x":"a\tb"} extra', '{"x":1,"x":2}',
                 '{"x":"a\tb","x":2}', '{"x":NaN}', '[]', '{"x":"a\\\tb"}',
                 '{"keywords":["reach out"\n  node",\n "support"]}']
        for text in cases:
            with self.subTest(text=repr(text)), self.assertRaises(v3.AMemV3Error):
                decode(raw_fixture(text))

    def test_response_and_request_eligibility_fail_closed(self):
        fixtures = []
        for finish in ("MAX_TOKENS", "SAFETY", None):
            raw = raw_fixture('{"x":"a\tb"}');raw["response"]["candidates"][0]["finishReason"] = finish;fixtures.append(raw)
        for model in (None, "gemini-2.5-flash-lite"):
            raw = raw_fixture('{"x":"a\tb"}');raw["response"]["modelVersion"] = model;fixtures.append(raw)
        raw = raw_fixture('{"x":"a\tb"}');raw["response"]["candidates"] *= 2;fixtures.append(raw)
        raw = raw_fixture('{"x":"a\tb"}');raw["response"]["candidates"][0]["content"]["parts"][0]["thought"] = True;fixtures.append(raw)
        raw = raw_fixture('{"x":"a\tb"}');raw["request"]["body"]["generationConfig"]["temperature"] = 0;fixtures.append(raw)
        raw = raw_fixture('{"x":"a\tb"}');raw["response"]["promptFeedback"] = {"blockReason": "SAFETY"};fixtures.append(raw)
        for raw in fixtures:
            with self.subTest(raw=raw), self.assertRaises(v3.AMemV3Error):decode(raw)

    def test_native_unknown_action_kept_and_later_action_runs(self):
        response = evolution(should_evolve=True, actions=["strestrengthen", "update_neighbor"],
                             suggested_connections=["0"], tags_to_update=["must not apply"],
                             new_context_neighborhood=["Updated neighbor"], new_tags_neighborhood=[["updated"]])
        with tempfile.TemporaryDirectory() as directory, patch.object(v3._runtime, "_get_encoder", return_value=FakeEncoder()):
            state = v3.build(conversation(), Path(directory), FakeAPI([evolution(), response]))
            self.assertEqual(state["notes"][1]["links"], [])
            self.assertNotEqual(state["notes"][1]["tags"], response["tags_to_update"])
            self.assertEqual(state["notes"][0]["context"], "Updated neighbor")
            self.assertEqual(state["evo_cnt"], 1)
            call = read(Path(directory) / "calls/note_000002_amem_evolution.json")
            self.assertEqual(call["parsed_response"]["actions"], response["actions"])
            self.assertEqual(call["normalization"]["native_action_dispatch"]["unknown_actions"], [{"position": 0, "value": "strestrengthen"}])

    def test_nonstring_actions_and_other_schema_errors_still_fail(self):
        for change in ({"actions": [2]}, {"actions": "strengthen"}, {"should_evolve": "true"}, {"new_tags_neighborhood": [2]}):
            with tempfile.TemporaryDirectory() as directory, patch.object(v3._runtime, "_get_encoder", return_value=FakeEncoder()):
                with self.assertRaises(v3.AMemError):
                    v3.build(conversation(), Path(directory), FakeAPI([evolution(), evolution(**change)]))

    def test_private_runtime_and_scientific_success_equal_v2(self):
        before = (v2._Controller, v2._runtime._Controller, v2.SCHEMA, v1._Controller)
        response = evolution(should_evolve=True, actions=["strengthen", "update_neighbor"], suggested_connections=["0"],
                             tags_to_update=["new"], new_context_neighborhood=["Updated"], new_tags_neighborhood=[["tag"]])
        with tempfile.TemporaryDirectory() as directory, patch.object(v3._runtime, "_get_encoder", return_value=FakeEncoder()), patch.object(v2._runtime, "_get_encoder", return_value=FakeEncoder()):
            old = v2.build(conversation(), Path(directory) / "v2", FakeAPI([evolution(), response]))
            new = v3.build(conversation(), Path(directory) / "v3", FakeAPI([evolution(), response]))
            for key in ("notes", "corpus", "memory_sha256", "evo_cnt"):self.assertEqual(old[key], new[key])
        self.assertEqual(before, (v2._Controller, v2._runtime._Controller, v2.SCHEMA, v1._Controller))

    @unittest.skipUnless(RUN.is_dir(), "Original evidence was cleaned; generic regressions remain")
    def test_real_three_failed_raws_and_actual_c5_c7_syntax_rejection(self):
        for ci, (note, key, expected) in ACTUAL.items():
            path = RUN / "work/api_method/api_cache" / (key + ".json")
            data = shared_read(path);self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
            value, journal = v3.decode_generation_json(json.loads(data), raw_artifact_sha256=expected)
            call = read(RUN / f"work/native/c{ci}/calls/note_{note:06d}_amem_evolution.json")
            controller = v3._Controller(None, Path("unused_offline"))
            controller.memory_count = note - 1
            controller.neighbor_indices = list(map(int, re.findall(r"(?:^|\n)\s*memory index:(\d+)\t", call["request"]["prompt"])))
            derived, normalization = controller._derive(value, call["request"]["response_format"]["json_schema"]["schema"], "amem_evolution")
            self.assertEqual(derived["actions"], value["actions"])
            self.assertEqual(bool(journal["changes"]), ci != 2)
            self.assertEqual(bool(normalization["native_action_dispatch"]["unknown_actions"]), ci == 2)
            self.assertEqual(hashlib.sha256(shared_read(path)).hexdigest(), expected)
        for ci, note in ((5, 506), (7, 624)):
            callpath = RUN / f"work/native/c{ci}/calls/note_{note:06d}_amem_analysis.json"
            if callpath.exists():
                req = read(callpath)["request"]
                sig = {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent",
                       "body": generation_body(req["prompt"], system=req["system"], json_output=True, max_output=16384)}
                data = shared_read(RUN / "work/api_method/api_cache" / (digest(sig) + ".json"))
                with self.subTest(conversation=ci), self.assertRaises(v3.AMemV3Error):
                    v3.decode_generation_json(json.loads(data), raw_artifact_sha256=hashlib.sha256(data).hexdigest())

    @unittest.skipUnless(RUN.is_dir(), "Original evidence was cleaned")
    def test_recorded_neighbor_native_completed_prefix_replay(self):
        """Replay native updates/prompts, holding recorded neighbor choices fixed.

        This proves memory/action equivalence, not recomputed MiniLM ranking.
        Source conversation IDs and successful raw values are taken from the
        immutable checkpoint/journals; no QA/gold enters this replay.
        """
        for ci, (failed_note, _, _) in ACTUAL.items():
            native = RUN / f"work/native/c{ci}"
            checkpoint = read(native / "checkpoint.json")
            records = [read(native / "calls" / f"note_{note:06d}_{stage}.json")
                       for note in range(1, failed_note) for stage in ("amem_analysis", "amem_evolution")]
            class RecordedAPI:
                def __init__(self):self.position = 0
                def generate(self, prompt, **kwargs):
                    record = records[self.position];self.position += 1
                    req = record["request"]
                    if prompt != req["prompt"] or kwargs != {key:req[key] for key in ("system", "json_output", "max_output", "stage")}:
                        raise AssertionError("Exact original prefix request changed")
                    return copy.deepcopy(record["raw_response"])
            api = RecordedAPI()
            with tempfile.TemporaryDirectory() as directory:
                controller = v3._Controller(api, Path(directory))
                system = v3._runtime._classes(FakeEncoder(), controller)["AgenticMemorySystem"]()
                for number, source in enumerate(checkpoint["provenance"], 1):
                    journal = read(native / "sessions" / source["session"] / f"note_{number:06d}.json")
                    expected = records[2 * number - 1]
                    indices = expected["normalization"]["actual_neighbor_indices"]
                    # Native neighbor formatting, process_memory, add_note and
                    # consolidation all run; only the recorded search is fixed.
                    system.retriever.search = lambda query, k=5, indices=indices: indices
                    controller.operation = f"note_{number:06d}"
                    controller.memory_count = len(system.memories)
                    system.add_note(journal["content"], time=source["timestamp"], id=source["note_id"], last_accessed=source["timestamp"])
                self.assertEqual(api.position, 2 * checkpoint["completed_notes"])
                self.assertEqual([v1._note_record(note) for note in system.memories.values()], checkpoint["notes"])
                self.assertEqual(system.evo_cnt, checkpoint["evo_cnt"])
                self.assertEqual(system.retriever.corpus, checkpoint["corpus"])

    @unittest.skipUnless(RUN.is_dir(), "Original evidence was cleaned")
    def test_actual_c2_raw_executes_native_noop_and_update_neighbor(self):
        native = RUN / "work/native/c2"
        checkpoint = read(native / "checkpoint.json")
        raw_path = RUN / "work/api_method/api_cache" / (ACTUAL[2][1] + ".json")
        raw = json.loads(shared_read(raw_path))
        value, _ = v3.decode_generation_json(raw, raw_artifact_sha256=ACTUAL[2][2])
        call = read(native / "calls/note_000246_amem_evolution.json")
        neighbors = list(map(int, re.findall(r"(?:^|\n)\s*memory index:(\d+)\t", call["request"]["prompt"])))
        value, _ = v2.normalize_evolution(value, memory_count=245, neighbor_indices=neighbors)
        # Compile the exact original successful native functions, rather than
        # reconstructing action dispatch in the test or using adapter methods.
        source = v1._sources()["memory_layer.py"]
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "AgenticMemorySystem")
        functions = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"process_memory", "add_note"}]
        class Note(SimpleNamespace):
            def __init__(self, content, **kwargs):
                super().__init__(content=content, id="test-new", context="new context", keywords=["kindness"], tags=["original"], links=[])
        namespace = {"json": json, "MemoryNote": Note, "print": lambda *a, **k: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "<exact-original-native-methods>", "exec"), namespace)
        def execute(actions):
            response = copy.deepcopy(value);response["actions"] = actions
            system = SimpleNamespace(memories={row["id"]:SimpleNamespace(**copy.deepcopy(row)) for row in checkpoint["notes"]},
                evo_cnt=checkpoint["evo_cnt"], evo_threshold=100, evolution_system_prompt="{content}",
                find_related_memories=lambda query,k=5:("recorded neighbors", neighbors),
                llm_controller=SimpleNamespace(llm=SimpleNamespace(get_completion=lambda *a,**k:json.dumps(response))),
                retriever=SimpleNamespace(add_documents=lambda *a:None), consolidate_memories=lambda:None)
            system.process_memory = MethodType(namespace["process_memory"], system)
            namespace["add_note"](system, "Actual note 246 controlled content fixture")
            return {key:vars(note) for key,note in system.memories.items()}, system.evo_cnt
        native, counter = execute(value["actions"])
        without_unknown, same_counter = execute(["update_neighbor"])
        typo_changed, _ = execute(["strengthen", "update_neighbor"])
        self.assertEqual(native, without_unknown)
        self.assertEqual(counter, same_counter)
        self.assertEqual(counter, checkpoint["evo_cnt"] + 1)
        self.assertEqual(native["test-new"]["links"], [])
        self.assertNotEqual(native, typo_changed)


if __name__ == "__main__":
    unittest.main()
