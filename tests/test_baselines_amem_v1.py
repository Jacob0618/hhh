"""Offline adapter checks: fake generation + fake 384-dimensional encoder only."""
import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import baselines_amem_v1 as amem


def evolution(**changes):
    result = {"should_evolve": False, "actions": [], "suggested_connections": [],
              "tags_to_update": [], "new_context_neighborhood": [], "new_tags_neighborhood": []}
    result.update(changes)
    return result


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        result = np.zeros((len(texts), 384), dtype=np.float32)
        # Unique enough for deterministic tests; deliberately not semantic.
        for row, text in enumerate(texts):
            result[row, 0] = 1
            result[row, 1] = len(text) / 100
            result[row, 2] = sum(map(ord, text)) % 59 / 100
        return result

    def get_config_dict(self):
        return {"model_name": "all-MiniLM-L6-v2"}


class FakeAPI:
    def __init__(self, evolutions=None, failure=None):
        self.calls = []
        self.evolutions = iter(evolutions or [])
        self.failure = failure

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.failure:
            raise self.failure
        stage = kwargs["stage"]
        if stage == "amem_analysis":
            return {"keywords": ["walk", "park", "friend"], "context": "A meeting in the park.",
                    "tags": ["social", "outdoors", "conversation"]}
        if stage == "amem_evolution":
            return next(self.evolutions, evolution())
        if stage == "amem_query_keywords":
            return '{"keywords":"park cosmos friend"}'
        raise AssertionError(stage)

    def embed(self, *args, **kwargs):
        raise AssertionError("A-Mem must retain native MiniLM, not use API embeddings")


def conversation(count=2):
    return {"speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A" if i % 2 == 0 else "B", "dia_id": f"D1:{i+1}",
                           "text": f"I walked in the park with friend number {i}."} for i in range(count)]}


class AMemAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.work = Path(self.directory.name)
        self.encoder = FakeEncoder()
        self.encoder_patch = patch.object(amem, "_get_encoder", return_value=self.encoder)
        self.encoder_patch.start()
        def fake_cosine(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return (x / np.linalg.norm(x, axis=1, keepdims=True)) @ (y / np.linalg.norm(y, axis=1, keepdims=True)).T
        self.cosine_patch = patch.object(amem, "_cosine_similarity", side_effect=fake_cosine)
        self.cosine_patch.start()

    def tearDown(self):
        self.encoder_patch.stop()
        self.cosine_patch.stop()
        self.directory.cleanup()

    def test_original_prompts_two_calls_per_note_and_offline_replay(self):
        api = FakeAPI()
        state = amem.build(conversation(), self.work, api)
        self.assertEqual([c["stage"] for c in api.calls], ["amem_analysis", "amem_evolution"] * 2)
        self.assertTrue(api.calls[0]["prompt"].startswith("Generate a structured analysis"))
        self.assertIn("The number of neighbors is 0.", api.calls[1]["prompt"])
        self.assertIn("memory index:0", api.calls[3]["prompt"])
        self.assertEqual(state["note_count"], 2)
        self.assertTrue(all("temperature" not in c and c["max_output"] == 16384 for c in api.calls))
        self.assertTrue((self.work / "sessions/session_1/state.json").is_file())
        self.assertTrue((self.work / state["export_paths"][0]).is_file())
        replay_api = FakeAPI(failure=AssertionError("Replay cannot call the API"))
        replay = amem.build(conversation(), self.work, replay_api)
        self.assertEqual(replay["memory_sha256"], state["memory_sha256"])
        self.assertEqual(replay_api.calls, [])

    def test_order_and_original_caption_loader(self):
        data = conversation(1)
        first = data.pop("session_1")
        time = data.pop("session_1_date_time")
        data.update({"session_10": [{"speaker": "B", "dia_id": "D10:1", "text": "A photo",
                                    "img_url": "irrelevant.jpg", "blip_caption": "a red bike"}],
                     "session_10_date_time": "2:00 pm on 9 May, 2023", "session_1": first,
                     "session_1_date_time": time})
        state = amem.build(data, self.work, FakeAPI())
        self.assertEqual([s["session"] for s in state["provenance"]], ["session_1", "session_10"])
        self.assertEqual(state["notes"][1]["content"], "Speaker Bsays : [Image: a red bike] A photo")

    def test_qa_or_summaries_never_accepted(self):
        for extra in ("qa", "session_summary", "answer", "evidence", "category"):
            data = conversation()
            data[extra] = "forbidden"
            with self.assertRaises(amem.AMemError):
                amem.build(data, self.work, FakeAPI())
        self.assertFalse((self.work / "checkpoint.json").exists())

    def test_neighbour_evolution_retains_stale_index_until_consolidation(self):
        api = FakeAPI([evolution(), evolution(should_evolve=True, actions=["strengthen", "update_neighbor"],
                      suggested_connections=[0], tags_to_update=["connected"],
                      new_context_neighborhood=["UPDATED"], new_tags_neighborhood=[["new tag"]])])
        state = amem.build(conversation(), self.work, api)
        self.assertEqual(state["notes"][0]["context"], "UPDATED")
        self.assertNotIn("UPDATED", state["corpus"][0])
        self.assertEqual(state["notes"][1]["links"], [0])
        self.assertEqual(state["evo_cnt"], 1)
        result = amem.retrieve("Who walked?", state, api)
        self.assertEqual(len(result["seed_indices"]), 2)
        self.assertEqual(len(result["evidence"]), 3)
        self.assertEqual(sum(e["via"] == "link" for e in result["evidence"]), 1)
        self.assertIn("memory context: UPDATED", result["context"])
        self.assertEqual(state["notes"][0]["retrieval_count"], 0)

    def test_exact_context_render_and_link_expansion_limit_no_dedup(self):
        api = FakeAPI([evolution(), evolution(should_evolve=True, actions=["strengthen"],
                      suggested_connections=[0] * 15, tags_to_update=["linked"])])
        state = amem.build(conversation(), self.work, api)
        result = amem.retrieve("park?", state, api)
        self.assertEqual(len(result["evidence"]), 13)  # 2 seeds + 11 repeated links
        rendered = ""
        for evidence in result["evidence"]:
            note = evidence["note"]
            rendered += ("talk start time:" + note["timestamp"] + "memory content: " + note["content"]
                         + "memory context: " + note["context"] + "memory keywords: " + str(note["keywords"])
                         + "memory tags: " + str(note["tags"]) + "\n")
        self.assertEqual(result["context"], rendered)

    def test_generation_errors_are_hard_failures_with_no_fake_note(self):
        api = FakeAPI(failure=RuntimeError("RESOURCE_EXHAUSTED"))
        with self.assertRaises(amem.AMemError):
            amem.build(conversation(1), self.work, api)
        checkpoint = json.loads((self.work / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["completed_notes"], 0)
        call = json.loads((self.work / "calls/note_000001_amem_analysis.json").read_text())
        self.assertEqual(call["status"], "failed")
        self.assertIn("RESOURCE_EXHAUSTED", call["error"])

    def test_baseexception_propagates_and_is_recorded(self):
        class Fatal(BaseException):
            pass
        with self.assertRaises(Fatal):
            amem.build(conversation(1), self.work, FakeAPI(failure=Fatal("safety")))
        call = json.loads((self.work / "calls/note_000001_amem_analysis.json").read_text())
        self.assertEqual(call["status"], "failed")

    def test_malformed_json_and_schema_do_not_use_original_fallbacks(self):
        for index, value in enumerate(("unknown", {}, {"keywords": [], "context": "", "tags": []})):
            api = FakeAPI()
            api.generate = lambda *args, **kwargs: value
            with self.assertRaises(amem.AMemError):
                amem.build(conversation(1), self.work / str(index), api)

    def test_bad_evolution_links_and_actions_fail(self):
        for index, value in enumerate((evolution(should_evolve=True, actions=["unknown"]),
                                       evolution(should_evolve=True, actions=["strengthen"], suggested_connections=[0]))):
            with self.assertRaises(amem.AMemError):
                amem.build(conversation(1), self.work / str(index), FakeAPI([value]))

    def test_query_failure_is_not_raw_unknown_keyword_fallback(self):
        state = amem.build(conversation(), self.work, FakeAPI())
        api = FakeAPI()
        api.generate = lambda *args, **kwargs: "unknown"
        with self.assertRaises(amem.AMemError):
            amem.retrieve("walk?", state, api)
        self.assertFalse((self.work / "retrieval").exists())

    def test_cache_removed_rebuilds_actual_index_and_query_replays_without_call(self):
        api = FakeAPI()
        state = amem.build(conversation(), self.work, api)
        (self.work / "cache/embeddings.npy").unlink()
        result = amem.retrieve("walking?", state, api)
        self.assertIn(state["corpus"], self.encoder.calls)
        previous = len(api.calls)
        replay = amem.retrieve("walking?", state, api)
        self.assertEqual(replay, result)
        self.assertEqual(len(api.calls), previous)

    def test_unknown_call_outcome_requires_recovery(self):
        api = FakeAPI()
        state = amem.build(conversation(1), self.work, api)
        call_path = self.work / "calls/note_000001_amem_analysis.json"
        record = json.loads(call_path.read_text())
        record["status"] = "started"
        call_path.write_text(json.dumps(record))
        controller = amem._Controller(api, self.work)
        controller.operation = "note_000001"
        with self.assertRaises(amem.AMemError):
            controller.get_completion(record["request"]["prompt"], record["request"]["response_format"])

    def test_parallel_duplicate_query_has_single_generation(self):
        api = FakeAPI()
        state = amem.build(conversation(), self.work, api)
        before = copy.deepcopy(state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: amem.retrieve("Who walked in park?", state, api), range(4)))
        self.assertEqual(sum(c["stage"] == "amem_query_keywords" for c in api.calls), 1)
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(before, state)

    def test_hundred_true_evolutions_use_original_consolidation(self):
        api = FakeAPI([evolution(should_evolve=True) for _ in range(100)])
        state = amem.build(conversation(100), self.work, api)
        self.assertEqual(state["evo_cnt"], 100)
        self.assertTrue(all(" , " in doc and not doc.startswith("content:") for doc in state["corpus"]))
        self.assertEqual(len(api.calls), 200)

    def test_source_and_state_integrity_gates(self):
        amem._sources()
        state = amem.build(conversation(), self.work, FakeAPI())
        state["notes"][0]["content"] = "tampered"
        with self.assertRaises(amem.AMemError):
            amem.retrieve("Who?", state, FakeAPI())


if __name__ == "__main__":
    unittest.main()
