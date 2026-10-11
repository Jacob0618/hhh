"""Offline tests: real native Memory.add with fake LLM/storage and blocked I/O."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_mem0_adapter as fixtures
from tests.test_mem0_regroup_sdk20 import Embedding
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0.memory.utils import parse_messages
from jev_mem.source_local_group_context_v1 import (
    SourceLocalMem0GroupAdapter, VERSION, build_context_plan,
)
from tools import run_mem0_regroup_sdk20 as frozen


def data():
    sessions = []
    for n in range(2):
        sessions.append({"session_id": "session_" + str(n + 1),
                         "observation_date": "2023-05-0" + str(n + 1),
                         "source_turn_ids": [f"D{n+1}:{k+1}" for k in range(8)],
                         "messages": [{"role": "user" if k % 2 == 0 else "assistant",
                                       "content": ("Alice" if k % 2 == 0 else "Bob") + f": raw turn {n*8+k}."}
                                      for k in range(8)]})
    segments = [{"segment_id": f"s{i}", "source_turn_ids": [tid]}
                for i, tid in enumerate(t for s in sessions for t in s["source_turn_ids"])]
    return sessions, segments


def envelope(indices, name="g0", sessions=None):
    sessions = sessions or data()[0]
    source = []
    for session in sessions:
        for tid, message in zip(session["source_turn_ids"], session["messages"]):
            source.append(({"source_turn_id": tid, "session_id": session["session_id"],
                            "observation_date": session["observation_date"],
                            "source_order_index": len(source)}, message))
    rows = [source[i] for i in indices]
    dates = sorted({s["observation_date"] for s, m in rows})
    return {"group_id": name, "segment_ids": [f"s{i}" for i in indices],
            "source_turn_ids": [s["source_turn_id"] for s, m in rows],
            "turn_sources": [copy.deepcopy(s) for s, m in rows],
            "messages": [copy.deepcopy(m) for s, m in rows],
            "source_order_indices": list(indices), "source_observation_dates": dates,
            "observation_date": dates[0] if len(dates) == 1 else None}


class ContextPlanTests(unittest.TestCase):
    def plan(self, indices, sessions=None, segments=None):
        base_sessions, base_segments = data()
        sessions = sessions or base_sessions
        return build_context_plan(envelope(indices, sessions=sessions),
                                  source_sessions=sessions, segments=segments or base_segments)

    def test_immediate_two_only_and_no_cross_session_history(self):
        p = self.plan([3, 8, 11])
        self.assertEqual([r["source_turn_id"] for r in p["history_context"]],
                         ["D1:2", "D1:3", "D2:2", "D2:3"])
        self.assertEqual(p["boundaries"][1]["context_turn_ids"], [])

    def test_shared_context_deduplicates_and_group_members_excluded(self):
        p = self.plan([3, 4])
        self.assertEqual([r["source_turn_id"] for r in p["history_context"]], ["D1:2", "D1:3"])
        self.assertEqual(p["history_context"][1]["for_segment_ids"], ["s3", "s4"])
        self.assertTrue(all(r["context_only"] for r in p["history_context"]))

    def test_unknown_changed_source_and_qa_rejected(self):
        sessions, segments = data()
        env = envelope([3])
        env["messages"][0]["content"] += " fabricated"
        with self.assertRaisesRegex(ValueError, "differs from original"):
            build_context_plan(env, source_sessions=sessions, segments=segments)
        sessions[0]["qa"] = []
        with self.assertRaisesRegex(ValueError, "without QA"):
            build_context_plan(envelope([3]), source_sessions=sessions, segments=segments)

    def test_noncontiguous_segment_and_missing_member_rejected(self):
        sessions, segments = data()
        segments[3]["source_turn_ids"] = ["D1:4", "D1:6"]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.plan([3], segments=segments)
        segments = data()[1]
        segments[3]["source_turn_ids"] = ["D1:4", "D1:5"]
        with self.assertRaisesRegex(ValueError, "complete segment union"):
            self.plan([3], segments=segments)

    def test_native_character_limit_is_explicit_and_keeps_body(self):
        sessions, _ = data()
        sessions[0]["messages"][1]["content"] += "x" * 700
        p = self.plan([3], sessions=sessions)
        r = p["history_context"][0]
        self.assertGreater(r["native_omitted_characters"], 0)
        self.assertEqual(len(r["native_visible_content"]), 303)
        self.assertIn("Bob: raw turn 1.", r["native_visible_content"])
        self.assertIn("date=2023-05-01", r["native_visible_content"])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jev_source_local_")
        self.addCleanup(self.tmp.cleanup)
        for name, value in {
            "lemmatize_for_bm25": lambda t: t.lower(), "extract_entities_batch": lambda ts: [[] for _ in ts],
            "capture_event": lambda *a, **k: None, "detect_scale_threshold_from_add_result": lambda *a: None,
            "detect_temporal_usage_from_metadata": lambda *a: None, "display_first_run_notice": lambda *a: None,
        }.items():
            manager = patch.object(fixtures.module, name, value)
            manager.start()
            self.addCleanup(manager.stop)

    def adapter(self, responses, sessions=None):
        source, segments = data()
        memory = fixtures.fake_memory(responses)
        memory.embedding_model = Embedding()
        return memory, SourceLocalMem0GroupAdapter(memory, self.tmp.name, current_date=frozen.CURRENT_DATE,
                                                   source_sessions=sessions or source, segments=segments)

    def test_native_messages_embedding_system_unchanged_context_writer_only(self):
        memory, writer = self.adapter([{"memory": [{"text": "Alice fact."}]}])
        native_embedding = memory.embedding_model
        env = envelope([3, 11])
        result = writer.add_group(user_id="c0", **env)
        messages = memory.llm.calls[0]["messages"]
        self.assertEqual(messages[0]["content"], ADDITIVE_EXTRACTION_PROMPT)
        prompt = messages[1]["content"]
        self.assertIn("[SEGMENT s3]", prompt)
        self.assertIn("[SEGMENT s11]", prompt)
        self.assertIn("[CONTEXT_ONLY target=s3 turn=D1:2", prompt)
        self.assertIn("Do not extract new standalone facts", prompt)
        self.assertNotIn("INCORRECT_DB_HISTORY", prompt)
        self.assertEqual(memory.db.saved[0][0], env["messages"])
        self.assertEqual(native_embedding.calls[0], (parse_messages(env["messages"]), "search"))
        self.assertIsNone(result["fact_records"][0]["observation_date"])
        fact = result["fact_records"][0]
        self.assertEqual(fact["input_scope"]["source_turn_ids"], ["D1:4", "D2:4"])
        self.assertEqual(fact["input_scope"]["historical_turn_ids"], ["D1:2", "D1:3", "D2:2", "D2:3"])
        self.assertEqual(fact["verified_support"], [])
        disk = json.loads(Path(result["audit_path"]).read_text(encoding="utf8"))
        self.assertEqual(disk["source_local_context"], result["source_local_context"])
        self.assertEqual(disk["effective_request_sha256"], result["effective_request_sha256"])
        self.assertEqual(disk["fact_records"], result["fact_records"])

    def test_nonlocal_prior_group_does_not_become_history(self):
        memory, writer = self.adapter([{"memory": []}, {"memory": []}])
        writer.add_group(user_id="c0", **envelope([0, 14], "first"))
        result = writer.add_group(user_id="c0", **envelope([3], "second"))
        self.assertEqual([r["source_turn_id"] for r in result["history_context"]], ["D1:2", "D1:3"])
        prompt = memory.llm.calls[1]["messages"][1]["content"]
        history = prompt.split("## Last k Messages\n", 1)[1].split("## Recently Extracted Memories", 1)[0]
        self.assertNotIn("raw turn 14", history)
        self.assertFalse(result["source_local_context"]["online_causal_claim"])
        self.assertTrue(result["source_local_context"]["existing_memory_retrieval_unchanged"])

    def test_success_replay_is_bound_to_local_source_context(self):
        memory, writer = self.adapter([{"memory": []}])
        first = writer.add_group(user_id="c0", **envelope([3]))
        resumed = writer.add_group(user_id="c0", **envelope([3]))
        self.assertTrue(resumed["replayed"])
        self.assertEqual(len(memory.llm.calls), 1)
        self.assertEqual(first["effective_request_sha256"], resumed["effective_request_sha256"])
        writer.source_sessions[0]["messages"][1]["content"] += " changed context"
        with self.assertRaisesRegex(frozen.original.SessionRecoveryRequired, "context changed"):
            writer.add_group(user_id="c0", **envelope([3]))

    def test_hooks_restore_after_failure_and_no_retry(self):
        memory, writer = self.adapter(["bad json"])
        builder, publish = frozen.adapt_group_builder, frozen.publish_json
        with self.assertRaises(Exception):
            writer.add_group(user_id="c0", **envelope([3]))
        self.assertIs(frozen.adapt_group_builder, builder)
        self.assertIs(frozen.publish_json, publish)
        self.assertIs(fixtures.module.generate_additive_extraction_prompt, generate_additive_extraction_prompt)
        with self.assertRaises(frozen.original.SessionRecoveryRequired):
            writer.add_group(user_id="c0", **envelope([3]))
        self.assertEqual(len(memory.llm.calls), 1)
        record = json.loads(next(Path(self.tmp.name).glob("*/*.json")).read_text(encoding="utf8"))
        self.assertEqual(record["adapter_protocol"], VERSION)
        self.assertEqual(record["status"], "requires_recovery")


if __name__ == "__main__":
    unittest.main()
