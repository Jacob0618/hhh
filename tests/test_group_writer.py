"""Offline source/date integrity and transactional tests for topic-group writing."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem.group_writer import GroupFactWriter, GroupWriteError, prepare_group


def source_turns():
    return [{"source_turn_id": f"t{i}", "session_id": f"s{i // 2}", "speaker": "Alice" if i % 2 else "Bob",
             "observation_date": f"2023-01-0{i // 2 + 1}", "text": f"Original turn {i}.",
             "role": "user" if i % 2 else "assistant"} for i in range(6)]


def response(ids, text="Alice changed a plan.", status="grounded"):
    return {"facts": [{"text": text, "source_turn_ids": ids, "temporal_status": status}]}


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.generations, self.embedding_calls, self.counts = [], [], []
        self.fail_embedding_once = False
        self.bad_vector = False

    def count_tokens(self, text, **kwargs):
        self.counts.append((text, kwargs))
        return len(text.split())

    def generate(self, prompt, **kwargs):
        self.generations.append((json.loads(prompt), kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def embed(self, texts, **kwargs):
        self.embedding_calls.append((list(texts), kwargs))
        if self.fail_embedding_once:
            self.fail_embedding_once = False
            raise RuntimeError("offline embedding failure")
        return [[float("nan") if self.bad_vector else 2.] + [0.] * 767 for _ in texts]


class GroupWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev_group_writer_")
        self.addCleanup(self.temp.cleanup)
        self.source = source_turns()

    def writer(self, responses):
        api = FakeAPI(responses)
        return api, GroupFactWriter(api, self.temp.name)

    def add(self, writer, indices=(2, 5), group="g1", **kwargs):
        return writer.add_group(user_id=kwargs.pop("user_id", "c1"), group_id=group,
                                members=[self.source[i] for i in indices], raw_turns=self.source,
                                grouping_method=kwargs.pop("grouping_method", "jev_topics"), **kwargs)

    def test_dates_and_original_text_with_member_context_support(self):
        untouched = copy.deepcopy(self.source)
        api, writer = self.writer([response(["t1", "t2", "t5"], status="uncertain")])
        record = self.add(writer)
        prompt, settings = api.generations[0]
        self.assertNotIn("observation_date", prompt)
        self.assertEqual(prompt["MEMBERS"], [self.source[2], self.source[5]])
        self.assertEqual(prompt["member_context_ids"], {"t2": ["t0", "t1"], "t5": ["t3", "t4"]})
        self.assertEqual(self.source, untouched)
        self.assertEqual(len(api.generations), 1)
        self.assertEqual(settings["model"], "gemini-2.5-flash")
        fact = record["fact_records"][0]
        self.assertIsNone(fact["observation_date"])
        self.assertEqual(fact["source_observation_dates"]["t5"], "2023-01-03")
        self.assertEqual(fact["temporal_status"], "uncertain")
        self.assertFalse(fact["temporal_status_verified"])
        self.assertEqual(fact["verified_support"], [])
        self.assertEqual(fact["embedding"][0], 1.)

    def test_context_is_deduplicated_and_members_never_context(self):
        prepared = prepare_group(members=[self.source[i] for i in (1, 2, 3)], raw_turns=self.source)
        self.assertEqual(prepared["context_source_turn_ids"], ["t0"])
        self.assertEqual(prepared["payload"]["member_context_ids"], {"t1": ["t0"], "t2": ["t0"], "t3": []})

    def test_context_only_fact_is_rejected(self):
        _, writer = self.writer([response(["t0"])])
        with self.assertRaises(GroupWriteError):
            self.add(writer)
        self.assertEqual(writer.get_facts(), [])

    def test_context_must_relate_to_a_cited_member(self):
        _, writer = self.writer([response(["t2", "t4"])])
        with self.assertRaises(GroupWriteError):
            self.add(writer)
        self.assertEqual(writer.get_facts(), [])

    def test_unseen_source_rejected(self):
        _, writer = self.writer([response(["t2", "future_unknown"])])
        with self.assertRaises(GroupWriteError):
            self.add(writer)

    def test_extra_context_tokens_and_turns_are_recorded(self):
        api, writer = self.writer([response(["t2"])])
        record = self.add(writer)
        cost = record["context_cost"]
        self.assertEqual(cost["extra_unique_context_turns"], 4)
        self.assertEqual(cost["context_links"], 4)
        self.assertGreater(cost["extra_prompt_text_tokens_from_context"], 0)
        self.assertFalse(cost["includes_system_prompt_or_provider_framing"])
        self.assertEqual(len(api.counts), 2)

    def test_all_member_group_has_no_extra_context(self):
        _, writer = self.writer([{"facts": []}])
        record = self.add(writer, indices=tuple(range(6)), grouping_method="original_sessions")
        self.assertEqual(record["context_cost"]["extra_unique_context_turns"], 0)
        self.assertEqual(record["context_cost"]["extra_prompt_text_tokens_from_context"], 0)
        self.assertTrue(writer.coverage()["complete"])

    def test_success_resume_and_changed_input_rejection(self):
        api, writer = self.writer([response(["t2"])])
        first = self.add(writer)
        reopened = GroupFactWriter(api, self.temp.name)
        replay = self.add(reopened)
        self.assertTrue(replay["replayed"])
        self.assertEqual(first["fact_records"], replay["fact_records"])
        self.assertEqual(len(api.generations), 1)
        self.assertEqual(len(api.counts), 2)
        with self.assertRaises(GroupWriteError):
            self.add(reopened, indices=(2,))

    def test_embedding_retry_reuses_extraction_and_token_checks(self):
        api, writer = self.writer([response(["t2"])])
        api.fail_embedding_once = True
        with self.assertRaises(RuntimeError):
            self.add(writer)
        self.assertEqual(writer.get_facts(), [])
        reopened = GroupFactWriter(api, self.temp.name)
        self.add(reopened)
        self.assertEqual(len(api.generations), 1)
        self.assertEqual(len(api.counts), 2)
        self.assertEqual(len(api.embedding_calls), 2)

    def test_commit_cleanup_crash_resumes_without_duplicates(self):
        api, writer = self.writer([response(["t2"])])
        with patch.object(writer, "_finish", side_effect=OSError("offline simulated crash")):
            with self.assertRaises(OSError):
                self.add(writer)
        self.assertEqual(len(writer.get_facts()), 1)
        reopened = GroupFactWriter(api, self.temp.name)
        self.assertTrue(self.add(reopened)["replayed"])
        self.assertEqual(len(api.generations), 1)
        self.assertFalse((Path(self.temp.name) / "pending_group.json").exists())

    def test_overlapping_group_and_mixed_method_rejected(self):
        _, writer = self.writer([response(["t0"])])
        self.add(writer, indices=(0, 2))
        with self.assertRaises(GroupWriteError):
            self.add(writer, indices=(2, 3), group="g2")
        with self.assertRaises(GroupWriteError):
            self.add(writer, indices=(3, 4), group="g2", grouping_method="embedding_topics")
        self.assertEqual(writer.coverage()["missing_source_turn_ids"], ["t1", "t3", "t4", "t5"])

    def test_no_fact_limit(self):
        data = {"facts": [{"text": f"Item {i}.", "source_turn_ids": ["t2"], "temporal_status": "not_temporal"} for i in range(25)]}
        _, writer = self.writer([data])
        self.assertEqual(len(self.add(writer)["fact_records"]), 25)

    def test_qa_and_source_rewrite_rejected_before_api(self):
        api, writer = self.writer([])
        altered = copy.deepcopy(self.source[2])
        altered["text"] = "Rewritten source"
        with self.assertRaises(ValueError):
            writer.add_group(user_id="c1", group_id="g1", members=[altered], raw_turns=self.source,
                             grouping_method="jev_topics")
        self.source[0]["gold"] = "answer"
        with self.assertRaises(ValueError):
            self.add(writer)
        self.assertEqual(api.generations, [])

    def test_bad_vector_and_missing_temporal_status_fail_closed(self):
        api, writer = self.writer([response(["t2"])])
        api.bad_vector = True
        with self.assertRaises(GroupWriteError):
            self.add(writer)
        self.assertEqual(writer.get_facts(), [])


if __name__ == "__main__":
    unittest.main()
