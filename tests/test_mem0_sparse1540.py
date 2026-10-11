"""Offline native-transfer contracts; no SDK instance, providers or real QA read."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_mem0_sparse1540 as target
from jev_mem.topic_grouping import raw_turns_from_locomo, render_group, TEMPORAL_INSTRUCTION


def source_item():
    # Deliberately insert sessions in reverse chronological order.
    return {"conversation": {
        "speaker_a": "Ada", "speaker_b": "Ben",
        "session_2_date_time": "2:00 pm on 2 January, 2023",
        "session_2": [{"dia_id": "D2:1", "speaker": "Ben", "text": "Tomorrow I leave."},
                      {"dia_id": "D2:2", "speaker": "Ada", "text": "Safe travels."}],
        "session_1_date_time": "2:00 pm on 1 January, 2023",
        "session_1": [{"dia_id": "D1:1", "speaker": "Ada", "text": "My painting.",
                       "blip_caption": "a blue bird", "query": "bird painting"},
                      {"dia_id": "D1:2", "speaker": "Ben", "text": "I like it."},
                      {"dia_id": "D1:3", "speaker": "Ada", "text": "A different topic."}]},
        "qa": [{"question": "DO_NOT_TRANSFER_QUESTION", "answer": "DO_NOT_TRANSFER_GOLD"}]}


def grouping(item=None, slices=None):
    item = source_item() if item is None else item
    raw = raw_turns_from_locomo(item["conversation"])
    slices = [[0, 1], [2], [3, 4]] if slices is None else slices
    groups = []
    for i, indices in enumerate(slices):
        turns = [copy.deepcopy(raw[j]) for j in indices]
        groups.append({"group_id": "sparse_topic_%04d" % i, "turns": turns,
            "source_turn_ids": [t["source_turn_id"] for t in turns], "text": render_group(turns),
            "cross_session": False, "cross_date": False, "segment_ids": ["segment_%d" % i]})
    return target._sealed({"status": "complete", "groups": groups})


def reseal(row):
    row = copy.deepcopy(row)
    row.pop("content_sha256", None)
    return target._sealed(row)


def pool(n=25):
    return [{"id": "fact_%d" % i, "text": "Unique fact %d." % i,
             "observation_date": "2023-01-01", "score": 1 - i / 100,
             "sources": ["mem0_hybrid"]} for i in range(n)]


QUESTION = {"question": "Which event?", "id": "c0_q0", "conversation": 0}


class NativeInputContracts(unittest.TestCase):
    def test_exact_partition_preserves_roles_images_dates_without_qa_or_group_prefix(self):
        result = target.group_envelopes(source_item(), grouping())
        self.assertEqual([g["source_turn_ids"] for g in result], [["D1:1", "D1:2"], ["D1:3"], ["D2:1", "D2:2"]])
        self.assertEqual([g["observation_date"] for g in result], ["2023-01-01", "2023-01-01", "2023-01-02"])
        originals = [m for s in target.session_envelopes(source_item()) for m in s["messages"]]
        actual = [m for g in result for m in g["messages"]]
        self.assertEqual(actual, originals)
        self.assertEqual(actual[0]["role"], "user")
        self.assertEqual(actual[1]["role"], "assistant")
        self.assertIn("The image shows: a blue bird", actual[0]["content"])
        self.assertNotIn("DO_NOT_TRANSFER", str(result))
        self.assertNotIn(TEMPORAL_INSTRUCTION, str(result))

    def test_missing_turn_rejected(self):
        with self.assertRaises(ValueError):
            target.group_envelopes(source_item(), grouping(slices=[[0, 1], [3, 4]]))

    def test_duplicate_turn_rejected(self):
        with self.assertRaises(ValueError):
            target.group_envelopes(source_item(), grouping(slices=[[0, 1], [1, 2], [3, 4]]))

    def test_rewritten_text_even_with_resealed_container_rejected(self):
        row = grouping()
        row["groups"][0]["turns"][0]["text"] = "Altered source"
        row["groups"][0]["text"] = render_group(row["groups"][0]["turns"])
        with self.assertRaises(ValueError):
            target.group_envelopes(source_item(), reseal(row))

    def test_turn_order_and_group_order_are_both_enforced(self):
        for slices in ([[1, 0], [2], [3, 4]], [[2], [0, 1], [3, 4]]):
            with self.subTest(slices=slices), self.assertRaises(ValueError):
                target.group_envelopes(source_item(), grouping(slices=slices))

    def test_cross_session_or_date_rejected_even_with_false_metadata_flags(self):
        with self.assertRaisesRegex(ValueError, "cross-session/date"):
            target.group_envelopes(source_item(), grouping(slices=[[0, 1], [2, 3, 4]]))

    def test_same_day_cross_session_is_still_rejected(self):
        item = source_item()
        item["conversation"]["session_2_date_time"] = "3:00 pm on 1 January, 2023"
        with self.assertRaisesRegex(ValueError, "cross-session/date"):
            target.group_envelopes(item, grouping(item, [[0, 1], [2, 3, 4]]))

    def test_declared_merges_rejected(self):
        for field, value in (("cross_date", True), ("cross_session", True), ("segment_ids", ["a", "b"])):
            row = grouping(); row["groups"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "not merged"):
                target.group_envelopes(source_item(), reseal(row))

    def test_bad_seal_and_noncomplete_rejected(self):
        row = grouping(); row["status"] = "partial"
        with self.assertRaises(ValueError):
            target.group_envelopes(source_item(), row)
        with self.assertRaisesRegex(ValueError, "complete"):
            target.group_envelopes(source_item(), reseal(row))


class NativeRetrievalContracts(unittest.TestCase):
    def test_rank_is_native_input_order_not_resort_by_score(self):
        candidates = pool(); candidates[24]["score"] = 100
        row = target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {})
        self.assertEqual([f["id"] for f in row["selected"]], ["fact_%d" % i for i in range(20)])
        self.assertEqual(row["pool_ids"], [f["id"] for f in candidates])
        self.assertEqual(row["pool_scores"], [f["score"] for f in candidates])
        self.assertEqual(row["selected"][0], target.fact_view(candidates[0]))
        self.assertEqual(row["selection_trace"]["judge_calls"], 0)
        target._verify_sealed(row)

    def test_identical_text_dedup_still_delivers20_unique(self):
        candidates = pool()
        candidates.insert(1, {**candidates[0], "id": "alternate_id", "text": " Unique   fact 0. "})
        row = target.make_retrieval(QUESTION, target.ARMS[1], candidates, "bankhash", {})
        self.assertEqual([f["id"] for f in row["selected"]], ["fact_%d" % i for i in range(20)])
        self.assertNotIn("alternate_id", [f["id"] for f in row["selected"]])

    def test_insufficient_unique_candidates_does_not_pad(self):
        candidates = pool(19) + [{**pool(1)[0], "id": "dupe"}]
        with self.assertRaisesRegex(ValueError, "20 unique"):
            target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {})

    def test_id_collision_different_text_rejected(self):
        candidates = pool(); candidates[-1]["id"] = candidates[0]["id"]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {})

    def test_full_replay_accepts_exact_pool_and_rejects_different_identity(self):
        candidates = pool(); bank = {"facts": candidates}
        row = target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {})
        self.assertEqual(target.validate_retrieval(row, QUESTION, target.ARMS[0], bank, "bankhash"), row)
        for q, arm, digest in (({**QUESTION, "id": "c0_q1"}, target.ARMS[0], "bankhash"),
                              (QUESTION, target.ARMS[1], "bankhash"),
                              (QUESTION, target.ARMS[0], "changedbank")):
            with self.subTest(q=q, arm=arm, digest=digest), self.assertRaises(ValueError):
                target.validate_retrieval(row, q, arm, bank, digest)

    def test_valid_bank_facts_in_wrong_rank_order_are_not_accepted(self):
        candidates = pool(); row = target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {})
        row["selected"][0], row["selected"][1] = row["selected"][1], row["selected"][0]
        with self.assertRaisesRegex(ValueError, "replay"):
            target.validate_retrieval(reseal(row), QUESTION, target.ARMS[0], {"facts": candidates}, "bankhash")

    def test_recorded_pool_budget_ids_scores_must_be_valid(self):
        candidates = pool(61)
        original = target.make_retrieval(QUESTION, target.ARMS[0], candidates[:25], "bankhash", {})
        bad_rows = []
        row = copy.deepcopy(original); row["pool_ids"][0] = "unknown"; bad_rows.append(row)
        row = copy.deepcopy(original); row["pool_ids"][1] = row["pool_ids"][0]; bad_rows.append(row)
        row = copy.deepcopy(original); row["pool_scores"].pop(); bad_rows.append(row)
        row = copy.deepcopy(original); row["pool_scores"][0] = True; bad_rows.append(row)
        row = target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {}); bad_rows.append(row)
        for row in bad_rows:
            with self.subTest(ids=len(row["pool_ids"])), self.assertRaises(ValueError):
                target.validate_retrieval(reseal(row), QUESTION, target.ARMS[0], {"facts": candidates}, "bankhash")

    def test_canonical_evidence_date_mutation_rejected_even_with_fresh_seal(self):
        candidates = pool(); row = target.make_retrieval(QUESTION, target.ARMS[0], candidates, "bankhash", {})
        row["selected"][0]["observation_date"] = "2030-01-01"
        with self.assertRaisesRegex(ValueError, "replay"):
            target.validate_retrieval(reseal(row), QUESTION, target.ARMS[0], {"facts": candidates}, "bankhash")


class FrozenManifestContracts(unittest.TestCase):
    def test_code_dependency_and_cohort_change_rejected_before_models(self):
        manifest = {"policy": target.POLICY, "source_dataset_sha256": target.SOURCE_HASH,
                    "code_sha256": {"code": "h1"}, "dependencies": {"dependency": "1"},
                    "question_ids": ["c0_q0"]}
        cases = [({"code": "changed"}, {"dependency": "1"}, [QUESTION]),
                 ({"code": "h1"}, {"dependency": "2"}, [QUESTION]),
                 ({"code": "h1"}, {"dependency": "1"}, [{**QUESTION, "id": "c0_q1"}])]
        for code, deps, qs in cases:
            with self.subTest(code=code, deps=deps, qs=qs), patch.object(target, "read", return_value=manifest), \
                    patch.object(target, "current_code", return_value=code), \
                    patch.object(target, "dependency_versions", return_value=deps), \
                    patch.object(target, "questions", return_value=qs), \
                    patch.object(target, "ModelAPI", side_effect=AssertionError("No provider allowed")), \
                    self.assertRaises(ValueError):
                target.verify_manifest()

    def test_existing_immutable_artifact_refuses_changed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            target.save_once(path, {"source": "original"})
            target.save_once(path, {"source": "original"})
            with self.assertRaisesRegex(ValueError, "Immutable"):
                target.save_once(path, {"source": "changed"})
            self.assertEqual(target.read(path), {"source": "original"})

    def test_prepared_input_or_seed_cache_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); folder = out / "conv_0"; folder.mkdir()
            target.publish_json(folder / "group_inputs.json", {"source": "original"})
            target.publish_json(folder / "session_bank.json", {"facts": []})
            seed_path = folder / "native/api_cache/request.json"
            target.publish_json(seed_path, {"response": {"embedding": [1]}})
            seed = {"source_sha256": target.sha(seed_path), "target_relative": seed_path.relative_to(out).as_posix()}
            target.publish_json(folder / "query_embedding_seed_lineage.json", {"entries": [seed]})
            manifest = {"policy": target.POLICY, "source_dataset_sha256": target.SOURCE_HASH,
                "code_sha256": {}, "dependencies": {}, "question_ids": [QUESTION["id"]],
                "conversations": [{"group_inputs_sha256": target.sha(folder / "group_inputs.json"),
                    "baseline_bank_sha256": target.sha(folder / "session_bank.json"),
                    "query_embedding_seed_lineage_sha256": target.sha(folder / "query_embedding_seed_lineage.json")}]}
            target.publish_json(out / "manifest.json", manifest)
            with patch.object(target, "OUT", out), patch.object(target, "current_code", return_value={}), \
                    patch.object(target, "dependency_versions", return_value={}), \
                    patch.object(target, "questions", return_value=[QUESTION]):
                self.assertEqual(target.verify_manifest(0), manifest)
                for path in (folder / "group_inputs.json", folder / "session_bank.json",
                             folder / "query_embedding_seed_lineage.json", seed_path):
                    original = path.read_bytes()
                    try:
                        path.write_bytes(original + b" ")
                        with self.subTest(path=path.name), self.assertRaises(ValueError):
                            target.verify_manifest(0)
                    finally:
                        path.write_bytes(original)

    def test_regroup_bank_must_match_committed_write_hash_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); folder = out / "conv_0"; folder.mkdir()
            target.publish_json(out / "manifest.json", {"protocol": "synthetic"})
            bank = {"arm": target.ARMS[1], "conversation": 0, "complete": True,
                    "source_dataset_sha256": target.SOURCE_HASH, "facts": [],
                    "manifest_sha256": target.sha(out / "manifest.json")}
            bank_path = folder / "regroup_bank.json"
            target.publish_json(bank_path, bank)
            state = {"status": "complete", "bank_sha256": target.sha(bank_path)}
            target.publish_json(folder / "write_status.json", state)
            with patch.object(target, "OUT", out):
                self.assertEqual(target.load_bank(0, target.ARMS[1], {})[0], bank)
                changed = {**bank, "extra_uncommitted": "change"}; target.publish_json(bank_path, changed)
                with self.assertRaisesRegex(ValueError, "committed hash"):
                    target.load_bank(0, target.ARMS[1], {})
                target.publish_json(bank_path, bank)
                target.publish_json(folder / "write_status.json", {**state, "status": "failed"})
                with self.assertRaisesRegex(ValueError, "not complete"):
                    target.load_bank(0, target.ARMS[1], {})


if __name__ == "__main__":
    unittest.main()
