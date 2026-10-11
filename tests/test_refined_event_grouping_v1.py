"""Offline invariants for identity checks and bounded raw-turn boundary repairs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_mem import global_segment_grouping as global_groups
from jev_mem.refined_event_grouping_v1 import (
    BOUNDARY_STAGE, PAIR_STAGE, PAIR_CRITERIA, RefinedGroupingError,
    refine_partial_segments, run_identity_only, run_refined_grouping,
)
from jev_mem.sparse_topic_pipeline import _group_record
from jev_mem.topic_grouping import verify_partition
from tools import run_refined_event_grouping_v1 as driver


def fixture(sizes=(12, 12), *, one_session=False, top_k=5):
    raw, segments, vectors = [], [], []
    for si, size in enumerate(sizes):
        start = len(raw)
        for ti in range(size):
            raw.append({"source_turn_id": f"s{si}_t{ti}",
                        "session_id": "session_0" if one_session else f"session_{si}",
                        "speaker": "Alice" if ti % 2 else "Bob",
                        "observation_date": "2026-09-01" if one_session else f"2026-09-{si+1:02d}",
                        "text": f"Original exact words {si}/{ti}",
                        "caption": f"caption {si}/{ti}"})
            # A real local change near the middle; the two sessions have the same shape.
            vectors.append(([1., 0.] if ti < size//2 else [0., 1.]) + [0.]*766)
        segment = _group_record(raw, list(range(start, len(raw))), f"segment_{si:04d}")
        segment["segment_id"] = segment["group_id"]
        segments.append(segment)
    plan = global_groups.prepare_candidates(0, segments, vectors, top_k=top_k)
    return raw, vectors, plan


class Judge:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def score_many(self, state, criteria, *, stage):
        parsed = json.loads(state)
        self.calls.append({"state": parsed, "criteria": deepcopy(criteria), "stage": stage})
        if self.answer:
            return self.answer(parsed, criteria, stage)
        if stage == BOUNDARY_STAGE:
            return {"topic_shift": .95, "right_depends_on_left": .1}
        return {"same_concrete_event": .9, "identity_conflict": .1, "partial_overlap": .9}


class RefinedGroupingTests(unittest.TestCase):
    def test_identity_veto_does_not_replace_raw_score(self):
        raw, _, plan = fixture()
        judge = Judge(lambda *_: {"same_concrete_event": .95,
                                   "identity_conflict": .8, "partial_overlap": .2})
        result = run_identity_only(plan, judge=judge)
        self.assertEqual(len(result["groups"]), 2)
        pair = next(iter(result["trace"]["pair_records"].values()))
        self.assertEqual(pair["scores"]["same_concrete_event"], .95)
        self.assertEqual(pair["effective_merge_score"], 0.)
        self.assertTrue(pair["identity_veto"])
        self.assertTrue(verify_partition(raw, result["groups"]))

    def test_partial_overlap_diagnostic_only_in_identity_arm(self):
        raw, _, plan = fixture()
        result = run_identity_only(plan, judge=Judge())
        self.assertEqual(len(result["groups"]), 1)
        self.assertFalse(result["trace"]["partial_overlap_is_merge_veto"])
        self.assertTrue(result["groups"][0]["cross_date"])
        self.assertIsNone(result["groups"][0]["observation_date"])
        self.assertEqual(result["groups"][0]["turns"], raw)

    def test_full_source_fields_and_multicriteria_are_sent(self):
        _, _, plan = fixture()
        judge = Judge()
        run_identity_only(plan, judge=judge)
        call = judge.calls[0]
        self.assertEqual(call["criteria"], PAIR_CRITERIA)
        self.assertEqual(call["stage"], PAIR_STAGE)
        self.assertEqual(call["state"]["segments"][0]["turns"], plan["segments"][0]["turns"])
        self.assertEqual(call["state"]["segments"][1]["turns"], plan["segments"][1]["turns"])

    def test_legacy_single_criterion_or_invalid_scores_fail_not_zero(self):
        _, _, plan = fixture()
        invalid = [{"same_concrete_event": .9}]
        for bad in (True, None, float("nan"), float("inf"), -.1, 1.1):
            invalid.append({"same_concrete_event": .9, "identity_conflict": bad, "partial_overlap": .5})
        for scores in invalid:
            with self.subTest(scores=scores), self.assertRaises(RefinedGroupingError) as caught:
                run_identity_only(plan, judge=Judge(lambda *_, value=scores: value))
            self.assertEqual(caught.exception.trace["requests"][0]["status"], "failed")
            self.assertNotIn("scores", caught.exception.trace["requests"][0])

    def test_durable_events_started_then_complete_and_exact_replay(self):
        _, _, plan = fixture()
        events = []
        first = run_identity_only(plan, judge=Judge(), on_progress=events.append)
        self.assertEqual([e["status"] for e in events], ["started", "complete"])
        self.assertNotIn("scores", events[0])
        cache = {e["request_sha256"]: e for e in first["trace"]["requests"]}
        judge = Judge(lambda *_: self.fail("Exact replay must avoid the judge hook"))
        second = run_identity_only(plan, judge=judge, replay_events=cache)
        self.assertEqual(second["groups"], first["groups"])
        self.assertEqual(second["trace"]["exact_event_replays"], 1)
        self.assertEqual(second["trace"]["judge_hook_calls"], 0)

    def test_replay_rejects_wrong_model_stage_state_criteria_and_failed(self):
        _, _, plan = fixture()
        first = run_identity_only(plan, judge=Judge())
        event = first["trace"]["requests"][0]
        for field, value in (("model", "different"), ("stage", "old"), ("state", "different"),
                             ("criteria", {"same_concrete_event": "old criterion"}),
                             ("status", "failed")):
            bad = deepcopy(event)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(RefinedGroupingError):
                run_identity_only(plan, judge=Judge(), replay_events={event["request_sha256"]: bad})

    def test_complete_link_supplements_with_new_identity_criteria(self):
        _, _, plan = fixture((8, 8, 8), top_k=1)
        judge = Judge()
        result = run_identity_only(plan, judge=judge, parameters={"top_k": 1})
        self.assertEqual(result["trace"]["initial_pairs_scored"], 2)
        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(len(result["groups"]), 1)
        self.assertTrue(all(call["criteria"] == PAIR_CRITERIA for call in judge.calls))

    def test_boundary_budget_shared_across_segments_same_original_session(self):
        raw, vectors, plan = fixture((12, 12, 12), one_session=True)
        identity = run_identity_only(plan, judge=Judge())
        judge = Judge()
        before = deepcopy(plan)
        result = refine_partial_segments(plan, identity, vectors, judge=judge)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(len(result["trace"]["sessions"]), 1)
        self.assertEqual(len(result["segments"]), 5)
        self.assertTrue(verify_partition(raw, result["segments"]))
        self.assertEqual(plan, before)
        self.assertTrue(result["trace"]["unassessed_gaps_are_not_certified_same_topic"])
        self.assertFalse(result["trace"]["convergence_guaranteed"])

    def test_boundary_window_dependency_veto_and_budget_per_session(self):
        raw, vectors, plan = fixture()
        identity = run_identity_only(plan, judge=Judge())
        judge = Judge(lambda *_: {"topic_shift": .99, "right_depends_on_left": .5})
        result = refine_partial_segments(plan, identity, vectors, judge=judge)
        self.assertEqual(len(judge.calls), 4)
        self.assertEqual(result["segments"], plan["segments"])
        self.assertTrue(all(d["dependency_veto"] for d in result["trace"]["decisions"]))
        for call in judge.calls:
            self.assertLessEqual(len(call["state"]["left_window"]), 4)
            self.assertLessEqual(len(call["state"]["right_window"]), 4)
            self.assertEqual(set(call["criteria"]), {"topic_shift", "right_depends_on_left"})
        self.assertTrue(verify_partition(raw, result["segments"]))

    def test_short_or_nonpartial_segments_never_trigger_extra_calls(self):
        for sizes, partial in (((7, 7), .9), ((12, 12), .1)):
            _, vectors, plan = fixture(sizes)
            identity = run_identity_only(plan, judge=Judge(lambda *_: {
                "same_concrete_event": .9, "identity_conflict": .1, "partial_overlap": partial}))
            judge = Judge(lambda *_: self.fail("No eligible segment should call the boundary judge"))
            result = refine_partial_segments(plan, identity, vectors, judge=judge)
            self.assertEqual(result["trace"]["logical_requests"], 0)
            self.assertEqual(result["segments"], plan["segments"])

    def test_end_to_end_no_change_replays_identity_pairs(self):
        _, vectors, plan = fixture()
        identity = run_identity_only(plan, judge=Judge(lambda *_: {
            "same_concrete_event": .9, "identity_conflict": .1, "partial_overlap": .1}))
        judge = Judge(lambda *_: self.fail("Unchanged full requests should replay"))
        final = run_refined_grouping(plan, identity, vectors, judge=judge)
        self.assertEqual(final["groups"], identity["groups"])
        self.assertEqual(final["trace"]["overall_judge_hook_calls_excluding_source_identity"], 0)
        self.assertEqual(final["trace"]["exact_event_replays"], 1)

    def test_changed_pairs_are_rejudged_and_all_source_survives(self):
        raw, vectors, plan = fixture()
        identity = run_identity_only(plan, judge=Judge())
        judge = Judge()
        final = run_refined_grouping(plan, identity, vectors, judge=judge)
        self.assertEqual(len(final["segments"]), 6)
        self.assertEqual(len([c for c in judge.calls if c["stage"] == BOUNDARY_STAGE]), 4)
        self.assertEqual(len([c for c in judge.calls if c["stage"] == PAIR_STAGE]), 15)
        self.assertEqual(final["trace"]["exact_event_replays"], 0)
        self.assertTrue(verify_partition(raw, final["groups"]))
        self.assertEqual(final["refined_candidate_plan"]["top_k"], 5)


class FakeCachedAPI:
    """Raw provider fixtures, never a socket or credential lookup."""
    hook_count = 0
    fail_after = None

    def __init__(self, run_dir, *, context=None, **_):
        self.run_dir = Path(run_dir)
        self.context = context or {}
        self.events = []

    def _redact(self, value):
        return str(value)

    def scoped(self, **context):
        return type(self)(self.run_dir, context={**self.context, **context})

    def score_many(self, state, criteria, *, stage):
        import hashlib
        type(self).hook_count += 1
        if self.fail_after is not None and type(self).hook_count >= self.fail_after:
            raise TimeoutError("mock provider failure")
        scores = ({"topic_shift": .95, "right_depends_on_left": .1} if stage == BOUNDARY_STAGE else
                  {"same_concrete_event": .9, "identity_conflict": .1, "partial_overlap": .9})
        event = {"state": state, "criteria": criteria}
        signature = driver._provider_signature(event)
        request_hash = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.run_dir / "api_cache" / (request_hash + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"request": signature, "response": {"model": "jev-1.13.0", "answers": {
            key: {"noul": value} for key, value in scores.items()}}}
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.events.append({"request_hash": request_hash, "stage": stage,
                            "provider": "jev", "model": "jev-1.13.0", "operation": "systemone",
                            "context": self.context, "cache_path": str(path),
                            "cache_sha256": driver.sha(path), "error_type": None})
        return scores


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "manifest.json").write_text('{"offline_mock": true}', encoding="utf-8")
        self.raw, vectors, self.plan = fixture()
        embedding_path = self.root / "source_embeddings.json"
        embedding_path.write_text(json.dumps({"vectors": vectors}), encoding="utf-8")
        self.plan["source_embeddings"] = driver.pointer(embedding_path)
        self.parent = {"plans": {"0": {"path": "mock_source_only", "sha256": "mock"}}}
        FakeCachedAPI.hook_count = 0
        FakeCachedAPI.fail_after = None

    def tearDown(self):
        self.temp.cleanup()

    def execute(self):
        with patch.object(driver, "load_plan", return_value=self.plan), \
                patch.object(driver, "CachedLegacyAPI", FakeCachedAPI):
            return driver.run_conversation(self.root, 0, {}, self.parent, {}, lambda *a, **kw: None)

    def test_driver_persists_both_arms_and_binds_all_raw_responses(self):
        result = self.execute()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["artifacts"]), set(driver.ARMS))
        self.assertEqual(result["counts"]["source_segments"], 2)
        self.assertEqual(result["counts"]["refined_segments"], 6)
        self.assertEqual(result["counts"]["boundary_requests"], 4)
        self.assertEqual(len(result["events"]), 20)
        for ref in result["events"]:
            event = driver.validate_event(ref["path"], self.root / "conv_0",
                                          driver.sha(self.root / "manifest.json"), 0)
            self.assertEqual(event["status"], "complete")

    def test_driver_complete_terminal_reuses_without_any_hooks(self):
        first = self.execute()
        calls = FakeCachedAPI.hook_count
        second = self.execute()
        self.assertEqual(second["counts"], first["counts"])
        self.assertEqual(FakeCachedAPI.hook_count, calls)

    def test_driver_unknown_or_failed_intent_not_reentered(self):
        FakeCachedAPI.fail_after = 2
        with self.assertRaises(RefinedGroupingError):
            self.execute()
        original = (self.root / "conv_0/result.json").read_bytes()
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual((self.root / "conv_0/result.json").read_bytes(), original)
        self.assertEqual(FakeCachedAPI.hook_count, 2)
        self.assertTrue(list((self.root / "conv_0/failed_events").glob("*.json")))

    def test_driver_rejects_raw_tampering_before_reuse(self):
        result = self.execute()
        event_wrapper = driver.read(result["events"][0]["path"])
        raw_path = Path(event_wrapper["api_events"][0]["cache_path"])
        raw = driver.read(raw_path)
        raw["response"]["answers"][next(iter(raw["response"]["answers"]))]["noul"] = .123
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.execute()


class RefinementFailureTests(unittest.TestCase):
    def test_failed_boundary_preserves_trace_never_returns_partial_partition(self):
        _, vectors, plan = fixture()
        identity = run_identity_only(plan, judge=Judge())
        def fail(*_):
            raise TimeoutError("Do not expose arbitrary provider error text")
        with self.assertRaises(RefinedGroupingError) as caught:
            refine_partial_segments(plan, identity, vectors, judge=Judge(fail))
        self.assertEqual(caught.exception.trace["judge_hook_calls"], 1)
        event = caught.exception.trace["requests"][0]
        self.assertEqual(event["error_type"], "TimeoutError")
        self.assertNotIn("error", event)
        self.assertNotIn("scores", event)

    def test_qa_contamination_plan_binding_and_embedding_changes_refused(self):
        _, vectors, plan = fixture()
        identity = run_identity_only(plan, judge=Judge())
        changed = deepcopy(plan)
        changed["segments"][0]["turns"][0]["question"] = "must not be forwarded"
        with self.assertRaises(ValueError):
            run_identity_only(changed, judge=Judge())
        altered_vectors = deepcopy(vectors)
        altered_vectors[0][0] = .7
        with self.assertRaises(ValueError):
            refine_partial_segments(plan, identity, altered_vectors, judge=Judge())
        bad_identity = deepcopy(identity)
        bad_identity["trace"]["source_plan_sha256"] = "bad"
        with self.assertRaises(ValueError):
            refine_partial_segments(plan, bad_identity, vectors, judge=Judge())

    def test_excess_budget_and_wrong_topk_rejected_before_hooks(self):
        _, _, plan = fixture()
        judge = Judge()
        with self.assertRaises(ValueError):
            run_identity_only(plan, judge=judge, parameters={"extra_boundaries_per_original_session": 3})
        with self.assertRaises(ValueError):
            run_identity_only(plan, judge=judge, parameters={"top_k": 2})
        self.assertEqual(judge.calls, [])

    def test_persistence_failure_is_fail_closed(self):
        _, _, plan = fixture()
        judge = Judge()
        def cannot_save(_):
            raise OSError("disk failure")
        with self.assertRaises(RefinedGroupingError) as caught:
            run_identity_only(plan, judge=judge, on_progress=cannot_save)
        self.assertEqual(judge.calls, [])
        self.assertEqual(caught.exception.trace["requests"][0]["status"], "failed")

    def test_single_segment_makes_no_model_calls(self):
        raw, vectors, plan = fixture((12,))
        judge = Judge(lambda *_: self.fail("No pairs"))
        identity = run_identity_only(plan, judge=judge)
        final = run_refined_grouping(plan, identity, vectors, judge=judge)
        self.assertEqual(identity["trace"]["logical_requests"], 0)
        self.assertEqual(final["trace"]["logical_requests"], 0)
        self.assertTrue(verify_partition(raw, final["groups"]))


if __name__ == "__main__":
    unittest.main()
