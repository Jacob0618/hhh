"""Isolated writer-shaped failures and mock audit dependencies; no network."""
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import summarize_pilot400_reviewed_v1 as s
from tests.test_group_writer_v2 import FakeAPI, http, dump


class ReviewedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.folder = self.run / "conversations/conv_1/banks/regroup"
        self.raw = [{"source_turn_id": f"D1:{i}", "session_id": "s1", "speaker": "A",
                     "observation_date": "2022-01-01", "text": f"Turn{i}"} for i in range(5)]
        self.api = FakeAPI(self.run / "api_cache")
        bad = {"facts": [{"text": "Context-only", "source_turn_ids": ["D1:4", "D1:0"], "temporal_status": "not_temporal"}]}
        self.api.responses = [http(bad), http(bad)]
        with self.assertRaises(s.writer.GroupWriteError):
            s.writer.GroupFactWriter(self.api, self.folder).add_group(user_id="pilot400_conv_1", group_id="g",
                members=[self.raw[2], self.raw[4]], raw_turns=self.raw, grouping_method="jev_topics")
        dump(self.folder / "group_fact_bank.json", {"schema": s.writer.SCHEMA,
             "source_sha256": s.driver.fingerprint(self.raw), "groups": [], "facts": []})
        self.manifest = {"protocol_sha256": "protocol", "protocol": {"parameters": {"current_date": "2026-09-27"}}}
        dump(self.run / "run_manifest.json", self.manifest)
        self.questions = [{"id": "c1_q0", "question": "Q", "reference": "R", "category": 1, "conversation": 1}]
        self.bound = s.driver._bank_binding(self.manifest, 1, self.raw, "regroup")
        self.error = {"type": "GroupWriteError", "stage": "write", "message": "Context citation is not associated with any cited member"}
        def fail():
            raise s.writer.GroupWriteError(self.error["message"])
        s.driver.artifact(self.folder / "bank_summary.json", self.bound, fail, stage="write")
        s.driver.artifact(self.folder / "grouping.json", {**self.bound, "operation": "grouping"},
                          lambda: {"groups": s.driver.original_groups(self.raw)}, stage="grouping")
        (self.run / "api_calls.jsonl").write_text("", encoding="utf-8")
        report = self.root / "review.md"
        report.write_text("Explicit mock failure admission, not all-complete approval.", encoding="utf-8")
        self.entry = {"conversation": 1, "bank": "regroup", "failure_class": "terminal_feedback_validation_failure",
                      "error": self.error, "affected_arms": [s.driver.ARMS[2]], "affected_query_ids": ["c1_q0"],
                      "disposition": s.DISPOSITION,
                      **{key:s.base.sha(self.folder / name) for key,name in s.FILES.items()}}
        self.review = {"schema": s.REVIEW_SCHEMA, "status": "reviewed", "run_dir": str(self.run.resolve()),
            "protocol_sha256": "protocol", "manifest_sha256": s.base.sha(self.run / "run_manifest.json"),
            "source_dataset_sha256": s.driver.SOURCE_HASH, "cohort_sha256": s.driver.COHORT_SHA256,
            "reviewed_utc": datetime.now(timezone.utc).isoformat(), "reviewed_by": "mock auditor",
            "report_path": "review.md", "report_sha256": s.base.sha(report), "failed_banks": [self.entry]}
        self.review_path = self.root / "review.json"
        dump(self.review_path, self.review)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(s, "ROOT", self.root))
        self.stack.enter_context(patch.object(s.driver, "ensure_manifest", return_value=self.manifest))
        self.stack.enter_context(patch.object(s.driver, "load_data", return_value=[{"conversation": {}} for _ in range(10)]))
        self.stack.enter_context(patch.object(s.driver, "load_cohort", return_value=({}, self.questions)))
        self.stack.enter_context(patch.object(s.driver, "raw_turns_from_locomo", return_value=self.raw))

    def validate(self, final=False):
        return s.validate_bank_failure_review(self.run, self.review_path, require_failed_qa=final)

    def qa(self, **changes):
        q = self.questions[0]
        arm = s.driver.ARMS[2]
        query = s.driver.binding(self.manifest, 1, self.raw, query_id=q["id"], question_sha256=s.driver.fingerprint(q["question"]))
        bound = {**query, "arm": arm, "bank_sha256": None, "tag_summary_sha256": None,
                 "reference_sha256": s.driver.fingerprint(q["reference"])}
        base = {"arm": arm, "query_id": q["id"], "conversation": 1, "category": 1,
                "cohort_sha256": s.driver.COHORT_SHA256, "protocol_sha256": "protocol",
                "bank_path": "conversations/conv_1/banks/regroup/group_fact_bank.json",
                "bank_sha256": None, "selected_ids": [], "result": None}
        def failed():
            raise s.driver.RecordedFailure("write: " + self.error["message"])
        path = self.run / "conversations/conv_1/qa" / arm / "c1_q0.json"
        row = s.driver.artifact(path, bound, failed, stage="qa", base=base, schema=s.driver.QA_SCHEMA)
        row.update(changes)
        row = s.driver._seal({k:v for k,v in row.items() if k != "content_sha256"})
        dump(path,row)
        return row

    def test_before_qa_checks_only_bound_failures_not_unwritten_banks(self):
        checked = self.validate()
        self.assertEqual(checked["affected_arm_questions"],1)
        self.assertEqual(checked["failed_banks"][0]["partial_groups"],0)
        with self.assertRaisesRegex(ValueError,"Missing declared failed"):
            self.validate(final=True)

    def test_declared_failed_qa_accepted_without_loading_partial_bank_as_evidence(self):
        row = self.qa()
        self.assertEqual(self.validate(final=True)["affected_arm_questions"],1)
        metric = s.base.v2_metrics()
        status, scores = metric.validate_artifact(row,self.questions[0],s.driver.ARMS[2],self.run,s.driver.COHORT_SHA256,{},lambda *_:1.)
        self.assertEqual(status,"failed")
        self.assertIsNone(scores)

    def test_completed_answer_or_partial_selection_rejected(self):
        self.qa(status="complete", result={"answer":"fabricated"}, selected_ids=["partial"])
        with self.assertRaisesRegex(ValueError,"scored, selected"):
            self.validate(final=True)

    def test_reader_cache_event_for_failed_arm_is_forbidden(self):
        self.qa()
        dump_row = {"request_hash":"mock", "stage":s.READER_STAGE,"cache_hit":True,
                    "context":{"conversation":1,"method":s.driver.ARMS[2],"query_id":"c1_q0"}}
        (self.run / "api_calls.jsonl").write_text(json.dumps(dump_row)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"including cache hits"):
            self.validate(final=True)

    def test_other_arm_reader_call_is_permitted(self):
        self.qa()
        row={"request_hash":"mock","stage":s.READER_STAGE,"cache_hit":False,
             "timestamp_utc":datetime.now(timezone.utc).isoformat(),
             "context":{"conversation":1,"method":s.driver.ARMS[0],"query_id":"c1_q0"}}
        (self.run / "api_calls.jsonl").write_text(json.dumps(row)+"\n",encoding="utf-8")
        self.validate(final=True)

    def test_review_cannot_change_affected_question_set(self):
        self.review["failed_banks"][0]["affected_query_ids"]=["c1_other"]
        dump(self.review_path,self.review)
        with self.assertRaisesRegex(ValueError,"fixed arm/question cohort"):
            self.validate()

    def test_pending_unknown_cannot_be_disguised_as_validation_failure(self):
        p=s.read(self.folder / "pending_group.json")
        p["status"]="unknown_operation_no_reissue"
        dump(self.folder / "pending_group.json",p)
        self.review["failed_banks"][0]["pending_sha256"]=s.base.sha(self.folder / "pending_group.json")
        dump(self.review_path,self.review)
        with self.assertRaisesRegex(ValueError,"Unsupported failure class"):
            self.validate()

    def test_hash_tampering_rejected_before_admission(self):
        with (self.folder / "group_fact_bank.json").open("a",encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            self.validate()

    def test_unreviewed_other_bank_failure_is_not_caught_and_skipped(self):
        self.qa()
        with patch.object(s.driver,"load_bank",side_effect=ValueError("unreviewed failed bank")):
            with self.assertRaisesRegex(ValueError,"unreviewed failed bank"):
                s.audit_reviewed_banks(self.run,self.review_path)


if __name__ == "__main__":
    unittest.main()
