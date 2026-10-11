"""Independent schema/scope regressions for the immutable prepare failure."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import run_five_baselines_amem_native_v3_admission_v2 as w

ROOT=Path(__file__).resolve().parents[1]
QUAL=ROOT/'research/baseline_execution_2026-10-08/amem_format_v3/failure_qualification_v4.json'


class IndependentAdmissionSchemaTests(unittest.TestCase):
    def fixture(self):
        q=json.loads(QUAL.read_text(encoding='utf-8'))
        record=copy.deepcopy(q['certificates']['1'])
        report=json.loads(Path(record['review_path']).read_text(encoding='utf-8'))
        return record,report

    def altered(self, transform):
        record,report=self.fixture();transform(report)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'changed-review.json';path.write_text(json.dumps(report),encoding='utf-8')
            record['review_path']=str(path);record['review_sha256']=w.sha(path)
            with patch.object(w,'ROOT',Path(directory)), self.assertRaises(ValueError):
                w.validate_nested_review(record)

    def test_all_six_actual_nested_reports_are_accepted_with_their_actual_scope(self):
        q=json.loads(QUAL.read_text(encoding='utf-8'))
        for ci,record in q['certificates'].items():
            with self.subTest(conversation=ci):w.validate_nested_review(record)

    def test_snapshot_cannot_be_relabeled_terminal_passed(self):
        self.altered(lambda r:r.update(status='passed'))

    def test_unrecognized_schema_even_with_passed_is_rejected(self):
        self.altered(lambda r:r.update(schema='arbitrary-report',status='passed'))

    def test_snapshot_binding_error_blocks_admission(self):
        self.altered(lambda r:r.update(binding_errors=['unknown intent']))

    def test_missing_known_failed_call_blocks_admission(self):
        self.altered(lambda r:r.update(failed_native_calls=[]))

    def test_changed_failed_raw_binding_blocks_admission(self):
        self.altered(lambda r:r['failed_native_calls'][0].update(raw_sha256='0'*64))

    def test_missing_or_corrupt_logical_consumer_blocks_admission(self):
        for change in ('missing','sha','error'):
            def transform(r):
                row=r['failed_native_calls'][0]
                if change=='missing':row['logical_events']=[]
                elif change=='sha':row['logical_events'][0]['cache_sha256']='0'*64
                else:row['logical_events'][0]['error_type']='unknown'
            with self.subTest(change=change):self.altered(transform)


if __name__=='__main__':unittest.main()
