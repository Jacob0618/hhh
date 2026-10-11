"""Explicit scoped-snapshot schema compatibility, with unchanged v3 behavior."""
import copy
import json
from unittest.mock import patch
from tests import test_five_baselines_amem_native_v3 as original_tests
from tools import run_five_baselines_amem_native_v3_admission_v2 as revised
from tools.baselines_common_v1 import save_new,sha


class RevisedContinuationTests(original_tests.ContinuationTests):
    def setUp(self):
        self.module_patch=patch.object(original_tests,'v3',revised);self.module_patch.start()
        super().setUp()
    def tearDown(self):
        try:super().tearDown()
        finally:self.module_patch.stop()

    def snapshot(self):
        record={'conversation':0,'request_hash':self.key,'stage':'amem_analysis','native_call_sha256':'native',
                'original_raw_sha256':sha(self.raw),'native_operation':'note_000001_amem_analysis',
                'original_raw_path':str(self.raw),'native_call_path':str(self.source/'work/native/c0/calls/note.json'),
                'context':self.context,'review_path':str(self.root/'snapshot.json')}
        call={'conversation':0,'request_hash':record['request_hash'],'stage':record['stage'],
              'native_call_sha256':record['native_call_sha256'],'raw_sha256':record['original_raw_sha256'],
              'operation':record['native_operation'],'raw_path':record['original_raw_path'],
              'native_call':record['native_call_path'],'binding_passed':True,'complete_http_raw_known':True,
              'logical_events':[{'context':self.context,'stage':record['stage'],'error_type':None,'cache_sha256':record['original_raw_sha256']}]}
        report={'schema':'amem-live-failure-chain-independent-review-v1','status':'snapshot_review_complete_no_recovery_started',
                'binding_errors':[],'paid_api_calls_during_review':0,'credential_reads':0,'run_files_modified':0,
                'recovery_started':False,'run':str(self.source),'failed_native_calls':[call]}
        save_new(record['review_path'],report);record['review_sha256']=sha(record['review_path'])
        return record,report

    def test_scoped_snapshot_is_evidence_and_remains_unchanged(self):
        record,report=self.snapshot();before=sha(record['review_path'])
        revised.validate_nested_review(record)
        self.assertEqual(sha(record['review_path']),before)
        self.assertNotEqual(report['status'],'passed')

    def test_scoped_snapshot_rejects_errors_or_different_failure(self):
        record,report=self.snapshot()
        for field in ('binding_errors','status','request_hash','native_call_sha256','raw_path','context','schema','missing_consumer','consumer_error','consumer_raw'):
            candidate=copy.deepcopy(report)
            if field=='binding_errors':candidate[field]=['unresolved']
            elif field=='status':candidate[field]='passed'
            elif field=='schema':candidate[field]='unrecognized-report'
            elif field=='context':candidate['failed_native_calls'][0]['logical_events'][0]['context']['conversation']=9
            elif field=='missing_consumer':candidate['failed_native_calls'][0]['logical_events']=[]
            elif field=='consumer_error':candidate['failed_native_calls'][0]['logical_events'][0]['error_type']='Unknown'
            elif field=='consumer_raw':candidate['failed_native_calls'][0]['logical_events'][0]['cache_sha256']='different'
            else:candidate['failed_native_calls'][0][field]='different'
            from pathlib import Path
            Path(record['review_path']).write_text(json.dumps(candidate));record['review_sha256']=sha(record['review_path'])
            with self.subTest(field=field),self.assertRaises(ValueError):revised.validate_nested_review(record)

    def test_passed_status_without_known_terminal_review_schema_is_rejected(self):
        record,report=self.snapshot();report['status']='passed';report['schema']='unknown'
        from pathlib import Path
        Path(record['review_path']).write_text(json.dumps(report));record['review_sha256']=sha(record['review_path'])
        with self.assertRaises(ValueError):revised.validate_nested_review(record)
