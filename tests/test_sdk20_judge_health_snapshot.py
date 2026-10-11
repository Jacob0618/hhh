"""Read-only synthetic health snapshots, no runner imports or network."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'research/snapshot_sdk20_judge_retest_v1.py'
spec=importlib.util.spec_from_file_location('judge_health',MODULE);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.jobs=[{'conversation':0,'method':'baseline_sdk20','query_id':'c0_q0','job_id':'baseline_sdk20/c0_q0','status':'ready'},
                   {'conversation':0,'method':'fact_types_sdk20','query_id':'c0_q0','job_id':'fact_types_sdk20/c0_q0','status':'blocked_original_reader'}]
        self.put('run_manifest.json',{'schema':'test','protocol_sha256':'mock','phase_id':'first','phase_kind':'firstpass','jobs':self.jobs})
        self.put('execution_intent.json',{'pid':42})
    def put(self,path,value):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value));return target
    def stage(self,stage,method,value,intent=False):
        return self.put(f'{stage}/conv_0/{method}/c0_q0'+('.intent' if intent else '')+'.json',value)
    def snapshot(self):return s.snapshot(self.root,process_reader=lambda:[{'ProcessId':42,'ParentProcessId':7,'CreationDate':'2026-09-29T00:00:00Z','CommandLine':'python fake'},{'ProcessId':999,'ParentProcessId':1,'CreationDate':'old','CommandLine':'python unrelated'}])

    def test_intents_not_counted_as_outputs_and_real_identity_visible(self):
        self.stage('strict','baseline_sdk20',{'pid':42,'created_utc':'2026-09-29T00:00:00+00:00'},True)
        snap=self.snapshot()
        self.assertEqual(snap['stage_counts']['strict'],{'missing':2})
        row=snap['conversations'][0];self.assertEqual(len(row['strict_or_compat_intents_without_output']),1)
        self.assertEqual([p['ProcessId'] for p in snap['processes']],[42])

    def test_complete_false_and_failed_label_are_distinct_outputs(self):
        self.stage('strict','baseline_sdk20',{'status':'complete','result':{'correct':False}})
        self.stage('compat','baseline_sdk20',{'status':'strict_valid'})
        self.stage('strict','fact_types_sdk20',{'status':'blocked_original_reader'})
        snap=self.snapshot();self.assertEqual(snap['stage_counts']['strict'],{'complete':1,'blocked_original_reader':1})
        self.assertEqual(snap['conversations'][0]['methods']['baseline_sdk20']['strict']['output_files'],1)

    def test_physical_cache_error_and_partial_tail_separately_counted(self):
        path=self.root/'conv_0/api/api_calls.jsonl';path.parent.mkdir(parents=True)
        rows=[{'request_hash':'a','status':200,'attempt':1,'timestamp_utc':'2026-09-29T00:00:01Z'},
              {'request_hash':'a','cache_hit':True,'timestamp_utc':'2026-09-29T00:00:02Z'},
              {'request_hash':'b','status':429,'attempt':1,'timestamp_utc':'2026-09-29T00:00:03Z'}]
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n{"partial":')
        value=s.ledger_summary(path)
        self.assertEqual((value['physical_attempts'],value['http_200'],value['physical_errors'],value['cache_events']),(2,1,1,1))
        self.assertEqual(value['malformed_line_numbers'],[4]);self.assertTrue(value['has_unterminated_tail'])

    def test_no_raw_intent_retains_terminal_consumer_status_not_assumed_running(self):
        self.stage('strict','baseline_sdk20',{'status':'failed','error':{'type':'RuntimeError'}})
        self.put('conv_0/api/request_intents/a.json',{'request_hash':'a','created_utc':'2026-09-29T00:00:00+00:00',
                 'context':{'conversation':0,'job_id':'baseline_sdk20/c0_q0'},'stage':'judge'})
        entry=self.snapshot()['conversations'][0]['request_intents_without_raw'][0]
        self.assertEqual(entry['strict_consumer_status'],'failed');self.assertNotEqual(entry['classification'],'inflight')

    def test_difference_is_evidence_without_automatic_health_verdict(self):
        before=self.snapshot();before['captured_utc']='2026-09-29T00:00:00+00:00'
        self.stage('strict','baseline_sdk20',{'status':'failed','error':{'type':'InvalidJudgeLabel'}})
        after=self.snapshot();after['captured_utc']='2026-09-29T00:00:30+00:00'
        diff=s.difference(before,after)
        self.assertEqual(diff['stage_count_deltas']['strict'],{'failed':1,'missing':-1})
        self.assertEqual(diff['conversations'][0]['methods']['baseline_sdk20']['strict']['output_files_added'],1)
        self.assertIsNone(diff['automatic_health_verdict'])
        bad=copy.deepcopy(after);bad['protocol_sha256']='other'
        with self.assertRaises(ValueError):s.difference(before,bad)


if __name__=='__main__':unittest.main()
