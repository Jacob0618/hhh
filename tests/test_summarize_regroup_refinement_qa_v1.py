"""Offline replay and single-arm400 denominator regressions; no real API."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import summarize_regroup_refinement_qa_v1 as s
from tests import test_sdk20_plain_qa_v1 as fixture


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixture.PlainTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        p=patch.object(fixture.d,'ARMS',s.ARMS);p.start();self.addCleanup(p.stop)

    def synthetic_rows(self):
        return [{'query_id':f'c{i%10}_q{i}','conversation':i%10,'category':i%4+1,'method':'jev_refined',
            'selection_status':'complete','reader_status':'complete','judge_status':'complete','qa_status':'complete',
            'scores':{'official_locomo_f1':.5,'custom_token_f1':.4,'judge_correct':1.}} for i in range(400)]

    def phase(self, outputs):
        j=fixture.job(arm='jev_refined');run,m=self.fx.phase([j]);out,_=self.fx.execute(run,outputs)
        source=fixture.d.verify(m['source_manifest']['path'])
        return run,m,out,j,source

    def adapt_mock_phase(self, run, m, out, source, *, source_run=None):
        source_run=source_run or run
        source_path=source_run/'source_jobs.json'
        if not source_path.exists():s.save_new(source_path,{k:v for k,v in source.items() if k!='content_sha256'})
        bound_source=s.plain.verify(source_path)
        adapted={**m,'schema':s.driver_v2.SCHEMA,'arm':'jev_refined'}
        (run/'run_manifest.json').write_text(json.dumps(adapted),encoding='utf-8')
        return adapted,bound_source,SimpleNamespace(terminal=lambda folder:(adapted,out))

    def test_single_arm400_independent_denominators_and_no_fake_pair(self):
        rows=self.synthetic_rows();rows[0]['scores']['judge_correct']=None
        value=s.aggregate(rows,'jev_refined');self.assertIsNone(value['paired'])
        self.assertEqual(set(value['arms']),{'jev_refined'})
        metrics=value['arms']['jev_refined']['metrics']
        self.assertEqual(metrics['official_locomo_f1']['valid_denominator'],400)
        self.assertEqual(metrics['judge_correct']['valid_denominator'],399)
        self.assertEqual(metrics['judge_correct']['planned_denominator'],400)
        self.assertEqual(metrics['judge_correct']['zero_imputed_sensitivity']['imputed_slots'],1)

    def test_missing_duplicate_or_second_arm_denominator_rejected(self):
        rows=self.synthetic_rows()
        for v in (rows[:-1],rows[:-1]+[rows[0]],rows+[dict(rows[0],method='source_local_context')]):
            with self.assertRaises(ValueError):s.aggregate(v,'jev_refined')

    def test_terminal_guard_never_seals_or_dispatches(self):
        run=self.fx.root/'closed';run.mkdir()
        facade=s.RunnerFacade(SimpleNamespace(terminal=lambda *_:self.fail('driver may not run')))
        with self.assertRaisesRegex(s.InvalidArtifact,'terminal QA'):facade.terminal(run)
        (run/'run_manifest.json').write_text(json.dumps({'phase_kind':'firstpass'}))
        s.save_new(run/'run_outcome.json',{'terminal':True})
        with self.assertRaisesRegex(s.InvalidArtifact,'sealed source_jobs'):facade.terminal(run)
        self.assertFalse((run/'source_jobs.json').exists())

    def test_actual_raw_reader_and_compat_replayed_without_scoring_changes(self):
        run,m,out,j,source=self.phase(['Paris','Explanation\n[[yes]]'])
        m,source,d=self.adapt_mock_phase(run,m,out,source)
        original=s.base.audit_phase;original_runner=original.__globals__['runner']
        with patch.object(s,'driver_for',return_value=d):result=s.audit_phase(run,source)
        self.assertEqual(result['errors'],[])
        strict=s.base.record_rows([j],result)[0];compat=s.base.record_rows([j],result,compatible=True)[0]
        self.assertEqual(strict['scores']['official_locomo_f1'],1.)
        self.assertIsNone(strict['scores']['judge_correct']);self.assertEqual(compat['scores']['judge_correct'],1.)
        self.assertIs(original.__globals__['runner'],original_runner)
        self.assertIs(s.base.audit_phase,original)

    def test_actual_recovery_preserves_reader_and_uses_retry_strict_no(self):
        run,m,out,j,source=self.phase(['Paris','Explanation\n[[yes]]'])
        fixture.d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1')
        retry_out,_=self.fx.execute(retry,['[[no]]']);retry_m=fixture.d.manifest(retry)
        m,source,first_driver=self.adapt_mock_phase(run,m,out,source)
        retry_m,_,retry_driver=self.adapt_mock_phase(retry,retry_m,retry_out,source,source_run=run)
        with patch.object(s,'driver_for',side_effect=lambda p:first_driver if Path(p)==run else retry_driver):
            first=s.audit_phase(run,source);second=s.audit_phase(retry,source,parent=first)
        self.assertEqual(first['errors'],[]);self.assertEqual(second['errors'],[])
        row=s.base.record_rows([j],first,second,compatible=True,final=True)[0]
        self.assertEqual(row['scores']['official_locomo_f1'],1.)
        self.assertEqual(row['scores']['judge_correct'],0.)
        self.assertEqual(second['usage']['physical_attempts'],1)

    def test_shared_ledger_filter_retains_original_hash_and_rejects_unknown_phase(self):
        run,m,out,j,source=self.phase(['Paris','[[yes]]'])
        p=run/'conv_0/api/api_calls.jsonl';raw=p.read_bytes();first=json.loads(raw.splitlines()[0])
        retrieval={**first,'context':{**first['context'],'phase':'retrieval'}}
        p.write_bytes(raw+json.dumps(retrieval).encode()+b'\n')
        snapshot=s.Snapshot();view=s.PhaseLedgerView(snapshot,'qa')
        self.assertEqual(view.raw(p).splitlines(),raw.splitlines())
        self.assertEqual(view.sha(p),s.plain.sha(p))
        bad={**retrieval,'context':{**retrieval['context'],'phase':'unrecognized'}}
        p.write_bytes(raw+json.dumps(bad).encode()+b'\n')
        with self.assertRaisesRegex(s.InvalidArtifact,'Unexpected phase'):
            s.PhaseLedgerView(s.Snapshot(),'qa').raw(p)

    def test_wrong_QA_model_is_not_hidden_by_phase_filter(self):
        run,m,out,j,source=self.phase(['Paris','[[yes]]'])
        _,source,d=self.adapt_mock_phase(run,m,out,source)
        p=run/'conv_0/api/api_calls.jsonl';p.write_text(p.read_text().replace('gemini-2.5-flash','wrong-model'))
        with patch.object(s,'driver_for',return_value=d):
            with self.assertRaises(s.InvalidArtifact):s.audit_phase(run,source)

    def test_public_summary_requires_full400_before_raw_replay(self):
        run,m,out,j,source=self.phase(['Paris','[[yes]]'])
        self.adapt_mock_phase(run,m,out,source)
        with patch.object(s,'audit_phase',side_effect=AssertionError('No partial summary replay')):
            with self.assertRaisesRegex(s.InvalidArtifact,'original400'):s.summarize(run)

    def test_retrieval_and_QA_ledger_accounting_are_separate(self):
        run,m,out,j,source=self.phase(['Paris','[[yes]]'])
        m,source,d=self.adapt_mock_phase(run,m,out,source)
        p=run/'conv_0/api/api_calls.jsonl';raw=p.read_bytes()
        row={'context':{'conversation':0,'architecture':s.driver_v2.SCHEMA,'phase':'retrieval',
                       'query_id':j['query_id'],'method':'jev_refined'},'provider':'gemini',
             'model':'gemini-embedding-001','operation':'batchEmbedContents','stage':'mem0_embedding_search',
             'request_hash':'embedding-fixture','attempt':1,'status':200,'usage':{}}
        p.write_bytes(raw+json.dumps(row).encode()+b'\n')
        with patch.object(s,'driver_for',return_value=d):phase=s.audit_phase(run,source)
        self.assertEqual(phase['usage']['physical_attempts'],2)
        def lineage(snapshot,run,m,rows,refs,missing):
            return {'scope_complete':not missing,'new_namespace_only':s.measure_usage(rows)}
        with patch.object(s.base.existing,'usage_and_lineage',side_effect=lineage):
            out=s.audit_retrieval_and_usage(phase,source)
        self.assertEqual(out['qa_usage']['physical_attempts'],2)
        self.assertEqual(out['retrieval_usage']['physical_attempts'],1)
        self.assertEqual(out['all_phase_usage']['physical_attempts'],3)
        row['model']='unexpected'
        p.write_bytes(raw+json.dumps(row).encode()+b'\n')
        with self.assertRaises(s.InvalidArtifact):s.audit_retrieval_and_usage(phase,source)

    def test_full400_summary_shell_single_arm_with_no_recovery(self):
        run=self.fx.root/'summary_only';run.mkdir()
        jobs=[fixture.job(f'c{i%10}_q{i}',arm='jev_refined',ci=i%10) for i in range(400)]
        for i,j in enumerate(jobs):j['category']=i%4+1
        source_ref=s.save_new(run/'source_jobs.json',{'jobs':jobs})
        gate={'path':'mock-gate','sha256':'not-read-without-writer-accounting'}
        m={'phase_kind':'firstpass','arm':'jev_refined','source_manifest':source_ref,'gate':gate}
        (run/'run_manifest.json').write_text(json.dumps(m));s.save_new(run/'run_outcome.json',{'counts':{}})
        records={j['job_id']:{'reader':{'status':'complete','result':{'metrics':{'official_locomo_f1':.5,'custom_token_f1':.4},
                     'prediction':{'answer':'Paris'}}},'judge':{'status':'complete','result':{'correct':True,'label':'[[yes]]'}},
                     'compat':None,'qa':None,'errors':[]} for j in jobs}
        phase={'run':str(run),'manifest':m,'outcome':{'counts':{}},'records':records,'errors':[],
               'ledger':[],'ledger_refs':[],'usage':s.measure_usage([]),'inputs':{}}
        account={'retrieval_records':[],'combined_lineage':{'scope_complete':True,'new_namespace_only':s.measure_usage([])}}
        with patch.object(s,'audit_phase',return_value=phase),patch.object(s,'audit_retrieval_and_usage',return_value=account):
            report=s.summarize(run)
        self.assertEqual(report['planned_total'],400);self.assertFalse(report['recovery_supplied'])
        self.assertIsNone(report['paired_comparison'])
        for view in ('strict_firstpass','compatible_firstpass','final'):
            self.assertEqual(set(report['views'][view]['arms']),{'jev_refined'})
            self.assertEqual(report['views'][view]['arms']['jev_refined']['metrics']['official_locomo_f1']['complete_only_mean'],.5)
        self.assertIsNone(report['writer_accounting']['usage'])

    def test_recovery_of_recovery_blocked_before_driver(self):
        first=self.fx.root/'fake_recovery';first.mkdir();second=first.with_name(first.name+'_recovery_v1');second.mkdir()
        (first/'run_manifest.json').write_text(json.dumps({'phase_kind':'technical_recovery'}))
        (second/'run_manifest.json').write_text(json.dumps({'phase_kind':'technical_recovery','parent':{'run_dir':str(first)}}))
        s.save_new(second/'run_outcome.json',{'terminal':True})
        with self.assertRaisesRegex(s.InvalidArtifact,'Recovery of recovery'):s.require_closed_files(second)


if __name__=='__main__':unittest.main()
