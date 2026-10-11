"""No credentials/network: population, old-response reuse and real mock HTTP."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_dated_judge_uniform_recovery_v1 as d
from tests.test_sdk20_judge_retest_v1 import Transport, KEYS, job as old_job
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI


def population():
    jobs=[];judges={}
    for arm in d.ARMS:
        for i in range(1540):
            qid='c2_q74' if arm==d.ARMS[0] and i==74 else f'c0_q{i}'
            j=old_job(qid,arm,ci=2 if qid=='c2_q74' else 0)
            j.update(category=1,reference_date='2023-07-01')
            jobs.append(j)
            fail=i<(110 if arm==d.ARMS[0] else 108)
            judges[d.key(j)]={'status':'failed' if fail else 'complete',
                'error':{'type':'InvalidJudgeLabel'} if fail else None,
                'result':None if fail else {'correct':bool(i%2)}}
    return jobs,judges


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dated_uniform_offline_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for target in ('requests.Session.post','requests.Session.request','socket.create_connection','jev_mem.providers.load_credentials'):
            p=patch(target,side_effect=AssertionError('No real HTTP or credentials'));p.start();self.addCleanup(p.stop)

    def case(self,text='Explained\n[[no]]'):
        run=self.root/'mock';run.mkdir()
        j=old_job(arm=d.ARMS[0]);j.update(category=1,reference_date='2023-07-01')
        source=self.root/'source.json';d.save_new(source,{'jobs':[j]})
        m={'schema':d.SCHEMA,'phase_kind':'uniform_dated_recovery','phase_id':d.SCHEMA,
           'protocol_sha256':'test-protocol','source_manifest':d.pointer(source),'jobs':[j]}
        api=PlainTextEvaluationAPI(run/'conv_0/api',keys=KEYS,cache_roots=(),retries=2,
            context={'architecture':d.SCHEMA,'conversation':0,'phase':'judge_retest','phase_id':d.SCHEMA})
        transport=Transport([text])
        with patch('requests.Session',return_value=transport):d.one(run,j,m,api)
        paths=d.legacy.result_paths(run,j)
        out={'results':[{'job_id':j['job_id'],**{k:d.pointer(p) for k,p in paths.items()}}]}
        return run,j,m,out,transport

    def test_population218_is_format_only_and_c2_excluded_from217(self):
        jobs,judges=population();fail=d.failure_population(jobs,judges)
        self.assertEqual(len(fail),218)
        self.assertEqual(len([j for j in fail if d.key(j)!=d.EXISTING_KEY]),217)
        before=[d.key(j) for j in fail]
        for r in judges.values():
            if r['result'] is not None:r['result']['correct']=not r['result']['correct']
        self.assertEqual([d.key(j) for j in d.failure_population(jobs,judges)],before)

    def test_unknown_failure_and_duplicate_slot_cannot_enter_recovery(self):
        jobs,judges=population();judges[d.key(jobs[0])]['error']['type']='Timeout'
        with self.assertRaisesRegex(ValueError,'unknown'):d.failure_population(jobs,judges)
        jobs[-1]=jobs[0]
        with self.assertRaisesRegex(ValueError,'3080'):d.failure_population(jobs,judges)

    def test_no_Reader_and_exact_unchanged_Judge_body_real_mock_transport(self):
        run,j,m,out,transport=self.case(' [[yes]]\n')
        self.assertEqual(transport.calls,[d.legacy.signature(j)['body']])
        self.assertFalse(any(run.rglob('reader')))
        self.assertEqual(d.verify(out['results'][0]['strict']['path'])['result']['label'],'[[yes]]')
        self.assertEqual(d.verify(out['results'][0]['compat']['path'])['status'],'strict_valid')
        self.assertIs(d.one.__code__,d.legacy.one.__code__)
        self.assertIs(d._LEGACY['run'].__code__,d.legacy.run.__code__)
        self.assertEqual(d.legacy.SCHEMA,'sdk20-plain-judge-retest-run-v2')

    def test_failed_format_raw_and_NO_compat_replay(self):
        run,j,m,out,_=self.case()
        with patch.object(d._SummaryRunner,'terminal',staticmethod(lambda p:(m,out))):result=d.audit_phase(run)
        record=result['records'][j['job_id']]
        self.assertEqual(record['strict']['status'],'failed')
        self.assertEqual(record['strict']['raw_invalid_label'],'Explained\n[[no]]')
        self.assertIs(record['compat']['interpretation']['correct'],False)
        self.assertEqual(result['usage']['physical_attempts'],1)

    def test_same_job_intent_never_implicitly_resends(self):
        run,j,m,out,_=self.case()
        with self.assertRaisesRegex(RuntimeError,'implicit replay'):d.one(run,j,m,None)

    def test_opposite_label_is_not_compatible(self):
        run,j,m,out,_=self.case('Maybe [[yes]]\n[[no]]')
        with patch.object(d._SummaryRunner,'terminal',staticmethod(lambda p:(m,out))):r=d.audit_phase(run)
        self.assertEqual(r['records'][j['job_id']]['compat']['status'],'rejected')
        self.assertIsNone(d.legacy.resolve_pair(None,None,r['records'][j['job_id']]['strict'],r['records'][j['job_id']]['compat'])['correct'])

    def test_extra_context_ledger_cannot_hide(self):
        run,j,m,out,_=self.case()
        p=run/'conv_0/api/api_calls.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
        extra=copy.deepcopy(rows[0]);extra['context']['job_id']='unknown/outside'
        with p.open('a') as f:f.write(json.dumps(extra)+'\n')
        with patch.object(d._SummaryRunner,'terminal',staticmethod(lambda p:(m,out))):
            with self.assertRaises(ValueError):d.audit_phase(run)

    def test_extension_gate_requires_two_closed_correct_schema_arms(self):
        from tools import run_regroup_refinement_extension1140_v1 as ext
        folders=[];values={}
        for arm in ('source_local_context','jev_refined'):
            folder=self.root/arm;folder.mkdir();folders.append(folder)
            m={'schema':ext.SCHEMA,'arm':arm,'phase_kind':'firstpass','protocol_sha256':'p'}
            (folder/'run_manifest.json').write_text(json.dumps(m));d.save_new(folder/'source_jobs.json',{})
            out={'terminal':True,'protocol_sha256':'p','total':1140};d.save_new(folder/'run_outcome.json',out)
            values[folder]=(m,out)
        with patch.object(ext,'terminal',side_effect=lambda p:values[p]):d.require_extensions_terminal(folders)
        values[folders[0]][1]['terminal']=False
        with patch.object(ext,'terminal',side_effect=lambda p:values[p]):
            with self.assertRaisesRegex(ValueError,'terminal1140'):d.require_extensions_terminal(folders)
        with self.assertRaisesRegex(ValueError,'distinct'): 
            with patch.object(ext,'terminal',side_effect=lambda p:values[p]):
                values[folders[0]][1]['terminal']=True;d.require_extensions_terminal([folders[0],folders[0]])

    def test_final_policy_preserves_strict_NO_and_uses_retry_strict_before_compat(self):
        yes={'status':'complete','result':{'correct':True,'label':'[[yes]]'}}
        no={'status':'complete','result':{'correct':False,'label':'[[no]]'}}
        cyes={'status':'compat_valid','interpretation':{'correct':True,'label':'[[yes]]'}}
        cno={'status':'compat_valid','interpretation':{'correct':False,'label':'[[no]]'}}
        self.assertEqual(d.legacy.resolve_pair(no,cno,yes,cyes)['source'],'firstpass_strict')
        self.assertIs(d.legacy.resolve_pair(None,cyes,no,cno)['correct'],False)
        self.assertEqual(d.legacy.resolve_pair(None,cyes,None,cno)['source'],'firstpass_eligible_compat')

    def test_prepare_never_overwrites_namespace(self):
        folder=self.root/'existing';folder.mkdir()
        with patch.object(d,'RUN',folder):
            with self.assertRaises(FileExistsError):d.prepare(folder)

    def test_offline_prepare_and_manifest_binding_preserve_exact217_jobs(self):
        jobs,judges=population()
        failed=d.failure_population(jobs,judges)
        source={'jobs':[j for j in failed if d.key(j)!=d.EXISTING_KEY],
                'official_metric_provenance':{'mock':'pinned'}}
        folder=self.root/'one_recovery';frozen=self.root/'lock.json';d.save_new(frozen,{'frozen':True})
        with patch.object(d,'RUN',folder),patch.object(d,'ROOT',self.root),patch.object(d,'CODE',()),\
             patch.object(d.lock,'LOCK',frozen),patch.object(d.lock,'verify_lock',return_value={}),\
             patch.object(d,'build_source',return_value=copy.deepcopy(source)):
            report=d.prepare(folder);self.assertEqual(report['new_Judge_jobs'],217)
            manifest=d.manifest(folder)
            self.assertEqual(manifest['jobs'],source['jobs'])
            self.assertNotIn(d.EXISTING_KEY,{d.key(j) for j in manifest['jobs']})
            self.assertFalse((folder/'execution_intent.json').exists())
            self.assertEqual(list(folder.iterdir()).__len__(),2)
            with self.assertRaises(FileExistsError):d.prepare(folder)


if __name__=='__main__':unittest.main()
