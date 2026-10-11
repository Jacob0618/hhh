"""Offline Judge-only orchestration tests. No credentials or remote transport."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import requests

from tools import run_sdk20_judge_retest_v1 as d
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.answer_format_compat import normalize_reader_prediction

KEYS={'GEMINI_API_KEY':'offline-secret-gemini','TYPESAFE_API_KEY':'offline-secret-jev'}


def job(qid='c0_q0',arm='baseline_sdk20',ci=0,answer='  exact answer\n'):
    return {'job_id':arm+'/'+qid,'conversation':ci,'method':arm,'query_id':qid,
        'question':'Question '+qid+'?', 'reference':'Reference','status':'ready','reader_status':'complete',
        'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),
        'source':{'original_marker':qid},'F1':{'official_locomo_f1':.5},'old_judge':{'correct':False},
        'main_cohort_member':True,'pending_original_list':None,'deferred_reader_resolution':None}


def blocked(qid='c0_q9'):
    value=job(qid);original=[' two ', 'one','one']
    normalized,proof=normalize_reader_prediction({'answer':original})
    value.update(status='blocked_original_reader',reader_status='failed',answer=None,answer_sha256=None,
        pending_original_list=original,F1={'official_locomo_f1':None},old_judge=None,
        deferred_reader_resolution={'status':'deferred_postbatch','reader_resolution_authorized':True,
            'source_choice':'original_firstpass_raw_only_not_old_recovery','normalized_answer':normalized['answer'],
            'answer_sha256':hashlib.sha256(normalized['answer'].encode()).hexdigest(),'normalization_proof':proof,
            'F1':{'official_locomo_f1':.75},'reader_model_calls':0,'main_firstpass_metrics_remain_null':True})
    return value


def response(text):return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}]}


class HTTP:
    status_code=200;ok=True;headers={}
    def __init__(self,data):self.data=data
    def json(self):return copy.deepcopy(self.data)


class Transport:
    def __init__(self,outputs):self.outputs=list(outputs);self.calls=[];self.lock=threading.Lock()
    def post(self,url,**kwargs):
        with self.lock:
            self.calls.append(copy.deepcopy(kwargs['json']))
            if not self.outputs:raise AssertionError('Unexpected extra physical request')
            value=self.outputs.pop(0)
        if isinstance(value,Exception):raise value
        return HTTP(response(value) if isinstance(value,str) else value)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='judge_only_offline_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'runs').mkdir()
        for target in ('requests.Session.post','requests.Session.request','socket.create_connection','jev_mem.providers.load_credentials'):
            p=patch(target,side_effect=AssertionError('No real transport or credentials'));p.start();self.addCleanup(p.stop)
        for name,value in (('ROOT',self.root),('CODE',())):
            p=patch.object(d,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(d,'source_validate',side_effect=lambda p:d.verify(p));p.start();self.addCleanup(p.stop)

    def phase(self,jobs,supplement=None,name='main'):
        source={'schema':d.SOURCE_SCHEMA,'phase_kind':'firstpass','phase_id':'mock_firstpass','jobs':jobs,'supplemental_jobs':supplement or []}
        sourcepath=self.root/(name+'_source.json');d.save_new(sourcepath,source)
        run=self.root/'runs'/name
        d._new_manifest(run,sourcepath,source,phase_kind='firstpass',phase_id=source['phase_id'],jobs=jobs)
        return run,d.manifest(run)

    def factory(self,path,ci,m):
        return PlainTextEvaluationAPI(path,keys=KEYS,cache_roots=(),retries=2,
            context={'architecture':d.SCHEMA,'conversation':ci,'phase':'judge_retest','phase_id':m['phase_id']})

    def execute(self,run,outputs):
        transport=Transport(outputs)
        with patch('requests.Session',return_value=transport),patch('jev_mem.providers.time.sleep'):
            result=d.run(run,api_factory=self.factory)
        return result,transport

    def artifact(self,run,j,stage):return d.verify(d.result_paths(run,j)[stage])

    def test_exact_prompt_body_and_no_json_or_reader(self):
        j=job();run,m=self.phase([j]);out,transport=self.execute(run,[' [[yes]]\n'])
        self.assertEqual(transport.calls,[d.signature(j)['body']])
        self.assertEqual(transport.calls[0]['generationConfig'],{'temperature':0,'maxOutputTokens':2048})
        self.assertIn(j['answer'],transport.calls[0]['contents'][0]['parts'][0]['text'])
        strict=self.artifact(run,j,'strict');compat=self.artifact(run,j,'compat')
        self.assertTrue(strict['result']['correct']);self.assertEqual(strict['result']['raw'],' [[yes]]\n')
        self.assertEqual(compat['status'],'strict_valid');self.assertEqual(out['reader_calls'],0)
        self.assertEqual(compat['raw_evidence']['physical_producer']['stage'],d.JUDGE_STAGE)
        self.assertFalse((run/'conv_0/api/embedding_pacing_events.jsonl').exists())

    def test_blocked_never_constructs_api(self):
        j=blocked();run,m=self.phase([j])
        out=d.run(run,api_factory=lambda *args:(_ for _ in ()).throw(AssertionError('No API for blocked')))
        self.assertEqual(out['counts']['strict'],{'blocked_original_reader':1})
        self.assertFalse((run/'conv_0').exists());self.assertIsNone(self.artifact(run,j,'strict')['result'])

    def test_firstpass_prepare_full_population_is_offline(self):
        jobs=[job(f'c0_q{i}',arm) for arm in ('baseline_sdk20','fact_types_sdk20') for i in range(1540)]
        source={'schema':d.SOURCE_SCHEMA,'phase_kind':'firstpass','phase_id':'p','jobs':jobs,'supplemental_jobs':[]}
        p=self.root/'source.json';d.save_new(p,source)
        with patch.object(d,'PlainTextEvaluationAPI',side_effect=AssertionError('No API before run')):
            result=d.prepare(self.root/'runs'/'prepared',p);self.assertEqual(result['jobs'],3080)
            self.assertEqual(len(d.manifest(self.root/'runs'/'prepared')['jobs']),3080)

    def test_strict_failure_and_compat_are_separate_and_immutable(self):
        j=job();run,m=self.phase([j]);self.execute(run,['Explanation\n[[no]]'])
        strict=self.artifact(run,j,'strict');compat=self.artifact(run,j,'compat')
        self.assertEqual(strict['status'],'failed');self.assertEqual(strict['error']['type'],'InvalidJudgeLabel')
        self.assertEqual(strict['raw_invalid_label'],'Explanation\n[[no]]')
        self.assertEqual(compat['status'],'compat_valid');self.assertFalse(compat['interpretation']['correct'])
        before=d.pointer(d.result_paths(run,j)['strict'])
        with self.assertRaisesRegex(RuntimeError,'no implicit replay'):d.one(run,j,m)
        self.assertEqual(before,d.pointer(d.result_paths(run,j)['strict']))

    def test_conflict_and_multipart_are_not_compat_eligible(self):
        for n,raw in enumerate((response('[[yes]] in explanation\n[[no]]'),
                {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'Explanation\n'},{'text':'[[yes]]'}]}}]},
                {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'hidden','thought':True},{'text':'Explanation\n[[yes]]'}]}}]})):
            j=job();run,m=self.phase([j],name=f'case{n}');self.execute(run,[raw])
            self.assertEqual(self.artifact(run,j,'strict')['status'],'failed')
            self.assertEqual(self.artifact(run,j,'compat')['status'],'rejected')

    def test_cache_is_same_phase_conv_exact_request_only(self):
        a=job();b=job(arm='fact_types_sdk20');run,m=self.phase([a,b])
        out,t=self.execute(run,['[[yes]]'])
        self.assertEqual(len(t.calls),1)
        self.assertEqual(out['counts']['strict'],{'complete':2})
        events=[e for j in (a,b) for e in self.artifact(run,j,'strict')['api_events']]
        self.assertEqual(sorted(e['source'] for e in events),['existing_run_cache','new_request'])
        self.assertTrue(all(self.artifact(run,j,'compat')['status']=='strict_valid' for j in (a,b)))

    def test_transport_retry_is_bounded_and_error_is_redacted(self):
        j=job();run,m=self.phase([j]);out,t=self.execute(run,[requests.exceptions.SSLError(KEYS['GEMINI_API_KEY'])]*3)
        self.assertEqual(len(t.calls),3);s=self.artifact(run,j,'strict');self.assertEqual(s['status'],'failed')
        self.assertNotIn(KEYS['GEMINI_API_KEY'],json.dumps(s));self.assertIn('<redacted>',s['error']['message'])
        self.assertEqual(self.artifact(run,j,'compat')['status'],'unavailable')
        self.assertEqual(len(d.run(run,api_factory=lambda *a:None)['results']),1)

    def test_strict_saved_before_compat_local_error(self):
        j=job();run,m=self.phase([j])
        with patch.object(d,'response_evidence',side_effect=ValueError('local evidence error')):
            self.execute(run,['[[no]]'])
        self.assertEqual(self.artifact(run,j,'strict')['result']['correct'],False)
        self.assertEqual(self.artifact(run,j,'compat')['status'],'unavailable')

    def test_source_or_manifest_change_is_refused_before_api(self):
        j=job();run,m=self.phase([j]);source=Path(m['source_manifest']['path'])
        value=d.verify(source);value['jobs'][0]['answer']='changed';source.write_text(json.dumps(value))
        with self.assertRaises(ValueError):d.run(run,api_factory=lambda *a:None)
        self.assertFalse((run/'execution_intent.json').exists())

    def test_missing_vault_refuses_before_execution_intent(self):
        run,m=self.phase([job()])
        with patch.object(d,'VAULT',self.root/'absent'):
            with self.assertRaisesRegex(RuntimeError,'DPAPI'):d.run(run)
        self.assertFalse((run/'execution_intent.json').exists())

    def test_unknown_execution_is_not_implicitly_restarted(self):
        run,m=self.phase([job()]);d.durable(run/'execution_intent.json',{'unknown':True})
        with self.assertRaisesRegex(RuntimeError,'failed/unknown'):d.run(run,api_factory=lambda *a:None)

    def test_terminal_wrong_slot_pointer_is_rejected(self):
        a=job();b=job('c0_q1');run,m=self.phase([a,b]);self.execute(run,['[[yes]]','[[no]]'])
        path=run/'run_outcome.json';out=d.verify(path);out.pop('content_sha256');out['results'][0]['strict']=out['results'][1]['strict']
        path.unlink();d.save_new(path,out)
        with self.assertRaisesRegex(ValueError,'another slot'):d.terminal(run)

    def test_postbatch_population_all_strict_failures_and_all_fixed_lists(self):
        a=job();b=job('c0_q1');c=blocked();extra=job('c2_q74','dated_unmerged385',2)
        extra.update(job_id='global_conflict/dated_unmerged385/c2_q74',status='deferred_postbatch',main_cohort_member=False)
        run,m=self.phase([a,b,c],[extra])
        with self.assertRaises(FileNotFoundError):d.prepare_recovery(run)
        # Per-job writes make output allocation deterministic, unlike concurrent scheduling.
        transport=Transport(['[[no]]','Explanation\n[[yes]]'])
        with patch('requests.Session',return_value=transport):
            api=self.factory(run/'conv_0/api',0,m)
            d.one(run,a,m,api);d.one(run,b,m,api);d.one(run,c,m,api)
        self.seal_existing(run,m)
        source,jobs,parent,resolutions=d.recovery_jobs(run)
        self.assertEqual({j['job_id'] for j in jobs},{b['job_id'],c['job_id'],extra['job_id']})
        deferred=next(j for j in jobs if j['job_id']==c['job_id'])
        self.assertEqual(deferred['answer'],' two \none\none');self.assertEqual(deferred['F1'],{'official_locomo_f1':.75})
        self.assertEqual(deferred['source'],c['source']);self.assertEqual(c['F1'],{'official_locomo_f1':None})
        self.assertEqual(deferred['attempt_kind'],'first_judge_after_explicit_reader_resolution')
        d.prepare_recovery(run)
        retry=run.with_name(run.name+'_recovery_v1');self.assertEqual(len(d.manifest(retry)['jobs']),3)
        with self.assertRaises(FileExistsError):d.prepare_recovery(run)
        _,t=self.execute(retry,['[[yes]]']*3);self.assertEqual(len(t.calls),3)
        with self.assertRaisesRegex(ValueError,'cannot itself'):d.prepare_recovery(retry)

    def seal_existing(self,run,m):
        rows=[]
        for j in m['jobs']:
            row={k:j[k] for k in ('job_id','conversation','method','query_id')}
            row.update({stage:d.pointer(p) if p.exists() else None for stage,p in d.result_paths(run,j).items()});rows.append(row)
        d.save_new(run/'run_outcome.json',{'terminal':True,'protocol_sha256':m['protocol_sha256'],'total':len(rows),'results':rows})

    def test_deferred_normalization_rejects_changed_answer_or_proof(self):
        for field,value in (('normalized_answer','different'),('normalization_proof',{}),('reader_resolution_authorized',False)):
            original=blocked();original['deferred_reader_resolution'][field]=value
            with self.assertRaises(ValueError):d.deferred_job(original)

    def test_followup_only_deferred_first_judge_failures_and_once(self):
        main=job();deferred=blocked();run,m=self.phase([main,deferred]);self.execute(run,['Explanation\n[[yes]]'])
        d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1')
        self.execute(retry,['Explanation\n[[no]]','Explanation\n[[yes]]'])
        source,jobs,parent=d.supplement_followup_jobs(retry)
        self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['job_id'],deferred['job_id'])
        self.assertEqual(jobs[0]['attempt_kind'],'deferred_reader_second_and_final_judge')
        d.prepare_supplement_followup(retry);last=retry.with_name(retry.name+'_supplement_followup_v1')
        self.assertEqual(d.manifest(last)['jobs'],jobs);self.execute(last,['Still bad\n[[no]]'])
        with self.assertRaisesRegex(ValueError,'Only the first'):d.prepare_supplement_followup(last)

    def test_unknown_firstpass_slot_is_selected_once(self):
        j=job();run,m=self.phase([j]);self.seal_existing(run,m)
        source,jobs,parent,resolutions=d.recovery_jobs(run)
        self.assertEqual(len(jobs),1);self.assertTrue(jobs[0]['parent_result']['unknown'])

    def test_resolution_uses_strict_first_even_when_labels_disagree(self):
        fs={'status':'failed'};fc={'status':'compat_valid','interpretation':{'correct':True,'label':'[[yes]]'}}
        rs={'status':'complete','result':{'correct':False,'label':'[[no]]'}}
        self.assertEqual(d.resolve_pair(fs,fc,rs,None),{'source':'retry_strict','correct':False,'label':'[[no]]'})
        self.assertEqual(d.resolve_pair(rs,fc,{'status':'complete','result':{'correct':True,'label':'[[yes]]'}},None)['source'],'firstpass_strict')
        self.assertEqual(d.resolve_pair(fs,fc)['source'],'firstpass_eligible_compat')
        self.assertIsNone(d.resolve_pair(fs,{'status':'rejected'})['correct'])

    def test_threads_bounded_four_per_conv_and_eight_convs(self):
        jobs=[job(f'c{ci}_q{k}',ci=ci) for ci in range(10) for k in range(6)]
        run,m=self.phase(jobs);lock=threading.Lock();active={};peaks={};total=peak=0
        def fake_one(run,j,m,api,initial_error):
            nonlocal total,peak
            ci=j['conversation']
            with lock:
                active[ci]=active.get(ci,0)+1;peaks[ci]=max(peaks.get(ci,0),active[ci]);total+=1;peak=max(peak,total)
            time.sleep(.02)
            with lock:active[ci]-=1;total-=1
        with patch.object(d,'one',side_effect=fake_one):d.run(run,api_factory=lambda *args:None)
        self.assertLessEqual(peak,32);self.assertTrue(all(v<=4 for v in peaks.values()))


if __name__=='__main__':unittest.main()
