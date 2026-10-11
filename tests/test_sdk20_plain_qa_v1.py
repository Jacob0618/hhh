"""Offline integration: real plain API wrapper, dummy keys, local mock HTTP."""
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

from tools import run_sdk20_plain_qa_v1 as d
from tests.test_sdk20_judge_retest_v1 import Transport,HTTP,response,KEYS
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem import evaluation_plain_text_v1 as ev


def job(qid='c0_q0',arm='baseline_sdk20',ci=0):
    facts=[{'id':f'actual-{i}','text':f'Paris is mentioned in memory {i}.','observation_date':'2023-06-01'} for i in range(20)]
    return {'job_id':arm+'/'+qid,'conversation':ci,'method':arm,'query_id':qid,'status':'ready',
        'question':'Where?','reference':'Paris','category':1,'reference_date':'2023-06-30',
        'selected_ids':[f['id'] for f in facts],'selected_facts':facts,'lineage':{'original_marker':qid}}


class PlainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='sdk20_plain_offline_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'runs').mkdir()
        for target in ('requests.Session.post','requests.Session.request','socket.create_connection','jev_mem.providers.load_credentials'):
            p=patch(target,side_effect=AssertionError('No real network or credentials'));p.start();self.addCleanup(p.stop)
        for name,value in (('ROOT',self.root),('CODE',())):
            p=patch.object(d,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(d,'source_validate',side_effect=lambda p:d.verify(p));p.start();self.addCleanup(p.stop)

    def phase(self,jobs,name='main'):
        _,provenance=d.verified_official_metric()
        source={'schema':d.SOURCE_SCHEMA,'jobs':jobs,'official_metric_provenance':provenance}
        p=self.root/(name+'_source.json');d.save_new(p,source);run=self.root/'runs'/name
        d._new_manifest(run,p,source,jobs=jobs,phase_kind='firstpass')
        return run,d.manifest(run)

    def factory(self,path,ci,m):
        return PlainTextEvaluationAPI(path,keys=KEYS,cache_roots=(),retries=2,
            context={'architecture':d.SCHEMA,'conversation':ci,'phase':'qa','phase_id':m['phase_id']})

    def execute(self,run,outputs):
        t=Transport(outputs)
        with patch('requests.Session',return_value=t),patch('jev_mem.providers.time.sleep'):
            out=d.run(run,api_factory=self.factory)
        return out,t

    def artifact(self,run,j,s):return d.verify(d.stage_files(run,j)[s])

    def test_exact_plain_inputs_order_numeric_gold_and_reader_before_judge(self):
        j=job();j['reference']=2022;j['selected_facts'].reverse();j['selected_ids'].reverse()
        run,m=self.phase([j]);t=Transport(['2022','[[yes]]']);original=t.post
        def post(url,**kw):
            if '3.5-flash-lite' in url:
                r=self.artifact(run,j,'reader');self.assertEqual(r['status'],'complete')
                self.assertEqual(r['result']['metrics']['official_locomo_f1'],1.)
            return original(url,**kw)
        t.post=post
        with patch('requests.Session',return_value=t):out=d.run(run,api_factory=self.factory)
        payload=json.loads(t.calls[0]['contents'][0]['parts'][0]['text'])
        self.assertEqual(payload,ev.reader_input(j['question'],j['selected_facts'],reference_date=j['reference_date']))
        self.assertNotIn('reference',payload);self.assertEqual(payload['facts'][0]['id'],'f1')
        self.assertEqual(t.calls[0]['generationConfig'],{'temperature':0,'maxOutputTokens':2048,'thinkingConfig':{'thinkingBudget':0}})
        self.assertEqual(t.calls[1]['generationConfig'],{'temperature':0,'maxOutputTokens':2048})
        self.assertEqual(self.artifact(run,j,'compat')['status'],'strict_valid')
        self.assertEqual(out['counts']['qa'],{'complete':1})

    def test_json_looking_reader_is_unmodified_plain_string(self):
        j=job();run,m=self.phase([j]);answer=' {"answer":["Paris"]}\n'
        self.execute(run,[answer,'[[no]]']);r=self.artifact(run,j,'reader')
        self.assertEqual(r['result']['prediction']['answer'],answer)
        self.assertIsNone(r['result']['prediction']['used_fact_ids'])
        self.assertIs(self.artifact(run,j,'judge')['result']['correct'],False)

    def test_invalid_judge_preserves_f1_and_eligible_compat_separately(self):
        j=job();run,m=self.phase([j]);self.execute(run,['Paris','Explanation\n[[no]]'])
        self.assertEqual(self.artifact(run,j,'reader')['result']['metrics']['official_locomo_f1'],1.)
        self.assertEqual(self.artifact(run,j,'judge')['status'],'failed')
        self.assertEqual(self.artifact(run,j,'compat')['status'],'compat_valid')
        q=self.artifact(run,j,'qa');self.assertIsNone(q['judge_correct']);self.assertIs(q['resolved_judge']['correct'],False)

    def test_reader_failure_blocks_judge_and_redacts_bounded_transport(self):
        j=job();run,m=self.phase([j]);out,t=self.execute(run,[requests.exceptions.SSLError(KEYS['GEMINI_API_KEY'])]*3)
        self.assertEqual(len(t.calls),3);r=self.artifact(run,j,'reader');self.assertEqual(r['status'],'failed')
        self.assertNotIn(KEYS['GEMINI_API_KEY'],json.dumps(r))
        self.assertEqual(self.artifact(run,j,'judge')['status'],'not_run')
        self.assertIsNone(self.artifact(run,j,'qa')['reader_metrics'])
        with self.assertRaisesRegex(ValueError,'transport-only'):d.prepare_recovery(run)

    def test_api_initialization_failure_is_explicit_no_dispatch_and_not_auto_retried(self):
        j=job();run,m=self.phase([j])
        def fail(*args):raise ValueError(KEYS['GEMINI_API_KEY'])
        out=d.run(run,api_factory=fail)
        r=self.artifact(run,j,'reader');self.assertEqual(r['status'],'not_run')
        self.assertEqual(r['error']['type'],'APIInitializationFailure');self.assertNotIn(KEYS['GEMINI_API_KEY'],json.dumps(r))
        self.assertEqual(self.artifact(run,j,'judge')['status'],'not_run')
        self.assertFalse(d.stage_files(run,j)['reader'].with_suffix('.intent.json').exists())
        with self.assertRaises(ValueError):d.prepare_recovery(run)

    def test_same_phase_cross_arm_identical_request_cache_is_allowed(self):
        a=job();b=job(arm='fact_types_sdk20');run,m=self.phase([a,b]);out,t=self.execute(run,['Paris','[[yes]]'])
        self.assertEqual(len(t.calls),2);self.assertEqual(out['counts']['qa'],{'complete':2})
        for j in (a,b):self.assertEqual(self.artifact(run,j,'compat')['status'],'strict_valid')
        self.assertFalse((run/'conv_0/api/cache_lineage').exists())

    def test_new_reader_for_all_3080_including_old_failures_prepare_offline(self):
        jobs=[job(f'c0_q{i}',arm) for arm in d.ARMS for i in range(1540)]
        _,prov=d.verified_official_metric();s={'schema':d.SOURCE_SCHEMA,'jobs':jobs,'official_metric_provenance':prov}
        p=self.root/'source.json';d.save_new(p,s)
        with patch.object(d,'PlainTextEvaluationAPI',side_effect=AssertionError('No API during prepare')):
            result=d.prepare(self.root/'runs'/'prepared',p);self.assertEqual(result['jobs'],3080)
            self.assertEqual(len(d.manifest(self.root/'runs'/'prepared')['jobs']),3080)

    def test_job_contract_rejects_budget_order_types_and_old_blocked_status(self):
        variants=[]
        j=job();j['selected_ids'].reverse();variants.append(j)
        j=job();j['selected_facts']=j['selected_facts'][:19];variants.append(j)
        j=job();j['reference']=True;variants.append(j)
        j=job();j['status']='blocked_original_reader';variants.append(j)
        for j in variants:
            with self.assertRaises(ValueError):d.check_jobs([j])

    def test_vault_missing_and_prior_execution_are_refused_without_dispatch(self):
        run,m=self.phase([job()])
        with patch.object(d,'VAULT',self.root/'absent'):
            with self.assertRaisesRegex(RuntimeError,'DPAPI'):d.run(run)
        self.assertFalse((run/'execution_intent.json').exists())
        d.durable_intent(run/'execution_intent.json',{'unknown':True})
        with self.assertRaisesRegex(RuntimeError,'implicit restart'):d.run(run,api_factory=self.factory)

    def test_unresolved_stage_intent_never_resends_in_same_phase(self):
        j=job();run,m=self.phase([j]);p=d.stage_files(run,j)['reader'].with_suffix('.intent.json')
        d.durable_intent(p,{**d.identity(j,m),'stage':'reader'})
        out,t=self.execute(run,[]);self.assertEqual(t.calls,[])
        self.assertEqual(self.artifact(run,j,'reader')['error']['type'],'UnresolvedStageIntent')
        with self.assertRaisesRegex(ValueError,'Unresolved'):d.prepare_recovery(run)

    def test_recovery_waits_for_terminal_inherits_reader_retries_compat_failure(self):
        j=job();run,m=self.phase([j])
        with self.assertRaises(FileNotFoundError):d.prepare_recovery(run)
        self.execute(run,['Paris','Explanation\n[[yes]]'])
        before=d.pointer(d.stage_files(run,j)['reader']);d.prepare_recovery(run)
        retry=run.with_name(run.name+'_recovery_v1');rm=d.manifest(retry)
        self.assertEqual(rm['jobs'][0]['reader_source'],before)
        out,t=self.execute(retry,['[[no]]']);self.assertEqual(len(t.calls),1)
        rr=self.artifact(retry,rm['jobs'][0],'reader')
        self.assertEqual(rr['inherited_from'],before);self.assertEqual(rr['api_events'],[])
        self.assertFalse(d.stage_files(retry,j)['reader'].with_suffix('.intent.json').exists())
        self.assertEqual(rr['result'],self.artifact(run,j,'reader')['result']);self.assertEqual(before,d.pointer(d.stage_files(run,j)['reader']))
        resolved=self.artifact(retry,rm['jobs'][0],'qa')['resolved_judge']
        self.assertEqual(resolved,{'source':'retry_strict','correct':False,'label':'[[no]]'})
        with self.assertRaisesRegex(ValueError,'Only one'):d.prepare_recovery(retry)
        with self.assertRaises(FileExistsError):d.prepare_recovery(run)

    def test_valid_no_skipped_from_recovery_without_score_selection(self):
        j=job();run,m=self.phase([j]);self.execute(run,['Wrong','[[no]]'])
        pm,out=d.terminal(run);self.assertEqual(d.recovery_jobs(pm,out),[])

    def test_recovery_initialization_failure_still_retains_successful_reader(self):
        j=job();run,m=self.phase([j]);self.execute(run,['Paris','Explanation\n[[yes]]'])
        d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1');rm=d.manifest(retry)
        def fail(*args):raise ValueError('local initialization')
        d.run(retry,api_factory=fail)
        self.assertEqual(self.artifact(retry,rm['jobs'][0],'reader')['result'],self.artifact(run,j,'reader')['result'])
        self.assertEqual(self.artifact(retry,rm['jobs'][0],'judge')['status'],'not_run')
        self.assertEqual(self.artifact(retry,rm['jobs'][0],'qa')['reader_metrics']['official_locomo_f1'],1.)

    def test_reader_recovery_then_first_judge_stops_after_one_phase(self):
        j=job();run,m=self.phase([j]);self.execute(run,[{'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':'unfinished'}]}}]}])
        d.prepare_recovery(run);retry=run.with_name(run.name+'_recovery_v1');rm=d.manifest(retry)
        self.assertIsNone(rm['jobs'][0]['reader_source'])
        out,t=self.execute(retry,['Paris','Explanation\n[[yes]]']);self.assertEqual(len(t.calls),2)
        self.assertEqual(self.artifact(retry,rm['jobs'][0],'qa')['resolved_judge']['source'],'retry_eligible_compat')
        with self.assertRaises(ValueError):d.prepare_recovery(retry)

    def test_recovery_replays_exact_job_body_even_if_raw_and_intent_agree(self):
        j=job();run,m=self.phase([j]);self.execute(run,[{'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':'unfinished'}]}}]}])
        rp=d.stage_files(run,j)['reader'];r=d.verify(rp);event=r['api_events'][0]
        rawpath=Path(event['cache_path']);raw=d.read(rawpath);raw['request']['body']['contents'][0]['parts'][0]['text']='other request'
        rawpath.write_text(json.dumps(raw));event['cache_sha256']=d.sha(rawpath)
        ip=run/'conv_0/api/request_intents'/(event['request_hash']+'.json');intent=d.read(ip);intent['signature']=raw['request'];ip.write_text(json.dumps(intent))
        r.pop('content_sha256');rp.unlink();d.save_new(rp,r)
        with self.assertRaises(Exception):d.known_failure(j,m,d.verify(rp),d.pointer(rp),'reader')

    def test_terminal_wrong_pointer_and_source_change_are_refused(self):
        a=job();b=job('c0_q1',arm='fact_types_sdk20');run,m=self.phase([a,b]);self.execute(run,['Paris','[[yes]]'])
        p=run/'run_outcome.json';out=d.verify(p);out.pop('content_sha256');out['results'][0]['reader']=out['results'][1]['reader']
        p.unlink();d.save_new(p,out)
        with self.assertRaisesRegex(ValueError,'another slot'):d.terminal(run)
        sp=Path(m['source_manifest']['path']);s=d.verify(sp);s['jobs'][0]['reference']='changed';sp.write_text(json.dumps(s))
        with self.assertRaises(ValueError):d.manifest(run)

    def test_unknown_slot_retained_and_recovery_blocked_pending_audit(self):
        j=job();run,m=self.phase([j])
        with patch.object(d,'one',side_effect=OSError('simulated local write failure')):out=d.run(run,api_factory=lambda *a:None)
        self.assertEqual(out['counts']['reader'],{'unknown':1})
        with self.assertRaisesRegex(ValueError,'offline audit'):d.prepare_recovery(run)

    def test_compatibility_source_error_blocks_recovery(self):
        j=job();run,m=self.phase([j])
        with patch.object(d,'replay_source',side_effect=ValueError('raw mismatch')):self.execute(run,['Paris','Explanation\n[[yes]]'])
        self.assertEqual(self.artifact(run,j,'compat')['error']['type'],'ValueError')
        with self.assertRaisesRegex(ValueError,'Compatibility source'):d.prepare_recovery(run)

    def test_thread_limit_is_eight_conversations_by_four_combined_jobs(self):
        jobs=[job(f'c{ci}_q{k}',ci=ci,arm=d.ARMS[k%2]) for ci in range(10) for k in range(6)]
        run,m=self.phase(jobs);lock=threading.Lock();active={};peakconv={};total=peak=0
        def fake(run,j,m,api,official,initial_error):
            nonlocal total,peak
            with lock:
                ci=j['conversation'];active[ci]=active.get(ci,0)+1;peakconv[ci]=max(peakconv.get(ci,0),active[ci]);total+=1;peak=max(peak,total)
            time.sleep(.02)
            with lock:active[ci]-=1;total-=1
        with patch.object(d,'one',side_effect=fake):d.run(run,api_factory=lambda *a:None)
        self.assertLessEqual(peak,32);self.assertTrue(all(n<=4 for n in peakconv.values()))


if __name__=='__main__':unittest.main()
