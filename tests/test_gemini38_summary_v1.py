"""Offline raw replay, immutable Reader inheritance and denominator/accounting tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import summarize_gemini38_seven_arm_v1 as s
from tools import run_gemini38_plain_qa_v1 as d
from jev_mem.gemini38_provider_v1 import Gemini38API,check_request,MODEL
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem import evaluation_gemini38_v1 as e


class RecordedFake:
    generate=Gemini38API.generate
    def __init__(self,folder,label='The answer is equivalent.\n[[no]]',context=None):
        self.folder=Path(folder);self.label=label;self.context=context or{};self.events=[]
    def scoped(self,**context):return RecordedFake(self.folder,self.label,{**self.context,**context})
    def _redact(self,value):return str(value)
    def request(self,provider,model,body,*,stage):
        check_request(provider,model,body,'generateContent',stage)
        signature={'provider':provider,'model':model,'operation':'generateContent','body':body};key=request_digest(signature)
        ip=self.folder/'request_intents'/f'{key}.json';rp=self.folder/'api_cache'/f'{key}.json'
        ip.parent.mkdir(parents=True,exist_ok=True);rp.parent.mkdir(parents=True,exist_ok=True)
        cached=rp.exists();usage={'promptTokenCount':9,'candidatesTokenCount':3,'thoughtsTokenCount':2,'totalTokenCount':14}
        if not cached:
            ip.write_text(json.dumps({'signature':signature,'request_hash':key,'stage':stage,'retries':2,'context':self.context}))
            text='alpha'if stage==e.READER_STAGE else self.label
            response={'modelVersion':MODEL,'usageMetadata':usage,'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}]}
            rp.write_text(json.dumps({'request':signature,'response':response,'stage':stage}))
        else:response=json.loads(rp.read_text())['response']
        self.events.append({'request_hash':key,'stage':stage,'provider':provider,'model':model,'operation':'generateContent',
                           'source':'existing_run_cache'if cached else'new_request','context':self.context,'cache_path':str(rp),
                           'cache_sha256':d._base.sha(rp),'error_type':None})
        row={'timestamp_utc':'2026-10-03T00:00:00+00:00','request_hash':key,'stage':stage,'provider':provider,
             'model':model,'operation':'generateContent','context':self.context,'cache_hit':cached,'usage':response['usageMetadata'],'resolved_model':MODEL}
        row.update({'usage_semantics':'reused_response_no_new_model_usage'}if cached else{'attempt':1,'status':200})
        with(self.folder/'api_calls.jsonl').open('a')as h:h.write(json.dumps(row)+'\n')
        return response


def job(method='baseline'):
    facts=[{'id':str(i),'text':'alpha '+str(i),'observation_date':'2023-05-08'}for i in range(20)]
    return {'conversation':0,'method':method,'query_id':'c0_q0','job_id':method+'/c0_q0','status':'ready',
            'question':'Which name?','reference':'alpha','category':4,'reference_date':'2023-05-08',
            'selected_ids':[f['id']for f in facts],'selected_facts':facts,'bank':{'path':'newbank','sha256':'new'},
            'selection':{'path':'newselection','sha256':'new'}}


def phase(run,jobs,kind='firstpass',label='The answer is equivalent.\n[[no]]'):
    m={'phase_kind':kind,'phase_id':'offline_mock_'+kind,'protocol_sha256':'offline_mock',
       'source_manifest':{'path':'mock_fresh_source','sha256':'mock'},'jobs':jobs,'official_metric_provenance':d._base.verified_official_metric()[1]}
    api=RecordedFake(run/'conv_0/api',label);official,_=d._base.verified_official_metric()
    for j in jobs:d.one(run,j,m,api,official)
    out={'counts':{},'results':[]};(run/'run_manifest.json').write_text(json.dumps(m));(run/'run_outcome.json').write_text(json.dumps(out))
    return m,out


class SummaryTests(unittest.TestCase):
    def test_terminal_no_replays_and_same_request_cache_cost_not_double_counted(self):
        with tempfile.TemporaryDirectory()as t:
            run=Path(t)/'first';jobs=[job(),job('hybrid_rank')];m,out=phase(run,jobs)
            with patch.object(d,'terminal',return_value=(m,out)):p=s.audit_phase(run,{})
            self.assertEqual(p['errors'],[])
            rows=s.record_rows(jobs,p)
            self.assertTrue(all(r['judge_parse_mode']=='terminal'and r['scores']['judge_correct']==0 for r in rows))
            self.assertEqual(rows[0]['reader_answer'],'alpha')
            usage=s.measured_usage(p['ledger'])
            self.assertEqual(usage['physical_attempts'],2);self.assertEqual(usage['cache_events'],2)
            self.assertEqual(usage['token_fields']['output']['recorded_token_sum'],6)
            self.assertEqual(usage['token_fields']['thinking']['recorded_token_sum'],4)
            self.assertIsNone(usage['total_cost_usd'])

    def test_invalid_first_label_recovery_inherits_exact_reader_and_rejects_change(self):
        with tempfile.TemporaryDirectory()as t:
            run=Path(t)/'first';j=job();m,out=phase(run,[j],label='no legal label')
            with patch.object(d,'terminal',return_value=(m,out)):first=s.audit_phase(run,{})
            self.assertEqual(first['errors'],[]);paths=d.stage_files(run,j)
            jr={**j,'reader_source':d._base.pointer(paths['reader']),'parent_results':{k:d._base.pointer(v)for k,v in paths.items()}}
            recovery=Path(t)/'recovery';mr,orr=phase(recovery,[jr],'technical_recovery','[[yes]]')
            with patch.object(d,'terminal',return_value=(mr,orr)):second=s.audit_phase(recovery,{},parent=first)
            self.assertEqual(second['errors'],[])
            self.assertIsNone(s.record_rows([j],first)[0]['scores']['judge_correct'])
            final=s.record_rows([j],first,second,final=True)[0]
            self.assertEqual(final['scores']['judge_correct'],1)
            self.assertEqual(final['judge_resolution']['source'],'recovery_accepted_label')
            self.assertEqual(final['reader_artifact'],d._base.pointer(paths['reader']))
            recovered=d.stage_files(recovery,jr)['reader'];r=d._base.verify(recovered)
            r['result']['prediction']['answer']='tampered';r.pop('content_sha256');recovered.unlink();d._base.save_new(recovered,r)
            with patch.object(d,'terminal',return_value=(mr,orr)):bad=s.audit_phase(recovery,{},parent=first)
            self.assertTrue(any(x['stage']=='reader'for x in bad['errors']))

    def test_wrong_model_in_raw_is_not_scored_as_a_no(self):
        with tempfile.TemporaryDirectory()as t:
            run=Path(t)/'first';j=job();m,out=phase(run,[j]);rp=d.stage_files(run,j)['judge']
            row=d._base.verify(rp);rawp=Path(row['api_events'][0]['cache_path']);raw=json.loads(rawp.read_text());raw['request']['model']='gemini-2.5-flash'
            rawp.write_text(json.dumps(raw))
            with patch.object(d,'terminal',return_value=(m,out)):bad=s.audit_phase(run,{})
            self.assertTrue(any(x['stage']=='judge'for x in bad['errors']))
            self.assertIsNone(s.record_rows([j],bad)[0]['scores']['judge_correct'])

    def test_independent_metric_denominators_full10780_and_newbaseline_flips(self):
        from tools.prepare_gemini38_hybrid_v1 import cohort
        rows=[]
        for arm in s.ARMS:
            for ci,questions in cohort()[0].items():
                for q in questions:
                    rows.append({'query_id':q['id'],'conversation':int(ci),'category':q['category'],'method':arm,
                       'selection_status':'complete','reader_status':'complete','judge_status':'missing','qa_status':'failed',
                       'scores':{'official_locomo_f1':.5,'custom_token_f1':.25,'judge_correct':None}})
        for r in rows:
            if r['query_id']=='c0_q0':r['scores']['judge_correct']=1 if r['method']=='baseline'else 0
        result=s.aggregate(rows,bootstrap_samples=2)
        self.assertEqual(len(rows),10780)
        for arm in s.ARMS:
            metrics=result['arms'][arm]['metrics'];self.assertEqual(metrics['official_locomo_f1']['valid_denominator'],1540)
            self.assertEqual(metrics['judge_correct']['valid_denominator'],1)
            self.assertIsNone(metrics['judge_correct']['full_cohort_observed_mean'])
        flip=result['paired_comparisons']['hybrid_rank_minus_baseline']['judge_transitions']
        self.assertEqual(flip['1_to_0']['query_ids'],['c0_q0']);self.assertEqual(flip['missing_either']['count'],1539)

    def test_missing_thinking_and_absent_ledger_are_not_zero_measurements(self):
        use=s.measured_usage([{'model':MODEL,'cache_hit':False,'status':200,'usage':{'promptTokenCount':8,'candidatesTokenCount':2}},
                             {'model':MODEL,'cache_hit':False,'status':'transport_error','usage':{}}],missing_ledgers=['absent'])
        self.assertFalse(use['available']);self.assertEqual(use['token_fields']['thinking']['missing_attempts'],2)
        self.assertEqual(use['physical_status_counts']['transport_or_unknown'],1);self.assertIsNone(use['total_cost_usd'])

    def test_unlogged_intent_is_retained_unknown_even_when_ledger_exists(self):
        with tempfile.TemporaryDirectory()as t:
            api=Path(t);folder=api/'request_intents';folder.mkdir()
            (folder/'unfinished.json').write_text(json.dumps({'request_hash':'unfinished','stage':e.READER_STAGE}))
            (api/'api_calls.jsonl').write_text('')
            result=s.intent_inventory(s.Snapshot(),api,[])
            self.assertEqual(result['unknown_intent_count'],1)
            self.assertEqual(result['unknown_intents_without_ended_physical_proof'][0]['request_hash'],'unfinished')

    def test_missing_source_or_unprepared_phase_fail_closed_readiness_has_no_score(self):
        with tempfile.TemporaryDirectory()as t:
            self.assertFalse(s.readiness(t)['ready_for_summary'])
            self.assertFalse(s.readiness(t)['results_computed'])
            with self.assertRaises(FileNotFoundError):s.summarize(t)


if __name__=='__main__':unittest.main()
