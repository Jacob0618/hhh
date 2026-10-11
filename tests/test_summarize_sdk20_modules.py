from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import patch

from tools import summarize_sdk20_modules as s


class PureSummaryTests(unittest.TestCase):
    def row(self):
        q={'id':'c0_q0','conversation':0,'question':'Who?','reference':'Alice','category':4}
        facts=[{'id':f'x{i}','text':'Alice','observation_date':'2023-01-01'} for i in range(20)]
        class Fake:
            def generate(self,*a,**kw):
                return ({'answer':'Alice','used_fact_ids':['f1','f1']} if kw['stage']=='qa_reader'
                        else {'correct':True,'completeness':1.,'reason':'same'})
        r={'schema':s.SCHEMA,'protocol_sha256':'protocol','method':'fact_types_sdk20',
           **{k:q[k] for k in ('conversation','question','reference','category')},'query_id':q['id'],
           'status':'complete','result':s.answer_and_judge(Fake(),q['question'],q['reference'],facts,reference_date='2023-01-01'),
           'selected_ids':[f['id'] for f in facts],'error':None}
        return q,r

    def test_existing_evaluator_citation_contract_preserved(self):
        q,row=self.row()
        self.assertIsNotNone(s.validate_row(row,q,'fact_types_sdk20','protocol'))

    def test_question_gold_protocol_and_metric_drift_rejected(self):
        q,row=self.row()
        changes=[('reference','someone else'),('protocol_sha256','other'),('question','Other?')]
        for k,v in changes:
            changed=deepcopy(row);changed[k]=v
            with self.assertRaises(s.InvalidArtifact):s.validate_row(changed,q,'fact_types_sdk20','protocol')
        changed=deepcopy(row);changed['result']['token_f1']=0
        with self.assertRaises(s.InvalidArtifact):s.validate_row(changed,q,'fact_types_sdk20','protocol')

    def test_failed_cannot_smuggle_real_score(self):
        q,row=self.row();row['status']='failed';row['error']={'message':'network'}
        with self.assertRaises(s.InvalidArtifact):s.validate_row(row,q,'fact_types_sdk20','protocol')
        row['result']=None
        self.assertIsNone(s.validate_row(row,q,'fact_types_sdk20','protocol'))

    def test_invalid_judge_and_wrong_fact_mapping_rejected(self):
        q,row=self.row()
        for val in (True,float('nan'),1.1):
            changed=deepcopy(row);changed['result']['judgment']['completeness']=val
            with self.assertRaises(s.InvalidArtifact):s.validate_row(changed,q,'fact_types_sdk20','protocol')
        changed=deepcopy(row);changed['result']['prediction']['used_fact_ids']=['x9','x9']
        with self.assertRaises(s.InvalidArtifact):s.validate_row(changed,q,'fact_types_sdk20','protocol')

    def test_failures_keep_denominator_and_pairing_excludes_without_imputing(self):
        qs=[{'id':f'c{ci}_q{i}','conversation':ci,'category':4} for ci in range(10) for i in range(3)]
        slots={}
        for arm in s.ARMS:
            for q in qs:
                scores={'query_id':q['id'],'conversation':q['conversation'],'category':4,**{m:1. for m in s.METRICS}}
                slots[arm,q['id']]={'conversation':q['conversation'],'category':4,'status':'complete','scores':scores}
        slots[s.ARMS[0],'c0_q0']={'conversation':0,'category':4,'status':'failed','scores':None}
        arms,pairs=s.aggregate(qs,slots,bootstrap_samples=20)
        a=arms[s.ARMS[0]]
        self.assertEqual(a['cohort_denominator'],30)
        self.assertEqual(a['completed_only_metrics']['official_locomo_f1'],1.)
        self.assertIsNone(a['full_cohort_observed_metrics']['official_locomo_f1'])
        self.assertAlmostEqual(a['missing_zero_sensitivity']['means']['official_locomo_f1'],29/30)
        p=pairs['fact_types_sdk20_minus_baseline_sdk20']
        self.assertEqual(p['common_success_count'],29);self.assertTrue(p['bootstrap']['eligible'])
        self.assertEqual(p['metrics']['official_locomo_f1']['other_minus_baseline'],0.)

    def test_not_all_ten_conversations_never_bootstrap_as_full_ten(self):
        qs=[{'id':'c0_q0','conversation':0,'category':4}]
        score={'query_id':'c0_q0','conversation':0,'category':4,**{m:1. for m in s.METRICS}}
        slots={(arm,'c0_q0'):{'conversation':0,'category':4,'status':'complete','scores':score} for arm in s.ARMS}
        _,pairs=s.aggregate(qs,slots,bootstrap_samples=20)
        p=pairs['fact_types_sdk20_minus_baseline_sdk20']
        self.assertFalse(p['bootstrap']['eligible']);self.assertIsNone(p['metrics']['judge_correct']['cluster_bootstrap_95_percentile_interval'])

    def test_fake_cache_error_consumer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'conv_0/api/api_calls.jsonl';p.parent.mkdir(parents=True)
            row={'context':{'conversation':0},'cache_hit':True,'status':503,'usage_semantics':'reused_response_no_new_model_usage'}
            p.write_text(json.dumps(row)+'\n',encoding='utf8')
            with self.assertRaises(s.InvalidArtifact):s.read_ledgers(root)

    def test_unknown_transport_and_embedding_cost_not_zero_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[{'model':'gemini-2.5-flash','status':'transport_error','attempt':1,'usage_unknown':True},
                  {'model':'gemini-embedding-001','status':200,'attempt':1,'usage':{'promptTokenCount':3}},
                  {'model':'gemini-2.5-flash','cache_hit':True,'usage':{'promptTokenCount':1000000}}]
            u=s.usage_report(Path(tmp),rows,[],[])
            self.assertIsNone(u['new_current_run']['total_cost_usd'])
            self.assertEqual(u['new_current_run']['unknown_cost_attempts'],2)
            self.assertEqual(u['new_current_run']['physical_attempts'],2)
            self.assertEqual(u['new_current_run']['cache_events'],1)

    def recorded(self,root,*,bad_answer=False):
        context={'conversation':0,'phase':'qa','query_id':'c0_q0','method':'fact_types_sdk20'}
        events=[];ledger=defaultdict(list)
        class Recording:
            generate=s.ModelAPI.generate
            def request(self,provider,model,body,*,operation='generateContent',stage='unspecified'):
                sig={'provider':provider,'model':model,'operation':operation,'body':body};key=s.request_digest(sig)
                result=({'answer':17 if bad_answer else 'Alice','used_fact_ids':['f1']} if stage=='qa_reader'
                        else {'correct':True,'completeness':1.,'reason':'yes'})
                raw={'request':sig,'response':{'modelVersion':model,'usageMetadata':{'promptTokenCount':12,'candidatesTokenCount':3},
                    'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(result)}]}}]}}
                target=root/'api_cache'/f'{key}.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(raw),encoding='utf8')
                intent=root/'request_intents'/f'{key}.json';intent.parent.mkdir(exist_ok=True)
                intent.write_text(json.dumps({'request_hash':key,'signature':sig,'context':context,'stage':stage,'retries':2}),encoding='utf8')
                events.append({'request_hash':key,'stage':stage,'provider':provider,'model':model,'operation':operation,
                    'context':context,'source':'new_request','cache_path':str(target),'cache_sha256':s.sha(target),'error_type':None})
                ledger[key,stage,s.fingerprint(context)].append({'provider':provider,'model':model,'operation':operation,
                    'request_hash':key,'stage':stage,'context':context,'status':200,'attempt':1,'cache_hit':False,
                    'usage':raw['response']['usageMetadata'],'resolved_model':model})
                return raw['response']
        facts=[{'id':f'x{i}','text':'Alice','observation_date':'2023-01-01'} for i in range(20)]
        result=None
        try:result=s.answer_and_judge(Recording(),'Who?','Alice',facts,reference_date='2023-01-01')
        except ValueError:
            if not bad_answer:raise
        return context,events,ledger,facts,result

    def test_exact_legacy_payload_raw_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,result=self.recorded(root)
            api=s.ReplayAPI(root,events,context,ledger)
            self.assertEqual(s.answer_and_judge(api,'Who?','Alice',facts,reference_date='2023-01-01'),result)
            api.assert_consumed()

    def test_raw_schema_or_missing_producer_cannot_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,_=self.recorded(root)
            with self.assertRaises(s.InvalidArtifact):
                s.answer_and_judge(s.ReplayAPI(root,events,context,{}),'Who?','Alice',facts,reference_date='2023-01-01')
            p=Path(events[0]['cache_path']);raw=json.loads(p.read_text(encoding='utf8'))
            raw['request']['body']['generationConfig']['responseSchema']={'type':'OBJECT'}
            p.write_text(json.dumps(raw),encoding='utf8');events[0]['cache_sha256']=s.sha(p)
            with self.assertRaises(s.InvalidArtifact):
                s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')

    def test_real_http200_nonstring_failure_replays_as_legacy_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,_=self.recorded(root,bad_answer=True)
            api=s.ReplayAPI(root,events,context,ledger)
            with self.assertRaisesRegex(ValueError,'Reader did not return an answer string'):
                s.answer_and_judge(api,'Who?','Alice',facts,reference_date='2023-01-01')
            api.assert_consumed()

    def test_inherited_raw_cache_requires_bound_source_and_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,result=self.recorded(root)
            old=root/'old_cache';old.mkdir();(root/'cache_lineage').mkdir()
            for event in events:
                target=Path(event['cache_path']);source=old/target.name;source.write_bytes(target.read_bytes())
                line={'source_path':str(source),'source_sha256':s.sha(source),'target_path':str(target),
                      'target_sha256':s.sha(target),'request_hash':event['request_hash']}
                (root/'cache_lineage'/target.name).write_text(json.dumps(line),encoding='utf8')
                event['source']='inherited_exact_request'
                (root/'request_intents'/target.name).unlink()
                raw=json.loads(target.read_text(encoding='utf8'))
                ledger[event['request_hash'],event['stage'],s.fingerprint(context)]=[{'context':context,'cache_hit':True,
                     'usage_semantics':'reused_response_no_new_model_usage','request_hash':event['request_hash'],'stage':event['stage'],
                     'provider':event['provider'],'model':event['model'],'operation':event['operation'],
                     'usage':raw['response']['usageMetadata'],'resolved_model':event['model']}]
            with patch.object(s.driver,'cache_roots',return_value=[old]):
                api=s.ReplayAPI(root,events,context,ledger)
                self.assertEqual(s.answer_and_judge(api,'Who?','Alice',facts,reference_date='2023-01-01'),result)
                api.assert_consumed()
                (old/Path(events[0]['cache_path']).name).write_text('{}',encoding='utf8')
                with self.assertRaises(s.InvalidArtifact):
                    s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')

    def test_transport_failure_requires_actual_failed_attempt_not_http200(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,_=self.recorded(root)
            events=events[:1];event=events[0];Path(event['cache_path']).unlink()
            event['cache_sha256']=None;event['error_type']='RuntimeError'
            with self.assertRaises(s.InvalidArtifact):
                s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')
            ledger[event['request_hash'],event['stage'],s.fingerprint(context)][0]['status']='transport_error'
            api=s.ReplayAPI(root,events,context,ledger)
            with self.assertRaises(s.DeclaredTransportFailure):
                s.answer_and_judge(api,'Who?','Alice',facts,reference_date='2023-01-01')
            api.assert_consumed()

    def test_inherited_cache_precedence_and_conflicting_intent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,_=self.recorded(root)
            old=root/'old';old.mkdir();first=root/'priority';first.mkdir();(root/'cache_lineage').mkdir()
            event=events[0];target=Path(event['cache_path']);source=old/target.name;source.write_bytes(target.read_bytes())
            raw=json.loads(target.read_text(encoding='utf8'))
            line={'source_path':str(source),'source_sha256':s.sha(source),'target_path':str(target),
                  'target_sha256':s.sha(target),'request_hash':event['request_hash']}
            (root/'cache_lineage'/target.name).write_text(json.dumps(line),encoding='utf8')
            event['source']='inherited_exact_request'
            cache={'context':context,'cache_hit':True,'usage_semantics':'reused_response_no_new_model_usage',
                'request_hash':event['request_hash'],'stage':event['stage'],'provider':event['provider'],
                'model':event['model'],'operation':event['operation'],'usage':raw['response']['usageMetadata'],
                'resolved_model':event['model']}
            ledger[event['request_hash'],event['stage'],s.fingerprint(context)]=[cache]
            with patch.object(s.driver,'cache_roots',return_value=[first,old]):
                with self.assertRaisesRegex(s.InvalidArtifact,'conflicts with local dispatch'):
                    s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')
                (root/'request_intents'/target.name).unlink();(first/target.name).write_bytes(target.read_bytes())
                with self.assertRaisesRegex(s.InvalidArtifact,'source precedence'):
                    s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')

    def test_ledger_identity_usage_and_resolved_model_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,_=self.recorded(root)
            key=(events[0]['request_hash'],events[0]['stage'],s.fingerprint(context))
            for field,value in [('provider','wrong'),('model','wrong'),('operation','wrong'),('usage',{}),('resolved_model','wrong')]:
                bad=deepcopy(ledger);bad[key][0][field]=value
                with self.assertRaises(s.InvalidArtifact,msg=field):
                    s.answer_and_judge(s.ReplayAPI(root,events,context,bad),'Who?','Alice',facts,reference_date='2023-01-01')
            p=Path(events[0]['cache_path']);raw=json.loads(p.read_text(encoding='utf8'))
            raw['response']['modelVersion']='unapproved-alias';p.write_text(json.dumps(raw),encoding='utf8')
            events[0]['cache_sha256']=s.sha(p);ledger[key][0]['resolved_model']='unapproved-alias'
            with self.assertRaisesRegex(s.InvalidArtifact,'Unapproved resolved model'):
                s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')

    def test_physical_retry_sequence_and_nonretryable_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,result=self.recorded(root)
            key=(events[0]['request_hash'],events[0]['stage'],s.fingerprint(context));base=ledger[key][0]
            cases=[[(1,503),(2,503),(3,503),(4,200)],[(1,503),(1,200)],[(2,503),(1,200)],[(1,401),(2,200)],[(1,200),(2,200)]]
            for sequence in cases:
                bad=deepcopy(ledger);bad[key]=[{**base,'attempt':n,'status':status} for n,status in sequence]
                with self.assertRaises(s.InvalidArtifact,msg=str(sequence)):
                    s.answer_and_judge(s.ReplayAPI(root,events,context,bad),'Who?','Alice',facts,reference_date='2023-01-01')
            ledger[key]=[{**base,'attempt':1,'status':503},{**base,'attempt':2,'status':200}]
            api=s.ReplayAPI(root,events,context,ledger)
            self.assertEqual(s.answer_and_judge(api,'Who?','Alice',facts,reference_date='2023-01-01'),result)
            api.assert_consumed()

    def test_same_request_independent_conversation_and_repeated_consumers_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,result=self.recorded(root)
            for event in events:
                key=(event['request_hash'],event['stage'],s.fingerprint(context));base=ledger[key][0]
                other={**context,'conversation':1}
                ledger[event['request_hash'],event['stage'],s.fingerprint(other)]=[{**base,'context':other}]
                cache={k:v for k,v in base.items() if k not in ('status','attempt')}
                cache.update(cache_hit=True,usage_semantics='reused_response_no_new_model_usage')
                ledger[key].extend([cache,deepcopy(cache)])
            api=s.ReplayAPI(root,events,context,ledger)
            self.assertEqual(s.answer_and_judge(api,'Who?','Alice',facts,reference_date='2023-01-01'),result)
            api.assert_consumed()

    def test_unknown_or_nonfailed_http_cannot_justify_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,events,ledger,facts,_=self.recorded(root)
            events=events[:1];event=events[0];Path(event['cache_path']).unlink();event.update(cache_sha256=None,error_type='RuntimeError')
            key=(event['request_hash'],event['stage'],s.fingerprint(context))
            for status in (301,'unknown',None,200,True,600):
                ledger[key][0]['status']=status
                with self.assertRaises(s.InvalidArtifact,msg=str(status)):
                    s.answer_and_judge(s.ReplayAPI(root,events,context,ledger),'Who?','Alice',facts,reference_date='2023-01-01')

    def test_historical_baseline_qa_ledger_redispatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'conv_0/api/api_calls.jsonl';p.parent.mkdir(parents=True)
            row={'context':{'conversation':0,'phase':'qa','method':'baseline_sdk20'},'stage':'qa_reader',
                 'provider':'gemini','operation':'generateContent','model':'gemini-2.5-flash','attempt':1,'status':200}
            p.write_text(json.dumps(row)+'\n',encoding='utf8')
            with self.assertRaisesRegex(s.InvalidArtifact,'module-only dispatch'):s.read_ledgers(root)

if __name__=='__main__':unittest.main()
