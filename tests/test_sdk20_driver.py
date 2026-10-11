"""Thin-driver protocol tests: synthetic banks, actual old evaluator, no HTTP."""
from contextlib import ExitStack
from contextlib import contextmanager
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_sdk20_modules as target
from jev_mem.providers import ModelAPI
from jev_mem.selection import COARSE_CRITERIA, FINE_CRITERIA


class FakeAPI(ModelAPI):
    calls = []
    def __init__(self, run_dir, *, context=None, **kwargs):
        super().__init__(run_dir, keys={'GEMINI_API_KEY':'test', 'TYPESAFE_API_KEY':'test'},
                         context=context)
        self.events = []
    def scoped(self, **context):
        return type(self)(self.run_dir, context={**self.context, **context})
    def request(self, provider, model, body, **kwargs):
        call={'provider':provider,'model':model,'body':deepcopy(body),**kwargs,'context':deepcopy(self.context)}
        self.calls.append(call)
        self.events.append({'stage':kwargs.get('stage'),'source':'mock'})
        if provider=='jev':
            return {'answers':{k:{'noul':0.5} for k in body['questions']}}
        value=({'answer':'A name','used_fact_ids':['f1']} if kwargs['stage']=='qa_reader' else
               {'correct':True,'completeness':1.0,'reason':'Equivalent'})
        return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(value)}]}}]}


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name);self.folder=self.run/'conv_0'
        self.m={'protocol_sha256':'mock-protocol'}
        self.q={'id':'c0_q0','question':'Which name?','reference':'QA_GOLD_SENTINEL','category':4}
        self.facts=[{'id':f'id{i}','text':f'Fact {i}','observation_date':'2023-01-01',
                     'input_scope':{'session_id':'session_1'}} for i in range(20)]
        self.sessions=[{'session_id':'session_1','observation_date':'2023-01-01',
                        'messages':[{'content':'Alice: Whole original first turn.'},{'content':'Bob: Whole second turn.'}]},
                       {'session_id':'session_2','observation_date':'2023-12-31',
                        'messages':[{'content':'Later unrelated session.'}]}]
        FakeAPI.calls=[]
    def bank(self,run,ci,kind,m):
        return {'facts':deepcopy(self.facts)},kind+'-hash'
    def common(self):
        stack=ExitStack()
        for name,value in [('CachedLegacyAPI',FakeAPI),('get_bank',self.bank),
                           ('load_data',lambda:[{}]),('queries',lambda item,ci:[self.q]),
                           ('session_envelopes',lambda item:self.sessions)]:
            stack.enter_context(patch.object(target,name,value))
        return stack
    def put(self,rel,row):
        target.save_new(self.folder/rel,{'schema':target.SCHEMA,'protocol_sha256':self.m['protocol_sha256'],**row})

    def test_both_qa_arms_use_actual_legacy_generate_and_last_session_date(self):
        ids=[f['id'] for f in self.facts]
        for method,kind in zip(target.ARMS[1:],['session','preseg']):
            self.put(f'selection/{method}/c0_q0.json',{'status':'complete','selected_ids':ids,'bank_sha256':kind+'-hash'})
        with self.common(),patch.object(target,'ThreadPoolExecutor',wraps=ThreadPoolExecutor) as workers:
            result=target.qa(self.run,0,self.m)
            self.assertEqual(result['failed'],0)
            self.assertEqual(workers.call_args.kwargs['max_workers'],4)
            self.assertEqual(len(FakeAPI.calls),4)
            for call in FakeAPI.calls:
                cfg=call['body']['generationConfig']
                self.assertEqual(cfg['maxOutputTokens'],2048)
                self.assertEqual(cfg['temperature'],0)
                self.assertEqual(cfg['responseMimeType'],'application/json')
                self.assertNotIn('responseSchema',cfg)
                prompt=json.loads(call['body']['contents'][0]['parts'][0]['text'])
                if call['stage']=='qa_reader':
                    self.assertEqual(prompt['conversation_reference_date'],'2023-12-31')
                    self.assertEqual(len(prompt['facts']),20)
                    self.assertEqual([f['id'] for f in prompt['facts']],[f'f{i+1}' for i in range(20)])
                    self.assertNotIn('QA_GOLD_SENTINEL',json.dumps(prompt))
                else:self.assertEqual(prompt['reference_answer'],'QA_GOLD_SENTINEL')
            target.qa(self.run,0,self.m)
            self.assertEqual(len(FakeAPI.calls),4,'Existing successful rows must not be rerun')

    def test_failed_selection_produces_null_no_reader_and_no_implicit_retry(self):
        for method,kind in zip(target.ARMS[1:],['session','preseg']):
            self.put(f'selection/{method}/c0_q0.json',{'status':'failed','selected_ids':[],
                      'bank_sha256':kind+'-hash','error':{'type':'ValueError','message':'known failure'}})
        with self.common():
            result=target.qa(self.run,0,self.m)
            self.assertEqual(result['failed'],2)
            for method in target.ARMS[1:]:
                row=target.verify(self.folder/f'qa/{method}/c0_q0.json')
                self.assertEqual(row['status'],'failed');self.assertIsNone(row['result'])
                self.assertEqual(row['selected_ids'],[])
            target.qa(self.run,0,self.m)
            self.assertEqual(FakeAPI.calls,[])

    def test_tag_uses_entire_original_session_and_ten_existing_criteria(self):
        self.facts=self.facts[:1]
        with self.common():
            outcome=target.tag(self.run,0,self.m)
            self.assertEqual(outcome['failed'],0)
            self.assertEqual(len(FakeAPI.calls),1)
            call=FakeAPI.calls[0]
            self.assertEqual(set(call['body']['questions']),set(COARSE_CRITERIA)|set(FINE_CRITERIA))
            for message in self.sessions[0]['messages']:self.assertIn(message['content'],call['body']['state'])
            self.assertNotIn('Later unrelated session.',call['body']['state'])
            self.assertNotIn('QA_GOLD_SENTINEL',call['body']['state'])
            row=target.verify(self.folder/'tags/id0.json')
            self.assertEqual(row['source_scope'],'whole_original_session_unverified_input_scope')
            self.assertEqual(row['source_context_sha256'],target.digest('\n'.join(m['content'] for m in self.sessions[0]['messages'])))
            target.tag(self.run,0,self.m)
            self.assertEqual(len(FakeAPI.calls),1)

    def test_baseline_copies_success_and_failed_null_without_qa_calls(self):
        second={**self.q,'id':'c0_q1'}
        ids=[f['id'] for f in self.facts]
        def old(ci,q):
            failed=q['id']=='c0_q1'
            return {'status':'failed' if failed else 'complete','result':None if failed else {'prediction':{'answer':'old'}},
                    'selected_ids':ids,'source':{'kind':'technical_recovery' if failed else 'firstpass'},
                    'retrieval_source':{'path':'frozen-original','sha256':'sha'},
                    'error':{'type':'ValueError','message':'Reader did not return an answer string'} if failed else None}
        with self.common(),patch.object(target,'queries',lambda item,ci:[self.q,second]),patch.object(target.sources,'baseline_row',side_effect=old) as inherited:
            result=target.baseline(self.run,0,self.m)
            self.assertEqual(result['completed'],2);self.assertEqual(result['failed'],1)
            failed=target.verify(self.folder/f'qa/{target.ARMS[0]}/c0_q1.json')
            self.assertTrue(failed['inherited_result']);self.assertIsNone(failed['result'])
            self.assertEqual(failed['api_events'],[])
            target.baseline(self.run,0,self.m)
            self.assertEqual(inherited.call_count,2)
            self.assertEqual(FakeAPI.calls,[])

    def test_query_type_scores_feed_actual_soft_selector_only_for_type_arm(self):
        from tests.test_sdk20_type_adapter import captured_memory
        _,captured,_=captured_memory()
        self.facts=[{'id':i,'text':'Fact '+i,'observation_date':'2023-01-01',
                     'input_scope':{'session_id':'session_1'}} for i in captured['eligible_ids']]
        for fact in self.facts:
            self.put('tags/'+fact['id']+'.json',{'status':'complete','bank_sha256':'session-hash',
                     'result':{'fact_id':fact['id'],'coarse_scores':{k:0.5 for k in COARSE_CRITERIA},
                               'fine_scores':{k:0.5 for k in FINE_CRITERIA}}})
        for kind in ['session','preseg']:
            self.put('retrieval/'+kind+'/c0_q0.json',{'status':'complete','bank_sha256':kind+'-hash',
                     'capture':captured,'selected_ids':captured['base20_ids']})
        with self.common():
            result=target.select_queries(self.run,0,self.m)
            self.assertEqual(result['failed'],0)
            self.assertEqual(len(FakeAPI.calls),1)
            call=FakeAPI.calls[0]
            self.assertEqual(call['stage'],'pilot_query_evidence_types')
            self.assertIn('Which name?',call['body']['state'])
            self.assertNotIn('QA_GOLD_SENTINEL',call['body']['state'])
            chosen=target.verify(self.folder/f'selection/{target.ARMS[1]}/c0_q0.json')
            self.assertEqual(chosen['selected_ids'],captured['base20_ids'])
            self.assertEqual(chosen['trace']['fallback_reason'],'equal_type_affinities')
            self.assertEqual(chosen['trace']['query_type_scores'],{k:0.5 for k in (*COARSE_CRITERIA,*FINE_CRITERIA)})
            target.select_queries(self.run,0,self.m)
            self.assertEqual(len(FakeAPI.calls),1)

    def test_native_sdk20_replay_drift_is_persisted_and_stops_retrieval(self):
        from types import SimpleNamespace
        from mem0.utils import spacy_models
        ids=[f['id'] for f in self.facts]
        previous=self.folder/'old_retrieval.json';previous.parent.mkdir(parents=True)
        previous.write_text(json.dumps({'sdk20':[{'id':i,'text':'Fact '+i[2:],
            'observation_date':'2023-01-01','score':0.5} for i in ids]}),encoding='utf-8')
        drift=[ids[1],ids[0],*ids[2:]]
        calls=[]
        def search(question,**kwargs):
            calls.append(kwargs)
            return {'results':[{'id':i,'score':0.5} for i in drift]}
        memory=SimpleNamespace(vector_store=SimpleNamespace(_get_bm25_encoder=lambda:object()),search=search)
        @contextmanager
        def capture():yield {'base20_ids':drift,'eligible_ids':ids}
        def bank(run,ci,kind,m):
            b,h=self.bank(run,ci,kind,m)
            b.update(native_dir=str(self.folder/kind/'native'),user_id='user',source_inventory={});return b,h
        with self.common(),patch.object(target,'get_bank',bank),patch.object(target,'capture_native_pool',capture),\
            patch.object(target.native,'open_memory',return_value=memory),patch.object(target.native,'close_memory'),\
            patch.object(target.sources,'_native_inventory',return_value={}),\
            patch.object(spacy_models,'get_nlp_full',return_value=object()),patch.object(spacy_models,'get_nlp_lemma',return_value=object()),\
            patch.object(target.sources,'baseline_row',return_value={'retrieval_source':{'path':str(previous)}}):
            with self.assertRaisesRegex(ValueError,'ordered IDs did not replay'):
                target.retrieve(self.run,0,self.m)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0],{'top_k':20,'filters':{'user_id':'user'},'rerank':False})
            failed=target.verify(self.folder/'retrieval/session/c0_q0.json')
            self.assertEqual(failed['status'],'failed');self.assertEqual(failed['selected_ids'],[])
            with self.assertRaisesRegex(RuntimeError,'Retained retrieval failure'):
                target.retrieve(self.run,0,self.m)
            self.assertEqual(len(calls),1,'Known drift must not re-search or silently retry')


if __name__=='__main__':unittest.main()
