"""Ten tiny synthetic source banks through real native/API/QA/audit/seal.

Only the admission review/grant is an explicitly isolated fixture. All science,
transport ledgers (fake HTTP), prefix ordering, native topic replay and QA checks
execute their actual implementations. This is ten questions, not real 1540 QA.
"""
from contextlib import ExitStack
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests
from tests.test_baselines_hypermem_v1 import FakeAPI,conversation
from tools import run_five_baselines_hypermem_native_v4 as c
from tools import audit_hypermem_continuation_v4 as audit
from tools import hypermem_prefix_api_v4 as prefix
from tools.baselines_common_v1 import read,save_new,sha
from jev_mem import providers
from jev_mem.gemini25_ablation_provider_v1 import READER_STAGE,JUDGE_STAGE

def response(text):
    return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}],
        'modelVersion':'gemini-2.5-flash'}

class FullSyntheticAudit(unittest.TestCase):
    def test_ten_real_banks_raw_native_qa_recovery_and_original_four_result_outputs(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research',prefix='v4_complete_tiny_') as tmp,ExitStack() as stack:
            root=Path(tmp);parent=root/'parent';new=root/'new';dataset=[];cases=[];registry={};questions=[]
            # Original tiny successful bank supplies a single exact retained
            # first request. The rest is fake first-seen transport below.
            for ci in range(10):
                conv=conversation();conv['speaker_a']=f'A{ci}';conv['speaker_b']=f'B{ci}'
                for i,t in enumerate(conv['session_1']):t.update(speaker=conv['speaker_a'] if i%2==0 else conv['speaker_b'],text=t['text']+f' Tiny ci {ci}.')
                dataset.append({'conversation':conv});bank=parent/f'work/native/c{ci}'
                c.adapter.build(conv,bank,FakeAPI())
                p=bank/'calls/episode_000001_hypermem_boundary.json';value=read(p);sig=audit.signature(value['request']);key=audit.digest(sig)
                raw=parent/'work/api_method/api_cache'/(key+'.json')
                save_new(raw,{'request':sig,'response':response(json.dumps(value['raw_response'])),'stage':'hypermem_boundary'})
                context={'method':'hypermem','conversation':ci,'phase':'build'}
                e={'ordinal':1,'signature':sig,'request_hash':key,'stage':'hypermem_boundary','operation':'generateContent',
                    'context':context,'raw':c.binding(raw),'native_call':c.binding(p),'mode':'inherit','native_status':'success'}
                cases.append({'conversation':ci,'prefix':[e],'source_messages':c.binding(bank/'source_messages.json')})
                registry[key]={'signature':sig,'raw':c.binding(raw),'stages':['hypermem_boundary']}
                q=deepcopy(next(q for q in c.chain.base.questions() if q['conversation']==ci))
                q.update(question=f'Where was the walk, tiny ci {ci}?',reference='park',category=1);questions.append(q)
            proof={'schema':prefix.PROOF_SCHEMA,'status':'candidate_evidence_only','source_bindings':{'fixture':True},
                'conversations':cases,'producer_registry':registry,'compatibility_cases':{},'replacement_cases':{},'role_cases':[],
                'questions':questions,'hash_environment':c.admission.hash_environment(),
                'original_representation_auditor_refusals':[], 'run_roots':{'tiny_parent':str(parent)},
                'run_inventories':{'tiny_parent':c.admission.inventory(parent)},
                'corrected_historical_usage':{'combined':c.admission.usage_presence([])}}
            pp=save_new(root/'proof.json',proof)
            fixed_review=save_new(root/'review.json',{'fixture_only':True})
            fixed_grant=save_new(root/'grant.json',{'fixture_only':True,'real_human_authorization':False})
            # This isolated fixture tests transport/science/audits, not the
            # independently adversarial-tested root authorization gate.
            stack.enter_context(patch.object(prefix,'HUMAN_GRANT_PATH',Path(fixed_grant['path'])))
            stack.enter_context(patch.object(prefix,'REVIEW_PATH',Path(fixed_review['path'])))
            stack.enter_context(patch.object(prefix,'require_human_grant',return_value={'fixture_only':True}))
            stack.enter_context(patch.object(c,'NEW',new));stack.enter_context(patch.object(c,'OLD',parent))
            stack.enter_context(patch.object(c,'check_review',return_value=None))
            stack.enter_context(patch.dict(c.private_controller.__globals__,{'NEW':new,'OLD':parent,'check_review':lambda *args:None}))
            stack.enter_context(patch.object(c.admission.chain.base,'load_data',return_value=dataset))
            stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No credentials')))
            stack.enter_context(patch.object(requests.Session,'request',side_effect=AssertionError('No network')))
            save_new(parent/'run_manifest.json',{'fixture_only':True})
            proof['run_inventories']['tiny_parent']=c.admission.inventory(parent)
            m={'schema':'synthetic-only','questions':questions,'code_sha256':{},'upstream_sha256':{},
                'review':fixed_review,'terminal_proof':pp,'parent_manifest':c.binding(parent/'run_manifest.json'),
                'human_grant':fixed_grant,'continuation_policy':{'fixture_only':True}}
            save_new(new/'run_manifest.json',m)
            class Ticket:
                def consume(self,*args):raise AssertionError('No production main/run in this fixture')
            ctl=c.private_controller(m,proof,Ticket())
            API=ctl.BaselineAPI
            class LocalAPI(API):
                def __init__(self,*args,**kwargs):
                    kwargs.setdefault('keys',{'GEMINI_API_KEY':'synthetic-unused-secret'})
                    kwargs.setdefault('gate_path',root/'gate.sqlite');kwargs.setdefault('interval_seconds',0.00001)
                    super().__init__(*args,**kwargs)
            ctl.BaselineAPI=LocalAPI;fake=FakeAPI();posts=[]
            class HTTP:
                status_code=200;ok=True;headers={}
                def __init__(self,value):self.value=value
                def json(self):return self.value
            def post(session,url,**kwargs):
                f=inspect.currentframe().f_back
                while f and not isinstance(f.f_locals.get('self'),prefix.PrefixAPI):f=f.f_back
                if f is None:raise AssertionError('No real PrefixAPI frame')
                api=f.f_locals['self'];stage=f.f_locals['stage'];context=api.context;body=kwargs['json']
                posts.append((stage,deepcopy(context)))
                if 'batchEmbedContents' in url:
                    vectors=[]
                    for request in body['requests']:
                        text=request['content']['parts'][0]['text'];v=[0.0]*768;v[0]=1;v[1]=len(text)/1000;v[2]=sum(map(ord,text))%41/100
                        vectors.append({'values':v})
                    return HTTP({'embeddings':vectors})
                if 'countTokens' in url:return HTTP({'totalTokens':100})
                if stage==READER_STAGE:text='park'
                elif stage==JUDGE_STAGE:text='invalid label' if context['conversation']==0 and context['phase']=='firstpass' else '[[yes]]'
                else:
                    value=fake.generate(body['contents'][0]['parts'][0]['text'],stage=stage)
                    if stage=='hypermem_episode_roles':
                        while f and f.f_code.co_name!='_assign_episode_roles_and_weights':f=f.f_back
                        if f is None:raise AssertionError('No actual native role caller')
                        for row in value['episode_roles']:row['episode_id']=f.f_locals['simple_to_real'][row['episode_id']]
                    text=json.dumps(value)
                return HTTP(response(text))
            stack.enter_context(patch.object(requests.Session,'post',post))
            outcomes=[]
            for ci,item in enumerate(dataset):
                row=ctl.conv_worker(new,'hypermem',ci,[questions[ci]],item)
                self.assertEqual(row['status'],'terminal',row);outcomes+=row['results']
            failed=[r for r in outcomes if r['judge']=='failed'];self.assertEqual(len(failed),1)
            save_new(new/'qa_recovery_plan.json',{'jobs':failed})
            q=questions[0];ret=read(new/f'results/c0/retrieval/{q["query_id"]}.json')['value']
            reader=read(new/'results/firstpass'/q['query_id']/'reader.json')
            recovered=ctl.qa_one(new,'hypermem',q,ret,phase='recovery',inherited_reader=reader)
            save_new(new/'qa_recovery_outcome.json',{'results':[recovered]})
            checked=audit.audit_all(new,proof)
            self.assertEqual(checked['status'],'passed');self.assertFalse(checked['historical_refusals_waived'])
            self.assertEqual(len(checked['native']['actual_topic_pipeline_replays']),10)
            self.assertEqual(sum(s==READER_STAGE and ctx['phase']=='recovery' for s,ctx in posts),0)
            summary=ctl.seal('hypermem')
            self.assertEqual(summary['effective']['reader_valid'],10);self.assertEqual(summary['effective']['judge_valid'],10)
            self.assertEqual(summary['effective']['slots'],1540) # The actual frozen denominator is retained.
            self.assertEqual(summary['effective']['missing_reader'],1530)
            for name in ('firstpass_per_query','per_query','raw_request_audit','summary'):
                self.assertTrue((new/'results'/(name+'.json')).is_file())
            journal=new/'work/prefix_v4/c0/requests.jsonl';original=journal.read_bytes()
            journal.write_bytes(original+original.splitlines(keepends=True)[-1])
            with self.assertRaisesRegex(ValueError,'Admission/logical count'):audit.audit_all(new,proof)

if __name__=='__main__':unittest.main()
