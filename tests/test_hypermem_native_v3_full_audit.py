"""Ten tiny real native pipelines, original-source prefixes, actual transport,
embeddings, first QA and one uniform Judge recovery. Fake HTTP, no paid APIs.
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
from tools import run_five_baselines_hypermem_native_v3 as c
from tools import baselines_hypermem_native_v2 as old_adapter
from tools import audit_hypermem_continuation_v3 as audit
from tools import hypermem_prefix_api_v3 as prefix
from tools.baselines_common_v1 import read,sha,save_new,BaselineFatal
from jev_mem import providers
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body,READER_STAGE,JUDGE_STAGE


def provider_response(text):
    return {'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':text}]}}],
        'modelVersion':'gemini-2.5-flash'}


def make_prefixes(root):
    old=root/'parent';ancestor=root/'ancestor';dataset=[];cases=[];registry={};roles=[]
    for ci in range(10):
        conv=conversation(4);conv['speaker_a']=f'A{ci}';conv['speaker_b']=f'B{ci}'
        for i,t in enumerate(conv['session_1']):t.update(speaker=conv['speaker_a'] if i%2==0 else conv['speaker_b'],text=t['text']+f' Unique {ci}.')
        dataset.append({'conversation':conv});work=old/f'work/native/c{ci}';entries=[];fake=FakeAPI()
        class Capture:
            def generate(self,prompt,**kwargs):
                stage=kwargs['stage'];value=fake.generate(prompt,**kwargs)
                if len(entries)==0:value.pop('should_wait')
                frame=inspect.currentframe().f_back
                native_request=frame.f_locals['request'];native_path=frame.f_locals['path']
                if stage=='hypermem_episode_roles':
                    caller=frame.f_back;mapping=caller.f_locals['simple_to_real']
                    for row in value['episode_roles']:row['episode_id']=mapping[row['episode_id']]
                text='{"invalid":' if ci==5 and stage=='hypermem_episode' else json.dumps(value)
                sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
                    'body':generation_body(prompt,system=None,json_output=True,max_output=16384)};key=digest(sig)
                raw=(ancestor if not entries else old)/'api_cache'/(key+'.json')
                save_new(raw,{'request':sig,'response':provider_response(text),'stage':stage})
                entry={'ordinal':len(entries)+1,'request_hash':key,'signature':sig,
                    'context':{'method':'hypermem','conversation':ci,'phase':'build'},'stage':stage,'operation':'generateContent',
                    'raw':c.binding(raw),'native_call':{'path':str(native_path),'sha256':'pending'},
                    'mode':'replace_once' if ci==5 and stage=='hypermem_episode' else 'inherit'}
                entries.append(entry)
                if text=='{"invalid":':raise BaselineFatal('Retained generation cannot be parsed: JSONDecodeError')
                return value
        with unittest.TestCase().assertRaises((old_adapter.HyperMemError,BaselineFatal)):old_adapter.build(conv,work,Capture())
        for e in entries:
            e['native_call']=c.binding(e['native_call']['path']);e['native_status']=read(e['native_call']['path'])['status']
            physical={'cache_hit':False,'usage':{},'status':200}
            registry[e['request_hash']]={'signature':e['signature'],'raw':e['raw'],'stages':[e['stage']],
                'physical_attempts':[] if e['ordinal']==1 else [physical],
                'inherited_parent':{'physical':physical} if e['ordinal']==1 else None}
        case={'conversation':ci,'prefix':entries,'source_messages':c.binding(work/'source_messages.json')};cases.append(case)
        if ci!=5:
            e=entries[-1];value=read(e['native_call']['path'])['raw_response'];episodes=read(work/'episodes.json')
            mapping={f'episode_{i+1}':ep['event_id'] for i,ep in enumerate(episodes)}
            derived,j=c.adapter.represent_episode_ids(value,mapping)
            roles.append({'conversation':ci,'request_hash':e['request_hash'],'raw':e['raw'],'native_call':e['native_call'],
                'mapping_from_complete_unique_native_display':[{'simple_id':k,'event_id':v} for k,v in mapping.items()],
                'original_parsed_sha256':j['original_parsed_sha256'],'derived_parsed_sha256':j['derived_parsed_sha256']})
    e=cases[5]['prefix'][-1]
    replacement={'kind':'known_complete_invalid_episode_json','retry_limit':1,'retry_permission_active':False,
        'conversation':5,'request_hash':e['request_hash'],'signature':e['signature'],'context':e['context'],
        'stage':e['stage'],'raw':e['raw'],'native_call':e['native_call'],'prefix_ordinal':e['ordinal']}
    rr=root/'role_qualification.json';save_new(rr,{'fixture_only':True,'rows':roles})
    proof={'schema':prefix.PROOF_SCHEMA,'status':'passed','conversations':cases,'producer_registry':registry,'role_cases':roles,
        'role_qualification':c.binding(rr),'replacement':replacement,
        'original_inventory':c.admission.inventory(ancestor),'current_inventory':c.admission.inventory(old)}
    return ancestor,old,proof,dataset


class CompleteAudit(unittest.TestCase):
    def test_actual_bad_replacement_transport_cannot_make_partial_bank_enter_qa(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);new=root/'continuation';ancestor,old,proof,dataset=make_prefixes(root)
            proof['questions']=c.chain.base.questions();pp=root/'proof.json';save_new(pp,proof)
            m={'questions':proof['questions'],'code_sha256':{},'upstream_sha256':{},'review':c.binding(pp),
                'terminal_proof':c.binding(pp),'parent_manifest':c.binding(pp),'continuation_policy':c.policy(proof,True)}
            save_new(new/'run_manifest.json',m)
            stack.enter_context(patch.object(c,'NEW',new))
            class Ticket:
                def consume(self,method):raise AssertionError('No production orchestration')
            controller=c.private_controller(m,proof,Ticket())
            API=prefix.admitted_api(proof,new,manifest=c.binding(new/'run_manifest.json'),proof_binding=m['terminal_proof'],replacement_enabled=True)
            class LocalAPI(API):
                def __init__(self,*args,**kw):
                    kw.setdefault('keys',{'GEMINI_API_KEY':'offline'});super().__init__(*args,**kw)
            controller.BaselineAPI=LocalAPI;sends=[]
            class Response:
                status_code=200;ok=True;headers={}
                def json(self):return provider_response('{"new_still_bad":')
            def post(*a,**kw):
                self.assertTrue((new/'work/replacement_v3/known_invalid_episode_once.json').exists());sends.append(1);return Response()
            stack.enter_context(patch.object(requests.Session,'post',post))
            stack.enter_context(patch.object(requests.Session,'request',side_effect=AssertionError('No network')))
            controller.qa_one=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No QA for partial build'))
            result=controller.conv_worker(new,'hypermem',5,[q for q in proof['questions'] if q['conversation']==5],dataset[5])
            self.assertEqual(result['status'],'needs_review');self.assertEqual(len(sends),1)
            self.assertTrue((new/'work/prefix_v3/c5/complete.json').exists())
            self.assertFalse((new/'results/c5/build.json').exists());self.assertFalse((new/'results/c5/retrieval').exists())
            self.assertFalse((new/'work/native/c5/state.json').exists());self.assertFalse((new/'results/firstpass').exists())
            key=proof['replacement']['request_hash'];e=proof['conversations'][5]['prefix'][-1];sig=e['signature']
            self.assertNotEqual(sha(new/'work/api_method/api_cache'/(key+'.json')),e['raw']['sha256'])
            self.assertEqual(sha(e['raw']['path']),e['raw']['sha256'])
            api=LocalAPI(new/'work/api_method',context=e['context'])
            with self.assertRaises(BaselineFatal):api.request(sig['provider'],sig['model'],sig['body'],stage=e['stage'])
            self.assertEqual(len(sends),1)

    def test_real_prefix_representation_replacement_native_qa_uniform_recovery_and_seal(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);new=root/'continuation';ancestor,old,proof,dataset=make_prefixes(root)
            qs=[]
            for ci in range(10):
                q=deepcopy(next(q for q in c.chain.base.questions() if q['conversation']==ci))
                q.update(question=f'Where did A walk, conversation {ci}?',reference='park',category=1);qs.append(q)
            proof['questions']=qs
            pp=root/'proof.json';save_new(pp,proof)
            m={'questions':qs,'code_sha256':{},'upstream_sha256':{},'review':c.binding(pp),'terminal_proof':c.binding(pp),
                'parent_manifest':c.binding(pp),'continuation_policy':c.policy(proof,True)}
            save_new(new/'run_manifest.json',m)
            stack.enter_context(patch.object(c,'NEW',new));stack.enter_context(patch.object(c.admission,'ANCESTOR',ancestor));stack.enter_context(patch.object(c.admission,'OLD',old))
            stack.enter_context(patch.object(c.admission.chain.base,'load_data',lambda:dataset))
            # This fixture tests the unmodified science and complete audits;
            # separate main fixtures own actual review/prepare admission.
            stack.enter_context(patch.object(c,'check_review',lambda *args:None))
            parent=old/'run_manifest.json';save_new(parent,{'fixture_only':True})
            proof['current_inventory']=c.admission.inventory(old)
            class Ticket:
                def consume(self,method):raise AssertionError('Production run not used')
            controller=c.private_controller(m,proof,Ticket())
            RealAPI=prefix.admitted_api(proof,new,manifest=c.binding(new/'run_manifest.json'),proof_binding=m['terminal_proof'],replacement_enabled=True,prospective_role_representation=True)
            class LocalAPI(RealAPI):
                def __init__(self,*args,**kw):
                    kw.setdefault('keys',{'GEMINI_API_KEY':'offline'});kw.setdefault('gate_path',root/'embedding_gate.sqlite');kw.setdefault('interval_seconds',0.00001)
                    super().__init__(*args,**kw)
            controller.BaselineAPI=LocalAPI;fake=FakeAPI();posts=[]
            class Response:
                status_code=200;ok=True;headers={}
                def __init__(self,value):self.value=value
                def json(self):return self.value
            def post(session,url,**kwargs):
                body=kwargs['json'];frame=inspect.currentframe().f_back
                while frame and not isinstance(frame.f_locals.get('self'),prefix.PrefixAPI):frame=frame.f_back
                if frame is None:raise AssertionError('Missing actual PrefixAPI frame')
                stage=frame.f_locals['stage'];context=frame.f_locals['self'].context;posts.append((stage,deepcopy(context)))
                if 'batchEmbedContents' in url:
                    rows=[]
                    for request in body['requests']:
                        text=request['content']['parts'][0]['text'];v=[0.0]*768;v[0]=1;v[1]=len(text)/1000;v[2]=sum(map(ord,text))%41/100;rows.append({'values':v})
                    return Response({'embeddings':rows})
                if 'countTokens' in url:return Response({'totalTokens':100})
                prompt=body['contents'][0]['parts'][0]['text']
                if stage==READER_STAGE:text='park'
                elif stage==JUDGE_STAGE:text='invalid label' if context['conversation']==0 and context['phase']=='firstpass' else '[[yes]]'
                else:
                    value=fake.generate(prompt,stage=stage)
                    # The c5 first unseen roles also uses the separately enabled
                    # prospective contract and actual current caller mapping.
                    if stage=='hypermem_episode_roles':
                        f=frame
                        while f and f.f_code.co_name!='_assign_episode_roles_and_weights':f=f.f_back
                        for row in value['episode_roles']:row['episode_id']=f.f_locals['simple_to_real'][row['episode_id']]
                    text=json.dumps(value)
                return Response(provider_response(text))
            stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')))
            stack.enter_context(patch.object(requests.Session,'request',side_effect=AssertionError('No network')))
            stack.enter_context(patch.object(requests.Session,'post',post))
            outcomes=[]
            for ci,item in enumerate(dataset):
                result=controller.conv_worker(new,'hypermem',ci,[qs[ci]],item)
                self.assertEqual(result['status'],'terminal',result);outcomes+=result['results']
            failed=[r for r in outcomes if r['judge']=='failed'];self.assertEqual(len(failed),1)
            save_new(new/'qa_recovery_plan.json',{'jobs':failed})
            q=qs[0];ret=read(new/f'results/c0/retrieval/{q["query_id"]}.json')['value'];reader=read(new/'results/firstpass'/q['query_id']/'reader.json')
            recovered=controller.qa_one(new,'hypermem',q,ret,phase='recovery',inherited_reader=reader)
            save_new(new/'qa_recovery_outcome.json',{'results':[recovered]})
            result=audit.audit_all(new,proof)
            self.assertEqual(result['status'],'passed');self.assertEqual(len(result['prefix']),10)
            self.assertEqual(result['ancestor_usage']['physical_attempts'],10)
            self.assertEqual(sum(x[0]==READER_STAGE and x[1]['phase']=='recovery' for x in posts),0)
            summary=controller.seal('hypermem')
            self.assertEqual(summary['effective']['reader_valid'],10);self.assertEqual(summary['effective']['judge_valid'],10)
            self.assertEqual(summary['usage']['combined_physical']['physical_attempts'],
                result['ancestor_usage']['physical_attempts']+result['parent_usage']['physical_attempts']+result['new_usage']['physical_attempts'])
            # Every phase is covered, including firstpass and recovery admissions.
            j=new/'work/prefix_v3/c0/requests.jsonl';saved=j.read_bytes();j.write_bytes(saved+saved.splitlines(keepends=True)[-1])
            with self.assertRaisesRegex(ValueError,'Admission/logical count'):audit.audit_all(new,proof)
            j.write_bytes(saved)
            p=new/'results/recovery'/qs[0]['query_id']/'reader.json';saved=p.read_bytes();v=read(p);v.pop('inherited');p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'Successful original Reader'):audit.audit_all(new,proof)
            p.write_bytes(saved)


if __name__=='__main__':unittest.main()
