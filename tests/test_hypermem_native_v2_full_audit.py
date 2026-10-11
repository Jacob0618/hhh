"""Ten tiny real native banks + genuine raw/ledger/QA artifacts; fake HTTP only.

This tests the new audit's success path, not real-cohort performance or launch.
"""
from contextlib import ExitStack
from copy import deepcopy
import inspect
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import requests
from tests.test_baselines_hypermem_v1 import FakeAPI,conversation
from tools import run_five_baselines_hypermem_native_v2 as c
from tools import audit_hypermem_continuation_v2 as audit
from tools.hypermem_prefix_api_v2 import admitted_api,PrefixAPI
from tools.baselines_common_v1 import read,sha,save_new
from jev_mem import providers
from jev_mem.sdk20_cached_api import digest
from jev_mem.gemini25_ablation_provider_v1 import generation_body,READER_STAGE,JUDGE_STAGE


class Capture(BaseException):pass


class CompleteNativeAudit(unittest.TestCase):
    def test_ten_real_native_banks_raw_embeddings_qa_and_recovery_full_audit(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);old=root/'parent';new=root/'continuation';dataset=[];rows=[]
            stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')))
            stack.enter_context(patch.object(socket.socket,'connect',side_effect=AssertionError('No network')))
            # Initialize the original native modules/asyncio under a real local
            # loop before globally blocking socket connects on Windows.
            stack.close()
            c.adapter._runtime._load()
            import asyncio
            loop=asyncio.new_event_loop();loop.close()
            stack.enter_context(patch.object(providers,'load_credentials',side_effect=AssertionError('No vault')))
            stack.enter_context(patch.object(requests.Session,'request',side_effect=AssertionError('No network')))
            actualqs=read(c.OLD/'run_manifest.json')['questions'];qs=[]
            for ci in range(10):
                conv=conversation(4);conv['speaker_a']=f'A{ci}';conv['speaker_b']=f'B{ci}'
                for i,turn in enumerate(conv['session_1']):turn.update(speaker=conv['speaker_a'] if i%2==0 else conv['speaker_b'],text=turn['text']+f' Conversation {ci}.')
                dataset.append({'conversation':conv});work=old/f'capture/c{ci}';record={}
                class First:
                    def generate(self,prompt,**kwargs):record.update(prompt=prompt,kwargs=kwargs);raise Capture()
                with self.assertRaises(c.adapter.HyperMemError):c.adapter.build(conv,work,First())
                sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':generation_body(record['prompt'],json_output=True,max_output=16384)}
                key=digest(sig);parsed={'should_end':False,'reasoning':'fixture original','confidence':1.0,'topic_summary':'Park'}
                response={'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':json.dumps(parsed)}]}}],'modelVersion':'gemini-2.5-flash'}
                raw=old/'api_cache'/(key+'.json');save_new(raw,{'request':sig,'response':response,'stage':'hypermem_boundary'})
                native=work/'calls/episode_000001_hypermem_boundary.json';nr=read(native);nr['raw_response']=parsed
                native.write_text(json.dumps(nr),encoding='utf8')
                rows.append({'conversation':ci,'request_hash':key,'signature':sig,'stage':'hypermem_boundary','context':{'method':'hypermem','conversation':ci,'phase':'build'},
                    'raw':c.binding(raw),'native_call':c.binding(native),'source_messages':c.binding(work/'source_messages.json'),
                    'physical':{'cache_hit':False,'usage':{},'status':200}})
                q=deepcopy(next(q for q in actualqs if q['conversation']==ci));q.update(question=f'Where did A walk, conversation {ci}?',reference='park',category=1);qs.append(q)
            proof={'rows':rows,'registry':{r['request_hash']:r for r in rows},'questions':qs,'original_inventory':c.admission.inventory(old),
                   'original_physical_usage':c.admission.usage_presence([r['physical'] for r in rows])}
            pp=root/'proof.json';rp=root/'review.json';save_new(pp,proof)
            parent=read(c.OLD/'run_manifest.json');codes=c.code_hashes(parent)
            save_new(rp,{'schema':c.REVIEW_SCHEMA,'status':'passed','fixture_only':True,'production_launch_authorized':False,
                'proof_sha256':sha(pp),'specification':parent['specification'],'code_sha256':codes,'upstream_sha256':parent['upstream_sha256'],
                'evidence_sha256':{str(pp):sha(pp)}})
            m={'questions':qs,'code_sha256':codes,'upstream_sha256':parent['upstream_sha256'],'review':c.binding(rp),'terminal_proof':c.binding(pp),'parent_manifest':c.binding(pp)}
            save_new(new/'run_manifest.json',m)
            stack.enter_context(patch.object(c,'NEW',new));stack.enter_context(patch.object(c.admission,'OLD',old))
            stack.enter_context(patch.object(c.admission.chain.base,'load_data',lambda:dataset))
            class Ticket:
                def consume(self,method):raise AssertionError('Not running production controller')
            controller=c.private_controller(m,proof,Ticket());RealAPI=admitted_api(proof,new)
            class LocalAPI(RealAPI):
                def __init__(self,*args,**kwargs):
                    kwargs.setdefault('keys',{'GEMINI_API_KEY':'synthetic'});kwargs.setdefault('gate_path',root/'fixture_gate.sqlite');kwargs.setdefault('interval_seconds',0.00001)
                    super().__init__(*args,**kwargs)
            controller.BaselineAPI=LocalAPI
            fake=FakeAPI();posts=[]
            class Response:
                status_code=200;ok=True;headers={}
                def __init__(self,value):self.value=value
                def json(self):return self.value
            def post(session,url,**kwargs):
                body=kwargs['json'];frame=inspect.currentframe().f_back
                while frame and not isinstance(frame.f_locals.get('self'),PrefixAPI):frame=frame.f_back
                if frame is None:raise AssertionError('Missing real PrefixAPI frame')
                stage=frame.f_locals['stage'];context=frame.f_locals['self'].context
                posts.append((stage,deepcopy(context)))
                if 'batchEmbedContents' in url:
                    values=[]
                    for request in body['requests']:
                        text=request['content']['parts'][0]['text'];v=[0.0]*768;v[0]=1.0;v[1]=len(text)/1000;v[2]=sum(map(ord,text))%41/100;values.append({'values':v})
                    return Response({'embeddings':values})
                if 'countTokens' in url:return Response({'totalTokens':100})
                prompt=body['contents'][0]['parts'][0]['text']
                if stage==READER_STAGE:text='park'
                elif stage==JUDGE_STAGE:text='invalid terminal label' if context['conversation']==0 and context['phase']=='firstpass' else '[[yes]]'
                else:text=json.dumps(fake.generate(prompt,stage=stage))
                return Response({'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':text}]}}],'modelVersion':'gemini-2.5-flash'})
            stack.enter_context(patch.object(requests.Session,'post',post))
            outcomes=[]
            for ci,item in enumerate(dataset):
                outcome=controller.conv_worker(new,'hypermem',ci,[qs[ci]],item)
                self.assertEqual(outcome['status'],'terminal',outcome);outcomes.extend(outcome['results'])
            failed=[r for r in outcomes if r['judge']=='failed'];self.assertEqual(len(failed),1)
            save_new(new/'qa_recovery_plan.json',{'jobs':failed})
            q=qs[0];ret=read(new/f'results/c0/retrieval/{q["query_id"]}.json')['value'];reader=read(new/'results/firstpass'/q['query_id']/'reader.json')
            recovered=controller.qa_one(new,'hypermem',q,ret,phase='recovery',inherited_reader=reader)
            save_new(new/'qa_recovery_outcome.json',{'results':[recovered]})
            value=audit.audit_all(new,proof)
            self.assertEqual(value['status'],'passed');self.assertEqual(value['native']['prefix_conversations'],10)
            self.assertEqual(len(value['native']['compatibility']),10)
            self.assertGreater(value['native']['generation_consumers'],10);self.assertGreater(value['native']['embedding_consumers'],10)
            for q in qs:
                for phase in ('firstpass','recovery') if q==qs[0] else ('firstpass',):controller.replay_qa_leaf(new,'hypermem',q,new/'results'/phase/q['query_id'])
            summary=controller.seal('hypermem')
            self.assertEqual(summary['effective']['reader_valid'],10);self.assertEqual(summary['effective']['judge_valid'],10)
            self.assertEqual(summary['usage']['parent_original']['physical_attempts'],10)
            self.assertEqual(summary['usage']['combined_physical']['physical_attempts'],summary['usage']['continuation']['physical_attempts']+10)
            # Reverse inheritance and coverage negatives use real completed artifacts.
            path=new/'results/recovery'/qs[0]['query_id']/'reader.json';saved=path.read_bytes();r=read(path);r.pop('inherited');path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Successful original Reader'):audit.audit_all(new,proof)
            path.write_bytes(saved)
            path=new/'qa_recovery_plan.json';saved=path.read_bytes();path.write_text('{"jobs":[]}')
            with self.assertRaises(ValueError):audit.audit_all(new,proof)
            path.write_bytes(saved)
            self.assertEqual(proof['original_inventory'],c.admission.inventory(old))
            self.assertEqual(sum(stage==READER_STAGE and ctx['phase']=='recovery' for stage,ctx in posts),0)

if __name__=='__main__':unittest.main()
