"""Independent real cache + fake HTTP adversarial prefix checks, temporary workspace only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import requests
from tools import hypermem_prefix_api_v3 as p
from tools import baselines_hypermem_native_v3 as a
from tools.baselines_common_v1 import BaselineFatal,read,save_new
from tests import test_hypermem_prefix_api_v3 as factory_module
response=factory_module.response


class PrefixBoundaries(unittest.TestCase):
    def setUp(self):
        self.f=factory_module.Prefix();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_two_conversations_same_old_key_are_exact_cache_consumers(self):
        first=deepcopy(self.f.proof['conversations'][0]['prefix'][0])
        first['context']['conversation']=1
        self.f.proof['conversations'][1]['prefix'][0]=first
        C=self.f.api_class();barrier=threading.Barrier(2)
        apis=[self.f.make(C,ci) for ci in (0,1)]
        original_save=p.save_new
        def delayed_save(path,value):
            if Path(path).parent.name=='cache_lineage':time.sleep(.08)
            return original_save(path,value)
        def worker(ci):
            barrier.wait(timeout=10)
            return self.f.call(apis[ci],ci)
        with patch.object(p,'save_new',delayed_save),patch.object(requests.Session,'post',side_effect=AssertionError('network')),ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(worker,(0,1)))
        self.assertEqual(values[0],values[1])
        self.assertFalse((self.f.new/'work/api_method/request_intents').exists())
        self.assertEqual(C.admission['progress'][0],1);self.assertEqual(C.admission['progress'][1],1)

    def test_replacement_http_does_not_hold_conversation_lock(self):
        C=self.f.api_class(replacement_enabled=True);api=self.f.make(C,5)
        class Resp:
            status_code=200;ok=True;headers={}
            def json(self):return response('{"title":"ok","summary":"ok","content":"ok"}')
        def probe():
            lock=C.admission['locks'][5]
            acquired=lock.acquire(timeout=.4)
            if acquired:lock.release()
            return acquired
        def post(*args,**kw):
            self.assertTrue((self.f.new/'work/replacement_v3/known_invalid_episode_once.json').exists())
            with ThreadPoolExecutor(max_workers=1) as pool:self.assertTrue(pool.submit(probe).result(timeout=2))
            return Resp()
        with patch.object(requests.Session,'post',post):self.f.call(api,5)

    def test_new_raw_representation_has_distinct_explicit_scope_and_unmodified_raw(self):
        C=self.f.api_class(prospective_role_representation=True);api=self.f.make(C)
        with patch.object(requests.Session,'post',side_effect=AssertionError('prefix network')):self.f.call(api)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,f'fresh-independent-{i}')) for i in range(2)]
        original={'episode_roles':[{'episode_id':x,'role':'initiating','weight':.7,'keep':['exact']} for x in ids], 'coherence_score':.8,'reasoning':'kept'}
        class Resp:
            status_code=200;ok=True;headers={}
            def json(self):return response(json.dumps(original))
        work=self.f.new/'work/native/c0';gen=a._Generation(api,work,'topic')
        async def _assign_episode_roles_and_weights():
            simple_to_real={f'episode_{i+1}':v for i,v in enumerate(ids)}
            episodes=[SimpleNamespace(event_id=v,to_dict=lambda x=v:{'event_id':x}) for v in ids]
            topic=SimpleNamespace(to_dict=lambda:{'topic_id':'fixture'})
            return await gen.generate('brand-new role prompt',response_format={'type':'json_object'})
        coroutine=_assign_episode_roles_and_weights()
        with patch.object(requests.Session,'post',lambda *args,**kw:Resp()):
            try:coroutine.send(None)
            except StopIteration:pass
            finally:coroutine.close()
        call=read(work/'calls/topic_000001_hypermem_episode_roles.json')
        self.assertEqual(call['raw_response'],original)
        self.assertEqual([x['episode_id'] for x in call['parsed_response']['episode_roles']],['episode_1','episode_2'])
        journal=read(work/'representations/topic_000001_hypermem_episode_roles.json')
        self.assertEqual(journal['authorization']['scope'],'first_seen_suffix_representation')
        self.assertFalse(journal['authorization']['model_regeneration_permitted'])
        # The actual consumer's raw binding, not a caller-supplied parsed object,
        # is needed by the authorization method.
        bad=deepcopy(original);bad['reasoning']='tampered'
        with self.assertRaises(BaselineFatal):
            api.authorize_episode_representation({k:v for k,v in journal.items() if k!='authorization'},original=bad,derived=call['parsed_response'])

    def test_same_admitted_class_native_cache_uses_sealed_durable_evidence_without_request(self):
        C=self.f.api_class(prospective_role_representation=True);api=self.f.make(C)
        with patch.object(requests.Session,'post',side_effect=AssertionError('prefix network')):self.f.call(api)
        uid=str(uuid.uuid5(uuid.NAMESPACE_URL,'independent-sealed-cached'))
        original={'episode_roles':[{'episode_id':uid,'role':'developing','weight':.7}]}
        class Resp:
            status_code=200;ok=True;headers={}
            def json(self):return response(json.dumps(original))
        work=self.f.new/'work/native/c0'
        gen=a._Generation(api,work,'topic')
        async def _assign_episode_roles_and_weights():
            simple_to_real={'episode_1':uid}
            episodes=[SimpleNamespace(event_id=uid,to_dict=lambda:{'event_id':uid})]
            topic=SimpleNamespace(to_dict=lambda:{'topic_id':'cache-fixture'})
            return await gen.generate('new cached-role prompt',response_format={'type':'json_object'})
        def run():
            coroutine=_assign_episode_roles_and_weights()
            try:coroutine.send(None)
            except StopIteration as result:return result.value
            finally:coroutine.close()
        with patch.object(requests.Session,'post',lambda *args,**kw:Resp()):first=run()
        records={str(x):x.read_bytes() for x in work.rglob('*') if x.is_file()}
        api=self.f.make(C);self.assertEqual(api.events,[])
        gen=a._Generation(api,work,'topic')
        with patch.object(p.PrefixAPI,'request',side_effect=AssertionError('sealed native cache cannot request')):
            self.assertEqual(run(),first)
        self.assertEqual(records,{str(x):x.read_bytes() for x in work.rglob('*') if x.is_file()})
        # Restarting admission never reconstructs the cursor from these artifacts.
        fresh=self.f.api_class(prospective_role_representation=True)
        self.assertEqual(fresh.admission['progress'][0],0)
        with patch.object(requests.Session,'post',side_effect=AssertionError('network')):
            with self.assertRaises((BaselineFatal,FileExistsError)):self.f.call(self.f.make(fresh))

    def test_qualified_raw_repeated_suffix_preserves_bytes_and_distinguishes_ordinal_scope(self):
        uid=str(uuid.uuid5(uuid.NAMESPACE_URL,'qualified-repeat-independent'))
        mapping={'episode_1':uid}
        original={'episode_roles':[{'episode_id':uid,'role':'developing','weight':.7}]}
        derived,pure=a.represent_episode_ids(original,mapping)
        prompt='qualified original role prompt'
        sig={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
             'body':p.generation_body(prompt,json_output=True,max_output=16384)}
        key=p.digest(sig);raw=self.f.old/'api_cache'/(key+'.json')
        save_new(raw,{'request':sig,'stage':'hypermem_episode_roles','response':response(json.dumps(original))})
        old_call=self.f.old/'native/c0/calls/topic_000001_hypermem_episode_roles.json'
        save_new(old_call,{'request':{'prompt':prompt},'raw_response':original})
        entry=self.f.proof['conversations'][0]['prefix'][0]
        entry.update(signature=sig,request_hash=key,stage='hypermem_episode_roles',
                     raw=p.binding(raw),native_call=p.binding(old_call))
        self.f.proof['producer_registry'][key]={'signature':sig,'raw':p.binding(raw),'stages':['hypermem_episode_roles']}
        self.f.proof['role_cases']=[{'conversation':0,'request_hash':key,'raw':p.binding(raw),
             'native_call':p.binding(old_call),'original_parsed_sha256':pure['original_parsed_sha256'],
             'derived_parsed_sha256':pure['derived_parsed_sha256'],
             'mapping_from_complete_unique_native_display':[{'simple_id':'episode_1','event_id':uid}]}]
        before=raw.read_bytes();C=self.f.api_class();api=self.f.make(C)
        work=self.f.new/'work/native/c0';gen=a._Generation(api,work,'topic')
        async def _assign_episode_roles_and_weights():
            simple_to_real=mapping
            episodes=[SimpleNamespace(event_id=uid,to_dict=lambda:{'event_id':uid})]
            topic=SimpleNamespace(to_dict=lambda:{'topic_id':'qualified-repeat'})
            return await gen.generate(prompt,response_format={'type':'json_object'})
        def one():
            coro=_assign_episode_roles_and_weights()
            try:coro.send(None)
            except StopIteration as result:return result.value
            finally:coro.close()
        with patch.object(requests.Session,'post',side_effect=AssertionError('no physical repeat')):
            self.assertEqual(json.loads(one()),derived);self.assertEqual(json.loads(one()),derived)
        journals=[read(work/'representations'/f'topic_{i:06d}_hypermem_episode_roles.json') for i in (1,2)]
        self.assertEqual([j['authorization']['scope'] for j in journals],
             ['qualified_original_response','qualified_original_response_reused_suffix'])
        self.assertEqual(before,raw.read_bytes())
        self.assertFalse((self.f.new/'work/api_method/request_intents').exists())
        self.assertEqual(C.admission['progress'][0],1)


if __name__=='__main__':unittest.main()
