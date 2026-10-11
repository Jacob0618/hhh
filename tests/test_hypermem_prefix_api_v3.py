"""Actual frozen cache/transport with synthetic bytes and no network."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import requests
from tools import hypermem_prefix_api_v3 as p
from tools.baselines_common_v1 import BaselineFatal, save_new, read, sha
from jev_mem import providers

ROOT = Path(__file__).resolve().parents[1]


def response(text):
    return {'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[{'text':text}]}}],
        'modelVersion':'gemini-2.5-flash','usageMetadata':{'promptTokenCount':7,'candidatesTokenCount':3,'totalTokenCount':10}}


class Prefix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'research'); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name); self.old = self.folder/'old'; self.new = self.folder/'new'
        self.proof = {'schema':p.PROOF_SCHEMA,'status':'passed','producer_registry':{},'conversations':[],
            'role_cases':[],'role_qualification':{'path':'synthetic-classification','sha256':'a'*64},
            'questions':[{'query_id':f'c{ci}_q0','conversation':ci} for ci in range(10)],'replacement':None}
        for ci in range(10):
            stage = 'hypermem_episode' if ci == 5 else 'hypermem_boundary'
            sig = {'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent',
                'body':p.generation_body(f'prefix-{ci}',json_output=True,max_output=16384)}
            key = p.digest(sig); raw = self.old/'api_cache'/(key+'.json')
            save_new(raw,{'request':sig,'stage':stage,'response':response('{"bad":' if ci == 5 else '{"should_end":false}')})
            call = self.old/f'native/c{ci}/calls/episode_000001_{stage}.json'; save_new(call,{'fixture_only':True})
            context = {'method':'hypermem','conversation':ci,'phase':'build'}
            entry = {'ordinal':1,'context':context,'signature':sig,'request_hash':key,'stage':stage,
                'raw':p.binding(raw),'native_call':p.binding(call),'mode':'replace_once' if ci == 5 else 'inherit'}
            self.proof['conversations'].append({'conversation':ci,'prefix':[entry]})
            self.proof['producer_registry'][key] = {'signature':sig,'raw':p.binding(raw),'stages':[stage]}
            if ci == 5:
                self.proof['replacement'] = {'kind':'known_complete_invalid_episode_json','retry_limit':1,
                    'conversation':ci,'request_hash':key,'signature':sig,'context':context,
                    'stage':stage,'raw':p.binding(raw),'prefix_ordinal':1}
        self.manifest = save_new(self.new/'run_manifest.json',{'fixture_only':True})
        self.proof_binding = save_new(self.folder/'proof.json',self.proof)
        self.vault = patch.object(providers,'load_credentials',side_effect=AssertionError('credentials prohibited'))
        self.vault.start(); self.addCleanup(self.vault.stop)

    def api_class(self, **kwargs):
        return p.admitted_api(self.proof,self.new,manifest=self.manifest,proof_binding=self.proof_binding,**kwargs)

    def make(self, C, ci=0, **kwargs):
        return C(self.new/'work/api_method',context={'method':'hypermem','conversation':ci,'phase':'build'},
            keys={'GEMINI_API_KEY':'synthetic-no-access'},**kwargs)

    def call(self, api, ci=0):
        e = self.proof['conversations'][ci]['prefix'][0]; s=e['signature']
        return api.request(s['provider'],s['model'],s['body'],operation=s['operation'],stage=e['stage'])

    def test_ordered_repeated_consumers_and_global_seen_key_never_produce(self):
        first = self.proof['conversations'][0]['prefix'][0]
        second = deepcopy(first); second['ordinal'] = 2
        self.proof['conversations'][0]['prefix'].append(second)
        C = self.api_class(); api=self.make(C)
        with patch.object(requests.Session,'post',side_effect=AssertionError('network')):
            self.call(api); self.call(api)
            # Cross-conversation known signature is still inherited, not redrawn.
            self.call(api,1)
        self.assertEqual(C.admission['progress'][0],2)
        self.assertFalse((self.new/'work/api_method/request_intents').exists())
        calls=[json.loads(x) for x in (self.new/'work/api_method/api_calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls),3); self.assertTrue(all(x['cache_hit'] for x in calls))
        self.assertEqual(read(self.new/'work/prefix_v3/c0/complete.json')['prefix_count'],2)

    def test_changed_order_and_context_rejected_before_provider(self):
        C=self.api_class(); api=self.make(C)
        with patch.object(requests.Session,'post',side_effect=AssertionError('network')):
            with self.assertRaises(BaselineFatal):self.call(api,1)
            with self.assertRaises(BaselineFatal):self.make(C,retries=2.0)
            with self.assertRaises(BaselineFatal):
                C(self.new/'work/api_method',context={'method':'hypermem','conversation':False,'phase':'build'})
        self.assertFalse((self.new/'work/prefix_v3').exists())

    def test_replacement_default_disabled_and_exact_once_marker_precedes_send(self):
        C=self.api_class(); api=self.make(C,5)
        with patch.object(requests.Session,'post',side_effect=AssertionError('network')):
            with self.assertRaises(BaselineFatal):self.call(api,5)
        self.assertFalse((self.new/'work/replacement_v3').exists())

    def test_enabled_replacement_bad_result_cannot_be_sent_again(self):
        C=self.api_class(replacement_enabled=True); api=self.make(C,5); sends=[]
        marker=self.new/'work/replacement_v3/known_invalid_episode_once.json'
        class Resp:
            status_code=200;ok=True;headers={}
            def json(self):return response('{"still_bad":')
        def fake_post(*args,**kwargs):
            self.assertTrue(marker.exists()); sends.append(1); return Resp()
        e=self.proof['conversations'][5]['prefix'][0]
        with patch.object(requests.Session,'post',fake_post):
            with self.assertRaises(BaselineFatal):
                api.generate('prefix-5',json_output=True,stage='hypermem_episode')
            with self.assertRaises(BaselineFatal):self.call(self.make(C,5),5)
        self.assertEqual(len(sends),1)
        self.assertEqual(sha(e['raw']['path']),e['raw']['sha256'])
        self.assertTrue((self.new/'work/api_method/api_cache'/(e['request_hash']+'.json')).exists())
        self.assertFalse((self.new/'work/api_method/cache_lineage'/(e['request_hash']+'.json')).exists())

    def test_successful_replacement_reuses_new_response_not_old_bad_raw(self):
        C=self.api_class(replacement_enabled=True); api=self.make(C,5); sends=[]
        class Resp:
            status_code=200;ok=True;headers={}
            def json(self):return response('{"title":"ok","summary":"ok","content":"ok"}')
        def fake_post(*args,**kwargs):sends.append(1);return Resp()
        with patch.object(requests.Session,'post',fake_post):
            first=self.call(api,5);second=self.call(self.make(C,5),5)
        self.assertEqual(first,second);self.assertEqual(len(sends),1)

    def test_two_unseen_suffix_requests_are_concurrent_and_unknown_not_redispatched(self):
        C=self.api_class();api=self.make(C)
        with patch.object(requests.Session,'post',side_effect=AssertionError('network')):self.call(api)
        barrier=threading.Barrier(2); sends=[]
        class Resp:
            status_code=200;ok=True;headers={}
            def json(self):return response('{"should_end":false}')
        def post(*args,**kwargs):sends.append(1);barrier.wait(timeout=3);return Resp()
        def one(i):return self.make(C).generate(f'suffix-{i}',json_output=True,stage='hypermem_boundary')
        with patch.object(requests.Session,'post',post),ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(len(list(pool.map(one,range(2)))),2)
        self.assertEqual(len(sends),2)
        failures=[]
        def unknown(*args,**kwargs):failures.append(1);raise RuntimeError('synthetic unknown')
        with patch.object(requests.Session,'post',unknown):
            for _ in range(2):
                with self.assertRaises(BaselineFatal):self.make(C).generate('unknown',json_output=True,stage='hypermem_boundary')
            with self.assertRaises(BaselineFatal):self.make(C).generate('wrongstage',json_output=True,stage='amem_analysis')
        self.assertEqual(len(failures),1)


if __name__=='__main__':unittest.main()
