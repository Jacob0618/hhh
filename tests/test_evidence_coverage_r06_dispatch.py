"""No-network regression tests for R06 phase and raw provenance boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev_mem.evidence_coverage_r06 import build_facet_request, build_request
from tools import score_evidence_coverage_r06 as s


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def records():
    return [{'id':str(i),'text':f'Person bought item {i}.','observation_date':'2023-01-01'} for i in range(20)]


def generation(text='{"subrequirements":[]}'):
    return {'modelVersion':'gemini-3.8-flash','candidates':[{'finishReason':'STOP',
        'content':{'parts':[{'text':text}]}}]}


class Fixture:
    def __init__(self, run, phase='facet', answer=None, statuses=(200,)):
        self.run, self.phase = run, phase
        sig = (build_facet_request('What item?', '2023-01-01', s.spec()) if phase=='facet'
               else build_request('What item?', '2023-01-01','What item?', records(),s.spec()))
        qid='c0_q0';rid=qid if phase=='facet' else qid+'_f00'
        paths=s.phase_paths(run,phase)
        self.rec={'request_id':rid,'query_id':qid,'conversation':0,'question':'What item?',
            'reference_date':'2023-01-01','protocol_sha256':'runtime-test','signature':sig,
            'request_hash':s.request_hash(sig),'signature_utf8_bytes':len(s.serialized_signature(sig)),
            'dispatch_eligible':True}
        if phase=='jev':self.rec.update(facet_index=0,logical_noul_count=20)
        self.rp=paths['requests']/(rid+'.json');dump(self.rp,self.rec)
        self.key=self.rec['request_hash'];self.sig=sig;self.paths=paths
        self.context=s.request_context(self.rec,phase)
        self.rawp=run/'api_cache'/(self.key+'.json')
        intent={'request_hash':self.key,'signature':sig,'context':self.context,
                'stage':paths['stage'],'retries':2}
        dump(run/'request_intents'/(self.key+'.json'),intent)
        self.events=[dict(request_hash=self.key,context=self.context,stage=paths['stage'],
            provider=sig['provider'],model=sig['model'],operation=sig['operation'],
            cache_hit=False,attempt=i+1,status=status,timestamp_utc=str(i)) for i,status in enumerate(statuses)]
        self.write_events()
        if statuses[-1]==200:
            dump(self.rawp,{'request':sig,'response':answer if answer is not None else generation()})
        self.api=SimpleNamespace(run_dir=run,context=self.context)

    def write_events(self):
        self.run.mkdir(parents=True,exist_ok=True)
        (self.run/'api_calls.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events),encoding='utf-8')

    def fake_api(self):
        fx=self
        class Fake:
            run_dir=fx.run
            cache_dir=fx.run/'api_cache'
            context=fx.context
            calls=0
            def scoped(self,**context):
                assert context==fx.context
                return self
            def request(self,*args,**kwargs):
                self.calls+=1
                if not fx.rawp.exists():raise RuntimeError('mock transport failure')
                return s.read(fx.rawp)['response']
        return Fake()

    def terminal(self, row):
        refs=[{'path':str(self.rp.resolve()),'sha256':s.sha(self.rp)}]
        dump(self.paths['manifest'],{'schema':'mock manifest'})
        s.publish(self.paths['outcome'],{'status':'terminal','local_errors':[],
            'expected_requests':1,'terminal_requests':1,'counts':{row['status']:1}})
        dump(self.paths['intent'],{'phase':self.phase,'request_files':refs,
            'manifest_sha256':s.sha(self.paths['manifest']),'dispatch_code_sha256':s.sha(s.__file__),
            'transport_retries':2})
        return patch.dict('sys.modules',{'tools.prepare_evidence_coverage_r06':
            SimpleNamespace(validate_prepared=lambda *a,**k: ({},refs))})


class DispatchTests(unittest.TestCase):
    def test_facet_gate_exact_no_network(self):
        sig=build_facet_request('What?',None,s.spec())
        api=object.__new__(s.GateAPI)
        with patch.object(s.PacedEmbeddingAPI,'request',return_value={'mock':True}) as p:
            self.assertEqual(api.request(sig['provider'],sig['model'],sig['body'],stage=s.FACET_STAGE),{'mock':True})
            p.assert_called_once()

    def test_facet_no_json_mime_or_model_change(self):
        sig=build_facet_request('What?',None,s.spec());api=object.__new__(s.GateAPI)
        for modify in ('model','mime','extra'):
            a=copy.deepcopy(sig)
            if modify=='model':a['model']='gemini-2.5-flash'
            if modify=='mime':a['body']['generationConfig']['responseMimeType']='application/json'
            if modify=='extra':a['body']['systemInstruction']={'parts':[{'text':'x'}]}
            with self.assertRaises(ValueError):api.request(a['provider'],a['model'],a['body'],stage=s.FACET_STAGE)

    def test_facet_forbidden_payload_rejected(self):
        sig=build_facet_request('What?',None,s.spec());api=object.__new__(s.GateAPI)
        text=sig['body']['contents'][0]['parts'][0]['text'];prefix=text.split('INPUT_JSON:\n')[0]+'INPUT_JSON:\n'
        sig['body']['contents'][0]['parts'][0]['text']=prefix+json.dumps({'question':'What?','reference_date':'','gold':'answer'})
        with self.assertRaises(ValueError):api.request('gemini','gemini-3.8-flash',sig['body'],stage=s.FACET_STAGE)

    def test_jev_key_and_primitive_guard(self):
        sig=build_request('Q','', 'Q',records(),s.spec())
        self.assertEqual(len(s.expected_keys(sig['body'])),20)
        sig['body']['questions']['f00']['type']='choice'
        with self.assertRaises(ValueError):s.expected_keys(sig['body'])

    def test_jev_global_inflight_bound(self):
        sig=build_request('Q','', 'Q',records(),s.spec());api=object.__new__(s.GateAPI)
        lock=threading.Lock();active=0;peak=0
        def request(*a,**k):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.01)
            with lock:active-=1
            return {}
        with patch.object(s.Gemini38API,'request',side_effect=request):
            with ThreadPoolExecutor(max_workers=32) as pool:
                list(pool.map(lambda _:api.request('jev','jev-1.13.0',sig['body'],operation='systemone',stage=s.JEV_STAGE),range(32)))
        self.assertLessEqual(peak,15);self.assertGreater(peak,1)

    def test_query_identity_not_filename_inference(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));fx.rec['conversation']=2
            with self.assertRaises(ValueError):s.validate_record(fx.rec,fx.rp,'facet')

    def test_success_facet_and_duplicate_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));api=fx.fake_api();row=s.one(fx.run,fx.rp,api,'facet')
            self.assertEqual(row['status'],'complete');self.assertEqual(row['facet_parse']['m'],1)
            with self.assertRaises(FileExistsError):s.one(fx.run,fx.rp,api,'facet')
            self.assertEqual(api.calls,1)

    def test_malformed200_preserved_no_repair(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),answer=generation('```json\n{}\n```'))
            row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            self.assertEqual((row['status'],row['reason']),('failed','response_validation_failed'))
            self.assertIsNone(row['facet_parse']);self.assertTrue(fx.rawp.exists())

    def test_transport_attempt_context_exact(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),statuses=('transport_error',)*3)
            state,reason,_=s.classify_failure(fx.api,fx.sig,fx.rawp,s.FACET_STAGE)
            self.assertEqual((state,reason),('failed','transport_exhausted_three_attempts'))
            fx.events[-1]['context']={'query_id':'another'};fx.write_events()
            self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.FACET_STAGE)[0],'unknown')

    def test_two_attempts_unknown_not_retry_permission(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),statuses=('transport_error',)*2)
            self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.FACET_STAGE)[0],'unknown')

    def test_retryable_and_nonretryable_http(self):
        for statuses,reason in [((429,429,429),'retryable_http_exhausted'),((401,),'nonretryable_http_terminal')]:
            with self.subTest(statuses=statuses),tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),statuses=statuses)
                self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.FACET_STAGE)[1],reason)

    def test_wrong_raw_signature_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));raw=s.read(fx.rawp);raw['request']['body']={};dump(fx.rawp,raw)
            self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.FACET_STAGE)[0],'unknown')

    def test_audit_raw_replay_pass(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            with fx.terminal(row):result=s.audit_phase(fx.run,'facet',fx.run/'review.json')
            self.assertEqual(result['status'],'passed');self.assertEqual(result['requests_count'],1)
            self.assertIn(str(fx.run/'request_intents'/(fx.key+'.json')),
                          [r['path'] for r in result['input_refs']])
            self.assertNotIn(str(fx.run/'api_calls.jsonl'),[r['path'] for r in result['input_refs']])
            self.assertEqual(result['ledger_projection']['selected_rows_sha256'],s.digest(fx.events))
            self.assertEqual(result['ledger_projection']['request_hashes'],[fx.key])

    def test_audit_malformed_fallback_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),answer=generation('bad json'));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            with fx.terminal(row):result=s.audit_phase(fx.run,'facet',fx.run/'review.json')
            self.assertEqual(result['status'],'passed')
            self.assertFalse(result['known_terminal_failures'][0]['technical_recovery_eligible'])

    def test_audit_transport_requires_recovery_review(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),statuses=('transport_error',)*3);row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            with fx.terminal(row):result=s.audit_phase(fx.run,'facet',fx.run/'review.json')
            self.assertEqual(result['status'],'needs_recovery_review')

    def test_audit_missing_consumer_or_producer_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            fx.events[0]['context']={'other':'query'};fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'facet',fx.run/'review.json')

    def test_audit_extra_api_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            fx.events.append(dict(fx.events[0],request_hash='unmanifested'));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'facet',fx.run/'review.json')

    def test_audit_same_hash_rogue_physical_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            fx.events.append(dict(fx.events[0],context={'conversation':9,'query_id':'c9_q9'}));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'facet',fx.run/'review.json')

    def test_audit_rogue_intent_producer_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            ip=fx.run/'request_intents'/(fx.key+'.json');intent=s.read(ip)
            intent['context']={'query_id':'unprepared'};dump(ip,intent)
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'facet',fx.run/'review.json')

    def test_audit_same_hash_rogue_stage_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            fx.events.append(dict(fx.events[0],stage='other_stage',cache_hit=True));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'facet',fx.run/'review.json')

    def test_audit_producer_cannot_reenter_as_cache(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            fx.events.append(dict(fx.events[0],cache_hit=True));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'facet',fx.run/'review.json')

    def test_audit_later_other_phase_is_not_bound_to_projection(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'facet')
            original=list(fx.events)
            fx.events.append(dict(fx.events[0],request_hash='other_phase_hash',stage=s.JEV_STAGE));fx.write_events()
            with fx.terminal(row):result=s.audit_phase(fx.run,'facet',fx.run/'review.json')
            self.assertEqual(result['ledger_projection']['selected_rows_sha256'],s.digest(original))
            self.assertEqual(result['ledger_projection']['selected_row_count'],1)


if __name__=='__main__':unittest.main()
