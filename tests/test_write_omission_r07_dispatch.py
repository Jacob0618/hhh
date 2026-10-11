"""No-network regression tests for R07 phase and raw provenance boundaries."""
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

from jev_mem.write_omission_r07 import build_proposer_request, build_request, parse_proposals, stable_candidates
from tools import score_write_omission_r07 as s


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def state():
    return {'session_id':'session_1','observation_date':'2023-01-01','current_source':[
        {'turn_id':'D1:1','speaker':'Amy','session_id':'session_1','observation_date':'2023-01-01','text':'Amy adopted a dog.'}],
        'historical_source':[],'baseline_prefix':[]}

def candidates():
    raw='{"memories":[{"statement":"Amy adopted a dog.","evidence":[{"turn_id":"D1:1","field":"text","quote":"Amy adopted a dog."}]}]}'
    return stable_candidates(parse_proposals(raw,state(),s.spec()),'c0_s001',state())


def generation(text='{"memories":[]}'):
    return {'modelVersion':'gemini-3.8-flash','candidates':[{'finishReason':'STOP',
        'content':{'parts':[{'text':text}]}}]}


class Fixture:
    def __init__(self, run, phase='proposer', answer=None, statuses=(200,)):
        self.run, self.phase = run, phase
        sig = (build_proposer_request(state(),s.spec()) if phase=='proposer'
               else build_request(state(),candidates(),s.spec()))
        qid='c0_s001';rid=qid
        paths=s.phase_paths(run,phase)
        self.rec={'request_id':rid,'query_id':qid,'conversation':0,'session_id':'session_1','state':state(),'protocol_sha256':'runtime-test','signature':sig,
            'request_hash':s.request_hash(sig),'signature_utf8_bytes':len(s.serialized_signature(sig)),
            'dispatch_eligible':True}
        if phase=='jev':self.rec.update(candidates=candidates())
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
        return patch.dict('sys.modules',{'tools.prepare_write_omission_r07':
            SimpleNamespace(validate_prepared=lambda *a,**k: ({},refs))})


class DispatchTests(unittest.TestCase):
    def test_proposer_gate_exact_no_network(self):
        sig=build_proposer_request(state(),s.spec())
        api=object.__new__(s.GateAPI)
        with patch.object(s.PacedEmbeddingAPI,'request',return_value={'mock':True}) as p:
            self.assertEqual(api.request(sig['provider'],sig['model'],sig['body'],stage=s.PROPOSER_STAGE),{'mock':True})
            p.assert_called_once()

    def test_proposer_no_json_mime_or_model_change(self):
        sig=build_proposer_request(state(),s.spec());api=object.__new__(s.GateAPI)
        for modify in ('model','mime','extra'):
            a=copy.deepcopy(sig)
            if modify=='model':a['model']='gemini-2.5-flash'
            if modify=='mime':a['body']['generationConfig']['responseMimeType']='application/json'
            if modify=='extra':a['body']['systemInstruction']={'parts':[{'text':'x'}]}
            with self.assertRaises(ValueError):api.request(a['provider'],a['model'],a['body'],stage=s.PROPOSER_STAGE)

    def test_proposer_forbidden_payload_rejected(self):
        sig=build_proposer_request(state(),s.spec());api=object.__new__(s.GateAPI)
        text=sig['body']['contents'][0]['parts'][0]['text'];prefix=text.split('INPUT_JSON:\n')[0]+'INPUT_JSON:\n'
        sig['body']['contents'][0]['parts'][0]['text']=prefix+json.dumps({'question':'What?','reference_date':'','gold':'answer'})
        with self.assertRaises(ValueError):api.request('gemini','gemini-3.8-flash',sig['body'],stage=s.PROPOSER_STAGE)

    def test_jev_key_and_primitive_guard(self):
        sig=build_request(state(),candidates(),s.spec())
        self.assertEqual(len(s.expected_keys(sig['body'])),2)
        sig['body']['questions']['c00_support']['type']='choice'
        with self.assertRaises(ValueError):s.expected_keys(sig['body'])

    def test_jev_global_inflight_bound(self):
        sig=build_request(state(),candidates(),s.spec());api=object.__new__(s.GateAPI)
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
            with self.assertRaises(ValueError):s.validate_record(fx.rec,fx.rp,'proposer')

    def test_success_proposer_and_duplicate_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));api=fx.fake_api();row=s.one(fx.run,fx.rp,api,'proposer')
            self.assertEqual(row['status'],'complete');self.assertEqual(row['proposal_parse']['retained_count'],0)
            with self.assertRaises(FileExistsError):s.one(fx.run,fx.rp,api,'proposer')
            self.assertEqual(api.calls,1)

    def test_malformed200_preserved_no_repair(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),answer=generation('```json\n{}\n```'))
            row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            self.assertEqual((row['status'],row['reason']),('failed','response_validation_failed'))
            self.assertIsNone(row['proposal_parse']);self.assertTrue(fx.rawp.exists())

    def test_absent_resolved_model_is_recorded_unavailable(self):
        with tempfile.TemporaryDirectory() as t:
            answer=generation();del answer['modelVersion']
            fx=Fixture(Path(t),answer=answer);row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            self.assertEqual(row['status'],'complete');self.assertIsNone(row['resolved_model_version'])
            with fx.terminal(row):self.assertEqual(s.audit_phase(fx.run,'proposer',fx.run/'audit.json')['status'],'passed')

    def test_dedup_keeps_second_original_index_but_axis_zero(self):
        from jev_mem.write_omission_r07 import admit_candidates
        st=state();st['baseline_prefix']=[{'id':'old','text':'Already known.','observation_date':'2023-01-01'}]
        ev=[{'turn_id':'D1:1','field':'text','quote':'Amy adopted a dog.'}]
        parsed=parse_proposals(json.dumps({'memories':[{'statement':'Already known.','evidence':ev},{'statement':'Amy adopted a dog.','evidence':ev}]}),st,s.spec())
        cs=stable_candidates(parsed,'c0_s001',st);self.assertEqual(cs[0]['proposal_index'],1)
        sig=build_request(st,cs,s.spec());self.assertEqual(set(sig['body']['questions']),{'c00_support','c00_novelty'})
        accepted=admit_candidates(cs,{'c00_support':.8,'c00_novelty':.5})
        self.assertEqual(accepted['accepted_ids'],['c0_s001_m01'])

    def test_zero_jev_requests_terminal_without_api(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);paths=s.phase_paths(run,'jev');dump(paths['manifest'],{})
            fake=SimpleNamespace(validate_prepared=lambda *a,**k:({},[]))
            with patch.dict('sys.modules',{'tools.prepare_write_omission_r07':fake}),patch.object(s,'GateAPI',side_effect=AssertionError('No credential/model initialization')):
                out=s.run_phase(run,'jev')
            self.assertEqual(out['status'],'terminal');self.assertEqual(out['terminal_requests'],0)

    def test_transport_attempt_context_exact(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),statuses=('transport_error',)*3)
            state,reason,_=s.classify_failure(fx.api,fx.sig,fx.rawp,s.PROPOSER_STAGE)
            self.assertEqual((state,reason),('failed','transport_exhausted_three_attempts'))
            fx.events[-1]['context']={'query_id':'another'};fx.write_events()
            self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.PROPOSER_STAGE)[0],'unknown')

    def test_two_attempts_unknown_not_retry_permission(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),statuses=('transport_error',)*2)
            self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.PROPOSER_STAGE)[0],'unknown')

    def test_retryable_and_nonretryable_http(self):
        for statuses,reason in [((429,429,429),'retryable_http_exhausted'),((401,),'nonretryable_http_terminal')]:
            with self.subTest(statuses=statuses),tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),statuses=statuses)
                self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.PROPOSER_STAGE)[1],reason)

    def test_wrong_raw_signature_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));raw=s.read(fx.rawp);raw['request']['body']={};dump(fx.rawp,raw)
            self.assertEqual(s.classify_failure(fx.api,fx.sig,fx.rawp,s.PROPOSER_STAGE)[0],'unknown')

    def test_audit_raw_replay_pass(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            with fx.terminal(row):result=s.audit_phase(fx.run,'proposer',fx.run/'review.json')
            self.assertEqual(result['status'],'passed');self.assertEqual(result['requests_count'],1)
            self.assertIn(str(fx.run/'request_intents'/(fx.key+'.json')),
                          [r['path'] for r in result['input_refs']])
            self.assertNotIn(str(fx.run/'api_calls.jsonl'),[r['path'] for r in result['input_refs']])
            self.assertEqual(result['ledger_projection']['selected_rows_sha256'],s.digest(fx.events))
            self.assertEqual(result['ledger_projection']['request_hashes'],[fx.key])

    def test_audit_malformed_fallback_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),answer=generation('bad json'));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            with fx.terminal(row):result=s.audit_phase(fx.run,'proposer',fx.run/'review.json')
            self.assertEqual(result['status'],'passed')
            self.assertFalse(result['known_terminal_failures'][0]['technical_recovery_eligible'])

    def test_audit_transport_requires_recovery_review(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),statuses=('transport_error',)*3);row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            with fx.terminal(row):result=s.audit_phase(fx.run,'proposer',fx.run/'review.json')
            self.assertEqual(result['status'],'needs_recovery_review')

    def test_audit_missing_consumer_or_producer_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            fx.events[0]['context']={'other':'query'};fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'proposer',fx.run/'review.json')

    def test_audit_extra_api_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            fx.events.append(dict(fx.events[0],request_hash='unmanifested'));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'proposer',fx.run/'review.json')

    def test_audit_same_hash_rogue_physical_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            fx.events.append(dict(fx.events[0],context={'conversation':9,'query_id':'c9_q9'}));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'proposer',fx.run/'review.json')

    def test_audit_rogue_intent_producer_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            ip=fx.run/'request_intents'/(fx.key+'.json');intent=s.read(ip)
            intent['context']={'query_id':'unprepared'};dump(ip,intent)
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'proposer',fx.run/'review.json')

    def test_audit_same_hash_rogue_stage_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            fx.events.append(dict(fx.events[0],stage='other_stage',cache_hit=True));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'proposer',fx.run/'review.json')

    def test_audit_producer_cannot_reenter_as_cache(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            fx.events.append(dict(fx.events[0],cache_hit=True));fx.write_events()
            with fx.terminal(row),self.assertRaises(ValueError):s.audit_phase(fx.run,'proposer',fx.run/'review.json')

    def test_audit_later_other_phase_is_not_bound_to_projection(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.fake_api(),'proposer')
            original=list(fx.events)
            fx.events.append(dict(fx.events[0],request_hash='other_phase_hash',stage=s.JEV_STAGE));fx.write_events()
            with fx.terminal(row):result=s.audit_phase(fx.run,'proposer',fx.run/'review.json')
            self.assertEqual(result['ledger_projection']['selected_rows_sha256'],s.digest(original))
            self.assertEqual(result['ledger_projection']['selected_row_count'],1)


if __name__=='__main__':unittest.main()
