"""Independent R08 durable dispatch fixtures; no provider client or network."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev_mem import facet_discovery_r08 as core
from tools import score_facet_discovery_r08 as s


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def embedding():
    return {'embeddings':[{'values':[3.,4.]+[0.]*766}, {'values':[0.,1.]+[0.]*766}]}


class Fixture:
    def __init__(self, run, phase='embedding', response=None, statuses=(200,)):
        self.run, self.phase = run, phase
        self.paths = s.phase_paths(run, phase)
        rid = 'c0_b000' if phase == 'embedding' else 'c0_q0'
        facts = [{'id':str(i), 'text':f'Person bought item {i}.', 'observation_date':'2023-01-01'} for i in range(80)]
        r03 = s.read(s.ROOT/'research/jev_autoresearch_2026-10-04/R03_selection_protocol_v2.json')
        sig = core.embedding_signature(['Which purchase?', 'What month?']) if phase == 'embedding' else core.build_request('Which item?', '2023-01-01', facts, r03)
        self.rec = {'request_id':rid, 'query_id':rid, 'conversation':0,
            'protocol_sha256':s.SPEC_SHA, 'signature':sig, 'request_hash':s.request_hash(sig),
            'signature_utf8_bytes':len(s.serialized_signature(sig)), 'signature_utf8_byte_cap':s.MAX_BYTES,
            'dispatch_eligible':True, 'max_physical_attempts_per_phase':3}
        if phase == 'embedding':
            self.rec['batch'] = {'request_id':rid, 'conversation':0, 'texts':['Which purchase?', 'What month?'], 'consumers':[]}
        else:
            self.rec.update(logical_noul_count=80, ordered_fact_ids=[str(i) for i in range(80)], score_cache=None)
        self.rp = self.paths['requests']/(rid+'.json'); dump(self.rp,self.rec)
        self.sig, self.key = sig, self.rec['request_hash']
        self.context = s.request_context(self.rec,phase)
        self.rawp = run/'api_cache'/(self.key+'.json')
        self.ip = run/'request_intents'/(self.key+'.json')
        dump(self.ip, {'request_hash':self.key,'signature':sig,'context':self.context,'stage':self.paths['stage'],'retries':2})
        self.events = [dict(request_hash=self.key, context=self.context, stage=self.paths['stage'],
            provider=sig['provider'], model=sig['model'], operation=sig['operation'], cache_hit=False,
            attempt=i+1, status=status, timestamp_utc=str(i)) for i,status in enumerate(statuses)]
        self.write_events()
        if statuses[-1] == 200:
            if response is None:
                response = embedding() if phase == 'embedding' else {'model':'jev-1.13.0', 'answers':{f'f{i:02d}':{'type':'noul','noul':i/100} for i in range(80)}}
            dump(self.rawp, {'request':sig, 'response':response})

    def write_events(self):
        (self.run/'api_calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in self.events),encoding='utf-8')

    def api(self):
        fx = self
        class Fake:
            run_dir = fx.run
            cache_dir = fx.run/'api_cache'
            context = fx.context
            calls = 0
            def scoped(self, **context):
                if context != fx.context: raise AssertionError('Wrong actual consumer context')
                return self
            def request(self, *args, **kwargs):
                self.calls += 1
                if not fx.rawp.exists(): raise RuntimeError('Synthetic transport failure')
                return s.read(fx.rawp)['response']
            def embed(self, texts, *, query, stage):
                if texts != fx.rec['batch']['texts'] or query is not True or stage != s.EMBEDDING_STAGE:
                    raise AssertionError('Embedding changed task/body/stage')
                return core.parse_embeddings(self.request(), len(texts))
        return Fake()

    def terminal(self, row):
        refs = [{'path':str(self.rp.resolve()), 'sha256':s.sha(self.rp)}]
        dump(self.paths['manifest'], {'schema':'synthetic-bound-manifest'})
        s.publish(self.paths['outcome'], {'status':'terminal','local_errors':[], 'expected_requests':1,
            'terminal_requests':1,'counts':{row['status']:1}})
        dump(self.paths['intent'], {'phase':self.phase,'request_files':refs,
            'manifest_sha256':s.sha(self.paths['manifest']), 'dispatch_code_sha256':s.sha(s.__file__), 'transport_retries':2})
        return refs


class DispatchTests(unittest.TestCase):
    def test_query_embedding_gate_uses_exact_original_body(self):
        sig = core.embedding_signature(['A', 'B']); api = object.__new__(s.GateAPI)
        with patch.object(s.PacedEmbeddingAPI,'request',return_value={'offline':True}) as call:
            self.assertEqual(api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage=s.EMBEDDING_STAGE), {'offline':True})
            call.assert_called_once()
        for key,value in [('taskType','RETRIEVAL_DOCUMENT'),('outputDimensionality',512)]:
            changed = copy.deepcopy(sig); changed['body']['requests'][0][key] = value
            with self.assertRaises(ValueError): api.request(changed['provider'],changed['model'],changed['body'],operation=changed['operation'],stage=s.EMBEDDING_STAGE)

    def test_success_embedding_has_exact_normalized_raw_and_no_duplicate_dispatch(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t)); api = fx.api(); row = s.one(fx.run,fx.rp,api,fx.phase)
            self.assertEqual(row['vectors'][0][:2],[.6,.8]); self.assertEqual(row['status'],'complete')
            audit = s.audit_replay(fx.run,fx.phase,fx.terminal(row)); self.assertEqual(audit['status'],'passed')
            with self.assertRaises(FileExistsError): s.one(fx.run,fx.rp,api,fx.phase)
            self.assertEqual(api.calls,1)

    def test_complete_jev_replays_all80_raw_values(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t),'jev'); row = s.one(fx.run,fx.rp,fx.api(),fx.phase)
            self.assertEqual(row['scores']['f79'],.79); self.assertIsNone(row['vectors'])
            self.assertEqual(s.audit_replay(fx.run,fx.phase,fx.terminal(row))['status'],'passed')

    def test_malformed200_batch_not_partial_or_retryable(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t),response={'embeddings':embedding()['embeddings'][:1]})
            row = s.one(fx.run,fx.rp,fx.api(),fx.phase)
            self.assertEqual((row['status'],row['reason']),('failed','response_validation_failed'))
            self.assertIsNone(row['vectors']); self.assertTrue(fx.rawp.exists())
            audit = s.audit_replay(fx.run,fx.phase,fx.terminal(row))
            self.assertEqual(audit['status'],'passed'); self.assertFalse(audit['known_terminal_failures'][0]['technical_recovery_eligible'])

    def test_exhausted_transport_needs_review_not_automatic_retry(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t),statuses=('transport_error',)*3)
            row = s.one(fx.run,fx.rp,fx.api(),fx.phase)
            audit = s.audit_replay(fx.run,fx.phase,fx.terminal(row))
            self.assertEqual(audit['status'],'needs_recovery_review'); self.assertTrue(audit['known_terminal_failures'][0]['technical_recovery_eligible'])

    def test_two_transport_attempts_are_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t),statuses=('transport_error',)*2)
            row = s.one(fx.run,fx.rp,fx.api(),fx.phase)
            audit = s.audit_replay(fx.run,fx.phase,fx.terminal(row))
            self.assertEqual(row['status'],'unknown'); self.assertEqual(audit['status'],'needs_review')

    def test_nonretryable_http_is_preserved_not_eligible(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t),statuses=(401,)); row = s.one(fx.run,fx.rp,fx.api(),fx.phase)
            audit = s.audit_replay(fx.run,fx.phase,fx.terminal(row))
            self.assertEqual(audit['status'],'passed'); self.assertFalse(audit['known_terminal_failures'][0]['technical_recovery_eligible'])

    def test_orphan_current_intent_without_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t)); row = s.one(fx.run,fx.rp,fx.api(),fx.phase); refs = fx.terminal(row)
            sig = core.embedding_signature(['Unmanifested physical request']); key=s.request_hash(sig)
            dump(fx.run/'request_intents'/(key+'.json'), {'request_hash':key,'signature':sig,'context':fx.context,'stage':s.EMBEDDING_STAGE,'retries':2})
            with self.assertRaisesRegex(ValueError,'Unmanifested'): s.audit_replay(fx.run,fx.phase,refs)

    def test_other_phase_intent_does_not_contaminate_embedding_projection(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t)); row = s.one(fx.run,fx.rp,fx.api(),fx.phase); refs=fx.terminal(row)
            sig={'provider':'jev','model':'jev-1.13.0','operation':'systemone','body':{'unrelated':'other-phase'}}; key=s.request_hash(sig)
            dump(fx.run/'request_intents'/(key+'.json'), {'request_hash':key,'signature':sig,'context':{'conversation':1},'stage':s.JEV_STAGE,'retries':2})
            fx.events.append({'request_hash':key,'stage':s.JEV_STAGE,'context':{'conversation':1},'status':200,'cache_hit':False});fx.write_events()
            self.assertEqual(s.audit_replay(fx.run,fx.phase,refs)['status'],'passed')

    def test_rogue_ended_event_same_hash_wrong_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t)); row=s.one(fx.run,fx.rp,fx.api(),fx.phase); refs=fx.terminal(row)
            fx.events.append({**fx.events[0], 'context':{**fx.context,'query_id':'c0_b999'}});fx.write_events()
            with self.assertRaisesRegex(ValueError,'unauthorized'): s.audit_replay(fx.run,fx.phase,refs)

    def test_raw_sha_change_rejected_even_with_valid_vectors(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(Path(t)); row=s.one(fx.run,fx.rp,fx.api(),fx.phase);refs=fx.terminal(row)
            with fx.rawp.open('a',encoding='utf-8') as out:out.write(' ')
            with self.assertRaisesRegex(ValueError,'Raw SHA'): s.audit_replay(fx.run,fx.phase,refs)

    def test_inherited_score_calls_no_api_and_requires_current_activity_absent(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),'jev'); proof={'scores':{f'f{i:02d}':.5 for i in range(80)},'raw':{'path':str(fx.rawp),'sha256':s.sha(fx.rawp)},'input_refs':[]}
            fx.rec['score_cache']=proof;dump(fx.rp,fx.rec); api=fx.api()
            with patch.object(s,'_cached_score',return_value=proof):
                row=s.one(fx.run,fx.rp,api,fx.phase)
                self.assertEqual(api.calls,0);self.assertEqual(row['score_source'],'r03_whole_request')
                with self.assertRaisesRegex(ValueError,'current API activity'):s.audit_replay(fx.run,fx.phase,fx.terminal(row))
            # This test isolates the dispatch branch; donor authentication remains the preparer owner's full audit.

    def test_inherited_raw_is_immutable_and_not_a_current_phase_producer(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t)/'new','jev'); donor=Path(t)/'old'/'raw.json'
            dump(donor,s.read(fx.rawp));fx.rawp.unlink();fx.ip.unlink();fx.events=[];fx.write_events()
            proof={'scores':s.parse_scores(s.read(donor)['response'],fx.sig),
                'raw':{'path':str(donor),'sha256':s.sha(donor)},
                'input_refs':[{'path':str(donor),'sha256':s.sha(donor)}]}
            fx.rec['score_cache']=proof;dump(fx.rp,fx.rec);api=fx.api()
            with patch.object(s,'_cached_score',return_value=proof):
                row=s.one(fx.run,fx.rp,api,fx.phase);refs=fx.terminal(row)
                self.assertEqual(api.calls,0);self.assertEqual(s.audit_replay(fx.run,fx.phase,refs)['status'],'passed')
                with donor.open('a',encoding='utf-8') as out:out.write(' ')
                with self.assertRaisesRegex(ValueError,'producer bytes changed'):s.audit_replay(fx.run,fx.phase,refs)

    def test_request_identity_cannot_be_reassigned_to_another_conversation(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));fx.rec['conversation']=2
            with self.assertRaises(ValueError):s.validate_record(fx.rec,fx.rp,fx.phase)


if __name__=='__main__':unittest.main()
