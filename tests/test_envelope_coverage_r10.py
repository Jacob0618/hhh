"""R10 decoder and durable whole-batch audit oracles; no actual imports or APIs."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from jev_mem import envelope_coverage_r10 as c
from tools import prepare_envelope_coverage_r10 as p
from tools import score_envelope_coverage_r10 as s

fxspec=importlib.util.spec_from_file_location('r09_fixture_only',s.ROOT/'tests/test_qualifier_repair_r09.py')
oldfixture=importlib.util.module_from_spec(fxspec);fxspec.loader.exec_module(oldfixture)
state,rev,dump,SPEC=oldfixture.state,oldfixture.rev,oldfixture.dump,oldfixture.SPEC
GUARD=None


def setUpModule():
    global GUARD
    GUARD=patch.object(socket,'socket',side_effect=AssertionError('Network forbidden'))
    GUARD.start()


def tearDownModule():GUARD.stop()


def response(text,**extra):
    return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}],**extra}


def literal(items=None):return json.dumps({'revisions':[rev()] if items is None else items})


def candidates():
    return c.stable_candidates(c.original.parse_proposals(literal(),state()),'c0_s001',state())


class Fixture:
    def __init__(self,run,statuses=(200,),response_override=None,inherited=False):
        self.run=run;self.paths=s.phase_paths(run,'jev')
        unit={'query_id':'c0_s001','conversation':0,'session_id':'s1','state':state()}
        self.rec=p.old._request_value(unit,SPEC,c.PROTOCOL_SHA256,candidates=candidates())
        self.rec.update(schema=p.REQUEST_SCHEMA,score_cache=None)
        self.rp=self.paths['requests']/'c0_s001.json';self.sig=self.rec['signature'];self.key=self.rec['request_hash']
        self.context=s.request_context(self.rec,'jev');self.rawp=run/'api_cache'/(self.key+'.json')
        self.ip=run/'request_intents'/(self.key+'.json');self.events=[]
        result=response_override or {'model':'jev-1.13.0','answers':{k:{'type':'noul','noul':v} for k,v in [('c00_support',.2),('c00_preservation',.9)]}}
        if inherited:
            self.rawp=run.parent/'historical'/f'{self.key}.json';dump(self.rawp,{'request':self.sig,'response':result})
            self.proof={'source':'r09_whole_request','scores':c.parse_scores(result,self.sig),
                        'raw':{'path':str(self.rawp),'sha256':s.sha(self.rawp)},
                        'input_refs':[{'path':str(self.rawp),'sha256':s.sha(self.rawp)}]}
            self.rec['score_cache']=self.proof
        else:
            dump(self.ip,{'request_hash':self.key,'signature':self.sig,'context':self.context,'stage':s.JEV_STAGE,'retries':2})
            self.events=[dict(request_hash=self.key,context=self.context,stage=s.JEV_STAGE,provider='jev',model='jev-1.13.0',operation='systemone',cache_hit=False,attempt=i+1,status=x,timestamp_utc=str(i)) for i,x in enumerate(statuses)]
            if statuses[-1]==200:dump(self.rawp,{'request':self.sig,'response':result})
        dump(self.rp,self.rec);self.events_write()
    def events_write(self):
        self.run.mkdir(parents=True,exist_ok=True)
        (self.run/'api_calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in self.events),encoding='utf8')
    def api(self):
        fx=self
        class Fake:
            run_dir=fx.run;cache_dir=fx.run/'api_cache';context=fx.context;calls=0
            def scoped(self,**ctx):
                if ctx!=fx.context:raise AssertionError('Wrong context')
                return self
            def request(self,*args,**kwargs):
                self.calls+=1
                if not fx.rawp.exists():raise RuntimeError('Offline exhausted request')
                return s.read(fx.rawp)['response']
        return Fake()
    def terminal(self,row):
        refs=[{'path':str(self.rp.resolve()),'sha256':s.sha(self.rp)}]
        dump(self.paths['manifest'],{'schema':'synthetic'})
        s.publish(self.paths['outcome'],dict(status='terminal',local_errors=[],expected_requests=1,terminal_requests=1,counts={row['status']:1}))
        dump(self.paths['intent'],dict(phase='jev',request_files=refs,manifest_sha256=s.sha(self.paths['manifest']),dispatch_code_sha256=s.sha(s.__file__),transport_retries=2))
        return refs


class DecoderTests(unittest.TestCase):
    def test_plain_exact_old_identity(self):
        text=' \n'+literal([rev('a'),rev()])+' \n'
        result=c.replay_proposal(response(text),state())
        self.assertEqual(result['parsed'],c.original.parse_proposals(text,state()))
        cs=c.stable_candidates(result['parsed'],'c0_s001',state())
        self.assertEqual([x['proposal_index'] for x in cs],[1,0])
        self.assertEqual(result['decoding']['branch'],'plain')
    def test_all_four_separator_pairs_and_exact_slice(self):
        body=' \r\n'+literal()+'  '
        for first in ('\n','\r\n'):
            for last in ('\n','\r\n'):
                text=' \t```json'+first+body+last+'```\n '
                decoded,meta=c.decode_text(text)
                self.assertEqual(decoded,body)
                lo,hi=meta['decoded_slice_codepoint_range'];self.assertEqual(text[lo:hi],body)
                self.assertEqual(meta['decoding_count'],1)
                self.assertEqual(c.replay_proposal(response(text),state())['status'],'complete')
    def test_single_envelope_only(self):
        for text in ['```JSON\n'+literal()+'\n```','```\n'+literal()+'\n```','```json \n'+literal()+'\n```','prose\n```json\n'+literal()+'\n```','```json\n'+literal()+'\n``` trailing','```json\n'+literal()+'\n```\n```json\n{}\n```','```json\n'+literal()]:
            self.assertEqual(c.replay_proposal(response(text),state())['status'],'failed',text)
    def test_literal_triple_backticks_inside_json_rejected(self):
        r=rev();r['replacement_text']='A ``` string'
        self.assertEqual(c.replay_proposal(response('```json\n'+literal([r])+'\n```'),state())['status'],'failed')
    def test_wrapper_bytes_count_before_decode(self):
        body='x'*24576
        with self.assertRaisesRegex(ValueError,'Whole visible'):c.decode_text('```json\n'+body+'\n```')
        self.assertEqual(c.decode_text(body)[0],body)
    def test_inner_not_repaired_or_partially_salvaged(self):
        bad=rev('a');bad['anchor_turn_ids']=['D1:9','D1:11']
        for inner in [literal([rev(),bad]),'{"revisions":[],"revisions":[]}',literal([rev(),rev()]),literal([rev(),rev('a'),rev()]),literal().replace('"z"','"future"')]:
            result=c.replay_proposal(response('```json\n'+inner+'\n```'),state())
            self.assertEqual(result['status'],'failed');self.assertIsNone(result['parsed'])
    def test_noop_empty_and_invalid_distinct(self):
        noop=rev();noop['replacement_text']=state()['current_facts'][0]['text']
        no=c.replay_proposal(response('```json\n'+literal([noop,rev('a')])+'\n```'),state())
        self.assertEqual(no['parsed']['exclusions'][0]['reason'],'literal_noop')
        cs=c.stable_candidates(no['parsed'],'c0_s001',state());self.assertEqual((cs[0]['candidate_key'],cs[0]['proposal_index']),('c00',1))
        empty=c.replay_proposal(response('```json\n'+literal([])+'\n```'),state())
        self.assertEqual(empty['parsed']['retained_count'],0);self.assertEqual(empty['status'],'complete')
        self.assertEqual(c.replay_proposal(response('not json'),state())['status'],'failed')
    def test_untrimmed_replacement_bytes_and_schema_exact(self):
        r=rev();r['replacement_text']='  '+r['replacement_text']+'  '
        got=c.replay_proposal(response('```json\n'+literal([r])+'\n```'),state())
        self.assertEqual(got['parsed']['candidates'][0]['replacement_text'],r['replacement_text'])
        r['replacement_text']='x'+' '*1536
        self.assertEqual(c.replay_proposal(response(literal([r])),state())['status'],'failed')
        r=rev();r['gold']='leak';self.assertEqual(c.replay_proposal(response(literal([r])),state())['status'],'failed')
    def test_raw_model_finished_and_visible_only_policy(self):
        for bad in [response(literal(),modelVersion='wrong'),{'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':literal()}]}}]}, {'candidates':[]}]:
            self.assertEqual(c.replay_proposal(bad,state())['status'],'failed')
        good=response(literal());good['candidates'][0]['content']['parts'].insert(0,{'text':'not JSON','thought':True})
        self.assertEqual(c.replay_proposal(good,state())['status'],'complete')
    def test_inner_original_full_request_predicates_identical(self):
        cs=candidates();sig=c.build_request(state(),cs,SPEC)
        self.assertEqual(sig,c.original.build_request(state(),cs,SPEC))
        low={'c00_support':.2,'c00_preservation':.9}
        self.assertEqual(c.admit_candidates(cs,low)['accepted_ids'],[])
        self.assertEqual(c.admit_candidates(cs,dict(c00_support=.8,c00_preservation=.8))['accepted_ids'],['z'])


class CacheTests(unittest.TestCase):
    def test_real_file_forward_slash_ref_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';p.save_new(path,{'schema':'fixture'})
            native={'path':str(path.resolve()),'sha256':p.file_sha(path)}
            forward={**native,'path':path.resolve().as_posix()}
            rd=p.Reader();rd.bound(p._canonical_ref(forward))
            self.assertEqual(rd.pointer(path),p._canonical_ref(forward))
            with self.assertRaisesRegex(ValueError,'source SHA'):
                rd.bound(p._canonical_ref({**forward,'sha256':'0'*64}))

    def registry(self):
        reader=p.Reader();cs=candidates();sig=c.build_request(state(),cs,SPEC)
        item={'query_id':'c0_s001','request':{'path':'req','sha256':'r'},'source_input':{'path':'src','sha256':'s'}}
        req={'signature':sig,'state':state(),'candidates':cs}
        result={'response':{'path':'res','sha256':'x'},'raw_response':{'path':'raw','sha256':'y'},'parsed':{'c00_support':.2,'c00_preservation':.9}}
        im={'registry':{c.request_hash(sig):[(item,req,result)]},'normative':SPEC,'audits':{'proposer':{'path':'pa','sha256':'a'},'jev':{'path':'ja','sha256':'b'}},'ledger':{'path':'ledger','sha256':'l'}}
        reader.bound=lambda ref:{'input_refs':[]}
        reader.raw_pointer=lambda pp:{'path':pp,'sha256':pp}
        return reader,cs,sig,im
    def test_whole_low_scores_preserved_and_cross_session_match(self):
        rd,cs,sig,im=self.registry()
        with patch.object(p,'_imports',return_value=im):proof=p.audit_old_score_cache(rd,sig,state(),cs,'c1_s002')
        self.assertEqual(proof['scores']['c00_support'],.2)
        self.assertEqual(proof['source_query_id'],'c0_s001');self.assertEqual(proof['consumer_query_id'],'c1_s002')
    def test_http_equal_proposal_ordinal_identity_mismatch_blocks(self):
        rd,cs,sig,im=self.registry();bad=copy.deepcopy(cs);bad[0]['proposal_index']=1
        self.assertEqual(c.build_request(state(),bad,SPEC),sig)
        with patch.object(p,'_imports',return_value=im),self.assertRaisesRegex(ValueError,'provenance'):
            p.audit_old_score_cache(rd,sig,state(),bad,'c0_s001')
    def test_genuine_complete_signature_miss(self):
        rd,cs,sig,im=self.registry();changed=copy.deepcopy(sig);changed['body']['questions']['c00_support']['instructions']+='different'
        with patch.object(p,'_imports',return_value=im):self.assertIsNone(p.audit_old_score_cache(rd,changed,state(),cs,'c0_s001'))


class DispatchTests(unittest.TestCase):
    def test_zero_proposer_dispatch_and_new_stage_literal(self):
        with self.assertRaises(ValueError):s.phase_paths(Path('x'),'proposer')
        api=object.__new__(s.GateAPI)
        sig=c.original.build_proposer_request(state(),SPEC)
        with self.assertRaises(ValueError):api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage=s.PROPOSER_STAGE)
        sig=c.build_request(state(),candidates(),SPEC)
        with patch.object(s.PacedEmbeddingAPI,'request',return_value={'offline':True}) as call:
            self.assertEqual(api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage=s.JEV_STAGE),{'offline':True});call.assert_called_once()
    def test_current_raw_full_replay_and_immutability(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));api=fx.api();row=s.one(fx.run,fx.rp,api,'jev')
            self.assertEqual(row['status'],'complete');self.assertEqual(api.calls,1)
            self.assertEqual(s.audit_replay(fx.run,'jev',fx.terminal(row))['status'],'passed')
            with self.assertRaises(FileExistsError):s.one(fx.run,fx.rp,api,'jev')
    def test_inherited_zero_api_and_old_low_scores(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t)/'run',inherited=True)
            with patch.object(s,'_cached_score',return_value=fx.proof):
                row=s.one(fx.run,fx.rp,None,'jev');refs=fx.terminal(row)
                self.assertEqual(row['score_source'],'r09_whole_request');self.assertEqual(row['scores']['c00_support'],.2)
                self.assertEqual(s.audit_replay(fx.run,'jev',refs)['status'],'passed')
    def test_inherited_current_intent_even_without_event_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t)/'run',inherited=True)
            with patch.object(s,'_cached_score',return_value=fx.proof):
                row=s.one(fx.run,fx.rp,None,'jev');refs=fx.terminal(row)
                dump(fx.ip,dict(request_hash=fx.key,signature=fx.sig,stage=s.JEV_STAGE,retries=2,context=fx.context))
                with self.assertRaisesRegex(ValueError,'current API activity'):s.audit_replay(fx.run,'jev',refs)
    def test_inherited_old_raw_mutation_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t)/'run',inherited=True)
            with patch.object(s,'_cached_score',return_value=fx.proof):
                row=s.one(fx.run,fx.rp,None,'jev');refs=fx.terminal(row)
                with fx.rawp.open('a') as f:f.write(' ')
                with self.assertRaisesRegex(ValueError,'donor changed'):s.audit_replay(fx.run,'jev',refs)
    def test_malformed200_preserves_failed_not_recovery(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),response_override={'model':'jev-1.13.0','answers':{}})
            row=s.one(fx.run,fx.rp,fx.api(),'jev');audit=s.audit_replay(fx.run,'jev',fx.terminal(row))
            self.assertEqual(row['status'],'failed');self.assertIsNone(row['scores']);self.assertEqual(audit['status'],'passed')
            self.assertFalse(audit['known_terminal_failures'][0]['technical_recovery_eligible'])
    def test_retryable_exhaustion_and_unknown_hard_blocks(self):
        for statuses,result in [(('transport_error',)*3,'needs_recovery_review'),((500,)*3,'needs_recovery_review'),(('transport_error',)*2,'needs_review')]:
            with tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),statuses=statuses);row=s.one(fx.run,fx.rp,fx.api(),'jev')
                self.assertEqual(s.audit_replay(fx.run,'jev',fx.terminal(row))['status'],result)
                with self.assertRaisesRegex(RuntimeError,'HARD BLOCKED'):s.recover_phase(fx.run,'jev')
    def test_unmanifested_stage_intent_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.api(),'jev');refs=fx.terminal(row)
            dump(fx.run/'request_intents/orphan.json',dict(request_hash='orphan',signature=fx.sig,stage=s.JEV_STAGE,context=fx.context))
            with self.assertRaisesRegex(ValueError,'Unmanifested'):s.audit_replay(fx.run,'jev',refs)


if __name__=='__main__':unittest.main()
