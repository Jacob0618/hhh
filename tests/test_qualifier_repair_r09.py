"""R09 owner offline oracles: strict whole-response and durable raw/intent audit."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from jev_mem import qualifier_repair_r09 as c
from tools import prepare_qualifier_repair_r09 as p
from tools import score_qualifier_repair_r09 as s

NETWORK_GUARD=None

def setUpModule():
    global NETWORK_GUARD
    NETWORK_GUARD=patch.object(socket,'socket',side_effect=AssertionError('Network forbidden'))
    NETWORK_GUARD.start()

def tearDownModule():
    NETWORK_GUARD.stop()

SPEC=json.loads((s.ROOT/'research/jev_autoresearch_2026-10-04/R09_qualifier_repair_protocol_final_v1.json').read_text(encoding='utf-8'))


def state():
    return {'session_id':'s1','observation_date':'2023-05-08',
        'current_source':[{'turn_id':tid,'speaker':who,'session_id':'s1','observation_date':'2023-05-08','text':text}
            for tid,who,text in [('D1:9','A','What is your favorite game?'),('D1:10','B','It Takes Two is my favorite game.'),('D1:11','B','Max was my childhood dog.')]],
        'historical_source':[{'turn_id':'D0:3','speaker':'B','session_id':'s0','observation_date':'2023-05-01','text':'I have four dogs now.'}],
        'current_facts':[{'id':'z','text':'B enjoys It Takes Two.','observation_date':'2023-05-08'},
                         {'id':'a','text':'B had a dog named Max.','observation_date':'2023-05-08'}]}


def rev(target='z'):
    if target=='z':return {'fact_id':'z','replacement_text':'It Takes Two is B favorite game.','edit_kind':'clarify',
        'anchor_turn_ids':['D1:9','D1:10'],'evidence':[{'turn_id':'D1:10','field':'text','quote':'It Takes Two is my favorite game.'}]}
    return {'fact_id':'a','replacement_text':'Max was B childhood dog.','edit_kind':'clarify','anchor_turn_ids':['D1:11'],
        'evidence':[{'turn_id':'D1:11','field':'text','quote':'Max was my childhood dog.'}]}


def candidates(st=None,items=None):
    st=state() if st is None else st
    parsed=c.parse_proposals(json.dumps({'revisions':[rev()] if items is None else items}),st)
    return c.stable_candidates(parsed,'c0_s001',st)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


class Fixture:
    def __init__(self,run,phase='proposer',statuses=(200,),response=None,empty=False):
        self.run,self.phase=run,phase;self.paths=s.phase_paths(run,phase)
        st=state()
        if empty:st['current_facts']=[]
        unit={'query_id':'c0_s001','conversation':0,'session_id':'s1','state':st}
        self.rec=p._request_value(unit,SPEC,c.PROTOCOL_SHA256,candidates=candidates(st) if phase=='jev' else None)
        self.rp=self.paths['requests']/'c0_s001.json';dump(self.rp,self.rec)
        self.sig=self.rec['signature'];self.key=self.rec['request_hash'];self.context=s.request_context(self.rec,phase)
        self.rawp=run/'api_cache'/(self.key+'.json');self.ip=run/'request_intents'/(self.key+'.json')
        self.events=[]
        if self.rec['dispatch_eligible']:
            dump(self.ip,{'request_hash':self.key,'signature':self.sig,'context':self.context,'stage':self.paths['stage'],'retries':2})
            self.events=[dict(request_hash=self.key,context=self.context,stage=self.paths['stage'],provider=self.sig['provider'],model=self.sig['model'],operation=self.sig['operation'],cache_hit=False,attempt=i+1,status=status,timestamp_utc=str(i)) for i,status in enumerate(statuses)]
            if statuses[-1]==200:
                if response is None:
                    response={'modelVersion':'gemini-3.8-flash','candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({'revisions':[rev()]})}]}}]} if phase=='proposer' else {'model':'jev-1.13.0','answers':{k:{'type':'noul','noul':.8} for k in ('c00_support','c00_preservation')}}
                dump(self.rawp,{'request':self.sig,'response':response})
        self.write_events()
    def write_events(self):
        (self.run/'api_calls.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events),encoding='utf-8')
    def api(self):
        fx=self
        class Fake:
            run_dir=fx.run;cache_dir=fx.run/'api_cache';context=fx.context;calls=0
            def scoped(self,**context):
                if context!=fx.context:raise AssertionError('wrong context')
                return self
            def request(self,*args,**kwargs):
                self.calls+=1
                if not fx.rawp.exists():raise RuntimeError('offline terminal transport')
                return s.read(fx.rawp)['response']
        return Fake()
    def terminal(self,row):
        refs=[{'path':str(self.rp.resolve()),'sha256':s.sha(self.rp)}]
        dump(self.paths['manifest'],{'schema':'synthetic'})
        s.publish(self.paths['outcome'],{'status':'terminal','local_errors':[],'expected_requests':1,'terminal_requests':1,'counts':{row['status']:1}})
        dump(self.paths['intent'],{'phase':self.phase,'request_files':refs,'manifest_sha256':s.sha(self.paths['manifest']),'dispatch_code_sha256':s.sha(s.__file__),'transport_retries':2})
        return refs


class CoreTests(unittest.TestCase):
    def test_five_fields_no_generated_prefix(self):
        st=state();self.assertEqual(set(st),c.STATE_FIELDS)
        for field in ('baseline_prefix','question','gold'):
            bad=copy.deepcopy(st);bad[field]=[]
            with self.assertRaises(ValueError):c.build_proposer_request(bad,SPEC)
        body=c.build_proposer_request(st,SPEC)['body'];literal=body['contents'][0]['parts'][0]['text']
        self.assertIn('\nINPUT_JSON:\n',literal);self.assertNotIn('baseline_prefix',literal)
        self.assertEqual(set(body),{'contents','generationConfig'})
    def test_current_target_and_date_only(self):
        for target in ('past','future'):
            bad=rev();bad['fact_id']=target
            with self.assertRaises(ValueError):c.parse_proposals(json.dumps({'revisions':[bad]}),state())
        st=state();st['current_facts'][0]['observation_date']='2023-05-01'
        with self.assertRaises(ValueError):c.validate_state(st)
    def test_source_order_not_lexicographic_anchor(self):
        self.assertEqual(len(candidates()),1)
        for anchor in (['D1:10','D1:9'],['D1:9','D1:11'],['D1:10','D1:10'],['D0:3'],['D1:9']):
            bad=rev();bad['anchor_turn_ids']=anchor
            with self.assertRaises(ValueError):c.parse_proposals(json.dumps({'revisions':[bad]}),state())
    def test_whole_response_invalid_no_salvage(self):
        bad=rev('a');bad['evidence'][0]['quote']='not in source'
        for items in ([rev(),bad],[rev(),rev()],[rev(),rev('a'),rev()]):
            with self.assertRaises(ValueError):c.parse_proposals(json.dumps({'revisions':items}),state())
        with self.assertRaises(ValueError):c.parse_proposals('{"revisions":[],"revisions":[]}',state())
        with self.assertRaises(ValueError):c.parse_proposals('```json\n{"revisions":[]}\n```',state())
    def test_order_and_noop_remap(self):
        cs=candidates(items=[rev('a'),rev()]);self.assertEqual([x['fact_id'] for x in cs],['z','a'])
        self.assertEqual([x['proposal_index'] for x in cs],[1,0])
        noop=rev();noop['replacement_text']=state()['current_facts'][0]['text']
        st=state();parsed=c.parse_proposals(json.dumps({'revisions':[noop,rev('a')]}),st)
        cs=c.stable_candidates(parsed,'c0_s001',st)
        self.assertEqual(cs[0]['candidate_key'],'c00');self.assertEqual(cs[0]['proposal_index'],1)
        self.assertEqual(parsed['exclusions'][0]['reason'],'literal_noop')
    def test_literal_generated_whitespace_is_preserved(self):
        r=rev();r['replacement_text']='  '+r['replacement_text']+'  '
        self.assertEqual(candidates(items=[r])[0]['replacement_text'],r['replacement_text'])
        for text in ('x'+' '*1600,' '*1600+'x'):
            bad=rev();bad['replacement_text']=text
            with self.assertRaises(ValueError):candidates(items=[bad])
    def test_bounds_and_exact_schema(self):
        for field,value in [('replacement_text','x'*1537),('edit_kind','add'),('anchor_turn_ids',[]),('evidence',[])]:
            bad=rev();bad[field]=value
            with self.assertRaises(ValueError):c.parse_proposals(json.dumps({'revisions':[bad]}),state())
        bad=rev();bad['question']='leak'
        with self.assertRaises(ValueError):c.parse_proposals(json.dumps({'revisions':[bad]}),state())
    def test_support_preservation_and_complete_scores(self):
        cs=candidates(items=[rev(),rev('a')]);scores={k:.8 for k in c.expected_keys(cs)}
        self.assertEqual(c.admit_candidates(cs,scores)['accepted_ids'],['z','a'])
        scores['c01_preservation']=.799999;self.assertEqual(c.admit_candidates(cs,scores)['accepted_ids'],['z'])
        for bad in ({k:v for k,v in scores.items() if k!='c00_support'},{**scores,'c00_support':True},{**scores,'c00_support':float('nan')}):
            with self.assertRaises(ValueError):c.admit_candidates(cs,bad)
        self.assertEqual(c.admit_candidates(cs,None,status='failed')['accepted_ids'],[])
        with self.assertRaises(ValueError):c.admit_candidates([],None,status='unknown')
    def test_candidate_target_provenance_and_full_prompts(self):
        cs=candidates();sig=c.build_request(state(),cs,SPEC)
        q=sig['body']['questions'];self.assertEqual(set(q),{'c00_support','c00_preservation'})
        self.assertIn('ORIGINAL_FACT_JSON:',q['c00_support']['instructions'])
        bad=copy.deepcopy(cs);bad[0]['original_fact']['text']='forged target'
        with self.assertRaises(ValueError):c.build_request(state(),bad,SPEC)
    def test_current_fact_projection_keeps_original_bank_order(self):
        st=state();old={k:copy.deepcopy(v) for k,v in st.items() if k!='current_facts'};old['baseline_prefix']=st['current_facts']
        unit={'session_id':'s1','observation_date':st['observation_date'],'sequence':0,'state':old}
        bank={'facts':st['current_facts']};journal={'fact_records':[st['current_facts'][1],st['current_facts'][0]]}
        with patch.object(p.original_source,'session_units',return_value=[unit]):rows=p.session_units({},bank,{},[journal])
        self.assertEqual(rows[0]['current_fact_ids'],['z','a']);self.assertEqual(set(rows[0]['state']),c.STATE_FIELDS)
        self.assertIn('offline_prefix_sha256',rows[0])


class DispatchTests(unittest.TestCase):
    def test_actual_new_stage_literal_gate_and_no_old_stage(self):
        api=object.__new__(s.GateAPI)
        for sig,stage in [(c.build_proposer_request(state(),SPEC),s.PROPOSER_STAGE),(c.build_request(state(),candidates(),SPEC),s.JEV_STAGE)]:
            with patch.object(s.PacedEmbeddingAPI,'request',return_value={'offline':True}) as call:
                self.assertEqual(api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage=stage),{'offline':True});call.assert_called_once()
            with self.assertRaises(ValueError):api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage='write_omission_proposer_r07')
        sig=c.build_request(state(),candidates(),SPEC);sig['body']['questions']['c00_preservation']['instructions']+='changed'
        with self.assertRaises(ValueError):api.request(sig['provider'],sig['model'],sig['body'],operation=sig['operation'],stage=s.JEV_STAGE)
    def test_both_success_phases_raw_replay(self):
        for phase in ('proposer','jev'):
            with tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),phase);api=fx.api();row=s.one(fx.run,fx.rp,api,phase)
                self.assertEqual(row['status'],'complete');self.assertEqual(s.audit_replay(fx.run,phase,fx.terminal(row))['status'],'passed')
                with self.assertRaises(FileExistsError):s.one(fx.run,fx.rp,api,phase)
                self.assertEqual(api.calls,1)
    def test_zero_fact_session_no_api_and_no_raw(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),empty=True);row=s.one(fx.run,fx.rp,None,'proposer')
            self.assertEqual((row['status'],row['reason']),('not_dispatched','zero_current_facts'))
            self.assertIsNone(row['raw']);refs=fx.terminal(row);self.assertEqual(s.audit_replay(fx.run,'proposer',refs)['status'],'passed')
            dump(fx.ip,{'request_hash':fx.key,'signature':fx.sig,'stage':s.PROPOSER_STAGE,'context':fx.context,'retries':2})
            with self.assertRaises(ValueError):s.audit_replay(fx.run,'proposer',refs)
    def test_malformed200_retained_not_recovered(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t),'jev',response={'model':'jev-1.13.0','answers':{'c00_support':{'type':'noul','noul':.9}}})
            row=s.one(fx.run,fx.rp,fx.api(),'jev');audit=s.audit_replay(fx.run,'jev',fx.terminal(row))
            self.assertEqual(row['status'],'failed');self.assertIsNone(row['scores']);self.assertEqual(audit['status'],'passed')
            self.assertFalse(audit['known_terminal_failures'][0]['technical_recovery_eligible'])
    def test_explicit_model_mismatch_and_valid_empty_differ(self):
        for response,expected in [
            ({'modelVersion':'wrong','candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"revisions":[]}'}]}}]},'failed'),
            ({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"revisions":[]}'}]}}]},'complete')]:
            with tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),response=response);row=s.one(fx.run,fx.rp,fx.api(),'proposer')
                self.assertEqual(row['status'],expected);self.assertEqual(s.audit_replay(fx.run,'proposer',fx.terminal(row))['status'],'passed')
                if expected=='complete':self.assertEqual(row['proposal_parse']['retained_count'],0);self.assertIsNone(row['resolved_model_version'])

    def test_retryable_exhaustion_hard_blocks(self):
        for statuses in [('transport_error',)*3,(429,)*3,(500,)*3]:
            with tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),statuses=statuses);row=s.one(fx.run,fx.rp,fx.api(),'proposer')
                self.assertEqual(s.audit_replay(fx.run,'proposer',fx.terminal(row))['status'],'needs_recovery_review')
                with self.assertRaisesRegex(RuntimeError,'HARD BLOCKED'):s.recover_phase(fx.run,'proposer')
    def test_incomplete_transport_unknown_nonretryable_terminal(self):
        for statuses,expected in [(('transport_error',)*2,'needs_review'),((401,),'passed')]:
            with tempfile.TemporaryDirectory() as t:
                fx=Fixture(Path(t),statuses=statuses);row=s.one(fx.run,fx.rp,fx.api(),'proposer')
                self.assertEqual(s.audit_replay(fx.run,'proposer',fx.terminal(row))['status'],expected)
    def test_extra_orphan_intent_without_event_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.api(),'proposer');refs=fx.terminal(row)
            sig=c.build_proposer_request({**state(),'session_id':'s2','current_source':[{**x,'session_id':'s2'} for x in state()['current_source']]},SPEC)
            key=s.request_hash(sig);dump(fx.run/'request_intents'/(key+'.json'),{'request_hash':key,'signature':sig,'stage':s.PROPOSER_STAGE,'context':fx.context,'retries':2})
            with self.assertRaisesRegex(ValueError,'Unmanifested'):s.audit_replay(fx.run,'proposer',refs)
    def test_raw_and_consumer_binding_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.api(),'proposer');refs=fx.terminal(row)
            fx.events.append({**fx.events[0],'context':{**fx.context,'query_id':'c0_s999'}});fx.write_events()
            with self.assertRaisesRegex(ValueError,'unauthorized'):s.audit_replay(fx.run,'proposer',refs)
        with tempfile.TemporaryDirectory() as t:
            fx=Fixture(Path(t));row=s.one(fx.run,fx.rp,fx.api(),'proposer');refs=fx.terminal(row)
            with fx.rawp.open('a',encoding='utf-8') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'Raw SHA'):s.audit_replay(fx.run,'proposer',refs)


if __name__=='__main__':
    # A bug in these fixtures may never become a live network call.
    with patch.object(socket,'socket',side_effect=AssertionError('Network forbidden')):
        unittest.main()
