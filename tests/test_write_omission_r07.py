"""Offline R07 provenance/admission tests; no API and no original store mutation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from jev_mem import write_omission_r07 as pure
from tools import prepare_write_omission_r07 as prep


def state():
    return {'session_id':'session_2','observation_date':'2023-01-02',
        'current_source':[{'turn_id':'D2:1','speaker':'Amy','session_id':'session_2','observation_date':'2023-01-02',
                          'text':'I plan to paint blue birds because they remind me of home.','blip_caption':'Two blue birds.'}],
        'historical_source':[{'turn_id':'D1:1','speaker':'Amy','session_id':'session_1','observation_date':'2023-01-01','text':'I like birds.'}],
        'baseline_prefix':[{'id':'old','text':'Amy likes birds.','observation_date':'2023-01-01'}]}


def proposal(statement='Amy plans to paint blue birds because they remind her of home.'):
    return {'statement':statement,'evidence':[{'turn_id':'D2:1','field':'text','quote':'I plan to paint blue birds because they remind me of home.'}]}


def spec():
    return json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))


class PureTests(unittest.TestCase):
    def test_exact_final_protocol(self):
        self.assertEqual(hashlib.sha256(prep.RESEARCH_PROTOCOL.read_bytes()).hexdigest(),pure.PROTOCOL_SHA256)

    def test_projection_rejects_question_and_query(self):
        for location,key in [('state','question'),('turn','query')]:
            s=state();target=s if location=='state'else s['current_source'][0];target[key]='forbidden'
            with self.assertRaises(ValueError):pure.validate_state(s)

    def test_full_batch_invalidates_after_valid(self):
        bad=proposal();bad['evidence'][0]['quote']='not in the source'
        with self.assertRaises(ValueError):pure.parse_proposals(json.dumps({'memories':[proposal(),bad]}),state())

    def test_reject_duplicate_json_keys_fences_trailing(self):
        for raw in ['{"memories":[],"memories":[]}','```json\n{"memories":[]}\n```','{"memories":[]} tail']:
            with self.assertRaises(ValueError):pure.parse_proposals(raw,state())

    def test_history_only_citation_rejected(self):
        p=proposal();p['evidence']=[{'turn_id':'D1:1','field':'text','quote':'I like birds.'}]
        with self.assertRaises(ValueError):pure.parse_proposals(json.dumps({'memories':[p]}),state())

    def test_case_sensitive_caption_field_and_duplicate_citations(self):
        p=proposal();p['evidence']=[{'turn_id':'D2:1','field':'blip_caption','quote':'Two blue birds.'}]
        self.assertEqual(pure.parse_proposals(json.dumps({'memories':[p]}),state())['retained_count'],1)
        p['evidence']*=2
        with self.assertRaises(ValueError):pure.parse_proposals(json.dumps({'memories':[p]}),state())

    def test_dedup_preserves_raw_index_but_axis_is_retained_order(self):
        parsed=pure.parse_proposals(json.dumps({'memories':[proposal('Amy likes birds.'),proposal()]}),state())
        candidates=pure.stable_candidates(parsed,'c0_s002',state())
        self.assertEqual(candidates[0]['candidate_id'],'c0_s002_m01');self.assertEqual(candidates[0]['proposal_index'],1)
        request=pure.build_request(state(),candidates,spec())
        self.assertEqual(set(request['body']['questions']),{'c00_support','c00_novelty'})
        decision=pure.admit_candidates(candidates,{'c00_support':.8,'c00_novelty':.5})
        self.assertEqual(decision['accepted_ids'],['c0_s002_m01'])

    def test_literal_not_semantic_dedup(self):
        raw={'memories':[proposal(),proposal()]};p=pure.parse_proposals(json.dumps(raw),state())
        self.assertEqual(p['retained_count'],1);self.assertEqual(p['exclusions'][0]['reason'],'duplicate_statement')

    def test_quote_statement_byte_caps(self):
        p=proposal('é'*385)
        with self.assertRaises(ValueError):pure.parse_proposals(json.dumps({'memories':[p]}),state())
        s=state();s['current_source'][0]['text']='é'*600;p=proposal();p['evidence'][0]['quote']='é'*513
        with self.assertRaises(ValueError):pure.parse_proposals(json.dumps({'memories':[p]}),s)

    def test_builders_do_not_forward_gold_and_config_is_exact(self):
        s=state();sig=pure.build_proposer_request(s,spec())
        self.assertEqual(set(sig['body']),{'contents','generationConfig'})
        self.assertNotIn('systemInstruction',sig['body']);self.assertNotIn('responseMimeType',sig['body']['generationConfig'])
        self.assertEqual(sig['body']['generationConfig']['thinkingConfig'],{'thinkingLevel':'low'})
        self.assertNotIn('\\nINPUT_JSON',sig['body']['contents'][0]['parts'][0]['text'])

    def test_threshold_complete_scores_unknown_no_imputation(self):
        candidates=pure.stable_candidates(pure.parse_proposals(json.dumps({'memories':[proposal()]}),state()),'c0_s002',state())
        self.assertFalse(pure.admit_candidates(candidates,{'c00_support':.799,'c00_novelty':1})['accepted_ids'])
        for scores in [{'c00_support':1},{'c00_support':True,'c00_novelty':1},{'c00_support':float('nan'),'c00_novelty':1}]:
            with self.assertRaises(ValueError):pure.admit_candidates(candidates,scores)
        with self.assertRaises(ValueError):pure.admit_candidates(candidates,None,status='unknown')
        self.assertEqual(pure.admit_candidates(candidates,None,status='failed')['accepted_ids'],[])
        self.assertEqual(pure.admit_candidates([],None)['reason'],'no_candidates')


class SourceSeamTests(unittest.TestCase):
    def test_actual_original_retrieval_manifest_publishing_contract(self):
        # This old artifact uses protocol_sha256, not generic content_sha256.
        path=prep.ROOT/'runs/gemini38_rebuild_v1_20261003/retrieval/run_manifest.json'
        value=prep.Reader().read(path,sealed=False)
        self.assertEqual(value['protocol_sha256'],pure.digest({k:v for k,v in value.items()if k!='protocol_sha256'}))
        self.assertNotIn('content_sha256',value)

    def fixtures(self):
        c={'speaker_a':'Amy','speaker_b':'Ben','session_1_date_time':'1:00 pm on 1 January, 2023',
           'session_2_date_time':'1:00 pm on 2 January, 2023',
           'session_1':[{'speaker':'Amy','dia_id':'D1:1','text':'Birds are nice.'}],
           'session_2':[{'speaker':'Ben','dia_id':'D2:1','text':'Good picture.','query':'secret image-search term','blip_caption':'Blue birds.'}]}
        env=list(prep.session_envelopes({'conversation':c}));facts=[];journals=[];history=[]
        for i,e in enumerate(env):
            f={'id':f'f{i}','text':f'Statement {i}.','event':'ADD','user_id':'u','observation_date':e['observation_date'],
               'input_scope':{'session_id':e['session_id'],'source_turn_ids':e['source_turn_ids'],
                   'historical_turn_ids':[h['source_turn_id']for h in history[-10:]],'existing_memory_ids':[]},
               'verified_support':[],'support_status':'unverified'}
            facts.append({**deepcopy(f),'embedding':[.1]});journals.append({**e,'status':'success','sequence':i,'user_id':'u',
                   'history_context':deepcopy(history[-10:]),'fact_records':[deepcopy(f)]})
            history.extend({'message':m,'session_id':e['session_id'],'source_turn_id':tid}for m,tid in zip(e['messages'],e['source_turn_ids']))
        return c,{'conversation':0,'user_id':'u','facts':facts},{'arm':'session','conversation':0,'envelopes':env},journals

    def test_actual_source_fields_and_future_prefix_exclusion(self):
        c,b,g,j=self.fixtures();units=prep.session_units(c,b,g,j)
        self.assertEqual([f['id']for f in units[0]['state']['baseline_prefix']],['f0'])
        self.assertEqual([f['id']for f in units[1]['state']['baseline_prefix']],['f0','f1'])
        t=units[1]['state']['current_source'][0]
        self.assertEqual(t['text'],'Good picture.');self.assertEqual(t['blip_caption'],'Blue birds.');self.assertNotIn('query',t)
        self.assertEqual(units[1]['state']['historical_source'][0]['turn_id'],'D1:1')

    def test_final_update_and_history_tamper_block(self):
        for kind in ('text','history','birth'):
            c,b,g,j=self.fixtures()
            if kind=='text':b['facts'][0]['text']='future UPDATE'
            elif kind=='history':j[1]['history_context'][0]['message']['content']='wrong'
            else:b['facts'][1]['input_scope']['session_id']='session_1'
            with self.assertRaises(ValueError):prep.session_units(c,b,g,j)

    def test_readonly_native_history_add_only(self):
        c,b,g,j=self.fixtures()
        with tempfile.TemporaryDirectory()as td:
            db=Path(td)/'history.sqlite';conn=sqlite3.connect(db)
            conn.execute('CREATE TABLE history(memory_id TEXT,event TEXT,is_deleted INT,old_memory TEXT,new_memory TEXT)')
            conn.executemany('INSERT INTO history VALUES(?,?,?,?,?)',[(f['id'],'ADD',0,None,f['text'])for f in b['facts']]);conn.commit();conn.close()
            bank={**b,'native_dir':td};before=db.read_bytes();prep._history(prep.Reader(),bank);self.assertEqual(before,db.read_bytes())
            conn=sqlite3.connect(db);conn.execute("UPDATE history SET event='UPDATE' WHERE memory_id='f0'");conn.commit();conn.close()
            with self.assertRaises(ValueError):prep._history(prep.Reader(),bank)

    def test_zero_candidates_generate_no_requests(self):
        self.assertEqual(pure.stable_candidates(pure.parse_proposals('{"memories":[]}',state()),'c0_s002',state()),[])
        with self.assertRaises(ValueError):pure.build_request(state(),[],spec())


class AuditGateTests(unittest.TestCase):
    def fixture(self,run):
        unit={'query_id':'c0_s002','conversation':0,'session_id':'session_2','state':state()}
        req=prep._request_value(unit,spec(),'p');rref=prep.save_new(run/'proposer_requests/c0_s002.json',req)
        mref=prep.save_new(run/'run_manifest.json',{'items':[{'request':rref}]})
        oref=prep.save_new(run/'proposer_outcome.json',{'status':'complete'})
        cref=prep.save_new(run/'execution_intent_proposer.json',{'status':'intent'})
        response=prep.save_new(run/'proposer_responses/c0_s002.json',{'status':'failed'})
        ledger=run/'api_calls.jsonl';ledger.write_text(json.dumps({'stage':'write_omission_proposer_r07','request_hash':req['request_hash'],'event':'known'})+'\n',encoding='utf8')
        stage='write_omission_proposer_r07';projection=prep.coverage.ledger_projection(prep.Reader(),ledger,stage,[req['request_hash']])
        auditor=prep.ROOT/'tools/score_write_omission_r07.py';pins={'path':str(auditor.resolve()),'sha256':prep.file_sha(auditor)}
        audit={'schema':prep.AUDIT_SCHEMA,'phase':'proposer','status':'passed','manifest':mref,'outcome':oref,
            'unknown':[],'requests_count':1,'known_terminal_failures':[],'codepins':pins,
            'input_refs':[mref,oref,cref,rref,response],'ledger_projection':projection,
            'response_refs':[{**response,'request_id':'c0_s002','query_id':'c0_s002','request_ref':rref}]}
        prep.save_new(run/'reviews/proposer_terminal_audit_v1.json',audit)
        return mref,rref,ledger,pins

    def test_other_phase_append_allowed_samehash_or_stage_mutation_rejected(self):
        with tempfile.TemporaryDirectory()as td:
            run=Path(td);mref,rref,ledger,pins=self.fixture(run)
            with patch.object(prep,'_codes',return_value={'tools/score_write_omission_r07.py':pins['sha256']}):
                prep._audit_gate(prep.Reader(),run,'proposer',mref,[rref])
                with ledger.open('a',encoding='utf8')as h:h.write(json.dumps({'stage':'write_omission_score_r07','request_hash':'other'})+'\n')
                prep._audit_gate(prep.Reader(),run,'proposer',mref,[rref])
                request=prep.Reader().bound(rref)
                with ledger.open('a',encoding='utf8')as h:h.write(json.dumps({'stage':'other','request_hash':request['request_hash']})+'\n')
                with self.assertRaises(ValueError):prep._audit_gate(prep.Reader(),run,'proposer',mref,[rref])

    def test_post_audit_immutable_response_tamper_rejected(self):
        with tempfile.TemporaryDirectory()as td:
            run=Path(td);mref,rref,ledger,pins=self.fixture(run)
            path=run/'proposer_responses/c0_s002.json';path.write_text('{"changed":true}',encoding='utf8')
            with patch.object(prep,'_codes',return_value={'tools/score_write_omission_r07.py':pins['sha256']}):
                with self.assertRaises(ValueError):prep._audit_gate(prep.Reader(),run,'proposer',mref,[rref])


if __name__=='__main__':unittest.main()
