import copy
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from jev_mem import answer_selection_r04 as p
from tools import prepare_answer_selection_r04 as prep

def facts(n=40):return [{'id':str(i),'text':'memory '+str(i),'observation_date':None} for i in range(n)]
def record(b='  original answer!\n',j='different answer'):
    spec=json.loads(prep.RESEARCH_PROTOCOL.read_bytes());cs={'baseline':{'answer':b},'jev_r03':{'answer':j}}
    sig=p.build_request('What happened?','2020-01-01',facts(20),cs,'c3_q4',spec)
    return {'query_id':'c3_q4','presentation':p.presentation('c3_q4'),'candidates':cs,
            'identical':b==j,'signature':sig}
def response(choice='first_better',probs=None):
    if probs is None:probs={k:(.7 if k==choice else .1) for k in p.CHOICES}
    return {'model':'jev-1.13.0','answers':{'answer_choice':{'type':'choice','choice':choice,
            'probabilities':probs,'confidence':.6}}}

class PureTests(unittest.TestCase):
    def test_fixed_protocol_byte_sha(self):
        self.assertEqual(prep.file_sha(prep.RESEARCH_PROTOCOL),p.PROTOCOL_SHA256)
    def test_blind_request_only_allowed_state_fields(self):
        r=record();s=r['signature']['body']['state']
        self.assertEqual(set(s),{'question','reference_date','memories','candidate_first','candidate_second'})
        self.assertNotIn('baseline',json.dumps(s));self.assertNotIn('jev_r03',json.dumps(s))
        self.assertEqual(set(r['signature']['body']['questions']['answer_choice']['criteria']),set(p.CHOICES))
    def test_hash_mapping_and_candidate_whitespace_preserved(self):
        r=record();first=r['presentation']['first']
        self.assertEqual(r['signature']['body']['state']['candidate_first'],r['candidates'][first]['answer'])
        self.assertEqual(p.presentation('c3_q4'),p.presentation('c3_q4'))
    def test_union_pool_order_exact_metadata(self):
        pool=facts();u=p.union_records(pool,pool[:20],pool[10:30])
        self.assertEqual(u,pool[:30]);self.assertIsNone(u[0]['observation_date'])
    def test_union_rejects_rewritten_or_outside_fact(self):
        pool=facts();s=copy.deepcopy(pool[10:30]);s[0]['text']='rewrite'
        with self.assertRaises(ValueError):p.union_records(pool,pool[:20],s)
        with self.assertRaises(ValueError):p.union_records(pool,pool[1:21],pool[10:30])
    def test_exact_abstention_and_no_substring(self):
        for phrase in p.POLICY['abstention_exact_set']:
            self.assertEqual(p.cheap_choice(record(' '+phrase.upper()+'.\n','supported')), 'jev_r03')
        self.assertEqual(p.cheap_choice(record('Insufficient evidence; however, he went home.','answer')),'baseline')
        self.assertEqual(p.cheap_choice(record('Insufficient evidence','The evidence is insufficient!')),'baseline')
    def test_missing_answer_hard_error(self):
        with self.assertRaises(ValueError):record('','answer')
    def test_choice_route_mapping(self):
        r=record()
        for choice,slot in [('first_better','first'),('second_better','second')]:
            parsed=p.parse_choice(response(choice),r['signature'])
            self.assertEqual(p.choose_jev(r,parsed)['route'],r['presentation'][slot])
    def test_valid_semantic_fallback(self):
        r=record()
        for choice in ('no_clear_preference','neither_supported'):
            d=p.choose_jev(r,p.parse_choice(response(choice),r['signature']))
            self.assertEqual(d['route'],'baseline');self.assertIn(choice,d['reason'])
    def test_tie_fallback_and_forged_tie_rejected(self):
        r=record();probs=dict(zip(p.CHOICES,[.4,.4,.1,.1]))
        parsed=p.parse_choice(response(probs=probs),r['signature'])
        self.assertTrue(parsed['tied_max']);self.assertEqual(p.choose_jev(r,parsed)['route'],'baseline')
        parsed['tied_max']=False
        with self.assertRaises(ValueError):p.choose_jev(r,parsed)
    def test_invalid_model_type_missing_key(self):
        r=record()
        for change in ('model','type','key'):
            raw=response()
            if change=='model':raw['model']='another'
            elif change=='type':raw['answers']['answer_choice']['type']='noul'
            else:raw['answers']['answer_choice']['probabilities'].pop('second_better')
            with self.assertRaises(ValueError):p.parse_choice(raw,r['signature'])
    def test_nonfinite_boolean_sum_and_argmax_rejected(self):
        r=record()
        for value in (True,math.nan,math.inf,-.1,1.1):
            raw=response();raw['answers']['answer_choice']['probabilities']['first_better']=value
            with self.assertRaises(ValueError):p.parse_choice(raw,r['signature'])
        raw=response();raw['answers']['answer_choice']['probabilities']['first_better']=.6
        with self.assertRaises(ValueError):p.parse_choice(raw,r['signature'])
        raw=response();raw['answers']['answer_choice']['choice']='second_better'
        with self.assertRaises(ValueError):p.parse_choice(raw,r['signature'])
    def test_identical_offline_and_unknown_no_zero_fill(self):
        r=record('same','same');self.assertEqual(p.choose_jev(r,status='identical_answer_no_dispatch')['route'],'baseline')
        r=record()
        for status in ('failed','unknown','not_dispatched'):
            self.assertEqual(p.choose_jev(r,status=status)['route'],'baseline')
            with self.assertRaises(ValueError):p.choose_jev(r,{},status=status)
        r['identical']=True
        with self.assertRaises(ValueError):p.choose_jev(r,status='unknown')
    def test_response_confidence_record_only(self):
        r=record();raw=response();raw['answers']['answer_choice']['confidence']=0
        parsed=p.parse_choice(raw,r['signature']);self.assertFalse(parsed['tied_max'])
        self.assertEqual(p.choose_jev(r,parsed)['reason'],'first_better')

class OfflineTests(unittest.TestCase):
    def test_actual_imported_replay_contracts(self):
        self.assertTrue(callable(prep.READER_LEDGER))
        self.assertIs(prep.READER_LEDGER.__globals__['runner'],prep.qa)
        self.assertTrue(callable(prep.audit.original.existing.bound))
        self.assertTrue(callable(prep.qa.Gemini38Replay))
        self.assertTrue(callable(prep.qa.identity))
        self.assertTrue(callable(prep.qa.stage_files))
    def test_published_proof_keeps_sha_binding_without_global_seal_exception(self):
        # Actual immutable publication convention; do not mutate the old artifact.
        reader=prep.Reader();m=reader.read(prep.ROOT/'runs/evidence_selection_qa_r03_v1/run_manifest.json',sealed=False)
        proof=prep._published_reuse_proof(reader,m)
        self.assertEqual(proof,prep.qa.proof_for(m))
        self.assertEqual(proof['schema'],prep.qa.SCHEMA+'-reuse-proof')
        with self.assertRaises(ValueError):reader.bound(m['baseline_reuse_proof'],sealed=True)
    def test_reader_signature_no_gold(self):
        from tools.run_evidence_exchange_qa_r01 import request_signature
        j={'question':'What happened?','reference':'SECRET_GOLD','category':4,
           'reference_date':'2020-01-01','selected_facts':facts(20)}
        self.assertEqual(prep._reader_signature(j),request_signature(j,'reader'))
        self.assertNotIn('SECRET_GOLD',json.dumps(prep._reader_signature(j)))
    def test_same_cached_bytes_can_still_enforce_sealing_and_sha(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.json';path.write_text('{"x":1}',encoding='utf-8');reader=prep.Reader()
            reader.read(path,sealed=False)
            with self.assertRaises(ValueError):reader.read(path,sealed=True)
            with self.assertRaises(ValueError):reader.raw_pointer(path,ref={'path':str(path.resolve()),'sha256':'bad'})
    def test_snapshot_no_repeat_byte_reads(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.json';path.write_text('{"x":1}',encoding='utf-8');reader=prep.Reader();snapshot=prep.Snapshot(reader)
            with patch.object(Path,'read_bytes',autospec=True,wraps=None) as read:
                read.return_value=b'{"x":1}';snapshot.read(path);snapshot.sha(path);reader.raw_pointer(path)
                self.assertEqual(read.call_count,1)

if __name__=='__main__':unittest.main()
