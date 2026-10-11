"""Offline R03 tests: member choice, frozenpool ordering, complete strict scores."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem import evidence_selection_r03 as x
from tools import prepare_evidence_selection_r03 as prep


def fixture(n=80):
    return [{'id':f'id{i}','text':f'neutral item {i}','observation_date':None}for i in range(n)]


class Membership(unittest.TestCase):
    def setUp(self):
        self.records=fixture();self.spec=json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))

    def test_protocol_predicate_exact_R02_and_pinned_finalv2(self):
        self.assertEqual(x.PROTOCOL_SHA256,prep.file_sha(prep.RESEARCH_PROTOCOL))
        r02=json.loads((ROOT/'research/jev_autoresearch_2026-10-04/R02_ordering_protocol.json').read_text(encoding='utf8'))
        self.assertEqual(r02['jev']['question_instructions_template'],self.spec['jev']['question_instructions_template'])

    def test_pool_count_unique_reader_record_shape(self):
        for n in (20,37,80):x.validate_records(fixture(n))
        for bad in (fixture(19),fixture(81),self.records[:-1]+[self.records[0]]):
            with self.assertRaises(ValueError):x.validate_records(bad)
        bad=deepcopy(self.records);bad[0]['gold']='forbiddenextra'
        with self.assertRaises(ValueError):x.validate_records(bad)

    def test_take_score_members_then_restore_poolorder(self):
        scores={f'f{i:02d}':i/100 for i in range(80)}
        decision=x.choose_jev(self.records,scores)
        self.assertEqual(list(reversed(range(60,80))),decision['score_ranked_member_indices'])
        self.assertEqual(list(range(60,80)),decision['pool_selected_indices'])
        self.assertEqual([f'id{i}'for i in range(60,80)],decision['selected_ids'])
        self.assertTrue(decision['changed_membership']);self.assertEqual(0,decision['retained_baseline_count'])

    def test_same_members_cannot_create_extra_order_intervention(self):
        scores={f'f{i:02d}':(i/100+.5 if i<20 else .01)for i in range(80)}
        decision=x.choose_jev(self.records,scores)
        self.assertEqual(list(range(20)),decision['pool_selected_indices'])
        self.assertFalse(decision['changed_membership']);self.assertFalse(decision['changed_order'])
        self.assertEqual(list(reversed(range(20))),decision['score_ranked_member_indices'])

    def test_tie_at_boundary_uses_frozenpoolindex_and_zero_valid(self):
        scores={f'f{i:02d}':0. for i in range(80)}
        d=x.choose_jev(self.records,scores)
        self.assertEqual(list(range(20)),d['pool_selected_indices'])
        self.assertEqual(3160,d['stable_tie_pairs'])
        scores['f79']=.001
        d=x.choose_jev(self.records,scores)
        self.assertEqual([*range(19),79],d['pool_selected_indices'])

    def test_any_unavailable_fullS0_and_never_zero_fill(self):
        for status in ('unknown','failed','not_dispatched'):
            d=x.choose_jev(self.records,None,score_status=status)
            self.assertEqual(list(range(20)),d['pool_selected_indices'])
            with self.assertRaises(ValueError):x.choose_jev(self.records,{},score_status=status)
        bad={f'f{i:02d}':.5 for i in range(79)}
        with self.assertRaises(ValueError):x.choose_jev(self.records,bad)

    def test_bool_null_range_nonfinite_are_not_valid_probabilities(self):
        for value in (True,None,float('nan'),float('inf'),-.1,1.1):
            bad={f'f{i:02d}':.5 for i in range(80)};bad['f79']=value
            with self.assertRaises(ValueError):x.choose_jev(self.records,bad)

    def test_cheap_entire_frozenpool_same_fields_fresh_scores(self):
        self.records[25]['observation_date']='2023-01-02'
        values=[0.]*80;values[25]=1.;values[79]=2.
        with patch.object(x,'stable_bm25',return_value=values)as scorer:
            d=x.choose_cheap('query',None,self.records)
        query,docs=scorer.call_args.args
        self.assertEqual('query\nReference date: ',query);self.assertEqual(80,len(docs))
        self.assertEqual('neutral item 25\nObservation date: 2023-01-02',docs[25])
        self.assertEqual([*range(18),25,79],d['pool_selected_indices'])
        self.assertIsNone(self.records[0]['observation_date'])

    def test_selected_records_preserved_not_mutated(self):
        ids=[*range(19),79];selected=x.selected_facts(self.records,ids)
        self.assertEqual([self.records[i]for i in ids],selected)
        selected[-1]['text']='mutated';self.assertEqual('neutral item 79',self.records[-1]['text'])
        for bad in (list(reversed(range(20))),[0]*20,[True,*range(1,20)],list(range(19))):
            with self.assertRaises(ValueError):x.selected_facts(self.records,bad)

    def test_one_batch_alln_fullrecord_no_ids_gold_or_priorp(self):
        sig=x.build_request('question',None,self.records,self.spec)
        self.assertEqual({'query'},set(sig['body']['state']))
        self.assertEqual([f'f{i:02d}'for i in range(80)],list(sig['body']['questions']))
        for i,q in enumerate(sig['body']['questions'].values()):
            literal=json.loads(q['instructions'].split('Memory record (JSON):\n')[1])
            self.assertEqual({'text':self.records[i]['text'],'observation_date':''},literal)
            self.assertNotIn('id',literal);self.assertEqual('noul',q['type'])
        self.assertEqual(x.request_hash(sig),__import__('hashlib').sha256(x.serialized_signature(sig)).hexdigest())
        sig20=x.build_request('question',None,self.records[:20],self.spec)
        self.assertNotEqual(x.request_hash(sig),x.request_hash(sig20))
        self.assertEqual(sig20['body']['questions']['f00'],sig['body']['questions']['f00'])

    def test_strict_model_type_and_alln_parser(self):
        sig=x.build_request('question',None,self.records,self.spec)
        response={'model':'jev-1.13.0','answers':{f'f{i:02d}':{'type':'noul','noul':.5}for i in range(80)}}
        self.assertEqual(80,len(x.parse_scores(response,sig)))
        for change in ('model','type','partial'):
            bad=deepcopy(response)
            if change=='model':bad['model']='other'
            if change=='type':bad['answers']['f79']['type']='choice'
            if change=='partial':del bad['answers']['f79']
            with self.assertRaises(ValueError):x.parse_scores(bad,sig)


class Provenance(unittest.TestCase):
    def test_actual_R01_pool_digest_path_first20_source(self):
        reader=prep.Reader();spec,_=prep._research_protocol(reader)
        manifest=reader.bound(prep._canonical_ref(spec['basis']['r01_pool_manifest']))
        item=manifest['items'][0];proposal=reader.bound(item['proposal'])
        self.assertEqual(proposal['pool_sha256'],x.digest(proposal['pool']))
        self.assertNotIn('pool_sha256',proposal['pool'])
        self.assertEqual(proposal['original_job']['selected_ids'],proposal['pool']['pool_ids'][:20])
        self.assertTrue(set(proposal['pool']['independent_ids']).isdisjoint(proposal['pool']['original_native80_ids']))

    def test_unknown_wrapper_bound_no_scores_and_missing_is_not_terminal(self):
        records=fixture(20);spec=json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))
        sig=x.build_request('query',None,records,spec)
        with tempfile.TemporaryDirectory()as tmp:
            run=Path(tmp);ref=prep.save_new(run/'requests/c0_q0.json',{'signature':sig,'request_hash':x.request_hash(sig)})
            v={'query_id':'c0_q0','protocol_sha256':'p','request':ref}
            with self.assertRaisesRegex(ValueError,'Missing result'):prep._score(prep.Reader(),run,v)
            prep.save_new(run/'responses/c0_q0.json',{'schema':prep.SCORE_SCHEMA,'query_id':'c0_q0',
                'protocol_sha256':'p','request_hash':x.request_hash(sig),'status':'unknown','scores':None,'raw':None})
            scores,status,_,_=prep._score(prep.Reader(),run,v)
            self.assertIsNone(scores);self.assertEqual('unknown',status)

    def test_complete_raw_response_binding_expanded80(self):
        records=fixture();spec=json.loads(prep.RESEARCH_PROTOCOL.read_text(encoding='utf8'))
        sig=x.build_request('query',None,records,spec)
        with tempfile.TemporaryDirectory()as tmp:
            run=Path(tmp);ref=prep.save_new(run/'requests/c0_q0.json',{'signature':sig,'request_hash':x.request_hash(sig),'dispatch_eligible':True})
            v={'query_id':'c0_q0','protocol_sha256':'p','request':ref}
            response={'model':'jev-1.13.0','answers':{f'f{i:02d}':{'type':'noul','noul':.4}for i in range(80)}}
            raw=run/'raw.json';raw.write_text(json.dumps({'request':sig,'response':response}),encoding='utf8')
            prep.save_new(run/'responses/c0_q0.json',{'schema':prep.SCORE_SCHEMA,'query_id':'c0_q0','protocol_sha256':'p',
                'request_hash':x.request_hash(sig),'status':'complete','scores':None,'raw':{'path':str(raw.resolve()),'sha256':prep.file_sha(raw)}})
            scores,status,_,_=prep._score(prep.Reader(),run,v)
            self.assertEqual('complete',status);self.assertEqual(80,len(scores))
            bad=deepcopy(sig);bad['body']['questions']['f79']['instructions']='wrongrecord'
            raw.write_text(json.dumps({'request':bad,'response':response}),encoding='utf8')
            with self.assertRaises(ValueError):prep._score(prep.Reader(),run,v)


if __name__=='__main__':unittest.main()
