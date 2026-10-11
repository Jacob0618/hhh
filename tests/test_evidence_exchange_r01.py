"""Network-free contract tests for bounded pools, full swaps and score provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem import evidence_exchange_r01 as x
from tools import prepare_evidence_exchange_r01 as prep


def fixture():
    facts = [{'id':f'f{i:03}', 'text':f'neutral item {i}', 'observation_date':'2023-01-01'} for i in range(120)]
    for i in range(20,120):
        facts[i]['text'] = f'special relation marker {i}'
    facts[19]['text'] = 'neutral shortest'
    return facts, [f['id'] for f in facts[:20]], [f['id'] for f in facts[:80]]


class PoolAndSelection(unittest.TestCase):
    def setUp(self):
        self.facts, self.base, self.native = fixture()
        self.pool = x.build_pool('special relation', self.facts, self.base, self.native)
        self.proposals = x.propose_exchanges(self.pool)

    def test_outside_means_entire_original_native80_and_full_scan(self):
        self.assertEqual(120,self.pool['bank_scan_count'])
        self.assertEqual(1,self.pool['bm25_query_count'])
        self.assertEqual(80,len(self.pool['pool_ids']))
        self.assertEqual(20,len(self.pool['independent_ids']))
        self.assertTrue(set(self.pool['independent_ids']).isdisjoint(self.native))
        self.assertEqual(self.base,self.pool['pool_ids'][:20])
        self.assertEqual(self.native[20:60],self.pool['native_other_ids'])

    def test_no_lexical_hits_native_fill_and_no_proposal(self):
        pool = x.build_pool('absent_z9',self.facts,self.base,self.native)
        self.assertEqual([],pool['independent_ids'])
        self.assertEqual(self.native,pool['pool_ids'])
        self.assertEqual([],x.propose_exchanges(pool))
        self.assertEqual(self.base,x.choose_jev(pool,[],None)['selected_ids'])
        self.assertIsNone(x.build_request('absent_z9','2023-01-01',self.facts,pool,[]))

    def test_duplicate_wrong_bank_or_baseline_not_native_refused(self):
        for ids in [self.native+['x'], self.native[:-1]+[self.native[0]], self.native[1:]]:
            with self.assertRaises(ValueError):
                x.build_pool('special',self.facts,self.base,ids)
        with self.assertRaises(ValueError):
            x.build_pool('special',self.facts+[self.facts[0]],self.base,self.native)

    def test_only_tail_donors_once_and_preserves_other_order(self):
        self.assertEqual(4,len(self.proposals))
        self.assertTrue(all(p['removed_id'] in self.base[-4:] for p in self.proposals))
        for p in self.proposals:
            self.assertEqual(20,len(set(p['selected_ids'])))
            self.assertEqual([i for i in self.base if i != p['removed_id']],
                [i for i in p['selected_ids'] if i != p['added_id']])
            self.assertEqual(p['added_id'],p['selected_ids'][p['slot']])
        chosen = x.choose_cheap(self.pool,self.proposals)
        self.assertEqual(1,chosen['swaps'])
        self.assertEqual('p0',chosen['accepted_proposal_id'])

    def test_exact_thresholds_first_pass_and_not_multiple_acceptance(self):
        scores = {f'p{i}_{label}': (.80 if label == 'G' else .10)
            for i in range(4) for label in 'GLX'}
        decision = x.choose_jev(self.pool,self.proposals,scores)
        self.assertEqual('p0',decision['accepted_proposal_id'])
        scores['p0_L'] = .100001
        self.assertEqual('p1',x.choose_jev(self.pool,self.proposals,scores)['accepted_proposal_id'])
        self.assertEqual(1,decision['swaps'])

    def test_unknown_not_zero_and_partial_nonfinite_bool_refused(self):
        d = x.choose_jev(self.pool,self.proposals,None,score_status='unknown')
        self.assertEqual(self.base,d['selected_ids'])
        self.assertEqual('unknown',d['score_status'])
        for invalid in ({}, {'p0_G':.9}, {f'p{i}_{l}':float('nan') for i in range(4) for l in 'GLX'},
            {f'p{i}_{l}':True for i in range(4) for l in 'GLX'}):
            with self.assertRaises(ValueError):
                x.choose_jev(self.pool,self.proposals,invalid)
        with self.assertRaises(ValueError):
            x.choose_jev(self.pool,self.proposals,{},score_status='unknown')

    def test_request_one_shared_state_no_gold_and_id_not_used_as_instruction(self):
        sig = x.build_request('special relation','2023-01-01',self.facts,self.pool,self.proposals)
        self.assertEqual(12,len(sig['body']['questions']))
        self.assertEqual(24,len(sig['body']['state']['facts']))
        self.assertEqual({'text','reference_date'},set(sig['body']['state']['query']))
        self.assertTrue(all('state.proposals[' in q['instructions'] for q in sig['body']['questions'].values()))
        reference_hash = hashlib.sha256(json.dumps(sig,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(reference_hash,x.request_hash(sig))
        self.assertEqual(131072,x.POLICY['signature_utf8_byte_cap'])

    def test_full_proposal_cannot_be_silently_modified(self):
        bad = deepcopy(self.proposals)
        bad[0]['selected_ids'][0] = bad[0]['added_id']
        with self.assertRaises(ValueError):
            x.choose_cheap(self.pool,bad)

    def test_raw_parser_preserves_low_yes_and_rejects_missing_null(self):
        sig = x.build_request('special relation','2023-01-01',self.facts,self.pool,self.proposals)
        answer = {'model':'jev-1.13.0','answers':{key:{'type':'noul','noul':.01} for key in sig['body']['questions']}}
        self.assertEqual(.01,x.parse_scores(answer,sig)['p0_G'])
        for value in (None,True,float('inf')):
            bad = deepcopy(answer)
            bad['answers']['p0_G']['noul'] = value
            with self.assertRaises(ValueError):
                x.parse_scores(bad,sig)
        bad = deepcopy(answer)
        del bad['answers']['p0_X']
        with self.assertRaises(ValueError):
            x.parse_scores(bad,sig)
        for model in (None,'jev-other'):
            bad = deepcopy(answer)
            bad['model'] = model
            with self.assertRaises(ValueError):
                x.parse_scores(bad,sig)
        bad = deepcopy(answer)
        bad['answers']['p0_G']['type'] = 'choice'
        with self.assertRaises(ValueError):
            x.parse_scores(bad,sig)

    def test_token_sensitivity_filters_action_without_new_state_or_score(self):
        fs = deepcopy(self.facts)
        for fact in fs[20:]:
            fact['text'] += ' very long '*200
        serializer = lambda q,f,d: json.dumps({'q':q,'facts':f,'date':d})
        result = x.token_eligible_proposals('special relation','2023-01-01',fs,self.pool,self.proposals,len,serializer)
        self.assertEqual([],result['eligible_proposal_ids'])
        self.assertEqual(0,result['new_jev_requests'])
        scores = {f'p{i}_{l}': (.9 if l == 'G' else .01) for i in range(4) for l in 'GLX'}
        self.assertEqual(self.base,x.choose_jev(self.pool,self.proposals,scores,
            eligible_proposal_ids=result['eligible_proposal_ids'])['selected_ids'])


class ArtifactBinding(unittest.TestCase):
    def test_single_read_byte_hash_content_hash_and_new_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'x.json'
            ref = prep.save_new(p,{'hello':'世界'})
            reader = prep.Reader()
            self.assertEqual('世界',reader.bound(ref)['hello'])
            self.assertEqual('世界',reader.bound(ref)['hello'])
            self.assertEqual(1,len(reader.rows))
            with self.assertRaises(FileExistsError):
                prep.save_new(p,{'hello':'overwrite'})
            bad = dict(ref,sha256='0'*64)
            with self.assertRaises(ValueError):
                reader.bound(bad)

    def test_raw_exact_signature_required_and_unavailable_no_scores(self):
        facts,base,native = fixture()
        pool = x.build_pool('special',facts,base,native)
        proposals = x.propose_exchanges(pool)
        sig = x.build_request('special','2023-01-01',facts,pool,proposals)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            reqref = prep.save_new(run/'requests/c0_q0.json', {'signature':sig,
                'request_hash':x.request_hash(sig),'dispatch_eligible':True})
            v = {'request':reqref,'query_id':'c0_q0','protocol_sha256':'proto'}
            response = {'model':'jev-1.13.0','answers':{key:{'type':'noul','noul':.5} for key in sig['body']['questions']}}
            raw = run/'api/raw.json'
            raw.parent.mkdir()
            raw.write_text(json.dumps({'request':sig,'response':response}),encoding='utf-8')
            rawref = {'path':str(raw.resolve()),'sha256':prep.file_sha(raw)}
            prep.save_new(run/'responses/c0_q0.json',{'schema':prep.SCORE_SCHEMA,'query_id':'c0_q0',
                'protocol_sha256':'proto','request_hash':x.request_hash(sig),'status':'complete','raw':rawref})
            scores,status,_,_ = prep._score(prep.Reader(),run,v)
            self.assertEqual('complete',status)
            self.assertEqual(12,len(scores))
            mutated = deepcopy(sig)
            mutated['body']['state']['query']['text'] = 'other'
            raw.write_text(json.dumps({'request':mutated,'response':response}),encoding='utf-8')
            with self.assertRaises(ValueError):
                prep._score(prep.Reader(),run,v)

    def test_missing_wrapper_not_fabricated_terminal_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            sig = {'provider':'jev','model':'jev-1.13.0','operation':'systemone','body':{}}
            reqref = prep.save_new(run/'request.json',{'signature':sig,'request_hash':x.request_hash(sig)})
            v = {'request':reqref,'query_id':'c0_q0','protocol_sha256':'p'}
            with self.assertRaisesRegex(ValueError,'Missing result'):
                prep._score(prep.Reader(),run,v)
            prep.save_new(run/'responses/c0_q0.json',{'schema':prep.SCORE_SCHEMA,'query_id':'c0_q0',
                'protocol_sha256':'p','request_hash':x.request_hash(sig),'status':'unknown','scores':None,'raw':None})
            scores,status,_,_ = prep._score(prep.Reader(),run,v)
            self.assertIsNone(scores)
            self.assertEqual('unknown',status)

    def test_pilot_is_group_balanced_order_invariant_not_outcome_selected(self):
        jobs = [{'conversation':ci,'query_id':f'c{ci}_q{q}','judge':q%2} for ci in range(10) for q in range(16)]
        ids = x.pilot_ids(jobs)
        self.assertEqual(80,len(ids))
        self.assertEqual(ids,x.pilot_ids(list(reversed(jobs))))
        for ci in range(10):
            self.assertEqual(8,len([q for q in ids if q.startswith(f'c{ci}_')]))


if __name__ == '__main__':
    unittest.main()
