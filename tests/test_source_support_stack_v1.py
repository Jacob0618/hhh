import unittest
from copy import deepcopy
import json
from pathlib import Path
from jev_mem import source_support_stack_v1 as core
from jev_mem import source_append_r05 as old


def fixture(n=12):
    inputs={'envelopes':[{'session_id':'session_1','observation_date':'2023-06-01',
        'source_turn_ids':[f'D1:{i}' for i in range(n)],
        'messages':[{'content':f'Alice: boots hiking item {i}; complete source sentence.'} for i in range(n)]}]}
    turns=old.source_turns(inputs,{'path':'fixture.json','sha256':'a'*64})
    records=[{'id':f'f{i}','text':f'Alice likes boots hiking item {i}.','observation_date':'2023-06-01'} for i in range(20)]
    scope={'session_id':'session_1','source_turn_ids':[t['source_turn_id'] for t in turns],
        'historical_turn_ids':['never-admitted'],'existing_memory_ids':[]}
    return records,[deepcopy(scope) for _ in records],turns


def base(records,scope,turns,slot,center):
    return core._base_packet(records[slot],slot,turns[center],core._scope(scope,core._turns(turns)),turns,0,qscore=2.,fscore=3.)


def scores(plan,value=.9,prefix='s'):
    return {key:value for req in plan['requests'] for key in req['key_to_window_index']}


class SupportStackTest(unittest.TestCase):
    def setUp(self):self.records,self.scopes,self.turns=fixture()

    def test_support_batch_mapping_and_query_independence(self):
        p=core.build_support_plan(self.records[0],self.scopes[0],self.turns,conversation=0)
        self.assertEqual(len(p['windows']),8)
        self.assertEqual([len(x['signature']['body']['questions']) for x in p['requests']],[4,4])
        self.assertEqual(p['requests'][1]['key_to_window_index'],{f's{i:02d}':i for i in range(4,8)})
        for req in p['requests']:
            self.assertEqual(core.validate_request(req['signature']),'s')
            self.assertEqual(set(req['signature']['body']['state']),{'original_memory'})
        # There is no Q/other20 argument; changes to query-side records cannot enter S.
        other=deepcopy(self.records);other[1]['text']='Completely different other memory'
        self.assertEqual(p,core.build_support_plan(other[0],self.scopes[0],self.turns,conversation=0))
        self.assertEqual(p['scope']['source_turn_ids'],self.scopes[0]['source_turn_ids'])
        self.assertNotIn('never-admitted',old.canonical(p))

    def test_support_threshold_tie_and_granularity(self):
        p=core.build_support_plan(self.records[0],self.scopes[0],self.turns,conversation=0)
        d=core.resolve_support(p,scores(p,.5));self.assertEqual(d['support_label'],'unknown')
        d=core.resolve_support(p,scores(p,.9))
        expected=min(range(8),key=lambda i:p['windows'][i]['origin']['center_global_index'])
        self.assertEqual(d['chosen_window_index'],expected)
        self.assertEqual(d['support_label'],'partial_support_locator')
        self.assertFalse(d['certified_fact_support'])

    def test_no_partial_batch_or_invalid_scores(self):
        p=core.build_support_plan(self.records[0],self.scopes[0],self.turns,conversation=0)
        s=scores(p);s.pop('s07')
        with self.assertRaises(ValueError):core.resolve_support(p,s)
        for bad in (True,float('nan'),float('inf'),-.1,1.1,'0.9'):
            s=scores(p);s['s00']=bad
            with self.assertRaises(ValueError):core.resolve_support(p,s)
        with self.assertRaises(ValueError):core.resolve_support(p,scores(p),score_status='unknown')
        d=core.resolve_support(p,None,score_status='unknown');self.assertIsNone(d['center_turn_id'])
        self.assertIsNone(d['scores'])

    def test_response_contract_and_gold_whitelist(self):
        p=core.build_support_plan(self.records[0],self.scopes[0],self.turns,conversation=0)
        sig=p['requests'][0]['signature'];answer={'model':core.MODEL,'answers':{k:{'type':'noul','noul':.9} for k in sig['body']['questions']}}
        self.assertEqual(core.parse_response(answer,sig),{k:.9 for k in sig['body']['questions']})
        bad=deepcopy(answer);bad['model']='other'
        with self.assertRaises(ValueError):core.parse_response(bad,sig)
        bad=deepcopy(answer);bad['answers']['s00']['type']='choice'
        with self.assertRaises(ValueError):core.parse_response(bad,sig)
        bad=deepcopy(sig);bad['body']['state']['reference']='gold'
        with self.assertRaises(ValueError):core.parse_response(answer,bad)
        bad=deepcopy(self.records[0]);bad['reference']='gold'
        with self.assertRaises(ValueError):core.build_support_plan(bad,self.scopes[0],self.turns,conversation=0)

    def test_s0_exact_old_locator_and_sidecar_replay(self):
        old_locator=old.build_candidates('boots hiking',self.records,self.scopes,self.turns,conversation=0)
        got=core.build_candidates('boots hiking',self.records,self.scopes,self.turns,conversation=0)
        self.assertEqual({k:v for k,v in got.items() if k!='support_locator'},old_locator)
        p=core.build_support_plan(self.records[0],self.scopes[0],self.turns,conversation=0)
        d=core.resolve_support(p,scores(p))
        got=core.build_candidates('boots hiking',self.records,self.scopes,self.turns,conversation=0,
            enable_support=True,support_sidecars={'f0':d})
        self.assertEqual(got['corpus_turn_ids'],old_locator['corpus_turn_ids'])
        self.assertEqual(got['query_bm25_scores'],old_locator['query_bm25_scores'])
        self.assertEqual(got['anchor_bm25_scores'],old_locator['anchor_bm25_scores'])
        self.assertEqual(got['support_locator']['support_used_count'],1)
        bad=deepcopy(d);bad['source_sha256']='wrong'
        with self.assertRaises(ValueError):core.build_candidates('boots hiking',self.records,self.scopes,self.turns,
            conversation=0,enable_support=True,support_sidecars={'f0':bad})

    def test_support_center_zero_question_overlap_falls_back(self):
        # S genuinely prefers a boots support turn, but query bicycle cannot admit it.
        inputs={'envelopes':[{'session_id':'session_1','observation_date':'2023-06-01',
            'source_turn_ids':['a','b','c','d','e'],'messages':[{'content':x} for x in
            ['Alice: boots support','Alice: boots','Alice: neutral','Alice: bicycle boots','Alice: bicycle']]}]}
        turns=old.source_turns(inputs,{'path':'fixture.json','sha256':'b'*64})
        scope={'session_id':'session_1','source_turn_ids':['a','b','c','d','e']}
        p=core.build_support_plan(self.records[0],scope,turns,conversation=0)
        s=scores(p,.1);idx=next(i for i,w in enumerate(p['windows']) if w['origin']['center_turn_id']=='a');s[f's{idx:02d}']=.9
        d=core.resolve_support(p,s)
        got=core.build_candidates('bicycle',self.records,[scope]*20,turns,conversation=0,enable_support=True,support_sidecars={'f0':d})
        self.assertEqual(got['support_locator']['per_fact'][0]['reason'],'support_center_not_dual_positive_or_oversize_keep_original_locator')
        self.assertEqual(got['support_locator']['per_fact'][0]['chosen_center'],'e')
        self.assertEqual(got['support_locator']['per_fact'][0]['chosen_center'],got['support_locator']['per_fact'][0]['original_locator_center'])

    def test_closure_complete_turns_tie_and_no_oversize_clipping(self):
        c=base(self.records,self.scopes[0],self.turns,0,4)
        p=core.build_closure_plan('boots hiking',None,self.records,c,self.scopes[0],self.turns,conversation=0)
        self.assertEqual([v['name'] for v in p['variants']],['left','right','both'])
        self.assertEqual([len(v['candidate']['packet_view']['turns']) for v in p['variants']],[4,4,5])
        d=core.resolve_closure(p,scores(p,.8))
        self.assertEqual(d['diagnostic']['chosen_variant_index'],0)
        self.assertEqual(d['diagnostic']['added_turn_count'],1)
        self.assertEqual(d['candidate']['packet_view']['turns'][0]['speaker_text'],self.turns[2]['view']['speaker_text'])
        self.assertEqual(core.resolve_closure(p,scores(p,.5))['candidate'],c)
        self.assertEqual(core.resolve_closure(p,None,score_status='failed')['candidate'],c)
        heavy=deepcopy(self.turns);heavy[2]['view']['speaker_text']='Alice: '+('字'*2000);heavy[2]['projection']['speaker_text']=heavy[2]['view']['speaker_text']
        p=core.build_closure_plan('boots hiking',None,self.records,c,self.scopes[0],heavy,conversation=0)
        self.assertEqual([v['name'] for v in p['variants']],['right'])
        self.assertTrue(any(e['reason']=='whole_attachment_oversize' for e in p['exclusions']))

    def test_scope_holes_and_cross_session_are_never_jumped(self):
        scope={'session_id':'session_1','source_turn_ids':['D1:3','D1:4','D1:5','D1:7']}
        c=base(self.records,scope,self.turns,0,4)
        p=core.build_closure_plan('boots',None,self.records,c,scope,self.turns,conversation=0)
        self.assertEqual(p['variants'],[])
        self.assertEqual(core.resolve_closure(p,{},score_status='complete')['candidate'],c)
        bad={'session_id':'other_session','source_turn_ids':['D1:4']}
        with self.assertRaises(ValueError):core.build_support_plan(self.records[0],bad,self.turns,conversation=0)

    def test_closure_collision_keeps_first_and_no_refill(self):
        a=base(self.records,self.scopes[0],self.turns,0,1);b=base(self.records,self.scopes[1],self.turns,1,2)
        plans=[core.build_closure_plan('boots',None,self.records,c,self.scopes[i],self.turns,conversation=0) for i,c in enumerate((a,b))]
        # First right, second left converge to exactly turns0..3.
        results=[]
        for p,name in zip(plans,('right','left')):
            s=scores(p,.1);i=next(i for i,v in enumerate(p['variants']) if v['name']==name);s[f'c{i:02d}']=.9
            results.append(core.resolve_closure(p,s))
        self.assertEqual(results[0]['candidate']['packet_id'],results[1]['candidate']['packet_id'])
        original=deepcopy((a,b,plans,results,self.records))
        got=core.apply_closures({'candidates':[a,b]},results,self.records,closure_plans=plans)
        self.assertEqual(len(got['candidates']),1)
        self.assertEqual(got['candidates'][0]['origin']['anchor_slot'],0)
        self.assertEqual(len(got['candidates'][0]['origin_aliases']),2)
        self.assertEqual(got['closure']['pre_closure_candidate_count'],2)
        self.assertTrue(got['closure']['no_outside_refill'])
        self.assertEqual((a,b,plans,results,self.records),original)
        fake=deepcopy(results);fake[0]['candidate']['attachment']+='invented'
        with self.assertRaises(ValueError):core.apply_closures({'candidates':[a,b]},fake,self.records,closure_plans=plans)

    def test_rank_exact_old_predicate_and_original_prefix(self):
        original_protocol=Path(__file__).resolve().parents[1]/'research/jev_autoresearch_2026-10-04/R05_source_append_protocol_final_v1.json'
        self.assertEqual(core.LOCAL_PREDICATE,json.loads(original_protocol.read_text(encoding='utf-8'))['jev']['predicate'])
        cs=[base(self.records,self.scopes[i],self.turns,i,2+3*i) for i in range(3)]
        sig=core.build_rank_request('boots hiking',None,self.records,cs)
        self.assertEqual(sig,old.build_request('boots hiking',None,self.records,cs,{'jev':{'predicate':core.LOCAL_PREDICATE}}))
        d=core.choose_rank(self.records,cs,{'f00':.1,'f01':.2,'f02':.3})
        facts=core.selected_facts(self.records,cs,d)
        self.assertEqual(d['candidate_indices'],[2,1]);self.assertEqual(len(facts),20)
        for oldfact,newfact in zip(self.records,facts):
            self.assertEqual(newfact['id'],oldfact['id']);self.assertEqual(newfact['observation_date'],oldfact['observation_date']);self.assertTrue(newfact['text'].startswith(oldfact['text']))
        failed=core.choose_rank(self.records,cs,None,score_status='failed')
        self.assertEqual(core.selected_facts(self.records,cs,failed),self.records)
        self.assertIsNone(core.build_rank_request('boots',None,self.records,cs[:2]))
        self.assertEqual(core.choose_rank(self.records,cs[:2],None,score_status='not_dispatched')['candidate_indices'],[0,1])


if __name__=='__main__':unittest.main()
