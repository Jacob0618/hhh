import unittest
from copy import deepcopy
from jev_mem import r05_refinement_v1 as core
from jev_mem import source_support_stack_v1 as stack
from jev_mem import source_append_r05 as old
from jev_mem.evidence_exchange_r01 import request_hash, digest


def fixture(n=14):
    inputs={'envelopes':[{'session_id':'session_1','observation_date':'2023-06-01',
        'source_turn_ids':[f'D1:{i}' for i in range(n)],
        'messages':[{'content':f'Alice: boots hiking item {i}; complete source sentence.'} for i in range(n)]}]}
    turns=old.source_turns(inputs,{'path':'fixture.json','sha256':'a'*64})
    records=[{'id':f'f{i}','text':f'Alice likes boots hiking item {i}.',
        'observation_date':'2023-06-01'} for i in range(20)]
    scope={'session_id':'session_1','source_turn_ids':[t['source_turn_id'] for t in turns],
        'historical_turn_ids':['never-admitted'],'existing_memory_ids':[]}
    return records,[deepcopy(scope) for _ in records],turns


def support(records,scopes,turns,value=.9):
    result={}
    for fact,scope in zip(records,scopes):
        plan=stack.build_support_plan(fact,scope,turns,conversation=0)
        result[fact['id']]={'plan':plan,'scores':{k:value for r in plan['requests']
            for k in r['key_to_window_index']},'status':'complete'}
    return result


class RefinementTest(unittest.TestCase):
    def setUp(self):
        self.records,self.scopes,self.turns=fixture()
        self.support=support(self.records,self.scopes,self.turns)

    def build(self,**kwargs):
        return core.build_candidate_variants('boots hiking',self.records,self.scopes,
            self.turns,conversation=0,support_results=kwargs.pop('support_results',self.support),**kwargs)

    def scores(self,p,value=.9):return {pair['key']:value for pair in p['pairs']}

    def test_original_admission_preserved_and_no_eager_replacement(self):
        got=self.build()
        original=old.build_candidates('boots hiking',self.records,self.scopes,self.turns,conversation=0)
        self.assertEqual(got['original_locator'],original)
        p=got['multi_plan'];self.assertEqual(p['original_locator'],original)
        self.assertEqual(len(original['candidates']),4)
        self.assertEqual(len(p['pairs']),4)
        self.assertEqual(p['variant_count'],8)
        self.assertEqual([x['anchor_slot'] for x in p['pairs']],
            [x['origin']['anchor_slot'] for x in original['candidates']])
        self.assertEqual(len(p['requests']),1)
        self.assertEqual(len(p['requests'][0]['signature']['body']['questions']),4)

    def test_support_plan_binding_and_complete_all_batches(self):
        for mutate in ('scope','scores','unknown'):
            bad=deepcopy(self.support);fid=self.build()['multi_plan']['pairs'][0]['original']['origin']['anchor_id']
            if mutate=='scope':bad[fid]['plan']['scope_sha256']='different'
            if mutate=='scores':bad[fid]['scores'].pop('s07')
            if mutate=='unknown':bad[fid].update(scores=None,status='unknown')
            with self.assertRaises(ValueError):self.build(support_results=bad)
        for value in (True,float('nan'),float('inf'),-.1,1.1,'0.9'):
            bad=deepcopy(self.support)
            for x in bad.values():x['scores']['s00']=value
            with self.assertRaises(ValueError):self.build(support_results=bad)

    def test_support_threshold_absence_and_known_failure_keep_original(self):
        for supplied in ({},support(self.records,self.scopes,self.turns,.5),
            {k:{'plan':v['plan'],'scores':None,'status':'failed'} for k,v in self.support.items()}):
            got=self.build(support_results=supplied);p=got['multi_plan']
            self.assertEqual(p['pairs'],[]);self.assertEqual(p['requests'],[])
            r=core.resolve_multi(p,{},score_status='complete')
            self.assertEqual(r['locator']['candidates'],got['original_locator']['candidates'])

    def test_whole_window_bm25_all_windows_and_stable_ties(self):
        got=self.build();p=got['multi_plan']
        for pair in p['pairs']:
            plan=self.support[self.records[pair['anchor_slot']]['id']]['plan']
            byid=stack._turns(self.turns)
            docs=['\n'.join(old.document(byid[t['source_turn_id']]) for t in w['packet_view']['turns'])
                for w in plan['windows']]
            qs=old.stable_bm25('boots hiking',docs)
            self.assertEqual(p['diagnostics'][pair['candidate_index']]['window_query_bm25'],qs)
            eligible=[j for j,w in enumerate(plan['windows']) if w['packet_id']!=pair['original']['packet_id']]
            winner=min(eligible,key=lambda j:(-qs[j],-.9,plan['windows'][j]['origin']['center_global_index']))
            self.assertEqual(pair['support_window_index'],winner)
            self.assertEqual(pair['whole_window_query_bm25'],qs[winner])

    def test_zero_question_window_score_not_extra_admission_gate(self):
        # Explicitly force the whole-window indexer to yield zero; original center
        # BM25 remains untouched. Zero-score alternatives still reach comparison.
        original_fn=old.stable_bm25
        def zero_windows(q,docs):
            if docs and '\n' in docs[0]:return [0.]*len(docs)
            return original_fn(q,docs)
        old.stable_bm25=zero_windows
        try:
            got=self.build();self.assertEqual(len(got['multi_plan']['pairs']),4)
            self.assertTrue(all(p['whole_window_query_bm25']==0 for p in got['multi_plan']['pairs']))
        finally:old.stable_bm25=original_fn

    def test_comparison_threshold_and_unavailable_no_score_imputation(self):
        p=self.build()['multi_plan']
        for value in (0.,.5):
            r=core.resolve_multi(p,self.scores(p,value))
            self.assertTrue(all(not x['replaced'] for x in r['decisions']))
            self.assertEqual(r['locator']['candidates'],p['original_locator']['candidates'])
        r=core.resolve_multi(p,self.scores(p,.50001))
        self.assertTrue(all(x['replaced'] for x in r['decisions']))
        failed=core.resolve_multi(p,None,score_status='failed')
        self.assertEqual(failed['locator']['candidates'],p['original_locator']['candidates'])
        self.assertTrue(all(x['comparison_probability'] is None for x in failed['decisions']))
        with self.assertRaises(ValueError):core.resolve_multi(p,None,score_status='unknown')
        with self.assertRaises(ValueError):core.resolve_multi(p,self.scores(p),score_status='failed')
        partial=self.scores(p);partial.pop(next(iter(partial)))
        with self.assertRaises(ValueError):core.resolve_multi(p,partial)
        for value in (True,float('nan'),float('inf'),-.1,1.1,'0.9'):
            with self.assertRaises(ValueError):core.resolve_multi(p,self.scores(p,value))

    def test_request_raw_contract_hash_and_gold_whitelist(self):
        req=self.build()['multi_plan']['requests'][0];sig=req['signature']
        self.assertEqual(core.validate_request(sig),'multi_compare')
        self.assertEqual(req['request_hash'],request_hash(sig))
        self.assertEqual(req['signature_sha256'],digest(sig))
        answer={'model':core.MODEL,'answers':{k:{'type':'noul','noul':.8} for k in sig['body']['questions']}}
        self.assertEqual(core.parse_response(answer,sig),{k:.8 for k in answer['answers']})
        for mutate in ('model','missing','bool','type'):
            bad=deepcopy(answer);first=next(iter(bad['answers']))
            if mutate=='model':bad['model']='other'
            if mutate=='missing':bad['answers'].pop(first)
            if mutate=='bool':bad['answers'][first]['noul']=True
            if mutate=='type':bad['answers'][first]['type']='choice'
            with self.assertRaises(ValueError):core.parse_response(bad,sig)
        bad=deepcopy(sig);bad['body']['state']['gold']='forbidden'
        with self.assertRaises(ValueError):core.validate_request(bad)
        bad=deepcopy(sig);first=next(iter(bad['body']['questions']))
        bad['body']['questions'][first]['instructions']+='\nGold: forbidden'
        with self.assertRaises(ValueError):core.validate_request(bad)

    def test_oversize_is_not_dispatched_without_text_truncation(self):
        question='boots hiking '+('x'*140000)
        got=core.build_candidate_variants(question,self.records,self.scopes,self.turns,
            conversation=0,support_results=self.support)
        p=got['multi_plan'];req=p['requests'][0]
        self.assertFalse(req['dispatch_eligible'])
        self.assertEqual(req['skip_reason'],'signature_oversize')
        self.assertEqual(req['signature']['body']['state']['query']['text'],question)
        core.validate_request(req['signature'],allow_oversize=True)
        with self.assertRaises(ValueError):core.validate_request(req['signature'])
        with self.assertRaises(ValueError):core.resolve_multi(p,self.scores(p))
        r=core.resolve_multi(p,None,score_status='not_dispatched')
        self.assertEqual(r['locator']['candidates'],got['original_locator']['candidates'])
        self.assertTrue(all(d['comparison_probability'] is None for d in r['decisions']))

    def test_scoped_whole_packets_distinct_consuming_anchor_and_collision_policy(self):
        p=self.build()['multi_plan'];byid=stack._turns(self.turns)
        for pair in p['pairs']:
            c=pair['alternative'];slot=pair['anchor_slot']
            self.assertLessEqual(c['attachment_utf8_bytes'],4096)
            self.assertEqual(c['attachment'],old.attachment(c['packet_view']))
            self.assertTrue(all(t['source_turn_id'] in self.scopes[slot]['source_turn_ids']
                for t in c['packet_view']['turns']))
            self.assertEqual(c['origin']['anchor_slot'],slot)
        r=core.resolve_multi(p,self.scores(p))
        cs=r['locator']['candidates']
        self.assertEqual(len({c['packet_id'] for c in cs}),len(cs))
        self.assertEqual(len({c['origin']['anchor_slot'] for c in cs}),len(cs))
        # Our symmetric fixture can send multiple anchors to the same window.
        self.assertTrue(r['locator']['multi_refinement']['collisions'])
        self.assertTrue(r['locator']['multi_refinement']['no_outside_refill'])

    def test_frozen_closure_rank_render_and_original20_prefix(self):
        got=self.build();p=got['multi_plan'];r=core.resolve_multi(p,self.scores(p))
        loc=r['locator'];plans=[];resolved=[]
        for c in loc['candidates']:
            plan=core.build_closure_plan('boots hiking',None,self.records,c,
                self.scopes[c['origin']['anchor_slot']],self.turns,conversation=0)
            plans.append(plan)
            resolved.append(core.resolve_closure(plan,{k:.5 for req in plan['requests'] for k in req['key_to_window_index']}))
        closed=core.apply_closures(loc,resolved,self.records,closure_plans=plans)
        cs=closed['candidates'];sig=core.build_rank_request('boots hiking',None,self.records,cs)
        self.assertEqual(sig,stack.build_rank_request('boots hiking',None,self.records,cs))
        scores={f'f{i:02d}':.9 for i in range(len(cs))} if len(cs)>2 else None
        decision=core.choose_rank(self.records,cs,scores,score_status='complete' if scores else 'not_needed')
        rendered=core.selected_facts(self.records,cs,decision)
        self.assertEqual(len(rendered),20)
        for original,new in zip(self.records,rendered):
            self.assertEqual(original['id'],new['id'])
            self.assertEqual(original['observation_date'],new['observation_date'])
            self.assertTrue(new['text'].startswith(original['text']))
        self.assertLessEqual(len(decision['candidate_indices']),2)
        self.assertIs(core.build_rank_request,stack.build_rank_request)
        self.assertIs(core.selected_facts,stack.selected_facts)


if __name__=='__main__':unittest.main()
