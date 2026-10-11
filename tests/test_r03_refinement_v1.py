import unittest
from copy import deepcopy
from jev_mem import r03_refinement_v1 as core


def facts(n=80):
    return [{'id':f'id{i}', 'text':f'Alice explicitly reports event {i}.',
             'observation_date':'2023-06-09'} for i in range(n)]


def fixture(n=80):
    pool=facts(n); original=deepcopy(pool[:20])
    p=core.build_precision_plan('What did Alice do before June?', '2023-06-10', pool, original)
    # P20 is indices20..39 for n80; no threshold removes low-scoring memories.
    scores={f'f{i:02d}':(.4 if 20<=i<40 else .1) for i in range(n)}
    r=core.resolve_precision(p,scores)
    c=core.build_complement_plan(p,r)
    cs={f'c{i:02d}':(.9 if i>=40 else .2) for i in c['candidate_indices']}
    cr=core.resolve_complement(c,cs)
    z=core.build_set_plan(c,cr)
    return p,r,c,cr,z


class RefinementTest(unittest.TestCase):
    def test_fixed_batches_and_keys(self):
        p,r,c,cr,z=fixture()
        self.assertEqual([len(x['signature']['body']['questions']) for x in p['requests']],[32,32,16])
        self.assertEqual([len(x['signature']['body']['questions']) for x in c['requests']],[32,28])
        self.assertEqual(len(z['requests']),1)
        self.assertEqual(len(z['proposals']),4)
        self.assertEqual(len(c['candidate_indices']),60)
        self.assertFalse(set(c['candidate_indices'])&set(r['preliminary_indices']))

    def test_low_pointwise_scores_still_return_twenty(self):
        p,r,c,cr,z=fixture()
        self.assertEqual(r['preliminary_indices'],list(range(20,40)))
        self.assertEqual(r['ranked_indices'][:20],list(range(20,40)))
        for proposal in z['proposals']:
            self.assertEqual(len(proposal['selected_indices']),20)
            self.assertEqual(proposal['selected_indices'],sorted(set(proposal['selected_indices'])))

    def test_swap_donors_low_precision_late_index(self):
        _,r,_,cr,z=fixture()
        byk={x['swap_count_from_P']:x['selected_indices'] for x in z['proposals']}
        self.assertEqual(byk[0],list(range(20,40)))
        self.assertEqual(byk[1],list(range(20,39))+[40])
        self.assertEqual(byk[2],list(range(20,38))+[40,41])
        self.assertEqual(byk[4],list(range(20,36))+[40,41,42,43])

    def test_guard_threshold_tie_and_unique_accept(self):
        p,_,_,_,z=fixture(); keys=[x['key'] for x in z['proposals']]
        for vals in ({k:.5 for k in keys},{k:.9 for k in keys}):
            d=core.choose_final(z,vals)
            self.assertIsNone(d['accepted_proposal_key'])
            self.assertEqual(d['selected_indices'],list(range(20)))
        scores={k:.3 for k in keys};scores['p02']=.51
        d=core.choose_final(z,scores)
        self.assertEqual(d['accepted_proposal_key'],'p02')
        self.assertEqual(core.selected_facts(p['base']['pool'],d),
            [p['base']['pool'][i] for i in z['proposals'][2]['selected_indices']])
        scores['p01']=.51-5e-13
        self.assertIsNone(core.choose_final(z,scores)['accepted_proposal_key'])

    def test_precision_failure_propagates_to_whole_OJ(self):
        p,_,_,_,_=fixture();r=core.resolve_precision(p,None,score_status='failed')
        c=core.build_complement_plan(p,r);self.assertEqual(c['requests'],[])
        cr=core.resolve_complement(c,None,score_status='not_dispatched')
        z=core.build_set_plan(c,cr);self.assertEqual(z['requests'],[])
        d=core.choose_final(z,None,score_status='not_dispatched')
        self.assertEqual(d['selected_indices'],list(range(20)))
        self.assertEqual(d['reason'],'precision_unavailable_keep_OJ')
        self.assertEqual(d['score_status'],'failed')

    def test_complement_or_set_failure_keeps_OJ_not_P(self):
        p,r,c,_,_=fixture();cr=core.resolve_complement(c,None,score_status='failed')
        z=core.build_set_plan(c,cr)
        self.assertEqual(core.choose_final(z,None,score_status='not_dispatched')['selected_indices'],list(range(20)))
        *_,z=fixture()
        self.assertEqual(core.choose_final(z,None,score_status='failed')['selected_indices'],list(range(20)))

    def test_unknown_never_silent_fallback(self):
        p,_,c,_,z=fixture()
        for fn,plan in ((core.resolve_precision,p),(core.resolve_complement,c),(core.choose_final,z)):
            with self.assertRaises(ValueError):fn(plan,None,score_status='unknown')

    def test_invalid_and_partial_scores_fail(self):
        p,r,c,cr,z=fixture()
        for fn,plan,scores in ((core.resolve_precision,p,r['scores']),(core.resolve_complement,c,cr['scores']),
                (core.choose_final,z,{v['key']:.8 for v in z['proposals']})):
            bad=deepcopy(scores);bad.pop(next(iter(bad)))
            with self.assertRaises(ValueError):fn(plan,bad)
            for value in (True,float('nan'),float('inf'),-.1,1.1,'0.9'):
                bad=deepcopy(scores);bad[next(iter(bad))]=value
                with self.assertRaises(ValueError):fn(plan,bad)
            with self.assertRaises(ValueError):fn(plan,scores,score_status='failed')

    def test_exact_pool_and_original_identity(self):
        pool=facts();old=deepcopy(pool[:20]);old[0]['text']='Changed'
        with self.assertRaises(ValueError):core.build_precision_plan('Q','D',pool,old)
        with self.assertRaises(ValueError):core.build_precision_plan('Q','D',pool,list(reversed(pool[:20])))
        old=deepcopy(pool[:20]);old[0]['gold']='hidden'
        with self.assertRaises(ValueError):core.build_precision_plan('Q','D',pool,old)

    def test_duplicate_same_OJ_proposals_removed(self):
        pool=facts(20);p=core.build_precision_plan('Q',None,pool,pool)
        r=core.resolve_precision(p,{f'f{i:02d}':0. for i in range(20)})
        c=core.build_complement_plan(p,r);self.assertEqual(c['requests'],[])
        cr=core.resolve_complement(c,{})
        z=core.build_set_plan(c,cr)
        self.assertEqual(z['proposals'],[])
        self.assertEqual(core.choose_final(z,{})['reason'],'all_proposals_equal_OJ')

    def test_full_plan_replay_rejects_tampering(self):
        p,r,c,cr,z=fixture()
        bad=deepcopy(p);bad['requests'][0]['key_to_pool_index']['f00']=1
        with self.assertRaises(ValueError):core.resolve_precision(bad,r['scores'])
        bad=deepcopy(r);bad['preliminary_indices']=list(range(20))
        with self.assertRaises(ValueError):core.build_complement_plan(p,bad)
        bad=deepcopy(c);bad['candidate_indices'].reverse()
        with self.assertRaises(ValueError):core.resolve_complement(bad,cr['scores'])
        bad=deepcopy(z);bad['proposals'][0]['selected_indices']=list(range(20))
        with self.assertRaises(ValueError):core.choose_final(bad,{v['key']:.9 for v in z['proposals']})

    def test_request_schema_model_parser_and_forbidden_fields(self):
        p,_,c,_,z=fixture()
        for stage,plan in [('precision',p),('complement',c),('set',z)]:
            sig=plan['requests'][0]['signature']
            self.assertEqual(core.validate_request(sig),stage)
            raw={'model':core.MODEL,'answers':{k:{'type':'noul','noul':.8} for k in sig['body']['questions']}}
            self.assertEqual(set(core.parse_response(raw,sig)),set(sig['body']['questions']))
            bad=deepcopy(sig);bad['body']['state']['gold']='answer'
            with self.assertRaises(ValueError):core.parse_response(raw,bad)
            bad=deepcopy(raw);bad['model']='different'
            with self.assertRaises(ValueError):core.parse_response(bad,sig)
            bad=deepcopy(raw);bad['answers'][next(iter(bad['answers']))]['type']='choice'
            with self.assertRaises(ValueError):core.parse_response(bad,sig)

    def test_oversize_not_truncated_or_rebatched(self):
        pool=facts();pool[0]['text']='X'*140000
        p=core.build_precision_plan('Q','D',pool,pool[:20]);sig=p['requests'][0]['signature']
        self.assertEqual(len(p['requests']),3)
        self.assertEqual(core.validate_request(sig,allow_oversize=True),'precision')
        with self.assertRaises(ValueError):core.validate_request(sig)

    def test_inputs_not_mutated(self):
        p,r,c,cr,z=fixture();prior=deepcopy((p,r,c,cr,z))
        d=core.choose_final(z,{x['key']:.9 for x in z['proposals']})
        f=core.selected_facts(p['base']['pool'],d);f[0]['text']='local mutation'
        self.assertEqual((p,r,c,cr,z),prior)

    def test_no_request_invalid_status_still_rejected(self):
        pool=facts(20);p=core.build_precision_plan('Q','D',pool,pool)
        r=core.resolve_precision(p,None,score_status='failed')
        c=core.build_complement_plan(p,r)
        with self.assertRaises(ValueError):core.resolve_complement(c,None,score_status='unknown_status')
        z=core.build_set_plan(c,core.resolve_complement(c,None,score_status='failed'))
        with self.assertRaises(ValueError):core.choose_final(z,None,score_status='unknown_status')


if __name__=='__main__':unittest.main()
