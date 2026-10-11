"""Offline semantic-budget and frozen-body invariants; no API construction."""
import json
import unittest
import tempfile
from pathlib import Path
from jev_mem import boundary_variants_v1 as v
from jev_mem import sparse_topic_pipeline as old
from jev_mem.topic_boundaries import boundary_requests


def units(n=16):
    return [{'source_turn_id':f'D1:{i}', 'session_id':'session_1', 'speaker':'A',
             'observation_date':'2023-01-01','text':f'exact original turn {i}'} for i in range(n)]


def embeddings(n=16, special=7):
    return [[1. if j == (1 if i == special else 0) else 0. for j in range(768)] for i in range(n)]


class Judge:
    def __init__(self, initial=.4, dependency=.1, neighbour=.9):
        self.calls=[];self.initial=initial;self.dependency=dependency;self.neighbour=neighbour
    def score_many(self,state,criteria,*,stage):
        self.calls.append((json.loads(state),dict(criteria),stage))
        return {'topic_shift':self.neighbour if stage=='boundary_variants_neighbour' else self.initial,
                'right_depends_on_left':self.dependency}


PARAMS={'cut_threshold':.7,'keep_threshold':.3,'dependency_threshold':.5}


class BoundaryVariants(unittest.TestCase):
    def test_c1_executes_frozen_body(self):
        self.assertIs(v._c1.__code__,old.segment_sparse_sessions.__code__)
        self.assertIsNot(v._c1.__globals__,old.segment_sparse_sessions.__globals__)
        self.assertIs(old.propose_sparse_boundaries,old.segment_sparse_sessions.__globals__['propose_sparse_boundaries'])

    def test_anchors_and_request_contents_are_identical(self):
        raw=units();vec=embeddings()
        p=v.propose_multiscale_boundaries(raw,vec)['sessions'][0]
        self.assertEqual([int(k) for k,r in p['selection_reasons'].items() if r=='fractional_coverage_anchor'],[5,11])
        original={r['right_source_index']:r for r in boundary_requests(raw,window_turns=2)}
        for request in p['initial_requests']:self.assertEqual(request,original[request['right_source_index']])
        self.assertEqual(set(p['multiscale_change_scores']),{'1','2','4'})
        self.assertEqual(len(p['initial_requests']),4)
        for g in p['multiscale_priority_scores']:
            self.assertEqual(p['multiscale_priority_scores'][g],max(p['multiscale_change_scores'][w][g] for w in ('1','2','4')))

    def test_same_candidates_replay_old_exactly(self):
        raw=units();vec=embeddings(special=-1)
        a=old.segment_sparse_sessions(raw,judge=Judge(),embeddings=vec,**PARAMS)
        b=v.segment_boundaries(raw,arm=v.ARMS[0],judge=Judge(),embeddings=vec,**PARAMS)
        self.assertEqual(a['segments'],b['segments'])
        self.assertEqual(a['trace']['requests'],b['trace']['requests'])
        self.assertEqual(a['trace']['decisions'],b['trace']['decisions'])

    def test_c1_c2_initial_same_then_neighbours_no_expansion(self):
        raw=units();vec=embeddings();j1=Judge();j2=Judge()
        a=v.segment_boundaries(raw,arm=v.ARMS[0],judge=j1,embeddings=vec,**PARAMS)
        b=v.segment_boundaries(raw,arm=v.ARMS[1],judge=j2,embeddings=vec,**PARAMS)
        self.assertEqual(j1.calls[:4],j2.calls[:4])
        self.assertEqual(len(j1.calls),6);self.assertEqual(len(j2.calls),6)
        initial={r[0]['boundary_before'] for r in j2.calls[:4]}
        self.assertTrue(all(r[0]['boundary_before'] not in initial for r in j2.calls[4:]))
        original={r['state']['boundary_before']:r for r in boundary_requests(raw,window_turns=2)}
        for state,criteria,stage in j2.calls[4:]:
            self.assertEqual((state,criteria),(original[state['boundary_before']]['state'],original[state['boundary_before']]['criteria']))
            self.assertEqual(stage,'boundary_variants_neighbour')
        self.assertEqual([t for s in b['segments'] for t in s['turns']],raw)
        self.assertTrue(b['trace']['unassessed_gap_indices'])
        self.assertEqual(b['trace']['sessions'][0]['total_logical_calls'],6)

    def test_low_score_cannot_certify_unassessed_half(self):
        raw=units();j=Judge(initial=.1)
        r=v.segment_boundaries(raw,arm=v.ARMS[1],judge=j,embeddings=embeddings(),**PARAMS)
        self.assertEqual(len(j.calls),4)
        self.assertEqual(len(r['trace']['unassessed_gap_indices']),11)
        self.assertTrue(all(d['scores'] is None for d in r['trace']['decisions'] if d['status']=='unassessed'))

    def test_dependency_veto_kept_for_neighbour(self):
        j=Judge(initial=.8,dependency=.8)
        r=v.segment_boundaries(units(),arm=v.ARMS[1],judge=j,embeddings=embeddings(),**PARAMS)
        self.assertEqual(r['trace']['semantic_cut_indices'],[])
        self.assertTrue(all(d['dependency_veto'] for d in r['trace']['decisions'] if d['status']=='assessed'))

    def test_tiny_sessions_never_duplicate_neighbour(self):
        for n in (1,2,3,4):
            for arm in v.ARMS:
                j=Judge();r=v.segment_boundaries(units(n),arm=arm,judge=j,embeddings=embeddings(n),**PARAMS)
                self.assertLessEqual(len(j.calls),6)
                self.assertEqual([t for s in r['segments'] for t in s['turns']],units(n))

    def test_failures_remain_failures(self):
        class Bad:
            def score_many(self,*args,**kwargs):raise RuntimeError('offline deliberately failed')
        for arm in v.ARMS:
            with self.assertRaises(old.SparseTopicPipelineError) as cm:
                v.segment_boundaries(units(),arm=arm,judge=Bad(),embeddings=embeddings(),**PARAMS)
            self.assertEqual(cm.exception.trace['logical_judge_calls'],1)
            self.assertEqual(cm.exception.trace['requests'][0]['status'],'error')

    def test_existing_plan_to_native_writer_build_offline(self):
        # Integration fixture reuses old decisions, NOT a C1/C2 model result.
        # Never publish it as an experiment arm; all files are temporary.
        from tools import run_boundary_variants_v1 as runner
        source=runner.ROOT/'runs/boundary_variants_v1_20261003'
        if not source.exists():self.skipTest('run offline prepare before integration fixture')
        with tempfile.TemporaryDirectory(prefix='boundary_v1_offline_fixture_') as tmp:
            run=Path(tmp);arm=v.ARMS[0];gf=run/arm/'global'
            (run/'manifest.json').write_bytes((source/'manifest.json').read_bytes())
            oldm=runner.global_runner.frozen(runner.GLOBAL_SOURCE)
            parent=runner.manifest(run)
            for ci in range(10):
                plan=runner.global_runner.load_plan(oldm,ci)
                boundary=runner.checked(plan['source_grouping']);boundary.pop('content_sha256')
                boundary.update(conversation=ci,parent_manifest_sha256=runner.sha(run/'manifest.json'))
                runner.save(run/arm/'boundary'/f'conv_{ci}'/'grouping.json',boundary)
            prepared=runner.global_prepare(run,arm)
            self.assertEqual(prepared['segments'],385)
            for ci in range(10):
                pp=gf/'conversations'/f'conv_{ci}'/'plan.json'
                gp=runner.GLOBAL_SOURCE/'conversations'/f'conv_{ci}'/'grouping.json'
                result=runner.read(gp);result.pop('content_sha256');result['plan_sha256']=runner.sha(pp)
                runner.save(pp.with_name('grouping.json'),result)
            ready=runner.build(run,arm)
            self.assertEqual(ready['writer_group_count'],338)
            self.assertEqual(ready['writer_and_qa_calls_at_build'],0)
            native_m=runner.writer.manifest(run/arm/'writer')
            self.assertEqual(native_m['arms'],[arm]);self.assertEqual(len(native_m['inputs']),10)
            self.assertEqual(native_m['provider_retries'],2)
            self.assertFalse(list(run.rglob('api_calls.jsonl')))


if __name__=='__main__':unittest.main()
