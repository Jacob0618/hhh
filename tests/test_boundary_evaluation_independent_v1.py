"""Independent scope, transport and terminal audits. No credentials/network."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from jev_mem import boundary_variants_v1 as variants
from jev_mem.sdk20_cached_api import CachedLegacyAPI
from tests.test_boundary_variants_v1 import Judge, PARAMS, embeddings, units
from tools import run_boundary_variants_parallel_v1 as parallel


def independent_global_status_body(run,arm,convs):
    # This body runs in the controller's isolated globals, like the old runner.
    global_runner.publish_json(Path(run)/arm/'global'/'score_status.json',{'fixture':True})
    global_runner.publish_json(Path(run)/arm/'global'/'other.json',{'fixture':True})
    return convs


class IndependentBoundaryEvaluation(unittest.TestCase):
    def test_multisession_exact_scope_and_no_cross_session_neighbour(self):
        raw=units(16)+units(16)
        for i,t in enumerate(raw[16:]):
            t.update(source_turn_id=f'D2:{i}',session_id='session_2',
                     observation_date='2023-02-02',text=f'not cancelled\n exact {i}')
        original=copy.deepcopy(raw)
        for arm in variants.ARMS:
            judge=Judge(initial=.3,dependency=.5)
            result=variants.segment_boundaries(raw,arm=arm,judge=judge,
                       embeddings=embeddings(32),**PARAMS)
            self.assertEqual(raw,original)
            self.assertEqual([t for s in result['segments'] for t in s['turns']],original)
            self.assertTrue(all(s['total_logical_calls']<=6 for s in result['trace']['sessions']))
            for state,criteria,stage in judge.calls:
                window=state['left_window']+state['right_window']
                self.assertEqual({t['session_id'] for t in window},{state['session_id']})
                self.assertEqual({t['observation_date'] for t in window},
                    {'2023-01-01' if state['session_id']=='session_1' else '2023-02-02'})
                self.assertEqual(set(criteria),{'topic_shift','right_depends_on_left'})

    def test_qa_field_rejected_before_any_judging(self):
        for arm in variants.ARMS:
            raw=units();raw[0]['gold_evidence']='inadmissible'
            judge=Judge()
            with self.assertRaises(ValueError):
                variants.segment_boundaries(raw,arm=arm,judge=judge,
                    embeddings=embeddings(),**PARAMS)
            self.assertEqual(judge.calls,[])

    def test_c2_exact_cut_and_dependency_thresholds(self):
        raw=units()
        for dependency,expected_cut in ((.499,True),(.5,False)):
            judge=Judge(initial=.7,neighbour=.7,dependency=dependency)
            result=variants.segment_boundaries(raw,arm=variants.ARMS[1],judge=judge,
                        embeddings=embeddings(),**PARAMS)
            assessed=[d for d in result['trace']['decisions'] if d['status']=='assessed']
            self.assertTrue(assessed)
            self.assertTrue(all(d['cut']==expected_cut for d in assessed))
            self.assertEqual(len(judge.calls),6)

    def test_two_criteria_one_logical_can_have_three_http_attempts(self):
        class Response:
            ok=False;status_code=429;headers={};text='test only'
            def json(self):return {'error':'test only'}
        with tempfile.TemporaryDirectory() as temp:
            api=CachedLegacyAPI(Path(temp),keys={'GEMINI_API_KEY':'dummy_g',
                'TYPESAFE_API_KEY':'dummy_t'},cache_roots=(),retries=2)
            with (patch('jev_mem.providers.requests.Session.post',return_value=Response()) as post,
                 patch('jev_mem.providers._pace_jev') as pace,
                 patch('jev_mem.providers.time.sleep')):
                with self.assertRaises(RuntimeError):
                    api.score_many('exact test input',{'topic_shift':'shift',
                        'right_depends_on_left':'dependency'},stage='independent_transport_audit')
                self.assertEqual(post.call_count,3)
                self.assertEqual(pace.call_count,3)
                self.assertEqual(len(api.events),1)
                self.assertEqual(len(list((Path(temp)/'request_intents').glob('*.json'))),1)
                with self.assertRaisesRegex(RuntimeError,'no implicit redispatch'):
                    api.score_many('exact test input',{'topic_shift':'shift',
                        'right_depends_on_left':'dependency'},stage='independent_transport_audit')
                self.assertEqual(post.call_count,3)

    def payload(self,folder):
        parallel.save(folder/'manifest.json',{'independent_fixture':True})
        return {'controller':str(folder),'run':str(folder),'stage':'boundary',
                'arms':list(variants.ARMS),'parent_pid':os.getpid(),'nonce':'offline',
                'parent_process_identity':parallel.process_identity(os.getpid())}

    def test_unknown_intent_and_failed_terminal_never_redispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);payload=self.payload(folder);calls=[]
            arm=variants.ARMS[0]
            intent=folder/'conversations/conv_0'/arm/'intent.json'
            parallel.save(intent,{'unknown':True})
            with self.assertRaisesRegex(RuntimeError,'no redispatch'):
                parallel.execute_job(payload,arm,0,operation=lambda *a:calls.append(a))
            self.assertEqual(calls,[])
            def fail(*args):calls.append(args);raise RuntimeError('offline failure')
            result=parallel.execute_job(payload,arm,1,operation=fail)
            self.assertEqual(result['status'],'failed')
            with self.assertRaisesRegex(RuntimeError,'terminal retained'):
                parallel.execute_job(payload,arm,1,operation=fail)
            self.assertEqual(len(calls),1)

    def test_success_terminal_reuse_rechecks_output_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);payload=self.payload(folder);arm=variants.ARMS[0];calls=[]
            output=folder/'independent_output.json'
            parallel.save(output,{'status':'complete'})
            validate=lambda *a:{'grouping':parallel.frozen.pointer(output)}
            operation=lambda *a:calls.append(a)
            first=parallel.execute_job(payload,arm,0,operation=operation,validate_output=validate)
            self.assertEqual(first['status'],'complete')
            reused=parallel.execute_job(payload,arm,0,operation=operation,validate_output=validate)
            self.assertTrue(reused['reused_success_terminal']);self.assertEqual(len(calls),1)
            output.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):
                parallel.execute_job(payload,arm,0,operation=operation,validate_output=validate)
            self.assertEqual(len(calls),1)

    def test_worker_runs_second_arm_after_first_failure(self):
        seen=[]
        def execute(payload,arm,ci):
            seen.append(arm)
            return {'status':'failed' if arm==variants.ARMS[0] else 'complete'}
        with patch.object(parallel,'coordination'),patch.object(parallel,'execute_job',side_effect=execute):
            result=parallel.worker({'arms':list(variants.ARMS)},3)
        self.assertEqual(seen,list(variants.ARMS));self.assertEqual(result['status'],'failed')

    def test_failed_conversation_does_not_cancel_remaining_nine(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);parallel.save(folder/'manifest.json',{'independent_fixture':True})
            seen=[];limits=[]
            def executor(*,max_workers):limits.append(max_workers);return ThreadPoolExecutor(max_workers=max_workers)
            def worker(payload,ci):
                seen.append(ci)
                return {'conversation':ci,'status':'failed' if ci==0 else 'complete'}
            from contextlib import nullcontext
            with (patch.object(parallel,'controller_manifest',return_value=(folder,{})),
                 patch.object(parallel,'stage_locks',return_value=nullcontext()),
                 patch.object(parallel,'ProcessPoolExecutor',side_effect=executor),
                 patch.object(parallel,'worker',side_effect=worker)):
                result=parallel.run_parallel(folder,'boundary',list(variants.ARMS),list(range(10)),{})
            self.assertEqual(limits,[8]);self.assertEqual(set(seen),set(range(10)))
            self.assertEqual(len(result['outcomes']),10);self.assertEqual(result['status'],'failed')

    def test_stale_reused_parent_pid_rejected_by_creation_identity(self):
        payload={'parent_pid':123,'parent_process_identity':{'pid':123,'creation_ticks':4}}
        with patch.object(parallel,'process_identity',return_value={'pid':123,'creation_ticks':5}):
            with self.assertRaisesRegex(RuntimeError,'creation identity changed'):
                parallel.parent_alive(payload)
        identity=parallel.process_identity(os.getpid())
        self.assertEqual(identity['pid'],os.getpid());self.assertGreater(identity['creation_ticks'],0)

    def test_global_status_is_per_conversation_without_patching_original(self):
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);arm=variants.ARMS[0];seen=[]
            old_publish=parallel.frozen.global_runner.publish_json
            def publish(path,value):seen.append(Path(path))
            with (patch.object(parallel,'coordination',return_value=nullcontext()),
                  patch.object(parallel.frozen,'global_score',independent_global_status_body),
                  patch.object(parallel.frozen.global_runner,'publish_json',side_effect=publish)):
                output=parallel.invoke({'stage':'global-score','run':str(folder)},arm,6)
            self.assertEqual(output,[6])
            self.assertEqual(seen,[folder/arm/'global/conversations/conv_6/score_status.json',
                                   folder/arm/'global/other.json'])
            self.assertIs(parallel.frozen.global_runner.publish_json,old_publish)

    def test_health_io_failure_does_not_stop_parent_lease_monitor(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);payload=self.payload(folder);reads=[]
            output=folder/'independent_output.json';parallel.save(output,{'status':'complete'})
            def snapshot(*args):
                reads.append(args)
                if len(reads)==1:raise OSError('offline transient health read failure')
                return {'fixture':'health read recovered'}
            def operation(*args):time.sleep(2.15)
            with (patch.object(parallel,'progress_snapshot',side_effect=snapshot),
                  patch.object(parallel,'parent_alive',wraps=parallel.parent_alive) as lease):
                result=parallel.execute_job(payload,variants.ARMS[0],0,operation=operation,
                         validate_output=lambda *a:{'grouping':parallel.frozen.pointer(output)})
            self.assertEqual(result['status'],'complete')
            self.assertGreaterEqual(lease.call_count,2)
            self.assertGreaterEqual(len(reads),3)


if __name__=='__main__':unittest.main()
