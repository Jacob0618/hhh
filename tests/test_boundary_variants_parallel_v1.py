"""Real 8-process offline coordination, guards and parent-lock invariants."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import run_boundary_variants_parallel_v1 as p


def offline_operation(payload,arm,ci):
    """Test fixture only, never exposed by the paid CLI."""
    folder=Path(payload['run'])/arm/'boundary'/f'conv_{ci}'
    time.sleep(.04)
    p.save(folder/'grouping.json',{'status':'complete','conversation':ci,'arm':arm,'model_calls':0})


def offline_failed_operation(payload,arm,ci):raise RuntimeError('offline intentional failure')


def offline_eight_worker(payload,ci):
    p.coordination(payload)
    started=time.monotonic()
    ready=Path(payload['run'])/'probe_ready';ready.mkdir(exist_ok=True)
    (ready/f'{ci}.json').write_text(json.dumps({'pid':os.getpid(),'start':started}),encoding='utf-8')
    deadline=time.monotonic()+30
    while len(list(ready.glob('*.json')))<8:
        if time.monotonic()>deadline:raise RuntimeError('Eight-worker barrier timed out')
        time.sleep(.01)
    # All 8 are live before any exits. The parent, not workers, owns the old lock.
    try:
        with p.frozen.global_runner.exclusive_driver(Path(payload['run'])):
            raise AssertionError('Parent lock unexpectedly obtainable by worker')
    except OSError:pass
    results=[p.execute_job(payload,arm,ci,operation=offline_operation) for arm in payload['arms']]
    return {'ci':ci,'pid':os.getpid(),'start':started,'finish':time.monotonic(),'results':results}


def fake_boundary_body(run,arm,convs):
    # The injected globals must substitute only the parent's already owned lock.
    with global_runner.exclusive_driver(run):
        return {'arm':arm,'convs':convs}


def fake_global_body(run,arm,convs):
    global_runner.publish_json(Path(run)/arm/'global'/'score_status.json',{'ci':convs[0]})
    global_runner.publish_json(Path(run)/arm/'global'/'other.json',{'unchanged':True})


def setup(root,parent_pid=None):
    root=Path(root);controller=root/'parallel_controller_v1'/'boundary';controller.mkdir(parents=True)
    p.save(controller/'manifest.json',{'schema':'offline_test_only'})
    payload={'run':str(root),'controller':str(controller),'stage':'boundary',
        'arms':list(p.frozen.variants.ARMS),'parent_pid':os.getpid() if parent_pid is None else parent_pid,'nonce':'offline_nonce'}
    payload['parent_process_identity']=p.process_identity(payload['parent_pid'])
    p.publish_json(controller/'parent_state.json',{'status':'running','pid':payload['parent_pid'],
        'nonce':payload['nonce'],'stage':'boundary','parent_process_identity':payload['parent_process_identity']})
    return payload


class ControllerTests(unittest.TestCase):
    def test_real_eight_processes_and_sequential_arms(self):
        with tempfile.TemporaryDirectory(prefix='boundary_parallel_offline_') as tmp:
            payload=setup(tmp)
            with p.stage_locks(tmp,'boundary',payload['arms']),ProcessPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(offline_eight_worker,[payload]*8,range(8)))
            self.assertEqual(len({r['pid'] for r in results}),8)
            self.assertLess(max(r['start'] for r in results),min(r['finish'] for r in results))
            for result in results:
                arms=result['results'];self.assertTrue(all(r['status']=='complete' for r in arms))
                self.assertLessEqual(arms[0]['finished_utc'],arms[1]['started_utc'])
                for arm in payload['arms']:
                    folder=Path(payload['controller'])/'conversations'/f"conv_{result['ci']}"/arm
                    state=p.read(folder/'worker_state.json')
                    self.assertEqual(state['pid'],result['pid']);self.assertEqual(state['status'],'complete')
                    self.assertTrue(state['started_utc']);self.assertTrue(state['finished_utc']);self.assertTrue(state['health'])
            self.assertEqual(len(list(Path(tmp).rglob('intent.json'))),16)
            self.assertFalse(list(Path(tmp).rglob('api_calls.jsonl')))

    def test_success_reuse_then_source_mutation_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp)
            first=p.execute_job(payload,payload['arms'][0],0,operation=offline_operation)
            second=p.execute_job(payload,payload['arms'][0],0,operation=offline_failed_operation)
            self.assertEqual(first['outputs'],second['outputs']);self.assertTrue(second['reused_success_terminal'])
            path=Path(first['outputs']['grouping']['path']);path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):p.execute_job(payload,payload['arms'][0],0,operation=offline_operation)

    def test_failure_retained_other_arm_can_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp)
            fail=p.execute_job(payload,payload['arms'][0],0,operation=offline_failed_operation)
            self.assertEqual(fail['status'],'failed')
            with self.assertRaises(RuntimeError):p.execute_job(payload,payload['arms'][0],0,operation=offline_operation)
            good=p.execute_job(payload,payload['arms'][1],0,operation=offline_operation)
            self.assertEqual(good['status'],'complete')

    def test_interrupted_intent_never_redispatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp);arm=payload['arms'][0]
            p.save(Path(payload['controller'])/'conversations'/'conv_0'/arm/'intent.json',{'status':'uncertain'})
            with self.assertRaises(RuntimeError):p.execute_job(payload,arm,0,operation=offline_operation)
            self.assertFalse(list(Path(tmp).rglob('grouping.json')))

    def test_wrong_parent_or_nonce_cannot_skip_original_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp,parent_pid=os.getppid())
            self.assertIsNotNone(p.coordination(payload))
            bad={**payload,'nonce':'wrong'}
            with self.assertRaises(RuntimeError):p.coordination(bad)
            bad={**payload,'parent_pid':os.getpid()}
            with self.assertRaises(RuntimeError):p.coordination(bad)
            bad={**payload,'parent_process_identity':{**payload['parent_process_identity'],'creation_ticks':-1}}
            with self.assertRaises(RuntimeError):p.coordination(bad)

    def test_function_body_lock_override_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp,parent_pid=os.getppid())
            original=p.frozen.global_runner.exclusive_driver
            with patch.object(p.frozen,'boundary',fake_boundary_body):
                result=p.invoke(payload,payload['arms'][0],0)
            self.assertEqual(result['convs'],[0])
            self.assertIs(p.frozen.global_runner.exclusive_driver,original)

    def test_global_status_redirect_only_exact_shared_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp,parent_pid=os.getppid());payload['stage']='global-score'
            claim=p.read(Path(payload['controller'])/'parent_state.json');claim['stage']='global-score'
            p.publish_json(Path(payload['controller'])/'parent_state.json',claim)
            arm=payload['arms'][0]
            with patch.object(p.frozen,'global_score',fake_global_body):p.invoke(payload,arm,3)
            folder=Path(tmp)/arm/'global'
            self.assertFalse((folder/'score_status.json').exists())
            self.assertEqual(p.read(folder/'conversations/conv_3/score_status.json'),{'ci':3})
            self.assertEqual(p.read(folder/'other.json'),{'unchanged':True})

    def test_health_timestamp_legacy_and_sharing_failure_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload=setup(tmp);arm=payload['arms'][0]
            ledger=Path(tmp)/arm/'boundary/conv_0/api/api_calls.jsonl';ledger.parent.mkdir(parents=True)
            ledger.write_text(json.dumps({'timestamp':'2026-10-03T01:00:00+00:00','request_hash':'h','cache_hit':False})+'\n',encoding='utf-8')
            health=p.progress_snapshot(tmp,'boundary',arm,0)
            self.assertEqual(health['last_provider_activity_utc'],'2026-10-03T01:00:00+00:00')
            self.assertEqual(health['physical_attempt_rows'],1)
            with patch.object(p,'progress_snapshot',side_effect=OSError('offline sharing simulation')):
                result=p.execute_job(payload,arm,0,operation=offline_operation)
            self.assertEqual(result['status'],'complete')
            state=p.read(Path(payload['controller'])/'conversations/conv_0'/arm/'worker_state.json')
            self.assertFalse(state['health_error']['health_valid'])

    def test_expansion_missing_audit_blocks_before_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(p.frozen,'manifest',return_value={}),patch.object(p.frozen.writer,'manifest',return_value={}):
                with self.assertRaisesRegex(ValueError,'development audit'):
                    p.validate_inputs(tmp,'write',p.frozen.variants.ARMS,[2,3],{})
            self.assertFalse(list(Path(tmp).rglob('intent.json')))

    def test_parent_acquires_all_original_stage_locks(self):
        entered=[]
        class Lock:
            def __init__(self,path):self.path=str(path)
            def __enter__(self):entered.append(self.path)
            def __exit__(self,*args):pass
        arms=p.frozen.variants.ARMS
        with patch.object(p.frozen.global_runner,'exclusive_driver',side_effect=Lock),patch.object(p.frozen.writer,'writer_process_lock',side_effect=Lock):
            with p.stage_locks(Path('fixture'),'global-score',arms):pass
            self.assertEqual(entered,['fixture',str(Path('fixture')/arms[0]/'global'),str(Path('fixture')/arms[1]/'global')])
            entered.clear()
            with p.stage_locks(Path('fixture'),'write',arms):pass
            self.assertEqual(entered,['fixture',str(Path('fixture')/arms[0]/'writer'),str(Path('fixture')/arms[1]/'writer')])


if __name__=='__main__':unittest.main()
