"""Persist real eight-process OFFLINE fixture evidence; never calls model API."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_boundary_variants_parallel_v1 as controller
from tests.test_boundary_variants_parallel_v1 import setup,offline_eight_worker


def main():
    with tempfile.TemporaryDirectory(prefix='boundary_eight_process_evidence_') as tmp:
        payload=setup(tmp)
        with controller.stage_locks(tmp,'boundary',payload['arms']),ProcessPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(offline_eight_worker,[payload]*8,range(8)))
        distinct={r['pid'] for r in results}
        overlap=max(r['start'] for r in results)<min(r['finish'] for r in results)
        assert len(distinct)==8 and overlap
        assert all(len(r['results'])==2 and all(x['status']=='complete' for x in r['results']) for r in results)
        assert not list(Path(tmp).rglob('api_calls.jsonl'))
        evidence={'schema':'boundary-variants-real-eight-process-offline-probe-v1','status':'passed',
            'distinct_worker_pids':sorted(distinct),'all_eight_overlap':overlap,'conversation_count':8,
            'sequential_arm_order':payload['arms'],'durable_logical_jobs_completed':16,
            'model_api_calls':0,'fixture_files_retained':False,
            'scope':'real Windows ProcessPool/lifecycle/parent lock probe, not paid pipeline throughput or method accuracy',
            'workers':[{'conversation':r['ci'],'pid':r['pid'],'start_monotonic':r['start'],'finish_monotonic':r['finish'],
                'arms':[{'arm':x['arm'],'started_utc':x['started_utc'],'finished_utc':x['finished_utc'],'status':x['status']} for x in r['results']]} for r in results],
            'code_sha256':{p:controller.frozen.sha(ROOT/p) for p in (
                'tools/run_boundary_variants_parallel_v1.py','tests/test_boundary_variants_parallel_v1.py',
                'tools/run_boundary_variants_parallel_probe_v1.py','tools/check_experiment_health.py')}}
    path=ROOT/'research/planned_execution_2026-10-03/boundary_parallel_offline_validation.json'
    controller.save(path,evidence)
    print({'evidence':str(path),'distinct_worker_pids':sorted(distinct),'all_eight_overlap':overlap,'calls':0})


if __name__=='__main__':main()
