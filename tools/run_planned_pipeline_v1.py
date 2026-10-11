"""Bound stage queue: wait for H QA, then C boundaries/grouping/development.

Stops at real development audit; never fabricates a gate or starts remaining
writers/QA. Every child uses its already-reviewed frozen implementation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_sdk20_modules import save_new, verify, sha
from tools.run_pilot400_experiments_v2 import publish_json
from tools.check_experiment_health import shared_read
from tools.run_mem0_regroup_plain_qa_v1 import durable_intent
from tools.freeze_sdk20_full1540_v1 import verify_lock
from tools import run_boundary_variants_v1 as boundary
from tools import run_boundary_variants_parallel_v1 as parallel
from tools import run_planned_plain_qa_v1 as planned

SCHEMA = 'planned-fixed1540-stage-queue-v1'
H = ROOT / 'runs/planned_hybrid_qa_v1'
C = ROOT / 'runs/boundary_variants_v1_20261003'
QUEUE = ROOT / 'runs/planned_pipeline_v1'
CODE = tuple(dict.fromkeys((*planned.CODE, *parallel.FILES,
        'tools/run_planned_pipeline_v1.py', 'tools/run_boundary_variants_v1.py')))
H_ARMS = ('hybrid_rank', 'hybrid_types', 'hybrid_types_no_relation')


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(shared_read(Path(path)))
def pointer(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def validate_h_terminal(out, manifest):
    jobs = manifest['jobs']
    expected = {(j['conversation'], j['method'], j['query_id']) for j in jobs}
    observed = {(r['conversation'], r['method'], r['query_id']) for r in out.get('results', [])}
    if (len(jobs) != 4620 or len(expected) != 4620 or
        Counter(j['method'] for j in jobs) != {a:1540 for a in H_ARMS} or
        manifest.get('phase_kind') != 'firstpass' or
        out.get('total') != 4620 or len(out.get('results', [])) != 4620 or
        expected != observed or out.get('terminal') is not True or
        out.get('protocol_sha256') != manifest['protocol_sha256']):
        raise ValueError('H entire4620 firstpass has not closed')
    if any(r.get('local_errors') for r in out.get('outcomes', [])):
        raise ValueError('Unresolved H local/controller errors require review before progression')
    originals={planned.job_key(j):j for j in jobs}
    for row in out['results']:
        j=originals[planned.job_key(row)];values={}
        paths=planned.stage_files(H,j)
        expected_identity=planned.identity(j,manifest)
        for stage in ('reader', 'judge', 'compat', 'qa'):
            ref = row.get(stage)
            if not ref or pointer(paths[stage]) != ref:
                raise ValueError('H missing/changed stage; cannot infer closed state')
            value=verify(paths[stage]);values[stage]=value
            if value.get('status') not in {
                'reader':{'complete','failed'}, 'judge':{'complete','failed','not_run'},
                'compat':{'strict_valid','compat_valid','rejected','unavailable'},
                'qa':{'complete','failed'}}[stage]:
                raise ValueError('H unknown/local stage state requires review')
            if value.get('stage')!=stage or any(value.get(k)!=v for k,v in expected_identity.items()):
                raise ValueError('H stage changed frozen slot identity')
        reader,judge,compat,qa=(values[s] for s in ('reader','judge','compat','qa'))
        if compat.get('error'):
            raise ValueError('H compatibility provenance error requires review')
        if reader['status']=='failed':
            planned.known_failure(j,manifest,reader,row['reader'],'reader')
            if (judge['status']!='not_run' or (judge.get('error')or{}).get('type')!='ReaderFailed'
                    or judge.get('result') is not None or judge.get('api_events') or compat['status']!='unavailable'):
                raise ValueError('H failed Reader has unexpected downstream state')
        elif judge['status']=='failed':
            planned.known_failure(j,manifest,judge,row['judge'],'judge')
        elif judge['status']!='complete':
            raise ValueError('H local/not-run Judge requires review')
        for stage in ('reader','judge'):
            value=values[stage]
            if value['status']=='complete':
                if value.get('error') or value.get('result') is None or len(value.get('api_events',[]))!=1:
                    raise ValueError('H successful stage lacks its request outcome')
                event=value['api_events'][0]
                if (event.get('error_type') is not None or not event.get('cache_sha256') or
                        pointer(event['cache_path'])['sha256']!=event['cache_sha256']):
                    raise ValueError('H successful stage raw cache missing/changed')
        expected_qa='complete' if reader['status']==judge['status']=='complete' else 'failed'
        if (qa['status']!=expected_qa or qa.get('reader_status')!=reader['status'] or
            qa.get('judge_status')!=judge['status'] or
            any(qa.get(s)!=row[s] for s in ('reader','judge','compat')) or
            judge.get('reader_sha256')!=row['reader']['sha256'] or compat.get('strict')!=row['judge']):
            raise ValueError('H persisted stage chain differs')
    # A strict invalid Judge label is a recorded technical outcome, not an excuse
    # to retry before the other predeclared arms finish their first pass.
    return True


def steps():
    base = ['--run-dir', str(C)]
    return [
        ('boundary', 'tools/run_boundary_variants_parallel_v1.py', ['run', *base, '--stage', 'boundary']),
        ('global_prepare_c1', 'tools/run_boundary_variants_v1.py', ['global-prepare', *base, '--arm', 'c1_multiscale']),
        ('global_prepare_c2', 'tools/run_boundary_variants_v1.py', ['global-prepare', *base, '--arm', 'c2_neighbour']),
        ('global_score', 'tools/run_boundary_variants_parallel_v1.py', ['run', *base, '--stage', 'global-score']),
        ('writer_build_c1', 'tools/run_boundary_variants_v1.py', ['build', *base, '--arm', 'c1_multiscale']),
        ('writer_build_c2', 'tools/run_boundary_variants_v1.py', ['build', *base, '--arm', 'c2_neighbour']),
        ('writer_development', 'tools/run_boundary_variants_parallel_v1.py',
            ['run', *base, '--stage', 'write', '--conversations', '0', '1']),
    ]


def verify_step(name):
    if name in ('boundary', 'global_score', 'writer_development'):
        scope = {'boundary': 'boundary', 'global_score': 'global-score', 'writer_development': 'write_development'}[name]
        path = C / 'parallel_controller_v1' / scope / 'parent_state.json'; row = read(path)
        convs = [0, 1] if name == 'writer_development' else list(range(10))
        stage={'boundary':'boundary','global_score':'global-score','writer_development':'write'}[name]
        controller=path.parent;cm=verify(controller/'manifest.json');cp=pointer(controller/'manifest.json')
        if (cm.get('schema')!=parallel.VERSION or cm.get('parent')!=pointer(C/'manifest.json') or
            cm.get('code_sha256')!={p:sha(ROOT/p) for p in parallel.FILES} or
            cm.get('policy',{}).get('stage')!=stage or cm['policy'].get('arms')!=list(boundary.variants.ARMS) or
            cm['policy'].get('conversations')!=convs):
            raise ValueError('Child controller frozen scope/source/code differs')
        if (row.get('status') != 'complete' or not row.get('finished_utc') or row.get('stage')!=stage or
            row.get('controller_manifest')!=cp or row.get('arms')!=list(boundary.variants.ARMS) or
            set(row.get('outcomes', {})) != set(map(str, convs)) or row.get('requested_conversations') != convs):
            raise ValueError('Child stage recorded failed/incomplete outcomes; not admitted by exit code')
        for ci in convs:
            outcome = row['outcomes'][str(ci)]
            if (outcome.get('status') != 'complete' or outcome.get('conversation')!=ci or
                len(outcome.get('arms', [])) != 2 or
                [a.get('arm') for a in outcome['arms']]!=list(boundary.variants.ARMS)):
                raise ValueError('Expected both complete declared arms per conversation')
            for arm in outcome['arms']:
                terminal=controller/'conversations'/f'conv_{ci}'/arm['arm']/'terminal.json'
                stored=verify(terminal)
                if (arm.get('status') != 'complete' or arm.get('conversation')!=ci or arm.get('stage')!=stage or
                    arm.get('controller_manifest')!=cp or not arm.get('finished_utc') or
                    {k:v for k,v in arm.items() if k not in ('content_sha256','reused_success_terminal')}!=
                    {k:v for k,v in stored.items() if k!='content_sha256'} or
                    arm.get('outputs')!=parallel.output_pointers(C,stage,arm['arm'],ci)):
                    raise ValueError('Recorded child terminal/source/outputs differ')
                for ref in arm.get('outputs', {}).values():
                    if pointer(ref['path']) != ref: raise ValueError('Child output pointer changed')
        return pointer(path)
    arm = 'c1_multiscale' if name.endswith('_c1') else 'c2_neighbour'
    if name.startswith('global_prepare'):
        boundary.global_manifest(C / arm / 'global')
        return pointer(C / arm / 'global/manifest.json')
    boundary.writer.manifest(C / arm / 'writer')
    ready=verify(C / arm / 'source_ready.json')
    if ready.get('status')!='source_ready_for_writer' or ready.get('arm')!=arm or ready.get('writer_manifest')!=pointer(C/arm/'writer/run_manifest.json'):
        raise ValueError('Writer build source-ready proof differs')
    return pointer(C / arm / 'source_ready.json')


def validate_queue_bindings(m):
    for key in ('h_manifest','h_execution','c_manifest'):
        if pointer(m[key]['path'])!=m[key]:raise ValueError('Pipeline source pointer changed')
    if m['code_sha256']!={p:sha(ROOT/p) for p in CODE} or m['steps']!=[list(x) for x in steps()]:
        raise ValueError('Frozen queue implementation/commands changed')


def h_identity_for_intent(intent,manifest):
    if intent.get('protocol_sha256')!=manifest['protocol_sha256']:
        raise ValueError('H execution intent protocol differs')
    identity=parallel.process_identity(intent['pid'])
    if os.name=='nt':
        created=(identity['creation_ticks']/10000000)-11644473600
        if created>datetime.fromisoformat(intent['created_utc']).timestamp():
            raise ValueError('H PID was created after original intent; reused process refused')
    return identity


def prepare():
    verify_lock(); boundary.manifest(C)
    parallel.validate_inputs(C, 'boundary', list(boundary.variants.ARMS), list(range(10)), {})
    if QUEUE.exists(): raise FileExistsError('A prior pipeline namespace must be retained')
    hm = read(H / 'run_manifest.json')
    if len(hm['jobs']) != 4620: raise ValueError('Unexpected H population')
    intent = read(H / 'execution_intent.json')
    if (H / 'run_outcome.json').exists():
        validate_h_terminal(verify(H / 'run_outcome.json'), hm)
        identity = None
    else:
        identity = h_identity_for_intent(intent,hm)
    value = {'schema': SCHEMA, 'created_utc': now(), 'h_manifest': pointer(H / 'run_manifest.json'),
        'h_execution': pointer(H / 'execution_intent.json'), 'h_process_identity': identity,
        'c_manifest': pointer(C / 'manifest.json'), 'code_sha256': {p: sha(ROOT / p) for p in CODE},
        'steps': steps(), 'stop_after': 'four_development_banks_then_required_source_index_semantic_review',
        'max_conversation_workers': 8, 'wait_seconds': 15, 'model_calls_during_prepare': 0}
    save_new(QUEUE / 'manifest.json', value)
    return {'prepared': str(QUEUE), 'model_calls': 0}


def run():
    m = verify(QUEUE / 'manifest.json')
    validate_queue_bindings(m)
    durable_intent(QUEUE / 'execution_intent.json', {'pid': os.getpid(), 'created_utc': now()})
    state = {'schema': SCHEMA, 'pid': os.getpid(), 'status': 'waiting_h_firstpass', 'started_utc': now(), 'steps': []}
    try:
        while not (H / 'run_outcome.json').exists():
            if not m['h_process_identity'] or parallel.process_identity(m['h_process_identity']['pid']) != m['h_process_identity']:
                raise RuntimeError('H process ended/replaced without closed outcome; no automatic restart')
            state['updated_utc'] = now(); publish_json(QUEUE / 'status.json', state)
            time.sleep(15)
        validate_h_terminal(verify(H / 'run_outcome.json'), read(H / 'run_manifest.json'))
        state['h_terminal'] = pointer(H / 'run_outcome.json')
        for name, script, args in steps():
            validate_queue_bindings(m); verify_lock(); boundary.manifest(C)
            folder = QUEUE / name
            durable_intent(folder / 'intent.json', {'stage': name, 'started_utc': now(), 'args': args})
            state.update(status='running', current_stage=name, updated_utc=now())
            publish_json(QUEUE / 'status.json', state)
            with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
                child = subprocess.Popen([sys.executable, str(ROOT / script), *args], cwd=ROOT,
                    stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                state['child_pid'] = child.pid
                publish_json(QUEUE / 'status.json', state)
                result = child.wait()
            if result != 0: raise RuntimeError('Child exited nonzero; original artifacts retained: ' + name)
            proof = verify_step(name)
            step = {'stage': name, 'status': 'complete', 'exit_code': result, 'proof': proof, 'finished_utc': now()}
            save_new(folder / 'outcome.json', step); state['steps'].append(step)
        state.update(status='awaiting_development_review', finished_utc=now(), child_pid=None)
    except Exception as exc:
        state.update(status='needs_review', error_type=type(exc).__name__,
                     message='Original stage failed or source/terminal guard refused; inspect stage artifacts. No automatic retry.',
                     finished_utc=now(), child_pid=None)
        raise
    finally:
        state['updated_utc'] = now(); publish_json(QUEUE / 'status.json', state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('prepare', 'run'))
    a = p.parse_args(); print(json.dumps(prepare() if a.stage == 'prepare' else run(), ensure_ascii=False))


if __name__ == '__main__': main()
