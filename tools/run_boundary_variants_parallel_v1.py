"""Eight-conversation controller around frozen C runners; no work on import.

The parent holds the original stage locks. Only its spawned workers may omit
those locks, in independent function globals. Each conversation runs arms in
order; all jobs finish before a failure is reported. Uncertain jobs never retry.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import FunctionType, SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_boundary_variants_v1 as frozen
from tools.run_pilot400_experiments_v2 import publish_json
from tools.check_experiment_health import shared_read

VERSION='boundary-variants-parallel-controller-v1'
FILES=('tools/run_boundary_variants_parallel_v1.py','tools/check_experiment_health.py')
STAGES=('boundary','global-score','write')
MAX_WORKERS=8


def now():return datetime.now(timezone.utc).isoformat()
def shared_bytes(path):
    for attempt in range(3):
        try:return shared_read(path)
        except OSError:
            if attempt==2:raise
            time.sleep(.02*(attempt+1))
def read(path):return json.loads(shared_bytes(path).decode('utf-8-sig'))
def save(path,value):frozen.save(path,value)
def checked(ptr):return frozen.checked(ptr)


def process_identity(pid):
    """Live PID plus OS creation identity; reject stale/reused Windows PIDs."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetProcessTimes.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME))
        kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        handle=kernel.OpenProcess(0x1000|0x100000,False,pid)
        if not handle:raise RuntimeError('Parent PID is not queryable/live')
        try:
            if kernel.WaitForSingleObject(handle,0)!=258:raise RuntimeError('Parent process has exited')
            creation,exit_time,user,kern=(wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle,creation,exit_time,user,kern):raise RuntimeError('Process creation identity unavailable')
            return {'pid':pid,'creation_ticks':(creation.dwHighDateTime<<32)|creation.dwLowDateTime}
        finally:kernel.CloseHandle(handle)
    os.kill(pid,0)
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    if stat[0]=='Z':raise RuntimeError('Parent process is a zombie')
    return {'pid':pid,'creation_ticks':int(stat[19])}


def parent_alive(payload):
    if process_identity(payload['parent_pid'])!=payload['parent_process_identity']:
        raise RuntimeError('Parent creation identity changed; no orphan/reused-PID work')


def scope_key(stage,convs):
    return 'write_development' if stage=='write' and set(convs)<={0,1} else 'write_expansion' if stage=='write' else stage


def development_gate(run,arm,path):
    """Require closed strict audit of exactly both development banks, never QA."""
    audit_path=Path(path)
    value=checked(frozen.pointer(audit_path))
    writer_root=Path(run)/arm/'writer';m=frozen.writer.manifest(writer_root)
    if (value.get('schema')!='boundary-variants-terminal-bank-audit-v1'
        or value.get('strict_passed') is not True or value.get('arm')!=arm
        or value.get('conversations')!=[0,1]
        or value.get('writer_manifest')!=frozen.pointer(writer_root/'run_manifest.json')):
        raise ValueError('Exactly two strict-complete development banks require bound audit')
    rows=value.get('rows',[])
    if len(rows)!=2 or {(r.get('conversation'),r.get('arm')) for r in rows}!={(0,arm),(1,arm)}:
        raise ValueError('Development audit must bind both conversation/arm identities')
    for row in rows:
        ci=row['conversation'];folder=writer_root/f'conv_{ci}'/arm
        for key,name in (('bank','bank.json'),('write_status','write_status.json'),('group_inputs','group_inputs.json')):
            if row.get(key)!=frozen.pointer(folder/name):raise ValueError('Development source pointer differs')
        bank=frozen.writer.check_pointer(row['bank']);state=frozen.writer.check_pointer(row['write_status'])
        if (bank.get('complete') is not True or (bank.get('conversation'),bank.get('arm'))!=(ci,arm)
            or bank.get('protocol_sha256')!=m['protocol_sha256'] or state.get('status')!='complete'
            or state.get('protocol_sha256')!=m['protocol_sha256']):
            raise ValueError('Development bank status/protocol differs')
        if not row.get('native_inventory'):raise ValueError('Closed native inventory evidence required')
    for path,hash_value in value.get('source_file_hashes',{}).items():
        if frozen.sha(path)!=hash_value:raise ValueError('Development audit source changed')
    return frozen.pointer(audit_path)


def validate_inputs(run,stage,arms,convs,gates):
    frozen.manifest(run)
    if stage not in STAGES or not arms or len(set(arms))!=len(arms) or any(a not in frozen.variants.ARMS for a in arms):
        raise ValueError('Declared stage and unique boundary arms required')
    if not convs or len(set(convs))!=len(convs) or any(type(ci)is not int or ci not in range(10) for ci in convs):
        raise ValueError('Unique conversation indices 0..9 required')
    bound={}
    for arm in arms:
        if stage=='global-score':frozen.global_manifest(Path(run)/arm/'global')
        if stage=='write':
            frozen.writer.manifest(Path(run)/arm/'writer')
            if any(ci>1 for ci in convs):
                if arm not in gates:raise ValueError('Expansion requires development audit for '+arm)
                bound[arm]=development_gate(run,arm,gates[arm])
    return bound


def controller_manifest(run,stage,arms,convs,gates):
    run=Path(run).resolve();bound=validate_inputs(run,stage,arms,convs,gates)
    folder=run/'parallel_controller_v1'/scope_key(stage,convs)
    policy={'stage':stage,'arms':list(arms),'conversations':list(convs),'max_workers':MAX_WORKERS,
            'conversation_policy':'one_worker_per_conv_arms_sequential_no_early_global_cancel',
            'uncertain_attempt_policy':'no_reentry_when_intent_without_success_terminal',
            'physical_retries':'unchanged_frozen_stage_retries_2',
            'parent_lease':'live_pid_creation_identity_each_arm_and_2sec_heartbeat_failclosed_exit',
            'hot_reads':'shared_READ_WRITE_DELETE_closed_before_decode_with_3_bounded_attempts'}
    value={'schema':VERSION,'parent':frozen.pointer(run/'manifest.json'),'policy':policy,
           'development_audits':bound,'code_sha256':{p:frozen.sha(ROOT/p) for p in FILES}}
    path=folder/'manifest.json'
    if path.exists():
        old=checked(frozen.pointer(path));old.pop('content_sha256')
        if old!=value:raise ValueError('Controller namespace/policy/code changed; use separate reviewed version')
    else:save(path,value)
    return folder,value


def stage_locks(run,stage,arms):
    stack=ExitStack()
    try:
        # The original boundary lock also prevents a second parallel controller.
        stack.enter_context(frozen.global_runner.exclusive_driver(Path(run)))
        if stage=='global-score':
            for arm in arms:stack.enter_context(frozen.global_runner.exclusive_driver(Path(run)/arm/'global'))
        elif stage=='write':
            for arm in arms:stack.enter_context(frozen.writer.writer_process_lock(Path(run)/arm/'writer'))
        return stack
    except Exception:
        stack.close();raise


def coordination(payload):
    parent_alive(payload)
    claim=read(Path(payload['controller'])/'parent_state.json')
    if (claim.get('status')!='running' or claim.get('nonce')!=payload['nonce']
        or claim.get('pid')!=payload['parent_pid'] or os.getppid()!=payload['parent_pid']
        or claim.get('stage')!=payload['stage']
        or claim.get('parent_process_identity')!=payload['parent_process_identity']):
        raise RuntimeError('Only a spawned worker under the live parent claim may omit stage locks')
    return nullcontext()


def invoke(payload,arm,ci):
    """Use original function bodies, omitting only parent-owned exact locks."""
    coordination(payload)
    stage=payload['stage'];run=Path(payload['run'])
    namespace=dict(vars(frozen))
    if stage in ('boundary','global-score'):
        attrs=dict(vars(frozen.global_runner));attrs['exclusive_driver']=lambda path:coordination(payload)
        if stage=='global-score':
            original_publish=attrs['publish_json']
            def publish_local(path,value):
                path=Path(path)
                if path.resolve()==(run/arm/'global'/'score_status.json').resolve():
                    path=run/arm/'global'/'conversations'/f'conv_{ci}'/'score_status.json'
                return original_publish(path,value)
            attrs['publish_json']=publish_local
        namespace['global_runner']=SimpleNamespace(**attrs)
        original=frozen.boundary if stage=='boundary' else frozen.global_score
    else:
        attrs=dict(vars(frozen.writer));attrs['writer_process_lock']=lambda path:coordination(payload)
        namespace['writer']=SimpleNamespace(**attrs);original=frozen.write
    fn=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    fn.__kwdefaults__=original.__kwdefaults__
    return fn(run,arm,[ci])


def output_pointers(run,stage,arm,ci):
    run=Path(run)
    if stage=='boundary':
        path=run/arm/'boundary'/f'conv_{ci}'/'grouping.json'
        value=checked(frozen.pointer(path))
        if value.get('status')!='complete' or value.get('conversation')!=ci:
            raise ValueError('Boundary output must be closed complete')
        return {'grouping':frozen.pointer(path)}
    if stage=='global-score':
        path=run/arm/'global'/'conversations'/f'conv_{ci}'/'grouping.json'
        value=checked(frozen.pointer(path))
        if value.get('status')!='complete' or value.get('conversation')!=ci:
            raise ValueError('Global grouping output must be closed complete')
        return {'grouping':frozen.pointer(path)}
    folder=run/arm/'writer'/f'conv_{ci}'/arm
    value=read(folder/'write_status.json')
    if value.get('status')!='complete' or (value.get('conversation'),value.get('arm'))!=(ci,arm):
        raise ValueError('Writer terminal must be closed complete')
    bank=frozen.writer.check_pointer(value['bank'])
    if bank.get('complete') is not True:raise ValueError('Writer bank incomplete')
    return {'write_status':frozen.pointer(folder/'write_status.json'),'bank':value['bank']}


def progress_snapshot(run,stage,arm,ci):
    """Best-effort health fields; never infer semantic success from counters."""
    run=Path(run)
    if stage=='boundary':folder=run/arm/'boundary'/f'conv_{ci}'/'api'
    elif stage=='global-score':folder=run/arm/'global'/'api'
    else:folder=run/arm/'writer'/f'conv_{ci}'/arm/'native'
    ledger=folder/'api_calls.jsonl';rows=[]
    if ledger.exists():
        for line in shared_bytes(ledger).decode('utf-8-sig').splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue  # unfinished current line is not an outcome
            if stage!='global-score' or row.get('context',{}).get('conversation')==ci:rows.append(row)
    result={'provider_ledger':str(ledger.resolve()),'physical_attempt_rows':sum(not x.get('cache_hit',False) for x in rows),
        'cache_hit_rows':sum(bool(x.get('cache_hit',False)) for x in rows),
        'distinct_logged_request_hashes':len({x.get('request_hash') for x in rows}),
        'last_provider_activity_utc':(rows[-1].get('timestamp_utc') or rows[-1].get('timestamp')) if rows else None,
        'last_activity_timestamp_status':'available' if rows and (rows[-1].get('timestamp_utc') or rows[-1].get('timestamp')) else 'unknown_or_no_rows_yet'}
    if stage=='write':
        status=run/arm/'writer'/f'conv_{ci}'/arm/'write_status.json'
        if status.exists():
            try:
                s=read(status);result['writer_progress']={k:s.get(k) for k in ('status','expected_groups','completed_groups','last_group','facts','updated_utc')}
            except json.JSONDecodeError:pass
    elif stage=='global-score':
        folder=run/arm/'global'/'conversations'/f'conv_{ci}'/'pairs'
        result['pair_result_files']=sum(1 for _ in folder.glob('*.json'))
    result['usage_policy']='raw provider usage in ledger/cache; missing usage unknown; retries/cache counted separately'
    return result


def execute_job(payload,arm,ci,operation=invoke,validate_output=output_pointers):
    """Durable logical attempt guard; failed or interrupted attempts are retained."""
    folder=Path(payload['controller'])/'conversations'/f'conv_{ci}'/arm
    intent=folder/'intent.json';terminal=folder/'terminal.json'
    parent_manifest=frozen.pointer(Path(payload['controller'])/'manifest.json')
    if terminal.exists():
        value=checked(frozen.pointer(terminal))
        if value.get('status')!='complete' or value.get('controller_manifest')!=parent_manifest:
            raise RuntimeError('Failed or mismatched controller terminal retained')
        for ptr in value['outputs'].values():frozen.global_runner.checked_pointer(ptr)
        return {**value,'reused_success_terminal':True}
    if intent.exists():raise RuntimeError('Prior uncertain controller intent retained; no redispatch')
    save(intent,{'schema':VERSION,'controller_manifest':parent_manifest,'conversation':ci,'arm':arm,
        'stage':payload['stage'],'pid':os.getpid(),'parent_pid':payload['parent_pid'],
        'nonce':payload['nonce'],'started_utc':now()})
    state={'schema':VERSION,'status':'running','conversation':ci,'arm':arm,'stage':payload['stage'],
           'pid':os.getpid(),'parent_pid':payload['parent_pid'],'started_utc':now(),'updated_utc':now()}
    state_lock=threading.Lock();stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            try:parent_alive(payload)
            except Exception:
                # Fail closed after parent death: the durable intent remains.
                # Killing this worker does not imply cancellation of a server's
                # already dispatched request; that usage/outcome stays unknown.
                try:
                    state.update(status='parent_lost',updated_utc=now(),error_type='ParentLeaseLost')
                    publish_json(folder/'worker_state.json',state)
                finally:os._exit(70)
            try:
                with state_lock:
                    state.update(updated_utc=now(),health=progress_snapshot(payload['run'],payload['stage'],arm,ci))
                    publish_json(folder/'worker_state.json',state)
            except Exception as exc:
                # A sharing/parse/publication failure must not disable the lease
                # thread. Keep an explicit error until a later healthy heartbeat.
                with state_lock:
                    state['health_error']={'type':type(exc).__name__,'utc':now(),'health_valid':False}
                    try:publish_json(folder/'worker_state.json',state)
                    except Exception:pass
            stop.wait(2)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    value={'schema':VERSION,'controller_manifest':parent_manifest,'stage':payload['stage'],
           'conversation':ci,'arm':arm,'pid':os.getpid(),'started_utc':state['started_utc']}
    try:
        operation(payload,arm,ci)
        value.update(status='complete',outputs=validate_output(payload['run'],payload['stage'],arm,ci))
    except Exception as exc:
        # Frozen stages redact provider failures before raising. Store type only
        # here to avoid an unredacted transport exception in a controller report.
        value.update(status='failed',outputs={},error={'type':type(exc).__name__,
                     'message':'Stage failed or guard rejected; inspect bound frozen stage artifacts/ledger.'})
    finally:
        stop.set();thread.join()
    value['finished_utc']=now();save(terminal,value)
    with state_lock:
        state.update(status=value['status'],finished_utc=value['finished_utc'],updated_utc=now(),terminal=frozen.pointer(terminal))
        try:state['health']=progress_snapshot(payload['run'],payload['stage'],arm,ci)
        except Exception as exc:state['health_error']={'type':type(exc).__name__,'utc':now(),'health_valid':False}
        publish_json(folder/'worker_state.json',state)
    return value


def worker(payload,ci):
    coordination(payload)
    results=[]
    for arm in payload['arms']:
        try:results.append(execute_job(payload,arm,ci))
        except Exception as exc:
            results.append({'status':'guard_rejected','conversation':ci,'arm':arm,'pid':os.getpid(),
                            'finished_utc':now(),'error_type':type(exc).__name__})
    return {'conversation':ci,'pid':os.getpid(),'finished_utc':now(),'arms':results,
            'status':'complete' if all(x['status']=='complete' for x in results) else 'failed'}


def run_parallel(run,stage,arms,convs,gates):
    folder,m=controller_manifest(run,stage,arms,convs,gates)
    with stage_locks(run,stage,arms):
        nonce=uuid.uuid4().hex
        identity=process_identity(os.getpid())
        state={'schema':VERSION,'stage':stage,'status':'running','pid':os.getpid(),'nonce':nonce,
            'started_utc':now(),'updated_utc':now(),'requested_conversations':list(convs),'arms':list(arms),
            'max_active_conversation_workers':min(MAX_WORKERS,len(convs)),'outcomes':{},
            'controller_manifest':frozen.pointer(folder/'manifest.json'),'parent_process_identity':identity}
        publish_json(folder/'parent_state.json',state)
        payload={'run':str(Path(run).resolve()),'controller':str(folder.resolve()),'stage':stage,
                 'arms':list(arms),'parent_pid':os.getpid(),'nonce':nonce}
        payload['parent_process_identity']=identity
        try:
            with ProcessPoolExecutor(max_workers=min(MAX_WORKERS,len(convs))) as pool:
                tasks={pool.submit(worker,payload,ci):ci for ci in convs}
                for future in as_completed(tasks):
                    ci=tasks[future]
                    try:result=future.result()
                    except Exception as exc:result={'conversation':ci,'status':'failed','error_type':type(exc).__name__,'finished_utc':now()}
                    state['outcomes'][str(ci)]=result;state['updated_utc']=now()
                    publish_json(folder/'parent_state.json',state)
            state.update(status='complete' if all(x['status']=='complete' for x in state['outcomes'].values()) else 'failed',finished_utc=now(),updated_utc=now())
        except BaseException:
            state.update(status='interrupted',finished_utc=now(),updated_utc=now())
            raise
        finally:publish_json(folder/'parent_state.json',state)
    return state


def parse_gates(values):
    result={}
    for value in values:
        arm,path=value.split('=',1)
        if arm in result:raise ValueError('Duplicate development audit arm')
        result[arm]=Path(path).resolve()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','check','run','health'))
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--arms',nargs='+',choices=frozen.variants.ARMS,default=list(frozen.variants.ARMS))
    p.add_argument('--conversations',nargs='+',type=int);p.add_argument('--development-audit',action='append',default=[])
    a=p.parse_args();convs=a.conversations if a.conversations is not None else [0,1] if a.stage=='write' else list(range(10))
    run=a.run_dir.resolve();gates=parse_gates(a.development_audit)
    if a.command=='prepare':
        folder,m=controller_manifest(run,a.stage,a.arms,convs,gates)
        result={'prepared':str(folder),'manifest':frozen.pointer(folder/'manifest.json'),'calls':0}
    elif a.command=='check':
        bound=validate_inputs(run,a.stage,a.arms,convs,gates)
        result={'valid':True,'stage':a.stage,'arms':a.arms,'conversations':convs,'development_audits':bound,'calls':0}
    elif a.command=='health':
        folder=run/'parallel_controller_v1'/scope_key(a.stage,convs)
        result={'parent':read(folder/'parent_state.json'),
                'workers':[read(path) for path in sorted(folder.glob('conversations/conv_*/*/worker_state.json'))]}
    else:result=run_parallel(run,a.stage,a.arms,convs,gates)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
