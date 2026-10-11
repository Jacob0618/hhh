"""Seal terminal strict QA without rewriting outcomes or dispatching a model.

Requires two externally captured terminal snapshots separated by >=30 seconds.
Imports only the read-only snapshot/shared-reader and existing digest helpers.
All run files except coordination .lock files are hashed, never copied.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_experiment_health import shared_read
from tools.snapshot_mem0_plain_qa import snapshot
from tools.run_sdk20_modules import digest

SCHEMA='mem0-plain-qa-firstpass-terminal-seal-v1'
DEFAULT_RUN=ROOT/'runs/mem0_global_regroup_plain_qa_v2'
DEFAULT_OUT=ROOT/'research/plain_qa_v2_firstpass_terminal_seal_2026-09-29.json'
STAGES=('selection','reader','judge','qa')


def terminal_evidence(value,run,protocol):
    if (Path(value['run_dir']).resolve()!=Path(run).resolve()
            or value['protocol_sha256']!=protocol or value['expected_slots']!=3080
            or value['processes']!=[]):
        raise ValueError('Snapshot identity or zero-process evidence differs')
    controller=value['controller']['run_status.json']
    if (controller.get('terminal') is not True or controller.get('conversations_terminal')!=10
            or controller.get('expected_qa')!=3080 or controller.get('failures')
            or len(controller.get('outcomes',[]))!=10):
        raise ValueError('All ten controller conversations must be terminal')
    if {x.get('conversation') for x in controller['outcomes']}!=set(range(10)):
        raise ValueError('Controller conversation population differs')
    compact=[]
    for row in value['conversations']:
        ci=row['conversation'];q=row['states']['qa_status.json'];expected=2*row['expected_per_arm']
        if (q.get('terminal') is not True or q.get('completed')!=expected or q.get('total')!=expected
                or row['api']['malformed_line_numbers']):
            raise ValueError('Conversation is not terminal or ledger has partial lines')
        outcome=next(x for x in controller['outcomes'] if x['conversation']==ci)
        if any(outcome.get(k)!=q.get(k) for k in ('completed','total','reader_complete','judge_complete','failed','terminal')):
            raise ValueError('Controller and worker final state disagree')
        for arm,stages in row['arms'].items():
            for stage in STAGES:
                count=stages[stage]
                if (count['output_files']!=row['expected_per_arm']
                        or sum(count['counts'].values())!=row['expected_per_arm']
                        or not set(count['counts'])<=({'complete','failed'} if stage in ('selection','qa') else {'complete','failed','not_run'})):
                    raise ValueError('Every frozen slot must have a terminal artifact')
        compact.append({'conversation':ci,'states':row['states'],'arms':row['arms'],'api':row['api']})
    if len(compact)!=10 or {x['conversation'] for x in compact}!=set(range(10)):
        raise ValueError('Exactly ten conversation snapshots required')
    return {'controller':value['controller'],'conversations':sorted(compact,key=lambda x:x['conversation'])}


def check_snapshot_pair(first,second,run,protocol):
    elapsed=(datetime.fromisoformat(second['captured_utc'])-datetime.fromisoformat(first['captured_utc'])).total_seconds()
    if elapsed<30:raise ValueError('Terminal snapshots must be at least30 seconds apart')
    a=terminal_evidence(first,run,protocol);b=terminal_evidence(second,run,protocol)
    if a!=b:raise ValueError('Terminal statuses/slots/ledger changed between snapshots')
    return elapsed,b


def processes():
    command="Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'python*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    r=subprocess.run(['powershell.exe','-NoProfile','-Command',command],check=True,capture_output=True,
        text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    value=json.loads(r.stdout or '[]')
    return [value] if isinstance(value,dict) else value


def reject_live_qa(items,pids):
    matches=[p for p in items if p['ProcessId'] in pids
             or any(name in (p.get('CommandLine') or '') for name in
                    ('run_mem0_regroup_plain_qa_v1.py','run_mem0_regroup_plain_qa_v2.py'))]
    if matches:raise ValueError('A recorded QA PID or named QA driver remains live')
    return matches


def run(run_dir,snapshot_paths,out):
    run_dir=Path(run_dir).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Never overwrite a terminal seal')
    if out.is_relative_to(run_dir):raise ValueError('Seal must be outside original run')
    hashes={}
    def raw(path):
        path=Path(path).resolve();data=shared_read(path);value=hashlib.sha256(data).hexdigest()
        if str(path) in hashes and hashes[str(path)]!=value:raise ValueError('Source changed while sealing: '+str(path))
        hashes[str(path)]=value;return data
    def read(path):return json.loads(raw(path).decode('utf-8-sig'))
    def ptr(path):
        data=raw(path);return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    def bound(pointer):
        value=read(pointer['path'])
        if hashes[str(Path(pointer['path']).resolve())]!=pointer['sha256']:raise ValueError('Bound pointer changed')
        return value
    def sealed(path):
        value=read(path)
        if value.get('content_sha256')!=digest({k:v for k,v in value.items() if k!='content_sha256'}):
            raise ValueError('Artifact content seal differs: '+str(path))
        return value
    manifest=read(run_dir/'run_manifest.json');protocol=manifest['protocol_sha256']
    if protocol!=digest({k:v for k,v in manifest.items() if k!='protocol_sha256'}):raise ValueError('Manifest protocol changed')
    first,second=[read(p) for p in snapshot_paths]
    elapsed,stable=check_snapshot_pair(first,second,run_dir,protocol)
    recorded_pids={stable['controller']['execution_intent.json']['pid'],stable['controller']['run_status.json']['pid']}
    for row in stable['conversations']:
        recorded_pids.update(v['pid'] for v in row['states'].values() if 'pid' in v)
    current_processes=processes();reject_live_qa(current_processes,recorded_pids)
    fresh=snapshot(run_dir,process_reader=lambda:current_processes)
    if terminal_evidence(fresh,run_dir,protocol)!=stable:raise ValueError('Terminal state changed after supplied snapshots')
    code=[]
    for key in ('code_sha256','runtime_sha256','sdk_sha256'):
        for name,expected in manifest[key].items():
            p=Path(name) if Path(name).is_absolute() else ROOT/name
            ref=ptr(p)
            if ref['sha256']!=expected:raise ValueError('Frozen execution source changed: '+str(p))
            code.append(ref)
    gate=bound(manifest['gate']);bound(manifest['plain_text_config'])
    slot_counts={s:Counter() for s in STAGES};arm_counts={};slots=[];errors=Counter()
    source_inventory=[];native_copies=[]
    for ci in range(10):
        expected_ids=manifest['query_ids'][str(ci)]
        if len(expected_ids)!=len(set(expected_ids)):raise ValueError('Duplicate manifest query IDs')
        for arm in manifest['policy']['arms']:
            binding=manifest['banks'][str(ci)][arm];bank=bound(binding)
            source=Path(bank['native_source']).resolve();copy_native=Path(bank['native_dir']).resolve()
            if copy_native!=run_dir/f'conv_{ci}'/arm/'native' or source==copy_native:raise ValueError('Native copy/source alias')
            actual={p.relative_to(source).as_posix():ptr(p)['sha256'] for p in list((source/'qdrant').rglob('*'))+[source/'history.sqlite'] if p.is_file() and p.name!='.lock'}
            if actual!=bank['source_inventory']:raise ValueError('Original writer native inventory changed')
            source_inventory.append({'conversation':ci,'arm':arm,'native_source':str(source),'inventory':actual,
                'source_bank':bank['source_bank'],'source_kind':bank['source_review']['source_kind']})
            bound(bank['source_bank'])
            postpath=run_dir/f'conv_{ci}'/arm/'post_retrieval_inventory.json';post=sealed(postpath)
            copy_actual={p.relative_to(copy_native).as_posix():ptr(p)['sha256'] for p in list((copy_native/'qdrant').rglob('*'))+[copy_native/'history.sqlite'] if p.is_file() and p.name!='.lock'}
            if post['inventory']!=copy_actual or post['source_native_opened'] is not False:raise ValueError('Post-retrieval copy inventory changed')
            native_copies.append({'conversation':ci,'arm':arm,'post_retrieval_inventory':ptr(postpath),'inventory':copy_actual})
            counts={s:Counter() for s in STAGES}
            for stage in STAGES:
                actual_ids={p.stem for p in (run_dir/f'conv_{ci}'/stage/arm).glob('*.json') if not p.name.endswith('.intent.json')}
                if actual_ids!=set(expected_ids):raise ValueError('Stage file IDs differ from frozen cohort')
            for qid in expected_ids:
                artifacts={stage:sealed(run_dir/f'conv_{ci}'/stage/arm/(qid+'.json')) for stage in STAGES}
                refs={stage:ptr(run_dir/f'conv_{ci}'/stage/arm/(qid+'.json')) for stage in STAGES}
                for stage,value in artifacts.items():
                    if (value.get('protocol_sha256')!=protocol or value.get('conversation')!=ci
                            or value.get('method')!=arm or value.get('query_id')!=qid):raise ValueError('Stage identity differs')
                    status=value['status'];counts[stage][status]+=1;slot_counts[stage][status]+=1
                    if value.get('error'):errors[stage+':'+value['error'].get('type','unknown')]+=1
                qa=artifacts['qa'];reader=artifacts['reader'];judge=artifacts['judge']
                if qa['reader_status']!=reader['status'] or qa['judge_status']!=judge['status']:raise ValueError('Final stage status differs')
                for stage in ('reader','judge'):
                    if qa[stage]['path']!=refs[stage]['path'] or qa[stage]['sha256']!=refs[stage]['sha256']:raise ValueError('QA stage pointer changed')
                expected_status='complete' if reader['status']==judge['status']=='complete' else 'failed'
                if qa['status']!=expected_status:raise ValueError('Strict QA failure state was altered')
                slots.append({'conversation':ci,'arm':arm,'query_id':qid,'statuses':{s:v['status'] for s,v in artifacts.items()},'artifacts':refs})
            arm_counts[f'{ci}:{arm}']={s:dict(v) for s,v in counts.items()}
    if len(slots)!=3080 or any(sum(v.values())!=3080 for v in slot_counts.values()):raise ValueError('Slot population incomplete')
    for ci in range(10):
        state=read(run_dir/f'conv_{ci}/qa_status.json')
        matched=[s for s in slots if s['conversation']==ci]
        if (state['reader_complete']!=sum(s['statuses']['reader']=='complete' for s in matched)
                or state['judge_complete']!=sum(s['statuses']['judge']=='complete' for s in matched)
                or state['failed']!=sum(s['statuses']['qa']=='failed' for s in matched)):
            raise ValueError('Worker terminal counts disagree with sealed artifacts')
    def file_set():return sorted(p.resolve() for p in run_dir.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
    before=file_set();inventory=[ptr(p) for p in before]
    if file_set()!=before:raise ValueError('Run file population changed during inventory')
    for p,expected in list(hashes.items()):
        if ptr(p)['sha256']!=expected:raise ValueError('Source changed at final hash pass')
    final_processes=processes();reject_live_qa(final_processes,recorded_pids)
    final=snapshot(run_dir,process_reader=lambda:final_processes)
    if terminal_evidence(final,run_dir,protocol)!=stable or file_set()!=before:raise ValueError('Run changed at final process/status check')
    result={'schema':SCHEMA,'status':'all10_terminal_strict_firstpass_frozen','created_utc':datetime.now(timezone.utc).isoformat(),
        'run_dir':str(run_dir),'protocol_sha256':protocol,'source_manifest':ptr(run_dir/'run_manifest.json'),
        'process_snapshots':[ptr(p) for p in snapshot_paths],'snapshot_gap_seconds':elapsed,
        'process_checks':{'recorded_pids':sorted(recorded_pids),'matching_live_processes_before':[],'matching_live_processes_after':[],
            'checked_fields':['ProcessId','ParentProcessId','CreationDate','CommandLine'],'fresh_before':fresh['captured_utc'],'fresh_after':final['captured_utc']},
        'controller':stable['controller'],'expected_slots':3080,'slot_count':len(slots),
        'stage_counts':{s:dict(v) for s,v in slot_counts.items()},'stage_counts_by_conversation_arm':arm_counts,
        'error_types':dict(errors),'slots':slots,'file_inventory':inventory,'file_count':len(inventory),
        'original_source_native_inventories':source_inventory,'post_retrieval_copy_inventories':native_copies,
        'frozen_code':code,'gate':manifest['gate'],'plain_text_config':manifest['plain_text_config'],
        'all_read_file_hashes_unchanged':True,'original_source_native_unchanged':True,
        'strict_judge_failures_preserved':True,'model_api_calls':0,'network_calls':0,'old_run_writes':0,
        'note':'A terminal evidence seal, not Judge format recovery or metric scoring. Strict failed responses and raw caches remain untouched.',
        'seal_tool':ptr(Path(__file__))}
    result['content_sha256']=digest(result)
    with out.open('x',encoding='utf8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    return {'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'stage_counts':result['stage_counts'],'file_count':len(inventory)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--snapshot',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if len(args.snapshot)!=2:parser.error('Exactly two terminal snapshots required')
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden'))
    socket.create_connection=socket.socket.connect
    print(json.dumps(run(args.run_dir,args.snapshot,args.output),ensure_ascii=False))


if __name__=='__main__':main()
