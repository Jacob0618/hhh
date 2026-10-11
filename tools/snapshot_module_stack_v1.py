"""Compact read-only module-stack health evidence; no API or credential reads.

Example: --run-dir runs/module_stack_v1 --output health_01.json
Take a second snapshot30s later with --compare health_01.json. A PID or HTTP200
alone never produces a health verdict. Only the explicit output file is written.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_experiment_health import shared_read

SCHEMA='module-stack-health-snapshot-v1'
MAX_OUTPUT_BYTES=524288
PHASES=('support','closure','rank')
SKIP_DIRS={'api_cache','request_intents','cache_lineage','requests','units','selections','.git'}


def utc(timestamp=None):
    return datetime.fromtimestamp(timestamp,timezone.utc).isoformat() if timestamp is not None else datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(shared_read(path).decode('utf-8-sig'))


def error_row(path,exc):
    # Exception messages can contain content; publish the type, never the content.
    return {'path':str(path),'error_type':type(exc).__name__}


def state(path,errors):
    if not path.is_file():return {'exists':False}
    try:
        d=read(path)
        if not isinstance(d,dict):raise ValueError('State object required')
        fields=('schema','status','phase','phase_kind','terminal','recovery','pid','started_utc',
            'updated_utc','finished_utc','created_utc','total','expected_requests','terminal_requests',
            'counts','conversations','outcomes')
        out={k:d[k] for k in fields if k in d and k not in ('conversations','outcomes')}
        conversations=d.get('conversations')
        if isinstance(conversations,dict):
            out['conversations']={str(k):{x:v[x] for x in ('expected','terminal','last_request_id','last_activity_utc') if x in v}
                for k,v in conversations.items() if isinstance(v,dict)}
        if isinstance(d.get('outcomes'),list):
            out['outcome_conversation_count']=len(d['outcomes'])
            out['outcome_terminal_conversations']=sum(x.get('terminal') is True for x in d['outcomes'] if isinstance(x,dict))
            out['outcome_local_error_count']=sum(len(x.get('local_errors',[])) for x in d['outcomes']
                if isinstance(x,dict) and isinstance(x.get('local_errors',[]),list))
        if isinstance(d.get('requests'),list):out['request_refs_count']=len(d['requests'])
        if isinstance(d.get('units'),list):out['unit_refs_count']=len(d['units'])
        errs=d.get('local_errors',[])
        out['local_error_count']=len(errs) if isinstance(errs,list) else None
        return {'exists':True,'mtime_utc':utc(path.stat().st_mtime),'state':out}
    except Exception as exc:errors.append(error_row(path,exc));return {'exists':True,'read_error':type(exc).__name__}


def artifacts(folder,errors):
    counts=Counter();perconv={};latest=None;bytes_total=0;unreadable=0
    for p in sorted(folder.glob('*.json')) if folder.is_dir() else []:
        if p.name.endswith('.intent.json'):continue
        try:
            row=read(p)
            if not isinstance(row,dict):raise ValueError('Artifact object required')
            st=str(row.get('status','status_missing'));counts[st]+=1
            ci=row.get('conversation')
            if ci is None:
                match=re.search(r'(?:^|[/\\])conv_(\d+)(?:[/\\]|$)',str(p))
                ci=match.group(1) if match else 'unknown'
            stamp=p.stat().st_mtime;bytes_total+=p.stat().st_size
            item=perconv.setdefault(str(ci),{'count':0,'latest_mtime_utc':None})
            item['count']+=1;when=utc(stamp)
            if item['latest_mtime_utc'] is None or when>item['latest_mtime_utc']:item['latest_mtime_utc']=when
            if latest is None or when>latest['mtime_utc']:
                latest={'filename':p.name,'mtime_utc':when,'status':st}
        except Exception as exc:unreadable+=1;errors.append(error_row(p,exc))
    return {'exists':folder.is_dir(),'readable_count':sum(counts.values()),'unreadable_count':unreadable,
        'file_count':sum(counts.values())+unreadable,'bytes':bytes_total,'statuses':dict(counts),'latest':latest,'per_conversation':perconv}


def prepared_files(folder,errors):
    """Observe build progress from file metadata before a manifest is published."""
    count=0;bytes_total=0;latest=None;unreadable=0
    for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
        if path.is_symlink():continue
        try:
            stat=path.stat()
            if not path.is_file():continue
            count+=1;bytes_total+=stat.st_size;when=utc(stat.st_mtime)
            if latest is None or when>latest['mtime_utc']:
                latest={'filename':path.name,'mtime_utc':when,'bytes':stat.st_size}
        except OSError as exc:
            unreadable+=1;errors.append(error_row(path,exc))
    return {'exists':folder.is_dir(),'file_count':count,'bytes':bytes_total,
        'latest':latest,'metadata_error_count':unreadable,
        'content_read':False,'semantics':'Observed JSON file metadata; manifest completeness and validity are not inferred.'}


def ledger(path,errors):
    raw=shared_read(path);complete=raw[:raw.rfind(b'\n')+1]
    counts=Counter();groups=Counter();perconv={};latest=None;malformed=0
    for line in complete.splitlines():
        if not line.strip():continue
        try:
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('Ledger object required')
            cached=row.get('cache_hit') is True;kind='cache' if cached else 'physical'
            counts[kind]+=1;counts['rows']+=1
            if not cached:counts['status_'+str(row.get('status'))]+=1
            if row.get('usage_unknown') is True:counts['unknown_usage_rows']+=1
            ctx=row.get('context') or {};ctx=ctx if isinstance(ctx,dict) else {}
            stage=str(row.get('stage'));phase=str(ctx.get('phase'));provider=str(row.get('provider'))
            groups['|'.join((provider,stage,phase,str(row.get('status')),kind))]+=1
            last={k:row.get(k) for k in ('timestamp_utc','stage','status','cache_hit','attempt')}
            last.update(provider=row.get('provider'),phase=ctx.get('phase'),method=ctx.get('method'),
                request_id=ctx.get('request_id'),request_hash_prefix=str(row.get('request_hash',''))[:16])
            stamp=str(row.get('timestamp_utc') or '')
            if latest is None or stamp>str(latest.get('timestamp_utc') or ''):latest=last
            ci=str(ctx.get('conversation','unknown'))
            if ci not in perconv or stamp>str(perconv[ci].get('timestamp_utc') or ''):perconv[ci]=last
        except Exception:malformed+=1
    if malformed:errors.append({'path':str(path),'error_type':'MalformedCompleteLedgerRows','count':malformed})
    return {'path':str(path),'bytes':len(raw),'complete_bytes':len(complete),
        'in_flight_tail_bytes':len(raw)-len(complete),'mtime_utc':utc(path.stat().st_mtime),
        'counts':dict(counts),'provider_stage_phase_status_cache_counts':dict(groups),
        'latest_request':latest,'conversation_latest_requests':perconv,'malformed_complete_rows':malformed}


def walk_ledgers(run):
    for current,dirs,files in os.walk(run,followlinks=False):
        dirs[:]=[d for d in dirs if d not in SKIP_DIRS and not (Path(current)/d).is_symlink()]
        if 'api_calls.jsonl' in files:yield Path(current)/'api_calls.jsonl'


def snapshot(run):
    run=Path(run).resolve();errors=[]
    if not run.is_dir():return {'run_dir':str(run),'exists':False}
    states={}
    for p in sorted(run.glob('*.json')):
        if p.name=='status.json' or p.name=='outcome.json' or p.name.endswith(('_status.json','_outcome.json')):
            states[p.name]=state(p,errors)
    phases={}
    for phase in PHASES:
        producers={}
        for kind in ('firstpass','recovery_v1'):
            folder=run/'phases'/phase/kind
            producers[kind]={'exists':folder.is_dir(),'status':state(folder/'status.json',errors),
                'execution_intent':state(folder/'execution_intent.json',errors),
                'outcome':state(folder/'outcome.json',errors),'responses':artifacts(folder/'responses',errors)}
        phase_folder=run/'phases'/phase
        phases[phase]={**producers,'directory_exists':phase_folder.is_dir(),
            'prepared_manifest':state(phase_folder/'manifest.json',errors),
            'prepared_files':{kind:prepared_files(phase_folder/kind,errors) for kind in ('units','requests')}}
    qa={}
    for conv in sorted(run.glob('conv_*')):
        if not conv.is_dir() or conv.is_symlink():continue
        for stage in ('reader','judge','qa','compat'):
            for folder in sorted((conv/stage).iterdir()) if (conv/stage).is_dir() else []:
                if folder.is_dir() and not folder.is_symlink():qa[str(folder.relative_to(run))]=artifacts(folder,errors)
        for p in sorted(conv.glob('*_status.json')):states[str(p.relative_to(run))]=state(p,errors)
    ledgers=[]
    for p in sorted(walk_ledgers(run)):
        try:ledgers.append(ledger(p,errors))
        except Exception as exc:errors.append(error_row(p,exc))
    counts=sum((Counter(x['counts']) for x in ledgers),Counter());perconv={}
    for item in ledgers:
        for ci,event in item['conversation_latest_requests'].items():
            if ci not in perconv or str(event.get('timestamp_utc') or '')>str(perconv[ci].get('timestamp_utc') or ''):perconv[ci]=event
    return {'run_dir':str(run),'exists':True,'states':states,'phases':phases,'qa_artifacts':qa,'api_ledgers':ledgers,
        'api_totals':{'bytes':sum(x['bytes'] for x in ledgers),'complete_bytes':sum(x['complete_bytes'] for x in ledgers),'counts':dict(counts)},
        'conversation_latest_requests':perconv,'read_error_count':len(errors),'read_errors':errors[:20]}


def _redact_command(command):
    command=str(command or '')
    command=re.sub(r'(?i)(\bBearer\s+)([^\s\"\']+)',r'\1[REDACTED]',command)
    command=re.sub(r'(?i)((?:--?|\b)(?:api[-_]?key|token|password|secret|authorization)\s*(?:=|\s)\s*)("[^"]*"|\S+)',r'\1[REDACTED]',command)
    command=re.sub(r'(https?://)[^\s/@:]+:[^\s/@]+@',r'\1[REDACTED]@',command)
    return command


def processes():
    script="""[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -like '*module_stack*' } |
ForEach-Object { [PSCustomObject]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;creation_utc=$_.CreationDate.ToUniversalTime().ToString('o');command_line=$_.CommandLine} } | ConvertTo-Json -Depth 3 -Compress"""
    try:
        p=subprocess.run(['powershell.exe','-NoProfile','-Command',script],capture_output=True,text=True,encoding='utf-8',check=True,
            timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        rows=json.loads(p.stdout or '[]');rows=rows if isinstance(rows,list) else [rows]
        for row in rows:
            row['command_line']=_redact_command(row.get('command_line'))
            row['role']='snapshot_observer' if row.get('pid')==os.getpid() else 'module_stack_python_process'
        return {'rows':rows,'error':None,'presence_is_not_health':True}
    except Exception as exc:return {'rows':[],'error':type(exc).__name__,'presence_is_not_health':True}


def compare(old,new):
    older={r['run_dir']:r for r in old.get('runs',[])};out=[]
    for run in new['runs']:
        prior=older.get(run['run_dir'])
        if prior is None:out.append({'run_dir':run['run_dir'],'prior_run_missing':True});continue
        before=prior.get('api_totals',{});after=run.get('api_totals',{})
        a=before.get('counts',{});b=after.get('counts',{})
        delta={k:b.get(k,0)-a.get(k,0) for k in sorted(set(a)|set(b))}
        phase={};prepared={}
        for name in PHASES:
            for kind in ('units','requests'):
                x=prior.get('phases',{}).get(name,{}).get('prepared_files',{}).get(kind,{})
                y=run.get('phases',{}).get(name,{}).get('prepared_files',{}).get(kind,{})
                prepared[name+'/'+kind]={'file_count_delta':y.get('file_count',0)-x.get('file_count',0),
                    'bytes_delta':y.get('bytes',0)-x.get('bytes',0),'latest':y.get('latest')}
            for kind in ('firstpass','recovery_v1'):
                x=prior.get('phases',{}).get(name,{}).get(kind,{});y=run.get('phases',{}).get(name,{}).get(kind,{})
                phase[name+'/'+kind]={'response_file_delta':y.get('responses',{}).get('file_count',0)-x.get('responses',{}).get('file_count',0),
                    'before_status':x.get('status',{}).get('state',{}).get('status'),'after_status':y.get('status',{}).get('state',{}).get('status'),
                    'terminal_requests_delta':y.get('status',{}).get('state',{}).get('terminal_requests',0)-x.get('status',{}).get('state',{}).get('terminal_requests',0)}
        qa={k:run.get('qa_artifacts',{}).get(k,{}).get('file_count',0)-prior.get('qa_artifacts',{}).get(k,{}).get('file_count',0)
            for k in sorted(set(prior.get('qa_artifacts',{}))|set(run.get('qa_artifacts',{})))}
        observable=(any(delta.values())
            or any(v['file_count_delta'] or v['bytes_delta'] for v in prepared.values())
            or any(v['response_file_delta'] or v['terminal_requests_delta'] or v['before_status']!=v['after_status'] for v in phase.values()) or any(qa.values()))
        out.append({'run_dir':run['run_dir'],'api_bytes_delta':after.get('bytes',0)-before.get('bytes',0),'api_count_deltas':delta,
            'phases':phase,'prepared_file_deltas':prepared,'qa_file_deltas':qa,'observable_counters_changed':bool(observable),
            'interpretation':'Counters changed; assess valid results and workload.' if observable else 'No observed counter change; long request/offline work may still be active.'})
    try:elapsed=(datetime.fromisoformat(new['captured_utc'])-datetime.fromisoformat(old['captured_utc'])).total_seconds()
    except Exception:elapsed=None
    return {'elapsed_seconds':elapsed,'runs':out,'not_a_health_verdict':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new snapshot output; no overwrite')
    data={'schema':SCHEMA,'captured_utc':utc(),'model_calls':0,'credential_reads':0,
        'runs':[snapshot(r) for r in a.run_dir],'processes':processes(),'health_verdict':None,
        'health_semantics':'Compare counters and valid artifacts across snapshots; PID presence/HTTP200 do not prove health.'}
    try:
        disk=shutil.disk_usage('D:/');data['disk_D']={'free_bytes':disk.free,'total_bytes':disk.total,'used_bytes':disk.used}
    except Exception as exc:data['disk_D']={'error':type(exc).__name__}
    if a.compare:data['comparison']=compare(read(a.compare),data);data['comparison_source']=str(a.compare.resolve())
    raw=json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8')
    if len(raw)>MAX_OUTPUT_BYTES:
        # Preserve counts and deltas; omit redundant per-folder conversation detail.
        for run in data['runs']:
            for folder in run.get('qa_artifacts',{}).values():folder.pop('per_conversation',None)
            for item in run.get('api_ledgers',[]):item.pop('conversation_latest_requests',None)
        data['compact_detail_reduced']=True
        raw=json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8')
    if len(raw)>MAX_OUTPUT_BYTES:raise ValueError('Snapshot exceeds compact524288byte output limit')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(raw+b'\n')
    print(json.dumps({'output':str(a.output.resolve()),'bytes':len(raw)+1,'runs':len(data['runs']),
        'read_errors':sum(x.get('read_error_count',0) for x in data['runs']),'model_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
