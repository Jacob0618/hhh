"""Read-only evidence for the separate selection/Reader/Judge plain QA stages.

No automatic health verdict: compare snapshots, actual processes and history.
Does not import any experiment runner, model provider or native database client.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_experiment_health import shared_read

def read(path):return json.loads(shared_read(path).decode('utf-8-sig'))

def artifact_counts(folder):
    counts=Counter();errors=[];latest=None;intents=0
    for path in sorted(folder.glob('*.json')):
        if path.name.endswith('.intent.json'):
            intents+=1;continue
        try:
            row=read(path);counts[row.get('status','unknown')]+=1
            if row.get('error'):
                errors.append({'file':path.name,'error_type':row['error'].get('type')})
            stamp=path.stat().st_mtime
            if latest is None or stamp>latest['mtime_unix']:
                latest={'file':path.name,'mtime_unix':stamp,'status':row.get('status')}
        except (ValueError,OSError,UnicodeDecodeError) as error:
            errors.append({'file':path.name,'unreadable_type':type(error).__name__})
            counts['unreadable']+=1
    return {'counts':dict(counts),'output_files':sum(counts.values()),'intent_files':intents,
            'latest':latest,'errors':errors}

def ledger_summary(path):
    data=shared_read(path) if path.exists() else b''
    events=[];bad=[]
    for index,line in enumerate(data.splitlines()):
        try:events.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):bad.append(index+1)
    recent=[{k:r.get(k) for k in ('timestamp_utc','stage','context','request_hash','status','attempt','cache_hit')} for r in events[-16:]]
    return {'bytes':len(data),'rows':len(events),'cache_events':sum(bool(r.get('cache_hit')) for r in events),
            'physical_attempts':sum(not r.get('cache_hit') for r in events),
            'statuses':dict(Counter(str(r.get('status','cache')) for r in events)),
            'physical_statuses':dict(Counter(str(r.get('status','unknown')) for r in events if not r.get('cache_hit'))),
            'by_stage':dict(Counter(r.get('stage','unknown') for r in events)),
            'last_events':recent,'malformed_line_numbers':bad}

def snapshot(run,*,process_reader=None):
    run=Path(run).resolve();manifest=read(run/'run_manifest.json')
    arms=manifest['policy']['arms'];rows=[];pids=set()
    for ci in range(10):
        folder=run/f'conv_{ci}';states={}
        for name in ('retrieval_status.json','qa_status.json','execution_intent.json'):
            path=folder/name
            if path.exists():
                states[name]=read(path)
                if states[name].get('pid'):pids.add(states[name]['pid'])
        rows.append({'conversation':ci,'expected_per_arm':len(manifest['query_ids'][str(ci)]),
            'states':states,'arms':{arm:{stage:artifact_counts(folder/stage/arm)
                       for stage in ('selection','reader','judge','qa')} for arm in arms},
            'api':ledger_summary(folder/'api/api_calls.jsonl')})
    controller={}
    for name in ('run_status.json','execution_intent.json'):
        if (run/name).exists():
            controller[name]=read(run/name)
            if controller[name].get('pid'):pids.add(controller[name]['pid'])
    if process_reader:
        processes=process_reader()
    else:
        command="Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'python*'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],check=True,capture_output=True,
            text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        processes=json.loads(result.stdout or '[]')
        if isinstance(processes,dict):processes=[processes]
    for _ in range(5):pids.update(p['ParentProcessId'] for p in processes if p['ProcessId'] in pids)
    return {'schema':'mem0-plain-qa-health-evidence-v1','captured_utc':datetime.now(timezone.utc).isoformat(),
        'run_dir':str(run),'protocol_sha256':manifest['protocol_sha256'],'expected_slots':manifest['policy']['total'],
        'controller':controller,'conversations':rows,'processes':[p for p in processes if p['ProcessId'] in pids],
        'note':'Counts are evidence only. Unchanged processes/HTTP200 do not prove progress; compare stages and prior snapshots. Technical errors are not answer errors.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new audit output')
    value=snapshot(args.run_dir);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)
    compact=[{'conversation':r['conversation'],'states':r['states'],
              'arms':{a:{s:v['counts'] for s,v in stages.items()} for a,stages in r['arms'].items()},
              'api_rows':r['api']['rows'],'api_bytes':r['api']['bytes'],'api_statuses':r['api']['statuses']}
              for r in value['conversations']]
    print(json.dumps({'output':str(args.output.resolve()),'captured_utc':value['captured_utc'],'conversations':compact,
                      'controller':value['controller'],'processes':value['processes']},ensure_ascii=False))

if __name__=='__main__':main()
