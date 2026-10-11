"""Read-only live observer. Never opens native indexes or initializes an API."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_experiment_health import shared_read
RUN = ROOT / 'runs/five_baselines_gemini25_v1_20261008'

def read(p):
    return json.loads(shared_read(p).decode('utf-8'))

def snapshot():
    errors=[]
    results={}
    for folder in sorted(RUN.iterdir()) if RUN.exists() else []:
        if not folder.is_dir(): continue
        item={'states':{},'by_conversation':{},'api':{}}
        for p in [folder/'status.json',*folder.glob('results/c*/status.json')]:
            if p.exists():
                try: item['states'][str(p.relative_to(folder))]=read(p)
                except Exception as e: errors.append({'path':str(p),'type':type(e).__name__})
        for ci in range(10):
            native=folder/'work/native'/f'c{ci}'
            row={'retrieval':len(list((folder/'results'/f'c{ci}'/'retrieval').glob('*.json'))),
                 'qa':len(list((folder/'results/firstpass').glob(f'c{ci}_q*/outcome.json'))),
                 'reader':len(list((folder/'results/firstpass').glob(f'c{ci}_q*/reader.json'))),
                 'judge':len(list((folder/'results/firstpass').glob(f'c{ci}_q*/judge.json')))}
            row['retrieval']-=len(list((folder/'results'/f'c{ci}'/'retrieval').glob('*.intent.json')))
            cp=native/'checkpoint.json'
            if cp.exists():
                try:
                    state=read(cp)
                    row.update({k:state[k] for k in ['completed_notes','completed_sessions','note_count','build_complete'] if k in state})
                except Exception as e: errors.append({'path':str(cp),'type':type(e).__name__})
            item['by_conversation'][str(ci)]=row
        for p in folder.glob('work/api_*/api_calls.jsonl'):
            try:
                raw=shared_read(p)
                rows=[json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
                item['api'][p.parent.name]={'bytes':len(raw),'rows':len(rows),'latest':rows[-1].get('timestamp_utc') if rows else None,
                    'physical':sum(not r.get('cache_hit') for r in rows),'cache':sum(bool(r.get('cache_hit')) for r in rows),
                    'statuses':dict(Counter(str(r.get('status')) for r in rows if not r.get('cache_hit'))),
                    'by_stage':dict(Counter(r.get('stage') for r in rows)),
                    'by_conversation':dict(Counter(str(r.get('context',{}).get('conversation')) for r in rows)),
                    'last_identity':[{k:r.get(k) for k in ['timestamp_utc','context','stage','request_hash','status','attempt']} for r in rows[-15:]]}
            except Exception as e: errors.append({'path':str(p),'type':type(e).__name__})
        results[folder.name]=item
    cmd="Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'five_baselines|baselines_'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 4 -Compress"
    proc=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True)
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'methods':results,'read_errors':errors,
        'd_free_bytes':shutil.disk_usage(ROOT).free,'processes':json.loads(proc.stdout) if proc.stdout.strip() else [],
        'process_probe_ok':proc.returncode==0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--compare',type=Path)
    args=p.parse_args();value=snapshot()
    if args.compare:
        old=read(args.compare)
        value['comparison']={'prior':str(args.compare),'actual_interval_seconds':(datetime.fromisoformat(value['created_utc'])-datetime.fromisoformat(old['created_utc'])).total_seconds(),
            'by_method':{m:{'qa_delta':sum(r['qa'] for r in v['by_conversation'].values())-sum(r['qa'] for r in old.get('methods',{}).get(m,{}).get('by_conversation',{}).values())} for m,v in value['methods'].items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(args.output),'methods':list(value['methods']),'read_errors':len(value['read_errors'])}))
