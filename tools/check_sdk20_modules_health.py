"""Read-only activity snapshots for the fixed SDK20 module run."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pilot400_experiments_v2 import read, publish_json

def snapshot(run):
    root=Path(run);out={'utc':datetime.now(timezone.utc).isoformat(),'run':str(root),'stages':{},'conversations':{}}
    for p in root.glob('*_status.json'):out['stages'][p.stem]=read(p)
    if (root/'active_stage.lock').exists():out['active']=read(root/'active_stage.lock')
    for ci in range(10):
        conv=root/f'conv_{ci}';row={'snapshot_start_utc':datetime.now(timezone.utc).isoformat(),'stages':{},'counts':{},'api':{'bytes':0,'physical':0,'success':0,'errors':0,'cache_hits':0,'last_utc':None}}
        for p in conv.glob('*_status.json'):row['stages'][p.stem]=read(p)
        for directory in ('retrieval/session','retrieval/preseg','tags','query_types',
                          'selection/fact_types_sdk20','selection/preseg_sdk20',
                          'qa/baseline_sdk20','qa/fact_types_sdk20','qa/preseg_sdk20'):
            paths=list((conv/directory).glob('*.json'))
            # Atomically published artifacts are counted here, not revalidated.
            # Success/failure counts come from stage status and final offline audit.
            row['counts'][directory]={'artifacts':len(paths)}
        p=conv/'api/api_calls.jsonl'
        if p.exists():
            # Writer appends one whole JSON line; ignore only incomplete final line.
            raw=p.read_bytes();row['api']['bytes']=len(raw)
            lines=raw.splitlines(keepends=True)
            for line in lines:
                if not line.endswith(b'\n'):continue
                event=json.loads(line)
                row['api']['last_utc']=event['timestamp_utc']
                if event.get('cache_hit'):row['api']['cache_hits']+=1
                else:
                    row['api']['physical']+=1
                    row['api']['success' if event.get('status')==200 else 'errors']+=1
        row['snapshot_end_utc']=datetime.now(timezone.utc).isoformat()
        out['conversations'][str(ci)]=row
    out['snapshot_end_utc']=datetime.now(timezone.utc).isoformat()
    return out

def compare(before,after):
    result={}
    for ci,b in before['conversations'].items():
        a=after['conversations'][ci]
        result[ci]={'api_delta':{k:a['api'][k]-b['api'][k] for k in ('bytes','physical','success','errors','cache_hits')},
                    'artifact_delta':{k:sum(a['counts'][k].values())-sum(v.values()) for k,v in b['counts'].items()},
                    'last_api_utc':a['api']['last_utc'],'stages':a['stages']}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default=str(ROOT/'runs/sdk20_modules_v1'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args()
    if not 1<=args.interval<=60:raise ValueError('Interval must be between 1 and 60 seconds')
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError('New audit directory required')
    out.mkdir(parents=True)
    a=snapshot(args.run_dir);publish_json(out/'before.json',a)
    print(json.dumps({'snapshot':'before','utc':a['utc'],'active':a.get('active')}),flush=True)
    time.sleep(args.interval)
    b=snapshot(args.run_dir);publish_json(out/'after.json',b)
    delta=compare(a,b);publish_json(out/'delta.json',delta)
    print(json.dumps({'snapshot':'after','utc':b['utc'],'active':b.get('active'),'delta':delta}),flush=True)

if __name__=='__main__':main()
