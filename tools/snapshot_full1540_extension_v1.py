"""Read-only progress evidence for complementary1140 QA; never dispatch models."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.snapshot_refinement_qa_v1 import read, difference, STAGES
from tools.snapshot_mem0_plain_qa import artifact_counts
from research.snapshot_sdk20_judge_retest_v1 import ledger_summary, process_rows


def snapshot(run):
    run=Path(run).resolve()
    m=read(run/'run_manifest.json')
    if 'extension' not in m['schema'] or m['policy']['slots_per_arm']!=1140:
        raise ValueError('Expected frozen complementary1140 QA manifest')
    if m['phase_kind']=='firstpass':
        qs={int(k):v for k,v in m['query_ids'].items()}
        if sum(map(len,qs.values()))!=1140:raise ValueError('Wrong extension size')
    else:
        qs={ci:[j['query_id'] for j in m['jobs'] if j['conversation']==ci] for ci in range(10)}
        qs={ci:ids for ci,ids in qs.items() if ids}
    start=datetime.now(timezone.utc).isoformat()
    states,pids={},set()
    for name in ('retrieval_execution_intent.json','retrieval_status.json','retrieval_outcome.json',
                 'qa_execution_intent.json','qa_status.json','run_outcome.json'):
        p=run/name
        if p.exists():
            v=read(p);states[name]={k:x for k,x in v.items() if k!='results'}
            if type(v.get('pid')) is int:pids.add(v['pid'])
    rows,totals=[],{s:Counter() for s in STAGES}
    for ci,ids in sorted(qs.items()):
        folder=run/f'conv_{ci}';local={}
        for name in ('retrieval_execution_intent.json','qa_progress.json'):
            p=folder/name
            if p.exists():
                local[name]=read(p)
                if type(local[name].get('pid')) is int:pids.add(local[name]['pid'])
        stages={s:artifact_counts(folder/s/m['arm']) for s in STAGES}
        for s in STAGES:totals[s].update(stages[s]['counts'])
        rows.append({'conversation':ci,'expected':len(ids),'states':local,'stages':stages,
                     'api':ledger_summary(folder/'api/api_calls.jsonl')})
    procs=process_rows();wanted=str(run).casefold().replace('/','\\')
    pids.update(p['ProcessId'] for p in procs if wanted in (p.get('CommandLine')or'').casefold().replace('/','\\'))
    for _ in range(3):pids.update(p['ProcessId'] for p in procs if p['ParentProcessId'] in pids)
    return {'schema':'full1540-extension-health-v1','captured_utc':start,
            'finished_utc':datetime.now(timezone.utc).isoformat(),'run_dir':str(run),
            'protocol_sha256':m['protocol_sha256'],'phase_id':m['phase_id'],'phase_kind':m['phase_kind'],
            'expected':sum(map(len,qs.values())),'arm':m['arm'],'controller':states,
            'stage_counts':{s:dict(v) for s,v in totals.items()},'conversations':rows,
            'processes':[p for p in procs if p['ProcessId'] in pids],
            'note':'Complementary1140 only; inherited400 unchanged. Real progress requires per-worker artifacts and request deltas.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.run_dir.resolve()):raise ValueError('Fresh external audit path required')
    v=snapshot(a.run_dir)
    if a.previous:v['difference']=difference(read(a.previous),v)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v[k] for k in ('captured_utc','expected','phase_kind','stage_counts','processes')}
                     |{'output':str(a.output),'difference':v.get('difference')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
