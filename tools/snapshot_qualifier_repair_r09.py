"""Read-only R09 stage products/ledger/process evidence. Never infer health from PID alone."""
import argparse,json,subprocess,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.snapshot_evidence_exchange_r01 import snapshot,read
from tools.score_evidence_exchange_r01 import now

def snapshot_qualifier_repair(run):
    run=Path(run).resolve();r=snapshot(run)
    if not r.get('exists'):return r
    for name in ('proposer_responses','responses'):
        folder=run/name
        if not folder.exists():continue
        counts=Counter();byconv={}
        for p in folder.glob('*.json'):
            try:
                x=read(p);ci=str(x['conversation']);counts[x.get('status','missing')]+=1
                c=byconv.setdefault(ci,{'count':0,'latest_mtime':0})
                c['count']+=1
                if p.stat().st_mtime>=c['latest_mtime']:
                    c['latest_mtime']=p.stat().st_mtime;c['last_request_id']=x['request_id']
            except Exception as ex:r['read_errors'].append({'path':str(p),'error_type':type(ex).__name__})
        r['artifacts'][name]={'count':sum(counts.values()),'statuses':dict(counts),'per_conversation':byconv}
    r['native_stage_files']={}
    state_paths=list(run.glob('conv_*/*/*status.json'))
    state_paths+=list((run/'materialization').glob('**/*outcome.json'))
    state_paths+=list((run/'materialization').glob('**/*intent.json'))
    state_paths+=list((run/'retrieval').glob('**/*outcome.json'))
    for p in sorted(set(state_paths)):
        try:r['native_stage_files'][str(p.relative_to(run))]={'mtime':p.stat().st_mtime,'state':read(p)}
        except Exception as ex:r['read_errors'].append({'path':str(p),'error_type':type(ex).__name__})
    r['native_banks']={}
    for folder in sorted((run/'materialization').glob('conv_*/*')):
        if not folder.is_dir() or folder.name not in ('cheap','jev'):continue
        item={'updates':0,'update_dispatches':0,'statuses':{},'retrieval_selections':0,'latest_mtime':0}
        counts=Counter()
        for p in folder.glob('updates/*.json'):
            try:
                row=read(p);counts[row.get('status','missing')]+=1;item['updates']+=1
                if p.stat().st_mtime>=item['latest_mtime']:
                    item.update(latest_mtime=p.stat().st_mtime,last_candidate_id=row.get('candidate',{}).get('candidate_id'))
            except Exception as ex:r['read_errors'].append({'path':str(p),'error_type':type(ex).__name__})
        item['statuses']=dict(counts)
        for p in folder.glob('update_dispatch/*.json'):
            item['update_dispatches']+=1
            item['latest_mtime']=max(item['latest_mtime'],p.stat().st_mtime)
        for p in folder.glob('selections/*.json'):
            item['retrieval_selections']+=1
            item['latest_mtime']=max(item['latest_mtime'],p.stat().st_mtime)
        r['native_banks'][str(folder.relative_to(run))]=item
    r['retrieval_banks']={}
    for folder in sorted((run/'retrieval').glob('conv_*/*')):
        if not folder.is_dir() or folder.name not in ('cheap','jev'):continue
        paths=list(folder.glob('selections/*.json'))
        r['retrieval_banks'][str(folder.relative_to(run))]={
            'selection_count':len(paths),
            'latest_mtime':max((p.stat().st_mtime for p in paths),default=0)}
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    script="$r09Processes=@(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\"); $r09Ids=@($r09Processes | Where-Object {$_.CommandLine -like '*qualifier_repair*'} | ForEach-Object {$_.ProcessId}); do {$r09Before=$r09Ids.Count; $r09Ids=@($r09Ids + @($r09Processes | Where-Object {$_.ParentProcessId -in $r09Ids} | ForEach-Object {$_.ProcessId}) | Select-Object -Unique)} while ($r09Ids.Count -gt $r09Before); $r09Processes | Where-Object {$_.ProcessId -in $r09Ids} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    q=subprocess.run(['powershell.exe','-NoProfile','-Command',script],capture_output=True,text=True,encoding='utf-8',check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    r={'created_utc':now(),'runs':[snapshot_qualifier_repair(v) for v in a.run],'processes':json.loads(q.stdout or '[]'),'health_verdict':'Compare actual stage products and contexts over time, including >=10minute history; no health claim from aggregate HTTP200 or PID.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(a.output),'runs':[{'run':x['run'],'artifacts':{k:v['count'] for k,v in x.get('artifacts',{}).items()},'read_errors':x.get('read_errors',[]),'ledgers':[{k:v[k] for k in ('bytes','counts')} for v in x.get('ledgers',[])]} for x in r['runs']]},ensure_ascii=False))
if __name__=='__main__':main()
