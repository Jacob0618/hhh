"""Read-only two-snapshot health evidence for recovered native retrieval or QA."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_mem0_sparse1540_health import inventory,process_identity,read,now,timestamp,save
from tools.check_experiment_health import shared_read
OUT=ROOT/'runs/mem0_sparse1540_index_recovery_v2'
POLICY='mem0-native-sparse1540-orphan-recovery-v2'

def snapshot(stage):
    procs=inventory();out=[]
    driver='recover_mem0_sparse1540_index_v3.py' if stage=='retrieve' else 'run_mem0_sparse1540_recovered_qa_v2.py'
    pattern='recovered/*/conv_3/retrieve_status.json' if stage=='retrieve' else 'conv_*/qa_status.json'
    for p in OUT.glob(pattern):
        s=read(p);worker=process_identity(s['pid'],procs);parent=process_identity(worker['parent_pid'],procs) if worker.get('parent_pid') else {}
        worker['identity_verified']=bool(worker.get('alive') and worker.get('cim_handle_creation_match') and
            'spawn_main' in worker.get('command_line','') and parent.get('alive') and parent.get('cim_handle_creation_match') and
            str(ROOT/'tools'/driver).lower() in parent.get('command_line','').lower() and stage in parent.get('command_line','') and
            timestamp(worker['created_utc'])<=timestamp(s['started_utc'])+1)
        lp=p.parent/('native' if stage=='retrieve' else 'qa_api')/'api_calls.jsonl'
        raw=shared_read(lp) if lp.exists() else b'';rows=[];malformed=0
        lines=raw.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except (ValueError,UnicodeError):
                if i<len(lines)-1 or line.endswith(b'\n'):malformed+=1
        scope=[r for r in rows if r.get('context',{}).get('phase')==stage and
            r.get('context',{}).get('architecture')==POLICY and r.get('context',{}).get('conversation')==s['conversation']]
        bad_scope=len(rows)-len(scope)
        physical=[r for r in scope if not r.get('cache_hit')]
        products=len(list((p.parent/'retrieval').glob('*.json'))) if stage=='retrieve' else len(list((p.parent/'qa').glob('*/*.json')))
        out.append({'key':s.get('arm',str(s['conversation'])),'state':s,'worker':worker,'parent':parent,
            'products':products,'ledger':str(lp),'bytes':len(raw),'malformed':malformed,'bad_scope':bad_scope,
            'http200':sum(r.get('status')==200 for r in physical),'http_errors':sum(r.get('status')!=200 for r in physical),
            'cache_hits':sum(r.get('cache_hit') is True for r in scope),'last_request':scope[-1] if scope else None})
    coordinator=None
    if stage=='qa':
        cp=OUT/'qa_coordinator_status.json'
        if cp.exists():
            cs=read(cp);identity=process_identity(cs['pid'],procs)
            identity['command_verified']=str(ROOT/'tools'/driver).lower() in identity.get('command_line','').lower() and 'qa' in identity.get('command_line','')
            coordinator={'state':cs,'identity':identity}
    return {'captured_utc':now(),'stage':stage,'rows':out,'coordinator':coordinator,
        'expected_tasks':10 if stage=='qa' else 2,'model_calls':0,'credentials_loaded':False}
def compare(first,second,history=None):
    before={r['key']:r for r in first['rows']};result=[]
    historical={r['key']:r for r in (history or {}).get('rows',[])}
    for row in second['rows']:
        old=before.get(row['key'],{});delta={k:row[k]-old.get(k,0) for k in ('products','bytes','http200','http_errors','cache_hits')}
        s=row['state'];terminal=s['status']!='running'
        if terminal:verdict='terminal_requires_integrity_review'
        elif not row['worker']['identity_verified']:verdict='identity_investigate'
        elif delta['products']>0:verdict='persisted_progress'
        elif delta['http200']>0 or delta['cache_hits']>0:verdict='requests_progress_pending_products'
        else:verdict='no_progress_investigate'
        if row['malformed'] or row['bad_scope'] or any(v<0 for v in delta.values()):verdict='ledger_integrity_investigate'
        if delta['http_errors']>0 and delta['http200']==0 and delta['products']==0 and not terminal:verdict='errors_only_investigate'
        h=historical.get(row['key'])
        if h and not terminal and timestamp(second['captured_utc'])-timestamp(history['captured_utc'])>=600:
            if row['worker'].get('creation_time_100ns')==h['worker'].get('creation_time_100ns') and row['products']==h['products']:
                verdict='ten_minutes_without_products_investigate'
        result.append({'key':row['key'],'status':s['status'],'done':s['done'],'expected':s['expected'],
            'completed':s.get('completed'),'failed':s.get('failed'),'delta':delta,'verdict':verdict})
    issues=[]
    if not second['rows']:issues.append('no_stage_workers_or_products_observed')
    if second['stage']=='qa':
        co=second.get('coordinator')
        if not co:issues.append('missing_coordinator_state')
        elif co['state']['status']=='running':
            if not (co['identity'].get('alive') and co['identity'].get('cim_handle_creation_match') and co['identity'].get('command_verified')):
                issues.append('active_coordinator_identity_missing')
        elif len(second['rows'])!=10 or any(r['state']['status']=='running' for r in second['rows']):
            issues.append('coordinator_terminal_with_missing_or_active_workers')
    return {'first_utc':first['captured_utc'],'second_utc':second['captured_utc'],'stage':second['stage'],
        'rows':result,'global_issues':issues,'pending_state_count':second['expected_tasks']-len(second['rows']),
        'requires_investigation':bool(issues) or any('investigate' in x['verdict'] for x in result),'model_calls':0}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('retrieve','qa'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--interval',type=int,default=30);p.add_argument('--previous',type=Path)
    a=p.parse_args()
    if not 1<=a.interval<=60 or a.output.resolve().is_relative_to(ROOT/'runs'):p.error('Use1–60secs and new research audit directory')
    a.output.mkdir(parents=True,exist_ok=False);first=snapshot(a.stage);save(a.output/'first.json',first)
    time.sleep(a.interval);second=snapshot(a.stage);save(a.output/'second.json',second)
    report=compare(first,second,read(a.previous) if a.previous else None);save(a.output/'report.json',report)
    print(json.dumps(report,indent=2))
