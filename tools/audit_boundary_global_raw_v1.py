"""Read-only pair/raw/decision replay, independent of a damaged shared ledger."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_boundary_variants_v1 as boundary
from tools import run_global_segment_grouping as original
from tools.run_sdk20_modules import save_new, now
from jev_mem.local_bm25_context_v2 import forbid_network

SCHEMA='boundary-global-pair-raw-independent-audit-v1'

def ledger_rows(data):
    rows=[];blank=[];malformed=[]
    for number,line in enumerate(data.decode('utf-8-sig').splitlines(),1):
        if not line.strip():blank.append(number);continue
        try:
            value=json.loads(line)
            if not isinstance(value,dict):raise ValueError('row is not an object')
            rows.append((number,value))
        except (ValueError,TypeError) as exc:
            malformed.append({'line':number,'length':len(line),'error_type':type(exc).__name__})
    return rows,blank,malformed

def ledger_mismatch(row,expected):
    return (row.get('context')!=expected['context'] or row.get('provider')!='jev'
        or row.get('model')!=original.grouping.MODEL or row.get('operation')!='systemone'
        or row.get('stage')!='global_segment_same_event' or row.get('cache_hit') not in (None,False)
        or type(row.get('attempt')) is not int or row['attempt'] not in (1,2,3)
        or (row.get('status')==200 and row.get('usage')!=expected['usage']))

def audit(run):
    run=Path(run).resolve();inputs={}
    def remember(path):
        path=Path(path).resolve();data=path.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        if path in inputs and inputs[path]!=digest:raise ValueError('Source changed during audit')
        inputs[path]=digest
        return data
    def read(path):return json.loads(remember(path).decode('utf-8-sig'))
    manifest=boundary.global_manifest(run);remember(run/'manifest.json')
    by_hash={};conversations=[]
    for ci in range(10):
        planpath=Path(manifest['plans'][str(ci)]['path']);remember(planpath)
        plan=original.load_plan(manifest,ci);folder=run/'conversations'/f'conv_{ci}'
        terminal=original.verify_seal(read(folder/'grouping.json'))
        if terminal['status']!='complete' or terminal['plan_sha256']!=manifest['plans'][str(ci)]['sha256']:
            raise ValueError('Grouping is not bound complete')
        observed=terminal['trace']['all_observed_pair_scores'];initial={e['pair_id'] for e in plan['candidate_pairs']}
        if not initial<=set(observed):raise ValueError('Initial pairs not all observed')
        files={p.stem:p for p in (folder/'pairs').glob('*.json')}
        if set(files)!=set(observed):raise ValueError('Terminal scores and pair artifacts differ')
        for key in sorted(observed):
            row=read(files[key]);score=original.validate_score(row,plan,manifest,row['left_index'],row['right_index'])
            if (row['conversation']!=ci or row['schema']!=original.grouping.VERSION
                or row['kind']!=('initial' if key in initial else 'complete_link_supplemental')
                or row['criterion_sha256']!=original.grouping.fingerprint({'same_concrete_event':original.grouping.CRITERION})):
                raise ValueError('Pair criterion or conversation/kind binding differs')
            if score!=observed[key]:raise ValueError('Trace score differs from validated raw')
            event=row['api_events'][0];cache=Path(event['cache_path']);raw=read(cache)
            intentpath=run/'api/request_intents'/(event['request_hash']+'.json');intent=read(intentpath)
            if (event['source']!='new_request' or intent['request_hash']!=event['request_hash']
                or intent['signature']!=raw['request'] or intent['context']!=event['context']
                or intent['stage']!=event['stage'] or intent['retries']!=2):
                raise ValueError('Durable intent differs from pair/raw request')
            if event['request_hash'] in by_hash:raise ValueError('Unexpected request hash reuse across pair contexts')
            by_hash[event['request_hash']]={'conversation':ci,'pair_id':key,'kind':row['kind'],
                'pair_artifact':str(files[key].resolve()),'raw_cache':str(cache.resolve()),
                'raw_sha256':event['cache_sha256'],'context':event['context'],
                'usage':raw['response'].get('usageMetadata',raw['response'].get('usage',{}))}
        replay=original.grouping.complete_link_groups(plan,{k:observed[k] for k in initial},
            supplemental=lambda l,r:observed[original.grouping.pair_key(l,r)],
            threshold=.7,supplemental_budget=2*len(plan['segments']))
        if any(terminal[k]!=value for k,value in replay.items()):raise ValueError('Full grouping decision replay differs')
        conversations.append({'conversation':ci,'initial_pairs':len(initial),'supplemental_pairs':len(observed)-len(initial),
            'validated_pairs':len(observed),'groups':len(replay['groups']),'full_decision_replay':True})
    intentset={p.stem for p in (run/'api/request_intents').glob('*.json')}
    cacheset={p.stem for p in (run/'api/api_cache').glob('*.json')}
    if intentset!=set(by_hash) or cacheset!=set(by_hash):raise ValueError('Orphan/missing global intent or raw cache')
    rows,blank,malformed=ledger_rows(remember(run/'api/api_calls.jsonl'))
    indexed=defaultdict(list);unbound=[];mismatch=[];counts=Counter()
    for lineno,row in rows:
        key=row.get('request_hash');indexed[key].append((lineno,row));counts[str(row.get('status'))]+=1
        if key not in by_hash:unbound.append({'line':lineno,'request_hash':key});continue
        expected=by_hash[key]
        if ledger_mismatch(row,expected):
            mismatch.append({'line':lineno,'request_hash':key})
    missing=[];attempt_gaps=[]
    for key,expected in by_hash.items():
        entries=indexed[key];success=[r for _,r in entries if type(r.get('status')) is int and 200<=r['status']<300]
        if len(success)!=1:missing.append({'request_hash':key,**{k:v for k,v in expected.items() if k!='usage'},'valid_success_rows':len(success)})
        attempts=[r.get('attempt') for _,r in entries]
        if not attempts or any(type(a) is not int for a in attempts) or sorted(attempts)!=list(range(1,max(attempts)+1)):
            attempt_gaps.append({'request_hash':key,'recorded_attempts':attempts})
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in inputs.items()):raise ValueError('Global source snapshot changed')
    complete=not (blank or malformed or unbound or mismatch or missing or attempt_gaps)
    return {'schema':SCHEMA,'created_utc':now(),'run_dir':str(run),'arm':manifest['boundary_variant'],
        'result_evidence_complete':True,'all_pair_scores_and_full_decisions_replayed':True,
        'conversations':conversations,'unique_pair_raw_requests':len(by_hash),
        'ledger':{'valid_rows':len(rows),'status_counts':dict(counts),'blank_lines':blank,'malformed_lines':malformed,
            'unbound_rows':unbound,'mismatched_rows':mismatch,'missing_or_duplicate_successes':missing,
            'attempt_sequence_gaps':attempt_gaps,'physical_attempt_ledger_complete':complete},
        'cost_usd':None,'transport_attempt_count_exact':None if not complete else len(rows),
        'result_integrity_is_not_billing_integrity':True,'model_calls':0,'source_writes':0,
        'input_sha256':{str(p):h for p,h in inputs.items()},
        'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Raw caches attest request/signature/parsed response and preserved usage, not HTTP status/headers or all transport attempts.',
            'Successful raw responses do not repair missing physical ledger rows. Known retries and damaged rows do not imply an exact total attempt count.',
            'Blank/malformed rows are preserved in the original ledger. This derived audit neither recreates nor substitutes physical ledger rows.',
            'Thread-only provider logging and observed damage are consistent with cross-process interference; exact lost byte mechanism is not established.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with forbid_network():result=audit(a.run_dir)
    save_new(a.output,result)
    print(json.dumps({k:result[k] for k in ('arm','unique_pair_raw_requests','result_evidence_complete')}))

if __name__=='__main__':main()
