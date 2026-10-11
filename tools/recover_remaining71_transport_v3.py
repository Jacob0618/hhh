"""One separately reviewed attempt for the v2 terminal transport-only failure."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_pilot400_remaining71_v2 as driver
from tools import pilot400_recovery_qa_v1 as qa
from tools.pilot400_recovery_writer_v1 import read,sha,save_once
from tools.run_pilot400_experiments_v2 import now
from jev_mem.pilot400 import fingerprint
from jev_mem.providers import ModelAPI

OUT=ROOT/'runs/pilot400_recovery_transport_v3'
POLICY='remaining71-reviewed-transport-recovery-v3'
POLICY_PATH=ROOT/'research/pilot400_remaining71_transport_recovery_v3_2026-09-28.md'

def prepare():
    old=driver.DEFAULT_OUTPUT
    status=read(old/'qa_remaining71_status.json')
    if (status['status'],status['done'],status['completed'],status['failed'])!=('requires_review',71,70,1):
        raise ValueError('Expected exact reviewed v2 terminal state')
    records=driver.collect(old);failures=[]
    for r in records:
        _,p=driver._paths(old,r)
        row=qa._verify_sealed(read(p))
        if row['status']=='failed':failures.append((r,p,row))
    if len(failures)!=1 or failures[0][0]['q']['id']!='c1_q43':
        raise ValueError('Reviewed terminal transport population changed')
    r,p,row=failures[0]
    if row['result'] is not None or len(row['api_events'])!=1:
        raise ValueError('Failure has a result or unexpected downstream calls')
    event=row['api_events'][0]
    if (event['source']!='failed_or_unknown_response' or event['stage']!=qa.evaluation.READER_STAGE
            or 'SSLEOFError' not in row['error']['message']):
        raise ValueError('Only the reviewed Reader TLS EOF is eligible')
    digest=event['request_hash']; intent=read(event['intent_path'])
    if qa.request_hash(intent['request'])!=digest or (old/'api_cache'/(digest+'.json')).exists():
        raise ValueError('Original request intent or no-response status changed')
    ledger=[json.loads(s) for s in (old/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    calls=[x for x in ledger if x.get('context',{}).get('phase')=='pilot_remaining71_qa'
           and x.get('context',{}).get('query_id')==r['q']['id']]
    if (len(calls)!=1 or calls[0]['status']!='transport_error' or not calls[0].get('usage_unknown')
            or calls[0]['request_hash']!=digest or calls[0].get('cache_hit')):
        raise ValueError('Original failed context/unknown usage ledger changed')
    retrieval_path,_=driver._paths(old,r)
    retrieval=qa._verify_sealed(read(retrieval_path))
    if retrieval!=driver.retrieval_for(r):raise ValueError('Original selection changed')
    binding={**r['binding'],'retrieval_sha256':sha(retrieval_path),
             'selected_sha256':fingerprint(retrieval['selected'])}
    if row['binding']!=binding:raise ValueError('Original failure binding changed')
    prepared=qa.evaluation.prepare_reader_input(r['q']['question'],retrieval['selected'],
        reference_date=binding['reference_date'])
    signature=qa.generation_signature(qa._json(prepared['payload']),model=qa.evaluation.READER_MODEL,
        system=qa.evaluation.READER_SYSTEM,json_output=True,max_output=qa.evaluation.MAX_OUTPUT,
        schema=qa.READER_SCHEMA)
    if signature!=intent['request']:raise ValueError('Retry Reader is not the exact original payload')
    plan={'schema':POLICY,'expected':1,'query_id':r['q']['id'],'conversation':1,'provider_retries':0,
        'original_status_sha256':sha(old/'qa_remaining71_status.json'),
        'original_failure_path':str(p),'original_failure_sha256':sha(p),
        'original_intent_path':event['intent_path'],'original_intent_sha256':sha(event['intent_path']),
        'original_ledger_sha256':sha(old/'api_calls.jsonl'),'original_request_hash':digest,
        'retrieval_path':str(retrieval_path),'retrieval_sha256':sha(retrieval_path),
        'binding':binding,'original_server_usage_unknown':True,'policy_sha256':sha(POLICY_PATH),
        'driver_sha256':sha(__file__)}
    return r,retrieval,plan

def run():
    r,retrieval,plan=prepare()
    OUT.mkdir(parents=True,exist_ok=True)
    with driver.run_lock(OUT):
        save_once(OUT/'plan.json',plan)
        path=OUT/'conversations/conv_1/qa'/driver.ARM/'c1_q43.json'
        if path.exists():raise ValueError('Separate attempt already has a terminal result; do not rerun')
        base=ModelAPI(OUT,retries=0,context={'architecture':POLICY,'driver_stage':'qa',
            'conversation':1,'query_id':'c1_q43','method':driver.ARM,'phase':'pilot_remaining71_qa',
            'recovery_version':3})
        api=qa.ResponseSchemaAPI(base);started=now();result=None;error=None
        try:result=qa.evaluate_new(r['q'],retrieval['selected'],plan['binding']['reference_date'],api)
        except Exception as exc:
            error={'type':type(exc).__name__,'stage':api.events[-1]['stage'] if api.events else 'qa',
                'message':api._redact(str(exc))[:1800]}
        row=qa._sealed({'schema':POLICY,'query_id':'c1_q43','conversation':1,'category':r['q']['category'],
            'arm':driver.ARM,'binding':plan['binding'],'status':'complete' if result is not None else 'failed',
            'result':result,'error':error,'selected_ids':[f['id'] for f in retrieval['selected']],
            'retrieval_path':plan['retrieval_path'],'api_events':api.events,'started_utc':started,
            'finished_utc':now(),'technical_recovery_plan_sha256':sha(OUT/'plan.json')})
        save_once(path,row)
        status={'schema':POLICY,'status':row['status'],'expected':1,'completed':int(result is not None),
            'failed':int(result is None),'query_id':'c1_q43','qa_sha256':sha(path),'finished_utc':now()}
        save_once(OUT/'status.json',status)
        print(json.dumps(status,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--check',action='store_true');a.add_argument('--run',action='store_true');args=p.parse_args()
    if args.check:
        _,_,plan=prepare();print(json.dumps({'eligible':plan['query_id'],'request_hash':plan['original_request_hash'],
            'identical_original_request':True,'credentials_loaded':False,'model_calls':0},indent=2))
    else:run()
