"""Declared JEV phases, original durable executor and producer/raw verifier."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
from types import FunctionType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import score_module_stack_v1 as prior
from tools.score_evidence_exchange_r01 import read,sha,publish,exclusive_intent,status_write,now
from jev_mem.gemini25_ablation_provider_v1 import Gemini25API
from jev_mem import evidence_exchange_r01 as shared
from jev_mem import evidence_selection_r03 as r03
from jev_mem import source_support_stack_v1 as stack
from jev_mem import r05_refinement_v1 as r05

STAGE='gemini25_ablation_jev_v1'
SCHEMA='gemini25-ablation-dispatch-v1'
RESULT_SCHEMA='gemini25-ablation-score-result-v1'
LOCK=threading.Lock()


def validate_request(sig,*,allow_oversize=False):
    if not isinstance(sig,dict) or set(sig)!={'provider','model','operation','body'}:
        raise ValueError('Exact request signature required')
    if (sig['provider'],sig['model'],sig['operation'])!=('jev','jev-1.13.0','systemone'):
        raise ValueError('Frozen JEV model/operation required')
    body=sig['body'];keys=body.get('questions',{})
    if len(keys) >= 20 and all(k.startswith('f') for k in keys):
        from tools.prepare_gemini25_ablation_v1 import spec
        protocol=spec();prefix=protocol['jev']['question_instructions_template'].split('{record_json}')[0]
        if not 20<=len(keys)<=80 or list(keys)!=[f'f{i:02d}' for i in range(len(keys))]:
            raise ValueError('Full ordered original R03 pointwise request required')
        records=[]
        for i,row in enumerate(keys.values()):
            if set(row)!={'type','instructions'} or row['type']!='noul' or not row['instructions'].startswith(prefix):
                raise ValueError('Original R03 predicate differs')
            value=json.loads(row['instructions'][len(prefix):])
            if set(value)!={'text','observation_date'}:raise ValueError('No extra R03 information permitted')
            records.append({'id':f'f{i:02d}',**value})
        query=body['state']['query']
        expected=r03.build_request(query['text'],query['reference_date'],records,protocol)
        if sig!=expected:raise ValueError('R03 signature does not replay exactly')
        kind='r03'
    elif keys and all(k.startswith('m') for k in keys):
        kind=r05.validate_request(sig,allow_oversize=allow_oversize)
    else:kind=stack.validate_request(sig,allow_oversize=allow_oversize)
    if not allow_oversize and len(shared.serialized_signature(sig))>131072:
        raise ValueError('Oversize cannot dispatch')
    return kind


def parse_response(response,sig):
    validate_request(sig)
    return shared.parse_scores(response,sig)


def clone(f,**updates):
    out=FunctionType(f.__code__,{**f.__globals__,**updates},f.__name__,f.__defaults__,f.__closure__)
    out.__kwdefaults__=dict(f.__kwdefaults__ or {})
    return out


core=SimpleNamespace(validate_request=validate_request,parse_response=parse_response,serialized_signature=shared.serialized_signature)
context=clone(prior.context,STAGE=STAGE)
classify_failure=clone(prior.classify_failure,STAGE=STAGE)
one=clone(prior.one,core=core,STAGE=STAGE,RESULT_SCHEMA=RESULT_SCHEMA,context=context,classify_failure=classify_failure)
verify_result=clone(prior.verify_result,core=core,STAGE=STAGE,RESULT_SCHEMA=RESULT_SCHEMA,context=context,classify_failure=classify_failure)


class GateAPI(Gemini25API):
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        if stage!=STAGE:raise ValueError('Declared JEV phases only')
        validate_request({'provider':provider,'model':model,'operation':operation,'body':body})
        return super().request(provider,model,body,operation=operation,stage=stage,cache=cache)


def phase_results(reader,run,phase,*,allow_recovery=True):
    pm=reader.read(run/'phases'/phase/'manifest.json');refs=pm['requests']
    def terminal(producer,requests):
        outcome=reader.read(producer/'outcome.json');intent=reader.read(producer/'execution_intent.json',sealed=False)
        if (outcome['schema']!=SCHEMA or outcome['status']!='terminal' or outcome.get('local_errors')
            or outcome['expected_requests']!=len(requests) or outcome['terminal_requests']!=len(requests)
            or intent['requests']!=requests or intent['transport_retries']!=2
            or intent['dispatch_code_sha256']!=sha(__file__) or intent['manifest_sha256']!=sha(run/'phases'/phase/'manifest.json')):
            raise ValueError('Whole phase is not audited terminal')
        rows={}
        for ref in requests:
            req=reader.bound(ref);rid=req['request_id'];result_ref=reader.pointer(producer/'responses'/(rid+'.json'))
            row=verify_result(reader,result_ref,ref,producer,2)
            rows[rid]={**row,'result_ref':result_ref}
        if {Path(r['result_ref']['path']).resolve() for r in rows.values()}!={p.resolve() for p in (producer/'responses').glob('*.json')}:
            raise ValueError('Missing/extra terminal responses')
        eligible={reader.bound(ref)['request_hash'] for ref in requests if reader.bound(ref)['dispatch_eligible']}
        if {p.stem for p in (producer/'request_intents').glob('*.json')}!=eligible:
            raise ValueError('Unmanifested/missing API intent')
        if {p.stem for p in (producer/'api_cache').glob('*.json')}!={r['request_hash'] for r in rows.values() if r.get('raw')}:
            raise ValueError('Unmanifested/missing raw response')
        allowed={(reader.bound(ref)['request_hash'],shared.digest(context(reader.bound(ref))))
                 for ref in requests if reader.bound(ref)['dispatch_eligible']}
        for p in (producer/'request_intents').glob('*.json'):
            iv=reader.read(p,sealed=False)
            if (p.stem,shared.digest(iv.get('context'))) not in allowed or shared.request_hash(iv['signature'])!=p.stem:
                raise ValueError('Foreign API producer')
        lp=producer/'api_calls.jsonl'
        if lp.exists():
            from tools.check_experiment_health import shared_read
            data=shared_read(lp);reader.raw_pointer(lp)
            if data and not data.endswith(b'\n'):raise ValueError('Incomplete ledger')
            for line in data.splitlines():
                e=json.loads(line)
                if ((e.get('request_hash'),shared.digest(e.get('context'))) not in allowed
                    or (e.get('stage'),e.get('provider'),e.get('model'),e.get('operation'))!=(STAGE,'jev','jev-1.13.0','systemone')):
                    raise ValueError('Foreign ledger request/context')
                if e.get('status') in (401,402,403):
                    raise ValueError('Resource/auth failure blocks downstream; no all-fallback evaluation')
        return rows
    result=terminal(run/'phases'/phase/'firstpass',refs)
    retry=run/'phases'/phase/'recovery_v1'
    failures=[ref for ref in refs if result[reader.bound(ref)['request_id']]['status']=='failed']
    if allow_recovery and failures and not retry.exists():raise ValueError('Uniform recovery is still required')
    if allow_recovery and retry.exists():
        if not failures:raise ValueError('Unnecessary recovery namespace')
        for rid,row in terminal(retry,failures).items():result[rid]={**row,'firstpass_result_ref':result[rid]['result_ref']}
    return result


def run_phase(run,phase,*,recovery=False):
    from tools.prepare_gemini25_ablation_v1 import validate_phase,Reader
    run=Path(run).resolve();pm,refs=validate_phase(run,phase)
    if recovery:
        prior_rows=phase_results(Reader(),run,phase,allow_recovery=False)
        refs=[ref for ref in refs if prior_rows[read(ref['path'])['request_id']]['status']=='failed']
        if not refs:raise ValueError('No known failures')
    dest=run/'phases'/phase/('recovery_v1' if recovery else 'firstpass')
    dest.mkdir(parents=True,exist_ok=False)
    exclusive_intent(dest/'execution_intent.json',{'schema':SCHEMA,'created_utc':now(),'pid':os.getpid(),
        'phase':phase,'recovery':recovery,'transport_retries':2,'requests':refs,
        'manifest_sha256':sha(run/'phases'/phase/'manifest.json'),'dispatch_code_sha256':sha(__file__)})
    api=GateAPI(dest,retries=2) if refs else None
    state={'schema':SCHEMA,'phase':phase,'pid':os.getpid(),'status':'running','started_utc':now(),
        'expected_requests':len(refs),'terminal_requests':0,'counts':{},'conversations':{},'local_errors':[],'recovery':recovery}
    rows=[];errors=[];byconv={ci:[] for ci in range(10)}
    for ref in refs:byconv[read(ref['path'])['conversation']].append(ref)
    status_write(dest/'status.json',state)
    def conversation(ci):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(one,dest,ref,api,pm['protocol_sha256']):ref for ref in byconv[ci]}
            for f in as_completed(futures):
                with LOCK:
                    try:
                        r=f.result();rows.append(r)
                        state['conversations'][str(ci)]={'terminal':sum(x['conversation']==ci for x in rows),
                            'expected':len(byconv[ci]),'last_request_id':r['request_id'],'last_activity_utc':now()}
                    except Exception as e:errors.append({'request':futures[f],'error_type':type(e).__name__})
                    state.update(updated_utc=now(),terminal_requests=len(rows),counts=dict(Counter(r['status'] for r in rows)),local_errors=deepcopy(errors))
                    status_write(dest/'status.json',state)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(conversation,ci) for ci in range(10)]):f.result()
    state.update(status='needs_review' if errors or len(rows)!=len(refs) or any(r['status']=='unknown' for r in rows) else 'terminal',finished_utc=now())
    publish(dest/'outcome.json',state);status_write(dest/'status.json',state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['r03','support','multi_compare','closure','rank'],required=True);p.add_argument('--recover',action='store_true')
    a=p.parse_args();print(json.dumps(run_phase(a.run_dir,a.phase,recovery=a.recover),ensure_ascii=False))
