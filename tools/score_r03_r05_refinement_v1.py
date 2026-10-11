"""Thin dispatcher reusing the audited module-stack producer and verifier.

Only the namespace, signature validator and frozen preparation entry change.
The per-request execution, physical-attempt audit and failure classifier are
the existing bytecode, with explicit module-local dependencies rebound.
"""
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
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import score_module_stack_v1 as prior
from jev_mem import r03_r05_request_envelope_v1 as core
from jev_mem.gemini38_provider_v1 import Gemini38API
from tools.score_evidence_exchange_r01 import now,read,sha,publish,exclusive_intent,status_write

STAGE='r03_r05_refinement_jev_v1'
SCHEMA='r03-r05-refinement-dispatch-v1'
RESULT_SCHEMA='r03-r05-refinement-score-result-v1'
LOCK=threading.Lock()


def _bind(fn, **updates):
    return FunctionType(fn.__code__,{**fn.__globals__,**updates},fn.__name__,
        fn.__defaults__,fn.__closure__)


context=_bind(prior.context,STAGE=STAGE)
classify_failure=_bind(prior.classify_failure,STAGE=STAGE)
one=_bind(prior.one,core=core,STAGE=STAGE,RESULT_SCHEMA=RESULT_SCHEMA,
    context=context,classify_failure=classify_failure)
verify_result=_bind(prior.verify_result,core=core,STAGE=STAGE,RESULT_SCHEMA=RESULT_SCHEMA,
    context=context,classify_failure=classify_failure)


class GateAPI(Gemini38API):
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        if stage!=STAGE:raise ValueError('Only frozen refinement scoring allowed')
        core.validate_request({'provider':provider,'model':model,'operation':operation,'body':body})
        return super().request(provider,model,body,operation=operation,stage=stage,cache=cache)


def run_phase(run_dir,phase,*,recovery=False):
    from tools.prepare_r03_r05_refinement_v1 import validate_phase,phase_results,Reader
    run=Path(run_dir).resolve()
    manifest,refs=validate_phase(run,phase)
    if recovery:
        first=phase_results(Reader(),run,phase,allow_recovery=False)
        refs=[ref for ref in refs if first[read(ref['path'])['request_id']]['status']=='failed']
        if not refs:raise ValueError('No technical failures to recover')
    producer=run/'phases'/phase/('recovery_v1' if recovery else 'firstpass')
    producer.mkdir(parents=True,exist_ok=True)
    if any((producer/'responses').glob('*.json')):raise FileExistsError('Immutable producer')
    retries=2
    exclusive_intent(producer/'execution_intent.json',{'schema':SCHEMA,'pid':os.getpid(),
        'created_utc':now(),'phase':phase,'recovery':recovery,'transport_retries':retries,
        'manifest_sha256':sha(run/'phases'/phase/'manifest.json'),
        'dispatch_code_sha256':sha(__file__),'requests':refs})
    api=GateAPI(producer,retries=retries)
    status={'schema':SCHEMA,'phase':phase,'pid':os.getpid(),'started_utc':now(),
        'status':'running','expected_requests':len(refs),'terminal_requests':0,
        'counts':{},'conversations':{},'recovery':recovery}
    sp=producer/'status.json';status_write(sp,status)
    rows,errors=[],[]
    byconv={ci:[] for ci in range(10)}
    for ref in refs:byconv[read(ref['path'])['conversation']].append(ref)
    def conv(ci):
        own=byconv[ci]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(one,producer,ref,api,manifest['protocol_sha256']):ref for ref in own}
            for future in as_completed(futures):
                with LOCK:
                    try:
                        row=future.result();rows.append(row)
                        status['conversations'][str(ci)]={'terminal':sum(r['conversation']==ci for r in rows),
                            'expected':len(own),'last_request_id':row['request_id'],'last_activity_utc':now()}
                    except Exception as exc:
                        errors.append({'request':futures[future],'error_type':type(exc).__name__})
                    status.update(terminal_requests=len(rows),updated_utc=now(),
                        counts=dict(Counter(r['status'] for r in rows)),local_errors=deepcopy(errors))
                    status_write(sp,status)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(conv,ci) for ci in range(10)]):future.result()
    status.update(status='needs_review' if errors or len(rows)!=len(refs)
        or any(r['status']=='unknown' for r in rows) else 'terminal',finished_utc=now(),local_errors=errors)
    publish(producer/'outcome.json',status);status_write(sp,status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--phase',required=True)
    p.add_argument('--recover',action='store_true')
    a=p.parse_args()
    print(json.dumps(run_phase(a.run_dir,a.phase,recovery=a.recover),ensure_ascii=False,indent=2))
