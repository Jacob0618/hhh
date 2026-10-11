"""Thin fixed-protocol adapters around the original SDK20 retrieval and QA.

No stage runs on import. Original sources, banks and cached outputs are read-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_session_experiments as native
from tools import sdk20_module_sources as sources
from tools.run_pilot400_experiments_v2 import read, publish_json
from jev_mem.data import load_data, queries, session_envelopes, SOURCE_HASH
from jev_mem.evaluation import answer_and_judge
from jev_mem.providers import Embeddings
from jev_mem.selection import tag_fact
from jev_mem.pilot400 import query_type_scores
from jev_mem.sdk20_cached_api import CachedLegacyAPI, sha
from jev_mem.sdk20_type_adapter import capture_native_pool, typed_select

SCHEMA = 'sdk20-modules-v1'
ARMS = ('baseline_sdk20', 'fact_types_sdk20', 'preseg_sdk20')
CODE = ('tools/run_sdk20_modules.py', 'tools/sdk20_module_sources.py',
        'tools/run_session_experiments.py', 'tools/run_mem0_sparse1540.py',
        'tools/run_pilot400_experiments_v2.py', 'jev_mem/sdk20_cached_api.py',
        'jev_mem/sdk20_type_adapter.py', 'jev_mem/providers.py', 'jev_mem/evaluation.py',
        'jev_mem/selection.py', 'jev_mem/routing.py', 'jev_mem/pilot400.py',
        'jev_mem/mem0_adapter.py', 'jev_mem/data.py')

def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sealed(value): return {**value, 'content_sha256': digest(value)}
def verify(path):
    row = read(path)
    if row.get('content_sha256') != digest({k:v for k,v in row.items() if k != 'content_sha256'}):
        raise ValueError('Artifact content binding differs: ' + str(path))
    return row
def save_new(path, value):
    path = Path(path)
    if path.exists(): raise FileExistsError('Refusing to overwrite artifact: ' + str(path))
    publish_json(path, sealed(value))

def cache_roots(ci):
    return [ROOT/'runs/technical_recovery_v1/transport/api_cache',
            ROOT/f'runs/session_top20_v1/conv_{ci}/api_cache',
            ROOT/f'runs/mem0_sparse1540_v1/conv_{ci}/native/api_cache',
            ROOT/'runs/pilot400_modules_v2/api_cache', ROOT/'runs/pilot400_modules_v1/api_cache']

def manifest(run):
    row = read(Path(run)/'run_manifest.json')
    body = {k:v for k,v in row.items() if k != 'protocol_sha256'}
    if row['protocol_sha256'] != digest(body): raise ValueError('Protocol digest differs')
    if row['schema'] != SCHEMA or row['arms'] != list(ARMS): raise ValueError('Protocol identity differs')
    for name, expected in row['code_sha256'].items():
        if sha(ROOT/name) != expected: raise ValueError('Frozen code changed: ' + name)
    if sha(ROOT/'data/locomo10.json') != SOURCE_HASH: raise ValueError('Dataset differs')
    if importlib.metadata.version('mem0ai') != '2.2.1': raise ValueError('Mem0 version differs')
    for name, expected in row['sdk_sha256'].items():
        if sha(Path(name)) != expected: raise ValueError('Native SDK code changed')
    return row

def prepare(run):
    run = Path(run).resolve()
    if run.exists(): raise FileExistsError('Prepare requires a new run directory')
    readiness = sources.readiness()
    run.mkdir(parents=True)
    save_new(run/'baseline_source_readiness.json', readiness)
    source_manifests = {}
    for ci in range(10):
        sources.prepare_conv(run, ci)
        source_manifests[str(ci)] = sha(run/f'conv_{ci}/source_manifest.json')
    sdkroot = Path(importlib.metadata.distribution('mem0ai').locate_file('mem0'))
    sdk = {str(p):sha(p) for p in sdkroot.rglob('*.py')}
    config = {'schema': SCHEMA, 'created_utc':now(), 'arms':list(ARMS),
              'dataset_sha256':SOURCE_HASH, 'questions':1540,
              'query_ids':{str(ci):[q['id'] for q in queries(item,ci)] for ci,item in enumerate(load_data())},
              'code_sha256':{p:sha(ROOT/p) for p in CODE}, 'sdk_sha256':sdk,
              'source_manifests':source_manifests,
              'reader_protocol':'brief_answer_local_evidence_ids_v2', 'response_schema':False,
              'models':{'reader':'gemini-2.5-flash','judge':'gemini-3.5-flash-lite',
                        'embedding':'gemini-embedding-001','dimensions':768,'jev':'jev-1.13.0'},
              'retrieval':{'top_k':20,'rerank':False,'threshold':0.1,'native_internal_limit':80},
              'type_fusion':{'native_weight':2,'type_weight':1,'rrf_k':60},
              'workers':{'conversations':8,'qa_per_conversation':4,'jev_http_per_second':15},
              'known_baseline':{'completed':1525,'failed':15,'correct':1001},
              'cache_policy':'fixed precedence: reviewed recovery, original session, sparse native, pilot v2, pilot v1; exact request only; never correctness-select or resend unresolved logical request',
              'native_state':'original_unrepaired', 'development_conversations':[0,1],
              'plan':{'path':str(ROOT/'research/sdk20_modules_execution_plan_2026-09-28.md'),
                      'sha256':sha(ROOT/'research/sdk20_modules_execution_plan_2026-09-28.md')}}
    publish_json(run/'run_manifest.json', {**config,'protocol_sha256':digest(config)})
    return {'prepared':str(run),'protocol_sha256':digest(config),'model_calls':0}

def get_bank(run, ci, kind, m):
    p = Path(run)/f'conv_{ci}/source_manifest.json'
    if sha(p) != m['source_manifests'][str(ci)]: raise ValueError('Source manifest changed')
    meta = read(p)['banks'][kind]
    if sha(meta['bank']['path']) != meta['bank']['sha256']: raise ValueError('Bank changed')
    return read(meta['bank']['path']), meta['bank']['sha256']

def identity(ci, q, m, method=None):
    return {'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'], 'conversation':ci,
            'query_id':q['id'],'question':q['question'],'reference':q['reference'],
            'category':q['category'], **({'method':method} if method else {})}

def existing(path, m):
    if not Path(path).exists(): return None
    row = verify(path)
    if row['protocol_sha256'] != m['protocol_sha256']: raise ValueError('Artifact protocol mismatch')
    return row

def status(folder, stage, completed, total, failed, **extra):
    publish_json(Path(folder)/(stage+'_status.json'), {'schema':SCHEMA,'stage':stage,'pid':os.getpid(),
                 'updated_utc':now(),'completed':completed,'total':total,'failed':failed,**extra})

def baseline(run, ci, m):
    folder=Path(run)/f'conv_{ci}';bank, bank_hash=get_bank(run,ci,'session',m)
    qs=list(queries(load_data()[ci],ci));done=failed=0
    for q in qs:
        p=folder/'qa'/ARMS[0]/(q['id']+'.json')
        row=existing(p,m)
        if row is None:
            old=sources.baseline_row(ci,q)
            row={**identity(ci,q,m,ARMS[0]),**old,'api_events':[], 'inherited_result':True}
            save_new(p,row)
        sp=folder/'selection'/ARMS[0]/(q['id']+'.json')
        if existing(sp,m) is None:
            save_new(sp,{**identity(ci,q,m,ARMS[0]),'status':'complete','selected_ids':row['selected_ids'],
                         'bank_sha256':bank_hash,'trace':{'method':'original_sdk20'},'source':row['retrieval_source']})
        done+=1;failed+=row['status']!='complete'
    status(folder,'baseline',done,len(qs),failed,terminal=True)
    return {'conversation':ci,'stage':'baseline','completed':done,'failed':failed}

class Warnings(logging.Handler):
    def __init__(self): super().__init__(logging.WARNING);self.items=[]
    def emit(self, record):
        self.items.append({'level':record.levelname,'function':record.funcName,'line':record.lineno})

def retrieve(run, ci, m):
    from mem0.utils import spacy_models
    folder=Path(run)/f'conv_{ci}';qs=list(queries(load_data()[ci],ci));done=0
    api=CachedLegacyAPI(folder/'api',cache_roots=cache_roots(ci),context={'conversation':ci,'phase':'retrieval'})
    for kind in ('session','preseg'):
        bank,bank_hash=get_bank(run,ci,kind,m);byid={f['id']:f for f in bank['facts']}
        if sources._native_inventory(bank['native_dir']) != bank['source_inventory']:
            raise ValueError('Copied native store changed before retrieval; review required')
        mem=native.open_memory(Path(bank['native_dir']),api)
        try:
            if spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None or mem.vector_store._get_bm25_encoder() is None:
                raise RuntimeError('Native NLP/BM25 unavailable')
            for q in qs:
                path=folder/'retrieval'/kind/(q['id']+'.json')
                old=existing(path,m)
                if old is not None:
                    if old['status']!='complete': raise RuntimeError('Retained retrieval failure requires review')
                    done+=1;continue
                scoped=api.scoped(query_id=q['id'],method=kind)
                mem.embedding_model=Embeddings(scoped)
                audit=Warnings();logging.getLogger('mem0').addHandler(audit)
                try:
                    with capture_native_pool() as captured:
                        result=mem.search(q['question'],top_k=20,filters={'user_id':bank['user_id']},rerank=False)
                    ids=[f['id'] for f in result['results']]
                    if ids!=captured['base20_ids'] or len(ids)!=20 or len(set(ids))!=20 or not set(captured['eligible_ids'])<=set(byid):
                        raise ValueError('Native retrieval budget or identity differs')
                    replay=None
                    if kind=='session':
                        source=sources.baseline_row(ci,q)['retrieval_source']
                        previous=read(source['path'])['sdk20']
                        if ids!=[f['id'] for f in previous]: raise ValueError('Original SDK20 ordered IDs did not replay')
                        for returned,prior in zip(result['results'],previous):
                            fact=byid[returned['id']]
                            if fact['text']!=prior['text'] or fact['observation_date']!=prior['observation_date'] or abs(returned['score']-prior['score'])>1e-7:
                                raise ValueError('Original SDK20 score/text/date did not replay')
                        replay={'matched':True,'source':source}
                    if audit.items: raise RuntimeError('Native search warning requires review')
                    save_new(path,{**identity(ci,q,m),'status':'complete','kind':kind,'bank_sha256':bank_hash,
                                   'capture':captured,'selected_ids':ids,'baseline_replay':replay,'api_events':scoped.events})
                except Exception as exc:
                    save_new(path,{**identity(ci,q,m),'status':'failed','kind':kind,'bank_sha256':bank_hash,
                                   'selected_ids':[],'error':{'type':type(exc).__name__,'message':scoped._redact(exc)},
                                   'sdk_warnings':audit.items,'api_events':scoped.events})
                    raise
                finally: logging.getLogger('mem0').removeHandler(audit)
                done+=1;status(folder,'retrieve',done,2*len(qs),0,terminal=False,last_query=q['id'],kind=kind)
        finally: native.close_memory(mem)
    status(folder,'retrieve',done,2*len(qs),0,terminal=True)
    return {'conversation':ci,'stage':'retrieve','completed':done,'failed':0}

def tag(run, ci, m):
    folder=Path(run)/f'conv_{ci}';bank,bank_hash=get_bank(run,ci,'session',m)
    sessions={s['session_id']:s for s in session_envelopes(load_data()[ci])}
    api=CachedLegacyAPI(folder/'api',cache_roots=cache_roots(ci),context={'conversation':ci,'phase':'write_typing','method':ARMS[1]})
    def one(fact):
        path=folder/'tags'/(fact['id']+'.json');old=existing(path,m)
        if old is not None:return old
        context='\n'.join(x['content'] for x in sessions[fact['input_scope']['session_id']]['messages'])
        scoped=api.scoped(fact_id=fact['id']);result=None;error=None
        try: result=tag_fact({**fact,'source_context':context},scoped)
        except Exception as exc: error={'type':type(exc).__name__,'message':scoped._redact(exc)}
        row={'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'fact_id':fact['id'],
             'bank_sha256':bank_hash,'source_context_sha256':digest(context),'fact_text_sha256':digest(fact['text']),
             'source_scope':'whole_original_session_unverified_input_scope','result':result,
             'status':'failed' if error else 'complete','error':error,'api_events':scoped.events}
        save_new(path,row);return row
    done=failed=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(one,f) for f in bank['facts']]):
            row=fut.result();done+=1;failed+=row['status']!='complete'
            status(folder,'tag',done,len(bank['facts']),failed,terminal=False,last_fact=row['fact_id'])
    status(folder,'tag',done,len(bank['facts']),failed,terminal=True)
    return {'conversation':ci,'stage':'tag','completed':done,'failed':failed}

def select_queries(run, ci, m):
    folder=Path(run)/f'conv_{ci}';bank,bhash=get_bank(run,ci,'session',m);_,chash=get_bank(run,ci,'preseg',m)
    tags={}
    for fact in bank['facts']:
        row=existing(folder/'tags'/(fact['id']+'.json'),m)
        if not row or row['status']!='complete' or row['bank_sha256']!=bhash: raise RuntimeError('All fact tags must complete before selection')
        tags[fact['id']]=row['result']
    api=CachedLegacyAPI(folder/'api',cache_roots=cache_roots(ci),context={'conversation':ci,'phase':'query_types','method':ARMS[1]})
    qs=list(queries(load_data()[ci],ci));done=failed=0
    def one(q):
        counts=0
        for kind,method,bank_hash in (('session',ARMS[1],bhash),('preseg',ARMS[2],chash)):
            path=folder/'selection'/method/(q['id']+'.json');old=existing(path,m)
            if old is not None:counts+=old['status']!='complete';continue
            retrieval=existing(folder/'retrieval'/kind/(q['id']+'.json'),m)
            if not retrieval or retrieval['status']!='complete' or retrieval['bank_sha256']!=bank_hash:raise RuntimeError('Complete bound retrieval required')
            scoped=api.scoped(query_id=q['id']);error=None;chosen=[];trace=None
            try:
                if kind=='session':
                    qp=folder/'query_types'/(q['id']+'.json');qt=existing(qp,m)
                    if qt is None:
                        scores,meta=query_type_scores(q['question'],scoped)
                        qt={**identity(ci,q,m,method),'status':'complete','scores':scores,'metadata':meta,'api_events':list(scoped.events)}
                        save_new(qp,qt)
                    selection=typed_select(retrieval['capture'],tags,qt['scores'])
                    chosen=selection['selected_ids'];trace=selection['trace']
                else:chosen=retrieval['selected_ids'];trace={'method':'native_sdk20_after_presegmentation'}
                if len(chosen)!=20 or len(set(chosen))!=20:raise ValueError('Expected 20 unique facts')
            except Exception as exc:error={'type':type(exc).__name__,'message':scoped._redact(exc)}
            save_new(path,{**identity(ci,q,m,method),'status':'failed' if error else 'complete',
                           'selected_ids':chosen if not error else [],'trace':trace,'bank_sha256':bank_hash,
                           'retrieval_sha256':sha(folder/'retrieval'/kind/(q['id']+'.json')),
                           'error':error,'api_events':scoped.events})
            counts+=bool(error)
        return counts
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(one,q) for q in qs]):
            failed+=fut.result();done+=2;status(folder,'select',done,2*len(qs),failed,terminal=False)
    status(folder,'select',done,2*len(qs),failed,terminal=True)
    return {'conversation':ci,'stage':'select','completed':done,'failed':failed}

def qa(run,ci,m):
    folder=Path(run)/f'conv_{ci}';item=load_data()[ci];qs=list(queries(item,ci))
    date=list(session_envelopes(item))[-1]['observation_date'];banks={}
    for kind,method in (('session',ARMS[1]),('preseg',ARMS[2])):
        bank,bhash=get_bank(run,ci,kind,m);banks[method]=({f['id']:f for f in bank['facts']},bhash)
    api=CachedLegacyAPI(folder/'api',cache_roots=cache_roots(ci),context={'conversation':ci,'phase':'qa'})
    def one(q):
        fails=0
        for method in ARMS[1:]:
            path=folder/'qa'/method/(q['id']+'.json');old=existing(path,m)
            if old is not None:fails+=old['status']!='complete';continue
            selection_path=folder/'selection'/method/(q['id']+'.json');selection=existing(selection_path,m)
            byid,bhash=banks[method]
            if not selection or selection['bank_sha256']!=bhash:raise RuntimeError('Missing selection or wrong bank')
            scoped=api.scoped(query_id=q['id'],method=method);result=None;error=None
            ids=selection['selected_ids']
            try:
                if selection['status']!='complete':raise RuntimeError('Selection failed; no Reader/Judge dispatch')
                result=answer_and_judge(scoped,q['question'],q['reference'],[byid[i] for i in ids],reference_date=date)
            except Exception as exc:error={'type':type(exc).__name__,'message':scoped._redact(exc)}
            row={**identity(ci,q,m,method),'status':'failed' if error else 'complete','result':result,
                 'selected_ids':ids,'reference_date':date,'bank_sha256':bhash,
                 'selection_sha256':sha(selection_path),'error':error,'api_events':scoped.events}
            save_new(path,row);fails+=bool(error)
        return fails
    done=failed=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(one,q) for q in qs]):
            failed+=fut.result();done+=2;status(folder,'qa',done,2*len(qs),failed,terminal=False)
    status(folder,'qa',done,2*len(qs),failed,terminal=True)
    return {'conversation':ci,'stage':'qa','completed':done,'failed':failed}

def worker(run,ci,stage):
    m=manifest(run)
    return {'baseline':baseline,'retrieve':retrieve,'tag':tag,'select':select_queries,'qa':qa}[stage](run,ci,m)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','check','baseline','retrieve','tag','select','qa'])
    parser.add_argument('--run-dir',default=str(ROOT/'runs/sdk20_modules_v1'))
    parser.add_argument('--conversations',default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();run=Path(args.run_dir).resolve()
    if args.stage=='prepare': print(json.dumps(prepare(run)));return 0
    m=manifest(run)
    if args.stage=='check':print(json.dumps({'protocol_sha256':m['protocol_sha256'],'valid':True,'model_calls':0}));return 0
    convs=[int(x) for x in args.conversations.split(',')]
    if not convs or len(set(convs))!=len(convs) or not set(convs)<=set(range(10)) or not 1<=args.workers<=8:raise ValueError('Invalid workers or conversations')
    prerequisite={'retrieve':'baseline','tag':'retrieve','select':'tag','qa':'select'}.get(args.stage)
    if prerequisite:
        previous=read(run/(prerequisite+'_status.json'))
        if not previous.get('terminal') or previous.get('completed')!=10 or previous.get('total')!=10:
            raise ValueError('The preceding full-cohort stage has not finished')
        if prerequisite!='baseline' and previous.get('failed'):
            raise ValueError('Preceding stage failures require review before continuing')
    lock=run/'active_stage.lock'
    with lock.open('x',encoding='utf-8') as handle:json.dump({'pid':os.getpid(),'stage':args.stage,'created_utc':now()},handle)
    failures=[];outcomes=[]
    try:
        status(run,args.stage,0,len(convs),0,terminal=False,conversations=convs)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(worker,str(run),ci,args.stage):ci for ci in convs}
            for fut in as_completed(pending):
                ci=pending[fut]
                try:row=fut.result();outcomes.append(row)
                except Exception as exc:
                    row={'conversation':ci,'stage':args.stage,'error_type':type(exc).__name__,'error':str(exc)}
                    failures.append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
                status(run,args.stage,len(outcomes)+len(failures),len(convs),len(failures)+sum(x['failed'] for x in outcomes),
                       terminal=False,outcomes=outcomes,failures=failures)
        status(run,args.stage,len(convs),len(convs),len(failures)+sum(x['failed'] for x in outcomes),
               terminal=True,outcomes=outcomes,failures=failures)
        return 1 if failures or any(x['failed'] for x in outcomes) and args.stage!='baseline' else 0
    finally:lock.unlink()

if __name__=='__main__':sys.exit(main())
