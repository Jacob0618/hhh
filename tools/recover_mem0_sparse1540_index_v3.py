"""Versioned continuation of pre-mutation float32 check failure; index policy unchanged."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mem0_sparse1540 as base
from tools.run_pilot400_experiments_v2 import read,publish_json
from jev_mem.providers import ModelAPI,Embeddings
from jev_mem.pilot400 import fingerprint
OUT=ROOT/'runs/mem0_sparse1540_index_recovery_v2'
POLICY='mem0-native-sparse1540-orphan-recovery-v2'
COVERAGE=ROOT/'research/mem0_sparse1540_twenty_bank_entity_coverage_2026-09-28.json'
POLICY_PATH=ROOT/'research/mem0_sparse1540_index_recovery_policy_v2_2026-09-28.json'
TARGETS={'session_rank':'bf8098df-18fc-4056-8232-2918dbc17257',
         'regroup_rank':'566fba15-8ba2-44e2-a364-4134c0ed732d'}
def binding(p):return {'path':str(Path(p).resolve()),'sha256':base.sha(p)}
def folder(arm):return OUT/'recovered'/arm/'conv_3'
def normalize(s):return ' '.join(s.strip().lower().split())
def points(client,collection):
    out={};offset=None
    while True:
        rows,offset=client.scroll(collection,limit=256,offset=offset,with_payload=True,with_vectors=True)
        for r in rows:out[str(r.id)]=r.model_dump(mode='json')
        if offset is None:return out
def linked_ids(entities):
    return {mid for r in entities.values() for mid in r['payload'].get('linked_memory_ids',[])}
def intended_entities(entities,fid):
    if fid in linked_ids(entities):raise ValueError('Target is not a fully orphaned fact')
    matches=[eid for eid,p in entities.items() if normalize(p['payload']['data'])=='video games']
    if len(matches)!=1:raise ValueError('No unique pre-existing exact entity destination')
    eid=matches[0];after=deepcopy(entities)
    after[eid]['payload']['linked_memory_ids']=sorted(set(after[eid]['payload'].get('linked_memory_ids',[]))|{fid})
    return eid,after
def inventory(path):
    return {str(p.relative_to(path)):base.sha(p) for p in Path(path).rglob('*') if p.is_file() and p.name!='.lock'}
def facts_and_journals(arm):
    if arm=='session_rank':
        bank=read(base.conv_path(3)/'session_bank.json')
        return bank['facts'],None
    inputs=read(base.conv_path(3)/'group_inputs.json')['envelopes']
    files={}
    for p in (base.conv_path(3)/'native/session_audit').rglob('*.json'):
        r=read(p)
        if r.get('status')=='success':
            if r['session_id'] in files:raise ValueError('Duplicate successful ADD journal')
            files[r['session_id']]=(p,r)
    if set(files)!={r['session_id'] for r in inputs}:raise ValueError('Missing/extra ADD journal')
    facts=[];journals=[]
    for envelope in inputs:
        p,r=files[envelope['session_id']]
        if any(r[k]!=v for k,v in envelope.items()):raise ValueError('Original message/date scope changed')
        facts.extend(deepcopy(r['fact_records']));journals.append(binding(p))
    return facts,journals
def native_gate(facts,entities,history):
    from mem0.memory.main import extract_entities_batch
    from mem0.utils import spacy_models
    if spacy_models.get_nlp_full() is None:raise ValueError('Native NLP unavailable')
    ids={f['id'] for f in facts};linked=linked_ids(entities)
    extracted=extract_entities_batch([f['text'] for f in facts])
    if len(extracted)!=len(facts):raise ValueError('Native NLP result cardinality mismatch')
    expected={f['id'] for f,e in zip(facts,extracted) if e}
    if linked-ids or expected-linked:raise ValueError('Native orphan/entity scope gate failed')
    db=sqlite3.connect(Path(history).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        for f in facts:
            rows=db.execute('SELECT new_memory,event,is_deleted FROM history WHERE memory_id=?',(f['id'],)).fetchall()
            if rows!=[(f['text'],'ADD',0)]:raise ValueError('Original ADD history mismatch')
    finally:db.close()
    return {'history_adds_verified':len(facts),'entities':len(entities),'linked_memory_ids':sorted(linked),
        'facts_with_local_entities':len(expected),'entity_facts_without_any_link':[],
        'scope':'Original orphan gate only; partial-link gaps remain unrepaired.'}
def validate_coverage():
    if base.sha(COVERAGE)!='f7845228541049d940ec2ac90e0dbd23eeb4af09fa787a2c65869680d46eee06':
        raise ValueError('Reviewed twenty-bank coverage bytes changed')
    if base.sha(POLICY_PATH)!='0321499a5abfc1573f6f894fd40ebcbebc0b5027477eb17f1da241c1ecbdb2b0':
        raise ValueError('Reviewed symmetric recovery policy changed')
    base.verify_manifest();r=read(COVERAGE);policy=read(POLICY_PATH)
    if not r['all_twenty_scanned'] or not r['original_files_unchanged'] or r['total_orphans']!=2:
        raise ValueError('Uniform source audit is incomplete or changed')
    expected={(a,i) for a in base.ARMS for i in range(10)}
    if {(x['arm'],x['conversation']) for x in r['rows']}!=expected or len(r['rows'])!=20:
        raise ValueError('Not all20 banks audited')
    for x in r['rows']:
        if x['orphan_fact_ids']!=([TARGETS[x['arm']]] if x['conversation']==3 else []):raise ValueError('Unreviewed orphan')
        for f in x['source_files']:
            if base.sha(f['source'])!=f['sha256'] or base.sha(f['copy'])!=f['sha256']:raise ValueError('Coverage source changed')
    return r,policy
def prepare():
    from qdrant_client import QdrantClient
    coverage,policy=validate_coverage()
    old=read(OUT/'prepare_intent.json')
    if old['helper_sha256']!='64b61be7376525be60846c27f2d2229795217d7ce6032d43dc507a0d9ef5c985':raise ValueError('Unknown prior preparation')
    if base.sha(ROOT/'tools/recover_mem0_sparse1540_index_v2.py')!=old['helper_sha256']:raise ValueError('Original failed helper changed')
    if old['source_manifest']!=binding(base.OUT/'manifest.json') or old['coverage']!=binding(COVERAGE) or old['policy_binding']!=binding(POLICY_PATH) or old['targets']!=TARGETS:
        raise ValueError('Prior preparation context differs')
    if set(p.name for p in OUT.iterdir())!={'prepare_intent.json','recovered'}:raise ValueError('Unreviewed continuation state')
    a=folder('session_rank')
    if set(p.name for p in a.iterdir())!={'native','original_snapshot'}:raise ValueError('Unexpected prior target artifacts')
    if set(p.name for p in (OUT/'recovered').iterdir())!={'session_rank'} or (a/'repair_intent.json').exists():raise ValueError('Prior index mutation not eligible for this continuation')
    if inventory(a/'native/qdrant')!=inventory(a/'original_snapshot/qdrant'):raise ValueError('Existing copy differs from immutable original')
    base.save_once(OUT/'prepare_continuation_intent.json',{'policy':POLICY,'started_utc':base.now(),'source_manifest':binding(base.OUT/'manifest.json'),
        'coverage':binding(COVERAGE),'policy_binding':binding(POLICY_PATH),'targets':TARGETS,
        'helper_sha256':base.sha(__file__),'prior_prepare_intent':binding(OUT/'prepare_intent.json'),
        'continuation_reason':'Pre-mutation float32 loading representation check; original copy unchanged.',
        'vector_absolute_tolerance':1e-6,'model_calls':0})
    results=[]
    for arm,fid in TARGETS.items():
        source=(base.BASE/'conv_3') if arm=='session_rank' else (base.conv_path(3)/'native')
        dest=folder(arm);snap=dest/'original_snapshot';native=dest/'native'
        existed=dest.exists()
        if existed and arm!='session_rank':raise ValueError('Unexpected pre-existing target')
        snap.mkdir(parents=True,exist_ok=True);native.mkdir(exist_ok=True)
        source_hashes={}
        for name in ('qdrant','history.sqlite','session_audit'):
            s=source/name;d=snap/name
            if not existed:
                if s.is_dir():shutil.copytree(s,d)
                else:shutil.copy2(s,d)
            if s.is_dir():
                if inventory(s)!=inventory(d):raise ValueError('Original snapshot copy mismatch')
                source_hashes[name]=inventory(s)
            else:
                if base.sha(s)!=base.sha(d):raise ValueError('Original history snapshot mismatch')
                source_hashes[name]=base.sha(s)
        if not existed:
            shutil.copytree(snap/'qdrant',native/'qdrant');shutil.copy2(snap/'history.sqlite',native/'history.sqlite')
        elif base.sha(native/'history.sqlite')!=base.sha(snap/'history.sqlite'):raise ValueError('Existing history copy differs')
        if arm=='regroup_rank':shutil.copy2(base.conv_path(3)/'write_status.json',snap/'original_failed_write_status.json')
        client=QdrantClient(path=str(native/'qdrant'));facts,journals=facts_and_journals(arm)
        max_vector_difference=0.0;max_norm_difference=0.0
        try:
            oldfacts=points(client,'facts');oldentities=points(client,'facts_entities')
            if set(oldfacts)!={f['id'] for f in facts}:raise ValueError('Fact set differs from original generation')
            for f in facts:
                p=oldfacts[f['id']]
                if p['payload']['data']!=f['text']:raise ValueError('Persisted fact text differs')
                vector=p['vector'].get('') if isinstance(p['vector'],dict) else p['vector']
                if 'embedding' in f:
                    if (len(f['embedding'])!=768 or len(vector)!=768 or
                        any(not math.isfinite(x) for x in vector+f['embedding']) or
                        abs(math.sqrt(sum(x*x for x in vector))-1)>1e-5 or
                        abs(math.sqrt(sum(x*x for x in f['embedding']))-1)>1e-5 or
                        max(abs(x-y) for x,y in zip(f['embedding'],vector))>1e-6):
                        raise ValueError('Baseline vector identity differs beyond float32 load tolerance')
                    max_vector_difference=max(max_vector_difference,max(abs(x-y) for x,y in zip(f['embedding'],vector)))
                    max_norm_difference=max(max_norm_difference,abs(math.sqrt(sum(x*x for x in f['embedding']))-math.sqrt(sum(x*x for x in vector))))
                else:f['embedding']=vector
            base.vector_ok(facts)
            target=next(f for f in facts if f['id']==fid)
            if target['text']!='Nate unwinds by playing video games or watching movies at home.':raise ValueError('Unexpected target text')
            eid,expected=intended_entities(oldentities,fid)
            intent={'policy':POLICY,'arm':arm,'conversation':3,'fact_id':fid,'entity_id':eid,
                'fact_sha256':fingerprint(target),'before':oldentities[eid],'intended_after':expected[eid],
                'fact_store_sha256':fingerprint(oldfacts),'entity_store_before_sha256':fingerprint(oldentities),
                'entity_store_after_sha256':fingerprint(expected),'source_path':str(source),'original_snapshot':str(snap),
                'source_hashes':source_hashes,'model_calls':0,'policy_binding':binding(POLICY_PATH),'coverage':binding(COVERAGE)}
            base.save_once(dest/'repair_intent.json',intent)
            client.set_payload('facts_entities',payload={'linked_memory_ids':expected[eid]['payload']['linked_memory_ids']},points=[eid])
            newfacts=points(client,'facts');newentities=points(client,'facts_entities')
            if newfacts!=oldfacts or newentities!=expected:raise ValueError('Repair changed more than declared link union')
            indexes=native_gate(facts,newentities,native/'history.sqlite')
        finally:client.close()
        for name,h in source_hashes.items():
            now=inventory(source/name) if isinstance(h,dict) else base.sha(source/name)
            if now!=h:raise ValueError('Original source changed during copy recovery')
        if arm=='session_rank':bankpath=base.conv_path(3)/'session_bank.json'
        else:
            bank={'schema':base.POLICY+'-bank','arm':arm,'conversation':3,'facts':facts,
                'source_dataset_sha256':base.SOURCE_HASH,'group_inputs_sha256':base.sha(base.conv_path(3)/'group_inputs.json'),
                'manifest_sha256':base.sha(base.OUT/'manifest.json'),'complete':True,'reused_native_bank':False,
                'fact_support':'input_scope_unverified_not_gold','user_id':'mem0_sparse1540_c3',
                'journal_files':journals,'native_index_audit':indexes,'sdk_warning_count':0,
                'index_recovery':binding(dest/'repair_intent.json'),'original_write_status':binding(snap/'original_failed_write_status.json')}
            bankpath=dest/'regroup_bank.json';base.save_once(bankpath,bank)
        record={**binding(dest/'repair_intent.json'),'arm':arm,'conversation':3,'bank':binding(bankpath),
            'native_dir':str(native),'source_dir':str(source),'native_index_audit':indexes,'model_calls':0,
            'post_index_files':inventory(native/'qdrant'),'history_sha256':base.sha(native/'history.sqlite'),
            'helper_sha256':base.sha(__file__),
            'inherited_ledger_bytes':0,
            'vector_representation_check':{'max_absolute_difference':max_vector_difference,'max_norm_difference':max_norm_difference,
                'historical_export_absolute_tolerance':1e-6,'actual_repair_before_after_all_points_exact':True},
            'original_unchanged':True,'status':'recovered_on_copy','finished_utc':base.now()}
        base.save_once(dest/'repair_result.json',record);results.append(record)
    base.save_once(OUT/'recovery_manifest.json',{'policy':POLICY,'source_manifest':binding(base.OUT/'manifest.json'),
        'coverage':binding(COVERAGE),'recovery_policy':binding(POLICY_PATH),'repairs':results,'created_utc':base.now(),'model_calls':0})
    return {'banks_recovered':len(results),'model_calls':0}
def seed_requests(arm):
    from mem0.memory.main import extract_entities
    from tools.audit_mem0_sparse1540_retrieval import native_expected_requests,validate_cache
    dest=folder(arm);source=Path(read(dest/'repair_result.json')['source_dir'])
    target=dest/'native/api_cache';target.mkdir(exist_ok=True);seen=set();rows=[]
    for q in base.questions(3):
        for spec in native_expected_requests(q['question'],extract_entities):
            h=base.request_digest(spec['signature'])
            if h in seen:continue
            seen.add(h);sp=source/'api_cache'/(h+'.json')
            if not sp.exists():continue
            validate_cache(read(sp),spec['signature']);dp=target/sp.name
            if dp.exists():raise ValueError('Seed destination already exists')
            shutil.copy2(sp,dp);rows.append({'request_hash':h,'source':binding(sp),'target':binding(dp)})
    base.save_once(dest/'cache_seed_lineage.json',{'entries':rows,'original_source_dir':str(source)})
    return rows
def retrieve(arm):
    validate_coverage();dest=folder(arm);rr=read(dest/'repair_result.json');bankpath=Path(rr['bank']['path'])
    rm=read(OUT/'recovery_manifest.json');intent=read(OUT/'prepare_continuation_intent.json')
    if rr!=next(x for x in rm['repairs'] if x['arm']==arm):raise ValueError('Recovery commit changed')
    if intent['helper_sha256']!=base.sha(__file__) or rr['helper_sha256']!=base.sha(__file__):raise ValueError('Recovery code changed')
    if base.sha(rr['path'])!=rr['sha256']:raise ValueError('Repair intent changed')
    if inventory(dest/'native/qdrant')!=rr['post_index_files'] or base.sha(dest/'native/history.sqlite')!=rr['history_sha256']:
        raise ValueError('Repaired index/history changed before retrieval')
    if base.sha(bankpath)!=rr['bank']['sha256']:raise ValueError('Recovery bank changed')
    if (dest/'retrieve_status.json').exists():raise ValueError('Retrieval already attempted; preserve terminal/unknown state')
    seed_requests(arm);bank=read(bankpath);facts={f['id']:f for f in bank['facts']}
    user_ids={f['user_id'] for f in facts.values()}
    if len(user_ids)!=1:raise ValueError('Native fact bank has mixed user scopes')
    user_id=next(iter(user_ids))
    api=ModelAPI(dest/'native',retries=2,context={'architecture':POLICY,'conversation':3,'method':arm,'phase':'retrieve'})
    state={'stage':'retrieve','policy':POLICY,'pid':os.getpid(),'arm':arm,'conversation':3,'status':'running','done':0,
        'expected':199,'started_utc':base.now()};publish_json(dest/'retrieve_status.json',state);memory=None
    try:
        memory=base.native.open_memory(dest/'native',api)
        if memory.vector_store._get_bm25_encoder() is None or not memory.vector_store._has_bm25_slot:raise ValueError('Native BM25 unavailable')
        original=memory.vector_store.keyword_search
        def required(*args,**kwargs):
            result=original(*args,**kwargs)
            if result is None:raise ValueError('Native keyword search failed')
            return result
        memory.vector_store.keyword_search=required
        for q in base.questions(3):
            memory.embedding_model=Embeddings(api.scoped(query_id=q['id']))
            results=memory.search(q['question'],top_k=60,filters={'user_id':user_id},rerank=False)['results']
            pool=[{**facts[r['id']],'score':r.get('score'),'sources':['mem0_hybrid']} for r in results]
            row=base.make_retrieval(q,arm,pool,rr['bank']['sha256'],{'new_native_search':True,'top_k':60,'threshold':0.1,
                'rerank':False,'index_recovery':binding(dest/'repair_result.json')})
            base.save_once(dest/'retrieval'/(q['id']+'.json'),row)
            state.update(done=state['done']+1,last_query_id=q['id'],updated_utc=base.now());publish_json(dest/'retrieve_status.json',state)
        state['status']='complete'
    except Exception as exc:state.update(status='failed',error={'type':type(exc).__name__,'message':api._redact(str(exc))[:1600]})
    finally:
        if memory is not None:base.native.close_memory(memory)
        state['finished_utc']=base.now();publish_json(dest/'retrieve_status.json',state)
    return state
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','retrieve'));a=p.parse_args()
    if a.stage=='prepare':result=prepare()
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:result=[f.result() for f in as_completed([pool.submit(retrieve,arm) for arm in base.ARMS])]
    print(json.dumps(result,indent=2))
