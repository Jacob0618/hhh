"""Frozen two-arm native Mem0 transfer; explicit prepare/write/retrieve/qa stages.

All original runs are read-only. Whole-session baseline banks and native search
pools are reused, never their answers. New sparse groups pass through the same
pinned Memory.add and native Memory.search. No independent writer or JEV reranker.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_session_experiments as native
from tools.run_pilot400_experiments_v2 import read,publish_json
from tools.pilot400_recovery_qa_v1 import _sealed,_verify_sealed
from jev_mem.data import load_data,queries,session_envelopes,SOURCE_HASH
from jev_mem.topic_grouping import raw_turns_from_locomo,verify_partition
from jev_mem.pilot400 import fingerprint
from jev_mem.selection import select
from jev_mem.providers import ModelAPI,Embeddings
from jev_mem.mem0_adapter import Mem0SessionAdapter

OUT=ROOT/'runs/mem0_sparse1540_v1'
BASE=ROOT/'runs/session_top20_v1'
GROUPS=ROOT/'runs/pilot400_modules_v2/conversations'
ARMS=('session_rank','regroup_rank')
POLICY='mem0-native-sparse1540-transfer-v1'
CURRENT_DATE='2026-09-27'
CODE_FILES=('tools/run_mem0_sparse1540.py','jev_mem/mem0_transfer_qa.py',
 'tools/run_session_experiments.py','jev_mem/mem0_adapter.py','jev_mem/providers.py',
 'jev_mem/data.py','jev_mem/evaluation.py','jev_mem/selection.py',
 'jev_mem/topic_grouping.py','jev_mem/pilot400.py','tools/pilot400_recovery_qa_v1.py',
 'tools/run_pilot400_experiments_v2.py',
 'upstream/mem0/mem0/memory/main.py','upstream/mem0/mem0/configs/prompts.py',
 'upstream/mem0/mem0/memory/storage.py','upstream/mem0/mem0/vector_stores/qdrant.py')

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_once(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Immutable artifact changed: '+str(path))
    else:publish_json(path,value)

def questions(ci=None):
    data=load_data()
    rows=[{**q,'conversation':i} for i,item in enumerate(data) for q in queries(item,i)]
    if len(rows)!=1540 or len({q['id'] for q in rows})!=1540:raise ValueError('Wrong1540 population')
    if Counter(q['category'] for q in rows)!=Counter({1:282,2:321,3:96,4:841}):
        raise ValueError('Wrong category population')
    return rows if ci is None else [q for q in rows if q['conversation']==ci]

def group_envelopes(item,grouping):
    _verify_sealed(grouping)
    if grouping['status']!='complete':raise ValueError('Only complete frozen grouping is reusable')
    raw=raw_turns_from_locomo(item['conversation'])
    verify_partition(raw,grouping['groups'])
    source=list(session_envelopes(item));by_id={}
    for session in source:
        for tid,msg in zip(session['source_turn_ids'],session['messages']):
            by_id[tid]=(session['session_id'],session['observation_date'],msg)
    expected=[t['source_turn_id'] for t in raw];assigned=[];envelopes=[]
    for group in grouping['groups']:
        ids=group['source_turn_ids'];members=[by_id[t] for t in ids]
        if len({x[0] for x in members})!=1 or len({x[1] for x in members})!=1:
            raise ValueError('This native single-date adapter rejects cross-session/date groups')
        if group['cross_session'] or group['cross_date'] or len(group['segment_ids'])!=1:
            raise ValueError('Frozen transfer is sparse splitting, not merged groups')
        assigned.extend(ids)
        envelopes.append({'session_id':group['group_id'],'observation_date':members[0][1],
            'messages':[x[2] for x in members],'source_turn_ids':ids})
    if assigned!=expected:raise ValueError('Native sequential ingestion needs exact source-order partition')
    return envelopes

def vector_ok(facts):
    if len({f['id'] for f in facts})!=len(facts):raise ValueError('Duplicate stored IDs')
    for f in facts:
        v=f['embedding'];norm=sum(x*x for x in v)
        if len(v)!=768 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) or abs(norm-1)>1e-5:
            raise ValueError('Invalid normalized768 embedding')

def query_signature(question):
    model='gemini-embedding-001'
    return {'provider':'gemini','model':model,'operation':'batchEmbedContents','body':{'requests':[
        {'model':'models/'+model,'content':{'parts':[{'text':question}]},
         'taskType':'RETRIEVAL_QUERY','outputDimensionality':768}]}}

def request_digest(signature):return hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def current_code():return {p:sha(ROOT/p) for p in CODE_FILES}

def dependency_versions():
    return {name:importlib.metadata.version(name) for name in
            ('mem0ai','qdrant-client','fastembed','spacy','en-core-web-sm','requests','numpy')}

def verify_manifest(ci=None):
    m=read(OUT/'manifest.json')
    if (m['policy']!=POLICY or m['source_dataset_sha256']!=SOURCE_HASH or m['code_sha256']!=current_code()
            or m['dependencies']!=dependency_versions()):
        raise ValueError('Frozen protocol/source changed; use a separate version')
    if m['question_ids']!=[q['id'] for q in questions()]:raise ValueError('Frozen cohort changed')
    for index in (range(10) if ci is None else [ci]):
        folder=conv_path(index);meta=m['conversations'][index]
        for name,key in (('group_inputs.json','group_inputs_sha256'),('session_bank.json','baseline_bank_sha256'),
                         ('query_embedding_seed_lineage.json','query_embedding_seed_lineage_sha256')):
            if sha(folder/name)!=meta[key]:raise ValueError('Prepared input/bank/seed lineage changed: '+name)
        for seed in read(folder/'query_embedding_seed_lineage.json')['entries']:
            target=(OUT/seed['target_relative']).resolve()
            if not target.is_relative_to((folder/'native/api_cache').resolve()) or sha(target)!=seed['source_sha256']:
                raise ValueError('Signed query embedding seed changed')
    return m

def fact_view(f):return {'id':f['id'],'text':f['text'],'observation_date':f['observation_date']}

def make_retrieval(q,arm,pool,bank_hash,source):
    selection=select(q['question'],pool,method='rank',top_k=20)
    if len(selection.selected)!=20:raise ValueError('Native search did not supply20 unique facts')
    return _sealed({'schema':POLICY+'-retrieval','status':'complete','arm':arm,
        'conversation':q['conversation'],'query_id':q['id'],'question_sha256':fingerprint(q['question']),
        'bank_sha256':bank_hash,'source':source,'pool_ids':[f['id'] for f in pool],
        'pool_scores':[f.get('score') for f in pool],
        'selected':[fact_view(f) for f in selection.selected],'selection_trace':selection.trace,
        'selection':'native_mem0_hybrid_top60_then_rank20_unique'})

def validate_retrieval(row,q,arm,bank,bank_hash):
    _verify_sealed(row)
    ids=row.get('pool_ids');scores=row.get('pool_scores');by_id={f['id']:f for f in bank['facts']}
    if (not isinstance(ids,list) or not 20<=len(ids)<=60 or len(set(ids))!=len(ids)
            or not isinstance(scores,list) or len(scores)!=len(ids) or not set(ids)<=set(by_id)):
        raise ValueError('Native candidate identities/budget invalid')
    if any(x is not None and (type(x) not in (int,float) or not math.isfinite(x)) for x in scores):
        raise ValueError('Invalid recorded native score')
    pool=[{**by_id[tid],'score':score,'sources':['mem0_hybrid']} for tid,score in zip(ids,scores)]
    expected=make_retrieval(q,arm,pool,bank_hash,row['source'])
    if row!=expected:raise ValueError('Native rank20 retrieval does not replay exactly')
    return row

def load_bank(ci,arm,manifest):
    folder=conv_path(ci);name='session_bank.json' if arm==ARMS[0] else 'regroup_bank.json'
    path=folder/name;bank=read(path);digest=sha(path)
    if (bank.get('arm')!=arm or bank.get('conversation')!=ci or not bank.get('complete')
            or bank.get('source_dataset_sha256')!=SOURCE_HASH):raise ValueError('Native bank identity changed')
    if arm==ARMS[0]:expected=manifest['conversations'][ci]['baseline_bank_sha256']
    else:
        state=read(folder/'write_status.json')
        if state['status']!='complete' or bank['manifest_sha256']!=sha(OUT/'manifest.json'):
            raise ValueError('Sparse native writing not complete or differently bound')
        expected=state['bank_sha256']
    if digest!=expected:raise ValueError('Native bank differs from committed hash')
    vector_ok(bank['facts']);return bank,digest

def prepare():
    if (OUT/'manifest.json').exists():return verify_manifest()
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_data();qs=questions();source_files={};conversation_meta=[]
    for ci,item in enumerate(data):
        target=OUT/('conv_'+str(ci));baseline=BASE/('conv_'+str(ci))
        gp=GROUPS/('conv_'+str(ci))/'banks/regroup/grouping.json'
        grouping=read(gp);envelopes=group_envelopes(item,grouping)
        inp={'schema':POLICY+'-group-inputs','conversation':ci,'raw_source_sha256':fingerprint(raw_turns_from_locomo(item['conversation'])),
            'grouping_path':str(gp),'grouping_sha256':sha(gp),'envelopes':envelopes,
            'source_turn_count':sum(len(e['messages']) for e in envelopes),'current_date':CURRENT_DATE,
            'rendering':'exact_session_envelopes_messages_subset_no_group_temporal_instruction'}
        save_once(target/'group_inputs.json',inp)
        summary=read(baseline/'ingestion_summary.json');facts=read(baseline/'facts.json')
        if not summary['complete'] or summary['facts']!=len(facts) or summary['sessions_total']!=len(list(session_envelopes(item))):
            raise ValueError('Baseline native bank incomplete')
        vector_ok(facts)
        native_hash=sha(baseline/'facts.json')
        bank={'schema':POLICY+'-bank','arm':ARMS[0],'conversation':ci,'facts':facts,
            'source_dataset_sha256':SOURCE_HASH,'upstream_fact_path':str(baseline/'facts.json'),
            'upstream_fact_sha256':native_hash,'summary_sha256':sha(baseline/'ingestion_summary.json'),
            'complete':True,'reused_native_bank':True,'fact_support':'input_scope_unverified_not_gold'}
        save_once(target/'session_bank.json',bank);bank_hash=sha(target/'session_bank.json')
        source_files[str(gp)]=sha(gp);source_files[str(baseline/'facts.json')]=native_hash
        source_files[str(baseline/'ingestion_summary.json')]=bank['summary_sha256']
        seeded=[]
        for q in [q for q in qs if q['conversation']==ci]:
            oldp=baseline/'retrieval'/(q['id']+'.json');old=read(oldp)
            native._verify_retrieval(old,q,summary,native_hash)
            if len(old['pool60'])>60:raise ValueError('Original native pool exceeds60')
            by_id={f['id']:f for f in facts}
            for f in old['pool60']:
                if any(f.get(k)!=by_id[f['id']][k] for k in ('text','observation_date','embedding')):
                    raise ValueError('Baseline cached candidate differs from frozen bank')
            old_hash=sha(oldp);source_files[str(oldp)]=old_hash
            row=make_retrieval(q,ARMS[0],old['pool60'],bank_hash,
                {'path':str(oldp),'sha256':old_hash,'reused_native_search':True})
            save_once(target/'retrieval'/ARMS[0]/(q['id']+'.json'),row)
            sig=query_signature(q['question']);digest=request_digest(sig);cache=baseline/'api_cache'/(digest+'.json')
            if cache.exists():
                value=read(cache)
                if value['request']!=sig:raise ValueError('Original query embedding signature mismatch')
                vectors=value['response'].get('embeddings',[])
                if len(vectors)!=1 or len(vectors[0]['values'])!=768:raise ValueError('Bad original query vector')
                dest=target/'native/api_cache'/cache.name
                if dest.exists():
                    if sha(dest)!=sha(cache):raise ValueError('Query cache seed differs')
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cache,dest)
                seeded.append({'query_id':q['id'],'request_hash':digest,'source_path':str(cache),
                    'source_sha256':sha(cache),'target_relative':dest.relative_to(OUT).as_posix()})
        save_once(target/'query_embedding_seed_lineage.json',{'entries':seeded,'new_http_calls':0})
        conversation_meta.append({'conversation':ci,'sessions':summary['sessions_total'],'groups':len(envelopes),
            'turns':inp['source_turn_count'],'baseline_facts':len(facts),'questions':sum(q['conversation']==ci for q in qs),
            'group_inputs_sha256':sha(target/'group_inputs.json'),'baseline_bank_sha256':bank_hash,
            'query_embedding_seed_lineage_sha256':sha(target/'query_embedding_seed_lineage.json')})
    if (sum(c['sessions'] for c in conversation_meta),sum(c['groups'] for c in conversation_meta),
            sum(c['turns'] for c in conversation_meta),sum(c['baseline_facts'] for c in conversation_meta))!=(272,385,5882,2376):
        raise ValueError('Known native/grouping population changed')
    m={'schema':POLICY,'policy':POLICY,'created_utc':now(),'source_dataset_sha256':SOURCE_HASH,
        'question_ids':[q['id'] for q in qs],'questions_per_arm':1540,'arms':list(ARMS),
        'current_date':CURRENT_DATE,'conversations':conversation_meta,'source_files':source_files,
        'code_sha256':current_code(),'dependencies':dependency_versions(),'models':{'writer_reader':'gemini-2.5-flash','judge':'gemini-3.5-flash-lite',
            'embedding':'gemini-embedding-001','dimensions':768,'grouping_reused_jev':'jev-1.13.0'},
        'provider_retries':2,'conversation_workers':8,'query_workers':4,'final_unique_facts':20,
        'qa_reader':'brief_answer_local_evidence_ids_v2_with_response_schema',
        'native_add_policy':'Mem0 2.2.1 additive ADD, upstream prompt plus existing date/ordered_history adapter',
        'baseline_reuse':'facts and native hybrid search only, no original answers',
        'limitations':['Whole conversation offline grouping; no online-causal claim.','All data previously inspected; no blind test claim.',
            'No independent-writer or assistant semantic filtering is transferred.',
            'Native input_scope is not fact-level verified supporting evidence.']}
    save_once(OUT/'manifest.json',m)
    return m

def conv_path(ci):return OUT/('conv_'+str(ci))
def start_state(ci,stage,expected):
    return {'policy':POLICY,'stage':stage,'conversation':ci,'status':'running','pid':os.getpid(),
        'started_utc':now(),'updated_utc':now(),'expected':expected,'done':0,'failed':0}
def state_write(ci,stage,state):
    state['updated_utc']=now();publish_json(conv_path(ci)/(stage+'_status.json'),state)

def sidecar_audit(memory,facts,user_id):
    from mem0.memory.main import extract_entities_batch
    for f in facts:
        history=memory.db.get_history(f['id'])
        if len(history)!=1 or history[0]['event']!='ADD' or history[0]['new_memory']!=f['text'] or history[0]['is_deleted']:
            raise ValueError('Native history missing or inconsistent for a persisted ADD')
    entities=memory._existing_entities_by_text({'user_id':user_id})
    linked=set()
    for point in entities.values():linked.update((point.payload or {}).get('linked_memory_ids',[]))
    expected={f['id'] for f,found in zip(facts,extract_entities_batch([f['text'] for f in facts])) if found}
    all_ids={f['id'] for f in facts}
    if linked-all_ids:raise ValueError('Native entity index links to another/absent memory')
    result={'history_adds_verified':len(facts),'entities':len(entities),'linked_memory_ids':sorted(linked),
        'facts_with_local_entities':len(expected),'entity_facts_without_any_link':sorted(expected-linked),
        'scope':'Native store consistency only, not factual semantic support.'}
    if expected-linked:raise ValueError('Native entity linking omitted extracted entity-bearing facts')
    return result

def write_conv(ci):
    manifest=verify_manifest(ci);folder=conv_path(ci);inp=read(folder/'group_inputs.json')
    expected=manifest['conversations'][ci]
    if sha(folder/'group_inputs.json')!=expected['group_inputs_sha256']:raise ValueError('Group inputs changed')
    state=start_state(ci,'write',len(inp['envelopes']));state_write(ci,'write',state)
    api=ModelAPI(folder/'native',retries=2,context={'architecture':POLICY,'conversation':ci,'method':ARMS[1],'phase':'write'})
    memory=None
    try:
        memory=native.open_memory(folder/'native',api)
        from mem0.utils import spacy_models
        if spacy_models.get_nlp_full() is None or spacy_models.get_nlp_lemma() is None or memory.vector_store._get_bm25_encoder() is None:
            raise RuntimeError('Native NLP/BM25 dependencies unavailable')
        writer=Mem0SessionAdapter(memory,folder/'native/session_audit',current_date=CURRENT_DATE)
        records=[]
        for envelope in inp['envelopes']:
            scoped=api.scoped(group_id=envelope['session_id'])
            memory.llm=native.ExtractionLLM(scoped);memory.embedding_model=Embeddings(scoped)
            r=writer.add_session(user_id='mem0_sparse1540_c'+str(ci),**envelope)
            records.append(r);state['done']+=1
            state['facts_added']=sum(len(x['fact_records']) for x in records)
            state['last_group']=envelope['session_id'];state_write(ci,'write',state)
        facts=[f for r in records for f in r['fact_records']]
        for f in facts:
            point=memory.vector_store.client.retrieve(collection_name='facts',ids=[f['id']],with_vectors=True,with_payload=True)[0]
            if point.payload['data']!=f['text']:raise ValueError('Native persisted fact changed')
            f['embedding']=point.vector.get('') if isinstance(point.vector,dict) else point.vector
        vector_ok(facts)
        if len(facts)<20:raise ValueError('Native sparse bank has fewer than20 facts')
        indexes=sidecar_audit(memory,facts,'mem0_sparse1540_c'+str(ci))
        bank={'schema':POLICY+'-bank','arm':ARMS[1],'conversation':ci,'facts':facts,
            'source_dataset_sha256':SOURCE_HASH,'group_inputs_sha256':sha(folder/'group_inputs.json'),
            'manifest_sha256':sha(OUT/'manifest.json'),'complete':True,'reused_native_bank':False,
            'fact_support':'input_scope_unverified_not_gold','user_id':'mem0_sparse1540_c'+str(ci),
            'journal_files':[{'path':r['audit_path'],'sha256':sha(r['audit_path'])} for r in records],
            'native_index_audit':indexes,
            'sdk_warning_count':sum(len(r['sdk_warnings']) for r in records)}
        save_once(folder/'regroup_bank.json',bank)
        state.update(status='complete',facts=len(facts),bank_sha256=sha(folder/'regroup_bank.json'),
                     sdk_warning_count=bank['sdk_warning_count'])
    except Exception as exc:
        state.update(status='failed',failed=1,error={'type':type(exc).__name__,'message':api._redact(str(exc))[:1500]})
    finally:
        if memory is not None:native.close_memory(memory)
        state['finished_utc']=now();state_write(ci,'write',state)
    return state

def retrieve_conv(ci):
    manifest=verify_manifest(ci);folder=conv_path(ci);bank,bank_hash=load_bank(ci,ARMS[1],manifest)
    by_id={f['id']:f for f in bank['facts']}
    qs=questions(ci);state=start_state(ci,'retrieve',len(qs));state_write(ci,'retrieve',state)
    api=ModelAPI(folder/'native',retries=2,context={'architecture':POLICY,'conversation':ci,'method':ARMS[1],'phase':'retrieve'})
    memory=None
    try:
        memory=native.open_memory(folder/'native',api)
        if memory.vector_store._get_bm25_encoder() is None or not memory.vector_store._has_bm25_slot:
            raise RuntimeError('Native keyword index unavailable')
        original_keyword_search=memory.vector_store.keyword_search
        def require_keyword_search(*args,**kwargs):
            result=original_keyword_search(*args,**kwargs)
            if result is None:raise RuntimeError('Native BM25 keyword search failed; no silent dense-only fallback')
            return result
        memory.vector_store.keyword_search=require_keyword_search
        for q in qs:
            path=folder/'retrieval'/ARMS[1]/(q['id']+'.json')
            if path.exists():
                row=validate_retrieval(read(path),q,ARMS[1],bank,bank_hash)
            else:
                memory.embedding_model=Embeddings(api.scoped(query_id=q['id']))
                result=memory.search(q['question'],top_k=60,filters={'user_id':bank['user_id']},rerank=False)
                pool=[{**by_id[r['id']],'score':r.get('score'),'sources':['mem0_hybrid']} for r in result['results']]
                row=make_retrieval(q,ARMS[1],pool,bank_hash,{'new_native_search':True,'top_k':60,'threshold':0.1,'rerank':False})
                save_once(path,row)
            state['done']+=1;state['last_query_id']=q['id'];state_write(ci,'retrieve',state)
        state.update(status='complete')
    except Exception as exc:
        state.update(status='failed',failed=1,error={'type':type(exc).__name__,'message':api._redact(str(exc))[:1500]})
    finally:
        if memory is not None:native.close_memory(memory)
        state['finished_utc']=now();state_write(ci,'retrieve',state)
    return state

def qa_conv(ci):
    from jev_mem.mem0_transfer_qa import StructuredMem0QAAPI
    manifest=verify_manifest(ci);folder=conv_path(ci);qs=questions(ci)
    if read(folder/'retrieve_status.json')['status']!='complete':raise ValueError('Both native pools must be complete beforeQA')
    reference_date=list(session_envelopes(load_data()[ci]))[-1]['observation_date']
    banks={};bank_hashes={}
    for arm in ARMS:banks[arm],bank_hashes[arm]=load_bank(ci,arm,manifest)
    bank_facts={a:{f['id']:f for f in b['facts']} for a,b in banks.items()}
    base=ModelAPI(folder/'qa_api',retries=2,context={'architecture':POLICY,'conversation':ci,'phase':'qa'})
    state=start_state(ci,'qa',len(qs)*2);state['completed']=0;state['arms']={a:{'completed':0,'failed':0} for a in ARMS}
    state_write(ci,'qa',state)
    def one(q,arm):
        rp=folder/'retrieval'/arm/(q['id']+'.json')
        retrieval=validate_retrieval(read(rp),q,arm,banks[arm],bank_hashes[arm]);selected=retrieval['selected']
        if retrieval['bank_sha256']!=bank_hashes[arm] or retrieval['question_sha256']!=fingerprint(q['question']):
            raise ValueError('QA retrieval/bank binding mismatch')
        if len(selected)!=20 or len({f['id'] for f in selected})!=20 or any(f!=fact_view(bank_facts[arm][f['id']]) for f in selected):
            raise ValueError('QA native evidence differs from bank')
        binding={'manifest_sha256':sha(OUT/'manifest.json'),'bank_sha256':bank_hashes[arm],
            'retrieval_sha256':sha(rp),'selected_sha256':fingerprint(selected),'question_sha256':fingerprint(q['question']),
            'reference_sha256':fingerprint(q['reference']),'reference_date':reference_date}
        path=folder/'qa'/arm/(q['id']+'.json')
        if path.exists():
            saved=_verify_sealed(read(path))
            if saved['binding']!=binding or saved['status'] not in ('complete','failed'):raise ValueError('Existing QA binding changed')
            return saved
        api=StructuredMem0QAAPI(base.scoped(query_id=q['id'],method=arm));started=now();result=None;error=None
        try:result=native.answer_and_judge(api,q['question'],q['reference'],selected,reference_date=reference_date)
        except Exception as exc:error={'type':type(exc).__name__,'message':base._redact(str(exc))[:1500]}
        row=_sealed({'schema':POLICY+'-qa','query_id':q['id'],'conversation':ci,'category':q['category'],'arm':arm,
            'binding':binding,'status':'complete' if result is not None else 'failed','result':result,'error':error,
            'selected_ids':[f['id'] for f in selected],'api_events':api.events,'started_utc':started,'finished_utc':now()})
        save_once(path,row);return row
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one,q,arm) for q in qs for arm in ARMS]
            for f in as_completed(futures):
                row=f.result();key='completed' if row['status']=='complete' else 'failed'
                state[key]+=1;state['arms'][row['arm']][key]+=1;state['done']+=1
                state['last_query_id']=row['query_id'];state_write(ci,'qa',state)
        state['status']='complete' if state['failed']==0 else 'requires_review'
    except Exception as exc:
        state.update(status='blocked',error={'type':type(exc).__name__,'message':base._redact(str(exc))[:1500]})
    state['finished_utc']=now();state_write(ci,'qa',state);return state

def launch(stage,conversations):
    verify_manifest();function={'write':write_conv,'retrieve':retrieve_conv,'qa':qa_conv}[stage]
    p=OUT/(stage+'_coordinator_status.json')
    if p.exists() and read(p)['status']=='running':raise ValueError('Coordinator marked running; inspect ownership before any restart')
    state={'policy':POLICY,'stage':stage,'pid':os.getpid(),'status':'running','started_utc':now(),'updated_utc':now(),
        'conversations':conversations,'workers':min(8,len(conversations)),'finished':[],'failed':[]}
    publish_json(p,state)
    try:
        with ProcessPoolExecutor(max_workers=min(8,len(conversations))) as pool:
            tasks={pool.submit(function,ci):ci for ci in conversations}
            for f in as_completed(tasks):
                ci=tasks[f]
                try:out=f.result();status=out['status']
                except Exception as exc:status='worker_error';state.setdefault('errors',[]).append({'conversation':ci,'type':type(exc).__name__})
                state['finished'].append(ci)
                if status!='complete':state['failed'].append(ci)
                state['updated_utc']=now();publish_json(p,state)
        state['status']='complete' if not state['failed'] else 'requires_review'
    except BaseException as exc:
        state.update(status='interrupted',error_type=type(exc).__name__)
        raise
    finally:
        state['finished_utc']=now();state['updated_utc']=now();publish_json(p,state)
    return state

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','write','retrieve','qa'))
    parser.add_argument('--conversations',default='0,1,2,3,4,5,6,7,8,9')
    args=parser.parse_args();cis=[int(x) for x in args.conversations.split(',')]
    if not cis or len(set(cis))!=len(cis) or any(i not in range(10) for i in cis):raise ValueError('Invalid conversation list')
    if args.stage in ('prepare','check'):
        m=prepare() if args.stage=='prepare' else verify_manifest()
        print(json.dumps({'policy':m['policy'],'questions_per_arm':1540,'arms':m['arms'],'conversations':m['conversations'],'model_calls':0}))
    else:print(json.dumps(launch(args.stage,cis),indent=2))

if __name__=='__main__':main()
