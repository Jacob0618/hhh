"""UPDATE-only native tracing, exact local replay, and durable embedding closure."""
from __future__ import annotations
from collections import Counter
import contextlib,copy,inspect,json,tempfile
from pathlib import Path
from tools import materialize_write_omission_r07 as old
from tools.run_sdk20_modules import digest
from jev_mem.providers import ModelAPI
from jev_mem.sdk20_cached_api import digest as request_hash
SCHEMA='qualifier-repair-r09-native-v1'
require=old.require

class ReplayEmbedding:
    embed=ModelAPI.embed
    def __init__(self,response):self.response=response;self.signature=None
    def request(self,provider,model,body,*,operation,stage):
        require(stage in ('mem0_embedding_add','mem0_embedding_update'),'Unexpected UPDATE embedding action')
        require(self.signature is None,'Expected one bounded native embedding request')
        self.signature={'provider':provider,'model':model,'operation':operation,'body':body}
        return self.response
    def count_tokens(self,*a,**k):raise ValueError('Bounded UPDATE unexpectedly requires countTokens')

def signature(text,action):
    x=ReplayEmbedding({'embeddings':[{'values':[1.]+[0.]*767}]})
    x.embed([text],query=False,stage='mem0_embedding_'+action);return x.signature

def runtime_pins():
    from qdrant_client import QdrantClient
    from qdrant_client.local import distances,local_collection,persistence
    from mem0.vector_stores.qdrant import Qdrant
    return {str(Path(inspect.getfile(x)).resolve()):old.sha(inspect.getfile(x))
            for x in (Qdrant,QdrantClient,distances,local_collection,persistence)}

@contextlib.contextmanager
def observe_update(memory,mid):
    """Delegate every original operation. No SDK/source/global patches."""
    memory.entity_store  # Existing collection must be initialized before cleanup.
    operations=[];embeddings=[];errors=[]
    stores={'fact':memory.vector_store,'entity':memory.entity_store}
    saved=[]
    for name,store in stores.items():
        methods=('update',) if name=='fact' else ('search','update','insert','delete')
        for method in methods:
            original=getattr(store,method);saved.append((store,method,original))
            def wrapper(*args,_original=original,_name=name,_method=method,**kwargs):
                require(not args,'Native store positional contract changed')
                op={'store':_name,'method':_method,'kwargs':copy.deepcopy(kwargs)}
                operations.append(op)
                try:
                    result=_original(**kwargs)
                    if _method=='search':
                        op['results']=[{'id':str(x.id),'score':x.score} for x in result]
                    return result
                except Exception as ex:
                    op['failure_type']=type(ex).__name__;errors.append(type(ex).__name__);raise
            setattr(store,method,wrapper)
    model=memory.embedding_model;original_embed=model.embed
    def embed(text,memory_action='add'):
        require(memory_action in ('add','update'),'Unexpected embedding action during UPDATE')
        value=original_embed(text,memory_action)
        embeddings.append({'text':text,'action':memory_action,'vector':copy.deepcopy(value)})
        return value
    model.embed=embed
    try:yield operations,embeddings,errors
    finally:
        model.embed=original_embed
        for store,method,original in saved:setattr(store,method,original)

def update_candidate(memory,candidate,user_id):
    mid=candidate['fact_id'];text=candidate['replacement_text']
    before=memory.vector_store.get(vector_id=mid)
    require(before is not None and before.payload.get('user_id')==user_id,'UPDATE target/scope missing')
    require(before.payload.get('data')==candidate['original_fact']['text']
            and before.payload.get('jev_mem_observation_date')==candidate['observation_date'],
            'UPDATE target original text/date differs')
    require(text!=before.payload['data'],'Noop must not dispatch native UPDATE')
    expected=old.expected_entities(memory,text)
    with observe_update(memory,mid) as (ops,embs,errors):
        result=memory.update(mid,text=text)
    require(result=={'message':'Memory updated successfully!'} and not errors,'Native UPDATE operation failed or swallowed')
    require(ops and ops[0]['store']=='fact' and ops[0]['method']=='update'
            and ops[0]['kwargs'].get('vector_id')==mid,'UPDATE fact mutation not first')
    require(sum(o['store']=='fact' for o in ops)==1,'More than one fact mutation in UPDATE')
    new=memory.vector_store.get(vector_id=mid)
    require(new.payload['data']==text,'Native UPDATE text not persisted')
    rows=memory.entity_store.list(filters={'user_id':user_id},top_k=10000)
    rows=rows[0] if rows and isinstance(rows[0],list) else rows
    linked={memory._normalize_entity_text((x.payload or {}).get('data','')) for x in rows
            if mid in (x.payload or {}).get('linked_memory_ids',[])}
    # Exact/semantic merge may canonicalize the name, so trace the actual link IDs.
    inserted={str(eid) for o in ops if o['store']=='entity' and o['method']=='insert'
              for eid in o['kwargs']['ids']}
    targets={str(o['kwargs']['vector_id']) for o in ops if o['store']=='entity'
             and o['method']=='update' and mid in o['kwargs'].get('payload',{}).get('linked_memory_ids',[])}|inserted
    linked_ids={str(x.id) for x in rows if mid in (x.payload or {}).get('linked_memory_ids',[])}
    require(targets==linked_ids and (not expected or linked_ids),'Native UPDATE entity relink missing or stale')
    # Every deduplicated NER entity calls the original upsert embed("add") once.
    require([memory._normalize_entity_text(e['text']) for e in embs if e['action']=='add']==expected,
            'Native UPDATE extraction/upsert swallowed or missing')
    return {'candidate':copy.deepcopy(candidate),'memory_id':mid,'status':'updated',
        'native_result':result,'operations':ops,'embeddings':embs,'expected_entity_keys':expected,
        'linked_entity_ids':sorted(linked_ids),'native_runtime_code_sha256':runtime_pins()}

def validate_delta(before,after,rows,user_id):
    import hashlib
    mids=[r['memory_id'] for r in rows]
    require(len(mids)==len(set(mids)) and set(after['facts'])==set(before['facts']),
            'UPDATE population/target IDs differ')
    by={r['memory_id']:r for r in rows}
    require(set(by)<=set(before['facts']),'UPDATE references foreign original ID')
    for mid,p in before['facts'].items():
        new=after['facts'][mid]
        if mid not in by:
            require(new==p,'Unaccepted original fact changed');continue
        r=by[mid];c=r['candidate'];n=new['payload'];o=p['payload']
        require(r['status']=='updated' and c['fact_id']==mid and c['original_fact']['id']==mid
                and c['original_fact']['text']==o['data']
                and c['observation_date']==o.get('jev_mem_observation_date') and
                n['data']==c['replacement_text'] and n['user_id']==user_id,'Accepted UPDATE text/scope differs')
        fact_ops=[x for x in r['operations'] if x['store']=='fact']
        require(len(fact_ops)==1 and fact_ops[0]==r['operations'][0] and fact_ops[0]['method']=='update'
                and fact_ops[0]['kwargs']['vector_id']==mid and fact_ops[0]['kwargs']['payload']==n,
                'UPDATE operation is not its accepted fact payload')
        require(r['embeddings'] and r['embeddings'][0]['action']=='update'
                and r['embeddings'][0]['text']==c['replacement_text']
                and fact_ops[0]['kwargs']['vector']==r['embeddings'][0]['vector'],
                'UPDATE fact vector not bound to original UPDATE embedding action')
        declared={(e['text'],e['action'],digest(e['vector'])) for e in r['embeddings']}
        for op in r['operations'][1:]:
            require(op['store']=='entity','Foreign fact UPDATE in trace')
            kw=op['kwargs']
            if op['method']=='update' and kw.get('vector') is not None:
                require((kw['payload']['data'],'update',digest(kw['vector'])) in declared,
                    'Shared entity reembedding not bound to UPDATE raw action')
            elif op['method']=='insert':
                require(len(kw['ids'])==len(kw['vectors'])==len(kw['payloads'])==1 and
                    (kw['payloads'][0]['data'],'add',digest(kw['vectors'][0])) in declared,
                    'New entity vector not bound to ADD raw action')
        unchanged={k:v for k,v in o.items() if k not in ('data','hash','text_lemmatized','updated_at')}
        require({k:v for k,v in n.items() if k not in ('data','hash','text_lemmatized','updated_at')}==unchanged,
                'Accepted UPDATE changed original identity/date/created_at/custom metadata')
        require(n['hash']==hashlib.md5(n['data'].encode()).hexdigest() and n.get('updated_at'),
                'Native UPDATE hash/storage time missing')
        require(old.dense(new['vector'].get('')) and old.sparse(new['vector'].get('bm25')),
                'Native UPDATE dense/BM25 invalid')
    prior=Counter(digest(x) for x in before['history']);final=Counter(digest(x) for x in after['history'])
    require(not prior-final and sum((final-prior).values())==len(rows),'Old history lost or extra UPDATE event')
    extra=[];left=prior.copy()
    for h in after['history']:
        key=digest(h)
        if left[key]:left[key]-=1
        else:extra.append(h)
    require(Counter(x['memory_id'] for x in extra)==Counter(mids),'History UPDATE target population differs')
    for h in extra:
        mid=h['memory_id'];o=before['facts'][mid]['payload'];n=after['facts'][mid]['payload']
        require(h['event']=='UPDATE' and not h['is_deleted'] and h['old_memory']==o['data']
                and h['new_memory']==n['data'] and h['created_at']==o['created_at']
                and h['updated_at']==n['updated_at'],'Actual UPDATE history differs')
    for eid,p in after['entities'].items():
        links=p['payload'].get('linked_memory_ids')
        require(isinstance(links,list) and links and len(links)==len(set(links)) and
                set(links)<=set(after['facts']) and p['payload'].get('user_id')==user_id,
                'Final entity link/scope malformed')
    for eid,p in before['entities'].items():
        preserved=set(p['payload'].get('linked_memory_ids',[]))-set(mids)
        require(not preserved or (eid in after['entities'] and preserved<=
                set(after['entities'][eid]['payload']['linked_memory_ids'])),'Unchanged memory entity link lost')
        if eid not in after['entities']:
            require(not preserved,'Shared old entity deleted')
    for r in rows:
        actual_links={eid for eid,p in after['entities'].items()
            if r['memory_id'] in p['payload']['linked_memory_ids']}
        require(set(r['linked_entity_ids'])==actual_links,
            'Declared relink target missing')
        require(r['native_runtime_code_sha256']==runtime_pins(),'UPDATE runtime code changed')
    return {'same_fact_ids_count':len(before['facts']),'updated':len(rows),'unchanged':len(before['facts'])-len(rows),
            'history_updates':len(extra),'semantic_support_certified':False}

def replay_exact(original_native,before,after,rows,resources):
    """Actual Qdrant operation replay including deletes/inserts/dense+sparse writes."""
    from jev_mem.local_bm25_context_v2 import local_bm25_context,forbid_network
    pins=runtime_pins();inv=old.inventory(original_native)
    require(old.snapshot(original_native)==before,'Original UPDATE snapshot differs')
    class NoAPI:
        keys={'GEMINI_API_KEY':'offline-only-placeholder'}
        def request(self,*a,**k):raise AssertionError('Numeric replay may not request a model')
    with tempfile.TemporaryDirectory(prefix='r09-update-replay-') as temp:
        dest=Path(temp)/'native';old.copy_closed(original_native,dest,inv);memory=None
        try:
            with local_bm25_context(resources['snapshot'],resources['files_sha256']),forbid_network():
                memory=old.writer.open_memory(dest,NoAPI());memory.entity_store
                for row in rows:
                    for op in row['operations']:
                        require(set(op)<= {'store','method','kwargs','results'} and op['store'] in ('fact','entity')
                            and op['method'] in (('update',) if op['store']=='fact' else
                                ('search','update','insert','delete')),'Invalid UPDATE operation trace')
                        store=memory.vector_store if op['store']=='fact' else memory.entity_store
                        result=getattr(store,op['method'])(**copy.deepcopy(op['kwargs']))
                        if op['method']=='search':
                            require([{'id':str(x.id),'score':x.score} for x in result]==op['results'],
                                    'Native UPDATE semantic search trace differs')
                old.native.close_memory(memory);memory=None
            actual=old.snapshot(dest)
            require(actual['facts']==after['facts'] and actual['entities']==after['entities'],
                    'Native full UPDATE operation replay not exact')
            require(actual['history']==before['history'],'Store-only replay changed history')
        finally:
            if memory is not None:old.native.close_memory(memory)
    require(old.inventory(original_native)==inv and runtime_pins()==pins,'Original/runtime changed during replay')
    return {'native_full_update_store_replay_exact':True,'before_sha256':digest(before),
        'after_sha256':digest(after),'ordered_updates_sha256':digest(rows),'native_code_sha256':pins,
        'model_calls':0,'epsilon_used':False,'history_verified_separately':True}

def audit_embeddings(api_dir,ci,by_arm):
    """All action occurrences/contexts, physical producers and raw inputs closed."""
    api_dir=Path(api_dir);p=api_dir/'api_calls.jsonl'
    data=old.shared_read(p) if p.exists() else b''
    ledger=[json.loads(x) for x in data.decode('utf-8-sig').splitlines() if x.strip()]
    declared=Counter();known={};consumer_proofs=[]
    for arm,rows in by_arm.items():
        for row in rows:
            c=row['candidate'];ctx={'architecture':SCHEMA,'conversation':ci,'method':arm,
                'phase':'qualifier_repair','candidate_id':c['fact_id'],'group_id':c['source_session_id']}
            for e in row['embeddings']:
                sig=signature(e['text'],e['action']);key=request_hash(sig);stage='mem0_embedding_'+e['action']
                declared[(key,digest(ctx),stage)]+=1;known[key]=sig
                rawref=old.pointer(api_dir/f'api_cache/{key}.json');raw=old.bound(rawref)
                require(raw['request']==sig,'UPDATE raw full signature differs')
                response=raw['response'];require(response.get('modelVersion',response.get('model')) in
                    (None,'gemini-embedding-001'),'UPDATE raw model differs')
                replay=ReplayEmbedding(response);v=replay.embed([e['text']],query=False,stage=stage)[0]
                require(replay.signature==sig and e['vector']==v,'UPDATE embedding trace is not raw response')
                consumer_proofs.append({'hash':key,'stage':stage,'context':ctx,'raw':rawref,
                    'resolved_model_available':response.get('modelVersion',response.get('model')) is not None})
    actual=Counter()
    for x in ledger:
        require(x.get('provider')=='gemini' and x.get('model')=='gemini-embedding-001' and
                x.get('operation')=='batchEmbedContents' and x.get('resolved_model') in
                (None,'gemini-embedding-001'),'UPDATE ledger model/provider differs')
        token=(x['request_hash'],digest(x.get('context')),x.get('stage'))
        require(token in declared,'Undeclared UPDATE physical/cache context or action')
        if x.get('cache_hit') is True:
            require('attempt' not in x and 'status' not in x and x.get('usage_semantics')==
                'reused_response_no_new_model_usage','Malformed UPDATE cache event');actual[token]+=1
        elif x.get('status')==200:actual[token]+=1
    require(actual==declared,'UPDATE invocation occurrence population differs')
    keys=set(known)
    require({x.stem for x in (api_dir/'api_cache').glob('*.json')}==keys and
            {x.stem for x in (api_dir/'request_intents').glob('*.json')}==keys and
            not list((api_dir/'cache_lineage').glob('*.json')),'Orphan/undeclared imported UPDATE raw/intent')
    producers=[]
    for key,sig in sorted(known.items()):
        ir=old.pointer(api_dir/f'request_intents/{key}.json');intent=old.bound(ir)
        raw=old.bound(old.pointer(api_dir/f'api_cache/{key}.json'))
        require(intent['signature']==sig and intent['request_hash']==key and intent['retries']==2 and
                intent['stage']==raw['stage'],'UPDATE producer intent/raw identity differs')
        physical=[(i,x) for i,x in enumerate(ledger) if x['request_hash']==key and x.get('cache_hit') is not True]
        require(physical and all(x['context']==intent['context'] and x['stage']==intent['stage'] for i,x in physical)
                and [x['attempt'] for i,x in physical]==list(range(1,len(physical)+1))
                and len(physical)<=3 and physical[-1][1]['status']==200
                and all(x['status'] in ('transport_error',429,500,502,503,504,529) for i,x in physical[:-1]),
                'UPDATE physical producer terminal/attempts differ')
        require(all(i>physical[-1][0] for i,x in enumerate(ledger) if x['request_hash']==key and
                    x.get('cache_hit') is True),'UPDATE cache consumer before producer success')
        producers.append({'request_hash':key,'intent':ir,'producer_context':intent['context']})
    require((old.shared_read(p) if p.exists() else b'')==data,'UPDATE ledger changed while auditing')
    return {'conversation':ci,'logical_invocations':sum(declared.values()),'unique_signatures':len(keys),
        'physical_attempts':sum(x.get('cache_hit') is not True for x in ledger),
        'cache_consumers':sum(x.get('cache_hit') is True for x in ledger),
        'unknown_usage_rows':sum(not x.get('usage') for x in ledger if x.get('cache_hit') is not True),
        'ledger':old.pointer(p) if ledger else None,'producers':producers,'consumers':consumer_proofs}

def arm_embedding_proof(proof,arm):
    """Stable per-arm raw lineage; the mutable shared ledger is bound by final gate."""
    consumers=[x for x in proof['consumers'] if x['context']['method']==arm]
    keys={x['hash'] for x in consumers}
    return {'consumers':consumers,'producers':[x for x in proof['producers'] if x['request_hash'] in keys]}
