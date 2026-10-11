"""Dated unmerged/merged input adapters around the original native Mem0 writer.

No work runs on import. This file changes source rendering and honest mixed-date
metadata, not Mem0 extraction semantics, models, SDK search or the QA reader.
"""
from __future__ import annotations
import argparse,contextlib,contextvars,copy,hashlib,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_session_experiments as native
from tools import run_mem0_sparse1540 as sparse
from tools.run_pilot400_experiments_v2 import read,publish_json
from tools.run_sdk20_modules import digest,sealed,verify,sha,now
from jev_mem import mem0_adapter as original
from jev_mem.data import load_data,session_envelopes,SOURCE_HASH
from jev_mem.topic_grouping import raw_turns_from_locomo,verify_partition
from jev_mem.providers import ModelAPI,Embeddings

SCHEMA='native-mem0-dated-regroup-sdk20-v1'
ARMS=('dated_unmerged385','dated_merged')
CURRENT_DATE='2026-09-27'
MULTIPLE_DATES=('PER_MESSAGE_OBSERVATION_DATES: These messages have multiple actual observation dates. '
    'For each message, its SOURCE observation_date is its Observation Date. Resolve relative time against '
    'that individual date, never a group date or Current Date. Preserve dates and source attribution; '
    'juxtaposition of turns does not establish a new relationship.')
_CONTEXT=contextvars.ContextVar('mem0_dated_group_context',default=None)
CODE=('tools/run_mem0_regroup_sdk20.py','tools/run_session_experiments.py','tools/run_mem0_sparse1540.py',
      'tools/run_pilot400_experiments_v2.py','tools/run_sdk20_modules.py','jev_mem/mem0_adapter.py',
      'jev_mem/providers.py','jev_mem/data.py','jev_mem/topic_grouping.py','jev_mem/evaluation.py')

def pointer(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check_pointer(p):
    if sha(p['path'])!=p['sha256']:raise ValueError('Changed source: '+p['path'])
    return read(p['path'])
def save_new(p,value):
    p=Path(p)
    if p.exists():raise FileExistsError('Refusing to replace immutable artifact: '+str(p))
    publish_json(p,sealed(value))

def render_source(message,source):
    header={k:source[k] for k in ('source_turn_id','session_id','observation_date')}
    return {**message,'content':'[SOURCE '+json.dumps(header,ensure_ascii=False,sort_keys=True)+']\n'+message['content']}

class GroupContextEmbeddings:
    """Adapt only the exact active group's native search context, without truncation."""
    def __init__(self,delegate,messages,source_turn_ids,*,audit_callback=None):
        from mem0.memory.utils import parse_messages
        self.delegate=delegate;self.config=delegate.config;self.api=delegate.api
        self.parts=[parse_messages([m]) for m in messages];self.text=''.join(self.parts)
        if self.text!=parse_messages(messages):raise ValueError('Native source serialization is not additive')
        self.ids=list(source_turn_ids);self.audit_callback=audit_callback;self.audit=None
        self._counts={};self._offsets=[];offset=0
        for part in self.parts:self._offsets.append((offset,offset+len(part)));offset+=len(part)
    def __getattr__(self,name):return getattr(self.delegate,name)
    def embed_batch(self,texts,memory_action='add'):return self.delegate.embed_batch(texts,memory_action)
    def _count(self,text):
        if text not in self._counts:
            value=self.api.count_tokens(text,model='gemini-embedding-001',stage='group_context_exact_tokens')
            if type(value)is not int or value<0:raise ValueError('Invalid exact group token count')
            self._counts[text]=value
        return self._counts[text]
    def _publish(self):
        if self.audit_callback:self.audit_callback(copy.deepcopy(self.audit))
    def embed(self,text,memory_action='add'):
        if memory_action!='search' or text!=self.text:return self.delegate.embed(text,memory_action)
        total=self._count(text)
        self.audit={'schema':SCHEMA+'-context-embedding','source_sha256':digest(text),'total_tokens':total,
            'triggered':total>2048,'limit':2048,'model':'gemini-embedding-001','dimensions':768,
            'task_type':'RETRIEVAL_QUERY','algorithm':'turn_midpoint_then_single_turn_character_midpoint_v1',
            'fusion':'exact_block_token_weighted_mean_then_l2_normalize','chunks':[],'status':'running'}
        self._publish()
        if total<=2048:
            vector=self.delegate.embed(text,memory_action);self.audit['status']='complete';self._publish();return vector
        spans=[]
        def split(start,end,boundaries):
            value=self._count(text[start:end])
            if value<=2048:spans.append((start,end,value));return
            internal=[b for b in boundaries if start<b<end]
            cut=internal[(len(internal)-1)//2] if internal else (start+end)//2
            if cut<=start or cut>=end:raise ValueError('Unsplittable single character exceeds embedding budget')
            split(start,cut,internal);split(cut,end,internal)
        split(0,len(text),[end for start,end in self._offsets[:-1]])
        if ''.join(text[a:b] for a,b,n in spans)!=text or any(n>2048 for a,b,n in spans):raise ValueError('Group embedding chunks lost source content')
        self.audit['chunks']=[{'index':i,'start_char':a,'end_char':b,'tokens':n,'text_sha256':digest(text[a:b]),
            'source_turn_ids':[tid for tid,(x,y) in zip(self.ids,self._offsets) if max(x,a)<min(y,b)]}
            for i,(a,b,n) in enumerate(spans)]
        self.audit['exact_count_requests']=len(self._counts);self._publish()
        weighted=[0.0]*768;weight=sum(n for a,b,n in spans)
        if weight<=0:raise ValueError('Group embedding has no positive token weight')
        for a,b,n in spans:
            v=self.delegate.embed(text[a:b],'search')
            if len(v)!=768 or any(type(x)not in (int,float) or not math.isfinite(x) for x in v):raise ValueError('Invalid group chunk vector')
            for i,x in enumerate(v):weighted[i]+=n*x/weight
        norm=math.sqrt(sum(x*x for x in weighted))
        if not math.isfinite(norm) or norm<=0:raise ValueError('Degenerate group context vector')
        vector=[x/norm for x in weighted];self.audit.update(status='complete',token_weight_sum=weight,
            fused_vector_sha256=digest(vector),norm_before_normalization=norm);self._publish();return vector

def make_envelopes(item,grouping,arm):
    """Bind group IDs to original messages, keeping arbitrary noncontiguous groups."""
    if arm not in ARMS or grouping.get('status')!='complete' or grouping.get('qa_visible')is not False:
        raise ValueError('Expected complete, QA-invisible global grouping')
    raw=raw_turns_from_locomo(item['conversation']);positions={t['source_turn_id']:i for i,t in enumerate(raw)}
    verify_partition(raw,grouping['segments']);verify_partition(raw,grouping['groups'])
    segments={s['segment_id']:s for s in grouping['segments']}
    if len(segments)!=len(grouping['segments']):raise ValueError('Duplicate segment IDs')
    assigned=[]
    for group in grouping['groups']:
        sids=group['segment_ids'];assigned.extend(sids)
        if not sids or len(set(sids))!=len(sids) or any(s not in segments for s in sids):raise ValueError('Unknown or duplicate merged segment')
        expected=sorted((t for sid in sids for t in segments[sid]['source_turn_ids']),key=positions.__getitem__)
        if expected!=group['source_turn_ids']:raise ValueError('Merged group does not contain its complete original segments')
    if len(assigned)!=len(segments) or set(assigned)!=set(segments):raise ValueError('Every segment must belong to one merged group')
    source={}
    for session in session_envelopes(item):
        for tid,message in zip(session['source_turn_ids'],session['messages']):
            source[tid]=({'source_turn_id':tid,'session_id':session['session_id'],'observation_date':session['observation_date'],
                          'source_order_index':positions[tid]},message)
    groups=grouping['segments'] if arm==ARMS[0] else grouping['groups']
    envelopes=[]
    for index,group in enumerate(sorted(groups,key=lambda g:min(positions[t] for t in g['source_turn_ids']))):
        ids=group['source_turn_ids'];members=[source[t] for t in ids]
        dates=sorted({s['observation_date'] for s,msg in members})
        sids=[group['segment_id']] if arm==ARMS[0] else list(group['segment_ids'])
        envelopes.append({'group_id':('dated_segment_'+str(index).zfill(4)) if arm==ARMS[0] else group['group_id'],
            'segment_ids':sids,'messages':[copy.deepcopy(msg) for s,msg in members],
            'turn_sources':[copy.deepcopy(s) for s,msg in members], 'source_turn_ids':list(ids),
            'source_order_indices':[positions[t] for t in ids],'source_observation_dates':dates,
            'observation_date':dates[0] if len(dates)==1 else None})
    return envelopes

@contextlib.contextmanager
def adapt_group_builder(record):
    from mem0.memory import main as module
    from mem0.memory.utils import parse_messages
    previous=module.generate_additive_extraction_prompt;token=_CONTEXT.set(record)
    def builder(*args,**kwargs):
        active=_CONTEXT.get()
        if active is not None:
            kwargs['timestamp']=active['observation_date'] or MULTIPLE_DATES
            kwargs['current_date']=active['current_date']
            kwargs['last_k_messages']=[x['writer_message'] for x in active['history_context']]
            kwargs['new_messages']=parse_messages(active['writer_messages'])
            active['builder_call_count']+=1
        return previous(*args,**kwargs)
    module.generate_additive_extraction_prompt=builder
    try:yield
    finally:module.generate_additive_extraction_prompt=previous;_CONTEXT.reset(token)

class Mem0GroupAdapter(original.Mem0SessionAdapter):
    """Reuse pinned SDK, capture wrappers and persistence contract for mixed dates."""
    def add_group(self,*,user_id,group_id,segment_ids,messages,turn_sources,source_turn_ids,
                  source_order_indices,source_observation_dates,observation_date):
        from mem0.memory import main as module
        if not user_id or not group_id or not messages or len(set(source_turn_ids))!=len(messages):raise ValueError('Malformed nonempty group identity')
        if not(len(messages)==len(turn_sources)==len(source_turn_ids)==len(source_order_indices)):raise ValueError('Group source dimensions differ')
        if source_order_indices!=sorted(source_order_indices) or len(set(source_order_indices))!=len(source_order_indices):raise ValueError('Group is not in source order')
        if not segment_ids or len(set(segment_ids))!=len(segment_ids):raise ValueError('Group requires unique segment IDs')
        dates=sorted({original._validate_date(s['observation_date']) for s in turn_sources})
        if dates!=source_observation_dates or observation_date!=(dates[0] if len(dates)==1 else None):raise ValueError('Mixed date groups must not claim one observation date')
        for message,source,tid,index in zip(messages,turn_sources,source_turn_ids,source_order_indices):
            if (message.get('role')not in ('user','assistant') or not isinstance(message.get('content'),str) or not message['content']
                or set(message)-{'role','content','name'} or source['source_turn_id']!=tid or source['source_order_index']!=index):
                raise ValueError('Original message/source mismatch')
        writer_messages=[render_source(m,s) for m,s in zip(messages,turn_sources)]
        request={'user_id':user_id,'group_id':group_id,'segment_ids':copy.deepcopy(segment_ids),'messages':copy.deepcopy(messages),
            'writer_messages':writer_messages,'turn_sources':copy.deepcopy(turn_sources),'source_turn_ids':list(source_turn_ids),
            'source_order_indices':list(source_order_indices),'source_observation_dates':list(dates),
            'observation_date':observation_date,'current_date':self.current_date}
        fingerprint=digest(request);path=self.audit_dir/original._safe_name(user_id)/(original._safe_name(group_id)+'.json')
        with original._INGEST_LOCK:
            prior=self._records(user_id)
            if path.exists():
                saved=read(path)
                if saved['request_sha256']!=fingerprint or saved['status']!='success':raise original.SessionRecoveryRequired('Prior group changed or unresolved; no automatic retry')
                for fact in saved['results']:
                    persisted=self.memory.vector_store.get(vector_id=fact['id'])
                    if persisted is None or (persisted.payload or {}).get('data')!=fact['memory']:raise original.SessionRecoveryRequired('Group checkpoint differs from store')
                return {**saved,'replayed':True}
            if any(r['status']!='success' for r in prior):raise original.SessionRecoveryRequired('Earlier group remains unresolved')
            if prior and min(source_order_indices)<=min(prior[-1]['source_order_indices']):raise ValueError('Groups must be submitted by first source position')
            already={t for r in prior for t in r['source_turn_ids']}
            if already.intersection(source_turn_ids):raise ValueError('Source turns reused across groups')
            history=[{**s,'message':m,'writer_message':wm,'group_id':r['group_id']}
                for r in prior for m,wm,s in zip(r['messages'],r['writer_messages'],r['turn_sources'])]
            record={**request,'status':'in_progress','request_sha256':fingerprint,'sequence':len(prior),
                'started_at':now(),'replayed':False,'history_context':history[-10:],'builder_call_count':0,
                'existing_memory_context':[],'llm_calls':[],'results':[],'fact_records':[],'sdk_warnings':[],
                'audit_path':str(path),'adapter_protocol':SCHEMA,'system_prompt_changed':False,'extraction_fact_limit':None,
                'patch_scope':['builder.new_messages_dated_source_rendering','builder.per_message_observation_dates',
                    'builder.current_date_frozen','builder.dated_ordered_history','mixed_date_metadata_null'],
                'offline_future_context_used':True,'support_status':'unverified_group_input_scope'}
            flush=lambda:publish_json(path,record);flush()
            llm,store,embeddings=self.memory.llm,self.memory.vector_store,self.memory.embedding_model;handler=original._WarningAudit(record);module.logger.addHandler(handler)
            self.memory.llm=original._CaptureLLM(llm,record,flush);self.memory.vector_store=original._CaptureVectorStore(store,record)
            try:
                def embedding_audit(value):record['group_context_embedding']=value;flush()
                self.memory.embedding_model=GroupContextEmbeddings(embeddings,messages,source_turn_ids,audit_callback=embedding_audit)
                metadata={'jev_mem_group_id':group_id,'jev_mem_observation_date':observation_date,
                    'jev_mem_source_observation_dates':dates,'jev_mem_source_turn_ids':list(source_turn_ids),
                    'jev_mem_segment_ids':list(segment_ids),'jev_mem_date_scope':'single_date_group' if len(dates)==1 else 'mixed_dates_unattributed'}
                # Native retrieval embedding sees the exact original role/content.
                # Only the extraction prompt gains the dated SOURCE wrappers;
                # oversized native search context uses an explicit audited mean.
                with adapt_group_builder(record):output=self.memory.add(copy.deepcopy(messages),user_id=user_id,metadata=metadata,infer=True)
                record['sdk_response']=output
                if record['builder_call_count']!=1 or len(record['llm_calls'])!=1:raise original.IncompleteSessionError('Expected exactly one native extraction per group')
                rows=output.get('results') if isinstance(output,dict) else None
                if not isinstance(rows,list) or any(not isinstance(r,dict) or r.get('event')!='ADD' or not r.get('id') or not r.get('memory') for r in rows):
                    raise original.IncompleteSessionError('Unexpected native ADD output')
                record['results']=rows;extracted=record['llm_calls'][0]['parsed_memory']
                # Preserve the original adapter's explicit memory:[] contract;
                # semantic adequacy is a later source-quality audit, not parsing.
                record['explicit_empty_extraction']=not extracted
                record['empty_extraction_source_audit_pending']=not extracted
                existing={r['hash'] for r in record['existing_memory_context'] if r['hash']}
                expected={r['text'] for r in extracted if hashlib.md5(r['text'].encode()).hexdigest()not in existing}
                actual={r['memory'] for r in rows}
                record['persistence_check']={'extracted_count':len(extracted),'expected_unique_nonduplicate_count':len(expected),
                    'returned_add_count':len(rows),'missing_texts':sorted(expected-actual),'unexpected_texts':sorted(actual-expected)}
                if expected!=actual:raise original.IncompleteSessionError('Native group ADDs are incomplete')
                for row in rows:
                    persisted=store.get(vector_id=row['id'])
                    if persisted is None or (persisted.payload or {}).get('data')!=row['memory']:raise original.IncompleteSessionError('Persisted native group ADD differs')
                    if any((persisted.payload or {}).get(k)!=v for k,v in metadata.items()):raise original.IncompleteSessionError('Persisted source/date metadata differs')
                scope={'group_id':group_id,'segment_ids':list(segment_ids),'source_turn_ids':list(source_turn_ids),
                    'turn_sources':copy.deepcopy(turn_sources),'historical_turn_ids':[s['source_turn_id'] for s in history[-10:]],
                    'existing_memory_ids':[r['id'] for r in record['existing_memory_context']]}
                record['fact_records']=[{'id':r['id'],'text':r['memory'],'event':'ADD','user_id':user_id,
                    'observation_date':observation_date,'source_observation_dates':dates,
                    'observation_date_scope':metadata['jev_mem_date_scope'],'input_scope':copy.deepcopy(scope),
                    'verified_support':[],'support_status':'unverified_group_input_scope'} for r in rows]
                record['status']='success';record['finished_at']=now();flush();return record
            except BaseException as ex:
                record['status']='requires_recovery';record['error_type']=type(ex).__name__;record['finished_at']=now();flush();raise
            finally:
                self.memory.llm,self.memory.vector_store,self.memory.embedding_model=llm,store,embeddings
                module.logger.removeHandler(handler)

def prepare(run_dir,grouping_dir):
    run=Path(run_dir).resolve();gp_root=Path(grouping_dir).resolve()
    if run.exists():raise FileExistsError('Writer preparation requires a new directory')
    inputs=[];sources={};data=load_data()
    for ci,item in enumerate(data):
        gp=gp_root/f'conversations/conv_{ci}/grouping.json';grouping=verify(gp)
        if grouping['conversation']!=ci:raise ValueError('Grouping crosses conversation scope')
        sources[str(gp)]=sha(gp)
        for arm in ARMS:
            env=make_envelopes(item,grouping,arm)
            inputs.append((ci,arm,{'schema':SCHEMA,'conversation':ci,'arm':arm,'grouping':pointer(gp),
                'envelopes':env,'turn_count':sum(len(e['messages']) for e in env),'dataset_sha256':SOURCE_HASH}))
    if sum(len(v['envelopes']) for ci,arm,v in inputs if arm==ARMS[0])!=385:raise ValueError('Expected385 fixed source segments')
    run.mkdir(parents=True);bindings=[]
    for ci,arm,value in inputs:
        p=run/f'conv_{ci}'/arm/'group_inputs.json';save_new(p,value);bindings.append({'conversation':ci,'arm':arm,'input':pointer(p)})
    m={'schema':SCHEMA,'created_utc':now(),'arms':list(ARMS),'dataset_sha256':SOURCE_HASH,'grouping_source_sha256':sources,
       'inputs':bindings,'code_sha256':{p:sha(ROOT/p) for p in CODE},'dependencies':sparse.dependency_versions(),
       'upstream_source_sha256':{str(ROOT/p):sha(ROOT/p) for p in sparse.CODE_FILES if p.startswith('upstream/')},
       'current_date':CURRENT_DATE,'writer':'native_Memory.add_original_ADDITIVE_EXTRACTION_PROMPT',
       'models':{'writer_reader':'gemini-2.5-flash','judge':'gemini-3.5-flash-lite','embedding':'gemini-embedding-001','dimensions':768},
       'conversation_workers':8,'arms_within_conversation':'sequential','provider_retries':2,
       'protocol_changes':['identical_dated_source_rendering_in_both_new_arms','mixed_date_fact_metadata_null',
                           'same_original_prompt_and_parser','group_order_first_source_position','group_history_previous10_ingested_turns',
                           'oversize_group_context_only_token_weighted_chunk_query_embedding'],
       'embedding_policy':'Only active group parse_messages search over2048 exact tokens: full turn-boundary blocks, oversize single-turn character bisection, original QUERY768 embeddings, token-weighted normalized mean; all other embeddings delegate unchanged',
       'empty_extraction_policy':'Explicit validated memory:[] is a valid empty SDK extraction, separately flagged for source-quality audit; malformed output, missing persisted ADDs and insufficient final bank remain failures',
       'downstream':'OriginalSDK20 top20 and original brief_answer_local_evidence_ids_v2; no new Reader or responseSchema',
       'limitations':['Both arms are new matched input-format controls; historical C is descriptive only.',
         'All grouping is offline within conversation; noncontiguous groups can expose future context.',
         'Group source dates are input scope, not verified supporting dates for each extracted fact.']}
    publish_json(run/'run_manifest.json',{**m,'protocol_sha256':digest(m)})
    return {'prepared':str(run),'model_calls':0,'protocol_sha256':digest(m)}

def manifest(run):
    m=read(Path(run)/'run_manifest.json')
    if m['schema']!=SCHEMA or m['protocol_sha256']!=digest({k:v for k,v in m.items() if k!='protocol_sha256'}):raise ValueError('Writer manifest changed')
    if sha(ROOT/'data/locomo10.json')!=SOURCE_HASH or m['dependencies']!=sparse.dependency_versions():raise ValueError('Writer dataset/dependencies changed')
    for p,h in m['code_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen writer source changed: '+p)
    for p,h in {**m['upstream_source_sha256'],**m['grouping_source_sha256']}.items():
        if sha(p)!=h:raise ValueError('Frozen upstream/grouping source changed: '+p)
    for v in m['inputs']:check_pointer(v['input'])
    return m

def write_arm(run,ci,arm,m):
    run=Path(run);folder=run/f'conv_{ci}'/arm;inp=verify(folder/'group_inputs.json');state_path=folder/'write_status.json'
    if state_path.exists() and read(state_path)['status']=='complete':
        state=read(state_path);check_pointer(state['bank']);return state
    state={'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,'pid':os.getpid(),
           'status':'running','started_utc':now(),'expected_groups':len(inp['envelopes']),'completed_groups':0,'facts':0}
    def status():state['updated_utc']=now();publish_json(state_path,state)
    if not Path(r'C:\Users\86139\Desktop\jev\experiments\live\credential_store.py').exists():raise RuntimeError('Existing DPAPI vault module is required; environment fallback is disabled')
    status();api=ModelAPI(folder/'native',retries=2,context={'architecture':SCHEMA,'conversation':ci,'method':arm,'phase':'write'})
    memory=None;records=[]
    try:
        memory=native.open_memory(folder/'native',api)
        from mem0.utils import spacy_models
        if spacy_models.get_nlp_full()is None or spacy_models.get_nlp_lemma()is None or memory.vector_store._get_bm25_encoder()is None:raise RuntimeError('Native NLP/BM25 unavailable')
        writer=Mem0GroupAdapter(memory,folder/'native/group_audit',current_date=CURRENT_DATE);uid=f'dated_regroup_c{ci}_{arm}'
        for envelope in inp['envelopes']:
            scoped=api.scoped(group_id=envelope['group_id']);memory.llm=native.ExtractionLLM(scoped);memory.embedding_model=Embeddings(scoped)
            record=writer.add_group(user_id=uid,**envelope);records.append(record)
            state.update(completed_groups=len(records),facts=sum(len(r['fact_records']) for r in records),last_group=envelope['group_id'],
                explicit_empty_groups=sum(r.get('explicit_empty_extraction',False) for r in records),
                chunk_embedding_groups=sum(r.get('group_context_embedding',{}).get('triggered',False) for r in records));status()
        facts=[f for r in records for f in r['fact_records']]
        for f in facts:
            p=memory.vector_store.client.retrieve(collection_name='facts',ids=[f['id']],with_vectors=True,with_payload=True)[0]
            if p.payload['data']!=f['text']:raise original.IncompleteSessionError('Exported native fact changed')
            f['embedding']=p.vector.get('') if isinstance(p.vector,dict) else p.vector
        sparse.vector_ok(facts)
        if len(facts)<20:raise original.IncompleteSessionError('Complete native bank requires20 facts')
        index=sparse.sidecar_audit(memory,facts,uid)
        bank={'schema':SCHEMA,'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,'complete':True,'user_id':uid,
            'facts':facts,'group_inputs':pointer(folder/'group_inputs.json'),'fact_support':'unverified_group_input_scope',
            'journal_files':[pointer(r['audit_path']) for r in records],'native_index_audit':index,
            'explicit_empty_group_ids':[r['group_id'] for r in records if r.get('explicit_empty_extraction')],
            'chunk_embedding_group_ids':[r['group_id'] for r in records if r.get('group_context_embedding',{}).get('triggered')],
            'native_dir':str((folder/'native').resolve()),'source_dataset_sha256':SOURCE_HASH}
        bp=folder/'bank.json';save_new(bp,bank);state.update(status='complete',bank=pointer(bp))
    except Exception as ex:state.update(status='failed',error={'type':type(ex).__name__,'message':api._redact(ex)[:1500]})
    finally:
        if memory is not None:native.close_memory(memory)
        state['finished_utc']=now();status()
    return state

def write_conv(run,ci):
    m=manifest(run)
    return [write_arm(run,ci,arm,m) for arm in ARMS]

def require_development_review(run,m):
    p=Path(run)/'development_review.json';review=verify(p)
    if review.get('protocol_sha256')!=m['protocol_sha256'] or review.get('status')!='passed':raise ValueError('Development source/date audit must pass before expansion')
    expected={str(Path(run).resolve()/f'conv_{ci}'/arm/'bank.json') for ci in (0,1) for arm in ARMS}
    if {b['path'] for b in review.get('banks',[])}!=expected:raise ValueError('Development review must bind allfour banks')
    for b in review['banks']:
        bank=check_pointer(b)
        if not bank.get('complete'):raise ValueError('Development bank is incomplete')

@contextlib.contextmanager
def writer_process_lock(run):
    """One CLI owner; Windows releases the byte lock if the owner exits."""
    import msvcrt
    f=(Path(run)/'writer_execution.lock').open('a+b');f.seek(0,2)
    if f.tell()==0:f.write(b'0');f.flush()
    f.seek(0)
    try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:f.close();raise RuntimeError('Another writer controller owns this run')
    try:yield
    finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--grouping-dir',type=Path,required=True)
    a=sub.add_parser('write');a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--conversations',default='0,1')
    a=sub.add_parser('check');a.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':result=prepare(args.run_dir,args.grouping_dir)
    elif args.command=='check':result={'protocol_sha256':manifest(args.run_dir)['protocol_sha256'],'model_calls':0}
    else:
        m=manifest(args.run_dir);convs=[int(x) for x in args.conversations.split(',')]
        if not convs or len(set(convs))!=len(convs) or any(c not in range(10) for c in convs):raise ValueError('Invalid conversation list')
        if any(c>1 for c in convs):require_development_review(args.run_dir,m)
        with writer_process_lock(args.run_dir):
            with ProcessPoolExecutor(max_workers=min(8,len(convs))) as pool:
                result=[f.result() for f in as_completed([pool.submit(write_conv,args.run_dir,ci) for ci in convs])]
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
