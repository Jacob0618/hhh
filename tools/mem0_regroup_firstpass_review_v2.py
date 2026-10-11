"""Offline first-pass seal, uniform20 native-index review and metadata-only c3 export.

No model, Mem0 or Qdrant client is created. Original stores/statuses are immutable.
This explicitly versioned post-failure policy never admits the partial c8 bank.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import socket
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_experiment_health import shared_read
ARMS=('dated_unmerged385','dated_merged')
REMAINING=ROOT/'runs/mem0_global_regroup_remaining_v1'
POLICY=ROOT/'research/mem0_regroup_failure_followup_policy_2026-09-29.md'
CORE=ROOT/'research/mem0_regroup_firstpass_complete_facts_core_v2_2026-09-29.json'
DEV_AUDITS=[ROOT/'research/mem0_regroup_development_three_bank_audit_2026-09-28.json',ROOT/'research/mem0_regroup_development_fourth_bank_audit_2026-09-29.json']
PARTIAL_AUDIT=ROOT/'research/mem0_regroup_conv8_merged_json_failure_audit_2026-09-29.json'
FIRSTPASS_SEAL=ROOT/'research/mem0_regroup_remaining_firstpass_seal_2026-09-29.json'
ORPHANS={ARMS[0]:'24c2aec4-2c78-4cfe-bdda-6fedc014d5da',ARMS[1]:'591694fe-7c20-4145-b363-ce80c983a2d4'}
ERROR='Native entity linking omitted extracted entity-bearing facts'
HASHES={}

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    path=Path(path).resolve();value=hashlib.sha256(shared_read(path)).hexdigest()
    if str(path) in HASHES and HASHES[str(path)]!=value:raise ValueError('Source changed: '+str(path))
    HASHES[str(path)]=value;return value
def pointer(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def read(path):sha(path);return json.loads(shared_read(path).decode('utf-8-sig'))
def bound(p):
    if pointer(p['path'])!=p:raise ValueError('Artifact binding changed')
    return read(p['path'])
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verify(path):
    v=read(path)
    if v['content_sha256']!=digest({k:x for k,x in v.items() if k!='content_sha256'}):raise ValueError('Seal content changed')
    return v
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value={**value,'content_sha256':digest(value)}
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,ensure_ascii=False,indent=2);h.write('\n')
    return pointer(path)
def origin(ci,arm):
    if ci>=2:return REMAINING
    return ROOT/('runs/mem0_global_regroup_bm25_recovery_v2' if (ci,arm)==(1,ARMS[1]) else 'runs/mem0_global_regroup_recovery_v1')
def folder(ci,arm):return origin(ci,arm)/f'conv_{ci}'/arm
def native_inventory(native):
    native=Path(native);out={}
    for p in list((native/'qdrant').rglob('*'))+[native/'history.sqlite']:
        if p.is_symlink():raise ValueError('Symlink forbidden')
        if p.is_file() and p.name!='.lock':out[p.relative_to(native).as_posix()]=sha(p)
    for p in native.rglob('*'):
        if p.is_file() and p.name.endswith(('-wal','-journal')) and p.stat().st_size:raise ValueError('Uncheckpointed native sidecar')
    return out
def rows(db,table):
    before=sha(db);c=sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    try:
        c.execute('PRAGMA query_only=ON');c.execute('BEGIN')
        if c.execute('PRAGMA query_only').fetchone()[0]!=1 or c.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity/read mode')
        answer=[dict(x) for x in c.execute('SELECT * FROM '+table)];c.rollback()
    finally:c.close()
    if sha(db)!=before:raise ValueError('SQLite source changed')
    return answer
def points(db):return {str(p.id):p for p in (pickle.loads(r['point']) for r in rows(db,'points'))}
def dense(v):return isinstance(v,list) and len(v)==768 and all(type(x) in (float,int) and math.isfinite(x) for x in v) and sum(x*x for x in v)>0
def sparse(v,allow_empty=False):
    return v is not None and len(v.indices)==len(v.values) and (allow_empty or len(v.indices)>0) and len(set(v.indices))==len(v.indices) and all(type(i)is int and i>=0 for i in v.indices) and all(math.isfinite(x) and x>=0 for x in v.values)
def norm(s):return ' '.join(s.lower().split())
def cosine(a,b):return sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
def require_terminal(states):
    if len(states)!=16 or {(s['conversation'],s['arm']) for s in states}!={(c,a) for c in range(2,10) for a in ARMS}:raise ValueError('Exact remaining16 required')
    if any(s['status'] not in ('complete','failed') for s in states):raise ValueError('Nonterminal bank: no SQLite/export permitted')
def terminal():
    states=[read(folder(ci,arm)/'write_status.json') for ci in range(2,10) for arm in ARMS];require_terminal(states)
    for s in states:
        f=folder(s['conversation'],s['arm']);o=verify(f/'write_outcome.json')
        if o['status']!=s['status'] or o['write_status']!=pointer(f/'write_status.json'):raise ValueError('Terminal outcome differs')
    return states
def load_extractor():
    p=ROOT/'upstream/mem0/mem0/utils/entity_extraction.py'
    if sha(p)!=sha(ROOT/'.venv/Lib/site-packages/mem0/utils/entity_extraction.py'):raise ValueError('Entity extractor drift')
    spec=importlib.util.spec_from_file_location('_readonly_firstpass_entities',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    import spacy
    if not spacy.util.is_package('en_core_web_sm'):raise ValueError('Local NLP model missing; download forbidden')
    return m,spacy.load('en_core_web_sm')
def validate_records(records,facts,history):
    byid={f['id']:f for f in records}
    if len(byid)!=len(records) or set(byid)!=set(facts):raise ValueError('Fact persistence missing/duplicate/extra')
    if Counter(h['memory_id'] for h in history)!=Counter(byid.keys()):raise ValueError('History ADD count differs')
    for h in history:
        if h['event']!='ADD' or h['is_deleted'] or h['old_memory']is not None or h['new_memory']!=byid[h['memory_id']]['text']:raise ValueError('History content differs')
    for mid,p in facts.items():
        f=byid[mid];payload=p.payload
        if payload['data']!=f['text'] or payload['hash']!=hashlib.md5(f['text'].encode()).hexdigest():raise ValueError('Fact text/hash differs')
        if not dense(p.vector.get('')) or not sparse(p.vector.get('bm25')):raise ValueError('Required fact vectors missing/invalid')
        if payload['jev_mem_group_id']!=f['input_scope']['group_id'] or payload['jev_mem_observation_date']!=f['observation_date'] or payload['jev_mem_source_observation_dates']!=f['source_observation_dates'] or payload['jev_mem_source_turn_ids']!=f['input_scope']['source_turn_ids']:raise ValueError('Fact source/date binding differs')
    return byid
def inspect(ci,arm,extractor,nlp):
    f=folder(ci,arm);native=f/'native';before=native_inventory(native)
    js=sorted([(read(p),p) for p in (native/'group_audit').rglob('*.json')],key=lambda x:x[0]['sequence'])
    success=[(j,p) for j,p in js if j['status']=='success'];records=[x for j,p in success for x in j['fact_records']]
    facts=points(native/'qdrant/collection/facts/storage.sqlite');entities=points(native/'qdrant/collection/facts_entities/storage.sqlite');history=rows(native/'history.sqlite','history')
    byid=validate_records(records,facts,history);ids=list(byid)
    extracted={mid:extractor._extract_entities_from_doc(doc) for mid,doc in zip(ids,nlp.pipe([byid[i]['text'] for i in ids],batch_size=32))}
    linked=set();texts=defaultdict(list);edges=0;missing_sparse=[];empty_sparse=[]
    for eid,p in entities.items():
        vectors=p.vector if isinstance(p.vector,dict) else {'':p.vector}
        if not dense(vectors.get('')):raise ValueError('Entity dense vector invalid')
        ls=p.payload['linked_memory_ids']
        if not ls or len(set(ls))!=len(ls) or not set(ls)<=set(facts):raise ValueError('Entity links empty/duplicate/foreign')
        if p.payload['user_id']!=f'dated_regroup_c{ci}_{arm}':raise ValueError('Foreign entity scope')
        linked.update(ls);edges+=len(ls);texts[norm(p.payload['data'])].append(eid)
        sv=vectors.get('bm25')
        if sv is None:missing_sparse.append(eid)
        elif not sparse(sv,allow_empty=True):raise ValueError('Entity sparse malformed')
        elif not sv.indices:empty_sparse.append(eid)
    expected={i for i,e in extracted.items() if e};orphans=sorted(expected-linked)
    state=read(f/'write_status.json');complete=len(success)==state['expected_groups']
    if state['facts']!=len(facts) or state['completed_groups']!=len(success):raise ValueError('Status/persisted prefix differs')
    expected_orphans=[ORPHANS[arm]] if ci==3 else []
    if orphans!=expected_orphans:raise ValueError(f'New orphan pattern: c{ci}/{arm} {orphans}')
    if native_inventory(native)!=before:raise ValueError('Native inventory changed')
    result={'conversation':ci,'arm':arm,'original_write_status':state['status'],'complete_fact_groups':complete,'groups':len(success),'expected_groups':state['expected_groups'],
        'facts':len(facts),'history_adds':len(history),'entities':len(entities),'entity_link_edges':edges,'facts_linked':len(linked),'entity_bearing_facts':len(expected),
        'orphan_fact_ids':orphans,'entity_sparse_missing':missing_sparse,'entity_sparse_empty':empty_sparse,'duplicate_normalized_entities':{k:v for k,v in texts.items() if len(v)>1},
        'bank':pointer(f/'bank.json') if (f/'bank.json').exists() else None,'write_status':pointer(f/'write_status.json'),
        'outcome':pointer(f/('write_outcome.json' if ci>=2 else 'recovery_outcome.json')),'source_manifest':pointer(origin(ci,arm)/'run_manifest.json'),
        'inputs':pointer(f/'group_inputs.json'),'native_dir':str(native),'native_inventory':before,
        'facts_dense_sparse_and_history_valid':True,'existing_entity_dense_and_links_valid':True,'native_index_policy':'original_unrepaired'}
    return result,records,facts,entities,js,extracted
def overwrite_proof(arm,records,entities,journals,extracted):
    target=ORPHANS[arm];byid={x['id']:x for x in records};j,jp=next((j,p) for j,p in journals if any(f['id']==target for f in j['fact_records']))
    seq={f['id']:g['sequence'] for g,p in journals for f in g['fact_records']};at=j['sequence']
    ge={}
    for f in j['fact_records']:
        for typ,text in extracted[f['id']]:ge.setdefault(norm(text),[typ,text,set()])[2].add(f['id'])
    if extracted[target]!=[('TOPIC','video games')]:raise ValueError('Orphan root cause entity differs')
    n=folder(3,arm)/'native';ledger=[json.loads(x) for x in shared_read(n/'api_calls.jsonl').decode().splitlines()];sha(n/'api_calls.jsonl')
    vectors={};raw_refs=[];texts_seen=[]
    for e in ledger:
        if e.get('context',{}).get('group_id')!=j['group_id'] or e.get('stage')!='mem0_embedding_add' or e.get('operation')!='batchEmbedContents':continue
        rp=n/'api_cache'/(e['request_hash']+'.json');raw=read(rp);texts=[x['content']['parts'][0]['text'] for x in raw['request']['body']['requests']]
        if texts[0] not in [v[1] for v in ge.values()]:continue
        vs=[v['values'] for v in raw['response']['embeddings']]
        if len(vs)!=len(texts) or not all(dense(v) for v in vs) or e.get('status')!=200:raise ValueError('Entity embedding source differs')
        vectors.update(zip(texts,vs));texts_seen.extend(texts);raw_refs.append(pointer(rp))
    if texts_seen!=[v[1] for v in ge.values()]:raise ValueError('Original entity batch differs from local entity extraction')
    prior={eid:copy.deepcopy(p.payload) for eid,p in entities.items() if any(seq[mid]<at for mid in p.payload['linked_memory_ids'])}
    for p in prior.values():p['linked_memory_ids']=[mid for mid in p['linked_memory_ids'] if seq[mid]<at]
    exact={norm(p['data']):eid for eid,p in prior.items()};matches=[]
    for text in ('video games','video game'):
        top=sorted([(cosine(vectors[text],entities[eid].vector['']),eid) for eid in prior],reverse=True)[:3]
        eid=exact.get(norm(text)) or (top[0][1] if top[0][0]>=.95 else None)
        matches.append({'text':text,'exact_id':exact.get(norm(text)),'selected_id':eid,'top3':[{'id':i,'text':prior[i]['data'],'cosine':v} for v,i in top],'memory_ids':sorted(ge[norm(text)][2])})
    canonical=matches[0]['selected_id']
    if not canonical or canonical!=matches[1]['selected_id']:raise ValueError('Not the reviewed same-canonical overwrite')
    base=prior[canonical]['linked_memory_ids'];first=sorted(set(base)|ge['video games'][2]);second=sorted(set(base)|ge['video game'][2])
    observed=sorted(mid for mid in entities[canonical].payload['linked_memory_ids'] if seq[mid]<=at)
    if target not in first or target in second or observed!=second:raise ValueError('Stale overwrite does not reproduce observed payload')
    return {'verified':True,'mechanism':'same_canonical_stale_payload_overwrite','fact_id':target,'fact_text':byid[target]['text'],'group_id':j['group_id'],
        'journal':pointer(jp),'entity_embedding_raws':raw_refs,'matches':matches,'canonical_entity_id':canonical,'before_links':base,'first_links':first,'second_links':second,
        'observed_links_through_group':observed,'warnings':j['sdk_warnings'],'qualification':'source/cached-vector/final-state reconstruction; no historical update trace'}
def audit(out,process_snapshot):
    states=terminal();ps=read(process_snapshot)
    if ps['processes'] or any(s['finished_utc']>ps['captured_utc'] for s in states):raise ValueError('Fresh completed-controller snapshot required')
    seal=verify(FIRSTPASS_SEAL)
    if seal['counts']!={'banks':16,'complete':13,'failed':3,'completed_groups':614,'facts_including_partial':4118}:raise ValueError('First-pass seal scope differs')
    for ref in seal['file_inventory']+seal['original_code']:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('First-pass source seal changed')
    core=read(CORE)
    if len(core['banks'])!=15 or not core['structural_integrity_passed']:raise ValueError('Complete-fact core missing')
    audits=[CORE,*DEV_AUDITS,PARTIAL_AUDIT]
    for p in audits:
        a=read(p)
        for path,value in a.get('audited_file_hashes',a.get('source_file_hashes',{})).items():
            if sha(path)!=value:raise ValueError('Prior source audit binding changed')
    extractor,nlp=load_extractor();banks=[];proofs={};raw_inventory=[];cost=[]
    for ci in range(10):
        for arm in ARMS:
            row,records,facts,entities,journals,extracted=inspect(ci,arm,extractor,nlp)
            if ci==3:proofs[arm]=overwrite_proof(arm,records,entities,journals,extracted)
            banks.append(row);n=folder(ci,arm)/'native';lp=n/'api_calls.jsonl';events=[json.loads(x) for x in shared_read(lp).decode().splitlines()];pointer(lp)
            by_model=defaultdict(lambda:{'physical_attempts':0,'http200':0,'failed_attempts':0,'cache_consumers':0,'unknown_usage_attempts':0,'reported_usage':Counter()})
            for e in events:
                v=by_model[str(e.get('model'))]
                if e.get('cache_hit'):v['cache_consumers']+=1;continue
                v['physical_attempts']+=1;v['http200']+=e.get('status')==200;v['failed_attempts']+=e.get('status')!=200;v['unknown_usage_attempts']+=bool(e.get('usage_unknown'))
                if e.get('status')==200:v['reported_usage'].update({k:x for k,x in e.get('usage',{}).items() if type(x)in(int,float)})
            cost.append({'conversation':ci,'arm':arm,'ledger':pointer(lp),'models':dict(by_model)})
            for p in (n/'api_cache').glob('*.json'):raw_inventory.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    from jev_mem.local_bm25_context_v2 import compare_persisted_fact_vectors
    m=read(REMAINING/'run_manifest.json');resource=m['remaining_expansion']['resources']
    remaining_banks=[b for b in banks if b['conversation']>=2]
    bm25=compare_persisted_fact_vectors([Path(b['native_dir']) for b in remaining_banks],resource['snapshot'],resource['files_sha256'])
    if bm25['facts_compared']!=bm25['exact_matches'] or bm25['facts_compared']!=sum(b['facts'] for b in remaining_banks):raise ValueError('Offline BM25 mismatch')
    bm25['scope']='remaining16 including explicitly ineligible partial c8; four development source/index audits reused with source hashes revalidated'
    failed=[b for b in banks if b['original_write_status']=='failed']
    if {(b['conversation'],b['arm']) for b in failed}!={(3,a) for a in ARMS}|{(8,ARMS[1])}:raise ValueError('Unexpected failure set')
    for path,value in list(HASHES.items()):
        if sha(path)!=value:raise ValueError('Source changed')
    result={'schema':'mem0-regroup-firstpass-uniform20-review-v2','status':'reviewed_with_three_original_failures','created_utc':now(),
        'policy':pointer(POLICY),'firstpass_seal':pointer(FIRSTPASS_SEAL),'process_snapshot':pointer(process_snapshot),'source_audits':[pointer(p) for p in audits],
        'banks':banks,'allbanks':banks,'counts':{'banks':20,'remaining':16,'remaining_complete':13,'original_complete':17,'original_failed':3,'complete_fact_banks':19,
            'facts_including_partial':sum(b['facts'] for b in banks),'orphans':sum(len(b['orphan_fact_ids']) for b in banks)},
        'c3_overwrite_proofs':proofs,'offline_bm25_equivalence':bm25,'raw_cache_inventory':raw_inventory,'cost_ledger_summary':cost,
        'cost_limitations':'Physical attempts/cache consumers and reported token usage only; unknown usage retained. No price assumptions or unsupported USD totals.',
        'checks':{'all16_terminal':True,'uniform20_index_and_orphan_review':True,'nineteen_complete_fact_chains_verified':True,'required_fact_vectors_valid':True,'existing_entity_dense_and_links_valid':True,
            'all_native_entity_bearing_facts_linked':False,'all20_complete_fact_banks':False,'original_sources_unchanged':True},
        'limitations':['Original unrepaired native entity behavior retained; partial individual entity-edge completeness is not claimed.','c8 merged remains13/29 and is excluded from complete-bank/export/QA admission.','Post-failure export policy is separate from strict v1; original failed statuses persist.','Full semantic/source-quality review remains separately bound by root.'],
        'source_file_hashes':HASHES,'audit_tool':pointer(Path(__file__)),'model_api_calls':0,'qa_gold_used':False,'native_clients':0,'source_mutations':0,'qa_gate_passed':False}
    return save(out,result)
def export_eligible(row,proof):
    return (row.get('conversation')==3 and row.get('arm')in ARMS and row.get('original_write_status')=='failed' and row.get('complete_fact_groups')is True
        and row.get('groups')==row.get('expected_groups') and row.get('facts_dense_sparse_and_history_valid')is True and row.get('existing_entity_dense_and_links_valid')is True
        and row.get('orphan_fact_ids')==[ORPHANS[row['arm']]] and proof.get('verified')is True and proof.get('mechanism')=='same_canonical_stale_payload_overwrite' and proof.get('fact_id')==ORPHANS[row['arm']])
def export(audit_path,destination):
    terminal();review=verify(audit_path);dest=Path(destination).resolve()
    if dest.parent!=(ROOT/'runs').resolve() or dest.exists():raise ValueError('Fresh direct runs namespace required')
    if review['schema']!='mem0-regroup-firstpass-uniform20-review-v2' or not review['checks']['all16_terminal']:raise ValueError('Wrong review')
    if pointer(review['policy']['path'])!=review['policy'] or pointer(review['audit_tool']['path'])!=review['audit_tool']:raise ValueError('Review policy/tool changed')
    for path,value in review['source_file_hashes'].items():
        if sha(path)!=value:raise ValueError('Reviewed source changed before export')
    targets=[b for b in review['banks'] if b['conversation']==3]
    if len(targets)!=2 or any(not export_eligible(b,review['c3_overwrite_proofs'][b['arm']]) for b in targets):raise ValueError('Only reviewed complete c3 twin banks eligible')
    dest.mkdir();save(dest/'export_intent.json',{'schema':'mem0-regroup-native-metadata-export-v2','review':pointer(audit_path),'policy':review['policy'],'model_api_calls':0,'native_index_policy':'original_unrepaired'})
    exported=[]
    for b in targets:
        arm=b['arm'];source=folder(3,arm);n=Path(b['native_dir']);before=native_inventory(n)
        if before!=b['native_inventory']:raise ValueError('Native source changed')
        js=sorted([(read(p),p) for p in (n/'group_audit').rglob('*.json')],key=lambda x:x[0]['sequence']);ps=points(n/'qdrant/collection/facts/storage.sqlite')
        facts=[dict(f,embedding=ps[f['id']].vector['']) for j,p in js for f in j['fact_records']]
        validate_records(facts,ps,rows(n/'history.sqlite','history'))
        bank={'schema':'mem0-dated-regroup-reviewed-bank-v2','protocol_sha256':read(origin(3,arm)/'run_manifest.json')['protocol_sha256'],
            'conversation':3,'arm':arm,'complete':True,'complete_scope':'all_source_groups_and_generated_facts; original_write_failed',
            'user_id':f'dated_regroup_c3_{arm}','facts':facts,'group_inputs':b['inputs'],'journal_files':[pointer(p) for j,p in js],
            'fact_support':'unverified_group_input_scope','native_dir':str(n),'source_dataset_sha256':read(origin(3,arm)/'run_manifest.json')['dataset_sha256'],
            'explicit_empty_group_ids':[j['group_id'] for j,p in js if j.get('explicit_empty_extraction')],
            'chunk_embedding_group_ids':[j['group_id'] for j,p in js if j.get('group_context_embedding',{}).get('triggered')],
            'source_kind':'reviewed_export','native_index_policy':'original_unrepaired','original_write_status':b['write_status'],'original_outcome':b['outcome'],
            'native_index_audit':{k:b[k] for k in ('facts','history_adds','entities','entity_link_edges','facts_linked','entity_bearing_facts','orphan_fact_ids','entity_sparse_missing','entity_sparse_empty')},
            'source_manifest':b['source_manifest'],'source_inventory':before,'source_review':pointer(audit_path)}
        base=dest/'conv_3'/arm;bp=save(base/'bank.json',bank)
        after=native_inventory(n)
        if before!=after or read(source/'write_status.json')['status']!='failed':raise ValueError('Original source changed during metadata export')
        record={'schema':'mem0-regroup-reviewed-export-v2','source_kind':'reviewed_export','conversation':3,'arm':arm,
            'original_write_status':'failed','reviewed_export_status':'complete','created_utc':now(),'bank':bp,
            'original_write_status_ref':b['write_status'],'original_outcome':b['outcome'],'source_manifest':b['source_manifest'],
            'native_dir':str(n),'native_inventory':before,'native_inventory_before':before,'native_inventory_after':after,
            'native_index_policy':'original_unrepaired','source_review':pointer(audit_path),'policy':review['policy'],'orphan_fact_ids':b['orphan_fact_ids'],
            'facts':len(facts),'groups':b['groups'],'native_source_opened_by_client':False,'native_bytes_copied':0,'model_api_calls':0,'source_mutations':0,'qa_gate_passed':False,
            'export_tool':pointer(Path(__file__))}
        ep=save(base/'reviewed_export.json',record);exported.append({'conversation':3,'arm':arm,'bank':bp,'reviewed_export':ep,'original_write_status':b['write_status'],'original_outcome':b['outcome'],'source_manifest':b['source_manifest'],'native_inventory':before,'native_index_policy':'original_unrepaired'})
    for path,value in review['source_file_hashes'].items():
        if sha(path)!=value:raise ValueError('Original reviewed source changed')
    return save(dest/'export_outcome.json',{'schema':'mem0-regroup-native-metadata-export-v2','status':'reviewed_metadata_exports_complete','review':pointer(audit_path),'exports':exported,
        'original_failed_banks_preserved':2,'partial_c8_exported':False,'native_bytes_copied':0,'source_hashes_before_after_equal':True,'model_api_calls':0,'qa_gate_passed':False})
def check_export_bank(binding_or_dir):
    """Read-only admission of one sealed metadata export; no SQLite/client/API.

    Accept an exact path/SHA pointer to reviewed_export.json, its path, or its
    containing directory. Return the common byte-copy adapter bank/native/review
    fields plus the original failure and export provenance.
    """
    if isinstance(binding_or_dir,dict):
        path=Path(binding_or_dir['path'])
        if pointer(path)!=binding_or_dir:raise ValueError('Reviewed export pointer changed')
    else:
        path=Path(binding_or_dir)
        if path.is_dir():path=path/'reviewed_export.json'
    record=verify(path)
    if record.get('schema')!='mem0-regroup-reviewed-export-v2' or record.get('source_kind')!='reviewed_export' or record.get('original_write_status')!='failed' or record.get('reviewed_export_status')!='complete' or record.get('native_index_policy')!='original_unrepaired':raise ValueError('Not an admitted original-failed metadata export')
    ci,arm=record['conversation'],record['arm']
    if ci!=3 or arm not in ARMS:raise ValueError('Only separately reviewed c3 twin banks')
    review=verify(record['source_review']['path'])
    if pointer(record['source_review']['path'])!=record['source_review']:raise ValueError('Source review hash changed')
    row=next(b for b in review['banks'] if (b['conversation'],b['arm'])==(ci,arm))
    if not export_eligible(row,review['c3_overwrite_proofs'][arm]):raise ValueError('Source audit ineligible')
    src=folder(ci,arm);native=src/'native'
    for key,expected in (('original_write_status_ref',src/'write_status.json'),('original_outcome',src/'write_outcome.json'),('source_manifest',origin(ci,arm)/'run_manifest.json')):
        if record[key]!=pointer(expected):raise ValueError('Original lineage differs')
    state=bound(record['original_write_status_ref']);outcome=verify(record['original_outcome']['path'])
    if state['status']!='failed' or outcome['status']!='failed' or state['error']['message']!=ERROR or state['completed_groups']!=state['expected_groups']:raise ValueError('Original failure scope differs')
    if outcome['write_status']!=record['original_write_status_ref']:raise ValueError('Outcome status pointer differs')
    bank=verify(record['bank']['path'])
    if record['bank']!=pointer(path.parent/'bank.json') or bank.get('schema')!='mem0-dated-regroup-reviewed-bank-v2' or not bank.get('complete'):raise ValueError('Exported bank differs')
    if (bank['conversation'],bank['arm'])!=(ci,arm) or Path(bank['native_dir']).resolve()!=native.resolve() or Path(record['native_dir']).resolve()!=native.resolve():raise ValueError('Export native lineage differs')
    manifest=bound(record['source_manifest'])
    inp=next(i['input'] for i in manifest['inputs'] if (i['conversation'],i['arm'])==(ci,arm))
    if bank['group_inputs']!=inp or bank['protocol_sha256']!=manifest['protocol_sha256'] or bank['source_dataset_sha256']!=manifest['dataset_sha256']:raise ValueError('Export input identity differs')
    inputs=bound(inp)
    js=sorted([bound(p) for p in bank['journal_files']],key=lambda j:j['sequence'])
    if len(js)!=len(inputs['envelopes']) or len(js)!=state['expected_groups'] or any(j['status']!='success' for j in js):raise ValueError('Incomplete exported group chain')
    if [f for j in js for f in j['fact_records']]!=[{k:v for k,v in f.items() if k!='embedding'} for f in bank['facts']]:raise ValueError('Exported facts differ from original journals')
    if len(bank['facts'])!=row['facts'] or not all(dense(f['embedding']) for f in bank['facts']):raise ValueError('Exported vectors/facts invalid')
    inv=native_inventory(native)
    if not inv==record['native_inventory']==record['native_inventory_before']==record['native_inventory_after']==bank['source_inventory']==row['native_inventory']:raise ValueError('Original native inventory changed')
    if bank['native_index_audit']['orphan_fact_ids']!=[ORPHANS[arm]] or record['orphan_fact_ids']!=[ORPHANS[arm]]:raise ValueError('Native anomaly omitted')
    aggregate=verify(path.parent.parent.parent/'export_outcome.json')
    if aggregate.get('status')!='reviewed_metadata_exports_complete' or not any(x['reviewed_export']==pointer(path) and x['bank']==record['bank'] for x in aggregate['exports']):raise ValueError('Export namespace has no complete aggregate outcome')
    if pointer(record['policy']['path'])!=record['policy'] or pointer(record['export_tool']['path'])!=record['export_tool']:raise ValueError('Export policy/tool drift')
    for source_path,value in review['source_file_hashes'].items():
        if Path(source_path).resolve().is_relative_to(src.resolve()) and sha(source_path)!=value:raise ValueError('Audited original source changed')
    return {'bank':bank,'native':native,'review':{'bank':record['bank'],'native_inventory':inv,'source_kind':'reviewed_export','original_write_status':record['original_write_status_ref'],
        'original_outcome':record['original_outcome'],'reviewed_export':pointer(path),'source_manifest':record['source_manifest'],'source_review':record['source_review']},
        'export':record,'source_kind':'reviewed_export'}
def main():
    sys.dont_write_bytecode=True
    def no_network(*args,**kwargs):raise RuntimeError('Network forbidden in read-only review/export')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=no_network
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('audit');p.add_argument('--output',type=Path,required=True);p.add_argument('--process-snapshot',type=Path,required=True)
    p=sub.add_parser('export');p.add_argument('--audit',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=ap.parse_args();result=audit(a.output,a.process_snapshot) if a.mode=='audit' else export(a.audit,a.destination)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
