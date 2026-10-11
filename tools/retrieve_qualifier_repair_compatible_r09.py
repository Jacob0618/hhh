"""Technical-only R09 cache-only retrieval on equivalent locked copy sources."""
from __future__ import annotations
import argparse,copy,json,os,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import materialize_qualifier_repair_snapshot_r09 as parent
from tools import materialize_qualifier_repair_r09 as original
from tools.recover_gemini38_session_io_v1 import pid_alive
from jev_mem.local_bm25_context_v2 import forbid_network
old=parent.old
require,pointer,bound,sealed_read,save_new,sha,digest,now=parent.require,parent.pointer,parent.bound,parent.sealed_read,parent.save_new,parent.sha,parent.digest,parent.now
SCHEMA=original.SCHEMA;SOURCE_SCHEMA=original.SOURCE_SCHEMA
MANIFEST_SCHEMA='qualifier-repair-r09-retrieval-compatibility-manifest-v1'
TECH_SHA='ce3c86a30b531dee4c4085f17818acf6c367eea9f2f24b1ddcd03aeafcbae8c2'
TECH=ROOT/'research/jev_autoresearch_2026-10-04/R09_retrieval_compatibility_protocol_v1.json'
APPROVED=ROOT/'runs/qualifier_repair_r09_native_snapshot_v1'
EXCEPTIONS={(ci,'jev') for ci in (0,1,3,9)}
CODE=tuple(dict.fromkeys((*parent.CODE,'tools/retrieve_qualifier_repair_compatible_r09.py',
    'tools/run_qualifier_repair_retrieval_qa_r09.py','tools/summarize_qualifier_repair_retrieval_r09.py')))
retrieval_folder=original.retrieval_folder
body=parent.body

def tech():
    require(sha(TECH)==TECH_SHA,'Retrieval compatibility protocol changed')
    t=sealed_read(TECH)
    for ref in t['input_refs'].values():require(pointer(ref['path'])==ref,'Retrieval protocol input bytes changed')
    return t

def selection_body(value):
    """Only frozen operational timestamp and the enclosing content seal excluded."""
    v=copy.deepcopy(body(value))
    for event in v['cache_events']:event.pop('timestamp_utc',None)
    return v

def assembly_lib(run,m):
    """Bind the legacy assembler's one native-gate alias without forging a gate."""
    alias=(Path(run)/'materialization/index_admission.json').resolve()
    def assembly_pointer(path):
        if Path(path).resolve()==alias:
            ref=m['approved_effective20_gate']
            require(pointer(ref['path'])==ref,'Approved assembler native gate changed')
            return ref
        return pointer(path)
    return original._lib({'bank_folder':retrieval_folder,'pointer':assembly_pointer})

def validate_cache_events(selection,job,entries,arm):
    """Replay only the frozen SDK's QUERY calls, never a store search or network."""
    from mem0.memory.main import Memory,extract_entities
    require(selection['schema']==SCHEMA and selection['status']=='complete'
        and selection['conversation']==job['conversation'] and selection['method']==arm
        and selection['query_id']==job['query_id'] and selection['model_calls']==0,
        'Selection fixed query/arm identity differs')
    api=old.CacheOnlyEmbeddings({e['request_hash']:e for e in entries}).scoped(
        conversation=job['conversation'],method=arm,query_id=job['query_id'],phase='retrieval')
    api.embed([job['question']],query=True,stage='mem0_embedding_search')
    seen=set();texts=[]
    for _,text in extract_entities(job['question'])[:8]:
        key=Memory._normalize_entity_text(text)
        if key and key not in seen:seen.add(key);texts.append(text)
    if texts:api.embed(texts,query=True,stage='mem0_embedding_search')
    actual=selection['cache_events'];expected=api.events
    require(len(actual)==len(expected)>0,'Exact frozen QUERY cache event cohort differs')
    for a,e in zip(actual,expected):
        require(isinstance(a.get('timestamp_utc'),str)
            and {k:v for k,v in a.items() if k!='timestamp_utc'}==
                {k:v for k,v in e.items() if k!='timestamp_utc'},
            'Exact frozen QUERY raw/cache context differs')
    return len(expected)

def approved_input_refs():
    """Byte-bind all closed stage proofs; native stores have separate inventories."""
    folder=APPROVED/'materialization'
    return [pointer(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and 'native' not in p.relative_to(folder).parts]

def validate_approved_inputs(refs):
    require(refs==approved_input_refs(),'Approved native proof byte inventory changed')

def copy_source(ci,arm,bank,info):
    """Return a pinned physical source, retaining the effective logical bank."""
    source=Path(bank['native_dir']).resolve();inv=old.inventory(source)
    require(inv==bank['native_inventory'],'Effective native inventory changed')
    if (ci,arm) not in EXCEPTIONS:
        require((source/'qdrant/.lock').is_file() and not (source/'qdrant/.lock').is_symlink(),
            'Nonexception effective source OS lock missing')
        return {'native_dir':str(source),'inventory':inv,'equivalence':None,'origin':'approved_effective_bank'}
    n0=Path(info['native_dir']).resolve();snapshot=parent.support.snapshot_body(bound(info['snapshot'],sealed=True))
    original_bank=bound(info['bank'],sealed=True)
    require(bank['updates']==[] and bank['user_id']==info['user_id']==original_bank['user_id']
        and bank['facts']==original_bank['facts'] and inv==info['inventory']==old.inventory(n0)
        and old.snapshot(source)==snapshot==old.snapshot(n0)
        and not (source/'qdrant/.lock').exists() and (n0/'qdrant/.lock').is_file()
        and not (n0/'qdrant/.lock').is_symlink(),'Zero-UPDATE original/effective full equivalence differs')
    return {'native_dir':str(n0),'inventory':info['inventory'],'origin':'equivalent_original_N0_locked_source',
        'equivalence':{'zero_updates':True,'effective_native_dir':str(source),'original_bank':info['bank'],
            'original_snapshot':info['snapshot'],'snapshot_body_sha256':digest(snapshot),
            'effective_bank_facts_sha256':digest(bank['facts']),'full_file_sha_inventory_equal':True,
            'full_logical_snapshot_equal':True,'effective_native_modified':False}}

def historical_retrieval(m0,gate):
    o=sealed_read(APPROVED/'retrieval_outcome.json');intent=sealed_read(APPROVED/'retrieval_execution_intent.json')
    failures=[r for r in o['banks'] if r['status']=='failed'];success=[r for r in o['banks'] if r['status']=='complete']
    require(o['status']=='failed' and o['model_calls']==0 and not pid_alive(intent['pid'])
        and [r['conversation'] for r in failures]==[0,1,3,9]
        and all(r['error_type']=='NativeGateError' for r in failures) and len(success)==12,
        'Original partial retrieval failure population differs')
    refs=[pointer(APPROVED/'retrieval_outcome.json'),pointer(APPROVED/'retrieval_execution_intent.json')]
    successful={};inventories={}
    jobs=old.old_jobs(bound(m0['protocol']['frozen_source']))
    for ci in (0,1,3,9):
        for arm in original.NEW_ARMS:
            f=retrieval_folder(APPROVED,ci,arm)
            require(not (f/'retrieval_outcome.json').exists() and not list((f/'selections').glob('*.json')),
                'Unknown/partial query activity in failed conversation')
        require(not (retrieval_folder(APPROVED,ci,'jev')/'native').exists()
            and old.inventory(retrieval_folder(APPROVED,ci,'cheap')/'native')==
                bound(gate['banks'][f'{ci}/cheap'],sealed=True)['native_inventory'],
                'Failed presearch copy state differs')
    for row in success:
        ci,arm=row['conversation'],row['arm'];f=retrieval_folder(APPROVED,ci,arm)
        require(row['schema']==SCHEMA and row['model_calls']==0 and row['error'] is None and row['warnings']==[]
            and row['bank']==gate['banks'][f'{ci}/{arm}'] and not pid_alive(row['pid'])
            and body(sealed_read(f/'retrieval_outcome.json'))==row,'Historical successful retrieval not bound/closed')
        refs.append(pointer(f/'retrieval_outcome.json'))
        qids=[j['query_id'] for j in jobs if j['conversation']==ci]
        expected=[pointer(f/f'selections/{qid}.json') for qid in qids]
        require(row['selections']==expected and set((f/'selections').glob('*.json'))==
            {Path(ref['path']) for ref in expected},'Historical successful query cohort differs')
        b=bound(row['bank'],sealed=True);inv=old.inventory(f/'native');snapshot=old.snapshot(f/'native')
        require(inv==b['native_inventory'] and snapshot==old.snapshot(b['native_dir']),
            'Historical successful copy differs from approved native')
        inventories[f'{ci}/{arm}']=inv
        for ref in expected:
            s=bound(ref,sealed=True);job=next(j for j in jobs if j['query_id']==s['query_id'])
            require(s['status']=='complete' and s['model_calls']==0 and s['bank']==row['bank']
                and s['conversation']==ci and s['method']==arm and s['original_job_sha256']==digest(job)
                and s['selected_ids']==old.validate_capture_bank(s['capture'],b,snapshot)
                and s['selected_facts']==old.selected_facts(b,s['selected_ids'])
                and all(e['cache_hit'] is True and e['model_calls']==0 for e in s['cache_events']),
                'Historical capture/selected facts/cache provenance differs')
            successful[f'{arm}/{s["query_id"]}']=ref;refs.append(ref)
    require(len(successful)==1900,'Historical success slot denominator differs')
    return {'outcome':pointer(APPROVED/'retrieval_outcome.json'),'selections':successful,
        'input_refs':refs,'successful_native_inventory':inventories,'unsearched_banks':8}

def prepare(run):
    run=Path(run).resolve();t=tech();require(run==ROOT/t['new_namespace'] and not run.exists(),'Fresh retrieval namespace required')
    m0=parent.manifest(APPROVED);gate=bound(t['input_refs']['approved_effective20_gate'],sealed=True)
    require(parent.audit_banks(APPROVED,publish=False)==body(gate),'Approved20 full native closure differs')
    historical=historical_retrieval(m0,gate);sources={}
    for ci in range(10):
        for arm in original.NEW_ARMS:
            ref=gate['banks'][f'{ci}/{arm}'];b=bound(ref,sealed=True)
            sources[f'{ci}/{arm}']={'effective_bank':ref,**copy_source(ci,arm,b,m0['source_banks'][str(ci)])}
    value={'schema':MANIFEST_SCHEMA,'created_utc':now(),'protocol':m0['protocol'],'admission':m0['admission'],
        'resources':m0['resources'],'embedding_cache_pack':m0['embedding_cache_pack'],
        'source_banks':m0['source_banks'],'approved_native_manifest':t['input_refs']['approved_native_manifest'],
        'approved_effective20_gate':t['input_refs']['approved_effective20_gate'],
        'retrieval_compatibility_protocol':pointer(TECH),'snapshot_compatibility_protocol':pointer(parent.TECH),
        'physical_copy_sources':sources,'historical_retrieval':historical,
        'approved_native_input_refs':approved_input_refs(),
        'code_sha256':{p:sha(ROOT/p) for p in CODE},'native_sdk_sha256':old.native_sdk_identity(),
        'native_update_runtime_code_sha256':m0['native_update_runtime_code_sha256'],
        'model_calls_at_prepare':0,'scientific_changes':False}
    return save_new(run/'run_manifest.json',value)

def manifest(run):
    run=Path(run).resolve();t=tech();require(run==ROOT/t['new_namespace'],'Foreign retrieval namespace')
    m=sealed_read(run/'run_manifest.json')
    require(m['schema']==MANIFEST_SCHEMA and m['retrieval_compatibility_protocol']==pointer(TECH)
        and m['approved_native_manifest']==t['input_refs']['approved_native_manifest']
        and m['approved_effective20_gate']==t['input_refs']['approved_effective20_gate']
        and m['snapshot_compatibility_protocol']==pointer(parent.TECH)
        and m['code_sha256']=={p:sha(ROOT/p) for p in CODE},'Retrieval manifest/code identity differs')
    m0=parent.manifest(APPROVED,validate_admission=False)
    validate_approved_inputs(m['approved_native_input_refs'])
    bound(m['admission'],sealed=True)
    for field in ('protocol','admission','resources','embedding_cache_pack','source_banks',
        'native_sdk_sha256','native_update_runtime_code_sha256'):
        require(m[field]==m0[field],'Inherited approved-native field differs')
    gate=bound(m['approved_effective20_gate'],sealed=True)
    expected={f'{ci}/{arm}':{'effective_bank':gate['banks'][f'{ci}/{arm}'],**copy_source(ci,arm,
        bound(gate['banks'][f'{ci}/{arm}'],sealed=True),m0['source_banks'][str(ci)])}
        for ci in range(10) for arm in original.NEW_ARMS}
    require(m['physical_copy_sources']==expected,'Physical copy source/equivalence changed')
    for ref in m['historical_retrieval']['input_refs']:require(pointer(ref['path'])==ref,'Historical success bytes changed')
    for key,inv in m['historical_retrieval']['successful_native_inventory'].items():
        ci,arm=key.split('/');require(old.inventory(retrieval_folder(APPROVED,int(ci),arm)/'native')==inv,
            'Historical successful native bytes changed')
    return m

def retrieve_conversation(run,ci,m,gate,jobs,entries):
    # Windows spawn enters this original top-level function; no cloned callable submitted.
    for arm in original.NEW_ARMS:
        source=m['physical_copy_sources'][f'{ci}/{arm}']
        old.copy_closed(source['native_dir'],retrieval_folder(run,ci,arm)/'native',source['inventory'])
    with forbid_network():
        return original._retrieval_lib().retrieve_conversation(run,ci,m,gate,jobs,entries)

def retrieve(run):
    run=Path(run).resolve();m=manifest(run);gate=bound(m['approved_effective20_gate'],sealed=True)
    jobs=old.old_jobs(bound(m['protocol']['frozen_source']));pack=old.packs.verify_pack(m['embedding_cache_pack'],check_sources=True)
    save_new(run/'retrieval_execution_intent.json',{'schema':SCHEMA,'pid':os.getpid(),'created_utc':now(),'model_calls':0})
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(retrieve_conversation,run,ci,m,gate,[j for j in jobs if j['conversation']==ci],
            [e for e in pack['entries'] if e['conversation']==ci]):ci for ci in range(10)}
        for f in as_completed(tasks):
            try:rows.extend(f.result())
            except Exception as ex:rows.append({'conversation':tasks[f],'status':'failed','error_type':type(ex).__name__})
    rows.sort(key=lambda r:(r['conversation'],r.get('arm','')))
    return save_new(run/'retrieval_outcome.json',{'schema':SCHEMA,'banks':rows,'model_calls':0,'finished_utc':now(),
        'status':'complete' if len(rows)==20 and all(r['status']=='complete' for r in rows) else 'failed'})

def audit(run,publish=True):
    run=Path(run).resolve();m=manifest(run);gate=bound(m['approved_effective20_gate'],sealed=True)
    jobs=old.old_jobs(bound(m['protocol']['frozen_source']));lib=assembly_lib(run,m)
    outcome=lib.validate_retrieval_terminal(run,m,gate,jobs)
    # Frozen assembler verifies every selected fact against bank/native payload and replay capture.
    assembled=lib.construct_jobs(run,m,gate,jobs)
    pack=old.packs.verify_pack(m['embedding_cache_pack'],check_sources=True)
    from jev_mem.local_bm25_context_v2 import local_bm25_context
    events=0
    with local_bm25_context(m['resources']['snapshot'],m['resources']['files_sha256']),forbid_network():
        for job in jobs:
            entries=[e for e in pack['entries'] if e['conversation']==job['conversation']]
            for arm in original.NEW_ARMS:
                selection=sealed_read(retrieval_folder(run,job['conversation'],arm)/f'selections/{job["query_id"]}.json')
                events+=validate_cache_events(selection,job,entries,arm)
    parity=[]
    for identity,ref in m['historical_retrieval']['selections'].items():
        arm,qid=identity.split('/');ci=int(qid.split('_')[0][1:])
        newer=pointer(retrieval_folder(run,ci,arm)/f'selections/{qid}.json')
        require(selection_body(bound(ref,sealed=True))==selection_body(bound(newer,sealed=True)),
            'Historical1900 full capture/selection parity differs')
        parity.append({'identity':identity,'original':ref,'current':newer})
    require(len(parity)==1900 and len(assembled)==4620,'Fixed parity/current cohort differs')
    require(all(old.inventory(retrieval_folder(run,int(key.split('/')[0]),key.split('/')[1])/'native')==
        s['inventory'] for key,s in m['physical_copy_sources'].items()),'Copied retrieval native changed')
    value={'schema':'qualifier-repair-r09-retrieval-compatibility-gate-v1','status':'passed',
        'retrieval_manifest':pointer(run/'run_manifest.json'),'retrieval_compatibility_protocol':pointer(TECH),
        'approved_effective20_gate':m['approved_effective20_gate'],'retrieval_outcome':pointer(run/'retrieval_outcome.json'),
        'banks':20,'selected_query_slots':3080,'historical_parity_slots':1900,'historical_parity':parity,
        'physical_copy_sources':m['physical_copy_sources'],'model_calls':0,
        'audited_query_cache_events':events,
        'approved_native_files_modified':False,'scientific_changes':False}
    return save_new(run/'retrieval_gate.json',value) if publish else value

def _source_value(run,m):
    native=bound(m['approved_native_manifest'],sealed=True);gate=bound(m['approved_effective20_gate'],sealed=True)
    p=m['protocol'];source=bound(p['frozen_source']);lib=assembly_lib(run,m);jobs=old.old_jobs(source)
    lib.validate_retrieval_terminal(run,m,gate,jobs)
    value={'schema':SOURCE_SCHEMA,'status':'ready','arms':list(original.ARMS),'questions_per_arm':1540,
        'jobs':lib.construct_jobs(run,m,gate,jobs),'qualifier_repair_protocol':p['research_protocol'],
        'selector_manifest':native['original_selector_manifest'],'original_selector_manifest':native['original_selector_manifest'],
        'admission_manifest':m['admission'],'materialization_manifest':m['approved_native_manifest'],
        'technical_manifest':m['approved_native_manifest'],'snapshot_compatibility_protocol':m['snapshot_compatibility_protocol'],
        'effective_bank_gate':m['approved_effective20_gate'],'index_admission':m['approved_effective20_gate'],
        'retrieval_compatibility_protocol':pointer(TECH),'retrieval_manifest':pointer(Path(run)/'run_manifest.json'),
        'retrieval_gate':pointer(Path(run)/'retrieval_gate.json'),'retrieval_outcome':pointer(Path(run)/'retrieval_outcome.json'),
        'historical_failed_retrieval_outcome':m['historical_retrieval']['outcome'],
        'protocol_sha256':digest(p),'code_sha256':m['code_sha256'],
        'official_metric_provenance':source['official_metric_provenance'],'source_dataset_sha256':source['source_dataset_sha256'],
        'selection_counts':{arm:{'jobs':1540} for arm in original.ARMS}}
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):value[f]=p[f]
    return value

def seal(run):
    run=Path(run).resolve();m=manifest(run);g=sealed_read(run/'retrieval_gate.json')
    require(audit(run,False)==body(g),'Full retrieval gate replay differs')
    return save_new(run/'selected_inputs.json',_source_value(run,m))

def validate_source(path,fresh=False):
    path=Path(path).resolve();before=sha(path);s=sealed_read(path);run=path.parent;m=manifest(run)
    if fresh:
        require(parent.audit_banks(APPROVED,publish=False)==body(bound(m['approved_effective20_gate'],sealed=True)),
            'Approved native full closure changed')
    require(audit(run,False)==body(sealed_read(run/'retrieval_gate.json')) and body(s)==_source_value(run,m),
        'Full compatible source/capture/request replay differs')
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):bound(s[f])
    require(sha(path)==before,'Compatible source changed while validating');return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','retrieve','audit','seal','check'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    f={'prepare':prepare,'retrieve':retrieve,'audit':audit,'seal':seal,
        'check':lambda r:{'jobs':len(validate_source(Path(r)/'selected_inputs.json')['jobs']),'model_calls':0}}
    print(json.dumps(f[a.command](a.run_dir),ensure_ascii=False))
if __name__=='__main__':main()
