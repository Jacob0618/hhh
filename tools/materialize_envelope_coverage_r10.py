"""R10 fresh original-bank UPDATE and cache-only SDK20, private frozen primitives."""
from __future__ import annotations
import argparse,copy,json,os,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import materialize_qualifier_repair_r09 as prior
from tools import r09_native_update_audit as original_audit
from tools import r09_snapshot_native_support as support
from tools import retrieve_qualifier_repair_compatible_r09 as cache_prior
from tools.recover_gemini38_session_io_v1 import pid_alive
from jev_mem.local_bm25_context_v2 import forbid_network
old=prior.old
require,pointer,bound,sealed_read,save_new,sha,digest,now=prior.require,prior.pointer,prior.bound,prior.sealed_read,prior.save_new,prior.sha,prior.digest,prior.now
SCHEMA='envelope-coverage-r10-native-v1'
SOURCE_SCHEMA='envelope-coverage-r10-selected-inputs-v1'
RETRIEVAL_SCHEMA='envelope-coverage-r10-retrieval-manifest-v1'
SPEC_SHA='c422886b79e17358851189839795af5dd44dc7cb3c574b9dc5bcd7917d9e872c'
SPEC=ROOT/'research/jev_autoresearch_2026-10-04/R10_envelope_coverage_protocol_v1.json'
R09_SHA=prior.SPEC_SHA
SELECTOR=ROOT/'runs/envelope_coverage_r10_v1'
NATIVE=ROOT/'runs/envelope_coverage_r10_native_v1'
RETRIEVAL=ROOT/'runs/envelope_coverage_r10_retrieval_v1'
ARMS=prior.ARMS;NEW_ARMS=prior.NEW_ARMS
CODE=tuple(dict.fromkeys((*prior.CODE,'tools/r09_snapshot_native_support.py',
    'tools/retrieve_qualifier_repair_compatible_r09.py','tools/materialize_envelope_coverage_r10.py',
    'jev_mem/envelope_coverage_r10.py','tools/prepare_envelope_coverage_r10.py',
    'tools/score_envelope_coverage_r10.py','tools/run_envelope_coverage_qa_r10.py',
    'tools/summarize_envelope_coverage_r10.py','tools/snapshot_envelope_coverage_r10.py')))
accepted=prior.accepted
audit=support.isolated(original_audit,{'SCHEMA':SCHEMA})

def body(value):return {k:v for k,v in value.items() if k!='content_sha256'}
def codes():return {p:sha(ROOT/p) for p in CODE}

def protocol():
    require(sha(SPEC)==SPEC_SHA,'R10 scientific protocol changed')
    value=bound(pointer(SPEC))
    require(value['namespaces']=={**value['namespaces'],'selector':str(SELECTOR.relative_to(ROOT)).replace('\\','/'),
        'native':str(NATIVE.relative_to(ROOT)).replace('\\','/'),
        'retrieval':str(RETRIEVAL.relative_to(ROOT)).replace('\\','/')},'R10 namespaces differ')
    return value

def admission(path_or_run=None):
    from tools.prepare_envelope_coverage_r10 import validate_admission
    a=validate_admission(SELECTOR)
    require(a['protocol']['research_protocol']==pointer(SPEC)
        and a['protocol']['qualifier_repair_protocol']['sha256']==R09_SHA,'Repair source protocol differs')
    bound(a['protocol']['qualifier_repair_protocol'])
    if path_or_run is not None and Path(path_or_run).resolve()==NATIVE:
        require(sha(NATIVE/'admission_manifest.json')==sha(SELECTOR/'admission_manifest.json'),
            'Native admission copy differs')
    return a

def snapshot_bound(ref,sealed=False):
    value=bound(ref,sealed=sealed)
    if sealed and Path(ref['path']).name=='original_native_snapshot.json':
        return support.snapshot_body(value)
    return value

def _publish(path,value):
    if Path(path).name=='run_manifest.json' and 'source_banks' in value:
        value={**value,'native_update_runtime_code_sha256':audit.runtime_pins(),
            'envelope_coverage_protocol':pointer(SPEC),'qualifier_repair_protocol':value['protocol']['qualifier_repair_protocol'],
            'original_selector_manifest':pointer(SELECTOR/'run_manifest.json'),
            'original_admission_manifest':pointer(SELECTOR/'admission_manifest.json'),
            'historical_native_embedding_cache_imported':False}
    return save_new(path,value)

def _old_lib(overrides=None):
    defaults={'SCHEMA':SCHEMA,'SOURCE_SCHEMA':SOURCE_SCHEMA,'SPEC_SHA':SPEC_SHA,'CODE':CODE,
        'admission':admission,'accepted':accepted,'save_new':_publish,'bound':snapshot_bound}
    defaults.update(overrides or {})
    return support.isolated(old,defaults)

def native_lib(overrides=None):
    defaults={'SCHEMA':SCHEMA,'SOURCE_SCHEMA':SOURCE_SCHEMA,'SPEC_SHA':SPEC_SHA,'CODE':CODE,
        'audit':audit,'admission':admission,'accepted':accepted,'bound':snapshot_bound,
        '_lib':_old_lib,'manifest':native_manifest,'save_new':save_new}
    defaults.update(overrides or {})
    return support.isolated(prior,defaults)

def prepare_native(run=NATIVE):
    run=Path(run).resolve();protocol();require(run==NATIVE and not run.exists(),'Fresh R10 native namespace required')
    admission();source=SELECTOR/'admission_manifest.json'
    # Preserve the exact admitted artifact bytes, not a differently resealed copy.
    run.mkdir(parents=True);target=run/'admission_manifest.json';target.write_bytes(old.shared_read(source))
    require(sha(target)==sha(source),'Admission bytes changed during copy')
    return _old_lib().prepare(run)

def native_manifest(run=NATIVE,validate_admission=True):
    run=Path(run).resolve();require(run==NATIVE,'Foreign R10 native namespace');protocol()
    m=_old_lib().manifest(run,validate_admission=validate_admission)
    require(m['code_sha256']==codes() and m['native_update_runtime_code_sha256']==audit.runtime_pins()
        and m['envelope_coverage_protocol']==pointer(SPEC)
        and m['qualifier_repair_protocol']==m['protocol']['qualifier_repair_protocol']
        and m['original_selector_manifest']==pointer(SELECTOR/'run_manifest.json')
        and m['original_admission_manifest']==pointer(SELECTOR/'admission_manifest.json')
        and m['admission']==pointer(NATIVE/'admission_manifest.json')
        and pointer(NATIVE/'admission_manifest.json')['sha256']==m['original_admission_manifest']['sha256'],
        'R10 native/runtime/admission identity differs')
    return m

def materialize_conversation(run,ci,m,admit):
    # A top-level spawn worker; never submit a cloned function to Windows spawn.
    return native_lib().materialize_conversation(run,ci,m,admit)

def materialize(run=NATIVE):return native_lib({'materialize_conversation':materialize_conversation}).materialize(run)
def audit_banks(run=NATIVE,publish=True):return native_lib().audit_banks(run,publish=publish)

def proof_refs():
    folder=NATIVE/'materialization'
    return [pointer(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and 'native' not in p.relative_to(folder).parts]

def validate_native_gate(m,gate):
    require(gate['schema']==SCHEMA and gate['status']=='passed' and gate['new_banks']==20
        and gate['admission']==m['admission'] and gate['model_calls']==0
        and set(gate['banks'])=={f'{ci}/{arm}' for ci in range(10) for arm in NEW_ARMS},'R10 native gate differs')
    for key,ref in gate['banks'].items():
        ci,arm=key.split('/');ci=int(ci);bank=bound(ref,sealed=True)
        require(ref==pointer(prior.bank_folder(NATIVE,ci,arm)/'bank.json')
            and bank['complete'] and bank['conversation']==ci and bank['arm']==arm
            and bank['admission']==m['admission'] and bank['original_bank']==m['source_banks'][str(ci)]['bank']
            and bank['protocol_sha256']==digest(m['protocol'])
            and old.inventory(bank['native_dir'])==bank['native_inventory'],'Effective R10 bank identity changed')

def copy_source(bank,info):
    source=Path(bank['native_dir']).resolve();inv=old.inventory(source)
    require(inv==bank['native_inventory'],'Effective native inventory changed')
    if bank['updates']:
        require((source/'qdrant/.lock').is_file() and not (source/'qdrant/.lock').is_symlink(),
            'Updated effective source lock missing')
        return {'native_dir':str(source),'inventory':inv,'origin':'effective_updated_bank','equivalence':None}
    original=Path(info['native_dir']).resolve();snap=support.snapshot_body(bound(info['snapshot'],sealed=True))
    n0=bound(info['bank'],sealed=True)
    require(inv==info['inventory']==old.inventory(original) and old.snapshot(source)==snap==old.snapshot(original)
        and bank['facts']==n0['facts'] and bank['user_id']==n0['user_id']==info['user_id']
        and (original/'qdrant/.lock').is_file() and not (original/'qdrant/.lock').is_symlink(),
        'Zero UPDATE original/effective exact equivalence differs')
    return {'native_dir':str(original),'inventory':inv,'origin':'equivalent_original_N0_locked_source',
        'equivalence':{'zero_updates':True,'effective_native_dir':str(source),'original_bank':info['bank'],
            'original_snapshot':info['snapshot'],'snapshot_body_sha256':digest(snap),
            'exported_facts_sha256':digest(bank['facts']),'full_file_inventory_equal':True,
            'full_logical_snapshot_equal':True,'effective_native_modified':False}}

def prepare_retrieval(run=RETRIEVAL):
    run=Path(run).resolve();require(run==RETRIEVAL and not run.exists(),'Fresh R10 retrieval namespace required')
    m=native_manifest();gate=sealed_read(NATIVE/'materialization/index_admission.json')
    validate_native_gate(m,gate)
    require(audit_banks(publish=False)==body(gate),'All20 native UPDATE audit replay differs')
    sources={k:{'effective_bank':ref,**copy_source(bound(ref,sealed=True),m['source_banks'][k.split('/')[0]])}
        for k,ref in gate['banks'].items()}
    value={'schema':RETRIEVAL_SCHEMA,'created_utc':now(),'native_manifest':pointer(NATIVE/'materialization/run_manifest.json'),
        'effective_bank_gate':pointer(NATIVE/'materialization/index_admission.json'),'protocol':m['protocol'],
        'admission':m['admission'],'envelope_coverage_protocol':pointer(SPEC),'qualifier_repair_protocol':m['qualifier_repair_protocol'],
        'original_selector_manifest':m['original_selector_manifest'],'source_banks':m['source_banks'],
        'resources':m['resources'],'embedding_cache_pack':m['embedding_cache_pack'],'physical_copy_sources':sources,
        'native_input_refs':proof_refs(),'code_sha256':codes(),'native_sdk_sha256':m['native_sdk_sha256'],
        'native_update_runtime_code_sha256':m['native_update_runtime_code_sha256'],'model_calls_at_prepare':0,
        'capture_admission':'Exact frozen scorer replay conditional on each current capture recorded boosts',
        'historical_numeric_capture_parity_required':False}
    return save_new(run/'run_manifest.json',value)

def retrieval_manifest(run=RETRIEVAL):
    run=Path(run).resolve();require(run==RETRIEVAL,'Foreign R10 retrieval namespace');m=sealed_read(run/'run_manifest.json')
    native=native_manifest(validate_admission=False)
    require(m['schema']==RETRIEVAL_SCHEMA and m['code_sha256']==codes()
        and m['native_manifest']==pointer(NATIVE/'materialization/run_manifest.json')
        and m['effective_bank_gate']==pointer(NATIVE/'materialization/index_admission.json')
        and m['native_input_refs']==proof_refs(),'Retrieval parent/code/proof bytes changed')
    for f in ('protocol','admission','envelope_coverage_protocol','qualifier_repair_protocol',
        'original_selector_manifest','source_banks','resources','embedding_cache_pack','native_sdk_sha256',
        'native_update_runtime_code_sha256'):require(m[f]==native[f],'Inherited R10 retrieval field changed')
    gate=bound(m['effective_bank_gate'],sealed=True);validate_native_gate(native,gate)
    require(set(m['physical_copy_sources'])==set(gate['banks']),'Physical copy source cohort differs')
    for key,ref in gate['banks'].items():
        require(m['physical_copy_sources'][key]=={'effective_bank':ref,
            **copy_source(bound(ref,sealed=True),m['source_banks'][key.split('/')[0]])},'Physical donor changed')
    return m

def retrieval_folder(run,ci,arm):return prior.retrieval_folder(run,ci,arm)
def assembly_lib(run,m):
    alias=(Path(run)/'materialization/index_admission.json').resolve()
    def assembly_pointer(path):
        if Path(path).resolve()==alias:
            ref=m['effective_bank_gate'];require(pointer(ref['path'])==ref,'Native assembler gate changed');return ref
        return pointer(path)
    return _old_lib({'bank_folder':retrieval_folder,'pointer':assembly_pointer,'bound':bound})

def retrieve_conversation(run,ci,m,gate,jobs,entries):
    for arm in NEW_ARMS:
        source=m['physical_copy_sources'][f'{ci}/{arm}']
        old.copy_closed(source['native_dir'],retrieval_folder(run,ci,arm)/'native',source['inventory'])
    with forbid_network():return assembly_lib(run,m).retrieve_conversation(run,ci,m,gate,jobs,entries)

def retrieve(run=RETRIEVAL):
    run=Path(run).resolve();m=retrieval_manifest(run);gate=bound(m['effective_bank_gate'],sealed=True)
    jobs=old.old_jobs(bound(m['protocol']['frozen_source']));pack=old.packs.verify_pack(m['embedding_cache_pack'],check_sources=True)
    require(not (run/'retrieval_execution_intent.json').exists(),'Retrieval reentry refused')
    save_new(run/'retrieval_execution_intent.json',{'schema':SCHEMA,'pid':os.getpid(),'created_utc':now(),'model_calls':0})
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(retrieve_conversation,run,ci,m,gate,[j for j in jobs if j['conversation']==ci],
            [e for e in pack['entries'] if e['conversation']==ci]):ci for ci in range(10)}
        for task in as_completed(tasks):
            try:rows.extend(task.result())
            except Exception as ex:rows.append({'conversation':tasks[task],'status':'failed','error_type':type(ex).__name__,'model_calls':0})
    rows.sort(key=lambda x:(x['conversation'],x.get('arm','')))
    return save_new(run/'retrieval_outcome.json',{'schema':SCHEMA,'banks':rows,'model_calls':0,'finished_utc':now(),
        'status':'complete' if len(rows)==20 and all(r['status']=='complete' for r in rows) else 'failed'})

def validate_cache_events(selection,job,entries,arm):
    from mem0.memory.main import _validate_and_trim_search_query
    derived=copy.deepcopy(job);derived['question']=_validate_and_trim_search_query(job['question'])
    return support.isolated(cache_prior,{'SCHEMA':SCHEMA}).validate_cache_events(selection,derived,entries,arm)

def retrieval_audit_value(run,m):
    gate=bound(m['effective_bank_gate'],sealed=True);jobs=old.old_jobs(bound(m['protocol']['frozen_source']))
    lib=assembly_lib(run,m);lib.validate_retrieval_terminal(run,m,gate,jobs)
    intent=sealed_read(Path(run)/'retrieval_execution_intent.json')
    require(intent['schema']==SCHEMA and intent['model_calls']==0 and not pid_alive(intent['pid']),
        'Retrieval controller not closed')
    pack=old.packs.verify_pack(m['embedding_cache_pack'],check_sources=True);events=0;refs=[]
    with old.local_bm25_context(m['resources']['snapshot'],m['resources']['files_sha256']),forbid_network():
        for arm in NEW_ARMS:
            for job in jobs:
                ref=pointer(retrieval_folder(run,job['conversation'],arm)/f'selections/{job["query_id"]}.json')
                events+=validate_cache_events(bound(ref,sealed=True),job,
                    [e for e in pack['entries'] if e['conversation']==job['conversation']],arm);refs.append(ref)
    selected=lib.construct_jobs(run,m,gate,jobs)
    require(len(refs)==3080 and len(selected)==4620,'Current SDK20 cohort differs')
    return {'schema':SCHEMA,'status':'passed','envelope_coverage_protocol':pointer(SPEC),
        'qualifier_repair_protocol':m['qualifier_repair_protocol'],'retrieval_manifest':pointer(Path(run)/'run_manifest.json'),
        'effective_bank_gate':m['effective_bank_gate'],'retrieval_outcome':pointer(Path(run)/'retrieval_outcome.json'),
        'selection_refs':refs,'current_capture_self_replay':3080,'exact_query_cache_events':events,'model_calls':0,
        'historical_numeric_capture_parity_required':False,'semantic_support_certified':False}

def audit_retrieval(run=RETRIEVAL,publish=True):
    run=Path(run).resolve();m=retrieval_manifest(run);value=retrieval_audit_value(run,m)
    return save_new(run/'retrieval_gate.json',value) if publish else value

def selection_diagnostics(jobs,gate):
    """Measured evidence/signature controls, never a gate forcing equal answers."""
    from tools.run_evidence_exchange_qa_r01 import request_signature
    from jev_mem.sdk20_cached_api import digest as request_hash
    baseline={j['query_id']:j for j in jobs if j['method']=='baseline'}
    banks={k:bound(ref,sealed=True) for k,ref in gate['banks'].items()};stats={};controls={}
    for arm in ARMS:
        selected=[j for j in jobs if j['method']==arm];changed=0;occurrences=0;zero=[]
        for j in selected:
            if arm=='baseline':continue
            bank=banks[f'{j["conversation"]}/{arm}'];revised={r['memory_id'] for r in bank['updates']}
            n=len(set(j['selected_ids'])&revised);changed+=bool(n);occurrences+=n
            if not revised:
                original=baseline[j['query_id']];a=request_signature(original,'reader');b=request_signature(j,'reader')
                zero.append({'query_id':j['query_id'],'conversation':j['conversation'],
                    'equal_full_reader_signature':a==b,'baseline_request_hash':request_hash(a),
                    'current_request_hash':request_hash(b),'equal_ordered_reader_facts':j['selected_facts']==original['selected_facts']})
        stats[arm]={'jobs':len(selected),'queries_selecting_revised_fact':changed,'selected_revised_occurrences':occurrences}
        if arm!='baseline':controls[arm]={'banks':[int(k.split('/')[0]) for k,v in banks.items()
            if k.endswith('/'+arm) and not v['updates']],'queries':len(zero),
            'equal_full_reader_requests':sum(x['equal_full_reader_signature'] for x in zero),
            'different_full_reader_requests':sum(not x['equal_full_reader_signature'] for x in zero),'rows':zero}
    return stats,controls

def source_value(run,m):
    source=bound(m['protocol']['frozen_source']);p=m['protocol'];gate=bound(m['effective_bank_gate'],sealed=True)
    jobs=assembly_lib(run,m).construct_jobs(run,m,gate,old.old_jobs(source))
    counts,controls=selection_diagnostics(jobs,gate)
    value={'schema':SOURCE_SCHEMA,'status':'ready','arms':list(ARMS),'questions_per_arm':1540,'jobs':jobs,
        'envelope_coverage_protocol':pointer(SPEC),'qualifier_repair_protocol':m['qualifier_repair_protocol'],
        'selector_manifest':m['original_selector_manifest'],'original_selector_manifest':m['original_selector_manifest'],
        'admission_manifest':pointer(SELECTOR/'admission_manifest.json'),'materialization_manifest':m['native_manifest'],
        'index_admission':m['effective_bank_gate'],'effective_bank_gate':m['effective_bank_gate'],
        'retrieval_manifest':pointer(Path(run)/'run_manifest.json'),'retrieval_gate':pointer(Path(run)/'retrieval_gate.json'),
        'retrieval_outcome':pointer(Path(run)/'retrieval_outcome.json'),'protocol_sha256':digest(p),'code_sha256':m['code_sha256'],
        'official_metric_provenance':source['official_metric_provenance'],'source_dataset_sha256':source['source_dataset_sha256'],
        'selection_counts':counts,'zero_revision_reader_controls':controls,
        'phase_sources':{'selector_run':str(SELECTOR),'native_run':str(NATIVE),'retrieval_run':str(Path(run).resolve())},
        'selector_lineage':{'admission':pointer(SELECTOR/'admission_manifest.json'),'manifest':m['original_selector_manifest'],
            'compiled':pointer(SELECTOR/'score_manifest.json'),'score_audit':pointer(SELECTOR/'reviews/jev_terminal_audit_v1.json')},
        'historical_numeric_capture_parity_required':False}
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):value[f]=p[f]
    return value

def seal(run=RETRIEVAL):
    run=Path(run).resolve();m=retrieval_manifest(run);gate=sealed_read(run/'retrieval_gate.json')
    require(body(gate)==retrieval_audit_value(run,m),'Current retrieval gate replay differs')
    return save_new(run/'selected_inputs.json',source_value(run,m))

def validate_source(path,fresh=False):
    path=Path(path).resolve();require(path==RETRIEVAL/'selected_inputs.json','Foreign R10 selected source')
    start=sha(path);s=sealed_read(path);m=retrieval_manifest(path.parent);gate=sealed_read(path.parent/'retrieval_gate.json')
    if fresh:
        require(audit_banks(publish=False)==body(bound(m['effective_bank_gate'],sealed=True)),'Fresh native replay differs')
    require(body(gate)==retrieval_audit_value(path.parent,m),'Exact current cache/capture gate differs')
    require(body(s)==source_value(path.parent,m),'Full selected source/lineage replay differs')
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):bound(s[f])
    require(sha(path)==start,'Selected input changed during validation');return s

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','materialize','audit','prepare-retrieval','retrieve','audit-retrieval','seal','check'))
    p.add_argument('--run-dir',type=Path);a=p.parse_args();native=a.command in ('prepare','materialize','audit')
    run=a.run_dir or (NATIVE if native else RETRIEVAL)
    functions={'prepare':prepare_native,'materialize':materialize,'audit':audit_banks,'prepare-retrieval':prepare_retrieval,
        'retrieve':retrieve,'audit-retrieval':audit_retrieval,'seal':seal,
        'check':lambda r:{'jobs':len(validate_source(Path(r)/'selected_inputs.json')['jobs']),'model_calls':0}}
    print(json.dumps(functions[a.command](run),ensure_ascii=False))
if __name__=='__main__':main()
