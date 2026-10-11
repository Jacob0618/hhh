"""Versioned R09 authenticated snapshot-body continuation; no import-time clients."""
from __future__ import annotations
import argparse,copy,json,os,sys,types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import materialize_qualifier_repair_r09 as original
from tools import r09_native_update_audit as native_audit
from tools import r09_snapshot_native_support as support
old=original.old
require,pointer,bound,sealed_read,save_new,sha=original.require,original.pointer,original.bound,original.sealed_read,original.save_new,original.sha
digest,now=old.digest,old.now
SCHEMA=original.SCHEMA;SOURCE_SCHEMA=original.SOURCE_SCHEMA;SPEC_SHA=original.SPEC_SHA
TECH_SHA='4836f4b2a9bb8684bdc89534cf635721e3fc2411408d9857ec5b7c317ce10321'
TECH=ROOT/'research/jev_autoresearch_2026-10-04/R09_snapshot_compatibility_protocol_v1.json'
ORIGINAL_RUN=ROOT/'runs/qualifier_repair_r09_v1'
CODE=tuple(dict.fromkeys((*original.CODE,'tools/materialize_qualifier_repair_snapshot_r09.py',
    'tools/r09_snapshot_native_support.py','tools/run_qualifier_repair_snapshot_qa_r09.py',
    'tools/summarize_qualifier_repair_snapshot_r09.py')))
bank_folder=original.bank_folder;retrieval_folder=original.retrieval_folder

def body(v):return {k:x for k,x in v.items() if k!='content_sha256'}
def tech():
    require(sha(TECH)==TECH_SHA,'Snapshot technical protocol changed')
    t=sealed_read(TECH)
    for ref in t['input_refs'].values():support.byte_ref(ref)
    return t
def admission(run=None):return original.admission(ORIGINAL_RUN)
def accepted(a,ci,arm):return original.accepted(a,ci,arm)

def original_terminal(m,a):
    from tools.recover_gemini38_session_io_v1 import pid_alive
    o=sealed_read(ORIGINAL_RUN/'materialization/controller_outcome.json')
    require(o['status']=='failed' and len(o['banks'])==10 and
        not pid_alive(bound(o['intent'],sealed=True)['pid']),'Original native stage is not closed terminal')
    require({(r['conversation'],r.get('arm')) for r in o['banks']}=={(ci,'cheap') for ci in range(10)},
        'Original failure population differs')
    for ci in range(10):
        f=bank_folder(ORIGINAL_RUN,ci,'cheap');row=sealed_read(f/'materialization_outcome.json')
        require(body(row)==next(r for r in o['banks'] if r['conversation']==ci)
            and row['status']=='failed' and row['bank'] is None and row['warnings']==[]
            and row['error']['phase']=='exact_native_update_replay'
            and row['error']['message']=='Original UPDATE snapshot differs' and not pid_alive(row['pid']),
            'Uncovered failure or unresolved original native owner')
        require(not bank_folder(ORIGINAL_RUN,ci,'jev').exists(),'Original JEV had an attempt')
        intent=bound(row['materialization_intent'],sealed=True)
        require(intent['conversation']==ci and intent['arm']=='cheap' and intent['admission']==m['admission']
            and intent['candidates_sha256']==digest(accepted(a,ci,'cheap')),'Original UPDATE intent differs')
        original.update_rows(f,accepted(a,ci,'cheap'))
    return o

def original_rows(ci,a):return original.update_rows(bank_folder(ORIGINAL_RUN,ci,'cheap'),accepted(a,ci,'cheap'))
def source_api(ci):return ORIGINAL_RUN/f'materialization/conv_{ci}/embedding_api'
def source_api_refs(ci,m):
    prefix=source_api(ci).resolve()
    return [ref for ref in m['input_refs'] if Path(ref['path']).is_relative_to(prefix)]

def original_input_refs():
    refs=[pointer(ORIGINAL_RUN/'materialization/controller_outcome.json')]
    for ci in range(10):
        for folder in (source_api(ci),bank_folder(ORIGINAL_RUN,ci,'cheap')):
            refs.extend(pointer(p) for p in sorted(folder.rglob('*')) if p.is_file() and
                not p.is_relative_to(folder/'native'))
    return sorted({r['path']:r for r in refs}.values(),key=lambda r:r['path'])

def review_cheap(ci,m,a):
    info=m['source_banks'][str(ci)];folder=bank_folder(ORIGINAL_RUN,ci,'cheap')
    rows=original_rows(ci,a);before=bound(info['snapshot'],sealed=True)
    canonical=support.snapshot_body(before);after=old.snapshot(folder/'native')
    delta=native_audit.validate_delta(canonical,after,rows,info['user_id'])
    replay=native_audit.replay_exact(info['native_dir'],canonical,after,rows,m['resources'])
    proof=native_audit.audit_embeddings(source_api(ci),ci,{'cheap':rows})
    bm25=old.compare_persisted_fact_vectors([folder/'native'],m['resources']['snapshot'],m['resources']['files_sha256'])
    require(bm25['facts_compared']==bm25['exact_matches']==len(after['facts']),'Reviewed cheap BM25 differs')
    return {'schema':SCHEMA,'complete':True,'conversation':ci,'arm':'cheap','user_id':info['user_id'],
        'native_dir':str((folder/'native').resolve()),'native_inventory':old.inventory(folder/'native'),
        'original_bank':info['bank'],'admission':m['admission'],
        'facts':original.project_bank(bound(info['bank'],sealed=True),after,rows),'updates':rows,
        'gate':delta,'numeric_replay':replay,'dense_embedding_proof':native_audit.arm_embedding_proof(proof,'cheap'),
        'bm25':bm25,'protocol_sha256':digest(m['protocol']),
        'origin_kind':'reviewed_old_failed_cheap_no_redispatch',
        'original_failed_outcome':pointer(folder/'materialization_outcome.json'),
        'authenticated_snapshot':{'original_ref':info['snapshot'],'body_sha256':digest(canonical)},
        'new_native_dispatches':0,'new_model_dispatches':0},proof

def prepare(run):
    run=Path(run).resolve();t=tech()
    require(run==ROOT/t['new_namespace'] and not run.exists(),'Fresh technical namespace required')
    base=original.manifest(ORIGINAL_RUN);a=admission();original_terminal(base,a)
    m={**body(base),'created_utc':now(),'code_sha256':{p:sha(ROOT/p) for p in CODE},
        'snapshot_compatibility_protocol':pointer(TECH),
        'original_native_manifest':t['input_refs']['original_native_manifest'],
        'original_selector_manifest':t['input_refs']['original_selector_manifest'],
        'input_refs':original_input_refs(),
        'execution':{**base['execution'],'reviewed_cheap_conversations':10,'first_attempt_jev_conversations':10},
        'technical_continuation_not_scientific_round':True}
    reviewed={};proofs={}
    for ci in range(10):reviewed[str(ci)],proofs[str(ci)]=review_cheap(ci,m,a)
    m['reviewed_cheap_native_inventory']={ci:b['native_inventory'] for ci,b in reviewed.items()}
    mref=save_new(run/'materialization/run_manifest.json',m)
    # The frozen source assembler reads this path before its selector alias is
    # explicitly restored to the original queryblind proposer manifest.
    save_new(run/'run_manifest.json',m)
    for ci,b in reviewed.items():
        f=bank_folder(run,int(ci),'cheap');br=save_new(f/'bank.json',b)
        save_new(f/'materialization_outcome.json',{'schema':SCHEMA,'conversation':int(ci),'arm':'cheap',
            'status':'complete','origin_kind':b['origin_kind'],'bank':br,'warnings':[],
            'original_failed_outcome':b['original_failed_outcome'],'new_native_dispatches':0,'new_model_dispatches':0})
    return mref

def manifest(run,validate_admission=True):
    run=Path(run).resolve();t=tech();require(run==ROOT/t['new_namespace'],'Foreign technical namespace')
    m=sealed_read(run/'materialization/run_manifest.json')
    require(sealed_read(run/'run_manifest.json')==m,'Technical manifest mirror differs')
    require(m['snapshot_compatibility_protocol']==pointer(TECH)
        and m['original_native_manifest']==t['input_refs']['original_native_manifest']
        and m['original_selector_manifest']==t['input_refs']['original_selector_manifest'],
        'Technical manifest parent pointers differ')
    base=bound(m['original_native_manifest'],sealed=True)
    for field in ('schema','admission','protocol','source_banks','resources','embedding_cache_pack',
                  'retrieval_manifest','native_sdk_sha256','native_update_runtime_code_sha256'):
        require(m[field]==base[field],'Inherited materialization field changed')
    for p,h in base['code_sha256'].items():require(sha(ROOT/p)==h,'Original frozen code changed')
    require(m['code_sha256']=={p:sha(ROOT/p) for p in CODE},'Technical code changed')
    require(m['native_sdk_sha256']==old.native_sdk_identity() and
        m['native_update_runtime_code_sha256']==native_audit.runtime_pins(),'Native runtime changed')
    require(m['input_refs']==original_input_refs(),'Original input ref inventory changed')
    for ref in m['input_refs']:support.byte_ref(ref)
    for ci,info in m['source_banks'].items():
        bound(info['bank'],sealed=True);support.snapshot_body(bound(info['snapshot'],sealed=True))
        require(old.inventory(info['native_dir'])==info['inventory'],'Original N0 native changed')
        require(old.inventory(bank_folder(ORIGINAL_RUN,int(ci),'cheap')/'native')==
                m['reviewed_cheap_native_inventory'][ci],'Approved old failed cheap store changed')
    if validate_admission:
        a=admission();require(a==bound(m['admission'],sealed=True),'Original admission changed')
        original_terminal(m,a)
    return m

class EmbeddingOnlyAPI(original.EmbeddingOnlyAPI):
    def __init__(self,cache_dir,*args,**kwargs):
        ci=kwargs.get('context',{}).get('conversation')
        require(type(ci) is int and ci in range(10),'Embedding import conversation missing')
        roots=(source_api(ci)/'api_cache',)
        require(not kwargs.get('cache_roots') or tuple(Path(x) for x in kwargs['cache_roots'])==roots,
            'Foreign embedding import roots')
        kwargs['cache_roots']=roots;super().__init__(cache_dir,*args,**kwargs)

def _native_lib(run,m):
    def rows(folder,candidates):
        if Path(folder).name=='cheap':return original.update_rows(bank_folder(ORIGINAL_RUN,
            int(Path(folder).parent.name.split('_')[1]),'cheap'),candidates)
        return original.update_rows(folder,candidates)
    def embeddings(api,ci,byarm):
        require(set(byarm)=={'cheap','jev'} and byarm['cheap']==original_rows(ci,admission()),
            'New embedding source rows differ')
        return support.audit_imported_embeddings(api,source_api(ci),ci,byarm['jev'],byarm['cheap'],source_api_refs(ci,m))
    proxy=types.SimpleNamespace(**vars(native_audit));proxy.replay_exact=support.replay_exact;proxy.audit_embeddings=embeddings
    return support.isolated(original,{'manifest':manifest,'admission':admission,'audit':proxy,
        'EmbeddingOnlyAPI':EmbeddingOnlyAPI,'update_rows':rows})

def materialize_one(run,ci,m,a):return _native_lib(run,m).materialize_bank(run,ci,'jev',m,a)
def materialize(run):
    run=Path(run).resolve();m=manifest(run);a=admission()
    intent=save_new(run/'materialization/controller_intent.json',{'schema':SCHEMA,'pid':os.getpid(),
        'created_utc':now(),'phase':'first_attempt_jev','original_cheap_no_redispatch':True})
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(materialize_one,run,ci,m,a):ci for ci in range(10)}
        for f in as_completed(tasks):
            try:rows.append(f.result())
            except Exception as ex:rows.append({'conversation':tasks[f],'arm':'jev','status':'failed','error_type':type(ex).__name__})
    rows.sort(key=lambda r:r['conversation'])
    return save_new(run/'materialization/controller_outcome.json',{'schema':SCHEMA,'banks':rows,'intent':intent,
        'finished_utc':now(),'status':'complete' if len(rows)==10 and all(r['status']=='complete' for r in rows) else 'failed'})

def audit_banks(run,publish=True):
    from tools.recover_gemini38_session_io_v1 import pid_alive
    run=Path(run).resolve();m=manifest(run);a=admission();o=sealed_read(run/'materialization/controller_outcome.json')
    require(o['status']=='complete' and len(o['banks'])==10 and
        {r['conversation'] for r in o['banks']}==set(range(10)) and
        not pid_alive(bound(o['intent'],sealed=True)['pid']),'First-JEV stage not closed complete')
    refs={};sourceproof={};newproof={}
    for ci in range(10):
        expected,sourceproof[str(ci)]=review_cheap(ci,m,a);cf=bank_folder(run,ci,'cheap')
        co=sealed_read(cf/'materialization_outcome.json');require(body(bound(co['bank'],sealed=True))==expected
            and co['bank']==pointer(cf/'bank.json') and co['origin_kind']==expected['origin_kind']
            and co['original_failed_outcome']==expected['original_failed_outcome']
            and co['schema']==SCHEMA and co['conversation']==ci and co['arm']=='cheap'
            and co['status']=='complete' and co['warnings']==[] and co['new_native_dispatches']==co['new_model_dispatches']==0,
            'Reviewed cheap metadata/closure changed')
        refs[f'{ci}/cheap']=co['bank']
        f=bank_folder(run,ci,'jev');row=sealed_read(f/'materialization_outcome.json')
        require(body(row)==next(r for r in o['banks'] if r['conversation']==ci) and row['arm']=='jev'
            and row['status']=='complete' and row['warnings']==[] and row['error'] is None and not pid_alive(row['pid'])
            and row['bank']==pointer(f/'bank.json'),'First-JEV terminal binding differs')
        b=bound(row['bank'],sealed=True);rows=original.update_rows(f,accepted(a,ci,'jev'));info=m['source_banks'][str(ci)]
        after=old.snapshot(f/'native');before=bound(info['snapshot'],sealed=True)
        require(b['conversation']==ci and b['arm']=='jev' and b['updates']==rows and b['admission']==m['admission']
            and b['original_bank']==info['bank'] and b['native_dir']==str((f/'native').resolve())
            and old.inventory(f/'native')==b['native_inventory'],'First-JEV native identity changed')
        require(b['gate']==native_audit.validate_delta(support.snapshot_body(before),after,rows,info['user_id'])
            and b['numeric_replay']==support.replay_exact(info['native_dir'],before,after,rows,m['resources'])
            and b['facts']==original.project_bank(bound(info['bank'],sealed=True),after,rows),'First-JEV full native replay differs')
        newproof[str(ci)]=support.audit_imported_embeddings(run/f'materialization/conv_{ci}/embedding_api',
            source_api(ci),ci,rows,original_rows(ci,a),source_api_refs(ci,m))
        require(b['dense_embedding_proof']==native_audit.arm_embedding_proof(newproof[str(ci)],'jev'),
            'First-JEV embedding lineage changed')
        bm25=old.compare_persisted_fact_vectors([f/'native'],m['resources']['snapshot'],m['resources']['files_sha256'])
        require(bm25==b['bm25'] and bm25['facts_compared']==bm25['exact_matches']==len(after['facts']),
            'First-JEV BM25 differs')
        refs[f'{ci}/jev']=row['bank']
    value={'schema':SCHEMA,'status':'passed','admission':m['admission'],'banks':refs,'new_banks':20,
        'source_original_facts':1637,'semantic_support_certified':False,'model_calls':0,
        'embedding_audits':{'original':sourceproof,'first_attempt':newproof},
        'effective_origins':{'cheap':'reviewed_old_failed_no_redispatch','jev':'fresh_first_attempt'},
        'snapshot_compatibility_protocol':pointer(TECH),'technical_manifest':pointer(run/'materialization/run_manifest.json')}
    return save_new(run/'materialization/index_admission.json',value) if publish else value

def retrieve_conversation(run,ci,m,gate,jobs,entries):
    return original.retrieve_conversation(run,ci,m,gate,jobs,entries)
def retrieve(run):
    return support.isolated(original,{'manifest':manifest,'retrieve_conversation':retrieve_conversation}).retrieve(run)
def _source_value(run,m,gate):
    value=original._source_value(run,m,gate)
    value.update(selector_manifest=m['original_selector_manifest'],original_selector_manifest=m['original_selector_manifest'],
        materialization_manifest=pointer(Path(run)/'materialization/run_manifest.json'),
        technical_manifest=pointer(Path(run)/'materialization/run_manifest.json'),
        snapshot_compatibility_protocol=pointer(TECH),effective_bank_gate=pointer(Path(run)/'materialization/index_admission.json'))
    return value
def seal(run):
    run=Path(run).resolve();m=manifest(run);gate=sealed_read(run/'materialization/index_admission.json')
    require(audit_banks(run,False)==body(gate),'Effective20 replay differs')
    return save_new(run/'selected_inputs.json',_source_value(run,m,gate))
def validate_source(path,fresh=False):
    path=Path(path).resolve();before=sha(path);s=sealed_read(path);run=path.parent;m=manifest(run)
    gate=sealed_read(run/'materialization/index_admission.json')
    require(audit_banks(run,False)==body(gate) and body(s)==_source_value(run,m,gate),
        'Full technical native/source/request replay differs')
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):bound(s[f])
    require(sha(path)==before,'Selected source changed while validating');return s
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','materialize','audit','retrieve','seal','check'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    functions={'prepare':prepare,'materialize':materialize,'audit':audit_banks,'retrieve':retrieve,'seal':seal,
        'check':lambda r:{'jobs':len(validate_source(Path(r)/'selected_inputs.json')['jobs']),'model_calls':0}}
    print(json.dumps(functions[a.command](a.run_dir),ensure_ascii=False))
if __name__=='__main__':main()
