"""R09 isolated native UPDATE and frozen SDK20 retrieval. No import-time clients."""
from __future__ import annotations
import argparse,copy,contextlib,json,logging,os,sys,types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import materialize_write_omission_r07 as old
from tools import r09_native_update_audit as audit
from tools.run_sdk20_modules import digest,now,Warnings
from jev_mem.paced_embedding_api import PacedEmbeddingAPI,DEFAULT_GATE
from jev_mem.providers import Embeddings
SCHEMA=audit.SCHEMA
SOURCE_SCHEMA='qualifier-repair-r09-selected-inputs-v1'
SPEC_SHA='59908bea9a269f5c79bb30a40bf30df165362bc5f5e291101f3e63b8731efc4b'
ARMS=old.ARMS;NEW_ARMS=old.NEW_ARMS
CODE=tuple(dict.fromkeys((*old.CODE,'tools/materialize_qualifier_repair_r09.py',
    'tools/r09_native_update_audit.py','jev_mem/qualifier_repair_r09.py',
    'tools/prepare_qualifier_repair_r09.py','tools/score_qualifier_repair_r09.py',
    'tools/run_qualifier_repair_qa_r09.py','tools/summarize_qualifier_repair_r09.py')))
require,pointer,bound,sealed_read,save_new,sha=old.require,old.pointer,old.bound,old.sealed_read,old.save_new,old.sha

def admission(path_or_run):
    from tools.prepare_qualifier_repair_r09 import validate_admission
    return validate_admission(path_or_run)

def accepted(admit,ci,arm):
    result=[]
    for item in admit['items']:
        if item['conversation']!=ci:continue
        candidates=item['candidates'];by={c['candidate_id']:c for c in candidates};ids=item['accepted'][arm]
        require(ids==[c['candidate_id'] for c in candidates if c['candidate_id'] in ids]
                and len(ids)==len(set(ids)) and set(ids)<=set(by),'Canonical admitted revision order differs')
        result.extend(copy.deepcopy(by[x]) for x in ids)
    require(len({c['fact_id'] for c in result})==len(result),'Same ID revised twice')
    return result

def _publish(path,value):
    if Path(path).name=='run_manifest.json' and 'source_banks' in value:
        value={**value,'native_update_runtime_code_sha256':audit.runtime_pins()}
    return save_new(path,value)

def _lib(overrides=None):
    """Clone legacy functions only into a private namespace; original globals immutable."""
    g={**old.__dict__,'SCHEMA':SCHEMA,'SOURCE_SCHEMA':SOURCE_SCHEMA,'SPEC_SHA':SPEC_SHA,'CODE':CODE,
       'admission':admission,'accepted':accepted,'save_new':_publish}
    for name,value in old.__dict__.items():
        if isinstance(value,types.FunctionType) and value.__module__==old.__name__:
            g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
            g[name].__kwdefaults__=copy.copy(value.__kwdefaults__)
    g.update(admission=admission,accepted=accepted,save_new=_publish)
    if overrides:g.update(overrides)
    return types.SimpleNamespace(**g)

def prepare(run):return _lib().prepare(run)

def manifest(run,validate_admission=True):
    m=_lib().manifest(run,validate_admission=validate_admission)
    require(m['native_update_runtime_code_sha256']==audit.runtime_pins(),'UPDATE numeric runtime changed')
    require(m['admission']==pointer(Path(run)/'admission_manifest.json'),'Foreign admission manifest')
    return m

def bank_folder(run,ci,arm):return old.bank_folder(run,ci,arm)
def retrieval_folder(run,ci,arm):
    require(ci in range(10) and arm in NEW_ARMS,'Foreign retrieval identity')
    return Path(run).resolve()/f'retrieval/conv_{ci}/{arm}'

class EmbeddingOnlyAPI(PacedEmbeddingAPI):
    def request(self,provider,model,body,*,operation='generateContent',stage='unspecified',cache=True):
        require((provider,model,operation)==('gemini','gemini-embedding-001','batchEmbedContents') and
                stage in ('mem0_embedding_add','mem0_embedding_update'),'Only original native UPDATE/entity embeddings allowed')
        require(isinstance(body,dict) and set(body)=={'requests'} and 1<=len(body['requests'])<=32,
                'Native embedding body/batch differs')
        for r in body['requests']:
            require(set(r)=={'model','content','taskType','outputDimensionality'} and
                r['model']=='models/gemini-embedding-001' and r['taskType']=='RETRIEVAL_DOCUMENT' and
                r['outputDimensionality']==768,'UPDATE fixed embedding request differs')
        require(len(json.dumps(body,ensure_ascii=False).encode())<=131072,'Embedding bytecap differs')
        return super().request(provider,model,body,operation=operation,stage=stage,cache=cache)

def update_rows(folder,candidates):
    folder=Path(folder);paths=sorted((folder/'updates').glob('*.json'))
    require([x.name for x in paths]==[f'{i:04d}.json' for i in range(len(candidates))],
            'Incomplete or orphan native UPDATE rows')
    rows=[{k:v for k,v in sealed_read(x).items() if k!='content_sha256'} for x in paths]
    require([r['candidate'] for r in rows]==candidates,'Native UPDATE acceptance changed')
    dispatch=sorted((folder/'update_dispatch').glob('*.json'))
    require([p.name for p in dispatch]==[f'{i:04d}.json' for i in range(len(candidates))],
            'UPDATE dispatch inventory differs')
    for i,p in enumerate(dispatch):
        d=sealed_read(p);require(d['candidate']==rows[i]['candidate'] and
            d['status']=='native_update_about_to_dispatch','UPDATE write-ahead intent differs')
    return rows

def project_bank(original,after,rows):
    facts=copy.deepcopy(original['facts']);by={r['memory_id']:r for r in rows}
    require([f['id'] for f in facts]==[f['id'] for f in original['facts']],'Canonical order differs')
    for f in facts:
        if f['id'] not in by:continue
        r=by[f['id']];f.update(text=r['candidate']['replacement_text'],embedding=after['facts'][f['id']]['vector'][''])
        f['repair']={'original_text':r['candidate']['original_fact']['text'],'candidate':r['candidate'],
                     'original_birth_event':f.get('event','ADD'),'native_event':'UPDATE',
                     'original_raw_certifies_replacement':False,'semantic_support_certified':False}
        f['support_status']='unverified_source_repair'
    return facts

def materialize_bank(run,ci,arm,m=None,admit=None):
    run=Path(run).resolve();m=m or manifest(run);admit=admit or admission(run);folder=bank_folder(run,ci,arm)
    require(not folder.exists(),'Native UPDATE reentry refused')
    info=m['source_banks'][str(ci)];bank=bound(info['bank'],sealed=True);cs=accepted(admit,ci,arm)
    save_new(folder/'materialization_intent.json',{'schema':SCHEMA,'pid':os.getpid(),'created_utc':now(),
        'conversation':ci,'arm':arm,'admission':m['admission'],'candidates_sha256':digest(cs)})
    memory=None;rows=[];failure=None;phase='copy';active=None;warnings=Warnings()
    logger=logging.getLogger('mem0');logger.addHandler(warnings)
    try:
        old.copy_closed(info['native_dir'],folder/'native',info['inventory']);before=bound(info['snapshot'],sealed=True)
        if cs:
            api=EmbeddingOnlyAPI(run/f'materialization/conv_{ci}/embedding_api',retries=2,gate_path=DEFAULT_GATE,
                context={'architecture':SCHEMA,'conversation':ci,'method':arm,'phase':'qualifier_repair'})
            with old.local_bm25_context(m['resources']['snapshot'],m['resources']['files_sha256']):
                memory=old.writer.open_memory(folder/'native',api);memory.entity_store
                for i,c in enumerate(cs):
                    phase='native_update';active=c['fact_id']
                    memory.embedding_model=Embeddings(api.scoped(candidate_id=c['fact_id'],group_id=c['source_session_id']))
                    save_new(folder/f'update_dispatch/{i:04d}.json',{'candidate':c,'created_utc':now(),
                        'status':'native_update_about_to_dispatch'})
                    row=audit.update_candidate(memory,c,bank['user_id']);rows.append(row)
                    save_new(folder/f'updates/{i:04d}.json',row)
                require(not warnings.items,'Native UPDATE emitted warnings')
                old.native.close_memory(memory);memory=None
        phase='native_update_delta';active=None;after=old.snapshot(folder/'native')
        delta=audit.validate_delta(before,after,rows,bank['user_id'])
        phase='exact_native_update_replay';numeric=audit.replay_exact(info['native_dir'],before,after,rows,m['resources'])
        phase='dense_raw_provenance';by_arm={arm:rows}
        if arm=='jev':by_arm['cheap']=update_rows(bank_folder(run,ci,'cheap'),accepted(admit,ci,'cheap'))
        proof=audit.arm_embedding_proof(audit.audit_embeddings(
            run/f'materialization/conv_{ci}/embedding_api',ci,by_arm),arm)
        phase='bm25';bm25=old.compare_persisted_fact_vectors([folder/'native'],m['resources']['snapshot'],m['resources']['files_sha256'])
        require(bm25['facts_compared']==bm25['exact_matches']==len(after['facts']),'Updated BM25 differs')
        bankref=save_new(folder/'bank.json',{'schema':SCHEMA,'complete':True,'conversation':ci,'arm':arm,
            'user_id':bank['user_id'],'native_dir':str((folder/'native').resolve()),'native_inventory':old.inventory(folder/'native'),
            'original_bank':info['bank'],'admission':m['admission'],'facts':project_bank(bank,after,rows),
            'updates':rows,'gate':delta,'numeric_replay':numeric,'dense_embedding_proof':proof,'bm25':bm25,
            'protocol_sha256':digest(m['protocol']),'created_utc':now()})
        require(old.inventory(info['native_dir'])==info['inventory'],'Original native source changed')
    except Exception as ex:failure=old.safe_error(ex,phase,active)
    finally:
        if memory is not None:
            try:old.native.close_memory(memory)
            except Exception as ex:failure=old.safe_error(ex,'native_close',active)
        logger.removeHandler(warnings)
    outcome={'schema':SCHEMA,'conversation':ci,'arm':arm,'pid':os.getpid(),'finished_utc':now(),
        'status':'failed' if failure else 'complete','error':failure,'warnings':warnings.items,
        'bank':None if failure else bankref,'materialization_intent':pointer(folder/'materialization_intent.json')}
    save_new(folder/'materialization_outcome.json',outcome);return outcome

def materialize_conversation(run,ci,m,admit):
    results=[]
    for arm in NEW_ARMS:
        row=materialize_bank(run,ci,arm,m,admit);results.append(row)
        if row['status']!='complete':break
    return results

def materialize(run):
    run=Path(run).resolve();m=manifest(run);a=admission(run)
    intent=save_new(run/'materialization/controller_intent.json',{'schema':SCHEMA,'pid':os.getpid(),'created_utc':now()})
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(materialize_conversation,run,ci,m,a):ci for ci in range(10)}
        for f in as_completed(tasks):
            try:rows.extend(f.result())
            except Exception as ex:rows.append({'conversation':tasks[f],'status':'failed','error_type':type(ex).__name__})
    rows.sort(key=lambda x:(x['conversation'],x.get('arm','')))
    return save_new(run/'materialization/controller_outcome.json',{'schema':SCHEMA,'banks':rows,'intent':intent,
        'finished_utc':now(),'status':'complete' if len(rows)==20 and all(r['status']=='complete' for r in rows) else 'failed'})

def audit_banks(run,publish=True):
    run=Path(run).resolve();m=manifest(run);a=admission(run);refs={};proofs={}
    from tools.recover_gemini38_session_io_v1 import pid_alive
    outcome=sealed_read(run/'materialization/controller_outcome.json')
    require(outcome['status']=='complete' and len(outcome['banks'])==20 and
        not pid_alive(bound(outcome['intent'],sealed=True)['pid']),'All UPDATE banks must be terminal')
    require({(x['conversation'],x['arm']) for x in outcome['banks']}==
        {(ci,arm) for ci in range(10) for arm in NEW_ARMS},'UPDATE controller identities differ')
    for ci in range(10):
        info=m['source_banks'][str(ci)];before=bound(info['snapshot'],sealed=True);byarm={}
        for arm in NEW_ARMS:
            folder=bank_folder(run,ci,arm);o=sealed_read(folder/'materialization_outcome.json')
            require({k:v for k,v in o.items() if k!='content_sha256'}==next(x for x in outcome['banks']
                if x['conversation']==ci and x['arm']==arm),'UPDATE controller/per-bank terminal differs')
            require(o['status']=='complete' and not pid_alive(o['pid']) and not o['warnings'],'UPDATE owner not closed')
            b=bound(o['bank'],sealed=True);rows=update_rows(folder,accepted(a,ci,arm));byarm[arm]=rows
            require(b['updates']==rows and b['admission']==m['admission'] and b['conversation']==ci and b['arm']==arm,
                    'UPDATE bank identity/accepted row differs')
            require(old.inventory(folder/'native')==b['native_inventory'],'Updated native bytes changed')
            after=old.snapshot(folder/'native')
            require(audit.validate_delta(before,after,rows,info['user_id'])==b['gate'],'UPDATE delta differs')
            require(audit.replay_exact(info['native_dir'],before,after,rows,m['resources'])==b['numeric_replay'],
                    'Exact UPDATE replay changed')
            require(b['facts']==project_bank(bound(info['bank'],sealed=True),after,rows),'Export differs from UPDATE source')
            refs[f'{ci}/{arm}']=o['bank']
        proofs[str(ci)]=audit.audit_embeddings(run/f'materialization/conv_{ci}/embedding_api',ci,byarm)
        for arm in NEW_ARMS:
            b=bound(refs[f'{ci}/{arm}'],sealed=True)
            require(b['dense_embedding_proof']==audit.arm_embedding_proof(proofs[str(ci)],arm),
                'UPDATE per-arm raw embedding provenance differs')
    value={'schema':SCHEMA,'status':'passed','admission':m['admission'],'banks':refs,'new_banks':20,
        'source_original_facts':1637,'semantic_support_certified':False,'model_calls':0,'embedding_audits':proofs}
    return save_new(run/'materialization/index_admission.json',value) if publish else value

def _retrieval_lib():return _lib({'bank_folder':retrieval_folder})
def retrieve_conversation(run,ci,m,gate,jobs,entries):
    for arm in NEW_ARMS:
        b=bound(gate['banks'][f'{ci}/{arm}'],sealed=True)
        old.copy_closed(b['native_dir'],retrieval_folder(run,ci,arm)/'native',b['native_inventory'])
    return _retrieval_lib().retrieve_conversation(run,ci,m,gate,jobs,entries)

def retrieve(run):
    run=Path(run).resolve();m=manifest(run);gate=sealed_read(run/'materialization/index_admission.json')
    require(gate['status']=='passed' and len(gate['banks'])==20,'UPDATE index admission absent')
    source=bound(m['protocol']['frozen_source']);jobs=old.old_jobs(source)
    pack=old.packs.verify_pack(m['embedding_cache_pack'],check_sources=True)
    save_new(run/'retrieval_execution_intent.json',{'schema':SCHEMA,'pid':os.getpid(),'created_utc':now(),'model_calls':0})
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(retrieve_conversation,run,ci,m,gate,[j for j in jobs if j['conversation']==ci],
            [e for e in pack['entries'] if e['conversation']==ci]):ci for ci in range(10)}
        for f in as_completed(tasks):
            try:rows.extend(f.result())
            except Exception as ex:rows.append({'conversation':tasks[f],'status':'failed','error_type':type(ex).__name__})
    rows.sort(key=lambda x:(x['conversation'],x.get('arm','')))
    return save_new(run/'retrieval_outcome.json',{'schema':SCHEMA,'banks':rows,'model_calls':0,'finished_utc':now(),
        'status':'complete' if len(rows)==20 and all(r['status']=='complete' for r in rows) else 'failed'})

def _source_value(run,m,gate):
    lib=_retrieval_lib();source=bound(m['protocol']['frozen_source']);jobs=old.old_jobs(source)
    lib.validate_retrieval_terminal(run,m,gate,jobs)
    p=m['protocol']
    value={'schema':SOURCE_SCHEMA,'status':'ready','arms':list(ARMS),'questions_per_arm':1540,
        'jobs':lib.construct_jobs(run,m,gate,jobs),'qualifier_repair_protocol':p['research_protocol'],
        'selector_manifest':pointer(Path(run)/'run_manifest.json'),'admission_manifest':m['admission'],
        'materialization_manifest':pointer(Path(run)/'materialization/run_manifest.json'),
        'index_admission':pointer(Path(run)/'materialization/index_admission.json'),
        'retrieval_outcome':pointer(Path(run)/'retrieval_outcome.json'),'protocol_sha256':digest(p),'code_sha256':m['code_sha256'],
        'official_metric_provenance':source['official_metric_provenance'],'source_dataset_sha256':source['source_dataset_sha256'],
        'selection_counts':{arm:{'jobs':1540} for arm in ARMS}}
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):value[f]=p[f]
    return value

def seal(run):
    run=Path(run).resolve();m=manifest(run);gate=sealed_read(run/'materialization/index_admission.json')
    require(audit_banks(run,publish=False)=={k:v for k,v in gate.items() if k!='content_sha256'},'Index gate replay differs')
    return save_new(run/'selected_inputs.json',_source_value(run,m,gate))

def validate_source(path,fresh=False):
    path=Path(path).resolve();before=sha(path);source=sealed_read(path);run=path.parent;m=manifest(run)
    gate=sealed_read(run/'materialization/index_admission.json')
    require(audit_banks(run,publish=False)=={k:v for k,v in gate.items() if k!='content_sha256'},'UPDATE full index replay differs')
    require({k:v for k,v in source.items() if k!='content_sha256'}==_source_value(run,m,gate),
            'Full selected input/request/lineage replay differs')
    for f in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest'):bound(source[f])
    require(sha(path)==before,'Selected source changed during validation');return source

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','materialize','audit','retrieve','seal','check'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    funcs={'prepare':prepare,'materialize':materialize,'audit':audit_banks,'retrieve':retrieve,'seal':seal,
        'check':lambda run:{'jobs':len(validate_source(Path(run)/'selected_inputs.json')['jobs']),'model_calls':0}}
    print(json.dumps(funcs[a.command](a.run_dir),ensure_ascii=False))
if __name__=='__main__':main()
