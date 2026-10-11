"""Uniform closed20 C review/export. No native repair, model calls, or implicit retry."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import review_regroup_refinement_banks_v2 as base
from tools import audit_boundary_variants_banks_v1 as cstrict
from tools import run_boundary_variants_parallel_v1 as parallel
from tools import run_boundary_variants_v1 as boundary

old=SimpleNamespace(**vars(base.old))
ROOT=base.ROOT
ARMS=tuple(cstrict.ARM_METHODS)
SCHEMA='boundary-variants-uniform20-review-v2'
EXPORT_SCHEMA='boundary-variants-reviewed-export-v2'
BANK_SCHEMA='boundary-variants-reviewed-bank-v2'
STATUS=base.STATUS
ERROR=base.ERROR
DEPENDENCIES=tuple(dict.fromkeys((*base.DEPENDENCIES,*cstrict.CODE,
    'tools/review_boundary_variants_banks_v2.py')))
# Must be filled only after the new independent proof is sealed and reviewed.
# No historical c3 exception is inherited; empty means fail closed.
SUPPORTED_FAILURES={(3,'c1_multiscale','599a163b102f70113a8271901601ee76e92426b6f7bf8fad0dbb798fc7418908'):
    '235f64910c386bcf7488759c341991075ea84c3fc22e49fccd51ad7c36cf505c',
    (3,'c2_neighbour','f27eb8e8f8c079125ed264787f53916da206886d8928929ed231ebdcc16095d8'):
    '1ceaeb59bc2fcd89b2d1445d7da24653c97b7b7bdc4203a72e467828599ccbe8'}
SUPPORTED_ORPHANS={(3,'c1_multiscale','599a163b102f70113a8271901601ee76e92426b6f7bf8fad0dbb798fc7418908'):
    ['7f386c3c-6437-462c-8e7b-2c58752fdcec'],
    (3,'c2_neighbour','f27eb8e8f8c079125ed264787f53916da206886d8928929ed231ebdcc16095d8'):
    ['6b65c2c9-fd68-428e-9096-468b82daf71c']}
digest,verify=base.digest,base.verify
require=base.require
forbid_network=base.forbid_network

def pointer(path):
    path=Path(path)
    return old.pointer(path.with_name('write_status.json') if path.name=='write_outcome.json' else path)

_checked=base._checked
_folder_inventory=base._folder_inventory
_isolated_overwrite_proof=base._isolated_overwrite_proof

def _population(rows):
    require(len(rows)==20 and {(r['conversation'],r['arm']) for r in rows}==
            {(ci,arm) for ci in range(10) for arm in ARMS},'Exactly current20 C banks required')

def _failed_state(state):
    require(state.get('status')=='failed' and state.get('error',{}).get('type')=='ValueError'
            and state.get('error',{}).get('message')==ERROR,'Unknown failure class')

def closed_controllers(runs,targets):
    """Closed parent may report failed, but every requested arm needs its own sealed terminal."""
    parents={Path(r).resolve().parents[1] for r in runs.values()}
    require(len(parents)==1,'One exact C parent required')
    parent=next(iter(parents));observed={};evidence=[]
    target={(t['conversation'],t['arm']):t for t in targets}
    for folder in sorted((parent/'parallel_controller_v1').glob('write_*')):
        mpath=folder/'manifest.json';cm=verify(mpath);policy=cm['policy']
        require(cm.get('schema')==parallel.VERSION and policy.get('stage')=='write'
                and cm.get('parent')==boundary.pointer(parent/'manifest.json')
                and cm.get('code_sha256')=={p:old.sha(ROOT/p) for p in parallel.FILES},
                'Controller source/policy/code differs')
        require(set(policy['arms'])<=set(ARMS) and set(policy['conversations'])<=set(range(10)),
                'Foreign controller scope')
        state=old.read(folder/'parent_state.json');mp=old.pointer(mpath)
        require(state.get('status') in ('complete','failed') and state.get('finished_utc')
                and state.get('controller_manifest')==mp and state.get('stage')=='write'
                and state.get('requested_conversations')==policy['conversations']
                and state.get('arms')==policy['arms']
                and set(state.get('outcomes',{}))=={str(ci) for ci in policy['conversations']},
                'Entire write controller must be closed terminal')
        try: live=parallel.process_identity(state['pid'])
        except RuntimeError: live=None
        require(live!=state.get('parent_process_identity'),'Matching controller remains live')
        workers=[];all_complete=True
        for ci in policy['conversations']:
            outcome=state['outcomes'][str(ci)];actual=[]
            require(outcome.get('conversation')==ci and outcome.get('finished_utc')
                    and len(outcome.get('arms',[]))==len(policy['arms']),
                    'Parent outcome lacks exact per-arm terminals')
            for arm in policy['arms']:
                key=ci,arm;require(key not in observed,'Duplicate controller ownership')
                t=target[key];job=folder/'conversations'/f'conv_{ci}'/arm
                term=verify(job/'terminal.json');worker=old.read(job/'worker_state.json');intent=verify(job/'intent.json')
                status=t['state']['status'];outputs=({'write_status':old.pointer(t['folder']/'write_status.json'),
                         'bank':old.pointer(t['folder']/'bank.json')} if status=='complete' else {})
                require(term.get('schema')==parallel.VERSION and term.get('controller_manifest')==mp
                        and (term.get('stage'),term.get('conversation'),term.get('arm'))==('write',ci,arm)
                        and term.get('status')==status and term.get('finished_utc') and term.get('outputs')==outputs
                        and worker.get('status')==status and worker.get('terminal')==old.pointer(job/'terminal.json')
                        and intent.get('controller_manifest')==mp and intent.get('nonce')==state.get('nonce')
                        and (intent.get('stage'),intent.get('conversation'),intent.get('arm'))==('write',ci,arm),
                        'Per-arm intent/terminal/worker source binding differs')
                terminal_body={k:v for k,v in term.items() if k!='content_sha256'}
                require([v for v in outcome['arms'] if (v.get('conversation'),v.get('arm'))==key]==[terminal_body],
                        'Parent arm terminal differs')
                require(t['state']['finished_utc']<=term['finished_utc']<=state['finished_utc'],
                        'Writer/worker/controller closure order differs')
                if status=='failed':_failed_state(t['state']);all_complete=False
                actual.append(status);observed[key]=old.pointer(job/'terminal.json')
                workers.append({'conversation':ci,'arm':arm,'terminal':observed[key],
                    'intent':old.pointer(job/'intent.json'),'worker_state':old.pointer(job/'worker_state.json')})
            require(outcome.get('status')==('complete' if all(s=='complete' for s in actual) else 'failed'),
                    'Parent conversation status differs')
        require(state['status']==('complete' if all_complete else 'failed'),'Parent aggregate status differs')
        evidence.append({'manifest':mp,'parent_state':old.pointer(folder/'parent_state.json'),'workers':workers})
    require(set(observed)==set(target),'All20 sealed controller terminals required')
    return {'mode':'parallel_closed_exact20_known_failures_disclosed','controllers':evidence}

def terminal(runs):
    require(set(runs)==set(ARMS),'Both exact C writer runs required');targets=[]
    for arm in ARMS:
        run=Path(runs[arm]).resolve();m=cstrict.writer.manifest(run)
        require(m['arms']==[arm] and m.get('boundary_variant',{}).get('arm')==arm,'Writer arm differs')
        for key in ('parent','global'):
            require(boundary.pointer(m['boundary_variant'][key]['path'])==m['boundary_variant'][key],
                    'Boundary/global pointer differs')
        for ci in range(10):
            folder=run/f'conv_{ci}'/arm;s=old.read(folder/'write_status.json')
            require(s.get('status') in ('complete','failed') and s.get('finished_utc')
                    and (s.get('conversation'),s.get('arm'),s.get('protocol_sha256'))==(ci,arm,m['protocol_sha256']),
                    'All20 exact source banks must be terminal before inspection')
            require(s.get('completed_groups')==s.get('expected_groups') and s.get('expected_groups',0)>0
                    and s.get('facts',0)>=20,'Partial facts never eligible')
            if s['status']=='failed':_failed_state(s)
            targets.append({'conversation':ci,'arm':arm,'run':run,'folder':folder,'manifest':m,
                'state':s,'outcome_value':s,'writer_terminal_kind':'write_status_only'})
    _population(targets);closed_controllers(runs,targets);return targets

def _failure_binding(path,target):
    ref=old.pointer(path);s=target['state'];m=target['manifest'];key=target['conversation'],target['arm'],m['protocol_sha256']
    require(SUPPORTED_FAILURES.get(key)==ref['sha256'],'Current independently pinned failure proof required')
    p=verify(path)
    require(p.get('schema')=='boundary-writer-failure-independent-review-v1'
            and p.get('status')=='closed_complete_facts_failed_entity_index_reviewed'
            and (p.get('conversation'),p.get('arm'))==key[:2]
            and p.get('source_manifest')==old.pointer(target['run']/'run_manifest.json')
            and p.get('write_status')==old.pointer(target['folder']/'write_status.json'),
            'Failure proof identity/class differs')
    require(p['groups']==p['expected_groups']==s['expected_groups'] and p['facts']==s['facts']
            and p.get('original_write_status')=='failed' and p.get('strict_passed') is False
            and p.get('source_generation_persistence_passed') is True and p.get('qa_gate_passed') is False
            and p.get('model_calls')==0 and p.get('source_mutations')==0 and p.get('qa_gold_used') is False,
            'Proof scope differs')
    for name in ('closed_bank_identity','group_source_date_raw_persistence','facts_dense_sparse_history',
                 'existing_entity_dense_links_valid','same_canonical_stale_payload_overwrite_proved',
                 'sqlite_read_only_transactions','no_source_change'):
        require(p.get('checks',{}).get(name) is True,'Complete independent source/required-index proof required')
    require(p.get('orphan_fact_ids')==SUPPORTED_ORPHANS.get(key)
            and p.get('cause_evidence',{}).get('verified') is True
            and p['cause_evidence'].get('mechanism')=='same_canonical_stale_payload_overwrite',
            'Unique current metadata defect required')
    controller_terminal=target['run'].parents[1]/'parallel_controller_v1/write_expansion/conversations'/f'conv_{key[0]}'/key[1]/'terminal.json'
    require(p.get('controller_terminal')==old.pointer(controller_terminal)
            and p.get('group_inputs')==old.pointer(target['folder']/'group_inputs.json')
            and Path(p.get('native_dir','')).resolve()==(target['folder']/'native').resolve()
            and p.get('original_error')==s.get('error'),'Current failure terminal/source binding differs')
    for name,h in p['code_sha256'].items():require(old.sha(ROOT/name)==h,'Independent proof code drift')
    for name,h in p['source_file_hashes'].items():require(old.sha(name)==h,'Proof source drift')
    require(old.native_inventory(target['folder']/'native')==p['native_inventory'],'Proof native inventory drift')
    require(not (target['folder']/'bank.json').exists(),'Failed original unexpectedly gained a bank')
    return p,ref

def _proof_inspection(proof):
    inspection=proof.get('source_date_raw_inspection',proof.get('strict_body_inspection'))
    require(isinstance(inspection,dict) and inspection.get('source_generation_persistence_passed') is True
            and inspection.get('writer_prompt_reconstructed_and_raw_verified') is True
            and inspection.get('strict_passed') is False
            and inspection.get('orphan_fact_ids')==proof['orphan_fact_ids']
            and inspection.get('groups')==proof['groups'] and inspection.get('facts')==proof['facts'],
            'Complete failure strict-body source inspection required')
    return inspection

def _complete_audit(run,convs):
    """Exact C strict inspection body, independent already-reviewed terminal scheduling."""
    run=Path(run);m=cstrict.writer.manifest(run);arm=m['boundary_variant']['arm'];data=old.read(ROOT/'data/locomo10.json')
    require(old.sha(ROOT/'data/locomo10.json')==m['dataset_sha256'],'Dataset drift')
    facade=SimpleNamespace(**vars(old));facade.pointer=pointer
    env=dict(cstrict.strict.inspect_bank.__globals__,index_audit=facade)
    fn=FunctionType(cstrict.strict.inspect_bank.__code__,env,cstrict.strict.inspect_bank.__name__,cstrict.strict.inspect_bank.__defaults__)
    prompt_env=cstrict.strict.prompt_environment();extractor,nlp=old.load_extractor();rows=[]
    for ci in convs:
        f=run/f'conv_{ci}'/arm;s=old.read(f/'write_status.json');b=verify(f/'bank.json')
        require(s['status']=='complete' and s['bank']==old.pointer(f/'bank.json'),'Strict bank source differs')
        rows.append(fn((ci,arm,f,s,s,b),m,data[ci]['conversation'],prompt_env,extractor,nlp))
    resource=m['boundary_variant']['resources']
    bm=cstrict.strict.compare_persisted_fact_vectors([Path(r['native_dir']) for r in rows],resource['snapshot'],resource['files_sha256'])
    require(bm['facts_compared']==bm['exact_matches']==sum(r['facts'] for r in rows),'Required fact BM25 incomplete')
    return {'banks':rows,'strict_passed':all(r['strict_passed'] for r in rows),'source_file_hashes':dict(old.HASHES),'offline_bm25_equivalence':bm}

# Reuse old review/check/failure projection bodies with only private bindings.
strict=SimpleNamespace(audit=_complete_audit)
_validate_row_body=FunctionType(base._validate_row.__code__,globals(),'validate_row_body',base._validate_row.__defaults__)
_audit_body=FunctionType(base.audit.__code__,globals(),'audit_body',base.audit.__defaults__)
_failure_facts_body=FunctionType(base._failure_facts.__code__,globals(),'failure_facts_body',base._failure_facts.__defaults__)
_check_review_body=FunctionType(base.check_review.__code__,globals(),'check_review_body',base.check_review.__defaults__)

def _failure_facts(target,proof):
    # Reuse the exact video-games/video-game overwrite algorithm ONLY for this
    # newly pinned proof whose canonical inputs were independently established.
    facts=_failure_facts_body(target,proof)
    require(digest(facts)==proof['fact_projection_sha256'],'Failed fact projection drift')
    resource=target['manifest']['boundary_variant']['resources']
    bm=cstrict.strict.compare_persisted_fact_vectors([target['folder']/'native'],resource['snapshot'],resource['files_sha256'])
    require(bm['facts_compared']==bm['exact_matches']==len(facts),'Failed required BM25 incomplete')
    return facts

def _validate_row(row,target):
    _validate_row_body(row,target)
    require(row.get('writer_terminal_kind')=='write_status_only'
            and row['outcome']==row['write_status'],'No invented write_outcome schema permitted')
    if row['source_kind']=='reviewed_export':
        proof,_=_failure_binding(row['proof']['path'],target)
        require(row['facts_projection_sha256']==proof['fact_projection_sha256'], 'Failed review projection differs from pinned proof')

_original_save=base.old.save
def _save(path,value):
    if value.get('schema')==SCHEMA:
        value=copy.deepcopy(value)
        for row in value['banks']:row['writer_terminal_kind']='write_status_only'
        value['writer_scheduling']=closed_controllers(value['runs'],terminal(value['runs']))
        value['strict_v1_policy_passed']=False
        value['export_policy']='uniform20_metadata_only_original_status_and_native_preserved'
        mixed=[]
        for row in value['banks']:
            if row['source_kind']=='strict_complete':
                report=_checked(row['audits']['source_date'][0])
                inspected=next(x for x in report['banks'] if (x['conversation'],x['arm'])==(row['conversation'],row['arm']))
            else:inspected=_proof_inspection(_checked(row['proof']))
            mixed.extend({'conversation':row['conversation'],'arm':row['arm'],'group_id':x['group_id'],
                'journal':x['journal'],'raw':x['raw']} for x in inspected['semantic_review_targets'] if x['mixed_dates'])
        value['source_review_scope']={'banks':20,'groups':sum(x['groups'] for x in value['banks']),
            'facts':sum(x['facts'] for x in value['banks']),'mixed_date_groups':len(mixed),
            'mixed_date_source_raw_bindings':mixed,'all_groups_original_source_date_effective_prompt_raw_add_reviewed':True,
            'semantic_correctness_verified':False}
    return _original_save(path,value)
old.save=_save

def _locks(runs):
    stack=ExitStack()
    try:
        for arm in ARMS:stack.enter_context(cstrict.source_locks(runs[arm]))
        return stack
    except BaseException:stack.close();raise

def audit(runs,failure_audits,output):
    with _locks(runs):return _audit_body(runs,failure_audits,output)

def check_review(path):
    review=verify(path)
    with _locks(review['runs']):return _check_review_body(path)

def export(review_path,destination):
    """Export metadata for ALL20, preserving UUID/text/date/vectors and original terminal state."""
    dest=Path(destination).resolve();require(dest.parent==(ROOT/'runs').resolve() and not dest.exists(),'Fresh direct runs namespace required')
    review=verify(review_path)
    with _locks(review['runs']),base.forbid_network():
        review=check_review(review_path);ref=old.pointer(review_path);dest.mkdir();old.save(dest/'export_intent.json',{'schema':EXPORT_SCHEMA,'source_review':ref,'native_index_policy':'original_unrepaired'})
        exported=[]
        for row in review['banks']:
            ci,arm=row['conversation'],row['arm'];n=Path(row['native_dir']);m=_checked(row['source_manifest'])
            if row['source_kind']=='strict_complete':bank=copy.deepcopy(verify(row['bank']['path']))
            else:
                target=next(t for t in terminal(review['runs']) if (t['conversation'],t['arm'])==(ci,arm))
                proof,_=_failure_binding(row['proof']['path'],target);facts=_failure_facts(target,proof)
                js=sorted([(old.read(p),p) for p in (n/'group_audit').rglob('*.json')],key=lambda x:x[0]['sequence'])
                bank={'protocol_sha256':m['protocol_sha256'],'conversation':ci,'arm':arm,'complete':True,
                    'user_id':f'dated_regroup_c{ci}_{arm}','facts':facts,'group_inputs':row['group_inputs'],
                    'journal_files':[old.pointer(p) for j,p in js],'native_dir':str(n),'source_dataset_sha256':m['dataset_sha256'],
                    'fact_support':'unverified_group_input_scope','explicit_empty_group_ids':[j['group_id'] for j,p in js if j.get('explicit_empty_extraction')],
                    'chunk_embedding_group_ids':[j['group_id'] for j,p in js if j.get('group_context_embedding',{}).get('triggered')],
                    'native_index_audit':{'orphan_fact_ids':row['orphan_fact_ids'],'limitations':row['native_index_limitations']}}
            bank.pop('content_sha256',None);require(digest(bank['facts'])==row['facts_projection_sha256'],'Exact fact projection changed')
            bank.update(schema=BANK_SCHEMA,source_kind=row['source_kind'],original_write_status=row['original_write_status'],
                source_inventory=row['native_inventory'],source_review=ref,native_index_policy='original_unrepaired')
            folder=dest/f'conv_{ci}'/arm;bp=old.save(folder/'bank.json',bank)
            record={'schema':EXPORT_SCHEMA,'conversation':ci,'arm':arm,'source_kind':row['source_kind'],
                'original_write_status':row['original_write_status'],'reviewed_export_status':'complete','bank':bp,
                'original_write_status_ref':row['write_status'],'original_outcome_ref':row['outcome'],
                'source_manifest':row['source_manifest'],'source_review':ref,'uniform_twenty_bank_review':ref,
                'review_row_sha256':digest(row),'native_dir':str(n),'native_inventory':row['native_inventory'],
                'native_index_policy':'original_unrepaired','orphan_fact_ids':row['orphan_fact_ids'],
                'facts':row['facts'],'groups':row['groups'],'model_calls':0,'source_mutations':0,'native_bytes_copied':0,
                'qa_gate_passed':False,'export_tool':old.pointer(__file__)}
            ep=old.save(folder/'reviewed_export.json',record);exported.append({'conversation':ci,'arm':arm,'bank':bp,'reviewed_export':ep})
        _population(exported);check_review(review_path)
        return old.save(dest/'export_outcome.json',{'schema':EXPORT_SCHEMA,'status':'reviewed_metadata_exports_complete',
            'source_review':ref,'exports':exported,'source_mutations':0,'native_bytes_copied':0,'model_calls':0})

def check_export_bank(binding_or_dir):
    """Static metadata/source revalidation; no SQLite opens or source changes."""
    if isinstance(binding_or_dir,dict):_checked(binding_or_dir);path=Path(binding_or_dir['path'])
    else:path=Path(binding_or_dir);path=path/'reviewed_export.json' if path.is_dir() else path
    record=verify(path);_checked(record['source_review']);review=check_review(record['source_review']['path'])
    row=next(r for r in review['banks'] if (r['conversation'],r['arm'])==(record['conversation'],record['arm']))
    require(record.get('schema')==EXPORT_SCHEMA and record.get('source_kind')==row['source_kind']
            and record.get('original_write_status')==row['original_write_status'] and record.get('reviewed_export_status')=='complete'
            and record.get('source_review')==record.get('uniform_twenty_bank_review')
            and record.get('review_row_sha256')==digest(row) and record.get('native_index_policy')=='original_unrepaired',
            'Uniform export identity/status differs')
    require(record['bank']==old.pointer(path.parent/'bank.json') and record['export_tool']==old.pointer(__file__), 'Export file/tool differs')
    for field,key in [('original_write_status_ref','write_status'),('original_outcome_ref','outcome'),('source_manifest','source_manifest')]:
        require(record[field]==row[key],'Original terminal lineage differs')
    bank=verify(record['bank']['path']);m=_checked(row['source_manifest']);n=Path(row['native_dir'])
    require(bank.get('schema')==BANK_SCHEMA and bank.get('complete') is True
            and (bank.get('conversation'),bank.get('arm'),bank.get('protocol_sha256'))==(row['conversation'],row['arm'],m['protocol_sha256'])
            and bank.get('source_dataset_sha256')==m['dataset_sha256'] and bank.get('group_inputs')==row['group_inputs']
            and bank.get('source_review')==record['source_review'] and bank.get('original_write_status')==row['original_write_status']
            and bank.get('source_kind')==row['source_kind'] and bank.get('native_index_policy')=='original_unrepaired', 'Export bank scope differs')
    js=[_checked(p) for p in bank['journal_files']]
    require(len(js)==row['groups'] and [j['sequence'] for j in js]==list(range(row['groups'])) and all(j['status']=='success' for j in js)
            and [f for j in js for f in j['fact_records']]==[{k:v for k,v in f.items() if k!='embedding'} for f in bank['facts']]
            and len(bank['facts'])==row['facts'] and digest(bank['facts'])==row['facts_projection_sha256'], 'Export original fact text/date/vector projection differs')
    require(Path(bank['native_dir']).resolve()==n.resolve()==Path(record['native_dir']).resolve()
            and record['native_inventory']==bank['source_inventory']==row['native_inventory']==old.native_inventory(n)
            and record['orphan_fact_ids']==row['orphan_fact_ids'],'Original native inventory/defect differs')
    if row['source_kind']=='strict_complete':
        original=verify(row['bank']['path'])
        for key in ('facts','native_index_audit','explicit_empty_group_ids','chunk_embedding_group_ids','journal_files','fact_support'):
            require(bank.get(key)==original.get(key),'Strict export changed original metadata')
    else:
        require(bank.get('native_index_audit',{}).get('orphan_fact_ids')==row['orphan_fact_ids']
                and bank['native_index_audit'].get('limitations')==row['native_index_limitations'], 'Failed defect disclosure erased')
    aggregate=verify(path.parent.parent.parent/'export_outcome.json');_population(aggregate['exports'])
    require(aggregate.get('schema')==EXPORT_SCHEMA and aggregate.get('status')=='reviewed_metadata_exports_complete'
            and aggregate.get('source_review')==record['source_review']
            and any(v['reviewed_export']==old.pointer(path) and v['bank']==record['bank'] for v in aggregate['exports']), 'Uniform20 export aggregate differs')
    return {'bank':bank,'native':n,'uniform_row':copy.deepcopy(row),'export':record,'source_kind':row['source_kind'],
            'review':{'bank':record['bank'],'native_inventory':row['native_inventory'],'source_kind':row['source_kind']}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('audit','check','export','check-export'))
    p.add_argument('--c1-run',type=Path);p.add_argument('--c2-run',type=Path);p.add_argument('--failure-audit',action='append',default=[])
    p.add_argument('--review',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args()
    if a.stage=='audit':
        proofs={}
        for item in a.failure_audit:
            ci,arm,path=item.split(':',2);key=int(ci),arm;require(key not in proofs,'Duplicate failure proof');proofs[key]=Path(path)
        result=audit(dict(zip(ARMS,(a.c1_run,a.c2_run))),proofs,a.review)
    elif a.stage=='check':result={'valid':bool(check_review(a.review))}
    elif a.stage=='export':result=export(a.review,a.destination)
    else:result={'valid':bool(check_export_bank(a.destination))}
    print(result)

if __name__=='__main__':main()
