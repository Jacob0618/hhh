"""Offline content-addressed evidence archive and post-cleanup QA resolver.

This separate lifecycle tool never calls a model, loads credentials, changes a
run, or removes files. The companion PowerShell script is the only deleter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tools import run_five_baselines_amem_native_v3 as continuation
from tools import run_five_baselines_gemini25_v1 as base
from tools.baselines_common_v1 import read, sha, save_new, now

ROOT=continuation.ROOT
ROOTS={'v1':continuation.ANCESTOR,'v2':continuation.SOURCE,'v3':continuation.NEW}
TOOL_FILES=('tools/archive_amem_continuation_v3.py','tools/cleanup_amem_continuation_v3.ps1')
SCHEMA='amem-cross-root-content-archive-v1'
PLAN_SCHEMA='amem-cross-root-cleanup-plan-v1'


def output_dir():return ROOTS['v3']/'results/continuation_archive'


def digest_bytes(raw):return hashlib.sha256(raw).hexdigest()


def stopped():
    if os.name!='nt':raise ValueError('Production process validation requires Windows')
    command=("$p=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne "+str(os.getpid())+" -and $_.Name -match '^python(?:w|[0-9]+(?:\\.[0-9]+)*)?\\.exe$' -and "
             "$_.CommandLine -match 'run_five_baselines[\\w]*(?:\\.py)?(?:[\"\\s]|$)' -and $_.CommandLine -match '(?:^|[\"\\s])run(?:[\"\\s]|$)' }; if ($p) { exit 23 }")
    if subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True).returncode:
        raise ValueError('Experiment controller active or process validation unavailable')


def safe_path(path):
    p=Path(path).absolute();root=ROOT.absolute()
    if not p.is_relative_to(root):raise ValueError('Foreign archive path: '+str(p))
    for part in (root,*reversed(p.relative_to(root).parents)):
        candidate=part if part==root else root/part
        if candidate.exists() and getattr(candidate.lstat(),'st_file_attributes',0) & 0x400:raise ValueError('Reparse point: '+str(candidate))
    if p.exists() and getattr(p.lstat(),'st_file_attributes',0)&0x400:raise ValueError('Reparse point: '+str(p))
    if p.resolve()!=p:raise ValueError('Path alias/traversal is forbidden')
    return p


def file_inventory(folder):
    safe_path(folder)
    result={}
    for p in sorted(Path(folder).rglob('*')):
        safe_path(p)
        if p.is_file():result[str(p.resolve())]=sha(p)
    return result


def disposable(path):
    for folder in ROOTS.values():
        native=folder/'work/native'
        if path.is_relative_to(native):
            parts=path.relative_to(native).parts
            if len(parts)>=3 and parts[1]=='sessions':return 'duplicate native per-session snapshot; full memory update journal retained'
            if len(parts)>=3 and parts[1]=='cache':return 'reproducible local MiniLM vector/index cache'
    return None


def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def verify_binding(value):
    p=safe_path(value['path'])
    if not p.is_file() or sha(p)!=value['sha256']:raise ValueError('Bound evidence changed: '+str(p))
    return p


def verify_result_review(path, *, live_inventory=True):
    review=read(safe_path(path));summary_path=ROOTS['v3']/'results/summary.json';summary=read(summary_path)
    if review.get('status')!='passed' or review.get('summary_sha256')!=sha(summary_path):raise ValueError('Passed independent full result review required')
    effective=summary.get('effective',{})
    if summary.get('audit_status')!='no_invalid_artifacts' or any(effective.get(k)!=1540 for k in ('slots','reader_valid','judge_valid')):
        raise ValueError('All 1540 effective Reader and Judge stages must be valid')
    expected=review.get('reviewed_run_inventory')
    if (Path(path).resolve().is_relative_to(ROOTS['v3'].resolve()) or not isinstance(expected,dict) or not expected
            or any(not Path(p).is_absolute() or str(Path(p).resolve())!=p or not Path(p).is_relative_to(ROOTS['v3'])
                   or not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for p,h in expected.items())):
        raise ValueError('Independent result review must bind a nonempty complete absolute run inventory and live outside the run')
    if live_inventory and file_inventory(ROOTS['v3'])!=expected:raise ValueError('Current run differs from independently reviewed full inventory')
    return review


def verify_tool_review(path):
    review=read(safe_path(path))
    if review.get('status')!='passed' or any(review.get('code_sha256',{}).get(p)!=sha(ROOT/p) for p in TOOL_FILES):
        raise ValueError('Independent archive/cleanup tool review must bind both exact source files')


def closure(initial):
    """Follow only explicit SHA-bound references, never arbitrary text paths."""
    records={};queue=list(initial)
    pairs=(('path','sha256'),('review_path','review_sha256'),('raw_path','raw_sha256'),
           ('source_path','source_sha256'),('native_call_path','native_call_sha256'))
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            for pk,hk in pairs:
                if isinstance(value.get(pk),str) and isinstance(value.get(hk),str):
                    queue.append((value[pk],value[hk]))
            for key in ('code_sha256','upstream_sha256','evidence_sha256','files_sha256'):
                for p,h in value.get(key,{}).items():queue.append((str(ROOT/p),h))
            for item in value.values():visit(item)
    while queue:
        path,expected=queue.pop();p=safe_path(path);name=str(p)
        if name in records:
            if records[name]!=expected:raise ValueError('Conflicting bound file SHA')
            continue
        if sha(p)!=expected:raise ValueError('Closure SHA differs: '+name)
        records[name]=expected
        if p.suffix=='.json':visit(read(p))
    return records


def verify_native_seal():
    """Revalidate current writer/API evidence, not merely its saved summary."""
    from tools import run_five_baselines_amem_native_v3_admission_v2 as frozen
    new=ROOTS['v3'];m=read(new/'run_manifest.json')
    for p,h in {**m['code_sha256'],**m['upstream_sha256']}.items():verify_binding({'path':str(ROOT/p),'sha256':h})
    proof=read(verify_binding(m['technical_continuation']['proof']))
    audit_requests=clone(frozen.audit_new_requests,NEW=new,SOURCE=ROOTS['v2'],ANCESTOR=ROOTS['v1'],_PROOF=proof)
    audit_native=clone(frozen.audit_native_outputs,NEW=new)
    if audit_requests()!=read(new/'results/raw_request_audit.json'):raise ValueError('Current API evidence differs from sealed raw audit')
    if audit_native(m)!=read(new/'results/native_output_audit.json'):raise ValueError('Current native writer evidence differs from sealed native audit')


def archive(result_review,tool_review):
    stopped();verify_result_review(result_review);verify_tool_review(tool_review)
    out=output_dir();plan_path=ROOTS['v3']/'cleanup_plan_native_v3.json'
    if out.exists() or plan_path.exists():raise FileExistsError('Archive namespace already exists; no implicit replacement')
    verify_native_seal()
    all_files={};work_inventories={}
    for name,folder in ROOTS.items():
        if not (folder/'work').is_dir():raise ValueError('All three original work roots required before archive')
        current=file_inventory(folder);all_files.update(current)
        work_inventories[name]={p:h for p,h in current.items() if Path(p).is_relative_to(folder/'work')}
    m=read(ROOTS['v3']/'run_manifest.json')
    initial=[(str(Path(result_review)),sha(result_review)),(str(Path(tool_review)),sha(tool_review))]
    initial.extend((str(folder/'run_manifest.json'),sha(folder/'run_manifest.json')) for folder in ROOTS.values())
    all_files.update(closure(initial))
    dataset=ROOT/'data/locomo10.json'
    if m.get('dataset_sha256')!=sha(dataset):raise ValueError('Original dataset changed')
    all_files[str(dataset.resolve())]=sha(dataset)
    frozen_config=ROOT/'configs/sdk20_full1540_locked_v1.json'
    all_files[str(frozen_config.resolve())]=sha(frozen_config)
    for p in TOOL_FILES:all_files[str((ROOT/p).resolve())]=sha(ROOT/p)
    proof=read(m['technical_continuation']['proof']['path'])
    for name,key in (('v1','ancestor_inventory'),('v2','source_inventory')):
        expected={str((ROOTS[name]/p).resolve()):h for p,h in proof[key].items()}
        observed={p:h for p,h in all_files.items() if Path(p).is_relative_to(ROOTS[name])}
        if observed!=expected:raise ValueError('Archived original source differs from terminal proof')
    records=[]
    for p,h in sorted(all_files.items()):
        reason=disposable(Path(p));record={'path':p,'sha256':h,'size':Path(p).stat().st_size,'retained':reason is None}
        if reason:record['disposal_reason']=reason
        else:record['member']='blobs/'+h
        records.append(record)
    manifest={'schema':SCHEMA,'created_utc':now(),'roots':{k:str(v.resolve()) for k,v in ROOTS.items()},
              'summary':binding(ROOTS['v3']/'results/summary.json'),'result_review':binding(result_review),
              'tool_review':binding(tool_review),'proof':m['technical_continuation']['proof'],
              'records':records,'work_inventories':work_inventories,
              'code_sha256':{p:sha(ROOT/p) for p in TOOL_FILES}}
    out.mkdir(parents=True,exist_ok=False)
    mp=out/'archive_manifest.json';save_new(mp,manifest);target=out/'evidence.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('archive_manifest.json',mp.read_bytes());written=set()
        for record in records:
            if not record['retained'] or record['member'] in written:continue
            raw=Path(record['path']).read_bytes()
            if digest_bytes(raw)!=record['sha256']:raise ValueError('Source changed while archiving')
            z.writestr(record['member'],raw);written.add(record['member'])
    # All source inventories are rechecked before a cleanup plan can exist.
    for p,h in all_files.items():
        if sha(p)!=h:raise ValueError('Source changed during archive: '+p)
    with Resolver(target,mp) as resolver:
        replay=replay_all(resolver)
    save_new(out/'archive_replay.json',replay)
    keep={p:h for p,h in all_files.items() if not any(Path(p).is_relative_to(f/'work') for f in ROOTS.values())}
    keep.update(file_inventory(out))
    plan={'schema':PLAN_SCHEMA,'created_utc':now(),'workspace':str(ROOT.resolve()),
          'allowed_delete_roots':[str((f/'work').resolve()) for f in ROOTS.values()],
          'archive':binding(target),'archive_manifest':binding(mp),'archive_replay':binding(out/'archive_replay.json'),
          'result_review':binding(result_review),'tool_review':binding(tool_review),'keep_sha256':keep,
          'work_inventories':work_inventories,'archive_crc_verified':True,'deletion_executed':False,
          'code_sha256':{p:sha(ROOT/p) for p in TOOL_FILES}}
    save_new(plan_path,plan)
    return {'status':'archive_verified','plan':str(plan_path),'plan_sha256':sha(plan_path),
            'logical_files':len(records),'unique_blobs':len(written),'disposed_files':sum(not x['retained'] for x in records)}


class Resolver:
    """Read original logical paths directly from verified immutable ZIP members."""
    def __init__(self,archive_path,manifest_path):
        self.manifest=read(manifest_path);self.zip=zipfile.ZipFile(archive_path)
        try:
            if self.manifest.get('schema')!=SCHEMA or self.zip.testzip() is not None:raise ValueError('Archive schema/CRC invalid')
            if self.zip.read('archive_manifest.json')!=Path(manifest_path).read_bytes():raise ValueError('External and archived manifests differ')
            self.records={os.path.normcase(str(Path(r['path']).resolve())):r for r in self.manifest['records']}
            if len(self.records)!=len(self.manifest['records']):raise ValueError('Duplicate logical archive paths')
            expected={'archive_manifest.json'}
            for record in self.records.values():
                if record['retained']:
                    if record.get('member')!='blobs/'+record['sha256']:raise ValueError('Content address differs')
                    expected.add(record['member'])
            if set(self.zip.namelist())!=expected or len(self.zip.namelist())!=len(expected):raise ValueError('Unexpected/duplicate ZIP entries')
            for member in expected-{'archive_manifest.json'}:
                if digest_bytes(self.zip.read(member))!=member.split('/')[1]:raise ValueError('Archive blob SHA differs')
        except BaseException:self.zip.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.zip.close()
    def bytes(self,path):
        record=self.records.get(os.path.normcase(str(Path(path).resolve())))
        if record is None or not record['retained']:raise ValueError('Logical artifact not retained: '+str(path))
        raw=self.zip.read(record['member'])
        if digest_bytes(raw)!=record['sha256'] or len(raw)!=record['size']:raise ValueError('Resolved archive evidence differs')
        return raw
    def read(self,path):return json.loads(self.bytes(path).decode('utf-8-sig'))
    def sha(self,path):return digest_bytes(self.bytes(path))


def clone(function,**changes):
    scope={**function.__globals__,**changes}
    result=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def qa_replayer(resolver):
    """Bind unchanged frozen QA functions to archive reads without global edits."""
    class ArchivedRawReplay(base.RawReplay):
        generate=clone(base.RawReplay.generate,read=resolver.read,sha=resolver.sha)
    known=clone(base.known_recoverable,RawReplay=ArchivedRawReplay,read=resolver.read,sha=resolver.sha)
    base_view=SimpleNamespace(**{**vars(base),'RawReplay':ArchivedRawReplay,'known_recoverable':known})
    return clone(continuation.replay_leaf,read=resolver.read,base=base_view)


def replay_all(resolver):
    """Run the frozen exact-request QA replay functions with archive-only reads."""
    replay=qa_replayer(resolver)
    roots={k:Path(v) for k,v in resolver.manifest['roots'].items()};new=roots['v3']
    m=resolver.read(new/'run_manifest.json');sources=resolver.read(new/'results/firstpass_sources.json')
    effective=resolver.read(new/'results/per_query.json');firstpass=resolver.read(new/'results/firstpass_per_query.json')
    if len(m['questions'])!=1540 or len({q['query_id'] for q in m['questions']})!=1540:raise ValueError('Archived full cohort missing')
    if any([r['query_id'] for r in values]!=[q['query_id'] for q in m['questions']] for values in (sources,effective,firstpass)):
        raise ValueError('Archived QA/source IDs differ')
    proof_binding=m['technical_continuation']['proof'];proof=resolver.read(proof_binding['path'])
    if resolver.sha(proof_binding['path'])!=proof_binding['sha256']:raise ValueError('Archived terminal proof differs')
    plan=resolver.read(new/'qa_recovery_plan.json');jobs=plan['jobs'];job_ids=[j['query_id'] for j in jobs]
    if plan.get('round')!=1 or len(job_ids)!=len(set(job_ids)):raise ValueError('Archived recovery round differs')
    official,_=base.verified_official_metric();checked=0;views={'firstpass':[],'effective':[]};eligible=[]
    for q,source,first,final in zip(m['questions'],sources,firstpass,effective):
        original_owner=Path(source['owner']);original_leaf=Path(source['leaf']);ret=Path(source['retrieval_path'])
        mode=proof['conversations'][str(q['conversation'])]['mode']
        if mode not in ('inherit','rebuild'):raise ValueError('Archived unknown conversation routing')
        expected_owner=roots['v2'] if mode=='inherit' else roots['v3']
        if (source['conversation']!=q['conversation'] or original_owner!=expected_owner
                or original_leaf!=expected_owner/'results/firstpass'/q['query_id']
                or ret!=expected_owner/'results'/f"c{q['conversation']}"/'retrieval'/(q['query_id']+'.json')):
            raise ValueError('Archived source route differs from terminal admission')
        for p,h in source['bindings'].items():
            if resolver.sha(p)!=h:raise ValueError('Archived firstpass source binding differs')
        for phase,value in (('firstpass',first),('effective',final)):
            owner=Path(value['source_owner']);leaf=Path(value['source_leaf'])
            recovery=phase=='effective' and q['query_id'] in job_ids
            if owner!=(new if recovery else original_owner) or leaf!=(new/'results/recovery'/q['query_id'] if recovery else original_leaf):
                raise ValueError('Archived result route differs from its phase/recovery plan')
            reader,judge=replay(q,leaf,owner,original_owner,original_leaf,ret)
            answer=reader['value']['prediction']['answer'] if reader['status']=='complete' else None
            label=base.evaluator.parse_judge_label(judge['value']['raw'])['correct'] if judge['status']=='complete' else None
            expected={**q,'answer':answer,'official_f1':official(answer,q['reference'],q['category']) if answer is not None else None,
                      'custom_metrics':base.evaluator.lexical_metrics(answer,q['reference']) if answer is not None else None,
                      'judge_correct':label,'reader_status':reader['status'],'judge_status':judge['status'],
                      'phase':phase,'source_owner':str(owner),'source_leaf':str(leaf),'retrieval_source':str(ret)}
            if expected!=value:raise ValueError('Archived complete per-query row differs from exact raw replay')
            if phase=='firstpass' and (reader['status']!='complete' or judge['status']!='complete'):eligible.append(q['query_id'])
            views[phase].append(expected)
            checked+=1
    if set(job_ids)!=set(eligible):raise ValueError('Archived recovery does not equal complete eligible cohort')
    summary=resolver.read(new/'results/summary.json')
    for phase,values in views.items():
        f1=[r['official_f1'] for r in values if r['official_f1'] is not None];labels=[r['judge_correct'] for r in values if r['judge_correct'] is not None]
        custom=[r['custom_metrics'].get('token_f1',r['custom_metrics'].get('f1')) for r in values if r['custom_metrics'] is not None]
        aggregate={'slots':1540,'reader_valid':len(f1),'judge_valid':len(labels),'custom_f1_valid_percent':100*sum(custom)/len(custom) if custom else None,
            'official_f1_valid_percent':100*sum(f1)/len(f1) if f1 else None,'official_f1_fixed_missing_zero_percent':100*sum(f1)/1540,
            'judge_valid_percent':100*sum(labels)/len(labels) if labels else None,'judge_fixed_missing_zero_percent':100*sum(labels)/1540,
            'missing_reader':1540-len(f1),'missing_judge':1540-len(labels)}
        if summary.get(phase)!=aggregate:raise ValueError('Archived summary differs from complete raw-replayed rows')
    return {'status':'passed','firstpass_and_effective_leaves':checked,'model_calls':0,'vault_reads':0,
            'method':'Original frozen RawReplay and v3 replay_leaf executed with archive resolver reads'}


def verify_cleanup(plan_path,*,require_work=True):
    stopped();plan=read(safe_path(plan_path))
    expected=[str((f/'work').resolve()) for f in ROOTS.values()]
    if (plan.get('schema')!=PLAN_SCHEMA or plan.get('workspace')!=str(ROOT.resolve())
            or plan.get('allowed_delete_roots')!=expected or not plan.get('archive_crc_verified')):raise ValueError('Wrong cleanup scope/schema')
    result_review=verify_result_review(verify_binding(plan['result_review']),live_inventory=False)
    verify_tool_review(verify_binding(plan['tool_review']))
    if plan.get('code_sha256')!={p:sha(ROOT/p) for p in TOOL_FILES}:raise ValueError('Cleanup implementation changed')
    for p,h in plan['keep_sha256'].items():verify_binding({'path':p,'sha256':h})
    target=verify_binding(plan['archive']);mp=verify_binding(plan['archive_manifest']);verify_binding(plan['archive_replay'])
    if require_work:
        for name,folder in ROOTS.items():
            if file_inventory(folder/'work')!=plan['work_inventories'][name]:raise ValueError('Work inventory changed before deletion')
    with Resolver(target,mp) as resolver:
        if resolver.manifest['work_inventories']!=plan['work_inventories']:raise ValueError('Archive/work inventory binding differs')
        reviewed={r['path']:r['sha256'] for r in resolver.manifest['records'] if Path(r['path']).is_relative_to(ROOTS['v3'])}
        if reviewed!=result_review['reviewed_run_inventory']:raise ValueError('Archived run differs from independently reviewed full inventory')
        replay=replay_all(resolver)
        if replay!=read(plan['archive_replay']['path']):raise ValueError('Independent archived replay differs')
    return {'status':'passed','plan_sha256':sha(plan_path),'archive_sha256':sha(target),'model_calls':0}


def complete(plan_path,cleanup_review):
    """Post-delete finalizer; no deletion or model access occurs here."""
    verify_cleanup(plan_path,require_work=False);plan=read(plan_path);review=read(verify_binding(binding(cleanup_review)))
    if (review.get('status')!='passed' or review.get('cleanup_plan_sha256')!=sha(plan_path)
            or review.get('archive_sha256')!=plan['archive']['sha256']):raise ValueError('Independent archive/cleanup approval differs')
    intent=read(output_dir()/'cleanup_execution_intent.json')
    if (intent.get('schema')!='amem-cross-root-cleanup-intent-v1' or intent.get('cleanup_plan_sha256')!=sha(plan_path)
            or intent.get('cleanup_review_sha256')!=sha(cleanup_review)
            or [r['path'] for r in intent.get('targets',[])]!=plan['allowed_delete_roots']):raise ValueError('Cleanup intent differs')
    receipts=[]
    for name,folder in ROOTS.items():
        if (folder/'work').exists():raise ValueError('A work root still exists')
        path=output_dir()/('cleanup_'+name+'.json');rec=read(path)
        if (rec.get('schema')!='amem-cross-root-removal-receipt-v1' or rec.get('removed_path')!=str((folder/'work').resolve())
                or rec.get('cleanup_plan_sha256')!=sha(plan_path) or rec.get('cleanup_review_sha256')!=sha(cleanup_review)):
            raise ValueError('Per-root cleanup receipt differs')
        receipts.append(binding(path))
    aggregate={'schema':'amem-cross-root-cleanup-receipt-v1','completed_utc':now(),'cleanup_plan':binding(plan_path),
               'archive':plan['archive'],'archive_manifest':plan['archive_manifest'],'cleanup_review':binding(cleanup_review),
               'per_root_receipts':receipts,'retained_evidence_verified':True}
    receipt=ROOTS['v3']/'results/cleanup_receipt.json';save_new(receipt,aggregate)
    completion={'schema':'amem-native-v3-predecessor-completion-v1','status':'passed','created_utc':now(),
                'summary_sha256':sha(ROOTS['v3']/'results/summary.json'),'cleanup_receipt_sha256':sha(receipt),
                'result_review':plan['result_review']}
    save_new(ROOTS['v3']/'results/predecessor_completion.json',completion)
    return {'status':'complete','cleanup_receipt_sha256':sha(receipt)}


def verify_completion(plan_path):
    """Strong post-cleanup prerequisite for separately reviewed later launchers."""
    verify_cleanup(plan_path,require_work=False);plan=read(plan_path)
    new=ROOTS['v3'];receipt_path=new/'results/cleanup_receipt.json';receipt=read(receipt_path)
    if (receipt.get('schema')!='amem-cross-root-cleanup-receipt-v1' or receipt.get('retained_evidence_verified') is not True
            or receipt.get('cleanup_plan')!=binding(plan_path) or receipt.get('archive')!=plan['archive']
            or receipt.get('archive_manifest')!=plan['archive_manifest']):raise ValueError('Aggregate cleanup receipt differs')
    review_path=verify_binding(receipt['cleanup_review']);review=read(review_path)
    if (review.get('status')!='passed' or review.get('cleanup_plan_sha256')!=sha(plan_path)
            or review.get('archive_sha256')!=plan['archive']['sha256']):raise ValueError('Cleanup independent review differs')
    intent=read(output_dir()/'cleanup_execution_intent.json')
    if (intent.get('schema')!='amem-cross-root-cleanup-intent-v1' or intent.get('cleanup_plan_sha256')!=sha(plan_path)
            or intent.get('cleanup_review_sha256')!=sha(review_path)
            or [r['path'] for r in intent.get('targets',[])]!=plan['allowed_delete_roots']):raise ValueError('Cleanup exclusive intent differs')
    expected_receipts=[]
    for name,folder in ROOTS.items():
        if (folder/'work').exists():raise ValueError('Work root remains/reappeared after cleanup')
        path=output_dir()/('cleanup_'+name+'.json');rec=read(path)
        if (rec.get('schema')!='amem-cross-root-removal-receipt-v1' or rec.get('removed_path')!=str((folder/'work').resolve())
                or rec.get('cleanup_plan_sha256')!=sha(plan_path) or rec.get('cleanup_review_sha256')!=sha(review_path)):
            raise ValueError('Removal receipt differs')
        expected_receipts.append(binding(path))
    if receipt.get('per_root_receipts')!=expected_receipts:raise ValueError('Removal receipt SHA list differs')
    completion=read(new/'results/predecessor_completion.json')
    if (completion.get('schema')!='amem-native-v3-predecessor-completion-v1' or completion.get('status')!='passed'
            or completion.get('summary_sha256')!=sha(new/'results/summary.json')
            or completion.get('cleanup_receipt_sha256')!=sha(receipt_path)
            or completion.get('result_review')!=plan['result_review']):raise ValueError('Predecessor completion differs')
    return {'status':'passed','archive_sha256':plan['archive']['sha256'],'cleanup_receipt_sha256':sha(receipt_path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('archive','verify','complete','replay','verify-completion'))
    p.add_argument('--result-review',type=Path);p.add_argument('--tool-review',type=Path);p.add_argument('--cleanup-review',type=Path)
    p.add_argument('--plan',type=Path,default=ROOTS['v3']/'cleanup_plan_native_v3.json');args=p.parse_args()
    if args.command=='archive':result=archive(args.result_review,args.tool_review)
    elif args.command=='verify':result=verify_cleanup(args.plan)
    elif args.command=='complete':result=complete(args.plan,args.cleanup_review)
    elif args.command=='verify-completion':result=verify_completion(args.plan)
    else:
        plan=read(args.plan)
        with Resolver(verify_binding(plan['archive']),verify_binding(plan['archive_manifest'])) as resolver:result=replay_all(resolver)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
