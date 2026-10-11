"""Four-root offline evidence archive and frozen-function post-cleanup replay.

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
import fnmatch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tools import run_five_baselines_amem_native_v4 as continuation
from tools import run_five_baselines_gemini25_v1 as base
from tools.baselines_common_v1 import read, sha, save_new, now

ROOT=continuation.ROOT
ROOTS={**continuation.HISTORY,'v4':continuation.NEW}
TOOL_FILES=('tools/archive_amem_continuation_v4.py','tools/cleanup_amem_continuation_v4.ps1',
            'tools/baseline_controller_guard_v2.py','tools/run_five_baselines_amem_native_v4.py',
            'tools/baselines_amem_v4.py')
SCHEMA='amem-four-root-content-archive-v1'
PLAN_SCHEMA='amem-four-root-cleanup-plan-v1'


def output_dir():return ROOTS['v4']/'results/continuation_archive'


def digest_bytes(raw):return hashlib.sha256(raw).hexdigest()


def stopped():
    from tools.baseline_controller_guard_v2 import assert_stopped
    assert_stopped()


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
    # Session journals include unique partial-build evidence. Retain all text
    # snapshots; only reproducible numeric encoder caches may be disposed.
    for folder in ROOTS.values():
        native=folder/'work/native'
        if path.is_relative_to(native):
            parts=path.relative_to(native).parts
            if len(parts)==3 and parts[1]=='cache' and parts[2] in ('embeddings.npy','embeddings.tmp.npy'):
                return 'original MiniLM numeric embedding cache; SHA and size retained'
    return None


def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def verify_binding(value):
    p=safe_path(value['path'])
    if not p.is_file() or sha(p)!=value['sha256']:raise ValueError('Bound evidence changed: '+str(p))
    return p


def verify_result_review(path, *, live_inventory=True):
    review=read(safe_path(path));summary_path=ROOTS['v4']/'results/summary.json';summary=read(summary_path)
    if review.get('status')!='passed' or review.get('summary_sha256')!=sha(summary_path):raise ValueError('Passed independent full result review required')
    effective=summary.get('effective',{})
    if summary.get('audit_status')!='no_invalid_artifacts' or any(effective.get(k)!=1540 for k in ('slots','reader_valid','judge_valid')):
        raise ValueError('All 1540 effective Reader and Judge stages must be valid')
    state=read(ROOTS['v4']/'status.json')
    if (summary.get('schema')!='five-baseline-result-native-continuation-v4'
            or summary.get('self_root')!=str(ROOTS['v4']) or summary.get('method')!='amem'
            or state.get('stage')!='completed_awaiting_result_review'
            or state.get('summary_sha256')!=sha(summary_path)):
        raise ValueError('Exact v4 terminal summary/status required')
    expected=review.get('reviewed_run_inventory')
    if (Path(path).resolve().is_relative_to(ROOTS['v4'].resolve()) or not isinstance(expected,dict) or not expected
            or any(not Path(p).is_absolute() or str(Path(p).resolve())!=p or not Path(p).is_relative_to(ROOTS['v4'])
                   or not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for p,h in expected.items())):
        raise ValueError('Independent result review must bind a nonempty complete absolute run inventory and live outside the run')
    if live_inventory and file_inventory(ROOTS['v4'])!=expected:raise ValueError('Current run differs from independently reviewed full inventory')
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
    frozen_manifest=continuation.manifest()
    summary=read(ROOTS['v4']/'results/summary.json')
    if read(ROOTS['v4']/'status.json').get('stage')!='completed_awaiting_result_review':
        raise ValueError('Final v4 run is not terminal')
    return replay_all(LiveResolver(frozen_manifest, summary))


def archive(result_review,tool_review):
    stopped();verify_result_review(result_review);verify_tool_review(tool_review)
    out=output_dir();plan_path=ROOTS['v4']/'cleanup_plan_native_v4.json'
    if out.exists() or plan_path.exists():raise FileExistsError('Archive namespace already exists; no implicit replacement')
    verify_native_seal()
    all_files={};work_inventories={}
    for name,folder in ROOTS.items():
        if not (folder/'work').is_dir():raise ValueError('All four original work roots required before archive')
        current=file_inventory(folder);all_files.update(current)
        work_inventories[name]={p:h for p,h in current.items() if Path(p).is_relative_to(folder/'work')}
    m=read(ROOTS['v4']/'run_manifest.json')
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
    for name,history in proof['history_inventories'].items():
        if name not in ('v1','v2','v3'):raise ValueError('Unexpected historical root')
        expected={str((ROOTS[name]/p).resolve()):h for p,h in history.items()}
        observed={p:h for p,h in all_files.items() if Path(p).is_relative_to(ROOTS[name])}
        if observed!=expected:raise ValueError('Archived original source differs from terminal proof')
    if set(proof['history_inventories'])!={'v1','v2','v3'}:raise ValueError('All three historical inventories required')
    records=[]
    for p,h in sorted(all_files.items()):
        reason=disposable(Path(p));record={'path':p,'sha256':h,'size':Path(p).stat().st_size,'retained':reason is None}
        if reason:record['disposal_reason']=reason
        else:record['member']='blobs/'+h
        records.append(record)
    manifest={'schema':SCHEMA,'created_utc':now(),'roots':{k:str(v.resolve()) for k,v in ROOTS.items()},
              'summary':binding(ROOTS['v4']/'results/summary.json'),'result_review':binding(result_review),
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
                if type(record.get('retained')) is not bool:
                    raise ValueError('Archive retention flag must be a boolean')
                if not record['retained']:
                    reason=disposable(Path(record['path']))
                    if reason is None or record.get('disposal_reason')!=reason or 'member' in record:
                        raise ValueError('Archive disposal is outside the exact numeric-cache policy')
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


class LiveResolver:
    """The identical seal replay before packing, using immutable live evidence."""
    def __init__(self, manifest, summary):
        self.manifest={'roots':{k:str(v) for k,v in ROOTS.items()},'records':[
            {'path':p,'sha256':h} for folder in ROOTS.values() for p,h in file_inventory(folder).items()]}
    read=staticmethod(read)
    sha=staticmethod(sha)


def archived_path_class(resolver):
    """Read-only virtual paths over the logical archive inventory; no extraction."""
    files={Path(r['path']) for r in resolver.manifest['records']}
    directories={parent for p in files for parent in p.parents}
    nodes=files|directories
    class ArchivedPath(type(Path())):
        def resolve(self, strict=False):
            if not self.is_absolute() or '..' in self.parts:
                raise ValueError('Noncanonical archived path')
            if strict and not self.exists():raise FileNotFoundError(str(self))
            return self
        def exists(self):return Path(self) in nodes
        def is_file(self):return Path(self) in files
        def is_dir(self):return Path(self) in directories
        def read_bytes(self):return resolver.bytes(self)
        def read_text(self, encoding=None, errors=None):
            return self.read_bytes().decode(encoding or 'utf-8',errors or 'strict')
        def glob(self, pattern):
            # Frozen audits use bounded ordinary globs, never recursive wildcards.
            if '**' in pattern:raise ValueError('Recursive virtual glob is not an admitted audit operation')
            parts=Path(pattern).parts
            for p in sorted(nodes):
                if p==Path(self) or not p.is_relative_to(self):continue
                relative=p.relative_to(self).parts
                if len(relative)==len(parts) and all(fnmatch.fnmatchcase(a.lower(),b.lower()) for a,b in zip(relative,parts)):
                    yield ArchivedPath(p)
    return ArchivedPath


def _archive_contract(resolver):
    roots={k:Path(v) for k,v in resolver.manifest['roots'].items()}
    if roots!=ROOTS:raise ValueError('Archive requires exactly the four original roots')
    new=roots['v4'];m=resolver.read(new/'run_manifest.json')
    proof_binding=m['technical_continuation']['proof'];proof=resolver.read(proof_binding['path'])
    if resolver.sha(proof_binding['path'])!=proof_binding['sha256']:raise ValueError('Archived admission proof differs')
    routes={str(ci):str(roots['v4'] if ci==2 else roots['v2'] if ci in (0,3,6,8) else roots['v3']) for ci in range(10)}
    summary=resolver.read(new/'results/summary.json')
    if (m.get('self_root')!=str(new) or proof.get('self_root')!=str(new) or summary.get('self_root')!=str(new)
            or any(x.get('conversation_sources')!=routes for x in (m,proof,summary))
            or m['technical_continuation']['rebuild_conversations']!=[2]
            or proof.get('paid_replacement_permissions')!=[]
            or set(proof['history_inventories'])!={'v1','v2','v3'}
            or proof['history_roots']!={k:str(roots[k]) for k in ('v1','v2','v3')}
            or m['questions']!=base.questions() or len(m['questions'])!=1540):
        raise ValueError('Archived v4 ownership, original cohort or no-replacement contract differs')
    for p,h in {**m['code_sha256'],**m['upstream_sha256']}.items():
        if resolver.sha(ROOT/p)!=h:raise ValueError('Archived frozen source differs')
    for name in ('v1','v2','v3'):
        expected={str((roots[name]/p).resolve()):h for p,h in proof['history_inventories'][name].items()}
        actual={r['path']:r['sha256'] for r in resolver.manifest['records'] if Path(r['path']).is_relative_to(roots[name])}
        if expected!=actual:raise ValueError('Historical logical inventory differs from terminal proof')
        if resolver.sha(roots[name]/'run_manifest.json')!=proof['history_manifests'][name]:
            raise ValueError('Historical manifest differs')
    return m,proof,summary


def qa_replayer(resolver):
    P=Path if isinstance(resolver,LiveResolver) else archived_path_class(resolver)
    class ArchivedRawReplay(base.RawReplay):
        generate=clone(base.RawReplay.generate,read=resolver.read,sha=resolver.sha,Path=P)
    known=clone(base.known_recoverable,RawReplay=ArchivedRawReplay,read=resolver.read,sha=resolver.sha,Path=P)
    view=SimpleNamespace(**{**vars(base),'RawReplay':ArchivedRawReplay,'known_recoverable':known})
    return clone(continuation.controller.replay_leaf,read=resolver.read,sha=resolver.sha,Path=P,base=view)


def audit_historical_source(resolver,common,P,proof):
    source_manifest=resolver.read(ROOTS['v3']/'run_manifest.json')
    old_proof=resolver.read(source_manifest['technical_continuation']['proof']['path'])
    source_common={**common,'NEW':P(ROOTS['v3']),'SOURCE':P(ROOTS['v2']),'ANCESTOR':P(ROOTS['v1']),'_PROOF':old_proof}
    source_api=clone(continuation.previous.audit_new_requests,**source_common)()
    source_native=clone(continuation._audit_source_native,**{**common,'SOURCE':P(ROOTS['v3'])})(old_proof)
    if source_api!=proof['source_raw_audit'] or source_native!=proof['source_native_audit']:
        raise ValueError('Archived v3 producer/consumer/native evidence differs from admission seal')
    return source_api,source_native


def replay_all(resolver):
    """Reexecute unchanged frozen seal/audits/QA using ZIP reads after deletion."""
    m,proof,summary=_archive_contract(resolver)
    P=Path if isinstance(resolver,LiveResolver) else archived_path_class(resolver)
    def bound(path,folder):
        p,f=P(path).resolve(),P(folder).resolve()
        if not p.is_relative_to(f) or not p.is_file():raise ValueError('Missing or foreign archived bound artifact')
        return p
    def checked(path,expected,folder):
        p=bound(path,folder)
        if resolver.sha(p)!=expected:raise ValueError('Archived bound SHA differs')
        return p
    class ArchivedRawReplay(base.RawReplay):
        generate=clone(base.RawReplay.generate,read=resolver.read,sha=resolver.sha,Path=P)
    known=clone(base.known_recoverable,RawReplay=ArchivedRawReplay,read=resolver.read,sha=resolver.sha,Path=P)
    view=SimpleNamespace(**{**vars(base),'RawReplay':ArchivedRawReplay,'known_recoverable':known})
    new=P(ROOTS['v4'])
    def source_for(q):
        owner=P(proof['conversation_sources'][str(q['conversation'])])
        return owner,owner/'results/firstpass'/q['query_id'],owner/'results'/f"c{q['conversation']}"/'retrieval'/(q['query_id']+'.json')
    common={'read':resolver.read,'sha':resolver.sha,'Path':P,'NEW':new,'_PROOF':proof,'base':view,
            'bound_path':bound,'check_binding':checked,'source_for':source_for}
    controller=continuation.controller
    leaf=clone(controller.replay_leaf,**common)
    api_audit=clone(controller.audit_new_requests,**common)
    native_audit=clone(controller.audit_native_outputs,**common)
    usage=clone(continuation.usage_union,**common,HISTORY={k:P(v) for k,v in continuation.HISTORY.items()})
    original_sources=clone(controller.firstpass_sources,**common,replay_leaf=leaf)(m)
    if original_sources!=resolver.read(new/'results/firstpass_sources.json'):
        raise ValueError('Archived complete firstpass source table differs from original raw replay')
    # Re-run v3 producer/consumer/native audit too, including its partial c2.
    source_api,source_native=audit_historical_source(resolver,common,P,proof)
    captured={}
    def capture(path,value):
        if str(path) in captured:raise ValueError('Duplicate frozen seal output')
        if value!=resolver.read(path):raise ValueError('Archived frozen seal output differs: '+str(path))
        captured[str(path)]=value
    def proof_check(path):
        if P(path)!=P(m['technical_continuation']['proof']['path']):raise ValueError('Foreign frozen seal proof')
        return proof
    sealed=clone(controller.seal,**common,validate_proof=proof_check,audit_new_requests=api_audit,
                 audit_native_outputs=native_audit,replay_leaf=leaf,usage_union=usage,
                 save_new=capture,now=lambda:summary['created_utc'])(m)
    if sealed!=summary or len(captured)!=5:raise ValueError('Incomplete archived full seal replay')
    return {'status':'passed','firstpass_and_effective_leaves':3080,'model_calls':0,'vault_reads':0,
            'raw_request_audit_sha256':resolver.sha(new/'results/raw_request_audit.json'),
            'native_output_audit_sha256':resolver.sha(new/'results/native_output_audit.json'),
            'source_raw_audit_status':source_api['status'],'source_native_audit_status':source_native['status'],
            'physical_and_inherited_usage':sealed['usage'],
            'method':'Unchanged frozen v4 seal, original QA and v3 source audit functions over read-only logical archive paths'}

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
        reviewed={r['path']:r['sha256'] for r in resolver.manifest['records'] if Path(r['path']).is_relative_to(ROOTS['v4'])}
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
    if (intent.get('schema')!='amem-four-root-cleanup-intent-v1' or intent.get('cleanup_plan_sha256')!=sha(plan_path)
            or intent.get('cleanup_review_sha256')!=sha(cleanup_review)
            or [r['path'] for r in intent.get('targets',[])]!=plan['allowed_delete_roots']):raise ValueError('Cleanup intent differs')
    receipts=[]
    for name,folder in ROOTS.items():
        if (folder/'work').exists():raise ValueError('A work root still exists')
        path=output_dir()/('cleanup_'+name+'.json');rec=read(path)
        if (rec.get('schema')!='amem-four-root-removal-receipt-v1' or rec.get('removed_path')!=str((folder/'work').resolve())
                or rec.get('cleanup_plan_sha256')!=sha(plan_path) or rec.get('cleanup_review_sha256')!=sha(cleanup_review)):
            raise ValueError('Per-root cleanup receipt differs')
        receipts.append(binding(path))
    aggregate={'schema':'amem-four-root-cleanup-receipt-v1','completed_utc':now(),'cleanup_plan':binding(plan_path),
               'archive':plan['archive'],'archive_manifest':plan['archive_manifest'],'cleanup_review':binding(cleanup_review),
               'per_root_receipts':receipts,'retained_evidence_verified':True}
    receipt=ROOTS['v4']/'results/cleanup_receipt.json';save_new(receipt,aggregate)
    completion={'schema':'amem-native-v4-predecessor-completion-v1','status':'passed','created_utc':now(),
                'summary_sha256':sha(ROOTS['v4']/'results/summary.json'),'cleanup_receipt_sha256':sha(receipt),
                'result_review':plan['result_review']}
    save_new(ROOTS['v4']/'results/predecessor_completion.json',completion)
    return {'status':'complete','cleanup_receipt_sha256':sha(receipt)}


def verify_completion(plan_path):
    """Strong post-cleanup prerequisite for separately reviewed later launchers."""
    verify_cleanup(plan_path,require_work=False);plan=read(plan_path)
    new=ROOTS['v4'];receipt_path=new/'results/cleanup_receipt.json';receipt=read(receipt_path)
    if (receipt.get('schema')!='amem-four-root-cleanup-receipt-v1' or receipt.get('retained_evidence_verified') is not True
            or receipt.get('cleanup_plan')!=binding(plan_path) or receipt.get('archive')!=plan['archive']
            or receipt.get('archive_manifest')!=plan['archive_manifest']):raise ValueError('Aggregate cleanup receipt differs')
    review_path=verify_binding(receipt['cleanup_review']);review=read(review_path)
    if (review.get('status')!='passed' or review.get('cleanup_plan_sha256')!=sha(plan_path)
            or review.get('archive_sha256')!=plan['archive']['sha256']):raise ValueError('Cleanup independent review differs')
    intent=read(output_dir()/'cleanup_execution_intent.json')
    if (intent.get('schema')!='amem-four-root-cleanup-intent-v1' or intent.get('cleanup_plan_sha256')!=sha(plan_path)
            or intent.get('cleanup_review_sha256')!=sha(review_path)
            or [r['path'] for r in intent.get('targets',[])]!=plan['allowed_delete_roots']):raise ValueError('Cleanup exclusive intent differs')
    expected_receipts=[]
    for name,folder in ROOTS.items():
        if (folder/'work').exists():raise ValueError('Work root remains/reappeared after cleanup')
        path=output_dir()/('cleanup_'+name+'.json');rec=read(path)
        if (rec.get('schema')!='amem-four-root-removal-receipt-v1' or rec.get('removed_path')!=str((folder/'work').resolve())
                or rec.get('cleanup_plan_sha256')!=sha(plan_path) or rec.get('cleanup_review_sha256')!=sha(review_path)):
            raise ValueError('Removal receipt differs')
        expected_receipts.append(binding(path))
    if receipt.get('per_root_receipts')!=expected_receipts:raise ValueError('Removal receipt SHA list differs')
    completion=read(new/'results/predecessor_completion.json')
    if (completion.get('schema')!='amem-native-v4-predecessor-completion-v1' or completion.get('status')!='passed'
            or completion.get('summary_sha256')!=sha(new/'results/summary.json')
            or completion.get('cleanup_receipt_sha256')!=sha(receipt_path)
            or completion.get('result_review')!=plan['result_review']):raise ValueError('Predecessor completion differs')
    return {'status':'passed','archive_sha256':plan['archive']['sha256'],'cleanup_receipt_sha256':sha(receipt_path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('archive','verify','complete','replay','verify-completion'))
    p.add_argument('--result-review',type=Path);p.add_argument('--tool-review',type=Path);p.add_argument('--cleanup-review',type=Path)
    p.add_argument('--plan',type=Path,default=ROOTS['v4']/'cleanup_plan_native_v4.json');args=p.parse_args()
    if args.command=='archive':result=archive(args.result_review,args.tool_review)
    elif args.command=='verify':result=verify_cleanup(args.plan)
    elif args.command=='complete':result=complete(args.plan,args.cleanup_review)
    elif args.command=='verify-completion':result=verify_completion(args.plan)
    else:
        plan=read(args.plan)
        with Resolver(verify_binding(plan['archive']),verify_binding(plan['archive_manifest'])) as resolver:result=replay_all(resolver)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
