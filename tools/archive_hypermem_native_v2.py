"""HyperMem dual-root lifecycle with explicit predelete index certification.

Only the NEW tools own this lifecycle. Frozen generation, retrieval and QA code
remain unchanged. A postdelete archive replay reads retained bytes only; its
declared private audit reads certified embedding identity/digests instead of
deleted runnable caches and checks raw embedding normalization again.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import FunctionType, SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import archive_simplemem_native_v2 as simple
from tools import baseline_lifecycle_v3 as single
from tools import archive_amem_metric_closure_v2 as metric
from tools import run_five_baselines_hypermem_native_v2 as core
from tools import audit_hypermem_terminal_v2 as terminal
from tools import audit_hypermem_continuation_v2 as audit
from tools import baseline_controller_guard_v3 as guard
from tools import seal_simplemem_native_v2 as shared_replay
from tools.baselines_common_v1 import read,sha,save_new,now

OLD=core.OLD; NEW=core.NEW
OUTPUT=ROOT/'research/baseline_execution_2026-10-08/hypermem_native_v2_lifecycle'
DEFAULT_PLAN=OUTPUT/'cleanup_plan.json'
SCHEMA='hypermem-native-v2-two-root-archive-v1'
PLAN_SCHEMA='hypermem-native-v2-two-root-cleanup-plan-v1'
REVIEW_SCHEMA='hypermem-native-v2-lifecycle-implementation-review-v1'
RESULT_SCHEMA='hypermem-native-v2-result-review-v1'
CERT_SCHEMA='hypermem-native-v2-predelete-bank-validation-v1'
BANK_REVIEW_SCHEMA='hypermem-native-v2-bank-validation-review-v1'
EXTRA=('tools/archive_hypermem_native_v2.py','tools/cleanup_hypermem_native_v2.ps1',
       'tools/run_five_baselines_reviewed_chain_v7.py')
OUTPUTS=('firstpass_per_query.json','per_query.json','raw_request_audit.json','summary.json')
SCOPE='real ZIP raw/native/provenance/QA replay and raw embedding normalization; disposed indexes and native vector caches validated before deletion under independent certificate; no index loading or retrieval search replay'
FROZEN_SIMPLE_SHA='4630e4209637a635ed096783d660609ded11aaeb13b2116df4d86ab839720080'
FROZEN_AUDIT_SHA='5f30bdea08a823923f56c5fd9dd28df65f5b94043c75f0ef27630af8b713cd63'
FROZEN_RUNTIME_SHA='5b4d73650f9c4d69e714468bc31f4da5accf8ce83d9e83bf2112ff51b1643df3'
INDEXES={'bm25':'hypergraph_bm25_index_conv_0.pkl','vectors':'hypergraph_embedding_index_conv_0.pkl','enhanced':'hypergraph_embedding_index_enhanced_conv_0.pkl'}
_ACTIVE=terminal.active_controllers
active_controllers=FunctionType(_ACTIVE.__code__,{**_ACTIVE.__globals__,
    'LAUNCHERS':(*_ACTIVE.__globals__['LAUNCHERS'],'run_five_baselines_reviewed_chain_v7')},
    'active_controllers',_ACTIVE.__defaults__)
_STOP=terminal.assert_stopped
_stop=FunctionType(_STOP.__code__,{**_STOP.__globals__,'active_controllers':active_controllers},'assert_stopped',_STOP.__defaults__)
class LiveView(shared_replay.LiveView):
    """Live artifact admission checks containment/reparse before content I/O."""
    read=staticmethod(lambda p:read(safe(p)))
    sha=staticmethod(lambda p:sha(safe(p)))
    bytes=staticmethod(lambda p:safe(p).read_bytes())

def view_bound(view,record):
    p=safe(record['path'])
    if view.sha(p)!=record['sha256']:raise ValueError('Bound lifecycle evidence changed: '+str(p))
    return view.read(p)

seal=SimpleNamespace(LiveView=LiveView,bound=view_bound)

def require(ok,message):
    if not ok:raise ValueError(message)


def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def roots():return {'original':OLD,'continuation':NEW}


def work_roots():return [r/'work' for r in roots().values()]


def safe(path):return single.clone(single.safe_path,ROOT=ROOT)(path)


def binding(path):
    p=safe(path);return {'path':str(p),'sha256':sha(p)}


def bound(record):
    p=safe(record['path']);require(sha(p)==record['sha256'],'Binding changed: '+str(p));return p


def inventory(folder):return single.clone(single.inventory,ROOT=ROOT,safe_path=safe)(folder)


def inventories():return {k:inventory(p) for k,p in roots().items()}


def hashes(inv):return {p:v['sha256'] for p,v in inv.items()}


def stopped():
    # Generic foreign-controller detection is unchanged. Only this precise
    # future launcher may have its actual verified venv parent ignored.
    _stop()


def tool_review(path):
    v=read(safe(path));require(v.get('schema')==REVIEW_SCHEMA and v.get('status')=='passed'
        and v.get('scope')=='two_root_lifecycle_with_explicit_predelete_bank_certificate'
        and v.get('code_sha256')==code_hashes() and bool(v.get('evidence_sha256')),'Independent exact lifecycle implementation review required')
    for p,h in v['evidence_sha256'].items():bound({'path':str(ROOT/p),'sha256':h})
    return v


def check_inventory(value,folder):
    require(isinstance(value,dict) and value,'Complete independent inventory required')
    for p,h in value.items():
        require(str(safe(p))==p and Path(p).is_relative_to(folder) and isinstance(h,str)
            and re.fullmatch('[0-9a-f]{64}',h),'Invalid inventory path or SHA')


def verify_certificate(view,bank_review_path,result_path,tool_path):
    br=view.read(bank_review_path)
    require(br.get('schema')==BANK_REVIEW_SCHEMA and br.get('status')=='passed'
        and br.get('scope')=='independently_verified_predelete_bank_bytes_and_declared_postdelete_scope'
        and br.get('postdelete_scope')==SCOPE and br.get('code_sha256')==code_hashes(),'Independent bank scope review required')
    cert=seal.bound(view,br['certificate'])
    require(br.get('result_review')=={'path':str(result_path),'sha256':view.sha(result_path)}
        and br.get('tool_review')=={'path':str(tool_path),'sha256':view.sha(tool_path)},'Bank review result/tool binding differs')
    require(cert.get('schema')==CERT_SCHEMA and cert.get('status')=='passed' and cert.get('scope')=='actual_predelete_bytes_and_unmodified_frozen_full_audits'
        and cert.get('code_sha256')==code_hashes() and cert.get('inventory_before_after_equal') is True
        and cert.get('database_engine_opened') is False and cert.get('retrieval_search_reexecuted') is False
        and type(cert.get('model_calls')) is int and cert['model_calls']==0 and type(cert.get('credential_reads')) is int and cert['credential_reads']==0
        and cert.get('postdelete_scope')==SCOPE,'Invalid bank validation certificate')
    require(cert['result_review']==br['result_review'] and cert['tool_review']==br['tool_review'],'Certificate authority differs')
    for path,h in cert['output_sha256'].items():require(view.sha(path)==h,'Certified output changed')
    require(set(cert['output_sha256'])=={str(NEW/'results'/n) for n in OUTPUTS},'Incomplete certified outputs')
    return cert


def function_source(view,path,name):
    raw=view.bytes(ROOT/path).decode('utf-8-sig');tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    require(len(nodes)==1,'Frozen function boundary differs')
    return '\n'.join(raw.splitlines()[nodes[0].lineno-1:nodes[0].end_lineno])+'\n'


def private_function(text,namespace,name,filename):
    code=compile(text,filename,'exec');scope=dict(namespace);exec(code,scope);return scope[name]


def virtual_paths(view):
    files={Path(r['path']) for r in view.manifest['records'] if r['retained']};dirs={p for f in files for p in f.parents};nodes=files|dirs
    class VirtualPath(type(Path())):
        def resolve(self,strict=False):
            require(self.is_absolute() and '..' not in self.parts,'Noncanonical virtual path')
            if strict and not self.exists():raise FileNotFoundError(str(self))
            return self
        def exists(self):return Path(self) in nodes
        def is_file(self):return Path(self) in files
        def is_dir(self):return Path(self) in dirs
        def read_bytes(self):return view.bytes(self)
        def read_text(self,encoding=None,errors=None):return self.read_bytes().decode(encoding or 'utf-8',errors or 'strict')
        def iterdir(self):return self.glob('*')
        def glob(self,pattern):
            require('**' not in pattern,'Use explicit virtual rglob')
            parts=Path(pattern).parts
            for p in sorted(nodes):
                if p==Path(self) or not p.is_relative_to(self):continue
                rel=p.relative_to(self).parts
                if len(rel)==len(parts) and all(fnmatch.fnmatchcase(a.lower(),b.lower()) for a,b in zip(rel,parts)):yield VirtualPath(p)
        def rglob(self,pattern):
            require('**' not in pattern,'Nested recursive pattern unsupported')
            parts=Path(pattern).parts
            for p in sorted(nodes):
                if p==Path(self) or not p.is_relative_to(self):continue
                rel=p.relative_to(self).parts
                if len(rel)>=len(parts) and all(fnmatch.fnmatchcase(a.lower(),b.lower()) for a,b in zip(rel[-len(parts):],parts)):yield VirtualPath(p)
    return VirtualPath


class Resolver:
    def __init__(self,target,manifest_path):
        self.manifest=read(safe(manifest_path));self.zip=zipfile.ZipFile(safe(target))
        try:
            require(self.manifest['schema']==SCHEMA and self.zip.testzip() is None,'Archive schema/CRC differs')
            require(self.zip.read('archive_manifest.json')==Path(manifest_path).read_bytes(),'Embedded manifest differs')
            self.records={os.path.normcase(r['path']):r for r in self.manifest['records']}
            require(len(self.records)==len(self.manifest['records']),'Duplicate logical archive path')
            members={'archive_manifest.json'}
            for p,r in self.records.items():
                require(str(safe(r['path']))==r['path'] and type(r['retained']) is bool,'Noncanonical record')
                reason=disposition(r['path']);require(r['retained']==(reason is None),'Wrong retention policy')
                if reason:require(r.get('disposal_reason')==reason and 'member' not in r,'Disposed artifact has bytes')
                else:require(r.get('member')=='blobs/'+r['sha256'],'Wrong content address');members.add(r['member'])
            require(set(self.zip.namelist())==members and len(self.zip.namelist())==len(members),'Extra/duplicate archive member')
            for member in members-{'archive_manifest.json'}:require(hashlib.sha256(self.zip.read(member)).hexdigest()==member[6:],'Blob SHA differs')
            self._hypermem_path_type=virtual_paths(self)
        except BaseException:self.zip.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.zip.close()
    def bytes(self,path):
        r=self.records.get(os.path.normcase(str(Path(path).absolute())))
        require(r is not None and r['retained'],'Archive bytes unavailable; no live fallback: '+str(path))
        data=self.zip.read(r['member']);require(len(data)==r['size'] and hashlib.sha256(data).hexdigest()==r['sha256'],'Archive bytes changed');return data
    def read(self,path):return json.loads(self.bytes(path).decode('utf-8-sig'))
    def sha(self,path):return hashlib.sha256(self.bytes(path)).hexdigest()


def strict_replay(view,runtime=None):
    import numpy  # Original double-normalization audit only; no encoder/DB.
    actual=metric.prepare_metric_runtime()
    if runtime is None:runtime=actual
    else:
        require(runtime.get('versions')==actual['versions'],'Recorded metric runtime versions changed')
        require(bool(runtime.get('runtime_modules_sha256')),'Recorded initialized runtime modules required')
        for p,h in runtime['runtime_modules_sha256'].items():
            require(actual['runtime_modules_sha256'].get(p)==h,'Recorded initialized runtime module differs')
    with metric.only_zip_workspace_reads(view,runtime):result=replay_all(view)
    return {**result,'runtime':runtime}


def archive(result_path,tool_path,bank_review_path):
    result_path=safe(result_path);tool_path=safe(tool_path);bank_review_path=safe(bank_review_path)
    stopped();tool_review(tool_path)
    require(not any((OUTPUT/name).exists() for name in ('cleanup_plan.json','evidence.zip','archive_manifest.json','archive_replay.json',
        'cleanup_execution_intent.json','removal_original.json','removal_continuation.json','cleanup_receipt.json','predecessor_completion.json')),'Archive already started; existing artifacts require review')
    m,p,s,review=result_review(result_path,live=True);cert=verify_certificate(seal.LiveView(),bank_review_path,result_path,tool_path)
    before=inventories();require(before==cert['run_inventories'],'Certified banks/run changed')
    # Explicit current-result closure. Nothing missing is silently optional.
    initial=[(str(path),sha(path)) for path in (result_path,tool_path,bank_review_path,NEW/'run_manifest.json',ROOT/metric.METRIC,ROOT/'data/locomo10.json',ROOT/'configs/sdk20_full1540_locked_v1.json')]
    print('hypermem lifecycle: explicit evidence closure',flush=True)
    annotations=[];evidence=evidence_closure(initial,annotations=annotations)
    allfiles={p:rec for inv in before.values() for p,rec in inv.items()}
    for path,h in evidence.items():
        require(path not in allfiles or allfiles[path]['sha256']==h,'Conflicting evidence inventory')
        allfiles.setdefault(path,{'sha256':h,'size':safe(path).stat().st_size})
    records=[]
    for path,rec in sorted(allfiles.items()):
        reason=disposition(path);row={'path':path,**rec,'retained':reason is None}
        if reason:row['disposal_reason']=reason
        else:row['member']='blobs/'+rec['sha256']
        records.append(row)
    am={'schema':SCHEMA,'created_utc':now(),'roots':{k:str(v) for k,v in roots().items()},'records':records,
        'result_review':binding(result_path),'tool_review':binding(tool_path),'bank_review':binding(bank_review_path),
        'bindings':{'bank_certificate':read(bank_review_path)['certificate']},'work_inventories':{k:{p:r for p,r in inv.items() if Path(p).is_relative_to(roots()[k]/'work')} for k,inv in before.items()},
        'code_sha256':code_hashes(),'scope':SCOPE,'dependency_annotations':annotations}
    mp=OUTPUT/'archive_manifest.json';target=OUTPUT/'evidence.zip';save_new(mp,am)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('archive_manifest.json',mp.read_bytes());written=set()
        for ordinal,rec in enumerate(records,1):
            if ordinal%2000==0:print('hypermem lifecycle: archive records '+str(ordinal)+'/'+str(len(records)),flush=True)
            if not rec['retained'] or rec['member'] in written:continue
            data=Path(rec['path']).read_bytes();require(hashlib.sha256(data).hexdigest()==rec['sha256'],'Source changed while archiving')
            z.writestr(rec['member'],data);written.add(rec['member'])
    require(inventories()==before,'Run changed during archive')
    for path,h in evidence.items():require(sha(path)==h,'Dependency changed during archive')
    print('hypermem lifecycle: actual ZIP-only replay',flush=True)
    with Resolver(target,mp) as view:replay=strict_replay(view)
    rp=OUTPUT/'archive_replay.json';save_new(rp,replay)
    keep={p:r['sha256'] for p,r in allfiles.items() if not any(Path(p).is_relative_to(w) for w in work_roots())}
    keep.update({str(x):sha(x) for x in (mp,target,rp)})
    plan={'schema':PLAN_SCHEMA,'method':'hypermem','created_utc':now(),'workspace':str(ROOT),'allowed_delete_roots':[str(w) for w in work_roots()],
        'archive':binding(target),'archive_manifest':binding(mp),'archive_replay':binding(rp),'archive_crc_verified':True,
        'result_review':am['result_review'],'tool_review':am['tool_review'],'bank_review':am['bank_review'],
        'keep_sha256':keep,'code_sha256':code_hashes(),'work_inventories':am['work_inventories'],'deletion_executed':False,'scope':SCOPE}
    save_new(DEFAULT_PLAN,plan);return {'status':'archive_verified_requires_independent_cleanup_review','plan':binding(DEFAULT_PLAN)}


def verify_archive(*,require_work=True):
    stopped();plan=read(safe(DEFAULT_PLAN))
    require(plan['schema']==PLAN_SCHEMA and plan['method']=='hypermem' and plan['workspace']==str(ROOT)
        and plan['allowed_delete_roots']==[str(w) for w in work_roots()] and plan['code_sha256']==code_hashes()
        and plan['archive_crc_verified'] is True and plan['scope']==SCOPE,'Wrong two-root plan or tool sources')
    tool_review(bound(plan['tool_review']))
    for p,h in plan['keep_sha256'].items():
        require(not any(Path(p).is_relative_to(w) for w in work_roots()),'Keep points into work');bound({'path':p,'sha256':h})
    for key,name in [('archive','evidence.zip'),('archive_manifest','archive_manifest.json'),('archive_replay','archive_replay.json')]:require(bound(plan[key])==OUTPUT/name,'Foreign archive output')
    with Resolver(plan['archive']['path'],plan['archive_manifest']['path']) as view:
        require(view.manifest['work_inventories']==plan['work_inventories'] and view.manifest['code_sha256']==plan['code_sha256'],'Plan/archive metadata differs')
        for key in ('result_review','tool_review','bank_review'):require(view.manifest[key]==plan[key],'Review binding differs')
        if require_work:
            for k,r in roots().items():require(inventory(r/'work')==plan['work_inventories'][k],'Work bytes changed before deletion')
        recorded=read(plan['archive_replay']['path'])
        require(strict_replay(view,runtime=recorded['runtime'])==recorded,'Saved full archive replay differs')
    return plan


def cleanup_review(path,plan):
    v=read(safe(path));require(v.get('schema')=='hypermem-native-v2-cleanup-review-v1' and v.get('status')=='passed'
        and v.get('cleanup_plan_sha256')==sha(DEFAULT_PLAN) and v.get('archive_sha256')==plan['archive']['sha256']
        and v.get('scope')==SCOPE and v.get('allowed_delete_roots')==plan['allowed_delete_roots'],'Independent exact cleanup review required')
    return binding(path)


def removal_evidence(plan,review):
    intent=binding(OUTPUT/'cleanup_execution_intent.json');iv=read(intent['path'])
    require(iv.get('schema')=='hypermem-native-v2-cleanup-intent-v1' and iv.get('targets')==plan['allowed_delete_roots']
        and iv.get('cleanup_plan_sha256')==sha(DEFAULT_PLAN) and iv.get('cleanup_review_sha256')==review['sha256'],'Removal intent differs')
    receipts=[]
    for k,r in roots().items():
        require(not safe(r/'work').exists(),'Work remains or reappeared')
        b=binding(OUTPUT/('removal_'+k+'.json'));v=read(b['path'])
        require(v.get('schema')=='hypermem-native-v2-removal-receipt-v1' and v.get('removed_path')==str(r/'work')
            and v.get('cleanup_plan_sha256')==sha(DEFAULT_PLAN) and v.get('cleanup_review_sha256')==review['sha256'],'Removal receipt differs')
        receipts.append(b)
    return intent,receipts


def completion_values(plan,review,intent,receipts):
    return {'schema':'hypermem-native-v2-cleanup-receipt-v1','method':'hypermem','cleanup_plan':binding(DEFAULT_PLAN),'cleanup_review':review,
        'archive':plan['archive'],'archive_manifest':plan['archive_manifest'],'intent':intent,'removal_receipts':receipts,'scope':SCOPE}


def complete(review_path):
    plan=verify_archive(require_work=False);review=cleanup_review(review_path,plan);intent,receipts=removal_evidence(plan,review)
    receipt=completion_values(plan,review,intent,receipts);receipt['completed_utc']=now();save_new(OUTPUT/'cleanup_receipt.json',receipt)
    completion={'schema':'hypermem-native-v2-predecessor-completion-v1','status':'passed','method':'hypermem','result_owner':str(NEW),
        'summary_sha256':sha(NEW/'results/summary.json'),'result_review':plan['result_review'],'cleanup_receipt':binding(OUTPUT/'cleanup_receipt.json'),'scope':SCOPE}
    save_new(OUTPUT/'predecessor_completion.json',completion);return {'status':'complete','scope':SCOPE}


def verify_completion():
    plan=verify_archive(require_work=False);receipt=read(OUTPUT/'cleanup_receipt.json');review=cleanup_review(bound(receipt['cleanup_review']),plan)
    intent,receipts=removal_evidence(plan,review)
    require({k:v for k,v in receipt.items() if k!='completed_utc'}==completion_values(plan,review,intent,receipts),'Aggregate receipt differs')
    expected={'schema':'hypermem-native-v2-predecessor-completion-v1','status':'passed','method':'hypermem','result_owner':str(NEW),
        'summary_sha256':sha(NEW/'results/summary.json'),'result_review':plan['result_review'],'cleanup_receipt':binding(OUTPUT/'cleanup_receipt.json'),'scope':SCOPE}
    require(read(OUTPUT/'predecessor_completion.json')==expected,'Strong completion differs')
    return {'status':'passed','method':'hypermem','result_owner':str(NEW),'scope':SCOPE}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('certify','archive','verify','complete','verify-completion'))
    for name in ('result-review','tool-review','bank-review','cleanup-review'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.command=='certify':v=certify(a.result_review,a.tool_review)
    elif a.command=='archive':v=archive(a.result_review,a.tool_review,a.bank_review)
    elif a.command=='verify':verify_archive();v={'status':'passed'}
    elif a.command=='complete':v=complete(a.cleanup_review)
    else:v=verify_completion()
    print(json.dumps(v,ensure_ascii=True))


def code_hashes():
    require(sha(ROOT/'tools/archive_simplemem_native_v2.py')==FROZEN_SIMPLE_SHA,'Reviewed lifecycle foundation changed')
    return {**core.chain.code_hashes(),**{p:sha(ROOT/p) for p in (*core.chain.base.SHARED_FILES,
        'tools/baselines_hypermem_v1.py',*core.CORE,*EXTRA)}}


def evidence_closure(initial,*,annotations=None):
    # Carry forward ONLY the already reviewed exact document classifications.
    require(sha(ROOT/'tools/archive_simplemem_native_v2.py')==FROZEN_SIMPLE_SHA,'Frozen typed closure changed')
    return simple.evidence_closure(initial,annotations=annotations)


def disposition(path):
    p=Path(path)
    for r in roots().values():
        if p.is_relative_to(r/'work'):
            rel=p.relative_to(r/'work').parts
            native=len(rel)>=4 and rel[0]=='native' and re.fullmatch('c[0-9]',rel[1])
            if native and len(rel)==4 and rel[2]=='cache' and rel[3] in INDEXES.values():
                return 'runnable_hypermem_bm25_vector_index'
            if native and len(rel)==4 and rel[2]=='embedding_evidence' and re.fullmatch(r'(document|query)_[0-9a-f]{64}\.json',rel[3]):
                return 'runnable_normalized_native_embedding_cache'
            if '__pycache__' in rel and p.suffix=='.pyc':return 'disposable_python_bytecode'
            require(p.suffix!='.tmp','Temporary artifact needs separate exact qualification: '+str(p))
            require(p.suffix in ('.json','.jsonl','.txt','.sha256'),'Unclassified work artifact: '+str(p))
    # Research preflight banks are NOT additional deletion roots or silent ZIP
    # members. A separate explicit disposition is needed if they ever appear.
    if p.suffix in ('.pkl','.lance') or 'embedding_evidence' in p.parts and re.fullmatch(r'(document|query)_[0-9a-f]{64}\.json',p.name):
        raise ValueError('External runnable research bank is outside this two-root disposition')
    if 'lancedb' in p.parts:return simple.disposition(p)
    return None


def result_review(path,view=None,*,live=False):
    view=view or seal.LiveView();v=view.read(path)
    require(not any(safe(path).is_relative_to(p) for p in roots().values()),'Result review must be outside both runs')
    require(v.get('schema')==RESULT_SCHEMA and v.get('status')=='passed' and v.get('method')=='hypermem','Independent actual HyperMem result review required')
    inv=v.get('reviewed_run_inventories',{});require(set(inv)==set(roots()),'Both full run inventories required')
    for k,r in roots().items():
        check_inventory(inv[k],r)
        if live:require(hashes(inventory(r))==inv[k],'Current independently reviewed run inventory differs')
    m=view.read(NEW/'run_manifest.json');p=seal.bound(view,m['terminal_proof']);s=view.read(NEW/'results/summary.json')
    require(m['original_run']==str(OLD) and m['new_run']==str(NEW) and m['schema']==core.SCHEMA,'Wrong continuation manifest')
    require(m['parent_manifest']=={'path':str(OLD/'run_manifest.json'),'sha256':view.sha(OLD/'run_manifest.json')}
            and inv['original']==p['original_inventory'],'Original immutable source differs')
    require(view.sha(NEW/'results/summary.json')==v.get('summary_sha256') and s.get('schema')=='five-baseline-result-v1'
            and s.get('method')=='hypermem' and s.get('audit_status')=='no_invalid_artifacts'
            and s.get('effective',{}).get('slots')==1540,'HyperMem final cohort is incomplete')
    for phase in ('firstpass','effective'):
        counts=s.get(phase,{})
        require(counts.get('slots')==1540 and all(type(counts.get(k)) is int and 0<=counts[k]<=1540 for k in ('reader_valid','judge_valid'))
            and counts.get('missing_reader')==1540-counts['reader_valid'] and counts.get('missing_judge')==1540-counts['judge_valid'],
            'Valid/missing fixed-denominator accounting differs')
    state=view.read(NEW/'status.json')
    require(state.get('stage')=='completed_awaiting_result_review' and state.get('summary_sha256')==v['summary_sha256'],'Exact successful terminal status required')
    require(view.read(OLD/'status.json').get('stage')=='needs_review','Original failed run was overwritten')
    for source,h in {**m['code_sha256'],**m['upstream_sha256']}.items():require(view.sha(ROOT/source)==h,'Scientific source differs')
    parent=seal.bound(view,m['parent_manifest']);require(m['questions']==p['questions']==parent['questions'] and len(m['questions'])==1540,'Full fixed cohort differs')
    cr=seal.bound(view,m['review'])
    require(cr.get('schema')==core.REVIEW_SCHEMA and cr.get('status')=='passed' and cr.get('proof_sha256')==m['terminal_proof']['sha256']
            and cr.get('code_sha256')==m['code_sha256'] and cr.get('upstream_sha256')==m['upstream_sha256']
            and cr.get('specification')==m['specification'],'Independent core admission differs')
    for source,h in cr.get('evidence_sha256',{}).items():require(view.sha(ROOT/source)==h,'Core review evidence differs')
    return m,p,s,v


def banks_for(proof,invs):
    require(hashes(invs['original'])==proof['original_inventory'],'Original whole inventory differs')
    banks={}
    for ci in range(10):
        native=NEW/f'work/native/c{ci}';state=read(native/'state.json')
        core.adapter._runtime._verify_state(state)
        core.adapter._runtime._verify_artifacts(native,state['document_embedding_sha256'])
        indexes={}
        for name,filename in INDEXES.items():
            p=native/'cache'/filename;require(sha(p)==state['cache_sha256'][name],'Actual native index bytes differ')
            indexes[str(p)]=invs['continuation'][str(p)]
        embeddings={}
        for p in sorted((native/'embedding_evidence').glob('*.json')):
            core.adapter._runtime._verify_seal(p);value=read(p)
            require(set(value)=={'identity','status','vectors'} and value['status']=='success','Unknown native embedding cache layout')
            require(p.name==('query_' if value['identity']['query'] else 'document_')+core.adapter._runtime._sha(value['identity'])+'.json','Native embedding cache identity path differs')
            embeddings[str(p)]={'original_binding':binding(p),'file':invs['continuation'][str(p)],
                'identity':value['identity'],'status':'success','normalized_vectors_sha256':digest(value['vectors']),
                'seal':binding(core.adapter._runtime._seal_path(p))}
        require(embeddings,'Actual embedding records required')
        banks[str(native)]={'conversation':ci,'owner':'continuation','state':binding(native/'state.json'),
            'index_files':indexes,'cache_sha256':state['cache_sha256'],
            'document_embedding_sha256':state['document_embedding_sha256'],'embeddings':embeddings}
    return banks


def certify(result_path,tool_path):
    result_path=safe(result_path);tool_path=safe(tool_path)
    stopped();tool_review(tool_path);require(not (OUTPUT/'bank_validation.json').exists(),'Certificate already exists')
    m,p,s,_=result_review(result_path,live=True);before=inventories()
    require(m==core.expected_manifest(m['terminal_proof']['path'],p,m['review']['path']),'Complete core manifest gate differs')
    require(terminal.validate_proof(m['terminal_proof']['path'])==p,'Parent terminal proof changed')
    original_audit=audit.audit_all(NEW,p)  # unmodified, actual indexes/native vectors
    require(original_audit==read(NEW/'results/raw_request_audit.json'),'Live frozen full audit differs')
    outputs=reconstruct(seal.LiveView(),m,p,created_utc=s['created_utc'],raw_audit=original_audit)
    for name,value in outputs.items():require(value==read(NEW/'results'/name),'Live frozen seal output differs')
    banks=banks_for(p,before)
    require(inventories()==before,'Runs changed during actual certification');stopped()
    value={'schema':CERT_SCHEMA,'status':'passed','created_utc':now(),'scope':'actual_predelete_bytes_and_unmodified_frozen_full_audits',
        'result_review':binding(result_path),'tool_review':binding(tool_path),'manifest':binding(NEW/'run_manifest.json'),'proof':m['terminal_proof'],
        'run_inventories':before,'banks':banks,'output_sha256':{str(NEW/'results'/n):sha(NEW/'results'/n) for n in OUTPUTS},
        'code_sha256':code_hashes(),'inventory_before_after_equal':True,'database_engine_opened':False,'retrieval_search_reexecuted':False,
        'model_calls':0,'credential_reads':0,'postdelete_scope':SCOPE,
        'embedding_certificate_semantics':'Identity, original byte SHA, seal and normalized-vector digest only. Vector values are not stored in this certificate; postdelete normalization is recomputed from actual retained provider responses.'}
    save_new(OUTPUT/'bank_validation.json',value)
    return {'status':'certificate_created_requires_independent_review','certificate':binding(OUTPUT/'bank_validation.json')}


def bank_structure(view,proof,cert):
    require(set(cert['banks'])=={str(NEW/f'work/native/c{ci}') for ci in range(10)},'Extra/missing actual bank certificate')
    for ci in range(10):
        native=NEW/f'work/native/c{ci}';rec=cert['banks'][str(native)];state=view.read(native/'state.json')
        require(rec['conversation']==ci and rec['owner']=='continuation' and rec['state']=={'path':str(native/'state.json'),'sha256':view.sha(native/'state.json')}
            and rec['cache_sha256']==state['cache_sha256'] and rec['document_embedding_sha256']==state['document_embedding_sha256'],'Certified bank metadata differs')
        inv=cert['run_inventories']['continuation']
        expected_indexes={str(native/'cache'/filename):inv[str(native/'cache'/filename)] for filename in INDEXES.values()}
        require(rec['index_files']==expected_indexes,'Exact certified index set differs')
        for name,filename in INDEXES.items():require(rec['index_files'][str(native/'cache'/filename)]['sha256']==state['cache_sha256'][name],'State/index byte certificate differs')
        expected_embeddings={p:v for p,v in inv.items() if Path(p).parent==native/'embedding_evidence'}
        require(set(rec['embeddings'])==set(expected_embeddings) and expected_embeddings,'Exact native embedding set differs')
        documents={}
        for path,e in rec['embeddings'].items():
            p=Path(path);identity=e['identity']
            require(e['file']==expected_embeddings[path] and e['original_binding']=={'path':path,'sha256':e['file']['sha256']}
                    and e['status']=='success' and re.fullmatch('[0-9a-f]{64}',e['normalized_vectors_sha256']), 'Certified embedding byte/digest layout differs')
            require(p.name==('query_' if identity['query'] else 'document_')+core.adapter._runtime._sha(identity)+'.json','Certified identity filename differs')
            sealpath=p.parent/'.integrity'/(p.name+'.sha256')
            require(e['seal']=={'path':str(sealpath),'sha256':view.sha(sealpath)} and view.bytes(sealpath).decode('ascii')==e['file']['sha256'],'Retained native seal differs')
            if not identity['query']:documents[str(p.relative_to(native))]=e['file']['sha256']
        require(documents==state['document_embedding_sha256'],'State bound document vector map differs')


def view_functions(view):
    P=Path if isinstance(view,seal.LiveView) else virtual_paths(view)
    def bind(p):return {'path':str(P(p).resolve()),'sha256':view.sha(p)}
    def lines(p):
        if not P(p).exists():return []
        value=view.bytes(p);require(not value or value.endswith(b'\n'),'Incomplete terminal JSONL')
        return [json.loads(x) for x in value.splitlines() if x.strip()]
    def inv(folder):
        return {str(p):view.sha(p) for p in sorted(P(folder).rglob('*')) if p.is_file()}
    return P,bind,lines,inv


def private_runtime(view):
    runtime=core.adapter._runtime;P,bind,lines,inv=view_functions(view)
    namespace={**vars(runtime),'Path':P,'_file_sha':view.sha,'_read_json':view.read}
    for name in ('_seal_path','_verify_seal','_verify_artifacts','_verify_state'):
        fn=getattr(runtime,name);namespace[name]=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    return SimpleNamespace(**namespace)


def qa_namespace(view,m):
    base=terminal.chain.base;P,bind,lines,inv=view_functions(view)
    class Replay(base.RawReplay):
        generate=shared_replay.clone(base.RawReplay.generate,read=view.read,sha=view.sha,Path=P)
    ns={**vars(base),'Path':P,'read':view.read,'sha':view.sha,'RawReplay':Replay,'paths':lambda method:P(NEW),'manifest':lambda method:m,
        'load_data':lambda:view.read(ROOT/'data/locomo10.json')}
    for name in ('known_recoverable','replay_qa_leaf'):
        f=getattr(base,name);ns[name]=FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
    return ns


def archived_auditors(view,cert,m):
    require(view.sha(ROOT/'tools/audit_hypermem_continuation_v2.py')==FROZEN_AUDIT_SHA
            and view.sha(ROOT/'tools/baselines_hypermem_v1.py')==FROZEN_RUNTIME_SHA,'Frozen auditor/runtime source differs')
    P,bind,lines,inv=view_functions(view);runtime=private_runtime(view);qa=qa_namespace(view,m)
    adm=SimpleNamespace(**{**vars(terminal),'OLD':P(OLD),'inventory':inv,
        'chain':SimpleNamespace(base=SimpleNamespace(**qa))})
    cm=SimpleNamespace(**{**vars(audit.common),'Path':P,'read':view.read,'file_sha':view.sha,'binding':bind,'jsonl':lines})
    checked=[]
    def bank(native):return cert['banks'][str(native)]
    def documents(native,state):
        require(bank(native)['document_embedding_sha256']==state['document_embedding_sha256'],'Certified document map differs')
        checked.append(('documents',str(native)))
    def index(native,name,filename,state):
        r=bank(native)['index_files'][str(native/'cache'/filename)]
        require(r['sha256']==state['cache_sha256'][name],'Certified index digest differs');checked.append(('index',str(native/'cache'/filename)))
    def embedding_paths(native):return [P(p) for p in sorted(bank(native)['embeddings'])]
    def embedding(p):
        rec=bank(p.parent.parent)['embeddings'][str(p)]
        require(view.bytes(rec['seal']['path']).decode('ascii')==rec['original_binding']['sha256'],'Retained embedding seal differs')
        checked.append(('embedding',str(p)));return rec
    text=function_source(view,'tools/audit_hypermem_continuation_v2.py','native_audit')
    changes=[("        runtime._verify_artifacts(native,state['document_embedding_sha256'])","        verify_certified_documents(native,state)"),
        ("            require(sha(native/'cache'/filename)==state['cache_sha256'][name],'Native index changed')","            verify_certified_index(native,name,filename,state)"),
        ("        for p in sorted((native/'embedding_evidence').glob('*.json')):\n            runtime._verify_seal(p); value=read(p); ident=value['identity']",
         "        for p in certified_embedding_paths(native):\n            value=certified_embedding(p); ident=value['identity']"),
        ("            require(value['vectors']==vectors,'Original normalized native embedding values differ')\n            native_bindings.append(binding(p))",
         "            require(vector_digest(vectors)==value['normalized_vectors_sha256'],'Original normalized native embedding digest differs')\n            native_bindings.append(value['original_binding'])")]
    altered=text
    for before,after in changes:
        require(altered.count(before)==1,'Exact archival certificate substitution boundary differs');altered=altered.replace(before,after)
    namespace={**vars(audit),'Path':P,'read':view.read,'sha':view.sha,'binding':bind,'jsonl':lines,'admission':adm,'common':cm,
        'adapter':SimpleNamespace(**{**vars(core.adapter),'_runtime':runtime}), 'verify_certified_documents':documents,
        'verify_certified_index':index,'certified_embedding_paths':embedding_paths,'certified_embedding':embedding,'vector_digest':digest}
    for name in ('api_audit','audit_all'):
        f=getattr(audit,name);namespace[name]=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
    exec(compile(altered,'<explicit-certified-hypermem-native-cache-audit>','exec'),namespace)
    transforms={'frozen_source_sha256':FROZEN_AUDIT_SHA,'original_function_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'archival_function_sha256':hashlib.sha256(altered.encode()).hexdigest(),'exact_substitutions':[{'before':a,'after':b} for a,b in changes],
        'scope':SCOPE,'remaining_native_and_api_functions_unchanged':True,
        'original_native_audit_scope_is_historical_certified_value':True,
        'historical_scope_interpretation':'The unchanged native.scope field in the reproduced historical raw_request_audit describes its actual predelete audit. It is not a claim that this archive read disposed bytes or reran retrieval.'}
    return namespace['audit_all'],transforms,checked


def reconstruct(view,m,proof,*,created_utc,raw_audit):
    """The original seal and exact frozen summary projection, without writes."""
    ns=qa_namespace(view,m);outputs={};base=terminal.chain.base
    def capture(path,value):
        require(Path(path).parent==NEW/'results' and Path(path).name in OUTPUTS,'Unexpected seal output path')
        require(Path(path).name not in outputs,'Duplicate seal output')
        outputs[Path(path).name]=value
    # Extract the actual frozen nested save hook instead of reimplementing the
    # usage/technical-continuation projection. Only its I/O is a memory sink.
    require(view.sha(ROOT/'tools/run_five_baselines_hypermem_native_v2.py')=='7f9d827f19812bdf775ceb0062b3945690dc9900b63d14bde02c213b0ff4f3f7','Frozen continuation controller differs')
    source=view.bytes(ROOT/'tools/run_five_baselines_hypermem_native_v2.py').decode('utf8')
    tree=ast.parse(source);outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='private_controller')
    hook=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='result_save')
    module=ast.Module(body=[hook],type_ignores=[])
    hookns={**vars(core),'NEW':NEW,'OLD':OLD,'m':m,'original_save':capture,'latest_audit':{'value':raw_audit},'binding':lambda p:{'path':str(p),'sha256':view.sha(p)}}
    exec(compile(ast.fix_missing_locations(module),'<unchanged-hypermem-summary-projection-memory-sink>','exec'),hookns)
    ns.update(save_new=hookns['result_save'],now=lambda:created_utc,audit_requests=lambda folder:raw_audit)
    runseal=FunctionType(base.seal.__code__,ns,'seal',base.seal.__defaults__)
    runseal('hypermem');require(set(outputs)==set(OUTPUTS),'Incomplete original four-output seal')
    return outputs


def audit_original(view,proof):
    """Same hash-locked original terminal body; ZIP paths replace file reads."""
    P,bind,lines,inv=view_functions(view);source=view.bytes(ROOT/terminal.AUDIT_SOURCE).decode('utf-8-sig')
    require(hashlib.sha256(view.bytes(ROOT/terminal.AUDIT_SOURCE)).hexdigest()==terminal.AUDIT_SHA,'Original terminal audit source differs')
    tree=ast.parse(source);block=next(n for n in tree.body if isinstance(n,ast.With));keep=[];removed=[]
    for n in block.body:
        names={x.id for x in ast.walk(n) if isinstance(x,ast.Name)}
        if (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('probe','process_rows') for x in n.targets)
            or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and (isinstance(n.value.func,ast.Name) and n.value.func.id=='print'
                or isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and 'process_rows' in names)
            or isinstance(n,ast.With) and 'OUT' in names):removed.append(n)
        else:keep.append(n)
    require(len(removed)==6,'Exact prior reviewed report/PID exclusion changed')
    import time
    from jev_mem.gemini25_ablation_provider_v1 import generation_body
    from jev_mem.sdk20_cached_api import digest as signature_digest
    from datetime import timezone
    env={'ROOT':P(ROOT),'RUN':P(OLD),'Path':P,'base':terminal.chain.base,'adapter':core.adapter._runtime,
        'shared_read':view.bytes,'read':view.read,'sha':view.sha,'binding':bind,'jsonl':lines,'inventory':lambda:inv(OLD),
        'before':inv(OLD),'require':require,'Counter':Counter,'ast':ast,'re':re,'json':json,'hashlib':hashlib,
        'generation_body':generation_body,'digest':signature_digest,'datetime':datetime,'timezone':timezone,'time':time,'start':time.monotonic()}
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ('unique','dt'):
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<original-terminal-pure-helpers>','exec'),env)
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[])),'<original-terminal-full-body-zip-only>','exec'),env)
    r=env['result'];require(r['original_inventory']==proof['original_inventory'] and r['rows']==proof['rows'],'Original ten full producer/native/source replay differs')
    require(terminal.usage_presence([x['physical'] for x in r['rows']])==proof['original_physical_usage'],'Original physical usage presence differs')
    return {'status':'passed','physical_producers':10,'logical_consumers':10,'source_inventory_sha256':digest(r['original_inventory']),
        'original_body_source_sha256':terminal.AUDIT_SHA,'excluded_ast_nodes':6,'excluded_scope':'Same frozen historical PID and report-publication nodes only'}


def replay_all(view):
    am=view.manifest;m,p,s,review=result_review(am['result_review']['path'],view)
    for record in am['bindings'].values():seal.bound(view,record)
    cert=verify_certificate(view,am['bank_review']['path'],am['result_review']['path'],am['tool_review']['path'])
    require(cert['manifest']=={'path':str(NEW/'run_manifest.json'),'sha256':view.sha(NEW/'run_manifest.json')}
            and cert['proof']==m['terminal_proof'],'Certificate source differs')
    for owner,r in roots().items():
        actual={rec['path']:rec['sha256'] for rec in am['records'] if Path(rec['path']).is_relative_to(r)}
        require(actual==review['reviewed_run_inventories'][owner]==hashes(cert['run_inventories'][owner]),'Whole two-root archived inventory differs')
    bank_structure(view,p,cert);parent=audit_original(view,p)
    auditor,transforms,checked=archived_auditors(view,cert,m);raw=auditor(virtual_paths(view)(NEW),p)
    outputs=reconstruct(view,m,p,created_utc=s['created_utc'],raw_audit=raw)
    for name,value in outputs.items():require(value==view.read(NEW/'results'/name),'Original archived full output differs: '+name)
    expected=Counter([('documents',str(NEW/f'work/native/c{ci}')) for ci in range(10)])
    for native,bank in cert['banks'].items():
        expected.update(('index',p) for p in bank['index_files']);expected.update(('embedding',p) for p in bank['embeddings'])
    require(Counter(checked)==expected,'Incomplete/extra certificate consumption')
    return {'status':'passed','scope':SCOPE,'parent_request_audit':parent,'replayed_rows':{n:len(outputs[n]) for n in OUTPUTS[:2]},
        'output_content_sha256':{n:digest(v) for n,v in outputs.items()},'explicit_private_transformations':transforms,
        'bank_certificate':am['bindings']['bank_certificate'],'bank_byte_checks_after_delete':0,'database_searches':0,'model_calls':0,
        'historical_native_audit_scope':raw['native']['scope'],
        'historical_scope_interpretation':transforms['historical_scope_interpretation']}


if __name__=='__main__':main()
