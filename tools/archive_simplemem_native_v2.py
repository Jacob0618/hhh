"""Two-root SimpleMem evidence lifecycle; no provider or database is opened.

The original live auditor is run before disposition. After deletion a precisely
declared private auditor replaces ONLY its two-line DB-byte check with an
independently reviewed predelete certificate. Disposed files never have virtual
bytes or a synthetic sha(). Text/native/API/QA replay still reads real ZIP bytes.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import FunctionType, SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import baseline_lifecycle_v3 as single
from tools import baseline_evidence_closure_v1 as typed
from tools import archive_amem_metric_closure_v2 as metric
from tools import run_five_baselines_simplemem_native_v2 as core
from tools import seal_simplemem_native_v2 as seal
from tools import audit_simplemem_terminal_v2 as terminal
from tools import baseline_controller_guard_v3 as guard
from tools.baselines_common_v1 import read,sha,save_new,now

OLD=core.OLD;NEW=core.NEW
OUTPUT=ROOT/'research/baseline_execution_2026-10-08/simplemem_native_v2_lifecycle'
DEFAULT_PLAN=OUTPUT/'cleanup_plan.json'
SCHEMA='simplemem-native-v2-two-root-archive-v1'
PLAN_SCHEMA='simplemem-native-v2-two-root-cleanup-plan-v1'
REVIEW_SCHEMA='simplemem-native-v2-lifecycle-implementation-review-v1'
RESULT_SCHEMA='simplemem-native-v2-result-review-v1'
CERT_SCHEMA='simplemem-native-v2-predelete-bank-validation-v1'
BANK_REVIEW_SCHEMA='simplemem-native-v2-bank-validation-review-v1'
EXTRA=('tools/archive_simplemem_native_v2.py','tools/cleanup_simplemem_native_v2.ps1',
       'tools/run_five_baselines_reviewed_chain_v5.py')
OUTPUTS=('firstpass_per_query.json','completed_firstpass_per_query.json','per_query.json',
         'raw_request_audit.json','method_request_audit.json','summary.json')
DB_BEFORE="""        actual_copy={str(path):view.sha(path) for path in P(bank/'lancedb').rglob('*') if path.is_file()}
        require(actual_copy==expected_copy,'Cloned database bytes or exact file set changed')"""
DB_AFTER="        verify_predelete_bank(bank, original, lineage, expected_copy)"
SEAL_BEFORE='    from tools import run_five_baselines_simplemem_native_v2 as controller'
SEAL_AFTER='    controller = archival_controller'
SCOPE='real ZIP raw/native/provenance/QA replay; disposed database bytes validated before deletion under independent certificate; no DB search replay'
NEGATIVE='research/baseline_execution_2026-10-08/heartbeat_20261009T0603/independent_admission_fixture_v2/wrong_code_fixture_review.json'
NEGATIVE_SHA='7337ace52b7508b008595fa060d7da021a723547a8c0021716215ca95ded26f2'
POSITIVE='research/baseline_execution_2026-10-08/heartbeat_20261009T0603/independent_admission_fixture_v2/fixture_review.json'
POSITIVE_SHA='9f23de6782890203f7d41120ce0ada3e5268491bee52a611166def915fbcd605'
NEGATIVE_DECISION='research/baseline_execution_2026-10-08/heartbeat_20261009T0603/root_negative_fixture_scope_v1.json'
NEGATIVE_DECISION_SHA='a52df1adb8e0c75c2c1d4765bf2853b5655854d595413f91d5b1692cc40bb781'
TYPED_SHA='8a478e382bfdef41a3ecc4af1f559a53690de4f7dee5d2b3111065bf1d1591a8'
LANCEDB_FTS_SOURCE='runtime/baseline_env_v1/Lib/site-packages/lancedb/fts.py'
LANCEDB_FTS_SOURCE_SHA='960450edf67d4870d7ee714373ef78fa51c7ab7e68d41e66c9b895c1cce5f0f7'
_GUARD_ACTIVE=guard.active_controllers
_GUARD_STOP=guard.assert_stopped
active_controllers=FunctionType(_GUARD_ACTIVE.__code__,{**_GUARD_ACTIVE.__globals__,
    'LAUNCHERS':(*guard.LAUNCHERS,'run_five_baselines_reviewed_chain_v4','run_five_baselines_reviewed_chain_v5')},
    'active_controllers',_GUARD_ACTIVE.__defaults__)
_stop=FunctionType(_GUARD_STOP.__code__,{**_GUARD_STOP.__globals__,'active_controllers':active_controllers},
    'assert_stopped',_GUARD_STOP.__defaults__)

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
def code_hashes():return {**metric.code_hashes(),**{p:sha(ROOT/p) for p in (*core.CORE,*EXTRA)}}

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

def result_review(path,view=None,*,live=False):
    view=view or seal.LiveView();v=view.read(path)
    require(not any(safe(path).is_relative_to(p) for p in roots().values()),'Result review must be outside both runs')
    require(v.get('schema')==RESULT_SCHEMA and v.get('status')=='passed' and v.get('method')=='simplemem','Independent union result review required')
    inv=v.get('reviewed_run_inventories',{});require(set(inv)==set(roots()),'Both full run inventories required')
    for k,r in roots().items():
        check_inventory(inv[k],r)
        if live:require(hashes(inventory(r))==inv[k],'Current reviewed run inventory differs')
    m=view.read(NEW/'run_manifest.json');p=seal.bound(view,m['proof']);s=view.read(NEW/'results/summary.json')
    require(m['self_root']==str(NEW) and m['parent_root']==str(OLD) and m['schema']==core.SCHEMA,'Wrong union manifest')
    require(inv['original']==p['original_inventory'],'Original immutable run changed')
    require(view.sha(NEW/'results/summary.json')==v.get('summary_sha256') and s.get('schema')==seal.SCHEMA
        and s.get('self_root')==str(NEW) and s.get('parent_root')==str(OLD) and s.get('audit_status')=='no_invalid_artifacts'
        and all(s.get('effective',{}).get(k)==1540 for k in ('slots','reader_valid','judge_valid')),'Union result not complete')
    state=view.read(NEW/'status.json')
    require(state.get('stage')=='complete_awaiting_result_review' and state.get('summary_sha256')==v['summary_sha256'],'Exact terminal status required')
    require(view.read(OLD/'status.json').get('stage')=='needs_review','Original failure status was overwritten')
    for path,h in {**m['code_sha256'],**m['upstream_sha256']}.items():require(view.sha(ROOT/path)==h,'Scientific source differs')
    require(m['questions']==p['questions'] and len(m['questions'])==1540,'Fixed cohort differs')
    cr=seal.bound(view,m['independent_review'])
    require(cr.get('schema')==core.REVIEW_SCHEMA and cr.get('status')=='passed' and cr.get('proof')==m['proof']
        and cr.get('code_sha256')==m['code_sha256'] and cr.get('upstream_sha256')==m['upstream_sha256']
        and cr.get('archive_cleanup_downstream_authorized') is False,'Core independent admission changed')
    return m,p,s,v

def disposition(path):
    p=Path(path)
    for r in roots().values():
        if p.is_relative_to(r/'work'):
            rel=p.relative_to(r/'work').parts
            if len(rel)>=5 and rel[0]=='native' and re.fullmatch('c[0-9]',rel[1]) and rel[2:4]==('simplemem','lancedb'):
                return 'runnable_lancedb_vector_and_fts_index'
            if '__pycache__' in rel and p.suffix=='.pyc':return 'disposable_python_bytecode'
            require(p.suffix!='.tmp','Temporary file requires explicit evidence qualification: '+str(p))
            require(p.suffix in ('.json','.jsonl','.txt'),'Unclassified work artifact: '+str(p))
    if 'lancedb' in p.parts and not any(p.is_relative_to(r/'work') for r in roots().values()):
        if p==ROOT/LANCEDB_FTS_SOURCE:
            require(sha(p)==LANCEDB_FTS_SOURCE_SHA,'Pinned LanceDB FTS Python source changed')
            return None
        raise ValueError('External runnable research bank requires a separate reviewed disposition; never archive silently')
    return None

def banks_for(proof,invs):
    result={}
    for ci,bank in proof['banks'].items():
        old=Path(bank['bank_dir']);new=NEW/'work/native'/('c'+ci)/'simplemem'
        expected={str(new/'lancedb'/Path(p).relative_to(old/'lancedb')):h for p,h in bank['lancedb_inventory'].items()}
        for owner,directory,wanted in [('original',old,bank['lancedb_inventory']),('continuation',new,expected)]:
            actual={p:rec for p,rec in invs[owner].items() if Path(p).is_relative_to(directory/'lancedb')}
            require(hashes(actual)==wanted and bool(actual),'Terminal bank bytes differ')
            result[str(directory)]={'owner':owner,'database_files':actual,'state':binding(directory/'state.json'),
                'original_bank':str(old),'original_database_inventory':bank['lancedb_inventory'],
                'database_integrity':bank['database_integrity']}
    return result

def certify(result_path,tool_path):
    stopped();tool_review(tool_path)
    require(not (OUTPUT/'bank_validation.json').exists(),'Certificate already exists')
    m,p,s,review=result_review(result_path,live=True);before=inventories()
    require(m==core.expected_manifest(m['proof']['path'],p,m['independent_review']['path']),'Complete core manifest gate differs')
    # A real complete revalidation, including original DB byte inventory.
    require(terminal.validate_proof(m['proof']['path'])==p,'Terminal proof changed')
    outputs=seal.reconstruct(m,p,created_utc=s['created_utc'])
    for name,value in outputs.items():require(value==read(NEW/'results'/name),'Live frozen seal output differs')
    banks=banks_for(p,before)
    require(inventories()==before,'Runs changed during certificate audit');stopped()
    certificate={'schema':CERT_SCHEMA,'status':'passed','created_utc':now(),'scope':'actual_predelete_bytes_and_unmodified_frozen_full_audits',
        'result_review':binding(result_path),'tool_review':binding(tool_path),'manifest':binding(NEW/'run_manifest.json'),'proof':m['proof'],
        'run_inventories':before,'banks':banks,'output_sha256':{str(NEW/'results'/n):sha(NEW/'results'/n) for n in OUTPUTS},
        'code_sha256':code_hashes(),'inventory_before_after_equal':True,'database_engine_opened':False,'retrieval_search_reexecuted':False,
        'model_calls':0,'credential_reads':0,'postdelete_scope':SCOPE}
    save_new(OUTPUT/'bank_validation.json',certificate)
    return {'status':'certificate_created_requires_independent_review','certificate':binding(OUTPUT/'bank_validation.json')}

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

def bank_structure(view,proof,cert):
    expected_banks=set()
    for ci,b in proof['banks'].items():
        old=Path(b['bank_dir']);new=NEW/'work/native'/('c'+ci)/'simplemem'
        for owner,directory in [('original',old),('continuation',new)]:
            expected_banks.add(str(directory));rec=cert['banks'][str(directory)]
            inv={p:r for p,r in cert['run_inventories'][owner].items() if Path(p).is_relative_to(directory/'lancedb')}
            wanted={str(directory/'lancedb'/Path(p).relative_to(old/'lancedb')):h for p,h in b['lancedb_inventory'].items()}
            require(rec['owner']==owner and rec['database_files']==inv and hashes(inv)==wanted and inv
                and rec['original_bank']==str(old) and rec['original_database_inventory']==b['lancedb_inventory']
                and rec['database_integrity']==b['database_integrity']
                and rec['state']=={'path':str(directory/'state.json'),'sha256':view.sha(directory/'state.json')},'Complete bank certificate metadata differs')
    require(set(cert['banks'])==expected_banks,'Extra/missing original/clone bank certificate')

def function_source(view,path,name):
    raw=view.bytes(ROOT/path).decode('utf-8-sig');tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    require(len(nodes)==1,'Frozen function boundary differs')
    return '\n'.join(raw.splitlines()[nodes[0].lineno-1:nodes[0].end_lineno])+'\n'

def private_function(text,namespace,name,filename):
    code=compile(text,filename,'exec');scope=dict(namespace);exec(code,scope);return scope[name]

def archived_auditors(view,cert):
    calls=[]
    def verify_bank(bank,original,lineage,expected_copy):
        record=cert['banks'].get(str(bank));require(record is not None,'Bank not independently certified')
        require(hashes(record['database_files'])==expected_copy
            and record['original_database_inventory']==original['lancedb_inventory']
            and record['database_integrity']==original['database_integrity']
            and record['state']=={'path':str(bank/'state.json'),'sha256':view.sha(bank/'state.json')},'Predelete bank certificate differs')
        calls.append(str(bank))
    text=function_source(view,'tools/run_five_baselines_simplemem_native_v2.py','audit_sidecar')
    require(text.count(DB_BEFORE)==1,'Exact two-line bank byte boundary differs')
    altered=text.replace(DB_BEFORE,DB_AFTER)
    method=private_function(altered,{**vars(core),'verify_predelete_bank':verify_bank},'audit_sidecar','<explicit-archival-bank-certificate-auditor>')
    source=function_source(view,'tools/seal_simplemem_native_v2.py','reconstruct')
    require(source.count(SEAL_BEFORE)==1,'Exact seal controller import boundary differs')
    reconstruct=private_function(source.replace(SEAL_BEFORE,SEAL_AFTER),{**vars(seal),'archival_controller':SimpleNamespace(audit_sidecar=method)},'reconstruct','<explicit-archival-union-replay>')
    transformations={'method_source_sha256':hashlib.sha256(text.encode()).hexdigest(),'method_archival_sha256':hashlib.sha256(altered.encode()).hexdigest(),
        'seal_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'seal_archival_sha256':hashlib.sha256(source.replace(SEAL_BEFORE,SEAL_AFTER).encode()).hexdigest(),
        'changed_method_lines':DB_BEFORE.splitlines(),'replacement':DB_AFTER,'scope':SCOPE}
    return reconstruct,transformations,calls

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
            self._simplemem_path_type=virtual_paths(self)
        except BaseException:self.zip.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.zip.close()
    def bytes(self,path):
        r=self.records.get(os.path.normcase(str(Path(path).absolute())))
        require(r is not None and r['retained'],'Archive bytes unavailable; no live fallback: '+str(path))
        data=self.zip.read(r['member']);require(len(data)==r['size'] and hashlib.sha256(data).hexdigest()==r['sha256'],'Archive bytes changed');return data
    def read(self,path):return json.loads(self.bytes(path).decode('utf-8-sig'))
    def sha(self,path):return hashlib.sha256(self.bytes(path)).hexdigest()

def audit_original(view,proof):
    a=core._audit_functions(view);reports={};records=None;events=None
    for namespace in ('api_method','api_firstpass'):
        summary,rr,ee=a.audit_api(OLD/'work'/namespace);require(summary==proof['api_audits'][namespace],'Original API audit differs');reports[namespace]=summary
        if namespace=='api_method':records,events=rr,ee
    grouped=defaultdict(list)
    for e in events:grouped[digest(e['context'])].append(e)
    for key,expected in proof['native_audits'].items():
        if key.startswith('build_c'):ctx={'method':'simplemem','conversation':int(key[7:]),'phase':'build'};build=True
        else:
            q=next(q for q in proof['questions'] if q['query_id']==key)
            ctx={'method':'simplemem','conversation':q['conversation'],'phase':'retrieval','query_id':key};build=False
        native=a.audit_native(Path(expected['log']['path']),grouped[digest(ctx)],records,ctx,build=build)
        require({k:v for k,v in native.items() if k!='prefix'}==expected,'Original native audit differs')
    fmt=a.original_formatter();memories={}
    for ci,b in proof['banks'].items():
        bank=Path(b['bank_dir']);state=view.read(bank/'state.json');memory=view.read(bank/'memory.json')
        require(state==b['state_value'] and memory['database_integrity']==state['database_integrity'],'Original terminal export differs')
        for boundfile in (*b['exports'],*b['permanent_exports']):
            require(view.sha(boundfile['path'])==boundfile['sha256'],'Original export changed')
        require(view.read(b['source']['path'])['conversation']==core.base.conversation_only(view.read(ROOT/'data/locomo10.json')[int(ci)]),'Original conversation source differs')
        memories[int(ci)]=memory
    for q in proof['questions']:
        ret=view.read(OLD/'results'/('c'+str(q['conversation']))/'retrieval'/(q['query_id']+'.json'))
        if ret['status']!='complete':continue
        value=ret['value'];memory=memories[q['conversation']];entries={e['entry_id']:e for e in memory['entries']};sources={s['dialogue_id']:s for s in memory['sources']}
        objects=[]
        for e in value['evidence']:
            eid=e['memory_id'];entry=entries[eid]
            require(e=={'memory_id':eid,'memory':entry,'source_dia_ids':[sources[d]['dia_id'] for d in memory['window_provenance'][eid]],
                'attribution':'source window, not individually verified fact evidence'},'Original full evidence provenance differs')
            objects.append(SimpleNamespace(**entry))
        require(value['context']==fmt(None,objects) and value['count']==len(objects),'Original full context differs')
    return {'status':'passed','api_summary_sha256':digest(reports),'native_logs':len(proof['native_audits'])}

def replay_all(view):
    am=view.manifest;m,p,s,review=result_review(am['result_review']['path'],view)
    for name,record in am['bindings'].items():seal.bound(view,record)
    cert=verify_certificate(view,am['bank_review']['path'],am['result_review']['path'],am['tool_review']['path'])
    require(cert['manifest']=={'path':str(NEW/'run_manifest.json'),'sha256':view.sha(NEW/'run_manifest.json')} and cert['proof']==m['proof'],'Certificate source differs')
    for owner,r in roots().items():
        actual={rec['path']:rec['sha256'] for rec in am['records'] if Path(rec['path']).is_relative_to(r)}
        require(actual==review['reviewed_run_inventories'][owner]==hashes(cert['run_inventories'][owner]),'Archived whole-run inventory differs')
    bank_structure(view,p,cert)
    previous=audit_original(view,p)
    reconstruct,transforms,checked=archived_auditors(view,cert)
    outputs=reconstruct(m,p,view=view,created_utc=s['created_utc'])
    require(set(outputs)==set(OUTPUTS),'Wrong frozen output set')
    for n,v in outputs.items():require(v==view.read(NEW/'results'/n),'Full archived union output differs: '+n)
    require(len(checked)==len(p['failed_retrievals']),'Missing per-query certified bank check')
    return {'status':'passed','scope':SCOPE,'original_request_audit':previous,'replayed_rows':{n:len(outputs[n]) for n in OUTPUTS[:3]},
        'output_content_sha256':{n:digest(v) for n,v in outputs.items()},'explicit_private_transformations':transforms,
        'bank_certificate':am['bindings']['bank_certificate'],'bank_byte_checks_after_delete':0,'database_searches':0,'model_calls':0}

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

def evidence_closure(initial,*,annotations=None):
    """One exact deliberately invalid test document, never a general exception."""
    require(sha(ROOT/'tools/baseline_evidence_closure_v1.py')==TYPED_SHA,'Frozen typed walker changed')
    source=function_source(seal.LiveView(),'tools/baseline_evidence_closure_v1.py','collect')
    marker='        p=safe_cache[name];canonical=str(p)'
    branch='            if p in prior:'
    require(source.count(marker)==source.count(branch)==1,'Typed leaf insertion boundary differs')
    source=source.replace(marker,marker+"\n        if p==root/NEGATIVE and expected!=NEGATIVE_SHA:raise ValueError('Exact negative fixture SHA differs')")
    source=source.replace(branch,"""            if p==root/NEGATIVE:
                accept_negative(value,p,expected,annotations)
                enqueue(NEGATIVE_DECISION,NEGATIVE_DECISION_SHA)
                enqueue(POSITIVE,POSITIVE_SHA)
                continue
"""+branch)
    def accept(value,p,h,journal):
        require(h==NEGATIVE_SHA and value.get('schema')==core.REVIEW_SCHEMA and value.get('fixture_only') is True
            and value.get('production_launch_authorized') is False
            and value.get('purpose')=='Synthetic review input for isolated research prepare test; not the final independent review or permission to launch.',
            'Exact negative fixture schema/flags differ')
        require(sha(ROOT/POSITIVE)==POSITIVE_SHA,'Positive fixture counterpart changed')
        from copy import deepcopy
        corrected=deepcopy(value)
        require(corrected['code_sha256']['tools/simplemem_prefix_api_v2.py']=='0'*64,'Negative fixture sole invalid source differs')
        corrected['code_sha256']['tools/simplemem_prefix_api_v2.py']='84a6f321445715939d7d1c36207b5ace72b8f80f9ccedbad35e0ab2c8d7a7d3e'
        require(corrected==read(ROOT/POSITIVE),'Negative fixture differs beyond the sole classified field')
        if journal is not None:journal.append({'document':str(p),'document_sha256':h,'json_pointer':'$',
            'treatment':'exact deliberately invalid nonoperative admission test leaf; original bytes retained, positive counterpart remains recursive',
            'classification':{'path':str(ROOT/NEGATIVE_DECISION),'sha256':NEGATIVE_DECISION_SHA}})
    fn=private_function(source,{**vars(typed),'NEGATIVE':NEGATIVE,'NEGATIVE_SHA':NEGATIVE_SHA,'NEGATIVE_DECISION':NEGATIVE_DECISION,
        'NEGATIVE_DECISION_SHA':NEGATIVE_DECISION_SHA,'POSITIVE':POSITIVE,'POSITIVE_SHA':POSITIVE_SHA,'accept_negative':accept},'collect','<one-exact-negative-fixture-typed-closure>')
    return fn(initial,root=ROOT,safe_path=safe,read=read,sha=sha,annotations=annotations)

def archive(result_path,tool_path,bank_review_path):
    stopped();tool_review(tool_path)
    require(not DEFAULT_PLAN.exists() and not (OUTPUT/'evidence.zip').exists(),'Archive already started')
    m,p,s,review=result_review(result_path,live=True);cert=verify_certificate(seal.LiveView(),bank_review_path,result_path,tool_path)
    before=inventories();require(before==cert['run_inventories'],'Certified banks/run changed')
    # Explicit current-result closure. Nothing missing is silently optional.
    initial=[(str(path),sha(path)) for path in (result_path,tool_path,bank_review_path,NEW/'run_manifest.json',ROOT/metric.METRIC,ROOT/'data/locomo10.json',ROOT/'configs/sdk20_full1540_locked_v1.json')]
    print('simplemem lifecycle: explicit evidence closure',flush=True)
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
            if ordinal%2000==0:print('simplemem lifecycle: archive records '+str(ordinal)+'/'+str(len(records)),flush=True)
            if not rec['retained'] or rec['member'] in written:continue
            data=Path(rec['path']).read_bytes();require(hashlib.sha256(data).hexdigest()==rec['sha256'],'Source changed while archiving')
            z.writestr(rec['member'],data);written.add(rec['member'])
    require(inventories()==before,'Run changed during archive')
    for path,h in evidence.items():require(sha(path)==h,'Dependency changed during archive')
    print('simplemem lifecycle: actual ZIP-only replay',flush=True)
    with Resolver(target,mp) as view:replay=strict_replay(view)
    rp=OUTPUT/'archive_replay.json';save_new(rp,replay)
    keep={p:r['sha256'] for p,r in allfiles.items() if not any(Path(p).is_relative_to(w) for w in work_roots())}
    keep.update({str(x):sha(x) for x in (mp,target,rp)})
    plan={'schema':PLAN_SCHEMA,'method':'simplemem','created_utc':now(),'workspace':str(ROOT),'allowed_delete_roots':[str(w) for w in work_roots()],
        'archive':binding(target),'archive_manifest':binding(mp),'archive_replay':binding(rp),'archive_crc_verified':True,
        'result_review':am['result_review'],'tool_review':am['tool_review'],'bank_review':am['bank_review'],
        'keep_sha256':keep,'code_sha256':code_hashes(),'work_inventories':am['work_inventories'],'deletion_executed':False,'scope':SCOPE}
    save_new(DEFAULT_PLAN,plan);return {'status':'archive_verified_requires_independent_cleanup_review','plan':binding(DEFAULT_PLAN)}

def verify_archive(*,require_work=True):
    stopped();plan=read(safe(DEFAULT_PLAN))
    require(plan['schema']==PLAN_SCHEMA and plan['method']=='simplemem' and plan['workspace']==str(ROOT)
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
    v=read(safe(path));require(v.get('schema')=='simplemem-native-v2-cleanup-review-v1' and v.get('status')=='passed'
        and v.get('cleanup_plan_sha256')==sha(DEFAULT_PLAN) and v.get('archive_sha256')==plan['archive']['sha256']
        and v.get('scope')==SCOPE and v.get('allowed_delete_roots')==plan['allowed_delete_roots'],'Independent exact cleanup review required')
    return binding(path)

def removal_evidence(plan,review):
    intent=binding(OUTPUT/'cleanup_execution_intent.json');iv=read(intent['path'])
    require(iv.get('schema')=='simplemem-native-v2-cleanup-intent-v1' and iv.get('targets')==plan['allowed_delete_roots']
        and iv.get('cleanup_plan_sha256')==sha(DEFAULT_PLAN) and iv.get('cleanup_review_sha256')==review['sha256'],'Removal intent differs')
    receipts=[]
    for k,r in roots().items():
        require(not safe(r/'work').exists(),'Work remains or reappeared')
        b=binding(OUTPUT/('removal_'+k+'.json'));v=read(b['path'])
        require(v.get('schema')=='simplemem-native-v2-removal-receipt-v1' and v.get('removed_path')==str(r/'work')
            and v.get('cleanup_plan_sha256')==sha(DEFAULT_PLAN) and v.get('cleanup_review_sha256')==review['sha256'],'Removal receipt differs')
        receipts.append(b)
    return intent,receipts

def completion_values(plan,review,intent,receipts):
    return {'schema':'simplemem-native-v2-cleanup-receipt-v1','method':'simplemem','cleanup_plan':binding(DEFAULT_PLAN),'cleanup_review':review,
        'archive':plan['archive'],'archive_manifest':plan['archive_manifest'],'intent':intent,'removal_receipts':receipts,'scope':SCOPE}

def complete(review_path):
    plan=verify_archive(require_work=False);review=cleanup_review(review_path,plan);intent,receipts=removal_evidence(plan,review)
    receipt=completion_values(plan,review,intent,receipts);receipt['completed_utc']=now();save_new(OUTPUT/'cleanup_receipt.json',receipt)
    completion={'schema':'simplemem-native-v2-predecessor-completion-v1','status':'passed','method':'simplemem','result_owner':str(NEW),
        'summary_sha256':sha(NEW/'results/summary.json'),'result_review':plan['result_review'],'cleanup_receipt':binding(OUTPUT/'cleanup_receipt.json'),'scope':SCOPE}
    save_new(OUTPUT/'predecessor_completion.json',completion);return {'status':'complete','scope':SCOPE}

def verify_completion():
    plan=verify_archive(require_work=False);receipt=read(OUTPUT/'cleanup_receipt.json');review=cleanup_review(bound(receipt['cleanup_review']),plan)
    intent,receipts=removal_evidence(plan,review)
    require({k:v for k,v in receipt.items() if k!='completed_utc'}==completion_values(plan,review,intent,receipts),'Aggregate receipt differs')
    expected={'schema':'simplemem-native-v2-predecessor-completion-v1','status':'passed','method':'simplemem','result_owner':str(NEW),
        'summary_sha256':sha(NEW/'results/summary.json'),'result_review':plan['result_review'],'cleanup_receipt':binding(OUTPUT/'cleanup_receipt.json'),'scope':SCOPE}
    require(read(OUTPUT/'predecessor_completion.json')==expected,'Strong completion differs')
    return {'status':'passed','method':'simplemem','result_owner':str(NEW),'scope':SCOPE}

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
if __name__=='__main__':main()
